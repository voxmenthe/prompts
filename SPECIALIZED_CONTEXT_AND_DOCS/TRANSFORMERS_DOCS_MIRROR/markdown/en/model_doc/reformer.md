*This model was released on 2020-01-13 and added to Hugging Face Transformers on 2020-11-16.*

# Reformer

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white)

## Overview

The Reformer model was proposed in the paper [Reformer: The Efficient Transformer](https://huggingface.co/papers/2001.04451) by Nikita Kitaev, Łukasz Kaiser, Anselm Levskaya.

The abstract from the paper is the following:

*Large Transformer models routinely achieve state-of-the-art results on a number of tasks but training these models can
be prohibitively costly, especially on long sequences. We introduce two techniques to improve the efficiency of
Transformers. For one, we replace dot-product attention by one that uses locality-sensitive hashing, changing its
complexity from O(L^2) to O(Llog(L)), where L is the length of the sequence. Furthermore, we use reversible residual
layers instead of the standard residuals, which allows storing activations only once in the training process instead of
N times, where N is the number of layers. The resulting model, the Reformer, performs on par with Transformer models
while being much more memory-efficient and much faster on long sequences.*

This model was contributed by [patrickvonplaten](https://huggingface.co/patrickvonplaten). The Authors’ code can be
found [here](https://github.com/google/trax/tree/master/trax/models/reformer).

## Usage tips

* Reformer does **not** work with *torch.nn.DataParallel* due to a bug in PyTorch, see [issue #36035](https://github.com/pytorch/pytorch/issues/36035).
* Use Axial position encoding (see below for more details). It’s a mechanism to avoid having a huge positional encoding matrix (when the sequence length is very big) by factorizing it into smaller matrices.
* Replace traditional attention by LSH (local-sensitive hashing) attention (see below for more details). It’s a technique to avoid computing the full product query-key in the attention layers.
* Avoid storing the intermediate results of each layer by using reversible transformer layers to obtain them during the backward pass (subtracting the residuals from the input of the next layer gives them back) or recomputing them for results inside a given layer (less efficient than storing them but saves memory).
* Compute the feedforward operations by chunks and not on the whole batch.

### Axial Positional Encodings

Axial Positional Encodings were first implemented in Google’s [trax library](https://github.com/google/trax/blob/4d99ad4965bab1deba227539758d59f0df0fef48/trax/layers/research/position_encodings.py#L29)
and developed by the authors of this model’s paper. In models that are treating very long input sequences, the
conventional position id encodings store an embeddings vector of sizeddd being the `config.hidden_size` for
every positioni,…,nsi, \ldots, n\_si,…,ns​, withnsn\_sns​ being `config.max_embedding_size`. This means that having
a sequence length ofns=219≈0.5Mn\_s = 2^{19} \approx 0.5Mns​=219≈0.5M and a `config.hidden_size` ofd=210≈1000d = 2^{10} \approx 1000d=210≈1000
would result in a position encoding matrix:
Xi,j, with i∈[1,…,d] and j∈[1,…,ns]X\_{i,j}, \text{ with } i \in \left[1,\ldots, d\right] \text{ and } j \in \left[1,\ldots, n\_s\right]Xi,j​, with i∈[1,…,d] and j∈[1,…,ns​]

which alone has over 500M parameters to store. Axial positional encodings factorizeXi,jX\_{i,j}Xi,j​ into two matrices:
Xi,j1, with i∈[1,…,d1] and j∈[1,…,ns1]X^{1}\_{i,j}, \text{ with } i \in \left[1,\ldots, d^1\right] \text{ and } j \in \left[1,\ldots, n\_s^1\right]Xi,j1​, with i∈[1,…,d1] and j∈[1,…,ns1​]

and
Xi,j2, with i∈[1,…,d2] and j∈[1,…,ns2]X^{2}\_{i,j}, \text{ with } i \in \left[1,\ldots, d^2\right] \text{ and } j \in \left[1,\ldots, n\_s^2\right]Xi,j2​, with i∈[1,…,d2] and j∈[1,…,ns2​]

with:
d=d1+d2 and ns=ns1×ns2.d = d^1 + d^2 \text{ and } n\_s = n\_s^1 \times n\_s^2 .d=d1+d2 and ns​=ns1​×ns2​.

Therefore the following holds:
Xi,j={Xi,k1,if  i<d1 with k=jmod  ns1Xi−d1,l2,if i≥d1 with l=⌊jns1⌋X\_{i,j} = \begin{cases}
X^{1}\_{i, k}, & \text{if }\ i < d^1 \text{ with } k = j \mod n\_s^1 \\
X^{2}\_{i - d^1, l}, & \text{if } i \ge d^1 \text{ with } l = \lfloor\frac{j}{n\_s^1}\rfloor
\end{cases}Xi,j​={Xi,k1​,Xi−d1,l2​,​if  i<d1 with k=jmodns1​if i≥d1 with l=⌊ns1​j​⌋​

Intuitively, this means that a position embedding vectorxj∈Rdx\_j \in \mathbb{R}^{d}xj​∈Rd is now the composition of two
factorized embedding vectors:xk,l1+xl,k2x^1\_{k, l} + x^2\_{l, k}xk,l1​+xl,k2​, where as the `config.max_embedding_size` dimensionjjj is factorized intok and lk \text{ and } lk and l. This design ensures that each position embedding vectorxjx\_jxj​ is unique.

Using the above example again, axial position encoding withd1=29,d2=29,ns1=29,ns2=210d^1 = 2^9, d^2 = 2^9, n\_s^1 = 2^9, n\_s^2 = 2^{10}d1=29,d2=29,ns1​=29,ns2​=210
can drastically reduced the number of parameters from 500 000 000 to218+219≈7800002^{18} + 2^{19} \approx 780 000218+219≈780000 parameters, this means 85% less memory usage.

In practice, the parameter `config.axial_pos_embds_dim` is set to a tuple(d1,d2)(d^1, d^2)(d1,d2) which sum has to be
equal to `config.hidden_size` and `config.axial_pos_shape` is set to a tuple(ns1,ns2)(n\_s^1, n\_s^2)(ns1​,ns2​) which
product has to be equal to `config.max_embedding_size`, which during training has to be equal to the *sequence
length* of the `input_ids`.

### LSH Self Attention

In Locality sensitive hashing (LSH) self attention the key and query projection weights are tied. Therefore, the key
query embedding vectors are also tied. LSH self attention uses the locality sensitive hashing mechanism proposed in
[Practical and Optimal LSH for Angular Distance](https://huggingface.co/papers/1509.02897) to assign each of the tied key
query embedding vectors to one of `config.num_buckets` possible buckets. The premise is that the more “similar”
key query embedding vectors (in terms of *cosine similarity*) are to each other, the more likely they are assigned to
the same bucket.

The accuracy of the LSH mechanism can be improved by increasing `config.num_hashes` or directly the argument
`num_hashes` of the forward function so that the output of the LSH self attention better approximates the output
of the “normal” full self attention. The buckets are then sorted and chunked into query key embedding vector chunks
each of length `config.lsh_chunk_length`. For each chunk, the query embedding vectors attend to its key vectors
(which are tied to themselves) and to the key embedding vectors of `config.lsh_num_chunks_before` previous
neighboring chunks and `config.lsh_num_chunks_after` following neighboring chunks.

For more information, see the [original Paper](https://huggingface.co/papers/2001.04451) or this great [blog post](https://www.pragmatic.ml/reformer-deep-dive/).

Note that `config.num_buckets` can also be factorized into a list(nbuckets1,nbuckets2)(n\_{\text{buckets}}^1,
n\_{\text{buckets}}^2)(nbuckets1​,nbuckets2​). This way instead of assigning the query key embedding vectors to one of(1,…,nbuckets)(1,\ldots,
n\_{\text{buckets}})(1,…,nbuckets​) they are assigned to one of(1−1,…,nbuckets1−1,…,1−nbuckets2,…,nbuckets1−nbuckets2)(1-1,\ldots, n\_{\text{buckets}}^1-1, \ldots,
1-n\_{\text{buckets}}^2, \ldots, n\_{\text{buckets}}^1-n\_{\text{buckets}}^2)(1−1,…,nbuckets1​−1,…,1−nbuckets2​,…,nbuckets1​−nbuckets2​). This is crucial for very long sequences to
save memory.

When training a model from scratch, it is recommended to leave `config.num_buckets=None`, so that depending on the
sequence length a good value for `num_buckets` is calculated on the fly. This value will then automatically be
saved in the config and should be reused for inference.

Using LSH self attention, the memory and time complexity of the query-key matmul operation can be reduced fromO(ns×ns)\mathcal{O}(n\_s \times n\_s)O(ns​×ns​) toO(ns×log⁡(ns))\mathcal{O}(n\_s \times \log(n\_s))O(ns​×log(ns​)), which usually represents the memory
and time bottleneck in a transformer model, withnsn\_sns​ being the sequence length.

### Local Self Attention

Local self attention is essentially a “normal” self attention layer with key, query and value projections, but is
chunked so that in each chunk of length `config.local_chunk_length` the query embedding vectors only attends to
the key embedding vectors in its chunk and to the key embedding vectors of `config.local_num_chunks_before`
previous neighboring chunks and `config.local_num_chunks_after` following neighboring chunks.

Using Local self attention, the memory and time complexity of the query-key matmul operation can be reduced fromO(ns×ns)\mathcal{O}(n\_s \times n\_s)O(ns​×ns​) toO(ns×log⁡(ns))\mathcal{O}(n\_s \times \log(n\_s))O(ns​×log(ns​)), which usually represents the memory
and time bottleneck in a transformer model, withnsn\_sns​ being the sequence length.

### Training

During training, we must ensure that the sequence length is set to a value that can be divided by the least common
multiple of `config.lsh_chunk_length` and `config.local_chunk_length` and that the parameters of the Axial
Positional Encodings are correctly set as described above. Reformer is very memory efficient so that the model can
easily be trained on sequences as long as 64000 tokens.

For training, the [ReformerModelWithLMHead](/docs/transformers/v4.56.2/en/model_doc/reformer#transformers.ReformerModelWithLMHead) should be used as follows:


```
input_ids = tokenizer.encode("This is a sentence from the training data", return_tensors="pt")
loss = model(input_ids, labels=input_ids)[0]
```

## Resources

* [Text classification task guide](../tasks/sequence_classification)
* [Question answering task guide](../tasks/question_answering)
* [Causal language modeling task guide](../tasks/language_modeling)
* [Masked language modeling task guide](../tasks/masked_language_modeling)

## ReformerConfig

### class transformers.ReformerConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/reformer/configuration_reformer.py#L25)

( attention\_head\_size = 64 attn\_layers = ['local', 'lsh', 'local', 'lsh', 'local', 'lsh'] axial\_norm\_std = 1.0 axial\_pos\_embds = True axial\_pos\_shape = [64, 64] axial\_pos\_embds\_dim = [64, 192] chunk\_size\_lm\_head = 0 eos\_token\_id = 2 feed\_forward\_size = 512 hash\_seed = None hidden\_act = 'relu' hidden\_dropout\_prob = 0.05 hidden\_size = 256 initializer\_range = 0.02 is\_decoder = False layer\_norm\_eps = 1e-12 local\_num\_chunks\_before = 1 local\_num\_chunks\_after = 0 local\_attention\_probs\_dropout\_prob = 0.05 local\_attn\_chunk\_length = 64 lsh\_attn\_chunk\_length = 64 lsh\_attention\_probs\_dropout\_prob = 0.0 lsh\_num\_chunks\_before = 1 lsh\_num\_chunks\_after = 0 max\_position\_embeddings = 4096 num\_attention\_heads = 12 num\_buckets = None num\_hashes = 1 pad\_token\_id = 0 vocab\_size = 320 tie\_word\_embeddings = False use\_cache = True classifier\_dropout = None \*\*kwargs  )

Parameters

* **attention\_head\_size** (`int`, *optional*, defaults to 64) —
  Dimensionality of the projected key, query and value vectors
* **attn\_layers** (`list[str]`, *optional*, defaults to `["local", "lsh", "local", "lsh", "local", "lsh"]`) —
  List of attention layer types in ascending order. It can be chosen between a LSHSelfAttention layer
  (`"lsh"`) and a LocalSelfAttention layer (`"local"`).

  For more information on LSHSelfAttention layer, see [LSH Self Attention](reformer#lsh-self-attention). For
  more information on LocalSelfAttention layer, see [Local Self Attention](reformer#local-self-attention).
* **axial\_pos\_embds** (`bool`, *optional*, defaults to `True`) —
  Whether or not to use axial position embeddings. For more information on how axial position embeddings
  work, see [Axial Position Encodings](reformer#axial-positional-encodings).
* **axial\_norm\_std** (`float`, *optional*, defaults to 1.0) —
  The standard deviation of the normal\_initializer for initializing the weight matrices of the axial
  positional encodings.
* **axial\_pos\_shape** (`list[int]`, *optional*, defaults to `[64, 64]`) —
  The position dims of the axial position encodings. During training, the product of the position dims has to
  be equal to the sequence length.

  For more information on how axial position embeddings work, see [Axial Position
  Encodings](reformer#axial-positional-encodings).
* **axial\_pos\_embds\_dim** (`list[int]`, *optional*, defaults to `[64, 192]`) —
  The embedding dims of the axial position encodings. The sum of the embedding dims has to be equal to the
  hidden size.

  For more information on how axial position embeddings work, see [Axial Position
  Encodings](reformer#axial-positional-encodings).
* **chunk\_size\_lm\_head** (`int`, *optional*, defaults to 0) —
  The chunk size of the final language model feed forward head layer. A chunk size of 0 means that the feed
  forward layer is not chunked. A chunk size of n means that the feed forward layer processes n <
  sequence\_length embeddings at a time.

  For more information on feed forward chunking, see [How does Feed Forward Chunking
  work?](../glossary#feed-forward-chunking).
* **eos\_token\_id** (`int`, *optional*, defaults to 2) —
  The token id for the end-of-sentence token.
* **feed\_forward\_size** (`int`, *optional*, defaults to 512) —
  Dimensionality of the feed\_forward layer in the residual attention block.
* **hash\_seed** (`int`, *optional*) —
  Seed that can be used to make local sensitive hashing in `LSHSelfAttention` deterministic. This should only
  be set for testing purposed. For evaluation and training purposes `hash_seed` should be left as `None` to
  ensure fully random rotations in local sensitive hashing scheme.
* **hidden\_act** (`str` or `Callable`, *optional*, defaults to `"relu"`) —
  The non-linear activation function (function or string) in the feed forward layer in the residual attention
  block. If string, `"gelu"`, `"relu"`, `"silu"` and `"gelu_new"` are supported.
* **hidden\_dropout\_prob** (`float`, *optional*, defaults to 0.05) —
  The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
* **hidden\_size** (`int`, *optional*, defaults to 256) —
  Dimensionality of the output hidden states of the residual attention blocks.
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
* **is\_decoder** (`bool`, *optional*, defaults to `False`) —
  Whether or not to use a causal mask in addition to the `attention_mask` passed to [ReformerModel](/docs/transformers/v4.56.2/en/model_doc/reformer#transformers.ReformerModel). When
  using the Reformer for causal language modeling, this argument should be set to `True`.
* **layer\_norm\_eps** (`float`, *optional*, defaults to 1e-12) —
  The epsilon used by the layer normalization layers.
* **local\_chunk\_length** (`int`, *optional*, defaults to 64) —
  Length of chunk which attends to itself in `LocalSelfAttention`. Chunking reduces memory complexity from
  sequence length x sequence length (self attention) to chunk length x chunk length x sequence length / chunk
  length (chunked self attention).
* **local\_num\_chunks\_before** (`int`, *optional*, defaults to 1) —
  Number of previous neighbouring chunks to attend to in `LocalSelfAttention` layer to itself.
* **local\_num\_chunks\_after** (`int`, *optional*, defaults to 0) —
  Number of following neighbouring chunks to attend to in `LocalSelfAttention` layer in addition to itself.
* **local\_attention\_probs\_dropout\_prob** (`float`, *optional*, defaults to 0.1) —
  The dropout ratio for the attention probabilities in `LocalSelfAttention`.
* **lsh\_attn\_chunk\_length** (`int`, *optional*, defaults to 64) —
  Length of chunk which attends to itself in `LSHSelfAttention`. Chunking reduces memory complexity from
  sequence length x sequence length (self attention) to chunk length x chunk length x sequence length / chunk
  length (chunked self attention).
* **lsh\_num\_chunks\_before** (`int`, *optional*, defaults to 1) —
  Number of previous neighbouring chunks to attend to in `LSHSelfAttention` layer to itself.
* **lsh\_num\_chunks\_after** (`int`, *optional*, defaults to 0) —
  Number of following neighbouring chunks to attend to in `LSHSelfAttention` layer to itself.
* **lsh\_attention\_probs\_dropout\_prob** (`float`, *optional*, defaults to 0.1) —
  The dropout ratio for the attention probabilities in `LSHSelfAttention`.
* **max\_position\_embeddings** (`int`, *optional*, defaults to 4096) —
  The maximum sequence length that this model might ever be used with. Typically set this to something large
  just in case (e.g., 512 or 1024 or 2048).
* **num\_attention\_heads** (`int`, *optional*, defaults to 12) —
  Number of attention heads for each attention layer in the Transformer encoder.
* **num\_buckets** (`int` or `list[int]`, *optional*) —
  Number of buckets, the key query vectors can be “hashed into” using the locality sensitive hashing scheme.
  Each query key vector is hashed into a hash in `1, ..., num_buckets`. The number of buckets can also be
  factorized into a list for improved memory complexity. In this case, each query key vector is hashed into a
  hash in `1-1, 1-2, ..., num_buckets[0]-1, ..., num_buckets[0]-num_buckets[1]` if `num_buckets` is
  factorized into two factors. The number of buckets (or the product the factors) should approximately equal
  sequence length / lsh\_chunk\_length. If `num_buckets` not set, a good value is calculated on the fly.
* **num\_hashes** (`int`, *optional*, defaults to 1) —
  Number of hashing rounds (e.g., number of random rotations) in Local Sensitive Hashing scheme. The higher
  `num_hashes`, the more accurate the `LSHSelfAttention` becomes, but also the more memory and time intensive
  the hashing becomes.
* **pad\_token\_id** (`int`, *optional*, defaults to 0) —
  The token id for the padding token.
* **vocab\_size** (`int`, *optional*, defaults to 320) —\
  Vocabulary size of the Reformer model. Defines the number of different tokens that can be represented by
  the `inputs_ids` passed when calling [ReformerModel](/docs/transformers/v4.56.2/en/model_doc/reformer#transformers.ReformerModel).
* **tie\_word\_embeddings** (`bool`, *optional*, defaults to `False`) —
  Whether to tie input and output embeddings.
* **use\_cache** (`bool`, *optional*, defaults to `True`) —
  Whether or not the model should return the last key/values attentions (not used by all models).
* **classifier\_dropout** (`float`, *optional*) —
  The dropout ratio for the classification head.

This is the configuration class to store the configuration of a [ReformerModel](/docs/transformers/v4.56.2/en/model_doc/reformer#transformers.ReformerModel). It is used to instantiate a
Reformer model according to the specified arguments, defining the model architecture. Instantiating a configuration
with the defaults will yield a similar configuration to that of the ReFormer
[google/reformer-crime-and-punishment](https://huggingface.co/google/reformer-crime-and-punishment) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Examples:


```
>>> from transformers import ReformerConfig, ReformerModel

>>> # Initializing a Reformer configuration
>>> configuration = ReformerConfig()

>>> # Initializing a Reformer model (with random weights)
>>> model = ReformerModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## ReformerTokenizer

### class transformers.ReformerTokenizer

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/reformer/tokenization_reformer.py#L37)

( vocab\_file eos\_token = '</s>' unk\_token = '<unk>' additional\_special\_tokens = [] sp\_model\_kwargs: typing.Optional[dict[str, typing.Any]] = None \*\*kwargs  )

Parameters

* **vocab\_file** (`str`) —
  [SentencePiece](https://github.com/google/sentencepiece) file (generally has a *.spm* extension) that
  contains the vocabulary necessary to instantiate a tokenizer.
* **eos\_token** (`str`, *optional*, defaults to `"</s>"`) —
  The end of sequence token.

  When building a sequence using special tokens, this is not the token that is used for the end of sequence.
  The token used is the `sep_token`.
* **unk\_token** (`str`, *optional*, defaults to `"<unk>"`) —
  The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
  token instead.
* **additional\_special\_tokens** (`list[str]`, *optional*, defaults to `[]`) —
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

Construct a Reformer tokenizer. Based on [SentencePiece](https://github.com/google/sentencepiece) .

This tokenizer inherits from [PreTrainedTokenizer](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer) which contains most of the main methods. Users should refer to
this superclass for more information regarding those methods.

#### save\_vocabulary

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/reformer/tokenization_reformer.py#L158)

( save\_directory: str filename\_prefix: typing.Optional[str] = None  )

## ReformerTokenizerFast

### class transformers.ReformerTokenizerFast

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/reformer/tokenization_reformer_fast.py#L39)

( vocab\_file = None tokenizer\_file = None eos\_token = '</s>' unk\_token = '<unk>' additional\_special\_tokens = [] \*\*kwargs  )

Parameters

* **vocab\_file** (`str`) —
  [SentencePiece](https://github.com/google/sentencepiece) file (generally has a *.spm* extension) that
  contains the vocabulary necessary to instantiate a tokenizer.
* **eos\_token** (`str`, *optional*, defaults to `"</s>"`) —
  The end of sequence token.

  When building a sequence using special tokens, this is not the token that is used for the end of sequence.
  The token used is the `sep_token`.
* **unk\_token** (`str`, *optional*, defaults to `"<unk>"`) —
  The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
  token instead.
* **pad\_token** (`str`, *optional*, defaults to `"<pad>"`) —
  The token used for padding, for example when batching sequences of different lengths.
* **additional\_special\_tokens** (`list[str]`, *optional*) —
  Additional special tokens used by the tokenizer.

Construct a “fast” Reformer tokenizer (backed by HuggingFace’s *tokenizers* library). Based on
[Unigram](https://huggingface.co/docs/tokenizers/python/latest/components.html?highlight=unigram#models).

This tokenizer inherits from [PreTrainedTokenizerFast](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast) which contains most of the main methods. Users should
refer to this superclass for more information regarding those methods.

## ReformerModel

### class transformers.ReformerModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/reformer/modeling_reformer.py#L1993)

( config  )

Parameters

* **config** ([ReformerModel](/docs/transformers/v4.56.2/en/model_doc/reformer#transformers.ReformerModel)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Reformer Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/reformer/modeling_reformer.py#L2021)

( input\_ids: typing.Optional[torch.Tensor] = None attention\_mask: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.Tensor] = None head\_mask: typing.Optional[torch.Tensor] = None inputs\_embeds: typing.Optional[torch.Tensor] = None num\_hashes: typing.Optional[int] = None past\_buckets\_states: typing.Optional[list[tuple[torch.Tensor]]] = None use\_cache: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → `transformers.models.reformer.modeling_reformer.ReformerModelOutput` or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) —
  Indices of input sequence tokens in the vocabulary. During training the input\_ids sequence\_length has to be
  a multiple of the relevant model’s chunk lengths (lsh’s, local’s or both). During evaluation, the indices
  are automatically padded to be a multiple of the chunk length.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
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
* **num\_hashes** (`int`, *optional*) —
  The number of hashing rounds that should be performed during bucketing. Setting this argument overwrites
  the default defined in `config.num_hashes`.

  For more information, see `num_hashes` in [ReformerConfig](/docs/transformers/v4.56.2/en/model_doc/reformer#transformers.ReformerConfig).
* **past\_buckets\_states** (`list[tuple(torch.LongTensor, torch.FloatTensor)]`, *optional*) —
  List of `tuple(torch.LongTensor, torch.FloatTensor` of length `config.n_layers`, with the first element
  being the previous *buckets* of shape `(batch_size, num_heads, num_hashes, sequence_length)`) and the
  second being the previous *hidden\_states* of shape `(batch_size, sequence_length, hidden_size)`).

  Contains precomputed hidden-states and buckets (only relevant for LSH Self-Attention). Can be used to speed
  up sequential decoding.
* **use\_cache** (`bool`, *optional*) —
  If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
  `past_key_values`).
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

`transformers.models.reformer.modeling_reformer.ReformerModelOutput` or `tuple(torch.FloatTensor)`

A `transformers.models.reformer.modeling_reformer.ReformerModelOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([ReformerConfig](/docs/transformers/v4.56.2/en/model_doc/reformer#transformers.ReformerConfig)) and inputs.

* **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, num_predict, hidden_size)`) — Sequence of hidden-states at the last layer of the model.

  `num_predict` corresponds to `target_mapping.shape[1]`. If `target_mapping` is `None`, then `num_predict`
  corresponds to `sequence_length`.
* **past\_buckets\_states** (`list[tuple(torch.LongTensor, torch.FloatTensor)]`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) — List of `tuple(torch.LongTensor, torch.FloatTensor` of length `config.n_layers`, with the first element
  being the previous *buckets* of shape `(batch_size, num_heads, num_hashes, sequence_length)`) and the
  second being the previous *hidden\_states* of shape `(batch_size, sequence_length, hidden_size)`).

  Contains precomputed buckets and hidden-states that can be used (see `past_buckets_states` input) to speed
  up sequential decoding.
* **hidden\_states** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [ReformerModel](/docs/transformers/v4.56.2/en/model_doc/reformer#transformers.ReformerModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

## ReformerModelWithLMHead

### class transformers.ReformerModelWithLMHead

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/reformer/modeling_reformer.py#L2227)

( config  )

Parameters

* **config** ([ReformerModelWithLMHead](/docs/transformers/v4.56.2/en/model_doc/reformer#transformers.ReformerModelWithLMHead)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Reformer Model with a `language modeling` head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/reformer/modeling_reformer.py#L2255)

( input\_ids: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.Tensor] = None attention\_mask: typing.Optional[torch.Tensor] = None head\_mask: typing.Optional[torch.Tensor] = None inputs\_embeds: typing.Optional[torch.Tensor] = None num\_hashes: typing.Optional[int] = None past\_buckets\_states: typing.Optional[list[tuple[torch.Tensor]]] = None use\_cache: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None labels: typing.Optional[torch.Tensor] = None \*\*kwargs  ) → [transformers.modeling\_outputs.CausalLMOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutput) or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) —
  Indices of input sequence tokens in the vocabulary. During training the input\_ids sequence\_length has to be
  a multiple of the relevant model’s chunk lengths (lsh’s, local’s or both). During evaluation, the indices
  are automatically padded to be a multiple of the chunk length.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **position\_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **head\_mask** (`torch.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **inputs\_embeds** (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model’s internal embedding lookup matrix.
* **num\_hashes** (`int`, *optional*) —
  The number of hashing rounds that should be performed during bucketing. Setting this argument overwrites
  the default defined in `config.num_hashes`.

  For more information, see `num_hashes` in [ReformerConfig](/docs/transformers/v4.56.2/en/model_doc/reformer#transformers.ReformerConfig).
* **past\_buckets\_states** (`list[tuple(torch.LongTensor, torch.FloatTensor)]`, *optional*) —
  List of `tuple(torch.LongTensor, torch.FloatTensor` of length `config.n_layers`, with the first element
  being the previous *buckets* of shape `(batch_size, num_heads, num_hashes, sequence_length)`) and the
  second being the previous *hidden\_states* of shape `(batch_size, sequence_length, hidden_size)`).

  Contains precomputed hidden-states and buckets (only relevant for LSH Self-Attention). Can be used to speed
  up sequential decoding.
* **use\_cache** (`bool`, *optional*) —
  If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
  `past_key_values`).
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
* **labels** (`torch.LongTensor` of shape `(batch_size,)`, *optional*) —
  Labels for computing the sequence classification/regression loss. Indices should be in `[-100, 0, ..., config.vocab_size - 1]`. All labels set to `-100` are ignored (masked), the loss is only computed for
  labels in `[0, ..., config.vocab_size]`

Returns

[transformers.modeling\_outputs.CausalLMOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutput) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.CausalLMOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([ReformerConfig](/docs/transformers/v4.56.2/en/model_doc/reformer#transformers.ReformerConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Language modeling loss (for next-token prediction).
* **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [ReformerModelWithLMHead](/docs/transformers/v4.56.2/en/model_doc/reformer#transformers.ReformerModelWithLMHead) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> import torch
>>> from transformers import AutoTokenizer, ReformerModelWithLMHead

>>> tokenizer = AutoTokenizer.from_pretrained("google/reformer-crime-and-punishment")
>>> model = ReformerModelWithLMHead.from_pretrained("google/reformer-crime-and-punishment")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
>>> outputs = model(**inputs, labels=inputs["input_ids"])
>>> loss = outputs.loss
>>> logits = outputs.logits
```

## ReformerForMaskedLM

### class transformers.ReformerForMaskedLM

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/reformer/modeling_reformer.py#L2376)

( config  )

Parameters

* **config** ([ReformerForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/reformer#transformers.ReformerForMaskedLM)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Reformer Model with a `language modeling` head on top.”

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/reformer/modeling_reformer.py#L2398)

( input\_ids: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.Tensor] = None attention\_mask: typing.Optional[torch.Tensor] = None head\_mask: typing.Optional[torch.Tensor] = None inputs\_embeds: typing.Optional[torch.Tensor] = None num\_hashes: typing.Optional[int] = None labels: typing.Optional[torch.Tensor] = None output\_hidden\_states: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → [transformers.modeling\_outputs.MaskedLMOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.MaskedLMOutput) or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) —
  Indices of input sequence tokens in the vocabulary. During training the input\_ids sequence\_length has to be
  a multiple of the relevant model’s chunk lengths (lsh’s, local’s or both). During evaluation, the indices
  are automatically padded to be a multiple of the chunk length.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **position\_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **head\_mask** (`torch.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **inputs\_embeds** (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model’s internal embedding lookup matrix.
* **num\_hashes** (`int`, *optional*) —
  The number of hashing rounds that should be performed during bucketing. Setting this argument overwrites
  the default defined in `config.num_hashes`.

  For more information, see `num_hashes` in [ReformerConfig](/docs/transformers/v4.56.2/en/model_doc/reformer#transformers.ReformerConfig).
* **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ..., config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked),
  the loss is only computed for the tokens with labels

  This example uses a false checkpoint since we don’t have any available pretrained model for the masked language
  modeling task with the Reformer architecture.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

[transformers.modeling\_outputs.MaskedLMOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.MaskedLMOutput) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.MaskedLMOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.MaskedLMOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([ReformerConfig](/docs/transformers/v4.56.2/en/model_doc/reformer#transformers.ReformerConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Masked language modeling (MLM) loss.
* **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [ReformerForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/reformer#transformers.ReformerForMaskedLM) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> import torch
>>> from transformers import AutoTokenizer, ReformerForMaskedLM

>>> tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-reformer")
>>> model = ReformerForMaskedLM.from_pretrained("hf-internal-testing/tiny-random-reformer")

>>> # add mask_token
>>> tokenizer.add_special_tokens({"mask_token": "[MASK]"})
>>> inputs = tokenizer("The capital of France is [MASK].", return_tensors="pt")

>>> # resize model's embedding matrix
>>> model.resize_token_embeddings(new_num_tokens=model.config.vocab_size + 1)
>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> # retrieve index of [MASK]
>>> mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]

>>> predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
>>> predicted_token = tokenizer.decode(predicted_token_id)
```


```
>>> labels = tokenizer("The capital of France is Paris.", return_tensors="pt")["input_ids"]
>>> # mask labels of non-[MASK] tokens
>>> labels = torch.where(
...     inputs.input_ids == tokenizer.mask_token_id, labels[:, : inputs["input_ids"].shape[-1]], -100
... )

>>> outputs = model(**inputs, labels=labels)
>>> loss = round(outputs.loss.item(), 2)
```

## ReformerForSequenceClassification

### class transformers.ReformerForSequenceClassification

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/reformer/modeling_reformer.py#L2517)

( config  )

Parameters

* **config** ([ReformerForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/reformer#transformers.ReformerForSequenceClassification)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Reformer Model transformer with a sequence classification/regression head on top (a linear layer on top of the
pooled output) e.g. for GLUE tasks.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/reformer/modeling_reformer.py#L2531)

( input\_ids: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.Tensor] = None attention\_mask: typing.Optional[torch.Tensor] = None head\_mask: typing.Optional[torch.Tensor] = None inputs\_embeds: typing.Optional[torch.Tensor] = None num\_hashes: typing.Optional[int] = None labels: typing.Optional[torch.Tensor] = None output\_hidden\_states: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → [transformers.modeling\_outputs.SequenceClassifierOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput) or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) —
  Indices of input sequence tokens in the vocabulary. During training the input\_ids sequence\_length has to be
  a multiple of the relevant model’s chunk lengths (lsh’s, local’s or both). During evaluation, the indices
  are automatically padded to be a multiple of the chunk length.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **position\_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **head\_mask** (`torch.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **inputs\_embeds** (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model’s internal embedding lookup matrix.
* **num\_hashes** (`int`, *optional*) —
  The number of hashing rounds that should be performed during bucketing. Setting this argument overwrites
  the default defined in `config.num_hashes`.

  For more information, see `num_hashes` in [ReformerConfig](/docs/transformers/v4.56.2/en/model_doc/reformer#transformers.ReformerConfig).
* **labels** (`torch.LongTensor` of shape `(batch_size,)`, *optional*) —
  Labels for computing the sequence classification/regression loss. Indices should be in `[0, ..., config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
  `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

[transformers.modeling\_outputs.SequenceClassifierOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.SequenceClassifierOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([ReformerConfig](/docs/transformers/v4.56.2/en/model_doc/reformer#transformers.ReformerConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Classification (or regression if config.num\_labels==1) loss.
* **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) — Classification (or regression if config.num\_labels==1) scores (before SoftMax).
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [ReformerForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/reformer#transformers.ReformerForSequenceClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example of single-label classification:


```
>>> import torch
>>> from transformers import AutoTokenizer, ReformerForSequenceClassification

>>> tokenizer = AutoTokenizer.from_pretrained("google/reformer-crime-and-punishment")
>>> model = ReformerForSequenceClassification.from_pretrained("google/reformer-crime-and-punishment")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> predicted_class_id = logits.argmax().item()
>>> label = model.config.id2label[predicted_class_id]
```


```
>>> # To train a model on `num_labels` classes, you can pass `num_labels=num_labels` to `.from_pretrained(...)`
>>> num_labels = len(model.config.id2label)
>>> model = ReformerForSequenceClassification.from_pretrained(
...     "google/reformer-crime-and-punishment", num_labels=num_labels
... )

>>> labels = torch.tensor(1)
>>> loss = model(**inputs, labels=labels).loss
```

## ReformerForQuestionAnswering

### class transformers.ReformerForQuestionAnswering

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/reformer/modeling_reformer.py#L2669)

( config  )

Parameters

* **config** ([ReformerForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/reformer#transformers.ReformerForQuestionAnswering)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Reformer transformer with a span classification head on top for extractive question-answering tasks like
SQuAD (a linear layer on top of the hidden-states output to compute `span start logits` and `span end logits`).

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/reformer/modeling_reformer.py#L2681)

( input\_ids: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.Tensor] = None attention\_mask: typing.Optional[torch.Tensor] = None head\_mask: typing.Optional[torch.Tensor] = None inputs\_embeds: typing.Optional[torch.Tensor] = None num\_hashes: typing.Optional[int] = None start\_positions: typing.Optional[torch.Tensor] = None end\_positions: typing.Optional[torch.Tensor] = None output\_hidden\_states: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → [transformers.modeling\_outputs.QuestionAnsweringModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.QuestionAnsweringModelOutput) or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) —
  Indices of input sequence tokens in the vocabulary. During training the input\_ids sequence\_length has to be
  a multiple of the relevant model’s chunk lengths (lsh’s, local’s or both). During evaluation, the indices
  are automatically padded to be a multiple of the chunk length.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **position\_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **head\_mask** (`torch.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **inputs\_embeds** (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model’s internal embedding lookup matrix.
* **num\_hashes** (`int`, *optional*) —
  The number of hashing rounds that should be performed during bucketing. Setting this argument overwrites
  the default defined in `config.num_hashes`.

  For more information, see `num_hashes` in [ReformerConfig](/docs/transformers/v4.56.2/en/model_doc/reformer#transformers.ReformerConfig).
* **start\_positions** (`torch.Tensor` of shape `(batch_size,)`, *optional*) —
  Labels for position (index) of the start of the labelled span for computing the token classification loss.
  Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
  are not taken into account for computing the loss.
* **end\_positions** (`torch.Tensor` of shape `(batch_size,)`, *optional*) —
  Labels for position (index) of the end of the labelled span for computing the token classification loss.
  Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
  are not taken into account for computing the loss.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

[transformers.modeling\_outputs.QuestionAnsweringModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.QuestionAnsweringModelOutput) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.QuestionAnsweringModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.QuestionAnsweringModelOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([ReformerConfig](/docs/transformers/v4.56.2/en/model_doc/reformer#transformers.ReformerConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
* **start\_logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`) — Span-start scores (before SoftMax).
* **end\_logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`) — Span-end scores (before SoftMax).
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [ReformerForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/reformer#transformers.ReformerForQuestionAnswering) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> from transformers import AutoTokenizer, ReformerForQuestionAnswering
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("google/reformer-crime-and-punishment")
>>> model = ReformerForQuestionAnswering.from_pretrained("google/reformer-crime-and-punishment")

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

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/reformer.md)
