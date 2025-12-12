*This model was released on 2021-06-09 and added to Hugging Face Transformers on 2023-06-20.*

# Graphormer

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white)

This model is in maintenance mode only, we don’t accept any new PRs changing its code.
If you run into any issues running this model, please reinstall the last version that supported this model: v4.40.2.
You can do so by running the following command: `pip install -U transformers==4.40.2`.

## Overview

The Graphormer model was proposed in [Do Transformers Really Perform Bad for Graph Representation?](https://huggingface.co/papers/2106.05234) by
Chengxuan Ying, Tianle Cai, Shengjie Luo, Shuxin Zheng, Guolin Ke, Di He, Yanming Shen and Tie-Yan Liu. It is a Graph Transformer model, modified to allow computations on graphs instead of text sequences by generating embeddings and features of interest during preprocessing and collation, then using a modified attention.

The abstract from the paper is the following:

*The Transformer architecture has become a dominant choice in many domains, such as natural language processing and computer vision. Yet, it has not achieved competitive performance on popular leaderboards of graph-level prediction compared to mainstream GNN variants. Therefore, it remains a mystery how Transformers could perform well for graph representation learning. In this paper, we solve this mystery by presenting Graphormer, which is built upon the standard Transformer architecture, and could attain excellent results on a broad range of graph representation learning tasks, especially on the recent OGB Large-Scale Challenge. Our key insight to utilizing Transformer in the graph is the necessity of effectively encoding the structural information of a graph into the model. To this end, we propose several simple yet effective structural encoding methods to help Graphormer better model graph-structured data. Besides, we mathematically characterize the expressive power of Graphormer and exhibit that with our ways of encoding the structural information of graphs, many popular GNN variants could be covered as the special cases of Graphormer.*

This model was contributed by [clefourrier](https://huggingface.co/clefourrier). The original code can be found [here](https://github.com/microsoft/Graphormer).

## Usage tips

This model will not work well on large graphs (more than 100 nodes/edges), as it will make the memory explode.
You can reduce the batch size, increase your RAM, or decrease the `UNREACHABLE_NODE_DISTANCE` parameter in algos\_graphormer.pyx, but it will be hard to go above 700 nodes/edges.

This model does not use a tokenizer, but instead a special collator during training.

## GraphormerConfig

### class transformers.GraphormerConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/graphormer/configuration_graphormer.py#L26)

( num\_classes: int = 1 num\_atoms: int = 4608 num\_edges: int = 1536 num\_in\_degree: int = 512 num\_out\_degree: int = 512 num\_spatial: int = 512 num\_edge\_dis: int = 128 multi\_hop\_max\_dist: int = 5 spatial\_pos\_max: int = 1024 edge\_type: str = 'multi\_hop' max\_nodes: int = 512 share\_input\_output\_embed: bool = False num\_hidden\_layers: int = 12 embedding\_dim: int = 768 ffn\_embedding\_dim: int = 768 num\_attention\_heads: int = 32 dropout: float = 0.1 attention\_dropout: float = 0.1 activation\_dropout: float = 0.1 layerdrop: float = 0.0 encoder\_normalize\_before: bool = False pre\_layernorm: bool = False apply\_graphormer\_init: bool = False activation\_fn: str = 'gelu' embed\_scale: typing.Optional[float] = None freeze\_embeddings: bool = False num\_trans\_layers\_to\_freeze: int = 0 traceable: bool = False q\_noise: float = 0.0 qn\_block\_size: int = 8 kdim: typing.Optional[int] = None vdim: typing.Optional[int] = None bias: bool = True self\_attention: bool = True pad\_token\_id = 0 bos\_token\_id = 1 eos\_token\_id = 2 \*\*kwargs  )

Parameters

* **num\_classes** (`int`, *optional*, defaults to 1) —
  Number of target classes or labels, set to n for binary classification of n tasks.
* **num\_atoms** (`int`, *optional*, defaults to 512\*9) —
  Number of node types in the graphs.
* **num\_edges** (`int`, *optional*, defaults to 512\*3) —
  Number of edges types in the graph.
* **num\_in\_degree** (`int`, *optional*, defaults to 512) —
  Number of in degrees types in the input graphs.
* **num\_out\_degree** (`int`, *optional*, defaults to 512) —
  Number of out degrees types in the input graphs.
* **num\_edge\_dis** (`int`, *optional*, defaults to 128) —
  Number of edge dis in the input graphs.
* **multi\_hop\_max\_dist** (`int`, *optional*, defaults to 20) —
  Maximum distance of multi hop edges between two nodes.
* **spatial\_pos\_max** (`int`, *optional*, defaults to 1024) —
  Maximum distance between nodes in the graph attention bias matrices, used during preprocessing and
  collation.
* **edge\_type** (`str`, *optional*, defaults to multihop) —
  Type of edge relation chosen.
* **max\_nodes** (`int`, *optional*, defaults to 512) —
  Maximum number of nodes which can be parsed for the input graphs.
* **share\_input\_output\_embed** (`bool`, *optional*, defaults to `False`) —
  Shares the embedding layer between encoder and decoder - careful, True is not implemented.
* **num\_layers** (`int`, *optional*, defaults to 12) —
  Number of layers.
* **embedding\_dim** (`int`, *optional*, defaults to 768) —
  Dimension of the embedding layer in encoder.
* **ffn\_embedding\_dim** (`int`, *optional*, defaults to 768) —
  Dimension of the “intermediate” (often named feed-forward) layer in encoder.
* **num\_attention\_heads** (`int`, *optional*, defaults to 32) —
  Number of attention heads in the encoder.
* **self\_attention** (`bool`, *optional*, defaults to `True`) —
  Model is self attentive (False not implemented).
* **activation\_function** (`str` or `function`, *optional*, defaults to `"gelu"`) —
  The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
  `"relu"`, `"silu"` and `"gelu_new"` are supported.
* **dropout** (`float`, *optional*, defaults to 0.1) —
  The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
* **attention\_dropout** (`float`, *optional*, defaults to 0.1) —
  The dropout probability for the attention weights.
* **activation\_dropout** (`float`, *optional*, defaults to 0.1) —
  The dropout probability for the activation of the linear transformer layer.
* **layerdrop** (`float`, *optional*, defaults to 0.0) —
  The LayerDrop probability for the encoder. See the [LayerDrop paper](see <https://huggingface.co/papers/1909.11556>)
  for more details.
* **bias** (`bool`, *optional*, defaults to `True`) —
  Uses bias in the attention module - unsupported at the moment.
* **embed\_scale(`float`,** *optional*, defaults to None) —
  Scaling factor for the node embeddings.
* **num\_trans\_layers\_to\_freeze** (`int`, *optional*, defaults to 0) —
  Number of transformer layers to freeze.
* **encoder\_normalize\_before** (`bool`, *optional*, defaults to `False`) —
  Normalize features before encoding the graph.
* **pre\_layernorm** (`bool`, *optional*, defaults to `False`) —
  Apply layernorm before self attention and the feed forward network. Without this, post layernorm will be
  used.
* **apply\_graphormer\_init** (`bool`, *optional*, defaults to `False`) —
  Apply a custom graphormer initialisation to the model before training.
* **freeze\_embeddings** (`bool`, *optional*, defaults to `False`) —
  Freeze the embedding layer, or train it along the model.
* **encoder\_normalize\_before** (`bool`, *optional*, defaults to `False`) —
  Apply the layer norm before each encoder block.
* **q\_noise** (`float`, *optional*, defaults to 0.0) —
  Amount of quantization noise (see “Training with Quantization Noise for Extreme Model Compression”). (For
  more detail, see fairseq’s documentation on quant\_noise).
* **qn\_block\_size** (`int`, *optional*, defaults to 8) —
  Size of the blocks for subsequent quantization with iPQ (see q\_noise).
* **kdim** (`int`, *optional*, defaults to None) —
  Dimension of the key in the attention, if different from the other values.
* **vdim** (`int`, *optional*, defaults to None) —
  Dimension of the value in the attention, if different from the other values.
* **use\_cache** (`bool`, *optional*, defaults to `True`) —
  Whether or not the model should return the last key/values attentions (not used by all models).
* **traceable** (`bool`, *optional*, defaults to `False`) —
  Changes return value of the encoder’s inner\_state to stacked tensors.
* **Example** —

This is the configuration class to store the configuration of a [~GraphormerModel](/docs/transformers/v4.56.2/en/model_doc/graphormer#transformers.GraphormerModel). It is used to instantiate an
Graphormer model according to the specified arguments, defining the model architecture. Instantiating a
configuration with the defaults will yield a similar configuration to that of the Graphormer
[graphormer-base-pcqm4mv1](https://huggingface.co/graphormer-base-pcqm4mv1) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

## GraphormerModel

### class transformers.GraphormerModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/graphormer/modeling_graphormer.py#L769)

( config: GraphormerConfig  )

The Graphormer model is a graph-encoder model.

It goes from a graph to its representation. If you want to use the model for a downstream classification task, use
GraphormerForGraphClassification instead. For any other downstream task, feel free to add a new class, or combine
this model with a downstream model of your choice, following the example in GraphormerForGraphClassification.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/graphormer/modeling_graphormer.py#L798)

( input\_nodes: LongTensor input\_edges: LongTensor attn\_bias: Tensor in\_degree: LongTensor out\_degree: LongTensor spatial\_pos: LongTensor attn\_edge\_type: LongTensor perturb: typing.Optional[torch.FloatTensor] = None masked\_tokens: None = None return\_dict: typing.Optional[bool] = None \*\*unused  )

## GraphormerForGraphClassification

### class transformers.GraphormerForGraphClassification

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/graphormer/modeling_graphormer.py#L840)

( config: GraphormerConfig  )

This model can be used for graph-level classification or regression tasks.

It can be trained on

* regression (by setting config.num\_classes to 1); there should be one float-type label per graph
* one task classification (by setting config.num\_classes to the number of classes); there should be one integer
  label per graph
* binary multi-task classification (by setting config.num\_classes to the number of labels); there should be a list
  of integer labels for each graph.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/graphormer/modeling_graphormer.py#L863)

( input\_nodes: LongTensor input\_edges: LongTensor attn\_bias: Tensor in\_degree: LongTensor out\_degree: LongTensor spatial\_pos: LongTensor attn\_edge\_type: LongTensor labels: typing.Optional[torch.LongTensor] = None return\_dict: typing.Optional[bool] = None \*\*unused  )

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/graphormer.md)
