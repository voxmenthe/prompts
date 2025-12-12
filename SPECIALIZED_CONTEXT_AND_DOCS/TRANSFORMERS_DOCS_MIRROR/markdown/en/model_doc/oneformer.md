*This model was released on 2022-11-10 and added to Hugging Face Transformers on 2023-01-19.*

# OneFormer

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white)

## Overview

The OneFormer model was proposed in [OneFormer: One Transformer to Rule Universal Image Segmentation](https://huggingface.co/papers/2211.06220) by Jitesh Jain, Jiachen Li, MangTik Chiu, Ali Hassani, Nikita Orlov, Humphrey Shi. OneFormer is a universal image segmentation framework that can be trained on a single panoptic dataset to perform semantic, instance, and panoptic segmentation tasks. OneFormer uses a task token to condition the model on the task in focus, making the architecture task-guided for training, and task-dynamic for inference.

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/oneformer_teaser.png)

The abstract from the paper is the following:

*Universal Image Segmentation is not a new concept. Past attempts to unify image segmentation in the last decades include scene parsing, panoptic segmentation, and, more recently, new panoptic architectures. However, such panoptic architectures do not truly unify image segmentation because they need to be trained individually on the semantic, instance, or panoptic segmentation to achieve the best performance. Ideally, a truly universal framework should be trained only once and achieve SOTA performance across all three image segmentation tasks. To that end, we propose OneFormer, a universal image segmentation framework that unifies segmentation with a multi-task train-once design. We first propose a task-conditioned joint training strategy that enables training on ground truths of each domain (semantic, instance, and panoptic segmentation) within a single multi-task training process. Secondly, we introduce a task token to condition our model on the task at hand, making our model task-dynamic to support multi-task training and inference. Thirdly, we propose using a query-text contrastive loss during training to establish better inter-task and inter-class distinctions. Notably, our single OneFormer model outperforms specialized Mask2Former models across all three segmentation tasks on ADE20k, CityScapes, and COCO, despite the latter being trained on each of the three tasks individually with three times the resources. With new ConvNeXt and DiNAT backbones, we observe even more performance improvement. We believe OneFormer is a significant step towards making image segmentation more universal and accessible.*

The figure below illustrates the architecture of OneFormer. Taken from the [original paper](https://huggingface.co/papers/2211.06220).

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/oneformer_architecture.png)

This model was contributed by [Jitesh Jain](https://huggingface.co/praeclarumjj3). The original code can be found [here](https://github.com/SHI-Labs/OneFormer).

## Usage tips

* OneFormer requires two inputs during inference: *image* and *task token*.
* During training, OneFormer only uses panoptic annotations.
* If you want to train the model in a distributed environment across multiple nodes, then one should update the
  `get_num_masks` function inside in the `OneFormerLoss` class of `modeling_oneformer.py`. When training on multiple nodes, this should be
  set to the average number of target masks across all nodes, as can be seen in the original implementation [here](https://github.com/SHI-Labs/OneFormer/blob/33ebb56ed34f970a30ae103e786c0cb64c653d9a/oneformer/modeling/criterion.py#L287).
* One can use [OneFormerProcessor](/docs/transformers/v4.56.2/en/model_doc/oneformer#transformers.OneFormerProcessor) to prepare input images and task inputs for the model and optional targets for the model. [OneFormerProcessor](/docs/transformers/v4.56.2/en/model_doc/oneformer#transformers.OneFormerProcessor) wraps [OneFormerImageProcessor](/docs/transformers/v4.56.2/en/model_doc/oneformer#transformers.OneFormerImageProcessor) and [CLIPTokenizer](/docs/transformers/v4.56.2/en/model_doc/clip#transformers.CLIPTokenizer) into a single instance to both prepare the images and encode the task inputs.
* To get the final segmentation, depending on the task, you can call [post\_process\_semantic\_segmentation()](/docs/transformers/v4.56.2/en/model_doc/oneformer#transformers.OneFormerProcessor.post_process_semantic_segmentation) or [post\_process\_instance\_segmentation()](/docs/transformers/v4.56.2/en/model_doc/oneformer#transformers.OneFormerImageProcessor.post_process_instance_segmentation) or [post\_process\_panoptic\_segmentation()](/docs/transformers/v4.56.2/en/model_doc/oneformer#transformers.OneFormerImageProcessor.post_process_panoptic_segmentation). All three tasks can be solved using [OneFormerForUniversalSegmentation](/docs/transformers/v4.56.2/en/model_doc/oneformer#transformers.OneFormerForUniversalSegmentation) output, panoptic segmentation accepts an optional `label_ids_to_fuse` argument to fuse instances of the target object/s (e.g. sky) together.

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with OneFormer.

* Demo notebooks regarding inference + fine-tuning on custom data can be found [here](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/OneFormer).

If you’re interested in submitting a resource to be included here, please feel free to open a Pull Request and we will review it.
The resource should ideally demonstrate something new instead of duplicating an existing resource.

## OneFormer specific outputs

### class transformers.models.oneformer.modeling\_oneformer.OneFormerModelOutput

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/oneformer/modeling_oneformer.py#L820)

( encoder\_hidden\_states: typing.Optional[tuple[torch.FloatTensor]] = None pixel\_decoder\_hidden\_states: typing.Optional[tuple[torch.FloatTensor]] = None transformer\_decoder\_hidden\_states: typing.Optional[torch.FloatTensor] = None transformer\_decoder\_object\_queries: typing.Optional[torch.FloatTensor] = None transformer\_decoder\_contrastive\_queries: typing.Optional[torch.FloatTensor] = None transformer\_decoder\_mask\_predictions: typing.Optional[torch.FloatTensor] = None transformer\_decoder\_class\_predictions: typing.Optional[torch.FloatTensor] = None transformer\_decoder\_auxiliary\_predictions: typing.Optional[tuple[dict[str, torch.FloatTensor]]] = None text\_queries: typing.Optional[torch.FloatTensor] = None task\_token: typing.Optional[torch.FloatTensor] = None attentions: typing.Optional[tuple[torch.FloatTensor]] = None  )

Parameters

* **encoder\_hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) —
  Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
  shape `(batch_size, num_channels, height, width)`. Hidden-states (also called feature maps) of the encoder
  model at the output of each stage.
* **pixel\_decoder\_hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) —
  Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
  shape `(batch_size, num_channels, height, width)`. Hidden-states (also called feature maps) of the pixel
  decoder model at the output of each stage.
* **transformer\_decoder\_hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) —
  Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
  shape `(batch_size, sequence_length, hidden_size)`. Hidden-states (also called feature maps) of the
  transformer decoder at the output of each stage.
* **transformer\_decoder\_object\_queries** (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_dim)`) —
  Output object queries from the last layer in the transformer decoder.
* **transformer\_decoder\_contrastive\_queries** (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_dim)`) —
  Contrastive queries from the transformer decoder.
* **transformer\_decoder\_mask\_predictions** (`torch.FloatTensor` of shape `(batch_size, num_queries, height, width)`) —
  Mask Predictions from the last layer in the transformer decoder.
* **transformer\_decoder\_class\_predictions** (`torch.FloatTensor` of shape `(batch_size, num_queries, num_classes+1)`) —
  Class Predictions from the last layer in the transformer decoder.
* **transformer\_decoder\_auxiliary\_predictions** (`Tuple` of Dict of `str, torch.FloatTensor`, *optional*) —
  Tuple of class and mask predictions from each layer of the transformer decoder.
* **text\_queries** (`torch.FloatTensor`, *optional* of shape `(batch_size, num_queries, hidden_dim)`) —
  Text queries derived from the input text list used for calculating contrastive loss during training.
* **task\_token** (`torch.FloatTensor` of shape `(batch_size, hidden_dim)`) —
  1D task token to condition the queries.
* **attentions** (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) —
  Tuple of `tuple(torch.FloatTensor)` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`. Self and Cross Attentions weights from transformer decoder.

Class for outputs of [OneFormerModel](/docs/transformers/v4.56.2/en/model_doc/oneformer#transformers.OneFormerModel). This class returns all the needed hidden states to compute the logits.

### class transformers.models.oneformer.modeling\_oneformer.OneFormerForUniversalSegmentationOutput

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/oneformer/modeling_oneformer.py#L877)

( loss: typing.Optional[torch.FloatTensor] = None class\_queries\_logits: typing.Optional[torch.FloatTensor] = None masks\_queries\_logits: typing.Optional[torch.FloatTensor] = None auxiliary\_predictions: list = None encoder\_hidden\_states: typing.Optional[tuple[torch.FloatTensor]] = None pixel\_decoder\_hidden\_states: typing.Optional[list[torch.FloatTensor]] = None transformer\_decoder\_hidden\_states: typing.Optional[torch.FloatTensor] = None transformer\_decoder\_object\_queries: typing.Optional[torch.FloatTensor] = None transformer\_decoder\_contrastive\_queries: typing.Optional[torch.FloatTensor] = None transformer\_decoder\_mask\_predictions: typing.Optional[torch.FloatTensor] = None transformer\_decoder\_class\_predictions: typing.Optional[torch.FloatTensor] = None transformer\_decoder\_auxiliary\_predictions: typing.Optional[list[dict[str, torch.FloatTensor]]] = None text\_queries: typing.Optional[torch.FloatTensor] = None task\_token: typing.Optional[torch.FloatTensor] = None attentions: typing.Optional[tuple[tuple[torch.FloatTensor]]] = None  )

Parameters

* **loss** (`torch.Tensor`, *optional*) —
  The computed loss, returned when labels are present.
* **class\_queries\_logits** (`torch.FloatTensor`, *optional*, defaults to `None`) —
  A tensor of shape `(batch_size, num_queries, num_labels + 1)` representing the proposed classes for each
  query. Note the `+ 1` is needed because we incorporate the null class.
* **masks\_queries\_logits** (`torch.FloatTensor`, *optional*, defaults to `None`) —
  A tensor of shape `(batch_size, num_queries, height, width)` representing the proposed masks for each
  query.
* **auxiliary\_predictions** (`List` of Dict of `str, torch.FloatTensor`, *optional*) —
  List of class and mask predictions from each layer of the transformer decoder.
* **encoder\_hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) —
  Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
  shape `(batch_size, num_channels, height, width)`. Hidden-states (also called feature maps) of the encoder
  model at the output of each stage.
* **pixel\_decoder\_hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) —
  Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
  shape `(batch_size, num_channels, height, width)`. Hidden-states (also called feature maps) of the pixel
  decoder model at the output of each stage.
* **transformer\_decoder\_hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) —
  Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
  shape `(batch_size, sequence_length, hidden_size)`. Hidden-states (also called feature maps) of the
  transformer decoder at the output of each stage.
* **transformer\_decoder\_object\_queries** (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_dim)`) —
  Output object queries from the last layer in the transformer decoder.
* **transformer\_decoder\_contrastive\_queries** (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_dim)`) —
  Contrastive queries from the transformer decoder.
* **transformer\_decoder\_mask\_predictions** (`torch.FloatTensor` of shape `(batch_size, num_queries, height, width)`) —
  Mask Predictions from the last layer in the transformer decoder.
* **transformer\_decoder\_class\_predictions** (`torch.FloatTensor` of shape `(batch_size, num_queries, num_classes+1)`) —
  Class Predictions from the last layer in the transformer decoder.
* **transformer\_decoder\_auxiliary\_predictions** (`List` of Dict of `str, torch.FloatTensor`, *optional*) —
  List of class and mask predictions from each layer of the transformer decoder.
* **text\_queries** (`torch.FloatTensor`, *optional* of shape `(batch_size, num_queries, hidden_dim)`) —
  Text queries derived from the input text list used for calculating contrastive loss during training.
* **task\_token** (`torch.FloatTensor` of shape `(batch_size, hidden_dim)`) —
  1D task token to condition the queries.
* **attentions** (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) —
  Tuple of `tuple(torch.FloatTensor)` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`. Self and Cross Attentions weights from transformer decoder.

Class for outputs of `OneFormerForUniversalSegmentationOutput`.

This output can be directly passed to [post\_process\_semantic\_segmentation()](/docs/transformers/v4.56.2/en/model_doc/oneformer#transformers.OneFormerImageProcessor.post_process_semantic_segmentation) or
[post\_process\_instance\_segmentation()](/docs/transformers/v4.56.2/en/model_doc/oneformer#transformers.OneFormerImageProcessor.post_process_instance_segmentation) or
[post\_process\_panoptic\_segmentation()](/docs/transformers/v4.56.2/en/model_doc/oneformer#transformers.OneFormerImageProcessor.post_process_panoptic_segmentation) depending on the task. Please, see
[`~OneFormerImageProcessor] for details regarding usage.

## OneFormerConfig

### class transformers.OneFormerConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/oneformer/configuration_oneformer.py#L28)

( backbone\_config: typing.Optional[dict] = None backbone: typing.Optional[str] = None use\_pretrained\_backbone: bool = False use\_timm\_backbone: bool = False backbone\_kwargs: typing.Optional[dict] = None ignore\_value: int = 255 num\_queries: int = 150 no\_object\_weight: int = 0.1 class\_weight: float = 2.0 mask\_weight: float = 5.0 dice\_weight: float = 5.0 contrastive\_weight: float = 0.5 contrastive\_temperature: float = 0.07 train\_num\_points: int = 12544 oversample\_ratio: float = 3.0 importance\_sample\_ratio: float = 0.75 init\_std: float = 0.02 init\_xavier\_std: float = 1.0 layer\_norm\_eps: float = 1e-05 is\_training: bool = False use\_auxiliary\_loss: bool = True output\_auxiliary\_logits: bool = True strides: typing.Optional[list] = [4, 8, 16, 32] task\_seq\_len: int = 77 text\_encoder\_width: int = 256 text\_encoder\_context\_length: int = 77 text\_encoder\_num\_layers: int = 6 text\_encoder\_vocab\_size: int = 49408 text\_encoder\_proj\_layers: int = 2 text\_encoder\_n\_ctx: int = 16 conv\_dim: int = 256 mask\_dim: int = 256 hidden\_dim: int = 256 encoder\_feedforward\_dim: int = 1024 norm: str = 'GN' encoder\_layers: int = 6 decoder\_layers: int = 10 use\_task\_norm: bool = True num\_attention\_heads: int = 8 dropout: float = 0.1 dim\_feedforward: int = 2048 pre\_norm: bool = False enforce\_input\_proj: bool = False query\_dec\_layers: int = 2 common\_stride: int = 4 \*\*kwargs  )

Parameters

* **backbone\_config** (`PretrainedConfig`, *optional*, defaults to `SwinConfig`) —
  The configuration of the backbone model.
* **backbone** (`str`, *optional*) —
  Name of backbone to use when `backbone_config` is `None`. If `use_pretrained_backbone` is `True`, this
  will load the corresponding pretrained weights from the timm or transformers library. If `use_pretrained_backbone`
  is `False`, this loads the backbone’s config and uses that to initialize the backbone with random weights.
* **use\_pretrained\_backbone** (`bool`, *optional*, defaults to `False`) —
  Whether to use pretrained weights for the backbone.
* **use\_timm\_backbone** (`bool`, *optional*, defaults to `False`) —
  Whether to load `backbone` from the timm library. If `False`, the backbone is loaded from the transformers
  library.
* **backbone\_kwargs** (`dict`, *optional*) —
  Keyword arguments to be passed to AutoBackbone when loading from a checkpoint
  e.g. `{'out_indices': (0, 1, 2, 3)}`. Cannot be specified if `backbone_config` is set.
* **ignore\_value** (`int`, *optional*, defaults to 255) —
  Values to be ignored in GT label while calculating loss.
* **num\_queries** (`int`, *optional*, defaults to 150) —
  Number of object queries.
* **no\_object\_weight** (`float`, *optional*, defaults to 0.1) —
  Weight for no-object class predictions.
* **class\_weight** (`float`, *optional*, defaults to 2.0) —
  Weight for Classification CE loss.
* **mask\_weight** (`float`, *optional*, defaults to 5.0) —
  Weight for binary CE loss.
* **dice\_weight** (`float`, *optional*, defaults to 5.0) —
  Weight for dice loss.
* **contrastive\_weight** (`float`, *optional*, defaults to 0.5) —
  Weight for contrastive loss.
* **contrastive\_temperature** (`float`, *optional*, defaults to 0.07) —
  Initial value for scaling the contrastive logits.
* **train\_num\_points** (`int`, *optional*, defaults to 12544) —
  Number of points to sample while calculating losses on mask predictions.
* **oversample\_ratio** (`float`, *optional*, defaults to 3.0) —
  Ratio to decide how many points to oversample.
* **importance\_sample\_ratio** (`float`, *optional*, defaults to 0.75) —
  Ratio of points that are sampled via importance sampling.
* **init\_std** (`float`, *optional*, defaults to 0.02) —
  Standard deviation for normal initialization.
* **init\_xavier\_std** (`float`, *optional*, defaults to 1.0) —
  Standard deviation for xavier uniform initialization.
* **layer\_norm\_eps** (`float`, *optional*, defaults to 1e-05) —
  Epsilon for layer normalization.
* **is\_training** (`bool`, *optional*, defaults to `False`) —
  Whether to run in training or inference mode.
* **use\_auxiliary\_loss** (`bool`, *optional*, defaults to `True`) —
  Whether to calculate loss using intermediate predictions from transformer decoder.
* **output\_auxiliary\_logits** (`bool`, *optional*, defaults to `True`) —
  Whether to return intermediate predictions from transformer decoder.
* **strides** (`list`, *optional*, defaults to `[4, 8, 16, 32]`) —
  List containing the strides for feature maps in the encoder.
* **task\_seq\_len** (`int`, *optional*, defaults to 77) —
  Sequence length for tokenizing text list input.
* **text\_encoder\_width** (`int`, *optional*, defaults to 256) —
  Hidden size for text encoder.
* **text\_encoder\_context\_length** (`int`, *optional*, defaults to 77) —
  Input sequence length for text encoder.
* **text\_encoder\_num\_layers** (`int`, *optional*, defaults to 6) —
  Number of layers for transformer in text encoder.
* **text\_encoder\_vocab\_size** (`int`, *optional*, defaults to 49408) —
  Vocabulary size for tokenizer.
* **text\_encoder\_proj\_layers** (`int`, *optional*, defaults to 2) —
  Number of layers in MLP for project text queries.
* **text\_encoder\_n\_ctx** (`int`, *optional*, defaults to 16) —
  Number of learnable text context queries.
* **conv\_dim** (`int`, *optional*, defaults to 256) —
  Feature map dimension to map outputs from the backbone.
* **mask\_dim** (`int`, *optional*, defaults to 256) —
  Dimension for feature maps in pixel decoder.
* **hidden\_dim** (`int`, *optional*, defaults to 256) —
  Dimension for hidden states in transformer decoder.
* **encoder\_feedforward\_dim** (`int`, *optional*, defaults to 1024) —
  Dimension for FFN layer in pixel decoder.
* **norm** (`str`, *optional*, defaults to `"GN"`) —
  Type of normalization.
* **encoder\_layers** (`int`, *optional*, defaults to 6) —
  Number of layers in pixel decoder.
* **decoder\_layers** (`int`, *optional*, defaults to 10) —
  Number of layers in transformer decoder.
* **use\_task\_norm** (`bool`, *optional*, defaults to `True`) —
  Whether to normalize the task token.
* **num\_attention\_heads** (`int`, *optional*, defaults to 8) —
  Number of attention heads in transformer layers in the pixel and transformer decoders.
* **dropout** (`float`, *optional*, defaults to 0.1) —
  Dropout probability for pixel and transformer decoders.
* **dim\_feedforward** (`int`, *optional*, defaults to 2048) —
  Dimension for FFN layer in transformer decoder.
* **pre\_norm** (`bool`, *optional*, defaults to `False`) —
  Whether to normalize hidden states before attention layers in transformer decoder.
* **enforce\_input\_proj** (`bool`, *optional*, defaults to `False`) —
  Whether to project hidden states in transformer decoder.
* **query\_dec\_layers** (`int`, *optional*, defaults to 2) —
  Number of layers in query transformer.
* **common\_stride** (`int`, *optional*, defaults to 4) —
  Common stride used for features in pixel decoder.

This is the configuration class to store the configuration of a [OneFormerModel](/docs/transformers/v4.56.2/en/model_doc/oneformer#transformers.OneFormerModel). It is used to instantiate a
OneFormer model according to the specified arguments, defining the model architecture. Instantiating a
configuration with the defaults will yield a similar configuration to that of the OneFormer
[shi-labs/oneformer\_ade20k\_swin\_tiny](https://huggingface.co/shi-labs/oneformer_ade20k_swin_tiny) architecture
trained on [ADE20k-150](https://huggingface.co/datasets/scene_parse_150).

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Examples:


```
>>> from transformers import OneFormerConfig, OneFormerModel

>>> # Initializing a OneFormer shi-labs/oneformer_ade20k_swin_tiny configuration
>>> configuration = OneFormerConfig()
>>> # Initializing a model (with random weights) from the shi-labs/oneformer_ade20k_swin_tiny style configuration
>>> model = OneFormerModel(configuration)
>>> # Accessing the model configuration
>>> configuration = model.config
```

## OneFormerImageProcessor

### class transformers.OneFormerImageProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/oneformer/image_processing_oneformer.py#L375)

( do\_resize: bool = True size: typing.Optional[dict[str, int]] = None resample: Resampling = <Resampling.BILINEAR: 2> do\_rescale: bool = True rescale\_factor: float = 0.00392156862745098 do\_normalize: bool = True image\_mean: typing.Union[float, list[float], NoneType] = None image\_std: typing.Union[float, list[float], NoneType] = None ignore\_index: typing.Optional[int] = None do\_reduce\_labels: bool = False repo\_path: typing.Optional[str] = 'shi-labs/oneformer\_demo' class\_info\_file: typing.Optional[str] = None num\_text: typing.Optional[int] = None num\_labels: typing.Optional[int] = None \*\*kwargs  )

Parameters

* **do\_resize** (`bool`, *optional*, defaults to `True`) —
  Whether to resize the input to a certain `size`.
* **size** (`int`, *optional*, defaults to 800) —
  Resize the input to the given size. Only has an effect if `do_resize` is set to `True`. If size is a
  sequence like `(width, height)`, output size will be matched to this. If size is an int, smaller edge of
  the image will be matched to this number. i.e, if `height > width`, then image will be rescaled to `(size * height / width, size)`.
* **resample** (`int`, *optional*, defaults to `Resampling.BILINEAR`) —
  An optional resampling filter. This can be one of `PIL.Image.Resampling.NEAREST`,
  `PIL.Image.Resampling.BOX`, `PIL.Image.Resampling.BILINEAR`, `PIL.Image.Resampling.HAMMING`,
  `PIL.Image.Resampling.BICUBIC` or `PIL.Image.Resampling.LANCZOS`. Only has an effect if `do_resize` is set
  to `True`.
* **do\_rescale** (`bool`, *optional*, defaults to `True`) —
  Whether to rescale the input to a certain `scale`.
* **rescale\_factor** (`float`, *optional*, defaults to `1/ 255`) —
  Rescale the input by the given factor. Only has an effect if `do_rescale` is set to `True`.
* **do\_normalize** (`bool`, *optional*, defaults to `True`) —
  Whether or not to normalize the input with mean and standard deviation.
* **image\_mean** (`int`, *optional*, defaults to `[0.485, 0.456, 0.406]`) —
  The sequence of means for each channel, to be used when normalizing images. Defaults to the ImageNet mean.
* **image\_std** (`int`, *optional*, defaults to `[0.229, 0.224, 0.225]`) —
  The sequence of standard deviations for each channel, to be used when normalizing images. Defaults to the
  ImageNet std.
* **ignore\_index** (`int`, *optional*) —
  Label to be assigned to background pixels in segmentation maps. If provided, segmentation map pixels
  denoted with 0 (background) will be replaced with `ignore_index`.
* **do\_reduce\_labels** (`bool`, *optional*, defaults to `False`) —
  Whether or not to decrement all label values of segmentation maps by 1. Usually used for datasets where 0
  is used for background, and background itself is not included in all classes of a dataset (e.g. ADE20k).
  The background label will be replaced by `ignore_index`.
* **repo\_path** (`str`, *optional*, defaults to `"shi-labs/oneformer_demo"`) —
  Path to hub repo or local directory containing the JSON file with class information for the dataset.
  If unset, will look for `class_info_file` in the current working directory.
* **class\_info\_file** (`str`, *optional*) —
  JSON file containing class information for the dataset. See `shi-labs/oneformer_demo/cityscapes_panoptic.json` for an example.
* **num\_text** (`int`, *optional*) —
  Number of text entries in the text input list.
* **num\_labels** (`int`, *optional*) —
  The number of labels in the segmentation map.

Constructs a OneFormer image processor. The image processor can be used to prepare image(s), task input(s) and
optional text inputs and targets for the model.

This image processor inherits from [BaseImageProcessor](/docs/transformers/v4.56.2/en/main_classes/image_processor#transformers.BaseImageProcessor) which contains most of the main methods. Users should
refer to this superclass for more information regarding those methods.

#### preprocess

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/oneformer/image_processing_oneformer.py#L665)

( images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']] task\_inputs: typing.Optional[list[str]] = None segmentation\_maps: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor'], NoneType] = None instance\_id\_to\_semantic\_id: typing.Optional[dict[int, int]] = None do\_resize: typing.Optional[bool] = None size: typing.Optional[dict[str, int]] = None resample: Resampling = None do\_rescale: typing.Optional[bool] = None rescale\_factor: typing.Optional[float] = None do\_normalize: typing.Optional[bool] = None image\_mean: typing.Union[float, list[float], NoneType] = None image\_std: typing.Union[float, list[float], NoneType] = None ignore\_index: typing.Optional[int] = None do\_reduce\_labels: typing.Optional[bool] = None return\_tensors: typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None data\_format: typing.Union[str, transformers.image\_utils.ChannelDimension] = <ChannelDimension.FIRST: 'channels\_first'> input\_data\_format: typing.Union[str, transformers.image\_utils.ChannelDimension, NoneType] = None  )

#### post\_process\_semantic\_segmentation

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/oneformer/image_processing_oneformer.py#L1082)

( outputs target\_sizes: typing.Optional[list[tuple[int, int]]] = None  ) → `list[torch.Tensor]`

Parameters

* **outputs** ([MaskFormerForInstanceSegmentation](/docs/transformers/v4.56.2/en/model_doc/maskformer#transformers.MaskFormerForInstanceSegmentation)) —
  Raw outputs of the model.
* **target\_sizes** (`list[tuple[int, int]]`, *optional*) —
  List of length (batch\_size), where each list item (`tuple[int, int]]`) corresponds to the requested
  final size (height, width) of each prediction. If left to None, predictions will not be resized.

Returns

`list[torch.Tensor]`

A list of length `batch_size`, where each item is a semantic segmentation map of shape (height, width)
corresponding to the target\_sizes entry (if `target_sizes` is specified). Each entry of each
`torch.Tensor` correspond to a semantic class id.

Converts the output of [MaskFormerForInstanceSegmentation](/docs/transformers/v4.56.2/en/model_doc/maskformer#transformers.MaskFormerForInstanceSegmentation) into semantic segmentation maps. Only supports
PyTorch.

#### post\_process\_instance\_segmentation

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/oneformer/image_processing_oneformer.py#L1132)

( outputs task\_type: str = 'instance' is\_demo: bool = True threshold: float = 0.5 mask\_threshold: float = 0.5 overlap\_mask\_area\_threshold: float = 0.8 target\_sizes: typing.Optional[list[tuple[int, int]]] = None return\_coco\_annotation: typing.Optional[bool] = False  ) → `list[Dict]`

Parameters

* **outputs** (`OneFormerForUniversalSegmentationOutput`) —
  The outputs from `OneFormerForUniversalSegmentationOutput`.
* **task\_type** (`str`, *optional*, defaults to “instance”) —
  The post processing depends on the task token input. If the `task_type` is “panoptic”, we need to
  ignore the stuff predictions.
* **is\_demo** (`bool`, *optional)*, defaults to `True`) —
  Whether the model is in demo mode. If true, use threshold to predict final masks.
* **threshold** (`float`, *optional*, defaults to 0.5) —
  The probability score threshold to keep predicted instance masks.
* **mask\_threshold** (`float`, *optional*, defaults to 0.5) —
  Threshold to use when turning the predicted masks into binary values.
* **overlap\_mask\_area\_threshold** (`float`, *optional*, defaults to 0.8) —
  The overlap mask area threshold to merge or discard small disconnected parts within each binary
  instance mask.
* **target\_sizes** (`list[Tuple]`, *optional*) —
  List of length (batch\_size), where each list item (`tuple[int, int]]`) corresponds to the requested
  final size (height, width) of each prediction in batch. If left to None, predictions will not be
  resized.
* **return\_coco\_annotation** (`bool`, *optional)*, defaults to `False`) —
  Whether to return predictions in COCO format.

Returns

`list[Dict]`

A list of dictionaries, one per image, each dictionary containing two keys:

* **segmentation** — a tensor of shape `(height, width)` where each pixel represents a `segment_id`, set
  to `None` if no mask if found above `threshold`. If `target_sizes` is specified, segmentation is resized
  to the corresponding `target_sizes` entry.
* **segments\_info** — A dictionary that contains additional information on each segment.
  + **id** — an integer representing the `segment_id`.
  + **label\_id** — An integer representing the label / semantic class id corresponding to `segment_id`.
  + **was\_fused** — a boolean, `True` if `label_id` was in `label_ids_to_fuse`, `False` otherwise.
    Multiple instances of the same class / label were fused and assigned a single `segment_id`.
  + **score** — Prediction score of segment with `segment_id`.

Converts the output of `OneFormerForUniversalSegmentationOutput` into image instance segmentation
predictions. Only supports PyTorch.

#### post\_process\_panoptic\_segmentation

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/oneformer/image_processing_oneformer.py#L1252)

( outputs threshold: float = 0.5 mask\_threshold: float = 0.5 overlap\_mask\_area\_threshold: float = 0.8 label\_ids\_to\_fuse: typing.Optional[set[int]] = None target\_sizes: typing.Optional[list[tuple[int, int]]] = None  ) → `list[Dict]`

Parameters

* **outputs** (`MaskFormerForInstanceSegmentationOutput`) —
  The outputs from [MaskFormerForInstanceSegmentation](/docs/transformers/v4.56.2/en/model_doc/maskformer#transformers.MaskFormerForInstanceSegmentation).
* **threshold** (`float`, *optional*, defaults to 0.5) —
  The probability score threshold to keep predicted instance masks.
* **mask\_threshold** (`float`, *optional*, defaults to 0.5) —
  Threshold to use when turning the predicted masks into binary values.
* **overlap\_mask\_area\_threshold** (`float`, *optional*, defaults to 0.8) —
  The overlap mask area threshold to merge or discard small disconnected parts within each binary
  instance mask.
* **label\_ids\_to\_fuse** (`Set[int]`, *optional*) —
  The labels in this state will have all their instances be fused together. For instance we could say
  there can only be one sky in an image, but several persons, so the label ID for sky would be in that
  set, but not the one for person.
* **target\_sizes** (`list[Tuple]`, *optional*) —
  List of length (batch\_size), where each list item (`tuple[int, int]]`) corresponds to the requested
  final size (height, width) of each prediction in batch. If left to None, predictions will not be
  resized.

Returns

`list[Dict]`

A list of dictionaries, one per image, each dictionary containing two keys:

* **segmentation** — a tensor of shape `(height, width)` where each pixel represents a `segment_id`, set
  to `None` if no mask if found above `threshold`. If `target_sizes` is specified, segmentation is resized
  to the corresponding `target_sizes` entry.
* **segments\_info** — A dictionary that contains additional information on each segment.
  + **id** — an integer representing the `segment_id`.
  + **label\_id** — An integer representing the label / semantic class id corresponding to `segment_id`.
  + **was\_fused** — a boolean, `True` if `label_id` was in `label_ids_to_fuse`, `False` otherwise.
    Multiple instances of the same class / label were fused and assigned a single `segment_id`.
  + **score** — Prediction score of segment with `segment_id`.

Converts the output of `MaskFormerForInstanceSegmentationOutput` into image panoptic segmentation
predictions. Only supports PyTorch.

## OneFormerImageProcessorFast

### class transformers.OneFormerImageProcessorFast

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/oneformer/image_processing_oneformer_fast.py#L337)

( \*\*kwargs: typing\_extensions.Unpack[transformers.models.oneformer.image\_processing\_oneformer\_fast.OneFormerFastImageProcessorKwargs]  )

Constructs a fast Oneformer image processor.

#### preprocess

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/oneformer/image_processing_oneformer_fast.py#L364)

( images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']] task\_inputs: typing.Optional[list[str]] = None segmentation\_maps: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor'], NoneType] = None instance\_id\_to\_semantic\_id: typing.Union[list[dict[int, int]], dict[int, int], NoneType] = None \*\*kwargs: typing\_extensions.Unpack[transformers.models.oneformer.image\_processing\_oneformer\_fast.OneFormerFastImageProcessorKwargs]  ) → `<class 'transformers.image_processing_base.BatchFeature'>`

Parameters

* **images** (`Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']]`) —
  Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
  passing in images with pixel values between 0 and 1, set `do_rescale=False`.
* **task\_inputs** (`list[str]`, *optional*) —
  List of tasks (`"panoptic"`, `"instance"`, `"semantic"`) for each image in the batch.
* **segmentation\_maps** (`ImageInput`, *optional*) —
  The segmentation maps.
* **instance\_id\_to\_semantic\_id** (`Union[list[dict[int, int]], dict[int, int]]`, *optional*) —
  A mapping from instance IDs to semantic IDs.
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
* **repo\_path** (`str`, *optional*, defaults to `shi-labs/oneformer_demo`) —
  Path to a local directory or Hugging Face Hub repository containing model metadata.
* **class\_info\_file** (`str`, *optional*) —
  Path to the JSON file within the repository that contains class metadata.
* **num\_text** (`int`, *optional*) —
  Number of text queries for the text encoder, used as task-guiding prompts.
* **num\_labels** (`int`, *optional*) —
  Number of semantic classes for segmentation, determining the output layer’s size.
* **ignore\_index** (`int`, *optional*) —
  Label to ignore in segmentation maps, often used for padding.
* **do\_reduce\_labels** (`bool`, *optional*, defaults to `False`) —
  Whether to decrement all label values by 1, mapping the background class to `ignore_index`.

Returns

`<class 'transformers.image_processing_base.BatchFeature'>`

* **data** (`dict`) — Dictionary of lists/arrays/tensors returned by the **call** method (‘pixel\_values’, etc.).
* **tensor\_type** (`Union[None, str, TensorType]`, *optional*) — You can give a tensor\_type here to convert the lists of integers in PyTorch/TensorFlow/Numpy Tensors at
  initialization.

#### post\_process\_semantic\_segmentation

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/oneformer/image_processing_oneformer_fast.py#L743)

( outputs target\_sizes: typing.Optional[list[tuple[int, int]]] = None  ) → `List[torch.Tensor]`

Parameters

* **outputs** ([MaskFormerForInstanceSegmentation](/docs/transformers/v4.56.2/en/model_doc/maskformer#transformers.MaskFormerForInstanceSegmentation)) —
  Raw outputs of the model.
* **target\_sizes** (`List[Tuple[int, int]]`, *optional*) —
  List of length (batch\_size), where each list item (`Tuple[int, int]]`) corresponds to the requested
  final size (height, width) of each prediction. If left to None, predictions will not be resized.

Returns

`List[torch.Tensor]`

A list of length `batch_size`, where each item is a semantic segmentation map of shape (height, width)
corresponding to the target\_sizes entry (if `target_sizes` is specified). Each entry of each
`torch.Tensor` correspond to a semantic class id.

Converts the output of [MaskFormerForInstanceSegmentation](/docs/transformers/v4.56.2/en/model_doc/maskformer#transformers.MaskFormerForInstanceSegmentation) into semantic segmentation maps. Only supports
PyTorch.

#### post\_process\_instance\_segmentation

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/oneformer/image_processing_oneformer_fast.py#L795)

( outputs task\_type: str = 'instance' is\_demo: bool = True threshold: float = 0.5 mask\_threshold: float = 0.5 overlap\_mask\_area\_threshold: float = 0.8 target\_sizes: typing.Optional[list[tuple[int, int]]] = None return\_coco\_annotation: typing.Optional[bool] = False  ) → `List[Dict]`

Parameters

* **outputs** (`OneFormerForUniversalSegmentationOutput`) —
  The outputs from `OneFormerForUniversalSegmentationOutput`.
* **task\_type** (`str`, *optional*, defaults to “instance”) —
  The post processing depends on the task token input. If the `task_type` is “panoptic”, we need to
  ignore the stuff predictions.
* **is\_demo** (`bool`, *optional)*, defaults to `True`) —
  Whether the model is in demo mode. If true, use threshold to predict final masks.
* **threshold** (`float`, *optional*, defaults to 0.5) —
  The probability score threshold to keep predicted instance masks.
* **mask\_threshold** (`float`, *optional*, defaults to 0.5) —
  Threshold to use when turning the predicted masks into binary values.
* **overlap\_mask\_area\_threshold** (`float`, *optional*, defaults to 0.8) —
  The overlap mask area threshold to merge or discard small disconnected parts within each binary
  instance mask.
* **target\_sizes** (`List[Tuple]`, *optional*) —
  List of length (batch\_size), where each list item (`Tuple[int, int]]`) corresponds to the requested
  final size (height, width) of each prediction in batch. If left to None, predictions will not be
  resized.
* **return\_coco\_annotation** (`bool`, *optional)*, defaults to `False`) —
  Whether to return predictions in COCO format.

Returns

`List[Dict]`

A list of dictionaries, one per image, each dictionary containing two keys:

* **segmentation** — a tensor of shape `(height, width)` where each pixel represents a `segment_id`, set
  to `None` if no mask if found above `threshold`. If `target_sizes` is specified, segmentation is resized
  to the corresponding `target_sizes` entry.
* **segments\_info** — A dictionary that contains additional information on each segment.
  + **id** — an integer representing the `segment_id`.
  + **label\_id** — An integer representing the label / semantic class id corresponding to `segment_id`.
  + **was\_fused** — a boolean, `True` if `label_id` was in `label_ids_to_fuse`, `False` otherwise.
    Multiple instances of the same class / label were fused and assigned a single `segment_id`.
  + **score** — Prediction score of segment with `segment_id`.

Converts the output of `OneFormerForUniversalSegmentationOutput` into image instance segmentation
predictions. Only supports PyTorch.

#### post\_process\_panoptic\_segmentation

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/oneformer/image_processing_oneformer_fast.py#L915)

( outputs threshold: float = 0.5 mask\_threshold: float = 0.5 overlap\_mask\_area\_threshold: float = 0.8 label\_ids\_to\_fuse: typing.Optional[set[int]] = None target\_sizes: typing.Optional[list[tuple[int, int]]] = None  ) → `list[Dict]`

Parameters

* **outputs** (`MaskFormerForInstanceSegmentationOutput`) —
  The outputs from [MaskFormerForInstanceSegmentation](/docs/transformers/v4.56.2/en/model_doc/maskformer#transformers.MaskFormerForInstanceSegmentation).
* **threshold** (`float`, *optional*, defaults to 0.5) —
  The probability score threshold to keep predicted instance masks.
* **mask\_threshold** (`float`, *optional*, defaults to 0.5) —
  Threshold to use when turning the predicted masks into binary values.
* **overlap\_mask\_area\_threshold** (`float`, *optional*, defaults to 0.8) —
  The overlap mask area threshold to merge or discard small disconnected parts within each binary
  instance mask.
* **label\_ids\_to\_fuse** (`Set[int]`, *optional*) —
  The labels in this state will have all their instances be fused together. For instance we could say
  there can only be one sky in an image, but several persons, so the label ID for sky would be in that
  set, but not the one for person.
* **target\_sizes** (`list[Tuple]`, *optional*) —
  List of length (batch\_size), where each list item (`tuple[int, int]]`) corresponds to the requested
  final size (height, width) of each prediction in batch. If left to None, predictions will not be
  resized.

Returns

`list[Dict]`

A list of dictionaries, one per image, each dictionary containing two keys:

* **segmentation** — a tensor of shape `(height, width)` where each pixel represents a `segment_id`, set
  to `None` if no mask if found above `threshold`. If `target_sizes` is specified, segmentation is resized
  to the corresponding `target_sizes` entry.
* **segments\_info** — A dictionary that contains additional information on each segment.
  + **id** — an integer representing the `segment_id`.
  + **label\_id** — An integer representing the label / semantic class id corresponding to `segment_id`.
  + **was\_fused** — a boolean, `True` if `label_id` was in `label_ids_to_fuse`, `False` otherwise.
    Multiple instances of the same class / label were fused and assigned a single `segment_id`.
  + **score** — Prediction score of segment with `segment_id`.

Converts the output of `MaskFormerForInstanceSegmentationOutput` into image panoptic segmentation
predictions. Only supports PyTorch.

## OneFormerProcessor

### class transformers.OneFormerProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/oneformer/processing_oneformer.py#L27)

( image\_processor = None tokenizer = None max\_seq\_length: int = 77 task\_seq\_length: int = 77 \*\*kwargs  )

Parameters

* **image\_processor** ([OneFormerImageProcessor](/docs/transformers/v4.56.2/en/model_doc/oneformer#transformers.OneFormerImageProcessor)) —
  The image processor is a required input.
* **tokenizer** ([`CLIPTokenizer`, `CLIPTokenizerFast`]) —
  The tokenizer is a required input.
* **max\_seq\_len** (`int`, *optional*, defaults to 77)) —
  Sequence length for input text list.
* **task\_seq\_len** (`int`, *optional*, defaults to 77) —
  Sequence length for input task token.

Constructs an OneFormer processor which wraps [OneFormerImageProcessor](/docs/transformers/v4.56.2/en/model_doc/oneformer#transformers.OneFormerImageProcessor) and
[CLIPTokenizer](/docs/transformers/v4.56.2/en/model_doc/clip#transformers.CLIPTokenizer)/[CLIPTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/clip#transformers.CLIPTokenizerFast) into a single processor that inherits both the image processor and
tokenizer functionalities.

#### encode\_inputs

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/oneformer/processing_oneformer.py#L143)

( images = None task\_inputs = None segmentation\_maps = None \*\*kwargs  )

This method forwards all its arguments to `OneFormerImageProcessor.encode_inputs()` and then tokenizes the
task\_inputs. Please refer to the docstring of this method for more information.

#### post\_process\_instance\_segmentation

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/oneformer/processing_oneformer.py#L190)

( \*args \*\*kwargs  )

This method forwards all its arguments to [OneFormerImageProcessor.post\_process\_instance\_segmentation()](/docs/transformers/v4.56.2/en/model_doc/oneformer#transformers.OneFormerImageProcessor.post_process_instance_segmentation).
Please refer to the docstring of this method for more information.

#### post\_process\_panoptic\_segmentation

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/oneformer/processing_oneformer.py#L197)

( \*args \*\*kwargs  )

This method forwards all its arguments to [OneFormerImageProcessor.post\_process\_panoptic\_segmentation()](/docs/transformers/v4.56.2/en/model_doc/oneformer#transformers.OneFormerImageProcessor.post_process_panoptic_segmentation).
Please refer to the docstring of this method for more information.

#### post\_process\_semantic\_segmentation

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/oneformer/processing_oneformer.py#L183)

( \*args \*\*kwargs  )

This method forwards all its arguments to [OneFormerImageProcessor.post\_process\_semantic\_segmentation()](/docs/transformers/v4.56.2/en/model_doc/oneformer#transformers.OneFormerImageProcessor.post_process_semantic_segmentation).
Please refer to the docstring of this method for more information.

## OneFormerModel

### class transformers.OneFormerModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/oneformer/modeling_oneformer.py#L2852)

( config: OneFormerConfig  )

Parameters

* **config** ([OneFormerConfig](/docs/transformers/v4.56.2/en/model_doc/oneformer#transformers.OneFormerConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Oneformer Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/oneformer/modeling_oneformer.py#L2869)

( pixel\_values: Tensor task\_inputs: Tensor text\_inputs: typing.Optional[torch.Tensor] = None pixel\_mask: typing.Optional[torch.Tensor] = None output\_hidden\_states: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → [transformers.models.oneformer.modeling\_oneformer.OneFormerModelOutput](/docs/transformers/v4.56.2/en/model_doc/oneformer#transformers.models.oneformer.modeling_oneformer.OneFormerModelOutput) or `tuple(torch.FloatTensor)`

Parameters

* **pixel\_values** (`torch.Tensor` of shape `(batch_size, num_channels, image_size, image_size)`) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [OneFormerImageProcessor](/docs/transformers/v4.56.2/en/model_doc/oneformer#transformers.OneFormerImageProcessor). See `OneFormerImageProcessor.__call__()` for details ([OneFormerProcessor](/docs/transformers/v4.56.2/en/model_doc/oneformer#transformers.OneFormerProcessor) uses
  [OneFormerImageProcessor](/docs/transformers/v4.56.2/en/model_doc/oneformer#transformers.OneFormerImageProcessor) for processing images).
* **task\_inputs** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`) —
  Task inputs. Task inputs can be obtained using [AutoImageProcessor](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoImageProcessor). See `OneFormerProcessor.__call__()`
  for details.
* **text\_inputs** (`list[torch.Tensor]`, *optional*) —
  Tensor fof shape `(num_queries, sequence_length)` to be fed to a model
* **pixel\_mask** (`torch.Tensor` of shape `(batch_size, height, width)`, *optional*) —
  Mask to avoid performing attention on padding pixel values. Mask values selected in `[0, 1]`:
  + 1 for pixels that are real (i.e. **not masked**),
  + 0 for pixels that are padding (i.e. **masked**).

  [What are attention masks?](../glossary#attention-mask)
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

[transformers.models.oneformer.modeling\_oneformer.OneFormerModelOutput](/docs/transformers/v4.56.2/en/model_doc/oneformer#transformers.models.oneformer.modeling_oneformer.OneFormerModelOutput) or `tuple(torch.FloatTensor)`

A [transformers.models.oneformer.modeling\_oneformer.OneFormerModelOutput](/docs/transformers/v4.56.2/en/model_doc/oneformer#transformers.models.oneformer.modeling_oneformer.OneFormerModelOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([OneFormerConfig](/docs/transformers/v4.56.2/en/model_doc/oneformer#transformers.OneFormerConfig)) and inputs.

* **encoder\_hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
  shape `(batch_size, num_channels, height, width)`. Hidden-states (also called feature maps) of the encoder
  model at the output of each stage.
* **pixel\_decoder\_hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
  shape `(batch_size, num_channels, height, width)`. Hidden-states (also called feature maps) of the pixel
  decoder model at the output of each stage.
* **transformer\_decoder\_hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
  shape `(batch_size, sequence_length, hidden_size)`. Hidden-states (also called feature maps) of the
  transformer decoder at the output of each stage.
* **transformer\_decoder\_object\_queries** (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_dim)`) — Output object queries from the last layer in the transformer decoder.
* **transformer\_decoder\_contrastive\_queries** (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_dim)`) — Contrastive queries from the transformer decoder.
* **transformer\_decoder\_mask\_predictions** (`torch.FloatTensor` of shape `(batch_size, num_queries, height, width)`) — Mask Predictions from the last layer in the transformer decoder.
* **transformer\_decoder\_class\_predictions** (`torch.FloatTensor` of shape `(batch_size, num_queries, num_classes+1)`) — Class Predictions from the last layer in the transformer decoder.
* **transformer\_decoder\_auxiliary\_predictions** (`Tuple` of Dict of `str, torch.FloatTensor`, *optional*) — Tuple of class and mask predictions from each layer of the transformer decoder.
* **text\_queries** (`torch.FloatTensor`, *optional* of shape `(batch_size, num_queries, hidden_dim)`) — Text queries derived from the input text list used for calculating contrastive loss during training.
* **task\_token** (`torch.FloatTensor` of shape `(batch_size, hidden_dim)`) — 1D task token to condition the queries.
* **attentions** (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `tuple(torch.FloatTensor)` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`. Self and Cross Attentions weights from transformer decoder.

The [OneFormerModel](/docs/transformers/v4.56.2/en/model_doc/oneformer#transformers.OneFormerModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> import torch
>>> from PIL import Image
>>> import requests
>>> from transformers import OneFormerProcessor, OneFormerModel

>>> # download texting image
>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> # load processor for preprocessing the inputs
>>> processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_ade20k_swin_tiny")
>>> model = OneFormerModel.from_pretrained("shi-labs/oneformer_ade20k_swin_tiny")
>>> inputs = processor(image, ["semantic"], return_tensors="pt")

>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> mask_predictions = outputs.transformer_decoder_mask_predictions
>>> class_predictions = outputs.transformer_decoder_class_predictions

>>> f"👉 Mask Predictions Shape: {list(mask_predictions.shape)}, Class Predictions Shape: {list(class_predictions.shape)}"
'👉 Mask Predictions Shape: [1, 150, 128, 171], Class Predictions Shape: [1, 150, 151]'
```

## OneFormerForUniversalSegmentation

### class transformers.OneFormerForUniversalSegmentation

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/oneformer/modeling_oneformer.py#L2985)

( config: OneFormerConfig  )

Parameters

* **config** ([OneFormerConfig](/docs/transformers/v4.56.2/en/model_doc/oneformer#transformers.OneFormerConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

OneFormer Model for instance, semantic and panoptic image segmentation.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/oneformer/modeling_oneformer.py#L3052)

( pixel\_values: Tensor task\_inputs: Tensor text\_inputs: typing.Optional[torch.Tensor] = None mask\_labels: typing.Optional[list[torch.Tensor]] = None class\_labels: typing.Optional[list[torch.Tensor]] = None pixel\_mask: typing.Optional[torch.Tensor] = None output\_auxiliary\_logits: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → [transformers.models.oneformer.modeling\_oneformer.OneFormerForUniversalSegmentationOutput](/docs/transformers/v4.56.2/en/model_doc/oneformer#transformers.models.oneformer.modeling_oneformer.OneFormerForUniversalSegmentationOutput) or `tuple(torch.FloatTensor)`

Parameters

* **pixel\_values** (`torch.Tensor` of shape `(batch_size, num_channels, image_size, image_size)`) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [OneFormerImageProcessor](/docs/transformers/v4.56.2/en/model_doc/oneformer#transformers.OneFormerImageProcessor). See `OneFormerImageProcessor.__call__()` for details ([OneFormerProcessor](/docs/transformers/v4.56.2/en/model_doc/oneformer#transformers.OneFormerProcessor) uses
  [OneFormerImageProcessor](/docs/transformers/v4.56.2/en/model_doc/oneformer#transformers.OneFormerImageProcessor) for processing images).
* **task\_inputs** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`) —
  Task inputs. Task inputs can be obtained using [AutoImageProcessor](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoImageProcessor). See `OneFormerProcessor.__call__()`
  for details.
* **text\_inputs** (`list[torch.Tensor]`, *optional*) —
  Tensor fof shape `(num_queries, sequence_length)` to be fed to a model
* **mask\_labels** (`list[torch.Tensor]`, *optional*) —
  List of mask labels of shape `(num_labels, height, width)` to be fed to a model
* **class\_labels** (`list[torch.LongTensor]`, *optional*) —
  list of target class labels of shape `(num_labels, height, width)` to be fed to a model. They identify the
  labels of `mask_labels`, e.g. the label of `mask_labels[i][j]` if `class_labels[i][j]`.
* **pixel\_mask** (`torch.Tensor` of shape `(batch_size, height, width)`, *optional*) —
  Mask to avoid performing attention on padding pixel values. Mask values selected in `[0, 1]`:
  + 1 for pixels that are real (i.e. **not masked**),
  + 0 for pixels that are padding (i.e. **masked**).

  [What are attention masks?](../glossary#attention-mask)
* **output\_auxiliary\_logits** (`bool`, *optional*) —
  Whether or not to output auxiliary logits.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

[transformers.models.oneformer.modeling\_oneformer.OneFormerForUniversalSegmentationOutput](/docs/transformers/v4.56.2/en/model_doc/oneformer#transformers.models.oneformer.modeling_oneformer.OneFormerForUniversalSegmentationOutput) or `tuple(torch.FloatTensor)`

A [transformers.models.oneformer.modeling\_oneformer.OneFormerForUniversalSegmentationOutput](/docs/transformers/v4.56.2/en/model_doc/oneformer#transformers.models.oneformer.modeling_oneformer.OneFormerForUniversalSegmentationOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([OneFormerConfig](/docs/transformers/v4.56.2/en/model_doc/oneformer#transformers.OneFormerConfig)) and inputs.

* **loss** (`torch.Tensor`, *optional*) — The computed loss, returned when labels are present.
* **class\_queries\_logits** (`torch.FloatTensor`, *optional*, defaults to `None`) — A tensor of shape `(batch_size, num_queries, num_labels + 1)` representing the proposed classes for each
  query. Note the `+ 1` is needed because we incorporate the null class.
* **masks\_queries\_logits** (`torch.FloatTensor`, *optional*, defaults to `None`) — A tensor of shape `(batch_size, num_queries, height, width)` representing the proposed masks for each
  query.
* **auxiliary\_predictions** (`List` of Dict of `str, torch.FloatTensor`, *optional*) — List of class and mask predictions from each layer of the transformer decoder.
* **encoder\_hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
  shape `(batch_size, num_channels, height, width)`. Hidden-states (also called feature maps) of the encoder
  model at the output of each stage.
* **pixel\_decoder\_hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
  shape `(batch_size, num_channels, height, width)`. Hidden-states (also called feature maps) of the pixel
  decoder model at the output of each stage.
* **transformer\_decoder\_hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
  shape `(batch_size, sequence_length, hidden_size)`. Hidden-states (also called feature maps) of the
  transformer decoder at the output of each stage.
* **transformer\_decoder\_object\_queries** (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_dim)`) — Output object queries from the last layer in the transformer decoder.
* **transformer\_decoder\_contrastive\_queries** (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_dim)`) — Contrastive queries from the transformer decoder.
* **transformer\_decoder\_mask\_predictions** (`torch.FloatTensor` of shape `(batch_size, num_queries, height, width)`) — Mask Predictions from the last layer in the transformer decoder.
* **transformer\_decoder\_class\_predictions** (`torch.FloatTensor` of shape `(batch_size, num_queries, num_classes+1)`) — Class Predictions from the last layer in the transformer decoder.
* **transformer\_decoder\_auxiliary\_predictions** (`List` of Dict of `str, torch.FloatTensor`, *optional*) — List of class and mask predictions from each layer of the transformer decoder.
* **text\_queries** (`torch.FloatTensor`, *optional* of shape `(batch_size, num_queries, hidden_dim)`) — Text queries derived from the input text list used for calculating contrastive loss during training.
* **task\_token** (`torch.FloatTensor` of shape `(batch_size, hidden_dim)`) — 1D task token to condition the queries.
* **attentions** (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `tuple(torch.FloatTensor)` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`. Self and Cross Attentions weights from transformer decoder.

The [OneFormerForUniversalSegmentation](/docs/transformers/v4.56.2/en/model_doc/oneformer#transformers.OneFormerForUniversalSegmentation) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:

Universal segmentation example:


```
>>> from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
>>> from PIL import Image
>>> import requests
>>> import torch

>>> # load OneFormer fine-tuned on ADE20k for universal segmentation
>>> processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_ade20k_swin_tiny")
>>> model = OneFormerForUniversalSegmentation.from_pretrained("shi-labs/oneformer_ade20k_swin_tiny")

>>> url = (
...     "https://huggingface.co/datasets/hf-internal-testing/fixtures_ade20k/resolve/main/ADE_val_00000001.jpg"
... )
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> # Semantic Segmentation
>>> inputs = processor(image, ["semantic"], return_tensors="pt")

>>> with torch.no_grad():
...     outputs = model(**inputs)
>>> # model predicts class_queries_logits of shape `(batch_size, num_queries)`
>>> # and masks_queries_logits of shape `(batch_size, num_queries, height, width)`
>>> class_queries_logits = outputs.class_queries_logits
>>> masks_queries_logits = outputs.masks_queries_logits

>>> # you can pass them to processor for semantic postprocessing
>>> predicted_semantic_map = processor.post_process_semantic_segmentation(
...     outputs, target_sizes=[(image.height, image.width)]
... )[0]
>>> f"👉 Semantic Predictions Shape: {list(predicted_semantic_map.shape)}"
'👉 Semantic Predictions Shape: [512, 683]'

>>> # Instance Segmentation
>>> inputs = processor(image, ["instance"], return_tensors="pt")

>>> with torch.no_grad():
...     outputs = model(**inputs)
>>> # model predicts class_queries_logits of shape `(batch_size, num_queries)`
>>> # and masks_queries_logits of shape `(batch_size, num_queries, height, width)`
>>> class_queries_logits = outputs.class_queries_logits
>>> masks_queries_logits = outputs.masks_queries_logits

>>> # you can pass them to processor for instance postprocessing
>>> predicted_instance_map = processor.post_process_instance_segmentation(
...     outputs, target_sizes=[(image.height, image.width)]
... )[0]["segmentation"]
>>> f"👉 Instance Predictions Shape: {list(predicted_instance_map.shape)}"
'👉 Instance Predictions Shape: [512, 683]'

>>> # Panoptic Segmentation
>>> inputs = processor(image, ["panoptic"], return_tensors="pt")

>>> with torch.no_grad():
...     outputs = model(**inputs)
>>> # model predicts class_queries_logits of shape `(batch_size, num_queries)`
>>> # and masks_queries_logits of shape `(batch_size, num_queries, height, width)`
>>> class_queries_logits = outputs.class_queries_logits
>>> masks_queries_logits = outputs.masks_queries_logits

>>> # you can pass them to processor for panoptic postprocessing
>>> predicted_panoptic_map = processor.post_process_panoptic_segmentation(
...     outputs, target_sizes=[(image.height, image.width)]
... )[0]["segmentation"]
>>> f"👉 Panoptic Predictions Shape: {list(predicted_panoptic_map.shape)}"
'👉 Panoptic Predictions Shape: [512, 683]'
```

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/oneformer.md)
