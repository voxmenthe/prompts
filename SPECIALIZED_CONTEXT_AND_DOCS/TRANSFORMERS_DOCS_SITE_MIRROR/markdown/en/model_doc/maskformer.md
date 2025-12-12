# MaskFormer

This is a recently introduced model so the API hasn't been tested extensively. There may be some bugs or slight
breaking changes to fix it in the future. If you see something strange, file a [Github Issue](https://github.com/huggingface/transformers/issues/new?assignees=&labels=&template=bug-report.md&title).

## Overview

The MaskFormer model was proposed in [Per-Pixel Classification is Not All You Need for Semantic Segmentation](https://huggingface.co/papers/2107.06278) by Bowen Cheng, Alexander G. Schwing, Alexander Kirillov. MaskFormer addresses semantic segmentation with a mask classification paradigm instead of performing classic pixel-level classification.

The abstract from the paper is the following:

*Modern approaches typically formulate semantic segmentation as a per-pixel classification task, while instance-level segmentation is handled with an alternative mask classification. Our key insight: mask classification is sufficiently general to solve both semantic- and instance-level segmentation tasks in a unified manner using the exact same model, loss, and training procedure. Following this observation, we propose MaskFormer, a simple mask classification model which predicts a set of binary masks, each associated with a single global class label prediction. Overall, the proposed mask classification-based method simplifies the landscape of effective approaches to semantic and panoptic segmentation tasks and shows excellent empirical results. In particular, we observe that MaskFormer outperforms per-pixel classification baselines when the number of classes is large. Our mask classification-based method outperforms both current state-of-the-art semantic (55.6 mIoU on ADE20K) and panoptic segmentation (52.7 PQ on COCO) models.*

The figure below illustrates the architecture of MaskFormer. Taken from the [original paper](https://huggingface.co/papers/2107.06278).

This model was contributed by [francesco](https://huggingface.co/francesco). The original code can be found [here](https://github.com/facebookresearch/MaskFormer).

## Usage tips

- MaskFormer's Transformer decoder is identical to the decoder of [DETR](detr). During training, the authors of DETR did find it helpful to use auxiliary losses in the decoder, especially to help the model output the correct number of objects of each class. If you set the parameter `use_auxiliary_loss` of [MaskFormerConfig](/docs/transformers/main/en/model_doc/maskformer#transformers.MaskFormerConfig) to `True`, then prediction feedforward neural networks and Hungarian losses are added after each decoder layer (with the FFNs sharing parameters).
- If you want to train the model in a distributed environment across multiple nodes, then one should update the
  `get_num_masks` function inside in the `MaskFormerLoss` class of `modeling_maskformer.py`. When training on multiple nodes, this should be
  set to the average number of target masks across all nodes, as can be seen in the original implementation [here](https://github.com/facebookresearch/MaskFormer/blob/da3e60d85fdeedcb31476b5edd7d328826ce56cc/mask_former/modeling/criterion.py#L169).
- One can use [MaskFormerImageProcessor](/docs/transformers/main/en/model_doc/maskformer#transformers.MaskFormerImageProcessor) to prepare images for the model and optional targets for the model.
- To get the final segmentation, depending on the task, you can call [post_process_semantic_segmentation()](/docs/transformers/main/en/model_doc/maskformer#transformers.MaskFormerImageProcessor.post_process_semantic_segmentation) or [post_process_panoptic_segmentation()](/docs/transformers/main/en/model_doc/maskformer#transformers.MaskFormerImageProcessor.post_process_panoptic_segmentation). Both tasks can be solved using [MaskFormerForInstanceSegmentation](/docs/transformers/main/en/model_doc/maskformer#transformers.MaskFormerForInstanceSegmentation) output, panoptic segmentation accepts an optional `label_ids_to_fuse` argument to fuse instances of the target object/s (e.g. sky) together.

## Resources

- All notebooks that illustrate inference as well as fine-tuning on custom data with MaskFormer can be found [here](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/MaskFormer).
- Scripts for finetuning `MaskFormer` with [Trainer](/docs/transformers/main/en/main_classes/trainer#transformers.Trainer) or [Accelerate](https://huggingface.co/docs/accelerate/index) can be found [here](https://github.com/huggingface/transformers/tree/main/examples/pytorch/instance-segmentation).

## MaskFormer specific outputs[[transformers.models.maskformer.modeling_maskformer.MaskFormerModelOutput]]

#### transformers.models.maskformer.modeling_maskformer.MaskFormerModelOutput[[transformers.models.maskformer.modeling_maskformer.MaskFormerModelOutput]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/maskformer/modeling_maskformer.py#L137)

Class for outputs of [MaskFormerModel](/docs/transformers/main/en/model_doc/maskformer#transformers.MaskFormerModel). This class returns all the needed hidden states to compute the logits.

**Parameters:**

encoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`) : Last hidden states (final feature map) of the last stage of the encoder model (backbone).

pixel_decoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`) : Last hidden states (final feature map) of the last stage of the pixel decoder model (FPN).

transformer_decoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) : Last hidden states (final feature map) of the last stage of the transformer decoder model.

encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) : Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of shape `(batch_size, num_channels, height, width)`. Hidden-states (also called feature maps) of the encoder model at the output of each stage.

pixel_decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) : Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of shape `(batch_size, num_channels, height, width)`. Hidden-states (also called feature maps) of the pixel decoder model at the output of each stage.

transformer_decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) : Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states (also called feature maps) of the transformer decoder at the output of each stage.

hidden_states `tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) : Tuple of `torch.FloatTensor` containing `encoder_hidden_states`, `pixel_decoder_hidden_states` and `decoder_hidden_states`

hidden_states (`tuple[torch.FloatTensor]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) : Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

attentions (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) : Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

#### transformers.models.maskformer.modeling_maskformer.MaskFormerForInstanceSegmentationOutput[[transformers.models.maskformer.modeling_maskformer.MaskFormerForInstanceSegmentationOutput]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/maskformer/modeling_maskformer.py#L183)

Class for outputs of [MaskFormerForInstanceSegmentation](/docs/transformers/main/en/model_doc/maskformer#transformers.MaskFormerForInstanceSegmentation).

This output can be directly passed to [post_process_semantic_segmentation()](/docs/transformers/main/en/model_doc/maskformer#transformers.MaskFormerImageProcessor.post_process_semantic_segmentation) or or
[post_process_instance_segmentation()](/docs/transformers/main/en/model_doc/maskformer#transformers.MaskFormerImageProcessor.post_process_instance_segmentation) or
[post_process_panoptic_segmentation()](/docs/transformers/main/en/model_doc/maskformer#transformers.MaskFormerImageProcessor.post_process_panoptic_segmentation) depending on the task. Please, see
[`~MaskFormerImageProcessor] for details regarding usage.

**Parameters:**

loss (`torch.Tensor`, *optional*) : The computed loss, returned when labels are present.

class_queries_logits (`torch.FloatTensor`, *optional*, defaults to `None`) : A tensor of shape `(batch_size, num_queries, num_labels + 1)` representing the proposed classes for each query. Note the `+ 1` is needed because we incorporate the null class.

masks_queries_logits (`torch.FloatTensor`, *optional*, defaults to `None`) : A tensor of shape `(batch_size, num_queries, height, width)` representing the proposed masks for each query.

auxiliary_logits (`Dict[str, torch.FloatTensor]`, *optional*, returned when `output_auxiliary_logits=True`) : Dictionary containing auxiliary predictions for each decoder layer when auxiliary losses are enabled.

encoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`) : Last hidden states (final feature map) of the last stage of the encoder model (backbone).

pixel_decoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`) : Last hidden states (final feature map) of the last stage of the pixel decoder model (FPN).

transformer_decoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) : Last hidden states (final feature map) of the last stage of the transformer decoder model.

encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) : Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of shape `(batch_size, num_channels, height, width)`. Hidden-states (also called feature maps) of the encoder model at the output of each stage.

pixel_decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) : Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of shape `(batch_size, num_channels, height, width)`. Hidden-states (also called feature maps) of the pixel decoder model at the output of each stage.

transformer_decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) : Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the transformer decoder at the output of each stage.

hidden_states `tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) : Tuple of `torch.FloatTensor` containing `encoder_hidden_states`, `pixel_decoder_hidden_states` and `decoder_hidden_states`.

hidden_states (`tuple[torch.FloatTensor]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) : Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

attentions (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) : Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

## MaskFormerConfig[[transformers.MaskFormerConfig]]

#### transformers.MaskFormerConfig[[transformers.MaskFormerConfig]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/maskformer/configuration_maskformer.py#L30)

This is the configuration class to store the configuration of a [MaskFormerModel](/docs/transformers/main/en/model_doc/maskformer#transformers.MaskFormerModel). It is used to instantiate a
MaskFormer model according to the specified arguments, defining the model architecture. Instantiating a
configuration with the defaults will yield a similar configuration to that of the MaskFormer
[facebook/maskformer-swin-base-ade](https://huggingface.co/facebook/maskformer-swin-base-ade) architecture trained
on [ADE20k-150](https://huggingface.co/datasets/scene_parse_150).

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

Currently, MaskFormer only supports the [Swin Transformer](swin) as backbone.

Examples:

```python
>>> from transformers import MaskFormerConfig, MaskFormerModel

>>> # Initializing a MaskFormer facebook/maskformer-swin-base-ade configuration
>>> configuration = MaskFormerConfig()

>>> # Initializing a model (with random weights) from the facebook/maskformer-swin-base-ade style configuration
>>> model = MaskFormerModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

**Parameters:**

mask_feature_size (`int`, *optional*, defaults to 256) : The masks' features size, this value will also be used to specify the Feature Pyramid Network features' size.

no_object_weight (`float`, *optional*, defaults to 0.1) : Weight to apply to the null (no object) class.

use_auxiliary_loss(`bool`, *optional*, defaults to `False`) : If `True` `MaskFormerForInstanceSegmentationOutput` will contain the auxiliary losses computed using the logits from each decoder's stage.

backbone_config (`Dict`, *optional*) : The configuration passed to the backbone, if unset, the configuration corresponding to `swin-base-patch4-window12-384` will be used.

backbone (`str`, *optional*) : Name of backbone to use when `backbone_config` is `None`. If `use_pretrained_backbone` is `True`, this will load the corresponding pretrained weights from the timm or transformers library. If `use_pretrained_backbone` is `False`, this loads the backbone's config and uses that to initialize the backbone with random weights.

use_pretrained_backbone (`bool`, *optional*, `False`) : Whether to use pretrained weights for the backbone.

use_timm_backbone (`bool`, *optional*, `False`) : Whether to load `backbone` from the timm library. If `False`, the backbone is loaded from the transformers library.

backbone_kwargs (`dict`, *optional*) : Keyword arguments to be passed to AutoBackbone when loading from a checkpoint e.g. `{'out_indices': (0, 1, 2, 3)}`. Cannot be specified if `backbone_config` is set.

decoder_config (`Dict`, *optional*) : The configuration passed to the transformer decoder model, if unset the base config for `detr-resnet-50` will be used.

init_std (`float`, *optional*, defaults to 0.02) : The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

init_xavier_std (`float`, *optional*, defaults to 1) : The scaling factor used for the Xavier initialization gain in the HM Attention map module.

dice_weight (`float`, *optional*, defaults to 1.0) : The weight for the dice loss.

cross_entropy_weight (`float`, *optional*, defaults to 1.0) : The weight for the cross entropy loss.

mask_weight (`float`, *optional*, defaults to 20.0) : The weight for the mask loss.

output_auxiliary_logits (`bool`, *optional*) : Should the model output its `auxiliary_logits` or not.

## MaskFormerImageProcessor[[transformers.MaskFormerImageProcessor]]

#### transformers.MaskFormerImageProcessor[[transformers.MaskFormerImageProcessor]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/maskformer/image_processing_maskformer.py#L371)

Constructs a MaskFormer image processor. The image processor can be used to prepare image(s) and optional targets
for the model.

This image processor inherits from [BaseImageProcessor](/docs/transformers/main/en/main_classes/image_processor#transformers.BaseImageProcessor) which contains most of the main methods. Users should
refer to this superclass for more information regarding those methods.

preprocesstransformers.MaskFormerImageProcessor.preprocesshttps://github.com/huggingface/transformers/blob/main/src/transformers/models/maskformer/image_processing_maskformer.py#L683[{"name": "images", "val": ": typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']]"}, {"name": "segmentation_maps", "val": ": typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor'], NoneType] = None"}, {"name": "instance_id_to_semantic_id", "val": ": typing.Optional[dict[int, int]] = None"}, {"name": "do_resize", "val": ": typing.Optional[bool] = None"}, {"name": "size", "val": ": typing.Optional[dict[str, int]] = None"}, {"name": "size_divisor", "val": ": typing.Optional[int] = None"}, {"name": "resample", "val": ": typing.Optional[PIL.Image.Resampling] = None"}, {"name": "do_rescale", "val": ": typing.Optional[bool] = None"}, {"name": "rescale_factor", "val": ": typing.Optional[float] = None"}, {"name": "do_normalize", "val": ": typing.Optional[bool] = None"}, {"name": "image_mean", "val": ": typing.Union[float, list[float], NoneType] = None"}, {"name": "image_std", "val": ": typing.Union[float, list[float], NoneType] = None"}, {"name": "ignore_index", "val": ": typing.Optional[int] = None"}, {"name": "do_reduce_labels", "val": ": typing.Optional[bool] = None"}, {"name": "return_tensors", "val": ": typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None"}, {"name": "data_format", "val": ": typing.Union[str, transformers.image_utils.ChannelDimension] = "}, {"name": "input_data_format", "val": ": typing.Union[str, transformers.image_utils.ChannelDimension, NoneType] = None"}, {"name": "pad_size", "val": ": typing.Optional[dict[str, int]] = None"}]

**Parameters:**

do_resize (`bool`, *optional*, defaults to `True`) : Whether to resize the input to a certain `size`.

size (`int`, *optional*, defaults to 800) : Resize the input to the given size. Only has an effect if `do_resize` is set to `True`. If size is a sequence like `(width, height)`, output size will be matched to this. If size is an int, smaller edge of the image will be matched to this number. i.e, if `height > width`, then image will be rescaled to `(size * height / width, size)`.

size_divisor (`int`, *optional*, defaults to 32) : Some backbones need images divisible by a certain number. If not passed, it defaults to the value used in Swin Transformer.

resample (`int`, *optional*, defaults to `Resampling.BILINEAR`) : An optional resampling filter. This can be one of `PIL.Image.Resampling.NEAREST`, `PIL.Image.Resampling.BOX`, `PIL.Image.Resampling.BILINEAR`, `PIL.Image.Resampling.HAMMING`, `PIL.Image.Resampling.BICUBIC` or `PIL.Image.Resampling.LANCZOS`. Only has an effect if `do_resize` is set to `True`.

do_rescale (`bool`, *optional*, defaults to `True`) : Whether to rescale the input to a certain `scale`.

rescale_factor (`float`, *optional*, defaults to `1/ 255`) : Rescale the input by the given factor. Only has an effect if `do_rescale` is set to `True`.

do_normalize (`bool`, *optional*, defaults to `True`) : Whether or not to normalize the input with mean and standard deviation.

image_mean (`int`, *optional*, defaults to `[0.485, 0.456, 0.406]`) : The sequence of means for each channel, to be used when normalizing images. Defaults to the ImageNet mean.

image_std (`int`, *optional*, defaults to `[0.229, 0.224, 0.225]`) : The sequence of standard deviations for each channel, to be used when normalizing images. Defaults to the ImageNet std.

ignore_index (`int`, *optional*) : Label to be assigned to background pixels in segmentation maps. If provided, segmentation map pixels denoted with 0 (background) will be replaced with `ignore_index`.

do_reduce_labels (`bool`, *optional*, defaults to `False`) : Whether or not to decrement all label values of segmentation maps by 1. Usually used for datasets where 0 is used for background, and background itself is not included in all classes of a dataset (e.g. ADE20k). The background label will be replaced by `ignore_index`.

num_labels (`int`, *optional*) : The number of labels in the segmentation map.

pad_size (`Dict[str, int]`, *optional*) : The size `{"height": int, "width" int}` to pad the images to. Must be larger than any image size provided for preprocessing. If `pad_size` is not provided, images will be padded to the largest height and width in the batch.
#### encode_inputs[[transformers.MaskFormerImageProcessor.encode_inputs]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/maskformer/image_processing_maskformer.py#L872)

Pad images up to the largest image in a batch and create a corresponding `pixel_mask`.

MaskFormer addresses semantic segmentation with a mask classification paradigm, thus input segmentation maps
will be converted to lists of binary masks and their respective labels. Let's see an example, assuming
`segmentation_maps = [[2,6,7,9]]`, the output will contain `mask_labels =
[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]` (four binary masks) and `class_labels = [2,6,7,9]`, the labels for
each mask.

**Parameters:**

pixel_values_list (`list[ImageInput]`) : List of images (pixel values) to be padded. Each image should be a tensor of shape `(channels, height, width)`. 

segmentation_maps (`ImageInput`, *optional*) : The corresponding semantic segmentation maps with the pixel-wise annotations.  (`bool`, *optional*, defaults to `True`): Whether or not to pad images up to the largest image in a batch and create a pixel mask.  If left to the default, will return a pixel mask that is:  - 1 for pixels that are real (i.e. **not masked**), - 0 for pixels that are padding (i.e. **masked**). 

instance_id_to_semantic_id (`list[dict[int, int]]` or `dict[int, int]`, *optional*) : A mapping between object instance ids and class ids. If passed, `segmentation_maps` is treated as an instance segmentation map where each pixel represents an instance id. Can be provided as a single dictionary with a global/dataset-level mapping or as a list of dictionaries (one per image), to map instance ids in each image separately. 

return_tensors (`str` or [TensorType](/docs/transformers/main/en/internal/file_utils#transformers.TensorType), *optional*) : If set, will return tensors instead of NumPy arrays. If set to `'pt'`, return PyTorch `torch.Tensor` objects. 

pad_size (`Dict[str, int]`, *optional*) : The size `{"height": int, "width" int}` to pad the images to. Must be larger than any image size provided for preprocessing. If `pad_size` is not provided, images will be padded to the largest height and width in the batch.

**Returns:**

`[BatchFeature](/docs/transformers/main/en/main_classes/image_processor#transformers.BatchFeature)`

A [BatchFeature](/docs/transformers/main/en/main_classes/image_processor#transformers.BatchFeature) with the following fields:

- **pixel_values** -- Pixel values to be fed to a model.
- **pixel_mask** -- Pixel mask to be fed to a model (when `=True` or if `pixel_mask` is in
  `self.model_input_names`).
- **mask_labels** -- Optional list of mask labels of shape `(labels, height, width)` to be fed to a model
  (when `annotations` are provided).
- **class_labels** -- Optional list of class labels of shape `(labels)` to be fed to a model (when
  `annotations` are provided). They identify the labels of `mask_labels`, e.g. the label of
  `mask_labels[i][j]` if `class_labels[i][j]`.
#### post_process_semantic_segmentation[[transformers.MaskFormerImageProcessor.post_process_semantic_segmentation]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/maskformer/image_processing_maskformer.py#L987)

Converts the output of [MaskFormerForInstanceSegmentation](/docs/transformers/main/en/model_doc/maskformer#transformers.MaskFormerForInstanceSegmentation) into semantic segmentation maps. Only supports
PyTorch.

**Parameters:**

outputs ([MaskFormerForInstanceSegmentation](/docs/transformers/main/en/model_doc/maskformer#transformers.MaskFormerForInstanceSegmentation)) : Raw outputs of the model.

target_sizes (`list[tuple[int, int]]`, *optional*) : List of length (batch_size), where each list item (`tuple[int, int]]`) corresponds to the requested final size (height, width) of each prediction. If left to None, predictions will not be resized.

**Returns:**

``list[torch.Tensor]``

A list of length `batch_size`, where each item is a semantic segmentation map of shape (height, width)
corresponding to the target_sizes entry (if `target_sizes` is specified). Each entry of each
`torch.Tensor` correspond to a semantic class id.
#### post_process_instance_segmentation[[transformers.MaskFormerImageProcessor.post_process_instance_segmentation]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/maskformer/image_processing_maskformer.py#L1037)

Converts the output of `MaskFormerForInstanceSegmentationOutput` into instance segmentation predictions. Only
supports PyTorch. If instances could overlap, set either return_coco_annotation or return_binary_maps
to `True` to get the correct segmentation result.

**Parameters:**

outputs ([MaskFormerForInstanceSegmentation](/docs/transformers/main/en/model_doc/maskformer#transformers.MaskFormerForInstanceSegmentation)) : Raw outputs of the model.

threshold (`float`, *optional*, defaults to 0.5) : The probability score threshold to keep predicted instance masks.

mask_threshold (`float`, *optional*, defaults to 0.5) : Threshold to use when turning the predicted masks into binary values.

overlap_mask_area_threshold (`float`, *optional*, defaults to 0.8) : The overlap mask area threshold to merge or discard small disconnected parts within each binary instance mask.

target_sizes (`list[Tuple]`, *optional*) : List of length (batch_size), where each list item (`tuple[int, int]]`) corresponds to the requested final size (height, width) of each prediction. If left to None, predictions will not be resized.

return_coco_annotation (`bool`, *optional*, defaults to `False`) : If set to `True`, segmentation maps are returned in COCO run-length encoding (RLE) format.

return_binary_maps (`bool`, *optional*, defaults to `False`) : If set to `True`, segmentation maps are returned as a concatenated tensor of binary segmentation maps (one per detected instance).

**Returns:**

``list[Dict]``

A list of dictionaries, one per image, each dictionary containing two keys:
- **segmentation** -- A tensor of shape `(height, width)` where each pixel represents a `segment_id`, or
  `list[List]` run-length encoding (RLE) of the segmentation map if return_coco_annotation is set to
  `True`, or a tensor of shape `(num_instances, height, width)` if return_binary_maps is set to `True`.
  Set to `None` if no mask if found above `threshold`.
- **segments_info** -- A dictionary that contains additional information on each segment.
  - **id** -- An integer representing the `segment_id`.
  - **label_id** -- An integer representing the label / semantic class id corresponding to `segment_id`.
  - **score** -- Prediction score of segment with `segment_id`.
#### post_process_panoptic_segmentation[[transformers.MaskFormerImageProcessor.post_process_panoptic_segmentation]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/maskformer/image_processing_maskformer.py#L1153)

Converts the output of `MaskFormerForInstanceSegmentationOutput` into image panoptic segmentation
predictions. Only supports PyTorch.

**Parameters:**

outputs (`MaskFormerForInstanceSegmentationOutput`) : The outputs from [MaskFormerForInstanceSegmentation](/docs/transformers/main/en/model_doc/maskformer#transformers.MaskFormerForInstanceSegmentation).

threshold (`float`, *optional*, defaults to 0.5) : The probability score threshold to keep predicted instance masks.

mask_threshold (`float`, *optional*, defaults to 0.5) : Threshold to use when turning the predicted masks into binary values.

overlap_mask_area_threshold (`float`, *optional*, defaults to 0.8) : The overlap mask area threshold to merge or discard small disconnected parts within each binary instance mask.

label_ids_to_fuse (`Set[int]`, *optional*) : The labels in this state will have all their instances be fused together. For instance we could say there can only be one sky in an image, but several persons, so the label ID for sky would be in that set, but not the one for person.

target_sizes (`list[Tuple]`, *optional*) : List of length (batch_size), where each list item (`tuple[int, int]]`) corresponds to the requested final size (height, width) of each prediction in batch. If left to None, predictions will not be resized.

**Returns:**

``list[Dict]``

A list of dictionaries, one per image, each dictionary containing two keys:
- **segmentation** -- a tensor of shape `(height, width)` where each pixel represents a `segment_id`, set
  to `None` if no mask if found above `threshold`. If `target_sizes` is specified, segmentation is resized
  to the corresponding `target_sizes` entry.
- **segments_info** -- A dictionary that contains additional information on each segment.
  - **id** -- an integer representing the `segment_id`.
  - **label_id** -- An integer representing the label / semantic class id corresponding to `segment_id`.
  - **was_fused** -- a boolean, `True` if `label_id` was in `label_ids_to_fuse`, `False` otherwise.
    Multiple instances of the same class / label were fused and assigned a single `segment_id`.
  - **score** -- Prediction score of segment with `segment_id`.

## MaskFormerImageProcessorFast[[transformers.MaskFormerImageProcessorFast]]

#### transformers.MaskFormerImageProcessorFast[[transformers.MaskFormerImageProcessorFast]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/maskformer/image_processing_maskformer_fast.py#L99)

Constructs a fast Maskformer image processor.

preprocesstransformers.MaskFormerImageProcessorFast.preprocesshttps://github.com/huggingface/transformers/blob/main/src/transformers/models/maskformer/image_processing_maskformer_fast.py#L236[{"name": "images", "val": ": typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']]"}, {"name": "segmentation_maps", "val": ": typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor'], NoneType] = None"}, {"name": "instance_id_to_semantic_id", "val": ": typing.Union[list[dict[int, int]], dict[int, int], NoneType] = None"}, {"name": "**kwargs", "val": ": typing_extensions.Unpack[transformers.models.maskformer.image_processing_maskformer.MaskFormerImageProcessorKwargs]"}]- **images** (`Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']]`) --
  Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
  passing in images with pixel values between 0 and 1, set `do_rescale=False`.
- **segmentation_maps** (`ImageInput`, *optional*) --
  The segmentation maps.
- **instance_id_to_semantic_id** (`Union[list[dict[int, int]], dict[int, int]]`, *optional*) --
  A mapping from instance IDs to semantic IDs.
- **do_convert_rgb** (`bool`, *optional*) --
  Whether to convert the image to RGB.
- **do_resize** (`bool`, *optional*) --
  Whether to resize the image.
- **size** (`Annotated[Union[int, list[int], tuple[int, ...], dict[str, int], NoneType], None]`) --
  Describes the maximum input dimensions to the model.
- **crop_size** (`Annotated[Union[int, list[int], tuple[int, ...], dict[str, int], NoneType], None]`) --
  Size of the output image after applying `center_crop`.
- **resample** (`Annotated[Union[PILImageResampling, int, NoneType], None]`) --
  Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`. Only
  has an effect if `do_resize` is set to `True`.
- **do_rescale** (`bool`, *optional*) --
  Whether to rescale the image.
- **rescale_factor** (`float`, *optional*) --
  Rescale factor to rescale the image by if `do_rescale` is set to `True`.
- **do_normalize** (`bool`, *optional*) --
  Whether to normalize the image.
- **image_mean** (`Union[float, list[float], tuple[float, ...], NoneType]`) --
  Image mean to use for normalization. Only has an effect if `do_normalize` is set to `True`.
- **image_std** (`Union[float, list[float], tuple[float, ...], NoneType]`) --
  Image standard deviation to use for normalization. Only has an effect if `do_normalize` is set to
  `True`.
- **do_pad** (`bool`, *optional*) --
  Whether to pad the image. Padding is done either to the largest size in the batch
  or to a fixed square size per image. The exact padding strategy depends on the model.
- **pad_size** (`Annotated[Union[int, list[int], tuple[int, ...], dict[str, int], NoneType], None]`) --
  The size in `{"height": int, "width" int}` to pad the images to. Must be larger than any image size
  provided for preprocessing. If `pad_size` is not provided, images will be padded to the largest
  height and width in the batch. Applied only when `do_pad=True.`
- **do_center_crop** (`bool`, *optional*) --
  Whether to center crop the image.
- **data_format** (`Union[~image_utils.ChannelDimension, str, NoneType]`) --
  Only `ChannelDimension.FIRST` is supported. Added for compatibility with slow processors.
- **input_data_format** (`Union[~image_utils.ChannelDimension, str, NoneType]`) --
  The channel dimension format for the input image. If unset, the channel dimension format is inferred
  from the input image. Can be one of:
  - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
  - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
  - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
- **device** (`Annotated[Union[str, torch.device, NoneType], None]`) --
  The device to process the images on. If unset, the device is inferred from the input images.
- **return_tensors** (`Annotated[Union[str, ~utils.generic.TensorType, NoneType], None]`) --
  Returns stacked tensors if set to `pt, otherwise returns a list of tensors.
- **disable_grouping** (`bool`, *optional*) --
  Whether to disable grouping of images by size to process them individually and not in batches.
  If None, will be set to True if the images are on CPU, and False otherwise. This choice is based on
  empirical observations, as detailed here: https://github.com/huggingface/transformers/pull/38157
- **image_seq_length** (`int`, *optional*) --
  The number of image tokens to be used for each image in the input.
  Added for backward compatibility but this should be set as a processor attribute in future models.
- **size_divisor** (`.size_divisor`) --
  The size by which to make sure both the height and width can be divided.
- **ignore_index** (`int`, *optional*) --
  Label to be assigned to background pixels in segmentation maps. If provided, segmentation map pixels
  denoted with 0 (background) will be replaced with `ignore_index`.
- **do_reduce_labels** (`bool`, *optional*, defaults to `False`) --
  Whether or not to decrement all label values of segmentation maps by 1. Usually used for datasets where 0
  is used for background, and background itself is not included in all classes of a dataset (e.g. ADE20k).
  The background label will be replaced by `ignore_index`.
- **num_labels** (`int`, *optional*) --
  The number of labels in the segmentation map.0``- **data** (`dict`) -- Dictionary of lists/arrays/tensors returned by the __call__ method ('pixel_values', etc.).
- **tensor_type** (`Union[None, str, TensorType]`, *optional*) -- You can give a tensor_type here to convert the lists of integers in PyTorch/Numpy Tensors at
  initialization.

**Parameters:**

images (`Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']]`) : Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If passing in images with pixel values between 0 and 1, set `do_rescale=False`.

segmentation_maps (`ImageInput`, *optional*) : The segmentation maps.

instance_id_to_semantic_id (`Union[list[dict[int, int]], dict[int, int]]`, *optional*) : A mapping from instance IDs to semantic IDs.

do_convert_rgb (`bool`, *optional*) : Whether to convert the image to RGB.

do_resize (`bool`, *optional*) : Whether to resize the image.

size (`Annotated[Union[int, list[int], tuple[int, ...], dict[str, int], NoneType], None]`) : Describes the maximum input dimensions to the model.

crop_size (`Annotated[Union[int, list[int], tuple[int, ...], dict[str, int], NoneType], None]`) : Size of the output image after applying `center_crop`.

resample (`Annotated[Union[PILImageResampling, int, NoneType], None]`) : Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`. Only has an effect if `do_resize` is set to `True`.

do_rescale (`bool`, *optional*) : Whether to rescale the image.

rescale_factor (`float`, *optional*) : Rescale factor to rescale the image by if `do_rescale` is set to `True`.

do_normalize (`bool`, *optional*) : Whether to normalize the image.

image_mean (`Union[float, list[float], tuple[float, ...], NoneType]`) : Image mean to use for normalization. Only has an effect if `do_normalize` is set to `True`.

image_std (`Union[float, list[float], tuple[float, ...], NoneType]`) : Image standard deviation to use for normalization. Only has an effect if `do_normalize` is set to `True`.

do_pad (`bool`, *optional*) : Whether to pad the image. Padding is done either to the largest size in the batch or to a fixed square size per image. The exact padding strategy depends on the model.

pad_size (`Annotated[Union[int, list[int], tuple[int, ...], dict[str, int], NoneType], None]`) : The size in `{"height": int, "width" int}` to pad the images to. Must be larger than any image size provided for preprocessing. If `pad_size` is not provided, images will be padded to the largest height and width in the batch. Applied only when `do_pad=True.`

do_center_crop (`bool`, *optional*) : Whether to center crop the image.

data_format (`Union[~image_utils.ChannelDimension, str, NoneType]`) : Only `ChannelDimension.FIRST` is supported. Added for compatibility with slow processors.

input_data_format (`Union[~image_utils.ChannelDimension, str, NoneType]`) : The channel dimension format for the input image. If unset, the channel dimension format is inferred from the input image. Can be one of: - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format. - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format. - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.

device (`Annotated[Union[str, torch.device, NoneType], None]`) : The device to process the images on. If unset, the device is inferred from the input images.

return_tensors (`Annotated[Union[str, ~utils.generic.TensorType, NoneType], None]`) : Returns stacked tensors if set to `pt, otherwise returns a list of tensors.

disable_grouping (`bool`, *optional*) : Whether to disable grouping of images by size to process them individually and not in batches. If None, will be set to True if the images are on CPU, and False otherwise. This choice is based on empirical observations, as detailed here: https://github.com/huggingface/transformers/pull/38157

image_seq_length (`int`, *optional*) : The number of image tokens to be used for each image in the input. Added for backward compatibility but this should be set as a processor attribute in future models.

size_divisor (`.size_divisor`) : The size by which to make sure both the height and width can be divided.

ignore_index (`int`, *optional*) : Label to be assigned to background pixels in segmentation maps. If provided, segmentation map pixels denoted with 0 (background) will be replaced with `ignore_index`.

do_reduce_labels (`bool`, *optional*, defaults to `False`) : Whether or not to decrement all label values of segmentation maps by 1. Usually used for datasets where 0 is used for background, and background itself is not included in all classes of a dataset (e.g. ADE20k). The background label will be replaced by `ignore_index`.

num_labels (`int`, *optional*) : The number of labels in the segmentation map.

**Returns:**

````

- **data** (`dict`) -- Dictionary of lists/arrays/tensors returned by the __call__ method ('pixel_values', etc.).
- **tensor_type** (`Union[None, str, TensorType]`, *optional*) -- You can give a tensor_type here to convert the lists of integers in PyTorch/Numpy Tensors at
  initialization.
#### post_process_semantic_segmentation[[transformers.MaskFormerImageProcessorFast.post_process_semantic_segmentation]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/maskformer/image_processing_maskformer_fast.py#L409)

Converts the output of [MaskFormerForInstanceSegmentation](/docs/transformers/main/en/model_doc/maskformer#transformers.MaskFormerForInstanceSegmentation) into semantic segmentation maps. Only supports
PyTorch.

**Parameters:**

outputs ([MaskFormerForInstanceSegmentation](/docs/transformers/main/en/model_doc/maskformer#transformers.MaskFormerForInstanceSegmentation)) : Raw outputs of the model.

target_sizes (`list[tuple[int, int]]`, *optional*) : List of length (batch_size), where each list item (`tuple[int, int]]`) corresponds to the requested final size (height, width) of each prediction. If left to None, predictions will not be resized.

**Returns:**

``list[torch.Tensor]``

A list of length `batch_size`, where each item is a semantic segmentation map of shape (height, width)
corresponding to the target_sizes entry (if `target_sizes` is specified). Each entry of each
`torch.Tensor` correspond to a semantic class id.
#### post_process_instance_segmentation[[transformers.MaskFormerImageProcessorFast.post_process_instance_segmentation]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/maskformer/image_processing_maskformer_fast.py#L460)

Converts the output of `MaskFormerForInstanceSegmentationOutput` into instance segmentation predictions. Only
supports PyTorch. If instances could overlap, set either return_coco_annotation or return_binary_maps
to `True` to get the correct segmentation result.

**Parameters:**

outputs ([MaskFormerForInstanceSegmentation](/docs/transformers/main/en/model_doc/maskformer#transformers.MaskFormerForInstanceSegmentation)) : Raw outputs of the model.

threshold (`float`, *optional*, defaults to 0.5) : The probability score threshold to keep predicted instance masks.

mask_threshold (`float`, *optional*, defaults to 0.5) : Threshold to use when turning the predicted masks into binary values.

overlap_mask_area_threshold (`float`, *optional*, defaults to 0.8) : The overlap mask area threshold to merge or discard small disconnected parts within each binary instance mask.

target_sizes (`list[Tuple]`, *optional*) : List of length (batch_size), where each list item (`tuple[int, int]]`) corresponds to the requested final size (height, width) of each prediction. If left to None, predictions will not be resized.

return_coco_annotation (`bool`, *optional*, defaults to `False`) : If set to `True`, segmentation maps are returned in COCO run-length encoding (RLE) format.

return_binary_maps (`bool`, *optional*, defaults to `False`) : If set to `True`, segmentation maps are returned as a concatenated tensor of binary segmentation maps (one per detected instance).

**Returns:**

``list[Dict]``

A list of dictionaries, one per image, each dictionary containing two keys:
- **segmentation** -- A tensor of shape `(height, width)` where each pixel represents a `segment_id`, or
  `list[List]` run-length encoding (RLE) of the segmentation map if return_coco_annotation is set to
  `True`, or a tensor of shape `(num_instances, height, width)` if return_binary_maps is set to `True`.
  Set to `None` if no mask if found above `threshold`.
- **segments_info** -- A dictionary that contains additional information on each segment.
  - **id** -- An integer representing the `segment_id`.
  - **label_id** -- An integer representing the label / semantic class id corresponding to `segment_id`.
  - **score** -- Prediction score of segment with `segment_id`.
#### post_process_panoptic_segmentation[[transformers.MaskFormerImageProcessorFast.post_process_panoptic_segmentation]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/maskformer/image_processing_maskformer_fast.py#L577)

Converts the output of `MaskFormerForInstanceSegmentationOutput` into image panoptic segmentation
predictions. Only supports PyTorch.

**Parameters:**

outputs (`MaskFormerForInstanceSegmentationOutput`) : The outputs from [MaskFormerForInstanceSegmentation](/docs/transformers/main/en/model_doc/maskformer#transformers.MaskFormerForInstanceSegmentation).

threshold (`float`, *optional*, defaults to 0.5) : The probability score threshold to keep predicted instance masks.

mask_threshold (`float`, *optional*, defaults to 0.5) : Threshold to use when turning the predicted masks into binary values.

overlap_mask_area_threshold (`float`, *optional*, defaults to 0.8) : The overlap mask area threshold to merge or discard small disconnected parts within each binary instance mask.

label_ids_to_fuse (`Set[int]`, *optional*) : The labels in this state will have all their instances be fused together. For instance we could say there can only be one sky in an image, but several persons, so the label ID for sky would be in that set, but not the one for person.

target_sizes (`list[Tuple]`, *optional*) : List of length (batch_size), where each list item (`tuple[int, int]]`) corresponds to the requested final size (height, width) of each prediction in batch. If left to None, predictions will not be resized.

**Returns:**

``list[Dict]``

A list of dictionaries, one per image, each dictionary containing two keys:
- **segmentation** -- a tensor of shape `(height, width)` where each pixel represents a `segment_id`, set
  to `None` if no mask if found above `threshold`. If `target_sizes` is specified, segmentation is resized
  to the corresponding `target_sizes` entry.
- **segments_info** -- A dictionary that contains additional information on each segment.
  - **id** -- an integer representing the `segment_id`.
  - **label_id** -- An integer representing the label / semantic class id corresponding to `segment_id`.
  - **was_fused** -- a boolean, `True` if `label_id` was in `label_ids_to_fuse`, `False` otherwise.
    Multiple instances of the same class / label were fused and assigned a single `segment_id`.
  - **score** -- Prediction score of segment with `segment_id`.

## MaskFormerModel[[transformers.MaskFormerModel]]

#### transformers.MaskFormerModel[[transformers.MaskFormerModel]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/maskformer/modeling_maskformer.py#L1481)

The bare Maskformer Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

forwardtransformers.MaskFormerModel.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/maskformer/modeling_maskformer.py#L1491[{"name": "pixel_values", "val": ": Tensor"}, {"name": "pixel_mask", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "output_hidden_states", "val": ": typing.Optional[bool] = None"}, {"name": "output_attentions", "val": ": typing.Optional[bool] = None"}, {"name": "return_dict", "val": ": typing.Optional[bool] = None"}, {"name": "**kwargs", "val": ""}]- **pixel_values** (`torch.Tensor` of shape `(batch_size, num_channels, image_size, image_size)`) --
  The tensors corresponding to the input images. Pixel values can be obtained using
  [MaskFormerImageProcessor](/docs/transformers/main/en/model_doc/maskformer#transformers.MaskFormerImageProcessor). See `MaskFormerImageProcessor.__call__()` for details (`processor_class` uses
  [MaskFormerImageProcessor](/docs/transformers/main/en/model_doc/maskformer#transformers.MaskFormerImageProcessor) for processing images).
- **pixel_mask** (`torch.Tensor` of shape `(batch_size, height, width)`, *optional*) --
  Mask to avoid performing attention on padding pixel values. Mask values selected in `[0, 1]`:

  - 1 for pixels that are real (i.e. **not masked**),
  - 0 for pixels that are padding (i.e. **masked**).

  [What are attention masks?](../glossary#attention-mask)
- **output_hidden_states** (`bool`, *optional*) --
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
- **output_attentions** (`bool`, *optional*) --
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
- **return_dict** (`bool`, *optional*) --
  Whether or not to return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.0[transformers.models.maskformer.modeling_maskformer.MaskFormerModelOutput](/docs/transformers/main/en/model_doc/maskformer#transformers.models.maskformer.modeling_maskformer.MaskFormerModelOutput) or `tuple(torch.FloatTensor)`A [transformers.models.maskformer.modeling_maskformer.MaskFormerModelOutput](/docs/transformers/main/en/model_doc/maskformer#transformers.models.maskformer.modeling_maskformer.MaskFormerModelOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([MaskFormerConfig](/docs/transformers/main/en/model_doc/maskformer#transformers.MaskFormerConfig)) and inputs.

- **encoder_last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`) -- Last hidden states (final feature map) of the last stage of the encoder model (backbone).
- **pixel_decoder_last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`) -- Last hidden states (final feature map) of the last stage of the pixel decoder model (FPN).
- **transformer_decoder_last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) -- Last hidden states (final feature map) of the last stage of the transformer decoder model.
- **encoder_hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
  shape `(batch_size, num_channels, height, width)`. Hidden-states (also called feature maps) of the encoder
  model at the output of each stage.
- **pixel_decoder_hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
  shape `(batch_size, num_channels, height, width)`. Hidden-states (also called feature maps) of the pixel
  decoder model at the output of each stage.
- **transformer_decoder_hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
  shape `(batch_size, sequence_length, hidden_size)`. Hidden-states (also called feature maps) of the
  transformer decoder at the output of each stage.
- **hidden_states** `tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` containing `encoder_hidden_states`, `pixel_decoder_hidden_states` and
  `decoder_hidden_states`
- **hidden_states** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
The [MaskFormerModel](/docs/transformers/main/en/model_doc/maskformer#transformers.MaskFormerModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:

```python
>>> from transformers import AutoImageProcessor, MaskFormerModel
>>> from PIL import Image
>>> import requests

>>> # load MaskFormer fine-tuned on ADE20k semantic segmentation
>>> image_processor = AutoImageProcessor.from_pretrained("facebook/maskformer-swin-base-ade")
>>> model = MaskFormerModel.from_pretrained("facebook/maskformer-swin-base-ade")

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> inputs = image_processor(image, return_tensors="pt")

>>> # forward pass
>>> outputs = model(**inputs)

>>> # the decoder of MaskFormer outputs hidden states of shape (batch_size, num_queries, hidden_size)
>>> transformer_decoder_last_hidden_state = outputs.transformer_decoder_last_hidden_state
>>> list(transformer_decoder_last_hidden_state.shape)
[1, 100, 256]
```

**Parameters:**

config ([MaskFormerConfig](/docs/transformers/main/en/model_doc/maskformer#transformers.MaskFormerConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

**Returns:**

`[transformers.models.maskformer.modeling_maskformer.MaskFormerModelOutput](/docs/transformers/main/en/model_doc/maskformer#transformers.models.maskformer.modeling_maskformer.MaskFormerModelOutput) or `tuple(torch.FloatTensor)``

A [transformers.models.maskformer.modeling_maskformer.MaskFormerModelOutput](/docs/transformers/main/en/model_doc/maskformer#transformers.models.maskformer.modeling_maskformer.MaskFormerModelOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([MaskFormerConfig](/docs/transformers/main/en/model_doc/maskformer#transformers.MaskFormerConfig)) and inputs.

- **encoder_last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`) -- Last hidden states (final feature map) of the last stage of the encoder model (backbone).
- **pixel_decoder_last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`) -- Last hidden states (final feature map) of the last stage of the pixel decoder model (FPN).
- **transformer_decoder_last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) -- Last hidden states (final feature map) of the last stage of the transformer decoder model.
- **encoder_hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
  shape `(batch_size, num_channels, height, width)`. Hidden-states (also called feature maps) of the encoder
  model at the output of each stage.
- **pixel_decoder_hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
  shape `(batch_size, num_channels, height, width)`. Hidden-states (also called feature maps) of the pixel
  decoder model at the output of each stage.
- **transformer_decoder_hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
  shape `(batch_size, sequence_length, hidden_size)`. Hidden-states (also called feature maps) of the
  transformer decoder at the output of each stage.
- **hidden_states** `tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` containing `encoder_hidden_states`, `pixel_decoder_hidden_states` and
  `decoder_hidden_states`
- **hidden_states** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

## MaskFormerForInstanceSegmentation[[transformers.MaskFormerForInstanceSegmentation]]

#### transformers.MaskFormerForInstanceSegmentation[[transformers.MaskFormerForInstanceSegmentation]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/maskformer/modeling_maskformer.py#L1578)

forwardtransformers.MaskFormerForInstanceSegmentation.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/maskformer/modeling_maskformer.py#L1660[{"name": "pixel_values", "val": ": Tensor"}, {"name": "mask_labels", "val": ": typing.Optional[list[torch.Tensor]] = None"}, {"name": "class_labels", "val": ": typing.Optional[list[torch.Tensor]] = None"}, {"name": "pixel_mask", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "output_auxiliary_logits", "val": ": typing.Optional[bool] = None"}, {"name": "output_hidden_states", "val": ": typing.Optional[bool] = None"}, {"name": "output_attentions", "val": ": typing.Optional[bool] = None"}, {"name": "return_dict", "val": ": typing.Optional[bool] = None"}, {"name": "**kwargs", "val": ""}]- **pixel_values** (`torch.Tensor` of shape `(batch_size, num_channels, image_size, image_size)`) --
  The tensors corresponding to the input images. Pixel values can be obtained using
  [MaskFormerImageProcessor](/docs/transformers/main/en/model_doc/maskformer#transformers.MaskFormerImageProcessor). See `MaskFormerImageProcessor.__call__()` for details (`processor_class` uses
  [MaskFormerImageProcessor](/docs/transformers/main/en/model_doc/maskformer#transformers.MaskFormerImageProcessor) for processing images).
- **mask_labels** (`list[torch.Tensor]`, *optional*) --
  List of mask labels of shape `(num_labels, height, width)` to be fed to a model
- **class_labels** (`list[torch.LongTensor]`, *optional*) --
  list of target class labels of shape `(num_labels, height, width)` to be fed to a model. They identify the
  labels of `mask_labels`, e.g. the label of `mask_labels[i][j]` if `class_labels[i][j]`.
- **pixel_mask** (`torch.Tensor` of shape `(batch_size, height, width)`, *optional*) --
  Mask to avoid performing attention on padding pixel values. Mask values selected in `[0, 1]`:

  - 1 for pixels that are real (i.e. **not masked**),
  - 0 for pixels that are padding (i.e. **masked**).

  [What are attention masks?](../glossary#attention-mask)
- **output_auxiliary_logits** (`bool`, *optional*) --
  Whether or not to output auxiliary logits.
- **output_hidden_states** (`bool`, *optional*) --
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
- **output_attentions** (`bool`, *optional*) --
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
- **return_dict** (`bool`, *optional*) --
  Whether or not to return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.0[transformers.models.maskformer.modeling_maskformer.MaskFormerForInstanceSegmentationOutput](/docs/transformers/main/en/model_doc/maskformer#transformers.models.maskformer.modeling_maskformer.MaskFormerForInstanceSegmentationOutput) or `tuple(torch.FloatTensor)`A [transformers.models.maskformer.modeling_maskformer.MaskFormerForInstanceSegmentationOutput](/docs/transformers/main/en/model_doc/maskformer#transformers.models.maskformer.modeling_maskformer.MaskFormerForInstanceSegmentationOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([MaskFormerConfig](/docs/transformers/main/en/model_doc/maskformer#transformers.MaskFormerConfig)) and inputs.

- **loss** (`torch.Tensor`, *optional*) -- The computed loss, returned when labels are present.
- **class_queries_logits** (`torch.FloatTensor`, *optional*, defaults to `None`) -- A tensor of shape `(batch_size, num_queries, num_labels + 1)` representing the proposed classes for each
  query. Note the `+ 1` is needed because we incorporate the null class.
- **masks_queries_logits** (`torch.FloatTensor`, *optional*, defaults to `None`) -- A tensor of shape `(batch_size, num_queries, height, width)` representing the proposed masks for each
  query.
- **auxiliary_logits** (`Dict[str, torch.FloatTensor]`, *optional*, returned when `output_auxiliary_logits=True`) -- Dictionary containing auxiliary predictions for each decoder layer when auxiliary losses are enabled.
- **encoder_last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`) -- Last hidden states (final feature map) of the last stage of the encoder model (backbone).
- **pixel_decoder_last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`) -- Last hidden states (final feature map) of the last stage of the pixel decoder model (FPN).
- **transformer_decoder_last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) -- Last hidden states (final feature map) of the last stage of the transformer decoder model.
- **encoder_hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
  shape `(batch_size, num_channels, height, width)`. Hidden-states (also called feature maps) of the encoder
  model at the output of each stage.
- **pixel_decoder_hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
  shape `(batch_size, num_channels, height, width)`. Hidden-states (also called feature maps) of the pixel
  decoder model at the output of each stage.
- **transformer_decoder_hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
  shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the transformer decoder at the output
  of each stage.
- **hidden_states** `tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` containing `encoder_hidden_states`, `pixel_decoder_hidden_states` and
  `decoder_hidden_states`.
- **hidden_states** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
The [MaskFormerForInstanceSegmentation](/docs/transformers/main/en/model_doc/maskformer#transformers.MaskFormerForInstanceSegmentation) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:

Semantic segmentation example:

```python
>>> from transformers import AutoImageProcessor, MaskFormerForInstanceSegmentation
>>> from PIL import Image
>>> import requests

>>> # load MaskFormer fine-tuned on ADE20k semantic segmentation
>>> image_processor = AutoImageProcessor.from_pretrained("facebook/maskformer-swin-base-ade")
>>> model = MaskFormerForInstanceSegmentation.from_pretrained("facebook/maskformer-swin-base-ade")

>>> url = (
...     "https://huggingface.co/datasets/hf-internal-testing/fixtures_ade20k/resolve/main/ADE_val_00000001.jpg"
... )
>>> image = Image.open(requests.get(url, stream=True).raw)
>>> inputs = image_processor(images=image, return_tensors="pt")

>>> outputs = model(**inputs)
>>> # model predicts class_queries_logits of shape `(batch_size, num_queries)`
>>> # and masks_queries_logits of shape `(batch_size, num_queries, height, width)`
>>> class_queries_logits = outputs.class_queries_logits
>>> masks_queries_logits = outputs.masks_queries_logits

>>> # you can pass them to image_processor for postprocessing
>>> predicted_semantic_map = image_processor.post_process_semantic_segmentation(
...     outputs, target_sizes=[(image.height, image.width)]
... )[0]

>>> # we refer to the demo notebooks for visualization (see "Resources" section in the MaskFormer docs)
>>> list(predicted_semantic_map.shape)
[512, 683]
```

Panoptic segmentation example:

```python
>>> from transformers import AutoImageProcessor, MaskFormerForInstanceSegmentation
>>> from PIL import Image
>>> import requests

>>> # load MaskFormer fine-tuned on COCO panoptic segmentation
>>> image_processor = AutoImageProcessor.from_pretrained("facebook/maskformer-swin-base-coco")
>>> model = MaskFormerForInstanceSegmentation.from_pretrained("facebook/maskformer-swin-base-coco")

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)
>>> inputs = image_processor(images=image, return_tensors="pt")

>>> outputs = model(**inputs)
>>> # model predicts class_queries_logits of shape `(batch_size, num_queries)`
>>> # and masks_queries_logits of shape `(batch_size, num_queries, height, width)`
>>> class_queries_logits = outputs.class_queries_logits
>>> masks_queries_logits = outputs.masks_queries_logits

>>> # you can pass them to image_processor for postprocessing
>>> result = image_processor.post_process_panoptic_segmentation(outputs, target_sizes=[(image.height, image.width)])[0]

>>> # we refer to the demo notebooks for visualization (see "Resources" section in the MaskFormer docs)
>>> predicted_panoptic_map = result["segmentation"]
>>> list(predicted_panoptic_map.shape)
[480, 640]
```

**Parameters:**

pixel_values (`torch.Tensor` of shape `(batch_size, num_channels, image_size, image_size)`) : The tensors corresponding to the input images. Pixel values can be obtained using [MaskFormerImageProcessor](/docs/transformers/main/en/model_doc/maskformer#transformers.MaskFormerImageProcessor). See `MaskFormerImageProcessor.__call__()` for details (`processor_class` uses [MaskFormerImageProcessor](/docs/transformers/main/en/model_doc/maskformer#transformers.MaskFormerImageProcessor) for processing images).

mask_labels (`list[torch.Tensor]`, *optional*) : List of mask labels of shape `(num_labels, height, width)` to be fed to a model

class_labels (`list[torch.LongTensor]`, *optional*) : list of target class labels of shape `(num_labels, height, width)` to be fed to a model. They identify the labels of `mask_labels`, e.g. the label of `mask_labels[i][j]` if `class_labels[i][j]`.

pixel_mask (`torch.Tensor` of shape `(batch_size, height, width)`, *optional*) : Mask to avoid performing attention on padding pixel values. Mask values selected in `[0, 1]`:  - 1 for pixels that are real (i.e. **not masked**), - 0 for pixels that are padding (i.e. **masked**).  [What are attention masks?](../glossary#attention-mask)

output_auxiliary_logits (`bool`, *optional*) : Whether or not to output auxiliary logits.

output_hidden_states (`bool`, *optional*) : Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.

output_attentions (`bool`, *optional*) : Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.

return_dict (`bool`, *optional*) : Whether or not to return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

**Returns:**

`[transformers.models.maskformer.modeling_maskformer.MaskFormerForInstanceSegmentationOutput](/docs/transformers/main/en/model_doc/maskformer#transformers.models.maskformer.modeling_maskformer.MaskFormerForInstanceSegmentationOutput) or `tuple(torch.FloatTensor)``

A [transformers.models.maskformer.modeling_maskformer.MaskFormerForInstanceSegmentationOutput](/docs/transformers/main/en/model_doc/maskformer#transformers.models.maskformer.modeling_maskformer.MaskFormerForInstanceSegmentationOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([MaskFormerConfig](/docs/transformers/main/en/model_doc/maskformer#transformers.MaskFormerConfig)) and inputs.

- **loss** (`torch.Tensor`, *optional*) -- The computed loss, returned when labels are present.
- **class_queries_logits** (`torch.FloatTensor`, *optional*, defaults to `None`) -- A tensor of shape `(batch_size, num_queries, num_labels + 1)` representing the proposed classes for each
  query. Note the `+ 1` is needed because we incorporate the null class.
- **masks_queries_logits** (`torch.FloatTensor`, *optional*, defaults to `None`) -- A tensor of shape `(batch_size, num_queries, height, width)` representing the proposed masks for each
  query.
- **auxiliary_logits** (`Dict[str, torch.FloatTensor]`, *optional*, returned when `output_auxiliary_logits=True`) -- Dictionary containing auxiliary predictions for each decoder layer when auxiliary losses are enabled.
- **encoder_last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`) -- Last hidden states (final feature map) of the last stage of the encoder model (backbone).
- **pixel_decoder_last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`) -- Last hidden states (final feature map) of the last stage of the pixel decoder model (FPN).
- **transformer_decoder_last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) -- Last hidden states (final feature map) of the last stage of the transformer decoder model.
- **encoder_hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
  shape `(batch_size, num_channels, height, width)`. Hidden-states (also called feature maps) of the encoder
  model at the output of each stage.
- **pixel_decoder_hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
  shape `(batch_size, num_channels, height, width)`. Hidden-states (also called feature maps) of the pixel
  decoder model at the output of each stage.
- **transformer_decoder_hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
  shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the transformer decoder at the output
  of each stage.
- **hidden_states** `tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` containing `encoder_hidden_states`, `pixel_decoder_hidden_states` and
  `decoder_hidden_states`.
- **hidden_states** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
