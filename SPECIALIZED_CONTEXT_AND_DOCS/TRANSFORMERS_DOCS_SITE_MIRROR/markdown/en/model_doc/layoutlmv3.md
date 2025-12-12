# LayoutLMv3

## Overview

The LayoutLMv3 model was proposed in [LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking](https://huggingface.co/papers/2204.08387) by Yupan Huang, Tengchao Lv, Lei Cui, Yutong Lu, Furu Wei.
LayoutLMv3 simplifies [LayoutLMv2](layoutlmv2) by using patch embeddings (as in [ViT](vit)) instead of leveraging a CNN backbone, and pre-trains the model on 3 objectives: masked language modeling (MLM), masked image modeling (MIM)
and word-patch alignment (WPA).

The abstract from the paper is the following:

*Self-supervised pre-training techniques have achieved remarkable progress in Document AI. Most multimodal pre-trained models use a masked language modeling objective to learn bidirectional representations on the text modality, but they differ in pre-training objectives for the image modality. This discrepancy adds difficulty to multimodal representation learning. In this paper, we propose LayoutLMv3 to pre-train multimodal Transformers for Document AI with unified text and image masking. Additionally, LayoutLMv3 is pre-trained with a word-patch alignment objective to learn cross-modal alignment by predicting whether the corresponding image patch of a text word is masked. The simple unified architecture and training objectives make LayoutLMv3 a general-purpose pre-trained model for both text-centric and image-centric Document AI tasks. Experimental results show that LayoutLMv3 achieves state-of-the-art performance not only in text-centric tasks, including form understanding, receipt understanding, and document visual question answering, but also in image-centric tasks such as document image classification and document layout analysis.*

 LayoutLMv3 architecture. Taken from the original paper. 

This model was contributed by [nielsr](https://huggingface.co/nielsr). The original code can be found [here](https://github.com/microsoft/unilm/tree/master/layoutlmv3).

## Usage tips

- In terms of data processing, LayoutLMv3 is identical to its predecessor [LayoutLMv2](layoutlmv2), except that:
  - images need to be resized and normalized with channels in regular RGB format. LayoutLMv2 on the other hand normalizes the images internally and expects the channels in BGR format.
  - text is tokenized using byte-pair encoding (BPE), as opposed to WordPiece.
  Due to these differences in data preprocessing, one can use [LayoutLMv3Processor](/docs/transformers/main/en/model_doc/layoutlmv3#transformers.LayoutLMv3Processor) which internally combines a [LayoutLMv3ImageProcessor](/docs/transformers/main/en/model_doc/layoutlmv3#transformers.LayoutLMv3ImageProcessor) (for the image modality) and a [LayoutLMv3Tokenizer](/docs/transformers/main/en/model_doc/layoutlmv3#transformers.LayoutLMv3Tokenizer)/[LayoutLMv3TokenizerFast](/docs/transformers/main/en/model_doc/layoutlmv3#transformers.LayoutLMv3Tokenizer) (for the text modality) to prepare all data for the model.
- Regarding usage of [LayoutLMv3Processor](/docs/transformers/main/en/model_doc/layoutlmv3#transformers.LayoutLMv3Processor), we refer to the [usage guide](layoutlmv2#usage-layoutlmv2processor) of its predecessor.

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with LayoutLMv3. If you're interested in submitting a resource to be included here, please feel free to open a Pull Request and we'll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

LayoutLMv3 is nearly identical to LayoutLMv2, so we've also included LayoutLMv2 resources you can adapt for LayoutLMv3 tasks. For these notebooks, take care to use [LayoutLMv2Processor](/docs/transformers/main/en/model_doc/layoutlmv2#transformers.LayoutLMv2Processor) instead when preparing data for the model!

- Demo notebooks for LayoutLMv3 can be found [here](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/LayoutLMv3).
- Demo scripts can be found [here](https://github.com/huggingface/transformers-research-projects/tree/main/layoutlmv3).

- [LayoutLMv2ForSequenceClassification](/docs/transformers/main/en/model_doc/layoutlmv2#transformers.LayoutLMv2ForSequenceClassification) is supported by this [notebook](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/LayoutLMv2/RVL-CDIP/Fine_tuning_LayoutLMv2ForSequenceClassification_on_RVL_CDIP.ipynb).
- [Text classification task guide](../tasks/sequence_classification)

- [LayoutLMv3ForTokenClassification](/docs/transformers/main/en/model_doc/layoutlmv3#transformers.LayoutLMv3ForTokenClassification) is supported by this [example script](https://github.com/huggingface/transformers-research-projects/tree/main/layoutlmv3) and [notebook](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/LayoutLMv3/Fine_tune_LayoutLMv3_on_FUNSD_(HuggingFace_Trainer).ipynb).
- A [notebook](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/LayoutLMv2/FUNSD/Inference_with_LayoutLMv2ForTokenClassification.ipynb) for how to perform inference with [LayoutLMv2ForTokenClassification](/docs/transformers/main/en/model_doc/layoutlmv2#transformers.LayoutLMv2ForTokenClassification) and a [notebook](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/LayoutLMv2/FUNSD/True_inference_with_LayoutLMv2ForTokenClassification_%2B_Gradio_demo.ipynb) for how to perform inference when no labels are available with [LayoutLMv2ForTokenClassification](/docs/transformers/main/en/model_doc/layoutlmv2#transformers.LayoutLMv2ForTokenClassification).
- A [notebook](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/LayoutLMv2/FUNSD/Fine_tuning_LayoutLMv2ForTokenClassification_on_FUNSD_using_HuggingFace_Trainer.ipynb) for how to finetune [LayoutLMv2ForTokenClassification](/docs/transformers/main/en/model_doc/layoutlmv2#transformers.LayoutLMv2ForTokenClassification) with the 🤗 Trainer.
- [Token classification task guide](../tasks/token_classification)

- [LayoutLMv2ForQuestionAnswering](/docs/transformers/main/en/model_doc/layoutlmv2#transformers.LayoutLMv2ForQuestionAnswering) is supported by this [notebook](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/LayoutLMv2/DocVQA/Fine_tuning_LayoutLMv2ForQuestionAnswering_on_DocVQA.ipynb).
- [Question answering task guide](../tasks/question_answering)

**Document question answering**

- [Document question answering task guide](../tasks/document_question_answering)

## LayoutLMv3Config[[transformers.LayoutLMv3Config]]

#### transformers.LayoutLMv3Config[[transformers.LayoutLMv3Config]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/layoutlmv3/configuration_layoutlmv3.py#L24)

This is the configuration class to store the configuration of a [LayoutLMv3Model](/docs/transformers/main/en/model_doc/layoutlmv3#transformers.LayoutLMv3Model). It is used to instantiate an
LayoutLMv3 model according to the specified arguments, defining the model architecture. Instantiating a
configuration with the defaults will yield a similar configuration to that of the LayoutLMv3
[microsoft/layoutlmv3-base](https://huggingface.co/microsoft/layoutlmv3-base) architecture.

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

Example:

```python
>>> from transformers import LayoutLMv3Config, LayoutLMv3Model

>>> # Initializing a LayoutLMv3 microsoft/layoutlmv3-base style configuration
>>> configuration = LayoutLMv3Config()

>>> # Initializing a model (with random weights) from the microsoft/layoutlmv3-base style configuration
>>> model = LayoutLMv3Model(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

**Parameters:**

vocab_size (`int`, *optional*, defaults to 50265) : Vocabulary size of the LayoutLMv3 model. Defines the number of different tokens that can be represented by the `inputs_ids` passed when calling [LayoutLMv3Model](/docs/transformers/main/en/model_doc/layoutlmv3#transformers.LayoutLMv3Model).

hidden_size (`int`, *optional*, defaults to 768) : Dimension of the encoder layers and the pooler layer.

num_hidden_layers (`int`, *optional*, defaults to 12) : Number of hidden layers in the Transformer encoder.

num_attention_heads (`int`, *optional*, defaults to 12) : Number of attention heads for each attention layer in the Transformer encoder.

intermediate_size (`int`, *optional*, defaults to 3072) : Dimension of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.

hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`) : The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`, `"relu"`, `"selu"` and `"gelu_new"` are supported.

hidden_dropout_prob (`float`, *optional*, defaults to 0.1) : The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.

attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1) : The dropout ratio for the attention probabilities.

max_position_embeddings (`int`, *optional*, defaults to 512) : The maximum sequence length that this model might ever be used with. Typically set this to something large just in case (e.g., 512 or 1024 or 2048).

type_vocab_size (`int`, *optional*, defaults to 2) : The vocabulary size of the `token_type_ids` passed when calling [LayoutLMv3Model](/docs/transformers/main/en/model_doc/layoutlmv3#transformers.LayoutLMv3Model).

initializer_range (`float`, *optional*, defaults to 0.02) : The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

layer_norm_eps (`float`, *optional*, defaults to 1e-5) : The epsilon used by the layer normalization layers.

max_2d_position_embeddings (`int`, *optional*, defaults to 1024) : The maximum value that the 2D position embedding might ever be used with. Typically set this to something large just in case (e.g., 1024).

coordinate_size (`int`, *optional*, defaults to `128`) : Dimension of the coordinate embeddings.

shape_size (`int`, *optional*, defaults to `128`) : Dimension of the width and height embeddings.

has_relative_attention_bias (`bool`, *optional*, defaults to `True`) : Whether or not to use a relative attention bias in the self-attention mechanism.

rel_pos_bins (`int`, *optional*, defaults to 32) : The number of relative position bins to be used in the self-attention mechanism.

max_rel_pos (`int`, *optional*, defaults to 128) : The maximum number of relative positions to be used in the self-attention mechanism.

max_rel_2d_pos (`int`, *optional*, defaults to 256) : The maximum number of relative 2D positions in the self-attention mechanism.

rel_2d_pos_bins (`int`, *optional*, defaults to 64) : The number of 2D relative position bins in the self-attention mechanism.

has_spatial_attention_bias (`bool`, *optional*, defaults to `True`) : Whether or not to use a spatial attention bias in the self-attention mechanism.

visual_embed (`bool`, *optional*, defaults to `True`) : Whether or not to add patch embeddings.

input_size (`int`, *optional*, defaults to `224`) : The size (resolution) of the images.

num_channels (`int`, *optional*, defaults to `3`) : The number of channels of the images.

patch_size (`int`, *optional*, defaults to `16`) : The size (resolution) of the patches.

classifier_dropout (`float`, *optional*) : The dropout ratio for the classification head.

## LayoutLMv3ImageProcessor[[transformers.LayoutLMv3ImageProcessor]]

#### transformers.LayoutLMv3ImageProcessor[[transformers.LayoutLMv3ImageProcessor]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/layoutlmv3/image_processing_layoutlmv3.py#L126)

Constructs a LayoutLMv3 image processor.

preprocesstransformers.LayoutLMv3ImageProcessor.preprocesshttps://github.com/huggingface/transformers/blob/main/src/transformers/models/layoutlmv3/image_processing_layoutlmv3.py#L250[{"name": "images", "val": ": typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']]"}, {"name": "do_resize", "val": ": typing.Optional[bool] = None"}, {"name": "size", "val": ": typing.Optional[dict[str, int]] = None"}, {"name": "resample", "val": " = None"}, {"name": "do_rescale", "val": ": typing.Optional[bool] = None"}, {"name": "rescale_factor", "val": ": typing.Optional[float] = None"}, {"name": "do_normalize", "val": ": typing.Optional[bool] = None"}, {"name": "image_mean", "val": ": typing.Union[float, collections.abc.Iterable[float], NoneType] = None"}, {"name": "image_std", "val": ": typing.Union[float, collections.abc.Iterable[float], NoneType] = None"}, {"name": "apply_ocr", "val": ": typing.Optional[bool] = None"}, {"name": "ocr_lang", "val": ": typing.Optional[str] = None"}, {"name": "tesseract_config", "val": ": typing.Optional[str] = None"}, {"name": "return_tensors", "val": ": typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None"}, {"name": "data_format", "val": ": ChannelDimension = "}, {"name": "input_data_format", "val": ": typing.Union[str, transformers.image_utils.ChannelDimension, NoneType] = None"}]- **images** (`ImageInput`) --
  Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
  passing in images with pixel values between 0 and 1, set `do_rescale=False`.
- **do_resize** (`bool`, *optional*, defaults to `self.do_resize`) --
  Whether to resize the image.
- **size** (`dict[str, int]`, *optional*, defaults to `self.size`) --
  Desired size of the output image after applying `resize`.
- **resample** (`int`, *optional*, defaults to `self.resample`) --
  Resampling filter to use if resizing the image. This can be one of the `PILImageResampling` filters.
  Only has an effect if `do_resize` is set to `True`.
- **do_rescale** (`bool`, *optional*, defaults to `self.do_rescale`) --
  Whether to rescale the image pixel values between [0, 1].
- **rescale_factor** (`float`, *optional*, defaults to `self.rescale_factor`) --
  Rescale factor to apply to the image pixel values. Only has an effect if `do_rescale` is set to `True`.
- **do_normalize** (`bool`, *optional*, defaults to `self.do_normalize`) --
  Whether to normalize the image.
- **image_mean** (`float` or `Iterable[float]`, *optional*, defaults to `self.image_mean`) --
  Mean values to be used for normalization. Only has an effect if `do_normalize` is set to `True`.
- **image_std** (`float` or `Iterable[float]`, *optional*, defaults to `self.image_std`) --
  Standard deviation values to be used for normalization. Only has an effect if `do_normalize` is set to
  `True`.
- **apply_ocr** (`bool`, *optional*, defaults to `self.apply_ocr`) --
  Whether to apply the Tesseract OCR engine to get words + normalized bounding boxes.
- **ocr_lang** (`str`, *optional*, defaults to `self.ocr_lang`) --
  The language, specified by its ISO code, to be used by the Tesseract OCR engine. By default, English is
  used.
- **tesseract_config** (`str`, *optional*, defaults to `self.tesseract_config`) --
  Any additional custom configuration flags that are forwarded to the `config` parameter when calling
  Tesseract.
- **return_tensors** (`str` or `TensorType`, *optional*) --
  The type of tensors to return. Can be one of:
  - Unset: Return a list of `np.ndarray`.
  - `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
  - `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
- **data_format** (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`) --
  The channel dimension format for the output image. Can be one of:
  - `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
  - `ChannelDimension.LAST`: image in (height, width, num_channels) format.
- **input_data_format** (`ChannelDimension` or `str`, *optional*) --
  The channel dimension format for the input image. If unset, the channel dimension format is inferred
  from the input image. Can be one of:
  - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
  - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
  - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.0

Preprocess an image or batch of images.

**Parameters:**

do_resize (`bool`, *optional*, defaults to `True`) : Whether to resize the image's (height, width) dimensions to `(size["height"], size["width"])`. Can be overridden by `do_resize` in `preprocess`.

size (`dict[str, int]` *optional*, defaults to `{"height" : 224, "width": 224}`): Size of the image after resizing. Can be overridden by `size` in `preprocess`.

resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BILINEAR`) : Resampling filter to use if resizing the image. Can be overridden by `resample` in `preprocess`.

do_rescale (`bool`, *optional*, defaults to `True`) : Whether to rescale the image's pixel values by the specified `rescale_value`. Can be overridden by `do_rescale` in `preprocess`.

rescale_factor (`float`, *optional*, defaults to 1 / 255) : Value by which the image's pixel values are rescaled. Can be overridden by `rescale_factor` in `preprocess`.

do_normalize (`bool`, *optional*, defaults to `True`) : Whether to normalize the image. Can be overridden by the `do_normalize` parameter in the `preprocess` method.

image_mean (`Iterable[float]` or `float`, *optional*, defaults to `IMAGENET_STANDARD_MEAN`) : Mean to use if normalizing the image. This is a float or list of floats the length of the number of channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method.

image_std (`Iterable[float]` or `float`, *optional*, defaults to `IMAGENET_STANDARD_STD`) : Standard deviation to use if normalizing the image. This is a float or list of floats the length of the number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.

apply_ocr (`bool`, *optional*, defaults to `True`) : Whether to apply the Tesseract OCR engine to get words + normalized bounding boxes. Can be overridden by the `apply_ocr` parameter in the `preprocess` method.

ocr_lang (`str`, *optional*) : The language, specified by its ISO code, to be used by the Tesseract OCR engine. By default, English is used. Can be overridden by the `ocr_lang` parameter in the `preprocess` method.

tesseract_config (`str`, *optional*) : Any additional custom configuration flags that are forwarded to the `config` parameter when calling Tesseract. For example: '--psm 6'. Can be overridden by the `tesseract_config` parameter in the `preprocess` method.

## LayoutLMv3ImageProcessorFast[[transformers.LayoutLMv3ImageProcessorFast]]

#### transformers.LayoutLMv3ImageProcessorFast[[transformers.LayoutLMv3ImageProcessorFast]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/layoutlmv3/image_processing_layoutlmv3_fast.py#L39)

Constructs a fast Layoutlmv3 image processor.

preprocesstransformers.LayoutLMv3ImageProcessorFast.preprocesshttps://github.com/huggingface/transformers/blob/main/src/transformers/models/layoutlmv3/image_processing_layoutlmv3_fast.py#L55[{"name": "images", "val": ": typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']]"}, {"name": "**kwargs", "val": ": typing_extensions.Unpack[transformers.models.layoutlmv3.image_processing_layoutlmv3.LayoutLMv3ImageProcessorKwargs]"}]- **images** (`Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']]`) --
  Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
  passing in images with pixel values between 0 and 1, set `do_rescale=False`.
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
- **apply_ocr** (`bool`, *optional*, defaults to `True`) --
  Whether to apply the Tesseract OCR engine to get words + normalized bounding boxes. Can be overridden by
  the `apply_ocr` parameter in the `preprocess` method.
- **ocr_lang** (`str`, *optional*) --
  The language, specified by its ISO code, to be used by the Tesseract OCR engine. By default, English is
  used. Can be overridden by the `ocr_lang` parameter in the `preprocess` method.
- **tesseract_config** (`str`, *optional*) --
  Any additional custom configuration flags that are forwarded to the `config` parameter when calling
  Tesseract. For example: '--psm 6'. Can be overridden by the `tesseract_config` parameter in the
  `preprocess` method.0``- **data** (`dict`) -- Dictionary of lists/arrays/tensors returned by the __call__ method ('pixel_values', etc.).
- **tensor_type** (`Union[None, str, TensorType]`, *optional*) -- You can give a tensor_type here to convert the lists of integers in PyTorch/Numpy Tensors at
  initialization.

**Parameters:**

images (`Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']]`) : Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If passing in images with pixel values between 0 and 1, set `do_rescale=False`.

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

apply_ocr (`bool`, *optional*, defaults to `True`) : Whether to apply the Tesseract OCR engine to get words + normalized bounding boxes. Can be overridden by the `apply_ocr` parameter in the `preprocess` method.

ocr_lang (`str`, *optional*) : The language, specified by its ISO code, to be used by the Tesseract OCR engine. By default, English is used. Can be overridden by the `ocr_lang` parameter in the `preprocess` method.

tesseract_config (`str`, *optional*) : Any additional custom configuration flags that are forwarded to the `config` parameter when calling Tesseract. For example: '--psm 6'. Can be overridden by the `tesseract_config` parameter in the `preprocess` method.

**Returns:**

````

- **data** (`dict`) -- Dictionary of lists/arrays/tensors returned by the __call__ method ('pixel_values', etc.).
- **tensor_type** (`Union[None, str, TensorType]`, *optional*) -- You can give a tensor_type here to convert the lists of integers in PyTorch/Numpy Tensors at
  initialization.

## LayoutLMv3Tokenizer[[transformers.LayoutLMv3Tokenizer]]

#### transformers.LayoutLMv3Tokenizer[[transformers.LayoutLMv3Tokenizer]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/layoutlmv3/tokenization_layoutlmv3.py#L117)

Construct a LayoutLMv3 tokenizer (backed by HuggingFace's *tokenizers* library). Based on byte-level BPE.

This tokenizer inherits from [TokenizersBackend](/docs/transformers/main/en/main_classes/tokenizer#transformers.TokenizersBackend) which contains most of the main methods. Users should
refer to this superclass for more information regarding those methods.

__call__transformers.LayoutLMv3Tokenizer.__call__https://github.com/huggingface/transformers/blob/main/src/transformers/models/layoutlmv3/tokenization_layoutlmv3.py#L243[{"name": "text", "val": ": typing.Union[str, list[str], list[list[str]]]"}, {"name": "text_pair", "val": ": typing.Union[list[str], list[list[str]], NoneType] = None"}, {"name": "boxes", "val": ": typing.Union[list[list[int]], list[list[list[int]]], NoneType] = None"}, {"name": "word_labels", "val": ": typing.Union[list[int], list[list[int]], NoneType] = None"}, {"name": "add_special_tokens", "val": ": bool = True"}, {"name": "padding", "val": ": typing.Union[bool, str, transformers.utils.generic.PaddingStrategy] = False"}, {"name": "truncation", "val": ": typing.Union[bool, str, transformers.tokenization_utils_base.TruncationStrategy] = None"}, {"name": "max_length", "val": ": typing.Optional[int] = None"}, {"name": "stride", "val": ": int = 0"}, {"name": "pad_to_multiple_of", "val": ": typing.Optional[int] = None"}, {"name": "padding_side", "val": ": typing.Optional[str] = None"}, {"name": "return_tensors", "val": ": typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None"}, {"name": "return_token_type_ids", "val": ": typing.Optional[bool] = None"}, {"name": "return_attention_mask", "val": ": typing.Optional[bool] = None"}, {"name": "return_overflowing_tokens", "val": ": bool = False"}, {"name": "return_special_tokens_mask", "val": ": bool = False"}, {"name": "return_offsets_mapping", "val": ": bool = False"}, {"name": "return_length", "val": ": bool = False"}, {"name": "verbose", "val": ": bool = True"}, {"name": "**kwargs", "val": ""}]- **text** (`str`, `List[str]`, `List[List[str]]`) --
  The sequence or batch of sequences to be encoded. Each sequence can be a string, a list of strings
  (words of a single example or questions of a batch of examples) or a list of list of strings (batch of
  words).
- **text_pair** (`List[str]`, `List[List[str]]`) --
  The sequence or batch of sequences to be encoded. Each sequence should be a list of strings
  (pretokenized string).
- **boxes** (`List[List[int]]`, `List[List[List[int]]]`) --
  Word-level bounding boxes. Each bounding box should be normalized to be on a 0-1000 scale.
- **word_labels** (`List[int]`, `List[List[int]]`, *optional*) --
  Word-level integer labels (for token classification tasks such as FUNSD, CORD).

- **add_special_tokens** (`bool`, *optional*, defaults to `True`) --
  Whether or not to encode the sequences with the special tokens relative to their model.
- **padding** (`bool`, `str` or [PaddingStrategy](/docs/transformers/main/en/internal/file_utils#transformers.utils.PaddingStrategy), *optional*, defaults to `False`) --
  Activates and controls padding. Accepts the following values:

  - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
    sequence if provided).
  - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
    acceptable input length for the model if that argument is not provided.
  - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
    lengths).
- **truncation** (`bool`, `str` or [TruncationStrategy](/docs/transformers/main/en/internal/tokenization_utils#transformers.tokenization_utils_base.TruncationStrategy), *optional*, defaults to `False`) --
  Activates and controls truncation. Accepts the following values:

  - `True` or `'longest_first'`: Truncate to a maximum length specified with the argument `max_length` or
    to the maximum acceptable input length for the model if that argument is not provided. This will
    truncate token by token, removing a token from the longest sequence in the pair if a pair of
    sequences (or a batch of pairs) is provided.
  - `'only_first'`: Truncate to a maximum length specified with the argument `max_length` or to the
    maximum acceptable input length for the model if that argument is not provided. This will only
    truncate the first sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
  - `'only_second'`: Truncate to a maximum length specified with the argument `max_length` or to the
    maximum acceptable input length for the model if that argument is not provided. This will only
    truncate the second sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
  - `False` or `'do_not_truncate'` (default): No truncation (i.e., can output batch with sequence lengths
    greater than the model maximum admissible input size).
- **max_length** (`int`, *optional*) --
  Controls the maximum length to use by one of the truncation/padding parameters.

  If left unset or set to `None`, this will use the predefined model maximum length if a maximum length
  is required by one of the truncation/padding parameters. If the model has no specific maximum input
  length (like XLNet) truncation/padding to a maximum length will be deactivated.
- **stride** (`int`, *optional*, defaults to 0) --
  If set to a number along with `max_length`, the overflowing tokens returned when
  `return_overflowing_tokens=True` will contain some tokens from the end of the truncated sequence
  returned to provide some overlap between truncated and overflowing sequences. The value of this
  argument defines the number of overlapping tokens.
- **pad_to_multiple_of** (`int`, *optional*) --
  If set will pad the sequence to a multiple of the provided value. This is especially useful to enable
  the use of Tensor Cores on NVIDIA hardware with compute capability `>= 7.5` (Volta).

- **return_token_type_ids** (`bool`, *optional*) --
  Whether to return token type IDs. If left to the default, will return the token type IDs according to
  the specific tokenizer's default, defined by the `return_outputs` attribute.

  [What are token type IDs?](../glossary#token-type-ids)
- **return_attention_mask** (`bool`, *optional*) --
  Whether to return the attention mask. If left to the default, will return the attention mask according
  to the specific tokenizer's default, defined by the `return_outputs` attribute.

  [What are attention masks?](../glossary#attention-mask)
- **return_overflowing_tokens** (`bool`, *optional*, defaults to `False`) --
  Whether or not to return overflowing token sequences. If a pair of sequences of input ids (or a batch
  of pairs) is provided with `truncation_strategy = longest_first` or `True`, an error is raised instead
  of returning overflowing tokens.
- **return_special_tokens_mask** (`bool`, *optional*, defaults to `False`) --
  Whether or not to return special tokens mask information.
- **return_offsets_mapping** (`bool`, *optional*, defaults to `False`) --
  Whether or not to return `(char_start, char_end)` for each token.

  This is only available on fast tokenizers inheriting from [PreTrainedTokenizerFast](/docs/transformers/main/en/main_classes/tokenizer#transformers.TokenizersBackend), if using
  Python's tokenizer, this method will raise `NotImplementedError`.
- **return_length**  (`bool`, *optional*, defaults to `False`) --
  Whether or not to return the lengths of the encoded inputs.
- **verbose** (`bool`, *optional*, defaults to `True`) --
  Whether or not to print more information and warnings.
- ****kwargs** -- passed to the `self.tokenize()` method0

Main method to tokenize and prepare for the model one or several sequence(s) or one or several pair(s) of
sequences with word-level normalized bounding boxes and optional labels.

**Parameters:**

errors (`str`, *optional*, defaults to `"replace"`) : Paradigm to follow when decoding bytes to UTF-8. See [bytes.decode](https://docs.python.org/3/library/stdtypes.html#bytes.decode) for more information.

bos_token (`str`, *optional*, defaults to `""`) : The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.

eos_token (`str`, *optional*, defaults to `""`) : The end of sequence token.

sep_token (`str`, *optional*, defaults to `""`) : The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for sequence classification or for a text and a question for question answering. It is also used as the last token of a sequence built with special tokens.

cls_token (`str`, *optional*, defaults to `""`) : The classifier token which is used when doing sequence classification (classification of the whole sequence instead of per-token classification). It is the first token of the sequence when built with special tokens.

unk_token (`str`, *optional*, defaults to `""`) : The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this token instead.

pad_token (`str`, *optional*, defaults to `""`) : The token used for padding, for example when batching sequences of different lengths.

mask_token (`str`, *optional*, defaults to `""`) : The token used for masking values. This is the token used when training this model with masked language modeling. This is the token which the model will try to predict.

add_prefix_space (`bool`, *optional*, defaults to `True`) : Whether or not to add an initial space to the input. This allows to treat the leading word just as any other word.

cls_token_box (`list[int]`, *optional*, defaults to `[0, 0, 0, 0]`) : The bounding box to use for the special [CLS] token.

sep_token_box (`list[int]`, *optional*, defaults to `[0, 0, 0, 0]`) : The bounding box to use for the special [SEP] token.

pad_token_box (`list[int]`, *optional*, defaults to `[0, 0, 0, 0]`) : The bounding box to use for the special [PAD] token.

pad_token_label (`int`, *optional*, defaults to -100) : The label to use for padding tokens. Defaults to -100, which is the `ignore_index` of PyTorch's CrossEntropyLoss.

only_label_first_subword (`bool`, *optional*, defaults to `True`) : Whether or not to only label the first subword, in case word labels are provided.

vocab (`str` or `dict[str, int]`, *optional*) : Custom vocabulary dictionary. If not provided, vocabulary is loaded from `vocab_file` when using `from_pretrained`.

merges (`str` or `list[str]`, *optional*) : Custom merges list. If not provided, merges are loaded from `merges_file` when using `from_pretrained`.
#### save_vocabulary[[transformers.LayoutLMv3Tokenizer.save_vocabulary]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_utils_tokenizers.py#L384)

## LayoutLMv3TokenizerFast[[transformers.LayoutLMv3Tokenizer]]

#### transformers.LayoutLMv3Tokenizer[[transformers.LayoutLMv3Tokenizer]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/layoutlmv3/tokenization_layoutlmv3.py#L117)

Construct a LayoutLMv3 tokenizer (backed by HuggingFace's *tokenizers* library). Based on byte-level BPE.

This tokenizer inherits from [TokenizersBackend](/docs/transformers/main/en/main_classes/tokenizer#transformers.TokenizersBackend) which contains most of the main methods. Users should
refer to this superclass for more information regarding those methods.

__call__transformers.LayoutLMv3Tokenizer.__call__https://github.com/huggingface/transformers/blob/main/src/transformers/models/layoutlmv3/tokenization_layoutlmv3.py#L243[{"name": "text", "val": ": typing.Union[str, list[str], list[list[str]]]"}, {"name": "text_pair", "val": ": typing.Union[list[str], list[list[str]], NoneType] = None"}, {"name": "boxes", "val": ": typing.Union[list[list[int]], list[list[list[int]]], NoneType] = None"}, {"name": "word_labels", "val": ": typing.Union[list[int], list[list[int]], NoneType] = None"}, {"name": "add_special_tokens", "val": ": bool = True"}, {"name": "padding", "val": ": typing.Union[bool, str, transformers.utils.generic.PaddingStrategy] = False"}, {"name": "truncation", "val": ": typing.Union[bool, str, transformers.tokenization_utils_base.TruncationStrategy] = None"}, {"name": "max_length", "val": ": typing.Optional[int] = None"}, {"name": "stride", "val": ": int = 0"}, {"name": "pad_to_multiple_of", "val": ": typing.Optional[int] = None"}, {"name": "padding_side", "val": ": typing.Optional[str] = None"}, {"name": "return_tensors", "val": ": typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None"}, {"name": "return_token_type_ids", "val": ": typing.Optional[bool] = None"}, {"name": "return_attention_mask", "val": ": typing.Optional[bool] = None"}, {"name": "return_overflowing_tokens", "val": ": bool = False"}, {"name": "return_special_tokens_mask", "val": ": bool = False"}, {"name": "return_offsets_mapping", "val": ": bool = False"}, {"name": "return_length", "val": ": bool = False"}, {"name": "verbose", "val": ": bool = True"}, {"name": "**kwargs", "val": ""}]- **text** (`str`, `List[str]`, `List[List[str]]`) --
  The sequence or batch of sequences to be encoded. Each sequence can be a string, a list of strings
  (words of a single example or questions of a batch of examples) or a list of list of strings (batch of
  words).
- **text_pair** (`List[str]`, `List[List[str]]`) --
  The sequence or batch of sequences to be encoded. Each sequence should be a list of strings
  (pretokenized string).
- **boxes** (`List[List[int]]`, `List[List[List[int]]]`) --
  Word-level bounding boxes. Each bounding box should be normalized to be on a 0-1000 scale.
- **word_labels** (`List[int]`, `List[List[int]]`, *optional*) --
  Word-level integer labels (for token classification tasks such as FUNSD, CORD).

- **add_special_tokens** (`bool`, *optional*, defaults to `True`) --
  Whether or not to encode the sequences with the special tokens relative to their model.
- **padding** (`bool`, `str` or [PaddingStrategy](/docs/transformers/main/en/internal/file_utils#transformers.utils.PaddingStrategy), *optional*, defaults to `False`) --
  Activates and controls padding. Accepts the following values:

  - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
    sequence if provided).
  - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
    acceptable input length for the model if that argument is not provided.
  - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
    lengths).
- **truncation** (`bool`, `str` or [TruncationStrategy](/docs/transformers/main/en/internal/tokenization_utils#transformers.tokenization_utils_base.TruncationStrategy), *optional*, defaults to `False`) --
  Activates and controls truncation. Accepts the following values:

  - `True` or `'longest_first'`: Truncate to a maximum length specified with the argument `max_length` or
    to the maximum acceptable input length for the model if that argument is not provided. This will
    truncate token by token, removing a token from the longest sequence in the pair if a pair of
    sequences (or a batch of pairs) is provided.
  - `'only_first'`: Truncate to a maximum length specified with the argument `max_length` or to the
    maximum acceptable input length for the model if that argument is not provided. This will only
    truncate the first sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
  - `'only_second'`: Truncate to a maximum length specified with the argument `max_length` or to the
    maximum acceptable input length for the model if that argument is not provided. This will only
    truncate the second sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
  - `False` or `'do_not_truncate'` (default): No truncation (i.e., can output batch with sequence lengths
    greater than the model maximum admissible input size).
- **max_length** (`int`, *optional*) --
  Controls the maximum length to use by one of the truncation/padding parameters.

  If left unset or set to `None`, this will use the predefined model maximum length if a maximum length
  is required by one of the truncation/padding parameters. If the model has no specific maximum input
  length (like XLNet) truncation/padding to a maximum length will be deactivated.
- **stride** (`int`, *optional*, defaults to 0) --
  If set to a number along with `max_length`, the overflowing tokens returned when
  `return_overflowing_tokens=True` will contain some tokens from the end of the truncated sequence
  returned to provide some overlap between truncated and overflowing sequences. The value of this
  argument defines the number of overlapping tokens.
- **pad_to_multiple_of** (`int`, *optional*) --
  If set will pad the sequence to a multiple of the provided value. This is especially useful to enable
  the use of Tensor Cores on NVIDIA hardware with compute capability `>= 7.5` (Volta).

- **return_token_type_ids** (`bool`, *optional*) --
  Whether to return token type IDs. If left to the default, will return the token type IDs according to
  the specific tokenizer's default, defined by the `return_outputs` attribute.

  [What are token type IDs?](../glossary#token-type-ids)
- **return_attention_mask** (`bool`, *optional*) --
  Whether to return the attention mask. If left to the default, will return the attention mask according
  to the specific tokenizer's default, defined by the `return_outputs` attribute.

  [What are attention masks?](../glossary#attention-mask)
- **return_overflowing_tokens** (`bool`, *optional*, defaults to `False`) --
  Whether or not to return overflowing token sequences. If a pair of sequences of input ids (or a batch
  of pairs) is provided with `truncation_strategy = longest_first` or `True`, an error is raised instead
  of returning overflowing tokens.
- **return_special_tokens_mask** (`bool`, *optional*, defaults to `False`) --
  Whether or not to return special tokens mask information.
- **return_offsets_mapping** (`bool`, *optional*, defaults to `False`) --
  Whether or not to return `(char_start, char_end)` for each token.

  This is only available on fast tokenizers inheriting from [PreTrainedTokenizerFast](/docs/transformers/main/en/main_classes/tokenizer#transformers.TokenizersBackend), if using
  Python's tokenizer, this method will raise `NotImplementedError`.
- **return_length**  (`bool`, *optional*, defaults to `False`) --
  Whether or not to return the lengths of the encoded inputs.
- **verbose** (`bool`, *optional*, defaults to `True`) --
  Whether or not to print more information and warnings.
- ****kwargs** -- passed to the `self.tokenize()` method0

Main method to tokenize and prepare for the model one or several sequence(s) or one or several pair(s) of
sequences with word-level normalized bounding boxes and optional labels.

**Parameters:**

errors (`str`, *optional*, defaults to `"replace"`) : Paradigm to follow when decoding bytes to UTF-8. See [bytes.decode](https://docs.python.org/3/library/stdtypes.html#bytes.decode) for more information.

bos_token (`str`, *optional*, defaults to `""`) : The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.

eos_token (`str`, *optional*, defaults to `""`) : The end of sequence token.

sep_token (`str`, *optional*, defaults to `""`) : The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for sequence classification or for a text and a question for question answering. It is also used as the last token of a sequence built with special tokens.

cls_token (`str`, *optional*, defaults to `""`) : The classifier token which is used when doing sequence classification (classification of the whole sequence instead of per-token classification). It is the first token of the sequence when built with special tokens.

unk_token (`str`, *optional*, defaults to `""`) : The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this token instead.

pad_token (`str`, *optional*, defaults to `""`) : The token used for padding, for example when batching sequences of different lengths.

mask_token (`str`, *optional*, defaults to `""`) : The token used for masking values. This is the token used when training this model with masked language modeling. This is the token which the model will try to predict.

add_prefix_space (`bool`, *optional*, defaults to `True`) : Whether or not to add an initial space to the input. This allows to treat the leading word just as any other word.

cls_token_box (`list[int]`, *optional*, defaults to `[0, 0, 0, 0]`) : The bounding box to use for the special [CLS] token.

sep_token_box (`list[int]`, *optional*, defaults to `[0, 0, 0, 0]`) : The bounding box to use for the special [SEP] token.

pad_token_box (`list[int]`, *optional*, defaults to `[0, 0, 0, 0]`) : The bounding box to use for the special [PAD] token.

pad_token_label (`int`, *optional*, defaults to -100) : The label to use for padding tokens. Defaults to -100, which is the `ignore_index` of PyTorch's CrossEntropyLoss.

only_label_first_subword (`bool`, *optional*, defaults to `True`) : Whether or not to only label the first subword, in case word labels are provided.

vocab (`str` or `dict[str, int]`, *optional*) : Custom vocabulary dictionary. If not provided, vocabulary is loaded from `vocab_file` when using `from_pretrained`.

merges (`str` or `list[str]`, *optional*) : Custom merges list. If not provided, merges are loaded from `merges_file` when using `from_pretrained`.

## LayoutLMv3Processor[[transformers.LayoutLMv3Processor]]

#### transformers.LayoutLMv3Processor[[transformers.LayoutLMv3Processor]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/layoutlmv3/processing_layoutlmv3.py#L26)

Constructs a LayoutLMv3 processor which combines a LayoutLMv3 image processor and a LayoutLMv3 tokenizer into a
single processor.

[LayoutLMv3Processor](/docs/transformers/main/en/model_doc/layoutlmv3#transformers.LayoutLMv3Processor) offers all the functionalities you need to prepare data for the model.

It first uses [LayoutLMv3ImageProcessor](/docs/transformers/main/en/model_doc/layoutlmv3#transformers.LayoutLMv3ImageProcessor) to resize and normalize document images, and optionally applies OCR to
get words and normalized bounding boxes. These are then provided to [LayoutLMv3Tokenizer](/docs/transformers/main/en/model_doc/layoutlmv3#transformers.LayoutLMv3Tokenizer) or
[LayoutLMv3TokenizerFast](/docs/transformers/main/en/model_doc/layoutlmv3#transformers.LayoutLMv3Tokenizer), which turns the words and bounding boxes into token-level `input_ids`,
`attention_mask`, `token_type_ids`, `bbox`. Optionally, one can provide integer `word_labels`, which are turned
into token-level `labels` for token classification tasks (such as FUNSD, CORD).

__call__transformers.LayoutLMv3Processor.__call__https://github.com/huggingface/transformers/blob/main/src/transformers/models/layoutlmv3/processing_layoutlmv3.py#L49[{"name": "images", "val": ""}, {"name": "text", "val": ": typing.Union[str, list[str], list[list[str]]] = None"}, {"name": "text_pair", "val": ": typing.Union[list[str], list[list[str]], NoneType] = None"}, {"name": "boxes", "val": ": typing.Union[list[list[int]], list[list[list[int]]], NoneType] = None"}, {"name": "word_labels", "val": ": typing.Union[list[int], list[list[int]], NoneType] = None"}, {"name": "add_special_tokens", "val": ": bool = True"}, {"name": "padding", "val": ": typing.Union[bool, str, transformers.utils.generic.PaddingStrategy] = False"}, {"name": "truncation", "val": ": typing.Union[bool, str, transformers.tokenization_utils_base.TruncationStrategy] = None"}, {"name": "max_length", "val": ": typing.Optional[int] = None"}, {"name": "stride", "val": ": int = 0"}, {"name": "pad_to_multiple_of", "val": ": typing.Optional[int] = None"}, {"name": "return_token_type_ids", "val": ": typing.Optional[bool] = None"}, {"name": "return_attention_mask", "val": ": typing.Optional[bool] = None"}, {"name": "return_overflowing_tokens", "val": ": bool = False"}, {"name": "return_special_tokens_mask", "val": ": bool = False"}, {"name": "return_offsets_mapping", "val": ": bool = False"}, {"name": "return_length", "val": ": bool = False"}, {"name": "verbose", "val": ": bool = True"}, {"name": "return_tensors", "val": ": typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None"}, {"name": "**kwargs", "val": ""}]

This method first forwards the `images` argument to [__call__()](/docs/transformers/main/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__). In case
[LayoutLMv3ImageProcessor](/docs/transformers/main/en/model_doc/layoutlmv3#transformers.LayoutLMv3ImageProcessor) was initialized with `apply_ocr` set to `True`, it passes the obtained words and
bounding boxes along with the additional arguments to [__call__()](/docs/transformers/main/en/model_doc/layoutlmv3#transformers.LayoutLMv3Tokenizer.__call__) and returns the output,
together with resized and normalized `pixel_values`. In case [LayoutLMv3ImageProcessor](/docs/transformers/main/en/model_doc/layoutlmv3#transformers.LayoutLMv3ImageProcessor) was initialized with
`apply_ocr` set to `False`, it passes the words (`text`/``text_pair`) and `boxes` specified by the user along
with the additional arguments to [__call__()](/docs/transformers/main/en/model_doc/layoutlmv3#transformers.LayoutLMv3Tokenizer.__call__) and returns the output, together with
resized and normalized `pixel_values`.

Please refer to the docstring of the above two methods for more information.

**Parameters:**

image_processor (`LayoutLMv3ImageProcessor`, *optional*) : An instance of [LayoutLMv3ImageProcessor](/docs/transformers/main/en/model_doc/layoutlmv3#transformers.LayoutLMv3ImageProcessor). The image processor is a required input.

tokenizer (`LayoutLMv3Tokenizer` or `LayoutLMv3TokenizerFast`, *optional*) : An instance of [LayoutLMv3Tokenizer](/docs/transformers/main/en/model_doc/layoutlmv3#transformers.LayoutLMv3Tokenizer) or [LayoutLMv3TokenizerFast](/docs/transformers/main/en/model_doc/layoutlmv3#transformers.LayoutLMv3Tokenizer). The tokenizer is a required input.

## LayoutLMv3Model[[transformers.LayoutLMv3Model]]

#### transformers.LayoutLMv3Model[[transformers.LayoutLMv3Model]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/layoutlmv3/modeling_layoutlmv3.py#L566)

The bare Layoutlmv3 Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

forwardtransformers.LayoutLMv3Model.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/layoutlmv3/modeling_layoutlmv3.py#L647[{"name": "input_ids", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "bbox", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "attention_mask", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "token_type_ids", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "position_ids", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "inputs_embeds", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "pixel_values", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "output_attentions", "val": ": typing.Optional[bool] = None"}, {"name": "output_hidden_states", "val": ": typing.Optional[bool] = None"}, {"name": "return_dict", "val": ": typing.Optional[bool] = None"}, {"name": "**kwargs", "val": ""}]- **input_ids** (`torch.LongTensor` of shape `(batch_size, token_sequence_length)`) --
  Indices of input sequence tokens in the vocabulary.

  Note that `sequence_length = token_sequence_length + patch_sequence_length + 1` where `1` is for [CLS]
  token. See `pixel_values` for `patch_sequence_length`.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
- **bbox** (`torch.LongTensor` of shape `(batch_size, token_sequence_length, 4)`, *optional*) --
  Bounding boxes of each input sequence tokens. Selected in the range `[0,
  config.max_2d_position_embeddings-1]`. Each bounding box should be a normalized version in (x0, y0, x1, y1)
  format, where (x0, y0) corresponds to the position of the upper left corner in the bounding box, and (x1,
  y1) represents the position of the lower right corner.

  Note that `sequence_length = token_sequence_length + patch_sequence_length + 1` where `1` is for [CLS]
  token. See `pixel_values` for `patch_sequence_length`.
- **attention_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

  - 1 for tokens that are **not masked**,
  - 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
- **token_type_ids** (`torch.LongTensor` of shape `(batch_size, token_sequence_length)`, *optional*) --
  Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
  1]`:

  - 0 corresponds to a *sentence A* token,
  - 1 corresponds to a *sentence B* token.

  Note that `sequence_length = token_sequence_length + patch_sequence_length + 1` where `1` is for [CLS]
  token. See `pixel_values` for `patch_sequence_length`.

  [What are token type IDs?](../glossary#token-type-ids)
- **position_ids** (`torch.LongTensor` of shape `(batch_size, token_sequence_length)`, *optional*) --
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
  config.max_position_embeddings - 1]`.

  Note that `sequence_length = token_sequence_length + patch_sequence_length + 1` where `1` is for [CLS]
  token. See `pixel_values` for `patch_sequence_length`.

  [What are position IDs?](../glossary#position-ids)
- **inputs_embeds** (`torch.FloatTensor` of shape `(batch_size, token_sequence_length, hidden_size)`, *optional*) --
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert *input_ids* indices into associated vectors than the
  model's internal embedding lookup matrix.
- **pixel_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) --
  The tensors corresponding to the input images. Pixel values can be obtained using
  [LayoutLMv3ImageProcessor](/docs/transformers/main/en/model_doc/layoutlmv3#transformers.LayoutLMv3ImageProcessor). See [LayoutLMv3ImageProcessor.__call__()](/docs/transformers/main/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([LayoutLMv3Processor](/docs/transformers/main/en/model_doc/layoutlmv3#transformers.LayoutLMv3Processor) uses
  [LayoutLMv3ImageProcessor](/docs/transformers/main/en/model_doc/layoutlmv3#transformers.LayoutLMv3ImageProcessor) for processing images).
- **output_attentions** (`bool`, *optional*) --
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
- **output_hidden_states** (`bool`, *optional*) --
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
- **return_dict** (`bool`, *optional*) --
  Whether or not to return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.0[transformers.modeling_outputs.BaseModelOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.BaseModelOutput) or `tuple(torch.FloatTensor)`A [transformers.modeling_outputs.BaseModelOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.BaseModelOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([LayoutLMv3Config](/docs/transformers/main/en/model_doc/layoutlmv3#transformers.LayoutLMv3Config)) and inputs.

- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) -- Sequence of hidden-states at the output of the last layer of the model.
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
The [LayoutLMv3Model](/docs/transformers/main/en/model_doc/layoutlmv3#transformers.LayoutLMv3Model) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:

```python
>>> from transformers import AutoProcessor, AutoModel
>>> from datasets import load_dataset

>>> processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
>>> model = AutoModel.from_pretrained("microsoft/layoutlmv3-base")

>>> dataset = load_dataset("nielsr/funsd-layoutlmv3", split="train")
>>> example = dataset[0]
>>> image = example["image"]
>>> words = example["tokens"]
>>> boxes = example["bboxes"]

>>> encoding = processor(image, words, boxes=boxes, return_tensors="pt")

>>> outputs = model(**encoding)
>>> last_hidden_states = outputs.last_hidden_state
```

**Parameters:**

config ([LayoutLMv3Model](/docs/transformers/main/en/model_doc/layoutlmv3#transformers.LayoutLMv3Model)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

**Returns:**

`[transformers.modeling_outputs.BaseModelOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.BaseModelOutput) or `tuple(torch.FloatTensor)``

A [transformers.modeling_outputs.BaseModelOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.BaseModelOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([LayoutLMv3Config](/docs/transformers/main/en/model_doc/layoutlmv3#transformers.LayoutLMv3Config)) and inputs.

- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) -- Sequence of hidden-states at the output of the last layer of the model.
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

## LayoutLMv3ForSequenceClassification[[transformers.LayoutLMv3ForSequenceClassification]]

#### transformers.LayoutLMv3ForSequenceClassification[[transformers.LayoutLMv3ForSequenceClassification]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/layoutlmv3/modeling_layoutlmv3.py#L1097)

LayoutLMv3 Model with a sequence classification head on top (a linear layer on top of the final hidden state of the
[CLS] token) e.g. for document image classification tasks such as the
[RVL-CDIP](https://www.cs.cmu.edu/~aharley/rvl-cdip/) dataset.

This model inherits from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

forwardtransformers.LayoutLMv3ForSequenceClassification.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/layoutlmv3/modeling_layoutlmv3.py#L1107[{"name": "input_ids", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "attention_mask", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "token_type_ids", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "position_ids", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "inputs_embeds", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "labels", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "output_attentions", "val": ": typing.Optional[bool] = None"}, {"name": "output_hidden_states", "val": ": typing.Optional[bool] = None"}, {"name": "return_dict", "val": ": typing.Optional[bool] = None"}, {"name": "bbox", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "pixel_values", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "**kwargs", "val": ""}]- **input_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
- **attention_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

  - 1 for tokens that are **not masked**,
  - 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
- **token_type_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:

  - 0 corresponds to a *sentence A* token,
  - 1 corresponds to a *sentence B* token.

  [What are token type IDs?](../glossary#token-type-ids)
- **position_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
- **inputs_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) --
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model's internal embedding lookup matrix.
- **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
  config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
  (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
- **output_attentions** (`bool`, *optional*) --
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
- **output_hidden_states** (`bool`, *optional*) --
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
- **return_dict** (`bool`, *optional*) --
  Whether or not to return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
- **bbox** (`torch.LongTensor` of shape `(batch_size, sequence_length, 4)`, *optional*) --
  Bounding boxes of each input sequence tokens. Selected in the range `[0,
  config.max_2d_position_embeddings-1]`. Each bounding box should be a normalized version in (x0, y0, x1, y1)
  format, where (x0, y0) corresponds to the position of the upper left corner in the bounding box, and (x1,
  y1) represents the position of the lower right corner.
- **pixel_values** (`torch.LongTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) --
  The tensors corresponding to the input images. Pixel values can be obtained using
  [LayoutLMv3ImageProcessor](/docs/transformers/main/en/model_doc/layoutlmv3#transformers.LayoutLMv3ImageProcessor). See [LayoutLMv3ImageProcessor.__call__()](/docs/transformers/main/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([LayoutLMv3Processor](/docs/transformers/main/en/model_doc/layoutlmv3#transformers.LayoutLMv3Processor) uses
  [LayoutLMv3ImageProcessor](/docs/transformers/main/en/model_doc/layoutlmv3#transformers.LayoutLMv3ImageProcessor) for processing images).0[transformers.modeling_outputs.SequenceClassifierOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput) or `tuple(torch.FloatTensor)`A [transformers.modeling_outputs.SequenceClassifierOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([LayoutLMv3Config](/docs/transformers/main/en/model_doc/layoutlmv3#transformers.LayoutLMv3Config)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Classification (or regression if config.num_labels==1) loss.
- **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) -- Classification (or regression if config.num_labels==1) scores (before SoftMax).
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
The [LayoutLMv3ForSequenceClassification](/docs/transformers/main/en/model_doc/layoutlmv3#transformers.LayoutLMv3ForSequenceClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:

```python
>>> from transformers import AutoProcessor, AutoModelForSequenceClassification
>>> from datasets import load_dataset
>>> import torch

>>> processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
>>> model = AutoModelForSequenceClassification.from_pretrained("microsoft/layoutlmv3-base")

>>> dataset = load_dataset("nielsr/funsd-layoutlmv3", split="train")
>>> example = dataset[0]
>>> image = example["image"]
>>> words = example["tokens"]
>>> boxes = example["bboxes"]

>>> encoding = processor(image, words, boxes=boxes, return_tensors="pt")
>>> sequence_label = torch.tensor([1])

>>> outputs = model(**encoding, labels=sequence_label)
>>> loss = outputs.loss
>>> logits = outputs.logits
```

**Parameters:**

config ([LayoutLMv3ForSequenceClassification](/docs/transformers/main/en/model_doc/layoutlmv3#transformers.LayoutLMv3ForSequenceClassification)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

**Returns:**

`[transformers.modeling_outputs.SequenceClassifierOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput) or `tuple(torch.FloatTensor)``

A [transformers.modeling_outputs.SequenceClassifierOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([LayoutLMv3Config](/docs/transformers/main/en/model_doc/layoutlmv3#transformers.LayoutLMv3Config)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Classification (or regression if config.num_labels==1) loss.
- **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) -- Classification (or regression if config.num_labels==1) scores (before SoftMax).
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

## LayoutLMv3ForTokenClassification[[transformers.LayoutLMv3ForTokenClassification]]

#### transformers.LayoutLMv3ForTokenClassification[[transformers.LayoutLMv3ForTokenClassification]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/layoutlmv3/modeling_layoutlmv3.py#L873)

LayoutLMv3 Model with a token classification head on top (a linear layer on top of the final hidden states) e.g.
for sequence labeling (information extraction) tasks such as [FUNSD](https://guillaumejaume.github.io/FUNSD/),
[SROIE](https://rrc.cvc.uab.es/?ch=13), [CORD](https://github.com/clovaai/cord) and
[Kleister-NDA](https://github.com/applicaai/kleister-nda).

This model inherits from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

forwardtransformers.LayoutLMv3ForTokenClassification.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/layoutlmv3/modeling_layoutlmv3.py#L887[{"name": "input_ids", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "bbox", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "attention_mask", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "token_type_ids", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "position_ids", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "inputs_embeds", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "labels", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "output_attentions", "val": ": typing.Optional[bool] = None"}, {"name": "output_hidden_states", "val": ": typing.Optional[bool] = None"}, {"name": "return_dict", "val": ": typing.Optional[bool] = None"}, {"name": "pixel_values", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "**kwargs", "val": ""}]- **input_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
- **bbox** (`torch.LongTensor` of shape `(batch_size, sequence_length, 4)`, *optional*) --
  Bounding boxes of each input sequence tokens. Selected in the range `[0,
  config.max_2d_position_embeddings-1]`. Each bounding box should be a normalized version in (x0, y0, x1, y1)
  format, where (x0, y0) corresponds to the position of the upper left corner in the bounding box, and (x1,
  y1) represents the position of the lower right corner.
- **attention_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

  - 1 for tokens that are **not masked**,
  - 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
- **token_type_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:

  - 0 corresponds to a *sentence A* token,
  - 1 corresponds to a *sentence B* token.

  [What are token type IDs?](../glossary#token-type-ids)
- **position_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
- **inputs_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) --
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model's internal embedding lookup matrix.
- **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
- **output_attentions** (`bool`, *optional*) --
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
- **output_hidden_states** (`bool`, *optional*) --
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
- **return_dict** (`bool`, *optional*) --
  Whether or not to return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
- **pixel_values** (`torch.LongTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) --
  The tensors corresponding to the input images. Pixel values can be obtained using
  [LayoutLMv3ImageProcessor](/docs/transformers/main/en/model_doc/layoutlmv3#transformers.LayoutLMv3ImageProcessor). See [LayoutLMv3ImageProcessor.__call__()](/docs/transformers/main/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([LayoutLMv3Processor](/docs/transformers/main/en/model_doc/layoutlmv3#transformers.LayoutLMv3Processor) uses
  [LayoutLMv3ImageProcessor](/docs/transformers/main/en/model_doc/layoutlmv3#transformers.LayoutLMv3ImageProcessor) for processing images).0[transformers.modeling_outputs.TokenClassifierOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput) or `tuple(torch.FloatTensor)`A [transformers.modeling_outputs.TokenClassifierOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([LayoutLMv3Config](/docs/transformers/main/en/model_doc/layoutlmv3#transformers.LayoutLMv3Config)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided)  -- Classification loss.
- **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_labels)`) -- Classification scores (before SoftMax).
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
The [LayoutLMv3ForTokenClassification](/docs/transformers/main/en/model_doc/layoutlmv3#transformers.LayoutLMv3ForTokenClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:

```python
>>> from transformers import AutoProcessor, AutoModelForTokenClassification
>>> from datasets import load_dataset

>>> processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
>>> model = AutoModelForTokenClassification.from_pretrained("microsoft/layoutlmv3-base", num_labels=7)

>>> dataset = load_dataset("nielsr/funsd-layoutlmv3", split="train")
>>> example = dataset[0]
>>> image = example["image"]
>>> words = example["tokens"]
>>> boxes = example["bboxes"]
>>> word_labels = example["ner_tags"]

>>> encoding = processor(image, words, boxes=boxes, word_labels=word_labels, return_tensors="pt")

>>> outputs = model(**encoding)
>>> loss = outputs.loss
>>> logits = outputs.logits
```

**Parameters:**

config ([LayoutLMv3ForTokenClassification](/docs/transformers/main/en/model_doc/layoutlmv3#transformers.LayoutLMv3ForTokenClassification)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

**Returns:**

`[transformers.modeling_outputs.TokenClassifierOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput) or `tuple(torch.FloatTensor)``

A [transformers.modeling_outputs.TokenClassifierOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([LayoutLMv3Config](/docs/transformers/main/en/model_doc/layoutlmv3#transformers.LayoutLMv3Config)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided)  -- Classification loss.
- **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_labels)`) -- Classification scores (before SoftMax).
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

## LayoutLMv3ForQuestionAnswering[[transformers.LayoutLMv3ForQuestionAnswering]]

#### transformers.LayoutLMv3ForQuestionAnswering[[transformers.LayoutLMv3ForQuestionAnswering]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/layoutlmv3/modeling_layoutlmv3.py#L977)

The Layoutlmv3 transformer with a span classification head on top for extractive question-answering tasks like
SQuAD (a linear layer on top of the hidden-states output to compute `span start logits` and `span end logits`).

This model inherits from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

forwardtransformers.LayoutLMv3ForQuestionAnswering.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/layoutlmv3/modeling_layoutlmv3.py#L987[{"name": "input_ids", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "attention_mask", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "token_type_ids", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "position_ids", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "inputs_embeds", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "start_positions", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "end_positions", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "output_attentions", "val": ": typing.Optional[bool] = None"}, {"name": "output_hidden_states", "val": ": typing.Optional[bool] = None"}, {"name": "return_dict", "val": ": typing.Optional[bool] = None"}, {"name": "bbox", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "pixel_values", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "**kwargs", "val": ""}]- **input_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
- **attention_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

  - 1 for tokens that are **not masked**,
  - 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
- **token_type_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:

  - 0 corresponds to a *sentence A* token,
  - 1 corresponds to a *sentence B* token.

  [What are token type IDs?](../glossary#token-type-ids)
- **position_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
- **inputs_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) --
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model's internal embedding lookup matrix.
- **start_positions** (`torch.LongTensor` of shape `(batch_size,)`, *optional*) --
  Labels for position (index) of the start of the labelled span for computing the token classification loss.
  Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
  are not taken into account for computing the loss.
- **end_positions** (`torch.LongTensor` of shape `(batch_size,)`, *optional*) --
  Labels for position (index) of the end of the labelled span for computing the token classification loss.
  Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
  are not taken into account for computing the loss.
- **output_attentions** (`bool`, *optional*) --
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
- **output_hidden_states** (`bool`, *optional*) --
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
- **return_dict** (`bool`, *optional*) --
  Whether or not to return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
- **bbox** (`torch.LongTensor` of shape `(batch_size, sequence_length, 4)`, *optional*) --
  Bounding boxes of each input sequence tokens. Selected in the range `[0,
  config.max_2d_position_embeddings-1]`. Each bounding box should be a normalized version in (x0, y0, x1, y1)
  format, where (x0, y0) corresponds to the position of the upper left corner in the bounding box, and (x1,
  y1) represents the position of the lower right corner.
- **pixel_values** (`torch.LongTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) --
  The tensors corresponding to the input images. Pixel values can be obtained using
  [LayoutLMv3ImageProcessor](/docs/transformers/main/en/model_doc/layoutlmv3#transformers.LayoutLMv3ImageProcessor). See [LayoutLMv3ImageProcessor.__call__()](/docs/transformers/main/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([LayoutLMv3Processor](/docs/transformers/main/en/model_doc/layoutlmv3#transformers.LayoutLMv3Processor) uses
  [LayoutLMv3ImageProcessor](/docs/transformers/main/en/model_doc/layoutlmv3#transformers.LayoutLMv3ImageProcessor) for processing images).0[transformers.modeling_outputs.QuestionAnsweringModelOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.QuestionAnsweringModelOutput) or `tuple(torch.FloatTensor)`A [transformers.modeling_outputs.QuestionAnsweringModelOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.QuestionAnsweringModelOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([LayoutLMv3Config](/docs/transformers/main/en/model_doc/layoutlmv3#transformers.LayoutLMv3Config)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
- **start_logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`) -- Span-start scores (before SoftMax).
- **end_logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`) -- Span-end scores (before SoftMax).
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
The [LayoutLMv3ForQuestionAnswering](/docs/transformers/main/en/model_doc/layoutlmv3#transformers.LayoutLMv3ForQuestionAnswering) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:

```python
>>> from transformers import AutoProcessor, AutoModelForQuestionAnswering
>>> from datasets import load_dataset
>>> import torch

>>> processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
>>> model = AutoModelForQuestionAnswering.from_pretrained("microsoft/layoutlmv3-base")

>>> dataset = load_dataset("nielsr/funsd-layoutlmv3", split="train")
>>> example = dataset[0]
>>> image = example["image"]
>>> question = "what's his name?"
>>> words = example["tokens"]
>>> boxes = example["bboxes"]

>>> encoding = processor(image, question, words, boxes=boxes, return_tensors="pt")
>>> start_positions = torch.tensor([1])
>>> end_positions = torch.tensor([3])

>>> outputs = model(**encoding, start_positions=start_positions, end_positions=end_positions)
>>> loss = outputs.loss
>>> start_scores = outputs.start_logits
>>> end_scores = outputs.end_logits
```

**Parameters:**

config ([LayoutLMv3ForQuestionAnswering](/docs/transformers/main/en/model_doc/layoutlmv3#transformers.LayoutLMv3ForQuestionAnswering)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

**Returns:**

`[transformers.modeling_outputs.QuestionAnsweringModelOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.QuestionAnsweringModelOutput) or `tuple(torch.FloatTensor)``

A [transformers.modeling_outputs.QuestionAnsweringModelOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.QuestionAnsweringModelOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([LayoutLMv3Config](/docs/transformers/main/en/model_doc/layoutlmv3#transformers.LayoutLMv3Config)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
- **start_logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`) -- Span-start scores (before SoftMax).
- **end_logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`) -- Span-end scores (before SoftMax).
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
