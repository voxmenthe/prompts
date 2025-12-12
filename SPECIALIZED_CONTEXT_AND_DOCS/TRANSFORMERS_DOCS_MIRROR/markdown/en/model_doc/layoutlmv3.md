*This model was released on 2022-04-18 and added to Hugging Face Transformers on 2022-05-24.*

# LayoutLMv3

## Overview

The LayoutLMv3 model was proposed in [LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking](https://huggingface.co/papers/2204.08387) by Yupan Huang, Tengchao Lv, Lei Cui, Yutong Lu, Furu Wei.
LayoutLMv3 simplifies [LayoutLMv2](layoutlmv2) by using patch embeddings (as in [ViT](vit)) instead of leveraging a CNN backbone, and pre-trains the model on 3 objectives: masked language modeling (MLM), masked image modeling (MIM)
and word-patch alignment (WPA).

The abstract from the paper is the following:

*Self-supervised pre-training techniques have achieved remarkable progress in Document AI. Most multimodal pre-trained models use a masked language modeling objective to learn bidirectional representations on the text modality, but they differ in pre-training objectives for the image modality. This discrepancy adds difficulty to multimodal representation learning. In this paper, we propose LayoutLMv3 to pre-train multimodal Transformers for Document AI with unified text and image masking. Additionally, LayoutLMv3 is pre-trained with a word-patch alignment objective to learn cross-modal alignment by predicting whether the corresponding image patch of a text word is masked. The simple unified architecture and training objectives make LayoutLMv3 a general-purpose pre-trained model for both text-centric and image-centric Document AI tasks. Experimental results show that LayoutLMv3 achieves state-of-the-art performance not only in text-centric tasks, including form understanding, receipt understanding, and document visual question answering, but also in image-centric tasks such as document image classification and document layout analysis.*

![drawing](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/layoutlmv3_architecture.png) LayoutLMv3 architecture. Taken from the [original paper](https://huggingface.co/papers/2204.08387).

This model was contributed by [nielsr](https://huggingface.co/nielsr). The original code can be found [here](https://github.com/microsoft/unilm/tree/master/layoutlmv3).

## Usage tips

* In terms of data processing, LayoutLMv3 is identical to its predecessor [LayoutLMv2](layoutlmv2), except that:
  + images need to be resized and normalized with channels in regular RGB format. LayoutLMv2 on the other hand normalizes the images internally and expects the channels in BGR format.
  + text is tokenized using byte-pair encoding (BPE), as opposed to WordPiece.
    Due to these differences in data preprocessing, one can use [LayoutLMv3Processor](/docs/transformers/v4.56.2/en/model_doc/layoutlmv3#transformers.LayoutLMv3Processor) which internally combines a [LayoutLMv3ImageProcessor](/docs/transformers/v4.56.2/en/model_doc/layoutlmv3#transformers.LayoutLMv3ImageProcessor) (for the image modality) and a [LayoutLMv3Tokenizer](/docs/transformers/v4.56.2/en/model_doc/layoutlmv3#transformers.LayoutLMv3Tokenizer)/[LayoutLMv3TokenizerFast](/docs/transformers/v4.56.2/en/model_doc/layoutlmv3#transformers.LayoutLMv3TokenizerFast) (for the text modality) to prepare all data for the model.
* Regarding usage of [LayoutLMv3Processor](/docs/transformers/v4.56.2/en/model_doc/layoutlmv3#transformers.LayoutLMv3Processor), we refer to the [usage guide](layoutlmv2#usage-layoutlmv2processor) of its predecessor.

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with LayoutLMv3. If you’re interested in submitting a resource to be included here, please feel free to open a Pull Request and we’ll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

LayoutLMv3 is nearly identical to LayoutLMv2, so we’ve also included LayoutLMv2 resources you can adapt for LayoutLMv3 tasks. For these notebooks, take care to use [LayoutLMv2Processor](/docs/transformers/v4.56.2/en/model_doc/layoutlmv2#transformers.LayoutLMv2Processor) instead when preparing data for the model!

* Demo notebooks for LayoutLMv3 can be found [here](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/LayoutLMv3).
* Demo scripts can be found [here](https://github.com/huggingface/transformers-research-projects/tree/main/layoutlmv3).

Text Classification

* [LayoutLMv2ForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/layoutlmv2#transformers.LayoutLMv2ForSequenceClassification) is supported by this [notebook](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/LayoutLMv2/RVL-CDIP/Fine_tuning_LayoutLMv2ForSequenceClassification_on_RVL_CDIP.ipynb).
* [Text classification task guide](../tasks/sequence_classification)

Token Classification

* [LayoutLMv3ForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/layoutlmv3#transformers.LayoutLMv3ForTokenClassification) is supported by this [example script](https://github.com/huggingface/transformers-research-projects/tree/main/layoutlmv3) and [notebook](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/LayoutLMv3/Fine_tune_LayoutLMv3_on_FUNSD_(HuggingFace_Trainer).ipynb).
* A [notebook](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/LayoutLMv2/FUNSD/Inference_with_LayoutLMv2ForTokenClassification.ipynb) for how to perform inference with [LayoutLMv2ForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/layoutlmv2#transformers.LayoutLMv2ForTokenClassification) and a [notebook](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/LayoutLMv2/FUNSD/True_inference_with_LayoutLMv2ForTokenClassification_%2B_Gradio_demo.ipynb) for how to perform inference when no labels are available with [LayoutLMv2ForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/layoutlmv2#transformers.LayoutLMv2ForTokenClassification).
* A [notebook](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/LayoutLMv2/FUNSD/Fine_tuning_LayoutLMv2ForTokenClassification_on_FUNSD_using_HuggingFace_Trainer.ipynb) for how to finetune [LayoutLMv2ForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/layoutlmv2#transformers.LayoutLMv2ForTokenClassification) with the 🤗 Trainer.
* [Token classification task guide](../tasks/token_classification)

Question Answering

* [LayoutLMv2ForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/layoutlmv2#transformers.LayoutLMv2ForQuestionAnswering) is supported by this [notebook](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/LayoutLMv2/DocVQA/Fine_tuning_LayoutLMv2ForQuestionAnswering_on_DocVQA.ipynb).
* [Question answering task guide](../tasks/question_answering)

**Document question answering**

* [Document question answering task guide](../tasks/document_question_answering)

## LayoutLMv3Config

### class transformers.LayoutLMv3Config

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/layoutlmv3/configuration_layoutlmv3.py#L37)

( vocab\_size = 50265 hidden\_size = 768 num\_hidden\_layers = 12 num\_attention\_heads = 12 intermediate\_size = 3072 hidden\_act = 'gelu' hidden\_dropout\_prob = 0.1 attention\_probs\_dropout\_prob = 0.1 max\_position\_embeddings = 512 type\_vocab\_size = 2 initializer\_range = 0.02 layer\_norm\_eps = 1e-05 pad\_token\_id = 1 bos\_token\_id = 0 eos\_token\_id = 2 max\_2d\_position\_embeddings = 1024 coordinate\_size = 128 shape\_size = 128 has\_relative\_attention\_bias = True rel\_pos\_bins = 32 max\_rel\_pos = 128 rel\_2d\_pos\_bins = 64 max\_rel\_2d\_pos = 256 has\_spatial\_attention\_bias = True text\_embed = True visual\_embed = True input\_size = 224 num\_channels = 3 patch\_size = 16 classifier\_dropout = None \*\*kwargs  )

Parameters

* **vocab\_size** (`int`, *optional*, defaults to 50265) —
  Vocabulary size of the LayoutLMv3 model. Defines the number of different tokens that can be represented by
  the `inputs_ids` passed when calling [LayoutLMv3Model](/docs/transformers/v4.56.2/en/model_doc/layoutlmv3#transformers.LayoutLMv3Model).
* **hidden\_size** (`int`, *optional*, defaults to 768) —
  Dimension of the encoder layers and the pooler layer.
* **num\_hidden\_layers** (`int`, *optional*, defaults to 12) —
  Number of hidden layers in the Transformer encoder.
* **num\_attention\_heads** (`int`, *optional*, defaults to 12) —
  Number of attention heads for each attention layer in the Transformer encoder.
* **intermediate\_size** (`int`, *optional*, defaults to 3072) —
  Dimension of the “intermediate” (i.e., feed-forward) layer in the Transformer encoder.
* **hidden\_act** (`str` or `function`, *optional*, defaults to `"gelu"`) —
  The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
  `"relu"`, `"selu"` and `"gelu_new"` are supported.
* **hidden\_dropout\_prob** (`float`, *optional*, defaults to 0.1) —
  The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
* **attention\_probs\_dropout\_prob** (`float`, *optional*, defaults to 0.1) —
  The dropout ratio for the attention probabilities.
* **max\_position\_embeddings** (`int`, *optional*, defaults to 512) —
  The maximum sequence length that this model might ever be used with. Typically set this to something large
  just in case (e.g., 512 or 1024 or 2048).
* **type\_vocab\_size** (`int`, *optional*, defaults to 2) —
  The vocabulary size of the `token_type_ids` passed when calling [LayoutLMv3Model](/docs/transformers/v4.56.2/en/model_doc/layoutlmv3#transformers.LayoutLMv3Model).
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
* **layer\_norm\_eps** (`float`, *optional*, defaults to 1e-5) —
  The epsilon used by the layer normalization layers.
* **max\_2d\_position\_embeddings** (`int`, *optional*, defaults to 1024) —
  The maximum value that the 2D position embedding might ever be used with. Typically set this to something
  large just in case (e.g., 1024).
* **coordinate\_size** (`int`, *optional*, defaults to `128`) —
  Dimension of the coordinate embeddings.
* **shape\_size** (`int`, *optional*, defaults to `128`) —
  Dimension of the width and height embeddings.
* **has\_relative\_attention\_bias** (`bool`, *optional*, defaults to `True`) —
  Whether or not to use a relative attention bias in the self-attention mechanism.
* **rel\_pos\_bins** (`int`, *optional*, defaults to 32) —
  The number of relative position bins to be used in the self-attention mechanism.
* **max\_rel\_pos** (`int`, *optional*, defaults to 128) —
  The maximum number of relative positions to be used in the self-attention mechanism.
* **max\_rel\_2d\_pos** (`int`, *optional*, defaults to 256) —
  The maximum number of relative 2D positions in the self-attention mechanism.
* **rel\_2d\_pos\_bins** (`int`, *optional*, defaults to 64) —
  The number of 2D relative position bins in the self-attention mechanism.
* **has\_spatial\_attention\_bias** (`bool`, *optional*, defaults to `True`) —
  Whether or not to use a spatial attention bias in the self-attention mechanism.
* **visual\_embed** (`bool`, *optional*, defaults to `True`) —
  Whether or not to add patch embeddings.
* **input\_size** (`int`, *optional*, defaults to `224`) —
  The size (resolution) of the images.
* **num\_channels** (`int`, *optional*, defaults to `3`) —
  The number of channels of the images.
* **patch\_size** (`int`, *optional*, defaults to `16`) —
  The size (resolution) of the patches.
* **classifier\_dropout** (`float`, *optional*) —
  The dropout ratio for the classification head.

This is the configuration class to store the configuration of a [LayoutLMv3Model](/docs/transformers/v4.56.2/en/model_doc/layoutlmv3#transformers.LayoutLMv3Model). It is used to instantiate an
LayoutLMv3 model according to the specified arguments, defining the model architecture. Instantiating a
configuration with the defaults will yield a similar configuration to that of the LayoutLMv3
[microsoft/layoutlmv3-base](https://huggingface.co/microsoft/layoutlmv3-base) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import LayoutLMv3Config, LayoutLMv3Model

>>> # Initializing a LayoutLMv3 microsoft/layoutlmv3-base style configuration
>>> configuration = LayoutLMv3Config()

>>> # Initializing a model (with random weights) from the microsoft/layoutlmv3-base style configuration
>>> model = LayoutLMv3Model(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## LayoutLMv3FeatureExtractor

### class transformers.LayoutLMv3FeatureExtractor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/layoutlmv3/feature_extraction_layoutlmv3.py#L30)

( \*args \*\*kwargs  )

#### \_\_call\_\_

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/image_processing_utils.py#L49)

( images \*\*kwargs  )

Preprocess an image or a batch of images.

## LayoutLMv3ImageProcessor

### class transformers.LayoutLMv3ImageProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/layoutlmv3/image_processing_layoutlmv3.py#L106)

( do\_resize: bool = True size: typing.Optional[dict[str, int]] = None resample: Resampling = <Resampling.BILINEAR: 2> do\_rescale: bool = True rescale\_value: float = 0.00392156862745098 do\_normalize: bool = True image\_mean: typing.Union[float, collections.abc.Iterable[float], NoneType] = None image\_std: typing.Union[float, collections.abc.Iterable[float], NoneType] = None apply\_ocr: bool = True ocr\_lang: typing.Optional[str] = None tesseract\_config: typing.Optional[str] = '' \*\*kwargs  )

Parameters

* **do\_resize** (`bool`, *optional*, defaults to `True`) —
  Whether to resize the image’s (height, width) dimensions to `(size["height"], size["width"])`. Can be
  overridden by `do_resize` in `preprocess`.
* **size** (`dict[str, int]` *optional*, defaults to `{"height" -- 224, "width": 224}`):
  Size of the image after resizing. Can be overridden by `size` in `preprocess`.
* **resample** (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BILINEAR`) —
  Resampling filter to use if resizing the image. Can be overridden by `resample` in `preprocess`.
* **do\_rescale** (`bool`, *optional*, defaults to `True`) —
  Whether to rescale the image’s pixel values by the specified `rescale_value`. Can be overridden by
  `do_rescale` in `preprocess`.
* **rescale\_factor** (`float`, *optional*, defaults to 1 / 255) —
  Value by which the image’s pixel values are rescaled. Can be overridden by `rescale_factor` in
  `preprocess`.
* **do\_normalize** (`bool`, *optional*, defaults to `True`) —
  Whether to normalize the image. Can be overridden by the `do_normalize` parameter in the `preprocess`
  method.
* **image\_mean** (`Iterable[float]` or `float`, *optional*, defaults to `IMAGENET_STANDARD_MEAN`) —
  Mean to use if normalizing the image. This is a float or list of floats the length of the number of
  channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method.
* **image\_std** (`Iterable[float]` or `float`, *optional*, defaults to `IMAGENET_STANDARD_STD`) —
  Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
  number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.
* **apply\_ocr** (`bool`, *optional*, defaults to `True`) —
  Whether to apply the Tesseract OCR engine to get words + normalized bounding boxes. Can be overridden by
  the `apply_ocr` parameter in the `preprocess` method.
* **ocr\_lang** (`str`, *optional*) —
  The language, specified by its ISO code, to be used by the Tesseract OCR engine. By default, English is
  used. Can be overridden by the `ocr_lang` parameter in the `preprocess` method.
* **tesseract\_config** (`str`, *optional*) —
  Any additional custom configuration flags that are forwarded to the `config` parameter when calling
  Tesseract. For example: ‘—psm 6’. Can be overridden by the `tesseract_config` parameter in the
  `preprocess` method.

Constructs a LayoutLMv3 image processor.

#### preprocess

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/layoutlmv3/image_processing_layoutlmv3.py#L227)

( images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']] do\_resize: typing.Optional[bool] = None size: typing.Optional[dict[str, int]] = None resample = None do\_rescale: typing.Optional[bool] = None rescale\_factor: typing.Optional[float] = None do\_normalize: typing.Optional[bool] = None image\_mean: typing.Union[float, collections.abc.Iterable[float], NoneType] = None image\_std: typing.Union[float, collections.abc.Iterable[float], NoneType] = None apply\_ocr: typing.Optional[bool] = None ocr\_lang: typing.Optional[str] = None tesseract\_config: typing.Optional[str] = None return\_tensors: typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None data\_format: ChannelDimension = <ChannelDimension.FIRST: 'channels\_first'> input\_data\_format: typing.Union[str, transformers.image\_utils.ChannelDimension, NoneType] = None  )

Parameters

* **images** (`ImageInput`) —
  Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
  passing in images with pixel values between 0 and 1, set `do_rescale=False`.
* **do\_resize** (`bool`, *optional*, defaults to `self.do_resize`) —
  Whether to resize the image.
* **size** (`dict[str, int]`, *optional*, defaults to `self.size`) —
  Desired size of the output image after applying `resize`.
* **resample** (`int`, *optional*, defaults to `self.resample`) —
  Resampling filter to use if resizing the image. This can be one of the `PILImageResampling` filters.
  Only has an effect if `do_resize` is set to `True`.
* **do\_rescale** (`bool`, *optional*, defaults to `self.do_rescale`) —
  Whether to rescale the image pixel values between [0, 1].
* **rescale\_factor** (`float`, *optional*, defaults to `self.rescale_factor`) —
  Rescale factor to apply to the image pixel values. Only has an effect if `do_rescale` is set to `True`.
* **do\_normalize** (`bool`, *optional*, defaults to `self.do_normalize`) —
  Whether to normalize the image.
* **image\_mean** (`float` or `Iterable[float]`, *optional*, defaults to `self.image_mean`) —
  Mean values to be used for normalization. Only has an effect if `do_normalize` is set to `True`.
* **image\_std** (`float` or `Iterable[float]`, *optional*, defaults to `self.image_std`) —
  Standard deviation values to be used for normalization. Only has an effect if `do_normalize` is set to
  `True`.
* **apply\_ocr** (`bool`, *optional*, defaults to `self.apply_ocr`) —
  Whether to apply the Tesseract OCR engine to get words + normalized bounding boxes.
* **ocr\_lang** (`str`, *optional*, defaults to `self.ocr_lang`) —
  The language, specified by its ISO code, to be used by the Tesseract OCR engine. By default, English is
  used.
* **tesseract\_config** (`str`, *optional*, defaults to `self.tesseract_config`) —
  Any additional custom configuration flags that are forwarded to the `config` parameter when calling
  Tesseract.
* **return\_tensors** (`str` or `TensorType`, *optional*) —
  The type of tensors to return. Can be one of:
  + Unset: Return a list of `np.ndarray`.
  + `TensorType.TENSORFLOW` or `'tf'`: Return a batch of type `tf.Tensor`.
  + `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
  + `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
  + `TensorType.JAX` or `'jax'`: Return a batch of type `jax.numpy.ndarray`.
* **data\_format** (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`) —
  The channel dimension format for the output image. Can be one of:
  + `ChannelDimension.FIRST`: image in (num\_channels, height, width) format.
  + `ChannelDimension.LAST`: image in (height, width, num\_channels) format.
* **input\_data\_format** (`ChannelDimension` or `str`, *optional*) —
  The channel dimension format for the input image. If unset, the channel dimension format is inferred
  from the input image. Can be one of:
  + `"channels_first"` or `ChannelDimension.FIRST`: image in (num\_channels, height, width) format.
  + `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num\_channels) format.
  + `"none"` or `ChannelDimension.NONE`: image in (height, width) format.

Preprocess an image or batch of images.

## LayoutLMv3ImageProcessorFast

### class transformers.LayoutLMv3ImageProcessorFast

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/layoutlmv3/image_processing_layoutlmv3_fast.py#L68)

( \*\*kwargs: typing\_extensions.Unpack[transformers.models.layoutlmv3.image\_processing\_layoutlmv3\_fast.LayoutLMv3FastImageProcessorKwargs]  )

Constructs a fast Layoutlmv3 image processor.

#### preprocess

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/layoutlmv3/image_processing_layoutlmv3_fast.py#L84)

( images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']] \*\*kwargs: typing\_extensions.Unpack[transformers.models.layoutlmv3.image\_processing\_layoutlmv3\_fast.LayoutLMv3FastImageProcessorKwargs]  ) → `<class 'transformers.image_processing_base.BatchFeature'>`

Parameters

* **images** (`Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']]`) —
  Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
  passing in images with pixel values between 0 and 1, set `do_rescale=False`.
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
* **apply\_ocr** (`bool`, *optional*, defaults to `True`) —
  Whether to apply the Tesseract OCR engine to get words + normalized bounding boxes. Can be overridden by
  the `apply_ocr` parameter in the `preprocess` method.
* **ocr\_lang** (`str`, *optional*) —
  The language, specified by its ISO code, to be used by the Tesseract OCR engine. By default, English is
  used. Can be overridden by the `ocr_lang` parameter in the `preprocess` method.
* **tesseract\_config** (`str`, *optional*) —
  Any additional custom configuration flags that are forwarded to the `config` parameter when calling
  Tesseract. For example: ‘—psm 6’. Can be overridden by the `tesseract_config` parameter in the
  `preprocess` method.

Returns

`<class 'transformers.image_processing_base.BatchFeature'>`

* **data** (`dict`) — Dictionary of lists/arrays/tensors returned by the **call** method (‘pixel\_values’, etc.).
* **tensor\_type** (`Union[None, str, TensorType]`, *optional*) — You can give a tensor\_type here to convert the lists of integers in PyTorch/TensorFlow/Numpy Tensors at
  initialization.

## LayoutLMv3Tokenizer

### class transformers.LayoutLMv3Tokenizer

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/layoutlmv3/tokenization_layoutlmv3.py#L184)

( vocab\_file merges\_file errors = 'replace' bos\_token = '<s>' eos\_token = '</s>' sep\_token = '</s>' cls\_token = '<s>' unk\_token = '<unk>' pad\_token = '<pad>' mask\_token = '<mask>' add\_prefix\_space = True cls\_token\_box = [0, 0, 0, 0] sep\_token\_box = [0, 0, 0, 0] pad\_token\_box = [0, 0, 0, 0] pad\_token\_label = -100 only\_label\_first\_subword = True \*\*kwargs  )

Parameters

* **vocab\_file** (`str`) —
  Path to the vocabulary file.
* **merges\_file** (`str`) —
  Path to the merges file.
* **errors** (`str`, *optional*, defaults to `"replace"`) —
  Paradigm to follow when decoding bytes to UTF-8. See
  [bytes.decode](https://docs.python.org/3/library/stdtypes.html#bytes.decode) for more information.
* **bos\_token** (`str`, *optional*, defaults to `"<s>"`) —
  The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.

  When building a sequence using special tokens, this is not the token that is used for the beginning of
  sequence. The token used is the `cls_token`.
* **eos\_token** (`str`, *optional*, defaults to `"</s>"`) —
  The end of sequence token.

  When building a sequence using special tokens, this is not the token that is used for the end of sequence.
  The token used is the `sep_token`.
* **sep\_token** (`str`, *optional*, defaults to `"</s>"`) —
  The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
  sequence classification or for a text and a question for question answering. It is also used as the last
  token of a sequence built with special tokens.
* **cls\_token** (`str`, *optional*, defaults to `"<s>"`) —
  The classifier token which is used when doing sequence classification (classification of the whole sequence
  instead of per-token classification). It is the first token of the sequence when built with special tokens.
* **unk\_token** (`str`, *optional*, defaults to `"<unk>"`) —
  The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
  token instead.
* **pad\_token** (`str`, *optional*, defaults to `"<pad>"`) —
  The token used for padding, for example when batching sequences of different lengths.
* **mask\_token** (`str`, *optional*, defaults to `"<mask>"`) —
  The token used for masking values. This is the token used when training this model with masked language
  modeling. This is the token which the model will try to predict.
* **add\_prefix\_space** (`bool`, *optional*, defaults to `True`) —
  Whether or not to add an initial space to the input. This allows to treat the leading word just as any
  other word. (RoBERTa tokenizer detect beginning of words by the preceding space).
* **cls\_token\_box** (`list[int]`, *optional*, defaults to `[0, 0, 0, 0]`) —
  The bounding box to use for the special [CLS] token.
* **sep\_token\_box** (`list[int]`, *optional*, defaults to `[0, 0, 0, 0]`) —
  The bounding box to use for the special [SEP] token.
* **pad\_token\_box** (`list[int]`, *optional*, defaults to `[0, 0, 0, 0]`) —
  The bounding box to use for the special [PAD] token.
* **pad\_token\_label** (`int`, *optional*, defaults to -100) —
  The label to use for padding tokens. Defaults to -100, which is the `ignore_index` of PyTorch’s
  CrossEntropyLoss.
* **only\_label\_first\_subword** (`bool`, *optional*, defaults to `True`) —
  Whether or not to only label the first subword, in case word labels are provided.

Construct a LayoutLMv3 tokenizer. Based on `RoBERTatokenizer` (Byte Pair Encoding or BPE).
[LayoutLMv3Tokenizer](/docs/transformers/v4.56.2/en/model_doc/layoutlmv3#transformers.LayoutLMv3Tokenizer) can be used to turn words, word-level bounding boxes and optional word labels to
token-level `input_ids`, `attention_mask`, `token_type_ids`, `bbox`, and optional `labels` (for token
classification).

This tokenizer inherits from [PreTrainedTokenizer](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer) which contains most of the main methods. Users should refer to
this superclass for more information regarding those methods.

[LayoutLMv3Tokenizer](/docs/transformers/v4.56.2/en/model_doc/layoutlmv3#transformers.LayoutLMv3Tokenizer) runs end-to-end tokenization: punctuation splitting and wordpiece. It also turns the
word-level bounding boxes into token-level bounding boxes.

#### \_\_call\_\_

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/layoutlmv3/tokenization_layoutlmv3.py#L532)

( text: typing.Union[str, list[str], list[list[str]]] text\_pair: typing.Union[list[str], list[list[str]], NoneType] = None boxes: typing.Union[list[list[int]], list[list[list[int]]], NoneType] = None word\_labels: typing.Union[list[int], list[list[int]], NoneType] = None add\_special\_tokens: bool = True padding: typing.Union[bool, str, transformers.utils.generic.PaddingStrategy] = False truncation: typing.Union[bool, str, transformers.tokenization\_utils\_base.TruncationStrategy] = None max\_length: typing.Optional[int] = None stride: int = 0 pad\_to\_multiple\_of: typing.Optional[int] = None padding\_side: typing.Optional[str] = None return\_tensors: typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None return\_token\_type\_ids: typing.Optional[bool] = None return\_attention\_mask: typing.Optional[bool] = None return\_overflowing\_tokens: bool = False return\_special\_tokens\_mask: bool = False return\_offsets\_mapping: bool = False return\_length: bool = False verbose: bool = True \*\*kwargs  )

Parameters

* **text** (`str`, `List[str]`, `List[List[str]]`) —
  The sequence or batch of sequences to be encoded. Each sequence can be a string, a list of strings
  (words of a single example or questions of a batch of examples) or a list of list of strings (batch of
  words).
* **text\_pair** (`List[str]`, `List[List[str]]`) —
  The sequence or batch of sequences to be encoded. Each sequence should be a list of strings
  (pretokenized string).
* **boxes** (`List[List[int]]`, `List[List[List[int]]]`) —
  Word-level bounding boxes. Each bounding box should be normalized to be on a 0-1000 scale.
* **word\_labels** (`List[int]`, `List[List[int]]`, *optional*) —
  Word-level integer labels (for token classification tasks such as FUNSD, CORD).
* **add\_special\_tokens** (`bool`, *optional*, defaults to `True`) —
  Whether or not to encode the sequences with the special tokens relative to their model.
* **padding** (`bool`, `str` or [PaddingStrategy](/docs/transformers/v4.56.2/en/internal/file_utils#transformers.utils.PaddingStrategy), *optional*, defaults to `False`) —
  Activates and controls padding. Accepts the following values:
  + `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
    sequence if provided).
  + `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
    acceptable input length for the model if that argument is not provided.
  + `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
    lengths).
* **truncation** (`bool`, `str` or [TruncationStrategy](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.tokenization_utils_base.TruncationStrategy), *optional*, defaults to `False`) —
  Activates and controls truncation. Accepts the following values:
  + `True` or `'longest_first'`: Truncate to a maximum length specified with the argument `max_length` or
    to the maximum acceptable input length for the model if that argument is not provided. This will
    truncate token by token, removing a token from the longest sequence in the pair if a pair of
    sequences (or a batch of pairs) is provided.
  + `'only_first'`: Truncate to a maximum length specified with the argument `max_length` or to the
    maximum acceptable input length for the model if that argument is not provided. This will only
    truncate the first sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
  + `'only_second'`: Truncate to a maximum length specified with the argument `max_length` or to the
    maximum acceptable input length for the model if that argument is not provided. This will only
    truncate the second sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
  + `False` or `'do_not_truncate'` (default): No truncation (i.e., can output batch with sequence lengths
    greater than the model maximum admissible input size).
* **max\_length** (`int`, *optional*) —
  Controls the maximum length to use by one of the truncation/padding parameters.

  If left unset or set to `None`, this will use the predefined model maximum length if a maximum length
  is required by one of the truncation/padding parameters. If the model has no specific maximum input
  length (like XLNet) truncation/padding to a maximum length will be deactivated.
* **stride** (`int`, *optional*, defaults to 0) —
  If set to a number along with `max_length`, the overflowing tokens returned when
  `return_overflowing_tokens=True` will contain some tokens from the end of the truncated sequence
  returned to provide some overlap between truncated and overflowing sequences. The value of this
  argument defines the number of overlapping tokens.
* **pad\_to\_multiple\_of** (`int`, *optional*) —
  If set will pad the sequence to a multiple of the provided value. This is especially useful to enable
  the use of Tensor Cores on NVIDIA hardware with compute capability `>= 7.5` (Volta).
* **return\_tensors** (`str` or [TensorType](/docs/transformers/v4.56.2/en/internal/file_utils#transformers.TensorType), *optional*) —
  If set, will return tensors instead of list of python integers. Acceptable values are:
  + `'tf'`: Return TensorFlow `tf.constant` objects.
  + `'pt'`: Return PyTorch `torch.Tensor` objects.
  + `'np'`: Return Numpy `np.ndarray` objects.
* **add\_special\_tokens** (`bool`, *optional*, defaults to `True`) —
  Whether or not to encode the sequences with the special tokens relative to their model.
* **padding** (`bool`, `str` or [PaddingStrategy](/docs/transformers/v4.56.2/en/internal/file_utils#transformers.utils.PaddingStrategy), *optional*, defaults to `False`) —
  Activates and controls padding. Accepts the following values:
  + `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
    sequence if provided).
  + `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
    acceptable input length for the model if that argument is not provided.
  + `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
    lengths).
* **truncation** (`bool`, `str` or [TruncationStrategy](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.tokenization_utils_base.TruncationStrategy), *optional*, defaults to `False`) —
  Activates and controls truncation. Accepts the following values:
  + `True` or `'longest_first'`: Truncate to a maximum length specified with the argument `max_length` or
    to the maximum acceptable input length for the model if that argument is not provided. This will
    truncate token by token, removing a token from the longest sequence in the pair if a pair of
    sequences (or a batch of pairs) is provided.
  + `'only_first'`: Truncate to a maximum length specified with the argument `max_length` or to the
    maximum acceptable input length for the model if that argument is not provided. This will only
    truncate the first sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
  + `'only_second'`: Truncate to a maximum length specified with the argument `max_length` or to the
    maximum acceptable input length for the model if that argument is not provided. This will only
    truncate the second sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
  + `False` or `'do_not_truncate'` (default): No truncation (i.e., can output batch with sequence lengths
    greater than the model maximum admissible input size).
* **max\_length** (`int`, *optional*) —
  Controls the maximum length to use by one of the truncation/padding parameters. If left unset or set to
  `None`, this will use the predefined model maximum length if a maximum length is required by one of the
  truncation/padding parameters. If the model has no specific maximum input length (like XLNet)
  truncation/padding to a maximum length will be deactivated.
* **stride** (`int`, *optional*, defaults to 0) —
  If set to a number along with `max_length`, the overflowing tokens returned when
  `return_overflowing_tokens=True` will contain some tokens from the end of the truncated sequence
  returned to provide some overlap between truncated and overflowing sequences. The value of this
  argument defines the number of overlapping tokens.
* **pad\_to\_multiple\_of** (`int`, *optional*) —
  If set will pad the sequence to a multiple of the provided value. This is especially useful to enable
  the use of Tensor Cores on NVIDIA hardware with compute capability `>= 7.5` (Volta).
* **return\_tensors** (`str` or [TensorType](/docs/transformers/v4.56.2/en/internal/file_utils#transformers.TensorType), *optional*) —
  If set, will return tensors instead of list of python integers. Acceptable values are:
  + `'tf'`: Return TensorFlow `tf.constant` objects.
  + `'pt'`: Return PyTorch `torch.Tensor` objects.
  + `'np'`: Return Numpy `np.ndarray` objects.

Main method to tokenize and prepare for the model one or several sequence(s) or one or several pair(s) of
sequences with word-level normalized bounding boxes and optional labels.

#### save\_vocabulary

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/layoutlmv3/tokenization_layoutlmv3.py#L413)

( save\_directory: str filename\_prefix: typing.Optional[str] = None  )

## LayoutLMv3TokenizerFast

### class transformers.LayoutLMv3TokenizerFast

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/layoutlmv3/tokenization_layoutlmv3_fast.py#L49)

( vocab\_file = None merges\_file = None tokenizer\_file = None errors = 'replace' bos\_token = '<s>' eos\_token = '</s>' sep\_token = '</s>' cls\_token = '<s>' unk\_token = '<unk>' pad\_token = '<pad>' mask\_token = '<mask>' add\_prefix\_space = True trim\_offsets = True cls\_token\_box = [0, 0, 0, 0] sep\_token\_box = [0, 0, 0, 0] pad\_token\_box = [0, 0, 0, 0] pad\_token\_label = -100 only\_label\_first\_subword = True \*\*kwargs  )

Parameters

* **vocab\_file** (`str`) —
  Path to the vocabulary file.
* **merges\_file** (`str`) —
  Path to the merges file.
* **errors** (`str`, *optional*, defaults to `"replace"`) —
  Paradigm to follow when decoding bytes to UTF-8. See
  [bytes.decode](https://docs.python.org/3/library/stdtypes.html#bytes.decode) for more information.
* **bos\_token** (`str`, *optional*, defaults to `"<s>"`) —
  The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.

  When building a sequence using special tokens, this is not the token that is used for the beginning of
  sequence. The token used is the `cls_token`.
* **eos\_token** (`str`, *optional*, defaults to `"</s>"`) —
  The end of sequence token.

  When building a sequence using special tokens, this is not the token that is used for the end of sequence.
  The token used is the `sep_token`.
* **sep\_token** (`str`, *optional*, defaults to `"</s>"`) —
  The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
  sequence classification or for a text and a question for question answering. It is also used as the last
  token of a sequence built with special tokens.
* **cls\_token** (`str`, *optional*, defaults to `"<s>"`) —
  The classifier token which is used when doing sequence classification (classification of the whole sequence
  instead of per-token classification). It is the first token of the sequence when built with special tokens.
* **unk\_token** (`str`, *optional*, defaults to `"<unk>"`) —
  The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
  token instead.
* **pad\_token** (`str`, *optional*, defaults to `"<pad>"`) —
  The token used for padding, for example when batching sequences of different lengths.
* **mask\_token** (`str`, *optional*, defaults to `"<mask>"`) —
  The token used for masking values. This is the token used when training this model with masked language
  modeling. This is the token which the model will try to predict.
* **add\_prefix\_space** (`bool`, *optional*, defaults to `False`) —
  Whether or not to add an initial space to the input. This allows to treat the leading word just as any
  other word. (RoBERTa tokenizer detect beginning of words by the preceding space).
* **trim\_offsets** (`bool`, *optional*, defaults to `True`) —
  Whether the post processing step should trim offsets to avoid including whitespaces.
* **cls\_token\_box** (`list[int]`, *optional*, defaults to `[0, 0, 0, 0]`) —
  The bounding box to use for the special [CLS] token.
* **sep\_token\_box** (`list[int]`, *optional*, defaults to `[0, 0, 0, 0]`) —
  The bounding box to use for the special [SEP] token.
* **pad\_token\_box** (`list[int]`, *optional*, defaults to `[0, 0, 0, 0]`) —
  The bounding box to use for the special [PAD] token.
* **pad\_token\_label** (`int`, *optional*, defaults to -100) —
  The label to use for padding tokens. Defaults to -100, which is the `ignore_index` of PyTorch’s
  CrossEntropyLoss.
* **only\_label\_first\_subword** (`bool`, *optional*, defaults to `True`) —
  Whether or not to only label the first subword, in case word labels are provided.

Construct a “fast” LayoutLMv3 tokenizer (backed by HuggingFace’s *tokenizers* library). Based on BPE.

This tokenizer inherits from [PreTrainedTokenizerFast](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast) which contains most of the main methods. Users should
refer to this superclass for more information regarding those methods.

#### \_\_call\_\_

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/layoutlmv3/tokenization_layoutlmv3_fast.py#L198)

( text: typing.Union[str, list[str], list[list[str]]] text\_pair: typing.Union[list[str], list[list[str]], NoneType] = None boxes: typing.Union[list[list[int]], list[list[list[int]]], NoneType] = None word\_labels: typing.Union[list[int], list[list[int]], NoneType] = None add\_special\_tokens: bool = True padding: typing.Union[bool, str, transformers.utils.generic.PaddingStrategy] = False truncation: typing.Union[bool, str, transformers.tokenization\_utils\_base.TruncationStrategy] = None max\_length: typing.Optional[int] = None stride: int = 0 pad\_to\_multiple\_of: typing.Optional[int] = None padding\_side: typing.Optional[str] = None return\_tensors: typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None return\_token\_type\_ids: typing.Optional[bool] = None return\_attention\_mask: typing.Optional[bool] = None return\_overflowing\_tokens: bool = False return\_special\_tokens\_mask: bool = False return\_offsets\_mapping: bool = False return\_length: bool = False verbose: bool = True \*\*kwargs  )

Parameters

* **text** (`str`, `List[str]`, `List[List[str]]`) —
  The sequence or batch of sequences to be encoded. Each sequence can be a string, a list of strings
  (words of a single example or questions of a batch of examples) or a list of list of strings (batch of
  words).
* **text\_pair** (`List[str]`, `List[List[str]]`) —
  The sequence or batch of sequences to be encoded. Each sequence should be a list of strings
  (pretokenized string).
* **boxes** (`List[List[int]]`, `List[List[List[int]]]`) —
  Word-level bounding boxes. Each bounding box should be normalized to be on a 0-1000 scale.
* **word\_labels** (`List[int]`, `List[List[int]]`, *optional*) —
  Word-level integer labels (for token classification tasks such as FUNSD, CORD).
* **add\_special\_tokens** (`bool`, *optional*, defaults to `True`) —
  Whether or not to encode the sequences with the special tokens relative to their model.
* **padding** (`bool`, `str` or [PaddingStrategy](/docs/transformers/v4.56.2/en/internal/file_utils#transformers.utils.PaddingStrategy), *optional*, defaults to `False`) —
  Activates and controls padding. Accepts the following values:
  + `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
    sequence if provided).
  + `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
    acceptable input length for the model if that argument is not provided.
  + `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
    lengths).
* **truncation** (`bool`, `str` or [TruncationStrategy](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.tokenization_utils_base.TruncationStrategy), *optional*, defaults to `False`) —
  Activates and controls truncation. Accepts the following values:
  + `True` or `'longest_first'`: Truncate to a maximum length specified with the argument `max_length` or
    to the maximum acceptable input length for the model if that argument is not provided. This will
    truncate token by token, removing a token from the longest sequence in the pair if a pair of
    sequences (or a batch of pairs) is provided.
  + `'only_first'`: Truncate to a maximum length specified with the argument `max_length` or to the
    maximum acceptable input length for the model if that argument is not provided. This will only
    truncate the first sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
  + `'only_second'`: Truncate to a maximum length specified with the argument `max_length` or to the
    maximum acceptable input length for the model if that argument is not provided. This will only
    truncate the second sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
  + `False` or `'do_not_truncate'` (default): No truncation (i.e., can output batch with sequence lengths
    greater than the model maximum admissible input size).
* **max\_length** (`int`, *optional*) —
  Controls the maximum length to use by one of the truncation/padding parameters.

  If left unset or set to `None`, this will use the predefined model maximum length if a maximum length
  is required by one of the truncation/padding parameters. If the model has no specific maximum input
  length (like XLNet) truncation/padding to a maximum length will be deactivated.
* **stride** (`int`, *optional*, defaults to 0) —
  If set to a number along with `max_length`, the overflowing tokens returned when
  `return_overflowing_tokens=True` will contain some tokens from the end of the truncated sequence
  returned to provide some overlap between truncated and overflowing sequences. The value of this
  argument defines the number of overlapping tokens.
* **pad\_to\_multiple\_of** (`int`, *optional*) —
  If set will pad the sequence to a multiple of the provided value. This is especially useful to enable
  the use of Tensor Cores on NVIDIA hardware with compute capability `>= 7.5` (Volta).
* **return\_tensors** (`str` or [TensorType](/docs/transformers/v4.56.2/en/internal/file_utils#transformers.TensorType), *optional*) —
  If set, will return tensors instead of list of python integers. Acceptable values are:
  + `'tf'`: Return TensorFlow `tf.constant` objects.
  + `'pt'`: Return PyTorch `torch.Tensor` objects.
  + `'np'`: Return Numpy `np.ndarray` objects.
* **add\_special\_tokens** (`bool`, *optional*, defaults to `True`) —
  Whether or not to encode the sequences with the special tokens relative to their model.
* **padding** (`bool`, `str` or [PaddingStrategy](/docs/transformers/v4.56.2/en/internal/file_utils#transformers.utils.PaddingStrategy), *optional*, defaults to `False`) —
  Activates and controls padding. Accepts the following values:
  + `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
    sequence if provided).
  + `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
    acceptable input length for the model if that argument is not provided.
  + `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
    lengths).
* **truncation** (`bool`, `str` or [TruncationStrategy](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.tokenization_utils_base.TruncationStrategy), *optional*, defaults to `False`) —
  Activates and controls truncation. Accepts the following values:
  + `True` or `'longest_first'`: Truncate to a maximum length specified with the argument `max_length` or
    to the maximum acceptable input length for the model if that argument is not provided. This will
    truncate token by token, removing a token from the longest sequence in the pair if a pair of
    sequences (or a batch of pairs) is provided.
  + `'only_first'`: Truncate to a maximum length specified with the argument `max_length` or to the
    maximum acceptable input length for the model if that argument is not provided. This will only
    truncate the first sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
  + `'only_second'`: Truncate to a maximum length specified with the argument `max_length` or to the
    maximum acceptable input length for the model if that argument is not provided. This will only
    truncate the second sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
  + `False` or `'do_not_truncate'` (default): No truncation (i.e., can output batch with sequence lengths
    greater than the model maximum admissible input size).
* **max\_length** (`int`, *optional*) —
  Controls the maximum length to use by one of the truncation/padding parameters. If left unset or set to
  `None`, this will use the predefined model maximum length if a maximum length is required by one of the
  truncation/padding parameters. If the model has no specific maximum input length (like XLNet)
  truncation/padding to a maximum length will be deactivated.
* **stride** (`int`, *optional*, defaults to 0) —
  If set to a number along with `max_length`, the overflowing tokens returned when
  `return_overflowing_tokens=True` will contain some tokens from the end of the truncated sequence
  returned to provide some overlap between truncated and overflowing sequences. The value of this
  argument defines the number of overlapping tokens.
* **pad\_to\_multiple\_of** (`int`, *optional*) —
  If set will pad the sequence to a multiple of the provided value. This is especially useful to enable
  the use of Tensor Cores on NVIDIA hardware with compute capability `>= 7.5` (Volta).
* **return\_tensors** (`str` or [TensorType](/docs/transformers/v4.56.2/en/internal/file_utils#transformers.TensorType), *optional*) —
  If set, will return tensors instead of list of python integers. Acceptable values are:
  + `'tf'`: Return TensorFlow `tf.constant` objects.
  + `'pt'`: Return PyTorch `torch.Tensor` objects.
  + `'np'`: Return Numpy `np.ndarray` objects.

Main method to tokenize and prepare for the model one or several sequence(s) or one or several pair(s) of
sequences with word-level normalized bounding boxes and optional labels.

## LayoutLMv3Processor

### class transformers.LayoutLMv3Processor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/layoutlmv3/processing_layoutlmv3.py#L27)

( image\_processor = None tokenizer = None \*\*kwargs  )

Parameters

* **image\_processor** (`LayoutLMv3ImageProcessor`, *optional*) —
  An instance of [LayoutLMv3ImageProcessor](/docs/transformers/v4.56.2/en/model_doc/layoutlmv3#transformers.LayoutLMv3ImageProcessor). The image processor is a required input.
* **tokenizer** (`LayoutLMv3Tokenizer` or `LayoutLMv3TokenizerFast`, *optional*) —
  An instance of [LayoutLMv3Tokenizer](/docs/transformers/v4.56.2/en/model_doc/layoutlmv3#transformers.LayoutLMv3Tokenizer) or [LayoutLMv3TokenizerFast](/docs/transformers/v4.56.2/en/model_doc/layoutlmv3#transformers.LayoutLMv3TokenizerFast). The tokenizer is a required input.

Constructs a LayoutLMv3 processor which combines a LayoutLMv3 image processor and a LayoutLMv3 tokenizer into a
single processor.

[LayoutLMv3Processor](/docs/transformers/v4.56.2/en/model_doc/layoutlmv3#transformers.LayoutLMv3Processor) offers all the functionalities you need to prepare data for the model.

It first uses [LayoutLMv3ImageProcessor](/docs/transformers/v4.56.2/en/model_doc/layoutlmv3#transformers.LayoutLMv3ImageProcessor) to resize and normalize document images, and optionally applies OCR to
get words and normalized bounding boxes. These are then provided to [LayoutLMv3Tokenizer](/docs/transformers/v4.56.2/en/model_doc/layoutlmv3#transformers.LayoutLMv3Tokenizer) or
[LayoutLMv3TokenizerFast](/docs/transformers/v4.56.2/en/model_doc/layoutlmv3#transformers.LayoutLMv3TokenizerFast), which turns the words and bounding boxes into token-level `input_ids`,
`attention_mask`, `token_type_ids`, `bbox`. Optionally, one can provide integer `word_labels`, which are turned
into token-level `labels` for token classification tasks (such as FUNSD, CORD).

#### \_\_call\_\_

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/layoutlmv3/processing_layoutlmv3.py#L69)

( images text: typing.Union[str, list[str], list[list[str]]] = None text\_pair: typing.Union[list[str], list[list[str]], NoneType] = None boxes: typing.Union[list[list[int]], list[list[list[int]]], NoneType] = None word\_labels: typing.Union[list[int], list[list[int]], NoneType] = None add\_special\_tokens: bool = True padding: typing.Union[bool, str, transformers.utils.generic.PaddingStrategy] = False truncation: typing.Union[bool, str, transformers.tokenization\_utils\_base.TruncationStrategy] = None max\_length: typing.Optional[int] = None stride: int = 0 pad\_to\_multiple\_of: typing.Optional[int] = None return\_token\_type\_ids: typing.Optional[bool] = None return\_attention\_mask: typing.Optional[bool] = None return\_overflowing\_tokens: bool = False return\_special\_tokens\_mask: bool = False return\_offsets\_mapping: bool = False return\_length: bool = False verbose: bool = True return\_tensors: typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None \*\*kwargs  )

This method first forwards the `images` argument to [**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__). In case
[LayoutLMv3ImageProcessor](/docs/transformers/v4.56.2/en/model_doc/layoutlmv3#transformers.LayoutLMv3ImageProcessor) was initialized with `apply_ocr` set to `True`, it passes the obtained words and
bounding boxes along with the additional arguments to [**call**()](/docs/transformers/v4.56.2/en/model_doc/layoutlmv3#transformers.LayoutLMv3Tokenizer.__call__) and returns the output,
together with resized and normalized `pixel_values`. In case [LayoutLMv3ImageProcessor](/docs/transformers/v4.56.2/en/model_doc/layoutlmv3#transformers.LayoutLMv3ImageProcessor) was initialized with
`apply_ocr` set to `False`, it passes the words (`text`/``text_pair`) and `boxes` specified by the user along
with the additional arguments to [**call**()](/docs/transformers/v4.56.2/en/model_doc/layoutlmv3#transformers.LayoutLMv3Tokenizer.__call__) and returns the output, together with
resized and normalized `pixel_values`.

Please refer to the docstring of the above two methods for more information.

## LayoutLMv3Model

### class transformers.LayoutLMv3Model

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/layoutlmv3/modeling_layoutlmv3.py#L589)

( config  )

Parameters

* **config** ([LayoutLMv3Model](/docs/transformers/v4.56.2/en/model_doc/layoutlmv3#transformers.LayoutLMv3Model)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Layoutlmv3 Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/layoutlmv3/modeling_layoutlmv3.py#L678)

( input\_ids: typing.Optional[torch.LongTensor] = None bbox: typing.Optional[torch.LongTensor] = None attention\_mask: typing.Optional[torch.FloatTensor] = None token\_type\_ids: typing.Optional[torch.LongTensor] = None position\_ids: typing.Optional[torch.LongTensor] = None head\_mask: typing.Optional[torch.FloatTensor] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None pixel\_values: typing.Optional[torch.FloatTensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → [transformers.modeling\_outputs.BaseModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutput) or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, token_sequence_length)`) —
  Indices of input sequence tokens in the vocabulary.

  Note that `sequence_length = token_sequence_length + patch_sequence_length + 1` where `1` is for [CLS]
  token. See `pixel_values` for `patch_sequence_length`.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **bbox** (`torch.LongTensor` of shape `(batch_size, token_sequence_length, 4)`, *optional*) —
  Bounding boxes of each input sequence tokens. Selected in the range `[0, config.max_2d_position_embeddings-1]`. Each bounding box should be a normalized version in (x0, y0, x1, y1)
  format, where (x0, y0) corresponds to the position of the upper left corner in the bounding box, and (x1,
  y1) represents the position of the lower right corner.

  Note that `sequence_length = token_sequence_length + patch_sequence_length + 1` where `1` is for [CLS]
  token. See `pixel_values` for `patch_sequence_length`.
* **attention\_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **token\_type\_ids** (`torch.LongTensor` of shape `(batch_size, token_sequence_length)`, *optional*) —
  Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
  + 0 corresponds to a *sentence A* token,
  + 1 corresponds to a *sentence B* token.

  Note that `sequence_length = token_sequence_length + patch_sequence_length + 1` where `1` is for [CLS]
  token. See `pixel_values` for `patch_sequence_length`.

  [What are token type IDs?](../glossary#token-type-ids)
* **position\_ids** (`torch.LongTensor` of shape `(batch_size, token_sequence_length)`, *optional*) —
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.max_position_embeddings - 1]`.

  Note that `sequence_length = token_sequence_length + patch_sequence_length + 1` where `1` is for [CLS]
  token. See `pixel_values` for `patch_sequence_length`.

  [What are position IDs?](../glossary#position-ids)
* **head\_mask** (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, token_sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert *input\_ids* indices into associated vectors than the
  model’s internal embedding lookup matrix.
* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [LayoutLMv3ImageProcessor](/docs/transformers/v4.56.2/en/model_doc/layoutlmv3#transformers.LayoutLMv3ImageProcessor). See [LayoutLMv3ImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([LayoutLMv3Processor](/docs/transformers/v4.56.2/en/model_doc/layoutlmv3#transformers.LayoutLMv3Processor) uses
  [LayoutLMv3ImageProcessor](/docs/transformers/v4.56.2/en/model_doc/layoutlmv3#transformers.LayoutLMv3ImageProcessor) for processing images).
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

[transformers.modeling\_outputs.BaseModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutput) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.BaseModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([LayoutLMv3Config](/docs/transformers/v4.56.2/en/model_doc/layoutlmv3#transformers.LayoutLMv3Config)) and inputs.

* **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the model.
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [LayoutLMv3Model](/docs/transformers/v4.56.2/en/model_doc/layoutlmv3#transformers.LayoutLMv3Model) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
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

## LayoutLMv3ForSequenceClassification

### class transformers.LayoutLMv3ForSequenceClassification

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/layoutlmv3/modeling_layoutlmv3.py#L1138)

( config  )

Parameters

* **config** ([LayoutLMv3ForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/layoutlmv3#transformers.LayoutLMv3ForSequenceClassification)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

LayoutLMv3 Model with a sequence classification head on top (a linear layer on top of the final hidden state of the
[CLS] token) e.g. for document image classification tasks such as the
[RVL-CDIP](https://www.cs.cmu.edu/~aharley/rvl-cdip/) dataset.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/layoutlmv3/modeling_layoutlmv3.py#L1148)

( input\_ids: typing.Optional[torch.LongTensor] = None attention\_mask: typing.Optional[torch.FloatTensor] = None token\_type\_ids: typing.Optional[torch.LongTensor] = None position\_ids: typing.Optional[torch.LongTensor] = None head\_mask: typing.Optional[torch.FloatTensor] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None labels: typing.Optional[torch.LongTensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None bbox: typing.Optional[torch.LongTensor] = None pixel\_values: typing.Optional[torch.LongTensor] = None  ) → [transformers.modeling\_outputs.SequenceClassifierOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput) or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **attention\_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **token\_type\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
  + 0 corresponds to a *sentence A* token,
  + 1 corresponds to a *sentence B* token.

  [What are token type IDs?](../glossary#token-type-ids)
* **position\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
* **head\_mask** (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model’s internal embedding lookup matrix.
* **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Labels for computing the masked language modeling loss. Indices should either be in `[0, ..., config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
  (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
* **bbox** (`torch.LongTensor` of shape `(batch_size, sequence_length, 4)`, *optional*) —
  Bounding boxes of each input sequence tokens. Selected in the range `[0, config.max_2d_position_embeddings-1]`. Each bounding box should be a normalized version in (x0, y0, x1, y1)
  format, where (x0, y0) corresponds to the position of the upper left corner in the bounding box, and (x1,
  y1) represents the position of the lower right corner.
* **pixel\_values** (`torch.LongTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [LayoutLMv3ImageProcessor](/docs/transformers/v4.56.2/en/model_doc/layoutlmv3#transformers.LayoutLMv3ImageProcessor). See [LayoutLMv3ImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([LayoutLMv3Processor](/docs/transformers/v4.56.2/en/model_doc/layoutlmv3#transformers.LayoutLMv3Processor) uses
  [LayoutLMv3ImageProcessor](/docs/transformers/v4.56.2/en/model_doc/layoutlmv3#transformers.LayoutLMv3ImageProcessor) for processing images).

Returns

[transformers.modeling\_outputs.SequenceClassifierOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.SequenceClassifierOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([LayoutLMv3Config](/docs/transformers/v4.56.2/en/model_doc/layoutlmv3#transformers.LayoutLMv3Config)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Classification (or regression if config.num\_labels==1) loss.
* **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) — Classification (or regression if config.num\_labels==1) scores (before SoftMax).
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [LayoutLMv3ForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/layoutlmv3#transformers.LayoutLMv3ForSequenceClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
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

## LayoutLMv3ForTokenClassification

### class transformers.LayoutLMv3ForTokenClassification

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/layoutlmv3/modeling_layoutlmv3.py#L912)

( config  )

Parameters

* **config** ([LayoutLMv3ForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/layoutlmv3#transformers.LayoutLMv3ForTokenClassification)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

LayoutLMv3 Model with a token classification head on top (a linear layer on top of the final hidden states) e.g.
for sequence labeling (information extraction) tasks such as [FUNSD](https://guillaumejaume.github.io/FUNSD/),
[SROIE](https://rrc.cvc.uab.es/?ch=13), [CORD](https://github.com/clovaai/cord) and
[Kleister-NDA](https://github.com/applicaai/kleister-nda).

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/layoutlmv3/modeling_layoutlmv3.py#L926)

( input\_ids: typing.Optional[torch.LongTensor] = None bbox: typing.Optional[torch.LongTensor] = None attention\_mask: typing.Optional[torch.FloatTensor] = None token\_type\_ids: typing.Optional[torch.LongTensor] = None position\_ids: typing.Optional[torch.LongTensor] = None head\_mask: typing.Optional[torch.FloatTensor] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None labels: typing.Optional[torch.LongTensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None pixel\_values: typing.Optional[torch.LongTensor] = None  ) → [transformers.modeling\_outputs.TokenClassifierOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput) or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **bbox** (`torch.LongTensor` of shape `(batch_size, sequence_length, 4)`, *optional*) —
  Bounding boxes of each input sequence tokens. Selected in the range `[0, config.max_2d_position_embeddings-1]`. Each bounding box should be a normalized version in (x0, y0, x1, y1)
  format, where (x0, y0) corresponds to the position of the upper left corner in the bounding box, and (x1,
  y1) represents the position of the lower right corner.
* **attention\_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **token\_type\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
  + 0 corresponds to a *sentence A* token,
  + 1 corresponds to a *sentence B* token.

  [What are token type IDs?](../glossary#token-type-ids)
* **position\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
* **head\_mask** (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model’s internal embedding lookup matrix.
* **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
* **pixel\_values** (`torch.LongTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [LayoutLMv3ImageProcessor](/docs/transformers/v4.56.2/en/model_doc/layoutlmv3#transformers.LayoutLMv3ImageProcessor). See [LayoutLMv3ImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([LayoutLMv3Processor](/docs/transformers/v4.56.2/en/model_doc/layoutlmv3#transformers.LayoutLMv3Processor) uses
  [LayoutLMv3ImageProcessor](/docs/transformers/v4.56.2/en/model_doc/layoutlmv3#transformers.LayoutLMv3ImageProcessor) for processing images).

Returns

[transformers.modeling\_outputs.TokenClassifierOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.TokenClassifierOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([LayoutLMv3Config](/docs/transformers/v4.56.2/en/model_doc/layoutlmv3#transformers.LayoutLMv3Config)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Classification loss.
* **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_labels)`) — Classification scores (before SoftMax).
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [LayoutLMv3ForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/layoutlmv3#transformers.LayoutLMv3ForTokenClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
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

## LayoutLMv3ForQuestionAnswering

### class transformers.LayoutLMv3ForQuestionAnswering

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/layoutlmv3/modeling_layoutlmv3.py#L1017)

( config  )

Parameters

* **config** ([LayoutLMv3ForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/layoutlmv3#transformers.LayoutLMv3ForQuestionAnswering)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Layoutlmv3 transformer with a span classification head on top for extractive question-answering tasks like
SQuAD (a linear layer on top of the hidden-states output to compute `span start logits` and `span end logits`).

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/layoutlmv3/modeling_layoutlmv3.py#L1027)

( input\_ids: typing.Optional[torch.LongTensor] = None attention\_mask: typing.Optional[torch.FloatTensor] = None token\_type\_ids: typing.Optional[torch.LongTensor] = None position\_ids: typing.Optional[torch.LongTensor] = None head\_mask: typing.Optional[torch.FloatTensor] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None start\_positions: typing.Optional[torch.LongTensor] = None end\_positions: typing.Optional[torch.LongTensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None bbox: typing.Optional[torch.LongTensor] = None pixel\_values: typing.Optional[torch.LongTensor] = None  ) → [transformers.modeling\_outputs.QuestionAnsweringModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.QuestionAnsweringModelOutput) or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **attention\_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **token\_type\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
  + 0 corresponds to a *sentence A* token,
  + 1 corresponds to a *sentence B* token.

  [What are token type IDs?](../glossary#token-type-ids)
* **position\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
* **head\_mask** (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model’s internal embedding lookup matrix.
* **start\_positions** (`torch.LongTensor` of shape `(batch_size,)`, *optional*) —
  Labels for position (index) of the start of the labelled span for computing the token classification loss.
  Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
  are not taken into account for computing the loss.
* **end\_positions** (`torch.LongTensor` of shape `(batch_size,)`, *optional*) —
  Labels for position (index) of the end of the labelled span for computing the token classification loss.
  Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
  are not taken into account for computing the loss.
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
* **bbox** (`torch.LongTensor` of shape `(batch_size, sequence_length, 4)`, *optional*) —
  Bounding boxes of each input sequence tokens. Selected in the range `[0, config.max_2d_position_embeddings-1]`. Each bounding box should be a normalized version in (x0, y0, x1, y1)
  format, where (x0, y0) corresponds to the position of the upper left corner in the bounding box, and (x1,
  y1) represents the position of the lower right corner.
* **pixel\_values** (`torch.LongTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [LayoutLMv3ImageProcessor](/docs/transformers/v4.56.2/en/model_doc/layoutlmv3#transformers.LayoutLMv3ImageProcessor). See [LayoutLMv3ImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([LayoutLMv3Processor](/docs/transformers/v4.56.2/en/model_doc/layoutlmv3#transformers.LayoutLMv3Processor) uses
  [LayoutLMv3ImageProcessor](/docs/transformers/v4.56.2/en/model_doc/layoutlmv3#transformers.LayoutLMv3ImageProcessor) for processing images).

Returns

[transformers.modeling\_outputs.QuestionAnsweringModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.QuestionAnsweringModelOutput) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.QuestionAnsweringModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.QuestionAnsweringModelOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([LayoutLMv3Config](/docs/transformers/v4.56.2/en/model_doc/layoutlmv3#transformers.LayoutLMv3Config)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
* **start\_logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`) — Span-start scores (before SoftMax).
* **end\_logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`) — Span-end scores (before SoftMax).
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [LayoutLMv3ForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/layoutlmv3#transformers.LayoutLMv3ForQuestionAnswering) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
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

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/layoutlmv3.md)
