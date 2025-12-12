*This model was released on 2023-08-25 and added to Hugging Face Transformers on 2023-09-26.*

# Nougat

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white)

## Overview

The Nougat model was proposed in [Nougat: Neural Optical Understanding for Academic Documents](https://huggingface.co/papers/2308.13418) by
Lukas Blecher, Guillem Cucurull, Thomas Scialom, Robert Stojnic. Nougat uses the same architecture as [Donut](donut), meaning an image Transformer
encoder and an autoregressive text Transformer decoder to translate scientific PDFs to markdown, enabling easier access to them.

The abstract from the paper is the following:

*Scientific knowledge is predominantly stored in books and scientific journals, often in the form of PDFs. However, the PDF format leads to a loss of semantic information, particularly for mathematical expressions. We propose Nougat (Neural Optical Understanding for Academic Documents), a Visual Transformer model that performs an Optical Character Recognition (OCR) task for processing scientific documents into a markup language, and demonstrate the effectiveness of our model on a new dataset of scientific documents. The proposed approach offers a promising solution to enhance the accessibility of scientific knowledge in the digital age, by bridging the gap between human-readable documents and machine-readable text. We release the models and code to accelerate future work on scientific text recognition.*

![drawing](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/nougat_architecture.jpg) Nougat high-level overview. Taken from the [original paper](https://huggingface.co/papers/2308.13418).

This model was contributed by [nielsr](https://huggingface.co/nielsr). The original code can be found
[here](https://github.com/facebookresearch/nougat).

## Usage tips

* The quickest way to get started with Nougat is by checking the [tutorial
  notebooks](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/Nougat), which show how to use the model
  at inference time as well as fine-tuning on custom data.
* Nougat is always used within the [VisionEncoderDecoder](vision-encoder-decoder) framework. The model is identical to [Donut](donut) in terms of architecture.

## Inference

Nougat’s `VisionEncoderDecoder` model accepts images as input and makes use of
[generate()](/docs/transformers/v4.56.2/en/main_classes/text_generation#transformers.GenerationMixin.generate) to autoregressively generate text given the input image.

The [NougatImageProcessor](/docs/transformers/v4.56.2/en/model_doc/nougat#transformers.NougatImageProcessor) class is responsible for preprocessing the input image and
[NougatTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/nougat#transformers.NougatTokenizerFast) decodes the generated target tokens to the target string. The
[NougatProcessor](/docs/transformers/v4.56.2/en/model_doc/nougat#transformers.NougatProcessor) wraps [NougatImageProcessor](/docs/transformers/v4.56.2/en/model_doc/nougat#transformers.NougatImageProcessor) and [NougatTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/nougat#transformers.NougatTokenizerFast) classes
into a single instance to both extract the input features and decode the predicted token ids.

* Step-by-step PDF transcription


```
>>> from huggingface_hub import hf_hub_download
>>> import re
>>> from PIL import Image

>>> from transformers import NougatProcessor, VisionEncoderDecoderModel, infer_device
>>> from datasets import load_dataset
>>> import torch

>>> processor = NougatProcessor.from_pretrained("facebook/nougat-base")
>>> model = VisionEncoderDecoderModel.from_pretrained("facebook/nougat-base")

>>> device = infer_device()
>>> model.to(device)
>>> # prepare PDF image for the model
>>> filepath = hf_hub_download(repo_id="hf-internal-testing/fixtures_docvqa", filename="nougat_paper.png", repo_type="dataset")
>>> image = Image.open(filepath)
>>> pixel_values = processor(image, return_tensors="pt").pixel_values

>>> # generate transcription (here we only generate 30 tokens)
>>> outputs = model.generate(
...     pixel_values.to(device),
...     min_length=1,
...     max_new_tokens=30,
...     bad_words_ids=[[processor.tokenizer.unk_token_id]],
... )

>>> sequence = processor.batch_decode(outputs, skip_special_tokens=True)[0]
>>> sequence = processor.post_process_generation(sequence, fix_markdown=False)
>>> # note: we're using repr here such for the sake of printing the \n characters, feel free to just print the sequence
>>> print(repr(sequence))
'\n\n# Nougat: Neural Optical Understanding for Academic Documents\n\n Lukas Blecher\n\nCorrespondence to: lblecher@'
```

See the [model hub](https://huggingface.co/models?filter=nougat) to look for Nougat checkpoints.

The model is identical to [Donut](donut) in terms of architecture.

## NougatImageProcessor

### class transformers.NougatImageProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/nougat/image_processing_nougat.py#L54)

( do\_crop\_margin: bool = True do\_resize: bool = True size: typing.Optional[dict[str, int]] = None resample: Resampling = <Resampling.BILINEAR: 2> do\_thumbnail: bool = True do\_align\_long\_axis: bool = False do\_pad: bool = True do\_rescale: bool = True rescale\_factor: typing.Union[int, float] = 0.00392156862745098 do\_normalize: bool = True image\_mean: typing.Union[float, list[float], NoneType] = None image\_std: typing.Union[float, list[float], NoneType] = None \*\*kwargs  )

Parameters

* **do\_crop\_margin** (`bool`, *optional*, defaults to `True`) —
  Whether to crop the image margins.
* **do\_resize** (`bool`, *optional*, defaults to `True`) —
  Whether to resize the image’s (height, width) dimensions to the specified `size`. Can be overridden by
  `do_resize` in the `preprocess` method.
* **size** (`dict[str, int]` *optional*, defaults to `{"height" -- 896, "width": 672}`):
  Size of the image after resizing. Can be overridden by `size` in the `preprocess` method.
* **resample** (`PILImageResampling`, *optional*, defaults to `Resampling.BILINEAR`) —
  Resampling filter to use if resizing the image. Can be overridden by `resample` in the `preprocess` method.
* **do\_thumbnail** (`bool`, *optional*, defaults to `True`) —
  Whether to resize the image using thumbnail method.
* **do\_align\_long\_axis** (`bool`, *optional*, defaults to `False`) —
  Whether to align the long axis of the image with the long axis of `size` by rotating by 90 degrees.
* **do\_pad** (`bool`, *optional*, defaults to `True`) —
  Whether to pad the images to the largest image size in the batch.
* **do\_rescale** (`bool`, *optional*, defaults to `True`) —
  Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by the `do_rescale`
  parameter in the `preprocess` method.
* **rescale\_factor** (`int` or `float`, *optional*, defaults to `1/255`) —
  Scale factor to use if rescaling the image. Can be overridden by the `rescale_factor` parameter in the
  `preprocess` method.
* **do\_normalize** (`bool`, *optional*, defaults to `True`) —
  Whether to normalize the image. Can be overridden by `do_normalize` in the `preprocess` method.
* **image\_mean** (`float` or `list[float]`, *optional*, defaults to `IMAGENET_DEFAULT_MEAN`) —
  Mean to use if normalizing the image. This is a float or list of floats the length of the number of
  channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method.
* **image\_std** (`float` or `list[float]`, *optional*, defaults to `IMAGENET_DEFAULT_STD`) —
  Image standard deviation.

Constructs a Nougat image processor.

#### preprocess

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/nougat/image_processing_nougat.py#L367)

( images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']] do\_crop\_margin: typing.Optional[bool] = None do\_resize: typing.Optional[bool] = None size: typing.Optional[dict[str, int]] = None resample: Resampling = None do\_thumbnail: typing.Optional[bool] = None do\_align\_long\_axis: typing.Optional[bool] = None do\_pad: typing.Optional[bool] = None do\_rescale: typing.Optional[bool] = None rescale\_factor: typing.Union[int, float, NoneType] = None do\_normalize: typing.Optional[bool] = None image\_mean: typing.Union[float, list[float], NoneType] = None image\_std: typing.Union[float, list[float], NoneType] = None return\_tensors: typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None data\_format: typing.Optional[transformers.image\_utils.ChannelDimension] = <ChannelDimension.FIRST: 'channels\_first'> input\_data\_format: typing.Union[str, transformers.image\_utils.ChannelDimension, NoneType] = None  )

Parameters

* **images** (`ImageInput`) —
  Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255.
* **do\_crop\_margin** (`bool`, *optional*, defaults to `self.do_crop_margin`) —
  Whether to crop the image margins.
* **do\_resize** (`bool`, *optional*, defaults to `self.do_resize`) —
  Whether to resize the image.
* **size** (`dict[str, int]`, *optional*, defaults to `self.size`) —
  Size of the image after resizing. Shortest edge of the image is resized to min(size[“height”],
  size[“width”]) with the longest edge resized to keep the input aspect ratio.
* **resample** (`int`, *optional*, defaults to `self.resample`) —
  Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`. Only
  has an effect if `do_resize` is set to `True`.
* **do\_thumbnail** (`bool`, *optional*, defaults to `self.do_thumbnail`) —
  Whether to resize the image using thumbnail method.
* **do\_align\_long\_axis** (`bool`, *optional*, defaults to `self.do_align_long_axis`) —
  Whether to align the long axis of the image with the long axis of `size` by rotating by 90 degrees.
* **do\_pad** (`bool`, *optional*, defaults to `self.do_pad`) —
  Whether to pad the images to the largest image size in the batch.
* **do\_rescale** (`bool`, *optional*, defaults to `self.do_rescale`) —
  Whether to rescale the image by the specified scale `rescale_factor`.
* **rescale\_factor** (`int` or `float`, *optional*, defaults to `self.rescale_factor`) —
  Scale factor to use if rescaling the image.
* **do\_normalize** (`bool`, *optional*, defaults to `self.do_normalize`) —
  Whether to normalize the image.
* **image\_mean** (`float` or `list[float]`, *optional*, defaults to `self.image_mean`) —
  Image mean to use for normalization.
* **image\_std** (`float` or `list[float]`, *optional*, defaults to `self.image_std`) —
  Image standard deviation to use for normalization.
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
  + Unset: defaults to the channel dimension format of the input image.
* **input\_data\_format** (`ChannelDimension` or `str`, *optional*) —
  The channel dimension format for the input image. If unset, the channel dimension format is inferred
  from the input image. Can be one of:
  + `"channels_first"` or `ChannelDimension.FIRST`: image in (num\_channels, height, width) format.
  + `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num\_channels) format.
  + `"none"` or `ChannelDimension.NONE`: image in (height, width) format.

Preprocess an image or batch of images.

## NougatImageProcessorFast

### class transformers.NougatImageProcessorFast

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/nougat/image_processing_nougat_fast.py#L77)

( \*\*kwargs: typing\_extensions.Unpack[transformers.models.nougat.image\_processing\_nougat\_fast.NougatFastImageProcessorKwargs]  )

Constructs a fast Nougat image processor.

#### preprocess

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/nougat/image_processing_nougat_fast.py#L94)

( images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']] \*\*kwargs: typing\_extensions.Unpack[transformers.models.nougat.image\_processing\_nougat\_fast.NougatFastImageProcessorKwargs]  ) → `<class 'transformers.image_processing_base.BatchFeature'>`

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
* **do\_crop\_margin** (`bool`, *optional*, defaults to `True`) —
  Whether to crop the image margins.
* **do\_thumbnail** (`bool`, *optional*, defaults to `True`) —
  Whether to resize the image using thumbnail method.
* **do\_align\_long\_axis** (`bool`, *optional*, defaults to `False`) —
  Whether to align the long axis of the image with the long axis of `size` by rotating by 90 degrees.
* **do\_pad** (`bool`, *optional*, defaults to `True`) —
  Whether to pad the images to the largest image size in the batch.

Returns

`<class 'transformers.image_processing_base.BatchFeature'>`

* **data** (`dict`) — Dictionary of lists/arrays/tensors returned by the **call** method (‘pixel\_values’, etc.).
* **tensor\_type** (`Union[None, str, TensorType]`, *optional*) — You can give a tensor\_type here to convert the lists of integers in PyTorch/TensorFlow/Numpy Tensors at
  initialization.

## NougatTokenizerFast

### class transformers.NougatTokenizerFast

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/nougat/tokenization_nougat_fast.py#L362)

( vocab\_file = None tokenizer\_file = None clean\_up\_tokenization\_spaces = False unk\_token = '<unk>' bos\_token = '<s>' eos\_token = '</s>' pad\_token = '<pad>' \*\*kwargs  )

Parameters

* **vocab\_file** (`str`, *optional*) —
  [SentencePiece](https://github.com/google/sentencepiece) file (generally has a .model extension) that
  contains the vocabulary necessary to instantiate a tokenizer.
* **tokenizer\_file** (`str`, *optional*) —
  [tokenizers](https://github.com/huggingface/tokenizers) file (generally has a .json extension) that
  contains everything needed to load the tokenizer.
* **clean\_up\_tokenization\_spaces** (`str`, *optional*, defaults to `False`) —
  Whether to cleanup spaces after decoding, cleanup consists in removing potential artifacts like extra
  spaces.
* **unk\_token** (`str`, *optional*, defaults to `"<unk>"`) —
  The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
  token instead.
* **bos\_token** (`str`, *optional*, defaults to `"<s>"`) —
  The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.
* **eos\_token** (`str`, *optional*, defaults to `"</s>"`) —
  The end of sequence token.
* **pad\_token** (`str`, *optional*, defaults to `"<pad>"`) —
  The token used for padding, for example when batching sequences of different lengths.
* **model\_max\_length** (`int`, *optional*) —
  The maximum length (in number of tokens) for the inputs to the transformer model. When the tokenizer is
  loaded with [from\_pretrained()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.from_pretrained), this will be set to the
  value stored for the associated model in `max_model_input_sizes` (see above). If no value is provided, will
  default to VERY\_LARGE\_INTEGER (`int(1e30)`).
* **padding\_side** (`str`, *optional*) —
  The side on which the model should have padding applied. Should be selected between [‘right’, ‘left’].
  Default value is picked from the class attribute of the same name.
* **truncation\_side** (`str`, *optional*) —
  The side on which the model should have truncation applied. Should be selected between [‘right’, ‘left’].
  Default value is picked from the class attribute of the same name.
* **chat\_template** (`str`, *optional*) —
  A Jinja template string that will be used to format lists of chat messages. See
  <https://huggingface.co/docs/transformers/chat_templating> for a full description.
* **model\_input\_names** (`list[string]`, *optional*) —
  The list of inputs accepted by the forward pass of the model (like `"token_type_ids"` or
  `"attention_mask"`). Default value is picked from the class attribute of the same name.
* **bos\_token** (`str` or `tokenizers.AddedToken`, *optional*) —
  A special token representing the beginning of a sentence. Will be associated to `self.bos_token` and
  `self.bos_token_id`.
* **eos\_token** (`str` or `tokenizers.AddedToken`, *optional*) —
  A special token representing the end of a sentence. Will be associated to `self.eos_token` and
  `self.eos_token_id`.
* **unk\_token** (`str` or `tokenizers.AddedToken`, *optional*) —
  A special token representing an out-of-vocabulary token. Will be associated to `self.unk_token` and
  `self.unk_token_id`.
* **sep\_token** (`str` or `tokenizers.AddedToken`, *optional*) —
  A special token separating two different sentences in the same input (used by BERT for instance). Will be
  associated to `self.sep_token` and `self.sep_token_id`.
* **pad\_token** (`str` or `tokenizers.AddedToken`, *optional*) —
  A special token used to make arrays of tokens the same size for batching purpose. Will then be ignored by
  attention mechanisms or loss computation. Will be associated to `self.pad_token` and `self.pad_token_id`.
* **cls\_token** (`str` or `tokenizers.AddedToken`, *optional*) —
  A special token representing the class of the input (used by BERT for instance). Will be associated to
  `self.cls_token` and `self.cls_token_id`.
* **mask\_token** (`str` or `tokenizers.AddedToken`, *optional*) —
  A special token representing a masked token (used by masked-language modeling pretraining objectives, like
  BERT). Will be associated to `self.mask_token` and `self.mask_token_id`.
* **additional\_special\_tokens** (tuple or list of `str` or `tokenizers.AddedToken`, *optional*) —
  A tuple or a list of additional special tokens. Add them here to ensure they are skipped when decoding with
  `skip_special_tokens` is set to True. If they are not part of the vocabulary, they will be added at the end
  of the vocabulary.
* **clean\_up\_tokenization\_spaces** (`bool`, *optional*, defaults to `True`) —
  Whether or not the model should cleanup the spaces that were added when splitting the input text during the
  tokenization process.
* **split\_special\_tokens** (`bool`, *optional*, defaults to `False`) —
  Whether or not the special tokens should be split during the tokenization process. Passing will affect the
  internal state of the tokenizer. The default behavior is to not split special tokens. This means that if
  `<s>` is the `bos_token`, then `tokenizer.tokenize("<s>") = ['<s>`]. Otherwise, if
  `split_special_tokens=True`, then `tokenizer.tokenize("<s>")` will be give `['<','s', '>']`.
* **tokenizer\_object** (`tokenizers.Tokenizer`) —
  A `tokenizers.Tokenizer` object from 🤗 tokenizers to instantiate from. See [Using tokenizers from 🤗
  tokenizers](../fast_tokenizers) for more information.
* **tokenizer\_file** (`str`) —
  A path to a local JSON file representing a previously serialized `tokenizers.Tokenizer` object from 🤗
  tokenizers.

Fast tokenizer for Nougat (backed by HuggingFace tokenizers library).

This tokenizer inherits from [PreTrainedTokenizerFast](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast) which contains most of the main methods. Users should
refer to this superclass for more information regarding those methods. This class mainly adds Nougat-specific
methods for postprocessing the generated text.

Class attributes (overridden by derived classes)

* **vocab\_files\_names** (`dict[str, str]`) — A dictionary with, as keys, the `__init__` keyword name of each
  vocabulary file required by the model, and as associated values, the filename for saving the associated file
  (string).
* **pretrained\_vocab\_files\_map** (`dict[str, dict[str, str]]`) — A dictionary of dictionaries, with the
  high-level keys being the `__init__` keyword name of each vocabulary file required by the model, the
  low-level being the `short-cut-names` of the pretrained models with, as associated values, the `url` to the
  associated pretrained vocabulary file.
* **model\_input\_names** (`list[str]`) — A list of inputs expected in the forward pass of the model.
* **padding\_side** (`str`) — The default value for the side on which the model should have padding applied.
  Should be `'right'` or `'left'`.
* **truncation\_side** (`str`) — The default value for the side on which the model should have truncation
  applied. Should be `'right'` or `'left'`.

#### correct\_tables

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/nougat/tokenization_nougat_fast.py#L453)

( generation: str  ) → str

Parameters

* **generation** (str) — The generated text to be postprocessed.

Returns

str

The postprocessed text.

Takes a generated string and fixes tables/tabulars to make them match the markdown format needed.

Example:


```
correct_tables("\begin{table} \begin{tabular}{l l} & \ \end{tabular} \end{table}")
"\begin{table}
abular}{l l} & \ \end{tabular}
le}"
```

#### post\_process\_generation

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/nougat/tokenization_nougat_fast.py#L583)

( generation: typing.Union[str, list[str]] fix\_markdown: bool = True num\_workers: typing.Optional[int] = None  ) → Union[str, list[str]]

Parameters

* **generation** (Union[str, list[str]]) —
  The generated text or a list of generated texts.
* **fix\_markdown** (`bool`, *optional*, defaults to `True`) —
  Whether to perform Markdown formatting fixes.
* **num\_workers** (`int`, *optional*) —
  Optional number of workers to pass to leverage multiprocessing (postprocessing several texts in
  parallel).

Returns

Union[str, list[str]]

The postprocessed text or list of postprocessed texts.

Postprocess a generated text or a list of generated texts.

This function can be used to perform postprocessing on generated text, such as fixing Markdown formatting.

Postprocessing is quite slow so it is recommended to use multiprocessing to speed up the process.

#### post\_process\_single

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/nougat/tokenization_nougat_fast.py#L488)

( generation: str fix\_markdown: bool = True  ) → str

Parameters

* **generation** (str) — The generated text to be postprocessed.
* **fix\_markdown** (bool, optional) — Whether to perform Markdown formatting fixes. Default is True.

Returns

str

The postprocessed text.

Postprocess a single generated text. Regular expressions used here are taken directly from the Nougat article
authors. These expressions are commented for clarity and tested end-to-end in most cases.

#### remove\_hallucinated\_references

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/nougat/tokenization_nougat_fast.py#L423)

( text: str  ) → `str`

Parameters

* **text** (`str`) —
  The input text containing references.

Returns

`str`

The text with hallucinated references removed.

Remove hallucinated or missing references from the text.

This function identifies and removes references that are marked as missing or hallucinated from the input text.

## NougatProcessor

### class transformers.NougatProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/nougat/processing_nougat.py#L27)

( image\_processor tokenizer  )

Parameters

* **image\_processor** ([NougatImageProcessor](/docs/transformers/v4.56.2/en/model_doc/nougat#transformers.NougatImageProcessor)) —
  An instance of [NougatImageProcessor](/docs/transformers/v4.56.2/en/model_doc/nougat#transformers.NougatImageProcessor). The image processor is a required input.
* **tokenizer** ([NougatTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/nougat#transformers.NougatTokenizerFast)) —
  An instance of [NougatTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/nougat#transformers.NougatTokenizerFast). The tokenizer is a required input.

Constructs a Nougat processor which wraps a Nougat image processor and a Nougat tokenizer into a single processor.

[NougatProcessor](/docs/transformers/v4.56.2/en/model_doc/nougat#transformers.NougatProcessor) offers all the functionalities of [NougatImageProcessor](/docs/transformers/v4.56.2/en/model_doc/nougat#transformers.NougatImageProcessor) and [NougatTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/nougat#transformers.NougatTokenizerFast). See the
[**call**()](/docs/transformers/v4.56.2/en/model_doc/nougat#transformers.NougatProcessor.__call__) and [decode()](/docs/transformers/v4.56.2/en/main_classes/processors#transformers.ProcessorMixin.decode) for more information.

#### \_\_call\_\_

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/nougat/processing_nougat.py#L49)

( images = None text = None do\_crop\_margin: typing.Optional[bool] = None do\_resize: typing.Optional[bool] = None size: typing.Optional[dict[str, int]] = None resample: PILImageResampling = None do\_thumbnail: typing.Optional[bool] = None do\_align\_long\_axis: typing.Optional[bool] = None do\_pad: typing.Optional[bool] = None do\_rescale: typing.Optional[bool] = None rescale\_factor: typing.Union[int, float, NoneType] = None do\_normalize: typing.Optional[bool] = None image\_mean: typing.Union[float, list[float], NoneType] = None image\_std: typing.Union[float, list[float], NoneType] = None data\_format: typing.Optional[ForwardRef('ChannelDimension')] = 'channels\_first' input\_data\_format: typing.Union[str, ForwardRef('ChannelDimension'), NoneType] = None text\_pair: typing.Union[str, list[str], list[list[str]], NoneType] = None text\_target: typing.Union[str, list[str], list[list[str]]] = None text\_pair\_target: typing.Union[str, list[str], list[list[str]], NoneType] = None add\_special\_tokens: bool = True padding: typing.Union[bool, str, transformers.utils.generic.PaddingStrategy] = False truncation: typing.Union[bool, str, transformers.tokenization\_utils\_base.TruncationStrategy] = None max\_length: typing.Optional[int] = None stride: int = 0 is\_split\_into\_words: bool = False pad\_to\_multiple\_of: typing.Optional[int] = None return\_tensors: typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None return\_token\_type\_ids: typing.Optional[bool] = None return\_attention\_mask: typing.Optional[bool] = None return\_overflowing\_tokens: bool = False return\_special\_tokens\_mask: bool = False return\_offsets\_mapping: bool = False return\_length: bool = False verbose: bool = True  )

#### from\_pretrained

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/processing_utils.py#L1272)

( pretrained\_model\_name\_or\_path: typing.Union[str, os.PathLike] cache\_dir: typing.Union[str, os.PathLike, NoneType] = None force\_download: bool = False local\_files\_only: bool = False token: typing.Union[bool, str, NoneType] = None revision: str = 'main' \*\*kwargs  )

Parameters

* **pretrained\_model\_name\_or\_path** (`str` or `os.PathLike`) —
  This can be either:
  + a string, the *model id* of a pretrained feature\_extractor hosted inside a model repo on
    huggingface.co.
  + a path to a *directory* containing a feature extractor file saved using the
    [save\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/feature_extractor#transformers.FeatureExtractionMixin.save_pretrained) method, e.g., `./my_model_directory/`.
  + a path or url to a saved feature extractor JSON *file*, e.g.,
    `./my_model_directory/preprocessor_config.json`.
* \***\*kwargs** —
  Additional keyword arguments passed along to both
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/feature_extractor#transformers.FeatureExtractionMixin.from_pretrained) and
  `~tokenization_utils_base.PreTrainedTokenizer.from_pretrained`.

Instantiate a processor associated with a pretrained model.

This class method is simply calling the feature extractor
[from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/feature_extractor#transformers.FeatureExtractionMixin.from_pretrained), image processor
[ImageProcessingMixin](/docs/transformers/v4.56.2/en/main_classes/image_processor#transformers.ImageProcessingMixin) and the tokenizer
`~tokenization_utils_base.PreTrainedTokenizer.from_pretrained` methods. Please refer to the docstrings of the
methods above for more information.

#### save\_pretrained

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/processing_utils.py#L653)

( save\_directory push\_to\_hub: bool = False legacy\_serialization: bool = True \*\*kwargs  )

Parameters

* **save\_directory** (`str` or `os.PathLike`) —
  Directory where the feature extractor JSON file and the tokenizer files will be saved (directory will
  be created if it does not exist).
* **push\_to\_hub** (`bool`, *optional*, defaults to `False`) —
  Whether or not to push your model to the Hugging Face model hub after saving it. You can specify the
  repository you want to push to with `repo_id` (will default to the name of `save_directory` in your
  namespace).
* **legacy\_serialization** (`bool`, *optional*, defaults to `True`) —
  Whether or not to save processor attributes in separate config files (legacy) or in processor’s config
  file as a nested dict. Saving all attributes in a single dict will become the default in future versions.
  Set to `legacy_serialization=True` until then.
* **kwargs** (`dict[str, Any]`, *optional*) —
  Additional key word arguments passed along to the [push\_to\_hub()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.utils.PushToHubMixin.push_to_hub) method.

Saves the attributes of this processor (feature extractor, tokenizer…) in the specified directory so that it
can be reloaded using the [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/processors#transformers.ProcessorMixin.from_pretrained) method.

This class method is simply calling [save\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/feature_extractor#transformers.FeatureExtractionMixin.save_pretrained) and
[save\_pretrained()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.save_pretrained). Please refer to the docstrings of the
methods above for more information.

#### batch\_decode

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/processing_utils.py#L1419)

( \*args \*\*kwargs  )

This method forwards all its arguments to PreTrainedTokenizer’s [batch\_decode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.batch_decode). Please
refer to the docstring of this method for more information.

#### decode

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/processing_utils.py#L1428)

( \*args \*\*kwargs  )

This method forwards all its arguments to PreTrainedTokenizer’s [decode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.decode). Please refer to
the docstring of this method for more information.

#### post\_process\_generation

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/nougat/processing_nougat.py#L141)

( \*args \*\*kwargs  )

This method forwards all its arguments to NougatTokenizer’s `~PreTrainedTokenizer.post_process_generation`.
Please refer to the docstring of this method for more information.

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/nougat.md)
