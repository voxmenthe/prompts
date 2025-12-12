*This model was released on 2020-12-29 and added to Hugging Face Transformers on 2021-08-30.*

# LayoutLMV2

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white)

## Overview

The LayoutLMV2 model was proposed in [LayoutLMv2: Multi-modal Pre-training for Visually-Rich Document Understanding](https://huggingface.co/papers/2012.14740) by Yang Xu, Yiheng Xu, Tengchao Lv, Lei Cui, Furu Wei, Guoxin Wang, Yijuan Lu,
Dinei Florencio, Cha Zhang, Wanxiang Che, Min Zhang, Lidong Zhou. LayoutLMV2 improves [LayoutLM](layoutlm) to obtain
state-of-the-art results across several document image understanding benchmarks:

* information extraction from scanned documents: the [FUNSD](https://guillaumejaume.github.io/FUNSD/) dataset (a
  collection of 199 annotated forms comprising more than 30,000 words), the [CORD](https://github.com/clovaai/cord)
  dataset (a collection of 800 receipts for training, 100 for validation and 100 for testing), the [SROIE](https://rrc.cvc.uab.es/?ch=13) dataset (a collection of 626 receipts for training and 347 receipts for testing)
  and the [Kleister-NDA](https://github.com/applicaai/kleister-nda) dataset (a collection of non-disclosure
  agreements from the EDGAR database, including 254 documents for training, 83 documents for validation, and 203
  documents for testing).
* document image classification: the [RVL-CDIP](https://www.cs.cmu.edu/~aharley/rvl-cdip/) dataset (a collection of
  400,000 images belonging to one of 16 classes).
* document visual question answering: the [DocVQA](https://huggingface.co/papers/2007.00398) dataset (a collection of 50,000
  questions defined on 12,000+ document images).

The abstract from the paper is the following:

*Pre-training of text and layout has proved effective in a variety of visually-rich document understanding tasks due to
its effective model architecture and the advantage of large-scale unlabeled scanned/digital-born documents. In this
paper, we present LayoutLMv2 by pre-training text, layout and image in a multi-modal framework, where new model
architectures and pre-training tasks are leveraged. Specifically, LayoutLMv2 not only uses the existing masked
visual-language modeling task but also the new text-image alignment and text-image matching tasks in the pre-training
stage, where cross-modality interaction is better learned. Meanwhile, it also integrates a spatial-aware self-attention
mechanism into the Transformer architecture, so that the model can fully understand the relative positional
relationship among different text blocks. Experiment results show that LayoutLMv2 outperforms strong baselines and
achieves new state-of-the-art results on a wide variety of downstream visually-rich document understanding tasks,
including FUNSD (0.7895 -> 0.8420), CORD (0.9493 -> 0.9601), SROIE (0.9524 -> 0.9781), Kleister-NDA (0.834 -> 0.852),
RVL-CDIP (0.9443 -> 0.9564), and DocVQA (0.7295 -> 0.8672). The pre-trained LayoutLMv2 model is publicly available at
this https URL.*

LayoutLMv2 depends on `detectron2`, `torchvision` and `tesseract`. Run the
following to install them:


```
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
python -m pip install torchvision tesseract
```

(If you are developing for LayoutLMv2, note that passing the doctests also requires the installation of these packages.)

## Usage tips

* The main difference between LayoutLMv1 and LayoutLMv2 is that the latter incorporates visual embeddings during
  pre-training (while LayoutLMv1 only adds visual embeddings during fine-tuning).
* LayoutLMv2 adds both a relative 1D attention bias as well as a spatial 2D attention bias to the attention scores in
  the self-attention layers. Details can be found on page 5 of the [paper](https://huggingface.co/papers/2012.14740).
* Demo notebooks on how to use the LayoutLMv2 model on RVL-CDIP, FUNSD, DocVQA, CORD can be found [here](https://github.com/NielsRogge/Transformers-Tutorials).
* LayoutLMv2 uses Facebook AI’s [Detectron2](https://github.com/facebookresearch/detectron2/) package for its visual
  backbone. See [this link](https://detectron2.readthedocs.io/en/latest/tutorials/install.html) for installation
  instructions.
* In addition to `input_ids`, [forward()](/docs/transformers/v4.56.2/en/model_doc/layoutlmv2#transformers.LayoutLMv2Model.forward) expects 2 additional inputs, namely
  `image` and `bbox`. The `image` input corresponds to the original document image in which the text
  tokens occur. The model expects each document image to be of size 224x224. This means that if you have a batch of
  document images, `image` should be a tensor of shape (batch\_size, 3, 224, 224). This can be either a
  `torch.Tensor` or a `Detectron2.structures.ImageList`. You don’t need to normalize the channels, as this is
  done by the model. Important to note is that the visual backbone expects BGR channels instead of RGB, as all models
  in Detectron2 are pre-trained using the BGR format. The `bbox` input are the bounding boxes (i.e. 2D-positions)
  of the input text tokens. This is identical to [LayoutLMModel](/docs/transformers/v4.56.2/en/model_doc/layoutlm#transformers.LayoutLMModel). These can be obtained using an
  external OCR engine such as Google’s [Tesseract](https://github.com/tesseract-ocr/tesseract) (there’s a [Python
  wrapper](https://pypi.org/project/pytesseract/) available). Each bounding box should be in (x0, y0, x1, y1)
  format, where (x0, y0) corresponds to the position of the upper left corner in the bounding box, and (x1, y1)
  represents the position of the lower right corner. Note that one first needs to normalize the bounding boxes to be on
  a 0-1000 scale. To normalize, you can use the following function:


```
def normalize_bbox(bbox, width, height):
    return [
        int(1000 * (bbox[0] / width)),
        int(1000 * (bbox[1] / height)),
        int(1000 * (bbox[2] / width)),
        int(1000 * (bbox[3] / height)),
    ]
```

Here, `width` and `height` correspond to the width and height of the original document in which the token
occurs (before resizing the image). Those can be obtained using the Python Image Library (PIL) library for example, as
follows:


```
from PIL import Image

image = Image.open(
    "name_of_your_document - can be a png, jpg, etc. of your documents (PDFs must be converted to images)."
)

width, height = image.size
```

However, this model includes a brand new [LayoutLMv2Processor](/docs/transformers/v4.56.2/en/model_doc/layoutlmv2#transformers.LayoutLMv2Processor) which can be used to directly
prepare data for the model (including applying OCR under the hood). More information can be found in the “Usage”
section below.

* Internally, [LayoutLMv2Model](/docs/transformers/v4.56.2/en/model_doc/layoutlmv2#transformers.LayoutLMv2Model) will send the `image` input through its visual backbone to
  obtain a lower-resolution feature map, whose shape is equal to the `image_feature_pool_shape` attribute of
  [LayoutLMv2Config](/docs/transformers/v4.56.2/en/model_doc/layoutlmv2#transformers.LayoutLMv2Config). This feature map is then flattened to obtain a sequence of image tokens. As
  the size of the feature map is 7x7 by default, one obtains 49 image tokens. These are then concatenated with the text
  tokens, and send through the Transformer encoder. This means that the last hidden states of the model will have a
  length of 512 + 49 = 561, if you pad the text tokens up to the max length. More generally, the last hidden states
  will have a shape of `seq_length` + `image_feature_pool_shape[0]` \*
  `config.image_feature_pool_shape[1]`.
* When calling [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained), a warning will be printed with a long list of
  parameter names that are not initialized. This is not a problem, as these parameters are batch normalization
  statistics, which are going to have values when fine-tuning on a custom dataset.
* If you want to train the model in a distributed environment, make sure to call `synchronize_batch_norm` on the
  model in order to properly synchronize the batch normalization layers of the visual backbone.

In addition, there’s LayoutXLM, which is a multilingual version of LayoutLMv2. More information can be found on
[LayoutXLM’s documentation page](layoutxlm).

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with LayoutLMv2. If you’re interested in submitting a resource to be included here, please feel free to open a Pull Request and we’ll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

Text Classification

* A notebook on how to [finetune LayoutLMv2 for text-classification on RVL-CDIP dataset](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/LayoutLMv2/RVL-CDIP/Fine_tuning_LayoutLMv2ForSequenceClassification_on_RVL_CDIP.ipynb).
* See also: [Text classification task guide](../tasks/sequence_classification)

Question Answering

* A notebook on how to [finetune LayoutLMv2 for question-answering on DocVQA dataset](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/LayoutLMv2/DocVQA/Fine_tuning_LayoutLMv2ForQuestionAnswering_on_DocVQA.ipynb).
* See also: [Question answering task guide](../tasks/question_answering)
* See also: [Document question answering task guide](../tasks/document_question_answering)

Token Classification

* A notebook on how to [finetune LayoutLMv2 for token-classification on CORD dataset](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/LayoutLMv2/CORD/Fine_tuning_LayoutLMv2ForTokenClassification_on_CORD.ipynb).
* A notebook on how to [finetune LayoutLMv2 for token-classification on FUNSD dataset](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/LayoutLMv2/FUNSD/Fine_tuning_LayoutLMv2ForTokenClassification_on_FUNSD_using_HuggingFace_Trainer.ipynb).
* See also: [Token classification task guide](../tasks/token_classification)

## Usage: LayoutLMv2Processor

The easiest way to prepare data for the model is to use [LayoutLMv2Processor](/docs/transformers/v4.56.2/en/model_doc/layoutlmv2#transformers.LayoutLMv2Processor), which internally
combines a image processor ([LayoutLMv2ImageProcessor](/docs/transformers/v4.56.2/en/model_doc/layoutlmv2#transformers.LayoutLMv2ImageProcessor)) and a tokenizer
([LayoutLMv2Tokenizer](/docs/transformers/v4.56.2/en/model_doc/layoutlmv2#transformers.LayoutLMv2Tokenizer) or [LayoutLMv2TokenizerFast](/docs/transformers/v4.56.2/en/model_doc/layoutlmv2#transformers.LayoutLMv2TokenizerFast)). The image processor
handles the image modality, while the tokenizer handles the text modality. A processor combines both, which is ideal
for a multi-modal model like LayoutLMv2. Note that you can still use both separately, if you only want to handle one
modality.


```
from transformers import LayoutLMv2ImageProcessor, LayoutLMv2TokenizerFast, LayoutLMv2Processor

image_processor = LayoutLMv2ImageProcessor()  # apply_ocr is set to True by default
tokenizer = LayoutLMv2TokenizerFast.from_pretrained("microsoft/layoutlmv2-base-uncased")
processor = LayoutLMv2Processor(image_processor, tokenizer)
```

In short, one can provide a document image (and possibly additional data) to [LayoutLMv2Processor](/docs/transformers/v4.56.2/en/model_doc/layoutlmv2#transformers.LayoutLMv2Processor),
and it will create the inputs expected by the model. Internally, the processor first uses
[LayoutLMv2ImageProcessor](/docs/transformers/v4.56.2/en/model_doc/layoutlmv2#transformers.LayoutLMv2ImageProcessor) to apply OCR on the image to get a list of words and normalized
bounding boxes, as well to resize the image to a given size in order to get the `image` input. The words and
normalized bounding boxes are then provided to [LayoutLMv2Tokenizer](/docs/transformers/v4.56.2/en/model_doc/layoutlmv2#transformers.LayoutLMv2Tokenizer) or
[LayoutLMv2TokenizerFast](/docs/transformers/v4.56.2/en/model_doc/layoutlmv2#transformers.LayoutLMv2TokenizerFast), which converts them to token-level `input_ids`,
`attention_mask`, `token_type_ids`, `bbox`. Optionally, one can provide word labels to the processor,
which are turned into token-level `labels`.

[LayoutLMv2Processor](/docs/transformers/v4.56.2/en/model_doc/layoutlmv2#transformers.LayoutLMv2Processor) uses [PyTesseract](https://pypi.org/project/pytesseract/), a Python
wrapper around Google’s Tesseract OCR engine, under the hood. Note that you can still use your own OCR engine of
choice, and provide the words and normalized boxes yourself. This requires initializing
[LayoutLMv2ImageProcessor](/docs/transformers/v4.56.2/en/model_doc/layoutlmv2#transformers.LayoutLMv2ImageProcessor) with `apply_ocr` set to `False`.

In total, there are 5 use cases that are supported by the processor. Below, we list them all. Note that each of these
use cases work for both batched and non-batched inputs (we illustrate them for non-batched inputs).

**Use case 1: document image classification (training, inference) + token classification (inference), apply\_ocr =
True**

This is the simplest case, in which the processor (actually the image processor) will perform OCR on the image to get
the words and normalized bounding boxes.


```
from transformers import LayoutLMv2Processor
from PIL import Image

processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased")

image = Image.open(
    "name_of_your_document - can be a png, jpg, etc. of your documents (PDFs must be converted to images)."
).convert("RGB")
encoding = processor(
    image, return_tensors="pt"
)  # you can also add all tokenizer parameters here such as padding, truncation
print(encoding.keys())
# dict_keys(['input_ids', 'token_type_ids', 'attention_mask', 'bbox', 'image'])
```

**Use case 2: document image classification (training, inference) + token classification (inference), apply\_ocr=False**

In case one wants to do OCR themselves, one can initialize the image processor with `apply_ocr` set to
`False`. In that case, one should provide the words and corresponding (normalized) bounding boxes themselves to
the processor.


```
from transformers import LayoutLMv2Processor
from PIL import Image

processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased", revision="no_ocr")

image = Image.open(
    "name_of_your_document - can be a png, jpg, etc. of your documents (PDFs must be converted to images)."
).convert("RGB")
words = ["hello", "world"]
boxes = [[1, 2, 3, 4], [5, 6, 7, 8]]  # make sure to normalize your bounding boxes
encoding = processor(image, words, boxes=boxes, return_tensors="pt")
print(encoding.keys())
# dict_keys(['input_ids', 'token_type_ids', 'attention_mask', 'bbox', 'image'])
```

**Use case 3: token classification (training), apply\_ocr=False**

For token classification tasks (such as FUNSD, CORD, SROIE, Kleister-NDA), one can also provide the corresponding word
labels in order to train a model. The processor will then convert these into token-level `labels`. By default, it
will only label the first wordpiece of a word, and label the remaining wordpieces with -100, which is the
`ignore_index` of PyTorch’s CrossEntropyLoss. In case you want all wordpieces of a word to be labeled, you can
initialize the tokenizer with `only_label_first_subword` set to `False`.


```
from transformers import LayoutLMv2Processor
from PIL import Image

processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased", revision="no_ocr")

image = Image.open(
    "name_of_your_document - can be a png, jpg, etc. of your documents (PDFs must be converted to images)."
).convert("RGB")
words = ["hello", "world"]
boxes = [[1, 2, 3, 4], [5, 6, 7, 8]]  # make sure to normalize your bounding boxes
word_labels = [1, 2]
encoding = processor(image, words, boxes=boxes, word_labels=word_labels, return_tensors="pt")
print(encoding.keys())
# dict_keys(['input_ids', 'token_type_ids', 'attention_mask', 'bbox', 'labels', 'image'])
```

**Use case 4: visual question answering (inference), apply\_ocr=True**

For visual question answering tasks (such as DocVQA), you can provide a question to the processor. By default, the
processor will apply OCR on the image, and create [CLS] question tokens [SEP] word tokens [SEP].


```
from transformers import LayoutLMv2Processor
from PIL import Image

processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased")

image = Image.open(
    "name_of_your_document - can be a png, jpg, etc. of your documents (PDFs must be converted to images)."
).convert("RGB")
question = "What's his name?"
encoding = processor(image, question, return_tensors="pt")
print(encoding.keys())
# dict_keys(['input_ids', 'token_type_ids', 'attention_mask', 'bbox', 'image'])
```

**Use case 5: visual question answering (inference), apply\_ocr=False**

For visual question answering tasks (such as DocVQA), you can provide a question to the processor. If you want to
perform OCR yourself, you can provide your own words and (normalized) bounding boxes to the processor.


```
from transformers import LayoutLMv2Processor
from PIL import Image

processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased", revision="no_ocr")

image = Image.open(
    "name_of_your_document - can be a png, jpg, etc. of your documents (PDFs must be converted to images)."
).convert("RGB")
question = "What's his name?"
words = ["hello", "world"]
boxes = [[1, 2, 3, 4], [5, 6, 7, 8]]  # make sure to normalize your bounding boxes
encoding = processor(image, question, words, boxes=boxes, return_tensors="pt")
print(encoding.keys())
# dict_keys(['input_ids', 'token_type_ids', 'attention_mask', 'bbox', 'image'])
```

## LayoutLMv2Config

### class transformers.LayoutLMv2Config

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/layoutlmv2/configuration_layoutlmv2.py#L29)

( vocab\_size = 30522 hidden\_size = 768 num\_hidden\_layers = 12 num\_attention\_heads = 12 intermediate\_size = 3072 hidden\_act = 'gelu' hidden\_dropout\_prob = 0.1 attention\_probs\_dropout\_prob = 0.1 max\_position\_embeddings = 512 type\_vocab\_size = 2 initializer\_range = 0.02 layer\_norm\_eps = 1e-12 pad\_token\_id = 0 max\_2d\_position\_embeddings = 1024 max\_rel\_pos = 128 rel\_pos\_bins = 32 fast\_qkv = True max\_rel\_2d\_pos = 256 rel\_2d\_pos\_bins = 64 convert\_sync\_batchnorm = True image\_feature\_pool\_shape = [7, 7, 256] coordinate\_size = 128 shape\_size = 128 has\_relative\_attention\_bias = True has\_spatial\_attention\_bias = True has\_visual\_segment\_embedding = False detectron2\_config\_args = None \*\*kwargs  )

Parameters

* **vocab\_size** (`int`, *optional*, defaults to 30522) —
  Vocabulary size of the LayoutLMv2 model. Defines the number of different tokens that can be represented by
  the `inputs_ids` passed when calling [LayoutLMv2Model](/docs/transformers/v4.56.2/en/model_doc/layoutlmv2#transformers.LayoutLMv2Model) or `TFLayoutLMv2Model`.
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
  The vocabulary size of the `token_type_ids` passed when calling [LayoutLMv2Model](/docs/transformers/v4.56.2/en/model_doc/layoutlmv2#transformers.LayoutLMv2Model) or
  `TFLayoutLMv2Model`.
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
* **layer\_norm\_eps** (`float`, *optional*, defaults to 1e-12) —
  The epsilon used by the layer normalization layers.
* **max\_2d\_position\_embeddings** (`int`, *optional*, defaults to 1024) —
  The maximum value that the 2D position embedding might ever be used with. Typically set this to something
  large just in case (e.g., 1024).
* **max\_rel\_pos** (`int`, *optional*, defaults to 128) —
  The maximum number of relative positions to be used in the self-attention mechanism.
* **rel\_pos\_bins** (`int`, *optional*, defaults to 32) —
  The number of relative position bins to be used in the self-attention mechanism.
* **fast\_qkv** (`bool`, *optional*, defaults to `True`) —
  Whether or not to use a single matrix for the queries, keys, values in the self-attention layers.
* **max\_rel\_2d\_pos** (`int`, *optional*, defaults to 256) —
  The maximum number of relative 2D positions in the self-attention mechanism.
* **rel\_2d\_pos\_bins** (`int`, *optional*, defaults to 64) —
  The number of 2D relative position bins in the self-attention mechanism.
* **image\_feature\_pool\_shape** (`list[int]`, *optional*, defaults to [7, 7, 256]) —
  The shape of the average-pooled feature map.
* **coordinate\_size** (`int`, *optional*, defaults to 128) —
  Dimension of the coordinate embeddings.
* **shape\_size** (`int`, *optional*, defaults to 128) —
  Dimension of the width and height embeddings.
* **has\_relative\_attention\_bias** (`bool`, *optional*, defaults to `True`) —
  Whether or not to use a relative attention bias in the self-attention mechanism.
* **has\_spatial\_attention\_bias** (`bool`, *optional*, defaults to `True`) —
  Whether or not to use a spatial attention bias in the self-attention mechanism.
* **has\_visual\_segment\_embedding** (`bool`, *optional*, defaults to `False`) —
  Whether or not to add visual segment embeddings.
* **detectron2\_config\_args** (`dict`, *optional*) —
  Dictionary containing the configuration arguments of the Detectron2 visual backbone. Refer to [this
  file](https://github.com/microsoft/unilm/blob/master/layoutlmft/layoutlmft/models/layoutlmv2/detectron2_config.py)
  for details regarding default values.

This is the configuration class to store the configuration of a [LayoutLMv2Model](/docs/transformers/v4.56.2/en/model_doc/layoutlmv2#transformers.LayoutLMv2Model). It is used to instantiate an
LayoutLMv2 model according to the specified arguments, defining the model architecture. Instantiating a
configuration with the defaults will yield a similar configuration to that of the LayoutLMv2
[microsoft/layoutlmv2-base-uncased](https://huggingface.co/microsoft/layoutlmv2-base-uncased) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import LayoutLMv2Config, LayoutLMv2Model

>>> # Initializing a LayoutLMv2 microsoft/layoutlmv2-base-uncased style configuration
>>> configuration = LayoutLMv2Config()

>>> # Initializing a model (with random weights) from the microsoft/layoutlmv2-base-uncased style configuration
>>> model = LayoutLMv2Model(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## LayoutLMv2FeatureExtractor

### class transformers.LayoutLMv2FeatureExtractor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/layoutlmv2/feature_extraction_layoutlmv2.py#L30)

( \*args \*\*kwargs  )

#### \_\_call\_\_

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/image_processing_utils.py#L49)

( images \*\*kwargs  )

Preprocess an image or a batch of images.

## LayoutLMv2ImageProcessor

### class transformers.LayoutLMv2ImageProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/layoutlmv2/image_processing_layoutlmv2.py#L103)

( do\_resize: bool = True size: typing.Optional[dict[str, int]] = None resample: Resampling = <Resampling.BILINEAR: 2> apply\_ocr: bool = True ocr\_lang: typing.Optional[str] = None tesseract\_config: typing.Optional[str] = '' \*\*kwargs  )

Parameters

* **do\_resize** (`bool`, *optional*, defaults to `True`) —
  Whether to resize the image’s (height, width) dimensions to `(size["height"], size["width"])`. Can be
  overridden by `do_resize` in `preprocess`.
* **size** (`dict[str, int]` *optional*, defaults to `{"height" -- 224, "width": 224}`):
  Size of the image after resizing. Can be overridden by `size` in `preprocess`.
* **resample** (`PILImageResampling`, *optional*, defaults to `Resampling.BILINEAR`) —
  Resampling filter to use if resizing the image. Can be overridden by the `resample` parameter in the
  `preprocess` method.
* **apply\_ocr** (`bool`, *optional*, defaults to `True`) —
  Whether to apply the Tesseract OCR engine to get words + normalized bounding boxes. Can be overridden by
  `apply_ocr` in `preprocess`.
* **ocr\_lang** (`str`, *optional*) —
  The language, specified by its ISO code, to be used by the Tesseract OCR engine. By default, English is
  used. Can be overridden by `ocr_lang` in `preprocess`.
* **tesseract\_config** (`str`, *optional*, defaults to `""`) —
  Any additional custom configuration flags that are forwarded to the `config` parameter when calling
  Tesseract. For example: ‘—psm 6’. Can be overridden by `tesseract_config` in `preprocess`.

Constructs a LayoutLMv2 image processor.

#### preprocess

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/layoutlmv2/image_processing_layoutlmv2.py#L199)

( images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']] do\_resize: typing.Optional[bool] = None size: typing.Optional[dict[str, int]] = None resample: Resampling = None apply\_ocr: typing.Optional[bool] = None ocr\_lang: typing.Optional[str] = None tesseract\_config: typing.Optional[str] = None return\_tensors: typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None data\_format: ChannelDimension = <ChannelDimension.FIRST: 'channels\_first'> input\_data\_format: typing.Union[str, transformers.image\_utils.ChannelDimension, NoneType] = None  )

Parameters

* **images** (`ImageInput`) —
  Image to preprocess.
* **do\_resize** (`bool`, *optional*, defaults to `self.do_resize`) —
  Whether to resize the image.
* **size** (`dict[str, int]`, *optional*, defaults to `self.size`) —
  Desired size of the output image after resizing.
* **resample** (`PILImageResampling`, *optional*, defaults to `self.resample`) —
  Resampling filter to use if resizing the image. This can be one of the enum `PIL.Image` resampling
  filter. Only has an effect if `do_resize` is set to `True`.
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

Preprocess an image or batch of images.

## LayoutLMv2ImageProcessorFast

### class transformers.LayoutLMv2ImageProcessorFast

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/layoutlmv2/image_processing_layoutlmv2_fast.py#L68)

( \*\*kwargs: typing\_extensions.Unpack[transformers.models.layoutlmv2.image\_processing\_layoutlmv2\_fast.LayoutLMv2FastImageProcessorKwargs]  )

Constructs a fast Layoutlmv2 image processor.

#### preprocess

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/layoutlmv2/image_processing_layoutlmv2_fast.py#L81)

( images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']] \*\*kwargs: typing\_extensions.Unpack[transformers.models.layoutlmv2.image\_processing\_layoutlmv2\_fast.LayoutLMv2FastImageProcessorKwargs]  ) → `<class 'transformers.image_processing_base.BatchFeature'>`

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

## LayoutLMv2Tokenizer

### class transformers.LayoutLMv2Tokenizer

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/layoutlmv2/tokenization_layoutlmv2.py#L183)

( vocab\_file do\_lower\_case = True do\_basic\_tokenize = True never\_split = None unk\_token = '[UNK]' sep\_token = '[SEP]' pad\_token = '[PAD]' cls\_token = '[CLS]' mask\_token = '[MASK]' cls\_token\_box = [0, 0, 0, 0] sep\_token\_box = [1000, 1000, 1000, 1000] pad\_token\_box = [0, 0, 0, 0] pad\_token\_label = -100 only\_label\_first\_subword = True tokenize\_chinese\_chars = True strip\_accents = None model\_max\_length: int = 512 additional\_special\_tokens: typing.Optional[list[str]] = None \*\*kwargs  )

Construct a LayoutLMv2 tokenizer. Based on WordPiece. [LayoutLMv2Tokenizer](/docs/transformers/v4.56.2/en/model_doc/layoutlmv2#transformers.LayoutLMv2Tokenizer) can be used to turn words, word-level
bounding boxes and optional word labels to token-level `input_ids`, `attention_mask`, `token_type_ids`, `bbox`, and
optional `labels` (for token classification).

This tokenizer inherits from [PreTrainedTokenizer](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer) which contains most of the main methods. Users should refer to
this superclass for more information regarding those methods.

[LayoutLMv2Tokenizer](/docs/transformers/v4.56.2/en/model_doc/layoutlmv2#transformers.LayoutLMv2Tokenizer) runs end-to-end tokenization: punctuation splitting and wordpiece. It also turns the
word-level bounding boxes into token-level bounding boxes.

#### \_\_call\_\_

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/layoutlmv2/tokenization_layoutlmv2.py#L381)

( text: typing.Union[str, list[str], list[list[str]]] text\_pair: typing.Union[list[str], list[list[str]], NoneType] = None boxes: typing.Union[list[list[int]], list[list[list[int]]], NoneType] = None word\_labels: typing.Union[list[int], list[list[int]], NoneType] = None add\_special\_tokens: bool = True padding: typing.Union[bool, str, transformers.utils.generic.PaddingStrategy] = False truncation: typing.Union[bool, str, transformers.tokenization\_utils\_base.TruncationStrategy] = None max\_length: typing.Optional[int] = None stride: int = 0 pad\_to\_multiple\_of: typing.Optional[int] = None padding\_side: typing.Optional[str] = None return\_tensors: typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None return\_token\_type\_ids: typing.Optional[bool] = None return\_attention\_mask: typing.Optional[bool] = None return\_overflowing\_tokens: bool = False return\_special\_tokens\_mask: bool = False return\_offsets\_mapping: bool = False return\_length: bool = False verbose: bool = True \*\*kwargs  ) → [BatchEncoding](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.BatchEncoding)

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
* **return\_token\_type\_ids** (`bool`, *optional*) —
  Whether to return token type IDs. If left to the default, will return the token type IDs according to
  the specific tokenizer’s default, defined by the `return_outputs` attribute.

  [What are token type IDs?](../glossary#token-type-ids)
* **return\_attention\_mask** (`bool`, *optional*) —
  Whether to return the attention mask. If left to the default, will return the attention mask according
  to the specific tokenizer’s default, defined by the `return_outputs` attribute.

  [What are attention masks?](../glossary#attention-mask)
* **return\_overflowing\_tokens** (`bool`, *optional*, defaults to `False`) —
  Whether or not to return overflowing token sequences. If a pair of sequences of input ids (or a batch
  of pairs) is provided with `truncation_strategy = longest_first` or `True`, an error is raised instead
  of returning overflowing tokens.
* **return\_special\_tokens\_mask** (`bool`, *optional*, defaults to `False`) —
  Whether or not to return special tokens mask information.
* **return\_offsets\_mapping** (`bool`, *optional*, defaults to `False`) —
  Whether or not to return `(char_start, char_end)` for each token.

  This is only available on fast tokenizers inheriting from [PreTrainedTokenizerFast](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast), if using
  Python’s tokenizer, this method will raise `NotImplementedError`.
* **return\_length** (`bool`, *optional*, defaults to `False`) —
  Whether or not to return the lengths of the encoded inputs.
* **verbose** (`bool`, *optional*, defaults to `True`) —
  Whether or not to print more information and warnings.
* \***\*kwargs** — passed to the `self.tokenize()` method

Returns

[BatchEncoding](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.BatchEncoding)

A [BatchEncoding](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.BatchEncoding) with the following fields:

* **input\_ids** — List of token ids to be fed to a model.

  [What are input IDs?](../glossary#input-ids)
* **bbox** — List of bounding boxes to be fed to a model.
* **token\_type\_ids** — List of token type ids to be fed to a model (when `return_token_type_ids=True` or
  if *“token\_type\_ids”* is in `self.model_input_names`).

  [What are token type IDs?](../glossary#token-type-ids)
* **attention\_mask** — List of indices specifying which tokens should be attended to by the model (when
  `return_attention_mask=True` or if *“attention\_mask”* is in `self.model_input_names`).

  [What are attention masks?](../glossary#attention-mask)
* **labels** — List of labels to be fed to a model. (when `word_labels` is specified).
* **overflowing\_tokens** — List of overflowing tokens sequences (when a `max_length` is specified and
  `return_overflowing_tokens=True`).
* **num\_truncated\_tokens** — Number of tokens truncated (when a `max_length` is specified and
  `return_overflowing_tokens=True`).
* **special\_tokens\_mask** — List of 0s and 1s, with 1 specifying added special tokens and 0 specifying
  regular sequence tokens (when `add_special_tokens=True` and `return_special_tokens_mask=True`).
* **length** — The length of the inputs (when `return_length=True`).

Main method to tokenize and prepare for the model one or several sequence(s) or one or several pair(s) of
sequences with word-level normalized bounding boxes and optional labels.

#### save\_vocabulary

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/layoutlmv2/tokenization_layoutlmv2.py#L361)

( save\_directory: str filename\_prefix: typing.Optional[str] = None  )

## LayoutLMv2TokenizerFast

### class transformers.LayoutLMv2TokenizerFast

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/layoutlmv2/tokenization_layoutlmv2_fast.py#L49)

( vocab\_file = None tokenizer\_file = None do\_lower\_case = True unk\_token = '[UNK]' sep\_token = '[SEP]' pad\_token = '[PAD]' cls\_token = '[CLS]' mask\_token = '[MASK]' cls\_token\_box = [0, 0, 0, 0] sep\_token\_box = [1000, 1000, 1000, 1000] pad\_token\_box = [0, 0, 0, 0] pad\_token\_label = -100 only\_label\_first\_subword = True tokenize\_chinese\_chars = True strip\_accents = None \*\*kwargs  )

Parameters

* **vocab\_file** (`str`) —
  File containing the vocabulary.
* **do\_lower\_case** (`bool`, *optional*, defaults to `True`) —
  Whether or not to lowercase the input when tokenizing.
* **unk\_token** (`str`, *optional*, defaults to `"[UNK]"`) —
  The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
  token instead.
* **sep\_token** (`str`, *optional*, defaults to `"[SEP]"`) —
  The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
  sequence classification or for a text and a question for question answering. It is also used as the last
  token of a sequence built with special tokens.
* **pad\_token** (`str`, *optional*, defaults to `"[PAD]"`) —
  The token used for padding, for example when batching sequences of different lengths.
* **cls\_token** (`str`, *optional*, defaults to `"[CLS]"`) —
  The classifier token which is used when doing sequence classification (classification of the whole sequence
  instead of per-token classification). It is the first token of the sequence when built with special tokens.
* **mask\_token** (`str`, *optional*, defaults to `"[MASK]"`) —
  The token used for masking values. This is the token used when training this model with masked language
  modeling. This is the token which the model will try to predict.
* **cls\_token\_box** (`List[int]`, *optional*, defaults to `[0, 0, 0, 0]`) —
  The bounding box to use for the special [CLS] token.
* **sep\_token\_box** (`List[int]`, *optional*, defaults to `[1000, 1000, 1000, 1000]`) —
  The bounding box to use for the special [SEP] token.
* **pad\_token\_box** (`List[int]`, *optional*, defaults to `[0, 0, 0, 0]`) —
  The bounding box to use for the special [PAD] token.
* **pad\_token\_label** (`int`, *optional*, defaults to -100) —
  The label to use for padding tokens. Defaults to -100, which is the `ignore_index` of PyTorch’s
  CrossEntropyLoss.
* **only\_label\_first\_subword** (`bool`, *optional*, defaults to `True`) —
  Whether or not to only label the first subword, in case word labels are provided.
* **tokenize\_chinese\_chars** (`bool`, *optional*, defaults to `True`) —
  Whether or not to tokenize Chinese characters. This should likely be deactivated for Japanese (see [this
  issue](https://github.com/huggingface/transformers/issues/328)).
* **strip\_accents** (`bool`, *optional*) —
  Whether or not to strip all accents. If this option is not specified, then it will be determined by the
  value for `lowercase` (as in the original LayoutLMv2).

Construct a “fast” LayoutLMv2 tokenizer (backed by HuggingFace’s *tokenizers* library). Based on WordPiece.

This tokenizer inherits from [PreTrainedTokenizerFast](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast) which contains most of the main methods. Users should
refer to this superclass for more information regarding those methods.

#### \_\_call\_\_

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/layoutlmv2/tokenization_layoutlmv2_fast.py#L155)

( text: typing.Union[str, list[str], list[list[str]]] text\_pair: typing.Union[list[str], list[list[str]], NoneType] = None boxes: typing.Union[list[list[int]], list[list[list[int]]], NoneType] = None word\_labels: typing.Union[list[int], list[list[int]], NoneType] = None add\_special\_tokens: bool = True padding: typing.Union[bool, str, transformers.utils.generic.PaddingStrategy] = False truncation: typing.Union[bool, str, transformers.tokenization\_utils\_base.TruncationStrategy] = None max\_length: typing.Optional[int] = None stride: int = 0 pad\_to\_multiple\_of: typing.Optional[int] = None padding\_side: typing.Optional[str] = None return\_tensors: typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None return\_token\_type\_ids: typing.Optional[bool] = None return\_attention\_mask: typing.Optional[bool] = None return\_overflowing\_tokens: bool = False return\_special\_tokens\_mask: bool = False return\_offsets\_mapping: bool = False return\_length: bool = False verbose: bool = True \*\*kwargs  ) → [BatchEncoding](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.BatchEncoding)

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
* **return\_token\_type\_ids** (`bool`, *optional*) —
  Whether to return token type IDs. If left to the default, will return the token type IDs according to
  the specific tokenizer’s default, defined by the `return_outputs` attribute.

  [What are token type IDs?](../glossary#token-type-ids)
* **return\_attention\_mask** (`bool`, *optional*) —
  Whether to return the attention mask. If left to the default, will return the attention mask according
  to the specific tokenizer’s default, defined by the `return_outputs` attribute.

  [What are attention masks?](../glossary#attention-mask)
* **return\_overflowing\_tokens** (`bool`, *optional*, defaults to `False`) —
  Whether or not to return overflowing token sequences. If a pair of sequences of input ids (or a batch
  of pairs) is provided with `truncation_strategy = longest_first` or `True`, an error is raised instead
  of returning overflowing tokens.
* **return\_special\_tokens\_mask** (`bool`, *optional*, defaults to `False`) —
  Whether or not to return special tokens mask information.
* **return\_offsets\_mapping** (`bool`, *optional*, defaults to `False`) —
  Whether or not to return `(char_start, char_end)` for each token.

  This is only available on fast tokenizers inheriting from [PreTrainedTokenizerFast](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast), if using
  Python’s tokenizer, this method will raise `NotImplementedError`.
* **return\_length** (`bool`, *optional*, defaults to `False`) —
  Whether or not to return the lengths of the encoded inputs.
* **verbose** (`bool`, *optional*, defaults to `True`) —
  Whether or not to print more information and warnings.
* \***\*kwargs** — passed to the `self.tokenize()` method

Returns

[BatchEncoding](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.BatchEncoding)

A [BatchEncoding](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.BatchEncoding) with the following fields:

* **input\_ids** — List of token ids to be fed to a model.

  [What are input IDs?](../glossary#input-ids)
* **bbox** — List of bounding boxes to be fed to a model.
* **token\_type\_ids** — List of token type ids to be fed to a model (when `return_token_type_ids=True` or
  if *“token\_type\_ids”* is in `self.model_input_names`).

  [What are token type IDs?](../glossary#token-type-ids)
* **attention\_mask** — List of indices specifying which tokens should be attended to by the model (when
  `return_attention_mask=True` or if *“attention\_mask”* is in `self.model_input_names`).

  [What are attention masks?](../glossary#attention-mask)
* **labels** — List of labels to be fed to a model. (when `word_labels` is specified).
* **overflowing\_tokens** — List of overflowing tokens sequences (when a `max_length` is specified and
  `return_overflowing_tokens=True`).
* **num\_truncated\_tokens** — Number of tokens truncated (when a `max_length` is specified and
  `return_overflowing_tokens=True`).
* **special\_tokens\_mask** — List of 0s and 1s, with 1 specifying added special tokens and 0 specifying
  regular sequence tokens (when `add_special_tokens=True` and `return_special_tokens_mask=True`).
* **length** — The length of the inputs (when `return_length=True`).

Main method to tokenize and prepare for the model one or several sequence(s) or one or several pair(s) of
sequences with word-level normalized bounding boxes and optional labels.

## LayoutLMv2Processor

### class transformers.LayoutLMv2Processor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/layoutlmv2/processing_layoutlmv2.py#L27)

( image\_processor = None tokenizer = None \*\*kwargs  )

Parameters

* **image\_processor** (`LayoutLMv2ImageProcessor`, *optional*) —
  An instance of [LayoutLMv2ImageProcessor](/docs/transformers/v4.56.2/en/model_doc/layoutlmv2#transformers.LayoutLMv2ImageProcessor). The image processor is a required input.
* **tokenizer** (`LayoutLMv2Tokenizer` or `LayoutLMv2TokenizerFast`, *optional*) —
  An instance of [LayoutLMv2Tokenizer](/docs/transformers/v4.56.2/en/model_doc/layoutlmv2#transformers.LayoutLMv2Tokenizer) or [LayoutLMv2TokenizerFast](/docs/transformers/v4.56.2/en/model_doc/layoutlmv2#transformers.LayoutLMv2TokenizerFast). The tokenizer is a required input.

Constructs a LayoutLMv2 processor which combines a LayoutLMv2 image processor and a LayoutLMv2 tokenizer into a
single processor.

[LayoutLMv2Processor](/docs/transformers/v4.56.2/en/model_doc/layoutlmv2#transformers.LayoutLMv2Processor) offers all the functionalities you need to prepare data for the model.

It first uses [LayoutLMv2ImageProcessor](/docs/transformers/v4.56.2/en/model_doc/layoutlmv2#transformers.LayoutLMv2ImageProcessor) to resize document images to a fixed size, and optionally applies OCR to
get words and normalized bounding boxes. These are then provided to [LayoutLMv2Tokenizer](/docs/transformers/v4.56.2/en/model_doc/layoutlmv2#transformers.LayoutLMv2Tokenizer) or
[LayoutLMv2TokenizerFast](/docs/transformers/v4.56.2/en/model_doc/layoutlmv2#transformers.LayoutLMv2TokenizerFast), which turns the words and bounding boxes into token-level `input_ids`,
`attention_mask`, `token_type_ids`, `bbox`. Optionally, one can provide integer `word_labels`, which are turned
into token-level `labels` for token classification tasks (such as FUNSD, CORD).

#### \_\_call\_\_

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/layoutlmv2/processing_layoutlmv2.py#L69)

( images text: typing.Union[str, list[str], list[list[str]]] = None text\_pair: typing.Union[list[str], list[list[str]], NoneType] = None boxes: typing.Union[list[list[int]], list[list[list[int]]], NoneType] = None word\_labels: typing.Union[list[int], list[list[int]], NoneType] = None add\_special\_tokens: bool = True padding: typing.Union[bool, str, transformers.utils.generic.PaddingStrategy] = False truncation: typing.Union[bool, str, transformers.tokenization\_utils\_base.TruncationStrategy] = False max\_length: typing.Optional[int] = None stride: int = 0 pad\_to\_multiple\_of: typing.Optional[int] = None return\_token\_type\_ids: typing.Optional[bool] = None return\_attention\_mask: typing.Optional[bool] = None return\_overflowing\_tokens: bool = False return\_special\_tokens\_mask: bool = False return\_offsets\_mapping: bool = False return\_length: bool = False verbose: bool = True return\_tensors: typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None \*\*kwargs  )

This method first forwards the `images` argument to [**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__). In case
[LayoutLMv2ImageProcessor](/docs/transformers/v4.56.2/en/model_doc/layoutlmv2#transformers.LayoutLMv2ImageProcessor) was initialized with `apply_ocr` set to `True`, it passes the obtained words and
bounding boxes along with the additional arguments to [**call**()](/docs/transformers/v4.56.2/en/model_doc/layoutlmv2#transformers.LayoutLMv2Tokenizer.__call__) and returns the output,
together with resized `images`. In case [LayoutLMv2ImageProcessor](/docs/transformers/v4.56.2/en/model_doc/layoutlmv2#transformers.LayoutLMv2ImageProcessor) was initialized with `apply_ocr` set to
`False`, it passes the words (`text`/`` text_pair`) and `boxes` specified by the user along with the additional arguments to [__call__()](/docs/transformers/v4.56.2/en/model_doc/layoutlmv2#transformers.LayoutLMv2Tokenizer.__call__) and returns the output, together with resized `images ``.

Please refer to the docstring of the above two methods for more information.

## LayoutLMv2Model

### class transformers.LayoutLMv2Model

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/layoutlmv2/modeling_layoutlmv2.py#L604)

( config  )

Parameters

* **config** ([LayoutLMv2Model](/docs/transformers/v4.56.2/en/model_doc/layoutlmv2#transformers.LayoutLMv2Model)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Layoutlmv2 Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/layoutlmv2/modeling_layoutlmv2.py#L714)

( input\_ids: typing.Optional[torch.LongTensor] = None bbox: typing.Optional[torch.LongTensor] = None image: typing.Optional[torch.FloatTensor] = None attention\_mask: typing.Optional[torch.FloatTensor] = None token\_type\_ids: typing.Optional[torch.LongTensor] = None position\_ids: typing.Optional[torch.LongTensor] = None head\_mask: typing.Optional[torch.FloatTensor] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → [transformers.modeling\_outputs.BaseModelOutputWithPooling](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **bbox** (`torch.LongTensor` of shape `((batch_size, sequence_length), 4)`, *optional*) —
  Bounding boxes of each input sequence tokens. Selected in the range `[0, config.max_2d_position_embeddings-1]`. Each bounding box should be a normalized version in (x0, y0, x1, y1)
  format, where (x0, y0) corresponds to the position of the upper left corner in the bounding box, and (x1,
  y1) represents the position of the lower right corner.
* **image** (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` or `detectron.structures.ImageList` whose `tensors` is of shape `(batch_size, num_channels, height, width)`) —
  Batch of document images.
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
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

[transformers.modeling\_outputs.BaseModelOutputWithPooling](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.BaseModelOutputWithPooling](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([LayoutLMv2Config](/docs/transformers/v4.56.2/en/model_doc/layoutlmv2#transformers.LayoutLMv2Config)) and inputs.

* **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the model.
* **pooler\_output** (`torch.FloatTensor` of shape `(batch_size, hidden_size)`) — Last layer hidden-state of the first token of the sequence (classification token) after further processing
  through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns
  the classification token after processing through a linear layer and a tanh activation function. The linear
  layer weights are trained from the next sentence prediction (classification) objective during pretraining.
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [LayoutLMv2Model](/docs/transformers/v4.56.2/en/model_doc/layoutlmv2#transformers.LayoutLMv2Model) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> from transformers import AutoProcessor, LayoutLMv2Model, set_seed
>>> from PIL import Image
>>> import torch
>>> from datasets import load_dataset

>>> set_seed(0)

>>> processor = AutoProcessor.from_pretrained("microsoft/layoutlmv2-base-uncased")
>>> model = LayoutLMv2Model.from_pretrained("microsoft/layoutlmv2-base-uncased")


>>> dataset = load_dataset("hf-internal-testing/fixtures_docvqa")
>>> image = dataset["test"][0]["image"]

>>> encoding = processor(image, return_tensors="pt")

>>> outputs = model(**encoding)
>>> last_hidden_states = outputs.last_hidden_state

>>> last_hidden_states.shape
torch.Size([1, 342, 768])
```

## LayoutLMv2ForSequenceClassification

### class transformers.LayoutLMv2ForSequenceClassification

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/layoutlmv2/modeling_layoutlmv2.py#L868)

( config  )

Parameters

* **config** ([LayoutLMv2ForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/layoutlmv2#transformers.LayoutLMv2ForSequenceClassification)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

LayoutLMv2 Model with a sequence classification head on top (a linear layer on top of the concatenation of the
final hidden state of the [CLS] token, average-pooled initial visual embeddings and average-pooled final visual
embeddings, e.g. for document image classification tasks such as the
[RVL-CDIP](https://www.cs.cmu.edu/~aharley/rvl-cdip/) dataset.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/layoutlmv2/modeling_layoutlmv2.py#L882)

( input\_ids: typing.Optional[torch.LongTensor] = None bbox: typing.Optional[torch.LongTensor] = None image: typing.Optional[torch.FloatTensor] = None attention\_mask: typing.Optional[torch.FloatTensor] = None token\_type\_ids: typing.Optional[torch.LongTensor] = None position\_ids: typing.Optional[torch.LongTensor] = None head\_mask: typing.Optional[torch.FloatTensor] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None labels: typing.Optional[torch.LongTensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → [transformers.modeling\_outputs.SequenceClassifierOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput) or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `batch_size, sequence_length`) —
  Indices of input sequence tokens in the vocabulary.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **bbox** (`torch.LongTensor` of shape `(batch_size, sequence_length, 4)`, *optional*) —
  Bounding boxes of each input sequence tokens. Selected in the range `[0, config.max_2d_position_embeddings-1]`. Each bounding box should be a normalized version in (x0, y0, x1, y1)
  format, where (x0, y0) corresponds to the position of the upper left corner in the bounding box, and (x1,
  y1) represents the position of the lower right corner.
* **image** (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` or `detectron.structures.ImageList` whose `tensors` is of shape `(batch_size, num_channels, height, width)`) —
  Batch of document images.
* **attention\_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **token\_type\_ids** (`torch.LongTensor` of shape `batch_size, sequence_length`, *optional*) —
  Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
  + 0 corresponds to a *sentence A* token,
  + 1 corresponds to a *sentence B* token.

  [What are token type IDs?](../glossary#token-type-ids)
* **position\_ids** (`torch.LongTensor` of shape `batch_size, sequence_length`, *optional*) —
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.max_position_embeddings - 1]`.

  [What are position IDs?](../glossary#position-ids)
* **head\_mask** (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model’s internal embedding lookup matrix.
* **labels** (`torch.LongTensor` of shape `(batch_size,)`, *optional*) —
  Labels for computing the sequence classification/regression loss. Indices should be in `[0, ..., config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
  `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

[transformers.modeling\_outputs.SequenceClassifierOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.SequenceClassifierOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([LayoutLMv2Config](/docs/transformers/v4.56.2/en/model_doc/layoutlmv2#transformers.LayoutLMv2Config)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Classification (or regression if config.num\_labels==1) loss.
* **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) — Classification (or regression if config.num\_labels==1) scores (before SoftMax).
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [LayoutLMv2ForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/layoutlmv2#transformers.LayoutLMv2ForSequenceClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> from transformers import AutoProcessor, LayoutLMv2ForSequenceClassification, set_seed
>>> from PIL import Image
>>> import torch
>>> from datasets import load_dataset

>>> set_seed(0)

>>> dataset = load_dataset("aharley/rvl_cdip", split="train", streaming=True)
>>> data = next(iter(dataset))
>>> image = data["image"].convert("RGB")

>>> processor = AutoProcessor.from_pretrained("microsoft/layoutlmv2-base-uncased")
>>> model = LayoutLMv2ForSequenceClassification.from_pretrained(
...     "microsoft/layoutlmv2-base-uncased", num_labels=dataset.info.features["label"].num_classes
... )

>>> encoding = processor(image, return_tensors="pt")
>>> sequence_label = torch.tensor([data["label"]])

>>> outputs = model(**encoding, labels=sequence_label)

>>> loss, logits = outputs.loss, outputs.logits
>>> predicted_idx = logits.argmax(dim=-1).item()
>>> predicted_answer = dataset.info.features["label"].names[4]
>>> predicted_idx, predicted_answer  # results are not good without further fine-tuning
(7, 'advertisement')
```

## LayoutLMv2ForTokenClassification

### class transformers.LayoutLMv2ForTokenClassification

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/layoutlmv2/modeling_layoutlmv2.py#L1073)

( config  )

Parameters

* **config** ([LayoutLMv2ForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/layoutlmv2#transformers.LayoutLMv2ForTokenClassification)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

LayoutLMv2 Model with a token classification head on top (a linear layer on top of the text part of the hidden
states) e.g. for sequence labeling (information extraction) tasks such as
[FUNSD](https://guillaumejaume.github.io/FUNSD/), [SROIE](https://rrc.cvc.uab.es/?ch=13),
[CORD](https://github.com/clovaai/cord) and [Kleister-NDA](https://github.com/applicaai/kleister-nda).

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/layoutlmv2/modeling_layoutlmv2.py#L1087)

( input\_ids: typing.Optional[torch.LongTensor] = None bbox: typing.Optional[torch.LongTensor] = None image: typing.Optional[torch.FloatTensor] = None attention\_mask: typing.Optional[torch.FloatTensor] = None token\_type\_ids: typing.Optional[torch.LongTensor] = None position\_ids: typing.Optional[torch.LongTensor] = None head\_mask: typing.Optional[torch.FloatTensor] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None labels: typing.Optional[torch.LongTensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → [transformers.modeling\_outputs.TokenClassifierOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput) or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `batch_size, sequence_length`) —
  Indices of input sequence tokens in the vocabulary.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **bbox** (`torch.LongTensor` of shape `(batch_size, sequence_length, 4)`, *optional*) —
  Bounding boxes of each input sequence tokens. Selected in the range `[0, config.max_2d_position_embeddings-1]`. Each bounding box should be a normalized version in (x0, y0, x1, y1)
  format, where (x0, y0) corresponds to the position of the upper left corner in the bounding box, and (x1,
  y1) represents the position of the lower right corner.
* **image** (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` or `detectron.structures.ImageList` whose `tensors` is of shape `(batch_size, num_channels, height, width)`) —
  Batch of document images.
* **attention\_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **token\_type\_ids** (`torch.LongTensor` of shape `batch_size, sequence_length`, *optional*) —
  Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
  + 0 corresponds to a *sentence A* token,
  + 1 corresponds to a *sentence B* token.

  [What are token type IDs?](../glossary#token-type-ids)
* **position\_ids** (`torch.LongTensor` of shape `batch_size, sequence_length`, *optional*) —
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.max_position_embeddings - 1]`.

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

Returns

[transformers.modeling\_outputs.TokenClassifierOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.TokenClassifierOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([LayoutLMv2Config](/docs/transformers/v4.56.2/en/model_doc/layoutlmv2#transformers.LayoutLMv2Config)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Classification loss.
* **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_labels)`) — Classification scores (before SoftMax).
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [LayoutLMv2ForTokenClassification](/docs/transformers/v4.56.2/en/model_doc/layoutlmv2#transformers.LayoutLMv2ForTokenClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> from transformers import AutoProcessor, LayoutLMv2ForTokenClassification, set_seed
>>> from PIL import Image
>>> from datasets import load_dataset

>>> set_seed(0)

>>> datasets = load_dataset("nielsr/funsd", split="test")
>>> labels = datasets.features["ner_tags"].feature.names
>>> id2label = {v: k for v, k in enumerate(labels)}

>>> processor = AutoProcessor.from_pretrained("microsoft/layoutlmv2-base-uncased", revision="no_ocr")
>>> model = LayoutLMv2ForTokenClassification.from_pretrained(
...     "microsoft/layoutlmv2-base-uncased", num_labels=len(labels)
... )

>>> data = datasets[0]
>>> image = Image.open(data["image_path"]).convert("RGB")
>>> words = data["words"]
>>> boxes = data["bboxes"]  # make sure to normalize your bounding boxes
>>> word_labels = data["ner_tags"]
>>> encoding = processor(
...     image,
...     words,
...     boxes=boxes,
...     word_labels=word_labels,
...     padding="max_length",
...     truncation=True,
...     return_tensors="pt",
... )

>>> outputs = model(**encoding)
>>> logits, loss = outputs.logits, outputs.loss

>>> predicted_token_class_ids = logits.argmax(-1)
>>> predicted_tokens_classes = [id2label[t.item()] for t in predicted_token_class_ids[0]]
>>> predicted_tokens_classes[:5]  # results are not good without further fine-tuning
['I-HEADER', 'I-HEADER', 'I-QUESTION', 'I-HEADER', 'I-QUESTION']
```

## LayoutLMv2ForQuestionAnswering

### class transformers.LayoutLMv2ForQuestionAnswering

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/layoutlmv2/modeling_layoutlmv2.py#L1221)

( config has\_visual\_segment\_embedding = True  )

Parameters

* **config** ([LayoutLMv2ForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/layoutlmv2#transformers.LayoutLMv2ForQuestionAnswering)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.
* **has\_visual\_segment\_embedding** (`bool`, *optional*, defaults to `True`) —
  Whether or not to add visual segment embeddings.

The Layoutlmv2 transformer with a span classification head on top for extractive question-answering tasks like
SQuAD (a linear layer on top of the hidden-states output to compute `span start logits` and `span end logits`).

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/layoutlmv2/modeling_layoutlmv2.py#L1239)

( input\_ids: typing.Optional[torch.LongTensor] = None bbox: typing.Optional[torch.LongTensor] = None image: typing.Optional[torch.FloatTensor] = None attention\_mask: typing.Optional[torch.FloatTensor] = None token\_type\_ids: typing.Optional[torch.LongTensor] = None position\_ids: typing.Optional[torch.LongTensor] = None head\_mask: typing.Optional[torch.FloatTensor] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None start\_positions: typing.Optional[torch.LongTensor] = None end\_positions: typing.Optional[torch.LongTensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → [transformers.modeling\_outputs.QuestionAnsweringModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.QuestionAnsweringModelOutput) or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `batch_size, sequence_length`) —
  Indices of input sequence tokens in the vocabulary.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **bbox** (`torch.LongTensor` of shape `(batch_size, sequence_length, 4)`, *optional*) —
  Bounding boxes of each input sequence tokens. Selected in the range `[0, config.max_2d_position_embeddings-1]`. Each bounding box should be a normalized version in (x0, y0, x1, y1)
  format, where (x0, y0) corresponds to the position of the upper left corner in the bounding box, and (x1,
  y1) represents the position of the lower right corner.
* **image** (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` or `detectron.structures.ImageList` whose `tensors` is of shape `(batch_size, num_channels, height, width)`) —
  Batch of document images.
* **attention\_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **token\_type\_ids** (`torch.LongTensor` of shape `batch_size, sequence_length`, *optional*) —
  Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
  + 0 corresponds to a *sentence A* token,
  + 1 corresponds to a *sentence B* token.

  [What are token type IDs?](../glossary#token-type-ids)
* **position\_ids** (`torch.LongTensor` of shape `batch_size, sequence_length`, *optional*) —
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.max_position_embeddings - 1]`.

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

Returns

[transformers.modeling\_outputs.QuestionAnsweringModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.QuestionAnsweringModelOutput) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.QuestionAnsweringModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.QuestionAnsweringModelOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([LayoutLMv2Config](/docs/transformers/v4.56.2/en/model_doc/layoutlmv2#transformers.LayoutLMv2Config)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
* **start\_logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`) — Span-start scores (before SoftMax).
* **end\_logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`) — Span-end scores (before SoftMax).
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [LayoutLMv2ForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/layoutlmv2#transformers.LayoutLMv2ForQuestionAnswering) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:

In this example below, we give the LayoutLMv2 model an image (of texts) and ask it a question. It will give us
a prediction of what it thinks the answer is (the span of the answer within the texts parsed from the image).


```
>>> from transformers import AutoProcessor, LayoutLMv2ForQuestionAnswering, set_seed
>>> import torch
>>> from PIL import Image
>>> from datasets import load_dataset

>>> set_seed(0)
>>> processor = AutoProcessor.from_pretrained("microsoft/layoutlmv2-base-uncased")
>>> model = LayoutLMv2ForQuestionAnswering.from_pretrained("microsoft/layoutlmv2-base-uncased")

>>> dataset = load_dataset("hf-internal-testing/fixtures_docvqa")
>>> image = dataset["test"][0]["image"]
>>> question = "When is coffee break?"
>>> encoding = processor(image, question, return_tensors="pt")

>>> outputs = model(**encoding)
>>> predicted_start_idx = outputs.start_logits.argmax(-1).item()
>>> predicted_end_idx = outputs.end_logits.argmax(-1).item()
>>> predicted_start_idx, predicted_end_idx
(30, 191)

>>> predicted_answer_tokens = encoding.input_ids.squeeze()[predicted_start_idx : predicted_end_idx + 1]
>>> predicted_answer = processor.tokenizer.decode(predicted_answer_tokens)
>>> predicted_answer  # results are not good without further fine-tuning
'44 a. m. to 12 : 25 p. m. 12 : 25 to 12 : 58 p. m. 12 : 58 to 4 : 00 p. m. 2 : 00 to 5 : 00 p. m. coffee break coffee will be served for men and women in the lobby adjacent to exhibit area. please move into exhibit area. ( exhibits open ) trrf general session ( part | ) presiding : lee a. waller trrf vice president “ introductory remarks ” lee a. waller, trrf vice presi - dent individual interviews with trrf public board members and sci - entific advisory council mem - bers conducted by trrf treasurer philip g. kuehn to get answers which the public refrigerated warehousing industry is looking for. plus questions from'
```


```
>>> target_start_index = torch.tensor([7])
>>> target_end_index = torch.tensor([14])
>>> outputs = model(**encoding, start_positions=target_start_index, end_positions=target_end_index)
>>> predicted_answer_span_start = outputs.start_logits.argmax(-1).item()
>>> predicted_answer_span_end = outputs.end_logits.argmax(-1).item()
>>> predicted_answer_span_start, predicted_answer_span_end
(30, 191)
```

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/layoutlmv2.md)
