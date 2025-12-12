# Processors

Processors can mean two different things in the Transformers library:

* the objects that pre-process inputs for multi-modal models such as [Wav2Vec2](../model_doc/wav2vec2) (speech and text)
  or [CLIP](../model_doc/clip) (text and vision)
* deprecated objects that were used in older versions of the library to preprocess data for GLUE or SQUAD.

## Multi-modal processors

Any multi-modal model will require an object to encode or decode the data that groups several modalities (among text,
vision and audio). This is handled by objects called processors, which group together two or more processing objects
such as tokenizers (for the text modality), image processors (for vision) and feature extractors (for audio).

Those processors inherit from the following base class that implements the saving and loading functionality:

### class transformers.ProcessorMixin

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/processing_utils.py#L498)

( \*args \*\*kwargs  )

This is a mixin used to provide saving/loading functionality for all processor classes.

#### apply\_chat\_template

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/processing_utils.py#L1459)

( conversation: typing.Union[list[dict[str, str]], list[list[dict[str, str]]]] chat\_template: typing.Optional[str] = None \*\*kwargs: typing\_extensions.Unpack[transformers.processing\_utils.AllKwargsForChatTemplate]  )

Parameters

* **conversation** (`Union[list[Dict, [str, str]], list[list[dict[str, str]]]]`) —
  The conversation to format.
* **chat\_template** (`Optional[str]`, *optional*) —
  The Jinja template to use for formatting the conversation. If not provided, the tokenizer’s
  chat template is used.

Similar to the `apply_chat_template` method on tokenizers, this method applies a Jinja template to input
conversations to turn them into a single tokenizable string.

The input is expected to be in the following format, where each message content is a list consisting of text and
optionally image or video inputs. One can also provide an image, video, URL or local path which will be used to form
`pixel_values` when `return_dict=True`. If not provided, one will get only the formatted text, optionally tokenized text.

conversation = [
{
“role”: “user”,
“content”: [
{“type”: “image”, “url”: “https://www.ilankelman.org/stopsigns/australia.jpg”},
{“type”: “text”, “text”: “Please describe this image in detail.”},
],
},
]

#### batch\_decode

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/processing_utils.py#L1419)

( \*args \*\*kwargs  )

This method forwards all its arguments to PreTrainedTokenizer’s [batch\_decode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.batch_decode). Please
refer to the docstring of this method for more information.

#### check\_argument\_for\_proper\_class

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/processing_utils.py#L550)

( argument\_name argument  )

Checks the passed argument’s class against the expected transformers class. In case of an unexpected
mismatch between expected and actual class, an error is raise. Otherwise, the proper retrieved class
is returned.

#### decode

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/processing_utils.py#L1428)

( \*args \*\*kwargs  )

This method forwards all its arguments to PreTrainedTokenizer’s [decode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.decode). Please refer to
the docstring of this method for more information.

#### from\_args\_and\_dict

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/processing_utils.py#L1086)

( args processor\_dict: dict \*\*kwargs  ) → `~processing_utils.ProcessingMixin`

Parameters

* **processor\_dict** (`dict[str, Any]`) —
  Dictionary that will be used to instantiate the processor object. Such a dictionary can be
  retrieved from a pretrained checkpoint by leveraging the
  `~processing_utils.ProcessingMixin.to_dict` method.
* **kwargs** (`dict[str, Any]`) —
  Additional parameters from which to initialize the processor object.

Returns

`~processing_utils.ProcessingMixin`

The processor object instantiated from those
parameters.

Instantiates a type of `~processing_utils.ProcessingMixin` from a Python dictionary of parameters.

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

#### get\_processor\_dict

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/processing_utils.py#L828)

( pretrained\_model\_name\_or\_path: typing.Union[str, os.PathLike] \*\*kwargs  ) → `tuple[Dict, Dict]`

Parameters

* **pretrained\_model\_name\_or\_path** (`str` or `os.PathLike`) —
  The identifier of the pre-trained checkpoint from which we want the dictionary of parameters.
* **subfolder** (`str`, *optional*, defaults to `""`) —
  In case the relevant files are located inside a subfolder of the model repo on huggingface.co, you can
  specify the folder name here.

Returns

`tuple[Dict, Dict]`

The dictionary(ies) that will be used to instantiate the processor object.

From a `pretrained_model_name_or_path`, resolve to a dictionary of parameters, to be used for instantiating a
processor of type `~processing_utils.ProcessingMixin` using `from_args_and_dict`.

#### post\_process\_image\_text\_to\_text

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/processing_utils.py#L1674)

( generated\_outputs skip\_special\_tokens = True \*\*kwargs  ) → `list[str]`

Parameters

* **generated\_outputs** (`torch.Tensor` or `np.ndarray`) —
  The output of the model `generate` function. The output is expected to be a tensor of shape `(batch_size, sequence_length)`
  or `(sequence_length,)`.
* **skip\_special\_tokens** (`bool`, *optional*, defaults to `True`) —
  Whether or not to remove special tokens in the output. Argument passed to the tokenizer’s `batch_decode` method.
* \***\*kwargs** —
  Additional arguments to be passed to the tokenizer’s `batch_decode method`.

Returns

`list[str]`

The decoded text.

Post-process the output of a vlm to decode the text.

#### push\_to\_hub

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/utils/hub.py#L847)

( repo\_id: str use\_temp\_dir: typing.Optional[bool] = None commit\_message: typing.Optional[str] = None private: typing.Optional[bool] = None token: typing.Union[bool, str, NoneType] = None max\_shard\_size: typing.Union[str, int, NoneType] = '5GB' create\_pr: bool = False safe\_serialization: bool = True revision: typing.Optional[str] = None commit\_description: typing.Optional[str] = None tags: typing.Optional[list[str]] = None \*\*deprecated\_kwargs  )

Parameters

* **repo\_id** (`str`) —
  The name of the repository you want to push your processor to. It should contain your organization name
  when pushing to a given organization.
* **use\_temp\_dir** (`bool`, *optional*) —
  Whether or not to use a temporary directory to store the files saved before they are pushed to the Hub.
  Will default to `True` if there is no directory named like `repo_id`, `False` otherwise.
* **commit\_message** (`str`, *optional*) —
  Message to commit while pushing. Will default to `"Upload processor"`.
* **private** (`bool`, *optional*) —
  Whether to make the repo private. If `None` (default), the repo will be public unless the organization’s default is private. This value is ignored if the repo already exists.
* **token** (`bool` or `str`, *optional*) —
  The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
  when running `hf auth login` (stored in `~/.huggingface`). Will default to `True` if `repo_url`
  is not specified.
* **max\_shard\_size** (`int` or `str`, *optional*, defaults to `"5GB"`) —
  Only applicable for models. The maximum size for a checkpoint before being sharded. Checkpoints shard
  will then be each of size lower than this size. If expressed as a string, needs to be digits followed
  by a unit (like `"5MB"`). We default it to `"5GB"` so that users can easily load models on free-tier
  Google Colab instances without any CPU OOM issues.
* **create\_pr** (`bool`, *optional*, defaults to `False`) —
  Whether or not to create a PR with the uploaded files or directly commit.
* **safe\_serialization** (`bool`, *optional*, defaults to `True`) —
  Whether or not to convert the model weights in safetensors format for safer serialization.
* **revision** (`str`, *optional*) —
  Branch to push the uploaded files to.
* **commit\_description** (`str`, *optional*) —
  The description of the commit that will be created
* **tags** (`list[str]`, *optional*) —
  List of tags to push on the Hub.

Upload the processor files to the 🤗 Model Hub.

Examples:


```
from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained("google-bert/bert-base-cased")

# Push the processor to your namespace with the name "my-finetuned-bert".
processor.push_to_hub("my-finetuned-bert")

# Push the processor to an organization with the name "my-finetuned-bert".
processor.push_to_hub("huggingface/my-finetuned-bert")
```

#### register\_for\_auto\_class

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/processing_utils.py#L1335)

( auto\_class = 'AutoProcessor'  )

Parameters

* **auto\_class** (`str` or `type`, *optional*, defaults to `"AutoProcessor"`) —
  The auto class to register this new feature extractor with.

Register this class with a given auto class. This should only be used for custom feature extractors as the ones
in the library are already mapped with `AutoProcessor`.

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

#### to\_dict

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/processing_utils.py#L571)

( legacy\_serialization = True  ) → `dict[str, Any]`

Returns

`dict[str, Any]`

Dictionary of all the attributes that make up this processor instance.

Serializes this instance to a Python dictionary.

#### to\_json\_file

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/processing_utils.py#L637)

( json\_file\_path: typing.Union[str, os.PathLike] legacy\_serialization = True  )

Parameters

* **json\_file\_path** (`str` or `os.PathLike`) —
  Path to the JSON file in which this processor instance’s parameters will be saved.

Save this instance to a JSON file.

#### to\_json\_string

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/processing_utils.py#L626)

( legacy\_serialization = True  ) → `str`

Returns

`str`

String containing all the attributes that make up this feature\_extractor instance in JSON format.

Serializes this instance to a JSON string.

## Deprecated processors

All processors follow the same architecture which is that of the
[DataProcessor](/docs/transformers/v4.56.2/en/main_classes/processors#transformers.DataProcessor). The processor returns a list of
[InputExample](/docs/transformers/v4.56.2/en/main_classes/processors#transformers.InputExample). These
[InputExample](/docs/transformers/v4.56.2/en/main_classes/processors#transformers.InputExample) can be converted to
[InputFeatures](/docs/transformers/v4.56.2/en/main_classes/processors#transformers.InputFeatures) in order to be fed to the model.

### class transformers.DataProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/data/processors/utils.py#L80)

( )

Base class for data converters for sequence classification data sets.

#### get\_dev\_examples

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/data/processors/utils.py#L97)

( data\_dir  )

Gets a collection of [InputExample](/docs/transformers/v4.56.2/en/main_classes/processors#transformers.InputExample) for the dev set.

#### get\_example\_from\_tensor\_dict

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/data/processors/utils.py#L83)

( tensor\_dict  )

Parameters

* **tensor\_dict** — Keys and values should match the corresponding Glue
  tensorflow\_dataset examples.

Gets an example from a dict with tensorflow tensors.

#### get\_labels

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/data/processors/utils.py#L105)

( )

Gets the list of labels for this data set.

#### get\_test\_examples

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/data/processors/utils.py#L101)

( data\_dir  )

Gets a collection of [InputExample](/docs/transformers/v4.56.2/en/main_classes/processors#transformers.InputExample) for the test set.

#### get\_train\_examples

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/data/processors/utils.py#L93)

( data\_dir  )

Gets a collection of [InputExample](/docs/transformers/v4.56.2/en/main_classes/processors#transformers.InputExample) for the train set.

#### tfds\_map

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/data/processors/utils.py#L109)

( example  )

Some tensorflow\_datasets datasets are not formatted the same way the GLUE datasets are. This method converts
examples to the correct format.

### class transformers.InputExample

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/data/processors/utils.py#L30)

( guid: str text\_a: str text\_b: typing.Optional[str] = None label: typing.Optional[str] = None  )

Parameters

* **guid** — Unique id for the example.
* **text\_a** — string. The untokenized text of the first sequence. For single
  sequence tasks, only this sequence must be specified.
* **text\_b** — (Optional) string. The untokenized text of the second sequence.
  Only must be specified for sequence pair tasks.
* **label** — (Optional) string. The label of the example. This should be
  specified for train and dev examples, but not for test examples.

A single training/test example for simple sequence classification.

#### to\_json\_string

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/data/processors/utils.py#L49)

( )

Serializes this instance to a JSON string.

### class transformers.InputFeatures

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/data/processors/utils.py#L55)

( input\_ids: list attention\_mask: typing.Optional[list[int]] = None token\_type\_ids: typing.Optional[list[int]] = None label: typing.Union[int, float, NoneType] = None  )

Parameters

* **input\_ids** — Indices of input sequence tokens in the vocabulary.
* **attention\_mask** — Mask to avoid performing attention on padding token indices.
  Mask values selected in `[0, 1]`: Usually `1` for tokens that are NOT MASKED, `0` for MASKED (padded)
  tokens.
* **token\_type\_ids** — (Optional) Segment token indices to indicate first and second
  portions of the inputs. Only some models use them.
* **label** — (Optional) Label corresponding to the input. Int for classification problems,
  float for regression problems.

A single set of features of data. Property names are the same names as the corresponding inputs to a model.

#### to\_json\_string

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/data/processors/utils.py#L75)

( )

Serializes this instance to a JSON string.

## GLUE

[General Language Understanding Evaluation (GLUE)](https://gluebenchmark.com/) is a benchmark that evaluates the
performance of models across a diverse set of existing NLU tasks. It was released together with the paper [GLUE: A
multi-task benchmark and analysis platform for natural language understanding](https://openreview.net/pdf?id=rJ4km2R5t7)

This library hosts a total of 10 processors for the following tasks: MRPC, MNLI, MNLI (mismatched), CoLA, SST2, STSB,
QQP, QNLI, RTE and WNLI.

Those processors are:

* `~data.processors.utils.MrpcProcessor`
* `~data.processors.utils.MnliProcessor`
* `~data.processors.utils.MnliMismatchedProcessor`
* `~data.processors.utils.Sst2Processor`
* `~data.processors.utils.StsbProcessor`
* `~data.processors.utils.QqpProcessor`
* `~data.processors.utils.QnliProcessor`
* `~data.processors.utils.RteProcessor`
* `~data.processors.utils.WnliProcessor`

Additionally, the following method can be used to load values from a data file and convert them to a list of
[InputExample](/docs/transformers/v4.56.2/en/main_classes/processors#transformers.InputExample).

#### transformers.glue\_convert\_examples\_to\_features

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/data/processors/glue.py#L41)

( examples: typing.Union[list[transformers.data.processors.utils.InputExample], ForwardRef('tf.data.Dataset')] tokenizer: PreTrainedTokenizer max\_length: typing.Optional[int] = None task = None label\_list = None output\_mode = None  )

Parameters

* **examples** — List of `InputExamples` or `tf.data.Dataset` containing the examples.
* **tokenizer** — Instance of a tokenizer that will tokenize the examples
* **max\_length** — Maximum example length. Defaults to the tokenizer’s max\_len
* **task** — GLUE task
* **label\_list** — List of labels. Can be obtained from the processor using the `processor.get_labels()` method
* **output\_mode** — String indicating the output mode. Either `regression` or `classification`

Loads a data file into a list of `InputFeatures`

## XNLI

[The Cross-Lingual NLI Corpus (XNLI)](https://www.nyu.edu/projects/bowman/xnli/) is a benchmark that evaluates the
quality of cross-lingual text representations. XNLI is crowd-sourced dataset based on [*MultiNLI*](http://www.nyu.edu/projects/bowman/multinli/): pairs of text are labeled with textual entailment annotations for 15
different languages (including both high-resource language such as English and low-resource languages such as Swahili).

It was released together with the paper [XNLI: Evaluating Cross-lingual Sentence Representations](https://huggingface.co/papers/1809.05053)

This library hosts the processor to load the XNLI data:

* `~data.processors.utils.XnliProcessor`

Please note that since the gold labels are available on the test set, evaluation is performed on the test set.

An example using these processors is given in the [run\_xnli.py](https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification/run_xnli.py) script.

## SQuAD

[The Stanford Question Answering Dataset (SQuAD)](https://rajpurkar.github.io/SQuAD-explorer//) is a benchmark that
evaluates the performance of models on question answering. Two versions are available, v1.1 and v2.0. The first version
(v1.1) was released together with the paper [SQuAD: 100,000+ Questions for Machine Comprehension of Text](https://huggingface.co/papers/1606.05250). The second version (v2.0) was released alongside the paper [Know What You Don’t
Know: Unanswerable Questions for SQuAD](https://huggingface.co/papers/1806.03822).

This library hosts a processor for each of the two versions:

### Processors

Those processors are:

* `~data.processors.utils.SquadV1Processor`
* `~data.processors.utils.SquadV2Processor`

They both inherit from the abstract class `~data.processors.utils.SquadProcessor`

### class transformers.data.processors.squad.SquadProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/data/processors/squad.py#L541)

( )

Processor for the SQuAD data set. overridden by SquadV1Processor and SquadV2Processor, used by the version 1.1 and
version 2.0 of SQuAD, respectively.

#### get\_dev\_examples

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/data/processors/squad.py#L629)

( data\_dir filename = None  )

Parameters

* **data\_dir** — Directory containing the data files used for training and evaluating.
* **filename** — None by default, specify this if the evaluation file has a different name than the original one
  which is `dev-v1.1.json` and `dev-v2.0.json` for squad versions 1.1 and 2.0 respectively.

Returns the evaluation example from the data directory.

#### get\_examples\_from\_dataset

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/data/processors/squad.py#L574)

( dataset evaluate = False  )

Parameters

* **dataset** — The tfds dataset loaded from *tensorflow\_datasets.load(“squad”)*
* **evaluate** — Boolean specifying if in evaluation mode or in training mode

Creates a list of `SquadExample` using a TFDS dataset.

Examples:


```
>>> import tensorflow_datasets as tfds

>>> dataset = tfds.load("squad")

>>> training_examples = get_examples_from_dataset(dataset, evaluate=False)
>>> evaluation_examples = get_examples_from_dataset(dataset, evaluate=True)
```

#### get\_train\_examples

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/data/processors/squad.py#L607)

( data\_dir filename = None  )

Parameters

* **data\_dir** — Directory containing the data files used for training and evaluating.
* **filename** — None by default, specify this if the training file has a different name than the original one
  which is `train-v1.1.json` and `train-v2.0.json` for squad versions 1.1 and 2.0 respectively.

Returns the training examples from the data directory.

Additionally, the following method can be used to convert SQuAD examples into
`~data.processors.utils.SquadFeatures` that can be used as model inputs.

#### transformers.squad\_convert\_examples\_to\_features

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/data/processors/squad.py#L317)

( examples tokenizer max\_seq\_length doc\_stride max\_query\_length is\_training padding\_strategy = 'max\_length' return\_dataset = False threads = 1 tqdm\_enabled = True  )

Parameters

* **examples** — list of `SquadExample`
* **tokenizer** — an instance of a child of [PreTrainedTokenizer](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer)
* **max\_seq\_length** — The maximum sequence length of the inputs.
* **doc\_stride** — The stride used when the context is too large and is split across several features.
* **max\_query\_length** — The maximum length of the query.
* **is\_training** — whether to create features for model evaluation or model training.
* **padding\_strategy** — Default to “max\_length”. Which padding strategy to use
* **return\_dataset** — Default False. Either ‘pt’ or ‘tf’.
  if ‘pt’: returns a torch.data.TensorDataset, if ‘tf’: returns a tf.data.Dataset
* **threads** — multiple processing threads.

Converts a list of examples into a list of features that can be directly given as input to a model. It is
model-dependant and takes advantage of many of the tokenizer’s features to create the model’s inputs.

Example:


```
processor = SquadV2Processor()
examples = processor.get_dev_examples(data_dir)

features = squad_convert_examples_to_features(
    examples=examples,
    tokenizer=tokenizer,
    max_seq_length=args.max_seq_length,
    doc_stride=args.doc_stride,
    max_query_length=args.max_query_length,
    is_training=not evaluate,
)
```

These processors as well as the aforementioned method can be used with files containing the data as well as with the
*tensorflow\_datasets* package. Examples are given below.

### Example usage

Here is an example using the processors as well as the conversion method using data files:


```
# Loading a V2 processor
processor = SquadV2Processor()
examples = processor.get_dev_examples(squad_v2_data_dir)

# Loading a V1 processor
processor = SquadV1Processor()
examples = processor.get_dev_examples(squad_v1_data_dir)

features = squad_convert_examples_to_features(
    examples=examples,
    tokenizer=tokenizer,
    max_seq_length=max_seq_length,
    doc_stride=args.doc_stride,
    max_query_length=max_query_length,
    is_training=not evaluate,
)
```

Using *tensorflow\_datasets* is as easy as using a data file:


```
# tensorflow_datasets only handle Squad V1.
tfds_examples = tfds.load("squad")
examples = SquadV1Processor().get_examples_from_dataset(tfds_examples, evaluate=evaluate)

features = squad_convert_examples_to_features(
    examples=examples,
    tokenizer=tokenizer,
    max_seq_length=max_seq_length,
    doc_stride=args.doc_stride,
    max_query_length=max_query_length,
    is_training=not evaluate,
)
```

Another example using these processors is given in the [run\_squad.py](https://github.com/huggingface/transformers/tree/main/examples/legacy/question-answering/run_squad.py) script.

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/main_classes/processors.md)
