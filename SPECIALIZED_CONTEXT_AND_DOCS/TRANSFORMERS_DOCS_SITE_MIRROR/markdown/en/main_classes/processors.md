# Processors

Processors can mean two different things in the Transformers library:

- the objects that pre-process inputs for multi-modal models such as [Wav2Vec2](../model_doc/wav2vec2) (speech and text)
  or [CLIP](../model_doc/clip) (text and vision)
- deprecated objects that were used in older versions of the library to preprocess data for GLUE or SQUAD.

## Multi-modal processors[[transformers.ProcessorMixin]]

Any multi-modal model will require an object to encode or decode the data that groups several modalities (among text,
vision and audio). This is handled by objects called processors, which group together two or more processing objects
such as tokenizers (for the text modality), image processors (for vision) and feature extractors (for audio).

Those processors inherit from the following base class that implements the saving and loading functionality:

#### transformers.ProcessorMixin[[transformers.ProcessorMixin]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/processing_utils.py#L552)

This is a mixin used to provide saving/loading functionality for all processor classes.

apply_chat_templatetransformers.ProcessorMixin.apply_chat_templatehttps://github.com/huggingface/transformers/blob/main/src/transformers/processing_utils.py#L1560[{"name": "conversation", "val": ": typing.Union[list[dict[str, str]], list[list[dict[str, str]]]]"}, {"name": "chat_template", "val": ": typing.Optional[str] = None"}, {"name": "**kwargs", "val": ": typing_extensions.Unpack[transformers.processing_utils.AllKwargsForChatTemplate]"}]- **conversation** (`Union[list[Dict, [str, str]], list[list[dict[str, str]]]]`) --
  The conversation to format.
- **chat_template** (`Optional[str]`, *optional*) --
  The Jinja template to use for formatting the conversation. If not provided, the tokenizer's
  chat template is used.0

Similar to the `apply_chat_template` method on tokenizers, this method applies a Jinja template to input
conversations to turn them into a single tokenizable string.

The input is expected to be in the following format, where each message content is a list consisting of text and
optionally image or video inputs. One can also provide an image, video, URL or local path which will be used to form
`pixel_values` when `return_dict=True`. If not provided, one will get only the formatted text, optionally tokenized text.

conversation = [
{
"role": "user",
"content": [
{"type": "image", "url": "https://www.ilankelman.org/stopsigns/australia.jpg"},
{"type": "text", "text": "Please describe this image in detail."},
],
},
]

**Parameters:**

conversation (`Union[list[Dict, [str, str]], list[list[dict[str, str]]]]`) : The conversation to format.

chat_template (`Optional[str]`, *optional*) : The Jinja template to use for formatting the conversation. If not provided, the tokenizer's chat template is used.
#### batch_decode[[transformers.ProcessorMixin.batch_decode]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/processing_utils.py#L1520)

This method forwards all its arguments to PreTrainedTokenizer's [batch_decode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.batch_decode). Please
refer to the docstring of this method for more information.
#### check_argument_for_proper_class[[transformers.ProcessorMixin.check_argument_for_proper_class]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/processing_utils.py#L660)

Checks the passed argument's class against the expected transformers class. In case of an unexpected
mismatch between expected and actual class, an error is raise. Otherwise, the proper retrieved class
is returned.
#### decode[[transformers.ProcessorMixin.decode]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/processing_utils.py#L1529)

This method forwards all its arguments to PreTrainedTokenizer's [decode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.decode). Please refer to
the docstring of this method for more information.
#### from_args_and_dict[[transformers.ProcessorMixin.from_args_and_dict]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/processing_utils.py#L1122)

Instantiates a type of `~processing_utils.ProcessingMixin` from a Python dictionary of parameters.

**Parameters:**

processor_dict (`dict[str, Any]`) : Dictionary that will be used to instantiate the processor object. Such a dictionary can be retrieved from a pretrained checkpoint by leveraging the `~processing_utils.ProcessingMixin.to_dict` method.

kwargs (`dict[str, Any]`) : Additional parameters from which to initialize the processor object.

**Returns:**

``~processing_utils.ProcessingMixin``

The processor object instantiated from those
parameters.
#### from_pretrained[[transformers.ProcessorMixin.from_pretrained]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/processing_utils.py#L1349)

Instantiate a processor associated with a pretrained model.

This class method is simply calling the feature extractor
[from_pretrained()](/docs/transformers/main/en/main_classes/feature_extractor#transformers.FeatureExtractionMixin.from_pretrained), image processor
[ImageProcessingMixin](/docs/transformers/main/en/main_classes/image_processor#transformers.ImageProcessingMixin) and the tokenizer
`~tokenization_utils_base.PreTrainedTokenizer.from_pretrained` methods. Please refer to the docstrings of the
methods above for more information.

**Parameters:**

pretrained_model_name_or_path (`str` or `os.PathLike`) : This can be either:  - a string, the *model id* of a pretrained feature_extractor hosted inside a model repo on huggingface.co. - a path to a *directory* containing a feature extractor file saved using the [save_pretrained()](/docs/transformers/main/en/main_classes/feature_extractor#transformers.FeatureExtractionMixin.save_pretrained) method, e.g., `./my_model_directory/`. - a path or url to a saved feature extractor JSON *file*, e.g., `./my_model_directory/preprocessor_config.json`.

- ****kwargs** : Additional keyword arguments passed along to both [from_pretrained()](/docs/transformers/main/en/main_classes/feature_extractor#transformers.FeatureExtractionMixin.from_pretrained) and `~tokenization_utils_base.PreTrainedTokenizer.from_pretrained`.
#### get_processor_dict[[transformers.ProcessorMixin.get_processor_dict]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/processing_utils.py#L883)

From a `pretrained_model_name_or_path`, resolve to a dictionary of parameters, to be used for instantiating a
processor of type `~processing_utils.ProcessingMixin` using `from_args_and_dict`.

**Parameters:**

pretrained_model_name_or_path (`str` or `os.PathLike`) : The identifier of the pre-trained checkpoint from which we want the dictionary of parameters.

subfolder (`str`, *optional*, defaults to `""`) : In case the relevant files are located inside a subfolder of the model repo on huggingface.co, you can specify the folder name here.

**Returns:**

``tuple[Dict, Dict]``

The dictionary(ies) that will be used to instantiate the processor object.
#### post_process_image_text_to_text[[transformers.ProcessorMixin.post_process_image_text_to_text]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/processing_utils.py#L1819)

Post-process the output of a vlm to decode the text.

**Parameters:**

generated_outputs (`torch.Tensor` or `np.ndarray`) : The output of the model `generate` function. The output is expected to be a tensor of shape `(batch_size, sequence_length)` or `(sequence_length,)`.

skip_special_tokens (`bool`, *optional*, defaults to `True`) : Whether or not to remove special tokens in the output. Argument passed to the tokenizer's `decode` method.

- ****kwargs** : Additional arguments to be passed to the tokenizer's `decode` method.

**Returns:**

``list[str]``

The decoded text.
#### post_process_multimodal_output[[transformers.ProcessorMixin.post_process_multimodal_output]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/processing_utils.py#L1790)

Post-process the output of a multimodal model to return the requested modality output.
If the model cannot generated the requested modality, an error will be raised.

**Parameters:**

generated_outputs (`torch.Tensor` or `np.ndarray`) : The output of the model `generate` function. The output is expected to be a tensor of shape `(batch_size, sequence_length)` or `(sequence_length,)`.

skip_special_tokens (`bool`, *optional*, defaults to `True`) : Whether or not to remove special tokens in the output. Argument passed to the tokenizer's `batch_decode` method.

generation_mode (`str`, *optional*) : Generation mode indicated which modality to output and can be one of `["text", "image", "audio"]`.

- ****kwargs** : Additional arguments to be passed to the tokenizer's `batch_decode method`.

**Returns:**

``list[str]``

The decoded text.
#### push_to_hub[[transformers.ProcessorMixin.push_to_hub]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/utils/hub.py#L711)

Upload the processor files to the 🤗 Model Hub.

Examples:

```python
from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained("google-bert/bert-base-cased")

# Push the processor to your namespace with the name "my-finetuned-bert".
processor.push_to_hub("my-finetuned-bert")

# Push the processor to an organization with the name "my-finetuned-bert".
processor.push_to_hub("huggingface/my-finetuned-bert")
```

**Parameters:**

repo_id (`str`) : The name of the repository you want to push your processor to. It should contain your organization name when pushing to a given organization.

commit_message (`str`, *optional*) : Message to commit while pushing. Will default to `"Upload processor"`.

commit_description (`str`, *optional*) : The description of the commit that will be created

private (`bool`, *optional*) : Whether to make the repo private. If `None` (default), the repo will be public unless the organization's default is private. This value is ignored if the repo already exists.

token (`bool` or `str`, *optional*) : The token to use as HTTP bearer authorization for remote files. If `True` (default), will use the token generated when running `hf auth login` (stored in `~/.huggingface`).

revision (`str`, *optional*) : Branch to push the uploaded files to.

create_pr (`bool`, *optional*, defaults to `False`) : Whether or not to create a PR with the uploaded files or directly commit.

max_shard_size (`int` or `str`, *optional*, defaults to `"50GB"`) : Only applicable for models. The maximum size for a checkpoint before being sharded. Checkpoints shard will then be each of size lower than this size. If expressed as a string, needs to be digits followed by a unit (like `"5MB"`).

tags (`list[str]`, *optional*) : List of tags to push on the Hub.
#### register_for_auto_class[[transformers.ProcessorMixin.register_for_auto_class]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/processing_utils.py#L1427)

Register this class with a given auto class. This should only be used for custom feature extractors as the ones
in the library are already mapped with `AutoProcessor`.

**Parameters:**

auto_class (`str` or `type`, *optional*, defaults to `"AutoProcessor"`) : The auto class to register this new feature extractor with.
#### save_pretrained[[transformers.ProcessorMixin.save_pretrained]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/processing_utils.py#L778)

Saves the attributes of this processor (feature extractor, tokenizer...) in the specified directory so that it
can be reloaded using the [from_pretrained()](/docs/transformers/main/en/main_classes/processors#transformers.ProcessorMixin.from_pretrained) method.

This class method is simply calling [save_pretrained()](/docs/transformers/main/en/main_classes/feature_extractor#transformers.FeatureExtractionMixin.save_pretrained) and
[save_pretrained()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.save_pretrained). Please refer to the docstrings of the
methods above for more information.

**Parameters:**

save_directory (`str` or `os.PathLike`) : Directory where the feature extractor JSON file and the tokenizer files will be saved (directory will be created if it does not exist).

push_to_hub (`bool`, *optional*, defaults to `False`) : Whether or not to push your model to the Hugging Face model hub after saving it. You can specify the repository you want to push to with `repo_id` (will default to the name of `save_directory` in your namespace).

kwargs (`dict[str, Any]`, *optional*) : Additional key word arguments passed along to the [push_to_hub()](/docs/transformers/main/en/main_classes/model#transformers.utils.PushToHubMixin.push_to_hub) method.
#### to_dict[[transformers.ProcessorMixin.to_dict]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/processing_utils.py#L681)

Serializes this instance to a Python dictionary.

**Returns:**

``dict[str, Any]``

Dictionary of all the attributes that make up this processor instance.
#### to_json_file[[transformers.ProcessorMixin.to_json_file]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/processing_utils.py#L762)

Save this instance to a JSON file.

**Parameters:**

json_file_path (`str` or `os.PathLike`) : Path to the JSON file in which this processor instance's parameters will be saved.
#### to_json_string[[transformers.ProcessorMixin.to_json_string]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/processing_utils.py#L751)

Serializes this instance to a JSON string.

**Returns:**

``str``

String containing all the attributes that make up this feature_extractor instance in JSON format.

## Deprecated processors[[transformers.DataProcessor]]

All processors follow the same architecture which is that of the
[DataProcessor](/docs/transformers/main/en/main_classes/processors#transformers.DataProcessor). The processor returns a list of
[InputExample](/docs/transformers/main/en/main_classes/processors#transformers.InputExample). These
[InputExample](/docs/transformers/main/en/main_classes/processors#transformers.InputExample) can be converted to
[InputFeatures](/docs/transformers/main/en/main_classes/processors#transformers.InputFeatures) in order to be fed to the model.

#### transformers.DataProcessor[[transformers.DataProcessor]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/data/processors/utils.py#L79)

Base class for data converters for sequence classification data sets.

get_dev_examplestransformers.DataProcessor.get_dev_exampleshttps://github.com/huggingface/transformers/blob/main/src/transformers/data/processors/utils.py#L96[{"name": "data_dir", "val": ""}]
Gets a collection of [InputExample](/docs/transformers/main/en/main_classes/processors#transformers.InputExample) for the dev set.
#### get_example_from_tensor_dict[[transformers.DataProcessor.get_example_from_tensor_dict]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/data/processors/utils.py#L82)

Gets an example from a dict.

**Parameters:**

tensor_dict : Keys and values should match the corresponding Glue tensorflow_dataset examples.
#### get_labels[[transformers.DataProcessor.get_labels]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/data/processors/utils.py#L104)

Gets the list of labels for this data set.
#### get_test_examples[[transformers.DataProcessor.get_test_examples]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/data/processors/utils.py#L100)

Gets a collection of [InputExample](/docs/transformers/main/en/main_classes/processors#transformers.InputExample) for the test set.
#### get_train_examples[[transformers.DataProcessor.get_train_examples]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/data/processors/utils.py#L92)

Gets a collection of [InputExample](/docs/transformers/main/en/main_classes/processors#transformers.InputExample) for the train set.
#### tfds_map[[transformers.DataProcessor.tfds_map]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/data/processors/utils.py#L108)

Some tensorflow_datasets datasets are not formatted the same way the GLUE datasets are. This method converts
examples to the correct format.

#### transformers.InputExample[[transformers.InputExample]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/data/processors/utils.py#L29)

A single training/test example for simple sequence classification.

to_json_stringtransformers.InputExample.to_json_stringhttps://github.com/huggingface/transformers/blob/main/src/transformers/data/processors/utils.py#L48[]
Serializes this instance to a JSON string.

**Parameters:**

guid : Unique id for the example.

text_a : string. The untokenized text of the first sequence. For single sequence tasks, only this sequence must be specified.

text_b : (Optional) string. The untokenized text of the second sequence. Only must be specified for sequence pair tasks.

label : (Optional) string. The label of the example. This should be specified for train and dev examples, but not for test examples.

#### transformers.InputFeatures[[transformers.InputFeatures]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/data/processors/utils.py#L54)

A single set of features of data. Property names are the same names as the corresponding inputs to a model.

to_json_stringtransformers.InputFeatures.to_json_stringhttps://github.com/huggingface/transformers/blob/main/src/transformers/data/processors/utils.py#L74[]
Serializes this instance to a JSON string.

**Parameters:**

input_ids : Indices of input sequence tokens in the vocabulary.

attention_mask : Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`: Usually `1` for tokens that are NOT MASKED, `0` for MASKED (padded) tokens.

token_type_ids : (Optional) Segment token indices to indicate first and second portions of the inputs. Only some models use them.

label : (Optional) Label corresponding to the input. Int for classification problems, float for regression problems.

## GLUE[[transformers.glue_convert_examples_to_features]]

[General Language Understanding Evaluation (GLUE)](https://gluebenchmark.com/) is a benchmark that evaluates the
performance of models across a diverse set of existing NLU tasks. It was released together with the paper [GLUE: A
multi-task benchmark and analysis platform for natural language understanding](https://openreview.net/pdf?id=rJ4km2R5t7)

This library hosts a total of 10 processors for the following tasks: MRPC, MNLI, MNLI (mismatched), CoLA, SST2, STSB,
QQP, QNLI, RTE and WNLI.

Those processors are:

- `~data.processors.utils.MrpcProcessor`
- `~data.processors.utils.MnliProcessor`
- `~data.processors.utils.MnliMismatchedProcessor`
- `~data.processors.utils.Sst2Processor`
- `~data.processors.utils.StsbProcessor`
- `~data.processors.utils.QqpProcessor`
- `~data.processors.utils.QnliProcessor`
- `~data.processors.utils.RteProcessor`
- `~data.processors.utils.WnliProcessor`

Additionally, the following method can be used to load values from a data file and convert them to a list of
[InputExample](/docs/transformers/main/en/main_classes/processors#transformers.InputExample).

#### transformers.glue_convert_examples_to_features[[transformers.glue_convert_examples_to_features]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/data/processors/glue.py#L36)

Loads a data file into a list of `InputFeatures`

**Parameters:**

examples : List of `InputExamples` containing the examples.

tokenizer : Instance of a tokenizer that will tokenize the examples

max_length : Maximum example length. Defaults to the tokenizer's max_len

task : GLUE task

label_list : List of labels. Can be obtained from the processor using the `processor.get_labels()` method

output_mode : String indicating the output mode. Either `regression` or `classification`

**Returns:**

Will return a list of task-specific `InputFeatures` which can be fed to the model.

## XNLI

[The Cross-Lingual NLI Corpus (XNLI)](https://www.nyu.edu/projects/bowman/xnli/) is a benchmark that evaluates the
quality of cross-lingual text representations. XNLI is crowd-sourced dataset based on [*MultiNLI*](http://www.nyu.edu/projects/bowman/multinli/): pairs of text are labeled with textual entailment annotations for 15
different languages (including both high-resource language such as English and low-resource languages such as Swahili).

It was released together with the paper [XNLI: Evaluating Cross-lingual Sentence Representations](https://huggingface.co/papers/1809.05053)

This library hosts the processor to load the XNLI data:

- `~data.processors.utils.XnliProcessor`

Please note that since the gold labels are available on the test set, evaluation is performed on the test set.

An example using these processors is given in the [run_xnli.py](https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification/run_xnli.py) script.

## SQuAD

[The Stanford Question Answering Dataset (SQuAD)](https://rajpurkar.github.io/SQuAD-explorer//) is a benchmark that
evaluates the performance of models on question answering. Two versions are available, v1.1 and v2.0. The first version
(v1.1) was released together with the paper [SQuAD: 100,000+ Questions for Machine Comprehension of Text](https://huggingface.co/papers/1606.05250). The second version (v2.0) was released alongside the paper [Know What You Don't
Know: Unanswerable Questions for SQuAD](https://huggingface.co/papers/1806.03822).

This library hosts a processor for each of the two versions:

### Processors[[transformers.data.processors.squad.SquadProcessor]]

Those processors are:

- `~data.processors.utils.SquadV1Processor`
- `~data.processors.utils.SquadV2Processor`

They both inherit from the abstract class `~data.processors.utils.SquadProcessor`

#### transformers.data.processors.squad.SquadProcessor[[transformers.data.processors.squad.SquadProcessor]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/data/processors/squad.py#L433)

Processor for the SQuAD data set. overridden by SquadV1Processor and SquadV2Processor, used by the version 1.1 and
version 2.0 of SQuAD, respectively.

get_dev_examplestransformers.data.processors.squad.SquadProcessor.get_dev_exampleshttps://github.com/huggingface/transformers/blob/main/src/transformers/data/processors/squad.py#L521[{"name": "data_dir", "val": ""}, {"name": "filename", "val": " = None"}]- **data_dir** -- Directory containing the data files used for training and evaluating.
- **filename** -- None by default, specify this if the evaluation file has a different name than the original one
  which is `dev-v1.1.json` and `dev-v2.0.json` for squad versions 1.1 and 2.0 respectively.0

Returns the evaluation example from the data directory.

**Parameters:**

data_dir : Directory containing the data files used for training and evaluating.

filename : None by default, specify this if the evaluation file has a different name than the original one which is `dev-v1.1.json` and `dev-v2.0.json` for squad versions 1.1 and 2.0 respectively.
#### get_examples_from_dataset[[transformers.data.processors.squad.SquadProcessor.get_examples_from_dataset]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/data/processors/squad.py#L466)

Creates a list of `SquadExample` using a TFDS dataset.

Examples:

```python
>>> import tensorflow_datasets as tfds

>>> dataset = tfds.load("squad")

>>> training_examples = get_examples_from_dataset(dataset, evaluate=False)
>>> evaluation_examples = get_examples_from_dataset(dataset, evaluate=True)
```

**Parameters:**

dataset : The tfds dataset loaded from *tensorflow_datasets.load("squad")*

evaluate : Boolean specifying if in evaluation mode or in training mode

**Returns:**

List of SquadExample
#### get_train_examples[[transformers.data.processors.squad.SquadProcessor.get_train_examples]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/data/processors/squad.py#L499)

Returns the training examples from the data directory.

**Parameters:**

data_dir : Directory containing the data files used for training and evaluating.

filename : None by default, specify this if the training file has a different name than the original one which is `train-v1.1.json` and `train-v2.0.json` for squad versions 1.1 and 2.0 respectively.

Additionally, the following method can be used to convert SQuAD examples into
`~data.processors.utils.SquadFeatures` that can be used as model inputs.

#### transformers.squad_convert_examples_to_features[[transformers.squad_convert_examples_to_features]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/data/processors/squad.py#L313)

Converts a list of examples into a list of features that can be directly given as input to a model. It is
model-dependant and takes advantage of many of the tokenizer's features to create the model's inputs.

Example:

```python
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

**Parameters:**

examples : list of `SquadExample`

tokenizer : an instance of a child of [PreTrainedTokenizer](/docs/transformers/main/en/main_classes/tokenizer#transformers.PythonBackend)

max_seq_length : The maximum sequence length of the inputs.

doc_stride : The stride used when the context is too large and is split across several features.

max_query_length : The maximum length of the query.

is_training : whether to create features for model evaluation or model training.

padding_strategy : Default to "max_length". Which padding strategy to use

return_dataset : Default False. Can also be 'pt'. if 'pt': returns a torch.data.TensorDataset.

threads : multiple processing threads.

**Returns:**

list of `SquadFeatures`

These processors as well as the aforementioned method can be used with files containing the data as well as with the
*tensorflow_datasets* package. Examples are given below.

### Example usage

Here is an example using the processors as well as the conversion method using data files:

```python
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

Using *tensorflow_datasets* is as easy as using a data file:

```python
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

Another example using these processors is given in the [run_squad.py](https://github.com/huggingface/transformers/tree/main/examples/legacy/question-answering/run_squad.py) script.
