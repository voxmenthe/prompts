# Utilities for pipelines

This page lists all the utility functions the library provides for pipelines.

Most of those are only useful if you are studying the code of the models in the library.

## Argument handling[[transformers.pipelines.ArgumentHandler]]

#### transformers.pipelines.ArgumentHandler[[transformers.pipelines.ArgumentHandler]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/base.py#L375)

Base interface for handling arguments for each [Pipeline](/docs/transformers/main/en/main_classes/pipelines#transformers.Pipeline).

#### transformers.pipelines.ZeroShotClassificationArgumentHandler[[transformers.pipelines.ZeroShotClassificationArgumentHandler]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/zero_shot_classification.py#L13)

Handles arguments for zero-shot for text classification by turning each possible label into an NLI
premise/hypothesis pair.

#### transformers.pipelines.QuestionAnsweringArgumentHandler[[transformers.pipelines.QuestionAnsweringArgumentHandler]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/question_answering.py#L142)

QuestionAnsweringPipeline requires the user to provide multiple arguments (i.e. question & context) to be mapped to
internal `SquadExample`.

QuestionAnsweringArgumentHandler manages all the possible to create a `SquadExample` from the command-line
supplied arguments.

## Data format[[transformers.PipelineDataFormat]]

#### transformers.PipelineDataFormat[[transformers.PipelineDataFormat]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/base.py#L385)

Base class for all the pipeline supported data format both for reading and writing. Supported data formats
currently includes:

- JSON
- CSV
- stdin/stdout (pipe)

`PipelineDataFormat` also includes some utilities to work with multi-columns like mapping from datasets columns to
pipelines keyword arguments through the `dataset_kwarg_1=dataset_column_1` format.

from_strtransformers.PipelineDataFormat.from_strhttps://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/base.py#L462[{"name": "format", "val": ": str"}, {"name": "output_path", "val": ": str | None"}, {"name": "input_path", "val": ": str | None"}, {"name": "column", "val": ": str | None"}, {"name": "overwrite", "val": " = False"}]- **format** (`str`) --
  The format of the desired pipeline. Acceptable values are `"json"`, `"csv"` or `"pipe"`.
- **output_path** (`str`, *optional*) --
  Where to save the outgoing data.
- **input_path** (`str`, *optional*) --
  Where to look for the input data.
- **column** (`str`, *optional*) --
  The column to read.
- **overwrite** (`bool`, *optional*, defaults to `False`) --
  Whether or not to overwrite the `output_path`.0[PipelineDataFormat](/docs/transformers/main/en/internal/pipelines_utils#transformers.PipelineDataFormat)The proper data format.

Creates an instance of the right subclass of [PipelineDataFormat](/docs/transformers/main/en/internal/pipelines_utils#transformers.PipelineDataFormat) depending on `format`.

**Parameters:**

output_path (`str`) : Where to save the outgoing data.

input_path (`str`) : Where to look for the input data.

column (`str`) : The column to read.

overwrite (`bool`, *optional*, defaults to `False`) : Whether or not to overwrite the `output_path`.

**Returns:**

`[PipelineDataFormat](/docs/transformers/main/en/internal/pipelines_utils#transformers.PipelineDataFormat)`

The proper data format.
#### save[[transformers.PipelineDataFormat.save]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/base.py#L434)

Save the provided data object with the representation for the current [PipelineDataFormat](/docs/transformers/main/en/internal/pipelines_utils#transformers.PipelineDataFormat).

**Parameters:**

data (`dict` or list of `dict`) : The data to store.
#### save_binary[[transformers.PipelineDataFormat.save_binary]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/base.py#L444)

Save the provided data object as a pickle-formatted binary data on the disk.

**Parameters:**

data (`dict` or list of `dict`) : The data to store.

**Returns:**

``str``

Path where the data has been saved.

#### transformers.CsvPipelineDataFormat[[transformers.CsvPipelineDataFormat]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/base.py#L498)

Support for pipelines using CSV data format.

savetransformers.CsvPipelineDataFormat.savehttps://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/base.py#L528[{"name": "data", "val": ": list"}]- **data** (`list[dict]`) -- The data to store.0

Save the provided data object with the representation for the current [PipelineDataFormat](/docs/transformers/main/en/internal/pipelines_utils#transformers.PipelineDataFormat).

**Parameters:**

output_path (`str`) : Where to save the outgoing data.

input_path (`str`) : Where to look for the input data.

column (`str`) : The column to read.

overwrite (`bool`, *optional*, defaults to `False`) : Whether or not to overwrite the `output_path`.

#### transformers.JsonPipelineDataFormat[[transformers.JsonPipelineDataFormat]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/base.py#L542)

Support for pipelines using JSON file format.

savetransformers.JsonPipelineDataFormat.savehttps://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/base.py#L573[{"name": "data", "val": ": dict"}]- **data** (`dict`) -- The data to store.0

Save the provided data object in a json file.

**Parameters:**

output_path (`str`) : Where to save the outgoing data.

input_path (`str`) : Where to look for the input data.

column (`str`) : The column to read.

overwrite (`bool`, *optional*, defaults to `False`) : Whether or not to overwrite the `output_path`.

#### transformers.PipedPipelineDataFormat[[transformers.PipedPipelineDataFormat]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/base.py#L584)

Read data from piped input to the python process. For multi columns data, columns should separated by 	

If columns are provided, then the output will be a dictionary with {column_x: value_x}

savetransformers.PipedPipelineDataFormat.savehttps://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/base.py#L613[{"name": "data", "val": ": dict"}]- **data** (`dict`) -- The data to store.0

Print the data.

**Parameters:**

output_path (`str`) : Where to save the outgoing data.

input_path (`str`) : Where to look for the input data.

column (`str`) : The column to read.

overwrite (`bool`, *optional*, defaults to `False`) : Whether or not to overwrite the `output_path`.

## Utilities[[transformers.pipelines.PipelineException]]

#### transformers.pipelines.PipelineException[[transformers.pipelines.PipelineException]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/base.py#L358)

Raised by a [Pipeline](/docs/transformers/main/en/main_classes/pipelines#transformers.Pipeline) when handling __call__.

**Parameters:**

task (`str`) : The task of the pipeline.

model (`str`) : The model used by the pipeline.

reason (`str`) : The error message to display.
