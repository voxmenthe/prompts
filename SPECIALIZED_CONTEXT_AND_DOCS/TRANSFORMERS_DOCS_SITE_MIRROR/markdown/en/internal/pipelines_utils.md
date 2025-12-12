# Utilities for pipelines

This page lists all the utility functions the library provides for pipelines.

Most of those are only useful if you are studying the code of the models in the library.

## Argument handling

### class transformers.pipelines.ArgumentHandler

 [< source >](https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/base.py#L375)

( )

Base interface for handling arguments for each [Pipeline](/docs/transformers/main/en/main_classes/pipelines#transformers.Pipeline).

### class transformers.pipelines.ZeroShotClassificationArgumentHandler

 [< source >](https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/zero_shot_classification.py#L13)

( )

Handles arguments for zero-shot for text classification by turning each possible label into an NLI
premise/hypothesis pair.

### class transformers.pipelines.QuestionAnsweringArgumentHandler

 [< source >](https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/question_answering.py#L142)

( )

QuestionAnsweringPipeline requires the user to provide multiple arguments (i.e. question & context) to be mapped to
internal `SquadExample`.

QuestionAnsweringArgumentHandler manages all the possible to create a `SquadExample` from the command-line
supplied arguments.

## Data format

### class transformers.PipelineDataFormat

 [< source >](https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/base.py#L385)

( output\_path: str | None input\_path: str | None column: str | None overwrite: bool = False  )

Parameters

* **output\_path** (`str`) — Where to save the outgoing data.
* **input\_path** (`str`) — Where to look for the input data.
* **column** (`str`) — The column to read.
* **overwrite** (`bool`, *optional*, defaults to `False`) —
  Whether or not to overwrite the `output_path`.

Base class for all the pipeline supported data format both for reading and writing. Supported data formats
currently includes:

* JSON
* CSV
* stdin/stdout (pipe)

`PipelineDataFormat` also includes some utilities to work with multi-columns like mapping from datasets columns to
pipelines keyword arguments through the `dataset_kwarg_1=dataset_column_1` format.

#### from\_str

 [< source >](https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/base.py#L462)

( format: str output\_path: str | None input\_path: str | None column: str | None overwrite = False  ) → [PipelineDataFormat](/docs/transformers/main/en/internal/pipelines_utils#transformers.PipelineDataFormat)

Parameters

* **format** (`str`) —
  The format of the desired pipeline. Acceptable values are `"json"`, `"csv"` or `"pipe"`.
* **output\_path** (`str`, *optional*) —
  Where to save the outgoing data.
* **input\_path** (`str`, *optional*) —
  Where to look for the input data.
* **column** (`str`, *optional*) —
  The column to read.
* **overwrite** (`bool`, *optional*, defaults to `False`) —
  Whether or not to overwrite the `output_path`.

Returns

[PipelineDataFormat](/docs/transformers/main/en/internal/pipelines_utils#transformers.PipelineDataFormat)

The proper data format.

Creates an instance of the right subclass of [PipelineDataFormat](/docs/transformers/main/en/internal/pipelines_utils#transformers.PipelineDataFormat) depending on `format`.

#### save

 [< source >](https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/base.py#L434)

( data: dict | list[dict]  )

Parameters

* **data** (`dict` or list of `dict`) — The data to store.

Save the provided data object with the representation for the current [PipelineDataFormat](/docs/transformers/main/en/internal/pipelines_utils#transformers.PipelineDataFormat).

#### save\_binary

 [< source >](https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/base.py#L444)

( data: dict | list[dict]  ) → `str`

Parameters

* **data** (`dict` or list of `dict`) — The data to store.

Returns

`str`

Path where the data has been saved.

Save the provided data object as a pickle-formatted binary data on the disk.

### class transformers.CsvPipelineDataFormat

 [< source >](https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/base.py#L498)

( output\_path: str | None input\_path: str | None column: str | None overwrite = False  )

Parameters

* **output\_path** (`str`) — Where to save the outgoing data.
* **input\_path** (`str`) — Where to look for the input data.
* **column** (`str`) — The column to read.
* **overwrite** (`bool`, *optional*, defaults to `False`) —
  Whether or not to overwrite the `output_path`.

Support for pipelines using CSV data format.

#### save

 [< source >](https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/base.py#L528)

( data: list  )

Parameters

* **data** (`list[dict]`) — The data to store.

Save the provided data object with the representation for the current [PipelineDataFormat](/docs/transformers/main/en/internal/pipelines_utils#transformers.PipelineDataFormat).

### class transformers.JsonPipelineDataFormat

 [< source >](https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/base.py#L542)

( output\_path: str | None input\_path: str | None column: str | None overwrite = False  )

Parameters

* **output\_path** (`str`) — Where to save the outgoing data.
* **input\_path** (`str`) — Where to look for the input data.
* **column** (`str`) — The column to read.
* **overwrite** (`bool`, *optional*, defaults to `False`) —
  Whether or not to overwrite the `output_path`.

Support for pipelines using JSON file format.

#### save

 [< source >](https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/base.py#L573)

( data: dict  )

Parameters

* **data** (`dict`) — The data to store.

Save the provided data object in a json file.

### class transformers.PipedPipelineDataFormat

 [< source >](https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/base.py#L584)

( output\_path: str | None input\_path: str | None column: str | None overwrite: bool = False  )

Parameters

* **output\_path** (`str`) — Where to save the outgoing data.
* **input\_path** (`str`) — Where to look for the input data.
* **column** (`str`) — The column to read.
* **overwrite** (`bool`, *optional*, defaults to `False`) —
  Whether or not to overwrite the `output_path`.

Read data from piped input to the python process. For multi columns data, columns should separated by

If columns are provided, then the output will be a dictionary with {column\_x: value\_x}

#### save

 [< source >](https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/base.py#L613)

( data: dict  )

Parameters

* **data** (`dict`) — The data to store.

Print the data.

## Utilities

### class transformers.pipelines.PipelineException

 [< source >](https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/base.py#L358)

( task: str model: str reason: str  )

Parameters

* **task** (`str`) — The task of the pipeline.
* **model** (`str`) — The model used by the pipeline.
* **reason** (`str`) — The error message to display.

Raised by a [Pipeline](/docs/transformers/main/en/main_classes/pipelines#transformers.Pipeline) when handling **call**.

 [Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/internal/pipelines_utils.md)
