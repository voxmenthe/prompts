# General Utilities

This page lists all of Transformers general utility functions that are found in the file `utils.py`.

Most of those are only useful if you are studying the general code in the library.

## Enums and namedtuples

### class transformers.utils.ExplicitEnum

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/utils/generic.py#L494)

( value names = None module = None qualname = None type = None start = 1  )

Enum with more explicit error message for missing values.

### class transformers.utils.PaddingStrategy

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/utils/generic.py#L506)

( value names = None module = None qualname = None type = None start = 1  )

Possible values for the `padding` argument in [PreTrainedTokenizerBase.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__). Useful for tab-completion in an
IDE.

### class transformers.TensorType

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/utils/generic.py#L517)

( value names = None module = None qualname = None type = None start = 1  )

Possible values for the `return_tensors` argument in [PreTrainedTokenizerBase.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__). Useful for
tab-completion in an IDE.

## Special Decorators

#### transformers.add\_start\_docstrings

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/utils/doc.py#L37)

( \*docstr  )

#### transformers.utils.add\_start\_docstrings\_to\_model\_forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/utils/doc.py#L45)

( \*docstr  )

#### transformers.add\_end\_docstrings

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/utils/doc.py#L81)

( \*docstr  )

#### transformers.utils.add\_code\_sample\_docstrings

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/utils/doc.py#L1455)

( \*docstr processor\_class = None checkpoint = None output\_type = None config\_class = None mask = '[MASK]' qa\_target\_start\_index = 14 qa\_target\_end\_index = 15 model\_cls = None modality = None expected\_output = None expected\_loss = None real\_checkpoint = None revision = None  )

#### transformers.utils.replace\_return\_docstrings

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/utils/doc.py#L1554)

( output\_type = None config\_class = None  )

## Special Properties

### class transformers.utils.cached\_property

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/utils/generic.py#L56)

( fget = None fset = None fdel = None doc = None  )

Descriptor that mimics @property but caches output in member variable.

From tensorflow\_datasets

Built-in in functools from Python 3.8.

## Other Utilities

### class transformers.utils.\_LazyModule

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/utils/import_utils.py#L2157)

( name: str module\_file: str import\_structure: dict module\_spec: typing.Optional[\_frozen\_importlib.ModuleSpec] = None extra\_objects: typing.Optional[dict[str, object]] = None explicit\_import\_shortcut: typing.Optional[dict[str, list[str]]] = None  )

Module class that surfaces all objects but only performs associated imports when the objects are requested.

#### transformers.infer\_device

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/pytorch_utils.py#L370)

( )

Infers available device.

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/internal/file_utils.md)
