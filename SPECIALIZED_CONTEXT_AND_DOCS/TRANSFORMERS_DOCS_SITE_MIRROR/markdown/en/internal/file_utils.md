# General Utilities

This page lists all of Transformers general utility functions that are found in the file `utils.py`.

Most of those are only useful if you are studying the general code in the library.

## Enums and namedtuples

### class transformers.utils.ExplicitEnum

 [< source >](https://github.com/huggingface/transformers/blob/main/src/transformers/utils/generic.py#L428)

( value names = None module = None qualname = None type = None start = 1  )

Enum with more explicit error message for missing values.

### class transformers.utils.PaddingStrategy

 [< source >](https://github.com/huggingface/transformers/blob/main/src/transformers/utils/generic.py#L440)

( value names = None module = None qualname = None type = None start = 1  )

Possible values for the `padding` argument in [PreTrainedTokenizerBase.**call**()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__). Useful for tab-completion in an
IDE.

### class transformers.TensorType

 [< source >](https://github.com/huggingface/transformers/blob/main/src/transformers/utils/generic.py#L451)

( value names = None module = None qualname = None type = None start = 1  )

Possible values for the `return_tensors` argument in [PreTrainedTokenizerBase.**call**()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__). Useful for
tab-completion in an IDE.

## Special Decorators

#### transformers.add\_start\_docstrings

 [< source >](https://github.com/huggingface/transformers/blob/main/src/transformers/utils/doc.py#L37)

( \*docstr  )

#### transformers.utils.add\_start\_docstrings\_to\_model\_forward

 [< source >](https://github.com/huggingface/transformers/blob/main/src/transformers/utils/doc.py#L45)

( \*docstr  )

#### transformers.add\_end\_docstrings

 [< source >](https://github.com/huggingface/transformers/blob/main/src/transformers/utils/doc.py#L81)

( \*docstr  )

#### transformers.utils.add\_code\_sample\_docstrings

 [< source >](https://github.com/huggingface/transformers/blob/main/src/transformers/utils/doc.py#L1006)

( \*docstr processor\_class = None checkpoint = None output\_type = None config\_class = None mask = '[MASK]' qa\_target\_start\_index = 14 qa\_target\_end\_index = 15 model\_cls = None modality = None expected\_output = None expected\_loss = None real\_checkpoint = None revision = None  )

#### transformers.utils.replace\_return\_docstrings

 [< source >](https://github.com/huggingface/transformers/blob/main/src/transformers/utils/doc.py#L1100)

( output\_type = None config\_class = None  )

## Other Utilities

### class transformers.utils.\_LazyModule

 [< source >](https://github.com/huggingface/transformers/blob/main/src/transformers/utils/import_utils.py#L1865)

( name: str module\_file: str import\_structure: dict module\_spec: \_frozen\_importlib.ModuleSpec | None = None extra\_objects: dict[str, object] | None = None explicit\_import\_shortcut: dict[str, list[str]] | None = None  )

Module class that surfaces all objects but only performs associated imports when the objects are requested.

 [Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/internal/file_utils.md)
