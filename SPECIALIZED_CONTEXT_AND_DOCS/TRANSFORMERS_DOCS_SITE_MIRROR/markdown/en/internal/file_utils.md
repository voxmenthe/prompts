# General Utilities

This page lists all of Transformers general utility functions that are found in the file `utils.py`.

Most of those are only useful if you are studying the general code in the library.

## Enums and namedtuples[[transformers.utils.ExplicitEnum]]

#### transformers.utils.ExplicitEnum[[transformers.utils.ExplicitEnum]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/utils/generic.py#L428)

Enum with more explicit error message for missing values.

#### transformers.utils.PaddingStrategy[[transformers.utils.PaddingStrategy]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/utils/generic.py#L440)

Possible values for the `padding` argument in [PreTrainedTokenizerBase.__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__). Useful for tab-completion in an
IDE.

#### transformers.TensorType[[transformers.TensorType]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/utils/generic.py#L451)

Possible values for the `return_tensors` argument in [PreTrainedTokenizerBase.__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__). Useful for
tab-completion in an IDE.

## Special Decorators[[transformers.add_start_docstrings]]

#### transformers.add_start_docstrings[[transformers.add_start_docstrings]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/utils/doc.py#L37)

#### transformers.utils.add_start_docstrings_to_model_forward[[transformers.utils.add_start_docstrings_to_model_forward]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/utils/doc.py#L45)

#### transformers.add_end_docstrings[[transformers.add_end_docstrings]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/utils/doc.py#L81)

#### transformers.utils.add_code_sample_docstrings[[transformers.utils.add_code_sample_docstrings]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/utils/doc.py#L1006)

#### transformers.utils.replace_return_docstrings[[transformers.utils.replace_return_docstrings]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/utils/doc.py#L1100)

## Other Utilities[[transformers.utils._LazyModule]]

#### transformers.utils._LazyModule[[transformers.utils._LazyModule]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/utils/import_utils.py#L1865)

Module class that surfaces all objects but only performs associated imports when the objects are requested.
