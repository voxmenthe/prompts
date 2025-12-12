# Import Utilities

This page goes through the transformers utilities to enable lazy and fast object import.
While we strive for minimal dependencies, some models have specific dependencies requirements that cannot be
worked around. We don’t want for all users of `transformers` to have to install those dependencies to use other models,
we therefore mark those as soft dependencies rather than hard dependencies.

The transformers toolkit is not made to error-out on import of a model that has a specific dependency; instead, an
object for which you are lacking a dependency will error-out when calling any method on it. As an example, if
`torchvision` isn’t installed, the fast image processors will not be available.

This object is still importable:


```
>>> from transformers import DetrImageProcessorFast
>>> print(DetrImageProcessorFast)
<class 'DetrImageProcessorFast'>
```

However, no method can be called on that object:


```
>>> DetrImageProcessorFast.from_pretrained()
ImportError: 
DetrImageProcessorFast requires the Torchvision library but it was not found in your environment. Check out the instructions on the
installation page: https://pytorch.org/get-started/locally/ and follow the ones that match your environment.
Please note that you may need to restart your runtime after installation.
```

Let’s see how to specify specific object dependencies.

## Specifying Object Dependencies

### Filename-based

All objects under a given filename have an automatic dependency to the tool linked to the filename

**TensorFlow**: All files starting with `modeling_tf_` have an automatic TensorFlow dependency.

**Flax**: All files starting with `modeling_flax_` have an automatic Flax dependency

**PyTorch**: All files starting with `modeling_` and not valid with the above (TensorFlow and Flax) have an automatic
PyTorch dependency

**Tokenizers**: All files starting with `tokenization_` and ending with `_fast` have an automatic `tokenizers` dependency

**Vision**: All files starting with `image_processing_` have an automatic dependency to the `vision` dependency group;
at the time of writing, this only contains the `pillow` dependency.

**Vision + Torch + Torchvision**: All files starting with `image_processing_` and ending with `_fast` have an automatic
dependency to `vision`, `torch`, and `torchvision`.

All of these automatic dependencies are added on top of the explicit dependencies that are detailed below.

### Explicit Object Dependencies

We add a method called `requires` that is used to explicitly specify the dependencies of a given object. As an
example, the `Trainer` class has two hard dependencies: `torch` and `accelerate`. Here is how we specify these
required dependencies:


```
from .utils.import_utils import requires

@requires(backends=("torch", "accelerate"))
class Trainer:
    ...
```

Backends that can be added here are all the backends that are available in the `import_utils.py` module.

Additionally, specific versions can be specified in each backend. For example, this is how you would specify
a requirement on torch>=2.6 on the `Trainer` class:


```
from .utils.import_utils import requires

@requires(backends=("torch>=2.6", "accelerate"))
class Trainer:
    ...
```

You can specify the following operators: `==`, `>`, `>=`, `<`, `<=`, `!=`.

## Methods

#### transformers.utils.import\_utils.define\_import\_structure

 

( module\_path: str prefix: typing.Optional[str] = None  )

This method takes a module\_path as input and creates an import structure digestible by a \_LazyModule.

Here’s an example of an output import structure at the src.transformers.models level:

{
frozenset({‘tokenizers’}): {
‘albert.tokenization\_albert\_fast’: {‘AlbertTokenizerFast’}
},
frozenset(): {
‘albert.configuration\_albert’: {‘AlbertConfig’, ‘AlbertOnnxConfig’},
‘align.processing\_align’: {‘AlignProcessor’},
‘align.configuration\_align’: {‘AlignConfig’, ‘AlignTextConfig’, ‘AlignVisionConfig’},
‘altclip.configuration\_altclip’: {‘AltCLIPConfig’, ‘AltCLIPTextConfig’, ‘AltCLIPVisionConfig’},
‘altclip.processing\_altclip’: {‘AltCLIPProcessor’}
}
}

The import structure is a dict defined with frozensets as keys, and dicts of strings to sets of objects.

If `prefix` is not None, it will add that prefix to all keys in the returned dict.

#### transformers.utils.import\_utils.requires

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/utils/import_utils.py#L2419)

( backends = ()  )

This decorator enables two things:

* Attaching a `__backends` tuple to an object to see what are the necessary backends for it
  to execute correctly without instantiating it
* The ‘@requires’ string is used to dynamically import objects

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/internal/import_utils.md)
