# Intel Gaudi

The Intel Gaudi AI accelerator family includes [Intel Gaudi 1](https://habana.ai/products/gaudi/), [Intel Gaudi 2](https://habana.ai/products/gaudi2/), and [Intel Gaudi 3](https://habana.ai/products/gaudi3/). Each server is equipped with 8 devices, known as Habana Processing Units (HPUs), providing 128GB of memory on Gaudi 3, 96GB on Gaudi 2, and 32GB on the first-gen Gaudi. For more details on the underlying hardware architecture, check out the [Gaudi Architecture](https://docs.habana.ai/en/latest/Gaudi_Overview/Gaudi_Architecture.html) overview.

[TrainingArguments](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.TrainingArguments), [Trainer](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Trainer) and [Pipeline](/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.Pipeline) detect and set the backend device to `hpu` if an Intel Gaudi device is available. No additional changes are required to enable training and inference on your device.

Some modeling code in Transformers is not optimized for HPU lazy mode. If you encounter any errors, set the environment variable below to use eager mode:


```
PT_HPU_LAZY_MODE=0
```

In some cases, you’ll also need to enable int64 support to avoid casting issues with long integers:


```
PT_ENABLE_INT64_SUPPORT=1
```

Refer to the [Gaudi docs](https://docs.habana.ai/en/latest/index.html) for more details.

For training and inference with Gaudi-optimized model implementations, we recommend using [Optimum for Intel Gaudi](https://huggingface.co/docs/optimum/main/en/habana/index).

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/perf_train_gaudi.md)
