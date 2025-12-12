# DeepSpeed

[DeepSpeed](https://github.com/deepspeedai/DeepSpeed), powered by Zero Redundancy Optimizer (ZeRO), is an optimization library for training and fitting very large models onto a GPU. It is available in several ZeRO stages, where each stage progressively saves more GPU memory by partitioning the optimizer state, gradients, parameters, and enabling offloading to a CPU or NVMe. DeepSpeed is integrated with the [Trainer](/docs/transformers/main/en/main_classes/trainer#transformers.Trainer) class and most of the setup is automatically taken care of for you.

However, if you want to use DeepSpeed without the [Trainer](/docs/transformers/main/en/main_classes/trainer#transformers.Trainer), Transformers provides a `HfDeepSpeedConfig` class.

> Learn more about using DeepSpeed with [Trainer](/docs/transformers/main/en/main_classes/trainer#transformers.Trainer) in the [DeepSpeed](../deepspeed) guide.

## HfDeepSpeedConfig

### class transformers.integrations.HfDeepSpeedConfig

 [< source >](https://github.com/huggingface/transformers/blob/main/src/transformers/integrations/deepspeed.py#L57)

( config\_file\_or\_dict  )

Parameters

* **config\_file\_or\_dict** (`Union[str, Dict]`) — path to DeepSpeed config file or dict.

This object contains a DeepSpeed configuration dictionary and can be quickly queried for things like zero stage.

A `weakref` of this object is stored in the module’s globals to be able to access the config from areas where
things like the Trainer object is not available (e.g. `from_pretrained` and `_get_resized_embeddings`). Therefore
it’s important that this object remains alive while the program is still running.

[Trainer](/docs/transformers/main/en/main_classes/trainer#transformers.Trainer) uses the `HfTrainerDeepSpeedConfig` subclass instead. That subclass has logic to sync the configuration
with values of [TrainingArguments](/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments) by replacing special placeholder values: `"auto"`. Without this special logic
the DeepSpeed configuration is not modified in any way.

 [Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/main_classes/deepspeed.md)
