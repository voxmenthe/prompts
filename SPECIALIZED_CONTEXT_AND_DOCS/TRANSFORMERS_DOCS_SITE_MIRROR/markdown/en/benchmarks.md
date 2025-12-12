# Benchmarks

Hugging Face’s Benchmarking tools are deprecated and it is advised to use external Benchmarking libraries to measure the speed
and memory complexity of Transformer models.

Let’s take a look at how 🤗 Transformers models can be benchmarked, best practices, and already available benchmarks.

A notebook explaining in more detail how to benchmark 🤗 Transformers models can be found [here](https://github.com/huggingface/notebooks/tree/main/examples/benchmark.ipynb).

## How to benchmark 🤗 Transformers models

The classes `PyTorchBenchmark` and `TensorFlowBenchmark` allow to flexibly benchmark 🤗 Transformers models. The benchmark classes allow us to measure the *peak memory usage* and *required time* for both *inference* and *training*.

Here, *inference* is defined by a single forward pass, and *training* is defined by a single forward pass and
backward pass.

The benchmark classes `PyTorchBenchmark` and `TensorFlowBenchmark` expect an object of type `PyTorchBenchmarkArguments` and
`TensorFlowBenchmarkArguments`, respectively, for instantiation. `PyTorchBenchmarkArguments` and `TensorFlowBenchmarkArguments` are data classes and contain all relevant configurations for their corresponding benchmark class. In the following example, it is shown how a BERT model of type *bert-base-cased* can be benchmarked.

Pytorch

Hide Pytorch content

```
>>> from transformers import PyTorchBenchmark, PyTorchBenchmarkArguments

>>> args = PyTorchBenchmarkArguments(models=["google-bert/bert-base-uncased"], batch_sizes=[8], sequence_lengths=[8, 32, 128, 512])
>>> benchmark = PyTorchBenchmark(args)
```

TensorFlow

Hide TensorFlow content

```
>>> from transformers import TensorFlowBenchmark, TensorFlowBenchmarkArguments

>>> args = TensorFlowBenchmarkArguments(
...     models=["google-bert/bert-base-uncased"], batch_sizes=[8], sequence_lengths=[8, 32, 128, 512]
... )
>>> benchmark = TensorFlowBenchmark(args)
```

Here, three arguments are given to the benchmark argument data classes, namely `models`, `batch_sizes`, and
`sequence_lengths`. The argument `models` is required and expects a `list` of model identifiers from the
[model hub](https://huggingface.co/models) The `list` arguments `batch_sizes` and `sequence_lengths` define
the size of the `input_ids` on which the model is benchmarked. There are many more parameters that can be configured
via the benchmark argument data classes. For more detail on these one can either directly consult the files
`src/transformers/benchmark/benchmark_args_utils.py`, `src/transformers/benchmark/benchmark_args.py` (for PyTorch)
and `src/transformers/benchmark/benchmark_args_tf.py` (for Tensorflow). Alternatively, running the following shell
commands from root will print out a descriptive list of all configurable parameters for PyTorch and Tensorflow
respectively.

Pytorch

Hide Pytorch content

```
python examples/pytorch/benchmarking/run_benchmark.py --help
```

An instantiated benchmark object can then simply be run by calling `benchmark.run()`.

```
>>> results = benchmark.run()
>>> print(results)
====================       INFERENCE - SPEED - RESULT       ====================
--------------------------------------------------------------------------------
Model Name             Batch Size     Seq Length     Time in s                  
--------------------------------------------------------------------------------
google-bert/bert-base-uncased          8               8             0.006     
google-bert/bert-base-uncased          8               32            0.006     
google-bert/bert-base-uncased          8              128            0.018     
google-bert/bert-base-uncased          8              512            0.088     
--------------------------------------------------------------------------------

====================      INFERENCE - MEMORY - RESULT       ====================
--------------------------------------------------------------------------------
Model Name             Batch Size     Seq Length    Memory in MB 
--------------------------------------------------------------------------------
google-bert/bert-base-uncased          8               8             1227
google-bert/bert-base-uncased          8               32            1281
google-bert/bert-base-uncased          8              128            1307
google-bert/bert-base-uncased          8              512            1539
--------------------------------------------------------------------------------

====================        ENVIRONMENT INFORMATION         ====================

- transformers_version: 2.11.0
- framework: PyTorch
- use_torchscript: False
- framework_version: 1.4.0
- python_version: 3.6.10
- system: Linux
- cpu: x86_64
- architecture: 64bit
- date: 2020-06-29
- time: 08:58:43.371351
- fp16: False
- use_multiprocessing: True
- only_pretrain_model: False
- cpu_ram_mb: 32088
- use_gpu: True
- num_gpus: 1
- gpu: TITAN RTX
- gpu_ram_mb: 24217
- gpu_power_watts: 280.0
- gpu_performance_state: 2
- use_tpu: False
```

TensorFlow

Hide TensorFlow content

```
python examples/tensorflow/benchmarking/run_benchmark_tf.py --help
```

An instantiated benchmark object can then simply be run by calling `benchmark.run()`.

```
>>> results = benchmark.run()
>>> print(results)
>>> results = benchmark.run()
>>> print(results)
====================       INFERENCE - SPEED - RESULT       ====================
--------------------------------------------------------------------------------
Model Name             Batch Size     Seq Length     Time in s                  
--------------------------------------------------------------------------------
google-bert/bert-base-uncased          8               8             0.005
google-bert/bert-base-uncased          8               32            0.008
google-bert/bert-base-uncased          8              128            0.022
google-bert/bert-base-uncased          8              512            0.105
--------------------------------------------------------------------------------

====================      INFERENCE - MEMORY - RESULT       ====================
--------------------------------------------------------------------------------
Model Name             Batch Size     Seq Length    Memory in MB 
--------------------------------------------------------------------------------
google-bert/bert-base-uncased          8               8             1330
google-bert/bert-base-uncased          8               32            1330
google-bert/bert-base-uncased          8              128            1330
google-bert/bert-base-uncased          8              512            1770
--------------------------------------------------------------------------------

====================        ENVIRONMENT INFORMATION         ====================

- transformers_version: 2.11.0
- framework: Tensorflow
- use_xla: False
- framework_version: 2.2.0
- python_version: 3.6.10
- system: Linux
- cpu: x86_64
- architecture: 64bit
- date: 2020-06-29
- time: 09:26:35.617317
- fp16: False
- use_multiprocessing: True
- only_pretrain_model: False
- cpu_ram_mb: 32088
- use_gpu: True
- num_gpus: 1
- gpu: TITAN RTX
- gpu_ram_mb: 24217
- gpu_power_watts: 280.0
- gpu_performance_state: 2
- use_tpu: False
```

By default, the *time* and the *required memory* for *inference* are benchmarked. In the example output above the first
two sections show the result corresponding to *inference time* and *inference memory*. In addition, all relevant
information about the computing environment, *e.g.* the GPU type, the system, the library versions, etc… are printed
out in the third section under *ENVIRONMENT INFORMATION*. This information can optionally be saved in a *.csv* file
when adding the argument `save_to_csv=True` to `PyTorchBenchmarkArguments` and
`TensorFlowBenchmarkArguments` respectively. In this case, every section is saved in a separate
*.csv* file. The path to each *.csv* file can optionally be defined via the argument data classes.

Instead of benchmarking pre-trained models via their model identifier, *e.g.* `google-bert/bert-base-uncased`, the user can
alternatively benchmark an arbitrary configuration of any available model class. In this case, a `list` of
configurations must be inserted with the benchmark args as follows.

Pytorch

Hide Pytorch content

```
>>> from transformers import PyTorchBenchmark, PyTorchBenchmarkArguments, BertConfig

>>> args = PyTorchBenchmarkArguments(
...     models=["bert-base", "bert-384-hid", "bert-6-lay"], batch_sizes=[8], sequence_lengths=[8, 32, 128, 512]
... )
>>> config_base = BertConfig()
>>> config_384_hid = BertConfig(hidden_size=384)
>>> config_6_lay = BertConfig(num_hidden_layers=6)

>>> benchmark = PyTorchBenchmark(args, configs=[config_base, config_384_hid, config_6_lay])
>>> benchmark.run()
====================       INFERENCE - SPEED - RESULT       ====================
--------------------------------------------------------------------------------
Model Name             Batch Size     Seq Length       Time in s                  
--------------------------------------------------------------------------------
bert-base                  8              128            0.006
bert-base                  8              512            0.006
bert-base                  8              128            0.018     
bert-base                  8              512            0.088     
bert-384-hid              8               8             0.006     
bert-384-hid              8               32            0.006     
bert-384-hid              8              128            0.011     
bert-384-hid              8              512            0.054     
bert-6-lay                 8               8             0.003     
bert-6-lay                 8               32            0.004     
bert-6-lay                 8              128            0.009     
bert-6-lay                 8              512            0.044
--------------------------------------------------------------------------------

====================      INFERENCE - MEMORY - RESULT       ====================
--------------------------------------------------------------------------------
Model Name             Batch Size     Seq Length      Memory in MB 
--------------------------------------------------------------------------------
bert-base                  8               8             1277
bert-base                  8               32            1281
bert-base                  8              128            1307     
bert-base                  8              512            1539     
bert-384-hid              8               8             1005     
bert-384-hid              8               32            1027     
bert-384-hid              8              128            1035     
bert-384-hid              8              512            1255     
bert-6-lay                 8               8             1097     
bert-6-lay                 8               32            1101     
bert-6-lay                 8              128            1127     
bert-6-lay                 8              512            1359
--------------------------------------------------------------------------------

====================        ENVIRONMENT INFORMATION         ====================

- transformers_version: 2.11.0
- framework: PyTorch
- use_torchscript: False
- framework_version: 1.4.0
- python_version: 3.6.10
- system: Linux
- cpu: x86_64
- architecture: 64bit
- date: 2020-06-29
- time: 09:35:25.143267
- fp16: False
- use_multiprocessing: True
- only_pretrain_model: False
- cpu_ram_mb: 32088
- use_gpu: True
- num_gpus: 1
- gpu: TITAN RTX
- gpu_ram_mb: 24217
- gpu_power_watts: 280.0
- gpu_performance_state: 2
- use_tpu: False
```

TensorFlow

Hide TensorFlow content

```
>>> from transformers import TensorFlowBenchmark, TensorFlowBenchmarkArguments, BertConfig

>>> args = TensorFlowBenchmarkArguments(
...     models=["bert-base", "bert-384-hid", "bert-6-lay"], batch_sizes=[8], sequence_lengths=[8, 32, 128, 512]
... )
>>> config_base = BertConfig()
>>> config_384_hid = BertConfig(hidden_size=384)
>>> config_6_lay = BertConfig(num_hidden_layers=6)

>>> benchmark = TensorFlowBenchmark(args, configs=[config_base, config_384_hid, config_6_lay])
>>> benchmark.run()
====================       INFERENCE - SPEED - RESULT       ====================
--------------------------------------------------------------------------------
Model Name             Batch Size     Seq Length       Time in s                  
--------------------------------------------------------------------------------
bert-base                  8               8             0.005
bert-base                  8               32            0.008
bert-base                  8              128            0.022
bert-base                  8              512            0.106
bert-384-hid              8               8             0.005
bert-384-hid              8               32            0.007
bert-384-hid              8              128            0.018
bert-384-hid              8              512            0.064
bert-6-lay                 8               8             0.002
bert-6-lay                 8               32            0.003
bert-6-lay                 8              128            0.0011
bert-6-lay                 8              512            0.074
--------------------------------------------------------------------------------

====================      INFERENCE - MEMORY - RESULT       ====================
--------------------------------------------------------------------------------
Model Name             Batch Size     Seq Length      Memory in MB 
--------------------------------------------------------------------------------
bert-base                  8               8             1330
bert-base                  8               32            1330
bert-base                  8              128            1330
bert-base                  8              512            1770
bert-384-hid              8               8             1330
bert-384-hid              8               32            1330
bert-384-hid              8              128            1330
bert-384-hid              8              512            1540
bert-6-lay                 8               8             1330
bert-6-lay                 8               32            1330
bert-6-lay                 8              128            1330
bert-6-lay                 8              512            1540
--------------------------------------------------------------------------------

====================        ENVIRONMENT INFORMATION         ====================

- transformers_version: 2.11.0
- framework: Tensorflow
- use_xla: False
- framework_version: 2.2.0
- python_version: 3.6.10
- system: Linux
- cpu: x86_64
- architecture: 64bit
- date: 2020-06-29
- time: 09:38:15.487125
- fp16: False
- use_multiprocessing: True
- only_pretrain_model: False
- cpu_ram_mb: 32088
- use_gpu: True
- num_gpus: 1
- gpu: TITAN RTX
- gpu_ram_mb: 24217
- gpu_power_watts: 280.0
- gpu_performance_state: 2
- use_tpu: False
```

Again, *inference time* and *required memory* for *inference* are measured, but this time for customized configurations
of the `BertModel` class. This feature can especially be helpful when deciding for which configuration the model
should be trained.

## Benchmark best practices

This section lists a couple of best practices one should be aware of when benchmarking a model.

* Currently, only single device benchmarking is supported. When benchmarking on GPU, it is recommended that the user
  specifies on which device the code should be run by setting the `CUDA_VISIBLE_DEVICES` environment variable in the
  shell, *e.g.* `export CUDA_VISIBLE_DEVICES=0` before running the code.
* The option `no_multi_processing` should only be set to `True` for testing and debugging. To ensure accurate
  memory measurement it is recommended to run each memory benchmark in a separate process by making sure
  `no_multi_processing` is set to `True`.
* One should always state the environment information when sharing the results of a model benchmark. Results can vary
  heavily between different GPU devices, library versions, etc., as a consequence, benchmark results on their own are not very
  useful for the community.

## Sharing your benchmark

Previously all available core models (10 at the time) have been benchmarked for *inference time*, across many different
settings: using PyTorch, with and without TorchScript, using TensorFlow, with and without XLA. All of those tests were
done across CPUs (except for TensorFlow XLA) and GPUs.

The approach is detailed in the [following blogpost](https://medium.com/huggingface/benchmarking-transformers-pytorch-and-tensorflow-e2917fb891c2) and the results are
available [here](https://docs.google.com/spreadsheets/d/1sryqufw2D0XlUH4sq3e9Wnxu5EAQkaohzrJbd5HdQ_w/edit?usp=sharing).

With the new *benchmark* tools, it is easier than ever to share your benchmark results with the community

* [PyTorch Benchmarking Results](https://github.com/huggingface/transformers/tree/main/examples/pytorch/benchmarking/README.md).
* [TensorFlow Benchmarking Results](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/benchmarking/README.md).

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/benchmarks.md)
