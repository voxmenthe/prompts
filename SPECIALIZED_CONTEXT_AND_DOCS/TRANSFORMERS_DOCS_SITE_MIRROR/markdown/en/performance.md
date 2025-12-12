# Performance and Scalability

Training large transformer models and deploying them to production present various challenges.  
During training, the model may require more GPU memory than available or exhibit slow training speed. In the deployment
phase, the model can struggle to handle the required throughput in a production environment.

This documentation aims to assist you in overcoming these challenges and finding the optimal settings for your use-case.
The guides are divided into training and inference sections, as each comes with different challenges and solutions.
Within each section you’ll find separate guides for different hardware configurations, such as single GPU vs. multi-GPU
for training or CPU vs. GPU for inference.

Use this document as your starting point to navigate further to the methods that match your scenario.

## Training

Training large transformer models efficiently requires an accelerator such as a GPU or TPU. The most common case is where
you have a single GPU. The methods that you can apply to improve training efficiency on a single GPU extend to other setups
such as multiple GPU. However, there are also techniques that are specific to multi-GPU or CPU training. We cover them in
separate sections.

* [Methods and tools for efficient training on a single GPU](perf_train_gpu_one): start here to learn common approaches that can help optimize GPU memory utilization, speed up the training, or both.
* [Multi-GPU training section](perf_train_gpu_many): explore this section to learn about further optimization methods that apply to a multi-GPU settings, such as data, tensor, and pipeline parallelism.
* [CPU training section](perf_train_cpu): learn about mixed precision training on CPU.
* [Efficient Training on Multiple CPUs](perf_train_cpu_many): learn about distributed CPU training.
* [Training on TPU with TensorFlow](perf_train_tpu_tf): if you are new to TPUs, refer to this section for an opinionated introduction to training on TPUs and using XLA.
* [Custom hardware for training](perf_hardware): find tips and tricks when building your own deep learning rig.
* [Hyperparameter Search using Trainer API](hpo_train)

## Inference

Efficient inference with large models in a production environment can be as challenging as training them. In the following
sections we go through the steps to run inference on CPU and single/multi-GPU setups.

* [Inference on a single CPU](perf_infer_cpu)
* [Inference on a single GPU](perf_infer_gpu_one)
* [Multi-GPU inference](perf_infer_gpu_multi)
* [XLA Integration for TensorFlow Models](tf_xla)

## Training and inference

Here you’ll find techniques, tips and tricks that apply whether you are training a model, or running inference with it.

* [Instantiating a big model](big_models)
* [Troubleshooting performance issues](debugging)

## Contribute

This document is far from being complete and a lot more needs to be added, so if you have additions or corrections to
make please don’t hesitate to open a PR or if you aren’t sure start an Issue and we can discuss the details there.

When making contributions that A is better than B, please try to include a reproducible benchmark and/or a link to the
source of that information (unless it comes directly from you).

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/performance.md)
