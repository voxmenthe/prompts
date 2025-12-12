# Accelerator selection

During distributed training, you can specify the number and order of accelerators (CUDA, XPU, MPS, HPU, etc.) to use. This can be useful when you have accelerators with different computing power and you want to use the faster accelerator first. Or you could only use a subset of the available accelerators. The selection process works for both [DistributedDataParallel](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) and [DataParallel](https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html). You don’t need Accelerate or [DeepSpeed integration](./main_classes/deepspeed).

This guide will show you how to select the number of accelerators to use and the order to use them in.

## Number of accelerators

For example, if there are 4 accelerators and you only want to use the first 2, run the command below.

torchrun

Accelerate

DeepSpeed

Use the `--nproc_per_node` to select how many accelerators to use.


```
torchrun --nproc_per_node=2  trainer-program.py ...
```

## Order of accelerators

To select specific accelerators to use and their order, use the environment variable appropriate for your hardware. This is often set on the command line for each run, but can also be added to your `~/.bashrc` or other startup config file.

For example, if there are 4 accelerators (0, 1, 2, 3) and you only want to run accelerators 0 and 2:

CUDA

Intel XPU


```
CUDA_VISIBLE_DEVICES=0,2 torchrun trainer-program.py ...
```

Only GPUs 0 and 2 are “visible” to PyTorch and are mapped to `cuda:0` and `cuda:1` respectively.  
To reverse the order (use GPU 2 as `cuda:0` and GPU 0 as `cuda:1`):


```
CUDA_VISIBLE_DEVICES=2,0 torchrun trainer-program.py ...
```

To run without any GPUs:


```
CUDA_VISIBLE_DEVICES= python trainer-program.py ...
```

You can also control the order of CUDA devices using `CUDA_DEVICE_ORDER`:

* Order by PCIe bus ID (matches `nvidia-smi`):


  ```
  export CUDA_DEVICE_ORDER=PCI_BUS_ID
  ```
* Order by compute capability (fastest first):


  ```
  export CUDA_DEVICE_ORDER=FASTEST_FIRST
  ```

Environment variables can be exported instead of being added to the command line. This is not recommended because it can be confusing if you forget how the environment variable was set up and you end up using the wrong accelerators. Instead, it is common practice to set the environment variable for a specific training run on the same command line.

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/accelerator_selection.md)
