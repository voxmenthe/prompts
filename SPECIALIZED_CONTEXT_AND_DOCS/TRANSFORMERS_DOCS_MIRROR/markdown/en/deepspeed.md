# DeepSpeed

[DeepSpeed](https://www.deepspeed.ai/) is designed to optimize distributed training for large models with data, model, pipeline, and even a combination of all three [parallelism](./perf_train_gpu_many) strategies to provide better memory efficiency and faster training speeds. This is achieved with the [Zero Redundancy Optimizer (ZeRO)](https://hf.co/papers/1910.02054) which consists of three stages.

| ZeRO stage | description |
| --- | --- |
| 1 | partition optimizer states |
| 2 | partition optimizer and gradient states |
| 3 | partition optimizer, gradient, and parameters |

Each stage progressively saves more memory, allowing really large models to fit and train on a single GPU. All ZeRO stages, offloading optimizer memory and computations from the GPU to the CPU are integrated with [Trainer](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Trainer). Provide a config file or one of the example templates to [Trainer](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Trainer) to enable DeepSpeed features.

This guide walks you through setting up a DeepSpeed config file, how to enable its features in [Trainer](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Trainer), and deploy for training.

Install DeepSpeed from either PyPI or Transformers. For more detailed installation instructions, refer to the DeepSpeed [installation](https://www.deepspeed.ai/tutorials/advanced-install/) or GitHUB [README](https://github.com/microsoft/deepspeed#installation).

PyPI

Transformers


```
pip install deepspeed
```

Refer to the [DeepSpeed CUDA installation](./debugging#deepspeed-cuda-issues) if you’re having trouble with your installation. While DeepSpeed has a pip installable package, it is highly recommended to [install it from source](https://www.deepspeed.ai/tutorials/advanced-install/#install-deepspeed-from-source) to ensure it matches your hardware and to support certain features which aren’t available in the PyPI distribution.

DeepSpeed provides a tool for estimating the required CPU and GPU memory for the parameters, optimizer and gradient states. You’ll also to need to reserve some memory for the CUDA kernels and activations.

Run the command below to check the memory requirements for [bigscience/T0\_3B](https://huggingface.co/docs/transformers/main/en/bigscience/T0_3B) on a single GPU.


```
$ python -c 'from transformers import AutoModel; \
from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live; \
model = AutoModel.from_pretrained("bigscience/T0_3B"); \
estimate_zero3_model_states_mem_needs_all_live(model, num_gpus_per_node=1, num_nodes=1)'
[...]
Estimated memory needed for params, optim states and gradients for a:
HW: Setup with 1 node, 1 GPU per node.
SW: Model with 2783M total params, 65M largest layer params.
  per CPU  |  per GPU |   Options
   70.00GB |   0.25GB | offload_param=cpu , offload_optimizer=cpu , zero_init=1
   70.00GB |   0.25GB | offload_param=cpu , offload_optimizer=cpu , zero_init=0
   62.23GB |   5.43GB | offload_param=none, offload_optimizer=cpu , zero_init=1
   62.23GB |   5.43GB | offload_param=none, offload_optimizer=cpu , zero_init=0
    0.37GB |  46.91GB | offload_param=none, offload_optimizer=none, zero_init=1
   15.56GB |  46.91GB | offload_param=none, offload_optimizer=none, zero_init=0
```

If you have enough GPU memory, disable CPU and NVMe offload to speed everything up.

## Choosing a ZeRO stage

Consider the table below to help you choose the appropriate ZeRO stage for training because there is a trade-off between training speed and memory usage. The table orders the ZeRO stages from fastest to slowest and from least memory usage to most.

| fastest | least memory usage |
| --- | --- |
| ZeRO-1 | ZeRO-3 + offload |
| ZeRO-2 | ZeRO-3 |
| ZeRO-2 + offload | ZeRO-2 + offload |
| ZeRO-3 | ZeRO-2 |
| ZeRO-3 + offload | ZeRO-1 |

Decide the type of performance you’re optimizing for, speed or memory, and then work backwards to discover the best ZeRO stage for your use case. For example, if you’re optimizing for speed, start with the fastest ZeRO stage and if you run out of memory, try the next stage which is slower but more memory efficient.

## Config file

Once you’ve decided on a ZeRO stage, set up a config file to enable DeepSpeed with [Trainer](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Trainer). The config file contains all the parameters for how to configure and set up your training. When the training script is executed, DeepSpeed logs the configuration from [Trainer](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Trainer) to the console so you can see exactly what’s being used.

Find a complete list of DeepSpeed configuration options on the [DeepSpeed Configuration JSON](https://www.deepspeed.ai/docs/config-json/) reference. There are also practical examples of various DeepSpeed configuration examples in the [DeepSpeedExamples](https://github.com/microsoft/DeepSpeedExamples) main [DeepSpeed](https://github.com/microsoft/DeepSpeed) repository. Run the command below to quickly find specific examples.


```
git clone https://github.com/microsoft/DeepSpeedExamples
cd DeepSpeedExamples
find . -name '*json'
# find examples with the Lamb optimizer
grep -i Lamb $(find . -name '*json')
```

The config file is passed as a path to a JSON file if you’re training from the command line interface or as a nested dict object if you’re using [Trainer](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Trainer) in a notebook.

path to file

nested dict


```
TrainingArguments(
    deepspeed="path/to/deepspeed_config.json",
    ...,
)
```

### DeepSpeed versus Trainer parameters

There are three types of config parameters.

1. Some config parameters are shared by DeepSpeed and [Trainer](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Trainer) making it difficult to identify errors when there are conflicting definitions. In this case, configure these parameters from the [Trainer](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Trainer) command line arguments.
2. Some config parameters are automatically derived from the model configuration and don’t need to be manually configured. [Trainer](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Trainer) uses the config value `auto` to set the most correct or efficient option. You could define these parameters explicitly, but you must take care to ensure the [Trainer](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Trainer) and DeepSpeed config parameters match. Mismatches may cause training to fail in very difficult to detect ways.
3. Some config parameters are specific to DeepSpeed and should be manually set based on your training requirements.

There are two ways to modify the config parameters.

Some values, such as `scheduler.params.total_num_steps`, are calculated by [Trainer](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Trainer) during training.

1. Create or load a DeepSpeed config to use as the main config.
2. Create a [TrainingArguments](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.TrainingArguments) object based on the DeepSpeed config values.

### ZeRO stage

Each ZeRO stage config is defined in `zero_optimization`.

For a more detailed explanation of each parameter, refer to the [DeepSpeed Configuration JSON](https://www.deepspeed.ai/docs/config-json/) reference. These parameters must be set up with DeepSpeed because [Trainer](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Trainer) doesn’t provide equivalent command line arguments.

DeepSpeed doesn’t validate parameter names and any typos will fallback on the parameters default setting. Observe the DeepSpeed engine startup log messages to see what values are being used.

ZeRO-1

ZeRO-2

ZeRO-3

ZeRO-1 shards the optimizer states across GPUs and you can expect a small speed up.


```
{
    "zero_optimization": {
        "stage": 1
    }
}
```

### NVMe

[ZeRO-Infinity](https://hf.co/papers/2104.07857) offloads model states to the CPU and/or NVMe to save even more memory. Smart partitioning and tiling algorithms allow each GPU to send and receive very small amounts of data during offloading such that a modern NVMe can fit an even larger total memory pool than is available to your training process. ZeRO-Infinity requires ZeRO-3.

Depending on the CPU and NVMe memory available, you can offload both the [optimizer states](https://www.deepspeed.ai/docs/config-json/#optimizer-offloading) and [parameters](https://www.deepspeed.ai/docs/config-json/#parameter-offloading), just one of them, or none of them. Make sure the `nvme_path` points to a NVMe device, because while it still works with a regular hard drive or solid state drive, it’ll be significantly slower. With a modern NVMe, you can expect peak transfer speeds of ~3.5GB/s for read operations and ~3GB/s for write operations.

Consider running a [benchmark](https://github.com/microsoft/DeepSpeed/issues/998) on your training setup to determine the optimal `aio` configuration.

The example ZeRO-3 and ZeRO-Infinity config below sets most of the parameter values to `auto`, but you can also manually set configure these values.


```
{
    "fp16": {
        "enabled": "auto",
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },

    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": "auto",
            "betas": "auto",
            "eps": "auto",
            "weight_decay": "auto"
        }
    },

    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": "auto",
            "warmup_max_lr": "auto",
            "warmup_num_steps": "auto"
        }
    },

    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "nvme",
            "nvme_path": "/local_nvme",
            "pin_memory": true,
            "buffer_count": 4,
            "fast_init": false
        },
        "offload_param": {
            "device": "nvme",
            "nvme_path": "/local_nvme",
            "pin_memory": true,
            "buffer_count": 5,
            "buffer_size": 1e8,
            "max_in_cpu": 1e9
        },
        "aio": {
            "block_size": 262144,
            "queue_depth": 32,
            "thread_count": 1,
            "single_submit": false,
            "overlap_events": true
        },
        "overlap_comm": true,
        "contiguous_gradients": true,
        "sub_group_size": 1e9,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": true
    },

    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "steps_per_print": 2000,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": false
}
```

## Training features

DeepSpeed supports many training features that can be configured in the config file. This section describes some of the most important features.

### Gradient checkpointing

Gradient checkpointing saves memory by only storing *some* of the intermediate activations instead of storing *all* of them. It is useful for fitting larger models on the GPU without running out of memory or to increase the batch size for better performance. Training speed is slower though.

* For a Transformers model, set `model.gradient_checkpointing_enable()` or add `--gradient_checkpointing` in the [TrainingArguments](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.TrainingArguments).
* For a non-Transformers model, use the DeepSpeed [Activation Checkpointing API](https://deepspeed.readthedocs.io/en/latest/activation-checkpointing.html). Replacing Transformers modeling code and [torch.utils.checkpoint](https://pytorch.org/docs/stable/checkpoint.html) with the DeepSpeed API gives you more flexibility because you can offload the forward activations to the CPU memory instead of recalculating them.

### Batch size

The batch size can be automatically configured or manually set. When you choose the `"auto"` option, [Trainer](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Trainer) sets `train_micro_batch_size_per_gpu` and `train_batch_size` to the value of `world_size * per_device_train_batch_size * gradient_accumulation_steps`.


```
{
    "train_micro_batch_size_per_gpu": "auto",
    "train_batch_size": "auto"
}
```

### Communication data type

A separate data type is used for communication collectives like reduction, gathering and scattering operations.

All gather and scatter operations are performed in the same data type the data is in. For example, if you’re training in bf16, the data is also gathered in bf16 because gathering is a non-lossy operation.

Reduce operations are lossy, for example, when gradients are averaged across multiple GPUs. When the communication is done in fp16 or bf16, it’s more likely to be lossy because adding multiple numbers in low precision isn’t exact. This is especially the case with bf16 which has a lower precision than fp16. For this reason, fp16 is the default for reduction operations because the loss is minimal when averaging gradients.

Choose the communication data type by setting the `communication_data_type` parameter in the config file. For example, choosing fp32 adds a small amount of overhead but ensures the reduction operation is accumulated in fp32 and when it is ready, it’s downcasted to whichever half-precision data type you’re training in.


```
{
    "communication_data_type": "fp32"
}
```

### Gradient accumulation

Gradient accumulation accumulates gradients over several mini-batches of data before updating parameters. It stores less gradients and enables training with a larger *effective batch size*. Training speed is slower though, but it’s useful for overcoming memory constraints.

Gradient accumulation can be automatically configured or manually set. When you choose the `"auto"` option, [Trainer](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Trainer) sets it to the value of `gradient_accumulation_steps`.


```
{
    "gradient_accumulation_steps": "auto"
}
```

### Gradient clipping

Gradient clipping is useful for preventing exploding gradients which can lead to instability during training. It sets a maximum threshold value and rescales the gradients if their norm exceeds the threshold.

Gradient clipping can be automatically configured or manually set. When you choose the `"auto"` option, [Trainer](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Trainer) sets it to the value of `max_grad_norm`.


```
{
    "gradient_clipping": "auto"
}
```

### Mixed precision training

Mixed precision accelerates training speed by performing some calculations in half-precision, but it also maintains some calculations in full-precision to preserve accuracy. DeepSpeed supports fp32, fp16, and bf16 data types.

fp32

fp16

bf16

Train in fp32 if a model wasn’t pretrained in mixed precision because it may cause underflow or overflow errors. Disable fp16, the default, in this case.


```
{
    "fp16": {
        "enabled": false
    }
}
```

For Ampere GPUs and PyTorch 1.7+, the more efficient [tf32](https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices) mode is automatically enabled for some operations but the results are still in fp32. Configure it in [Trainer](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Trainer) by setting `--tf32` to enable it, and `--tf32 0` or `--no_tf32` to disable it.

### Optimizer and scheduler

DeepSpeed and Transformers optimizers and schedulers can be mixed and matched if `offload_optimizer` isn’t enabled. When `offload_optimizer` is enabled, use a non-DeepSpeed optimizer (except for LAMB) as long as it has it a CPU and GPU implementation.

Set the optimizer and scheduler parameters for the config file from the command line to avoid hard to find errors. For example, if the learning rate is set to a different value in another place, you can override it from the command line.

optimizer

scheduler

DeepSpeed offers several [optimizers](https://www.deepspeed.ai/docs/config-json/#optimizer-parameters) (Adam, AdamW, OneBitAdam, and LAMB) but you can also import other optimizers from PyTorch. If you don’t configure the optimizer in the config, [Trainer](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Trainer) automatically selects AdamW and either uses the supplied values or the default values for the following parameters from the command line: `lr`, `adam_beta1`, `adam_beta2`, `adam_epsilon`, `weight_decay`.

You can set the parameters to `"auto"` or manually input your own values.


```
{
   "optimizer": {
       "type": "AdamW",
       "params": {
         "lr": "auto",
         "betas": "auto",
         "eps": "auto",
         "weight_decay": "auto"
       }
   }
}
```

Use an unsupported optimizer by adding the following to the top level configuration.


```
{
   "zero_allow_untested_optimizer": true
}
```

From DeepSpeed 0.8.3+, if you want to use offload, you’ll also need to add the following to the top level configuration because offload works best with DeepSpeed’s CPU Adam optimizer.


```
{
   "zero_force_ds_cpu_optimizer": false
}
```

### Universal checkpointing

[Universal Checkpointing](https://www.deepspeed.ai/tutorials/universal-checkpointing) saves and loads model, optimizer and training scheduler states across different model architectures, parallelism techniques, and training configurations. By saving them in a Universal format, it enables easier model training continuation and fine-tuning.

Resume training with a Universal checkpoint by setting `load_universal` to `true` in the config file.


```
{
    "checkpoint": {
        "load_universal": true
    }
}
```

## Deploy

DeepSpeed can be deployed with its native launcher, [torchrun](https://pytorch.org/docs/stable/elastic/run.html) or [Accelerate](https://huggingface.co/docs/accelerate/basic_tutorials/launch#using-accelerate-launch).

Add the `--deepspeed ds_config.json` argument to [Trainer](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Trainer) in the command line. It is recommended to use DeepSpeeds [add\_config\_arguments](https://deepspeed.readthedocs.io/en/latest/initialize.html#argument-parsing) utility to add any other command line arguments to your code.

multi-GPU

single-GPU

To deploy DeepSpeed on multiple GPUs, add `--num_gpus`. You don’t need to add `--num_gpus` if you’re planning on using all available GPUs.


```
deepspeed --num_gpus=2 examples/pytorch/translation/run_translation.py \
--deepspeed tests/deepspeed/ds_config_zero3.json \
--model_name_or_path google-t5/t5-small --per_device_train_batch_size 1 \
--output_dir output_dir --overwrite_output_dir --fp16 \
--do_train --max_train_samples 500 --num_train_epochs 1 \
--dataset_name wmt16 --dataset_config "ro-en" \
--source_lang en --target_lang ro
```

### Multi-node

A multi-node setup consists of multiple nodes, where each node has one of more GPUs running a workload. DeepSpeed expects a shared storage system, but if this is not the case, you need to adjust the config file to include a [checkpoint](https://www.deepspeed.ai/docs/config-json/#checkpoint-options) to allow loading without access to a shared filesystem.


```
{
  "checkpoint": {
    "use_node_local_storage": true
  }
}
```

You could also use the `--save_on_each_node` parameter in [TrainingArguments](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.TrainingArguments) to automatically add the above `checkpoint` to your config.

The examples below for the torchrun and DeepSpeed launcher shows how to deploy two nodes with eight GPUs each. Access the first node with `ssh hostname1` and the second node with `ssh hostname2`. Both nodes must be able to communicate with each other locally over ssh without a password.

torchrun

DeepSpeed

With [torchrun](https://pytorch.org/docs/stable/elastic/run.html), ssh to each node and run the following command on both of them. The launcher waits until both nodes are synchronized before launching the training.


```
torchrun --nproc_per_node=8 --nnode=2 --node_rank=0 --master_addr=hostname1 \
--master_port=9901 your_program.py <normal cl args> --deepspeed ds_config.json
```

### Slurm

[Slurm](https://slurm.schedmd.com/documentation.html) is a cluster management and job scheduling system. An example Slurm script is shown below.


```
#SBATCH --job-name=test-nodes        # name
#SBATCH --nodes=2                    # nodes
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=10           # number of cores per tasks
#SBATCH --gres=gpu:8                 # number of gpus
#SBATCH --time 20:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=%x-%j.out           # output file name

export GPUS_PER_NODE=8
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=9901

srun --jobid $SLURM_JOBID bash -c 'python -m torch.distributed.run \
 --nproc_per_node $GPUS_PER_NODE --nnodes $SLURM_NNODES --node_rank $SLURM_PROCID \
 --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
your_program.py <normal cl args> --deepspeed ds_config.json'
```

Launch training simultaneously on all nodes with the command below.


```
sbatch launch.slurm
```

### Jupyter Notebook

To use DeepSpeed in a Jupyter Notebook, you need to emulate a distributed environment because the launcher doesn’t support deployment from a notebook. This is only supported for one GPU. To use multiple GPUs, you must use a multi-process environment, which means you have to use the DeepSpeed launcher which can’t be emulated as shown here.


```
# emulate a launcher in the notebook
import os

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "9994"  # modify if RuntimeError: Address already in use
os.environ["RANK"] = "0"
os.environ["LOCAL_RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"

training_args = TrainingArguments(..., deepspeed="ds_config_zero3.json")
trainer = Trainer(...)
trainer.train()
```

Create a config file on the fly in the notebook in the current directory with a dedicated cell.


```
%%bash
cat <<'EOT' > ds_config_zero3.json
{
    "fp16": {
        "enabled": "auto",
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },

    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": "auto",
            "betas": "auto",
            "eps": "auto",
            "weight_decay": "auto"
        }
    },

    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": "auto",
            "warmup_max_lr": "auto",
            "warmup_num_steps": "auto"
        }
    },

    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": true
        },
        "overlap_comm": true,
        "contiguous_gradients": true,
        "sub_group_size": 1e9,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": true
    },

    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "steps_per_print": 2000,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": false
}
EOT
```

If the training script is in a file and not a notebook cell, launch DeepSpeed from the shell in the notebook cell.


```
!git clone https://github.com/huggingface/transformers
!cd transformers; deepspeed examples/pytorch/translation/run_translation.py ...
```

Another option is to use `%%bash` to run the shell program without emulating the distributed environment. However, you won’t be able to view the logs until training is complete.


```
%%bash

git clone https://github.com/huggingface/transformers
cd transformers
deepspeed examples/pytorch/translation/run_translation.py ...
```

## Save model weights

DeepSpeed stores the main fp32 weights in custom checkpoint optimizer files (`global_step*/*optim_states.pt`) which are saved under the normal checkpoint.

### fp16

ZeRO-2 saves the model weights in fp16. To save the weights in fp16 for ZeRO-3, set `"stage3_gather_16bit_weights_on_model_save": true` in the config file, because the weights are distributed across multiple GPUs.

If you don’t, [Trainer](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Trainer) won’t save the weights in fp16 and won’t create a `pytorch_model.bin` file. This is because DeepSpeed’s state\_dict contains a placeholder instead of the real weights, so you won’t be able to load it.


```
{
    "zero_optimization": {
        "stage": 3,
        "stage3_gather_16bit_weights_on_model_save": true
    }
}
```

### fp32

Unless you have a lot of free CPU memory, fp32 weights shouldn’t be saved during training because it can require a lot of memory. It is usually best to save the fp32 weights offline after training is complete.

offline

online

DeepSpeed provides a [zero\_to\_fp32.py](https://github.com/microsoft/DeepSpeed/blob/91829476a8fd4d0d9268c03c1d56795d20a51c12/deepspeed/utils/zero_to_fp32.py#L14) script at the top-level checkpoint folder for extracting weights at any point. This is a standalone script and you don’t need a config file or [Trainer](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Trainer).

For example, if your checkpoint folder looks like the one shown below, then you can run the following command to create and consolidate the fp32 weights from multiple GPUs into a single `pytorch_model.bin` file. The script automatically discovers the subfolder `global_step1` which contains the checkpoint.


```
$ ls -l output_dir/checkpoint-1/
-rw-rw-r-- 1 stas stas 1.4K Mar 27 20:42 config.json
drwxrwxr-x 2 stas stas 4.0K Mar 25 19:52 global_step1/
-rw-rw-r-- 1 stas stas   12 Mar 27 13:16 latest
-rw-rw-r-- 1 stas stas 827K Mar 27 20:42 optimizer.pt
-rw-rw-r-- 1 stas stas 231M Mar 27 20:42 pytorch_model.bin
-rw-rw-r-- 1 stas stas  623 Mar 27 20:42 scheduler.pt
-rw-rw-r-- 1 stas stas 1.8K Mar 27 20:42 special_tokens_map.json
-rw-rw-r-- 1 stas stas 774K Mar 27 20:42 spiece.model
-rw-rw-r-- 1 stas stas 1.9K Mar 27 20:42 tokenizer_config.json
-rw-rw-r-- 1 stas stas  339 Mar 27 20:42 trainer_state.json
-rw-rw-r-- 1 stas stas 2.3K Mar 27 20:42 training_args.bin
-rwxrw-r-- 1 stas stas 5.5K Mar 27 13:16 zero_to_fp32.py*
```

Run `python zero_to_fp32.py -h` for more usage details. The script requires 2x the general RAM of the final fp32 weights.


```
python zero_to_fp32.py . pytorch_model.bin
```

## Non-Trainer integration

DeepSpeed also works with Transformers without [Trainer](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Trainer). The [HfDeepSpeedConfig](/docs/transformers/v4.56.2/en/main_classes/deepspeed#transformers.integrations.HfDeepSpeedConfig) is responsible for gathering ZeRO-3 parameters and partitioning a model across multiple GPUs when [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) is called.

You must instantiate [HfDeepSpeedConfig](/docs/transformers/v4.56.2/en/main_classes/deepspeed#transformers.integrations.HfDeepSpeedConfig) before loading a model to efficiently deploy ZeRO-3.

pretrained model

non-pretrained model


```
from transformers.integrations import HfDeepSpeedConfig
from transformers import AutoModel
import deepspeed

# DeepSpeed config object or path to the file
ds_config = {...}
# must run before instantiating the model to detect ZeRO-3
dschf = HfDeepSpeedConfig(ds_config)  # keep this object alive
model = AutoModel.from_pretrained("openai-community/gpt2")
engine = deepspeed.initialize(model=model, config_params=ds_config, ...)
```

## Troubleshoot

One of the first things to check when you encounter an error is whether DeepSpeed is the cause (because often it isn’t). Retry your setup without DeepSpeed, and if the error persists, report the issue. If the issue is unrelated to the Transformers integration, please open the issue on the DeepSpeed [repository](https://github.com/microsoft/DeepSpeed).

For issues related to the Transformers integration, please provide the following information.

* The full DeepSpeed config file.
* The command line arguments for [Trainer](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Trainer) or the [TrainingArguments](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.TrainingArguments) if you’re scripting the [Trainer](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Trainer) setup yourself (don’t dump the entire [TrainingArguments](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.TrainingArguments) which contains many irrelevant entries).
* The outputs of the following commands.


  ```
  python -c 'import torch; print(f"torch: {torch.__version__}")'
  python -c 'import transformers; print(f"transformers: {transformers.__version__}")'
  python -c 'import deepspeed; print(f"deepspeed: {deepspeed.__version__}")'
  ```
* A link to a Google Colab notebook to reproduce the issue.
* A standard or non-custom dataset or an existing example to reproduce the issue.

The following sections provide a guide for resolving two of the most common issues.

### Process killed at startup

When the DeepSpeed process is killed during launch without a traceback, that usually means the program tried to allocate more CPU memory than is available on your system. Or the process may have tried to allocate more CPU memory than allowed, leading the OS kernel to terminate the process.

In this case, check whether your config file has either `offload_optimizer`, `offlload_param`, or both configured to offload to the CPU.

If you have NVM3 and ZeRO-3 set up, experiment with offloading to the NVMe ([estimate](https://deepspeed.readthedocs.io/en/latest/memory.html) the memory requirements of a model first) instead.

### NaN loss

NaN loss often occurs when a model is pretrained in bf16 and you try to use it with fp16 (especially relevant to TPU trained models). To resolve this, use fp32 or bf16 if your hardware (TPUs, Ampere GPUs or newer) supports it.

It is also possible that fp16 is causing overflow. For example, if your config file looks like the one below, you may see the following overflow errors in the logs.


```
{
    "fp16": {
        "enabled": "auto",
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    }
}
```

The `OVERFLOW!` error below is a result of the DeepSpeed loss scaler unable to find a scaling coefficient to overcome the loss overflow. Try a higher `initial_scale_power` value in this case (32 usually works).


```
0%|                                                                                                                             | 0/189 [00:00<?, ?it/s]
 [deepscale] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 262144, reducing to 262144
  1%|▌                                                                                                                    | 1/189 [00:00<01:26,  2.17it/s]
 [deepscale] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 262144, reducing to 131072.0
  1%|█▏
 [...]
 [deepscale] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 1, reducing to 1
 14%|████████████████▌                                                                                                   | 27/189 [00:14<01:13,  2.21it/s]
 [deepscale] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 1, reducing to 1
 15%|█████████████████▏                                                                                                  | 28/189 [00:14<01:13,  2.18it/s]
 [deepscale] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 1, reducing to 1
 15%|█████████████████▊                                                                                                  | 29/189 [00:15<01:13,  2.18it/s]
 [deepscale] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 1, reducing to 1
[...]
```

## Resources

DeepSpeed is a powerful technology for scaling large model training. To learn more about DeepSpeed, take a look at their [blog posts](https://www.microsoft.com/en-us/research/search/?q=deepspeed), [documentation](https://www.deepspeed.ai/getting-started/), and [GitHub](https://github.com/microsoft/deepspeed).

The papers below provide additional details about ZeRO.

* [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://hf.co/papers/1910.02054)
* [ZeRO-Offload: Democratizing Billion-Scale Model Training](https://hf.co/papers/2101.06840)
* [ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning](https://hf.co/papers/2104.07857)

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/deepspeed.md)
