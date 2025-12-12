# HQQ

[Half-Quadratic Quantization (HQQ)](https://github.com/mobiusml/hqq/) supports fast on-the-fly quantization for 8, 4, 3, 2, and even 1-bits. It doesn’t require calibration data, and it is compatible with any model modality (LLMs, vision, etc.).

HQQ further supports fine-tuning with [PEFT](https://huggingface.co/docs/peft) and is fully compatible with [torch.compile](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) for even faster inference and training.

Install HQQ with the following command to get the latest version and to build its corresponding CUDA kernels if you are using a cuda device. It also support Intel XPU with pure pytorch implementation.


```
pip install hqq
```

You can choose to either replace all the linear layers in a model with the same quantization config or dedicate a specific quantization config for specific linear layers.

replace all layers

specific layers only

Quantize a model by creating a [HqqConfig](/docs/transformers/v4.56.2/en/main_classes/quantization#transformers.HqqConfig) and specifying the `nbits` and `group_size` to replace for all the linear layers ([torch.nn.Linear](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html)) of the model.


```
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, HqqConfig

quant_config = HqqConfig(nbits=8, group_size=64)
model = transformers.AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B", 
    dtype=torch.float16, 
    device_map="auto", 
    quantization_config=quant_config
)
```

## Backends

HQQ supports various backends, including pure PyTorch and custom dequantization CUDA kernels. These backends are suitable for older GPUs and PEFT/QLoRA training.


```
from hqq.core.quantize import *

HQQLinear.set_backend(HQQBackend.PYTORCH)
```

For faster inference, HQQ supports 4-bit fused kernels (torchao and Marlin) after a model is quantized. These can reach up to 200 tokens/sec on a single 4090. The example below demonstrates enabling the torchao\_int4 backend.


```
from hqq.utils.patching import prepare_for_inference

prepare_for_inference("model", backend="torchao_int4")
```

Refer to the [Backend](https://github.com/mobiusml/hqq/#backend) guide for more details.

## Resources

Read the [Half-Quadratic Quantization of Large Machine Learning Models](https://mobiusml.github.io/hqq_blog/) blog post for more details about HQQ.

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/quantization/hqq.md)
