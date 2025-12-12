# SpQR

The [SpQR](https://hf.co/papers/2306.03078) quantization algorithm involves a 16x16 tiled bi-level group 3-bit quantization structure with sparse outliers.

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/spqr-diagram.png)

To quantize a model with SpQR, refer to the [Vahe1994/SpQR](https://github.com/Vahe1994/SpQR) repository.

Load a SpQR-quantized model with [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained).


```
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

quantized_model = AutoModelForCausalLM.from_pretrained(
    "elvircrn/Llama-2-7b-SPQR-3Bit-16x16-red_pajama-hf",
    dtype=torch.half,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("elvircrn/Llama-2-7b-SPQR-3Bit-16x16-red_pajama-hf")
```

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/quantization/spqr.md)
