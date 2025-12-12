*This model was released on 2024-12-17 and added to Hugging Face Transformers on 2024-12-17.*

# Falcon3

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white)

## Overview

[Falcon3](https://falconllm.tii.ae/falcon3/index.html) represents a natural evolution from previous releases, emphasizing expanding the models’ science, math, and code capabilities. This iteration includes five base models: Falcon3-1B-Base, Falcon3-3B-Base, Falcon3-Mamba-7B-Base, Falcon3-7B-Base, and Falcon3-10B-Base. In developing these models, we incorporated several key innovations aimed at improving the models’ performances while reducing training costs:

One pre-training: We conducted a single large-scale pretraining run on the 7B model, using 2048 H100 GPU chips, leveraging 14 trillion tokens featuring web, code, STEM, and curated high-quality and multilingual data.
Depth up-scaling for improved reasoning: Building on recent studies on the effects of model depth, we upscaled the 7B model to a 10B parameters model by duplicating the redundant layers and continuing pre-training with 2TT of high-quality data. This yielded Falcon3-10B-Base which achieves state-of-the-art zero-shot and few-shot performance for models under 13B parameters.
Knowledge distillation for better tiny models: To provide compact and efficient alternatives, we developed Falcon3-1B-Base and Falcon3-3B-Base by leveraging pruning and knowledge distillation techniques, using less than 100GT of curated high-quality data, thereby redefining pre-training efficiency.

## Resources

* [Blog post](https://huggingface.co/blog/falcon3)
* [Models on Huggingface](https://huggingface.co/collections/tiiuae/falcon3-67605ae03578be86e4e87026)

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/falcon3.md)
