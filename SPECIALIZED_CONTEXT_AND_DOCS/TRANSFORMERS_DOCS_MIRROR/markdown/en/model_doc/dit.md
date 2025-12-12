*This model was released on 2022-03-04 and added to Hugging Face Transformers on 2022-03-10.*

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white)

# DiT

[DiT](https://huggingface.co/papers/2203.02378) is an image transformer pretrained on large-scale unlabeled document images. It learns to predict the missing visual tokens from a corrupted input image. The pretrained DiT model can be used as a backbone in other models for visual document tasks like document image classification and table detection.

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/dit_architecture.jpg)

You can find all the original DiT checkpoints under the [Microsoft](https://huggingface.co/microsoft?search_models=dit) organization.

Refer to the [BEiT](./beit) docs for more examples of how to apply DiT to different vision tasks.

The example below demonstrates how to classify an image with [Pipeline](/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.Pipeline) or the [AutoModel](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel) class.

Pipeline

AutoModel


```
import torch
from transformers import pipeline

pipeline = pipeline(
    task="image-classification",
    model="microsoft/dit-base-finetuned-rvlcdip",
    dtype=torch.float16,
    device=0
)
pipeline("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/dit-example.jpg")
```

## Notes

* The pretrained DiT weights can be loaded in a [BEiT] model with a modeling head to predict visual tokens.


  ```
  from transformers import BeitForMaskedImageModeling

  model = BeitForMaskedImageModeling.from_pretraining("microsoft/dit-base")
  ```

## Resources

* Refer to this [notebook](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/DiT/Inference_with_DiT_(Document_Image_Transformer)_for_document_image_classification.ipynb) for a document image classification inference example.

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/dit.md)
