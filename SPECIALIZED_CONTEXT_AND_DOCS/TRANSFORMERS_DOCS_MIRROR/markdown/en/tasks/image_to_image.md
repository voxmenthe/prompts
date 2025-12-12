# Image-to-Image Task Guide

![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)

![Open In Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)

Image-to-Image task is the task where an application receives an image and outputs another image. This has various subtasks, including image enhancement (super resolution, low light enhancement, deraining and so on), image inpainting, and more.

This guide will show you how to:

* Use an image-to-image pipeline for super resolution task,
* Run image-to-image models for same task without a pipeline.

Note that as of the time this guide is released, `image-to-image` pipeline only supports super resolution task.

Let’s begin by installing the necessary libraries.


```
pip install transformers
```

We can now initialize the pipeline with a [Swin2SR model](https://huggingface.co/caidas/swin2SR-lightweight-x2-64). We can then infer with the pipeline by calling it with an image. As of now, only [Swin2SR models](https://huggingface.co/models?sort=trending&search=swin2sr) are supported in this pipeline.


```
from transformers import pipeline, infer_device
import torch
# automatically detects the underlying device type (CUDA, CPU, XPU, MPS, etc.)
device = infer_device()
pipe = pipeline(task="image-to-image", model="caidas/swin2SR-lightweight-x2-64", device=device)
```

Now, let’s load an image.


```
from PIL import Image
import requests

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/cat.jpg"
image = Image.open(requests.get(url, stream=True).raw)

print(image.size)
```


```
# (532, 432)
```

![Photo of a cat](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/cat.jpg)

We can now do inference with the pipeline. We will get an upscaled version of the cat image.


```
upscaled = pipe(image)
print(upscaled.size)
```


```
# (1072, 880)
```

If you wish to do inference yourself with no pipeline, you can use the `Swin2SRForImageSuperResolution` and `Swin2SRImageProcessor` classes of transformers. We will use the same model checkpoint for this. Let’s initialize the model and the processor.


```
from transformers import Swin2SRForImageSuperResolution, Swin2SRImageProcessor 

model = Swin2SRForImageSuperResolution.from_pretrained("caidas/swin2SR-lightweight-x2-64").to(device)
processor = Swin2SRImageProcessor("caidas/swin2SR-lightweight-x2-64")
```

`pipeline` abstracts away the preprocessing and postprocessing steps that we have to do ourselves, so let’s preprocess the image. We will pass the image to the processor and then move the pixel values to GPU.


```
pixel_values = processor(image, return_tensors="pt").pixel_values
print(pixel_values.shape)

pixel_values = pixel_values.to(device)
```

We can now infer the image by passing pixel values to the model.


```
import torch

with torch.no_grad():
  outputs = model(pixel_values)
```

Output is an object of type `ImageSuperResolutionOutput` that looks like below 👇


```
(loss=None, reconstruction=tensor([[[[0.8270, 0.8269, 0.8275,  ..., 0.7463, 0.7446, 0.7453],
          [0.8287, 0.8278, 0.8283,  ..., 0.7451, 0.7448, 0.7457],
          [0.8280, 0.8273, 0.8269,  ..., 0.7447, 0.7446, 0.7452],
          ...,
          [0.5923, 0.5933, 0.5924,  ..., 0.0697, 0.0695, 0.0706],
          [0.5926, 0.5932, 0.5926,  ..., 0.0673, 0.0687, 0.0705],
          [0.5927, 0.5914, 0.5922,  ..., 0.0664, 0.0694, 0.0718]]]],
       device='cuda:0'), hidden_states=None, attentions=None)
```

We need to get the `reconstruction` and post-process it for visualization. Let’s see how it looks like.


```
outputs.reconstruction.data.shape
# torch.Size([1, 3, 880, 1072])
```

We need to squeeze the output and get rid of axis 0, clip the values, then convert it to be numpy float. Then we will arrange axes to have the shape [1072, 880], and finally, bring the output back to range [0, 255].


```
import numpy as np

# squeeze, take to CPU and clip the values
output = outputs.reconstruction.data.squeeze().cpu().clamp_(0, 1).numpy()
# rearrange the axes
output = np.moveaxis(output, source=0, destination=-1)
# bring values back to pixel values range
output = (output * 255.0).round().astype(np.uint8)
Image.fromarray(output)
```

![Upscaled photo of a cat](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/cat_upscaled.png)

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/tasks/image_to_image.md)
