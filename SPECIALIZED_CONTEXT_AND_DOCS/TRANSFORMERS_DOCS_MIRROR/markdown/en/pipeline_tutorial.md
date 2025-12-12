# Pipeline

The [Pipeline](/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.Pipeline) is a simple but powerful inference API that is readily available for a variety of machine learning tasks with any model from the Hugging Face [Hub](https://hf.co/models).

Tailor the [Pipeline](/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.Pipeline) to your task with task specific parameters such as adding timestamps to an automatic speech recognition (ASR) pipeline for transcribing meeting notes. [Pipeline](/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.Pipeline) supports GPUs, Apple Silicon, and half-precision weights to accelerate inference and save memory.

Transformers has two pipeline classes, a generic [Pipeline](/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.Pipeline) and many individual task-specific pipelines like [TextGenerationPipeline](/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.TextGenerationPipeline) or [VisualQuestionAnsweringPipeline](/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.VisualQuestionAnsweringPipeline). Load these individual pipelines by setting the task identifier in the `task` parameter in [Pipeline](/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.Pipeline). You can find the task identifier for each pipeline in their API documentation.

Each task is configured to use a default pretrained model and preprocessor, but this can be overridden with the `model` parameter if you want to use a different model.

For example, to use the [TextGenerationPipeline](/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.TextGenerationPipeline) with [Gemma 2](./model_doc/gemma2), set `task="text-generation"` and `model="google/gemma-2-2b"`.


```
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="google/gemma-2-2b")
pipeline("the secret to baking a really good cake is ")
[{'generated_text': 'the secret to baking a really good cake is 1. the right ingredients 2. the'}]
```

When you have more than one input, pass them as a list.


```
from transformers import pipeline, infer_device

device = infer_device()

pipeline = pipeline(task="text-generation", model="google/gemma-2-2b", device=device)
pipeline(["the secret to baking a really good cake is ", "a baguette is "])
[[{'generated_text': 'the secret to baking a really good cake is 1. the right ingredients 2. the'}],
 [{'generated_text': 'a baguette is 100% bread.\n\na baguette is 100%'}]]
```

This guide will introduce you to the [Pipeline](/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.Pipeline), demonstrate its features, and show how to configure its various parameters.

## Tasks

[Pipeline](/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.Pipeline) is compatible with many machine learning tasks across different modalities. Pass an appropriate input to the pipeline and it will handle the rest.

Here are some examples of how to use [Pipeline](/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.Pipeline) for different tasks and modalities.

summarization

automatic speech recognition

image classification

visual question answering


```
from transformers import pipeline

pipeline = pipeline(task="summarization", model="google/pegasus-billsum")
pipeline("Section was formerly set out as section 44 of this title. As originally enacted, this section contained two further provisions that 'nothing in this act shall be construed as in any wise affecting the grant of lands made to the State of California by virtue of the act entitled 'An act authorizing a grant to the State of California of the Yosemite Valley, and of the land' embracing the Mariposa Big-Tree Grove, approved June thirtieth, eighteen hundred and sixty-four; or as affecting any bona-fide entry of land made within the limits above described under any law of the United States prior to the approval of this act.' The first quoted provision was omitted from the Code because the land, granted to the state of California pursuant to the Act cite, was receded to the United States. Resolution June 11, 1906, No. 27, accepted the recession.")
[{'summary_text': 'Instructs the Secretary of the Interior to convey to the State of California all right, title, and interest of the United States in and to specified lands which are located within the Yosemite and Mariposa National Forests, California.'}]
```

## Parameters

At a minimum, [Pipeline](/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.Pipeline) only requires a task identifier, model, and the appropriate input. But there are many parameters available to configure the pipeline with, from task-specific parameters to optimizing performance.

This section introduces you to some of the more important parameters.

### Device

[Pipeline](/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.Pipeline) is compatible with many hardware types, including GPUs, CPUs, Apple Silicon, and more. Configure the hardware type with the `device` parameter. By default, [Pipeline](/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.Pipeline) runs on a CPU which is given by `device=-1`.

GPU

Apple silicon

To run [Pipeline](/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.Pipeline) on a GPU, set `device` to the associated CUDA device id. For example, `device=0` runs on the first GPU.


```
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="google/gemma-2-2b", device=0)
pipeline("the secret to baking a really good cake is ")
```

You could also let [Accelerate](https://hf.co/docs/accelerate/index), a library for distributed training, automatically choose how to load and store the model weights on the appropriate device. This is especially useful if you have multiple devices. Accelerate loads and stores the model weights on the fastest device first, and then moves the weights to other devices (CPU, hard drive) as needed. Set `device_map="auto"` to let Accelerate choose the device.

Make sure have [Accelerate](https://hf.co/docs/accelerate/basic_tutorials/install) is installed.


```
!pip install -U accelerate
```


```
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="google/gemma-2-2b", device_map="auto")
pipeline("the secret to baking a really good cake is ")
```

### Batch inference

[Pipeline](/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.Pipeline) can also process batches of inputs with the `batch_size` parameter. Batch inference may improve speed, especially on a GPU, but it isn’t guaranteed. Other variables such as hardware, data, and the model itself can affect whether batch inference improves speed. For this reason, batch inference is disabled by default.

In the example below, when there are 4 inputs and `batch_size` is set to 2, [Pipeline](/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.Pipeline) passes a batch of 2 inputs to the model at a time.


```
from transformers import pipeline, infer_device()

device = infer_device()

pipeline = pipeline(task="text-generation", model="google/gemma-2-2b", device=device, batch_size=2)
pipeline(["the secret to baking a really good cake is", "a baguette is", "paris is the", "hotdogs are"])
[[{'generated_text': 'the secret to baking a really good cake is to use a good cake mix.\n\ni’'}],
 [{'generated_text': 'a baguette is'}],
 [{'generated_text': 'paris is the most beautiful city in the world.\n\ni’ve been to paris 3'}],
 [{'generated_text': 'hotdogs are a staple of the american diet. they are a great source of protein and can'}]]
```

Another good use case for batch inference is for streaming data in [Pipeline](/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.Pipeline).


```
from transformers import pipeline, infer_device
from transformers.pipelines.pt_utils import KeyDataset
import datasets

device = infer_device()

# KeyDataset is a utility that returns the item in the dict returned by the dataset
dataset = datasets.load_dataset("imdb", name="plain_text", split="unsupervised")
pipeline = pipeline(task="text-classification", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english", device=device)
for out in pipeline(KeyDataset(dataset, "text"), batch_size=8, truncation="only_first"):
    print(out)
```

Keep the following general rules of thumb in mind for determining whether batch inference can help improve performance.

1. The only way to know for sure is to measure performance on your model, data, and hardware.
2. Don’t batch inference if you’re constrained by latency (a live inference product for example).
3. Don’t batch inference if you’re using a CPU.
4. Don’t batch inference if you don’t know the `sequence_length` of your data. Measure performance, iteratively add to `sequence_length`, and include out-of-memory (OOM) checks to recover from failures.
5. Do batch inference if your `sequence_length` is regular, and keep pushing it until you reach an OOM error. The larger the GPU, the more helpful batch inference is.
6. Do make sure you can handle OOM errors if you decide to do batch inference.

### Task-specific parameters

[Pipeline](/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.Pipeline) accepts any parameters that are supported by each individual task pipeline. Make sure to check out each individual task pipeline to see what type of parameters are available. If you can’t find a parameter that is useful for your use case, please feel free to open a GitHub [issue](https://github.com/huggingface/transformers/issues/new?assignees=&labels=feature&template=feature-request.yml) to request it!

The examples below demonstrate some of the task-specific parameters available.

automatic speech recognition

text generation

Pass the `return_timestamps="word"` parameter to [Pipeline](/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.Pipeline) to return when each word was spoken.


```
from transformers import pipeline

pipeline = pipeline(task="automatic-speech-recognition", model="openai/whisper-large-v3")
pipeline(audio="https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac", return_timestamp="word")
{'text': ' I have a dream that one day this nation will rise up and live out the true meaning of its creed.',
 'chunks': [{'text': ' I', 'timestamp': (0.0, 1.1)},
  {'text': ' have', 'timestamp': (1.1, 1.44)},
  {'text': ' a', 'timestamp': (1.44, 1.62)},
  {'text': ' dream', 'timestamp': (1.62, 1.92)},
  {'text': ' that', 'timestamp': (1.92, 3.7)},
  {'text': ' one', 'timestamp': (3.7, 3.88)},
  {'text': ' day', 'timestamp': (3.88, 4.24)},
  {'text': ' this', 'timestamp': (4.24, 5.82)},
  {'text': ' nation', 'timestamp': (5.82, 6.78)},
  {'text': ' will', 'timestamp': (6.78, 7.36)},
  {'text': ' rise', 'timestamp': (7.36, 7.88)},
  {'text': ' up', 'timestamp': (7.88, 8.46)},
  {'text': ' and', 'timestamp': (8.46, 9.2)},
  {'text': ' live', 'timestamp': (9.2, 10.34)},
  {'text': ' out', 'timestamp': (10.34, 10.58)},
  {'text': ' the', 'timestamp': (10.58, 10.8)},
  {'text': ' true', 'timestamp': (10.8, 11.04)},
  {'text': ' meaning', 'timestamp': (11.04, 11.4)},
  {'text': ' of', 'timestamp': (11.4, 11.64)},
  {'text': ' its', 'timestamp': (11.64, 11.8)},
  {'text': ' creed.', 'timestamp': (11.8, 12.3)}]}
```

## Chunk batching

There are some instances where you need to process data in chunks.

* for some data types, a single input (for example, a really long audio file) may need to be chunked into multiple parts before it can be processed
* for some tasks, like zero-shot classification or question answering, a single input may need multiple forward passes which can cause issues with the `batch_size` parameter

The [ChunkPipeline](https://github.com/huggingface/transformers/blob/99e0ab6ed888136ea4877c6d8ab03690a1478363/src/transformers/pipelines/base.py#L1387) class is designed to handle these use cases. Both pipeline classes are used in the same way, but since [ChunkPipeline](https://github.com/huggingface/transformers/blob/99e0ab6ed888136ea4877c6d8ab03690a1478363/src/transformers/pipelines/base.py#L1387) can automatically handle batching, you don’t need to worry about the number of forward passes your inputs trigger. Instead, you can optimize `batch_size` independently of the inputs.

The example below shows how it differs from [Pipeline](/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.Pipeline).


```
# ChunkPipeline
all_model_outputs = []
for preprocessed in pipeline.preprocess(inputs):
    model_outputs = pipeline.model_forward(preprocessed)
    all_model_outputs.append(model_outputs)
outputs =pipeline.postprocess(all_model_outputs)

# Pipeline
preprocessed = pipeline.preprocess(inputs)
model_outputs = pipeline.forward(preprocessed)
outputs = pipeline.postprocess(model_outputs)
```

## Large datasets

For inference with large datasets, you can iterate directly over the dataset itself. This avoids immediately allocating memory for the entire dataset, and you don’t need to worry about creating batches yourself. Try [Batch inference](#batch-inference) with the `batch_size` parameter to see if it improves performance.


```
from transformers.pipelines.pt_utils import KeyDataset
from transformers import pipeline, infer_device
from datasets import load_dataset

device = infer_device()

dataset = datasets.load_dataset("imdb", name="plain_text", split="unsupervised")
pipeline = pipeline(task="text-classification", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english", device=device)
for out in pipeline(KeyDataset(dataset, "text"), batch_size=8, truncation="only_first"):
    print(out)
```

Other ways to run inference on large datasets with [Pipeline](/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.Pipeline) include using an iterator or generator.


```
def data():
    for i in range(1000):
        yield f"My example {i}"

pipeline = pipeline(model="openai-community/gpt2", device=0)
generated_characters = 0
for out in pipeline(data()):
    generated_characters += len(out[0]["generated_text"])
```

## Large models

[Accelerate](https://hf.co/docs/accelerate/index) enables a couple of optimizations for running large models with [Pipeline](/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.Pipeline). Make sure Accelerate is installed first.


```
!pip install -U accelerate
```

The `device_map="auto"` setting is useful for automatically distributing the model across the fastest devices (GPUs) first before dispatching to other slower devices if available (CPU, hard drive).

[Pipeline](/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.Pipeline) supports half-precision weights (torch.float16), which can be significantly faster and save memory. Performance loss is negligible for most models, especially for larger ones. If your hardware supports it, you can enable torch.bfloat16 instead for more range.

Inputs are internally converted to torch.float16 and it only works for models with a PyTorch backend.

Lastly, [Pipeline](/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.Pipeline) also accepts quantized models to reduce memory usage even further. Make sure you have the [bitsandbytes](https://hf.co/docs/bitsandbytes/installation) library installed first, and then add `load_in_8bit=True` to `model_kwargs` in the pipeline.


```
import torch
from transformers import pipeline, BitsAndBytesConfig

pipeline = pipeline(model="google/gemma-7b", dtype=torch.bfloat16, device_map="auto", model_kwargs={"quantization_config": BitsAndBytesConfig(load_in_8bit=True)})
pipeline("the secret to baking a good cake is ")
[{'generated_text': 'the secret to baking a good cake is 1. the right ingredients 2. the right'}]
```

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/pipeline_tutorial.md)
