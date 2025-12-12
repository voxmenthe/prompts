# Quickstart

![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)

![Open In Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)

Transformers is designed to be fast and easy to use so that everyone can start learning or building with transformer models.

The number of user-facing abstractions is limited to only three classes for instantiating a model, and two APIs for inference or training. This quickstart introduces you to Transformers’ key features and shows you how to:

* load a pretrained model
* run inference with [Pipeline](/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.Pipeline)
* fine-tune a model with [Trainer](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Trainer)

## Set up

To start, we recommend creating a Hugging Face [account](https://hf.co/join). An account lets you host and access version controlled models, datasets, and [Spaces](https://hf.co/spaces) on the Hugging Face [Hub](https://hf.co/docs/hub/index), a collaborative platform for discovery and building.

Create a [User Access Token](https://hf.co/docs/hub/security-tokens#user-access-tokens) and log in to your account.

notebook

CLI

Paste your User Access Token into `notebook_login` when prompted to log in.


```
from huggingface_hub import notebook_login

notebook_login()
```

Install Pytorch.


```
!pip install torch
```

Then install an up-to-date version of Transformers and some additional libraries from the Hugging Face ecosystem for accessing datasets and vision models, evaluating training, and optimizing training for large models.


```
!pip install -U transformers datasets evaluate accelerate timm
```

## Pretrained models

Each pretrained model inherits from three base classes.

| **Class** | **Description** |
| --- | --- |
| [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) | A file that specifies a models attributes such as the number of attention heads or vocabulary size. |
| [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel) | A model (or architecture) defined by the model attributes from the configuration file. A pretrained model only returns the raw hidden states. For a specific task, use the appropriate model head to convert the raw hidden states into a meaningful result (for example, [LlamaModel](/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaModel) versus [LlamaForCausalLM](/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaForCausalLM)). |
| Preprocessor | A class for converting raw inputs (text, images, audio, multimodal) into numerical inputs to the model. For example, [PreTrainedTokenizer](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer) converts text into tensors and [ImageProcessingMixin](/docs/transformers/v4.56.2/en/main_classes/image_processor#transformers.ImageProcessingMixin) converts pixels into tensors. |

We recommend using the [AutoClass](./model_doc/auto) API to load models and preprocessors because it automatically infers the appropriate architecture for each task and machine learning framework based on the name or path to the pretrained weights and configuration file.

Use [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) to load the weights and configuration file from the Hub into the model and preprocessor class.

When you load a model, configure the following parameters to ensure the model is optimally loaded.

* `device_map="auto"` automatically allocates the model weights to your fastest device first.
* `dtype="auto"` directly initializes the model weights in the data type they’re stored in, which can help avoid loading the weights twice (PyTorch loads weights in `torch.float32` by default).


```
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
```

Tokenize the text and return PyTorch tensors with the tokenizer. Move the model to an accelerator if it’s available to accelerate inference.


```
model_inputs = tokenizer(["The secret to baking a good cake is "], return_tensors="pt").to(model.device)
```

The model is now ready for inference or training.

For inference, pass the tokenized inputs to [generate()](/docs/transformers/v4.56.2/en/main_classes/text_generation#transformers.GenerationMixin.generate) to generate text. Decode the token ids back into text with [batch\_decode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.batch_decode).


```
generated_ids = model.generate(**model_inputs, max_length=30)
tokenizer.batch_decode(generated_ids)[0]
'<s> The secret to baking a good cake is 100% in the preparation. There are so many recipes out there,'
```

Skip ahead to the [Trainer](#trainer-api) section to learn how to fine-tune a model.

## Pipeline

The [Pipeline](/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.Pipeline) class is the most convenient way to inference with a pretrained model. It supports many tasks such as text generation, image segmentation, automatic speech recognition, document question answering, and more.

Refer to the [Pipeline](./main_classes/pipelines) API reference for a complete list of available tasks.

Create a [Pipeline](/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.Pipeline) object and select a task. By default, [Pipeline](/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.Pipeline) downloads and caches a default pretrained model for a given task. Pass the model name to the `model` parameter to choose a specific model.

text generation

image segmentation

automatic speech recognition

Use [~infer\_device()](/docs/transformers/v4.56.2/en/internal/file_utils#transformers.infer_device) to automatically detect an available accelerator for inference.


```
from transformers import pipeline, infer_device

device = infer_device()

pipeline = pipeline("text-generation", model="meta-llama/Llama-2-7b-hf", device=device)
```

Prompt [Pipeline](/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.Pipeline) with some initial text to generate more text.


```
pipeline("The secret to baking a good cake is ", max_length=50)
[{'generated_text': 'The secret to baking a good cake is 100% in the batter. The secret to a great cake is the icing.\nThis is why we’ve created the best buttercream frosting reci'}]
```

## Trainer

[Trainer](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Trainer) is a complete training and evaluation loop for PyTorch models. It abstracts away a lot of the boilerplate usually involved in manually writing a training loop, so you can start training faster and focus on training design choices. You only need a model, dataset, a preprocessor, and a data collator to build batches of data from the dataset.

Use the [TrainingArguments](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.TrainingArguments) class to customize the training process. It provides many options for training, evaluation, and more. Experiment with training hyperparameters and features like batch size, learning rate, mixed precision, torch.compile, and more to meet your training needs. You could also use the default training parameters to quickly produce a baseline.

Load a model, tokenizer, and dataset for training.


```
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset

model = AutoModelForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
dataset = load_dataset("rotten_tomatoes")
```

Create a function to tokenize the text and convert it into PyTorch tensors. Apply this function to the whole dataset with the [map](https://huggingface.co/docs/datasets/v4.1.0/en/package_reference/main_classes#datasets.Dataset.map) method.


```
def tokenize_dataset(dataset):
    return tokenizer(dataset["text"])
dataset = dataset.map(tokenize_dataset, batched=True)
```

Load a data collator to create batches of data and pass the tokenizer to it.


```
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
```

Next, set up [TrainingArguments](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.TrainingArguments) with the training features and hyperparameters.


```
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="distilbert-rotten-tomatoes",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    push_to_hub=True,
)
```

Finally, pass all these separate components to [Trainer](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Trainer) and call [train()](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Trainer.train) to start.


```
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()
```

Share your model and tokenizer to the Hub with [push\_to\_hub()](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Trainer.push_to_hub).


```
trainer.push_to_hub()
```

Congratulations, you just trained your first model with Transformers!

## Next steps

Now that you have a better understanding of Transformers and what it offers, it’s time to keep exploring and learning what interests you the most.

* **Base classes**: Learn more about the configuration, model and processor classes. This will help you understand how to create and customize models, preprocess different types of inputs (audio, images, multimodal), and how to share your model.
* **Inference**: Explore the [Pipeline](/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.Pipeline) further, inference and chatting with LLMs, agents, and how to optimize inference with your machine learning framework and hardware.
* **Training**: Study the [Trainer](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Trainer) in more detail, as well as distributed training and optimizing training on specific hardware.
* **Quantization**: Reduce memory and storage requirements with quantization and speed up inference by representing weights with fewer bits.
* **Resources**: Looking for end-to-end recipes for how to train and inference with a model for a specific task? Check out the task recipes!

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/quicktour.md)
