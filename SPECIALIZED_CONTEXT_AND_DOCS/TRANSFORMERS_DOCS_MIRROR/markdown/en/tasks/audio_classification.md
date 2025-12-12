# Audio classification

![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)

![Open In Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)

Audio classification - just like with text - assigns a class label as output from the input data. The only difference is instead of text inputs, you have raw audio waveforms. Some practical applications of audio classification include identifying speaker intent, language classification, and even animal species by their sounds.

This guide will show you how to:

1. Fine-tune [Wav2Vec2](https://huggingface.co/facebook/wav2vec2-base) on the [MInDS-14](https://huggingface.co/datasets/PolyAI/minds14) dataset to classify speaker intent.
2. Use your fine-tuned model for inference.

To see all architectures and checkpoints compatible with this task, we recommend checking the [task-page](https://huggingface.co/tasks/audio-classification)

Before you begin, make sure you have all the necessary libraries installed:


```
pip install transformers datasets evaluate
```

We encourage you to login to your Hugging Face account so you can upload and share your model with the community. When prompted, enter your token to login:


```
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

## Load MInDS-14 dataset

Start by loading the MInDS-14 dataset from the 🤗 Datasets library:


```
>>> from datasets import load_dataset, Audio

>>> minds = load_dataset("PolyAI/minds14", name="en-US", split="train")
```

Split the dataset’s `train` split into a smaller train and test set with the [train\_test\_split](https://huggingface.co/docs/datasets/v4.1.0/en/package_reference/main_classes#datasets.Dataset.train_test_split) method. This will give you a chance to experiment and make sure everything works before spending more time on the full dataset.


```
>>> minds = minds.train_test_split(test_size=0.2)
```

Then take a look at the dataset:


```
>>> minds
DatasetDict({
    train: Dataset({
        features: ['path', 'audio', 'transcription', 'english_transcription', 'intent_class', 'lang_id'],
        num_rows: 450
    })
    test: Dataset({
        features: ['path', 'audio', 'transcription', 'english_transcription', 'intent_class', 'lang_id'],
        num_rows: 113
    })
})
```

While the dataset contains a lot of useful information, like `lang_id` and `english_transcription`, you will focus on the `audio` and `intent_class` in this guide. Remove the other columns with the [remove\_columns](https://huggingface.co/docs/datasets/v4.1.0/en/package_reference/main_classes#datasets.Dataset.remove_columns) method:


```
>>> minds = minds.remove_columns(["path", "transcription", "english_transcription", "lang_id"])
```

Here’s an example:


```
>>> minds["train"][0]
{'audio': {'array': array([ 0.        ,  0.        ,  0.        , ..., -0.00048828,
         -0.00024414, -0.00024414], dtype=float32),
  'path': '/root/.cache/huggingface/datasets/downloads/extracted/f14948e0e84be638dd7943ac36518a4cf3324e8b7aa331c5ab11541518e9368c/en-US~APP_ERROR/602b9a5fbb1e6d0fbce91f52.wav',
  'sampling_rate': 8000},
 'intent_class': 2}
```

There are two fields:

* `audio`: a 1-dimensional `array` of the speech signal that must be called to load and resample the audio file.
* `intent_class`: represents the class id of the speaker’s intent.

To make it easier for the model to get the label name from the label id, create a dictionary that maps the label name to an integer and vice versa:


```
>>> labels = minds["train"].features["intent_class"].names
>>> label2id, id2label = dict(), dict()
>>> for i, label in enumerate(labels):
...     label2id[label] = str(i)
...     id2label[str(i)] = label
```

Now you can convert the label id to a label name:


```
>>> id2label[str(2)]
'app_error'
```

## Preprocess

The next step is to load a Wav2Vec2 feature extractor to process the audio signal:


```
>>> from transformers import AutoFeatureExtractor

>>> feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
```

The MInDS-14 dataset has a sampling rate of 8kHz (you can find this information in its [dataset card](https://huggingface.co/datasets/PolyAI/minds14)), which means you’ll need to resample the dataset to 16kHz to use the pretrained Wav2Vec2 model:


```
>>> minds = minds.cast_column("audio", Audio(sampling_rate=16_000))
>>> minds["train"][0]
{'audio': {'array': array([ 2.2098757e-05,  4.6582241e-05, -2.2803260e-05, ...,
         -2.8419291e-04, -2.3305941e-04, -1.1425107e-04], dtype=float32),
  'path': '/root/.cache/huggingface/datasets/downloads/extracted/f14948e0e84be638dd7943ac36518a4cf3324e8b7aa331c5ab11541518e9368c/en-US~APP_ERROR/602b9a5fbb1e6d0fbce91f52.wav',
  'sampling_rate': 16000},
 'intent_class': 2}
```

Now create a preprocessing function that:

1. Calls the `audio` column to load, and if necessary, resample the audio file.
2. Checks if the sampling rate of the audio file matches the sampling rate of the audio data a model was pretrained with. You can find this information in the Wav2Vec2 [model card](https://huggingface.co/facebook/wav2vec2-base).
3. Set a maximum input length to batch longer inputs without truncating them.


```
>>> def preprocess_function(examples):
...     audio_arrays = [x["array"] for x in examples["audio"]]
...     inputs = feature_extractor(
...         audio_arrays, sampling_rate=feature_extractor.sampling_rate, max_length=16000, truncation=True
...     )
...     return inputs
```

To apply the preprocessing function over the entire dataset, use 🤗 Datasets [map](https://huggingface.co/docs/datasets/v4.1.0/en/package_reference/main_classes#datasets.Dataset.map) function. You can speed up `map` by setting `batched=True` to process multiple elements of the dataset at once. Remove unnecessary columns and rename `intent_class` to `label`, as required by the model:


```
>>> encoded_minds = minds.map(preprocess_function, remove_columns="audio", batched=True)
>>> encoded_minds = encoded_minds.rename_column("intent_class", "label")
```

## Evaluate

Including a metric during training is often helpful for evaluating your model’s performance. You can quickly load an evaluation method with the 🤗 [Evaluate](https://huggingface.co/docs/evaluate/index) library. For this task, load the [accuracy](https://huggingface.co/spaces/evaluate-metric/accuracy) metric (see the 🤗 Evaluate [quick tour](https://huggingface.co/docs/evaluate/a_quick_tour) to learn more about how to load and compute a metric):


```
>>> import evaluate

>>> accuracy = evaluate.load("accuracy")
```

Then create a function that passes your predictions and labels to [compute](https://huggingface.co/docs/evaluate/v0.4.5/en/package_reference/main_classes#evaluate.EvaluationModule.compute) to calculate the accuracy:


```
>>> import numpy as np


>>> def compute_metrics(eval_pred):
...     predictions = np.argmax(eval_pred.predictions, axis=1)
...     return accuracy.compute(predictions=predictions, references=eval_pred.label_ids)
```

Your `compute_metrics` function is ready to go now, and you’ll return to it when you setup your training.

## Train

Pytorch

Hide Pytorch content

If you aren’t familiar with finetuning a model with the [Trainer](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Trainer), take a look at the basic tutorial [here](../training#train-with-pytorch-trainer)!

You’re ready to start training your model now! Load Wav2Vec2 with [AutoModelForAudioClassification](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModelForAudioClassification) along with the number of expected labels, and the label mappings:


```
>>> from transformers import AutoModelForAudioClassification, TrainingArguments, Trainer

>>> num_labels = len(id2label)
>>> model = AutoModelForAudioClassification.from_pretrained(
...     "facebook/wav2vec2-base", num_labels=num_labels, label2id=label2id, id2label=id2label
... )
```

At this point, only three steps remain:

1. Define your training hyperparameters in [TrainingArguments](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.TrainingArguments). The only required parameter is `output_dir`, which specifies where to save your model. You’ll push this model to the Hub by setting `push_to_hub=True` (you need to be signed in to Hugging Face to upload your model). At the end of each epoch, the [Trainer](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Trainer) will evaluate the accuracy and save the training checkpoint.
2. Pass the training arguments to [Trainer](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Trainer) along with the model, dataset, tokenizer, data collator, and `compute_metrics` function.
3. Call [train()](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Trainer.train) to fine-tune your model.


```
>>> training_args = TrainingArguments(
...     output_dir="my_awesome_mind_model",
...     eval_strategy="epoch",
...     save_strategy="epoch",
...     learning_rate=3e-5,
...     per_device_train_batch_size=32,
...     gradient_accumulation_steps=4,
...     per_device_eval_batch_size=32,
...     num_train_epochs=10,
...     warmup_ratio=0.1,
...     logging_steps=10,
...     load_best_model_at_end=True,
...     metric_for_best_model="accuracy",
...     push_to_hub=True,
... )

>>> trainer = Trainer(
...     model=model,
...     args=training_args,
...     train_dataset=encoded_minds["train"],
...     eval_dataset=encoded_minds["test"],
...     processing_class=feature_extractor,
...     compute_metrics=compute_metrics,
... )

>>> trainer.train()
```

Once training is completed, share your model to the Hub with the [push\_to\_hub()](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Trainer.push_to_hub) method so everyone can use your model:


```
>>> trainer.push_to_hub()
```

For a more in-depth example of how to fine-tune a model for audio classification, take a look at the corresponding [PyTorch notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/audio_classification.ipynb).

## Inference

Great, now that you’ve fine-tuned a model, you can use it for inference!

Load an audio file for inference. Remember to resample the sampling rate of the audio file to match the model’s sampling rate, if necessary.


```
>>> from datasets import load_dataset, Audio

>>> dataset = load_dataset("PolyAI/minds14", name="en-US", split="train")
>>> dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
>>> sampling_rate = dataset.features["audio"].sampling_rate
>>> audio_file = dataset[0]["audio"]["path"]
```

The simplest way to try out your fine-tuned model for inference is to use it in a [pipeline()](/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.pipeline). Instantiate a `pipeline` for audio classification with your model, and pass your audio file to it:


```
>>> from transformers import pipeline

>>> classifier = pipeline("audio-classification", model="stevhliu/my_awesome_minds_model")
>>> classifier(audio_file)
[
    {'score': 0.09766869246959686, 'label': 'cash_deposit'},
    {'score': 0.07998877018690109, 'label': 'app_error'},
    {'score': 0.0781070664525032, 'label': 'joint_account'},
    {'score': 0.07667109370231628, 'label': 'pay_bill'},
    {'score': 0.0755252093076706, 'label': 'balance'}
]
```

You can also manually replicate the results of the `pipeline` if you’d like:

Pytorch

Hide Pytorch content

Load a feature extractor to preprocess the audio file and return the `input` as PyTorch tensors:


```
>>> from transformers import AutoFeatureExtractor

>>> feature_extractor = AutoFeatureExtractor.from_pretrained("stevhliu/my_awesome_minds_model")
>>> inputs = feature_extractor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")
```

Pass your inputs to the model and return the logits:


```
>>> from transformers import AutoModelForAudioClassification

>>> model = AutoModelForAudioClassification.from_pretrained("stevhliu/my_awesome_minds_model")
>>> with torch.no_grad():
...     logits = model(**inputs).logits
```

Get the class with the highest probability, and use the model’s `id2label` mapping to convert it to a label:


```
>>> import torch

>>> predicted_class_ids = torch.argmax(logits).item()
>>> predicted_label = model.config.id2label[predicted_class_ids]
>>> predicted_label
'cash_deposit'
```

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/tasks/audio_classification.md)
