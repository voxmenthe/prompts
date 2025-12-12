# Question answering

![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)

![Open In Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)

Question answering tasks return an answer given a question. If you’ve ever asked a virtual assistant like Alexa, Siri or Google what the weather is, then you’ve used a question answering model before. There are two common types of question answering tasks:

* Extractive: extract the answer from the given context.
* Abstractive: generate an answer from the context that correctly answers the question.

This guide will show you how to:

1. Finetune [DistilBERT](https://huggingface.co/distilbert/distilbert-base-uncased) on the [SQuAD](https://huggingface.co/datasets/squad) dataset for extractive question answering.
2. Use your finetuned model for inference.

To see all architectures and checkpoints compatible with this task, we recommend checking the [task-page](https://huggingface.co/tasks/question-answering)

Before you begin, make sure you have all the necessary libraries installed:


```
pip install transformers datasets evaluate
```

We encourage you to login to your Hugging Face account so you can upload and share your model with the community. When prompted, enter your token to login:


```
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

## Load SQuAD dataset

Start by loading a smaller subset of the SQuAD dataset from the 🤗 Datasets library. This’ll give you a chance to experiment and make sure everything works before spending more time training on the full dataset.


```
>>> from datasets import load_dataset

>>> squad = load_dataset("squad", split="train[:5000]")
```

Split the dataset’s `train` split into a train and test set with the [train\_test\_split](https://huggingface.co/docs/datasets/v4.1.0/en/package_reference/main_classes#datasets.Dataset.train_test_split) method:


```
>>> squad = squad.train_test_split(test_size=0.2)
```

Then take a look at an example:


```
>>> squad["train"][0]
{'answers': {'answer_start': [515], 'text': ['Saint Bernadette Soubirous']},
 'context': 'Architecturally, the school has a Catholic character. Atop the Main Building\'s gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend "Venite Ad Me Omnes". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.',
 'id': '5733be284776f41900661182',
 'question': 'To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?',
 'title': 'University_of_Notre_Dame'
}
```

There are several important fields here:

* `answers`: the starting location of the answer token and the answer text.
* `context`: background information from which the model needs to extract the answer.
* `question`: the question a model should answer.

## Preprocess

The next step is to load a DistilBERT tokenizer to process the `question` and `context` fields:


```
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
```

There are a few preprocessing steps particular to question answering tasks you should be aware of:

1. Some examples in a dataset may have a very long `context` that exceeds the maximum input length of the model. To deal with longer sequences, truncate only the `context` by setting `truncation="only_second"`.
2. Next, map the start and end positions of the answer to the original `context` by setting
   `return_offset_mapping=True`.
3. With the mapping in hand, now you can find the start and end tokens of the answer. Use the `sequence_ids` method to
   find which part of the offset corresponds to the `question` and which corresponds to the `context`.

Here is how you can create a function to truncate and map the start and end tokens of the `answer` to the `context`:


```
>>> def preprocess_function(examples):
...     questions = [q.strip() for q in examples["question"]]
...     inputs = tokenizer(
...         questions,
...         examples["context"],
...         max_length=384,
...         truncation="only_second",
...         return_offsets_mapping=True,
...         padding="max_length",
...     )

...     offset_mapping = inputs.pop("offset_mapping")
...     answers = examples["answers"]
...     start_positions = []
...     end_positions = []

...     for i, offset in enumerate(offset_mapping):
...         answer = answers[i]
...         start_char = answer["answer_start"][0]
...         end_char = answer["answer_start"][0] + len(answer["text"][0])
...         sequence_ids = inputs.sequence_ids(i)

...         # Find the start and end of the context
...         idx = 0
...         while sequence_ids[idx] != 1:
...             idx += 1
...         context_start = idx
...         while sequence_ids[idx] == 1:
...             idx += 1
...         context_end = idx - 1

...         # If the answer is not fully inside the context, label it (0, 0)
...         if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
...             start_positions.append(0)
...             end_positions.append(0)
...         else:
...             # Otherwise it's the start and end token positions
...             idx = context_start
...             while idx <= context_end and offset[idx][0] <= start_char:
...                 idx += 1
...             start_positions.append(idx - 1)

...             idx = context_end
...             while idx >= context_start and offset[idx][1] >= end_char:
...                 idx -= 1
...             end_positions.append(idx + 1)

...     inputs["start_positions"] = start_positions
...     inputs["end_positions"] = end_positions
...     return inputs
```

To apply the preprocessing function over the entire dataset, use 🤗 Datasets [map](https://huggingface.co/docs/datasets/v4.1.0/en/package_reference/main_classes#datasets.Dataset.map) function. You can speed up the `map` function by setting `batched=True` to process multiple elements of the dataset at once. Remove any columns you don’t need:


```
>>> tokenized_squad = squad.map(preprocess_function, batched=True, remove_columns=squad["train"].column_names)
```

Now create a batch of examples using [DefaultDataCollator](/docs/transformers/v4.56.2/en/main_classes/data_collator#transformers.DefaultDataCollator). Unlike other data collators in 🤗 Transformers, the [DefaultDataCollator](/docs/transformers/v4.56.2/en/main_classes/data_collator#transformers.DefaultDataCollator) does not apply any additional preprocessing such as padding.


```
>>> from transformers import DefaultDataCollator

>>> data_collator = DefaultDataCollator()
```

## Train

If you aren’t familiar with finetuning a model with the [Trainer](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Trainer), take a look at the basic tutorial [here](../training#train-with-pytorch-trainer)!

You’re ready to start training your model now! Load DistilBERT with [AutoModelForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModelForQuestionAnswering):


```
>>> from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer

>>> model = AutoModelForQuestionAnswering.from_pretrained("distilbert/distilbert-base-uncased")
```

At this point, only three steps remain:

1. Define your training hyperparameters in [TrainingArguments](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.TrainingArguments). The only required parameter is `output_dir` which specifies where to save your model. You’ll push this model to the Hub by setting `push_to_hub=True` (you need to be signed in to Hugging Face to upload your model).
2. Pass the training arguments to [Trainer](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Trainer) along with the model, dataset, tokenizer, and data collator.
3. Call [train()](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Trainer.train) to finetune your model.


```
>>> training_args = TrainingArguments(
...     output_dir="my_awesome_qa_model",
...     eval_strategy="epoch",
...     learning_rate=2e-5,
...     per_device_train_batch_size=16,
...     per_device_eval_batch_size=16,
...     num_train_epochs=3,
...     weight_decay=0.01,
...     push_to_hub=True,
... )

>>> trainer = Trainer(
...     model=model,
...     args=training_args,
...     train_dataset=tokenized_squad["train"],
...     eval_dataset=tokenized_squad["test"],
...     processing_class=tokenizer,
...     data_collator=data_collator,
... )

>>> trainer.train()
```

Once training is completed, share your model to the Hub with the [push\_to\_hub()](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Trainer.push_to_hub) method so everyone can use your model:


```
>>> trainer.push_to_hub()
```

For a more in-depth example of how to finetune a model for question answering, take a look at the corresponding
[PyTorch notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/question_answering.ipynb).

## Evaluate

Evaluation for question answering requires a significant amount of postprocessing. To avoid taking up too much of your time, this guide skips the evaluation step. The [Trainer](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Trainer) still calculates the evaluation loss during training so you’re not completely in the dark about your model’s performance.

If you have more time and you’re interested in how to evaluate your model for question answering, take a look at the [Question answering](https://huggingface.co/course/chapter7/7?fw=pt#post-processing) chapter from the 🤗 Hugging Face Course!

## Inference

Great, now that you’ve finetuned a model, you can use it for inference!

Come up with a question and some context you’d like the model to predict:


```
>>> question = "How many programming languages does BLOOM support?"
>>> context = "BLOOM has 176 billion parameters and can generate text in 46 languages natural languages and 13 programming languages."
```

The simplest way to try out your finetuned model for inference is to use it in a [pipeline()](/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.pipeline). Instantiate a `pipeline` for question answering with your model, and pass your text to it:


```
>>> from transformers import pipeline

>>> question_answerer = pipeline("question-answering", model="my_awesome_qa_model")
>>> question_answerer(question=question, context=context)
{'score': 0.2058267742395401,
 'start': 10,
 'end': 95,
 'answer': '176 billion parameters and can generate text in 46 languages natural languages and 13'}
```

You can also manually replicate the results of the `pipeline` if you’d like:

Tokenize the text and return PyTorch tensors:


```
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("my_awesome_qa_model")
>>> inputs = tokenizer(question, context, return_tensors="pt")
```

Pass your inputs to the model and return the `logits`:


```
>>> import torch
>>> from transformers import AutoModelForQuestionAnswering

>>> model = AutoModelForQuestionAnswering.from_pretrained("my_awesome_qa_model")
>>> with torch.no_grad():
...     outputs = model(**inputs)
```

Get the highest probability from the model output for the start and end positions:


```
>>> answer_start_index = outputs.start_logits.argmax()
>>> answer_end_index = outputs.end_logits.argmax()
```

Decode the predicted tokens to get the answer:


```
>>> predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
>>> tokenizer.decode(predict_answer_tokens)
'176 billion parameters and can generate text in 46 languages natural languages and 13'
```

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/tasks/question_answering.md)
