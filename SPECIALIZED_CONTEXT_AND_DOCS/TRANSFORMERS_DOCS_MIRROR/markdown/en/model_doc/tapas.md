*This model was released on 2020-04-05 and added to Hugging Face Transformers on 2020-12-15.*

# TAPAS

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white)

## Overview

The TAPAS model was proposed in [TAPAS: Weakly Supervised Table Parsing via Pre-training](https://huggingface.co/papers/2004.02349)
by Jonathan Herzig, Paweł Krzysztof Nowak, Thomas Müller, Francesco Piccinno and Julian Martin Eisenschlos. It’s a BERT-based model specifically
designed (and pre-trained) for answering questions about tabular data. Compared to BERT, TAPAS uses relative position embeddings and has 7
token types that encode tabular structure. TAPAS is pre-trained on the masked language modeling (MLM) objective on a large dataset comprising
millions of tables from English Wikipedia and corresponding texts.

For question answering, TAPAS has 2 heads on top: a cell selection head and an aggregation head, for (optionally) performing aggregations (such as counting or summing) among selected cells. TAPAS has been fine-tuned on several datasets:

* [SQA](https://www.microsoft.com/en-us/download/details.aspx?id=54253) (Sequential Question Answering by Microsoft)
* [WTQ](https://github.com/ppasupat/WikiTableQuestions) (Wiki Table Questions by Stanford University)
* [WikiSQL](https://github.com/salesforce/WikiSQL) (by Salesforce).

It achieves state-of-the-art on both SQA and WTQ, while having comparable performance to SOTA on WikiSQL, with a much simpler architecture.

The abstract from the paper is the following:

*Answering natural language questions over tables is usually seen as a semantic parsing task. To alleviate the collection cost of full logical forms, one popular approach focuses on weak supervision consisting of denotations instead of logical forms. However, training semantic parsers from weak supervision poses difficulties, and in addition, the generated logical forms are only used as an intermediate step prior to retrieving the denotation. In this paper, we present TAPAS, an approach to question answering over tables without generating logical forms. TAPAS trains from weak supervision, and predicts the denotation by selecting table cells and optionally applying a corresponding aggregation operator to such selection. TAPAS extends BERT’s architecture to encode tables as input, initializes from an effective joint pre-training of text segments and tables crawled from Wikipedia, and is trained end-to-end. We experiment with three different semantic parsing datasets, and find that TAPAS outperforms or rivals semantic parsing models by improving state-of-the-art accuracy on SQA from 55.1 to 67.2 and performing on par with the state-of-the-art on WIKISQL and WIKITQ, but with a simpler model architecture. We additionally find that transfer learning, which is trivial in our setting, from WIKISQL to WIKITQ, yields 48.7 accuracy, 4.2 points above the state-of-the-art.*

In addition, the authors have further pre-trained TAPAS to recognize **table entailment**, by creating a balanced dataset of millions of automatically created training examples which are learned in an intermediate step prior to fine-tuning. The authors of TAPAS call this further pre-training intermediate pre-training (since TAPAS is first pre-trained on MLM, and then on another dataset). They found that intermediate pre-training further improves performance on SQA, achieving a new state-of-the-art as well as state-of-the-art on [TabFact](https://github.com/wenhuchen/Table-Fact-Checking), a large-scale dataset with 16k Wikipedia tables for table entailment (a binary classification task). For more details, see their follow-up paper: [Understanding tables with intermediate pre-training](https://www.aclweb.org/anthology/2020.findings-emnlp.27/) by Julian Martin Eisenschlos, Syrine Krichene and Thomas Müller.

![drawing](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/tapas_architecture.png) TAPAS architecture. Taken from the [original blog post](https://ai.googleblog.com/2020/04/using-neural-networks-to-find-answers.html).

This model was contributed by [nielsr](https://huggingface.co/nielsr). The original code can be found [here](https://github.com/google-research/tapas).

## Usage tips

* TAPAS is a model that uses relative position embeddings by default (restarting the position embeddings at every cell of the table). Note that this is something that was added after the publication of the original TAPAS paper. According to the authors, this usually results in a slightly better performance, and allows you to encode longer sequences without running out of embeddings. This is reflected in the `reset_position_index_per_cell` parameter of [TapasConfig](/docs/transformers/v4.56.2/en/model_doc/tapas#transformers.TapasConfig), which is set to `True` by default. The default versions of the models available on the [hub](https://huggingface.co/models?search=tapas) all use relative position embeddings. You can still use the ones with absolute position embeddings by passing in an additional argument `revision="no_reset"` when calling the `from_pretrained()` method. Note that it’s usually advised to pad the inputs on the right rather than the left.
* TAPAS is based on BERT, so `TAPAS-base` for example corresponds to a `BERT-base` architecture. Of course, `TAPAS-large` will result in the best performance (the results reported in the paper are from `TAPAS-large`). Results of the various sized models are shown on the [original GitHub repository](https://github.com/google-research/tapas).
* TAPAS has checkpoints fine-tuned on SQA, which are capable of answering questions related to a table in a conversational set-up. This means that you can ask follow-up questions such as “what is his age?” related to the previous question. Note that the forward pass of TAPAS is a bit different in case of a conversational set-up: in that case, you have to feed every table-question pair one by one to the model, such that the `prev_labels` token type ids can be overwritten by the predicted `labels` of the model to the previous question. See “Usage” section for more info.
* TAPAS is similar to BERT and therefore relies on the masked language modeling (MLM) objective. It is therefore efficient at predicting masked tokens and at NLU in general, but is not optimal for text generation. Models trained with a causal language modeling (CLM) objective are better in that regard. Note that TAPAS can be used as an encoder in the EncoderDecoderModel framework, to combine it with an autoregressive text decoder such as GPT-2.

## Usage: fine-tuning

Here we explain how you can fine-tune [TapasForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/tapas#transformers.TapasForQuestionAnswering) on your own dataset.

**STEP 1: Choose one of the 3 ways in which you can use TAPAS - or experiment**

Basically, there are 3 different ways in which one can fine-tune [TapasForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/tapas#transformers.TapasForQuestionAnswering), corresponding to the different datasets on which Tapas was fine-tuned:

1. SQA: if you’re interested in asking follow-up questions related to a table, in a conversational set-up. For example if you first ask “what’s the name of the first actor?” then you can ask a follow-up question such as “how old is he?“. Here, questions do not involve any aggregation (all questions are cell selection questions).
2. WTQ: if you’re not interested in asking questions in a conversational set-up, but rather just asking questions related to a table, which might involve aggregation, such as counting a number of rows, summing up cell values or averaging cell values. You can then for example ask “what’s the total number of goals Cristiano Ronaldo made in his career?“. This case is also called **weak supervision**, since the model itself must learn the appropriate aggregation operator (SUM/COUNT/AVERAGE/NONE) given only the answer to the question as supervision.
3. WikiSQL-supervised: this dataset is based on WikiSQL with the model being given the ground truth aggregation operator during training. This is also called **strong supervision**. Here, learning the appropriate aggregation operator is much easier.

To summarize:

| **Task** | **Example dataset** | **Description** |
| --- | --- | --- |
| Conversational | SQA | Conversational, only cell selection questions |
| Weak supervision for aggregation | WTQ | Questions might involve aggregation, and the model must learn this given only the answer as supervision |
| Strong supervision for aggregation | WikiSQL-supervised | Questions might involve aggregation, and the model must learn this given the gold aggregation operator |

Initializing a model with a pre-trained base and randomly initialized classification heads from the hub can be done as shown below.


```
>>> from transformers import TapasConfig, TapasForQuestionAnswering

>>> # for example, the base sized model with default SQA configuration
>>> model = TapasForQuestionAnswering.from_pretrained("google/tapas-base")

>>> # or, the base sized model with WTQ configuration
>>> config = TapasConfig.from_pretrained("google/tapas-base-finetuned-wtq")
>>> model = TapasForQuestionAnswering.from_pretrained("google/tapas-base", config=config)

>>> # or, the base sized model with WikiSQL configuration
>>> config = TapasConfig("google-base-finetuned-wikisql-supervised")
>>> model = TapasForQuestionAnswering.from_pretrained("google/tapas-base", config=config)
```

Of course, you don’t necessarily have to follow one of these three ways in which TAPAS was fine-tuned. You can also experiment by defining any hyperparameters you want when initializing [TapasConfig](/docs/transformers/v4.56.2/en/model_doc/tapas#transformers.TapasConfig), and then create a [TapasForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/tapas#transformers.TapasForQuestionAnswering) based on that configuration. For example, if you have a dataset that has both conversational questions and questions that might involve aggregation, then you can do it this way. Here’s an example:


```
>>> from transformers import TapasConfig, TapasForQuestionAnswering

>>> # you can initialize the classification heads any way you want (see docs of TapasConfig)
>>> config = TapasConfig(num_aggregation_labels=3, average_logits_per_cell=True)
>>> # initializing the pre-trained base sized model with our custom classification heads
>>> model = TapasForQuestionAnswering.from_pretrained("google/tapas-base", config=config)
```

What you can also do is start from an already fine-tuned checkpoint. A note here is that the already fine-tuned checkpoint on WTQ has some issues due to the L2-loss which is somewhat brittle. See [here](https://github.com/google-research/tapas/issues/91#issuecomment-735719340) for more info.

For a list of all pre-trained and fine-tuned TAPAS checkpoints available on HuggingFace’s hub, see [here](https://huggingface.co/models?search=tapas).

**STEP 2: Prepare your data in the SQA format**

Second, no matter what you picked above, you should prepare your dataset in the [SQA](https://www.microsoft.com/en-us/download/details.aspx?id=54253) format. This format is a TSV/CSV file with the following columns:

* `id`: optional, id of the table-question pair, for bookkeeping purposes.
* `annotator`: optional, id of the person who annotated the table-question pair, for bookkeeping purposes.
* `position`: integer indicating if the question is the first, second, third,… related to the table. Only required in case of conversational setup (SQA). You don’t need this column in case you’re going for WTQ/WikiSQL-supervised.
* `question`: string
* `table_file`: string, name of a csv file containing the tabular data
* `answer_coordinates`: list of one or more tuples (each tuple being a cell coordinate, i.e. row, column pair that is part of the answer)
* `answer_text`: list of one or more strings (each string being a cell value that is part of the answer)
* `aggregation_label`: index of the aggregation operator. Only required in case of strong supervision for aggregation (the WikiSQL-supervised case)
* `float_answer`: the float answer to the question, if there is one (np.nan if there isn’t). Only required in case of weak supervision for aggregation (such as WTQ and WikiSQL)

The tables themselves should be present in a folder, each table being a separate csv file. Note that the authors of the TAPAS algorithm used conversion scripts with some automated logic to convert the other datasets (WTQ, WikiSQL) into the SQA format. The author explains this [here](https://github.com/google-research/tapas/issues/50#issuecomment-705465960). A conversion of this script that works with HuggingFace’s implementation can be found [here](https://github.com/NielsRogge/tapas_utils). Interestingly, these conversion scripts are not perfect (the `answer_coordinates` and `float_answer` fields are populated based on the `answer_text`), meaning that WTQ and WikiSQL results could actually be improved.

**STEP 3: Convert your data into tensors using TapasTokenizer**

Third, given that you’ve prepared your data in this TSV/CSV format (and corresponding CSV files containing the tabular data), you can then use [TapasTokenizer](/docs/transformers/v4.56.2/en/model_doc/tapas#transformers.TapasTokenizer) to convert table-question pairs into `input_ids`, `attention_mask`, `token_type_ids` and so on. Again, based on which of the three cases you picked above, [TapasForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/tapas#transformers.TapasForQuestionAnswering) requires different
inputs to be fine-tuned:

| **Task** | **Required inputs** |
| --- | --- |
| Conversational | `input_ids`, `attention_mask`, `token_type_ids`, `labels` |
| Weak supervision for aggregation | `input_ids`, `attention_mask`, `token_type_ids`, `labels`, `numeric_values`, `numeric_values_scale`, `float_answer` |
| Strong supervision for aggregation | `input ids`, `attention mask`, `token type ids`, `labels`, `aggregation_labels` |

[TapasTokenizer](/docs/transformers/v4.56.2/en/model_doc/tapas#transformers.TapasTokenizer) creates the `labels`, `numeric_values` and `numeric_values_scale` based on the `answer_coordinates` and `answer_text` columns of the TSV file. The `float_answer` and `aggregation_labels` are already in the TSV file of step 2. Here’s an example:


```
>>> from transformers import TapasTokenizer
>>> import pandas as pd

>>> model_name = "google/tapas-base"
>>> tokenizer = TapasTokenizer.from_pretrained(model_name)

>>> data = {"Actors": ["Brad Pitt", "Leonardo Di Caprio", "George Clooney"], "Number of movies": ["87", "53", "69"]}
>>> queries = [
...     "What is the name of the first actor?",
...     "How many movies has George Clooney played in?",
...     "What is the total number of movies?",
... ]
>>> answer_coordinates = [[(0, 0)], [(2, 1)], [(0, 1), (1, 1), (2, 1)]]
>>> answer_text = [["Brad Pitt"], ["69"], ["209"]]
>>> table = pd.DataFrame.from_dict(data)
>>> inputs = tokenizer(
...     table=table,
...     queries=queries,
...     answer_coordinates=answer_coordinates,
...     answer_text=answer_text,
...     padding="max_length",
...     return_tensors="pt",
... )
>>> inputs
{'input_ids': tensor([[ ... ]]), 'attention_mask': tensor([[...]]), 'token_type_ids': tensor([[[...]]]),
'numeric_values': tensor([[ ... ]]), 'numeric_values_scale: tensor([[ ... ]]), labels: tensor([[ ... ]])}
```

Note that [TapasTokenizer](/docs/transformers/v4.56.2/en/model_doc/tapas#transformers.TapasTokenizer) expects the data of the table to be **text-only**. You can use `.astype(str)` on a dataframe to turn it into text-only data.
Of course, this only shows how to encode a single training example. It is advised to create a dataloader to iterate over batches:


```
>>> import torch
>>> import pandas as pd

>>> tsv_path = "your_path_to_the_tsv_file"
>>> table_csv_path = "your_path_to_a_directory_containing_all_csv_files"


>>> class TableDataset(torch.utils.data.Dataset):
...     def __init__(self, data, tokenizer):
...         self.data = data
...         self.tokenizer = tokenizer

...     def __getitem__(self, idx):
...         item = data.iloc[idx]
...         table = pd.read_csv(table_csv_path + item.table_file).astype(
...             str
...         )  # be sure to make your table data text only
...         encoding = self.tokenizer(
...             table=table,
...             queries=item.question,
...             answer_coordinates=item.answer_coordinates,
...             answer_text=item.answer_text,
...             truncation=True,
...             padding="max_length",
...             return_tensors="pt",
...         )
...         # remove the batch dimension which the tokenizer adds by default
...         encoding = {key: val.squeeze(0) for key, val in encoding.items()}
...         # add the float_answer which is also required (weak supervision for aggregation case)
...         encoding["float_answer"] = torch.tensor(item.float_answer)
...         return encoding

...     def __len__(self):
...         return len(self.data)


>>> data = pd.read_csv(tsv_path, sep="\t")
>>> train_dataset = TableDataset(data, tokenizer)
>>> train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32)
```

Note that here, we encode each table-question pair independently. This is fine as long as your dataset is **not conversational**. In case your dataset involves conversational questions (such as in SQA), then you should first group together the `queries`, `answer_coordinates` and `answer_text` per table (in the order of their `position`
index) and batch encode each table with its questions. This will make sure that the `prev_labels` token types (see docs of [TapasTokenizer](/docs/transformers/v4.56.2/en/model_doc/tapas#transformers.TapasTokenizer)) are set correctly. See [this notebook](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/TAPAS/Fine_tuning_TapasForQuestionAnswering_on_SQA.ipynb) for more info.

\*\*STEP 4: Train (fine-tune) the model

You can then fine-tune [TapasForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/tapas#transformers.TapasForQuestionAnswering) as follows (shown here for the weak supervision for aggregation case):


```
>>> from transformers import TapasConfig, TapasForQuestionAnswering, AdamW

>>> # this is the default WTQ configuration
>>> config = TapasConfig(
...     num_aggregation_labels=4,
...     use_answer_as_supervision=True,
...     answer_loss_cutoff=0.664694,
...     cell_selection_preference=0.207951,
...     huber_loss_delta=0.121194,
...     init_cell_selection_weights_to_zero=True,
...     select_one_column=True,
...     allow_empty_column_selection=False,
...     temperature=0.0352513,
... )
>>> model = TapasForQuestionAnswering.from_pretrained("google/tapas-base", config=config)

>>> optimizer = AdamW(model.parameters(), lr=5e-5)

>>> model.train()
>>> for epoch in range(2):  # loop over the dataset multiple times
...     for batch in train_dataloader:
...         # get the inputs;
...         input_ids = batch["input_ids"]
...         attention_mask = batch["attention_mask"]
...         token_type_ids = batch["token_type_ids"]
...         labels = batch["labels"]
...         numeric_values = batch["numeric_values"]
...         numeric_values_scale = batch["numeric_values_scale"]
...         float_answer = batch["float_answer"]

...         # zero the parameter gradients
...         optimizer.zero_grad()

...         # forward + backward + optimize
...         outputs = model(
...             input_ids=input_ids,
...             attention_mask=attention_mask,
...             token_type_ids=token_type_ids,
...             labels=labels,
...             numeric_values=numeric_values,
...             numeric_values_scale=numeric_values_scale,
...             float_answer=float_answer,
...         )
...         loss = outputs.loss
...         loss.backward()
...         optimizer.step()
```

## Usage: inference

Here we explain how you can use [TapasForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/tapas#transformers.TapasForQuestionAnswering) for inference (i.e. making predictions on new data). For inference, only `input_ids`, `attention_mask` and `token_type_ids` (which you can obtain using [TapasTokenizer](/docs/transformers/v4.56.2/en/model_doc/tapas#transformers.TapasTokenizer)) have to be provided to the model to obtain the logits. Next, you can use the handy `~models.tapas.tokenization_tapas.convert_logits_to_predictions` method to convert these into predicted coordinates and optional aggregation indices.

However, note that inference is **different** depending on whether or not the setup is conversational. In a non-conversational set-up, inference can be done in parallel on all table-question pairs of a batch. Here’s an example of that:


```
>>> from transformers import TapasTokenizer, TapasForQuestionAnswering
>>> import pandas as pd

>>> model_name = "google/tapas-base-finetuned-wtq"
>>> model = TapasForQuestionAnswering.from_pretrained(model_name)
>>> tokenizer = TapasTokenizer.from_pretrained(model_name)

>>> data = {"Actors": ["Brad Pitt", "Leonardo Di Caprio", "George Clooney"], "Number of movies": ["87", "53", "69"]}
>>> queries = [
...     "What is the name of the first actor?",
...     "How many movies has George Clooney played in?",
...     "What is the total number of movies?",
... ]
>>> table = pd.DataFrame.from_dict(data)
>>> inputs = tokenizer(table=table, queries=queries, padding="max_length", return_tensors="pt")
>>> outputs = model(**inputs)
>>> predicted_answer_coordinates, predicted_aggregation_indices = tokenizer.convert_logits_to_predictions(
...     inputs, outputs.logits.detach(), outputs.logits_aggregation.detach()
... )

>>> # let's print out the results:
>>> id2aggregation = {0: "NONE", 1: "SUM", 2: "AVERAGE", 3: "COUNT"}
>>> aggregation_predictions_string = [id2aggregation[x] for x in predicted_aggregation_indices]

>>> answers = []
>>> for coordinates in predicted_answer_coordinates:
...     if len(coordinates) == 1:
...         # only a single cell:
...         answers.append(table.iat[coordinates[0]])
...     else:
...         # multiple cells
...         cell_values = []
...         for coordinate in coordinates:
...             cell_values.append(table.iat[coordinate])
...         answers.append(", ".join(cell_values))

>>> display(table)
>>> print("")
>>> for query, answer, predicted_agg in zip(queries, answers, aggregation_predictions_string):
...     print(query)
...     if predicted_agg == "NONE":
...         print("Predicted answer: " + answer)
...     else:
...         print("Predicted answer: " + predicted_agg + " > " + answer)
What is the name of the first actor?
Predicted answer: Brad Pitt
How many movies has George Clooney played in?
Predicted answer: COUNT > 69
What is the total number of movies?
Predicted answer: SUM > 87, 53, 69
```

In case of a conversational set-up, then each table-question pair must be provided **sequentially** to the model, such that the `prev_labels` token types can be overwritten by the predicted `labels` of the previous table-question pair. Again, more info can be found in [this notebook](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/TAPAS/Fine_tuning_TapasForQuestionAnswering_on_SQA.ipynb).

## Resources

* [Text classification task guide](../tasks/sequence_classification)
* [Masked language modeling task guide](../tasks/masked_language_modeling)

## TAPAS specific outputs

### class transformers.models.tapas.modeling\_tapas.TableQuestionAnsweringOutput

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/tapas/modeling_tapas.py#L52)

( loss: typing.Optional[torch.FloatTensor] = None logits: typing.Optional[torch.FloatTensor] = None logits\_aggregation: typing.Optional[torch.FloatTensor] = None hidden\_states: typing.Optional[tuple[torch.FloatTensor]] = None attentions: typing.Optional[tuple[torch.FloatTensor]] = None  )

Parameters

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` (and possibly `answer`, `aggregation_labels`, `numeric_values` and `numeric_values_scale` are provided)) —
  Total loss as the sum of the hierarchical cell selection log-likelihood loss and (optionally) the
  semi-supervised regression loss and (optionally) supervised loss for aggregations.
* **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`) —
  Prediction scores of the cell selection head, for every token.
* **logits\_aggregation** (`torch.FloatTensor`, *optional*, of shape `(batch_size, num_aggregation_labels)`) —
  Prediction scores of the aggregation head, for every aggregation operator.
* **hidden\_states** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) —
  Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) —
  Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

Output type of [TapasForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/tapas#transformers.TapasForQuestionAnswering).

## TapasConfig

### class transformers.TapasConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/tapas/configuration_tapas.py#L28)

( vocab\_size = 30522 hidden\_size = 768 num\_hidden\_layers = 12 num\_attention\_heads = 12 intermediate\_size = 3072 hidden\_act = 'gelu' hidden\_dropout\_prob = 0.1 attention\_probs\_dropout\_prob = 0.1 max\_position\_embeddings = 1024 type\_vocab\_sizes = [3, 256, 256, 2, 256, 256, 10] initializer\_range = 0.02 layer\_norm\_eps = 1e-12 pad\_token\_id = 0 positive\_label\_weight = 10.0 num\_aggregation\_labels = 0 aggregation\_loss\_weight = 1.0 use\_answer\_as\_supervision = None answer\_loss\_importance = 1.0 use\_normalized\_answer\_loss = False huber\_loss\_delta = None temperature = 1.0 aggregation\_temperature = 1.0 use\_gumbel\_for\_cells = False use\_gumbel\_for\_aggregation = False average\_approximation\_function = 'ratio' cell\_selection\_preference = None answer\_loss\_cutoff = None max\_num\_rows = 64 max\_num\_columns = 32 average\_logits\_per\_cell = False select\_one\_column = True allow\_empty\_column\_selection = False init\_cell\_selection\_weights\_to\_zero = False reset\_position\_index\_per\_cell = True disable\_per\_token\_loss = False aggregation\_labels = None no\_aggregation\_label\_index = None \*\*kwargs  )

Parameters

* **vocab\_size** (`int`, *optional*, defaults to 30522) —
  Vocabulary size of the TAPAS model. Defines the number of different tokens that can be represented by the
  `inputs_ids` passed when calling [TapasModel](/docs/transformers/v4.56.2/en/model_doc/tapas#transformers.TapasModel).
* **hidden\_size** (`int`, *optional*, defaults to 768) —
  Dimensionality of the encoder layers and the pooler layer.
* **num\_hidden\_layers** (`int`, *optional*, defaults to 12) —
  Number of hidden layers in the Transformer encoder.
* **num\_attention\_heads** (`int`, *optional*, defaults to 12) —
  Number of attention heads for each attention layer in the Transformer encoder.
* **intermediate\_size** (`int`, *optional*, defaults to 3072) —
  Dimensionality of the “intermediate” (often named feed-forward) layer in the Transformer encoder.
* **hidden\_act** (`str` or `Callable`, *optional*, defaults to `"gelu"`) —
  The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
  `"relu"`, `"swish"` and `"gelu_new"` are supported.
* **hidden\_dropout\_prob** (`float`, *optional*, defaults to 0.1) —
  The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
* **attention\_probs\_dropout\_prob** (`float`, *optional*, defaults to 0.1) —
  The dropout ratio for the attention probabilities.
* **max\_position\_embeddings** (`int`, *optional*, defaults to 1024) —
  The maximum sequence length that this model might ever be used with. Typically set this to something large
  just in case (e.g., 512 or 1024 or 2048).
* **type\_vocab\_sizes** (`list[int]`, *optional*, defaults to `[3, 256, 256, 2, 256, 256, 10]`) —
  The vocabulary sizes of the `token_type_ids` passed when calling [TapasModel](/docs/transformers/v4.56.2/en/model_doc/tapas#transformers.TapasModel).
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
* **layer\_norm\_eps** (`float`, *optional*, defaults to 1e-12) —
  The epsilon used by the layer normalization layers.
* **positive\_label\_weight** (`float`, *optional*, defaults to 10.0) —
  Weight for positive labels.
* **num\_aggregation\_labels** (`int`, *optional*, defaults to 0) —
  The number of aggregation operators to predict.
* **aggregation\_loss\_weight** (`float`, *optional*, defaults to 1.0) —
  Importance weight for the aggregation loss.
* **use\_answer\_as\_supervision** (`bool`, *optional*) —
  Whether to use the answer as the only supervision for aggregation examples.
* **answer\_loss\_importance** (`float`, *optional*, defaults to 1.0) —
  Importance weight for the regression loss.
* **use\_normalized\_answer\_loss** (`bool`, *optional*, defaults to `False`) —
  Whether to normalize the answer loss by the maximum of the predicted and expected value.
* **huber\_loss\_delta** (`float`, *optional*) —
  Delta parameter used to calculate the regression loss.
* **temperature** (`float`, *optional*, defaults to 1.0) —
  Value used to control (OR change) the skewness of cell logits probabilities.
* **aggregation\_temperature** (`float`, *optional*, defaults to 1.0) —
  Scales aggregation logits to control the skewness of probabilities.
* **use\_gumbel\_for\_cells** (`bool`, *optional*, defaults to `False`) —
  Whether to apply Gumbel-Softmax to cell selection.
* **use\_gumbel\_for\_aggregation** (`bool`, *optional*, defaults to `False`) —
  Whether to apply Gumbel-Softmax to aggregation selection.
* **average\_approximation\_function** (`string`, *optional*, defaults to `"ratio"`) —
  Method to calculate the expected average of cells in the weak supervision case. One of `"ratio"`,
  `"first_order"` or `"second_order"`.
* **cell\_selection\_preference** (`float`, *optional*) —
  Preference for cell selection in ambiguous cases. Only applicable in case of weak supervision for
  aggregation (WTQ, WikiSQL). If the total mass of the aggregation probabilities (excluding the “NONE”
  operator) is higher than this hyperparameter, then aggregation is predicted for an example.
* **answer\_loss\_cutoff** (`float`, *optional*) —
  Ignore examples with answer loss larger than cutoff.
* **max\_num\_rows** (`int`, *optional*, defaults to 64) —
  Maximum number of rows.
* **max\_num\_columns** (`int`, *optional*, defaults to 32) —
  Maximum number of columns.
* **average\_logits\_per\_cell** (`bool`, *optional*, defaults to `False`) —
  Whether to average logits per cell.
* **select\_one\_column** (`bool`, *optional*, defaults to `True`) —
  Whether to constrain the model to only select cells from a single column.
* **allow\_empty\_column\_selection** (`bool`, *optional*, defaults to `False`) —
  Whether to allow not to select any column.
* **init\_cell\_selection\_weights\_to\_zero** (`bool`, *optional*, defaults to `False`) —
  Whether to initialize cell selection weights to 0 so that the initial probabilities are 50%.
* **reset\_position\_index\_per\_cell** (`bool`, *optional*, defaults to `True`) —
  Whether to restart position indexes at every cell (i.e. use relative position embeddings).
* **disable\_per\_token\_loss** (`bool`, *optional*, defaults to `False`) —
  Whether to disable any (strong or weak) supervision on cells.
* **aggregation\_labels** (`dict[int, label]`, *optional*) —
  The aggregation labels used to aggregate the results. For example, the WTQ models have the following
  aggregation labels: `{0: "NONE", 1: "SUM", 2: "AVERAGE", 3: "COUNT"}`
* **no\_aggregation\_label\_index** (`int`, *optional*) —
  If the aggregation labels are defined and one of these labels represents “No aggregation”, this should be
  set to its index. For example, the WTQ models have the “NONE” aggregation label at index 0, so that value
  should be set to 0 for these models.

This is the configuration class to store the configuration of a [TapasModel](/docs/transformers/v4.56.2/en/model_doc/tapas#transformers.TapasModel). It is used to instantiate a TAPAS
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the TAPAS
[google/tapas-base-finetuned-sqa](https://huggingface.co/google/tapas-base-finetuned-sqa) architecture.

Configuration objects inherit from `PreTrainedConfig` and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Hyperparameters additional to BERT are taken from run\_task\_main.py and hparam\_utils.py of the original
implementation. Original implementation available at <https://github.com/google-research/tapas/tree/master>.

Example:


```
>>> from transformers import TapasModel, TapasConfig

>>> # Initializing a default (SQA) Tapas configuration
>>> configuration = TapasConfig()
>>> # Initializing a model from the configuration
>>> model = TapasModel(configuration)
>>> # Accessing the model configuration
>>> configuration = model.config
```

## TapasTokenizer

### class transformers.TapasTokenizer

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/tapas/tokenization_tapas.py#L151)

( vocab\_file do\_lower\_case = True do\_basic\_tokenize = True never\_split = None unk\_token = '[UNK]' sep\_token = '[SEP]' pad\_token = '[PAD]' cls\_token = '[CLS]' mask\_token = '[MASK]' empty\_token = '[EMPTY]' tokenize\_chinese\_chars = True strip\_accents = None cell\_trim\_length: int = -1 max\_column\_id: typing.Optional[int] = None max\_row\_id: typing.Optional[int] = None strip\_column\_names: bool = False update\_answer\_coordinates: bool = False min\_question\_length = None max\_question\_length = None model\_max\_length: int = 512 additional\_special\_tokens: typing.Optional[list[str]] = None clean\_up\_tokenization\_spaces = True \*\*kwargs  )

Parameters

* **vocab\_file** (`str`) —
  File containing the vocabulary.
* **do\_lower\_case** (`bool`, *optional*, defaults to `True`) —
  Whether or not to lowercase the input when tokenizing.
* **do\_basic\_tokenize** (`bool`, *optional*, defaults to `True`) —
  Whether or not to do basic tokenization before WordPiece.
* **never\_split** (`Iterable`, *optional*) —
  Collection of tokens which will never be split during tokenization. Only has an effect when
  `do_basic_tokenize=True`
* **unk\_token** (`str`, *optional*, defaults to `"[UNK]"`) —
  The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
  token instead.
* **sep\_token** (`str`, *optional*, defaults to `"[SEP]"`) —
  The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
  sequence classification or for a text and a question for question answering. It is also used as the last
  token of a sequence built with special tokens.
* **pad\_token** (`str`, *optional*, defaults to `"[PAD]"`) —
  The token used for padding, for example when batching sequences of different lengths.
* **cls\_token** (`str`, *optional*, defaults to `"[CLS]"`) —
  The classifier token which is used when doing sequence classification (classification of the whole sequence
  instead of per-token classification). It is the first token of the sequence when built with special tokens.
* **mask\_token** (`str`, *optional*, defaults to `"[MASK]"`) —
  The token used for masking values. This is the token used when training this model with masked language
  modeling. This is the token which the model will try to predict.
* **empty\_token** (`str`, *optional*, defaults to `"[EMPTY]"`) —
  The token used for empty cell values in a table. Empty cell values include "", “n/a”, “nan” and ”?“.
* **tokenize\_chinese\_chars** (`bool`, *optional*, defaults to `True`) —
  Whether or not to tokenize Chinese characters. This should likely be deactivated for Japanese (see this
  [issue](https://github.com/huggingface/transformers/issues/328)).
* **strip\_accents** (`bool`, *optional*) —
  Whether or not to strip all accents. If this option is not specified, then it will be determined by the
  value for `lowercase` (as in the original BERT).
* **cell\_trim\_length** (`int`, *optional*, defaults to -1) —
  If > 0: Trim cells so that the length is <= this value. Also disables further cell trimming, should thus be
  used with `truncation` set to `True`.
* **max\_column\_id** (`int`, *optional*) —
  Max column id to extract.
* **max\_row\_id** (`int`, *optional*) —
  Max row id to extract.
* **strip\_column\_names** (`bool`, *optional*, defaults to `False`) —
  Whether to add empty strings instead of column names.
* **update\_answer\_coordinates** (`bool`, *optional*, defaults to `False`) —
  Whether to recompute the answer coordinates from the answer text.
* **min\_question\_length** (`int`, *optional*) —
  Minimum length of each question in terms of tokens (will be skipped otherwise).
* **max\_question\_length** (`int`, *optional*) —
  Maximum length of each question in terms of tokens (will be skipped otherwise).
* **clean\_up\_tokenization\_spaces** (`bool`, *optional*, defaults to `True`) —
  Whether or not to cleanup spaces after decoding, cleanup consists in removing potential artifacts like
  extra spaces.

Construct a TAPAS tokenizer. Based on WordPiece. Flattens a table and one or more related sentences to be used by
TAPAS models.

This tokenizer inherits from [PreTrainedTokenizer](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer) which contains most of the main methods. Users should refer to
this superclass for more information regarding those methods. [TapasTokenizer](/docs/transformers/v4.56.2/en/model_doc/tapas#transformers.TapasTokenizer) creates several token type ids to
encode tabular structure. To be more precise, it adds 7 token type ids, in the following order: `segment_ids`,
`column_ids`, `row_ids`, `prev_labels`, `column_ranks`, `inv_column_ranks` and `numeric_relations`:

* segment\_ids: indicate whether a token belongs to the question (0) or the table (1). 0 for special tokens and
  padding.
* column\_ids: indicate to which column of the table a token belongs (starting from 1). Is 0 for all question
  tokens, special tokens and padding.
* row\_ids: indicate to which row of the table a token belongs (starting from 1). Is 0 for all question tokens,
  special tokens and padding. Tokens of column headers are also 0.
* prev\_labels: indicate whether a token was (part of) an answer to the previous question (1) or not (0). Useful in
  a conversational setup (such as SQA).
* column\_ranks: indicate the rank of a table token relative to a column, if applicable. For example, if you have a
  column “number of movies” with values 87, 53 and 69, then the column ranks of these tokens are 3, 1 and 2
  respectively. 0 for all question tokens, special tokens and padding.
* inv\_column\_ranks: indicate the inverse rank of a table token relative to a column, if applicable. For example, if
  you have a column “number of movies” with values 87, 53 and 69, then the inverse column ranks of these tokens are
  1, 3 and 2 respectively. 0 for all question tokens, special tokens and padding.
* numeric\_relations: indicate numeric relations between the question and the tokens of the table. 0 for all
  question tokens, special tokens and padding.

[TapasTokenizer](/docs/transformers/v4.56.2/en/model_doc/tapas#transformers.TapasTokenizer) runs end-to-end tokenization on a table and associated sentences: punctuation splitting and
wordpiece.

#### \_\_call\_\_

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/tapas/tokenization_tapas.py#L505)

( table: pd.DataFrame queries: typing.Union[str, list[str], list[int], list[list[str]], list[list[int]], NoneType] = None answer\_coordinates: typing.Union[list[tuple], list[list[tuple]], NoneType] = None answer\_text: typing.Union[list[str], list[list[str]], NoneType] = None add\_special\_tokens: bool = True padding: typing.Union[bool, str, transformers.utils.generic.PaddingStrategy] = False truncation: typing.Union[bool, str, transformers.models.tapas.tokenization\_tapas.TapasTruncationStrategy] = False max\_length: typing.Optional[int] = None pad\_to\_multiple\_of: typing.Optional[int] = None padding\_side: typing.Optional[str] = None return\_tensors: typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None return\_token\_type\_ids: typing.Optional[bool] = None return\_attention\_mask: typing.Optional[bool] = None return\_overflowing\_tokens: bool = False return\_special\_tokens\_mask: bool = False return\_offsets\_mapping: bool = False return\_length: bool = False verbose: bool = True \*\*kwargs  )

Parameters

* **table** (`pd.DataFrame`) —
  Table containing tabular data. Note that all cell values must be text. Use *.astype(str)* on a Pandas
  dataframe to convert it to string.
* **queries** (`str` or `list[str]`) —
  Question or batch of questions related to a table to be encoded. Note that in case of a batch, all
  questions must refer to the **same** table.
* **answer\_coordinates** (`list[Tuple]` or `list[list[Tuple]]`, *optional*) —
  Answer coordinates of each table-question pair in the batch. In case only a single table-question pair
  is provided, then the answer\_coordinates must be a single list of one or more tuples. Each tuple must
  be a (row\_index, column\_index) pair. The first data row (not the column header row) has index 0. The
  first column has index 0. In case a batch of table-question pairs is provided, then the
  answer\_coordinates must be a list of lists of tuples (each list corresponding to a single
  table-question pair).
* **answer\_text** (`list[str]` or `list[list[str]]`, *optional*) —
  Answer text of each table-question pair in the batch. In case only a single table-question pair is
  provided, then the answer\_text must be a single list of one or more strings. Each string must be the
  answer text of a corresponding answer coordinate. In case a batch of table-question pairs is provided,
  then the answer\_coordinates must be a list of lists of strings (each list corresponding to a single
  table-question pair).
* **add\_special\_tokens** (`bool`, *optional*, defaults to `True`) —
  Whether or not to encode the sequences with the special tokens relative to their model.
* **padding** (`bool`, `str` or [PaddingStrategy](/docs/transformers/v4.56.2/en/internal/file_utils#transformers.utils.PaddingStrategy), *optional*, defaults to `False`) —
  Activates and controls padding. Accepts the following values:
  + `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
    sequence if provided).
  + `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
    acceptable input length for the model if that argument is not provided.
  + `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
    lengths).
* **truncation** (`bool`, `str` or `TapasTruncationStrategy`, *optional*, defaults to `False`) —
  Activates and controls truncation. Accepts the following values:
  + `True` or `'drop_rows_to_fit'`: Truncate to a maximum length specified with the argument `max_length`
    or to the maximum acceptable input length for the model if that argument is not provided. This will
    truncate row by row, removing rows from the table.
  + `False` or `'do_not_truncate'` (default): No truncation (i.e., can output batch with sequence lengths
    greater than the model maximum admissible input size).
* **max\_length** (`int`, *optional*) —
  Controls the maximum length to use by one of the truncation/padding parameters.

  If left unset or set to `None`, this will use the predefined model maximum length if a maximum length
  is required by one of the truncation/padding parameters. If the model has no specific maximum input
  length (like XLNet) truncation/padding to a maximum length will be deactivated.
* **is\_split\_into\_words** (`bool`, *optional*, defaults to `False`) —
  Whether or not the input is already pre-tokenized (e.g., split into words). If set to `True`, the
  tokenizer assumes the input is already split into words (for instance, by splitting it on whitespace)
  which it will tokenize. This is useful for NER or token classification.
* **pad\_to\_multiple\_of** (`int`, *optional*) —
  If set will pad the sequence to a multiple of the provided value. This is especially useful to enable
  the use of Tensor Cores on NVIDIA hardware with compute capability `>= 7.5` (Volta).
* **return\_tensors** (`str` or [TensorType](/docs/transformers/v4.56.2/en/internal/file_utils#transformers.TensorType), *optional*) —
  If set, will return tensors instead of list of python integers. Acceptable values are:
  + `'tf'`: Return TensorFlow `tf.constant` objects.
  + `'pt'`: Return PyTorch `torch.Tensor` objects.
  + `'np'`: Return Numpy `np.ndarray` objects.

Main method to tokenize and prepare for the model one or several sequence(s) related to a table.

#### convert\_logits\_to\_predictions

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/tapas/tokenization_tapas.py#L1889)

( data logits logits\_agg = None cell\_classification\_threshold = 0.5  ) → `tuple` comprising various elements depending on the inputs

Parameters

* **data** (`dict`) —
  Dictionary mapping features to actual values. Should be created using [TapasTokenizer](/docs/transformers/v4.56.2/en/model_doc/tapas#transformers.TapasTokenizer).
* **logits** (`torch.Tensor` or `tf.Tensor` of shape `(batch_size, sequence_length)`) —
  Tensor containing the logits at the token level.
* **logits\_agg** (`torch.Tensor` or `tf.Tensor` of shape `(batch_size, num_aggregation_labels)`, *optional*) —
  Tensor containing the aggregation logits.
* **cell\_classification\_threshold** (`float`, *optional*, defaults to 0.5) —
  Threshold to be used for cell selection. All table cells for which their probability is larger than
  this threshold will be selected.

Returns

`tuple` comprising various elements depending on the inputs

* predicted\_answer\_coordinates (`list[list[[tuple]]` of length `batch_size`): Predicted answer coordinates
  as a list of lists of tuples. Each element in the list contains the predicted answer coordinates of a
  single example in the batch, as a list of tuples. Each tuple is a cell, i.e. (row index, column index).
* predicted\_aggregation\_indices (`list[int]`of length `batch_size`, *optional*, returned when
  `logits_aggregation` is provided): Predicted aggregation operator indices of the aggregation head.

Converts logits of [TapasForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/tapas#transformers.TapasForQuestionAnswering) to actual predicted answer coordinates and optional
aggregation indices.

The original implementation, on which this function is based, can be found
[here](https://github.com/google-research/tapas/blob/4908213eb4df7aa988573350278b44c4dbe3f71b/tapas/experiments/prediction_utils.py#L288).

#### save\_vocabulary

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/tapas/tokenization_tapas.py#L373)

( save\_directory: str filename\_prefix: typing.Optional[str] = None  )

## TapasModel

### class transformers.TapasModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/tapas/modeling_tapas.py#L719)

( config add\_pooling\_layer = True  )

Parameters

* **config** ([TapasModel](/docs/transformers/v4.56.2/en/model_doc/tapas#transformers.TapasModel)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.
* **add\_pooling\_layer** (`bool`, *optional*, defaults to `True`) —
  Whether to add a pooling layer

The bare Tapas Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/tapas/modeling_tapas.py#L760)

( input\_ids: typing.Optional[torch.LongTensor] = None attention\_mask: typing.Optional[torch.FloatTensor] = None token\_type\_ids: typing.Optional[torch.LongTensor] = None position\_ids: typing.Optional[torch.LongTensor] = None head\_mask: typing.Optional[torch.FloatTensor] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None encoder\_hidden\_states: typing.Optional[torch.FloatTensor] = None encoder\_attention\_mask: typing.Optional[torch.FloatTensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → [transformers.modeling\_outputs.BaseModelOutputWithPooling](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **attention\_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **token\_type\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length, 7)`, *optional*) —
  Token indices that encode tabular structure. Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See this
  class for more info.

  [What are token type IDs?](../glossary#token-type-ids)
* **position\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of positions of each input sequence tokens in the position embeddings. If
  `reset_position_index_per_cell` of [TapasConfig](/docs/transformers/v4.56.2/en/model_doc/tapas#transformers.TapasConfig) is set to `True`, relative position embeddings will be
  used. Selected in the range `[0, config.max_position_embeddings - 1]`.

  [What are position IDs?](../glossary#position-ids)
* **head\_mask** (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model’s internal embedding lookup matrix.
* **encoder\_hidden\_states** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
  if the model is configured as a decoder.
* **encoder\_attention\_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
  the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

[transformers.modeling\_outputs.BaseModelOutputWithPooling](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.BaseModelOutputWithPooling](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([TapasConfig](/docs/transformers/v4.56.2/en/model_doc/tapas#transformers.TapasConfig)) and inputs.

* **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the model.
* **pooler\_output** (`torch.FloatTensor` of shape `(batch_size, hidden_size)`) — Last layer hidden-state of the first token of the sequence (classification token) after further processing
  through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns
  the classification token after processing through a linear layer and a tanh activation function. The linear
  layer weights are trained from the next sentence prediction (classification) objective during pretraining.
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [TapasModel](/docs/transformers/v4.56.2/en/model_doc/tapas#transformers.TapasModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> from transformers import AutoTokenizer, TapasModel
>>> import pandas as pd

>>> tokenizer = AutoTokenizer.from_pretrained("google/tapas-base")
>>> model = TapasModel.from_pretrained("google/tapas-base")

>>> data = {
...     "Actors": ["Brad Pitt", "Leonardo Di Caprio", "George Clooney"],
...     "Age": ["56", "45", "59"],
...     "Number of movies": ["87", "53", "69"],
... }
>>> table = pd.DataFrame.from_dict(data)
>>> queries = ["How many movies has George Clooney played in?", "How old is Brad Pitt?"]

>>> inputs = tokenizer(table=table, queries=queries, padding="max_length", return_tensors="pt")
>>> outputs = model(**inputs)

>>> last_hidden_states = outputs.last_hidden_state
```

## TapasForMaskedLM

### class transformers.TapasForMaskedLM

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/tapas/modeling_tapas.py#L885)

( config  )

Parameters

* **config** ([TapasForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/tapas#transformers.TapasForMaskedLM)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Tapas Model with a `language modeling` head on top.”

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/tapas/modeling_tapas.py#L906)

( input\_ids: typing.Optional[torch.LongTensor] = None attention\_mask: typing.Optional[torch.FloatTensor] = None token\_type\_ids: typing.Optional[torch.LongTensor] = None position\_ids: typing.Optional[torch.LongTensor] = None head\_mask: typing.Optional[torch.FloatTensor] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None encoder\_hidden\_states: typing.Optional[torch.FloatTensor] = None encoder\_attention\_mask: typing.Optional[torch.FloatTensor] = None labels: typing.Optional[torch.LongTensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None \*\*kwargs  ) → [transformers.modeling\_outputs.MaskedLMOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.MaskedLMOutput) or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **attention\_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **token\_type\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length, 7)`, *optional*) —
  Token indices that encode tabular structure. Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See this
  class for more info.

  [What are token type IDs?](../glossary#token-type-ids)
* **position\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of positions of each input sequence tokens in the position embeddings. If
  `reset_position_index_per_cell` of [TapasConfig](/docs/transformers/v4.56.2/en/model_doc/tapas#transformers.TapasConfig) is set to `True`, relative position embeddings will be
  used. Selected in the range `[0, config.max_position_embeddings - 1]`.

  [What are position IDs?](../glossary#position-ids)
* **head\_mask** (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model’s internal embedding lookup matrix.
* **encoder\_hidden\_states** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
  if the model is configured as a decoder.
* **encoder\_attention\_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
  the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.
* **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ..., config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
  loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

[transformers.modeling\_outputs.MaskedLMOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.MaskedLMOutput) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.MaskedLMOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.MaskedLMOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([TapasConfig](/docs/transformers/v4.56.2/en/model_doc/tapas#transformers.TapasConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Masked language modeling (MLM) loss.
* **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [TapasForMaskedLM](/docs/transformers/v4.56.2/en/model_doc/tapas#transformers.TapasForMaskedLM) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> from transformers import AutoTokenizer, TapasForMaskedLM
>>> import pandas as pd

>>> tokenizer = AutoTokenizer.from_pretrained("google/tapas-base")
>>> model = TapasForMaskedLM.from_pretrained("google/tapas-base")

>>> data = {
...     "Actors": ["Brad Pitt", "Leonardo Di Caprio", "George Clooney"],
...     "Age": ["56", "45", "59"],
...     "Number of movies": ["87", "53", "69"],
... }
>>> table = pd.DataFrame.from_dict(data)

>>> inputs = tokenizer(
...     table=table, queries="How many [MASK] has George [MASK] played in?", return_tensors="pt"
... )
>>> labels = tokenizer(
...     table=table, queries="How many movies has George Clooney played in?", return_tensors="pt"
... )["input_ids"]

>>> outputs = model(**inputs, labels=labels)
>>> logits = outputs.logits
```

## TapasForSequenceClassification

### class transformers.TapasForSequenceClassification

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/tapas/modeling_tapas.py#L1360)

( config  )

Parameters

* **config** ([TapasForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/tapas#transformers.TapasForSequenceClassification)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Tapas Model with a sequence classification head on top (a linear layer on top of the pooled output), e.g. for table
entailment tasks, such as TabFact (Chen et al., 2020).

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/tapas/modeling_tapas.py#L1372)

( input\_ids: typing.Optional[torch.LongTensor] = None attention\_mask: typing.Optional[torch.FloatTensor] = None token\_type\_ids: typing.Optional[torch.LongTensor] = None position\_ids: typing.Optional[torch.LongTensor] = None head\_mask: typing.Optional[torch.FloatTensor] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None labels: typing.Optional[torch.LongTensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → [transformers.modeling\_outputs.SequenceClassifierOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput) or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **attention\_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **token\_type\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length, 7)`, *optional*) —
  Token indices that encode tabular structure. Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See this
  class for more info.

  [What are token type IDs?](../glossary#token-type-ids)
* **position\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of positions of each input sequence tokens in the position embeddings. If
  `reset_position_index_per_cell` of [TapasConfig](/docs/transformers/v4.56.2/en/model_doc/tapas#transformers.TapasConfig) is set to `True`, relative position embeddings will be
  used. Selected in the range `[0, config.max_position_embeddings - 1]`.

  [What are position IDs?](../glossary#position-ids)
* **head\_mask** (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model’s internal embedding lookup matrix.
* **labels** (`torch.LongTensor` of shape `(batch_size,)`, *optional*) —
  Labels for computing the sequence classification/regression loss. Indices should be in `[0, ..., config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
  `config.num_labels > 1` a classification loss is computed (Cross-Entropy). Note: this is called
  “classification\_class\_index” in the original implementation.
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

[transformers.modeling\_outputs.SequenceClassifierOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.SequenceClassifierOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([TapasConfig](/docs/transformers/v4.56.2/en/model_doc/tapas#transformers.TapasConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Classification (or regression if config.num\_labels==1) loss.
* **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) — Classification (or regression if config.num\_labels==1) scores (before SoftMax).
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [TapasForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/tapas#transformers.TapasForSequenceClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> from transformers import AutoTokenizer, TapasForSequenceClassification
>>> import torch
>>> import pandas as pd

>>> tokenizer = AutoTokenizer.from_pretrained("google/tapas-base-finetuned-tabfact")
>>> model = TapasForSequenceClassification.from_pretrained("google/tapas-base-finetuned-tabfact")

>>> data = {
...     "Actors": ["Brad Pitt", "Leonardo Di Caprio", "George Clooney"],
...     "Age": ["56", "45", "59"],
...     "Number of movies": ["87", "53", "69"],
... }
>>> table = pd.DataFrame.from_dict(data)
>>> queries = [
...     "There is only one actor who is 45 years old",
...     "There are 3 actors which played in more than 60 movies",
... ]

>>> inputs = tokenizer(table=table, queries=queries, padding="max_length", return_tensors="pt")
>>> labels = torch.tensor([1, 0])  # 1 means entailed, 0 means refuted

>>> outputs = model(**inputs, labels=labels)
>>> loss = outputs.loss
>>> logits = outputs.logits
```

## TapasForQuestionAnswering

### class transformers.TapasForQuestionAnswering

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/tapas/modeling_tapas.py#L1009)

( config: TapasConfig  )

Parameters

* **config** ([TapasConfig](/docs/transformers/v4.56.2/en/model_doc/tapas#transformers.TapasConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Tapas Model with a cell selection head and optional aggregation head on top for question-answering tasks on tables
(linear layers on top of the hidden-states output to compute `logits` and optional `logits_aggregation`), e.g. for
SQA, WTQ or WikiSQL-supervised tasks.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/tapas/modeling_tapas.py#L1044)

( input\_ids: typing.Optional[torch.LongTensor] = None attention\_mask: typing.Optional[torch.FloatTensor] = None token\_type\_ids: typing.Optional[torch.LongTensor] = None position\_ids: typing.Optional[torch.LongTensor] = None head\_mask: typing.Optional[torch.FloatTensor] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None table\_mask: typing.Optional[torch.LongTensor] = None labels: typing.Optional[torch.LongTensor] = None aggregation\_labels: typing.Optional[torch.LongTensor] = None float\_answer: typing.Optional[torch.FloatTensor] = None numeric\_values: typing.Optional[torch.FloatTensor] = None numeric\_values\_scale: typing.Optional[torch.FloatTensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → [transformers.models.tapas.modeling\_tapas.TableQuestionAnsweringOutput](/docs/transformers/v4.56.2/en/model_doc/tapas#transformers.models.tapas.modeling_tapas.TableQuestionAnsweringOutput) or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **attention\_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **token\_type\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length, 7)`, *optional*) —
  Token indices that encode tabular structure. Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See this
  class for more info.

  [What are token type IDs?](../glossary#token-type-ids)
* **position\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of positions of each input sequence tokens in the position embeddings. If
  `reset_position_index_per_cell` of [TapasConfig](/docs/transformers/v4.56.2/en/model_doc/tapas#transformers.TapasConfig) is set to `True`, relative position embeddings will be
  used. Selected in the range `[0, config.max_position_embeddings - 1]`.

  [What are position IDs?](../glossary#position-ids)
* **head\_mask** (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model’s internal embedding lookup matrix.
* **table\_mask** (`torch.LongTensor` of shape `(batch_size, seq_length)`, *optional*) —
  Mask for the table. Indicates which tokens belong to the table (1). Question tokens, table headers and
  padding are 0.
* **labels** (`torch.LongTensor` of shape `(batch_size, seq_length)`, *optional*) —
  Labels per token for computing the hierarchical cell selection loss. This encodes the positions of the
  answer appearing in the table. Can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer).
  + 1 for tokens that are **part of the answer**,
  + 0 for tokens that are **not part of the answer**.
* **aggregation\_labels** (`torch.LongTensor` of shape `(batch_size, )`, *optional*) —
  Aggregation function index for every example in the batch for computing the aggregation loss. Indices
  should be in `[0, ..., config.num_aggregation_labels - 1]`. Only required in case of strong supervision for
  aggregation (WikiSQL-supervised).
* **float\_answer** (`torch.FloatTensor` of shape `(batch_size, )`, *optional*) —
  Float answer for every example in the batch. Set to *float(‘nan’)* for cell selection questions. Only
  required in case of weak supervision (WTQ) to calculate the aggregate mask and regression loss.
* **numeric\_values** (`torch.FloatTensor` of shape `(batch_size, seq_length)`, *optional*) —
  Numeric values of every token, NaN for tokens which are not numeric values. Can be obtained using
  [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). Only required in case of weak supervision for aggregation (WTQ) to calculate the
  regression loss.
* **numeric\_values\_scale** (`torch.FloatTensor` of shape `(batch_size, seq_length)`, *optional*) —
  Scale of the numeric values of every token. Can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). Only required in case
  of weak supervision for aggregation (WTQ) to calculate the regression loss.
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

[transformers.models.tapas.modeling\_tapas.TableQuestionAnsweringOutput](/docs/transformers/v4.56.2/en/model_doc/tapas#transformers.models.tapas.modeling_tapas.TableQuestionAnsweringOutput) or `tuple(torch.FloatTensor)`

A [transformers.models.tapas.modeling\_tapas.TableQuestionAnsweringOutput](/docs/transformers/v4.56.2/en/model_doc/tapas#transformers.models.tapas.modeling_tapas.TableQuestionAnsweringOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([TapasConfig](/docs/transformers/v4.56.2/en/model_doc/tapas#transformers.TapasConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` (and possibly `answer`, `aggregation_labels`, `numeric_values` and `numeric_values_scale` are provided)) — Total loss as the sum of the hierarchical cell selection log-likelihood loss and (optionally) the
  semi-supervised regression loss and (optionally) supervised loss for aggregations.
* **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`) — Prediction scores of the cell selection head, for every token.
* **logits\_aggregation** (`torch.FloatTensor`, *optional*, of shape `(batch_size, num_aggregation_labels)`) — Prediction scores of the aggregation head, for every aggregation operator.
* **hidden\_states** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [TapasForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/tapas#transformers.TapasForQuestionAnswering) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> from transformers import AutoTokenizer, TapasForQuestionAnswering
>>> import pandas as pd

>>> tokenizer = AutoTokenizer.from_pretrained("google/tapas-base-finetuned-wtq")
>>> model = TapasForQuestionAnswering.from_pretrained("google/tapas-base-finetuned-wtq")

>>> data = {
...     "Actors": ["Brad Pitt", "Leonardo Di Caprio", "George Clooney"],
...     "Age": ["56", "45", "59"],
...     "Number of movies": ["87", "53", "69"],
... }
>>> table = pd.DataFrame.from_dict(data)
>>> queries = ["How many movies has George Clooney played in?", "How old is Brad Pitt?"]

>>> inputs = tokenizer(table=table, queries=queries, padding="max_length", return_tensors="pt")
>>> outputs = model(**inputs)

>>> logits = outputs.logits
>>> logits_aggregation = outputs.logits_aggregation
```

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/tapas.md)
