# Sharing

The Hugging Face [Hub](https://hf.co/models) is a platform for sharing, discovering, and consuming models of all different types and sizes. We highly recommend sharing your model on the Hub to push open-source machine learning forward for everyone!

This guide will show you how to share a model to the Hub from Transformers.

## Set up

To share a model to the Hub, you need a Hugging Face [account](https://hf.co/join). Create a [User Access Token](https://hf.co/docs/hub/security-tokens#user-access-tokens) (stored in the [cache](./installation#cache-directory) by default) and login to your account from either the command line or notebook.

```bash
hf auth login
```

```py
from huggingface_hub import notebook_login

notebook_login()
```

## Repository features

Each model repository features versioning, commit history, and diff visualization.

    

Versioning is based on [Git](https://git-scm.com/) and [Git Large File Storage (LFS)](https://git-lfs.github.com/), and it enables revisions, a way to specify a model version with a commit hash, tag or branch.

For example, use the `revision` parameter in [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) to load a specific model version from a commit hash.

```py
model = AutoModel.from_pretrained(
    "julien-c/EsperBERTo-small", revision="4c77982"
)
```

Model repositories also support [gating](https://hf.co/docs/hub/models-gated) to control who can access a model. Gating is common for allowing a select group of users to preview a research model before it's made public.

    

A model repository also includes an inference [widget](https://hf.co/docs/hub/models-widgets) for users to directly interact with a model on the Hub.

Check out the Hub [Models](https://hf.co/docs/hub/models) documentation to for more information.

## Uploading a model

There are several ways to upload a model to the Hub depending on your workflow preference. You can push a model with [Trainer](/docs/transformers/main/en/main_classes/trainer#transformers.Trainer), call [push_to_hub()](/docs/transformers/main/en/main_classes/model#transformers.utils.PushToHubMixin.push_to_hub) directly on a model, or use the Hub web interface.

### Trainer

[Trainer](/docs/transformers/main/en/main_classes/trainer#transformers.Trainer) can push a model directly to the Hub after training. Set `push_to_hub=True` in [TrainingArguments](/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments) and pass it to [Trainer](/docs/transformers/main/en/main_classes/trainer#transformers.Trainer). Once training is complete, call [push_to_hub()](/docs/transformers/main/en/main_classes/trainer#transformers.Trainer.push_to_hub) to upload the model.

[push_to_hub()](/docs/transformers/main/en/main_classes/trainer#transformers.Trainer.push_to_hub) automatically adds useful information like training hyperparameters and results to the model card.

```py
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(output_dir="my-awesome-model", push_to_hub=True)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)
trainer.push_to_hub()
```

### PushToHubMixin

The [PushToHubMixin](/docs/transformers/main/en/main_classes/model#transformers.utils.PushToHubMixin) provides functionality for pushing a model or tokenizer to the Hub.

Call [push_to_hub()](/docs/transformers/main/en/main_classes/model#transformers.utils.PushToHubMixin.push_to_hub) directly on a model to upload it to the Hub. It creates a repository under your namespace with the model name specified in [push_to_hub()](/docs/transformers/main/en/main_classes/model#transformers.utils.PushToHubMixin.push_to_hub).

```py
model.push_to_hub("my-awesome-model")
```

Other objects like a tokenizer are also pushed to the Hub in the same way.

```py
tokenizer.push_to_hub("my-awesome-model")
```

Your Hugging Face profile should now display the newly created model repository. Navigate to the **Files** tab to see all the uploaded files.

Refer to the [Upload files to the Hub](https://hf.co/docs/hub/how-to-upstream) guide for more information about pushing files to the Hub.

### Hub web interface

The Hub web interface is a no-code approach for uploading a model.

1. Create a new repository by selecting [**New Model**](https://huggingface.co/new).

    

Add some information about your model:

- Select the **owner** of the repository. This can be yourself or any of the organizations you belong to.
- Pick a name for your model, which will also be the repository name.
- Choose whether your model is public or private.
- Set the license usage.

2. Click on **Create model** to create the model repository.

3. Select the **Files** tab and click on the **Add file** button to drag-and-drop a file to your repository. Add a commit message and click on **Commit changes to main** to commit the file.

    

## Model card

[Model cards](https://hf.co/docs/hub/model-cards#model-cards) inform users about a models performance, limitations, potential biases, and ethical considerations. It is highly recommended to add a model card to your repository!

A model card is a `README.md` file in your repository. Add this file by:

- manually creating and uploading a `README.md` file
- clicking on the **Edit model card** button in the repository

Take a look at the Llama 3.1 [model card](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct) for an example of what to include on a model card.

Learn more about other model card metadata (carbon emissions, license, link to paper, etc.) available in the [Model Cards](https://hf.co/docs/hub/model-cards#model-cards) guide.
