# Hyperparameter search

Hyperparameter search discovers an optimal set of hyperparameters that produces the best model performance. [Trainer](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Trainer) supports several hyperparameter search backends - [Optuna](https://optuna.readthedocs.io/en/stable/index.html), [SigOpt](https://docs.sigopt.com/), [Weights & Biases](https://docs.wandb.ai/), [Ray Tune](https://docs.ray.io/en/latest/tune/index.html) - through [hyperparameter\_search()](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Trainer.hyperparameter_search) to optimize an objective or even multiple objectives.

This guide will go over how to set up a hyperparameter search for each of the backends.

> [!WARNING][SigOpt](<https://github.com/sigopt/sigopt-server>) is in public archive mode and is no longer actively maintained. Try using Optuna, Weights & Biases or Ray Tune instead.


```
pip install optuna/sigopt/wandb/ray[tune]
```

To use [hyperparameter\_search()](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Trainer.hyperparameter_search), you need to create a `model_init` function. This function includes basic model information (arguments and configuration) because it needs to be reinitialized for each search trial in the run.

The `model_init` function is incompatible with the [optimizers](./main_classes/trainer#transformers.Trainer.optimizers) parameter. Subclass [Trainer](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Trainer) and override the [create\_optimizer\_and\_scheduler()](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Trainer.create_optimizer_and_scheduler) method to create a custom optimizer and scheduler.

An example `model_init` function is shown below.


```
def model_init(trial):
    return AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=True if model_args.use_auth_token else None,
    )
```

Pass `model_init` to [Trainer](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Trainer) along with everything else you need for training. Then you can call [hyperparameter\_search()](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Trainer.hyperparameter_search) to start the search.

[hyperparameter\_search()](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Trainer.hyperparameter_search) accepts a [direction](./main_classes/trainer#transformers.Trainer.hyperparameter_search.direction) parameter to specify whether to minimize, maximize, or minimize and maximize multiple objectives. You’ll also need to set the [backend](./main_classes/trainer#transformers.Trainer.hyperparameter_search.backend) you’re using, an [object](./main_classes/trainer#transformers.Trainer.hyperparameter_search.hp_space) containing the hyperparameters to optimize for, the [number of trials](./main_classes/trainer#transformers.Trainer.hyperparameter_search.n_trials) to run, and a [compute\_objective](./main_classes/trainer#transformers.Trainer.hyperparameter_search.compute_objective) to return the objective values.

If [compute\_objective](./main_classes/trainer#transformers.Trainer.hyperparameter_search.compute_objective) isn’t defined, the default [compute\_objective](./main_classes/trainer#transformers.Trainer.hyperparameter_search.compute_objective) is called which is the sum of an evaluation metric like F1.


```
from transformers import Trainer

trainer = Trainer(
    model=None,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
    processing_class=tokenizer,
    model_init=model_init,
    data_collator=data_collator,
)
trainer.hyperparameter_search(...)
```

The following examples demonstrate how to perform a hyperparameter search for the learning rate and training batch size using the different backends.

Optuna

Ray Tune

SigOpt

Weights & Biases

[Optuna](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/002_configurations.html#sphx-glr-tutorial-10-key-features-002-configurations-py) optimizes categories, integers, and floats.


```
def optuna_hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [16, 32, 64, 128]),
    }

best_trials = trainer.hyperparameter_search(
    direction=["minimize", "maximize"],
    backend="optuna",
    hp_space=optuna_hp_space,
    n_trials=20,
    compute_objective=compute_objective,
)
```

## Distributed Data Parallel

[Trainer](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Trainer) only supports hyperparameter search for distributed data parallel (DDP) on the Optuna and SigOpt backends. Only the rank-zero process is used to generate the search trial, and the resulting parameters are passed along to the other ranks.

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/hpo_train.md)
