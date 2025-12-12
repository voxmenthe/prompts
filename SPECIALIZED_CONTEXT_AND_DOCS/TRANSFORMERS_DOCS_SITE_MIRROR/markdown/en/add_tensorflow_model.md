# How to convert a 🤗 Transformers model to TensorFlow?

Having multiple frameworks available to use with 🤗 Transformers gives you flexibility to play their strengths when
designing your application, but it implies that compatibility must be added on a per-model basis. The good news is that
adding TensorFlow compatibility to an existing model is simpler than [adding a new model from scratch](add_new_model)!
Whether you wish to have a deeper understanding of large TensorFlow models, make a major open-source contribution, or
enable TensorFlow for your model of choice, this guide is for you.

This guide empowers you, a member of our community, to contribute TensorFlow model weights and/or
architectures to be used in 🤗 Transformers, with minimal supervision from the Hugging Face team. Writing a new model
is no small feat, but hopefully this guide will make it less of a rollercoaster 🎢 and more of a walk in the park 🚶.
Harnessing our collective experiences is absolutely critical to make this process increasingly easier, and thus we
highly encourage that you suggest improvements to this guide!

Before you dive deeper, it is recommended that you check the following resources if you’re new to 🤗 Transformers:

* [General overview of 🤗 Transformers](add_new_model#general-overview-of-transformers)
* [Hugging Face’s TensorFlow Philosophy](https://huggingface.co/blog/tensorflow-philosophy)

In the remainder of this guide, you will learn what’s needed to add a new TensorFlow model architecture, the
procedure to convert PyTorch into TensorFlow model weights, and how to efficiently debug mismatches across ML
frameworks. Let’s get started!

Are you unsure whether the model you wish to use already has a corresponding TensorFlow architecture?

Check the `model_type` field of the `config.json` of your model of choice
([example](https://huggingface.co/google-bert/bert-base-uncased/blob/main/config.json#L14)). If the corresponding model folder in
🤗 Transformers has a file whose name starts with “modeling\_tf”, it means that it has a corresponding TensorFlow
architecture ([example](https://github.com/huggingface/transformers/tree/main/src/transformers/models/bert)).

## Step-by-step guide to add TensorFlow model architecture code

There are many ways to design a large model architecture, and multiple ways of implementing said design. However,
you might recall from our [general overview of 🤗 Transformers](add_new_model#general-overview-of-transformers)
that we are an opinionated bunch - the ease of use of 🤗 Transformers relies on consistent design choices. From
experience, we can tell you a few important things about adding TensorFlow models:

* Don’t reinvent the wheel! More often than not, there are at least two reference implementations you should check: the
  PyTorch equivalent of the model you are implementing and other TensorFlow models for the same class of problems.
* Great model implementations survive the test of time. This doesn’t happen because the code is pretty, but rather
  because the code is clear, easy to debug and build upon. If you make the life of the maintainers easy with your
  TensorFlow implementation, by replicating the same patterns as in other TensorFlow models and minimizing the mismatch
  to the PyTorch implementation, you ensure your contribution will be long lived.
* Ask for help when you’re stuck! The 🤗 Transformers team is here to help, and we’ve probably found solutions to the same
  problems you’re facing.

Here’s an overview of the steps needed to add a TensorFlow model architecture:

1. Select the model you wish to convert
2. Prepare transformers dev environment
3. (Optional) Understand theoretical aspects and the existing implementation
4. Implement the model architecture
5. Implement model tests
6. Submit the pull request
7. (Optional) Build demos and share with the world

### 1.-3. Prepare your model contribution

**1. Select the model you wish to convert**

Let’s start off with the basics: the first thing you need to know is the architecture you want to convert. If you
don’t have your eyes set on a specific architecture, asking the 🤗 Transformers team for suggestions is a great way to
maximize your impact - we will guide you towards the most prominent architectures that are missing on the TensorFlow
side. If the specific model you want to use with TensorFlow already has a TensorFlow architecture implementation in
🤗 Transformers but is lacking weights, feel free to jump straight into the
[weight conversion section](#adding-tensorflow-weights-to--hub)
of this page.

For simplicity, the remainder of this guide assumes you’ve decided to contribute with the TensorFlow version of
*BrandNewBert* (the same example as in the [guide](add_new_model) to add a new model from scratch).

Before starting the work on a TensorFlow model architecture, double-check that there is no ongoing effort to do so.
You can search for `BrandNewBert` on the
[pull request GitHub page](https://github.com/huggingface/transformers/pulls?q=is%3Apr) to confirm that there is no
TensorFlow-related pull request.

**2. Prepare transformers dev environment**

Having selected the model architecture, open a draft PR to signal your intention to work on it. Follow the
instructions below to set up your environment and open a draft PR.

1. Fork the [repository](https://github.com/huggingface/transformers) by clicking on the ‘Fork’ button on the
   repository’s page. This creates a copy of the code under your GitHub user account.
2. Clone your `transformers` fork to your local disk, and add the base repository as a remote:

   ```
   git clone https://github.com/[your Github handle]/transformers.git
   cd transformers
   git remote add upstream https://github.com/huggingface/transformers.git
   ```
3. Set up a development environment, for instance by running the following commands:

   ```
   python -m venv .env
   source .env/bin/activate
   pip install -e ".[dev]"
   ```

   Depending on your OS, and since the number of optional dependencies of Transformers is growing, you might get a
   failure with this command. If that’s the case make sure to install TensorFlow then do:

   ```
   pip install -e ".[quality]"
   ```

   **Note:** You don’t need to have CUDA installed. Making the new model work on CPU is sufficient.
4. Create a branch with a descriptive name from your main branch:

   ```
   git checkout -b add_tf_brand_new_bert
   ```
5. Fetch and rebase to current main:

   ```
   git fetch upstream
   git rebase upstream/main
   ```
6. Add an empty `.py` file in `transformers/src/models/brandnewbert/` named `modeling_tf_brandnewbert.py`. This will
   be your TensorFlow model file.
7. Push the changes to your account using:

   ```
   git add .
   git commit -m "initial commit"
   git push -u origin add_tf_brand_new_bert
   ```
8. Once you are satisfied, go to the webpage of your fork on GitHub. Click on “Pull request”. Make sure to add the
   GitHub handle of some members of the Hugging Face team as reviewers, so that the Hugging Face team gets notified for
   future changes.
9. Change the PR into a draft by clicking on “Convert to draft” on the right of the GitHub pull request web page.

Now you have set up a development environment to port *BrandNewBert* to TensorFlow in 🤗 Transformers.

**3. (Optional) Understand theoretical aspects and the existing implementation**

You should take some time to read *BrandNewBert’s* paper, if such descriptive work exists. There might be large
sections of the paper that are difficult to understand. If this is the case, this is fine - don’t worry! The goal is
not to get a deep theoretical understanding of the paper, but to extract the necessary information required to
effectively re-implement the model in 🤗 Transformers using TensorFlow. That being said, you don’t have to spend too
much time on the theoretical aspects, but rather focus on the practical ones, namely the existing model documentation
page (e.g. [model docs for BERT](model_doc/bert)).

After you’ve grasped the basics of the models you are about to implement, it’s important to understand the existing
implementation. This is a great chance to confirm that a working implementation matches your expectations for the
model, as well as to foresee technical challenges on the TensorFlow side.

It’s perfectly natural that you feel overwhelmed with the amount of information that you’ve just absorbed. It is
definitely not a requirement that you understand all facets of the model at this stage. Nevertheless, we highly
encourage you to clear any pressing questions in our [forum](https://discuss.huggingface.co/).

### 4. Model implementation

Now it’s time to finally start coding. Our suggested starting point is the PyTorch file itself: copy the contents of
`modeling_brand_new_bert.py` inside `src/transformers/models/brand_new_bert/` into
`modeling_tf_brand_new_bert.py`. The goal of this section is to modify the file and update the import structure of
🤗 Transformers such that you can import `TFBrandNewBert` and
`TFBrandNewBert.from_pretrained(model_repo, from_pt=True)` successfully loads a working TensorFlow *BrandNewBert* model.

Sadly, there is no prescription to convert a PyTorch model into TensorFlow. You can, however, follow our selection of
tips to make the process as smooth as possible:

* Prepend `TF` to the name of all classes (e.g. `BrandNewBert` becomes `TFBrandNewBert`).
* Most PyTorch operations have a direct TensorFlow replacement. For example, `torch.nn.Linear` corresponds to
  `tf.keras.layers.Dense`, `torch.nn.Dropout` corresponds to `tf.keras.layers.Dropout`, etc. If you’re not sure
  about a specific operation, you can use the [TensorFlow documentation](https://www.tensorflow.org/api_docs/python/tf)
  or the [PyTorch documentation](https://pytorch.org/docs/stable/).
* Look for patterns in the 🤗 Transformers codebase. If you come across a certain operation that doesn’t have a direct
  replacement, the odds are that someone else already had the same problem.
* By default, keep the same variable names and structure as in PyTorch. This will make it easier to debug, track
  issues, and add fixes down the line.
* Some layers have different default values in each framework. A notable example is the batch normalization layer’s
  epsilon (`1e-5` in [PyTorch](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html#torch.nn.BatchNorm2d)
  and `1e-3` in [TensorFlow](https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization)).
  Double-check the documentation!
* PyTorch’s `nn.Parameter` variables typically need to be initialized within TF Layer’s `build()`. See the following
  example: [PyTorch](https://github.com/huggingface/transformers/blob/655f72a6896c0533b1bdee519ed65a059c2425ac/src/transformers/models/vit_mae/modeling_vit_mae.py#L212) /
  [TensorFlow](https://github.com/huggingface/transformers/blob/655f72a6896c0533b1bdee519ed65a059c2425ac/src/transformers/models/vit_mae/modeling_tf_vit_mae.py#L220)
* If the PyTorch model has a `#copied from ...` on top of a function, the odds are that your TensorFlow model can also
  borrow that function from the architecture it was copied from, assuming it has a TensorFlow architecture.
* Assigning the `name` attribute correctly in TensorFlow functions is critical to do the `from_pt=True` weight
  cross-loading. `name` is almost always the name of the corresponding variable in the PyTorch code. If `name` is not
  properly set, you will see it in the error message when loading the model weights.
* The logic of the base model class, `BrandNewBertModel`, will actually reside in `TFBrandNewBertMainLayer`, a Keras
  layer subclass ([example](https://github.com/huggingface/transformers/blob/4fd32a1f499e45f009c2c0dea4d81c321cba7e02/src/transformers/models/bert/modeling_tf_bert.py#L719)).
  `TFBrandNewBertModel` will simply be a wrapper around this layer.
* Keras models need to be built in order to load pretrained weights. For that reason, `TFBrandNewBertPreTrainedModel`
  will need to hold an example of inputs to the model, the `dummy_inputs`
  ([example](https://github.com/huggingface/transformers/blob/4fd32a1f499e45f009c2c0dea4d81c321cba7e02/src/transformers/models/bert/modeling_tf_bert.py#L916)).
* If you get stuck, ask for help - we’re here to help you! 🤗

In addition to the model file itself, you will also need to add the pointers to the model classes and related
documentation pages. You can complete this part entirely following the patterns in other PRs
([example](https://github.com/huggingface/transformers/pull/18020/files)). Here’s a list of the needed manual
changes:

* Include all public classes of *BrandNewBert* in `src/transformers/__init__.py`
* Add *BrandNewBert* classes to the corresponding Auto classes in `src/transformers/models/auto/modeling_tf_auto.py`
* Add the lazy loading classes related to *BrandNewBert* in `src/transformers/utils/dummy_tf_objects.py`
* Update the import structures for the public classes in `src/transformers/models/brand_new_bert/__init__.py`
* Add the documentation pointers to the public methods of *BrandNewBert* in `docs/source/en/model_doc/brand_new_bert.md`
* Add yourself to the list of contributors to *BrandNewBert* in `docs/source/en/model_doc/brand_new_bert.md`
* Finally, add a green tick ✅ to the TensorFlow column of *BrandNewBert* in `docs/source/en/index.md`

When you’re happy with your implementation, run the following checklist to confirm that your model architecture is
ready:

1. All layers that behave differently at train time (e.g. Dropout) are called with a `training` argument, which is
   propagated all the way from the top-level classes
2. You have used `#copied from ...` whenever possible
3. `TFBrandNewBertMainLayer` and all classes that use it have their `call` function decorated with `@unpack_inputs`
4. `TFBrandNewBertMainLayer` is decorated with `@keras_serializable`
5. A TensorFlow model can be loaded from PyTorch weights using `TFBrandNewBert.from_pretrained(model_repo, from_pt=True)`
6. You can call the TensorFlow model using the expected input format

### 5. Add model tests

Hurray, you’ve implemented a TensorFlow model! Now it’s time to add tests to make sure that your model behaves as
expected. As in the previous section, we suggest you start by copying the `test_modeling_brand_new_bert.py` file in
`tests/models/brand_new_bert/` into `test_modeling_tf_brand_new_bert.py`, and continue by making the necessary
TensorFlow replacements. For now, in all `.from_pretrained()` calls, you should use the `from_pt=True` flag to load
the existing PyTorch weights.

After you’re done, it’s time for the moment of truth: run the tests! 😬

```
NVIDIA_TF32_OVERRIDE=0 RUN_SLOW=1 RUN_PT_TF_CROSS_TESTS=1 \
py.test -vv tests/models/brand_new_bert/test_modeling_tf_brand_new_bert.py
```

The most likely outcome is that you’ll see a bunch of errors. Don’t worry, this is expected! Debugging ML models is
notoriously hard, and the key ingredient to success is patience (and `breakpoint()`). In our experience, the hardest
problems arise from subtle mismatches between ML frameworks, for which we have a few pointers at the end of this guide.
In other cases, a general test might not be directly applicable to your model, in which case we suggest an override
at the model test class level. Regardless of the issue, don’t hesitate to ask for help in your draft pull request if
you’re stuck.

When all tests pass, congratulations, your model is nearly ready to be added to the 🤗 Transformers library! 🎉

### 6.-7. Ensure everyone can use your model

**6. Submit the pull request**

Once you’re done with the implementation and the tests, it’s time to submit a pull request. Before pushing your code,
run our code formatting utility, `make fixup` 🪄. This will automatically fix any formatting issues, which would cause
our automatic checks to fail.

It’s now time to convert your draft pull request into a real pull request. To do so, click on the “Ready for
review” button and add Joao (`@gante`) and Matt (`@Rocketknight1`) as reviewers. A model pull request will need
at least 3 reviewers, but they will take care of finding appropriate additional reviewers for your model.

After all reviewers are happy with the state of your PR, the final action point is to remove the `from_pt=True` flag in
`.from_pretrained()` calls. Since there are no TensorFlow weights, you will have to add them! Check the section
below for instructions on how to do it.

Finally, when the TensorFlow weights get merged, you have at least 3 reviewer approvals, and all CI checks are
green, double-check the tests locally one last time

```
NVIDIA_TF32_OVERRIDE=0 RUN_SLOW=1 RUN_PT_TF_CROSS_TESTS=1 \
py.test -vv tests/models/brand_new_bert/test_modeling_tf_brand_new_bert.py
```

and we will merge your PR! Congratulations on the milestone 🎉

**7. (Optional) Build demos and share with the world**

One of the hardest parts about open-source is discovery. How can the other users learn about the existence of your
fabulous TensorFlow contribution? With proper communication, of course! 📣

There are two main ways to share your model with the community:

* Build demos. These include Gradio demos, notebooks, and other fun ways to show off your model. We highly
  encourage you to add a notebook to our [community-driven demos](https://huggingface.co/docs/transformers/community).
* Share stories on social media like Twitter and LinkedIn. You should be proud of your work and share
  your achievement with the community - your model can now be used by thousands of engineers and researchers around
  the world 🌍! We will be happy to retweet your posts and help you share your work with the community.

## Adding TensorFlow weights to 🤗 Hub

Assuming that the TensorFlow model architecture is available in 🤗 Transformers, converting PyTorch weights into
TensorFlow weights is a breeze!

Here’s how to do it:

1. Make sure you are logged into your Hugging Face account in your terminal. You can log in using the command
   `huggingface-cli login` (you can find your access tokens [here](https://huggingface.co/settings/tokens))
2. Run `transformers-cli pt-to-tf --model-name foo/bar`, where `foo/bar` is the name of the model repository
   containing the PyTorch weights you want to convert
3. Tag `@joaogante` and `@Rocketknight1` in the 🤗 Hub PR the command above has just created

That’s it! 🎉

## Debugging mismatches across ML frameworks 🐛

At some point, when adding a new architecture or when creating TensorFlow weights for an existing architecture, you
might come across errors complaining about mismatches between PyTorch and TensorFlow. You might even decide to open the
model architecture code for the two frameworks, and find that they look identical. What’s going on? 🤔

First of all, let’s talk about why understanding these mismatches matters. Many community members will use 🤗
Transformers models out of the box, and trust that our models behave as expected. When there is a large mismatch
between the two frameworks, it implies that the model is not following the reference implementation for at least one
of the frameworks. This might lead to silent failures, in which the model runs but has poor performance. This is
arguably worse than a model that fails to run at all! To that end, we aim at having a framework mismatch smaller than
`1e-5` at all stages of the model.

As in other numerical problems, the devil is in the details. And as in any detail-oriented craft, the secret
ingredient here is patience. Here is our suggested workflow for when you come across this type of issues:

1. Locate the source of mismatches. The model you’re converting probably has near identical inner variables up to a
   certain point. Place `breakpoint()` statements in the two frameworks’ architectures, and compare the values of the
   numerical variables in a top-down fashion until you find the source of the problems.
2. Now that you’ve pinpointed the source of the issue, get in touch with the 🤗 Transformers team. It is possible
   that we’ve seen a similar problem before and can promptly provide a solution. As a fallback, scan popular pages
   like StackOverflow and GitHub issues.
3. If there is no solution in sight, it means you’ll have to go deeper. The good news is that you’ve located the
   issue, so you can focus on the problematic instruction, abstracting away the rest of the model! The bad news is
   that you’ll have to venture into the source implementation of said instruction. In some cases, you might find an
   issue with a reference implementation - don’t abstain from opening an issue in the upstream repository.

In some cases, in discussion with the 🤗 Transformers team, we might find that fixing the mismatch is infeasible.
When the mismatch is very small in the output layers of the model (but potentially large in the hidden states), we
might decide to ignore it in favor of distributing the model. The `pt-to-tf` CLI mentioned above has a `--max-error`
flag to override the error message at weight conversion time.

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/add_tensorflow_model.md)
