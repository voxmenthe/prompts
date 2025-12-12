# Video classification

![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)

![Open In Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)

Video classification is the task of assigning a label or class to an entire video. Videos are expected to have only one class for each video. Video classification models take a video as input and return a prediction about which class the video belongs to. These models can be used to categorize what a video is all about. A real-world application of video classification is action / activity recognition, which is useful for fitness applications. It is also helpful for vision-impaired individuals, especially when they are commuting.

This guide will show you how to:

1. Fine-tune [VideoMAE](https://huggingface.co/docs/transformers/main/en/model_doc/videomae) on a subset of the [UCF101](https://www.crcv.ucf.edu/data/UCF101.php) dataset.
2. Use your fine-tuned model for inference.

To see all architectures and checkpoints compatible with this task, we recommend checking the [task-page](https://huggingface.co/tasks/video-classification).

Before you begin, make sure you have all the necessary libraries installed:


```
pip install -q pytorchvideo transformers evaluate
```

You will use [PyTorchVideo](https://pytorchvideo.org/) (dubbed `pytorchvideo`) to process and prepare the videos.

We encourage you to log in to your Hugging Face account so you can upload and share your model with the community. When prompted, enter your token to log in:


```
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

## Load UCF101 dataset

Start by loading a subset of the [UCF-101 dataset](https://www.crcv.ucf.edu/data/UCF101.php). This will give you a chance to experiment and make sure everything works before spending more time training on the full dataset.


```
>>> from huggingface_hub import hf_hub_download

>>> hf_dataset_identifier = "sayakpaul/ucf101-subset"
>>> filename = "UCF101_subset.tar.gz"
>>> file_path = hf_hub_download(repo_id=hf_dataset_identifier, filename=filename, repo_type="dataset")
```

After the subset has been downloaded, you need to extract the compressed archive:


```
>>> import tarfile

>>> with tarfile.open(file_path) as t:
...      t.extractall(".")
```

At a high level, the dataset is organized like so:


```
UCF101_subset/
    train/
        BandMarching/
            video_1.mp4
            video_2.mp4
            ...
        Archery
            video_1.mp4
            video_2.mp4
            ...
        ...
    val/
        BandMarching/
            video_1.mp4
            video_2.mp4
            ...
        Archery
            video_1.mp4
            video_2.mp4
            ...
        ...
    test/
        BandMarching/
            video_1.mp4
            video_2.mp4
            ...
        Archery
            video_1.mp4
            video_2.mp4
            ...
        ...
```

You can then count the number of total videos.


```
>>> import pathlib
>>> dataset_root_path = "UCF101_subset"
>>> dataset_root_path = pathlib.Path(dataset_root_path)
```


```
>>> video_count_train = len(list(dataset_root_path.glob("train/*/*.avi")))
>>> video_count_val = len(list(dataset_root_path.glob("val/*/*.avi")))
>>> video_count_test = len(list(dataset_root_path.glob("test/*/*.avi")))
>>> video_total = video_count_train + video_count_val + video_count_test
>>> print(f"Total videos: {video_total}")
```


```
>>> all_video_file_paths = (
...     list(dataset_root_path.glob("train/*/*.avi"))
...     + list(dataset_root_path.glob("val/*/*.avi"))
...     + list(dataset_root_path.glob("test/*/*.avi"))
...  )
>>> all_video_file_paths[:5]
```

The (`sorted`) video paths appear like so:


```
...
'UCF101_subset/train/ApplyEyeMakeup/v_ApplyEyeMakeup_g07_c04.avi',
'UCF101_subset/train/ApplyEyeMakeup/v_ApplyEyeMakeup_g07_c06.avi',
'UCF101_subset/train/ApplyEyeMakeup/v_ApplyEyeMakeup_g08_c01.avi',
'UCF101_subset/train/ApplyEyeMakeup/v_ApplyEyeMakeup_g09_c02.avi',
'UCF101_subset/train/ApplyEyeMakeup/v_ApplyEyeMakeup_g09_c06.avi'
...
```

You will notice that there are video clips belonging to the same group / scene where group is denoted by `g` in the video file paths. `v_ApplyEyeMakeup_g07_c04.avi` and `v_ApplyEyeMakeup_g07_c06.avi`, for example.

For the validation and evaluation splits, you wouldn’t want to have video clips from the same group / scene to prevent [data leakage](https://www.kaggle.com/code/alexisbcook/data-leakage). The subset that you are using in this tutorial takes this information into account.

Next up, you will derive the set of labels present in the dataset. Also, create two dictionaries that’ll be helpful when initializing the model:

* `label2id`: maps the class names to integers.
* `id2label`: maps the integers to class names.


```
>>> class_labels = sorted({str(path).split("/")[2] for path in all_video_file_paths})
>>> label2id = {label: i for i, label in enumerate(class_labels)}
>>> id2label = {i: label for label, i in label2id.items()}

>>> print(f"Unique classes: {list(label2id.keys())}.")

# Unique classes: ['ApplyEyeMakeup', 'ApplyLipstick', 'Archery', 'BabyCrawling', 'BalanceBeam', 'BandMarching', 'BaseballPitch', 'Basketball', 'BasketballDunk', 'BenchPress'].
```

There are 10 unique classes. For each class, there are 30 videos in the training set.

## Load a model to fine-tune

Instantiate a video classification model from a pretrained checkpoint and its associated image processor. The model’s encoder comes with pre-trained parameters, and the classification head is randomly initialized. The image processor will come in handy when writing the preprocessing pipeline for our dataset.


```
>>> from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification

>>> model_ckpt = "MCG-NJU/videomae-base"
>>> image_processor = VideoMAEImageProcessor.from_pretrained(model_ckpt)
>>> model = VideoMAEForVideoClassification.from_pretrained(
...     model_ckpt,
...     label2id=label2id,
...     id2label=id2label,
...     ignore_mismatched_sizes=True,  # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
... )
```

While the model is loading, you might notice the following warning:


```
Some weights of the model checkpoint at MCG-NJU/videomae-base were not used when initializing VideoMAEForVideoClassification: [..., 'decoder.decoder_layers.1.attention.output.dense.bias', 'decoder.decoder_layers.2.attention.attention.key.weight']
- This IS expected if you are initializing VideoMAEForVideoClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing VideoMAEForVideoClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
Some weights of VideoMAEForVideoClassification were not initialized from the model checkpoint at MCG-NJU/videomae-base and are newly initialized: ['classifier.bias', 'classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
```

The warning is telling us we are throwing away some weights (e.g. the weights and bias of the `classifier` layer) and randomly initializing some others (the weights and bias of a new `classifier` layer). This is expected in this case, because we are adding a new head for which we don’t have pretrained weights, so the library warns us we should fine-tune this model before using it for inference, which is exactly what we are going to do.

**Note** that [this checkpoint](https://huggingface.co/MCG-NJU/videomae-base-finetuned-kinetics) leads to better performance on this task as the checkpoint was obtained fine-tuning on a similar downstream task having considerable domain overlap. You can check out [this checkpoint](https://huggingface.co/sayakpaul/videomae-base-finetuned-kinetics-finetuned-ucf101-subset) which was obtained by fine-tuning `MCG-NJU/videomae-base-finetuned-kinetics`.

## Prepare the datasets for training

For preprocessing the videos, you will leverage the [PyTorchVideo library](https://pytorchvideo.org/). Start by importing the dependencies we need.


```
>>> import pytorchvideo.data

>>> from pytorchvideo.transforms import (
...     ApplyTransformToKey,
...     Normalize,
...     RandomShortSideScale,
...     RemoveKey,
...     ShortSideScale,
...     UniformTemporalSubsample,
... )

>>> from torchvision.transforms import (
...     Compose,
...     Lambda,
...     RandomCrop,
...     RandomHorizontalFlip,
...     Resize,
... )
```

For the training dataset transformations, use a combination of uniform temporal subsampling, pixel normalization, random cropping, and random horizontal flipping. For the validation and evaluation dataset transformations, keep the same transformation chain except for random cropping and horizontal flipping. To learn more about the details of these transformations check out the [official documentation of PyTorchVideo](https://pytorchvideo.org).

Use the `image_processor` associated with the pre-trained model to obtain the following information:

* Image mean and standard deviation with which the video frame pixels will be normalized.
* Spatial resolution to which the video frames will be resized.

Start by defining some constants.


```
>>> mean = image_processor.image_mean
>>> std = image_processor.image_std
>>> if "shortest_edge" in image_processor.size:
...     height = width = image_processor.size["shortest_edge"]
>>> else:
...     height = image_processor.size["height"]
...     width = image_processor.size["width"]
>>> resize_to = (height, width)

>>> num_frames_to_sample = model.config.num_frames
>>> sample_rate = 4
>>> fps = 30
>>> clip_duration = num_frames_to_sample * sample_rate / fps
```

Now, define the dataset-specific transformations and the datasets respectively. Starting with the training set:


```
>>> train_transform = Compose(
...     [
...         ApplyTransformToKey(
...             key="video",
...             transform=Compose(
...                 [
...                     UniformTemporalSubsample(num_frames_to_sample),
...                     Lambda(lambda x: x / 255.0),
...                     Normalize(mean, std),
...                     RandomShortSideScale(min_size=256, max_size=320),
...                     RandomCrop(resize_to),
...                     RandomHorizontalFlip(p=0.5),
...                 ]
...             ),
...         ),
...     ]
... )

>>> train_dataset = pytorchvideo.data.Ucf101(
...     data_path=os.path.join(dataset_root_path, "train"),
...     clip_sampler=pytorchvideo.data.make_clip_sampler("random", clip_duration),
...     decode_audio=False,
...     transform=train_transform,
... )
```

The same sequence of workflow can be applied to the validation and evaluation sets:


```
>>> val_transform = Compose(
...     [
...         ApplyTransformToKey(
...             key="video",
...             transform=Compose(
...                 [
...                     UniformTemporalSubsample(num_frames_to_sample),
...                     Lambda(lambda x: x / 255.0),
...                     Normalize(mean, std),
...                     Resize(resize_to),
...                 ]
...             ),
...         ),
...     ]
... )

>>> val_dataset = pytorchvideo.data.Ucf101(
...     data_path=os.path.join(dataset_root_path, "val"),
...     clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", clip_duration),
...     decode_audio=False,
...     transform=val_transform,
... )

>>> test_dataset = pytorchvideo.data.Ucf101(
...     data_path=os.path.join(dataset_root_path, "test"),
...     clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", clip_duration),
...     decode_audio=False,
...     transform=val_transform,
... )
```

**Note**: The above dataset pipelines are taken from the [official PyTorchVideo example](https://pytorchvideo.org/docs/tutorial_classification#dataset). We’re using the [`pytorchvideo.data.Ucf101()`](https://pytorchvideo.readthedocs.io/en/latest/api/data/data.html#pytorchvideo.data.Ucf101) function because it’s tailored for the UCF-101 dataset. Under the hood, it returns a [`pytorchvideo.data.labeled_video_dataset.LabeledVideoDataset`](https://pytorchvideo.readthedocs.io/en/latest/api/data/data.html#pytorchvideo.data.LabeledVideoDataset) object. `LabeledVideoDataset` class is the base class for all things video in the PyTorchVideo dataset. So, if you want to use a custom dataset not supported off-the-shelf by PyTorchVideo, you can extend the `LabeledVideoDataset` class accordingly. Refer to the `data` API [documentation to](https://pytorchvideo.readthedocs.io/en/latest/api/data/data.html) learn more. Also, if your dataset follows a similar structure (as shown above), then using the `pytorchvideo.data.Ucf101()` should work just fine.

You can access the `num_videos` argument to know the number of videos in the dataset.


```
>>> print(train_dataset.num_videos, val_dataset.num_videos, test_dataset.num_videos)
# (300, 30, 75)
```

## Visualize the preprocessed video for better debugging


```
>>> import imageio
>>> import numpy as np
>>> from IPython.display import Image

>>> def unnormalize_img(img):
...     """Un-normalizes the image pixels."""
...     img = (img * std) + mean
...     img = (img * 255).astype("uint8")
...     return img.clip(0, 255)

>>> def create_gif(video_tensor, filename="sample.gif"):
...     """Prepares a GIF from a video tensor.
...
...     The video tensor is expected to have the following shape:
...     (num_frames, num_channels, height, width).
...     """
...     frames = []
...     for video_frame in video_tensor:
...         frame_unnormalized = unnormalize_img(video_frame.permute(1, 2, 0).numpy())
...         frames.append(frame_unnormalized)
...     kargs = {"duration": 0.25}
...     imageio.mimsave(filename, frames, "GIF", **kargs)
...     return filename

>>> def display_gif(video_tensor, gif_name="sample.gif"):
...     """Prepares and displays a GIF from a video tensor."""
...     video_tensor = video_tensor.permute(1, 0, 2, 3)
...     gif_filename = create_gif(video_tensor, gif_name)
...     return Image(filename=gif_filename)

>>> sample_video = next(iter(train_dataset))
>>> video_tensor = sample_video["video"]
>>> display_gif(video_tensor)
```

![Person playing basketball](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/sample_gif.gif)

## Train the model

Leverage [`Trainer`](https://huggingface.co/docs/transformers/main_classes/trainer) from 🤗 Transformers for training the model. To instantiate a `Trainer`, you need to define the training configuration and an evaluation metric. The most important is the [`TrainingArguments`](https://huggingface.co/transformers/main_classes/trainer.html#transformers.TrainingArguments), which is a class that contains all the attributes to configure the training. It requires an output folder name, which will be used to save the checkpoints of the model. It also helps sync all the information in the model repository on 🤗 Hub.

Most of the training arguments are self-explanatory, but one that is quite important here is `remove_unused_columns=False`. This one will drop any features not used by the model’s call function. By default it’s `True` because usually it’s ideal to drop unused feature columns, making it easier to unpack inputs into the model’s call function. But, in this case, you need the unused features (‘video’ in particular) in order to create `pixel_values` (which is a mandatory key our model expects in its inputs).


```
>>> from transformers import TrainingArguments, Trainer

>>> model_name = model_ckpt.split("/")[-1]
>>> new_model_name = f"{model_name}-finetuned-ucf101-subset"
>>> num_epochs = 4

>>> args = TrainingArguments(
...     new_model_name,
...     remove_unused_columns=False,
...     eval_strategy="epoch",
...     save_strategy="epoch",
...     learning_rate=5e-5,
...     per_device_train_batch_size=batch_size,
...     per_device_eval_batch_size=batch_size,
...     warmup_ratio=0.1,
...     logging_steps=10,
...     load_best_model_at_end=True,
...     metric_for_best_model="accuracy",
...     push_to_hub=True,
...     max_steps=(train_dataset.num_videos // batch_size) * num_epochs,
... )
```

The dataset returned by `pytorchvideo.data.Ucf101()` doesn’t implement the `__len__` method. As such, we must define `max_steps` when instantiating `TrainingArguments`.

Next, you need to define a function to compute the metrics from the predictions, which will use the `metric` you’ll load now. The only preprocessing you have to do is to take the argmax of our predicted logits:


```
import evaluate

metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)
```

**A note on evaluation**:

In the [VideoMAE paper](https://huggingface.co/papers/2203.12602), the authors use the following evaluation strategy. They evaluate the model on several clips from test videos and apply different crops to those clips and report the aggregate score. However, in the interest of simplicity and brevity, we don’t consider that in this tutorial.

Also, define a `collate_fn`, which will be used to batch examples together. Each batch consists of 2 keys, namely `pixel_values` and `labels`.


```
>>> def collate_fn(examples):
...     # permute to (num_frames, num_channels, height, width)
...     pixel_values = torch.stack(
...         [example["video"].permute(1, 0, 2, 3) for example in examples]
...     )
...     labels = torch.tensor([example["label"] for example in examples])
...     return {"pixel_values": pixel_values, "labels": labels}
```

Then you just pass all of this along with the datasets to `Trainer`:


```
>>> trainer = Trainer(
...     model,
...     args,
...     train_dataset=train_dataset,
...     eval_dataset=val_dataset,
...     processing_class=image_processor,
...     compute_metrics=compute_metrics,
...     data_collator=collate_fn,
... )
```

You might wonder why you passed along the `image_processor` as a tokenizer when you preprocessed the data already. This is only to make sure the image processor configuration file (stored as JSON) will also be uploaded to the repo on the Hub.

Now fine-tune our model by calling the `train` method:


```
>>> train_results = trainer.train()
```

Once training is completed, share your model to the Hub with the [push\_to\_hub()](/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Trainer.push_to_hub) method so everyone can use your model:


```
>>> trainer.push_to_hub()
```

## Inference

Great, now that you have fine-tuned a model, you can use it for inference!

Load a video for inference:


```
>>> sample_test_video = next(iter(test_dataset))
```

![Teams playing basketball](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/sample_gif_two.gif)

The simplest way to try out your fine-tuned model for inference is to use it in a [`pipeline`](https://huggingface.co/docs/transformers/main/en/main_classes/pipelines#transformers.VideoClassificationPipeline). Instantiate a `pipeline` for video classification with your model, and pass your video to it:


```
>>> from transformers import pipeline, infer_device

>>> video_cls = pipeline(model="my_awesome_video_cls_model")
>>> video_cls("https://huggingface.co/datasets/sayakpaul/ucf101-subset/resolve/main/v_BasketballDunk_g14_c06.avi")
[{'score': 0.9272987842559814, 'label': 'BasketballDunk'},
 {'score': 0.017777055501937866, 'label': 'BabyCrawling'},
 {'score': 0.01663011871278286, 'label': 'BalanceBeam'},
 {'score': 0.009560945443809032, 'label': 'BandMarching'},
 {'score': 0.0068979403004050255, 'label': 'BaseballPitch'}]
```

You can also manually replicate the results of the `pipeline` if you’d like.


```
>>> def run_inference(model, video):
...     # (num_frames, num_channels, height, width)
...     perumuted_sample_test_video = video.permute(1, 0, 2, 3)
...     inputs = {
...         "pixel_values": perumuted_sample_test_video.unsqueeze(0),
...         "labels": torch.tensor(
...             [sample_test_video["label"]]
...         ),  # this can be skipped if you don't have labels available.
...     }

...     device = torch.device(infer_device())
...     inputs = {k: v.to(device) for k, v in inputs.items()}
...     model = model.to(device)

...     # forward pass
...     with torch.no_grad():
...         outputs = model(**inputs)
...         logits = outputs.logits

...     return logits
```

Now, pass your input to the model and return the `logits`:


```
>>> logits = run_inference(trained_model, sample_test_video["video"])
```

Decoding the `logits`, we get:


```
>>> predicted_class_idx = logits.argmax(-1).item()
>>> print("Predicted class:", model.config.id2label[predicted_class_idx])
# Predicted class: BasketballDunk
```

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/tasks/video_classification.md)
