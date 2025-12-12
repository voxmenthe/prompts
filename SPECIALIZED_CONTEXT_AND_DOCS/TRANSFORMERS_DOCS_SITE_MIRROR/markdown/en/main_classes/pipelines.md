# Pipelines

The pipelines are a great and easy way to use models for inference. These pipelines are objects that abstract most of
the complex code from the library, offering a simple API dedicated to several tasks, including Named Entity
Recognition, Masked Language Modeling, Sentiment Analysis, Feature Extraction and Question Answering. See the
[task summary](../task_summary) for examples of use.

There are two categories of pipeline abstractions to be aware about:

- The [pipeline()](/docs/transformers/main/en/main_classes/pipelines#transformers.pipeline) which is the most powerful object encapsulating all other pipelines.
- Task-specific pipelines are available for [audio](#audio), [computer vision](#computer-vision), [natural language processing](#natural-language-processing), and [multimodal](#multimodal) tasks.

## The pipeline abstraction[[transformers.pipeline]]

The *pipeline* abstraction is a wrapper around all the other available pipelines. It is instantiated as any other
pipeline but can provide additional quality of life.

Simple call on one item:

```python
>>> pipe = pipeline("text-classification")
>>> pipe("This restaurant is awesome")
[{'label': 'POSITIVE', 'score': 0.9998743534088135}]
```

If you want to use a specific model from the [hub](https://huggingface.co) you can ignore the task if the model on
the hub already defines it:

```python
>>> pipe = pipeline(model="FacebookAI/roberta-large-mnli")
>>> pipe("This restaurant is awesome")
[{'label': 'NEUTRAL', 'score': 0.7313136458396912}]
```

To call a pipeline on many items, you can call it with a *list*.

```python
>>> pipe = pipeline("text-classification")
>>> pipe(["This restaurant is awesome", "This restaurant is awful"])
[{'label': 'POSITIVE', 'score': 0.9998743534088135},
 {'label': 'NEGATIVE', 'score': 0.9996669292449951}]
```

To iterate over full datasets it is recommended to use a `dataset` directly. This means you don't need to allocate
the whole dataset at once, nor do you need to do batching yourself. This should work just as fast as custom loops on
GPU. If it doesn't don't hesitate to create an issue.

```python
import datasets
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
from tqdm.auto import tqdm

pipe = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h", device=0)
dataset = datasets.load_dataset("superb", name="asr", split="test")

# KeyDataset (only *pt*) will simply return the item in the dict returned by the dataset item
# as we're not interested in the *target* part of the dataset. For sentence pair use KeyPairDataset
for out in tqdm(pipe(KeyDataset(dataset, "file"))):
    print(out)
    # {"text": "NUMBER TEN FRESH NELLY IS WAITING ON YOU GOOD NIGHT HUSBAND"}
    # {"text": ....}
    # ....
```

For ease of use, a generator is also possible:

```python
from transformers import pipeline

pipe = pipeline("text-classification")

def data():
    while True:
        # This could come from a dataset, a database, a queue or HTTP request
        # in a server
        # Caveat: because this is iterative, you cannot use `num_workers > 1` variable
        # to use multiple threads to preprocess data. You can still have 1 thread that
        # does the preprocessing while the main runs the big inference
        yield "This is a test"

for out in pipe(data()):
    print(out)
    # {"text": "NUMBER TEN FRESH NELLY IS WAITING ON YOU GOOD NIGHT HUSBAND"}
    # {"text": ....}
    # ....
```

#### transformers.pipeline[[transformers.pipeline]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/__init__.py#L516)

Utility factory method to build a [Pipeline](/docs/transformers/main/en/main_classes/pipelines#transformers.Pipeline).

A pipeline consists of:

- One or more components for pre-processing model inputs, such as a [tokenizer](tokenizer),
[image_processor](image_processor), [feature_extractor](feature_extractor), or [processor](processors).
- A [model](model) that generates predictions from the inputs.
- Optional post-processing steps to refine the model's output, which can also be handled by processors.

While there are such optional arguments as `tokenizer`, `feature_extractor`, `image_processor`, and `processor`,
they shouldn't be specified all at once. If these components are not provided, `pipeline` will try to load
required ones automatically. In case you want to provide these components explicitly, please refer to a
specific pipeline in order to get more details regarding what components are required.

Examples:

```python
>>> from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer

>>> # Sentiment analysis pipeline
>>> analyzer = pipeline("sentiment-analysis")

>>> # Question answering pipeline, specifying the checkpoint identifier
>>> oracle = pipeline(
...     "question-answering", model="distilbert/distilbert-base-cased-distilled-squad", tokenizer="google-bert/bert-base-cased"
... )

>>> # Named entity recognition pipeline, passing in a specific model and tokenizer
>>> model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
>>> tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
>>> recognizer = pipeline("ner", model=model, tokenizer=tokenizer)
```

**Parameters:**

task (`str`) : The task defining which pipeline will be returned. Currently accepted tasks are:  - `"audio-classification"`: will return a [AudioClassificationPipeline](/docs/transformers/main/en/main_classes/pipelines#transformers.AudioClassificationPipeline). - `"automatic-speech-recognition"`: will return a [AutomaticSpeechRecognitionPipeline](/docs/transformers/main/en/main_classes/pipelines#transformers.AutomaticSpeechRecognitionPipeline). - `"depth-estimation"`: will return a [DepthEstimationPipeline](/docs/transformers/main/en/main_classes/pipelines#transformers.DepthEstimationPipeline). - `"document-question-answering"`: will return a [DocumentQuestionAnsweringPipeline](/docs/transformers/main/en/main_classes/pipelines#transformers.DocumentQuestionAnsweringPipeline). - `"feature-extraction"`: will return a [FeatureExtractionPipeline](/docs/transformers/main/en/main_classes/pipelines#transformers.FeatureExtractionPipeline). - `"fill-mask"`: will return a [FillMaskPipeline](/docs/transformers/main/en/main_classes/pipelines#transformers.FillMaskPipeline):. - `"image-classification"`: will return a [ImageClassificationPipeline](/docs/transformers/main/en/main_classes/pipelines#transformers.ImageClassificationPipeline). - `"image-feature-extraction"`: will return an [ImageFeatureExtractionPipeline](/docs/transformers/main/en/main_classes/pipelines#transformers.ImageFeatureExtractionPipeline). - `"image-segmentation"`: will return a [ImageSegmentationPipeline](/docs/transformers/main/en/main_classes/pipelines#transformers.ImageSegmentationPipeline). - `"image-text-to-text"`: will return a [ImageTextToTextPipeline](/docs/transformers/main/en/main_classes/pipelines#transformers.ImageTextToTextPipeline). - `"image-to-image"`: will return a [ImageToImagePipeline](/docs/transformers/main/en/main_classes/pipelines#transformers.ImageToImagePipeline). - `"image-to-text"`: will return a [ImageToTextPipeline](/docs/transformers/main/en/main_classes/pipelines#transformers.ImageToTextPipeline). - `"keypoint-matching"`: will return a [KeypointMatchingPipeline](/docs/transformers/main/en/main_classes/pipelines#transformers.KeypointMatchingPipeline). - `"mask-generation"`: will return a [MaskGenerationPipeline](/docs/transformers/main/en/main_classes/pipelines#transformers.MaskGenerationPipeline). - `"object-detection"`: will return a [ObjectDetectionPipeline](/docs/transformers/main/en/main_classes/pipelines#transformers.ObjectDetectionPipeline). - `"question-answering"`: will return a [QuestionAnsweringPipeline](/docs/transformers/main/en/main_classes/pipelines#transformers.QuestionAnsweringPipeline). - `"summarization"`: will return a [SummarizationPipeline](/docs/transformers/main/en/main_classes/pipelines#transformers.SummarizationPipeline). - `"table-question-answering"`: will return a [TableQuestionAnsweringPipeline](/docs/transformers/main/en/main_classes/pipelines#transformers.TableQuestionAnsweringPipeline). - `"text2text-generation"`: will return a [Text2TextGenerationPipeline](/docs/transformers/main/en/main_classes/pipelines#transformers.Text2TextGenerationPipeline). - `"text-classification"` (alias `"sentiment-analysis"` available): will return a [TextClassificationPipeline](/docs/transformers/main/en/main_classes/pipelines#transformers.TextClassificationPipeline). - `"text-generation"`: will return a [TextGenerationPipeline](/docs/transformers/main/en/main_classes/pipelines#transformers.TextGenerationPipeline):. - `"text-to-audio"` (alias `"text-to-speech"` available): will return a [TextToAudioPipeline](/docs/transformers/main/en/main_classes/pipelines#transformers.TextToAudioPipeline):. - `"token-classification"` (alias `"ner"` available): will return a [TokenClassificationPipeline](/docs/transformers/main/en/main_classes/pipelines#transformers.TokenClassificationPipeline). - `"translation"`: will return a [TranslationPipeline](/docs/transformers/main/en/main_classes/pipelines#transformers.TranslationPipeline). - `"translation_xx_to_yy"`: will return a [TranslationPipeline](/docs/transformers/main/en/main_classes/pipelines#transformers.TranslationPipeline). - `"video-classification"`: will return a [VideoClassificationPipeline](/docs/transformers/main/en/main_classes/pipelines#transformers.VideoClassificationPipeline). - `"visual-question-answering"`: will return a [VisualQuestionAnsweringPipeline](/docs/transformers/main/en/main_classes/pipelines#transformers.VisualQuestionAnsweringPipeline). - `"zero-shot-classification"`: will return a [ZeroShotClassificationPipeline](/docs/transformers/main/en/main_classes/pipelines#transformers.ZeroShotClassificationPipeline). - `"zero-shot-image-classification"`: will return a [ZeroShotImageClassificationPipeline](/docs/transformers/main/en/main_classes/pipelines#transformers.ZeroShotImageClassificationPipeline). - `"zero-shot-audio-classification"`: will return a [ZeroShotAudioClassificationPipeline](/docs/transformers/main/en/main_classes/pipelines#transformers.ZeroShotAudioClassificationPipeline). - `"zero-shot-object-detection"`: will return a [ZeroShotObjectDetectionPipeline](/docs/transformers/main/en/main_classes/pipelines#transformers.ZeroShotObjectDetectionPipeline). 

model (`str` or [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel), *optional*) : The model that will be used by the pipeline to make predictions. This can be a model identifier or an actual instance of a pretrained model inheriting from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel).  If not provided, the default for the `task` will be loaded.

config (`str` or [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig), *optional*) : The configuration that will be used by the pipeline to instantiate the model. This can be a model identifier or an actual pretrained model configuration inheriting from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig).  If not provided, the default configuration file for the requested model will be used. That means that if `model` is given, its default configuration will be used. However, if `model` is not supplied, this `task`'s default model's config is used instead.

tokenizer (`str` or [PreTrainedTokenizer](/docs/transformers/main/en/main_classes/tokenizer#transformers.PythonBackend), *optional*) : The tokenizer that will be used by the pipeline to encode data for the model. This can be a model identifier or an actual pretrained tokenizer inheriting from [PreTrainedTokenizer](/docs/transformers/main/en/main_classes/tokenizer#transformers.PythonBackend).  If not provided, the default tokenizer for the given `model` will be loaded (if it is a string). If `model` is not specified or not a string, then the default tokenizer for `config` is loaded (if it is a string). However, if `config` is also not given or not a string, then the default tokenizer for the given `task` will be loaded.

feature_extractor (`str` or `PreTrainedFeatureExtractor`, *optional*) : The feature extractor that will be used by the pipeline to encode data for the model. This can be a model identifier or an actual pretrained feature extractor inheriting from `PreTrainedFeatureExtractor`.  Feature extractors are used for non-NLP models, such as Speech or Vision models as well as multi-modal models. Multi-modal models will also require a tokenizer to be passed.  If not provided, the default feature extractor for the given `model` will be loaded (if it is a string). If `model` is not specified or not a string, then the default feature extractor for `config` is loaded (if it is a string). However, if `config` is also not given or not a string, then the default feature extractor for the given `task` will be loaded.

image_processor (`str` or [BaseImageProcessor](/docs/transformers/main/en/main_classes/image_processor#transformers.BaseImageProcessor), *optional*) : The image processor that will be used by the pipeline to preprocess images for the model. This can be a model identifier or an actual image processor inheriting from [BaseImageProcessor](/docs/transformers/main/en/main_classes/image_processor#transformers.BaseImageProcessor).  Image processors are used for Vision models and multi-modal models that require image inputs. Multi-modal models will also require a tokenizer to be passed.  If not provided, the default image processor for the given `model` will be loaded (if it is a string). If `model` is not specified or not a string, then the default image processor for `config` is loaded (if it is a string).

processor (`str` or [ProcessorMixin](/docs/transformers/main/en/main_classes/processors#transformers.ProcessorMixin), *optional*) : The processor that will be used by the pipeline to preprocess data for the model. This can be a model identifier or an actual processor inheriting from [ProcessorMixin](/docs/transformers/main/en/main_classes/processors#transformers.ProcessorMixin).  Processors are used for multi-modal models that require multi-modal inputs, for example, a model that requires both text and image inputs.  If not provided, the default processor for the given `model` will be loaded (if it is a string). If `model` is not specified or not a string, then the default processor for `config` is loaded (if it is a string).

revision (`str`, *optional*, defaults to `"main"`) : When passing a task name or a string model identifier: The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any identifier allowed by git.

use_fast (`bool`, *optional*, defaults to `True`) : Whether or not to use a Fast tokenizer if possible (a [PreTrainedTokenizerFast](/docs/transformers/main/en/main_classes/tokenizer#transformers.TokenizersBackend)).

token (`str` or *bool*, *optional*) : The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated when running `hf auth login`.

device (`int` or `str` or `torch.device`) : Defines the device (*e.g.*, `"cpu"`, `"cuda:1"`, `"mps"`, or a GPU ordinal rank like `1`) on which this pipeline will be allocated.

device_map (`str` or `dict[str, Union[int, str, torch.device]`, *optional*) : Sent directly as `model_kwargs` (just a simpler shortcut). When `accelerate` library is present, set `device_map="auto"` to compute the most optimized `device_map` automatically (see [here](https://huggingface.co/docs/accelerate/main/en/package_reference/big_modeling#accelerate.cpu_offload) for more information).    Do not use `device_map` AND `device` at the same time as they will conflict   

dtype (`str` or `torch.dtype`, *optional*) : Sent directly as `model_kwargs` (just a simpler shortcut) to use the available precision for this model (`torch.float16`, `torch.bfloat16`, ... or `"auto"`).

trust_remote_code (`bool`, *optional*, defaults to `False`) : Whether or not to allow for custom code defined on the Hub in their own modeling, configuration, tokenization or even pipeline files. This option should only be set to `True` for repositories you trust and in which you have read the code, as it will execute code present on the Hub on your local machine.

model_kwargs (`dict[str, Any]`, *optional*) : Additional dictionary of keyword arguments passed along to the model's `from_pretrained(..., **model_kwargs)` function.

kwargs (`dict[str, Any]`, *optional*) : Additional keyword arguments passed along to the specific pipeline init (see the documentation for the corresponding pipeline class for possible values).

**Returns:**

`[Pipeline](/docs/transformers/main/en/main_classes/pipelines#transformers.Pipeline)`

A suitable pipeline for the task.

## Pipeline batching

All pipelines can use batching. This will work
whenever the pipeline uses its streaming ability (so when passing lists or `Dataset` or `generator`).

```python
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
import datasets

dataset = datasets.load_dataset("imdb", name="plain_text", split="unsupervised")
pipe = pipeline("text-classification", device=0)
for out in pipe(KeyDataset(dataset, "text"), batch_size=8, truncation="only_first"):
    print(out)
    # [{'label': 'POSITIVE', 'score': 0.9998743534088135}]
    # Exactly the same output as before, but the content are passed
    # as batches to the model
```

However, this is not automatically a win for performance. It can be either a 10x speedup or 5x slowdown depending
on hardware, data and the actual model being used.

Example where it's mostly a speedup:

```python
from transformers import pipeline
from torch.utils.data import Dataset
from tqdm.auto import tqdm

pipe = pipeline("text-classification", device=0)

class MyDataset(Dataset):
    def __len__(self):
        return 5000

    def __getitem__(self, i):
        return "This is a test"

dataset = MyDataset()

for batch_size in [1, 8, 64, 256]:
    print("-" * 30)
    print(f"Streaming batch_size={batch_size}")
    for out in tqdm(pipe(dataset, batch_size=batch_size), total=len(dataset)):
        pass
```

```text
# On GTX 970
------------------------------
Streaming no batching
100%|██████████████████████████████████████████████████████████████████████| 5000/5000 [00:26
    for out in tqdm(pipe(dataset, batch_size=256), total=len(dataset)):
....
    q = q / math.sqrt(dim_per_head)  # (bs, n_heads, q_length, dim_per_head)
RuntimeError: CUDA out of memory. Tried to allocate 376.00 MiB (GPU 0; 3.95 GiB total capacity; 1.72 GiB already allocated; 354.88 MiB free; 2.46 GiB reserved in total by PyTorch)
```

There are no good (general) solutions for this problem, and your mileage may vary depending on your use cases. Rule of
thumb:

For users, a rule of thumb is:

- **Measure performance on your load, with your hardware. Measure, measure, and keep measuring. Real numbers are the
  only way to go.**
- If you are latency constrained (live product doing inference), don't batch.
- If you are using CPU, don't batch.
- If you are using throughput (you want to run your model on a bunch of static data), on GPU, then:

  - If you have no clue about the size of the sequence_length ("natural" data), by default don't batch, measure and
    try tentatively to add it, add OOM checks to recover when it will fail (and it will at some point if you don't
    control the sequence_length.)
  - If your sequence_length is super regular, then batching is more likely to be VERY interesting, measure and push
    it until you get OOMs.
  - The larger the GPU the more likely batching is going to be more interesting
- As soon as you enable batching, make sure you can handle OOMs nicely.

## Pipeline chunk batching

`zero-shot-classification` and `question-answering` are slightly specific in the sense, that a single input might yield
multiple forward pass of a model. Under normal circumstances, this would yield issues with `batch_size` argument.

In order to circumvent this issue, both of these pipelines are a bit specific, they are `ChunkPipeline` instead of
regular `Pipeline`. In short:

```python
preprocessed = pipe.preprocess(inputs)
model_outputs = pipe.forward(preprocessed)
outputs = pipe.postprocess(model_outputs)
```

Now becomes:

```python
all_model_outputs = []
for preprocessed in pipe.preprocess(inputs):
    model_outputs = pipe.forward(preprocessed)
    all_model_outputs.append(model_outputs)
outputs = pipe.postprocess(all_model_outputs)
```

This should be very transparent to your code because the pipelines are used in
the same way.

This is a simplified view, since the pipeline can handle automatically the batch to ! Meaning you don't have to care
about how many forward passes you inputs are actually going to trigger, you can optimize the `batch_size`
independently of the inputs. The caveats from the previous section still apply.

## Pipeline FP16 inference

Models can be run in FP16 which can be significantly faster on GPU while saving memory. Most models will not suffer noticeable performance loss from this. The larger the model, the less likely that it will.

To enable FP16 inference, you can simply pass `dtype=torch.float16` or `dtype='float16'` to the pipeline constructor. Note that this only works for models with a PyTorch backend. Your inputs will be converted to FP16 internally.

## Pipeline custom code

If you want to override a specific pipeline.

Don't hesitate to create an issue for your task at hand, the goal of the pipeline is to be easy to use and support most
cases, so `transformers` could maybe support your use case.

If you want to try simply you can:

- Subclass your pipeline of choice

```python
class MyPipeline(TextClassificationPipeline):
    def postprocess():
        # Your code goes here
        scores = scores * 100
        # And here

my_pipeline = MyPipeline(model=model, tokenizer=tokenizer, ...)
# or if you use *pipeline* function, then:
my_pipeline = pipeline(model="xxxx", pipeline_class=MyPipeline)
```

That should enable you to do all the custom code you want.

## Implementing a pipeline

[Implementing a new pipeline](../add_new_pipeline)

## Audio

Pipelines available for audio tasks include the following.

### AudioClassificationPipeline[[transformers.AudioClassificationPipeline]]

#### transformers.AudioClassificationPipeline[[transformers.AudioClassificationPipeline]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/audio_classification.py#L67)

Audio classification pipeline using any `AutoModelForAudioClassification`. This pipeline predicts the class of a
raw waveform or an audio file. In case of an audio file, ffmpeg should be installed to support multiple audio
formats.

Example:

```python
>>> from transformers import pipeline

>>> classifier = pipeline(model="superb/wav2vec2-base-superb-ks")
>>> classifier("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac")
[{'score': 0.997, 'label': '_unknown_'}, {'score': 0.002, 'label': 'left'}, {'score': 0.0, 'label': 'yes'}, {'score': 0.0, 'label': 'down'}, {'score': 0.0, 'label': 'stop'}]
```

Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

This pipeline can currently be loaded from [pipeline()](/docs/transformers/main/en/main_classes/pipelines#transformers.pipeline) using the following task identifier:
`"audio-classification"`.

See the list of available models on
[huggingface.co/models](https://huggingface.co/models?filter=audio-classification).

__call__transformers.AudioClassificationPipeline.__call__https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/audio_classification.py#L108[{"name": "inputs", "val": ": numpy.ndarray | bytes | str | dict"}, {"name": "**kwargs", "val": ": typing.Any"}]- **inputs** (`np.ndarray` or `bytes` or `str` or `dict`) --
  The inputs is either :
  - `str` that is the filename of the audio file, the file will be read at the correct sampling rate
    to get the waveform using *ffmpeg*. This requires *ffmpeg* to be installed on the system.
  - `bytes` it is supposed to be the content of an audio file and is interpreted by *ffmpeg* in the
    same way.
  - (`np.ndarray` of shape (n, ) of type `np.float32` or `np.float64`)
    Raw audio at the correct sampling rate (no further check will be done)
  - `dict` form can be used to pass raw audio sampled at arbitrary `sampling_rate` and let this
    pipeline do the resampling. The dict must be either be in the format `{"sampling_rate": int,
    "raw": np.array}`, or `{"sampling_rate": int, "array": np.array}`, where the key `"raw"` or
    `"array"` is used to denote the raw audio waveform.
- **top_k** (`int`, *optional*, defaults to None) --
  The number of top labels that will be returned by the pipeline. If the provided number is `None` or
  higher than the number of labels available in the model configuration, it will default to the number of
  labels.
- **function_to_apply(`str`,** *optional*, defaults to "softmax") --
  The function to apply to the model output. By default, the pipeline will apply the softmax function to
  the output of the model. Valid options: ["softmax", "sigmoid", "none"]. Note that passing Python's
  built-in `None` will default to "softmax", so you need to pass the string "none" to disable any
  post-processing.0A list of `dict` with the following keys- **label** (`str`) -- The label predicted.
- **score** (`float`) -- The corresponding probability.

Classify the sequence(s) given as inputs. See the [AutomaticSpeechRecognitionPipeline](/docs/transformers/main/en/main_classes/pipelines#transformers.AutomaticSpeechRecognitionPipeline) documentation for more
information.

**Parameters:**

model ([PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel)) : The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel).

feature_extractor ([SequenceFeatureExtractor](/docs/transformers/main/en/main_classes/feature_extractor#transformers.SequenceFeatureExtractor)) : The feature extractor that will be used by the pipeline to encode data for the model. This object inherits from [SequenceFeatureExtractor](/docs/transformers/main/en/main_classes/feature_extractor#transformers.SequenceFeatureExtractor).

modelcard (`str` or `ModelCard`, *optional*) : Model card attributed to the model for this pipeline.

task (`str`, defaults to `""`) : A task-identifier for the pipeline.

num_workers (`int`, *optional*, defaults to 8) : When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the number of workers to be used.

batch_size (`int`, *optional*, defaults to 1) : When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the size of the batch to use, for inference this is not always beneficial, please read [Batching with pipelines](https://huggingface.co/transformers/main_classes/pipelines.html#pipeline-batching) .

args_parser ([ArgumentHandler](/docs/transformers/main/en/internal/pipelines_utils#transformers.pipelines.ArgumentHandler), *optional*) : Reference to the object in charge of parsing supplied pipeline parameters.

device (`int`, *optional*, defaults to -1) : Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, a positive will run the model on the associated CUDA device id. You can pass native `torch.device` or a `str` too

dtype (`str` or `torch.dtype`, *optional*) : Sent directly as `model_kwargs` (just a simpler shortcut) to use the available precision for this model (`torch.float16`, `torch.bfloat16`, ... or `"auto"`)

binary_output (`bool`, *optional*, defaults to `False`) : Flag indicating if the output the pipeline should happen in a serialized format (i.e., pickle) or as the raw output data e.g. text.

**Returns:**

`A list of `dict` with the following keys`

- **label** (`str`) -- The label predicted.
- **score** (`float`) -- The corresponding probability.

### AutomaticSpeechRecognitionPipeline[[transformers.AutomaticSpeechRecognitionPipeline]]

#### transformers.AutomaticSpeechRecognitionPipeline[[transformers.AutomaticSpeechRecognitionPipeline]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/automatic_speech_recognition.py#L112)

Pipeline that aims at extracting spoken text contained within some audio.

The input can be either a raw waveform or a audio file. In case of the audio file, ffmpeg should be installed for
to support multiple audio formats

Unless the model you're using explicitly sets these generation parameters in its configuration files
(`generation_config.json`), the following default values will be used:
- max_new_tokens: 256
- num_beams: 5

Example:

```python
>>> from transformers import pipeline

>>> transcriber = pipeline(model="openai/whisper-base")
>>> transcriber("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac")
{'text': ' He hoped there would be stew for dinner, turnips and carrots and bruised potatoes and fat mutton pieces to be ladled out in thick, peppered flour-fatten sauce.'}
```

Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

__call__transformers.AutomaticSpeechRecognitionPipeline.__call__https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/automatic_speech_recognition.py#L213[{"name": "inputs", "val": ": numpy.ndarray | bytes | str | dict"}, {"name": "**kwargs", "val": ": typing.Any"}]- **inputs** (`np.ndarray` or `bytes` or `str` or `dict`) --
  The inputs is either :
  - `str` that is either the filename of a local audio file, or a public URL address to download the
    audio file. The file will be read at the correct sampling rate to get the waveform using
    *ffmpeg*. This requires *ffmpeg* to be installed on the system.
  - `bytes` it is supposed to be the content of an audio file and is interpreted by *ffmpeg* in the
    same way.
  - (`np.ndarray` of shape (n, ) of type `np.float32` or `np.float64`)
    Raw audio at the correct sampling rate (no further check will be done)
  - `dict` form can be used to pass raw audio sampled at arbitrary `sampling_rate` and let this
    pipeline do the resampling. The dict must be in the format `{"sampling_rate": int, "raw":
    np.array}` with optionally a `"stride": (left: int, right: int)` than can ask the pipeline to
    treat the first `left` samples and last `right` samples to be ignored in decoding (but used at
    inference to provide more context to the model). Only use `stride` with CTC models.
- **return_timestamps** (*optional*, `str` or `bool`) --
  Only available for pure CTC models (Wav2Vec2, HuBERT, etc) and the Whisper model. Not available for
  other sequence-to-sequence models.

  For CTC models, timestamps can take one of two formats:
  - `"char"`: the pipeline will return timestamps along the text for every character in the text. For
    instance, if you get `[{"text": "h", "timestamp": (0.5, 0.6)}, {"text": "i", "timestamp": (0.7,
    0.9)}]`, then it means the model predicts that the letter "h" was spoken after `0.5` and before
    `0.6` seconds.
  - `"word"`: the pipeline will return timestamps along the text for every word in the text. For
    instance, if you get `[{"text": "hi ", "timestamp": (0.5, 0.9)}, {"text": "there", "timestamp":
    (1.0, 1.5)}]`, then it means the model predicts that the word "hi" was spoken after `0.5` and
    before `0.9` seconds.

  For the Whisper model, timestamps can take one of two formats:
  - `"word"`: same as above for word-level CTC timestamps. Word-level timestamps are predicted
    through the *dynamic-time warping (DTW)* algorithm, an approximation to word-level timestamps
    by inspecting the cross-attention weights.
  - `True`: the pipeline will return timestamps along the text for *segments* of words in the text.
    For instance, if you get `[{"text": " Hi there!", "timestamp": (0.5, 1.5)}]`, then it means the
    model predicts that the segment "Hi there!" was spoken after `0.5` and before `1.5` seconds.
    Note that a segment of text refers to a sequence of one or more words, rather than individual
    words as with word-level timestamps.
- **generate_kwargs** (`dict`, *optional*) --
  The dictionary of ad-hoc parametrization of `generate_config` to be used for the generation call. For a
  complete overview of generate, check the [following
  guide](https://huggingface.co/docs/transformers/en/main_classes/text_generation).0`Dict`A dictionary with the following keys:
- **text** (`str`): The recognized text.
- **chunks** (*optional(, `list[Dict]`)
  When using `return_timestamps`, the `chunks` will become a list containing all the various text
  chunks identified by the model, *e.g.* `[{"text": "hi ", "timestamp": (0.5, 0.9)}, {"text":
  "there", "timestamp": (1.0, 1.5)}]`. The original full text can roughly be recovered by doing
  `"".join(chunk["text"] for chunk in output["chunks"])`.

Transcribe the audio sequence(s) given as inputs to text. See the [AutomaticSpeechRecognitionPipeline](/docs/transformers/main/en/main_classes/pipelines#transformers.AutomaticSpeechRecognitionPipeline)
documentation for more information.

**Parameters:**

model ([PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel)) : The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel).

feature_extractor ([SequenceFeatureExtractor](/docs/transformers/main/en/main_classes/feature_extractor#transformers.SequenceFeatureExtractor)) : The feature extractor that will be used by the pipeline to encode waveform for the model.

tokenizer ([PreTrainedTokenizer](/docs/transformers/main/en/main_classes/tokenizer#transformers.PythonBackend)) : The tokenizer that will be used by the pipeline to encode data for the model. This object inherits from [PreTrainedTokenizer](/docs/transformers/main/en/main_classes/tokenizer#transformers.PythonBackend).

decoder (`pyctcdecode.BeamSearchDecoderCTC`, *optional*) : [PyCTCDecode's BeamSearchDecoderCTC](https://github.com/kensho-technologies/pyctcdecode/blob/2fd33dc37c4111417e08d89ccd23d28e9b308d19/pyctcdecode/decoder.py#L180) can be passed for language model boosted decoding. See [Wav2Vec2ProcessorWithLM](/docs/transformers/main/en/model_doc/wav2vec2#transformers.Wav2Vec2ProcessorWithLM) for more information.

chunk_length_s (`float`, *optional*, defaults to 0) : The input length for in each chunk. If `chunk_length_s = 0` then chunking is disabled (default).    For more information on how to effectively use `chunk_length_s`, please have a look at the [ASR chunking blog post](https://huggingface.co/blog/asr-chunking).   

stride_length_s (`float`, *optional*, defaults to `chunk_length_s / 6`) : The length of stride on the left and right of each chunk. Used only with `chunk_length_s > 0`. This enables the model to *see* more context and infer letters better than without this context but the pipeline discards the stride bits at the end to make the final reconstitution as perfect as possible.    For more information on how to effectively use `stride_length_s`, please have a look at the [ASR chunking blog post](https://huggingface.co/blog/asr-chunking).   

device (Union[`int`, `torch.device`], *optional*) : Device ordinal for CPU/GPU supports. Setting this to `None` will leverage CPU, a positive will run the model on the associated CUDA device id.

**Returns:**

``Dict``

A dictionary with the following keys:
- **text** (`str`): The recognized text.
- **chunks** (*optional(, `list[Dict]`)
  When using `return_timestamps`, the `chunks` will become a list containing all the various text
  chunks identified by the model, *e.g.* `[{"text": "hi ", "timestamp": (0.5, 0.9)}, {"text":
  "there", "timestamp": (1.0, 1.5)}]`. The original full text can roughly be recovered by doing
  `"".join(chunk["text"] for chunk in output["chunks"])`.

### TextToAudioPipeline[[transformers.TextToAudioPipeline]]

#### transformers.TextToAudioPipeline[[transformers.TextToAudioPipeline]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/text_to_audio.py#L45)

Text-to-audio generation pipeline using any `AutoModelForTextToWaveform` or `AutoModelForTextToSpectrogram`. This
pipeline generates an audio file from an input text and optional other conditional inputs.

Unless the model you're using explicitly sets these generation parameters in its configuration files
(`generation_config.json`), the following default values will be used:
- max_new_tokens: 256

Example:

```python
>>> from transformers import pipeline

>>> pipe = pipeline(model="suno/bark-small")
>>> output = pipe("Hey it's HuggingFace on the phone!")

>>> audio = output["audio"]
>>> sampling_rate = output["sampling_rate"]
```

Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

You can specify parameters passed to the model by using `TextToAudioPipeline.__call__.forward_params` or
`TextToAudioPipeline.__call__.generate_kwargs`.

Example:

```python
>>> from transformers import pipeline

>>> music_generator = pipeline(task="text-to-audio", model="facebook/musicgen-small")

>>> # diversify the music generation by adding randomness with a high temperature and set a maximum music length
>>> generate_kwargs = {
...     "do_sample": True,
...     "temperature": 0.7,
...     "max_new_tokens": 35,
... }

>>> outputs = music_generator("Techno music with high melodic riffs", generate_kwargs=generate_kwargs)
```

This pipeline can currently be loaded from [pipeline()](/docs/transformers/main/en/main_classes/pipelines#transformers.pipeline) using the following task identifiers: `"text-to-speech"` or
`"text-to-audio"`.

See the list of available models on [huggingface.co/models](https://huggingface.co/models?filter=text-to-speech).

__call__transformers.TextToAudioPipeline.__call__https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/text_to_audio.py#L238[{"name": "text_inputs", "val": ""}, {"name": "**forward_params", "val": ""}]- **text_inputs** (`str`, `list[str]`, `ChatType`, or `list[ChatType]`) --
  One or several texts to generate. If strings or a list of string are passed, this pipeline will
  generate the corresponding text. Alternatively, a "chat", in the form of a list of dicts with "role"
  and "content" keys, can be passed, or a list of such chats. When chats are passed, the model's chat
  template will be used to format them before passing them to the model.
- **forward_params** (`dict`, *optional*) --
  Parameters passed to the model generation/forward method. `forward_params` are always passed to the
  underlying model.
- **generate_kwargs** (`dict`, *optional*) --
  The dictionary of ad-hoc parametrization of `generate_config` to be used for the generation call. For a
  complete overview of generate, check the [following
  guide](https://huggingface.co/docs/transformers/en/main_classes/text_generation). `generate_kwargs` are
  only passed to the underlying model if the latter is a generative model.0`AudioOutput` or a list of `AudioOutput`, which is a `TypedDict` with two keys- **audio** (`np.ndarray` of shape `(nb_channels, audio_length)`) -- The generated audio waveform.
- **sampling_rate** (`int`) -- The sampling rate of the generated audio waveform.

Generates speech/audio from the inputs. See the [TextToAudioPipeline](/docs/transformers/main/en/main_classes/pipelines#transformers.TextToAudioPipeline) documentation for more information.

**Parameters:**

text_inputs (`str`, `list[str]`, `ChatType`, or `list[ChatType]`) : One or several texts to generate. If strings or a list of string are passed, this pipeline will generate the corresponding text. Alternatively, a "chat", in the form of a list of dicts with "role" and "content" keys, can be passed, or a list of such chats. When chats are passed, the model's chat template will be used to format them before passing them to the model.

forward_params (`dict`, *optional*) : Parameters passed to the model generation/forward method. `forward_params` are always passed to the underlying model.

generate_kwargs (`dict`, *optional*) : The dictionary of ad-hoc parametrization of `generate_config` to be used for the generation call. For a complete overview of generate, check the [following guide](https://huggingface.co/docs/transformers/en/main_classes/text_generation). `generate_kwargs` are only passed to the underlying model if the latter is a generative model.

**Returns:**

``AudioOutput` or a list of `AudioOutput`, which is a `TypedDict` with two keys`

- **audio** (`np.ndarray` of shape `(nb_channels, audio_length)`) -- The generated audio waveform.
- **sampling_rate** (`int`) -- The sampling rate of the generated audio waveform.

### ZeroShotAudioClassificationPipeline[[transformers.ZeroShotAudioClassificationPipeline]]

#### transformers.ZeroShotAudioClassificationPipeline[[transformers.ZeroShotAudioClassificationPipeline]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/zero_shot_audio_classification.py#L33)

Zero shot audio classification pipeline using `ClapModel`. This pipeline predicts the class of an audio when you
provide an audio and a set of `candidate_labels`.

The default `hypothesis_template` is : `"This is a sound of {}."`. Make sure you update it for your usage.

Example:
```python
>>> from transformers import pipeline
>>> from datasets import load_dataset

>>> dataset = load_dataset("ashraq/esc50")
>>> audio = next(iter(dataset["train"]["audio"]))["array"]
>>> classifier = pipeline(task="zero-shot-audio-classification", model="laion/clap-htsat-unfused")
>>> classifier(audio, candidate_labels=["Sound of a dog", "Sound of vacuum cleaner"])
[{'score': 0.9996, 'label': 'Sound of a dog'}, {'score': 0.0004, 'label': 'Sound of vacuum cleaner'}]
```

Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial) This audio
classification pipeline can currently be loaded from [pipeline()](/docs/transformers/main/en/main_classes/pipelines#transformers.pipeline) using the following task identifier:
`"zero-shot-audio-classification"`. See the list of available models on
[huggingface.co/models](https://huggingface.co/models?filter=zero-shot-audio-classification).

__call__transformers.ZeroShotAudioClassificationPipeline.__call__https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/zero_shot_audio_classification.py#L71[{"name": "audios", "val": ": numpy.ndarray | bytes | str | dict"}, {"name": "**kwargs", "val": ": typing.Any"}]- **audios** (`str`, `list[str]`, `np.array` or `list[np.array]`) --
  The pipeline handles three types of inputs:
  - A string containing a http link pointing to an audio
  - A string containing a local path to an audio
  - An audio loaded in numpy
- **candidate_labels** (`list[str]`) --
  The candidate labels for this audio. They will be formatted using *hypothesis_template*.
- **hypothesis_template** (`str`, *optional*, defaults to `"This is a sound of {}"`) --
  The format used in conjunction with *candidate_labels* to attempt the audio classification by
  replacing the placeholder with the candidate_labels. Pass "{}" if *candidate_labels* are
  already formatted.0A list of dictionaries containing one entry per proposed label. Each dictionary contains the
following keys:
- **label** (`str`) -- One of the suggested *candidate_labels*.
- **score** (`float`) -- The score attributed by the model to that label. It is a value between
  0 and 1, computed as the `softmax` of `logits_per_audio`.

Assign labels to the audio(s) passed as inputs.

**Parameters:**

model ([PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel)) : The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel).

tokenizer ([PreTrainedTokenizer](/docs/transformers/main/en/main_classes/tokenizer#transformers.PythonBackend)) : The tokenizer that will be used by the pipeline to encode data for the model. This object inherits from [PreTrainedTokenizer](/docs/transformers/main/en/main_classes/tokenizer#transformers.PythonBackend).

feature_extractor ([SequenceFeatureExtractor](/docs/transformers/main/en/main_classes/feature_extractor#transformers.SequenceFeatureExtractor)) : The feature extractor that will be used by the pipeline to encode data for the model. This object inherits from [SequenceFeatureExtractor](/docs/transformers/main/en/main_classes/feature_extractor#transformers.SequenceFeatureExtractor).

modelcard (`str` or `ModelCard`, *optional*) : Model card attributed to the model for this pipeline.

task (`str`, defaults to `""`) : A task-identifier for the pipeline.

num_workers (`int`, *optional*, defaults to 8) : When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the number of workers to be used.

batch_size (`int`, *optional*, defaults to 1) : When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the size of the batch to use, for inference this is not always beneficial, please read [Batching with pipelines](https://huggingface.co/transformers/main_classes/pipelines.html#pipeline-batching) .

args_parser ([ArgumentHandler](/docs/transformers/main/en/internal/pipelines_utils#transformers.pipelines.ArgumentHandler), *optional*) : Reference to the object in charge of parsing supplied pipeline parameters.

device (`int`, *optional*, defaults to -1) : Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, a positive will run the model on the associated CUDA device id. You can pass native `torch.device` or a `str` too

dtype (`str` or `torch.dtype`, *optional*) : Sent directly as `model_kwargs` (just a simpler shortcut) to use the available precision for this model (`torch.float16`, `torch.bfloat16`, ... or `"auto"`)

binary_output (`bool`, *optional*, defaults to `False`) : Flag indicating if the output the pipeline should happen in a serialized format (i.e., pickle) or as the raw output data e.g. text.

**Returns:**

A list of dictionaries containing one entry per proposed label. Each dictionary contains the
following keys:
- **label** (`str`) -- One of the suggested *candidate_labels*.
- **score** (`float`) -- The score attributed by the model to that label. It is a value between
  0 and 1, computed as the `softmax` of `logits_per_audio`.

## Computer vision

Pipelines available for computer vision tasks include the following.

### DepthEstimationPipeline[[transformers.DepthEstimationPipeline]]

#### transformers.DepthEstimationPipeline[[transformers.DepthEstimationPipeline]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/depth_estimation.py#L25)

Depth estimation pipeline using any `AutoModelForDepthEstimation`. This pipeline predicts the depth of an image.

Example:

```python
>>> from transformers import pipeline

>>> depth_estimator = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-base-hf")
>>> output = depth_estimator("http://images.cocodataset.org/val2017/000000039769.jpg")
>>> # This is a tensor with the values being the depth expressed in meters for each pixel
>>> output["predicted_depth"].shape
torch.Size([1, 384, 384])
```

Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

This depth estimation pipeline can currently be loaded from [pipeline()](/docs/transformers/main/en/main_classes/pipelines#transformers.pipeline) using the following task identifier:
`"depth-estimation"`.

See the list of available models on [huggingface.co/models](https://huggingface.co/models?filter=depth-estimation).

__call__transformers.DepthEstimationPipeline.__call__https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/depth_estimation.py#L66[{"name": "inputs", "val": ": typing.Union[str, list[str], ForwardRef('Image.Image'), list['Image.Image']]"}, {"name": "**kwargs", "val": ": typing.Any"}]- **inputs** (`str`, `list[str]`, `PIL.Image` or `list[PIL.Image]`) --
  The pipeline handles three types of images:

  - A string containing a http link pointing to an image
  - A string containing a local path to an image
  - An image loaded in PIL directly

  The pipeline accepts either a single image or a batch of images, which must then be passed as a string.
  Images in a batch must all be in the same format: all as http links, all as local paths, or all as PIL
  images.
- **parameters** (`Dict`, *optional*) --
  A dictionary of argument names to parameter values, to control pipeline behaviour.
  The only parameter available right now is `timeout`, which is the length of time, in seconds,
  that the pipeline should wait before giving up on trying to download an image.
- **timeout** (`float`, *optional*, defaults to None) --
  The maximum time in seconds to wait for fetching images from the web. If None, no timeout is set and
  the call may block forever.0A dictionary or a list of dictionaries containing result. If the input is a single image, will return a
dictionary, if the input is a list of several images, will return a list of dictionaries corresponding to
the images.

The dictionaries contain the following keys:

- **predicted_depth** (`torch.Tensor`) -- The predicted depth by the model as a `torch.Tensor`.
- **depth** (`PIL.Image`) -- The predicted depth by the model as a `PIL.Image`.

Predict the depth(s) of the image(s) passed as inputs.

**Parameters:**

model ([PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel)) : The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel).

image_processor ([BaseImageProcessor](/docs/transformers/main/en/main_classes/image_processor#transformers.BaseImageProcessor)) : The image processor that will be used by the pipeline to encode data for the model. This object inherits from [BaseImageProcessor](/docs/transformers/main/en/main_classes/image_processor#transformers.BaseImageProcessor).

modelcard (`str` or `ModelCard`, *optional*) : Model card attributed to the model for this pipeline.

task (`str`, defaults to `""`) : A task-identifier for the pipeline.

num_workers (`int`, *optional*, defaults to 8) : When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the number of workers to be used.

batch_size (`int`, *optional*, defaults to 1) : When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the size of the batch to use, for inference this is not always beneficial, please read [Batching with pipelines](https://huggingface.co/transformers/main_classes/pipelines.html#pipeline-batching) .

args_parser ([ArgumentHandler](/docs/transformers/main/en/internal/pipelines_utils#transformers.pipelines.ArgumentHandler), *optional*) : Reference to the object in charge of parsing supplied pipeline parameters.

device (`int`, *optional*, defaults to -1) : Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, a positive will run the model on the associated CUDA device id. You can pass native `torch.device` or a `str` too

dtype (`str` or `torch.dtype`, *optional*) : Sent directly as `model_kwargs` (just a simpler shortcut) to use the available precision for this model (`torch.float16`, `torch.bfloat16`, ... or `"auto"`)

binary_output (`bool`, *optional*, defaults to `False`) : Flag indicating if the output the pipeline should happen in a serialized format (i.e., pickle) or as the raw output data e.g. text.

**Returns:**

A dictionary or a list of dictionaries containing result. If the input is a single image, will return a
dictionary, if the input is a list of several images, will return a list of dictionaries corresponding to
the images.

The dictionaries contain the following keys:

- **predicted_depth** (`torch.Tensor`) -- The predicted depth by the model as a `torch.Tensor`.
- **depth** (`PIL.Image`) -- The predicted depth by the model as a `PIL.Image`.

### ImageClassificationPipeline[[transformers.ImageClassificationPipeline]]

#### transformers.ImageClassificationPipeline[[transformers.ImageClassificationPipeline]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/image_classification.py#L73)

Image classification pipeline using any `AutoModelForImageClassification`. This pipeline predicts the class of an
image.

Example:

```python
>>> from transformers import pipeline

>>> classifier = pipeline(model="microsoft/beit-base-patch16-224-pt22k-ft22k")
>>> classifier("https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png")
[{'score': 0.442, 'label': 'macaw'}, {'score': 0.088, 'label': 'popinjay'}, {'score': 0.075, 'label': 'parrot'}, {'score': 0.073, 'label': 'parodist, lampooner'}, {'score': 0.046, 'label': 'poll, poll_parrot'}]
```

Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

This image classification pipeline can currently be loaded from [pipeline()](/docs/transformers/main/en/main_classes/pipelines#transformers.pipeline) using the following task identifier:
`"image-classification"`.

See the list of available models on
[huggingface.co/models](https://huggingface.co/models?filter=image-classification).

__call__transformers.ImageClassificationPipeline.__call__https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/image_classification.py#L127[{"name": "inputs", "val": ": typing.Union[str, list[str], ForwardRef('Image.Image'), list['Image.Image']]"}, {"name": "**kwargs", "val": ": typing.Any"}]- **inputs** (`str`, `list[str]`, `PIL.Image` or `list[PIL.Image]`) --
  The pipeline handles three types of images:

  - A string containing a http link pointing to an image
  - A string containing a local path to an image
  - An image loaded in PIL directly

  The pipeline accepts either a single image or a batch of images, which must then be passed as a string.
  Images in a batch must all be in the same format: all as http links, all as local paths, or all as PIL
  images.
- **function_to_apply** (`str`, *optional*, defaults to `"default"`) --
  The function to apply to the model outputs in order to retrieve the scores. Accepts four different
  values:

  If this argument is not specified, then it will apply the following functions according to the number
  of labels:

  - If the model has a single label, will apply the sigmoid function on the output.
  - If the model has several labels, will apply the softmax function on the output.

  Possible values are:

  - `"sigmoid"`: Applies the sigmoid function on the output.
  - `"softmax"`: Applies the softmax function on the output.
  - `"none"`: Does not apply any function on the output.
- **top_k** (`int`, *optional*, defaults to 5) --
  The number of top labels that will be returned by the pipeline. If the provided number is higher than
  the number of labels available in the model configuration, it will default to the number of labels.
- **timeout** (`float`, *optional*, defaults to None) --
  The maximum time in seconds to wait for fetching images from the web. If None, no timeout is set and
  the call may block forever.0A dictionary or a list of dictionaries containing result. If the input is a single image, will return a
dictionary, if the input is a list of several images, will return a list of dictionaries corresponding to
the images.

The dictionaries contain the following keys:

- **label** (`str`) -- The label identified by the model.
- **score** (`int`) -- The score attributed by the model for that label.

Assign labels to the image(s) passed as inputs.

**Parameters:**

model ([PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel)) : The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel).

image_processor ([BaseImageProcessor](/docs/transformers/main/en/main_classes/image_processor#transformers.BaseImageProcessor)) : The image processor that will be used by the pipeline to encode data for the model. This object inherits from [BaseImageProcessor](/docs/transformers/main/en/main_classes/image_processor#transformers.BaseImageProcessor).

modelcard (`str` or `ModelCard`, *optional*) : Model card attributed to the model for this pipeline.

task (`str`, defaults to `""`) : A task-identifier for the pipeline.

num_workers (`int`, *optional*, defaults to 8) : When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the number of workers to be used.

batch_size (`int`, *optional*, defaults to 1) : When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the size of the batch to use, for inference this is not always beneficial, please read [Batching with pipelines](https://huggingface.co/transformers/main_classes/pipelines.html#pipeline-batching) .

args_parser ([ArgumentHandler](/docs/transformers/main/en/internal/pipelines_utils#transformers.pipelines.ArgumentHandler), *optional*) : Reference to the object in charge of parsing supplied pipeline parameters.

device (`int`, *optional*, defaults to -1) : Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, a positive will run the model on the associated CUDA device id. You can pass native `torch.device` or a `str` too

dtype (`str` or `torch.dtype`, *optional*) : Sent directly as `model_kwargs` (just a simpler shortcut) to use the available precision for this model (`torch.float16`, `torch.bfloat16`, ... or `"auto"`)

binary_output (`bool`, *optional*, defaults to `False`) : Flag indicating if the output the pipeline should happen in a serialized format (i.e., pickle) or as the raw output data e.g. text.

function_to_apply (`str`, *optional*, defaults to `"default"`) : The function to apply to the model outputs in order to retrieve the scores. Accepts four different values:  - `"default"`: if the model has a single label, will apply the sigmoid function on the output. If the model has several labels, will apply the softmax function on the output. - `"sigmoid"`: Applies the sigmoid function on the output. - `"softmax"`: Applies the softmax function on the output. - `"none"`: Does not apply any function on the output.

**Returns:**

A dictionary or a list of dictionaries containing result. If the input is a single image, will return a
dictionary, if the input is a list of several images, will return a list of dictionaries corresponding to
the images.

The dictionaries contain the following keys:

- **label** (`str`) -- The label identified by the model.
- **score** (`int`) -- The score attributed by the model for that label.

### ImageSegmentationPipeline[[transformers.ImageSegmentationPipeline]]

#### transformers.ImageSegmentationPipeline[[transformers.ImageSegmentationPipeline]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/image_segmentation.py#L27)

Image segmentation pipeline using any `AutoModelForXXXSegmentation`. This pipeline predicts masks of objects and
their classes.

Example:

```python
>>> from transformers import pipeline

>>> segmenter = pipeline(model="facebook/detr-resnet-50-panoptic")
>>> segments = segmenter("https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png")
>>> len(segments)
2

>>> segments[0]["label"]
'bird'

>>> segments[1]["label"]
'bird'

>>> type(segments[0]["mask"])  # This is a black and white mask showing where is the bird on the original image.

>>> segments[0]["mask"].size
(768, 512)
```

This image segmentation pipeline can currently be loaded from [pipeline()](/docs/transformers/main/en/main_classes/pipelines#transformers.pipeline) using the following task identifier:
`"image-segmentation"`.

See the list of available models on
[huggingface.co/models](https://huggingface.co/models?filter=image-segmentation).

__call__transformers.ImageSegmentationPipeline.__call__https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/image_segmentation.py#L101[{"name": "inputs", "val": ": typing.Union[str, ForwardRef('Image.Image'), list[str], list['Image.Image']]"}, {"name": "**kwargs", "val": ": typing.Any"}]- **inputs** (`str`, `list[str]`, `PIL.Image` or `list[PIL.Image]`) --
  The pipeline handles three types of images:

  - A string containing an HTTP(S) link pointing to an image
  - A string containing a local path to an image
  - An image loaded in PIL directly

  The pipeline accepts either a single image or a batch of images. Images in a batch must all be in the
  same format: all as HTTP(S) links, all as local paths, or all as PIL images.
- **subtask** (`str`, *optional*) --
  Segmentation task to be performed, choose [`semantic`, `instance` and `panoptic`] depending on model
  capabilities. If not set, the pipeline will attempt tp resolve in the following order:
  `panoptic`, `instance`, `semantic`.
- **threshold** (`float`, *optional*, defaults to 0.9) --
  Probability threshold to filter out predicted masks.
- **mask_threshold** (`float`, *optional*, defaults to 0.5) --
  Threshold to use when turning the predicted masks into binary values.
- **overlap_mask_area_threshold** (`float`, *optional*, defaults to 0.5) --
  Mask overlap threshold to eliminate small, disconnected segments.
- **timeout** (`float`, *optional*, defaults to None) --
  The maximum time in seconds to wait for fetching images from the web. If None, no timeout is set and
  the call may block forever.0If the input is a single image, will return a list of dictionaries, if the input is a list of several images,
will return a list of list of dictionaries corresponding to each image.

The dictionaries contain the mask, label and score (where applicable) of each detected object and contains
the following keys:

- **label** (`str`) -- The class label identified by the model.
- **mask** (`PIL.Image`) -- A binary mask of the detected object as a Pil Image of shape (width, height) of
  the original image. Returns a mask filled with zeros if no object is found.
- **score** (*optional* `float`) -- Optionally, when the model is capable of estimating a confidence of the
  "object" described by the label and the mask.

Perform segmentation (detect masks & classes) in the image(s) passed as inputs.

**Parameters:**

model ([PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel)) : The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel).

image_processor ([BaseImageProcessor](/docs/transformers/main/en/main_classes/image_processor#transformers.BaseImageProcessor)) : The image processor that will be used by the pipeline to encode data for the model. This object inherits from [BaseImageProcessor](/docs/transformers/main/en/main_classes/image_processor#transformers.BaseImageProcessor).

modelcard (`str` or `ModelCard`, *optional*) : Model card attributed to the model for this pipeline.

task (`str`, defaults to `""`) : A task-identifier for the pipeline.

num_workers (`int`, *optional*, defaults to 8) : When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the number of workers to be used.

batch_size (`int`, *optional*, defaults to 1) : When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the size of the batch to use, for inference this is not always beneficial, please read [Batching with pipelines](https://huggingface.co/transformers/main_classes/pipelines.html#pipeline-batching) .

args_parser ([ArgumentHandler](/docs/transformers/main/en/internal/pipelines_utils#transformers.pipelines.ArgumentHandler), *optional*) : Reference to the object in charge of parsing supplied pipeline parameters.

device (`int`, *optional*, defaults to -1) : Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, a positive will run the model on the associated CUDA device id. You can pass native `torch.device` or a `str` too

dtype (`str` or `torch.dtype`, *optional*) : Sent directly as `model_kwargs` (just a simpler shortcut) to use the available precision for this model (`torch.float16`, `torch.bfloat16`, ... or `"auto"`)

binary_output (`bool`, *optional*, defaults to `False`) : Flag indicating if the output the pipeline should happen in a serialized format (i.e., pickle) or as the raw output data e.g. text.

**Returns:**

If the input is a single image, will return a list of dictionaries, if the input is a list of several images,
will return a list of list of dictionaries corresponding to each image.

The dictionaries contain the mask, label and score (where applicable) of each detected object and contains
the following keys:

- **label** (`str`) -- The class label identified by the model.
- **mask** (`PIL.Image`) -- A binary mask of the detected object as a Pil Image of shape (width, height) of
  the original image. Returns a mask filled with zeros if no object is found.
- **score** (*optional* `float`) -- Optionally, when the model is capable of estimating a confidence of the
  "object" described by the label and the mask.

### ImageToImagePipeline[[transformers.ImageToImagePipeline]]

#### transformers.ImageToImagePipeline[[transformers.ImageToImagePipeline]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/image_to_image.py#L40)

Image to Image pipeline using any `AutoModelForImageToImage`. This pipeline generates an image based on a previous
image input.

Example:

```python
>>> from PIL import Image
>>> import httpx
>>> import io

>>> from transformers import pipeline

>>> upscaler = pipeline("image-to-image", model="caidas/swin2SR-classical-sr-x2-64")
>>> img = Image.open(io.BytesIO(httpx.get("http://images.cocodataset.org/val2017/000000039769.jpg").content))
>>> img = img.resize((64, 64))
>>> upscaled_img = upscaler(img)
>>> img.size
(64, 64)

>>> upscaled_img.size
(144, 144)
```

This image to image pipeline can currently be loaded from [pipeline()](/docs/transformers/main/en/main_classes/pipelines#transformers.pipeline) using the following task identifier:
`"image-to-image"`.

See the list of available models on [huggingface.co/models](https://huggingface.co/models?filter=image-to-image).

__call__transformers.ImageToImagePipeline.__call__https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/image_to_image.py#L97[{"name": "images", "val": ": typing.Union[str, list[str], ForwardRef('Image.Image'), list['Image.Image']]"}, {"name": "**kwargs", "val": ": typing.Any"}]- **images** (`str`, `list[str]`, `PIL.Image` or `list[PIL.Image]`) --
  The pipeline handles three types of images:

  - A string containing a http link pointing to an image
  - A string containing a local path to an image
  - An image loaded in PIL directly

  The pipeline accepts either a single image or a batch of images, which must then be passed as a string.
  Images in a batch must all be in the same format: all as http links, all as local paths, or all as PIL
  images.
- **timeout** (`float`, *optional*, defaults to None) --
  The maximum time in seconds to wait for fetching images from the web. If None, no timeout is used and
  the call may block forever.0An image (Image.Image) or a list of images (list["Image.Image"]) containing result(s). If the input is a
single image, the return will be also a single image, if the input is a list of several images, it will
return a list of transformed images.

Transform the image(s) passed as inputs.

**Parameters:**

model ([PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel)) : The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel).

image_processor ([BaseImageProcessor](/docs/transformers/main/en/main_classes/image_processor#transformers.BaseImageProcessor)) : The image processor that will be used by the pipeline to encode data for the model. This object inherits from [BaseImageProcessor](/docs/transformers/main/en/main_classes/image_processor#transformers.BaseImageProcessor).

modelcard (`str` or `ModelCard`, *optional*) : Model card attributed to the model for this pipeline.

task (`str`, defaults to `""`) : A task-identifier for the pipeline.

num_workers (`int`, *optional*, defaults to 8) : When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the number of workers to be used.

batch_size (`int`, *optional*, defaults to 1) : When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the size of the batch to use, for inference this is not always beneficial, please read [Batching with pipelines](https://huggingface.co/transformers/main_classes/pipelines.html#pipeline-batching) .

args_parser ([ArgumentHandler](/docs/transformers/main/en/internal/pipelines_utils#transformers.pipelines.ArgumentHandler), *optional*) : Reference to the object in charge of parsing supplied pipeline parameters.

device (`int`, *optional*, defaults to -1) : Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, a positive will run the model on the associated CUDA device id. You can pass native `torch.device` or a `str` too

dtype (`str` or `torch.dtype`, *optional*) : Sent directly as `model_kwargs` (just a simpler shortcut) to use the available precision for this model (`torch.float16`, `torch.bfloat16`, ... or `"auto"`)

binary_output (`bool`, *optional*, defaults to `False`) : Flag indicating if the output the pipeline should happen in a serialized format (i.e., pickle) or as the raw output data e.g. text.

**Returns:**

An image (Image.Image) or a list of images (list["Image.Image"]) containing result(s). If the input is a
single image, the return will be also a single image, if the input is a list of several images, it will
return a list of transformed images.

### KeypointMatchingPipeline[[transformers.KeypointMatchingPipeline]]

#### transformers.KeypointMatchingPipeline[[transformers.KeypointMatchingPipeline]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/keypoint_matching.py#L69)

Keypoint matching pipeline using any `AutoModelForKeypointMatching`. This pipeline matches keypoints between two images.

__call__transformers.KeypointMatchingPipeline.__call__https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/keypoint_matching.py#L98[{"name": "inputs", "val": ": list[collections.abc.Sequence[typing.Union[ForwardRef('Image.Image'), str]]] | collections.abc.Sequence[typing.Union[ForwardRef('Image.Image'), str]]"}, {"name": "threshold", "val": ": float = 0.0"}, {"name": "**kwargs", "val": ": typing.Any"}]- **inputs** (`str`, `list[str]`, `PIL.Image` or `list[PIL.Image]`) --
  The pipeline handles three types of images:

  - A string containing a http link pointing to an image
  - A string containing a local path to an image
  - An image loaded in PIL directly

  The pipeline accepts either a single pair of images or a batch of image pairs, which must then be passed as a string.
  Images in a batch must all be in the same format: all as http links, all as local paths, or all as PIL
  images.

- **threshold** (`float`, *optional*, defaults to 0.0) --
  The threshold to use for keypoint matching. Keypoints matched with a lower matching score will be filtered out.
  A value of 0 means that all matched keypoints will be returned.

- **kwargs** --
  `timeout (`float`, *optional*, defaults to None)`
  The maximum time in seconds to wait for fetching images from the web. If None, no timeout is set and
  the call may block forever.0Union[list[Match], list[list[Match]]]A list of matches or a list if a single image pair is provided, or of lists of matches if a batch
of image pairs is provided. Each match is a dictionary containing the following keys:

- **keypoint_image_0** (`Keypoint`): The keypoint in the first image (x, y coordinates).
- **keypoint_image_1** (`Keypoint`): The keypoint in the second image (x, y coordinates).
- **score** (`float`): The matching score between the two keypoints.

Find matches between keypoints in two images.

**Parameters:**

inputs (`str`, `list[str]`, `PIL.Image` or `list[PIL.Image]`) : The pipeline handles three types of images:  - A string containing a http link pointing to an image - A string containing a local path to an image - An image loaded in PIL directly  The pipeline accepts either a single pair of images or a batch of image pairs, which must then be passed as a string. Images in a batch must all be in the same format: all as http links, all as local paths, or all as PIL images. 

threshold (`float`, *optional*, defaults to 0.0) : The threshold to use for keypoint matching. Keypoints matched with a lower matching score will be filtered out. A value of 0 means that all matched keypoints will be returned. 

kwargs : `timeout (`float`, *optional*, defaults to None)` The maximum time in seconds to wait for fetching images from the web. If None, no timeout is set and the call may block forever.

**Returns:**

`Union[list[Match], list[list[Match]]]`

A list of matches or a list if a single image pair is provided, or of lists of matches if a batch
of image pairs is provided. Each match is a dictionary containing the following keys:

- **keypoint_image_0** (`Keypoint`): The keypoint in the first image (x, y coordinates).
- **keypoint_image_1** (`Keypoint`): The keypoint in the second image (x, y coordinates).
- **score** (`float`): The matching score between the two keypoints.

### ObjectDetectionPipeline[[transformers.ObjectDetectionPipeline]]

#### transformers.ObjectDetectionPipeline[[transformers.ObjectDetectionPipeline]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/object_detection.py#L26)

Object detection pipeline using any `AutoModelForObjectDetection`. This pipeline predicts bounding boxes of objects
and their classes.

Example:

```python
>>> from transformers import pipeline

>>> detector = pipeline(model="facebook/detr-resnet-50")
>>> detector("https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png")
[{'score': 0.997, 'label': 'bird', 'box': {'xmin': 69, 'ymin': 171, 'xmax': 396, 'ymax': 507}}, {'score': 0.999, 'label': 'bird', 'box': {'xmin': 398, 'ymin': 105, 'xmax': 767, 'ymax': 507}}]

>>> # x, y  are expressed relative to the top left hand corner.
```

Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

This object detection pipeline can currently be loaded from [pipeline()](/docs/transformers/main/en/main_classes/pipelines#transformers.pipeline) using the following task identifier:
`"object-detection"`.

See the list of available models on [huggingface.co/models](https://huggingface.co/models?filter=object-detection).

__call__transformers.ObjectDetectionPipeline.__call__https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/object_detection.py#L81[{"name": "*args", "val": ""}, {"name": "**kwargs", "val": ""}]- **inputs** (`str`, `list[str]`, `PIL.Image` or `list[PIL.Image]`) --
  The pipeline handles three types of images:

  - A string containing an HTTP(S) link pointing to an image
  - A string containing a local path to an image
  - An image loaded in PIL directly

  The pipeline accepts either a single image or a batch of images. Images in a batch must all be in the
  same format: all as HTTP(S) links, all as local paths, or all as PIL images.
- **threshold** (`float`, *optional*, defaults to 0.5) --
  The probability necessary to make a prediction.
- **timeout** (`float`, *optional*, defaults to None) --
  The maximum time in seconds to wait for fetching images from the web. If None, no timeout is set and
  the call may block forever.0A list of dictionaries or a list of list of dictionaries containing the result. If the input is a single
image, will return a list of dictionaries, if the input is a list of several images, will return a list of
list of dictionaries corresponding to each image.

The dictionaries contain the following keys:

- **label** (`str`) -- The class label identified by the model.
- **score** (`float`) -- The score attributed by the model for that label.
- **box** (`list[dict[str, int]]`) -- The bounding box of detected object in image's original size.

Detect objects (bounding boxes & classes) in the image(s) passed as inputs.

**Parameters:**

model ([PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel)) : The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel).

image_processor ([BaseImageProcessor](/docs/transformers/main/en/main_classes/image_processor#transformers.BaseImageProcessor)) : The image processor that will be used by the pipeline to encode data for the model. This object inherits from [BaseImageProcessor](/docs/transformers/main/en/main_classes/image_processor#transformers.BaseImageProcessor).

modelcard (`str` or `ModelCard`, *optional*) : Model card attributed to the model for this pipeline.

task (`str`, defaults to `""`) : A task-identifier for the pipeline.

num_workers (`int`, *optional*, defaults to 8) : When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the number of workers to be used.

batch_size (`int`, *optional*, defaults to 1) : When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the size of the batch to use, for inference this is not always beneficial, please read [Batching with pipelines](https://huggingface.co/transformers/main_classes/pipelines.html#pipeline-batching) .

args_parser ([ArgumentHandler](/docs/transformers/main/en/internal/pipelines_utils#transformers.pipelines.ArgumentHandler), *optional*) : Reference to the object in charge of parsing supplied pipeline parameters.

device (`int`, *optional*, defaults to -1) : Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, a positive will run the model on the associated CUDA device id. You can pass native `torch.device` or a `str` too

dtype (`str` or `torch.dtype`, *optional*) : Sent directly as `model_kwargs` (just a simpler shortcut) to use the available precision for this model (`torch.float16`, `torch.bfloat16`, ... or `"auto"`)

binary_output (`bool`, *optional*, defaults to `False`) : Flag indicating if the output the pipeline should happen in a serialized format (i.e., pickle) or as the raw output data e.g. text.

**Returns:**

A list of dictionaries or a list of list of dictionaries containing the result. If the input is a single
image, will return a list of dictionaries, if the input is a list of several images, will return a list of
list of dictionaries corresponding to each image.

The dictionaries contain the following keys:

- **label** (`str`) -- The class label identified by the model.
- **score** (`float`) -- The score attributed by the model for that label.
- **box** (`list[dict[str, int]]`) -- The bounding box of detected object in image's original size.

### VideoClassificationPipeline[[transformers.VideoClassificationPipeline]]

#### transformers.VideoClassificationPipeline[[transformers.VideoClassificationPipeline]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/video_classification.py#L42)

Video classification pipeline using any `AutoModelForVideoClassification`. This pipeline predicts the class of a
video.

This video classification pipeline can currently be loaded from [pipeline()](/docs/transformers/main/en/main_classes/pipelines#transformers.pipeline) using the following task identifier:
`"video-classification"`.

See the list of available models on
[huggingface.co/models](https://huggingface.co/models?filter=video-classification).

__call__transformers.VideoClassificationPipeline.__call__https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/video_classification.py#L91[{"name": "inputs", "val": ": str | list[str] | None = None"}, {"name": "**kwargs", "val": ""}]- **inputs** (`str`, `list[str]`) --
  The pipeline handles three types of videos:

  - A string containing a http link pointing to a video
  - A string containing a local path to a video

  The pipeline accepts either a single video or a batch of videos, which must then be passed as a string.
  Videos in a batch must all be in the same format: all as http links or all as local paths.
- **top_k** (`int`, *optional*, defaults to 5) --
  The number of top labels that will be returned by the pipeline. If the provided number is higher than
  the number of labels available in the model configuration, it will default to the number of labels.
- **num_frames** (`int`, *optional*, defaults to `self.model.config.num_frames`) --
  The number of frames sampled from the video to run the classification on. If not provided, will default
  to the number of frames specified in the model configuration.
- **frame_sampling_rate** (`int`, *optional*, defaults to 1) --
  The sampling rate used to select frames from the video. If not provided, will default to 1, i.e. every
  frame will be used.
- **function_to_apply(`str`,** *optional*, defaults to "softmax") --
  The function to apply to the model output. By default, the pipeline will apply the softmax function to
  the output of the model. Valid options: ["softmax", "sigmoid", "none"]. Note that passing Python's
  built-in `None` will default to "softmax", so you need to pass the string "none" to disable any
  post-processing.0A list of dictionaries or a list of list of dictionaries containing result. If the input is a single video,
will return a list of `top_k` dictionaries, if the input is a list of several videos, will return a list of list of
`top_k` dictionaries corresponding to the videos.

The dictionaries contain the following keys:

- **label** (`str`) -- The label identified by the model.
- **score** (`int`) -- The score attributed by the model for that label.

Assign labels to the video(s) passed as inputs.

**Parameters:**

model ([PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel)) : The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel).

image_processor ([BaseImageProcessor](/docs/transformers/main/en/main_classes/image_processor#transformers.BaseImageProcessor)) : The image processor that will be used by the pipeline to encode data for the model. This object inherits from [BaseImageProcessor](/docs/transformers/main/en/main_classes/image_processor#transformers.BaseImageProcessor).

modelcard (`str` or `ModelCard`, *optional*) : Model card attributed to the model for this pipeline.

task (`str`, defaults to `""`) : A task-identifier for the pipeline.

num_workers (`int`, *optional*, defaults to 8) : When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the number of workers to be used.

batch_size (`int`, *optional*, defaults to 1) : When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the size of the batch to use, for inference this is not always beneficial, please read [Batching with pipelines](https://huggingface.co/transformers/main_classes/pipelines.html#pipeline-batching) .

args_parser ([ArgumentHandler](/docs/transformers/main/en/internal/pipelines_utils#transformers.pipelines.ArgumentHandler), *optional*) : Reference to the object in charge of parsing supplied pipeline parameters.

device (`int`, *optional*, defaults to -1) : Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, a positive will run the model on the associated CUDA device id. You can pass native `torch.device` or a `str` too

dtype (`str` or `torch.dtype`, *optional*) : Sent directly as `model_kwargs` (just a simpler shortcut) to use the available precision for this model (`torch.float16`, `torch.bfloat16`, ... or `"auto"`)

binary_output (`bool`, *optional*, defaults to `False`) : Flag indicating if the output the pipeline should happen in a serialized format (i.e., pickle) or as the raw output data e.g. text.

**Returns:**

A list of dictionaries or a list of list of dictionaries containing result. If the input is a single video,
will return a list of `top_k` dictionaries, if the input is a list of several videos, will return a list of list of
`top_k` dictionaries corresponding to the videos.

The dictionaries contain the following keys:

- **label** (`str`) -- The label identified by the model.
- **score** (`int`) -- The score attributed by the model for that label.

### ZeroShotImageClassificationPipeline[[transformers.ZeroShotImageClassificationPipeline]]

#### transformers.ZeroShotImageClassificationPipeline[[transformers.ZeroShotImageClassificationPipeline]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/zero_shot_image_classification.py#L30)

Zero shot image classification pipeline using `CLIPModel`. This pipeline predicts the class of an image when you
provide an image and a set of `candidate_labels`.

Example:

```python
>>> from transformers import pipeline

>>> classifier = pipeline(model="google/siglip-so400m-patch14-384")
>>> classifier(
...     "https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png",
...     candidate_labels=["animals", "humans", "landscape"],
... )
[{'score': 0.965, 'label': 'animals'}, {'score': 0.03, 'label': 'humans'}, {'score': 0.005, 'label': 'landscape'}]

>>> classifier(
...     "https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png",
...     candidate_labels=["black and white", "photorealist", "painting"],
... )
[{'score': 0.996, 'label': 'black and white'}, {'score': 0.003, 'label': 'photorealist'}, {'score': 0.0, 'label': 'painting'}]
```

Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

This image classification pipeline can currently be loaded from [pipeline()](/docs/transformers/main/en/main_classes/pipelines#transformers.pipeline) using the following task identifier:
`"zero-shot-image-classification"`.

See the list of available models on
[huggingface.co/models](https://huggingface.co/models?filter=zero-shot-image-classification).

__call__transformers.ZeroShotImageClassificationPipeline.__call__https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/zero_shot_image_classification.py#L84[{"name": "image", "val": ": typing.Union[str, list[str], ForwardRef('Image.Image'), list['Image.Image']]"}, {"name": "candidate_labels", "val": ": list"}, {"name": "**kwargs", "val": ": typing.Any"}]- **image** (`str`, `list[str]`, `PIL.Image` or `list[PIL.Image]`) --
  The pipeline handles three types of images:

  - A string containing a http link pointing to an image
  - A string containing a local path to an image
  - An image loaded in PIL directly

- **candidate_labels** (`list[str]`) --
  The candidate labels for this image. They will be formatted using *hypothesis_template*.

- **hypothesis_template** (`str`, *optional*, defaults to `"This is a photo of {}"`) --
  The format used in conjunction with *candidate_labels* to attempt the image classification by
  replacing the placeholder with the candidate_labels. Pass "{}" if *candidate_labels* are
  already formatted.

- **timeout** (`float`, *optional*, defaults to None) --
  The maximum time in seconds to wait for fetching images from the web. If None, no timeout is set and
  the call may block forever.0A list of dictionaries containing one entry per proposed label. Each dictionary contains the
following keys:
- **label** (`str`) -- One of the suggested *candidate_labels*.
- **score** (`float`) -- The score attributed by the model to that label. It is a value between
  0 and 1, computed as the `softmax` of `logits_per_image`.

Assign labels to the image(s) passed as inputs.

**Parameters:**

model ([PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel)) : The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel).

image_processor ([BaseImageProcessor](/docs/transformers/main/en/main_classes/image_processor#transformers.BaseImageProcessor)) : The image processor that will be used by the pipeline to encode data for the model. This object inherits from [BaseImageProcessor](/docs/transformers/main/en/main_classes/image_processor#transformers.BaseImageProcessor).

modelcard (`str` or `ModelCard`, *optional*) : Model card attributed to the model for this pipeline.

task (`str`, defaults to `""`) : A task-identifier for the pipeline.

num_workers (`int`, *optional*, defaults to 8) : When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the number of workers to be used.

batch_size (`int`, *optional*, defaults to 1) : When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the size of the batch to use, for inference this is not always beneficial, please read [Batching with pipelines](https://huggingface.co/transformers/main_classes/pipelines.html#pipeline-batching) .

args_parser ([ArgumentHandler](/docs/transformers/main/en/internal/pipelines_utils#transformers.pipelines.ArgumentHandler), *optional*) : Reference to the object in charge of parsing supplied pipeline parameters.

device (`int`, *optional*, defaults to -1) : Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, a positive will run the model on the associated CUDA device id. You can pass native `torch.device` or a `str` too

dtype (`str` or `torch.dtype`, *optional*) : Sent directly as `model_kwargs` (just a simpler shortcut) to use the available precision for this model (`torch.float16`, `torch.bfloat16`, ... or `"auto"`)

binary_output (`bool`, *optional*, defaults to `False`) : Flag indicating if the output the pipeline should happen in a serialized format (i.e., pickle) or as the raw output data e.g. text.

**Returns:**

A list of dictionaries containing one entry per proposed label. Each dictionary contains the
following keys:
- **label** (`str`) -- One of the suggested *candidate_labels*.
- **score** (`float`) -- The score attributed by the model to that label. It is a value between
  0 and 1, computed as the `softmax` of `logits_per_image`.

### ZeroShotObjectDetectionPipeline[[transformers.ZeroShotObjectDetectionPipeline]]

#### transformers.ZeroShotObjectDetectionPipeline[[transformers.ZeroShotObjectDetectionPipeline]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/zero_shot_object_detection.py#L23)

Zero shot object detection pipeline using `OwlViTForObjectDetection`. This pipeline predicts bounding boxes of
objects when you provide an image and a set of `candidate_labels`.

Example:

```python
>>> from transformers import pipeline

>>> detector = pipeline(model="google/owlvit-base-patch32", task="zero-shot-object-detection")
>>> detector(
...     "http://images.cocodataset.org/val2017/000000039769.jpg",
...     candidate_labels=["cat", "couch"],
... )
[{'score': 0.287, 'label': 'cat', 'box': {'xmin': 324, 'ymin': 20, 'xmax': 640, 'ymax': 373}}, {'score': 0.254, 'label': 'cat', 'box': {'xmin': 1, 'ymin': 55, 'xmax': 315, 'ymax': 472}}, {'score': 0.121, 'label': 'couch', 'box': {'xmin': 4, 'ymin': 0, 'xmax': 642, 'ymax': 476}}]

>>> detector(
...     "https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png",
...     candidate_labels=["head", "bird"],
... )
[{'score': 0.119, 'label': 'bird', 'box': {'xmin': 71, 'ymin': 170, 'xmax': 410, 'ymax': 508}}]
```

Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

This object detection pipeline can currently be loaded from [pipeline()](/docs/transformers/main/en/main_classes/pipelines#transformers.pipeline) using the following task identifier:
`"zero-shot-object-detection"`.

See the list of available models on
[huggingface.co/models](https://huggingface.co/models?filter=zero-shot-object-detection).

__call__transformers.ZeroShotObjectDetectionPipeline.__call__https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/zero_shot_object_detection.py#L75[{"name": "image", "val": ": typing.Union[str, ForwardRef('Image.Image'), list[dict[str, typing.Any]]]"}, {"name": "candidate_labels", "val": ": str | list[str] | None = None"}, {"name": "**kwargs", "val": ": typing.Any"}]- **image** (`str`, `PIL.Image` or `list[dict[str, Any]]`) --
  The pipeline handles three types of images:

  - A string containing an http url pointing to an image
  - A string containing a local path to an image
  - An image loaded in PIL directly

  You can use this parameter to send directly a list of images, or a dataset or a generator like so:

```python
>>> from transformers import pipeline

>>> detector = pipeline(model="google/owlvit-base-patch32", task="zero-shot-object-detection")
>>> detector(
...     [
...         {
...             "image": "http://images.cocodataset.org/val2017/000000039769.jpg",
...             "candidate_labels": ["cat", "couch"],
...         },
...         {
...             "image": "http://images.cocodataset.org/val2017/000000039769.jpg",
...             "candidate_labels": ["cat", "couch"],
...         },
...     ]
... )
[[{'score': 0.287, 'label': 'cat', 'box': {'xmin': 324, 'ymin': 20, 'xmax': 640, 'ymax': 373}}, {'score': 0.25, 'label': 'cat', 'box': {'xmin': 1, 'ymin': 55, 'xmax': 315, 'ymax': 472}}, {'score': 0.121, 'label': 'couch', 'box': {'xmin': 4, 'ymin': 0, 'xmax': 642, 'ymax': 476}}], [{'score': 0.287, 'label': 'cat', 'box': {'xmin': 324, 'ymin': 20, 'xmax': 640, 'ymax': 373}}, {'score': 0.254, 'label': 'cat', 'box': {'xmin': 1, 'ymin': 55, 'xmax': 315, 'ymax': 472}}, {'score': 0.121, 'label': 'couch', 'box': {'xmin': 4, 'ymin': 0, 'xmax': 642, 'ymax': 476}}]]
```

- **candidate_labels** (`str` or `list[str]` or `list[list[str]]`) --
  What the model should recognize in the image.

- **threshold** (`float`, *optional*, defaults to 0.1) --
  The probability necessary to make a prediction.

- **top_k** (`int`, *optional*, defaults to None) --
  The number of top predictions that will be returned by the pipeline. If the provided number is `None`
  or higher than the number of predictions available, it will default to the number of predictions.

- **timeout** (`float`, *optional*, defaults to None) --
  The maximum time in seconds to wait for fetching images from the web. If None, no timeout is set and
  the call may block forever.0A list of lists containing prediction results, one list per input image. Each list contains dictionaries
with the following keys:

- **label** (`str`) -- Text query corresponding to the found object.
- **score** (`float`) -- Score corresponding to the object (between 0 and 1).
- **box** (`dict[str,int]`) -- Bounding box of the detected object in image's original size. It is a
  dictionary with `x_min`, `x_max`, `y_min`, `y_max` keys.

Detect objects (bounding boxes & classes) in the image(s) passed as inputs.

**Parameters:**

model ([PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel)) : The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel).

image_processor ([BaseImageProcessor](/docs/transformers/main/en/main_classes/image_processor#transformers.BaseImageProcessor)) : The image processor that will be used by the pipeline to encode data for the model. This object inherits from [BaseImageProcessor](/docs/transformers/main/en/main_classes/image_processor#transformers.BaseImageProcessor).

modelcard (`str` or `ModelCard`, *optional*) : Model card attributed to the model for this pipeline.

task (`str`, defaults to `""`) : A task-identifier for the pipeline.

num_workers (`int`, *optional*, defaults to 8) : When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the number of workers to be used.

batch_size (`int`, *optional*, defaults to 1) : When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the size of the batch to use, for inference this is not always beneficial, please read [Batching with pipelines](https://huggingface.co/transformers/main_classes/pipelines.html#pipeline-batching) .

args_parser ([ArgumentHandler](/docs/transformers/main/en/internal/pipelines_utils#transformers.pipelines.ArgumentHandler), *optional*) : Reference to the object in charge of parsing supplied pipeline parameters.

device (`int`, *optional*, defaults to -1) : Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, a positive will run the model on the associated CUDA device id. You can pass native `torch.device` or a `str` too

dtype (`str` or `torch.dtype`, *optional*) : Sent directly as `model_kwargs` (just a simpler shortcut) to use the available precision for this model (`torch.float16`, `torch.bfloat16`, ... or `"auto"`)

binary_output (`bool`, *optional*, defaults to `False`) : Flag indicating if the output the pipeline should happen in a serialized format (i.e., pickle) or as the raw output data e.g. text.

**Returns:**

A list of lists containing prediction results, one list per input image. Each list contains dictionaries
with the following keys:

- **label** (`str`) -- Text query corresponding to the found object.
- **score** (`float`) -- Score corresponding to the object (between 0 and 1).
- **box** (`dict[str,int]`) -- Bounding box of the detected object in image's original size. It is a
  dictionary with `x_min`, `x_max`, `y_min`, `y_max` keys.

## Natural Language Processing

Pipelines available for natural language processing tasks include the following.

### FillMaskPipeline[[transformers.FillMaskPipeline]]

#### transformers.FillMaskPipeline[[transformers.FillMaskPipeline]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/fill_mask.py#L28)

__call__transformers.FillMaskPipeline.__call__https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/fill_mask.py#L234[{"name": "inputs", "val": ": str | list[str]"}, {"name": "**kwargs", "val": ": typing.Any"}]- **inputs** (`str` or `list[str]`) --
  One or several texts (or one list of prompts) with masked tokens.
- **targets** (`str` or `list[str]`, *optional*) --
  When passed, the model will limit the scores to the passed targets instead of looking up in the whole
  vocab. If the provided targets are not in the model vocab, they will be tokenized and the first
  resulting token will be used (with a warning, and that might be slower).
- **top_k** (`int`, *optional*) --
  When passed, overrides the number of predictions to return.0A list or a list of list of `dict`Each result comes as list of dictionaries with the following keys:

- **sequence** (`str`) -- The corresponding input with the mask token prediction.
- **score** (`float`) -- The corresponding probability.
- **token** (`int`) -- The predicted token id (to replace the masked one).
- **token_str** (`str`) -- The predicted token (to replace the masked one).

Fill the masked token in the text(s) given as inputs.

**Parameters:**

model ([PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel)) : The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel).

tokenizer ([PreTrainedTokenizer](/docs/transformers/main/en/main_classes/tokenizer#transformers.PythonBackend)) : The tokenizer that will be used by the pipeline to encode data for the model. This object inherits from [PreTrainedTokenizer](/docs/transformers/main/en/main_classes/tokenizer#transformers.PythonBackend).

modelcard (`str` or `ModelCard`, *optional*) : Model card attributed to the model for this pipeline.

task (`str`, defaults to `""`) : A task-identifier for the pipeline.

num_workers (`int`, *optional*, defaults to 8) : When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the number of workers to be used.

batch_size (`int`, *optional*, defaults to 1) : When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the size of the batch to use, for inference this is not always beneficial, please read [Batching with pipelines](https://huggingface.co/transformers/main_classes/pipelines.html#pipeline-batching) .

args_parser ([ArgumentHandler](/docs/transformers/main/en/internal/pipelines_utils#transformers.pipelines.ArgumentHandler), *optional*) : Reference to the object in charge of parsing supplied pipeline parameters.

device (`int`, *optional*, defaults to -1) : Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, a positive will run the model on the associated CUDA device id. You can pass native `torch.device` or a `str` too

dtype (`str` or `torch.dtype`, *optional*) : Sent directly as `model_kwargs` (just a simpler shortcut) to use the available precision for this model (`torch.float16`, `torch.bfloat16`, ... or `"auto"`)

binary_output (`bool`, *optional*, defaults to `False`) : Flag indicating if the output the pipeline should happen in a serialized format (i.e., pickle) or as the raw output data e.g. text.

top_k (`int`, *optional*, defaults to 5) : The number of predictions to return.

targets (`str` or `list[str]`, *optional*) : When passed, the model will limit the scores to the passed targets instead of looking up in the whole vocab. If the provided targets are not in the model vocab, they will be tokenized and the first resulting token will be used (with a warning, and that might be slower).

tokenizer_kwargs (`dict`, *optional*) : Additional dictionary of keyword arguments passed along to the tokenizer.

**Returns:**

`A list or a list of list of `dict``

Each result comes as list of dictionaries with the following keys:

- **sequence** (`str`) -- The corresponding input with the mask token prediction.
- **score** (`float`) -- The corresponding probability.
- **token** (`int`) -- The predicted token id (to replace the masked one).
- **token_str** (`str`) -- The predicted token (to replace the masked one).

### QuestionAnsweringPipeline[[transformers.QuestionAnsweringPipeline]]

#### transformers.QuestionAnsweringPipeline[[transformers.QuestionAnsweringPipeline]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/question_answering.py#L230)

Question Answering pipeline using any `ModelForQuestionAnswering`. See the [question answering
examples](../task_summary#question-answering) for more information.

Example:

```python
>>> from transformers import pipeline

>>> oracle = pipeline(model="deepset/roberta-base-squad2")
>>> oracle(question="Where do I live?", context="My name is Wolfgang and I live in Berlin")
{'score': 0.9191, 'start': 34, 'end': 40, 'answer': 'Berlin'}
```

Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

This question answering pipeline can currently be loaded from [pipeline()](/docs/transformers/main/en/main_classes/pipelines#transformers.pipeline) using the following task identifier:
`"question-answering"`.

The models that this pipeline can use are models that have been fine-tuned on a question answering task. See the
up-to-date list of available models on
[huggingface.co/models](https://huggingface.co/models?filter=question-answering).

__call__transformers.QuestionAnsweringPipeline.__call__https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/question_answering.py#L339[{"name": "*args", "val": ""}, {"name": "**kwargs", "val": ""}]- **question** (`str` or `list[str]`) --
  One or several question(s) (must be used in conjunction with the `context` argument).
- **context** (`str` or `list[str]`) --
  One or several context(s) associated with the question(s) (must be used in conjunction with the
  `question` argument).
- **top_k** (`int`, *optional*, defaults to 1) --
  The number of answers to return (will be chosen by order of likelihood). Note that we return less than
  top_k answers if there are not enough options available within the context.
- **doc_stride** (`int`, *optional*, defaults to 128) --
  If the context is too long to fit with the question for the model, it will be split in several chunks
  with some overlap. This argument controls the size of that overlap.
- **max_answer_len** (`int`, *optional*, defaults to 15) --
  The maximum length of predicted answers (e.g., only answers with a shorter length are considered).
- **max_seq_len** (`int`, *optional*, defaults to 384) --
  The maximum length of the total sentence (context + question) in tokens of each chunk passed to the
  model. The context will be split in several chunks (using `doc_stride` as overlap) if needed.
- **max_question_len** (`int`, *optional*, defaults to 64) --
  The maximum length of the question after tokenization. It will be truncated if needed.
- **handle_impossible_answer** (`bool`, *optional*, defaults to `False`) --
  Whether or not we accept impossible as an answer.
- **align_to_words** (`bool`, *optional*, defaults to `True`) --
  Attempts to align the answer to real words. Improves quality on space separated languages. Might hurt on
  non-space-separated languages (like Japanese or Chinese)0A `dict` or a list of `dict`Each result comes as a dictionary with the following keys:

- **score** (`float`) -- The probability associated to the answer.
- **start** (`int`) -- The character start index of the answer (in the tokenized version of the input).
- **end** (`int`) -- The character end index of the answer (in the tokenized version of the input).
- **answer** (`str`) -- The answer to the question.

Answer the question(s) given as inputs by using the context(s).

**Parameters:**

model ([PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel)) : The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel).

tokenizer ([PreTrainedTokenizer](/docs/transformers/main/en/main_classes/tokenizer#transformers.PythonBackend)) : The tokenizer that will be used by the pipeline to encode data for the model. This object inherits from [PreTrainedTokenizer](/docs/transformers/main/en/main_classes/tokenizer#transformers.PythonBackend).

modelcard (`str` or `ModelCard`, *optional*) : Model card attributed to the model for this pipeline.

task (`str`, defaults to `""`) : A task-identifier for the pipeline.

num_workers (`int`, *optional*, defaults to 8) : When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the number of workers to be used.

batch_size (`int`, *optional*, defaults to 1) : When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the size of the batch to use, for inference this is not always beneficial, please read [Batching with pipelines](https://huggingface.co/transformers/main_classes/pipelines.html#pipeline-batching) .

args_parser ([ArgumentHandler](/docs/transformers/main/en/internal/pipelines_utils#transformers.pipelines.ArgumentHandler), *optional*) : Reference to the object in charge of parsing supplied pipeline parameters.

device (`int`, *optional*, defaults to -1) : Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, a positive will run the model on the associated CUDA device id. You can pass native `torch.device` or a `str` too

dtype (`str` or `torch.dtype`, *optional*) : Sent directly as `model_kwargs` (just a simpler shortcut) to use the available precision for this model (`torch.float16`, `torch.bfloat16`, ... or `"auto"`)

binary_output (`bool`, *optional*, defaults to `False`) : Flag indicating if the output the pipeline should happen in a serialized format (i.e., pickle) or as the raw output data e.g. text.

**Returns:**

`A `dict` or a list of `dict``

Each result comes as a dictionary with the following keys:

- **score** (`float`) -- The probability associated to the answer.
- **start** (`int`) -- The character start index of the answer (in the tokenized version of the input).
- **end** (`int`) -- The character end index of the answer (in the tokenized version of the input).
- **answer** (`str`) -- The answer to the question.
#### create_sample[[transformers.QuestionAnsweringPipeline.create_sample]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/question_answering.py#L277)

QuestionAnsweringPipeline leverages the `SquadExample` internally. This helper method encapsulate all the
logic for converting question(s) and context(s) to `SquadExample`.

We currently support extractive question answering.

**Parameters:**

question (`str` or `list[str]`) : The question(s) asked.

context (`str` or `list[str]`) : The context(s) in which we will look for the answer.

**Returns:**

`One or a list of `SquadExample``

The corresponding `SquadExample` grouping question and context.
#### span_to_answer[[transformers.QuestionAnsweringPipeline.span_to_answer]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/question_answering.py#L644)

When decoding from token probabilities, this method maps token indexes to actual word in the initial context.

**Parameters:**

text (`str`) : The actual context to extract the answer from.

start (`int`) : The answer starting token index.

end (`int`) : The answer end token index.

**Returns:**

`Dictionary like `{'answer'`

str, 'start': int, 'end': int}`

### SummarizationPipeline[[transformers.SummarizationPipeline]]

#### transformers.SummarizationPipeline[[transformers.SummarizationPipeline]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/deprecated/text2text_generation.py#L233)

Summarize news articles and other documents.

This summarizing pipeline can currently be loaded from [pipeline()](/docs/transformers/main/en/main_classes/pipelines#transformers.pipeline) using the following task identifier:
`"summarization"`.

The models that this pipeline can use are models that have been fine-tuned on a summarization task, which is
currently, '*bart-large-cnn*', '*google-t5/t5-small*', '*google-t5/t5-base*', '*google-t5/t5-large*', '*google-t5/t5-3b*', '*google-t5/t5-11b*'. See the up-to-date
list of available models on [huggingface.co/models](https://huggingface.co/models?filter=summarization). For a list
of available parameters, see the [following
documentation](https://huggingface.co/docs/transformers/en/main_classes/text_generation#transformers.generation.GenerationMixin.generate)

Unless the model you're using explicitly sets these generation parameters in its configuration files
(`generation_config.json`), the following default values will be used:
- max_new_tokens: 256
- num_beams: 4

Usage:

```python
# use bart
summarizer = pipeline("summarization")
summarizer("An apple a day, keeps the doctor away", min_length=5, max_length=20)

```

__call__transformers.SummarizationPipeline.__call__https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/deprecated/text2text_generation.py#L271[{"name": "*args", "val": ""}, {"name": "**kwargs", "val": ""}]- **documents** (*str* or `list[str]`) --
  One or several articles (or one list of articles) to summarize.
- **return_text** (`bool`, *optional*, defaults to `True`) --
  Whether or not to include the decoded texts in the outputs
- **return_tensors** (`bool`, *optional*, defaults to `False`) --
  Whether or not to include the tensors of predictions (as token indices) in the outputs.
- **clean_up_tokenization_spaces** (`bool`, *optional*, defaults to `False`) --
  Whether or not to clean up the potential extra spaces in the text output.
- **generate_kwargs** --
  Additional keyword arguments to pass along to the generate method of the model (see the generate method
  [here](./text_generation)).0A list or a list of list of `dict`Each result comes as a dictionary with the following keys:

- **summary_text** (`str`, present when `return_text=True`) -- The summary of the corresponding input.
- **summary_token_ids** (`torch.Tensor`, present when `return_tensors=True`) -- The token
  ids of the summary.

Summarize the text(s) given as inputs.

**Parameters:**

model ([PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel)) : The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel).

tokenizer ([PreTrainedTokenizer](/docs/transformers/main/en/main_classes/tokenizer#transformers.PythonBackend)) : The tokenizer that will be used by the pipeline to encode data for the model. This object inherits from [PreTrainedTokenizer](/docs/transformers/main/en/main_classes/tokenizer#transformers.PythonBackend).

modelcard (`str` or `ModelCard`, *optional*) : Model card attributed to the model for this pipeline.

task (`str`, defaults to `""`) : A task-identifier for the pipeline.

num_workers (`int`, *optional*, defaults to 8) : When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the number of workers to be used.

batch_size (`int`, *optional*, defaults to 1) : When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the size of the batch to use, for inference this is not always beneficial, please read [Batching with pipelines](https://huggingface.co/transformers/main_classes/pipelines.html#pipeline-batching) .

args_parser ([ArgumentHandler](/docs/transformers/main/en/internal/pipelines_utils#transformers.pipelines.ArgumentHandler), *optional*) : Reference to the object in charge of parsing supplied pipeline parameters.

device (`int`, *optional*, defaults to -1) : Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, a positive will run the model on the associated CUDA device id. You can pass native `torch.device` or a `str` too

dtype (`str` or `torch.dtype`, *optional*) : Sent directly as `model_kwargs` (just a simpler shortcut) to use the available precision for this model (`torch.float16`, `torch.bfloat16`, ... or `"auto"`)

binary_output (`bool`, *optional*, defaults to `False`) : Flag indicating if the output the pipeline should happen in a serialized format (i.e., pickle) or as the raw output data e.g. text.

**Returns:**

`A list or a list of list of `dict``

Each result comes as a dictionary with the following keys:

- **summary_text** (`str`, present when `return_text=True`) -- The summary of the corresponding input.
- **summary_token_ids** (`torch.Tensor`, present when `return_tensors=True`) -- The token
  ids of the summary.

### TableQuestionAnsweringPipeline[[transformers.TableQuestionAnsweringPipeline]]

#### transformers.TableQuestionAnsweringPipeline[[transformers.TableQuestionAnsweringPipeline]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/table_question_answering.py#L78)

Table Question Answering pipeline using a `ModelForTableQuestionAnswering`. This pipeline is only available in
PyTorch.

Unless the model you're using explicitly sets these generation parameters in its configuration files
(`generation_config.json`), the following default values will be used:
- max_new_tokens: 256

Example:

```python
>>> from transformers import pipeline

>>> oracle = pipeline(model="google/tapas-base-finetuned-wtq")
>>> table = {
...     "Repository": ["Transformers", "Datasets", "Tokenizers"],
...     "Stars": ["36542", "4512", "3934"],
...     "Contributors": ["651", "77", "34"],
...     "Programming language": ["Python", "Python", "Rust, Python and NodeJS"],
... }
>>> oracle(query="How many stars does the transformers repository have?", table=table)
{'answer': 'AVERAGE > 36542', 'coordinates': [(0, 1)], 'cells': ['36542'], 'aggregator': 'AVERAGE'}
```

Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

This tabular question answering pipeline can currently be loaded from [pipeline()](/docs/transformers/main/en/main_classes/pipelines#transformers.pipeline) using the following task
identifier: `"table-question-answering"`.

The models that this pipeline can use are models that have been fine-tuned on a tabular question answering task.
See the up-to-date list of available models on
[huggingface.co/models](https://huggingface.co/models?filter=table-question-answering).

__call__transformers.TableQuestionAnsweringPipeline.__call__https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/table_question_answering.py#L208[{"name": "*args", "val": ""}, {"name": "**kwargs", "val": ""}]- **table** (`pd.DataFrame` or `Dict`) --
  Pandas DataFrame or dictionary that will be converted to a DataFrame containing all the table values.
  See above for an example of dictionary.
- **query** (`str` or `list[str]`) --
  Query or list of queries that will be sent to the model alongside the table.
- **sequential** (`bool`, *optional*, defaults to `False`) --
  Whether to do inference sequentially or as a batch. Batching is faster, but models like SQA require the
  inference to be done sequentially to extract relations within sequences, given their conversational
  nature.
- **padding** (`bool`, `str` or [PaddingStrategy](/docs/transformers/main/en/internal/file_utils#transformers.utils.PaddingStrategy), *optional*, defaults to `False`) --
  Activates and controls padding. Accepts the following values:

  - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
    sequence if provided).
  - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
    acceptable input length for the model if that argument is not provided.
  - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
    lengths).

- **truncation** (`bool`, `str` or `TapasTruncationStrategy`, *optional*, defaults to `False`) --
  Activates and controls truncation. Accepts the following values:

  - `True` or `'drop_rows_to_fit'`: Truncate to a maximum length specified with the argument `max_length`
    or to the maximum acceptable input length for the model if that argument is not provided. This will
    truncate row by row, removing rows from the table.
  - `False` or `'do_not_truncate'` (default): No truncation (i.e., can output batch with sequence lengths
    greater than the model maximum admissible input size).0A dictionary or a list of dictionaries containing resultsEach result is a dictionary with the following
keys:

- **answer** (`str`) -- The answer of the query given the table. If there is an aggregator, the answer will
  be preceded by `AGGREGATOR >`.
- **coordinates** (`list[tuple[int, int]]`) -- Coordinates of the cells of the answers.
- **cells** (`list[str]`) -- List of strings made up of the answer cell values.
- **aggregator** (`str`) -- If the model has an aggregator, this returns the aggregator.

Answers queries according to a table. The pipeline accepts several types of inputs which are detailed below:

- `pipeline(table, query)`
- `pipeline(table, [query])`
- `pipeline(table=table, query=query)`
- `pipeline(table=table, query=[query])`
- `pipeline({"table": table, "query": query})`
- `pipeline({"table": table, "query": [query]})`
- `pipeline([{"table": table, "query": query}, {"table": table, "query": query}])`

The `table` argument should be a dict or a DataFrame built from that dict, containing the whole table:

Example:

```python
data = {
    "actors": ["brad pitt", "leonardo di caprio", "george clooney"],
    "age": ["56", "45", "59"],
    "number of movies": ["87", "53", "69"],
    "date of birth": ["7 february 1967", "10 june 1996", "28 november 1967"],
}
```

This dictionary can be passed in as such, or can be converted to a pandas DataFrame:

Example:

```python
import pandas as pd

table = pd.DataFrame.from_dict(data)
```

**Parameters:**

model ([PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel)) : The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel).

tokenizer ([PreTrainedTokenizer](/docs/transformers/main/en/main_classes/tokenizer#transformers.PythonBackend)) : The tokenizer that will be used by the pipeline to encode data for the model. This object inherits from [PreTrainedTokenizer](/docs/transformers/main/en/main_classes/tokenizer#transformers.PythonBackend).

modelcard (`str` or `ModelCard`, *optional*) : Model card attributed to the model for this pipeline.

task (`str`, defaults to `""`) : A task-identifier for the pipeline.

num_workers (`int`, *optional*, defaults to 8) : When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the number of workers to be used.

batch_size (`int`, *optional*, defaults to 1) : When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the size of the batch to use, for inference this is not always beneficial, please read [Batching with pipelines](https://huggingface.co/transformers/main_classes/pipelines.html#pipeline-batching) .

args_parser ([ArgumentHandler](/docs/transformers/main/en/internal/pipelines_utils#transformers.pipelines.ArgumentHandler), *optional*) : Reference to the object in charge of parsing supplied pipeline parameters.

device (`int`, *optional*, defaults to -1) : Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, a positive will run the model on the associated CUDA device id. You can pass native `torch.device` or a `str` too

dtype (`str` or `torch.dtype`, *optional*) : Sent directly as `model_kwargs` (just a simpler shortcut) to use the available precision for this model (`torch.float16`, `torch.bfloat16`, ... or `"auto"`)

binary_output (`bool`, *optional*, defaults to `False`) : Flag indicating if the output the pipeline should happen in a serialized format (i.e., pickle) or as the raw output data e.g. text.

**Returns:**

`A dictionary or a list of dictionaries containing results`

Each result is a dictionary with the following
keys:

- **answer** (`str`) -- The answer of the query given the table. If there is an aggregator, the answer will
  be preceded by `AGGREGATOR >`.
- **coordinates** (`list[tuple[int, int]]`) -- Coordinates of the cells of the answers.
- **cells** (`list[str]`) -- List of strings made up of the answer cell values.
- **aggregator** (`str`) -- If the model has an aggregator, this returns the aggregator.

### TextClassificationPipeline[[transformers.TextClassificationPipeline]]

#### transformers.TextClassificationPipeline[[transformers.TextClassificationPipeline]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/text_classification.py#L46)

Text classification pipeline using any `ModelForSequenceClassification`. See the [sequence classification
examples](../task_summary#sequence-classification) for more information.

Example:

```python
>>> from transformers import pipeline

>>> classifier = pipeline(model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")
>>> classifier("This movie is disgustingly good !")
[{'label': 'POSITIVE', 'score': 1.0}]

>>> classifier("Director tried too much.")
[{'label': 'NEGATIVE', 'score': 0.996}]
```

Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

This text classification pipeline can currently be loaded from [pipeline()](/docs/transformers/main/en/main_classes/pipelines#transformers.pipeline) using the following task identifier:
`"sentiment-analysis"` (for classifying sequences according to positive or negative sentiments).

If multiple classification labels are available (`model.config.num_labels >= 2`), the pipeline will run a softmax
over the results. If there is a single label, the pipeline will run a sigmoid over the result. In case of regression
tasks (`model.config.problem_type == "regression"`), will not apply any function on the output.

The models that this pipeline can use are models that have been fine-tuned on a sequence classification task. See
the up-to-date list of available models on
[huggingface.co/models](https://huggingface.co/models?filter=text-classification).

__call__transformers.TextClassificationPipeline.__call__https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/text_classification.py#L121[{"name": "inputs", "val": ": str | list[str] | dict[str, str] | list[dict[str, str]]"}, {"name": "**kwargs", "val": ": typing.Any"}]- **inputs** (`str` or `list[str]` or `dict[str]`, or `list[dict[str]]`) --
  One or several texts to classify. In order to use text pairs for your classification, you can send a
  dictionary containing `{"text", "text_pair"}` keys, or a list of those.
- **top_k** (`int`, *optional*, defaults to `1`) --
  How many results to return.
- **function_to_apply** (`str`, *optional*, defaults to `"default"`) --
  The function to apply to the model outputs in order to retrieve the scores. Accepts four different
  values:

  If this argument is not specified, then it will apply the following functions according to the number
  of labels:

  - If problem type is regression, will not apply any function on the output.
  - If the model has a single label, will apply the sigmoid function on the output.
  - If the model has several labels, will apply the softmax function on the output.

  Possible values are:

  - `"sigmoid"`: Applies the sigmoid function on the output.
  - `"softmax"`: Applies the softmax function on the output.
  - `"none"`: Does not apply any function on the output.0A list of `dict`Each result comes as list of dictionaries with the following keys:

- **label** (`str`) -- The label predicted.
- **score** (`float`) -- The corresponding probability.

If `top_k` is used, one such dictionary is returned per label.

Classify the text(s) given as inputs.

**Parameters:**

model ([PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel)) : The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel).

tokenizer ([PreTrainedTokenizer](/docs/transformers/main/en/main_classes/tokenizer#transformers.PythonBackend)) : The tokenizer that will be used by the pipeline to encode data for the model. This object inherits from [PreTrainedTokenizer](/docs/transformers/main/en/main_classes/tokenizer#transformers.PythonBackend).

modelcard (`str` or `ModelCard`, *optional*) : Model card attributed to the model for this pipeline.

task (`str`, defaults to `""`) : A task-identifier for the pipeline.

num_workers (`int`, *optional*, defaults to 8) : When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the number of workers to be used.

batch_size (`int`, *optional*, defaults to 1) : When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the size of the batch to use, for inference this is not always beneficial, please read [Batching with pipelines](https://huggingface.co/transformers/main_classes/pipelines.html#pipeline-batching) .

args_parser ([ArgumentHandler](/docs/transformers/main/en/internal/pipelines_utils#transformers.pipelines.ArgumentHandler), *optional*) : Reference to the object in charge of parsing supplied pipeline parameters.

device (`int`, *optional*, defaults to -1) : Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, a positive will run the model on the associated CUDA device id. You can pass native `torch.device` or a `str` too

dtype (`str` or `torch.dtype`, *optional*) : Sent directly as `model_kwargs` (just a simpler shortcut) to use the available precision for this model (`torch.float16`, `torch.bfloat16`, ... or `"auto"`)

binary_output (`bool`, *optional*, defaults to `False`) : Flag indicating if the output the pipeline should happen in a serialized format (i.e., pickle) or as the raw output data e.g. text.

return_all_scores (`bool`, *optional*, defaults to `False`) : Whether to return all prediction scores or just the one of the predicted class.

function_to_apply (`str`, *optional*, defaults to `"default"`) : The function to apply to the model outputs in order to retrieve the scores. Accepts four different values:  - `"default"`: if the model has a single label, will apply the sigmoid function on the output. If the model has several labels, will apply the softmax function on the output. In case of regression tasks, will not apply any function on the output. - `"sigmoid"`: Applies the sigmoid function on the output. - `"softmax"`: Applies the softmax function on the output. - `"none"`: Does not apply any function on the output.

**Returns:**

`A list of `dict``

Each result comes as list of dictionaries with the following keys:

- **label** (`str`) -- The label predicted.
- **score** (`float`) -- The corresponding probability.

If `top_k` is used, one such dictionary is returned per label.

### TextGenerationPipeline[[transformers.TextGenerationPipeline]]

#### transformers.TextGenerationPipeline[[transformers.TextGenerationPipeline]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/text_generation.py#L23)

Language generation pipeline using any `ModelWithLMHead` or `ModelForCausalLM`. This pipeline predicts the words
that will follow a specified text prompt. When the underlying model is a conversational model, it can also accept
one or more chats, in which case the pipeline will operate in chat mode and will continue the chat(s) by adding
its response(s). Each chat takes the form of a list of dicts, where each dict contains "role" and "content" keys.

Unless the model you're using explicitly sets these generation parameters in its configuration files
(`generation_config.json`), the following default values will be used:
- max_new_tokens: 256
- do_sample: True
- temperature: 0.7

Examples:

```python
>>> from transformers import pipeline

>>> generator = pipeline(model="openai-community/gpt2")
>>> generator("I can't believe you did such a ", do_sample=False)
[{'generated_text': "I can't believe you did such a icky thing to me. I'm so sorry. I'm so sorry. I'm so sorry. I'm so sorry. I'm so sorry. I'm so sorry. I'm so sorry. I"}]

>>> # These parameters will return suggestions, and only the newly created text making it easier for prompting suggestions.
>>> outputs = generator("My tart needs some", num_return_sequences=4, return_full_text=False)
```

```python
>>> from transformers import pipeline

>>> generator = pipeline(model="HuggingFaceH4/zephyr-7b-beta")
>>> # Zephyr-beta is a conversational model, so let's pass it a chat instead of a single string
>>> generator([{"role": "user", "content": "What is the capital of France? Answer in one word."}], do_sample=False, max_new_tokens=2)
[{'generated_text': [{'role': 'user', 'content': 'What is the capital of France? Answer in one word.'}, {'role': 'assistant', 'content': 'Paris'}]}]
```

Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial). You can pass text
generation parameters to this pipeline to control stopping criteria, decoding strategy, and more. Learn more about
text generation parameters in [Text generation strategies](../generation_strategies) and [Text
generation](text_generation).

This language generation pipeline can currently be loaded from [pipeline()](/docs/transformers/main/en/main_classes/pipelines#transformers.pipeline) using the following task identifier:
`"text-generation"`.

The models that this pipeline can use are models that have been trained with an autoregressive language modeling
objective. See the list of available [text completion models](https://huggingface.co/models?filter=text-generation)
and the list of [conversational models](https://huggingface.co/models?other=conversational)
on [huggingface.co/models].

__call__transformers.TextGenerationPipeline.__call__https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/text_generation.py#L243[{"name": "text_inputs", "val": ""}, {"name": "**kwargs", "val": ""}]- **text_inputs** (`str`, `list[str]`, `ChatType`, or `list[ChatType]`) --
  One or several prompts (or one list of prompts) to complete. If strings or a list of string are
  passed, this pipeline will continue each prompt. Alternatively, a "chat", in the form of a list
  of dicts with "role" and "content" keys, can be passed, or a list of such chats. When chats are passed,
  the model's chat template will be used to format them before passing them to the model.
- **return_tensors** (`bool`, *optional*, defaults to `False`) --
  Returns the tensors of predictions (as token indices) in the outputs. If set to
  `True`, the decoded text is not returned.
- **return_text** (`bool`, *optional*) --
  Returns the decoded texts in the outputs.
- **return_full_text** (`bool`, *optional*, defaults to `True`) --
  If set to `False` only added text is returned, otherwise the full text is returned. Cannot be
  specified at the same time as `return_text`.
- **clean_up_tokenization_spaces** (`bool`, *optional*, defaults to `True`) --
  Whether or not to clean up the potential extra spaces in the text output.
- **continue_final_message(** `bool`, *optional*) -- This indicates that you want the model to continue the
  last message in the input chat rather than starting a new one, allowing you to "prefill" its response.
  By default this is `True` when the final message in the input chat has the `assistant` role and
  `False` otherwise, but you can manually override that behaviour by setting this flag.
- **prefix** (`str`, *optional*) --
  Prefix added to prompt.
- **handle_long_generation** (`str`, *optional*) --
  By default, this pipelines does not handle long generation (ones that exceed in one form or the other
  the model maximum length). There is no perfect way to address this (more info
  :https://github.com/huggingface/transformers/issues/14033#issuecomment-948385227). This provides common
  strategies to work around that problem depending on your use case.

  - `None` : default strategy where nothing in particular happens
  - `"hole"`: Truncates left of input, and leaves a gap wide enough to let generation happen (might
    truncate a lot of the prompt and not suitable when generation exceed the model capacity)
- **tokenizer_encode_kwargs** (`dict`, *optional*) --
  Additional keyword arguments to pass along to the encoding step of the tokenizer. If the text input is
  a chat, it is passed to `apply_chat_template`. Otherwise, it is passed to `__call__`.
- **generate_kwargs** (`dict`, *optional*) --
  Additional keyword arguments to pass along to the generate method of the model (see the generate method
  [here](./text_generation)).0A list or a list of lists of `dict`Returns one of the following dictionaries (cannot return a combination
of both `generated_text` and `generated_token_ids`):

- **generated_text** (`str`, present when `return_text=True`) -- The generated text.
- **generated_token_ids** (`torch.Tensor`, present when `return_tensors=True`) -- The token
  ids of the generated text.

Complete the prompt(s) given as inputs.

**Parameters:**

model ([PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel)) : The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel).

tokenizer ([PreTrainedTokenizer](/docs/transformers/main/en/main_classes/tokenizer#transformers.PythonBackend)) : The tokenizer that will be used by the pipeline to encode data for the model. This object inherits from [PreTrainedTokenizer](/docs/transformers/main/en/main_classes/tokenizer#transformers.PythonBackend).

modelcard (`str` or `ModelCard`, *optional*) : Model card attributed to the model for this pipeline.

task (`str`, defaults to `""`) : A task-identifier for the pipeline.

num_workers (`int`, *optional*, defaults to 8) : When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the number of workers to be used.

batch_size (`int`, *optional*, defaults to 1) : When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the size of the batch to use, for inference this is not always beneficial, please read [Batching with pipelines](https://huggingface.co/transformers/main_classes/pipelines.html#pipeline-batching) .

args_parser ([ArgumentHandler](/docs/transformers/main/en/internal/pipelines_utils#transformers.pipelines.ArgumentHandler), *optional*) : Reference to the object in charge of parsing supplied pipeline parameters.

device (`int`, *optional*, defaults to -1) : Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, a positive will run the model on the associated CUDA device id. You can pass native `torch.device` or a `str` too

dtype (`str` or `torch.dtype`, *optional*) : Sent directly as `model_kwargs` (just a simpler shortcut) to use the available precision for this model (`torch.float16`, `torch.bfloat16`, ... or `"auto"`)

binary_output (`bool`, *optional*, defaults to `False`) : Flag indicating if the output the pipeline should happen in a serialized format (i.e., pickle) or as the raw output data e.g. text.

**Returns:**

`A list or a list of lists of `dict``

Returns one of the following dictionaries (cannot return a combination
of both `generated_text` and `generated_token_ids`):

- **generated_text** (`str`, present when `return_text=True`) -- The generated text.
- **generated_token_ids** (`torch.Tensor`, present when `return_tensors=True`) -- The token
  ids of the generated text.

### Text2TextGenerationPipeline[[transformers.Text2TextGenerationPipeline]]

#### transformers.Text2TextGenerationPipeline[[transformers.Text2TextGenerationPipeline]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/deprecated/text2text_generation.py#L23)

Pipeline for text to text generation using seq2seq models.

Unless the model you're using explicitly sets these generation parameters in its configuration files
(`generation_config.json`), the following default values will be used:
- max_new_tokens: 256
- num_beams: 4

Example:

```python
>>> from transformers import pipeline

>>> generator = pipeline(model="mrm8488/t5-base-finetuned-question-generation-ap")
>>> generator(
...     "answer: Manuel context: Manuel has created RuPERTa-base with the support of HF-Transformers and Google"
... )
[{'generated_text': 'question: Who created the RuPERTa-base?'}]
```

Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial). You can pass text
generation parameters to this pipeline to control stopping criteria, decoding strategy, and more. Learn more about
text generation parameters in [Text generation strategies](../generation_strategies) and [Text
generation](text_generation).

This Text2TextGenerationPipeline pipeline can currently be loaded from [pipeline()](/docs/transformers/main/en/main_classes/pipelines#transformers.pipeline) using the following task
identifier: `"text2text-generation"`.

The models that this pipeline can use are models that have been fine-tuned on a translation task. See the
up-to-date list of available models on
[huggingface.co/models](https://huggingface.co/models?filter=text2text-generation). For a list of available
parameters, see the [following
documentation](https://huggingface.co/docs/transformers/en/main_classes/text_generation#transformers.generation.GenerationMixin.generate)

Usage:

```python
text2text_generator = pipeline("text2text-generation")
text2text_generator("question: What is 42 ? context: 42 is the answer to life, the universe and everything")
```

__call__transformers.Text2TextGenerationPipeline.__call__https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/deprecated/text2text_generation.py#L159[{"name": "*args", "val": ": str | list[str]"}, {"name": "**kwargs", "val": ": typing.Any"}]- **args** (`str` or `list[str]`) --
  Input text for the encoder.
- **return_tensors** (`bool`, *optional*, defaults to `False`) --
  Whether or not to include the tensors of predictions (as token indices) in the outputs.
- **return_text** (`bool`, *optional*, defaults to `True`) --
  Whether or not to include the decoded texts in the outputs.
- **clean_up_tokenization_spaces** (`bool`, *optional*, defaults to `False`) --
  Whether or not to clean up the potential extra spaces in the text output.
- **truncation** (`TruncationStrategy`, *optional*, defaults to `TruncationStrategy.DO_NOT_TRUNCATE`) --
  The truncation strategy for the tokenization within the pipeline. `TruncationStrategy.DO_NOT_TRUNCATE`
  (default) will never truncate, but it is sometimes desirable to truncate the input to fit the model's
  max_length instead of throwing an error down the line.
- **generate_kwargs** --
  Additional keyword arguments to pass along to the generate method of the model (see the generate method
  [here](./text_generation)).0A list or a list of list of `dict`Each result comes as a dictionary with the following keys:

- **generated_text** (`str`, present when `return_text=True`) -- The generated text.
- **generated_token_ids** (`torch.Tensor`, present when `return_tensors=True`) -- The token
  ids of the generated text.

Generate the output text(s) using text(s) given as inputs.

**Parameters:**

model ([PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel)) : The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel).

tokenizer ([PreTrainedTokenizer](/docs/transformers/main/en/main_classes/tokenizer#transformers.PythonBackend)) : The tokenizer that will be used by the pipeline to encode data for the model. This object inherits from [PreTrainedTokenizer](/docs/transformers/main/en/main_classes/tokenizer#transformers.PythonBackend).

modelcard (`str` or `ModelCard`, *optional*) : Model card attributed to the model for this pipeline.

task (`str`, defaults to `""`) : A task-identifier for the pipeline.

num_workers (`int`, *optional*, defaults to 8) : When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the number of workers to be used.

batch_size (`int`, *optional*, defaults to 1) : When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the size of the batch to use, for inference this is not always beneficial, please read [Batching with pipelines](https://huggingface.co/transformers/main_classes/pipelines.html#pipeline-batching) .

args_parser ([ArgumentHandler](/docs/transformers/main/en/internal/pipelines_utils#transformers.pipelines.ArgumentHandler), *optional*) : Reference to the object in charge of parsing supplied pipeline parameters.

device (`int`, *optional*, defaults to -1) : Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, a positive will run the model on the associated CUDA device id. You can pass native `torch.device` or a `str` too

dtype (`str` or `torch.dtype`, *optional*) : Sent directly as `model_kwargs` (just a simpler shortcut) to use the available precision for this model (`torch.float16`, `torch.bfloat16`, ... or `"auto"`)

binary_output (`bool`, *optional*, defaults to `False`) : Flag indicating if the output the pipeline should happen in a serialized format (i.e., pickle) or as the raw output data e.g. text.

**Returns:**

`A list or a list of list of `dict``

Each result comes as a dictionary with the following keys:

- **generated_text** (`str`, present when `return_text=True`) -- The generated text.
- **generated_token_ids** (`torch.Tensor`, present when `return_tensors=True`) -- The token
  ids of the generated text.
#### check_inputs[[transformers.Text2TextGenerationPipeline.check_inputs]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/deprecated/text2text_generation.py#L132)

Checks whether there might be something wrong with given input with regard to the model.

### TokenClassificationPipeline[[transformers.TokenClassificationPipeline]]

#### transformers.TokenClassificationPipeline[[transformers.TokenClassificationPipeline]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/token_classification.py#L95)

Named Entity Recognition pipeline using any `ModelForTokenClassification`. See the [named entity recognition
examples](../task_summary#named-entity-recognition) for more information.

Example:

```python
>>> from transformers import pipeline

>>> token_classifier = pipeline(model="Jean-Baptiste/camembert-ner", aggregation_strategy="simple")
>>> sentence = "Je m'appelle jean-baptiste et je vis à montréal"
>>> tokens = token_classifier(sentence)
>>> tokens
[{'entity_group': 'PER', 'score': 0.9931, 'word': 'jean-baptiste', 'start': 12, 'end': 26}, {'entity_group': 'LOC', 'score': 0.998, 'word': 'montréal', 'start': 38, 'end': 47}]

>>> token = tokens[0]
>>> # Start and end provide an easy way to highlight words in the original text.
>>> sentence[token["start"] : token["end"]]
' jean-baptiste'

>>> # Some models use the same idea to do part of speech.
>>> syntaxer = pipeline(model="vblagoje/bert-english-uncased-finetuned-pos", aggregation_strategy="simple")
>>> syntaxer("My name is Sarah and I live in London")
[{'entity_group': 'PRON', 'score': 0.999, 'word': 'my', 'start': 0, 'end': 2}, {'entity_group': 'NOUN', 'score': 0.997, 'word': 'name', 'start': 3, 'end': 7}, {'entity_group': 'AUX', 'score': 0.994, 'word': 'is', 'start': 8, 'end': 10}, {'entity_group': 'PROPN', 'score': 0.999, 'word': 'sarah', 'start': 11, 'end': 16}, {'entity_group': 'CCONJ', 'score': 0.999, 'word': 'and', 'start': 17, 'end': 20}, {'entity_group': 'PRON', 'score': 0.999, 'word': 'i', 'start': 21, 'end': 22}, {'entity_group': 'VERB', 'score': 0.998, 'word': 'live', 'start': 23, 'end': 27}, {'entity_group': 'ADP', 'score': 0.999, 'word': 'in', 'start': 28, 'end': 30}, {'entity_group': 'PROPN', 'score': 0.999, 'word': 'london', 'start': 31, 'end': 37}]
```

Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

This token recognition pipeline can currently be loaded from [pipeline()](/docs/transformers/main/en/main_classes/pipelines#transformers.pipeline) using the following task identifier:
`"ner"` (for predicting the classes of tokens in a sequence: person, organisation, location or miscellaneous).

The models that this pipeline can use are models that have been fine-tuned on a token classification task. See the
up-to-date list of available models on
[huggingface.co/models](https://huggingface.co/models?filter=token-classification).

__call__transformers.TokenClassificationPipeline.__call__https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/token_classification.py#L233[{"name": "inputs", "val": ": str | list[str]"}, {"name": "**kwargs", "val": ": typing.Any"}]- **inputs** (`str` or `List[str]`) --
  One or several texts (or one list of texts) for token classification. Can be pre-tokenized when
  `is_split_into_words=True`.0A list or a list of list of `dict`Each result comes as a list of dictionaries (one for each token in the
corresponding input, or each entity if this pipeline was instantiated with an aggregation_strategy) with
the following keys:

- **word** (`str`) -- The token/word classified. This is obtained by decoding the selected tokens. If you
  want to have the exact string in the original sentence, use `start` and `end`.
- **score** (`float`) -- The corresponding probability for `entity`.
- **entity** (`str`) -- The entity predicted for that token/word (it is named *entity_group* when
  *aggregation_strategy* is not `"none"`.
- **index** (`int`, only present when `aggregation_strategy="none"`) -- The index of the corresponding
  token in the sentence.
- **start** (`int`, *optional*) -- The index of the start of the corresponding entity in the sentence. Only
  exists if the offsets are available within the tokenizer
- **end** (`int`, *optional*) -- The index of the end of the corresponding entity in the sentence. Only
  exists if the offsets are available within the tokenizer

Classify each token of the text(s) given as inputs.

**Parameters:**

model ([PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel)) : The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel).

tokenizer ([PreTrainedTokenizer](/docs/transformers/main/en/main_classes/tokenizer#transformers.PythonBackend)) : The tokenizer that will be used by the pipeline to encode data for the model. This object inherits from [PreTrainedTokenizer](/docs/transformers/main/en/main_classes/tokenizer#transformers.PythonBackend).

modelcard (`str` or `ModelCard`, *optional*) : Model card attributed to the model for this pipeline.

task (`str`, defaults to `""`) : A task-identifier for the pipeline.

num_workers (`int`, *optional*, defaults to 8) : When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the number of workers to be used.

batch_size (`int`, *optional*, defaults to 1) : When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the size of the batch to use, for inference this is not always beneficial, please read [Batching with pipelines](https://huggingface.co/transformers/main_classes/pipelines.html#pipeline-batching) .

args_parser ([ArgumentHandler](/docs/transformers/main/en/internal/pipelines_utils#transformers.pipelines.ArgumentHandler), *optional*) : Reference to the object in charge of parsing supplied pipeline parameters.

device (`int`, *optional*, defaults to -1) : Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, a positive will run the model on the associated CUDA device id. You can pass native `torch.device` or a `str` too

dtype (`str` or `torch.dtype`, *optional*) : Sent directly as `model_kwargs` (just a simpler shortcut) to use the available precision for this model (`torch.float16`, `torch.bfloat16`, ... or `"auto"`)

binary_output (`bool`, *optional*, defaults to `False`) : Flag indicating if the output the pipeline should happen in a serialized format (i.e., pickle) or as the raw output data e.g. text.

ignore_labels (`list[str]`, defaults to `["O"]`) : A list of labels to ignore.

grouped_entities (`bool`, *optional*, defaults to `False`) : DEPRECATED, use `aggregation_strategy` instead. Whether or not to group the tokens corresponding to the same entity together in the predictions or not.

stride (`int`, *optional*) : If stride is provided, the pipeline is applied on all the text. The text is split into chunks of size model_max_length. Works only with fast tokenizers and `aggregation_strategy` different from `NONE`. The value of this argument defines the number of overlapping tokens between chunks. In other words, the model will shift forward by `tokenizer.model_max_length - stride` tokens each step.

aggregation_strategy (`str`, *optional*, defaults to `"none"`) : The strategy to fuse (or not) tokens based on the model prediction.  - "none" : Will simply not do any aggregation and simply return raw results from the model - "simple" : Will attempt to group entities following the default schema. (A, B-TAG), (B, I-TAG), (C, I-TAG), (D, B-TAG2) (E, B-TAG2) will end up being [{"word": ABC, "entity": "TAG"}, {"word": "D", "entity": "TAG2"}, {"word": "E", "entity": "TAG2"}] Notice that two consecutive B tags will end up as different entities. On word based languages, we might end up splitting words undesirably : Imagine Microsoft being tagged as [{"word": "Micro", "entity": "ENTERPRISE"}, {"word": "soft", "entity": "NAME"}]. Look for FIRST, MAX, AVERAGE for ways to mitigate that and disambiguate words (on languages that support that meaning, which is basically tokens separated by a space). These mitigations will only work on real words, "New york" might still be tagged with two different entities. - "first" : (works only on word based models) Will use the `SIMPLE` strategy except that words, cannot end up with different tags. Words will simply use the tag of the first token of the word when there is ambiguity. - "average" : (works only on word based models) Will use the `SIMPLE` strategy except that words, cannot end up with different tags. scores will be averaged first across tokens, and then the maximum label is applied. - "max" : (works only on word based models) Will use the `SIMPLE` strategy except that words, cannot end up with different tags. Word entity will simply be the token with the maximum score.

**Returns:**

`A list or a list of list of `dict``

Each result comes as a list of dictionaries (one for each token in the
corresponding input, or each entity if this pipeline was instantiated with an aggregation_strategy) with
the following keys:

- **word** (`str`) -- The token/word classified. This is obtained by decoding the selected tokens. If you
  want to have the exact string in the original sentence, use `start` and `end`.
- **score** (`float`) -- The corresponding probability for `entity`.
- **entity** (`str`) -- The entity predicted for that token/word (it is named *entity_group* when
  *aggregation_strategy* is not `"none"`.
- **index** (`int`, only present when `aggregation_strategy="none"`) -- The index of the corresponding
  token in the sentence.
- **start** (`int`, *optional*) -- The index of the start of the corresponding entity in the sentence. Only
  exists if the offsets are available within the tokenizer
- **end** (`int`, *optional*) -- The index of the end of the corresponding entity in the sentence. Only
  exists if the offsets are available within the tokenizer
#### aggregate_words[[transformers.TokenClassificationPipeline.aggregate_words]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/token_classification.py#L545)

Override tokens from a given word that disagree to force agreement on word boundaries.

Example: micro|soft| com|pany| B-ENT I-NAME I-ENT I-ENT will be rewritten with first strategy as microsoft|
company| B-ENT I-ENT
#### gather_pre_entities[[transformers.TokenClassificationPipeline.gather_pre_entities]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/token_classification.py#L421)

Fuse various numpy arrays into dicts with all the information needed for aggregation
#### group_entities[[transformers.TokenClassificationPipeline.group_entities]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/token_classification.py#L608)

Find and group together the adjacent tokens with the same entity predicted.

**Parameters:**

entities (`dict`) : The entities predicted by the pipeline.
#### group_sub_entities[[transformers.TokenClassificationPipeline.group_sub_entities]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/token_classification.py#L573)

Group together the adjacent tokens with the same entity predicted.

**Parameters:**

entities (`dict`) : The entities predicted by the pipeline.

### TranslationPipeline[[transformers.TranslationPipeline]]

#### transformers.TranslationPipeline[[transformers.TranslationPipeline]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/deprecated/text2text_generation.py#L313)

Translates from one language to another.

This translation pipeline can currently be loaded from [pipeline()](/docs/transformers/main/en/main_classes/pipelines#transformers.pipeline) using the following task identifier:
`"translation_xx_to_yy"`.

The models that this pipeline can use are models that have been fine-tuned on a translation task. See the
up-to-date list of available models on [huggingface.co/models](https://huggingface.co/models?filter=translation).
For a list of available parameters, see the [following
documentation](https://huggingface.co/docs/transformers/en/main_classes/text_generation#transformers.generation.GenerationMixin.generate)

Unless the model you're using explicitly sets these generation parameters in its configuration files
(`generation_config.json`), the following default values will be used:
- max_new_tokens: 256
- num_beams: 4

Usage:

```python
en_fr_translator = pipeline("translation_en_to_fr")
en_fr_translator("How old are you?")
```

__call__transformers.TranslationPipeline.__call__https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/deprecated/text2text_generation.py#L378[{"name": "*args", "val": ""}, {"name": "**kwargs", "val": ""}]- **args** (`str` or `list[str]`) --
  Texts to be translated.
- **return_tensors** (`bool`, *optional*, defaults to `False`) --
  Whether or not to include the tensors of predictions (as token indices) in the outputs.
- **return_text** (`bool`, *optional*, defaults to `True`) --
  Whether or not to include the decoded texts in the outputs.
- **clean_up_tokenization_spaces** (`bool`, *optional*, defaults to `False`) --
  Whether or not to clean up the potential extra spaces in the text output.
- **src_lang** (`str`, *optional*) --
  The language of the input. Might be required for multilingual models. Will not have any effect for
  single pair translation models
- **tgt_lang** (`str`, *optional*) --
  The language of the desired output. Might be required for multilingual models. Will not have any effect
  for single pair translation models
- **generate_kwargs** --
  Additional keyword arguments to pass along to the generate method of the model (see the generate method
  [here](./text_generation)).0A list or a list of list of `dict`Each result comes as a dictionary with the following keys:

- **translation_text** (`str`, present when `return_text=True`) -- The translation.
- **translation_token_ids** (`torch.Tensor`, present when `return_tensors=True`) -- The
  token ids of the translation.

Translate the text(s) given as inputs.

**Parameters:**

model ([PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel)) : The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel).

tokenizer ([PreTrainedTokenizer](/docs/transformers/main/en/main_classes/tokenizer#transformers.PythonBackend)) : The tokenizer that will be used by the pipeline to encode data for the model. This object inherits from [PreTrainedTokenizer](/docs/transformers/main/en/main_classes/tokenizer#transformers.PythonBackend).

modelcard (`str` or `ModelCard`, *optional*) : Model card attributed to the model for this pipeline.

task (`str`, defaults to `""`) : A task-identifier for the pipeline.

num_workers (`int`, *optional*, defaults to 8) : When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the number of workers to be used.

batch_size (`int`, *optional*, defaults to 1) : When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the size of the batch to use, for inference this is not always beneficial, please read [Batching with pipelines](https://huggingface.co/transformers/main_classes/pipelines.html#pipeline-batching) .

args_parser ([ArgumentHandler](/docs/transformers/main/en/internal/pipelines_utils#transformers.pipelines.ArgumentHandler), *optional*) : Reference to the object in charge of parsing supplied pipeline parameters.

device (`int`, *optional*, defaults to -1) : Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, a positive will run the model on the associated CUDA device id. You can pass native `torch.device` or a `str` too

dtype (`str` or `torch.dtype`, *optional*) : Sent directly as `model_kwargs` (just a simpler shortcut) to use the available precision for this model (`torch.float16`, `torch.bfloat16`, ... or `"auto"`)

binary_output (`bool`, *optional*, defaults to `False`) : Flag indicating if the output the pipeline should happen in a serialized format (i.e., pickle) or as the raw output data e.g. text.

**Returns:**

`A list or a list of list of `dict``

Each result comes as a dictionary with the following keys:

- **translation_text** (`str`, present when `return_text=True`) -- The translation.
- **translation_token_ids** (`torch.Tensor`, present when `return_tensors=True`) -- The
  token ids of the translation.
#### check_inputs[[transformers.TranslationPipeline.check_inputs]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/deprecated/text2text_generation.py#L348)

Removed input length check - unnecessary with max_new_tokens (previously relevant for max_length)

### ZeroShotClassificationPipeline[[transformers.ZeroShotClassificationPipeline]]

#### transformers.ZeroShotClassificationPipeline[[transformers.ZeroShotClassificationPipeline]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/zero_shot_classification.py#L44)

NLI-based zero-shot classification pipeline using a `ModelForSequenceClassification` trained on NLI (natural
language inference) tasks. Equivalent of `text-classification` pipelines, but these models don't require a
hardcoded number of potential classes, they can be chosen at runtime. It usually means it's slower but it is
**much** more flexible.

Any combination of sequences and labels can be passed and each combination will be posed as a premise/hypothesis
pair and passed to the pretrained model. Then, the logit for *entailment* is taken as the logit for the candidate
label being valid. Any NLI model can be used, but the id of the *entailment* label must be included in the model
config's :attr:*~transformers.PreTrainedConfig.label2id*.

Example:

```python
>>> from transformers import pipeline

>>> oracle = pipeline(model="facebook/bart-large-mnli")
>>> oracle(
...     "I have a problem with my iphone that needs to be resolved asap!!",
...     candidate_labels=["urgent", "not urgent", "phone", "tablet", "computer"],
... )
{'sequence': 'I have a problem with my iphone that needs to be resolved asap!!', 'labels': ['urgent', 'phone', 'computer', 'not urgent', 'tablet'], 'scores': [0.504, 0.479, 0.013, 0.003, 0.002]}

>>> oracle(
...     "I have a problem with my iphone that needs to be resolved asap!!",
...     candidate_labels=["english", "german"],
... )
{'sequence': 'I have a problem with my iphone that needs to be resolved asap!!', 'labels': ['english', 'german'], 'scores': [0.814, 0.186]}
```

Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

This NLI pipeline can currently be loaded from [pipeline()](/docs/transformers/main/en/main_classes/pipelines#transformers.pipeline) using the following task identifier:
`"zero-shot-classification"`.

The models that this pipeline can use are models that have been fine-tuned on an NLI task. See the up-to-date list
of available models on [huggingface.co/models](https://huggingface.co/models?search=nli).

__call__transformers.ZeroShotClassificationPipeline.__call__https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/zero_shot_classification.py#L165[{"name": "sequences", "val": ": str | list[str]"}, {"name": "*args", "val": ""}, {"name": "**kwargs", "val": ""}]- **sequences** (`str` or `list[str]`) --
  The sequence(s) to classify, will be truncated if the model input is too large.
- **candidate_labels** (`str` or `list[str]`) --
  The set of possible class labels to classify each sequence into. Can be a single label, a string of
  comma-separated labels, or a list of labels.
- **hypothesis_template** (`str`, *optional*, defaults to `"This example is {}."`) --
  The template used to turn each label into an NLI-style hypothesis. This template must include a {} or
  similar syntax for the candidate label to be inserted into the template. For example, the default
  template is `"This example is {}."` With the candidate label `"sports"`, this would be fed into the
  model like `" sequence to classify  This example is sports . "`. The default template
  works well in many cases, but it may be worthwhile to experiment with different templates depending on
  the task setting.
- **multi_label** (`bool`, *optional*, defaults to `False`) --
  Whether or not multiple candidate labels can be true. If `False`, the scores are normalized such that
  the sum of the label likelihoods for each sequence is 1. If `True`, the labels are considered
  independent and probabilities are normalized for each candidate by doing a softmax of the entailment
  score vs. the contradiction score.0A `dict` or a list of `dict`Each result comes as a dictionary with the following keys:

- **sequence** (`str`) -- The sequence for which this is the output.
- **labels** (`list[str]`) -- The labels sorted by order of likelihood.
- **scores** (`list[float]`) -- The probabilities for each of the labels.

Classify the sequence(s) given as inputs. See the [ZeroShotClassificationPipeline](/docs/transformers/main/en/main_classes/pipelines#transformers.ZeroShotClassificationPipeline) documentation for more
information.

**Parameters:**

model ([PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel)) : The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel).

tokenizer ([PreTrainedTokenizer](/docs/transformers/main/en/main_classes/tokenizer#transformers.PythonBackend)) : The tokenizer that will be used by the pipeline to encode data for the model. This object inherits from [PreTrainedTokenizer](/docs/transformers/main/en/main_classes/tokenizer#transformers.PythonBackend).

modelcard (`str` or `ModelCard`, *optional*) : Model card attributed to the model for this pipeline.

task (`str`, defaults to `""`) : A task-identifier for the pipeline.

num_workers (`int`, *optional*, defaults to 8) : When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the number of workers to be used.

batch_size (`int`, *optional*, defaults to 1) : When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the size of the batch to use, for inference this is not always beneficial, please read [Batching with pipelines](https://huggingface.co/transformers/main_classes/pipelines.html#pipeline-batching) .

args_parser ([ArgumentHandler](/docs/transformers/main/en/internal/pipelines_utils#transformers.pipelines.ArgumentHandler), *optional*) : Reference to the object in charge of parsing supplied pipeline parameters.

device (`int`, *optional*, defaults to -1) : Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, a positive will run the model on the associated CUDA device id. You can pass native `torch.device` or a `str` too

dtype (`str` or `torch.dtype`, *optional*) : Sent directly as `model_kwargs` (just a simpler shortcut) to use the available precision for this model (`torch.float16`, `torch.bfloat16`, ... or `"auto"`)

binary_output (`bool`, *optional*, defaults to `False`) : Flag indicating if the output the pipeline should happen in a serialized format (i.e., pickle) or as the raw output data e.g. text.

**Returns:**

`A `dict` or a list of `dict``

Each result comes as a dictionary with the following keys:

- **sequence** (`str`) -- The sequence for which this is the output.
- **labels** (`list[str]`) -- The labels sorted by order of likelihood.
- **scores** (`list[float]`) -- The probabilities for each of the labels.

## Multimodal

Pipelines available for multimodal tasks include the following.

### DocumentQuestionAnsweringPipeline[[transformers.DocumentQuestionAnsweringPipeline]]

#### transformers.DocumentQuestionAnsweringPipeline[[transformers.DocumentQuestionAnsweringPipeline]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/document_question_answering.py#L103)

Document Question Answering pipeline using any `AutoModelForDocumentQuestionAnswering`. The inputs/outputs are
similar to the (extractive) question answering pipeline; however, the pipeline takes an image (and optional OCR'd
words/boxes) as input instead of text context.

Unless the model you're using explicitly sets these generation parameters in its configuration files
(`generation_config.json`), the following default values will be used:
- max_new_tokens: 256

Example:

```python
>>> from transformers import pipeline

>>> document_qa = pipeline(model="impira/layoutlm-document-qa")
>>> document_qa(
...     image="https://huggingface.co/spaces/impira/docquery/resolve/2359223c1837a7587402bda0f2643382a6eefeab/invoice.png",
...     question="What is the invoice number?",
... )
[{'score': 0.425, 'answer': 'us-001', 'start': 16, 'end': 16}]
```

Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

This document question answering pipeline can currently be loaded from [pipeline()](/docs/transformers/main/en/main_classes/pipelines#transformers.pipeline) using the following task
identifier: `"document-question-answering"`.

The models that this pipeline can use are models that have been fine-tuned on a document question answering task.
See the up-to-date list of available models on
[huggingface.co/models](https://huggingface.co/models?filter=document-question-answering).

__call__transformers.DocumentQuestionAnsweringPipeline.__call__https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/document_question_answering.py#L233[{"name": "image", "val": ": typing.Union[ForwardRef('Image.Image'), str, list[dict[str, typing.Any]]]"}, {"name": "question", "val": ": str | None = None"}, {"name": "word_boxes", "val": ": tuple[str, list[float]] | None = None"}, {"name": "**kwargs", "val": ": typing.Any"}]- **image** (`str` or `PIL.Image`) --
  The pipeline handles three types of images:

  - A string containing a http link pointing to an image
  - A string containing a local path to an image
  - An image loaded in PIL directly

  The pipeline accepts either a single image or a batch of images. If given a single image, it can be
  broadcasted to multiple questions.
- **question** (`str`) --
  A question to ask of the document.
- **word_boxes** (`list[str, tuple[float, float, float, float]]`, *optional*) --
  A list of words and bounding boxes (normalized 0->1000). If you provide this optional input, then the
  pipeline will use these words and boxes instead of running OCR on the image to derive them for models
  that need them (e.g. LayoutLM). This allows you to reuse OCR'd results across many invocations of the
  pipeline without having to re-run it each time.
- **top_k** (`int`, *optional*, defaults to 1) --
  The number of answers to return (will be chosen by order of likelihood). Note that we return less than
  top_k answers if there are not enough options available within the context.
- **doc_stride** (`int`, *optional*, defaults to 128) --
  If the words in the document are too long to fit with the question for the model, it will be split in
  several chunks with some overlap. This argument controls the size of that overlap.
- **max_answer_len** (`int`, *optional*, defaults to 15) --
  The maximum length of predicted answers (e.g., only answers with a shorter length are considered).
- **max_seq_len** (`int`, *optional*, defaults to 384) --
  The maximum length of the total sentence (context + question) in tokens of each chunk passed to the
  model. The context will be split in several chunks (using `doc_stride` as overlap) if needed.
- **max_question_len** (`int`, *optional*, defaults to 64) --
  The maximum length of the question after tokenization. It will be truncated if needed.
- **handle_impossible_answer** (`bool`, *optional*, defaults to `False`) --
  Whether or not we accept impossible as an answer.
- **lang** (`str`, *optional*) --
  Language to use while running OCR. Defaults to english.
- **tesseract_config** (`str`, *optional*) --
  Additional flags to pass to tesseract while running OCR.
- **timeout** (`float`, *optional*, defaults to None) --
  The maximum time in seconds to wait for fetching images from the web. If None, no timeout is set and
  the call may block forever.0A `dict` or a list of `dict`Each result comes as a dictionary with the following keys:

- **score** (`float`) -- The probability associated to the answer.
- **start** (`int`) -- The start word index of the answer (in the OCR'd version of the input or provided
  `word_boxes`).
- **end** (`int`) -- The end word index of the answer (in the OCR'd version of the input or provided
  `word_boxes`).
- **answer** (`str`) -- The answer to the question.
- **words** (`list[int]`) -- The index of each word/box pair that is in the answer

Answer the question(s) given as inputs by using the document(s). A document is defined as an image and an
optional list of (word, box) tuples which represent the text in the document. If the `word_boxes` are not
provided, it will use the Tesseract OCR engine (if available) to extract the words and boxes automatically for
LayoutLM-like models which require them as input. For Donut, no OCR is run.

You can invoke the pipeline several ways:

- `pipeline(image=image, question=question)`
- `pipeline(image=image, question=question, word_boxes=word_boxes)`
- `pipeline([{"image": image, "question": question}])`
- `pipeline([{"image": image, "question": question, "word_boxes": word_boxes}])`

**Parameters:**

model ([PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel)) : The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel).

tokenizer ([PreTrainedTokenizer](/docs/transformers/main/en/main_classes/tokenizer#transformers.PythonBackend)) : The tokenizer that will be used by the pipeline to encode data for the model. This object inherits from [PreTrainedTokenizer](/docs/transformers/main/en/main_classes/tokenizer#transformers.PythonBackend).

image_processor ([BaseImageProcessor](/docs/transformers/main/en/main_classes/image_processor#transformers.BaseImageProcessor)) : The image processor that will be used by the pipeline to encode data for the model. This object inherits from [BaseImageProcessor](/docs/transformers/main/en/main_classes/image_processor#transformers.BaseImageProcessor).

modelcard (`str` or `ModelCard`, *optional*) : Model card attributed to the model for this pipeline.

task (`str`, defaults to `""`) : A task-identifier for the pipeline.

num_workers (`int`, *optional*, defaults to 8) : When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the number of workers to be used.

batch_size (`int`, *optional*, defaults to 1) : When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the size of the batch to use, for inference this is not always beneficial, please read [Batching with pipelines](https://huggingface.co/transformers/main_classes/pipelines.html#pipeline-batching) .

args_parser ([ArgumentHandler](/docs/transformers/main/en/internal/pipelines_utils#transformers.pipelines.ArgumentHandler), *optional*) : Reference to the object in charge of parsing supplied pipeline parameters.

device (`int`, *optional*, defaults to -1) : Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, a positive will run the model on the associated CUDA device id. You can pass native `torch.device` or a `str` too

dtype (`str` or `torch.dtype`, *optional*) : Sent directly as `model_kwargs` (just a simpler shortcut) to use the available precision for this model (`torch.float16`, `torch.bfloat16`, ... or `"auto"`)

binary_output (`bool`, *optional*, defaults to `False`) : Flag indicating if the output the pipeline should happen in a serialized format (i.e., pickle) or as the raw output data e.g. text.

**Returns:**

`A `dict` or a list of `dict``

Each result comes as a dictionary with the following keys:

- **score** (`float`) -- The probability associated to the answer.
- **start** (`int`) -- The start word index of the answer (in the OCR'd version of the input or provided
  `word_boxes`).
- **end** (`int`) -- The end word index of the answer (in the OCR'd version of the input or provided
  `word_boxes`).
- **answer** (`str`) -- The answer to the question.
- **words** (`list[int]`) -- The index of each word/box pair that is in the answer

### FeatureExtractionPipeline[[transformers.FeatureExtractionPipeline]]

#### transformers.FeatureExtractionPipeline[[transformers.FeatureExtractionPipeline]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/feature_extraction.py#L15)

Feature extraction pipeline uses no model head. This pipeline extracts the hidden states from the base
transformer, which can be used as features in downstream tasks.

Example:

```python
>>> from transformers import pipeline

>>> extractor = pipeline(model="google-bert/bert-base-uncased", task="feature-extraction")
>>> result = extractor("This is a simple test.", return_tensors=True)
>>> result.shape  # This is a tensor of shape [1, sequence_length, hidden_dimension] representing the input string.
torch.Size([1, 8, 768])
```

Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

This feature extraction pipeline can currently be loaded from [pipeline()](/docs/transformers/main/en/main_classes/pipelines#transformers.pipeline) using the task identifier:
`"feature-extraction"`.

All models may be used for this pipeline. See a list of all models, including community-contributed models on
[huggingface.co/models](https://huggingface.co/models).

__call__transformers.FeatureExtractionPipeline.__call__https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/feature_extraction.py#L78[{"name": "*args", "val": ": str | list[str]"}, {"name": "**kwargs", "val": ": typing.Any"}]- **args** (`str` or `list[str]`) -- One or several texts (or one list of texts) to get the features of.0A nested list of `float`The features computed by the model.

Extract the features of the input(s) text.

**Parameters:**

model ([PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel)) : The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel).

tokenizer ([PreTrainedTokenizer](/docs/transformers/main/en/main_classes/tokenizer#transformers.PythonBackend)) : The tokenizer that will be used by the pipeline to encode data for the model. This object inherits from [PreTrainedTokenizer](/docs/transformers/main/en/main_classes/tokenizer#transformers.PythonBackend).

modelcard (`str` or `ModelCard`, *optional*) : Model card attributed to the model for this pipeline.

task (`str`, defaults to `""`) : A task-identifier for the pipeline.

num_workers (`int`, *optional*, defaults to 8) : When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the number of workers to be used.

batch_size (`int`, *optional*, defaults to 1) : When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the size of the batch to use, for inference this is not always beneficial, please read [Batching with pipelines](https://huggingface.co/transformers/main_classes/pipelines.html#pipeline-batching) .

args_parser ([ArgumentHandler](/docs/transformers/main/en/internal/pipelines_utils#transformers.pipelines.ArgumentHandler), *optional*) : Reference to the object in charge of parsing supplied pipeline parameters.

device (`int`, *optional*, defaults to -1) : Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, a positive will run the model on the associated CUDA device id. You can pass native `torch.device` or a `str` too

dtype (`str` or `torch.dtype`, *optional*) : Sent directly as `model_kwargs` (just a simpler shortcut) to use the available precision for this model (`torch.float16`, `torch.bfloat16`, ... or `"auto"`)

tokenize_kwargs (`dict`, *optional*) : Additional dictionary of keyword arguments passed along to the tokenizer.

return_tensors (`bool`, *optional*) : If `True`, returns a tensor according to the specified framework, otherwise returns a list.

**Returns:**

`A nested list of `float``

The features computed by the model.

### ImageFeatureExtractionPipeline[[transformers.ImageFeatureExtractionPipeline]]

#### transformers.ImageFeatureExtractionPipeline[[transformers.ImageFeatureExtractionPipeline]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/image_feature_extraction.py#L23)

Image feature extraction pipeline uses no model head. This pipeline extracts the hidden states from the base
transformer, which can be used as features in downstream tasks.

Example:

```python
>>> from transformers import pipeline

>>> extractor = pipeline(model="google/vit-base-patch16-224", task="image-feature-extraction")
>>> result = extractor("https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png", return_tensors=True)
>>> result.shape  # This is a tensor of shape [1, sequence_length, hidden_dimension] representing the input image.
torch.Size([1, 197, 768])
```

Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

This image feature extraction pipeline can currently be loaded from [pipeline()](/docs/transformers/main/en/main_classes/pipelines#transformers.pipeline) using the task identifier:
`"image-feature-extraction"`.

All vision models may be used for this pipeline. See a list of all models, including community-contributed models on
[huggingface.co/models](https://huggingface.co/models).

__call__transformers.ImageFeatureExtractionPipeline.__call__https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/image_feature_extraction.py#L94[{"name": "*args", "val": ": typing.Union[str, ForwardRef('Image.Image'), list['Image.Image'], list[str]]"}, {"name": "**kwargs", "val": ": typing.Any"}]- **images** (`str`, `list[str]`, `PIL.Image` or `list[PIL.Image]`) --
  The pipeline handles three types of images:

  - A string containing a http link pointing to an image
  - A string containing a local path to an image
  - An image loaded in PIL directly

  The pipeline accepts either a single image or a batch of images, which must then be passed as a string.
  Images in a batch must all be in the same format: all as http links, all as local paths, or all as PIL
  images.
- **timeout** (`float`, *optional*, defaults to None) --
  The maximum time in seconds to wait for fetching images from the web. If None, no timeout is used and
  the call may block forever.0A nested list of `float`The features computed by the model.

Extract the features of the input(s).

**Parameters:**

model ([PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel)) : The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel).

image_processor ([BaseImageProcessor](/docs/transformers/main/en/main_classes/image_processor#transformers.BaseImageProcessor)) : The image processor that will be used by the pipeline to encode data for the model. This object inherits from [BaseImageProcessor](/docs/transformers/main/en/main_classes/image_processor#transformers.BaseImageProcessor).

modelcard (`str` or `ModelCard`, *optional*) : Model card attributed to the model for this pipeline.

task (`str`, defaults to `""`) : A task-identifier for the pipeline.

num_workers (`int`, *optional*, defaults to 8) : When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the number of workers to be used.

batch_size (`int`, *optional*, defaults to 1) : When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the size of the batch to use, for inference this is not always beneficial, please read [Batching with pipelines](https://huggingface.co/transformers/main_classes/pipelines.html#pipeline-batching) .

args_parser ([ArgumentHandler](/docs/transformers/main/en/internal/pipelines_utils#transformers.pipelines.ArgumentHandler), *optional*) : Reference to the object in charge of parsing supplied pipeline parameters.

device (`int`, *optional*, defaults to -1) : Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, a positive will run the model on the associated CUDA device id. You can pass native `torch.device` or a `str` too

dtype (`str` or `torch.dtype`, *optional*) : Sent directly as `model_kwargs` (just a simpler shortcut) to use the available precision for this model (`torch.float16`, `torch.bfloat16`, ... or `"auto"`)

binary_output (`bool`, *optional*, defaults to `False`) : Flag indicating if the output the pipeline should happen in a serialized format (i.e., pickle) or as the raw output data e.g. text.

image_processor_kwargs (`dict`, *optional*) : Additional dictionary of keyword arguments passed along to the image processor e.g. {"size": {"height": 100, "width": 100}}

pool (`bool`, *optional*, defaults to `False`) : Whether or not to return the pooled output. If `False`, the model will return the raw hidden states.

**Returns:**

`A nested list of `float``

The features computed by the model.

### ImageToTextPipeline[[transformers.ImageToTextPipeline]]

#### transformers.ImageToTextPipeline[[transformers.ImageToTextPipeline]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/image_to_text.py#L43)

Image To Text pipeline using a `AutoModelForImageTextToText`. This pipeline predicts a caption for a given image.

Unless the model you're using explicitly sets these generation parameters in its configuration files
(`generation_config.json`), the following default values will be used:
- max_new_tokens: 256

Example:

```python
>>> from transformers import pipeline

>>> captioner = pipeline(model="ydshieh/vit-gpt2-coco-en")
>>> captioner("https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png")
[{'generated_text': 'two birds are standing next to each other '}]
```

Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

This image to text pipeline can currently be loaded from pipeline() using the following task identifier:
"image-to-text".

See the list of available models on
[huggingface.co/models](https://huggingface.co/models?pipeline_tag=image-to-text).

__call__transformers.ImageToTextPipeline.__call__https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/image_to_text.py#L118[{"name": "inputs", "val": ": typing.Union[str, list[str], ForwardRef('Image.Image'), list['Image.Image']]"}, {"name": "**kwargs", "val": ""}]- **inputs** (`str`, `list[str]`, `PIL.Image` or `list[PIL.Image]`) --
  The pipeline handles three types of images:

  - A string containing a HTTP(s) link pointing to an image
  - A string containing a local path to an image
  - An image loaded in PIL directly

  The pipeline accepts either a single image or a batch of images.

- **max_new_tokens** (`int`, *optional*) --
  The amount of maximum tokens to generate. By default it will use `generate` default.

- **generate_kwargs** (`Dict`, *optional*) --
  Pass it to send all of these arguments directly to `generate` allowing full control of this function.

- **timeout** (`float`, *optional*, defaults to None) --
  The maximum time in seconds to wait for fetching images from the web. If None, no timeout is set and
  the call may block forever.0A list or a list of list of `dict`Each result comes as a dictionary with the following key:

- **generated_text** (`str`) -- The generated text.

Assign labels to the image(s) passed as inputs.

**Parameters:**

model ([PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel)) : The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel).

tokenizer ([PreTrainedTokenizer](/docs/transformers/main/en/main_classes/tokenizer#transformers.PythonBackend)) : The tokenizer that will be used by the pipeline to encode data for the model. This object inherits from [PreTrainedTokenizer](/docs/transformers/main/en/main_classes/tokenizer#transformers.PythonBackend).

image_processor ([BaseImageProcessor](/docs/transformers/main/en/main_classes/image_processor#transformers.BaseImageProcessor)) : The image processor that will be used by the pipeline to encode data for the model. This object inherits from [BaseImageProcessor](/docs/transformers/main/en/main_classes/image_processor#transformers.BaseImageProcessor).

modelcard (`str` or `ModelCard`, *optional*) : Model card attributed to the model for this pipeline.

task (`str`, defaults to `""`) : A task-identifier for the pipeline.

num_workers (`int`, *optional*, defaults to 8) : When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the number of workers to be used.

batch_size (`int`, *optional*, defaults to 1) : When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the size of the batch to use, for inference this is not always beneficial, please read [Batching with pipelines](https://huggingface.co/transformers/main_classes/pipelines.html#pipeline-batching) .

args_parser ([ArgumentHandler](/docs/transformers/main/en/internal/pipelines_utils#transformers.pipelines.ArgumentHandler), *optional*) : Reference to the object in charge of parsing supplied pipeline parameters.

device (`int`, *optional*, defaults to -1) : Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, a positive will run the model on the associated CUDA device id. You can pass native `torch.device` or a `str` too

dtype (`str` or `torch.dtype`, *optional*) : Sent directly as `model_kwargs` (just a simpler shortcut) to use the available precision for this model (`torch.float16`, `torch.bfloat16`, ... or `"auto"`)

binary_output (`bool`, *optional*, defaults to `False`) : Flag indicating if the output the pipeline should happen in a serialized format (i.e., pickle) or as the raw output data e.g. text.

**Returns:**

`A list or a list of list of `dict``

Each result comes as a dictionary with the following key:

- **generated_text** (`str`) -- The generated text.

### ImageTextToTextPipeline[[transformers.ImageTextToTextPipeline]]

#### transformers.ImageTextToTextPipeline[[transformers.ImageTextToTextPipeline]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/image_text_to_text.py#L54)

Image-text-to-text pipeline using an `AutoModelForImageTextToText`. This pipeline generates text given an image and text.
When the underlying model is a conversational model, it can also accept one or more chats,
in which case the pipeline will operate in chat mode and will continue the chat(s) by adding its response(s).
Each chat takes the form of a list of dicts, where each dict contains "role" and "content" keys.

Unless the model you're using explicitly sets these generation parameters in its configuration files
(`generation_config.json`), the following default values will be used:
- max_new_tokens: 256

Example:

```python
>>> from transformers import pipeline

>>> pipe = pipeline(task="image-text-to-text", model="Salesforce/blip-image-captioning-base")
>>> pipe("https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png", text="A photo of")
[{'generated_text': 'a photo of two birds'}]
```

```python
>>> from transformers import pipeline

>>> pipe = pipeline("image-text-to-text", model="llava-hf/llava-interleave-qwen-0.5b-hf")
>>> messages = [
>>>     {
>>>         "role": "user",
>>>         "content": [
>>>             {
>>>                 "type": "image",
>>>                 "url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
>>>             },
>>>             {"type": "text", "text": "Describe this image."},
>>>         ],
>>>     },
>>>     {
>>>         "role": "assistant",
>>>         "content": [
>>>             {"type": "text", "text": "There is a dog and"},
>>>         ],
>>>     },
>>> ]
>>> pipe(text=messages, max_new_tokens=20, return_full_text=False)
[{'input_text': [{'role': 'user',
    'content': [{'type': 'image',
    'url': 'https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg'},
    {'type': 'text', 'text': 'Describe this image.'}]},
{'role': 'assistant',
    'content': [{'type': 'text', 'text': 'There is a dog and'}]}],
'generated_text': ' a person in the image. The dog is sitting on the sand, and the person is sitting on'}]
```

Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

This image-text to text pipeline can currently be loaded from pipeline() using the following task identifier:
"image-text-to-text".

See the list of available models on
[huggingface.co/models](https://huggingface.co/models?pipeline_tag=image-text-to-text).

__call__transformers.ImageTextToTextPipeline.__call__https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/image_text_to_text.py#L214[{"name": "images", "val": ": typing.Union[str, list[str], list[list[str]], ForwardRef('Image.Image'), list['Image.Image'], list[list['Image.Image']], list[dict], NoneType] = None"}, {"name": "text", "val": ": str | list[str] | list[dict] | None = None"}, {"name": "**kwargs", "val": ""}]- **images** (`str`, `list[str]`, `PIL.Image, `list[PIL.Image]`, `list[dict[str, Union[str, PIL.Image]]]`) --
  The pipeline handles three types of images:

  - A string containing a HTTP(s) link pointing to an image
  - A string containing a local path to an image
  - An image loaded in PIL directly

  The pipeline accepts either a single image or a batch of images. Finally, this pipeline also supports
  the chat format (see `text`) containing images and text in this argument.
- **text** (str, list[str], `list[dict[str, Union[str, PIL.Image]]]`) --
  The text to be used for generation. If a list of strings is passed, the length of the list should be
  the same as the number of images. Text can also follow the chat format: a list of dictionaries where
  each dictionary represents a message in a conversation. Each dictionary should have two keys: 'role'
  and 'content'. 'role' should be one of 'user', 'system' or 'assistant'. 'content' should be a list of
  dictionary containing the text of the message and the type of the message. The type of the message
  can be either 'text' or 'image'. If the type is 'image', no text is needed.
- **return_tensors** (`bool`, *optional*, defaults to `False`) --
  Returns the tensors of predictions (as token indices) in the outputs. If set to
  `True`, the decoded text is not returned.
- **return_text** (`bool`, *optional*) --
  Returns the decoded texts in the outputs.
- **return_full_text** (`bool`, *optional*, defaults to `True`) --
  If set to `False` only added text is returned, otherwise the full text is returned. Cannot be
  specified at the same time as `return_text`.
- **clean_up_tokenization_spaces** (`bool`, *optional*, defaults to `True`) --
  Whether or not to clean up the potential extra spaces in the text output.
- **continue_final_message(** `bool`, *optional*) -- This indicates that you want the model to continue the
  last message in the input chat rather than starting a new one, allowing you to "prefill" its response.
  By default this is `True` when the final message in the input chat has the `assistant` role and
  `False` otherwise, but you can manually override that behaviour by setting this flag.0A list or a list of list of `dict`Each result comes as a dictionary with the following key (cannot
return a combination of both `generated_text` and `generated_token_ids`):

- **generated_text** (`str`, present when `return_text=True`) -- The generated text.
- **generated_token_ids** (`torch.Tensor`, present when `return_tensors=True`) -- The token
  ids of the generated text.
- **input_text** (`str`) -- The input text.

Generate a text given text and the image(s) passed as inputs.

**Parameters:**

model ([PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel)) : The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel).

processor ([ProcessorMixin](/docs/transformers/main/en/main_classes/processors#transformers.ProcessorMixin)) : The processor that will be used by the pipeline to encode data for the model. This object inherits from [ProcessorMixin](/docs/transformers/main/en/main_classes/processors#transformers.ProcessorMixin). Processor is a composite object that might contain `tokenizer`, `feature_extractor`, and `image_processor`.

modelcard (`str` or `ModelCard`, *optional*) : Model card attributed to the model for this pipeline.

task (`str`, defaults to `""`) : A task-identifier for the pipeline.

num_workers (`int`, *optional*, defaults to 8) : When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the number of workers to be used.

batch_size (`int`, *optional*, defaults to 1) : When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the size of the batch to use, for inference this is not always beneficial, please read [Batching with pipelines](https://huggingface.co/transformers/main_classes/pipelines.html#pipeline-batching) .

args_parser ([ArgumentHandler](/docs/transformers/main/en/internal/pipelines_utils#transformers.pipelines.ArgumentHandler), *optional*) : Reference to the object in charge of parsing supplied pipeline parameters.

device (`int`, *optional*, defaults to -1) : Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, a positive will run the model on the associated CUDA device id. You can pass native `torch.device` or a `str` too

dtype (`str` or `torch.dtype`, *optional*) : Sent directly as `model_kwargs` (just a simpler shortcut) to use the available precision for this model (`torch.float16`, `torch.bfloat16`, ... or `"auto"`)

binary_output (`bool`, *optional*, defaults to `False`) : Flag indicating if the output the pipeline should happen in a serialized format (i.e., pickle) or as the raw output data e.g. text.

**Returns:**

`A list or a list of list of `dict``

Each result comes as a dictionary with the following key (cannot
return a combination of both `generated_text` and `generated_token_ids`):

- **generated_text** (`str`, present when `return_text=True`) -- The generated text.
- **generated_token_ids** (`torch.Tensor`, present when `return_tensors=True`) -- The token
  ids of the generated text.
- **input_text** (`str`) -- The input text.

### AnyToAnyPipeline[[transformers.AnyToAnyPipeline]]

#### transformers.AnyToAnyPipeline[[transformers.AnyToAnyPipeline]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/any_to_any.py#L67)

Multimodal Generation pipeline using an `AutoModelForMultimodalLM`. This pipeline generates text given any
combination of multimodal data and text.When the underlying model is a conversational model, it can also
accept one or more chats, in which case the pipeline will operate in chat mode and will continue the
chat(s) by adding its response(s). Each chat takes the form of a list of dicts, where each dict contains
"role" and "content" keys.

Unless the model you're using explicitly sets these generation parameters in its configuration files
(`generation_config.json`), the following default values will be used:
- max_new_tokens: 256

Example:

```python
>>> from transformers import pipeline

>>> pipe = pipeline(task="any-to-any", model="google/gemma-3n-E4B-it")
>>> pipe("https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png", text="A photo of")
[{'generated_text': 'a photo of two birds'}]
```

```python
>>> from transformers import pipeline

>>> pipe = pipeline("any-to-any", model="google/gemma-3n-E4B-it")
>>> messages = [
>>>     {
>>>         "role": "user",
>>>         "content": [
>>>             {
>>>                 "type": "image",
>>>                 "url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
>>>             },
>>>             {"type": "text", "text": "Describe this image."},
>>>         ],
>>>     },
>>>     {
>>>         "role": "assistant",
>>>         "content": [
>>>             {"type": "text", "text": "There is a dog and"},
>>>         ],
>>>     },
>>> ]
>>> pipe(text=messages, max_new_tokens=20, return_full_text=False)
[{'input_text': [{'role': 'user',
    'content': [{'type': 'image',
    'url': 'https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg'},
    {'type': 'text', 'text': 'Describe this image.'}]},
{'role': 'assistant',
    'content': [{'type': 'text', 'text': 'There is a dog and'}]}],
'generated_text': ' a person in the image. The dog is sitting on the sand, and the person is sitting on'}]
```

Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

This multimodal pipeline can currently be loaded from pipeline() using the following task identifier:
"any-to-any".

See the list of available models on
[huggingface.co/models](https://huggingface.co/models?pipeline_tag=any-to-any).

__call__transformers.AnyToAnyPipeline.__call__https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/any_to_any.py#L254[{"name": "text", "val": ": typing.Union[str, list[str], list[dict]]"}, {"name": "images", "val": ": typing.Union[str, list[str], list[list[str]], ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor'], NoneType] = None"}, {"name": "videos", "val": ": typing.Union[str, list[str], list['PIL.Image.Image'], numpy.ndarray, ForwardRef('torch.Tensor'), list[numpy.ndarray], list['torch.Tensor'], list[list['PIL.Image.Image']], list[list[numpy.ndarray]], list[list['torch.Tensor']], transformers.video_utils.URL, list[transformers.video_utils.URL], list[list[transformers.video_utils.URL]], transformers.video_utils.Path, list[transformers.video_utils.Path], list[list[transformers.video_utils.Path]], NoneType] = None"}, {"name": "audio", "val": ": typing.Union[str, list[str], numpy.ndarray, ForwardRef('torch.Tensor'), collections.abc.Sequence[numpy.ndarray], collections.abc.Sequence['torch.Tensor'], NoneType] = None"}, {"name": "**kwargs", "val": ""}]- **text** (`str`, `list[str]`, `list[dict]`) --
  The text to be used for generation. If a list of strings is passed, the length of the list should be
  the same as the number of images. Text can also follow the chat format: a list of dictionaries where
  each dictionary represents a message in a conversation. Each dictionary should have two keys: 'role'
  and 'content'. 'role' should be one of 'user', 'system' or 'assistant'. 'content' should be a list of
  dictionary containing the text of the message and the type of the message.
- **images** (`str`, `list[str]`, `ImageInput`) --
  The pipeline handles three types of images:

  - A string containing a HTTP(s) link pointing to an image
  - A string containing a local path to an image
  - An image loaded in PIL directly

  The pipeline accepts either a single image or a batch of images. Finally, this pipeline also supports
  the chat format (see `text`) containing images and text in this argument.
- **videos** (`str`, `list[str]`, `VideoInput`) --
  The pipeline handles three types of videos:

  - A string containing a HTTP(s) link pointing to a video
  - A string containing a local path to a video
  - A video loaded and decoded to array format

  The pipeline accepts either a single video or a batch of videos. Finally, this pipeline also supports
  the chat format (see `text`) containing videos and text in this argument.
- **audio** (`str`, `list[str]`, `AudioInput`) --
  The pipeline handles three types of audios:

  - A string containing a HTTP(s) link pointing to an audio
  - A string containing a local path to an audio
  - An audio loaded in PIL directly

  The pipeline accepts either a single audios or a batch of audios. Finally, this pipeline also supports
  the chat format (see `text`) containing audios and text in this argument.
- **return_tensors** (`bool`, *optional*, defaults to `False`) --
  Returns the tensors of predictions (as token indices) in the outputs. If set to
  `True`, the decoded text is not returned.
- **return_text** (`bool`, *optional*) --
  Returns the decoded texts in the outputs.
- **return_full_text** (`bool`, *optional*, defaults to `True`) --
  If set to `False` only added text is returned, otherwise the full text is returned. Cannot be
  specified at the same time as `return_text`.
- **clean_up_tokenization_spaces** (`bool`, *optional*, defaults to `True`) --
  Whether or not to clean up the potential extra spaces in the text output.
- **continue_final_message(** `bool`, *optional*) -- This indicates that you want the model to continue the
  last message in the input chat rather than starting a new one, allowing you to "prefill" its response.
  By default this is `True` when the final message in the input chat has the `assistant` role and
  `False` otherwise, but you can manually override that behaviour by setting this flag.0A list or a list of list of `dict`Each result comes as a dictionary with the following key (cannot
return a combination of both `generated_text` and `generated_token_ids`):

- **generated_text** (`str`, present when `return_text=True` and `generation_mode="text") -- The generated text.
- **generated_audio** (`np.ndarray`, present when `generation_mode="audio") -- The generated audio.
- **generated_image** (`PIL.Image.Image`, present when `generation_mode="image") -- The generated image.
- **generated_token_ids** (`torch.Tensor`, present when `return_tensors=True` and `generation_mode="text") -- The token
  ids of the generated text.
- **input_text** (`str`) -- The input text.

Generate a text given text and optionally multimodal data passed as inputs.

**Parameters:**

model ([PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel)) : The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel).

processor ([ProcessorMixin](/docs/transformers/main/en/main_classes/processors#transformers.ProcessorMixin)) : The processor that will be used by the pipeline to encode data for the model. This object inherits from [ProcessorMixin](/docs/transformers/main/en/main_classes/processors#transformers.ProcessorMixin). Processor is a composite object that might contain `tokenizer`, `feature_extractor`, and `image_processor`.

modelcard (`str` or `ModelCard`, *optional*) : Model card attributed to the model for this pipeline.

task (`str`, defaults to `""`) : A task-identifier for the pipeline.

num_workers (`int`, *optional*, defaults to 8) : When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the number of workers to be used.

batch_size (`int`, *optional*, defaults to 1) : When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the size of the batch to use, for inference this is not always beneficial, please read [Batching with pipelines](https://huggingface.co/transformers/main_classes/pipelines.html#pipeline-batching) .

args_parser ([ArgumentHandler](/docs/transformers/main/en/internal/pipelines_utils#transformers.pipelines.ArgumentHandler), *optional*) : Reference to the object in charge of parsing supplied pipeline parameters.

device (`int`, *optional*, defaults to -1) : Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, a positive will run the model on the associated CUDA device id. You can pass native `torch.device` or a `str` too

dtype (`str` or `torch.dtype`, *optional*) : Sent directly as `model_kwargs` (just a simpler shortcut) to use the available precision for this model (`torch.float16`, `torch.bfloat16`, ... or `"auto"`)

binary_output (`bool`, *optional*, defaults to `False`) : Flag indicating if the output the pipeline should happen in a serialized format (i.e., pickle) or as the raw output data e.g. text.

**Returns:**

`A list or a list of list of `dict``

Each result comes as a dictionary with the following key (cannot
return a combination of both `generated_text` and `generated_token_ids`):

- **generated_text** (`str`, present when `return_text=True` and `generation_mode="text") -- The generated text.
- **generated_audio** (`np.ndarray`, present when `generation_mode="audio") -- The generated audio.
- **generated_image** (`PIL.Image.Image`, present when `generation_mode="image") -- The generated image.
- **generated_token_ids** (`torch.Tensor`, present when `return_tensors=True` and `generation_mode="text") -- The token
  ids of the generated text.
- **input_text** (`str`) -- The input text.

### MaskGenerationPipeline[[transformers.MaskGenerationPipeline]]

#### transformers.MaskGenerationPipeline[[transformers.MaskGenerationPipeline]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/mask_generation.py#L36)

Automatic mask generation for images using `SamForMaskGeneration`. This pipeline predicts binary masks for an
image, given an image. It is a `ChunkPipeline` because you can separate the points in a mini-batch in order to
avoid OOM issues. Use the `points_per_batch` argument to control the number of points that will be processed at the
same time. Default is `64`.

The pipeline works in 3 steps:
1. `preprocess`: A grid of 1024 points evenly separated is generated along with bounding boxes and point
   labels.
   For more details on how the points and bounding boxes are created, check the `_generate_crop_boxes`
   function. The image is also preprocessed using the `image_processor`. This function `yields` a minibatch of
   `points_per_batch`.

2. `forward`: feeds the outputs of `preprocess` to the model. The image embedding is computed only once.
   Calls both `self.model.get_image_embeddings` and makes sure that the gradients are not computed, and the
   tensors and models are on the same device.

3. `postprocess`: The most important part of the automatic mask generation happens here. Three steps
   are induced:
   - image_processor.postprocess_masks (run on each minibatch loop): takes in the raw output masks,
     resizes them according
   to the image size, and transforms there to binary masks.
   - image_processor.filter_masks (on each minibatch loop): uses both `pred_iou_thresh` and
     `stability_scores`. Also
   applies a variety of filters based on non maximum suppression to remove bad masks.
   - image_processor.postprocess_masks_for_amg applies the NSM on the mask to only keep relevant ones.

Example:

```python
>>> from transformers import pipeline

>>> generator = pipeline(model="facebook/sam-vit-base", task="mask-generation")
>>> outputs = generator(
...     "http://images.cocodataset.org/val2017/000000039769.jpg",
... )

>>> outputs = generator(
...     "https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png", points_per_batch=128
... )
```

Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

This segmentation pipeline can currently be loaded from [pipeline()](/docs/transformers/main/en/main_classes/pipelines#transformers.pipeline) using the following task identifier:
`"mask-generation"`.

See the list of available models on [huggingface.co/models](https://huggingface.co/models?filter=mask-generation).

__call__transformers.MaskGenerationPipeline.__call__https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/mask_generation.py#L143[{"name": "image", "val": ": typing.Union[str, ForwardRef('Image.Image'), list[str], list['Image.Image']]"}, {"name": "*args", "val": ": typing.Any"}, {"name": "**kwargs", "val": ": typing.Any"}]- **image** (`str`, `List[str]`, `PIL.Image` or `List[PIL.Image]`) --
  Image or list of images.
- **mask_threshold** (`float`, *optional*, defaults to 0.0) --
  Threshold to use when turning the predicted masks into binary values.
- **pred_iou_thresh** (`float`, *optional*, defaults to 0.88) --
  A filtering threshold in `[0,1]` applied on the model's predicted mask quality.
- **stability_score_thresh** (`float`, *optional*, defaults to 0.95) --
  A filtering threshold in `[0,1]`, using the stability of the mask under changes to the cutoff used to
  binarize the model's mask predictions.
- **stability_score_offset** (`int`, *optional*, defaults to 1) --
  The amount to shift the cutoff when calculated the stability score.
- **crops_nms_thresh** (`float`, *optional*, defaults to 0.7) --
  The box IoU cutoff used by non-maximal suppression to filter duplicate masks.
- **crops_n_layers** (`int`, *optional*, defaults to 0) --
  If `crops_n_layers>0`, mask prediction will be run again on crops of the image. Sets the number of
  layers to run, where each layer has 2**i_layer number of image crops.
- **crop_overlap_ratio** (`float`, *optional*, defaults to `512 / 1500`) --
  Sets the degree to which crops overlap. In the first crop layer, crops will overlap by this fraction of
  the image length. Later layers with more crops scale down this overlap.
- **crop_n_points_downscale_factor** (`int`, *optional*, defaults to `1`) --
  The number of points-per-side sampled in layer n is scaled down by crop_n_points_downscale_factor**n.
- **timeout** (`float`, *optional*, defaults to None) --
  The maximum time in seconds to wait for fetching images from the web. If None, no timeout is set and
  the call may block forever.0`Dict`A dictionary with the following keys:
- **mask** (`PIL.Image`) -- A binary mask of the detected object as a PIL Image of shape `(width,
  height)` of the original image. Returns a mask filled with zeros if no object is found.
- **score** (*optional* `float`) -- Optionally, when the model is capable of estimating a confidence of
  the "object" described by the label and the mask.

Generates binary segmentation masks

**Parameters:**

model ([PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel)) : The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel).

image_processor ([BaseImageProcessor](/docs/transformers/main/en/main_classes/image_processor#transformers.BaseImageProcessor)) : The image processor that will be used by the pipeline to encode data for the model. This object inherits from [BaseImageProcessor](/docs/transformers/main/en/main_classes/image_processor#transformers.BaseImageProcessor).

modelcard (`str` or `ModelCard`, *optional*) : Model card attributed to the model for this pipeline.

task (`str`, defaults to `""`) : A task-identifier for the pipeline.

num_workers (`int`, *optional*, defaults to 8) : When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the number of workers to be used.

batch_size (`int`, *optional*, defaults to 1) : When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the size of the batch to use, for inference this is not always beneficial, please read [Batching with pipelines](https://huggingface.co/transformers/main_classes/pipelines.html#pipeline-batching) .

args_parser ([ArgumentHandler](/docs/transformers/main/en/internal/pipelines_utils#transformers.pipelines.ArgumentHandler), *optional*) : Reference to the object in charge of parsing supplied pipeline parameters.

device (`int`, *optional*, defaults to -1) : Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, a positive will run the model on the associated CUDA device id. You can pass native `torch.device` or a `str` too

dtype (`str` or `torch.dtype`, *optional*) : Sent directly as `model_kwargs` (just a simpler shortcut) to use the available precision for this model (`torch.float16`, `torch.bfloat16`, ... or `"auto"`)

binary_output (`bool`, *optional*, defaults to `False`) : Flag indicating if the output the pipeline should happen in a serialized format (i.e., pickle) or as the raw output data e.g. text.

points_per_batch (*optional*, int, default to 64) : Sets the number of points run simultaneously by the model. Higher numbers may be faster but use more GPU memory.

output_bboxes_mask (`bool`, *optional*, default to `False`) : Whether or not to output the bounding box predictions.

output_rle_masks (`bool`, *optional*, default to `False`) : Whether or not to output the masks in `RLE` format

**Returns:**

``Dict``

A dictionary with the following keys:
- **mask** (`PIL.Image`) -- A binary mask of the detected object as a PIL Image of shape `(width,
  height)` of the original image. Returns a mask filled with zeros if no object is found.
- **score** (*optional* `float`) -- Optionally, when the model is capable of estimating a confidence of
  the "object" described by the label and the mask.

### VisualQuestionAnsweringPipeline[[transformers.VisualQuestionAnsweringPipeline]]

#### transformers.VisualQuestionAnsweringPipeline[[transformers.VisualQuestionAnsweringPipeline]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/visual_question_answering.py#L21)

Visual Question Answering pipeline using a `AutoModelForVisualQuestionAnswering`. This pipeline is currently only
available in PyTorch.

Unless the model you're using explicitly sets these generation parameters in its configuration files
(`generation_config.json`), the following default values will be used:
- max_new_tokens: 256

Example:

```python
>>> from transformers import pipeline

>>> oracle = pipeline(model="dandelin/vilt-b32-finetuned-vqa")
>>> image_url = "https://huggingface.co/datasets/Narsil/image_dummy/raw/main/lena.png"
>>> oracle(question="What is she wearing ?", image=image_url)
[{'score': 0.948, 'answer': 'hat'}, {'score': 0.009, 'answer': 'fedora'}, {'score': 0.003, 'answer': 'clothes'}, {'score': 0.003, 'answer': 'sun hat'}, {'score': 0.002, 'answer': 'nothing'}]

>>> oracle(question="What is she wearing ?", image=image_url, top_k=1)
[{'score': 0.948, 'answer': 'hat'}]

>>> oracle(question="Is this a person ?", image=image_url, top_k=1)
[{'score': 0.993, 'answer': 'yes'}]

>>> oracle(question="Is this a man ?", image=image_url, top_k=1)
[{'score': 0.996, 'answer': 'no'}]
```

Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

This visual question answering pipeline can currently be loaded from [pipeline()](/docs/transformers/main/en/main_classes/pipelines#transformers.pipeline) using the following task
identifiers: `"visual-question-answering", "vqa"`.

The models that this pipeline can use are models that have been fine-tuned on a visual question answering task. See
the up-to-date list of available models on
[huggingface.co/models](https://huggingface.co/models?filter=visual-question-answering).

__call__transformers.VisualQuestionAnsweringPipeline.__call__https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/visual_question_answering.py#L95[{"name": "image", "val": ": typing.Union[ForwardRef('Image.Image'), str, list['Image.Image'], list[str], ForwardRef('KeyDataset')]"}, {"name": "question", "val": ": str | list[str] | None = None"}, {"name": "**kwargs", "val": ""}]- **image** (`str`, `list[str]`, `PIL.Image`, `list[PIL.Image]` or `KeyDataset`) --
  The pipeline handles three types of images:

  - A string containing a http link pointing to an image
  - A string containing a local path to an image
  - An image loaded in PIL directly

  The pipeline accepts either a single image or a batch of images. If given a single image, it can be
  broadcasted to multiple questions.
  For dataset: the passed in dataset must be of type `transformers.pipelines.pt_utils.KeyDataset`
  Example:
```python
>>> from transformers.pipelines.pt_utils import KeyDataset
>>> from datasets import load_dataset

>>> dataset = load_dataset("detection-datasets/coco")
>>> oracle(image=KeyDataset(dataset, "image"), question="What's in this image?")

```
- **question** (`str`, `list[str]`) --
  The question(s) asked. If given a single question, it can be broadcasted to multiple images.
  If multiple images and questions are given, each and every question will be broadcasted to all images
  (same effect as a Cartesian product)
- **top_k** (`int`, *optional*, defaults to 5) --
  The number of top labels that will be returned by the pipeline. If the provided number is higher than
  the number of labels available in the model configuration, it will default to the number of labels.
- **timeout** (`float`, *optional*, defaults to None) --
  The maximum time in seconds to wait for fetching images from the web. If None, no timeout is set and
  the call may block forever.0A dictionary or a list of dictionaries containing the result. The dictionaries contain the following keys- **label** (`str`) -- The label identified by the model.
- **score** (`int`) -- The score attributed by the model for that label.

Answers open-ended questions about images. The pipeline accepts several types of inputs which are detailed
below:

- `pipeline(image=image, question=question)`
- `pipeline({"image": image, "question": question})`
- `pipeline([{"image": image, "question": question}])`
- `pipeline([{"image": image, "question": question}, {"image": image, "question": question}])`

**Parameters:**

model ([PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel)) : The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel).

tokenizer ([PreTrainedTokenizer](/docs/transformers/main/en/main_classes/tokenizer#transformers.PythonBackend)) : The tokenizer that will be used by the pipeline to encode data for the model. This object inherits from [PreTrainedTokenizer](/docs/transformers/main/en/main_classes/tokenizer#transformers.PythonBackend).

image_processor ([BaseImageProcessor](/docs/transformers/main/en/main_classes/image_processor#transformers.BaseImageProcessor)) : The image processor that will be used by the pipeline to encode data for the model. This object inherits from [BaseImageProcessor](/docs/transformers/main/en/main_classes/image_processor#transformers.BaseImageProcessor).

modelcard (`str` or `ModelCard`, *optional*) : Model card attributed to the model for this pipeline.

task (`str`, defaults to `""`) : A task-identifier for the pipeline.

num_workers (`int`, *optional*, defaults to 8) : When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the number of workers to be used.

batch_size (`int`, *optional*, defaults to 1) : When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the size of the batch to use, for inference this is not always beneficial, please read [Batching with pipelines](https://huggingface.co/transformers/main_classes/pipelines.html#pipeline-batching) .

args_parser ([ArgumentHandler](/docs/transformers/main/en/internal/pipelines_utils#transformers.pipelines.ArgumentHandler), *optional*) : Reference to the object in charge of parsing supplied pipeline parameters.

device (`int`, *optional*, defaults to -1) : Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, a positive will run the model on the associated CUDA device id. You can pass native `torch.device` or a `str` too

dtype (`str` or `torch.dtype`, *optional*) : Sent directly as `model_kwargs` (just a simpler shortcut) to use the available precision for this model (`torch.float16`, `torch.bfloat16`, ... or `"auto"`)

binary_output (`bool`, *optional*, defaults to `False`) : Flag indicating if the output the pipeline should happen in a serialized format (i.e., pickle) or as the raw output data e.g. text.

**Returns:**

`A dictionary or a list of dictionaries containing the result. The dictionaries contain the following keys`

- **label** (`str`) -- The label identified by the model.
- **score** (`int`) -- The score attributed by the model for that label.

## Parent class: `Pipeline`[[transformers.Pipeline]]

#### transformers.Pipeline[[transformers.Pipeline]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/base.py#L746)

The Pipeline class is the class from which all pipelines inherit. Refer to this class for methods shared across
different pipelines.

Base class implementing pipelined operations. Pipeline workflow is defined as a sequence of the following
operations:

Input -> Tokenization -> Model Inference -> Post-Processing (task dependent) -> Output

Pipeline supports running on CPU or GPU through the device argument (see below).

Some pipeline, like for instance [FeatureExtractionPipeline](/docs/transformers/main/en/main_classes/pipelines#transformers.FeatureExtractionPipeline) (`'feature-extraction'`) output large tensor object
as nested-lists. In order to avoid dumping such large structure as textual data we provide the `binary_output`
constructor argument. If set to `True`, the output will be stored in the pickle format.

check_model_typetransformers.Pipeline.check_model_typehttps://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/base.py#L1093[{"name": "supported_models", "val": ": list[str] | dict"}]- **supported_models** (`list[str]` or `dict`) --
  The list of models supported by the pipeline, or a dictionary with model class values.0

Check if the model class is in supported by the pipeline.

**Parameters:**

model ([PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel)) : The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel).

tokenizer ([PreTrainedTokenizer](/docs/transformers/main/en/main_classes/tokenizer#transformers.PythonBackend)) : The tokenizer that will be used by the pipeline to encode data for the model. This object inherits from [PreTrainedTokenizer](/docs/transformers/main/en/main_classes/tokenizer#transformers.PythonBackend).

feature_extractor ([SequenceFeatureExtractor](/docs/transformers/main/en/main_classes/feature_extractor#transformers.SequenceFeatureExtractor)) : The feature extractor that will be used by the pipeline to encode data for the model. This object inherits from [SequenceFeatureExtractor](/docs/transformers/main/en/main_classes/feature_extractor#transformers.SequenceFeatureExtractor).

image_processor ([BaseImageProcessor](/docs/transformers/main/en/main_classes/image_processor#transformers.BaseImageProcessor)) : The image processor that will be used by the pipeline to encode data for the model. This object inherits from [BaseImageProcessor](/docs/transformers/main/en/main_classes/image_processor#transformers.BaseImageProcessor).

processor ([ProcessorMixin](/docs/transformers/main/en/main_classes/processors#transformers.ProcessorMixin)) : The processor that will be used by the pipeline to encode data for the model. This object inherits from [ProcessorMixin](/docs/transformers/main/en/main_classes/processors#transformers.ProcessorMixin). Processor is a composite object that might contain `tokenizer`, `feature_extractor`, and `image_processor`.

modelcard (`str` or `ModelCard`, *optional*) : Model card attributed to the model for this pipeline.

task (`str`, defaults to `""`) : A task-identifier for the pipeline.

num_workers (`int`, *optional*, defaults to 8) : When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the number of workers to be used.

batch_size (`int`, *optional*, defaults to 1) : When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the size of the batch to use, for inference this is not always beneficial, please read [Batching with pipelines](https://huggingface.co/transformers/main_classes/pipelines.html#pipeline-batching) .

args_parser ([ArgumentHandler](/docs/transformers/main/en/internal/pipelines_utils#transformers.pipelines.ArgumentHandler), *optional*) : Reference to the object in charge of parsing supplied pipeline parameters.

device (`int`, *optional*, defaults to -1) : Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, a positive will run the model on the associated CUDA device id. You can pass native `torch.device` or a `str` too

dtype (`str` or `torch.dtype`, *optional*) : Sent directly as `model_kwargs` (just a simpler shortcut) to use the available precision for this model (`torch.float16`, `torch.bfloat16`, ... or `"auto"`)

binary_output (`bool`, *optional*, defaults to `False`) : Flag indicating if the output the pipeline should happen in a serialized format (i.e., pickle) or as the raw output data e.g. text.
#### device_placement[[transformers.Pipeline.device_placement]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/base.py#L1029)

Context Manager allowing tensor allocation on the user-specified device.

Examples:

```python
# Explicitly ask for tensor allocation on CUDA device :0
pipe = pipeline(..., device=0)
with pipe.device_placement():
    # Every tensor allocation will be done on the request device
    output = pipe(...)
```

**Returns:**

Context manager
#### ensure_tensor_on_device[[transformers.Pipeline.ensure_tensor_on_device]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/base.py#L1061)

Ensure PyTorch tensors are on the specified device.

**Parameters:**

inputs (keyword arguments that should be `torch.Tensor`, the rest is ignored) : The tensors to place on `self.device`.

Recursive on lists **only**. --

**Returns:**

``dict[str, torch.Tensor]``

The same as `inputs` but on the proper device.
#### postprocess[[transformers.Pipeline.postprocess]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/base.py#L1160)

Postprocess will receive the raw outputs of the `_forward` method, generally tensors, and reformat them into
something more friendly. Generally it will output a list or a dict or results (containing just strings and
numbers).
#### predict[[transformers.Pipeline.predict]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/base.py#L1008)

Scikit / Keras interface to transformers' pipelines. This method will forward to __call__().
#### preprocess[[transformers.Pipeline.preprocess]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/base.py#L1139)

Preprocess will take the `input_` of a specific pipeline and return a dictionary of everything necessary for
`_forward` to run properly. It should contain at least one tensor, but might have arbitrary other items.
#### push_to_hub[[transformers.Pipeline.push_to_hub]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/utils/hub.py#L711)

Upload the pipeline file to the 🤗 Model Hub.

Examples:

```python
from transformers import pipeline

pipe = pipeline("google-bert/bert-base-cased")

# Push the pipe to your namespace with the name "my-finetuned-bert".
pipe.push_to_hub("my-finetuned-bert")

# Push the pipe to an organization with the name "my-finetuned-bert".
pipe.push_to_hub("huggingface/my-finetuned-bert")
```

**Parameters:**

repo_id (`str`) : The name of the repository you want to push your pipe to. It should contain your organization name when pushing to a given organization.

commit_message (`str`, *optional*) : Message to commit while pushing. Will default to `"Upload pipe"`.

commit_description (`str`, *optional*) : The description of the commit that will be created

private (`bool`, *optional*) : Whether to make the repo private. If `None` (default), the repo will be public unless the organization's default is private. This value is ignored if the repo already exists.

token (`bool` or `str`, *optional*) : The token to use as HTTP bearer authorization for remote files. If `True` (default), will use the token generated when running `hf auth login` (stored in `~/.huggingface`).

revision (`str`, *optional*) : Branch to push the uploaded files to.

create_pr (`bool`, *optional*, defaults to `False`) : Whether or not to create a PR with the uploaded files or directly commit.

max_shard_size (`int` or `str`, *optional*, defaults to `"50GB"`) : Only applicable for models. The maximum size for a checkpoint before being sharded. Checkpoints shard will then be each of size lower than this size. If expressed as a string, needs to be digits followed by a unit (like `"5MB"`).

tags (`list[str]`, *optional*) : List of tags to push on the Hub.
#### save_pretrained[[transformers.Pipeline.save_pretrained]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/base.py#L953)

Save the pipeline's model and tokenizer.

**Parameters:**

save_directory (`str` or `os.PathLike`) : A path to the directory where to saved. It will be created if it doesn't exist.

kwargs (`dict[str, Any]`, *optional*) : Additional key word arguments passed along to the [push_to_hub()](/docs/transformers/main/en/main_classes/model#transformers.utils.PushToHubMixin.push_to_hub) method.
#### transform[[transformers.Pipeline.transform]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/base.py#L1002)

Scikit / Keras interface to transformers' pipelines. This method will forward to __call__().
