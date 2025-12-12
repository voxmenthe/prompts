# Environment Variables

## HF\_ENABLE\_PARALLEL\_LOADING

By default this is disabled. Enables the loading of torch and safetensor based weights to be loaded in parallel. Can decrease the time to load large models significantly, often times producing speed ups around ~50%.

Can be set to a string equal to `"false"` or `"true"`. e.g. `os.environ["HF_ENABLE_PARALLEL_LOADING"] = "true"`.

e.g. `facebook/opt-30b` on an AWS EC2 g4dn.metal instance can be made to load in ~30s with this enabled vs ~55s without it.

Profile before committing to using this environment variable, this will not produce speed ups for smaller models.


```
import os

os.environ["HF_ENABLE_PARALLEL_LOADING"] = "true"

from transformers import pipeline

model = pipeline(task="text-generation", model="facebook/opt-30b", device_map="auto")
```

## HF\_PARALLEL\_LOADING\_WORKERS

Determines how many threads should be used when parallel loading is enabled. Default is `8`.

If the number of files that are being loaded is less than the number of threads specified, the number that is actually spawned will be equal to the number of files.

e.g. If you specify 8 workers, and there are only 2 files, only 2 workers will be spawned.

Tune as you see fit.


```
import os

os.environ["HF_ENABLE_PARALLEL_LOADING"] = "true"
os.environ["HF_PARALLEL_LOADING_WORKERS"] = "4"

from transformers import pipeline

model = pipeline(task="text-generation", model="facebook/opt-30b", device_map="auto")
```

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/reference/environment_variables.md)
