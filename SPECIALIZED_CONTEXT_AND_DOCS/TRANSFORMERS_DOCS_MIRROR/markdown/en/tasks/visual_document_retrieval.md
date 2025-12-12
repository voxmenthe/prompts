# Visual document retrieval

Documents can contain multimodal data if they include charts, tables, and visuals in addition to text. Retrieving information from these documents is challenging because text retrieval models alone can’t handle visual data and image retrieval models lack the granularity and document processing capabilities.

Visual document retrieval can help retrieve information from all types of documents, including multimodal retrieval augmented generation (RAG). These models accept documents (as images) and texts and calculates the similarity scores between them.

This guide demonstrates how to index and retrieve documents with [ColPali](../model_doc/colpali).

For large scale use cases, you may want to index and retrieve documents with a vector database.

Make sure Transformers and Datasets is installed.


```
pip install -q datasets transformers
```

We will index a dataset of documents related to UFO sightings. We filter the examples where our column of interest is missing. It contains several columns, we are interested in the column `specific_detail_query` where it contains short summary of the document, and `image` column that contains our documents.


```
from datasets import load_dataset

dataset = load_dataset("davanstrien/ufo-ColPali")
dataset = dataset["train"]
dataset = dataset.filter(lambda example: example["specific_detail_query"] is not None)
dataset
```


```
Dataset({
    features: ['image', 'raw_queries', 'broad_topical_query', 'broad_topical_explanation', 'specific_detail_query', 'specific_detail_explanation', 'visual_element_query', 'visual_element_explanation', 'parsed_into_json'],
    num_rows: 2172
})
```

Let’s load the model and the tokenizer.


```
import torch
from transformers import ColPaliForRetrieval, ColPaliProcessor, infer_device

device = infer_device()

model_name = "vidore/colpali-v1.2-hf"

processor = ColPaliProcessor.from_pretrained(model_name)

model = ColPaliForRetrieval.from_pretrained(
    model_name,
    dtype=torch.bfloat16,
    device_map="auto",
).eval()
```

Pass the text query to the processor and return the indexed text embeddings from the model. For image-to-text search, replace the `text` parameter in [ColPaliProcessor](/docs/transformers/v4.56.2/en/model_doc/colpali#transformers.ColPaliProcessor) with the `images` parameter to pass images.


```
inputs = processor(text="a document about Mars expedition").to(model.device)
with torch.no_grad():
  text_embeds = model(**inputs, return_tensors="pt").embeddings
```

Index the images offline, and during inference, return the query text embeddings to get its closest image embeddings.

Store the image and image embeddings by writing them to the dataset with [map](https://huggingface.co/docs/datasets/v4.1.0/en/package_reference/main_classes#datasets.Dataset.map) as shown below. Add an `embeddings` column that contains the indexed embeddings. ColPali embeddings take up a lot of storage, so remove them from the accelerator and store them in the CPU as NumPy vectors.


```
ds_with_embeddings = dataset.map(lambda example: {'embeddings': model(**processor(images=example["image"]).to(devide), return_tensors="pt").embeddings.to(torch.float32).detach().cpu().numpy()})
```

For online inference, create a function to search the image embeddings in batches and retrieve the k-most relevant images. The function below returns the indices in the dataset and their scores for a given indexed dataset, text embeddings, number of top results, and the batch size.


```
def find_top_k_indices_batched(dataset, text_embedding, processor, k=10, batch_size=4):
    scores_and_indices = []

    for start_idx in range(0, len(dataset), batch_size):

        end_idx = min(start_idx + batch_size, len(dataset))
        batch = dataset[start_idx:end_idx]        
        batch_embeddings = [torch.tensor(emb[0], dtype=torch.float32) for emb in batch["embeddings"]]
        scores = processor.score_retrieval(text_embedding.to("cpu").to(torch.float32), batch_embeddings)

        if hasattr(scores, "tolist"):
            scores = scores.tolist()[0]

        for i, score in enumerate(scores):
            scores_and_indices.append((score, start_idx + i))

    sorted_results = sorted(scores_and_indices, key=lambda x: -x[0])

    topk = sorted_results[:k]
    indices = [idx for _, idx in topk]
    scores = [score for score, _ in topk]

    return indices, scores
```

Generate the text embeddings and pass them to the function above to return the dataset indices and scores.


```
with torch.no_grad():
  text_embeds = model(**processor(text="a document about Mars expedition").to(model.device), return_tensors="pt").embeddings
indices, scores = find_top_k_indices_batched(ds_with_embeddings, text_embeds, processor, k=3, batch_size=4)
print(indices, scores)
```


```
([440, 442, 443],
 [14.370786666870117,
  13.675487518310547,
  12.9899320602417])
```

Display the images to view the Mars related documents.


```
for i in indices:
  display(dataset[i]["image"])
```

![Document 1](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/doc_1.png) ![Document 2](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/doc_2.png) ![Document 3](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/doc_3.png)

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/tasks/visual_document_retrieval.md)
