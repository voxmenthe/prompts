# Searching Data

## Searching with Shelf Content Intelligence

This tutorial demonstrates how to efficiently search for content items using the Shelf Content Intelligence API in Python. By employing specific search criteria or keywords, you can swiftly find the information you need. We'll go over initializing the `ContentIntelligenceClient` object and executing a focused search with filters.

### Environment Setup

First, ensure your environment is prepared with the necessary configurations. You should have:

* The `python-dotenv` package to manage environment variables.
* Your `SHELF_API_TOKEN` securely stored in a `.env` file.

Load and initialize the `ContentIntelligenceClient` object as follows:

```python
from dotenv import load_dotenv
from shelf.ci.client import ContentIntelligenceClient

# Load environment variables
load_dotenv()

# Initialize the ContentIntelligenceClient object
ci = ContentIntelligenceClient()
```

### Conducting a Search

With the `ContentIntelligenceClient` object set up, you can proceed to search for content items. You can tailor the search based on your needs, such as looking for documents containing specific keywords, or filtering by tags or custom fields.

#### Example: Searching Content Items with Filters

```python
# Define your search query and filters
search_query = "training module"
filters = {
    "search_language": "en",
    "created_after": "2024-01-01",
    "parent_id": "8a22a6bb-d31c-4c66-8ba2-f442bd5851db",
}

# Perform the search with both query and filters
search_results = ci.content_items.search(query=search_query, **filters)

# Iterate over and print out the search results
for item in search_results:
    print(item)
```

In this example, `search_query` defines what you're looking for, while `filters` allows you to narrow down the results based on additional criteria, such as the `search_language` of the content, creation date (`created_after`) and content location (`parent_id`).

### Search sections

#### Semantic sections

Basic sections search API

```python
# Example of performing a custom POST request
search_query = "How to build powerful RAG pipelines?"

result = ci.client.post(
    path="/cil-search/content/semantic-sections",
    json={
        "query": search_query,
        "size": 5,
        "origin": "shelf",
        "format": "markdown",
        "sortBy": "RELEVANCE",
        "sortOrder": "DESC",
    }
)
```

Chunks API

```python
# Example of performing a custom POST request
search_query = "How to build powerful RAG pipelines?"

result = ci.client.post(
    path="/cil-search/content/sections",
    json={
        "query": search_query,
        "size": 5,
        "origin": "shelf",
        "format": "markdown",
    }
)
```

GEN APP Search API for chunks with enrichments and filters

```python
# Example of performing a custom POST request
search_query = "How to build powerful RAG pipelines?"

result = ci.client.post(
    path="/cip-genai/content/search",
    json={
        "userInput": search_query,
        "filters": {
            "content": [],
            "quality": []
        },
        "enrichments": [
            "acronyms.disambiguation"
        ]
    }
)
```

### Conclusion

You've now seen how to execute a targeted search for content items using Shelf Content Intelligence. This method can significantly reduce the time you spend looking for specific pieces of information by allowing you to apply a range of filters to your search. Leveraging these techniques can improve the efficiency of content retrieval within your applications or workflows.
