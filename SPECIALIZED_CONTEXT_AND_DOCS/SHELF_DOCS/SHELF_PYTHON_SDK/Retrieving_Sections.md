# Retrieving Sections

## Tutorial: How to Retrieve All Account Semantic Sections

In this tutorial, we will walk through the process of retrieving all account semantic sections using the Shelf Content Intelligence SDK. We'll be using Python and several libraries, including `dotenv`, `tqdm`, and `concurrent.futures`.

### Prerequisites

1. **Python installed**: You should have Python installed on your machine.
2. **Libraries installed**:
   * `dotenv`: For loading environment variables.
   * `tqdm`: For progress bars.
   * `shelf.ci.client`: For accessing the Shelf Content IntelligenceClient.

### Step 1: Import Libraries

```python
from dotenv import load_dotenv
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from shelf.ci.client import ContentIntelligenceClient
```

* `dotenv`: Helps in loading environment variables from a `.env` file.
* `tqdm`: Provides a progress bar for loops.
* `concurrent.futures`: Contains `ThreadPoolExecutor` for running tasks concurrently.
* `shelf.ci.client`: Contains `ContentIntelligenceClient` to interact with the Shelf API.

### Step 2: Load Environment Variables

```python
# Load environment variables
load_dotenv()
```

* `load_dotenv()`: Loads environment variables from a `.env` file into Python's `os.environ`.

### Step 3: Initialize the Shelf Content Intelligence SDK

```python
# Initialize Shelf Content Intelligence SDK
ci = ContentIntelligenceClient()
```

* `ContentIntelligenceClient()`: Initializes the client to interact with the Shelf Content Intelligence API.

### Step 4: Function to Fetch Section Data

```python
def fetch_section_data(content_item, ci):
    content_item_id = content_item.external_id
    try:
        semantic_sections_iterator = ci.sections.list(item_id=content_item_id)
        sections_data = []
        for section in semantic_sections_iterator:
            semantic_section_id = section.id
            section_content = ci.sections.retrieve(content_item_id, semantic_section_id)
            metadata = {
                "content_item_id": content_item_id,
                "content_item_title": content_item.title,
                "section_id": semantic_section_id,
                "section_title": section.title,
                "section_level": section.level,
                "section_order": section.order,
                "section_type": section.type,
            }
            sections_data.append({
                "metadata": metadata,
                "content": section_content,
            })
        return sections_data
    except Exception as e:
        print(f"Error fetching data for content_item_id {content_item_id}: {e}")
        return []
```

* **Function Purpose**: Fetch the semantic sections of a given content item.
* **Parameters**:
  * `content_item`: An instance representing a content item.
  * `ci`: Initialized Shelf Content Intelligence Client.
* **Try Block**: Attempts to fetch sections for a content item and their detailed content.
* **Exception Handling**: Catches and prints exceptions if any occur.

### Step 5: Function to Retrieve Semantic Sections

```python
def retrieve_semantic_sections(content_items, ci, max_workers=10):
    semantic_sections = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_item = {executor.submit(fetch_section_data, content_item, ci): content_item for content_item in content_items}

        with tqdm(total=len(content_items)) as pbar:
            for future in as_completed(future_to_item):
                content_item = future_to_item[future]
                try:
                    sections_data = future.result()
                    semantic_sections.extend(sections_data)
                except Exception as e:
                    print(f"Failed to retrieve data for {content_item.external_id}: {e}")
                finally:
                    pbar.update(1)
    return semantic_sections
```

* **Function Purpose**: Retrieve semantic sections for multiple content items concurrently.
* **Parameters**:
  * `content_items`: List of content items.
  * `ci`: Initialized Shelf Content Intelligence Client.
  * `max_workers`: Maximum number of threads to use for concurrent execution.
* **ThreadPoolExecutor**: Manages a pool of threads.
* **tqdm**: Displays a progress bar for the number of content items processed.
* **Exception Handling**: Catches and prints exceptions if any occur during the retrieval of section data.

### Step 6: Pull Content Items and Fetch All Sections

```python
content_items = list(ci.content_items.list())

semantic_sections = retrieve_semantic_sections(content_items, ci, max_workers=10)
```

* **Fetch Content Items**: Retrieves all content items from the account.
* **Retrieve Semantic Sections**: Calls the `retrieve_semantic_sections` function to fetch sections for all content items concurrently.

### Conclusion

In this tutorial, we covered how to use the Shelf Content Intelligence SDK to retrieve semantic sections for content items. By leveraging the power of concurrent futures and progress bars, we can efficiently handle multiple content items at scale.
