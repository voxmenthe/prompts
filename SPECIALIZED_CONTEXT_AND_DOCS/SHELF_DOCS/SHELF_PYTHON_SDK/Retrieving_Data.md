# Retrieving Data

## Retrieving Content Items in Shelf Content Intelligence

In this tutorial, we'll explore how to retrieve and list content items from Shelf Content Intelligence. You'll learn to initialize the `ShelfContentIntelligenceClient` object, configure it with your API token, and list content items using specific filtering criteria.

### Prerequisites

Before you begin, ensure you have the following:

* Python environment setup
* Access to Shelf Content Intelligence and an API token
* `python-dotenv` package for loading environment variables
* Shelf Content Intelligence Python SDK installed in your environment

Refer to the [Quick Start: Making the 1st API call](https://docs.shelf.io/dev-portal/python-sdk/quick-start-making-1st-api-call) for detailed setup instructions.

#### Initialize the ShelfContentIntelligence Object

First, create a `.env` file in your project directory and populate it with your Shelf API token and URL:

```plaintext
SHELF_API_TOKEN=your_api_token_here
SHELF_API_URL=your_api_url_here
```

Next, use the following Python code snippet to initialize the `ContentIntelligenceClient` object by loading the API token from your `.env` file:

```python
import os

from dotenv import load_dotenv
from shelf.ci.client import ContentIntelligenceClient

load_dotenv()

ci = ContentIntelligenceClient()
```

### Content Items

Now that your `ContentIntelligenceClient` object is ready, let's explore how to query and list content items effectively.

#### Retrieve Content Items

If your goal is to list a handful of content items, you can achieve this easily by invoking the `content_items.list` method and traversing through the returned items:

```python
content_items = ci.content_items.list(items_limit=5)
for item in content_items:
    print(item)
```

This snippet fetches and displays details for the first 5 content items. To fetch all available items, omit the `items_limit` parameter.

Our SDK offers robust filtering capabilities. For example, if you need items from a specific time range, items within certain locations like `parent_id` or `collection_ids`, items in a particular language, or to include/exclude specific items, it's all seamlessly achievable. For a comprehensive list of filtering options, refer to our API documentation.

```python
content_items = ci.content_items.list(
    created_after="2024-01-01",
    updated_after="2024-01-01",
    created_before="2024-01-31",
    updated_before="2024-01-31",
    parent_id="<parent-id>",
    ids_to_include=["<external-item-id>"],
    ids_to_exclude=["<other-external-item-id>"]
)
```

#### Delving Into Semantic Sections of Content Items

A crucial aspect of content items is their semantic sections, which might be pivotal for generating answers or enhancing search capabilities. To parse these sections and extract their content in a particular format, such as markdown, follow the example below:

```python
for item in ci.content_items.list():
    for section in ci.sections.list(item_id=item.external_id):
        content = ci.sections.retrieve(item_id=item.external_id, section_id=section.id, content_format="markdown")
        print({"content": content, "metadata": dict(section)})
```

#### Exploring Chunks

In scenarios where detailed examination of sections is needed, you may want to break down the sections into smaller, more manageable chunks. This can be done as follows:

```python
for item in ci.content_items.list():
    for chunk in ci.chunks.list(item_id=item.external_id):
        print(chunk)
```

### Get Content Item Attachments

You can also retrieve content item attachments. First, check what kind of attachments are available for a specific content item:

```python
item_id = "d5b27bd3-9228-4f5e-9c5a-9894d7748cf1"
content_item = ci.content_items.get(item_id)
print(content_item.attachments)
```

Among the content item fields, you will find `attachments`, which is a list of dictionaries. Each dictionary contains the attachment's metadata, including the `attachmentId`, `extension`, and a list of available conversions (e.g., text, md, html, etc.). Once you have the `attachmentId`, you can retrieve the attachment URL to download it:

```python
attachment_id = "72eab2beec54b7a633165321aa3007f5"
url = ci.content_items._get_attachment_url(item_id, attachment_id)
```

Remember, the URL is temporary and will expire after some time.

Now, you can use it to download the attachment:

```python
import requests
response = requests.get(url)
with open("file_name.extension", "wb") as file:
    file.write(response.content)
```

### Conclusion

Congratulations! You now know how to configure the `ContentIntelligenceClient` with your API token, list content items using various filters, dive into semantic sections, retrieve technical chunks, and download attachments from Shelf Content Intelligence.

This foundation allows you to integrate Shelf Content Intelligence with your applications seamlessly, enabling robust content management and retrieval functionalities. Feel free to explore further and customize the filtering criteria as per your requirements.
