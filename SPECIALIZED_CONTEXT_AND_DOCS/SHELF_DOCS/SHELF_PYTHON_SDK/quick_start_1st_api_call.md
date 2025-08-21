# Quick Start: Making 1st API call

## First API Call Using Shelf Content Intelligence Python SDK

Welcome to the quick start guide for getting up and running with the Shelf Content Intelligence Python SDK. In just a few steps, learn how to perform API calls to access and analyze your content like never before!

***

## Tutorial Colab notebook

We encourage you to explore the package with our [Tutorial Colab notebook](https://colab.research.google.com/drive/1srG2Lih8xbEit50rJ_M1hhedGdB_dhxL?usp=sharing), which showcases the package in action.

***

### Prerequisites

Before we dive in, make sure you have the following ready:

* A Python environment is set up on your machine.
* Access credentials for Shelf Content Intelligence, including a Shelf API token.
* The Shelf Content Intelligence Python SDK is installed in your environment.

If you need some pointers on setting these up, consider checking out the following guides:

* [Setting Up Shelf API Token](https://github.com/shelfio/shelf-dev-portal/blob/master/gitbook/content-intelligence/README.md)
* [Installing Shelf Content Intelligence Python SDK](installation-and-making-1st-api-call)

### Setting Up Your Environment

#### 1. Prepare Environment Variables

For a smooth start, ensure you have a `.env` file in your project's root directory with the following content:

```plaintext
SHELF_API_TOKEN=your_api_token_here
SHELF_API_URL=https://your_api_url_here
```

This step stores your API token and URL securely.

**Note:** Ensure that the environment `.env` file does not have quotation marks `"`.

\- `SHELF_API_TOKEN="EXAMPLE_TOKEN"` - <mark style="color:red;">incorrect</mark>

\- `SHELF_API_TOKEN=EXAMPLE_TOKEN`- <mark style="color:green;">correct</mark>

#### 2. Manage Environment Variables Easily

The `python-dotenv`package comes in handy to manage your environmental variables conveniently. Install it by entering in your terminal:

```bash
pip install python-dotenv
```

#### 3. Initialize the SDK

With your environment variables set, it's time to initialize the `ContentIntelligenceClient` object. Here's how you do it in your Python code:

```python
from dotenv import load_dotenv
from shelf.ci.client import ContentIntelligenceClient

# Load environment variables
load_dotenv()

# Initialize Shelf Content Intelligence SDK
ci = ContentIntelligenceClient()
```

Alternatively, you can pass the `token` and `api_url` directly:

```python
# Initialize Shelf Content Intelligence SDK with explicit token and URL
ci = ContentIntelligenceClient(token="SHELF_API_TOKEN", api_url="https://SHELF_API_URL")
```

### Example API Call: Listing Content Items

Now, let's explore how we can use the SDK to list content items based on a specific criterion.

#### Fetch and Print Content Items

Imagine you need a list of content items. You can accomplish this using the `ci.content_items.list()` method as shown below:

```python
# List all content items
content_items = ci.content_items.list()

# Iterate and print each content item's details
for item in content_items:
    print(item)
```

This snippet fetches content items and prints the details of each item in a readable format.

### Advanced: Perform Custom REST Requests

While the Python SDK is comprehensive, you might occasionally need to perform REST requests directly—for functionality not yet available in the SDK or for specialized requests.

```python
# Assuming environment variables and the `ContentIntelligenceClient` object (ci) are already set up as before

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

This snippet showcases how you can conduct a custom POST request to the Shelf Content Intelligence REST API, potentially expanding the use cases you can handle with the SDK.

***

We hope this guide helps you get started with the Shelf Content Intelligence Python SDK smoothly. As you embark on your journey to unlock the full potential of your content, remember to explore more use cases and examples that can inspire what you build next.

Happy coding!
