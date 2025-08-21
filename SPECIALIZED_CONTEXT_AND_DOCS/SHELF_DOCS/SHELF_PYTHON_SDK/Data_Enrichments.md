# Data Enrichments

## Instructions on How to Use Enrichments in Shelf Content Intelligence

This code is used to list content items and their enrichments from the Shelf Content Intelligence (CI) platform. Here's how to use it:

### Prerequisites

1. Python 3.9 or higher installed on your machine.
2.  The `dotenv` Python package installed. You can install these using pip:

    ```
    pip install python-dotenv
    ```
3. A `.env` file located at `shelf-python/.env` containing your Shelf CI credentials.

### Steps

1.  Import the necessary modules:

    ```python
    from dotenv import load_dotenv
    ```
2.  Load your Shelf CI credentials from the `.env` file:

    ```python
    load_dotenv(dotenv_path="shelf-python/.env")
    ```
3.  Initialize the ContentIntelligenceClient object:

    ```python
    from shelf.ci.client import ContentIntelligenceClient
    ci = ContentIntelligenceClient()
    ```
4.  List all content items:

    ```python
    content_items = ci.content_items.list()
    ```
5.  For each content item, list its enrichments and semantic sections:

    ```python
    for item in content_items:
        for enrichment in ci.content_items.get_enrichments(item.external_id):
            print(enrichment)

        for section in ci.sections.list(item.external_id):
            for enrichment in ci.sections.get_enrichments(item.external_id, section.id):
                print(enrichment)
    ```
6.  List topics for each content item:

    ```python
    for content_item in ci.content_items.list():
        for topic in ci.content_items.get_topics(content_item.external_id):
            print(topic)
    ```
7.  List content items for a specific topic:

    ```python
    topic_id = "..." # "topic_id"
    for item in ci.topics.list_items(topic_id):
        print(item)
    ```
8.  Enrich text with a specific enrichment:

    ```python
    for res in ci.enrich(["what is NASDAQ ?"], ["acronyms.disambiguation"]):
        print(res)
    ```
9.  Get Image Enrichment:

    ```python
    def get_image_enrichments(item_id) -> list[(str, dict)]:
        content_item = ci.content_items.get(item_id=item_id)
        return [
            (section.id, enrichment.data)
            for section in ci.sections.list(content_item.external_id)
            for enrichment in ci.sections.get_enrichments(content_item.external_id, section.id)
            if enrichment.kind == "section.ocr_summaries"
        ]


    get_image_enrichments("content_external_id")
    ```

The `.env` file should contain the necessary credentials to access the Shelf CI platform.
