from shelf.ci.client import ContentIntelligenceClient

def format_source_informarion(sections: list[dict], filters_outcome: list[dict]) -> str:
    """
    Formats the source information based on the provided sections and filter outcomes.

    This function extracts metadata from the given sections, counts the number of documents
    skipped based on the filters outcome, and formats the source documents. It then combines
    this information into a single formatted string which includes the number of documents
    filtered out.

    Args:
        sections (list[dict]): A list of dictionaries, where each dictionary contains information
            about a section of the source documents.
        filters_outcome (list[dict]): A list of dictionaries, where each dictionary represents
            the outcome of applying a filter to the source documents, including information
            about which documents were skipped.

    Returns:
        str: A formatted string that includes the metadata of the sections, the formatted source
            documents, and the number of documents that were filtered out.
    """
    metadata = extract_sections_metadata(sections)
    filtered_docs_count = count_skipped_items(filters_outcome)
    formatted_source_output = format_source_documents(metadata)
    formatted_source_output += f"\n\n*{filtered_docs_count} docs filtered out.*"
    return formatted_source_output


def count_skipped_items(filters_outcome: list[dict]) -> int:
    """
    Counts the total number of skipped items based on the filters outcome.

    Args:
        filters_outcome (list[dict]): The list of filter outcomes.

    Returns:
        int: The total count of skipped items.
    """
    skipped = 0
    for filter_outcome in filters_outcome:
        skipped_elements = filter_outcome.get("skippedEntityIds")
        skipped += len(skipped_elements)
    return skipped

def add_filtered_items_title(filters_outcome: list[dict], shelf_client: ContentIntelligenceClient) -> list[dict]:
    """
    Formats the filtered items based on the filters outcome.
    """
    
    for filter_outcome in filters_outcome:
        filtered_items = []
        skipped_elements_ids = filter_outcome.get("skippedEntityIds")
        content_items = shelf_client.content_items.list(
            ids_to_include=skipped_elements_ids,
        )
        for content_item in content_items:
            filtered_items.append(content_item.title)
        filter_outcome["skippedContentItemsTitle"] = filtered_items

    return filters_outcome


def extract_sections_metadata(retrieved_sections: dict) -> list[dict]:
    """
    Extracts metadata information from the retrieved sections.

    Args:
        retrieved_sections (dict): The dictionary containing retrieved sections data.

    Returns:
        list[dict]: A list of dictionaries containing sections metadata.
    """
    sections_metadata = []
    for data in retrieved_sections:
        information = {"content_id": data["externalId"], "title": data["title"], "url": data["externalURL"]}
        if information not in sections_metadata:
            sections_metadata.append(information)
    return sections_metadata


def format_source_documents(source_documents: dict) -> str:
    """
    Formats the source documents into a string with markdown links.

    Args:
        source_documents (dict): The dictionary containing source document information.

    Returns:
        str: The formatted string with markdown links to the source documents.
    """
    formatted_sources = "\nSources:\n"
    for document in source_documents:
        formatted_sources += f"- [{document['title']}]({document['url']})\n"
    return formatted_sources
