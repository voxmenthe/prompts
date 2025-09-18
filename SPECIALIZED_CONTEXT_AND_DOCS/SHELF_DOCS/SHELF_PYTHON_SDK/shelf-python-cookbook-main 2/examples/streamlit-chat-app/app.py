import os
import streamlit as st
from dotenv import load_dotenv
from jinja2 import Template
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from shelf.ci.client import ContentIntelligenceClient
from src.formating import format_source_informarion, add_filtered_items_title
from src.clients import get_llm_provider_options, initialize_azure_openai_client, initialize_openai_client, init_shelf_api_client
from src.session_state import (
    add_llm_response_to_chat_history,
    clear_chat_history,
    display_chat_messages_from_history,
    initialize_session_state,
)

from shelf.__init__ import __version__

st.set_page_config(layout="wide")

# Load environment variables
load_dotenv()


shelf_api_client = init_shelf_api_client()
shelf_client = ContentIntelligenceClient(client=shelf_api_client)


def main():
    """
    The main function for the Streamlit app that orchestrates the chat interface.
    It handles user input, interacts with Shelf Content Intelligence and OpenAI APIs,
    and displays the chat interface along with the responses.
    """
    # Title of the chat interface
    st.title("Content Intelligence Chat Sandbox")

    initialize_session_state()
    display_chat_messages_from_history()

    app_settings = get_app_settings()

    if app_settings["llm_provider"] == "OpenAI":
        llm_client = initialize_openai_client(app_settings["llm"])
    else:
        llm_client = initialize_azure_openai_client(app_settings["llm"])

    # Accept user input
    question = st.chat_input("Ask a question:")
    if question:
        # Display user question in chat message container
        with st.chat_message("user"):
            st.markdown(question)
        # Add user question to chat history
        st.session_state.messages.append({"role": "user", "content": question})

        with st.spinner("Start searching for an answer..."):
            sections, filters_outcome = search_sections_with_filters(question, app_settings)
            llm_prompt = prepare_llm_prompt(question, sections, app_settings)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            if app_settings["chat"]["show_intermediate_steps"]:
                display_intermediate_steps(question, sections, filters_outcome, llm_prompt)

            llm_response = stream_llm_completions(llm_client, llm_prompt)
            source_inforamtion = format_source_informarion(sections, filters_outcome)
            st.markdown(source_inforamtion)

        add_llm_response_to_chat_history(llm_response, source_inforamtion)


def get_app_settings() -> dict:
    """
    This function is responsible for rendering the sidebar settings,
    where users can select and modify different parameters.
    It returns a dictionary with the selected settings.
    """
    # Settings for the GPT model
    st.sidebar.header("LLM Settings")

    llm_provider = st.sidebar.selectbox("LLM Provider", get_llm_provider_options())
    if llm_provider == "OpenAI":
        gpt_model_name = st.sidebar.selectbox("GPT Model", ["gpt-3.5-turbo", "gpt-4-turbo-preview"])
    else:
        gpt_model_name = os.environ.get("AZURE_DEPLOYMENT")
    temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.0, 0.1)

    # Settings for the search
    st.sidebar.header("Retrieval Settings")
    number_of_sections = st.sidebar.slider("Number of sections to retrieve", 0, 10, 3, 1)

    # Sidebar for quality filters
    st.sidebar.subheader("Quality filters")

    exclude_wip_documents = st.sidebar.checkbox("Exclude documents marked work in progress", value=True)
    exclude_duplicated_documents = st.sidebar.checkbox(
        "Exclude duplicates except the last updated document", value=True
    )
    exclude_toxic_documents = st.sidebar.checkbox("Exclude documents with toxic content", value=True)
    exclude_crossed_out_sections = st.sidebar.checkbox("Exclude sections with crossed-out text", value=True)
    exclude_contradicting_documents = st.sidebar.checkbox("Exclude documents with contradicting content", value=True)

    exclude_link_health_sections = st.sidebar.checkbox("Exclude sections with unhealth links", value=True)

    st.sidebar.subheader("Content filters")

    search_filters_options = facets_for_search_filters()

    collections_keys = list(search_filters_options["collections"].keys())
    selected_collections_keys = st.sidebar.multiselect("Collections/Libraries", collections_keys, placeholder="All")
    selected_collections_ids = [
        search_filters_options["collections"][collection_name] for collection_name in selected_collections_keys
    ]

    tags_keys = list(search_filters_options["tags"].keys())
    selected_tags_keys = st.sidebar.multiselect("Tags", tags_keys, placeholder="All")
    selected_tags = [search_filters_options["tags"][tag_name] for tag_name in selected_tags_keys]

    # Sidebar for enrichment options
    st.sidebar.header("Enrichment Options")
    acronyms_disambiguation = st.sidebar.checkbox("Acronyms Disambiguation", value=True)

    # Sidebar for chat settings
    st.sidebar.subheader("Chat Settings")
    show_intermediate_steps = st.sidebar.checkbox("Show intermediate steps", value=False)

    clear_chat = st.sidebar.button("Clear chat history")
    if clear_chat:
        clear_chat_history()

    with st.sidebar.expander("Prompt template"):
        prompt_template = st.text_area(
            "Prompt template",
            default_prompt_template(),
            height=300,
        )

    settings = {
        "llm_provider": llm_provider,
        "llm": {
            "model": gpt_model_name,
            "temperature": temperature,
        },
        "search": {
            "number_of_sections": number_of_sections,
        },
        "filters": {
            "quality": {
                "exclude_wip_documents": exclude_wip_documents,
                "exclude_duplicated_documents": exclude_duplicated_documents,
                "exclude_crossed_out_sections": exclude_crossed_out_sections,
                "exclude_contradicting_documents": exclude_contradicting_documents,
                "exclude_toxic_documents": exclude_toxic_documents,
                "exclude_link_health_sections": exclude_link_health_sections,
            },
            "content": {
                "collections": selected_collections_ids,
                "tags": selected_tags,
            },
        },
        "enrichments": {
            "acronyms_disambiguation": acronyms_disambiguation,
        },
        "chat": {
            "show_intermediate_steps": show_intermediate_steps,
            "prompt_template": prompt_template,
        },
    }
    return settings


def default_prompt_template() -> str:
    """
    Returns the default prompt template for the language model.
    """
    return r"""
            Please answer the following question based on the information found
            within the sections enclosed by triplet quotes (\`\`\`).
            Your response should be concise, well-written, and follow markdown formatting guidelines:

            - Use bullet points for list items.
            - Use **bold** text for emphasis where necessary.

            **Question:** {{question}}

            Thank you for your detailed attention to the request
            **Context information**:
            ```
            {% for item in sections %}
                ---
                Document title: {{ item["title"] }}
                Document information: {{ item["text"] }}
                ---
            {% endfor %}
            ```

            **User Question:** {{question}}
            Answer:
            """


def facets_for_search_filters() -> dict:
    """
    Retrieves the facets for search filters from the session state.
    If not already stored, fetches the facets by making an API call.
    """
    if st.session_state.get("facets", None) is None:
        st.session_state["facets"] = fetch_facets_for_search_filters()

    search_facets_response = st.session_state["facets"]
    return search_facets_response


def fetch_facets_for_search_filters() -> dict:
    """
    Fetches the facets for search filters by making an API call.

    Returns:
        dict: A dictionary containing the facets for search filters.
    """
    payload = {"facets": ["collection", "tag", "lang", "connector"]}

    facets_response = shelf_client.client.post(path="/cil-search/facets", json=payload)

    return process_facets_response(facets_response)


def process_facets_response(facets_response: dict) -> dict:
    """
    Processes the facets response obtained from the API call.

    Args:
        facets_response (dict): The response containing facets data.

    Returns:
        dict: A dictionary containing processed facets data categorized by collections, tags, langs, and connectors.
    """
    collections = {
        f"{collection['label']} - {collection['count']} docs": collection["value"]
        for collection in facets_response["results"]["collection"]
    }

    tags = {f"{tag['label']} - {tag['count']} docs": tag["value"] for tag in facets_response["results"]["tag"]}

    langs = {f"{lang['label']} - {lang['count']} docs": lang["value"] for lang in facets_response["results"]["lang"]}

    connectors = {
        f"{connector['label']} - {connector['count']} docs": connector["value"]
        for connector in facets_response["results"]["connector"]
    }

    facets_filters = {
        "collections": collections,
        "tags": tags,
        "langs": langs,
        "connectors": connectors,
    }
    return facets_filters


def generate_payload_for_search_api(question: str, settings: dict) -> dict:
    """
    Generates the payload for the search API based on the question and settings provided.

    Args:
        question (str): The search query or question.
        settings (dict): The settings containing filters and enrichments configuration.

    Returns:
        dict: A dictionary representing the payload for the search API.
    """
    quality_filters = []
    quality_filters_settings = settings["filters"]["quality"]

    if quality_filters_settings["exclude_wip_documents"] is True:
        quality_filters.append({"type": "wipDocuments", "level": "document"})
    if quality_filters_settings["exclude_crossed_out_sections"] is True:
        quality_filters.append({"type": "crossedOutContent", "level": "section"})
    if quality_filters_settings["exclude_duplicated_documents"] is True:
        quality_filters.append({"type": "duplicates", "level": "document", "value": {"exclude": "last"}})
    if quality_filters_settings["exclude_contradicting_documents"] is True:
        quality_filters.append({"type": "contradictions", "level": "document"})
    if quality_filters_settings["exclude_toxic_documents"] is True:
        quality_filters.append({"type": "toxicity", "level": "document"})
    if quality_filters_settings["exclude_link_health_sections"] is True:
        quality_filters.append({"type": "linkHealth", "level": "section"})

    content_filters = []
    collection_ids = settings["filters"]["content"]["collections"]
    if len(collection_ids) > 0:
        content_filters.append({"type": "collections", "value": {"eq": collection_ids}})

    tags = settings["filters"]["content"]["tags"]
    if len(tags) > 0:
        content_filters.append({"type": "tags", "value": {"eq": tags}})

    enrichments = []
    if settings["enrichments"]["acronyms_disambiguation"] is True:
        enrichments.append("acronyms.disambiguation")

    return {
        "userInput": question,
        "filters": {
            "quality": quality_filters,
            "content": content_filters,
        },
        "enrichments": enrichments,
    }


def fetch_search_response(payload: dict) -> dict:
    """
    Fetches the search response by making a POST request to the search API.

    Args:
        payload (dict): The payload containing search parameters.

    Returns:
        dict: The response from the search API.
    """

    search_response = shelf_client.client.post(path="/cip-genai/content/search", json=payload)
    return search_response


def search_sections_with_filters(search_query: str, settings: dict) -> tuple[list[dict], list[dict]]:
    """
    Searches for sections with applied filters based on the search query and settings provided.

    Args:
        search_query (str): The search query or question.
        settings (dict): The settings containing filters and search configuration.

    Returns:
        tuple[list[dict], list[dict]]: A tuple containing a list of sections and filters outcome.
    """
    payload = generate_payload_for_search_api(search_query, settings)
    search_response = fetch_search_response(payload)

    filters_outcome = search_response.get("filtersOutcome", [])
    top_k = int(settings["search"]["number_of_sections"])
    sections = search_response.get("items", [])[:top_k]

    return sections, filters_outcome


def display_intermediate_steps(query: str, sections: list[dict], filters_outcome: list[dict], prompt: str) -> None:
    """
    Display intermediate steps in chat message container.

    Args:
        search_response (dict): Search response.
        prompt (str): Prompt for generation.
    """
    st.info("**Search Query:** " + query)
    with st.expander("Search Items"):
        for item in sections:
            st.json(item, expanded=False)
    with st.expander("Filters Outcome"):
        filters_outcome = add_filtered_items_title(filters_outcome, shelf_client)
        for filter_outcome in filters_outcome:
            st.write(filter_outcome, expanded=False)

    with st.expander("Prompt for generation"):
        st.write(prompt, unsafe_allow_html=False)


def prepare_llm_prompt(search_query: str, retrieved_sections: dict, settings: dict) -> str:
    """
    Prepares the prompt for the Language Model (LLM) based on the search query, retrieved sections, and settings.

    Args:
        search_query (str): The search query or question.
        retrieved_sections (dict): The retrieved sections from the search.
        settings (dict): The settings containing prompt template.

    Returns:
        str: The prepared prompt for the Language Model.
    """
    prompt_template = settings["chat"]["prompt_template"]

    template = Template(prompt_template)
    llm_promt = template.render(question=search_query, sections=retrieved_sections)
    return llm_promt


def stream_llm_completions(llm_client: ChatOpenAI, prompt: str) -> str:
    """
    Streams completions from a language model in real-time and displays them using Streamlit.

    Args:
        llm_client (ChatOpenAI): The language model client.
        prompt (str): The prompt for generation.
    """
    stream = llm_client.stream([HumanMessage(role="user", content=prompt)])
    response = st.write_stream(stream)
    return response


if __name__ == "__main__":
    main()
