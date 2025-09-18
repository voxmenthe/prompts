import os
import httpx

from langchain_openai import AzureChatOpenAI, ChatOpenAI
from shelf.core.base_client import BaseAPIClient


def init_shelf_api_client():
    """
    Initialize the Shelf API client with custom timeout value
    """
    token = os.environ.get("SHELF_API_TOKEN")
    api_url = os.environ.get("SHELF_API_URL")

    if token is None or token == "your_shelf_api_token_here":
        raise ValueError("Please provide a valid API token for the Shelf API.")
    
    if api_url is None or api_url == "https://api_url_here":
        raise ValueError("Please provide a valid API URL for the Shelf API.")

    headers = {
        "Authorization": token,
        "Content-Type": "application/json",
        "User-Agent": "Shelf-SDK/Python/0.1.0 (Darwin; Python 3.11.7)",
    }

    timeout = httpx.Timeout(120)

    httpx_client = httpx.Client(
        headers=headers, base_url=api_url, timeout=timeout
    )

    api_client = BaseAPIClient(base_url=api_url, token=token, httpx_client=httpx_client)
    return api_client



def initialize_openai_client(llm_settings: dict) -> ChatOpenAI:
    """
    Initializes the OpenAI client for chat based on the provided settings.

    Args:
        llm_settings (dict): The settings for the Language Model.

    Returns:
        ChatOpenAI: An instance of the ChatOpenAI class for OpenAI API.

    Raises:
        ValueError: If the OpenAI API key is not provided in the environment variables.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key and api_key != "your_openai_api_key_here":
        return ChatOpenAI(**llm_settings)
    raise ValueError("Please provide a valid API key for the OpenAI API or choose a suitable LLM provider.")


def initialize_azure_openai_client(
    llm_settings: dict,
) -> AzureChatOpenAI:
    """
     Initializes the Azure OpenAI client for chat based on the provided settings.

    Args:
        llm_settings (dict): The settings for the Language Model.

    Returns:
        AzureChatOpenAI: An instance of the AzureChatOpenAI class for Azure OpenAI API.

    Raises:
        ValueError: If the Azure OpenAI API key is not provided in the environment variables.
    """
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    if api_key and api_key != "your_azure_openai_api_key_here":
        return AzureChatOpenAI(
            azure_deployment=os.environ.get("AZURE_DEPLOYMENT"),
            openai_api_version=os.environ.get("OPENAI_API_VERSION"),
            azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
            openai_api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
            **llm_settings,
        )
    raise ValueError("Please provide a proper API key for Azure OpenAI API or choose a suitable LLM provider.")


def get_llm_provider_options() -> list[str]:
    """
    Returns the list of available LLM providers based on environment variables.
    """
    options = []
    if os.environ.get("AZURE_OPENAI_API_KEY"):
        options.append("Azure OpenAI")
    if os.environ.get("OPENAI_API_KEY"):
        options.append("OpenAI")
    if len(options) == 0:
        raise ValueError(
            "Please provide a valid API key for the OpenAI API or Azure OpenAI API in the environment variables."
        )
    return options
