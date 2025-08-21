# Shelf Content Intelligence as a Retriever

## Using Shelf Content Intelligence Search as a Retriever in a LangChain RAG Pipeline

This tutorial guides you through the process of integrating Shelf Content Intelligence as a retriever within a LangChain Retrieve and Generate (RAG) pipeline. The goal is to leverage the powerful search capabilities of Shelf to retrieve relevant document sections and use them as context for generating answers to questions.

### Prerequisites

Before starting, ensure you have installed the necessary packages by running the following commands:

```bash
pip install python-dotenv==1.0.1
pip install langchain==0.1.9
pip install langchain-community==0.0.24
pip install langchain-core==0.1.28
pip install langchain_openai==0.0.8
```

Ensure you're using these specific versions or later to avoid compatibility issues.

### Setup

Before diving into the code, ensure you have `.env` file with your environment variables and API keys for Shelf Content Intelligence access. We use `dotenv` to simplify environment variable management:

```python
from dotenv import load_dotenv
load_dotenv()
```

### Building a Custom Retriever

First, we'll implement a custom retriever by inheriting from `BaseRetriever` and overriding the `_get_relevant_documents` method. This retriever will use the Shelf platform to fetch document sections relevant to a given query.

#### Defining the `ShelfSectionsRetriever` Class

The class has several important attributes and methods:

* `client`: An instance of `ShelfContentIntelligenceClient` used to interact with the Shelf platform.
* `number`: Specifies the number of document sections to retrieve.
* `origin`: An optional string indicating the origin of the request.
* `search_filter`: An optional dictionary to specify filter parameters for the search.
* `_get_relevant_documents()`: This overridden method performs the actual querying and retrieval of documents from Shelf using the client, converting the results into a list of `Document` objects compatible with LangChain..

The `validate_client` root validator ensures `client` is correctly instanced, preventing errors during runtime.

```python
from typing import Any

from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import root_validator

from shelf.ci.client import ContentIntelligenceClient

class ShelfSectionsRetriever(BaseRetriever):
    """
    `Shelf hybrid search` retriever.

    See the documentation:
      https://docs.shelf.io/dev-portal
    """

    client: Any
    """'shelf_ci_client' instance to use."""
    number: int = 5
    """Number of sections to retrieve."""
    origin: str = "unknown"
    """Origin of the request."""
    search_filter: dict = {}
    """Filter to apply to search the sections."""

    @root_validator(pre=True)
    def validate_client(
        cls,
        values: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            import shelf

        except ImportError:
            raise ImportError("Could not import shelf python package. " "Please install it with `pip install shelf`.")
        if not isinstance(values["client"], ContentIntelligenceClient):
            client = values["client"]
            raise ValueError(f"client should be an instance of ContentIntelligenceClient, got {type(client)}")
        return values

    def _search_api_call(self, search_query):
        response = self.client.client.post(
            path="/cil-search/content/sections",
            json={
                "query": search_query,
                "size": self.number,
                "origin": "shelf",
                "format": "markdown",
                **self.search_filter,
            }
        )
        return response.get("items", [])

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        sections = self._search_api_call(query)

        langchain_documents = [
            Document(
                page_content=section["content"],
                metadata={
                    "section_id": section["sectionId"],
                    "semantic_section_id": section["semanticSectionId"],
                    "content_title": section["title"],
                    "content_item_id":  section["externalId"],
                    "url": section["externalURL"],
                }
            )
            for section in sections
        ]

        return langchain_documents
```

This retriever fetches a specified number of technical sections from Shelf that match the given query and then converts them into a format (`Document`) that LangChain can work with.

#### Utilizing the Retriever

After defining the class, we instantiate `ShelfSectionsRetriever` with our specific parameters, including the client, number of sections to retrieve, origin, and any search filters. In this example, we're filtering by a `parent_id`.

```python
shelf_retriever = ShelfSectionsRetriever(
    client=ContentIntelligenceClient(),  # Initialize with your Shelf client
    number=5,  # Number of sections to retrieve
    origin="shelf",
    search_filter={ "parent_id": "8a22a6bb-d31c-4c66-8ba2-f442bd5851db" },  # Custom search filter
)
```

The `get_relevant_documents` method is called with a query string to fetch relevant document sections.

```python
 docs = shelf_retriever.get_relevant_documents("How to Build a Powerful RAG Pipeline with Shelf Content Intelligence?")
```

### Integrating the Retriever into a LangChain RAG Pipeline

LangChain pipelines allow for flexible composition of processing steps. Here, we're constructing a pipeline that includes our custom retriever, a formatting function to process documents, and a prompt template to guide the generation.

#### Pipeline Components

* `template`: Defines the structure of the input to our model, including context and question.
* `prompt`: A `ChatPromptTemplate` that formats input using our predefined template.
* `model`: In this example, an instance of `ChatOpenAI` is used as our generation model.
* `format_docs`: A custom function to format the retrieved documents into a single string of context.

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Template to structure the context and question for the OpenAI model
template = """Answer the question based only on the following context: {context} Question: {question}"""
prompt = ChatPromptTemplate.from_template(template)

# Initialize the model
model = ChatOpenAI()

# Function to format the documents into a single string of contexts
def format_docs(docs: list[Document]):
    return "\n\n".join([d.page_content for d in docs])

# Construct the pipeline
chain = (
    {"context": shelf_retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

# Invoke the pipeline with a sample question
answer = chain.invoke("How to update the order?")
print(answer)
```

This RAG pipeline takes a question, retrieves relevant sections from the Shelf platform, formats those sections into a context string, uses that context along with the question to generate a prompt for the OpenAI model, and finally parses the model's response into a human-readable string.

And that's it! You've successfully integrated Shelf Content Intelligence as a retriever in a LangChain RAG pipeline. This setup enables you to create powerful question-answering systems that utilize the rich content available on the Shelf platform.
