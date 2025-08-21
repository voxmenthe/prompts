# Shelf Content Intelligence as a Data Loader

This tutorial guides you through setting up a basic example of Langchain RAG (Retrieval-Augmented Generation) using the Shelf Content Intelligence loader. We will walk through installing necessary libraries, loading content from Shelf Content Intelligence, and setting up a simple RAG pipeline to answer questions based on the loaded content.

### Prerequisites

Ensure you have Python installed on your machine. This example uses Python 3.9 or above.

### Setup

First, you need to install the required dependencies. Run the following commands in your terminal or Jupyter notebook:

```bash
%pip install python-dotenv==1.0.1
%pip install langchain==0.1.9
%pip install langchain-community==0.0.24
%pip install langchain-core==0.1.28
%pip install langchain_openai==0.0.8
%pip install omegaconf==2.3.0
%pip install chromadb==0.4.24
```

### Loading Content with Shelf Content Intelligence

Before proceeding, make sure to load your environment variables using `load_dotenv()`. This typically involves having a `.env` file with necessary API keys and configurations.

Here's the core class for loading sections from Shelf Content Intelligence:

```python
from typing import Optional
from dotenv import load_dotenv
from langchain.schema.document import Document
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from omegaconf import OmegaConf
from shelf.ci.client import ContentIntelligenceClient

load_dotenv()

class ShelfSectionsLoader:
    """
    Load sections from Shelf Content Intelligence.
    """
    def __init__(self):
        self.ci = ContentIntelligenceClient()

    def load(self, **kwargs):
        langchain_documents = []
        content_items_iterator = self.ci.content_items.list(**kwargs)

        for content_item in content_items_iterator:
            content_item_id = content_item.external_id

            semantic_sections_iterator = self.ci.sections.list(item_id=content_item_id)

            for section in semantic_sections_iterator:
                semantic_section_id = section.id
                section_content = self.ci.sections.retrieve(content_item_id, semantic_section_id)
                metadata = {
                    "content_item_id": content_item_id,
                    "content_item_title": content_item.title,
                    "section_id": semantic_section_id,
                    "section_title": section.title,
                }

                langain_document = Document(page_content=section_content, metadata=metadata)
                langchain_documents.append(langain_document)

        return langchain_documents
```

### Configuring Your Loader and Documents

Define your configuration and load the documents using the `ShelfSectionsLoader` class:

```python
conf = OmegaConf.create({
    "parent_id": "8a22a6bb-d31c-4c66-8ba2-f442bd5851db",
    "items_limit": 5,
})

loader = ShelfSectionsLoader()
documents = loader.load(**conf)
```

### Setting Up the RAG Pipeline

Now, set up the vector store, retriever, prompt template, and the final RAG model:

```python
vectorstore = Chroma.from_documents(documents, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

model = ChatOpenAI()

chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | model | StrOutputParser()
```

### Running Your RAG Pipeline

Finally, you can run your RAG pipeline by invoking the chain with a question:

```python
response = chain.invoke("How to say if adding an item")
print(response)
```

This command queries the loaded Shelf Content Intelligence sections to find relevant information and generate an answer to the question.

Congratulations! You've just completed a basic setup of a Langchain RAG example with Shelf Content Intelligence loader.

***
