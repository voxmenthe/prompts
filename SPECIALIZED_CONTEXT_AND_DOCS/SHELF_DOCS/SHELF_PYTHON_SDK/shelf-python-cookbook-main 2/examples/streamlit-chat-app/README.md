# Quick Start Guide: Chat RAG Application

Hello! Ready to set up an exciting chat app with Shelf Content Intelligence retrieval-augmented generation features? You're in the right spot! Whether you're comfortable with Python or prefer Docker, this guide will help you get started smoothly and quickly.

Before jumping in, ensure you have Python 3.9 or newer if you're going the Python route.

## Setting Up Your Environment

The initial step involves preparing your API credentials for the application to function correctly. Follow these instructions to get started:

1. Copy the example environment file:
    ```
    cp .env.example examples/streamlit-chat-app/.env
    ```

2. Navigate to the `cd examples/streamlit-chat-app` directory.
3. Add your API keys by updating the `.env` file as illustrated below:

   ```
    # Shelf Content Intelligence
    SHELF_API_TOKEN=your_shelf_api_token_here
    SHELF_API_URL=https://api_url_here

    # Select one section based on the service you are using: OpenAI or Azure OpenAI
    # Fill in the corresponding credentials for the service you choose

    # OpenAI Credentials
    OPENAI_API_KEY=your_openai_api_key_here

    # Azure OpenAI Configuration
    AZURE_OPENAI_API_KEY=your_azure_openai_api_key_here
    AZURE_OPENAI_ENDPOINT=https://your_azure_openai_endpoint_here
    OPENAI_API_VERSION=your_openai_api_version_here
    AZURE_DEPLOYMENT=your_azure_deployment_name_here
   ```

Once you've updated your API credentials based on your chosen service (OpenAI or Azure OpenAI), your setup is complete!

## Choosing Your Setup Method

Here are two pathways you can choose from, depending on what you prefer.

### 🐳 Option 1: Using Docker

If you like the simplicity of Docker, follow these steps:

1. Ensure Docker is installed and operational on your device.
2. Navigate to `examples/streamlit-chat-app` using your terminal.
3. Execute `make build` to let Docker prepare your app.

That’s it! Your chat app is now containerized and ready within Docker.

### 🐍 Option 2: Python Setup

Prefer setting things up manually? Let's dive into the Python setup:

#### Preparing Your Environment

1. Open your terminal and navigate to `examples/streamlit-chat-app`.
2. Create a Python virtual environment with `python -m venv venv`.
3. Activate the virtual environment:
   - **Unix/MacOS**: `source venv/bin/activate`
   - **Windows**: `.\venv\Scripts\activate`
4. Install the required dependencies using `pip install -r requirements.txt`.

#### Starting the Application

With your environment ready, launch your app using:

```bash
streamlit run app.py
```

## Enjoying Your Application

Open your browser and visit `http://localhost:8501`. Your chat application is now up and running!

Explore the capabilities of Shelf Content Intelligence in your chat app. Experiment, ask questions, and enjoy what you've built. 

Happy coding! 🚀
