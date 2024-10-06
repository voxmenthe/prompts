#!/bin/bash

# Install dependencies with poetry
pip install -U poetry
poetry install
poetry lock

# Create and install the IPython kernel for the project
#python -m ipykernel install --user --name=prompts --display-name "Prompts"
poetry run python -m ipykernel install --sys-prefix --name=prompts --display-name "Prompts"

echo "Jupyter kernel 'Prompts' has been installed."