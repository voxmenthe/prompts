# prompts

## Setup Instructions

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/yourproject.git
   cd yourproject
   ```

2. **Create a virtual environment:**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install the dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Install the package:**

   ```bash
   pip install -e .
   ```

5. **Running Jupyter Notebooks:**

   Ensure that your Jupyter Notebook is using the virtual environment's Python interpreter. You can set it up by running:

   ```bash
   python -m ipykernel install --user --name=your_project_name
   ```

   Then, select `your_project_name` as the kernel in Jupyter Notebook.
