# prompts

## Overview`

This repository contains a collection of prompts for various applications, including:

- **Agents**: Multi-agent simulations and role-playing scenarios
- **Personas**: Detailed character profiles for AI models
- **Scenarios**: Real-world and hypothetical situations for scenario testing
- **Instructions**: Instruction-style prompts for a wide range of tasks


## Setup Instructions

1. **Install pyenv:**

   First, install pyenv to manage Python versions:

   ```bash
   # On macOS
   brew install pyenv

   # On Linux
   curl https://pyenv.run | bash
   ```

   Add pyenv to your shell configuration file (e.g., `.bashrc`, `.zshrc`):

   ```bash
   echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
   echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
   echo 'eval "$(pyenv init -)"' >> ~/.bashrc
   ```

   Restart your shell or run `source ~/.bashrc` (or your appropriate shell config file).

2. **Install Python using pyenv:**

   ```bash
   pyenv install 3.x.x  # Replace x.x with the desired Python version
   pyenv global 3.x.x
   ```

3. **Clone the repository:**

   ```bash
   git clone https://github.com/voxmenthe/prompts.git
   cd prompts
   ```

4. **Create a virtual environment:**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

5. **Install the dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

6. **Install the package:**

   ```bash
   pip install -e .
   ```

7. **Running Jupyter Notebooks:**

   Ensure that your Jupyter Notebook is using the virtual environment's Python interpreter. You can set it up by running:

   ```bash
   python -m ipykernel install --user --name=prompts
   ```

   Then, select `prompts` as the kernel in Jupyter Notebook.
