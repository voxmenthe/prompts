

### **Instructions for Writing Clean, Modular, Readable, and Maintainable Code**  
*(Python, NumPy, PyTorch, JAX)*  

---

### **1. Clean Code**  
Clean code is simple, efficient, and free of unnecessary complexity.  

#### **Key Practices**  
1. **Follow Style Guidelines**  
   - Use **PEP8** for Python (e.g., 4-space indentation, snake_case for variables).  
   - Auto-format with tools like `black` or `autopep8` to enforce consistency.  
   - Example:  
     ```python  
     # Bad  
     def calculate(x):return x*2  
     # Good  
     def calculate(x):  
         return x * 2  
     ```

2. **Avoid Magic Numbers**  
   - Replace hardcoded values with named constants.  
   - Example (NumPy):  
     ```python  
     # Bad  
     normalized_data = data / 255.0  
     # Good  
     MAX_PIXEL_VALUE = 255.0  
     normalized_data = data / MAX_PIXEL_VALUE  
     ```

3. **Vectorize Operations**  
   - Use NumPy/PyTorch/JAX vectorization instead of Python loops for performance and clarity.  
   - Example (NumPy):  
     ```python  
     # Bad  
     squares = [x**2 for x in np.arange(1000)]  
     # Good  
     squares = np.arange(1000)**2  
     ```

4. **Avoid Side Effects**  
   - Functions should not modify global state. Critical for JAX’s JIT compilation.  
   - Example (JAX):  
     ```python  
     # Bad  
     global_counter = 0  
     def increment():  
         global global_counter  
         global_counter += 1  
     # Good  
     def increment(counter):  
         return counter + 1  
     ```

---

### **2. Modular Code**  
Modular code is organized into reusable, self-contained components.  

#### **Key Practices**  
1. **Single Responsibility Principle**  
   - Each function/class should do one thing.  
   - Example (PyTorch):  
     ```python  
     # Bad: A single function handling data loading, training, and logging.  
     # Good:  
     def train_epoch(model, dataloader):  
         ...  
     def log_metrics(metrics):  
         ...  
     ```

2. **Use Classes for State Management**  
   - Encapsulate related data and methods in classes.  
   - Example (PyTorch):  
     ```python  
     class NeuralNetwork(nn.Module):  
         def __init__(self):  
             super().__init__()  
             self.layer = nn.Linear(10, 2)  
         def forward(self, x):  
             return self.layer(x)  
     ```

3. **Leverage Modules and Packages**  
   - Split code into modules (e.g., `data.py`, `models.py`, `utils.py`).  
   - Example directory structure:  
     ```  
     project/  
         data_loader.py  
         models/  
             __init__.py  
             cnn.py  
             transformer.py  
         train.py  
     ```

4. **Functional Programming in JAX**  
   - Use pure functions and JAX transformations (e.g., `jit`, `vmap`).  
   - Example:  
     ```python  
     @jax.jit  
     def loss_fn(params, x, y):  
         preds = model.apply(params, x)  
         return jnp.mean((preds - y) ** 2)  
     ```

---

### **3. Readable Code**  
Readable code is easy to understand at a glance.  

#### **Key Practices**  
1. **Meaningful Names**  
   - Use descriptive variable/function names.  
   - Example:  
     ```python  
     # Bad  
     def process(a, b):  
         ...  
     # Good  
     def compute_similarity(embedding1, embedding2):  
         ...  
     ```

2. **Docstrings and Type Hints**  
   - Document functions with docstrings and type hints.  
   - Example (PyTorch):  
     ```python  
     def train(model: nn.Module, dataloader: DataLoader, lr: float = 1e-3) -> float:  
         """Trains the model for one epoch and returns average loss."""  
         ...  
     ```

3. **Avoid Overly Clever Code**  
   - Prioritize clarity over brevity.  
   - Example (JAX):  
     ```python  
     # Bad (unclear axis manipulation)  
     reshaped = jnp.transpose(x, (2, 0, 1))  
     # Good  
     # Reshape from (batch, height, width, channels) to (batch, channels, height, width)  
     reshaped = x.transpose(0, 3, 1, 2)  
     ```

4. **Consistent Formatting**  
   - Use vertical alignment and whitespace.  
   - Example:  
     ```python  
     # Bad  
     result=compute(a,b,c)  
     # Good  
     result = compute(a, b, c)  
     ```

---

### **4. Maintainable Code**  
Maintainable code is easy to debug, extend, and refactor.  

#### **Key Practices**  
1. **Write Tests**  
   - Use `pytest` for unit/integration tests.  
   - Example (NumPy):  
     ```python  
     def test_normalize():  
         data = np.array([0, 255])  
         expected = np.array([0.0, 1.0])  
         assert np.allclose(normalize(data), expected)  
     ```

2. **Version Control**  
   - Commit small, logical changes with clear messages.  
   - Example:  
     ```  
     git commit -m "Fix gradient clipping in JAX optimizer"  
     ```

3. **Reproducibility**  
   - Set random seeds for NumPy/PyTorch/JAX.  
     ```python  
     import numpy as np  
     import torch  
     import jax  
     np.random.seed(42)  
     torch.manual_seed(42)  
     jax.config.update("jax_enable_x64", True)  
     ```

4. **Configuration Management**  
   - Use config files (YAML/JSON) for hyperparameters.  
   - Example:  
     ```python  
     # config.yaml  
     learning_rate: 0.001  
     batch_size: 32  
     # In code  
     with open("config.yaml", "r") as f:  
         config = yaml.safe_load(f)  
     ```

5. **Logging Over Print Statements**  
   - Use Python’s `logging` module.  
     ```python  
     import logging  
     logging.info(f"Epoch {epoch}: Loss = {loss:.4f}")  
     ```

---

### **Tools to Adopt**  
- **Linting**: `flake8`, `pylint`  
- **Formatting**: `black`, `isort`  
- **Testing**: `pytest`, `unittest`  
- **Dependencies**: `poetry` or `pipenv`  
- **Type Checking**: `mypy` (for Python)  

By following these guidelines, you’ll write code that is easier to debug, share, and scale. Always prioritize clarity and simplicity over cleverness.