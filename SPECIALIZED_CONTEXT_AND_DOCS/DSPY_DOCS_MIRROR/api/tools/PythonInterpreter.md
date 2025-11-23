# dspy.PythonInterpreter

## dspy.PythonInterpreter

```python
class PythonInterpreter(deno_command=None, enable_read_paths=None, enable_write_paths=None, enable_env_vars=None, enable_network_access=None, sync_files=True)
```

PythonInterpreter that runs code in a sandboxed environment using Deno and Pyodide.

Prerequisites:
- Deno (https://docs.deno.com/runtime/getting_started/installation/).

Example Usage:
```python
code_string = "print('Hello'); 1 + 2"
with PythonInterpreter() as interp:
    output = interp(code_string) # If final statement is non-None, prints the numeric result, else prints captured output
```


### __call__

```python
def __call__(self, code, variables=None)
```

### execute

```python
def execute(self, code, variables=None)
```

### shutdown

```python
def shutdown(self)
```
Source: `/Volumes/cdrive/repos/OTHER_PEOPLES_REPOS/dspy/dspy/primitives/python_interpreter.py` (lines 15–294)

