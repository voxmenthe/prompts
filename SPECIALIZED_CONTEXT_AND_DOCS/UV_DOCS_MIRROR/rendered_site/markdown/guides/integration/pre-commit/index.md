# [Using uv in pre-commit](#using-uv-in-pre-commit)

An official pre-commit hook is provided at [`astral-sh/uv-pre-commit `](https://github.com/astral-sh/uv-pre-commit). 

To use uv with pre-commit, add one of the following examples to the `repos `list in the `.pre-commit-config.yaml `. 

To make sure your `uv.lock `file is up to date even if your `pyproject.toml `file was changed: 

.pre-commit-config.yaml 

```
[#__codelineno-0-1](#__codelineno-0-1)repos:
[#__codelineno-0-2](#__codelineno-0-2)  - repo: https://github.com/astral-sh/uv-pre-commit
[#__codelineno-0-3](#__codelineno-0-3)    # uv version.
[#__codelineno-0-4](#__codelineno-0-4)    rev: 0.9.9
[#__codelineno-0-5](#__codelineno-0-5)    hooks:
[#__codelineno-0-6](#__codelineno-0-6)      - id: uv-lock

```

To keep a `requirements.txt `file in sync with your `uv.lock `file: 

.pre-commit-config.yaml 

```
[#__codelineno-1-1](#__codelineno-1-1)repos:
[#__codelineno-1-2](#__codelineno-1-2)  - repo: https://github.com/astral-sh/uv-pre-commit
[#__codelineno-1-3](#__codelineno-1-3)    # uv version.
[#__codelineno-1-4](#__codelineno-1-4)    rev: 0.9.9
[#__codelineno-1-5](#__codelineno-1-5)    hooks:
[#__codelineno-1-6](#__codelineno-1-6)      - id: uv-export

```

To compile requirements files: 

.pre-commit-config.yaml 

```
[#__codelineno-2-1](#__codelineno-2-1)repos:
[#__codelineno-2-2](#__codelineno-2-2)  - repo: https://github.com/astral-sh/uv-pre-commit
[#__codelineno-2-3](#__codelineno-2-3)    # uv version.
[#__codelineno-2-4](#__codelineno-2-4)    rev: 0.9.9
[#__codelineno-2-5](#__codelineno-2-5)    hooks:
[#__codelineno-2-6](#__codelineno-2-6)      # Compile requirements
[#__codelineno-2-7](#__codelineno-2-7)      - id: pip-compile
[#__codelineno-2-8](#__codelineno-2-8)        args: [requirements.in, -o, requirements.txt]

```

To compile alternative requirements files, modify `args `and `files `: 

.pre-commit-config.yaml 

```
[#__codelineno-3-1](#__codelineno-3-1)repos:
[#__codelineno-3-2](#__codelineno-3-2)  - repo: https://github.com/astral-sh/uv-pre-commit
[#__codelineno-3-3](#__codelineno-3-3)    # uv version.
[#__codelineno-3-4](#__codelineno-3-4)    rev: 0.9.9
[#__codelineno-3-5](#__codelineno-3-5)    hooks:
[#__codelineno-3-6](#__codelineno-3-6)      # Compile requirements
[#__codelineno-3-7](#__codelineno-3-7)      - id: pip-compile
[#__codelineno-3-8](#__codelineno-3-8)        args: [requirements-dev.in, -o, requirements-dev.txt]
[#__codelineno-3-9](#__codelineno-3-9)        files: ^requirements-dev\.(in|txt)$

```

To run the hook over multiple files at the same time, add additional entries: 

.pre-commit-config.yaml 

```
[#__codelineno-4-1](#__codelineno-4-1)repos:
[#__codelineno-4-2](#__codelineno-4-2)  - repo: https://github.com/astral-sh/uv-pre-commit
[#__codelineno-4-3](#__codelineno-4-3)    # uv version.
[#__codelineno-4-4](#__codelineno-4-4)    rev: 0.9.9
[#__codelineno-4-5](#__codelineno-4-5)    hooks:
[#__codelineno-4-6](#__codelineno-4-6)      # Compile requirements
[#__codelineno-4-7](#__codelineno-4-7)      - id: pip-compile
[#__codelineno-4-8](#__codelineno-4-8)        name: pip-compile requirements.in
[#__codelineno-4-9](#__codelineno-4-9)        args: [requirements.in, -o, requirements.txt]
[#__codelineno-4-10](#__codelineno-4-10)      - id: pip-compile
[#__codelineno-4-11](#__codelineno-4-11)        name: pip-compile requirements-dev.in
[#__codelineno-4-12](#__codelineno-4-12)        args: [requirements-dev.in, -o, requirements-dev.txt]
[#__codelineno-4-13](#__codelineno-4-13)        files: ^requirements-dev\.(in|txt)$

```

November 12, 2025
