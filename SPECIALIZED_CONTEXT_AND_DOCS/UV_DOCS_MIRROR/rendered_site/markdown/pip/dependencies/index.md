# [Declaring dependencies](#declaring-dependencies)

It is best practice to declare dependencies in a static file instead of modifying environments with ad-hoc installations. Once dependencies are defined, they can be [locked](../compile/)to create a consistent, reproducible environment. 

## [Using `pyproject.toml `](#using-pyprojecttoml)

The `pyproject.toml `file is the Python standard for defining configuration for a project. 

To define project dependencies in a `pyproject.toml `file: 

pyproject.toml 

```
[#__codelineno-0-1](#__codelineno-0-1)[project]
[#__codelineno-0-2](#__codelineno-0-2)dependencies = [
[#__codelineno-0-3](#__codelineno-0-3)  "httpx",
[#__codelineno-0-4](#__codelineno-0-4)  "ruff>=0.3.0"
[#__codelineno-0-5](#__codelineno-0-5)]

```

To define optional dependencies in a `pyproject.toml `file: 

pyproject.toml 

```
[#__codelineno-1-1](#__codelineno-1-1)[project.optional-dependencies]
[#__codelineno-1-2](#__codelineno-1-2)cli = [
[#__codelineno-1-3](#__codelineno-1-3)  "rich",
[#__codelineno-1-4](#__codelineno-1-4)  "click",
[#__codelineno-1-5](#__codelineno-1-5)]

```

Each of the keys defines an "extra", which can be installed using the `--extra `and `--all-extras `flags or `package[ ] `syntax. See the documentation on [installing packages](../packages/#installing-packages-from-files)for more details. 

See the official [`pyproject.toml `guide](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/)for more details on getting started with a `pyproject.toml `. 

## [Using `requirements.in `](#using-requirementsin)

It is also common to use a lightweight `requirements.txt `format to declare the dependencies for the project. Each requirement is defined on its own line. Commonly, this file is called `requirements.in `to distinguish it from `requirements.txt `which is used for the locked dependencies. 

To define dependencies in a `requirements.in `file: 

requirements.in 

```
[#__codelineno-2-1](#__codelineno-2-1)httpx
[#__codelineno-2-2](#__codelineno-2-2)ruff>=0.3.0

```

Optional dependencies groups are not supported in this format. 

August 27, 2024
