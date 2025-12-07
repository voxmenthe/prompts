# [Working on projects](#working-on-projects)

uv supports managing Python projects, which define their dependencies in a `pyproject.toml `file. 

## [Creating a new project](#creating-a-new-project)

You can create a new Python project using the `uv init `command: 

```
[#__codelineno-0-1](#__codelineno-0-1)$ uv init hello-world
[#__codelineno-0-2](#__codelineno-0-2)$ cd hello-world

```

Alternatively, you can initialize a project in the working directory: 

```
[#__codelineno-1-1](#__codelineno-1-1)$ mkdir hello-world
[#__codelineno-1-2](#__codelineno-1-2)$ cd hello-world
[#__codelineno-1-3](#__codelineno-1-3)$ uv init

```

uv will create the following files: 

```
[#__codelineno-2-1](#__codelineno-2-1)├── .gitignore
[#__codelineno-2-2](#__codelineno-2-2)├── .python-version
[#__codelineno-2-3](#__codelineno-2-3)├── README.md
[#__codelineno-2-4](#__codelineno-2-4)├── main.py
[#__codelineno-2-5](#__codelineno-2-5)└── pyproject.toml

```

The `main.py `file contains a simple "Hello world" program. Try it out with `uv run `: 

```
[#__codelineno-3-1](#__codelineno-3-1)$ uv run main.py
[#__codelineno-3-2](#__codelineno-3-2)Hello from hello-world!

```

## [Project structure](#project-structure)

A project consists of a few important parts that work together and allow uv to manage your project. In addition to the files created by `uv init `, uv will create a virtual environment and `uv.lock `file in the root of your project the first time you run a project command, i.e., `uv run `, `uv sync `, or `uv lock `. 

A complete listing would look like: 

```
[#__codelineno-4-1](#__codelineno-4-1).
[#__codelineno-4-2](#__codelineno-4-2)├── .venv
[#__codelineno-4-3](#__codelineno-4-3)│   ├── bin
[#__codelineno-4-4](#__codelineno-4-4)│   ├── lib
[#__codelineno-4-5](#__codelineno-4-5)│   └── pyvenv.cfg
[#__codelineno-4-6](#__codelineno-4-6)├── .python-version
[#__codelineno-4-7](#__codelineno-4-7)├── README.md
[#__codelineno-4-8](#__codelineno-4-8)├── main.py
[#__codelineno-4-9](#__codelineno-4-9)├── pyproject.toml
[#__codelineno-4-10](#__codelineno-4-10)└── uv.lock

```

### [`pyproject.toml `](#pyprojecttoml)

The `pyproject.toml `contains metadata about your project: 

pyproject.toml 

```
[#__codelineno-5-1](#__codelineno-5-1)[project]
[#__codelineno-5-2](#__codelineno-5-2)name = "hello-world"
[#__codelineno-5-3](#__codelineno-5-3)version = "0.1.0"
[#__codelineno-5-4](#__codelineno-5-4)description = "Add your description here"
[#__codelineno-5-5](#__codelineno-5-5)readme = "README.md"
[#__codelineno-5-6](#__codelineno-5-6)dependencies = []

```

You'll use this file to specify dependencies, as well as details about the project such as its description or license. You can edit this file manually, or use commands like `uv add `and `uv remove `to manage your project from the terminal. 

!!! tip "Tip"

    See the official [`pyproject.toml `guide](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/)for more details on getting started with the `pyproject.toml `format. 

You'll also use this file to specify uv [configuration options](../../concepts/configuration-files/)in a [`[tool.uv] `](../../reference/settings/)section. 

### [`.python-version `](#python-version)

The `.python-version `file contains the project's default Python version. This file tells uv which Python version to use when creating the project's virtual environment. 

### [`.venv `](#venv)

The `.venv `folder contains your project's virtual environment, a Python environment that is isolated from the rest of your system. This is where uv will install your project's dependencies. 

See the [project environment](../../concepts/projects/layout/#the-project-environment)documentation for more details. 

### [`uv.lock `](#uvlock)

`uv.lock `is a cross-platform lockfile that contains exact information about your project's dependencies. Unlike the `pyproject.toml `which is used to specify the broad requirements of your project, the lockfile contains the exact resolved versions that are installed in the project environment. This file should be checked into version control, allowing for consistent and reproducible installations across machines. 

`uv.lock `is a human-readable TOML file but is managed by uv and should not be edited manually. 

See the [lockfile](../../concepts/projects/layout/#the-lockfile)documentation for more details. 

## [Managing dependencies](#managing-dependencies)

You can add dependencies to your `pyproject.toml `with the `uv add `command. This will also update the lockfile and project environment: 

```
[#__codelineno-6-1](#__codelineno-6-1)$ uv add requests

```

You can also specify version constraints or alternative sources: 

```
[#__codelineno-7-1](#__codelineno-7-1)$ # Specify a version constraint
[#__codelineno-7-2](#__codelineno-7-2)$ uv add 'requests==2.31.0'
[#__codelineno-7-3](#__codelineno-7-3)
[#__codelineno-7-4](#__codelineno-7-4)$ # Add a git dependency
[#__codelineno-7-5](#__codelineno-7-5)$ uv add git+https://github.com/psf/requests

```

If you're migrating from a `requirements.txt `file, you can use `uv add `with the `-r `flag to add all dependencies from the file: 

```
[#__codelineno-8-1](#__codelineno-8-1)$ # Add all dependencies from `requirements.txt`.
[#__codelineno-8-2](#__codelineno-8-2)$ uv add -r requirements.txt -c constraints.txt

```

To remove a package, you can use `uv remove `: 

```
[#__codelineno-9-1](#__codelineno-9-1)$ uv remove requests

```

To upgrade a package, run `uv lock `with the `--upgrade-package `flag: 

```
[#__codelineno-10-1](#__codelineno-10-1)$ uv lock --upgrade-package requests

```

The `--upgrade-package `flag will attempt to update the specified package to the latest compatible version, while keeping the rest of the lockfile intact. 

See the documentation on [managing dependencies](../../concepts/projects/dependencies/)for more details. 

## [Viewing your version](#viewing-your-version)

The `uv version `command can be used to read your package's version. 

To get the version of your package, run `uv version `: 

```
[#__codelineno-11-1](#__codelineno-11-1)$ uv version
[#__codelineno-11-2](#__codelineno-11-2)hello-world 0.7.0

```

To get the version without the package name, use the `--short `option: 

```
[#__codelineno-12-1](#__codelineno-12-1)$ uv version --short
[#__codelineno-12-2](#__codelineno-12-2)0.7.0

```

To get version information in a JSON format, use the `--output-format json `option: 

```
[#__codelineno-13-1](#__codelineno-13-1)$ uv version --output-format json
[#__codelineno-13-2](#__codelineno-13-2){
[#__codelineno-13-3](#__codelineno-13-3)    "package_name": "hello-world",
[#__codelineno-13-4](#__codelineno-13-4)    "version": "0.7.0",
[#__codelineno-13-5](#__codelineno-13-5)    "commit_info": null
[#__codelineno-13-6](#__codelineno-13-6)}

```

See the [publishing guide](../package/#updating-your-version)for details on updating your package version. 

## [Running commands](#running-commands)

`uv run `can be used to run arbitrary scripts or commands in your project environment. 

Prior to every `uv run `invocation, uv will verify that the lockfile is up-to-date with the `pyproject.toml `, and that the environment is up-to-date with the lockfile, keeping your project in-sync without the need for manual intervention. `uv run `guarantees that your command is run in a consistent, locked environment. 

For example, to use `flask `: 

```
[#__codelineno-14-1](#__codelineno-14-1)$ uv add flask
[#__codelineno-14-2](#__codelineno-14-2)$ uv run -- flask run -p 3000

```

Or, to run a script: 

example.py 

```
[#__codelineno-15-1](#__codelineno-15-1)# Require a project dependency
[#__codelineno-15-2](#__codelineno-15-2)import flask
[#__codelineno-15-3](#__codelineno-15-3)
[#__codelineno-15-4](#__codelineno-15-4)print("hello world")

```

```
[#__codelineno-16-1](#__codelineno-16-1)$ uv run example.py

```

Alternatively, you can use `uv sync `to manually update the environment then activate it before executing a command: 

=== "macOS and Linux"

```
[#__codelineno-17-1](#__codelineno-17-1)$ uv sync
[#__codelineno-17-2](#__codelineno-17-2)$ source .venv/bin/activate
[#__codelineno-17-3](#__codelineno-17-3)$ flask run -p 3000
[#__codelineno-17-4](#__codelineno-17-4)$ python example.py

```

=== "Windows"

```
[#__codelineno-18-1](#__codelineno-18-1)PS> uv sync
[#__codelineno-18-2](#__codelineno-18-2)PS> .venv\Scripts\activate
[#__codelineno-18-3](#__codelineno-18-3)PS> flask run -p 3000
[#__codelineno-18-4](#__codelineno-18-4)PS> python example.py

```

!!! note "Note"

    The virtual environment must be active to run scripts and commands in the project without `uv run `. Virtual environment activation differs per shell and platform. 

See the documentation on [running commands and scripts](../../concepts/projects/run/)in projects for more details. 

## [Building distributions](#building-distributions)

`uv build `can be used to build source distributions and binary distributions (wheel) for your project. 

By default, `uv build `will build the project in the current directory, and place the built artifacts in a `dist/ `subdirectory: 

```
[#__codelineno-19-1](#__codelineno-19-1)$ uv build
[#__codelineno-19-2](#__codelineno-19-2)$ ls dist/
[#__codelineno-19-3](#__codelineno-19-3)hello-world-0.1.0-py3-none-any.whl
[#__codelineno-19-4](#__codelineno-19-4)hello-world-0.1.0.tar.gz

```

See the documentation on [building projects](../../concepts/projects/build/)for more details. 

## [Next steps](#next-steps)

To learn more about working on projects with uv, see the [projects concept](../../concepts/projects/)page and the [command reference](../../reference/cli/#uv). 

Or, read on to learn how to [export a uv lockfile to different formats](../../concepts/projects/export/). 

November 24, 2025
