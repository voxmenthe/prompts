# [Migrating from pip to a uv project](#migrating-from-pip-to-a-uv-project)

This guide will discuss converting from a `pip `and `pip-tools `workflow centered on `requirements `files to uv's project workflow using a `pyproject.toml `and `uv.lock `file. 

!!! note "Note"

    If you're looking to migrate from `pip `and `pip-tools `to uv's drop-in interface or from an existing workflow where you're already using a `pyproject.toml `, those guides are not yet written. See [#5200](https://github.com/astral-sh/uv/issues/5200)to track progress. 

We'll start with an overview of developing with `pip `, then discuss migrating to uv. 

!!! tip "Tip"

    If you're familiar with the ecosystem, you can jump ahead to the [requirements file import](#importing-requirements-files)instructions. 

## [Understanding pip workflows](#understanding-pip-workflows)

### [Project dependencies](#project-dependencies)

When you want to use a package in your project, you need to install it first. `pip `supports imperative installation of packages, e.g.: 

```
[#__codelineno-0-1](#__codelineno-0-1)$ pip install fastapi

```

This installs the package into the environment that `pip `is installed in. This may be a virtual environment, or, the global environment of your system's Python installation. 

Then, you can run a Python script that requires the package: 

example.py 

```
[#__codelineno-1-1](#__codelineno-1-1)import fastapi

```

It's best practice to create a virtual environment for each project, to avoid mixing packages between them. For example: 

```
[#__codelineno-2-1](#__codelineno-2-1)$ python -m venv
[#__codelineno-2-2](#__codelineno-2-2)$ source .venv/bin/activate
[#__codelineno-2-3](#__codelineno-2-3)$ pip ...

```

We will revisit this topic in the [project environments section](#project-environments)below. 

### [Requirements files](#requirements-files)

When sharing projects with others, it's useful to declare all the packages you require upfront. `pip `supports installing requirements from a file, e.g.: 

requirements.txt 

```
[#__codelineno-3-1](#__codelineno-3-1)fastapi

```

```
[#__codelineno-4-1](#__codelineno-4-1)$ pip install -r requirements.txt

```

Notice above that `fastapi `is not "locked" to a specific version — each person working on the project may have a different version of `fastapi `installed. `pip-tools `was created to improve this experience. 

When using `pip-tools `, requirements files specify both the dependencies for your project and lock dependencies to a specific version — the file extension is used to differentiate between the two. For example, if you require `fastapi `and `pydantic `, you'd specify these in a `requirements.in `file: 

requirements.in 

```
[#__codelineno-5-1](#__codelineno-5-1)fastapi
[#__codelineno-5-2](#__codelineno-5-2)pydantic>2

```

Notice there's a version constraint on `pydantic `— this means only `pydantic `versions later than `2.0.0 `can be used. In contrast, `fastapi `does not have a version constraint — any version can be used. 

These dependencies can be compiled into a `requirements.txt `file: 

```
[#__codelineno-6-1](#__codelineno-6-1)$ pip-compile requirements.in -o requirements.txt

```

requirements.txt 

```
[#__codelineno-7-1](#__codelineno-7-1)annotated-types==0.7.0
[#__codelineno-7-2](#__codelineno-7-2)    # via pydantic
[#__codelineno-7-3](#__codelineno-7-3)anyio==4.8.0
[#__codelineno-7-4](#__codelineno-7-4)    # via starlette
[#__codelineno-7-5](#__codelineno-7-5)fastapi==0.115.11
[#__codelineno-7-6](#__codelineno-7-6)    # via -r requirements.in
[#__codelineno-7-7](#__codelineno-7-7)idna==3.10
[#__codelineno-7-8](#__codelineno-7-8)    # via anyio
[#__codelineno-7-9](#__codelineno-7-9)pydantic==2.10.6
[#__codelineno-7-10](#__codelineno-7-10)    # via
[#__codelineno-7-11](#__codelineno-7-11)    #   -r requirements.in
[#__codelineno-7-12](#__codelineno-7-12)    #   fastapi
[#__codelineno-7-13](#__codelineno-7-13)pydantic-core==2.27.2
[#__codelineno-7-14](#__codelineno-7-14)    # via pydantic
[#__codelineno-7-15](#__codelineno-7-15)sniffio==1.3.1
[#__codelineno-7-16](#__codelineno-7-16)    # via anyio
[#__codelineno-7-17](#__codelineno-7-17)starlette==0.46.1
[#__codelineno-7-18](#__codelineno-7-18)    # via fastapi
[#__codelineno-7-19](#__codelineno-7-19)typing-extensions==4.12.2
[#__codelineno-7-20](#__codelineno-7-20)    # via
[#__codelineno-7-21](#__codelineno-7-21)    #   fastapi
[#__codelineno-7-22](#__codelineno-7-22)    #   pydantic
[#__codelineno-7-23](#__codelineno-7-23)    #   pydantic-core

```

Here, all the versions constraints are _exact _. Only a single version of each package can be used. The above example was generated with `uv pip compile `, but could also be generated with `pip-compile `from `pip-tools `. 

Though less common, the `requirements.txt `can also be generated using `pip freeze `, by first installing the input dependencies into the environment then exporting the installed versions: 

```
[#__codelineno-8-1](#__codelineno-8-1)$ pip install -r requirements.in
[#__codelineno-8-2](#__codelineno-8-2)$ pip freeze > requirements.txt

```

requirements.txt 

```
[#__codelineno-9-1](#__codelineno-9-1)annotated-types==0.7.0
[#__codelineno-9-2](#__codelineno-9-2)anyio==4.8.0
[#__codelineno-9-3](#__codelineno-9-3)fastapi==0.115.11
[#__codelineno-9-4](#__codelineno-9-4)idna==3.10
[#__codelineno-9-5](#__codelineno-9-5)pydantic==2.10.6
[#__codelineno-9-6](#__codelineno-9-6)pydantic-core==2.27.2
[#__codelineno-9-7](#__codelineno-9-7)sniffio==1.3.1
[#__codelineno-9-8](#__codelineno-9-8)starlette==0.46.1
[#__codelineno-9-9](#__codelineno-9-9)typing-extensions==4.12.2

```

After compiling dependencies into a locked set of versions, these files are committed to version control and distributed with the project. 

Then, when someone wants to use the project, they install from the requirements file: 

```
[#__codelineno-10-1](#__codelineno-10-1)$ pip install -r requirements.txt

```

### [Development dependencies](#development-dependencies)

The requirements file format can only describe a single set of dependencies at once. This means if you have additional _groups _of dependencies, such as development dependencies, they need separate files. For example, we'll create a `-dev `dependency file: 

requirements-dev.in 

```
[#__codelineno-11-1](#__codelineno-11-1)-r requirements.in
[#__codelineno-11-2](#__codelineno-11-2)-c requirements.txt
[#__codelineno-11-3](#__codelineno-11-3)
[#__codelineno-11-4](#__codelineno-11-4)pytest

```

Notice the base requirements are included with `-r requirements.in `. This ensures your development environment considers _all _of the dependencies together. The `-c requirements.txt `_constrains _the package version to ensure that the `requirements-dev.txt `uses the same versions as `requirements.txt `. 

!!! note "Note"

    It's common to use `-r requirements.txt `directly instead of using both `-r requirements.in `, and `-c requirements.txt `. There's no difference in the resulting package versions, but using both files produces annotations which allow you to determine which dependencies are _direct _(annotated with `-r requirements.in `) and which are _indirect _(only annotated with `-c requirements.txt `). 

The compiled development dependencies look like: 

requirements-dev.txt 

```
[#__codelineno-12-1](#__codelineno-12-1)annotated-types==0.7.0
[#__codelineno-12-2](#__codelineno-12-2)    # via
[#__codelineno-12-3](#__codelineno-12-3)    #   -c requirements.txt
[#__codelineno-12-4](#__codelineno-12-4)    #   pydantic
[#__codelineno-12-5](#__codelineno-12-5)anyio==4.8.0
[#__codelineno-12-6](#__codelineno-12-6)    # via
[#__codelineno-12-7](#__codelineno-12-7)    #   -c requirements.txt
[#__codelineno-12-8](#__codelineno-12-8)    #   starlette
[#__codelineno-12-9](#__codelineno-12-9)fastapi==0.115.11
[#__codelineno-12-10](#__codelineno-12-10)    # via
[#__codelineno-12-11](#__codelineno-12-11)    #   -c requirements.txt
[#__codelineno-12-12](#__codelineno-12-12)    #   -r requirements.in
[#__codelineno-12-13](#__codelineno-12-13)idna==3.10
[#__codelineno-12-14](#__codelineno-12-14)    # via
[#__codelineno-12-15](#__codelineno-12-15)    #   -c requirements.txt
[#__codelineno-12-16](#__codelineno-12-16)    #   anyio
[#__codelineno-12-17](#__codelineno-12-17)iniconfig==2.0.0
[#__codelineno-12-18](#__codelineno-12-18)    # via pytest
[#__codelineno-12-19](#__codelineno-12-19)packaging==24.2
[#__codelineno-12-20](#__codelineno-12-20)    # via pytest
[#__codelineno-12-21](#__codelineno-12-21)pluggy==1.5.0
[#__codelineno-12-22](#__codelineno-12-22)    # via pytest
[#__codelineno-12-23](#__codelineno-12-23)pydantic==2.10.6
[#__codelineno-12-24](#__codelineno-12-24)    # via
[#__codelineno-12-25](#__codelineno-12-25)    #   -c requirements.txt
[#__codelineno-12-26](#__codelineno-12-26)    #   -r requirements.in
[#__codelineno-12-27](#__codelineno-12-27)    #   fastapi
[#__codelineno-12-28](#__codelineno-12-28)pydantic-core==2.27.2
[#__codelineno-12-29](#__codelineno-12-29)    # via
[#__codelineno-12-30](#__codelineno-12-30)    #   -c requirements.txt
[#__codelineno-12-31](#__codelineno-12-31)    #   pydantic
[#__codelineno-12-32](#__codelineno-12-32)pytest==8.3.5
[#__codelineno-12-33](#__codelineno-12-33)    # via -r requirements-dev.in
[#__codelineno-12-34](#__codelineno-12-34)sniffio==1.3.1
[#__codelineno-12-35](#__codelineno-12-35)    # via
[#__codelineno-12-36](#__codelineno-12-36)    #   -c requirements.txt
[#__codelineno-12-37](#__codelineno-12-37)    #   anyio
[#__codelineno-12-38](#__codelineno-12-38)starlette==0.46.1
[#__codelineno-12-39](#__codelineno-12-39)    # via
[#__codelineno-12-40](#__codelineno-12-40)    #   -c requirements.txt
[#__codelineno-12-41](#__codelineno-12-41)    #   fastapi
[#__codelineno-12-42](#__codelineno-12-42)typing-extensions==4.12.2
[#__codelineno-12-43](#__codelineno-12-43)    # via
[#__codelineno-12-44](#__codelineno-12-44)    #   -c requirements.txt
[#__codelineno-12-45](#__codelineno-12-45)    #   fastapi
[#__codelineno-12-46](#__codelineno-12-46)    #   pydantic
[#__codelineno-12-47](#__codelineno-12-47)    #   pydantic-core

```

As with the base dependency files, these are committed to version control and distributed with the project. When someone wants to work on the project, they'll install from the requirements file: 

```
[#__codelineno-13-1](#__codelineno-13-1)$ pip install -r requirements-dev.txt

```

### [Platform-specific dependencies](#platform-specific-dependencies)

When compiling dependencies with `pip `or `pip-tools `, the result is only usable on the same platform as it is generated on. This poses a problem for projects which need to be usable on multiple platforms, such as Windows and macOS. 

For example, take a simple dependency: 

requirements.in 

```
[#__codelineno-14-1](#__codelineno-14-1)tqdm

```

On Linux, this compiles to: 

requirements-linux.txt 

```
[#__codelineno-15-1](#__codelineno-15-1)tqdm==4.67.1
[#__codelineno-15-2](#__codelineno-15-2)    # via -r requirements.in

```

While on Windows, this compiles to: 

requirements-win.txt 

```
[#__codelineno-16-1](#__codelineno-16-1)colorama==0.4.6
[#__codelineno-16-2](#__codelineno-16-2)    # via tqdm
[#__codelineno-16-3](#__codelineno-16-3)tqdm==4.67.1
[#__codelineno-16-4](#__codelineno-16-4)    # via -r requirements.in

```

`colorama `is a Windows-only dependency of `tqdm `. 

When using `pip `and `pip-tools `, a project needs to declare a requirements lock file for each supported platform. 

!!! note "Note"

    uv's resolver can compile dependencies for multiple platforms at once (see ["universal resolution"](../../../concepts/resolution/#universal-resolution)), allowing you to use a single `requirements.txt `for all platforms: 

    ```
[#__codelineno-17-1](#__codelineno-17-1)$ uv pip compile --universal requirements.in

```

    requirements.txt 

    ```
[#__codelineno-18-1](#__codelineno-18-1)colorama==0.4.6 ; sys_platform == 'win32'
[#__codelineno-18-2](#__codelineno-18-2)    # via tqdm
[#__codelineno-18-3](#__codelineno-18-3)tqdm==4.67.1
[#__codelineno-18-4](#__codelineno-18-4)    # via -r requirements.in

```

    This resolution mode is also used when using a `pyproject.toml `and `uv.lock `. 

## [Migrating to a uv project](#migrating-to-a-uv-project)

### [The `pyproject.toml `](#the-pyprojecttoml)

The `pyproject.toml `is a standardized file for Python project metadata. It replaces `requirements.in `files, allowing you to represent arbitrary groups of project dependencies. It also provides a centralized location for metadata about your project, such as the build system or tool settings. 

For example, the `requirements.in `and `requirements-dev.in `files above can be translated to a `pyproject.toml `as follows: 

pyproject.toml 

```
[#__codelineno-19-1](#__codelineno-19-1)[project]
[#__codelineno-19-2](#__codelineno-19-2)name = "example"
[#__codelineno-19-3](#__codelineno-19-3)version = "0.0.1"
[#__codelineno-19-4](#__codelineno-19-4)dependencies = [
[#__codelineno-19-5](#__codelineno-19-5)    "fastapi",
[#__codelineno-19-6](#__codelineno-19-6)    "pydantic>2"
[#__codelineno-19-7](#__codelineno-19-7)]
[#__codelineno-19-8](#__codelineno-19-8)
[#__codelineno-19-9](#__codelineno-19-9)[dependency-groups]
[#__codelineno-19-10](#__codelineno-19-10)dev = ["pytest"]

```

We'll discuss the commands necessary to automate these imports below. 

### [The uv lockfile](#the-uv-lockfile)

uv uses a lockfile ( `uv.lock `) file to lock package versions. The format of this file is specific to uv, allowing uv to support advanced features. It replaces `requirements.txt `files. 

The lockfile will be automatically created and populated when adding dependencies, but you can explicitly create it with `uv lock `. 

Unlike `requirements.txt `files, the `uv.lock `file can represent arbitrary groups of dependencies, so multiple files are not needed to lock development dependencies. 

The uv lockfile is always [universal](../../../concepts/resolution/#universal-resolution), so multiple files are not needed to [lock dependencies for each platform](#platform-specific-dependencies). This ensures that all developers are using consistent, locked versions of dependencies regardless of their machine. 

The uv lockfile also supports concepts like [pinning packages to specific indexes](../../../concepts/indexes/#pinning-a-package-to-an-index), which is not representable in `requirements.txt `files. 

!!! tip "Tip"

    If you only need to lock for a subset of platforms, use the [`tool.uv.environments `](../../../concepts/resolution/#limited-resolution-environments)setting to limit the resolution and lockfile. 

To learn more, see the [lockfile](../../../concepts/projects/layout/#the-lockfile)documentation. 

### [Importing requirements files](#importing-requirements-files)

First, create a `pyproject.toml `if you have not already: 

```
[#__codelineno-20-1](#__codelineno-20-1)$ uv init

```

Then, the easiest way to import requirements is with `uv add `: 

```
[#__codelineno-21-1](#__codelineno-21-1)$ uv add -r requirements.in

```

However, there is some nuance to this transition. Notice we used the `requirements.in `file, which does not pin to exact versions of packages so uv will solve for new versions of these packages. You may want to continue using your previously locked versions from your `requirements.txt `so, when switching over to uv, none of your dependency versions change. 

The solution is to add your locked versions as _constraints _. uv supports using these on `add `to preserve locked versions: 

```
[#__codelineno-22-1](#__codelineno-22-1)$ uv add -r requirements.in -c requirements.txt

```

Your existing versions will be retained when producing a `uv.lock `file. 

#### [Importing platform-specific constraints](#importing-platform-specific-constraints)

If your platform-specific dependencies have been compiled into separate files, you can still transition to a universal lockfile. However, you cannot just use `-c `to specify constraints from your existing platform-specific `requirements.txt `files because they do not include markers describing the environment and will consequently conflict. 

To add the necessary markers, use `uv pip compile `to convert your existing files. For example, given the following: 

requirements-win.txt 

```
[#__codelineno-23-1](#__codelineno-23-1)colorama==0.4.6
[#__codelineno-23-2](#__codelineno-23-2)    # via tqdm
[#__codelineno-23-3](#__codelineno-23-3)tqdm==4.67.1
[#__codelineno-23-4](#__codelineno-23-4)    # via -r requirements.in

```

The markers can be added with: 

```
[#__codelineno-24-1](#__codelineno-24-1)$ uv pip compile requirements.in -o requirements-win.txt --python-platform windows --no-strip-markers

```

Notice the resulting output includes a Windows marker on `colorama `: 

requirements-win.txt 

```
[#__codelineno-25-1](#__codelineno-25-1)colorama==0.4.6 ; sys_platform == 'win32'
[#__codelineno-25-2](#__codelineno-25-2)    # via tqdm
[#__codelineno-25-3](#__codelineno-25-3)tqdm==4.67.1
[#__codelineno-25-4](#__codelineno-25-4)    # via -r requirements.in

```

When using `-o `, uv will constrain the versions to match the existing output file, if it can. 

Markers can be added for other platforms by changing the `--python-platform `and `-o `values for each requirements file you need to import, e.g., to `linux `and `macos `. 

Once each `requirements.txt `file has been transformed, the dependencies can be imported to the `pyproject.toml `and `uv.lock `with `uv add `: 

```
[#__codelineno-26-1](#__codelineno-26-1)$ uv add -r requirements.in -c requirements-win.txt -c requirements-linux.txt

```

#### [Importing development dependency files](#importing-development-dependency-files)

As discussed in the [development dependencies](#development-dependencies)section, it's common to have groups of dependencies for development purposes. 

To import development dependencies, use the `--dev `flag during `uv add `: 

```
[#__codelineno-27-1](#__codelineno-27-1)$ uv add --dev -r requirements-dev.in -c requirements-dev.txt

```

If the `requirements-dev.in `includes the parent `requirements.in `via `-r `, it will need to be stripped to avoid adding the base requirements to the `dev `dependency group. The following example uses `sed `to strip lines that start with `-r `, then pipes the result to `uv add `: 

```
[#__codelineno-28-1](#__codelineno-28-1)$ sed '/^-r /d' requirements-dev.in | uv add --dev -r - -c requirements-dev.txt

```

In addition to the `dev `dependency group, uv supports arbitrary group names. For example, if you also have a dedicated set of dependencies for building your documentation, those can be imported to a `docs `group: 

```
[#__codelineno-29-1](#__codelineno-29-1)$ uv add -r requirements-docs.in -c requirements-docs.txt --group docs

```

### [Project environments](#project-environments)

Unlike `pip `, uv is not centered around the concept of an "active" virtual environment. Instead, uv uses a dedicated virtual environment for each project in a `.venv `directory. This environment is automatically managed, so when you run a command, like `uv add `, the environment is synced with the project dependencies. 

The preferred way to execute commands in the environment is with `uv run `, e.g.: 

```
[#__codelineno-30-1](#__codelineno-30-1)$ uv run pytest

```

Prior to every `uv run `invocation, uv will verify that the lockfile is up-to-date with the `pyproject.toml `, and that the environment is up-to-date with the lockfile, keeping your project in-sync without the need for manual intervention. `uv run `guarantees that your command is run in a consistent, locked environment. 

The project environment can also be explicitly created with `uv sync `, e.g., for use with editors. 

!!! note "Note"

    When in projects, uv will prefer a `.venv `in the project directory and ignore the active environment as declared by the `VIRTUAL_ENV `variable by default. You can opt-in to using the active environment with the `--active `flag. 

To learn more, see the [project environment](../../../concepts/projects/layout/#the-project-environment)documentation. 

## [Next steps](#next-steps)

Now that you've migrated to uv, take a look at the [project concept](../../../concepts/projects/)page for more details about uv projects. 

July 3, 2025
