# [Creating projects](#creating-projects)

uv supports creating a project with `uv init `. 

When creating projects, uv supports two basic templates: [**applications **](#applications)and [**libraries **](#libraries). By default, uv will create a project for an application. The `--lib `flag can be used to create a project for a library instead. 

## [Target directory](#target-directory)

uv will create a project in the working directory, or, in a target directory by providing a name, e.g., `uv init foo `. The working directory can be modified with the `--directory `option, which will cause the target directory path will be interpreted relative to the specified working directory. If there's already a project in the target directory, i.e., if there's a `pyproject.toml `, uv will exit with an error. 

## [Applications](#applications)

Application projects are suitable for web servers, scripts, and command-line interfaces. 

Applications are the default target for `uv init `, but can also be specified with the `--app `flag. 

```
[#__codelineno-0-1](#__codelineno-0-1)$ uv init example-app

```

The project includes a `pyproject.toml `, a sample file ( `main.py `), a readme, and a Python version pin file ( `.python-version `). 

```
[#__codelineno-1-1](#__codelineno-1-1)$ tree example-app
[#__codelineno-1-2](#__codelineno-1-2)example-app
[#__codelineno-1-3](#__codelineno-1-3)├── .python-version
[#__codelineno-1-4](#__codelineno-1-4)├── README.md
[#__codelineno-1-5](#__codelineno-1-5)├── main.py
[#__codelineno-1-6](#__codelineno-1-6)└── pyproject.toml

```

!!! note "Note"

    Prior to v0.6.0, uv created a file named `hello.py `instead of `main.py `. 

The `pyproject.toml `includes basic metadata. It does not include a build system, it is not a [package](../config/#project-packaging)and will not be installed into the environment: 

pyproject.toml 

```
[#__codelineno-2-1](#__codelineno-2-1)[project]
[#__codelineno-2-2](#__codelineno-2-2)name = "example-app"
[#__codelineno-2-3](#__codelineno-2-3)version = "0.1.0"
[#__codelineno-2-4](#__codelineno-2-4)description = "Add your description here"
[#__codelineno-2-5](#__codelineno-2-5)readme = "README.md"
[#__codelineno-2-6](#__codelineno-2-6)requires-python = ">=3.11"
[#__codelineno-2-7](#__codelineno-2-7)dependencies = []

```

The sample file defines a `main `function with some standard boilerplate: 

main.py 

```
[#__codelineno-3-1](#__codelineno-3-1)def main():
[#__codelineno-3-2](#__codelineno-3-2)    print("Hello from example-app!")
[#__codelineno-3-3](#__codelineno-3-3)
[#__codelineno-3-4](#__codelineno-3-4)
[#__codelineno-3-5](#__codelineno-3-5)if __name__ == "__main__":
[#__codelineno-3-6](#__codelineno-3-6)    main()

```

Python files can be executed with `uv run `: 

```
[#__codelineno-4-1](#__codelineno-4-1)$ cd example-app
[#__codelineno-4-2](#__codelineno-4-2)$ uv run main.py
[#__codelineno-4-3](#__codelineno-4-3)Hello from example-project!

```

## [Packaged applications](#packaged-applications)

Many use-cases require a [package](../config/#project-packaging). For example, if you are creating a command-line interface that will be published to PyPI or if you want to define tests in a dedicated directory. 

The `--package `flag can be used to create a packaged application: 

```
[#__codelineno-5-1](#__codelineno-5-1)$ uv init --package example-pkg

```

The source code is moved into a `src `directory with a module directory and an `__init__.py `file: 

```
[#__codelineno-6-1](#__codelineno-6-1)$ tree example-pkg
[#__codelineno-6-2](#__codelineno-6-2)example-pkg
[#__codelineno-6-3](#__codelineno-6-3)├── .python-version
[#__codelineno-6-4](#__codelineno-6-4)├── README.md
[#__codelineno-6-5](#__codelineno-6-5)├── pyproject.toml
[#__codelineno-6-6](#__codelineno-6-6)└── src
[#__codelineno-6-7](#__codelineno-6-7)    └── example_pkg
[#__codelineno-6-8](#__codelineno-6-8)        └── __init__.py

```

A [build system](../config/#build-systems)is defined, so the project will be installed into the environment: 

pyproject.toml 

```
[#__codelineno-7-1](#__codelineno-7-1)[project]
[#__codelineno-7-2](#__codelineno-7-2)name = "example-pkg"
[#__codelineno-7-3](#__codelineno-7-3)version = "0.1.0"
[#__codelineno-7-4](#__codelineno-7-4)description = "Add your description here"
[#__codelineno-7-5](#__codelineno-7-5)readme = "README.md"
[#__codelineno-7-6](#__codelineno-7-6)requires-python = ">=3.11"
[#__codelineno-7-7](#__codelineno-7-7)dependencies = []
[#__codelineno-7-8](#__codelineno-7-8)
[#__codelineno-7-9](#__codelineno-7-9)[project.scripts]
[#__codelineno-7-10](#__codelineno-7-10)example-pkg = "example_pkg:main"
[#__codelineno-7-11](#__codelineno-7-11)
[#__codelineno-7-12](#__codelineno-7-12)[build-system]
[#__codelineno-7-13](#__codelineno-7-13)requires = ["uv_build>=0.9.16,<0.10.0"]
[#__codelineno-7-14](#__codelineno-7-14)build-backend = "uv_build"

```

!!! tip "Tip"

    The `--build-backend `option can be used to request an alternative build system. 

A [command](../config/#entry-points)definition is included: 

pyproject.toml 

```
[#__codelineno-8-1](#__codelineno-8-1)[project]
[#__codelineno-8-2](#__codelineno-8-2)name = "example-pkg"
[#__codelineno-8-3](#__codelineno-8-3)version = "0.1.0"
[#__codelineno-8-4](#__codelineno-8-4)description = "Add your description here"
[#__codelineno-8-5](#__codelineno-8-5)readme = "README.md"
[#__codelineno-8-6](#__codelineno-8-6)requires-python = ">=3.11"
[#__codelineno-8-7](#__codelineno-8-7)dependencies = []
[#__codelineno-8-8](#__codelineno-8-8)
[#__codelineno-8-9](#__codelineno-8-9)[project.scripts]
[#__codelineno-8-10](#__codelineno-8-10)example-pkg = "example_pkg:main"
[#__codelineno-8-11](#__codelineno-8-11)
[#__codelineno-8-12](#__codelineno-8-12)[build-system]
[#__codelineno-8-13](#__codelineno-8-13)requires = ["uv_build>=0.9.16,<0.10.0"]
[#__codelineno-8-14](#__codelineno-8-14)build-backend = "uv_build"

```

The command can be executed with `uv run `: 

```
[#__codelineno-9-1](#__codelineno-9-1)$ cd example-pkg
[#__codelineno-9-2](#__codelineno-9-2)$ uv run example-pkg
[#__codelineno-9-3](#__codelineno-9-3)Hello from example-pkg!

```

## [Libraries](#libraries)

A library provides functions and objects for other projects to consume. Libraries are intended to be built and distributed, e.g., by uploading them to PyPI. 

Libraries can be created by using the `--lib `flag: 

```
[#__codelineno-10-1](#__codelineno-10-1)$ uv init --lib example-lib

```

!!! note "Note"

    Using `--lib `implies `--package `. Libraries always require a packaged project. 

As with a [packaged application](#packaged-applications), a `src `layout is used. A `py.typed `marker is included to indicate to consumers that types can be read from the library: 

```
[#__codelineno-11-1](#__codelineno-11-1)$ tree example-lib
[#__codelineno-11-2](#__codelineno-11-2)example-lib
[#__codelineno-11-3](#__codelineno-11-3)├── .python-version
[#__codelineno-11-4](#__codelineno-11-4)├── README.md
[#__codelineno-11-5](#__codelineno-11-5)├── pyproject.toml
[#__codelineno-11-6](#__codelineno-11-6)└── src
[#__codelineno-11-7](#__codelineno-11-7)    └── example_lib
[#__codelineno-11-8](#__codelineno-11-8)        ├── py.typed
[#__codelineno-11-9](#__codelineno-11-9)        └── __init__.py

```

!!! note "Note"

    A `src `layout is particularly valuable when developing libraries. It ensures that the library is isolated from any `python `invocations in the project root and that distributed library code is well separated from the rest of the project source. 

A [build system](../config/#build-systems)is defined, so the project will be installed into the environment: 

pyproject.toml 

```
[#__codelineno-12-1](#__codelineno-12-1)[project]
[#__codelineno-12-2](#__codelineno-12-2)name = "example-lib"
[#__codelineno-12-3](#__codelineno-12-3)version = "0.1.0"
[#__codelineno-12-4](#__codelineno-12-4)description = "Add your description here"
[#__codelineno-12-5](#__codelineno-12-5)readme = "README.md"
[#__codelineno-12-6](#__codelineno-12-6)requires-python = ">=3.11"
[#__codelineno-12-7](#__codelineno-12-7)dependencies = []
[#__codelineno-12-8](#__codelineno-12-8)
[#__codelineno-12-9](#__codelineno-12-9)[build-system]
[#__codelineno-12-10](#__codelineno-12-10)requires = ["uv_build>=0.9.16,<0.10.0"]
[#__codelineno-12-11](#__codelineno-12-11)build-backend = "uv_build"

```

!!! tip "Tip"

    You can select a different build backend template by using `--build-backend `with `hatchling `, `uv_build `, `flit-core `, `pdm-backend `, `setuptools `, `maturin `, or `scikit-build-core `. An alternative backend is required if you want to create a [library with extension modules](#projects-with-extension-modules). 

The created module defines a simple API function: 

__init__.py 

```
[#__codelineno-13-1](#__codelineno-13-1)def hello() -> str:
[#__codelineno-13-2](#__codelineno-13-2)    return "Hello from example-lib!"

```

And you can import and execute it using `uv run `: 

```
[#__codelineno-14-1](#__codelineno-14-1)$ cd example-lib
[#__codelineno-14-2](#__codelineno-14-2)$ uv run python -c "import example_lib; print(example_lib.hello())"
[#__codelineno-14-3](#__codelineno-14-3)Hello from example-lib!

```

## [Projects with extension modules](#projects-with-extension-modules)

Most Python projects are "pure Python", meaning they do not define modules in other languages like C, C++, FORTRAN, or Rust. However, projects with extension modules are often used for performance sensitive code. 

Creating a project with an extension module requires choosing an alternative build system. uv supports creating projects with the following build systems that support building extension modules: 

- [`maturin `](https://www.maturin.rs)for projects with Rust 

- [`scikit-build-core `](https://github.com/scikit-build/scikit-build-core)for projects with C, C++, FORTRAN, Cython 

Specify the build system with the `--build-backend `flag: 

```
[#__codelineno-15-1](#__codelineno-15-1)$ uv init --build-backend maturin example-ext

```

!!! note "Note"

    Using `--build-backend `implies `--package `. 

The project contains a `Cargo.toml `and a `lib.rs `file in addition to the typical Python project files: 

```
[#__codelineno-16-1](#__codelineno-16-1)$ tree example-ext
[#__codelineno-16-2](#__codelineno-16-2)example-ext
[#__codelineno-16-3](#__codelineno-16-3)├── .python-version
[#__codelineno-16-4](#__codelineno-16-4)├── Cargo.toml
[#__codelineno-16-5](#__codelineno-16-5)├── README.md
[#__codelineno-16-6](#__codelineno-16-6)├── pyproject.toml
[#__codelineno-16-7](#__codelineno-16-7)└── src
[#__codelineno-16-8](#__codelineno-16-8)    ├── lib.rs
[#__codelineno-16-9](#__codelineno-16-9)    └── example_ext
[#__codelineno-16-10](#__codelineno-16-10)        ├── __init__.py
[#__codelineno-16-11](#__codelineno-16-11)        └── _core.pyi

```

!!! note "Note"

    If using `scikit-build-core `, you'll see CMake configuration and a `main.cpp `file instead. 

The Rust library defines a simple function: 

src/lib.rs 

```
[#__codelineno-17-1](#__codelineno-17-1)use pyo3::prelude::*;
[#__codelineno-17-2](#__codelineno-17-2)
[#__codelineno-17-3](#__codelineno-17-3)#[pymodule]
[#__codelineno-17-4](#__codelineno-17-4)mod _core {
[#__codelineno-17-5](#__codelineno-17-5)    use pyo3::prelude::*;
[#__codelineno-17-6](#__codelineno-17-6)
[#__codelineno-17-7](#__codelineno-17-7)    #[pyfunction]
[#__codelineno-17-8](#__codelineno-17-8)    fn hello_from_bin() -> String {
[#__codelineno-17-9](#__codelineno-17-9)        "Hello from example-ext!".to_string()
[#__codelineno-17-10](#__codelineno-17-10)    }
[#__codelineno-17-11](#__codelineno-17-11)}

```

And the Python module imports it: 

src/example_ext/__init__.py 

```
[#__codelineno-18-1](#__codelineno-18-1)from example_ext._core import hello_from_bin
[#__codelineno-18-2](#__codelineno-18-2)
[#__codelineno-18-3](#__codelineno-18-3)
[#__codelineno-18-4](#__codelineno-18-4)def main() -> None:
[#__codelineno-18-5](#__codelineno-18-5)    print(hello_from_bin())

```

The command can be executed with `uv run `: 

```
[#__codelineno-19-1](#__codelineno-19-1)$ cd example-ext
[#__codelineno-19-2](#__codelineno-19-2)$ uv run example-ext
[#__codelineno-19-3](#__codelineno-19-3)Hello from example-ext!

```

!!! important "Important"

    When creating a project with maturin or scikit-build-core, uv configures [`tool.uv.cache-keys `](https://docs.astral.sh/uv/reference/settings/#cache-keys)to include common source file types. To force a rebuild, e.g. when changing files outside `cache-keys `or when not using `cache-keys `, use `--reinstall `. 

## [Creating a minimal project](#creating-a-minimal-project)

If you only want to create a `pyproject.toml `, use the `--bare `option: 

```
[#__codelineno-20-1](#__codelineno-20-1)$ uv init example --bare

```

uv will skip creating a Python version pin file, a README, and any source directories or files. Additionally, uv will not initialize a version control system (i.e., `git `). 

```
[#__codelineno-21-1](#__codelineno-21-1)$ tree example-bare
[#__codelineno-21-2](#__codelineno-21-2)example-bare
[#__codelineno-21-3](#__codelineno-21-3)└── pyproject.toml

```

uv will also not add extra metadata to the `pyproject.toml `, such as the `description `or `authors `. 

```
[#__codelineno-22-1](#__codelineno-22-1)[project]
[#__codelineno-22-2](#__codelineno-22-2)name = "example"
[#__codelineno-22-3](#__codelineno-22-3)version = "0.1.0"
[#__codelineno-22-4](#__codelineno-22-4)requires-python = ">=3.12"
[#__codelineno-22-5](#__codelineno-22-5)dependencies = []

```

The `--bare `option can be used with other options like `--lib `or `--build-backend `— in these cases uv will still configure a build system but will not create the expected file structure. 

When `--bare `is used, additional features can still be used opt-in: 

```
[#__codelineno-23-1](#__codelineno-23-1)$ uv init example --bare --description "Hello world" --author-from git --vcs git --python-pin

```

December 6, 2025
