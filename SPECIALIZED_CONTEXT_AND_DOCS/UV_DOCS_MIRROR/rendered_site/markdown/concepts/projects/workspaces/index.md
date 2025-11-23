# [Using workspaces](#using-workspaces)

Inspired by the [Cargo](https://doc.rust-lang.org/cargo/reference/workspaces.html)concept of the same name, a workspace is "a collection of one or more packages, called _workspace members _, that are managed together." 

Workspaces organize large codebases by splitting them into multiple packages with common dependencies. Think: a FastAPI-based web application, alongside a series of libraries that are versioned and maintained as separate Python packages, all in the same Git repository. 

In a workspace, each package defines its own `pyproject.toml `, but the workspace shares a single lockfile, ensuring that the workspace operates with a consistent set of dependencies. 

As such, `uv lock `operates on the entire workspace at once, while `uv run `and `uv sync `operate on the workspace root by default, though both accept a `--package `argument, allowing you to run a command in a particular workspace member from any workspace directory. 

## [Getting started](#getting-started)

To create a workspace, add a `tool.uv.workspace `table to a `pyproject.toml `, which will implicitly create a workspace rooted at that package. 

!!! tip "Tip"

    By default, running `uv init `inside an existing package will add the newly created member to the workspace, creating a `tool.uv.workspace `table in the workspace root if it doesn't already exist. 

In defining a workspace, you must specify the `members `(required) and `exclude `(optional) keys, which direct the workspace to include or exclude specific directories as members respectively, and accept lists of globs: 

pyproject.toml 

```
[#__codelineno-0-1](#__codelineno-0-1)[project]
[#__codelineno-0-2](#__codelineno-0-2)name = "albatross"
[#__codelineno-0-3](#__codelineno-0-3)version = "0.1.0"
[#__codelineno-0-4](#__codelineno-0-4)requires-python = ">=3.12"
[#__codelineno-0-5](#__codelineno-0-5)dependencies = ["bird-feeder", "tqdm>=4,<5"]
[#__codelineno-0-6](#__codelineno-0-6)
[#__codelineno-0-7](#__codelineno-0-7)[tool.uv.sources]
[#__codelineno-0-8](#__codelineno-0-8)bird-feeder = { workspace = true }
[#__codelineno-0-9](#__codelineno-0-9)
[#__codelineno-0-10](#__codelineno-0-10)[tool.uv.workspace]
[#__codelineno-0-11](#__codelineno-0-11)members = ["packages/*"]
[#__codelineno-0-12](#__codelineno-0-12)exclude = ["packages/seeds"]

```

Every directory included by the `members `globs (and not excluded by the `exclude `globs) must contain a `pyproject.toml `file. However, workspace members can be _either _[applications](../init/#applications)or [libraries](../init/#libraries); both are supported in the workspace context. 

Every workspace needs a root, which is _also _a workspace member. In the above example, `albatross `is the workspace root, and the workspace members include all projects under the `packages `directory, except `seeds `. 

By default, `uv run `and `uv sync `operates on the workspace root. For example, in the above example, `uv run `and `uv run --package albatross `would be equivalent, while `uv run --package bird-feeder `would run the command in the `bird-feeder `package. 

## [Workspace sources](#workspace-sources)

Within a workspace, dependencies on workspace members are facilitated via [`tool.uv.sources `](../dependencies/), as in: 

pyproject.toml 

```
[#__codelineno-1-1](#__codelineno-1-1)[project]
[#__codelineno-1-2](#__codelineno-1-2)name = "albatross"
[#__codelineno-1-3](#__codelineno-1-3)version = "0.1.0"
[#__codelineno-1-4](#__codelineno-1-4)requires-python = ">=3.12"
[#__codelineno-1-5](#__codelineno-1-5)dependencies = ["bird-feeder", "tqdm>=4,<5"]
[#__codelineno-1-6](#__codelineno-1-6)
[#__codelineno-1-7](#__codelineno-1-7)[tool.uv.sources]
[#__codelineno-1-8](#__codelineno-1-8)bird-feeder = { workspace = true }
[#__codelineno-1-9](#__codelineno-1-9)
[#__codelineno-1-10](#__codelineno-1-10)[tool.uv.workspace]
[#__codelineno-1-11](#__codelineno-1-11)members = ["packages/*"]
[#__codelineno-1-12](#__codelineno-1-12)
[#__codelineno-1-13](#__codelineno-1-13)[build-system]
[#__codelineno-1-14](#__codelineno-1-14)requires = ["uv_build>=0.9.9,<0.10.0"]
[#__codelineno-1-15](#__codelineno-1-15)build-backend = "uv_build"

```

In this example, the `albatross `project depends on the `bird-feeder `project, which is a member of the workspace. The `workspace = true `key-value pair in the `tool.uv.sources `table indicates the `bird-feeder `dependency should be provided by the workspace, rather than fetched from PyPI or another registry. 

!!! note "Note"

    Dependencies between workspace members are editable. 

Any `tool.uv.sources `definitions in the workspace root apply to all members, unless overridden in the `tool.uv.sources `of a specific member. For example, given the following `pyproject.toml `: 

pyproject.toml 

```
[#__codelineno-2-1](#__codelineno-2-1)[project]
[#__codelineno-2-2](#__codelineno-2-2)name = "albatross"
[#__codelineno-2-3](#__codelineno-2-3)version = "0.1.0"
[#__codelineno-2-4](#__codelineno-2-4)requires-python = ">=3.12"
[#__codelineno-2-5](#__codelineno-2-5)dependencies = ["bird-feeder", "tqdm>=4,<5"]
[#__codelineno-2-6](#__codelineno-2-6)
[#__codelineno-2-7](#__codelineno-2-7)[tool.uv.sources]
[#__codelineno-2-8](#__codelineno-2-8)bird-feeder = { workspace = true }
[#__codelineno-2-9](#__codelineno-2-9)tqdm = { git = "https://github.com/tqdm/tqdm" }
[#__codelineno-2-10](#__codelineno-2-10)
[#__codelineno-2-11](#__codelineno-2-11)[tool.uv.workspace]
[#__codelineno-2-12](#__codelineno-2-12)members = ["packages/*"]
[#__codelineno-2-13](#__codelineno-2-13)
[#__codelineno-2-14](#__codelineno-2-14)[build-system]
[#__codelineno-2-15](#__codelineno-2-15)requires = ["uv_build>=0.9.9,<0.10.0"]
[#__codelineno-2-16](#__codelineno-2-16)build-backend = "uv_build"

```

Every workspace member would, by default, install `tqdm `from GitHub, unless a specific member overrides the `tqdm `entry in its own `tool.uv.sources `table. 

!!! note "Note"

    If a workspace member provides `tool.uv.sources `for some dependency, it will ignore any `tool.uv.sources `for the same dependency in the workspace root, even if the member's source is limited by a [marker](../dependencies/#platform-specific-sources)that doesn't match the current platform. 

## [Workspace layouts](#workspace-layouts)

The most common workspace layout can be thought of as a root project with a series of accompanying libraries. 

For example, continuing with the above example, this workspace has an explicit root at `albatross `, with two libraries ( `bird-feeder `and `seeds `) in the `packages `directory: 

```
[#__codelineno-3-1](#__codelineno-3-1)albatross
[#__codelineno-3-2](#__codelineno-3-2)├── packages
[#__codelineno-3-3](#__codelineno-3-3)│   ├── bird-feeder
[#__codelineno-3-4](#__codelineno-3-4)│   │   ├── pyproject.toml
[#__codelineno-3-5](#__codelineno-3-5)│   │   └── src
[#__codelineno-3-6](#__codelineno-3-6)│   │       └── bird_feeder
[#__codelineno-3-7](#__codelineno-3-7)│   │           ├── __init__.py
[#__codelineno-3-8](#__codelineno-3-8)│   │           └── foo.py
[#__codelineno-3-9](#__codelineno-3-9)│   └── seeds
[#__codelineno-3-10](#__codelineno-3-10)│       ├── pyproject.toml
[#__codelineno-3-11](#__codelineno-3-11)│       └── src
[#__codelineno-3-12](#__codelineno-3-12)│           └── seeds
[#__codelineno-3-13](#__codelineno-3-13)│               ├── __init__.py
[#__codelineno-3-14](#__codelineno-3-14)│               └── bar.py
[#__codelineno-3-15](#__codelineno-3-15)├── pyproject.toml
[#__codelineno-3-16](#__codelineno-3-16)├── README.md
[#__codelineno-3-17](#__codelineno-3-17)├── uv.lock
[#__codelineno-3-18](#__codelineno-3-18)└── src
[#__codelineno-3-19](#__codelineno-3-19)    └── albatross
[#__codelineno-3-20](#__codelineno-3-20)        └── main.py

```

Since `seeds `was excluded in the `pyproject.toml `, the workspace has two members total: `albatross `(the root) and `bird-feeder `. 

## [When (not) to use workspaces](#when-not-to-use-workspaces)

Workspaces are intended to facilitate the development of multiple interconnected packages within a single repository. As a codebase grows in complexity, it can be helpful to split it into smaller, composable packages, each with their own dependencies and version constraints. 

Workspaces help enforce isolation and separation of concerns. For example, in uv, we have separate packages for the core library and the command-line interface, enabling us to test the core library independently of the CLI, and vice versa. 

Other common use cases for workspaces include: 

- A library with a performance-critical subroutine implemented in an extension module (Rust, C++, etc.). 

- A library with a plugin system, where each plugin is a separate workspace package with a dependency on the root. 

Workspaces are _not _suited for cases in which members have conflicting requirements, or desire a separate virtual environment for each member. In this case, path dependencies are often preferable. For example, rather than grouping `albatross `and its members in a workspace, you can always define each package as its own independent project, with inter-package dependencies defined as path dependencies in `tool.uv.sources `: 

pyproject.toml 

```
[#__codelineno-4-1](#__codelineno-4-1)[project]
[#__codelineno-4-2](#__codelineno-4-2)name = "albatross"
[#__codelineno-4-3](#__codelineno-4-3)version = "0.1.0"
[#__codelineno-4-4](#__codelineno-4-4)requires-python = ">=3.12"
[#__codelineno-4-5](#__codelineno-4-5)dependencies = ["bird-feeder", "tqdm>=4,<5"]
[#__codelineno-4-6](#__codelineno-4-6)
[#__codelineno-4-7](#__codelineno-4-7)[tool.uv.sources]
[#__codelineno-4-8](#__codelineno-4-8)bird-feeder = { path = "packages/bird-feeder" }
[#__codelineno-4-9](#__codelineno-4-9)
[#__codelineno-4-10](#__codelineno-4-10)[build-system]
[#__codelineno-4-11](#__codelineno-4-11)requires = ["uv_build>=0.9.9,<0.10.0"]
[#__codelineno-4-12](#__codelineno-4-12)build-backend = "uv_build"

```

This approach conveys many of the same benefits, but allows for more fine-grained control over dependency resolution and virtual environment management (with the downside that `uv run --package `is no longer available; instead, commands must be run from the relevant package directory). 

Finally, uv's workspaces enforce a single `requires-python `for the entire workspace, taking the intersection of all members' `requires-python `values. If you need to support testing a given member on a Python version that isn't supported by the rest of the workspace, you may need to use `uv pip `to install that member in a separate virtual environment. 

!!! note "Note"

    As Python does not provide dependency isolation, uv can't ensure that a package uses its declared dependencies and nothing else. For workspaces specifically, uv can't ensure that packages don't import dependencies declared by another workspace member. 

November 12, 2025
