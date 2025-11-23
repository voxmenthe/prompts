# [Configuring projects](#configuring-projects)

## [Python version requirement](#python-version-requirement)

Projects may declare the Python versions supported by the project in the `project.requires-python `field of the `pyproject.toml `. 

It is recommended to set a `requires-python `value: 

pyproject.toml 

```
[#__codelineno-0-1](#__codelineno-0-1)[project]
[#__codelineno-0-2](#__codelineno-0-2)name = "example"
[#__codelineno-0-3](#__codelineno-0-3)version = "0.1.0"
[#__codelineno-0-4](#__codelineno-0-4)requires-python = ">=3.12"

```

The Python version requirement determines the Python syntax that is allowed in the project and affects selection of dependency versions (they must support the same Python version range). 

## [Entry points](#entry-points)

[Entry points](https://packaging.python.org/en/latest/specifications/entry-points/#entry-points)are the official term for an installed package to advertise interfaces. These include: 

- [Command line interfaces](#command-line-interfaces)

- [Graphical user interfaces](#graphical-user-interfaces)

- [Plugin entry points](#plugin-entry-points)

!!! important "Important"

    Using the entry point tables requires a [build system](#build-systems)to be defined. 

### [Command-line interfaces](#command-line-interfaces)

Projects may define command line interfaces (CLIs) for the project in the `[project.scripts] `table of the `pyproject.toml `. 

For example, to declare a command called `hello `that invokes the `hello `function in the `example `module: 

pyproject.toml 

```
[#__codelineno-1-1](#__codelineno-1-1)[project.scripts]
[#__codelineno-1-2](#__codelineno-1-2)hello = "example:hello"

```

Then, the command can be run from a console: 

```
[#__codelineno-2-1](#__codelineno-2-1)$ uv run hello

```

### [Graphical user interfaces](#graphical-user-interfaces)

Projects may define graphical user interfaces (GUIs) for the project in the `[project.gui-scripts] `table of the `pyproject.toml `. 

!!! important "Important"

    These are only different from [command-line interfaces](#command-line-interfaces)on Windows, where they are wrapped by a GUI executable so they can be started without a console. On other platforms, they behave the same. 

For example, to declare a command called `hello `that invokes the `app `function in the `example `module: 

pyproject.toml 

```
[#__codelineno-3-1](#__codelineno-3-1)[project.gui-scripts]
[#__codelineno-3-2](#__codelineno-3-2)hello = "example:app"

```

### [Plugin entry points](#plugin-entry-points)

Projects may define entry points for plugin discovery in the [`[project.entry-points] `](https://packaging.python.org/en/latest/guides/creating-and-discovering-plugins/#using-package-metadata)table of the `pyproject.toml `. 

For example, to register the `example-plugin-a `package as a plugin for `example `: 

pyproject.toml 

```
[#__codelineno-4-1](#__codelineno-4-1)[project.entry-points.'example.plugins']
[#__codelineno-4-2](#__codelineno-4-2)a = "example_plugin_a"

```

Then, in `example `, plugins would be loaded with: 

example/__init__.py 

```
[#__codelineno-5-1](#__codelineno-5-1)from importlib.metadata import entry_points
[#__codelineno-5-2](#__codelineno-5-2)
[#__codelineno-5-3](#__codelineno-5-3)for plugin in entry_points(group='example.plugins'):
[#__codelineno-5-4](#__codelineno-5-4)    plugin.load()

```

!!! note "Note"

    The `group `key can be an arbitrary value, it does not need to include the package name or "plugins". However, it is recommended to namespace the key by the package name to avoid collisions with other packages. 

## [Build systems](#build-systems)

A build system determines how the project should be packaged and installed. Projects may declare and configure a build system in the `[build-system] `table of the `pyproject.toml `. 

uv uses the presence of a build system to determine if a project contains a package that should be installed in the project virtual environment. If a build system is not defined, uv will not attempt to build or install the project itself, just its dependencies. If a build system is defined, uv will build and install the project into the project environment. 

The `--build-backend `option can be provided to `uv init `to create a packaged project with an appropriate layout. The `--package `option can be provided to `uv init `to create a packaged project with the default build system. 

!!! note "Note"

    While uv will not build and install the current project without a build system definition, the presence of a `[build-system] `table is not required in other packages. For legacy reasons, if a build system is not defined, then `setuptools.build_meta:__legacy__ `is used to build the package. Packages you depend on may not explicitly declare their build system but are still installable. Similarly, if you [add a dependency on a local project](../dependencies/#path)or install it with `uv pip `, uv will attempt to build and install it regardless of the presence of a `[build-system] `table. 

Build systems are used to power the following features: 

- Including or excluding files from distributions 

- Editable installation behavior 

- Dynamic project metadata 

- Compilation of native code 

- Vendoring shared libraries 

To configure these features, refer to the documentation of your chosen build system. 

## [Project packaging](#project-packaging)

As discussed in [build systems](#build-systems), a Python project must be built to be installed. This process is generally referred to as "packaging". 

You probably need a package if you want to: 

- Add commands to the project 

- Distribute the project to others 

- Use a `src `and `test `layout 

- Write a library 

You probably _do not _need a package if you are: 

- Writing scripts 

- Building a simple application 

- Using a flat layout 

While uv usually uses the declaration of a [build system](#build-systems)to determine if a project should be packaged, uv also allows overriding this behavior with the [`tool.uv.package `](../../../reference/settings/#package)setting. 

Setting `tool.uv.package = true `will force a project to be built and installed into the project environment. If no build system is defined, uv will use the setuptools legacy backend. 

Setting `tool.uv.package = false `will force a project package _not _to be built and installed into the project environment. uv will ignore a declared build system when interacting with the project; however, uv will still respect explicit attempts to build the project such as invoking `uv build `. 

## [Project environment path](#project-environment-path)

The `UV_PROJECT_ENVIRONMENT `environment variable can be used to configure the project virtual environment path ( `.venv `by default). 

If a relative path is provided, it will be resolved relative to the workspace root. If an absolute path is provided, it will be used as-is, i.e., a child directory will not be created for the environment. If an environment is not present at the provided path, uv will create it. 

This option can be used to write to the system Python environment, though it is not recommended. `uv sync `will remove extraneous packages from the environment by default and, as such, may leave the system in a broken state. 

To target the system environment, set `UV_PROJECT_ENVIRONMENT `to the prefix of the Python installation. For example, on Debian-based systems, this is usually `/usr/local `: 

```
[#__codelineno-6-1](#__codelineno-6-1)$ python -c "import sysconfig; print(sysconfig.get_config_var('prefix'))"
[#__codelineno-6-2](#__codelineno-6-2)/usr/local

```

To target this environment, you'd export `UV_PROJECT_ENVIRONMENT=/usr/local `. 

!!! important "Important"

    If an absolute path is provided and the setting is used across multiple projects, the environment will be overwritten by invocations in each project. This setting is only recommended for use for a single project in CI or Docker images. 

!!! note "Note"

    By default, uv does not read the `VIRTUAL_ENV `environment variable during project operations. A warning will be displayed if `VIRTUAL_ENV `is set to a different path than the project's environment. The `--active `flag can be used to opt-in to respecting `VIRTUAL_ENV `. The `--no-active `flag can be used to silence the warning. 

## [Build isolation](#build-isolation)

By default, uv builds all packages in isolated virtual environments alongside their declared build dependencies, as per [PEP 517](https://peps.python.org/pep-0517/). 

Some packages are incompatible with this approach to build isolation, be it intentionally or unintentionally. 

For example, packages like [`flash-attn `](https://pypi.org/project/flash-attn/)and [`deepspeed `](https://pypi.org/project/deepspeed/)need to build against the same version of PyTorch that is installed in the project environment; by building them in an isolated environment, they may inadvertently build against a different version of PyTorch, leading to runtime errors. 

In other cases, packages may accidentally omit necessary dependencies in their declared build dependency list. For example, [`cchardet `](https://pypi.org/project/cchardet/)requires `cython `to be installed in the project environment prior to installing `cchardet `, but does not declare it as a build dependency. 

To address these issues, uv supports two separate approaches to modifying the build isolation behavior: 

1. 

**Augmenting the list of build dependencies **: This allows you to install a package in an isolated environment, but with additional build dependencies that are not declared by the package itself via the [`extra-build-dependencies `](../../../reference/settings/#extra-build-dependencies)setting. For packages like `flash-attn `, you can even enforce that those build dependencies (like `torch `) match the version of the package that is or will be installed in the project environment. 


2. 

**Disabling build isolation for specific packages **: This allows you to install a package without building it in an isolated environment. 



When possible, we recommend augmenting the build dependencies rather than disabling build isolation entirely, as the latter approach requires that the build dependencies are installed in the project environment _prior _to installing the package itself, which can lead to more complex installation steps, the inclusion of extraneous packages in the project environment, and difficulty in reproducing the project environment in other contexts. 

### [Augmenting build dependencies](#augmenting-build-dependencies)

To augment the list of build dependencies for a specific package, add it to the [`extra-build-dependencies `](../../../reference/settings/#extra-build-dependencies)list in your `pyproject.toml `. 

For example, to build `cchardet `with `cython `as an additional build dependency, include the following in your `pyproject.toml `: 

pyproject.toml 

```
[#__codelineno-7-1](#__codelineno-7-1)[project]
[#__codelineno-7-2](#__codelineno-7-2)name = "project"
[#__codelineno-7-3](#__codelineno-7-3)version = "0.1.0"
[#__codelineno-7-4](#__codelineno-7-4)description = "..."
[#__codelineno-7-5](#__codelineno-7-5)readme = "README.md"
[#__codelineno-7-6](#__codelineno-7-6)requires-python = ">=3.12"
[#__codelineno-7-7](#__codelineno-7-7)dependencies = ["cchardet"]
[#__codelineno-7-8](#__codelineno-7-8)
[#__codelineno-7-9](#__codelineno-7-9)[tool.uv.extra-build-dependencies]
[#__codelineno-7-10](#__codelineno-7-10)cchardet = ["cython"]

```

To ensure that a build dependency matches the version of the package that is or will be installed in the project environment, set `match-runtime = true `in the `extra-build-dependencies `table. For example, to build `deepspeed `with `torch `as an additional build dependency, include the following in your `pyproject.toml `: 

pyproject.toml 

```
[#__codelineno-8-1](#__codelineno-8-1)[project]
[#__codelineno-8-2](#__codelineno-8-2)name = "project"
[#__codelineno-8-3](#__codelineno-8-3)version = "0.1.0"
[#__codelineno-8-4](#__codelineno-8-4)description = "..."
[#__codelineno-8-5](#__codelineno-8-5)readme = "README.md"
[#__codelineno-8-6](#__codelineno-8-6)requires-python = ">=3.12"
[#__codelineno-8-7](#__codelineno-8-7)dependencies = ["deepspeed", "torch"]
[#__codelineno-8-8](#__codelineno-8-8)
[#__codelineno-8-9](#__codelineno-8-9)[tool.uv.extra-build-dependencies]
[#__codelineno-8-10](#__codelineno-8-10)deepspeed = [{ requirement = "torch", match-runtime = true }]

```

This will ensure that `deepspeed `is built with the same version of `torch `that is installed in the project environment. 

Similarly, to build `flash-attn `with `torch `as an additional build dependency, include the following in your `pyproject.toml `: 

pyproject.toml 

```
[#__codelineno-9-1](#__codelineno-9-1)[project]
[#__codelineno-9-2](#__codelineno-9-2)name = "project"
[#__codelineno-9-3](#__codelineno-9-3)version = "0.1.0"
[#__codelineno-9-4](#__codelineno-9-4)description = "..."
[#__codelineno-9-5](#__codelineno-9-5)readme = "README.md"
[#__codelineno-9-6](#__codelineno-9-6)requires-python = ">=3.12"
[#__codelineno-9-7](#__codelineno-9-7)dependencies = ["flash-attn", "torch"]
[#__codelineno-9-8](#__codelineno-9-8)
[#__codelineno-9-9](#__codelineno-9-9)[tool.uv.extra-build-dependencies]
[#__codelineno-9-10](#__codelineno-9-10)flash-attn = [{ requirement = "torch", match-runtime = true }]
[#__codelineno-9-11](#__codelineno-9-11)
[#__codelineno-9-12](#__codelineno-9-12)[tool.uv.extra-build-variables]
[#__codelineno-9-13](#__codelineno-9-13)flash-attn = { FLASH_ATTENTION_SKIP_CUDA_BUILD = "TRUE" }

```

!!! note "Note"

    The `FLASH_ATTENTION_SKIP_CUDA_BUILD `environment variable ensures that `flash-attn `is installed from a compatible, pre-built wheel, rather than attempting to build it from source, which requires access to the CUDA development toolkit. If the CUDA toolkit is not available, the environment variable can be omitted, and `flash-attn `will be installed from a pre-built wheel if one is available for the current platform, Python version, and PyTorch version. 

Similarly, [`deep_gemm `](https://github.com/deepseek-ai/DeepGEMM)follows the same pattern: 

pyproject.toml 

```
[#__codelineno-10-1](#__codelineno-10-1)[project]
[#__codelineno-10-2](#__codelineno-10-2)name = "project"
[#__codelineno-10-3](#__codelineno-10-3)version = "0.1.0"
[#__codelineno-10-4](#__codelineno-10-4)description = "..."
[#__codelineno-10-5](#__codelineno-10-5)readme = "README.md"
[#__codelineno-10-6](#__codelineno-10-6)requires-python = ">=3.12"
[#__codelineno-10-7](#__codelineno-10-7)dependencies = ["deep_gemm", "torch"]
[#__codelineno-10-8](#__codelineno-10-8)
[#__codelineno-10-9](#__codelineno-10-9)[tool.uv.sources]
[#__codelineno-10-10](#__codelineno-10-10)deep_gemm = { git = "https://github.com/deepseek-ai/DeepGEMM" }
[#__codelineno-10-11](#__codelineno-10-11)
[#__codelineno-10-12](#__codelineno-10-12)[tool.uv.extra-build-dependencies]
[#__codelineno-10-13](#__codelineno-10-13)deep_gemm = [{ requirement = "torch", match-runtime = true }]

```

The use of `extra-build-dependencies `and `extra-build-variables `are tracked in the uv cache, such that changes to these settings will trigger a reinstall and rebuild of the affected packages. For example, in the case of `flash-attn `, upgrading the version of `torch `used in your project would subsequently trigger a rebuild of `flash-attn `with the new version of `torch `. 

#### [Dynamic metadata](#dynamic-metadata)

The use of `match-runtime = true `is only available for packages like `flash-attn `that declare static metadata. If static metadata is unavailable, uv is required to build the package during the dependency resolution phase; as such, uv cannot determine the version of the build dependency that would ultimately be installed in the project environment. 

In other words, if `flash-attn `did not declare static metadata, uv would not be able to determine the version of `torch `that would be installed in the project environment, since it would need to build `flash-attn `prior to resolving the `torch `version. 

As a concrete example, [`axolotl `](https://pypi.org/project/axolotl/)is a popular package that requires augmented build dependencies, but does not declare static metadata, as the package's dependencies vary based on the version of `torch `that is installed in the project environment. In this case, users should instead specify the exact version of `torch `that they intend to use in their project, and then augment the build dependencies with that version. 

For example, to build `axolotl `against `torch==2.6.0 `, include the following in your `pyproject.toml `: 

pyproject.toml 

```
[#__codelineno-11-1](#__codelineno-11-1)[project]
[#__codelineno-11-2](#__codelineno-11-2)name = "project"
[#__codelineno-11-3](#__codelineno-11-3)version = "0.1.0"
[#__codelineno-11-4](#__codelineno-11-4)description = "..."
[#__codelineno-11-5](#__codelineno-11-5)readme = "README.md"
[#__codelineno-11-6](#__codelineno-11-6)requires-python = ">=3.12"
[#__codelineno-11-7](#__codelineno-11-7)dependencies = ["axolotl[deepspeed, flash-attn]", "torch==2.6.0"]
[#__codelineno-11-8](#__codelineno-11-8)
[#__codelineno-11-9](#__codelineno-11-9)[tool.uv.extra-build-dependencies]
[#__codelineno-11-10](#__codelineno-11-10)axolotl = ["torch==2.6.0"]
[#__codelineno-11-11](#__codelineno-11-11)deepspeed = ["torch==2.6.0"]
[#__codelineno-11-12](#__codelineno-11-12)flash-attn = ["torch==2.6.0"]

```

Similarly, older versions of `flash-attn `did not declare static metadata, and thus would not have supported `match-runtime = true `out of the box. Unlike `axolotl `, though, `flash-attn `did not vary its dependencies based on dynamic properties of the build environment. As such, users could instead provide the `flash-attn `metadata upfront via the [`dependency-metadata `](../../../reference/settings/#dependency-metadata)setting, thereby forgoing the need to build the package during the dependency resolution phase. For example, to provide the `flash-attn `metadata upfront: 

pyproject.toml 

```
[#__codelineno-12-1](#__codelineno-12-1)[[tool.uv.dependency-metadata]]
[#__codelineno-12-2](#__codelineno-12-2)name = "flash-attn"
[#__codelineno-12-3](#__codelineno-12-3)version = "2.6.3"
[#__codelineno-12-4](#__codelineno-12-4)requires-dist = ["torch", "einops"]

```

!!! tip "Tip"

    To determine the package metadata for a package like `flash-attn `, navigate to the appropriate Git repository, or look it up on [PyPI](https://pypi.org/project/flash-attn)and download the package's source distribution. The package requirements can typically be found in the `setup.py `or `setup.cfg `file. 

    (If the package includes a built distribution, you can unzip it to find the `METADATA `file; however, the presence of a built distribution would negate the need to provide the metadata upfront, since it would already be available to uv.) 

    The `version `field in `tool.uv.dependency-metadata `is optional for registry-based dependencies (when omitted, uv will assume the metadata applies to all versions of the package), but _required _for direct URL dependencies (like Git dependencies). 

### [Disabling build isolation](#disabling-build-isolation)

Installing packages without build isolation requires that the package's build dependencies are installed in the project environment _prior _to building the package itself. 

For example, historically, to install `cchardet `without build isolation, you would first need to install the `cython `and `setuptools `packages in the project environment, followed by a separate invocation to install `cchardet `without build isolation: 

```
[#__codelineno-13-1](#__codelineno-13-1)$ uv venv
[#__codelineno-13-2](#__codelineno-13-2)$ uv pip install cython setuptools
[#__codelineno-13-3](#__codelineno-13-3)$ uv pip install cchardet --no-build-isolation

```

uv simplifies this process by allowing you to specify packages that should not be built in isolation via the `no-build-isolation-package `setting in your `pyproject.toml `and the `--no-build-isolation-package `flag in the command line. Further, when a package is marked for disabling build isolation, uv will perform a two-phase install, first installing any packages that support build isolation, followed by those that do not. As a result, if a project's build dependencies are included as project dependencies, uv will automatically install them before installing the package that requires build isolation to be disabled. 

For example, to install `cchardet `without build isolation, include the following in your `pyproject.toml `: 

pyproject.toml 

```
[#__codelineno-14-1](#__codelineno-14-1)[project]
[#__codelineno-14-2](#__codelineno-14-2)name = "project"
[#__codelineno-14-3](#__codelineno-14-3)version = "0.1.0"
[#__codelineno-14-4](#__codelineno-14-4)description = "..."
[#__codelineno-14-5](#__codelineno-14-5)readme = "README.md"
[#__codelineno-14-6](#__codelineno-14-6)requires-python = ">=3.12"
[#__codelineno-14-7](#__codelineno-14-7)dependencies = ["cchardet", "cython", "setuptools"]
[#__codelineno-14-8](#__codelineno-14-8)
[#__codelineno-14-9](#__codelineno-14-9)[tool.uv]
[#__codelineno-14-10](#__codelineno-14-10)no-build-isolation-package = ["cchardet"]

```

When running `uv sync `, uv will first install `cython `and `setuptools `in the project environment, followed by `cchardet `(without build isolation): 

```
[#__codelineno-15-1](#__codelineno-15-1)$ uv sync --extra build
[#__codelineno-15-2](#__codelineno-15-2) + cchardet==2.1.7
[#__codelineno-15-3](#__codelineno-15-3) + cython==3.1.3
[#__codelineno-15-4](#__codelineno-15-4) + setuptools==80.9.0

```

Similarly, to install `flash-attn `without build isolation, include the following in your `pyproject.toml `: 

pyproject.toml 

```
[#__codelineno-16-1](#__codelineno-16-1)[project]
[#__codelineno-16-2](#__codelineno-16-2)name = "project"
[#__codelineno-16-3](#__codelineno-16-3)version = "0.1.0"
[#__codelineno-16-4](#__codelineno-16-4)description = "..."
[#__codelineno-16-5](#__codelineno-16-5)readme = "README.md"
[#__codelineno-16-6](#__codelineno-16-6)requires-python = ">=3.12"
[#__codelineno-16-7](#__codelineno-16-7)dependencies = ["flash-attn", "torch"]
[#__codelineno-16-8](#__codelineno-16-8)
[#__codelineno-16-9](#__codelineno-16-9)[tool.uv]
[#__codelineno-16-10](#__codelineno-16-10)no-build-isolation-package = ["flash-attn"]

```

When running `uv sync `, uv will first install `torch `in the project environment, followed by `flash-attn `(without build isolation). As `torch `is both a project dependency and a build dependency, the version of `torch `is guaranteed to be consistent between the build and runtime environments. 

A downside of the above approach is that it requires the build dependencies to be installed in the project environment, which is appropriate for `flash-attn `(which requires `torch `both at build-time and runtime), but not for `cchardet `(which only requires `cython `at build-time). 

To avoid including build dependencies in the project environment, uv supports a two-step installation process that allows you to separate the build dependencies from the packages that require them. 

For example, the build dependencies for `cchardet `can be isolated to an optional `build `group, as in: 

pyproject.toml 

```
[#__codelineno-17-1](#__codelineno-17-1)[project]
[#__codelineno-17-2](#__codelineno-17-2)name = "project"
[#__codelineno-17-3](#__codelineno-17-3)version = "0.1.0"
[#__codelineno-17-4](#__codelineno-17-4)description = "..."
[#__codelineno-17-5](#__codelineno-17-5)readme = "README.md"
[#__codelineno-17-6](#__codelineno-17-6)requires-python = ">=3.12"
[#__codelineno-17-7](#__codelineno-17-7)dependencies = ["cchardet"]
[#__codelineno-17-8](#__codelineno-17-8)
[#__codelineno-17-9](#__codelineno-17-9)[project.optional-dependencies]
[#__codelineno-17-10](#__codelineno-17-10)build = ["setuptools", "cython"]
[#__codelineno-17-11](#__codelineno-17-11)
[#__codelineno-17-12](#__codelineno-17-12)[tool.uv]
[#__codelineno-17-13](#__codelineno-17-13)no-build-isolation-package = ["cchardet"]

```

Given the above, a user would first sync with the `build `optional group, and then without it to remove the build dependencies: 

```
[#__codelineno-18-1](#__codelineno-18-1)$ uv sync --extra build
[#__codelineno-18-2](#__codelineno-18-2) + cchardet==2.1.7
[#__codelineno-18-3](#__codelineno-18-3) + cython==3.1.3
[#__codelineno-18-4](#__codelineno-18-4) + setuptools==80.9.0
[#__codelineno-18-5](#__codelineno-18-5)$ uv sync
[#__codelineno-18-6](#__codelineno-18-6) - cython==3.1.3
[#__codelineno-18-7](#__codelineno-18-7) - setuptools==80.9.0

```

Some packages, like `cchardet `, only require build dependencies for the _installation _phase of `uv sync `. Others require their build dependencies to be present even just to resolve the project's dependencies during the _resolution _phase. 

In such cases, the build dependencies can be installed prior to running any `uv lock `or `uv sync `commands, using the lower lower-level `uv pip `API. For example, given: 

pyproject.toml 

```
[#__codelineno-19-1](#__codelineno-19-1)[project]
[#__codelineno-19-2](#__codelineno-19-2)name = "project"
[#__codelineno-19-3](#__codelineno-19-3)version = "0.1.0"
[#__codelineno-19-4](#__codelineno-19-4)description = "..."
[#__codelineno-19-5](#__codelineno-19-5)readme = "README.md"
[#__codelineno-19-6](#__codelineno-19-6)requires-python = ">=3.12"
[#__codelineno-19-7](#__codelineno-19-7)dependencies = ["flash-attn"]
[#__codelineno-19-8](#__codelineno-19-8)
[#__codelineno-19-9](#__codelineno-19-9)[tool.uv]
[#__codelineno-19-10](#__codelineno-19-10)no-build-isolation-package = ["flash-attn"]

```

You could run the following sequence of commands to sync `flash-attn `: 

```
[#__codelineno-20-1](#__codelineno-20-1)$ uv venv
[#__codelineno-20-2](#__codelineno-20-2)$ uv pip install torch setuptools
[#__codelineno-20-3](#__codelineno-20-3)$ uv sync

```

Alternatively, users can instead provide the `flash-attn `metadata upfront via the [`dependency-metadata `](../../../reference/settings/#dependency-metadata)setting, thereby forgoing the need to build the package during the dependency resolution phase. For example, to provide the `flash-attn `metadata upfront: 

pyproject.toml 

```
[#__codelineno-21-1](#__codelineno-21-1)[[tool.uv.dependency-metadata]]
[#__codelineno-21-2](#__codelineno-21-2)name = "flash-attn"
[#__codelineno-21-3](#__codelineno-21-3)version = "2.6.3"
[#__codelineno-21-4](#__codelineno-21-4)requires-dist = ["torch", "einops"]

```

## [Editable mode](#editable-mode)

By default, the project will be installed in editable mode, such that changes to the source code are immediately reflected in the environment. `uv sync `and `uv run `both accept a `--no-editable `flag, which instructs uv to install the project in non-editable mode. `--no-editable `is intended for deployment use-cases, such as building a Docker container, in which the project should be included in the deployed environment without a dependency on the originating source code. 

## [Conflicting dependencies](#conflicting-dependencies)

uv resolves all project dependencies together, including optional dependencies ("extras") and dependency groups. If dependencies declared in one section are not compatible with those in another section, uv will fail to resolve the requirements of the project with an error. 

uv supports explicit declaration of conflicting dependency groups. For example, to declare that the `optional-dependency `groups `extra1 `and `extra2 `are incompatible: 

pyproject.toml 

```
[#__codelineno-22-1](#__codelineno-22-1)[tool.uv]
[#__codelineno-22-2](#__codelineno-22-2)conflicts = [
[#__codelineno-22-3](#__codelineno-22-3)    [
[#__codelineno-22-4](#__codelineno-22-4)      { extra = "extra1" },
[#__codelineno-22-5](#__codelineno-22-5)      { extra = "extra2" },
[#__codelineno-22-6](#__codelineno-22-6)    ],
[#__codelineno-22-7](#__codelineno-22-7)]

```

Or, to declare the development dependency groups `group1 `and `group2 `incompatible: 

pyproject.toml 

```
[#__codelineno-23-1](#__codelineno-23-1)[tool.uv]
[#__codelineno-23-2](#__codelineno-23-2)conflicts = [
[#__codelineno-23-3](#__codelineno-23-3)    [
[#__codelineno-23-4](#__codelineno-23-4)      { group = "group1" },
[#__codelineno-23-5](#__codelineno-23-5)      { group = "group2" },
[#__codelineno-23-6](#__codelineno-23-6)    ],
[#__codelineno-23-7](#__codelineno-23-7)]

```

See the [resolution documentation](../../resolution/#conflicting-dependencies)for more. 

## [Limited resolution environments](#limited-resolution-environments)

If your project supports a more limited set of platforms or Python versions, you can constrain the set of solved platforms via the `environments `setting, which accepts a list of PEP 508 environment markers. For example, to constrain the lockfile to macOS and Linux, and exclude Windows: 

pyproject.toml 

```
[#__codelineno-24-1](#__codelineno-24-1)[tool.uv]
[#__codelineno-24-2](#__codelineno-24-2)environments = [
[#__codelineno-24-3](#__codelineno-24-3)    "sys_platform == 'darwin'",
[#__codelineno-24-4](#__codelineno-24-4)    "sys_platform == 'linux'",
[#__codelineno-24-5](#__codelineno-24-5)]

```

See the [resolution documentation](../../resolution/#limited-resolution-environments)for more. 

## [Required environments](#required-environments)

If your project _must _support a specific platform or Python version, you can mark that platform as required via the `required-environments `setting. For example, to require that the project supports Intel macOS: 

pyproject.toml 

```
[#__codelineno-25-1](#__codelineno-25-1)[tool.uv]
[#__codelineno-25-2](#__codelineno-25-2)required-environments = [
[#__codelineno-25-3](#__codelineno-25-3)    "sys_platform == 'darwin' and platform_machine == 'x86_64'",
[#__codelineno-25-4](#__codelineno-25-4)]

```

The `required-environments `setting is only relevant for packages that do not publish a source distribution (like PyTorch), as such packages can _only _be installed on environments covered by the set of pre-built binary distributions (wheels) published by that package. 

See the [resolution documentation](../../resolution/#required-environments)for more. 

August 27, 2025
