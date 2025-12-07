# [Using tools](#using-tools)

Many Python packages provide applications that can be used as tools. uv has specialized support for easily invoking and installing tools. 

## [Running tools](#running-tools)

The `uvx `command invokes a tool without installing it. 

For example, to run `ruff `: 

```
[#__codelineno-0-1](#__codelineno-0-1)$ uvx ruff

```

!!! note "Note"

    This is exactly equivalent to: 

    ```
[#__codelineno-1-1](#__codelineno-1-1)$ uv tool run ruff

```

    `uvx `is provided as an alias for convenience. 

Arguments can be provided after the tool name: 

```
[#__codelineno-2-1](#__codelineno-2-1)$ uvx pycowsay hello from uv
[#__codelineno-2-2](#__codelineno-2-2)
[#__codelineno-2-3](#__codelineno-2-3)  -------------
[#__codelineno-2-4](#__codelineno-2-4)< hello from uv >
[#__codelineno-2-5](#__codelineno-2-5)  -------------
[#__codelineno-2-6](#__codelineno-2-6)   \   ^__^
[#__codelineno-2-7](#__codelineno-2-7)    \  (oo)\_______
[#__codelineno-2-8](#__codelineno-2-8)       (__)\       )\/\
[#__codelineno-2-9](#__codelineno-2-9)           ||----w |
[#__codelineno-2-10](#__codelineno-2-10)           ||     ||

```

Tools are installed into temporary, isolated environments when using `uvx `. 

!!! note "Note"

    If you are running a tool in a [_project _](../../concepts/projects/)and the tool requires that your project is installed, e.g., when using `pytest `or `mypy `, you'll want to use [`uv run `](../projects/#running-commands)instead of `uvx `. Otherwise, the tool will be run in a virtual environment that is isolated from your project. 

    If your project has a flat structure, e.g., instead of using a `src `directory for modules, the project itself does not need to be installed and `uvx `is fine. In this case, using `uv run `is only beneficial if you want to pin the version of the tool in the project's dependencies. 

## [Commands with different package names](#commands-with-different-package-names)

When `uvx ruff `is invoked, uv installs the `ruff `package which provides the `ruff `command. However, sometimes the package and command names differ. 

The `--from `option can be used to invoke a command from a specific package, e.g., `http `which is provided by `httpie `: 

```
[#__codelineno-3-1](#__codelineno-3-1)$ uvx --from httpie http

```

## [Requesting specific versions](#requesting-specific-versions)

To run a tool at a specific version, use `command@ `: 

```
[#__codelineno-4-1](#__codelineno-4-1)$ uvx [[email protected]](/cdn-cgi/l/email-protection) check

```

To run a tool at the latest version, use `command@latest `: 

```
[#__codelineno-5-1](#__codelineno-5-1)$ uvx ruff@latest check

```

The `--from `option can also be used to specify package versions, as above: 

```
[#__codelineno-6-1](#__codelineno-6-1)$ uvx --from 'ruff==0.3.0' ruff check

```

Or, to constrain to a range of versions: 

```
[#__codelineno-7-1](#__codelineno-7-1)$ uvx --from 'ruff>0.2.0,<0.3.0' ruff check

```

Note the `@ `syntax cannot be used for anything other than an exact version. 

## [Requesting extras](#requesting-extras)

The `--from `option can be used to run a tool with extras: 

```
[#__codelineno-8-1](#__codelineno-8-1)$ uvx --from 'mypy[faster-cache,reports]' mypy --xml-report mypy_report

```

This can also be combined with version selection: 

```
[#__codelineno-9-1](#__codelineno-9-1)$ uvx --from 'mypy[faster-cache,reports]==1.13.0' mypy --xml-report mypy_report

```

## [Requesting different sources](#requesting-different-sources)

The `--from `option can also be used to install from alternative sources. 

For example, to pull from git: 

```
[#__codelineno-10-1](#__codelineno-10-1)$ uvx --from git+https://github.com/httpie/cli httpie

```

You can also pull the latest commit from a specific named branch: 

```
[#__codelineno-11-1](#__codelineno-11-1)$ uvx --from git+https://github.com/httpie/cli@master httpie

```

Or pull a specific tag: 

```
[#__codelineno-12-1](#__codelineno-12-1)$ uvx --from git+https://github.com/httpie/[[email protected]](/cdn-cgi/l/email-protection) httpie

```

Or even a specific commit: 

```
[#__codelineno-13-1](#__codelineno-13-1)$ uvx --from git+https://github.com/httpie/cli@2843b87 httpie

```

Or with [Git LFS](https://git-lfs.com)support: 

```
[#__codelineno-14-1](#__codelineno-14-1)$ uvx --lfs --from git+https://github.com/astral-sh/lfs-cowsay lfs-cowsay

```

## [Commands with plugins](#commands-with-plugins)

Additional dependencies can be included, e.g., to include `mkdocs-material `when running `mkdocs `: 

```
[#__codelineno-15-1](#__codelineno-15-1)$ uvx --with mkdocs-material mkdocs --help

```

## [Installing tools](#installing-tools)

If a tool is used often, it is useful to install it to a persistent environment and add it to the `PATH `instead of invoking `uvx `repeatedly. 

!!! tip "Tip"

    `uvx `is a convenient alias for `uv tool run `. All of the other commands for interacting with tools require the full `uv tool `prefix. 

To install `ruff `: 

```
[#__codelineno-16-1](#__codelineno-16-1)$ uv tool install ruff

```

When a tool is installed, its executables are placed in a `bin `directory in the `PATH `which allows the tool to be run without uv. If it's not on the `PATH `, a warning will be displayed and `uv tool update-shell `can be used to add it to the `PATH `. 

After installing `ruff `, it should be available: 

```
[#__codelineno-17-1](#__codelineno-17-1)$ ruff --version

```

Unlike `uv pip install `, installing a tool does not make its modules available in the current environment. For example, the following command will fail: 

```
[#__codelineno-18-1](#__codelineno-18-1)$ python -c "import ruff"

```

This isolation is important for reducing interactions and conflicts between dependencies of tools, scripts, and projects. 

Unlike `uvx `, `uv tool install `operates on a _package _and will install all executables provided by the tool. 

For example, the following will install the `http `, `https `, and `httpie `executables: 

```
[#__codelineno-19-1](#__codelineno-19-1)$ uv tool install httpie

```

Additionally, package versions can be included without `--from `: 

```
[#__codelineno-20-1](#__codelineno-20-1)$ uv tool install 'httpie>0.1.0'

```

And, similarly, for package sources: 

```
[#__codelineno-21-1](#__codelineno-21-1)$ uv tool install git+https://github.com/httpie/cli

```

Or package sources with [Git LFS](https://git-lfs.com): 

```
[#__codelineno-22-1](#__codelineno-22-1)$ uv tool install --lfs git+https://github.com/astral-sh/lfs-cowsay

```

As with `uvx `, installations can include additional packages: 

```
[#__codelineno-23-1](#__codelineno-23-1)$ uv tool install mkdocs --with mkdocs-material

```

Multiple related executables can be installed together in the same tool environment, using the `--with-executables-from `flag. For example, the following will install the executables from `ansible `, plus those ones provided by `ansible-core `and `ansible-lint `: 

```
[#__codelineno-24-1](#__codelineno-24-1)$ uv tool install --with-executables-from ansible-core,ansible-lint ansible

```

## [Upgrading tools](#upgrading-tools)

To upgrade a tool, use `uv tool upgrade `: 

```
[#__codelineno-25-1](#__codelineno-25-1)$ uv tool upgrade ruff

```

Tool upgrades will respect the version constraints provided when installing the tool. For example, `uv tool install ruff >=0.3, < 0.4 `followed by `uv tool upgrade ruff `will upgrade Ruff to the latest version in the range `>=0.3, < 0.4 `. 

To instead replace the version constraints, re-install the tool with `uv tool install `: 

```
[#__codelineno-26-1](#__codelineno-26-1)$ uv tool install ruff>=0.4

```

To instead upgrade all tools: 

```
[#__codelineno-27-1](#__codelineno-27-1)$ uv tool upgrade --all

```

## [Requesting Python versions](#requesting-python-versions)

By default, uv will use your default Python interpreter (the first it finds) when running, installing, or upgrading tools. You can specify the Python interpreter to use with the `--python `option. 

For example, to request a specific Python version when running a tool: 

```
[#__codelineno-28-1](#__codelineno-28-1)$ uvx --python 3.10 ruff

```

Or, when installing a tool: 

```
[#__codelineno-29-1](#__codelineno-29-1)$ uv tool install --python 3.10 ruff

```

Or, when upgrading a tool: 

```
[#__codelineno-30-1](#__codelineno-30-1)$ uv tool upgrade --python 3.10 ruff

```

For more details on requesting Python versions, see the [Python version](../../concepts/python-versions/#requesting-a-version)concept page. 

## [Legacy Windows Scripts](#legacy-windows-scripts)

Tools also support running [legacy setuptools scripts](https://packaging.python.org/en/latest/guides/distributing-packages-using-setuptools/#scripts). These scripts are available via `$(uv tool dir)\ \Scripts `when installed. 

Currently only legacy scripts with the `.ps1 `, `.cmd `, and `.bat `extensions are supported. 

For example, below is an example running a Command Prompt script. 

```
[#__codelineno-31-1](#__codelineno-31-1)$ uv tool run --from nuitka==2.6.7 nuitka.cmd --version

```

In addition, you don't need to specify the extension. `uvx `will automatically look for files ending in `.ps1 `, `.cmd `, and `.bat `in that order of execution on your behalf. 

```
[#__codelineno-32-1](#__codelineno-32-1)$ uv tool run --from nuitka==2.6.7 nuitka --version

```

## [Next steps](#next-steps)

To learn more about managing tools with uv, see the [Tools concept](../../concepts/tools/)page and the [command reference](../../reference/cli/#uv-tool). 

Or, read on to learn how to [work on projects](../projects/). 

December 2, 2025
