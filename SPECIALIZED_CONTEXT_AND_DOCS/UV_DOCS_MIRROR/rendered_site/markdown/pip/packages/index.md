# [Managing packages](#managing-packages)

## [Installing a package](#installing-a-package)

To install a package into the virtual environment, e.g., Flask: 

```
[#__codelineno-0-1](#__codelineno-0-1)$ uv pip install flask

```

To install a package with optional dependencies enabled, e.g., Flask with the "dotenv" extra: 

```
[#__codelineno-1-1](#__codelineno-1-1)$ uv pip install "flask[dotenv]"

```

To install multiple packages, e.g., Flask and Ruff: 

```
[#__codelineno-2-1](#__codelineno-2-1)$ uv pip install flask ruff

```

To install a package with a constraint, e.g., Ruff v0.2.0 or newer: 

```
[#__codelineno-3-1](#__codelineno-3-1)$ uv pip install 'ruff>=0.2.0'

```

To install a package at a specific version, e.g., Ruff v0.3.0: 

```
[#__codelineno-4-1](#__codelineno-4-1)$ uv pip install 'ruff==0.3.0'

```

To install a package from the disk: 

```
[#__codelineno-5-1](#__codelineno-5-1)$ uv pip install "ruff @ ./projects/ruff"

```

To install a package from GitHub: 

```
[#__codelineno-6-1](#__codelineno-6-1)$ uv pip install "git+https://github.com/astral-sh/ruff"

```

To install a package from GitHub at a specific reference: 

```
[#__codelineno-7-1](#__codelineno-7-1)$ # Install a tag
[#__codelineno-7-2](#__codelineno-7-2)$ uv pip install "git+https://github.com/astral-sh/[[email protected]](/cdn-cgi/l/email-protection)"
[#__codelineno-7-3](#__codelineno-7-3)
[#__codelineno-7-4](#__codelineno-7-4)$ # Install a commit
[#__codelineno-7-5](#__codelineno-7-5)$ uv pip install "git+https://github.com/astral-sh/ruff@1fadefa67b26508cc59cf38e6130bde2243c929d"
[#__codelineno-7-6](#__codelineno-7-6)
[#__codelineno-7-7](#__codelineno-7-7)$ # Install a branch
[#__codelineno-7-8](#__codelineno-7-8)$ uv pip install "git+https://github.com/astral-sh/ruff@main"

```

See the [Git authentication](../../concepts/authentication/git/)documentation for installation from a private repository. 

## [Editable packages](#editable-packages)

Editable packages do not need to be reinstalled for changes to their source code to be active. 

To install the current project as an editable package 

```
[#__codelineno-8-1](#__codelineno-8-1)$ uv pip install -e .

```

To install a project in another directory as an editable package: 

```
[#__codelineno-9-1](#__codelineno-9-1)$ uv pip install -e "ruff @ ./project/ruff"

```

## [Installing packages from files](#installing-packages-from-files)

Multiple packages can be installed at once from standard file formats. 

Install from a `requirements.txt `file: 

```
[#__codelineno-10-1](#__codelineno-10-1)$ uv pip install -r requirements.txt

```

See the [`uv pip compile `](../compile/)documentation for more information on `requirements.txt `files. 

Install from a `pyproject.toml `file: 

```
[#__codelineno-11-1](#__codelineno-11-1)$ uv pip install -r pyproject.toml

```

Install from a `pyproject.toml `file with optional dependencies enabled, e.g., the "foo" extra: 

```
[#__codelineno-12-1](#__codelineno-12-1)$ uv pip install -r pyproject.toml --extra foo

```

Install from a `pyproject.toml `file with all optional dependencies enabled: 

```
[#__codelineno-13-1](#__codelineno-13-1)$ uv pip install -r pyproject.toml --all-extras

```

To install dependency groups in the current project directory's `pyproject.toml `, for example the group `foo `: 

```
[#__codelineno-14-1](#__codelineno-14-1)$ uv pip install --group foo

```

To specify the project directory where groups should be sourced from: 

```
[#__codelineno-15-1](#__codelineno-15-1)$ uv pip install --project some/path/ --group foo --group bar

```

Alternatively, you can specify a path to a `pyproject.toml `for each group: 

```
[#__codelineno-16-1](#__codelineno-16-1)$ uv pip install --group some/path/pyproject.toml:foo --group other/pyproject.toml:bar

```

!!! note "Note"

    As in pip, `--group `flags do not apply to other sources specified with flags like `-r `or `-e `. For instance, `uv pip install -r some/path/pyproject.toml --group foo `sources `foo `from `./pyproject.toml `and **not **`some/path/pyproject.toml `. 

## [Uninstalling a package](#uninstalling-a-package)

To uninstall a package, e.g., Flask: 

```
[#__codelineno-17-1](#__codelineno-17-1)$ uv pip uninstall flask

```

To uninstall multiple packages, e.g., Flask and Ruff: 

```
[#__codelineno-18-1](#__codelineno-18-1)$ uv pip uninstall flask ruff

```

August 28, 2025
