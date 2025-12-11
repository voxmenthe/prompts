# [Using uv in GitHub Actions](#using-uv-in-github-actions)

## [Installation](#installation)

For use with GitHub Actions, we recommend the official [`astral-sh/setup-uv `](https://github.com/astral-sh/setup-uv)action, which installs uv, adds it to PATH, (optionally) persists the cache, and more, with support for all uv-supported platforms. 

To install the latest version of uv: 

example.yml 

```
[#__codelineno-0-1](#__codelineno-0-1)name: Example
[#__codelineno-0-2](#__codelineno-0-2)
[#__codelineno-0-3](#__codelineno-0-3)jobs:
[#__codelineno-0-4](#__codelineno-0-4)  uv-example:
[#__codelineno-0-5](#__codelineno-0-5)    name: python
[#__codelineno-0-6](#__codelineno-0-6)    runs-on: ubuntu-latest
[#__codelineno-0-7](#__codelineno-0-7)
[#__codelineno-0-8](#__codelineno-0-8)    steps:
[#__codelineno-0-9](#__codelineno-0-9)      - uses: actions/checkout@v5
[#__codelineno-0-10](#__codelineno-0-10)
[#__codelineno-0-11](#__codelineno-0-11)      - name: Install uv
[#__codelineno-0-12](#__codelineno-0-12)        uses: astral-sh/setup-uv@v7

```

It is considered best practice to pin to a specific uv version, e.g., with: 

example.yml 

```
[#__codelineno-1-1](#__codelineno-1-1)name: Example
[#__codelineno-1-2](#__codelineno-1-2)
[#__codelineno-1-3](#__codelineno-1-3)jobs:
[#__codelineno-1-4](#__codelineno-1-4)  uv-example:
[#__codelineno-1-5](#__codelineno-1-5)    name: python
[#__codelineno-1-6](#__codelineno-1-6)    runs-on: ubuntu-latest
[#__codelineno-1-7](#__codelineno-1-7)
[#__codelineno-1-8](#__codelineno-1-8)    steps:
[#__codelineno-1-9](#__codelineno-1-9)      - uses: actions/checkout@v5
[#__codelineno-1-10](#__codelineno-1-10)
[#__codelineno-1-11](#__codelineno-1-11)      - name: Install uv
[#__codelineno-1-12](#__codelineno-1-12)        uses: astral-sh/setup-uv@v7
[#__codelineno-1-13](#__codelineno-1-13)        with:
[#__codelineno-1-14](#__codelineno-1-14)          # Install a specific version of uv.
[#__codelineno-1-15](#__codelineno-1-15)          version: "0.9.17"

```

## [Setting up Python](#setting-up-python)

Python can be installed with the `python install `command: 

example.yml 

```
[#__codelineno-2-1](#__codelineno-2-1)name: Example
[#__codelineno-2-2](#__codelineno-2-2)
[#__codelineno-2-3](#__codelineno-2-3)jobs:
[#__codelineno-2-4](#__codelineno-2-4)  uv-example:
[#__codelineno-2-5](#__codelineno-2-5)    name: python
[#__codelineno-2-6](#__codelineno-2-6)    runs-on: ubuntu-latest
[#__codelineno-2-7](#__codelineno-2-7)
[#__codelineno-2-8](#__codelineno-2-8)    steps:
[#__codelineno-2-9](#__codelineno-2-9)      - uses: actions/checkout@v5
[#__codelineno-2-10](#__codelineno-2-10)
[#__codelineno-2-11](#__codelineno-2-11)      - name: Install uv
[#__codelineno-2-12](#__codelineno-2-12)        uses: astral-sh/setup-uv@v7
[#__codelineno-2-13](#__codelineno-2-13)
[#__codelineno-2-14](#__codelineno-2-14)      - name: Set up Python
[#__codelineno-2-15](#__codelineno-2-15)        run: uv python install

```

This will respect the Python version pinned in the project. 

Alternatively, the official GitHub `setup-python `action can be used. This can be faster, because GitHub caches the Python versions alongside the runner. 

Set the [`python-version-file `](https://github.com/actions/setup-python/blob/main/docs/advanced-usage.md#using-the-python-version-file-input)option to use the pinned version for the project: 

example.yml 

```
[#__codelineno-3-1](#__codelineno-3-1)name: Example
[#__codelineno-3-2](#__codelineno-3-2)
[#__codelineno-3-3](#__codelineno-3-3)jobs:
[#__codelineno-3-4](#__codelineno-3-4)  uv-example:
[#__codelineno-3-5](#__codelineno-3-5)    name: python
[#__codelineno-3-6](#__codelineno-3-6)    runs-on: ubuntu-latest
[#__codelineno-3-7](#__codelineno-3-7)
[#__codelineno-3-8](#__codelineno-3-8)    steps:
[#__codelineno-3-9](#__codelineno-3-9)      - uses: actions/checkout@v5
[#__codelineno-3-10](#__codelineno-3-10)
[#__codelineno-3-11](#__codelineno-3-11)      - name: "Set up Python"
[#__codelineno-3-12](#__codelineno-3-12)        uses: actions/setup-python@v6
[#__codelineno-3-13](#__codelineno-3-13)        with:
[#__codelineno-3-14](#__codelineno-3-14)          python-version-file: ".python-version"
[#__codelineno-3-15](#__codelineno-3-15)
[#__codelineno-3-16](#__codelineno-3-16)      - name: Install uv
[#__codelineno-3-17](#__codelineno-3-17)        uses: astral-sh/setup-uv@v7

```

Or, specify the `pyproject.toml `file to ignore the pin and use the latest version compatible with the project's `requires-python `constraint: 

example.yml 

```
[#__codelineno-4-1](#__codelineno-4-1)name: Example
[#__codelineno-4-2](#__codelineno-4-2)
[#__codelineno-4-3](#__codelineno-4-3)jobs:
[#__codelineno-4-4](#__codelineno-4-4)  uv-example:
[#__codelineno-4-5](#__codelineno-4-5)    name: python
[#__codelineno-4-6](#__codelineno-4-6)    runs-on: ubuntu-latest
[#__codelineno-4-7](#__codelineno-4-7)
[#__codelineno-4-8](#__codelineno-4-8)    steps:
[#__codelineno-4-9](#__codelineno-4-9)      - uses: actions/checkout@v5
[#__codelineno-4-10](#__codelineno-4-10)
[#__codelineno-4-11](#__codelineno-4-11)      - name: "Set up Python"
[#__codelineno-4-12](#__codelineno-4-12)        uses: actions/setup-python@v6
[#__codelineno-4-13](#__codelineno-4-13)        with:
[#__codelineno-4-14](#__codelineno-4-14)          python-version-file: "pyproject.toml"
[#__codelineno-4-15](#__codelineno-4-15)
[#__codelineno-4-16](#__codelineno-4-16)      - name: Install uv
[#__codelineno-4-17](#__codelineno-4-17)        uses: astral-sh/setup-uv@v7

```

## [Multiple Python versions](#multiple-python-versions)

When using a matrix to test multiple Python versions, set the Python version using `astral-sh/setup-uv `, which will override the Python version specification in the `pyproject.toml `or `.python-version `files: 

example.yml 

```
[#__codelineno-5-1](#__codelineno-5-1)jobs:
[#__codelineno-5-2](#__codelineno-5-2)  build:
[#__codelineno-5-3](#__codelineno-5-3)    name: continuous-integration
[#__codelineno-5-4](#__codelineno-5-4)    runs-on: ubuntu-latest
[#__codelineno-5-5](#__codelineno-5-5)    strategy:
[#__codelineno-5-6](#__codelineno-5-6)      matrix:
[#__codelineno-5-7](#__codelineno-5-7)        python-version:
[#__codelineno-5-8](#__codelineno-5-8)          - "3.10"
[#__codelineno-5-9](#__codelineno-5-9)          - "3.11"
[#__codelineno-5-10](#__codelineno-5-10)          - "3.12"
[#__codelineno-5-11](#__codelineno-5-11)
[#__codelineno-5-12](#__codelineno-5-12)    steps:
[#__codelineno-5-13](#__codelineno-5-13)      - uses: actions/checkout@v5
[#__codelineno-5-14](#__codelineno-5-14)
[#__codelineno-5-15](#__codelineno-5-15)      - name: Install uv and set the Python version
[#__codelineno-5-16](#__codelineno-5-16)        uses: astral-sh/setup-uv@v7
[#__codelineno-5-17](#__codelineno-5-17)        with:
[#__codelineno-5-18](#__codelineno-5-18)          python-version: ${{ matrix.python-version }}

```

If not using the `setup-uv `action, you can set the `UV_PYTHON `environment variable: 

example.yml 

```
[#__codelineno-6-1](#__codelineno-6-1)jobs:
[#__codelineno-6-2](#__codelineno-6-2)  build:
[#__codelineno-6-3](#__codelineno-6-3)    name: continuous-integration
[#__codelineno-6-4](#__codelineno-6-4)    runs-on: ubuntu-latest
[#__codelineno-6-5](#__codelineno-6-5)    strategy:
[#__codelineno-6-6](#__codelineno-6-6)      matrix:
[#__codelineno-6-7](#__codelineno-6-7)        python-version:
[#__codelineno-6-8](#__codelineno-6-8)          - "3.10"
[#__codelineno-6-9](#__codelineno-6-9)          - "3.11"
[#__codelineno-6-10](#__codelineno-6-10)          - "3.12"
[#__codelineno-6-11](#__codelineno-6-11)    env:
[#__codelineno-6-12](#__codelineno-6-12)      UV_PYTHON: ${{ matrix.python-version }}
[#__codelineno-6-13](#__codelineno-6-13)    steps:
[#__codelineno-6-14](#__codelineno-6-14)      - uses: actions/checkout@v5

```

## [Syncing and running](#syncing-and-running)

Once uv and Python are installed, the project can be installed with `uv sync `and commands can be run in the environment with `uv run `: 

example.yml 

```
[#__codelineno-7-1](#__codelineno-7-1)name: Example
[#__codelineno-7-2](#__codelineno-7-2)
[#__codelineno-7-3](#__codelineno-7-3)jobs:
[#__codelineno-7-4](#__codelineno-7-4)  uv-example:
[#__codelineno-7-5](#__codelineno-7-5)    name: python
[#__codelineno-7-6](#__codelineno-7-6)    runs-on: ubuntu-latest
[#__codelineno-7-7](#__codelineno-7-7)
[#__codelineno-7-8](#__codelineno-7-8)    steps:
[#__codelineno-7-9](#__codelineno-7-9)      - uses: actions/checkout@v5
[#__codelineno-7-10](#__codelineno-7-10)
[#__codelineno-7-11](#__codelineno-7-11)      - name: Install uv
[#__codelineno-7-12](#__codelineno-7-12)        uses: astral-sh/setup-uv@v7
[#__codelineno-7-13](#__codelineno-7-13)
[#__codelineno-7-14](#__codelineno-7-14)      - name: Install the project
[#__codelineno-7-15](#__codelineno-7-15)        run: uv sync --locked --all-extras --dev
[#__codelineno-7-16](#__codelineno-7-16)
[#__codelineno-7-17](#__codelineno-7-17)      - name: Run tests
[#__codelineno-7-18](#__codelineno-7-18)        # For example, using `pytest`
[#__codelineno-7-19](#__codelineno-7-19)        run: uv run pytest tests

```

!!! tip "Tip"

    The [`UV_PROJECT_ENVIRONMENT `setting](../../../concepts/projects/config/#project-environment-path)can be used to install to the system Python environment instead of creating a virtual environment. 

## [Caching](#caching)

It may improve CI times to store uv's cache across workflow runs. 

The [`astral-sh/setup-uv `](https://github.com/astral-sh/setup-uv)has built-in support for persisting the cache: 

example.yml 

```
[#__codelineno-8-1](#__codelineno-8-1)- name: Enable caching
[#__codelineno-8-2](#__codelineno-8-2)  uses: astral-sh/setup-uv@v7
[#__codelineno-8-3](#__codelineno-8-3)  with:
[#__codelineno-8-4](#__codelineno-8-4)    enable-cache: true

```

Alternatively, you can manage the cache manually with the `actions/cache `action: 

example.yml 

```
[#__codelineno-9-1](#__codelineno-9-1)jobs:
[#__codelineno-9-2](#__codelineno-9-2)  install_job:
[#__codelineno-9-3](#__codelineno-9-3)    env:
[#__codelineno-9-4](#__codelineno-9-4)      # Configure a constant location for the uv cache
[#__codelineno-9-5](#__codelineno-9-5)      UV_CACHE_DIR: /tmp/.uv-cache
[#__codelineno-9-6](#__codelineno-9-6)
[#__codelineno-9-7](#__codelineno-9-7)    steps:
[#__codelineno-9-8](#__codelineno-9-8)      # ... setup up Python and uv ...
[#__codelineno-9-9](#__codelineno-9-9)
[#__codelineno-9-10](#__codelineno-9-10)      - name: Restore uv cache
[#__codelineno-9-11](#__codelineno-9-11)        uses: actions/cache@v4
[#__codelineno-9-12](#__codelineno-9-12)        with:
[#__codelineno-9-13](#__codelineno-9-13)          path: /tmp/.uv-cache
[#__codelineno-9-14](#__codelineno-9-14)          key: uv-${{ runner.os }}-${{ hashFiles('uv.lock') }}
[#__codelineno-9-15](#__codelineno-9-15)          restore-keys: |
[#__codelineno-9-16](#__codelineno-9-16)            uv-${{ runner.os }}-${{ hashFiles('uv.lock') }}
[#__codelineno-9-17](#__codelineno-9-17)            uv-${{ runner.os }}
[#__codelineno-9-18](#__codelineno-9-18)
[#__codelineno-9-19](#__codelineno-9-19)      # ... install packages, run tests, etc ...
[#__codelineno-9-20](#__codelineno-9-20)
[#__codelineno-9-21](#__codelineno-9-21)      - name: Minimize uv cache
[#__codelineno-9-22](#__codelineno-9-22)        run: uv cache prune --ci

```

The `uv cache prune --ci `command is used to reduce the size of the cache and is optimized for CI. Its effect on performance is dependent on the packages being installed. 

!!! tip "Tip"

    If using `uv pip `, use `requirements.txt `instead of `uv.lock `in the cache key. 

!!! note "Note"

    When using non-ephemeral, self-hosted runners the default cache directory can grow unbounded. In this case, it may not be optimal to share the cache between jobs. Instead, move the cache inside the GitHub Workspace and remove it once the job finishes using a [Post Job Hook](https://docs.github.com/en/actions/hosting-your-own-runners/managing-self-hosted-runners/running-scripts-before-or-after-a-job). 

    ```
[#__codelineno-10-1](#__codelineno-10-1)install_job:
[#__codelineno-10-2](#__codelineno-10-2)  env:
[#__codelineno-10-3](#__codelineno-10-3)    # Configure a relative location for the uv cache
[#__codelineno-10-4](#__codelineno-10-4)    UV_CACHE_DIR: ${{ github.workspace }}/.cache/uv

```

    Using a post job hook requires setting the `ACTIONS_RUNNER_HOOK_JOB_STARTED `environment variable on the self-hosted runner to the path of a cleanup script such as the one shown below. 

    clean-uv-cache.sh 

    ```
[#__codelineno-11-1](#__codelineno-11-1)#!/usr/bin/env sh
[#__codelineno-11-2](#__codelineno-11-2)uv cache clean

```

## [Using `uv pip `](#using-uv-pip)

If using the `uv pip `interface instead of the uv project interface, uv requires a virtual environment by default. To allow installing packages into the system environment, use the `--system `flag on all `uv `invocations or set the `UV_SYSTEM_PYTHON `variable. 

The `UV_SYSTEM_PYTHON `variable can be defined in at different scopes. 

Opt-in for the entire workflow by defining it at the top level: 

example.yml 

```
[#__codelineno-12-1](#__codelineno-12-1)env:
[#__codelineno-12-2](#__codelineno-12-2)  UV_SYSTEM_PYTHON: 1
[#__codelineno-12-3](#__codelineno-12-3)
[#__codelineno-12-4](#__codelineno-12-4)jobs: ...

```

Or, opt-in for a specific job in the workflow: 

example.yml 

```
[#__codelineno-13-1](#__codelineno-13-1)jobs:
[#__codelineno-13-2](#__codelineno-13-2)  install_job:
[#__codelineno-13-3](#__codelineno-13-3)    env:
[#__codelineno-13-4](#__codelineno-13-4)      UV_SYSTEM_PYTHON: 1
[#__codelineno-13-5](#__codelineno-13-5)    ...

```

Or, opt-in for a specific step in a job: 

example.yml 

```
[#__codelineno-14-1](#__codelineno-14-1)steps:
[#__codelineno-14-2](#__codelineno-14-2)  - name: Install requirements
[#__codelineno-14-3](#__codelineno-14-3)    run: uv pip install -r requirements.txt
[#__codelineno-14-4](#__codelineno-14-4)    env:
[#__codelineno-14-5](#__codelineno-14-5)      UV_SYSTEM_PYTHON: 1

```

To opt-out again, the `--no-system `flag can be used in any uv invocation. 

## [Private repos](#private-repos)

If your project has [dependencies](../../../concepts/projects/dependencies/#git)on private GitHub repositories, you will need to configure a [personal access token (PAT)](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens)to allow uv to fetch them. 

After creating a PAT that has read access to the private repositories, add it as a [repository secret](https://docs.github.com/en/actions/security-for-github-actions/security-guides/using-secrets-in-github-actions#creating-secrets-for-a-repository). 

Then, you can use the [`gh `](https://cli.github.com/)CLI (which is installed in GitHub Actions runners by default) to configure a [credential helper for Git](../../../concepts/authentication/git/#git-credential-helpers)to use the PAT for queries to repositories hosted on `github.com `. 

For example, if you called your repository secret `MY_PAT `: 

example.yml 

```
[#__codelineno-15-1](#__codelineno-15-1)steps:
[#__codelineno-15-2](#__codelineno-15-2)  - name: Register the personal access token
[#__codelineno-15-3](#__codelineno-15-3)    run: echo "${{ secrets.MY_PAT }}" | gh auth login --with-token
[#__codelineno-15-4](#__codelineno-15-4)  - name: Configure the Git credential helper
[#__codelineno-15-5](#__codelineno-15-5)    run: gh auth setup-git

```

## [Publishing to PyPI](#publishing-to-pypi)

uv can be used to build and publish your package to PyPI from GitHub Actions. We provide a standalone example alongside this guide in [astral-sh/trusted-publishing-examples](https://github.com/astral-sh/trusted-publishing-examples). The workflow uses [trusted publishing](https://docs.pypi.org/trusted-publishers/), so no credentials need to be configured. 

In the example workflow, we use a script to test that the source distribution and the wheel are both functional and we didn't miss any files. This step is recommended, but optional. 

First, add a release workflow to your project: 

.github/workflows/publish.yml 

```
[#__codelineno-16-1](#__codelineno-16-1)name: "Publish"
[#__codelineno-16-2](#__codelineno-16-2)
[#__codelineno-16-3](#__codelineno-16-3)on:
[#__codelineno-16-4](#__codelineno-16-4)  push:
[#__codelineno-16-5](#__codelineno-16-5)    tags:
[#__codelineno-16-6](#__codelineno-16-6)      # Publish on any tag starting with a `v`, e.g., v0.1.0
[#__codelineno-16-7](#__codelineno-16-7)      - v*
[#__codelineno-16-8](#__codelineno-16-8)
[#__codelineno-16-9](#__codelineno-16-9)jobs:
[#__codelineno-16-10](#__codelineno-16-10)  run:
[#__codelineno-16-11](#__codelineno-16-11)    runs-on: ubuntu-latest
[#__codelineno-16-12](#__codelineno-16-12)    environment:
[#__codelineno-16-13](#__codelineno-16-13)      name: pypi
[#__codelineno-16-14](#__codelineno-16-14)    permissions:
[#__codelineno-16-15](#__codelineno-16-15)      id-token: write
[#__codelineno-16-16](#__codelineno-16-16)      contents: read
[#__codelineno-16-17](#__codelineno-16-17)    steps:
[#__codelineno-16-18](#__codelineno-16-18)      - name: Checkout
[#__codelineno-16-19](#__codelineno-16-19)        uses: actions/checkout@v5
[#__codelineno-16-20](#__codelineno-16-20)      - name: Install uv
[#__codelineno-16-21](#__codelineno-16-21)        uses: astral-sh/setup-uv@v7
[#__codelineno-16-22](#__codelineno-16-22)      - name: Install Python 3.13
[#__codelineno-16-23](#__codelineno-16-23)        run: uv python install 3.13
[#__codelineno-16-24](#__codelineno-16-24)      - name: Build
[#__codelineno-16-25](#__codelineno-16-25)        run: uv build
[#__codelineno-16-26](#__codelineno-16-26)      # Check that basic features work and we didn't miss to include crucial files
[#__codelineno-16-27](#__codelineno-16-27)      - name: Smoke test (wheel)
[#__codelineno-16-28](#__codelineno-16-28)        run: uv run --isolated --no-project --with dist/*.whl tests/smoke_test.py
[#__codelineno-16-29](#__codelineno-16-29)      - name: Smoke test (source distribution)
[#__codelineno-16-30](#__codelineno-16-30)        run: uv run --isolated --no-project --with dist/*.tar.gz tests/smoke_test.py
[#__codelineno-16-31](#__codelineno-16-31)      - name: Publish
[#__codelineno-16-32](#__codelineno-16-32)        run: uv publish

```

Then, create the environment defined in the workflow in the GitHub repository under "Settings" -> "Environments". 

![GitHub settings dialog showing how to add the "pypi" environment under "Settings" -> "Environments"](../../../assets/github-add-environment.png)

Add a [trusted publisher](https://docs.pypi.org/trusted-publishers/adding-a-publisher/)to your PyPI project in the project settings under "Publishing". Ensure that all fields match with your GitHub configuration. 

![PyPI project publishing settings dialog showing how to set all fields for a trusted publisher configuration](../../../assets/pypi-add-trusted-publisher.png)

After saving: 

![PyPI project publishing settings dialog showing the configured trusted publishing settings](../../../assets/pypi-with-trusted-publisher.png)

Finally, tag a release and push it. Make sure it starts with `v `to match the pattern in the workflow. 

```
[#__codelineno-17-1](#__codelineno-17-1)$ git tag -a v0.1.0 -m v0.1.0
[#__codelineno-17-2](#__codelineno-17-2)$ git push --tags

```

December 9, 2025
