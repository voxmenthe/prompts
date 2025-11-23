# [CLI Reference](#cli-reference)

## [uv](#uv)

An extremely fast Python package manager. 

### Usage 

```
[#__codelineno-0-1](#__codelineno-0-1)uv [OPTIONS] 

```

### Commands 

[`uv auth `](#uv-auth)

Manage authentication 

[`uv run `](#uv-run)

Run a command or script 

[`uv init `](#uv-init)

Create a new project 

[`uv add `](#uv-add)

Add dependencies to the project 

[`uv remove `](#uv-remove)

Remove dependencies from the project 

[`uv version `](#uv-version)

Read or update the project's version 

[`uv sync `](#uv-sync)

Update the project's environment 

[`uv lock `](#uv-lock)

Update the project's lockfile 

[`uv export `](#uv-export)

Export the project's lockfile to an alternate format 

[`uv tree `](#uv-tree)

Display the project's dependency tree 

[`uv format `](#uv-format)

Format Python code in the project 

[`uv tool `](#uv-tool)

Run and install commands provided by Python packages 

[`uv python `](#uv-python)

Manage Python versions and installations 

[`uv pip `](#uv-pip)

Manage Python packages with a pip-compatible interface 

[`uv venv `](#uv-venv)

Create a virtual environment 

[`uv build `](#uv-build)

Build Python packages into source distributions and wheels 

[`uv publish `](#uv-publish)

Upload distributions to an index 

[`uv cache `](#uv-cache)

Manage uv's cache 

[`uv self `](#uv-self)

Manage the uv executable 

[`uv help `](#uv-help)

Display documentation for a command 

## [uv auth](#uv-auth)

Manage authentication 

### Usage 

```
[#__codelineno-1-1](#__codelineno-1-1)uv auth [OPTIONS] 

```

### Commands 

[`uv auth login `](#uv-auth-login)

Login to a service 

[`uv auth logout `](#uv-auth-logout)

Logout of a service 

[`uv auth token `](#uv-auth-token)

Show the authentication token for a service 

[`uv auth dir `](#uv-auth-dir)

Show the path to the uv credentials directory 

### [uv auth login](#uv-auth-login)

Login to a service 

### Usage 

```
[#__codelineno-2-1](#__codelineno-2-1)uv auth login [OPTIONS] 

```

### Arguments 

[SERVICE](#uv-auth-login--service)

The domain or URL of the service to log into
