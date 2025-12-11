# [Using uv in Docker](#using-uv-in-docker)

## [Getting started](#getting-started)

!!! tip "Tip"

    Check out the [`uv-docker-example `](https://github.com/astral-sh/uv-docker-example)project for an example of best practices when using uv to build an application in Docker. 

uv provides both _distroless _Docker images, which are useful for [copying uv binaries](#installing-uv)into your own image builds, and images derived from popular base images, which are useful for using uv in a container. The distroless images do not contain anything but the uv binaries. In contrast, the derived images include an operating system with uv pre-installed. 

As an example, to run uv in a container using a Debian-based image: 

```
[#__codelineno-0-1](#__codelineno-0-1)$ docker run --rm -it ghcr.io/astral-sh/uv:debian uv --help

```

### [Available images](#available-images)

The following distroless images are available: 

- `ghcr.io/astral-sh/uv:latest `

- `ghcr.io/astral-sh/uv:{major}.{minor}.{patch} `, e.g., `ghcr.io/astral-sh/uv:0.9.17 `

- `ghcr.io/astral-sh/uv:{major}.{minor} `, e.g., `ghcr.io/astral-sh/uv:0.8 `(the latest patch version) 

And the following derived images are available: 

- Based on `alpine:3.22 `: 

  - `ghcr.io/astral-sh/uv:alpine `

  - `ghcr.io/astral-sh/uv:alpine3.22 `


- Based on `alpine:3.21 `: 

  - `ghcr.io/astral-sh/uv:alpine3.21 `


- Based on `debian:trixie-slim `: 

  - `ghcr.io/astral-sh/uv:debian-slim `

  - `ghcr.io/astral-sh/uv:trixie-slim `


- Based on `debian:bookworm-slim `: 

  - `ghcr.io/astral-sh/uv:bookworm-slim `


- Based on `buildpack-deps:trixie `: 

  - `ghcr.io/astral-sh/uv:debian `

  - `ghcr.io/astral-sh/uv:trixie `


- Based on `buildpack-deps:bookworm `: 

  - `ghcr.io/astral-sh/uv:bookworm `


- Based on `python3.x-alpine `: 

  - `ghcr.io/astral-sh/uv:python3.14-alpine `

  - `ghcr.io/astral-sh/uv:python3.13-alpine `

  - `ghcr.io/astral-sh/uv:python3.12-alpine `

  - `ghcr.io/astral-sh/uv:python3.11-alpine `

  - `ghcr.io/astral-sh/uv:python3.10-alpine `

  - `ghcr.io/astral-sh/uv:python3.9-alpine `

  - `ghcr.io/astral-sh/uv:python3.8-alpine `


- Based on `python3.x-trixie `: 

  - `ghcr.io/astral-sh/uv:python3.14-trixie `

  - `ghcr.io/astral-sh/uv:python3.13-trixie `

  - `ghcr.io/astral-sh/uv:python3.12-trixie `

  - `ghcr.io/astral-sh/uv:python3.11-trixie `

  - `ghcr.io/astral-sh/uv:python3.10-trixie `

  - `ghcr.io/astral-sh/uv:python3.9-trixie `


- Based on `python3.x-slim-trixie `: 

  - `ghcr.io/astral-sh/uv:python3.14-trixie-slim `

  - `ghcr.io/astral-sh/uv:python3.13-trixie-slim `

  - `ghcr.io/astral-sh/uv:python3.12-trixie-slim `

  - `ghcr.io/astral-sh/uv:python3.11-trixie-slim `

  - `ghcr.io/astral-sh/uv:python3.10-trixie-slim `

  - `ghcr.io/astral-sh/uv:python3.9-trixie-slim `


- Based on `python3.x-bookworm `: 

  - `ghcr.io/astral-sh/uv:python3.14-bookworm `

  - `ghcr.io/astral-sh/uv:python3.13-bookworm `

  - `ghcr.io/astral-sh/uv:python3.12-bookworm `

  - `ghcr.io/astral-sh/uv:python3.11-bookworm `

  - `ghcr.io/astral-sh/uv:python3.10-bookworm `

  - `ghcr.io/astral-sh/uv:python3.9-bookworm `

  - `ghcr.io/astral-sh/uv:python3.8-bookworm `


- Based on `python3.x-slim-bookworm `: 

  - `ghcr.io/astral-sh/uv:python3.14-bookworm-slim `

  - `ghcr.io/astral-sh/uv:python3.13-bookworm-slim `

  - `ghcr.io/astral-sh/uv:python3.12-bookworm-slim `

  - `ghcr.io/astral-sh/uv:python3.11-bookworm-slim `

  - `ghcr.io/astral-sh/uv:python3.10-bookworm-slim `

  - `ghcr.io/astral-sh/uv:python3.9-bookworm-slim `

  - `ghcr.io/astral-sh/uv:python3.8-bookworm-slim `



As with the distroless image, each derived image is published with uv version tags as `ghcr.io/astral-sh/uv:{major}.{minor}.{patch}-{base} `and `ghcr.io/astral-sh/uv:{major}.{minor}-{base} `, e.g., `ghcr.io/astral-sh/uv:0.9.17-alpine `. 

In addition, starting with `0.8 `each derived image also sets `UV_TOOL_BIN_DIR `to `/usr/local/bin `to allow `uv tool install `to work as expected with the default user. 

For more details, see the [GitHub Container](https://github.com/astral-sh/uv/pkgs/container/uv)page. 

### [Installing uv](#installing-uv)

Use one of the above images with uv pre-installed or install uv by copying the binary from the official distroless Docker image: 

Dockerfile 

```
[#__codelineno-1-1](#__codelineno-1-1)FROM python:3.12-slim-trixie
[#__codelineno-1-2](#__codelineno-1-2)COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

```

Or, with the installer: 

Dockerfile 

```
[#__codelineno-2-1](#__codelineno-2-1)FROM python:3.12-slim-trixie
[#__codelineno-2-2](#__codelineno-2-2)
[#__codelineno-2-3](#__codelineno-2-3)# The installer requires curl (and certificates) to download the release archive
[#__codelineno-2-4](#__codelineno-2-4)RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates
[#__codelineno-2-5](#__codelineno-2-5)
[#__codelineno-2-6](#__codelineno-2-6)# Download the latest installer
[#__codelineno-2-7](#__codelineno-2-7)ADD https://astral.sh/uv/install.sh /uv-installer.sh
[#__codelineno-2-8](#__codelineno-2-8)
[#__codelineno-2-9](#__codelineno-2-9)# Run the installer then remove it
[#__codelineno-2-10](#__codelineno-2-10)RUN sh /uv-installer.sh && rm /uv-installer.sh
[#__codelineno-2-11](#__codelineno-2-11)
[#__codelineno-2-12](#__codelineno-2-12)# Ensure the installed binary is on the `PATH`
[#__codelineno-2-13](#__codelineno-2-13)ENV PATH="/root/.local/bin/:$PATH"

```

Note this requires `curl `to be available. 

In either case, it is best practice to pin to a specific uv version, e.g., with: 

```
[#__codelineno-3-1](#__codelineno-3-1)COPY --from=ghcr.io/astral-sh/uv:0.9.17 /uv /uvx /bin/

```

!!! tip "Tip"

    While the Dockerfile example above pins to a specific tag, it's also possible to pin a specific SHA256. Pinning a specific SHA256 is considered best practice in environments that require reproducible builds as tags can be moved across different commit SHAs. 

    ```
[#__codelineno-4-1](#__codelineno-4-1)# e.g., using a hash from a previous release
[#__codelineno-4-2](#__codelineno-4-2)COPY --from=ghcr.io/astral-sh/uv@sha256:2381d6aa60c326b71fd40023f921a0a3b8f91b14d5db6b90402e65a635053709 /uv /uvx /bin/

```

Or, with the installer: 

```
[#__codelineno-5-1](#__codelineno-5-1)ADD https://astral.sh/uv/0.9.17/install.sh /uv-installer.sh

```

### [Installing a project](#installing-a-project)

If you're using uv to manage your project, you can copy it into the image and install it: 

Dockerfile 

```
[#__codelineno-6-1](#__codelineno-6-1)# Copy the project into the image
[#__codelineno-6-2](#__codelineno-6-2)COPY . /app
[#__codelineno-6-3](#__codelineno-6-3)
[#__codelineno-6-4](#__codelineno-6-4)# Disable development dependencies
[#__codelineno-6-5](#__codelineno-6-5)ENV UV_NO_DEV=1
[#__codelineno-6-6](#__codelineno-6-6)
[#__codelineno-6-7](#__codelineno-6-7)# Sync the project into a new environment, asserting the lockfile is up to date
[#__codelineno-6-8](#__codelineno-6-8)WORKDIR /app
[#__codelineno-6-9](#__codelineno-6-9)RUN uv sync --locked

```

!!! important "Important"

    It is best practice to add `.venv `to a [`.dockerignore `file](https://docs.docker.com/build/concepts/context/#dockerignore-files)in your repository to prevent it from being included in image builds. The project virtual environment is dependent on your local platform and should be created from scratch in the image. 

Then, to start your application by default: 

Dockerfile 

```
[#__codelineno-7-1](#__codelineno-7-1)# Presuming there is a `my_app` command provided by the project
[#__codelineno-7-2](#__codelineno-7-2)CMD ["uv", "run", "my_app"]

```

!!! tip "Tip"

    It is best practice to use [intermediate layers](#intermediate-layers)separating installation of dependencies and the project itself to improve Docker image build times. 

See a complete example in the [`uv-docker-example `project](https://github.com/astral-sh/uv-docker-example/blob/main/Dockerfile). 

### [Using the environment](#using-the-environment)

Once the project is installed, you can either _activate _the project virtual environment by placing its binary directory at the front of the path: 

Dockerfile 

```
[#__codelineno-8-1](#__codelineno-8-1)ENV PATH="/app/.venv/bin:$PATH"

```

Or, you can use `uv run `for any commands that require the environment: 

Dockerfile 

```
[#__codelineno-9-1](#__codelineno-9-1)RUN uv run some_script.py

```

!!! tip "Tip"

    Alternatively, the [`UV_PROJECT_ENVIRONMENT `setting](../../../concepts/projects/config/#project-environment-path)can be set before syncing to install to the system Python environment and skip environment activation entirely. 

### [Using installed tools](#using-installed-tools)

To use installed tools, ensure the [tool bin directory](../../../concepts/tools/#tool-executables)is on the path: 

Dockerfile 

```
[#__codelineno-10-1](#__codelineno-10-1)ENV PATH=/root/.local/bin:$PATH
[#__codelineno-10-2](#__codelineno-10-2)RUN uv tool install cowsay

```

```
[#__codelineno-11-1](#__codelineno-11-1)$ docker run -it $(docker build -q .) /bin/bash -c "cowsay -t hello"
[#__codelineno-11-2](#__codelineno-11-2)  _____
[#__codelineno-11-3](#__codelineno-11-3)| hello |
[#__codelineno-11-4](#__codelineno-11-4)  =====
[#__codelineno-11-5](#__codelineno-11-5)     \
[#__codelineno-11-6](#__codelineno-11-6)      \
[#__codelineno-11-7](#__codelineno-11-7)        ^__^
[#__codelineno-11-8](#__codelineno-11-8)        (oo)\_______
[#__codelineno-11-9](#__codelineno-11-9)        (__)\       )\/\
[#__codelineno-11-10](#__codelineno-11-10)            ||----w |
[#__codelineno-11-11](#__codelineno-11-11)            ||     ||

```

!!! note "Note"

    The tool bin directory's location can be determined by running the `uv tool dir --bin `command in the container. 

    Alternatively, it can be set to a constant location: 

    Dockerfile 

    ```
[#__codelineno-12-1](#__codelineno-12-1)ENV UV_TOOL_BIN_DIR=/opt/uv-bin/

```

### [Installing Python in ARM musl images](#installing-python-in-arm-musl-images)

While uv will attempt to [install a compatible Python version](../../install-python/)if no such version is available in the image, uv does not yet support installing Python for musl Linux on ARM. For example, if you are using an Alpine Linux base image on an ARM machine, you may need to add it with the system package manager: 

```
[#__codelineno-13-1](#__codelineno-13-1)apk add --no-cache python3~=3.12

```

## [Developing in a container](#developing-in-a-container)

When developing, it's useful to mount the project directory into a container. With this setup, changes to the project can be immediately reflected in a containerized service without rebuilding the image. However, it is important _not _to include the project virtual environment ( `.venv `) in the mount, because the virtual environment is platform specific and the one built for the image should be kept. 

### [Mounting the project with `docker run `](#mounting-the-project-with-docker-run)

Bind mount the project (in the working directory) to `/app `while retaining the `.venv `directory with an [anonymous volume](https://docs.docker.com/engine/storage/#volumes): 

```
[#__codelineno-14-1](#__codelineno-14-1)$ docker run --rm --volume .:/app --volume /app/.venv [...]

```

!!! tip "Tip"

    The `--rm `flag is included to ensure the container and anonymous volume are cleaned up when the container exits. 

See a complete example in the [`uv-docker-example `project](https://github.com/astral-sh/uv-docker-example/blob/main/run.sh). 

### [Configuring `watch `with `docker compose `](#configuring-watch-with-docker-compose)

When using Docker compose, more sophisticated tooling is available for container development. The [`watch `](https://docs.docker.com/compose/file-watch/#compose-watch-versus-bind-mounts)option allows for greater granularity than is practical with a bind mount and supports triggering updates to the containerized service when files change. 

!!! note "Note"

    This feature requires Compose 2.22.0 which is bundled with Docker Desktop 4.24. 

Configure `watch `in your [Docker compose file](https://docs.docker.com/compose/compose-application-model/#the-compose-file)to mount the project directory without syncing the project virtual environment and to rebuild the image when the configuration changes: 

compose.yaml 

```
[#__codelineno-15-1](#__codelineno-15-1)services:
[#__codelineno-15-2](#__codelineno-15-2)  example:
[#__codelineno-15-3](#__codelineno-15-3)    build: .
[#__codelineno-15-4](#__codelineno-15-4)
[#__codelineno-15-5](#__codelineno-15-5)    # ...
[#__codelineno-15-6](#__codelineno-15-6)
[#__codelineno-15-7](#__codelineno-15-7)    develop:
[#__codelineno-15-8](#__codelineno-15-8)      # Create a `watch` configuration to update the app
[#__codelineno-15-9](#__codelineno-15-9)      #
[#__codelineno-15-10](#__codelineno-15-10)      watch:
[#__codelineno-15-11](#__codelineno-15-11)        # Sync the working directory with the `/app` directory in the container
[#__codelineno-15-12](#__codelineno-15-12)        - action: sync
[#__codelineno-15-13](#__codelineno-15-13)          path: .
[#__codelineno-15-14](#__codelineno-15-14)          target: /app
[#__codelineno-15-15](#__codelineno-15-15)          # Exclude the project virtual environment
[#__codelineno-15-16](#__codelineno-15-16)          ignore:
[#__codelineno-15-17](#__codelineno-15-17)            - .venv/
[#__codelineno-15-18](#__codelineno-15-18)
[#__codelineno-15-19](#__codelineno-15-19)        # Rebuild the image on changes to the `pyproject.toml`
[#__codelineno-15-20](#__codelineno-15-20)        - action: rebuild
[#__codelineno-15-21](#__codelineno-15-21)          path: ./pyproject.toml

```

Then, run `docker compose watch `to run the container with the development setup. 

See a complete example in the [`uv-docker-example `project](https://github.com/astral-sh/uv-docker-example/blob/main/compose.yml). 

## [Optimizations](#optimizations)

### [Compiling bytecode](#compiling-bytecode)

Compiling Python source files to bytecode is typically desirable for production images as it tends to improve startup time (at the cost of increased installation time). 

To enable bytecode compilation, use the `--compile-bytecode `flag: 

Dockerfile 

```
[#__codelineno-16-1](#__codelineno-16-1)RUN uv sync --compile-bytecode

```

Alternatively, you can set the `UV_COMPILE_BYTECODE `environment variable to ensure that all commands within the Dockerfile compile bytecode: 

Dockerfile 

```
[#__codelineno-17-1](#__codelineno-17-1)ENV UV_COMPILE_BYTECODE=1

```

### [Caching](#caching)

A [cache mount](https://docs.docker.com/build/guide/mounts/#add-a-cache-mount)can be used to improve performance across builds: 

Dockerfile 

```
[#__codelineno-18-1](#__codelineno-18-1)ENV UV_LINK_MODE=copy
[#__codelineno-18-2](#__codelineno-18-2)
[#__codelineno-18-3](#__codelineno-18-3)RUN --mount=type=cache,target=/root/.cache/uv \
[#__codelineno-18-4](#__codelineno-18-4)    uv sync

```

Changing the default [`UV_LINK_MODE `](../../../reference/settings/#link-mode)silences warnings about not being able to use hard links since the cache and sync target are on separate file systems. 

If you're not mounting the cache, image size can be reduced by using the `--no-cache `flag or setting `UV_NO_CACHE `. 

By default, managed Python installations are not cached before being installed. Setting `UV_PYTHON_CACHE_DIR `can be used in combination with a cache mount: 

Dockerfile 

```
[#__codelineno-19-1](#__codelineno-19-1)ENV UV_PYTHON_CACHE_DIR=/root/.cache/uv/python
[#__codelineno-19-2](#__codelineno-19-2)
[#__codelineno-19-3](#__codelineno-19-3)RUN --mount=type=cache,target=/root/.cache/uv \
[#__codelineno-19-4](#__codelineno-19-4)    uv python install

```

!!! note "Note"

    The cache directory's location can be determined by running the `uv cache dir `command in the container. 

    Alternatively, the cache can be set to a constant location: 

    Dockerfile 

    ```
[#__codelineno-20-1](#__codelineno-20-1)ENV UV_CACHE_DIR=/opt/uv-cache/

```

### [Intermediate layers](#intermediate-layers)

If you're using uv to manage your project, you can improve build times by moving your transitive dependency installation into its own layer via the `--no-install `options. 

`uv sync --no-install-project `will install the dependencies of the project but not the project itself. Since the project changes frequently, but its dependencies are generally static, this can be a big time saver. 

Dockerfile 

```
[#__codelineno-21-1](#__codelineno-21-1)# Install uv
[#__codelineno-21-2](#__codelineno-21-2)FROM python:3.12-slim
[#__codelineno-21-3](#__codelineno-21-3)COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
[#__codelineno-21-4](#__codelineno-21-4)
[#__codelineno-21-5](#__codelineno-21-5)# Change the working directory to the `app` directory
[#__codelineno-21-6](#__codelineno-21-6)WORKDIR /app
[#__codelineno-21-7](#__codelineno-21-7)
[#__codelineno-21-8](#__codelineno-21-8)# Install dependencies
[#__codelineno-21-9](#__codelineno-21-9)RUN --mount=type=cache,target=/root/.cache/uv \
[#__codelineno-21-10](#__codelineno-21-10)    --mount=type=bind,source=uv.lock,target=uv.lock \
[#__codelineno-21-11](#__codelineno-21-11)    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
[#__codelineno-21-12](#__codelineno-21-12)    uv sync --locked --no-install-project
[#__codelineno-21-13](#__codelineno-21-13)
[#__codelineno-21-14](#__codelineno-21-14)# Copy the project into the image
[#__codelineno-21-15](#__codelineno-21-15)COPY . /app
[#__codelineno-21-16](#__codelineno-21-16)
[#__codelineno-21-17](#__codelineno-21-17)# Sync the project
[#__codelineno-21-18](#__codelineno-21-18)RUN --mount=type=cache,target=/root/.cache/uv \
[#__codelineno-21-19](#__codelineno-21-19)    uv sync --locked

```

Note that the `pyproject.toml `is required to identify the project root and name, but the project _contents _are not copied into the image until the final `uv sync `command. 

!!! tip "Tip"

    If you want to remove additional, specific packages from the sync, use `--no-install-package `. 

#### [Intermediate layers in workspaces](#intermediate-layers-in-workspaces)

If you're using a [workspace](../../../concepts/projects/workspaces/), then a couple changes are needed: 

- Use `--frozen `instead of `--locked `during the initially sync. 

- Use the `--no-install-workspace `flag which excludes the project _and _any workspace members. 

Dockerfile 

```
[#__codelineno-22-1](#__codelineno-22-1)# Install uv
[#__codelineno-22-2](#__codelineno-22-2)FROM python:3.12-slim
[#__codelineno-22-3](#__codelineno-22-3)COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
[#__codelineno-22-4](#__codelineno-22-4)
[#__codelineno-22-5](#__codelineno-22-5)WORKDIR /app
[#__codelineno-22-6](#__codelineno-22-6)
[#__codelineno-22-7](#__codelineno-22-7)RUN --mount=type=cache,target=/root/.cache/uv \
[#__codelineno-22-8](#__codelineno-22-8)    --mount=type=bind,source=uv.lock,target=uv.lock \
[#__codelineno-22-9](#__codelineno-22-9)    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
[#__codelineno-22-10](#__codelineno-22-10)    uv sync --frozen --no-install-workspace
[#__codelineno-22-11](#__codelineno-22-11)
[#__codelineno-22-12](#__codelineno-22-12)COPY . /app
[#__codelineno-22-13](#__codelineno-22-13)
[#__codelineno-22-14](#__codelineno-22-14)RUN --mount=type=cache,target=/root/.cache/uv \
[#__codelineno-22-15](#__codelineno-22-15)    uv sync --locked

```

uv cannot assert that the `uv.lock `file is up-to-date without each of the workspace member `pyproject.toml `files, so we use `--frozen `instead of `--locked `to skip the check during the initial sync. The next sync, after all the workspace members have been copied, can still use `--locked `and will validate that the lockfile is correct for all workspace members. 

### [Non-editable installs](#non-editable-installs)

By default, uv installs projects and workspace members in editable mode, such that changes to the source code are immediately reflected in the environment. 

`uv sync `and `uv run `both accept a `--no-editable `flag, which instructs uv to install the project in non-editable mode, removing any dependency on the source code. 

In the context of a multi-stage Docker image, `--no-editable `can be used to include the project in the synced virtual environment from one stage, then copy the virtual environment alone (and not the source code) into the final image. 

For example: 

Dockerfile 

```
[#__codelineno-23-1](#__codelineno-23-1)# Install uv
[#__codelineno-23-2](#__codelineno-23-2)FROM python:3.12-slim AS builder
[#__codelineno-23-3](#__codelineno-23-3)COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
[#__codelineno-23-4](#__codelineno-23-4)
[#__codelineno-23-5](#__codelineno-23-5)# Change the working directory to the `app` directory
[#__codelineno-23-6](#__codelineno-23-6)WORKDIR /app
[#__codelineno-23-7](#__codelineno-23-7)
[#__codelineno-23-8](#__codelineno-23-8)# Install dependencies
[#__codelineno-23-9](#__codelineno-23-9)RUN --mount=type=cache,target=/root/.cache/uv \
[#__codelineno-23-10](#__codelineno-23-10)    --mount=type=bind,source=uv.lock,target=uv.lock \
[#__codelineno-23-11](#__codelineno-23-11)    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
[#__codelineno-23-12](#__codelineno-23-12)    uv sync --locked --no-install-project --no-editable
[#__codelineno-23-13](#__codelineno-23-13)
[#__codelineno-23-14](#__codelineno-23-14)# Copy the project into the intermediate image
[#__codelineno-23-15](#__codelineno-23-15)COPY . /app
[#__codelineno-23-16](#__codelineno-23-16)
[#__codelineno-23-17](#__codelineno-23-17)# Sync the project
[#__codelineno-23-18](#__codelineno-23-18)RUN --mount=type=cache,target=/root/.cache/uv \
[#__codelineno-23-19](#__codelineno-23-19)    uv sync --locked --no-editable
[#__codelineno-23-20](#__codelineno-23-20)
[#__codelineno-23-21](#__codelineno-23-21)FROM python:3.12-slim
[#__codelineno-23-22](#__codelineno-23-22)
[#__codelineno-23-23](#__codelineno-23-23)# Copy the environment, but not the source code
[#__codelineno-23-24](#__codelineno-23-24)COPY --from=builder --chown=app:app /app/.venv /app/.venv
[#__codelineno-23-25](#__codelineno-23-25)
[#__codelineno-23-26](#__codelineno-23-26)# Run the application
[#__codelineno-23-27](#__codelineno-23-27)CMD ["/app/.venv/bin/hello"]

```

### [Using uv temporarily](#using-uv-temporarily)

If uv isn't needed in the final image, the binary can be mounted in each invocation: 

Dockerfile 

```
[#__codelineno-24-1](#__codelineno-24-1)RUN --mount=from=ghcr.io/astral-sh/uv,source=/uv,target=/bin/uv \
[#__codelineno-24-2](#__codelineno-24-2)    uv sync

```

## [Using the pip interface](#using-the-pip-interface)

### [Installing a package](#installing-a-package)

The system Python environment is safe to use this context, since a container is already isolated. The `--system `flag can be used to install in the system environment: 

Dockerfile 

```
[#__codelineno-25-1](#__codelineno-25-1)RUN uv pip install --system ruff

```

To use the system Python environment by default, set the `UV_SYSTEM_PYTHON `variable: 

Dockerfile 

```
[#__codelineno-26-1](#__codelineno-26-1)ENV UV_SYSTEM_PYTHON=1

```

Alternatively, a virtual environment can be created and activated: 

Dockerfile 

```
[#__codelineno-27-1](#__codelineno-27-1)RUN uv venv /opt/venv
[#__codelineno-27-2](#__codelineno-27-2)# Use the virtual environment automatically
[#__codelineno-27-3](#__codelineno-27-3)ENV VIRTUAL_ENV=/opt/venv
[#__codelineno-27-4](#__codelineno-27-4)# Place entry points in the environment at the front of the path
[#__codelineno-27-5](#__codelineno-27-5)ENV PATH="/opt/venv/bin:$PATH"

```

When using a virtual environment, the `--system `flag should be omitted from uv invocations: 

Dockerfile 

```
[#__codelineno-28-1](#__codelineno-28-1)RUN uv pip install ruff

```

### [Installing requirements](#installing-requirements)

To install requirements files, copy them into the container: 

Dockerfile 

```
[#__codelineno-29-1](#__codelineno-29-1)COPY requirements.txt .
[#__codelineno-29-2](#__codelineno-29-2)RUN uv pip install -r requirements.txt

```

### [Installing a project](#installing-a-project_1)

When installing a project alongside requirements, it is best practice to separate copying the requirements from the rest of the source code. This allows the dependencies of the project (which do not change often) to be cached separately from the project itself (which changes very frequently). 

Dockerfile 

```
[#__codelineno-30-1](#__codelineno-30-1)COPY pyproject.toml .
[#__codelineno-30-2](#__codelineno-30-2)RUN uv pip install -r pyproject.toml
[#__codelineno-30-3](#__codelineno-30-3)COPY . .
[#__codelineno-30-4](#__codelineno-30-4)RUN uv pip install -e .

```

## [Verifying image provenance](#verifying-image-provenance)

The Docker images are signed during the build process to provide proof of their origin. These attestations can be used to verify that an image was produced from an official channel. 

For example, you can verify the attestations with the [GitHub CLI tool `gh `](https://cli.github.com/): 

```
[#__codelineno-31-1](#__codelineno-31-1)$ gh attestation verify --owner astral-sh oci://ghcr.io/astral-sh/uv:latest
[#__codelineno-31-2](#__codelineno-31-2)Loaded digest sha256:xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx for oci://ghcr.io/astral-sh/uv:latest
[#__codelineno-31-3](#__codelineno-31-3)Loaded 1 attestation from GitHub API
[#__codelineno-31-4](#__codelineno-31-4)
[#__codelineno-31-5](#__codelineno-31-5)The following policy criteria will be enforced:
[#__codelineno-31-6](#__codelineno-31-6)- OIDC Issuer must match:................... https://token.actions.githubusercontent.com
[#__codelineno-31-7](#__codelineno-31-7)- Source Repository Owner URI must match:... https://github.com/astral-sh
[#__codelineno-31-8](#__codelineno-31-8)- Predicate type must match:................ https://slsa.dev/provenance/v1
[#__codelineno-31-9](#__codelineno-31-9)- Subject Alternative Name must match regex: (?i)^https://github.com/astral-sh/
[#__codelineno-31-10](#__codelineno-31-10)
[#__codelineno-31-11](#__codelineno-31-11)✓ Verification succeeded!
[#__codelineno-31-12](#__codelineno-31-12)
[#__codelineno-31-13](#__codelineno-31-13)sha256:xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx was attested by:
[#__codelineno-31-14](#__codelineno-31-14)REPO          PREDICATE_TYPE                  WORKFLOW
[#__codelineno-31-15](#__codelineno-31-15)astral-sh/uv  https://slsa.dev/provenance/v1  .github/workflows/build-docker.yml@refs/heads/main

```

This tells you that the specific Docker image was built by the official uv GitHub release workflow and hasn't been tampered with since. 

GitHub attestations build on the [sigstore.dev infrastructure](https://www.sigstore.dev/). As such you can also use the [`cosign `command](https://github.com/sigstore/cosign)to verify the attestation blob against the (multi-platform) manifest for `uv `: 

```
[#__codelineno-32-1](#__codelineno-32-1)$ REPO=astral-sh/uv
[#__codelineno-32-2](#__codelineno-32-2)$ gh attestation download --repo $REPO oci://ghcr.io/${REPO}:latest
[#__codelineno-32-3](#__codelineno-32-3)Wrote attestations to file sha256:xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx.jsonl.
[#__codelineno-32-4](#__codelineno-32-4)Any previous content has been overwritten
[#__codelineno-32-5](#__codelineno-32-5)
[#__codelineno-32-6](#__codelineno-32-6)The trusted metadata is now available at sha256:xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx.jsonl
[#__codelineno-32-7](#__codelineno-32-7)$ docker buildx imagetools inspect ghcr.io/${REPO}:latest --format "{{json .Manifest}}" > manifest.json
[#__codelineno-32-8](#__codelineno-32-8)$ cosign verify-blob-attestation \
[#__codelineno-32-9](#__codelineno-32-9)    --new-bundle-format \
[#__codelineno-32-10](#__codelineno-32-10)    --bundle "$(jq -r .digest manifest.json).jsonl"  \
[#__codelineno-32-11](#__codelineno-32-11)    --certificate-oidc-issuer="https://token.actions.githubusercontent.com" \
[#__codelineno-32-12](#__codelineno-32-12)    --certificate-identity-regexp="^https://github\.com/${REPO}/.*" \
[#__codelineno-32-13](#__codelineno-32-13)    <(jq -j '.|del(.digest,.size)' manifest.json)
[#__codelineno-32-14](#__codelineno-32-14)Verified OK

```

!!! tip "Tip"

    These examples use `latest `, but best practice is to verify the attestation for a specific version tag, e.g., `ghcr.io/astral-sh/uv:0.9.17 `, or (even better) the specific image digest, such as `ghcr.io/astral-sh/uv:0.5.27@sha256:5adf09a5a526f380237408032a9308000d14d5947eafa687ad6c6a2476787b4f `. 

December 9, 2025
