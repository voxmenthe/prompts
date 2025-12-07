# [Using uv with FastAPI](#using-uv-with-fastapi)

[FastAPI](https://github.com/fastapi/fastapi)is a modern, high-performance Python web framework. You can use uv to manage your FastAPI project, including installing dependencies, managing environments, running FastAPI applications, and more. 

!!! note "Note"

    You can view the source code for this guide in the [uv-fastapi-example](https://github.com/astral-sh/uv-fastapi-example)repository. 

## [Migrating an existing FastAPI project](#migrating-an-existing-fastapi-project)

As an example, consider the sample application defined in the [FastAPI documentation](https://fastapi.tiangolo.com/tutorial/bigger-applications/), structured as follows: 

```
[#__codelineno-0-1](#__codelineno-0-1)project
[#__codelineno-0-2](#__codelineno-0-2)└── app
[#__codelineno-0-3](#__codelineno-0-3)    ├── __init__.py
[#__codelineno-0-4](#__codelineno-0-4)    ├── main.py
[#__codelineno-0-5](#__codelineno-0-5)    ├── dependencies.py
[#__codelineno-0-6](#__codelineno-0-6)    ├── routers
[#__codelineno-0-7](#__codelineno-0-7)    │   ├── __init__.py
[#__codelineno-0-8](#__codelineno-0-8)    │   ├── items.py
[#__codelineno-0-9](#__codelineno-0-9)    │   └── users.py
[#__codelineno-0-10](#__codelineno-0-10)    └── internal
[#__codelineno-0-11](#__codelineno-0-11)        ├── __init__.py
[#__codelineno-0-12](#__codelineno-0-12)        └── admin.py

```

To use uv with this application, inside the `project `directory run: 

```
[#__codelineno-1-1](#__codelineno-1-1)$ uv init --app

```

This creates a [project with an application layout](../../../concepts/projects/init/#applications)and a `pyproject.toml `file. 

Then, add a dependency on FastAPI: 

```
[#__codelineno-2-1](#__codelineno-2-1)$ uv add fastapi --extra standard

```

You should now have the following structure: 

```
[#__codelineno-3-1](#__codelineno-3-1)project
[#__codelineno-3-2](#__codelineno-3-2)├── pyproject.toml
[#__codelineno-3-3](#__codelineno-3-3)└── app
[#__codelineno-3-4](#__codelineno-3-4)    ├── __init__.py
[#__codelineno-3-5](#__codelineno-3-5)    ├── main.py
[#__codelineno-3-6](#__codelineno-3-6)    ├── dependencies.py
[#__codelineno-3-7](#__codelineno-3-7)    ├── routers
[#__codelineno-3-8](#__codelineno-3-8)    │   ├── __init__.py
[#__codelineno-3-9](#__codelineno-3-9)    │   ├── items.py
[#__codelineno-3-10](#__codelineno-3-10)    │   └── users.py
[#__codelineno-3-11](#__codelineno-3-11)    └── internal
[#__codelineno-3-12](#__codelineno-3-12)        ├── __init__.py
[#__codelineno-3-13](#__codelineno-3-13)        └── admin.py

```

And the contents of the `pyproject.toml `file should look something like this: 

pyproject.toml 

```
[#__codelineno-4-1](#__codelineno-4-1)[project]
[#__codelineno-4-2](#__codelineno-4-2)name = "uv-fastapi-example"
[#__codelineno-4-3](#__codelineno-4-3)version = "0.1.0"
[#__codelineno-4-4](#__codelineno-4-4)description = "FastAPI project"
[#__codelineno-4-5](#__codelineno-4-5)readme = "README.md"
[#__codelineno-4-6](#__codelineno-4-6)requires-python = ">=3.12"
[#__codelineno-4-7](#__codelineno-4-7)dependencies = [
[#__codelineno-4-8](#__codelineno-4-8)    "fastapi[standard]",
[#__codelineno-4-9](#__codelineno-4-9)]

```

From there, you can run the FastAPI application with: 

```
[#__codelineno-5-1](#__codelineno-5-1)$ uv run fastapi dev

```

`uv run `will automatically resolve and lock the project dependencies (i.e., create a `uv.lock `alongside the `pyproject.toml `), create a virtual environment, and run the command in that environment. 

Test the app by opening [http://127.0.0.1:8000/?token=jessica](http://127.0.0.1:8000/?token=jessica)in a web browser. 

## [Deployment](#deployment)

To deploy the FastAPI application with Docker, you can use the following `Dockerfile `: 

Dockerfile 

```
[#__codelineno-6-1](#__codelineno-6-1)FROM python:3.12-slim
[#__codelineno-6-2](#__codelineno-6-2)
[#__codelineno-6-3](#__codelineno-6-3)# Install uv.
[#__codelineno-6-4](#__codelineno-6-4)COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
[#__codelineno-6-5](#__codelineno-6-5)
[#__codelineno-6-6](#__codelineno-6-6)# Copy the application into the container.
[#__codelineno-6-7](#__codelineno-6-7)COPY . /app
[#__codelineno-6-8](#__codelineno-6-8)
[#__codelineno-6-9](#__codelineno-6-9)# Install the application dependencies.
[#__codelineno-6-10](#__codelineno-6-10)WORKDIR /app
[#__codelineno-6-11](#__codelineno-6-11)RUN uv sync --frozen --no-cache
[#__codelineno-6-12](#__codelineno-6-12)
[#__codelineno-6-13](#__codelineno-6-13)# Run the application.
[#__codelineno-6-14](#__codelineno-6-14)CMD ["/app/.venv/bin/fastapi", "run", "app/main.py", "--port", "80", "--host", "0.0.0.0"]

```

Build the Docker image with: 

```
[#__codelineno-7-1](#__codelineno-7-1)$ docker build -t fastapi-app .

```

Run the Docker container locally with: 

```
[#__codelineno-8-1](#__codelineno-8-1)$ docker run -p 8000:80 fastapi-app

```

Navigate to [http://127.0.0.1:8000/?token=jessica](http://127.0.0.1:8000/?token=jessica)in your browser to verify that the app is running correctly. 

!!! tip "Tip"

    For more on using uv with Docker, see the [Docker guide](../docker/). 

November 17, 2025
