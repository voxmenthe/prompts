# [Using uv in GitLab CI/CD](#using-uv-in-gitlab-cicd)

## [Using the uv image](#using-the-uv-image)

Astral provides [Docker images](../docker/#available-images)with uv preinstalled. Select a variant that is suitable for your workflow. 

gitlab-ci.yml 

```
[#__codelineno-0-1](#__codelineno-0-1)variables:
[#__codelineno-0-2](#__codelineno-0-2)  UV_VERSION: "0.9.17"
[#__codelineno-0-3](#__codelineno-0-3)  PYTHON_VERSION: "3.12"
[#__codelineno-0-4](#__codelineno-0-4)  BASE_LAYER: bookworm-slim
[#__codelineno-0-5](#__codelineno-0-5)  # GitLab CI creates a separate mountpoint for the build directory,
[#__codelineno-0-6](#__codelineno-0-6)  # so we need to copy instead of using hard links.
[#__codelineno-0-7](#__codelineno-0-7)  UV_LINK_MODE: copy
[#__codelineno-0-8](#__codelineno-0-8)
[#__codelineno-0-9](#__codelineno-0-9)uv:
[#__codelineno-0-10](#__codelineno-0-10)  image: ghcr.io/astral-sh/uv:$UV_VERSION-python$PYTHON_VERSION-$BASE_LAYER
[#__codelineno-0-11](#__codelineno-0-11)  script:
[#__codelineno-0-12](#__codelineno-0-12)    # your `uv` commands

```

!!! note "Note"

    If you are using a distroless image, you have to specify the entrypoint: 

    ```
[#__codelineno-1-1](#__codelineno-1-1)uv:
[#__codelineno-1-2](#__codelineno-1-2)  image:
[#__codelineno-1-3](#__codelineno-1-3)    name: ghcr.io/astral-sh/uv:$UV_VERSION
[#__codelineno-1-4](#__codelineno-1-4)    entrypoint: [""]
[#__codelineno-1-5](#__codelineno-1-5)  # ...

```



## [Caching](#caching)

Persisting the uv cache between workflow runs can improve performance. 

```
[#__codelineno-2-1](#__codelineno-2-1)uv-install:
[#__codelineno-2-2](#__codelineno-2-2)  variables:
[#__codelineno-2-3](#__codelineno-2-3)    UV_CACHE_DIR: .uv-cache
[#__codelineno-2-4](#__codelineno-2-4)  cache:
[#__codelineno-2-5](#__codelineno-2-5)    - key:
[#__codelineno-2-6](#__codelineno-2-6)        files:
[#__codelineno-2-7](#__codelineno-2-7)          - uv.lock
[#__codelineno-2-8](#__codelineno-2-8)      paths:
[#__codelineno-2-9](#__codelineno-2-9)        - $UV_CACHE_DIR
[#__codelineno-2-10](#__codelineno-2-10)  script:
[#__codelineno-2-11](#__codelineno-2-11)    # Your `uv` commands
[#__codelineno-2-12](#__codelineno-2-12)    - uv cache prune --ci

```

See the [GitLab caching documentation](https://docs.gitlab.com/ee/ci/caching/)for more details on configuring caching. 

Using `uv cache prune --ci `at the end of the job is recommended to reduce cache size. See the [uv cache documentation](../../../concepts/cache/#caching-in-continuous-integration)for more details. 

## [Using `uv pip `](#using-uv-pip)

If using the `uv pip `interface instead of the uv project interface, uv requires a virtual environment by default. To allow installing packages into the system environment, use the `--system `flag on all uv invocations or set the `UV_SYSTEM_PYTHON `variable. 

The `UV_SYSTEM_PYTHON `variable can be defined in at different scopes. You can read more about how [variables and their precedence works in GitLab here](https://docs.gitlab.com/ee/ci/variables/)

Opt-in for the entire workflow by defining it at the top level: 

gitlab-ci.yml 

```
[#__codelineno-3-1](#__codelineno-3-1)variables:
[#__codelineno-3-2](#__codelineno-3-2)  UV_SYSTEM_PYTHON: 1
[#__codelineno-3-3](#__codelineno-3-3)
[#__codelineno-3-4](#__codelineno-3-4)# [...]

```

To opt-out again, the `--no-system `flag can be used in any uv invocation. 

When persisting the cache, you may want to use `requirements.txt `or `pyproject.toml `as your cache key files instead of `uv.lock `. 

December 9, 2025
