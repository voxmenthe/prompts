# [Using uv with AWS Lambda](#using-uv-with-aws-lambda)

[AWS Lambda](https://aws.amazon.com/lambda/)is a serverless computing service that lets you run code without provisioning or managing servers. 

You can use uv with AWS Lambda to manage your Python dependencies, build your deployment package, and deploy your Lambda functions. 

!!! tip "Tip"

    Check out the [`uv-aws-lambda-example `](https://github.com/astral-sh/uv-aws-lambda-example)project for an example of best practices when using uv to deploy an application to AWS Lambda. 

## [Getting started](#getting-started)

To start, assume we have a minimal FastAPI application with the following structure: 

```
[#__codelineno-0-1](#__codelineno-0-1)project
[#__codelineno-0-2](#__codelineno-0-2)├── pyproject.toml
[#__codelineno-0-3](#__codelineno-0-3)└── app
[#__codelineno-0-4](#__codelineno-0-4)    ├── __init__.py
[#__codelineno-0-5](#__codelineno-0-5)    └── main.py

```

Where the `pyproject.toml `contains: 

pyproject.toml 

```
[#__codelineno-1-1](#__codelineno-1-1)[project]
[#__codelineno-1-2](#__codelineno-1-2)name = "uv-aws-lambda-example"
[#__codelineno-1-3](#__codelineno-1-3)version = "0.1.0"
[#__codelineno-1-4](#__codelineno-1-4)requires-python = ">=3.13"
[#__codelineno-1-5](#__codelineno-1-5)dependencies = [
[#__codelineno-1-6](#__codelineno-1-6)    # FastAPI is a modern web framework for building APIs with Python.
[#__codelineno-1-7](#__codelineno-1-7)    "fastapi",
[#__codelineno-1-8](#__codelineno-1-8)    # Mangum is a library that adapts ASGI applications to AWS Lambda and API Gateway.
[#__codelineno-1-9](#__codelineno-1-9)    "mangum",
[#__codelineno-1-10](#__codelineno-1-10)]
[#__codelineno-1-11](#__codelineno-1-11)
[#__codelineno-1-12](#__codelineno-1-12)[dependency-groups]
[#__codelineno-1-13](#__codelineno-1-13)dev = [
[#__codelineno-1-14](#__codelineno-1-14)    # In development mode, include the FastAPI development server.
[#__codelineno-1-15](#__codelineno-1-15)    "fastapi[standard]>=0.115",
[#__codelineno-1-16](#__codelineno-1-16)]

```

And the `main.py `file contains: 

app/main.py 

```
[#__codelineno-2-1](#__codelineno-2-1)import logging
[#__codelineno-2-2](#__codelineno-2-2)
[#__codelineno-2-3](#__codelineno-2-3)from fastapi import FastAPI
[#__codelineno-2-4](#__codelineno-2-4)from mangum import Mangum
[#__codelineno-2-5](#__codelineno-2-5)
[#__codelineno-2-6](#__codelineno-2-6)logger = logging.getLogger()
[#__codelineno-2-7](#__codelineno-2-7)logger.setLevel(logging.INFO)
[#__codelineno-2-8](#__codelineno-2-8)
[#__codelineno-2-9](#__codelineno-2-9)app = FastAPI()
[#__codelineno-2-10](#__codelineno-2-10)handler = Mangum(app)
[#__codelineno-2-11](#__codelineno-2-11)
[#__codelineno-2-12](#__codelineno-2-12)
[#__codelineno-2-13](#__codelineno-2-13)@app.get("/")
[#__codelineno-2-14](#__codelineno-2-14)async def root() -> str:
[#__codelineno-2-15](#__codelineno-2-15)    return "Hello, world!"

```

We can run this application locally with: 

```
[#__codelineno-3-1](#__codelineno-3-1)$ uv run fastapi dev

```

From there, opening [http://127.0.0.1:8000/](http://127.0.0.1:8000/)in a web browser will display "Hello, world!" 

## [Deploying a Docker image](#deploying-a-docker-image)

To deploy to AWS Lambda, we need to build a container image that includes the application code and dependencies in a single output directory. 

We'll follow the principles outlined in the [Docker guide](../docker/)(in particular, a multi-stage build) to ensure that the final image is as small and cache-friendly as possible. 

In the first stage, we'll populate a single directory with all application code and dependencies. In the second stage, we'll copy this directory over to the final image, omitting the build tools and other unnecessary files. 

Dockerfile 

```
[#__codelineno-4-1](#__codelineno-4-1)FROM ghcr.io/astral-sh/uv:0.9.17 AS uv
[#__codelineno-4-2](#__codelineno-4-2)
[#__codelineno-4-3](#__codelineno-4-3)# First, bundle the dependencies into the task root.
[#__codelineno-4-4](#__codelineno-4-4)FROM public.ecr.aws/lambda/python:3.13 AS builder
[#__codelineno-4-5](#__codelineno-4-5)
[#__codelineno-4-6](#__codelineno-4-6)# Enable bytecode compilation, to improve cold-start performance.
[#__codelineno-4-7](#__codelineno-4-7)ENV UV_COMPILE_BYTECODE=1
[#__codelineno-4-8](#__codelineno-4-8)
[#__codelineno-4-9](#__codelineno-4-9)# Disable installer metadata, to create a deterministic layer.
[#__codelineno-4-10](#__codelineno-4-10)ENV UV_NO_INSTALLER_METADATA=1
[#__codelineno-4-11](#__codelineno-4-11)
[#__codelineno-4-12](#__codelineno-4-12)# Enable copy mode to support bind mount caching.
[#__codelineno-4-13](#__codelineno-4-13)ENV UV_LINK_MODE=copy
[#__codelineno-4-14](#__codelineno-4-14)
[#__codelineno-4-15](#__codelineno-4-15)# Bundle the dependencies into the Lambda task root via `uv pip install --target`.
[#__codelineno-4-16](#__codelineno-4-16)#
[#__codelineno-4-17](#__codelineno-4-17)# Omit any local packages (`--no-emit-workspace`) and development dependencies (`--no-dev`).
[#__codelineno-4-18](#__codelineno-4-18)# This ensures that the Docker layer cache is only invalidated when the `pyproject.toml` or `uv.lock`
[#__codelineno-4-19](#__codelineno-4-19)# files change, but remains robust to changes in the application code.
[#__codelineno-4-20](#__codelineno-4-20)RUN --mount=from=uv,source=/uv,target=/bin/uv \
[#__codelineno-4-21](#__codelineno-4-21)    --mount=type=cache,target=/root/.cache/uv \
[#__codelineno-4-22](#__codelineno-4-22)    --mount=type=bind,source=uv.lock,target=uv.lock \
[#__codelineno-4-23](#__codelineno-4-23)    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
[#__codelineno-4-24](#__codelineno-4-24)    uv export --frozen --no-emit-workspace --no-dev --no-editable -o requirements.txt && \
[#__codelineno-4-25](#__codelineno-4-25)    uv pip install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"
[#__codelineno-4-26](#__codelineno-4-26)
[#__codelineno-4-27](#__codelineno-4-27)FROM public.ecr.aws/lambda/python:3.13
[#__codelineno-4-28](#__codelineno-4-28)
[#__codelineno-4-29](#__codelineno-4-29)# Copy the runtime dependencies from the builder stage.
[#__codelineno-4-30](#__codelineno-4-30)COPY --from=builder ${LAMBDA_TASK_ROOT} ${LAMBDA_TASK_ROOT}
[#__codelineno-4-31](#__codelineno-4-31)
[#__codelineno-4-32](#__codelineno-4-32)# Copy the application code.
[#__codelineno-4-33](#__codelineno-4-33)COPY ./app ${LAMBDA_TASK_ROOT}/app
[#__codelineno-4-34](#__codelineno-4-34)
[#__codelineno-4-35](#__codelineno-4-35)# Set the AWS Lambda handler.
[#__codelineno-4-36](#__codelineno-4-36)CMD ["app.main.handler"]

```

!!! tip "Tip"

    To deploy to ARM-based AWS Lambda runtimes, replace `public.ecr.aws/lambda/python:3.13 `with `public.ecr.aws/lambda/python:3.13-arm64 `. 

We can build the image with, e.g.: 

```
[#__codelineno-5-1](#__codelineno-5-1)$ uv lock
[#__codelineno-5-2](#__codelineno-5-2)$ docker build -t fastapi-app .

```

The core benefits of this Dockerfile structure are as follows: 

1. **Minimal image size. **By using a multi-stage build, we can ensure that the final image only includes the application code and dependencies. For example, the uv binary itself is not included in the final image. 

2. **Maximal cache reuse. **By installing application dependencies separately from the application code, we can ensure that the Docker layer cache is only invalidated when the dependencies change. 

Concretely, rebuilding the image after modifying the application source code can reuse the cached layers, resulting in millisecond builds: 

```
[#__codelineno-6-1](#__codelineno-6-1) => [internal] load build definition from Dockerfile                                                                 0.0s
[#__codelineno-6-2](#__codelineno-6-2) => => transferring dockerfile: 1.31kB                                                                               0.0s
[#__codelineno-6-3](#__codelineno-6-3) => [internal] load metadata for public.ecr.aws/lambda/python:3.13                                                   0.3s
[#__codelineno-6-4](#__codelineno-6-4) => [internal] load metadata for ghcr.io/astral-sh/uv:latest                                                         0.3s
[#__codelineno-6-5](#__codelineno-6-5) => [internal] load .dockerignore                                                                                    0.0s
[#__codelineno-6-6](#__codelineno-6-6) => => transferring context: 106B                                                                                    0.0s
[#__codelineno-6-7](#__codelineno-6-7) => [uv 1/1] FROM ghcr.io/astral-sh/uv:latest@sha256:ea61e006cfec0e8d81fae901ad703e09d2c6cf1aa58abcb6507d124b50286f  0.0s
[#__codelineno-6-8](#__codelineno-6-8) => [builder 1/2] FROM public.ecr.aws/lambda/python:3.13@sha256:f5b51b377b80bd303fe8055084e2763336ea8920d12955b23ef  0.0s
[#__codelineno-6-9](#__codelineno-6-9) => [internal] load build context                                                                                    0.0s
[#__codelineno-6-10](#__codelineno-6-10) => => transferring context: 185B                                                                                    0.0s
[#__codelineno-6-11](#__codelineno-6-11) => CACHED [builder 2/2] RUN --mount=from=uv,source=/uv,target=/bin/uv     --mount=type=cache,target=/root/.cache/u  0.0s
[#__codelineno-6-12](#__codelineno-6-12) => CACHED [stage-2 2/3] COPY --from=builder /var/task /var/task                                                     0.0s
[#__codelineno-6-13](#__codelineno-6-13) => CACHED [stage-2 3/3] COPY ./app /var/task                                                                        0.0s
[#__codelineno-6-14](#__codelineno-6-14) => exporting to image                                                                                               0.0s
[#__codelineno-6-15](#__codelineno-6-15) => => exporting layers                                                                                              0.0s
[#__codelineno-6-16](#__codelineno-6-16) => => writing image sha256:6f8f9ef715a7cda466b677a9df4046ebbb90c8e88595242ade3b4771f547652d                         0.0

```

After building, we can push the image to [Elastic Container Registry (ECR)](https://aws.amazon.com/ecr/)with, e.g.: 

```
[#__codelineno-7-1](#__codelineno-7-1)$ aws ecr get-login-password --region region | docker login --username AWS --password-stdin aws_account_id.dkr.ecr.region.amazonaws.com
[#__codelineno-7-2](#__codelineno-7-2)$ docker tag fastapi-app:latest aws_account_id.dkr.ecr.region.amazonaws.com/fastapi-app:latest
[#__codelineno-7-3](#__codelineno-7-3)$ docker push aws_account_id.dkr.ecr.region.amazonaws.com/fastapi-app:latest

```

Finally, we can deploy the image to AWS Lambda using the AWS Management Console or the AWS CLI, e.g.: 

```
[#__codelineno-8-1](#__codelineno-8-1)$ aws lambda create-function \
[#__codelineno-8-2](#__codelineno-8-2)   --function-name myFunction \
[#__codelineno-8-3](#__codelineno-8-3)   --package-type Image \
[#__codelineno-8-4](#__codelineno-8-4)   --code ImageUri=aws_account_id.dkr.ecr.region.amazonaws.com/fastapi-app:latest \
[#__codelineno-8-5](#__codelineno-8-5)   --role arn:aws:iam::111122223333:role/my-lambda-role

```

Where the [execution role](https://docs.aws.amazon.com/lambda/latest/dg/lambda-intro-execution-role.html#permissions-executionrole-api)is created via: 

```
[#__codelineno-9-1](#__codelineno-9-1)$ aws iam create-role \
[#__codelineno-9-2](#__codelineno-9-2)   --role-name my-lambda-role \
[#__codelineno-9-3](#__codelineno-9-3)   --assume-role-policy-document '{"Version": "2012-10-17", "Statement": [{ "Effect": "Allow", "Principal": {"Service": "lambda.amazonaws.com"}, "Action": "sts:AssumeRole"}]}'

```

Or, update an existing function with: 

```
[#__codelineno-10-1](#__codelineno-10-1)$ aws lambda update-function-code \
[#__codelineno-10-2](#__codelineno-10-2)   --function-name myFunction \
[#__codelineno-10-3](#__codelineno-10-3)   --image-uri aws_account_id.dkr.ecr.region.amazonaws.com/fastapi-app:latest \
[#__codelineno-10-4](#__codelineno-10-4)   --publish

```

To test the Lambda, we can invoke it via the AWS Management Console or the AWS CLI, e.g.: 

```
[#__codelineno-11-1](#__codelineno-11-1)$ aws lambda invoke \
[#__codelineno-11-2](#__codelineno-11-2)   --function-name myFunction \
[#__codelineno-11-3](#__codelineno-11-3)   --payload file://event.json \
[#__codelineno-11-4](#__codelineno-11-4)   --cli-binary-format raw-in-base64-out \
[#__codelineno-11-5](#__codelineno-11-5)   response.json
[#__codelineno-11-6](#__codelineno-11-6){
[#__codelineno-11-7](#__codelineno-11-7)  "StatusCode": 200,
[#__codelineno-11-8](#__codelineno-11-8)  "ExecutedVersion": "$LATEST"
[#__codelineno-11-9](#__codelineno-11-9)}

```

Where `event.json `contains the event payload to pass to the Lambda function: 

event.json 

```
[#__codelineno-12-1](#__codelineno-12-1){
[#__codelineno-12-2](#__codelineno-12-2)  "httpMethod": "GET",
[#__codelineno-12-3](#__codelineno-12-3)  "path": "/",
[#__codelineno-12-4](#__codelineno-12-4)  "requestContext": {},
[#__codelineno-12-5](#__codelineno-12-5)  "version": "1.0"
[#__codelineno-12-6](#__codelineno-12-6)}

```

And `response.json `contains the response from the Lambda function: 

response.json 

```
[#__codelineno-13-1](#__codelineno-13-1){
[#__codelineno-13-2](#__codelineno-13-2)  "statusCode": 200,
[#__codelineno-13-3](#__codelineno-13-3)  "headers": {
[#__codelineno-13-4](#__codelineno-13-4)    "content-length": "14",
[#__codelineno-13-5](#__codelineno-13-5)    "content-type": "application/json"
[#__codelineno-13-6](#__codelineno-13-6)  },
[#__codelineno-13-7](#__codelineno-13-7)  "multiValueHeaders": {},
[#__codelineno-13-8](#__codelineno-13-8)  "body": "\"Hello, world!\"",
[#__codelineno-13-9](#__codelineno-13-9)  "isBase64Encoded": false
[#__codelineno-13-10](#__codelineno-13-10)}

```

For details, see the [AWS Lambda documentation](https://docs.aws.amazon.com/lambda/latest/dg/python-image.html). 

### [Workspace support](#workspace-support)

If a project includes local dependencies (e.g., via [Workspaces](../../../concepts/projects/workspaces/)), those too must be included in the deployment package. 

We'll start by extending the above example to include a dependency on a locally-developed library named `library `. 

First, we'll create the library itself: 

```
[#__codelineno-14-1](#__codelineno-14-1)$ uv init --lib library
[#__codelineno-14-2](#__codelineno-14-2)$ uv add ./library

```

Running `uv init `within the `project `directory will automatically convert `project `to a workspace and add `library `as a workspace member: 

pyproject.toml 

```
[#__codelineno-15-1](#__codelineno-15-1)[project]
[#__codelineno-15-2](#__codelineno-15-2)name = "uv-aws-lambda-example"
[#__codelineno-15-3](#__codelineno-15-3)version = "0.1.0"
[#__codelineno-15-4](#__codelineno-15-4)requires-python = ">=3.13"
[#__codelineno-15-5](#__codelineno-15-5)dependencies = [
[#__codelineno-15-6](#__codelineno-15-6)    # FastAPI is a modern web framework for building APIs with Python.
[#__codelineno-15-7](#__codelineno-15-7)    "fastapi",
[#__codelineno-15-8](#__codelineno-15-8)    # A local library.
[#__codelineno-15-9](#__codelineno-15-9)    "library",
[#__codelineno-15-10](#__codelineno-15-10)    # Mangum is a library that adapts ASGI applications to AWS Lambda and API Gateway.
[#__codelineno-15-11](#__codelineno-15-11)    "mangum",
[#__codelineno-15-12](#__codelineno-15-12)]
[#__codelineno-15-13](#__codelineno-15-13)
[#__codelineno-15-14](#__codelineno-15-14)[dependency-groups]
[#__codelineno-15-15](#__codelineno-15-15)dev = [
[#__codelineno-15-16](#__codelineno-15-16)    # In development mode, include the FastAPI development server.
[#__codelineno-15-17](#__codelineno-15-17)    "fastapi[standard]",
[#__codelineno-15-18](#__codelineno-15-18)]
[#__codelineno-15-19](#__codelineno-15-19)
[#__codelineno-15-20](#__codelineno-15-20)[tool.uv.workspace]
[#__codelineno-15-21](#__codelineno-15-21)members = ["library"]
[#__codelineno-15-22](#__codelineno-15-22)
[#__codelineno-15-23](#__codelineno-15-23)[tool.uv.sources]
[#__codelineno-15-24](#__codelineno-15-24)lib = { workspace = true }

```

By default, `uv init --lib `will create a package that exports a `hello `function. We'll modify the application source code to call that function: 

app/main.py 

```
[#__codelineno-16-1](#__codelineno-16-1)import logging
[#__codelineno-16-2](#__codelineno-16-2)
[#__codelineno-16-3](#__codelineno-16-3)from fastapi import FastAPI
[#__codelineno-16-4](#__codelineno-16-4)from mangum import Mangum
[#__codelineno-16-5](#__codelineno-16-5)
[#__codelineno-16-6](#__codelineno-16-6)from library import hello
[#__codelineno-16-7](#__codelineno-16-7)
[#__codelineno-16-8](#__codelineno-16-8)logger = logging.getLogger()
[#__codelineno-16-9](#__codelineno-16-9)logger.setLevel(logging.INFO)
[#__codelineno-16-10](#__codelineno-16-10)
[#__codelineno-16-11](#__codelineno-16-11)app = FastAPI()
[#__codelineno-16-12](#__codelineno-16-12)handler = Mangum(app)
[#__codelineno-16-13](#__codelineno-16-13)
[#__codelineno-16-14](#__codelineno-16-14)
[#__codelineno-16-15](#__codelineno-16-15)@app.get("/")
[#__codelineno-16-16](#__codelineno-16-16)async def root() -> str:
[#__codelineno-16-17](#__codelineno-16-17)    return hello()

```

We can run the modified application locally with: 

```
[#__codelineno-17-1](#__codelineno-17-1)$ uv run fastapi dev

```

And confirm that opening [http://127.0.0.1:8000/](http://127.0.0.1:8000/)in a web browser displays, "Hello from library!" (instead of "Hello, World!") 

Finally, we'll update the Dockerfile to include the local library in the deployment package: 

Dockerfile 

```
[#__codelineno-18-1](#__codelineno-18-1)FROM ghcr.io/astral-sh/uv:0.9.17 AS uv
[#__codelineno-18-2](#__codelineno-18-2)
[#__codelineno-18-3](#__codelineno-18-3)# First, bundle the dependencies into the task root.
[#__codelineno-18-4](#__codelineno-18-4)FROM public.ecr.aws/lambda/python:3.13 AS builder
[#__codelineno-18-5](#__codelineno-18-5)
[#__codelineno-18-6](#__codelineno-18-6)# Enable bytecode compilation, to improve cold-start performance.
[#__codelineno-18-7](#__codelineno-18-7)ENV UV_COMPILE_BYTECODE=1
[#__codelineno-18-8](#__codelineno-18-8)
[#__codelineno-18-9](#__codelineno-18-9)# Disable installer metadata, to create a deterministic layer.
[#__codelineno-18-10](#__codelineno-18-10)ENV UV_NO_INSTALLER_METADATA=1
[#__codelineno-18-11](#__codelineno-18-11)
[#__codelineno-18-12](#__codelineno-18-12)# Enable copy mode to support bind mount caching.
[#__codelineno-18-13](#__codelineno-18-13)ENV UV_LINK_MODE=copy
[#__codelineno-18-14](#__codelineno-18-14)
[#__codelineno-18-15](#__codelineno-18-15)# Bundle the dependencies into the Lambda task root via `uv pip install --target`.
[#__codelineno-18-16](#__codelineno-18-16)#
[#__codelineno-18-17](#__codelineno-18-17)# Omit any local packages (`--no-emit-workspace`) and development dependencies (`--no-dev`).
[#__codelineno-18-18](#__codelineno-18-18)# This ensures that the Docker layer cache is only invalidated when the `pyproject.toml` or `uv.lock`
[#__codelineno-18-19](#__codelineno-18-19)# files change, but remains robust to changes in the application code.
[#__codelineno-18-20](#__codelineno-18-20)RUN --mount=from=uv,source=/uv,target=/bin/uv \
[#__codelineno-18-21](#__codelineno-18-21)    --mount=type=cache,target=/root/.cache/uv \
[#__codelineno-18-22](#__codelineno-18-22)    --mount=type=bind,source=uv.lock,target=uv.lock \
[#__codelineno-18-23](#__codelineno-18-23)    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
[#__codelineno-18-24](#__codelineno-18-24)    uv export --frozen --no-emit-workspace --no-dev --no-editable -o requirements.txt && \
[#__codelineno-18-25](#__codelineno-18-25)    uv pip install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"
[#__codelineno-18-26](#__codelineno-18-26)
[#__codelineno-18-27](#__codelineno-18-27)# If you have a workspace, copy it over and install it too.
[#__codelineno-18-28](#__codelineno-18-28)#
[#__codelineno-18-29](#__codelineno-18-29)# By omitting `--no-emit-workspace`, `library` will be copied into the task root. Using a separate
[#__codelineno-18-30](#__codelineno-18-30)# `RUN` command ensures that all third-party dependencies are cached separately and remain
[#__codelineno-18-31](#__codelineno-18-31)# robust to changes in the workspace.
[#__codelineno-18-32](#__codelineno-18-32)RUN --mount=from=uv,source=/uv,target=/bin/uv \
[#__codelineno-18-33](#__codelineno-18-33)    --mount=type=cache,target=/root/.cache/uv \
[#__codelineno-18-34](#__codelineno-18-34)    --mount=type=bind,source=uv.lock,target=uv.lock \
[#__codelineno-18-35](#__codelineno-18-35)    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
[#__codelineno-18-36](#__codelineno-18-36)    --mount=type=bind,source=library,target=library \
[#__codelineno-18-37](#__codelineno-18-37)    uv export --frozen --no-dev --no-editable -o requirements.txt && \
[#__codelineno-18-38](#__codelineno-18-38)    uv pip install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"
[#__codelineno-18-39](#__codelineno-18-39)
[#__codelineno-18-40](#__codelineno-18-40)FROM public.ecr.aws/lambda/python:3.13
[#__codelineno-18-41](#__codelineno-18-41)
[#__codelineno-18-42](#__codelineno-18-42)# Copy the runtime dependencies from the builder stage.
[#__codelineno-18-43](#__codelineno-18-43)COPY --from=builder ${LAMBDA_TASK_ROOT} ${LAMBDA_TASK_ROOT}
[#__codelineno-18-44](#__codelineno-18-44)
[#__codelineno-18-45](#__codelineno-18-45)# Copy the application code.
[#__codelineno-18-46](#__codelineno-18-46)COPY ./app ${LAMBDA_TASK_ROOT}/app
[#__codelineno-18-47](#__codelineno-18-47)
[#__codelineno-18-48](#__codelineno-18-48)# Set the AWS Lambda handler.
[#__codelineno-18-49](#__codelineno-18-49)CMD ["app.main.handler"]

```

!!! tip "Tip"

    To deploy to ARM-based AWS Lambda runtimes, replace `public.ecr.aws/lambda/python:3.13 `with `public.ecr.aws/lambda/python:3.13-arm64 `. 

From there, we can build and deploy the updated image as before. 

## [Deploying a zip archive](#deploying-a-zip-archive)

AWS Lambda also supports deployment via zip archives. For simple applications, zip archives can be a more straightforward and efficient deployment method than Docker images; however, zip archives are limited to [250 MB](https://docs.aws.amazon.com/lambda/latest/dg/python-package.html#python-package-create-update)in size. 

Returning to the FastAPI example, we can bundle the application dependencies into a local directory for AWS Lambda via: 

```
[#__codelineno-19-1](#__codelineno-19-1)$ uv export --frozen --no-dev --no-editable -o requirements.txt
[#__codelineno-19-2](#__codelineno-19-2)$ uv pip install \
[#__codelineno-19-3](#__codelineno-19-3)   --no-installer-metadata \
[#__codelineno-19-4](#__codelineno-19-4)   --no-compile-bytecode \
[#__codelineno-19-5](#__codelineno-19-5)   --python-platform x86_64-manylinux2014 \
[#__codelineno-19-6](#__codelineno-19-6)   --python 3.13 \
[#__codelineno-19-7](#__codelineno-19-7)   --target packages \
[#__codelineno-19-8](#__codelineno-19-8)   -r requirements.txt

```

!!! tip "Tip"

    To deploy to ARM-based AWS Lambda runtimes, replace `x86_64-manylinux2014 `with `aarch64-manylinux2014 `. 

Following the [AWS Lambda documentation](https://docs.aws.amazon.com/lambda/latest/dg/python-package.html), we can then bundle these dependencies into a zip as follows: 

```
[#__codelineno-20-1](#__codelineno-20-1)$ cd packages
[#__codelineno-20-2](#__codelineno-20-2)$ zip -r ../package.zip .
[#__codelineno-20-3](#__codelineno-20-3)$ cd ..

```

Finally, we can add the application code to the zip archive: 

```
[#__codelineno-21-1](#__codelineno-21-1)$ zip -r package.zip app

```

We can then deploy the zip archive to AWS Lambda via the AWS Management Console or the AWS CLI, e.g.: 

```
[#__codelineno-22-1](#__codelineno-22-1)$ aws lambda create-function \
[#__codelineno-22-2](#__codelineno-22-2)   --function-name myFunction \
[#__codelineno-22-3](#__codelineno-22-3)   --runtime python3.13 \
[#__codelineno-22-4](#__codelineno-22-4)   --zip-file fileb://package.zip \
[#__codelineno-22-5](#__codelineno-22-5)   --handler app.main.handler \
[#__codelineno-22-6](#__codelineno-22-6)   --role arn:aws:iam::111122223333:role/service-role/my-lambda-role

```

Where the [execution role](https://docs.aws.amazon.com/lambda/latest/dg/lambda-intro-execution-role.html#permissions-executionrole-api)is created via: 

```
[#__codelineno-23-1](#__codelineno-23-1)$ aws iam create-role \
[#__codelineno-23-2](#__codelineno-23-2)   --role-name my-lambda-role \
[#__codelineno-23-3](#__codelineno-23-3)   --assume-role-policy-document '{"Version": "2012-10-17", "Statement": [{ "Effect": "Allow", "Principal": {"Service": "lambda.amazonaws.com"}, "Action": "sts:AssumeRole"}]}'

```

Or, update an existing function with: 

```
[#__codelineno-24-1](#__codelineno-24-1)$ aws lambda update-function-code \
[#__codelineno-24-2](#__codelineno-24-2)   --function-name myFunction \
[#__codelineno-24-3](#__codelineno-24-3)   --zip-file fileb://package.zip

```

!!! note "Note"

    By default, the AWS Management Console assumes a Lambda entrypoint of `lambda_function.lambda_handler `. If your application uses a different entrypoint, you'll need to modify it in the AWS Management Console. For example, the above FastAPI application uses `app.main.handler `. 

To test the Lambda, we can invoke it via the AWS Management Console or the AWS CLI, e.g.: 

```
[#__codelineno-25-1](#__codelineno-25-1)$ aws lambda invoke \
[#__codelineno-25-2](#__codelineno-25-2)   --function-name myFunction \
[#__codelineno-25-3](#__codelineno-25-3)   --payload file://event.json \
[#__codelineno-25-4](#__codelineno-25-4)   --cli-binary-format raw-in-base64-out \
[#__codelineno-25-5](#__codelineno-25-5)   response.json
[#__codelineno-25-6](#__codelineno-25-6){
[#__codelineno-25-7](#__codelineno-25-7)  "StatusCode": 200,
[#__codelineno-25-8](#__codelineno-25-8)  "ExecutedVersion": "$LATEST"
[#__codelineno-25-9](#__codelineno-25-9)}

```

Where `event.json `contains the event payload to pass to the Lambda function: 

event.json 

```
[#__codelineno-26-1](#__codelineno-26-1){
[#__codelineno-26-2](#__codelineno-26-2)  "httpMethod": "GET",
[#__codelineno-26-3](#__codelineno-26-3)  "path": "/",
[#__codelineno-26-4](#__codelineno-26-4)  "requestContext": {},
[#__codelineno-26-5](#__codelineno-26-5)  "version": "1.0"
[#__codelineno-26-6](#__codelineno-26-6)}

```

And `response.json `contains the response from the Lambda function: 

response.json 

```
[#__codelineno-27-1](#__codelineno-27-1){
[#__codelineno-27-2](#__codelineno-27-2)  "statusCode": 200,
[#__codelineno-27-3](#__codelineno-27-3)  "headers": {
[#__codelineno-27-4](#__codelineno-27-4)    "content-length": "14",
[#__codelineno-27-5](#__codelineno-27-5)    "content-type": "application/json"
[#__codelineno-27-6](#__codelineno-27-6)  },
[#__codelineno-27-7](#__codelineno-27-7)  "multiValueHeaders": {},
[#__codelineno-27-8](#__codelineno-27-8)  "body": "\"Hello, world!\"",
[#__codelineno-27-9](#__codelineno-27-9)  "isBase64Encoded": false
[#__codelineno-27-10](#__codelineno-27-10)}

```

### [Using a Lambda layer](#using-a-lambda-layer)

AWS Lambda also supports the deployment of multiple composed [Lambda layers](https://docs.aws.amazon.com/lambda/latest/dg/python-layers.html)when working with zip archives. These layers are conceptually similar to layers in a Docker image, allowing you to separate application code from dependencies. 

In particular, we can create a lambda layer for application dependencies and attach it to the Lambda function, separate from the application code itself. This setup can improve cold-start performance for application updates, as the dependencies layer can be reused across deployments. 

To create a Lambda layer, we'll follow similar steps, but create two separate zip archives: one for the application code and one for the application dependencies. 

First, we'll create the dependency layer. Lambda layers are expected to follow a slightly different structure, so we'll use `--prefix `rather than `--target `: 

```
[#__codelineno-28-1](#__codelineno-28-1)$ uv export --frozen --no-dev --no-editable -o requirements.txt
[#__codelineno-28-2](#__codelineno-28-2)$ uv pip install \
[#__codelineno-28-3](#__codelineno-28-3)   --no-installer-metadata \
[#__codelineno-28-4](#__codelineno-28-4)   --no-compile-bytecode \
[#__codelineno-28-5](#__codelineno-28-5)   --python-platform x86_64-manylinux2014 \
[#__codelineno-28-6](#__codelineno-28-6)   --python 3.13 \
[#__codelineno-28-7](#__codelineno-28-7)   --prefix packages \
[#__codelineno-28-8](#__codelineno-28-8)   -r requirements.txt

```

We'll then zip the dependencies in adherence with the expected layout for Lambda layers: 

```
[#__codelineno-29-1](#__codelineno-29-1)$ mkdir python
[#__codelineno-29-2](#__codelineno-29-2)$ cp -r packages/lib python/
[#__codelineno-29-3](#__codelineno-29-3)$ zip -r layer_content.zip python

```

!!! tip "Tip"

    To generate deterministic zip archives, consider passing the `-X `flag to `zip `to exclude extended attributes and file system metadata. 

And publish the Lambda layer: 

```
[#__codelineno-30-1](#__codelineno-30-1)$ aws lambda publish-layer-version --layer-name dependencies-layer \
[#__codelineno-30-2](#__codelineno-30-2)   --zip-file fileb://layer_content.zip \
[#__codelineno-30-3](#__codelineno-30-3)   --compatible-runtimes python3.13 \
[#__codelineno-30-4](#__codelineno-30-4)   --compatible-architectures "x86_64"

```

We can then create the Lambda function as in the previous example, omitting the dependencies: 

```
[#__codelineno-31-1](#__codelineno-31-1)$ # Zip the application code.
[#__codelineno-31-2](#__codelineno-31-2)$ zip -r app.zip app
[#__codelineno-31-3](#__codelineno-31-3)
[#__codelineno-31-4](#__codelineno-31-4)$ # Create the Lambda function.
[#__codelineno-31-5](#__codelineno-31-5)$ aws lambda create-function \
[#__codelineno-31-6](#__codelineno-31-6)   --function-name myFunction \
[#__codelineno-31-7](#__codelineno-31-7)   --runtime python3.13 \
[#__codelineno-31-8](#__codelineno-31-8)   --zip-file fileb://app.zip \
[#__codelineno-31-9](#__codelineno-31-9)   --handler app.main.handler \
[#__codelineno-31-10](#__codelineno-31-10)   --role arn:aws:iam::111122223333:role/service-role/my-lambda-role

```

Finally, we can attach the dependencies layer to the Lambda function, using the ARN returned by the `publish-layer-version `step: 

```
[#__codelineno-32-1](#__codelineno-32-1)$ aws lambda update-function-configuration --function-name myFunction \
[#__codelineno-32-2](#__codelineno-32-2)    --cli-binary-format raw-in-base64-out \
[#__codelineno-32-3](#__codelineno-32-3)    --layers "arn:aws:lambda:region:111122223333:layer:dependencies-layer:1"

```

When the application dependencies change, the layer can be updated independently of the application by republishing the layer and updating the Lambda function configuration: 

```
[#__codelineno-33-1](#__codelineno-33-1)$ # Update the dependencies in the layer.
[#__codelineno-33-2](#__codelineno-33-2)$ aws lambda publish-layer-version --layer-name dependencies-layer \
[#__codelineno-33-3](#__codelineno-33-3)   --zip-file fileb://layer_content.zip \
[#__codelineno-33-4](#__codelineno-33-4)   --compatible-runtimes python3.13 \
[#__codelineno-33-5](#__codelineno-33-5)   --compatible-architectures "x86_64"
[#__codelineno-33-6](#__codelineno-33-6)
[#__codelineno-33-7](#__codelineno-33-7)$ # Update the Lambda function configuration.
[#__codelineno-33-8](#__codelineno-33-8)$ aws lambda update-function-configuration --function-name myFunction \
[#__codelineno-33-9](#__codelineno-33-9)    --cli-binary-format raw-in-base64-out \
[#__codelineno-33-10](#__codelineno-33-10)    --layers "arn:aws:lambda:region:111122223333:layer:dependencies-layer:2"

```

December 9, 2025
