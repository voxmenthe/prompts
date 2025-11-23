# [Using alternative package indexes](#using-alternative-package-indexes)

While uv uses the official Python Package Index (PyPI) by default, it also supports [alternative package indexes](../../../concepts/indexes/). Most alternative indexes require various forms of authentication, which require some initial setup. 

!!! important "Important"

    If using the pip interface, please read the documentation on [using multiple indexes](../../../pip/compatibility/#packages-that-exist-on-multiple-indexes)in uv — the default behavior is different from pip to prevent dependency confusion attacks, but this means that uv may not find the versions of a package as you'd expect. 

## [Azure Artifacts](#azure-artifacts)

uv can install packages from [Azure Artifacts](https://learn.microsoft.com/en-us/azure/devops/artifacts/start-using-azure-artifacts?view=azure-devops&tabs=nuget%2Cnugetserver), either by using a [Personal Access Token](https://learn.microsoft.com/en-us/azure/devops/organizations/accounts/use-personal-access-tokens-to-authenticate?view=azure-devops&tabs=Windows)(PAT), or using the [`keyring `](https://github.com/jaraco/keyring)package. 

To use Azure Artifacts, add the index to your project: 

pyproject.toml 

```
[#__codelineno-0-1](#__codelineno-0-1)[[tool.uv.index]]
[#__codelineno-0-2](#__codelineno-0-2)name = "private-registry"
[#__codelineno-0-3](#__codelineno-0-3)url = "https://pkgs.dev.azure.com///_packaging//pypi/simple/"

```

### [Authenticate with an Azure access token](#authenticate-with-an-azure-access-token)

If there is a personal access token (PAT) available (e.g., [`$(System.AccessToken) `in an Azure pipeline](https://learn.microsoft.com/en-us/azure/devops/pipelines/build/variables?view=azure-devops&tabs=yaml#systemaccesstoken)), credentials can be provided via "Basic" HTTP authentication scheme. Include the PAT in the password field of the URL. A username must be included as well, but can be any string. 

For example, with the token stored in the `$AZURE_ARTIFACTS_TOKEN `environment variable, set credentials for the index with: 

```
[#__codelineno-1-1](#__codelineno-1-1)export UV_INDEX_PRIVATE_REGISTRY_USERNAME=dummy
[#__codelineno-1-2](#__codelineno-1-2)export UV_INDEX_PRIVATE_REGISTRY_PASSWORD="$AZURE_ARTIFACTS_TOKEN"

```

!!! note "Note"

    `PRIVATE_REGISTRY `should match the name of the index defined in your `pyproject.toml `. 

### [Authenticate with `keyring `and `artifacts-keyring `](#authenticate-with-keyring-and-artifacts-keyring)

You can also authenticate to Artifacts using [`keyring `](https://github.com/jaraco/keyring)package with the [`artifacts-keyring `plugin](https://github.com/Microsoft/artifacts-keyring). Because these two packages are required to authenticate to Azure Artifacts, they must be pre-installed from a source other than Artifacts. 

The `artifacts-keyring `plugin wraps the [Azure Artifacts Credential Provider tool](https://github.com/microsoft/artifacts-credprovider). The credential provider supports a few different authentication modes including interactive login — see the [tool's documentation](https://github.com/microsoft/artifacts-credprovider)for information on configuration. 

uv only supports using the `keyring `package in [subprocess mode](../../../reference/settings/#keyring-provider). The `keyring `executable must be in the `PATH `, i.e., installed globally or in the active environment. The `keyring `CLI requires a username in the URL, and it must be `VssSessionToken `. 

```
[#__codelineno-2-1](#__codelineno-2-1)# Pre-install keyring and the Artifacts plugin from the public PyPI
[#__codelineno-2-2](#__codelineno-2-2)uv tool install keyring --with artifacts-keyring
[#__codelineno-2-3](#__codelineno-2-3)
[#__codelineno-2-4](#__codelineno-2-4)# Enable keyring authentication
[#__codelineno-2-5](#__codelineno-2-5)export UV_KEYRING_PROVIDER=subprocess
[#__codelineno-2-6](#__codelineno-2-6)
[#__codelineno-2-7](#__codelineno-2-7)# Set the username for the index
[#__codelineno-2-8](#__codelineno-2-8)export UV_INDEX_PRIVATE_REGISTRY_USERNAME=VssSessionToken

```

!!! note "Note"

    The [`tool.uv.keyring-provider `](../../../reference/settings/#keyring-provider)setting can be used to enable keyring in your `uv.toml `or `pyproject.toml `. 

    Similarly, the username for the index can be added directly to the index URL. 

### [Publishing packages to Azure Artifacts](#publishing-packages-to-azure-artifacts)

If you also want to publish your own packages to Azure Artifacts, you can use `uv publish `as described in the [Building and publishing guide](../../package/). 

First, add a `publish-url `to the index you want to publish packages to. For example: 

pyproject.toml 

```
[#__codelineno-3-1](#__codelineno-3-1)[[tool.uv.index]]
[#__codelineno-3-2](#__codelineno-3-2)name = "private-registry"
[#__codelineno-3-3](#__codelineno-3-3)url = "https://pkgs.dev.azure.com///_packaging//pypi/simple/"
[#__codelineno-3-4](#__codelineno-3-4)publish-url = "https://pkgs.dev.azure.com///_packaging//pypi/upload/"

```

Then, configure credentials (if not using keyring): 

```
[#__codelineno-4-1](#__codelineno-4-1)$ export UV_PUBLISH_USERNAME=dummy
[#__codelineno-4-2](#__codelineno-4-2)$ export UV_PUBLISH_PASSWORD="$AZURE_ARTIFACTS_TOKEN"

```

And publish the package: 

```
[#__codelineno-5-1](#__codelineno-5-1)$ uv publish --index private-registry

```

To use `uv publish `without adding the `publish-url `to the project, you can set `UV_PUBLISH_URL `: 

```
[#__codelineno-6-1](#__codelineno-6-1)$ export UV_PUBLISH_URL=https://pkgs.dev.azure.com///_packaging//pypi/upload/
[#__codelineno-6-2](#__codelineno-6-2)$ uv publish

```

Note this method is not preferable because uv cannot check if the package is already published before uploading artifacts. 

## [Google Artifact Registry](#google-artifact-registry)

uv can install packages from [Google Artifact Registry](https://cloud.google.com/artifact-registry/docs), either by using an access token, or using the [`keyring `](https://github.com/jaraco/keyring)package. 

!!! note "Note"

    This guide assumes that [`gcloud `](https://cloud.google.com/sdk/gcloud)CLI is installed and authenticated. 

To use Google Artifact Registry, add the index to your project: 

pyproject.toml 

```
[#__codelineno-7-1](#__codelineno-7-1)[[tool.uv.index]]
[#__codelineno-7-2](#__codelineno-7-2)name = "private-registry"
[#__codelineno-7-3](#__codelineno-7-3)url = "https://-python.pkg.dev///simple/"

```

### [Authenticate with a Google access token](#authenticate-with-a-google-access-token)

Credentials can be provided via "Basic" HTTP authentication scheme. Include access token in the password field of the URL. Username must be `oauth2accesstoken `, otherwise authentication will fail. 

Generate a token with `gcloud `: 

```
[#__codelineno-8-1](#__codelineno-8-1)export ARTIFACT_REGISTRY_TOKEN=$(
[#__codelineno-8-2](#__codelineno-8-2)    gcloud auth application-default print-access-token
[#__codelineno-8-3](#__codelineno-8-3))

```

!!! note "Note"

    You might need to pass extra parameters to properly generate the token (like `--project `), this is a basic example. 

Then set credentials for the index with: 

```
[#__codelineno-9-1](#__codelineno-9-1)export UV_INDEX_PRIVATE_REGISTRY_USERNAME=oauth2accesstoken
[#__codelineno-9-2](#__codelineno-9-2)export UV_INDEX_PRIVATE_REGISTRY_PASSWORD="$ARTIFACT_REGISTRY_TOKEN"

```

!!! note "Note"

    `PRIVATE_REGISTRY `should match the name of the index defined in your `pyproject.toml `. 

### [Authenticate with `keyring `and `keyrings.google-artifactregistry-auth `](#authenticate-with-keyring-and-keyringsgoogle-artifactregistry-auth)

You can also authenticate to Artifact Registry using [`keyring `](https://github.com/jaraco/keyring)package with the [`keyrings.google-artifactregistry-auth `plugin](https://github.com/GoogleCloudPlatform/artifact-registry-python-tools). Because these two packages are required to authenticate to Artifact Registry, they must be pre-installed from a source other than Artifact Registry. 

The `keyrings.google-artifactregistry-auth `plugin wraps [gcloud CLI](https://cloud.google.com/sdk/gcloud)to generate short-lived access tokens, securely store them in system keyring, and refresh them when they are expired. 

uv only supports using the `keyring `package in [subprocess mode](../../../reference/settings/#keyring-provider). The `keyring `executable must be in the `PATH `, i.e., installed globally or in the active environment. The `keyring `CLI requires a username in the URL and it must be `oauth2accesstoken `. 

```
[#__codelineno-10-1](#__codelineno-10-1)# Pre-install keyring and Artifact Registry plugin from the public PyPI
[#__codelineno-10-2](#__codelineno-10-2)uv tool install keyring --with keyrings.google-artifactregistry-auth
[#__codelineno-10-3](#__codelineno-10-3)
[#__codelineno-10-4](#__codelineno-10-4)# Enable keyring authentication
[#__codelineno-10-5](#__codelineno-10-5)export UV_KEYRING_PROVIDER=subprocess
[#__codelineno-10-6](#__codelineno-10-6)
[#__codelineno-10-7](#__codelineno-10-7)# Set the username for the index
[#__codelineno-10-8](#__codelineno-10-8)export UV_INDEX_PRIVATE_REGISTRY_USERNAME=oauth2accesstoken

```

!!! note "Note"

    The [`tool.uv.keyring-provider `](../../../reference/settings/#keyring-provider)setting can be used to enable keyring in your `uv.toml `or `pyproject.toml `. 

    Similarly, the username for the index can be added directly to the index URL. 

### [Publishing packages to Google Artifact Registry](#publishing-packages-to-google-artifact-registry)

If you also want to publish your own packages to Google Artifact Registry, you can use `uv publish `as described in the [Building and publishing guide](../../package/). 

First, add a `publish-url `to the index you want to publish packages to. For example: 

pyproject.toml 

```
[#__codelineno-11-1](#__codelineno-11-1)[[tool.uv.index]]
[#__codelineno-11-2](#__codelineno-11-2)name = "private-registry"
[#__codelineno-11-3](#__codelineno-11-3)url = "https://-python.pkg.dev///simple/"
[#__codelineno-11-4](#__codelineno-11-4)publish-url = "https://-python.pkg.dev///"

```

Then, configure credentials (if not using keyring): 

```
[#__codelineno-12-1](#__codelineno-12-1)$ export UV_PUBLISH_USERNAME=oauth2accesstoken
[#__codelineno-12-2](#__codelineno-12-2)$ export UV_PUBLISH_PASSWORD="$ARTIFACT_REGISTRY_TOKEN"

```

And publish the package: 

```
[#__codelineno-13-1](#__codelineno-13-1)$ uv publish --index private-registry

```

To use `uv publish `without adding the `publish-url `to the project, you can set `UV_PUBLISH_URL `: 

```
[#__codelineno-14-1](#__codelineno-14-1)$ export UV_PUBLISH_URL=https://-python.pkg.dev///
[#__codelineno-14-2](#__codelineno-14-2)$ uv publish

```

Note this method is not preferable because uv cannot check if the package is already published before uploading artifacts. 

## [AWS CodeArtifact](#aws-codeartifact)

uv can install packages from [AWS CodeArtifact](https://docs.aws.amazon.com/codeartifact/latest/ug/using-python.html), either by using an access token, or using the [`keyring `](https://github.com/jaraco/keyring)package. 

!!! note "Note"

    This guide assumes that [`awscli `](https://aws.amazon.com/cli/)is installed and authenticated. 

The index can be declared like so: 

pyproject.toml 

```
[#__codelineno-15-1](#__codelineno-15-1)[[tool.uv.index]]
[#__codelineno-15-2](#__codelineno-15-2)name = "private-registry"
[#__codelineno-15-3](#__codelineno-15-3)url = "https://-.d.codeartifact..amazonaws.com/pypi//simple/"

```

### [Authenticate with an AWS access token](#authenticate-with-an-aws-access-token)

Credentials can be provided via "Basic" HTTP authentication scheme. Include access token in the password field of the URL. Username must be `aws `, otherwise authentication will fail. 

Generate a token with `awscli `: 

```
[#__codelineno-16-1](#__codelineno-16-1)export AWS_CODEARTIFACT_TOKEN="$(
[#__codelineno-16-2](#__codelineno-16-2)    aws codeartifact get-authorization-token \
[#__codelineno-16-3](#__codelineno-16-3)    --domain  \
[#__codelineno-16-4](#__codelineno-16-4)    --domain-owner  \
[#__codelineno-16-5](#__codelineno-16-5)    --query authorizationToken \
[#__codelineno-16-6](#__codelineno-16-6)    --output text
[#__codelineno-16-7](#__codelineno-16-7))"

```

!!! note "Note"

    You might need to pass extra parameters to properly generate the token (like `--region `), this is a basic example. 

Then set credentials for the index with: 

```
[#__codelineno-17-1](#__codelineno-17-1)export UV_INDEX_PRIVATE_REGISTRY_USERNAME=aws
[#__codelineno-17-2](#__codelineno-17-2)export UV_INDEX_PRIVATE_REGISTRY_PASSWORD="$AWS_CODEARTIFACT_TOKEN"

```

!!! note "Note"

    `PRIVATE_REGISTRY `should match the name of the index defined in your `pyproject.toml `. 

### [Authenticate with `keyring `and `keyrings.codeartifact `](#authenticate-with-keyring-and-keyringscodeartifact)

You can also authenticate to Artifact Registry using [`keyring `](https://github.com/jaraco/keyring)package with the [`keyrings.codeartifact `plugin](https://github.com/jmkeyes/keyrings.codeartifact). Because these two packages are required to authenticate to Artifact Registry, they must be pre-installed from a source other than Artifact Registry. 

The `keyrings.codeartifact `plugin wraps [boto3](https://pypi.org/project/boto3/)to generate short-lived access tokens, securely store them in system keyring, and refresh them when they are expired. 

uv only supports using the `keyring `package in [subprocess mode](../../../reference/settings/#keyring-provider). The `keyring `executable must be in the `PATH `, i.e., installed globally or in the active environment. The `keyring `CLI requires a username in the URL and it must be `aws `. 

```
[#__codelineno-18-1](#__codelineno-18-1)# Pre-install keyring and AWS CodeArtifact plugin from the public PyPI
[#__codelineno-18-2](#__codelineno-18-2)uv tool install keyring --with keyrings.codeartifact
[#__codelineno-18-3](#__codelineno-18-3)
[#__codelineno-18-4](#__codelineno-18-4)# Enable keyring authentication
[#__codelineno-18-5](#__codelineno-18-5)export UV_KEYRING_PROVIDER=subprocess
[#__codelineno-18-6](#__codelineno-18-6)
[#__codelineno-18-7](#__codelineno-18-7)# Set the username for the index
[#__codelineno-18-8](#__codelineno-18-8)export UV_INDEX_PRIVATE_REGISTRY_USERNAME=aws

```

!!! note "Note"

    The [`tool.uv.keyring-provider `](../../../reference/settings/#keyring-provider)setting can be used to enable keyring in your `uv.toml `or `pyproject.toml `. 

    Similarly, the username for the index can be added directly to the index URL. 

### [Publishing packages to AWS CodeArtifact](#publishing-packages-to-aws-codeartifact)

If you also want to publish your own packages to AWS CodeArtifact, you can use `uv publish `as described in the [Building and publishing guide](../../package/). 

First, add a `publish-url `to the index you want to publish packages to. For example: 

pyproject.toml 

```
[#__codelineno-19-1](#__codelineno-19-1)[[tool.uv.index]]
[#__codelineno-19-2](#__codelineno-19-2)name = "private-registry"
[#__codelineno-19-3](#__codelineno-19-3)url = "https://-.d.codeartifact..amazonaws.com/pypi//simple/"
[#__codelineno-19-4](#__codelineno-19-4)publish-url = "https://-.d.codeartifact..amazonaws.com/pypi//"

```

Then, configure credentials (if not using keyring): 

```
[#__codelineno-20-1](#__codelineno-20-1)$ export UV_PUBLISH_USERNAME=aws
[#__codelineno-20-2](#__codelineno-20-2)$ export UV_PUBLISH_PASSWORD="$AWS_CODEARTIFACT_TOKEN"

```

And publish the package: 

```
[#__codelineno-21-1](#__codelineno-21-1)$ uv publish --index private-registry

```

To use `uv publish `without adding the `publish-url `to the project, you can set `UV_PUBLISH_URL `: 

```
[#__codelineno-22-1](#__codelineno-22-1)$ export UV_PUBLISH_URL=https://-.d.codeartifact..amazonaws.com/pypi//
[#__codelineno-22-2](#__codelineno-22-2)$ uv publish

```

Note this method is not preferable because uv cannot check if the package is already published before uploading artifacts. 

## [JFrog Artifactory](#jfrog-artifactory)

uv can install packages from JFrog Artifactory, either by using a username and password or a JWT token. 

To use it, add the index to your project: 

pyproject.toml 

```
[#__codelineno-23-1](#__codelineno-23-1)[[tool.uv.index]]
[#__codelineno-23-2](#__codelineno-23-2)name = "private-registry"
[#__codelineno-23-3](#__codelineno-23-3)url = "https://.jfrog.io/artifactory/api/pypi//simple"

```

### [Authenticate with username and password](#authenticate-with-username-and-password)

```
[#__codelineno-24-1](#__codelineno-24-1)$ export UV_INDEX_PRIVATE_REGISTRY_USERNAME=""
[#__codelineno-24-2](#__codelineno-24-2)$ export UV_INDEX_PRIVATE_REGISTRY_PASSWORD=""

```

### [Authenticate with JWT token](#authenticate-with-jwt-token)

```
[#__codelineno-25-1](#__codelineno-25-1)$ export UV_INDEX_PRIVATE_REGISTRY_USERNAME=""
[#__codelineno-25-2](#__codelineno-25-2)$ export UV_INDEX_PRIVATE_REGISTRY_PASSWORD="$JFROG_JWT_TOKEN"

```

!!! note "Note"

    Replace `PRIVATE_REGISTRY `in the environment variable names with the actual index name defined in your `pyproject.toml `. 

### [Publishing packages to JFrog Artifactory](#publishing-packages-to-jfrog-artifactory)

Add a `publish-url `to your index definition: 

pyproject.toml 

```
[#__codelineno-26-1](#__codelineno-26-1)[[tool.uv.index]]
[#__codelineno-26-2](#__codelineno-26-2)name = "private-registry"
[#__codelineno-26-3](#__codelineno-26-3)url = "https://.jfrog.io/artifactory/api/pypi//simple"
[#__codelineno-26-4](#__codelineno-26-4)publish-url = "https://.jfrog.io/artifactory/api/pypi/"

```

!!! important "Important"

    If you use `--token "$JFROG_TOKEN" `or `UV_PUBLISH_TOKEN `with JFrog, you will receive a 401 Unauthorized error as JFrog requires an empty username but uv passes `__token__ `for as the username when `--token `is used. 

To authenticate, pass your token as the password and set the username to an empty string: 

```
[#__codelineno-27-1](#__codelineno-27-1)$ uv publish --index  -u "" -p "$JFROG_TOKEN"

```

Alternatively, you can set environment variables: 

```
[#__codelineno-28-1](#__codelineno-28-1)$ export UV_PUBLISH_USERNAME=""
[#__codelineno-28-2](#__codelineno-28-2)$ export UV_PUBLISH_PASSWORD="$JFROG_TOKEN"
[#__codelineno-28-3](#__codelineno-28-3)$ uv publish --index private-registry

```

!!! note "Note"

    The publish environment variables ( `UV_PUBLISH_USERNAME `and `UV_PUBLISH_PASSWORD `) do not include the index name. 

June 30, 2025
