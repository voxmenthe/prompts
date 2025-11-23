# [The `uv auth `CLI](#the-uv-auth-cli)

uv provides a high-level interface for storing and retrieving credentials from services. 

## [Logging in to a service](#logging-in-to-a-service)

To add credentials for service, use the `uv auth login `command: 

```
[#__codelineno-0-1](#__codelineno-0-1)$ uv auth login example.com

```

This will prompt for the credentials. 

The credentials can also be provided using the `--username `and `--password `options, or the `--token `option for services which use a `__token__ `or arbitrary username. 

!!! note "Note"

    We recommend providing the secret via stdin. Use `- `to indicate the value should be read from stdin, e.g., for `--password `: 

    ```
[#__codelineno-1-1](#__codelineno-1-1)$ echo 'my-password' | uv auth login example.com --password -

```

    The same pattern can be used with `--token `. 

Once credentials are added, uv will use them for packaging operations that require fetching content from the given service. At this time, only HTTPS Basic authentication is supported. The credentials will not yet be used for Git requests. 

!!! note "Note"

    The credentials will not be validated, i.e., incorrect credentials will not fail. 

## [Logging out of a service](#logging-out-of-a-service)

To remove credentials, use the `uv auth logout `command: 

```
[#__codelineno-2-1](#__codelineno-2-1)$ uv auth logout example.com

```

!!! note "Note"

    The credentials will not be invalidated with the remote server, i.e., they will only be removed from local storage not rendered unusable. 

## [Showing credentials for a service](#showing-credentials-for-a-service)

To show the credential stored for a given URL, use the `uv auth token `command: 

```
[#__codelineno-3-1](#__codelineno-3-1)$ uv auth token example.com

```

If a username was used to log in, it will need to be provided as well, e.g.: 

```
[#__codelineno-4-1](#__codelineno-4-1)$ uv auth token --username foo example.com

```

## [Configuring the storage backend](#configuring-the-storage-backend)

Credentials are persisted to the uv [credentials store](../http/#the-uv-credentials-store). 

By default, credentials are written to a plaintext file. An encrypted system-native storage backend can be enabled with `UV_PREVIEW_FEATURES=native-auth `. 

September 2, 2025
