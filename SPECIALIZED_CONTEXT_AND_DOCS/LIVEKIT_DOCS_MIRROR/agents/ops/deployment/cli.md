LiveKit docs › Agent deployment › Deploying to LiveKit Cloud › CLI reference

---

# Agent deployment CLI reference

> Reference for the LiveKit Cloud agent deployment commands in the LiveKit CLI.

## Overview

The LiveKit CLI is the primary interface for managing agents [deployed to LiveKit Cloud](https://docs.livekit.io/agents/ops/deployment.md). All agent commands are prefixed with `lk agent`.

For instructions on installing the CLI, see the LiveKit CLI [Getting started](https://docs.livekit.io/home/cli.md) guide.

```shell
lk agent [command] [command options] [working-dir]

```

> 🔥 **CLI version requirement**
> 
> Update the CLI regularly to ensure you have the latest version. You must have an up-to-date CLI to deploy and manage agents. See [Update the CLI](https://docs.livekit.io/home/cli.md#updates) for instructions.

### Working directory

The default working directory for each command is the current directory. You can override the working directory by passing it as the first regular argument.

For example, this command deploys the agent in the current directory:

```shell
lk agent deploy

```

While this command deploys the agent in the named directory:

```shell
lk agent deploy ~/my-agent

```

### Project and agent identification

If a `livekit.toml` file is present in the working directory, the CLI uses the project and agent configuration from that file by default.

If no `livekit.toml` file is present, the CLI uses the [default project](https://docs.livekit.io/home/cli/projects.md#set-default). You must also specify the agent ID with the `--id` flag for commands that operate on an existing agent.

## Agent subcommands

The following agent subcommands are available:

### Create

Create a new agent using configuration in the working directory and optional secrets. You must not already have a configuration file for the agent (default name is `livekit.toml`). If no `Dockerfile` is present, the CLI creates one for you.

```shell
lk agent create [options] [working-dir]

```

Options for `create`:

- `--region REGION`: [Region code](https://docs.livekit.io/agents/ops/deployment.md#regions) for the agent deployment. If no value is provided, the CLI prompts you to select a region.
- `--secrets KEY=VALUE [--secrets KEY=VALUE]`: Comma-separated `KEY=VALUE` secrets. Injected as environment variables into the agent. Individual values take precedence over values in `--secrets-file`, in the case of duplicate keys.
- `--secrets-file FILE`: File containing secret `KEY=VALUE` pairs, one per line. Injected as environment variables into the agent.
- `--secret-mount FILE`: Path to a file to load as a [file-mounted secret](https://docs.livekit.io/agents/ops/deployment/secrets.md#file-mounted-secrets) in the agent container.
- `--config FILE`: Name of the configuration file to create for the new deployment. If no value is provided, the default name is `livekit.toml`.
- `--silent`: Do not prompt for interactive confirmation. Default: `false`.

#### Examples

Create and [deploy a new agent](https://docs.livekit.io/agents/ops/deployment.md#create) to `us-east` from the current directory, providing secrets inline and via file:

```shell
lk agent create \
  --region us-east \
  --secrets OPENAI_API_KEY=sk-xxx,GOOGLE_API_KEY=ya29.xxx \
  --secrets-file ./secrets.env \
  .

```

### Deploy

[Build and deploy](https://docs.livekit.io/agents/ops/deployment/builds.md) a new agent version based on the working directory. You must have a `livekit.toml` and `Dockerfile` in the working directory.

```shell
lk agent deploy [options] [working-dir]

```

Options for `deploy`:

- `--secrets KEY=VALUE [--secrets KEY=VALUE]`: Comma-separated `KEY=VALUE` secrets. Injected as environment variables into the agent. Takes precedence over `--secrets-file`.
- `--secrets-file FILE`: File containing secret `KEY=VALUE` pairs, one per line. Injected as environment variables into the agent.
- `--secret-mount FILE`: Path to a file to load as a [file-mounted secret](https://docs.livekit.io/agents/ops/deployment/secrets.md#file-mounted-secrets) in the agent container.

#### Examples

Deploy a new version from the current directory:

```shell
lk agent deploy

```

Deploy a new version from the subdirectory `./agent`:

```shell
lk agent deploy ./agent

```

### Status

Show the current status of the specified agent:

```shell
lk agent status [options] [working-dir]

```

Options for `status`:

- `--id AGENT_ID`: Agent ID. By default, uses the ID found in the `livekit.toml` file in the working directory.

#### Examples

Show the status of the agent in the current directory:

```shell
lk agent status

```

Show the status of the agent with the ID `CA_MyAgentId`:

```shell
lk agent status --id CA_MyAgentId

```

Example output:

```shell
Using default project [my-project]
Using agent [CA_MyAgentId]
┌─────────────────┬────────────────┬─────────┬──────────┬────────────┬─────────┬───────────┬──────────────────────┐
│ ID              │ Version        │ Region  │ Status   │ CPU        │ Mem     │ Replicas  │ Deployed At          │
├─────────────────┼────────────────┼─────────┼──────────┼────────────┼─────────┼───────────┼──────────────────────┤
│ CA_MyAgentId    │ 20250809003117 │ us-east │ Sleeping │ 0m / 2000m │ 0 / 4GB │ 1 / 1 / 1 │ 2025-08-09T00:31:48Z │
└─────────────────┴────────────────┴─────────┴──────────┴────────────┴─────────┴───────────┴──────────────────────┘

```

#### Status values

The `status` field indicates the current state of the agent.

##### Normal statuses

These indicate that the agent is running or deploying normally.

| Agent status | Description |
| Setting Up | Agent created; waiting for provisioning. |
| Building | Building images for a new version. |
| Running | Agent is running and serving users. |
| Updating | Agent is pending update. |
| Scheduling | Agent is being deployed. |
| Deleting | Agent is pending delete. |

##### Sleep

Agents on certain plans may be scaled down to zero active instances. See [cold start](https://docs.livekit.io/agents/ops/deployment.md#cold-start) for more info.

| Agent status | Description |
| Sleeping | Agent has been scaled down. |
| Waking | Agent is scaling back up to serve a new user. |

##### Errors

These indicate that the agent is in an error state.

| Agent status | Description |
| Error | Agent program exited with a non-zero error code. |
| CrashLoop | Agent pod is crash looping. |
| Build Failed | Latest build failed. |
| Server Error | LiveKit Cloud Agents infrastructure error (not customer-related). See the live [Status page](https://status.livekit.io) for more info. |
| Deleted | Agent has been deleted. |
| Suspended | Project suspended for suspicious behavior. |

### Update

Update secrets for an existing agent. This command restarts the agent servers, but does not interrupt any active sessions.

```shell
lk agent update [options] [working-dir]

```

Options for `update`:

- `--secrets KEY=VALUE [--secrets KEY=VALUE]`: Comma-separated `KEY=VALUE` secrets. Injected as environment variables into the agent. Takes precedence over `--secrets-file`.
- `--secrets-file FILE`: File containing secret `KEY=VALUE` pairs, one per line. Injected as environment variables into the agent.
- `--secret-mount FILE`: Path to a file to load as a [file-mounted secret](https://docs.livekit.io/agents/ops/deployment/secrets.md#file-mounted-secrets) in the agent container.
- `--id AGENT_ID`: Agent ID. By default, uses the ID found in the `livekit.toml` file in the working directory.

#### Examples

Update secrets and restart the agent:

```shell
lk agent update \
  --secrets OPENAI_API_KEY=sk-new

```

### Restart

Restart the agent server pool for the specified agent. This command does not interrupt any active sessions.

```shell
lk agent restart [options] [working-dir]

```

Options for `restart`:

- `--id AGENT_ID`: Agent ID. By default, uses the ID found in the `livekit.toml` file in the working directory.

#### Examples

```shell
lk agent restart --id CA_MyAgentId

```

### Rollback

[Rollback](https://docs.livekit.io/agents/ops/deployment.md#rollback) the specified agent to a prior version:

```shell
lk agent rollback [options] [working-dir]

```

Options for `rollback`:

- `--version string`: Version to roll back to. Defaults to the most recent version prior to the current.
- `--id ID`: Agent ID. If unset and `livekit.toml` is present, uses the ID found there.

#### Examples

Roll back to a specific version:

```shell
lk agent rollback --id CA_MyAgentId --version 20250809003117

```

### Logs

Stream [logs](https://docs.livekit.io/agents/ops/deployment/logs.md) for the specified agent and log type. Also available as `tail`.

```shell
lk agent logs [options] [working-dir]
# or
lk agent tail [options] [working-dir]

```

Options for `logs`/`tail`:

- `--id ID`: Agent ID. If unset and `livekit.toml` is present, uses the ID found there.
- `--log-type string`: Log type to retrieve. Valid values: `deploy`, `build`. Default: `deploy`.

#### Examples

Tail deploy logs:

```shell
lk agent logs --id CA_MyAgentId --log-type deploy

```

### Delete

Delete the specified agent. Also available as `destroy`.

```shell
lk agent delete [options] [working-dir]
# or
lk agent destroy [options] [working-dir]

```

Options for `delete`/`destroy`:

- `--id ID`: Agent ID. If unset and `livekit.toml` is present, uses the ID found there.

#### Examples

```shell
lk agent delete --id CA_MyAgentId

```

### Versions

List versions associated with the specified agent, which can be used to [rollback](https://docs.livekit.io/agents/ops/deployment.md#rollback).

```shell
lk agent versions [options] [working-dir]

```

Options for `versions`:

- `--id ID`: Agent ID. If unset and `livekit.toml` is present, uses the ID found there.

#### Examples

```shell
lk agent versions --id CA_MyAgentId

```

Example output:

```shell
Using default project [my-project]
Using agent [CA_MyAgentId]
┌────────────────┬─────────┬──────────────────────┐
│ Version        │ Current │ Deployed At          │
├────────────────┼─────────┼──────────────────────┤
│ 20250809003117 │ true    │ 2025-08-09T00:31:48Z │
└────────────────┴─────────┴──────────────────────┘

```

### List

List all deployed agents in the current project:

```shell
lk agent list [options]

```

Options for `list`:

- `--id IDs [--id IDs]`: Filter to one or more agent IDs. Repeatable.
- `--project PROJECT_NAME`: The project name to list agents for. By default, use the project from the current `livekit.toml` file or the [default project](https://docs.livekit.io/home/cli/projects.md#set-default).

#### Examples

```shell
lk agent list

```

Example output:

```shell
Using default project [my-project]
┌─────────────────┬─────────┬────────────────┬──────────────────────┐
│ ID              │ Regions │ Version        │ Deployed At          │
├─────────────────┼─────────┼────────────────┼──────────────────────┤
│ CA_MyAgentId    │ us-east │ 20250809003117 │ 2025-08-09T00:31:48Z │
└─────────────────┴─────────┴────────────────┴──────────────────────┘

```

### Secrets

Show the current [secret](https://docs.livekit.io/agents/ops/deployment/secrets.md) keys for the specified agent. Does not include secret values.

```shell
lk agent secrets [options] [working-dir]

```

Options for `secrets`:

- `--id AGENT_ID`: Agent ID. By default, uses the ID found in the `livekit.toml` file in the working directory.

#### Examples

```shell
lk agent secrets --id CA_MyAgentId

```

Example output:

```shell
Using default project [my-project]
Using agent [CA_MyAgentId]
┌────────────────┬──────────────────────┬──────────────────────┐
│ Name           │ Created At           │ Updated At           │
├────────────────┼──────────────────────┼──────────────────────┤
│ OPENAI_API_KEY │ 2025-08-08T23:32:29Z │ 2025-08-09T00:31:10Z │
│ GOOGLE_API_KEY │ 2025-08-08T23:32:29Z │ 2025-08-09T00:31:10Z │
│ HEDRA_API_KEY  │ 2025-08-08T23:32:29Z │ 2025-08-09T00:31:10Z │
└────────────────┴──────────────────────┴──────────────────────┘

```

### Update secrets

Update secrets for the specified agent. This command restarts the agent:

```shell
lk agent update-secrets [options] [working-dir]

```

Options for `update-secrets`:

- `--secrets KEY=VALUE [--secrets KEY=VALUE]`: Comma-separated `KEY=VALUE` secrets. Injected as environment variables into the agent. Takes precedence over `--secrets-file`.
- `--secrets-file FILE`: File containing secret `KEY=VALUE` pairs, one per line. Injected as environment variables into the agent.
- `--secret-mount FILE`: Path to a file to load as a [file-mounted secret](https://docs.livekit.io/agents/ops/deployment/secrets.md#file-mounted-secrets) in the agent container.
- `--id ID`: Agent ID. If unset and `livekit.toml` is present, uses the ID found there.
- `--overwrite`: Overwrite existing secrets. Default: `false`.

#### Examples

Update secrets without overwriting existing keys:

```shell
lk agent update-secrets --id CA_MyAgentId \
  --secrets-file ./secrets.env

```

Overwrite existing keys explicitly:

```shell
lk agent update-secrets --id CA_MyAgentId \
  --secrets OPENAI_API_KEY=sk-xxx \
  --overwrite

```

Mount a file as a secret:

```shell
lk agent update-secrets --id CA_MyAgentId \
  --secret-mount ./google-appplication-credentials.json

```

### Config

Generate a new `livekit.toml` in the working directory for an existing agent:

```shell
lk agent config --id AGENT_ID [options] [working-dir]

```

Options for `config`:

- `--id AGENT_ID`: Agent ID. Uses the provided ID to generate a new `livekit.toml` file.

### Generate Dockerfile

Generate a new `Dockerfile` and `.dockerignore` file in the working directory. To overwrite existing files, use the `--overwrite` flag.

```shell
lk agent dockerfile [options] [working-dir]

```

Options for `dockerfile`:

- `--overwrite`: Overwrite existing files. Default: `false`.

#### Examples

```shell
lk agent dockerfile

```

---


For the latest version of this document, see [https://docs.livekit.io/agents/ops/deployment/cli.md](https://docs.livekit.io/agents/ops/deployment/cli.md).

To explore all LiveKit documentation, see [llms.txt](https://docs.livekit.io/llms.txt).