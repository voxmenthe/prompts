LiveKit docs › Agent server › Server options

---

# Agent server options

> Learn about the options available for creating an agent server.

## Options

The constructor for `AgentServer` includes some parameters for configuring the agent server. The following includes some of the available parameters. For the complete list, see the [AgentServer reference](https://docs.livekit.io/reference/python/v1/livekit/agents/index.html.md#livekit.agents.AgentServer).

> ℹ️ **Python and Node.js differences**
> 
> In Python, the `@sever.rtc_session()` decorator is used to define some options for the agent server. In Node.js, these options are set up using the `ServerOptions` class.

> 💡 **Use the quickstart first**
> 
> You can edit the agent created in the [Voice AI quickstart](https://docs.livekit.io/agents/start/voice-ai.md) to try out the code samples in this topic.

**Python**:

```python
server = AgentServer(
    # Whether the agent can subscribe to tracks, publish data, update metadata, etc.
    permissions,
    # Amount of time to wait for existing jobs to finish when SIGTERM or SIGINT is received
    drain_timeout,
    # The maximum value of load_fnc, above which no new processes will spawn
    load_threshold,
    # A function to perform any necessary initialization before the job starts.
    setup_fnc,
    # Function to determine the current load of the worker. Should return a value between 0 and 1.
    load_fnc,
)

# start the agent server
cli.run_app(server)

```

---

**Node.js**:

```ts
const server = new AgentServer({
  // inspect the request and decide if the current agent server should handle it.
  requestFunc,
  // whether the agent can subscribe to tracks, publish data, update metadata, etc.
  permissions,
  // the type of agent server to create, either JT_ROOM or JT_PUBLISHER
  serverType=ServerType.JT_ROOM,
  // a function that reports the current load of the agent server. returns a value between 0-1.
  loadFunc,
  // the maximum value of loadFunc, above which agent server is marked as unavailable.
  loadThreshold,
})

// Start the agent server
cli.runApp(server);

```

> 🔥 **Caution**
> 
> For security purposes, set the LiveKit API key and secret as environment variables rather than as `ServerAgent` parameters.

### Entrypoint function

The entrypoint function is the main function called for each new job, and is the core of your agent app. To learn more, see the [entrypoint documentation](https://docs.livekit.io/agents/server/job.md#entrypoint) in the job lifecycle topic.

**Python**:

In Python, the entrypoint function is defined using the `@sever.rtc_session()` decorator on the agent function:

```python
@sever.rtc_session()
async def my_agent(ctx: JobContext):
    # connect to the room

    # handle the session
    ...

```

---

**Node.js**:

In Node.js, the entrypoint function is defined as a property of the default export of the agent file:

```ts
export default defineAgent({
  entry: async (ctx: JobContext) => {
    // connect to the room
    await ctx.connect();
    // handle the session
  },
});

```

### Request handler

The `on_request` function runs each time the server has a job for the agent. The framework expects agent servers to explicitly accept or reject each job request. If the agent server accepts the request, your [entrypoint function](#entrypoint) is called. If the request is rejected, it's sent to the next available agent server. A rejection indicates that the agent server is unable to handle the job, not that the job itself is invalid. The framework simply reassigns it to another agent server.

If `on_request` is not defined, the default behavior is to automatically accept all requests dispatched to the agent server.

**Python**:

```python
async def request_fnc(req: JobRequest):
    # accept the job request
    await req.accept(
        # the agent's name (Participant.name), defaults to ""
        name="agent",
        # the agent's identity (Participant.identity), defaults to "agent-<jobid>"
        identity="identity",
        # attributes to set on the agent participant upon join
        attributes={"myagent": "rocks"},
    )

    # or reject it
    # await req.reject()

server = AgentServer()

@server.rtc_session(on_request=request_fnc)
async def my_agent(ctx: JobContext):
    # set up entrypoint function
    # handle the session
    ...

```

---

**Node.js**:

```ts
const requestFunc = async (req: JobRequest) => {
  // accept the job request
  await req.accept(
    // the agent's name (Participant.name), defaults to ""
    'agent',
    // the agent's identity (Participant.identity), defaults to "agent-<jobid>"
    'identity',
  );
};

const server = new AgentServer({
  requestFunc,
});

```

> ℹ️ **Agent display name**
> 
> The `name` parameter is the display name of the agent, used to identify the agent in the room. It defaults to the agent's identity. This parameter is _not_ the same as the `agent_name` parameter for the `@sever.realtime_session()` decorator, which is used to [explicitly dispatch](https://docs.livekit.io/agents/server/agent-dispatch.md) the agent to a room.

### Prewarm function

For isolation and performance reasons, the framework runs each agent job in its own process. Agents often need access to model files that take time to load. To address this, you can use a `prewarm` function to warm up the process before assigning any jobs to it. You can control the number of processes to keep warm using the `num_idle_processes` parameter.

**Python**:

In Python, set the `setup_fnc` for `AgentServer` to your prewarm function:

```python
server = AgentServer()

def prewarm(proc: JobProcess):
    # load silero weights and store to process userdata
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm

@server.rtc_session()
async def my_agent(ctx: JobContext):
    # access the loaded silero instance
    vad: silero.VAD = ctx.proc.userdata["vad"]


```

---

**Node.js**:

In Node.js, the prewarm function is defined as a property of the default export of the agent file:

```ts
export default defineAgent({
  prewarm: async (proc: JobProcess) => {
    // load silero weights and store to process userdata
    proc.userData.vad = await silero.VAD.load();
  },
  entry: async (ctx: JobContext) => {
    // access the loaded silero instance
    const vad = ctx.proc.userData.vad! as silero.VAD;
  },
});

```

### Agent server load

In [custom deployments](https://docs.livekit.io/agents/ops/deployment/custom.md), you can configure the conditions under which the agent server stops accepting new jobs through the `load_fnc` and `load_threshold` parameters.

- `load_fnc`: A function that returns the current load of the agent server as a float between 0 and 1.0.
- `load_threshold`: The maximum load value at which the agent server still accepts new jobs.

The default `load_fnc` is the agent server's average CPU utilization over a 5-second window. The default `load_threshold` is `0.7`.

**Python**:

The following example shows how to define a custom load function that limits the agent server to 9 concurrent jobs, independent of CPU usage:

```python
from livekit.agents import AgentServer

server = AgentServer(
    load_threshold=0.9,
)

def compute_load(agent server: AgentServer) -> float:
    return min(len(agent server.active_jobs) / 10, 1.0)

server.load_fnc=compute_load

```

---

**Node.js**:

```ts
import { AgentServer } from '@livekit/agents';

const computeLoad = (agentServer: AgentServer): Promise<number> => {
  return Math.min(agentServer.activeJobs.length / 10, 1.0);
};

const server = new AgentServer({
  loadFunc: computeLoad,
  loadThreshold: 0.9,
});

```

> ℹ️ **Not available in LiveKit Cloud**
> 
> The `load_fnc` and `load_threshold` parameters cannot be changed in LiveKit Cloud deployments.

### Drain timeout

Agent sessions are stateful and should **not** be terminated abruptly. The Agents framework supports graceful termination: when a `SIGTERM` or `SIGINT` signal is received, the agent server enters a `draining` state. In this state, it stops accepting new jobs but allows existing ones to complete, up to a configured timeout.

The `drain_timeout` parameter sets the maximum time to wait for active jobs to finish. It defaults to 30 minutes.

### Permissions

By default, agents can both publish to and subscribe from the other participants in the same room. However, you can customize these permissions by setting the `permissions` parameter. To see the full list of parameters, see the [AgentServerPermissions reference](https://docs.livekit.io/reference/python/v1/livekit/agents/index.html.md#livekit.agents.AgentServerPermissions).

**Python**:

```python
server = AgentServer(
    ...
    permissions=AgentServerPermissions(
        can_publish=True,
        can_subscribe=True,
        can_publish_data=True,
        # when set to true, the agent won't be visible to others in the room.
        # when hidden, it will also not be able to publish tracks to the room as it won't be visible.
        hidden=False,
    ),
)

```

---

**Node.js**:

```ts
const server = new AgentServer({
  permissions: new AgentServerPermissions({
    canPublish: true,
    canSubscribe: true,
    // when set to true, the agent won't be visible to others in the room.
    // when hidden, it will also not be able to publish tracks to the room as it won't be visible
    hidden: false,
  }),
});

```

### Agent server type

You can choose to start a new instance of the agent for each room or for each publisher in the room. This can be set when you register your agent server:

**Python**:

In Python, the agent server type can be set using the `type` parameter for the `@sever.rtc_session()` decorator:

```python
@server.rtc_session(type=ServerType.ROOM)
async def my_agent(ctx: JobContext):
    # ...

```

---

**Node.js**:

```ts
const server = new AgentServer({
  // agent: ...
  // when omitted, the default is ServerType.JT_ROOM
  agent serverType: ServerType.JT_ROOM,
});

```

The `ServerType` enum has two options:

- `ROOM`: Create a new instance of the agent for each room.
- `PUBLISHER`: Create a new instance of the agent for each publisher in the room.

If the agent is performing resource-intensive operations in a room that could potentially include multiple publishers (for example, processing incoming video from a set of security cameras), you can set `agent server_type` to `JT_PUBLISHER` to ensure that each publisher has its own instance of the agent.

For `PUBLISHER` jobs, call the entrypoint function once for each publisher in the room. The `JobContext.publisher` object contains a `RemoteParticipant` representing that publisher.

## Starting the agent server

To spin up an agent server with the configuration defined in the `AgentServer` constructor, call the CLI:

**Python**:

```python
if __name__ == "__main__":
    cli.run_app(server)

```

---

**Node.js**:

```ts
cli.runApp(server);

```

The Agents agent server CLI provides two subcommands: `start` and `dev`. The former outputs raw JSON data to stdout, and is recommended for production. `dev` is recommended to use for development, as it outputs human-friendly colored logs, and supports hot reloading on Python.

## Log levels

By default, your agent server and all of its job processes output logs at the `INFO` level or higher. You can configure this behavior with the `--log-level` flag.

**Python**:

```shell
uv run agent.py start --log-level=DEBUG

```

---

**Node.js**:

> ℹ️ **Run script must be set up in package.json**
> 
> The `start` script must be set up in your `package.json` file to run the following command. If you haven't already, see [Agent CLI modes](https://docs.livekit.io/agents/start/voice-ai.md#cli-modes) for the command to add it.

```shell
pnpm run start --log-level=debug

```

The following log levels are available:

- `DEBUG`: Detailed information for debugging.
- `INFO`: Default level for general information.
- `WARNING`: Warning messages.
- `ERROR`: Error messages.
- `CRITICAL`: Critical error messages.

---


For the latest version of this document, see [https://docs.livekit.io/agents/server/options.md](https://docs.livekit.io/agents/server/options.md).

To explore all LiveKit documentation, see [llms.txt](https://docs.livekit.io/llms.txt).