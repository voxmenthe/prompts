LiveKit Docs › Building voice agents › Workflows

---

# Workflows

> How to model repeatable, accurate workflows through agents, handoffs, and tasks.

## Overview

LiveKit Agents enables you to compose reliable workflows to tackle complex scenarios.

An **Agent** takes indefinite control of a session. It can include custom prompts, tools, and other logic. If needed, it can invoke tasks or hand off control to a different agent. This is useful for scenarios such as the following:

- Including multiple personas with unique traits within a single session.
- Moving through different predetermined conversation phases.
- Offering multiple modes or functionality within a single voice agent.

The framework also includes experimental support for **Tasks**, which take temporary control of a session to complete a specific task and return a specific result. For more information, see [Tasks](#tasks).

## Agents

Agents form the backbone of a session’s functionality and are responsible for overall orchestration.

### Defining an agent

Extend the `Agent` class to define a custom agent.

```python
from livekit.agents import Agent

class HelpfulAssistant(Agent):
    def __init__(self):
        super().__init__(instructions="You are a helpful voice AI assistant.")

    async def on_enter(self) -> None:
        await self.session.generate_reply(instructions="Greet the user and ask how you can help them.")

```

You can also create an instance of `Agent` class directly:

```python
agent = Agent(instructions="You are a helpful voice AI assistant.")

```

### Setting the active agent

Specify the initial agent in the `AgentSession` constructor:

```python
session = AgentSession(
    agent=CustomerServiceAgent()
    # ...
)

```

To set a new agent, use the `update_agent` method:

```python
session.update_agent(CustomerServiceAgent())

```

### Handing off from tool call

Return a different agent from within a tool call to hand off control automatically. This allows the LLM to make decisions about when handoff should occur. For more information, see [tool return value](https://docs.livekit.io/agents/build/tools.md#return-value).

```python
from livekit.agents import Agent, function_tool

class CustomerServiceAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="""You are a friendly customer service representative. Help customers with 
            general inquiries, account questions, and technical support. If a customer needs 
            specialized help, transfer them to the appropriate specialist."""
        )

    async def on_enter(self) -> None:
        await self.session.generate_reply(instructions="Greet the user warmly and offer your assistance.")

    @function_tool()
    async def transfer_to_billing(self, context: RunContext):
        """Transfer the customer to a billing specialist for account and payment questions."""
        return "Transferring to billing", BillingAgent(chat_ctx=self.chat_ctx)

    @function_tool()
    async def transfer_to_technical_support(self, context: RunContext):
        """Transfer the customer to technical support for product issues and troubleshooting."""
        return "Transferring to technical support", TechnicalSupportAgent(chat_ctx=self.chat_ctx)

class BillingAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="""You are a billing specialist. Help customers with account questions, 
            payments, refunds, and billing inquiries. Be thorough and empathetic."""
        )

    async def on_enter(self) -> None:
        await self.session.generate_reply(instructions="Introduce yourself as a billing specialist and ask how you can help with their account.")

class TechnicalSupportAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="""You are a technical support specialist. Help customers troubleshoot 
            product issues, setup problems, and technical questions. Ask clarifying questions 
            to diagnose problems effectively."""
        )

    async def on_enter(self) -> None:
        await self.session.generate_reply(instructions="Introduce yourself as a technical support specialist and offer to help with any technical issues.")

```

### Passing state

To store custom state within your session, use the `userdata` attribute. The type of userdata is up to you, but the recommended approach is to use a `dataclass`.

```python
from livekit.agents import AgentSession
from dataclasses import dataclass

@dataclass
class MySessionInfo:
    user_name: str | None = None
    age: int | None = None

```

To add userdata to your session, pass it in the constructor. You must also specify the type of userdata on the `AgentSession` itself.

```python
session = AgentSession[MySessionInfo](
    userdata=MySessionInfo(),
    # ... tts, stt, llm, etc.
)

```

Userdata is available as `session.userdata`, and is also available within function tools on the `RunContext`. The following example shows how to use userdata in an agent workflow that starts with the `IntakeAgent`.

```python
class IntakeAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="""Your are an intake agent. Learn the user's name and age."""
        )
        
    @function_tool()
    async def record_name(self, context: RunContext[MySessionInfo], name: str):
        """Use this tool to record the user's name."""
        context.userdata.user_name = name
        return self._handoff_if_done()
    
    @function_tool()
    async def record_age(self, context: RunContext[MySessionInfo], age: int):
        """Use this tool to record the user's age."""
        context.userdata.age = age
        return self._handoff_if_done()
    
    def _handoff_if_done(self):
        if self.session.userdata.user_name and self.session.userdata.age:
            return CustomerServiceAgent()
        else:
            return None

class CustomerServiceAgent(Agent):
    def __init__(self):
        super().__init__(instructions="You are a friendly customer service representative.")

    async def on_enter(self) -> None:
        userdata: MySessionInfo = self.session.userdata
        await self.session.generate_reply(
            instructions=f"Greet {userdata.user_name} personally and offer your assistance."
        )

```

## Tasks

Tasks allow you to create focused, reusable components that complete specific tasks and return typed results. Unlike regular agents that take indefinite control of a session, tasks are used within agents or other tasks, complete their objective, and return control along with their result.

Tasks are useful for scenarios such as:

- Acquiring recording consent at the beginning of a call.
- Collecting specific structured information, such as an address or a credit card number.
- Moving through a series of questions, one at a time.
- Any discrete task that should complete and return control to the caller.

> 🔥 **Experimental feature**
> 
> Tasks are currently experimental and the API might change in a future release.

### Defining a task

Extend the `AgentTask` class and specify a result type using [generics](https://typing.python.org/en/latest/reference/generics.html). Use the `on_enter` method to begin the task's interaction with the user, and call the `complete` method with a result when complete. The task has full support for tools, similar to an agent.

```python
from livekit.agents import AgentTask, function_tool

class CollectConsent(AgentTask[bool]):
    def __init__(self):
        super().__init__(
            instructions="Ask for recording consent and get a clear yes or no answer."
        )

    async def on_enter(self) -> None:
        await self.session.generate_reply(instructions="Ask for permission to record the call for quality assurance purposes.")

    @function_tool
    async def consent_given(self) -> None:
        """Use this when the user gives consent to record."""
        self.complete(True)

    @function_tool
    async def consent_denied(self) -> None:
        """Use this when the user denies consent to record."""
        self.complete(False)

```

### Running a task

The task runs automatically upon creation. It must be created within the context of an existing `Agent` which is active within an `AgentSession`. The task takes control of the session until it returns a result. Await the task to receive its result.

```python
from livekit.agents import Agent, function_tool, get_job_context

class CustomerServiceAgent(Agent):
    def __init__(self):
        super().__init__(instructions="You are a friendly customer service representative.")

    async def on_enter(self) -> None:
        if await CollectConsent(chat_ctx=self.chat_ctx):
            await self.session.generate_reply(instructions="Offer your assistance to the user.")
        else:
            await self.session.generate_reply(instructions="Inform the user that you are unable to proceed and will end the call.")
            job_ctx = get_job_context()
            await job_ctx.api.room.delete_room(api.DeleteRoomRequest(room=job_ctx.room.name))

```

### Task results

Use any result type you want. For complex results, use a custom dataclass.

```python
from dataclasses import dataclass

@dataclass
class ContactInfoResult:
    name: str
    email_address: str
    phone_number: str

class GetContactInfoTask(AgentTask[ContactInfoResult]):
    # ....

```

### Prebuilt tasks

The framework will include prebuilt tasks for common use cases within the module [livekit.agents.beta.workflows](https://docs.livekit.io/reference/python/v1/livekit/agents/beta/workflows/index.html.md). As of initial release, only the `GetEmailTask` is available.

#### GetEmailTask

Use `GetEmailTask` to reliably collect and validate an email address from the user.

```python
from livekit.agents.beta.workflows import GetEmailTask

# ... within your agent ...
email_result = await GetEmailTask(chat_ctx=self.chat_ctx)
print(f"Collected email: {email_result.email_address}")

```

## Context preservation

By default, each new agent or task starts with a fresh conversation history for their LLM prompt. To include the prior conversation, set the `chat_ctx` parameter in the `Agent` or `AgentTask` constructor. You can either copy the prior agent's `chat_ctx`, or construct a new one based on custom business logic to provide the appropriate context.

```python
from livekit.agents import ChatContext, function_tool, Agent

class TechnicalSupportAgent(Agent):
    def __init__(self, chat_ctx: ChatContext):
        super().__init__(
            instructions="""You are a technical support specialist. Help customers troubleshoot 
            product issues, setup problems, and technical questions.""",
            chat_ctx=chat_ctx
        )

class CustomerServiceAgent(Agent):
    # ...

    @function_tool()
    async def transfer_to_technical_support(self):
        """Transfer the customer to technical support for product issues and troubleshooting."""
        await self.session.generate_reply(instructions="Inform the customer that you're transferring them to the technical support team.")
        
        # Pass the chat context during handoff
        return TechnicalSupportAgent(chat_ctx=self.session.chat_ctx)

```

The complete conversation history for the session is always available in `session.history`.

## Overriding plugins

You can override any of the plugins used in the session by setting the corresponding attributes in your `Agent` or `AgentTask` constructor. For instance, you can change the voice for a specific agent by overriding the `tts` attribute.

```python
from livekit.agents import Agent
from livekit.plugins import cartesia

class CustomerServiceManager(Agent):
    def __init__(self):
        super().__init__(
            instructions="You are a customer service manager who can handle escalated issues.",
            tts=cartesia.TTS(voice="6f84f4b8-58a2-430c-8c79-688dad597532")
        )

```

## Examples

These examples show how to build more complex workflows with multiple agents:

- **[Drive-thru agent](https://github.com/livekit/agents/blob/main/examples/drive-thru)**: A complex food ordering agent with tasks, tools, and a complete evaluation suite.

- **[Front-desk agent](https://github.com/livekit/agents/blob/main/examples/frontdesk)**: A calendar booking agent with tasks, tools, and evaluations.

- **[Medical Office Triage](https://github.com/livekit-examples/python-agents-examples/tree/main/complex-agents/medical_office_triage)**: Agent that triages patients based on symptoms and medical history.

- **[Restaurant Agent](https://github.com/livekit/agents/blob/main/examples/voice_agents/restaurant_agent.py)**: A restaurant front-of-house agent that can take orders, add items to a shared cart, and checkout.

## Further reading

For more information on concepts touched on in this article, see the following related articles:

- **[Tool definition and use](https://docs.livekit.io/agents/build/tools.md)**: Complete guide to defining and using tools in your agents.

- **[Nodes](https://docs.livekit.io/agents/build/nodes.md)**: Add custom behavior to any component of the voice pipeline.

- **[Agent speech](https://docs.livekit.io/agents/build/audio.md)**: Customize the speech output of your agents.

- **[Testing & evaluation](https://docs.livekit.io/agents/build/testing.md)**: Test every aspect of your agents with a custom test suite.

---

This document was rendered at 2025-08-04T05:46:54.752Z.
For the latest version of this document, see [https://docs.livekit.io/agents/build/workflows.md](https://docs.livekit.io/agents/build/workflows.md).

To explore all LiveKit documentation, see [llms.txt](https://docs.livekit.io/llms.txt).