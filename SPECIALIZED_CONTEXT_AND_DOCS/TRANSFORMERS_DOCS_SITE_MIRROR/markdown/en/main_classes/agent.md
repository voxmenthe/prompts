# Agents & Tools

Transformers Agents is an experimental API which is subject to change at any time. Results returned by the agents
can vary as the APIs or underlying models are prone to change.

To learn more about agents and tools make sure to read the [introductory guide](../transformers_agents). This page
contains the API docs for the underlying classes.

## Agents

We provide two types of agents, based on the main [Agent](/docs/transformers/main/en/main_classes/agent#transformers.Agent) class:

* [CodeAgent](/docs/transformers/main/en/main_classes/agent#transformers.CodeAgent) acts in one shot, generating code to solve the task, then executes it at once.
* [ReactAgent](/docs/transformers/main/en/main_classes/agent#transformers.ReactAgent) acts step by step, each step consisting of one thought, then one tool call and execution. It has two classes:
  + [ReactJsonAgent](/docs/transformers/main/en/main_classes/agent#transformers.ReactJsonAgent) writes its tool calls in JSON.
  + [ReactCodeAgent](/docs/transformers/main/en/main_classes/agent#transformers.ReactCodeAgent) writes its tool calls in Python code.

### Agent

### class transformers.Agent

 [< source >](https://github.com/huggingface/transformers/blob/main/src/transformers/agents/agents.py#L356)

( tools: typing.Union[typing.List[transformers.agents.tools.Tool], transformers.agents.agents.Toolbox] llm\_engine: typing.Callable = None system\_prompt: typing.Optional[str] = None tool\_description\_template: typing.Optional[str] = None additional\_args: typing.Dict = {} max\_iterations: int = 6 tool\_parser: typing.Optional[typing.Callable] = None add\_base\_tools: bool = False verbose: int = 0 grammar: typing.Optional[typing.Dict[str, str]] = None managed\_agents: typing.Optional[typing.List] = None step\_callbacks: typing.Optional[typing.List[typing.Callable]] = None monitor\_metrics: bool = True  )

#### execute\_tool\_call

 [< source >](https://github.com/huggingface/transformers/blob/main/src/transformers/agents/agents.py#L532)

( tool\_name: str arguments: typing.Dict[str, str]  )

Parameters

* **tool\_name** (`str`) — Name of the Tool to execute (should be one from self.toolbox).
* **arguments** (Dict[str, str]) — Arguments passed to the Tool.

Execute tool with the provided input and returns the result.
This method replaces arguments with the actual values from the state if they refer to state variables.

#### extract\_action

 [< source >](https://github.com/huggingface/transformers/blob/main/src/transformers/agents/agents.py#L511)

( llm\_output: str split\_token: str  )

Parameters

* **llm\_output** (`str`) — Output of the LLM
* **split\_token** (`str`) — Separator for the action. Should match the example in the system prompt.

Parse action from the LLM output

#### run

 [< source >](https://github.com/huggingface/transformers/blob/main/src/transformers/agents/agents.py#L587)

( \*\*kwargs  )

To be implemented in the child class

#### write\_inner\_memory\_from\_logs

 [< source >](https://github.com/huggingface/transformers/blob/main/src/transformers/agents/agents.py#L451)

( summary\_mode: typing.Optional[bool] = False  )

Reads past llm\_outputs, actions, and observations or errors from the logs into a series of messages
that can be used as input to the LLM.

### CodeAgent

### class transformers.CodeAgent

 [< source >](https://github.com/huggingface/transformers/blob/main/src/transformers/agents/agents.py#L592)

( tools: typing.List[transformers.agents.tools.Tool] llm\_engine: typing.Optional[typing.Callable] = None system\_prompt: typing.Optional[str] = None tool\_description\_template: typing.Optional[str] = None grammar: typing.Optional[typing.Dict[str, str]] = None additional\_authorized\_imports: typing.Optional[typing.List[str]] = None \*\*kwargs  )

A class for an agent that solves the given task using a single block of code. It plans all its actions, then executes all in one shot.

#### parse\_code\_blob

 [< source >](https://github.com/huggingface/transformers/blob/main/src/transformers/agents/agents.py#L634)

( result: str  )

Override this method if you want to change the way the code is
cleaned in the `run` method.

#### run

 [< source >](https://github.com/huggingface/transformers/blob/main/src/transformers/agents/agents.py#L641)

( task: str return\_generated\_code: bool = False \*\*kwargs  )

Parameters

* **task** (`str`) — The task to perform
* **return\_generated\_code** (`bool`, *optional*, defaults to `False`) — Whether to return the generated code instead of running it
* **kwargs** (additional keyword arguments, *optional*) —
  Any keyword argument to send to the agent when evaluating the code.

Runs the agent for the given task.

Example:

```
from transformers.agents import CodeAgent

agent = CodeAgent(tools=[])
agent.run("What is the result of 2 power 3.7384?")
```

### React agents

### class transformers.ReactAgent

 [< source >](https://github.com/huggingface/transformers/blob/main/src/transformers/agents/agents.py#L718)

( tools: typing.List[transformers.agents.tools.Tool] llm\_engine: typing.Optional[typing.Callable] = None system\_prompt: typing.Optional[str] = None tool\_description\_template: typing.Optional[str] = None grammar: typing.Optional[typing.Dict[str, str]] = None plan\_type: typing.Optional[str] = None planning\_interval: typing.Optional[int] = None \*\*kwargs  )

This agent that solves the given task step by step, using the ReAct framework:
While the objective is not reached, the agent will perform a cycle of thinking and acting.
The action will be parsed from the LLM output: it consists in calls to tools from the toolbox, with arguments chosen by the LLM engine.

#### direct\_run

 [< source >](https://github.com/huggingface/transformers/blob/main/src/transformers/agents/agents.py#L854)

( task: str  )

Runs the agent in direct mode, returning outputs only at the end: should be launched only in the `run` method.

#### planning\_step

 [< source >](https://github.com/huggingface/transformers/blob/main/src/transformers/agents/agents.py#L894)

( task is\_first\_step: bool = False iteration: typing.Optional[int] = None  )

Parameters

* **task** (`str`) — The task to perform
* **is\_first\_step** (`bool`) — If this step is not the first one, the plan should be an update over a previous plan.
* **iteration** (`int`) — The number of the current step, used as an indication for the LLM.

Used periodically by the agent to plan the next steps to reach the objective.

#### provide\_final\_answer

 [< source >](https://github.com/huggingface/transformers/blob/main/src/transformers/agents/agents.py#L761)

( task  )

This method provides a final answer to the task, based on the logs of the agent’s interactions.

#### run

 [< source >](https://github.com/huggingface/transformers/blob/main/src/transformers/agents/agents.py#L783)

( task: str stream: bool = False reset: bool = True \*\*kwargs  )

Parameters

* **task** (`str`) — The task to perform

Runs the agent for the given task.

Example:

```
from transformers.agents import ReactCodeAgent
agent = ReactCodeAgent(tools=[])
agent.run("What is the result of 2 power 3.7384?")
```

#### stream\_run

 [< source >](https://github.com/huggingface/transformers/blob/main/src/transformers/agents/agents.py#L814)

( task: str  )

Runs the agent in streaming mode, yielding steps as they are executed: should be launched only in the `run` method.

### class transformers.ReactJsonAgent

 [< source >](https://github.com/huggingface/transformers/blob/main/src/transformers/agents/agents.py#L994)

( tools: typing.List[transformers.agents.tools.Tool] llm\_engine: typing.Optional[typing.Callable] = None system\_prompt: typing.Optional[str] = None tool\_description\_template: typing.Optional[str] = None grammar: typing.Optional[typing.Dict[str, str]] = None planning\_interval: typing.Optional[int] = None \*\*kwargs  )

This agent that solves the given task step by step, using the ReAct framework:
While the objective is not reached, the agent will perform a cycle of thinking and acting.
The tool calls will be formulated by the LLM in JSON format, then parsed and executed.

#### step

 [< source >](https://github.com/huggingface/transformers/blob/main/src/transformers/agents/agents.py#L1027)

( log\_entry: typing.Dict[str, typing.Any]  )

Perform one step in the ReAct framework: the agent thinks, acts, and observes the result.
The errors are raised here, they are caught and logged in the run() method.

### class transformers.ReactCodeAgent

 [< source >](https://github.com/huggingface/transformers/blob/main/src/transformers/agents/agents.py#L1105)

( tools: typing.List[transformers.agents.tools.Tool] llm\_engine: typing.Optional[typing.Callable] = None system\_prompt: typing.Optional[str] = None tool\_description\_template: typing.Optional[str] = None grammar: typing.Optional[typing.Dict[str, str]] = None additional\_authorized\_imports: typing.Optional[typing.List[str]] = None planning\_interval: typing.Optional[int] = None \*\*kwargs  )

This agent that solves the given task step by step, using the ReAct framework:
While the objective is not reached, the agent will perform a cycle of thinking and acting.
The tool calls will be formulated by the LLM in code format, then parsed and executed.

#### step

 [< source >](https://github.com/huggingface/transformers/blob/main/src/transformers/agents/agents.py#L1152)

( log\_entry: typing.Dict[str, typing.Any]  )

Perform one step in the ReAct framework: the agent thinks, acts, and observes the result.
The errors are raised here, they are caught and logged in the run() method.

### ManagedAgent

### class transformers.ManagedAgent

 [< source >](https://github.com/huggingface/transformers/blob/main/src/transformers/agents/agents.py#L1237)

( agent name description additional\_prompting = None provide\_run\_summary = False  )

## Tools

### load\_tool

#### transformers.load\_tool

 [< source >](https://github.com/huggingface/transformers/blob/main/src/transformers/agents/tools.py#L834)

( task\_or\_repo\_id model\_repo\_id = None token = None \*\*kwargs  )

Parameters

* **task\_or\_repo\_id** (`str`) —
  The task for which to load the tool or a repo ID of a tool on the Hub. Tasks implemented in Transformers
  are:
  + `"document_question_answering"`
  + `"image_question_answering"`
  + `"speech_to_text"`
  + `"text_to_speech"`
  + `"translation"`
* **model\_repo\_id** (`str`, *optional*) —
  Use this argument to use a different model than the default one for the tool you selected.
* **token** (`str`, *optional*) —
  The token to identify you on hf.co. If unset, will use the token generated when running `huggingface-cli login` (stored in `~/.huggingface`).
* **kwargs** (additional keyword arguments, *optional*) —
  Additional keyword arguments that will be split in two: all arguments relevant to the Hub (such as
  `cache_dir`, `revision`, `subfolder`) will be used when downloading the files for your tool, and the others
  will be passed along to its init.

Main function to quickly load a tool, be it on the Hub or in the Transformers library.

Loading a tool means that you’ll download the tool and execute it locally.
ALWAYS inspect the tool you’re downloading before loading it within your runtime, as you would do when
installing a package using pip/npm/apt.

### tool

#### transformers.tool

 [< source >](https://github.com/huggingface/transformers/blob/main/src/transformers/agents/tools.py#L981)

( tool\_function: typing.Callable  )

Parameters

* **tool\_function** — Your function. Should have type hints for each input and a type hint for the output.
* **Should** also have a docstring description including an ‘Args —’ part where each argument is described.

Converts a function into an instance of a Tool subclass.

### Tool

### class transformers.Tool

 [< source >](https://github.com/huggingface/transformers/blob/main/src/transformers/agents/tools.py#L109)

( \*args \*\*kwargs  )

A base class for the functions used by the agent. Subclass this and implement the `__call__` method as well as the
following class attributes:

* **description** (`str`) — A short description of what your tool does, the inputs it expects and the output(s) it
  will return. For instance ‘This is a tool that downloads a file from a `url`. It takes the `url` as input, and
  returns the text contained in the file’.
* **name** (`str`) — A performative name that will be used for your tool in the prompt to the agent. For instance
  `"text-classifier"` or `"image_generator"`.
* **inputs** (`Dict[str, Dict[str, Union[str, type]]]`) — The dict of modalities expected for the inputs.
  It has one `type`key and a `description`key.
  This is used by `launch_gradio_demo` or to make a nice space from your tool, and also can be used in the generated
  description for your tool.
* **output\_type** (`type`) — The type of the tool output. This is used by `launch_gradio_demo`
  or to make a nice space from your tool, and also can be used in the generated description for your tool.

You can also override the method [setup()](/docs/transformers/main/en/main_classes/agent#transformers.Tool.setup) if your tool as an expensive operation to perform before being
usable (such as loading a model). [setup()](/docs/transformers/main/en/main_classes/agent#transformers.Tool.setup) will be called the first time you use your tool, but not at
instantiation.

#### from\_gradio

 [< source >](https://github.com/huggingface/transformers/blob/main/src/transformers/agents/tools.py#L546)

( gradio\_tool  )

Creates a [Tool](/docs/transformers/main/en/main_classes/agent#transformers.Tool) from a gradio tool.

#### from\_hub

 [< source >](https://github.com/huggingface/transformers/blob/main/src/transformers/agents/tools.py#L262)

( repo\_id: str token: typing.Optional[str] = None \*\*kwargs  )

Parameters

* **repo\_id** (`str`) —
  The name of the repo on the Hub where your tool is defined.
* **token** (`str`, *optional*) —
  The token to identify you on hf.co. If unset, will use the token generated when running
  `huggingface-cli login` (stored in `~/.huggingface`).
* **kwargs** (additional keyword arguments, *optional*) —
  Additional keyword arguments that will be split in two: all arguments relevant to the Hub (such as
  `cache_dir`, `revision`, `subfolder`) will be used when downloading the files for your tool, and the
  others will be passed along to its init.

Loads a tool defined on the Hub.

Loading a tool from the Hub means that you’ll download the tool and execute it locally.
ALWAYS inspect the tool you’re downloading before loading it within your runtime, as you would do when
installing a package using pip/npm/apt.

#### from\_langchain

 [< source >](https://github.com/huggingface/transformers/blob/main/src/transformers/agents/tools.py#L567)

( langchain\_tool  )

Creates a [Tool](/docs/transformers/main/en/main_classes/agent#transformers.Tool) from a langchain tool.

#### from\_space

 [< source >](https://github.com/huggingface/transformers/blob/main/src/transformers/agents/tools.py#L430)

( space\_id: str name: str description: str api\_name: typing.Optional[str] = None token: typing.Optional[str] = None  ) → [Tool](/docs/transformers/main/en/main_classes/agent#transformers.Tool)

Parameters

* **space\_id** (`str`) —
  The id of the Space on the Hub.
* **name** (`str`) —
  The name of the tool.
* **description** (`str`) —
  The description of the tool.
* **api\_name** (`str`, *optional*) —
  The specific api\_name to use, if the space has several tabs. If not precised, will default to the first available api.
* **token** (`str`, *optional*) —
  Add your token to access private spaces or increase your GPU quotas.

Returns

[Tool](/docs/transformers/main/en/main_classes/agent#transformers.Tool)

The Space, as a tool.

Creates a [Tool](/docs/transformers/main/en/main_classes/agent#transformers.Tool) from a Space given its id on the Hub.

Examples:

```
image_generator = Tool.from_space(
    space_id="black-forest-labs/FLUX.1-schnell",
    name="image-generator",
    description="Generate an image from a prompt"
)
image = image_generator("Generate an image of a cool surfer in Tahiti")
```

```
face_swapper = Tool.from_space(
    "tuan2308/face-swap",
    "face_swapper",
    "Tool that puts the face shown on the first image on the second image. You can give it paths to images.",
)
image = face_swapper('./aymeric.jpeg', './ruth.jpg')
```

#### push\_to\_hub

 [< source >](https://github.com/huggingface/transformers/blob/main/src/transformers/agents/tools.py#L373)

( repo\_id: str commit\_message: str = 'Upload tool' private: typing.Optional[bool] = None token: typing.Union[bool, str, NoneType] = None create\_pr: bool = False  )

Parameters

* **repo\_id** (`str`) —
  The name of the repository you want to push your tool to. It should contain your organization name when
  pushing to a given organization.
* **commit\_message** (`str`, *optional*, defaults to `"Upload tool"`) —
  Message to commit while pushing.
* **private** (`bool`, *optional*) —
  Whether to make the repo private. If `None` (default), the repo will be public unless the organization’s default is private. This value is ignored if the repo already exists.
* **token** (`bool` or `str`, *optional*) —
  The token to use as HTTP bearer authorization for remote files. If unset, will use the token generated
  when running `huggingface-cli login` (stored in `~/.huggingface`).
* **create\_pr** (`bool`, *optional*, defaults to `False`) —
  Whether or not to create a PR with the uploaded files or directly commit.

Upload the tool to the Hub.

For this method to work properly, your tool must have been defined in a separate module (not `__main__`).

For instance:

```
from my_tool_module import MyTool
my_tool = MyTool()
my_tool.push_to_hub("my-username/my-space")
```

#### save

 [< source >](https://github.com/huggingface/transformers/blob/main/src/transformers/agents/tools.py#L202)

( output\_dir  )

Parameters

* **output\_dir** (`str`) — The folder in which you want to save your tool.

Saves the relevant code files for your tool so it can be pushed to the Hub. This will copy the code of your
tool in `output_dir` as well as autogenerate:

* a config file named `tool_config.json`
* an `app.py` file so that your tool can be converted to a space
* a `requirements.txt` containing the names of the module used by your tool (as detected when inspecting its
  code)

You should only use this method to save tools that are defined in a separate module (not `__main__`).

#### setup

 [< source >](https://github.com/huggingface/transformers/blob/main/src/transformers/agents/tools.py#L195)

( )

Overwrite this method here for any operation that is expensive and needs to be executed before you start using
your tool. Such as loading a big model.

### Toolbox

### class transformers.Toolbox

 [< source >](https://github.com/huggingface/transformers/blob/main/src/transformers/agents/agents.py#L186)

( tools: typing.List[transformers.agents.tools.Tool] add\_base\_tools: bool = False  )

Parameters

* **tools** (`List[Tool]`) —
  The list of tools to instantiate the toolbox with
* **add\_base\_tools** (`bool`, defaults to `False`, *optional*, defaults to `False`) —
  Whether to add the tools available within `transformers` to the toolbox.

The toolbox contains all tools that the agent can perform operations with, as well as a few methods to
manage them.

#### add\_tool

 [< source >](https://github.com/huggingface/transformers/blob/main/src/transformers/agents/agents.py#L232)

( tool: Tool  )

Parameters

* **tool** (`Tool`) —
  The tool to add to the toolbox.

Adds a tool to the toolbox

#### clear\_toolbox

 [< source >](https://github.com/huggingface/transformers/blob/main/src/transformers/agents/agents.py#L272)

( )

Clears the toolbox

#### remove\_tool

 [< source >](https://github.com/huggingface/transformers/blob/main/src/transformers/agents/agents.py#L244)

( tool\_name: str  )

Parameters

* **tool\_name** (`str`) —
  The tool to remove from the toolbox.

Removes a tool from the toolbox

#### show\_tool\_descriptions

 [< source >](https://github.com/huggingface/transformers/blob/main/src/transformers/agents/agents.py#L220)

( tool\_description\_template: typing.Optional[str] = None  )

Parameters

* **tool\_description\_template** (`str`, *optional*) —
  The template to use to describe the tools. If not provided, the default template will be used.

Returns the description of all tools in the toolbox

#### update\_tool

 [< source >](https://github.com/huggingface/transformers/blob/main/src/transformers/agents/agents.py#L258)

( tool: Tool  )

Parameters

* **tool** (`Tool`) —
  The tool to update to the toolbox.

Updates a tool in the toolbox according to its name.

### PipelineTool

### class transformers.PipelineTool

 [< source >](https://github.com/huggingface/transformers/blob/main/src/transformers/agents/tools.py#L631)

( model = None pre\_processor = None post\_processor = None device = None device\_map = None model\_kwargs = None token = None \*\*hub\_kwargs  )

Parameters

* **model** (`str` or [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel), *optional*) —
  The name of the checkpoint to use for the model, or the instantiated model. If unset, will default to the
  value of the class attribute `default_checkpoint`.
* **pre\_processor** (`str` or `Any`, *optional*) —
  The name of the checkpoint to use for the pre-processor, or the instantiated pre-processor (can be a
  tokenizer, an image processor, a feature extractor or a processor). Will default to the value of `model` if
  unset.
* **post\_processor** (`str` or `Any`, *optional*) —
  The name of the checkpoint to use for the post-processor, or the instantiated pre-processor (can be a
  tokenizer, an image processor, a feature extractor or a processor). Will default to the `pre_processor` if
  unset.
* **device** (`int`, `str` or `torch.device`, *optional*) —
  The device on which to execute the model. Will default to any accelerator available (GPU, MPS etc…), the
  CPU otherwise.
* **device\_map** (`str` or `dict`, *optional*) —
  If passed along, will be used to instantiate the model.
* **model\_kwargs** (`dict`, *optional*) —
  Any keyword argument to send to the model instantiation.
* **token** (`str`, *optional*) —
  The token to use as HTTP bearer authorization for remote files. If unset, will use the token generated when
  running `huggingface-cli login` (stored in `~/.huggingface`).
* **hub\_kwargs** (additional keyword arguments, *optional*) —
  Any additional keyword argument to send to the methods that will load the data from the Hub.

A [Tool](/docs/transformers/main/en/main_classes/agent#transformers.Tool) tailored towards Transformer models. On top of the class attributes of the base class [Tool](/docs/transformers/main/en/main_classes/agent#transformers.Tool), you will
need to specify:

* **model\_class** (`type`) — The class to use to load the model in this tool.
* **default\_checkpoint** (`str`) — The default checkpoint that should be used when the user doesn’t specify one.
* **pre\_processor\_class** (`type`, *optional*, defaults to [AutoProcessor](/docs/transformers/main/en/model_doc/auto#transformers.AutoProcessor)) — The class to use to load the
  pre-processor
* **post\_processor\_class** (`type`, *optional*, defaults to [AutoProcessor](/docs/transformers/main/en/model_doc/auto#transformers.AutoProcessor)) — The class to use to load the
  post-processor (when different from the pre-processor).

#### decode

 [< source >](https://github.com/huggingface/transformers/blob/main/src/transformers/agents/tools.py#L754)

( outputs  )

Uses the `post_processor` to decode the model output.

#### encode

 [< source >](https://github.com/huggingface/transformers/blob/main/src/transformers/agents/tools.py#L741)

( raw\_inputs  )

Uses the `pre_processor` to prepare the inputs for the `model`.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/main/src/transformers/agents/tools.py#L747)

( inputs  )

Sends the inputs through the `model`.

#### setup

 [< source >](https://github.com/huggingface/transformers/blob/main/src/transformers/agents/tools.py#L715)

( )

Instantiates the `pre_processor`, `model` and `post_processor` if necessary.

### launch\_gradio\_demo

#### transformers.launch\_gradio\_demo

 [< source >](https://github.com/huggingface/transformers/blob/main/src/transformers/agents/tools.py#L779)

( tool\_class: Tool  )

Parameters

* **tool\_class** (`type`) — The class of the tool for which to launch the demo.

Launches a gradio demo for a tool. The corresponding tool class needs to properly implement the class attributes
`inputs` and `output_type`.

### stream\_to\_gradio

#### transformers.stream\_to\_gradio

 [< source >](https://github.com/huggingface/transformers/blob/main/src/transformers/agents/monitoring.py#L60)

( agent task: str test\_mode: bool = False \*\*kwargs  )

Runs an agent with the given task and streams the messages from the agent as gradio ChatMessages.

### ToolCollection

### class transformers.ToolCollection

 [< source >](https://github.com/huggingface/transformers/blob/main/src/transformers/agents/tools.py#L949)

( collection\_slug: str token: typing.Optional[str] = None  )

Parameters

* **collection\_slug** (str) —
  The collection slug referencing the collection.
* **token** (str, *optional*) —
  The authentication token if the collection is private.

Tool collections enable loading all Spaces from a collection in order to be added to the agent’s toolbox.

> [!NOTE]
> Only Spaces will be fetched, so you can feel free to add models and datasets to your collection if you’d
> like for this collection to showcase them.

Example:

```
>>> from transformers import ToolCollection, ReactCodeAgent

>>> image_tool_collection = ToolCollection(collection_slug="huggingface-tools/diffusion-tools-6630bb19a942c2306a2cdb6f")
>>> agent = ReactCodeAgent(tools=[*image_tool_collection.tools], add_base_tools=True)

>>> agent.run("Please draw me a picture of rivers and lakes.")
```

## Engines

You’re free to create and use your own engines to be usable by the Agents framework.
These engines have the following specification:

1. Follow the [messages format](../chat_templating.md) for its input (`List[Dict[str, str]]`) and return a string.
2. Stop generating outputs *before* the sequences passed in the argument `stop_sequences`

### TransformersEngine

For convenience, we have added a `TransformersEngine` that implements the points above, taking a pre-initialized `Pipeline` as input.

```
>>> from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, TransformersEngine

>>> model_name = "HuggingFaceTB/SmolLM-135M-Instruct"
>>> tokenizer = AutoTokenizer.from_pretrained(model_name)
>>> model = AutoModelForCausalLM.from_pretrained(model_name)

>>> pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

>>> engine = TransformersEngine(pipe)
>>> engine([{"role": "user", "content": "Ok!"}], stop_sequences=["great"])

"What a "
```

### class transformers.TransformersEngine

 [< source >](https://github.com/huggingface/transformers/blob/main/src/transformers/agents/llm_engine.py#L201)

( pipeline: Pipeline model\_id: typing.Optional[str] = None  )

This engine uses a pre-initialized local text-generation pipeline.

### HfApiEngine

The `HfApiEngine` is an engine that wraps an [HF Inference API](https://huggingface.co/docs/api-inference/index) client for the execution of the LLM.

```
>>> from transformers import HfApiEngine

>>> messages = [
...   {"role": "user", "content": "Hello, how are you?"},
...   {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
...   {"role": "user", "content": "No need to help, take it easy."},
... ]

>>> HfApiEngine()(messages, stop_sequences=["conversation"])

"That's very kind of you to say! It's always nice to have a relaxed "
```

### class transformers.HfApiEngine

 [< source >](https://github.com/huggingface/transformers/blob/main/src/transformers/agents/llm_engine.py#L150)

( model: str = 'meta-llama/Meta-Llama-3.1-8B-Instruct' token: typing.Optional[str] = None max\_tokens: typing.Optional[int] = 1500 timeout: typing.Optional[int] = 120  )

Parameters

* **model** (`str`, *optional*, defaults to `"meta-llama/Meta-Llama-3.1-8B-Instruct"`) —
  The Hugging Face model ID to be used for inference. This can be a path or model identifier from the Hugging Face model hub.
* **token** (`str`, *optional*) —
  Token used by the Hugging Face API for authentication.
  If not provided, the class will use the token stored in the Hugging Face CLI configuration.
* **max\_tokens** (`int`, *optional*, defaults to 1500) —
  The maximum number of tokens allowed in the output.
* **timeout** (`int`, *optional*, defaults to 120) —
  Timeout for the API request, in seconds.

Raises

`ValueError`

* `ValueError` —
  If the model name is not provided.

A class to interact with Hugging Face’s Inference API for language model interaction.

This engine allows you to communicate with Hugging Face’s models using the Inference API. It can be used in both serverless mode or with a dedicated endpoint, supporting features like stop sequences and grammar customization.

## Agent Types

Agents can handle any type of object in-between tools; tools, being completely multimodal, can accept and return
text, image, audio, video, among other types. In order to increase compatibility between tools, as well as to
correctly render these returns in ipython (jupyter, colab, ipython notebooks, …), we implement wrapper classes
around these types.

The wrapped objects should continue behaving as initially; a text object should still behave as a string, an image
object should still behave as a `PIL.Image`.

These types have three specific purposes:

* Calling `to_raw` on the type should return the underlying object
* Calling `to_string` on the type should return the object as a string: that can be the string in case of an `AgentText`
  but will be the path of the serialized version of the object in other instances
* Displaying it in an ipython kernel should display the object correctly

### AgentText

### class transformers.agents.agent\_types.AgentText

 [< source >](https://github.com/huggingface/transformers/blob/main/src/transformers/agents/agent_types.py#L73)

( value  )

Text type returned by the agent. Behaves as a string.

### AgentImage

### class transformers.agents.agent\_types.AgentImage

 [< source >](https://github.com/huggingface/transformers/blob/main/src/transformers/agents/agent_types.py#L85)

( value  )

Image type returned by the agent. Behaves as a PIL.Image.

#### save

 [< source >](https://github.com/huggingface/transformers/blob/main/src/transformers/agents/agent_types.py#L162)

( output\_bytes format \*\*params  )

Parameters

* **output\_bytes** (bytes) — The output bytes to save the image to.
* **format** (str) — The format to use for the output image. The format is the same as in PIL.Image.save.
* \***\*params** — Additional parameters to pass to PIL.Image.save.

Saves the image to a file.

#### to\_raw

 [< source >](https://github.com/huggingface/transformers/blob/main/src/transformers/agents/agent_types.py#L120)

( )

Returns the “raw” version of that object. In the case of an AgentImage, it is a PIL.Image.

#### to\_string

 [< source >](https://github.com/huggingface/transformers/blob/main/src/transformers/agents/agent_types.py#L135)

( )

Returns the stringified version of that object. In the case of an AgentImage, it is a path to the serialized
version of the image.

### AgentAudio

### class transformers.agents.agent\_types.AgentAudio

 [< source >](https://github.com/huggingface/transformers/blob/main/src/transformers/agents/agent_types.py#L174)

( value samplerate = 16000  )

Audio type returned by the agent.

#### to\_raw

 [< source >](https://github.com/huggingface/transformers/blob/main/src/transformers/agents/agent_types.py#L210)

( )

Returns the “raw” version of that object. It is a `torch.Tensor` object.

#### to\_string

 [< source >](https://github.com/huggingface/transformers/blob/main/src/transformers/agents/agent_types.py#L222)

( )

Returns the stringified version of that object. In the case of an AgentAudio, it is a path to the serialized
version of the audio.

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/main_classes/agent.md)
