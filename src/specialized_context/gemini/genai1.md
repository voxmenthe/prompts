ContentsMenuExpandLight modeDark modeAuto light/dark, in light modeAuto light/dark, in dark mode[Skip to content](https://googleapis.github.io/python-genai/#furo-main-content)

[Back to top](https://googleapis.github.io/python-genai/#)

[View this page](https://googleapis.github.io/python-genai/_sources/index.rst.txt "View this page")

Toggle Light / Dark / Auto color theme

Toggle table of contents sidebar

# Google Gen AI SDK [¶](https://googleapis.github.io/python-genai/\#google-gen-ai-sdk "Link to this heading")

[![pypi](https://img.shields.io/pypi/v/google-genai.svg)](https://pypi.org/project/google-genai/)

[https://github.com/googleapis/python-genai](https://github.com/googleapis/python-genai)

**google-genai** is an initial Python client library for interacting with
Google’s Generative AI APIs.

Google Gen AI Python SDK provides an interface for developers to integrate Google’s generative models into their Python applications. It supports the [Gemini Developer API](https://ai.google.dev/gemini-api/docs) and [Vertex AI](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/overview) APIs.

## Installation [¶](https://googleapis.github.io/python-genai/\#installation "Link to this heading")

```
pip install google-genai

```

## Imports [¶](https://googleapis.github.io/python-genai/\#imports "Link to this heading")

```
from google import genai
from google.genai import types

```

## Create a client [¶](https://googleapis.github.io/python-genai/\#create-a-client "Link to this heading")

Please run one of the following code blocks to create a client for
different services ( [Gemini Developer API](https://ai.google.dev/gemini-api/docs) or [Vertex AI](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/overview)). Feel free to switch the client and
run all the examples to see how it behaves under different APIs.

```
from google import genai

# Only run this block for Gemini Developer API
client = genai.Client(api_key='GEMINI_API_KEY')

```

```
from google import genai

# Only run this block for Vertex AI API
client = genai.Client(
    vertexai=True, project='your-project-id', location='us-central1'
)

```

**(Optional) Using environment variables:**

You can create a client by configuring the necessary environment variables.
Configuration setup instructions depends on whether you’re using the Gemini
Developer API or the Gemini API in Vertex AI.

**Gemini Developer API:** Set GOOGLE\_API\_KEY as shown below:

```
export GOOGLE_API_KEY='your-api-key'

```

**Gemini API in Vertex AI:** Set GOOGLE\_GENAI\_USE\_VERTEXAI, GOOGLE\_CLOUD\_PROJECT
and GOOGLE\_CLOUD\_LOCATION, as shown below:

```
export GOOGLE_GENAI_USE_VERTEXAI=true
export GOOGLE_CLOUD_PROJECT='your-project-id'
export GOOGLE_CLOUD_LOCATION='us-central1'

```

```
from google import genai

client = genai.Client()

```

### API Selection [¶](https://googleapis.github.io/python-genai/\#api-selection "Link to this heading")

By default, the SDK uses the beta API endpoints provided by Google to support preview features in the APIs. The stable API endpoints can be selected by setting the API version to v1.

To set the API version use `http_options`. For example, to set the API version to `v1` for Vertex AI:

```
from google import genai
from google.genai import types

client = genai.Client(
    vertexai=True,
    project='your-project-id',
    location='us-central1',
    http_options=types.HttpOptions(api_version='v1')
)

```

To set the API version to v1alpha for the Gemini Developer API:

```
from google import genai
from google.genai import types

# Only run this block for Gemini Developer API
client = genai.Client(
    api_key='GEMINI_API_KEY',
    http_options=types.HttpOptions(api_version='v1alpha')
)

```

### Faster async client option: Aiohttp [¶](https://googleapis.github.io/python-genai/\#faster-async-client-option-aiohttp "Link to this heading")

By default we use httpx for both sync and async client implementations. In order
to have faster performance, you may install google-genai\[aiohttp\]. In Gen AI
SDK we configure trust\_env=True to match with the default behavior of httpx.
Additional args of aiohttp.ClientSession.request()
( [see \_RequestOptions args](https://github.com/aio-libs/aiohttp/blob/v3.12.13/aiohttp/client.py#L170))
can be passed through the following way:

```
http_options = types.HttpOptions(
    async_client_args={'cookies': ..., 'ssl': ...},
)

client=Client(..., http_options=http_options)

```

### Proxy [¶](https://googleapis.github.io/python-genai/\#proxy "Link to this heading")

Both httpx and aiohttp libraries use urllib.request.getproxies from
environment variables. Before client initialization, you may set proxy (and
optional SSL\_CERT\_FILE) by setting the environment variables:

```
export HTTPS_PROXY='http://username:password@proxy_uri:port'
export SSL_CERT_FILE='client.pem'

```

If you need socks5 proxy, httpx [supports](https://www.python-httpx.org/advanced/proxies/#socks) socks5 proxy if you pass it via args to httpx.Client(). You may install
httpx\[socks\] to use it. Then you can pass it through the following way:

```
http_options = types.HttpOptions(
    client_args={'proxy': 'socks5://user:pass@host:port'},
    async_client_args={'proxy': 'socks5://user:pass@host:port'},
)

client=Client(..., http_options=http_options)

```

## Types [¶](https://googleapis.github.io/python-genai/\#types "Link to this heading")

Parameter types can be specified as either dictionaries( `TypedDict`) or [Pydantic Models](https://pydantic.readthedocs.io/en/stable/model.html).
Pydantic model types are available in the `types` module.

# Models [¶](https://googleapis.github.io/python-genai/\#models "Link to this heading")

The `client.models` modules exposes model inferencing and model
getters. See the ‘Create a client’ section above to initialize a client.

## Generate Content [¶](https://googleapis.github.io/python-genai/\#generate-content "Link to this heading")

### with text content [¶](https://googleapis.github.io/python-genai/\#with-text-content "Link to this heading")

```
response = client.models.generate_content(
    model='gemini-2.0-flash-001', contents='Why is the sky blue?'
)
print(response.text)

```

### with uploaded file (Gemini Developer API only) [¶](https://googleapis.github.io/python-genai/\#with-uploaded-file-gemini-developer-api-only "Link to this heading")

download the file in console.

```
!wget -q https://storage.googleapis.com/generativeai-downloads/data/a11.txt

```

python code.

```
file = client.files.upload(file='a11.txt')
response = client.models.generate_content(
    model='gemini-2.0-flash-001',
    contents=['Could you summarize this file?', file]
)
print(response.text)

```

### How to structure contents argument for generate\_content [¶](https://googleapis.github.io/python-genai/\#how-to-structure-contents-argument-for-generate-content "Link to this heading")

The SDK always converts the inputs to the contents argument into
list\[types.Content\].
The following shows some common ways to provide your inputs.

#### Provide a list\[types.Content\] [¶](https://googleapis.github.io/python-genai/\#provide-a-list-types-content "Link to this heading")

This is the canonical way to provide contents, SDK will not do any conversion.

#### Provide a types.Content instance [¶](https://googleapis.github.io/python-genai/\#provide-a-types-content-instance "Link to this heading")

```
from google.genai import types

contents = types.Content(
role='user',
parts=[types.Part.from_text(text='Why is the sky blue?')]
)

```

SDK converts this to

```
[\
types.Content(\
    role='user',\
    parts=[types.Part.from_text(text='Why is the sky blue?')]\
)\
]

```

#### Provide a string [¶](https://googleapis.github.io/python-genai/\#provide-a-string "Link to this heading")

```
contents='Why is the sky blue?'

```

The SDK will assume this is a text part, and it converts this into the following:

```
[\
types.UserContent(\
    parts=[\
    types.Part.from_text(text='Why is the sky blue?')\
    ]\
)\
]

```

Where a types.UserContent is a subclass of types.Content, it sets the
role field to be user.

#### Provide a list of string [¶](https://googleapis.github.io/python-genai/\#provide-a-list-of-string "Link to this heading")

The SDK assumes these are 2 text parts, it converts this into a single content,
like the following:

```
[\
types.UserContent(\
    parts=[\
    types.Part.from_text(text='Why is the sky blue?'),\
    types.Part.from_text(text='Why is the cloud white?'),\
    ]\
)\
]

```

Where a types.UserContent is a subclass of types.Content, the
role field in types.UserContent is fixed to be user.

#### Provide a function call part [¶](https://googleapis.github.io/python-genai/\#provide-a-function-call-part "Link to this heading")

```
from google.genai import types

contents = types.Part.from_function_call(
name='get_weather_by_location',
args={'location': 'Boston'}
)

```

The SDK converts a function call part to a content with a model role:

```
[\
types.ModelContent(\
    parts=[\
    types.Part.from_function_call(\
        name='get_weather_by_location',\
        args={'location': 'Boston'}\
    )\
    ]\
)\
]

```

Where a types.ModelContent is a subclass of types.Content, the
role field in types.ModelContent is fixed to be model.

#### Provide a list of function call parts [¶](https://googleapis.github.io/python-genai/\#provide-a-list-of-function-call-parts "Link to this heading")

```
from google.genai import types

contents = [\
types.Part.from_function_call(\
    name='get_weather_by_location',\
    args={'location': 'Boston'}\
),\
types.Part.from_function_call(\
    name='get_weather_by_location',\
    args={'location': 'New York'}\
),\
]

```

The SDK converts a list of function call parts to the a content with a model role:

```
[\
types.ModelContent(\
    parts=[\
    types.Part.from_function_call(\
        name='get_weather_by_location',\
        args={'location': 'Boston'}\
    ),\
    types.Part.from_function_call(\
        name='get_weather_by_location',\
        args={'location': 'New York'}\
    )\
    ]\
)\
]

```

Where a types.ModelContent is a subclass of types.Content, the
role field in types.ModelContent is fixed to be model.

#### Provide a non function call part [¶](https://googleapis.github.io/python-genai/\#provide-a-non-function-call-part "Link to this heading")

```
from google.genai import types

contents = types.Part.from_uri(
file_uri: 'gs://generativeai-downloads/images/scones.jpg',
mime_type: 'image/jpeg',
)

```

The SDK converts all non function call parts into a content with a user role.

```
[\
types.UserContent(parts=[\
    types.Part.from_uri(\
    file_uri: 'gs://generativeai-downloads/images/scones.jpg',\
    mime_type: 'image/jpeg',\
    )\
])\
]

```

#### Provide a list of non function call parts [¶](https://googleapis.github.io/python-genai/\#provide-a-list-of-non-function-call-parts "Link to this heading")

```
from google.genai import types

contents = [\
types.Part.from_text('What is this image about?'),\
types.Part.from_uri(\
    file_uri: 'gs://generativeai-downloads/images/scones.jpg',\
    mime_type: 'image/jpeg',\
)\
]

```

The SDK will convert the list of parts into a content with a user role

```
[\
types.UserContent(\
    parts=[\
    types.Part.from_text('What is this image about?'),\
    types.Part.from_uri(\
        file_uri: 'gs://generativeai-downloads/images/scones.jpg',\
        mime_type: 'image/jpeg',\
    )\
    ]\
)\
]

```

#### Mix types in contents [¶](https://googleapis.github.io/python-genai/\#mix-types-in-contents "Link to this heading")

You can also provide a list of types.ContentUnion. The SDK leaves items of
types.Content as is, it groups consecutive non function call parts into a
single types.UserContent, and it groups consecutive function call parts into
a single types.ModelContent.

If you put a list within a list, the inner list can only contain
types.PartUnion items. The SDK will convert the inner list into a single
types.UserContent.

## System Instructions and Other Configs [¶](https://googleapis.github.io/python-genai/\#system-instructions-and-other-configs "Link to this heading")

The output of the model can be influenced by several optional settings
available in generate\_content’s config parameter. For example, increasing
max\_output\_tokens is essential for longer model responses. To make a model more
deterministic, lowering the temperature parameter reduces randomness, with
values near 0 minimizing variability. Capabilities and parameter defaults for
each model is shown in the
[Vertex AI docs](https://cloud.google.com/vertex-ai/generative-ai/docs/models/gemini/2-5-flash)
and [Gemini API docs](https://ai.google.dev/gemini-api/docs/models) respectively.

```
from google.genai import types

response = client.models.generate_content(
    model='gemini-2.0-flash-001',
    contents='high',
    config=types.GenerateContentConfig(
        system_instruction='I say high, you say low',
        max_output_tokens=3,
        temperature=0.3,
    ),
)
print(response.text)

```

## Typed Config [¶](https://googleapis.github.io/python-genai/\#typed-config "Link to this heading")

All API methods support Pydantic types for parameters as well as
dictionaries. You can get the type from `google.genai.types`.

```
from google.genai import types

response = client.models.generate_content(
    model='gemini-2.0-flash-001',
    contents=types.Part.from_text(text='Why is the sky blue?'),
    config=types.GenerateContentConfig(
        temperature=0,
        top_p=0.95,
        top_k=20,
        candidate_count=1,
        seed=5,
        max_output_tokens=100,
        stop_sequences=['STOP!'],
        presence_penalty=0.0,
        frequency_penalty=0.0,
    ),
)

print(response.text)

```

## List Base Models [¶](https://googleapis.github.io/python-genai/\#list-base-models "Link to this heading")

To retrieve tuned models, see: List Tuned Models

```
for model in client.models.list():
    print(model)

```

```
pager = client.models.list(config={'page_size': 10})
print(pager.page_size)
print(pager[0])
pager.next_page()
print(pager[0])

```

### List Base Models (Asynchronous) [¶](https://googleapis.github.io/python-genai/\#list-base-models-asynchronous "Link to this heading")

```
async for job in await client.aio.models.list():
    print(job)

```

```
async_pager = await client.aio.models.list(config={'page_size': 10})
print(async_pager.page_size)
print(async_pager[0])
await async_pager.next_page()
print(async_pager[0])

```

## Safety Settings [¶](https://googleapis.github.io/python-genai/\#safety-settings "Link to this heading")

```
from google.genai import types

response = client.models.generate_content(
    model='gemini-2.0-flash-001',
    contents='Say something bad.',
    config=types.GenerateContentConfig(
        safety_settings=[\
            types.SafetySetting(\
                category='HARM_CATEGORY_HATE_SPEECH',\
                threshold='BLOCK_ONLY_HIGH',\
            )\
        ]
    ),
)
print(response.text)

```

## Function Calling [¶](https://googleapis.github.io/python-genai/\#function-calling "Link to this heading")

Automatic Python function Support:

You can pass a Python function directly and it will be automatically
called and responded.

```
from google.genai import types

def get_current_weather(location: str) -> str:
    """Returns the current weather.

    Args:
      location: The city and state, e.g. San Francisco, CA
    """
    return 'sunny'

response = client.models.generate_content(
    model='gemini-2.0-flash-001',
    contents='What is the weather like in Boston?',
    config=types.GenerateContentConfig(
        tools=[get_current_weather],
    ),
)

print(response.text)

```

### Disabling automatic function calling [¶](https://googleapis.github.io/python-genai/\#disabling-automatic-function-calling "Link to this heading")

If you pass in a python function as a tool directly, and do not want
automatic function calling, you can disable automatic function calling
as follows:

```
from google.genai import types

response = client.models.generate_content(
    model='gemini-2.0-flash-001',
    contents='What is the weather like in Boston?',
    config=types.GenerateContentConfig(
        tools=[get_current_weather],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(
            disable=True
        ),
    ),
)

```

With automatic function calling disabled, you will get a list of function call
parts in the response:

### Manually declare and invoke a function for function calling [¶](https://googleapis.github.io/python-genai/\#manually-declare-and-invoke-a-function-for-function-calling "Link to this heading")

If you don’t want to use the automatic function support, you can manually
declare the function and invoke it.

The following example shows how to declare a function and pass it as a tool.
Then you will receive a function call part in the response.

```
from google.genai import types

function = types.FunctionDeclaration(
    name='get_current_weather',
    description='Get the current weather in a given location',
    parameters=types.Schema(
        type='OBJECT',
        properties={
            'location': types.Schema(
                type='STRING',
                description='The city and state, e.g. San Francisco, CA',
            ),
        },
        required=['location'],
    ),
)

tool = types.Tool(function_declarations=[function])

response = client.models.generate_content(
    model='gemini-2.0-flash-001',
    contents='What is the weather like in Boston?',
    config=types.GenerateContentConfig(
        tools=[tool],
    ),
)
print(response.function_calls[0])

```

After you receive the function call part from the model, you can invoke the function
and get the function response. And then you can pass the function response to
the model.
The following example shows how to do it for a simple function invocation.

```
from google.genai import types

user_prompt_content = types.Content(
    role='user',
    parts=[types.Part.from_text(text='What is the weather like in Boston?')],
)
function_call_part = response.function_calls[0]
function_call_content = response.candidates[0].content

try:
    function_result = get_current_weather(
        **function_call_part.function_call.args
    )
    function_response = {'result': function_result}
except (
    Exception
) as e:  # instead of raising the exception, you can let the model handle it
    function_response = {'error': str(e)}

function_response_part = types.Part.from_function_response(
    name=function_call_part.name,
    response=function_response,
)
function_response_content = types.Content(
    role='tool', parts=[function_response_part]
)

response = client.models.generate_content(
    model='gemini-2.0-flash-001',
    contents=[\
        user_prompt_content,\
        function_call_content,\
        function_response_content,\
    ],
    config=types.GenerateContentConfig(
        tools=[tool],
    ),
)

print(response.text)

```

### Function calling with `ANY` tools config mode [¶](https://googleapis.github.io/python-genai/\#function-calling-with-any-tools-config-mode "Link to this heading")

If you configure function calling mode to be ANY, then the model will always
return function call parts. If you also pass a python function as a tool, by
default the SDK will perform automatic function calling until the remote calls
exceed the maximum remote call for automatic function calling (default to 10 times).

If you’d like to disable automatic function calling in ANY mode:

```
from google.genai import types

def get_current_weather(location: str) -> str:
    """Returns the current weather.

    Args:
        location: The city and state, e.g. San Francisco, CA
    """
    return "sunny"

response = client.models.generate_content(
    model="gemini-2.0-flash-001",
    contents="What is the weather like in Boston?",
    config=types.GenerateContentConfig(
        tools=[get_current_weather],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(
            disable=True
        ),
        tool_config=types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(mode='ANY')
        ),
    ),
)

```

If you’d like to set `x` number of automatic function call turns, you can
configure the maximum remote calls to be `x + 1`.
Assuming you prefer `1` turn for automatic function calling:

```
from google.genai import types

def get_current_weather(location: str) -> str:
    """Returns the current weather.

    Args:
        location: The city and state, e.g. San Francisco, CA
    """
    return "sunny"

response = client.models.generate_content(
    model="gemini-2.0-flash-001",
    contents="What is the weather like in Boston?",
    config=types.GenerateContentConfig(
        tools=[get_current_weather],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(
            maximum_remote_calls=2
        ),
        tool_config=types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(mode='ANY')
        ),
    ),
)

```

## JSON Response Schema [¶](https://googleapis.github.io/python-genai/\#json-response-schema "Link to this heading")

### Pydantic Model Schema support [¶](https://googleapis.github.io/python-genai/\#pydantic-model-schema-support "Link to this heading")

Schemas can be provided as Pydantic Models.

```
from pydantic import BaseModel
from google.genai import types

class CountryInfo(BaseModel):
    name: str
    population: int
    capital: str
    continent: str
    gdp: int
    official_language: str
    total_area_sq_mi: int

response = client.models.generate_content(
    model='gemini-2.0-flash-001',
    contents='Give me information for the United States.',
    config=types.GenerateContentConfig(
        response_mime_type='application/json',
        response_schema=CountryInfo,
    ),
)
print(response.text)

```

```
from google.genai import types

response = client.models.generate_content(
    model='gemini-2.0-flash-001',
    contents='Give me information for the United States.',
    config=types.GenerateContentConfig(
        response_mime_type='application/json',
        response_schema={
            'required': [\
                'name',\
                'population',\
                'capital',\
                'continent',\
                'gdp',\
                'official_language',\
                'total_area_sq_mi',\
            ],
            'properties': {
                'name': {'type': 'STRING'},
                'population': {'type': 'INTEGER'},
                'capital': {'type': 'STRING'},
                'continent': {'type': 'STRING'},
                'gdp': {'type': 'INTEGER'},
                'official_language': {'type': 'STRING'},
                'total_area_sq_mi': {'type': 'INTEGER'},
            },
            'type': 'OBJECT',
        },
    ),
)
print(response.text)

```

## Enum Response Schema [¶](https://googleapis.github.io/python-genai/\#enum-response-schema "Link to this heading")

### Text Response [¶](https://googleapis.github.io/python-genai/\#text-response "Link to this heading")

You can set response\_mime\_type to ‘text/x.enum’ to return one of those enum
values as the response.

```
from enum import Enum

class InstrumentEnum(Enum):
    PERCUSSION = 'Percussion'
    STRING = 'String'
    WOODWIND = 'Woodwind'
    BRASS = 'Brass'
    KEYBOARD = 'Keyboard'

response = client.models.generate_content(
    model='gemini-2.0-flash-001',
    contents='What instrument plays multiple notes at once?',
    config={
        'response_mime_type': 'text/x.enum',
        'response_schema': InstrumentEnum,
    },
)
print(response.text)

```

### JSON Response [¶](https://googleapis.github.io/python-genai/\#json-response "Link to this heading")

You can also set response\_mime\_type to ‘application/json’, the response will be
identical but in quotes.

```
class InstrumentEnum(Enum):
    PERCUSSION = 'Percussion'
    STRING = 'String'
    WOODWIND = 'Woodwind'
    BRASS = 'Brass'
    KEYBOARD = 'Keyboard'

response = client.models.generate_content(
    model='gemini-2.0-flash-001',
    contents='What instrument plays multiple notes at once?',
    config={
        'response_mime_type': 'application/json',
        'response_schema': InstrumentEnum,
    },
)
print(response.text)

```

## Generate Content (Synchronous Streaming) [¶](https://googleapis.github.io/python-genai/\#generate-content-synchronous-streaming "Link to this heading")

Generate content in a streaming format so that the model outputs streams back
to you, rather than being returned as one chunk.

### Streaming for text content [¶](https://googleapis.github.io/python-genai/\#streaming-for-text-content "Link to this heading")

```
for chunk in client.models.generate_content_stream(
    model='gemini-2.0-flash-001', contents='Tell me a story in 300 words.'
):
    print(chunk.text, end='')

```

### Streaming for image content [¶](https://googleapis.github.io/python-genai/\#streaming-for-image-content "Link to this heading")

If your image is stored in [Google Cloud Storage](https://cloud.google.com/storage), you can use the from\_uri class method to create a Part object.

```
from google.genai import types

for chunk in client.models.generate_content_stream(
    model='gemini-2.0-flash-001',
    contents=[\
        'What is this image about?',\
        types.Part.from_uri(\
            file_uri='gs://generativeai-downloads/images/scones.jpg',\
            mime_type='image/jpeg',\
        ),\
    ],
):
    print(chunk.text, end='')

```

If your image is stored in your local file system, you can read it in as bytes
data and use the `from_bytes` class method to create a `Part` object.

```
from google.genai import types

YOUR_IMAGE_PATH = 'your_image_path'
YOUR_IMAGE_MIME_TYPE = 'your_image_mime_type'
with open(YOUR_IMAGE_PATH, 'rb') as f:
    image_bytes = f.read()

for chunk in client.models.generate_content_stream(
    model='gemini-2.0-flash-001',
    contents=[\
        'What is this image about?',\
        types.Part.from_bytes(data=image_bytes, mime_type=YOUR_IMAGE_MIME_TYPE),\
    ],
):
    print(chunk.text, end='')

```

## Generate Content (Asynchronous Non-Streaming) [¶](https://googleapis.github.io/python-genai/\#generate-content-asynchronous-non-streaming "Link to this heading")

`client.aio` exposes all the analogous [async methods](https://docs.python.org/3/library/asyncio.html) that are available on `client`.
Note that it applies to all the modules.

For example, `client.aio.models.generate_content` is the `async` version of `client.models.generate_content`

```
response = await client.aio.models.generate_content(
    model='gemini-2.0-flash-001', contents='Tell me a story in 300 words.'
)

print(response.text)

```

## Generate Content (Asynchronous Streaming) [¶](https://googleapis.github.io/python-genai/\#generate-content-asynchronous-streaming "Link to this heading")

```
async for chunk in await client.aio.models.generate_content_stream(
    model='gemini-2.0-flash-001', contents='Tell me a story in 300 words.'
):
    print(chunk.text, end='')

```

## Count Tokens and Compute Tokens [¶](https://googleapis.github.io/python-genai/\#count-tokens-and-compute-tokens "Link to this heading")

```
response = client.models.count_tokens(
    model='gemini-2.0-flash-001',
    contents='why is the sky blue?',
)
print(response)

```

### Compute Tokens [¶](https://googleapis.github.io/python-genai/\#compute-tokens "Link to this heading")

Compute tokens is only supported in Vertex AI.

```
response = client.models.compute_tokens(
    model='gemini-2.0-flash-001',
    contents='why is the sky blue?',
)
print(response)

```

### Count Tokens (Asynchronous) [¶](https://googleapis.github.io/python-genai/\#count-tokens-asynchronous "Link to this heading")

```
response = await client.aio.models.count_tokens(
    model='gemini-2.0-flash-001',
    contents='why is the sky blue?',
)
print(response)

```

## Embed Content [¶](https://googleapis.github.io/python-genai/\#embed-content "Link to this heading")

```
response = client.models.embed_content(
    model='text-embedding-004',
    contents='why is the sky blue?',
)
print(response)

```

```
from google.genai import types

# multiple contents with config
response = client.models.embed_content(
    model='text-embedding-004',
    contents=['why is the sky blue?', 'What is your age?'],
    config=types.EmbedContentConfig(output_dimensionality=10),
)

print(response)

```

## Imagen [¶](https://googleapis.github.io/python-genai/\#imagen "Link to this heading")

### Generate Image [¶](https://googleapis.github.io/python-genai/\#generate-image "Link to this heading")

Support for generate image in Gemini Developer API is behind an allowlist

```
from google.genai import types

# Generate Image
response1 = client.models.generate_images(
    model='imagen-3.0-generate-002',
    prompt='An umbrella in the foreground, and a rainy night sky in the background',
    config=types.GenerateImagesConfig(
        number_of_images=1,
        include_rai_reason=True,
        output_mime_type='image/jpeg',
    ),
)
response1.generated_images[0].image.show()

```

Upscale image is only supported in Vertex AI.

```
from google.genai import types

# Upscale the generated image from above
response2 = client.models.upscale_image(
    model='imagen-3.0-generate-002',
    image=response1.generated_images[0].image,
    upscale_factor='x2',
    config=types.UpscaleImageConfig(
        include_rai_reason=True,
        output_mime_type='image/jpeg',
    ),
)
response2.generated_images[0].image.show()

```

### Edit Image [¶](https://googleapis.github.io/python-genai/\#edit-image "Link to this heading")

Edit image uses a separate model from generate and upscale.

Edit image is only supported in Vertex AI.

```
# Edit the generated image from above
from google.genai import types
from google.genai.types import RawReferenceImage, MaskReferenceImage

raw_ref_image = RawReferenceImage(
    reference_id=1,
    reference_image=response1.generated_images[0].image,
)

# Model computes a mask of the background
mask_ref_image = MaskReferenceImage(
    reference_id=2,
    config=types.MaskReferenceConfig(
        mask_mode='MASK_MODE_BACKGROUND',
        mask_dilation=0,
    ),
)

response3 = client.models.edit_image(
    model='imagen-3.0-capability-001',
    prompt='Sunlight and clear sky',
    reference_images=[raw_ref_image, mask_ref_image],
    config=types.EditImageConfig(
        edit_mode='EDIT_MODE_INPAINT_INSERTION',
        number_of_images=1,
        include_rai_reason=True,
        output_mime_type='image/jpeg',
    ),
)
response3.generated_images[0].image.show()

```

## Veo [¶](https://googleapis.github.io/python-genai/\#veo "Link to this heading")

### Generate Videos [¶](https://googleapis.github.io/python-genai/\#generate-videos "Link to this heading")

Support for generate videos in Vertex and Gemini Developer API is behind an allowlist

```
from google.genai import types

# Create operation
operation = client.models.generate_videos(
    model='veo-2.0-generate-001',
    prompt='A neon hologram of a cat driving at top speed',
    config=types.GenerateVideosConfig(
        number_of_videos=1,
        fps=24,
        duration_seconds=5,
        enhance_prompt=True,
    ),
)

# Poll operation
while not operation.done:
    time.sleep(20)
    operation = client.operations.get(operation)

video = operation.result.generated_videos[0].video
video.show()

```

# Chats [¶](https://googleapis.github.io/python-genai/\#chats "Link to this heading")

Create a chat session to start a multi-turn conversations with the model. Then,
use chat.send\_message function multiple times within the same chat session so
that it can reflect on its previous responses (i.e., engage in an ongoing
conversation). See the ‘Create a client’ section above to initialize a client.

## Send Message (Synchronous Non-Streaming) [¶](https://googleapis.github.io/python-genai/\#send-message-synchronous-non-streaming "Link to this heading")

```
chat = client.chats.create(model='gemini-2.0-flash-001')
response = chat.send_message('tell me a story')
print(response.text)
response = chat.send_message('summarize the story you told me in 1 sentence')
print(response.text)

```

## Send Message (Synchronous Streaming) [¶](https://googleapis.github.io/python-genai/\#send-message-synchronous-streaming "Link to this heading")

```
chat = client.chats.create(model='gemini-2.0-flash-001')
for chunk in chat.send_message_stream('tell me a story'):
    print(chunk.text, end='')  # end='' is optional, for demo purposes.

```

## Send Message (Asynchronous Non-Streaming) [¶](https://googleapis.github.io/python-genai/\#send-message-asynchronous-non-streaming "Link to this heading")

```
chat = client.aio.chats.create(model='gemini-2.0-flash-001')
response = await chat.send_message('tell me a story')
print(response.text)

```

## Send Message (Asynchronous Streaming) [¶](https://googleapis.github.io/python-genai/\#send-message-asynchronous-streaming "Link to this heading")

```
chat = client.aio.chats.create(model='gemini-2.0-flash-001')
async for chunk in await chat.send_message_stream('tell me a story'):
    print(chunk.text, end='') # end='' is optional, for demo purposes.

```

# Files [¶](https://googleapis.github.io/python-genai/\#files "Link to this heading")

Files are only supported in Gemini Developer API. See the ‘Create a client’
section above to initialize a client.

```
gsutil cp gs://cloud-samples-data/generative-ai/pdf/2312.11805v3.pdf .
gsutil cp gs://cloud-samples-data/generative-ai/pdf/2403.05530.pdf .

```

## Upload [¶](https://googleapis.github.io/python-genai/\#upload "Link to this heading")

```
file1 = client.files.upload(file='2312.11805v3.pdf')
file2 = client.files.upload(file='2403.05530.pdf')

print(file1)
print(file2)

```

## Get [¶](https://googleapis.github.io/python-genai/\#get "Link to this heading")

```
file1 = client.files.upload(file='2312.11805v3.pdf')
file_info = client.files.get(name=file1.name)

```

## Delete [¶](https://googleapis.github.io/python-genai/\#delete "Link to this heading")

```
file3 = client.files.upload(file='2312.11805v3.pdf')

client.files.delete(name=file3.name)

```

# Caches [¶](https://googleapis.github.io/python-genai/\#caches "Link to this heading")

`client.caches` contains the control plane APIs for cached content.

See the ‘Create a client’ section above to initialize a client.

## Create [¶](https://googleapis.github.io/python-genai/\#create "Link to this heading")

```
from google.genai import types

if client.vertexai:
    file_uris = [\
        'gs://cloud-samples-data/generative-ai/pdf/2312.11805v3.pdf',\
        'gs://cloud-samples-data/generative-ai/pdf/2403.05530.pdf',\
    ]
else:
    file_uris = [file1.uri, file2.uri]

cached_content = client.caches.create(
    model='gemini-2.0-flash-001',
    config=types.CreateCachedContentConfig(
        contents=[\
            types.Content(\
                role='user',\
                parts=[\
                    types.Part.from_uri(\
                        file_uri=file_uris[0], mime_type='application/pdf'\
                    ),\
                    types.Part.from_uri(\
                        file_uri=file_uris[1],\
                        mime_type='application/pdf',\
                    ),\
                ],\
            )\
        ],
        system_instruction='What is the sum of the two pdfs?',
        display_name='test cache',
        ttl='3600s',
    ),
)

```

## Get [¶](https://googleapis.github.io/python-genai/\#id3 "Link to this heading")

```
cached_content = client.caches.get(name=cached_content.name)

```

## Generate Content with Caches [¶](https://googleapis.github.io/python-genai/\#generate-content-with-caches "Link to this heading")

```
from google.genai import types

response = client.models.generate_content(
    model='gemini-2.0-flash-001',
    contents='Summarize the pdfs',
    config=types.GenerateContentConfig(
        cached_content=cached_content.name,
    ),
)
print(response.text)

```

# Tunings [¶](https://googleapis.github.io/python-genai/\#tunings "Link to this heading")

`client.tunings` contains tuning job APIs and supports supervised fine
tuning through `tune`. See the ‘Create a client’ section above to initialize a
client.

## Tune [¶](https://googleapis.github.io/python-genai/\#tune "Link to this heading")

- Vertex AI supports tuning from GCS source

- Gemini Developer API supports tuning from inline examples


```
from google.genai import types

if client.vertexai:
    model = 'gemini-2.0-flash-001'
    training_dataset = types.TuningDataset(
        gcs_uri='gs://cloud-samples-data/ai-platform/generative_ai/gemini-1_5/text/sft_train_data.jsonl',
    )
else:
    model = 'models/gemini-2.0-flash-001'
    training_dataset = types.TuningDataset(
        examples=[\
            types.TuningExample(\
                text_input=f'Input text {i}',\
                output=f'Output text {i}',\
            )\
            for i in range(5)\
        ],
    )

```

```
from google.genai import types

tuning_job = client.tunings.tune(
    base_model=model,
    training_dataset=training_dataset,
    config=types.CreateTuningJobConfig(
        epoch_count=1, tuned_model_display_name='test_dataset_examples model'
    ),
)
print(tuning_job)

```

## Get Tuning Job [¶](https://googleapis.github.io/python-genai/\#get-tuning-job "Link to this heading")

```
tuning_job = client.tunings.get(name=tuning_job.name)
print(tuning_job)

```

```
import time

running_states = set(
    [\
        'JOB_STATE_PENDING',\
        'JOB_STATE_RUNNING',\
    ]
)

while tuning_job.state in running_states:
    print(tuning_job.state)
    tuning_job = client.tunings.get(name=tuning_job.name)
    time.sleep(10)

```

## Use Tuned Model [¶](https://googleapis.github.io/python-genai/\#use-tuned-model "Link to this heading")

```
response = client.models.generate_content(
    model=tuning_job.tuned_model.endpoint,
    contents='why is the sky blue?',
)

print(response.text)

```

## Get Tuned Model [¶](https://googleapis.github.io/python-genai/\#get-tuned-model "Link to this heading")

```
tuned_model = client.models.get(model=tuning_job.tuned_model.model)
print(tuned_model)

```

## Update Tuned Model [¶](https://googleapis.github.io/python-genai/\#update-tuned-model "Link to this heading")

```
from google.genai import types

tuned_model = client.models.update(
    model=tuning_job.tuned_model.model,
    config=types.UpdateModelConfig(
        display_name='my tuned model', description='my tuned model description'
    ),
)
print(tuned_model)

```

## List Tuned Models [¶](https://googleapis.github.io/python-genai/\#list-tuned-models "Link to this heading")

To retrieve base models, see: List Base Models

```
for model in client.models.list(config={'page_size': 10, 'query_base': False}}):
    print(model)

```

```
pager = client.models.list(config={'page_size': 10, 'query_base': False}})
print(pager.page_size)
print(pager[0])
pager.next_page()
print(pager[0])

```

### List Tuned Models (Asynchronous) [¶](https://googleapis.github.io/python-genai/\#list-tuned-models-asynchronous "Link to this heading")

```
async for job in await client.aio.models.list(config={'page_size': 10, 'query_base': False}}):
    print(job)

```

```
async_pager = await client.aio.models.list(config={'page_size': 10, 'query_base': False}})
print(async_pager.page_size)
print(async_pager[0])
await async_pager.next_page()
print(async_pager[0])

```

## Update Tuned Model [¶](https://googleapis.github.io/python-genai/\#id4 "Link to this heading")

```
model = pager[0]

model = client.models.update(
    model=model.name,
    config=types.UpdateModelConfig(
        display_name='my tuned model', description='my tuned model description'
    ),
)

print(model)

```

## List Tuning Jobs [¶](https://googleapis.github.io/python-genai/\#list-tuning-jobs "Link to this heading")

```
for job in client.tunings.list(config={'page_size': 10}):
    print(job)

```

```
pager = client.tunings.list(config={'page_size': 10})
print(pager.page_size)
print(pager[0])
pager.next_page()
print(pager[0])

```

List Tuning Jobs (Asynchronous):

```
async for job in await client.aio.tunings.list(config={'page_size': 10}):
    print(job)

```

```
async_pager = await client.aio.tunings.list(config={'page_size': 10})
print(async_pager.page_size)
print(async_pager[0])
await async_pager.next_page()
print(async_pager[0])

```

# Batch Prediction [¶](https://googleapis.github.io/python-genai/\#batch-prediction "Link to this heading")

Only supported in Vertex AI. See the ‘Create a client’ section above to
initialize a client.

## Create [¶](https://googleapis.github.io/python-genai/\#id5 "Link to this heading")

```
# Specify model and source file only, destination and job display name will be auto-populated
job = client.batches.create(
    model='gemini-2.0-flash-001',
    src='bq://my-project.my-dataset.my-table',
)

job

```

```
# Get a job by name
job = client.batches.get(name=job.name)

job.state

```

```
completed_states = set(
    [\
        'JOB_STATE_SUCCEEDED',\
        'JOB_STATE_FAILED',\
        'JOB_STATE_CANCELLED',\
        'JOB_STATE_PAUSED',\
    ]
)

while job.state not in completed_states:
    print(job.state)
    job = client.batches.get(name=job.name)
    time.sleep(30)

job

```

## List [¶](https://googleapis.github.io/python-genai/\#list "Link to this heading")

```
from google.genai import types

for job in client.batches.list(config=types.ListBatchJobsConfig(page_size=10)):
    print(job)

```

### List Batch Jobs with Pager [¶](https://googleapis.github.io/python-genai/\#list-batch-jobs-with-pager "Link to this heading")

```
from google.genai import types

pager = client.batches.list(config=types.ListBatchJobsConfig(page_size=10))
print(pager.page_size)
print(pager[0])
pager.next_page()
print(pager[0])

```

### List Batch Jobs (Asynchronous) [¶](https://googleapis.github.io/python-genai/\#list-batch-jobs-asynchronous "Link to this heading")

```
from google.genai import types

async for job in await client.aio.batches.list(
    config=types.ListBatchJobsConfig(page_size=10)
):
    print(job)

```

### List Batch Jobs with Pager (Asynchronous) [¶](https://googleapis.github.io/python-genai/\#list-batch-jobs-with-pager-asynchronous "Link to this heading")

```
from google.genai import types

async_pager = await client.aio.batches.list(
    config=types.ListBatchJobsConfig(page_size=10)
)
print(async_pager.page_size)
print(async_pager[0])
await async_pager.next_page()
print(async_pager[0])

```

## Delete [¶](https://googleapis.github.io/python-genai/\#id6 "Link to this heading")

```
# Delete the job resource
delete_job = client.batches.delete(name=job.name)

delete_job

```

# Error Handling [¶](https://googleapis.github.io/python-genai/\#error-handling "Link to this heading")

To handle errors raised by the model, the SDK provides this [APIError](https://github.com/googleapis/python-genai/blob/main/google/genai/errors.py) class.

```
try:
    client.models.generate_content(
        model="invalid-model-name",
        contents="What is your name?",
    )
except errors.APIError as e:
    print(e.code) # 404
    print(e.message)

```

# Reference [¶](https://googleapis.github.io/python-genai/\#reference "Link to this heading")

- [Submodules](https://googleapis.github.io/python-genai/genai.html)
- [genai.client module](https://googleapis.github.io/python-genai/genai.html#module-genai.client)
  - [`AsyncClient`](https://googleapis.github.io/python-genai/genai.html#genai.client.AsyncClient)
    - [`AsyncClient.auth_tokens`](https://googleapis.github.io/python-genai/genai.html#genai.client.AsyncClient.auth_tokens)
    - [`AsyncClient.batches`](https://googleapis.github.io/python-genai/genai.html#genai.client.AsyncClient.batches)
    - [`AsyncClient.caches`](https://googleapis.github.io/python-genai/genai.html#genai.client.AsyncClient.caches)
    - [`AsyncClient.chats`](https://googleapis.github.io/python-genai/genai.html#genai.client.AsyncClient.chats)
    - [`AsyncClient.files`](https://googleapis.github.io/python-genai/genai.html#genai.client.AsyncClient.files)
    - [`AsyncClient.live`](https://googleapis.github.io/python-genai/genai.html#genai.client.AsyncClient.live)
    - [`AsyncClient.models`](https://googleapis.github.io/python-genai/genai.html#genai.client.AsyncClient.models)
    - [`AsyncClient.operations`](https://googleapis.github.io/python-genai/genai.html#genai.client.AsyncClient.operations)
    - [`AsyncClient.tunings`](https://googleapis.github.io/python-genai/genai.html#genai.client.AsyncClient.tunings)
  - [`Client`](https://googleapis.github.io/python-genai/genai.html#genai.client.Client)
    - [`Client.api_key`](https://googleapis.github.io/python-genai/genai.html#genai.client.Client.api_key)
    - [`Client.vertexai`](https://googleapis.github.io/python-genai/genai.html#genai.client.Client.vertexai)
    - [`Client.credentials`](https://googleapis.github.io/python-genai/genai.html#genai.client.Client.credentials)
    - [`Client.project`](https://googleapis.github.io/python-genai/genai.html#genai.client.Client.project)
    - [`Client.location`](https://googleapis.github.io/python-genai/genai.html#genai.client.Client.location)
    - [`Client.debug_config`](https://googleapis.github.io/python-genai/genai.html#genai.client.Client.debug_config)
    - [`Client.http_options`](https://googleapis.github.io/python-genai/genai.html#genai.client.Client.http_options)
    - [`Client.aio`](https://googleapis.github.io/python-genai/genai.html#genai.client.Client.aio)
    - [`Client.auth_tokens`](https://googleapis.github.io/python-genai/genai.html#genai.client.Client.auth_tokens)
    - [`Client.batches`](https://googleapis.github.io/python-genai/genai.html#genai.client.Client.batches)
    - [`Client.caches`](https://googleapis.github.io/python-genai/genai.html#genai.client.Client.caches)
    - [`Client.chats`](https://googleapis.github.io/python-genai/genai.html#genai.client.Client.chats)
    - [`Client.files`](https://googleapis.github.io/python-genai/genai.html#genai.client.Client.files)
    - [`Client.models`](https://googleapis.github.io/python-genai/genai.html#genai.client.Client.models)
    - [`Client.operations`](https://googleapis.github.io/python-genai/genai.html#genai.client.Client.operations)
    - [`Client.tunings`](https://googleapis.github.io/python-genai/genai.html#genai.client.Client.tunings)
    - [`Client.vertexai`](https://googleapis.github.io/python-genai/genai.html#id0)
  - [`DebugConfig`](https://googleapis.github.io/python-genai/genai.html#genai.client.DebugConfig)
    - [`DebugConfig.client_mode`](https://googleapis.github.io/python-genai/genai.html#genai.client.DebugConfig.client_mode)
    - [`DebugConfig.replay_id`](https://googleapis.github.io/python-genai/genai.html#genai.client.DebugConfig.replay_id)
    - [`DebugConfig.replays_directory`](https://googleapis.github.io/python-genai/genai.html#genai.client.DebugConfig.replays_directory)
- [genai.batches module](https://googleapis.github.io/python-genai/genai.html#module-genai.batches)
  - [`AsyncBatches`](https://googleapis.github.io/python-genai/genai.html#genai.batches.AsyncBatches)
    - [`AsyncBatches.cancel()`](https://googleapis.github.io/python-genai/genai.html#genai.batches.AsyncBatches.cancel)
    - [`AsyncBatches.create()`](https://googleapis.github.io/python-genai/genai.html#genai.batches.AsyncBatches.create)
    - [`AsyncBatches.delete()`](https://googleapis.github.io/python-genai/genai.html#genai.batches.AsyncBatches.delete)
    - [`AsyncBatches.get()`](https://googleapis.github.io/python-genai/genai.html#genai.batches.AsyncBatches.get)
    - [`AsyncBatches.list()`](https://googleapis.github.io/python-genai/genai.html#genai.batches.AsyncBatches.list)
  - [`Batches`](https://googleapis.github.io/python-genai/genai.html#genai.batches.Batches)
    - [`Batches.cancel()`](https://googleapis.github.io/python-genai/genai.html#genai.batches.Batches.cancel)
    - [`Batches.create()`](https://googleapis.github.io/python-genai/genai.html#genai.batches.Batches.create)
    - [`Batches.delete()`](https://googleapis.github.io/python-genai/genai.html#genai.batches.Batches.delete)
    - [`Batches.get()`](https://googleapis.github.io/python-genai/genai.html#genai.batches.Batches.get)
    - [`Batches.list()`](https://googleapis.github.io/python-genai/genai.html#genai.batches.Batches.list)
- [genai.caches module](https://googleapis.github.io/python-genai/genai.html#module-genai.caches)
  - [`AsyncCaches`](https://googleapis.github.io/python-genai/genai.html#genai.caches.AsyncCaches)
    - [`AsyncCaches.create()`](https://googleapis.github.io/python-genai/genai.html#genai.caches.AsyncCaches.create)
    - [`AsyncCaches.delete()`](https://googleapis.github.io/python-genai/genai.html#genai.caches.AsyncCaches.delete)
    - [`AsyncCaches.get()`](https://googleapis.github.io/python-genai/genai.html#genai.caches.AsyncCaches.get)
    - [`AsyncCaches.list()`](https://googleapis.github.io/python-genai/genai.html#genai.caches.AsyncCaches.list)
    - [`AsyncCaches.update()`](https://googleapis.github.io/python-genai/genai.html#genai.caches.AsyncCaches.update)
  - [`Caches`](https://googleapis.github.io/python-genai/genai.html#genai.caches.Caches)
    - [`Caches.create()`](https://googleapis.github.io/python-genai/genai.html#genai.caches.Caches.create)
    - [`Caches.delete()`](https://googleapis.github.io/python-genai/genai.html#genai.caches.Caches.delete)
    - [`Caches.get()`](https://googleapis.github.io/python-genai/genai.html#genai.caches.Caches.get)
    - [`Caches.list()`](https://googleapis.github.io/python-genai/genai.html#genai.caches.Caches.list)
    - [`Caches.update()`](https://googleapis.github.io/python-genai/genai.html#genai.caches.Caches.update)
- [genai.chats module](https://googleapis.github.io/python-genai/genai.html#module-genai.chats)
  - [`AsyncChat`](https://googleapis.github.io/python-genai/genai.html#genai.chats.AsyncChat)
    - [`AsyncChat.send_message()`](https://googleapis.github.io/python-genai/genai.html#genai.chats.AsyncChat.send_message)
    - [`AsyncChat.send_message_stream()`](https://googleapis.github.io/python-genai/genai.html#genai.chats.AsyncChat.send_message_stream)
  - [`AsyncChats`](https://googleapis.github.io/python-genai/genai.html#genai.chats.AsyncChats)
    - [`AsyncChats.create()`](https://googleapis.github.io/python-genai/genai.html#genai.chats.AsyncChats.create)
  - [`Chat`](https://googleapis.github.io/python-genai/genai.html#genai.chats.Chat)
    - [`Chat.send_message()`](https://googleapis.github.io/python-genai/genai.html#genai.chats.Chat.send_message)
    - [`Chat.send_message_stream()`](https://googleapis.github.io/python-genai/genai.html#genai.chats.Chat.send_message_stream)
  - [`Chats`](https://googleapis.github.io/python-genai/genai.html#genai.chats.Chats)
    - [`Chats.create()`](https://googleapis.github.io/python-genai/genai.html#genai.chats.Chats.create)
- [genai.files module](https://googleapis.github.io/python-genai/genai.html#module-genai.files)
  - [`AsyncFiles`](https://googleapis.github.io/python-genai/genai.html#genai.files.AsyncFiles)
    - [`AsyncFiles.delete()`](https://googleapis.github.io/python-genai/genai.html#genai.files.AsyncFiles.delete)
    - [`AsyncFiles.download()`](https://googleapis.github.io/python-genai/genai.html#genai.files.AsyncFiles.download)
    - [`AsyncFiles.get()`](https://googleapis.github.io/python-genai/genai.html#genai.files.AsyncFiles.get)
    - [`AsyncFiles.list()`](https://googleapis.github.io/python-genai/genai.html#genai.files.AsyncFiles.list)
    - [`AsyncFiles.upload()`](https://googleapis.github.io/python-genai/genai.html#genai.files.AsyncFiles.upload)
  - [`Files`](https://googleapis.github.io/python-genai/genai.html#genai.files.Files)
    - [`Files.delete()`](https://googleapis.github.io/python-genai/genai.html#genai.files.Files.delete)
    - [`Files.download()`](https://googleapis.github.io/python-genai/genai.html#genai.files.Files.download)
    - [`Files.get()`](https://googleapis.github.io/python-genai/genai.html#genai.files.Files.get)
    - [`Files.list()`](https://googleapis.github.io/python-genai/genai.html#genai.files.Files.list)
    - [`Files.upload()`](https://googleapis.github.io/python-genai/genai.html#genai.files.Files.upload)
- [genai.live module](https://googleapis.github.io/python-genai/genai.html#module-genai.live)
  - [`AsyncLive`](https://googleapis.github.io/python-genai/genai.html#genai.live.AsyncLive)
    - [`AsyncLive.connect()`](https://googleapis.github.io/python-genai/genai.html#genai.live.AsyncLive.connect)
    - [`AsyncLive.music`](https://googleapis.github.io/python-genai/genai.html#genai.live.AsyncLive.music)
  - [`AsyncSession`](https://googleapis.github.io/python-genai/genai.html#genai.live.AsyncSession)
    - [`AsyncSession.close()`](https://googleapis.github.io/python-genai/genai.html#genai.live.AsyncSession.close)
    - [`AsyncSession.receive()`](https://googleapis.github.io/python-genai/genai.html#genai.live.AsyncSession.receive)
    - [`AsyncSession.send()`](https://googleapis.github.io/python-genai/genai.html#genai.live.AsyncSession.send)
    - [`AsyncSession.send_client_content()`](https://googleapis.github.io/python-genai/genai.html#genai.live.AsyncSession.send_client_content)
    - [`AsyncSession.send_realtime_input()`](https://googleapis.github.io/python-genai/genai.html#genai.live.AsyncSession.send_realtime_input)
    - [`AsyncSession.send_tool_response()`](https://googleapis.github.io/python-genai/genai.html#genai.live.AsyncSession.send_tool_response)
    - [`AsyncSession.start_stream()`](https://googleapis.github.io/python-genai/genai.html#genai.live.AsyncSession.start_stream)
- [genai.models module](https://googleapis.github.io/python-genai/genai.html#module-genai.models)
  - [`AsyncModels`](https://googleapis.github.io/python-genai/genai.html#genai.models.AsyncModels)
    - [`AsyncModels.compute_tokens()`](https://googleapis.github.io/python-genai/genai.html#genai.models.AsyncModels.compute_tokens)
    - [`AsyncModels.count_tokens()`](https://googleapis.github.io/python-genai/genai.html#genai.models.AsyncModels.count_tokens)
    - [`AsyncModels.delete()`](https://googleapis.github.io/python-genai/genai.html#genai.models.AsyncModels.delete)
    - [`AsyncModels.edit_image()`](https://googleapis.github.io/python-genai/genai.html#genai.models.AsyncModels.edit_image)
    - [`AsyncModels.embed_content()`](https://googleapis.github.io/python-genai/genai.html#genai.models.AsyncModels.embed_content)
    - [`AsyncModels.generate_content()`](https://googleapis.github.io/python-genai/genai.html#genai.models.AsyncModels.generate_content)
    - [`AsyncModels.generate_content_stream()`](https://googleapis.github.io/python-genai/genai.html#genai.models.AsyncModels.generate_content_stream)
    - [`AsyncModels.generate_images()`](https://googleapis.github.io/python-genai/genai.html#genai.models.AsyncModels.generate_images)
    - [`AsyncModels.generate_videos()`](https://googleapis.github.io/python-genai/genai.html#genai.models.AsyncModels.generate_videos)
    - [`AsyncModels.get()`](https://googleapis.github.io/python-genai/genai.html#genai.models.AsyncModels.get)
    - [`AsyncModels.list()`](https://googleapis.github.io/python-genai/genai.html#genai.models.AsyncModels.list)
    - [`AsyncModels.update()`](https://googleapis.github.io/python-genai/genai.html#genai.models.AsyncModels.update)
    - [`AsyncModels.upscale_image()`](https://googleapis.github.io/python-genai/genai.html#genai.models.AsyncModels.upscale_image)
  - [`Models`](https://googleapis.github.io/python-genai/genai.html#genai.models.Models)
    - [`Models.compute_tokens()`](https://googleapis.github.io/python-genai/genai.html#genai.models.Models.compute_tokens)
    - [`Models.count_tokens()`](https://googleapis.github.io/python-genai/genai.html#genai.models.Models.count_tokens)
    - [`Models.delete()`](https://googleapis.github.io/python-genai/genai.html#genai.models.Models.delete)
    - [`Models.edit_image()`](https://googleapis.github.io/python-genai/genai.html#genai.models.Models.edit_image)
    - [`Models.embed_content()`](https://googleapis.github.io/python-genai/genai.html#genai.models.Models.embed_content)
    - [`Models.generate_content()`](https://googleapis.github.io/python-genai/genai.html#genai.models.Models.generate_content)
    - [`Models.generate_content_stream()`](https://googleapis.github.io/python-genai/genai.html#genai.models.Models.generate_content_stream)
    - [`Models.generate_images()`](https://googleapis.github.io/python-genai/genai.html#genai.models.Models.generate_images)
    - [`Models.generate_videos()`](https://googleapis.github.io/python-genai/genai.html#genai.models.Models.generate_videos)
    - [`Models.get()`](https://googleapis.github.io/python-genai/genai.html#genai.models.Models.get)
    - [`Models.list()`](https://googleapis.github.io/python-genai/genai.html#genai.models.Models.list)
    - [`Models.update()`](https://googleapis.github.io/python-genai/genai.html#genai.models.Models.update)
    - [`Models.upscale_image()`](https://googleapis.github.io/python-genai/genai.html#genai.models.Models.upscale_image)
- [genai.tokens module](https://googleapis.github.io/python-genai/genai.html#module-genai.tokens)
  - [`AsyncTokens`](https://googleapis.github.io/python-genai/genai.html#genai.tokens.AsyncTokens)
    - [`AsyncTokens.create()`](https://googleapis.github.io/python-genai/genai.html#genai.tokens.AsyncTokens.create)
  - [`Tokens`](https://googleapis.github.io/python-genai/genai.html#genai.tokens.Tokens)
    - [`Tokens.create()`](https://googleapis.github.io/python-genai/genai.html#genai.tokens.Tokens.create)
- [genai.tunings module](https://googleapis.github.io/python-genai/genai.html#module-genai.tunings)
  - [`AsyncTunings`](https://googleapis.github.io/python-genai/genai.html#genai.tunings.AsyncTunings)
    - [`AsyncTunings.get()`](https://googleapis.github.io/python-genai/genai.html#genai.tunings.AsyncTunings.get)
    - [`AsyncTunings.list()`](https://googleapis.github.io/python-genai/genai.html#genai.tunings.AsyncTunings.list)
    - [`AsyncTunings.tune()`](https://googleapis.github.io/python-genai/genai.html#genai.tunings.AsyncTunings.tune)
  - [`Tunings`](https://googleapis.github.io/python-genai/genai.html#genai.tunings.Tunings)
    - [`Tunings.get()`](https://googleapis.github.io/python-genai/genai.html#genai.tunings.Tunings.get)
    - [`Tunings.list()`](https://googleapis.github.io/python-genai/genai.html#genai.tunings.Tunings.list)
    - [`Tunings.tune()`](https://googleapis.github.io/python-genai/genai.html#genai.tunings.Tunings.tune)
- [genai.types module](https://googleapis.github.io/python-genai/genai.html#module-genai.types)
  - [`ActivityEnd`](https://googleapis.github.io/python-genai/genai.html#genai.types.ActivityEnd)
  - [`ActivityEndDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.ActivityEndDict)
  - [`ActivityHandling`](https://googleapis.github.io/python-genai/genai.html#genai.types.ActivityHandling)
    - [`ActivityHandling.ACTIVITY_HANDLING_UNSPECIFIED`](https://googleapis.github.io/python-genai/genai.html#genai.types.ActivityHandling.ACTIVITY_HANDLING_UNSPECIFIED)
    - [`ActivityHandling.NO_INTERRUPTION`](https://googleapis.github.io/python-genai/genai.html#genai.types.ActivityHandling.NO_INTERRUPTION)
    - [`ActivityHandling.START_OF_ACTIVITY_INTERRUPTS`](https://googleapis.github.io/python-genai/genai.html#genai.types.ActivityHandling.START_OF_ACTIVITY_INTERRUPTS)
  - [`ActivityStart`](https://googleapis.github.io/python-genai/genai.html#genai.types.ActivityStart)
  - [`ActivityStartDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.ActivityStartDict)
  - [`AdapterSize`](https://googleapis.github.io/python-genai/genai.html#genai.types.AdapterSize)
    - [`AdapterSize.ADAPTER_SIZE_EIGHT`](https://googleapis.github.io/python-genai/genai.html#genai.types.AdapterSize.ADAPTER_SIZE_EIGHT)
    - [`AdapterSize.ADAPTER_SIZE_FOUR`](https://googleapis.github.io/python-genai/genai.html#genai.types.AdapterSize.ADAPTER_SIZE_FOUR)
    - [`AdapterSize.ADAPTER_SIZE_ONE`](https://googleapis.github.io/python-genai/genai.html#genai.types.AdapterSize.ADAPTER_SIZE_ONE)
    - [`AdapterSize.ADAPTER_SIZE_SIXTEEN`](https://googleapis.github.io/python-genai/genai.html#genai.types.AdapterSize.ADAPTER_SIZE_SIXTEEN)
    - [`AdapterSize.ADAPTER_SIZE_THIRTY_TWO`](https://googleapis.github.io/python-genai/genai.html#genai.types.AdapterSize.ADAPTER_SIZE_THIRTY_TWO)
    - [`AdapterSize.ADAPTER_SIZE_TWO`](https://googleapis.github.io/python-genai/genai.html#genai.types.AdapterSize.ADAPTER_SIZE_TWO)
    - [`AdapterSize.ADAPTER_SIZE_UNSPECIFIED`](https://googleapis.github.io/python-genai/genai.html#genai.types.AdapterSize.ADAPTER_SIZE_UNSPECIFIED)
  - [`ApiKeyConfig`](https://googleapis.github.io/python-genai/genai.html#genai.types.ApiKeyConfig)
    - [`ApiKeyConfig.api_key_string`](https://googleapis.github.io/python-genai/genai.html#genai.types.ApiKeyConfig.api_key_string)
  - [`ApiKeyConfigDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.ApiKeyConfigDict)
    - [`ApiKeyConfigDict.api_key_string`](https://googleapis.github.io/python-genai/genai.html#genai.types.ApiKeyConfigDict.api_key_string)
  - [`AudioChunk`](https://googleapis.github.io/python-genai/genai.html#genai.types.AudioChunk)
    - [`AudioChunk.data`](https://googleapis.github.io/python-genai/genai.html#genai.types.AudioChunk.data)
    - [`AudioChunk.mime_type`](https://googleapis.github.io/python-genai/genai.html#genai.types.AudioChunk.mime_type)
    - [`AudioChunk.source_metadata`](https://googleapis.github.io/python-genai/genai.html#genai.types.AudioChunk.source_metadata)
  - [`AudioChunkDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.AudioChunkDict)
    - [`AudioChunkDict.data`](https://googleapis.github.io/python-genai/genai.html#genai.types.AudioChunkDict.data)
    - [`AudioChunkDict.mime_type`](https://googleapis.github.io/python-genai/genai.html#genai.types.AudioChunkDict.mime_type)
    - [`AudioChunkDict.source_metadata`](https://googleapis.github.io/python-genai/genai.html#genai.types.AudioChunkDict.source_metadata)
  - [`AudioTranscriptionConfig`](https://googleapis.github.io/python-genai/genai.html#genai.types.AudioTranscriptionConfig)
  - [`AudioTranscriptionConfigDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.AudioTranscriptionConfigDict)
  - [`AuthConfig`](https://googleapis.github.io/python-genai/genai.html#genai.types.AuthConfig)
    - [`AuthConfig.api_key_config`](https://googleapis.github.io/python-genai/genai.html#genai.types.AuthConfig.api_key_config)
    - [`AuthConfig.auth_type`](https://googleapis.github.io/python-genai/genai.html#genai.types.AuthConfig.auth_type)
    - [`AuthConfig.google_service_account_config`](https://googleapis.github.io/python-genai/genai.html#genai.types.AuthConfig.google_service_account_config)
    - [`AuthConfig.http_basic_auth_config`](https://googleapis.github.io/python-genai/genai.html#genai.types.AuthConfig.http_basic_auth_config)
    - [`AuthConfig.oauth_config`](https://googleapis.github.io/python-genai/genai.html#genai.types.AuthConfig.oauth_config)
    - [`AuthConfig.oidc_config`](https://googleapis.github.io/python-genai/genai.html#genai.types.AuthConfig.oidc_config)
  - [`AuthConfigDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.AuthConfigDict)
    - [`AuthConfigDict.api_key_config`](https://googleapis.github.io/python-genai/genai.html#genai.types.AuthConfigDict.api_key_config)
    - [`AuthConfigDict.auth_type`](https://googleapis.github.io/python-genai/genai.html#genai.types.AuthConfigDict.auth_type)
    - [`AuthConfigDict.google_service_account_config`](https://googleapis.github.io/python-genai/genai.html#genai.types.AuthConfigDict.google_service_account_config)
    - [`AuthConfigDict.http_basic_auth_config`](https://googleapis.github.io/python-genai/genai.html#genai.types.AuthConfigDict.http_basic_auth_config)
    - [`AuthConfigDict.oauth_config`](https://googleapis.github.io/python-genai/genai.html#genai.types.AuthConfigDict.oauth_config)
    - [`AuthConfigDict.oidc_config`](https://googleapis.github.io/python-genai/genai.html#genai.types.AuthConfigDict.oidc_config)
  - [`AuthConfigGoogleServiceAccountConfig`](https://googleapis.github.io/python-genai/genai.html#genai.types.AuthConfigGoogleServiceAccountConfig)
    - [`AuthConfigGoogleServiceAccountConfig.service_account`](https://googleapis.github.io/python-genai/genai.html#genai.types.AuthConfigGoogleServiceAccountConfig.service_account)
  - [`AuthConfigGoogleServiceAccountConfigDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.AuthConfigGoogleServiceAccountConfigDict)
    - [`AuthConfigGoogleServiceAccountConfigDict.service_account`](https://googleapis.github.io/python-genai/genai.html#genai.types.AuthConfigGoogleServiceAccountConfigDict.service_account)
  - [`AuthConfigHttpBasicAuthConfig`](https://googleapis.github.io/python-genai/genai.html#genai.types.AuthConfigHttpBasicAuthConfig)
    - [`AuthConfigHttpBasicAuthConfig.credential_secret`](https://googleapis.github.io/python-genai/genai.html#genai.types.AuthConfigHttpBasicAuthConfig.credential_secret)
  - [`AuthConfigHttpBasicAuthConfigDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.AuthConfigHttpBasicAuthConfigDict)
    - [`AuthConfigHttpBasicAuthConfigDict.credential_secret`](https://googleapis.github.io/python-genai/genai.html#genai.types.AuthConfigHttpBasicAuthConfigDict.credential_secret)
  - [`AuthConfigOauthConfig`](https://googleapis.github.io/python-genai/genai.html#genai.types.AuthConfigOauthConfig)
    - [`AuthConfigOauthConfig.access_token`](https://googleapis.github.io/python-genai/genai.html#genai.types.AuthConfigOauthConfig.access_token)
    - [`AuthConfigOauthConfig.service_account`](https://googleapis.github.io/python-genai/genai.html#genai.types.AuthConfigOauthConfig.service_account)
  - [`AuthConfigOauthConfigDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.AuthConfigOauthConfigDict)
    - [`AuthConfigOauthConfigDict.access_token`](https://googleapis.github.io/python-genai/genai.html#genai.types.AuthConfigOauthConfigDict.access_token)
    - [`AuthConfigOauthConfigDict.service_account`](https://googleapis.github.io/python-genai/genai.html#genai.types.AuthConfigOauthConfigDict.service_account)
  - [`AuthConfigOidcConfig`](https://googleapis.github.io/python-genai/genai.html#genai.types.AuthConfigOidcConfig)
    - [`AuthConfigOidcConfig.id_token`](https://googleapis.github.io/python-genai/genai.html#genai.types.AuthConfigOidcConfig.id_token)
    - [`AuthConfigOidcConfig.service_account`](https://googleapis.github.io/python-genai/genai.html#genai.types.AuthConfigOidcConfig.service_account)
  - [`AuthConfigOidcConfigDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.AuthConfigOidcConfigDict)
    - [`AuthConfigOidcConfigDict.id_token`](https://googleapis.github.io/python-genai/genai.html#genai.types.AuthConfigOidcConfigDict.id_token)
    - [`AuthConfigOidcConfigDict.service_account`](https://googleapis.github.io/python-genai/genai.html#genai.types.AuthConfigOidcConfigDict.service_account)
  - [`AuthToken`](https://googleapis.github.io/python-genai/genai.html#genai.types.AuthToken)
    - [`AuthToken.name`](https://googleapis.github.io/python-genai/genai.html#genai.types.AuthToken.name)
  - [`AuthTokenDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.AuthTokenDict)
    - [`AuthTokenDict.name`](https://googleapis.github.io/python-genai/genai.html#genai.types.AuthTokenDict.name)
  - [`AuthType`](https://googleapis.github.io/python-genai/genai.html#genai.types.AuthType)
    - [`AuthType.API_KEY_AUTH`](https://googleapis.github.io/python-genai/genai.html#genai.types.AuthType.API_KEY_AUTH)
    - [`AuthType.AUTH_TYPE_UNSPECIFIED`](https://googleapis.github.io/python-genai/genai.html#genai.types.AuthType.AUTH_TYPE_UNSPECIFIED)
    - [`AuthType.GOOGLE_SERVICE_ACCOUNT_AUTH`](https://googleapis.github.io/python-genai/genai.html#genai.types.AuthType.GOOGLE_SERVICE_ACCOUNT_AUTH)
    - [`AuthType.HTTP_BASIC_AUTH`](https://googleapis.github.io/python-genai/genai.html#genai.types.AuthType.HTTP_BASIC_AUTH)
    - [`AuthType.NO_AUTH`](https://googleapis.github.io/python-genai/genai.html#genai.types.AuthType.NO_AUTH)
    - [`AuthType.OAUTH`](https://googleapis.github.io/python-genai/genai.html#genai.types.AuthType.OAUTH)
    - [`AuthType.OIDC_AUTH`](https://googleapis.github.io/python-genai/genai.html#genai.types.AuthType.OIDC_AUTH)
  - [`AutomaticActivityDetection`](https://googleapis.github.io/python-genai/genai.html#genai.types.AutomaticActivityDetection)
    - [`AutomaticActivityDetection.disabled`](https://googleapis.github.io/python-genai/genai.html#genai.types.AutomaticActivityDetection.disabled)
    - [`AutomaticActivityDetection.end_of_speech_sensitivity`](https://googleapis.github.io/python-genai/genai.html#genai.types.AutomaticActivityDetection.end_of_speech_sensitivity)
    - [`AutomaticActivityDetection.prefix_padding_ms`](https://googleapis.github.io/python-genai/genai.html#genai.types.AutomaticActivityDetection.prefix_padding_ms)
    - [`AutomaticActivityDetection.silence_duration_ms`](https://googleapis.github.io/python-genai/genai.html#genai.types.AutomaticActivityDetection.silence_duration_ms)
    - [`AutomaticActivityDetection.start_of_speech_sensitivity`](https://googleapis.github.io/python-genai/genai.html#genai.types.AutomaticActivityDetection.start_of_speech_sensitivity)
  - [`AutomaticActivityDetectionDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.AutomaticActivityDetectionDict)
    - [`AutomaticActivityDetectionDict.disabled`](https://googleapis.github.io/python-genai/genai.html#genai.types.AutomaticActivityDetectionDict.disabled)
    - [`AutomaticActivityDetectionDict.end_of_speech_sensitivity`](https://googleapis.github.io/python-genai/genai.html#genai.types.AutomaticActivityDetectionDict.end_of_speech_sensitivity)
    - [`AutomaticActivityDetectionDict.prefix_padding_ms`](https://googleapis.github.io/python-genai/genai.html#genai.types.AutomaticActivityDetectionDict.prefix_padding_ms)
    - [`AutomaticActivityDetectionDict.silence_duration_ms`](https://googleapis.github.io/python-genai/genai.html#genai.types.AutomaticActivityDetectionDict.silence_duration_ms)
    - [`AutomaticActivityDetectionDict.start_of_speech_sensitivity`](https://googleapis.github.io/python-genai/genai.html#genai.types.AutomaticActivityDetectionDict.start_of_speech_sensitivity)
  - [`AutomaticFunctionCallingConfig`](https://googleapis.github.io/python-genai/genai.html#genai.types.AutomaticFunctionCallingConfig)
    - [`AutomaticFunctionCallingConfig.disable`](https://googleapis.github.io/python-genai/genai.html#genai.types.AutomaticFunctionCallingConfig.disable)
    - [`AutomaticFunctionCallingConfig.ignore_call_history`](https://googleapis.github.io/python-genai/genai.html#genai.types.AutomaticFunctionCallingConfig.ignore_call_history)
    - [`AutomaticFunctionCallingConfig.maximum_remote_calls`](https://googleapis.github.io/python-genai/genai.html#genai.types.AutomaticFunctionCallingConfig.maximum_remote_calls)
  - [`AutomaticFunctionCallingConfigDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.AutomaticFunctionCallingConfigDict)
    - [`AutomaticFunctionCallingConfigDict.disable`](https://googleapis.github.io/python-genai/genai.html#genai.types.AutomaticFunctionCallingConfigDict.disable)
    - [`AutomaticFunctionCallingConfigDict.ignore_call_history`](https://googleapis.github.io/python-genai/genai.html#genai.types.AutomaticFunctionCallingConfigDict.ignore_call_history)
    - [`AutomaticFunctionCallingConfigDict.maximum_remote_calls`](https://googleapis.github.io/python-genai/genai.html#genai.types.AutomaticFunctionCallingConfigDict.maximum_remote_calls)
  - [`BatchJob`](https://googleapis.github.io/python-genai/genai.html#genai.types.BatchJob)
    - [`BatchJob.create_time`](https://googleapis.github.io/python-genai/genai.html#genai.types.BatchJob.create_time)
    - [`BatchJob.dest`](https://googleapis.github.io/python-genai/genai.html#genai.types.BatchJob.dest)
    - [`BatchJob.display_name`](https://googleapis.github.io/python-genai/genai.html#genai.types.BatchJob.display_name)
    - [`BatchJob.end_time`](https://googleapis.github.io/python-genai/genai.html#genai.types.BatchJob.end_time)
    - [`BatchJob.error`](https://googleapis.github.io/python-genai/genai.html#genai.types.BatchJob.error)
    - [`BatchJob.model`](https://googleapis.github.io/python-genai/genai.html#genai.types.BatchJob.model)
    - [`BatchJob.name`](https://googleapis.github.io/python-genai/genai.html#genai.types.BatchJob.name)
    - [`BatchJob.src`](https://googleapis.github.io/python-genai/genai.html#genai.types.BatchJob.src)
    - [`BatchJob.start_time`](https://googleapis.github.io/python-genai/genai.html#genai.types.BatchJob.start_time)
    - [`BatchJob.state`](https://googleapis.github.io/python-genai/genai.html#genai.types.BatchJob.state)
    - [`BatchJob.update_time`](https://googleapis.github.io/python-genai/genai.html#genai.types.BatchJob.update_time)
  - [`BatchJobDestination`](https://googleapis.github.io/python-genai/genai.html#genai.types.BatchJobDestination)
    - [`BatchJobDestination.bigquery_uri`](https://googleapis.github.io/python-genai/genai.html#genai.types.BatchJobDestination.bigquery_uri)
    - [`BatchJobDestination.format`](https://googleapis.github.io/python-genai/genai.html#genai.types.BatchJobDestination.format)
    - [`BatchJobDestination.gcs_uri`](https://googleapis.github.io/python-genai/genai.html#genai.types.BatchJobDestination.gcs_uri)
  - [`BatchJobDestinationDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.BatchJobDestinationDict)
    - [`BatchJobDestinationDict.bigquery_uri`](https://googleapis.github.io/python-genai/genai.html#genai.types.BatchJobDestinationDict.bigquery_uri)
    - [`BatchJobDestinationDict.format`](https://googleapis.github.io/python-genai/genai.html#genai.types.BatchJobDestinationDict.format)
    - [`BatchJobDestinationDict.gcs_uri`](https://googleapis.github.io/python-genai/genai.html#genai.types.BatchJobDestinationDict.gcs_uri)
  - [`BatchJobDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.BatchJobDict)
    - [`BatchJobDict.create_time`](https://googleapis.github.io/python-genai/genai.html#genai.types.BatchJobDict.create_time)
    - [`BatchJobDict.dest`](https://googleapis.github.io/python-genai/genai.html#genai.types.BatchJobDict.dest)
    - [`BatchJobDict.display_name`](https://googleapis.github.io/python-genai/genai.html#genai.types.BatchJobDict.display_name)
    - [`BatchJobDict.end_time`](https://googleapis.github.io/python-genai/genai.html#genai.types.BatchJobDict.end_time)
    - [`BatchJobDict.error`](https://googleapis.github.io/python-genai/genai.html#genai.types.BatchJobDict.error)
    - [`BatchJobDict.model`](https://googleapis.github.io/python-genai/genai.html#genai.types.BatchJobDict.model)
    - [`BatchJobDict.name`](https://googleapis.github.io/python-genai/genai.html#genai.types.BatchJobDict.name)
    - [`BatchJobDict.src`](https://googleapis.github.io/python-genai/genai.html#genai.types.BatchJobDict.src)
    - [`BatchJobDict.start_time`](https://googleapis.github.io/python-genai/genai.html#genai.types.BatchJobDict.start_time)
    - [`BatchJobDict.state`](https://googleapis.github.io/python-genai/genai.html#genai.types.BatchJobDict.state)
    - [`BatchJobDict.update_time`](https://googleapis.github.io/python-genai/genai.html#genai.types.BatchJobDict.update_time)
  - [`BatchJobSource`](https://googleapis.github.io/python-genai/genai.html#genai.types.BatchJobSource)
    - [`BatchJobSource.bigquery_uri`](https://googleapis.github.io/python-genai/genai.html#genai.types.BatchJobSource.bigquery_uri)
    - [`BatchJobSource.format`](https://googleapis.github.io/python-genai/genai.html#genai.types.BatchJobSource.format)
    - [`BatchJobSource.gcs_uri`](https://googleapis.github.io/python-genai/genai.html#genai.types.BatchJobSource.gcs_uri)
  - [`BatchJobSourceDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.BatchJobSourceDict)
    - [`BatchJobSourceDict.bigquery_uri`](https://googleapis.github.io/python-genai/genai.html#genai.types.BatchJobSourceDict.bigquery_uri)
    - [`BatchJobSourceDict.format`](https://googleapis.github.io/python-genai/genai.html#genai.types.BatchJobSourceDict.format)
    - [`BatchJobSourceDict.gcs_uri`](https://googleapis.github.io/python-genai/genai.html#genai.types.BatchJobSourceDict.gcs_uri)
  - [`Behavior`](https://googleapis.github.io/python-genai/genai.html#genai.types.Behavior)
    - [`Behavior.BLOCKING`](https://googleapis.github.io/python-genai/genai.html#genai.types.Behavior.BLOCKING)
    - [`Behavior.NON_BLOCKING`](https://googleapis.github.io/python-genai/genai.html#genai.types.Behavior.NON_BLOCKING)
    - [`Behavior.UNSPECIFIED`](https://googleapis.github.io/python-genai/genai.html#genai.types.Behavior.UNSPECIFIED)
  - [`Blob`](https://googleapis.github.io/python-genai/genai.html#genai.types.Blob)
    - [`Blob.data`](https://googleapis.github.io/python-genai/genai.html#genai.types.Blob.data)
    - [`Blob.display_name`](https://googleapis.github.io/python-genai/genai.html#genai.types.Blob.display_name)
    - [`Blob.mime_type`](https://googleapis.github.io/python-genai/genai.html#genai.types.Blob.mime_type)
  - [`BlobDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.BlobDict)
    - [`BlobDict.data`](https://googleapis.github.io/python-genai/genai.html#genai.types.BlobDict.data)
    - [`BlobDict.display_name`](https://googleapis.github.io/python-genai/genai.html#genai.types.BlobDict.display_name)
    - [`BlobDict.mime_type`](https://googleapis.github.io/python-genai/genai.html#genai.types.BlobDict.mime_type)
  - [`BlockedReason`](https://googleapis.github.io/python-genai/genai.html#genai.types.BlockedReason)
    - [`BlockedReason.BLOCKED_REASON_UNSPECIFIED`](https://googleapis.github.io/python-genai/genai.html#genai.types.BlockedReason.BLOCKED_REASON_UNSPECIFIED)
    - [`BlockedReason.BLOCKLIST`](https://googleapis.github.io/python-genai/genai.html#genai.types.BlockedReason.BLOCKLIST)
    - [`BlockedReason.OTHER`](https://googleapis.github.io/python-genai/genai.html#genai.types.BlockedReason.OTHER)
    - [`BlockedReason.PROHIBITED_CONTENT`](https://googleapis.github.io/python-genai/genai.html#genai.types.BlockedReason.PROHIBITED_CONTENT)
    - [`BlockedReason.SAFETY`](https://googleapis.github.io/python-genai/genai.html#genai.types.BlockedReason.SAFETY)
  - [`CachedContent`](https://googleapis.github.io/python-genai/genai.html#genai.types.CachedContent)
    - [`CachedContent.create_time`](https://googleapis.github.io/python-genai/genai.html#genai.types.CachedContent.create_time)
    - [`CachedContent.display_name`](https://googleapis.github.io/python-genai/genai.html#genai.types.CachedContent.display_name)
    - [`CachedContent.expire_time`](https://googleapis.github.io/python-genai/genai.html#genai.types.CachedContent.expire_time)
    - [`CachedContent.model`](https://googleapis.github.io/python-genai/genai.html#genai.types.CachedContent.model)
    - [`CachedContent.name`](https://googleapis.github.io/python-genai/genai.html#genai.types.CachedContent.name)
    - [`CachedContent.update_time`](https://googleapis.github.io/python-genai/genai.html#genai.types.CachedContent.update_time)
    - [`CachedContent.usage_metadata`](https://googleapis.github.io/python-genai/genai.html#genai.types.CachedContent.usage_metadata)
  - [`CachedContentDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.CachedContentDict)
    - [`CachedContentDict.create_time`](https://googleapis.github.io/python-genai/genai.html#genai.types.CachedContentDict.create_time)
    - [`CachedContentDict.display_name`](https://googleapis.github.io/python-genai/genai.html#genai.types.CachedContentDict.display_name)
    - [`CachedContentDict.expire_time`](https://googleapis.github.io/python-genai/genai.html#genai.types.CachedContentDict.expire_time)
    - [`CachedContentDict.model`](https://googleapis.github.io/python-genai/genai.html#genai.types.CachedContentDict.model)
    - [`CachedContentDict.name`](https://googleapis.github.io/python-genai/genai.html#genai.types.CachedContentDict.name)
    - [`CachedContentDict.update_time`](https://googleapis.github.io/python-genai/genai.html#genai.types.CachedContentDict.update_time)
    - [`CachedContentDict.usage_metadata`](https://googleapis.github.io/python-genai/genai.html#genai.types.CachedContentDict.usage_metadata)
  - [`CachedContentUsageMetadata`](https://googleapis.github.io/python-genai/genai.html#genai.types.CachedContentUsageMetadata)
    - [`CachedContentUsageMetadata.audio_duration_seconds`](https://googleapis.github.io/python-genai/genai.html#genai.types.CachedContentUsageMetadata.audio_duration_seconds)
    - [`CachedContentUsageMetadata.image_count`](https://googleapis.github.io/python-genai/genai.html#genai.types.CachedContentUsageMetadata.image_count)
    - [`CachedContentUsageMetadata.text_count`](https://googleapis.github.io/python-genai/genai.html#genai.types.CachedContentUsageMetadata.text_count)
    - [`CachedContentUsageMetadata.total_token_count`](https://googleapis.github.io/python-genai/genai.html#genai.types.CachedContentUsageMetadata.total_token_count)
    - [`CachedContentUsageMetadata.video_duration_seconds`](https://googleapis.github.io/python-genai/genai.html#genai.types.CachedContentUsageMetadata.video_duration_seconds)
  - [`CachedContentUsageMetadataDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.CachedContentUsageMetadataDict)
    - [`CachedContentUsageMetadataDict.audio_duration_seconds`](https://googleapis.github.io/python-genai/genai.html#genai.types.CachedContentUsageMetadataDict.audio_duration_seconds)
    - [`CachedContentUsageMetadataDict.image_count`](https://googleapis.github.io/python-genai/genai.html#genai.types.CachedContentUsageMetadataDict.image_count)
    - [`CachedContentUsageMetadataDict.text_count`](https://googleapis.github.io/python-genai/genai.html#genai.types.CachedContentUsageMetadataDict.text_count)
    - [`CachedContentUsageMetadataDict.total_token_count`](https://googleapis.github.io/python-genai/genai.html#genai.types.CachedContentUsageMetadataDict.total_token_count)
    - [`CachedContentUsageMetadataDict.video_duration_seconds`](https://googleapis.github.io/python-genai/genai.html#genai.types.CachedContentUsageMetadataDict.video_duration_seconds)
  - [`CancelBatchJobConfig`](https://googleapis.github.io/python-genai/genai.html#genai.types.CancelBatchJobConfig)
    - [`CancelBatchJobConfig.http_options`](https://googleapis.github.io/python-genai/genai.html#genai.types.CancelBatchJobConfig.http_options)
  - [`CancelBatchJobConfigDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.CancelBatchJobConfigDict)
    - [`CancelBatchJobConfigDict.http_options`](https://googleapis.github.io/python-genai/genai.html#genai.types.CancelBatchJobConfigDict.http_options)
  - [`Candidate`](https://googleapis.github.io/python-genai/genai.html#genai.types.Candidate)
    - [`Candidate.avg_logprobs`](https://googleapis.github.io/python-genai/genai.html#genai.types.Candidate.avg_logprobs)
    - [`Candidate.citation_metadata`](https://googleapis.github.io/python-genai/genai.html#genai.types.Candidate.citation_metadata)
    - [`Candidate.content`](https://googleapis.github.io/python-genai/genai.html#genai.types.Candidate.content)
    - [`Candidate.finish_message`](https://googleapis.github.io/python-genai/genai.html#genai.types.Candidate.finish_message)
    - [`Candidate.finish_reason`](https://googleapis.github.io/python-genai/genai.html#genai.types.Candidate.finish_reason)
    - [`Candidate.grounding_metadata`](https://googleapis.github.io/python-genai/genai.html#genai.types.Candidate.grounding_metadata)
    - [`Candidate.index`](https://googleapis.github.io/python-genai/genai.html#genai.types.Candidate.index)
    - [`Candidate.logprobs_result`](https://googleapis.github.io/python-genai/genai.html#genai.types.Candidate.logprobs_result)
    - [`Candidate.safety_ratings`](https://googleapis.github.io/python-genai/genai.html#genai.types.Candidate.safety_ratings)
    - [`Candidate.token_count`](https://googleapis.github.io/python-genai/genai.html#genai.types.Candidate.token_count)
    - [`Candidate.url_context_metadata`](https://googleapis.github.io/python-genai/genai.html#genai.types.Candidate.url_context_metadata)
  - [`CandidateDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.CandidateDict)
    - [`CandidateDict.avg_logprobs`](https://googleapis.github.io/python-genai/genai.html#genai.types.CandidateDict.avg_logprobs)
    - [`CandidateDict.citation_metadata`](https://googleapis.github.io/python-genai/genai.html#genai.types.CandidateDict.citation_metadata)
    - [`CandidateDict.content`](https://googleapis.github.io/python-genai/genai.html#genai.types.CandidateDict.content)
    - [`CandidateDict.finish_message`](https://googleapis.github.io/python-genai/genai.html#genai.types.CandidateDict.finish_message)
    - [`CandidateDict.finish_reason`](https://googleapis.github.io/python-genai/genai.html#genai.types.CandidateDict.finish_reason)
    - [`CandidateDict.grounding_metadata`](https://googleapis.github.io/python-genai/genai.html#genai.types.CandidateDict.grounding_metadata)
    - [`CandidateDict.index`](https://googleapis.github.io/python-genai/genai.html#genai.types.CandidateDict.index)
    - [`CandidateDict.logprobs_result`](https://googleapis.github.io/python-genai/genai.html#genai.types.CandidateDict.logprobs_result)
    - [`CandidateDict.safety_ratings`](https://googleapis.github.io/python-genai/genai.html#genai.types.CandidateDict.safety_ratings)
    - [`CandidateDict.token_count`](https://googleapis.github.io/python-genai/genai.html#genai.types.CandidateDict.token_count)
    - [`CandidateDict.url_context_metadata`](https://googleapis.github.io/python-genai/genai.html#genai.types.CandidateDict.url_context_metadata)
  - [`Checkpoint`](https://googleapis.github.io/python-genai/genai.html#genai.types.Checkpoint)
    - [`Checkpoint.checkpoint_id`](https://googleapis.github.io/python-genai/genai.html#genai.types.Checkpoint.checkpoint_id)
    - [`Checkpoint.epoch`](https://googleapis.github.io/python-genai/genai.html#genai.types.Checkpoint.epoch)
    - [`Checkpoint.step`](https://googleapis.github.io/python-genai/genai.html#genai.types.Checkpoint.step)
  - [`CheckpointDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.CheckpointDict)
    - [`CheckpointDict.checkpoint_id`](https://googleapis.github.io/python-genai/genai.html#genai.types.CheckpointDict.checkpoint_id)
    - [`CheckpointDict.epoch`](https://googleapis.github.io/python-genai/genai.html#genai.types.CheckpointDict.epoch)
    - [`CheckpointDict.step`](https://googleapis.github.io/python-genai/genai.html#genai.types.CheckpointDict.step)
  - [`Citation`](https://googleapis.github.io/python-genai/genai.html#genai.types.Citation)
    - [`Citation.end_index`](https://googleapis.github.io/python-genai/genai.html#genai.types.Citation.end_index)
    - [`Citation.license`](https://googleapis.github.io/python-genai/genai.html#genai.types.Citation.license)
    - [`Citation.publication_date`](https://googleapis.github.io/python-genai/genai.html#genai.types.Citation.publication_date)
    - [`Citation.start_index`](https://googleapis.github.io/python-genai/genai.html#genai.types.Citation.start_index)
    - [`Citation.title`](https://googleapis.github.io/python-genai/genai.html#genai.types.Citation.title)
    - [`Citation.uri`](https://googleapis.github.io/python-genai/genai.html#genai.types.Citation.uri)
  - [`CitationDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.CitationDict)
    - [`CitationDict.end_index`](https://googleapis.github.io/python-genai/genai.html#genai.types.CitationDict.end_index)
    - [`CitationDict.license`](https://googleapis.github.io/python-genai/genai.html#genai.types.CitationDict.license)
    - [`CitationDict.publication_date`](https://googleapis.github.io/python-genai/genai.html#genai.types.CitationDict.publication_date)
    - [`CitationDict.start_index`](https://googleapis.github.io/python-genai/genai.html#genai.types.CitationDict.start_index)
    - [`CitationDict.title`](https://googleapis.github.io/python-genai/genai.html#genai.types.CitationDict.title)
    - [`CitationDict.uri`](https://googleapis.github.io/python-genai/genai.html#genai.types.CitationDict.uri)
  - [`CitationMetadata`](https://googleapis.github.io/python-genai/genai.html#genai.types.CitationMetadata)
    - [`CitationMetadata.citations`](https://googleapis.github.io/python-genai/genai.html#genai.types.CitationMetadata.citations)
  - [`CitationMetadataDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.CitationMetadataDict)
    - [`CitationMetadataDict.citations`](https://googleapis.github.io/python-genai/genai.html#genai.types.CitationMetadataDict.citations)
  - [`CodeExecutionResult`](https://googleapis.github.io/python-genai/genai.html#genai.types.CodeExecutionResult)
    - [`CodeExecutionResult.outcome`](https://googleapis.github.io/python-genai/genai.html#genai.types.CodeExecutionResult.outcome)
    - [`CodeExecutionResult.output`](https://googleapis.github.io/python-genai/genai.html#genai.types.CodeExecutionResult.output)
  - [`CodeExecutionResultDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.CodeExecutionResultDict)
    - [`CodeExecutionResultDict.outcome`](https://googleapis.github.io/python-genai/genai.html#genai.types.CodeExecutionResultDict.outcome)
    - [`CodeExecutionResultDict.output`](https://googleapis.github.io/python-genai/genai.html#genai.types.CodeExecutionResultDict.output)
  - [`ComputeTokensConfig`](https://googleapis.github.io/python-genai/genai.html#genai.types.ComputeTokensConfig)
    - [`ComputeTokensConfig.http_options`](https://googleapis.github.io/python-genai/genai.html#genai.types.ComputeTokensConfig.http_options)
  - [`ComputeTokensConfigDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.ComputeTokensConfigDict)
    - [`ComputeTokensConfigDict.http_options`](https://googleapis.github.io/python-genai/genai.html#genai.types.ComputeTokensConfigDict.http_options)
  - [`ComputeTokensResponse`](https://googleapis.github.io/python-genai/genai.html#genai.types.ComputeTokensResponse)
    - [`ComputeTokensResponse.tokens_info`](https://googleapis.github.io/python-genai/genai.html#genai.types.ComputeTokensResponse.tokens_info)
  - [`ComputeTokensResponseDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.ComputeTokensResponseDict)
    - [`ComputeTokensResponseDict.tokens_info`](https://googleapis.github.io/python-genai/genai.html#genai.types.ComputeTokensResponseDict.tokens_info)
  - [`Content`](https://googleapis.github.io/python-genai/genai.html#genai.types.Content)
    - [`Content.parts`](https://googleapis.github.io/python-genai/genai.html#genai.types.Content.parts)
    - [`Content.role`](https://googleapis.github.io/python-genai/genai.html#genai.types.Content.role)
  - [`ContentDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.ContentDict)
    - [`ContentDict.parts`](https://googleapis.github.io/python-genai/genai.html#genai.types.ContentDict.parts)
    - [`ContentDict.role`](https://googleapis.github.io/python-genai/genai.html#genai.types.ContentDict.role)
  - [`ContentEmbedding`](https://googleapis.github.io/python-genai/genai.html#genai.types.ContentEmbedding)
    - [`ContentEmbedding.statistics`](https://googleapis.github.io/python-genai/genai.html#genai.types.ContentEmbedding.statistics)
    - [`ContentEmbedding.values`](https://googleapis.github.io/python-genai/genai.html#genai.types.ContentEmbedding.values)
  - [`ContentEmbeddingDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.ContentEmbeddingDict)
    - [`ContentEmbeddingDict.statistics`](https://googleapis.github.io/python-genai/genai.html#genai.types.ContentEmbeddingDict.statistics)
  - [`ContentEmbeddingStatistics`](https://googleapis.github.io/python-genai/genai.html#genai.types.ContentEmbeddingStatistics)
    - [`ContentEmbeddingStatistics.token_count`](https://googleapis.github.io/python-genai/genai.html#genai.types.ContentEmbeddingStatistics.token_count)
    - [`ContentEmbeddingStatistics.truncated`](https://googleapis.github.io/python-genai/genai.html#genai.types.ContentEmbeddingStatistics.truncated)
  - [`ContentEmbeddingStatisticsDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.ContentEmbeddingStatisticsDict)
    - [`ContentEmbeddingStatisticsDict.token_count`](https://googleapis.github.io/python-genai/genai.html#genai.types.ContentEmbeddingStatisticsDict.token_count)
    - [`ContentEmbeddingStatisticsDict.truncated`](https://googleapis.github.io/python-genai/genai.html#genai.types.ContentEmbeddingStatisticsDict.truncated)
  - [`ContextWindowCompressionConfig`](https://googleapis.github.io/python-genai/genai.html#genai.types.ContextWindowCompressionConfig)
    - [`ContextWindowCompressionConfig.sliding_window`](https://googleapis.github.io/python-genai/genai.html#genai.types.ContextWindowCompressionConfig.sliding_window)
    - [`ContextWindowCompressionConfig.trigger_tokens`](https://googleapis.github.io/python-genai/genai.html#genai.types.ContextWindowCompressionConfig.trigger_tokens)
  - [`ContextWindowCompressionConfigDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.ContextWindowCompressionConfigDict)
    - [`ContextWindowCompressionConfigDict.sliding_window`](https://googleapis.github.io/python-genai/genai.html#genai.types.ContextWindowCompressionConfigDict.sliding_window)
    - [`ContextWindowCompressionConfigDict.trigger_tokens`](https://googleapis.github.io/python-genai/genai.html#genai.types.ContextWindowCompressionConfigDict.trigger_tokens)
  - [`ControlReferenceConfig`](https://googleapis.github.io/python-genai/genai.html#genai.types.ControlReferenceConfig)
    - [`ControlReferenceConfig.control_type`](https://googleapis.github.io/python-genai/genai.html#genai.types.ControlReferenceConfig.control_type)
    - [`ControlReferenceConfig.enable_control_image_computation`](https://googleapis.github.io/python-genai/genai.html#genai.types.ControlReferenceConfig.enable_control_image_computation)
  - [`ControlReferenceConfigDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.ControlReferenceConfigDict)
    - [`ControlReferenceConfigDict.control_type`](https://googleapis.github.io/python-genai/genai.html#genai.types.ControlReferenceConfigDict.control_type)
    - [`ControlReferenceConfigDict.enable_control_image_computation`](https://googleapis.github.io/python-genai/genai.html#genai.types.ControlReferenceConfigDict.enable_control_image_computation)
  - [`ControlReferenceImage`](https://googleapis.github.io/python-genai/genai.html#genai.types.ControlReferenceImage)
    - [`ControlReferenceImage.config`](https://googleapis.github.io/python-genai/genai.html#genai.types.ControlReferenceImage.config)
    - [`ControlReferenceImage.control_image_config`](https://googleapis.github.io/python-genai/genai.html#genai.types.ControlReferenceImage.control_image_config)
    - [`ControlReferenceImage.reference_id`](https://googleapis.github.io/python-genai/genai.html#genai.types.ControlReferenceImage.reference_id)
    - [`ControlReferenceImage.reference_image`](https://googleapis.github.io/python-genai/genai.html#genai.types.ControlReferenceImage.reference_image)
    - [`ControlReferenceImage.reference_type`](https://googleapis.github.io/python-genai/genai.html#genai.types.ControlReferenceImage.reference_type)
  - [`ControlReferenceImageDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.ControlReferenceImageDict)
    - [`ControlReferenceImageDict.config`](https://googleapis.github.io/python-genai/genai.html#genai.types.ControlReferenceImageDict.config)
    - [`ControlReferenceImageDict.reference_id`](https://googleapis.github.io/python-genai/genai.html#genai.types.ControlReferenceImageDict.reference_id)
    - [`ControlReferenceImageDict.reference_image`](https://googleapis.github.io/python-genai/genai.html#genai.types.ControlReferenceImageDict.reference_image)
    - [`ControlReferenceImageDict.reference_type`](https://googleapis.github.io/python-genai/genai.html#genai.types.ControlReferenceImageDict.reference_type)
  - [`ControlReferenceType`](https://googleapis.github.io/python-genai/genai.html#genai.types.ControlReferenceType)
    - [`ControlReferenceType.CONTROL_TYPE_CANNY`](https://googleapis.github.io/python-genai/genai.html#genai.types.ControlReferenceType.CONTROL_TYPE_CANNY)
    - [`ControlReferenceType.CONTROL_TYPE_DEFAULT`](https://googleapis.github.io/python-genai/genai.html#genai.types.ControlReferenceType.CONTROL_TYPE_DEFAULT)
    - [`ControlReferenceType.CONTROL_TYPE_FACE_MESH`](https://googleapis.github.io/python-genai/genai.html#genai.types.ControlReferenceType.CONTROL_TYPE_FACE_MESH)
    - [`ControlReferenceType.CONTROL_TYPE_SCRIBBLE`](https://googleapis.github.io/python-genai/genai.html#genai.types.ControlReferenceType.CONTROL_TYPE_SCRIBBLE)
  - [`CountTokensConfig`](https://googleapis.github.io/python-genai/genai.html#genai.types.CountTokensConfig)
    - [`CountTokensConfig.generation_config`](https://googleapis.github.io/python-genai/genai.html#genai.types.CountTokensConfig.generation_config)
    - [`CountTokensConfig.http_options`](https://googleapis.github.io/python-genai/genai.html#genai.types.CountTokensConfig.http_options)
    - [`CountTokensConfig.system_instruction`](https://googleapis.github.io/python-genai/genai.html#genai.types.CountTokensConfig.system_instruction)
    - [`CountTokensConfig.tools`](https://googleapis.github.io/python-genai/genai.html#genai.types.CountTokensConfig.tools)
  - [`CountTokensConfigDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.CountTokensConfigDict)
    - [`CountTokensConfigDict.generation_config`](https://googleapis.github.io/python-genai/genai.html#genai.types.CountTokensConfigDict.generation_config)
    - [`CountTokensConfigDict.http_options`](https://googleapis.github.io/python-genai/genai.html#genai.types.CountTokensConfigDict.http_options)
    - [`CountTokensConfigDict.system_instruction`](https://googleapis.github.io/python-genai/genai.html#genai.types.CountTokensConfigDict.system_instruction)
    - [`CountTokensConfigDict.tools`](https://googleapis.github.io/python-genai/genai.html#genai.types.CountTokensConfigDict.tools)
  - [`CountTokensResponse`](https://googleapis.github.io/python-genai/genai.html#genai.types.CountTokensResponse)
    - [`CountTokensResponse.cached_content_token_count`](https://googleapis.github.io/python-genai/genai.html#genai.types.CountTokensResponse.cached_content_token_count)
    - [`CountTokensResponse.total_tokens`](https://googleapis.github.io/python-genai/genai.html#genai.types.CountTokensResponse.total_tokens)
  - [`CountTokensResponseDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.CountTokensResponseDict)
    - [`CountTokensResponseDict.cached_content_token_count`](https://googleapis.github.io/python-genai/genai.html#genai.types.CountTokensResponseDict.cached_content_token_count)
    - [`CountTokensResponseDict.total_tokens`](https://googleapis.github.io/python-genai/genai.html#genai.types.CountTokensResponseDict.total_tokens)
  - [`CreateAuthTokenConfig`](https://googleapis.github.io/python-genai/genai.html#genai.types.CreateAuthTokenConfig)
    - [`CreateAuthTokenConfig.expire_time`](https://googleapis.github.io/python-genai/genai.html#genai.types.CreateAuthTokenConfig.expire_time)
    - [`CreateAuthTokenConfig.http_options`](https://googleapis.github.io/python-genai/genai.html#genai.types.CreateAuthTokenConfig.http_options)
    - [`CreateAuthTokenConfig.live_connect_constraints`](https://googleapis.github.io/python-genai/genai.html#genai.types.CreateAuthTokenConfig.live_connect_constraints)
    - [`CreateAuthTokenConfig.lock_additional_fields`](https://googleapis.github.io/python-genai/genai.html#genai.types.CreateAuthTokenConfig.lock_additional_fields)
    - [`CreateAuthTokenConfig.new_session_expire_time`](https://googleapis.github.io/python-genai/genai.html#genai.types.CreateAuthTokenConfig.new_session_expire_time)
    - [`CreateAuthTokenConfig.uses`](https://googleapis.github.io/python-genai/genai.html#genai.types.CreateAuthTokenConfig.uses)
  - [`CreateAuthTokenConfigDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.CreateAuthTokenConfigDict)
    - [`CreateAuthTokenConfigDict.expire_time`](https://googleapis.github.io/python-genai/genai.html#genai.types.CreateAuthTokenConfigDict.expire_time)
    - [`CreateAuthTokenConfigDict.http_options`](https://googleapis.github.io/python-genai/genai.html#genai.types.CreateAuthTokenConfigDict.http_options)
    - [`CreateAuthTokenConfigDict.live_connect_constraints`](https://googleapis.github.io/python-genai/genai.html#genai.types.CreateAuthTokenConfigDict.live_connect_constraints)
    - [`CreateAuthTokenConfigDict.lock_additional_fields`](https://googleapis.github.io/python-genai/genai.html#genai.types.CreateAuthTokenConfigDict.lock_additional_fields)
    - [`CreateAuthTokenConfigDict.new_session_expire_time`](https://googleapis.github.io/python-genai/genai.html#genai.types.CreateAuthTokenConfigDict.new_session_expire_time)
    - [`CreateAuthTokenConfigDict.uses`](https://googleapis.github.io/python-genai/genai.html#genai.types.CreateAuthTokenConfigDict.uses)
  - [`CreateAuthTokenParameters`](https://googleapis.github.io/python-genai/genai.html#genai.types.CreateAuthTokenParameters)
    - [`CreateAuthTokenParameters.config`](https://googleapis.github.io/python-genai/genai.html#genai.types.CreateAuthTokenParameters.config)
  - [`CreateAuthTokenParametersDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.CreateAuthTokenParametersDict)
    - [`CreateAuthTokenParametersDict.config`](https://googleapis.github.io/python-genai/genai.html#genai.types.CreateAuthTokenParametersDict.config)
  - [`CreateBatchJobConfig`](https://googleapis.github.io/python-genai/genai.html#genai.types.CreateBatchJobConfig)
    - [`CreateBatchJobConfig.dest`](https://googleapis.github.io/python-genai/genai.html#genai.types.CreateBatchJobConfig.dest)
    - [`CreateBatchJobConfig.display_name`](https://googleapis.github.io/python-genai/genai.html#genai.types.CreateBatchJobConfig.display_name)
    - [`CreateBatchJobConfig.http_options`](https://googleapis.github.io/python-genai/genai.html#genai.types.CreateBatchJobConfig.http_options)
  - [`CreateBatchJobConfigDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.CreateBatchJobConfigDict)
    - [`CreateBatchJobConfigDict.dest`](https://googleapis.github.io/python-genai/genai.html#genai.types.CreateBatchJobConfigDict.dest)
    - [`CreateBatchJobConfigDict.display_name`](https://googleapis.github.io/python-genai/genai.html#genai.types.CreateBatchJobConfigDict.display_name)
    - [`CreateBatchJobConfigDict.http_options`](https://googleapis.github.io/python-genai/genai.html#genai.types.CreateBatchJobConfigDict.http_options)
  - [`CreateCachedContentConfig`](https://googleapis.github.io/python-genai/genai.html#genai.types.CreateCachedContentConfig)
    - [`CreateCachedContentConfig.contents`](https://googleapis.github.io/python-genai/genai.html#genai.types.CreateCachedContentConfig.contents)
    - [`CreateCachedContentConfig.display_name`](https://googleapis.github.io/python-genai/genai.html#genai.types.CreateCachedContentConfig.display_name)
    - [`CreateCachedContentConfig.expire_time`](https://googleapis.github.io/python-genai/genai.html#genai.types.CreateCachedContentConfig.expire_time)
    - [`CreateCachedContentConfig.http_options`](https://googleapis.github.io/python-genai/genai.html#genai.types.CreateCachedContentConfig.http_options)
    - [`CreateCachedContentConfig.kms_key_name`](https://googleapis.github.io/python-genai/genai.html#genai.types.CreateCachedContentConfig.kms_key_name)
    - [`CreateCachedContentConfig.system_instruction`](https://googleapis.github.io/python-genai/genai.html#genai.types.CreateCachedContentConfig.system_instruction)
    - [`CreateCachedContentConfig.tool_config`](https://googleapis.github.io/python-genai/genai.html#genai.types.CreateCachedContentConfig.tool_config)
    - [`CreateCachedContentConfig.tools`](https://googleapis.github.io/python-genai/genai.html#genai.types.CreateCachedContentConfig.tools)
    - [`CreateCachedContentConfig.ttl`](https://googleapis.github.io/python-genai/genai.html#genai.types.CreateCachedContentConfig.ttl)
  - [`CreateCachedContentConfigDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.CreateCachedContentConfigDict)
    - [`CreateCachedContentConfigDict.contents`](https://googleapis.github.io/python-genai/genai.html#genai.types.CreateCachedContentConfigDict.contents)
    - [`CreateCachedContentConfigDict.display_name`](https://googleapis.github.io/python-genai/genai.html#genai.types.CreateCachedContentConfigDict.display_name)
    - [`CreateCachedContentConfigDict.expire_time`](https://googleapis.github.io/python-genai/genai.html#genai.types.CreateCachedContentConfigDict.expire_time)
    - [`CreateCachedContentConfigDict.http_options`](https://googleapis.github.io/python-genai/genai.html#genai.types.CreateCachedContentConfigDict.http_options)
    - [`CreateCachedContentConfigDict.kms_key_name`](https://googleapis.github.io/python-genai/genai.html#genai.types.CreateCachedContentConfigDict.kms_key_name)
    - [`CreateCachedContentConfigDict.system_instruction`](https://googleapis.github.io/python-genai/genai.html#genai.types.CreateCachedContentConfigDict.system_instruction)
    - [`CreateCachedContentConfigDict.tool_config`](https://googleapis.github.io/python-genai/genai.html#genai.types.CreateCachedContentConfigDict.tool_config)
    - [`CreateCachedContentConfigDict.tools`](https://googleapis.github.io/python-genai/genai.html#genai.types.CreateCachedContentConfigDict.tools)
    - [`CreateCachedContentConfigDict.ttl`](https://googleapis.github.io/python-genai/genai.html#genai.types.CreateCachedContentConfigDict.ttl)
  - [`CreateFileConfig`](https://googleapis.github.io/python-genai/genai.html#genai.types.CreateFileConfig)
    - [`CreateFileConfig.http_options`](https://googleapis.github.io/python-genai/genai.html#genai.types.CreateFileConfig.http_options)
    - [`CreateFileConfig.should_return_http_response`](https://googleapis.github.io/python-genai/genai.html#genai.types.CreateFileConfig.should_return_http_response)
  - [`CreateFileConfigDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.CreateFileConfigDict)
    - [`CreateFileConfigDict.http_options`](https://googleapis.github.io/python-genai/genai.html#genai.types.CreateFileConfigDict.http_options)
    - [`CreateFileConfigDict.should_return_http_response`](https://googleapis.github.io/python-genai/genai.html#genai.types.CreateFileConfigDict.should_return_http_response)
  - [`CreateFileResponse`](https://googleapis.github.io/python-genai/genai.html#genai.types.CreateFileResponse)
    - [`CreateFileResponse.sdk_http_response`](https://googleapis.github.io/python-genai/genai.html#genai.types.CreateFileResponse.sdk_http_response)
  - [`CreateFileResponseDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.CreateFileResponseDict)
    - [`CreateFileResponseDict.sdk_http_response`](https://googleapis.github.io/python-genai/genai.html#genai.types.CreateFileResponseDict.sdk_http_response)
  - [`CreateTuningJobConfig`](https://googleapis.github.io/python-genai/genai.html#genai.types.CreateTuningJobConfig)
    - [`CreateTuningJobConfig.adapter_size`](https://googleapis.github.io/python-genai/genai.html#genai.types.CreateTuningJobConfig.adapter_size)
    - [`CreateTuningJobConfig.batch_size`](https://googleapis.github.io/python-genai/genai.html#genai.types.CreateTuningJobConfig.batch_size)
    - [`CreateTuningJobConfig.description`](https://googleapis.github.io/python-genai/genai.html#genai.types.CreateTuningJobConfig.description)
    - [`CreateTuningJobConfig.epoch_count`](https://googleapis.github.io/python-genai/genai.html#genai.types.CreateTuningJobConfig.epoch_count)
    - [`CreateTuningJobConfig.export_last_checkpoint_only`](https://googleapis.github.io/python-genai/genai.html#genai.types.CreateTuningJobConfig.export_last_checkpoint_only)
    - [`CreateTuningJobConfig.http_options`](https://googleapis.github.io/python-genai/genai.html#genai.types.CreateTuningJobConfig.http_options)
    - [`CreateTuningJobConfig.learning_rate`](https://googleapis.github.io/python-genai/genai.html#genai.types.CreateTuningJobConfig.learning_rate)
    - [`CreateTuningJobConfig.learning_rate_multiplier`](https://googleapis.github.io/python-genai/genai.html#genai.types.CreateTuningJobConfig.learning_rate_multiplier)
    - [`CreateTuningJobConfig.tuned_model_display_name`](https://googleapis.github.io/python-genai/genai.html#genai.types.CreateTuningJobConfig.tuned_model_display_name)
    - [`CreateTuningJobConfig.validation_dataset`](https://googleapis.github.io/python-genai/genai.html#genai.types.CreateTuningJobConfig.validation_dataset)
  - [`CreateTuningJobConfigDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.CreateTuningJobConfigDict)
    - [`CreateTuningJobConfigDict.adapter_size`](https://googleapis.github.io/python-genai/genai.html#genai.types.CreateTuningJobConfigDict.adapter_size)
    - [`CreateTuningJobConfigDict.batch_size`](https://googleapis.github.io/python-genai/genai.html#genai.types.CreateTuningJobConfigDict.batch_size)
    - [`CreateTuningJobConfigDict.description`](https://googleapis.github.io/python-genai/genai.html#genai.types.CreateTuningJobConfigDict.description)
    - [`CreateTuningJobConfigDict.epoch_count`](https://googleapis.github.io/python-genai/genai.html#genai.types.CreateTuningJobConfigDict.epoch_count)
    - [`CreateTuningJobConfigDict.export_last_checkpoint_only`](https://googleapis.github.io/python-genai/genai.html#genai.types.CreateTuningJobConfigDict.export_last_checkpoint_only)
    - [`CreateTuningJobConfigDict.http_options`](https://googleapis.github.io/python-genai/genai.html#genai.types.CreateTuningJobConfigDict.http_options)
    - [`CreateTuningJobConfigDict.learning_rate`](https://googleapis.github.io/python-genai/genai.html#genai.types.CreateTuningJobConfigDict.learning_rate)
    - [`CreateTuningJobConfigDict.learning_rate_multiplier`](https://googleapis.github.io/python-genai/genai.html#genai.types.CreateTuningJobConfigDict.learning_rate_multiplier)
    - [`CreateTuningJobConfigDict.tuned_model_display_name`](https://googleapis.github.io/python-genai/genai.html#genai.types.CreateTuningJobConfigDict.tuned_model_display_name)
    - [`CreateTuningJobConfigDict.validation_dataset`](https://googleapis.github.io/python-genai/genai.html#genai.types.CreateTuningJobConfigDict.validation_dataset)
  - [`DatasetDistribution`](https://googleapis.github.io/python-genai/genai.html#genai.types.DatasetDistribution)
    - [`DatasetDistribution.buckets`](https://googleapis.github.io/python-genai/genai.html#genai.types.DatasetDistribution.buckets)
    - [`DatasetDistribution.max`](https://googleapis.github.io/python-genai/genai.html#genai.types.DatasetDistribution.max)
    - [`DatasetDistribution.mean`](https://googleapis.github.io/python-genai/genai.html#genai.types.DatasetDistribution.mean)
    - [`DatasetDistribution.median`](https://googleapis.github.io/python-genai/genai.html#genai.types.DatasetDistribution.median)
    - [`DatasetDistribution.min`](https://googleapis.github.io/python-genai/genai.html#genai.types.DatasetDistribution.min)
    - [`DatasetDistribution.p5`](https://googleapis.github.io/python-genai/genai.html#genai.types.DatasetDistribution.p5)
    - [`DatasetDistribution.p95`](https://googleapis.github.io/python-genai/genai.html#genai.types.DatasetDistribution.p95)
    - [`DatasetDistribution.sum`](https://googleapis.github.io/python-genai/genai.html#genai.types.DatasetDistribution.sum)
  - [`DatasetDistributionDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.DatasetDistributionDict)
    - [`DatasetDistributionDict.buckets`](https://googleapis.github.io/python-genai/genai.html#genai.types.DatasetDistributionDict.buckets)
    - [`DatasetDistributionDict.max`](https://googleapis.github.io/python-genai/genai.html#genai.types.DatasetDistributionDict.max)
    - [`DatasetDistributionDict.mean`](https://googleapis.github.io/python-genai/genai.html#genai.types.DatasetDistributionDict.mean)
    - [`DatasetDistributionDict.median`](https://googleapis.github.io/python-genai/genai.html#genai.types.DatasetDistributionDict.median)
    - [`DatasetDistributionDict.min`](https://googleapis.github.io/python-genai/genai.html#genai.types.DatasetDistributionDict.min)
    - [`DatasetDistributionDict.p5`](https://googleapis.github.io/python-genai/genai.html#genai.types.DatasetDistributionDict.p5)
    - [`DatasetDistributionDict.p95`](https://googleapis.github.io/python-genai/genai.html#genai.types.DatasetDistributionDict.p95)
    - [`DatasetDistributionDict.sum`](https://googleapis.github.io/python-genai/genai.html#genai.types.DatasetDistributionDict.sum)
  - [`DatasetDistributionDistributionBucket`](https://googleapis.github.io/python-genai/genai.html#genai.types.DatasetDistributionDistributionBucket)
    - [`DatasetDistributionDistributionBucket.count`](https://googleapis.github.io/python-genai/genai.html#genai.types.DatasetDistributionDistributionBucket.count)
    - [`DatasetDistributionDistributionBucket.left`](https://googleapis.github.io/python-genai/genai.html#genai.types.DatasetDistributionDistributionBucket.left)
    - [`DatasetDistributionDistributionBucket.right`](https://googleapis.github.io/python-genai/genai.html#genai.types.DatasetDistributionDistributionBucket.right)
  - [`DatasetDistributionDistributionBucketDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.DatasetDistributionDistributionBucketDict)
    - [`DatasetDistributionDistributionBucketDict.count`](https://googleapis.github.io/python-genai/genai.html#genai.types.DatasetDistributionDistributionBucketDict.count)
    - [`DatasetDistributionDistributionBucketDict.left`](https://googleapis.github.io/python-genai/genai.html#genai.types.DatasetDistributionDistributionBucketDict.left)
    - [`DatasetDistributionDistributionBucketDict.right`](https://googleapis.github.io/python-genai/genai.html#genai.types.DatasetDistributionDistributionBucketDict.right)
  - [`DatasetStats`](https://googleapis.github.io/python-genai/genai.html#genai.types.DatasetStats)
    - [`DatasetStats.total_billable_character_count`](https://googleapis.github.io/python-genai/genai.html#genai.types.DatasetStats.total_billable_character_count)
    - [`DatasetStats.total_tuning_character_count`](https://googleapis.github.io/python-genai/genai.html#genai.types.DatasetStats.total_tuning_character_count)
    - [`DatasetStats.tuning_dataset_example_count`](https://googleapis.github.io/python-genai/genai.html#genai.types.DatasetStats.tuning_dataset_example_count)
    - [`DatasetStats.tuning_step_count`](https://googleapis.github.io/python-genai/genai.html#genai.types.DatasetStats.tuning_step_count)
    - [`DatasetStats.user_dataset_examples`](https://googleapis.github.io/python-genai/genai.html#genai.types.DatasetStats.user_dataset_examples)
    - [`DatasetStats.user_input_token_distribution`](https://googleapis.github.io/python-genai/genai.html#genai.types.DatasetStats.user_input_token_distribution)
    - [`DatasetStats.user_message_per_example_distribution`](https://googleapis.github.io/python-genai/genai.html#genai.types.DatasetStats.user_message_per_example_distribution)
    - [`DatasetStats.user_output_token_distribution`](https://googleapis.github.io/python-genai/genai.html#genai.types.DatasetStats.user_output_token_distribution)
  - [`DatasetStatsDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.DatasetStatsDict)
    - [`DatasetStatsDict.total_billable_character_count`](https://googleapis.github.io/python-genai/genai.html#genai.types.DatasetStatsDict.total_billable_character_count)
    - [`DatasetStatsDict.total_tuning_character_count`](https://googleapis.github.io/python-genai/genai.html#genai.types.DatasetStatsDict.total_tuning_character_count)
    - [`DatasetStatsDict.tuning_dataset_example_count`](https://googleapis.github.io/python-genai/genai.html#genai.types.DatasetStatsDict.tuning_dataset_example_count)
    - [`DatasetStatsDict.tuning_step_count`](https://googleapis.github.io/python-genai/genai.html#genai.types.DatasetStatsDict.tuning_step_count)
    - [`DatasetStatsDict.user_dataset_examples`](https://googleapis.github.io/python-genai/genai.html#genai.types.DatasetStatsDict.user_dataset_examples)
    - [`DatasetStatsDict.user_input_token_distribution`](https://googleapis.github.io/python-genai/genai.html#genai.types.DatasetStatsDict.user_input_token_distribution)
    - [`DatasetStatsDict.user_message_per_example_distribution`](https://googleapis.github.io/python-genai/genai.html#genai.types.DatasetStatsDict.user_message_per_example_distribution)
    - [`DatasetStatsDict.user_output_token_distribution`](https://googleapis.github.io/python-genai/genai.html#genai.types.DatasetStatsDict.user_output_token_distribution)
  - [`DeleteBatchJobConfig`](https://googleapis.github.io/python-genai/genai.html#genai.types.DeleteBatchJobConfig)
    - [`DeleteBatchJobConfig.http_options`](https://googleapis.github.io/python-genai/genai.html#genai.types.DeleteBatchJobConfig.http_options)
  - [`DeleteBatchJobConfigDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.DeleteBatchJobConfigDict)
    - [`DeleteBatchJobConfigDict.http_options`](https://googleapis.github.io/python-genai/genai.html#genai.types.DeleteBatchJobConfigDict.http_options)
  - [`DeleteCachedContentConfig`](https://googleapis.github.io/python-genai/genai.html#genai.types.DeleteCachedContentConfig)
    - [`DeleteCachedContentConfig.http_options`](https://googleapis.github.io/python-genai/genai.html#genai.types.DeleteCachedContentConfig.http_options)
  - [`DeleteCachedContentConfigDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.DeleteCachedContentConfigDict)
    - [`DeleteCachedContentConfigDict.http_options`](https://googleapis.github.io/python-genai/genai.html#genai.types.DeleteCachedContentConfigDict.http_options)
  - [`DeleteCachedContentResponse`](https://googleapis.github.io/python-genai/genai.html#genai.types.DeleteCachedContentResponse)
  - [`DeleteCachedContentResponseDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.DeleteCachedContentResponseDict)
  - [`DeleteFileConfig`](https://googleapis.github.io/python-genai/genai.html#genai.types.DeleteFileConfig)
    - [`DeleteFileConfig.http_options`](https://googleapis.github.io/python-genai/genai.html#genai.types.DeleteFileConfig.http_options)
  - [`DeleteFileConfigDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.DeleteFileConfigDict)
    - [`DeleteFileConfigDict.http_options`](https://googleapis.github.io/python-genai/genai.html#genai.types.DeleteFileConfigDict.http_options)
  - [`DeleteFileResponse`](https://googleapis.github.io/python-genai/genai.html#genai.types.DeleteFileResponse)
  - [`DeleteFileResponseDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.DeleteFileResponseDict)
  - [`DeleteModelConfig`](https://googleapis.github.io/python-genai/genai.html#genai.types.DeleteModelConfig)
    - [`DeleteModelConfig.http_options`](https://googleapis.github.io/python-genai/genai.html#genai.types.DeleteModelConfig.http_options)
  - [`DeleteModelConfigDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.DeleteModelConfigDict)
    - [`DeleteModelConfigDict.http_options`](https://googleapis.github.io/python-genai/genai.html#genai.types.DeleteModelConfigDict.http_options)
  - [`DeleteModelResponse`](https://googleapis.github.io/python-genai/genai.html#genai.types.DeleteModelResponse)
  - [`DeleteModelResponseDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.DeleteModelResponseDict)
  - [`DeleteResourceJob`](https://googleapis.github.io/python-genai/genai.html#genai.types.DeleteResourceJob)
    - [`DeleteResourceJob.done`](https://googleapis.github.io/python-genai/genai.html#genai.types.DeleteResourceJob.done)
    - [`DeleteResourceJob.error`](https://googleapis.github.io/python-genai/genai.html#genai.types.DeleteResourceJob.error)
    - [`DeleteResourceJob.name`](https://googleapis.github.io/python-genai/genai.html#genai.types.DeleteResourceJob.name)
  - [`DeleteResourceJobDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.DeleteResourceJobDict)
    - [`DeleteResourceJobDict.done`](https://googleapis.github.io/python-genai/genai.html#genai.types.DeleteResourceJobDict.done)
    - [`DeleteResourceJobDict.error`](https://googleapis.github.io/python-genai/genai.html#genai.types.DeleteResourceJobDict.error)
    - [`DeleteResourceJobDict.name`](https://googleapis.github.io/python-genai/genai.html#genai.types.DeleteResourceJobDict.name)
  - [`DistillationDataStats`](https://googleapis.github.io/python-genai/genai.html#genai.types.DistillationDataStats)
    - [`DistillationDataStats.training_dataset_stats`](https://googleapis.github.io/python-genai/genai.html#genai.types.DistillationDataStats.training_dataset_stats)
  - [`DistillationDataStatsDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.DistillationDataStatsDict)
    - [`DistillationDataStatsDict.training_dataset_stats`](https://googleapis.github.io/python-genai/genai.html#genai.types.DistillationDataStatsDict.training_dataset_stats)
  - [`DistillationHyperParameters`](https://googleapis.github.io/python-genai/genai.html#genai.types.DistillationHyperParameters)
    - [`DistillationHyperParameters.adapter_size`](https://googleapis.github.io/python-genai/genai.html#genai.types.DistillationHyperParameters.adapter_size)
    - [`DistillationHyperParameters.epoch_count`](https://googleapis.github.io/python-genai/genai.html#genai.types.DistillationHyperParameters.epoch_count)
    - [`DistillationHyperParameters.learning_rate_multiplier`](https://googleapis.github.io/python-genai/genai.html#genai.types.DistillationHyperParameters.learning_rate_multiplier)
  - [`DistillationHyperParametersDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.DistillationHyperParametersDict)
    - [`DistillationHyperParametersDict.adapter_size`](https://googleapis.github.io/python-genai/genai.html#genai.types.DistillationHyperParametersDict.adapter_size)
    - [`DistillationHyperParametersDict.epoch_count`](https://googleapis.github.io/python-genai/genai.html#genai.types.DistillationHyperParametersDict.epoch_count)
    - [`DistillationHyperParametersDict.learning_rate_multiplier`](https://googleapis.github.io/python-genai/genai.html#genai.types.DistillationHyperParametersDict.learning_rate_multiplier)
  - [`DistillationSpec`](https://googleapis.github.io/python-genai/genai.html#genai.types.DistillationSpec)
    - [`DistillationSpec.base_teacher_model`](https://googleapis.github.io/python-genai/genai.html#genai.types.DistillationSpec.base_teacher_model)
    - [`DistillationSpec.hyper_parameters`](https://googleapis.github.io/python-genai/genai.html#genai.types.DistillationSpec.hyper_parameters)
    - [`DistillationSpec.pipeline_root_directory`](https://googleapis.github.io/python-genai/genai.html#genai.types.DistillationSpec.pipeline_root_directory)
    - [`DistillationSpec.student_model`](https://googleapis.github.io/python-genai/genai.html#genai.types.DistillationSpec.student_model)
    - [`DistillationSpec.training_dataset_uri`](https://googleapis.github.io/python-genai/genai.html#genai.types.DistillationSpec.training_dataset_uri)
    - [`DistillationSpec.tuned_teacher_model_source`](https://googleapis.github.io/python-genai/genai.html#genai.types.DistillationSpec.tuned_teacher_model_source)
    - [`DistillationSpec.validation_dataset_uri`](https://googleapis.github.io/python-genai/genai.html#genai.types.DistillationSpec.validation_dataset_uri)
  - [`DistillationSpecDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.DistillationSpecDict)
    - [`DistillationSpecDict.base_teacher_model`](https://googleapis.github.io/python-genai/genai.html#genai.types.DistillationSpecDict.base_teacher_model)
    - [`DistillationSpecDict.hyper_parameters`](https://googleapis.github.io/python-genai/genai.html#genai.types.DistillationSpecDict.hyper_parameters)
    - [`DistillationSpecDict.pipeline_root_directory`](https://googleapis.github.io/python-genai/genai.html#genai.types.DistillationSpecDict.pipeline_root_directory)
    - [`DistillationSpecDict.student_model`](https://googleapis.github.io/python-genai/genai.html#genai.types.DistillationSpecDict.student_model)
    - [`DistillationSpecDict.training_dataset_uri`](https://googleapis.github.io/python-genai/genai.html#genai.types.DistillationSpecDict.training_dataset_uri)
    - [`DistillationSpecDict.tuned_teacher_model_source`](https://googleapis.github.io/python-genai/genai.html#genai.types.DistillationSpecDict.tuned_teacher_model_source)
    - [`DistillationSpecDict.validation_dataset_uri`](https://googleapis.github.io/python-genai/genai.html#genai.types.DistillationSpecDict.validation_dataset_uri)
  - [`DownloadFileConfig`](https://googleapis.github.io/python-genai/genai.html#genai.types.DownloadFileConfig)
    - [`DownloadFileConfig.http_options`](https://googleapis.github.io/python-genai/genai.html#genai.types.DownloadFileConfig.http_options)
  - [`DownloadFileConfigDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.DownloadFileConfigDict)
    - [`DownloadFileConfigDict.http_options`](https://googleapis.github.io/python-genai/genai.html#genai.types.DownloadFileConfigDict.http_options)
  - [`DynamicRetrievalConfig`](https://googleapis.github.io/python-genai/genai.html#genai.types.DynamicRetrievalConfig)
    - [`DynamicRetrievalConfig.dynamic_threshold`](https://googleapis.github.io/python-genai/genai.html#genai.types.DynamicRetrievalConfig.dynamic_threshold)
    - [`DynamicRetrievalConfig.mode`](https://googleapis.github.io/python-genai/genai.html#genai.types.DynamicRetrievalConfig.mode)
  - [`DynamicRetrievalConfigDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.DynamicRetrievalConfigDict)
    - [`DynamicRetrievalConfigDict.dynamic_threshold`](https://googleapis.github.io/python-genai/genai.html#genai.types.DynamicRetrievalConfigDict.dynamic_threshold)
    - [`DynamicRetrievalConfigDict.mode`](https://googleapis.github.io/python-genai/genai.html#genai.types.DynamicRetrievalConfigDict.mode)
  - [`DynamicRetrievalConfigMode`](https://googleapis.github.io/python-genai/genai.html#genai.types.DynamicRetrievalConfigMode)
    - [`DynamicRetrievalConfigMode.MODE_DYNAMIC`](https://googleapis.github.io/python-genai/genai.html#genai.types.DynamicRetrievalConfigMode.MODE_DYNAMIC)
    - [`DynamicRetrievalConfigMode.MODE_UNSPECIFIED`](https://googleapis.github.io/python-genai/genai.html#genai.types.DynamicRetrievalConfigMode.MODE_UNSPECIFIED)
  - [`EditImageConfig`](https://googleapis.github.io/python-genai/genai.html#genai.types.EditImageConfig)
    - [`EditImageConfig.aspect_ratio`](https://googleapis.github.io/python-genai/genai.html#genai.types.EditImageConfig.aspect_ratio)
    - [`EditImageConfig.base_steps`](https://googleapis.github.io/python-genai/genai.html#genai.types.EditImageConfig.base_steps)
    - [`EditImageConfig.edit_mode`](https://googleapis.github.io/python-genai/genai.html#genai.types.EditImageConfig.edit_mode)
    - [`EditImageConfig.guidance_scale`](https://googleapis.github.io/python-genai/genai.html#genai.types.EditImageConfig.guidance_scale)
    - [`EditImageConfig.http_options`](https://googleapis.github.io/python-genai/genai.html#genai.types.EditImageConfig.http_options)
    - [`EditImageConfig.include_rai_reason`](https://googleapis.github.io/python-genai/genai.html#genai.types.EditImageConfig.include_rai_reason)
    - [`EditImageConfig.include_safety_attributes`](https://googleapis.github.io/python-genai/genai.html#genai.types.EditImageConfig.include_safety_attributes)
    - [`EditImageConfig.language`](https://googleapis.github.io/python-genai/genai.html#genai.types.EditImageConfig.language)
    - [`EditImageConfig.negative_prompt`](https://googleapis.github.io/python-genai/genai.html#genai.types.EditImageConfig.negative_prompt)
    - [`EditImageConfig.number_of_images`](https://googleapis.github.io/python-genai/genai.html#genai.types.EditImageConfig.number_of_images)
    - [`EditImageConfig.output_compression_quality`](https://googleapis.github.io/python-genai/genai.html#genai.types.EditImageConfig.output_compression_quality)
    - [`EditImageConfig.output_gcs_uri`](https://googleapis.github.io/python-genai/genai.html#genai.types.EditImageConfig.output_gcs_uri)
    - [`EditImageConfig.output_mime_type`](https://googleapis.github.io/python-genai/genai.html#genai.types.EditImageConfig.output_mime_type)
    - [`EditImageConfig.person_generation`](https://googleapis.github.io/python-genai/genai.html#genai.types.EditImageConfig.person_generation)
    - [`EditImageConfig.safety_filter_level`](https://googleapis.github.io/python-genai/genai.html#genai.types.EditImageConfig.safety_filter_level)
    - [`EditImageConfig.seed`](https://googleapis.github.io/python-genai/genai.html#genai.types.EditImageConfig.seed)
  - [`EditImageConfigDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.EditImageConfigDict)
    - [`EditImageConfigDict.aspect_ratio`](https://googleapis.github.io/python-genai/genai.html#genai.types.EditImageConfigDict.aspect_ratio)
    - [`EditImageConfigDict.base_steps`](https://googleapis.github.io/python-genai/genai.html#genai.types.EditImageConfigDict.base_steps)
    - [`EditImageConfigDict.edit_mode`](https://googleapis.github.io/python-genai/genai.html#genai.types.EditImageConfigDict.edit_mode)
    - [`EditImageConfigDict.guidance_scale`](https://googleapis.github.io/python-genai/genai.html#genai.types.EditImageConfigDict.guidance_scale)
    - [`EditImageConfigDict.http_options`](https://googleapis.github.io/python-genai/genai.html#genai.types.EditImageConfigDict.http_options)
    - [`EditImageConfigDict.include_rai_reason`](https://googleapis.github.io/python-genai/genai.html#genai.types.EditImageConfigDict.include_rai_reason)
    - [`EditImageConfigDict.include_safety_attributes`](https://googleapis.github.io/python-genai/genai.html#genai.types.EditImageConfigDict.include_safety_attributes)
    - [`EditImageConfigDict.language`](https://googleapis.github.io/python-genai/genai.html#genai.types.EditImageConfigDict.language)
    - [`EditImageConfigDict.negative_prompt`](https://googleapis.github.io/python-genai/genai.html#genai.types.EditImageConfigDict.negative_prompt)
    - [`EditImageConfigDict.number_of_images`](https://googleapis.github.io/python-genai/genai.html#genai.types.EditImageConfigDict.number_of_images)
    - [`EditImageConfigDict.output_compression_quality`](https://googleapis.github.io/python-genai/genai.html#genai.types.EditImageConfigDict.output_compression_quality)
    - [`EditImageConfigDict.output_gcs_uri`](https://googleapis.github.io/python-genai/genai.html#genai.types.EditImageConfigDict.output_gcs_uri)
    - [`EditImageConfigDict.output_mime_type`](https://googleapis.github.io/python-genai/genai.html#genai.types.EditImageConfigDict.output_mime_type)
    - [`EditImageConfigDict.person_generation`](https://googleapis.github.io/python-genai/genai.html#genai.types.EditImageConfigDict.person_generation)
    - [`EditImageConfigDict.safety_filter_level`](https://googleapis.github.io/python-genai/genai.html#genai.types.EditImageConfigDict.safety_filter_level)
    - [`EditImageConfigDict.seed`](https://googleapis.github.io/python-genai/genai.html#genai.types.EditImageConfigDict.seed)
  - [`EditImageResponse`](https://googleapis.github.io/python-genai/genai.html#genai.types.EditImageResponse)
    - [`EditImageResponse.generated_images`](https://googleapis.github.io/python-genai/genai.html#genai.types.EditImageResponse.generated_images)
  - [`EditImageResponseDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.EditImageResponseDict)
    - [`EditImageResponseDict.generated_images`](https://googleapis.github.io/python-genai/genai.html#genai.types.EditImageResponseDict.generated_images)
  - [`EditMode`](https://googleapis.github.io/python-genai/genai.html#genai.types.EditMode)
    - [`EditMode.EDIT_MODE_BGSWAP`](https://googleapis.github.io/python-genai/genai.html#genai.types.EditMode.EDIT_MODE_BGSWAP)
    - [`EditMode.EDIT_MODE_CONTROLLED_EDITING`](https://googleapis.github.io/python-genai/genai.html#genai.types.EditMode.EDIT_MODE_CONTROLLED_EDITING)
    - [`EditMode.EDIT_MODE_DEFAULT`](https://googleapis.github.io/python-genai/genai.html#genai.types.EditMode.EDIT_MODE_DEFAULT)
    - [`EditMode.EDIT_MODE_INPAINT_INSERTION`](https://googleapis.github.io/python-genai/genai.html#genai.types.EditMode.EDIT_MODE_INPAINT_INSERTION)
    - [`EditMode.EDIT_MODE_INPAINT_REMOVAL`](https://googleapis.github.io/python-genai/genai.html#genai.types.EditMode.EDIT_MODE_INPAINT_REMOVAL)
    - [`EditMode.EDIT_MODE_OUTPAINT`](https://googleapis.github.io/python-genai/genai.html#genai.types.EditMode.EDIT_MODE_OUTPAINT)
    - [`EditMode.EDIT_MODE_PRODUCT_IMAGE`](https://googleapis.github.io/python-genai/genai.html#genai.types.EditMode.EDIT_MODE_PRODUCT_IMAGE)
    - [`EditMode.EDIT_MODE_STYLE`](https://googleapis.github.io/python-genai/genai.html#genai.types.EditMode.EDIT_MODE_STYLE)
  - [`EmbedContentConfig`](https://googleapis.github.io/python-genai/genai.html#genai.types.EmbedContentConfig)
    - [`EmbedContentConfig.auto_truncate`](https://googleapis.github.io/python-genai/genai.html#genai.types.EmbedContentConfig.auto_truncate)
    - [`EmbedContentConfig.http_options`](https://googleapis.github.io/python-genai/genai.html#genai.types.EmbedContentConfig.http_options)
    - [`EmbedContentConfig.mime_type`](https://googleapis.github.io/python-genai/genai.html#genai.types.EmbedContentConfig.mime_type)
    - [`EmbedContentConfig.output_dimensionality`](https://googleapis.github.io/python-genai/genai.html#genai.types.EmbedContentConfig.output_dimensionality)
    - [`EmbedContentConfig.task_type`](https://googleapis.github.io/python-genai/genai.html#genai.types.EmbedContentConfig.task_type)
    - [`EmbedContentConfig.title`](https://googleapis.github.io/python-genai/genai.html#genai.types.EmbedContentConfig.title)
  - [`EmbedContentConfigDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.EmbedContentConfigDict)
    - [`EmbedContentConfigDict.auto_truncate`](https://googleapis.github.io/python-genai/genai.html#genai.types.EmbedContentConfigDict.auto_truncate)
    - [`EmbedContentConfigDict.http_options`](https://googleapis.github.io/python-genai/genai.html#genai.types.EmbedContentConfigDict.http_options)
    - [`EmbedContentConfigDict.mime_type`](https://googleapis.github.io/python-genai/genai.html#genai.types.EmbedContentConfigDict.mime_type)
    - [`EmbedContentConfigDict.output_dimensionality`](https://googleapis.github.io/python-genai/genai.html#genai.types.EmbedContentConfigDict.output_dimensionality)
    - [`EmbedContentConfigDict.task_type`](https://googleapis.github.io/python-genai/genai.html#genai.types.EmbedContentConfigDict.task_type)
    - [`EmbedContentConfigDict.title`](https://googleapis.github.io/python-genai/genai.html#genai.types.EmbedContentConfigDict.title)
  - [`EmbedContentMetadata`](https://googleapis.github.io/python-genai/genai.html#genai.types.EmbedContentMetadata)
    - [`EmbedContentMetadata.billable_character_count`](https://googleapis.github.io/python-genai/genai.html#genai.types.EmbedContentMetadata.billable_character_count)
  - [`EmbedContentMetadataDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.EmbedContentMetadataDict)
    - [`EmbedContentMetadataDict.billable_character_count`](https://googleapis.github.io/python-genai/genai.html#genai.types.EmbedContentMetadataDict.billable_character_count)
  - [`EmbedContentResponse`](https://googleapis.github.io/python-genai/genai.html#genai.types.EmbedContentResponse)
    - [`EmbedContentResponse.embeddings`](https://googleapis.github.io/python-genai/genai.html#genai.types.EmbedContentResponse.embeddings)
    - [`EmbedContentResponse.metadata`](https://googleapis.github.io/python-genai/genai.html#genai.types.EmbedContentResponse.metadata)
  - [`EmbedContentResponseDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.EmbedContentResponseDict)
    - [`EmbedContentResponseDict.embeddings`](https://googleapis.github.io/python-genai/genai.html#genai.types.EmbedContentResponseDict.embeddings)
    - [`EmbedContentResponseDict.metadata`](https://googleapis.github.io/python-genai/genai.html#genai.types.EmbedContentResponseDict.metadata)
  - [`EncryptionSpec`](https://googleapis.github.io/python-genai/genai.html#genai.types.EncryptionSpec)
    - [`EncryptionSpec.kms_key_name`](https://googleapis.github.io/python-genai/genai.html#genai.types.EncryptionSpec.kms_key_name)
  - [`EncryptionSpecDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.EncryptionSpecDict)
    - [`EncryptionSpecDict.kms_key_name`](https://googleapis.github.io/python-genai/genai.html#genai.types.EncryptionSpecDict.kms_key_name)
  - [`EndSensitivity`](https://googleapis.github.io/python-genai/genai.html#genai.types.EndSensitivity)
    - [`EndSensitivity.END_SENSITIVITY_HIGH`](https://googleapis.github.io/python-genai/genai.html#genai.types.EndSensitivity.END_SENSITIVITY_HIGH)
    - [`EndSensitivity.END_SENSITIVITY_LOW`](https://googleapis.github.io/python-genai/genai.html#genai.types.EndSensitivity.END_SENSITIVITY_LOW)
    - [`EndSensitivity.END_SENSITIVITY_UNSPECIFIED`](https://googleapis.github.io/python-genai/genai.html#genai.types.EndSensitivity.END_SENSITIVITY_UNSPECIFIED)
  - [`Endpoint`](https://googleapis.github.io/python-genai/genai.html#genai.types.Endpoint)
    - [`Endpoint.deployed_model_id`](https://googleapis.github.io/python-genai/genai.html#genai.types.Endpoint.deployed_model_id)
    - [`Endpoint.name`](https://googleapis.github.io/python-genai/genai.html#genai.types.Endpoint.name)
  - [`EndpointDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.EndpointDict)
    - [`EndpointDict.deployed_model_id`](https://googleapis.github.io/python-genai/genai.html#genai.types.EndpointDict.deployed_model_id)
    - [`EndpointDict.name`](https://googleapis.github.io/python-genai/genai.html#genai.types.EndpointDict.name)
  - [`EnterpriseWebSearch`](https://googleapis.github.io/python-genai/genai.html#genai.types.EnterpriseWebSearch)
  - [`EnterpriseWebSearchDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.EnterpriseWebSearchDict)
  - [`ExecutableCode`](https://googleapis.github.io/python-genai/genai.html#genai.types.ExecutableCode)
    - [`ExecutableCode.code`](https://googleapis.github.io/python-genai/genai.html#genai.types.ExecutableCode.code)
    - [`ExecutableCode.language`](https://googleapis.github.io/python-genai/genai.html#genai.types.ExecutableCode.language)
  - [`ExecutableCodeDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.ExecutableCodeDict)
    - [`ExecutableCodeDict.code`](https://googleapis.github.io/python-genai/genai.html#genai.types.ExecutableCodeDict.code)
    - [`ExecutableCodeDict.language`](https://googleapis.github.io/python-genai/genai.html#genai.types.ExecutableCodeDict.language)
  - [`FeatureSelectionPreference`](https://googleapis.github.io/python-genai/genai.html#genai.types.FeatureSelectionPreference)
    - [`FeatureSelectionPreference.BALANCED`](https://googleapis.github.io/python-genai/genai.html#genai.types.FeatureSelectionPreference.BALANCED)
    - [`FeatureSelectionPreference.FEATURE_SELECTION_PREFERENCE_UNSPECIFIED`](https://googleapis.github.io/python-genai/genai.html#genai.types.FeatureSelectionPreference.FEATURE_SELECTION_PREFERENCE_UNSPECIFIED)
    - [`FeatureSelectionPreference.PRIORITIZE_COST`](https://googleapis.github.io/python-genai/genai.html#genai.types.FeatureSelectionPreference.PRIORITIZE_COST)
    - [`FeatureSelectionPreference.PRIORITIZE_QUALITY`](https://googleapis.github.io/python-genai/genai.html#genai.types.FeatureSelectionPreference.PRIORITIZE_QUALITY)
  - [`FetchPredictOperationConfig`](https://googleapis.github.io/python-genai/genai.html#genai.types.FetchPredictOperationConfig)
    - [`FetchPredictOperationConfig.http_options`](https://googleapis.github.io/python-genai/genai.html#genai.types.FetchPredictOperationConfig.http_options)
  - [`FetchPredictOperationConfigDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.FetchPredictOperationConfigDict)
    - [`FetchPredictOperationConfigDict.http_options`](https://googleapis.github.io/python-genai/genai.html#genai.types.FetchPredictOperationConfigDict.http_options)
  - [`File`](https://googleapis.github.io/python-genai/genai.html#genai.types.File)
    - [`File.create_time`](https://googleapis.github.io/python-genai/genai.html#genai.types.File.create_time)
    - [`File.display_name`](https://googleapis.github.io/python-genai/genai.html#genai.types.File.display_name)
    - [`File.download_uri`](https://googleapis.github.io/python-genai/genai.html#genai.types.File.download_uri)
    - [`File.error`](https://googleapis.github.io/python-genai/genai.html#genai.types.File.error)
    - [`File.expiration_time`](https://googleapis.github.io/python-genai/genai.html#genai.types.File.expiration_time)
    - [`File.mime_type`](https://googleapis.github.io/python-genai/genai.html#genai.types.File.mime_type)
    - [`File.name`](https://googleapis.github.io/python-genai/genai.html#genai.types.File.name)
    - [`File.sha256_hash`](https://googleapis.github.io/python-genai/genai.html#genai.types.File.sha256_hash)
    - [`File.size_bytes`](https://googleapis.github.io/python-genai/genai.html#genai.types.File.size_bytes)
    - [`File.source`](https://googleapis.github.io/python-genai/genai.html#genai.types.File.source)
    - [`File.state`](https://googleapis.github.io/python-genai/genai.html#genai.types.File.state)
    - [`File.update_time`](https://googleapis.github.io/python-genai/genai.html#genai.types.File.update_time)
    - [`File.uri`](https://googleapis.github.io/python-genai/genai.html#genai.types.File.uri)
    - [`File.video_metadata`](https://googleapis.github.io/python-genai/genai.html#genai.types.File.video_metadata)
  - [`FileData`](https://googleapis.github.io/python-genai/genai.html#genai.types.FileData)
    - [`FileData.display_name`](https://googleapis.github.io/python-genai/genai.html#genai.types.FileData.display_name)
    - [`FileData.file_uri`](https://googleapis.github.io/python-genai/genai.html#genai.types.FileData.file_uri)
    - [`FileData.mime_type`](https://googleapis.github.io/python-genai/genai.html#genai.types.FileData.mime_type)
  - [`FileDataDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.FileDataDict)
    - [`FileDataDict.display_name`](https://googleapis.github.io/python-genai/genai.html#genai.types.FileDataDict.display_name)
    - [`FileDataDict.file_uri`](https://googleapis.github.io/python-genai/genai.html#genai.types.FileDataDict.file_uri)
    - [`FileDataDict.mime_type`](https://googleapis.github.io/python-genai/genai.html#genai.types.FileDataDict.mime_type)
  - [`FileDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.FileDict)
    - [`FileDict.create_time`](https://googleapis.github.io/python-genai/genai.html#genai.types.FileDict.create_time)
    - [`FileDict.display_name`](https://googleapis.github.io/python-genai/genai.html#genai.types.FileDict.display_name)
    - [`FileDict.download_uri`](https://googleapis.github.io/python-genai/genai.html#genai.types.FileDict.download_uri)
    - [`FileDict.error`](https://googleapis.github.io/python-genai/genai.html#genai.types.FileDict.error)
    - [`FileDict.expiration_time`](https://googleapis.github.io/python-genai/genai.html#genai.types.FileDict.expiration_time)
    - [`FileDict.mime_type`](https://googleapis.github.io/python-genai/genai.html#genai.types.FileDict.mime_type)
    - [`FileDict.name`](https://googleapis.github.io/python-genai/genai.html#genai.types.FileDict.name)
    - [`FileDict.sha256_hash`](https://googleapis.github.io/python-genai/genai.html#genai.types.FileDict.sha256_hash)
    - [`FileDict.size_bytes`](https://googleapis.github.io/python-genai/genai.html#genai.types.FileDict.size_bytes)
    - [`FileDict.source`](https://googleapis.github.io/python-genai/genai.html#genai.types.FileDict.source)
    - [`FileDict.state`](https://googleapis.github.io/python-genai/genai.html#genai.types.FileDict.state)
    - [`FileDict.update_time`](https://googleapis.github.io/python-genai/genai.html#genai.types.FileDict.update_time)
    - [`FileDict.uri`](https://googleapis.github.io/python-genai/genai.html#genai.types.FileDict.uri)
    - [`FileDict.video_metadata`](https://googleapis.github.io/python-genai/genai.html#genai.types.FileDict.video_metadata)
  - [`FileSource`](https://googleapis.github.io/python-genai/genai.html#genai.types.FileSource)
    - [`FileSource.GENERATED`](https://googleapis.github.io/python-genai/genai.html#genai.types.FileSource.GENERATED)
    - [`FileSource.SOURCE_UNSPECIFIED`](https://googleapis.github.io/python-genai/genai.html#genai.types.FileSource.SOURCE_UNSPECIFIED)
    - [`FileSource.UPLOADED`](https://googleapis.github.io/python-genai/genai.html#genai.types.FileSource.UPLOADED)
  - [`FileState`](https://googleapis.github.io/python-genai/genai.html#genai.types.FileState)
    - [`FileState.ACTIVE`](https://googleapis.github.io/python-genai/genai.html#genai.types.FileState.ACTIVE)
    - [`FileState.FAILED`](https://googleapis.github.io/python-genai/genai.html#genai.types.FileState.FAILED)
    - [`FileState.PROCESSING`](https://googleapis.github.io/python-genai/genai.html#genai.types.FileState.PROCESSING)
    - [`FileState.STATE_UNSPECIFIED`](https://googleapis.github.io/python-genai/genai.html#genai.types.FileState.STATE_UNSPECIFIED)
  - [`FileStatus`](https://googleapis.github.io/python-genai/genai.html#genai.types.FileStatus)
    - [`FileStatus.code`](https://googleapis.github.io/python-genai/genai.html#genai.types.FileStatus.code)
    - [`FileStatus.details`](https://googleapis.github.io/python-genai/genai.html#genai.types.FileStatus.details)
    - [`FileStatus.message`](https://googleapis.github.io/python-genai/genai.html#genai.types.FileStatus.message)
  - [`FileStatusDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.FileStatusDict)
    - [`FileStatusDict.code`](https://googleapis.github.io/python-genai/genai.html#genai.types.FileStatusDict.code)
    - [`FileStatusDict.details`](https://googleapis.github.io/python-genai/genai.html#genai.types.FileStatusDict.details)
    - [`FileStatusDict.message`](https://googleapis.github.io/python-genai/genai.html#genai.types.FileStatusDict.message)
  - [`FinishReason`](https://googleapis.github.io/python-genai/genai.html#genai.types.FinishReason)
    - [`FinishReason.BLOCKLIST`](https://googleapis.github.io/python-genai/genai.html#genai.types.FinishReason.BLOCKLIST)
    - [`FinishReason.FINISH_REASON_UNSPECIFIED`](https://googleapis.github.io/python-genai/genai.html#genai.types.FinishReason.FINISH_REASON_UNSPECIFIED)
    - [`FinishReason.IMAGE_SAFETY`](https://googleapis.github.io/python-genai/genai.html#genai.types.FinishReason.IMAGE_SAFETY)
    - [`FinishReason.LANGUAGE`](https://googleapis.github.io/python-genai/genai.html#genai.types.FinishReason.LANGUAGE)
    - [`FinishReason.MALFORMED_FUNCTION_CALL`](https://googleapis.github.io/python-genai/genai.html#genai.types.FinishReason.MALFORMED_FUNCTION_CALL)
    - [`FinishReason.MAX_TOKENS`](https://googleapis.github.io/python-genai/genai.html#genai.types.FinishReason.MAX_TOKENS)
    - [`FinishReason.OTHER`](https://googleapis.github.io/python-genai/genai.html#genai.types.FinishReason.OTHER)
    - [`FinishReason.PROHIBITED_CONTENT`](https://googleapis.github.io/python-genai/genai.html#genai.types.FinishReason.PROHIBITED_CONTENT)
    - [`FinishReason.RECITATION`](https://googleapis.github.io/python-genai/genai.html#genai.types.FinishReason.RECITATION)
    - [`FinishReason.SAFETY`](https://googleapis.github.io/python-genai/genai.html#genai.types.FinishReason.SAFETY)
    - [`FinishReason.SPII`](https://googleapis.github.io/python-genai/genai.html#genai.types.FinishReason.SPII)
    - [`FinishReason.STOP`](https://googleapis.github.io/python-genai/genai.html#genai.types.FinishReason.STOP)
    - [`FinishReason.UNEXPECTED_TOOL_CALL`](https://googleapis.github.io/python-genai/genai.html#genai.types.FinishReason.UNEXPECTED_TOOL_CALL)
  - [`FunctionCall`](https://googleapis.github.io/python-genai/genai.html#genai.types.FunctionCall)
    - [`FunctionCall.args`](https://googleapis.github.io/python-genai/genai.html#genai.types.FunctionCall.args)
    - [`FunctionCall.id`](https://googleapis.github.io/python-genai/genai.html#genai.types.FunctionCall.id)
    - [`FunctionCall.name`](https://googleapis.github.io/python-genai/genai.html#genai.types.FunctionCall.name)
  - [`FunctionCallDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.FunctionCallDict)
    - [`FunctionCallDict.args`](https://googleapis.github.io/python-genai/genai.html#genai.types.FunctionCallDict.args)
    - [`FunctionCallDict.id`](https://googleapis.github.io/python-genai/genai.html#genai.types.FunctionCallDict.id)
    - [`FunctionCallDict.name`](https://googleapis.github.io/python-genai/genai.html#genai.types.FunctionCallDict.name)
  - [`FunctionCallingConfig`](https://googleapis.github.io/python-genai/genai.html#genai.types.FunctionCallingConfig)
    - [`FunctionCallingConfig.allowed_function_names`](https://googleapis.github.io/python-genai/genai.html#genai.types.FunctionCallingConfig.allowed_function_names)
    - [`FunctionCallingConfig.mode`](https://googleapis.github.io/python-genai/genai.html#genai.types.FunctionCallingConfig.mode)
  - [`FunctionCallingConfigDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.FunctionCallingConfigDict)
    - [`FunctionCallingConfigDict.allowed_function_names`](https://googleapis.github.io/python-genai/genai.html#genai.types.FunctionCallingConfigDict.allowed_function_names)
    - [`FunctionCallingConfigDict.mode`](https://googleapis.github.io/python-genai/genai.html#genai.types.FunctionCallingConfigDict.mode)
  - [`FunctionCallingConfigMode`](https://googleapis.github.io/python-genai/genai.html#genai.types.FunctionCallingConfigMode)
    - [`FunctionCallingConfigMode.ANY`](https://googleapis.github.io/python-genai/genai.html#genai.types.FunctionCallingConfigMode.ANY)
    - [`FunctionCallingConfigMode.AUTO`](https://googleapis.github.io/python-genai/genai.html#genai.types.FunctionCallingConfigMode.AUTO)
    - [`FunctionCallingConfigMode.MODE_UNSPECIFIED`](https://googleapis.github.io/python-genai/genai.html#genai.types.FunctionCallingConfigMode.MODE_UNSPECIFIED)
    - [`FunctionCallingConfigMode.NONE`](https://googleapis.github.io/python-genai/genai.html#genai.types.FunctionCallingConfigMode.NONE)
  - [`FunctionDeclaration`](https://googleapis.github.io/python-genai/genai.html#genai.types.FunctionDeclaration)
    - [`FunctionDeclaration.behavior`](https://googleapis.github.io/python-genai/genai.html#genai.types.FunctionDeclaration.behavior)
    - [`FunctionDeclaration.description`](https://googleapis.github.io/python-genai/genai.html#genai.types.FunctionDeclaration.description)
    - [`FunctionDeclaration.name`](https://googleapis.github.io/python-genai/genai.html#genai.types.FunctionDeclaration.name)
    - [`FunctionDeclaration.parameters`](https://googleapis.github.io/python-genai/genai.html#genai.types.FunctionDeclaration.parameters)
    - [`FunctionDeclaration.parameters_json_schema`](https://googleapis.github.io/python-genai/genai.html#genai.types.FunctionDeclaration.parameters_json_schema)
    - [`FunctionDeclaration.response`](https://googleapis.github.io/python-genai/genai.html#genai.types.FunctionDeclaration.response)
    - [`FunctionDeclaration.response_json_schema`](https://googleapis.github.io/python-genai/genai.html#genai.types.FunctionDeclaration.response_json_schema)
    - [`FunctionDeclaration.from_callable()`](https://googleapis.github.io/python-genai/genai.html#genai.types.FunctionDeclaration.from_callable)
    - [`FunctionDeclaration.from_callable_with_api_option()`](https://googleapis.github.io/python-genai/genai.html#genai.types.FunctionDeclaration.from_callable_with_api_option)
  - [`FunctionDeclarationDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.FunctionDeclarationDict)
    - [`FunctionDeclarationDict.behavior`](https://googleapis.github.io/python-genai/genai.html#genai.types.FunctionDeclarationDict.behavior)
    - [`FunctionDeclarationDict.description`](https://googleapis.github.io/python-genai/genai.html#genai.types.FunctionDeclarationDict.description)
    - [`FunctionDeclarationDict.name`](https://googleapis.github.io/python-genai/genai.html#genai.types.FunctionDeclarationDict.name)
    - [`FunctionDeclarationDict.parameters`](https://googleapis.github.io/python-genai/genai.html#genai.types.FunctionDeclarationDict.parameters)
    - [`FunctionDeclarationDict.parameters_json_schema`](https://googleapis.github.io/python-genai/genai.html#genai.types.FunctionDeclarationDict.parameters_json_schema)
    - [`FunctionDeclarationDict.response`](https://googleapis.github.io/python-genai/genai.html#genai.types.FunctionDeclarationDict.response)
    - [`FunctionDeclarationDict.response_json_schema`](https://googleapis.github.io/python-genai/genai.html#genai.types.FunctionDeclarationDict.response_json_schema)
  - [`FunctionResponse`](https://googleapis.github.io/python-genai/genai.html#genai.types.FunctionResponse)
    - [`FunctionResponse.id`](https://googleapis.github.io/python-genai/genai.html#genai.types.FunctionResponse.id)
    - [`FunctionResponse.name`](https://googleapis.github.io/python-genai/genai.html#genai.types.FunctionResponse.name)
    - [`FunctionResponse.response`](https://googleapis.github.io/python-genai/genai.html#genai.types.FunctionResponse.response)
    - [`FunctionResponse.scheduling`](https://googleapis.github.io/python-genai/genai.html#genai.types.FunctionResponse.scheduling)
    - [`FunctionResponse.will_continue`](https://googleapis.github.io/python-genai/genai.html#genai.types.FunctionResponse.will_continue)
    - [`FunctionResponse.from_mcp_response()`](https://googleapis.github.io/python-genai/genai.html#genai.types.FunctionResponse.from_mcp_response)
  - [`FunctionResponseDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.FunctionResponseDict)
    - [`FunctionResponseDict.id`](https://googleapis.github.io/python-genai/genai.html#genai.types.FunctionResponseDict.id)
    - [`FunctionResponseDict.name`](https://googleapis.github.io/python-genai/genai.html#genai.types.FunctionResponseDict.name)
    - [`FunctionResponseDict.response`](https://googleapis.github.io/python-genai/genai.html#genai.types.FunctionResponseDict.response)
    - [`FunctionResponseDict.scheduling`](https://googleapis.github.io/python-genai/genai.html#genai.types.FunctionResponseDict.scheduling)
    - [`FunctionResponseDict.will_continue`](https://googleapis.github.io/python-genai/genai.html#genai.types.FunctionResponseDict.will_continue)
  - [`FunctionResponseScheduling`](https://googleapis.github.io/python-genai/genai.html#genai.types.FunctionResponseScheduling)
    - [`FunctionResponseScheduling.INTERRUPT`](https://googleapis.github.io/python-genai/genai.html#genai.types.FunctionResponseScheduling.INTERRUPT)
    - [`FunctionResponseScheduling.SCHEDULING_UNSPECIFIED`](https://googleapis.github.io/python-genai/genai.html#genai.types.FunctionResponseScheduling.SCHEDULING_UNSPECIFIED)
    - [`FunctionResponseScheduling.SILENT`](https://googleapis.github.io/python-genai/genai.html#genai.types.FunctionResponseScheduling.SILENT)
    - [`FunctionResponseScheduling.WHEN_IDLE`](https://googleapis.github.io/python-genai/genai.html#genai.types.FunctionResponseScheduling.WHEN_IDLE)
  - [`GenerateContentConfig`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateContentConfig)
    - [`GenerateContentConfig.audio_timestamp`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateContentConfig.audio_timestamp)
    - [`GenerateContentConfig.automatic_function_calling`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateContentConfig.automatic_function_calling)
    - [`GenerateContentConfig.cached_content`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateContentConfig.cached_content)
    - [`GenerateContentConfig.candidate_count`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateContentConfig.candidate_count)
    - [`GenerateContentConfig.frequency_penalty`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateContentConfig.frequency_penalty)
    - [`GenerateContentConfig.http_options`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateContentConfig.http_options)
    - [`GenerateContentConfig.labels`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateContentConfig.labels)
    - [`GenerateContentConfig.logprobs`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateContentConfig.logprobs)
    - [`GenerateContentConfig.max_output_tokens`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateContentConfig.max_output_tokens)
    - [`GenerateContentConfig.media_resolution`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateContentConfig.media_resolution)
    - [`GenerateContentConfig.model_selection_config`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateContentConfig.model_selection_config)
    - [`GenerateContentConfig.presence_penalty`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateContentConfig.presence_penalty)
    - [`GenerateContentConfig.response_logprobs`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateContentConfig.response_logprobs)
    - [`GenerateContentConfig.response_mime_type`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateContentConfig.response_mime_type)
    - [`GenerateContentConfig.response_modalities`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateContentConfig.response_modalities)
    - [`GenerateContentConfig.response_schema`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateContentConfig.response_schema)
    - [`GenerateContentConfig.routing_config`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateContentConfig.routing_config)
    - [`GenerateContentConfig.safety_settings`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateContentConfig.safety_settings)
    - [`GenerateContentConfig.seed`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateContentConfig.seed)
    - [`GenerateContentConfig.speech_config`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateContentConfig.speech_config)
    - [`GenerateContentConfig.stop_sequences`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateContentConfig.stop_sequences)
    - [`GenerateContentConfig.system_instruction`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateContentConfig.system_instruction)
    - [`GenerateContentConfig.temperature`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateContentConfig.temperature)
    - [`GenerateContentConfig.thinking_config`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateContentConfig.thinking_config)
    - [`GenerateContentConfig.tool_config`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateContentConfig.tool_config)
    - [`GenerateContentConfig.tools`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateContentConfig.tools)
    - [`GenerateContentConfig.top_k`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateContentConfig.top_k)
    - [`GenerateContentConfig.top_p`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateContentConfig.top_p)
  - [`GenerateContentConfigDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateContentConfigDict)
    - [`GenerateContentConfigDict.audio_timestamp`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateContentConfigDict.audio_timestamp)
    - [`GenerateContentConfigDict.automatic_function_calling`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateContentConfigDict.automatic_function_calling)
    - [`GenerateContentConfigDict.cached_content`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateContentConfigDict.cached_content)
    - [`GenerateContentConfigDict.candidate_count`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateContentConfigDict.candidate_count)
    - [`GenerateContentConfigDict.frequency_penalty`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateContentConfigDict.frequency_penalty)
    - [`GenerateContentConfigDict.http_options`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateContentConfigDict.http_options)
    - [`GenerateContentConfigDict.labels`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateContentConfigDict.labels)
    - [`GenerateContentConfigDict.logprobs`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateContentConfigDict.logprobs)
    - [`GenerateContentConfigDict.max_output_tokens`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateContentConfigDict.max_output_tokens)
    - [`GenerateContentConfigDict.media_resolution`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateContentConfigDict.media_resolution)
    - [`GenerateContentConfigDict.model_selection_config`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateContentConfigDict.model_selection_config)
    - [`GenerateContentConfigDict.presence_penalty`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateContentConfigDict.presence_penalty)
    - [`GenerateContentConfigDict.response_logprobs`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateContentConfigDict.response_logprobs)
    - [`GenerateContentConfigDict.response_mime_type`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateContentConfigDict.response_mime_type)
    - [`GenerateContentConfigDict.response_modalities`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateContentConfigDict.response_modalities)
    - [`GenerateContentConfigDict.response_schema`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateContentConfigDict.response_schema)
    - [`GenerateContentConfigDict.routing_config`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateContentConfigDict.routing_config)
    - [`GenerateContentConfigDict.safety_settings`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateContentConfigDict.safety_settings)
    - [`GenerateContentConfigDict.seed`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateContentConfigDict.seed)
    - [`GenerateContentConfigDict.speech_config`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateContentConfigDict.speech_config)
    - [`GenerateContentConfigDict.stop_sequences`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateContentConfigDict.stop_sequences)
    - [`GenerateContentConfigDict.system_instruction`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateContentConfigDict.system_instruction)
    - [`GenerateContentConfigDict.temperature`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateContentConfigDict.temperature)
    - [`GenerateContentConfigDict.thinking_config`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateContentConfigDict.thinking_config)
    - [`GenerateContentConfigDict.tool_config`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateContentConfigDict.tool_config)
    - [`GenerateContentConfigDict.tools`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateContentConfigDict.tools)
    - [`GenerateContentConfigDict.top_k`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateContentConfigDict.top_k)
    - [`GenerateContentConfigDict.top_p`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateContentConfigDict.top_p)
  - [`GenerateContentResponse`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateContentResponse)
    - [`GenerateContentResponse.automatic_function_calling_history`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateContentResponse.automatic_function_calling_history)
    - [`GenerateContentResponse.candidates`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateContentResponse.candidates)
    - [`GenerateContentResponse.create_time`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateContentResponse.create_time)
    - [`GenerateContentResponse.model_version`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateContentResponse.model_version)
    - [`GenerateContentResponse.parsed`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateContentResponse.parsed)
    - [`GenerateContentResponse.prompt_feedback`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateContentResponse.prompt_feedback)
    - [`GenerateContentResponse.response_id`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateContentResponse.response_id)
    - [`GenerateContentResponse.usage_metadata`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateContentResponse.usage_metadata)
    - [`GenerateContentResponse.code_execution_result`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateContentResponse.code_execution_result)
    - [`GenerateContentResponse.executable_code`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateContentResponse.executable_code)
    - [`GenerateContentResponse.function_calls`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateContentResponse.function_calls)
    - [`GenerateContentResponse.text`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateContentResponse.text)
  - [`GenerateContentResponseDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateContentResponseDict)
    - [`GenerateContentResponseDict.candidates`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateContentResponseDict.candidates)
    - [`GenerateContentResponseDict.create_time`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateContentResponseDict.create_time)
    - [`GenerateContentResponseDict.model_version`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateContentResponseDict.model_version)
    - [`GenerateContentResponseDict.prompt_feedback`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateContentResponseDict.prompt_feedback)
    - [`GenerateContentResponseDict.response_id`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateContentResponseDict.response_id)
    - [`GenerateContentResponseDict.usage_metadata`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateContentResponseDict.usage_metadata)
  - [`GenerateContentResponsePromptFeedback`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateContentResponsePromptFeedback)
    - [`GenerateContentResponsePromptFeedback.block_reason`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateContentResponsePromptFeedback.block_reason)
    - [`GenerateContentResponsePromptFeedback.block_reason_message`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateContentResponsePromptFeedback.block_reason_message)
    - [`GenerateContentResponsePromptFeedback.safety_ratings`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateContentResponsePromptFeedback.safety_ratings)
  - [`GenerateContentResponsePromptFeedbackDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateContentResponsePromptFeedbackDict)
    - [`GenerateContentResponsePromptFeedbackDict.block_reason`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateContentResponsePromptFeedbackDict.block_reason)
    - [`GenerateContentResponsePromptFeedbackDict.block_reason_message`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateContentResponsePromptFeedbackDict.block_reason_message)
    - [`GenerateContentResponsePromptFeedbackDict.safety_ratings`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateContentResponsePromptFeedbackDict.safety_ratings)
  - [`GenerateContentResponseUsageMetadata`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateContentResponseUsageMetadata)
    - [`GenerateContentResponseUsageMetadata.cache_tokens_details`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateContentResponseUsageMetadata.cache_tokens_details)
    - [`GenerateContentResponseUsageMetadata.cached_content_token_count`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateContentResponseUsageMetadata.cached_content_token_count)
    - [`GenerateContentResponseUsageMetadata.candidates_token_count`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateContentResponseUsageMetadata.candidates_token_count)
    - [`GenerateContentResponseUsageMetadata.candidates_tokens_details`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateContentResponseUsageMetadata.candidates_tokens_details)
    - [`GenerateContentResponseUsageMetadata.prompt_token_count`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateContentResponseUsageMetadata.prompt_token_count)
    - [`GenerateContentResponseUsageMetadata.prompt_tokens_details`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateContentResponseUsageMetadata.prompt_tokens_details)
    - [`GenerateContentResponseUsageMetadata.thoughts_token_count`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateContentResponseUsageMetadata.thoughts_token_count)
    - [`GenerateContentResponseUsageMetadata.tool_use_prompt_token_count`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateContentResponseUsageMetadata.tool_use_prompt_token_count)
    - [`GenerateContentResponseUsageMetadata.tool_use_prompt_tokens_details`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateContentResponseUsageMetadata.tool_use_prompt_tokens_details)
    - [`GenerateContentResponseUsageMetadata.total_token_count`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateContentResponseUsageMetadata.total_token_count)
    - [`GenerateContentResponseUsageMetadata.traffic_type`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateContentResponseUsageMetadata.traffic_type)
  - [`GenerateContentResponseUsageMetadataDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateContentResponseUsageMetadataDict)
    - [`GenerateContentResponseUsageMetadataDict.cache_tokens_details`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateContentResponseUsageMetadataDict.cache_tokens_details)
    - [`GenerateContentResponseUsageMetadataDict.cached_content_token_count`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateContentResponseUsageMetadataDict.cached_content_token_count)
    - [`GenerateContentResponseUsageMetadataDict.candidates_token_count`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateContentResponseUsageMetadataDict.candidates_token_count)
    - [`GenerateContentResponseUsageMetadataDict.candidates_tokens_details`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateContentResponseUsageMetadataDict.candidates_tokens_details)
    - [`GenerateContentResponseUsageMetadataDict.prompt_token_count`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateContentResponseUsageMetadataDict.prompt_token_count)
    - [`GenerateContentResponseUsageMetadataDict.prompt_tokens_details`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateContentResponseUsageMetadataDict.prompt_tokens_details)
    - [`GenerateContentResponseUsageMetadataDict.thoughts_token_count`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateContentResponseUsageMetadataDict.thoughts_token_count)
    - [`GenerateContentResponseUsageMetadataDict.tool_use_prompt_token_count`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateContentResponseUsageMetadataDict.tool_use_prompt_token_count)
    - [`GenerateContentResponseUsageMetadataDict.tool_use_prompt_tokens_details`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateContentResponseUsageMetadataDict.tool_use_prompt_tokens_details)
    - [`GenerateContentResponseUsageMetadataDict.total_token_count`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateContentResponseUsageMetadataDict.total_token_count)
    - [`GenerateContentResponseUsageMetadataDict.traffic_type`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateContentResponseUsageMetadataDict.traffic_type)
  - [`GenerateImagesConfig`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateImagesConfig)
    - [`GenerateImagesConfig.add_watermark`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateImagesConfig.add_watermark)
    - [`GenerateImagesConfig.aspect_ratio`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateImagesConfig.aspect_ratio)
    - [`GenerateImagesConfig.enhance_prompt`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateImagesConfig.enhance_prompt)
    - [`GenerateImagesConfig.guidance_scale`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateImagesConfig.guidance_scale)
    - [`GenerateImagesConfig.http_options`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateImagesConfig.http_options)
    - [`GenerateImagesConfig.include_rai_reason`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateImagesConfig.include_rai_reason)
    - [`GenerateImagesConfig.include_safety_attributes`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateImagesConfig.include_safety_attributes)
    - [`GenerateImagesConfig.language`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateImagesConfig.language)
    - [`GenerateImagesConfig.negative_prompt`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateImagesConfig.negative_prompt)
    - [`GenerateImagesConfig.number_of_images`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateImagesConfig.number_of_images)
    - [`GenerateImagesConfig.output_compression_quality`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateImagesConfig.output_compression_quality)
    - [`GenerateImagesConfig.output_gcs_uri`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateImagesConfig.output_gcs_uri)
    - [`GenerateImagesConfig.output_mime_type`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateImagesConfig.output_mime_type)
    - [`GenerateImagesConfig.person_generation`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateImagesConfig.person_generation)
    - [`GenerateImagesConfig.safety_filter_level`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateImagesConfig.safety_filter_level)
    - [`GenerateImagesConfig.seed`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateImagesConfig.seed)
  - [`GenerateImagesConfigDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateImagesConfigDict)
    - [`GenerateImagesConfigDict.add_watermark`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateImagesConfigDict.add_watermark)
    - [`GenerateImagesConfigDict.aspect_ratio`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateImagesConfigDict.aspect_ratio)
    - [`GenerateImagesConfigDict.enhance_prompt`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateImagesConfigDict.enhance_prompt)
    - [`GenerateImagesConfigDict.guidance_scale`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateImagesConfigDict.guidance_scale)
    - [`GenerateImagesConfigDict.http_options`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateImagesConfigDict.http_options)
    - [`GenerateImagesConfigDict.include_rai_reason`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateImagesConfigDict.include_rai_reason)
    - [`GenerateImagesConfigDict.include_safety_attributes`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateImagesConfigDict.include_safety_attributes)
    - [`GenerateImagesConfigDict.language`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateImagesConfigDict.language)
    - [`GenerateImagesConfigDict.negative_prompt`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateImagesConfigDict.negative_prompt)
    - [`GenerateImagesConfigDict.number_of_images`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateImagesConfigDict.number_of_images)
    - [`GenerateImagesConfigDict.output_compression_quality`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateImagesConfigDict.output_compression_quality)
    - [`GenerateImagesConfigDict.output_gcs_uri`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateImagesConfigDict.output_gcs_uri)
    - [`GenerateImagesConfigDict.output_mime_type`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateImagesConfigDict.output_mime_type)
    - [`GenerateImagesConfigDict.person_generation`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateImagesConfigDict.person_generation)
    - [`GenerateImagesConfigDict.safety_filter_level`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateImagesConfigDict.safety_filter_level)
    - [`GenerateImagesConfigDict.seed`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateImagesConfigDict.seed)
  - [`GenerateImagesResponse`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateImagesResponse)
    - [`GenerateImagesResponse.generated_images`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateImagesResponse.generated_images)
    - [`GenerateImagesResponse.positive_prompt_safety_attributes`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateImagesResponse.positive_prompt_safety_attributes)
  - [`GenerateImagesResponseDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateImagesResponseDict)
    - [`GenerateImagesResponseDict.generated_images`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateImagesResponseDict.generated_images)
    - [`GenerateImagesResponseDict.positive_prompt_safety_attributes`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateImagesResponseDict.positive_prompt_safety_attributes)
  - [`GenerateVideosConfig`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateVideosConfig)
    - [`GenerateVideosConfig.aspect_ratio`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateVideosConfig.aspect_ratio)
    - [`GenerateVideosConfig.duration_seconds`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateVideosConfig.duration_seconds)
    - [`GenerateVideosConfig.enhance_prompt`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateVideosConfig.enhance_prompt)
    - [`GenerateVideosConfig.fps`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateVideosConfig.fps)
    - [`GenerateVideosConfig.generate_audio`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateVideosConfig.generate_audio)
    - [`GenerateVideosConfig.http_options`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateVideosConfig.http_options)
    - [`GenerateVideosConfig.last_frame`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateVideosConfig.last_frame)
    - [`GenerateVideosConfig.negative_prompt`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateVideosConfig.negative_prompt)
    - [`GenerateVideosConfig.number_of_videos`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateVideosConfig.number_of_videos)
    - [`GenerateVideosConfig.output_gcs_uri`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateVideosConfig.output_gcs_uri)
    - [`GenerateVideosConfig.person_generation`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateVideosConfig.person_generation)
    - [`GenerateVideosConfig.pubsub_topic`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateVideosConfig.pubsub_topic)
    - [`GenerateVideosConfig.resolution`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateVideosConfig.resolution)
    - [`GenerateVideosConfig.seed`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateVideosConfig.seed)
  - [`GenerateVideosConfigDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateVideosConfigDict)
    - [`GenerateVideosConfigDict.aspect_ratio`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateVideosConfigDict.aspect_ratio)
    - [`GenerateVideosConfigDict.duration_seconds`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateVideosConfigDict.duration_seconds)
    - [`GenerateVideosConfigDict.enhance_prompt`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateVideosConfigDict.enhance_prompt)
    - [`GenerateVideosConfigDict.fps`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateVideosConfigDict.fps)
    - [`GenerateVideosConfigDict.generate_audio`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateVideosConfigDict.generate_audio)
    - [`GenerateVideosConfigDict.http_options`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateVideosConfigDict.http_options)
    - [`GenerateVideosConfigDict.last_frame`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateVideosConfigDict.last_frame)
    - [`GenerateVideosConfigDict.negative_prompt`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateVideosConfigDict.negative_prompt)
    - [`GenerateVideosConfigDict.number_of_videos`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateVideosConfigDict.number_of_videos)
    - [`GenerateVideosConfigDict.output_gcs_uri`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateVideosConfigDict.output_gcs_uri)
    - [`GenerateVideosConfigDict.person_generation`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateVideosConfigDict.person_generation)
    - [`GenerateVideosConfigDict.pubsub_topic`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateVideosConfigDict.pubsub_topic)
    - [`GenerateVideosConfigDict.resolution`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateVideosConfigDict.resolution)
    - [`GenerateVideosConfigDict.seed`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateVideosConfigDict.seed)
  - [`GenerateVideosOperation`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateVideosOperation)
    - [`GenerateVideosOperation.done`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateVideosOperation.done)
    - [`GenerateVideosOperation.error`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateVideosOperation.error)
    - [`GenerateVideosOperation.metadata`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateVideosOperation.metadata)
    - [`GenerateVideosOperation.name`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateVideosOperation.name)
    - [`GenerateVideosOperation.response`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateVideosOperation.response)
    - [`GenerateVideosOperation.result`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateVideosOperation.result)
  - [`GenerateVideosOperationDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateVideosOperationDict)
    - [`GenerateVideosOperationDict.done`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateVideosOperationDict.done)
    - [`GenerateVideosOperationDict.error`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateVideosOperationDict.error)
    - [`GenerateVideosOperationDict.metadata`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateVideosOperationDict.metadata)
    - [`GenerateVideosOperationDict.name`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateVideosOperationDict.name)
    - [`GenerateVideosOperationDict.response`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateVideosOperationDict.response)
    - [`GenerateVideosOperationDict.result`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateVideosOperationDict.result)
  - [`GenerateVideosResponse`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateVideosResponse)
    - [`GenerateVideosResponse.generated_videos`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateVideosResponse.generated_videos)
    - [`GenerateVideosResponse.rai_media_filtered_count`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateVideosResponse.rai_media_filtered_count)
    - [`GenerateVideosResponse.rai_media_filtered_reasons`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateVideosResponse.rai_media_filtered_reasons)
  - [`GenerateVideosResponseDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateVideosResponseDict)
    - [`GenerateVideosResponseDict.generated_videos`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateVideosResponseDict.generated_videos)
    - [`GenerateVideosResponseDict.rai_media_filtered_count`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateVideosResponseDict.rai_media_filtered_count)
    - [`GenerateVideosResponseDict.rai_media_filtered_reasons`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateVideosResponseDict.rai_media_filtered_reasons)
  - [`GeneratedImage`](https://googleapis.github.io/python-genai/genai.html#genai.types.GeneratedImage)
    - [`GeneratedImage.enhanced_prompt`](https://googleapis.github.io/python-genai/genai.html#genai.types.GeneratedImage.enhanced_prompt)
    - [`GeneratedImage.image`](https://googleapis.github.io/python-genai/genai.html#genai.types.GeneratedImage.image)
    - [`GeneratedImage.rai_filtered_reason`](https://googleapis.github.io/python-genai/genai.html#genai.types.GeneratedImage.rai_filtered_reason)
    - [`GeneratedImage.safety_attributes`](https://googleapis.github.io/python-genai/genai.html#genai.types.GeneratedImage.safety_attributes)
  - [`GeneratedImageDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.GeneratedImageDict)
    - [`GeneratedImageDict.enhanced_prompt`](https://googleapis.github.io/python-genai/genai.html#genai.types.GeneratedImageDict.enhanced_prompt)
    - [`GeneratedImageDict.image`](https://googleapis.github.io/python-genai/genai.html#genai.types.GeneratedImageDict.image)
    - [`GeneratedImageDict.rai_filtered_reason`](https://googleapis.github.io/python-genai/genai.html#genai.types.GeneratedImageDict.rai_filtered_reason)
    - [`GeneratedImageDict.safety_attributes`](https://googleapis.github.io/python-genai/genai.html#genai.types.GeneratedImageDict.safety_attributes)
  - [`GeneratedVideo`](https://googleapis.github.io/python-genai/genai.html#genai.types.GeneratedVideo)
    - [`GeneratedVideo.video`](https://googleapis.github.io/python-genai/genai.html#genai.types.GeneratedVideo.video)
  - [`GeneratedVideoDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.GeneratedVideoDict)
    - [`GeneratedVideoDict.video`](https://googleapis.github.io/python-genai/genai.html#genai.types.GeneratedVideoDict.video)
  - [`GenerationConfig`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerationConfig)
    - [`GenerationConfig.audio_timestamp`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerationConfig.audio_timestamp)
    - [`GenerationConfig.candidate_count`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerationConfig.candidate_count)
    - [`GenerationConfig.frequency_penalty`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerationConfig.frequency_penalty)
    - [`GenerationConfig.logprobs`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerationConfig.logprobs)
    - [`GenerationConfig.max_output_tokens`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerationConfig.max_output_tokens)
    - [`GenerationConfig.media_resolution`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerationConfig.media_resolution)
    - [`GenerationConfig.model_selection_config`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerationConfig.model_selection_config)
    - [`GenerationConfig.presence_penalty`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerationConfig.presence_penalty)
    - [`GenerationConfig.response_json_schema`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerationConfig.response_json_schema)
    - [`GenerationConfig.response_logprobs`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerationConfig.response_logprobs)
    - [`GenerationConfig.response_mime_type`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerationConfig.response_mime_type)
    - [`GenerationConfig.response_modalities`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerationConfig.response_modalities)
    - [`GenerationConfig.response_schema`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerationConfig.response_schema)
    - [`GenerationConfig.routing_config`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerationConfig.routing_config)
    - [`GenerationConfig.seed`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerationConfig.seed)
    - [`GenerationConfig.speech_config`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerationConfig.speech_config)
    - [`GenerationConfig.stop_sequences`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerationConfig.stop_sequences)
    - [`GenerationConfig.temperature`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerationConfig.temperature)
    - [`GenerationConfig.thinking_config`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerationConfig.thinking_config)
    - [`GenerationConfig.top_k`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerationConfig.top_k)
    - [`GenerationConfig.top_p`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerationConfig.top_p)
  - [`GenerationConfigDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerationConfigDict)
    - [`GenerationConfigDict.audio_timestamp`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerationConfigDict.audio_timestamp)
    - [`GenerationConfigDict.candidate_count`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerationConfigDict.candidate_count)
    - [`GenerationConfigDict.frequency_penalty`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerationConfigDict.frequency_penalty)
    - [`GenerationConfigDict.logprobs`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerationConfigDict.logprobs)
    - [`GenerationConfigDict.max_output_tokens`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerationConfigDict.max_output_tokens)
    - [`GenerationConfigDict.media_resolution`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerationConfigDict.media_resolution)
    - [`GenerationConfigDict.model_selection_config`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerationConfigDict.model_selection_config)
    - [`GenerationConfigDict.presence_penalty`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerationConfigDict.presence_penalty)
    - [`GenerationConfigDict.response_json_schema`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerationConfigDict.response_json_schema)
    - [`GenerationConfigDict.response_logprobs`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerationConfigDict.response_logprobs)
    - [`GenerationConfigDict.response_mime_type`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerationConfigDict.response_mime_type)
    - [`GenerationConfigDict.response_modalities`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerationConfigDict.response_modalities)
    - [`GenerationConfigDict.response_schema`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerationConfigDict.response_schema)
    - [`GenerationConfigDict.routing_config`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerationConfigDict.routing_config)
    - [`GenerationConfigDict.seed`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerationConfigDict.seed)
    - [`GenerationConfigDict.speech_config`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerationConfigDict.speech_config)
    - [`GenerationConfigDict.stop_sequences`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerationConfigDict.stop_sequences)
    - [`GenerationConfigDict.temperature`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerationConfigDict.temperature)
    - [`GenerationConfigDict.thinking_config`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerationConfigDict.thinking_config)
    - [`GenerationConfigDict.top_k`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerationConfigDict.top_k)
    - [`GenerationConfigDict.top_p`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerationConfigDict.top_p)
  - [`GenerationConfigRoutingConfig`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerationConfigRoutingConfig)
    - [`GenerationConfigRoutingConfig.auto_mode`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerationConfigRoutingConfig.auto_mode)
    - [`GenerationConfigRoutingConfig.manual_mode`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerationConfigRoutingConfig.manual_mode)
  - [`GenerationConfigRoutingConfigAutoRoutingMode`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerationConfigRoutingConfigAutoRoutingMode)
    - [`GenerationConfigRoutingConfigAutoRoutingMode.model_routing_preference`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerationConfigRoutingConfigAutoRoutingMode.model_routing_preference)
  - [`GenerationConfigRoutingConfigAutoRoutingModeDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerationConfigRoutingConfigAutoRoutingModeDict)
    - [`GenerationConfigRoutingConfigAutoRoutingModeDict.model_routing_preference`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerationConfigRoutingConfigAutoRoutingModeDict.model_routing_preference)
  - [`GenerationConfigRoutingConfigDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerationConfigRoutingConfigDict)
    - [`GenerationConfigRoutingConfigDict.auto_mode`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerationConfigRoutingConfigDict.auto_mode)
    - [`GenerationConfigRoutingConfigDict.manual_mode`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerationConfigRoutingConfigDict.manual_mode)
  - [`GenerationConfigRoutingConfigManualRoutingMode`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerationConfigRoutingConfigManualRoutingMode)
    - [`GenerationConfigRoutingConfigManualRoutingMode.model_name`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerationConfigRoutingConfigManualRoutingMode.model_name)
  - [`GenerationConfigRoutingConfigManualRoutingModeDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerationConfigRoutingConfigManualRoutingModeDict)
    - [`GenerationConfigRoutingConfigManualRoutingModeDict.model_name`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerationConfigRoutingConfigManualRoutingModeDict.model_name)
  - [`GenerationConfigThinkingConfig`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerationConfigThinkingConfig)
    - [`GenerationConfigThinkingConfig.include_thoughts`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerationConfigThinkingConfig.include_thoughts)
    - [`GenerationConfigThinkingConfig.thinking_budget`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerationConfigThinkingConfig.thinking_budget)
  - [`GenerationConfigThinkingConfigDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerationConfigThinkingConfigDict)
    - [`GenerationConfigThinkingConfigDict.include_thoughts`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerationConfigThinkingConfigDict.include_thoughts)
    - [`GenerationConfigThinkingConfigDict.thinking_budget`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerationConfigThinkingConfigDict.thinking_budget)
  - [`GetBatchJobConfig`](https://googleapis.github.io/python-genai/genai.html#genai.types.GetBatchJobConfig)
    - [`GetBatchJobConfig.http_options`](https://googleapis.github.io/python-genai/genai.html#genai.types.GetBatchJobConfig.http_options)
  - [`GetBatchJobConfigDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.GetBatchJobConfigDict)
    - [`GetBatchJobConfigDict.http_options`](https://googleapis.github.io/python-genai/genai.html#genai.types.GetBatchJobConfigDict.http_options)
  - [`GetCachedContentConfig`](https://googleapis.github.io/python-genai/genai.html#genai.types.GetCachedContentConfig)
    - [`GetCachedContentConfig.http_options`](https://googleapis.github.io/python-genai/genai.html#genai.types.GetCachedContentConfig.http_options)
  - [`GetCachedContentConfigDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.GetCachedContentConfigDict)
    - [`GetCachedContentConfigDict.http_options`](https://googleapis.github.io/python-genai/genai.html#genai.types.GetCachedContentConfigDict.http_options)
  - [`GetFileConfig`](https://googleapis.github.io/python-genai/genai.html#genai.types.GetFileConfig)
    - [`GetFileConfig.http_options`](https://googleapis.github.io/python-genai/genai.html#genai.types.GetFileConfig.http_options)
  - [`GetFileConfigDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.GetFileConfigDict)
    - [`GetFileConfigDict.http_options`](https://googleapis.github.io/python-genai/genai.html#genai.types.GetFileConfigDict.http_options)
  - [`GetModelConfig`](https://googleapis.github.io/python-genai/genai.html#genai.types.GetModelConfig)
    - [`GetModelConfig.http_options`](https://googleapis.github.io/python-genai/genai.html#genai.types.GetModelConfig.http_options)
  - [`GetModelConfigDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.GetModelConfigDict)
    - [`GetModelConfigDict.http_options`](https://googleapis.github.io/python-genai/genai.html#genai.types.GetModelConfigDict.http_options)
  - [`GetOperationConfig`](https://googleapis.github.io/python-genai/genai.html#genai.types.GetOperationConfig)
    - [`GetOperationConfig.http_options`](https://googleapis.github.io/python-genai/genai.html#genai.types.GetOperationConfig.http_options)
  - [`GetOperationConfigDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.GetOperationConfigDict)
    - [`GetOperationConfigDict.http_options`](https://googleapis.github.io/python-genai/genai.html#genai.types.GetOperationConfigDict.http_options)
  - [`GetTuningJobConfig`](https://googleapis.github.io/python-genai/genai.html#genai.types.GetTuningJobConfig)
    - [`GetTuningJobConfig.http_options`](https://googleapis.github.io/python-genai/genai.html#genai.types.GetTuningJobConfig.http_options)
  - [`GetTuningJobConfigDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.GetTuningJobConfigDict)
    - [`GetTuningJobConfigDict.http_options`](https://googleapis.github.io/python-genai/genai.html#genai.types.GetTuningJobConfigDict.http_options)
  - [`GoogleMaps`](https://googleapis.github.io/python-genai/genai.html#genai.types.GoogleMaps)
    - [`GoogleMaps.auth_config`](https://googleapis.github.io/python-genai/genai.html#genai.types.GoogleMaps.auth_config)
  - [`GoogleMapsDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.GoogleMapsDict)
    - [`GoogleMapsDict.auth_config`](https://googleapis.github.io/python-genai/genai.html#genai.types.GoogleMapsDict.auth_config)
  - [`GoogleRpcStatus`](https://googleapis.github.io/python-genai/genai.html#genai.types.GoogleRpcStatus)
    - [`GoogleRpcStatus.code`](https://googleapis.github.io/python-genai/genai.html#genai.types.GoogleRpcStatus.code)
    - [`GoogleRpcStatus.details`](https://googleapis.github.io/python-genai/genai.html#genai.types.GoogleRpcStatus.details)
    - [`GoogleRpcStatus.message`](https://googleapis.github.io/python-genai/genai.html#genai.types.GoogleRpcStatus.message)
  - [`GoogleRpcStatusDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.GoogleRpcStatusDict)
    - [`GoogleRpcStatusDict.code`](https://googleapis.github.io/python-genai/genai.html#genai.types.GoogleRpcStatusDict.code)
    - [`GoogleRpcStatusDict.details`](https://googleapis.github.io/python-genai/genai.html#genai.types.GoogleRpcStatusDict.details)
    - [`GoogleRpcStatusDict.message`](https://googleapis.github.io/python-genai/genai.html#genai.types.GoogleRpcStatusDict.message)
  - [`GoogleSearch`](https://googleapis.github.io/python-genai/genai.html#genai.types.GoogleSearch)
    - [`GoogleSearch.time_range_filter`](https://googleapis.github.io/python-genai/genai.html#genai.types.GoogleSearch.time_range_filter)
  - [`GoogleSearchDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.GoogleSearchDict)
    - [`GoogleSearchDict.time_range_filter`](https://googleapis.github.io/python-genai/genai.html#genai.types.GoogleSearchDict.time_range_filter)
  - [`GoogleSearchRetrieval`](https://googleapis.github.io/python-genai/genai.html#genai.types.GoogleSearchRetrieval)
    - [`GoogleSearchRetrieval.dynamic_retrieval_config`](https://googleapis.github.io/python-genai/genai.html#genai.types.GoogleSearchRetrieval.dynamic_retrieval_config)
  - [`GoogleSearchRetrievalDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.GoogleSearchRetrievalDict)
    - [`GoogleSearchRetrievalDict.dynamic_retrieval_config`](https://googleapis.github.io/python-genai/genai.html#genai.types.GoogleSearchRetrievalDict.dynamic_retrieval_config)
  - [`GoogleTypeDate`](https://googleapis.github.io/python-genai/genai.html#genai.types.GoogleTypeDate)
    - [`GoogleTypeDate.day`](https://googleapis.github.io/python-genai/genai.html#genai.types.GoogleTypeDate.day)
    - [`GoogleTypeDate.month`](https://googleapis.github.io/python-genai/genai.html#genai.types.GoogleTypeDate.month)
    - [`GoogleTypeDate.year`](https://googleapis.github.io/python-genai/genai.html#genai.types.GoogleTypeDate.year)
  - [`GoogleTypeDateDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.GoogleTypeDateDict)
    - [`GoogleTypeDateDict.day`](https://googleapis.github.io/python-genai/genai.html#genai.types.GoogleTypeDateDict.day)
    - [`GoogleTypeDateDict.month`](https://googleapis.github.io/python-genai/genai.html#genai.types.GoogleTypeDateDict.month)
    - [`GoogleTypeDateDict.year`](https://googleapis.github.io/python-genai/genai.html#genai.types.GoogleTypeDateDict.year)
  - [`GroundingChunk`](https://googleapis.github.io/python-genai/genai.html#genai.types.GroundingChunk)
    - [`GroundingChunk.retrieved_context`](https://googleapis.github.io/python-genai/genai.html#genai.types.GroundingChunk.retrieved_context)
    - [`GroundingChunk.web`](https://googleapis.github.io/python-genai/genai.html#genai.types.GroundingChunk.web)
  - [`GroundingChunkDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.GroundingChunkDict)
    - [`GroundingChunkDict.retrieved_context`](https://googleapis.github.io/python-genai/genai.html#genai.types.GroundingChunkDict.retrieved_context)
    - [`GroundingChunkDict.web`](https://googleapis.github.io/python-genai/genai.html#genai.types.GroundingChunkDict.web)
  - [`GroundingChunkRetrievedContext`](https://googleapis.github.io/python-genai/genai.html#genai.types.GroundingChunkRetrievedContext)
    - [`GroundingChunkRetrievedContext.rag_chunk`](https://googleapis.github.io/python-genai/genai.html#genai.types.GroundingChunkRetrievedContext.rag_chunk)
    - [`GroundingChunkRetrievedContext.text`](https://googleapis.github.io/python-genai/genai.html#genai.types.GroundingChunkRetrievedContext.text)
    - [`GroundingChunkRetrievedContext.title`](https://googleapis.github.io/python-genai/genai.html#genai.types.GroundingChunkRetrievedContext.title)
    - [`GroundingChunkRetrievedContext.uri`](https://googleapis.github.io/python-genai/genai.html#genai.types.GroundingChunkRetrievedContext.uri)
  - [`GroundingChunkRetrievedContextDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.GroundingChunkRetrievedContextDict)
    - [`GroundingChunkRetrievedContextDict.rag_chunk`](https://googleapis.github.io/python-genai/genai.html#genai.types.GroundingChunkRetrievedContextDict.rag_chunk)
    - [`GroundingChunkRetrievedContextDict.text`](https://googleapis.github.io/python-genai/genai.html#genai.types.GroundingChunkRetrievedContextDict.text)
    - [`GroundingChunkRetrievedContextDict.title`](https://googleapis.github.io/python-genai/genai.html#genai.types.GroundingChunkRetrievedContextDict.title)
    - [`GroundingChunkRetrievedContextDict.uri`](https://googleapis.github.io/python-genai/genai.html#genai.types.GroundingChunkRetrievedContextDict.uri)
  - [`GroundingChunkWeb`](https://googleapis.github.io/python-genai/genai.html#genai.types.GroundingChunkWeb)
    - [`GroundingChunkWeb.domain`](https://googleapis.github.io/python-genai/genai.html#genai.types.GroundingChunkWeb.domain)
    - [`GroundingChunkWeb.title`](https://googleapis.github.io/python-genai/genai.html#genai.types.GroundingChunkWeb.title)
    - [`GroundingChunkWeb.uri`](https://googleapis.github.io/python-genai/genai.html#genai.types.GroundingChunkWeb.uri)
  - [`GroundingChunkWebDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.GroundingChunkWebDict)
    - [`GroundingChunkWebDict.domain`](https://googleapis.github.io/python-genai/genai.html#genai.types.GroundingChunkWebDict.domain)
    - [`GroundingChunkWebDict.title`](https://googleapis.github.io/python-genai/genai.html#genai.types.GroundingChunkWebDict.title)
    - [`GroundingChunkWebDict.uri`](https://googleapis.github.io/python-genai/genai.html#genai.types.GroundingChunkWebDict.uri)
  - [`GroundingMetadata`](https://googleapis.github.io/python-genai/genai.html#genai.types.GroundingMetadata)
    - [`GroundingMetadata.grounding_chunks`](https://googleapis.github.io/python-genai/genai.html#genai.types.GroundingMetadata.grounding_chunks)
    - [`GroundingMetadata.grounding_supports`](https://googleapis.github.io/python-genai/genai.html#genai.types.GroundingMetadata.grounding_supports)
    - [`GroundingMetadata.retrieval_metadata`](https://googleapis.github.io/python-genai/genai.html#genai.types.GroundingMetadata.retrieval_metadata)
    - [`GroundingMetadata.retrieval_queries`](https://googleapis.github.io/python-genai/genai.html#genai.types.GroundingMetadata.retrieval_queries)
    - [`GroundingMetadata.search_entry_point`](https://googleapis.github.io/python-genai/genai.html#genai.types.GroundingMetadata.search_entry_point)
    - [`GroundingMetadata.web_search_queries`](https://googleapis.github.io/python-genai/genai.html#genai.types.GroundingMetadata.web_search_queries)
  - [`GroundingMetadataDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.GroundingMetadataDict)
    - [`GroundingMetadataDict.grounding_chunks`](https://googleapis.github.io/python-genai/genai.html#genai.types.GroundingMetadataDict.grounding_chunks)
    - [`GroundingMetadataDict.grounding_supports`](https://googleapis.github.io/python-genai/genai.html#genai.types.GroundingMetadataDict.grounding_supports)
    - [`GroundingMetadataDict.retrieval_metadata`](https://googleapis.github.io/python-genai/genai.html#genai.types.GroundingMetadataDict.retrieval_metadata)
    - [`GroundingMetadataDict.retrieval_queries`](https://googleapis.github.io/python-genai/genai.html#genai.types.GroundingMetadataDict.retrieval_queries)
    - [`GroundingMetadataDict.search_entry_point`](https://googleapis.github.io/python-genai/genai.html#genai.types.GroundingMetadataDict.search_entry_point)
    - [`GroundingMetadataDict.web_search_queries`](https://googleapis.github.io/python-genai/genai.html#genai.types.GroundingMetadataDict.web_search_queries)
  - [`GroundingSupport`](https://googleapis.github.io/python-genai/genai.html#genai.types.GroundingSupport)
    - [`GroundingSupport.confidence_scores`](https://googleapis.github.io/python-genai/genai.html#genai.types.GroundingSupport.confidence_scores)
    - [`GroundingSupport.grounding_chunk_indices`](https://googleapis.github.io/python-genai/genai.html#genai.types.GroundingSupport.grounding_chunk_indices)
    - [`GroundingSupport.segment`](https://googleapis.github.io/python-genai/genai.html#genai.types.GroundingSupport.segment)
  - [`GroundingSupportDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.GroundingSupportDict)
    - [`GroundingSupportDict.confidence_scores`](https://googleapis.github.io/python-genai/genai.html#genai.types.GroundingSupportDict.confidence_scores)
    - [`GroundingSupportDict.grounding_chunk_indices`](https://googleapis.github.io/python-genai/genai.html#genai.types.GroundingSupportDict.grounding_chunk_indices)
    - [`GroundingSupportDict.segment`](https://googleapis.github.io/python-genai/genai.html#genai.types.GroundingSupportDict.segment)
  - [`HarmBlockMethod`](https://googleapis.github.io/python-genai/genai.html#genai.types.HarmBlockMethod)
    - [`HarmBlockMethod.HARM_BLOCK_METHOD_UNSPECIFIED`](https://googleapis.github.io/python-genai/genai.html#genai.types.HarmBlockMethod.HARM_BLOCK_METHOD_UNSPECIFIED)
    - [`HarmBlockMethod.PROBABILITY`](https://googleapis.github.io/python-genai/genai.html#genai.types.HarmBlockMethod.PROBABILITY)
    - [`HarmBlockMethod.SEVERITY`](https://googleapis.github.io/python-genai/genai.html#genai.types.HarmBlockMethod.SEVERITY)
  - [`HarmBlockThreshold`](https://googleapis.github.io/python-genai/genai.html#genai.types.HarmBlockThreshold)
    - [`HarmBlockThreshold.BLOCK_LOW_AND_ABOVE`](https://googleapis.github.io/python-genai/genai.html#genai.types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE)
    - [`HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE`](https://googleapis.github.io/python-genai/genai.html#genai.types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE)
    - [`HarmBlockThreshold.BLOCK_NONE`](https://googleapis.github.io/python-genai/genai.html#genai.types.HarmBlockThreshold.BLOCK_NONE)
    - [`HarmBlockThreshold.BLOCK_ONLY_HIGH`](https://googleapis.github.io/python-genai/genai.html#genai.types.HarmBlockThreshold.BLOCK_ONLY_HIGH)
    - [`HarmBlockThreshold.HARM_BLOCK_THRESHOLD_UNSPECIFIED`](https://googleapis.github.io/python-genai/genai.html#genai.types.HarmBlockThreshold.HARM_BLOCK_THRESHOLD_UNSPECIFIED)
    - [`HarmBlockThreshold.OFF`](https://googleapis.github.io/python-genai/genai.html#genai.types.HarmBlockThreshold.OFF)
  - [`HarmCategory`](https://googleapis.github.io/python-genai/genai.html#genai.types.HarmCategory)
    - [`HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY`](https://googleapis.github.io/python-genai/genai.html#genai.types.HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY)
    - [`HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT`](https://googleapis.github.io/python-genai/genai.html#genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT)
    - [`HarmCategory.HARM_CATEGORY_HARASSMENT`](https://googleapis.github.io/python-genai/genai.html#genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT)
    - [`HarmCategory.HARM_CATEGORY_HATE_SPEECH`](https://googleapis.github.io/python-genai/genai.html#genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH)
    - [`HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT`](https://googleapis.github.io/python-genai/genai.html#genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT)
    - [`HarmCategory.HARM_CATEGORY_UNSPECIFIED`](https://googleapis.github.io/python-genai/genai.html#genai.types.HarmCategory.HARM_CATEGORY_UNSPECIFIED)
  - [`HarmProbability`](https://googleapis.github.io/python-genai/genai.html#genai.types.HarmProbability)
    - [`HarmProbability.HARM_PROBABILITY_UNSPECIFIED`](https://googleapis.github.io/python-genai/genai.html#genai.types.HarmProbability.HARM_PROBABILITY_UNSPECIFIED)
    - [`HarmProbability.HIGH`](https://googleapis.github.io/python-genai/genai.html#genai.types.HarmProbability.HIGH)
    - [`HarmProbability.LOW`](https://googleapis.github.io/python-genai/genai.html#genai.types.HarmProbability.LOW)
    - [`HarmProbability.MEDIUM`](https://googleapis.github.io/python-genai/genai.html#genai.types.HarmProbability.MEDIUM)
    - [`HarmProbability.NEGLIGIBLE`](https://googleapis.github.io/python-genai/genai.html#genai.types.HarmProbability.NEGLIGIBLE)
  - [`HarmSeverity`](https://googleapis.github.io/python-genai/genai.html#genai.types.HarmSeverity)
    - [`HarmSeverity.HARM_SEVERITY_HIGH`](https://googleapis.github.io/python-genai/genai.html#genai.types.HarmSeverity.HARM_SEVERITY_HIGH)
    - [`HarmSeverity.HARM_SEVERITY_LOW`](https://googleapis.github.io/python-genai/genai.html#genai.types.HarmSeverity.HARM_SEVERITY_LOW)
    - [`HarmSeverity.HARM_SEVERITY_MEDIUM`](https://googleapis.github.io/python-genai/genai.html#genai.types.HarmSeverity.HARM_SEVERITY_MEDIUM)
    - [`HarmSeverity.HARM_SEVERITY_NEGLIGIBLE`](https://googleapis.github.io/python-genai/genai.html#genai.types.HarmSeverity.HARM_SEVERITY_NEGLIGIBLE)
    - [`HarmSeverity.HARM_SEVERITY_UNSPECIFIED`](https://googleapis.github.io/python-genai/genai.html#genai.types.HarmSeverity.HARM_SEVERITY_UNSPECIFIED)
  - [`HttpOptions`](https://googleapis.github.io/python-genai/genai.html#genai.types.HttpOptions)
    - [`HttpOptions.api_version`](https://googleapis.github.io/python-genai/genai.html#genai.types.HttpOptions.api_version)
    - [`HttpOptions.async_client_args`](https://googleapis.github.io/python-genai/genai.html#genai.types.HttpOptions.async_client_args)
    - [`HttpOptions.base_url`](https://googleapis.github.io/python-genai/genai.html#genai.types.HttpOptions.base_url)
    - [`HttpOptions.client_args`](https://googleapis.github.io/python-genai/genai.html#genai.types.HttpOptions.client_args)
    - [`HttpOptions.extra_body`](https://googleapis.github.io/python-genai/genai.html#genai.types.HttpOptions.extra_body)
    - [`HttpOptions.headers`](https://googleapis.github.io/python-genai/genai.html#genai.types.HttpOptions.headers)
    - [`HttpOptions.retry_options`](https://googleapis.github.io/python-genai/genai.html#genai.types.HttpOptions.retry_options)
    - [`HttpOptions.timeout`](https://googleapis.github.io/python-genai/genai.html#genai.types.HttpOptions.timeout)
  - [`HttpOptionsDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.HttpOptionsDict)
    - [`HttpOptionsDict.api_version`](https://googleapis.github.io/python-genai/genai.html#genai.types.HttpOptionsDict.api_version)
    - [`HttpOptionsDict.async_client_args`](https://googleapis.github.io/python-genai/genai.html#genai.types.HttpOptionsDict.async_client_args)
    - [`HttpOptionsDict.base_url`](https://googleapis.github.io/python-genai/genai.html#genai.types.HttpOptionsDict.base_url)
    - [`HttpOptionsDict.client_args`](https://googleapis.github.io/python-genai/genai.html#genai.types.HttpOptionsDict.client_args)
    - [`HttpOptionsDict.extra_body`](https://googleapis.github.io/python-genai/genai.html#genai.types.HttpOptionsDict.extra_body)
    - [`HttpOptionsDict.headers`](https://googleapis.github.io/python-genai/genai.html#genai.types.HttpOptionsDict.headers)
    - [`HttpOptionsDict.retry_options`](https://googleapis.github.io/python-genai/genai.html#genai.types.HttpOptionsDict.retry_options)
    - [`HttpOptionsDict.timeout`](https://googleapis.github.io/python-genai/genai.html#genai.types.HttpOptionsDict.timeout)
  - [`HttpResponse`](https://googleapis.github.io/python-genai/genai.html#genai.types.HttpResponse)
    - [`HttpResponse.body`](https://googleapis.github.io/python-genai/genai.html#genai.types.HttpResponse.body)
    - [`HttpResponse.headers`](https://googleapis.github.io/python-genai/genai.html#genai.types.HttpResponse.headers)
  - [`HttpResponseDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.HttpResponseDict)
    - [`HttpResponseDict.body`](https://googleapis.github.io/python-genai/genai.html#genai.types.HttpResponseDict.body)
    - [`HttpResponseDict.headers`](https://googleapis.github.io/python-genai/genai.html#genai.types.HttpResponseDict.headers)
  - [`HttpRetryOptions`](https://googleapis.github.io/python-genai/genai.html#genai.types.HttpRetryOptions)
    - [`HttpRetryOptions.attempts`](https://googleapis.github.io/python-genai/genai.html#genai.types.HttpRetryOptions.attempts)
    - [`HttpRetryOptions.exp_base`](https://googleapis.github.io/python-genai/genai.html#genai.types.HttpRetryOptions.exp_base)
    - [`HttpRetryOptions.http_status_codes`](https://googleapis.github.io/python-genai/genai.html#genai.types.HttpRetryOptions.http_status_codes)
    - [`HttpRetryOptions.initial_delay`](https://googleapis.github.io/python-genai/genai.html#genai.types.HttpRetryOptions.initial_delay)
    - [`HttpRetryOptions.jitter`](https://googleapis.github.io/python-genai/genai.html#genai.types.HttpRetryOptions.jitter)
    - [`HttpRetryOptions.max_delay`](https://googleapis.github.io/python-genai/genai.html#genai.types.HttpRetryOptions.max_delay)
  - [`HttpRetryOptionsDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.HttpRetryOptionsDict)
    - [`HttpRetryOptionsDict.attempts`](https://googleapis.github.io/python-genai/genai.html#genai.types.HttpRetryOptionsDict.attempts)
    - [`HttpRetryOptionsDict.exp_base`](https://googleapis.github.io/python-genai/genai.html#genai.types.HttpRetryOptionsDict.exp_base)
    - [`HttpRetryOptionsDict.http_status_codes`](https://googleapis.github.io/python-genai/genai.html#genai.types.HttpRetryOptionsDict.http_status_codes)
    - [`HttpRetryOptionsDict.initial_delay`](https://googleapis.github.io/python-genai/genai.html#genai.types.HttpRetryOptionsDict.initial_delay)
    - [`HttpRetryOptionsDict.jitter`](https://googleapis.github.io/python-genai/genai.html#genai.types.HttpRetryOptionsDict.jitter)
    - [`HttpRetryOptionsDict.max_delay`](https://googleapis.github.io/python-genai/genai.html#genai.types.HttpRetryOptionsDict.max_delay)
  - [`Image`](https://googleapis.github.io/python-genai/genai.html#genai.types.Image)
    - [`Image.gcs_uri`](https://googleapis.github.io/python-genai/genai.html#genai.types.Image.gcs_uri)
    - [`Image.image_bytes`](https://googleapis.github.io/python-genai/genai.html#genai.types.Image.image_bytes)
    - [`Image.mime_type`](https://googleapis.github.io/python-genai/genai.html#genai.types.Image.mime_type)
    - [`Image.from_file()`](https://googleapis.github.io/python-genai/genai.html#genai.types.Image.from_file)
    - [`Image.model_post_init()`](https://googleapis.github.io/python-genai/genai.html#genai.types.Image.model_post_init)
    - [`Image.save()`](https://googleapis.github.io/python-genai/genai.html#genai.types.Image.save)
    - [`Image.show()`](https://googleapis.github.io/python-genai/genai.html#genai.types.Image.show)
  - [`ImageDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.ImageDict)
    - [`ImageDict.gcs_uri`](https://googleapis.github.io/python-genai/genai.html#genai.types.ImageDict.gcs_uri)
    - [`ImageDict.image_bytes`](https://googleapis.github.io/python-genai/genai.html#genai.types.ImageDict.image_bytes)
    - [`ImageDict.mime_type`](https://googleapis.github.io/python-genai/genai.html#genai.types.ImageDict.mime_type)
  - [`ImagePromptLanguage`](https://googleapis.github.io/python-genai/genai.html#genai.types.ImagePromptLanguage)
    - [`ImagePromptLanguage.auto`](https://googleapis.github.io/python-genai/genai.html#genai.types.ImagePromptLanguage.auto)
    - [`ImagePromptLanguage.en`](https://googleapis.github.io/python-genai/genai.html#genai.types.ImagePromptLanguage.en)
    - [`ImagePromptLanguage.hi`](https://googleapis.github.io/python-genai/genai.html#genai.types.ImagePromptLanguage.hi)
    - [`ImagePromptLanguage.ja`](https://googleapis.github.io/python-genai/genai.html#genai.types.ImagePromptLanguage.ja)
    - [`ImagePromptLanguage.ko`](https://googleapis.github.io/python-genai/genai.html#genai.types.ImagePromptLanguage.ko)
  - [`Interval`](https://googleapis.github.io/python-genai/genai.html#genai.types.Interval)
    - [`Interval.end_time`](https://googleapis.github.io/python-genai/genai.html#genai.types.Interval.end_time)
    - [`Interval.start_time`](https://googleapis.github.io/python-genai/genai.html#genai.types.Interval.start_time)
  - [`IntervalDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.IntervalDict)
    - [`IntervalDict.end_time`](https://googleapis.github.io/python-genai/genai.html#genai.types.IntervalDict.end_time)
    - [`IntervalDict.start_time`](https://googleapis.github.io/python-genai/genai.html#genai.types.IntervalDict.start_time)
  - [`JSONSchema`](https://googleapis.github.io/python-genai/genai.html#genai.types.JSONSchema)
    - [`JSONSchema.any_of`](https://googleapis.github.io/python-genai/genai.html#genai.types.JSONSchema.any_of)
    - [`JSONSchema.default`](https://googleapis.github.io/python-genai/genai.html#genai.types.JSONSchema.default)
    - [`JSONSchema.description`](https://googleapis.github.io/python-genai/genai.html#genai.types.JSONSchema.description)
    - [`JSONSchema.enum`](https://googleapis.github.io/python-genai/genai.html#genai.types.JSONSchema.enum)
    - [`JSONSchema.format`](https://googleapis.github.io/python-genai/genai.html#genai.types.JSONSchema.format)
    - [`JSONSchema.items`](https://googleapis.github.io/python-genai/genai.html#genai.types.JSONSchema.items)
    - [`JSONSchema.max_items`](https://googleapis.github.io/python-genai/genai.html#genai.types.JSONSchema.max_items)
    - [`JSONSchema.max_length`](https://googleapis.github.io/python-genai/genai.html#genai.types.JSONSchema.max_length)
    - [`JSONSchema.max_properties`](https://googleapis.github.io/python-genai/genai.html#genai.types.JSONSchema.max_properties)
    - [`JSONSchema.maximum`](https://googleapis.github.io/python-genai/genai.html#genai.types.JSONSchema.maximum)
    - [`JSONSchema.min_items`](https://googleapis.github.io/python-genai/genai.html#genai.types.JSONSchema.min_items)
    - [`JSONSchema.min_length`](https://googleapis.github.io/python-genai/genai.html#genai.types.JSONSchema.min_length)
    - [`JSONSchema.min_properties`](https://googleapis.github.io/python-genai/genai.html#genai.types.JSONSchema.min_properties)
    - [`JSONSchema.minimum`](https://googleapis.github.io/python-genai/genai.html#genai.types.JSONSchema.minimum)
    - [`JSONSchema.pattern`](https://googleapis.github.io/python-genai/genai.html#genai.types.JSONSchema.pattern)
    - [`JSONSchema.properties`](https://googleapis.github.io/python-genai/genai.html#genai.types.JSONSchema.properties)
    - [`JSONSchema.required`](https://googleapis.github.io/python-genai/genai.html#genai.types.JSONSchema.required)
    - [`JSONSchema.title`](https://googleapis.github.io/python-genai/genai.html#genai.types.JSONSchema.title)
    - [`JSONSchema.type`](https://googleapis.github.io/python-genai/genai.html#genai.types.JSONSchema.type)
  - [`JSONSchemaType`](https://googleapis.github.io/python-genai/genai.html#genai.types.JSONSchemaType)
    - [`JSONSchemaType.ARRAY`](https://googleapis.github.io/python-genai/genai.html#genai.types.JSONSchemaType.ARRAY)
    - [`JSONSchemaType.BOOLEAN`](https://googleapis.github.io/python-genai/genai.html#genai.types.JSONSchemaType.BOOLEAN)
    - [`JSONSchemaType.INTEGER`](https://googleapis.github.io/python-genai/genai.html#genai.types.JSONSchemaType.INTEGER)
    - [`JSONSchemaType.NULL`](https://googleapis.github.io/python-genai/genai.html#genai.types.JSONSchemaType.NULL)
    - [`JSONSchemaType.NUMBER`](https://googleapis.github.io/python-genai/genai.html#genai.types.JSONSchemaType.NUMBER)
    - [`JSONSchemaType.OBJECT`](https://googleapis.github.io/python-genai/genai.html#genai.types.JSONSchemaType.OBJECT)
    - [`JSONSchemaType.STRING`](https://googleapis.github.io/python-genai/genai.html#genai.types.JSONSchemaType.STRING)
  - [`JobError`](https://googleapis.github.io/python-genai/genai.html#genai.types.JobError)
    - [`JobError.code`](https://googleapis.github.io/python-genai/genai.html#genai.types.JobError.code)
    - [`JobError.details`](https://googleapis.github.io/python-genai/genai.html#genai.types.JobError.details)
    - [`JobError.message`](https://googleapis.github.io/python-genai/genai.html#genai.types.JobError.message)
  - [`JobErrorDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.JobErrorDict)
    - [`JobErrorDict.code`](https://googleapis.github.io/python-genai/genai.html#genai.types.JobErrorDict.code)
    - [`JobErrorDict.details`](https://googleapis.github.io/python-genai/genai.html#genai.types.JobErrorDict.details)
    - [`JobErrorDict.message`](https://googleapis.github.io/python-genai/genai.html#genai.types.JobErrorDict.message)
  - [`JobState`](https://googleapis.github.io/python-genai/genai.html#genai.types.JobState)
    - [`JobState.JOB_STATE_CANCELLED`](https://googleapis.github.io/python-genai/genai.html#genai.types.JobState.JOB_STATE_CANCELLED)
    - [`JobState.JOB_STATE_CANCELLING`](https://googleapis.github.io/python-genai/genai.html#genai.types.JobState.JOB_STATE_CANCELLING)
    - [`JobState.JOB_STATE_EXPIRED`](https://googleapis.github.io/python-genai/genai.html#genai.types.JobState.JOB_STATE_EXPIRED)
    - [`JobState.JOB_STATE_FAILED`](https://googleapis.github.io/python-genai/genai.html#genai.types.JobState.JOB_STATE_FAILED)
    - [`JobState.JOB_STATE_PARTIALLY_SUCCEEDED`](https://googleapis.github.io/python-genai/genai.html#genai.types.JobState.JOB_STATE_PARTIALLY_SUCCEEDED)
    - [`JobState.JOB_STATE_PAUSED`](https://googleapis.github.io/python-genai/genai.html#genai.types.JobState.JOB_STATE_PAUSED)
    - [`JobState.JOB_STATE_PENDING`](https://googleapis.github.io/python-genai/genai.html#genai.types.JobState.JOB_STATE_PENDING)
    - [`JobState.JOB_STATE_QUEUED`](https://googleapis.github.io/python-genai/genai.html#genai.types.JobState.JOB_STATE_QUEUED)
    - [`JobState.JOB_STATE_RUNNING`](https://googleapis.github.io/python-genai/genai.html#genai.types.JobState.JOB_STATE_RUNNING)
    - [`JobState.JOB_STATE_SUCCEEDED`](https://googleapis.github.io/python-genai/genai.html#genai.types.JobState.JOB_STATE_SUCCEEDED)
    - [`JobState.JOB_STATE_UNSPECIFIED`](https://googleapis.github.io/python-genai/genai.html#genai.types.JobState.JOB_STATE_UNSPECIFIED)
    - [`JobState.JOB_STATE_UPDATING`](https://googleapis.github.io/python-genai/genai.html#genai.types.JobState.JOB_STATE_UPDATING)
  - [`Language`](https://googleapis.github.io/python-genai/genai.html#genai.types.Language)
    - [`Language.LANGUAGE_UNSPECIFIED`](https://googleapis.github.io/python-genai/genai.html#genai.types.Language.LANGUAGE_UNSPECIFIED)
    - [`Language.PYTHON`](https://googleapis.github.io/python-genai/genai.html#genai.types.Language.PYTHON)
  - [`LatLng`](https://googleapis.github.io/python-genai/genai.html#genai.types.LatLng)
    - [`LatLng.latitude`](https://googleapis.github.io/python-genai/genai.html#genai.types.LatLng.latitude)
    - [`LatLng.longitude`](https://googleapis.github.io/python-genai/genai.html#genai.types.LatLng.longitude)
  - [`LatLngDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.LatLngDict)
    - [`LatLngDict.latitude`](https://googleapis.github.io/python-genai/genai.html#genai.types.LatLngDict.latitude)
    - [`LatLngDict.longitude`](https://googleapis.github.io/python-genai/genai.html#genai.types.LatLngDict.longitude)
  - [`ListBatchJobsConfig`](https://googleapis.github.io/python-genai/genai.html#genai.types.ListBatchJobsConfig)
    - [`ListBatchJobsConfig.filter`](https://googleapis.github.io/python-genai/genai.html#genai.types.ListBatchJobsConfig.filter)
    - [`ListBatchJobsConfig.http_options`](https://googleapis.github.io/python-genai/genai.html#genai.types.ListBatchJobsConfig.http_options)
    - [`ListBatchJobsConfig.page_size`](https://googleapis.github.io/python-genai/genai.html#genai.types.ListBatchJobsConfig.page_size)
    - [`ListBatchJobsConfig.page_token`](https://googleapis.github.io/python-genai/genai.html#genai.types.ListBatchJobsConfig.page_token)
  - [`ListBatchJobsConfigDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.ListBatchJobsConfigDict)
    - [`ListBatchJobsConfigDict.filter`](https://googleapis.github.io/python-genai/genai.html#genai.types.ListBatchJobsConfigDict.filter)
    - [`ListBatchJobsConfigDict.http_options`](https://googleapis.github.io/python-genai/genai.html#genai.types.ListBatchJobsConfigDict.http_options)
    - [`ListBatchJobsConfigDict.page_size`](https://googleapis.github.io/python-genai/genai.html#genai.types.ListBatchJobsConfigDict.page_size)
    - [`ListBatchJobsConfigDict.page_token`](https://googleapis.github.io/python-genai/genai.html#genai.types.ListBatchJobsConfigDict.page_token)
  - [`ListBatchJobsResponse`](https://googleapis.github.io/python-genai/genai.html#genai.types.ListBatchJobsResponse)
    - [`ListBatchJobsResponse.batch_jobs`](https://googleapis.github.io/python-genai/genai.html#genai.types.ListBatchJobsResponse.batch_jobs)
    - [`ListBatchJobsResponse.next_page_token`](https://googleapis.github.io/python-genai/genai.html#genai.types.ListBatchJobsResponse.next_page_token)
  - [`ListBatchJobsResponseDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.ListBatchJobsResponseDict)
    - [`ListBatchJobsResponseDict.batch_jobs`](https://googleapis.github.io/python-genai/genai.html#genai.types.ListBatchJobsResponseDict.batch_jobs)
    - [`ListBatchJobsResponseDict.next_page_token`](https://googleapis.github.io/python-genai/genai.html#genai.types.ListBatchJobsResponseDict.next_page_token)
  - [`ListCachedContentsConfig`](https://googleapis.github.io/python-genai/genai.html#genai.types.ListCachedContentsConfig)
    - [`ListCachedContentsConfig.http_options`](https://googleapis.github.io/python-genai/genai.html#genai.types.ListCachedContentsConfig.http_options)
    - [`ListCachedContentsConfig.page_size`](https://googleapis.github.io/python-genai/genai.html#genai.types.ListCachedContentsConfig.page_size)
    - [`ListCachedContentsConfig.page_token`](https://googleapis.github.io/python-genai/genai.html#genai.types.ListCachedContentsConfig.page_token)
  - [`ListCachedContentsConfigDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.ListCachedContentsConfigDict)
    - [`ListCachedContentsConfigDict.http_options`](https://googleapis.github.io/python-genai/genai.html#genai.types.ListCachedContentsConfigDict.http_options)
    - [`ListCachedContentsConfigDict.page_size`](https://googleapis.github.io/python-genai/genai.html#genai.types.ListCachedContentsConfigDict.page_size)
    - [`ListCachedContentsConfigDict.page_token`](https://googleapis.github.io/python-genai/genai.html#genai.types.ListCachedContentsConfigDict.page_token)
  - [`ListCachedContentsResponse`](https://googleapis.github.io/python-genai/genai.html#genai.types.ListCachedContentsResponse)
    - [`ListCachedContentsResponse.cached_contents`](https://googleapis.github.io/python-genai/genai.html#genai.types.ListCachedContentsResponse.cached_contents)
    - [`ListCachedContentsResponse.next_page_token`](https://googleapis.github.io/python-genai/genai.html#genai.types.ListCachedContentsResponse.next_page_token)
  - [`ListCachedContentsResponseDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.ListCachedContentsResponseDict)
    - [`ListCachedContentsResponseDict.cached_contents`](https://googleapis.github.io/python-genai/genai.html#genai.types.ListCachedContentsResponseDict.cached_contents)
    - [`ListCachedContentsResponseDict.next_page_token`](https://googleapis.github.io/python-genai/genai.html#genai.types.ListCachedContentsResponseDict.next_page_token)
  - [`ListFilesConfig`](https://googleapis.github.io/python-genai/genai.html#genai.types.ListFilesConfig)
    - [`ListFilesConfig.http_options`](https://googleapis.github.io/python-genai/genai.html#genai.types.ListFilesConfig.http_options)
    - [`ListFilesConfig.page_size`](https://googleapis.github.io/python-genai/genai.html#genai.types.ListFilesConfig.page_size)
    - [`ListFilesConfig.page_token`](https://googleapis.github.io/python-genai/genai.html#genai.types.ListFilesConfig.page_token)
  - [`ListFilesConfigDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.ListFilesConfigDict)
    - [`ListFilesConfigDict.http_options`](https://googleapis.github.io/python-genai/genai.html#genai.types.ListFilesConfigDict.http_options)
    - [`ListFilesConfigDict.page_size`](https://googleapis.github.io/python-genai/genai.html#genai.types.ListFilesConfigDict.page_size)
    - [`ListFilesConfigDict.page_token`](https://googleapis.github.io/python-genai/genai.html#genai.types.ListFilesConfigDict.page_token)
  - [`ListFilesResponse`](https://googleapis.github.io/python-genai/genai.html#genai.types.ListFilesResponse)
    - [`ListFilesResponse.files`](https://googleapis.github.io/python-genai/genai.html#genai.types.ListFilesResponse.files)
    - [`ListFilesResponse.next_page_token`](https://googleapis.github.io/python-genai/genai.html#genai.types.ListFilesResponse.next_page_token)
  - [`ListFilesResponseDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.ListFilesResponseDict)
    - [`ListFilesResponseDict.files`](https://googleapis.github.io/python-genai/genai.html#genai.types.ListFilesResponseDict.files)
    - [`ListFilesResponseDict.next_page_token`](https://googleapis.github.io/python-genai/genai.html#genai.types.ListFilesResponseDict.next_page_token)
  - [`ListModelsConfig`](https://googleapis.github.io/python-genai/genai.html#genai.types.ListModelsConfig)
    - [`ListModelsConfig.filter`](https://googleapis.github.io/python-genai/genai.html#genai.types.ListModelsConfig.filter)
    - [`ListModelsConfig.http_options`](https://googleapis.github.io/python-genai/genai.html#genai.types.ListModelsConfig.http_options)
    - [`ListModelsConfig.page_size`](https://googleapis.github.io/python-genai/genai.html#genai.types.ListModelsConfig.page_size)
    - [`ListModelsConfig.page_token`](https://googleapis.github.io/python-genai/genai.html#genai.types.ListModelsConfig.page_token)
    - [`ListModelsConfig.query_base`](https://googleapis.github.io/python-genai/genai.html#genai.types.ListModelsConfig.query_base)
  - [`ListModelsConfigDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.ListModelsConfigDict)
    - [`ListModelsConfigDict.filter`](https://googleapis.github.io/python-genai/genai.html#genai.types.ListModelsConfigDict.filter)
    - [`ListModelsConfigDict.http_options`](https://googleapis.github.io/python-genai/genai.html#genai.types.ListModelsConfigDict.http_options)
    - [`ListModelsConfigDict.page_size`](https://googleapis.github.io/python-genai/genai.html#genai.types.ListModelsConfigDict.page_size)
    - [`ListModelsConfigDict.page_token`](https://googleapis.github.io/python-genai/genai.html#genai.types.ListModelsConfigDict.page_token)
    - [`ListModelsConfigDict.query_base`](https://googleapis.github.io/python-genai/genai.html#genai.types.ListModelsConfigDict.query_base)
  - [`ListModelsResponse`](https://googleapis.github.io/python-genai/genai.html#genai.types.ListModelsResponse)
    - [`ListModelsResponse.models`](https://googleapis.github.io/python-genai/genai.html#genai.types.ListModelsResponse.models)
    - [`ListModelsResponse.next_page_token`](https://googleapis.github.io/python-genai/genai.html#genai.types.ListModelsResponse.next_page_token)
  - [`ListModelsResponseDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.ListModelsResponseDict)
    - [`ListModelsResponseDict.models`](https://googleapis.github.io/python-genai/genai.html#genai.types.ListModelsResponseDict.models)
    - [`ListModelsResponseDict.next_page_token`](https://googleapis.github.io/python-genai/genai.html#genai.types.ListModelsResponseDict.next_page_token)
  - [`ListTuningJobsConfig`](https://googleapis.github.io/python-genai/genai.html#genai.types.ListTuningJobsConfig)
    - [`ListTuningJobsConfig.filter`](https://googleapis.github.io/python-genai/genai.html#genai.types.ListTuningJobsConfig.filter)
    - [`ListTuningJobsConfig.http_options`](https://googleapis.github.io/python-genai/genai.html#genai.types.ListTuningJobsConfig.http_options)
    - [`ListTuningJobsConfig.page_size`](https://googleapis.github.io/python-genai/genai.html#genai.types.ListTuningJobsConfig.page_size)
    - [`ListTuningJobsConfig.page_token`](https://googleapis.github.io/python-genai/genai.html#genai.types.ListTuningJobsConfig.page_token)
  - [`ListTuningJobsConfigDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.ListTuningJobsConfigDict)
    - [`ListTuningJobsConfigDict.filter`](https://googleapis.github.io/python-genai/genai.html#genai.types.ListTuningJobsConfigDict.filter)
    - [`ListTuningJobsConfigDict.http_options`](https://googleapis.github.io/python-genai/genai.html#genai.types.ListTuningJobsConfigDict.http_options)
    - [`ListTuningJobsConfigDict.page_size`](https://googleapis.github.io/python-genai/genai.html#genai.types.ListTuningJobsConfigDict.page_size)
    - [`ListTuningJobsConfigDict.page_token`](https://googleapis.github.io/python-genai/genai.html#genai.types.ListTuningJobsConfigDict.page_token)
  - [`ListTuningJobsResponse`](https://googleapis.github.io/python-genai/genai.html#genai.types.ListTuningJobsResponse)
    - [`ListTuningJobsResponse.next_page_token`](https://googleapis.github.io/python-genai/genai.html#genai.types.ListTuningJobsResponse.next_page_token)
    - [`ListTuningJobsResponse.tuning_jobs`](https://googleapis.github.io/python-genai/genai.html#genai.types.ListTuningJobsResponse.tuning_jobs)
  - [`ListTuningJobsResponseDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.ListTuningJobsResponseDict)
    - [`ListTuningJobsResponseDict.next_page_token`](https://googleapis.github.io/python-genai/genai.html#genai.types.ListTuningJobsResponseDict.next_page_token)
    - [`ListTuningJobsResponseDict.tuning_jobs`](https://googleapis.github.io/python-genai/genai.html#genai.types.ListTuningJobsResponseDict.tuning_jobs)
  - [`LiveClientContent`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveClientContent)
    - [`LiveClientContent.turn_complete`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveClientContent.turn_complete)
    - [`LiveClientContent.turns`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveClientContent.turns)
  - [`LiveClientContentDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveClientContentDict)
    - [`LiveClientContentDict.turn_complete`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveClientContentDict.turn_complete)
    - [`LiveClientContentDict.turns`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveClientContentDict.turns)
  - [`LiveClientMessage`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveClientMessage)
    - [`LiveClientMessage.client_content`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveClientMessage.client_content)
    - [`LiveClientMessage.realtime_input`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveClientMessage.realtime_input)
    - [`LiveClientMessage.setup`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveClientMessage.setup)
    - [`LiveClientMessage.tool_response`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveClientMessage.tool_response)
  - [`LiveClientMessageDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveClientMessageDict)
    - [`LiveClientMessageDict.client_content`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveClientMessageDict.client_content)
    - [`LiveClientMessageDict.realtime_input`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveClientMessageDict.realtime_input)
    - [`LiveClientMessageDict.setup`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveClientMessageDict.setup)
    - [`LiveClientMessageDict.tool_response`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveClientMessageDict.tool_response)
  - [`LiveClientRealtimeInput`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveClientRealtimeInput)
    - [`LiveClientRealtimeInput.activity_end`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveClientRealtimeInput.activity_end)
    - [`LiveClientRealtimeInput.activity_start`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveClientRealtimeInput.activity_start)
    - [`LiveClientRealtimeInput.audio`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveClientRealtimeInput.audio)
    - [`LiveClientRealtimeInput.audio_stream_end`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveClientRealtimeInput.audio_stream_end)
    - [`LiveClientRealtimeInput.media_chunks`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveClientRealtimeInput.media_chunks)
    - [`LiveClientRealtimeInput.text`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveClientRealtimeInput.text)
    - [`LiveClientRealtimeInput.video`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveClientRealtimeInput.video)
  - [`LiveClientRealtimeInputDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveClientRealtimeInputDict)
    - [`LiveClientRealtimeInputDict.activity_end`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveClientRealtimeInputDict.activity_end)
    - [`LiveClientRealtimeInputDict.activity_start`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveClientRealtimeInputDict.activity_start)
    - [`LiveClientRealtimeInputDict.audio`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveClientRealtimeInputDict.audio)
    - [`LiveClientRealtimeInputDict.audio_stream_end`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveClientRealtimeInputDict.audio_stream_end)
    - [`LiveClientRealtimeInputDict.media_chunks`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveClientRealtimeInputDict.media_chunks)
    - [`LiveClientRealtimeInputDict.text`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveClientRealtimeInputDict.text)
    - [`LiveClientRealtimeInputDict.video`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveClientRealtimeInputDict.video)
  - [`LiveClientSetup`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveClientSetup)
    - [`LiveClientSetup.context_window_compression`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveClientSetup.context_window_compression)
    - [`LiveClientSetup.generation_config`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveClientSetup.generation_config)
    - [`LiveClientSetup.input_audio_transcription`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveClientSetup.input_audio_transcription)
    - [`LiveClientSetup.model`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveClientSetup.model)
    - [`LiveClientSetup.output_audio_transcription`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveClientSetup.output_audio_transcription)
    - [`LiveClientSetup.proactivity`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveClientSetup.proactivity)
    - [`LiveClientSetup.session_resumption`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveClientSetup.session_resumption)
    - [`LiveClientSetup.system_instruction`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveClientSetup.system_instruction)
    - [`LiveClientSetup.tools`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveClientSetup.tools)
  - [`LiveClientSetupDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveClientSetupDict)
    - [`LiveClientSetupDict.context_window_compression`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveClientSetupDict.context_window_compression)
    - [`LiveClientSetupDict.generation_config`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveClientSetupDict.generation_config)
    - [`LiveClientSetupDict.input_audio_transcription`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveClientSetupDict.input_audio_transcription)
    - [`LiveClientSetupDict.model`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveClientSetupDict.model)
    - [`LiveClientSetupDict.output_audio_transcription`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveClientSetupDict.output_audio_transcription)
    - [`LiveClientSetupDict.proactivity`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveClientSetupDict.proactivity)
    - [`LiveClientSetupDict.session_resumption`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveClientSetupDict.session_resumption)
    - [`LiveClientSetupDict.system_instruction`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveClientSetupDict.system_instruction)
    - [`LiveClientSetupDict.tools`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveClientSetupDict.tools)
  - [`LiveClientToolResponse`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveClientToolResponse)
    - [`LiveClientToolResponse.function_responses`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveClientToolResponse.function_responses)
  - [`LiveClientToolResponseDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveClientToolResponseDict)
    - [`LiveClientToolResponseDict.function_responses`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveClientToolResponseDict.function_responses)
  - [`LiveConnectConfig`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveConnectConfig)
    - [`LiveConnectConfig.context_window_compression`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveConnectConfig.context_window_compression)
    - [`LiveConnectConfig.enable_affective_dialog`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveConnectConfig.enable_affective_dialog)
    - [`LiveConnectConfig.generation_config`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveConnectConfig.generation_config)
    - [`LiveConnectConfig.http_options`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveConnectConfig.http_options)
    - [`LiveConnectConfig.input_audio_transcription`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveConnectConfig.input_audio_transcription)
    - [`LiveConnectConfig.max_output_tokens`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveConnectConfig.max_output_tokens)
    - [`LiveConnectConfig.media_resolution`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveConnectConfig.media_resolution)
    - [`LiveConnectConfig.output_audio_transcription`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveConnectConfig.output_audio_transcription)
    - [`LiveConnectConfig.proactivity`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveConnectConfig.proactivity)
    - [`LiveConnectConfig.realtime_input_config`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveConnectConfig.realtime_input_config)
    - [`LiveConnectConfig.response_modalities`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveConnectConfig.response_modalities)
    - [`LiveConnectConfig.seed`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveConnectConfig.seed)
    - [`LiveConnectConfig.session_resumption`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveConnectConfig.session_resumption)
    - [`LiveConnectConfig.speech_config`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveConnectConfig.speech_config)
    - [`LiveConnectConfig.system_instruction`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveConnectConfig.system_instruction)
    - [`LiveConnectConfig.temperature`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveConnectConfig.temperature)
    - [`LiveConnectConfig.tools`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveConnectConfig.tools)
    - [`LiveConnectConfig.top_k`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveConnectConfig.top_k)
    - [`LiveConnectConfig.top_p`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveConnectConfig.top_p)
  - [`LiveConnectConfigDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveConnectConfigDict)
    - [`LiveConnectConfigDict.context_window_compression`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveConnectConfigDict.context_window_compression)
    - [`LiveConnectConfigDict.enable_affective_dialog`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveConnectConfigDict.enable_affective_dialog)
    - [`LiveConnectConfigDict.generation_config`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveConnectConfigDict.generation_config)
    - [`LiveConnectConfigDict.http_options`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveConnectConfigDict.http_options)
    - [`LiveConnectConfigDict.input_audio_transcription`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveConnectConfigDict.input_audio_transcription)
    - [`LiveConnectConfigDict.max_output_tokens`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveConnectConfigDict.max_output_tokens)
    - [`LiveConnectConfigDict.media_resolution`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveConnectConfigDict.media_resolution)
    - [`LiveConnectConfigDict.output_audio_transcription`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveConnectConfigDict.output_audio_transcription)
    - [`LiveConnectConfigDict.proactivity`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveConnectConfigDict.proactivity)
    - [`LiveConnectConfigDict.realtime_input_config`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveConnectConfigDict.realtime_input_config)
    - [`LiveConnectConfigDict.response_modalities`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveConnectConfigDict.response_modalities)
    - [`LiveConnectConfigDict.seed`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveConnectConfigDict.seed)
    - [`LiveConnectConfigDict.session_resumption`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveConnectConfigDict.session_resumption)
    - [`LiveConnectConfigDict.speech_config`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveConnectConfigDict.speech_config)
    - [`LiveConnectConfigDict.system_instruction`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveConnectConfigDict.system_instruction)
    - [`LiveConnectConfigDict.temperature`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveConnectConfigDict.temperature)
    - [`LiveConnectConfigDict.tools`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveConnectConfigDict.tools)
    - [`LiveConnectConfigDict.top_k`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveConnectConfigDict.top_k)
    - [`LiveConnectConfigDict.top_p`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveConnectConfigDict.top_p)
  - [`LiveConnectConstraints`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveConnectConstraints)
    - [`LiveConnectConstraints.config`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveConnectConstraints.config)
    - [`LiveConnectConstraints.model`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveConnectConstraints.model)
  - [`LiveConnectConstraintsDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveConnectConstraintsDict)
    - [`LiveConnectConstraintsDict.config`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveConnectConstraintsDict.config)
    - [`LiveConnectConstraintsDict.model`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveConnectConstraintsDict.model)
  - [`LiveConnectParameters`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveConnectParameters)
    - [`LiveConnectParameters.config`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveConnectParameters.config)
    - [`LiveConnectParameters.model`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveConnectParameters.model)
  - [`LiveConnectParametersDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveConnectParametersDict)
    - [`LiveConnectParametersDict.config`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveConnectParametersDict.config)
    - [`LiveConnectParametersDict.model`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveConnectParametersDict.model)
  - [`LiveMusicClientContent`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveMusicClientContent)
    - [`LiveMusicClientContent.weighted_prompts`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveMusicClientContent.weighted_prompts)
  - [`LiveMusicClientContentDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveMusicClientContentDict)
    - [`LiveMusicClientContentDict.weighted_prompts`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveMusicClientContentDict.weighted_prompts)
  - [`LiveMusicClientMessage`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveMusicClientMessage)
    - [`LiveMusicClientMessage.client_content`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveMusicClientMessage.client_content)
    - [`LiveMusicClientMessage.music_generation_config`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveMusicClientMessage.music_generation_config)
    - [`LiveMusicClientMessage.playback_control`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveMusicClientMessage.playback_control)
    - [`LiveMusicClientMessage.setup`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveMusicClientMessage.setup)
  - [`LiveMusicClientMessageDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveMusicClientMessageDict)
    - [`LiveMusicClientMessageDict.client_content`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveMusicClientMessageDict.client_content)
    - [`LiveMusicClientMessageDict.music_generation_config`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveMusicClientMessageDict.music_generation_config)
    - [`LiveMusicClientMessageDict.playback_control`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveMusicClientMessageDict.playback_control)
    - [`LiveMusicClientMessageDict.setup`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveMusicClientMessageDict.setup)
  - [`LiveMusicClientSetup`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveMusicClientSetup)
    - [`LiveMusicClientSetup.model`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveMusicClientSetup.model)
  - [`LiveMusicClientSetupDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveMusicClientSetupDict)
    - [`LiveMusicClientSetupDict.model`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveMusicClientSetupDict.model)
  - [`LiveMusicConnectParameters`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveMusicConnectParameters)
    - [`LiveMusicConnectParameters.model`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveMusicConnectParameters.model)
  - [`LiveMusicConnectParametersDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveMusicConnectParametersDict)
    - [`LiveMusicConnectParametersDict.model`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveMusicConnectParametersDict.model)
  - [`LiveMusicFilteredPrompt`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveMusicFilteredPrompt)
    - [`LiveMusicFilteredPrompt.filtered_reason`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveMusicFilteredPrompt.filtered_reason)
    - [`LiveMusicFilteredPrompt.text`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveMusicFilteredPrompt.text)
  - [`LiveMusicFilteredPromptDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveMusicFilteredPromptDict)
    - [`LiveMusicFilteredPromptDict.filtered_reason`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveMusicFilteredPromptDict.filtered_reason)
    - [`LiveMusicFilteredPromptDict.text`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveMusicFilteredPromptDict.text)
  - [`LiveMusicGenerationConfig`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveMusicGenerationConfig)
    - [`LiveMusicGenerationConfig.bpm`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveMusicGenerationConfig.bpm)
    - [`LiveMusicGenerationConfig.brightness`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveMusicGenerationConfig.brightness)
    - [`LiveMusicGenerationConfig.density`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveMusicGenerationConfig.density)
    - [`LiveMusicGenerationConfig.guidance`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveMusicGenerationConfig.guidance)
    - [`LiveMusicGenerationConfig.mute_bass`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveMusicGenerationConfig.mute_bass)
    - [`LiveMusicGenerationConfig.mute_drums`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveMusicGenerationConfig.mute_drums)
    - [`LiveMusicGenerationConfig.only_bass_and_drums`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveMusicGenerationConfig.only_bass_and_drums)
    - [`LiveMusicGenerationConfig.scale`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveMusicGenerationConfig.scale)
    - [`LiveMusicGenerationConfig.seed`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveMusicGenerationConfig.seed)
    - [`LiveMusicGenerationConfig.temperature`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveMusicGenerationConfig.temperature)
    - [`LiveMusicGenerationConfig.top_k`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveMusicGenerationConfig.top_k)
  - [`LiveMusicGenerationConfigDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveMusicGenerationConfigDict)
    - [`LiveMusicGenerationConfigDict.bpm`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveMusicGenerationConfigDict.bpm)
    - [`LiveMusicGenerationConfigDict.brightness`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveMusicGenerationConfigDict.brightness)
    - [`LiveMusicGenerationConfigDict.density`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveMusicGenerationConfigDict.density)
    - [`LiveMusicGenerationConfigDict.guidance`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveMusicGenerationConfigDict.guidance)
    - [`LiveMusicGenerationConfigDict.mute_bass`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveMusicGenerationConfigDict.mute_bass)
    - [`LiveMusicGenerationConfigDict.mute_drums`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveMusicGenerationConfigDict.mute_drums)
    - [`LiveMusicGenerationConfigDict.only_bass_and_drums`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveMusicGenerationConfigDict.only_bass_and_drums)
    - [`LiveMusicGenerationConfigDict.scale`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveMusicGenerationConfigDict.scale)
    - [`LiveMusicGenerationConfigDict.seed`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveMusicGenerationConfigDict.seed)
    - [`LiveMusicGenerationConfigDict.temperature`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveMusicGenerationConfigDict.temperature)
    - [`LiveMusicGenerationConfigDict.top_k`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveMusicGenerationConfigDict.top_k)
  - [`LiveMusicPlaybackControl`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveMusicPlaybackControl)
    - [`LiveMusicPlaybackControl.PAUSE`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveMusicPlaybackControl.PAUSE)
    - [`LiveMusicPlaybackControl.PLAY`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveMusicPlaybackControl.PLAY)
    - [`LiveMusicPlaybackControl.PLAYBACK_CONTROL_UNSPECIFIED`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveMusicPlaybackControl.PLAYBACK_CONTROL_UNSPECIFIED)
    - [`LiveMusicPlaybackControl.RESET_CONTEXT`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveMusicPlaybackControl.RESET_CONTEXT)
    - [`LiveMusicPlaybackControl.STOP`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveMusicPlaybackControl.STOP)
  - [`LiveMusicServerContent`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveMusicServerContent)
    - [`LiveMusicServerContent.audio_chunks`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveMusicServerContent.audio_chunks)
  - [`LiveMusicServerContentDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveMusicServerContentDict)
    - [`LiveMusicServerContentDict.audio_chunks`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveMusicServerContentDict.audio_chunks)
  - [`LiveMusicServerMessage`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveMusicServerMessage)
    - [`LiveMusicServerMessage.filtered_prompt`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveMusicServerMessage.filtered_prompt)
    - [`LiveMusicServerMessage.server_content`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveMusicServerMessage.server_content)
    - [`LiveMusicServerMessage.setup_complete`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveMusicServerMessage.setup_complete)
  - [`LiveMusicServerMessageDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveMusicServerMessageDict)
    - [`LiveMusicServerMessageDict.filtered_prompt`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveMusicServerMessageDict.filtered_prompt)
    - [`LiveMusicServerMessageDict.server_content`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveMusicServerMessageDict.server_content)
    - [`LiveMusicServerMessageDict.setup_complete`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveMusicServerMessageDict.setup_complete)
  - [`LiveMusicServerSetupComplete`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveMusicServerSetupComplete)
  - [`LiveMusicServerSetupCompleteDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveMusicServerSetupCompleteDict)
  - [`LiveMusicSetConfigParameters`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveMusicSetConfigParameters)
    - [`LiveMusicSetConfigParameters.music_generation_config`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveMusicSetConfigParameters.music_generation_config)
  - [`LiveMusicSetConfigParametersDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveMusicSetConfigParametersDict)
    - [`LiveMusicSetConfigParametersDict.music_generation_config`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveMusicSetConfigParametersDict.music_generation_config)
  - [`LiveMusicSetWeightedPromptsParameters`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveMusicSetWeightedPromptsParameters)
    - [`LiveMusicSetWeightedPromptsParameters.weighted_prompts`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveMusicSetWeightedPromptsParameters.weighted_prompts)
  - [`LiveMusicSetWeightedPromptsParametersDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveMusicSetWeightedPromptsParametersDict)
    - [`LiveMusicSetWeightedPromptsParametersDict.weighted_prompts`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveMusicSetWeightedPromptsParametersDict.weighted_prompts)
  - [`LiveMusicSourceMetadata`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveMusicSourceMetadata)
    - [`LiveMusicSourceMetadata.client_content`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveMusicSourceMetadata.client_content)
    - [`LiveMusicSourceMetadata.music_generation_config`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveMusicSourceMetadata.music_generation_config)
  - [`LiveMusicSourceMetadataDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveMusicSourceMetadataDict)
    - [`LiveMusicSourceMetadataDict.client_content`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveMusicSourceMetadataDict.client_content)
    - [`LiveMusicSourceMetadataDict.music_generation_config`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveMusicSourceMetadataDict.music_generation_config)
  - [`LiveSendRealtimeInputParameters`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveSendRealtimeInputParameters)
    - [`LiveSendRealtimeInputParameters.activity_end`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveSendRealtimeInputParameters.activity_end)
    - [`LiveSendRealtimeInputParameters.activity_start`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveSendRealtimeInputParameters.activity_start)
    - [`LiveSendRealtimeInputParameters.audio`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveSendRealtimeInputParameters.audio)
    - [`LiveSendRealtimeInputParameters.audio_stream_end`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveSendRealtimeInputParameters.audio_stream_end)
    - [`LiveSendRealtimeInputParameters.media`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveSendRealtimeInputParameters.media)
    - [`LiveSendRealtimeInputParameters.text`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveSendRealtimeInputParameters.text)
    - [`LiveSendRealtimeInputParameters.video`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveSendRealtimeInputParameters.video)
  - [`LiveSendRealtimeInputParametersDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveSendRealtimeInputParametersDict)
    - [`LiveSendRealtimeInputParametersDict.activity_end`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveSendRealtimeInputParametersDict.activity_end)
    - [`LiveSendRealtimeInputParametersDict.activity_start`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveSendRealtimeInputParametersDict.activity_start)
    - [`LiveSendRealtimeInputParametersDict.audio`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveSendRealtimeInputParametersDict.audio)
    - [`LiveSendRealtimeInputParametersDict.audio_stream_end`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveSendRealtimeInputParametersDict.audio_stream_end)
    - [`LiveSendRealtimeInputParametersDict.media`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveSendRealtimeInputParametersDict.media)
    - [`LiveSendRealtimeInputParametersDict.text`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveSendRealtimeInputParametersDict.text)
    - [`LiveSendRealtimeInputParametersDict.video`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveSendRealtimeInputParametersDict.video)
  - [`LiveServerContent`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveServerContent)
    - [`LiveServerContent.generation_complete`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveServerContent.generation_complete)
    - [`LiveServerContent.grounding_metadata`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveServerContent.grounding_metadata)
    - [`LiveServerContent.input_transcription`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveServerContent.input_transcription)
    - [`LiveServerContent.interrupted`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveServerContent.interrupted)
    - [`LiveServerContent.model_turn`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveServerContent.model_turn)
    - [`LiveServerContent.output_transcription`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveServerContent.output_transcription)
    - [`LiveServerContent.turn_complete`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveServerContent.turn_complete)
    - [`LiveServerContent.url_context_metadata`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveServerContent.url_context_metadata)
  - [`LiveServerContentDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveServerContentDict)
    - [`LiveServerContentDict.generation_complete`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveServerContentDict.generation_complete)
    - [`LiveServerContentDict.grounding_metadata`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveServerContentDict.grounding_metadata)
    - [`LiveServerContentDict.input_transcription`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveServerContentDict.input_transcription)
    - [`LiveServerContentDict.interrupted`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveServerContentDict.interrupted)
    - [`LiveServerContentDict.model_turn`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveServerContentDict.model_turn)
    - [`LiveServerContentDict.output_transcription`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveServerContentDict.output_transcription)
    - [`LiveServerContentDict.turn_complete`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveServerContentDict.turn_complete)
    - [`LiveServerContentDict.url_context_metadata`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveServerContentDict.url_context_metadata)
  - [`LiveServerGoAway`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveServerGoAway)
    - [`LiveServerGoAway.time_left`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveServerGoAway.time_left)
  - [`LiveServerGoAwayDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveServerGoAwayDict)
    - [`LiveServerGoAwayDict.time_left`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveServerGoAwayDict.time_left)
  - [`LiveServerMessage`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveServerMessage)
    - [`LiveServerMessage.go_away`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveServerMessage.go_away)
    - [`LiveServerMessage.server_content`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveServerMessage.server_content)
    - [`LiveServerMessage.session_resumption_update`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveServerMessage.session_resumption_update)
    - [`LiveServerMessage.setup_complete`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveServerMessage.setup_complete)
    - [`LiveServerMessage.tool_call`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveServerMessage.tool_call)
    - [`LiveServerMessage.tool_call_cancellation`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveServerMessage.tool_call_cancellation)
    - [`LiveServerMessage.usage_metadata`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveServerMessage.usage_metadata)
    - [`LiveServerMessage.data`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveServerMessage.data)
    - [`LiveServerMessage.text`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveServerMessage.text)
  - [`LiveServerMessageDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveServerMessageDict)
    - [`LiveServerMessageDict.go_away`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveServerMessageDict.go_away)
    - [`LiveServerMessageDict.server_content`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveServerMessageDict.server_content)
    - [`LiveServerMessageDict.session_resumption_update`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveServerMessageDict.session_resumption_update)
    - [`LiveServerMessageDict.setup_complete`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveServerMessageDict.setup_complete)
    - [`LiveServerMessageDict.tool_call`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveServerMessageDict.tool_call)
    - [`LiveServerMessageDict.tool_call_cancellation`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveServerMessageDict.tool_call_cancellation)
    - [`LiveServerMessageDict.usage_metadata`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveServerMessageDict.usage_metadata)
  - [`LiveServerSessionResumptionUpdate`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveServerSessionResumptionUpdate)
    - [`LiveServerSessionResumptionUpdate.last_consumed_client_message_index`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveServerSessionResumptionUpdate.last_consumed_client_message_index)
    - [`LiveServerSessionResumptionUpdate.new_handle`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveServerSessionResumptionUpdate.new_handle)
    - [`LiveServerSessionResumptionUpdate.resumable`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveServerSessionResumptionUpdate.resumable)
  - [`LiveServerSessionResumptionUpdateDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveServerSessionResumptionUpdateDict)
    - [`LiveServerSessionResumptionUpdateDict.last_consumed_client_message_index`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveServerSessionResumptionUpdateDict.last_consumed_client_message_index)
    - [`LiveServerSessionResumptionUpdateDict.new_handle`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveServerSessionResumptionUpdateDict.new_handle)
    - [`LiveServerSessionResumptionUpdateDict.resumable`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveServerSessionResumptionUpdateDict.resumable)
  - [`LiveServerSetupComplete`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveServerSetupComplete)
  - [`LiveServerSetupCompleteDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveServerSetupCompleteDict)
  - [`LiveServerToolCall`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveServerToolCall)
    - [`LiveServerToolCall.function_calls`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveServerToolCall.function_calls)
  - [`LiveServerToolCallCancellation`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveServerToolCallCancellation)
    - [`LiveServerToolCallCancellation.ids`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveServerToolCallCancellation.ids)
  - [`LiveServerToolCallCancellationDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveServerToolCallCancellationDict)
    - [`LiveServerToolCallCancellationDict.ids`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveServerToolCallCancellationDict.ids)
  - [`LiveServerToolCallDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveServerToolCallDict)
    - [`LiveServerToolCallDict.function_calls`](https://googleapis.github.io/python-genai/genai.html#genai.types.LiveServerToolCallDict.function_calls)
  - [`LogprobsResult`](https://googleapis.github.io/python-genai/genai.html#genai.types.LogprobsResult)
    - [`LogprobsResult.chosen_candidates`](https://googleapis.github.io/python-genai/genai.html#genai.types.LogprobsResult.chosen_candidates)
    - [`LogprobsResult.top_candidates`](https://googleapis.github.io/python-genai/genai.html#genai.types.LogprobsResult.top_candidates)
  - [`LogprobsResultCandidate`](https://googleapis.github.io/python-genai/genai.html#genai.types.LogprobsResultCandidate)
    - [`LogprobsResultCandidate.log_probability`](https://googleapis.github.io/python-genai/genai.html#genai.types.LogprobsResultCandidate.log_probability)
    - [`LogprobsResultCandidate.token`](https://googleapis.github.io/python-genai/genai.html#genai.types.LogprobsResultCandidate.token)
    - [`LogprobsResultCandidate.token_id`](https://googleapis.github.io/python-genai/genai.html#genai.types.LogprobsResultCandidate.token_id)
  - [`LogprobsResultCandidateDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.LogprobsResultCandidateDict)
    - [`LogprobsResultCandidateDict.log_probability`](https://googleapis.github.io/python-genai/genai.html#genai.types.LogprobsResultCandidateDict.log_probability)
    - [`LogprobsResultCandidateDict.token`](https://googleapis.github.io/python-genai/genai.html#genai.types.LogprobsResultCandidateDict.token)
    - [`LogprobsResultCandidateDict.token_id`](https://googleapis.github.io/python-genai/genai.html#genai.types.LogprobsResultCandidateDict.token_id)
  - [`LogprobsResultDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.LogprobsResultDict)
    - [`LogprobsResultDict.chosen_candidates`](https://googleapis.github.io/python-genai/genai.html#genai.types.LogprobsResultDict.chosen_candidates)
    - [`LogprobsResultDict.top_candidates`](https://googleapis.github.io/python-genai/genai.html#genai.types.LogprobsResultDict.top_candidates)
  - [`LogprobsResultTopCandidates`](https://googleapis.github.io/python-genai/genai.html#genai.types.LogprobsResultTopCandidates)
    - [`LogprobsResultTopCandidates.candidates`](https://googleapis.github.io/python-genai/genai.html#genai.types.LogprobsResultTopCandidates.candidates)
  - [`LogprobsResultTopCandidatesDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.LogprobsResultTopCandidatesDict)
    - [`LogprobsResultTopCandidatesDict.candidates`](https://googleapis.github.io/python-genai/genai.html#genai.types.LogprobsResultTopCandidatesDict.candidates)
  - [`MaskReferenceConfig`](https://googleapis.github.io/python-genai/genai.html#genai.types.MaskReferenceConfig)
    - [`MaskReferenceConfig.mask_dilation`](https://googleapis.github.io/python-genai/genai.html#genai.types.MaskReferenceConfig.mask_dilation)
    - [`MaskReferenceConfig.mask_mode`](https://googleapis.github.io/python-genai/genai.html#genai.types.MaskReferenceConfig.mask_mode)
    - [`MaskReferenceConfig.segmentation_classes`](https://googleapis.github.io/python-genai/genai.html#genai.types.MaskReferenceConfig.segmentation_classes)
  - [`MaskReferenceConfigDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.MaskReferenceConfigDict)
    - [`MaskReferenceConfigDict.mask_dilation`](https://googleapis.github.io/python-genai/genai.html#genai.types.MaskReferenceConfigDict.mask_dilation)
    - [`MaskReferenceConfigDict.mask_mode`](https://googleapis.github.io/python-genai/genai.html#genai.types.MaskReferenceConfigDict.mask_mode)
    - [`MaskReferenceConfigDict.segmentation_classes`](https://googleapis.github.io/python-genai/genai.html#genai.types.MaskReferenceConfigDict.segmentation_classes)
  - [`MaskReferenceImage`](https://googleapis.github.io/python-genai/genai.html#genai.types.MaskReferenceImage)
    - [`MaskReferenceImage.config`](https://googleapis.github.io/python-genai/genai.html#genai.types.MaskReferenceImage.config)
    - [`MaskReferenceImage.mask_image_config`](https://googleapis.github.io/python-genai/genai.html#genai.types.MaskReferenceImage.mask_image_config)
    - [`MaskReferenceImage.reference_id`](https://googleapis.github.io/python-genai/genai.html#genai.types.MaskReferenceImage.reference_id)
    - [`MaskReferenceImage.reference_image`](https://googleapis.github.io/python-genai/genai.html#genai.types.MaskReferenceImage.reference_image)
    - [`MaskReferenceImage.reference_type`](https://googleapis.github.io/python-genai/genai.html#genai.types.MaskReferenceImage.reference_type)
  - [`MaskReferenceImageDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.MaskReferenceImageDict)
    - [`MaskReferenceImageDict.config`](https://googleapis.github.io/python-genai/genai.html#genai.types.MaskReferenceImageDict.config)
    - [`MaskReferenceImageDict.reference_id`](https://googleapis.github.io/python-genai/genai.html#genai.types.MaskReferenceImageDict.reference_id)
    - [`MaskReferenceImageDict.reference_image`](https://googleapis.github.io/python-genai/genai.html#genai.types.MaskReferenceImageDict.reference_image)
    - [`MaskReferenceImageDict.reference_type`](https://googleapis.github.io/python-genai/genai.html#genai.types.MaskReferenceImageDict.reference_type)
  - [`MaskReferenceMode`](https://googleapis.github.io/python-genai/genai.html#genai.types.MaskReferenceMode)
    - [`MaskReferenceMode.MASK_MODE_BACKGROUND`](https://googleapis.github.io/python-genai/genai.html#genai.types.MaskReferenceMode.MASK_MODE_BACKGROUND)
    - [`MaskReferenceMode.MASK_MODE_DEFAULT`](https://googleapis.github.io/python-genai/genai.html#genai.types.MaskReferenceMode.MASK_MODE_DEFAULT)
    - [`MaskReferenceMode.MASK_MODE_FOREGROUND`](https://googleapis.github.io/python-genai/genai.html#genai.types.MaskReferenceMode.MASK_MODE_FOREGROUND)
    - [`MaskReferenceMode.MASK_MODE_SEMANTIC`](https://googleapis.github.io/python-genai/genai.html#genai.types.MaskReferenceMode.MASK_MODE_SEMANTIC)
    - [`MaskReferenceMode.MASK_MODE_USER_PROVIDED`](https://googleapis.github.io/python-genai/genai.html#genai.types.MaskReferenceMode.MASK_MODE_USER_PROVIDED)
  - [`MediaModality`](https://googleapis.github.io/python-genai/genai.html#genai.types.MediaModality)
    - [`MediaModality.AUDIO`](https://googleapis.github.io/python-genai/genai.html#genai.types.MediaModality.AUDIO)
    - [`MediaModality.DOCUMENT`](https://googleapis.github.io/python-genai/genai.html#genai.types.MediaModality.DOCUMENT)
    - [`MediaModality.IMAGE`](https://googleapis.github.io/python-genai/genai.html#genai.types.MediaModality.IMAGE)
    - [`MediaModality.MODALITY_UNSPECIFIED`](https://googleapis.github.io/python-genai/genai.html#genai.types.MediaModality.MODALITY_UNSPECIFIED)
    - [`MediaModality.TEXT`](https://googleapis.github.io/python-genai/genai.html#genai.types.MediaModality.TEXT)
    - [`MediaModality.VIDEO`](https://googleapis.github.io/python-genai/genai.html#genai.types.MediaModality.VIDEO)
  - [`MediaResolution`](https://googleapis.github.io/python-genai/genai.html#genai.types.MediaResolution)
    - [`MediaResolution.MEDIA_RESOLUTION_HIGH`](https://googleapis.github.io/python-genai/genai.html#genai.types.MediaResolution.MEDIA_RESOLUTION_HIGH)
    - [`MediaResolution.MEDIA_RESOLUTION_LOW`](https://googleapis.github.io/python-genai/genai.html#genai.types.MediaResolution.MEDIA_RESOLUTION_LOW)
    - [`MediaResolution.MEDIA_RESOLUTION_MEDIUM`](https://googleapis.github.io/python-genai/genai.html#genai.types.MediaResolution.MEDIA_RESOLUTION_MEDIUM)
    - [`MediaResolution.MEDIA_RESOLUTION_UNSPECIFIED`](https://googleapis.github.io/python-genai/genai.html#genai.types.MediaResolution.MEDIA_RESOLUTION_UNSPECIFIED)
  - [`Modality`](https://googleapis.github.io/python-genai/genai.html#genai.types.Modality)
    - [`Modality.AUDIO`](https://googleapis.github.io/python-genai/genai.html#genai.types.Modality.AUDIO)
    - [`Modality.IMAGE`](https://googleapis.github.io/python-genai/genai.html#genai.types.Modality.IMAGE)
    - [`Modality.MODALITY_UNSPECIFIED`](https://googleapis.github.io/python-genai/genai.html#genai.types.Modality.MODALITY_UNSPECIFIED)
    - [`Modality.TEXT`](https://googleapis.github.io/python-genai/genai.html#genai.types.Modality.TEXT)
  - [`ModalityTokenCount`](https://googleapis.github.io/python-genai/genai.html#genai.types.ModalityTokenCount)
    - [`ModalityTokenCount.modality`](https://googleapis.github.io/python-genai/genai.html#genai.types.ModalityTokenCount.modality)
    - [`ModalityTokenCount.token_count`](https://googleapis.github.io/python-genai/genai.html#genai.types.ModalityTokenCount.token_count)
  - [`ModalityTokenCountDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.ModalityTokenCountDict)
    - [`ModalityTokenCountDict.modality`](https://googleapis.github.io/python-genai/genai.html#genai.types.ModalityTokenCountDict.modality)
    - [`ModalityTokenCountDict.token_count`](https://googleapis.github.io/python-genai/genai.html#genai.types.ModalityTokenCountDict.token_count)
  - [`Mode`](https://googleapis.github.io/python-genai/genai.html#genai.types.Mode)
    - [`Mode.MODE_DYNAMIC`](https://googleapis.github.io/python-genai/genai.html#genai.types.Mode.MODE_DYNAMIC)
    - [`Mode.MODE_UNSPECIFIED`](https://googleapis.github.io/python-genai/genai.html#genai.types.Mode.MODE_UNSPECIFIED)
  - [`Model`](https://googleapis.github.io/python-genai/genai.html#genai.types.Model)
    - [`Model.checkpoints`](https://googleapis.github.io/python-genai/genai.html#genai.types.Model.checkpoints)
    - [`Model.default_checkpoint_id`](https://googleapis.github.io/python-genai/genai.html#genai.types.Model.default_checkpoint_id)
    - [`Model.description`](https://googleapis.github.io/python-genai/genai.html#genai.types.Model.description)
    - [`Model.display_name`](https://googleapis.github.io/python-genai/genai.html#genai.types.Model.display_name)
    - [`Model.endpoints`](https://googleapis.github.io/python-genai/genai.html#genai.types.Model.endpoints)
    - [`Model.input_token_limit`](https://googleapis.github.io/python-genai/genai.html#genai.types.Model.input_token_limit)
    - [`Model.labels`](https://googleapis.github.io/python-genai/genai.html#genai.types.Model.labels)
    - [`Model.name`](https://googleapis.github.io/python-genai/genai.html#genai.types.Model.name)
    - [`Model.output_token_limit`](https://googleapis.github.io/python-genai/genai.html#genai.types.Model.output_token_limit)
    - [`Model.supported_actions`](https://googleapis.github.io/python-genai/genai.html#genai.types.Model.supported_actions)
    - [`Model.tuned_model_info`](https://googleapis.github.io/python-genai/genai.html#genai.types.Model.tuned_model_info)
    - [`Model.version`](https://googleapis.github.io/python-genai/genai.html#genai.types.Model.version)
  - [`ModelContent`](https://googleapis.github.io/python-genai/genai.html#genai.types.ModelContent)
    - [`ModelContent.parts`](https://googleapis.github.io/python-genai/genai.html#genai.types.ModelContent.parts)
    - [`ModelContent.role`](https://googleapis.github.io/python-genai/genai.html#genai.types.ModelContent.role)
  - [`ModelDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.ModelDict)
    - [`ModelDict.checkpoints`](https://googleapis.github.io/python-genai/genai.html#genai.types.ModelDict.checkpoints)
    - [`ModelDict.default_checkpoint_id`](https://googleapis.github.io/python-genai/genai.html#genai.types.ModelDict.default_checkpoint_id)
    - [`ModelDict.description`](https://googleapis.github.io/python-genai/genai.html#genai.types.ModelDict.description)
    - [`ModelDict.display_name`](https://googleapis.github.io/python-genai/genai.html#genai.types.ModelDict.display_name)
    - [`ModelDict.endpoints`](https://googleapis.github.io/python-genai/genai.html#genai.types.ModelDict.endpoints)
    - [`ModelDict.input_token_limit`](https://googleapis.github.io/python-genai/genai.html#genai.types.ModelDict.input_token_limit)
    - [`ModelDict.labels`](https://googleapis.github.io/python-genai/genai.html#genai.types.ModelDict.labels)
    - [`ModelDict.name`](https://googleapis.github.io/python-genai/genai.html#genai.types.ModelDict.name)
    - [`ModelDict.output_token_limit`](https://googleapis.github.io/python-genai/genai.html#genai.types.ModelDict.output_token_limit)
    - [`ModelDict.supported_actions`](https://googleapis.github.io/python-genai/genai.html#genai.types.ModelDict.supported_actions)
    - [`ModelDict.tuned_model_info`](https://googleapis.github.io/python-genai/genai.html#genai.types.ModelDict.tuned_model_info)
    - [`ModelDict.version`](https://googleapis.github.io/python-genai/genai.html#genai.types.ModelDict.version)
  - [`ModelSelectionConfig`](https://googleapis.github.io/python-genai/genai.html#genai.types.ModelSelectionConfig)
    - [`ModelSelectionConfig.feature_selection_preference`](https://googleapis.github.io/python-genai/genai.html#genai.types.ModelSelectionConfig.feature_selection_preference)
  - [`ModelSelectionConfigDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.ModelSelectionConfigDict)
    - [`ModelSelectionConfigDict.feature_selection_preference`](https://googleapis.github.io/python-genai/genai.html#genai.types.ModelSelectionConfigDict.feature_selection_preference)
  - [`MultiSpeakerVoiceConfig`](https://googleapis.github.io/python-genai/genai.html#genai.types.MultiSpeakerVoiceConfig)
    - [`MultiSpeakerVoiceConfig.speaker_voice_configs`](https://googleapis.github.io/python-genai/genai.html#genai.types.MultiSpeakerVoiceConfig.speaker_voice_configs)
  - [`MultiSpeakerVoiceConfigDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.MultiSpeakerVoiceConfigDict)
    - [`MultiSpeakerVoiceConfigDict.speaker_voice_configs`](https://googleapis.github.io/python-genai/genai.html#genai.types.MultiSpeakerVoiceConfigDict.speaker_voice_configs)
  - [`Operation`](https://googleapis.github.io/python-genai/genai.html#genai.types.Operation)
    - [`Operation.done`](https://googleapis.github.io/python-genai/genai.html#genai.types.Operation.done)
    - [`Operation.error`](https://googleapis.github.io/python-genai/genai.html#genai.types.Operation.error)
    - [`Operation.metadata`](https://googleapis.github.io/python-genai/genai.html#genai.types.Operation.metadata)
    - [`Operation.name`](https://googleapis.github.io/python-genai/genai.html#genai.types.Operation.name)
  - [`OperationDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.OperationDict)
    - [`OperationDict.done`](https://googleapis.github.io/python-genai/genai.html#genai.types.OperationDict.done)
    - [`OperationDict.error`](https://googleapis.github.io/python-genai/genai.html#genai.types.OperationDict.error)
    - [`OperationDict.metadata`](https://googleapis.github.io/python-genai/genai.html#genai.types.OperationDict.metadata)
    - [`OperationDict.name`](https://googleapis.github.io/python-genai/genai.html#genai.types.OperationDict.name)
  - [`Outcome`](https://googleapis.github.io/python-genai/genai.html#genai.types.Outcome)
    - [`Outcome.OUTCOME_DEADLINE_EXCEEDED`](https://googleapis.github.io/python-genai/genai.html#genai.types.Outcome.OUTCOME_DEADLINE_EXCEEDED)
    - [`Outcome.OUTCOME_FAILED`](https://googleapis.github.io/python-genai/genai.html#genai.types.Outcome.OUTCOME_FAILED)
    - [`Outcome.OUTCOME_OK`](https://googleapis.github.io/python-genai/genai.html#genai.types.Outcome.OUTCOME_OK)
    - [`Outcome.OUTCOME_UNSPECIFIED`](https://googleapis.github.io/python-genai/genai.html#genai.types.Outcome.OUTCOME_UNSPECIFIED)
  - [`Part`](https://googleapis.github.io/python-genai/genai.html#genai.types.Part)
    - [`Part.code_execution_result`](https://googleapis.github.io/python-genai/genai.html#genai.types.Part.code_execution_result)
    - [`Part.executable_code`](https://googleapis.github.io/python-genai/genai.html#genai.types.Part.executable_code)
    - [`Part.file_data`](https://googleapis.github.io/python-genai/genai.html#genai.types.Part.file_data)
    - [`Part.function_call`](https://googleapis.github.io/python-genai/genai.html#genai.types.Part.function_call)
    - [`Part.function_response`](https://googleapis.github.io/python-genai/genai.html#genai.types.Part.function_response)
    - [`Part.inline_data`](https://googleapis.github.io/python-genai/genai.html#genai.types.Part.inline_data)
    - [`Part.text`](https://googleapis.github.io/python-genai/genai.html#genai.types.Part.text)
    - [`Part.thought`](https://googleapis.github.io/python-genai/genai.html#genai.types.Part.thought)
    - [`Part.thought_signature`](https://googleapis.github.io/python-genai/genai.html#genai.types.Part.thought_signature)
    - [`Part.video_metadata`](https://googleapis.github.io/python-genai/genai.html#genai.types.Part.video_metadata)
    - [`Part.from_bytes()`](https://googleapis.github.io/python-genai/genai.html#genai.types.Part.from_bytes)
    - [`Part.from_code_execution_result()`](https://googleapis.github.io/python-genai/genai.html#genai.types.Part.from_code_execution_result)
    - [`Part.from_executable_code()`](https://googleapis.github.io/python-genai/genai.html#genai.types.Part.from_executable_code)
    - [`Part.from_function_call()`](https://googleapis.github.io/python-genai/genai.html#genai.types.Part.from_function_call)
    - [`Part.from_function_response()`](https://googleapis.github.io/python-genai/genai.html#genai.types.Part.from_function_response)
    - [`Part.from_text()`](https://googleapis.github.io/python-genai/genai.html#genai.types.Part.from_text)
    - [`Part.from_uri()`](https://googleapis.github.io/python-genai/genai.html#genai.types.Part.from_uri)
  - [`PartDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.PartDict)
    - [`PartDict.code_execution_result`](https://googleapis.github.io/python-genai/genai.html#genai.types.PartDict.code_execution_result)
    - [`PartDict.executable_code`](https://googleapis.github.io/python-genai/genai.html#genai.types.PartDict.executable_code)
    - [`PartDict.file_data`](https://googleapis.github.io/python-genai/genai.html#genai.types.PartDict.file_data)
    - [`PartDict.function_call`](https://googleapis.github.io/python-genai/genai.html#genai.types.PartDict.function_call)
    - [`PartDict.function_response`](https://googleapis.github.io/python-genai/genai.html#genai.types.PartDict.function_response)
    - [`PartDict.inline_data`](https://googleapis.github.io/python-genai/genai.html#genai.types.PartDict.inline_data)
    - [`PartDict.text`](https://googleapis.github.io/python-genai/genai.html#genai.types.PartDict.text)
    - [`PartDict.thought`](https://googleapis.github.io/python-genai/genai.html#genai.types.PartDict.thought)
    - [`PartDict.thought_signature`](https://googleapis.github.io/python-genai/genai.html#genai.types.PartDict.thought_signature)
    - [`PartDict.video_metadata`](https://googleapis.github.io/python-genai/genai.html#genai.types.PartDict.video_metadata)
  - [`PartnerModelTuningSpec`](https://googleapis.github.io/python-genai/genai.html#genai.types.PartnerModelTuningSpec)
    - [`PartnerModelTuningSpec.hyper_parameters`](https://googleapis.github.io/python-genai/genai.html#genai.types.PartnerModelTuningSpec.hyper_parameters)
    - [`PartnerModelTuningSpec.training_dataset_uri`](https://googleapis.github.io/python-genai/genai.html#genai.types.PartnerModelTuningSpec.training_dataset_uri)
    - [`PartnerModelTuningSpec.validation_dataset_uri`](https://googleapis.github.io/python-genai/genai.html#genai.types.PartnerModelTuningSpec.validation_dataset_uri)
  - [`PartnerModelTuningSpecDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.PartnerModelTuningSpecDict)
    - [`PartnerModelTuningSpecDict.hyper_parameters`](https://googleapis.github.io/python-genai/genai.html#genai.types.PartnerModelTuningSpecDict.hyper_parameters)
    - [`PartnerModelTuningSpecDict.training_dataset_uri`](https://googleapis.github.io/python-genai/genai.html#genai.types.PartnerModelTuningSpecDict.training_dataset_uri)
    - [`PartnerModelTuningSpecDict.validation_dataset_uri`](https://googleapis.github.io/python-genai/genai.html#genai.types.PartnerModelTuningSpecDict.validation_dataset_uri)
  - [`PersonGeneration`](https://googleapis.github.io/python-genai/genai.html#genai.types.PersonGeneration)
    - [`PersonGeneration.ALLOW_ADULT`](https://googleapis.github.io/python-genai/genai.html#genai.types.PersonGeneration.ALLOW_ADULT)
    - [`PersonGeneration.ALLOW_ALL`](https://googleapis.github.io/python-genai/genai.html#genai.types.PersonGeneration.ALLOW_ALL)
    - [`PersonGeneration.DONT_ALLOW`](https://googleapis.github.io/python-genai/genai.html#genai.types.PersonGeneration.DONT_ALLOW)
  - [`PrebuiltVoiceConfig`](https://googleapis.github.io/python-genai/genai.html#genai.types.PrebuiltVoiceConfig)
    - [`PrebuiltVoiceConfig.voice_name`](https://googleapis.github.io/python-genai/genai.html#genai.types.PrebuiltVoiceConfig.voice_name)
  - [`PrebuiltVoiceConfigDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.PrebuiltVoiceConfigDict)
    - [`PrebuiltVoiceConfigDict.voice_name`](https://googleapis.github.io/python-genai/genai.html#genai.types.PrebuiltVoiceConfigDict.voice_name)
  - [`ProactivityConfig`](https://googleapis.github.io/python-genai/genai.html#genai.types.ProactivityConfig)
    - [`ProactivityConfig.proactive_audio`](https://googleapis.github.io/python-genai/genai.html#genai.types.ProactivityConfig.proactive_audio)
  - [`ProactivityConfigDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.ProactivityConfigDict)
    - [`ProactivityConfigDict.proactive_audio`](https://googleapis.github.io/python-genai/genai.html#genai.types.ProactivityConfigDict.proactive_audio)
  - [`RagChunk`](https://googleapis.github.io/python-genai/genai.html#genai.types.RagChunk)
    - [`RagChunk.page_span`](https://googleapis.github.io/python-genai/genai.html#genai.types.RagChunk.page_span)
    - [`RagChunk.text`](https://googleapis.github.io/python-genai/genai.html#genai.types.RagChunk.text)
  - [`RagChunkDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.RagChunkDict)
    - [`RagChunkDict.page_span`](https://googleapis.github.io/python-genai/genai.html#genai.types.RagChunkDict.page_span)
    - [`RagChunkDict.text`](https://googleapis.github.io/python-genai/genai.html#genai.types.RagChunkDict.text)
  - [`RagChunkPageSpan`](https://googleapis.github.io/python-genai/genai.html#genai.types.RagChunkPageSpan)
    - [`RagChunkPageSpan.first_page`](https://googleapis.github.io/python-genai/genai.html#genai.types.RagChunkPageSpan.first_page)
    - [`RagChunkPageSpan.last_page`](https://googleapis.github.io/python-genai/genai.html#genai.types.RagChunkPageSpan.last_page)
  - [`RagChunkPageSpanDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.RagChunkPageSpanDict)
    - [`RagChunkPageSpanDict.first_page`](https://googleapis.github.io/python-genai/genai.html#genai.types.RagChunkPageSpanDict.first_page)
    - [`RagChunkPageSpanDict.last_page`](https://googleapis.github.io/python-genai/genai.html#genai.types.RagChunkPageSpanDict.last_page)
  - [`RagRetrievalConfig`](https://googleapis.github.io/python-genai/genai.html#genai.types.RagRetrievalConfig)
    - [`RagRetrievalConfig.filter`](https://googleapis.github.io/python-genai/genai.html#genai.types.RagRetrievalConfig.filter)
    - [`RagRetrievalConfig.hybrid_search`](https://googleapis.github.io/python-genai/genai.html#genai.types.RagRetrievalConfig.hybrid_search)
    - [`RagRetrievalConfig.ranking`](https://googleapis.github.io/python-genai/genai.html#genai.types.RagRetrievalConfig.ranking)
    - [`RagRetrievalConfig.top_k`](https://googleapis.github.io/python-genai/genai.html#genai.types.RagRetrievalConfig.top_k)
  - [`RagRetrievalConfigDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.RagRetrievalConfigDict)
    - [`RagRetrievalConfigDict.filter`](https://googleapis.github.io/python-genai/genai.html#genai.types.RagRetrievalConfigDict.filter)
    - [`RagRetrievalConfigDict.hybrid_search`](https://googleapis.github.io/python-genai/genai.html#genai.types.RagRetrievalConfigDict.hybrid_search)
    - [`RagRetrievalConfigDict.ranking`](https://googleapis.github.io/python-genai/genai.html#genai.types.RagRetrievalConfigDict.ranking)
    - [`RagRetrievalConfigDict.top_k`](https://googleapis.github.io/python-genai/genai.html#genai.types.RagRetrievalConfigDict.top_k)
  - [`RagRetrievalConfigFilter`](https://googleapis.github.io/python-genai/genai.html#genai.types.RagRetrievalConfigFilter)
    - [`RagRetrievalConfigFilter.metadata_filter`](https://googleapis.github.io/python-genai/genai.html#genai.types.RagRetrievalConfigFilter.metadata_filter)
    - [`RagRetrievalConfigFilter.vector_distance_threshold`](https://googleapis.github.io/python-genai/genai.html#genai.types.RagRetrievalConfigFilter.vector_distance_threshold)
    - [`RagRetrievalConfigFilter.vector_similarity_threshold`](https://googleapis.github.io/python-genai/genai.html#genai.types.RagRetrievalConfigFilter.vector_similarity_threshold)
  - [`RagRetrievalConfigFilterDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.RagRetrievalConfigFilterDict)
    - [`RagRetrievalConfigFilterDict.metadata_filter`](https://googleapis.github.io/python-genai/genai.html#genai.types.RagRetrievalConfigFilterDict.metadata_filter)
    - [`RagRetrievalConfigFilterDict.vector_distance_threshold`](https://googleapis.github.io/python-genai/genai.html#genai.types.RagRetrievalConfigFilterDict.vector_distance_threshold)
    - [`RagRetrievalConfigFilterDict.vector_similarity_threshold`](https://googleapis.github.io/python-genai/genai.html#genai.types.RagRetrievalConfigFilterDict.vector_similarity_threshold)
  - [`RagRetrievalConfigHybridSearch`](https://googleapis.github.io/python-genai/genai.html#genai.types.RagRetrievalConfigHybridSearch)
    - [`RagRetrievalConfigHybridSearch.alpha`](https://googleapis.github.io/python-genai/genai.html#genai.types.RagRetrievalConfigHybridSearch.alpha)
  - [`RagRetrievalConfigHybridSearchDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.RagRetrievalConfigHybridSearchDict)
    - [`RagRetrievalConfigHybridSearchDict.alpha`](https://googleapis.github.io/python-genai/genai.html#genai.types.RagRetrievalConfigHybridSearchDict.alpha)
  - [`RagRetrievalConfigRanking`](https://googleapis.github.io/python-genai/genai.html#genai.types.RagRetrievalConfigRanking)
    - [`RagRetrievalConfigRanking.llm_ranker`](https://googleapis.github.io/python-genai/genai.html#genai.types.RagRetrievalConfigRanking.llm_ranker)
    - [`RagRetrievalConfigRanking.rank_service`](https://googleapis.github.io/python-genai/genai.html#genai.types.RagRetrievalConfigRanking.rank_service)
  - [`RagRetrievalConfigRankingDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.RagRetrievalConfigRankingDict)
    - [`RagRetrievalConfigRankingDict.llm_ranker`](https://googleapis.github.io/python-genai/genai.html#genai.types.RagRetrievalConfigRankingDict.llm_ranker)
    - [`RagRetrievalConfigRankingDict.rank_service`](https://googleapis.github.io/python-genai/genai.html#genai.types.RagRetrievalConfigRankingDict.rank_service)
  - [`RagRetrievalConfigRankingLlmRanker`](https://googleapis.github.io/python-genai/genai.html#genai.types.RagRetrievalConfigRankingLlmRanker)
    - [`RagRetrievalConfigRankingLlmRanker.model_name`](https://googleapis.github.io/python-genai/genai.html#genai.types.RagRetrievalConfigRankingLlmRanker.model_name)
  - [`RagRetrievalConfigRankingLlmRankerDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.RagRetrievalConfigRankingLlmRankerDict)
    - [`RagRetrievalConfigRankingLlmRankerDict.model_name`](https://googleapis.github.io/python-genai/genai.html#genai.types.RagRetrievalConfigRankingLlmRankerDict.model_name)
  - [`RagRetrievalConfigRankingRankService`](https://googleapis.github.io/python-genai/genai.html#genai.types.RagRetrievalConfigRankingRankService)
    - [`RagRetrievalConfigRankingRankService.model_name`](https://googleapis.github.io/python-genai/genai.html#genai.types.RagRetrievalConfigRankingRankService.model_name)
  - [`RagRetrievalConfigRankingRankServiceDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.RagRetrievalConfigRankingRankServiceDict)
    - [`RagRetrievalConfigRankingRankServiceDict.model_name`](https://googleapis.github.io/python-genai/genai.html#genai.types.RagRetrievalConfigRankingRankServiceDict.model_name)
  - [`RawReferenceImage`](https://googleapis.github.io/python-genai/genai.html#genai.types.RawReferenceImage)
    - [`RawReferenceImage.reference_id`](https://googleapis.github.io/python-genai/genai.html#genai.types.RawReferenceImage.reference_id)
    - [`RawReferenceImage.reference_image`](https://googleapis.github.io/python-genai/genai.html#genai.types.RawReferenceImage.reference_image)
    - [`RawReferenceImage.reference_type`](https://googleapis.github.io/python-genai/genai.html#genai.types.RawReferenceImage.reference_type)
  - [`RawReferenceImageDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.RawReferenceImageDict)
    - [`RawReferenceImageDict.reference_id`](https://googleapis.github.io/python-genai/genai.html#genai.types.RawReferenceImageDict.reference_id)
    - [`RawReferenceImageDict.reference_image`](https://googleapis.github.io/python-genai/genai.html#genai.types.RawReferenceImageDict.reference_image)
    - [`RawReferenceImageDict.reference_type`](https://googleapis.github.io/python-genai/genai.html#genai.types.RawReferenceImageDict.reference_type)
  - [`RealtimeInputConfig`](https://googleapis.github.io/python-genai/genai.html#genai.types.RealtimeInputConfig)
    - [`RealtimeInputConfig.activity_handling`](https://googleapis.github.io/python-genai/genai.html#genai.types.RealtimeInputConfig.activity_handling)
    - [`RealtimeInputConfig.automatic_activity_detection`](https://googleapis.github.io/python-genai/genai.html#genai.types.RealtimeInputConfig.automatic_activity_detection)
    - [`RealtimeInputConfig.turn_coverage`](https://googleapis.github.io/python-genai/genai.html#genai.types.RealtimeInputConfig.turn_coverage)
  - [`RealtimeInputConfigDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.RealtimeInputConfigDict)
    - [`RealtimeInputConfigDict.activity_handling`](https://googleapis.github.io/python-genai/genai.html#genai.types.RealtimeInputConfigDict.activity_handling)
    - [`RealtimeInputConfigDict.automatic_activity_detection`](https://googleapis.github.io/python-genai/genai.html#genai.types.RealtimeInputConfigDict.automatic_activity_detection)
    - [`RealtimeInputConfigDict.turn_coverage`](https://googleapis.github.io/python-genai/genai.html#genai.types.RealtimeInputConfigDict.turn_coverage)
  - [`ReplayFile`](https://googleapis.github.io/python-genai/genai.html#genai.types.ReplayFile)
    - [`ReplayFile.interactions`](https://googleapis.github.io/python-genai/genai.html#genai.types.ReplayFile.interactions)
    - [`ReplayFile.replay_id`](https://googleapis.github.io/python-genai/genai.html#genai.types.ReplayFile.replay_id)
  - [`ReplayFileDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.ReplayFileDict)
    - [`ReplayFileDict.interactions`](https://googleapis.github.io/python-genai/genai.html#genai.types.ReplayFileDict.interactions)
    - [`ReplayFileDict.replay_id`](https://googleapis.github.io/python-genai/genai.html#genai.types.ReplayFileDict.replay_id)
  - [`ReplayInteraction`](https://googleapis.github.io/python-genai/genai.html#genai.types.ReplayInteraction)
    - [`ReplayInteraction.request`](https://googleapis.github.io/python-genai/genai.html#genai.types.ReplayInteraction.request)
    - [`ReplayInteraction.response`](https://googleapis.github.io/python-genai/genai.html#genai.types.ReplayInteraction.response)
  - [`ReplayInteractionDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.ReplayInteractionDict)
    - [`ReplayInteractionDict.request`](https://googleapis.github.io/python-genai/genai.html#genai.types.ReplayInteractionDict.request)
    - [`ReplayInteractionDict.response`](https://googleapis.github.io/python-genai/genai.html#genai.types.ReplayInteractionDict.response)
  - [`ReplayRequest`](https://googleapis.github.io/python-genai/genai.html#genai.types.ReplayRequest)
    - [`ReplayRequest.body_segments`](https://googleapis.github.io/python-genai/genai.html#genai.types.ReplayRequest.body_segments)
    - [`ReplayRequest.headers`](https://googleapis.github.io/python-genai/genai.html#genai.types.ReplayRequest.headers)
    - [`ReplayRequest.method`](https://googleapis.github.io/python-genai/genai.html#genai.types.ReplayRequest.method)
    - [`ReplayRequest.url`](https://googleapis.github.io/python-genai/genai.html#genai.types.ReplayRequest.url)
  - [`ReplayRequestDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.ReplayRequestDict)
    - [`ReplayRequestDict.body_segments`](https://googleapis.github.io/python-genai/genai.html#genai.types.ReplayRequestDict.body_segments)
    - [`ReplayRequestDict.headers`](https://googleapis.github.io/python-genai/genai.html#genai.types.ReplayRequestDict.headers)
    - [`ReplayRequestDict.method`](https://googleapis.github.io/python-genai/genai.html#genai.types.ReplayRequestDict.method)
    - [`ReplayRequestDict.url`](https://googleapis.github.io/python-genai/genai.html#genai.types.ReplayRequestDict.url)
  - [`ReplayResponse`](https://googleapis.github.io/python-genai/genai.html#genai.types.ReplayResponse)
    - [`ReplayResponse.body_segments`](https://googleapis.github.io/python-genai/genai.html#genai.types.ReplayResponse.body_segments)
    - [`ReplayResponse.headers`](https://googleapis.github.io/python-genai/genai.html#genai.types.ReplayResponse.headers)
    - [`ReplayResponse.sdk_response_segments`](https://googleapis.github.io/python-genai/genai.html#genai.types.ReplayResponse.sdk_response_segments)
    - [`ReplayResponse.status_code`](https://googleapis.github.io/python-genai/genai.html#genai.types.ReplayResponse.status_code)
  - [`ReplayResponseDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.ReplayResponseDict)
    - [`ReplayResponseDict.body_segments`](https://googleapis.github.io/python-genai/genai.html#genai.types.ReplayResponseDict.body_segments)
    - [`ReplayResponseDict.headers`](https://googleapis.github.io/python-genai/genai.html#genai.types.ReplayResponseDict.headers)
    - [`ReplayResponseDict.sdk_response_segments`](https://googleapis.github.io/python-genai/genai.html#genai.types.ReplayResponseDict.sdk_response_segments)
    - [`ReplayResponseDict.status_code`](https://googleapis.github.io/python-genai/genai.html#genai.types.ReplayResponseDict.status_code)
  - [`Retrieval`](https://googleapis.github.io/python-genai/genai.html#genai.types.Retrieval)
    - [`Retrieval.disable_attribution`](https://googleapis.github.io/python-genai/genai.html#genai.types.Retrieval.disable_attribution)
    - [`Retrieval.vertex_ai_search`](https://googleapis.github.io/python-genai/genai.html#genai.types.Retrieval.vertex_ai_search)
    - [`Retrieval.vertex_rag_store`](https://googleapis.github.io/python-genai/genai.html#genai.types.Retrieval.vertex_rag_store)
  - [`RetrievalConfig`](https://googleapis.github.io/python-genai/genai.html#genai.types.RetrievalConfig)
    - [`RetrievalConfig.language_code`](https://googleapis.github.io/python-genai/genai.html#genai.types.RetrievalConfig.language_code)
    - [`RetrievalConfig.lat_lng`](https://googleapis.github.io/python-genai/genai.html#genai.types.RetrievalConfig.lat_lng)
  - [`RetrievalConfigDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.RetrievalConfigDict)
    - [`RetrievalConfigDict.language_code`](https://googleapis.github.io/python-genai/genai.html#genai.types.RetrievalConfigDict.language_code)
    - [`RetrievalConfigDict.lat_lng`](https://googleapis.github.io/python-genai/genai.html#genai.types.RetrievalConfigDict.lat_lng)
  - [`RetrievalDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.RetrievalDict)
    - [`RetrievalDict.disable_attribution`](https://googleapis.github.io/python-genai/genai.html#genai.types.RetrievalDict.disable_attribution)
    - [`RetrievalDict.vertex_ai_search`](https://googleapis.github.io/python-genai/genai.html#genai.types.RetrievalDict.vertex_ai_search)
    - [`RetrievalDict.vertex_rag_store`](https://googleapis.github.io/python-genai/genai.html#genai.types.RetrievalDict.vertex_rag_store)
  - [`RetrievalMetadata`](https://googleapis.github.io/python-genai/genai.html#genai.types.RetrievalMetadata)
    - [`RetrievalMetadata.google_search_dynamic_retrieval_score`](https://googleapis.github.io/python-genai/genai.html#genai.types.RetrievalMetadata.google_search_dynamic_retrieval_score)
  - [`RetrievalMetadataDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.RetrievalMetadataDict)
    - [`RetrievalMetadataDict.google_search_dynamic_retrieval_score`](https://googleapis.github.io/python-genai/genai.html#genai.types.RetrievalMetadataDict.google_search_dynamic_retrieval_score)
  - [`SafetyAttributes`](https://googleapis.github.io/python-genai/genai.html#genai.types.SafetyAttributes)
    - [`SafetyAttributes.categories`](https://googleapis.github.io/python-genai/genai.html#genai.types.SafetyAttributes.categories)
    - [`SafetyAttributes.content_type`](https://googleapis.github.io/python-genai/genai.html#genai.types.SafetyAttributes.content_type)
    - [`SafetyAttributes.scores`](https://googleapis.github.io/python-genai/genai.html#genai.types.SafetyAttributes.scores)
  - [`SafetyAttributesDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.SafetyAttributesDict)
    - [`SafetyAttributesDict.categories`](https://googleapis.github.io/python-genai/genai.html#genai.types.SafetyAttributesDict.categories)
    - [`SafetyAttributesDict.content_type`](https://googleapis.github.io/python-genai/genai.html#genai.types.SafetyAttributesDict.content_type)
    - [`SafetyAttributesDict.scores`](https://googleapis.github.io/python-genai/genai.html#genai.types.SafetyAttributesDict.scores)
  - [`SafetyFilterLevel`](https://googleapis.github.io/python-genai/genai.html#genai.types.SafetyFilterLevel)
    - [`SafetyFilterLevel.BLOCK_LOW_AND_ABOVE`](https://googleapis.github.io/python-genai/genai.html#genai.types.SafetyFilterLevel.BLOCK_LOW_AND_ABOVE)
    - [`SafetyFilterLevel.BLOCK_MEDIUM_AND_ABOVE`](https://googleapis.github.io/python-genai/genai.html#genai.types.SafetyFilterLevel.BLOCK_MEDIUM_AND_ABOVE)
    - [`SafetyFilterLevel.BLOCK_NONE`](https://googleapis.github.io/python-genai/genai.html#genai.types.SafetyFilterLevel.BLOCK_NONE)
    - [`SafetyFilterLevel.BLOCK_ONLY_HIGH`](https://googleapis.github.io/python-genai/genai.html#genai.types.SafetyFilterLevel.BLOCK_ONLY_HIGH)
  - [`SafetyRating`](https://googleapis.github.io/python-genai/genai.html#genai.types.SafetyRating)
    - [`SafetyRating.blocked`](https://googleapis.github.io/python-genai/genai.html#genai.types.SafetyRating.blocked)
    - [`SafetyRating.category`](https://googleapis.github.io/python-genai/genai.html#genai.types.SafetyRating.category)
    - [`SafetyRating.probability`](https://googleapis.github.io/python-genai/genai.html#genai.types.SafetyRating.probability)
    - [`SafetyRating.probability_score`](https://googleapis.github.io/python-genai/genai.html#genai.types.SafetyRating.probability_score)
    - [`SafetyRating.severity`](https://googleapis.github.io/python-genai/genai.html#genai.types.SafetyRating.severity)
    - [`SafetyRating.severity_score`](https://googleapis.github.io/python-genai/genai.html#genai.types.SafetyRating.severity_score)
  - [`SafetyRatingDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.SafetyRatingDict)
    - [`SafetyRatingDict.blocked`](https://googleapis.github.io/python-genai/genai.html#genai.types.SafetyRatingDict.blocked)
    - [`SafetyRatingDict.category`](https://googleapis.github.io/python-genai/genai.html#genai.types.SafetyRatingDict.category)
    - [`SafetyRatingDict.probability`](https://googleapis.github.io/python-genai/genai.html#genai.types.SafetyRatingDict.probability)
    - [`SafetyRatingDict.probability_score`](https://googleapis.github.io/python-genai/genai.html#genai.types.SafetyRatingDict.probability_score)
    - [`SafetyRatingDict.severity`](https://googleapis.github.io/python-genai/genai.html#genai.types.SafetyRatingDict.severity)
    - [`SafetyRatingDict.severity_score`](https://googleapis.github.io/python-genai/genai.html#genai.types.SafetyRatingDict.severity_score)
  - [`SafetySetting`](https://googleapis.github.io/python-genai/genai.html#genai.types.SafetySetting)
    - [`SafetySetting.category`](https://googleapis.github.io/python-genai/genai.html#genai.types.SafetySetting.category)
    - [`SafetySetting.method`](https://googleapis.github.io/python-genai/genai.html#genai.types.SafetySetting.method)
    - [`SafetySetting.threshold`](https://googleapis.github.io/python-genai/genai.html#genai.types.SafetySetting.threshold)
  - [`SafetySettingDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.SafetySettingDict)
    - [`SafetySettingDict.category`](https://googleapis.github.io/python-genai/genai.html#genai.types.SafetySettingDict.category)
    - [`SafetySettingDict.method`](https://googleapis.github.io/python-genai/genai.html#genai.types.SafetySettingDict.method)
    - [`SafetySettingDict.threshold`](https://googleapis.github.io/python-genai/genai.html#genai.types.SafetySettingDict.threshold)
  - [`Scale`](https://googleapis.github.io/python-genai/genai.html#genai.types.Scale)
    - [`Scale.A_FLAT_MAJOR_F_MINOR`](https://googleapis.github.io/python-genai/genai.html#genai.types.Scale.A_FLAT_MAJOR_F_MINOR)
    - [`Scale.A_MAJOR_G_FLAT_MINOR`](https://googleapis.github.io/python-genai/genai.html#genai.types.Scale.A_MAJOR_G_FLAT_MINOR)
    - [`Scale.B_FLAT_MAJOR_G_MINOR`](https://googleapis.github.io/python-genai/genai.html#genai.types.Scale.B_FLAT_MAJOR_G_MINOR)
    - [`Scale.B_MAJOR_A_FLAT_MINOR`](https://googleapis.github.io/python-genai/genai.html#genai.types.Scale.B_MAJOR_A_FLAT_MINOR)
    - [`Scale.C_MAJOR_A_MINOR`](https://googleapis.github.io/python-genai/genai.html#genai.types.Scale.C_MAJOR_A_MINOR)
    - [`Scale.D_FLAT_MAJOR_B_FLAT_MINOR`](https://googleapis.github.io/python-genai/genai.html#genai.types.Scale.D_FLAT_MAJOR_B_FLAT_MINOR)
    - [`Scale.D_MAJOR_B_MINOR`](https://googleapis.github.io/python-genai/genai.html#genai.types.Scale.D_MAJOR_B_MINOR)
    - [`Scale.E_FLAT_MAJOR_C_MINOR`](https://googleapis.github.io/python-genai/genai.html#genai.types.Scale.E_FLAT_MAJOR_C_MINOR)
    - [`Scale.E_MAJOR_D_FLAT_MINOR`](https://googleapis.github.io/python-genai/genai.html#genai.types.Scale.E_MAJOR_D_FLAT_MINOR)
    - [`Scale.F_MAJOR_D_MINOR`](https://googleapis.github.io/python-genai/genai.html#genai.types.Scale.F_MAJOR_D_MINOR)
    - [`Scale.G_FLAT_MAJOR_E_FLAT_MINOR`](https://googleapis.github.io/python-genai/genai.html#genai.types.Scale.G_FLAT_MAJOR_E_FLAT_MINOR)
    - [`Scale.G_MAJOR_E_MINOR`](https://googleapis.github.io/python-genai/genai.html#genai.types.Scale.G_MAJOR_E_MINOR)
    - [`Scale.SCALE_UNSPECIFIED`](https://googleapis.github.io/python-genai/genai.html#genai.types.Scale.SCALE_UNSPECIFIED)
  - [`Schema`](https://googleapis.github.io/python-genai/genai.html#genai.types.Schema)
    - [`Schema.additional_properties`](https://googleapis.github.io/python-genai/genai.html#genai.types.Schema.additional_properties)
    - [`Schema.any_of`](https://googleapis.github.io/python-genai/genai.html#genai.types.Schema.any_of)
    - [`Schema.default`](https://googleapis.github.io/python-genai/genai.html#genai.types.Schema.default)
    - [`Schema.defs`](https://googleapis.github.io/python-genai/genai.html#genai.types.Schema.defs)
    - [`Schema.description`](https://googleapis.github.io/python-genai/genai.html#genai.types.Schema.description)
    - [`Schema.enum`](https://googleapis.github.io/python-genai/genai.html#genai.types.Schema.enum)
    - [`Schema.example`](https://googleapis.github.io/python-genai/genai.html#genai.types.Schema.example)
    - [`Schema.format`](https://googleapis.github.io/python-genai/genai.html#genai.types.Schema.format)
    - [`Schema.items`](https://googleapis.github.io/python-genai/genai.html#genai.types.Schema.items)
    - [`Schema.max_items`](https://googleapis.github.io/python-genai/genai.html#genai.types.Schema.max_items)
    - [`Schema.max_length`](https://googleapis.github.io/python-genai/genai.html#genai.types.Schema.max_length)
    - [`Schema.max_properties`](https://googleapis.github.io/python-genai/genai.html#genai.types.Schema.max_properties)
    - [`Schema.maximum`](https://googleapis.github.io/python-genai/genai.html#genai.types.Schema.maximum)
    - [`Schema.min_items`](https://googleapis.github.io/python-genai/genai.html#genai.types.Schema.min_items)
    - [`Schema.min_length`](https://googleapis.github.io/python-genai/genai.html#genai.types.Schema.min_length)
    - [`Schema.min_properties`](https://googleapis.github.io/python-genai/genai.html#genai.types.Schema.min_properties)
    - [`Schema.minimum`](https://googleapis.github.io/python-genai/genai.html#genai.types.Schema.minimum)
    - [`Schema.nullable`](https://googleapis.github.io/python-genai/genai.html#genai.types.Schema.nullable)
    - [`Schema.pattern`](https://googleapis.github.io/python-genai/genai.html#genai.types.Schema.pattern)
    - [`Schema.properties`](https://googleapis.github.io/python-genai/genai.html#genai.types.Schema.properties)
    - [`Schema.property_ordering`](https://googleapis.github.io/python-genai/genai.html#genai.types.Schema.property_ordering)
    - [`Schema.ref`](https://googleapis.github.io/python-genai/genai.html#genai.types.Schema.ref)
    - [`Schema.required`](https://googleapis.github.io/python-genai/genai.html#genai.types.Schema.required)
    - [`Schema.title`](https://googleapis.github.io/python-genai/genai.html#genai.types.Schema.title)
    - [`Schema.type`](https://googleapis.github.io/python-genai/genai.html#genai.types.Schema.type)
    - [`Schema.from_json_schema()`](https://googleapis.github.io/python-genai/genai.html#genai.types.Schema.from_json_schema)
    - [`Schema.json_schema`](https://googleapis.github.io/python-genai/genai.html#genai.types.Schema.json_schema)
  - [`SchemaDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.SchemaDict)
    - [`SchemaDict.additional_properties`](https://googleapis.github.io/python-genai/genai.html#genai.types.SchemaDict.additional_properties)
    - [`SchemaDict.any_of`](https://googleapis.github.io/python-genai/genai.html#genai.types.SchemaDict.any_of)
    - [`SchemaDict.default`](https://googleapis.github.io/python-genai/genai.html#genai.types.SchemaDict.default)
    - [`SchemaDict.defs`](https://googleapis.github.io/python-genai/genai.html#genai.types.SchemaDict.defs)
    - [`SchemaDict.description`](https://googleapis.github.io/python-genai/genai.html#genai.types.SchemaDict.description)
    - [`SchemaDict.enum`](https://googleapis.github.io/python-genai/genai.html#genai.types.SchemaDict.enum)
    - [`SchemaDict.example`](https://googleapis.github.io/python-genai/genai.html#genai.types.SchemaDict.example)
    - [`SchemaDict.format`](https://googleapis.github.io/python-genai/genai.html#genai.types.SchemaDict.format)
    - [`SchemaDict.max_items`](https://googleapis.github.io/python-genai/genai.html#genai.types.SchemaDict.max_items)
    - [`SchemaDict.max_length`](https://googleapis.github.io/python-genai/genai.html#genai.types.SchemaDict.max_length)
    - [`SchemaDict.max_properties`](https://googleapis.github.io/python-genai/genai.html#genai.types.SchemaDict.max_properties)
    - [`SchemaDict.maximum`](https://googleapis.github.io/python-genai/genai.html#genai.types.SchemaDict.maximum)
    - [`SchemaDict.min_items`](https://googleapis.github.io/python-genai/genai.html#genai.types.SchemaDict.min_items)
    - [`SchemaDict.min_length`](https://googleapis.github.io/python-genai/genai.html#genai.types.SchemaDict.min_length)
    - [`SchemaDict.min_properties`](https://googleapis.github.io/python-genai/genai.html#genai.types.SchemaDict.min_properties)
    - [`SchemaDict.minimum`](https://googleapis.github.io/python-genai/genai.html#genai.types.SchemaDict.minimum)
    - [`SchemaDict.nullable`](https://googleapis.github.io/python-genai/genai.html#genai.types.SchemaDict.nullable)
    - [`SchemaDict.pattern`](https://googleapis.github.io/python-genai/genai.html#genai.types.SchemaDict.pattern)
    - [`SchemaDict.properties`](https://googleapis.github.io/python-genai/genai.html#genai.types.SchemaDict.properties)
    - [`SchemaDict.property_ordering`](https://googleapis.github.io/python-genai/genai.html#genai.types.SchemaDict.property_ordering)
    - [`SchemaDict.ref`](https://googleapis.github.io/python-genai/genai.html#genai.types.SchemaDict.ref)
    - [`SchemaDict.required`](https://googleapis.github.io/python-genai/genai.html#genai.types.SchemaDict.required)
    - [`SchemaDict.title`](https://googleapis.github.io/python-genai/genai.html#genai.types.SchemaDict.title)
    - [`SchemaDict.type`](https://googleapis.github.io/python-genai/genai.html#genai.types.SchemaDict.type)
  - [`SearchEntryPoint`](https://googleapis.github.io/python-genai/genai.html#genai.types.SearchEntryPoint)
    - [`SearchEntryPoint.rendered_content`](https://googleapis.github.io/python-genai/genai.html#genai.types.SearchEntryPoint.rendered_content)
    - [`SearchEntryPoint.sdk_blob`](https://googleapis.github.io/python-genai/genai.html#genai.types.SearchEntryPoint.sdk_blob)
  - [`SearchEntryPointDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.SearchEntryPointDict)
    - [`SearchEntryPointDict.rendered_content`](https://googleapis.github.io/python-genai/genai.html#genai.types.SearchEntryPointDict.rendered_content)
    - [`SearchEntryPointDict.sdk_blob`](https://googleapis.github.io/python-genai/genai.html#genai.types.SearchEntryPointDict.sdk_blob)
  - [`Segment`](https://googleapis.github.io/python-genai/genai.html#genai.types.Segment)
    - [`Segment.end_index`](https://googleapis.github.io/python-genai/genai.html#genai.types.Segment.end_index)
    - [`Segment.part_index`](https://googleapis.github.io/python-genai/genai.html#genai.types.Segment.part_index)
    - [`Segment.start_index`](https://googleapis.github.io/python-genai/genai.html#genai.types.Segment.start_index)
    - [`Segment.text`](https://googleapis.github.io/python-genai/genai.html#genai.types.Segment.text)
  - [`SegmentDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.SegmentDict)
    - [`SegmentDict.end_index`](https://googleapis.github.io/python-genai/genai.html#genai.types.SegmentDict.end_index)
    - [`SegmentDict.part_index`](https://googleapis.github.io/python-genai/genai.html#genai.types.SegmentDict.part_index)
    - [`SegmentDict.start_index`](https://googleapis.github.io/python-genai/genai.html#genai.types.SegmentDict.start_index)
    - [`SegmentDict.text`](https://googleapis.github.io/python-genai/genai.html#genai.types.SegmentDict.text)
  - [`SessionResumptionConfig`](https://googleapis.github.io/python-genai/genai.html#genai.types.SessionResumptionConfig)
    - [`SessionResumptionConfig.handle`](https://googleapis.github.io/python-genai/genai.html#genai.types.SessionResumptionConfig.handle)
    - [`SessionResumptionConfig.transparent`](https://googleapis.github.io/python-genai/genai.html#genai.types.SessionResumptionConfig.transparent)
  - [`SessionResumptionConfigDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.SessionResumptionConfigDict)
    - [`SessionResumptionConfigDict.handle`](https://googleapis.github.io/python-genai/genai.html#genai.types.SessionResumptionConfigDict.handle)
    - [`SessionResumptionConfigDict.transparent`](https://googleapis.github.io/python-genai/genai.html#genai.types.SessionResumptionConfigDict.transparent)
  - [`SlidingWindow`](https://googleapis.github.io/python-genai/genai.html#genai.types.SlidingWindow)
    - [`SlidingWindow.target_tokens`](https://googleapis.github.io/python-genai/genai.html#genai.types.SlidingWindow.target_tokens)
  - [`SlidingWindowDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.SlidingWindowDict)
    - [`SlidingWindowDict.target_tokens`](https://googleapis.github.io/python-genai/genai.html#genai.types.SlidingWindowDict.target_tokens)
  - [`SpeakerVoiceConfig`](https://googleapis.github.io/python-genai/genai.html#genai.types.SpeakerVoiceConfig)
    - [`SpeakerVoiceConfig.speaker`](https://googleapis.github.io/python-genai/genai.html#genai.types.SpeakerVoiceConfig.speaker)
    - [`SpeakerVoiceConfig.voice_config`](https://googleapis.github.io/python-genai/genai.html#genai.types.SpeakerVoiceConfig.voice_config)
  - [`SpeakerVoiceConfigDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.SpeakerVoiceConfigDict)
    - [`SpeakerVoiceConfigDict.speaker`](https://googleapis.github.io/python-genai/genai.html#genai.types.SpeakerVoiceConfigDict.speaker)
    - [`SpeakerVoiceConfigDict.voice_config`](https://googleapis.github.io/python-genai/genai.html#genai.types.SpeakerVoiceConfigDict.voice_config)
  - [`SpeechConfig`](https://googleapis.github.io/python-genai/genai.html#genai.types.SpeechConfig)
    - [`SpeechConfig.language_code`](https://googleapis.github.io/python-genai/genai.html#genai.types.SpeechConfig.language_code)
    - [`SpeechConfig.multi_speaker_voice_config`](https://googleapis.github.io/python-genai/genai.html#genai.types.SpeechConfig.multi_speaker_voice_config)
    - [`SpeechConfig.voice_config`](https://googleapis.github.io/python-genai/genai.html#genai.types.SpeechConfig.voice_config)
  - [`SpeechConfigDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.SpeechConfigDict)
    - [`SpeechConfigDict.language_code`](https://googleapis.github.io/python-genai/genai.html#genai.types.SpeechConfigDict.language_code)
    - [`SpeechConfigDict.multi_speaker_voice_config`](https://googleapis.github.io/python-genai/genai.html#genai.types.SpeechConfigDict.multi_speaker_voice_config)
    - [`SpeechConfigDict.voice_config`](https://googleapis.github.io/python-genai/genai.html#genai.types.SpeechConfigDict.voice_config)
  - [`StartSensitivity`](https://googleapis.github.io/python-genai/genai.html#genai.types.StartSensitivity)
    - [`StartSensitivity.START_SENSITIVITY_HIGH`](https://googleapis.github.io/python-genai/genai.html#genai.types.StartSensitivity.START_SENSITIVITY_HIGH)
    - [`StartSensitivity.START_SENSITIVITY_LOW`](https://googleapis.github.io/python-genai/genai.html#genai.types.StartSensitivity.START_SENSITIVITY_LOW)
    - [`StartSensitivity.START_SENSITIVITY_UNSPECIFIED`](https://googleapis.github.io/python-genai/genai.html#genai.types.StartSensitivity.START_SENSITIVITY_UNSPECIFIED)
  - [`StyleReferenceConfig`](https://googleapis.github.io/python-genai/genai.html#genai.types.StyleReferenceConfig)
    - [`StyleReferenceConfig.style_description`](https://googleapis.github.io/python-genai/genai.html#genai.types.StyleReferenceConfig.style_description)
  - [`StyleReferenceConfigDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.StyleReferenceConfigDict)
    - [`StyleReferenceConfigDict.style_description`](https://googleapis.github.io/python-genai/genai.html#genai.types.StyleReferenceConfigDict.style_description)
  - [`StyleReferenceImage`](https://googleapis.github.io/python-genai/genai.html#genai.types.StyleReferenceImage)
    - [`StyleReferenceImage.config`](https://googleapis.github.io/python-genai/genai.html#genai.types.StyleReferenceImage.config)
    - [`StyleReferenceImage.reference_id`](https://googleapis.github.io/python-genai/genai.html#genai.types.StyleReferenceImage.reference_id)
    - [`StyleReferenceImage.reference_image`](https://googleapis.github.io/python-genai/genai.html#genai.types.StyleReferenceImage.reference_image)
    - [`StyleReferenceImage.reference_type`](https://googleapis.github.io/python-genai/genai.html#genai.types.StyleReferenceImage.reference_type)
    - [`StyleReferenceImage.style_image_config`](https://googleapis.github.io/python-genai/genai.html#genai.types.StyleReferenceImage.style_image_config)
  - [`StyleReferenceImageDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.StyleReferenceImageDict)
    - [`StyleReferenceImageDict.config`](https://googleapis.github.io/python-genai/genai.html#genai.types.StyleReferenceImageDict.config)
    - [`StyleReferenceImageDict.reference_id`](https://googleapis.github.io/python-genai/genai.html#genai.types.StyleReferenceImageDict.reference_id)
    - [`StyleReferenceImageDict.reference_image`](https://googleapis.github.io/python-genai/genai.html#genai.types.StyleReferenceImageDict.reference_image)
    - [`StyleReferenceImageDict.reference_type`](https://googleapis.github.io/python-genai/genai.html#genai.types.StyleReferenceImageDict.reference_type)
  - [`SubjectReferenceConfig`](https://googleapis.github.io/python-genai/genai.html#genai.types.SubjectReferenceConfig)
    - [`SubjectReferenceConfig.subject_description`](https://googleapis.github.io/python-genai/genai.html#genai.types.SubjectReferenceConfig.subject_description)
    - [`SubjectReferenceConfig.subject_type`](https://googleapis.github.io/python-genai/genai.html#genai.types.SubjectReferenceConfig.subject_type)
  - [`SubjectReferenceConfigDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.SubjectReferenceConfigDict)
    - [`SubjectReferenceConfigDict.subject_description`](https://googleapis.github.io/python-genai/genai.html#genai.types.SubjectReferenceConfigDict.subject_description)
    - [`SubjectReferenceConfigDict.subject_type`](https://googleapis.github.io/python-genai/genai.html#genai.types.SubjectReferenceConfigDict.subject_type)
  - [`SubjectReferenceImage`](https://googleapis.github.io/python-genai/genai.html#genai.types.SubjectReferenceImage)
    - [`SubjectReferenceImage.config`](https://googleapis.github.io/python-genai/genai.html#genai.types.SubjectReferenceImage.config)
    - [`SubjectReferenceImage.reference_id`](https://googleapis.github.io/python-genai/genai.html#genai.types.SubjectReferenceImage.reference_id)
    - [`SubjectReferenceImage.reference_image`](https://googleapis.github.io/python-genai/genai.html#genai.types.SubjectReferenceImage.reference_image)
    - [`SubjectReferenceImage.reference_type`](https://googleapis.github.io/python-genai/genai.html#genai.types.SubjectReferenceImage.reference_type)
    - [`SubjectReferenceImage.subject_image_config`](https://googleapis.github.io/python-genai/genai.html#genai.types.SubjectReferenceImage.subject_image_config)
  - [`SubjectReferenceImageDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.SubjectReferenceImageDict)
    - [`SubjectReferenceImageDict.config`](https://googleapis.github.io/python-genai/genai.html#genai.types.SubjectReferenceImageDict.config)
    - [`SubjectReferenceImageDict.reference_id`](https://googleapis.github.io/python-genai/genai.html#genai.types.SubjectReferenceImageDict.reference_id)
    - [`SubjectReferenceImageDict.reference_image`](https://googleapis.github.io/python-genai/genai.html#genai.types.SubjectReferenceImageDict.reference_image)
    - [`SubjectReferenceImageDict.reference_type`](https://googleapis.github.io/python-genai/genai.html#genai.types.SubjectReferenceImageDict.reference_type)
  - [`SubjectReferenceType`](https://googleapis.github.io/python-genai/genai.html#genai.types.SubjectReferenceType)
    - [`SubjectReferenceType.SUBJECT_TYPE_ANIMAL`](https://googleapis.github.io/python-genai/genai.html#genai.types.SubjectReferenceType.SUBJECT_TYPE_ANIMAL)
    - [`SubjectReferenceType.SUBJECT_TYPE_DEFAULT`](https://googleapis.github.io/python-genai/genai.html#genai.types.SubjectReferenceType.SUBJECT_TYPE_DEFAULT)
    - [`SubjectReferenceType.SUBJECT_TYPE_PERSON`](https://googleapis.github.io/python-genai/genai.html#genai.types.SubjectReferenceType.SUBJECT_TYPE_PERSON)
    - [`SubjectReferenceType.SUBJECT_TYPE_PRODUCT`](https://googleapis.github.io/python-genai/genai.html#genai.types.SubjectReferenceType.SUBJECT_TYPE_PRODUCT)
  - [`SupervisedHyperParameters`](https://googleapis.github.io/python-genai/genai.html#genai.types.SupervisedHyperParameters)
    - [`SupervisedHyperParameters.adapter_size`](https://googleapis.github.io/python-genai/genai.html#genai.types.SupervisedHyperParameters.adapter_size)
    - [`SupervisedHyperParameters.epoch_count`](https://googleapis.github.io/python-genai/genai.html#genai.types.SupervisedHyperParameters.epoch_count)
    - [`SupervisedHyperParameters.learning_rate_multiplier`](https://googleapis.github.io/python-genai/genai.html#genai.types.SupervisedHyperParameters.learning_rate_multiplier)
  - [`SupervisedHyperParametersDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.SupervisedHyperParametersDict)
    - [`SupervisedHyperParametersDict.adapter_size`](https://googleapis.github.io/python-genai/genai.html#genai.types.SupervisedHyperParametersDict.adapter_size)
    - [`SupervisedHyperParametersDict.epoch_count`](https://googleapis.github.io/python-genai/genai.html#genai.types.SupervisedHyperParametersDict.epoch_count)
    - [`SupervisedHyperParametersDict.learning_rate_multiplier`](https://googleapis.github.io/python-genai/genai.html#genai.types.SupervisedHyperParametersDict.learning_rate_multiplier)
  - [`SupervisedTuningDataStats`](https://googleapis.github.io/python-genai/genai.html#genai.types.SupervisedTuningDataStats)
    - [`SupervisedTuningDataStats.dropped_example_reasons`](https://googleapis.github.io/python-genai/genai.html#genai.types.SupervisedTuningDataStats.dropped_example_reasons)
    - [`SupervisedTuningDataStats.total_billable_character_count`](https://googleapis.github.io/python-genai/genai.html#genai.types.SupervisedTuningDataStats.total_billable_character_count)
    - [`SupervisedTuningDataStats.total_billable_token_count`](https://googleapis.github.io/python-genai/genai.html#genai.types.SupervisedTuningDataStats.total_billable_token_count)
    - [`SupervisedTuningDataStats.total_truncated_example_count`](https://googleapis.github.io/python-genai/genai.html#genai.types.SupervisedTuningDataStats.total_truncated_example_count)
    - [`SupervisedTuningDataStats.total_tuning_character_count`](https://googleapis.github.io/python-genai/genai.html#genai.types.SupervisedTuningDataStats.total_tuning_character_count)
    - [`SupervisedTuningDataStats.truncated_example_indices`](https://googleapis.github.io/python-genai/genai.html#genai.types.SupervisedTuningDataStats.truncated_example_indices)
    - [`SupervisedTuningDataStats.tuning_dataset_example_count`](https://googleapis.github.io/python-genai/genai.html#genai.types.SupervisedTuningDataStats.tuning_dataset_example_count)
    - [`SupervisedTuningDataStats.tuning_step_count`](https://googleapis.github.io/python-genai/genai.html#genai.types.SupervisedTuningDataStats.tuning_step_count)
    - [`SupervisedTuningDataStats.user_dataset_examples`](https://googleapis.github.io/python-genai/genai.html#genai.types.SupervisedTuningDataStats.user_dataset_examples)
    - [`SupervisedTuningDataStats.user_input_token_distribution`](https://googleapis.github.io/python-genai/genai.html#genai.types.SupervisedTuningDataStats.user_input_token_distribution)
    - [`SupervisedTuningDataStats.user_message_per_example_distribution`](https://googleapis.github.io/python-genai/genai.html#genai.types.SupervisedTuningDataStats.user_message_per_example_distribution)
    - [`SupervisedTuningDataStats.user_output_token_distribution`](https://googleapis.github.io/python-genai/genai.html#genai.types.SupervisedTuningDataStats.user_output_token_distribution)
  - [`SupervisedTuningDataStatsDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.SupervisedTuningDataStatsDict)
    - [`SupervisedTuningDataStatsDict.dropped_example_reasons`](https://googleapis.github.io/python-genai/genai.html#genai.types.SupervisedTuningDataStatsDict.dropped_example_reasons)
    - [`SupervisedTuningDataStatsDict.total_billable_character_count`](https://googleapis.github.io/python-genai/genai.html#genai.types.SupervisedTuningDataStatsDict.total_billable_character_count)
    - [`SupervisedTuningDataStatsDict.total_billable_token_count`](https://googleapis.github.io/python-genai/genai.html#genai.types.SupervisedTuningDataStatsDict.total_billable_token_count)
    - [`SupervisedTuningDataStatsDict.total_truncated_example_count`](https://googleapis.github.io/python-genai/genai.html#genai.types.SupervisedTuningDataStatsDict.total_truncated_example_count)
    - [`SupervisedTuningDataStatsDict.total_tuning_character_count`](https://googleapis.github.io/python-genai/genai.html#genai.types.SupervisedTuningDataStatsDict.total_tuning_character_count)
    - [`SupervisedTuningDataStatsDict.truncated_example_indices`](https://googleapis.github.io/python-genai/genai.html#genai.types.SupervisedTuningDataStatsDict.truncated_example_indices)
    - [`SupervisedTuningDataStatsDict.tuning_dataset_example_count`](https://googleapis.github.io/python-genai/genai.html#genai.types.SupervisedTuningDataStatsDict.tuning_dataset_example_count)
    - [`SupervisedTuningDataStatsDict.tuning_step_count`](https://googleapis.github.io/python-genai/genai.html#genai.types.SupervisedTuningDataStatsDict.tuning_step_count)
    - [`SupervisedTuningDataStatsDict.user_dataset_examples`](https://googleapis.github.io/python-genai/genai.html#genai.types.SupervisedTuningDataStatsDict.user_dataset_examples)
    - [`SupervisedTuningDataStatsDict.user_input_token_distribution`](https://googleapis.github.io/python-genai/genai.html#genai.types.SupervisedTuningDataStatsDict.user_input_token_distribution)
    - [`SupervisedTuningDataStatsDict.user_message_per_example_distribution`](https://googleapis.github.io/python-genai/genai.html#genai.types.SupervisedTuningDataStatsDict.user_message_per_example_distribution)
    - [`SupervisedTuningDataStatsDict.user_output_token_distribution`](https://googleapis.github.io/python-genai/genai.html#genai.types.SupervisedTuningDataStatsDict.user_output_token_distribution)
  - [`SupervisedTuningDatasetDistribution`](https://googleapis.github.io/python-genai/genai.html#genai.types.SupervisedTuningDatasetDistribution)
    - [`SupervisedTuningDatasetDistribution.billable_sum`](https://googleapis.github.io/python-genai/genai.html#genai.types.SupervisedTuningDatasetDistribution.billable_sum)
    - [`SupervisedTuningDatasetDistribution.buckets`](https://googleapis.github.io/python-genai/genai.html#genai.types.SupervisedTuningDatasetDistribution.buckets)
    - [`SupervisedTuningDatasetDistribution.max`](https://googleapis.github.io/python-genai/genai.html#genai.types.SupervisedTuningDatasetDistribution.max)
    - [`SupervisedTuningDatasetDistribution.mean`](https://googleapis.github.io/python-genai/genai.html#genai.types.SupervisedTuningDatasetDistribution.mean)
    - [`SupervisedTuningDatasetDistribution.median`](https://googleapis.github.io/python-genai/genai.html#genai.types.SupervisedTuningDatasetDistribution.median)
    - [`SupervisedTuningDatasetDistribution.min`](https://googleapis.github.io/python-genai/genai.html#genai.types.SupervisedTuningDatasetDistribution.min)
    - [`SupervisedTuningDatasetDistribution.p5`](https://googleapis.github.io/python-genai/genai.html#genai.types.SupervisedTuningDatasetDistribution.p5)
    - [`SupervisedTuningDatasetDistribution.p95`](https://googleapis.github.io/python-genai/genai.html#genai.types.SupervisedTuningDatasetDistribution.p95)
    - [`SupervisedTuningDatasetDistribution.sum`](https://googleapis.github.io/python-genai/genai.html#genai.types.SupervisedTuningDatasetDistribution.sum)
  - [`SupervisedTuningDatasetDistributionDatasetBucket`](https://googleapis.github.io/python-genai/genai.html#genai.types.SupervisedTuningDatasetDistributionDatasetBucket)
    - [`SupervisedTuningDatasetDistributionDatasetBucket.count`](https://googleapis.github.io/python-genai/genai.html#genai.types.SupervisedTuningDatasetDistributionDatasetBucket.count)
    - [`SupervisedTuningDatasetDistributionDatasetBucket.left`](https://googleapis.github.io/python-genai/genai.html#genai.types.SupervisedTuningDatasetDistributionDatasetBucket.left)
    - [`SupervisedTuningDatasetDistributionDatasetBucket.right`](https://googleapis.github.io/python-genai/genai.html#genai.types.SupervisedTuningDatasetDistributionDatasetBucket.right)
  - [`SupervisedTuningDatasetDistributionDatasetBucketDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.SupervisedTuningDatasetDistributionDatasetBucketDict)
    - [`SupervisedTuningDatasetDistributionDatasetBucketDict.count`](https://googleapis.github.io/python-genai/genai.html#genai.types.SupervisedTuningDatasetDistributionDatasetBucketDict.count)
    - [`SupervisedTuningDatasetDistributionDatasetBucketDict.left`](https://googleapis.github.io/python-genai/genai.html#genai.types.SupervisedTuningDatasetDistributionDatasetBucketDict.left)
    - [`SupervisedTuningDatasetDistributionDatasetBucketDict.right`](https://googleapis.github.io/python-genai/genai.html#genai.types.SupervisedTuningDatasetDistributionDatasetBucketDict.right)
  - [`SupervisedTuningDatasetDistributionDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.SupervisedTuningDatasetDistributionDict)
    - [`SupervisedTuningDatasetDistributionDict.billable_sum`](https://googleapis.github.io/python-genai/genai.html#genai.types.SupervisedTuningDatasetDistributionDict.billable_sum)
    - [`SupervisedTuningDatasetDistributionDict.buckets`](https://googleapis.github.io/python-genai/genai.html#genai.types.SupervisedTuningDatasetDistributionDict.buckets)
    - [`SupervisedTuningDatasetDistributionDict.max`](https://googleapis.github.io/python-genai/genai.html#genai.types.SupervisedTuningDatasetDistributionDict.max)
    - [`SupervisedTuningDatasetDistributionDict.mean`](https://googleapis.github.io/python-genai/genai.html#genai.types.SupervisedTuningDatasetDistributionDict.mean)
    - [`SupervisedTuningDatasetDistributionDict.median`](https://googleapis.github.io/python-genai/genai.html#genai.types.SupervisedTuningDatasetDistributionDict.median)
    - [`SupervisedTuningDatasetDistributionDict.min`](https://googleapis.github.io/python-genai/genai.html#genai.types.SupervisedTuningDatasetDistributionDict.min)
    - [`SupervisedTuningDatasetDistributionDict.p5`](https://googleapis.github.io/python-genai/genai.html#genai.types.SupervisedTuningDatasetDistributionDict.p5)
    - [`SupervisedTuningDatasetDistributionDict.p95`](https://googleapis.github.io/python-genai/genai.html#genai.types.SupervisedTuningDatasetDistributionDict.p95)
    - [`SupervisedTuningDatasetDistributionDict.sum`](https://googleapis.github.io/python-genai/genai.html#genai.types.SupervisedTuningDatasetDistributionDict.sum)
  - [`SupervisedTuningSpec`](https://googleapis.github.io/python-genai/genai.html#genai.types.SupervisedTuningSpec)
    - [`SupervisedTuningSpec.export_last_checkpoint_only`](https://googleapis.github.io/python-genai/genai.html#genai.types.SupervisedTuningSpec.export_last_checkpoint_only)
    - [`SupervisedTuningSpec.hyper_parameters`](https://googleapis.github.io/python-genai/genai.html#genai.types.SupervisedTuningSpec.hyper_parameters)
    - [`SupervisedTuningSpec.training_dataset_uri`](https://googleapis.github.io/python-genai/genai.html#genai.types.SupervisedTuningSpec.training_dataset_uri)
    - [`SupervisedTuningSpec.validation_dataset_uri`](https://googleapis.github.io/python-genai/genai.html#genai.types.SupervisedTuningSpec.validation_dataset_uri)
  - [`SupervisedTuningSpecDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.SupervisedTuningSpecDict)
    - [`SupervisedTuningSpecDict.export_last_checkpoint_only`](https://googleapis.github.io/python-genai/genai.html#genai.types.SupervisedTuningSpecDict.export_last_checkpoint_only)
    - [`SupervisedTuningSpecDict.hyper_parameters`](https://googleapis.github.io/python-genai/genai.html#genai.types.SupervisedTuningSpecDict.hyper_parameters)
    - [`SupervisedTuningSpecDict.training_dataset_uri`](https://googleapis.github.io/python-genai/genai.html#genai.types.SupervisedTuningSpecDict.training_dataset_uri)
    - [`SupervisedTuningSpecDict.validation_dataset_uri`](https://googleapis.github.io/python-genai/genai.html#genai.types.SupervisedTuningSpecDict.validation_dataset_uri)
  - [`TestTableFile`](https://googleapis.github.io/python-genai/genai.html#genai.types.TestTableFile)
    - [`TestTableFile.comment`](https://googleapis.github.io/python-genai/genai.html#genai.types.TestTableFile.comment)
    - [`TestTableFile.parameter_names`](https://googleapis.github.io/python-genai/genai.html#genai.types.TestTableFile.parameter_names)
    - [`TestTableFile.test_method`](https://googleapis.github.io/python-genai/genai.html#genai.types.TestTableFile.test_method)
    - [`TestTableFile.test_table`](https://googleapis.github.io/python-genai/genai.html#genai.types.TestTableFile.test_table)
  - [`TestTableFileDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.TestTableFileDict)
    - [`TestTableFileDict.comment`](https://googleapis.github.io/python-genai/genai.html#genai.types.TestTableFileDict.comment)
    - [`TestTableFileDict.parameter_names`](https://googleapis.github.io/python-genai/genai.html#genai.types.TestTableFileDict.parameter_names)
    - [`TestTableFileDict.test_method`](https://googleapis.github.io/python-genai/genai.html#genai.types.TestTableFileDict.test_method)
    - [`TestTableFileDict.test_table`](https://googleapis.github.io/python-genai/genai.html#genai.types.TestTableFileDict.test_table)
  - [`TestTableItem`](https://googleapis.github.io/python-genai/genai.html#genai.types.TestTableItem)
    - [`TestTableItem.exception_if_mldev`](https://googleapis.github.io/python-genai/genai.html#genai.types.TestTableItem.exception_if_mldev)
    - [`TestTableItem.exception_if_vertex`](https://googleapis.github.io/python-genai/genai.html#genai.types.TestTableItem.exception_if_vertex)
    - [`TestTableItem.has_union`](https://googleapis.github.io/python-genai/genai.html#genai.types.TestTableItem.has_union)
    - [`TestTableItem.ignore_keys`](https://googleapis.github.io/python-genai/genai.html#genai.types.TestTableItem.ignore_keys)
    - [`TestTableItem.name`](https://googleapis.github.io/python-genai/genai.html#genai.types.TestTableItem.name)
    - [`TestTableItem.override_replay_id`](https://googleapis.github.io/python-genai/genai.html#genai.types.TestTableItem.override_replay_id)
    - [`TestTableItem.parameters`](https://googleapis.github.io/python-genai/genai.html#genai.types.TestTableItem.parameters)
    - [`TestTableItem.skip_in_api_mode`](https://googleapis.github.io/python-genai/genai.html#genai.types.TestTableItem.skip_in_api_mode)
  - [`TestTableItemDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.TestTableItemDict)
    - [`TestTableItemDict.exception_if_mldev`](https://googleapis.github.io/python-genai/genai.html#genai.types.TestTableItemDict.exception_if_mldev)
    - [`TestTableItemDict.exception_if_vertex`](https://googleapis.github.io/python-genai/genai.html#genai.types.TestTableItemDict.exception_if_vertex)
    - [`TestTableItemDict.has_union`](https://googleapis.github.io/python-genai/genai.html#genai.types.TestTableItemDict.has_union)
    - [`TestTableItemDict.ignore_keys`](https://googleapis.github.io/python-genai/genai.html#genai.types.TestTableItemDict.ignore_keys)
    - [`TestTableItemDict.name`](https://googleapis.github.io/python-genai/genai.html#genai.types.TestTableItemDict.name)
    - [`TestTableItemDict.override_replay_id`](https://googleapis.github.io/python-genai/genai.html#genai.types.TestTableItemDict.override_replay_id)
    - [`TestTableItemDict.parameters`](https://googleapis.github.io/python-genai/genai.html#genai.types.TestTableItemDict.parameters)
    - [`TestTableItemDict.skip_in_api_mode`](https://googleapis.github.io/python-genai/genai.html#genai.types.TestTableItemDict.skip_in_api_mode)
  - [`ThinkingConfig`](https://googleapis.github.io/python-genai/genai.html#genai.types.ThinkingConfig)
    - [`ThinkingConfig.include_thoughts`](https://googleapis.github.io/python-genai/genai.html#genai.types.ThinkingConfig.include_thoughts)
    - [`ThinkingConfig.thinking_budget`](https://googleapis.github.io/python-genai/genai.html#genai.types.ThinkingConfig.thinking_budget)
  - [`ThinkingConfigDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.ThinkingConfigDict)
    - [`ThinkingConfigDict.include_thoughts`](https://googleapis.github.io/python-genai/genai.html#genai.types.ThinkingConfigDict.include_thoughts)
    - [`ThinkingConfigDict.thinking_budget`](https://googleapis.github.io/python-genai/genai.html#genai.types.ThinkingConfigDict.thinking_budget)
  - [`TokensInfo`](https://googleapis.github.io/python-genai/genai.html#genai.types.TokensInfo)
    - [`TokensInfo.role`](https://googleapis.github.io/python-genai/genai.html#genai.types.TokensInfo.role)
    - [`TokensInfo.token_ids`](https://googleapis.github.io/python-genai/genai.html#genai.types.TokensInfo.token_ids)
    - [`TokensInfo.tokens`](https://googleapis.github.io/python-genai/genai.html#genai.types.TokensInfo.tokens)
  - [`TokensInfoDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.TokensInfoDict)
    - [`TokensInfoDict.role`](https://googleapis.github.io/python-genai/genai.html#genai.types.TokensInfoDict.role)
    - [`TokensInfoDict.token_ids`](https://googleapis.github.io/python-genai/genai.html#genai.types.TokensInfoDict.token_ids)
    - [`TokensInfoDict.tokens`](https://googleapis.github.io/python-genai/genai.html#genai.types.TokensInfoDict.tokens)
  - [`Tool`](https://googleapis.github.io/python-genai/genai.html#genai.types.Tool)
    - [`Tool.code_execution`](https://googleapis.github.io/python-genai/genai.html#genai.types.Tool.code_execution)
    - [`Tool.enterprise_web_search`](https://googleapis.github.io/python-genai/genai.html#genai.types.Tool.enterprise_web_search)
    - [`Tool.function_declarations`](https://googleapis.github.io/python-genai/genai.html#genai.types.Tool.function_declarations)
    - [`Tool.google_maps`](https://googleapis.github.io/python-genai/genai.html#genai.types.Tool.google_maps)
    - [`Tool.google_search`](https://googleapis.github.io/python-genai/genai.html#genai.types.Tool.google_search)
    - [`Tool.google_search_retrieval`](https://googleapis.github.io/python-genai/genai.html#genai.types.Tool.google_search_retrieval)
    - [`Tool.retrieval`](https://googleapis.github.io/python-genai/genai.html#genai.types.Tool.retrieval)
    - [`Tool.url_context`](https://googleapis.github.io/python-genai/genai.html#genai.types.Tool.url_context)
  - [`ToolCodeExecution`](https://googleapis.github.io/python-genai/genai.html#genai.types.ToolCodeExecution)
  - [`ToolCodeExecutionDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.ToolCodeExecutionDict)
  - [`ToolConfig`](https://googleapis.github.io/python-genai/genai.html#genai.types.ToolConfig)
    - [`ToolConfig.function_calling_config`](https://googleapis.github.io/python-genai/genai.html#genai.types.ToolConfig.function_calling_config)
    - [`ToolConfig.retrieval_config`](https://googleapis.github.io/python-genai/genai.html#genai.types.ToolConfig.retrieval_config)
  - [`ToolConfigDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.ToolConfigDict)
    - [`ToolConfigDict.function_calling_config`](https://googleapis.github.io/python-genai/genai.html#genai.types.ToolConfigDict.function_calling_config)
    - [`ToolConfigDict.retrieval_config`](https://googleapis.github.io/python-genai/genai.html#genai.types.ToolConfigDict.retrieval_config)
  - [`ToolDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.ToolDict)
    - [`ToolDict.code_execution`](https://googleapis.github.io/python-genai/genai.html#genai.types.ToolDict.code_execution)
    - [`ToolDict.enterprise_web_search`](https://googleapis.github.io/python-genai/genai.html#genai.types.ToolDict.enterprise_web_search)
    - [`ToolDict.function_declarations`](https://googleapis.github.io/python-genai/genai.html#genai.types.ToolDict.function_declarations)
    - [`ToolDict.google_maps`](https://googleapis.github.io/python-genai/genai.html#genai.types.ToolDict.google_maps)
    - [`ToolDict.google_search`](https://googleapis.github.io/python-genai/genai.html#genai.types.ToolDict.google_search)
    - [`ToolDict.google_search_retrieval`](https://googleapis.github.io/python-genai/genai.html#genai.types.ToolDict.google_search_retrieval)
    - [`ToolDict.retrieval`](https://googleapis.github.io/python-genai/genai.html#genai.types.ToolDict.retrieval)
    - [`ToolDict.url_context`](https://googleapis.github.io/python-genai/genai.html#genai.types.ToolDict.url_context)
  - [`TrafficType`](https://googleapis.github.io/python-genai/genai.html#genai.types.TrafficType)
    - [`TrafficType.ON_DEMAND`](https://googleapis.github.io/python-genai/genai.html#genai.types.TrafficType.ON_DEMAND)
    - [`TrafficType.PROVISIONED_THROUGHPUT`](https://googleapis.github.io/python-genai/genai.html#genai.types.TrafficType.PROVISIONED_THROUGHPUT)
    - [`TrafficType.TRAFFIC_TYPE_UNSPECIFIED`](https://googleapis.github.io/python-genai/genai.html#genai.types.TrafficType.TRAFFIC_TYPE_UNSPECIFIED)
  - [`Transcription`](https://googleapis.github.io/python-genai/genai.html#genai.types.Transcription)
    - [`Transcription.finished`](https://googleapis.github.io/python-genai/genai.html#genai.types.Transcription.finished)
    - [`Transcription.text`](https://googleapis.github.io/python-genai/genai.html#genai.types.Transcription.text)
  - [`TranscriptionDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.TranscriptionDict)
    - [`TranscriptionDict.finished`](https://googleapis.github.io/python-genai/genai.html#genai.types.TranscriptionDict.finished)
    - [`TranscriptionDict.text`](https://googleapis.github.io/python-genai/genai.html#genai.types.TranscriptionDict.text)
  - [`TunedModel`](https://googleapis.github.io/python-genai/genai.html#genai.types.TunedModel)
    - [`TunedModel.checkpoints`](https://googleapis.github.io/python-genai/genai.html#genai.types.TunedModel.checkpoints)
    - [`TunedModel.endpoint`](https://googleapis.github.io/python-genai/genai.html#genai.types.TunedModel.endpoint)
    - [`TunedModel.model`](https://googleapis.github.io/python-genai/genai.html#genai.types.TunedModel.model)
  - [`TunedModelCheckpoint`](https://googleapis.github.io/python-genai/genai.html#genai.types.TunedModelCheckpoint)
    - [`TunedModelCheckpoint.checkpoint_id`](https://googleapis.github.io/python-genai/genai.html#genai.types.TunedModelCheckpoint.checkpoint_id)
    - [`TunedModelCheckpoint.endpoint`](https://googleapis.github.io/python-genai/genai.html#genai.types.TunedModelCheckpoint.endpoint)
    - [`TunedModelCheckpoint.epoch`](https://googleapis.github.io/python-genai/genai.html#genai.types.TunedModelCheckpoint.epoch)
    - [`TunedModelCheckpoint.step`](https://googleapis.github.io/python-genai/genai.html#genai.types.TunedModelCheckpoint.step)
  - [`TunedModelCheckpointDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.TunedModelCheckpointDict)
    - [`TunedModelCheckpointDict.checkpoint_id`](https://googleapis.github.io/python-genai/genai.html#genai.types.TunedModelCheckpointDict.checkpoint_id)
    - [`TunedModelCheckpointDict.endpoint`](https://googleapis.github.io/python-genai/genai.html#genai.types.TunedModelCheckpointDict.endpoint)
    - [`TunedModelCheckpointDict.epoch`](https://googleapis.github.io/python-genai/genai.html#genai.types.TunedModelCheckpointDict.epoch)
    - [`TunedModelCheckpointDict.step`](https://googleapis.github.io/python-genai/genai.html#genai.types.TunedModelCheckpointDict.step)
  - [`TunedModelDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.TunedModelDict)
    - [`TunedModelDict.checkpoints`](https://googleapis.github.io/python-genai/genai.html#genai.types.TunedModelDict.checkpoints)
    - [`TunedModelDict.endpoint`](https://googleapis.github.io/python-genai/genai.html#genai.types.TunedModelDict.endpoint)
    - [`TunedModelDict.model`](https://googleapis.github.io/python-genai/genai.html#genai.types.TunedModelDict.model)
  - [`TunedModelInfo`](https://googleapis.github.io/python-genai/genai.html#genai.types.TunedModelInfo)
    - [`TunedModelInfo.base_model`](https://googleapis.github.io/python-genai/genai.html#genai.types.TunedModelInfo.base_model)
    - [`TunedModelInfo.create_time`](https://googleapis.github.io/python-genai/genai.html#genai.types.TunedModelInfo.create_time)
    - [`TunedModelInfo.update_time`](https://googleapis.github.io/python-genai/genai.html#genai.types.TunedModelInfo.update_time)
  - [`TunedModelInfoDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.TunedModelInfoDict)
    - [`TunedModelInfoDict.base_model`](https://googleapis.github.io/python-genai/genai.html#genai.types.TunedModelInfoDict.base_model)
    - [`TunedModelInfoDict.create_time`](https://googleapis.github.io/python-genai/genai.html#genai.types.TunedModelInfoDict.create_time)
    - [`TunedModelInfoDict.update_time`](https://googleapis.github.io/python-genai/genai.html#genai.types.TunedModelInfoDict.update_time)
  - [`TuningDataStats`](https://googleapis.github.io/python-genai/genai.html#genai.types.TuningDataStats)
    - [`TuningDataStats.distillation_data_stats`](https://googleapis.github.io/python-genai/genai.html#genai.types.TuningDataStats.distillation_data_stats)
    - [`TuningDataStats.supervised_tuning_data_stats`](https://googleapis.github.io/python-genai/genai.html#genai.types.TuningDataStats.supervised_tuning_data_stats)
  - [`TuningDataStatsDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.TuningDataStatsDict)
    - [`TuningDataStatsDict.distillation_data_stats`](https://googleapis.github.io/python-genai/genai.html#genai.types.TuningDataStatsDict.distillation_data_stats)
    - [`TuningDataStatsDict.supervised_tuning_data_stats`](https://googleapis.github.io/python-genai/genai.html#genai.types.TuningDataStatsDict.supervised_tuning_data_stats)
  - [`TuningDataset`](https://googleapis.github.io/python-genai/genai.html#genai.types.TuningDataset)
    - [`TuningDataset.examples`](https://googleapis.github.io/python-genai/genai.html#genai.types.TuningDataset.examples)
    - [`TuningDataset.gcs_uri`](https://googleapis.github.io/python-genai/genai.html#genai.types.TuningDataset.gcs_uri)
  - [`TuningDatasetDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.TuningDatasetDict)
    - [`TuningDatasetDict.examples`](https://googleapis.github.io/python-genai/genai.html#genai.types.TuningDatasetDict.examples)
    - [`TuningDatasetDict.gcs_uri`](https://googleapis.github.io/python-genai/genai.html#genai.types.TuningDatasetDict.gcs_uri)
  - [`TuningExample`](https://googleapis.github.io/python-genai/genai.html#genai.types.TuningExample)
    - [`TuningExample.output`](https://googleapis.github.io/python-genai/genai.html#genai.types.TuningExample.output)
    - [`TuningExample.text_input`](https://googleapis.github.io/python-genai/genai.html#genai.types.TuningExample.text_input)
  - [`TuningExampleDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.TuningExampleDict)
    - [`TuningExampleDict.output`](https://googleapis.github.io/python-genai/genai.html#genai.types.TuningExampleDict.output)
    - [`TuningExampleDict.text_input`](https://googleapis.github.io/python-genai/genai.html#genai.types.TuningExampleDict.text_input)
  - [`TuningJob`](https://googleapis.github.io/python-genai/genai.html#genai.types.TuningJob)
    - [`TuningJob.base_model`](https://googleapis.github.io/python-genai/genai.html#genai.types.TuningJob.base_model)
    - [`TuningJob.create_time`](https://googleapis.github.io/python-genai/genai.html#genai.types.TuningJob.create_time)
    - [`TuningJob.description`](https://googleapis.github.io/python-genai/genai.html#genai.types.TuningJob.description)
    - [`TuningJob.distillation_spec`](https://googleapis.github.io/python-genai/genai.html#genai.types.TuningJob.distillation_spec)
    - [`TuningJob.encryption_spec`](https://googleapis.github.io/python-genai/genai.html#genai.types.TuningJob.encryption_spec)
    - [`TuningJob.end_time`](https://googleapis.github.io/python-genai/genai.html#genai.types.TuningJob.end_time)
    - [`TuningJob.error`](https://googleapis.github.io/python-genai/genai.html#genai.types.TuningJob.error)
    - [`TuningJob.experiment`](https://googleapis.github.io/python-genai/genai.html#genai.types.TuningJob.experiment)
    - [`TuningJob.labels`](https://googleapis.github.io/python-genai/genai.html#genai.types.TuningJob.labels)
    - [`TuningJob.name`](https://googleapis.github.io/python-genai/genai.html#genai.types.TuningJob.name)
    - [`TuningJob.partner_model_tuning_spec`](https://googleapis.github.io/python-genai/genai.html#genai.types.TuningJob.partner_model_tuning_spec)
    - [`TuningJob.pipeline_job`](https://googleapis.github.io/python-genai/genai.html#genai.types.TuningJob.pipeline_job)
    - [`TuningJob.service_account`](https://googleapis.github.io/python-genai/genai.html#genai.types.TuningJob.service_account)
    - [`TuningJob.start_time`](https://googleapis.github.io/python-genai/genai.html#genai.types.TuningJob.start_time)
    - [`TuningJob.state`](https://googleapis.github.io/python-genai/genai.html#genai.types.TuningJob.state)
    - [`TuningJob.supervised_tuning_spec`](https://googleapis.github.io/python-genai/genai.html#genai.types.TuningJob.supervised_tuning_spec)
    - [`TuningJob.tuned_model`](https://googleapis.github.io/python-genai/genai.html#genai.types.TuningJob.tuned_model)
    - [`TuningJob.tuned_model_display_name`](https://googleapis.github.io/python-genai/genai.html#genai.types.TuningJob.tuned_model_display_name)
    - [`TuningJob.tuning_data_stats`](https://googleapis.github.io/python-genai/genai.html#genai.types.TuningJob.tuning_data_stats)
    - [`TuningJob.update_time`](https://googleapis.github.io/python-genai/genai.html#genai.types.TuningJob.update_time)
    - [`TuningJob.has_ended`](https://googleapis.github.io/python-genai/genai.html#genai.types.TuningJob.has_ended)
    - [`TuningJob.has_succeeded`](https://googleapis.github.io/python-genai/genai.html#genai.types.TuningJob.has_succeeded)
  - [`TuningJobDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.TuningJobDict)
    - [`TuningJobDict.base_model`](https://googleapis.github.io/python-genai/genai.html#genai.types.TuningJobDict.base_model)
    - [`TuningJobDict.create_time`](https://googleapis.github.io/python-genai/genai.html#genai.types.TuningJobDict.create_time)
    - [`TuningJobDict.description`](https://googleapis.github.io/python-genai/genai.html#genai.types.TuningJobDict.description)
    - [`TuningJobDict.distillation_spec`](https://googleapis.github.io/python-genai/genai.html#genai.types.TuningJobDict.distillation_spec)
    - [`TuningJobDict.encryption_spec`](https://googleapis.github.io/python-genai/genai.html#genai.types.TuningJobDict.encryption_spec)
    - [`TuningJobDict.end_time`](https://googleapis.github.io/python-genai/genai.html#genai.types.TuningJobDict.end_time)
    - [`TuningJobDict.error`](https://googleapis.github.io/python-genai/genai.html#genai.types.TuningJobDict.error)
    - [`TuningJobDict.experiment`](https://googleapis.github.io/python-genai/genai.html#genai.types.TuningJobDict.experiment)
    - [`TuningJobDict.labels`](https://googleapis.github.io/python-genai/genai.html#genai.types.TuningJobDict.labels)
    - [`TuningJobDict.name`](https://googleapis.github.io/python-genai/genai.html#genai.types.TuningJobDict.name)
    - [`TuningJobDict.partner_model_tuning_spec`](https://googleapis.github.io/python-genai/genai.html#genai.types.TuningJobDict.partner_model_tuning_spec)
    - [`TuningJobDict.pipeline_job`](https://googleapis.github.io/python-genai/genai.html#genai.types.TuningJobDict.pipeline_job)
    - [`TuningJobDict.service_account`](https://googleapis.github.io/python-genai/genai.html#genai.types.TuningJobDict.service_account)
    - [`TuningJobDict.start_time`](https://googleapis.github.io/python-genai/genai.html#genai.types.TuningJobDict.start_time)
    - [`TuningJobDict.state`](https://googleapis.github.io/python-genai/genai.html#genai.types.TuningJobDict.state)
    - [`TuningJobDict.supervised_tuning_spec`](https://googleapis.github.io/python-genai/genai.html#genai.types.TuningJobDict.supervised_tuning_spec)
    - [`TuningJobDict.tuned_model`](https://googleapis.github.io/python-genai/genai.html#genai.types.TuningJobDict.tuned_model)
    - [`TuningJobDict.tuned_model_display_name`](https://googleapis.github.io/python-genai/genai.html#genai.types.TuningJobDict.tuned_model_display_name)
    - [`TuningJobDict.tuning_data_stats`](https://googleapis.github.io/python-genai/genai.html#genai.types.TuningJobDict.tuning_data_stats)
    - [`TuningJobDict.update_time`](https://googleapis.github.io/python-genai/genai.html#genai.types.TuningJobDict.update_time)
  - [`TuningValidationDataset`](https://googleapis.github.io/python-genai/genai.html#genai.types.TuningValidationDataset)
    - [`TuningValidationDataset.gcs_uri`](https://googleapis.github.io/python-genai/genai.html#genai.types.TuningValidationDataset.gcs_uri)
  - [`TuningValidationDatasetDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.TuningValidationDatasetDict)
    - [`TuningValidationDatasetDict.gcs_uri`](https://googleapis.github.io/python-genai/genai.html#genai.types.TuningValidationDatasetDict.gcs_uri)
  - [`TurnCoverage`](https://googleapis.github.io/python-genai/genai.html#genai.types.TurnCoverage)
    - [`TurnCoverage.TURN_COVERAGE_UNSPECIFIED`](https://googleapis.github.io/python-genai/genai.html#genai.types.TurnCoverage.TURN_COVERAGE_UNSPECIFIED)
    - [`TurnCoverage.TURN_INCLUDES_ALL_INPUT`](https://googleapis.github.io/python-genai/genai.html#genai.types.TurnCoverage.TURN_INCLUDES_ALL_INPUT)
    - [`TurnCoverage.TURN_INCLUDES_ONLY_ACTIVITY`](https://googleapis.github.io/python-genai/genai.html#genai.types.TurnCoverage.TURN_INCLUDES_ONLY_ACTIVITY)
  - [`Type`](https://googleapis.github.io/python-genai/genai.html#genai.types.Type)
    - [`Type.ARRAY`](https://googleapis.github.io/python-genai/genai.html#genai.types.Type.ARRAY)
    - [`Type.BOOLEAN`](https://googleapis.github.io/python-genai/genai.html#genai.types.Type.BOOLEAN)
    - [`Type.INTEGER`](https://googleapis.github.io/python-genai/genai.html#genai.types.Type.INTEGER)
    - [`Type.NULL`](https://googleapis.github.io/python-genai/genai.html#genai.types.Type.NULL)
    - [`Type.NUMBER`](https://googleapis.github.io/python-genai/genai.html#genai.types.Type.NUMBER)
    - [`Type.OBJECT`](https://googleapis.github.io/python-genai/genai.html#genai.types.Type.OBJECT)
    - [`Type.STRING`](https://googleapis.github.io/python-genai/genai.html#genai.types.Type.STRING)
    - [`Type.TYPE_UNSPECIFIED`](https://googleapis.github.io/python-genai/genai.html#genai.types.Type.TYPE_UNSPECIFIED)
  - [`UpdateCachedContentConfig`](https://googleapis.github.io/python-genai/genai.html#genai.types.UpdateCachedContentConfig)
    - [`UpdateCachedContentConfig.expire_time`](https://googleapis.github.io/python-genai/genai.html#genai.types.UpdateCachedContentConfig.expire_time)
    - [`UpdateCachedContentConfig.http_options`](https://googleapis.github.io/python-genai/genai.html#genai.types.UpdateCachedContentConfig.http_options)
    - [`UpdateCachedContentConfig.ttl`](https://googleapis.github.io/python-genai/genai.html#genai.types.UpdateCachedContentConfig.ttl)
  - [`UpdateCachedContentConfigDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.UpdateCachedContentConfigDict)
    - [`UpdateCachedContentConfigDict.expire_time`](https://googleapis.github.io/python-genai/genai.html#genai.types.UpdateCachedContentConfigDict.expire_time)
    - [`UpdateCachedContentConfigDict.http_options`](https://googleapis.github.io/python-genai/genai.html#genai.types.UpdateCachedContentConfigDict.http_options)
    - [`UpdateCachedContentConfigDict.ttl`](https://googleapis.github.io/python-genai/genai.html#genai.types.UpdateCachedContentConfigDict.ttl)
  - [`UpdateModelConfig`](https://googleapis.github.io/python-genai/genai.html#genai.types.UpdateModelConfig)
    - [`UpdateModelConfig.default_checkpoint_id`](https://googleapis.github.io/python-genai/genai.html#genai.types.UpdateModelConfig.default_checkpoint_id)
    - [`UpdateModelConfig.description`](https://googleapis.github.io/python-genai/genai.html#genai.types.UpdateModelConfig.description)
    - [`UpdateModelConfig.display_name`](https://googleapis.github.io/python-genai/genai.html#genai.types.UpdateModelConfig.display_name)
    - [`UpdateModelConfig.http_options`](https://googleapis.github.io/python-genai/genai.html#genai.types.UpdateModelConfig.http_options)
  - [`UpdateModelConfigDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.UpdateModelConfigDict)
    - [`UpdateModelConfigDict.default_checkpoint_id`](https://googleapis.github.io/python-genai/genai.html#genai.types.UpdateModelConfigDict.default_checkpoint_id)
    - [`UpdateModelConfigDict.description`](https://googleapis.github.io/python-genai/genai.html#genai.types.UpdateModelConfigDict.description)
    - [`UpdateModelConfigDict.display_name`](https://googleapis.github.io/python-genai/genai.html#genai.types.UpdateModelConfigDict.display_name)
    - [`UpdateModelConfigDict.http_options`](https://googleapis.github.io/python-genai/genai.html#genai.types.UpdateModelConfigDict.http_options)
  - [`UploadFileConfig`](https://googleapis.github.io/python-genai/genai.html#genai.types.UploadFileConfig)
    - [`UploadFileConfig.display_name`](https://googleapis.github.io/python-genai/genai.html#genai.types.UploadFileConfig.display_name)
    - [`UploadFileConfig.http_options`](https://googleapis.github.io/python-genai/genai.html#genai.types.UploadFileConfig.http_options)
    - [`UploadFileConfig.mime_type`](https://googleapis.github.io/python-genai/genai.html#genai.types.UploadFileConfig.mime_type)
    - [`UploadFileConfig.name`](https://googleapis.github.io/python-genai/genai.html#genai.types.UploadFileConfig.name)
  - [`UploadFileConfigDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.UploadFileConfigDict)
    - [`UploadFileConfigDict.display_name`](https://googleapis.github.io/python-genai/genai.html#genai.types.UploadFileConfigDict.display_name)
    - [`UploadFileConfigDict.http_options`](https://googleapis.github.io/python-genai/genai.html#genai.types.UploadFileConfigDict.http_options)
    - [`UploadFileConfigDict.mime_type`](https://googleapis.github.io/python-genai/genai.html#genai.types.UploadFileConfigDict.mime_type)
    - [`UploadFileConfigDict.name`](https://googleapis.github.io/python-genai/genai.html#genai.types.UploadFileConfigDict.name)
  - [`UpscaleImageConfig`](https://googleapis.github.io/python-genai/genai.html#genai.types.UpscaleImageConfig)
    - [`UpscaleImageConfig.http_options`](https://googleapis.github.io/python-genai/genai.html#genai.types.UpscaleImageConfig.http_options)
    - [`UpscaleImageConfig.include_rai_reason`](https://googleapis.github.io/python-genai/genai.html#genai.types.UpscaleImageConfig.include_rai_reason)
    - [`UpscaleImageConfig.output_compression_quality`](https://googleapis.github.io/python-genai/genai.html#genai.types.UpscaleImageConfig.output_compression_quality)
    - [`UpscaleImageConfig.output_mime_type`](https://googleapis.github.io/python-genai/genai.html#genai.types.UpscaleImageConfig.output_mime_type)
  - [`UpscaleImageConfigDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.UpscaleImageConfigDict)
    - [`UpscaleImageConfigDict.http_options`](https://googleapis.github.io/python-genai/genai.html#genai.types.UpscaleImageConfigDict.http_options)
    - [`UpscaleImageConfigDict.include_rai_reason`](https://googleapis.github.io/python-genai/genai.html#genai.types.UpscaleImageConfigDict.include_rai_reason)
    - [`UpscaleImageConfigDict.output_compression_quality`](https://googleapis.github.io/python-genai/genai.html#genai.types.UpscaleImageConfigDict.output_compression_quality)
    - [`UpscaleImageConfigDict.output_mime_type`](https://googleapis.github.io/python-genai/genai.html#genai.types.UpscaleImageConfigDict.output_mime_type)
  - [`UpscaleImageParameters`](https://googleapis.github.io/python-genai/genai.html#genai.types.UpscaleImageParameters)
    - [`UpscaleImageParameters.config`](https://googleapis.github.io/python-genai/genai.html#genai.types.UpscaleImageParameters.config)
    - [`UpscaleImageParameters.image`](https://googleapis.github.io/python-genai/genai.html#genai.types.UpscaleImageParameters.image)
    - [`UpscaleImageParameters.model`](https://googleapis.github.io/python-genai/genai.html#genai.types.UpscaleImageParameters.model)
    - [`UpscaleImageParameters.upscale_factor`](https://googleapis.github.io/python-genai/genai.html#genai.types.UpscaleImageParameters.upscale_factor)
  - [`UpscaleImageParametersDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.UpscaleImageParametersDict)
    - [`UpscaleImageParametersDict.config`](https://googleapis.github.io/python-genai/genai.html#genai.types.UpscaleImageParametersDict.config)
    - [`UpscaleImageParametersDict.image`](https://googleapis.github.io/python-genai/genai.html#genai.types.UpscaleImageParametersDict.image)
    - [`UpscaleImageParametersDict.model`](https://googleapis.github.io/python-genai/genai.html#genai.types.UpscaleImageParametersDict.model)
    - [`UpscaleImageParametersDict.upscale_factor`](https://googleapis.github.io/python-genai/genai.html#genai.types.UpscaleImageParametersDict.upscale_factor)
  - [`UpscaleImageResponse`](https://googleapis.github.io/python-genai/genai.html#genai.types.UpscaleImageResponse)
    - [`UpscaleImageResponse.generated_images`](https://googleapis.github.io/python-genai/genai.html#genai.types.UpscaleImageResponse.generated_images)
  - [`UpscaleImageResponseDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.UpscaleImageResponseDict)
    - [`UpscaleImageResponseDict.generated_images`](https://googleapis.github.io/python-genai/genai.html#genai.types.UpscaleImageResponseDict.generated_images)
  - [`UrlContext`](https://googleapis.github.io/python-genai/genai.html#genai.types.UrlContext)
  - [`UrlContextDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.UrlContextDict)
  - [`UrlContextMetadata`](https://googleapis.github.io/python-genai/genai.html#genai.types.UrlContextMetadata)
    - [`UrlContextMetadata.url_metadata`](https://googleapis.github.io/python-genai/genai.html#genai.types.UrlContextMetadata.url_metadata)
  - [`UrlContextMetadataDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.UrlContextMetadataDict)
    - [`UrlContextMetadataDict.url_metadata`](https://googleapis.github.io/python-genai/genai.html#genai.types.UrlContextMetadataDict.url_metadata)
  - [`UrlMetadata`](https://googleapis.github.io/python-genai/genai.html#genai.types.UrlMetadata)
    - [`UrlMetadata.retrieved_url`](https://googleapis.github.io/python-genai/genai.html#genai.types.UrlMetadata.retrieved_url)
    - [`UrlMetadata.url_retrieval_status`](https://googleapis.github.io/python-genai/genai.html#genai.types.UrlMetadata.url_retrieval_status)
  - [`UrlMetadataDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.UrlMetadataDict)
    - [`UrlMetadataDict.retrieved_url`](https://googleapis.github.io/python-genai/genai.html#genai.types.UrlMetadataDict.retrieved_url)
    - [`UrlMetadataDict.url_retrieval_status`](https://googleapis.github.io/python-genai/genai.html#genai.types.UrlMetadataDict.url_retrieval_status)
  - [`UrlRetrievalStatus`](https://googleapis.github.io/python-genai/genai.html#genai.types.UrlRetrievalStatus)
    - [`UrlRetrievalStatus.URL_RETRIEVAL_STATUS_ERROR`](https://googleapis.github.io/python-genai/genai.html#genai.types.UrlRetrievalStatus.URL_RETRIEVAL_STATUS_ERROR)
    - [`UrlRetrievalStatus.URL_RETRIEVAL_STATUS_SUCCESS`](https://googleapis.github.io/python-genai/genai.html#genai.types.UrlRetrievalStatus.URL_RETRIEVAL_STATUS_SUCCESS)
    - [`UrlRetrievalStatus.URL_RETRIEVAL_STATUS_UNSPECIFIED`](https://googleapis.github.io/python-genai/genai.html#genai.types.UrlRetrievalStatus.URL_RETRIEVAL_STATUS_UNSPECIFIED)
  - [`UsageMetadata`](https://googleapis.github.io/python-genai/genai.html#genai.types.UsageMetadata)
    - [`UsageMetadata.cache_tokens_details`](https://googleapis.github.io/python-genai/genai.html#genai.types.UsageMetadata.cache_tokens_details)
    - [`UsageMetadata.cached_content_token_count`](https://googleapis.github.io/python-genai/genai.html#genai.types.UsageMetadata.cached_content_token_count)
    - [`UsageMetadata.prompt_token_count`](https://googleapis.github.io/python-genai/genai.html#genai.types.UsageMetadata.prompt_token_count)
    - [`UsageMetadata.prompt_tokens_details`](https://googleapis.github.io/python-genai/genai.html#genai.types.UsageMetadata.prompt_tokens_details)
    - [`UsageMetadata.response_token_count`](https://googleapis.github.io/python-genai/genai.html#genai.types.UsageMetadata.response_token_count)
    - [`UsageMetadata.response_tokens_details`](https://googleapis.github.io/python-genai/genai.html#genai.types.UsageMetadata.response_tokens_details)
    - [`UsageMetadata.thoughts_token_count`](https://googleapis.github.io/python-genai/genai.html#genai.types.UsageMetadata.thoughts_token_count)
    - [`UsageMetadata.tool_use_prompt_token_count`](https://googleapis.github.io/python-genai/genai.html#genai.types.UsageMetadata.tool_use_prompt_token_count)
    - [`UsageMetadata.tool_use_prompt_tokens_details`](https://googleapis.github.io/python-genai/genai.html#genai.types.UsageMetadata.tool_use_prompt_tokens_details)
    - [`UsageMetadata.total_token_count`](https://googleapis.github.io/python-genai/genai.html#genai.types.UsageMetadata.total_token_count)
    - [`UsageMetadata.traffic_type`](https://googleapis.github.io/python-genai/genai.html#genai.types.UsageMetadata.traffic_type)
  - [`UsageMetadataDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.UsageMetadataDict)
    - [`UsageMetadataDict.cache_tokens_details`](https://googleapis.github.io/python-genai/genai.html#genai.types.UsageMetadataDict.cache_tokens_details)
    - [`UsageMetadataDict.cached_content_token_count`](https://googleapis.github.io/python-genai/genai.html#genai.types.UsageMetadataDict.cached_content_token_count)
    - [`UsageMetadataDict.prompt_token_count`](https://googleapis.github.io/python-genai/genai.html#genai.types.UsageMetadataDict.prompt_token_count)
    - [`UsageMetadataDict.prompt_tokens_details`](https://googleapis.github.io/python-genai/genai.html#genai.types.UsageMetadataDict.prompt_tokens_details)
    - [`UsageMetadataDict.response_token_count`](https://googleapis.github.io/python-genai/genai.html#genai.types.UsageMetadataDict.response_token_count)
    - [`UsageMetadataDict.response_tokens_details`](https://googleapis.github.io/python-genai/genai.html#genai.types.UsageMetadataDict.response_tokens_details)
    - [`UsageMetadataDict.thoughts_token_count`](https://googleapis.github.io/python-genai/genai.html#genai.types.UsageMetadataDict.thoughts_token_count)
    - [`UsageMetadataDict.tool_use_prompt_token_count`](https://googleapis.github.io/python-genai/genai.html#genai.types.UsageMetadataDict.tool_use_prompt_token_count)
    - [`UsageMetadataDict.tool_use_prompt_tokens_details`](https://googleapis.github.io/python-genai/genai.html#genai.types.UsageMetadataDict.tool_use_prompt_tokens_details)
    - [`UsageMetadataDict.total_token_count`](https://googleapis.github.io/python-genai/genai.html#genai.types.UsageMetadataDict.total_token_count)
    - [`UsageMetadataDict.traffic_type`](https://googleapis.github.io/python-genai/genai.html#genai.types.UsageMetadataDict.traffic_type)
  - [`UserContent`](https://googleapis.github.io/python-genai/genai.html#genai.types.UserContent)
    - [`UserContent.parts`](https://googleapis.github.io/python-genai/genai.html#genai.types.UserContent.parts)
    - [`UserContent.role`](https://googleapis.github.io/python-genai/genai.html#genai.types.UserContent.role)
  - [`VertexAISearch`](https://googleapis.github.io/python-genai/genai.html#genai.types.VertexAISearch)
    - [`VertexAISearch.data_store_specs`](https://googleapis.github.io/python-genai/genai.html#genai.types.VertexAISearch.data_store_specs)
    - [`VertexAISearch.datastore`](https://googleapis.github.io/python-genai/genai.html#genai.types.VertexAISearch.datastore)
    - [`VertexAISearch.engine`](https://googleapis.github.io/python-genai/genai.html#genai.types.VertexAISearch.engine)
    - [`VertexAISearch.filter`](https://googleapis.github.io/python-genai/genai.html#genai.types.VertexAISearch.filter)
    - [`VertexAISearch.max_results`](https://googleapis.github.io/python-genai/genai.html#genai.types.VertexAISearch.max_results)
  - [`VertexAISearchDataStoreSpec`](https://googleapis.github.io/python-genai/genai.html#genai.types.VertexAISearchDataStoreSpec)
    - [`VertexAISearchDataStoreSpec.data_store`](https://googleapis.github.io/python-genai/genai.html#genai.types.VertexAISearchDataStoreSpec.data_store)
    - [`VertexAISearchDataStoreSpec.filter`](https://googleapis.github.io/python-genai/genai.html#genai.types.VertexAISearchDataStoreSpec.filter)
  - [`VertexAISearchDataStoreSpecDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.VertexAISearchDataStoreSpecDict)
    - [`VertexAISearchDataStoreSpecDict.data_store`](https://googleapis.github.io/python-genai/genai.html#genai.types.VertexAISearchDataStoreSpecDict.data_store)
    - [`VertexAISearchDataStoreSpecDict.filter`](https://googleapis.github.io/python-genai/genai.html#genai.types.VertexAISearchDataStoreSpecDict.filter)
  - [`VertexAISearchDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.VertexAISearchDict)
    - [`VertexAISearchDict.data_store_specs`](https://googleapis.github.io/python-genai/genai.html#genai.types.VertexAISearchDict.data_store_specs)
    - [`VertexAISearchDict.datastore`](https://googleapis.github.io/python-genai/genai.html#genai.types.VertexAISearchDict.datastore)
    - [`VertexAISearchDict.engine`](https://googleapis.github.io/python-genai/genai.html#genai.types.VertexAISearchDict.engine)
    - [`VertexAISearchDict.filter`](https://googleapis.github.io/python-genai/genai.html#genai.types.VertexAISearchDict.filter)
    - [`VertexAISearchDict.max_results`](https://googleapis.github.io/python-genai/genai.html#genai.types.VertexAISearchDict.max_results)
  - [`VertexRagStore`](https://googleapis.github.io/python-genai/genai.html#genai.types.VertexRagStore)
    - [`VertexRagStore.rag_corpora`](https://googleapis.github.io/python-genai/genai.html#genai.types.VertexRagStore.rag_corpora)
    - [`VertexRagStore.rag_resources`](https://googleapis.github.io/python-genai/genai.html#genai.types.VertexRagStore.rag_resources)
    - [`VertexRagStore.rag_retrieval_config`](https://googleapis.github.io/python-genai/genai.html#genai.types.VertexRagStore.rag_retrieval_config)
    - [`VertexRagStore.similarity_top_k`](https://googleapis.github.io/python-genai/genai.html#genai.types.VertexRagStore.similarity_top_k)
    - [`VertexRagStore.store_context`](https://googleapis.github.io/python-genai/genai.html#genai.types.VertexRagStore.store_context)
    - [`VertexRagStore.vector_distance_threshold`](https://googleapis.github.io/python-genai/genai.html#genai.types.VertexRagStore.vector_distance_threshold)
  - [`VertexRagStoreDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.VertexRagStoreDict)
    - [`VertexRagStoreDict.rag_corpora`](https://googleapis.github.io/python-genai/genai.html#genai.types.VertexRagStoreDict.rag_corpora)
    - [`VertexRagStoreDict.rag_resources`](https://googleapis.github.io/python-genai/genai.html#genai.types.VertexRagStoreDict.rag_resources)
    - [`VertexRagStoreDict.rag_retrieval_config`](https://googleapis.github.io/python-genai/genai.html#genai.types.VertexRagStoreDict.rag_retrieval_config)
    - [`VertexRagStoreDict.similarity_top_k`](https://googleapis.github.io/python-genai/genai.html#genai.types.VertexRagStoreDict.similarity_top_k)
    - [`VertexRagStoreDict.store_context`](https://googleapis.github.io/python-genai/genai.html#genai.types.VertexRagStoreDict.store_context)
    - [`VertexRagStoreDict.vector_distance_threshold`](https://googleapis.github.io/python-genai/genai.html#genai.types.VertexRagStoreDict.vector_distance_threshold)
  - [`VertexRagStoreRagResource`](https://googleapis.github.io/python-genai/genai.html#genai.types.VertexRagStoreRagResource)
    - [`VertexRagStoreRagResource.rag_corpus`](https://googleapis.github.io/python-genai/genai.html#genai.types.VertexRagStoreRagResource.rag_corpus)
    - [`VertexRagStoreRagResource.rag_file_ids`](https://googleapis.github.io/python-genai/genai.html#genai.types.VertexRagStoreRagResource.rag_file_ids)
  - [`VertexRagStoreRagResourceDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.VertexRagStoreRagResourceDict)
    - [`VertexRagStoreRagResourceDict.rag_corpus`](https://googleapis.github.io/python-genai/genai.html#genai.types.VertexRagStoreRagResourceDict.rag_corpus)
    - [`VertexRagStoreRagResourceDict.rag_file_ids`](https://googleapis.github.io/python-genai/genai.html#genai.types.VertexRagStoreRagResourceDict.rag_file_ids)
  - [`Video`](https://googleapis.github.io/python-genai/genai.html#genai.types.Video)
    - [`Video.mime_type`](https://googleapis.github.io/python-genai/genai.html#genai.types.Video.mime_type)
    - [`Video.uri`](https://googleapis.github.io/python-genai/genai.html#genai.types.Video.uri)
    - [`Video.video_bytes`](https://googleapis.github.io/python-genai/genai.html#genai.types.Video.video_bytes)
    - [`Video.from_file()`](https://googleapis.github.io/python-genai/genai.html#genai.types.Video.from_file)
    - [`Video.save()`](https://googleapis.github.io/python-genai/genai.html#genai.types.Video.save)
    - [`Video.show()`](https://googleapis.github.io/python-genai/genai.html#genai.types.Video.show)
  - [`VideoDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.VideoDict)
    - [`VideoDict.mime_type`](https://googleapis.github.io/python-genai/genai.html#genai.types.VideoDict.mime_type)
    - [`VideoDict.uri`](https://googleapis.github.io/python-genai/genai.html#genai.types.VideoDict.uri)
    - [`VideoDict.video_bytes`](https://googleapis.github.io/python-genai/genai.html#genai.types.VideoDict.video_bytes)
  - [`VideoMetadata`](https://googleapis.github.io/python-genai/genai.html#genai.types.VideoMetadata)
    - [`VideoMetadata.end_offset`](https://googleapis.github.io/python-genai/genai.html#genai.types.VideoMetadata.end_offset)
    - [`VideoMetadata.fps`](https://googleapis.github.io/python-genai/genai.html#genai.types.VideoMetadata.fps)
    - [`VideoMetadata.start_offset`](https://googleapis.github.io/python-genai/genai.html#genai.types.VideoMetadata.start_offset)
  - [`VideoMetadataDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.VideoMetadataDict)
    - [`VideoMetadataDict.end_offset`](https://googleapis.github.io/python-genai/genai.html#genai.types.VideoMetadataDict.end_offset)
    - [`VideoMetadataDict.fps`](https://googleapis.github.io/python-genai/genai.html#genai.types.VideoMetadataDict.fps)
    - [`VideoMetadataDict.start_offset`](https://googleapis.github.io/python-genai/genai.html#genai.types.VideoMetadataDict.start_offset)
  - [`VoiceConfig`](https://googleapis.github.io/python-genai/genai.html#genai.types.VoiceConfig)
    - [`VoiceConfig.prebuilt_voice_config`](https://googleapis.github.io/python-genai/genai.html#genai.types.VoiceConfig.prebuilt_voice_config)
  - [`VoiceConfigDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.VoiceConfigDict)
    - [`VoiceConfigDict.prebuilt_voice_config`](https://googleapis.github.io/python-genai/genai.html#genai.types.VoiceConfigDict.prebuilt_voice_config)
  - [`WeightedPrompt`](https://googleapis.github.io/python-genai/genai.html#genai.types.WeightedPrompt)
    - [`WeightedPrompt.text`](https://googleapis.github.io/python-genai/genai.html#genai.types.WeightedPrompt.text)
    - [`WeightedPrompt.weight`](https://googleapis.github.io/python-genai/genai.html#genai.types.WeightedPrompt.weight)
  - [`WeightedPromptDict`](https://googleapis.github.io/python-genai/genai.html#genai.types.WeightedPromptDict)
    - [`WeightedPromptDict.text`](https://googleapis.github.io/python-genai/genai.html#genai.types.WeightedPromptDict.text)
    - [`WeightedPromptDict.weight`](https://googleapis.github.io/python-genai/genai.html#genai.types.WeightedPromptDict.weight)