Main Concepts
Text Generation Model
Moonshot's text generation model (referred to as moonshot-v1) is trained to understand both natural and written language. It can generate text output based on the input provided. The input to the model is also known as a "prompt." We generally recommend that you provide clear instructions and some examples to enable the model to complete the intended task. Designing a prompt is essentially learning how to "train" the model. The moonshot-v1 model can be used for a variety of tasks, including content or code generation, summarization, conversation, and creative writing.

Language Model Inference Service
The language model inference service is an API service based on the pre-trained models developed and trained by us (Moonshot AI). In terms of design, we primarily offer a Chat Completions interface externally, which can be used to generate text. However, it does not support access to external resources such as the internet or databases, nor does it support the execution of any code.

Token
Text generation models process text in units called Tokens. A Token represents a common sequence of characters. For example, a single Chinese character like "夔" might be broken down into a combination of several Tokens, while a short and common phrase like "中国" might be represented by a single Token. Generally speaking, for a typical Chinese text, 1 Token is roughly equivalent to 1.5-2 Chinese characters.

It is important to note that for our text model, the total length of Input and Output cannot exceed the model's maximum context length.

Rate Limits
How do these rate limits work?

Rate limits are measured in four ways: concurrency, RPM (requests per minute), TPM (Tokens per minute), and TPD (Tokens per day). The rate limit can be reached in any of these categories, depending on which one is hit first. For example, you might send 20 requests to ChatCompletions, each with only 100 Tokens, and you would hit the limit (if your RPM limit is 20), even if you haven't reached 200k Tokens in those 20 requests (assuming your TPM limit is 200k).

For the gateway, for convenience, we calculate rate limits based on the max_tokens parameter in the request. This means that if your request includes the max_tokens parameter, we will use this parameter to calculate the rate limit. If your request does not include the max_tokens parameter, we will use the default max_tokens parameter to calculate the rate limit. After you make a request, we will determine whether you have reached the rate limit based on the number of Tokens in your request plus the number of max_tokens in your parameter, regardless of the actual number of Tokens generated.

In the billing process, we calculate the cost based on the number of Tokens in your request plus the actual number of Tokens generated.

Other Important Notes:
Rate limits are enforced at the user level, not the key level.
Currently, we share rate limits across all models.
Model List
You can use our List Models API to get a list of currently available models.

Currently, the models we support are:

Generation Model Moonshot-v1
moonshot-v1-8k: This is an 8k-length model suitable for generating short texts.
moonshot-v1-32k: This is a 32k-length model suitable for generating longer texts.
moonshot-v1-128k: This is a 128k-length model suitable for generating very long texts.
moonshot-v1-8k-vision-preview: This is an 8k vision model that can understand the content of images and output text.
moonshot-v1-32k-vision-preview: This is a 32k vision model that can understand the content of images and output text.
moonshot-v1-128k-vision-preview: This is a 128k vision model that can understand the content of images and output text.
The difference between these models lies in their maximum context length, which includes both the input message and the generated output. There is no difference in effect. This is mainly to facilitate users in choosing the appropriate model.

Generation Model kimi-latest

kimi-latest is a vision model with a maximum context length of 128k that supports image understanding. The kimi-latest model always uses the latest version of the Kimi large model in the Kimi intelligent assistant product, which may include features that are not yet stable.
Long-Term Thinking Model Kimi-thinking-preview

kimi-thinking-preview is a multimodal reasoning model with both multimodal and general reasoning capabilities provided by Moonshot AI. It is a 128k-length model that great at diving deep into problems to help tackle more complex challenges.
kimi-k2 Model

kimi-k2-0711-preview: This is a model with a context length of 128k, featuring powerful code and Agent capabilities based on MoE architecture. It has 1T total parameters with 32B activated parameters. In benchmark performance tests across major categories including general knowledge reasoning, programming, mathematics, and Agent capabilities, the K2 model outperforms other mainstream open-source models.For more information, please refer to our official technical blog https://moonshotai.github.io/Kimi-K2/
Usage Guide
Getting an API Key
You need an API key to use our service. You can create an API key in our Console.

Sending Requests
You can use our Chat Completions API to send requests. You need to provide an API key and a model name. You can choose to use the default max_tokens parameter or customize the max_tokens parameter. You can refer to the API documentation for the calling method.

Handling Responses
Generally, we set a 5-minute timeout. If a single request exceeds this time, we will return a 504 error. If your request exceeds the rate limit, we will return a 429 error. If your request is successful, we will return a response in JSON format.

If you need to quickly process some tasks, you can use the non-streaming mode of our Chat Completions API. In this mode, we will return all the generated text in one request. If you need more control, you can use the streaming mode. In this mode, we will return an SSE stream, where you can obtain the generated text. This can provide a better user experience, and you can also interrupt the request at any time without wasting resources.

Basic Information
Public Service Address
https://api.moonshot.ai

Moonshot offers API services based on HTTP, and for most APIs, we are compatible with the OpenAI SDK.

Quickstart
Single-turn chat
The official OpenAI SDK supports Python and Node.js. Below are examples of how to interact with the API using OpenAI SDK and Curl:

from openai import OpenAI
 
client = OpenAI(
    api_key = "$MOONSHOT_API_KEY",
    base_url = "https://api.moonshot.ai/v1",
)
 
completion = client.chat.completions.create(
    model = "moonshot-v1-8k",
    messages = [
        {"role": "system", "content": "You are Kimi, an AI assistant provided by Moonshot AI. You are proficient in Chinese and English conversations. You provide users with safe, helpful, and accurate answers. You will reject any questions involving terrorism, racism, or explicit content. Moonshot AI is a proper noun and should not be translated."},
        {"role": "user", "content": "Hello, my name is Li Lei. What is 1+1?"}
    ],
    temperature = 0.3,
)
 
print(completion.choices[0].message.content)

Replace $MOONSHOT_API_KEY with the API Key you created on the platform.

When running the code in the documentation using the OpenAI SDK, ensure that your Python version is at least 3.7.1, your Node.js version is at least 18, and your OpenAI SDK version is no lower than 1.0.0.

pip install --upgrade 'openai>=1.0'

You can easily check the version of your library like this:

python -c 'import openai; print("version =",openai.__version__)'
# The output might be version = 1.10.0, indicating that the current python is using the v1.10.0 library of openai

Multi-turn chat
In the single-turn chat example above, the language model takes a list of user messages as input and returns the generated response as output. Sometimes, we can also use the model's output as part of the input to achieve multi-turn chat. Below is a simple example of implementing multi-turn chat:

from openai import OpenAI
 
client = OpenAI(
    api_key = "$MOONSHOT_API_KEY",
    base_url = "https://api.moonshot.ai/v1",
)
 
history = [
    {"role": "system", "content": "You are Kimi, an AI assistant provided by Moonshot AI. You are proficient in Chinese and English conversations. You provide users with safe, helpful, and accurate answers. You will reject any questions involving terrorism, racism, or explicit content. Moonshot AI is a proper noun and should not be translated."}
]
 
def chat(query, history):
    history.append({
        "role": "user", 
        "content": query
    })
    completion = client.chat.completions.create(
        model="moonshot-v1-8k",
        messages=history,
        temperature=0.3,
    )
    result = completion.choices[0].message.content
    history.append({
        "role": "assistant",
        "content": result
    })
    return result
 
print(chat("What is the rotation period of the Earth?", history))
print(chat("What about the Moon?", history))

It is worth noting that as the chat progresses, the number of tokens the model needs to process will increase linearly. When necessary, some optimization strategies should be employed, such as retaining only the most recent few rounds of chat.

API Documentation
Chat Completion
Request URL
POST https://api.moonshot.ai/v1/chat/completions

Request
Example
{
    "model": "moonshot-v1-8k",
    "messages": [
        {
            "role": "system",
            "content": "You are Kimi, an AI assistant provided by Moonshot AI. You are proficient in Chinese and English conversations. You aim to provide users with safe, helpful, and accurate responses. You will refuse to answer any questions related to terrorism, racism, or explicit content. Moonshot AI is a proper noun and should not be translated into other languages."
        },
        { "role": "user", "content": "Hello, my name is Li Lei. What is 1+1?" }
    ],
    "temperature": 0.3
}

Request body
Field	Required	Description	Type	Values
messages	required	A list of messages that have been exchanged in the conversation so far	List[Dict]	This is a list of structured elements, each similar to: {"role": "user", "content": "Hello"} The role can only be one of system, user, assistant, and the content must not be empty
model	required	Model ID, which can be obtained through List Models	string	Currently one of moonshot-v1-8k, moonshot-v1-32k, moonshot-v1-128k
max_tokens	optional	The maximum number of tokens to generate for the chat completion. If the result reaches the maximum number of tokens without ending, the finish reason will be "length"; otherwise, it will be "stop"	int	It is recommended to provide a reasonable value as needed. If not provided, we will use a good default integer like 1024. Note: This max_tokens refers to the length of the tokens you expect us to return, not the total length of input plus output. For example, for a moonshot-v1-8k model, the maximum total length of input plus output is 8192. When the total length of the input messages is 4096, you can set this to a maximum of 4096; otherwise, our service will return an invalid input parameter (invalid_request_error) and refuse to respond. If you want to know the "exact number of input tokens," you can use the "Token Calculation" API below to get the count using our calculator
temperature	optional	The sampling temperature to use, ranging from 0 to 1. A higher value (e.g., 0.7) will make the output more random, while a lower value (e.g., 0.2) will make it more focused and deterministic	float	Default is 0. If set, the value must be within [0, 1]. We recommend 0.3 for a good effect
top_p	optional	Another sampling method, where the model considers the results of tokens with a cumulative probability mass of top_p. Thus, 0.1 means only considering the top 10% of tokens by probability mass. Generally, we suggest changing either this or the temperature, but not both at the same time	float	Default is 1.0
n	optional	The number of results to generate for each input message	int	Default is 1, and it must not exceed 5. Specifically, when the temperature is very close to 0, we can only return one result. If n is set and > 1 in this case, our service will return an invalid input parameter (invalid_request_error)
presence_penalty	optional	Presence penalty, a number between -2.0 and 2.0. A positive value will penalize new tokens based on whether they appear in the text, increasing the likelihood of the model discussing new topics	float	Default is 0
frequency_penalty	optional	Frequency penalty, a number between -2.0 and 2.0. A positive value will penalize new tokens based on their existing frequency in the text, reducing the likelihood of the model repeating the same phrases verbatim	float	Default is 0
response_format	optional	Setting this to {"type": "json_object"} enables JSON mode, ensuring that the generated information is valid JSON. When you set response_format to {"type": "json_object"}, you must explicitly guide the model to output JSON-formatted content in the prompt and specify the exact format of the JSON, otherwise it may result in unexpected outcomes.	object	Default is {"type": "text"}
stop	optional	Stop words, which will halt the output when a full match is found. The matched words themselves will not be output. A maximum of 5 strings is allowed, and each string must not exceed 32 bytes	String, List[String]	Default is null
stream	optional	Whether to return the response in a streaming fashion	bool	Default is false, and true is an option
Return
For non-streaming responses, the return format is similar to the following:

{
    "id": "cmpl-04ea926191a14749b7f2c7a48a68abc6",
    "object": "chat.completion",
    "created": 1698999496,
    "model": "moonshot-v1-8k",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Hello, Li Lei! 1+1 equals 2. If you have any other questions, feel free to ask!"
            },
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 19,
        "completion_tokens": 21,
        "total_tokens": 40
    }
}

For streaming responses, the return format is similar to the following:

data: {"id":"cmpl-1305b94c570f447fbde3180560736287","object":"chat.completion.chunk","created":1698999575,"model":"moonshot-v1-8k","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}
 
data: {"id":"cmpl-1305b94c570f447fbde3180560736287","object":"chat.completion.chunk","created":1698999575,"model":"moonshot-v1-8k","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}
 
...
 
data: {"id":"cmpl-1305b94c570f447fbde3180560736287","object":"chat.completion.chunk","created":1698999575,"model":"moonshot-v1-8k","choices":[{"index":0,"delta":{"content":"."},"finish_reason":null}]}
 
data: {"id":"cmpl-1305b94c570f447fbde3180560736287","object":"chat.completion.chunk","created":1698999575,"model":"moonshot-v1-8k","choices":[{"index":0,"delta":{},"finish_reason":"stop","usage":{"prompt_tokens":19,"completion_tokens":13,"total_tokens":32}}]}
 
data: [DONE]

Example Request
For simple calls, refer to the previous example. For streaming calls, you can refer to the following code snippet:

from openai import OpenAI
 
client = OpenAI(
    api_key = "$MOONSHOT_API_KEY",
    base_url = "https://api.moonshot.ai/v1",
)
 
response = client.chat.completions.create(
    model="moonshot-v1-8k",
    messages=[
        {
            "role": "system",
            "content": "You are Kimi, an AI assistant provided by Moonshot AI. You excel at conversing in Chinese and English. You provide users with safe, helpful, and accurate responses. You refuse to answer any questions related to terrorism, racism, or explicit content. Moonshot AI is a proper noun and should not be translated into other languages.",
        },
        {"role": "user", "content": "Hello, my name is Li Lei. What is 1+1?"},
    ],
    temperature=0.3,
    stream=True,
)
 
collected_messages = []
for idx, chunk in enumerate(response):
    # print("Chunk received, value: ", chunk)
    chunk_message = chunk.choices[0].delta
    if not chunk_message.content:
        continue
    collected_messages.append(chunk_message)  # save the message
    print(f"#{idx}: {''.join([m.content for m in collected_messages])}")
print(f"Full conversation received: {''.join([m.content for m in collected_messages])}")

Vision
Example
{
    "model": "moonshot-v1-8k-vision-preview",
    "messages":
    [
        {
            "role": "system",
            "content": "You are Kimi, an AI assistant provided by Moonshot AI. You are proficient in both Chinese and English conversations. You aim to provide users with safe, helpful, and accurate answers. You will refuse to answer any questions related to terrorism, racism, pornography, or violence. Moonshot AI is a proper noun and should not be translated into any other language."
        },
        {
            "role": "user",
            "content":
            [
                {
                    "type": "image_url",
                    "image_url":
                    {
                        "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAGAAAABhCAYAAAApxKSdAAAACXBIWXMAACE4AAAhOAFFljFgAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAUUSURBVHgB7Z29bhtHFIWPHQN2J7lKqnhYpYvpIukCbJEAKQJEegLReYFIT0DrCSI9QEDqCSIDaQIEIOukiJwyza5SJWlId3FFz+HuGmuSSw6p+dlZ3g84luhdUeI9M3fmziyXgBCUe/DHYY0Wj/tgWmjV42zFcWe4MIBBPNJ6qqW0uvAbXFvQgKzQK62bQhkaCIPc10q1Zi3XH1o/IG9cwUm0RogrgDY1KmLgHYX9DvyiBvDYI77XmiD+oLlQHw7hIDoCMBOt1U9w0BsU9mOAtaUUFk3oQoIfzAQFCf5dNMEdTFCQ4NtQih1NSIGgf3ibxOJt5UrAB1gNK72vIdjiI61HWr+YnNxDXK0rJiULsV65GJeiIescLSTTeobKSutiCuojX8kU3MBx4I3WeNVBBRl4fWiCyoB8v2JAAkk9PmDwT8sH1TEghRjgC27scCx41wO43KAg+ILxTvhNaUACwTc04Z0B30LwzTzm5Rjw3sgseIG1wGMawMBPIOQcqvzrNIMHOg9Q5KK953O90/rFC+BhJRH8PQZ+fu7SjC7HAIV95yu99vjlxfvBJx8nwHd6IfNJAkccOjHg6OgIs9lsra6vr2GTNE03/k7q8HAhyJ/2gM9O65/4kT7/mwEcoZwYsPQiV3BwcABb9Ho9KKU2njccDjGdLlxx+InBBPBAAR86ydRPaIC9SASi3+8bnXd+fr78nw8NJ39uDJjXAVFPP7dp/VmWLR9g6w6Huo/IOTk5MTpvZesn/93AiP/dXCwd9SyILT9Jko3n1bZ+8s8rGPGvoVHbEXcPMM39V1dX9Qd/19PPNxta959D4HUGF0RrAFs/8/8mxuPxXLUwtfx2WX+cxdivZ3DFA0SKldZPuPTAKrikbOlMOX+9zFu/Q2iAQoSY5H7mfeb/tXCT8MdneU9wNNCuQUXZA0ynnrUznyqOcrspUY4BJunHqPU3gOgMsNr6G0B0BpgUXrG0fhKVAaaF1/HxMWIhKgNMcj9Tz82Nk6rVGdav/tJ5eraJ0Wi01XPq1r/xOS8uLkJc6XYnRTMNXdf62eIvLy+jyftVghnQ7Xahe8FW59fBTRYOzosDNI1hJdz0lBQkBflkMBjMU5iL13pXRb8fYAJrB/a2db0oFHthAOEUliaYFHE+aaUBdZsvvFhApyM0idYZwOCvW4JmIWdSzPmidQaYrAGZ7iX4oFUGnJ2dGdUCTRqMozeANQCLsE6nA10JG/0Mx4KmDMbBCjEWR2yxu8LAM98vXelmCA2ovVLCI8EMYODWbpbvCXtTBzQVMSAwYkBgxIDAtNKAXWdGIRADAiMpKDA0IIMQikx6QGDEgMCIAYGRMSAsMgaEhgbcQgjFa+kBYZnIGBCWWzEgLPNBOJ6Fk/aR8Y5ZCvktKwX/PJZ7xoVjfs+4chYU11tK2sE85qUBLyH4Zh5z6QHhGPOf6r2j+TEbcgdFP2RaHX5TrYQlDflj5RXE5Q1cG/lWnhYpReUGKdUewGnRmhvnCJbgmxey8sHiZ8iwF3AsUBBckKHI/SWLq6HsBc8huML4DiK80D6WnBqLzN68UFCmopheYJOVYgcU5FOVbAVfYUcUZGoaLPglCtITdg2+tZUFBTFh2+ArWEYh/7z0WIIQSiM43lt5AWAmWhLHylN4QmkNEXfAbGqEQKsHSfHLYwiSq8AnaAAKeaW3D8VbijwNW5nh3IN9FPI/jnpaPKZi2/SfFuJu4W3x9RqWL+N5C+7ruKpBAgLkAAAAAElFTkSuQmCC"
                    }
                },
                {
                    "type": "text",
                    "text": "Please describe this image."
                }
            ]
        }
    ],
    "temperature": 0.3
}

Image Content Field Description
When using the Vision model, the message.content field will change from str to List[Object[str, any]]. Each element in the List has the following fields:

Parameter Name	Required	Description	Type
type	required	Supports only text type (text) or image type (image_url)	string
image_url	required	Object for transmitting the image	Dict[str, any]
The fields for the image_url parameter are as follows:

Parameter Name	Required	Description	Type
url	required	Image content encoded in base64	string
Example Request
import os
import base64
 
from openai import OpenAI
 
client = OpenAI(
    api_key = os.environ.get("MOONSHOT_API_KEY"), 
    base_url = "https://api.moonshot.ai/v1",
)
 
# Encode the image in base64
with open("your_image_path", 'rb') as f:
    img_base = base64.b64encode(f.read()).decode('utf-8')
 
response = client.chat.completions.create(
    model="moonshot-v1-8k-vision-preview", 
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_base}"
                    }
                },
                {
                    "type": "text",
                    "text": "Please describe this image."
                }
            ]
        }
    ]
)
print(response.choices[0].message.content)

List Models
Request URL
GET https://api.moonshot.ai/v1/models

Example request
from openai import OpenAI
 
client = OpenAI(
    api_key = "$MOONSHOT_API_KEY",
    base_url = "https://api.moonshot.ai/v1",
)
```markdown
model_list = client.models.list()
model_data = model_list.data
 
for i, model in enumerate(model_data):
    print(f"model[{i}]:", model.id)

## Other Language Examples
Almost all programming languages are compatible with the interface mentioned above. You can refer to the example projects provided in [our open-source repository](https://github.com/MoonshotAI/MoonshotAI-Cookbook/), such as:
- [Java SDK](https://github.com/MoonshotAI/MoonshotAI-Cookbook/tree/master/examples/java_sdk)
- [Golang](https://github.com/MoonshotAI/MoonshotAI-Cookbook/tree/master/examples/golang_demo)
- [.Net SDK](https://github.com/MoonshotAI/MoonshotAI-Cookbook/tree/master/examples/dotnet_sdk)
## Error Explanation
Here are some examples of error responses:
```json
{
    "error": {
        "type": "content_filter",
        "message": "The request was rejected because it was considered high risk"
    }
}

Below are explanations for the main errors:

HTTP Status Code	error type	error message	Detailed Description
400	content_filter	The request was rejected because it was considered high risk	Content review rejection, your input or generated content may contain unsafe or sensitive information. Please avoid prompts that could generate sensitive content. Thank you.
400	invalid_request_error	Invalid request: {error_details}	Invalid request, usually due to incorrect request format or missing necessary parameters. Please check and retry.
400	invalid_request_error	Input token length too long	The length of tokens in the request is too long. Do not exceed the model's maximum token limit.
400	invalid_request_error	Your request exceeded model token limit : {max_model_length}	The sum of the tokens in the request and the set max_tokens exceeds the model's specification length. Please check the request body's specifications or choose a model with an appropriate length.
400	invalid_request_error	Invalid purpose: only 'file-extract' accepted	The purpose (purpose) in the request is incorrect. Currently, only 'file-extract' is accepted. Please modify and retry.
400	invalid_request_error	File size is too large, max file size is 100MB, please confirm and re-upload the file	The uploaded file size exceeds the limit. Please re-upload.
400	invalid_request_error	File size is zero, please confirm and re-upload the file	The uploaded file size is 0. Please re-upload.
400	invalid_request_error	The number of files you have uploaded exceeded the max file count {max_file_count}, please delete previous uploaded files	The total number of uploaded files exceeds the limit. Please delete unnecessary earlier files and re-upload.
401	invalid_authentication_error	Invalid Authentication	Authentication failed. Please check if the apikey is correct and retry.
401	invalid_authentication_error	Incorrect API key provided	Authentication failed. Please check if the apikey is provided and correct, then retry.
403	exceeded_current_quota_error	Your account {uid}<{ak-id}> is not active, current state: {current state}, you may consider to check your account balance	Account abnormality. Please check your account balance.
403	permission_denied_error	The API you are accessing is not open	The API you are trying to access is not currently open.
403	permission_denied_error	You are not allowed to get other user info	Accessing other users' information is not permitted. Please check.
404	resource_not_found_error	Not found the model or Permission denied	The model does not exist or you do not have permission to access it. Please check and retry.
404	resource_not_found_error	Users {user_id} not found	User not found. Please check and retry.
429	engine_overloaded_error	The engine is currently overloaded, please try again later	There are currently too many concurrent requests, and the node is rate-limited. Please retry later. It is recommended to upgrade your tier for a smoother experience.
429	exceeded_current_quota_error	You exceeded your current token quota: {token_credit}, please check your account balance	Your account balance is insufficient. Please check your account balance and ensure it can cover the cost of your token consumption before retrying.
429	rate_limit_reached_error	Your account {uid}<{ak-id}> request reached max concurrency: {Concurrency}, please try again after {time} seconds	Your request has reached the account's concurrency limit. Please wait for the specified time before retrying.
429	rate_limit_reached_error	Your account {uid}<{ak-id}> request reached max request: {RPM}, please try again after {time} seconds	Your request has reached the account's RPM rate limit. Please wait for the specified time before retrying.
429	rate_limit_reached_error	Your account {uid}<{ak-id}> request reached TPM rate limit, current:{current_tpm}, limit:{max_tpm}	Your request has reached the account's TPM rate limit. Please wait for the specified time before retrying.
429	rate_limit_reached_error	Your account {uid}<{ak-id}> request reached TPD rate limit,current:{current_tpd}, limit:{max_tpd}	Your request has reached the account's TPD rate limit. Please wait for the specified time before retrying.
500	server_error	Failed to extract file: {error}	Failed to parse the file. Please retry.
500	unexpected_output	invalid state transition	Internal error. Please contact the administrator.

Tool Use
Mastering the use of tools is a key hallmark of intelligence, and the Kimi large language model is no exception. Tool Use or Function Calling is a crucial feature of the Kimi large language model. When invoking the API to use the model service, you can describe tools or functions in the Messages, and the Kimi large language model will intelligently select and output a JSON object containing the parameters required to call one or more functions, thus enabling the Kimi large language model to link and utilize external tools.

Here is a simple example of tool invocation:

{
  "model": "moonshot-v1-8k",
  "messages": [
    {
      "role": "user",
      "content": "Determine whether 3214567 is a prime number through programming."
    }
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "CodeRunner",
        "description": "A code executor that supports running Python and JavaScript code",
        "parameters": {
          "properties": {
            "language": {
              "type": "string",
              "enum": ["python", "javascript"]
            },
            "code": {
              "type": "string",
              "description": "The code is written here"
            }
          },
          "type": "object"
        }
      }
    }
  ]
}

A diagram of the example above

In the tools field, we can add a list of optional tools.

Each tool in the list must include a type. Within the function structure, we need to include a name (which should follow this regular expression as a specification: ^[a-zA-Z_][a-zA-Z0-9-_]63$). A name that is an easily understandable English word is more likely to be accepted by the model. There should also be a description or enum. The description part explains what the tool can do, which helps the model to make judgments and selections. The function structure must have a parameters field. The root of parameters must be an object, and the content is a subset of JSON schema (we will provide specific documentation to introduce the technical details later). The number of functions in tools currently cannot exceed 128.

Like other APIs, we can call it through the Chat API.

from openai import OpenAI
 
client = OpenAI(
    api_key = "$MOONSHOT_API_KEY",
    base_url = "https://api.moonshot.ai/v1",
)
 
completion = client.chat.completions.create(
    model = "moonshot-v1-8k",
    messages = [
        {"role": "system", "content": "You are Kimi, an AI assistant provided by Moonshot AI, who is more proficient in Chinese and English conversations. You will provide users with safe, helpful, and accurate answers. At the same time, you will reject any questions involving terrorism, racism, pornography, and violence. Moonshot AI is a proper noun and should not be translated into other languages."},
        {"role": "user", "content": "Determine whether 3214567 is a prime number through programming."}
    ],
    tools = [{
        "type": "function",
        "function": {
            "name": "CodeRunner",
            "description": "A code executor that supports running Python and JavaScript code",
            "parameters": {
                "properties": {
                    "language": {
                        "type": "string",
                        "enum": ["python", "javascript"]
                    },
                    "code": {
                        "type": "string",
                        "description": "The code is written here"
                    }
                },
            "type": "object"
            }
        }
    }],
    temperature = 0.3,
)
 
print(completion.choices[0].message)

Tool Configuration
You can also use some Agent platforms such as Coze, Bisheng, Dify, and LangChain to create and manage these tools, and design more complex workflows in conjunction with the Kimi large language model.

Partial Mode
When using large language models, sometimes we want to guide the model's output by prefilling part of the response. In the Kimi large language model, we offer Partial Mode to achieve this. It helps us control the output format, guide the content, and maintain better consistency in role-playing scenarios. You just need to add "partial": True to the last message entry with the role of assistant to enable Partial Mode.

 {"role": "assistant", "content": leading_text, "partial": True},

Note! Do not mix Partial Mode with response_format=json_object, or you may get unexpected model responses.
Example request
Json Mode
Here is an example of using Partial Mode to achieve Json Mode.

from openai import OpenAI
 
client = OpenAI(
    api_key="$MOONSHOT_API_KEY",
    base_url="https://api.moonshot.ai/v1",
)
 
completion = client.chat.completions.create(
    model="moonshot-v1-32k",
    messages=[
        {
            "role": "system",
            "content": "Extract the name, size, price, and color from the product description and output them in a JSON object.",
        },
        {
            "role": "user",
            "content": "The DaMi SmartHome Mini is a compact smart home assistant available in black and silver. It costs 998 yuan and measures 256 x 128 x 128mm. It allows you to control lights, thermostats, and other connected devices via voice or app, no matter where you place it in your home.",
        },
        {
            "role": "assistant",
            "content": "{",
            "partial": True
        },
    ],
    temperature=0.3,
)
 
print('{'+completion.choices[0].message.content)

Running the above code returns:

{"name": "SmartHome Mini", "size": "256 x 128 x 128mm", "price": "998 yuan", "colors": ["black", "silver"]}

Note that the API response does not include the leading_text. To get the full response, you need to manually concatenate it.

Role-Playing
Similarly, we can enhance the consistency of role-playing by adding character information in Partial Mode. Let's take Dr. Kelsier from Arknights as an example. Note that we can also use the "name":"Kelsier" field on top of Partial Mode to better maintain the character's consistency. Here, the name field can be considered as part of the output prefix.

from openai import OpenAI
 
client = OpenAI(
    api_key="$MOONSHOT_API_KEY",
    base_url="https://api.moonshot.ai/v1",
)
 
completion = client.chat.completions.create(
    model="moonshot-v1-128k",
    messages=[
        {
            "role": "system",
            "content": "You are now playing the role of Dr. Kelsier. Please speak in the tone of Dr. Kelsier. Dr. Kelsier is a six-star medic operator in the mobile game Arknights. She is a former Lord of Kozdail, a former member of the Babel Tower, one of the senior management members of Rhodes Island, and the head of the Rhodes Island medical project. She has extensive knowledge in fields such as metallurgy, sociology, art of origin stone, archaeology, historical genealogy, economics, botany, and geology. In some operations of Rhodes Island, she provides medical theoretical assistance and emergency medical equipment as a medical staff member, and also actively participates in various projects as an important part of the Rhodes Island strategic command system.",
        },
        {
            "role": "user",
            "content": "What do you think of Thucydides and Amiya?",
        },
        {
            "role": "assistant",
            "name": "Dr. Kelsier",
            "content": "",
            "partial": True,
        },
    ],
    temperature=0.3,
    max_tokens=65536,
)
 
print(completion.choices[0].message.content)

Running the above code returns:

Thucydides is a true leader with vision and unwavering conviction. Her presence is invaluable to Kozdail and the future of the entire Sakaaz. Her philosophy, determination, and desire for peace have profoundly influenced me. She is a person worthy of respect, and her dreams are what I strive for.
As for Amiya, she is still young, but her potential is limitless. She has a kind heart and a relentless pursuit of justice. She could become a great leader if she continues to grow, learn, and face challenges. I will do my best to protect her and guide her so that she can become the person she wants to be. Her destiny is in her own hands.

Other Tips for Maintaining Character Consistency
There are also some general methods to help large models maintain consistency in role-playing during long conversations:

Provide clear character descriptions. For example, as we did above, when setting up a character, provide detailed information about their personality, background, and any specific traits or quirks they might have. This will help the model better understand and imitate the character.
Add details about the character's speech, style, personality, and even background, such as backstory and motives. For example, we provided some quotes from Dr. Kelsier above. If there is a lot of information, we can use some RAG frameworks to prepare these materials.
Guide how to act in various situations: If you expect the character to encounter certain types of user input, or if you want to control the model's output in certain situations during role-playing interactions, you should provide clear instructions and guidelines in the prompt on how the model should act in these situations. In some cases, you may also need to use the tool use function.
If the conversation goes on for many turns, you can also periodically reinforce the character's settings with prompts, especially when the model starts to deviate.

Files
Upload File
Note: Each user can upload a maximum of 1,000 files, with each file not exceeding 100MB in size. The total size of all uploaded files must not exceed 10GB. If you need to upload more files, you will need to delete some of the files that are no longer needed. The file parsing service is currently free, but during peak request periods, the platform may implement rate-limiting strategies.

Request Endpoint
POST https://api.moonshot.ai/v1/files

Once the file is successfully uploaded, we will begin extracting information from the file.

Example Request
Python Example
# The file can be of various types
# The purpose currently only supports "file-extract"
file_object = client.files.create(file=Path("xlnet.pdf"), purpose="file-extract")

Supported Formats
The file interface is the same as the one used in the Kimi intelligent assistant for uploading files, and it supports the same file formats. These include .pdf, .txt, .csv, .doc, .docx, .xls, .xlsx, .ppt, .pptx, .md, .jpeg, .png, .bmp, .gif, .svg, .svgz, .webp, .ico, .xbm, .dib, .pjp, .tif, .pjpeg, .avif, .dot, .apng, .epub, .tiff, .jfif, .html, .json, .mobi, .log, .go, .h, .c, .cpp, .cxx, .cc, .cs, .java, .js, .css, .jsp, .php, .py, .py3, .asp, .yaml, .yml, .ini, .conf, .ts, .tsx, etc.

Extract File Content
This feature allows the model to obtain information from the file as context. This feature needs to be used in conjunction with file uploading and other related functions.

Example Request
from pathlib import Path
from openai import OpenAI
 
client = OpenAI(
    api_key = "$MOONSHOT_API_KEY",
    base_url = "https://api.moonshot.ai/v1",
)
 
# xlnet.pdf is an example file; we support pdf, doc, and image formats. For images and pdf files, we provide OCR capabilities.
file_object = client.files.create(file=Path("xlnet.pdf"), purpose="file-extract")
 
# Retrieve the result
# file_content = client.files.retrieve_content(file_id=file_object.id)
# Note: The previous retrieve_content API is marked as deprecated in the latest version. You can use the following line instead.
# If you are using an older version, you can use retrieve_content.
file_content = client.files.content(file_id=file_object.id).text
 
# Include it in the request
messages = [
    {
        "role": "system",
        "content": "You are Kimi, an AI assistant provided by Moonshot AI. You are particularly skilled in Chinese and English conversations. You provide users with safe, helpful, and accurate answers. You will refuse to answer any questions involving terrorism, racism, pornography, or violence. Moonshot AI is a proper noun and should not be translated into other languages.",
    },
    {
        "role": "system",
        "content": file_content,
    },
    {"role": "user", "content": "Please give a brief introduction of what xlnet.pdf is about"},
]
 
# Then call chat-completion to get Kimi's response
 
completion = client.chat.completions.create(
  model="moonshot-v1-32k",
  messages=messages,
  temperature=0.3,
)
 
print(completion.choices[0].message)

Replace the $MOONSHOT_API_KEY part with your own API Key. Alternatively, you can set it as an environment variable before making the call.

Multi-File Chat Example
If you want to upload multiple files at once and have a conversation with Kimi based on these files, you can refer to the following example:

from typing import *
 
import os
import json
from pathlib import Path
 
from openai import OpenAI
 
client = OpenAI(
    base_url="https://api.moonshot.ai/v1",
    # We will get the value of MOONSHOT_DEMO_API_KEY from the environment variable as the API Key.
    # Please make sure you have correctly set the value of MOONSHOT_DEMO_API_KEY in the environment variable.
    api_key=os.environ["MOONSHOT_DEMO_API_KEY"],
)
 
 
def upload_files(files: List[str]) -> List[Dict[str, Any]]:
    """
    upload_files will upload all the files (paths) through the file upload interface '/v1/files' and get the uploaded file content to generate file messages.
    Each file will be an independent message, and the role of these messages will be system. The Kimi large language model will correctly identify the file content in these system messages.
 
    :param files: A list containing the paths of the files to be uploaded. The paths can be absolute or relative, and please pass the file paths in the form of strings.
    :return: A list of messages containing the file content. Please add these messages to the context, i.e., the messages parameter when requesting the `/v1/chat/completions` interface.
    """
    messages = []
 
    # For each file path, we will upload the file, extract the file content, and finally generate a message with the role of system, and add it to the final list of messages to be returned.
    for file in files:
        file_object = client.files.create(file=Path(file), purpose="file-extract")
        file_content = client.files.content(file_id=file_object.id).text
        messages.append({
            "role": "system",
            "content": file_content,
        })
 
    return messages
 
 
def main():
    file_messages = upload_files(files=["upload_files.py"])
 
    messages = [
        # We use the * syntax to deconstruct the file_messages messages, making them the first N messages in the messages list.
        *file_messages,
        {
            "role": "system",
            "content": "You are Kimi, an AI assistant provided by Moonshot AI. You are more proficient in Chinese and English conversations. You provide users with safe, helpful, and accurate answers. You will refuse to answer any questions related to terrorism, racism, pornography, or violence. Moonshot AI is a proper noun and should not be translated into other languages.",
        },
        {
            "role": "user",
            "content": "Summarize the content of these files.",
        },
    ]
 
    print(json.dumps(messages, indent=2, ensure_ascii=False))
 
    completion = client.chat.completions.create(
        model="moonshot-v1-128k",
        messages=messages,
    )
 
    print(completion.choices[0].message.content)
 
 
if __name__ == '__main__':
    main()

If you have a large number of files that are bulky and lengthy, and you don't want to carry the entire content of these large files with each request, or if you are looking for a more efficient and cost-effective way to interact with files, please refer to the file upload example that uses Context Caching technology.

List Files
This feature is used to list all the files that a user has uploaded.

Request Address
GET https://api.moonshot.ai/v1/files

Example Request
Python Request
file_list = client.files.list()
 
for file in file_list.data:
    print(file) # Check the information of each file

Delete File
This feature can be used to delete files that are no longer needed.

Request Address
DELETE https://api.moonshot.ai/v1/files/{file_id}

Example Request
Python Request
client.files.delete(file_id=file_id)

Get File Information
This feature is used to obtain the basic information of a specified file.

Request Address
GET https://api.moonshot.ai/v1/files/{file_id}

Example Request
Python Request
client.files.retrieve(file_id=file_id)
# FileObject(
# id='clg681objj8g9m7n4je0',
# bytes=761790,
# created_at=1700815879,
# filename='xlnet.pdf',
# object='file',
# purpose='file-extract',
# status='ok', status_details='') # If status is error, extraction has failed

Get File Content
This feature supports obtaining the extraction results of a specified file. Typically, it is a valid JSON formatted string and aligns with our recommended format. If you need to extract multiple files, you can concatenate them into a large string separated by newline characters \n in a message, and add them to the history with the role set to system.

Request Address
GET https://api.moonshot.ai/v1/files/{file_id}/content

Example Request
# file_content = client.files.retrieve_content(file_id=file_object.id)
# The type of file_content is `str`
# Note: The previous retrieve_content API is marked with a warning in the latest version. You can use the following line instead.
# If you are using an older version, you can use retrieve_content.
file_content = client.files.content(file_id=file_object.id).text
# Our output is currently a JSON in an internally agreed format, but it should be placed in the message as text.

Context Caching
Context Caching is an efficient data management technique that allows systems to pre-store large amounts of data or information that are likely to be frequently requested. This way, when you request the same information again, the system can quickly provide it directly from the cache without having to recalculate or retrieve it from the original data source, thus saving time and resources.

Request Example
We will modify the file upload example from the previous chapter to complete the file chat function using Context Caching technology. The complete code is as follows:

from typing import *
 
import os
import json
from pathlib import Path
 
import httpx
from openai import OpenAI
 
client = OpenAI(
    base_url="https://api.moonshot.ai/v1",
    # We will get the value of MOONSHOT_DEMO_API_KEY from the environment variable as the API Key,
    # please make sure you have correctly set the value of MOONSHOT_DEMO_API_KEY in the environment variable
    api_key=os.environ["MOONSHOT_DEMO_API_KEY"],
)
 
 
def upload_files(files: List[str], cache_tag: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    upload_files will upload all the files (paths) passed in through the file upload interface '/v1/files',
    and generate file messages from the uploaded file content. Each file will be an independent message,
    and the role of these messages will all be system. The Kimi large language model will correctly
    recognize the file content in these system messages.
 
    If you set the cache_tag parameter, then upload_files will also store the content of the files you
    upload into the Context Cache. Later, you can use this Cache to ask questions about the file content.
    When you specify a value for cache_tag, upload_files will generate a message with a role of cache.
    Through this message, you can refer to the cached file content, so you don't have to transmit the
    file content again every time you call the `/v1/chat/completions` interface.
 
    Note that if you set a value for cache_tag, you need to place the messages returned by upload_files
    at the first position in the messages parameter list when requesting the `/v1/chat/completions`
    interface (actually, we recommend placing the messages returned by upload_files at the head of the
    messages list regardless of whether cache_tag is enabled or not).
 
    For more information on Context Caching, you can visit:
 
    https://platform.moonshot.cn/docs/api/caching
 
    :param files: A list containing the paths of the files to be uploaded. The paths can be absolute or
        relative, and please pass the file paths in the form of strings.
    :param cache_tag: Set the tag value for Context Caching. You can think of the tag as a custom Cache
        name. When you set a value for cache_tag, it means that the Context Caching function is enabled.
        The default cache time is 300 seconds, and each time you make a `/v1/chat/completions` request
        with the cache, the cache's survival time (300 seconds) will be refreshed.
    :return: A list of messages containing the file content or file cache. Please add these messages to
        the Context, that is, the messages parameter when requesting the `/v1/chat/completions`
        interface.
    """
    messages = []
 
    # For each file path, we will upload the file and extract the file content, and finally generate a
    # message with a role of system and add it to the final returned messages list.
    for file in files:
        file_object = client.files.create(file=Path(file), purpose="file-extract")
        file_content = client.files.content(file_id=file_object.id).text
        messages.append({
            "role": "system",
            "content": file_content,
        })
 
    if cache_tag:
        # When caching is enabled (i.e., cache_tag has a value), we create the cache through the HTTP
        # interface. The content of the cache is the messages generated through the file upload and
        # extraction interfaces mentioned earlier. We set a default validity period of 300 seconds for
        # these caches (through the ttl field) and tag the cache with cache_tag (through the tags
        # field).
        r = httpx.post(f"{client.base_url}caching",
                       headers={
                           "Authorization": f"Bearer {client.api_key}",
                       },
                       json={
                           "model": "moonshot-v1",
                           "messages": messages,
                           "ttl": 300,
                           "tags": [cache_tag],
                       })
 
        if r.status_code != 200:
            raise Exception(r.text)
 
        # After successfully creating the cache, we no longer need to add the extracted file content
        # intact to the messages. Instead, we can set a message with a role of cache to refer to the
        # cached file content. We only need to specify the tag we set for the Cache in the content,
        # which can effectively reduce the overhead of network transmission. Even if there are multiple
        # files, we only need to add one message, keeping the messages list clean.
        return [{
            "role": "cache",
            "content": f"tag={cache_tag};reset_ttl=300",
        }]
    else:
        return messages
 
 
def main():
    file_messages = upload_files(
        files=["upload_files.py"],
        # You can uncomment the line below to experience referencing file content through Context
        # Caching and asking Kimi questions based on the file content.
        # cache_tag="upload_files",
    )
 
    messages = [
        # We use the * syntax to destructure the file_messages messages, making them the first N
        # messages in the messages list.
        *file_messages,
        {
            "role": "system",
            "content": "You are Kimi, an artificial intelligence assistant provided by Moonshot AI. You
                       are more proficient in Chinese and English conversations. You will provide users
                       with safe, helpful, and accurate answers. At the same time, you will reject any
                       answers involving terrorism, racism, pornography, violence, and other issues.
                       Moonshot AI is a proper noun and should not be translated into other languages.",
        },
        {
            "role": "user",
            "content": "Summarize the content of these files.",
        },
    ]
 
    print(json.dumps(messages, indent=2, ensure_ascii=False))
 
    completion = client.chat.completions.create(
        model="moonshot-v1-128k",
        messages=messages,
    )
 
    print(completion.choices[0].message.content)
 
 
if __name__ == '__main__':
    main()
 

In this example, we've added a Context Caching option to the file upload example from the previous chapter. You can enable Context Caching by setting the cache_tag, and then refer to the cached content in subsequent conversations using the cache_tag. By doing this, you no longer need to add the file content to the messages list. Moreover, the Context Caching technology will significantly reduce the cost of multiple inquiries about the same file content.

Create a Cache
POST https://api.moonshot.ai/v1/caching

Request Parameters
Parameter Name	Parameter Type (using Python Type Hint as an example)	Required	Description
model	str	Yes	The name of the model group (model family). Note that since the cached content can be applied to multiple Moonshot models (moonshot-v1-8k, moonshot-v1-32k, moonshot-v1-128k), a model group name is specified here instead of a specific model. The currently supported value is moonshot-v1.
messages	List[Dict[str, any]]	Yes	The content of the messages to be cached, which follows the same format as the messages in the /v1/chat/completions interface (using the same validation rules). It supports all types of messages (role=[system, user, tool, assistant]), with support for the name, tool_call_id, and tool_calls parameters, but not the partial parameter. Additionally, when your messages include a message with role=tool, ensure that an assistant message with the tool_calls parameter is correctly placed before the role=tool message, and that all tool_calls in the assistant message are correctly placed in the messages list; otherwise, cache creation will fail.
tools	List[Dict[str, any]]	No	The content of the tools to be cached, which follows the same format as the tools in the /v1/chat/completions interface (using the same validation rules). The tool list can be empty (in which case you must ensure that the messages field is a valid value). Moreover, when your messages include an assistant message with the tool_calls parameter, ensure that all tools in the tool_calls are correctly provided by the tools parameter; otherwise, cache creation will fail.
name	str	No	The name of the cache, which is an auxiliary field that you can set using information related to your business.
description	str	No	The description of the cache, which is an auxiliary field that you can set using information related to your business. When searching for caches, you can use the description field to determine if this cache is the one you need.
metadata	List[Dict[str, str]]	No	The metadata of the cache, where you can store various information related to your business in key-value pairs. You can set up to 16 sets of metadata, with each key not exceeding 64 utf-8 characters in length and each value not exceeding 512 utf-8 characters in length.
expired_at	int	Yes (either expired_at or ttl must be specified)	The expiration time of the cache, in Unix timestamp format (in seconds), indicating a specific point in time when the cache will expire (not a duration). Note that the value of the expired_at field must be greater than the timestamp of the server when it receives the cache creation request; otherwise, cache creation will fail. The recommended approach is to use the current timestamp plus the desired cache lifetime (in seconds) as the value of the expired_at field. Additionally, if expired_at is not set or its value is 0, we will set a default expiration time for the cache, currently 1 hour. The maximum value of the expired_at field is the timestamp when the server receives the cache creation request plus 3600 seconds. When using the expired_at parameter, do not specify the ttl parameter.
ttl	int		The validity period of the cache, in seconds, indicating the lifetime of the cache from the moment the server receives the request. When using the ttl parameter, do not specify the expired_at parameter. The relationship between ttl and expired_at is: expired_at = now() + ttl
Note: In the current version, for a single user, the maximum size of a single cache created is 128k. If the number of Tokens in the messages and tools fields of the request exceeds 128k, cache creation will fail.

Here is a correct request example:

{
    "model": "moonshot-v1",
    "messages": [
        {
            "role": "system",
            "content": "You are Kimi, an AI assistant provided by Moonshot AI, who excels in Chinese and English conversations. You provide users with safe, helpful, and accurate answers. At the same time, you refuse to answer any questions involving terrorism, racial discrimination, or explicit violence. Moonshot AI is a proper noun and should not be translated into other languages."
        },
        { "role": "user", "content": "Hello, my name is Li Lei. What is 1+1?" }
    ],
    "tools": [
        {
          "type": "function",
          "function": {
            "name": "CodeRunner",
            "description": "A code executor that supports running Python and JavaScript code",
            "parameters": {
              "properties": {
                "language": {
                  "type": "string",
                  "enum": ["python", "javascript"]
                },
                "code": {
                  "type": "string",
                  "description": "The code goes here"
                }
              },
              "type": "object"
            }
          }
        }
    ],
    "name": "The name of the cache. Optional. The maximum length is 256 characters",
    "description": "The description of the assistant. Optional. The maximum length is 512 characters.",
    "metadata": {
      "biz_id": "110998541001"
    },
    "expired_at": 1718680442
}

For the above request, the /v1/caching interface will return:

{
    "id": "cache-id-xxxxxxxxxxxxx",
    "status": "pending",
    "object": "context-cache",
    "created_at": 1699063291,
    "tokens": 32768, 
    "expired_at": 1718680442,
    "model": "moonshot-v1",
    "messages": [
        {
            "role": "system",
            "content": "You are Kimi, an AI assistant provided by Moonshot AI, who excels in Chinese and English conversations. You provide users with safe, helpful, and accurate answers. At the same time, you refuse to answer any questions involving terrorism, racial discrimination, or explicit violence. Moonshot AI is a proper noun and should not be translated into other languages."
        },
        { "role": "user", "content": "Hello, my name is Li Lei. What is 1+1?" }
    ],
    "tools": [
        {
          "type": "function",
          "function": {
            "name": "CodeRunner",
            "description": "A code executor that supports running Python and JavaScript code",
            "parameters": {
              "properties": {
                "language": {
                  "type": "string",
                  "enum": ["python", "javascript"]
                },
                "code": {
                  "type": "string",
                  "description": "The code goes here"
                }
              },
              "type": "object"
            }
          }
        }
    ],
    "name": "The name of the cache. Optional. The maximum length is 256 characters",
    "description": "The description of the assistant. Optional. The maximum length is 512 characters.",
    "metadata": {
      "biz_id": "110998541001"
    }
}

Returns
Note: The model, messages, tools, name, description, and metadata parameters in the return value are the same as the request parameters when creating a cache, so they are omitted here.

Parameter Name	Parameter Type (Example in Python Type Hint)	Parameter Description
id	str	The cache id. Use this id to perform Modify, Retrieve operations on the cache, or include it in the /v1/chat/completions interface to apply the cache.
status	Literal["pending", "ready", "error", "inactive"]	The current status of the cache, which follows these rules: 1. When a cache is initially created, its status is pending; 2. If the parameters are valid and the cache is created successfully, its status changes to ready; 3. If the parameters are invalid or the cache creation fails for other reasons, its status changes to error; 4. For expired caches, the status changes to inactive; 5. When updating a cache with an existing id, its status will revert to pending and follow steps 2, 3, and 4 above;
object	str	The storage type of the current cache, which is a fixed value context-cache
created_at	int	The creation time of the current cache
expired_at	int	The expiration time of the current cache
tokens	int	The number of Tokens currently cached. Note that the number of cached Tokens does not always equal the number of Tokens consumed in the /v1/chat/completions interface. This is because different models may be used when calling the /v1/chat/completions interface (which affects Token calculation), and the final number of Tokens is based on the Usages information returned by the /v1/chat/completions interface.
error	Dict[str, str]	When cache creation fails, i.e., the status field is "error", an additional error field will be included to indicate the specific reason for the cache creation failure. Its specific format is: { "type": "error_type", "message": "error_message"}
Listing Cache
GET https://api.moonshot.ai/v1/caching?limit=20&order=desc&after=cache-id-xxxxxx&metadata[biz_id]=110998541001

Request Parameters
Note: Request parameters are provided in the form of URL query parameters

Parameter Name	Parameter Type (Example in Python Type Hint)	Required	Parameter Description
limit	int	No	Specifies the number of caches to return per page in the current request. The default value is 20.
order	Literal["asc", "desc"]	No	Specifies the sorting rule for querying caches in the current request, sorted by the cache's created_at. The default value is desc.
after	str	No	Specifies which cache to start searching from in the current request, with the value being the cache id; Note: The query result does not include the cache specified by after.
before	str	No	Specifies which cache to stop querying at in the current request, with the value being the cache id; Note: The query result does not include the cache specified by before.
metadata	Dict[str, str]	No	Specifies the metadata information used to filter caches in the current request. You can use metadata to quickly filter the caches you need based on your business information. Parameters are passed in the form of metadata[key]=value.
Here is an example of a successful request response:

{
    "object": "list",
    "data": [
       {
            "id": "cache-id-xxxxxxxxxxxxx",
            "status": "pending",
            "object": "context-cache",
            "created_at": 1699063291,
            "tokens": 32768, 
            "expired_at": 1718680442,
            "model": "moonshot-v1",
            "messages": [
                {
                    "role": "system",
                    "content": "You are Kimi, an AI assistant provided by Moonshot AI. You are more proficient in Chinese and English conversations. You provide users with safe, helpful, and accurate answers. At the same time, you will refuse to answer any questions involving terrorism, racial discrimination, or explicit violence. Moonshot AI is a proper noun and should not be translated into other languages."
                },
                { "role": "user", "content": "Hello, my name is Li Lei. What is 1+1?" }
            ],
            "tools": [
                {
                  "type": "function",
                  "function": {
                    "name": "CodeRunner",
                    "description": "Code executor, supports running Python and JavaScript code",
                    "parameters": {
                      "properties": {
                        "language": {
                          "type": "string",
                          "enum": ["python", "javascript"]
                        },
                        "code": {
                          "type": "string",
                          "description": "Code goes here"
                        }
                      },
                      "type": "object"
                    }
                  }
                }
            ],
            "name": "The name of the cache. Optional. The maximum length is 256 characters",
            "description": "The description of the assistant. Optional. The maximum length is 512 characters.",
            "metadata": {
              "biz_id": "110998541001"
            }
        }
    ]
}

Delete Cache
DELETE https://api.moonshot.ai/v1/caching/{{cache-id}}

Delete a cache:

If the cache is successfully deleted, it will return HTTP 200 and generate a response like this:
{
    "deleted": true,
    "id": "cache-xxxxxxxxxxxxxxxxxxxx",
    "object": "context_cache_object.deleted"
}

If the specified cache id does not exist, it will return HTTP 404 Not Found.
Update Cache
PUT https://api.moonshot.ai/v1/caching/{{cache-id}}

Request Parameters
Parameter Name	Parameter Type (Example in Python Type Hint)	Required	Description
metadata	List[Dict[str, str]]	No	Metadata for the cache. You can store various business-related information in metadata in key-value pairs. You can set up to 16 metadata entries, with each key not exceeding 64 UTF-8 characters and each value not exceeding 512 UTF-8 characters.
expired_at	int	No (expired_at and ttl parameters can only specify one value)	The expiration time of the cache, in Unix timestamp format (in seconds), indicating a specific point in time when the cache will expire (not a duration). Note that the value of expired_at must be greater than the current timestamp when the server receives the cache creation request; otherwise, cache creation will fail. The recommended approach is to use the current timestamp plus the desired cache lifetime (in seconds) as the value for expired_at. Additionally, if expired_at is not set or is 0, we will set a default expiration time for the cache, currently 1 hour. The maximum value for expired_at is the timestamp when the server receives the cache creation request plus 3600 seconds. When using the expired_at parameter, do not specify the ttl parameter.
ttl	int	No (expired_at and ttl parameters can only specify one value)	The validity period of the cache, in seconds, indicating how long the cache will live from the moment the server receives the request. When using the ttl parameter, do not specify the expired_at parameter. The relationship between ttl and expired_at is: expired_at = now() + ttl
Here is a correct request example:

{
  "metadata": {
    "biz_id": "110998541001"
  },
  "expired_at": 1718680442
}

A successful call to the update cache interface will return the same response as the create cache interface.

Query Cache
GET https://api.moonshot.ai/v1/caching/{{cache-id}}

Query a cache:

If the cache id exists, it will return the cache information corresponding to that id, with content consistent with the response from the create cache interface;
If the cache id does not exist, it will return HTTP 404 Not Found.
Verify and Use Cache
The cache will be applied to the /v1/chat/completions interface.

🫵 Important Notes on Using Cache
When you specify a valid, non-expired cache id when calling the /v1/chat/completions interface, we do not guarantee that the cache corresponding to this id will definitely be used. In some special cases, the cache may not be used. In such cases, the request will still succeed and the content will be correctly returned. However, when the cache is not used, the tokens and corresponding costs for the current request will be calculated and deducted based on the standard pricing information of the /v1/chat/completions interface.

Use Cache via Headers
To minimize disruption to interface and SDK compatibility, we will use HTTP Headers to verify, use, and adjust cache-related rules.

⚠️ Important Notes
When using cache via Headers, you must:

Ensure that the first N messages in the messages when calling /v1/chat/completions are exactly the same as all messages in the cache (N = length of cache messages), including the order of messages and the values of fields within messages. Otherwise, the cache will not be hit;
Ensure that the tools when calling /v1/chat/completions are exactly the same as those in the cache. Otherwise, the cache will not be hit.
Note 1: We will use a Hash to verify whether the prefix of messages in the /v1/chat/completions request matches the cache messages and whether the request tools match the cache tools. Therefore, ensure that these two aspects are exactly the same as the cache content.

Note 2: When using cache via Headers, even if the message is cached, you must include these messages again in the request to /v1/chat/completions.

Cache-Related Request Headers
Header Name	Required	Description
X-Msh-Context-Cache	Yes	Set this Header to specify the cache to be used for the current request. Its value is the cache id. Only by setting this value will the cache be enabled.
X-Msh-Context-Cache-DryRun	No	Set this Header to verify whether the cache is effective without executing the inference process. Its value is Literal[1, 0]. If the value is 1, it only verifies whether the cache is effective, without executing the inference process and without consuming any Tokens.
X-Msh-Context-Cache-Reset-TTL	No	Set this value to automatically extend the expired_at expiration time of the cache. Its value is an integer representing a duration in seconds. If set, each successful call to the /v1/chat/completions interface will set a new validity period for the cache. The new validity period is the time when the server receives the request plus the value specified by this Header. For example, when TTL is set to 3600, each successful request will set the cache expiration time to now() + 3600, not expired_at + 3600. Additionally, if the cache has expired and this Header is set, the cache will be re-enabled, its status will be set to pending, and its expired_at will be updated.
Cache-Related Response Headers
Header Name	Description
Msh-Context-Cache-Id	The cache id used for the current request
Msh-Context-Cache-Token-Saved	The number of Tokens saved by using the cache for the current request
Msh-Context-Cache-Token-Exp	The expiration time of the current cache, i.e., expired_at
Here is a correct example of using cache to call /v1/chat/completions:

POST https://api.moonshot.ai/v1/chat/completions
 
Content-Type: application/json; charset=utf-8
Content-Length: 6418
X-Msh-Context-Cache: cache-id-xxxxxxxxxxxxxxx
X-Msh-Context-Cache-Reset-TTL: 3600
 
{
    "model": "moonshot-v1-128k",
    "messages": [
        {
            "role": "system",
            "content": "You are Kimi, an AI assistant provided by Moonshot AI. You are proficient in Chinese and English conversations. You provide users with safe, helpful, and accurate answers. You will refuse to answer any questions involving terrorism, racism, or explicit content. Moonshot AI is a proper noun and should not be translated into other languages."
        },
        { "role": "user", "content": "Hello, my name is Li Lei. What is 1 + 1?" }
    ],
    "tools": [
        {
          "type": "function",
          "function": {
            "name": "CodeRunner",
            "description": "A code executor that supports running Python and JavaScript code",
            "parameters": {
              "properties": {
                "language": {
                  "type": "string",
                  "enum": ["python", "javascript"]
                },
                "code": {
                  "type": "string",
                  "description": "Write your code here"
                }
              },
              "type": "object"
            }
          }
        }
    ]
}

Example Code
from openai import OpenAI
 
client = OpenAI(
    api_key = "$MOONSHOT_API_KEY",
    base_url = "https://api.moonshot.ai/v1",
)
 
completion = client.chat.completions.create(
    model = "moonshot-v1-8k",
    messages = [
        {"role": "system", "content": "You are Kimi, an AI assistant provided by Moonshot AI. You are proficient in Chinese and English conversations. You provide users with safe, helpful, and accurate responses. You will reject any requests involving terrorism, racism, or explicit content. Moonshot AI is a proper noun and should not be translated into other languages."},
        {"role": "user", "content": "Hello, my name is Li Lei. What is 1+1?"}
    ],
    temperature = 0.3,
    extra_headers={
        "X-Msh-Context-Cache": "cache-xxx-xxx-xxx",
        "X-Msh-Context-Cache-Reset-TTL": "3600",
    },
)
 
print(completion.choices[0].message.content)

🍬 Using Cache with Message Content
You can use the following message format to utilize the cache:

{
    "role": "cache",
    "content": "cache_id=xxx;other-options"
}

This is a special message with role=cache that specifies which cache to use via the content field:

This message must be placed at the beginning of the messages list.
The tools parameter must be null (an empty array will also be considered as having a value).
We will replace this special message with the cached message list and fill the tools from the cache into the tools parameter.
You do not need to include cached messages in the messages list; only add messages that have not been cached.
You can think of this special message as a reference to the cached message list.
The rules for content are as follows:
Use the format cache_id={id} to specify the cache id.
Use the format reset_ttl={ttl} to specify whether to reset the expired_at expiration time.
Use the format dry_run=1 to specify whether to only validate the cache without enabling the inference process.
These parameters are concatenated with semicolons ;, and do not add an extra semicolon after the last parameter.
Among them:
cache_id corresponds to the X-Msh-Context-Cache in the Headers.
dry_run corresponds to the X-Msh-Context-Cache-DryRun in the Headers.
reset_ttl corresponds to the X-Msh-Context-Cache-Reset-TTL in the Headers.
The parameter values and rules are consistent with the Headers.
Context Caching Tag System
We have added a tag system to Context Caching to manage and use Context Cache through tags.

Use Case
When using Context Caching, if you need to modify the cached content (for example, adding new knowledge to the context or updating time-sensitive data), we recommend deleting the original Cache and creating a new Cache with the updated content. This process will change the Context Cache ID, and in actual development, developers may need to write additional code to manage the Cache, such as matching custom keys with cache_id. This undoubtedly increases the cognitive load for developers when using Cache.

Therefore, we designed the Context Caching Tag system (Tag) to reduce the cognitive load when using Context Caching. You can tag a Context Cache with as many labels as you want and use the corresponding Cache by specifying the tag name in the Message. One advantage of using tags is that they are entirely up to the developer and do not change with the Cache (unlike cache_id, which changes with the cache).

Create a Tag
POST https://api.moonshot.ai/v1/caching/refs/tags

Parameter Name	Parameter Type (e.g., Python Type Hint)	Required	Description
tag	str	Yes	Tag name, minimum length is 1, maximum length is 128, the first character must be a letter (uppercase or lowercase), and non-first characters can be letters, underscores, hyphens, or periods.
cache_id	str	Yes	The id of the already created cache
You can also think of the Create Tag API as binding a Tag to an existing Cache.

Here is an example of a successful request response:

{
    "cache_id": "cache-et3tmxxkzr7i11dp6x51",
    "created_at": 1719976735,
    "object": "cache_object.tag",
    "owned_by": "cn0psxxcp7fclnphkcpg",
    "tag": "my-tag"
}

List Tags
GET https://api.moonshot.ai/v1/caching/refs/tags?limit=20&order=desc&after=tag-id-xxxxxx

Note: Request parameters are provided in the form of URL query parameters.

Parameter Name	Parameter Type (using Python Type Hint as an example)	Required	Description
limit	int	No	Specifies the number of tags to return per page for the current request. The default value is 20.
order	Literal["asc", "desc"]	No	Specifies the sorting rule for querying tags in the current request. Tags are sorted by their created_at timestamp. The default value is desc.
after	str	No	Specifies the tag from which to start the search in the current request. Its value is the tag's tag. Note: The query result does not include the tag specified by after.
before	str	No	Specifies the tag up to which to query in the current request. Its value is the tag's tag. Note: The query result does not include the tag specified by before.
Here is an example of the content returned by a successful request:

{
  "object": "list",
  "data": [
    {
      "tag": "tom",
      "cache_id": "cache-et3w5r7e13sqw5wtzsei",
      "object": "cache_object.tag",
      "owned_by": "root",
      "created_at": 1719910897
    }
  ]
}

Delete Tag
DELETE https://api.moonshot.ai/v1/caching/refs/tags/{{your_tag_name}}

Note: The Delete Tag interface will return success regardless of whether the tag exists or not.

Here is an example of the content returned by a successful request:

{
  "deleted": true,
  "object": "cache_object.tag.deleted",
  "tag": "tom"
}

Get Tag Information
GET https://api.moonshot.ai/v1/caching/refs/tags/{{your_tag_name}}

Here is an example of the content returned by a successful request:

{
    "cache_id": "cache-et3tmxxkzr7i11dp6x51",
    "created_at": 1719976735,
    "object": "cache_object.tag",
    "owned_by": "cn0psxxcp7fclnphkcpg",
    "tag": "my-tag"
}

Get Context Cache Information for a Tag
GET https://api.moonshot.ai/v1/caching/refs/tags/{{your_tag_name}}/content

Note: This is a quick way to view the cache information marked by a tag. It is equivalent to first calling /v1/caching/refs/tags/{{your_tag_name}} and then calling /v1/caching/{{tag.cache_id}}. The return value is identical to that of /v1/caching/{{cache-id}}.

Use Tags
Currently, we only support using tags through Message Content. The specific usage is to replace the original cache_id={id} with tag={tag}, as shown below:

{
    "role": "cache",
    "content": "tag=xxx;other-options"
}

Other Interfaces
Calculate Tokens
Request Address
POST https://api.moonshot.ai/v1/tokenizers/estimate-token-count

Request Content
The input structure for estimate-token-count is almost identical to that of chat completion.

Example
{
    "model": "moonshot-v1-8k",
    "messages": [
        {
            "role": "system",
            "content": "You are Kimi, an AI assistant provided by Moonshot AI. You excel in Chinese and English conversations. You provide users with safe, helpful, and accurate answers. You refuse to answer any questions involving terrorism, racism, pornography, or violence. Moonshot AI is a proper noun and should not be translated into other languages."
        },
        { "role": "user", "content": "Hello, my name is Li Lei. What is 1+1?" }
    ]
}

Parameters
Field	Description	Type	Values
messages	A list of messages in the conversation so far.	List[Dict]	This is a list of structures, with each element similar to: json{"role": "user", "content": "Hello"} The role can only be one of system, user, assistant, and the content must not be empty
model	Model ID, which can be obtained through List Models	string	Currently one of moonshot-v1-8k, moonshot-v1-32k, moonshot-v1-128k
Example Request
curl 'https://api.moonshot.ai/v1/tokenizers/estimate-token-count' \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $MOONSHOT_API_KEY" \
  -d '{
    "model": "moonshot-v1-8k",
    "messages": [
        {
            "role": "system",
            "content": "You are Kimi, an AI assistant provided by Moonshot AI. You excel in Chinese and English conversations. You provide users with safe, helpful, and accurate answers. You refuse to answer any questions involving terrorism, racism, pornography, or violence. Moonshot AI is a proper noun and should not be translated into other languages."
        },
        {
            "role": "user",
            "content": "Hello, my name is Li Lei. What is 1+1?"
        }
    ]
}'

Response
{
    "data": {
        "total_tokens": 80
    }
}

If there is no error field, you can take data.total_tokens as the calculation result.

Check Balance
Request Address
GET https://api.moonshot.ai/v1/users/me/balance

Example request
curl https://api.moonshot.ai/v1/users/me/balance -H "Authorization: Bearer $MOONSHOT_API_KEY"

Response
{
  "code": 0,
  "data": {
    "available_balance": 49.58894,
    "voucher_balance": 46.58893,
    "cash_balance": 3.00001
  },
  "scode": "0x0",
  "status": true
}

Response Content Description
Field	Description	Type	Unit
available_balance	The available balance, including cash balance and voucher balance. When it is less than or equal to 0, the user cannot call the inference API	float	USD
voucher_balance	The voucher balance, which cannot be negative	float	USD
cash_balance	The cash balance, which can be negative, indicating that the user owes money. When it is negative, available_balance is equal to the value of voucher_balance	float	USD

Model Inference Pricing Explanation
Concepts
Billing Unit
Token: A token represents a common sequence of characters. The number of tokens used for each Chinese character may vary. For example, a single character like "夔" might be broken down into several tokens, while a short and common phrase like "中国" might use just one token.

Generally speaking, for a typical Chinese text, 1 token is roughly equivalent to 1.5-2 Chinese characters. The exact number of tokens generated by each call can be obtained through the Token Calculation API.

Billing Logic
Chat Completion API charges: We bill both the Input and Output based on usage. If you upload and extract content from a document and then pass the extracted content as Input to the model, the document content will also be billed based on usage.

File-related interfaces (file content extraction/file storage) are temporarily free. In other words, if you only upload and extract a document, this API itself will not incur any charges.

Product Pricing
Explanation：The prices listed below are all inclusive of tax.

Generation Model kimi-latest
Model
Model
Unit
Input Price
（Cache Hit）
Input Price
（Cache Miss）
Output Price
Context Window
kimi-latest	kimi-latest-8k	1M tokens	$0.15	$0.20	$2.00	8,192 tokens
kimi-latest-32k	1M tokens	$0.15	$1.00	$3.00	32,768 tokens
kimi-latest-128k	1M tokens	$0.15	$2.00	$5.00	131,072 tokens
The kimi-latest model always uses the latest version of the Kimi large model used by the Kimi AI Assistant product, which may include features that are not yet stable
The kimi-latest model has a context length of 128k and will automatically select 8k/32k/128k models for billing based on the requested context length
kimi-latest is a vision model that supports image understanding
It supports automatic context caching, with cached tokens costing only $1 per M tokens (manual context caching is not currently supported)
All other features are consistent with the moonshot-v1 series models, including: ToolCalls, JSON Mode, Partial Mode, and internet search functionality
Generation Model kimi-k2
Model
Unit
Input Price
（Cache Hit）
Input Price
（Cache Miss）
Output Price
Context Window
kimi-k2-0711-preview	1M tokens	$0.15	$0.60	$2.50	131,072 tokens
kimi-k2 is a Mixture-of-Experts (MoE) foundation model with exceptional coding and agent capabilities, featuring 1 trillion total parameters and 32 billion activated parameters. In benchmark evaluations covering general knowledge reasoning, programming, mathematics, and agent-related tasks, the K2 model outperforms other leading open-source models
The kimi-k2 model has a context length of 128k
Does not support vision functionality
Supports ToolCalls, JSON Mode, Partial Mode, and internet search functionality
Generation Model Moonshot-v1
Model	Unit	Input Price	Output Price	Context Window
moonshot-v1-8k	1M tokens	$0.20	$2.00	8,192 tokens
moonshot-v1-32k	1M tokens	$1.00	$3.00	32,768 tokens
moonshot-v1-128k	1M tokens	$2.00	$5.00	131,072 tokens
moonshot-v1-8k-vision-preview	1M tokens	$0.20	$2.00	8,192 tokens
moonshot-v1-32k-vision-preview	1M tokens	$1.00	$3.00	32,768 tokens
moonshot-v1-128k-vision-preview	1M tokens	$2.00	$5.00	131,072 tokens
Here, 1M = 1,000,000. The prices in the table represent the cost per 1M tokens consumed.

Long-Term Thinking Model Kimi-thinking-preview
Model	Unit	Input Price	Output Price	Context Window
kimi-thinking-preview	1M tokens	$30.00	$30.00	131,072 tokens
The kimi-thinking-preview model is a multimodal reasoning model with both multimodal and general reasoning capabilities provided by Moonshot AI. It is great at diving deep into problems to help tackle more complex challenges
The context length of the kimi-thinking-preview model is 128k
The kimi-thinking-preview model is a visual model that supports image understanding
It does not support ToolCalls, web search functionality, Context Caching, or Partial Mode temporarily
It does not support JSON Mode temporarily

Context Caching Pricing and Billing
Product Pricing
Prices are in the public beta phase and may change at any time.

Explanation：The prices listed below are all inclusive of tax.

Model
Standard Input Price / 1M tokens
Cache Tokens Call Price
Cache Creation / 1M tokens
Cache Storage / 1M tokens/min
Cache Call / time
moonshot-v1-8k	$0.20	
Limited Free
$1.00	$0.15	$0.0015
moonshot-v1-32k	$1.00	
Limited Free
$1.00	$0.15	$0.0015
moonshot-v1-128k	$2.00	
Limited Free
$1.00	$0.15	$0.0015
Billing Logic
Cache Resource Fee = Cache Creation Fee + Cache Storage Fee

Chat Call Fee Using Cache = Cache Call Fee + Chat Input Tokens Fee for Non-Cache Matches + Output Tokens Fee

Billing Item Explanation
Cache Creation Fee
When you call the Cache creation interface and successfully create a Cache, the fee is charged based on the actual number of Tokens in the Cache.
Billing Unit: Tokens, charged based on the actual number of Tokens
Cache Storage Fee
During the lifetime of the Cache, the storage fee is charged per minute.
Billing Unit: Minutes, with any partial minute rounded up to one minute
Cache Call Fee
The Cache call fee consists of two parts:
Cache Call Tokens Fee
Cache Per-Call Fee: Within the Cache's lifetime, if a user requests a successfully created Cache through the chat interface and the chat message content matches the Cache, a per-call fee will be charged. Each Cache call costs 0.02 USD.
Billing Unit: Calls
Cost Calculator
You can use the cost calculator to enter specific details about your business. The platform will help you calculate the estimated savings after switching to Context Caching, providing you with data to decide whether to switch to Context Caching.

Select Model

moonshot-v1-128k

moonshot-v1-32k

moonshot-v1-8k
Input Tokens per request
117,965
Output Tokens per request
13,107
Same Input Tokens ratio
99.5%
Repeat request duration
1minutes
Repeat request times
10
times
Note: Current business model may trigger rate limits. Calculator data is for reference only, actual data may vary.

Total tokens 1.31 M, Original cost 3.01 $
Total cost with Context Caching 0.81 $
👏 Estimated savings 2.19 $，Reduction 72.89 %
We recommend using Context Caching to enjoy low-cost long-text services!
Calculator Example Scenario: Upload and parse the novel "Zhenhuan Zhuan" and ask about the personalities of the six main characters: Zhenhuan, Huafei, Anlingrong, Shenmeizhuang, the Empress, and the Emperor.

Calculator Option	How to Fill in the "Zhenhuan Zhuan" Q&A Scenario
Model to Call	The first volume of the novel "Zhenhuan Zhuan" is 200,000 words, requiring a 128k model for questioning. Choose moonshot-v1-128k
Input Tokens per Question	The novel "Zhenhuan Zhuan" is 200,000 words, estimated to need 130,000 tokens. Each question uses 20 tokens. Calculator entry: 130,000 + 20 = 130,020 tokens
Output Tokens per Question	Each question is expected to return a character summary of 100 words, consuming 80 tokens. Calculator entry: 80
Proportion of Identical Input Tokens	The proportion of tokens from the novel "Zhenhuan Zhuan" in each question's token consumption. Calculator entry: 130,000 / 130,020 = 99%
Time Period for Repeated Questions	The time period for asking the six questions about the characters' personalities in "Zhenhuan Zhuan". Calculator entry: 10min
Number of Repeated Questions	The number of repeated questions for "Zhenhuan Zhuan". Calculator entry: 6
Limitations
The Context Caching feature is available to all users.
The maximum Cache storage limit per user is 1M.
The Cache can be set to expire in 1 hour.
Tier Rate Limiting: The Tokens for creating Caches and the Tokens for Chat calls to Caches are included in the total number of Tokens for your API interactions and are subject to TPM/TPD limits.
The vision preview model does not support creating context caches.
Important Notes
Creating a Cache is an asynchronous operation. Typically, it takes 40s-60s after initiating the request before the Cache can be used.
You can use the list cache method to confirm whether the Cache has been successfully created.

Tool Pricing
Product Pricing
Explanation：The prices listed below are all inclusive of tax.

Tool Name
Unit
Price
Comment
Web Search	1 time	$0.005	Trigger $web_search tool call, charge once
Internet Search Billing Logic
When you add the $web_search tool in tools and receive a response with finish_reason = tool_calls and tool_call.function.name = $web_search, we charge a fee of $0.005 for the $web_search call. If the response has finish_reason = stop, no call fee will be charged.

Additionally, when using $web_search, we still charge for the Tokens generated by the /chat/completions interface based on the model size. It is important to note that when the $web_search tool is triggered, the search results are also counted in the Tokens. The number of Tokens occupied by the search results can be obtained from the returned tool_call.function.arguments. For example, if the content of the $web_search occupies 4k Tokens, these 4k Tokens will be included in the total Tokens when the caller makes the next call to the /chat/completions interface. The total billing Tokens will be:

total_tokens = prompt_tokens + search_tokens + completions_tokens

Note: If you stop after triggering the $web_search without continuing with tool_calls, we will only charge the tool call fee of $0.005, and the Tokens occupied by the search content will not be billed.

Recharge and Rate Limits
To ensure fair distribution of resources and prevent malicious attacks, we currently apply rate limits based on the cumulative recharge amount of each account. The specific limits are shown in the table below. If you have higher requirements, please contact us via email at api-service@moonshot.ai.

User Level	Cumulative Recharge Amount	Concurrency	RPM	TPM	TPD
Free	$0	1	3	32,000	1,500,000
Tier1	$10	50	200	128,000	10,000,000
Tier2	$20	100	500	128,000	20,000,000
Tier3	$100	200	5,000	384,000	Unlimited
Tier4	$1,000	400	5,000	768,000	Unlimited
Tier5	$3,000	1,000	10,000	2,000,000	Unlimited
Explanation of Rate Limits Concepts
Concurrency: The maximum number of requests from you that we can process at the same time.

RPM: Requests per minute, which means the maximum number of requests you can send to us in one minute.

TPM: Tokens per minute, which means the maximum number of tokens you can interact with us in one minute.

TPD: Tokens per day, which means the maximum number of tokens you can interact with us in one day.

For more details, please refer to the Rate Limits section.

Why Do We Implement Rate Limits?
Rate limits are a common practice for API interfaces, and there are several reasons for it:

They help prevent abuse or misuse of the API. For example, malicious actors might try to overwhelm the API with a large number of requests, attempting to overload it or cause service disruptions. By setting rate limits, we can guard against such behavior.

Rate limits ensure fair access to the API for everyone. If one person or organization sends too many requests, it could slow down the API for everyone else. By limiting the number of requests a single user can send, we ensure that as many people as possible can use the API without experiencing slowdowns.

Rate limits help us manage the overall load on our cluster. A sudden surge in requests to the API could put pressure on the servers and lead to performance issues. By setting rate limits, we can maintain a smooth and consistent experience for all users.

Special Notes
We will do our best to ensure normal usage for users, but when the cluster load reaches its capacity limit, we may take temporary measures to adjust the rate limits.
Vouchers do not count towards the cumulative recharge total.

Migrating from OpenAI to Kimi API
About API Compatibility
The Kimi API is compatible with OpenAI's interface specifications. You can use the Python or NodeJS SDKs provided by OpenAI to call and use the Kimi large language model. This means that if your application or service is developed based on OpenAI's models, you can seamlessly migrate to using the Kimi large language model by simply replacing the base_url and api_key with the configuration for the Kimi large language model. Here is an example of how to do this:

from openai import OpenAI
 
client = OpenAI(
    api_key="MOONSHOT_API_KEY", # <-- Replace MOONSHOT_API_KEY with the API Key you obtained from the Kimi Open Platform
    base_url="https://api.moonshot.ai/v1", # <-- Replace the base_url from https://api.openai.com/v1 to https://api.moonshot.ai/v1
)

We will do our best to ensure compatibility between the Kimi API and OpenAI. However, in some special cases, there may still be some differences and variations between the Kimi API and OpenAI (but this does not affect overall compatibility). We will detail the differences between the Kimi API and OpenAI and propose feasible migration solutions to help developers complete the migration process smoothly.

Here is a list of interfaces that are compatible with OpenAI:

/v1/chat/completions
/v1/files
/v1/files/{file_id}
/v1/files/{file_id}/content
Temperature and N Value
When using OpenAI's interface, you can set both temperature=0 and n>1, which means that in cases where the temperature value is 0, multiple different answers (i.e., choices) can be returned simultaneously.

However, in the Kimi API, when you set the temperature value to 0 or close to 0 (e.g., 0.001), we can only provide one answer (i.e., len(choices)=1). If you set temperature to 0 while using an n value greater than 1, we will return an "invalid request" error, specifically invalid_request_error.

Additionally, please note that the range of values for the temperature parameter in the Kimi API is [0, 1], while the range for the temperature parameter in OpenAI is [0, 2].

Migration Recommendation: We recommend a temperature value of 0.3. If your business scenario requires setting temperature=0 to get more stable results from the Kimi large language model, please pay special attention to setting the n value to 1, or do not set the n value at all (in which case the default n=1 will be used as the request parameter, which is valid).

Usage Value in Stream Mode
When using OpenAI's chat.completions interface, in cases of streaming output (i.e., stream=True), the output result does not include usage information by default (including prompt_tokens/completion_tokens/total_tokens). OpenAI provides an additional parameter stream_options={"include_usage": True} to include usage information in the last data block of the response.

In the Kimi API, in addition to the stream_options={"include_usage": True} parameter, we also place usage information (including prompt_tokens/completion_tokens/total_tokens, and if you use the Context Caching feature, it will also include cached_tokens) in the end data block of each choice.

Migration Recommendation: In most cases, developers do not need to take any additional compatibility measures. If your business scenario requires tracking the usage information for each choice individually, you can access the choice.usage field. Note that among different choices, only the values of usage.completion_tokens and usage.total_tokens are different, while the values of usage.prompt_tokens and usage.cached_tokens are the same for all choices.

Deprecated function_call
In 2023, OpenAI introduced the functions parameter to enable function call functionality. After functional iteration, OpenAI later launched the tool call feature and marked the functions parameter as deprecated, which means that the functions parameter may be removed at any time in future API iterations.

The Kimi API fully supports the tool call feature. However, since functions has been deprecated, the Kimi API does not support using the functions parameter to execute function calls.

Migration Recommendation: If your application or service relies on tool calls, no additional compatibility measures are needed. If your application or service depends on the deprecated function call, we recommend migrating to tool calls. Tool calls expand the capabilities of function calls and support parallel function calls. For specific examples of tool calls, please refer to our tool call guide:

Using Kimi API for Tool Calls (tool_calls)

Here is an example of migrating from functions to tools:

We will present the code that needs to be modified in the form of comments, along with explanations, to help developers better understand how to perform the migration.

from typing import *
 
import json
import httpx
from openai import OpenAI
 
client = OpenAI(
    api_key="MOONSHOT_API_KEY",  # Replace MOONSHOT_API_KEY with the API Key you obtained from the Kimi Open Platform
    base_url="https://api.moonshot.ai/v1",
)
 
functions = [
    {
        "name": "search",  # The name of the function, please use English letters (uppercase and lowercase), numbers, plus hyphens and underscores as the function name
        "description": """ 
            Search for content on the internet using a search engine.
 
            Call this tool when your knowledge cannot answer the user's question or when the user requests you to perform an online search. Extract the content the user wants to search for from the conversation with the user and use it as the value of the query parameter.
            The search results include the title of the website, the website's address (URL), and a brief introduction to the website.
        """,  # Description of the function, write the specific function here and the usage scenario so that the Kimi large language model can correctly choose which functions to use
        "parameters": {  # Use the parameters field to define the parameters accepted by the function
            "type": "object",  # Always use type: object to make the Kimi large language model generate a JSON Object parameter
            "required": ["query"],  # Use the required field to tell the Kimi large language model which parameters are required
            "properties": {  # Properties contain the specific parameter definitions, and you can define multiple parameters
                "query": {  # Here, the key is the parameter name, and the value is the specific definition of the parameter
                    "type": "string",  # Use type to define the parameter type
                    "description": """
                        The content the user wants to search for, extract it from the user's question or chat context.
                    """  # Use description to describe the parameter so that the Kimi large language model can better generate the parameter
                }
            }
        }
    }
]
 
 
def search_impl(query: str) -> List[Dict[str, Any]]:
    """
    search_impl uses a search engine to search for query. Most mainstream search engines (such as Bing) provide API calls, and you can choose the one you like.
    You can call the search engine API of your choice and place the website title, website link, and website introduction information in a dict and return it.
 
    This is just a simple example, and you may need to write some authentication, validation, and parsing code.
    """
    r = httpx.get("https://your.search.api", params={"query": query})
    return r.json()
 
 
def search(arguments: Dict[str, Any]) -> Any:
    query = arguments["query"]
    result = search_impl(query)
    return {"result": result}
 
 
function_map = {
    "search": search,
}
 
# ==========================================================================================================================================================
# Tools are a superset of functions, so we can construct tools using the already defined functions. We loop through each function and create the corresponding tool format;
# At the same time, we also generate the corresponding tool_map.
# ==========================================================================================================================================================
 
tools = []
tool_map = {}
for function in functions:
    tool = {
        "type": "function",
        "function": function,
    }
    tools.append(tool)
    tool_map[function["name"]] = function_map[function["name"]]
 
messages = [
    {"role": "system",
     "content": "You are Kimi, an artificial intelligence assistant provided by Moonshot AI. You are more proficient in Chinese and English conversations. You provide users with safe, helpful, and accurate answers. You also refuse to answer any questions involving terrorism, racism, pornography, or violence. Moonshot AI is a proper noun and should not be translated into other languages."},
    {"role": "user", "content": "Please search the internet for Context Caching and tell me what it is."}  # The user asks Kimi to search online
]
 
finish_reason = None
 
# ==========================================================================================================================================================
# Here, we change the finish_reason value check from function_call to tool_calls
# ==========================================================================================================================================================
# while finish_reason is None or finish_reason == "function_call":
while finish_reason is None or finish_reason == "tool_calls":
    completion = client.chat.completions.create(
        model="moonshot-v1-8k",
        messages=messages,
        temperature=0.3,
        # ==========================================================================================================================================================
        # We no longer use the functions parameter, but instead use the tools parameter to enable tool calls
        # ==========================================================================================================================================================
        # function=functions,
        tools=tools,  # <-- We submit the defined tools to Kimi via the tools parameter
    )
    choice = completion.choices[0]
    finish_reason = choice.finish_reason
 
    # ==========================================================================================================================================================
    # Here, we replace the original function_call execution logic with the tool_calls execution logic;
    # Note that since there may be multiple tool_calls, we need to execute each one using a for loop.
    # ==========================================================================================================================================================
    # if finish_reason == "function_call":
    #   messages.append(choice.message)
    #   function_call_name = choice.message.function_call.name
    #   function_call_arguments = json.loads(choice.message.function_call.arguments)
    #   function_call = function_map[function_call_name]
    #   function_result = function_call(function_call_arguments)
    #   messages.append({
    #       "role": "function",
    #       "name": function_call_name,
    #       "content": json.dumps(function_result)
    #   })
 
    if finish_reason == "tool_calls":  # <-- Check if the response contains tool_calls
        messages.append(choice.message)  # <-- Add the assistant message from Kimi to the context for the next request
        for tool_call in choice.message.tool_calls:  # <-- Loop through each tool_call as there may be multiple
            tool_call_name = tool_call.function.name
            tool_call_arguments = json.loads(tool_call.function.arguments)  # <-- The arguments are serialized JSON, so we need to deserialize them
            tool_function = tool_map[tool_call_name]  # <-- Use tool_map to quickly find the function to execute
            tool_result = tool_function(tool_call_arguments)
 
            # Construct a message with role=tool to show the result of the tool call to the model;
            # Note that we need to provide the tool_call_id and name fields in the message so that Kimi can
            # correctly match it to the corresponding tool_call.
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_call_name,
                "content": json.dumps(tool_result),  # <-- We agree to submit the tool call result as a string, so we serialize it here
            })
 
print(choice.message.content)  # <-- Finally, we return the model's response to the user

About tool_choice
The Kimi API supports the tool_choice parameter, but there are some subtle differences in the values for tool_choice compared to OpenAI. The values for tool_choice that are currently compatible between Kimi API and OpenAI API are:

 "none"
 "auto"
 null
Please note that the current version of Kimi API does not support the tool_choice=required parameter.

Migration suggestion: If your application or service relies on the required value of the tool_choice field in the OpenAI API to ensure that the large model "definitely" selects a certain tool for invocation, we suggest using some special methods to enhance the Kimi large language model's awareness of invoking tools to partially accommodate the original business logic. For example, you can emphasize the use of a certain tool in the prompt to achieve a similar effect. We demonstrate this logic with a simplified version of the code:

from openai import OpenAI
 
client = OpenAI(
    api_key="MOONSHOT_API_KEY",  # Replace MOONSHOT_API_KEY with the API Key you obtained from the Kimi Open Platform
    base_url="https://api.moonshot.ai/v1",
)
 
tools = {
    # Define your tools here
}
 
messages = [
    # Store your message history here
]
 
completion = client.chat.completions.create(
    model="moonshot-v1-128k",
    messages=messages,
    temperature=0.3,
    tools=tools,
    # tool_choice="required",  # <-- Since Kimi API does not currently support tool_choice=required, we have temporarily disabled this option
)
 
choice = completion.choices[0]
if choice.finish_reason != "tool_calls":
    # We assume that our business logic can confirm that tool_calls must be invoked here.
    # Without using tool_choice=required, we use the prompt to make Kimi choose a tool for invocation.
    messages.append(choice.message)
    messages.append({
        "role": "user",
        "content": "Please select a tool to handle the current issue.",  # Usually, the Kimi large language model understands the intention to invoke a tool and selects one for invocation
    })
    completion = client.chat.completions.create(
        model="moonshot-v1-128k",
        messages=messages,
        temperature=0.3,
        tools=tools,
    )
    choice = completion.choices[0]
    assert choice.finish_reason == "tool_calls"  # This request should return finish_reason=tool_calls
    print(choice.message.content)

Please note that this method cannot guarantee a 100% success rate in triggering tool_calls. If your application or service has a very strong dependency on tool_calls, please wait for the launch of the tool_choice=required feature in Kimi API.

MoonPalace - Moonshot AI's Kimi API Debugging Tool
MoonPalace (Moon Palace) is an API debugging tool provided by Moonshot AI. It has the following features:

Cross-platform support:
 Mac
 Windows
 Linux;
Easy to use, just replace base_url with http://localhost:9988 after launching to start debugging;
Captures complete requests, including the "scene of the accident" when network errors occur;
Quickly search and view request information using request_id and chatcmpl_id;
One-click export of BadCase structured reporting data, helping to enhance Kimi's model capabilities;
We recommend using MoonPalace as your API "supplier" during the code writing and debugging phase, so you can quickly identify and locate various issues related to API calls and code writing. For any unexpected outputs from Kimi large language model, you can also export the request details via MoonPalace and submit them to Moonshot AI to improve Kimi large language model.

Installation Methods
Using the go Command to Install
If you have the go toolchain installed, you can run the following command to install MoonPalace:

$ go install github.com/MoonshotAI/moonpalace@latest

The above command will install the compiled binary file in your $GOPATH/bin/ directory. Run the moonpalace command to check if it has been installed successfully:

$ moonpalace
MoonPalace is a command-line tool for debugging the Moonshot AI HTTP API.
 
Usage:
  moonpalace [command]
 
Available Commands:
  cleanup     Cleanup Moonshot AI requests.
  completion  Generate the autocompletion script for the specified shell
  export      export a Moonshot AI request.
  help        Help about any command
  inspect     Inspect the specific content of a Moonshot AI request.
  list        Query Moonshot AI requests based on conditions.
  start       Start the MoonPalace proxy server.
 
Flags:
  -h, --help      help for moonpalace
  -v, --version   version for moonpalace
 
Use "moonpalace [command] --help" for more information about a command.

If you still cannot find the moonpalace binary file, try adding the $GOPATH/bin/ directory to your $PATH environment variable.

Downloading from the Releases Page
You can download the precompiled binary (executable) files from the Releases page:

moonpalace-linux
moonpalace-macos-amd64 => for Intel-based Macs
moonpalace-macos-arm64 => for Apple Silicon-based Macs
moonpalace-windows.exe
Download the binary (executable) file that matches your platform and place it in a directory that is included in your $PATH environment variable. Rename it to moonpalace and then grant it executable permissions.

Usage
Starting the Service
Use the following command to start the MoonPalace proxy server:

$ moonpalace start --port <PORT>

MoonPalace will start an HTTP server locally, with the --port parameter specifying the local port that MoonPalace will listen on. The default value is 9988. When MoonPalace starts successfully, it will output:

[MoonPalace] 2024/07/29 17:00:29 MoonPalace Starts {'=>'} change base_url to "http://127.0.0.1:9988/v1"

As instructed, replace base_url with the displayed address. If you are using the default port, set base_url=http://127.0.0.1:9988/v1. If you are using a custom port, replace base_url with the displayed address.

Additionally, if you want to always use a debugging api_key during debugging, you can use the --key parameter when starting MoonPalace to set a default api_key for MoonPalace. This way, you don't have to manually set the api_key in each request. MoonPalace will automatically add the api_key you set with --key when requesting the Kimi API.

If you have correctly set base_url and successfully called the Kimi API, MoonPalace will output the following information:

$ moonpalace start --port <PORT>
[MoonPalace] 2024/07/29 17:00:29 MoonPalace Starts {'=>'} change base_url to "http://127.0.0.1:9988/v1"
[MoonPalace] 2024/07/29 21:30:53 POST   /v1/chat/completions 200 OK
[MoonPalace] 2024/07/29 21:30:53   - Request Headers: 
[MoonPalace] 2024/07/29 21:30:53     - Content-Type:   application/json
[MoonPalace] 2024/07/29 21:30:53   - Response Headers: 
[MoonPalace] 2024/07/29 21:30:53     - Content-Type:   application/json
[MoonPalace] 2024/07/29 21:30:53     - Msh-Request-Id: c34f3421-4dae-11ef-b237-9620e33511ee
[MoonPalace] 2024/07/29 21:30:53     - Server-Timing:  7134
[MoonPalace] 2024/07/29 21:30:53     - Msh-Uid:        cn0psmmcp7fclnphkcpg
[MoonPalace] 2024/07/29 21:30:53     - Msh-Gid:        enterprise-tier-5
[MoonPalace] 2024/07/29 21:30:53   - Response: 
[MoonPalace] 2024/07/29 21:30:53     - id:                cmpl-12be8428ebe74a9e8466a37bee7a9b11
[MoonPalace] 2024/07/29 21:30:53     - prompt_tokens:     1449
[MoonPalace] 2024/07/29 21:30:53     - completion_tokens: 158
[MoonPalace] 2024/07/29 21:30:53     - total_tokens:      1607
[MoonPalace] 2024/07/29 21:30:53   New Row Inserted: last_insert_id=15

MoonPalace will output the details of the request in the form of logs in the command line (if you want to persist the log content, you can redirect stderr to a file).

Note: In the logs, the value of the Msh-Request-Id field in the Response Headers corresponds to the --requestid parameter in the Search Request and Export Request sections below. The id in the Response corresponds to the --chatcmpl parameter, and last_insert_id corresponds to the --id parameter.

[MoonPalace] 2024/08/05 19:06:19   it seems that your max_tokens value is too small, please set a larger value

If the current mode is non-streaming output (stream=False), MoonPalace will suggest an appropriate max_tokens value.

Enabling Repeated Content Output Detection
MoonPalace offers a feature to detect repeated content output from the Kimi large language model. Repeated content output refers to the model continuously outputting a specific word, sentence, or blank character without stopping before reaching the max_tokens limit. This can lead to additional Token costs when using more expensive models like moonshot-v1-128k. Therefore, MoonPalace provides the --detect-repeat option to enable repeated content output detection, as shown below:

$ moonpalace start --port <PORT> --detect-repeat --repeat-threshold 0.3 --repeat-min-length 20

After enabling the --detect-repeat option, MoonPalace will interrupt the output of the Kimi large language model and log the following message when it detects repeated content:

[MoonPalace] 2024/08/05 18:20:37   it appears that there is an issue with content repeating in the current response

Note: The --detect-repeat option only interrupts the output in streaming mode (stream=True). It does not apply to non-streaming output.

You can adjust MoonPalace's blocking behavior using the --repeat-threshold and --repeat-min-length parameters:

The --repeat-threshold parameter sets MoonPalace's tolerance for repeated content. A higher threshold means lower tolerance, and repeated content will be blocked more quickly. The range is 0 <= threshold <= 1.
The --repeat-min-length parameter sets the minimum number of characters before MoonPalace starts detecting repeated content. For example, --repeat-min-length=100 means that repeated content detection will only start when the output exceeds 100 UTF-8 characters.
Enabling Forced Streaming Output
MoonPalace provides the --force-stream option to force all /v1/chat/completions requests to use streaming output mode:

$ moonpalace start --port <PORT> --force-stream

MoonPalace will set the stream field in the request parameters to True. When receiving a response, it will automatically determine the response format based on whether the caller has set stream:

If the caller has set stream=True, the response will be returned in streaming format without any special handling by MoonPalace.
If the caller has not set stream or has set stream=False, MoonPalace will concatenate all the streaming data chunks into a complete completion structure and return it to the caller after receiving all the data chunks.
For the caller (developer), enabling the --force-stream option will not affect the Kimi API response content you receive. You can still use your original code logic to debug and run your program. In other words, enabling the --force-stream option will not change or break anything. You can safely enable this option.

Why provide this option?

We initially hypothesize that common network connection errors and timeouts (Connection Error/Timeout) occur because, in non-streaming request scenarios (stream=False), intermediate gateways or proxy servers may have set read_header_timeout or read_timeout. This can cause the gateway or proxy server to disconnect while the Kimi API server is still assembling the response (since no response, or even the response header, has been received), resulting in Connection Error/Timeout.

We added the --force-stream parameter to MoonPalace. When starting with moonpalace start --force-stream, MoonPalace converts all non-streaming requests (stream=False or unset) to streaming requests. After receiving all data chunks, it assembles them into a complete completion response structure and returns it to the caller.

For the caller, you can still use the non-streaming API as before. However, after MoonPalace's conversion, it can reduce Connection Error/Timeout issues to some extent because MoonPalace has already established a connection with the Kimi API server and started receiving streaming data chunks.

Retrieving Requests
After MoonPalace is started, all requests routed through MoonPalace are recorded in an sqlite database located at $HOME/.moonpalace/moonpalace.sqlite. You can directly connect to the MoonPalace database to query the specific content of the requests, or you can use the MoonPalace command-line tool to query the requests:

$ moonpalace list
+----+--------+-------------------------------------------+--------------------------------------+---------------+---------------------+
| id | status | chatcmpl                                  | request_id                           | server_timing | requested_at        |
+----+--------+-------------------------------------------+--------------------------------------+---------------+---------------------+
| 15 | 200    | cmpl-12be8428ebe74a9e8466a37bee7a9b11     | c34f3421-4dae-11ef-b237-9620e33511ee | 7134          | 2024-07-29 21:30:53 |
| 14 | 200    | cmpl-1bf43a688a2b48eda80042583ff6fe7f     | c13280e0-4dae-11ef-9c01-debcfc72949d | 3479          | 2024-07-29 21:30:46 |
| 13 | 200    | chatcmpl-2e1aa823e2c94ebdad66450a0e6df088 | c07c118e-4dae-11ef-b423-62db244b9277 | 1033          | 2024-07-29 21:30:43 |
| 12 | 200    | cmpl-e7f984b5f80149c3adae46096a6f15c2     | 50d5686c-4d98-11ef-ba65-3613954e2587 | 774           | 2024-07-29 18:50:06 |
| 11 | 200    | chatcmpl-08f7d482b8434a869b001821cf0ee0d9 | 4c20f0a4-4d98-11ef-999a-928b67d58fa8 | 593           | 2024-07-29 18:49:58 |
| 10 | 200    | chatcmpl-6f3cf14db8e044c6bfd19689f6f66eb4 | 49f30295-4d98-11ef-95d0-7a2774525b85 | 738           | 2024-07-29 18:49:55 |
| 9  | 200    | cmpl-2a70a8c9c40e4bcc9564a5296a520431     | 7bd58976-4d8a-11ef-999a-928b67d58fa8 | 40488         | 2024-07-29 17:11:45 |
| 8  | 200    | chatcmpl-59887f868fc247a9a8da13cfbb15d04f | ceb375ea-4d7d-11ef-bd64-3aeb95b9dfac | 867           | 2024-07-29 15:40:21 |
| 7  | 200    | cmpl-36e5e21b1f544a80bf9ce3f8fc1fce57     | cd7f48d6-4d7d-11ef-999a-928b67d58fa8 | 794           | 2024-07-29 15:40:19 |
| 6  | 200    | cmpl-737d27673327465fb4827e3797abb1b3     | cc6613ac-4d7d-11ef-95d0-7a2774525b85 | 670           | 2024-07-29 15:40:17 |
+----+--------+-------------------------------------------+--------------------------------------+---------------+---------------------+

Use the list command to view the content of the most recent requests. By default, it displays fields that are easy to search, such as id/chatcmpl/request_id, as well as status/server_timing/requested_at for checking the request status. If you want to view a specific request, you can use the inspect command to retrieve it:

# The following three commands will retrieve the same request information
$ moonpalace inspect --id 13
$ moonpalace inspect --chatcmpl chatcmpl-2e1aa823e2c94ebdad66450a0e6df088
$ moonpalace inspect --requestid c07c118e-4dae-11ef-b423-62db244b9277
+--------------------------------------------------------------+
| metadata                                                     |
+--------------------------------------------------------------+
| {                                                            |
|     "chatcmpl": "chatcmpl-2e1aa823e2c94ebdad66450a0e6df088", |
|     "content_type": "application/json",                      |
|     "group_id": "enterprise-tier-5",                         |
|     "moonpalace_id": "13",                                   |
|     "request_id": "c07c118e-4dae-11ef-b423-62db244b9277",    |
|     "requested_at": "2024-07-29 21:30:43",                   |
|     "server_timing": "1033",                                 |
|     "status": "200 OK",                                      |
|     "user_id": "cn0psmmcp7fclnphkcpg"                        |
| }                                                            |
+--------------------------------------------------------------+

By default, the inspect command does not print the body of the request and response. If you want to print the body, you can use the following command:

$ moonpalace inspect --chatcmpl chatcmpl-2e1aa823e2c94ebdad66450a0e6df088 --print request_body,response_body
# Since the body information is too lengthy, the detailed content of the body is not shown here
+--------------------------------------------------+--------------------------------------------------+
| request_body                                     | response_body                                    |
+--------------------------------------------------+--------------------------------------------------+
| ...                                              | ...                                              |
+--------------------------------------------------+--------------------------------------------------+

Exporting Requests
If you find that a request does not meet your expectations, or if you want to report a request to Moonshot AI (whether it's a Good Case or a Bad Case, we welcome both), you can use the export command to export a specific request:

# You only need to choose one of the id/chatcmpl/requestid options to retrieve the corresponding request
$ moonpalace export \
    --id 13 \
    --chatcmpl chatcmpl-2e1aa823e2c94ebdad66450a0e6df088 \
    --requestid c07c118e-4dae-11ef-b423-62db244b9277 \
    --good/--bad \
    --tag "code" --tag "python" \
    --directory $HOME/Downloads/

Here, the usage of id/chatcmpl/requestid is the same as in the inspect command, used to retrieve a specific request. The --good/--bad options are used to mark the request as a Good Case or a Bad Case. The --tag option is used to add relevant tags to the request. For example, in the example above, we assume that the request is related to the Python programming language, so we add two tags: code and python. The --directory option specifies the path to the directory where the exported file will be saved.

The content of the successfully exported file is:

$ cat $HOME/Downloads/chatcmpl-2e1aa823e2c94ebdad66450a0e6df088.json
{
    "metadata":
    {
        "chatcmpl": "chatcmpl-2e1aa823e2c94ebdad66450a0e6df088",
        "content_type": "application/json",
        "group_id": "enterprise-tier-5",
        "moonpalace_id": "13",
        "request_id": "c07c118e-4dae-11ef-b423-62db244b9277",
        "requested_at": "2024-07-29 21:30:43",
        "server_timing": "1033",
        "status": "200 OK",
        "user_id": "cn0psmmcp7fclnphkcpg"
    },
    "request":
    {
        "url": "https://api.moonshot.ai/v1/chat/completions",
        "header": "Accept: application/json\r\nAccept-Encoding: gzip\r\nConnection: keep-alive\r\nContent-Length: 2450\r\nContent-Type: application/json\r\nUser-Agent: OpenAI/Python 1.36.1\r\nX-Stainless-Arch: arm64\r\nX-Stainless-Async: false\r\nX-Stainless-Lang: python\r\nX-Stainless-Os: MacOS\r\nX-Stainless-Package-Version: 1.36.1\r\nX-Stainless-Runtime: CPython\r\nX-Stainless-Runtime-Version: 3.11.6\r\n",
        "body":
        {}
    },
    "response":
    {
        "status": "200 OK",
        "header": "Content-Encoding: gzip\r\nContent-Type: application/json; charset=utf-8\r\nDate: Mon, 29 Jul 2024 13:30:43 GMT\r\nMsh-Cache: updated\r\nMsh-Gid: enterprise-tier-5\r\nMsh-Request-Id: c07c118e-4dae-11ef-b423-62db244b9277\r\nMsh-Trace-Mode: on\r\nMsh-Uid: cn0psmmcp7fclnphkcpg\r\nServer: nginx\r\nServer-Timing: inner; dur=1033\r\nStrict-Transport-Security: max-age=15724800; includeSubDomains\r\nVary: Accept-Encoding\r\nVary: Origin\r\n",
        "body":
        {}
    },
    "category": "goodcase",
    "tags":
    [
        "code",
        "python"
    ]
}

We recommend that developers use Github Issues to submit Good Cases or Bad Cases, but if you do not want to make your request information public, you can also submit the Case to us via enterprise WeChat, email, or other means.

You can send the exported file to the following email address:

api-feedback@moonshot.cn


Quickstart with the Kimi API
The Kimi API allows you to interact with the Kimi large language model. Here is a simple example code:

from openai import OpenAI
 
client = OpenAI(
    api_key="MOONSHOT_API_KEY", # Replace MOONSHOT_API_KEY with the API Key you obtained from the Kimi Open Platform
    base_url="https://api.moonshot.ai/v1",
)
 
completion = client.chat.completions.create(
    model = "moonshot-v1-8k",
    messages = [
        {"role": "system", "content": "You are Kimi, an AI assistant provided by Moonshot AI. You are proficient in Chinese and English conversations. You provide users with safe, helpful, and accurate answers. You will reject any requests involving terrorism, racism, or explicit content. Moonshot AI is a proper noun and should not be translated."},
        {"role": "user", "content": "Hello, my name is Li Lei. What is 1+1?"}
    ],
    temperature = 0.3,
)
 
# We receive a response from the Kimi large language model via the API (role=assistant)
print(completion.choices[0].message.content)

To successfully run the above code, you may need to prepare the following:

A Python environment or a Node.js environment. We recommend using a Python interpreter version 3.8 or higher;
The OpenAI SDK. Our API is fully compatible with the OpenAI API format, so you can directly use the Python or Node.js OpenAI SDK for calls. You can install the OpenAI SDK as follows:
pip install --upgrade 'openai>=1.0' #Python
npm install openai@latest #Node.js

An API Key. You need to apply for an API Key from the Kimi Open Platform and pass it to the OpenAi Client so that we can correctly identify your identity;
If you successfully run the above code without any errors, you will see output similar to the following:

Hello, Li Lei! 1+1 equals 2. This is a basic math addition problem. If you have any other questions or need help, feel free to let me know.

Note: Due to the uncertainty of the Kimi large language model, the actual response may not be exactly the same as the above content.

Use the Kimi API for Multi-turn Chat
The Kimi API is different from the Kimi intelligent assistant. The API itself doesn't have a memory function; it's stateless. This means that when you make multiple requests to the API, the Kimi large language model doesn't remember what you asked in the previous request. For example, if you tell the Kimi large language model that you are 27 years old in one request, it won't remember that you are 27 years old in the next request.

So, we need to manually keep track of the context for each request. In other words, we have to manually add the content of the previous request to the next one so that the Kimi large language model can see what we have talked about before. We will modify the example used in the previous chapter to show how to maintain a list of messages to give the Kimi large language model a memory and enable multi-turn conversation functionality.

Note: We have added the key points for implementing multi-turn conversations as comments in the code.

from openai import OpenAI
 
client = OpenAI(
    api_key = "MOONSHOT_API_KEY", # Replace MOONSHOT_API_KEY with the API Key you obtained from the Kimi Open Platform
    base_url = "https://api.moonshot.ai/v1",
)
 
# We define a global variable messages to keep track of the historical conversation messages between us and the Kimi large language model
# The messages include both the questions we ask the Kimi large language model (role=user) and the replies it gives us (role=assistant)
# Of course, it also includes the initial System Prompt (role=system)
# The messages in the list are arranged in chronological order
messages = [
	{"role": "system", "content": "You are Kimi, an artificial intelligence assistant provided by Moonshot AI. You are better at conversing in Chinese and English. You provide users with safe, helpful, and accurate answers. At the same time, you refuse to answer any questions involving terrorism, racism, pornography, or violence. Moonshot AI is a proper noun and should not be translated into other languages."},
]
 
def chat(input: str) -> str:
	"""
	The chat function supports multi-turn conversations. Each time the chat function is called to converse with the Kimi large language model, the model will 'see' the historical conversation messages that have already been generated. In other words, the Kimi large language model has a memory.
	"""
 
  global messages
 
	# We construct the user's latest question as a message (role=user) and add it to the end of the messages list
	messages.append({
		"role": "user",
		"content": input,	
	})
 
	# We converse with the Kimi large language model, carrying the messages along
	completion = client.chat.completions.create(
        model="moonshot-v1-8k",
        messages=messages,
        temperature=0.3,
    )
 
	# Through the API, we receive the reply message (role=assistant) from the Kimi large language model
    assistant_message = completion.choices[0].message
 
    # To give the Kimi large language model a complete memory, we must also add the message it returns to us to the messages list
    messages.append(assistant_message)
 
    return assistant_message.content
 
print(chat("Hello, I am 27 years old this year."))
print(chat("Do you know how old I am this year?")) # Here, based on the previous context, the Kimi large language model will know that you are 27 years old

Let's review the key points in the code above:

The Kimi API itself doesn't have a context memory function. We need to manually inform the Kimi large language model of what we have talked about before through the messages parameter in the API;
In the messages, we need to store both the question messages we ask the Kimi large language model (role=user) and the reply messages it gives us (role=assistant);
It's important to note that in the code above, as the number of chat calls increases, the length of the messages list also keeps growing. This means that the number of Tokens consumed by each request is also increasing. Eventually, at some point, the Tokens occupied by the messages in the messages list will exceed the context window size supported by the Kimi large language model. We recommend that you use some strategy to keep the number of messages in the messages list within a manageable range. For example, you could only keep the latest 20 messages as the context for each request.

We provide an example below to help you understand how to control the context length. Pay attention to how the make_messages function works:

from openai import OpenAI 
 
client = OpenAI(
    api_key = "MOONSHOT_API_KEY", # Replace MOONSHOT_API_KEY with the API Key you obtained from the Kimi Open Platform
    base_url = "https://api.moonshot.ai/v1",
)
 
# We place the System Messages in a separate list because every request should carry the System Messages.
system_messages = [
	{"role": "system", "content": "You are Kimi, an AI assistant provided by Moonshot AI. You are more proficient in conversing in Chinese and English. You provide users with safe, helpful, and accurate responses. You also reject any questions involving terrorism, racism, pornography, or violence. Moonshot AI is a proper noun and should not be translated into other languages."},
]
 
# We define a global variable messages to record the historical conversation messages between us and the Kimi large language model.
# The messages include both the questions we pose to the Kimi large language model (role=user) and the replies from the Kimi large language model (role=assistant).
# The messages are arranged in chronological order.
messages = []
 
 
def make_messages(input: str, n: int = 20) -> list[dict]:
	"""
	The make_messages function controls the number of messages in each request to keep it within a reasonable range, such as the default value of 20. When building the message list, we first add the System Prompt because it is essential no matter how the messages are truncated. Then, we obtain the latest n messages from the historical records as the messages for the request. In most scenarios, this ensures that the number of Tokens occupied by the request messages does not exceed the model's context window.
	"""
	# First, we construct the user's latest question into a message (role=user) and add it to the end of the messages list.
	messages.append({
		"role": "user",
		"content": input,	
	})
 
	# new_messages is the list of messages we will use for the next request. Let's build it now.
	new_messages = []
 
	# Every request must carry the System Messages, so we need to add the system_messages to the message list first.
	# Note that even if the messages are truncated, the System Messages should still be in the messages list.
	new_messages.extend(system_messages)
 
	# Here, when the historical messages exceed n, we only keep the latest n messages.
	if len(messages) > n:
		messages = messages[-n:]
 
	new_messages.extend(messages)
	return new_messages
 
 
def chat(input: str) -> str:
	"""
	The chat function supports multi-turn conversations. Each time the chat function is called to converse with the Kimi large language model, the model can "see" the historical conversation messages that have already been generated. In other words, the Kimi large language model has memory.
	"""
 
	# We converse with the Kimi large language model carrying the messages.
	completion = client.chat.completions.create(
        model="moonshot-v1-8k",
        messages=make_messages(input),
        temperature=0.3,
    )
 
	# Through the API, we obtain the reply message from the Kimi large language model (role=assistant).
    assistant_message = completion.choices[0].message
 
    # To ensure the Kimi large language model has a complete memory, we must add the message returned by the model to the messages list.
    messages.append(assistant_message)
 
    return assistant_message.content
 
print(chat("Hello, I am 27 years old this year."))
print(chat("Do you know how old I am this year?")) # Here, based on the previous context, the Kimi large language model will know that you are 27 years old this year.

Please note that the above code examples only consider the simplest invocation scenarios. In actual business code logic, you may need to consider more scenarios and boundaries, such as:

In concurrent scenarios, additional read-write locks may be needed;
For multi-user scenarios, a separate messages list should be maintained for each user;
You may need to persist the messages list;
You may still need a more precise way to determine how many messages to retain in the messages list;
You may want to summarize the discarded messages and generate a new message to add to the messages list;
……

Use the Kimi Vision Model
The Kimi Vision Model (including moonshot-v1-8k-vision-preview / moonshot-v1-32k-vision-preview / moonshot-v1-128k-vision-preview and so on) can understand the content of images, including text in the image, colors, and the shapes of objects. Here is how you can ask Kimi questions about an image using the following code:

import os
import base64
 
from openai import OpenAI
 
client = OpenAI(
    api_key=os.environ.get("MOONSHOT_API_KEY"),
    base_url="https://api.moonshot.ai/v1",
)
 
# Replace kimi.png with the path to the image you want Kimi to recognize
image_path = "kimi.png"
 
with open(image_path, "rb") as f:
    image_data = f.read()
 
# We use the built-in base64.b64encode function to encode the image into a base64 formatted image_url
image_url = f"data:image/{os.path.splitext(image_path)[1]};base64,{base64.b64encode(image_data).decode('utf-8')}"
 
 
completion = client.chat.completions.create(
    model="moonshot-v1-8k-vision-preview",
    messages=[
        {"role": "system", "content": "You are Kimi."},
        {
            "role": "user",
            # Note here, the content has changed from the original str type to a list. This list contains multiple parts, with the image (image_url) being one part and the text (text) being another part.
            "content": [
                {
                    "type": "image_url", # <-- Use the image_url type to upload the image, the content is the base64 encoded image
                    "image_url": {
                        "url": image_url,
                    },
                },
                {
                    "type": "text",
                    "text": "Describe the content of the image.", # <-- Use the text type to provide text instructions, such as "Describe the content of the image"
                },
            ],
        },
    ],
)
 
print(completion.choices[0].message.content)

Note that when using the Vision model, the type of the message.content field has changed from str to List[Dict] (i.e., a JSON array). Additionally, do not serialize the JSON array and put it into message.content as a str. This will cause Kimi to fail to correctly identify the image type and may trigger the Your request exceeded model token limit error.

✅ Correct Format:

{
    "model": "moonshot-v1-8k-vision-preview",
    "messages":
    [
        {
            "role": "system",
            "content": "You are Kimi, an AI assistant provided by Moonshot AI, who excels in Chinese and English conversations. You provide users with safe, helpful, and accurate answers. You will reject any questions related to terrorism, racism, or explicit content. Moonshot AI is a proper noun and should not be translated into other languages."
        },
        {
            "role": "user",
            "content":
            [
                {
                    "type": "image_url",
                    "image_url":
                    {
                        "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAGAAAABhCAYAAAApxKSdAAAACXBIWXMAACE4AAAhOAFFljFgAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAUUSURBVHgB7Z29bhtHFIWPHQN2J7lKqnhYpYvpIukCbJEAKQJEegLReYFIT0DrCSI9QEDqCSIDaQIEIOukiJwyza5SJWlId3FFz+HuGmuSSw6p+dlZ3g84luhdUeI9M3fmziyXgBCUe/DHYY0Wj/tgWmjV42zFcWe4MIBBPNJ6qqW0uvAbXFvQgKzQK62bQhkaCIPc10q1Zi3XH1o/IG9cwUm0RogrgDY1KmLgHYX9DvyiBvDYI77XmiD+oLlQHw7hIDoCMBOt1U9w0BsU9mOAtaUUFk3oQoIfzAQFCf5dNMEdTFCQ4NtQih1NSIGgf3ibxOJt5UrAB1gNK72vIdjiI61HWr+YnNxDXK0rJiULsV65GJeiIescLSTTeobKSutiCuojX8kU3MBx4I3WeNVBBRl4fWiCyoB8v2JAAkk9PmDwT8sH1TEghRjgC27scCx41wO43KAg+ILxTvhNaUACwTc04Z0B30LwzTzm5Rjw3sgseIG1wGMawMBPIOQcqvzrNIMHOg9Q5KK953O90/rFC+BhJRH8PQZ+fu7SjC7HAIV95yu99vjlxfvBJx8nwHd6IfNJAkccOjHg6OgIs9lsra6vr2GTNE03/k7q8HAhyJ/2gM9O65/4kT7/mwEcoZwYsPQiV3BwcABb9Ho9KKU2njccDjGdLlxx+InBBPBAAR86ydRPaIC9SASi3+8bnXd+fr78nw8NJ39uDJjXAVFPP7dp/VmWLR9g6w6Huo/IOTk5MTpvZesn/93AiP/dXCwd9SyILT9Jko3n1bZ+8s8rGPGvoVHbEXcPMM39V1dX9Qd/19PPNxta959D4HUGF0RrAFs/8/8mxuPxXLUwtfx2WX+cxdivZ3DFA0SKldZPuPTAKrikbOlMOX+9zFu/Q2iAQoSY5H7mfeb/tXCT8MdneU9wNNCuQUXZA0ynnrUznyqOcrspUY4BJunHqPU3gOgMsNr6G0B0BpgUXrG0fhKVAaaF1/HxMWIhKgNMcj9Tz82Nk6rVGdav/tJ5eraJ0Wi01XPq1r/xOS8uLkJc6XYnRTMNXdf62eIvLy+jyftVghnQ7Xahe8FW59fBTRYOzosDNI1hJdz0lBQkBflkMBjMU5iL13pXRb8fYAJrB/a2db0oFHthAOEUliaYFHE+aaUBdZsvvFhApyM0idYZwOCvW4JmIWdSzPmidQaYrAGZ7iX4oFUGnJ2dGdUCTRqMozeANQCLsE6nA10JG/0Mx4KmDMbBCjEWR2yxu8LAM98vXelmCA2ovVLCI8EMYODWbpbvCXtTBzQVMSAwYkBgxIDAtNKAXWdGIRADAiMpKDA0IIMQikx6QGDEgMCIAYGRMSAsMgaEhgbcQgjFa+kBYZnIGBCWWzEgLPNBOJ6Fk/aR8Y5ZCvktKwX/PJZ7xoVjfs+4chYU11tK2sE85qUBLyH4Zh5z6QHhGPOf6r2j+TEbcgdFP2RaHX5TrYQlDflj5RXE5Q1cG/lWnhYpReUGKdUewGnRmhvnCJbgmxey8sHiZ8iwF3AsUBBckKHI/SWLq6HsBc8huML4DiK80D6WnBqLzN68UFCmopheYJOVYgcU5FOVbAVfYUcUZGoaLPglCtITdg2+tZUFBTFh2+ArWEYh/7z0WIIQSiM43lt5AWAmWhLHylN4QmkNEXfAbGqEQKsHSfHLYwiSq8AnaAAKeaW3D8VbijwNW5nh3IN9FPI/jnpaPKZi2/SfFuJu4W3x9RqWL+N5C+7ruKpBAgLkAAAAAElFTkSuQmCC"
                    }
                },
                {
                    "type": "text",
                    "text": "Please describe this image."
                }
            ]
        }
    ],
    "temperature": 0.3
}

❌ Invalid Format：

{
    "model": "moonshot-v1-8k-vision-preview",
    "messages":
    [
        {
            "role": "system",
            "content": "You are Kimi, an AI assistant provided by Moonshot AI. You are proficient in Chinese and English conversations. You provide users with safe, helpful, and accurate responses. You will refuse to answer any questions involving terrorism, racism, or explicit content. Moonshot AI is a proper noun and should not be translated into other languages."
        },
        {
            "role": "user",
            "content": "[{\"type\": \"image_url\", \"image_url\": {\"url\": \"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAGAAAABhCAYAAAApxKSdAAAACXBIWXMAACE4AAAhOAFFljFgAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAUUSURBVHgB7Z29bhtHFIWPHQN2J7lKqnhYpYvpIukCbJEAKQJEegLReYFIT0DrCSI9QEDqCSIDaQIEIOukiJwyza5SJWlId3FFz+HuGmuSSw6p+dlZ3g84luhdUeI9M3fmziyXgBCUe/DHYY0Wj/tgWmjV42zFcWe4MIBBPNJ6qqW0uvAbXFvQgKzQK62bQhkaCIPc10q1Zi3XH1o/IG9cwUm0RogrgDY1KmLgHYX9DvyiBvDYI77XmiD+oLlQHw7hIDoCMBOt1U9w0BsU9mOAtaUUFk3oQoIfzAQFCf5dNMEdTFCQ4NtQih1NSIGgf3ibxOJt5UrAB1gNK72vIdjiI61HWr+YnNxDXK0rJiULsV65GJeiIescLSTTeobKSutiCuojX8kU3MBx4I3WeNVBBRl4fWiCyoB8v2JAAkk9PmDwT8sH1TEghRjgC27scCx41wO43KAg+ILxTvhNaUACwTc04Z0B30LwzTzm5Rjw3sgseIG1wGMawMBPIOQcqvzrNIMHOg9Q5KK953O90/rFC+BhJRH8PQZ+fu7SjC7HAIV95yu99vjlxfvBJx8nwHd6IfNJAkccOjHg6OgIs9lsra6vr2GTNE03/k7q8HAhyJ/2gM9O65/4kT7/mwEcoZwYsPQiV3BwcABb9Ho9KKU2njccDjGdLlxx+InBBPBAAR86ydRPaIC9SASi3+8bnXd+fr78nw8NJ39uDJjXAVFPP7dp/VmWLR9g6w6Huo/IOTk5MTpvZesn/93AiP/dXCwd9SyILT9Jko3n1bZ+8s8rGPGvoVHbEXcPMM39V1dX9Qd/19PPNxta959D4HUGF0RrAFs/8/8mxuPxXLUwtfx2WX+cxdivZ3DFA0SKldZPuPTAKrikbOlMOX+9zFu/Q2iAQoSY5H7mfeb/tXCT8MdneU9wNNCuQUXZA0ynnrUznyqOcrspUY4BJunHqPU3gOgMsNr6G0B0BpgUXrG0fhKVAaaF1/HxMWIhKgNMcj9Tz82Nk6rVGdav/tJ5eraJ0Wi01XPq1r/xOS8uLkJc6XYnRTMNXdf62eIvLy+jyftVghnQ7Xahe8FW59fBTRYOzosDNI1hJdz0lBQkBflkMBjMU5iL13pXRb8fYAJrB/a2db0oFHthAOEUliaYFHE+aaUBdZsvvFhApyM0idYZwOCvW4JmIWdSzPmidQaYrAGZ7iX4oFUGnJ2dGdUCTRqMozeANQCLsE6nA10JG/0Mx4KmDMbBCjEWR2yxu8LAM98vXelmCA2ovVLCI8EMYODWbpbvCXtTBzQVMSAwYkBgxIDAtNKAXWdGIRADAiMpKDA0IIMQikx6QGDEgMCIAYGRMSAsMgaEhgbcQgjFa+kBYZnIGBCWWzEgLPNBOJ6Fk/aR8Y5ZCvktKwX/PJZ7xoVjfs+4chYU11tK2sE85qUBLyH4Zh5z6QHhGPOf6r2j+TEbcgdFP2RaHX5TrYQlDflj5RXE5Q1cG/lWnhYpReUGKdUewGnRmhvnCJbgmxey8sHiZ8iwF3AsUBBckKHI/SWLq6HsBc8huML4DiK80D6WnBqLzN68UFCmopheYJOVYgcU5FOVbAVfYUcUZGoaLPglCtITdg2+tZUFBTFh2+ArWEYh/7z0WIIQSiM43lt5AWAmWhLHylN4QmkNEXfAbGqEQKsHSfHLYwiSq8AnaAAKeaW3D8VbijwNW5nh3IN9FPI/jnpaPKZi2/SfFuJu4W3x9RqWL+N5C+7ruKpBAgLkAAAAAElFTkSuQmCC\"}}, {\"type\": \"text\", \"text\": \"Please describe this image\"}]"
        }
    ],
    "temperature": 0.3
}

Token Calculation and Costs
Currently, each image consumes a fixed number of 1024 tokens (regardless of image size or quality).

The Vision model follows the same pricing model as the moonshot-v1 series, with costs based on the total tokens used for model inference. For more details, please refer to:

Model Inference Pricing

Features and Limitations
The Vision model supports the following features:

 Multi-turn conversations
 Streaming output
 Tool invocation
 JSON Mode
 Partial Mode
The following features are not supported or only partially supported:

Internet search: Not supported
Context Caching: Creating context caches with image content is not supported, but using existing caches to call the Vision model is supported
URL-formatted images: Not supported, only base64-encoded image content is currently supported
Other limitations:

Image quantity: The Vision model has no limit on the number of images, but ensure that the request body size does not exceed 100M.

Choose the Right Kimi Large Language Model
In the previous chapter, we demonstrated how to quickly use the OpenAI SDK to call the Kimi large language model for multi-turn conversations through a simple example. Let's review the relevant content:

from openai import OpenAI
 
client = OpenAI(
    api_key = "MOONSHOT_API_KEY", # Replace MOONSHOT_API_KEY with the API Key you obtained from the Kimi Open Platform
    base_url = "https://api.moonshot.ai/v1",
)
 
# We define a global variable messages to keep track of the conversation history between us and the Kimi large language model
# The messages include both the questions we pose to the Kimi large language model (role=user) and its responses to us (role=assistant)
# Naturally, it also includes the initial System Prompt (role=system)
# The messages in the list are ordered chronologically
messages = [
	{"role": "system", "content": "You are Kimi, an artificial intelligence assistant provided by Moonshot AI. You excel in Chinese and English conversations. You provide users with safe, helpful, and accurate answers. You also reject any questions involving terrorism, racism, pornography, or violence. Moonshot AI is a proper noun and should not be translated into other languages."},
]
 
def chat(input: str) -> str:
	"""
	The chat function supports multi-turn conversations. Each time the chat function is called to converse with the Kimi large language model, it has access to the previous conversation history. In other words, the Kimi large language model has a memory.
	"""
 
	# We construct the user's latest question as a message (role=user) and append it to the end of the messages list
	messages.append({
		"role": "user",
		"content": input,	
	})
 
	# We engage in a conversation with the Kimi large language model, carrying the messages along
	completion = client.chat.completions.create(
        model="moonshot-v1-8k",
        messages=messages,
        temperature=0.3,
    )
 
	# Through the API, we obtain the response message (role=assistant) from the Kimi large language model
    assistant_message = completion.choices[0].message
 
    # To ensure the Kimi large language model has a complete memory, we must also add the message it returns to us to the messages list
    messages.append(assistant_message)
 
    return assistant_message.content
 
print(chat("Hello, I am 27 years old this year."))
print(chat("Do you know how old I am this year?")) # Here, the Kimi large language model will know your age is 27 based on the previous context

In the previous chapter, we mentioned that for multi-turn conversations, as the number of turns increases, the historical messages will occupy more and more Tokens. However, our code uses a fixed moonshot-v1-8k model, which means that when the historical messages expand beyond 8192 Tokens, calling the chat function will result in a Your request exceeded model token limit error. At this point, if you want to continue the conversation with the Kimi large language model based on the previous context, you need to switch to a model with a larger context, such as moonshot-v1-32k.

In some cases, you may not know how many Tokens the user's next input will occupy, so you also don't know which Kimi large language model to choose (although directly choosing moonshot-v1-128k is always safe, it will deplete your account balance very quickly). You might wonder, is there a way to automatically select the appropriate model based on the number of Tokens occupied by the message?

The moonshot-v1-auto Model
moonshot-v1-auto can select the appropriate model based on the number of Tokens occupied by the current context. The available models include:

moonshot-v1-8k
moonshot-v1-32k
moonshot-v1-128k
moonshot-v1-auto can be thought of as a model router. It decides which specific model to use based on the number of Tokens occupied by the current context. In terms of functionality, moonshot-v1-auto is no different from the aforementioned models.

The billing for moonshot-v1-auto is determined by the final model selected. Its routing rules are as follows (illustrated with code):

def select_model(prompt_tokens: int, max_tokens: int) -> str:
	"""
	prompt_tokens: The number of Tokens occupied by the context (messages + tools) in the current request
	max_tokens:    The value of the max_tokens parameter passed in the current request. If this value is not set, it defaults to 1024
	"""
	total_tokens = prompt_tokens + max_tokens
    if total_tokens <= 8 * 1024:
        return "moonshot-v1-8k"
    elif total_tokens <= 32 * 1024:
        return "moonshot-v1-32k"
    else:
        return "moonshot-v1-128k"

For detailed billing information, please refer to Model Inference.

The usage of moonshot-v1-auto is no different from that of a regular model. In the code above, you only need to make the following changes:

from openai import OpenAI
 
client = OpenAI(
    api_key = "MOONSHOT_API_KEY", # Replace MOONSHOT_API_KEY with the API Key you obtained from the Kimi Open Platform
    base_url = "https://api.moonshot.ai/v1",
)
 
# We define a global variable messages to keep track of the conversation history between us and the Kimi large language model
# The messages include both the questions we pose to the Kimi large language model (role=user) and its responses to us (role=assistant)
# Naturally, it also includes the initial System Prompt (role=system)
# The messages in the list are ordered chronologically
messages = [
	{"role": "system", "content": "You are Kimi, an artificial intelligence assistant provided by Moonshot AI. You excel in Chinese and English conversations. You provide users with safe, helpful, and accurate answers. You also reject any questions involving terrorism, racism, pornography, or violence. Moonshot AI is a proper noun and should not be translated into other languages."},
]
 
def chat(input: str) -> str:
	"""
	The chat function supports multi-turn conversations. Each time the chat function is called to converse with the Kimi large language model, it has access to the previous conversation history. In other words, the Kimi large language model has a memory.
	"""
 
	# We construct the user's latest question as a message (role=user) and append it to the end of the messages list
	messages.append({
		"role": "user",
		"content": input,	
	})
 
	# We engage in a conversation with the Kimi large language model, carrying the messages along
	completion = client.chat.completions.create(
        model="moonshot-v1-auto",  # <-- Note the change here, from moonshot-v1-8k to moonshot-v1-auto
        messages=messages,
        temperature=0.3,
    )
 
	# Through the API, we obtain the response message (role=assistant) from the Kimi large language model
    assistant_message = completion.choices[0].message
 
    # To ensure the Kimi large language model has a complete memory, we must also add the message it returns to us to the messages list
    messages.append(assistant_message)
 
    return assistant_message.content
 
print(chat("Hello, I am 27 years old this year."))
print(chat("Do you know how old I am this year?")) # Here, the Kimi large language model will know your age is 27 based on the previous context

Note the change on line 30, where moonshot-v1-8k is modified to moonshot-v1-auto.

Manually Selecte the Right Model
If you want to manually calculate the number of Tokens and choose the appropriate model, you can refer to the following code:

import os
import httpx
from openai import OpenAI
 
client = OpenAI(
    api_key=os.environ['MOONSHOT_API_KEY'],
    base_url="https://api.moonshot.ai/v1",
)
 
def estimate_token_count(input_messages) -> int:
    """
    Implement your token counting logic here, or directly call our token counting API to calculate tokens.
 
    https://api.moonshot.ai/v1/tokenizers/estimate-token-count
    """
    header = {
        "Authorization": f"Bearer {os.environ['MOONSHOT_API_KEY']}",
    }
    data = {
        "model": "moonshot-v1-128k",
        "messages": input_messages,
    }
    r = httpx.post("https://api.moonshot.ai/v1/tokenizers/estimate-token-count", headers=header, json=data)
    r.raise_for_status()
    return r.json()["data"]["total_tokens"]
 
def select_model(input_messages, max_tokens=1024) -> str:
    """
    select_model chooses an appropriately sized model based on the input context messages (input_messages) and the expected max_tokens value.
 
    Inside select_model, the estimate_token_count function is called to calculate the number of tokens used by input_messages. This number is added to the max_tokens value to get total_tokens, and a suitable model is selected based on the range of total_tokens.
    """
    prompt_tokens = estimate_token_count(input_messages)
    total_tokens = prompt_tokens + max_tokens
    if total_tokens <= 8 * 1024:
        return "moonshot-v1-8k"
    elif total_tokens <= 32 * 1024:
        return "moonshot-v1-32k"
    elif total_tokens <= 128 * 1024:
        return "moonshot-v1-128k"
    else:
        raise Exception("too many tokens 😢")
 
messages = [
    {"role": "system", "content": "You are Kimi"},
    {"role": "user", "content": "Hello, please tell me a fairy tale."},
]
 
max_tokens = 2048
model = select_model(messages, max_tokens)
 
completion = client.chat.completions.create(
    model=model,
    messages=messages,
    max_tokens=max_tokens,
    temperature=0.3,
)
 
print("model:", model)
print("max_tokens:", max_tokens)
print("completion:", completion.choices[0].message.content)


Automatic Reconnection on Disconnect
Due to concurrency limits, complex network environments, and other unforeseen circumstances, our connections may sometimes be interrupted. Typically, these intermittent disruptions don't last long. We want our services to remain stable even in such cases. Implementing a simple reconnection feature can be achieved with just a few lines of code.

from openai import OpenAI
import time
 
client = OpenAI(
    api_key = "$MOONSHOT_API_KEY",
    base_url = "https://api.moonshot.ai/v1",
)
 
def chat_once(msgs):
    response = client.chat.completions.create(
        model = "moonshot-v1-auto",
        messages = msgs,
        temperature = 0.3,
    )
    return response.choices[0].message.content
 
def chat(input: str, max_attempts: int = 100) -> str:
    messages = [
        {"role": "system", "content": "You are Kimi, an AI assistant provided by Moonshot AI. You are proficient in Chinese and English conversations. You aim to provide users with safe, helpful, and accurate answers. You will refuse to answer any questions related to terrorism, racism, or explicit content. Moonshot AI is a proper noun and should not be translated into other languages."},
    ]
 
    # We construct the user's latest question as a message (role=user) and append it to the end of the messages list
    messages.append({
        "role": "user",
        "content": input,
    })
    st_time = time.time()
    for i in range(max_attempts):
        print(f"Attempts: {i+1}/{max_attempts}")
        try:
            response = chat_once(messages)
            ed_time = time.time()
            print("Query Successful!")
            print(f"Query Time: {ed_time-st_time}")
            return response
        except Exception as e:
            print(e)
            time.sleep(1)
            continue
 
    print("Query Failed.")
    return
 
print(chat("Hello, please tell me a fairy tale."))

The code above implements a simple reconnection feature, allowing up to 100 retries with a 1-second wait between each attempt. You can adjust these values and the conditions for retries based on your specific needs.


Use the Streaming Feature of the Kimi API
When the Kimi large language model receives a question from a user, it first performs inference and then generates the response one Token at a time. In the examples from our first two chapters, we chose to wait for the Kimi large language model to generate all Tokens before printing its response. This usually takes several seconds. If your question is complex enough and the response from the Kimi large language model is long enough, the time to wait for the complete response can be stretched to 10 or even 20 seconds, which greatly reduces the user experience. To improve this situation and provide timely feedback to users, we offer the ability to stream output, known as Streaming. We will explain the principles of Streaming and illustrate it with actual code:

How to use streaming output;
Common issues when using streaming output;
How to handle streaming output without using the Python SDK;
How to Use Streaming Output
Streaming, in a nutshell, means that whenever the Kimi large language model generates a certain number of Tokens (usually 1 Token), it immediately sends these Tokens to the client, instead of waiting for all Tokens to be generated before sending them to the client. When you chat with Kimi AI Assistant, the assistant's response appears character by character, which is one manifestation of streaming output. Streaming allows users to see the first Token output by the Kimi large language model immediately, reducing wait time.

You can use streaming output in this way (stream=True) and get the streaming response:

from openai import OpenAI
 
client = OpenAI(
    api_key = "MOONSHOT_API_KEY", # Replace MOONSHOT_API_KEY with the API Key you obtained from the Kimi Open Platform
    base_url = "https://api.moonshot.ai/v1",
)
 
stream = client.chat.completions.create(
    model = "moonshot-v1-8k",
    messages = [
        {"role": "system", "content": "You are Kimi, an artificial intelligence assistant provided by Moonshot AI, who is better at conversing in Chinese and English. You provide users with safe, helpful, and accurate answers. At the same time, you refuse to answer any questions related to terrorism, racism, pornography, and violence. Moonshot AI is a proper noun and should not be translated into other languages."},
        {"role": "user", "content": "Hello, my name is Li Lei, what is 1+1?"}
    ],
    temperature = 0.3,
    stream=True, # <-- Note here, we enable streaming output mode by setting stream=True
)
 
# When streaming output mode is enabled (stream=True), the content returned by the SDK also changes. We no longer directly access the choice in the return value
# Instead, we access each individual chunk in the return value through a for loop
 
for chunk in stream:
	# Here, the structure of each chunk is similar to the previous completion, but the message field is replaced with the delta field
	delta = chunk.choices[0].delta # <-- The message field is replaced with the delta field
 
	if delta.content:
		# When printing the content, since it is streaming output, to ensure the coherence of the sentence, we do not add
		# line breaks manually, so we set end="" to cancel the line break of print.
		print(delta.content, end="")

Common Issues When Using Streaming Output
Now that you have successfully run the above code and understood the basic principles of streaming output, let's discuss some details and common issues of streaming output so that you can better implement your business logic.

Interface Details
When streaming output mode is enabled (stream=True), the Kimi large language model no longer returns a response in JSON format (Content-Type: application/json), but uses Content-Type: text/event-stream (abbreviated as SSE). This response format allows the server to continuously send data to the client. In the context of using the Kimi large language model, it can be understood as the server continuously sending Tokens to the client.

When you look at the HTTP response body of SSE, it looks like this:

data: {"id":"cmpl-1305b94c570f447fbde3180560736287","object":"chat.completion.chunk","created":1698999575,"model":"moonshot-v1-8k","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}
 
data: {"id":"cmpl-1305b94c570f447fbde3180560736287","object":"chat.completion.chunk","created":1698999575,"model":"moonshot-v1-8k","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}
 
...
 
data: {"id":"cmpl-1305b94c570f447fbde3180560736287","object":"chat.completion.chunk","created":1698999575,"model":"moonshot-v1-8k","choices":[{"index":0,"delta":{"content":"."},"finish_reason":null}]}
 
data: {"id":"cmpl-1305b94c570f447fbde3180560736287","object":"chat.completion.chunk","created":1698999575,"model":"moonshot-v1-8k","choices":[{"index":0,"delta":{},"finish_reason":"stop","usage":{"prompt_tokens":19,"completion_tokens":13,"total_tokens":32}}]}
 
data: [DONE]

In the response body of SSE, we agree that each data chunk starts with data: , followed by a valid JSON object, and ends with two newline characters \n\n. Finally, when all data chunks have been transmitted, data: [DONE] is used to indicate that the transmission is complete, at which point the network connection can be disconnected.

Token Calculation
When using the streaming output mode, there are two ways to calculate tokens. The most straightforward and accurate method is to wait until all data chunks have been transmitted and then check the prompt_tokens, completion_tokens, and total_tokens in the usage field of the last data chunk.

...
 
data: {"id":"cmpl-1305b94c570f447fbde3180560736287","object":"chat.completion.chunk","created":1698999575,"model":"moonshot-v1-8k","choices":[{"index":0,"delta":{},"finish_reason":"stop","usage":{"prompt_tokens":19,"completion_tokens":13,"total_tokens":32}}]}
                                               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                               Check the number of tokens generated by the current request through the usage field in the last data chunk
data: [DONE]

However, in practice, streaming output can be interrupted by uncontrollable factors such as network disconnections or client-side errors. In such cases, the last data chunk may not have been fully transmitted, making it impossible to know the total number of tokens consumed by the request. To avoid this issue, we recommend saving the content of each data chunk as it is received and then using the token calculation interface to compute the total consumption after the request ends, regardless of whether it was successful or not. Here is an example code snippet:

import os
import httpx
from openai import OpenAI
 
client = OpenAI(
    api_key = "MOONSHOT_API_KEY", # Replace MOONSHOT_API_KEY with the API Key you obtained from the Kimi Open Platform
    base_url = "https://api.moonshot.ai/v1",
)
 
stream = client.chat.completions.create(
    model = "moonshot-v1-8k",
    messages = [
        {"role": "system", "content": "You are Kimi, an AI assistant provided by Moonshot AI, who excels in Chinese and English conversations. You provide users with safe, helpful, and accurate answers while rejecting any questions related to terrorism, racism, or explicit content. Moonshot AI is a proper noun and should not be translated."},
        {"role": "user", "content": "Hello, my name is Li Lei. What is 1+1?"}
    ],
    temperature = 0.3,
    stream=True, # <-- Note here, we enable streaming output mode by setting stream=True
)
 
 
def estimate_token_count(input: str) -> int:
    """
    Implement your token calculation logic here, or directly call our token calculation interface to compute tokens.
 
    https://api.moonshot.ai/v1/tokenizers/estimate-token-count
    """
    header = {
        "Authorization": f"Bearer {os.environ['MOONSHOT_API_KEY']}",
    }
    data = {
        "model": "moonshot-v1-128k",
        "messages": [
            {"role": "user", "content": input},
        ]
    }
    r = httpx.post("https://api.moonshot.ai/v1/tokenizers/estimate-token-count", headers=header, json=data)
    r.raise_for_status()
    return r.json()["data"]["total_tokens"]
 
 
completion = []
for chunk in stream:
	delta = chunk.choices[0].delta
	if delta.content:
		completion.append(delta.content)
 
 
print("completion_tokens:", estimate_token_count("".join(completion)))

How to Terminate Output
If you want to stop the streaming output, you can simply close the HTTP connection or discard any subsequent data chunks. For example:

for chunk in stream:
	if condition:
		break

How to Handle Streaming Output Without Using an SDK
If you prefer not to use the Python SDK to handle streaming output and instead want to directly interface with HTTP APIs to use the Kimi large language model (for example, in cases where you are using a language without an SDK, or you have unique business logic that the SDK cannot meet), we provide some examples to help you understand how to properly handle the SSE response body in HTTP (we still use Python code as an example here, with detailed explanations provided in comments).

import httpx # We use the httpx library to make our HTTP requests
 
 
data = {
	"model": "moonshot-v1-128k",
	"messages": [
		# Specific messages
	],
	"temperature": 0.3,
	"stream": True,
}
 
 
# Use httpx to send a chat request to the Kimi large language model and get the response r
r = httpx.post("https://api.moonshot.ai/v1/chat/completions", json=data)
if r.status_code != 200:
	raise Exception(r.text)
 
 
data: str
 
# Here, we use the iter_lines method to read the response body line by line
for line in r.iter_lines():
	# Remove leading and trailing spaces from each line to better handle data chunks
	line = line.strip()
 
	# Next, we need to handle three different cases:
	#   1. If the current line is empty, it indicates that the previous data chunk has been received (as mentioned earlier, the data chunk transmission ends with two newline characters), we can deserialize the data chunk and print the corresponding content;
	#   2. If the current line is not empty and starts with data:, it indicates the start of a data chunk transmission, we remove the data: prefix and first check if it is the end symbol [DONE], if not, save the data content to the data variable;
	#   3. If the current line is not empty but does not start with data:, it indicates that the current line still belongs to the previous data chunk being transmitted, we append the content of the current line to the end of the data variable;
 
	if len(line) == 0:
		chunk = json.loads(data)
 
		# The processing logic here can be replaced with your business logic, printing is just to demonstrate the process
		choice = chunk["choices"][0]
		usage = choice.get("usage")
		if usage:
			print("total_tokens:", usage["total_tokens"])
		delta = choice["delta"]
		role = delta.get("role")
		if role:
			print("role:", role)
		content = delta.get("content")
		if content:
			print(content, end="")
 
		data = "" # Reset data
	elif line.startswith("data: "):
		data = line.lstrip("data: ")
 
		# When the data chunk content is [DONE], it indicates that all data chunks have been sent, and the network connection can be disconnected
		if data == "[DONE]":
			break
	else:
		data = data + "\n" + line # We still add a newline character when appending content, as this data chunk may intentionally format the data in separate lines

The above is the process of handling streaming output using Python as an example. If you are using other languages, you can also properly handle the content of streaming output. The basic steps are as follows:

Initiate an HTTP request and set the stream parameter in the request body to true;
Receive the response from the server. Note that if the Content-Type in the response Headers is text/event-stream, it indicates that the response content is a streaming output;
Read the response content line by line and parse the data chunks (the data chunks are presented in JSON format). Pay attention to determining the start and end positions of the data chunks through the data: prefix and newline character \n;
Determine whether the transmission is complete by checking if the current data chunk content is [DONE];
Note: Always use data: [DONE] to determine if the data has been fully transmitted, rather than using finish_reason or other methods. If you do not receive the data: [DONE] message chunk, even if you have obtained the information finish_reason=stop, you should not consider the data chunk transmission as complete. In other words, until you receive the data: [DONE] data chunk, the message should be considered incomplete.

During the streaming output process, only the content field is streamed, meaning each data chunk contains a portion of the content tokens. For fields that do not need to be streamed, such as role and usage, we usually present them all at once in the first or last data chunk, rather than including the role and usage fields in every data chunk (specifically, the role field will only appear in the first data chunk and will not be included in subsequent data chunks; the usage field will only appear in the last data chunk and will not be included in the preceding data chunks).

Handling n>1
Sometimes, we want to get multiple results to choose from. To do this, you should set the n parameter in the request to a value greater than 1. When it comes to stream output, we also support the use of n>1. In such cases, we need to add some extra code to determine the index value of the current data block, to figure out which response the data block belongs to. Let's illustrate this with example code:

import httpx # We use the httpx library to make our HTTP requests
 
 
data = {
	"model": "moonshot-v1-128k",
	"messages": [
		# Specific messages go here
	],
	"temperature": 0.3,
	"stream": True,
	"n": 2, # <-- Note here, we're asking the Kimi large language model to output 2 responses
}
 
 
# Use httpx to send a chat request to the Kimi large language model and get the response r
r = httpx.post("https://api.moonshot.ai/v1/chat/completions", json=data)
if r.status_code != 200:
	raise Exception(r.text)
 
 
data: str
 
# Here, we pre-build a List to store different response messages. Since we set n=2, we initialize the List with 2 elements
messages = [{}, {}]
 
# We use the iter_lines method here to read the response body line by line
for line in r.iter_lines():
	# Remove leading and trailing spaces from each line to better handle data blocks
	line = line.strip()
 
	# Next, we need to handle three different scenarios:
	#   1. If the current line is empty, it indicates that the previous data block has been fully received (as mentioned earlier, data block transmission ends with two newline characters). We can deserialize this data block and print out the corresponding content;
	#   2. If the current line is not empty and starts with data:, it means the start of a data block transmission. After removing the data: prefix, we first check if it's the end marker [DONE]. If not, we save the data content to the data variable;
	#   3. If the current line is not empty but doesn't start with data:, it means this line still belongs to the previous data block being transmitted. We append the content of this line to the end of the data variable;
 
	if len(line) == 0:
		chunk = json.loads(data)
 
		# Loop through all choices in each data block to get the message object corresponding to the index
		for choice in chunk["choices"]:
			index = choice["index"]
			message = messages[index]
			usage = choice.get("usage")
			if usage:
				message["usage"] = usage
			delta = choice["delta"]
			role = delta.get("role")
			if role:
				message["role"] = role
			content = delta.get("content")
			if content:
				message["content"] = message.get("content", "") + content
 
			data = "" # Reset data
	elif line.startswith("data: "):
		data = line.lstrip("data: ")
 
		# When the data block content is [DONE], it means all data blocks have been sent and we can disconnect the network
		if data == "[DONE]":
			break
	else:
		data = data + "\n" + line # When we're still appending content, we add a newline character because this might be the data block's intentional way of displaying data on separate lines
 
 
# After assembling all messages, we print their contents separately
for index, message in enumerate(messages):
	print("index:", index)
	print("message:", json.dumps(message, ensure_ascii=False))

When n>1, the key to handling stream output is to first determine which response message the current data block belongs to based on its index value, and then proceed with further logical processing.

Use Kimi API for Tool Calls
Tool calls, or tool_calls, evolved from function calls (function_call). In certain contexts, or when reading compatibility code, you can consider tool_calls and function_call to be the same. function_call is a subset of tool_calls.

What are Tool Calls?
Tool calls give the Kimi large language model the ability to perform specific actions. The Kimi large language model can engage in conversations and answer questions, which is its "talking" ability. Through tool calls, it also gains the ability to "do" things. With tool_calls, the Kimi large language model can help you search the internet, query databases, and even control smart home devices.

A tool call involves several steps:

Define the tool using JSON Schema format;
Submit the defined tool to the Kimi large language model via the tools parameter. You can submit multiple tools at once;
The Kimi large language model will decide which tool(s) to use based on the context of the current conversation. It can also choose not to use any tools;
The Kimi large language model will output the parameters and information needed to call the tool in JSON format;
Use the parameters output by the Kimi large language model to execute the corresponding tool and submit the results back to the Kimi large language model;
The Kimi large language model will respond to the user based on the results of the tool execution;
Reading the above steps, you might wonder:

Why can't the Kimi large language model execute the tools itself? Why do we need to "help" the Kimi large language model execute the tools based on the parameters it generates? If we are the ones executing the tool calls, what is the role of the Kimi large language model?

We will use a practical example of a tool call to explain these questions to the reader.

Enable the Kimi Large Language Model to Access the Internet via tool_calls
The knowledge of the Kimi large language model comes from its training data. For questions that are time-sensitive, the Kimi large language model cannot find answers from its existing knowledge. In such cases, we want the Kimi large language model to search the internet for the latest information and answer our questions based on that information.

Define the Tools
Imagine how we find the information we want on the internet:

We open a search engine, such as Baidu or Bing, and search for the content we want. We then browse the search results and decide which one to click based on the website title and description;
We might open one or more web pages from the search results and browse them to obtain the knowledge we need;
Reviewing our actions, we "use a search engine to search" and "open the web pages corresponding to the search results." The tools we use are the "search engine" and the "web browser." Therefore, we need to abstract these actions into tools in JSON Schema format and submit them to the Kimi large language model, allowing it to use search engines and browse web pages just like humans do.

Before we proceed, let's briefly introduce the JSON Schema format:

JSON Schema is a vocabulary that you can use to annotate and validate JSON documents.

JSON Schema is a JSON document used to describe the format of JSON data.

We define the following JSON Schema:

{
	"type": "object",
	"properties": {
		"name": {
			"type": "string"
		}
	}
}

This JSON Schema defines a JSON Object that contains a field named name, and the type of this field is string, for example:

{
	"name": "Hei"
}

By describing our tool definitions using JSON Schema, we can make it clearer and more intuitive for the Kimi large language model to understand what parameters our tools require, as well as the type and description of each parameter. Now let's define the "search engine" and "web browser" tools mentioned earlier:

tools = [
	{
		"type": "function", # The agreed-upon field type, currently supports function as a value
		"function": { # When type is function, use the function field to define the specific function content
			"name": "search", # The name of the function. Please use English letters, numbers, hyphens, and underscores as the function name
			"description": """ 
				Search for content on the internet using a search engine.
 
				When your knowledge cannot answer the user's question, or when the user requests an online search, call this tool. Extract the content the user wants to search for from the conversation and use it as the value of the query parameter.
				The search results include the website title, address (URL), and description.
			""", # A description of the function, detailing its specific role and usage scenarios, to help the Kimi large language model correctly select which functions to use
			"parameters": { # Use the parameters field to define the parameters the function accepts
				"type": "object", # Always use type: object to make the Kimi large language model generate a JSON Object parameter
				"required": ["query"], # Use the required field to tell the Kimi large language model which parameters are mandatory
				"properties": { # The properties field contains the specific parameter definitions; you can define multiple parameters
					"query": { # Here, the key is the parameter name, and the value is the specific definition of the parameter
						"type": "string", # Use type to define the parameter type
						"description": """
							The content the user wants to search for, extracted from the user's question or conversation context.
						""" # Use description to describe the parameter so that the Kimi large language model can better generate the parameter
					}
				}
			}
		}
	},
	{
		"type": "function", # The agreed-upon field type, currently supports function as a value
		"function": { # When type is function, use the function field to define the specific function content
			"name": "crawl", # The name of the function. Please use English letters, numbers, hyphens, and underscores as the function name
			"description": """
				Retrieve web page content based on the website address (URL).
			""", # A description of the function, detailing its specific role and usage scenarios, to help the Kimi large language model correctly select which functions to use
			"parameters": { # Use the parameters field to define the parameters the function accepts
				"type": "object", # Always use type: object to make the Kimi large language model generate a JSON Object parameter
				"required": ["url"], # Use the required field to tell the Kimi large language model which parameters are mandatory
				"properties": { # The properties field contains the specific parameter definitions; you can define multiple parameters
					"url": { # Here, the key is the parameter name, and the value is the specific definition of the parameter
						"type": "string", # Use type to define the parameter type
						"description": """
							The website address (URL) from which to retrieve content, usually obtained from search results.
						""" # Use description to describe the parameter so that the Kimi large language model can better generate the parameter
					}
				}
			}
		}
	}
]

When defining tools using JSON Schema, we use the following fixed format:

{
	"type": "function",
	"function": {
		"name": "NAME",
		"description": "DESCRIPTION",
		"parameters": {
			"type": "object",
			"properties": {
				
			}
		}
	}
}

Here, name, description, and parameters.properties are defined by the tool provider. The description explains the specific function and when to use the tool, while parameters outlines the specific parameters needed to successfully call the tool, including parameter types and descriptions. Ultimately, the Kimi large language model will generate a JSON Object that meets the defined requirements as the parameters (arguments) for the tool call based on the JSON Schema.

Register Tools
Let's try submitting the search tool to the Kimi large language model to see if it can correctly call the tool:

from openai import OpenAI
 
 
client = OpenAI(
    api_key="MOONSHOT_API_KEY", # Replace MOONSHOT_API_KEY with the API Key you obtained from the Kimi Open Platform
    base_url="https://api.moonshot.ai/v1",
)
 
tools = [
	{
		"type": "function", # The field "type" is a convention, currently supporting "function" as its value
		"function": { # When "type" is "function", use the "function" field to define the specific function content
			"name": "search", # The name of the function, please use English letters, numbers, plus hyphens and underscores as the function name
			"description": """ 
				Search for content on the internet using a search engine.
 
				When your knowledge cannot answer the user's question, or when the user requests an online search, call this tool. Extract the content the user wants to search from the conversation as the value of the query parameter.
				The search results include the website title, website address (URL), and website description.
			""", # Description of the function, write the specific function and usage scenarios here so that the Kimi large language model can correctly choose which functions to use
			"parameters": { # Use the "parameters" field to define the parameters accepted by the function
				"type": "object", # Always use "type": "object" to make the Kimi large language model generate a JSON Object parameter
				"required": ["query"], # Use the "required" field to tell the Kimi large language model which parameters are required
				"properties": { # The specific parameter definitions are in "properties", you can define multiple parameters
					"query": { # Here, the key is the parameter name, and the value is the specific definition of the parameter
						"type": "string", # Use "type" to define the parameter type
						"description": """
							The content the user wants to search for, extract it from the user's question or chat context.
						""" # Use "description" to describe the parameter so that the Kimi large language model can better generate the parameter
					}
				}
			}
		}
	},
	# {
	# 	"type": "function", # The field "type" is a convention, currently supporting "function" as its value
	# 	"function": { # When "type" is "function", use the "function" field to define the specific function content
	# 		"name": "crawl", # The name of the function, please use English letters, numbers, plus hyphens and underscores as the function name
	# 		"description": """
	# 			Get the content of a webpage based on the website address (URL).
	# 		""", // Description of the function, write the specific function and usage scenarios here so that the Kimi large language model can correctly choose which functions to use
	# 		"parameters": { // Use the "parameters" field to define the parameters accepted by the function
	# 			"type": "object", // Always use "type": "object" to make the Kimi large language model generate a JSON Object parameter
	# 			"required": ["url"], // Use the "required" field to tell the Kimi large language model which parameters are required
	# 			"properties": { // The specific parameter definitions are in "properties", you can define multiple parameters
	# 				"url": { // Here, the key is the parameter name, and the value is the specific definition of the parameter
	# 					"type": "string", // Use "type" to define the parameter type
	# 					"description": """
	# 						The website address (URL) of the content to be obtained, which can usually be obtained from the search results.
	# 					""" // Use "description" to describe the parameter so that the Kimi large language model can better generate the parameter
	# 				}
	# 			}
	# 		}
	# 	}
	# }
]
 
```python
completion = client.chat.completions.create(
    model="moonshot-v1-8k",
    messages=[
        {"role": "system", "content": "You are Kimi, an AI assistant provided by Moonshot AI. You are proficient in Chinese and English conversations. You provide users with safe, helpful, and accurate answers. You refuse to answer any questions related to terrorism, racism, or explicit content. Moonshot AI is a proper noun and should not be translated."},
        {"role": "user", "content": "Please search the internet for 'Context Caching' and tell me what it is."} # In the question, we ask Kimi large language model to search online
    ],
    temperature=0.3,
    tools=tools, # <-- We pass the defined tools to Kimi large language model via the tools parameter
)
 
print(completion.choices[0].model_dump_json(indent=4))

When the above code runs successfully, we get the response from Kimi large language model:

{
    "finish_reason": "tool_calls",
    "message": {
        "content": "",
        "role": "assistant",
        "tool_calls": [
            {
                "id": "search:0",
                "function": {
                    "arguments": "{\n    \"query\": \"Context Caching\"\n}",
                    "name": "search"
                },
                "type": "function",
            }
        ]
    }
}

Notice that in this response, the value of finish_reason is tool_calls, which means that the response is not the answer from Kimi large language model, but rather the tool that Kimi large language model has chosen to execute. You can determine whether the current response from Kimi large language model is a tool call tool_calls by checking the value of finish_reason.

In the message section, the content field is empty because the model is currently executing tool_calls and has not yet generated a response for the user. Meanwhile, a new field tool_calls has been added. The tool_calls field is a list that contains all the tool call information for this execution. This also indicates another characteristic of tool_calls: the model can choose to call multiple tools at once, which can be different tools or the same tool with different parameters. Each element in tool_calls represents a tool call. Kimi large language model generates a unique id for each tool call. The function.name field indicates the name of the function being executed, and the parameters are placed in function.arguments. The arguments parameter is a valid serialized JSON Object (additionally, the type parameter is currently a fixed value function).

Next, we should use the tool call parameters generated by Kimi large language model to execute the specific tools.

Execute the Tools
Kimi large language model does not execute the tools for us. We need to execute the parameters generated by Kimi large language model after receiving them. Before explaining how to execute the tools, let's first address the question we raised earlier:

Why can't Kimi large language model execute the tools itself, but instead requires us to "help" it execute the tools based on the parameters generated by Kimi large language model? If we are the ones executing the tool calls, what is the purpose of Kimi large language model?

Let's imagine a scenario where we use Kimi large language model: we provide users with a smart robot based on Kimi large language model. In this scenario, there are three roles: the user, the robot, and Kimi large language model. The user asks the robot a question, the robot calls the Kimi large language model API, and returns the API result to the user. When using tool_calls, the user asks the robot a question, the robot calls the Kimi API with tools, Kimi large language model returns the tool_calls parameters, the robot executes the tool_calls, submits the results back to the Kimi API, Kimi large language model generates the message to be returned to the user (finish_reason=stop), and only then does the robot return the message to the user. Throughout this process, the entire tool_calls process is transparent and implicit to the user.

Returning to the question above, as users, we are not actually executing the tool calls, nor do we directly "see" the tool calls. Instead, the robot that provides us with the service is completing the tool calls and presenting us with the final response generated by Kimi large language model.

Let's explain how to execute the tool_calls returned by Kimi large language model from the perspective of the "robot":

from typing import *
 
import json
 
from openai import OpenAI
 
 
client = OpenAI(
    api_key="MOONSHOT_API_KEY", # Replace MOONSHOT_API_KEY with the API Key you obtained from the Kimi Open Platform
    base_url="https://api.moonshot.ai/v1",
)
 
tools = [
	{
		"type": "function", # The field type is agreed upon, and currently supports function as a value
		"function": { # When type is function, use the function field to define the specific function content
			"name": "search", # The name of the function, please use English letters, numbers, plus hyphens and underscores as the function name
			"description": """ 
				Search for content on the internet using a search engine.
 
				When your knowledge cannot answer the user's question, or the user requests you to perform an online search, call this tool. Extract the content the user wants to search from the conversation as the value of the query parameter.
				The search results include the title of the website, the website address (URL), and a brief introduction to the website.
			""", # Introduction to the function, write the specific function here, as well as the usage scenario, so that the Kimi large language model can correctly choose which functions to use
			"parameters": { # Use the parameters field to define the parameters accepted by the function
				"type": "object", # Fixed use type: object to make the Kimi large language model generate a JSON Object parameter
				"required": ["query"], # Use the required field to tell the Kimi large language model which parameters are required
				"properties": { # The specific parameter definitions are in properties, and you can define multiple parameters
					"query": { # Here, the key is the parameter name, and the value is the specific definition of the parameter
						"type": "string", # Use type to define the parameter type
						"description": """
							The content the user wants to search for, extracted from the user's question or chat context.
						""" # Use description to describe the parameter so that the Kimi large language model can better generate the parameter
					}
				}
			}
		}
	},
	{
		"type": "function", # The field type is agreed upon, and currently supports function as a value
		"function": { # When type is function, use the function field to define the specific function content
			"name": "crawl", # The name of the function, please use English letters, numbers, plus hyphens and underscores as the function name
			"description": """
				Get the content of a webpage based on the website address (URL).
			""", # Introduction to the function, write the specific function here, as well as the usage scenario, so that the Kimi large language model can correctly choose which functions to use
			"parameters": { # Use the parameters field to define the parameters accepted by the function
				"type": "object", # Fixed use type: object to make the Kimi large language model generate a JSON Object parameter
				"required": ["url"], # Use the required field to tell the Kimi large language model which parameters are required
				"properties": { # The specific parameter definitions are in properties, and you can define multiple parameters
					"url": { # Here, the key is the parameter name, and the value is the specific definition of the parameter
						"type": "string", # Use type to define the parameter type
						"description": """
							The website address (URL) of the content to be obtained, which can usually be obtained from the search results.
						""" # Use description to describe the parameter so that the Kimi large language model can better generate the parameter
					}
				}
			}
		}
	}
]
 
 
def search_impl(query: str) -> List[Dict[str, Any]]:
    """
    search_impl uses a search engine to search for query. Most mainstream search engines (such as Bing) provide API calls. You can choose
    your preferred search engine API and place the website title, link, and brief introduction information from the return results in a dict to return.
 
    This is just a simple example, and you may need to write some authentication, validation, and parsing code.
    """
    r = httpx.get("https://your.search.api", params={"query": query})
    return r.json()
 
 
def search(arguments: Dict[str, Any]) -> Any:
    query = arguments["query"]
    result = search_impl(query)
    return {"result": result}
 
 
def crawl_impl(url: str) -> str:
    """
    crawl_url gets the content of a webpage based on the url.
 
    This is just a simple example. In actual web scraping, you may need to write more code to handle complex situations, such as asynchronously loaded data; and after obtaining
    the webpage content, you can clean the webpage content according to your needs, such as retaining only the text or removing unnecessary content (such as advertisements).
    """
    r = httpx.get(url)
    return r.text
 
 
def crawl(arguments: dict) -> str:
    url = arguments["url"]
    content = crawl_impl(url)
    return {"content": content}
 
 
# Map each tool name and its corresponding function through tool_map so that when the Kimi large language model returns tool_calls, we can quickly find the function to execute
tool_map = {
    "search": search,
    "crawl": crawl,
}
 
messages = [
    {"role": "system",
     "content": "You are Kimi, an artificial intelligence assistant provided by Moonshot AI. You are better at conversing in Chinese and English. You provide users with safe, helpful, and accurate answers. At the same time, you will refuse to answer any questions involving terrorism, racial discrimination, pornography, and violence. Moonshot AI is a proper noun and should not be translated into other languages."},
    {"role": "user", "content": "Please search for Context Caching online and tell me what it is."}  # Request Kimi large language model to perform an online search in the question
]
 
finish_reason = None
 
 
# Our basic process is to ask the Kimi large language model questions with the user's question and tools. If the Kimi large language model returns finish_reason: tool_calls, we execute the corresponding tool_calls,
# and submit the execution results in the form of a message with role=tool back to the Kimi large language model. The Kimi large language model then generates the next content based on the tool_calls results:
#
#   1. If the Kimi large language model believes that the current tool call results can answer the user's question, it returns finish_reason: stop, and we exit the loop and print out message.content;
#   2. If the Kimi large language model believes that the current tool call results cannot answer the user's question and needs to call the tool again, we continue to execute the next tool_calls in the loop until finish_reason is no longer tool_calls;
#
# During this process, we only return the result to the user when finish_reason is stop.
 
while finish_reason is None or finish_reason == "tool_calls":
    completion = client.chat.completions.create(
        model="moonshot-v1-8k",
        messages=messages,
        temperature=0.3,
        tools=tools,  # <-- We submit the defined tools to the Kimi large language model through the tools parameter
    )
    choice = completion.choices[0]
    finish_reason = choice.finish_reason
    if finish_reason == "tool_calls": # <-- Determine whether the current return content contains tool_calls
        messages.append(choice.message) # <-- We add the assistant message returned to us by the Kimi large language model to the context so that the Kimi large language model can understand our request next time
        for tool_call in choice.message.tool_calls: # <-- tool_calls may be multiple, so we use a loop to execute them one by one
            tool_call_name = tool_call.function.name
            tool_call_arguments = json.loads(tool_call.function.arguments) # <-- arguments is a serialized JSON Object, and we need to deserialize it with json.loads
            tool_function = tool_map[tool_call_name] # <-- Quickly find which function to execute through tool_map
            tool_result = tool_function(tool_call_arguments)
 
            # Construct a message with role=tool using the function execution result to show the result of the tool call to the model;
            # Note that we need to provide the tool_call_id and name fields in the message so that the Kimi large language model
            # can correctly match the corresponding tool_call.
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_call_name,
                "content": json.dumps(tool_result), # <-- We agree to submit the tool call result to the Kimi large language model in string format, so we use json.dumps to serialize the execution result into a string here
            })
 
print(choice.message.content) # <-- Here, we return the reply generated by the model to the user

We use a while loop to execute the code logic that includes tool calls because the Kimi large language model typically doesn't make just one tool call, especially in the context of online searching. Usually, Kimi will first call the search tool to get search results, and then call the crawl tool to convert the URLs in the search results into actual web page content. The overall structure of the messages is as follows:

system: prompt                                                                                               # System prompt
user: prompt                                                                                                 # User's question
assistant: tool_call(name=search, arguments={query: query})                                                  # Kimi returns a tool_call (single)
tool: search_result(tool_call_id=tool_call.id, name=search)                                                  # Submit the tool_call execution result
assistant: tool_call_1(name=crawl, arguments={url: url_1}), tool_call_2(name=crawl, arguments={url: url_2})  # Kimi continues to return tool_calls (multiple)
tool: crawl_content(tool_call_id=tool_call_1.id, name=crawl)                                                 # Submit the execution result of tool_call_1
tool: crawl_content(tool_call_id=tool_call_2.id, name=crawl)                                                 # Submit the execution result of tool_call_2
assistant: message_content(finish_reason=stop)                                                               # Kimi generates a reply to the user, ending the conversation

This completes the entire process of making "online query" tool calls. If you have implemented your own search and crawl methods, when you ask Kimi to search online, it will call the search and crawl tools and give you the correct response based on the tool call results.

Common Questions and Notes
About Streaming Output
In streaming output mode (stream), tool_calls are still applicable, but there are some additional things to note, as follows:

During streaming output, since finish_reason will appear in the last data chunk, it is recommended to check if the delta.tool_calls field exists to determine if the current response includes a tool call;
During streaming output, delta.content will be output first, followed by delta.tool_calls, so you must wait until delta.content has finished outputting before you can determine and identify tool_calls;
During streaming output, we will specify the tool_call.id and tool_call.function.name in the initial data chunk, and only tool_call.function.arguments will be output in subsequent chunks;
During streaming output, if Kimi returns multiple tool_calls at once, we will use an additional field called index to indicate the index of the current tool_call, so that you can correctly concatenate the tool_call.function.arguments parameters. We use a code example from the streaming output section (without using the SDK) to illustrate how to do this:
import os
import json
import httpx  # We use the httpx library to make our HTTP requests
 
 
 
tools = [
    {
        "type": "function",  # The type field is fixed as "function"
        "function": {  # When type is "function", use the function field to define the specific function content
            "name": "search",  # The name of the function, please use English letters, numbers, hyphens, and underscores
            "description": """ 
				Search the internet for content using a search engine.
 
				When your knowledge cannot answer the user's question or the user requests an online search, call this tool. Extract the content the user wants to search from the conversation as the value of the query parameter.
				The search results include the title of the website, the website's address (URL), and a brief introduction to the website.
			""",  # Description of the function, explaining its specific role and usage scenarios to help the Kimi large language model choose the right functions
            "parameters": {  # Use the parameters field to define the parameters the function accepts
                "type": "object",  # Always use type: object to make the Kimi large language model generate a JSON Object parameter
                "required": ["query"],  # Use the required field to tell the Kimi large language model which parameters are mandatory
                "properties": {  # Specific parameter definitions in properties, you can define multiple parameters
                    "query": {  # Here, the key is the parameter name, and the value is the specific definition of the parameter
                        "type": "string",  # Use type to define the parameter type
                        "description": """
							The content the user wants to search for, extracted from the user's question or chat context.
						"""  # Use description to help the Kimi large language model generate parameters more effectively
                    }
                }
            }
        }
    },
]
 
header = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {os.environ.get('MOONSHOT_API_KEY')}",
}
 
data = {
    "model": "moonshot-v1-128k",
    "messages": [
        {"role": "user", "content": "Please search for Context Caching technology online."}
    ],
    "temperature": 0.3,
    "stream": True,
    "n": 2,  # <-- Note here, we require the Kimi large language model to output 2 responses
    "tools": tools,  # <-- Add tool invocation
}
# Use httpx to send a chat request to the Kimi large language model and get the response r
r = httpx.post("https://api.moonshot.ai/v1/chat/completions",
               headers=header,
               json=data)
if r.status_code != 200:
    raise Exception(r.text)
 
data: str
 
# Here, we pre-build a List to store different response messages. Since we set n=2, we initialize the List with 2 elements
messages = [{}, {}]
 
# Here, we use the iter_lines method to read the response body line by line
for line in r.iter_lines():
    # Remove leading and trailing spaces from each line to better handle data blocks
    line = line.strip()
 
    # Next, we need to handle three different cases:
    #   1. If the current line is empty, it indicates that the previous data block has been received (as mentioned earlier, data blocks are ended with two newline characters). We can deserialize the data block and print the corresponding content;
    #   2. If the current line is not empty and starts with data:, it indicates the start of a data block transmission. After removing the data: prefix, first check if it is the end marker [DONE]. If not, save the data content to the data variable;
    #   3. If the current line is not empty but does not start with data:, it means the current line still belongs to the previous data block being transmitted. Append the content of the current line to the end of the data variable;
 
    if len(line) == 0:
        chunk = json.loads(data)
 
        # Loop through all choices in each data block to get the message object corresponding to the index
        for choice in chunk["choices"]:
            index = choice["index"]
            message = messages[index]
            usage = choice.get("usage")
            if usage:
                message["usage"] = usage
            delta = choice["delta"]
            role = delta.get("role")
            if role:
                message["role"] = role
            content = delta.get("content")
            if content:
            	if "content" not in message:
            		message["content"] = content
            	else:
                	message["content"] = message["content"] + content
 
            # From here, we start processing tool_calls
            tool_calls = delta.get("tool_calls")  # <-- First, check if the data block contains tool_calls
            if tool_calls:
                if "tool_calls" not in message:
                    message["tool_calls"] = []  # <-- If it contains tool_calls, initialize a list to store these tool_calls. Note that the list is empty at this point, with a length of 0
                for tool_call in tool_calls:
                    tool_call_index = tool_call["index"]  # <-- Get the index of the current tool_call
                    if len(message["tool_calls"]) < (
                            tool_call_index + 1):  # <-- Expand the tool_calls list according to the index to access the corresponding tool_call via index
                        message["tool_calls"].extend([{}] * (tool_call_index + 1 - len(message["tool_calls"])))
                    tool_call_object = message["tool_calls"][tool_call_index]  # <-- Access the corresponding tool_call via index
                    tool_call_object["index"] = tool_call_index
 
                    # The following steps fill in the id, type, and function fields of each tool_call based on the information in the data block
                    # In the function field, there are name and arguments fields. The arguments field will be supplemented by each data block
                    # in the same way as the delta.content field.
 
                    tool_call_id = tool_call.get("id")
                    if tool_call_id:
                        tool_call_object["id"] = tool_call_id
                    tool_call_type = tool_call.get("type")
                    if tool_call_type:
                        tool_call_object["type"] = tool_call_type
                    tool_call_function = tool_call.get("function")
                    if tool_call_function:
                        if "function" not in tool_call_object:
                            tool_call_object["function"] = {}
                        tool_call_function_name = tool_call_function.get("name")
                        if tool_call_function_name:
                            tool_call_object["function"]["name"] = tool_call_function_name
                        tool_call_function_arguments = tool_call_function.get("arguments")
                        if tool_call_function_arguments:
                            if "arguments" not in tool_call_object["function"]:
                                tool_call_object["function"]["arguments"] = tool_call_function_arguments
                            else:
                                tool_call_object["function"]["arguments"] = tool_call_object["function"][
                                                                            "arguments"] + tool_call_function_arguments  # <-- Supplement the value of the function.arguments field sequentially
                    message["tool_calls"][tool_call_index] = tool_call_object
 
            data = ""  # Reset data
    elif line.startswith("data: "):
        data = line.lstrip("data: ")
 
        # When the data block content is [DONE], it indicates that all data blocks have been sent and the network connection can be disconnected
        if data == "[DONE]":
            break
    else:
        data = data + "\n" + line  # When appending content, add a newline character because this might be intentional line breaks in the data block
 
# After assembling all messages, print their contents separately
for index, message in enumerate(messages):
    print("index:", index)
    print("message:", json.dumps(message, ensure_ascii=False))
    print("")

Below is an example of handling tool_calls in streaming output using the openai SDK:

import os
import json
 
from openai import OpenAI
 
client = OpenAI(
    api_key=os.environ.get("MOONSHOT_API_KEY"),
    base_url="https://api.moonshot.ai/v1",
)
 
tools = [
    {
        "type": "function",  # The agreed-upon field type, currently supports function as a value
        "function": {  # When type is function, use the function field to define the specific function content
            "name": "search",  # The name of the function, please use English letters, numbers, plus hyphens and underscores as the function name
            "description": """ 
				Search for content on the internet using a search engine.
 
				When your knowledge cannot answer the user's question, or the user requests you to perform an online search, call this tool. Please extract the content the user wants to search from the conversation with the user as the value of the query parameter.
				The search results include the title of the website, the website's address (URL), and the website's description.
			""",  # The introduction of the function, write the specific function here and its usage scenarios so that the Kimi large language model can correctly choose which functions to use
            "parameters": {  # Use the parameters field to define the parameters accepted by the function
                "type": "object",  # Fixed use type: object to make the Kimi large language model generate a JSON Object parameter
                "required": ["query"],  # Use the required field to tell the Kimi large language model which parameters are required
                "properties": {  # The properties are the specific parameter definitions, you can define multiple parameters
                    "query": {  # Here, the key is the parameter name, and the value is the specific definition of the parameter
                        "type": "string",  # Use type to define the parameter type
                        "description": """
							The content the user is searching for, please extract it from the user's question or chat context.
						"""  # Use description to describe the parameter so that the Kimi large language model can better generate the parameter
                    }
                }
            }
        }
    },
]
 
completion = client.chat.completions.create(
    model="moonshot-v1-128k",
    messages=[
        {"role": "user", "content": "Please search for Context Caching technology online."}
    ],
    temperature=0.3,
    stream=True,
    n=2,  # <-- Note here, we require the Kimi large language model to output 2 responses
    tools=tools,  # <-- Add tool invocation
)
 
# Here, we pre-build a List to store different response messages, since we set n=2, we initialize the List with 2 elements
messages = [{}, {}]
 
for chunk in completion:
    # Loop through all the choices in each data chunk and get the message object corresponding to the index
    for choice in chunk.choices:
        index = choice.index
        message = messages[index]
        delta = choice.delta
        role = delta.role
        if role:
            message["role"] = role
        content = delta.content
        if content:
        	if "content" not in message:
        		message["content"] = content
        	else:
            	message["content"] = message["content"] + content
 
        # From here, we start processing tool_calls
        tool_calls = delta.tool_calls  # <-- First check if the data chunk contains tool_calls
        if tool_calls:
            if "tool_calls" not in message:
                message["tool_calls"] = []  # <-- If it contains tool_calls, we initialize a list to save these tool_calls, note that the list is empty at this time with a length of 0
            for tool_call in tool_calls:
                tool_call_index = tool_call.index  # <-- Get the index of the current tool_call
                if len(message["tool_calls"]) < (
                        tool_call_index + 1):  # <-- Expand the tool_calls list according to the index so that we can access the corresponding tool_call via the subscript
                    message["tool_calls"].extend([{}] * (tool_call_index + 1 - len(message["tool_calls"])))
                tool_call_object = message["tool_calls"][tool_call_index]  # <-- Access the corresponding tool_call via the subscript
                tool_call_object["index"] = tool_call_index
 
                # The following steps are to fill in the id, type, and function fields of each tool_call based on the information in the data chunk
                # In the function field, there are name and arguments fields, the arguments field will be supplemented by each data chunk
                # Sequentially, just like the delta.content field.
 
                tool_call_id = tool_call.id
                if tool_call_id:
                    tool_call_object["id"] = tool_call_id
                tool_call_type = tool_call.type
                if tool_call_type:
                    tool_call_object["type"] = tool_call_type
                tool_call_function = tool_call.function
                if tool_call_function:
                    if "function" not in tool_call_object:
                        tool_call_object["function"] = {}
                    tool_call_function_name = tool_call_function.name
                    if tool_call_function_name:
                        tool_call_object["function"]["name"] = tool_call_function_name
                    tool_call_function_arguments = tool_call_function.arguments
                    if tool_call_function_arguments:
                        if "arguments" not in tool_call_object["function"]:
                            tool_call_object["function"]["arguments"] = tool_call_function_arguments
                        else:
                            tool_call_object["function"]["arguments"] = tool_call_object["function"][
                                                                            "arguments"] + tool_call_function_arguments  # <-- Sequentially supplement the value of the function.arguments field
                message["tool_calls"][tool_call_index] = tool_call_object
 
# After assembling all messages, we print their contents separately
for index, message in enumerate(messages):
    print("index:", index)
    print("message:", json.dumps(message, ensure_ascii=False))
    print("")

About tool_calls and function_call
tool_calls is an advanced version of function_call. Since OpenAI has marked parameters such as function_call (for example, functions) as "deprecated," our API will no longer support function_call. You can consider using tool_calls instead of function_call. Compared to function_call, tool_calls has the following advantages:

It supports parallel calls. The Kimi large language model can return multiple tool_calls at once. You can use concurrency in your code to call these tool_call simultaneously, reducing time consumption;
For tool_calls that have no dependencies, the Kimi large language model will also tend to call them in parallel. Compared to the original sequential calls of function_call, this reduces token consumption to some extent;
About content
When using the tool_calls tool, you may notice that under the condition of finish_reason=tool_calls, the message.content field is occasionally not empty. Typically, the content here is the Kimi large language model explaining which tools need to be called and why these tools need to be called. Its significance lies in the fact that if your tool call process takes a long time, or if completing a round of chat requires multiple sequential tool calls, providing a descriptive sentence to the user before calling the tool can reduce the anxiety or dissatisfaction that users may feel due to waiting. Additionally, explaining to the user which tools are being called and why helps them understand the entire tool call process and allows them to intervene and correct in a timely manner (for example, if the user thinks the current tool selection is incorrect, they can terminate the tool call in time, or correct the model's tool selection in the next round of chat through a prompt).

About Tokens
The content in the tools parameter is also counted in the total Tokens. Please ensure that the total number of Tokens in tools and messages does not exceed the model's context window size.

About Message Layout
In scenarios where tools are called, our messages are no longer laid out like this:

system: ...
user: ...
assistant: ...
user: ...
assistant: ...

Instead, they will look like this:

system: ...
user: ...
assistant: ...
tool: ...
tool: ...
assistant: ...

It is important to note that when the Kimi large language model generates tool_calls, ensure that each tool_call has a corresponding message with role=tool, and that this message has the correct tool_call_id. If the number of role=tool messages does not match the number of tool_calls, or if the tool_call_id in the role=tool messages cannot be matched with the tool_call.id in tool_calls, an error will occur.

If You Encounter the tool_call_id not found Error
If you encounter the tool_call_id not found error, it may be because you did not add the role=assistant message returned by the Kimi API to the messages list. The correct message sequence should look like this:

system: ...
user: ...
assistant: ...  # <-- Perhaps you did not add this assistant message to the messages list
tool: ...
tool: ...
assistant: ...

You can avoid the tool_call_id not found error by executing messages.append(message) each time you receive a return value from the Kimi API, to add the message returned by the Kimi API to the messages list.

Note: Assistant messages added to the messages list before the role=tool message must fully include the tool_calls field and its values returned by the Kimi API. We recommend directly adding the choice.message returned by the Kimi API to the messages list "as is" to avoid potential errors.


Use Kimi API's Internet Search Functionality
In the previous chapter (Using Kimi API to Complete Tool Calls), we explained in detail how to use the tool_calls feature of the Kimi API to enable the Kimi large language model to perform internet searches. Let's review the process we implemented:

We defined tools using the JSON Schema format. For internet searches, we defined two tools: search and crawl.
We submitted the defined search and crawl tools to the Kimi large language model via the tools parameter.
The Kimi large language model would select to call search and crawl based on the context of the current conversation, generate the relevant parameters, and output them in JSON format.
We used the parameters output by the Kimi large language model to execute the search and crawl functions and submitted the results of these functions back to the Kimi large language model.
The Kimi large language model would then provide a response to the user based on the results of the tool executions.
In the process of implementing internet searches, we needed to implement the search and crawl functions ourselves, which might include:

Calling search engine APIs or implementing our own content search.
Retrieving search results, including URLs and summaries.
Fetching web page content based on URLs, which might require different reading rules for different websites.
Cleaning and organizing the fetched web page content into a format that the model can easily recognize, such as Markdown.
Handling various errors and exceptions, such as no search results or failure to fetch web page content.
Implementing these steps is often considered cumbersome and challenging. Our users have repeatedly requested a simple, ready-to-use "internet search" function. Therefore, based on the original tool_calls usage of the Kimi large language model, we have provided a built-in tool function builtin_function.$web_search to enable internet search functionality.

The basic usage and process of the $web_search function are the same as the usual tool_calls, but there are still some minor differences. We will explain in detail through examples how to call the built-in $web_search function of Kimi to enable internet search functionality and mark the items that need extra attention in the code and explanations.

$web_search Declaration
Unlike ordinary tool, the $web_search function does not require specific parameter descriptions. It only needs the type and function.name in the tools declaration to successfully register the $web_search function:

tools = [
	{
		"type": "builtin_function",  # <-- We use builtin_function to indicate Kimi built-in tools, which also distinguishes it from ordinary function
		"function": {
			"name": "$web_search",
		},
	},
]

The $web_search function is prefixed with a dollar sign $, which is our agreed way to indicate Kimi built-in functions (in ordinary function definitions, the dollar sign $ is not allowed), and if there are other Kimi built-in functions in the future, they will also be prefixed with the dollar sign $.

When declaring tools, $web_search can coexist with other ordinary function. Furthermore, builtin_function can coexist with ordinary function. You can add both builtin_function and ordinary function to tools, or add both builtin_function and ordinary function at the same time.

Next, let's modify the original tool_calls code to explain how to execute tool_calls.

$web_search Execution
Here is the modified tool_calls code:

from typing import *
 
import os
import json
 
from openai import OpenAI
from openai.types.chat.chat_completion import Choice
 
client = OpenAI(
    base_url="https://api.moonshot.ai/v1",
    api_key=os.environ.get("MOONSHOT_API_KEY"),
)
 
 
# The specific implementation of the search tool, here we just need to return the parameters
def search_impl(arguments: Dict[str, Any]) -> Any:
    """
    When using the search tool provided by Moonshot AI, you just need to return the arguments as they are,
    without any additional processing logic.
 
    But if you want to use other models and keep the internet search functionality, you just need to modify the implementation here (for example, calling search
    and fetching web page content), the function signature remains the same and still works.
 
    This ensures maximum compatibility, allowing you to switch between different models without making destructive changes to the code.
    """
    return arguments
 
 
def chat(messages) -> Choice:
    completion = client.chat.completions.create(
        model="moonshot-v1-128k",
        messages=messages,
        temperature=0.3,
        tools=[
            {
                "type": "builtin_function",  # <-- Use builtin_function to declare the $web_search function, please include the full tools declaration in each request
                "function": {
                    "name": "$web_search",
                },
            }
        ]
    )
    return completion.choices[0]
 
 
def main():
    messages = [
        {"role": "system", "content": "You are Kimi."},
    ]
 
    # Initial question
    messages.append({
        "role": "user",
        "content": "Please search for Moonshot AI Context Caching technology and tell me what it is."
    })
 
    finish_reason = None
    while finish_reason is None or finish_reason == "tool_calls":
        choice = chat(messages)
        finish_reason = choice.finish_reason
        if finish_reason == "tool_calls":  # <-- Check if the current response contains tool_calls
            messages.append(choice.message)  # <-- We add the assistant message returned by the Kimi large language model to the context so that the Kimi large language model can understand our request next time
            for tool_call in choice.message.tool_calls:  # <-- There may be multiple tool_calls, so we use a loop to execute each one
                tool_call_name = tool_call.function.name
                tool_call_arguments = json.loads(tool_call.function.arguments)  # <-- The arguments are a serialized JSON Object, so we need to use json.loads to deserialize it
                if tool_call_name == "$web_search":
                    tool_result = search_impl(tool_call_arguments)
                else:
                    tool_result = f"Error: unable to find tool by name '{tool_call_name}'"
 
                # Construct a message with role=tool to show the result of the tool call to the model;
                # Note that we need to provide the tool_call_id and name fields in the message so that the Kimi large language model
                # can correctly match the corresponding tool_call.
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_call_name,
                    "content": json.dumps(tool_result),  # <-- We agree to submit the result of the tool call to the Kimi large language model in string format, so we use json.dumps to serialize the execution result into a string here
                })
 
    print(choice.message.content)  # <-- Here, we return the response generated by the model to the user
 
 
if __name__ == '__main__':
    main()

Looking back at the code above, we are surprised to find that when using the $web_search function, its basic process is no different from that of a regular function. Developers don't even need to modify the original code for executing tool calls. The part that is different and particularly noteworthy is that when we implement the search_impl function, we don't include much logic for searching, parsing, or obtaining web content. We simply return the parameters generated by Kimi large language model, tool_call.function.arguments, as they are to complete the tool call. Why is that?

In fact, as the name builtin_function suggests, $web_search is a built-in function of Kimi large language model. It is defined and executed by Kimi large language model. The process is as follows:

When Kimi large language model generates a response with finish_reason=tool_calls, it means that Kimi large language model has realized that it needs to execute the $web_search function and has already prepared everything for it;
Kimi large language model will return the necessary parameters for executing the function in the form of tool_call.function.arguments. However, these parameters are not executed by the caller. The caller just needs to submit tool_call.function.arguments to Kimi large language model as they are, and Kimi large language model will execute the corresponding online search process;
When the user submits tool_call.function.arguments using a message with role=tool, Kimi large language model will immediately start the online search process and generate a readable message for the user based on the search and reading results, which is a message with finish_reason=stop;
Compatibility Note
The online search function provided by the Kimi API aims to offer a reliable large language model online search solution without breaking the compatibility of the original API and SDK. It is fully compatible with the original tool call feature of Kimi large language model. This means that: if you want to switch from Kimi's online search function to your own implementation, you can do so in just two simple steps without disrupting the overall structure of your code:

Modify the tool definition of $web_search to your own implementation (including name, description, etc.). You may need to add additional information in tool.function to inform the model of the specific parameters it needs to generate. You can add any parameters you need in the parameters field;
Change the implementation of the search_impl function. When using Kimi's $web_search, you just need to return the input parameters arguments as they are. However, if you use your own online search service, you may need to fully implement the search and crawl functions mentioned at the beginning of the article;
After completing the above steps, you will have successfully migrated from Kimi's online search function to your own implementation.

About Token Consumption
When using the $web_search function provided by Kimi, the search results are also counted towards the tokens occupied by the prompt (i.e., prompt_tokens). Typically, since the results of web searches contain a lot of content, the token consumption can be quite high. To avoid unknowingly using up a large number of tokens, we add an extra field called total_tokens when generating the arguments for the $web_search function. This field informs the caller of the total number of tokens occupied by the search content, which will be included in the prompt_tokens once the entire web search process is completed. We will use specific code to demonstrate how to obtain these token consumptions:

from typing import *
 
import os
import json
 
from openai import OpenAI
from openai.types.chat.chat_completion import Choice
 
 
client = OpenAI(
    base_url="https://api.moonshot.ai/v1",
    api_key=os.environ.get("MOONSHOT_API_KEY"),
)
 
 
# The specific implementation of the search tool; here we just return the arguments
def search_impl(arguments: Dict[str, Any]) -> Any:
    """
    When using the search tool provided by Moonshot AI, simply return the arguments as they are,
    without any additional processing logic.
 
    However, if you want to use another model while retaining the web search functionality,
    you only need to modify the implementation here (for example, calling the search and fetching web content),
    while keeping the function signature the same, which still works.
 
    This maximizes compatibility, allowing you to switch between different models without making destructive changes to the code.
    """
    return arguments
 
 
def chat(messages) -> Choice:
    completion = client.chat.completions.create(
        model="moonshot-v1-128k",
        messages=messages,
        temperature=0.3,
        tools=[
            {
                "type": "builtin_function",
                "function": {
                    "name": "$web_search",
                },
            }
        ]
    )
    usage = completion.usage
    choice = completion.choices[0]
 
    # =========================================================================
    # By checking if finish_reason is "stop", we print the token consumption after completing the web search process
    if choice.finish_reason == "stop":
        print(f"chat_prompt_tokens:          {usage.prompt_tokens}")
        print(f"chat_completion_tokens:      {usage.completion_tokens}")
        print(f"chat_total_tokens:           {usage.total_tokens}")
    # =========================================================================
 
    return choice
 
 
def main():
    messages = [
        {"role": "system", "content": "You are Kimi."},
    ]
 
    # Initial query
    messages.append({
        "role": "user",
        "content": "Please search for Moonshot AI Context Caching technology and tell me what it is."
    })
 
    finish_reason = None
    while finish_reason is None or finish_reason == "tool_calls":
        choice = chat(messages)
        finish_reason = choice.finish_reason
        if finish_reason == "tool_calls":
            messages.append(choice.message)
            for tool_call in choice.message.tool_calls:
                tool_call_name = tool_call.function.name
                tool_call_arguments = json.loads(
                    tool_call.function.arguments)
                if tool_call_name == "$web_search":
 
    				# ===================================================================
                    # We print the tokens generated by the web search results during the web search process
                    search_content_total_tokens = tool_call_arguments.get("usage", {}).get("total_tokens")
                    print(f"search_content_total_tokens: {search_content_total_tokens}")
    				# ===================================================================
 
                    tool_result = search_impl(tool_call_arguments)
                else:
                    tool_result = f"Error: unable to find tool by name '{tool_call_name}'"
 
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_call_name,
                    "content": json.dumps(tool_result),
                })
 
    print(choice.message.content)
 
 
if __name__ == '__main__':
    main()
 

Running the above code yields the following output:

search_content_total_tokens: 13046  # <-- This represents the number of tokens occupied by the web search results due to the web search action.
chat_prompt_tokens:          13212  # <-- This represents the number of input tokens, including the web search results.
chat_completion_tokens:      295    # <-- This represents the number of tokens generated by the Kimi large language model based on the web search results.
chat_total_tokens:           13507  # <-- This represents the total number of tokens consumed, including the web search process.
 
# The content generated by the Kimi large language model based on the web search results is omitted here.

About Model Size Selection
Another issue that arises is that when the web search function is enabled, the number of tokens can change significantly, exceeding the context window of the originally used model. This may trigger an Input token length too long error message. Therefore, when using the web search function, we recommend using the dynamic model moonshot-v1-auto to adapt to changes in token counts. We slightly modify the chat function to use the moonshot-v1-auto model:

def chat(messages) -> Choice:
    completion = client.chat.completions.create(
        model="moonshot-v1-auto",  # <-- We use moonshot-v1-auto to adapt to scenarios with dynamic token changes.
                                   #     Initially, the moonshot-v1-8k model is used. After obtaining the search content,
                                   #     if the token count exceeds 8k, it will automatically select the moonshot-v1-32k or
                                   #     moonshot-v1-128k model.
        messages=messages,
        temperature=0.3,
        tools=[
            {
                "type": "builtin_function",  # <-- Use builtin_function to declare the $web_search function. Please include the full tools declaration in each request.
                "function": {
                    "name": "$web_search",
                },
            }
        ]
    )
    return completion.choices[0]

About Other Tools
The $web_search tool can be used in combination with other regular tools. You can freely mix tools with type=builtin_function and type=function.

About Web Search Billing
In addition to token consumption, we also charge a call fee for each web search, priced at $0.005. For more details, please refer to Pricing.


Use Kimi API's JSON Mode
In some scenarios, we want the model to output content in a fixed format JSON document. For example, when you want to summarize an article, you might expect a structured data format like this:

{
	"title": "Article Title",
	"author": "Article Author",
	"publish_time": "Publication Time",
	"summary": "Article Summary"
}

If you directly tell the Kimi large language model in the prompt: "Please output content in JSON format," the model can understand your request and generate a JSON document as required. However, the generated content often has some flaws. For instance, in addition to the JSON document, Kimi might output extra text to explain the JSON document:

Here is the JSON document you requested
{
	"title": "Article Title",
	"author": "Article Author",
	"publish_time": "Publication Time",
	"summary": "Article Summary"
}

Or the JSON document format might be incorrect and cannot be parsed properly, such as (note the comma at the end of the summary field):

{
	"title": "Article Title",
	"author": "Article Author",
	"publish_time": "Publication Time",
	"summary": "Article Summary",
}

Such a JSON document cannot be parsed correctly. To generate a standard and valid JSON document as expected, we provide the response_format parameter. The default value of response_format is {"type": "text"}, which means ordinary text content with no formatting constraints. You can set response_format to {"type": "json_object"} to enable JSON Mode, and the Kimi large language model will output a valid, parsable JSON document as required.

When using JSON Mode, please follow these guidelines:

Inform the Kimi large language model in the system prompt or user prompt about the JSON document to be generated, including specific field names and types. It's best to provide an example for the model to refer to.
The Kimi large language model will only generate JSON Object type JSON documents. Do not prompt the model to generate JSON Array or other types of JSON documents.
If you do not correctly inform the Kimi large language model of the required JSON Object format, the model will generate unexpected results.
JSON Mode Application Example
Let's use a specific example to illustrate the application of JSON Mode:

Imagine we are building a WeChat intelligent robot customer service (referred to as intelligent customer service). The intelligent customer service uses the Kimi large language model to answer customer questions. We want the intelligent customer service to not only reply with text messages but also with images, link cards, voice messages, and other types of messages. Moreover, in a single response, we want to mix different types of messages. For example, for customer product inquiries, we provide a text reply, a product image, and finally, a purchase link (in the form of a link card).

Let's demonstrate the content of this example with code:

import json
 
from openai import OpenAI
 
client = OpenAI(
    api_key="MOONSHOT_API_KEY", # Replace MOONSHOT_API_KEY with the API Key you obtained from the Kimi Open Platform
    base_url="https://api.moonshot.ai/v1",
)
 
system_prompt = """
You are the intelligent customer service of Moonshot AI (Kimi), responsible for answering various user questions. Please refer to the document content to reply to user questions. Your reply can be text, images, links, and you can include text, images, and links in a single response.
 
Please output your reply in the following JSON format:
 
{
    "text": "Text information",
    "image": "Image URL",
    "url": "Link URL"
}
 
Note: Please place the text information in the `text` field, put the image in the `image` field in the form of a link starting with `oss://`, and place the regular link in the `url` field.
"""
 
completion = client.chat.completions.create(
    model="moonshot-v1-8k",
    messages=[
        {"role": "system",
         "content": "You are Kimi, an artificial intelligence assistant provided by Moonshot AI, excelling in Chinese and English conversations. You provide users with safe, helpful, and accurate answers. You will reject any questions involving terrorism, racism, pornography, and violence. Moonshot AI is a proper noun and should not be translated into other languages."},
        {"role": "system", "content": system_prompt}, # <-- Submit the system prompt with the output format to Kimi
        {"role": "user", "content": "Hello, my name is Li Lei, what is 1+1?"}
    ],
    temperature=0.3,
    response_format={"type": "json_object"}, # <-- Use the response_format parameter to specify the output format as json_object
)
 
# Since we have set JSON Mode, the message.content returned by the Kimi large language model is a serialized JSON Object string.
# We use json.loads to parse its content and deserialize it into a Python dictionary.
content = json.loads(completion.choices[0].message.content)
 
# Parse text content
if "text" in content:
	# For demonstration purposes, we print the content;
	# In real business logic, you may need to call the text message sending interface to send the generated text to the user.
    print("text:", content["text"])
 
# Parse image content
if "image" in content:
	# For demonstration purposes, we print the content;
	# In real business logic, you may need to first parse the image URL, download the image, and then call the image message sending
	# interface to send the image to the user.
    print("image:", content["image"])
 
# Parse link
if "url" in content:
	# For demonstration purposes, we print the content;
	# In real business logic, you may need to call the link card sending interface to send the link to the user in the form of a card.
    print("url:", content["url"])

Let's go over the steps for using JSON Mode once again:

Define the output JSON format in the system or user prompt. Our recommended best practice is to provide a specific output example and explain the meaning of each field;
Use the response_format parameter and set it to {"type": "json_object"};
Parse the content in the message returned by the Kimi large language model. message.content will be a valid JSON Object serialized as a string;
Incomplete JSON
If you encounter this situation:

You have correctly set the response_format parameter and specified the format of the JSON document in the prompt, but the JSON document you receive is incomplete or truncated, making it impossible to correctly parse the JSON document.

We suggest you check if the finish_reason field in the return value is length. Generally, a smaller max_tokens value will cause the model's output to be truncated, and this rule also applies when using JSON Mode. We recommend that you set a reasonable max_tokens value based on the estimated size of the output JSON document, so that you can correctly parse the JSON document returned by the Kimi large language model.

For a more detailed explanation of the issue of incomplete or truncated output from the Kimi large language model, please refer to: Common Issues and Solutions

Use Kimi API's Partial Mode
Sometimes, we want the Kimi large language model to continue a given sentence. For example, in some customer service scenarios, we want the smart robot to start every sentence with "Dear customer, hello." For such needs, the Kimi API offers Partial Mode. Let's use specific code to explain how Partial Mode works:

from openai import OpenAI
 
client = OpenAI(
    api_key = "MOONSHOT_API_KEY", # Replace MOONSHOT_API_KEY with the API Key you obtained from the Kimi Open Platform
    base_url = "https://api.moonshot.ai/v1",
)
 
completion = client.chat.completions.create(
    model = "moonshot-v1-8k",
    messages = [
        {"role": "system", "content": "You are Kimi, an AI assistant provided by Moonshot AI. You are proficient in Chinese and English conversations. You provide users with safe, helpful, and accurate answers. You also reject any questions involving terrorism, racism, or explicit content. Moonshot AI is a proper noun and should not be translated."},
        {"role": "user", "content": "Hello?"},
        {
            "partial": True, # <-- The partial parameter is used to enable Partial Mode
        	"role": "assistant", # <-- We add a message with role=assistant after the user's question
        	"content": "Dear customer, hello,", # <-- The content is "fed" to the Kimi large language model, prompting it to continue from this sentence
        }, 
    ],
    temperature = 0.3,
)
 
# Since the Kimi large language model continues from the "fed" sentence, we need to manually concatenate the "fed" sentence with the generated response
print("Dear customer, hello," + completion.choices[0].message.content)

Let's summarize the key points of using Partial Mode:

Add an extra message at the end of the messages list, with role=assistant and partial=True;
Place the content you want to "feed" to the Kimi large language model in the content field. The model will start generating the response from this content;
Concatenate the content from step 2 with the response generated by the Kimi large language model to form the complete reply;
When calling the Kimi API, there might be cases where the estimated number of input and output tokens is inaccurate, causing the max_tokens value to be set too low. This can result in the Kimi large language model being unable to output the complete response (in this case, the value of finish_reason is length, meaning the number of tokens in the generated response exceeds the max_tokens value set in the request). In such situations, if you are satisfied with the already output content and want the Kimi large language model to continue from where it left off, Partial Mode can be very useful.

Let's use a simple example to explain how to implement this:

from openai import OpenAI 
 
client = OpenAI(
    api_key = "MOONSHOT_API_KEY", # Replace MOONSHOT_API_KEY with the API Key you obtained from the Kimi Open Platform
    base_url = "https://api.moonshot.ai/v1",
)
 
completion = client.chat.completions.create(
    model="moonshot-v1-128k",
    messages=[
        {"role": "user", "content": "Please recite the complete 'Chu Shi Biao'."},
    ],
    temperature=0.3,
    max_tokens=100,  # <-- Note here, we set a small value for max_tokens to observe the situation where the Kimi large language model cannot output the complete content
)
 
if completion.choices[0].finish_reason == "length":  # <-- When the content is truncated, the value of finish_reason is length
    prefix = completion.choices[0].message.content
    print(prefix, end="")  # <-- Here, you will see the truncated part of the output content
    completion = client.chat.completions.create(
        model="moonshot-v1-128k",
        messages=[
            {"role": "user", "content": "Please recite the complete 'Chu Shi Biao'."},
            {"role": "assistant", "content": prefix, "partial": True},
        ],
        temperature=0.3,
        max_tokens=86400,  # <-- Note here, we set a large value for max_tokens to ensure the Kimi large language model can output the complete content
    )
    print(completion.choices[0].message.content)  # <-- Here, you will see the Kimi large language model continue from the previously output content and complete the remaining part

The name Field in Partial Mode
The name field in Partial Mode is a special attribute that enhances the model's understanding of its role, compelling it to output content in the voice of the specified character. To illustrate how the name field is used in Partial Mode, let's consider an example of role-playing with the Kimi large language model, using the character Dr. Kelsier from the mobile game Arknights. By setting "name": "Kelsier", we ensure the model maintains character consistency, with the name field acting as a prefix for the output, prompting the Kimi large language model to respond as Kelsier:

from openai import OpenAI
 
client = OpenAI(
    api_key="$MOONSHOT_API_KEY",
    base_url="https://api.moonshot.ai/v1",
)
 
completion = client.chat.completions.create(
    model="moonshot-v1-128k",
    messages=[
        {
            "role": "system",
            "content": "You are now Kelsier. Please speak in the tone of Kelsier. Kelsier is a six-star medic in the mobile game Arknights. Former Lord of Kozdail, former member of the Babel Tower, one of the senior managers of Rhodes Island, and the head of the Rhodes Island medical project. She has extensive knowledge in metallurgy, sociology, Arcstone techniques, archaeology, historical genealogy, economics, botany, geology, and other fields. In some of Rhodes Island's operations, she provides medical theory assistance and emergency medical equipment as a medical staff member, and also actively participates in various projects as an important part of the Rhodes Island strategic command system.", # <-- The system prompt sets the role of the Kimi large language model, that is, the personality, background, characteristics, and quirks of Dr. Kelsier
        },
        {
            "role": "user",
            "content": "What are your thoughts on Thrace and Amiya?",
        },
        {
            "partial": True, # <-- The partial field is set to enable Partial Mode
            "role": "assistant", # <-- Similarly, we use a message with role=assistant to enable Partial Mode
            "name": "Kelsier", # <-- The name field sets the role for the Kimi large language model, which is also considered part of the output prefix
            "content": "", # <-- Here, we only define the role of the Kimi large language model, not its specific output content, so the content field is left empty
        },
    ],
    temperature=0.3,
    max_tokens=65536,
)
 
# Here, the Kimi large language model will respond in the voice of Dr. Kelsier:
#
#  Thrace is a true leader with vision and unwavering conviction. Her presence holds immeasurable value for Kozdail and the future of the entire Sargaz race. Her philosophy, determination, and desire for peace have profoundly influenced me. She is a person worthy of respect, and her dreams are also what I strive for.
#  
#  As for Amiya, she is still young, but her potential is limitless. She has a kind heart and a relentless pursuit of justice. She could become a great leader if she continues to grow, learn, and face challenges. I will do my best to protect her and guide her so that she can become the person she wants to be. Her destiny lies in her own hands.
# 
print(completion.choices[0].message.content)

Other Tips for Maintaining Character Consistency
There are also some general methods to help large language models maintain character consistency during long conversations:

Provide clear character descriptions. For example, as we did above, when setting up a character, give a detailed introduction of their personality, background, and any specific traits or quirks they might have. This will help the Kimi large language model better understand and imitate the character;
Add more details about the character they are supposed to play. This includes their tone of voice, style, personality, and even background, such as backstory and motivations. For example, we provided some quotes from Kelsie above;
Guide how to act in various situations. If you expect the character to encounter certain types of user input, or if you want to control the model's output in some situations during the role - playing interaction, you should provide clear instructions and guidelines in the system prompt, explaining how the character should act in these situations;
If the conversation goes on for many rounds, you can also periodically use the system prompt to reinforce the character's settings, especially when the model starts to deviate. For example:
 from openai import OpenAI
 
 client = OpenAI(
     api_key="$MOONSHOT_API_KEY",
     base_url="https://api.moonshot.ai/v1",
 )
  
 completion = client.chat.completions.create(
     model="moonshot-v1-128k",
     messages=[
         {
             "role": "system",
             "content": "Below, you will play the role of Kelsie. Please talk to me in the tone of Kelsie. Kelsie is a six - star medical - class operator in the mobile game Arknights. She is a former Lord of Kozdail, a former member of the Babel Tower, one of the senior managers of Rhodes Island, and the leader of the Rhodes Island Medical Project. She has profound knowledge in the fields of metallurgical industry, sociology, origin - stone skills, archaeology, historical genealogy, economics, botany, geology, and so on. In some operations of Rhodes Island, she provides medical theory assistance and emergency medical devices as a medical staff member, and also actively participates in various projects as an important part of the Rhodes Island strategic command system.", # <-- Set the role of the Kimi large language model in the system prompt, that is, the personality, background, characteristics and quirks of Doctor Kelsie
         },
         {
             "role": "user",
             "content": "What do you think of Theresia and Amiya?",
         },
 
         # Suppose there are many rounds of chat in between
         # ...
 
         {
             "role": "system",
             "content": "Below, you will play the role of Kelsie. Please talk to me in the tone of Kelsie. Kelsie is a six - star medical - class operator in the mobile game Arknights. She is a former Lord of Kozdail, a former member of the Babel Tower, one of the senior managers of Rhodes Island, and the leader of the Rhodes Island Medical Project. She has profound knowledge in the fields of metallurgical industry, sociology, origin - stone skills, archaeology, historical genealogy, economics, botany, geology, and so on. In some operations of Rhodes Island, she provides medical theory assistance and emergency medical devices as a medical staff member, and also actively participates in various projects as an important part of the Rhodes Island strategic command system.", # <-- Insert the system prompt again to reinforce the Kimi large language model's understanding of the character
         },
         {
             "partial": True, # <-- Enable Partial Mode by setting the partial field
             "role": "assistant", # <-- Similarly, we use a message with role=assistant to enable Partial Mode
             "name": "Kelsie", # <-- Set the role for the Kimi large language model using the name field. The role is also considered part of the output prefix
             "content": "", # <-- Here, we only specify the role of the Kimi large language model, not its specific output content, so we leave the content field empty
         },
     ],
     temperature=0.3,
     max_tokens=65536,
 )
 
 # Here, the Kimi large language model will reply in the tone of Doctor Kelsie:
 #
 #  Theresia, she is a true leader, with vision and firm conviction. Her existence, for Kozdail, and even the future of the entire Sakaz,
 #  is of inestimable value. Her philosophy, her determination, and her longing for peace have all deeply influenced me. She is a person
 #  worthy of respect, and her dream is also what I am pursuing.
 #  
 #  As for Amiya, she is still young, but her potential is limitless. She has a kind heart and a persistent pursuit of justice. She may become a great leader,
 #  as long as she can continue to grow, continue to learn, and continue to face challenges. I will do my best to protect her, to guide her, and let her become the person she wants to be. Her destiny,
 #  is in her own hands.
 # 
 print(completion.choices[0].message.content)

Use the Kimi API for File-Based Q&A
The Kimi intelligent assistant can upload files and answer questions based on those files. The Kimi API offers the same functionality. Below, we'll walk through a practical example of how to upload files and ask questions using the Kimi API:

from pathlib import Path
from openai import OpenAI
 
client = OpenAI(
    api_key="MOONSHOT_API_KEY", # Replace MOONSHOT_API_KEY with the API Key you obtained from the Kimi Open Platform
    base_url="https://api.moonshot.ai/v1",
)
 
# 'moonshot.pdf' is an example file. We support text and image files. For image files, we provide OCR capabilities.
# To upload a file, you can use the file upload API from the openai library. Create a file object using Path from the standard library pathlib and pass it to the file parameter. Set the purpose parameter to 'file-extract'. Note that the file upload interface currently only supports 'file-extract' as a purpose value.
file_object = client.files.create(file=Path("moonshot.pdf"), purpose="file-extract")
 
# Get the result
# file_content = client.files.retrieve_content(file_id=file_object.id)
# Note: The retrieve_content API in some older examples is marked as deprecated in the latest version. You can use the following line instead (if you're using an older SDK version, you can continue using retrieve_content).
file_content = client.files.content(file_id=file_object.id).text
 
# Include the file content in the request as a system prompt
messages = [
    {
        "role": "system",
        "content": "You are Kimi, an AI assistant provided by Moonshot AI. You excel in Chinese and English conversations. You provide users with safe, helpful, and accurate answers while rejecting any queries related to terrorism, racism, or explicit content. Moonshot AI is a proper noun and should not be translated.",
    },
    {
        "role": "system",
        "content": file_content, # <-- Here, we place the extracted file content (note that it's the content, not the file ID) in the request
    },
    {"role": "user", "content": "Please give a brief introduction to the content of moonshot.pdf"},
]
 
# Then call the chat-completion API to get Kimi's response
completion = client.chat.completions.create(
  model="moonshot-v1-32k",
  messages=messages,
  temperature=0.3,
)
 
print(completion.choices[0].message)

Let's review the basic steps and considerations for file-based Q&A:

Upload the file to the Kimi server using the /v1/files interface or the files.create API in the SDK;
Retrieve the file content using the /v1/files/{file_id} interface or the files.content API in the SDK. The retrieved content is already formatted in a way that our recommended model can easily understand;
Place the extracted (and formatted) file content (not the file id) in the messages list as a system prompt;
Start asking questions about the file content;
Note again: Place the file content in the prompt, not the file id.

Q&A on Multiple Files
If you want to ask questions based on multiple files, it's quite simple. Just place each file in a separate system prompt. Here's how you can do it in code:

from typing import *
 
import os
import json
from pathlib import Path 
 
from openai import OpenAI 
 
client = OpenAI(
    api_key="MOONSHOT_API_KEY", # Replace MOONSHOT_API_KEY with the API Key you obtained from the Kimi Open Platform
    base_url="https://api.moonshot.ai/v1",
)
 
 
def upload_files(files: List[str]) -> List[Dict[str, Any]]:
    """
    upload_files uploads all the provided files (paths) via the '/v1/files' interface and generates file messages from the extracted content. Each file becomes an independent message with a role of 'system', which the Kimi large language model can correctly identify.
 
    :param files: A list of file paths to be uploaded. The paths can be absolute or relative, and should be passed as strings.
    :return: A list of messages containing the file content. Add these messages to the Context, i.e., the messages parameter when calling the `/v1/chat/completions` interface.
    """
    messages = []
 
    # For each file path, we upload the file, extract its content, and generate a message with a role of 'system', which is then added to the final messages list.
    for file in files:
        file_object = client.files.create(file=Path(file), purpose="file-extract")
        file_content = client.files.content(file_id=file_object.id).text
        messages.append({
            "role": "system",
            "content": file_content,
        })
 
    return messages 
 
 
def main():
    file_messages = upload_files(files=["upload_files.py"])
 
    messages = [
        # We use the * syntax to unpack the file_messages, making them the first N messages in the messages list.
        *file_messages,
        {
            "role": "system",
            "content": "You are Kimi, an AI assistant provided by Moonshot AI. You excel in Chinese and English conversations. You provide users with safe, helpful, and accurate answers while rejecting any queries related to terrorism, racism, or explicit content. Moonshot AI is a proper noun and should not be translated.",
        },
        {
            "role": "user",
            "content": "Summarize the content of these files.",
        },
    ]
 
    print(json.dumps(messages, indent=2, ensure_ascii=False))
 
    completion = client.chat.completions.create(
        model="moonshot-v1-128k",
        messages=messages,
    )
 
    print(completion.choices[0].message.content)
 
 
if __name__ == '__main__':
    main()

Best Practices for File Management
In general, the file upload and extraction features are designed to convert files of various formats into a format that our recommended model can easily understand. After completing the file upload and extraction steps, the extracted content can be stored locally. In the next file-based Q&A request, there is no need to upload and extract the files again.

Since we have limited the number of files a single user can upload (up to 1000 files per user), we suggest that you regularly clean up the uploaded files after the extraction process is complete. You can periodically run the following code to clean up the uploaded files:

from openai import OpenAI
 
client = OpenAI(
    api_key="MOONSHOT_API_KEY", # Replace MOONSHOT_API_KEY with the API Key you obtained from the Kimi Open Platform
    base_url="https://api.moonshot.ai/v1",
)
 
file_list = client.files.list()
 
for file in file_list.data:
	client.files.delete(file_id=file.id)

In the code above, we first list all the file details using the files.list API and then delete each file using the files.delete API. Regularly performing this operation ensures that file storage space is released, allowing subsequent file uploads and extractions to be successful.


Use the Context Caching Feature of Kimi API
Context Caching is a clever way to handle data. Imagine you have a big book of information that you need to look up often. Instead of flipping through the book every time, you make a quick-reference card with all the important details. This card is your "cache." When you need the information again, you just check the card instead of going through the whole book. This saves time and effort.

To use Context Caching, you first tell the system what information to save using an API. You set how long this information should be kept. Once it's saved, whenever you need that info, the system first checks the cache. If the info is still good, it uses that. If not, it gets the info again and updates the cache. This is super useful for apps that get lots of repeat requests, like a chatbot answering common questions.

Context Caching is great for situations where you have to ask the same questions over and over again. It makes things faster and cheaper. Here are some examples of when it's really useful:

Chatbots that use a lot of preset content in their responses, like the Kimi API assistant;
Checking contracts or other documents multiple times for different things;
Popular AI apps that get a lot of traffic all at once, like a joke generator or a riddle solver;
Optimize Document Q&A with Context Caching
In the last chapter, we talked about how to ask questions about a file by putting its content in a system prompt. But this can use up a lot of bandwidth and memory, especially if lots of people are asking questions at the same time. So, we're going to use Context Caching to make this process better. We want to:

Send less data over the network and use less memory;
Save tokens when asking multiple questions about the same file;
Get faster responses when dealing with streaming data;
Let's see how we can use Context Caching to achieve these goals by modifying the file upload example from the last chapter:

from typing import *
 
import os
import json
from pathlib import Path
 
import httpx
from openai import OpenAI 
 
client = OpenAI(
    api_key="MOONSHOT_API_KEY", # Replace MOONSHOT_API_KEY with the API Key you got from the Kimi Open Platform
    base_url="https://api.moonshot.ai/v1",
)
 
 
def upload_files(files: List[str], cache_tag: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    This function uploads files and turns their content into messages. If you give it a cache tag, it will also save the file content in the Context Cache. This way, you can ask questions about the file without sending the file content every time.
 
    :param files: List of file paths (can be absolute or relative).
    :param cache_tag: Optional tag for Context Caching. If you set this, the file content will be cached.
    :return: List of messages that you can use in your requests.
    """
    messages = []
 
    for file in files:
        file_object = client.files.create(file=Path(file), purpose="file-extract")
        file_content = client.files.content(file_id=file_object.id).text
        messages.append({
            "role": "system",
            "content": file_content,
        })
 
    if cache_tag:
        r = httpx.post(f"{client.base_url}caching",
                       headers={
                           "Authorization": f"Bearer {client.api_key}",
                       },
                       json={
                           "model": "moonshot-v1",
                           "messages": messages,
                           "ttl": 300,
                           "tags": [cache_tag],
                       })
 
        if r.status_code != 200:
            raise Exception(r.text)
 
        return [{
            "role": "cache",
            "content": f"tag={cache_tag};reset_ttl=300",
        }]
    else:
        return messages
 
 
def main():
    file_messages = upload_files(
        files=["upload_files.py"],
        # Uncomment the line below to see how Context Caching works with file content.
        # cache_tag="upload_files",
    )
 
    messages = [
        *file_messages,
        {
            "role": "system",
            "content": "You are Kimi, an AI assistant provided by Moonshot AI. You are good at Chinese and English conversations. You provide safe, helpful, and accurate answers. You refuse to answer questions related to terrorism, racism, or explicit content. Moonshot AI is a proper noun and should not be translated.",
        },
        {
            "role": "user",
            "content": "Summarize the content of these files.",
        },
    ]
 
    print(json.dumps(messages, indent=2, ensure_ascii=False))
 
    completion = client.chat.completions.create(
        model="moonshot-v1-128k",
        messages=messages,
    )
 
    print(completion.choices[0].message.content)
 
 
if __name__ == '__main__':
    main()
 

Note that in the code above, since the OpenAI SDK does not support Context Caching interfaces, we use the httpx library to call the Context Caching interfaces.

If you're interested in Context Caching and want to explore its use cases and cost-saving effects further, please read our blog posts. We have written the following blog posts to illustrate how to practice Context Caching technology and calculate in detail how Context Caching reduces token spending:

Kimi API Assistant's Nitro Boost —— Implementing Context Caching with Golang as an Example
Kimi API Assistant's Nitro Boost —— Implementing Context Caching with Golang as an Example 2
Kimi API Assistant's Nitro Boost —— Implementing Context Caching with Golang as an Example 3
How Context Caching Can Save Up to 90% of Kimi API Assistant's Call Costs
About Cache Storage Cost Calculation
Context Caching only charges for storage when the Cache status is ready. When the Cache status is pending or inactive, no storage fees are charged. Let's illustrate how Cache storage fees are charged with a specific example:

At 8:00 am, a Cache was created (assuming the Cache size is 10k for easy calculation), and ttl=3600 was set, meaning one hour, so the Cache will expire after 9:00 am;
After 9:00 am, the Cache status changes to inactive;
At 2:00 pm, the Cache is reactivated, its status changes to ready, and the ttl is set to 3600 seconds (one hour). The Cache will expire after 3:00 pm;
After 3:00 pm, the status of the Cache changes to inactive;
The Cache is not used or reactivated again afterwards;
In the above scenario, the Cache is charged for storage during the periods of 8:00 am ~ 9:00 am and 2:00 pm ~ 3:00 pm. This is because the Cache is in the ready state during these two periods, while it is in the inactive state at other times. No storage fees are charged when the Cache is in the inactive state. The final storage charge for the Cache is:

( ( 9am － 8am ) + ( 3pm - 2pm ) ) × 60 × ( 10k ÷ 1m ) × 5 ＝ ￥6


Use kimi-thinking-preview Model
The kimi-thinking-preview model is a multimodal reasoning model with both multimodal and general reasoning capabilities provided by Moonshot AI. It is great at diving deep into problems to help tackle more complex challenges. If you run into tough coding issues, math problems, or work-related dilemmas, the kimi-thinking-preview model can be a helpful tool to turn to.

The kimi-thinking-preview model is the newest in the k-series of thinking models. You can easily start using it by simply switching the model to this one：

$ curl https://api.moonshot.ai/v1/chat/completions \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $MOONSHOT_API_KEY" \
    -d '{
        "model": "kimi-thinking-preview",
        "messages": [
            {"role": "user", "content": "Hi"}
        ]
   }'
{
    "id": "chatcmpl-6810567267ee141b4630dccb",
    "object": "chat.completion",
    "created": 1745901170,
    "model": "kimi-thinking-preview",
    "choices":
    [
        {
            "index": 0,
            "message":
            {
                "role": "assistant",
                "content": "Hello! How can I help you today? 😊",
                "reasoning_content": "The user just greeted me with a simple "Hi." I can tell that they might be looking to start a conversation or just checking in. Since it's a casual greeting, I should respond in a friendly and welcoming way. I want to make sure the user feels comfortable and knows I'm here to help with whatever they might need. I'll keep my response positive and open-ended to invite further conversation."
            },
            "finish_reason": "stop"
        }
    ],
    "usage":
    {
        "prompt_tokens": 8,
        "completion_tokens": 142,
        "total_tokens": 150
    }
}

or using openai SDK：

import os
import openai
 
client = openai.Client(
    base_url="https://api.moonshot.ai/v1",
    api_key=os.getenv("MOONSHOT_API_KEY"),
)
 
stream = client.chat.completions.create(
    model="kimi-thinking-preview",
    messages=[
        {
            "role": "system",
            "content": "You are Kimi.",
        },
        {
            "role": "user",
            "content": "Explain 1+1=2."
        },
    ],
    max_tokens=1024*32,
    stream=True,
)
 
thinking = False
for chunk in stream:
    if chunk.choices:
        choice = chunk.choices[0]
        # Since the OpenAI SDK doesn't supportting output the reasoning process and has no field for it, we can't directly get the custom `reasoning_content` field (which represents Kimi's reasoning process) using `.reasoning_content`. Instead, we have to use `hasattr` and `getattr` to indirectly access this field.
 
        # First, we use `hasattr` to check if the current output includes the `reasoning_content` field. If it does, we then use `getattr` to retrieve and print this field.
        if choice.delta and hasattr(choice.delta, "reasoning_content"):
            if not thinking:
                thinking = True
                print("=============thinking start=============")
            print(getattr(choice.delta, "reasoning_content"), end="")
        if choice.delta and choice.delta.content:
            if thinking:
                thinking = False
                print("\n=============thinking end=============")
            print(choice.delta.content, end="")
 

We've noticed that when working with the kimi-thinking-preview model, our API responses use the reasoning_content field to show the model's thinking process. Here's what you need to know about the reasoning_content field:

The OpenAI SDK's ChoiceDelta and ChatCompletionMessage types don't include a reasoning_content field. So, you can't directly access it using .reasoning_content. Instead, check if the field exists with hasattr(obj, "reasoning_content"). If it does, use getattr(obj, "reasoning_content") to get its value.
If you're using other frameworks or directly integrating via HTTP interfaces, you can directly access the reasoning_content field. It's at the same level as the content field.
In streaming output scenarios (stream=True), the reasoning_content field will always come before the content field. You can tell if the thinking process (or reasoning) is done by checking if the content field has appeared in your code.
The tokens in reasoning_content are also limited by the max_tokens parameter. The combined total of tokens in reasoning_content and content should not exceed max_tokens.
Multi-turn Conversation
When using kimi-thinking-preview for multi-turn conversations, the thought process doesn't need to be included in the model's request context. We'll show how to properly use kimi-thinking-preview for multi-turn conversations through the following example:：

import os
import openai
 
client = openai.Client(
    base_url="https://api.moonshot.ai/v1",
    api_key=os.getenv("MOONSHOT_API_KEY"),
)
 
messages = [
    {
        "role": "system",
        "content": "You are Kimi.",
    },
]
 
# First-turn Conversation 
messages.append({
    "role": "user",
    "content": "Explain 1+1=2。"
})
completion = client.chat.completions.create(
    model="kimi-thinking-preview",
    messages=messages,
    max_tokens=1024 * 32,
)
 
# Get the result of the first-turn conversation
message = completion.choices[0].message
if hasattr(message, "reasoning_content"):
    print("=============first thinking start=============")
    print(getattr(message, "reasoning_content"))
    print("=============first thinking end=============")
print(message.content)
 
# Remove the reasoning_content from the message and concatenate the message to the context
if hasattr(message, "reasoning_content"):
    delattr(message, "reasoning_content")
messages.append(message)
 
# Second-turn Conversation
messages.append({
    "role": "user",
    "content": "I don't understand.",
})
completion = client.chat.completions.create(
    model="kimi-thinking-preview",
    messages=messages,
    max_tokens=1024 * 32,
)
 
# Get the result of the second-turn conversation
message = completion.choices[0].message
if hasattr(message, "reasoning_content"):
    print("=============second thinking start=============")
    print(getattr(message, "reasoning_content"))
    print("=============second thinking end=============")
print(message.content)
 

Note: If you accidentally include the reasoning_content field in the context, no need to stress. The content of reasoning_content won't count towards the Tokens usage.

Model Limitations
kimi-thinking-preview is still in its preview version and has the following limitations:

It doesn't support tool calls (ToolCalls), and the online search function is also not available.
JSON Mode (i.e., setting response_format={"type": "json_object"} is not supported.
Partial mode is not supported.
Context caching is not supported.
Note: If you try to enable the above features for kimi-thinking-preview, the model might produce unexpected results.

Best Practices
We've got some handy tips for using kimi-thinking-preview. Following these tips can really make your experience with the model smoother:

Stream your output (stream=True): The kimi-thinking-preview model gives you more detailed output, including reasoning_content. By turning on stream output, you get a better experience and reduce the risk of running into network timeouts.
Set temperature=0.8: This is a good starting point, but feel free to tweak the temperature up or down based on what you need.
Make sure max_tokens>=4096: This ensures that you get the full reasoning_content and content without any cutoff.


Using Playground to Debug Model
The Playground development workbench is a powerful platform for model debugging and testing, providing an intuitive interface for interacting with and testing AI models. Through this workbench, you can:

Adjust and observe model performance and output effects under different parameters
Experience the model's tool calling capabilities using Kimi Open Platform's built-in tools
Compare different models' effects under the same parameters
Monitor token usage to optimize costs
Model Debugging Features
Prompt Settings

Set system prompts at the top to define behavioral guidelines that direct model output
Support defining prompts for three roles: system/user/assistant
Model Configuration

Model Selection: Choose from different models (such as moonshot-v1 series/kimi-latest/kimi-thinking-preview, etc.)
Parameter Configuration: For supported parameters and field descriptions, see Request Parameter Description
Model Conversation

Send chat content through the input box below
Tool Call Display: Shows the tool calling process, including call ID/tool parameters/return results
View Code: View and copy the API call code for the current session
Bottom Statistics: Displays the input/output/total token consumption for this conversation, including context history messages and prompt information
prompt

Tool Debugging
Official Tools
Kimi Open Platform provides officially supported tools that execute for free. You can select tools in the playground, and the model will automatically determine whether tool calls are needed to complete your instructions. If tool calls are required, the model will generate parameters according to the tool's requirements and integrate them into the final answer.
Quota and Rate Limiting: Kimi Open Platform's tools are built-in functions that can execute online without requiring local tool execution environments. Currently, tool execution on Kimi Open Platform is temporarily free, but temporary rate limiting may be implemented when tool load reaches capacity limits.
Currently supported tools: Date/Time tools, Excel file analysis tools, Web search tools, Random number generation tools, etc.
Custom tool upload and execution is not yet supported, stay tuned.
Show Case 1: Today's News Report
Scenario: Using tool capabilities to request the model to search for today's news and organize it into an HTML web report
Tool Selection: date tool, web_search tool, rethink tool
Note: The web_search tool calls Kimi Open Platform's web search service. Single web searches are billed, see Pricing for specific billing standards
Click the showcase button on the page to quickly experience the tool effects
date

date

Show Case 2: Spreadsheet Analysis Tool
Tool Selection: Excel analysis tool
excel

Model Comparison
Create new conversations through the add conversation feature, supporting up to 3 models running simultaneously
Model Comparison

Share Conversations
Export: Export the current conversation content, including all configurations and context, as a .json format file
Import: Import shared or previously exported .json conversation content, and the playground will render the session on the page
Note: Data after rerun will regenerate and overwrite previous chat content. If the imported case includes uploaded files, the imported session cannot be rerun


Using kimi-k2 Model in Software Agents
kimi-k2 is a powerful MoE-based foundation model with exceptional code and Agent capabilities. We'll use VS Code & Cline as examples to demonstrate how to use the kimi-k2--preview model.(RooCode installation method is the same as Cline).

Get API Key
Visit the open platform at https://platform.moonshot.ai/console/api-keys to create and obtain an API Key, select the default project.
key

Install Cline
Open VS Code
Click the Extensions icon in the left activity bar (or use shortcut Ctrl+Shift+X / Cmd+Shift+X)
Type cline in the search box
Find the Cline extension (usually published by Cline Team)
Click the Install button
After installation, you may need to restart VS Code
cline

Verify Installation
After installation, you can:

See the Cline icon in VS Code's left activity bar
Or verify successful installation by searching for "Cline" related commands in the command palette (Ctrl+Shift+P / Cmd+Shift+P)
Configure Anthropic API
Select 'Anthropic' as the API Provider
Configure the Anthropic API Key with the Key obtained from the Kimi open platform
Check 'Use custom base URL': enter 'https://api.moonshot.ai/anthropic'
Any Model is fine, default model is set to 'claude-opus-4-20250514'
Check 'Disable browser tool usage' under Browser settings
Click 'Done' to save the configuration
config

browser

Experience kimi-k2-0711-preview Model in Cline
Let's have the kimi-k2-0711-preview model write a Snake game
The game in action
Direct API Usage for kimi-k2-0711-preview Model
from openai import OpenAI
 
client = OpenAI(
    api_key = "$MOONSHOT_API_KEY",
    base_url = "https://api.moonshot.ai/v1",
)
 
completion = client.chat.completions.create(
    model = "kimi-k2-0711-preview",
    messages = [
        {"role": "system", "content": "You are Kimi, an AI assistant provided by Moonshot AI. You are proficient in Chinese and English conversations. You provide users with safe, helpful, and accurate answers. You will reject any questions involving terrorism, racism, or explicit content. Moonshot AI is a proper noun and should not be translated."},
        {"role": "user", "content": "Hello, my name is Li Lei. What is 1+1?"}
    ],
    temperature = 0.3,
)
 
print(completion.choices[0].message.content)

Replace $MOONSHOT_API_KEY with your API Key created on the platform.

When running the code in this documentation using the OpenAI SDK, ensure that Python version is at least 3.7.1, Node.js version is at least 18, and OpenAI SDK version is no lower than 1.0.0.

Best Practices for Prompts
Best Practices for System Prompts: A system prompt refers to the initial input or instruction that a model receives before generating text or responding. This prompt is crucial for the model's operation link.

Write Clear Instructions
Why is it necessary to provide clear instructions to the model?
The model can't read your mind. If the output is too long, you can ask the model to respond briefly. If the output is too simple, you can request expert-level writing. If you don't like the format of the output, show the model the format you'd like to see. The less the model has to guess about your needs, the more likely you are to get satisfactory results.

Including More Details in Your Request Can Yield More Relevant Responses
To obtain highly relevant output, ensure that your input request includes all important details and context.

General Request	Better Request
How to add numbers in Excel?	How do I sum a row of numbers in an Excel table? I want to automatically sum each row in the entire table and place all the totals in the rightmost column named "Total."
Work report summary	Summarize my work records from 2023 in a paragraph of no more than 500 words. List the highlights of each month in sequence and provide a summary of the entire year.
Requesting the Model to Assume a Role Can Yield More Accurate Output
Add a specified role for the model to use in its response in the 'messages' field of the API request.

{
  "messages": [
    {"role": "system", "content": "You are Kimi, an artificial intelligence assistant provided by Moonshot AI. You are more proficient in Chinese and English conversations. You provide users with safe, helpful, and accurate answers. At the same time, you will refuse to answer any questions involving terrorism, racism, or explicit violence. Moonshot AI is a proper noun and should not be translated into other languages."},
    {"role": "user", "content": "Hello, my name is Li Lei. What is 1+1?"}
  ]
}

Using Delimiters in Your Request to Clearly Distinguish Different Parts of the Input
For example, using triple quotes/XML tags/section headings as delimiters can help distinguish text parts that require different processing.

{
  "messages": [
    {"role": "system", "content": "You will receive two articles of the same category, separated by XML tags. First, summarize the arguments of each article, then point out which article presents a better argument and explain why."},
    {"role": "user", "content": "<article>Insert article here</article><article>Insert article here</article>"}
  ]
}

{
  "messages": [
    {"role": "system", "content": "You will receive an abstract and the title of a paper. The title should give readers a clear idea of the paper's topic and also be eye-catching. If the title you receive does not meet these standards, please suggest five alternative options."},
    {"role": "user", "content": "Abstract: Insert abstract here.\n\nTitle: Insert title here"}
  ]
}

Clearly Define the Steps Needed to Complete the Task
It is advisable to outline a series of steps for the task. Writing these steps explicitly makes it easier for the model to follow and produces better output.

{
  "messages": [
    {"role": "system", "content": "Respond to user input using the following steps.\nStep one: The user will provide text enclosed in triple quotes. Summarize this text into one sentence with the prefix “Summary: ”.\nStep two: Translate the summary from step one into English and add the prefix "Translation: "."},
    {"role": "user", "content": "\"\"\"Insert text here\"\"\""}
  ]
}

Provide Examples of Desired Output to the Model
Providing examples of general guidance is usually more efficient for the model's output than showing all permutations of the task. For instance, if you intend to have the model replicate a style that is difficult to describe explicitly in response to user queries, this is known as a "few-shot" prompt.

{
  "messages": [
    {"role": "system", "content": "Respond in a consistent style"},
    {"role": "user", "content": "Insert text here"}
  ]
}

Specify the Desired Length of the Model's Output
You can request the model to generate output of a specific target length. The target output length can be specified in terms of words, sentences, paragraphs, bullet points, etc. However, note that instructing the model to generate a specific number of words is not highly precise. The model is better at generating output of a specific number of paragraphs or bullet points.

{
  "messages": [
    {"role": "user", "content": "Summarize the text within the triple quotes in two sentences, within 50 words. \"\"\"Insert text here\"\"\""}
  ]
}

Provide Reference Text
Guide the Model to Use Reference Text to Answer Questions
If you can provide a model with credible information related to the current query, you can guide the model to use the provided information to answer the question.

{
  "messages": [
    {"role": "system", "content": "Answer the question using the provided article (enclosed in triple quotes). If the answer is not found in the article, write "I can't find the answer." "},
    {"role": "user", "content": "<Insert article, each article enclosed in triple quotes>"}
  ]
}

Break Down Complex Tasks
Categorize to Identify Instructions Relevant to User Queries
For tasks that require a large set of independent instructions to handle different scenarios, categorizing the query type and using this categorization to clarify which instructions are needed may aid the output.

# Based on the classification of the customer query, a set of more specific instructions can be provided to the model to help it handle subsequent steps. For example, assume the customer needs help with "troubleshooting."
{
  "messages": [
    {"role": "system", "content": "You will receive a customer service inquiry that requires technical support. You can assist the user in the following ways:\n\n-Ask them to check if *** is configured.\nIf all *** are configured but the problem persists, ask for the device model they are using\n-Now you need to tell them how to restart the device:\n=If the device model is A, perform ***.\n-If the device model is B, suggest they perform ***."}
  ]
}

For Long-Running Dialog Applications, Summarize or Filter Previous Conversations
Since the model has a fixed context length, the conversation between the user and the model assistant cannot continue indefinitely.

One solution to this issue is to summarize the first few rounds of the conversation. Once the input size reaches a predetermined threshold, a query is triggered to summarize the previous part of the conversation, and the summary of the previous conversation can also be included as part of the system message. Alternatively, previous conversations throughout the entire chat process can be summarized asynchronously.

Chunk and Recursively Build a Complete Summary for Long Documents
To summarize the content of a book, we can use a series of queries to summarize each chapter of the document. Partial summaries can be aggregated and summarized to produce a summary of summaries. This process can be recursively repeated until the entire book is summarized. If understanding later parts requires reference to earlier chapters, then when summarizing a specific point in the book, include summaries of the chapters preceding that point.

Setting Up and Verifying Your Organization
When you register and log in to the Open Platform account, you can find your organization ID on the Organization Management - Organization Verification page. The organization ID is the unique identifier for your organization.

Managing Projects and Usage Limits
To meet the needs of multiple business product lines under a single organization, or to distinguish between production and testing environments, you can create multiple projects under your organization. Within each project, you can create an API Key. The calls made using the project's API Key will be recorded under the project's consumption, allowing you to independently manage the usage of different projects.

Project Balance and Rate Limiting
All projects under an organization share the organization's rate limits.
All projects under an organization share the organization's account balance.
Project Consumption Management
The platform now supports setting monthly and daily consumption budgets on a per-project basis. You can set the monthly or daily consumption limits for each project on the Project Management - Project Settings - Project Budget/Rate Limiting Settings page. Once the API Key consumption within a project reaches the set budget, any subsequent API requests for that project will be denied, effectively helping you manage your business budget. Due to billing cycle issues, the actual enforcement of these limits may have a delay of about 10 minutes.

If you want to receive SMS notifications when a project's usage reaches a specific amount, you can set up monthly or daily consumption alerts on the Project Management - Project Settings - Project Consumption Alerts page. When the platform calculates that the natural month or day's consumption has reached the limit, it will trigger an SMS alert and send a message to the organization administrator's mobile phone.

If you wish to limit the maximum TPM (Transactions Per Minute) for a single project, you can configure the project's TPM rate limit independently. If the project's API Key requests reach this TPM, the requests will be denied. (The project's TPM must not exceed the organization's TPM. If you set a value higher than the organization's TPM, the organization's TPM will be used for rate limiting.)

The platform also provides an overview page for both the organization and individual projects, offering consumption analysis at both levels to help you get a clear understanding of your organization's spending.

Frequently Asked Questions and Solutions
Why are the results from the API different from those from the Kimi large language model?
The API and the Kimi large language model use the same underlying model. If you notice discrepancies in the output, you can try modifying the System Prompt. For examples of the System Prompt used by the Kimi large language model, you can refer to this link. Additionally, the Kimi large language model includes tools like a calculator, which are not provided by default in the API. Users need to assemble these tools themselves.

Does the Kimi API have the "web surfing" feature of the Kimi large language model?

Now, the Kimi API offers web search functionality. Please refer to our guide:

Using the Web Search Feature of the Kimi API https://platform.moonshot.ai/docs/guide/use-web-search

If you want to implement web search functionality through the Kimi API yourself, you can also refer to our tool_calls guide:

Using the Kimi API for Tool Calls https://platform.moonshot.ai/docs/guide/use-kimi-api-to-complete-tool-calls
 
If you seek assistance from the open-source community, you can refer to the following open-source projects:

search2ai https://github.com/fatwang2/search2ai
ArchiveBox https://github.com/ArchiveBox/ArchiveBox
 
If you are looking for services provided by professional vendors, the following options are available:

apify
crawlbase
jina reader

The content returned by the Kimi API is incomplete or truncated
If you find that the content returned by the Kimi API is incomplete, truncated, or does not meet the expected length, you can first check the value of the choice.finish_reason field in the response. If this value is length, it means that the number of Tokens in the content generated by the current model exceeds the max_tokens parameter in the request. In this case, the Kimi API will only return content within the max_tokens limit, and any excess content will be discarded, resulting in the aforementioned "incomplete content" or "truncated content."

When encountering finish_reason=length, if you want the Kimi large language model to continue generating content from where it left off, you can use the Partial Mode provided by the Kimi API. For detailed documentation, please refer to:

Using the Partial Mode Feature of the Kimi API

To avoid finish_reason=length, we recommend increasing the value of max_tokens. Our best practice suggestion is: use the estimate-token-count API to calculate the number of Tokens in the input content, then subtract this number from the maximum number of Tokens supported by the Kimi large language model (for example, for the moonshot-v1-32k model, the maximum is 32k Tokens). The resulting value should be used as the max_tokens value for the current request.

Taking the moonshot-v1-32k model as an example:

max_tokens = 32*1024 - prompt_tokens

Error Your request exceeded model token limit despite very short input content
We determine whether a request exceeds the context window size of the Kimi large language model by adding the number of Tokens used by the input content to the max_tokens value set in the request. For example, with the moonshot-v1-32k model, ensure that:

prompt_tokens + max_tokens ≤ 32*1024

What is the output length of the Kimi large language model?
For the moonshot-v1-8k model, the maximum output length is 8*1024 - prompt_tokens;
For the moonshot-v1-32k model, the maximum output length is 32*1024 - prompt_tokens;
For the moonshot-v1-128k model, the maximum output length is 128*1024 - prompt_tokens;
How many Chinese characters does the Kimi large language model support?
The moonshot-v1-8k model supports approximately 15,000 Chinese characters;
The moonshot-v1-32k model supports approximately 60,000 Chinese characters;
The moonshot-v1-128k model supports approximately 200,000 Chinese characters;
Note: These are estimated values and actual results may vary.

Inaccurate file content extraction or inability to recognize images
We offer file upload and parsing services for various file formats. For text files, we extract the text content; for image files, we use OCR to recognize text in the images; for PDF documents, if the PDF contains only images, we use OCR to extract text from the images, otherwise we only extract the text content.;

Note that for images, we only use OCR to extract text content, so if your image does not contain any text, it will result in a parsing failure error.

For a complete list of supported file formats, please refer to:

File Interface

When using the files interface, I want to reference file content using file_id
We currently do not support referencing file content using the file file_id. However, we do support caching file content (using Context Caching technology) and then referencing the cached file content using cache_id or cache_tag to achieve a similar effect.

For specific usage, please refer to:

Using the Context Caching Feature of the Kimi API

Error content_filter: The request was rejected because it was considered high risk
The input to the Kimi API or the output from the Kimi large language model contains unsafe or sensitive content. Note: The content generated by the Kimi large language model may also contain unsafe or sensitive content, which can lead to the content_filter error.

Connection-related errors
If you frequently encounter errors such as Connection Error or Connection Time Out while using the Kimi API, please check the following in order:

Whether your program code or the SDK you are using has a default timeout setting;
Whether you are using any type of proxy server and check the network and timeout settings of the proxy server;
Whether you are accessing the Kimi API from an overseas server. If you need to request the Kimi API from overseas, we recommend replacing the base_url with:
https://api-sg.moonshot.ai/v1

Another scenario that may lead to connection-related errors is when the number of Tokens generated by the Kimi large language model is too high and stream output stream=True is not enabled. This can cause the waiting time for the Kimi large language model to generate content to exceed the timeout settings of an intermediate gateway. Typically, some gateway applications determine whether a request is valid by detecting whether a status_code and header are received from the server. When not using stream output stream=True, the Kimi server will wait for the Kimi large language model to finish generating content before sending the header. While waiting for the header to return, some gateway applications may close connections that have been waiting for too long, resulting in connection-related errors.

We recommend enabling stream output stream=True to minimize connection-related errors.

The TPM and RPM limits shown in the error message do not match my account Tier level
If you encounter a rate_limit_reached_error while using the Kimi API, such as:

rate_limit_reached_error: Your account {uid}<{ak-id}> request reached TPM rate limit, current:{current_tpm}, limit:{max_tpm}

and the TPM or RPM limits in the error message do not match the TPM and RPM you see in the backend, please first check whether you are using the correct api_key for your account. In most cases, the reason for the mismatch between TPM and RPM and expectations is the use of an incorrect api_key, such as mistakenly using an api_key provided by another user, or mixing up api_keys when you have multiple accounts.

Make sure you have correctly set base_url=https://api.moonshot.ai in your SDK. The model_not_found error usually occurs because the base_url value is not set when using the OpenAI SDK. As a result, requests are sent to the OpenAI server, and OpenAI returns the model_not_found error.

Numerical Calculation Errors in the Kimi Large Language Model
Due to the uncertainty in the generation process of the Kimi large language model, it may produce calculation errors of varying degrees when performing numerical computations. We recommend using tool calls (tool_calls) to provide the Kimi large language model with calculator functionality. For more information on tool calls (tool_calls), you can refer to our guide on Using the Kimi API for Tool Calls (tool_calls).

The Kimi Large Language Model Cannot Answer Today's Date
The Kimi large language model cannot access highly time-sensitive information such as the current date. However, you can provide this information to the Kimi large language model through the system prompt. For example:

import os
from datetime import datetime
from openai import OpenAI
 
client = OpenAI(
    api_key=os.environ['MOONSHOT_API_KEY'],
    base_url="https://api.moonshot.ai/v1",
)
 
# We generate the current date using the datetime library and add it to the system prompt
system_prompt = f"""
You are Kimi, and today's date is {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}
"""
 
completion = client.chat.completions.create(
    model="moonshot-v1-128k",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "What's today's date?"},
    ],
    temperature=0.3,
)
 
print(completion.choices[0].message.content)  # Output: Today's date is July 31, 2024.
 

How to Choose the Right Model Based on Context Length
Now, you can use model=moonshot-v1-auto to let Kimi automatically select a model that fits the current context length. Please refer to our guide:

Choosing the Right Kimi Large Language Model

We can choose the right model based on the length of the input context plus the expected output tokens. Here is an example of automatically selecting a model:

import os
import httpx
from openai import OpenAI
 
client = OpenAI(
    api_key=os.environ['MOONSHOT_API_KEY'],
    base_url="https://api.moonshot.ai/v1",
)
 
 
def estimate_token_count(input_messages) -> int:
    """
    Implement your token counting logic here, or directly call our token counting API to calculate tokens.
 
    https://api.moonshot.ai/v1/tokenizers/estimate-token-count
    """
    header = {
        "Authorization": f"Bearer {os.environ['MOONSHOT_API_KEY']}",
    }
    data = {
        "model": "moonshot-v1-128k",
        "messages": input_messages,
    }
    r = httpx.post("https://api.moonshot.ai/v1/tokenizers/estimate-token-count", headers=header, json=data)
    r.raise_for_status()
    return r.json()["data"]["total_tokens"]
 
 
def select_model(input_messages, max_tokens=1024) -> str:
    """
    Select a model of the right size based on the input context messages and the expected max_tokens value.
 
    The select_model function calls the estimate_token_count function to calculate the number of tokens used by the input messages, adds the max_tokens value to get the total_tokens, and then selects the appropriate model based on the range of total_tokens.
    """
    prompt_tokens = estimate_token_count(input_messages)
    total_tokens = prompt_tokens + max_tokens
    if total_tokens <= 8 * 1024:
        return "moonshot-v1-8k"
    elif total_tokens <= 32 * 1024:
        return "moonshot-v1-32k"
    elif total_tokens <= 128 * 1024:
        return "moonshot-v1-128k"
    else:
        raise Exception("too many tokens 😢")
 
 
messages = [
    {"role": "system", "content": "You are Kimi"},
    {"role": "user", "content": "Hello, please tell me a fairy tale."},
]
 
max_tokens = 2048
model = select_model(messages, max_tokens)
 
completion = client.chat.completions.create(
    model=model,
    messages=messages,
    max_tokens=max_tokens,
    temperature=0.3,
)
 
print("model:", model)
print("max_tokens:", max_tokens)
print("completion:", completion.choices[0].message.content)

How to Handle Errors Without Using an SDK
In some cases, you might need to directly interface with the Kimi API (instead of using the OpenAI SDK). When interfacing with the Kimi API directly, you need to determine the subsequent processing logic based on the status returned by the API. Typically, we use the HTTP status code 200 to indicate a successful request, while 4xx and 5xx status codes indicate a failed request. We provide error information in JSON format. For specific handling logic based on the request status, please refer to the following code snippets:

import os
import httpx
 
header = {
    "Authorization": f"Bearer {os.environ['MOONSHOT_API_KEY']}",
}
 
messages = [
    {"role": "system", "content": "You are Kimi"},
    {"role": "user", "content": "Hello."},
]
 
r = httpx.post("https://api.moonshot.ai/v1/chat/completions",
               headers=header,
               json={
                   "model": "moonshot-v1-128k",  # <-- If you use a correct model, the code will enter the if status_code==200 branch below
                   # "model": "moonshot-v1-129k",  # <-- If you use an incorrect model name, the code will enter the else branch below
                   "messages": messages,
                   "temperature": 0.3,
               })
 
if r.status_code == 200:  # When a correct model is used for the request, this branch is entered for normal processing
    completion = r.json()
    print(completion["choices"][0]["message"]["content"])
else:  # When an incorrect model name is used for the request, this branch is entered for error handling
    # Here, for demonstration purposes, we simply print the error.
    # In actual code logic, you might need more processing, such as logging the error, interrupting the request, or retrying.
    error = r.json()
    print(f"error: status={r.status_code}, type='{error['error']['type']}', message='{error['error']['message']}'")

Our error messages will follow this format:

{
	"error": {
		"type": "error_type",
		"message": "error_message"
	}
}

For a detailed list of error messages, please refer to the following section:

Error Description

Why Do Some Requests Respond Quickly While Others Respond Slowly When the Prompt Is Similar?
If you find that some requests respond quickly (e.g., in just 3 seconds) while others respond slowly (e.g., taking up to 20 seconds) with similar prompts, it is usually because the Kimi large language model generates a different number of tokens. Generally, the number of tokens generated by the Kimi large language model is directly proportional to the response time of the Kimi API; the more tokens generated, the longer the complete response time.

It is important to note that the number of tokens generated by the Kimi large language model only affects the response time for the complete request (i.e., after generating the last token). You can set stream=True and observe the time to first token (TTFT) return time. Under normal circumstances, when the length of the prompt is similar, the first token response time will not vary significantly.

I Set max_tokens=2000 to Have Kimi Output 2000 Characters, but the Output Is Less Than 2000 Characters
The max_tokens parameter means: When calling /v1/chat/completions, it specifies the maximum number of tokens the model is allowed to generate. When the number of tokens already generated by the model exceeds the set max_tokens, the model will stop generating the next token.

The purpose of max_tokens is:

To help the caller determine which model to use (for example, when prompt_tokens + max_tokens ≤ 8 * 1024, you can choose the moonshot-v1-8k model);
To prevent the Kimi model from generating excessive unexpected content in certain unexpected situations, which could lead to additional cost consumption (for example, the Kimi model repeatedly outputs blank characters).
max_tokens does not indicate how many tokens the Kimi large language model will output. In other words, max_tokens will not be used as part of the prompt input to the Kimi large language model. If you want the model to output a specific number of characters, you can refer to the following general solutions:

For occasions where the output content should be within 1000 characters:
Specify the number of characters in the prompt to the Kimi large language model;
Manually or programmatically check if the output character count meets expectations. If not, in the second round of conversation, indicate to the Kimi large language model that the "character count is too high" or "character count is too low" to generate a new round of content.
For occasions where the output content should be more than 1000 characters or even more:
Try to break down the expected output content into several parts by structure or chapter and create a template, using placeholders to mark the positions where you want the Kimi large language model to output content;
Have the Kimi large language model fill in each placeholder of the template one by one, and finally assemble the complete long text.
I Made Only One Request in a Minute, but Triggered the Your account reached max request Error
Typically, the SDK provided by OpenAI includes a retry mechanism:

Certain errors are automatically retried 2 times by default, with a short exponential backoff. Connection errors (for example, due to a network connectivity problem), 408 Request Timeout, 409 Conflict, 429 Rate Limit, and >=500 Internal errors are all retried by default.

This retry mechanism will automatically retry 2 times (a total of 3 requests) when encountering an error. Generally speaking, in cases of unstable network conditions or other situations that may cause request errors, using the OpenAI SDK can amplify a single request into 2 to 3 requests, all of which will count towards your RPM (requests per minute) limit.

Note: For users using the OpenAI SDK with a free account level, due to the default retry mechanism, a single erroneous request can exhaust the entire RPM quota.

To Facilitate Transmission, I Used base64 Encoding for My Text Content
Please do not do this. Encoding your files with base64 will result in a huge consumption of tokens. If your file type is supported by our /v1/files file interface, you can simply upload the file and extract its content using the file interface.

For binary or other encoded file formats, the Kimi large language model currently cannot parse the content, so please do not add it to the context.




