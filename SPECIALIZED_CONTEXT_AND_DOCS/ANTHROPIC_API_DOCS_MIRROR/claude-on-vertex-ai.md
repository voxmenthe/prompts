# Vertex AI API

> Anthropic's Claude models are now generally available through [Vertex AI](https://cloud.google.com/vertex-ai).

The Vertex API for accessing Claude is nearly-identical to the [Messages API](/en/api/messages) and supports all of the same options, with two key differences:

* In Vertex, `model` is not passed in the request body. Instead, it is specified in the Google Cloud endpoint URL.
* In Vertex, `anthropic_version` is passed in the request body (rather than as a header), and must be set to the value `vertex-2023-10-16`.

Vertex is also supported by Anthropic's official [client SDKs](/en/api/client-sdks). This guide will walk you through the process of making a request to Claude on Vertex AI in either Python or TypeScript.

Note that this guide assumes you have already have a GCP project that is able to use Vertex AI. See [using the Claude 3 models from Anthropic](https://cloud.google.com/vertex-ai/generative-ai/docs/partner-models/use-claude) for more information on the setup required, as well as a full walkthrough.

## Install an SDK for accessing Vertex AI

First, install Anthropic's [client SDK](/en/api/client-sdks) for your language of choice.

<CodeGroup>
  ```Python Python
  pip install -U google-cloud-aiplatform "anthropic[vertex]"
  ```

  ```TypeScript TypeScript
  npm install @anthropic-ai/vertex-sdk
  ```
</CodeGroup>

## Accessing Vertex AI

### Model Availability

Note that Anthropic model availability varies by region. Search for "Claude" in the [Vertex AI Model Garden](https://cloud.google.com/model-garden) or go to [Use Claude 3](https://cloud.google.com/vertex-ai/generative-ai/docs/partner-models/use-claude) for the latest information.

#### API model names

| Model                                                                           | Vertex AI API model name       |
| ------------------------------------------------------------------------------- | ------------------------------ |
| Claude Opus 4.1                                                                 | claude-opus-4-1\@20250805      |
| Claude Opus 4                                                                   | claude-opus-4\@20250514        |
| Claude Sonnet 4                                                                 | claude-sonnet-4\@20250514      |
| Claude Sonnet 3.7                                                               | claude-3-7-sonnet\@20250219    |
| Claude Haiku 3.5                                                                | claude-3-5-haiku\@20241022     |
| Claude Sonnet 3.5 <Tooltip tip="Deprecated as of August 13, 2025.">⚠️</Tooltip> | claude-3-5-sonnet-v2\@20241022 |
| Claude Opus 3 <Tooltip tip="Deprecated as of June 30, 2025.">⚠️</Tooltip>       | claude-3-opus\@20240229        |
| Claude Haiku 3                                                                  | claude-3-haiku\@20240307       |

### Making requests

Before running requests you may need to run `gcloud auth application-default login` to authenticate with GCP.

The following examples shows how to generate text from Claude on Vertex AI:

<CodeGroup>
  ```Python Python
  from anthropic import AnthropicVertex

  project_id = "MY_PROJECT_ID"
  # Where the model is running
  region = "us-east5"

  client = AnthropicVertex(project_id=project_id, region=region)

  message = client.messages.create(
      model="claude-opus-4-1@20250805",
      max_tokens=100,
      messages=[
          {
              "role": "user",
              "content": "Hey Claude!",
          }
      ],
  )
  print(message)
  ```

  ```TypeScript TypeScript
  import { AnthropicVertex } from '@anthropic-ai/vertex-sdk';

  const projectId = 'MY_PROJECT_ID';
  // Where the model is running
  const region = 'us-east5';

  // Goes through the standard `google-auth-library` flow.
  const client = new AnthropicVertex({
    projectId,
    region,
  });

  async function main() {
    const result = await client.messages.create({
      model: 'claude-opus-4-1@20250805',
      max_tokens: 100,
      messages: [
        {
          role: 'user',
          content: 'Hey Claude!',
        },
      ],
    });
    console.log(JSON.stringify(result, null, 2));
  }

  main();
  ```

  ```bash Shell
  MODEL_ID=claude-opus-4-1@20250805
  LOCATION=us-east5
  PROJECT_ID=MY_PROJECT_ID

  curl \
  -X POST \
  -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  -H "Content-Type: application/json" \
  https://$LOCATION-aiplatform.googleapis.com/v1/projects/${PROJECT_ID}/locations/${LOCATION}/publishers/anthropic/models/${MODEL_ID}:streamRawPredict -d \
  '{
    "anthropic_version": "vertex-2023-10-16",
    "messages": [{
      "role": "user",
      "content": "Hey Claude!"
    }],
    "max_tokens": 100,
  }'
  ```
</CodeGroup>

See our [client SDKs](/en/api/client-sdks) and the official [Vertex AI docs](https://cloud.google.com/vertex-ai/docs) for more details.

## Activity logging

Vertex provides a [request-response logging service](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/request-response-logging) that allows customers to log the prompts and completions associated with your usage.

Anthropic recommends that you log your activity on at least a 30-day rolling basis in order to understand your activity and investigate any potential misuse.

<Note>
  Turning on this service does not give Google or Anthropic any access to your content.
</Note>

## Feature support

You can find all the features currently supported on Vertex [here](/en/docs/build-with-claude/overview).
