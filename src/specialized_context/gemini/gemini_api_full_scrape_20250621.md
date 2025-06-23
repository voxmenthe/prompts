[Skip to main content](https://ai.google.dev/gemini-api/docs#main-content)

[![Google AI for Developers](https://www.gstatic.com/devrel-devsite/prod/va55008f56463f12ba1a0c4ec3fdc81dac4d4d331f95ef7b209d2570e7d9e879b/googledevai/images/lockup-new.svg)](https://ai.google.dev/)

`/`

- [English](https://ai.google.dev/gemini-api/docs)


[Sign in](https://ai.google.dev/_d/signin?continue=https%3A%2F%2Fai.google.dev%2Fgemini-api%2Fdocs&prompt=select_account)

Introducing updates to our 2.5 family of thinking models. [Learn more](https://ai.google.dev/gemini-api/docs/models)

- [Home](https://ai.google.dev/)
- [Gemini API](https://ai.google.dev/gemini-api)
- [Models](https://ai.google.dev/gemini-api/docs)

# Gemini Developer API

[Get a Gemini API Key](https://aistudio.google.com/apikey)

Get a Gemini API key and make your first API request in minutes.

[Python](https://ai.google.dev/gemini-api/docs#python)[JavaScript](https://ai.google.dev/gemini-api/docs#javascript)[Go](https://ai.google.dev/gemini-api/docs#go)[Java](https://ai.google.dev/gemini-api/docs#java)[REST](https://ai.google.dev/gemini-api/docs#rest)More

```
from google import genai

client = genai.Client(api_key="YOUR_API_KEY")

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Explain how AI works in a few words",
)

print(response.text)

```

```
import { GoogleGenAI } from "@google/genai";

const ai = new GoogleGenAI({ apiKey: "YOUR_API_KEY" });

async function main() {
  const response = await ai.models.generateContent({
    model: "gemini-2.5-flash",
    contents: "Explain how AI works in a few words",
  });
  console.log(response.text);
}

await main();

```

```
package main

import (
    "context"
    "fmt"
    "log"

    "google.golang.org/genai"
)

func main() {
    ctx := context.Background()
    client, err := genai.NewClient(ctx, &genai.ClientConfig{
        APIKey:  "YOUR_API_KEY",
        Backend: genai.BackendGeminiAPI,
    })
    if err != nil {
        log.Fatal(err)
    }

    result, err := client.Models.GenerateContent(
        ctx,
        "gemini-2.5-flash",
        genai.Text("Explain how AI works in a few words"),
        nil,
    )
    if err != nil {
        log.Fatal(err)
    }
    fmt.Println(result.Text())
}

```

```
package com.example;

import com.google.genai.Client;
import com.google.genai.types.GenerateContentResponse;

public class GenerateTextFromTextInput {
  public static void main(String[] args) {
    // The client gets the API key from the environment variable `GOOGLE_API_KEY`.
    Client client = new Client();

    GenerateContentResponse response =
        client.models.generateContent(
            "gemini-2.5-flash",
            "Explain how AI works in a few words",
            null);

    System.out.println(response.text());
  }
}

```

```
curl "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=YOUR_API_KEY" \
  -H 'Content-Type: application/json' \
  -X POST \
  -d '{
    "contents": [\
      {\
        "parts": [\
          {\
            "text": "Explain how AI works in a few words"\
          }\
        ]\
      }\
    ]
  }'

```

## Meet the models

[Use Gemini in Google AI Studio](https://aistudio.google.com/)

2.5 Pro
spark

Our most powerful thinking model with features for complex reasoning and much more


[Learn more about 2.5 Pro](https://ai.google.dev/gemini-api/docs/models#gemini-2.5-pro)

2.5 Flash
spark

Our newest multimodal model, with next generation features and improved
capabilities


[Learn more about 2.5 Flash](https://ai.google.dev/gemini-api/docs/models/gemini#gemini-2.5-flash)

2.5 Flash-Lite
spark

Our fastest and most cost-efficient multimodal model with great performance
for high-frequency tasks


[Learn more about 2.5 Flash-Lite](https://ai.google.dev/gemini-api/docs/models/gemini#gemini-2.5-flash-lite)

## Explore the API

![](https://ai.google.dev/static/site-assets/images/image-generation-index.png)

### Native Image Generation

Generate and edit highly contextual images natively with Gemini 2.0 Flash.

![](https://ai.google.dev/static/site-assets/images/long-context-overview.png)

### Explore long context

Input millions of tokens to Gemini models and derive understanding from unstructured images, videos, and documents.

![](https://ai.google.dev/static/site-assets/images/structured-outputs-index.png)

### Generate structured outputs

Constrain Gemini to respond with JSON, a structured data format suitable for automated processing.

### Start building with the Gemini API

[Get started](https://ai.google.dev/gemini-api/docs/quickstart)

Except as otherwise noted, the content of this page is licensed under the [Creative Commons Attribution 4.0 License](https://creativecommons.org/licenses/by/4.0/), and code samples are licensed under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0). For details, see the [Google Developers Site Policies](https://developers.google.com/site-policies). Java is a registered trademark of Oracle and/or its affiliates.

Last updated 2025-06-17 UTC.