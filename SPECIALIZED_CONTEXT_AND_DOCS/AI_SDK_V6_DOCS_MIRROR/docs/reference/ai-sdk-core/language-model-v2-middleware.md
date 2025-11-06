# `LanguageModelV3Middleware`

Language model middleware is an experimental feature.

Language model middleware provides a way to enhance the behavior of language models
by intercepting and modifying the calls to the language model. It can be used to add
features like guardrails, RAG, caching, and logging in a language model agnostic way.

See [Language Model Middleware](../../ai-sdk-core/middleware.md) for more information.

## Import

```
import { LanguageModelV3Middleware } from "ai"
```

## API Signature

### transformParams:

({ type: "generate" | "stream", params: LanguageModelV3CallOptions }) => Promise<LanguageModelV3CallOptions>

Transforms the parameters before they are passed to the language model.

### wrapGenerate:

({ doGenerate: DoGenerateFunction, params: LanguageModelV3CallOptions, model: LanguageModelV3 }) => Promise<DoGenerateResult>

Wraps the generate operation of the language model.

### wrapStream:

({ doStream: DoStreamFunction, params: LanguageModelV3CallOptions, model: LanguageModelV3 }) => Promise<DoStreamResult>

Wraps the stream operation of the language model.
