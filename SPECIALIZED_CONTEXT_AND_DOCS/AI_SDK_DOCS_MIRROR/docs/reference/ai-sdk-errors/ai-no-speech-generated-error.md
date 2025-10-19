# AI_NoSpeechGeneratedError

This error occurs when no audio could be generated from the input.

## Properties

- `responses`: Array of responses
- `message`: The error message

## Checking for this Error

You can check if an error is an instance of `AI_NoSpeechGeneratedError` using:

```typescript
import { NoSpeechGeneratedError } from 'ai';

if (NoSpeechGeneratedError.isInstance(error)) {
  // Handle the error
}
```
