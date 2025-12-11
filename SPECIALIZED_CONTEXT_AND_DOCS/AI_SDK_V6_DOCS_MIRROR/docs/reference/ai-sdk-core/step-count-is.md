# `stepCountIs()`

Creates a stop condition that stops when the number of steps reaches a specified count.

This function is used with `stopWhen` in `generateText` and `streamText` to control when a tool-calling loop should stop based on the number of steps executed.

```ts
import { generateText, stepCountIs } from 'ai';

const result = await generateText({
  model: "anthropic/claude-sonnet-4.5",
  tools: {
    // your tools
  },
  // Stop after 5 steps
  stopWhen: stepCountIs(5),
});
```

## Import

```
import { stepCountIs } from "ai"
```

## API Signature

### Parameters

### count:

number

The maximum number of steps to execute before stopping the tool-calling loop.

### Returns

A `StopCondition` function that returns `true` when the step count reaches the specified number. The function can be used with the `stopWhen` parameter in `generateText` and `streamText`.

## Examples

### Basic Usage

Stop after 3 steps:

```ts
import { generateText, stepCountIs } from 'ai';

const result = await generateText({
  model: yourModel,
  tools: yourTools,
  stopWhen: stepCountIs(3),
});
```

### Combining with Other Conditions

You can combine multiple stop conditions in an array:

```ts
import { generateText, stepCountIs, hasToolCall } from 'ai';

const result = await generateText({
  model: yourModel,
  tools: yourTools,
  // Stop after 10 steps OR when finalAnswer tool is called
  stopWhen: [stepCountIs(10), hasToolCall('finalAnswer')],
});
```

## See also

- [`hasToolCall()`](has-tool-call.md)
- [`generateText()`](generate-text.md)
- [`streamText()`](stream-text.md)
