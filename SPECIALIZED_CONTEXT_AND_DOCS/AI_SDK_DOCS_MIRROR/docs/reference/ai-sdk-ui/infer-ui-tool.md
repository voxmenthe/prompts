# InferUITool

Infers the input and output types of a tool.

This type helper is useful when working with individual tools to ensure type safety for your tool inputs and outputs in `UIMessage`s.

## Import

```tsx
import { InferUITool } from 'ai';
```

## API Signature

### Type Parameters

### TOOL:

Tool

The tool to infer types from.

### Returns

A type that contains the inferred input and output types of the tool.

The resulting type has the shape:

```typescript
{
  input: InferToolInput<TOOL>;
  output: InferToolOutput<TOOL>;
}
```

## Examples

### Basic Usage

```tsx
import { InferUITool } from 'ai';
import { z } from 'zod';

const weatherTool = {
  description: 'Get the current weather',
  parameters: z.object({
    location: z.string().describe('The city and state'),
  }),
  execute: async ({ location }) => {
    return `The weather in ${location} is sunny.`;
  },
};

// Infer the types from the tool
type WeatherUITool = InferUITool<typeof weatherTool>;
// This creates a type with:
// {
//   input: { location: string };
//   output: string;
// }
```

## Related

- [`InferUITools`](infer-ui-tools.md) - Infer types for a tool set
- [`ToolUIPart`](tool-ui-part.md) - Tool part type for UI messages
