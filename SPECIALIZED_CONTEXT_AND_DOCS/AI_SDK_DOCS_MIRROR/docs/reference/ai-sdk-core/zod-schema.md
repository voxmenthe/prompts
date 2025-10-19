# `zodSchema()`

`zodSchema` is a helper function that converts a Zod schema into a JSON schema object that is compatible with the AI SDK.
It takes a Zod schema and optional configuration as inputs, and returns a typed schema.

You can use it to [generate structured data](../../ai-sdk-core/generating-structured-data.md) and in [tools](../../ai-sdk-core/tools-and-tool-calling.md).

You can also pass Zod objects directly to the AI SDK functions. Internally,
the AI SDK will convert the Zod schema to a JSON schema using `zodSchema()`.
However, if you want to specify options such as `useReferences`, you can pass
the `zodSchema()` helper function instead.

## Example with recursive schemas

```ts
import { zodSchema } from 'ai';
import { z } from 'zod';

// Define a base category schema
const baseCategorySchema = z.object({
  name: z.string(),
});

// Define the recursive Category type
type Category = z.infer<typeof baseCategorySchema> & {
  subcategories: Category[];
};

// Create the recursive schema using z.lazy
const categorySchema: z.ZodType<Category> = baseCategorySchema.extend({
  subcategories: z.lazy(() => categorySchema.array()),
});

// Create the final schema with useReferences enabled for recursive support
const mySchema = zodSchema(
  z.object({
    category: categorySchema,
  }),
  { useReferences: true },
);
```

## Import

```
import { zodSchema } from "ai"
```

## API Signature

### Parameters

### zodSchema:

z.Schema

The Zod schema definition.

### options:

object

Additional options for the schema conversion.

object

### useReferences?:

boolean

Enables support for references in the schema. This is required for recursive schemas, e.g. with `z.lazy`. However, not all language models and providers support such references. Defaults to `false`.

### Returns

A Schema object that is compatible with the AI SDK, containing both the JSON schema representation and validation functionality.
