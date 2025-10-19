# Model is not assignable to type "LanguageModelV1"

## Issue

I have updated the AI SDK and now I get the following error: `Type 'SomeModel' is not assignable to type 'LanguageModelV1'.`

Similar errors can occur with `EmbeddingModelV2` as well.

## Background

Sometimes new features are being added to the model specification.
This can cause incompatibilities with older provider versions.

## Solution

Update your provider packages and the AI SDK to the latest version.
