---
description: When creating or fixing tests
globs: *test*, *mock*
alwaysApply: false
---
Do not rely on the expected format of the tests being correct. Of course you should not rely on the actual code we are testing being 100% correct either, but they are most likely designed for a specific reason, so, before modifying the code to fit the tests, first fully understand how the code is working, and how it should work, and then decide whether to modify the code, or the tests.

When encountering an error in tests, first create a plan for how to fix the errors by first understanding the intentions of the original code that is being tested, and then making sure the tests are logically designed to test the key functionality of the underlying code. Do not modify any of the code or tests simply for the sake of tests passing. First make sure the underlying code is working correctly, and then make sure that the tests are logically designed to test the things we actually care about.

Additional note: We should assume QuantLib calculations stemming directly from quantlib functions are returning correct results. We just need to make sure we're using them correctly, and that we've properly implemented them in the code.

If needed, make sure to ask clarifying questions about the implementation of the original code, and its desired functionality.