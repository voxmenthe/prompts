# Rate limits

> To mitigate misuse and manage capacity on our API, we have implemented limits on how much an organization can use the Claude API.

We have two types of limits:

1. **Spend limits** set a maximum monthly cost an organization can incur for API usage.
2. **Rate limits** set the maximum number of API requests an organization can make over a defined period of time.

We enforce service-configured limits at the organization level, but you may also set user-configurable limits for your organization's workspaces.

These limits apply to both Standard and Priority Tier usage. For more information about Priority Tier, which offers enhanced service levels in exchange for committed spend, see [Service Tiers](/en/api/service-tiers).

## About our limits

* Limits are designed to prevent API abuse, while minimizing impact on common customer usage patterns.
* Limits are defined by **usage tier**, where each tier is associated with a different set of spend and rate limits.
* Your organization will increase tiers automatically as you reach certain thresholds while using the API.
  Limits are set at the organization level. You can see your organization's limits in the [Limits page](https://console.anthropic.com/settings/limits) in the [Claude Console](https://console.anthropic.com/).
* You may hit rate limits over shorter time intervals. For instance, a rate of 60 requests per minute (RPM) may be enforced as 1 request per second. Short bursts of requests at a high volume can surpass the rate limit and result in rate limit errors.
* The limits outlined below are our standard tier limits. If you're seeking higher, custom limits or Priority Tier for enhanced service levels, contact sales through the [Claude Console](https://console.anthropic.com/settings/limits).
* We use the [token bucket algorithm](https://en.wikipedia.org/wiki/Token_bucket) to do rate limiting. This means that your capacity is continuously replenished up to your maximum limit, rather than being reset at fixed intervals.
* All limits described here represent maximum allowed usage, not guaranteed minimums. These limits are intended to reduce unintentional overspend and ensure fair distribution of resources among users.

## Spend limits

Each usage tier has a limit on how much you can spend on the API each calendar month. Once you reach the spend limit of your tier, until you qualify for the next tier, you will have to wait until the next month to be able to use the API again.

To qualify for the next tier, you must meet a deposit requirement. To minimize the risk of overfunding your account, you cannot deposit more than your monthly spend limit.

### Requirements to advance tier

<table>
  <thead>
    <tr><th>Usage Tier</th><th>Credit Purchase</th><th>Max Usage per Month</th></tr>
  </thead>

  <tbody>
    <tr><td>Tier 1</td><td>\$5</td><td>\$100</td></tr>
    <tr><td>Tier 2</td><td>\$40</td><td>\$500</td></tr>
    <tr><td>Tier 3</td><td>\$200</td><td>\$1,000</td></tr>
    <tr><td>Tier 4</td><td>\$400</td><td>\$5,000</td></tr>
    <tr><td>Monthly Invoicing</td><td>N/A</td><td>N/A</td></tr>
  </tbody>
</table>

## Rate limits

Our rate limits for the Messages API are measured in requests per minute (RPM), input tokens per minute (ITPM), and output tokens per minute (OTPM) for each model class.
If you exceed any of the rate limits you will get a [429 error](/en/api/errors) describing which rate limit was exceeded, along with a `retry-after` header indicating how long to wait.

<Note>
  You might also encounter 429 errors due to acceleration limits on the API if your organization has a sharp increase in usage. To avoid hitting acceleration limits, ramp up your traffic gradually and maintain consistent usage patterns.
</Note>

ITPM rate limits are estimated at the beginning of each request, and the estimate is adjusted during the request to reflect the actual number of input tokens used.
The final adjustment counts `input_tokens` and `cache_creation_input_tokens` towards ITPM rate limits.

<Note>
  For some models, `cache_read_input_tokens` also count towards ITPM rate limits. The maximum ITPM for these models is marked with † in the rate limit tables below.

  For all other models, `cache_read_input_tokens` do not count towards ITPM rate limits (though they are still billed).
</Note>

OTPM rate limits are estimated based on `max_tokens` at the beginning of each request, and the estimate is adjusted at the end of the request to reflect the actual number of output tokens used.
If you're hitting OTPM limits earlier than expected, try reducing `max_tokens` to better approximate the size of your completions.

Rate limits are applied separately for each model; therefore you can use different models up to their respective limits simultaneously.
You can check your current rate limits and behavior in the [Claude Console](https://console.anthropic.com/settings/limits).

<Note>
  For long context requests (>200K tokens) when using the `context-1m-2025-08-07` beta header with Claude Sonnet 4, separate rate limits apply. See [Long context rate limits](#long-context-rate-limits) below.
</Note>

<Tabs>
  <Tab title="Tier 1">
    | Model                                                                                        | Maximum requests per minute (RPM) | Maximum input tokens per minute (ITPM) | Maximum output tokens per minute (OTPM) |
    | -------------------------------------------------------------------------------------------- | --------------------------------- | -------------------------------------- | --------------------------------------- |
    | Claude Opus 4.x<sup>\*</sup>                                                                 | 50                                | 30,000                                 | 8,000                                   |
    | Claude Sonnet 4                                                                              | 50                                | 30,000                                 | 8,000                                   |
    | Claude Sonnet 3.7                                                                            | 50                                | 20,000                                 | 8,000                                   |
    | Claude Sonnet 3.5 <br /> 2024-10-22 ([deprecated](/en/docs/about-claude/model-deprecations)) | 50                                | 40,000<sup>†</sup>                     | 8,000                                   |
    | Claude Sonnet 3.5 <br /> 2024-06-20 ([deprecated](/en/docs/about-claude/model-deprecations)) | 50                                | 40,000<sup>†</sup>                     | 8,000                                   |
    | Claude Haiku 3.5                                                                             | 50                                | 50,000<sup>†</sup>                     | 10,000                                  |
    | Claude Opus 3 ([deprecated](/en/docs/about-claude/model-deprecations))                       | 50                                | 20,000<sup>†</sup>                     | 4,000                                   |
    | Claude Haiku 3                                                                               | 50                                | 50,000<sup>†</sup>                     | 10,000                                  |
  </Tab>

  <Tab title="Tier 2">
    | Model                                                                                        | Maximum requests per minute (RPM) | Maximum input tokens per minute (ITPM) | Maximum output tokens per minute (OTPM) |
    | -------------------------------------------------------------------------------------------- | --------------------------------- | -------------------------------------- | --------------------------------------- |
    | Claude Opus 4.x<sup>\*</sup>                                                                 | 1,000                             | 450,000                                | 90,000                                  |
    | Claude Sonnet 4                                                                              | 1,000                             | 450,000                                | 90,000                                  |
    | Claude Sonnet 3.7                                                                            | 1,000                             | 40,000                                 | 16,000                                  |
    | Claude Sonnet 3.5 <br /> 2024-10-22 ([deprecated](/en/docs/about-claude/model-deprecations)) | 1,000                             | 80,000<sup>†</sup>                     | 16,000                                  |
    | Claude Sonnet 3.5 <br /> 2024-06-20 ([deprecated](/en/docs/about-claude/model-deprecations)) | 1,000                             | 80,000<sup>†</sup>                     | 16,000                                  |
    | Claude Haiku 3.5                                                                             | 1,000                             | 100,000<sup>†</sup>                    | 20,000                                  |
    | Claude Opus 3 ([deprecated](/en/docs/about-claude/model-deprecations))                       | 1,000                             | 40,000<sup>†</sup>                     | 8,000                                   |
    | Claude Haiku 3                                                                               | 1,000                             | 100,000<sup>†</sup>                    | 20,000                                  |
  </Tab>

  <Tab title="Tier 3">
    | Model                                                                                        | Maximum requests per minute (RPM) | Maximum input tokens per minute (ITPM) | Maximum output tokens per minute (OTPM) |
    | -------------------------------------------------------------------------------------------- | --------------------------------- | -------------------------------------- | --------------------------------------- |
    | Claude Opus 4.x<sup>\*</sup>                                                                 | 2,000                             | 800,000                                | 160,000                                 |
    | Claude Sonnet 4                                                                              | 2,000                             | 800,000                                | 160,000                                 |
    | Claude Sonnet 3.7                                                                            | 2,000                             | 80,000                                 | 32,000                                  |
    | Claude Sonnet 3.5 <br /> 2024-10-22 ([deprecated](/en/docs/about-claude/model-deprecations)) | 2,000                             | 160,000<sup>†</sup>                    | 32,000                                  |
    | Claude Sonnet 3.5 <br /> 2024-06-20 ([deprecated](/en/docs/about-claude/model-deprecations)) | 2,000                             | 160,000<sup>†</sup>                    | 32,000                                  |
    | Claude Haiku 3.5                                                                             | 2,000                             | 200,000<sup>†</sup>                    | 40,000                                  |
    | Claude Opus 3 ([deprecated](/en/docs/about-claude/model-deprecations))                       | 2,000                             | 80,000<sup>†</sup>                     | 16,000                                  |
    | Claude Haiku 3                                                                               | 2,000                             | 200,000<sup>†</sup>                    | 40,000                                  |
  </Tab>

  <Tab title="Tier 4">
    | Model                                                                                        | Maximum requests per minute (RPM) | Maximum input tokens per minute (ITPM) | Maximum output tokens per minute (OTPM) |
    | -------------------------------------------------------------------------------------------- | --------------------------------- | -------------------------------------- | --------------------------------------- |
    | Claude Opus 4.x<sup>\*</sup>                                                                 | 4,000                             | 2,000,000                              | 400,000                                 |
    | Claude Sonnet 4                                                                              | 4,000                             | 2,000,000                              | 400,000                                 |
    | Claude Sonnet 3.7                                                                            | 4,000                             | 200,000                                | 80,000                                  |
    | Claude Sonnet 3.5 <br /> 2024-10-22 ([deprecated](/en/docs/about-claude/model-deprecations)) | 4,000                             | 400,000<sup>†</sup>                    | 80,000                                  |
    | Claude Sonnet 3.5 <br /> 2024-06-20 ([deprecated](/en/docs/about-claude/model-deprecations)) | 4,000                             | 400,000<sup>†</sup>                    | 80,000                                  |
    | Claude Haiku 3.5                                                                             | 4,000                             | 400,000<sup>†</sup>                    | 80,000                                  |
    | Claude Opus 3 ([deprecated](/en/docs/about-claude/model-deprecations))                       | 4,000                             | 400,000<sup>†</sup>                    | 80,000                                  |
    | Claude Haiku 3                                                                               | 4,000                             | 400,000<sup>†</sup>                    | 80,000                                  |
  </Tab>

  <Tab title="Custom">
    If you're seeking higher limits for an Enterprise use case, contact sales through the [Claude Console](https://console.anthropic.com/settings/limits).
  </Tab>
</Tabs>

*<sup>\* - Opus 4.x rate limit is a total limit that applies to combined traffic across both Opus 4.0 and Opus 4.1.</sup>*

*<sup>† - Limit counts `cache_read_input_tokens` towards ITPM usage.</sup>*

### Message Batches API

The Message Batches API has its own set of rate limits which are shared across all models. These include a requests per minute (RPM) limit to all API endpoints and a limit on the number of batch requests that can be in the processing queue at the same time. A "batch request" here refers to part of a Message Batch. You may create a Message Batch containing thousands of batch requests, each of which count towards this limit. A batch request is considered part of the processing queue when it has yet to be successfully processed by the model.

<Tabs>
  <Tab title="Tier 1">
    | Maximum requests per minute (RPM) | Maximum batch requests in processing queue | Maximum batch requests per batch |
    | --------------------------------- | ------------------------------------------ | -------------------------------- |
    | 50                                | 100,000                                    | 100,000                          |
  </Tab>

  <Tab title="Tier 2">
    | Maximum requests per minute (RPM) | Maximum batch requests in processing queue | Maximum batch requests per batch |
    | --------------------------------- | ------------------------------------------ | -------------------------------- |
    | 1,000                             | 200,000                                    | 100,000                          |
  </Tab>

  <Tab title="Tier 3">
    | Maximum requests per minute (RPM) | Maximum batch requests in processing queue | Maximum batch requests per batch |
    | --------------------------------- | ------------------------------------------ | -------------------------------- |
    | 2,000                             | 300,000                                    | 100,000                          |
  </Tab>

  <Tab title="Tier 4">
    | Maximum requests per minute (RPM) | Maximum batch requests in processing queue | Maximum batch requests per batch |
    | --------------------------------- | ------------------------------------------ | -------------------------------- |
    | 4,000                             | 500,000                                    | 100,000                          |
  </Tab>

  <Tab title="Custom">
    If you're seeking higher limits for an Enterprise use case, contact sales through the [Claude Console](https://console.anthropic.com/settings/limits).
  </Tab>
</Tabs>

### Long context rate limits

When using Claude Sonnet 4 with the [1M token context window enabled](/en/docs/build-with-claude/context-windows#1m-token-context-window), the following dedicated rate limits apply to requests exceeding 200K tokens.

<Note>
  The 1M token context window is currently in beta for organizations in usage tier 4 and organizations with custom rate limits. The 1M token context window is only available for Claude Sonnet 4.
</Note>

<Tabs>
  <Tab title="Tier 4">
    | Maximum input tokens per minute (ITPM) | Maximum output tokens per minute (OTPM) |
    | -------------------------------------- | --------------------------------------- |
    | 1,000,000                              | 200,000                                 |
  </Tab>

  <Tab title="Custom">
    For custom long context rate limits for enterprise use cases, contact sales through the [Claude Console](https://console.anthropic.com/settings/limits).
  </Tab>
</Tabs>

<Tip>
  To get the most out of the 1M token context window with rate limits, use [prompt caching](/en/docs/build-with-claude/prompt-caching).
</Tip>

### Monitoring your rate limits in the Console

You can monitor your rate limit usage on the [Usage](https://console.anthropic.com/settings/usage) page of the [Claude Console](https://console.anthropic.com/).

In addition to providing token and request charts, the Usage page provides two separate rate limit charts. Use these charts to see what headroom you have to grow, when you may be hitting peak use, better undersand what rate limits to request, or how you can improve your caching rates. The charts visualize a number of metrics for a given rate limit (e.g. per model):

* The **Rate Limit - Input Tokens** chart includes:
  * Hourly maximum uncached input tokens per minute
  * Your current input tokens per minute rate limit
  * The cache rate for your input tokens (i.e. the percentage of input tokens read from the cache)
* The **Rate Limit - Output Tokens** chart includes:
  * Hourly maximum output tokens per minute
  * Your current output tokens per minute rate limit

## Setting lower limits for Workspaces

In order to protect Workspaces in your Organization from potential overuse, you can set custom spend and rate limits per Workspace.

Example: If your Organization's limit is 40,000 input tokens per minute and 8,000 output tokens per minute, you might limit one Workspace to 30,000 total tokens per minute. This protects other Workspaces from potential overuse and ensures a more equitable distribution of resources across your Organization. The remaining unused tokens per minute (or more, if that Workspace doesn't use the limit) are then available for other Workspaces to use.

Note:

* You can't set limits on the default Workspace.
* If not set, Workspace limits match the Organization's limit.
* Organization-wide limits always apply, even if Workspace limits add up to more.
* Support for input and output token limits will be added to Workspaces in the future.

## Response headers

The API response includes headers that show you the rate limit enforced, current usage, and when the limit will be reset.

The following headers are returned:

| Header                                        | Description                                                                                                                           |
| --------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| `retry-after`                                 | The number of seconds to wait until you can retry the request. Earlier retries will fail.                                             |
| `anthropic-ratelimit-requests-limit`          | The maximum number of requests allowed within any rate limit period.                                                                  |
| `anthropic-ratelimit-requests-remaining`      | The number of requests remaining before being rate limited.                                                                           |
| `anthropic-ratelimit-requests-reset`          | The time when the request rate limit will be fully replenished, provided in RFC 3339 format.                                          |
| `anthropic-ratelimit-tokens-limit`            | The maximum number of tokens allowed within any rate limit period.                                                                    |
| `anthropic-ratelimit-tokens-remaining`        | The number of tokens remaining (rounded to the nearest thousand) before being rate limited.                                           |
| `anthropic-ratelimit-tokens-reset`            | The time when the token rate limit will be fully replenished, provided in RFC 3339 format.                                            |
| `anthropic-ratelimit-input-tokens-limit`      | The maximum number of input tokens allowed within any rate limit period.                                                              |
| `anthropic-ratelimit-input-tokens-remaining`  | The number of input tokens remaining (rounded to the nearest thousand) before being rate limited.                                     |
| `anthropic-ratelimit-input-tokens-reset`      | The time when the input token rate limit will be fully replenished, provided in RFC 3339 format.                                      |
| `anthropic-ratelimit-output-tokens-limit`     | The maximum number of output tokens allowed within any rate limit period.                                                             |
| `anthropic-ratelimit-output-tokens-remaining` | The number of output tokens remaining (rounded to the nearest thousand) before being rate limited.                                    |
| `anthropic-ratelimit-output-tokens-reset`     | The time when the output token rate limit will be fully replenished, provided in RFC 3339 format.                                     |
| `anthropic-priority-input-tokens-limit`       | The maximum number of Priority Tier input tokens allowed within any rate limit period. (Priority Tier only)                           |
| `anthropic-priority-input-tokens-remaining`   | The number of Priority Tier input tokens remaining (rounded to the nearest thousand) before being rate limited. (Priority Tier only)  |
| `anthropic-priority-input-tokens-reset`       | The time when the Priority Tier input token rate limit will be fully replenished, provided in RFC 3339 format. (Priority Tier only)   |
| `anthropic-priority-output-tokens-limit`      | The maximum number of Priority Tier output tokens allowed within any rate limit period. (Priority Tier only)                          |
| `anthropic-priority-output-tokens-remaining`  | The number of Priority Tier output tokens remaining (rounded to the nearest thousand) before being rate limited. (Priority Tier only) |
| `anthropic-priority-output-tokens-reset`      | The time when the Priority Tier output token rate limit will be fully replenished, provided in RFC 3339 format. (Priority Tier only)  |

The `anthropic-ratelimit-tokens-*` headers display the values for the most restrictive limit currently in effect. For instance, if you have exceeded the Workspace per-minute token limit, the headers will contain the Workspace per-minute token rate limit values. If Workspace limits do not apply, the headers will return the total tokens remaining, where total is the sum of input and output tokens. This approach ensures that you have visibility into the most relevant constraint on your current API usage.
