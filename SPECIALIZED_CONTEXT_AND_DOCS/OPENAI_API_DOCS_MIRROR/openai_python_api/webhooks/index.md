# Webhooks


Types:

```python
from openai.types.webhooks import (
    BatchCancelledWebhookEvent,
    BatchCompletedWebhookEvent,
    BatchExpiredWebhookEvent,
    BatchFailedWebhookEvent,
    EvalRunCanceledWebhookEvent,
    EvalRunFailedWebhookEvent,
    EvalRunSucceededWebhookEvent,
    FineTuningJobCancelledWebhookEvent,
    FineTuningJobFailedWebhookEvent,
    FineTuningJobSucceededWebhookEvent,
    RealtimeCallIncomingWebhookEvent,
    ResponseCancelledWebhookEvent,
    ResponseCompletedWebhookEvent,
    ResponseFailedWebhookEvent,
    ResponseIncompleteWebhookEvent,
    UnwrapWebhookEvent,
)
```

Methods:

- <code>client.webhooks.<a href="./src/openai/resources/webhooks.py">unwrap</a>(payload, headers, \*, secret) -> UnwrapWebhookEvent</code>
- <code>client.webhooks.<a href="./src/openai/resources/webhooks.py">verify_signature</a>(payload, headers, \*, secret, tolerance) -> None</code>
