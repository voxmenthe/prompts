LiveKit docs › Getting started › Cloud › Phone numbers

---

# LiveKit Phone Numbers

> How to purchase and configure phone numbers directly through LiveKit.

## Overview

LiveKit Phone Numbers lets you purchase and manage US phone numbers for voice applications. It provides the telephony infrastructure and phone number inventory, without requiring separate SIP trunk configuration. Buy local or toll-free numbers directly through LiveKit and assign them to voice agents using dispatch rules.

> ℹ️ **Inbound calling only**
> 
> LiveKit Phone Numbers currently only supports inbound calling. Support for outbound calls is coming soon.

- **Buy numbers directly**: Select local or toll-free US numbers for inbound calling with your preferred area code.
- **Streamlined setup**: Purchase phone numbers and configure voice agents without SIP trunk complexity.
- **High-definition (HD) voice**: Ensure clear, professional audio quality on all calls, from agent dialogue to hold music.
- **Unified management**: Use LiveKit Cloud to procure and manage numbers, configure dispatch rules, and review call metrics and logs.

You can manage your phone numbers using the [LiveKit Cloud dashboard](https://cloud.livekit.io/projects/p_/telephony/phone-numbers), [LiveKit CLI](#cli-reference), or the [Phone Numbers APIs](https://docs.livekit.io/sip/phone-numbers-api.md).

## Setting up a LiveKit phone number

To set up a LiveKit phone number, you need to purchase a phone number and assign it to a dispatch rule. The following steps guide you through the process.

### Step 1: Search for an available number

Search for available phone numbers by country and area code.

**LiveKit Cloud**:

Search for available numbers by area code:

1. Sign in to the **LiveKit Cloud** [dashboard](https://cloud.livekit.io/).
2. Select **Telephony** → [**Phone Numbers**](https://cloud.livekit.io/projects/p_/telephony/phone-numbers).
3. Select **Buy a number**.
4. Select the search icon and enter an area code.

---

**LiveKit CLI**:

Search for phone numbers in the United States with area code 415:

```shell
lk number search --country-code US --area-code 415

```

### Step 2: Buy a number

Select an available phone number and purchase it.

**LiveKit Cloud**:

After you [search for available numbers](#search), purchase the number by clicking **Buy** in the row with the number you want to purchase:

1. Select **Buy** for the number you want to purchase.
2. Select **Confirm purchase**.

---

**LiveKit CLI**:

To buy the number `+14155550100`, run the following command:

```shell
lk number purchase --numbers +14155550100

```

### Step 3: Assign the number to a dispatch rule

Assign the number to a dispatch rule. LiveKit recommends using [explicit dispatch](https://docs.livekit.io/agents/worker/agent-dispatch.md#explicit-dispatch) for agents that receive inbound calls. Define the agent you want to respond to calls to a number in the dispatch rule. To learn more, see [Dispatch from inbound SIP calls](https://docs.livekit.io/agents/worker/agent-dispatch.md#dispatch-from-inbound-sip-calls).

**LiveKit Cloud**:

After you successfully purchase a phone number, you can select **Options** to assign or create a dispatch rule for the number. Otherwise, use the following steps to assign a dispatch rule:

1. Navigate to the [Phone Numbers page](https://cloud.livekit.io/projects/p_/telephony/phone-numbers) and find the number you want to assign a dispatch rule to.
2. Select the more menu (**⋮**) and select **Assign dispatch rule**.
3. Select the dispatch rule you want to assign to the number.
4. Select **Save**.

---

**LiveKit CLI**:

For example, to assign a phone number to a dispatch rule, replace the `<PHONE_NUMBER_ID>` and `<DISPATCH_RULE_ID>` placeholders, and run the following command:

```shell
lk number update --id <PHONE_NUMBER_ID> --sip-dispatch-rule-id <DISPATCH_RULE_ID>

```

> ℹ️ **Find your phone number ID**
> 
> You can find your phone number ID by listing all phone numbers using the `lk number list` command.

### Create an agent that responds to inbound calls

Follow the [Voice AI quickstart](https://docs.livekit.io/agents/start/voice-ai.md) to create an agent. Start your agent and call your phone number.

## Considerations

The following limitations and considerations apply to LiveKit Phone Numbers:

- Available only in the US. Support for additional countries is coming in a future release.
- Only inbound calling is supported. Support for outbound calling is coming in a future release.
- Forwarding calls using the `TransferSipParticipant` API is not yet supported.
- If you release a phone number before the end of the month, you are still billed for the entirety of the month. For details on pricing, see the [LiveKit Phone Numbers pricing page](https://livekit.io/pricing/phone-numbers).

## CLI reference

The LiveKit CLI provides phone number management commands for searching, purchasing, and managing phone numbers for your SIP applications. Prefix all phone number commands with `lk number`.

For instructions on installing the CLI, see the LiveKit CLI [Getting started](https://docs.livekit.io/home/cli.md) guide.

```shell
lk number [command] [command options]

```

> 🔥 **CLI version requirement**
> 
> Update the CLI regularly to ensure you have the latest version. You must have an up-to-date CLI to manage phone numbers. See [Update the CLI](https://docs.livekit.io/home/cli.md#updates) for instructions.

### Search

Search available phone numbers in inventory for purchase.

```shell
lk number search [options]

```

Options for `search`:

- `--country-code STRING`: Filter by country code (for example, "US," "CA"). Required.
- `--area-code STRING`: Filter by area code (for example, "415").
- `--limit INT`: Maximum number of results. Default: 50.
- `--json, -j`: Output as JSON. Default: false.

#### Examples

Search for phone numbers in the US with area code 415:

```shell
lk number search --country-code US --area-code 415 --limit 10

```

Search for phone numbers with JSON output:

```shell
lk number search --country-code US --area-code 415 --json

```

### Purchase

Purchase phone numbers from inventory.

```shell
lk number purchase [options]

```

Options for `purchase`:

- `--numbers STRING`: Phone numbers to purchase (for example, "+16505550010"). Required.
- `--sip-dispatch-rule-id STRING`: SIP dispatch rule ID to apply to all purchased numbers.

#### Examples

Purchase a single phone number:

```shell
lk number purchase --numbers +16505550010

```

### List

List phone numbers for a project.

```shell
lk number list [options]

```

Options for `list`:

- `--limit INT`: Maximum number of results. Default: 50.
- `--status STRING`: Filter by statuses: `active`, `pending`, `released`. You can specify multiple statuses by repeating the flag.
- `--sip-dispatch-rule-id STRING`: Filter by SIP dispatch rule ID.
- `--json, -j`: Output as JSON. Default: false.

#### Examples

List all `active`phone numbers:

```shell
lk number list

```

List `active` and `released` phone numbers:

```shell
lk number list --status active --status released

```

### Get

Get details for a specific phone number.

```shell
lk number get [options]

```

Options for `get`:

- `--id STRING`: Phone number ID for direct lookup.
- `--number STRING`: Phone number string for lookup (for example, "+16505550010").

**Note**: you must specify either `--id` or `--number`.

#### Examples

Get phone number by ID:

```shell
lk number get --id <PHONE_NUMBER_ID>

```

Get phone number by number string:

```shell
lk number get --number +16505550010

```

### Update

Update a phone number configuration.

```shell
lk number update [options]

```

Options for `update`:

- `--id STRING`: Phone number ID for direct lookup.
- `--number STRING`: Phone number string for lookup.
- `--sip-dispatch-rule-id STRING`: SIP dispatch rule ID to assign to the phone number.

**Note**: you must specify either `--id` or `--number`.

#### Examples

Update phone number dispatch rule by ID:

```shell
lk number update --id <PHONE_NUMBER_ID> --sip-dispatch-rule-id <DISPATCH_RULE_ID>

```

Update phone number dispatch rule by number:

```shell
lk number update \
  --number +16505550010 \
  --sip-dispatch-rule-id <DISPATCH_RULE_ID>

```

### Release

Release phone numbers by ID or phone number string.

```shell
lk number release [options]

```

Options for `release`:

- `--ids STRING`: Phone number ID for direct lookup.
- `--numbers STRING`: Phone number string for lookup.

**Note**: you must specify either `--ids` or `--numbers`.

#### Examples

Release phone numbers by ID:

```shell
lk number release --ids <PHONE_NUMBER_ID>

```

Release phone numbers by number strings:

```shell
lk number release --numbers +16505550010

```

## Additional resources

The following topics provide more information on managing LiveKit Phone Numbers and LiveKit SIP.

- **[Dispatch rules](https://docs.livekit.io/sip/dispatch-rules.md)**: Create dispatch rules to determine how callers to your LiveKit Phone Number are dispatched to rooms.

- **[Phone Number APIs](https://docs.livekit.io/sip/phone-numbers-api.md)**: Reference for the phone number management commands in the LiveKit CLI.

---


For the latest version of this document, see [https://docs.livekit.io/sip/cloud/phone-numbers.md](https://docs.livekit.io/sip/cloud/phone-numbers.md).

To explore all LiveKit documentation, see [llms.txt](https://docs.livekit.io/llms.txt).