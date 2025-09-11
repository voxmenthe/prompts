To run **Claude Code** with **Amazon Bedrock credentials** you only need to do two things:

1. Give Claude Code a **working AWS credential** (any method you already use for the CLI/SDK is fine).  
2. Tell Claude Code to **use Bedrock instead of the Anthropic API** by exporting a couple of environment variables.

Below are the three most common ways teams do this (pick ONE).

--------------------------------------------------------
1. Quickest – static access-key (laptop / sandbox)
--------------------------------------------------------
```bash
# 1.  Make sure the AWS CLI can already call Bedrock
aws bedrock list-foundation-models --region us-east-1

# 2.  Point Claude Code at Bedrock
export CLAUDE_CODE_USE_BEDROCK=1
export AWS_REGION=us-east-1   # or eu-central-1, us-west-2, etc.
# (optional) pick a model other than the default Sonnet 3.7
export ANTHROPIC_MODEL="us.anthropic.claude-3-5-sonnet-20241022-v2:0"

# 3.  Start the CLI
claude
```
That’s it—Claude Code will read your default AWS credential (`~/.aws/credentials`) and talk to Bedrock.

--------------------------------------------------------
2. Safer – AWS SSO (IAM Identity Center)
--------------------------------------------------------
```bash
# 1.  Create / login to an SSO profile
aws configure sso --profile claude-bedrock   # first-time only
aws sso login   --profile claude-bedrock

# 2.  Export the profile name so Claude Code uses it
export AWS_PROFILE=claude-bedrock
export CLAUDE_CODE_USE_BEDROCK=1
export AWS_REGION=us-east-1

# 3.  Launch
claude
```
No long-lived keys ever touch your disk; credentials are refreshed automatically while the SSO session is valid.

--------------------------------------------------------
3. Enterprise – OIDC → Cognito → temporary credentials
--------------------------------------------------------
If your company already uses Okta, Azure AD, Auth0, etc., the AWS Solutions team published a ready-made package that:

- Lets developers sign in with their corporate IdP  
- Exchanges the OIDC token for **temporary** Bedrock-scope credentials  
- Drops a local AWS profile named `ClaudeCode`  

Deploy once (CloudFormation + Cognito provided), then developers only run:

```bash
# one-time installer handed out by IT
./install-claude-bedrock.sh

# daily usage
export AWS_PROFILE=ClaudeCode   # created by installer
claude
```

Full instructions and source:  
https://github.com/aws-solutions-library-samples/guidance-for-claude-code-with-amazon-bedrock

--------------------------------------------------------
Check-list before you start
--------------------------------------------------------
1. In the AWS console open **Bedrock → Model access** and enable the **Anthropic** models you plan to use (Claude 3.5 Sonnet, Haiku, etc.).  
2. Your IAM user/role needs at least the policy `AmazonBedrockFullAccess` (or a tighter custom policy that allows `bedrock:InvokeModel` on the desired ARNs).  
3. Use a region that supports the model:  
   - **us-east-1** (N. Virginia) – all models  
   - **us-west-2** (Oregon) – most models  
   - **eu-central-1** (Frankfurt) – Sonnet & Haiku  

--------------------------------------------------------
Common env-vars you can tweak
--------------------------------------------------------
| Variable | Purpose | Default |
|----------|---------|---------|
| `CLAUDE_CODE_USE_BEDROCK=1` | Switch from Anthropic API to Bedrock | — |
| `AWS_REGION` | Region for Bedrock calls | — |
| `ANTHROPIC_MODEL` | Big model ID | `us.anthropic.claude-3-7-sonnet-20250219-v1:0` |
| `ANTHROPIC_SMALL_FAST_MODEL` | Small model ID | `us.anthropic.claude-3-5-haiku-20241022-v1:0` |
| `DISABLE_PROMPT_CACHING=1` | Turn off prompt caching | caching ON by default |

--------------------------------------------------------
Quick test
--------------------------------------------------------
After `claude` starts you should see the welcome banner **without** being asked for an “Anthropic API key”.  
Run `/cost` inside Claude Code; if the table shows **“Amazon Bedrock”** in the provider column you’re connected.

That’s all—happy coding!