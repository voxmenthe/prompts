<!-- Source: https://docs.anthropic.com/en/api/admin-api/organization/get-me -->

# Get Current Organization

get/v1/organizations/me

Retrieve information about the organization associated with the authenticated API key.

##### ReturnsExpand Collapse 

Organization = object { id, name, type } 

id: string

ID of the Organization.

formatuuid

name: string

Name of the Organization.

type: "organization"

Object type.

For Organizations, this is always `"organization"`.

Accepts one of the following:

"organization"

Get Current Organization
[code]
    curl https://api.anthropic.com/v1/organizations/me \
        -H "X-Api-Key: $ANTHROPIC_ADMIN_API_KEY"
[/code]
[code]
    {
      "id": "12345678-1234-5678-1234-567812345678",
      "name": "Organization Name",
      "type": "organization"
    }
[/code]

##### Returns Examples
[code]
    {
      "id": "12345678-1234-5678-1234-567812345678",
      "name": "Organization Name",
      "type": "organization"
    }
[/code]