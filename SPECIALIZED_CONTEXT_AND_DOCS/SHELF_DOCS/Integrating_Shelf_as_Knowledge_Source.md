# Integrating Shelf as a Knowledge Source for Third-Party AI Applications

## **Overview**

This use case demonstrates how to leverage Shelf's knowledge management platform as a comprehensive content source for chatbots and generative AI (GenAI) initiatives. By following this implementation path, organizations can enrich their AI applications with accurate, up-to-date information stored in Shelf.

## **Business Challenge**

Organizations need to ensure their AI applications have access to reliable, current information to provide accurate responses to users. Maintaining knowledge consistency between internal knowledge bases and AI applications is critical for customer service quality and operational efficiency.

## **Solution**

Integrate Shelf's content repository with AI applications through a systematic API-based approach that extracts various content types (Wikis, Decision Trees, and file attachments) and prepares them for AI consumption.

## **Implementation Steps**

### **1. Authentication Setup**

**Action**: _Generate an API token with appropriate permissions to access Shelf content._

* Navigate to the Shelf admin portal to create an API token
* Assign necessary read permissions for content access
* Securely store the token in your application environment.

**Documentation Reference**: [_Managing API Tokens Guide_](https://docs.shelf.io/dev-portal/api-essentials/managing-api-tokens-guide)_._

### **2. Content Discovery** <a href="#id-2.-content-discovery" id="id-2.-content-discovery"></a>

{% hint style="warning" %}
Before starting any content discovery actions, you need to know that the best format for interacting with Gen AI applications or chatbots is Markdown <img src="https://3746171570-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2F7yKmgyutbCyeUlzPhhYi%2Fuploads%2FwuJELSZqASIELj0gNz62%2Fmkdown.png?alt=media&#x26;token=f74d8d81-b04f-4c77-be3c-5174ac848c9c" alt="" data-size="line">. If you are trying to get the content, Shelf will transform it to Markdown regardless of the content's underlying format.
{% endhint %}

**Action**: _Retrieve a catalog of available content items (Gems) from Shelf to be used as knowledge sources._

* Identify content item types you have on your Shelf account by running the dedicated API query

## Get content types

> An endpoint for getting all content types in the account\
>

```json
{"openapi":"3.1.0","info":{"title":"Shelf Content Types API","version":"1.0.0"},"servers":[{"description":"US region","url":"https://api.shelf.io"},{"description":"EU region","url":"https://api.shelf-eu.com"},{"description":"CA region","url":"https://api.shelf-ca.com"}],"security":[{"auth_token":[]}],"components":{"securitySchemes":{"auth_token":{"type":"apiKey","description":"Shelf API Token. See [Create API Token](https://docs.shelf.io/dev-portal/authentication/create-api-token) for more details.","in":"header","name":"Authorization"}}},"paths":{"/content-types/types":{"get":{"operationId":"getContentTypes","summary":"Get content types","description":"An endpoint for getting all content types in the account\n","responses":{"200":{"description":"Success - Content types in the account","content":{"application/json":{"schema":{"type":"object","required":["types"],"properties":{"types":{"type":"array","items":{"allOf":[{"type":"object","allOf":[{"type":"object","description":"Content type icons","properties":{"iconS3URL":{"type":"string"},"iconURL":{"type":"string"}}}],"required":["id","accountId","name","isPublished","isDefault","isLocked","createdAt","updatedAt","createdBy","updatedBy"],"properties":{"description":{"type":"string"},"accountId":{"type":"string"},"createdAt":{"type":"string","format":"date-time"},"createdBy":{"type":"string"},"id":{"type":"string"},"isDefault":{"type":"boolean"},"isLocked":{"type":"boolean"},"isPublished":{"type":"boolean"},"key":{"type":"string"},"name":{"type":"string"},"updatedAt":{"type":"string","format":"date-time"},"updatedBy":{"type":"string"}}},{"type":"object","description":"Content type user info","required":["createdByUsername","updatedByUsername"],"properties":{"createdByUsername":{"type":"string"},"updatedByUsername":{"type":"string"}}}]}}}}}}},"403":{"description":"Forbidden","content":{"application/json":{"schema":{"type":"object","required":["error"],"properties":{"error":{"type":"object","description":"API Error","required":["status","message","code"],"properties":{"code":{"type":"string","description":"ERROR_CODE"},"detail":{"description":"Detail info","oneOf":[{"type":"string"},{"type":"array","items":{"type":"object","properties":{"dataPath":{"type":"string"},"keyword":{"type":"string"},"message":{"type":"string"},"params":{"type":"object"},"schemaPath":{"type":"string"}}}}]},"message":{"type":"string","description":"Error message"},"status":{"type":"number","description":"Status code"}}}}}}}}},"tags":["Types"]}}}}
```

{% hint style="warning" %}
From the response you get after running the above query, the important parameter is `id`.  This parameter's value shows the Shelf content type identifier that you will need further.
{% endhint %}

<figure><img src="https://3746171570-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2F7yKmgyutbCyeUlzPhhYi%2Fuploads%2FycIxEyR8ndptTq4KG5oC%2FUseCase-ContentItemType-Response.png?alt=media&#x26;token=2f1db63a-b9cb-40fd-a4b0-75e7230a4fdb" alt=""><figcaption></figcaption></figure>

You can then extract the retrieved content type ids and save them for further use.

<details>

<summary>Example of Retrieved Content Type Ids</summary>

```json
[
  { "id": "01HKW4F4JVG3NWBS2DK7KGGZD1", "key": "Article",       "name": "Article" },
  { "id": "01HKW4F4NDAWJ872HDHRRQARN7", "key": "Audio",         "name": "Audio" },
  { "id": "01HKW4F4S1TBX124B9003JHVN9", "key": "Decision Tree", "name": "Decision Tree" },
  { "id": "01HKW4F4H0EVQ4Q1EMW0MMANY0", "key": "Document",      "name": "Document" },
  { "id": "01HKW4F4VGRQS0427YZ84QXETS", "key": "FAQ",           "name": "FAQ" },
  { "id": "01HKW4F4J3AT5R8C20Z223BBKX", "key": "Image",         "name": "Image" },
  { "id": "01HKW4F4KJ60W5Z4M5662C4BG4", "key": "Bookmark",      "name": "Link" },
  { "id": "01HKW4F4PJ6Q5VS84QQ8BK43PH", "key": "Organization",  "name": "Organization" },
  { "id": "01HKW4F4Q7TMAHE7MS1Z6QEK4J", "key": "Person",        "name": "Person" },
  { "id": "01HKW4F4WSB4GSP7CGZMY7MWF9", "key": "Post",          "name": "Post" },
  { "id": "01HKW4F4MAHNJGYSBHM7320WDS", "key": "Video",         "name": "Video" },
  { "id": "01HKW4F4MQHT7C1JZJP9SGMAFS", "key": "Note",          "name": "Wiki Page" }
]
```

</details>

where&#x20;

**`id`**: Use for API filtering (e.g. `"contentTypeId"`)

**`key`**: Short code/slug for the type

**`name`**: Human-friendly display name.



* Make a request to the content item list endpoint specified below

## List Content Items

> Used to list and paginate over CIL content items. Does not apply vector search.\
> Useful to pull content synced from various sources via a single API.\
> \
> Permissions: users will be able to search only across collections they have access to.\
>

```json
{"openapi":"3.0.0","info":{"title":"Shelf Content Integration Layer API","version":"1.0.0"},"tags":[{"name":"Content Items","description":"There two endpoints: \"List Content Items\" and \"Search Content Items\". They are designed to be used in different scenarios. If you want to sync content from Shelf into your database - \"List CIL Content API\" is the right choice.\n\n**Search CIL Content API**\n- **Search event submission**: When using the Search API, a search event is submitted, which is then processed by the Content Insights pipeline. This could be useful for tracking and analyzing user search behavior over time.\n- **Appear on Search Queries reports**: The data from the search event will eventually appear on Search Queries reports, providing insights into the types of queries that users are performing.\n- **Vector search**: It applies vector search to find the most relevant content items based on the search term.\n\n**List CIL Content API**\n- **Data listing without search events**: This API lists CIL content items without creating a search event or contributing to the Content Insights pipeline.\n- **No `searchResultsUrl`**: The response from the List API does not include a `searchResultsUrl`, which would otherwise be used to link directly to the search results.\n- **Paginate over content items**: It's designed to paginate over content items rather than perform a relevance-based search.\n\nSo, to decide which API to use, consider whether you need insights and tracking of user search behavior (use the \"Search CIL Content API\") or whether you simply need to list content items without additional analytics (use the \"List CIL Content API\").\n\n**Pagination Mechanics**\n1. **Initial Request**: The initial API request includes a pagination parameter `size`, which defines the number of items to be returned in a single response. Additionally, clients can specify a `from` parameter to skip a certain number of items from the start, effectively controlling the starting point of the data retrieval.\n\n2. **Iterative Pagination**: To navigate through the content items, the API uses a `nextToken` parameter. After the initial request, the API response includes a `nextToken` value, which is a unique identifier for the next page of results.\n\n3. **Subsequent Requests**: For subsequent requests, clients include this `nextToken` in their request to retrieve the next set of items. This token ensures that each request fetches a new page of results, starting exactly where the last one ended.\n\n4. **Page Info**: Each response also contains a `pageInfo` object, providing metadata about the pagination state. This includes information like `currentPage`, `hasNextPage`, `totalPages`, and `totalResultsCount`, giving a comprehensive overview of the pagination context.\n\n5. **Efficient Data Retrieval**: This pagination system allows for efficient data retrieval, especially in scenarios where the total number of content items is large. Clients can systematically fetch all items in a controlled and manageable manner."}],"servers":[{"url":"https://api.shelf.io","description":"US region"},{"url":"https://api.shelf-eu.com","description":"EU region"},{"url":"https://api.shelf-ca.com","description":"CA region"}],"security":[{"auth_token":[]}],"components":{"securitySchemes":{"auth_token":{"description":"Shelf API Token. See [Create API Token](https://docs.shelf.io/dev-portal/authentication/create-api-token) for more details.","type":"apiKey","in":"header","name":"Authorization"}}},"paths":{"/cil-content/items/list":{"post":{"tags":["Content Items"],"summary":"List Content Items","description":"Used to list and paginate over CIL content items. Does not apply vector search.\nUseful to pull content synced from various sources via a single API.\n\nPermissions: users will be able to search only across collections they have access to.\n","operationId":"listCILContentItems","requestBody":{"content":{"application/json":{"schema":{"allOf":[{"type":"object","properties":{"fieldsFilters":{"type":"object","properties":{"contentTypeId":{"type":"array","items":{"type":"string"}},"shelf_categoryId":{"type":"array","minItems":0,"maxItems":20,"items":{"type":"string"}},"shelf_categoryOp":{"type":"string","enum":["and","or"],"default":"or","description":"Specifies how to filter content by ids of categories if multiple ids provided. \"and\" - matching content should have all provided category ids. \"or\" - matching content should have any one of provided category ids."},"shelf_categoryIdToExclude":{"type":"array","description":"Filter content by ids of categories it does not belong to.","minItems":0,"maxItems":20,"items":{"type":"string"}},"shelf_reviewStatus":{"description":"Filter content by review status. Can be provided only if Content Review feature is enabled.","type":"array","minLength":0,"maxLength":3,"items":{"type":"string","enum":["up-to-date","out-of-date","none"]}},"ownerId":{"type":"array","items":{"type":"string"}},"source":{"type":"array","items":{"type":"string"}},"ratingMoreThan":{"type":"array","minItems":1,"maxItems":1,"items":{"type":"number","minimum":0,"maximum":4}},"type":{"type":"array","items":{"type":"string"}}}}}},{"type":"object","properties":{"query":{"type":"string","description":"Filters content by searchable text, like `title`, `description`, `text` and `tags` fields."},"queryFields":{"type":"array","description":"Filters content by specific fields by query provided. If not specified, all searchable fields are used.","minItems":1,"maxItems":5,"items":{"oneOf":[{"type":"string","enum":["title"]},{"type":"string","pattern":"^fields\\\\.shelf_customField_.+$"}]}},"searchLanguage":{"type":"string","description":"Filters content by language code. Specify `any` to search across all languages.","minLength":2,"maxLength":5},"tags":{"type":"array","description":"Filters content by tags. Uses `tagsOp` parameter to specify how to treat multiple tags.","minItems":0,"maxItems":200,"items":{"type":"string","minLength":1,"maxLength":40}},"tagsOp":{"type":"string","enum":["and","or"],"default":"or","description":"Specifies how to filter content by tags if multiple tags provided. \"and\" - matching content should have all provided tags. \"or\" - matching content should have any one of provided tags."},"tagsToExclude":{"type":"array","description":"Filter content by tags it does not belong to.","minItems":0,"maxItems":200,"items":{"type":"string","minLength":1,"maxLength":40}},"collectionIds":{"type":"array","description":"Filter content by ids of collections it belongs to.","minItems":0,"maxItems":50,"items":{"type":"string","minLength":6,"maxLength":64}},"collectionIdsToExclude":{"type":"array","description":"Filter content by ids of collections it does not belong to.","minItems":0,"maxItems":50,"items":{"type":"string","minLength":6,"maxLength":64}},"connectorIds":{"type":"array","description":"Filter content by ids of connectors it was synchronized with.","minItems":0,"maxItems":50,"items":{"type":"string","minLength":6,"maxLength":64}},"createdAfter":{"description":"Filter content that was created starting from the given date. Creation date from source system is used.","type":"string","format":"date"},"createdBefore":{"description":"Filter content that was created up to the given date. Creation date from source system is used.","type":"string","format":"date"},"updatedAfter":{"description":"Filter content that was updated after or equal the given date. Update date from source system is used.","type":"string","format":"date"},"updatedBefore":{"description":"Filter content that was updated before or equal the given date. Update date from source system is used.","type":"string","format":"date"},"updatedAfterStrict":{"description":"Filter content that was updated strictly after the given date. Update date from source system is used.","type":"string","format":"date"},"updatedBeforeStrict":{"description":"Filter content that was updated strictly before the given date. Update date from source system is used.","type":"string","format":"date"},"parentId":{"description":"Filter content that is in given location","type":"string"},"includeDeepResults":{"description":"If specified `parentId` would be also searched through grand parent locations","type":"boolean"},"includePrivateContent":{"description":"If specified private content would be included in search results","type":"boolean","default":true},"idsToExclude":{"description":"Filter out specific external ids","type":"array","minItems":0,"maxItems":1000,"items":{"type":"string","minLength":6,"maxLength":64}},"idsToInclude":{"description":"Filter specific external ids","type":"array","minItems":0,"maxItems":1000,"items":{"type":"string","minLength":6,"maxLength":64}}}},{"type":"object","properties":{"sortBy":{"type":"string","enum":["TITLE","CREATED_DATE","UPDATED_DATE","RELEVANCE","VIEWS","RATING","LAST_REVIEWED_DATE","SCHEDULED_REVIEW_BY","REVIEW_FREQUENCY"],"description":"Specify how (by what) to sort search results. Review sorting may be provided only if Content Review feature is enabled."},"sortOrder":{"type":"string","description":"Specify either ascending or descnding sort order to apply to `sortBy` parameter.","enum":["ASC","DESC"]}}},{"type":"object","properties":{"from":{"type":"integer","minimum":0,"maximum":10000,"description":"Filters out content by numbered position in the search results, for example if the value is `3`, then the first item of the search will be the 4th one, meaning that the first three were skipped."},"size":{"type":"integer","minimum":0,"maximum":1000,"description":"Using this parameter it's possible to filter out the amount of gems in response."},"nextToken":{"type":"string","description":"Next token should be used to retrieve next portion of the results (next page). Next token from the response of this API can be used correctly ONLY if filters & sorting in new request have not changed comparing to the request where next token was returned. If filters or sorting have changed, use this API without next token, and after retrieving first page with changed filters or sorting next token will be returned by this API and can be used. Use only nextToken returned by this API, do not change it."}}}]}}}},"responses":{"200":{"description":"Page of CIL Items","content":{"application/json":{"schema":{"type":"object","required":["items","pageInfo"],"properties":{"items":{"type":"array","items":{"allOf":[{"type":"object","required":["fields"],"properties":{"id":{"type":"string","description":"Content item ID (internal to CIL)"},"accountId":{"description":"ID of account","type":"string","minLength":6,"maxLength":64},"internalId":{"type":"string","description":"Content item ID (internal to CIL)"},"externalId":{"type":"string","description":"External content item ID, provided by CIL Connector, corresponding to id in source system"},"title":{"type":"string","description":"Content item title"},"createdAt":{"type":"string","format":"date","description":"Date when content was first created in CIL"},"updatedAt":{"type":"string","format":"date","description":"Date when content was last updated in CIL"},"connectorId":{"type":"string","description":"Connector ID which synced this content item to CIL"},"externalURL":{"type":"string","description":"Url pointing to content in source system"},"lang":{"type":"string","description":"Language code of content item"},"lastViewedAt":{"type":"string","format":"date","description":"Date when content was last viewed at"},"originalCreatedAt":{"type":"string","format":"date","description":"Date when content was first created in source system"},"originalUpdatedAt":{"type":"string","format":"date","description":"Date when content was last updated in source system"},"tags":{"type":"array","description":"List of tags for this content item","items":{"type":"string"}},"collectionIds":{"type":"array","description":"List of collection IDs this content item belongs to in external system","items":{"type":"string","minLength":1,"maxLength":512,"description":"Collection ID from external system"}},"internalCollectionIds":{"type":"array","description":"List of internal collection IDs this content item belongs to in CIL","items":{"type":"string","minLength":26,"maxLength":26}},"locations":{"description":"List of locations of this content item belongs to an external system","type":"array","items":{"type":"object","required":["grandParentIds","collectionId"],"properties":{"grandParentIds":{"type":"array","items":{"type":"string","minLength":6,"maxLength":64}},"collectionId":{"type":"string","minLength":1,"maxLength":512,"description":"Collection ID from external system"},"parentId":{"type":"string","minLength":6,"maxLength":64}}}},"iconURL":{"type":"string","description":"Icon url for this content item. Present if this content item corresponds to Shelf KMS gem."},"fields":{"type":"array","items":{"type":"object","required":["key","value","type","label"],"properties":{"key":{"type":"string"},"value":{"type":"string"},"type":{"type":"string"},"label":{"type":"string"},"data":{"type":"object","additionalProperties":true}}}}}},{"type":"object","properties":{"description":{"type":"string","description":"CIL item description"},"attachments":{"type":"array","description":"List of attachments for this content item","items":{"type":"object","description":"Attachments","properties":{"attachmentId":{"type":"string","description":"Attachment ID"},"extension":{"type":"string","enum":["pdf","docx","txt","md","html","pptx"],"description":"The file extension of the attachment"},"converted":{"type":"array","description":"An array of objects, each representing a converted file format available for the attachment","items":{"type":"object","properties":{"attachmentId":{"type":"string","description":"Attachment ID"},"extension":{"type":"string","description":"The file extension of the attachment","enum":["md","txt","html","json"]}}}}}}}}}]}},"pageInfo":{"allOf":[{"type":"object","description":"This object holds information about pagination. Use it to know how to iterate over array of results","properties":{"currentPage":{"type":"number","minimum":1,"maximum":100,"default":1},"hasNextPage":{"type":"boolean"},"hasPreviousPage":{"type":"boolean"},"pageSize":{"type":"number"},"totalPages":{"type":"number","minimum":1,"maximum":100},"totalResultsCount":{"type":"number","minimum":0,"maximum":999999}}},{"type":"object","properties":{"nextToken":{"type":"string"}}}]}}}}}},"400":{"description":"Bad Request","content":{"application/json":{"schema":{"oneOf":[{"type":"object","title":"API Error with code","required":["error"],"properties":{"error":{"description":"API Error with code","type":"object","required":["status","message","code"],"properties":{"status":{"type":"number","description":"Status code"},"message":{"type":"string","description":"Error message"},"code":{"type":"string","description":"Error code"}}}}},{"type":"object","title":"Bad request payload (AJV)","required":["error"],"properties":{"error":{"description":"API Error With detail","type":"object","required":["status","message","detail"],"properties":{"status":{"type":"number","description":"Status code"},"message":{"type":"string","description":"Error message"},"detail":{"description":"Detail info","type":"array","items":{"type":"object","properties":{"keyword":{"type":"string"},"instancePath":{"type":"string"},"schemaPath":{"type":"string"},"params":{"type":"object"},"message":{"type":"string"}}}}}}}}]}}}},"403":{"description":"Forbidden","content":{"application/json":{"schema":{"type":"object","title":"API Error with code","required":["error"],"properties":{"error":{"description":"API Error with code","type":"object","required":["status","message","code"],"properties":{"status":{"type":"number","description":"Status code"},"message":{"type":"string","description":"Error message"},"code":{"type":"string","description":"Error code"}}}}}}}}}}}}}
```

You can also filter the content list by the needed type(s) only, using the specific content type id—`contentTypeId`—in your API query body.

<details>

<summary>Example of Content Type Filtering in Query JSON Payload</summary>

```json
{
  "fieldsFilters": {
    "contentTypeId": ["01HKW4F4H0EVQ4Q1EMW0MMANY0"]
  },
  "size": 10
}
```

</details>

<figure><img src="https://3746171570-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2F7yKmgyutbCyeUlzPhhYi%2Fuploads%2Fhcraj2BzPlfj34ZF4LVq%2FUseCase-ContentList-FilterByType.png?alt=media&#x26;token=b687dd8e-f55e-4041-a881-bd6494d0cef1" alt=""><figcaption></figcaption></figure>



If the request returns the <mark style="color:green;">`200`</mark> (_success_) response, in this response, `"externalId": "56b32d3d-4338-4ad6-aeb3-53220f94e334"` represents the **Gem ID** of a content item in Shelf KMS, while other parameters and their values show what type this content item is of, where it is located, what attachment(s) it has (if any), and other valuable metadata of this content item.&#x20;

{% hint style="info" %}
The `externalId` parameter is a unique identifier of the content item that allows accessing this item from external systems or platforms; meanwhile, the `internalId` and `id` parameters are for identifying content items internally, in Shelf.
{% endhint %}

<figure><img src="https://3746171570-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2F7yKmgyutbCyeUlzPhhYi%2Fuploads%2FcHkzOpYggU6Ocxauq63I%2FUseCase-ContentItemId_External.png?alt=media&#x26;token=db407952-0667-42d2-9aa4-d2b75fc67763" alt=""><figcaption></figcaption></figure>

If you open this content item in your browser, you can easily find its id in the URL address.

<figure><img src="https://3746171570-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2F7yKmgyutbCyeUlzPhhYi%2Fuploads%2FkjJrrmi2UNetTM9vaMAJ%2FUseCase-ContentItemId_External-url.png?alt=media&#x26;token=21695446-c4f4-4dee-86ec-2f61445a0ec9" alt=""><figcaption></figcaption></figure>

{% hint style="warning" %}
You are able to search only across collections you have access to. So, if you need to cover content stored in multiple collections, make sure first you have access to such collections.
{% endhint %}

* Once you have retrieved IDs of the content, you can filter the content based on relevance to your AI application needs
* Finally, when you have filtered content item IDs, save them for further processing and using with your AI applications. For example, you can save them in the following format, which can help easily understand what content item it is.

<details>

<summary>Example of Retrieved Content Item Ids</summary>

```json
[
  {
    "externalId": "25a0845e-2d5b-48ba-aacf-94e21fb6a70b",
    "title": "Shelf Announcements Feature Guide",
    "contentTypeId": "01HKW4F4H0EVQ4Q1EMW0MMANY0",
    "contentTypeLabel": "Document"
  },
  {
    "externalId": "44890dde-7447-4401-bb0a-6ceb527e750a",
    "title": "Shelf Content Intelligence Core Guide",
    "contentTypeId": "01HKW4F4H0EVQ4Q1EMW0MMANY0",
    "contentTypeLabel": "Document"
  }
]
```

</details>

where

**`externalId`**: Unique identifier for the content item (for further API queries)

`title`: Title of the document

`contentTypeId`: The content type id

`contentTypeLabel`: The human label for the content type

**Documentation**: [_Content Item List API_](https://docs.shelf.io/dev-portal/rest-api/cil-api)

### **3. Content Search Capabilities (Optional)**

**Action**: Implement search functionality to find specific content.

{% hint style="warning" %}
This optional step can be taken to later pull only those content items that are needed and relevant for your needs.
{% endhint %}

Shelf provides flexible API search and filtering for both entire documents (content items) and for finer-grained _semantic sections_ within those documents.

Full code samples, endpoint coverage, and deep filtering strategies are documented in our respective [documentation](https://docs.shelf.io/dev-portal/recipes/content-searches).

#### Document-level search

If you work with documents and want to store them in your system, use the **document-level search**.

## Search Content Items

> Returns search results over CIL content items. Applies vector search.\
> \
> Permissions: users will be able to search only across collections that they have access to.\
>

```json
{"openapi":"3.0.0","info":{"title":"Shelf Content Integration Layer API","version":"1.0.0"},"tags":[{"name":"Content Items","description":"There two endpoints: \"List Content Items\" and \"Search Content Items\". They are designed to be used in different scenarios. If you want to sync content from Shelf into your database - \"List CIL Content API\" is the right choice.\n\n**Search CIL Content API**\n- **Search event submission**: When using the Search API, a search event is submitted, which is then processed by the Content Insights pipeline. This could be useful for tracking and analyzing user search behavior over time.\n- **Appear on Search Queries reports**: The data from the search event will eventually appear on Search Queries reports, providing insights into the types of queries that users are performing.\n- **Vector search**: It applies vector search to find the most relevant content items based on the search term.\n\n**List CIL Content API**\n- **Data listing without search events**: This API lists CIL content items without creating a search event or contributing to the Content Insights pipeline.\n- **No `searchResultsUrl`**: The response from the List API does not include a `searchResultsUrl`, which would otherwise be used to link directly to the search results.\n- **Paginate over content items**: It's designed to paginate over content items rather than perform a relevance-based search.\n\nSo, to decide which API to use, consider whether you need insights and tracking of user search behavior (use the \"Search CIL Content API\") or whether you simply need to list content items without additional analytics (use the \"List CIL Content API\").\n\n**Pagination Mechanics**\n1. **Initial Request**: The initial API request includes a pagination parameter `size`, which defines the number of items to be returned in a single response. Additionally, clients can specify a `from` parameter to skip a certain number of items from the start, effectively controlling the starting point of the data retrieval.\n\n2. **Iterative Pagination**: To navigate through the content items, the API uses a `nextToken` parameter. After the initial request, the API response includes a `nextToken` value, which is a unique identifier for the next page of results.\n\n3. **Subsequent Requests**: For subsequent requests, clients include this `nextToken` in their request to retrieve the next set of items. This token ensures that each request fetches a new page of results, starting exactly where the last one ended.\n\n4. **Page Info**: Each response also contains a `pageInfo` object, providing metadata about the pagination state. This includes information like `currentPage`, `hasNextPage`, `totalPages`, and `totalResultsCount`, giving a comprehensive overview of the pagination context.\n\n5. **Efficient Data Retrieval**: This pagination system allows for efficient data retrieval, especially in scenarios where the total number of content items is large. Clients can systematically fetch all items in a controlled and manageable manner."}],"servers":[{"url":"https://api.shelf.io","description":"US region"},{"url":"https://api.shelf-eu.com","description":"EU region"},{"url":"https://api.shelf-ca.com","description":"CA region"}],"security":[{"auth_token":[]}],"components":{"securitySchemes":{"auth_token":{"description":"Shelf API Token. See [Create API Token](https://docs.shelf.io/dev-portal/authentication/create-api-token) for more details.","type":"apiKey","in":"header","name":"Authorization"}}},"paths":{"/cil-search/content":{"post":{"tags":["Content Items"],"summary":"Search Content Items","description":"Returns search results over CIL content items. Applies vector search.\n\nPermissions: users will be able to search only across collections that they have access to.\n","operationId":"searchContent","requestBody":{"content":{"application/json":{"schema":{"allOf":[{"type":"object","required":["origin"],"properties":{"origin":{"type":"string","description":"System where search request originated from."}}},{"type":"object","properties":{"purpose":{"type":"string","enum":["search-results","ask-copilot"],"default":"search-results","description":"Reason to trigger search request."}}},{"type":"object","properties":{"fieldsFilters":{"type":"object","properties":{"contentTypeId":{"type":"array","items":{"type":"string"}},"shelf_categoryId":{"type":"array","minItems":0,"maxItems":20,"items":{"type":"string"}},"shelf_categoryOp":{"type":"string","enum":["and","or"],"default":"or","description":"Specifies how to filter content by ids of categories if multiple ids provided. \"and\" - matching content should have all provided category ids. \"or\" - matching content should have any one of provided category ids."},"shelf_categoryIdToExclude":{"type":"array","description":"Filter content by ids of categories it does not belong to.","minItems":0,"maxItems":20,"items":{"type":"string"}},"shelf_reviewStatus":{"description":"Filter content by review status. Can be provided only if Content Review feature is enabled.","type":"array","minLength":0,"maxLength":3,"items":{"type":"string","enum":["up-to-date","out-of-date","none"]}},"ownerId":{"type":"array","items":{"type":"string"}},"source":{"type":"array","items":{"type":"string"}},"ratingMoreThan":{"type":"array","minItems":1,"maxItems":1,"items":{"type":"number","minimum":0,"maximum":4}},"type":{"type":"array","items":{"type":"string"}}}}}},{"type":"object","properties":{"enrichmentsFilters":{"type":"object","properties":{"ids":{"type":"array","description":"Filter content by ids of enrichments instance ids.","items":{"type":"string"}},"types":{"type":"array","description":"Filter content by types of enrichments.","items":{"type":"string","enum":["NER"]}},"values":{"type":"array","description":"Filter content by values of enrichments.","items":{"type":"object","required":["key","value"],"properties":{"key":{"type":"string","description":"Key of the enrichment"},"value":{"description":"Value of the enrichment. Can be a string, number, boolean or array of strings.","oneOf":[{"type":"array","items":{"type":"string"}},{"type":"number"},{"type":"string"},{"type":"boolean"}]}}}}}}}},{"type":"object","properties":{"query":{"type":"string","description":"Filters content by searchable text, like `title`, `description`, `text` and `tags` fields."},"queryFields":{"type":"array","description":"Filters content by specific fields by query provided. If not specified, all searchable fields are used.","minItems":1,"maxItems":5,"items":{"oneOf":[{"type":"string","enum":["title"]},{"type":"string","pattern":"^fields\\\\.shelf_customField_.+$"}]}},"searchLanguage":{"type":"string","description":"Filters content by language code. Specify `any` to search across all languages.","minLength":2,"maxLength":5},"tags":{"type":"array","description":"Filters content by tags. Uses `tagsOp` parameter to specify how to treat multiple tags.","minItems":0,"maxItems":200,"items":{"type":"string","minLength":1,"maxLength":40}},"tagsOp":{"type":"string","enum":["and","or"],"default":"or","description":"Specifies how to filter content by tags if multiple tags provided. \"and\" - matching content should have all provided tags. \"or\" - matching content should have any one of provided tags."},"tagsToExclude":{"type":"array","description":"Filter content by tags it does not belong to.","minItems":0,"maxItems":200,"items":{"type":"string","minLength":1,"maxLength":40}},"collectionIds":{"type":"array","description":"Filter content by ids of collections it belongs to.","minItems":0,"maxItems":50,"items":{"type":"string","minLength":6,"maxLength":64}},"collectionIdsToExclude":{"type":"array","description":"Filter content by ids of collections it does not belong to.","minItems":0,"maxItems":50,"items":{"type":"string","minLength":6,"maxLength":64}},"connectorIds":{"type":"array","description":"Filter content by ids of connectors it was synchronized with.","minItems":0,"maxItems":50,"items":{"type":"string","minLength":6,"maxLength":64}},"createdAfter":{"description":"Filter content that was created starting from the given date. Creation date from source system is used.","type":"string","format":"date"},"createdBefore":{"description":"Filter content that was created up to the given date. Creation date from source system is used.","type":"string","format":"date"},"updatedAfter":{"description":"Filter content that was updated after or equal the given date. Update date from source system is used.","type":"string","format":"date"},"updatedBefore":{"description":"Filter content that was updated before or equal the given date. Update date from source system is used.","type":"string","format":"date"},"updatedAfterStrict":{"description":"Filter content that was updated strictly after the given date. Update date from source system is used.","type":"string","format":"date"},"updatedBeforeStrict":{"description":"Filter content that was updated strictly before the given date. Update date from source system is used.","type":"string","format":"date"},"parentId":{"description":"Filter content that is in given location","type":"string"},"includeDeepResults":{"description":"If specified `parentId` would be also searched through grand parent locations","type":"boolean"},"includePrivateContent":{"description":"If specified private content would be included in search results","type":"boolean","default":true},"idsToExclude":{"description":"Filter out specific external ids","type":"array","minItems":0,"maxItems":1000,"items":{"type":"string","minLength":6,"maxLength":64}},"idsToInclude":{"description":"Filter specific external ids","type":"array","minItems":0,"maxItems":1000,"items":{"type":"string","minLength":6,"maxLength":64}}}},{"type":"object","properties":{"sortBy":{"type":"string","enum":["TITLE","CREATED_DATE","UPDATED_DATE","RELEVANCE","VIEWS","RATING"],"description":"Specify how (by what) to sort search results."},"sortOrder":{"type":"string","description":"Specify either ascending or descnding sort order to apply to `sortBy` parameter.","enum":["ASC","DESC"]}}},{"type":"object","properties":{"from":{"type":"integer","minimum":0,"maximum":10000,"description":"Filters out content by numbered position in the search results, for example if the value is `3`, then the first item of the search will be the 4th one, meaning that the first three were skipped."},"size":{"type":"integer","minimum":0,"maximum":1000,"description":"Using this parameter it's possible to filter out the amount of gems in response."},"nextToken":{"type":"string","description":"Next token should be used to retrieve next portion of the results (next page). Next token from the response of this API can be used correctly ONLY if filters & sorting in new request have not changed comparing to the request where next token was returned. If filters or sorting have changed, use this API without next token, and after retrieving first page with changed filters or sorting next token will be returned by this API and can be used. Use only nextToken returned by this API, do not change it."}}}]}}}},"responses":{"200":{"description":"Search results","content":{"application/json":{"schema":{"type":"object","required":["items","pageInfo","searchLanguage","searchEventId"],"properties":{"items":{"type":"array","items":{"allOf":[{"allOf":[{"type":"object","required":["fields"],"properties":{"id":{"type":"string","description":"Content item ID (internal to CIL)"},"accountId":{"description":"ID of account","type":"string","minLength":6,"maxLength":64},"internalId":{"type":"string","description":"Content item ID (internal to CIL)"},"externalId":{"type":"string","description":"External content item ID, provided by CIL Connector, corresponding to id in source system"},"title":{"type":"string","description":"Content item title"},"createdAt":{"type":"string","format":"date","description":"Date when content was first created in CIL"},"updatedAt":{"type":"string","format":"date","description":"Date when content was last updated in CIL"},"connectorId":{"type":"string","description":"Connector ID which synced this content item to CIL"},"externalURL":{"type":"string","description":"Url pointing to content in source system"},"lang":{"type":"string","description":"Language code of content item"},"lastViewedAt":{"type":"string","format":"date","description":"Date when content was last viewed at"},"originalCreatedAt":{"type":"string","format":"date","description":"Date when content was first created in source system"},"originalUpdatedAt":{"type":"string","format":"date","description":"Date when content was last updated in source system"},"tags":{"type":"array","description":"List of tags for this content item","items":{"type":"string"}},"collectionIds":{"type":"array","description":"List of collection IDs this content item belongs to in external system","items":{"type":"string","minLength":1,"maxLength":512,"description":"Collection ID from external system"}},"internalCollectionIds":{"type":"array","description":"List of internal collection IDs this content item belongs to in CIL","items":{"type":"string","minLength":26,"maxLength":26}},"locations":{"description":"List of locations of this content item belongs to an external system","type":"array","items":{"type":"object","required":["grandParentIds","collectionId"],"properties":{"grandParentIds":{"type":"array","items":{"type":"string","minLength":6,"maxLength":64}},"collectionId":{"type":"string","minLength":1,"maxLength":512,"description":"Collection ID from external system"},"parentId":{"type":"string","minLength":6,"maxLength":64}}}},"iconURL":{"type":"string","description":"Icon url for this content item. Present if this content item corresponds to Shelf KMS gem."},"fields":{"type":"array","items":{"type":"object","required":["key","value","type","label"],"properties":{"key":{"type":"string"},"value":{"type":"string"},"type":{"type":"string"},"label":{"type":"string"},"data":{"type":"object","additionalProperties":true}}}}}},{"type":"object","properties":{"iconURL":{"type":"string","description":"Icon url for this content item. Present if this content item corresponds to Shelf KMS gem."},"description":{"type":"string","description":"Content item description"},"jobId":{"type":"string","description":"Job ID that synced this content item to CIL"},"syncFlowId":{"allOf":[{"type":"string","minLength":26,"maxLength":26},{"description":"Sync Flow ID"}]}}},{"type":"object","properties":{"enrichmentsCount":{"type":"object","description":"Count of enrichments for this content item","properties":{"NER":{"type":"number","description":"Count of NER enrichments for this content item"}}},"enrichments":{"type":"array","description":"List of enrichments for this content item, list is limited to up to 5 items.","items":{"type":"object","required":["id","type","values"],"properties":{"id":{"type":"string"},"type":{"type":"string"},"values":{"type":"array","items":{"type":"object","required":["key","label","value"],"properties":{"key":{"type":"string"},"label":{"type":"string"},"value":{"oneOf":[{"type":"string"},{"type":"number"},{"type":"boolean"},{"type":"array","items":{"type":"string"}}]}}}}}}}}}]},{"type":"object","properties":{"text":{"type":"string","description":"Content item main text"},"type":{"type":"string","enum":["content-item"],"description":"Specifies that this is a content item"}}}]}},"pageInfo":{"allOf":[{"type":"object","description":"This object holds information about pagination. Use it to know how to iterate over array of results","properties":{"currentPage":{"type":"number","minimum":1,"maximum":100,"default":1},"hasNextPage":{"type":"boolean"},"hasPreviousPage":{"type":"boolean"},"pageSize":{"type":"number"},"totalPages":{"type":"number","minimum":1,"maximum":100},"totalResultsCount":{"type":"number","minimum":0,"maximum":999999}}},{"type":"object","properties":{"nextToken":{"type":"string"}}}]},"searchLanguage":{"type":"string","description":"Language code with which search request was executed"},"searchEventId":{"type":"string","description":"Search event ID associated with this search request"},"searchResultsUrl":{"type":"string","description":"Url pointing to Shelf KMS Search results with similar filters applied. Will not return exactly same results, because CIL Search covers more content than KMS Search."}}}}}},"400":{"description":"Bad Request","content":{"application/json":{"schema":{"oneOf":[{"type":"object","title":"API Error with code","required":["error"],"properties":{"error":{"description":"API Error with code","type":"object","required":["status","message","code"],"properties":{"status":{"type":"number","description":"Status code"},"message":{"type":"string","description":"Error message"},"code":{"type":"string","description":"Error code"}}}}},{"type":"object","title":"Bad request payload (AJV)","required":["error"],"properties":{"error":{"description":"API Error With detail","type":"object","required":["status","message","detail"],"properties":{"status":{"type":"number","description":"Status code"},"message":{"type":"string","description":"Error message"},"detail":{"description":"Detail info","type":"array","items":{"type":"object","properties":{"keyword":{"type":"string"},"instancePath":{"type":"string"},"schemaPath":{"type":"string"},"params":{"type":"object"},"message":{"type":"string"}}}}}}}}]}}}},"403":{"description":"Forbidden","content":{"application/json":{"schema":{"type":"object","title":"API Error with code","required":["error"],"properties":{"error":{"description":"API Error with code","type":"object","required":["status","message","code"],"properties":{"status":{"type":"number","description":"Status code"},"message":{"type":"string","description":"Error message"},"code":{"type":"string","description":"Error code"}}}}}}}}}}}}}
```

You may also filter the search by various parameters and criteria.  \
For example, you want to search for documents with the category `"Cars"` in Shelf.&#x20;

In this case, you need to use the `https://api.shelf.io/cil-search/content` endpoint with the dedicated payload.

<details>

<summary>Example of Content Category Filtering JSON Payload</summary>

```json
{
  "origin": "my-app",
  "query": "*",
  "fieldsFilters": {
    "shelf_categoryId": [
      "setId#default#categoryId#01HKW83S7FRMJZ4EB4VATT3SV1"
    ]
  }
}
```

</details>

* Replace `"my-app"` with your app name.
* Update `shelf_categoryId` with your actual Shelf category ID. For retrieving `shelf_categoryId` , please refer to the [Shelf Categories API Documentation](https://docs.shelf.io/dev-portal/rest-api/categories-api) and see the two figures below. Make sure to use the following formatting for category id: `setId#default#categoryId#somecategoryid`.

<figure><img src="https://3746171570-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2F7yKmgyutbCyeUlzPhhYi%2Fuploads%2FqGeLN4wTUWFq739d2WVJ%2FUseCase-Postman-CatSetsAPI.png?alt=media&#x26;token=c8523030-aaec-4c40-bc05-506e8f17ad72" alt=""><figcaption></figcaption></figure>

<figure><img src="https://3746171570-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2F7yKmgyutbCyeUlzPhhYi%2Fuploads%2FxsT4LKYe9vGHS33eqMmx%2FUseCase-Postman-CatIdAPI.png?alt=media&#x26;token=2c165928-32de-4967-9cf3-5c603768aed4" alt=""><figcaption></figcaption></figure>

#### Semantic section search

If your documents are big and you want smaller sections for more refined use in your RAG applications, the search by semantic sections should be run.

## Search CIL Semantic Sections

> Returns search results over CIL content as semantic sections to which items were split.\
> \
> Permissions: users will be able to search only across collections they have access to.\
>

```json
{"openapi":"3.0.0","info":{"title":"Shelf Content Integration Layer API","version":"1.0.0"},"tags":[],"servers":[{"url":"https://api.shelf.io","description":"US region"},{"url":"https://api.shelf-eu.com","description":"EU region"},{"url":"https://api.shelf-ca.com","description":"CA region"}],"security":[{"auth_token":[]}],"components":{"securitySchemes":{"auth_token":{"description":"Shelf API Token. See [Create API Token](https://docs.shelf.io/dev-portal/authentication/create-api-token) for more details.","type":"apiKey","in":"header","name":"Authorization"}}},"paths":{"/cil-search/content/semantic-sections":{"post":{"tags":["Search"],"summary":"Search CIL Semantic Sections","description":"Returns search results over CIL content as semantic sections to which items were split.\n\nPermissions: users will be able to search only across collections they have access to.\n","operationId":"searchSemanticSections","requestBody":{"content":{"application/json":{"schema":{"allOf":[{"allOf":[{"type":"object","required":["origin"],"properties":{"origin":{"type":"string","description":"System where search request originated from."}}},{"type":"object","properties":{"format":{"enum":["html","markdown","text"],"default":"html","description":"Specifies format of `content` field in response. `html` - text with html tags `markdown` - text with markdown tags `text` - plain text"}}},{"type":"object","properties":{"fieldsFilters":{"type":"object","properties":{"contentTypeId":{"type":"array","items":{"type":"string"}},"shelf_categoryId":{"type":"array","minItems":0,"maxItems":20,"items":{"type":"string"}},"shelf_categoryOp":{"type":"string","enum":["and","or"],"default":"or","description":"Specifies how to filter content by ids of categories if multiple ids provided. \"and\" - matching content should have all provided category ids. \"or\" - matching content should have any one of provided category ids."},"shelf_categoryIdToExclude":{"type":"array","description":"Filter content by ids of categories it does not belong to.","minItems":0,"maxItems":20,"items":{"type":"string"}},"shelf_reviewStatus":{"description":"Filter content by review status. Can be provided only if Content Review feature is enabled.","type":"array","minLength":0,"maxLength":3,"items":{"type":"string","enum":["up-to-date","out-of-date","none"]}},"ownerId":{"type":"array","items":{"type":"string"}},"source":{"type":"array","items":{"type":"string"}},"ratingMoreThan":{"type":"array","minItems":1,"maxItems":1,"items":{"type":"number","minimum":0,"maximum":4}},"type":{"type":"array","items":{"type":"string"}}}}}},{"type":"object","properties":{"enrichmentsFilters":{"type":"object","properties":{"ids":{"type":"array","description":"Filter content by ids of enrichments instance ids.","items":{"type":"string"}},"types":{"type":"array","description":"Filter content by types of enrichments.","items":{"type":"string","enum":["NER"]}},"values":{"type":"array","description":"Filter content by values of enrichments.","items":{"type":"object","required":["key","value"],"properties":{"key":{"type":"string","description":"Key of the enrichment"},"value":{"description":"Value of the enrichment. Can be a string, number, boolean or array of strings.","oneOf":[{"type":"array","items":{"type":"string"}},{"type":"number"},{"type":"string"},{"type":"boolean"}]}}}}}}}},{"type":"object","properties":{"enrichmentsFilters":{"type":"object","properties":{"ids":{"type":"array","description":"Filter content by ids of enrichments instance ids.","items":{"type":"string"}},"types":{"type":"array","description":"Filter content by types of enrichments.","items":{"type":"string","enum":["NER"]}},"values":{"type":"array","description":"Filter content by values of enrichments.","items":{"type":"object","required":["key","value"],"properties":{"key":{"type":"string","description":"Key of the enrichment"},"value":{"description":"Value of the enrichment. Can be a string, number, boolean or array of strings.","oneOf":[{"type":"array","items":{"type":"string"}},{"type":"number"},{"type":"string"},{"type":"boolean"}]}}}}}}}},{"type":"object","properties":{"query":{"type":"string","description":"Filters content by searchable text, like `title`, `description`, `text` and `tags` fields."},"queryFields":{"type":"array","description":"Filters content by specific fields by query provided. If not specified, all searchable fields are used.","minItems":1,"maxItems":5,"items":{"oneOf":[{"type":"string","enum":["title"]},{"type":"string","pattern":"^fields\\\\.shelf_customField_.+$"}]}},"searchLanguage":{"type":"string","description":"Filters content by language code. Specify `any` to search across all languages.","minLength":2,"maxLength":5},"tags":{"type":"array","description":"Filters content by tags. Uses `tagsOp` parameter to specify how to treat multiple tags.","minItems":0,"maxItems":200,"items":{"type":"string","minLength":1,"maxLength":40}},"tagsOp":{"type":"string","enum":["and","or"],"default":"or","description":"Specifies how to filter content by tags if multiple tags provided. \"and\" - matching content should have all provided tags. \"or\" - matching content should have any one of provided tags."},"tagsToExclude":{"type":"array","description":"Filter content by tags it does not belong to.","minItems":0,"maxItems":200,"items":{"type":"string","minLength":1,"maxLength":40}},"collectionIds":{"type":"array","description":"Filter content by ids of collections it belongs to.","minItems":0,"maxItems":50,"items":{"type":"string","minLength":6,"maxLength":64}},"collectionIdsToExclude":{"type":"array","description":"Filter content by ids of collections it does not belong to.","minItems":0,"maxItems":50,"items":{"type":"string","minLength":6,"maxLength":64}},"connectorIds":{"type":"array","description":"Filter content by ids of connectors it was synchronized with.","minItems":0,"maxItems":50,"items":{"type":"string","minLength":6,"maxLength":64}},"createdAfter":{"description":"Filter content that was created starting from the given date. Creation date from source system is used.","type":"string","format":"date"},"createdBefore":{"description":"Filter content that was created up to the given date. Creation date from source system is used.","type":"string","format":"date"},"updatedAfter":{"description":"Filter content that was updated after or equal the given date. Update date from source system is used.","type":"string","format":"date"},"updatedBefore":{"description":"Filter content that was updated before or equal the given date. Update date from source system is used.","type":"string","format":"date"},"updatedAfterStrict":{"description":"Filter content that was updated strictly after the given date. Update date from source system is used.","type":"string","format":"date"},"updatedBeforeStrict":{"description":"Filter content that was updated strictly before the given date. Update date from source system is used.","type":"string","format":"date"},"parentId":{"description":"Filter content that is in given location","type":"string"},"includeDeepResults":{"description":"If specified `parentId` would be also searched through grand parent locations","type":"boolean"},"includePrivateContent":{"description":"If specified private content would be included in search results","type":"boolean","default":true},"idsToExclude":{"description":"Filter out specific external ids","type":"array","minItems":0,"maxItems":1000,"items":{"type":"string","minLength":6,"maxLength":64}},"idsToInclude":{"description":"Filter specific external ids","type":"array","minItems":0,"maxItems":1000,"items":{"type":"string","minLength":6,"maxLength":64}}}},{"type":"object","properties":{"sortBy":{"type":"string","enum":["TITLE","CREATED_DATE","UPDATED_DATE","RELEVANCE","VIEWS","RATING"],"description":"Specify how (by what) to sort search results."},"sortOrder":{"type":"string","description":"Specify either ascending or descnding sort order to apply to `sortBy` parameter.","enum":["ASC","DESC"]}}},{"type":"object","properties":{"from":{"type":"integer","minimum":0,"maximum":10000,"description":"Filters out content by numbered position in the search results, for example if the value is `3`, then the first item of the search will be the 4th one, meaning that the first three were skipped."},"size":{"type":"integer","minimum":0,"maximum":1000,"description":"Using this parameter it's possible to filter out the amount of gems in response."},"nextToken":{"type":"string","description":"Next token should be used to retrieve next portion of the results (next page). Next token from the response of this API can be used correctly ONLY if filters & sorting in new request have not changed comparing to the request where next token was returned. If filters or sorting have changed, use this API without next token, and after retrieving first page with changed filters or sorting next token will be returned by this API and can be used. Use only nextToken returned by this API, do not change it."}}}]},{"type":"object","properties":{"ranking":{"type":"object","description":"Optional Re-ranking step configuration to be additionally performed after getting page of search results. Requires OpenAI feature to be enabled.","required":["scoring"],"properties":{"scoring":{"type":"string","enum":["similarity"]},"reorder":{"type":"boolean","default":false,"description":"If true will reorder page of search results based on ranking similarity"}}}}}]}}}},"responses":{"200":{"description":"Search results","content":{"application/json":{"schema":{"type":"object","required":["sections","pageInfo"],"properties":{"sections":{"type":"array","items":{"allOf":[{"type":"object","properties":{"id":{"type":"string","description":"Semantic section ID"}}},{"allOf":[{"type":"object","required":["fields"],"properties":{"id":{"type":"string","description":"Content item ID (internal to CIL)"},"accountId":{"description":"ID of account","type":"string","minLength":6,"maxLength":64},"internalId":{"type":"string","description":"Content item ID (internal to CIL)"},"externalId":{"type":"string","description":"External content item ID, provided by CIL Connector, corresponding to id in source system"},"title":{"type":"string","description":"Content item title"},"createdAt":{"type":"string","format":"date","description":"Date when content was first created in CIL"},"updatedAt":{"type":"string","format":"date","description":"Date when content was last updated in CIL"},"connectorId":{"type":"string","description":"Connector ID which synced this content item to CIL"},"externalURL":{"type":"string","description":"Url pointing to content in source system"},"lang":{"type":"string","description":"Language code of content item"},"lastViewedAt":{"type":"string","format":"date","description":"Date when content was last viewed at"},"originalCreatedAt":{"type":"string","format":"date","description":"Date when content was first created in source system"},"originalUpdatedAt":{"type":"string","format":"date","description":"Date when content was last updated in source system"},"tags":{"type":"array","description":"List of tags for this content item","items":{"type":"string"}},"collectionIds":{"type":"array","description":"List of collection IDs this content item belongs to in external system","items":{"type":"string","minLength":1,"maxLength":512,"description":"Collection ID from external system"}},"internalCollectionIds":{"type":"array","description":"List of internal collection IDs this content item belongs to in CIL","items":{"type":"string","minLength":26,"maxLength":26}},"locations":{"description":"List of locations of this content item belongs to an external system","type":"array","items":{"type":"object","required":["grandParentIds","collectionId"],"properties":{"grandParentIds":{"type":"array","items":{"type":"string","minLength":6,"maxLength":64}},"collectionId":{"type":"string","minLength":1,"maxLength":512,"description":"Collection ID from external system"},"parentId":{"type":"string","minLength":6,"maxLength":64}}}},"iconURL":{"type":"string","description":"Icon url for this content item. Present if this content item corresponds to Shelf KMS gem."},"fields":{"type":"array","items":{"type":"object","required":["key","value","type","label"],"properties":{"key":{"type":"string"},"value":{"type":"string"},"type":{"type":"string"},"label":{"type":"string"},"data":{"type":"object","additionalProperties":true}}}}}},{"type":"object","properties":{"iconURL":{"type":"string","description":"Icon url for this content item. Present if this content item corresponds to Shelf KMS gem."},"description":{"type":"string","description":"Content item description"},"jobId":{"type":"string","description":"Job ID that synced this content item to CIL"},"syncFlowId":{"allOf":[{"type":"string","minLength":26,"maxLength":26},{"description":"Sync Flow ID"}]}}},{"type":"object","properties":{"enrichmentsCount":{"type":"object","description":"Count of enrichments for this content item","properties":{"NER":{"type":"number","description":"Count of NER enrichments for this content item"}}},"enrichments":{"type":"array","description":"List of enrichments for this content item, list is limited to up to 5 items.","items":{"type":"object","required":["id","type","values"],"properties":{"id":{"type":"string"},"type":{"type":"string"},"values":{"type":"array","items":{"type":"object","required":["key","label","value"],"properties":{"key":{"type":"string"},"label":{"type":"string"},"value":{"oneOf":[{"type":"string"},{"type":"number"},{"type":"boolean"},{"type":"array","items":{"type":"string"}}]}}}}}}}}}]},{"type":"object","required":["sectionId"],"properties":{"type":{"type":"string","enum":["content-section"],"description":"Specifies that this is a content section"},"sectionId":{"type":"string","description":"Semantic section ID"},"sectionHash":{"type":"string","description":"Fingerprint for the content"},"sectionType":{"type":"string","enum":["table","text","image"],"description":"Type of the section"},"sectionTitle":{"type":"string","description":"The title of the section"},"sectionTitleHTML":{"type":"string","description":"The HTML version of the section title"},"sectionTitleBreadcrumbs":{"type":"array","items":{"type":"string"},"description":"List of section titles from parent section titles to title of this section"},"sectionLevel":{"type":"integer","description":"Hierarchical level of the section"},"sectionOrder":{"type":"integer","description":"The order of the section in the document"},"sectionParentId":{"type":"string","description":"ID of the parent section"},"sectionPreviousId":{"type":"string","description":"ID of the previous section"},"sectionNextId":{"type":"string","description":"ID of the next section"},"content":{"type":"string","description":"Content of this content section in requested format"},"ranking":{"type":"object","description":"Ranking information for this section, present if ranking was requested","required":["similarity"],"properties":{"similarity":{"type":"number","description":"Similarity score showing how relevant this section is to the search query","minimum":0,"maximum":1},"isAvailable":{"type":"boolean","description":"Equals to false if ranking information is not available for this section"}}}}}]}},"pageInfo":{"allOf":[{"type":"object","description":"This object holds information about pagination. Use it to know how to iterate over array of results","properties":{"currentPage":{"type":"number","minimum":1,"maximum":100,"default":1},"hasNextPage":{"type":"boolean"},"hasPreviousPage":{"type":"boolean"},"pageSize":{"type":"number"},"totalPages":{"type":"number","minimum":1,"maximum":100},"totalResultsCount":{"type":"number","minimum":0,"maximum":999999}}},{"type":"object","properties":{"nextToken":{"type":"string"}}}]}}}}}},"400":{"description":"Bad Request","content":{"application/json":{"schema":{"oneOf":[{"type":"object","title":"API Error with code","required":["error"],"properties":{"error":{"description":"API Error with code","type":"object","required":["status","message","code"],"properties":{"status":{"type":"number","description":"Status code"},"message":{"type":"string","description":"Error message"},"code":{"type":"string","description":"Error code"}}}}},{"type":"object","title":"Bad request payload (AJV)","required":["error"],"properties":{"error":{"description":"API Error With detail","type":"object","required":["status","message","detail"],"properties":{"status":{"type":"number","description":"Status code"},"message":{"type":"string","description":"Error message"},"detail":{"description":"Detail info","type":"array","items":{"type":"object","properties":{"keyword":{"type":"string"},"instancePath":{"type":"string"},"schemaPath":{"type":"string"},"params":{"type":"object"},"message":{"type":"string"}}}}}}}}]}}}},"403":{"description":"Forbidden","content":{"application/json":{"schema":{"type":"object","title":"API Error with code","required":["error"],"properties":{"error":{"description":"API Error with code","type":"object","required":["status","message","code"],"properties":{"status":{"type":"number","description":"Status code"},"message":{"type":"string","description":"Error message"},"code":{"type":"string","description":"Error code"}}}}}}}}}}}}}
```

If you want to look for the semantic section across the content items belonging to some specific category, the request body payload is similar to the one you use for retrieving whole documents of a specific category.

<details>

<summary>Example of JSON Payload</summary>

```json
{
  "origin": "my-app",
  "query": "Alfa Romeo",
  "fieldsFilters": {
    "shelf_categoryId": [
      "setId#default#categoryId#01HKW83S7FRMJZ4EB4VATT3SV1"
    ]
  }
}
```

</details>

{% hint style="warning" %}
The `origin` parameter is **required** in both document-level (`/cil-search/content`) and semantic section (`/cil-search/content/semantic-sections`) Shelf API search requests because **it identifies the source of the search request** (e.g., `"web"`, `"api"`, `"bot"`, or your app name).&#x20;

If you omit the `origin` key, your request may be rejected with a <mark style="color:red;">400</mark> or <mark style="color:red;">422</mark> error ("_missing required property '`origin`_'") as Shelf API cannot identify who or what is making the query, so it cannot enforce appropriate security and behavior.
{% endhint %}

### **4. Content Extraction**

**Action**: _Extract different types of content from Shelf—including wiki pages, attachments, and advanced structures like Decision Trees—as actionable payloads for AI, LLM, and analytics scenarios_

{% hint style="warning" %}
Shelf supports and stores multiple types of content. Before you proceed to extracting the needed content, you first would like to check to which specific type your content belongs. See the [Content Discovery section](#get-content-types-types) for more details on identifying content types.
{% endhint %}

#### **Wiki Content**

Shelf stores Wiki Gems, which often contain rich text. When extracting this content via API, you might encounter different formats depending on the endpoint used. For AI applications, _**Markdown**_ is typically the preferred format due to its cleaner structure.

**Action**: _Extract Wiki Gem content and ensure it is available in **Markdown** format_

**(1)** Identify Wiki-type Gems from the content list: [Filter](#get-content-types-types) your content list to isolate Gems categorized as Wiki content.&#x20;

<details>

<summary>Example of Response with Wiki Content Type Id</summary>

```json
{
    "accountId": "{{accountId}}",
    "createdAt": "2024-01-11T11:28:09.367Z",
    "updatedBy": "root",
    "createdBy": "root",
    "name": "Wiki Page",
    "isLocked": true,
    "updatedAt": "2024-01-11T11:28:09.367Z",
    "isPublished": true,
    "isDefault": true,
    "description": "Create different media-rich (images, videos, tables, etc.) articles, text documents, and step-by-step processes using a feature-rich text editor",
    "id": "01HKW4F4MQHT7C1JZJP9SGMAFS",
    "key": "Note",
    "iconURL": "https://thumbnails.shelf.io/{{accountId}}/content-types/01HKW4F4MQHT7C1JZJP9SGMAFS/icons/type_icon",
    "iconS3URL": "https://shelf.io/{{accountId}}/content-types/01HKW4F4MQHT7C1JZJP9SGMAFS/icons/type_icon",
    "createdByUsername": "Shelf",
    "updatedByUsername": "Shelf"
}
```

</details>

Note the `"id": "01HKW4F4MQHT7C1JZJP9SGMAFS"` in the response - it is the Wiki content type id.&#x20;

**(2)** Get the needed Gem's `externalId` (or `itemId`) from the response of the [content item list](#post-cil-content-items-list) call.

<details>

<summary>Example of Response with Wiki Content Item Id</summary>

```json
{
  "items": [
    {
      "id": "01JNKGXRA03Y3X0SNDZFCVAHHC",
      "accountId": "{{accountId}}",
      "internalId": "01JNKGXRA03Y3X0SNDZFCVAHHC",
      "externalId": "56b32d3d-4338-4ad6-aeb3-53220f94e334",
      "title": "Insurance policy ACME IP-002",
      "createdAt": "2025-03-05T16:06:23.297Z",
      "updatedAt": "2025-04-28T15:19:52.568Z",
      "connectorId": "default-shelf-{{accountId}}",
      "externalURL": "https://{{userSubdomain}}.shelf.io/read/56b32d3d-4338-4ad6-aeb3-53220f94e334",
      "lang": "en",
      "lastViewedAt": "2025-04-28T15:19:50.505Z",
      "originalCreatedAt": "2025-03-05T16:06:22.421Z",
      "originalUpdatedAt": "2025-03-05T16:18:22.921Z",
      "tags": [
        "finance",
        "insurance",
        "policy"
      ]
      // ... rest of the item fields
    }
    // ... more items
  ]
}
```

</details>

**(3)** Retrieve the complete Wiki content in its **default** format, `HTML`: Shelf provides a direct endpoint to access the rendered content of a Wiki Gem.&#x20;

## Download gem's Wiki Content

> User must have access to view the gem in order to download wiki content.\
> Gem types with wiki content include: Article, Post, Wiki Page, Person, Organization, Project.\
>

```json
{"openapi":"3.1.0","info":{"title":"Shelf KMS Gems API","version":"1.0.0"},"servers":[{"description":"US region","url":"https://api.shelf.io"},{"description":"EU region","url":"https://api.shelf-eu.com"},{"description":"CA region","url":"https://api.shelf-ca.com"}],"security":[{"auth_token":[]}],"components":{"securitySchemes":{"auth_token":{"type":"apiKey","description":"Shelf API Token. See [Create API Token](https://docs.shelf.io/dev-portal/authentication/create-api-token) for more details.","in":"header","name":"Authorization"}}},"paths":{"/content/v1/gems/{id}/wiki-content":{"get":{"operationId":"downloadGemWikiContent","summary":"Download gem's Wiki Content","description":"User must have access to view the gem in order to download wiki content.\nGem types with wiki content include: Article, Post, Wiki Page, Person, Organization, Project.\n","parameters":[{"description":"ID of the gem with wiki content","schema":{"type":"string","maximum":64,"minimum":6},"in":"path","name":"id","required":true},{"description":"Version of wiki content to download. Defaults to the latest version.\nRequires `variant` query param set if used\n","schema":{"type":"string","maximum":64,"minimum":10},"in":"query","name":"version","required":false},{"description":"Indicates which wiki content to download from a specific version.\nCan be used only if `version` query param is set\n","schema":{"type":"string","enum":["old","new"]},"in":"query","name":"variant","required":false}],"responses":{"200":{"description":"Success - wiki content HTML text","content":{"text/html":{}}},"403":{"description":"You don't have permission to download this gem","content":{"application/json":{"schema":{"type":"object"}}}},"404":{"description":"No such gem","content":{"application/json":{"schema":{"type":"object"}}}},"422":{"description":"Gem has no wiki content to download","content":{"application/json":{"schema":{"type":"object"}}}}},"tags":["Download"]}}}}
```

**Path Parameter**: Replace `{id}` with the `externalId` of the Wiki Gem you have received in Step **(2)** above.&#x20;

As a result of running this request, you get Wiki content rendered as **HTML**. While this represents the visual look within Shelf, HTML often contains styling and structural elements less suitable for direct AI ingestion compared to **Markdown**. &#x20;

**(4)** Retrieve Wiki content as **Markdown** via the conversion: To obtain the content in the AI-preferred **Markdown** format, Shelf provides a specific mechanism using its attachment conversion capabilities. Even if the underlying storage or default representation involves **HTML**, this process allows you to request the **Markdown** version.

**(4a) Find the Primary Content Attachment ID:** First, you need the specific `attachmentId` that represents the Wiki's main content. Make a request to get the Gem's details:

## Get single CIL Content Item

> Used to fetch basic metadata about a single CIL content item.\
>

```json
{"openapi":"3.1.0","info":{"title":"Shelf Content Integration Layer API","version":"1.0.0"},"tags":[{"name":"Retrieve Content"}],"servers":[{"description":"US region","url":"https://api.shelf.io"},{"description":"EU region","url":"https://api.shelf-eu.com"},{"description":"CA region","url":"https://api.shelf-ca.com"}],"security":[{"auth_token":[]}],"components":{"securitySchemes":{"auth_token":{"type":"apiKey","description":"Shelf API Token. See [Create API Token](https://docs.shelf.io/dev-portal/authentication/create-api-token) for more details.","in":"header","name":"Authorization"}}},"paths":{"/cil-content/items/{itemId}":{"get":{"operationId":"getCILItem","summary":"Get single CIL Content Item","description":"Used to fetch basic metadata about a single CIL content item.\n","parameters":[{"description":"External Content ID","schema":{"type":"string","maxLength":26,"minLength":26},"in":"path","name":"itemId","required":true},{"description":"Content type of the request payload.","schema":{"type":"string","default":"application/json"},"in":"header","name":"Content-Type","required":true}],"responses":{"200":{"description":"CIL Item Metadata","content":{"application/json":{"schema":{"allOf":[{"type":"object","required":["fields"],"properties":{"accountId":{"type":"string","description":"ID of account","maxLength":64,"minLength":6},"collectionIds":{"type":"array","description":"List of collection IDs this content item belongs to in external system","items":{"type":"string","description":"Collection ID from external system","maxLength":512,"minLength":1}},"connectorId":{"type":"string","description":"Connector ID which synced this content item to CIL"},"createdAt":{"type":"string","description":"Date when content was first created in CIL","format":"date"},"externalId":{"type":"string","description":"External content item ID, provided by CIL Connector, corresponding to id in source system"},"externalURL":{"type":"string","description":"Url pointing to content in source system"},"fields":{"type":"array","items":{"type":"object","required":["key","value","type","label"],"properties":{"type":{"type":"string"},"data":{"type":"object","additionalProperties":true},"key":{"type":"string"},"label":{"type":"string"},"value":{"type":"string"}}}},"iconURL":{"type":"string","description":"Icon url for this content item. Present if this content item corresponds to Shelf KMS gem."},"id":{"type":"string","description":"Content item ID (internal to CIL)"},"internalCollectionIds":{"type":"array","description":"List of internal collection IDs this content item belongs to in CIL","items":{"type":"string","maxLength":26,"minLength":26}},"internalId":{"type":"string","description":"Content item ID (internal to CIL)"},"lang":{"type":"string","description":"Language code of content item"},"lastViewedAt":{"type":"string","description":"Date when content was last viewed at","format":"date"},"locations":{"type":"array","description":"List of locations of this content item belongs to an external system","items":{"type":"object","required":["grandParentIds","collectionId"],"properties":{"collectionId":{"type":"string","description":"Collection ID from external system","maxLength":512,"minLength":1},"grandParentIds":{"type":"array","items":{"type":"string","maxLength":64,"minLength":6}},"parentId":{"type":"string","maxLength":64,"minLength":6}}}},"originalCreatedAt":{"type":"string","description":"Date when content was first created in source system","format":"date"},"originalUpdatedAt":{"type":"string","description":"Date when content was last updated in source system","format":"date"},"tags":{"type":"array","description":"List of tags for this content item","items":{"type":"string"}},"title":{"type":"string","description":"Content item title"},"updatedAt":{"type":"string","description":"Date when content was last updated in CIL","format":"date"}}},{"type":"object","properties":{"description":{"type":"string","description":"CIL item description"},"attachments":{"type":"array","description":"List of attachments for this content item","items":{"type":"object","description":"Attachments","properties":{"attachmentId":{"type":"string","description":"Attachment ID"},"converted":{"type":"array","description":"An array of objects, each representing a converted file format available for the attachment","items":{"type":"object","properties":{"attachmentId":{"type":"string","description":"Attachment ID"},"extension":{"type":"string","description":"The file extension of the attachment","enum":["md","txt","html","json"]}}}},"extension":{"type":"string","description":"The file extension of the attachment","enum":["pdf","docx","txt","md","html","pptx"]}}}}}}]}}}},"400":{"description":"Bad Request","content":{"application/json":{"schema":{"oneOf":[{"type":"object","title":"API Error with code","required":["error"],"properties":{"error":{"type":"object","description":"API Error with code","required":["status","message","code"],"properties":{"code":{"type":"string","description":"Error code"},"message":{"type":"string","description":"Error message"},"status":{"type":"number","description":"Status code"}}}}},{"type":"object","title":"Bad request payload (AJV)","required":["error"],"properties":{"error":{"type":"object","description":"API Error With detail","required":["status","message","detail"],"properties":{"detail":{"type":"array","description":"Detail info","items":{"type":"object","properties":{"instancePath":{"type":"string"},"keyword":{"type":"string"},"message":{"type":"string"},"params":{"type":"object"},"schemaPath":{"type":"string"}}}},"message":{"type":"string","description":"Error message"},"status":{"type":"number","description":"Status code"}}}}}]}}}},"403":{"description":"Forbidden","content":{"application/json":{"schema":{"type":"object","title":"API Error with code","required":["error"],"properties":{"error":{"type":"object","description":"API Error with code","required":["status","message","code"],"properties":{"code":{"type":"string","description":"Error code"},"message":{"type":"string","description":"Error message"},"status":{"type":"number","description":"Status code"}}}}}}}}},"tags":["Retrieve Content"]}}}}
```

**Path Parameter**: Replace `{id}` with the `externalId` of the Wiki Gem you have received in Step **(2)** above.&#x20;

<details>

<summary>Example of Response with Content Item's Attachments</summary>

```json
"description": "Comprehensive Overview of Policy Form ACME IP-002",
"iconURL": "https://thumbnails.shelf.io/{{accountId}}/content-types/01HKW4F4MQHT7C1JZJP9SGMAFS/icons/type_icon",
"attachments": [
    {
        "attachmentId": "56427db5bba2b4ad4a7e246e738323b5",
        "extension": "html",
        "converted": [
            {
                "attachmentId": "8801b3b6440b7dd9f839b84284c6bbdf",
                "extension": "txt"
            },
            {
                "attachmentId": "8b294d4487fb43cdd8d1de57ab4e07da",
                "extension": "md"
            }
        ]
    }
]
```

</details>

In the response you get, find and copy the main attachment's id - the first `attachmentId` key in the `attachments` array.&#x20;

**(4b) Request Markdown Conversion of the Attachment:** Use the attachment conversion endpoint, explicitly requesting the `md` extension. This tells Shelf to provide the content _as_ Markdown, performing a conversion if necessary.

## Download Attachment By Extension Format

> Download Content Item Attachment By Extension Format\
>

```json
{"openapi":"3.1.0","info":{"title":"Shelf Content Integration Layer API","version":"1.0.0"},"tags":[{"name":"Retrieve Content"}],"servers":[{"description":"US region","url":"https://api.shelf.io"},{"description":"EU region","url":"https://api.shelf-eu.com"},{"description":"CA region","url":"https://api.shelf-ca.com"}],"security":[{"auth_token":[]}],"components":{"securitySchemes":{"auth_token":{"type":"apiKey","description":"Shelf API Token. See [Create API Token](https://docs.shelf.io/dev-portal/authentication/create-api-token) for more details.","in":"header","name":"Authorization"}}},"paths":{"/cil-content/items/{itemId}/attachments/{attachmentId}/content/converted/{extension}":{"get":{"operationId":"getItemAttachmentContentByExtensions","summary":"Download Attachment By Extension Format","description":"Download Content Item Attachment By Extension Format\n","parameters":[{"description":"Content Item ID","schema":{"type":"string","maxLength":64,"minLength":6},"in":"path","name":"itemId","required":true},{"description":"Attachment ID. Could be obtained from API to get 1 item.","schema":{"type":"string","maxLength":26,"minLength":26},"in":"path","name":"attachmentId","required":true},{"description":"The file extension of the converted file","schema":{"enum":["txt","html","md"]},"in":"path","name":"extension","required":true}],"responses":{"200":{"description":"Success - download url","content":{"application/json":{"schema":{"type":"object","properties":{"url":{"type":"string","description":"URL to download the file. It is temporary and will expire after some time.","format":"uri"}}}}}},"404":{"description":"Not Found","content":{"application/json":{"schema":{"type":"object","required":["error"],"properties":{"error":{"type":"object","description":"API Error","required":["status","message","code"],"properties":{"code":{"type":"string","description":"ERROR_CODE"},"detail":{"description":"Detail info","oneOf":[{"type":"string"},{"type":"array","items":{"type":"object","properties":{"dataPath":{"type":"string"},"keyword":{"type":"string"},"message":{"type":"string"},"params":{"type":"object"},"schemaPath":{"type":"string"}}}}]},"message":{"type":"string","description":"Error message"},"status":{"type":"number","description":"Status code"}}}}}}}}},"tags":["Retrieve Content"]}}}}
```

**Path Parameters:**
\
\- Replace `{itemId}` with the Gem's `externalId`.
\
\- Replace `{attachmentId}` with the ID of the main content attachment found in Step **(4a)** above.
\
\- Replace `{extension}` with `md`.

**Query Parameter:** Add `origin=YOUR_APP_NAME` (e.g., `origin=web`).

The response body of this targeted request will contain the download link to content formatted as Markdown, suitable for your AI pipeline.

<details>

<summary>Markdown Download Link</summary>

```json
{
    "url": "https://s3.shelf.amazon-aws.com/bucket-name/60b11695-41a0-4d65-9b44-1638f5bfa206/sync-flows/01HKW4FBRN8WC848NHDFF0F44X/content-items/56b32d3d-4338-4ad6-aeb3-53220f94e334/attachments/01JNKGXZCA6SQ5QT4JP20FFHEE/2025-03-05T16%3A06%3A22.527Z.html.01JNKGXZC9N0EWM7870T1WA0M9.md?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=YOUR_CREDENTIAL_PLACEHOLDER%2FYYYYMMDD%2Feu-central-1%2Fs3%2Faws4_request&X-Amz-Date=YYYYMMDDTHHMMSSZ&X-Amz-Expires=300&X-Amz-Security-Token=YOUR_SECURITY_TOKEN_PLACEHOLDER&X-Amz-Signature=YOUR_SIGNATURE_PLACEHOLDER&X-Amz-SignedHeaders=host&response-content-disposition=attachment%3B%20filename%3D%222025-03-05T16%3A06%3A22.527Z.html.01JNKGXZC9N0EWM7870T1WA0M9.md%22&x-id=GetObject"
}
```

</details>

Now, by following this link, you can download your Wiki content item as a Markdown file.&#x20;

<details>

<summary>Wiki Gem's Markdown Version</summary>

{% code overflow="wrap" %}
```markdown
### 1\. Introduction

ACME Financial Company is pleased to introduce **Policy Form ACME IP-002**, our enhanced auto insurance product tailored for both personal and commercial-use vehicles in diverse environments. This policy offers comprehensive coverage options, including liability, collision, comprehensive events, and additional protections, ensuring policyholders receive the optimal protection needed without compromising their financial stability. Leveraging decades of underwriting expertise, ACME Financial adheres to **GAAP (Generally Accepted Accounting Principles)** and maintains our **Loss Ratio** well below industry averages. 

---

### 2\. Coverage Features of ACME IP-002

Under **ACME IP-002**, policyholders benefit from an extensive range of auto insurance protections designed to meet both personal and commercial needs. Core features include:

...
```
{% endcode %}

</details>

Make sure to do it immediately as you get the URL: it has the limited expiration period (<mark style="color:red;">**5 minutes**</mark>), and once it is expired, you will get the <mark style="color:red;">`403 Forbidden`</mark> error. In this case, you need to rerun the conversion API call.

<details>

<summary>Example of Expired URL Error</summary>

```xml
<?xml version="1.0" encoding="UTF-8"?>
<Error>
    <Code>AccessDenied</Code>
    <Message>Request has expired</Message>
    <X-Amz-Expires>300</X-Amz-Expires>
    <Expires>2025-04-30T13:29:33Z</Expires>
    <ServerTime>2025-04-30T15:24:22Z</ServerTime>
    <RequestId>B78G7V0W6HR308QE</RequestId>
    <HostId>M6yctucCFuY/ODzZKuDSSeopYSC+BbZiTErrq1WAorlQKqSc5rIOcthJOn+MwlAv08auyKn+1QU=</HostId>
</Error>
```

</details>

By using the `/content/converted/md` endpoint, you ensure you receive Markdown, regardless of whether Shelf's internal primary storage for that specific Wiki attachment is HTML or Markdown already. This endpoint handles the necessary steps to deliver the requested format.

#### Attachment-type Content (Document, Video, Image Gems)

Some content items in Shelf are specifically intended as attachment-type Gems, where the primary value is the file attached rather than a large body of structured Wiki content. More details about this type of content items can be found [here](https://docs.shelf.io/dev-portal/data-ingestion/content-item-structure#attachments).&#x20;

If you need to extract an attachment from such Gems, you first should be aware of the following:

{% hint style="warning" %}
Shelf stores all binary attachments (such as documents, PDFs, images, videos, etc.) in a secure object storage. So, instead of directly downloading any attachment file by calling the dedicated Shelf API endpoint, such query returns a **signed, time-limited URL** that points directly to the respective binary file in Shelf's managed object storage. You can then download the attachment file.
{% endhint %}

For obtaining the direct object storage URL attachment file download link, you need to call the following API endpoint:&#x20;

## Download gem's file attachment

> User must have access to view the gem in order to download its attachment

```json
{"openapi":"3.1.0","info":{"title":"Shelf KMS Gems API","version":"1.0.0"},"servers":[{"description":"US region","url":"https://api.shelf.io"},{"description":"EU region","url":"https://api.shelf-eu.com"},{"description":"CA region","url":"https://api.shelf-ca.com"}],"security":[{"auth_token":[]}],"components":{"securitySchemes":{"auth_token":{"type":"apiKey","description":"Shelf API Token. See [Create API Token](https://docs.shelf.io/dev-portal/authentication/create-api-token) for more details.","in":"header","name":"Authorization"}}},"paths":{"/content/v1/gems/{id}/attachment":{"get":{"operationId":"downloadGemAttachment","summary":"Download gem's file attachment","description":"User must have access to view the gem in order to download its attachment","parameters":[{"description":"ID of the gem with file attachment","schema":{"type":"string","maximum":64,"minimum":6},"in":"path","name":"id","required":true},{"description":"Set this to get download URL in response body as JSON instead of 302 redirect","schema":{"type":"string","enum":["true"]},"in":"query","name":"skipRedirect","required":false},{"description":"Version of the file attachment to download. Defaults to the latest attachment version.\nRequires `variant` query param set if used\n","schema":{"type":"string","maximum":64,"minimum":10},"in":"query","name":"version","required":false},{"description":"Indicates which file attachment to download from a specific version.\nCan be used only if `version` query param is set\n","schema":{"type":"string","enum":["old","new"]},"in":"query","name":"variant","required":false},{"description":"Indicates how many minutes URL will be valid.\n","schema":{"type":"integer","default":5,"maximum":480,"minimum":1},"in":"query","name":"expiresInMinutes","required":false}],"responses":{"200":{"description":"Success - when `skipRedirect` query string is set","content":{"application/json":{"schema":{"type":"object","properties":{"url":{"type":"string"}}}}}},"302":{"description":"Success - default","headers":{"Location":{"schema":{"type":"string","description":"Presigned URL for file on AWS S3, valid for 5 minutes"}}}},"403":{"description":"You don't have permission to download this gem","content":{"application/json":{"schema":{"type":"object"}}}},"404":{"description":"No such gem","content":{"application/json":{"schema":{"type":"object"}}}},"422":{"description":"Gem has no attachment to download","content":{"application/json":{"schema":{"type":"object","title":"API Error with code","required":["error"],"properties":{"error":{"type":"object","description":"API Error with code","required":["status","message","code"],"properties":{"code":{"type":"string","description":"Error code"},"message":{"type":"string","description":"Error message"},"status":{"type":"number","description":"Status code"}}}}}}}}},"tags":["Download"]}}}}
```

Once you get the response with the object storage URL link, you can follow it to download the attachment file.

{% hint style="warning" %}
Note that **object storage URLs** you expect to retrieve **have a temporary nature with a limited expiration** **period** (**5 minutes** in our case). So, in order to not needing to rerun the API requests due to the expired/inaccessible object storage URL, make sure to download the attachment(s) immediately after obtaining them.
{% endhint %}

But in this case, you may also face a problem: the original attachment may be in a format other than needed. To resolve this issue, use the conversion API request `https://api.shelf.io/cil-content/items/{itemId}/attachments/{attachmentId}/content/converted/{extension}` as already shown above in Step **(4b)** above.

For obtaining the `attachmentId` parameter value needed for the above API request, you can run <mark style="color:green;">`GET`</mark>` ``https://api.shelf.io/cil-content/items/{itemId}` API request, detailed in Step **(4a)**, to retrieve the specific content item details, including the information about its attachments.

<details>

<summary>Example of Response with Attachment Ids</summary>

```json
{
  "id": "01ABCD...",
  ...
  "attachments": [
    {
      "attachmentId": "168b4140d6755b1246c6f58c81bed78c",
      "extension": "html",
      ...
    },
    {
      "attachmentId": "e3b631990771361db5779b714fece9b3",
      "extension": "pdf",
      ...
    }
  ]
}
```

</details>

In the response, find and copy `attachmentId` for the file type/format you want (check the `"extension"` field: `html`, `pdf`, `md`) and use it in the [Download File Attachment by Extension Format API call](#get-cil-content-items-itemid-attachments-attachmentid-content-converted-extension) detailed above.

#### **Decision Trees**

> **Decision Tree Gems** are interactive knowledge artifacts within the Shelf Knowledge Management System (KMS) that guide users through a structured series of questions and answers to reach specific solutions or outcomes. Unlike static content types, Decision Trees represent branching logic that adapts based on user input, making them ideal for troubleshooting processes, diagnostic workflows, and guided decision-making.

Shelf Decision Trees allow you to model complex workflows, guided processes, or decision logic within your knowledge base. Below are practical approaches for extracting the content and structure of Decision Trees for integration with downstream AI or automation workflows.

**Exporting the Entire Decision Tree as PDF**

You can easily download the full visual representation of a Decision Tree in PDF format for reference or sharing:

## Download gem's PDF preview

> User must have access to view the gem in order to download its pdf preview

```json
{"openapi":"3.1.0","info":{"title":"Shelf KMS Gems API","version":"1.0.0"},"servers":[{"description":"US region","url":"https://api.shelf.io"},{"description":"EU region","url":"https://api.shelf-eu.com"},{"description":"CA region","url":"https://api.shelf-ca.com"}],"security":[{"auth_token":[]}],"components":{"securitySchemes":{"auth_token":{"type":"apiKey","description":"Shelf API Token. See [Create API Token](https://docs.shelf.io/dev-portal/authentication/create-api-token) for more details.","in":"header","name":"Authorization"}}},"paths":{"/content/v1/gems/{id}/attachment/preview/pdf":{"get":{"operationId":"downloadGemPDFPreview","summary":"Download gem's PDF preview","description":"User must have access to view the gem in order to download its pdf preview","parameters":[{"description":"ID of the gem with pdf preview","schema":{"type":"string","maximum":64,"minimum":6},"in":"path","name":"id","required":true},{"description":"Set this to get download URL in response body as JSON instead of 302 redirect","schema":{"type":"string","enum":["true"]},"in":"query","name":"skipRedirect","required":false},{"description":"Version of the file PDF preview. Defaults to the latest PDF preview version.\nRequires `variant` query param set if used\n","schema":{"type":"string","maximum":64,"minimum":10},"in":"query","name":"version","required":false},{"description":"Indicates which file attachment to download from a specific version.\nCan be used only if `version` query param is set\n","schema":{"type":"string","enum":["old","new"]},"in":"query","name":"variant","required":false}],"responses":{"200":{"description":"Success - when `skipRedirect` query string is set","content":{"application/json":{"schema":{"type":"object","properties":{"url":{"type":"string"}}}}}},"302":{"description":"Success - default","headers":{"Location":{"schema":{"type":"string","description":"Presigned URL for file on AWS S3, valid for 5 minutes"}}}},"403":{"description":"You don't have permission to download PDF preview","content":{"application/json":{"schema":{"type":"object"}}}},"404":{"description":"No such gem","content":{"application/json":{"schema":{"type":"object"}}}},"422":{"description":"Gem has no PDF preview","content":{"application/json":{"schema":{"type":"object"}}}}},"tags":["Previews"]}}}}
```

{% hint style="warning" %}
Make sure to include the required `skipRedirect=true` parameter in your API request URL. This parameter is responsible for providing you with the PDF file download link. If you don't include this parameter, you will get the binary in the response.
{% endhint %}

#### Retrieving Decision Tree Structure and Steps

To integrate Decision Trees with AI tools or for detailed analysis, you need to extract their full structure, which includes all steps and their types.&#x20;

## Get Decision Tree Steps Settings

> Getting Decision Tree Steps and Settings using either a Gem Id, an Archived Gem Id, or a Draft Id.\
> \- Use \`gemId\` to retrieve steps and settings for an active Decision Tree Gem.\
> \- Use \`archivedGemId\` to retrieve steps and settings for an archived Decision Tree Gem.\
> \- Use \`draftId\` to retrieve steps and settings for a Decision Tree Draft.\
> \- Only one of \`gemId\`, \`archivedGemId\`, or \`draftId\` should be provided in a single request.\
>

```json
{"openapi":"3.0.0","info":{"title":"Shelf Decision Trees API","version":"1.0.0"},"servers":[{"description":"US region","url":"https://api.shelf.io"},{"description":"EU region","url":"https://api.shelf-eu.com"},{"description":"CA region","url":"https://api.shelf-ca.com"}],"security":[{"auth_token":[]}],"components":{"securitySchemes":{"auth_token":{"type":"apiKey","description":"Shelf API Token. See [Create API Token](https://docs.shelf.io/dev-portal/authentication/create-api-token) for more details.","in":"header","name":"Authorization"}}},"paths":{"/decision-tree/steps-settings":{"get":{"operationId":"getStepsSettings","summary":"Get Decision Tree Steps Settings","description":"Getting Decision Tree Steps and Settings using either a Gem Id, an Archived Gem Id, or a Draft Id.\n- Use `gemId` to retrieve steps and settings for an active Decision Tree Gem.\n- Use `archivedGemId` to retrieve steps and settings for an archived Decision Tree Gem.\n- Use `draftId` to retrieve steps and settings for a Decision Tree Draft.\n- Only one of `gemId`, `archivedGemId`, or `draftId` should be provided in a single request.\n","parameters":[{"description":"Id of Decision Tree Gem","schema":{"type":"string","maxLength":64,"minLength":6},"in":"query","name":"gemId","required":true},{"description":"Id of Decision Tree Draft","schema":{"type":"string","maxLength":26,"minLength":26},"in":"query","name":"draftId","required":true},{"description":"Id of archived Decision Tree Gem","schema":{"type":"string","maxLength":64,"minLength":6},"in":"query","name":"archivedGemId","required":true}],"responses":{"200":{"description":"Get Decision Tree Steps Settings","content":{"application/json":{"schema":{"type":"object","properties":{"settings":{"type":"object","title":"Decision Tree Settings","properties":{"isAllowedNavigateToStep":{"type":"boolean"},"isContextualVariablesEnabled":{"type":"boolean"},"isMinimizedViewEnabled":{"type":"boolean"},"isOpenDTLinkInNewTabEnabled":{"type":"boolean"}}},"steps":{"type":"array","items":{"anyOf":[{"allOf":[{"type":"object","title":"Question","required":["id","type"],"properties":{"type":{"type":"string","enum":["Question"]},"id":{"type":"string","maxLength":26,"minLength":6},"isFirstStep":{"type":"boolean","description":"The first step of the tree (only one for the tree)"},"question":{"type":"object","required":["id"],"properties":{"description":{"type":"string"},"answers":{"type":"array","description":"List of answers","items":{"type":"object","required":["id"],"properties":{"assignments":{"type":"array","description":"List of assignments with Contextual Variables","maxItems":100,"minItems":0,"items":{"type":"object","required":["variableId","value"],"properties":{"value":{"type":"string","description":"Value of Contextual Variable","maximum":250,"minimum":1},"variableId":{"type":"string","maxLength":26,"minLength":26}}}},"id":{"type":"string","maxLength":26,"minLength":6},"nextStep":{"type":"object","required":["id"],"properties":{"id":{"type":"string","maxLength":26,"minLength":6}}},"title":{"type":"string","description":"Title of the answer"}}}},"id":{"type":"string","maxLength":26,"minLength":6},"title":{"type":"string","description":"Title of the question"}}},"settings":{"type":"object","properties":{"borderColor":{"type":"string","description":"The color of the step border. Must be in the `Hexadecimal(#RRGGBB)` format"},"position":{"type":"object","description":"The positon of the step by `X`,`Y` coordinate axes","required":["x","y"],"properties":{"x":{"type":"number"},"y":{"type":"number"}}}}},"title":{"type":"string"}}},{"type":"object","properties":{"textS3Key":{"type":"string","description":"The S3 key of the tree step text"}}}]},{"allOf":[{"type":"object","title":"Solution","required":["id","type"],"properties":{"type":{"type":"string","enum":["Solution"]},"id":{"type":"string","maxLength":26,"minLength":6},"settings":{"type":"object","properties":{"borderColor":{"type":"string","description":"The color of the step border. Must be in the `Hexadecimal(#RRGGBB)` format"},"position":{"type":"object","description":"The positon of the step by `X`,`Y` coordinate axes","required":["x","y"],"properties":{"x":{"type":"number"},"y":{"type":"number"}}}}},"title":{"type":"string"}}},{"type":"object","properties":{"textS3Key":{"type":"string","description":"The S3 key of the tree step text"}}}]},{"allOf":[{"type":"object","title":"Link","required":["id","type"],"properties":{"type":{"type":"string","enum":["Link"]},"id":{"type":"string","maxLength":26,"minLength":6},"settings":{"type":"object","properties":{"borderColor":{"type":"string","description":"The color of the step border. Must be in the `Hexadecimal(#RRGGBB)` format"},"position":{"type":"object","description":"The positon of the step by `X`,`Y` coordinate axes","required":["x","y"],"properties":{"x":{"type":"number"},"y":{"type":"number"}}}}},"title":{"type":"string"},"url":{"type":"string","description":"The link to redirect. Begins with \"https://\""}}}]},{"allOf":[{"type":"object","title":"Condition","required":["id","type"],"properties":{"type":{"type":"string","enum":["Condition"]},"conditions":{"type":"array","maxItems":100,"minItems":0,"items":{"type":"object","additionalProperties":false,"required":["id","expression"],"properties":{"expression":{"oneOf":[{"type":"object","additionalProperties":false,"required":["rules","joiner"],"properties":{"joiner":{"type":"string","enum":["and"]},"rules":{"type":"array","maxItems":5,"minItems":2,"items":{"oneOf":[{"type":"object","additionalProperties":false,"required":["id","variableId","operator","value"],"properties":{"id":{"type":"string","maxLength":26,"minLength":26},"operator":{"type":"string","enum":["eq","neq","contains","not_contains"]},"value":{"type":"string","maxLength":250,"minLength":1},"variableId":{"type":"string","maxLength":26,"minLength":26}}},{"type":"object","additionalProperties":false,"title":"Complex Expression with joiner","required":["rules","joiner"],"properties":{"joiner":{"type":"string","enum":["and"]},"rules":{"type":"array","maxItems":5,"minItems":2,"items":{"oneOf":[{"type":"object","additionalProperties":false,"required":["id","variableId","operator","value"],"properties":{"id":{"type":"string","maxLength":26,"minLength":26},"operator":{"type":"string","enum":["eq","neq","contains","not_contains"]},"value":{"type":"string","maxLength":250,"minLength":1},"variableId":{"type":"string","maxLength":26,"minLength":26}}},{"type":"object","title":"Complex Expression with joiner"},{"type":"object","title":"Complex Expression without joiner"}]}}}},{"type":"object","additionalProperties":false,"title":"Complex Expression without joiner","required":["rules"],"properties":{"rules":{"type":"array","maxItems":1,"minItems":1,"items":{"oneOf":[{"type":"object","additionalProperties":false,"required":["id","variableId","operator","value"],"properties":{"id":{"type":"string","maxLength":26,"minLength":26},"operator":{"type":"string","enum":["eq","neq","contains","not_contains"]},"value":{"type":"string","maxLength":250,"minLength":1},"variableId":{"type":"string","maxLength":26,"minLength":26}}},{"type":"object","title":"Complex Expression with joiner"},{"type":"object","title":"Complex Expression without joiner"}]}}}}]}}}},{"type":"object","additionalProperties":false,"required":["rules"],"properties":{"rules":{"type":"array","maxItems":1,"minItems":1,"items":{"oneOf":[{"type":"object","additionalProperties":false,"required":["id","variableId","operator","value"],"properties":{"id":{"type":"string","maxLength":26,"minLength":26},"operator":{"type":"string","enum":["eq","neq","contains","not_contains"]},"value":{"type":"string","maxLength":250,"minLength":1},"variableId":{"type":"string","maxLength":26,"minLength":26}}},{"type":"object","additionalProperties":false,"title":"Complex Expression with joiner","required":["rules","joiner"],"properties":{"joiner":{"type":"string","enum":["and"]},"rules":{"type":"array","maxItems":5,"minItems":2,"items":{"oneOf":[{"type":"object","additionalProperties":false,"required":["id","variableId","operator","value"],"properties":{"id":{"type":"string","maxLength":26,"minLength":26},"operator":{"type":"string","enum":["eq","neq","contains","not_contains"]},"value":{"type":"string","maxLength":250,"minLength":1},"variableId":{"type":"string","maxLength":26,"minLength":26}}},{"type":"object","title":"Complex Expression with joiner"},{"type":"object","title":"Complex Expression without joiner"}]}}}},{"type":"object","additionalProperties":false,"title":"Complex Expression without joiner","required":["rules"],"properties":{"rules":{"type":"array","maxItems":1,"minItems":1,"items":{"oneOf":[{"type":"object","additionalProperties":false,"required":["id","variableId","operator","value"],"properties":{"id":{"type":"string","maxLength":26,"minLength":26},"operator":{"type":"string","enum":["eq","neq","contains","not_contains"]},"value":{"type":"string","maxLength":250,"minLength":1},"variableId":{"type":"string","maxLength":26,"minLength":26}}},{"type":"object","title":"Complex Expression with joiner"},{"type":"object","title":"Complex Expression without joiner"}]}}}}]}}}}]},"id":{"type":"string","maxLength":26,"minLength":26},"nextStep":{"type":"object","required":["id"],"properties":{"id":{"type":"string","maxLength":26,"minLength":6}}}}}},"defaultNextStep":{"type":"object","required":["id"],"properties":{"id":{"type":"string","maxLength":26,"minLength":6}}},"id":{"type":"string","maxLength":26,"minLength":6},"settings":{"type":"object","properties":{"borderColor":{"type":"string","description":"The color of the step border. Must be in the `Hexadecimal(#RRGGBB)` format"},"position":{"type":"object","description":"The positon of the step by `X`,`Y` coordinate axes","required":["x","y"],"properties":{"x":{"type":"number"},"y":{"type":"number"}}}}},"title":{"type":"string","maxLength":250,"minLength":0}}}]}]}}}}}}},"400":{"description":"Bad Request","content":{"application/json":{"schema":{"type":"object","title":"Bad request payload (AJV)","required":["error"],"properties":{"error":{"type":"object","description":"API Error With detail","required":["status","message","detail"],"properties":{"detail":{"type":"array","description":"Detail info","items":{"type":"object","properties":{"instancePath":{"type":"string"},"keyword":{"type":"string"},"message":{"type":"string"},"params":{"type":"object"},"schemaPath":{"type":"string"}}}},"message":{"type":"string","description":"Error message"},"status":{"type":"number","description":"Status code"}}}}}}}},"403":{"description":"Forbidden","content":{"application/json":{"schema":{"type":"object","title":"API Error with code","required":["error"],"properties":{"error":{"type":"object","description":"API Error with code","required":["status","message","code"],"properties":{"code":{"type":"string","description":"Error code"},"message":{"type":"string","description":"Error message"},"status":{"type":"number","description":"Status code"}}}}}}}},"404":{"description":"Not Found","content":{"application/json":{"schema":{"type":"object","title":"API Error with code","required":["error"],"properties":{"error":{"type":"object","description":"API Error with code","required":["status","message","code"],"properties":{"code":{"type":"string","description":"Error code"},"message":{"type":"string","description":"Error message"},"status":{"type":"number","description":"Status code"}}}}}}}}},"tags":["Decision Trees"]}}}}
```

The example query would look like this

<mark style="color:green;">`GET`</mark>` ``https://api.shelf.io/decision-tree/steps-settings?gemId=de0cba94-c11d-4d75-811c-301e4fe2ab62`

**Path Parameters:**&#x20;

* `gemId`: Same as `externalId` for individual content items, and the **mandatory** parameter

This endpoint returns a JSON object containing the entire Decision Tree definition, including an array named `steps`. Each object within this array represents a single step (question, solution, or link) in the tree.

<details>

<summary>Example Response with Decision Tree Structure (Steps)</summary>

```json
{
    "steps": [
        {
            "id": "01J08ZXEG5VSKXYNGA8XBWSN27",
            "type": "Question",
            "title": "Do you need an Alfa Romeo car?",
            "settings": {
                "position": {
                    "x": 0,
                    "y": 0
                }
            },
            "question": {
                "id": "01J08ZXEG52KF7X66JZCS9VWS8",
                "title": "Do you need an Alfa Romeo car",
                "description": "",
                "answers": [
                    {
                        "id": "01J08ZXEG5VXXTWRZDD454JKBE",
                        "title": "Yes",
                        "nextStep": {
                            "id": "01J0903QZTKXAF405H3EVXEV9B"
                        }
                    },
                    {
                        "id": "01J0902YYXKHVV3M52Z7BXNB9F",
                        "title": "No",
                        "nextStep": {
                            "id": "01J09042WAGTKP51R733T43NMS"
                        }
                    }
                ]
            },
            "isFirstStep": true
        },
        {
            "id": "01J0903QZTKXAF405H3EVXEV9B",
            "type": "Question",
            "title": "",
            "textS3Key": "60b11695-41a0-4d65-9b44-1638f5bfa206/gems/de0cba94-c11d-4d75-811c-301e4fe2ab62/steps/step_01J0903QZTKXAF405H3EVXEV9B_2024-06-13T14:37:27.005Z.html",
            "settings": {
                "position": {
                    "x": 500,
                    "y": 0
                }
            },
            "question": {
                "id": "01J0903QZTMJ1JEPCP7HKZNNSC",
                "title": "Do you need a modern or a vintage model?",
                "description": "",
                "answers": [
                    {
                        "id": "01J0903QZTX7SNSNTBZPPBEKDA",
                        "title": "Modern model",
                        "nextStep": {
                            "id": "01J0907K8YC4TDWHZ18471JQQ7"
                        }
                    },
                    {
                        "id": "01J090704HSBJEB4Z8J50JSTE4",
                        "title": "Vintage model",
                        "nextStep": {
                            "id": "01J0909DGZHW9QYNNBVFXCTKVS"
                        }
                    }
                ]
            }
        },
        {
            "id": "01J09042WAGTKP51R733T43NMS",
            "type": "Solution",
            "title": "Refer to Other Brand Car Selection Procedure",
            "settings": {
                "position": {
                    "x": 160,
                    "y": 310
                }
            }
        },
        {
            "id": "01J0907K8YC4TDWHZ18471JQQ7",
            "type": "Question",
            "title": "",
            "textS3Key": "60b11695-41a0-4d65-9b44-1638f5bfa206/gems/de0cba94-c11d-4d75-811c-301e4fe2ab62/steps/step_01J0907K8YC4TDWHZ18471JQQ7_2024-06-13T14:37:27.005Z.html",
            "settings": {
                "position": {
                    "x": 920,
                    "y": -90
                }
            },
            "question": {
                "id": "01J0907K8YR341QGXBXBAHMGR5",
                "title": "Do you need a sport or an everyday car?",
                "description": "",
                "answers": [
                    {
                        "id": "01J0907K8Y5JT8REZAWFCV5TB8",
                        "title": "Sport car",
                        "nextStep": {
                            "id": "01J0909DGZHW9QYNNBVFXCTKVS"
                        }
                    },
                    {
                        "id": "01J0908RN699YDNRBR1KVD9ER5",
                        "title": "Family or everyday car",
                        "nextStep": {
                            "id": "01J090EF430DP8G1QBZPQDNRME"
                        }
                    }
                ]
            }
        },
        {
            "id": "01J0909DGZHW9QYNNBVFXCTKVS",
            "type": "Solution",
            "title": "Alfa Romeo Giulietta Sprint Speciale",
            "textS3Key": "60b11695-41a0-4d65-9b44-1638f5bfa206/gems/de0cba94-c11d-4d75-811c-301e4fe2ab62/steps/step_01J0909DGZHW9QYNNBVFXCTKVS_2024-06-13T14:37:27.005Z.html",
            "settings": {
                "position": {
                    "x": 920,
                    "y": 180
                }
            }
        },
        {
            "id": "01J090EF430DP8G1QBZPQDNRME",
            "type": "Solution",
            "title": "Alfa Romeo Stelvio",
            "textS3Key": "60b11695-41a0-4d65-9b44-1638f5bfa206/gems/de0cba94-c11d-4d75-811c-301e4fe2ab62/steps/step_01J090EF430DP8G1QBZPQDNRME_2024-06-13T14:37:27.005Z.html",
            "settings": {
                "position": {
                    "x": 1300,
                    "y": -20
                }
            }
        }
    ],
    "settings": {
        "isAllowedNavigateToStep": true
    }
}
```

</details>

**Understanding Decision Tree Step IDs**

The `id` field within each step object in the response (e.g., `"01QSTN1"`) serves as a unique identifier for that step within this specific Decision Tree structure. The `nextStepId` field found in "`question`" type steps uses these IDs to map the logical flow. Consult the Shelf API documentation to confirm if these step IDs remain persistent if the tree is modified, or if they should be treated as specific to the retrieved version.

The `type` field in the response helps you identify the purpose of each step:

* `question`: prompts user input
* `solution`: provides an answer or instruction
* `link`: references external content.

#### Extracting Decision Tree Steps by Specific Type

You can programmatically select just the steps of a certain type—such as `questions`, `solutions` (answers), or `links`—by filtering the `steps` array received in the API response from the `/steps-settings` endpoint. No filter is required in the API request payload as it returns all the steps. The filtering is performed on the client side. The logic here is as follows:

```
filteredSteps = allSteps.filter(step => step["type"] === "solution");
```

After filtering the API response, you can expect to see something like this:

<details>

<summary>Example Filtered List of Specific Decision Tree Steps</summary>

```json
[
  {
    "id": "01SOL1",
    "type": "solution",
    "title": "Proceed to diagnostics..."
  },
  {
    "id": "01SOL2",
    "type": "solution",
    "title": "Plug in the device and retry."
  },
  {
    "id": "01SOL3",
    "type": "solution",
    "title": "Contact technical support if the issue persists."
  }
]
```

</details>

#### Individual Steps of Decision Tree Gem

Once you have received the general structure of the Decision Tree, you can now pull and process each individual step of the content item.

## Get Decision Tree Step

> Get a Decision Tree Step using either Gem Id or Draft Id.\
> \- Use \`gemId\` with \`stepId\` to retrieve by Gem.\
> \- Use \`draftId\` with \`stepId\` to retrieve by Draft.\
>

```json
{"openapi":"3.0.0","info":{"title":"Shelf Decision Trees API","version":"1.0.0"},"servers":[{"description":"US region","url":"https://api.shelf.io"},{"description":"EU region","url":"https://api.shelf-eu.com"},{"description":"CA region","url":"https://api.shelf-ca.com"}],"security":[{"auth_token":[]}],"components":{"securitySchemes":{"auth_token":{"type":"apiKey","description":"Shelf API Token. See [Create API Token](https://docs.shelf.io/dev-portal/authentication/create-api-token) for more details.","in":"header","name":"Authorization"}}},"paths":{"/decision-tree/step":{"get":{"operationId":"getDTStep","summary":"Get Decision Tree Step","description":"Get a Decision Tree Step using either Gem Id or Draft Id.\n- Use `gemId` with `stepId` to retrieve by Gem.\n- Use `draftId` with `stepId` to retrieve by Draft.\n","parameters":[{"description":"Id of Decision Tree Gem","schema":{"type":"string","maxLength":64,"minLength":6},"in":"query","name":"gemId","required":true},{"description":"Id of Decision Tree Draft","schema":{"type":"string","maxLength":26,"minLength":26},"in":"query","name":"draftId","required":true},{"description":"Step Id of Decision Tree","schema":{"type":"string","maxLength":26,"minLength":6},"in":"query","name":"stepId","required":true}],"responses":{"200":{"description":"Get Decision Tree Step","content":{"application/json":{"schema":{"type":"object","properties":{"step":{"anyOf":[{"allOf":[{"type":"object","title":"Question","required":["id","type"],"properties":{"type":{"type":"string","enum":["Question"]},"id":{"type":"string","maxLength":26,"minLength":6},"isFirstStep":{"type":"boolean","description":"The first step of the tree (only one for the tree)"},"question":{"type":"object","required":["id"],"properties":{"description":{"type":"string"},"answers":{"type":"array","description":"List of answers","items":{"type":"object","required":["id"],"properties":{"assignments":{"type":"array","description":"List of assignments with Contextual Variables","maxItems":100,"minItems":0,"items":{"type":"object","required":["variableId","value"],"properties":{"value":{"type":"string","description":"Value of Contextual Variable","maximum":250,"minimum":1},"variableId":{"type":"string","maxLength":26,"minLength":26}}}},"id":{"type":"string","maxLength":26,"minLength":6},"nextStep":{"type":"object","required":["id"],"properties":{"id":{"type":"string","maxLength":26,"minLength":6}}},"title":{"type":"string","description":"Title of the answer"}}}},"id":{"type":"string","maxLength":26,"minLength":6},"title":{"type":"string","description":"Title of the question"}}},"settings":{"type":"object","properties":{"borderColor":{"type":"string","description":"The color of the step border. Must be in the `Hexadecimal(#RRGGBB)` format"},"position":{"type":"object","description":"The positon of the step by `X`,`Y` coordinate axes","required":["x","y"],"properties":{"x":{"type":"number"},"y":{"type":"number"}}}}},"title":{"type":"string"}}},{"type":"object","properties":{"text":{"type":"string","maxLength":3145728,"minLength":0}}}]},{"allOf":[{"type":"object","title":"Solution","required":["id","type"],"properties":{"type":{"type":"string","enum":["Solution"]},"id":{"type":"string","maxLength":26,"minLength":6},"settings":{"type":"object","properties":{"borderColor":{"type":"string","description":"The color of the step border. Must be in the `Hexadecimal(#RRGGBB)` format"},"position":{"type":"object","description":"The positon of the step by `X`,`Y` coordinate axes","required":["x","y"],"properties":{"x":{"type":"number"},"y":{"type":"number"}}}}},"title":{"type":"string"}}},{"type":"object","properties":{"text":{"type":"string","maxLength":3145728,"minLength":0}}}]},{"allOf":[{"type":"object","title":"Link","required":["id","type"],"properties":{"type":{"type":"string","enum":["Link"]},"id":{"type":"string","maxLength":26,"minLength":6},"settings":{"type":"object","properties":{"borderColor":{"type":"string","description":"The color of the step border. Must be in the `Hexadecimal(#RRGGBB)` format"},"position":{"type":"object","description":"The positon of the step by `X`,`Y` coordinate axes","required":["x","y"],"properties":{"x":{"type":"number"},"y":{"type":"number"}}}}},"title":{"type":"string"},"url":{"type":"string","description":"The link to redirect. Begins with \"https://\""}}}]},{"allOf":[{"type":"object","title":"Condition","required":["id","type"],"properties":{"type":{"type":"string","enum":["Condition"]},"conditions":{"type":"array","maxItems":100,"minItems":0,"items":{"type":"object","additionalProperties":false,"required":["id","expression"],"properties":{"expression":{"oneOf":[{"type":"object","additionalProperties":false,"required":["rules","joiner"],"properties":{"joiner":{"type":"string","enum":["and"]},"rules":{"type":"array","maxItems":5,"minItems":2,"items":{"oneOf":[{"type":"object","additionalProperties":false,"required":["id","variableId","operator","value"],"properties":{"id":{"type":"string","maxLength":26,"minLength":26},"operator":{"type":"string","enum":["eq","neq","contains","not_contains"]},"value":{"type":"string","maxLength":250,"minLength":1},"variableId":{"type":"string","maxLength":26,"minLength":26}}},{"type":"object","additionalProperties":false,"title":"Complex Expression with joiner","required":["rules","joiner"],"properties":{"joiner":{"type":"string","enum":["and"]},"rules":{"type":"array","maxItems":5,"minItems":2,"items":{"oneOf":[{"type":"object","additionalProperties":false,"required":["id","variableId","operator","value"],"properties":{"id":{"type":"string","maxLength":26,"minLength":26},"operator":{"type":"string","enum":["eq","neq","contains","not_contains"]},"value":{"type":"string","maxLength":250,"minLength":1},"variableId":{"type":"string","maxLength":26,"minLength":26}}},{"type":"object","title":"Complex Expression with joiner"},{"type":"object","title":"Complex Expression without joiner"}]}}}},{"type":"object","additionalProperties":false,"title":"Complex Expression without joiner","required":["rules"],"properties":{"rules":{"type":"array","maxItems":1,"minItems":1,"items":{"oneOf":[{"type":"object","additionalProperties":false,"required":["id","variableId","operator","value"],"properties":{"id":{"type":"string","maxLength":26,"minLength":26},"operator":{"type":"string","enum":["eq","neq","contains","not_contains"]},"value":{"type":"string","maxLength":250,"minLength":1},"variableId":{"type":"string","maxLength":26,"minLength":26}}},{"type":"object","title":"Complex Expression with joiner"},{"type":"object","title":"Complex Expression without joiner"}]}}}}]}}}},{"type":"object","additionalProperties":false,"required":["rules"],"properties":{"rules":{"type":"array","maxItems":1,"minItems":1,"items":{"oneOf":[{"type":"object","additionalProperties":false,"required":["id","variableId","operator","value"],"properties":{"id":{"type":"string","maxLength":26,"minLength":26},"operator":{"type":"string","enum":["eq","neq","contains","not_contains"]},"value":{"type":"string","maxLength":250,"minLength":1},"variableId":{"type":"string","maxLength":26,"minLength":26}}},{"type":"object","additionalProperties":false,"title":"Complex Expression with joiner","required":["rules","joiner"],"properties":{"joiner":{"type":"string","enum":["and"]},"rules":{"type":"array","maxItems":5,"minItems":2,"items":{"oneOf":[{"type":"object","additionalProperties":false,"required":["id","variableId","operator","value"],"properties":{"id":{"type":"string","maxLength":26,"minLength":26},"operator":{"type":"string","enum":["eq","neq","contains","not_contains"]},"value":{"type":"string","maxLength":250,"minLength":1},"variableId":{"type":"string","maxLength":26,"minLength":26}}},{"type":"object","title":"Complex Expression with joiner"},{"type":"object","title":"Complex Expression without joiner"}]}}}},{"type":"object","additionalProperties":false,"title":"Complex Expression without joiner","required":["rules"],"properties":{"rules":{"type":"array","maxItems":1,"minItems":1,"items":{"oneOf":[{"type":"object","additionalProperties":false,"required":["id","variableId","operator","value"],"properties":{"id":{"type":"string","maxLength":26,"minLength":26},"operator":{"type":"string","enum":["eq","neq","contains","not_contains"]},"value":{"type":"string","maxLength":250,"minLength":1},"variableId":{"type":"string","maxLength":26,"minLength":26}}},{"type":"object","title":"Complex Expression with joiner"},{"type":"object","title":"Complex Expression without joiner"}]}}}}]}}}}]},"id":{"type":"string","maxLength":26,"minLength":26},"nextStep":{"type":"object","required":["id"],"properties":{"id":{"type":"string","maxLength":26,"minLength":6}}}}}},"defaultNextStep":{"type":"object","required":["id"],"properties":{"id":{"type":"string","maxLength":26,"minLength":6}}},"id":{"type":"string","maxLength":26,"minLength":6},"settings":{"type":"object","properties":{"borderColor":{"type":"string","description":"The color of the step border. Must be in the `Hexadecimal(#RRGGBB)` format"},"position":{"type":"object","description":"The positon of the step by `X`,`Y` coordinate axes","required":["x","y"],"properties":{"x":{"type":"number"},"y":{"type":"number"}}}}},"title":{"type":"string","maxLength":250,"minLength":0}}}]}]}}}}}},"400":{"description":"Bad Request","content":{"application/json":{"schema":{"type":"object","title":"Bad request payload (AJV)","required":["error"],"properties":{"error":{"type":"object","description":"API Error With detail","required":["status","message","detail"],"properties":{"detail":{"type":"array","description":"Detail info","items":{"type":"object","properties":{"instancePath":{"type":"string"},"keyword":{"type":"string"},"message":{"type":"string"},"params":{"type":"object"},"schemaPath":{"type":"string"}}}},"message":{"type":"string","description":"Error message"},"status":{"type":"number","description":"Status code"}}}}}}}},"403":{"description":"Forbidden","content":{"application/json":{"schema":{"type":"object","title":"API Error with code","required":["error"],"properties":{"error":{"type":"object","description":"API Error with code","required":["status","message","code"],"properties":{"code":{"type":"string","description":"Error code"},"message":{"type":"string","description":"Error message"},"status":{"type":"number","description":"Status code"}}}}}}}},"404":{"description":"Not Found","content":{"application/json":{"schema":{"type":"object","title":"API Error with code","required":["error"],"properties":{"error":{"type":"object","description":"API Error with code","required":["status","message","code"],"properties":{"code":{"type":"string","description":"Error code"},"message":{"type":"string","description":"Error message"},"status":{"type":"number","description":"Status code"}}}}}}}}},"tags":["Decision Trees"]}}}}
```

**Query Parameters:**

* This endpoint requires **both** the `gemId` of the Decision Tree and the specific `stepId` of the step you want to retrieve.&#x20;

<details>

<summary>Example API Query URL</summary>

```
https://api.shelf.io/decision-tree/step?gemId=YOUR_DECISION_TREE_EXTERNAL_ID&stepId=STEP_ID_FROM_STRUCTURE
```

</details>

#### Handling Steps by Their Type

Process each step sequentially according to its type: Iterate through the `Steps` array, check each step’s `"type"`, and extract the relevant field(s) as shown below.

<details>

<summary><strong>(a)</strong> Extract question content from the <strong>Question</strong> type steps</summary>

```json
/{
  "id": "01QSTN1",
  "type": "question",
  "title": "Is the device powered on?",
  "answers": [
    { "text": "Yes", "nextStepId": "01SOL1" },
    { "text": "No",  "nextStepId": "01SOL2" }
  ]
}
```

</details>

**Extraction:**

* `question`: `"Is the device powered on?"` (from `"title"`)
* `answers`: `"Yes"`, `"No"` (from `answers[].text`)

<details>

<summary><strong>(b)</strong> Capture solution information from the <strong>Solution</strong> type steps</summary>

```json
{
  "id": "01SOL1",
  "type": "solution",
  "title": "Proceed to diagnostics."
}
```

</details>

**Extraction:**

* `solution`: `"Proceed to diagnostics."` (from `"title"`)

<details>

<summary><strong>(c)</strong> Process URL references from the <strong>Link</strong> type steps</summary>

```json
{
  "id": "01LNK1",
  "type": "link",
  "title": "View troubleshooting guide",
  "url": "https://example.com/troubleshooting"
}
```

</details>

**Extraction:**

* `linkTitle`: `"View troubleshooting guide"` (from `"title"`)
* `url`: `"https://example.com/troubleshooting"` (from `"url"`)



**(d)** Map the logical flow of the Decision Tree: Use the `id` and `nextStepId` fields (primarily from question steps) to understand and potentially replicate the branching logic of the Decision Tree for AI context understanding.

This structured approach facilitates AI integration, enabling it to navigate and utilize decision trees effectively.

#### Handling Non-text Content in Steps

But what if a Decision Tree step contains not a text, but an image, diagram, or other non-textual data, such as video or music?&#x20;

If a step contains an **image,** a **video** or a **musical track (audio)**, your query response will have a direct URL to the media. Save both the `title` and the non-textual content URL (for accessing, watching/listening, or further processing).&#x20;

<details>

<summary><strong>Example Step with Image Only</strong></summary>

<pre class="language-json" data-overflow="wrap"><code class="lang-json">{
    "step": {
        "id": "01J0909DGZHW9QYNNBVFXCTKVS",
        "type": "Solution",
        "title": "Alfa Romeo Giulietta Sprint Speciale",
        "settings": {
            "position": {
                "x": 920,
                "y": 180
            }
        },
        "gemId": "de0cba94-c11d-4d75-811c-301e4fe2ab62",
        "text": "&#x3C;div>&#x3C;br />&#x3C;/div>&#x3C;div>&#x3C;br />&#x3C;/div>&#x3C;div><a data-footnote-ref href="#user-content-fn-1">&#x3C;img src=\"https://thumbnails.shelf.io/60b11695-41a0-4d65-9b44-1638f5bfa206%2Fdrafts%2F01JT67F4HAQ6ZG3Y7TNJBN6TD3%2Fwiki%2Fimages%2F1746113505535-1961-Alfa-Romeo-Giulietta-SS-by-Bertone-110.jpg\"</a> style=\"width:826px\" class=\"fr-fic fr-dib\" />&#x3C;/div>"
    }
}
</code></pre>

</details>

### **5. Content Integration**

**Action**: _Prepare the extracted content for AI consumption._

One of the most important actions preceding your content integration into AI applications is to structure this content in a consistent format.&#x20;

Format all content, which you have extracted as a result of the respective API requests above, in a predictable structure, such as _**markdown**_ files or plain text.\
The best practice would be to start each file with a clear title line (e.g., `# Document Title`), optionally include metadata like tags or categories, followed by the main content.\
This consistent structure ensures the AI or retrieval pipeline can reliably segment, parse, and understand each document’s contex&#x74;_._

<details>

<summary>Example of Resulting Markdown File</summary>

```markdown
# How to Close an Account
**Tags:** accounts, closure, banking

Step-by-step instructions for properly closing a customer account...
```

</details>

## **Benefits**

* **Single Source of Truth**: Maintain consistency between knowledge management and AI responses.
* **Updated Information**: Automated processes ensure AI accesses the latest information.
* **Content Type Diversity**: Incorporate various content formats (text, decision flows, files) into AI knowledge.
* **Scalability**: Efficiently process large knowledge bases through API automation.

## **Success Metrics**

* Reduction in AI response inaccuracies.
* Decreased time spent manually updating AI knowledge sources.
* Improved user satisfaction with AI application responses.
* Higher consistency between information provided through human and AI channels.

## **Technical Requirements**

* Development resources familiar with REST API integration.
* Secure credential management for API tokens.
* Processing capacity for content extraction and transformation.
* Storage solution for processed content.

This implementation enables organizations to leverage their existing Shelf knowledge base as a comprehensive foundation for AI applications, ensuring consistent and accurate information delivery across all channels.



[^1]: Direct URL to a non-textual media in the Decision Tree step
