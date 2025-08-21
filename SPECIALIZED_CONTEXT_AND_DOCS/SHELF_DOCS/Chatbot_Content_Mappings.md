# Chatbot Content Mappings

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://shelf.io)

Agenda

* Fields in the CSV
* Create CSV
* Submit CSV
* Get mapped values
* Magics

## Fields in the CSV

| field            | type                                     | description                                                                                                                                                                                                                     | example                                                                                                                                                         |
| ---------------- | ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| intent           | string                                   | Uniq code for the system combination of the intentGroupId and intent that will be used to identify what data should be retrieved to the requester                                                                               | `some-intent`                                                                                                                                                   |
| intentGroupId    | string                                   | group label for few intents                                                                                                                                                                                                     | `someSupportLabel`                                                                                                                                              |
| mappingValue     | url                                      | value of the mapping, for mappingType GEM it would be gem url, for the search it would be full search link that you want to retrieve                                                                                            | `https://SHELF_DOMAIN/some-gem-id` **OR** `https://SHELF_DOMAIN/?cp=1&fi=917ff821-7b9e-1111-938b-8a9a27asgfsag6&it=snippet&ps=20&sb=RELEVANCE&so=ASC&term=help` |
| mappingType      | string enum: **GEM** OR **SEARCH**       | defines what type of the lookup you want to setup by mapping value                                                                                                                                                              | `GEM`                                                                                                                                                           |
| allowedLibraries | String array. Array members split by `,` | list of the links to allowed libraries for document fetch. If you would set up gemId without library gem id listed here, it would not fetch anything, same for the search. Search will only be used in the provided library ids | `library-gem-url,library-gem-url2,library-gem-url3`                                                                                                             |

### CSV Structure example

| intent | intentGroupId | mappingValue                                                                                                                | mappingType | allowedLibraries                                                                                                                   |
| ------ | ------------- | --------------------------------------------------------------------------------------------------------------------------- | ----------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| 1      | 88005553535   | https://SHELF-DOMAIN.shelf.io/d13b48ec-0799-46ec-af10-caigj5353sv                                                           | GEM         | https://SHELF-DOMAIN.shelf.io/c917ff821-7b9e-4a69-938b-gjbjb24442xb                                                                |
| 2      | 88005553535   | https://SHELF-DOMAIN.shelf.io/cp=1\&fi=917ff821-7b9e-4ass-938b-9asdgu91\&it=snippet\&ps=20\&sb=RELEVANCE\&so=ASC\&term=help | SEARCH      | https://SHELF-DOMAIN.shelf.io/c917ff821-7b9e-4a69-938b-adfgasfsaf,https://SHELF-DOMAIN.shelf.io/c917ff821-7b9e-4a69-938b-adghsdjgb |

### Ready to go csv example

https://docs.google.com/spreadsheets/d/1Xz35Nq6YZtgI\_ebRzc7k7KrgniVEIWDQBZ7KbNqF-TY/edit?usp=sharing

### Submit CSV file

Go to the Admin Panel at your Shelf, and submit it in the according section.

### Limitations

1. Favorites filter will not be used for the document fetch. So the result might be slightly different from what you saw
2. Search or Gem fetch will be used only for the provided allowedLibraries
3. csv should be valid and contain at least one mapping

## Get mapped values

Get values of the mapped documents in response. Response allays will be _**array**_

### Request description

Request type: _**GET**_ Request url: _**/chatbot/{accountId}/{intentGroupId}/{intent}**_ **!!** _**accountId**_ - your shelf account ID **!!** Request parameters should be used from the CVS that you imported earlier in the _**Bulk Import**_ section.

### Responses

List of the possible responses:

| status code | description                                                  | value                                                                      |
| ----------- | ------------------------------------------------------------ | -------------------------------------------------------------------------- |
| 200         | fetched requested documents                                  | \[{array with fetched documents}]                                          |
| 400         | not valid intent mapping request or account-related problems | Invalid intent mapping                                                     |
| 400         | Validation failed                                            | invalid fields provided by search string in the SEARCH type intent mapping |

### Response types

Types of the returned document fields

| Field                        | Type             | Sub Properties                      | Sub Properties Types      |
| ---------------------------- | ---------------- | ----------------------------------- | ------------------------- |
| ratingsCount                 | number           |                                     |                           |
| averageRating                | float            |                                     |                           |
| previewSnippetImageUrl       | string           |                                     |                           |
| previewImageURL              | string           |                                     |                           |
| detailedSection              | string           |                                     |                           |
| description                  | string           |                                     |                           |
| meta                         | object           | creationSource, mimeType, viewCount | string, string, number    |
| tags                         | array of objects | tag, ownerId                        | string, string            |
| lastModifiedAt               | date string      |                                     |                           |
| contentUpdatedByUserFullName | string           |                                     |                           |
| contentUpdatedByUserId       | string           |                                     |                           |
| contentUpdatedAt             | date string      |                                     |                           |
| createdAt                    | timestamp number |                                     |                           |
| title                        | string           |                                     |                           |
| ownerUsername                | string           |                                     |                           |
| informationSourceURL         | string           |                                     |                           |
| pins                         | object           | dashboard, group, folder            | boolean, boolean, boolean |
| path                         | string           |                                     |                           |
| type                         | string           |                                     |                           |
| ownerId                      | string           |                                     |                           |
| accountId                    | string           |                                     |                           |
| gemId                        | string           |                                     |                           |
| publicURL                    | string           |                                     |                           |
| gemPageURL                   | string           |                                     |                           |

### Response example

```json
[
  {
    "averageRating": 5,
    "description": "Blog",
    "detailedSection": "some text here",
    "gemId": "some-gem-id-1",
    "ratingsCount": 33
  },
  {
    "gemId": "666666-5555-4444-aaaa-bbbbbbbbbbc",
    "accountId": "some-account-id",
    "ownerId": "some-owner-id",
    "type": "Note",
    "path": ",666666-5555-4444-aaaa-bbbbbbbbbbb,",
    "pins": {},
    "ownerUsername": "some owner username",
    "title": "some-title",
    "createdAt": "2019-12-16T10:03:18",
    "contentUpdatedAt": "2019-12-16T10:03:24",
    "contentUpdatedByUserId": "some-owner-id",
    "contentUpdatedByUserFullName": "some owner username",
    "lastModifiedAt": "2019-12-16T10:03:24",
    "tags": [],
    "averageRating": 4.5,
    "description": "description that can help you",
    "detailedSection": "details about something",
    "ratingsCount": 33,
    "publicURL": "https://shelf_domain/ssp/libraries/666666-5555-4444-aaaa-bbbbbbbbbbb/gems/666666-5555-4444-aaaa-bbbbbbbbbbc/portal/redirect",
    "gemPageURL": "https://my-subdomain.shelf.io/read/some-gem-id?source=chatbot",
    "meta": {
      "creationSource": "website",
      "viewCount": 5
    }
  }
]
```
