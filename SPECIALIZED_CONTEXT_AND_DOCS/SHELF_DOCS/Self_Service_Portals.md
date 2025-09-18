# SSP Content API's docs

## Intro

Let's say you have the following articles in your Shelf account:

<figure><img src="https://3746171570-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2F7yKmgyutbCyeUlzPhhYi%2Fuploads%2Fgit-blob-66ee6473acd298924ee16999b8b6e05140855647%2F01-ssp-lib.png?alt=media" alt=""><figcaption></figcaption></figure>

And you want to build your Self-Service Portal experience where you want to search, display your articles and more like this:

<figure><img src="https://3746171570-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2F7yKmgyutbCyeUlzPhhYi%2Fuploads%2Fgit-blob-ee813d1b7150a8af9e7aefa72d60d8b6118f4e6c%2F02-ssp-portal.png?alt=media" alt=""><figcaption></figcaption></figure>

Let's go through the API's that help to build this page:

## Display Articles List

To display articles from your library you can use

## Search the Self-Service Portal

> Search for articles within the published Self-Service Portal. This API returns only articles and doesn't search for folders.\
> \
> The search algorithm analyzes not only tags but search keywords, gem title, description, Decision Tree step's info and actual text to present the most relevant results according to your initial search query.\
>

OpenAPI:

```json
{"openapi":"3.1.0","info":{"title":"Shelf Self-Service Portals API","version":"1.0.0"},"tags":[{"description":"APIs for finding and suggesting portal content","name":"Content Discovery APIs"}],"servers":[{"description":"US region","url":"https://api.shelf.io"},{"description":"EU region","url":"https://api.shelf-eu.com"},{"description":"CA region","url":"https://api.shelf-ca.com"}],"paths":{"/ssp/accounts/{accountId}/libraries/{libId}/search":{"get":{"operationId":"searchInSSP","summary":"Search the Self-Service Portal","description":"Search for articles within the published Self-Service Portal. This API returns only articles and doesn't search for folders.\n\nThe search algorithm analyzes not only tags but search keywords, gem title, description, Decision Tree step's info and actual text to present the most relevant results according to your initial search query.\n","parameters":[{"description":"Search Term to find articles for","schema":{"type":"string","maxLength":256,"minLength":1},"in":"query","name":"term"},{"description":"Page of search results","schema":{"type":"integer"},"in":"query","name":"page"},{"description":"Tag to filter search results","schema":{"type":"string","maxLength":256,"minLength":1},"in":"query","name":"tag"},{"description":"CategoryId to filter search results (category will be ignored if `categoryId` provided)","schema":{"type":"string","maxLength":64,"minLength":6},"in":"query","name":"categoryId"},{"description":"Language to filter search results","schema":{"type":"string","enum":["es","pt","...","en","en-GB"],"maxLength":5,"minLength":2,"title":"Language Code"},"in":"query","name":"lang","required":true},{"description":"The library id","schema":{"maxLength":64,"minLength":6},"in":"path","name":"libId","required":true},{"description":"Id of account","schema":{"maxLength":64,"minLength":6},"in":"path","name":"accountId","required":true},{"description":"Category names to filter search results","schema":{"type":"array","maxItems":20,"minItems":0,"items":{"type":"string","maxLength":512,"minLength":1}},"in":"query","name":"category"}],"responses":{"200":{"description":"Search Results","content":{"application/json":{"schema":{"type":"object","required":["items","totalCount"],"properties":{"totalCount":{"type":"number"},"items":{"type":"array","items":{"type":"object","required":["gemId","type","title","description","createdAt","ownerUsername","url"],"properties":{"type":{"type":"string","enum":["Note","Decision Tree"],"title":"SSP content Gem types"},"description":{"type":"string","description":"Description of a gem","maxLength":500,"minLength":0},"categories":{"type":"array","items":{"type":"object","required":["categoryId","categoryName"],"properties":{"categoryId":{"type":"string","description":"ID of a Category","maxLength":64,"minLength":6},"categoryName":{"type":"string"},"topParentCategoryName":{"type":"string"}}}},"contentUpdatedAt":{"type":"string","description":"Date of updating","format":"date-time"},"contentUpdatedByUserFullName":{"type":"string","description":"User's full name","maxLength":512,"minLength":0},"createdAt":{"type":"string","description":"Date of creation","format":"date-time"},"gemId":{"type":"string","description":"ID of a Gem","maxLength":64,"minLength":6,"title":"Gem ID"},"ownerUsername":{"type":"string","description":"User's full name","maxLength":512,"minLength":0},"tags":{"type":"array","description":"Array of gem tags","maxItems":50,"minItems":0,"items":{"type":"string","maxLength":40,"minLength":1}},"title":{"type":"string","description":"Title of a gem","maxLength":255,"minLength":1},"url":{"type":"string","description":"Gem page redirect URL. Adhered to the redirect endpoint settings & logic.\n  - If <b style=\"color=;color: darkgreen;\">attachUseReferrerToRedirectURL</b> is <b>true</b> AND <b style=\"color=;color: darkblue;\">skipUseReferrerInRedirectEndpoint</b> SSL setting is OFF:\n    - to the <b>url</b> will be attached URL parameter: <i>useReferer=true</i>\n  - If <b style=\"color=;color: darkgreen;\">attachUseReferrerToRedirectURL</b> is <b>false</b>:\n    - the URL parameter <i>useReferer=true</i> will not be attached\n  - If <b style=\"color=;color: darkgreen;\">attachAndForwardCurrentURLSearchParamsForRedirectEndpoint</b> SSL setting has some values <i>(Example: ['tag', 'category'])</i>:\n    - <b style=\"color=;color: darkorange;\">X-Shelf-URL-Search</b> search parameters will be filtered by values from the\n<b style=\"color=;color: darkblue;\">attachAndForwardCurrentURLSearchParamsForRedirectEndpoint</b> setting and will be attached\nas URL parameters <i>(Example: 'tag=some1&category=some2')</i>.\n      - So, <b style=\"color=;color: darkblue;\">attachAndForwardCurrentURLSearchParamsForRedirectEndpoint</b> setting defines\nwhich parameters will be attached to the Redirect URL from the <b style=\"color=;color: darkorange;\">X-Shelf-URL-Search</b> header.\n"}}}}}}}}},"400":{"description":"Bad Request","content":{"application/json":{"schema":{"oneOf":[{"allOf":[{"type":"object","required":["message"],"properties":{"message":{"type":"string"}}},{}],"title":"Missing required request parameters"},{"allOf":[{"type":"object","title":"Bad request payload (AJV)","required":["error"],"properties":{"error":{"type":"object","description":"API Error With detail","required":["status","message","detail"],"properties":{"detail":{"type":"array","description":"Detail info","items":{"type":"object","properties":{"instancePath":{"type":"string"},"keyword":{"type":"string"},"message":{"type":"string"},"params":{"type":"object"},"schemaPath":{"type":"string"}}}},"message":{"type":"string","description":"Error message"},"status":{"type":"number","description":"Status code"}}}}},{}],"title":"Wrong language parameter type"}]}}}},"404":{"description":"Gems were not found","content":{"application/json":{"schema":{"allOf":[{"type":"object","deprecated":true,"required":["error"],"properties":{"error":{"type":"object","description":"API Error","required":["status","message"],"properties":{"message":{"type":"string","description":"Error message"},"status":{"type":"number","description":"Status code"}}}}},{}]}}}}},"tags":["Content Discovery APIs"]}}}}
```

from the Self-Service Search API. It will let you get the list of articles on your account' library and filter them by language, category, tag or search term. Since endpoint returns a `totalCount` of all articles you can use this to build pagination got the search results.

#### Example code

```ts
async function searchArticles({accountId, libraryId, page, category, tag, term, lang}) {
  const filters = `?page=${page}&category=${category}&tag=${tag}&term=${term}&lang=${lang}`;
  const url = `https://api.shelf.io/ssp/accounts/${accountId}/libraries/${libraryId}/search/${filters}`;

  const resp = await fetch(url);
  return await resp.json();
}

export default async function () {
  const accountId = 'your-account-id';
  const libraryId = 'your-library-id';

  const page = 1; // Page of search results. Can be used for pagination. 1 page is 20 search results items
  const category = 'your-category'; // category to filter search results
  const tag = 'your tag'; // tag to filter search results
  const term = 'your search term'; // search Term to find articles for
  const lang = 'en'; // language to filter search results

  return await searchArticles({
    accountId,
    libraryId,
    page,
    category,
    tag,
    term,
    lang,
  });
}
```

{% embed url="https://codesandbox.io/embed/search-articles-z7emc?fontsize=14&hidenavigation=1&theme=dark&view=preview" %}

## Display Article Page

When you need to display a specific article page you can use [Wiki Gem endpoint](https://docs.shelf.io/dev-portal/api-reference/self-service-portals/content-access-apis#get-ssp-accounts-accountid-libraries-libid-gems-gemid). Take the `gemId` of article you want to display and pass it to the request:

### Example code

```ts
async function getWikiArticle({accountId, libraryId, gemId, lang}) {
  const url = `https://api.shelf.io/ssp/accounts/${accountId}/libraries/${libraryId}/gems/${gemId}?lang=${lang}`;

  const resp = await fetch(url);
  return await resp.json();
}

export default async function () {
  const accountId = 'your-account-id';
  const libraryId = 'your-library-id';
  const gemId = 'your-gem-id';
  const lang = 'en';

  return await getWikiArticle({
    accountId,
    libraryId,
    gemId,
    lang,
  });
}
```

{% embed url="https://codesandbox.io/embed/get-wiki-rqd6p?fontsize=14&hidenavigation=1&theme=dark&view=preview" %}

## Get Most Popular Articles

When you need to display a list of most viewed articles you have a [Most viewed articles endpoint](https://github.com/shelfio/shelf-dev-portal/blob/master/ssp/README.md#tag/Rendering/operation/mostViewedArticles).

### Example code

```ts
async function getMostViewedArticles({accountId, libraryId, lang, size, days}) {
  const url = `https://api.shelf.io/ssp/${accountId}/libraries/${libraryId}/languages/${lang}/articles/most-viewed/?size=${size}&days=${days}`;
  const resp = await fetch(url);
  return await resp.json();
}

export default async function () {
  const accountId = 'your-account-id';
  const libraryId = 'your-library-id';
  const lang = 'en';
  const size = 5;
  const days = 90;

  return await getMostViewedArticles({
    accountId,
    libraryId,
    lang,
    size,
    days,
  });
}
```

{% embed url="https://codesandbox.io/embed/get-most-viewed-articles-wk7np?fontsize=14&hidenavigation=1&module=/src/get-most-viewed-articles.js&theme=dark&view=preview" %}

## Get Popular Tags

If you want to display the list of popular tags that can be used as a filter of your articles you can use [Popular tags endpoint](https://github.com/shelfio/shelf-dev-portal/blob/master/ssp/README.md#tag/Rendering/operation/popularTags).

### Example code

```ts
async function getPopularTags({accountId, libraryId, lang, size}) {
  const url = `https://api.shelf.io/ssp/accounts/${accountId}/libraries/${libraryId}/popular-tags/?size=20&lang=${lang}`;
  const resp = await fetch(url);
  return await resp.json();
}

export default async function () {
  const accountId = 'your-account-id';
  const libraryId = 'your-library-id';
  const lang = 'en';
  const size = 5;

  return await getPopularTags({
    accountId,
    libraryId,
    lang,
    size,
  });
}
```

{% embed url="https://codesandbox.io/embed/get-most-popular-tags-nktvo?fontsize=14&hidenavigation=1&module=/src/get-most-popular-tags.js&theme=dark&view=preview" %}

## Get Popular Categories

Similar to the popular tags you can display [Popular categories endpoint](https://github.com/shelfio/shelf-dev-portal/blob/master/ssp/README.md#tag/Rendering/operation/mostPopularCategories).

### Example code

```ts
async function getPopularCategories({accountId, libraryId, lang, size}) {
  const url = `https://api.shelf.io/ssp/accounts/${accountId}/libraries/${libraryId}/most-popular-categories/?size=20&lang=${lang}`;
  const resp = await fetch(url);
  return await resp.json();
}

export default async function () {
  const accountId = 'your-account-id';
  const libraryId = 'your-library-id';
  const lang = 'en';
  const size = 5;

  return await getPopularCategories({
    accountId,
    libraryId,
    lang,
    size,
  });
}
```

{% embed url="https://codesandbox.io/embed/get-most-popular-categories-hbg9v?fontsize=14&hidenavigation=1&module=/src/get-most-popular-categories.js&theme=dark&view=preview" %}
