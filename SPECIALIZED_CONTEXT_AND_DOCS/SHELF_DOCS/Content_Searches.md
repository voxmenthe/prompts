# Content Searches

This technical guide will assist you in understanding how to search, filter, and view content using Shelf APIs. For the purpose of this guide, we will be using **FAQ** and **Wiki** content, and **category-based** filters for simplicity.

### Authorization

After the creation of your [**Api Token**](../api-essentials/managing-api-tokens-guide), you will need to use it as an authorization header while making API calls.

### Predefined Constants

{% code overflow="wrap" %}
```javascript
const TOKEN = '<your-token>';
const API_HOST = 'api.shelf.io'; // depending on the region 'api.shelf-eu.com' | 'api.shelf-ca.com'
const SEARCHABLE_CONTENT_TYPES_IDS = []; // will be populated in the next step
const FILTERS_FACETS = ['shelf_categoryId'];

const HEADER_LIST = {
  Accept: 'application/json',
  "Content-Type": "application/json"
  Authorization: TOKEN,
};

```
{% endcode %}

### Content Types

Since we are going to use **FAQ** and **Wiki** content we need to find out the ids of these types for subsequent content filtering/searching. **SearchFacets** will be used.

<details>

<summary>Code Snippet</summary>

{% code overflow="wrap" %}
```javascript
async function getContentTypesFacets() {

  const bodyContent = JSON.stringify({
  "fieldsFacets": [
    "contentTypeId" // we want to retrieve content types only
  ]
});

  const response = await fetch(`https://${API_HOST}/cil-search/facets`, {
    method: 'POST',
    headers: HEADER_LIST,
    body: bodyContent,
  });

  return await response.json();
}

await getContentTypesFacets();

/*
The response will include the list of all content types of your account. We will be using ids of FAQ and Wiki content types.
Below is a partial response example. More details can be found in the endpoint documentation.
*/


{
  "results": {
    "fields": {
      "contentTypeId": [
        {
          "value": "wiki-id-3P4PCDEQ1SRC42WY86P", // we are using this
          "count": 15,
          "label": "Wiki Page"
        },
        {
          "value": "01GTC3P4KFDKGZH98D01TGJBH1",
          "count": 10,
          "label": "Image"
        },
        {
          "value": "01GTC3P4NPRD4HG4AJJ829VGHA",
          "count": 17,
          "label": "Video"
        },
        {
          "value": "faq-id-01GTC3P4V9QVQ8BJV3A", // we are using this
          "count": 11,
          "label": "FAQ"
        },
      ]
    }
  }
}

```
{% endcode %}

</details>

### APIs <a href="#apis" id="apis"></a>

#### Get search facets values & counts <a href="#get-search-facets-values--counts" id="get-search-facets-values--counts"></a>

Get available facet values along with their quantities. For more information, review the [Documentation](https://docs.shelf.io/dev-portal/rest-api/cil-api/search#post-cil-search-facets)

It can be utilized on the UI to manifest available filters.

<details>

<summary>Code snippet</summary>

{% code overflow="wrap" %}
```javascript
async function getFiltersFacets() {
  const bodyContent = JSON.stringify({
    // Tells what facets will be retrieved. Here we want a list of categories.
    fieldsFacets: FILTERS_FACETS,
    fieldsFilters: {
      contentTypeId: SEARCHABLE_CONTENT_TYPES_IDS,
    },
  });

  const response = await fetch(`https://${API_HOST}/cil-search/facets`, {
    method: 'POST',
    body: bodyContent,
    headers: headersList,
  });

  const data = await response.json();

  return data.results.fields.shelf_categoryId;
}

await getFiltersFacets();
```
{% endcode %}

</details>

<details>

<summary>Response Example</summary>

{% code overflow="wrap" %}
```javascript
"results": {
    "fields": {
      "shelf_categoryId": [
        {
          // Category ID for subsequent content filtering/searching.
          "value": "setId#default#categoryId#01H9MXVEFV8VH2XJ5S2DYVJ08N",
          // 3 content items have a "Car Fuel Efficiency" category
          "count": 3,
          // Suitable to display on the UI
          "label": "Car Fuel Efficiency",
          ...
        },
        {
          "value": "setId#default#categoryId#01H9MSD43NBZ1R5D44MSNPANEF",
          "count": 2,
          "label": "Maintenance",
          ...
        },
        {
          "value": "setId#default#categoryId#01H9MXF47EPQZY5BMGD58FTCCQ",
          "count": 1,
          "label": "Insurance",
          ...
        },
        ...
        {
          // one Content item does not have a category attached
          "value": "uncategorized",
          "count": 1,
          "label": "Uncategorized",
          ...
        }
      ]
    }
  }
```
{% endcode %}

</details>

#### Search Content <a href="#search-content" id="search-content"></a>

This API allows you to filter the Gems based on certain criteria.&#x20;

For more information, review the [**Documentation**](https://app.gitbook.com/s/uYKEY4aDh92wDlKTf0kQ/content-integration-layer/search-and-retrieval)[.](https://github.com/shelfio/shelf-dev-portal/blob/master/gitbook/recipes/broken-reference/README.md)

<details>

<summary>Code snippet</summary>

{% code overflow="wrap" %}
```javascript
// `getFiltersFacets` from a preceding step fetches all categories.
const categories = await getFiltersFacets();
/*
 This will be used as a filter to search content. For simplicity, we use the first categoryId.
 Multiple categories can also be applied as needed.
*/
const categoryId = categories[0].value;

// Below line can be used when multiple categories are needed
// const categoryIds = results.map(category => category.value);

async function searchItems({query}) {
  const url = `https://${API_HOST}/cil-search/content`;

  const bodyContent = JSON.stringify({
    origin: '<your App name>',
    query, // User typed search query
    fieldsFilters: {
      shelf_categoryId: [categoryId],
      contentTypeId: SEARCHABLE_CONTENT_TYPES_IDS,
    },
  });

  const response = await fetch(url, {
    method: 'POST',
    body: bodyContent,
    headers: HEADER_LIST,
  });

  return await response.json();
}

await searchItems({
  query: 'how to change',
});
```
{% endcode %}

</details>

<details>

<summary>Response Example</summary>

```javascript
```

</details>

#### Calculating search relevance score <a href="#calculating-search-relevance-score" id="calculating-search-relevance-score"></a>

This API allows you to calculate the relevance score of search results according to search query. [**Documentation**](https://app.gitbook.com/s/uYKEY4aDh92wDlKTf0kQ/content-integration-layer/search-and-retrieval)

<details>

<summary>Code snippet</summary>

```javascript
const query = `How can I prolong`;

// `searchItems` from the "Search Content" step.
const {items} = await searchItems({query});

const externalIds = items.map(item => item.externalId);

async function calculateRelevanceScore({query, externalIds}) {
  const url = `https://${API_HOST}/cil-search/relevance`;

  const bodyContent = JSON.stringify({
    query,
    externalIds,
  });

  const response = await fetch(url, {
    method: 'POST',
    body: bodyContent,
    headers: HEADER_LIST,
  });

  return await response.json();
}

const relevanceScore = await calculateRelevanceScore({
  query,
  externalIds,
});
```

</details>

<details>

<summary>Response Example</summary>

```javascript
{
  "results": [
    {
      "externalId": "e888fafb-86c3-4259-9e1e-6b0d62b6ed85",
      "relevance": 0.82
    },
    {
      "externalId": "61cd2a86-8264-4fae-bfd4-8cb1e544a261",
      "relevance": 0.71
    },
    {
      "externalId": "479ebbfd-c43a-40ea-9fe7-8f47063488de",
      "relevance": 0.43
    },
    {
      "externalId": "c887a954-66e7-4797-bfcf-f5ecf500cdeb",
      "relevance": 0.18
    }
  ]
}
```

</details>

#### Download gem's Wiki Content <a href="#download-gems-wiki-content" id="download-gems-wiki-content"></a>

Get wiki content about a specific gem in html format.

<details>

<summary>Code snippet</summary>

```javascript
// `searchItems` from the "Search Content" step.
const {items} = await searchItems({
  query: `How can I prolong`,
});

const gemId = items[0].externalId;

async function getGemWikiContent({id}) {
  const headers = {
    Accept: 'text/html',
    Authorization: TOKEN,
  };

  const response = await fetch(`https://${API_HOST}/content/v1/gems/${id}/wiki-content`, {
    method: 'GET',
    headers,
  });

  return await response.text();
}

const wikiContent = await getGemWikiContent({
  id: gemId,
});
```

</details>

<details>

<summary>Response Example</summary>

```
<p>
    Your car's brakes play a critical role in keeping you and others safe on the road. Here
    are some tips to help you extend the life of your car's brakes:
  </p>
    <br />
  <ul>
    <li>
      <strong>Avoid aggressive driving:</strong> Frequent hard braking, rapid acceleration,
      and sudden stops can wear down your brake pads and rotors more quickly. Practice
      smooth and gradual braking to reduce unnecessary stress on your brakes.
    </li>
    <li>
      <strong>Use engine braking:</strong> Whenever possible, downshift your automatic
      transmission or use engine braking in manual transmissions to slow down rather than
      relying solely on your brakes. This technique reduces brake usage.
    </li>
    <li>
      <strong>Schedule regular brake maintenance:</strong> Have your brakes inspected and
      serviced as recommended by your car's manufacturer. Routine maintenance can catch
      issues early and prevent costly repairs.
    </li>
  </ul>

```

</details>

