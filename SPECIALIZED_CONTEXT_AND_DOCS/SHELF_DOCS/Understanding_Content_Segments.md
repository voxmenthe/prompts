---
description: Organize, analyze, and manage content by focusing on specific data subsets.
---

# Understanding Content Segments

Content Segments are a powerful feature within Shelf Content Intelligence designed to help you organize, analyze, and manage your content more effectively by focusing on specific subsets of your data. This page introduces what Content Segments are, why they are beneficial, and how they function.

## What are Content Segments?

A Content Segment is a **predefined set of content filters** that allows for in-depth analysis and exploration of content based on business domains, geographies, or other pertinent segments you define. Think of them as saved, dynamic views into specific slices of your content.

They are **dynamic**; Content Intelligence continuously evaluates your content against the segment's criteria. If a document is updated to match or no longer match a segment's filters, its inclusion is updated automatically.

Key aspects include:

* **Purpose-Driven Filtering:** Group content according to shared characteristics like source, collection, tags, categories, language, and importantly, **any custom metadata field defined in your Connectors**. This allows for highly specific and relevant segmentation.
* **Dynamic Application:** Filters are applied dynamically. As your content changes or new content matching the segment criteria is added, it automatically becomes part of that segment.
* **Overlapping Segments:** Content Segments can overlap. A single document might meet the criteria for multiple segments, allowing for flexible and multi-faceted analysis (as illustrated below).

## Key Characteristics and How They Work

Content Segments operate based on a set of defined rules that precisely filter your content, allowing you to sculpt specific views into your data. These rules are built using various attributes of your content. As you can see when adding a new segment (see image below), you define them based on several criteria:

<figure><img src="https://3746171570-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2F7yKmgyutbCyeUlzPhhYi%2Fuploads%2FgmyGJf93oADUDLz05ko9%2Fcontent-segment.png?alt=media&#x26;token=fa338629-b6c0-4302-acf7-d0a46bc85483" alt=""><figcaption></figcaption></figure>

You'll start by giving your segment a **Segment name**. Then, you'll specify **Collections**, pinpointing specific collections from your connected sources. It's important to note that you select particular collections within a source (like specific Libraries in Shelf KMS, or top-level folders in OneDrive/Dropbox, as shown in the "Select locations" dialog below), rather than selecting an entire connector.

You can further refine your segment using **Tags**, **Categories** (currently for Shelf KMS content), and **Content language**.

A particularly powerful option is the ability to use **Custom fields**. This allows you to leverage any metadata field you've defined in your Connectors, enabling highly granular segmentation based on your organization's unique data structure (e.g., "ProductLine," "DepartmentID").

### Building Conditions for Custom Fields

Content segmentation mentioned above occurs through the properly defined conditions, which are based on specific datatypes. In its turn, each datatype has its operators and values that help ensure precise filter configuration.

For example, you can configure condition(s) to customize your content segments(s) for better content filtering.

<figure><img src="https://3746171570-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2F7yKmgyutbCyeUlzPhhYi%2Fuploads%2Fgit-blob-56aae3bdd3174ffb6b5fc7a8918654dd238fb5d7%2FConditions-addfilter-condition.png?alt=media" alt="" width="375"><figcaption></figcaption></figure>

Before you start building conditions, it’s important to understand the types of data you can filter, the operators available for each type, and examples of how these work in practice. The tabs below show all supported data types (string (`string`), string list (`string[]`), numerical(`number`), and Boolean(`boolean`)), operators, and value formats you can use to tailor Custom field logic in Content Intelligence.

You can use these operators to:

* Target text fields, numbers, lists, or boolean flags with both simple and advanced rules,
* Combine multiple conditions for precise segmenting or analysis,
* Apply regular expressions, ranges, lists, and more—depending on your quality and coverage goals.

{% tabs %}
{% tab title="String operators" %}
|    Operator / Value    | Description                                                                                                                                 |
| :--------------------: | ------------------------------------------------------------------------------------------------------------------------------------------- |
|           is           | Matches exact text (e.g., "**Category is Electronics**")                                                                                    |
|        is any of       | Matches exactly at least one item from the list (e.g., "**Category is any of \[Electronics, Food, Cosmetics]**")                            |
|       is none of       | Same as above but vice versa                                                                                                                |
|         is not         | Excludes exact matches                                                                                                                      |
|        contains        | Contains the specified text anywhere                                                                                                        |
|    does not contain    | Excludes content with the specified text                                                                                                    |
|     contains any of    | Contains at least one item of the specified text anywhere (e.g., "**Category is any of \[Electronics, Food, Cosmetics]**")                  |
| doesn't contain any of | Excludes at least one item of the specified text anywhere                                                                                   |
|       starts with      | Matches text beginning with a specific phrase/word                                                                                          |
|        ends with       | Matches text ending with a specific phrase/word                                                                                             |
|      length equals     | Regular number                                                                                                                              |
|    length more than    | <<...>>                                                                                                                                     |
|    length less than    | <<...>>                                                                                                                                     |
|      matches regex     | Allows pattern-based filtering using regular expressions (e.g., extend or modify Unsecured credentials gap to detect some specific secrets) |
{% endtab %}

{% tab title="Number operators" %}
|     Operator / Value     | Description                                                                        |
| :----------------------: | ---------------------------------------------------------------------------------- |
|            is            | Matches an exact numeric value (e.g., **price = 100**)                             |
|          is not          | Excludes a specific number (e.g., **price != 100**)                                |
|       greater than       | Matches numbers greater than a given value (e.g., **age > 18**)                    |
| greater than or equal to | Matches numbers greater than or equal to a given value (e.g., **score >= 90**)     |
|         less than        | Matches numbers less than a given value (e.g., **temperature < 0**)                |
|   less than or equal to  | Matches numbers less than or equal to a given value (e.g., **discount <= 50**)     |
|          between         | Matches numbers falling within a specific range (**quantity between 10 (and) 20**) |
{% endtab %}

{% tab title="String list operators" %}
| Operator / Value | Description                                    |
| :--------------: | ---------------------------------------------- |
|     contains     | Contains the specified text in any item        |
| does not contain | Excludes content with the specified text       |
|    has any of    | Contains any of the listed items               |
|    has all of    | Contains all of the listed items               |
|    has none of   | Does not contain any of the listed items       |
|   count equals   | A numerical value equal to a specified one     |
|  count less than | A numerical value less than a specified one    |
|  count more than | A numerical value higher than a specified one. |
{% endtab %}

{% tab title="Boolean operators" %}
| Operator / Value | Description                                                                                                    |
| :--------------: | -------------------------------------------------------------------------------------------------------------- |
|        is        | This operator requires a selectable value that works under the logic `Yes` or `No`, that is `true` or `false`. |
{% endtab %}
{% endtabs %}

The way the filters mentioned above combine is key to their power. Content Intelligence applies an **OR** logic _within_ a single filter type (like Tags or Collections) and an **AND** logic _between_ different filter types.

For example, if you select Tags "HR" **and** "IT", along with Collections "Guides" **and** "Policies", the segment will include documents that have (the "HR" tag **OR** the "IT" tag) **AND** also belong to (the "Guides" collection **OR** the "Policies" collection).

<img src="https://3746171570-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2F7yKmgyutbCyeUlzPhhYi%2Fuploads%2FiRgr2KJD3ALKdAwIxeHe%2Ffile.excalidraw.svg?alt=media&#x26;token=c6ada44f-72b1-4c66-bf0c-52d0b0bd8fd3" alt="" class="gitbook-drawing">

Furthermore, segments can **overlap**. A single document might satisfy the criteria for multiple segments, allowing you to analyze it from various perspectives. For example, a document could be part of a "Q4 Marketing Campaigns" segment and also a "High Priority Legal Review" segment if it meets the rules for both.

<img src="https://3746171570-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2F7yKmgyutbCyeUlzPhhYi%2Fuploads%2FtODPrrXst62u6F3eY0eW%2Ffile.excalidraw.svg?alt=media&#x26;token=b0526eeb-155f-4b79-bcc6-2495fc914419" alt="" class="gitbook-drawing">

## Where are Content Segments Useful?

Content Segments offer several key benefits for managing and improving your unstructured data, especially for preparing data for GenAI:

<details>

<summary>Targeted Analysis &#x26; Reporting</summary>

Focus your content quality analysis (e.g., identifying gaps like broken links, duplicates, or contradictions) on specific, business-relevant slices of your content inventory.

Filter dashboards and reports (like Quality Assessment, Link Health Explorer) to show data only for a particular segment.

</details>

<details>

<summary>Focused Quality Improvement</summary>

Understand the unique risk profile of different content sets (e.g., "All Customer-Facing FAQs" vs. "Internal Engineering Documentation").

Prioritize content cleanup and enrichment efforts where they matter most.

</details>

<details>

<summary>Efficient Filtering &#x26; Navigation</summary>

Save and reuse complex filter combinations. Instead of manually selecting multiple filters (source, tags, custom metadata, etc.) each time, simply select a pre-configured Content Segment.

Quickly access and manage content relevant to specific areas of your organization from the Content Intelligence Home dashboard or the "Content segments" menu.

</details>

<details>

<summary>Tailored Data Governance</summary>

Admins can customize which content "gaps" (potential quality issues) are prioritized or tracked for different segments via the "Gap settings" tab within a segment's configuration. This allows for applying different quality standards to different content sets.

</details>
