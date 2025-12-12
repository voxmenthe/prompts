import{s as Lr,z as $r,o as Ur,n as oe}from"../chunks/scheduler.18a86fab.js";import{S as Jr,i as zr,g as i,s as o,r as m,A as qr,h as l,f as n,c as s,j as w,x as p,u,k as T,y as a,a as r,v as h,d as f,t as g,w as _}from"../chunks/index.98837b22.js";import{T as Qo}from"../chunks/Tip.77304350.js";import{D as x}from"../chunks/Docstring.a1ef7999.js";import{C as S}from"../chunks/CodeBlock.8d0c2e8a.js";import{E as Zn}from"../chunks/ExampleCodeBlock.8c3ee1f9.js";import{H as R,E as jr}from"../chunks/getInferenceSnippets.06c2775f.js";function Cr($){let c,y="Examples:",k,M,b;return M=new S({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyME1hcmt1cExNTW9kZWwlMkMlMjBNYXJrdXBMTUNvbmZpZyUwQSUwQSUyMyUyMEluaXRpYWxpemluZyUyMGElMjBNYXJrdXBMTSUyMG1pY3Jvc29mdCUyRm1hcmt1cGxtLWJhc2UlMjBzdHlsZSUyMGNvbmZpZ3VyYXRpb24lMEFjb25maWd1cmF0aW9uJTIwJTNEJTIwTWFya3VwTE1Db25maWcoKSUwQSUwQSUyMyUyMEluaXRpYWxpemluZyUyMGElMjBtb2RlbCUyMGZyb20lMjB0aGUlMjBtaWNyb3NvZnQlMkZtYXJrdXBsbS1iYXNlJTIwc3R5bGUlMjBjb25maWd1cmF0aW9uJTBBbW9kZWwlMjAlM0QlMjBNYXJrdXBMTU1vZGVsKGNvbmZpZ3VyYXRpb24pJTBBJTBBJTIzJTIwQWNjZXNzaW5nJTIwdGhlJTIwbW9kZWwlMjBjb25maWd1cmF0aW9uJTBBY29uZmlndXJhdGlvbiUyMCUzRCUyMG1vZGVsLmNvbmZpZw==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> MarkupLMModel, MarkupLMConfig

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a MarkupLM microsoft/markuplm-base style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = MarkupLMConfig()

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a model from the microsoft/markuplm-base style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = MarkupLMModel(configuration)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Accessing the model configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = model.config`,wrap:!1}}),{c(){c=i("p"),c.textContent=y,k=o(),m(M.$$.fragment)},l(d){c=l(d,"P",{"data-svelte-h":!0}),p(c)!=="svelte-kvfsh7"&&(c.textContent=y),k=s(d),u(M.$$.fragment,d)},m(d,v){r(d,c,v),r(d,k,v),h(M,d,v),b=!0},p:oe,i(d){b||(f(M.$$.fragment,d),b=!0)},o(d){g(M.$$.fragment,d),b=!1},d(d){d&&(n(c),n(k)),_(M,d)}}}function Fr($){let c,y="Examples:",k,M,b;return M=new S({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyME1hcmt1cExNRmVhdHVyZUV4dHJhY3RvciUwQSUwQXBhZ2VfbmFtZV8xJTIwJTNEJTIwJTIycGFnZTEuaHRtbCUyMiUwQXBhZ2VfbmFtZV8yJTIwJTNEJTIwJTIycGFnZTIuaHRtbCUyMiUwQXBhZ2VfbmFtZV8zJTIwJTNEJTIwJTIycGFnZTMuaHRtbCUyMiUwQSUwQXdpdGglMjBvcGVuKHBhZ2VfbmFtZV8xKSUyMGFzJTIwZiUzQSUwQSUyMCUyMCUyMCUyMHNpbmdsZV9odG1sX3N0cmluZyUyMCUzRCUyMGYucmVhZCgpJTBBJTBBZmVhdHVyZV9leHRyYWN0b3IlMjAlM0QlMjBNYXJrdXBMTUZlYXR1cmVFeHRyYWN0b3IoKSUwQSUwQSUyMyUyMHNpbmdsZSUyMGV4YW1wbGUlMEFlbmNvZGluZyUyMCUzRCUyMGZlYXR1cmVfZXh0cmFjdG9yKHNpbmdsZV9odG1sX3N0cmluZyklMEFwcmludChlbmNvZGluZy5rZXlzKCkpJTBBJTIzJTIwZGljdF9rZXlzKCU1Qidub2RlcyclMkMlMjAneHBhdGhzJyU1RCklMEElMEElMjMlMjBiYXRjaGVkJTIwZXhhbXBsZSUwQSUwQW11bHRpX2h0bWxfc3RyaW5ncyUyMCUzRCUyMCU1QiU1RCUwQSUwQXdpdGglMjBvcGVuKHBhZ2VfbmFtZV8yKSUyMGFzJTIwZiUzQSUwQSUyMCUyMCUyMCUyMG11bHRpX2h0bWxfc3RyaW5ncy5hcHBlbmQoZi5yZWFkKCkpJTBBd2l0aCUyMG9wZW4ocGFnZV9uYW1lXzMpJTIwYXMlMjBmJTNBJTBBJTIwJTIwJTIwJTIwbXVsdGlfaHRtbF9zdHJpbmdzLmFwcGVuZChmLnJlYWQoKSklMEElMEFlbmNvZGluZyUyMCUzRCUyMGZlYXR1cmVfZXh0cmFjdG9yKG11bHRpX2h0bWxfc3RyaW5ncyklMEFwcmludChlbmNvZGluZy5rZXlzKCkpJTBBJTIzJTIwZGljdF9rZXlzKCU1Qidub2RlcyclMkMlMjAneHBhdGhzJyU1RCk=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> MarkupLMFeatureExtractor

<span class="hljs-meta">&gt;&gt;&gt; </span>page_name_1 = <span class="hljs-string">&quot;page1.html&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>page_name_2 = <span class="hljs-string">&quot;page2.html&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>page_name_3 = <span class="hljs-string">&quot;page3.html&quot;</span>

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> <span class="hljs-built_in">open</span>(page_name_1) <span class="hljs-keyword">as</span> f:
<span class="hljs-meta">... </span>    single_html_string = f.read()

<span class="hljs-meta">&gt;&gt;&gt; </span>feature_extractor = MarkupLMFeatureExtractor()

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># single example</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>encoding = feature_extractor(single_html_string)
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">print</span>(encoding.keys())
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># dict_keys([&#x27;nodes&#x27;, &#x27;xpaths&#x27;])</span>

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># batched example</span>

<span class="hljs-meta">&gt;&gt;&gt; </span>multi_html_strings = []

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> <span class="hljs-built_in">open</span>(page_name_2) <span class="hljs-keyword">as</span> f:
<span class="hljs-meta">... </span>    multi_html_strings.append(f.read())
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> <span class="hljs-built_in">open</span>(page_name_3) <span class="hljs-keyword">as</span> f:
<span class="hljs-meta">... </span>    multi_html_strings.append(f.read())

<span class="hljs-meta">&gt;&gt;&gt; </span>encoding = feature_extractor(multi_html_strings)
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">print</span>(encoding.keys())
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># dict_keys([&#x27;nodes&#x27;, &#x27;xpaths&#x27;])</span>`,wrap:!1}}),{c(){c=i("p"),c.textContent=y,k=o(),m(M.$$.fragment)},l(d){c=l(d,"P",{"data-svelte-h":!0}),p(c)!=="svelte-kvfsh7"&&(c.textContent=y),k=s(d),u(M.$$.fragment,d)},m(d,v){r(d,c,v),r(d,k,v),h(M,d,v),b=!0},p:oe,i(d){b||(f(M.$$.fragment,d),b=!0)},o(d){g(M.$$.fragment,d),b=!1},d(d){d&&(n(c),n(k)),_(M,d)}}}function Nr($){let c,y=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){c=i("p"),c.innerHTML=y},l(k){c=l(k,"P",{"data-svelte-h":!0}),p(c)!=="svelte-fincs2"&&(c.innerHTML=y)},m(k,M){r(k,c,M)},p:oe,d(k){k&&n(c)}}}function Ir($){let c,y="Examples:",k,M,b;return M=new S({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Qcm9jZXNzb3IlMkMlMjBNYXJrdXBMTU1vZGVsJTBBJTBBcHJvY2Vzc29yJTIwJTNEJTIwQXV0b1Byb2Nlc3Nvci5mcm9tX3ByZXRyYWluZWQoJTIybWljcm9zb2Z0JTJGbWFya3VwbG0tYmFzZSUyMiklMEFtb2RlbCUyMCUzRCUyME1hcmt1cExNTW9kZWwuZnJvbV9wcmV0cmFpbmVkKCUyMm1pY3Jvc29mdCUyRm1hcmt1cGxtLWJhc2UlMjIpJTBBJTBBaHRtbF9zdHJpbmclMjAlM0QlMjAlMjIlM0NodG1sJTNFJTIwJTNDaGVhZCUzRSUyMCUzQ3RpdGxlJTNFUGFnZSUyMFRpdGxlJTNDJTJGdGl0bGUlM0UlMjAlM0MlMkZoZWFkJTNFJTIwJTNDJTJGaHRtbCUzRSUyMiUwQSUwQWVuY29kaW5nJTIwJTNEJTIwcHJvY2Vzc29yKGh0bWxfc3RyaW5nJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiklMEElMEFvdXRwdXRzJTIwJTNEJTIwbW9kZWwoKiplbmNvZGluZyklMEFsYXN0X2hpZGRlbl9zdGF0ZXMlMjAlM0QlMjBvdXRwdXRzLmxhc3RfaGlkZGVuX3N0YXRlJTBBbGlzdChsYXN0X2hpZGRlbl9zdGF0ZXMuc2hhcGUp",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoProcessor, MarkupLMModel

<span class="hljs-meta">&gt;&gt;&gt; </span>processor = AutoProcessor.from_pretrained(<span class="hljs-string">&quot;microsoft/markuplm-base&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = MarkupLMModel.from_pretrained(<span class="hljs-string">&quot;microsoft/markuplm-base&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>html_string = <span class="hljs-string">&quot;&lt;html&gt; &lt;head&gt; &lt;title&gt;Page Title&lt;/title&gt; &lt;/head&gt; &lt;/html&gt;&quot;</span>

<span class="hljs-meta">&gt;&gt;&gt; </span>encoding = processor(html_string, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**encoding)
<span class="hljs-meta">&gt;&gt;&gt; </span>last_hidden_states = outputs.last_hidden_state
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">list</span>(last_hidden_states.shape)
[<span class="hljs-number">1</span>, <span class="hljs-number">4</span>, <span class="hljs-number">768</span>]`,wrap:!1}}),{c(){c=i("p"),c.textContent=y,k=o(),m(M.$$.fragment)},l(d){c=l(d,"P",{"data-svelte-h":!0}),p(c)!=="svelte-kvfsh7"&&(c.textContent=y),k=s(d),u(M.$$.fragment,d)},m(d,v){r(d,c,v),r(d,k,v),h(M,d,v),b=!0},p:oe,i(d){b||(f(M.$$.fragment,d),b=!0)},o(d){g(M.$$.fragment,d),b=!1},d(d){d&&(n(c),n(k)),_(M,d)}}}function Zr($){let c,y=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){c=i("p"),c.innerHTML=y},l(k){c=l(k,"P",{"data-svelte-h":!0}),p(c)!=="svelte-fincs2"&&(c.innerHTML=y)},m(k,M){r(k,c,M)},p:oe,d(k){k&&n(c)}}}function Rr($){let c,y="Examples:",k,M,b;return M=new S({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Qcm9jZXNzb3IlMkMlMjBBdXRvTW9kZWxGb3JTZXF1ZW5jZUNsYXNzaWZpY2F0aW9uJTBBaW1wb3J0JTIwdG9yY2glMEElMEFwcm9jZXNzb3IlMjAlM0QlMjBBdXRvUHJvY2Vzc29yLmZyb21fcHJldHJhaW5lZCglMjJtaWNyb3NvZnQlMkZtYXJrdXBsbS1iYXNlJTIyKSUwQW1vZGVsJTIwJTNEJTIwQXV0b01vZGVsRm9yU2VxdWVuY2VDbGFzc2lmaWNhdGlvbi5mcm9tX3ByZXRyYWluZWQoJTIybWljcm9zb2Z0JTJGbWFya3VwbG0tYmFzZSUyMiUyQyUyMG51bV9sYWJlbHMlM0Q3KSUwQSUwQWh0bWxfc3RyaW5nJTIwJTNEJTIwJTIyJTNDaHRtbCUzRSUyMCUzQ2hlYWQlM0UlMjAlM0N0aXRsZSUzRVBhZ2UlMjBUaXRsZSUzQyUyRnRpdGxlJTNFJTIwJTNDJTJGaGVhZCUzRSUyMCUzQyUyRmh0bWwlM0UlMjIlMEFlbmNvZGluZyUyMCUzRCUyMHByb2Nlc3NvcihodG1sX3N0cmluZyUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpJTBBJTBBd2l0aCUyMHRvcmNoLm5vX2dyYWQoKSUzQSUwQSUyMCUyMCUyMCUyMG91dHB1dHMlMjAlM0QlMjBtb2RlbCgqKmVuY29kaW5nKSUwQSUwQWxvc3MlMjAlM0QlMjBvdXRwdXRzLmxvc3MlMEFsb2dpdHMlMjAlM0QlMjBvdXRwdXRzLmxvZ2l0cw==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoProcessor, AutoModelForSequenceClassification
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>processor = AutoProcessor.from_pretrained(<span class="hljs-string">&quot;microsoft/markuplm-base&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = AutoModelForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;microsoft/markuplm-base&quot;</span>, num_labels=<span class="hljs-number">7</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>html_string = <span class="hljs-string">&quot;&lt;html&gt; &lt;head&gt; &lt;title&gt;Page Title&lt;/title&gt; &lt;/head&gt; &lt;/html&gt;&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>encoding = processor(html_string, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    outputs = model(**encoding)

<span class="hljs-meta">&gt;&gt;&gt; </span>loss = outputs.loss
<span class="hljs-meta">&gt;&gt;&gt; </span>logits = outputs.logits`,wrap:!1}}),{c(){c=i("p"),c.textContent=y,k=o(),m(M.$$.fragment)},l(d){c=l(d,"P",{"data-svelte-h":!0}),p(c)!=="svelte-kvfsh7"&&(c.textContent=y),k=s(d),u(M.$$.fragment,d)},m(d,v){r(d,c,v),r(d,k,v),h(M,d,v),b=!0},p:oe,i(d){b||(f(M.$$.fragment,d),b=!0)},o(d){g(M.$$.fragment,d),b=!1},d(d){d&&(n(c),n(k)),_(M,d)}}}function Br($){let c,y=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){c=i("p"),c.innerHTML=y},l(k){c=l(k,"P",{"data-svelte-h":!0}),p(c)!=="svelte-fincs2"&&(c.innerHTML=y)},m(k,M){r(k,c,M)},p:oe,d(k){k&&n(c)}}}function Gr($){let c,y="Examples:",k,M,b;return M=new S({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Qcm9jZXNzb3IlMkMlMjBBdXRvTW9kZWxGb3JUb2tlbkNsYXNzaWZpY2F0aW9uJTBBaW1wb3J0JTIwdG9yY2glMEElMEFwcm9jZXNzb3IlMjAlM0QlMjBBdXRvUHJvY2Vzc29yLmZyb21fcHJldHJhaW5lZCglMjJtaWNyb3NvZnQlMkZtYXJrdXBsbS1iYXNlJTIyKSUwQXByb2Nlc3Nvci5wYXJzZV9odG1sJTIwJTNEJTIwRmFsc2UlMEFtb2RlbCUyMCUzRCUyMEF1dG9Nb2RlbEZvclRva2VuQ2xhc3NpZmljYXRpb24uZnJvbV9wcmV0cmFpbmVkKCUyMm1pY3Jvc29mdCUyRm1hcmt1cGxtLWJhc2UlMjIlMkMlMjBudW1fbGFiZWxzJTNENyklMEElMEFub2RlcyUyMCUzRCUyMCU1QiUyMmhlbGxvJTIyJTJDJTIwJTIyd29ybGQlMjIlNUQlMEF4cGF0aHMlMjAlM0QlMjAlNUIlMjIlMkZodG1sJTJGYm9keSUyRmRpdiUyRmxpJTVCMSU1RCUyRmRpdiUyRnNwYW4lMjIlMkMlMjAlMjIlMkZodG1sJTJGYm9keSUyRmRpdiUyRmxpJTVCMSU1RCUyRmRpdiUyRnNwYW4lMjIlNUQlMEFub2RlX2xhYmVscyUyMCUzRCUyMCU1QjElMkMlMjAyJTVEJTBBZW5jb2RpbmclMjAlM0QlMjBwcm9jZXNzb3Iobm9kZXMlM0Rub2RlcyUyQyUyMHhwYXRocyUzRHhwYXRocyUyQyUyMG5vZGVfbGFiZWxzJTNEbm9kZV9sYWJlbHMlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQSUwQXdpdGglMjB0b3JjaC5ub19ncmFkKCklM0ElMEElMjAlMjAlMjAlMjBvdXRwdXRzJTIwJTNEJTIwbW9kZWwoKiplbmNvZGluZyklMEElMEFsb3NzJTIwJTNEJTIwb3V0cHV0cy5sb3NzJTBBbG9naXRzJTIwJTNEJTIwb3V0cHV0cy5sb2dpdHM=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoProcessor, AutoModelForTokenClassification
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>processor = AutoProcessor.from_pretrained(<span class="hljs-string">&quot;microsoft/markuplm-base&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>processor.parse_html = <span class="hljs-literal">False</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = AutoModelForTokenClassification.from_pretrained(<span class="hljs-string">&quot;microsoft/markuplm-base&quot;</span>, num_labels=<span class="hljs-number">7</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>nodes = [<span class="hljs-string">&quot;hello&quot;</span>, <span class="hljs-string">&quot;world&quot;</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span>xpaths = [<span class="hljs-string">&quot;/html/body/div/li[1]/div/span&quot;</span>, <span class="hljs-string">&quot;/html/body/div/li[1]/div/span&quot;</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span>node_labels = [<span class="hljs-number">1</span>, <span class="hljs-number">2</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span>encoding = processor(nodes=nodes, xpaths=xpaths, node_labels=node_labels, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    outputs = model(**encoding)

<span class="hljs-meta">&gt;&gt;&gt; </span>loss = outputs.loss
<span class="hljs-meta">&gt;&gt;&gt; </span>logits = outputs.logits`,wrap:!1}}),{c(){c=i("p"),c.textContent=y,k=o(),m(M.$$.fragment)},l(d){c=l(d,"P",{"data-svelte-h":!0}),p(c)!=="svelte-kvfsh7"&&(c.textContent=y),k=s(d),u(M.$$.fragment,d)},m(d,v){r(d,c,v),r(d,k,v),h(M,d,v),b=!0},p:oe,i(d){b||(f(M.$$.fragment,d),b=!0)},o(d){g(M.$$.fragment,d),b=!1},d(d){d&&(n(c),n(k)),_(M,d)}}}function Hr($){let c,y=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){c=i("p"),c.innerHTML=y},l(k){c=l(k,"P",{"data-svelte-h":!0}),p(c)!=="svelte-fincs2"&&(c.innerHTML=y)},m(k,M){r(k,c,M)},p:oe,d(k){k&&n(c)}}}function Wr($){let c,y="Examples:",k,M,b;return M=new S({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Qcm9jZXNzb3IlMkMlMjBNYXJrdXBMTUZvclF1ZXN0aW9uQW5zd2VyaW5nJTBBaW1wb3J0JTIwdG9yY2glMEElMEFwcm9jZXNzb3IlMjAlM0QlMjBBdXRvUHJvY2Vzc29yLmZyb21fcHJldHJhaW5lZCglMjJtaWNyb3NvZnQlMkZtYXJrdXBsbS1iYXNlLWZpbmV0dW5lZC13ZWJzcmMlMjIpJTBBbW9kZWwlMjAlM0QlMjBNYXJrdXBMTUZvclF1ZXN0aW9uQW5zd2VyaW5nLmZyb21fcHJldHJhaW5lZCglMjJtaWNyb3NvZnQlMkZtYXJrdXBsbS1iYXNlLWZpbmV0dW5lZC13ZWJzcmMlMjIpJTBBJTBBaHRtbF9zdHJpbmclMjAlM0QlMjAlMjIlM0NodG1sJTNFJTIwJTNDaGVhZCUzRSUyMCUzQ3RpdGxlJTNFTXklMjBuYW1lJTIwaXMlMjBOaWVscyUzQyUyRnRpdGxlJTNFJTIwJTNDJTJGaGVhZCUzRSUyMCUzQyUyRmh0bWwlM0UlMjIlMEFxdWVzdGlvbiUyMCUzRCUyMCUyMldoYXQncyUyMGhpcyUyMG5hbWUlM0YlMjIlMEElMEFlbmNvZGluZyUyMCUzRCUyMHByb2Nlc3NvcihodG1sX3N0cmluZyUyQyUyMHF1ZXN0aW9ucyUzRHF1ZXN0aW9uJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiklMEElMEF3aXRoJTIwdG9yY2gubm9fZ3JhZCgpJTNBJTBBJTIwJTIwJTIwJTIwb3V0cHV0cyUyMCUzRCUyMG1vZGVsKCoqZW5jb2RpbmcpJTBBJTBBYW5zd2VyX3N0YXJ0X2luZGV4JTIwJTNEJTIwb3V0cHV0cy5zdGFydF9sb2dpdHMuYXJnbWF4KCklMEFhbnN3ZXJfZW5kX2luZGV4JTIwJTNEJTIwb3V0cHV0cy5lbmRfbG9naXRzLmFyZ21heCgpJTBBJTBBcHJlZGljdF9hbnN3ZXJfdG9rZW5zJTIwJTNEJTIwZW5jb2RpbmcuaW5wdXRfaWRzJTVCMCUyQyUyMGFuc3dlcl9zdGFydF9pbmRleCUyMCUzQSUyMGFuc3dlcl9lbmRfaW5kZXglMjAlMkIlMjAxJTVEJTBBcHJvY2Vzc29yLmRlY29kZShwcmVkaWN0X2Fuc3dlcl90b2tlbnMpLnN0cmlwKCk=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoProcessor, MarkupLMForQuestionAnswering
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>processor = AutoProcessor.from_pretrained(<span class="hljs-string">&quot;microsoft/markuplm-base-finetuned-websrc&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = MarkupLMForQuestionAnswering.from_pretrained(<span class="hljs-string">&quot;microsoft/markuplm-base-finetuned-websrc&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>html_string = <span class="hljs-string">&quot;&lt;html&gt; &lt;head&gt; &lt;title&gt;My name is Niels&lt;/title&gt; &lt;/head&gt; &lt;/html&gt;&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>question = <span class="hljs-string">&quot;What&#x27;s his name?&quot;</span>

<span class="hljs-meta">&gt;&gt;&gt; </span>encoding = processor(html_string, questions=question, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    outputs = model(**encoding)

<span class="hljs-meta">&gt;&gt;&gt; </span>answer_start_index = outputs.start_logits.argmax()
<span class="hljs-meta">&gt;&gt;&gt; </span>answer_end_index = outputs.end_logits.argmax()

<span class="hljs-meta">&gt;&gt;&gt; </span>predict_answer_tokens = encoding.input_ids[<span class="hljs-number">0</span>, answer_start_index : answer_end_index + <span class="hljs-number">1</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span>processor.decode(predict_answer_tokens).strip()
<span class="hljs-string">&#x27;Niels&#x27;</span>`,wrap:!1}}),{c(){c=i("p"),c.textContent=y,k=o(),m(M.$$.fragment)},l(d){c=l(d,"P",{"data-svelte-h":!0}),p(c)!=="svelte-kvfsh7"&&(c.textContent=y),k=s(d),u(M.$$.fragment,d)},m(d,v){r(d,c,v),r(d,k,v),h(M,d,v),b=!0},p:oe,i(d){b||(f(M.$$.fragment,d),b=!0)},o(d){g(M.$$.fragment,d),b=!1},d(d){d&&(n(c),n(k)),_(M,d)}}}function Er($){let c,y,k,M,b,d="<em>This model was released on 2021-10-16 and added to Hugging Face Transformers on 2022-09-30.</em>",v,$e,Hn,re,ca='<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/>',Wn,Ue,En,Je,pa=`The MarkupLM model was proposed in <a href="https://huggingface.co/papers/2110.08518" rel="nofollow">MarkupLM: Pre-training of Text and Markup Language for Visually-rich Document
Understanding</a> by Junlong Li, Yiheng Xu, Lei Cui, Furu Wei. MarkupLM is BERT, but
applied to HTML pages instead of raw text documents. The model incorporates additional embedding layers to improve
performance, similar to <a href="layoutlm">LayoutLM</a>.`,Vn,ze,ma=`The model can be used for tasks like question answering on web pages or information extraction from web pages. It obtains
state-of-the-art results on 2 important benchmarks:`,Pn,qe,ua=`<li><a href="https://x-lance.github.io/WebSRC/" rel="nofollow">WebSRC</a>, a dataset for Web-Based Structural Reading Comprehension (a bit like SQuAD but for web pages)</li> <li><a href="https://www.researchgate.net/publication/221299838_From_one_tree_to_a_forest_a_unified_solution_for_structured_web_data_extraction" rel="nofollow">SWDE</a>, a dataset
for information extraction from web pages (basically named-entity recognition on web pages)</li>`,Sn,je,ha="The abstract from the paper is the following:",Qn,Ce,fa=`<em>Multimodal pre-training with text, layout, and image has made significant progress for Visually-rich Document
Understanding (VrDU), especially the fixed-layout documents such as scanned document images. While, there are still a
large number of digital documents where the layout information is not fixed and needs to be interactively and
dynamically rendered for visualization, making existing layout-based pre-training approaches not easy to apply. In this
paper, we propose MarkupLM for document understanding tasks with markup languages as the backbone such as
HTML/XML-based documents, where text and markup information is jointly pre-trained. Experiment results show that the
pre-trained MarkupLM significantly outperforms the existing strong baseline models on several document understanding
tasks. The pre-trained model and code will be publicly available.</em>`,Xn,Fe,ga='This model was contributed by <a href="https://huggingface.co/nielsr" rel="nofollow">nielsr</a>. The original code can be found <a href="https://github.com/microsoft/unilm/tree/master/markuplm" rel="nofollow">here</a>.',An,Ne,Yn,Ie,_a=`<li>In addition to <code>input_ids</code>, <a href="/docs/transformers/v4.56.2/en/model_doc/markuplm#transformers.MarkupLMModel.forward">forward()</a> expects 2 additional inputs, namely <code>xpath_tags_seq</code> and <code>xpath_subs_seq</code>.
These are the XPATH tags and subscripts respectively for each token in the input sequence.</li> <li>One can use <a href="/docs/transformers/v4.56.2/en/model_doc/markuplm#transformers.MarkupLMProcessor">MarkupLMProcessor</a> to prepare all data for the model. Refer to the <a href="#usage-markuplmprocessor">usage guide</a> for more info.</li>`,Dn,ie,Ma,On,Ze,ka='MarkupLM architecture. Taken from the <a href="https://huggingface.co/papers/2110.08518">original paper.</a>',Kn,Re,eo,Be,ba=`The easiest way to prepare data for the model is to use <a href="/docs/transformers/v4.56.2/en/model_doc/markuplm#transformers.MarkupLMProcessor">MarkupLMProcessor</a>, which internally combines a feature extractor
(<a href="/docs/transformers/v4.56.2/en/model_doc/markuplm#transformers.MarkupLMFeatureExtractor">MarkupLMFeatureExtractor</a>) and a tokenizer (<a href="/docs/transformers/v4.56.2/en/model_doc/markuplm#transformers.MarkupLMTokenizer">MarkupLMTokenizer</a> or <a href="/docs/transformers/v4.56.2/en/model_doc/markuplm#transformers.MarkupLMTokenizerFast">MarkupLMTokenizerFast</a>). The feature extractor is
used to extract all nodes and xpaths from the HTML strings, which are then provided to the tokenizer, which turns them into the
token-level inputs of the model (<code>input_ids</code> etc.). Note that you can still use the feature extractor and tokenizer separately,
if you only want to handle one of the two tasks.`,to,Ge,no,He,ya=`In short, one can provide HTML strings (and possibly additional data) to <a href="/docs/transformers/v4.56.2/en/model_doc/markuplm#transformers.MarkupLMProcessor">MarkupLMProcessor</a>,
and it will create the inputs expected by the model. Internally, the processor first uses
<a href="/docs/transformers/v4.56.2/en/model_doc/markuplm#transformers.MarkupLMFeatureExtractor">MarkupLMFeatureExtractor</a> to get a list of nodes and corresponding xpaths. The nodes and
xpaths are then provided to <a href="/docs/transformers/v4.56.2/en/model_doc/markuplm#transformers.MarkupLMTokenizer">MarkupLMTokenizer</a> or <a href="/docs/transformers/v4.56.2/en/model_doc/markuplm#transformers.MarkupLMTokenizerFast">MarkupLMTokenizerFast</a>, which converts them
to token-level <code>input_ids</code>, <code>attention_mask</code>, <code>token_type_ids</code>, <code>xpath_subs_seq</code>, <code>xpath_tags_seq</code>.
Optionally, one can provide node labels to the processor, which are turned into token-level <code>labels</code>.`,oo,We,Ta=`<a href="/docs/transformers/v4.56.2/en/model_doc/markuplm#transformers.MarkupLMFeatureExtractor">MarkupLMFeatureExtractor</a> uses <a href="https://www.crummy.com/software/BeautifulSoup/bs4/doc/" rel="nofollow">Beautiful Soup</a>, a Python library for
pulling data out of HTML and XML files, under the hood. Note that you can still use your own parsing solution of
choice, and provide the nodes and xpaths yourself to <a href="/docs/transformers/v4.56.2/en/model_doc/markuplm#transformers.MarkupLMTokenizer">MarkupLMTokenizer</a> or <a href="/docs/transformers/v4.56.2/en/model_doc/markuplm#transformers.MarkupLMTokenizerFast">MarkupLMTokenizerFast</a>.`,so,Ee,va=`In total, there are 5 use cases that are supported by the processor. Below, we list them all. Note that each of these
use cases work for both batched and non-batched inputs (we illustrate them for non-batched inputs).`,ao,Ve,wa="<strong>Use case 1: web page classification (training, inference) + token classification (inference), parse_html = True</strong>",ro,Pe,xa="This is the simplest case, in which the processor will use the feature extractor to get all nodes and xpaths from the HTML.",io,Se,lo,Qe,La="<strong>Use case 2: web page classification (training, inference) + token classification (inference), parse_html=False</strong>",co,Xe,$a=`In case one already has obtained all nodes and xpaths, one doesn’t need the feature extractor. In that case, one should
provide the nodes and corresponding xpaths themselves to the processor, and make sure to set <code>parse_html</code> to <code>False</code>.`,po,Ae,mo,Ye,Ua="<strong>Use case 3: token classification (training), parse_html=False</strong>",uo,De,Ja=`For token classification tasks (such as <a href="https://paperswithcode.com/dataset/swde" rel="nofollow">SWDE</a>), one can also provide the
corresponding node labels in order to train a model. The processor will then convert these into token-level <code>labels</code>.
By default, it will only label the first wordpiece of a word, and label the remaining wordpieces with -100, which is the
<code>ignore_index</code> of PyTorch’s CrossEntropyLoss. In case you want all wordpieces of a word to be labeled, you can
initialize the tokenizer with <code>only_label_first_subword</code> set to <code>False</code>.`,ho,Oe,fo,Ke,za="<strong>Use case 4: web page question answering (inference), parse_html=True</strong>",go,et,qa=`For question answering tasks on web pages, you can provide a question to the processor. By default, the
processor will use the feature extractor to get all nodes and xpaths, and create [CLS] question tokens [SEP] word tokens [SEP].`,_o,tt,Mo,nt,ja="<strong>Use case 5: web page question answering (inference), parse_html=False</strong>",ko,ot,Ca=`For question answering tasks (such as WebSRC), you can provide a question to the processor. If you have extracted
all nodes and xpaths yourself, you can provide them directly to the processor. Make sure to set <code>parse_html</code> to <code>False</code>.`,bo,st,yo,at,To,rt,Fa='<li><a href="https://github.com/NielsRogge/Transformers-Tutorials/tree/master/MarkupLM" rel="nofollow">Demo notebooks</a></li> <li><a href="../tasks/sequence_classification">Text classification task guide</a></li> <li><a href="../tasks/token_classification">Token classification task guide</a></li> <li><a href="../tasks/question_answering">Question answering task guide</a></li>',vo,it,wo,B,lt,Xo,Wt,Na=`This is the configuration class to store the configuration of a <a href="/docs/transformers/v4.56.2/en/model_doc/markuplm#transformers.MarkupLMModel">MarkupLMModel</a>. It is used to instantiate a
MarkupLM model according to the specified arguments, defining the model architecture. Instantiating a configuration
with the defaults will yield a similar configuration to that of the MarkupLM
<a href="https://huggingface.co/microsoft/markuplm-base" rel="nofollow">microsoft/markuplm-base</a> architecture.`,Ao,Et,Ia=`Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertConfig">BertConfig</a> and can be used to control the model outputs. Read the
documentation from <a href="/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertConfig">BertConfig</a> for more information.`,Yo,le,xo,dt,Lo,G,ct,Do,Vt,Za=`Constructs a MarkupLM feature extractor. This can be used to get a list of nodes and corresponding xpaths from HTML
strings.`,Oo,Pt,Ra=`This feature extractor inherits from <code>PreTrainedFeatureExtractor()</code> which contains most
of the main methods. Users should refer to this superclass for more information regarding those methods.`,Ko,A,pt,es,St,Ba="Main method to prepare for the model one or several HTML strings.",ts,de,$o,mt,Uo,z,ut,ns,Qt,Ga=`Construct a MarkupLM tokenizer. Based on byte-level Byte-Pair-Encoding (BPE). <a href="/docs/transformers/v4.56.2/en/model_doc/markuplm#transformers.MarkupLMTokenizer">MarkupLMTokenizer</a> can be used to
turn HTML strings into to token-level <code>input_ids</code>, <code>attention_mask</code>, <code>token_type_ids</code>, <code>xpath_tags_seq</code> and
<code>xpath_tags_seq</code>. This tokenizer inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer">PreTrainedTokenizer</a> which contains most of the main methods.
Users should refer to this superclass for more information regarding those methods.`,os,Y,ht,ss,Xt,Ha=`Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
adding special tokens. A RoBERTa sequence has the following format:`,as,At,Wa="<li>single sequence: <code>&lt;s&gt; X &lt;/s&gt;</code></li> <li>pair of sequences: <code>&lt;s&gt; A &lt;/s&gt;&lt;/s&gt; B &lt;/s&gt;</code></li>",rs,Yt,ft,is,ce,gt,ls,Dt,Ea=`Create a mask from the two sequences passed to be used in a sequence-pair classification task. RoBERTa does not
make use of token type ids, therefore a list of zeros is returned.`,ds,Ot,_t,Jo,Mt,zo,L,kt,cs,Kt,Va="Construct a MarkupLM tokenizer. Based on byte-level Byte-Pair-Encoding (BPE).",ps,en,Pa=`<a href="/docs/transformers/v4.56.2/en/model_doc/markuplm#transformers.MarkupLMTokenizerFast">MarkupLMTokenizerFast</a> can be used to turn HTML strings into to token-level <code>input_ids</code>, <code>attention_mask</code>,
<code>token_type_ids</code>, <code>xpath_tags_seq</code> and <code>xpath_tags_seq</code>. This tokenizer inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer">PreTrainedTokenizer</a> which
contains most of the main methods.`,ms,tn,Sa="Users should refer to this superclass for more information regarding those methods.",us,J,bt,hs,nn,Qa=`add_special_tokens (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>):
Whether or not to add special tokens when encoding the sequences. This will use the underlying
<code>PretrainedTokenizerBase.build_inputs_with_special_tokens</code> function, which defines which tokens are
automatically added to the input ids. This is useful if you want to add <code>bos</code> or <code>eos</code> tokens
automatically.
padding (<code>bool</code>, <code>str</code> or <a href="/docs/transformers/v4.56.2/en/internal/file_utils#transformers.utils.PaddingStrategy">PaddingStrategy</a>, <em>optional</em>, defaults to <code>False</code>):
Activates and controls padding. Accepts the following values:`,fs,on,Xa=`<li><p><code>True</code> or <code>&#39;longest&#39;</code>: Pad to the longest sequence in the batch (or no padding if only a single
sequence is provided).</p></li> <li><p><code>&#39;max_length&#39;</code>: Pad to a maximum length specified with the argument <code>max_length</code> or to the maximum
acceptable input length for the model if that argument is not provided.</p></li> <li><p><code>False</code> or <code>&#39;do_not_pad&#39;</code> (default): No padding (i.e., can output a batch with sequences of different
lengths).
truncation (<code>bool</code>, <code>str</code> or <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.tokenization_utils_base.TruncationStrategy">TruncationStrategy</a>, <em>optional</em>, defaults to <code>False</code>):
Activates and controls truncation. Accepts the following values:</p></li> <li><p><code>True</code> or <code>&#39;longest_first&#39;</code>: Truncate to a maximum length specified with the argument <code>max_length</code> or
to the maximum acceptable input length for the model if that argument is not provided. This will
truncate token by token, removing a token from the longest sequence in the pair if a pair of
sequences (or a batch of pairs) is provided.</p></li> <li><p><code>&#39;only_first&#39;</code>: Truncate to a maximum length specified with the argument <code>max_length</code> or to the
maximum acceptable input length for the model if that argument is not provided. This will only
truncate the first sequence of a pair if a pair of sequences (or a batch of pairs) is provided.</p></li> <li><p><code>&#39;only_second&#39;</code>: Truncate to a maximum length specified with the argument <code>max_length</code> or to the
maximum acceptable input length for the model if that argument is not provided. This will only
truncate the second sequence of a pair if a pair of sequences (or a batch of pairs) is provided.</p></li> <li><p><code>False</code> or <code>&#39;do_not_truncate&#39;</code> (default): No truncation (i.e., can output batch with sequence lengths
greater than the model maximum admissible input size).
max_length (<code>int</code>, <em>optional</em>):
Controls the maximum length to use by one of the truncation/padding parameters.</p></li>`,gs,sn,Aa=`If left unset or set to <code>None</code>, this will use the predefined model maximum length if a maximum length
is required by one of the truncation/padding parameters. If the model has no specific maximum input
length (like XLNet) truncation/padding to a maximum length will be deactivated.
stride (<code>int</code>, <em>optional</em>, defaults to 0):
If set to a number along with <code>max_length</code>, the overflowing tokens returned when
<code>return_overflowing_tokens=True</code> will contain some tokens from the end of the truncated sequence
returned to provide some overlap between truncated and overflowing sequences. The value of this
argument defines the number of overlapping tokens.
is_split_into_words (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>):
Whether or not the input is already pre-tokenized (e.g., split into words). If set to <code>True</code>, the
tokenizer assumes the input is already split into words (for instance, by splitting it on whitespace)
which it will tokenize. This is useful for NER or token classification.
pad_to_multiple_of (<code>int</code>, <em>optional</em>):
If set will pad the sequence to a multiple of the provided value. Requires <code>padding</code> to be activated.
This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
<code>&gt;= 7.5</code> (Volta).
padding_side (<code>str</code>, <em>optional</em>):
The side on which the model should have padding applied. Should be selected between [‘right’, ‘left’].
Default value is picked from the class attribute of the same name.
return_tensors (<code>str</code> or <a href="/docs/transformers/v4.56.2/en/internal/file_utils#transformers.TensorType">TensorType</a>, <em>optional</em>):
If set, will return tensors instead of list of python integers. Acceptable values are:`,_s,an,Ya="<li><code>&#39;tf&#39;</code>: Return TensorFlow <code>tf.constant</code> objects.</li> <li><code>&#39;pt&#39;</code>: Return PyTorch <code>torch.Tensor</code> objects.</li> <li><code>&#39;np&#39;</code>: Return Numpy <code>np.ndarray</code> objects.</li>",Ms,rn,Da=`add_special_tokens (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>):
Whether or not to encode the sequences with the special tokens relative to their model.
padding (<code>bool</code>, <code>str</code> or <a href="/docs/transformers/v4.56.2/en/internal/file_utils#transformers.utils.PaddingStrategy">PaddingStrategy</a>, <em>optional</em>, defaults to <code>False</code>):
Activates and controls padding. Accepts the following values:`,ks,ln,Oa=`<li><p><code>True</code> or <code>&#39;longest&#39;</code>: Pad to the longest sequence in the batch (or no padding if only a single
sequence if provided).</p></li> <li><p><code>&#39;max_length&#39;</code>: Pad to a maximum length specified with the argument <code>max_length</code> or to the maximum
acceptable input length for the model if that argument is not provided.</p></li> <li><p><code>False</code> or <code>&#39;do_not_pad&#39;</code> (default): No padding (i.e., can output a batch with sequences of different
lengths).
truncation (<code>bool</code>, <code>str</code> or <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.tokenization_utils_base.TruncationStrategy">TruncationStrategy</a>, <em>optional</em>, defaults to <code>False</code>):
Activates and controls truncation. Accepts the following values:</p></li> <li><p><code>True</code> or <code>&#39;longest_first&#39;</code>: Truncate to a maximum length specified with the argument <code>max_length</code> or
to the maximum acceptable input length for the model if that argument is not provided. This will
truncate token by token, removing a token from the longest sequence in the pair if a pair of
sequences (or a batch of pairs) is provided.</p></li> <li><p><code>&#39;only_first&#39;</code>: Truncate to a maximum length specified with the argument <code>max_length</code> or to the
maximum acceptable input length for the model if that argument is not provided. This will only
truncate the first sequence of a pair if a pair of sequences (or a batch of pairs) is provided.</p></li> <li><p><code>&#39;only_second&#39;</code>: Truncate to a maximum length specified with the argument <code>max_length</code> or to the
maximum acceptable input length for the model if that argument is not provided. This will only
truncate the second sequence of a pair if a pair of sequences (or a batch of pairs) is provided.</p></li> <li><p><code>False</code> or <code>&#39;do_not_truncate&#39;</code> (default): No truncation (i.e., can output batch with sequence lengths
greater than the model maximum admissible input size).
max_length (<code>int</code>, <em>optional</em>):
Controls the maximum length to use by one of the truncation/padding parameters. If left unset or set to
<code>None</code>, this will use the predefined model maximum length if a maximum length is required by one of the
truncation/padding parameters. If the model has no specific maximum input length (like XLNet)
truncation/padding to a maximum length will be deactivated.
stride (<code>int</code>, <em>optional</em>, defaults to 0):
If set to a number along with <code>max_length</code>, the overflowing tokens returned when
<code>return_overflowing_tokens=True</code> will contain some tokens from the end of the truncated sequence
returned to provide some overlap between truncated and overflowing sequences. The value of this
argument defines the number of overlapping tokens.
pad_to_multiple_of (<code>int</code>, <em>optional</em>):
If set will pad the sequence to a multiple of the provided value. This is especially useful to enable
the use of Tensor Cores on NVIDIA hardware with compute capability <code>&gt;= 7.5</code> (Volta).
return_tensors (<code>str</code> or <a href="/docs/transformers/v4.56.2/en/internal/file_utils#transformers.TensorType">TensorType</a>, <em>optional</em>):
If set, will return tensors instead of list of python integers. Acceptable values are:</p></li> <li><p><code>&#39;tf&#39;</code>: Return TensorFlow <code>tf.constant</code> objects.</p></li> <li><p><code>&#39;pt&#39;</code>: Return PyTorch <code>torch.Tensor</code> objects.</p></li> <li><p><code>&#39;np&#39;</code>: Return Numpy <code>np.ndarray</code> objects.</p></li>`,bs,D,yt,ys,dn,Ka=`Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
adding special tokens. A RoBERTa sequence has the following format:`,Ts,cn,er="<li>single sequence: <code>&lt;s&gt; X &lt;/s&gt;</code></li> <li>pair of sequences: <code>&lt;s&gt; A &lt;/s&gt;&lt;/s&gt; B &lt;/s&gt;</code></li>",vs,pe,Tt,ws,pn,tr=`Create a mask from the two sequences passed to be used in a sequence-pair classification task. RoBERTa does not
make use of token type ids, therefore a list of zeros is returned.`,xs,me,vt,Ls,mn,nr=`Tokenize and prepare for the model a sequence or a pair of sequences. .. warning:: This method is deprecated,
<code>__call__</code> should be used instead.`,$s,ue,wt,Us,un,or=`Given the xpath expression of one particular node (like “/html/body/div/li[1]/div/span[2]”), return a list of
tag IDs and corresponding subscripts, taking into account max depth.`,qo,xt,jo,j,Lt,Js,hn,sr=`Constructs a MarkupLM processor which combines a MarkupLM feature extractor and a MarkupLM tokenizer into a single
processor.`,zs,fn,ar='<a href="/docs/transformers/v4.56.2/en/model_doc/markuplm#transformers.MarkupLMProcessor">MarkupLMProcessor</a> offers all the functionalities you need to prepare data for the model.',qs,gn,rr=`It first uses <a href="/docs/transformers/v4.56.2/en/model_doc/markuplm#transformers.MarkupLMFeatureExtractor">MarkupLMFeatureExtractor</a> to extract nodes and corresponding xpaths from one or more HTML strings.
Next, these are provided to <a href="/docs/transformers/v4.56.2/en/model_doc/markuplm#transformers.MarkupLMTokenizer">MarkupLMTokenizer</a> or <a href="/docs/transformers/v4.56.2/en/model_doc/markuplm#transformers.MarkupLMTokenizerFast">MarkupLMTokenizerFast</a>, which turns them into token-level
<code>input_ids</code>, <code>attention_mask</code>, <code>token_type_ids</code>, <code>xpath_tags_seq</code> and <code>xpath_subs_seq</code>.`,js,H,$t,Cs,_n,ir=`This method first forwards the <code>html_strings</code> argument to <a href="/docs/transformers/v4.56.2/en/model_doc/markuplm#transformers.MarkupLMFeatureExtractor.__call__"><strong>call</strong>()</a>. Next, it
passes the <code>nodes</code> and <code>xpaths</code> along with the additional arguments to <code>__call__()</code> and
returns the output.`,Fs,Mn,lr="Optionally, one can also provide a <code>text</code> argument which is passed along as first sequence.",Ns,kn,dr="Please refer to the docstring of the above two methods for more information.",Co,Ut,Fo,C,Jt,Is,bn,cr="The bare Markuplm Model outputting raw hidden-states without any specific head on top.",Zs,yn,pr=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Rs,Tn,mr=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Bs,W,zt,Gs,vn,ur='The <a href="/docs/transformers/v4.56.2/en/model_doc/markuplm#transformers.MarkupLMModel">MarkupLMModel</a> forward method, overrides the <code>__call__</code> special method.',Hs,he,Ws,fe,No,qt,Io,F,jt,Es,wn,hr=`MarkupLM Model transformer with a sequence classification/regression head on top (a linear layer on top of the
pooled output) e.g. for GLUE tasks.`,Vs,xn,fr=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Ps,Ln,gr=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Ss,E,Ct,Qs,$n,_r='The <a href="/docs/transformers/v4.56.2/en/model_doc/markuplm#transformers.MarkupLMForSequenceClassification">MarkupLMForSequenceClassification</a> forward method, overrides the <code>__call__</code> special method.',Xs,ge,As,_e,Zo,Ft,Ro,N,Nt,Ys,Un,Mr="MarkupLM Model with a <code>token_classification</code> head on top.",Ds,Jn,kr=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Os,zn,br=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Ks,V,It,ea,qn,yr='The <a href="/docs/transformers/v4.56.2/en/model_doc/markuplm#transformers.MarkupLMForTokenClassification">MarkupLMForTokenClassification</a> forward method, overrides the <code>__call__</code> special method.',ta,Me,na,ke,Bo,Zt,Go,I,Rt,oa,jn,Tr=`The Markuplm transformer with a span classification head on top for extractive question-answering tasks like
SQuAD (a linear layer on top of the hidden-states output to compute <code>span start logits</code> and <code>span end logits</code>).`,sa,Cn,vr=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,aa,Fn,wr=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,ra,P,Bt,ia,Nn,xr='The <a href="/docs/transformers/v4.56.2/en/model_doc/markuplm#transformers.MarkupLMForQuestionAnswering">MarkupLMForQuestionAnswering</a> forward method, overrides the <code>__call__</code> special method.',la,be,da,ye,Ho,Gt,Wo,Rn,Eo;return $e=new R({props:{title:"MarkupLM",local:"markuplm",headingTag:"h1"}}),Ue=new R({props:{title:"Overview",local:"overview",headingTag:"h2"}}),Ne=new R({props:{title:"Usage tips",local:"usage-tips",headingTag:"h2"}}),Re=new R({props:{title:"Usage: MarkupLMProcessor",local:"usage-markuplmprocessor",headingTag:"h2"}}),Ge=new S({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyME1hcmt1cExNRmVhdHVyZUV4dHJhY3RvciUyQyUyME1hcmt1cExNVG9rZW5pemVyRmFzdCUyQyUyME1hcmt1cExNUHJvY2Vzc29yJTBBJTBBZmVhdHVyZV9leHRyYWN0b3IlMjAlM0QlMjBNYXJrdXBMTUZlYXR1cmVFeHRyYWN0b3IoKSUwQXRva2VuaXplciUyMCUzRCUyME1hcmt1cExNVG9rZW5pemVyRmFzdC5mcm9tX3ByZXRyYWluZWQoJTIybWljcm9zb2Z0JTJGbWFya3VwbG0tYmFzZSUyMiklMEFwcm9jZXNzb3IlMjAlM0QlMjBNYXJrdXBMTVByb2Nlc3NvcihmZWF0dXJlX2V4dHJhY3RvciUyQyUyMHRva2VuaXplcik=",highlighted:`<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> MarkupLMFeatureExtractor, MarkupLMTokenizerFast, MarkupLMProcessor

feature_extractor = MarkupLMFeatureExtractor()
tokenizer = MarkupLMTokenizerFast.from_pretrained(<span class="hljs-string">&quot;microsoft/markuplm-base&quot;</span>)
processor = MarkupLMProcessor(feature_extractor, tokenizer)`,wrap:!1}}),Se=new S({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyME1hcmt1cExNUHJvY2Vzc29yJTBBJTBBcHJvY2Vzc29yJTIwJTNEJTIwTWFya3VwTE1Qcm9jZXNzb3IuZnJvbV9wcmV0cmFpbmVkKCUyMm1pY3Jvc29mdCUyRm1hcmt1cGxtLWJhc2UlMjIpJTBBJTBBaHRtbF9zdHJpbmclMjAlM0QlMjAlMjIlMjIlMjIlMEElMjAlM0MhRE9DVFlQRSUyMGh0bWwlM0UlMEElMjAlM0NodG1sJTNFJTBBJTIwJTNDaGVhZCUzRSUwQSUyMCUzQ3RpdGxlJTNFSGVsbG8lMjB3b3JsZCUzQyUyRnRpdGxlJTNFJTBBJTIwJTNDJTJGaGVhZCUzRSUwQSUyMCUzQ2JvZHklM0UlMEElMjAlM0NoMSUzRVdlbGNvbWUlM0MlMkZoMSUzRSUwQSUyMCUzQ3AlM0VIZXJlJTIwaXMlMjBteSUyMHdlYnNpdGUuJTNDJTJGcCUzRSUwQSUyMCUzQyUyRmJvZHklM0UlMEElMjAlM0MlMkZodG1sJTNFJTIyJTIyJTIyJTBBJTBBJTIzJTIwbm90ZSUyMHRoYXQlMjB5b3UlMjBjYW4lMjBhbHNvJTIwYWRkJTIwcHJvdmlkZSUyMGFsbCUyMHRva2VuaXplciUyMHBhcmFtZXRlcnMlMjBoZXJlJTIwc3VjaCUyMGFzJTIwcGFkZGluZyUyQyUyMHRydW5jYXRpb24lMEFlbmNvZGluZyUyMCUzRCUyMHByb2Nlc3NvcihodG1sX3N0cmluZyUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpJTBBcHJpbnQoZW5jb2Rpbmcua2V5cygpKQ==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> MarkupLMProcessor

<span class="hljs-meta">&gt;&gt;&gt; </span>processor = MarkupLMProcessor.from_pretrained(<span class="hljs-string">&quot;microsoft/markuplm-base&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>html_string = <span class="hljs-string">&quot;&quot;&quot;
<span class="hljs-meta">... </span> &lt;!DOCTYPE html&gt;
<span class="hljs-meta">... </span> &lt;html&gt;
<span class="hljs-meta">... </span> &lt;head&gt;
<span class="hljs-meta">... </span> &lt;title&gt;Hello world&lt;/title&gt;
<span class="hljs-meta">... </span> &lt;/head&gt;
<span class="hljs-meta">... </span> &lt;body&gt;
<span class="hljs-meta">... </span> &lt;h1&gt;Welcome&lt;/h1&gt;
<span class="hljs-meta">... </span> &lt;p&gt;Here is my website.&lt;/p&gt;
<span class="hljs-meta">... </span> &lt;/body&gt;
<span class="hljs-meta">... </span> &lt;/html&gt;&quot;&quot;&quot;</span>

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># note that you can also add provide all tokenizer parameters here such as padding, truncation</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>encoding = processor(html_string, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">print</span>(encoding.keys())
dict_keys([<span class="hljs-string">&#x27;input_ids&#x27;</span>, <span class="hljs-string">&#x27;token_type_ids&#x27;</span>, <span class="hljs-string">&#x27;attention_mask&#x27;</span>, <span class="hljs-string">&#x27;xpath_tags_seq&#x27;</span>, <span class="hljs-string">&#x27;xpath_subs_seq&#x27;</span>])`,wrap:!1}}),Ae=new S({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyME1hcmt1cExNUHJvY2Vzc29yJTBBJTBBcHJvY2Vzc29yJTIwJTNEJTIwTWFya3VwTE1Qcm9jZXNzb3IuZnJvbV9wcmV0cmFpbmVkKCUyMm1pY3Jvc29mdCUyRm1hcmt1cGxtLWJhc2UlMjIpJTBBcHJvY2Vzc29yLnBhcnNlX2h0bWwlMjAlM0QlMjBGYWxzZSUwQSUwQW5vZGVzJTIwJTNEJTIwJTVCJTIyaGVsbG8lMjIlMkMlMjAlMjJ3b3JsZCUyMiUyQyUyMCUyMmhvdyUyMiUyQyUyMCUyMmFyZSUyMiU1RCUwQXhwYXRocyUyMCUzRCUyMCU1QiUyMiUyRmh0bWwlMkZib2R5JTJGZGl2JTJGbGklNUIxJTVEJTJGZGl2JTJGc3BhbiUyMiUyQyUyMCUyMiUyRmh0bWwlMkZib2R5JTJGZGl2JTJGbGklNUIxJTVEJTJGZGl2JTJGc3BhbiUyMiUyQyUyMCUyMmh0bWwlMkZib2R5JTIyJTJDJTIwJTIyaHRtbCUyRmJvZHklMkZkaXYlMjIlNUQlMEFlbmNvZGluZyUyMCUzRCUyMHByb2Nlc3Nvcihub2RlcyUzRG5vZGVzJTJDJTIweHBhdGhzJTNEeHBhdGhzJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiklMEFwcmludChlbmNvZGluZy5rZXlzKCkp",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> MarkupLMProcessor

<span class="hljs-meta">&gt;&gt;&gt; </span>processor = MarkupLMProcessor.from_pretrained(<span class="hljs-string">&quot;microsoft/markuplm-base&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>processor.parse_html = <span class="hljs-literal">False</span>

<span class="hljs-meta">&gt;&gt;&gt; </span>nodes = [<span class="hljs-string">&quot;hello&quot;</span>, <span class="hljs-string">&quot;world&quot;</span>, <span class="hljs-string">&quot;how&quot;</span>, <span class="hljs-string">&quot;are&quot;</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span>xpaths = [<span class="hljs-string">&quot;/html/body/div/li[1]/div/span&quot;</span>, <span class="hljs-string">&quot;/html/body/div/li[1]/div/span&quot;</span>, <span class="hljs-string">&quot;html/body&quot;</span>, <span class="hljs-string">&quot;html/body/div&quot;</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span>encoding = processor(nodes=nodes, xpaths=xpaths, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">print</span>(encoding.keys())
dict_keys([<span class="hljs-string">&#x27;input_ids&#x27;</span>, <span class="hljs-string">&#x27;token_type_ids&#x27;</span>, <span class="hljs-string">&#x27;attention_mask&#x27;</span>, <span class="hljs-string">&#x27;xpath_tags_seq&#x27;</span>, <span class="hljs-string">&#x27;xpath_subs_seq&#x27;</span>])`,wrap:!1}}),Oe=new S({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyME1hcmt1cExNUHJvY2Vzc29yJTBBJTBBcHJvY2Vzc29yJTIwJTNEJTIwTWFya3VwTE1Qcm9jZXNzb3IuZnJvbV9wcmV0cmFpbmVkKCUyMm1pY3Jvc29mdCUyRm1hcmt1cGxtLWJhc2UlMjIpJTBBcHJvY2Vzc29yLnBhcnNlX2h0bWwlMjAlM0QlMjBGYWxzZSUwQSUwQW5vZGVzJTIwJTNEJTIwJTVCJTIyaGVsbG8lMjIlMkMlMjAlMjJ3b3JsZCUyMiUyQyUyMCUyMmhvdyUyMiUyQyUyMCUyMmFyZSUyMiU1RCUwQXhwYXRocyUyMCUzRCUyMCU1QiUyMiUyRmh0bWwlMkZib2R5JTJGZGl2JTJGbGklNUIxJTVEJTJGZGl2JTJGc3BhbiUyMiUyQyUyMCUyMiUyRmh0bWwlMkZib2R5JTJGZGl2JTJGbGklNUIxJTVEJTJGZGl2JTJGc3BhbiUyMiUyQyUyMCUyMmh0bWwlMkZib2R5JTIyJTJDJTIwJTIyaHRtbCUyRmJvZHklMkZkaXYlMjIlNUQlMEFub2RlX2xhYmVscyUyMCUzRCUyMCU1QjElMkMlMjAyJTJDJTIwMiUyQyUyMDElNUQlMEFlbmNvZGluZyUyMCUzRCUyMHByb2Nlc3Nvcihub2RlcyUzRG5vZGVzJTJDJTIweHBhdGhzJTNEeHBhdGhzJTJDJTIwbm9kZV9sYWJlbHMlM0Rub2RlX2xhYmVscyUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpJTBBcHJpbnQoZW5jb2Rpbmcua2V5cygpKQ==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> MarkupLMProcessor

<span class="hljs-meta">&gt;&gt;&gt; </span>processor = MarkupLMProcessor.from_pretrained(<span class="hljs-string">&quot;microsoft/markuplm-base&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>processor.parse_html = <span class="hljs-literal">False</span>

<span class="hljs-meta">&gt;&gt;&gt; </span>nodes = [<span class="hljs-string">&quot;hello&quot;</span>, <span class="hljs-string">&quot;world&quot;</span>, <span class="hljs-string">&quot;how&quot;</span>, <span class="hljs-string">&quot;are&quot;</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span>xpaths = [<span class="hljs-string">&quot;/html/body/div/li[1]/div/span&quot;</span>, <span class="hljs-string">&quot;/html/body/div/li[1]/div/span&quot;</span>, <span class="hljs-string">&quot;html/body&quot;</span>, <span class="hljs-string">&quot;html/body/div&quot;</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span>node_labels = [<span class="hljs-number">1</span>, <span class="hljs-number">2</span>, <span class="hljs-number">2</span>, <span class="hljs-number">1</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span>encoding = processor(nodes=nodes, xpaths=xpaths, node_labels=node_labels, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">print</span>(encoding.keys())
dict_keys([<span class="hljs-string">&#x27;input_ids&#x27;</span>, <span class="hljs-string">&#x27;token_type_ids&#x27;</span>, <span class="hljs-string">&#x27;attention_mask&#x27;</span>, <span class="hljs-string">&#x27;xpath_tags_seq&#x27;</span>, <span class="hljs-string">&#x27;xpath_subs_seq&#x27;</span>, <span class="hljs-string">&#x27;labels&#x27;</span>])`,wrap:!1}}),tt=new S({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyME1hcmt1cExNUHJvY2Vzc29yJTBBJTBBcHJvY2Vzc29yJTIwJTNEJTIwTWFya3VwTE1Qcm9jZXNzb3IuZnJvbV9wcmV0cmFpbmVkKCUyMm1pY3Jvc29mdCUyRm1hcmt1cGxtLWJhc2UlMjIpJTBBJTBBaHRtbF9zdHJpbmclMjAlM0QlMjAlMjIlMjIlMjIlMEElMjAlM0MhRE9DVFlQRSUyMGh0bWwlM0UlMEElMjAlM0NodG1sJTNFJTBBJTIwJTNDaGVhZCUzRSUwQSUyMCUzQ3RpdGxlJTNFSGVsbG8lMjB3b3JsZCUzQyUyRnRpdGxlJTNFJTBBJTIwJTNDJTJGaGVhZCUzRSUwQSUyMCUzQ2JvZHklM0UlMEElMjAlM0NoMSUzRVdlbGNvbWUlM0MlMkZoMSUzRSUwQSUyMCUzQ3AlM0VNeSUyMG5hbWUlMjBpcyUyME5pZWxzLiUzQyUyRnAlM0UlMEElMjAlM0MlMkZib2R5JTNFJTBBJTIwJTNDJTJGaHRtbCUzRSUyMiUyMiUyMiUwQSUwQXF1ZXN0aW9uJTIwJTNEJTIwJTIyV2hhdCdzJTIwaGlzJTIwbmFtZSUzRiUyMiUwQWVuY29kaW5nJTIwJTNEJTIwcHJvY2Vzc29yKGh0bWxfc3RyaW5nJTJDJTIwcXVlc3Rpb25zJTNEcXVlc3Rpb24lMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQXByaW50KGVuY29kaW5nLmtleXMoKSk=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> MarkupLMProcessor

<span class="hljs-meta">&gt;&gt;&gt; </span>processor = MarkupLMProcessor.from_pretrained(<span class="hljs-string">&quot;microsoft/markuplm-base&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>html_string = <span class="hljs-string">&quot;&quot;&quot;
<span class="hljs-meta">... </span> &lt;!DOCTYPE html&gt;
<span class="hljs-meta">... </span> &lt;html&gt;
<span class="hljs-meta">... </span> &lt;head&gt;
<span class="hljs-meta">... </span> &lt;title&gt;Hello world&lt;/title&gt;
<span class="hljs-meta">... </span> &lt;/head&gt;
<span class="hljs-meta">... </span> &lt;body&gt;
<span class="hljs-meta">... </span> &lt;h1&gt;Welcome&lt;/h1&gt;
<span class="hljs-meta">... </span> &lt;p&gt;My name is Niels.&lt;/p&gt;
<span class="hljs-meta">... </span> &lt;/body&gt;
<span class="hljs-meta">... </span> &lt;/html&gt;&quot;&quot;&quot;</span>

<span class="hljs-meta">&gt;&gt;&gt; </span>question = <span class="hljs-string">&quot;What&#x27;s his name?&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>encoding = processor(html_string, questions=question, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">print</span>(encoding.keys())
dict_keys([<span class="hljs-string">&#x27;input_ids&#x27;</span>, <span class="hljs-string">&#x27;token_type_ids&#x27;</span>, <span class="hljs-string">&#x27;attention_mask&#x27;</span>, <span class="hljs-string">&#x27;xpath_tags_seq&#x27;</span>, <span class="hljs-string">&#x27;xpath_subs_seq&#x27;</span>])`,wrap:!1}}),st=new S({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyME1hcmt1cExNUHJvY2Vzc29yJTBBJTBBcHJvY2Vzc29yJTIwJTNEJTIwTWFya3VwTE1Qcm9jZXNzb3IuZnJvbV9wcmV0cmFpbmVkKCUyMm1pY3Jvc29mdCUyRm1hcmt1cGxtLWJhc2UlMjIpJTBBcHJvY2Vzc29yLnBhcnNlX2h0bWwlMjAlM0QlMjBGYWxzZSUwQSUwQW5vZGVzJTIwJTNEJTIwJTVCJTIyaGVsbG8lMjIlMkMlMjAlMjJ3b3JsZCUyMiUyQyUyMCUyMmhvdyUyMiUyQyUyMCUyMmFyZSUyMiU1RCUwQXhwYXRocyUyMCUzRCUyMCU1QiUyMiUyRmh0bWwlMkZib2R5JTJGZGl2JTJGbGklNUIxJTVEJTJGZGl2JTJGc3BhbiUyMiUyQyUyMCUyMiUyRmh0bWwlMkZib2R5JTJGZGl2JTJGbGklNUIxJTVEJTJGZGl2JTJGc3BhbiUyMiUyQyUyMCUyMmh0bWwlMkZib2R5JTIyJTJDJTIwJTIyaHRtbCUyRmJvZHklMkZkaXYlMjIlNUQlMEFxdWVzdGlvbiUyMCUzRCUyMCUyMldoYXQncyUyMGhpcyUyMG5hbWUlM0YlMjIlMEFlbmNvZGluZyUyMCUzRCUyMHByb2Nlc3Nvcihub2RlcyUzRG5vZGVzJTJDJTIweHBhdGhzJTNEeHBhdGhzJTJDJTIwcXVlc3Rpb25zJTNEcXVlc3Rpb24lMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQXByaW50KGVuY29kaW5nLmtleXMoKSk=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> MarkupLMProcessor

<span class="hljs-meta">&gt;&gt;&gt; </span>processor = MarkupLMProcessor.from_pretrained(<span class="hljs-string">&quot;microsoft/markuplm-base&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>processor.parse_html = <span class="hljs-literal">False</span>

<span class="hljs-meta">&gt;&gt;&gt; </span>nodes = [<span class="hljs-string">&quot;hello&quot;</span>, <span class="hljs-string">&quot;world&quot;</span>, <span class="hljs-string">&quot;how&quot;</span>, <span class="hljs-string">&quot;are&quot;</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span>xpaths = [<span class="hljs-string">&quot;/html/body/div/li[1]/div/span&quot;</span>, <span class="hljs-string">&quot;/html/body/div/li[1]/div/span&quot;</span>, <span class="hljs-string">&quot;html/body&quot;</span>, <span class="hljs-string">&quot;html/body/div&quot;</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span>question = <span class="hljs-string">&quot;What&#x27;s his name?&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>encoding = processor(nodes=nodes, xpaths=xpaths, questions=question, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">print</span>(encoding.keys())
dict_keys([<span class="hljs-string">&#x27;input_ids&#x27;</span>, <span class="hljs-string">&#x27;token_type_ids&#x27;</span>, <span class="hljs-string">&#x27;attention_mask&#x27;</span>, <span class="hljs-string">&#x27;xpath_tags_seq&#x27;</span>, <span class="hljs-string">&#x27;xpath_subs_seq&#x27;</span>])`,wrap:!1}}),at=new R({props:{title:"Resources",local:"resources",headingTag:"h2"}}),it=new R({props:{title:"MarkupLMConfig",local:"transformers.MarkupLMConfig",headingTag:"h2"}}),lt=new x({props:{name:"class transformers.MarkupLMConfig",anchor:"transformers.MarkupLMConfig",parameters:[{name:"vocab_size",val:" = 30522"},{name:"hidden_size",val:" = 768"},{name:"num_hidden_layers",val:" = 12"},{name:"num_attention_heads",val:" = 12"},{name:"intermediate_size",val:" = 3072"},{name:"hidden_act",val:" = 'gelu'"},{name:"hidden_dropout_prob",val:" = 0.1"},{name:"attention_probs_dropout_prob",val:" = 0.1"},{name:"max_position_embeddings",val:" = 512"},{name:"type_vocab_size",val:" = 2"},{name:"initializer_range",val:" = 0.02"},{name:"layer_norm_eps",val:" = 1e-12"},{name:"pad_token_id",val:" = 0"},{name:"bos_token_id",val:" = 0"},{name:"eos_token_id",val:" = 2"},{name:"max_xpath_tag_unit_embeddings",val:" = 256"},{name:"max_xpath_subs_unit_embeddings",val:" = 1024"},{name:"tag_pad_id",val:" = 216"},{name:"subs_pad_id",val:" = 1001"},{name:"xpath_unit_hidden_size",val:" = 32"},{name:"max_depth",val:" = 50"},{name:"position_embedding_type",val:" = 'absolute'"},{name:"use_cache",val:" = True"},{name:"classifier_dropout",val:" = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.MarkupLMConfig.vocab_size",description:`<strong>vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 30522) &#x2014;
Vocabulary size of the MarkupLM model. Defines the different tokens that can be represented by the
<em>inputs_ids</em> passed to the forward method of <a href="/docs/transformers/v4.56.2/en/model_doc/markuplm#transformers.MarkupLMModel">MarkupLMModel</a>.`,name:"vocab_size"},{anchor:"transformers.MarkupLMConfig.hidden_size",description:`<strong>hidden_size</strong> (<code>int</code>, <em>optional</em>, defaults to 768) &#x2014;
Dimensionality of the encoder layers and the pooler layer.`,name:"hidden_size"},{anchor:"transformers.MarkupLMConfig.num_hidden_layers",description:`<strong>num_hidden_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 12) &#x2014;
Number of hidden layers in the Transformer encoder.`,name:"num_hidden_layers"},{anchor:"transformers.MarkupLMConfig.num_attention_heads",description:`<strong>num_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 12) &#x2014;
Number of attention heads for each attention layer in the Transformer encoder.`,name:"num_attention_heads"},{anchor:"transformers.MarkupLMConfig.intermediate_size",description:`<strong>intermediate_size</strong> (<code>int</code>, <em>optional</em>, defaults to 3072) &#x2014;
Dimensionality of the &#x201C;intermediate&#x201D; (i.e., feed-forward) layer in the Transformer encoder.`,name:"intermediate_size"},{anchor:"transformers.MarkupLMConfig.hidden_act",description:`<strong>hidden_act</strong> (<code>str</code> or <code>function</code>, <em>optional</em>, defaults to <code>&quot;gelu&quot;</code>) &#x2014;
The non-linear activation function (function or string) in the encoder and pooler. If string, <code>&quot;gelu&quot;</code>,
<code>&quot;relu&quot;</code>, <code>&quot;silu&quot;</code> and <code>&quot;gelu_new&quot;</code> are supported.`,name:"hidden_act"},{anchor:"transformers.MarkupLMConfig.hidden_dropout_prob",description:`<strong>hidden_dropout_prob</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.`,name:"hidden_dropout_prob"},{anchor:"transformers.MarkupLMConfig.attention_probs_dropout_prob",description:`<strong>attention_probs_dropout_prob</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout ratio for the attention probabilities.`,name:"attention_probs_dropout_prob"},{anchor:"transformers.MarkupLMConfig.max_position_embeddings",description:`<strong>max_position_embeddings</strong> (<code>int</code>, <em>optional</em>, defaults to 512) &#x2014;
The maximum sequence length that this model might ever be used with. Typically set this to something large
just in case (e.g., 512 or 1024 or 2048).`,name:"max_position_embeddings"},{anchor:"transformers.MarkupLMConfig.type_vocab_size",description:`<strong>type_vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 2) &#x2014;
The vocabulary size of the <code>token_type_ids</code> passed into <a href="/docs/transformers/v4.56.2/en/model_doc/markuplm#transformers.MarkupLMModel">MarkupLMModel</a>.`,name:"type_vocab_size"},{anchor:"transformers.MarkupLMConfig.initializer_range",description:`<strong>initializer_range</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.`,name:"initializer_range"},{anchor:"transformers.MarkupLMConfig.layer_norm_eps",description:`<strong>layer_norm_eps</strong> (<code>float</code>, <em>optional</em>, defaults to 1e-12) &#x2014;
The epsilon used by the layer normalization layers.`,name:"layer_norm_eps"},{anchor:"transformers.MarkupLMConfig.max_tree_id_unit_embeddings",description:`<strong>max_tree_id_unit_embeddings</strong> (<code>int</code>, <em>optional</em>, defaults to 1024) &#x2014;
The maximum value that the tree id unit embedding might ever use. Typically set this to something large
just in case (e.g., 1024).`,name:"max_tree_id_unit_embeddings"},{anchor:"transformers.MarkupLMConfig.max_xpath_tag_unit_embeddings",description:`<strong>max_xpath_tag_unit_embeddings</strong> (<code>int</code>, <em>optional</em>, defaults to 256) &#x2014;
The maximum value that the xpath tag unit embedding might ever use. Typically set this to something large
just in case (e.g., 256).`,name:"max_xpath_tag_unit_embeddings"},{anchor:"transformers.MarkupLMConfig.max_xpath_subs_unit_embeddings",description:`<strong>max_xpath_subs_unit_embeddings</strong> (<code>int</code>, <em>optional</em>, defaults to 1024) &#x2014;
The maximum value that the xpath subscript unit embedding might ever use. Typically set this to something
large just in case (e.g., 1024).`,name:"max_xpath_subs_unit_embeddings"},{anchor:"transformers.MarkupLMConfig.tag_pad_id",description:`<strong>tag_pad_id</strong> (<code>int</code>, <em>optional</em>, defaults to 216) &#x2014;
The id of the padding token in the xpath tags.`,name:"tag_pad_id"},{anchor:"transformers.MarkupLMConfig.subs_pad_id",description:`<strong>subs_pad_id</strong> (<code>int</code>, <em>optional</em>, defaults to 1001) &#x2014;
The id of the padding token in the xpath subscripts.`,name:"subs_pad_id"},{anchor:"transformers.MarkupLMConfig.xpath_tag_unit_hidden_size",description:`<strong>xpath_tag_unit_hidden_size</strong> (<code>int</code>, <em>optional</em>, defaults to 32) &#x2014;
The hidden size of each tree id unit. One complete tree index will have
(50*xpath_tag_unit_hidden_size)-dim.`,name:"xpath_tag_unit_hidden_size"},{anchor:"transformers.MarkupLMConfig.max_depth",description:`<strong>max_depth</strong> (<code>int</code>, <em>optional</em>, defaults to 50) &#x2014;
The maximum depth in xpath.`,name:"max_depth"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/markuplm/configuration_markuplm.py#L26"}}),le=new Zn({props:{anchor:"transformers.MarkupLMConfig.example",$$slots:{default:[Cr]},$$scope:{ctx:$}}}),dt=new R({props:{title:"MarkupLMFeatureExtractor",local:"transformers.MarkupLMFeatureExtractor",headingTag:"h2"}}),ct=new x({props:{name:"class transformers.MarkupLMFeatureExtractor",anchor:"transformers.MarkupLMFeatureExtractor",parameters:[{name:"**kwargs",val:""}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/markuplm/feature_extraction_markuplm.py#L33"}}),pt=new x({props:{name:"__call__",anchor:"transformers.MarkupLMFeatureExtractor.__call__",parameters:[{name:"html_strings",val:""}],parametersDescription:[{anchor:"transformers.MarkupLMFeatureExtractor.__call__.html_strings",description:`<strong>html_strings</strong> (<code>str</code>, <code>list[str]</code>) &#x2014;
The HTML string or batch of HTML strings from which to extract nodes and corresponding xpaths.`,name:"html_strings"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/markuplm/feature_extraction_markuplm.py#L99",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/image_processor#transformers.BatchFeature"
>BatchFeature</a> with the following fields:</p>
<ul>
<li><strong>nodes</strong> — Nodes.</li>
<li><strong>xpaths</strong> — Corresponding xpaths.</li>
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><a
  href="/docs/transformers/v4.56.2/en/main_classes/image_processor#transformers.BatchFeature"
>BatchFeature</a></p>
`}}),de=new Zn({props:{anchor:"transformers.MarkupLMFeatureExtractor.__call__.example",$$slots:{default:[Fr]},$$scope:{ctx:$}}}),mt=new R({props:{title:"MarkupLMTokenizer",local:"transformers.MarkupLMTokenizer",headingTag:"h2"}}),ut=new x({props:{name:"class transformers.MarkupLMTokenizer",anchor:"transformers.MarkupLMTokenizer",parameters:[{name:"vocab_file",val:""},{name:"merges_file",val:""},{name:"tags_dict",val:""},{name:"errors",val:" = 'replace'"},{name:"bos_token",val:" = '<s>'"},{name:"eos_token",val:" = '</s>'"},{name:"sep_token",val:" = '</s>'"},{name:"cls_token",val:" = '<s>'"},{name:"unk_token",val:" = '<unk>'"},{name:"pad_token",val:" = '<pad>'"},{name:"mask_token",val:" = '<mask>'"},{name:"add_prefix_space",val:" = False"},{name:"max_depth",val:" = 50"},{name:"max_width",val:" = 1000"},{name:"pad_width",val:" = 1001"},{name:"pad_token_label",val:" = -100"},{name:"only_label_first_subword",val:" = True"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.MarkupLMTokenizer.vocab_file",description:`<strong>vocab_file</strong> (<code>str</code>) &#x2014;
Path to the vocabulary file.`,name:"vocab_file"},{anchor:"transformers.MarkupLMTokenizer.merges_file",description:`<strong>merges_file</strong> (<code>str</code>) &#x2014;
Path to the merges file.`,name:"merges_file"},{anchor:"transformers.MarkupLMTokenizer.errors",description:`<strong>errors</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;replace&quot;</code>) &#x2014;
Paradigm to follow when decoding bytes to UTF-8. See
<a href="https://docs.python.org/3/library/stdtypes.html#bytes.decode" rel="nofollow">bytes.decode</a> for more information.`,name:"errors"},{anchor:"transformers.MarkupLMTokenizer.bos_token",description:`<strong>bos_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;s&gt;&quot;</code>) &#x2014;
The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.</p>
<div class="course-tip  bg-gradient-to-br dark:bg-gradient-to-r before:border-green-500 dark:before:border-green-800 from-green-50 dark:from-gray-900 to-white dark:to-gray-950 border border-green-50 text-green-700 dark:text-gray-400">
						
<p>When building a sequence using special tokens, this is not the token that is used for the beginning of
sequence. The token used is the <code>cls_token</code>.</p>

					</div>`,name:"bos_token"},{anchor:"transformers.MarkupLMTokenizer.eos_token",description:`<strong>eos_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;/s&gt;&quot;</code>) &#x2014;
The end of sequence token.</p>
<div class="course-tip  bg-gradient-to-br dark:bg-gradient-to-r before:border-green-500 dark:before:border-green-800 from-green-50 dark:from-gray-900 to-white dark:to-gray-950 border border-green-50 text-green-700 dark:text-gray-400">
						
<p>When building a sequence using special tokens, this is not the token that is used for the end of sequence.
The token used is the <code>sep_token</code>.</p>

					</div>`,name:"eos_token"},{anchor:"transformers.MarkupLMTokenizer.sep_token",description:`<strong>sep_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;/s&gt;&quot;</code>) &#x2014;
The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
sequence classification or for a text and a question for question answering. It is also used as the last
token of a sequence built with special tokens.`,name:"sep_token"},{anchor:"transformers.MarkupLMTokenizer.cls_token",description:`<strong>cls_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;s&gt;&quot;</code>) &#x2014;
The classifier token which is used when doing sequence classification (classification of the whole sequence
instead of per-token classification). It is the first token of the sequence when built with special tokens.`,name:"cls_token"},{anchor:"transformers.MarkupLMTokenizer.unk_token",description:`<strong>unk_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;unk&gt;&quot;</code>) &#x2014;
The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
token instead.`,name:"unk_token"},{anchor:"transformers.MarkupLMTokenizer.pad_token",description:`<strong>pad_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;pad&gt;&quot;</code>) &#x2014;
The token used for padding, for example when batching sequences of different lengths.`,name:"pad_token"},{anchor:"transformers.MarkupLMTokenizer.mask_token",description:`<strong>mask_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;mask&gt;&quot;</code>) &#x2014;
The token used for masking values. This is the token used when training this model with masked language
modeling. This is the token which the model will try to predict.`,name:"mask_token"},{anchor:"transformers.MarkupLMTokenizer.add_prefix_space",description:`<strong>add_prefix_space</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to add an initial space to the input. This allows to treat the leading word just as any
other word. (RoBERTa tokenizer detect beginning of words by the preceding space).`,name:"add_prefix_space"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/markuplm/tokenization_markuplm.py#L128"}}),ht=new x({props:{name:"build_inputs_with_special_tokens",anchor:"transformers.MarkupLMTokenizer.build_inputs_with_special_tokens",parameters:[{name:"token_ids_0",val:": list"},{name:"token_ids_1",val:": typing.Optional[list[int]] = None"}],parametersDescription:[{anchor:"transformers.MarkupLMTokenizer.build_inputs_with_special_tokens.token_ids_0",description:`<strong>token_ids_0</strong> (<code>list[int]</code>) &#x2014;
List of IDs to which the special tokens will be added.`,name:"token_ids_0"},{anchor:"transformers.MarkupLMTokenizer.build_inputs_with_special_tokens.token_ids_1",description:`<strong>token_ids_1</strong> (<code>list[int]</code>, <em>optional</em>) &#x2014;
Optional second list of IDs for sequence pairs.`,name:"token_ids_1"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/markuplm/tokenization_markuplm.py#L407",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>List of <a href="../glossary#input-ids">input IDs</a> with the appropriate special tokens.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>list[int]</code></p>
`}}),ft=new x({props:{name:"get_special_tokens_mask",anchor:"transformers.MarkupLMTokenizer.get_special_tokens_mask",parameters:[{name:"token_ids_0",val:": list"},{name:"token_ids_1",val:": typing.Optional[list[int]] = None"},{name:"already_has_special_tokens",val:": bool = False"}],parametersDescription:[{anchor:"transformers.MarkupLMTokenizer.get_special_tokens_mask.Retrieve",description:"<strong>Retrieve</strong> sequence ids from a token list that has no special tokens added. This method is called when adding &#x2014;",name:"Retrieve"},{anchor:"transformers.MarkupLMTokenizer.get_special_tokens_mask.special",description:`<strong>special</strong> tokens using the tokenizer <code>prepare_for_model</code> method. &#x2014;
token_ids_0 (<code>list[int]</code>):
List of IDs.
token_ids_1 (<code>list[int]</code>, <em>optional</em>):
Optional second list of IDs for sequence pairs.
already_has_special_tokens (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>):
Whether or not the token list is already formatted with special tokens for the model.`,name:"special"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/markuplm/tokenization_markuplm.py#L446",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>list[int]</code></p>
`}}),gt=new x({props:{name:"create_token_type_ids_from_sequences",anchor:"transformers.MarkupLMTokenizer.create_token_type_ids_from_sequences",parameters:[{name:"token_ids_0",val:": list"},{name:"token_ids_1",val:": typing.Optional[list[int]] = None"}],parametersDescription:[{anchor:"transformers.MarkupLMTokenizer.create_token_type_ids_from_sequences.token_ids_0",description:`<strong>token_ids_0</strong> (<code>list[int]</code>) &#x2014;
List of IDs.`,name:"token_ids_0"},{anchor:"transformers.MarkupLMTokenizer.create_token_type_ids_from_sequences.token_ids_1",description:`<strong>token_ids_1</strong> (<code>list[int]</code>, <em>optional</em>) &#x2014;
Optional second list of IDs for sequence pairs.`,name:"token_ids_1"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/markuplm/tokenization_markuplm.py#L471",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>List of zeros.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>list[int]</code></p>
`}}),_t=new x({props:{name:"save_vocabulary",anchor:"transformers.MarkupLMTokenizer.save_vocabulary",parameters:[{name:"save_directory",val:": str"},{name:"filename_prefix",val:": typing.Optional[str] = None"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/markuplm/tokenization_markuplm.py#L370"}}),Mt=new R({props:{title:"MarkupLMTokenizerFast",local:"transformers.MarkupLMTokenizerFast",headingTag:"h2"}}),kt=new x({props:{name:"class transformers.MarkupLMTokenizerFast",anchor:"transformers.MarkupLMTokenizerFast",parameters:[{name:"vocab_file",val:""},{name:"merges_file",val:""},{name:"tags_dict",val:""},{name:"tokenizer_file",val:" = None"},{name:"errors",val:" = 'replace'"},{name:"bos_token",val:" = '<s>'"},{name:"eos_token",val:" = '</s>'"},{name:"sep_token",val:" = '</s>'"},{name:"cls_token",val:" = '<s>'"},{name:"unk_token",val:" = '<unk>'"},{name:"pad_token",val:" = '<pad>'"},{name:"mask_token",val:" = '<mask>'"},{name:"add_prefix_space",val:" = False"},{name:"max_depth",val:" = 50"},{name:"max_width",val:" = 1000"},{name:"pad_width",val:" = 1001"},{name:"pad_token_label",val:" = -100"},{name:"only_label_first_subword",val:" = True"},{name:"trim_offsets",val:" = False"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.MarkupLMTokenizerFast.vocab_file",description:`<strong>vocab_file</strong> (<code>str</code>) &#x2014;
Path to the vocabulary file.`,name:"vocab_file"},{anchor:"transformers.MarkupLMTokenizerFast.merges_file",description:`<strong>merges_file</strong> (<code>str</code>) &#x2014;
Path to the merges file.`,name:"merges_file"},{anchor:"transformers.MarkupLMTokenizerFast.errors",description:`<strong>errors</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;replace&quot;</code>) &#x2014;
Paradigm to follow when decoding bytes to UTF-8. See
<a href="https://docs.python.org/3/library/stdtypes.html#bytes.decode" rel="nofollow">bytes.decode</a> for more information.`,name:"errors"},{anchor:"transformers.MarkupLMTokenizerFast.bos_token",description:`<strong>bos_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;s&gt;&quot;</code>) &#x2014;
The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.</p>
<div class="course-tip  bg-gradient-to-br dark:bg-gradient-to-r before:border-green-500 dark:before:border-green-800 from-green-50 dark:from-gray-900 to-white dark:to-gray-950 border border-green-50 text-green-700 dark:text-gray-400">
						
<p>When building a sequence using special tokens, this is not the token that is used for the beginning of
sequence. The token used is the <code>cls_token</code>.</p>

					</div>`,name:"bos_token"},{anchor:"transformers.MarkupLMTokenizerFast.eos_token",description:`<strong>eos_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;/s&gt;&quot;</code>) &#x2014;
The end of sequence token.</p>
<div class="course-tip  bg-gradient-to-br dark:bg-gradient-to-r before:border-green-500 dark:before:border-green-800 from-green-50 dark:from-gray-900 to-white dark:to-gray-950 border border-green-50 text-green-700 dark:text-gray-400">
						
<p>When building a sequence using special tokens, this is not the token that is used for the end of sequence.
The token used is the <code>sep_token</code>.</p>

					</div>`,name:"eos_token"},{anchor:"transformers.MarkupLMTokenizerFast.sep_token",description:`<strong>sep_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;/s&gt;&quot;</code>) &#x2014;
The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
sequence classification or for a text and a question for question answering. It is also used as the last
token of a sequence built with special tokens.`,name:"sep_token"},{anchor:"transformers.MarkupLMTokenizerFast.cls_token",description:`<strong>cls_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;s&gt;&quot;</code>) &#x2014;
The classifier token which is used when doing sequence classification (classification of the whole sequence
instead of per-token classification). It is the first token of the sequence when built with special tokens.`,name:"cls_token"},{anchor:"transformers.MarkupLMTokenizerFast.unk_token",description:`<strong>unk_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;unk&gt;&quot;</code>) &#x2014;
The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
token instead.`,name:"unk_token"},{anchor:"transformers.MarkupLMTokenizerFast.pad_token",description:`<strong>pad_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;pad&gt;&quot;</code>) &#x2014;
The token used for padding, for example when batching sequences of different lengths.`,name:"pad_token"},{anchor:"transformers.MarkupLMTokenizerFast.mask_token",description:`<strong>mask_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;mask&gt;&quot;</code>) &#x2014;
The token used for masking values. This is the token used when training this model with masked language
modeling. This is the token which the model will try to predict.`,name:"mask_token"},{anchor:"transformers.MarkupLMTokenizerFast.add_prefix_space",description:`<strong>add_prefix_space</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to add an initial space to the input. This allows to treat the leading word just as any
other word. (RoBERTa tokenizer detect beginning of words by the preceding space).`,name:"add_prefix_space"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/markuplm/tokenization_markuplm_fast.py#L83"}}),bt=new x({props:{name:"batch_encode_plus",anchor:"transformers.MarkupLMTokenizerFast.batch_encode_plus",parameters:[{name:"batch_text_or_text_pairs",val:": typing.Union[list[str], list[tuple[str, str]], list[list[str]]]"},{name:"is_pair",val:": typing.Optional[bool] = None"},{name:"xpaths",val:": typing.Optional[list[list[list[int]]]] = None"},{name:"node_labels",val:": typing.Union[list[int], list[list[int]], NoneType] = None"},{name:"add_special_tokens",val:": bool = True"},{name:"padding",val:": typing.Union[bool, str, transformers.utils.generic.PaddingStrategy] = False"},{name:"truncation",val:": typing.Union[bool, str, transformers.tokenization_utils_base.TruncationStrategy] = None"},{name:"max_length",val:": typing.Optional[int] = None"},{name:"stride",val:": int = 0"},{name:"pad_to_multiple_of",val:": typing.Optional[int] = None"},{name:"padding_side",val:": typing.Optional[str] = None"},{name:"return_tensors",val:": typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None"},{name:"return_token_type_ids",val:": typing.Optional[bool] = None"},{name:"return_attention_mask",val:": typing.Optional[bool] = None"},{name:"return_overflowing_tokens",val:": bool = False"},{name:"return_special_tokens_mask",val:": bool = False"},{name:"return_offsets_mapping",val:": bool = False"},{name:"return_length",val:": bool = False"},{name:"verbose",val:": bool = True"},{name:"**kwargs",val:""}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/markuplm/tokenization_markuplm_fast.py#L416"}}),yt=new x({props:{name:"build_inputs_with_special_tokens",anchor:"transformers.MarkupLMTokenizerFast.build_inputs_with_special_tokens",parameters:[{name:"token_ids_0",val:": list"},{name:"token_ids_1",val:": typing.Optional[list[int]] = None"}],parametersDescription:[{anchor:"transformers.MarkupLMTokenizerFast.build_inputs_with_special_tokens.token_ids_0",description:`<strong>token_ids_0</strong> (<code>list[int]</code>) &#x2014;
List of IDs to which the special tokens will be added.`,name:"token_ids_0"},{anchor:"transformers.MarkupLMTokenizerFast.build_inputs_with_special_tokens.token_ids_1",description:`<strong>token_ids_1</strong> (<code>list[int]</code>, <em>optional</em>) &#x2014;
Optional second list of IDs for sequence pairs.`,name:"token_ids_1"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/markuplm/tokenization_markuplm_fast.py#L879",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>List of <a href="../glossary#input-ids">input IDs</a> with the appropriate special tokens.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>list[int]</code></p>
`}}),Tt=new x({props:{name:"create_token_type_ids_from_sequences",anchor:"transformers.MarkupLMTokenizerFast.create_token_type_ids_from_sequences",parameters:[{name:"token_ids_0",val:": list"},{name:"token_ids_1",val:": typing.Optional[list[int]] = None"}],parametersDescription:[{anchor:"transformers.MarkupLMTokenizerFast.create_token_type_ids_from_sequences.token_ids_0",description:`<strong>token_ids_0</strong> (<code>list[int]</code>) &#x2014;
List of IDs.`,name:"token_ids_0"},{anchor:"transformers.MarkupLMTokenizerFast.create_token_type_ids_from_sequences.token_ids_1",description:`<strong>token_ids_1</strong> (<code>list[int]</code>, <em>optional</em>) &#x2014;
Optional second list of IDs for sequence pairs.`,name:"token_ids_1"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/markuplm/tokenization_markuplm_fast.py#L902",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>List of zeros.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>list[int]</code></p>
`}}),vt=new x({props:{name:"encode_plus",anchor:"transformers.MarkupLMTokenizerFast.encode_plus",parameters:[{name:"text",val:": typing.Union[str, list[str]]"},{name:"text_pair",val:": typing.Optional[list[str]] = None"},{name:"xpaths",val:": typing.Optional[list[list[int]]] = None"},{name:"node_labels",val:": typing.Optional[list[int]] = None"},{name:"add_special_tokens",val:": bool = True"},{name:"padding",val:": typing.Union[bool, str, transformers.utils.generic.PaddingStrategy] = False"},{name:"truncation",val:": typing.Union[bool, str, transformers.tokenization_utils_base.TruncationStrategy] = None"},{name:"max_length",val:": typing.Optional[int] = None"},{name:"stride",val:": int = 0"},{name:"pad_to_multiple_of",val:": typing.Optional[int] = None"},{name:"padding_side",val:": typing.Optional[str] = None"},{name:"return_tensors",val:": typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None"},{name:"return_token_type_ids",val:": typing.Optional[bool] = None"},{name:"return_attention_mask",val:": typing.Optional[bool] = None"},{name:"return_overflowing_tokens",val:": bool = False"},{name:"return_special_tokens_mask",val:": bool = False"},{name:"return_offsets_mapping",val:": bool = False"},{name:"return_length",val:": bool = False"},{name:"verbose",val:": bool = True"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.MarkupLMTokenizerFast.encode_plus.text",description:`<strong>text</strong> (<code>str</code>, <code>list[str]</code>, <code>list[list[str]]</code>) &#x2014;
The first sequence to be encoded. This can be a string, a list of strings or a list of list of strings.`,name:"text"},{anchor:"transformers.MarkupLMTokenizerFast.encode_plus.text_pair",description:`<strong>text_pair</strong> (<code>list[str]</code> or <code>list[int]</code>, <em>optional</em>) &#x2014;
Optional second sequence to be encoded. This can be a list of strings (words of a single example) or a
list of list of strings (words of a batch of examples).`,name:"text_pair"},{anchor:"transformers.MarkupLMTokenizerFast.encode_plus.add_special_tokens",description:`<strong>add_special_tokens</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to add special tokens when encoding the sequences. This will use the underlying
<code>PretrainedTokenizerBase.build_inputs_with_special_tokens</code> function, which defines which tokens are
automatically added to the input ids. This is useful if you want to add <code>bos</code> or <code>eos</code> tokens
automatically.`,name:"add_special_tokens"},{anchor:"transformers.MarkupLMTokenizerFast.encode_plus.padding",description:`<strong>padding</strong> (<code>bool</code>, <code>str</code> or <a href="/docs/transformers/v4.56.2/en/internal/file_utils#transformers.utils.PaddingStrategy">PaddingStrategy</a>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Activates and controls padding. Accepts the following values:</p>
<ul>
<li><code>True</code> or <code>&apos;longest&apos;</code>: Pad to the longest sequence in the batch (or no padding if only a single
sequence is provided).</li>
<li><code>&apos;max_length&apos;</code>: Pad to a maximum length specified with the argument <code>max_length</code> or to the maximum
acceptable input length for the model if that argument is not provided.</li>
<li><code>False</code> or <code>&apos;do_not_pad&apos;</code> (default): No padding (i.e., can output a batch with sequences of different
lengths).</li>
</ul>`,name:"padding"},{anchor:"transformers.MarkupLMTokenizerFast.encode_plus.truncation",description:`<strong>truncation</strong> (<code>bool</code>, <code>str</code> or <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.tokenization_utils_base.TruncationStrategy">TruncationStrategy</a>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Activates and controls truncation. Accepts the following values:</p>
<ul>
<li><code>True</code> or <code>&apos;longest_first&apos;</code>: Truncate to a maximum length specified with the argument <code>max_length</code> or
to the maximum acceptable input length for the model if that argument is not provided. This will
truncate token by token, removing a token from the longest sequence in the pair if a pair of
sequences (or a batch of pairs) is provided.</li>
<li><code>&apos;only_first&apos;</code>: Truncate to a maximum length specified with the argument <code>max_length</code> or to the
maximum acceptable input length for the model if that argument is not provided. This will only
truncate the first sequence of a pair if a pair of sequences (or a batch of pairs) is provided.</li>
<li><code>&apos;only_second&apos;</code>: Truncate to a maximum length specified with the argument <code>max_length</code> or to the
maximum acceptable input length for the model if that argument is not provided. This will only
truncate the second sequence of a pair if a pair of sequences (or a batch of pairs) is provided.</li>
<li><code>False</code> or <code>&apos;do_not_truncate&apos;</code> (default): No truncation (i.e., can output batch with sequence lengths
greater than the model maximum admissible input size).</li>
</ul>`,name:"truncation"},{anchor:"transformers.MarkupLMTokenizerFast.encode_plus.max_length",description:`<strong>max_length</strong> (<code>int</code>, <em>optional</em>) &#x2014;
Controls the maximum length to use by one of the truncation/padding parameters.</p>
<p>If left unset or set to <code>None</code>, this will use the predefined model maximum length if a maximum length
is required by one of the truncation/padding parameters. If the model has no specific maximum input
length (like XLNet) truncation/padding to a maximum length will be deactivated.`,name:"max_length"},{anchor:"transformers.MarkupLMTokenizerFast.encode_plus.stride",description:`<strong>stride</strong> (<code>int</code>, <em>optional</em>, defaults to 0) &#x2014;
If set to a number along with <code>max_length</code>, the overflowing tokens returned when
<code>return_overflowing_tokens=True</code> will contain some tokens from the end of the truncated sequence
returned to provide some overlap between truncated and overflowing sequences. The value of this
argument defines the number of overlapping tokens.`,name:"stride"},{anchor:"transformers.MarkupLMTokenizerFast.encode_plus.is_split_into_words",description:`<strong>is_split_into_words</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not the input is already pre-tokenized (e.g., split into words). If set to <code>True</code>, the
tokenizer assumes the input is already split into words (for instance, by splitting it on whitespace)
which it will tokenize. This is useful for NER or token classification.`,name:"is_split_into_words"},{anchor:"transformers.MarkupLMTokenizerFast.encode_plus.pad_to_multiple_of",description:`<strong>pad_to_multiple_of</strong> (<code>int</code>, <em>optional</em>) &#x2014;
If set will pad the sequence to a multiple of the provided value. Requires <code>padding</code> to be activated.
This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
<code>&gt;= 7.5</code> (Volta).`,name:"pad_to_multiple_of"},{anchor:"transformers.MarkupLMTokenizerFast.encode_plus.padding_side",description:`<strong>padding_side</strong> (<code>str</code>, <em>optional</em>) &#x2014;
The side on which the model should have padding applied. Should be selected between [&#x2018;right&#x2019;, &#x2018;left&#x2019;].
Default value is picked from the class attribute of the same name.`,name:"padding_side"},{anchor:"transformers.MarkupLMTokenizerFast.encode_plus.return_tensors",description:`<strong>return_tensors</strong> (<code>str</code> or <a href="/docs/transformers/v4.56.2/en/internal/file_utils#transformers.TensorType">TensorType</a>, <em>optional</em>) &#x2014;
If set, will return tensors instead of list of python integers. Acceptable values are:</p>
<ul>
<li><code>&apos;tf&apos;</code>: Return TensorFlow <code>tf.constant</code> objects.</li>
<li><code>&apos;pt&apos;</code>: Return PyTorch <code>torch.Tensor</code> objects.</li>
<li><code>&apos;np&apos;</code>: Return Numpy <code>np.ndarray</code> objects.</li>
</ul>`,name:"return_tensors"},{anchor:"transformers.MarkupLMTokenizerFast.encode_plus.add_special_tokens",description:`<strong>add_special_tokens</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to encode the sequences with the special tokens relative to their model.`,name:"add_special_tokens"},{anchor:"transformers.MarkupLMTokenizerFast.encode_plus.padding",description:`<strong>padding</strong> (<code>bool</code>, <code>str</code> or <a href="/docs/transformers/v4.56.2/en/internal/file_utils#transformers.utils.PaddingStrategy">PaddingStrategy</a>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Activates and controls padding. Accepts the following values:</p>
<ul>
<li><code>True</code> or <code>&apos;longest&apos;</code>: Pad to the longest sequence in the batch (or no padding if only a single
sequence if provided).</li>
<li><code>&apos;max_length&apos;</code>: Pad to a maximum length specified with the argument <code>max_length</code> or to the maximum
acceptable input length for the model if that argument is not provided.</li>
<li><code>False</code> or <code>&apos;do_not_pad&apos;</code> (default): No padding (i.e., can output a batch with sequences of different
lengths).</li>
</ul>`,name:"padding"},{anchor:"transformers.MarkupLMTokenizerFast.encode_plus.truncation",description:`<strong>truncation</strong> (<code>bool</code>, <code>str</code> or <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.tokenization_utils_base.TruncationStrategy">TruncationStrategy</a>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Activates and controls truncation. Accepts the following values:</p>
<ul>
<li><code>True</code> or <code>&apos;longest_first&apos;</code>: Truncate to a maximum length specified with the argument <code>max_length</code> or
to the maximum acceptable input length for the model if that argument is not provided. This will
truncate token by token, removing a token from the longest sequence in the pair if a pair of
sequences (or a batch of pairs) is provided.</li>
<li><code>&apos;only_first&apos;</code>: Truncate to a maximum length specified with the argument <code>max_length</code> or to the
maximum acceptable input length for the model if that argument is not provided. This will only
truncate the first sequence of a pair if a pair of sequences (or a batch of pairs) is provided.</li>
<li><code>&apos;only_second&apos;</code>: Truncate to a maximum length specified with the argument <code>max_length</code> or to the
maximum acceptable input length for the model if that argument is not provided. This will only
truncate the second sequence of a pair if a pair of sequences (or a batch of pairs) is provided.</li>
<li><code>False</code> or <code>&apos;do_not_truncate&apos;</code> (default): No truncation (i.e., can output batch with sequence lengths
greater than the model maximum admissible input size).</li>
</ul>`,name:"truncation"},{anchor:"transformers.MarkupLMTokenizerFast.encode_plus.max_length",description:`<strong>max_length</strong> (<code>int</code>, <em>optional</em>) &#x2014;
Controls the maximum length to use by one of the truncation/padding parameters. If left unset or set to
<code>None</code>, this will use the predefined model maximum length if a maximum length is required by one of the
truncation/padding parameters. If the model has no specific maximum input length (like XLNet)
truncation/padding to a maximum length will be deactivated.`,name:"max_length"},{anchor:"transformers.MarkupLMTokenizerFast.encode_plus.stride",description:`<strong>stride</strong> (<code>int</code>, <em>optional</em>, defaults to 0) &#x2014;
If set to a number along with <code>max_length</code>, the overflowing tokens returned when
<code>return_overflowing_tokens=True</code> will contain some tokens from the end of the truncated sequence
returned to provide some overlap between truncated and overflowing sequences. The value of this
argument defines the number of overlapping tokens.`,name:"stride"},{anchor:"transformers.MarkupLMTokenizerFast.encode_plus.pad_to_multiple_of",description:`<strong>pad_to_multiple_of</strong> (<code>int</code>, <em>optional</em>) &#x2014;
If set will pad the sequence to a multiple of the provided value. This is especially useful to enable
the use of Tensor Cores on NVIDIA hardware with compute capability <code>&gt;= 7.5</code> (Volta).`,name:"pad_to_multiple_of"},{anchor:"transformers.MarkupLMTokenizerFast.encode_plus.return_tensors",description:`<strong>return_tensors</strong> (<code>str</code> or <a href="/docs/transformers/v4.56.2/en/internal/file_utils#transformers.TensorType">TensorType</a>, <em>optional</em>) &#x2014;
If set, will return tensors instead of list of python integers. Acceptable values are:</p>
<ul>
<li><code>&apos;tf&apos;</code>: Return TensorFlow <code>tf.constant</code> objects.</li>
<li><code>&apos;pt&apos;</code>: Return PyTorch <code>torch.Tensor</code> objects.</li>
<li><code>&apos;np&apos;</code>: Return Numpy <code>np.ndarray</code> objects.</li>
</ul>`,name:"return_tensors"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/markuplm/tokenization_markuplm_fast.py#L485"}}),wt=new x({props:{name:"get_xpath_seq",anchor:"transformers.MarkupLMTokenizerFast.get_xpath_seq",parameters:[{name:"xpath",val:""}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/markuplm/tokenization_markuplm_fast.py#L243"}}),xt=new R({props:{title:"MarkupLMProcessor",local:"transformers.MarkupLMProcessor",headingTag:"h2"}}),Lt=new x({props:{name:"class transformers.MarkupLMProcessor",anchor:"transformers.MarkupLMProcessor",parameters:[{name:"*args",val:""},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.MarkupLMProcessor.feature_extractor",description:`<strong>feature_extractor</strong> (<code>MarkupLMFeatureExtractor</code>) &#x2014;
An instance of <a href="/docs/transformers/v4.56.2/en/model_doc/markuplm#transformers.MarkupLMFeatureExtractor">MarkupLMFeatureExtractor</a>. The feature extractor is a required input.`,name:"feature_extractor"},{anchor:"transformers.MarkupLMProcessor.tokenizer",description:`<strong>tokenizer</strong> (<code>MarkupLMTokenizer</code> or <code>MarkupLMTokenizerFast</code>) &#x2014;
An instance of <a href="/docs/transformers/v4.56.2/en/model_doc/markuplm#transformers.MarkupLMTokenizer">MarkupLMTokenizer</a> or <a href="/docs/transformers/v4.56.2/en/model_doc/markuplm#transformers.MarkupLMTokenizerFast">MarkupLMTokenizerFast</a>. The tokenizer is a required input.`,name:"tokenizer"},{anchor:"transformers.MarkupLMProcessor.parse_html",description:`<strong>parse_html</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to use <code>MarkupLMFeatureExtractor</code> to parse HTML strings into nodes and corresponding xpaths.`,name:"parse_html"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/markuplm/processing_markuplm.py#L26"}}),$t=new x({props:{name:"__call__",anchor:"transformers.MarkupLMProcessor.__call__",parameters:[{name:"html_strings",val:" = None"},{name:"nodes",val:" = None"},{name:"xpaths",val:" = None"},{name:"node_labels",val:" = None"},{name:"questions",val:" = None"},{name:"add_special_tokens",val:": bool = True"},{name:"padding",val:": typing.Union[bool, str, transformers.utils.generic.PaddingStrategy] = False"},{name:"truncation",val:": typing.Union[bool, str, transformers.tokenization_utils_base.TruncationStrategy] = None"},{name:"max_length",val:": typing.Optional[int] = None"},{name:"stride",val:": int = 0"},{name:"pad_to_multiple_of",val:": typing.Optional[int] = None"},{name:"return_token_type_ids",val:": typing.Optional[bool] = None"},{name:"return_attention_mask",val:": typing.Optional[bool] = None"},{name:"return_overflowing_tokens",val:": bool = False"},{name:"return_special_tokens_mask",val:": bool = False"},{name:"return_offsets_mapping",val:": bool = False"},{name:"return_length",val:": bool = False"},{name:"verbose",val:": bool = True"},{name:"return_tensors",val:": typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None"},{name:"**kwargs",val:""}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/markuplm/processing_markuplm.py#L50"}}),Ut=new R({props:{title:"MarkupLMModel",local:"transformers.MarkupLMModel",headingTag:"h2"}}),Jt=new x({props:{name:"class transformers.MarkupLMModel",anchor:"transformers.MarkupLMModel",parameters:[{name:"config",val:""},{name:"add_pooling_layer",val:" = True"}],parametersDescription:[{anchor:"transformers.MarkupLMModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/markuplm#transformers.MarkupLMModel">MarkupLMModel</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"},{anchor:"transformers.MarkupLMModel.add_pooling_layer",description:`<strong>add_pooling_layer</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether to add a pooling layer`,name:"add_pooling_layer"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/markuplm/modeling_markuplm.py#L578"}}),zt=new x({props:{name:"forward",anchor:"transformers.MarkupLMModel.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"xpath_tags_seq",val:": typing.Optional[torch.LongTensor] = None"},{name:"xpath_subs_seq",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.MarkupLMModel.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.MarkupLMModel.forward.xpath_tags_seq",description:`<strong>xpath_tags_seq</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length, config.max_depth)</code>, <em>optional</em>) &#x2014;
Tag IDs for each token in the input sequence, padded up to config.max_depth.`,name:"xpath_tags_seq"},{anchor:"transformers.MarkupLMModel.forward.xpath_subs_seq",description:`<strong>xpath_subs_seq</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length, config.max_depth)</code>, <em>optional</em>) &#x2014;
Subscript IDs for each token in the input sequence, padded up to config.max_depth.`,name:"xpath_subs_seq"},{anchor:"transformers.MarkupLMModel.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.MarkupLMModel.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.MarkupLMModel.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.MarkupLMModel.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.MarkupLMModel.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.MarkupLMModel.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.MarkupLMModel.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.MarkupLMModel.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/markuplm/modeling_markuplm.py#L610",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling"
>transformers.modeling_outputs.BaseModelOutputWithPooling</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/markuplm#transformers.MarkupLMConfig"
>MarkupLMConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>last_hidden_state</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>) — Sequence of hidden-states at the output of the last layer of the model.</p>
</li>
<li>
<p><strong>pooler_output</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, hidden_size)</code>) — Last layer hidden-state of the first token of the sequence (classification token) after further processing
through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns
the classification token after processing through a linear layer and a tanh activation function. The linear
layer weights are trained from the next sentence prediction (classification) objective during pretraining.</p>
</li>
<li>
<p><strong>hidden_states</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings, if the model has an embedding layer, +
one for the output of each layer) of shape <code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.</p>
</li>
<li>
<p><strong>attentions</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
heads.</p>
</li>
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling"
>transformers.modeling_outputs.BaseModelOutputWithPooling</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),he=new Qo({props:{$$slots:{default:[Nr]},$$scope:{ctx:$}}}),fe=new Zn({props:{anchor:"transformers.MarkupLMModel.forward.example",$$slots:{default:[Ir]},$$scope:{ctx:$}}}),qt=new R({props:{title:"MarkupLMForSequenceClassification",local:"transformers.MarkupLMForSequenceClassification",headingTag:"h2"}}),jt=new x({props:{name:"class transformers.MarkupLMForSequenceClassification",anchor:"transformers.MarkupLMForSequenceClassification",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.MarkupLMForSequenceClassification.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/markuplm#transformers.MarkupLMForSequenceClassification">MarkupLMForSequenceClassification</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/markuplm/modeling_markuplm.py#L932"}}),Ct=new x({props:{name:"forward",anchor:"transformers.MarkupLMForSequenceClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"xpath_tags_seq",val:": typing.Optional[torch.Tensor] = None"},{name:"xpath_subs_seq",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"labels",val:": typing.Optional[torch.Tensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.MarkupLMForSequenceClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.MarkupLMForSequenceClassification.forward.xpath_tags_seq",description:`<strong>xpath_tags_seq</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length, config.max_depth)</code>, <em>optional</em>) &#x2014;
Tag IDs for each token in the input sequence, padded up to config.max_depth.`,name:"xpath_tags_seq"},{anchor:"transformers.MarkupLMForSequenceClassification.forward.xpath_subs_seq",description:`<strong>xpath_subs_seq</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length, config.max_depth)</code>, <em>optional</em>) &#x2014;
Subscript IDs for each token in the input sequence, padded up to config.max_depth.`,name:"xpath_subs_seq"},{anchor:"transformers.MarkupLMForSequenceClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.MarkupLMForSequenceClassification.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.MarkupLMForSequenceClassification.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.MarkupLMForSequenceClassification.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.MarkupLMForSequenceClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.MarkupLMForSequenceClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the sequence classification/regression loss. Indices should be in <code>[0, ..., config.num_labels - 1]</code>. If <code>config.num_labels == 1</code> a regression loss is computed (Mean-Square loss), If
<code>config.num_labels &gt; 1</code> a classification loss is computed (Cross-Entropy).`,name:"labels"},{anchor:"transformers.MarkupLMForSequenceClassification.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.MarkupLMForSequenceClassification.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.MarkupLMForSequenceClassification.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/markuplm/modeling_markuplm.py#L949",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput"
>transformers.modeling_outputs.SequenceClassifierOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/markuplm#transformers.MarkupLMConfig"
>MarkupLMConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>loss</strong> (<code>torch.FloatTensor</code> of shape <code>(1,)</code>, <em>optional</em>, returned when <code>labels</code> is provided) — Classification (or regression if config.num_labels==1) loss.</p>
</li>
<li>
<p><strong>logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, config.num_labels)</code>) — Classification (or regression if config.num_labels==1) scores (before SoftMax).</p>
</li>
<li>
<p><strong>hidden_states</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings, if the model has an embedding layer, +
one for the output of each layer) of shape <code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.</p>
</li>
<li>
<p><strong>attentions</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
heads.</p>
</li>
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput"
>transformers.modeling_outputs.SequenceClassifierOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),ge=new Qo({props:{$$slots:{default:[Zr]},$$scope:{ctx:$}}}),_e=new Zn({props:{anchor:"transformers.MarkupLMForSequenceClassification.forward.example",$$slots:{default:[Rr]},$$scope:{ctx:$}}}),Ft=new R({props:{title:"MarkupLMForTokenClassification",local:"transformers.MarkupLMForTokenClassification",headingTag:"h2"}}),Nt=new x({props:{name:"class transformers.MarkupLMForTokenClassification",anchor:"transformers.MarkupLMForTokenClassification",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.MarkupLMForTokenClassification.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/markuplm#transformers.MarkupLMForTokenClassification">MarkupLMForTokenClassification</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/markuplm/modeling_markuplm.py#L829"}}),It=new x({props:{name:"forward",anchor:"transformers.MarkupLMForTokenClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"xpath_tags_seq",val:": typing.Optional[torch.Tensor] = None"},{name:"xpath_subs_seq",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"labels",val:": typing.Optional[torch.Tensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.MarkupLMForTokenClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.MarkupLMForTokenClassification.forward.xpath_tags_seq",description:`<strong>xpath_tags_seq</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length, config.max_depth)</code>, <em>optional</em>) &#x2014;
Tag IDs for each token in the input sequence, padded up to config.max_depth.`,name:"xpath_tags_seq"},{anchor:"transformers.MarkupLMForTokenClassification.forward.xpath_subs_seq",description:`<strong>xpath_subs_seq</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length, config.max_depth)</code>, <em>optional</em>) &#x2014;
Subscript IDs for each token in the input sequence, padded up to config.max_depth.`,name:"xpath_subs_seq"},{anchor:"transformers.MarkupLMForTokenClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.MarkupLMForTokenClassification.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.MarkupLMForTokenClassification.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.MarkupLMForTokenClassification.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.MarkupLMForTokenClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.MarkupLMForTokenClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the token classification loss. Indices should be in <code>[0, ..., config.num_labels - 1]</code>.`,name:"labels"},{anchor:"transformers.MarkupLMForTokenClassification.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.MarkupLMForTokenClassification.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.MarkupLMForTokenClassification.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/markuplm/modeling_markuplm.py#L845",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.MaskedLMOutput"
>transformers.modeling_outputs.MaskedLMOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/markuplm#transformers.MarkupLMConfig"
>MarkupLMConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>loss</strong> (<code>torch.FloatTensor</code> of shape <code>(1,)</code>, <em>optional</em>, returned when <code>labels</code> is provided) — Masked language modeling (MLM) loss.</p>
</li>
<li>
<p><strong>logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, config.vocab_size)</code>) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).</p>
</li>
<li>
<p><strong>hidden_states</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings, if the model has an embedding layer, +
one for the output of each layer) of shape <code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.</p>
</li>
<li>
<p><strong>attentions</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
heads.</p>
</li>
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.MaskedLMOutput"
>transformers.modeling_outputs.MaskedLMOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),Me=new Qo({props:{$$slots:{default:[Br]},$$scope:{ctx:$}}}),ke=new Zn({props:{anchor:"transformers.MarkupLMForTokenClassification.forward.example",$$slots:{default:[Gr]},$$scope:{ctx:$}}}),Zt=new R({props:{title:"MarkupLMForQuestionAnswering",local:"transformers.MarkupLMForQuestionAnswering",headingTag:"h2"}}),Rt=new x({props:{name:"class transformers.MarkupLMForQuestionAnswering",anchor:"transformers.MarkupLMForQuestionAnswering",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.MarkupLMForQuestionAnswering.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/markuplm#transformers.MarkupLMForQuestionAnswering">MarkupLMForQuestionAnswering</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/markuplm/modeling_markuplm.py#L715"}}),Bt=new x({props:{name:"forward",anchor:"transformers.MarkupLMForQuestionAnswering.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"xpath_tags_seq",val:": typing.Optional[torch.Tensor] = None"},{name:"xpath_subs_seq",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"start_positions",val:": typing.Optional[torch.Tensor] = None"},{name:"end_positions",val:": typing.Optional[torch.Tensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.MarkupLMForQuestionAnswering.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.MarkupLMForQuestionAnswering.forward.xpath_tags_seq",description:`<strong>xpath_tags_seq</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length, config.max_depth)</code>, <em>optional</em>) &#x2014;
Tag IDs for each token in the input sequence, padded up to config.max_depth.`,name:"xpath_tags_seq"},{anchor:"transformers.MarkupLMForQuestionAnswering.forward.xpath_subs_seq",description:`<strong>xpath_subs_seq</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length, config.max_depth)</code>, <em>optional</em>) &#x2014;
Subscript IDs for each token in the input sequence, padded up to config.max_depth.`,name:"xpath_subs_seq"},{anchor:"transformers.MarkupLMForQuestionAnswering.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.MarkupLMForQuestionAnswering.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.MarkupLMForQuestionAnswering.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.MarkupLMForQuestionAnswering.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.MarkupLMForQuestionAnswering.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.MarkupLMForQuestionAnswering.forward.start_positions",description:`<strong>start_positions</strong> (<code>torch.Tensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for position (index) of the start of the labelled span for computing the token classification loss.
Positions are clamped to the length of the sequence (<code>sequence_length</code>). Position outside of the sequence
are not taken into account for computing the loss.`,name:"start_positions"},{anchor:"transformers.MarkupLMForQuestionAnswering.forward.end_positions",description:`<strong>end_positions</strong> (<code>torch.Tensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for position (index) of the end of the labelled span for computing the token classification loss.
Positions are clamped to the length of the sequence (<code>sequence_length</code>). Position outside of the sequence
are not taken into account for computing the loss.`,name:"end_positions"},{anchor:"transformers.MarkupLMForQuestionAnswering.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.MarkupLMForQuestionAnswering.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.MarkupLMForQuestionAnswering.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/markuplm/modeling_markuplm.py#L727",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.QuestionAnsweringModelOutput"
>transformers.modeling_outputs.QuestionAnsweringModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/markuplm#transformers.MarkupLMConfig"
>MarkupLMConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>loss</strong> (<code>torch.FloatTensor</code> of shape <code>(1,)</code>, <em>optional</em>, returned when <code>labels</code> is provided) — Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.</p>
</li>
<li>
<p><strong>start_logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>) — Span-start scores (before SoftMax).</p>
</li>
<li>
<p><strong>end_logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>) — Span-end scores (before SoftMax).</p>
</li>
<li>
<p><strong>hidden_states</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings, if the model has an embedding layer, +
one for the output of each layer) of shape <code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.</p>
</li>
<li>
<p><strong>attentions</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
heads.</p>
</li>
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.QuestionAnsweringModelOutput"
>transformers.modeling_outputs.QuestionAnsweringModelOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),be=new Qo({props:{$$slots:{default:[Hr]},$$scope:{ctx:$}}}),ye=new Zn({props:{anchor:"transformers.MarkupLMForQuestionAnswering.forward.example",$$slots:{default:[Wr]},$$scope:{ctx:$}}}),Gt=new jr({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/markuplm.md"}}),{c(){c=i("meta"),y=o(),k=i("p"),M=o(),b=i("p"),b.innerHTML=d,v=o(),m($e.$$.fragment),Hn=o(),re=i("div"),re.innerHTML=ca,Wn=o(),m(Ue.$$.fragment),En=o(),Je=i("p"),Je.innerHTML=pa,Vn=o(),ze=i("p"),ze.textContent=ma,Pn=o(),qe=i("ul"),qe.innerHTML=ua,Sn=o(),je=i("p"),je.textContent=ha,Qn=o(),Ce=i("p"),Ce.innerHTML=fa,Xn=o(),Fe=i("p"),Fe.innerHTML=ga,An=o(),m(Ne.$$.fragment),Yn=o(),Ie=i("ul"),Ie.innerHTML=_a,Dn=o(),ie=i("img"),On=o(),Ze=i("small"),Ze.innerHTML=ka,Kn=o(),m(Re.$$.fragment),eo=o(),Be=i("p"),Be.innerHTML=ba,to=o(),m(Ge.$$.fragment),no=o(),He=i("p"),He.innerHTML=ya,oo=o(),We=i("p"),We.innerHTML=Ta,so=o(),Ee=i("p"),Ee.textContent=va,ao=o(),Ve=i("p"),Ve.innerHTML=wa,ro=o(),Pe=i("p"),Pe.textContent=xa,io=o(),m(Se.$$.fragment),lo=o(),Qe=i("p"),Qe.innerHTML=La,co=o(),Xe=i("p"),Xe.innerHTML=$a,po=o(),m(Ae.$$.fragment),mo=o(),Ye=i("p"),Ye.innerHTML=Ua,uo=o(),De=i("p"),De.innerHTML=Ja,ho=o(),m(Oe.$$.fragment),fo=o(),Ke=i("p"),Ke.innerHTML=za,go=o(),et=i("p"),et.textContent=qa,_o=o(),m(tt.$$.fragment),Mo=o(),nt=i("p"),nt.innerHTML=ja,ko=o(),ot=i("p"),ot.innerHTML=Ca,bo=o(),m(st.$$.fragment),yo=o(),m(at.$$.fragment),To=o(),rt=i("ul"),rt.innerHTML=Fa,vo=o(),m(it.$$.fragment),wo=o(),B=i("div"),m(lt.$$.fragment),Xo=o(),Wt=i("p"),Wt.innerHTML=Na,Ao=o(),Et=i("p"),Et.innerHTML=Ia,Yo=o(),m(le.$$.fragment),xo=o(),m(dt.$$.fragment),Lo=o(),G=i("div"),m(ct.$$.fragment),Do=o(),Vt=i("p"),Vt.textContent=Za,Oo=o(),Pt=i("p"),Pt.innerHTML=Ra,Ko=o(),A=i("div"),m(pt.$$.fragment),es=o(),St=i("p"),St.textContent=Ba,ts=o(),m(de.$$.fragment),$o=o(),m(mt.$$.fragment),Uo=o(),z=i("div"),m(ut.$$.fragment),ns=o(),Qt=i("p"),Qt.innerHTML=Ga,os=o(),Y=i("div"),m(ht.$$.fragment),ss=o(),Xt=i("p"),Xt.textContent=Ha,as=o(),At=i("ul"),At.innerHTML=Wa,rs=o(),Yt=i("div"),m(ft.$$.fragment),is=o(),ce=i("div"),m(gt.$$.fragment),ls=o(),Dt=i("p"),Dt.textContent=Ea,ds=o(),Ot=i("div"),m(_t.$$.fragment),Jo=o(),m(Mt.$$.fragment),zo=o(),L=i("div"),m(kt.$$.fragment),cs=o(),Kt=i("p"),Kt.textContent=Va,ps=o(),en=i("p"),en.innerHTML=Pa,ms=o(),tn=i("p"),tn.textContent=Sa,us=o(),J=i("div"),m(bt.$$.fragment),hs=o(),nn=i("p"),nn.innerHTML=Qa,fs=o(),on=i("ul"),on.innerHTML=Xa,gs=o(),sn=i("p"),sn.innerHTML=Aa,_s=o(),an=i("ul"),an.innerHTML=Ya,Ms=o(),rn=i("p"),rn.innerHTML=Da,ks=o(),ln=i("ul"),ln.innerHTML=Oa,bs=o(),D=i("div"),m(yt.$$.fragment),ys=o(),dn=i("p"),dn.textContent=Ka,Ts=o(),cn=i("ul"),cn.innerHTML=er,vs=o(),pe=i("div"),m(Tt.$$.fragment),ws=o(),pn=i("p"),pn.textContent=tr,xs=o(),me=i("div"),m(vt.$$.fragment),Ls=o(),mn=i("p"),mn.innerHTML=nr,$s=o(),ue=i("div"),m(wt.$$.fragment),Us=o(),un=i("p"),un.textContent=or,qo=o(),m(xt.$$.fragment),jo=o(),j=i("div"),m(Lt.$$.fragment),Js=o(),hn=i("p"),hn.textContent=sr,zs=o(),fn=i("p"),fn.innerHTML=ar,qs=o(),gn=i("p"),gn.innerHTML=rr,js=o(),H=i("div"),m($t.$$.fragment),Cs=o(),_n=i("p"),_n.innerHTML=ir,Fs=o(),Mn=i("p"),Mn.innerHTML=lr,Ns=o(),kn=i("p"),kn.textContent=dr,Co=o(),m(Ut.$$.fragment),Fo=o(),C=i("div"),m(Jt.$$.fragment),Is=o(),bn=i("p"),bn.textContent=cr,Zs=o(),yn=i("p"),yn.innerHTML=pr,Rs=o(),Tn=i("p"),Tn.innerHTML=mr,Bs=o(),W=i("div"),m(zt.$$.fragment),Gs=o(),vn=i("p"),vn.innerHTML=ur,Hs=o(),m(he.$$.fragment),Ws=o(),m(fe.$$.fragment),No=o(),m(qt.$$.fragment),Io=o(),F=i("div"),m(jt.$$.fragment),Es=o(),wn=i("p"),wn.textContent=hr,Vs=o(),xn=i("p"),xn.innerHTML=fr,Ps=o(),Ln=i("p"),Ln.innerHTML=gr,Ss=o(),E=i("div"),m(Ct.$$.fragment),Qs=o(),$n=i("p"),$n.innerHTML=_r,Xs=o(),m(ge.$$.fragment),As=o(),m(_e.$$.fragment),Zo=o(),m(Ft.$$.fragment),Ro=o(),N=i("div"),m(Nt.$$.fragment),Ys=o(),Un=i("p"),Un.innerHTML=Mr,Ds=o(),Jn=i("p"),Jn.innerHTML=kr,Os=o(),zn=i("p"),zn.innerHTML=br,Ks=o(),V=i("div"),m(It.$$.fragment),ea=o(),qn=i("p"),qn.innerHTML=yr,ta=o(),m(Me.$$.fragment),na=o(),m(ke.$$.fragment),Bo=o(),m(Zt.$$.fragment),Go=o(),I=i("div"),m(Rt.$$.fragment),oa=o(),jn=i("p"),jn.innerHTML=Tr,sa=o(),Cn=i("p"),Cn.innerHTML=vr,aa=o(),Fn=i("p"),Fn.innerHTML=wr,ra=o(),P=i("div"),m(Bt.$$.fragment),ia=o(),Nn=i("p"),Nn.innerHTML=xr,la=o(),m(be.$$.fragment),da=o(),m(ye.$$.fragment),Ho=o(),m(Gt.$$.fragment),Wo=o(),Rn=i("p"),this.h()},l(e){const t=qr("svelte-u9bgzb",document.head);c=l(t,"META",{name:!0,content:!0}),t.forEach(n),y=s(e),k=l(e,"P",{}),w(k).forEach(n),M=s(e),b=l(e,"P",{"data-svelte-h":!0}),p(b)!=="svelte-c4l8bd"&&(b.innerHTML=d),v=s(e),u($e.$$.fragment,e),Hn=s(e),re=l(e,"DIV",{class:!0,"data-svelte-h":!0}),p(re)!=="svelte-13t8s2t"&&(re.innerHTML=ca),Wn=s(e),u(Ue.$$.fragment,e),En=s(e),Je=l(e,"P",{"data-svelte-h":!0}),p(Je)!=="svelte-16qq0ht"&&(Je.innerHTML=pa),Vn=s(e),ze=l(e,"P",{"data-svelte-h":!0}),p(ze)!=="svelte-1ytlk74"&&(ze.textContent=ma),Pn=s(e),qe=l(e,"UL",{"data-svelte-h":!0}),p(qe)!=="svelte-6h13ie"&&(qe.innerHTML=ua),Sn=s(e),je=l(e,"P",{"data-svelte-h":!0}),p(je)!=="svelte-vfdo9a"&&(je.textContent=ha),Qn=s(e),Ce=l(e,"P",{"data-svelte-h":!0}),p(Ce)!=="svelte-hdnp88"&&(Ce.innerHTML=fa),Xn=s(e),Fe=l(e,"P",{"data-svelte-h":!0}),p(Fe)!=="svelte-rdxa92"&&(Fe.innerHTML=ga),An=s(e),u(Ne.$$.fragment,e),Yn=s(e),Ie=l(e,"UL",{"data-svelte-h":!0}),p(Ie)!=="svelte-nrbn5b"&&(Ie.innerHTML=_a),Dn=s(e),ie=l(e,"IMG",{src:!0,alt:!0,width:!0}),On=s(e),Ze=l(e,"SMALL",{"data-svelte-h":!0}),p(Ze)!=="svelte-v7qfsu"&&(Ze.innerHTML=ka),Kn=s(e),u(Re.$$.fragment,e),eo=s(e),Be=l(e,"P",{"data-svelte-h":!0}),p(Be)!=="svelte-1kh26q"&&(Be.innerHTML=ba),to=s(e),u(Ge.$$.fragment,e),no=s(e),He=l(e,"P",{"data-svelte-h":!0}),p(He)!=="svelte-drm92f"&&(He.innerHTML=ya),oo=s(e),We=l(e,"P",{"data-svelte-h":!0}),p(We)!=="svelte-irw5rx"&&(We.innerHTML=Ta),so=s(e),Ee=l(e,"P",{"data-svelte-h":!0}),p(Ee)!=="svelte-jv0had"&&(Ee.textContent=va),ao=s(e),Ve=l(e,"P",{"data-svelte-h":!0}),p(Ve)!=="svelte-dkcsne"&&(Ve.innerHTML=wa),ro=s(e),Pe=l(e,"P",{"data-svelte-h":!0}),p(Pe)!=="svelte-6kpuj"&&(Pe.textContent=xa),io=s(e),u(Se.$$.fragment,e),lo=s(e),Qe=l(e,"P",{"data-svelte-h":!0}),p(Qe)!=="svelte-1i783jq"&&(Qe.innerHTML=La),co=s(e),Xe=l(e,"P",{"data-svelte-h":!0}),p(Xe)!=="svelte-glfvnv"&&(Xe.innerHTML=$a),po=s(e),u(Ae.$$.fragment,e),mo=s(e),Ye=l(e,"P",{"data-svelte-h":!0}),p(Ye)!=="svelte-owkrw8"&&(Ye.innerHTML=Ua),uo=s(e),De=l(e,"P",{"data-svelte-h":!0}),p(De)!=="svelte-1x7hckg"&&(De.innerHTML=Ja),ho=s(e),u(Oe.$$.fragment,e),fo=s(e),Ke=l(e,"P",{"data-svelte-h":!0}),p(Ke)!=="svelte-34jdzj"&&(Ke.innerHTML=za),go=s(e),et=l(e,"P",{"data-svelte-h":!0}),p(et)!=="svelte-11r3tgp"&&(et.textContent=qa),_o=s(e),u(tt.$$.fragment,e),Mo=s(e),nt=l(e,"P",{"data-svelte-h":!0}),p(nt)!=="svelte-147vwql"&&(nt.innerHTML=ja),ko=s(e),ot=l(e,"P",{"data-svelte-h":!0}),p(ot)!=="svelte-1wesasz"&&(ot.innerHTML=Ca),bo=s(e),u(st.$$.fragment,e),yo=s(e),u(at.$$.fragment,e),To=s(e),rt=l(e,"UL",{"data-svelte-h":!0}),p(rt)!=="svelte-1o7x1ln"&&(rt.innerHTML=Fa),vo=s(e),u(it.$$.fragment,e),wo=s(e),B=l(e,"DIV",{class:!0});var Q=w(B);u(lt.$$.fragment,Q),Xo=s(Q),Wt=l(Q,"P",{"data-svelte-h":!0}),p(Wt)!=="svelte-bbpwuk"&&(Wt.innerHTML=Na),Ao=s(Q),Et=l(Q,"P",{"data-svelte-h":!0}),p(Et)!=="svelte-xa1djz"&&(Et.innerHTML=Ia),Yo=s(Q),u(le.$$.fragment,Q),Q.forEach(n),xo=s(e),u(dt.$$.fragment,e),Lo=s(e),G=l(e,"DIV",{class:!0});var X=w(G);u(ct.$$.fragment,X),Do=s(X),Vt=l(X,"P",{"data-svelte-h":!0}),p(Vt)!=="svelte-t29t5v"&&(Vt.textContent=Za),Oo=s(X),Pt=l(X,"P",{"data-svelte-h":!0}),p(Pt)!=="svelte-1xyp3q9"&&(Pt.innerHTML=Ra),Ko=s(X),A=l(X,"DIV",{class:!0});var se=w(A);u(pt.$$.fragment,se),es=s(se),St=l(se,"P",{"data-svelte-h":!0}),p(St)!=="svelte-ep3db5"&&(St.textContent=Ba),ts=s(se),u(de.$$.fragment,se),se.forEach(n),X.forEach(n),$o=s(e),u(mt.$$.fragment,e),Uo=s(e),z=l(e,"DIV",{class:!0});var Z=w(z);u(ut.$$.fragment,Z),ns=s(Z),Qt=l(Z,"P",{"data-svelte-h":!0}),p(Qt)!=="svelte-zli1zx"&&(Qt.innerHTML=Ga),os=s(Z),Y=l(Z,"DIV",{class:!0});var ae=w(Y);u(ht.$$.fragment,ae),ss=s(ae),Xt=l(ae,"P",{"data-svelte-h":!0}),p(Xt)!=="svelte-og4clw"&&(Xt.textContent=Ha),as=s(ae),At=l(ae,"UL",{"data-svelte-h":!0}),p(At)!=="svelte-rq8uot"&&(At.innerHTML=Wa),ae.forEach(n),rs=s(Z),Yt=l(Z,"DIV",{class:!0});var Bn=w(Yt);u(ft.$$.fragment,Bn),Bn.forEach(n),is=s(Z),ce=l(Z,"DIV",{class:!0});var Ht=w(ce);u(gt.$$.fragment,Ht),ls=s(Ht),Dt=l(Ht,"P",{"data-svelte-h":!0}),p(Dt)!=="svelte-wwxeoo"&&(Dt.textContent=Ea),Ht.forEach(n),ds=s(Z),Ot=l(Z,"DIV",{class:!0});var Gn=w(Ot);u(_t.$$.fragment,Gn),Gn.forEach(n),Z.forEach(n),Jo=s(e),u(Mt.$$.fragment,e),zo=s(e),L=l(e,"DIV",{class:!0});var U=w(L);u(kt.$$.fragment,U),cs=s(U),Kt=l(U,"P",{"data-svelte-h":!0}),p(Kt)!=="svelte-w71jv6"&&(Kt.textContent=Va),ps=s(U),en=l(U,"P",{"data-svelte-h":!0}),p(en)!=="svelte-ve9m9s"&&(en.innerHTML=Pa),ms=s(U),tn=l(U,"P",{"data-svelte-h":!0}),p(tn)!=="svelte-1x24yjd"&&(tn.textContent=Sa),us=s(U),J=l(U,"DIV",{class:!0});var q=w(J);u(bt.$$.fragment,q),hs=s(q),nn=l(q,"P",{"data-svelte-h":!0}),p(nn)!=="svelte-1iqx5t9"&&(nn.innerHTML=Qa),fs=s(q),on=l(q,"UL",{"data-svelte-h":!0}),p(on)!=="svelte-ib4j9k"&&(on.innerHTML=Xa),gs=s(q),sn=l(q,"P",{"data-svelte-h":!0}),p(sn)!=="svelte-yhko47"&&(sn.innerHTML=Aa),_s=s(q),an=l(q,"UL",{"data-svelte-h":!0}),p(an)!=="svelte-sxb3sg"&&(an.innerHTML=Ya),Ms=s(q),rn=l(q,"P",{"data-svelte-h":!0}),p(rn)!=="svelte-oamp4"&&(rn.innerHTML=Da),ks=s(q),ln=l(q,"UL",{"data-svelte-h":!0}),p(ln)!=="svelte-1uk56nj"&&(ln.innerHTML=Oa),q.forEach(n),bs=s(U),D=l(U,"DIV",{class:!0});var In=w(D);u(yt.$$.fragment,In),ys=s(In),dn=l(In,"P",{"data-svelte-h":!0}),p(dn)!=="svelte-og4clw"&&(dn.textContent=Ka),Ts=s(In),cn=l(In,"UL",{"data-svelte-h":!0}),p(cn)!=="svelte-rq8uot"&&(cn.innerHTML=er),In.forEach(n),vs=s(U),pe=l(U,"DIV",{class:!0});var Vo=w(pe);u(Tt.$$.fragment,Vo),ws=s(Vo),pn=l(Vo,"P",{"data-svelte-h":!0}),p(pn)!=="svelte-wwxeoo"&&(pn.textContent=tr),Vo.forEach(n),xs=s(U),me=l(U,"DIV",{class:!0});var Po=w(me);u(vt.$$.fragment,Po),Ls=s(Po),mn=l(Po,"P",{"data-svelte-h":!0}),p(mn)!=="svelte-1eppb6b"&&(mn.innerHTML=nr),Po.forEach(n),$s=s(U),ue=l(U,"DIV",{class:!0});var So=w(ue);u(wt.$$.fragment,So),Us=s(So),un=l(So,"P",{"data-svelte-h":!0}),p(un)!=="svelte-19qak01"&&(un.textContent=or),So.forEach(n),U.forEach(n),qo=s(e),u(xt.$$.fragment,e),jo=s(e),j=l(e,"DIV",{class:!0});var O=w(j);u(Lt.$$.fragment,O),Js=s(O),hn=l(O,"P",{"data-svelte-h":!0}),p(hn)!=="svelte-jukxuq"&&(hn.textContent=sr),zs=s(O),fn=l(O,"P",{"data-svelte-h":!0}),p(fn)!=="svelte-1hszrsm"&&(fn.innerHTML=ar),qs=s(O),gn=l(O,"P",{"data-svelte-h":!0}),p(gn)!=="svelte-1xd2a3i"&&(gn.innerHTML=rr),js=s(O),H=l(O,"DIV",{class:!0});var Te=w(H);u($t.$$.fragment,Te),Cs=s(Te),_n=l(Te,"P",{"data-svelte-h":!0}),p(_n)!=="svelte-1o67nws"&&(_n.innerHTML=ir),Fs=s(Te),Mn=l(Te,"P",{"data-svelte-h":!0}),p(Mn)!=="svelte-yz8mju"&&(Mn.innerHTML=lr),Ns=s(Te),kn=l(Te,"P",{"data-svelte-h":!0}),p(kn)!=="svelte-ws0hzs"&&(kn.textContent=dr),Te.forEach(n),O.forEach(n),Co=s(e),u(Ut.$$.fragment,e),Fo=s(e),C=l(e,"DIV",{class:!0});var K=w(C);u(Jt.$$.fragment,K),Is=s(K),bn=l(K,"P",{"data-svelte-h":!0}),p(bn)!=="svelte-22lbx3"&&(bn.textContent=cr),Zs=s(K),yn=l(K,"P",{"data-svelte-h":!0}),p(yn)!=="svelte-q52n56"&&(yn.innerHTML=pr),Rs=s(K),Tn=l(K,"P",{"data-svelte-h":!0}),p(Tn)!=="svelte-hswkmf"&&(Tn.innerHTML=mr),Bs=s(K),W=l(K,"DIV",{class:!0});var ve=w(W);u(zt.$$.fragment,ve),Gs=s(ve),vn=l(ve,"P",{"data-svelte-h":!0}),p(vn)!=="svelte-c8vs40"&&(vn.innerHTML=ur),Hs=s(ve),u(he.$$.fragment,ve),Ws=s(ve),u(fe.$$.fragment,ve),ve.forEach(n),K.forEach(n),No=s(e),u(qt.$$.fragment,e),Io=s(e),F=l(e,"DIV",{class:!0});var ee=w(F);u(jt.$$.fragment,ee),Es=s(ee),wn=l(ee,"P",{"data-svelte-h":!0}),p(wn)!=="svelte-hwwz9z"&&(wn.textContent=hr),Vs=s(ee),xn=l(ee,"P",{"data-svelte-h":!0}),p(xn)!=="svelte-q52n56"&&(xn.innerHTML=fr),Ps=s(ee),Ln=l(ee,"P",{"data-svelte-h":!0}),p(Ln)!=="svelte-hswkmf"&&(Ln.innerHTML=gr),Ss=s(ee),E=l(ee,"DIV",{class:!0});var we=w(E);u(Ct.$$.fragment,we),Qs=s(we),$n=l(we,"P",{"data-svelte-h":!0}),p($n)!=="svelte-1r80ea0"&&($n.innerHTML=_r),Xs=s(we),u(ge.$$.fragment,we),As=s(we),u(_e.$$.fragment,we),we.forEach(n),ee.forEach(n),Zo=s(e),u(Ft.$$.fragment,e),Ro=s(e),N=l(e,"DIV",{class:!0});var te=w(N);u(Nt.$$.fragment,te),Ys=s(te),Un=l(te,"P",{"data-svelte-h":!0}),p(Un)!=="svelte-13zzcv4"&&(Un.innerHTML=Mr),Ds=s(te),Jn=l(te,"P",{"data-svelte-h":!0}),p(Jn)!=="svelte-q52n56"&&(Jn.innerHTML=kr),Os=s(te),zn=l(te,"P",{"data-svelte-h":!0}),p(zn)!=="svelte-hswkmf"&&(zn.innerHTML=br),Ks=s(te),V=l(te,"DIV",{class:!0});var xe=w(V);u(It.$$.fragment,xe),ea=s(xe),qn=l(xe,"P",{"data-svelte-h":!0}),p(qn)!=="svelte-q3bb4q"&&(qn.innerHTML=yr),ta=s(xe),u(Me.$$.fragment,xe),na=s(xe),u(ke.$$.fragment,xe),xe.forEach(n),te.forEach(n),Bo=s(e),u(Zt.$$.fragment,e),Go=s(e),I=l(e,"DIV",{class:!0});var ne=w(I);u(Rt.$$.fragment,ne),oa=s(ne),jn=l(ne,"P",{"data-svelte-h":!0}),p(jn)!=="svelte-h0zih4"&&(jn.innerHTML=Tr),sa=s(ne),Cn=l(ne,"P",{"data-svelte-h":!0}),p(Cn)!=="svelte-q52n56"&&(Cn.innerHTML=vr),aa=s(ne),Fn=l(ne,"P",{"data-svelte-h":!0}),p(Fn)!=="svelte-hswkmf"&&(Fn.innerHTML=wr),ra=s(ne),P=l(ne,"DIV",{class:!0});var Le=w(P);u(Bt.$$.fragment,Le),ia=s(Le),Nn=l(Le,"P",{"data-svelte-h":!0}),p(Nn)!=="svelte-vmq738"&&(Nn.innerHTML=xr),la=s(Le),u(be.$$.fragment,Le),da=s(Le),u(ye.$$.fragment,Le),Le.forEach(n),ne.forEach(n),Ho=s(e),u(Gt.$$.fragment,e),Wo=s(e),Rn=l(e,"P",{}),w(Rn).forEach(n),this.h()},h(){T(c,"name","hf:doc:metadata"),T(c,"content",Vr),T(re,"class","flex flex-wrap space-x-1"),$r(ie.src,Ma="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/markuplm_architecture.jpg")||T(ie,"src",Ma),T(ie,"alt","drawing"),T(ie,"width","600"),T(B,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),T(A,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),T(G,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),T(Y,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),T(Yt,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),T(ce,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),T(Ot,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),T(z,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),T(J,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),T(D,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),T(pe,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),T(me,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),T(ue,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),T(L,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),T(H,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),T(j,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),T(W,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),T(C,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),T(E,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),T(F,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),T(V,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),T(N,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),T(P,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),T(I,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,t){a(document.head,c),r(e,y,t),r(e,k,t),r(e,M,t),r(e,b,t),r(e,v,t),h($e,e,t),r(e,Hn,t),r(e,re,t),r(e,Wn,t),h(Ue,e,t),r(e,En,t),r(e,Je,t),r(e,Vn,t),r(e,ze,t),r(e,Pn,t),r(e,qe,t),r(e,Sn,t),r(e,je,t),r(e,Qn,t),r(e,Ce,t),r(e,Xn,t),r(e,Fe,t),r(e,An,t),h(Ne,e,t),r(e,Yn,t),r(e,Ie,t),r(e,Dn,t),r(e,ie,t),r(e,On,t),r(e,Ze,t),r(e,Kn,t),h(Re,e,t),r(e,eo,t),r(e,Be,t),r(e,to,t),h(Ge,e,t),r(e,no,t),r(e,He,t),r(e,oo,t),r(e,We,t),r(e,so,t),r(e,Ee,t),r(e,ao,t),r(e,Ve,t),r(e,ro,t),r(e,Pe,t),r(e,io,t),h(Se,e,t),r(e,lo,t),r(e,Qe,t),r(e,co,t),r(e,Xe,t),r(e,po,t),h(Ae,e,t),r(e,mo,t),r(e,Ye,t),r(e,uo,t),r(e,De,t),r(e,ho,t),h(Oe,e,t),r(e,fo,t),r(e,Ke,t),r(e,go,t),r(e,et,t),r(e,_o,t),h(tt,e,t),r(e,Mo,t),r(e,nt,t),r(e,ko,t),r(e,ot,t),r(e,bo,t),h(st,e,t),r(e,yo,t),h(at,e,t),r(e,To,t),r(e,rt,t),r(e,vo,t),h(it,e,t),r(e,wo,t),r(e,B,t),h(lt,B,null),a(B,Xo),a(B,Wt),a(B,Ao),a(B,Et),a(B,Yo),h(le,B,null),r(e,xo,t),h(dt,e,t),r(e,Lo,t),r(e,G,t),h(ct,G,null),a(G,Do),a(G,Vt),a(G,Oo),a(G,Pt),a(G,Ko),a(G,A),h(pt,A,null),a(A,es),a(A,St),a(A,ts),h(de,A,null),r(e,$o,t),h(mt,e,t),r(e,Uo,t),r(e,z,t),h(ut,z,null),a(z,ns),a(z,Qt),a(z,os),a(z,Y),h(ht,Y,null),a(Y,ss),a(Y,Xt),a(Y,as),a(Y,At),a(z,rs),a(z,Yt),h(ft,Yt,null),a(z,is),a(z,ce),h(gt,ce,null),a(ce,ls),a(ce,Dt),a(z,ds),a(z,Ot),h(_t,Ot,null),r(e,Jo,t),h(Mt,e,t),r(e,zo,t),r(e,L,t),h(kt,L,null),a(L,cs),a(L,Kt),a(L,ps),a(L,en),a(L,ms),a(L,tn),a(L,us),a(L,J),h(bt,J,null),a(J,hs),a(J,nn),a(J,fs),a(J,on),a(J,gs),a(J,sn),a(J,_s),a(J,an),a(J,Ms),a(J,rn),a(J,ks),a(J,ln),a(L,bs),a(L,D),h(yt,D,null),a(D,ys),a(D,dn),a(D,Ts),a(D,cn),a(L,vs),a(L,pe),h(Tt,pe,null),a(pe,ws),a(pe,pn),a(L,xs),a(L,me),h(vt,me,null),a(me,Ls),a(me,mn),a(L,$s),a(L,ue),h(wt,ue,null),a(ue,Us),a(ue,un),r(e,qo,t),h(xt,e,t),r(e,jo,t),r(e,j,t),h(Lt,j,null),a(j,Js),a(j,hn),a(j,zs),a(j,fn),a(j,qs),a(j,gn),a(j,js),a(j,H),h($t,H,null),a(H,Cs),a(H,_n),a(H,Fs),a(H,Mn),a(H,Ns),a(H,kn),r(e,Co,t),h(Ut,e,t),r(e,Fo,t),r(e,C,t),h(Jt,C,null),a(C,Is),a(C,bn),a(C,Zs),a(C,yn),a(C,Rs),a(C,Tn),a(C,Bs),a(C,W),h(zt,W,null),a(W,Gs),a(W,vn),a(W,Hs),h(he,W,null),a(W,Ws),h(fe,W,null),r(e,No,t),h(qt,e,t),r(e,Io,t),r(e,F,t),h(jt,F,null),a(F,Es),a(F,wn),a(F,Vs),a(F,xn),a(F,Ps),a(F,Ln),a(F,Ss),a(F,E),h(Ct,E,null),a(E,Qs),a(E,$n),a(E,Xs),h(ge,E,null),a(E,As),h(_e,E,null),r(e,Zo,t),h(Ft,e,t),r(e,Ro,t),r(e,N,t),h(Nt,N,null),a(N,Ys),a(N,Un),a(N,Ds),a(N,Jn),a(N,Os),a(N,zn),a(N,Ks),a(N,V),h(It,V,null),a(V,ea),a(V,qn),a(V,ta),h(Me,V,null),a(V,na),h(ke,V,null),r(e,Bo,t),h(Zt,e,t),r(e,Go,t),r(e,I,t),h(Rt,I,null),a(I,oa),a(I,jn),a(I,sa),a(I,Cn),a(I,aa),a(I,Fn),a(I,ra),a(I,P),h(Bt,P,null),a(P,ia),a(P,Nn),a(P,la),h(be,P,null),a(P,da),h(ye,P,null),r(e,Ho,t),h(Gt,e,t),r(e,Wo,t),r(e,Rn,t),Eo=!0},p(e,[t]){const Q={};t&2&&(Q.$$scope={dirty:t,ctx:e}),le.$set(Q);const X={};t&2&&(X.$$scope={dirty:t,ctx:e}),de.$set(X);const se={};t&2&&(se.$$scope={dirty:t,ctx:e}),he.$set(se);const Z={};t&2&&(Z.$$scope={dirty:t,ctx:e}),fe.$set(Z);const ae={};t&2&&(ae.$$scope={dirty:t,ctx:e}),ge.$set(ae);const Bn={};t&2&&(Bn.$$scope={dirty:t,ctx:e}),_e.$set(Bn);const Ht={};t&2&&(Ht.$$scope={dirty:t,ctx:e}),Me.$set(Ht);const Gn={};t&2&&(Gn.$$scope={dirty:t,ctx:e}),ke.$set(Gn);const U={};t&2&&(U.$$scope={dirty:t,ctx:e}),be.$set(U);const q={};t&2&&(q.$$scope={dirty:t,ctx:e}),ye.$set(q)},i(e){Eo||(f($e.$$.fragment,e),f(Ue.$$.fragment,e),f(Ne.$$.fragment,e),f(Re.$$.fragment,e),f(Ge.$$.fragment,e),f(Se.$$.fragment,e),f(Ae.$$.fragment,e),f(Oe.$$.fragment,e),f(tt.$$.fragment,e),f(st.$$.fragment,e),f(at.$$.fragment,e),f(it.$$.fragment,e),f(lt.$$.fragment,e),f(le.$$.fragment,e),f(dt.$$.fragment,e),f(ct.$$.fragment,e),f(pt.$$.fragment,e),f(de.$$.fragment,e),f(mt.$$.fragment,e),f(ut.$$.fragment,e),f(ht.$$.fragment,e),f(ft.$$.fragment,e),f(gt.$$.fragment,e),f(_t.$$.fragment,e),f(Mt.$$.fragment,e),f(kt.$$.fragment,e),f(bt.$$.fragment,e),f(yt.$$.fragment,e),f(Tt.$$.fragment,e),f(vt.$$.fragment,e),f(wt.$$.fragment,e),f(xt.$$.fragment,e),f(Lt.$$.fragment,e),f($t.$$.fragment,e),f(Ut.$$.fragment,e),f(Jt.$$.fragment,e),f(zt.$$.fragment,e),f(he.$$.fragment,e),f(fe.$$.fragment,e),f(qt.$$.fragment,e),f(jt.$$.fragment,e),f(Ct.$$.fragment,e),f(ge.$$.fragment,e),f(_e.$$.fragment,e),f(Ft.$$.fragment,e),f(Nt.$$.fragment,e),f(It.$$.fragment,e),f(Me.$$.fragment,e),f(ke.$$.fragment,e),f(Zt.$$.fragment,e),f(Rt.$$.fragment,e),f(Bt.$$.fragment,e),f(be.$$.fragment,e),f(ye.$$.fragment,e),f(Gt.$$.fragment,e),Eo=!0)},o(e){g($e.$$.fragment,e),g(Ue.$$.fragment,e),g(Ne.$$.fragment,e),g(Re.$$.fragment,e),g(Ge.$$.fragment,e),g(Se.$$.fragment,e),g(Ae.$$.fragment,e),g(Oe.$$.fragment,e),g(tt.$$.fragment,e),g(st.$$.fragment,e),g(at.$$.fragment,e),g(it.$$.fragment,e),g(lt.$$.fragment,e),g(le.$$.fragment,e),g(dt.$$.fragment,e),g(ct.$$.fragment,e),g(pt.$$.fragment,e),g(de.$$.fragment,e),g(mt.$$.fragment,e),g(ut.$$.fragment,e),g(ht.$$.fragment,e),g(ft.$$.fragment,e),g(gt.$$.fragment,e),g(_t.$$.fragment,e),g(Mt.$$.fragment,e),g(kt.$$.fragment,e),g(bt.$$.fragment,e),g(yt.$$.fragment,e),g(Tt.$$.fragment,e),g(vt.$$.fragment,e),g(wt.$$.fragment,e),g(xt.$$.fragment,e),g(Lt.$$.fragment,e),g($t.$$.fragment,e),g(Ut.$$.fragment,e),g(Jt.$$.fragment,e),g(zt.$$.fragment,e),g(he.$$.fragment,e),g(fe.$$.fragment,e),g(qt.$$.fragment,e),g(jt.$$.fragment,e),g(Ct.$$.fragment,e),g(ge.$$.fragment,e),g(_e.$$.fragment,e),g(Ft.$$.fragment,e),g(Nt.$$.fragment,e),g(It.$$.fragment,e),g(Me.$$.fragment,e),g(ke.$$.fragment,e),g(Zt.$$.fragment,e),g(Rt.$$.fragment,e),g(Bt.$$.fragment,e),g(be.$$.fragment,e),g(ye.$$.fragment,e),g(Gt.$$.fragment,e),Eo=!1},d(e){e&&(n(y),n(k),n(M),n(b),n(v),n(Hn),n(re),n(Wn),n(En),n(Je),n(Vn),n(ze),n(Pn),n(qe),n(Sn),n(je),n(Qn),n(Ce),n(Xn),n(Fe),n(An),n(Yn),n(Ie),n(Dn),n(ie),n(On),n(Ze),n(Kn),n(eo),n(Be),n(to),n(no),n(He),n(oo),n(We),n(so),n(Ee),n(ao),n(Ve),n(ro),n(Pe),n(io),n(lo),n(Qe),n(co),n(Xe),n(po),n(mo),n(Ye),n(uo),n(De),n(ho),n(fo),n(Ke),n(go),n(et),n(_o),n(Mo),n(nt),n(ko),n(ot),n(bo),n(yo),n(To),n(rt),n(vo),n(wo),n(B),n(xo),n(Lo),n(G),n($o),n(Uo),n(z),n(Jo),n(zo),n(L),n(qo),n(jo),n(j),n(Co),n(Fo),n(C),n(No),n(Io),n(F),n(Zo),n(Ro),n(N),n(Bo),n(Go),n(I),n(Ho),n(Wo),n(Rn)),n(c),_($e,e),_(Ue,e),_(Ne,e),_(Re,e),_(Ge,e),_(Se,e),_(Ae,e),_(Oe,e),_(tt,e),_(st,e),_(at,e),_(it,e),_(lt),_(le),_(dt,e),_(ct),_(pt),_(de),_(mt,e),_(ut),_(ht),_(ft),_(gt),_(_t),_(Mt,e),_(kt),_(bt),_(yt),_(Tt),_(vt),_(wt),_(xt,e),_(Lt),_($t),_(Ut,e),_(Jt),_(zt),_(he),_(fe),_(qt,e),_(jt),_(Ct),_(ge),_(_e),_(Ft,e),_(Nt),_(It),_(Me),_(ke),_(Zt,e),_(Rt),_(Bt),_(be),_(ye),_(Gt,e)}}}const Vr='{"title":"MarkupLM","local":"markuplm","sections":[{"title":"Overview","local":"overview","sections":[],"depth":2},{"title":"Usage tips","local":"usage-tips","sections":[],"depth":2},{"title":"Usage: MarkupLMProcessor","local":"usage-markuplmprocessor","sections":[],"depth":2},{"title":"Resources","local":"resources","sections":[],"depth":2},{"title":"MarkupLMConfig","local":"transformers.MarkupLMConfig","sections":[],"depth":2},{"title":"MarkupLMFeatureExtractor","local":"transformers.MarkupLMFeatureExtractor","sections":[],"depth":2},{"title":"MarkupLMTokenizer","local":"transformers.MarkupLMTokenizer","sections":[],"depth":2},{"title":"MarkupLMTokenizerFast","local":"transformers.MarkupLMTokenizerFast","sections":[],"depth":2},{"title":"MarkupLMProcessor","local":"transformers.MarkupLMProcessor","sections":[],"depth":2},{"title":"MarkupLMModel","local":"transformers.MarkupLMModel","sections":[],"depth":2},{"title":"MarkupLMForSequenceClassification","local":"transformers.MarkupLMForSequenceClassification","sections":[],"depth":2},{"title":"MarkupLMForTokenClassification","local":"transformers.MarkupLMForTokenClassification","sections":[],"depth":2},{"title":"MarkupLMForQuestionAnswering","local":"transformers.MarkupLMForQuestionAnswering","sections":[],"depth":2}],"depth":1}';function Pr($){return Ur(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class Kr extends Jr{constructor(c){super(),zr(this,c,Pr,Er,Lr,{})}}export{Kr as component};
