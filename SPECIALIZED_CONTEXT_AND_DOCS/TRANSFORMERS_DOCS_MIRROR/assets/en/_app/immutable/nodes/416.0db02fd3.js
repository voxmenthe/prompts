import{s as _t,o as Tt,n as Te}from"../chunks/scheduler.18a86fab.js";import{S as yt,i as vt,g as l,s as n,r as p,A as bt,h as m,f as s,c as a,j as ne,x as v,u as f,k as V,y as c,a as i,v as g,d as _,t as T,w as y}from"../chunks/index.98837b22.js";import{T as gt}from"../chunks/Tip.77304350.js";import{D as _e}from"../chunks/Docstring.a1ef7999.js";import{C as et}from"../chunks/CodeBlock.8d0c2e8a.js";import{E as Ke}from"../chunks/ExampleCodeBlock.8c3ee1f9.js";import{H as ae,E as wt}from"../chunks/getInferenceSnippets.06c2775f.js";function Mt(S){let o,u;return o=new et({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMFRpbWVTZXJpZXNUcmFuc2Zvcm1lckNvbmZpZyUyQyUyMFRpbWVTZXJpZXNUcmFuc2Zvcm1lck1vZGVsJTBBJTBBJTIzJTIwSW5pdGlhbGl6aW5nJTIwYSUyMFRpbWUlMjBTZXJpZXMlMjBUcmFuc2Zvcm1lciUyMGNvbmZpZ3VyYXRpb24lMjB3aXRoJTIwMTIlMjB0aW1lJTIwc3RlcHMlMjBmb3IlMjBwcmVkaWN0aW9uJTBBY29uZmlndXJhdGlvbiUyMCUzRCUyMFRpbWVTZXJpZXNUcmFuc2Zvcm1lckNvbmZpZyhwcmVkaWN0aW9uX2xlbmd0aCUzRDEyKSUwQSUwQSUyMyUyMFJhbmRvbWx5JTIwaW5pdGlhbGl6aW5nJTIwYSUyMG1vZGVsJTIwKHdpdGglMjByYW5kb20lMjB3ZWlnaHRzKSUyMGZyb20lMjB0aGUlMjBjb25maWd1cmF0aW9uJTBBbW9kZWwlMjAlM0QlMjBUaW1lU2VyaWVzVHJhbnNmb3JtZXJNb2RlbChjb25maWd1cmF0aW9uKSUwQSUwQSUyMyUyMEFjY2Vzc2luZyUyMHRoZSUyMG1vZGVsJTIwY29uZmlndXJhdGlvbiUwQWNvbmZpZ3VyYXRpb24lMjAlM0QlMjBtb2RlbC5jb25maWc=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> TimeSeriesTransformerConfig, TimeSeriesTransformerModel

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a Time Series Transformer configuration with 12 time steps for prediction</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = TimeSeriesTransformerConfig(prediction_length=<span class="hljs-number">12</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Randomly initializing a model (with random weights) from the configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = TimeSeriesTransformerModel(configuration)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Accessing the model configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = model.config`,wrap:!1}}),{c(){p(o.$$.fragment)},l(r){f(o.$$.fragment,r)},m(r,h){g(o,r,h),u=!0},p:Te,i(r){u||(_(o.$$.fragment,r),u=!0)},o(r){T(o.$$.fragment,r),u=!1},d(r){y(o,r)}}}function xt(S){let o,u=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){o=l("p"),o.innerHTML=u},l(r){o=m(r,"P",{"data-svelte-h":!0}),v(o)!=="svelte-fincs2"&&(o.innerHTML=u)},m(r,h){i(r,o,h)},p:Te,d(r){r&&s(o)}}}function kt(S){let o,u="Examples:",r,h,b;return h=new et({props:{code:"ZnJvbSUyMGh1Z2dpbmdmYWNlX2h1YiUyMGltcG9ydCUyMGhmX2h1Yl9kb3dubG9hZCUwQWltcG9ydCUyMHRvcmNoJTBBZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMFRpbWVTZXJpZXNUcmFuc2Zvcm1lck1vZGVsJTBBJTBBZmlsZSUyMCUzRCUyMGhmX2h1Yl9kb3dubG9hZCglMEElMjAlMjAlMjAlMjByZXBvX2lkJTNEJTIyaGYtaW50ZXJuYWwtdGVzdGluZyUyRnRvdXJpc20tbW9udGhseS1iYXRjaCUyMiUyQyUyMGZpbGVuYW1lJTNEJTIydHJhaW4tYmF0Y2gucHQlMjIlMkMlMjByZXBvX3R5cGUlM0QlMjJkYXRhc2V0JTIyJTBBKSUwQWJhdGNoJTIwJTNEJTIwdG9yY2gubG9hZChmaWxlKSUwQSUwQW1vZGVsJTIwJTNEJTIwVGltZVNlcmllc1RyYW5zZm9ybWVyTW9kZWwuZnJvbV9wcmV0cmFpbmVkKCUyMmh1Z2dpbmdmYWNlJTJGdGltZS1zZXJpZXMtdHJhbnNmb3JtZXItdG91cmlzbS1tb250aGx5JTIyKSUwQSUwQSUyMyUyMGR1cmluZyUyMHRyYWluaW5nJTJDJTIwb25lJTIwcHJvdmlkZXMlMjBib3RoJTIwcGFzdCUyMGFuZCUyMGZ1dHVyZSUyMHZhbHVlcyUwQSUyMyUyMGFzJTIwd2VsbCUyMGFzJTIwcG9zc2libGUlMjBhZGRpdGlvbmFsJTIwZmVhdHVyZXMlMEFvdXRwdXRzJTIwJTNEJTIwbW9kZWwoJTBBJTIwJTIwJTIwJTIwcGFzdF92YWx1ZXMlM0RiYXRjaCU1QiUyMnBhc3RfdmFsdWVzJTIyJTVEJTJDJTBBJTIwJTIwJTIwJTIwcGFzdF90aW1lX2ZlYXR1cmVzJTNEYmF0Y2glNUIlMjJwYXN0X3RpbWVfZmVhdHVyZXMlMjIlNUQlMkMlMEElMjAlMjAlMjAlMjBwYXN0X29ic2VydmVkX21hc2slM0RiYXRjaCU1QiUyMnBhc3Rfb2JzZXJ2ZWRfbWFzayUyMiU1RCUyQyUwQSUyMCUyMCUyMCUyMHN0YXRpY19jYXRlZ29yaWNhbF9mZWF0dXJlcyUzRGJhdGNoJTVCJTIyc3RhdGljX2NhdGVnb3JpY2FsX2ZlYXR1cmVzJTIyJTVEJTJDJTBBJTIwJTIwJTIwJTIwc3RhdGljX3JlYWxfZmVhdHVyZXMlM0RiYXRjaCU1QiUyMnN0YXRpY19yZWFsX2ZlYXR1cmVzJTIyJTVEJTJDJTBBJTIwJTIwJTIwJTIwZnV0dXJlX3ZhbHVlcyUzRGJhdGNoJTVCJTIyZnV0dXJlX3ZhbHVlcyUyMiU1RCUyQyUwQSUyMCUyMCUyMCUyMGZ1dHVyZV90aW1lX2ZlYXR1cmVzJTNEYmF0Y2glNUIlMjJmdXR1cmVfdGltZV9mZWF0dXJlcyUyMiU1RCUyQyUwQSklMEElMEFsYXN0X2hpZGRlbl9zdGF0ZSUyMCUzRCUyMG91dHB1dHMubGFzdF9oaWRkZW5fc3RhdGU=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> huggingface_hub <span class="hljs-keyword">import</span> hf_hub_download
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> TimeSeriesTransformerModel

<span class="hljs-meta">&gt;&gt;&gt; </span>file = hf_hub_download(
<span class="hljs-meta">... </span>    repo_id=<span class="hljs-string">&quot;hf-internal-testing/tourism-monthly-batch&quot;</span>, filename=<span class="hljs-string">&quot;train-batch.pt&quot;</span>, repo_type=<span class="hljs-string">&quot;dataset&quot;</span>
<span class="hljs-meta">... </span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>batch = torch.load(file)

<span class="hljs-meta">&gt;&gt;&gt; </span>model = TimeSeriesTransformerModel.from_pretrained(<span class="hljs-string">&quot;huggingface/time-series-transformer-tourism-monthly&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># during training, one provides both past and future values</span>
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># as well as possible additional features</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(
<span class="hljs-meta">... </span>    past_values=batch[<span class="hljs-string">&quot;past_values&quot;</span>],
<span class="hljs-meta">... </span>    past_time_features=batch[<span class="hljs-string">&quot;past_time_features&quot;</span>],
<span class="hljs-meta">... </span>    past_observed_mask=batch[<span class="hljs-string">&quot;past_observed_mask&quot;</span>],
<span class="hljs-meta">... </span>    static_categorical_features=batch[<span class="hljs-string">&quot;static_categorical_features&quot;</span>],
<span class="hljs-meta">... </span>    static_real_features=batch[<span class="hljs-string">&quot;static_real_features&quot;</span>],
<span class="hljs-meta">... </span>    future_values=batch[<span class="hljs-string">&quot;future_values&quot;</span>],
<span class="hljs-meta">... </span>    future_time_features=batch[<span class="hljs-string">&quot;future_time_features&quot;</span>],
<span class="hljs-meta">... </span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>last_hidden_state = outputs.last_hidden_state`,wrap:!1}}),{c(){o=l("p"),o.textContent=u,r=n(),p(h.$$.fragment)},l(d){o=m(d,"P",{"data-svelte-h":!0}),v(o)!=="svelte-kvfsh7"&&(o.textContent=u),r=a(d),f(h.$$.fragment,d)},m(d,C){i(d,o,C),i(d,r,C),g(h,d,C),b=!0},p:Te,i(d){b||(_(h.$$.fragment,d),b=!0)},o(d){T(h.$$.fragment,d),b=!1},d(d){d&&(s(o),s(r)),y(h,d)}}}function Ut(S){let o,u=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){o=l("p"),o.innerHTML=u},l(r){o=m(r,"P",{"data-svelte-h":!0}),v(o)!=="svelte-fincs2"&&(o.innerHTML=u)},m(r,h){i(r,o,h)},p:Te,d(r){r&&s(o)}}}function St(S){let o,u="Examples:",r,h,b;return h=new et({props:{code:"ZnJvbSUyMGh1Z2dpbmdmYWNlX2h1YiUyMGltcG9ydCUyMGhmX2h1Yl9kb3dubG9hZCUwQWltcG9ydCUyMHRvcmNoJTBBZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMFRpbWVTZXJpZXNUcmFuc2Zvcm1lckZvclByZWRpY3Rpb24lMEElMEFmaWxlJTIwJTNEJTIwaGZfaHViX2Rvd25sb2FkKCUwQSUyMCUyMCUyMCUyMHJlcG9faWQlM0QlMjJoZi1pbnRlcm5hbC10ZXN0aW5nJTJGdG91cmlzbS1tb250aGx5LWJhdGNoJTIyJTJDJTIwZmlsZW5hbWUlM0QlMjJ0cmFpbi1iYXRjaC5wdCUyMiUyQyUyMHJlcG9fdHlwZSUzRCUyMmRhdGFzZXQlMjIlMEEpJTBBYmF0Y2glMjAlM0QlMjB0b3JjaC5sb2FkKGZpbGUpJTBBJTBBbW9kZWwlMjAlM0QlMjBUaW1lU2VyaWVzVHJhbnNmb3JtZXJGb3JQcmVkaWN0aW9uLmZyb21fcHJldHJhaW5lZCglMEElMjAlMjAlMjAlMjAlMjJodWdnaW5nZmFjZSUyRnRpbWUtc2VyaWVzLXRyYW5zZm9ybWVyLXRvdXJpc20tbW9udGhseSUyMiUwQSklMEElMEElMjMlMjBkdXJpbmclMjB0cmFpbmluZyUyQyUyMG9uZSUyMHByb3ZpZGVzJTIwYm90aCUyMHBhc3QlMjBhbmQlMjBmdXR1cmUlMjB2YWx1ZXMlMEElMjMlMjBhcyUyMHdlbGwlMjBhcyUyMHBvc3NpYmxlJTIwYWRkaXRpb25hbCUyMGZlYXR1cmVzJTBBb3V0cHV0cyUyMCUzRCUyMG1vZGVsKCUwQSUyMCUyMCUyMCUyMHBhc3RfdmFsdWVzJTNEYmF0Y2glNUIlMjJwYXN0X3ZhbHVlcyUyMiU1RCUyQyUwQSUyMCUyMCUyMCUyMHBhc3RfdGltZV9mZWF0dXJlcyUzRGJhdGNoJTVCJTIycGFzdF90aW1lX2ZlYXR1cmVzJTIyJTVEJTJDJTBBJTIwJTIwJTIwJTIwcGFzdF9vYnNlcnZlZF9tYXNrJTNEYmF0Y2glNUIlMjJwYXN0X29ic2VydmVkX21hc2slMjIlNUQlMkMlMEElMjAlMjAlMjAlMjBzdGF0aWNfY2F0ZWdvcmljYWxfZmVhdHVyZXMlM0RiYXRjaCU1QiUyMnN0YXRpY19jYXRlZ29yaWNhbF9mZWF0dXJlcyUyMiU1RCUyQyUwQSUyMCUyMCUyMCUyMHN0YXRpY19yZWFsX2ZlYXR1cmVzJTNEYmF0Y2glNUIlMjJzdGF0aWNfcmVhbF9mZWF0dXJlcyUyMiU1RCUyQyUwQSUyMCUyMCUyMCUyMGZ1dHVyZV92YWx1ZXMlM0RiYXRjaCU1QiUyMmZ1dHVyZV92YWx1ZXMlMjIlNUQlMkMlMEElMjAlMjAlMjAlMjBmdXR1cmVfdGltZV9mZWF0dXJlcyUzRGJhdGNoJTVCJTIyZnV0dXJlX3RpbWVfZmVhdHVyZXMlMjIlNUQlMkMlMEEpJTBBJTBBbG9zcyUyMCUzRCUyMG91dHB1dHMubG9zcyUwQWxvc3MuYmFja3dhcmQoKSUwQSUwQSUyMyUyMGR1cmluZyUyMGluZmVyZW5jZSUyQyUyMG9uZSUyMG9ubHklMjBwcm92aWRlcyUyMHBhc3QlMjB2YWx1ZXMlMEElMjMlMjBhcyUyMHdlbGwlMjBhcyUyMHBvc3NpYmxlJTIwYWRkaXRpb25hbCUyMGZlYXR1cmVzJTBBJTIzJTIwdGhlJTIwbW9kZWwlMjBhdXRvcmVncmVzc2l2ZWx5JTIwZ2VuZXJhdGVzJTIwZnV0dXJlJTIwdmFsdWVzJTBBb3V0cHV0cyUyMCUzRCUyMG1vZGVsLmdlbmVyYXRlKCUwQSUyMCUyMCUyMCUyMHBhc3RfdmFsdWVzJTNEYmF0Y2glNUIlMjJwYXN0X3ZhbHVlcyUyMiU1RCUyQyUwQSUyMCUyMCUyMCUyMHBhc3RfdGltZV9mZWF0dXJlcyUzRGJhdGNoJTVCJTIycGFzdF90aW1lX2ZlYXR1cmVzJTIyJTVEJTJDJTBBJTIwJTIwJTIwJTIwcGFzdF9vYnNlcnZlZF9tYXNrJTNEYmF0Y2glNUIlMjJwYXN0X29ic2VydmVkX21hc2slMjIlNUQlMkMlMEElMjAlMjAlMjAlMjBzdGF0aWNfY2F0ZWdvcmljYWxfZmVhdHVyZXMlM0RiYXRjaCU1QiUyMnN0YXRpY19jYXRlZ29yaWNhbF9mZWF0dXJlcyUyMiU1RCUyQyUwQSUyMCUyMCUyMCUyMHN0YXRpY19yZWFsX2ZlYXR1cmVzJTNEYmF0Y2glNUIlMjJzdGF0aWNfcmVhbF9mZWF0dXJlcyUyMiU1RCUyQyUwQSUyMCUyMCUyMCUyMGZ1dHVyZV90aW1lX2ZlYXR1cmVzJTNEYmF0Y2glNUIlMjJmdXR1cmVfdGltZV9mZWF0dXJlcyUyMiU1RCUyQyUwQSklMEElMEFtZWFuX3ByZWRpY3Rpb24lMjAlM0QlMjBvdXRwdXRzLnNlcXVlbmNlcy5tZWFuKGRpbSUzRDEp",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> huggingface_hub <span class="hljs-keyword">import</span> hf_hub_download
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> TimeSeriesTransformerForPrediction

<span class="hljs-meta">&gt;&gt;&gt; </span>file = hf_hub_download(
<span class="hljs-meta">... </span>    repo_id=<span class="hljs-string">&quot;hf-internal-testing/tourism-monthly-batch&quot;</span>, filename=<span class="hljs-string">&quot;train-batch.pt&quot;</span>, repo_type=<span class="hljs-string">&quot;dataset&quot;</span>
<span class="hljs-meta">... </span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>batch = torch.load(file)

<span class="hljs-meta">&gt;&gt;&gt; </span>model = TimeSeriesTransformerForPrediction.from_pretrained(
<span class="hljs-meta">... </span>    <span class="hljs-string">&quot;huggingface/time-series-transformer-tourism-monthly&quot;</span>
<span class="hljs-meta">... </span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># during training, one provides both past and future values</span>
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># as well as possible additional features</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(
<span class="hljs-meta">... </span>    past_values=batch[<span class="hljs-string">&quot;past_values&quot;</span>],
<span class="hljs-meta">... </span>    past_time_features=batch[<span class="hljs-string">&quot;past_time_features&quot;</span>],
<span class="hljs-meta">... </span>    past_observed_mask=batch[<span class="hljs-string">&quot;past_observed_mask&quot;</span>],
<span class="hljs-meta">... </span>    static_categorical_features=batch[<span class="hljs-string">&quot;static_categorical_features&quot;</span>],
<span class="hljs-meta">... </span>    static_real_features=batch[<span class="hljs-string">&quot;static_real_features&quot;</span>],
<span class="hljs-meta">... </span>    future_values=batch[<span class="hljs-string">&quot;future_values&quot;</span>],
<span class="hljs-meta">... </span>    future_time_features=batch[<span class="hljs-string">&quot;future_time_features&quot;</span>],
<span class="hljs-meta">... </span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>loss = outputs.loss
<span class="hljs-meta">&gt;&gt;&gt; </span>loss.backward()

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># during inference, one only provides past values</span>
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># as well as possible additional features</span>
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># the model autoregressively generates future values</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model.generate(
<span class="hljs-meta">... </span>    past_values=batch[<span class="hljs-string">&quot;past_values&quot;</span>],
<span class="hljs-meta">... </span>    past_time_features=batch[<span class="hljs-string">&quot;past_time_features&quot;</span>],
<span class="hljs-meta">... </span>    past_observed_mask=batch[<span class="hljs-string">&quot;past_observed_mask&quot;</span>],
<span class="hljs-meta">... </span>    static_categorical_features=batch[<span class="hljs-string">&quot;static_categorical_features&quot;</span>],
<span class="hljs-meta">... </span>    static_real_features=batch[<span class="hljs-string">&quot;static_real_features&quot;</span>],
<span class="hljs-meta">... </span>    future_time_features=batch[<span class="hljs-string">&quot;future_time_features&quot;</span>],
<span class="hljs-meta">... </span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>mean_prediction = outputs.sequences.mean(dim=<span class="hljs-number">1</span>)`,wrap:!1}}),{c(){o=l("p"),o.textContent=u,r=n(),p(h.$$.fragment)},l(d){o=m(d,"P",{"data-svelte-h":!0}),v(o)!=="svelte-kvfsh7"&&(o.textContent=u),r=a(d),f(h.$$.fragment,d)},m(d,C){i(d,o,C),i(d,r,C),g(h,d,C),b=!0},p:Te,i(d){b||(_(h.$$.fragment,d),b=!0)},o(d){T(h.$$.fragment,d),b=!1},d(d){d&&(s(o),s(r)),y(h,d)}}}function Ct(S){let o,u,r,h,b,d="<em>This model was released on 2022-12-01 and added to Hugging Face Transformers on 2022-09-30.</em>",C,G,ye,q,tt='<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/>',ve,I,be,Y,ot=`The Time Series Transformer model is a vanilla encoder-decoder Transformer for time series forecasting.
This model was contributed by <a href="https://huggingface.co/kashif" rel="nofollow">kashif</a>.`,we,B,Me,H,st=`<li>Similar to other models in the library, <a href="/docs/transformers/v4.56.2/en/model_doc/time_series_transformer#transformers.TimeSeriesTransformerModel">TimeSeriesTransformerModel</a> is the raw Transformer without any head on top, and <a href="/docs/transformers/v4.56.2/en/model_doc/time_series_transformer#transformers.TimeSeriesTransformerForPrediction">TimeSeriesTransformerForPrediction</a>
adds a distribution head on top of the former, which can be used for time-series forecasting. Note that this is a so-called probabilistic forecasting model, not a
point forecasting model. This means that the model learns a distribution, from which one can sample. The model doesn’t directly output values.</li> <li><a href="/docs/transformers/v4.56.2/en/model_doc/time_series_transformer#transformers.TimeSeriesTransformerForPrediction">TimeSeriesTransformerForPrediction</a> consists of 2 blocks: an encoder, which takes a <code>context_length</code> of time series values as input (called <code>past_values</code>),
and a decoder, which predicts a <code>prediction_length</code> of time series values into the future (called <code>future_values</code>). During training, one needs to provide
pairs of (<code>past_values</code> and <code>future_values</code>) to the model.</li> <li>In addition to the raw (<code>past_values</code> and <code>future_values</code>), one typically provides additional features to the model. These can be the following:<ul><li><code>past_time_features</code>: temporal features which the model will add to <code>past_values</code>. These serve as “positional encodings” for the Transformer encoder.
Examples are “day of the month”, “month of the year”, etc. as scalar values (and then stacked together as a vector).
e.g. if a given time-series value was obtained on the 11th of August, then one could have [11, 8] as time feature vector (11 being “day of the month”, 8 being “month of the year”).</li> <li><code>future_time_features</code>: temporal features which the model will add to <code>future_values</code>. These serve as “positional encodings” for the Transformer decoder.
Examples are “day of the month”, “month of the year”, etc. as scalar values (and then stacked together as a vector).
e.g. if a given time-series value was obtained on the 11th of August, then one could have [11, 8] as time feature vector (11 being “day of the month”, 8 being “month of the year”).</li> <li><code>static_categorical_features</code>: categorical features which are static over time (i.e., have the same value for all <code>past_values</code> and <code>future_values</code>).
An example here is the store ID or region ID that identifies a given time-series.
Note that these features need to be known for ALL data points (also those in the future).</li> <li><code>static_real_features</code>: real-valued features which are static over time (i.e., have the same value for all <code>past_values</code> and <code>future_values</code>).
An example here is the image representation of the product for which you have the time-series values (like the <a href="resnet">ResNet</a> embedding of a “shoe” picture,
if your time-series is about the sales of shoes).
Note that these features need to be known for ALL data points (also those in the future).</li></ul></li> <li>The model is trained using “teacher-forcing”, similar to how a Transformer is trained for machine translation. This means that, during training, one shifts the
<code>future_values</code> one position to the right as input to the decoder, prepended by the last value of <code>past_values</code>. At each time step, the model needs to predict the
next target. So the set-up of training is similar to a GPT model for language, except that there’s no notion of <code>decoder_start_token_id</code> (we just use the last value
of the context as initial input for the decoder).</li> <li>At inference time, we give the final value of the <code>past_values</code> as input to the decoder. Next, we can sample from the model to make a prediction at the next time step,
which is then fed to the decoder in order to make the next prediction (also called autoregressive generation).</li>`,xe,E,ke,P,nt="A list of official Hugging Face and community (indicated by 🌎) resources to help you get started. If you’re interested in submitting a resource to be included here, please feel free to open a Pull Request and we’ll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.",Ue,D,at='<li>Check out the Time Series Transformer blog-post in HuggingFace blog: <a href="https://huggingface.co/blog/time-series-transformers" rel="nofollow">Probabilistic Time Series Forecasting with 🤗 Transformers</a></li>',Se,Q,Ce,x,O,Xe,re,rt=`This is the configuration class to store the configuration of a <a href="/docs/transformers/v4.56.2/en/model_doc/time_series_transformer#transformers.TimeSeriesTransformerModel">TimeSeriesTransformerModel</a>. It is used to
instantiate a Time Series Transformer model according to the specified arguments, defining the model architecture.
Instantiating a configuration with the defaults will yield a similar configuration to that of the Time Series
Transformer
<a href="https://huggingface.co/huggingface/time-series-transformer-tourism-monthly" rel="nofollow">huggingface/time-series-transformer-tourism-monthly</a>
architecture.`,Re,ie,it=`Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> can be used to control the model outputs. Read the
documentation from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> for more information.`,Ne,z,je,A,Je,w,L,We,de,dt="The bare Time Series Transformer Model outputting raw hidden-states without any specific head on top.",Ve,ce,ct=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Ge,le,lt=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Ie,j,K,Ye,me,mt='The <a href="/docs/transformers/v4.56.2/en/model_doc/time_series_transformer#transformers.TimeSeriesTransformerModel">TimeSeriesTransformerModel</a> forward method, overrides the <code>__call__</code> special method.',Be,X,He,R,$e,ee,Fe,M,te,Ee,he,ht="The Time Series Transformer Model with a distribution head on top for time-series forecasting.",Pe,ue,ut=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,De,pe,pt=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Qe,J,oe,Oe,fe,ft='The <a href="/docs/transformers/v4.56.2/en/model_doc/time_series_transformer#transformers.TimeSeriesTransformerForPrediction">TimeSeriesTransformerForPrediction</a> forward method, overrides the <code>__call__</code> special method.',Ae,N,Le,W,Ze,se,qe,ge,ze;return G=new ae({props:{title:"Time Series Transformer",local:"time-series-transformer",headingTag:"h1"}}),I=new ae({props:{title:"Overview",local:"overview",headingTag:"h2"}}),B=new ae({props:{title:"Usage tips",local:"usage-tips",headingTag:"h2"}}),E=new ae({props:{title:"Resources",local:"resources",headingTag:"h2"}}),Q=new ae({props:{title:"TimeSeriesTransformerConfig",local:"transformers.TimeSeriesTransformerConfig",headingTag:"h2"}}),O=new _e({props:{name:"class transformers.TimeSeriesTransformerConfig",anchor:"transformers.TimeSeriesTransformerConfig",parameters:[{name:"prediction_length",val:": typing.Optional[int] = None"},{name:"context_length",val:": typing.Optional[int] = None"},{name:"distribution_output",val:": str = 'student_t'"},{name:"loss",val:": str = 'nll'"},{name:"input_size",val:": int = 1"},{name:"lags_sequence",val:": list = [1, 2, 3, 4, 5, 6, 7]"},{name:"scaling",val:": typing.Union[str, bool, NoneType] = 'mean'"},{name:"num_dynamic_real_features",val:": int = 0"},{name:"num_static_categorical_features",val:": int = 0"},{name:"num_static_real_features",val:": int = 0"},{name:"num_time_features",val:": int = 0"},{name:"cardinality",val:": typing.Optional[list[int]] = None"},{name:"embedding_dimension",val:": typing.Optional[list[int]] = None"},{name:"encoder_ffn_dim",val:": int = 32"},{name:"decoder_ffn_dim",val:": int = 32"},{name:"encoder_attention_heads",val:": int = 2"},{name:"decoder_attention_heads",val:": int = 2"},{name:"encoder_layers",val:": int = 2"},{name:"decoder_layers",val:": int = 2"},{name:"is_encoder_decoder",val:": bool = True"},{name:"activation_function",val:": str = 'gelu'"},{name:"d_model",val:": int = 64"},{name:"dropout",val:": float = 0.1"},{name:"encoder_layerdrop",val:": float = 0.1"},{name:"decoder_layerdrop",val:": float = 0.1"},{name:"attention_dropout",val:": float = 0.1"},{name:"activation_dropout",val:": float = 0.1"},{name:"num_parallel_samples",val:": int = 100"},{name:"init_std",val:": float = 0.02"},{name:"use_cache",val:" = True"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.TimeSeriesTransformerConfig.prediction_length",description:`<strong>prediction_length</strong> (<code>int</code>) &#x2014;
The prediction length for the decoder. In other words, the prediction horizon of the model. This value is
typically dictated by the dataset and we recommend to set it appropriately.`,name:"prediction_length"},{anchor:"transformers.TimeSeriesTransformerConfig.context_length",description:`<strong>context_length</strong> (<code>int</code>, <em>optional</em>, defaults to <code>prediction_length</code>) &#x2014;
The context length for the encoder. If <code>None</code>, the context length will be the same as the
<code>prediction_length</code>.`,name:"context_length"},{anchor:"transformers.TimeSeriesTransformerConfig.distribution_output",description:`<strong>distribution_output</strong> (<code>string</code>, <em>optional</em>, defaults to <code>&quot;student_t&quot;</code>) &#x2014;
The distribution emission head for the model. Could be either &#x201C;student_t&#x201D;, &#x201C;normal&#x201D; or &#x201C;negative_binomial&#x201D;.`,name:"distribution_output"},{anchor:"transformers.TimeSeriesTransformerConfig.loss",description:`<strong>loss</strong> (<code>string</code>, <em>optional</em>, defaults to <code>&quot;nll&quot;</code>) &#x2014;
The loss function for the model corresponding to the <code>distribution_output</code> head. For parametric
distributions it is the negative log likelihood (nll) - which currently is the only supported one.`,name:"loss"},{anchor:"transformers.TimeSeriesTransformerConfig.input_size",description:`<strong>input_size</strong> (<code>int</code>, <em>optional</em>, defaults to 1) &#x2014;
The size of the target variable which by default is 1 for univariate targets. Would be &gt; 1 in case of
multivariate targets.`,name:"input_size"},{anchor:"transformers.TimeSeriesTransformerConfig.scaling",description:`<strong>scaling</strong> (<code>string</code> or <code>bool</code>, <em>optional</em> defaults to <code>&quot;mean&quot;</code>) &#x2014;
Whether to scale the input targets via &#x201C;mean&#x201D; scaler, &#x201C;std&#x201D; scaler or no scaler if <code>None</code>. If <code>True</code>, the
scaler is set to &#x201C;mean&#x201D;.`,name:"scaling"},{anchor:"transformers.TimeSeriesTransformerConfig.lags_sequence",description:`<strong>lags_sequence</strong> (<code>list[int]</code>, <em>optional</em>, defaults to <code>[1, 2, 3, 4, 5, 6, 7]</code>) &#x2014;
The lags of the input time series as covariates often dictated by the frequency of the data. Default is
<code>[1, 2, 3, 4, 5, 6, 7]</code> but we recommend to change it based on the dataset appropriately.`,name:"lags_sequence"},{anchor:"transformers.TimeSeriesTransformerConfig.num_time_features",description:`<strong>num_time_features</strong> (<code>int</code>, <em>optional</em>, defaults to 0) &#x2014;
The number of time features in the input time series.`,name:"num_time_features"},{anchor:"transformers.TimeSeriesTransformerConfig.num_dynamic_real_features",description:`<strong>num_dynamic_real_features</strong> (<code>int</code>, <em>optional</em>, defaults to 0) &#x2014;
The number of dynamic real valued features.`,name:"num_dynamic_real_features"},{anchor:"transformers.TimeSeriesTransformerConfig.num_static_categorical_features",description:`<strong>num_static_categorical_features</strong> (<code>int</code>, <em>optional</em>, defaults to 0) &#x2014;
The number of static categorical features.`,name:"num_static_categorical_features"},{anchor:"transformers.TimeSeriesTransformerConfig.num_static_real_features",description:`<strong>num_static_real_features</strong> (<code>int</code>, <em>optional</em>, defaults to 0) &#x2014;
The number of static real valued features.`,name:"num_static_real_features"},{anchor:"transformers.TimeSeriesTransformerConfig.cardinality",description:`<strong>cardinality</strong> (<code>list[int]</code>, <em>optional</em>) &#x2014;
The cardinality (number of different values) for each of the static categorical features. Should be a list
of integers, having the same length as <code>num_static_categorical_features</code>. Cannot be <code>None</code> if
<code>num_static_categorical_features</code> is &gt; 0.`,name:"cardinality"},{anchor:"transformers.TimeSeriesTransformerConfig.embedding_dimension",description:`<strong>embedding_dimension</strong> (<code>list[int]</code>, <em>optional</em>) &#x2014;
The dimension of the embedding for each of the static categorical features. Should be a list of integers,
having the same length as <code>num_static_categorical_features</code>. Cannot be <code>None</code> if
<code>num_static_categorical_features</code> is &gt; 0.`,name:"embedding_dimension"},{anchor:"transformers.TimeSeriesTransformerConfig.d_model",description:`<strong>d_model</strong> (<code>int</code>, <em>optional</em>, defaults to 64) &#x2014;
Dimensionality of the transformer layers.`,name:"d_model"},{anchor:"transformers.TimeSeriesTransformerConfig.encoder_layers",description:`<strong>encoder_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 2) &#x2014;
Number of encoder layers.`,name:"encoder_layers"},{anchor:"transformers.TimeSeriesTransformerConfig.decoder_layers",description:`<strong>decoder_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 2) &#x2014;
Number of decoder layers.`,name:"decoder_layers"},{anchor:"transformers.TimeSeriesTransformerConfig.encoder_attention_heads",description:`<strong>encoder_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 2) &#x2014;
Number of attention heads for each attention layer in the Transformer encoder.`,name:"encoder_attention_heads"},{anchor:"transformers.TimeSeriesTransformerConfig.decoder_attention_heads",description:`<strong>decoder_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 2) &#x2014;
Number of attention heads for each attention layer in the Transformer decoder.`,name:"decoder_attention_heads"},{anchor:"transformers.TimeSeriesTransformerConfig.encoder_ffn_dim",description:`<strong>encoder_ffn_dim</strong> (<code>int</code>, <em>optional</em>, defaults to 32) &#x2014;
Dimension of the &#x201C;intermediate&#x201D; (often named feed-forward) layer in encoder.`,name:"encoder_ffn_dim"},{anchor:"transformers.TimeSeriesTransformerConfig.decoder_ffn_dim",description:`<strong>decoder_ffn_dim</strong> (<code>int</code>, <em>optional</em>, defaults to 32) &#x2014;
Dimension of the &#x201C;intermediate&#x201D; (often named feed-forward) layer in decoder.`,name:"decoder_ffn_dim"},{anchor:"transformers.TimeSeriesTransformerConfig.activation_function",description:`<strong>activation_function</strong> (<code>str</code> or <code>function</code>, <em>optional</em>, defaults to <code>&quot;gelu&quot;</code>) &#x2014;
The non-linear activation function (function or string) in the encoder and decoder. If string, <code>&quot;gelu&quot;</code> and
<code>&quot;relu&quot;</code> are supported.`,name:"activation_function"},{anchor:"transformers.TimeSeriesTransformerConfig.dropout",description:`<strong>dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout probability for all fully connected layers in the encoder, and decoder.`,name:"dropout"},{anchor:"transformers.TimeSeriesTransformerConfig.encoder_layerdrop",description:`<strong>encoder_layerdrop</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout probability for the attention and fully connected layers for each encoder layer.`,name:"encoder_layerdrop"},{anchor:"transformers.TimeSeriesTransformerConfig.decoder_layerdrop",description:`<strong>decoder_layerdrop</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout probability for the attention and fully connected layers for each decoder layer.`,name:"decoder_layerdrop"},{anchor:"transformers.TimeSeriesTransformerConfig.attention_dropout",description:`<strong>attention_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout probability for the attention probabilities.`,name:"attention_dropout"},{anchor:"transformers.TimeSeriesTransformerConfig.activation_dropout",description:`<strong>activation_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout probability used between the two layers of the feed-forward networks.`,name:"activation_dropout"},{anchor:"transformers.TimeSeriesTransformerConfig.num_parallel_samples",description:`<strong>num_parallel_samples</strong> (<code>int</code>, <em>optional</em>, defaults to 100) &#x2014;
The number of samples to generate in parallel for each time step of inference.`,name:"num_parallel_samples"},{anchor:"transformers.TimeSeriesTransformerConfig.init_std",description:`<strong>init_std</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated normal weight initialization distribution.`,name:"init_std"},{anchor:"transformers.TimeSeriesTransformerConfig.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether to use the past key/values attentions (if applicable to the model) to speed up decoding.`,name:"use_cache"},{anchor:"transformers.TimeSeriesTransformerConfig.Example",description:"<strong>Example</strong> &#x2014;",name:"Example"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/time_series_transformer/configuration_time_series_transformer.py#L26"}}),z=new Ke({props:{anchor:"transformers.TimeSeriesTransformerConfig.example",$$slots:{default:[Mt]},$$scope:{ctx:S}}}),A=new ae({props:{title:"TimeSeriesTransformerModel",local:"transformers.TimeSeriesTransformerModel",headingTag:"h2"}}),L=new _e({props:{name:"class transformers.TimeSeriesTransformerModel",anchor:"transformers.TimeSeriesTransformerModel",parameters:[{name:"config",val:": TimeSeriesTransformerConfig"}],parametersDescription:[{anchor:"transformers.TimeSeriesTransformerModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/time_series_transformer#transformers.TimeSeriesTransformerConfig">TimeSeriesTransformerConfig</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/time_series_transformer/modeling_time_series_transformer.py#L1100"}}),K=new _e({props:{name:"forward",anchor:"transformers.TimeSeriesTransformerModel.forward",parameters:[{name:"past_values",val:": Tensor"},{name:"past_time_features",val:": Tensor"},{name:"past_observed_mask",val:": Tensor"},{name:"static_categorical_features",val:": typing.Optional[torch.Tensor] = None"},{name:"static_real_features",val:": typing.Optional[torch.Tensor] = None"},{name:"future_values",val:": typing.Optional[torch.Tensor] = None"},{name:"future_time_features",val:": typing.Optional[torch.Tensor] = None"},{name:"decoder_attention_mask",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"decoder_head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"cross_attn_head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_outputs",val:": typing.Optional[list[torch.FloatTensor]] = None"},{name:"past_key_values",val:": typing.Optional[list[torch.FloatTensor]] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"}],parametersDescription:[{anchor:"transformers.TimeSeriesTransformerModel.forward.past_values",description:`<strong>past_values</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code> or <code>(batch_size, sequence_length, input_size)</code>) &#x2014;
Past values of the time series, that serve as context in order to predict the future. The sequence size of
this tensor must be larger than the <code>context_length</code> of the model, since the model will use the larger size
to construct lag features, i.e. additional values from the past which are added in order to serve as &#x201C;extra
context&#x201D;.</p>
<p>The <code>sequence_length</code> here is equal to <code>config.context_length</code> + <code>max(config.lags_sequence)</code>, which if no
<code>lags_sequence</code> is configured, is equal to <code>config.context_length</code> + 7 (as by default, the largest
look-back index in <code>config.lags_sequence</code> is 7). The property <code>_past_length</code> returns the actual length of
the past.</p>
<p>The <code>past_values</code> is what the Transformer encoder gets as input (with optional additional features, such as
<code>static_categorical_features</code>, <code>static_real_features</code>, <code>past_time_features</code> and lags).</p>
<p>Optionally, missing values need to be replaced with zeros and indicated via the <code>past_observed_mask</code>.</p>
<p>For multivariate time series, the <code>input_size</code> &gt; 1 dimension is required and corresponds to the number of
variates in the time series per time step.`,name:"past_values"},{anchor:"transformers.TimeSeriesTransformerModel.forward.past_time_features",description:`<strong>past_time_features</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, num_features)</code>) &#x2014;
Required time features, which the model internally will add to <code>past_values</code>. These could be things like
&#x201C;month of year&#x201D;, &#x201C;day of the month&#x201D;, etc. encoded as vectors (for instance as Fourier features). These
could also be so-called &#x201C;age&#x201D; features, which basically help the model know &#x201C;at which point in life&#x201D; a
time-series is. Age features have small values for distant past time steps and increase monotonically the
more we approach the current time step. Holiday features are also a good example of time features.</p>
<p>These features serve as the &#x201C;positional encodings&#x201D; of the inputs. So contrary to a model like BERT, where
the position encodings are learned from scratch internally as parameters of the model, the Time Series
Transformer requires to provide additional time features. The Time Series Transformer only learns
additional embeddings for <code>static_categorical_features</code>.</p>
<p>Additional dynamic real covariates can be concatenated to this tensor, with the caveat that these features
must but known at prediction time.</p>
<p>The <code>num_features</code> here is equal to <code>config.</code>num_time_features<code>+</code>config.num_dynamic_real_features\`.`,name:"past_time_features"},{anchor:"transformers.TimeSeriesTransformerModel.forward.past_observed_mask",description:`<strong>past_observed_mask</strong> (<code>torch.BoolTensor</code> of shape <code>(batch_size, sequence_length)</code> or <code>(batch_size, sequence_length, input_size)</code>, <em>optional</em>) &#x2014;
Boolean mask to indicate which <code>past_values</code> were observed and which were missing. Mask values selected in
<code>[0, 1]</code>:</p>
<ul>
<li>1 for values that are <strong>observed</strong>,</li>
<li>0 for values that are <strong>missing</strong> (i.e. NaNs that were replaced by zeros).</li>
</ul>`,name:"past_observed_mask"},{anchor:"transformers.TimeSeriesTransformerModel.forward.static_categorical_features",description:`<strong>static_categorical_features</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, number of static categorical features)</code>, <em>optional</em>) &#x2014;
Optional static categorical features for which the model will learn an embedding, which it will add to the
values of the time series.</p>
<p>Static categorical features are features which have the same value for all time steps (static over time).</p>
<p>A typical example of a static categorical feature is a time series ID.`,name:"static_categorical_features"},{anchor:"transformers.TimeSeriesTransformerModel.forward.static_real_features",description:`<strong>static_real_features</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, number of static real features)</code>, <em>optional</em>) &#x2014;
Optional static real features which the model will add to the values of the time series.</p>
<p>Static real features are features which have the same value for all time steps (static over time).</p>
<p>A typical example of a static real feature is promotion information.`,name:"static_real_features"},{anchor:"transformers.TimeSeriesTransformerModel.forward.future_values",description:`<strong>future_values</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, prediction_length)</code> or <code>(batch_size, prediction_length, input_size)</code>, <em>optional</em>) &#x2014;
Future values of the time series, that serve as labels for the model. The <code>future_values</code> is what the
Transformer needs during training to learn to output, given the <code>past_values</code>.</p>
<p>The sequence length here is equal to <code>prediction_length</code>.</p>
<p>See the demo notebook and code snippets for details.</p>
<p>Optionally, during training any missing values need to be replaced with zeros and indicated via the
<code>future_observed_mask</code>.</p>
<p>For multivariate time series, the <code>input_size</code> &gt; 1 dimension is required and corresponds to the number of
variates in the time series per time step.`,name:"future_values"},{anchor:"transformers.TimeSeriesTransformerModel.forward.future_time_features",description:`<strong>future_time_features</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, prediction_length, num_features)</code>) &#x2014;
Required time features for the prediction window, which the model internally will add to <code>future_values</code>.
These could be things like &#x201C;month of year&#x201D;, &#x201C;day of the month&#x201D;, etc. encoded as vectors (for instance as
Fourier features). These could also be so-called &#x201C;age&#x201D; features, which basically help the model know &#x201C;at
which point in life&#x201D; a time-series is. Age features have small values for distant past time steps and
increase monotonically the more we approach the current time step. Holiday features are also a good example
of time features.</p>
<p>These features serve as the &#x201C;positional encodings&#x201D; of the inputs. So contrary to a model like BERT, where
the position encodings are learned from scratch internally as parameters of the model, the Time Series
Transformer requires to provide additional time features. The Time Series Transformer only learns
additional embeddings for <code>static_categorical_features</code>.</p>
<p>Additional dynamic real covariates can be concatenated to this tensor, with the caveat that these features
must but known at prediction time.</p>
<p>The <code>num_features</code> here is equal to <code>config.</code>num_time_features<code>+</code>config.num_dynamic_real_features\`.`,name:"future_time_features"},{anchor:"transformers.TimeSeriesTransformerModel.forward.decoder_attention_mask",description:`<strong>decoder_attention_mask</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on certain token indices. By default, a causal mask will be used, to
make sure the model can only look at previous inputs in order to predict the future.`,name:"decoder_attention_mask"},{anchor:"transformers.TimeSeriesTransformerModel.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.TimeSeriesTransformerModel.forward.decoder_head_mask",description:`<strong>decoder_head_mask</strong> (<code>torch.Tensor</code> of shape <code>(decoder_layers, decoder_attention_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the attention modules in the decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"decoder_head_mask"},{anchor:"transformers.TimeSeriesTransformerModel.forward.cross_attn_head_mask",description:`<strong>cross_attn_head_mask</strong> (<code>torch.Tensor</code> of shape <code>(decoder_layers, decoder_attention_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the cross-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"cross_attn_head_mask"},{anchor:"transformers.TimeSeriesTransformerModel.forward.encoder_outputs",description:`<strong>encoder_outputs</strong> (<code>tuple(tuple(torch.FloatTensor)</code>, <em>optional</em>) &#x2014;
Tuple consists of <code>last_hidden_state</code>, <code>hidden_states</code> (<em>optional</em>) and <code>attentions</code> (<em>optional</em>)
<code>last_hidden_state</code> of shape <code>(batch_size, sequence_length, hidden_size)</code> (<em>optional</em>) is a sequence of
hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.`,name:"encoder_outputs"},{anchor:"transformers.TimeSeriesTransformerModel.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>list[torch.FloatTensor]</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.TimeSeriesTransformerModel.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.TimeSeriesTransformerModel.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.TimeSeriesTransformerModel.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.TimeSeriesTransformerModel.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.TimeSeriesTransformerModel.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/time_series_transformer/modeling_time_series_transformer.py#L1241",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqTSModelOutput"
>transformers.modeling_outputs.Seq2SeqTSModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/time_series_transformer#transformers.TimeSeriesTransformerConfig"
>TimeSeriesTransformerConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>last_hidden_state</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>) — Sequence of hidden-states at the output of the last layer of the decoder of the model.</p>
<p>If <code>past_key_values</code> is used only the last hidden-state of the sequences of shape <code>(batch_size, 1, hidden_size)</code> is output.</p>
</li>
<li>
<p><strong>past_key_values</strong> (<code>EncoderDecoderCache</code>, <em>optional</em>, returned when <code>use_cache=True</code> is passed or when <code>config.use_cache=True</code>) — It is a <a
  href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.EncoderDecoderCache"
>EncoderDecoderCache</a> instance. For more details, see our <a
  href="https://huggingface.co/docs/transformers/en/kv_cache"
  rel="nofollow"
>kv cache guide</a>.</p>
<p>Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used (see <code>past_key_values</code> input) to speed up sequential decoding.</p>
</li>
<li>
<p><strong>decoder_hidden_states</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings, if the model has an embedding layer, +
one for the output of each layer) of shape <code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the decoder at the output of each layer plus the optional initial embedding outputs.</p>
</li>
<li>
<p><strong>decoder_attentions</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
self-attention heads.</p>
</li>
<li>
<p><strong>cross_attentions</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the
weighted average in the cross-attention heads.</p>
</li>
<li>
<p><strong>encoder_last_hidden_state</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) — Sequence of hidden-states at the output of the last layer of the encoder of the model.</p>
</li>
<li>
<p><strong>encoder_hidden_states</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings, if the model has an embedding layer, +
one for the output of each layer) of shape <code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the encoder at the output of each layer plus the optional initial embedding outputs.</p>
</li>
<li>
<p><strong>encoder_attentions</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
self-attention heads.</p>
</li>
<li>
<p><strong>loc</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size,)</code> or <code>(batch_size, input_size)</code>, <em>optional</em>) — Shift values of each time series’ context window which is used to give the model inputs of the same
magnitude and then used to shift back to the original magnitude.</p>
</li>
<li>
<p><strong>scale</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size,)</code> or <code>(batch_size, input_size)</code>, <em>optional</em>) — Scaling values of each time series’ context window which is used to give the model inputs of the same
magnitude and then used to rescale back to the original magnitude.</p>
</li>
<li>
<p><strong>static_features</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, feature size)</code>, <em>optional</em>) — Static features of each time series’ in a batch which are copied to the covariates at inference time.</p>
</li>
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqTSModelOutput"
>transformers.modeling_outputs.Seq2SeqTSModelOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),X=new gt({props:{$$slots:{default:[xt]},$$scope:{ctx:S}}}),R=new Ke({props:{anchor:"transformers.TimeSeriesTransformerModel.forward.example",$$slots:{default:[kt]},$$scope:{ctx:S}}}),ee=new ae({props:{title:"TimeSeriesTransformerForPrediction",local:"transformers.TimeSeriesTransformerForPrediction",headingTag:"h2"}}),te=new _e({props:{name:"class transformers.TimeSeriesTransformerForPrediction",anchor:"transformers.TimeSeriesTransformerForPrediction",parameters:[{name:"config",val:": TimeSeriesTransformerConfig"}],parametersDescription:[{anchor:"transformers.TimeSeriesTransformerForPrediction.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/time_series_transformer#transformers.TimeSeriesTransformerConfig">TimeSeriesTransformerConfig</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/time_series_transformer/modeling_time_series_transformer.py#L1462"}}),oe=new _e({props:{name:"forward",anchor:"transformers.TimeSeriesTransformerForPrediction.forward",parameters:[{name:"past_values",val:": Tensor"},{name:"past_time_features",val:": Tensor"},{name:"past_observed_mask",val:": Tensor"},{name:"static_categorical_features",val:": typing.Optional[torch.Tensor] = None"},{name:"static_real_features",val:": typing.Optional[torch.Tensor] = None"},{name:"future_values",val:": typing.Optional[torch.Tensor] = None"},{name:"future_time_features",val:": typing.Optional[torch.Tensor] = None"},{name:"future_observed_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"decoder_attention_mask",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"decoder_head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"cross_attn_head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_outputs",val:": typing.Optional[list[torch.FloatTensor]] = None"},{name:"past_key_values",val:": typing.Optional[list[torch.FloatTensor]] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"}],parametersDescription:[{anchor:"transformers.TimeSeriesTransformerForPrediction.forward.past_values",description:`<strong>past_values</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code> or <code>(batch_size, sequence_length, input_size)</code>) &#x2014;
Past values of the time series, that serve as context in order to predict the future. The sequence size of
this tensor must be larger than the <code>context_length</code> of the model, since the model will use the larger size
to construct lag features, i.e. additional values from the past which are added in order to serve as &#x201C;extra
context&#x201D;.</p>
<p>The <code>sequence_length</code> here is equal to <code>config.context_length</code> + <code>max(config.lags_sequence)</code>, which if no
<code>lags_sequence</code> is configured, is equal to <code>config.context_length</code> + 7 (as by default, the largest
look-back index in <code>config.lags_sequence</code> is 7). The property <code>_past_length</code> returns the actual length of
the past.</p>
<p>The <code>past_values</code> is what the Transformer encoder gets as input (with optional additional features, such as
<code>static_categorical_features</code>, <code>static_real_features</code>, <code>past_time_features</code> and lags).</p>
<p>Optionally, missing values need to be replaced with zeros and indicated via the <code>past_observed_mask</code>.</p>
<p>For multivariate time series, the <code>input_size</code> &gt; 1 dimension is required and corresponds to the number of
variates in the time series per time step.`,name:"past_values"},{anchor:"transformers.TimeSeriesTransformerForPrediction.forward.past_time_features",description:`<strong>past_time_features</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, num_features)</code>) &#x2014;
Required time features, which the model internally will add to <code>past_values</code>. These could be things like
&#x201C;month of year&#x201D;, &#x201C;day of the month&#x201D;, etc. encoded as vectors (for instance as Fourier features). These
could also be so-called &#x201C;age&#x201D; features, which basically help the model know &#x201C;at which point in life&#x201D; a
time-series is. Age features have small values for distant past time steps and increase monotonically the
more we approach the current time step. Holiday features are also a good example of time features.</p>
<p>These features serve as the &#x201C;positional encodings&#x201D; of the inputs. So contrary to a model like BERT, where
the position encodings are learned from scratch internally as parameters of the model, the Time Series
Transformer requires to provide additional time features. The Time Series Transformer only learns
additional embeddings for <code>static_categorical_features</code>.</p>
<p>Additional dynamic real covariates can be concatenated to this tensor, with the caveat that these features
must but known at prediction time.</p>
<p>The <code>num_features</code> here is equal to <code>config.</code>num_time_features<code>+</code>config.num_dynamic_real_features\`.`,name:"past_time_features"},{anchor:"transformers.TimeSeriesTransformerForPrediction.forward.past_observed_mask",description:`<strong>past_observed_mask</strong> (<code>torch.BoolTensor</code> of shape <code>(batch_size, sequence_length)</code> or <code>(batch_size, sequence_length, input_size)</code>, <em>optional</em>) &#x2014;
Boolean mask to indicate which <code>past_values</code> were observed and which were missing. Mask values selected in
<code>[0, 1]</code>:</p>
<ul>
<li>1 for values that are <strong>observed</strong>,</li>
<li>0 for values that are <strong>missing</strong> (i.e. NaNs that were replaced by zeros).</li>
</ul>`,name:"past_observed_mask"},{anchor:"transformers.TimeSeriesTransformerForPrediction.forward.static_categorical_features",description:`<strong>static_categorical_features</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, number of static categorical features)</code>, <em>optional</em>) &#x2014;
Optional static categorical features for which the model will learn an embedding, which it will add to the
values of the time series.</p>
<p>Static categorical features are features which have the same value for all time steps (static over time).</p>
<p>A typical example of a static categorical feature is a time series ID.`,name:"static_categorical_features"},{anchor:"transformers.TimeSeriesTransformerForPrediction.forward.static_real_features",description:`<strong>static_real_features</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, number of static real features)</code>, <em>optional</em>) &#x2014;
Optional static real features which the model will add to the values of the time series.</p>
<p>Static real features are features which have the same value for all time steps (static over time).</p>
<p>A typical example of a static real feature is promotion information.`,name:"static_real_features"},{anchor:"transformers.TimeSeriesTransformerForPrediction.forward.future_values",description:`<strong>future_values</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, prediction_length)</code> or <code>(batch_size, prediction_length, input_size)</code>, <em>optional</em>) &#x2014;
Future values of the time series, that serve as labels for the model. The <code>future_values</code> is what the
Transformer needs during training to learn to output, given the <code>past_values</code>.</p>
<p>The sequence length here is equal to <code>prediction_length</code>.</p>
<p>See the demo notebook and code snippets for details.</p>
<p>Optionally, during training any missing values need to be replaced with zeros and indicated via the
<code>future_observed_mask</code>.</p>
<p>For multivariate time series, the <code>input_size</code> &gt; 1 dimension is required and corresponds to the number of
variates in the time series per time step.`,name:"future_values"},{anchor:"transformers.TimeSeriesTransformerForPrediction.forward.future_time_features",description:`<strong>future_time_features</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, prediction_length, num_features)</code>) &#x2014;
Required time features for the prediction window, which the model internally will add to <code>future_values</code>.
These could be things like &#x201C;month of year&#x201D;, &#x201C;day of the month&#x201D;, etc. encoded as vectors (for instance as
Fourier features). These could also be so-called &#x201C;age&#x201D; features, which basically help the model know &#x201C;at
which point in life&#x201D; a time-series is. Age features have small values for distant past time steps and
increase monotonically the more we approach the current time step. Holiday features are also a good example
of time features.</p>
<p>These features serve as the &#x201C;positional encodings&#x201D; of the inputs. So contrary to a model like BERT, where
the position encodings are learned from scratch internally as parameters of the model, the Time Series
Transformer requires to provide additional time features. The Time Series Transformer only learns
additional embeddings for <code>static_categorical_features</code>.</p>
<p>Additional dynamic real covariates can be concatenated to this tensor, with the caveat that these features
must but known at prediction time.</p>
<p>The <code>num_features</code> here is equal to <code>config.</code>num_time_features<code>+</code>config.num_dynamic_real_features\`.`,name:"future_time_features"},{anchor:"transformers.TimeSeriesTransformerForPrediction.forward.future_observed_mask",description:`<strong>future_observed_mask</strong> (<code>torch.BoolTensor</code> of shape <code>(batch_size, sequence_length)</code> or <code>(batch_size, sequence_length, input_size)</code>, <em>optional</em>) &#x2014;
Boolean mask to indicate which <code>future_values</code> were observed and which were missing. Mask values selected
in <code>[0, 1]</code>:</p>
<ul>
<li>1 for values that are <strong>observed</strong>,</li>
<li>0 for values that are <strong>missing</strong> (i.e. NaNs that were replaced by zeros).</li>
</ul>
<p>This mask is used to filter out missing values for the final loss calculation.`,name:"future_observed_mask"},{anchor:"transformers.TimeSeriesTransformerForPrediction.forward.decoder_attention_mask",description:`<strong>decoder_attention_mask</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on certain token indices. By default, a causal mask will be used, to
make sure the model can only look at previous inputs in order to predict the future.`,name:"decoder_attention_mask"},{anchor:"transformers.TimeSeriesTransformerForPrediction.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.TimeSeriesTransformerForPrediction.forward.decoder_head_mask",description:`<strong>decoder_head_mask</strong> (<code>torch.Tensor</code> of shape <code>(decoder_layers, decoder_attention_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the attention modules in the decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"decoder_head_mask"},{anchor:"transformers.TimeSeriesTransformerForPrediction.forward.cross_attn_head_mask",description:`<strong>cross_attn_head_mask</strong> (<code>torch.Tensor</code> of shape <code>(decoder_layers, decoder_attention_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the cross-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"cross_attn_head_mask"},{anchor:"transformers.TimeSeriesTransformerForPrediction.forward.encoder_outputs",description:`<strong>encoder_outputs</strong> (<code>tuple(tuple(torch.FloatTensor)</code>, <em>optional</em>) &#x2014;
Tuple consists of <code>last_hidden_state</code>, <code>hidden_states</code> (<em>optional</em>) and <code>attentions</code> (<em>optional</em>)
<code>last_hidden_state</code> of shape <code>(batch_size, sequence_length, hidden_size)</code> (<em>optional</em>) is a sequence of
hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.`,name:"encoder_outputs"},{anchor:"transformers.TimeSeriesTransformerForPrediction.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>list[torch.FloatTensor]</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.TimeSeriesTransformerForPrediction.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.TimeSeriesTransformerForPrediction.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.TimeSeriesTransformerForPrediction.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.TimeSeriesTransformerForPrediction.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.TimeSeriesTransformerForPrediction.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/time_series_transformer/modeling_time_series_transformer.py#L1502",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqTSModelOutput"
>transformers.modeling_outputs.Seq2SeqTSModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/time_series_transformer#transformers.TimeSeriesTransformerConfig"
>TimeSeriesTransformerConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>last_hidden_state</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>) — Sequence of hidden-states at the output of the last layer of the decoder of the model.</p>
<p>If <code>past_key_values</code> is used only the last hidden-state of the sequences of shape <code>(batch_size, 1, hidden_size)</code> is output.</p>
</li>
<li>
<p><strong>past_key_values</strong> (<code>EncoderDecoderCache</code>, <em>optional</em>, returned when <code>use_cache=True</code> is passed or when <code>config.use_cache=True</code>) — It is a <a
  href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.EncoderDecoderCache"
>EncoderDecoderCache</a> instance. For more details, see our <a
  href="https://huggingface.co/docs/transformers/en/kv_cache"
  rel="nofollow"
>kv cache guide</a>.</p>
<p>Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used (see <code>past_key_values</code> input) to speed up sequential decoding.</p>
</li>
<li>
<p><strong>decoder_hidden_states</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings, if the model has an embedding layer, +
one for the output of each layer) of shape <code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the decoder at the output of each layer plus the optional initial embedding outputs.</p>
</li>
<li>
<p><strong>decoder_attentions</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
self-attention heads.</p>
</li>
<li>
<p><strong>cross_attentions</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the
weighted average in the cross-attention heads.</p>
</li>
<li>
<p><strong>encoder_last_hidden_state</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) — Sequence of hidden-states at the output of the last layer of the encoder of the model.</p>
</li>
<li>
<p><strong>encoder_hidden_states</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings, if the model has an embedding layer, +
one for the output of each layer) of shape <code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the encoder at the output of each layer plus the optional initial embedding outputs.</p>
</li>
<li>
<p><strong>encoder_attentions</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
self-attention heads.</p>
</li>
<li>
<p><strong>loc</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size,)</code> or <code>(batch_size, input_size)</code>, <em>optional</em>) — Shift values of each time series’ context window which is used to give the model inputs of the same
magnitude and then used to shift back to the original magnitude.</p>
</li>
<li>
<p><strong>scale</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size,)</code> or <code>(batch_size, input_size)</code>, <em>optional</em>) — Scaling values of each time series’ context window which is used to give the model inputs of the same
magnitude and then used to rescale back to the original magnitude.</p>
</li>
<li>
<p><strong>static_features</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, feature size)</code>, <em>optional</em>) — Static features of each time series’ in a batch which are copied to the covariates at inference time.</p>
</li>
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqTSModelOutput"
>transformers.modeling_outputs.Seq2SeqTSModelOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),N=new gt({props:{$$slots:{default:[Ut]},$$scope:{ctx:S}}}),W=new Ke({props:{anchor:"transformers.TimeSeriesTransformerForPrediction.forward.example",$$slots:{default:[St]},$$scope:{ctx:S}}}),se=new wt({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/time_series_transformer.md"}}),{c(){o=l("meta"),u=n(),r=l("p"),h=n(),b=l("p"),b.innerHTML=d,C=n(),p(G.$$.fragment),ye=n(),q=l("div"),q.innerHTML=tt,ve=n(),p(I.$$.fragment),be=n(),Y=l("p"),Y.innerHTML=ot,we=n(),p(B.$$.fragment),Me=n(),H=l("ul"),H.innerHTML=st,xe=n(),p(E.$$.fragment),ke=n(),P=l("p"),P.textContent=nt,Ue=n(),D=l("ul"),D.innerHTML=at,Se=n(),p(Q.$$.fragment),Ce=n(),x=l("div"),p(O.$$.fragment),Xe=n(),re=l("p"),re.innerHTML=rt,Re=n(),ie=l("p"),ie.innerHTML=it,Ne=n(),p(z.$$.fragment),je=n(),p(A.$$.fragment),Je=n(),w=l("div"),p(L.$$.fragment),We=n(),de=l("p"),de.textContent=dt,Ve=n(),ce=l("p"),ce.innerHTML=ct,Ge=n(),le=l("p"),le.innerHTML=lt,Ie=n(),j=l("div"),p(K.$$.fragment),Ye=n(),me=l("p"),me.innerHTML=mt,Be=n(),p(X.$$.fragment),He=n(),p(R.$$.fragment),$e=n(),p(ee.$$.fragment),Fe=n(),M=l("div"),p(te.$$.fragment),Ee=n(),he=l("p"),he.textContent=ht,Pe=n(),ue=l("p"),ue.innerHTML=ut,De=n(),pe=l("p"),pe.innerHTML=pt,Qe=n(),J=l("div"),p(oe.$$.fragment),Oe=n(),fe=l("p"),fe.innerHTML=ft,Ae=n(),p(N.$$.fragment),Le=n(),p(W.$$.fragment),Ze=n(),p(se.$$.fragment),qe=n(),ge=l("p"),this.h()},l(e){const t=bt("svelte-u9bgzb",document.head);o=m(t,"META",{name:!0,content:!0}),t.forEach(s),u=a(e),r=m(e,"P",{}),ne(r).forEach(s),h=a(e),b=m(e,"P",{"data-svelte-h":!0}),v(b)!=="svelte-117bkdu"&&(b.innerHTML=d),C=a(e),f(G.$$.fragment,e),ye=a(e),q=m(e,"DIV",{class:!0,"data-svelte-h":!0}),v(q)!=="svelte-13t8s2t"&&(q.innerHTML=tt),ve=a(e),f(I.$$.fragment,e),be=a(e),Y=m(e,"P",{"data-svelte-h":!0}),v(Y)!=="svelte-gds1oq"&&(Y.innerHTML=ot),we=a(e),f(B.$$.fragment,e),Me=a(e),H=m(e,"UL",{"data-svelte-h":!0}),v(H)!=="svelte-hejnay"&&(H.innerHTML=st),xe=a(e),f(E.$$.fragment,e),ke=a(e),P=m(e,"P",{"data-svelte-h":!0}),v(P)!=="svelte-1e7xzkp"&&(P.textContent=nt),Ue=a(e),D=m(e,"UL",{"data-svelte-h":!0}),v(D)!=="svelte-noat2w"&&(D.innerHTML=at),Se=a(e),f(Q.$$.fragment,e),Ce=a(e),x=m(e,"DIV",{class:!0});var $=ne(x);f(O.$$.fragment,$),Xe=a($),re=m($,"P",{"data-svelte-h":!0}),v(re)!=="svelte-1htz1sy"&&(re.innerHTML=rt),Re=a($),ie=m($,"P",{"data-svelte-h":!0}),v(ie)!=="svelte-1ynyot8"&&(ie.innerHTML=it),Ne=a($),f(z.$$.fragment,$),$.forEach(s),je=a(e),f(A.$$.fragment,e),Je=a(e),w=m(e,"DIV",{class:!0});var k=ne(w);f(L.$$.fragment,k),We=a(k),de=m(k,"P",{"data-svelte-h":!0}),v(de)!=="svelte-14ffg19"&&(de.textContent=dt),Ve=a(k),ce=m(k,"P",{"data-svelte-h":!0}),v(ce)!=="svelte-q52n56"&&(ce.innerHTML=ct),Ge=a(k),le=m(k,"P",{"data-svelte-h":!0}),v(le)!=="svelte-hswkmf"&&(le.innerHTML=lt),Ie=a(k),j=m(k,"DIV",{class:!0});var F=ne(j);f(K.$$.fragment,F),Ye=a(F),me=m(F,"P",{"data-svelte-h":!0}),v(me)!=="svelte-fji8g2"&&(me.innerHTML=mt),Be=a(F),f(X.$$.fragment,F),He=a(F),f(R.$$.fragment,F),F.forEach(s),k.forEach(s),$e=a(e),f(ee.$$.fragment,e),Fe=a(e),M=m(e,"DIV",{class:!0});var U=ne(M);f(te.$$.fragment,U),Ee=a(U),he=m(U,"P",{"data-svelte-h":!0}),v(he)!=="svelte-axb8n3"&&(he.textContent=ht),Pe=a(U),ue=m(U,"P",{"data-svelte-h":!0}),v(ue)!=="svelte-q52n56"&&(ue.innerHTML=ut),De=a(U),pe=m(U,"P",{"data-svelte-h":!0}),v(pe)!=="svelte-hswkmf"&&(pe.innerHTML=pt),Qe=a(U),J=m(U,"DIV",{class:!0});var Z=ne(J);f(oe.$$.fragment,Z),Oe=a(Z),fe=m(Z,"P",{"data-svelte-h":!0}),v(fe)!=="svelte-g1t8vg"&&(fe.innerHTML=ft),Ae=a(Z),f(N.$$.fragment,Z),Le=a(Z),f(W.$$.fragment,Z),Z.forEach(s),U.forEach(s),Ze=a(e),f(se.$$.fragment,e),qe=a(e),ge=m(e,"P",{}),ne(ge).forEach(s),this.h()},h(){V(o,"name","hf:doc:metadata"),V(o,"content",jt),V(q,"class","flex flex-wrap space-x-1"),V(x,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),V(j,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),V(w,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),V(J,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),V(M,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,t){c(document.head,o),i(e,u,t),i(e,r,t),i(e,h,t),i(e,b,t),i(e,C,t),g(G,e,t),i(e,ye,t),i(e,q,t),i(e,ve,t),g(I,e,t),i(e,be,t),i(e,Y,t),i(e,we,t),g(B,e,t),i(e,Me,t),i(e,H,t),i(e,xe,t),g(E,e,t),i(e,ke,t),i(e,P,t),i(e,Ue,t),i(e,D,t),i(e,Se,t),g(Q,e,t),i(e,Ce,t),i(e,x,t),g(O,x,null),c(x,Xe),c(x,re),c(x,Re),c(x,ie),c(x,Ne),g(z,x,null),i(e,je,t),g(A,e,t),i(e,Je,t),i(e,w,t),g(L,w,null),c(w,We),c(w,de),c(w,Ve),c(w,ce),c(w,Ge),c(w,le),c(w,Ie),c(w,j),g(K,j,null),c(j,Ye),c(j,me),c(j,Be),g(X,j,null),c(j,He),g(R,j,null),i(e,$e,t),g(ee,e,t),i(e,Fe,t),i(e,M,t),g(te,M,null),c(M,Ee),c(M,he),c(M,Pe),c(M,ue),c(M,De),c(M,pe),c(M,Qe),c(M,J),g(oe,J,null),c(J,Oe),c(J,fe),c(J,Ae),g(N,J,null),c(J,Le),g(W,J,null),i(e,Ze,t),g(se,e,t),i(e,qe,t),i(e,ge,t),ze=!0},p(e,[t]){const $={};t&2&&($.$$scope={dirty:t,ctx:e}),z.$set($);const k={};t&2&&(k.$$scope={dirty:t,ctx:e}),X.$set(k);const F={};t&2&&(F.$$scope={dirty:t,ctx:e}),R.$set(F);const U={};t&2&&(U.$$scope={dirty:t,ctx:e}),N.$set(U);const Z={};t&2&&(Z.$$scope={dirty:t,ctx:e}),W.$set(Z)},i(e){ze||(_(G.$$.fragment,e),_(I.$$.fragment,e),_(B.$$.fragment,e),_(E.$$.fragment,e),_(Q.$$.fragment,e),_(O.$$.fragment,e),_(z.$$.fragment,e),_(A.$$.fragment,e),_(L.$$.fragment,e),_(K.$$.fragment,e),_(X.$$.fragment,e),_(R.$$.fragment,e),_(ee.$$.fragment,e),_(te.$$.fragment,e),_(oe.$$.fragment,e),_(N.$$.fragment,e),_(W.$$.fragment,e),_(se.$$.fragment,e),ze=!0)},o(e){T(G.$$.fragment,e),T(I.$$.fragment,e),T(B.$$.fragment,e),T(E.$$.fragment,e),T(Q.$$.fragment,e),T(O.$$.fragment,e),T(z.$$.fragment,e),T(A.$$.fragment,e),T(L.$$.fragment,e),T(K.$$.fragment,e),T(X.$$.fragment,e),T(R.$$.fragment,e),T(ee.$$.fragment,e),T(te.$$.fragment,e),T(oe.$$.fragment,e),T(N.$$.fragment,e),T(W.$$.fragment,e),T(se.$$.fragment,e),ze=!1},d(e){e&&(s(u),s(r),s(h),s(b),s(C),s(ye),s(q),s(ve),s(be),s(Y),s(we),s(Me),s(H),s(xe),s(ke),s(P),s(Ue),s(D),s(Se),s(Ce),s(x),s(je),s(Je),s(w),s($e),s(Fe),s(M),s(Ze),s(qe),s(ge)),s(o),y(G,e),y(I,e),y(B,e),y(E,e),y(Q,e),y(O),y(z),y(A,e),y(L),y(K),y(X),y(R),y(ee,e),y(te),y(oe),y(N),y(W),y(se,e)}}}const jt='{"title":"Time Series Transformer","local":"time-series-transformer","sections":[{"title":"Overview","local":"overview","sections":[],"depth":2},{"title":"Usage tips","local":"usage-tips","sections":[],"depth":2},{"title":"Resources","local":"resources","sections":[],"depth":2},{"title":"TimeSeriesTransformerConfig","local":"transformers.TimeSeriesTransformerConfig","sections":[],"depth":2},{"title":"TimeSeriesTransformerModel","local":"transformers.TimeSeriesTransformerModel","sections":[],"depth":2},{"title":"TimeSeriesTransformerForPrediction","local":"transformers.TimeSeriesTransformerForPrediction","sections":[],"depth":2}],"depth":1}';function Jt(S){return Tt(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class Nt extends yt{constructor(o){super(),vt(this,o,Jt,Ct,_t,{})}}export{Nt as component};
