import{s as xt,o as It,n as be}from"../chunks/scheduler.18a86fab.js";import{S as kt,i as jt,g as l,s as r,r as f,A as Ct,h as m,f as o,c as i,j as re,x as p,u as g,k as V,y as h,a as s,v as _,d as y,t as v,w as b}from"../chunks/index.98837b22.js";import{T as Mt}from"../chunks/Tip.77304350.js";import{D as ve}from"../chunks/Docstring.a1ef7999.js";import{C as at}from"../chunks/CodeBlock.8d0c2e8a.js";import{E as nt}from"../chunks/ExampleCodeBlock.8c3ee1f9.js";import{H as _e,E as Jt}from"../chunks/getInferenceSnippets.06c2775f.js";function qt(C){let n,T="Example:",c,d,u;return d=new at({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEluZm9ybWVyQ29uZmlnJTJDJTIwSW5mb3JtZXJNb2RlbCUwQSUwQSUyMyUyMEluaXRpYWxpemluZyUyMGFuJTIwSW5mb3JtZXIlMjBjb25maWd1cmF0aW9uJTIwd2l0aCUyMDEyJTIwdGltZSUyMHN0ZXBzJTIwZm9yJTIwcHJlZGljdGlvbiUwQWNvbmZpZ3VyYXRpb24lMjAlM0QlMjBJbmZvcm1lckNvbmZpZyhwcmVkaWN0aW9uX2xlbmd0aCUzRDEyKSUwQSUwQSUyMyUyMFJhbmRvbWx5JTIwaW5pdGlhbGl6aW5nJTIwYSUyMG1vZGVsJTIwKHdpdGglMjByYW5kb20lMjB3ZWlnaHRzKSUyMGZyb20lMjB0aGUlMjBjb25maWd1cmF0aW9uJTBBbW9kZWwlMjAlM0QlMjBJbmZvcm1lck1vZGVsKGNvbmZpZ3VyYXRpb24pJTBBJTBBJTIzJTIwQWNjZXNzaW5nJTIwdGhlJTIwbW9kZWwlMjBjb25maWd1cmF0aW9uJTBBY29uZmlndXJhdGlvbiUyMCUzRCUyMG1vZGVsLmNvbmZpZw==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> InformerConfig, InformerModel

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing an Informer configuration with 12 time steps for prediction</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = InformerConfig(prediction_length=<span class="hljs-number">12</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Randomly initializing a model (with random weights) from the configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = InformerModel(configuration)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Accessing the model configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = model.config`,wrap:!1}}),{c(){n=l("p"),n.textContent=T,c=r(),f(d.$$.fragment)},l(a){n=m(a,"P",{"data-svelte-h":!0}),p(n)!=="svelte-11lpom8"&&(n.textContent=T),c=i(a),g(d.$$.fragment,a)},m(a,w){s(a,n,w),s(a,c,w),_(d,a,w),u=!0},p:be,i(a){u||(y(d.$$.fragment,a),u=!0)},o(a){v(d.$$.fragment,a),u=!1},d(a){a&&(o(n),o(c)),b(d,a)}}}function $t(C){let n,T=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){n=l("p"),n.innerHTML=T},l(c){n=m(c,"P",{"data-svelte-h":!0}),p(n)!=="svelte-fincs2"&&(n.innerHTML=T)},m(c,d){s(c,n,d)},p:be,d(c){c&&o(n)}}}function Ft(C){let n,T="Examples:",c,d,u;return d=new at({props:{code:"ZnJvbSUyMGh1Z2dpbmdmYWNlX2h1YiUyMGltcG9ydCUyMGhmX2h1Yl9kb3dubG9hZCUwQWltcG9ydCUyMHRvcmNoJTBBZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEluZm9ybWVyTW9kZWwlMEElMEFmaWxlJTIwJTNEJTIwaGZfaHViX2Rvd25sb2FkKCUwQSUyMCUyMCUyMCUyMHJlcG9faWQlM0QlMjJoZi1pbnRlcm5hbC10ZXN0aW5nJTJGdG91cmlzbS1tb250aGx5LWJhdGNoJTIyJTJDJTIwZmlsZW5hbWUlM0QlMjJ0cmFpbi1iYXRjaC5wdCUyMiUyQyUyMHJlcG9fdHlwZSUzRCUyMmRhdGFzZXQlMjIlMEEpJTBBYmF0Y2glMjAlM0QlMjB0b3JjaC5sb2FkKGZpbGUpJTBBJTBBbW9kZWwlMjAlM0QlMjBJbmZvcm1lck1vZGVsLmZyb21fcHJldHJhaW5lZCglMjJodWdnaW5nZmFjZSUyRmluZm9ybWVyLXRvdXJpc20tbW9udGhseSUyMiklMEElMEElMjMlMjBkdXJpbmclMjB0cmFpbmluZyUyQyUyMG9uZSUyMHByb3ZpZGVzJTIwYm90aCUyMHBhc3QlMjBhbmQlMjBmdXR1cmUlMjB2YWx1ZXMlMEElMjMlMjBhcyUyMHdlbGwlMjBhcyUyMHBvc3NpYmxlJTIwYWRkaXRpb25hbCUyMGZlYXR1cmVzJTBBb3V0cHV0cyUyMCUzRCUyMG1vZGVsKCUwQSUyMCUyMCUyMCUyMHBhc3RfdmFsdWVzJTNEYmF0Y2glNUIlMjJwYXN0X3ZhbHVlcyUyMiU1RCUyQyUwQSUyMCUyMCUyMCUyMHBhc3RfdGltZV9mZWF0dXJlcyUzRGJhdGNoJTVCJTIycGFzdF90aW1lX2ZlYXR1cmVzJTIyJTVEJTJDJTBBJTIwJTIwJTIwJTIwcGFzdF9vYnNlcnZlZF9tYXNrJTNEYmF0Y2glNUIlMjJwYXN0X29ic2VydmVkX21hc2slMjIlNUQlMkMlMEElMjAlMjAlMjAlMjBzdGF0aWNfY2F0ZWdvcmljYWxfZmVhdHVyZXMlM0RiYXRjaCU1QiUyMnN0YXRpY19jYXRlZ29yaWNhbF9mZWF0dXJlcyUyMiU1RCUyQyUwQSUyMCUyMCUyMCUyMHN0YXRpY19yZWFsX2ZlYXR1cmVzJTNEYmF0Y2glNUIlMjJzdGF0aWNfcmVhbF9mZWF0dXJlcyUyMiU1RCUyQyUwQSUyMCUyMCUyMCUyMGZ1dHVyZV92YWx1ZXMlM0RiYXRjaCU1QiUyMmZ1dHVyZV92YWx1ZXMlMjIlNUQlMkMlMEElMjAlMjAlMjAlMjBmdXR1cmVfdGltZV9mZWF0dXJlcyUzRGJhdGNoJTVCJTIyZnV0dXJlX3RpbWVfZmVhdHVyZXMlMjIlNUQlMkMlMEEpJTBBJTBBbGFzdF9oaWRkZW5fc3RhdGUlMjAlM0QlMjBvdXRwdXRzLmxhc3RfaGlkZGVuX3N0YXRl",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> huggingface_hub <span class="hljs-keyword">import</span> hf_hub_download
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> InformerModel

<span class="hljs-meta">&gt;&gt;&gt; </span>file = hf_hub_download(
<span class="hljs-meta">... </span>    repo_id=<span class="hljs-string">&quot;hf-internal-testing/tourism-monthly-batch&quot;</span>, filename=<span class="hljs-string">&quot;train-batch.pt&quot;</span>, repo_type=<span class="hljs-string">&quot;dataset&quot;</span>
<span class="hljs-meta">... </span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>batch = torch.load(file)

<span class="hljs-meta">&gt;&gt;&gt; </span>model = InformerModel.from_pretrained(<span class="hljs-string">&quot;huggingface/informer-tourism-monthly&quot;</span>)

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

<span class="hljs-meta">&gt;&gt;&gt; </span>last_hidden_state = outputs.last_hidden_state`,wrap:!1}}),{c(){n=l("p"),n.textContent=T,c=r(),f(d.$$.fragment)},l(a){n=m(a,"P",{"data-svelte-h":!0}),p(n)!=="svelte-kvfsh7"&&(n.textContent=T),c=i(a),g(d.$$.fragment,a)},m(a,w){s(a,n,w),s(a,c,w),_(d,a,w),u=!0},p:be,i(a){u||(y(d.$$.fragment,a),u=!0)},o(a){v(d.$$.fragment,a),u=!1},d(a){a&&(o(n),o(c)),b(d,a)}}}function Ut(C){let n,T=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){n=l("p"),n.innerHTML=T},l(c){n=m(c,"P",{"data-svelte-h":!0}),p(n)!=="svelte-fincs2"&&(n.innerHTML=T)},m(c,d){s(c,n,d)},p:be,d(c){c&&o(n)}}}function Zt(C){let n,T="Examples:",c,d,u;return d=new at({props:{code:"ZnJvbSUyMGh1Z2dpbmdmYWNlX2h1YiUyMGltcG9ydCUyMGhmX2h1Yl9kb3dubG9hZCUwQWltcG9ydCUyMHRvcmNoJTBBZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEluZm9ybWVyRm9yUHJlZGljdGlvbiUwQSUwQWZpbGUlMjAlM0QlMjBoZl9odWJfZG93bmxvYWQoJTBBJTIwJTIwJTIwJTIwcmVwb19pZCUzRCUyMmhmLWludGVybmFsLXRlc3RpbmclMkZ0b3VyaXNtLW1vbnRobHktYmF0Y2glMjIlMkMlMjBmaWxlbmFtZSUzRCUyMnRyYWluLWJhdGNoLnB0JTIyJTJDJTIwcmVwb190eXBlJTNEJTIyZGF0YXNldCUyMiUwQSklMEFiYXRjaCUyMCUzRCUyMHRvcmNoLmxvYWQoZmlsZSklMEElMEFtb2RlbCUyMCUzRCUyMEluZm9ybWVyRm9yUHJlZGljdGlvbi5mcm9tX3ByZXRyYWluZWQoJTBBJTIwJTIwJTIwJTIwJTIyaHVnZ2luZ2ZhY2UlMkZpbmZvcm1lci10b3VyaXNtLW1vbnRobHklMjIlMEEpJTBBJTBBJTIzJTIwZHVyaW5nJTIwdHJhaW5pbmclMkMlMjBvbmUlMjBwcm92aWRlcyUyMGJvdGglMjBwYXN0JTIwYW5kJTIwZnV0dXJlJTIwdmFsdWVzJTBBJTIzJTIwYXMlMjB3ZWxsJTIwYXMlMjBwb3NzaWJsZSUyMGFkZGl0aW9uYWwlMjBmZWF0dXJlcyUwQW91dHB1dHMlMjAlM0QlMjBtb2RlbCglMEElMjAlMjAlMjAlMjBwYXN0X3ZhbHVlcyUzRGJhdGNoJTVCJTIycGFzdF92YWx1ZXMlMjIlNUQlMkMlMEElMjAlMjAlMjAlMjBwYXN0X3RpbWVfZmVhdHVyZXMlM0RiYXRjaCU1QiUyMnBhc3RfdGltZV9mZWF0dXJlcyUyMiU1RCUyQyUwQSUyMCUyMCUyMCUyMHBhc3Rfb2JzZXJ2ZWRfbWFzayUzRGJhdGNoJTVCJTIycGFzdF9vYnNlcnZlZF9tYXNrJTIyJTVEJTJDJTBBJTIwJTIwJTIwJTIwc3RhdGljX2NhdGVnb3JpY2FsX2ZlYXR1cmVzJTNEYmF0Y2glNUIlMjJzdGF0aWNfY2F0ZWdvcmljYWxfZmVhdHVyZXMlMjIlNUQlMkMlMEElMjAlMjAlMjAlMjBzdGF0aWNfcmVhbF9mZWF0dXJlcyUzRGJhdGNoJTVCJTIyc3RhdGljX3JlYWxfZmVhdHVyZXMlMjIlNUQlMkMlMEElMjAlMjAlMjAlMjBmdXR1cmVfdmFsdWVzJTNEYmF0Y2glNUIlMjJmdXR1cmVfdmFsdWVzJTIyJTVEJTJDJTBBJTIwJTIwJTIwJTIwZnV0dXJlX3RpbWVfZmVhdHVyZXMlM0RiYXRjaCU1QiUyMmZ1dHVyZV90aW1lX2ZlYXR1cmVzJTIyJTVEJTJDJTBBKSUwQSUwQWxvc3MlMjAlM0QlMjBvdXRwdXRzLmxvc3MlMEFsb3NzLmJhY2t3YXJkKCklMEElMEElMjMlMjBkdXJpbmclMjBpbmZlcmVuY2UlMkMlMjBvbmUlMjBvbmx5JTIwcHJvdmlkZXMlMjBwYXN0JTIwdmFsdWVzJTBBJTIzJTIwYXMlMjB3ZWxsJTIwYXMlMjBwb3NzaWJsZSUyMGFkZGl0aW9uYWwlMjBmZWF0dXJlcyUwQSUyMyUyMHRoZSUyMG1vZGVsJTIwYXV0b3JlZ3Jlc3NpdmVseSUyMGdlbmVyYXRlcyUyMGZ1dHVyZSUyMHZhbHVlcyUwQW91dHB1dHMlMjAlM0QlMjBtb2RlbC5nZW5lcmF0ZSglMEElMjAlMjAlMjAlMjBwYXN0X3ZhbHVlcyUzRGJhdGNoJTVCJTIycGFzdF92YWx1ZXMlMjIlNUQlMkMlMEElMjAlMjAlMjAlMjBwYXN0X3RpbWVfZmVhdHVyZXMlM0RiYXRjaCU1QiUyMnBhc3RfdGltZV9mZWF0dXJlcyUyMiU1RCUyQyUwQSUyMCUyMCUyMCUyMHBhc3Rfb2JzZXJ2ZWRfbWFzayUzRGJhdGNoJTVCJTIycGFzdF9vYnNlcnZlZF9tYXNrJTIyJTVEJTJDJTBBJTIwJTIwJTIwJTIwc3RhdGljX2NhdGVnb3JpY2FsX2ZlYXR1cmVzJTNEYmF0Y2glNUIlMjJzdGF0aWNfY2F0ZWdvcmljYWxfZmVhdHVyZXMlMjIlNUQlMkMlMEElMjAlMjAlMjAlMjBzdGF0aWNfcmVhbF9mZWF0dXJlcyUzRGJhdGNoJTVCJTIyc3RhdGljX3JlYWxfZmVhdHVyZXMlMjIlNUQlMkMlMEElMjAlMjAlMjAlMjBmdXR1cmVfdGltZV9mZWF0dXJlcyUzRGJhdGNoJTVCJTIyZnV0dXJlX3RpbWVfZmVhdHVyZXMlMjIlNUQlMkMlMEEpJTBBJTBBbWVhbl9wcmVkaWN0aW9uJTIwJTNEJTIwb3V0cHV0cy5zZXF1ZW5jZXMubWVhbihkaW0lM0QxKQ==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> huggingface_hub <span class="hljs-keyword">import</span> hf_hub_download
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> InformerForPrediction

<span class="hljs-meta">&gt;&gt;&gt; </span>file = hf_hub_download(
<span class="hljs-meta">... </span>    repo_id=<span class="hljs-string">&quot;hf-internal-testing/tourism-monthly-batch&quot;</span>, filename=<span class="hljs-string">&quot;train-batch.pt&quot;</span>, repo_type=<span class="hljs-string">&quot;dataset&quot;</span>
<span class="hljs-meta">... </span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>batch = torch.load(file)

<span class="hljs-meta">&gt;&gt;&gt; </span>model = InformerForPrediction.from_pretrained(
<span class="hljs-meta">... </span>    <span class="hljs-string">&quot;huggingface/informer-tourism-monthly&quot;</span>
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

<span class="hljs-meta">&gt;&gt;&gt; </span>mean_prediction = outputs.sequences.mean(dim=<span class="hljs-number">1</span>)`,wrap:!1}}),{c(){n=l("p"),n.textContent=T,c=r(),f(d.$$.fragment)},l(a){n=m(a,"P",{"data-svelte-h":!0}),p(n)!=="svelte-kvfsh7"&&(n.textContent=T),c=i(a),g(d.$$.fragment,a)},m(a,w){s(a,n,w),s(a,c,w),_(d,a,w),u=!0},p:be,i(a){u||(y(d.$$.fragment,a),u=!0)},o(a){v(d.$$.fragment,a),u=!1},d(a){a&&(o(n),o(c)),b(d,a)}}}function zt(C){let n,T,c,d,u,a="<em>This model was released on 2020-12-14 and added to Hugging Face Transformers on 2023-03-08.</em>",w,R,Te,Z,st='<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/>',we,G,Me,S,rt='The Informer model was proposed in <a href="https://huggingface.co/papers/2012.07436" rel="nofollow">Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting</a> by Haoyi Zhou, Shanghang Zhang, Jieqi Peng, Shuai Zhang, Jianxin Li, Hui Xiong, and Wancai Zhang.',xe,H,it="This method introduces a Probabilistic Attention mechanism to select the “active” queries rather than the “lazy” queries and provides a sparse Transformer thus mitigating the quadratic compute and memory requirements of vanilla attention.",Ie,Y,ct="The abstract from the paper is the following:",ke,E,dt="<em>Many real-world applications require the prediction of long sequence time-series, such as electricity consumption planning. Long sequence time-series forecasting (LSTF) demands a high prediction capacity of the model, which is the ability to capture precise long-range dependency coupling between output and input efficiently. Recent studies have shown the potential of Transformer to increase the prediction capacity. However, there are several severe issues with Transformer that prevent it from being directly applicable to LSTF, including quadratic time complexity, high memory usage, and inherent limitation of the encoder-decoder architecture. To address these issues, we design an efficient transformer-based model for LSTF, named Informer, with three distinctive characteristics: (i) a ProbSparse self-attention mechanism, which achieves O(L logL) in time complexity and memory usage, and has comparable performance on sequences’ dependency alignment. (ii) the self-attention distilling highlights dominating attention by halving cascading layer input, and efficiently handles extreme long input sequences. (iii) the generative style decoder, while conceptually simple, predicts the long time-series sequences at one forward operation rather than a step-by-step way, which drastically improves the inference speed of long-sequence predictions. Extensive experiments on four large-scale datasets demonstrate that Informer significantly outperforms existing methods and provides a new solution to the LSTF problem.</em>",je,P,lt=`This model was contributed by <a href="https://huggingface.co/elisim" rel="nofollow">elisim</a> and <a href="https://huggingface.co/kashif" rel="nofollow">kashif</a>.
The original code can be found <a href="https://github.com/zhouhaoyi/Informer2020" rel="nofollow">here</a>.`,Ce,D,Je,L,mt="A list of official Hugging Face and community (indicated by 🌎) resources to help you get started. If you’re interested in submitting a resource to be included here, please feel free to open a Pull Request and we’ll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.",qe,A,ht='<li>Check out the Informer blog-post in HuggingFace blog: <a href="https://huggingface.co/blog/informer" rel="nofollow">Multivariate Probabilistic Time Series Forecasting with Informer</a></li>',$e,Q,Fe,I,O,Ve,ie,pt=`This is the configuration class to store the configuration of an <a href="/docs/transformers/v4.56.2/en/model_doc/informer#transformers.InformerModel">InformerModel</a>. It is used to instantiate an
Informer model according to the specified arguments, defining the model architecture. Instantiating a configuration
with the defaults will yield a similar configuration to that of the Informer
<a href="https://huggingface.co/huggingface/informer-tourism-monthly" rel="nofollow">huggingface/informer-tourism-monthly</a> architecture.`,Re,ce,ut=`Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> can be used to control the model outputs. Read the
documentation from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> for more information.`,Ge,z,Ue,K,Ze,M,ee,Se,de,ft="The bare Informer Model outputting raw hidden-states without any specific head on top.",He,le,gt=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Ye,me,_t=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Ee,J,te,Pe,he,yt='The <a href="/docs/transformers/v4.56.2/en/model_doc/informer#transformers.InformerModel">InformerModel</a> forward method, overrides the <code>__call__</code> special method.',De,W,Le,N,ze,oe,We,x,ne,Ae,pe,vt="The Informer Model with a distribution head on top for time-series forecasting.",Qe,ue,bt=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Oe,fe,Tt=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Ke,q,ae,et,ge,wt='The <a href="/docs/transformers/v4.56.2/en/model_doc/informer#transformers.InformerForPrediction">InformerForPrediction</a> forward method, overrides the <code>__call__</code> special method.',tt,X,ot,B,Ne,se,Xe,ye,Be;return R=new _e({props:{title:"Informer",local:"informer",headingTag:"h1"}}),G=new _e({props:{title:"Overview",local:"overview",headingTag:"h2"}}),D=new _e({props:{title:"Resources",local:"resources",headingTag:"h2"}}),Q=new _e({props:{title:"InformerConfig",local:"transformers.InformerConfig",headingTag:"h2"}}),O=new ve({props:{name:"class transformers.InformerConfig",anchor:"transformers.InformerConfig",parameters:[{name:"prediction_length",val:": typing.Optional[int] = None"},{name:"context_length",val:": typing.Optional[int] = None"},{name:"distribution_output",val:": str = 'student_t'"},{name:"loss",val:": str = 'nll'"},{name:"input_size",val:": int = 1"},{name:"lags_sequence",val:": typing.Optional[list[int]] = None"},{name:"scaling",val:": typing.Union[str, bool, NoneType] = 'mean'"},{name:"num_dynamic_real_features",val:": int = 0"},{name:"num_static_real_features",val:": int = 0"},{name:"num_static_categorical_features",val:": int = 0"},{name:"num_time_features",val:": int = 0"},{name:"cardinality",val:": typing.Optional[list[int]] = None"},{name:"embedding_dimension",val:": typing.Optional[list[int]] = None"},{name:"d_model",val:": int = 64"},{name:"encoder_ffn_dim",val:": int = 32"},{name:"decoder_ffn_dim",val:": int = 32"},{name:"encoder_attention_heads",val:": int = 2"},{name:"decoder_attention_heads",val:": int = 2"},{name:"encoder_layers",val:": int = 2"},{name:"decoder_layers",val:": int = 2"},{name:"is_encoder_decoder",val:": bool = True"},{name:"activation_function",val:": str = 'gelu'"},{name:"dropout",val:": float = 0.05"},{name:"encoder_layerdrop",val:": float = 0.1"},{name:"decoder_layerdrop",val:": float = 0.1"},{name:"attention_dropout",val:": float = 0.1"},{name:"activation_dropout",val:": float = 0.1"},{name:"num_parallel_samples",val:": int = 100"},{name:"init_std",val:": float = 0.02"},{name:"use_cache",val:" = True"},{name:"attention_type",val:": str = 'prob'"},{name:"sampling_factor",val:": int = 5"},{name:"distil",val:": bool = True"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.InformerConfig.prediction_length",description:`<strong>prediction_length</strong> (<code>int</code>) &#x2014;
The prediction length for the decoder. In other words, the prediction horizon of the model. This value is
typically dictated by the dataset and we recommend to set it appropriately.`,name:"prediction_length"},{anchor:"transformers.InformerConfig.context_length",description:`<strong>context_length</strong> (<code>int</code>, <em>optional</em>, defaults to <code>prediction_length</code>) &#x2014;
The context length for the encoder. If <code>None</code>, the context length will be the same as the
<code>prediction_length</code>.`,name:"context_length"},{anchor:"transformers.InformerConfig.distribution_output",description:`<strong>distribution_output</strong> (<code>string</code>, <em>optional</em>, defaults to <code>&quot;student_t&quot;</code>) &#x2014;
The distribution emission head for the model. Could be either &#x201C;student_t&#x201D;, &#x201C;normal&#x201D; or &#x201C;negative_binomial&#x201D;.`,name:"distribution_output"},{anchor:"transformers.InformerConfig.loss",description:`<strong>loss</strong> (<code>string</code>, <em>optional</em>, defaults to <code>&quot;nll&quot;</code>) &#x2014;
The loss function for the model corresponding to the <code>distribution_output</code> head. For parametric
distributions it is the negative log likelihood (nll) - which currently is the only supported one.`,name:"loss"},{anchor:"transformers.InformerConfig.input_size",description:`<strong>input_size</strong> (<code>int</code>, <em>optional</em>, defaults to 1) &#x2014;
The size of the target variable which by default is 1 for univariate targets. Would be &gt; 1 in case of
multivariate targets.`,name:"input_size"},{anchor:"transformers.InformerConfig.scaling",description:`<strong>scaling</strong> (<code>string</code> or <code>bool</code>, <em>optional</em> defaults to <code>&quot;mean&quot;</code>) &#x2014;
Whether to scale the input targets via &#x201C;mean&#x201D; scaler, &#x201C;std&#x201D; scaler or no scaler if <code>None</code>. If <code>True</code>, the
scaler is set to &#x201C;mean&#x201D;.`,name:"scaling"},{anchor:"transformers.InformerConfig.lags_sequence",description:`<strong>lags_sequence</strong> (<code>list[int]</code>, <em>optional</em>, defaults to <code>[1, 2, 3, 4, 5, 6, 7]</code>) &#x2014;
The lags of the input time series as covariates often dictated by the frequency of the data. Default is
<code>[1, 2, 3, 4, 5, 6, 7]</code> but we recommend to change it based on the dataset appropriately.`,name:"lags_sequence"},{anchor:"transformers.InformerConfig.num_time_features",description:`<strong>num_time_features</strong> (<code>int</code>, <em>optional</em>, defaults to 0) &#x2014;
The number of time features in the input time series.`,name:"num_time_features"},{anchor:"transformers.InformerConfig.num_dynamic_real_features",description:`<strong>num_dynamic_real_features</strong> (<code>int</code>, <em>optional</em>, defaults to 0) &#x2014;
The number of dynamic real valued features.`,name:"num_dynamic_real_features"},{anchor:"transformers.InformerConfig.num_static_categorical_features",description:`<strong>num_static_categorical_features</strong> (<code>int</code>, <em>optional</em>, defaults to 0) &#x2014;
The number of static categorical features.`,name:"num_static_categorical_features"},{anchor:"transformers.InformerConfig.num_static_real_features",description:`<strong>num_static_real_features</strong> (<code>int</code>, <em>optional</em>, defaults to 0) &#x2014;
The number of static real valued features.`,name:"num_static_real_features"},{anchor:"transformers.InformerConfig.cardinality",description:`<strong>cardinality</strong> (<code>list[int]</code>, <em>optional</em>) &#x2014;
The cardinality (number of different values) for each of the static categorical features. Should be a list
of integers, having the same length as <code>num_static_categorical_features</code>. Cannot be <code>None</code> if
<code>num_static_categorical_features</code> is &gt; 0.`,name:"cardinality"},{anchor:"transformers.InformerConfig.embedding_dimension",description:`<strong>embedding_dimension</strong> (<code>list[int]</code>, <em>optional</em>) &#x2014;
The dimension of the embedding for each of the static categorical features. Should be a list of integers,
having the same length as <code>num_static_categorical_features</code>. Cannot be <code>None</code> if
<code>num_static_categorical_features</code> is &gt; 0.`,name:"embedding_dimension"},{anchor:"transformers.InformerConfig.d_model",description:`<strong>d_model</strong> (<code>int</code>, <em>optional</em>, defaults to 64) &#x2014;
Dimensionality of the transformer layers.`,name:"d_model"},{anchor:"transformers.InformerConfig.encoder_layers",description:`<strong>encoder_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 2) &#x2014;
Number of encoder layers.`,name:"encoder_layers"},{anchor:"transformers.InformerConfig.decoder_layers",description:`<strong>decoder_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 2) &#x2014;
Number of decoder layers.`,name:"decoder_layers"},{anchor:"transformers.InformerConfig.encoder_attention_heads",description:`<strong>encoder_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 2) &#x2014;
Number of attention heads for each attention layer in the Transformer encoder.`,name:"encoder_attention_heads"},{anchor:"transformers.InformerConfig.decoder_attention_heads",description:`<strong>decoder_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 2) &#x2014;
Number of attention heads for each attention layer in the Transformer decoder.`,name:"decoder_attention_heads"},{anchor:"transformers.InformerConfig.encoder_ffn_dim",description:`<strong>encoder_ffn_dim</strong> (<code>int</code>, <em>optional</em>, defaults to 32) &#x2014;
Dimension of the &#x201C;intermediate&#x201D; (often named feed-forward) layer in encoder.`,name:"encoder_ffn_dim"},{anchor:"transformers.InformerConfig.decoder_ffn_dim",description:`<strong>decoder_ffn_dim</strong> (<code>int</code>, <em>optional</em>, defaults to 32) &#x2014;
Dimension of the &#x201C;intermediate&#x201D; (often named feed-forward) layer in decoder.`,name:"decoder_ffn_dim"},{anchor:"transformers.InformerConfig.activation_function",description:`<strong>activation_function</strong> (<code>str</code> or <code>function</code>, <em>optional</em>, defaults to <code>&quot;gelu&quot;</code>) &#x2014;
The non-linear activation function (function or string) in the encoder and decoder. If string, <code>&quot;gelu&quot;</code> and
<code>&quot;relu&quot;</code> are supported.`,name:"activation_function"},{anchor:"transformers.InformerConfig.dropout",description:`<strong>dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout probability for all fully connected layers in the encoder, and decoder.`,name:"dropout"},{anchor:"transformers.InformerConfig.encoder_layerdrop",description:`<strong>encoder_layerdrop</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout probability for the attention and fully connected layers for each encoder layer.`,name:"encoder_layerdrop"},{anchor:"transformers.InformerConfig.decoder_layerdrop",description:`<strong>decoder_layerdrop</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout probability for the attention and fully connected layers for each decoder layer.`,name:"decoder_layerdrop"},{anchor:"transformers.InformerConfig.attention_dropout",description:`<strong>attention_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout probability for the attention probabilities.`,name:"attention_dropout"},{anchor:"transformers.InformerConfig.activation_dropout",description:`<strong>activation_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout probability used between the two layers of the feed-forward networks.`,name:"activation_dropout"},{anchor:"transformers.InformerConfig.num_parallel_samples",description:`<strong>num_parallel_samples</strong> (<code>int</code>, <em>optional</em>, defaults to 100) &#x2014;
The number of samples to generate in parallel for each time step of inference.`,name:"num_parallel_samples"},{anchor:"transformers.InformerConfig.init_std",description:`<strong>init_std</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated normal weight initialization distribution.`,name:"init_std"},{anchor:"transformers.InformerConfig.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether to use the past key/values attentions (if applicable to the model) to speed up decoding.`,name:"use_cache"},{anchor:"transformers.InformerConfig.attention_type",description:`<strong>attention_type</strong> (<code>str</code>, <em>optional</em>, defaults to &#x201C;prob&#x201D;) &#x2014;
Attention used in encoder. This can be set to &#x201C;prob&#x201D; (Informer&#x2019;s ProbAttention) or &#x201C;full&#x201D; (vanilla
transformer&#x2019;s canonical self-attention).`,name:"attention_type"},{anchor:"transformers.InformerConfig.sampling_factor",description:`<strong>sampling_factor</strong> (<code>int</code>, <em>optional</em>, defaults to 5) &#x2014;
ProbSparse sampling factor (only makes affect when <code>attention_type</code>=&#x201C;prob&#x201D;). It is used to control the
reduced query matrix (Q_reduce) input length.`,name:"sampling_factor"},{anchor:"transformers.InformerConfig.distil",description:`<strong>distil</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether to use distilling in encoder.`,name:"distil"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/informer/configuration_informer.py#L26"}}),z=new nt({props:{anchor:"transformers.InformerConfig.example",$$slots:{default:[qt]},$$scope:{ctx:C}}}),K=new _e({props:{title:"InformerModel",local:"transformers.InformerModel",headingTag:"h2"}}),ee=new ve({props:{name:"class transformers.InformerModel",anchor:"transformers.InformerModel",parameters:[{name:"config",val:": InformerConfig"}],parametersDescription:[{anchor:"transformers.InformerModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/informer#transformers.InformerConfig">InformerConfig</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/informer/modeling_informer.py#L1323"}}),te=new ve({props:{name:"forward",anchor:"transformers.InformerModel.forward",parameters:[{name:"past_values",val:": Tensor"},{name:"past_time_features",val:": Tensor"},{name:"past_observed_mask",val:": Tensor"},{name:"static_categorical_features",val:": typing.Optional[torch.Tensor] = None"},{name:"static_real_features",val:": typing.Optional[torch.Tensor] = None"},{name:"future_values",val:": typing.Optional[torch.Tensor] = None"},{name:"future_time_features",val:": typing.Optional[torch.Tensor] = None"},{name:"decoder_attention_mask",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"decoder_head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"cross_attn_head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_outputs",val:": typing.Optional[list[torch.FloatTensor]] = None"},{name:"past_key_values",val:": typing.Optional[list[torch.FloatTensor]] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"}],parametersDescription:[{anchor:"transformers.InformerModel.forward.past_values",description:`<strong>past_values</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code> or <code>(batch_size, sequence_length, input_size)</code>) &#x2014;
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
variates in the time series per time step.`,name:"past_values"},{anchor:"transformers.InformerModel.forward.past_time_features",description:`<strong>past_time_features</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, num_features)</code>) &#x2014;
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
<p>The <code>num_features</code> here is equal to <code>config.</code>num_time_features<code>+</code>config.num_dynamic_real_features\`.`,name:"past_time_features"},{anchor:"transformers.InformerModel.forward.past_observed_mask",description:`<strong>past_observed_mask</strong> (<code>torch.BoolTensor</code> of shape <code>(batch_size, sequence_length)</code> or <code>(batch_size, sequence_length, input_size)</code>, <em>optional</em>) &#x2014;
Boolean mask to indicate which <code>past_values</code> were observed and which were missing. Mask values selected in
<code>[0, 1]</code>:</p>
<ul>
<li>1 for values that are <strong>observed</strong>,</li>
<li>0 for values that are <strong>missing</strong> (i.e. NaNs that were replaced by zeros).</li>
</ul>`,name:"past_observed_mask"},{anchor:"transformers.InformerModel.forward.static_categorical_features",description:`<strong>static_categorical_features</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, number of static categorical features)</code>, <em>optional</em>) &#x2014;
Optional static categorical features for which the model will learn an embedding, which it will add to the
values of the time series.</p>
<p>Static categorical features are features which have the same value for all time steps (static over time).</p>
<p>A typical example of a static categorical feature is a time series ID.`,name:"static_categorical_features"},{anchor:"transformers.InformerModel.forward.static_real_features",description:`<strong>static_real_features</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, number of static real features)</code>, <em>optional</em>) &#x2014;
Optional static real features which the model will add to the values of the time series.</p>
<p>Static real features are features which have the same value for all time steps (static over time).</p>
<p>A typical example of a static real feature is promotion information.`,name:"static_real_features"},{anchor:"transformers.InformerModel.forward.future_values",description:`<strong>future_values</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, prediction_length)</code> or <code>(batch_size, prediction_length, input_size)</code>, <em>optional</em>) &#x2014;
Future values of the time series, that serve as labels for the model. The <code>future_values</code> is what the
Transformer needs during training to learn to output, given the <code>past_values</code>.</p>
<p>The sequence length here is equal to <code>prediction_length</code>.</p>
<p>See the demo notebook and code snippets for details.</p>
<p>Optionally, during training any missing values need to be replaced with zeros and indicated via the
<code>future_observed_mask</code>.</p>
<p>For multivariate time series, the <code>input_size</code> &gt; 1 dimension is required and corresponds to the number of
variates in the time series per time step.`,name:"future_values"},{anchor:"transformers.InformerModel.forward.future_time_features",description:`<strong>future_time_features</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, prediction_length, num_features)</code>) &#x2014;
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
<p>The <code>num_features</code> here is equal to <code>config.</code>num_time_features<code>+</code>config.num_dynamic_real_features\`.`,name:"future_time_features"},{anchor:"transformers.InformerModel.forward.decoder_attention_mask",description:`<strong>decoder_attention_mask</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on certain token indices. By default, a causal mask will be used, to
make sure the model can only look at previous inputs in order to predict the future.`,name:"decoder_attention_mask"},{anchor:"transformers.InformerModel.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.InformerModel.forward.decoder_head_mask",description:`<strong>decoder_head_mask</strong> (<code>torch.Tensor</code> of shape <code>(decoder_layers, decoder_attention_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the attention modules in the decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"decoder_head_mask"},{anchor:"transformers.InformerModel.forward.cross_attn_head_mask",description:`<strong>cross_attn_head_mask</strong> (<code>torch.Tensor</code> of shape <code>(decoder_layers, decoder_attention_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the cross-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"cross_attn_head_mask"},{anchor:"transformers.InformerModel.forward.encoder_outputs",description:`<strong>encoder_outputs</strong> (<code>tuple(tuple(torch.FloatTensor)</code>, <em>optional</em>) &#x2014;
Tuple consists of <code>last_hidden_state</code>, <code>hidden_states</code> (<em>optional</em>) and <code>attentions</code> (<em>optional</em>)
<code>last_hidden_state</code> of shape <code>(batch_size, sequence_length, hidden_size)</code> (<em>optional</em>) is a sequence of
hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.`,name:"encoder_outputs"},{anchor:"transformers.InformerModel.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>list[torch.FloatTensor]</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.InformerModel.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.InformerModel.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.InformerModel.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.InformerModel.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.InformerModel.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/informer/modeling_informer.py#L1464",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqTSModelOutput"
>transformers.modeling_outputs.Seq2SeqTSModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/informer#transformers.InformerConfig"
>InformerConfig</a>) and inputs.</p>
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
`}}),W=new Mt({props:{$$slots:{default:[$t]},$$scope:{ctx:C}}}),N=new nt({props:{anchor:"transformers.InformerModel.forward.example",$$slots:{default:[Ft]},$$scope:{ctx:C}}}),oe=new _e({props:{title:"InformerForPrediction",local:"transformers.InformerForPrediction",headingTag:"h2"}}),ne=new ve({props:{name:"class transformers.InformerForPrediction",anchor:"transformers.InformerForPrediction",parameters:[{name:"config",val:": InformerConfig"}],parametersDescription:[{anchor:"transformers.InformerForPrediction.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/informer#transformers.InformerConfig">InformerConfig</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/informer/modeling_informer.py#L1716"}}),ae=new ve({props:{name:"forward",anchor:"transformers.InformerForPrediction.forward",parameters:[{name:"past_values",val:": Tensor"},{name:"past_time_features",val:": Tensor"},{name:"past_observed_mask",val:": Tensor"},{name:"static_categorical_features",val:": typing.Optional[torch.Tensor] = None"},{name:"static_real_features",val:": typing.Optional[torch.Tensor] = None"},{name:"future_values",val:": typing.Optional[torch.Tensor] = None"},{name:"future_time_features",val:": typing.Optional[torch.Tensor] = None"},{name:"future_observed_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"decoder_attention_mask",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"decoder_head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"cross_attn_head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_outputs",val:": typing.Optional[list[torch.FloatTensor]] = None"},{name:"past_key_values",val:": typing.Optional[list[torch.FloatTensor]] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"}],parametersDescription:[{anchor:"transformers.InformerForPrediction.forward.past_values",description:`<strong>past_values</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code> or <code>(batch_size, sequence_length, input_size)</code>) &#x2014;
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
variates in the time series per time step.`,name:"past_values"},{anchor:"transformers.InformerForPrediction.forward.past_time_features",description:`<strong>past_time_features</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, num_features)</code>) &#x2014;
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
<p>The <code>num_features</code> here is equal to <code>config.</code>num_time_features<code>+</code>config.num_dynamic_real_features\`.`,name:"past_time_features"},{anchor:"transformers.InformerForPrediction.forward.past_observed_mask",description:`<strong>past_observed_mask</strong> (<code>torch.BoolTensor</code> of shape <code>(batch_size, sequence_length)</code> or <code>(batch_size, sequence_length, input_size)</code>, <em>optional</em>) &#x2014;
Boolean mask to indicate which <code>past_values</code> were observed and which were missing. Mask values selected in
<code>[0, 1]</code>:</p>
<ul>
<li>1 for values that are <strong>observed</strong>,</li>
<li>0 for values that are <strong>missing</strong> (i.e. NaNs that were replaced by zeros).</li>
</ul>`,name:"past_observed_mask"},{anchor:"transformers.InformerForPrediction.forward.static_categorical_features",description:`<strong>static_categorical_features</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, number of static categorical features)</code>, <em>optional</em>) &#x2014;
Optional static categorical features for which the model will learn an embedding, which it will add to the
values of the time series.</p>
<p>Static categorical features are features which have the same value for all time steps (static over time).</p>
<p>A typical example of a static categorical feature is a time series ID.`,name:"static_categorical_features"},{anchor:"transformers.InformerForPrediction.forward.static_real_features",description:`<strong>static_real_features</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, number of static real features)</code>, <em>optional</em>) &#x2014;
Optional static real features which the model will add to the values of the time series.</p>
<p>Static real features are features which have the same value for all time steps (static over time).</p>
<p>A typical example of a static real feature is promotion information.`,name:"static_real_features"},{anchor:"transformers.InformerForPrediction.forward.future_values",description:`<strong>future_values</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, prediction_length)</code> or <code>(batch_size, prediction_length, input_size)</code>, <em>optional</em>) &#x2014;
Future values of the time series, that serve as labels for the model. The <code>future_values</code> is what the
Transformer needs during training to learn to output, given the <code>past_values</code>.</p>
<p>The sequence length here is equal to <code>prediction_length</code>.</p>
<p>See the demo notebook and code snippets for details.</p>
<p>Optionally, during training any missing values need to be replaced with zeros and indicated via the
<code>future_observed_mask</code>.</p>
<p>For multivariate time series, the <code>input_size</code> &gt; 1 dimension is required and corresponds to the number of
variates in the time series per time step.`,name:"future_values"},{anchor:"transformers.InformerForPrediction.forward.future_time_features",description:`<strong>future_time_features</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, prediction_length, num_features)</code>) &#x2014;
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
<p>The <code>num_features</code> here is equal to <code>config.</code>num_time_features<code>+</code>config.num_dynamic_real_features\`.`,name:"future_time_features"},{anchor:"transformers.InformerForPrediction.forward.future_observed_mask",description:`<strong>future_observed_mask</strong> (<code>torch.BoolTensor</code> of shape <code>(batch_size, sequence_length)</code> or <code>(batch_size, sequence_length, input_size)</code>, <em>optional</em>) &#x2014;
Boolean mask to indicate which <code>future_values</code> were observed and which were missing. Mask values selected
in <code>[0, 1]</code>:</p>
<ul>
<li>1 for values that are <strong>observed</strong>,</li>
<li>0 for values that are <strong>missing</strong> (i.e. NaNs that were replaced by zeros).</li>
</ul>
<p>This mask is used to filter out missing values for the final loss calculation.`,name:"future_observed_mask"},{anchor:"transformers.InformerForPrediction.forward.decoder_attention_mask",description:`<strong>decoder_attention_mask</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on certain token indices. By default, a causal mask will be used, to
make sure the model can only look at previous inputs in order to predict the future.`,name:"decoder_attention_mask"},{anchor:"transformers.InformerForPrediction.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.InformerForPrediction.forward.decoder_head_mask",description:`<strong>decoder_head_mask</strong> (<code>torch.Tensor</code> of shape <code>(decoder_layers, decoder_attention_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the attention modules in the decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"decoder_head_mask"},{anchor:"transformers.InformerForPrediction.forward.cross_attn_head_mask",description:`<strong>cross_attn_head_mask</strong> (<code>torch.Tensor</code> of shape <code>(decoder_layers, decoder_attention_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the cross-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"cross_attn_head_mask"},{anchor:"transformers.InformerForPrediction.forward.encoder_outputs",description:`<strong>encoder_outputs</strong> (<code>tuple(tuple(torch.FloatTensor)</code>, <em>optional</em>) &#x2014;
Tuple consists of <code>last_hidden_state</code>, <code>hidden_states</code> (<em>optional</em>) and <code>attentions</code> (<em>optional</em>)
<code>last_hidden_state</code> of shape <code>(batch_size, sequence_length, hidden_size)</code> (<em>optional</em>) is a sequence of
hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.`,name:"encoder_outputs"},{anchor:"transformers.InformerForPrediction.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>list[torch.FloatTensor]</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.InformerForPrediction.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.InformerForPrediction.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.InformerForPrediction.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.InformerForPrediction.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.InformerForPrediction.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/informer/modeling_informer.py#L1757",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqTSModelOutput"
>transformers.modeling_outputs.Seq2SeqTSModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/informer#transformers.InformerConfig"
>InformerConfig</a>) and inputs.</p>
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
`}}),X=new Mt({props:{$$slots:{default:[Ut]},$$scope:{ctx:C}}}),B=new nt({props:{anchor:"transformers.InformerForPrediction.forward.example",$$slots:{default:[Zt]},$$scope:{ctx:C}}}),se=new Jt({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/informer.md"}}),{c(){n=l("meta"),T=r(),c=l("p"),d=r(),u=l("p"),u.innerHTML=a,w=r(),f(R.$$.fragment),Te=r(),Z=l("div"),Z.innerHTML=st,we=r(),f(G.$$.fragment),Me=r(),S=l("p"),S.innerHTML=rt,xe=r(),H=l("p"),H.textContent=it,Ie=r(),Y=l("p"),Y.textContent=ct,ke=r(),E=l("p"),E.innerHTML=dt,je=r(),P=l("p"),P.innerHTML=lt,Ce=r(),f(D.$$.fragment),Je=r(),L=l("p"),L.textContent=mt,qe=r(),A=l("ul"),A.innerHTML=ht,$e=r(),f(Q.$$.fragment),Fe=r(),I=l("div"),f(O.$$.fragment),Ve=r(),ie=l("p"),ie.innerHTML=pt,Re=r(),ce=l("p"),ce.innerHTML=ut,Ge=r(),f(z.$$.fragment),Ue=r(),f(K.$$.fragment),Ze=r(),M=l("div"),f(ee.$$.fragment),Se=r(),de=l("p"),de.textContent=ft,He=r(),le=l("p"),le.innerHTML=gt,Ye=r(),me=l("p"),me.innerHTML=_t,Ee=r(),J=l("div"),f(te.$$.fragment),Pe=r(),he=l("p"),he.innerHTML=yt,De=r(),f(W.$$.fragment),Le=r(),f(N.$$.fragment),ze=r(),f(oe.$$.fragment),We=r(),x=l("div"),f(ne.$$.fragment),Ae=r(),pe=l("p"),pe.textContent=vt,Qe=r(),ue=l("p"),ue.innerHTML=bt,Oe=r(),fe=l("p"),fe.innerHTML=Tt,Ke=r(),q=l("div"),f(ae.$$.fragment),et=r(),ge=l("p"),ge.innerHTML=wt,tt=r(),f(X.$$.fragment),ot=r(),f(B.$$.fragment),Ne=r(),f(se.$$.fragment),Xe=r(),ye=l("p"),this.h()},l(e){const t=Ct("svelte-u9bgzb",document.head);n=m(t,"META",{name:!0,content:!0}),t.forEach(o),T=i(e),c=m(e,"P",{}),re(c).forEach(o),d=i(e),u=m(e,"P",{"data-svelte-h":!0}),p(u)!=="svelte-e9u0z0"&&(u.innerHTML=a),w=i(e),g(R.$$.fragment,e),Te=i(e),Z=m(e,"DIV",{class:!0,"data-svelte-h":!0}),p(Z)!=="svelte-13t8s2t"&&(Z.innerHTML=st),we=i(e),g(G.$$.fragment,e),Me=i(e),S=m(e,"P",{"data-svelte-h":!0}),p(S)!=="svelte-wzphmc"&&(S.innerHTML=rt),xe=i(e),H=m(e,"P",{"data-svelte-h":!0}),p(H)!=="svelte-1dm22a4"&&(H.textContent=it),Ie=i(e),Y=m(e,"P",{"data-svelte-h":!0}),p(Y)!=="svelte-vfdo9a"&&(Y.textContent=ct),ke=i(e),E=m(e,"P",{"data-svelte-h":!0}),p(E)!=="svelte-halvao"&&(E.innerHTML=dt),je=i(e),P=m(e,"P",{"data-svelte-h":!0}),p(P)!=="svelte-1sw49wp"&&(P.innerHTML=lt),Ce=i(e),g(D.$$.fragment,e),Je=i(e),L=m(e,"P",{"data-svelte-h":!0}),p(L)!=="svelte-1e7xzkp"&&(L.textContent=mt),qe=i(e),A=m(e,"UL",{"data-svelte-h":!0}),p(A)!=="svelte-pj6p0d"&&(A.innerHTML=ht),$e=i(e),g(Q.$$.fragment,e),Fe=i(e),I=m(e,"DIV",{class:!0});var $=re(I);g(O.$$.fragment,$),Ve=i($),ie=m($,"P",{"data-svelte-h":!0}),p(ie)!=="svelte-14u83jz"&&(ie.innerHTML=pt),Re=i($),ce=m($,"P",{"data-svelte-h":!0}),p(ce)!=="svelte-1ynyot8"&&(ce.innerHTML=ut),Ge=i($),g(z.$$.fragment,$),$.forEach(o),Ue=i(e),g(K.$$.fragment,e),Ze=i(e),M=m(e,"DIV",{class:!0});var k=re(M);g(ee.$$.fragment,k),Se=i(k),de=m(k,"P",{"data-svelte-h":!0}),p(de)!=="svelte-s6cqlq"&&(de.textContent=ft),He=i(k),le=m(k,"P",{"data-svelte-h":!0}),p(le)!=="svelte-q52n56"&&(le.innerHTML=gt),Ye=i(k),me=m(k,"P",{"data-svelte-h":!0}),p(me)!=="svelte-hswkmf"&&(me.innerHTML=_t),Ee=i(k),J=m(k,"DIV",{class:!0});var F=re(J);g(te.$$.fragment,F),Pe=i(F),he=m(F,"P",{"data-svelte-h":!0}),p(he)!=="svelte-1am9h41"&&(he.innerHTML=yt),De=i(F),g(W.$$.fragment,F),Le=i(F),g(N.$$.fragment,F),F.forEach(o),k.forEach(o),ze=i(e),g(oe.$$.fragment,e),We=i(e),x=m(e,"DIV",{class:!0});var j=re(x);g(ne.$$.fragment,j),Ae=i(j),pe=m(j,"P",{"data-svelte-h":!0}),p(pe)!=="svelte-mz7fxi"&&(pe.textContent=vt),Qe=i(j),ue=m(j,"P",{"data-svelte-h":!0}),p(ue)!=="svelte-q52n56"&&(ue.innerHTML=bt),Oe=i(j),fe=m(j,"P",{"data-svelte-h":!0}),p(fe)!=="svelte-hswkmf"&&(fe.innerHTML=Tt),Ke=i(j),q=m(j,"DIV",{class:!0});var U=re(q);g(ae.$$.fragment,U),et=i(U),ge=m(U,"P",{"data-svelte-h":!0}),p(ge)!=="svelte-ivw4c9"&&(ge.innerHTML=wt),tt=i(U),g(X.$$.fragment,U),ot=i(U),g(B.$$.fragment,U),U.forEach(o),j.forEach(o),Ne=i(e),g(se.$$.fragment,e),Xe=i(e),ye=m(e,"P",{}),re(ye).forEach(o),this.h()},h(){V(n,"name","hf:doc:metadata"),V(n,"content",Wt),V(Z,"class","flex flex-wrap space-x-1"),V(I,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),V(J,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),V(M,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),V(q,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),V(x,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,t){h(document.head,n),s(e,T,t),s(e,c,t),s(e,d,t),s(e,u,t),s(e,w,t),_(R,e,t),s(e,Te,t),s(e,Z,t),s(e,we,t),_(G,e,t),s(e,Me,t),s(e,S,t),s(e,xe,t),s(e,H,t),s(e,Ie,t),s(e,Y,t),s(e,ke,t),s(e,E,t),s(e,je,t),s(e,P,t),s(e,Ce,t),_(D,e,t),s(e,Je,t),s(e,L,t),s(e,qe,t),s(e,A,t),s(e,$e,t),_(Q,e,t),s(e,Fe,t),s(e,I,t),_(O,I,null),h(I,Ve),h(I,ie),h(I,Re),h(I,ce),h(I,Ge),_(z,I,null),s(e,Ue,t),_(K,e,t),s(e,Ze,t),s(e,M,t),_(ee,M,null),h(M,Se),h(M,de),h(M,He),h(M,le),h(M,Ye),h(M,me),h(M,Ee),h(M,J),_(te,J,null),h(J,Pe),h(J,he),h(J,De),_(W,J,null),h(J,Le),_(N,J,null),s(e,ze,t),_(oe,e,t),s(e,We,t),s(e,x,t),_(ne,x,null),h(x,Ae),h(x,pe),h(x,Qe),h(x,ue),h(x,Oe),h(x,fe),h(x,Ke),h(x,q),_(ae,q,null),h(q,et),h(q,ge),h(q,tt),_(X,q,null),h(q,ot),_(B,q,null),s(e,Ne,t),_(se,e,t),s(e,Xe,t),s(e,ye,t),Be=!0},p(e,[t]){const $={};t&2&&($.$$scope={dirty:t,ctx:e}),z.$set($);const k={};t&2&&(k.$$scope={dirty:t,ctx:e}),W.$set(k);const F={};t&2&&(F.$$scope={dirty:t,ctx:e}),N.$set(F);const j={};t&2&&(j.$$scope={dirty:t,ctx:e}),X.$set(j);const U={};t&2&&(U.$$scope={dirty:t,ctx:e}),B.$set(U)},i(e){Be||(y(R.$$.fragment,e),y(G.$$.fragment,e),y(D.$$.fragment,e),y(Q.$$.fragment,e),y(O.$$.fragment,e),y(z.$$.fragment,e),y(K.$$.fragment,e),y(ee.$$.fragment,e),y(te.$$.fragment,e),y(W.$$.fragment,e),y(N.$$.fragment,e),y(oe.$$.fragment,e),y(ne.$$.fragment,e),y(ae.$$.fragment,e),y(X.$$.fragment,e),y(B.$$.fragment,e),y(se.$$.fragment,e),Be=!0)},o(e){v(R.$$.fragment,e),v(G.$$.fragment,e),v(D.$$.fragment,e),v(Q.$$.fragment,e),v(O.$$.fragment,e),v(z.$$.fragment,e),v(K.$$.fragment,e),v(ee.$$.fragment,e),v(te.$$.fragment,e),v(W.$$.fragment,e),v(N.$$.fragment,e),v(oe.$$.fragment,e),v(ne.$$.fragment,e),v(ae.$$.fragment,e),v(X.$$.fragment,e),v(B.$$.fragment,e),v(se.$$.fragment,e),Be=!1},d(e){e&&(o(T),o(c),o(d),o(u),o(w),o(Te),o(Z),o(we),o(Me),o(S),o(xe),o(H),o(Ie),o(Y),o(ke),o(E),o(je),o(P),o(Ce),o(Je),o(L),o(qe),o(A),o($e),o(Fe),o(I),o(Ue),o(Ze),o(M),o(ze),o(We),o(x),o(Ne),o(Xe),o(ye)),o(n),b(R,e),b(G,e),b(D,e),b(Q,e),b(O),b(z),b(K,e),b(ee),b(te),b(W),b(N),b(oe,e),b(ne),b(ae),b(X),b(B),b(se,e)}}}const Wt='{"title":"Informer","local":"informer","sections":[{"title":"Overview","local":"overview","sections":[],"depth":2},{"title":"Resources","local":"resources","sections":[],"depth":2},{"title":"InformerConfig","local":"transformers.InformerConfig","sections":[],"depth":2},{"title":"InformerModel","local":"transformers.InformerModel","sections":[],"depth":2},{"title":"InformerForPrediction","local":"transformers.InformerForPrediction","sections":[],"depth":2}],"depth":1}';function Nt(C){return It(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class Yt extends kt{constructor(n){super(),jt(this,n,Nt,zt,xt,{})}}export{Yt as component};
