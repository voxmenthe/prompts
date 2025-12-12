import{s as Ct,o as xt,n as be}from"../chunks/scheduler.18a86fab.js";import{S as Ut,i as Zt,g as d,s as l,r as _,A as $t,h as m,f as a,c,j as ie,x as f,u as y,k as N,y as p,a as r,v as b,d as v,t as M,w as T}from"../chunks/index.98837b22.js";import{T as it}from"../chunks/Tip.77304350.js";import{D as Me}from"../chunks/Docstring.a1ef7999.js";import{C as Ne}from"../chunks/CodeBlock.8d0c2e8a.js";import{E as Ge}from"../chunks/ExampleCodeBlock.8c3ee1f9.js";import{H as ye,E as At}from"../chunks/getInferenceSnippets.06c2775f.js";function Ft(w){let t,u;return t=new Ne({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9mb3JtZXJDb25maWclMkMlMjBBdXRvZm9ybWVyTW9kZWwlMEElMEElMjMlMjBJbml0aWFsaXppbmclMjBhJTIwZGVmYXVsdCUyMEF1dG9mb3JtZXIlMjBjb25maWd1cmF0aW9uJTBBY29uZmlndXJhdGlvbiUyMCUzRCUyMEF1dG9mb3JtZXJDb25maWcoKSUwQSUwQSUyMyUyMFJhbmRvbWx5JTIwaW5pdGlhbGl6aW5nJTIwYSUyMG1vZGVsJTIwKHdpdGglMjByYW5kb20lMjB3ZWlnaHRzKSUyMGZyb20lMjB0aGUlMjBjb25maWd1cmF0aW9uJTBBbW9kZWwlMjAlM0QlMjBBdXRvZm9ybWVyTW9kZWwoY29uZmlndXJhdGlvbiklMEElMEElMjMlMjBBY2Nlc3NpbmclMjB0aGUlMjBtb2RlbCUyMGNvbmZpZ3VyYXRpb24lMEFjb25maWd1cmF0aW9uJTIwJTNEJTIwbW9kZWwuY29uZmln",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoformerConfig, AutoformerModel

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a default Autoformer configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = AutoformerConfig()

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Randomly initializing a model (with random weights) from the configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = AutoformerModel(configuration)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Accessing the model configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = model.config`,wrap:!1}}),{c(){_(t.$$.fragment)},l(s){y(t.$$.fragment,s)},m(s,i){b(t,s,i),u=!0},p:be,i(s){u||(v(t.$$.fragment,s),u=!0)},o(s){M(t.$$.fragment,s),u=!1},d(s){T(t,s)}}}function Xt(w){let t,u=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=d("p"),t.innerHTML=u},l(s){t=m(s,"P",{"data-svelte-h":!0}),f(t)!=="svelte-fincs2"&&(t.innerHTML=u)},m(s,i){r(s,t,i)},p:be,d(s){s&&a(t)}}}function Wt(w){let t,u="Examples:",s,i,h;return i=new Ne({props:{code:"ZnJvbSUyMGh1Z2dpbmdmYWNlX2h1YiUyMGltcG9ydCUyMGhmX2h1Yl9kb3dubG9hZCUwQWltcG9ydCUyMHRvcmNoJTBBZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9mb3JtZXJNb2RlbCUwQSUwQWZpbGUlMjAlM0QlMjBoZl9odWJfZG93bmxvYWQoJTBBJTIwJTIwJTIwJTIwcmVwb19pZCUzRCUyMmhmLWludGVybmFsLXRlc3RpbmclMkZ0b3VyaXNtLW1vbnRobHktYmF0Y2glMjIlMkMlMjBmaWxlbmFtZSUzRCUyMnRyYWluLWJhdGNoLnB0JTIyJTJDJTIwcmVwb190eXBlJTNEJTIyZGF0YXNldCUyMiUwQSklMEFiYXRjaCUyMCUzRCUyMHRvcmNoLmxvYWQoZmlsZSklMEElMEFtb2RlbCUyMCUzRCUyMEF1dG9mb3JtZXJNb2RlbC5mcm9tX3ByZXRyYWluZWQoJTIyaHVnZ2luZ2ZhY2UlMkZhdXRvZm9ybWVyLXRvdXJpc20tbW9udGhseSUyMiklMEElMEElMjMlMjBkdXJpbmclMjB0cmFpbmluZyUyQyUyMG9uZSUyMHByb3ZpZGVzJTIwYm90aCUyMHBhc3QlMjBhbmQlMjBmdXR1cmUlMjB2YWx1ZXMlMEElMjMlMjBhcyUyMHdlbGwlMjBhcyUyMHBvc3NpYmxlJTIwYWRkaXRpb25hbCUyMGZlYXR1cmVzJTBBb3V0cHV0cyUyMCUzRCUyMG1vZGVsKCUwQSUyMCUyMCUyMCUyMHBhc3RfdmFsdWVzJTNEYmF0Y2glNUIlMjJwYXN0X3ZhbHVlcyUyMiU1RCUyQyUwQSUyMCUyMCUyMCUyMHBhc3RfdGltZV9mZWF0dXJlcyUzRGJhdGNoJTVCJTIycGFzdF90aW1lX2ZlYXR1cmVzJTIyJTVEJTJDJTBBJTIwJTIwJTIwJTIwcGFzdF9vYnNlcnZlZF9tYXNrJTNEYmF0Y2glNUIlMjJwYXN0X29ic2VydmVkX21hc2slMjIlNUQlMkMlMEElMjAlMjAlMjAlMjBzdGF0aWNfY2F0ZWdvcmljYWxfZmVhdHVyZXMlM0RiYXRjaCU1QiUyMnN0YXRpY19jYXRlZ29yaWNhbF9mZWF0dXJlcyUyMiU1RCUyQyUwQSUyMCUyMCUyMCUyMGZ1dHVyZV92YWx1ZXMlM0RiYXRjaCU1QiUyMmZ1dHVyZV92YWx1ZXMlMjIlNUQlMkMlMEElMjAlMjAlMjAlMjBmdXR1cmVfdGltZV9mZWF0dXJlcyUzRGJhdGNoJTVCJTIyZnV0dXJlX3RpbWVfZmVhdHVyZXMlMjIlNUQlMkMlMEEpJTBBJTBBbGFzdF9oaWRkZW5fc3RhdGUlMjAlM0QlMjBvdXRwdXRzLmxhc3RfaGlkZGVuX3N0YXRl",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> huggingface_hub <span class="hljs-keyword">import</span> hf_hub_download
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoformerModel

<span class="hljs-meta">&gt;&gt;&gt; </span>file = hf_hub_download(
<span class="hljs-meta">... </span>    repo_id=<span class="hljs-string">&quot;hf-internal-testing/tourism-monthly-batch&quot;</span>, filename=<span class="hljs-string">&quot;train-batch.pt&quot;</span>, repo_type=<span class="hljs-string">&quot;dataset&quot;</span>
<span class="hljs-meta">... </span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>batch = torch.load(file)

<span class="hljs-meta">&gt;&gt;&gt; </span>model = AutoformerModel.from_pretrained(<span class="hljs-string">&quot;huggingface/autoformer-tourism-monthly&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># during training, one provides both past and future values</span>
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># as well as possible additional features</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(
<span class="hljs-meta">... </span>    past_values=batch[<span class="hljs-string">&quot;past_values&quot;</span>],
<span class="hljs-meta">... </span>    past_time_features=batch[<span class="hljs-string">&quot;past_time_features&quot;</span>],
<span class="hljs-meta">... </span>    past_observed_mask=batch[<span class="hljs-string">&quot;past_observed_mask&quot;</span>],
<span class="hljs-meta">... </span>    static_categorical_features=batch[<span class="hljs-string">&quot;static_categorical_features&quot;</span>],
<span class="hljs-meta">... </span>    future_values=batch[<span class="hljs-string">&quot;future_values&quot;</span>],
<span class="hljs-meta">... </span>    future_time_features=batch[<span class="hljs-string">&quot;future_time_features&quot;</span>],
<span class="hljs-meta">... </span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>last_hidden_state = outputs.last_hidden_state`,wrap:!1}}),{c(){t=d("p"),t.textContent=u,s=l(),_(i.$$.fragment)},l(o){t=m(o,"P",{"data-svelte-h":!0}),f(t)!=="svelte-kvfsh7"&&(t.textContent=u),s=c(o),y(i.$$.fragment,o)},m(o,g){r(o,t,g),r(o,s,g),b(i,o,g),h=!0},p:be,i(o){h||(v(i.$$.fragment,o),h=!0)},o(o){M(i.$$.fragment,o),h=!1},d(o){o&&(a(t),a(s)),T(i,o)}}}function zt(w){let t,u=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=d("p"),t.innerHTML=u},l(s){t=m(s,"P",{"data-svelte-h":!0}),f(t)!=="svelte-fincs2"&&(t.innerHTML=u)},m(s,i){r(s,t,i)},p:be,d(s){s&&a(t)}}}function Vt(w){let t,u="Examples:",s,i,h;return i=new Ne({props:{code:"ZnJvbSUyMGh1Z2dpbmdmYWNlX2h1YiUyMGltcG9ydCUyMGhmX2h1Yl9kb3dubG9hZCUwQWltcG9ydCUyMHRvcmNoJTBBZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9mb3JtZXJGb3JQcmVkaWN0aW9uJTBBJTBBZmlsZSUyMCUzRCUyMGhmX2h1Yl9kb3dubG9hZCglMEElMjAlMjAlMjAlMjByZXBvX2lkJTNEJTIyaGYtaW50ZXJuYWwtdGVzdGluZyUyRnRvdXJpc20tbW9udGhseS1iYXRjaCUyMiUyQyUyMGZpbGVuYW1lJTNEJTIydHJhaW4tYmF0Y2gucHQlMjIlMkMlMjByZXBvX3R5cGUlM0QlMjJkYXRhc2V0JTIyJTBBKSUwQWJhdGNoJTIwJTNEJTIwdG9yY2gubG9hZChmaWxlKSUwQSUwQW1vZGVsJTIwJTNEJTIwQXV0b2Zvcm1lckZvclByZWRpY3Rpb24uZnJvbV9wcmV0cmFpbmVkKCUyMmh1Z2dpbmdmYWNlJTJGYXV0b2Zvcm1lci10b3VyaXNtLW1vbnRobHklMjIpJTBBJTBBJTIzJTIwZHVyaW5nJTIwdHJhaW5pbmclMkMlMjBvbmUlMjBwcm92aWRlcyUyMGJvdGglMjBwYXN0JTIwYW5kJTIwZnV0dXJlJTIwdmFsdWVzJTBBJTIzJTIwYXMlMjB3ZWxsJTIwYXMlMjBwb3NzaWJsZSUyMGFkZGl0aW9uYWwlMjBmZWF0dXJlcyUwQW91dHB1dHMlMjAlM0QlMjBtb2RlbCglMEElMjAlMjAlMjAlMjBwYXN0X3ZhbHVlcyUzRGJhdGNoJTVCJTIycGFzdF92YWx1ZXMlMjIlNUQlMkMlMEElMjAlMjAlMjAlMjBwYXN0X3RpbWVfZmVhdHVyZXMlM0RiYXRjaCU1QiUyMnBhc3RfdGltZV9mZWF0dXJlcyUyMiU1RCUyQyUwQSUyMCUyMCUyMCUyMHBhc3Rfb2JzZXJ2ZWRfbWFzayUzRGJhdGNoJTVCJTIycGFzdF9vYnNlcnZlZF9tYXNrJTIyJTVEJTJDJTBBJTIwJTIwJTIwJTIwc3RhdGljX2NhdGVnb3JpY2FsX2ZlYXR1cmVzJTNEYmF0Y2glNUIlMjJzdGF0aWNfY2F0ZWdvcmljYWxfZmVhdHVyZXMlMjIlNUQlMkMlMEElMjAlMjAlMjAlMjBmdXR1cmVfdmFsdWVzJTNEYmF0Y2glNUIlMjJmdXR1cmVfdmFsdWVzJTIyJTVEJTJDJTBBJTIwJTIwJTIwJTIwZnV0dXJlX3RpbWVfZmVhdHVyZXMlM0RiYXRjaCU1QiUyMmZ1dHVyZV90aW1lX2ZlYXR1cmVzJTIyJTVEJTJDJTBBKSUwQSUwQWxvc3MlMjAlM0QlMjBvdXRwdXRzLmxvc3MlMEFsb3NzLmJhY2t3YXJkKCklMEElMEElMjMlMjBkdXJpbmclMjBpbmZlcmVuY2UlMkMlMjBvbmUlMjBvbmx5JTIwcHJvdmlkZXMlMjBwYXN0JTIwdmFsdWVzJTBBJTIzJTIwYXMlMjB3ZWxsJTIwYXMlMjBwb3NzaWJsZSUyMGFkZGl0aW9uYWwlMjBmZWF0dXJlcyUwQSUyMyUyMHRoZSUyMG1vZGVsJTIwYXV0b3JlZ3Jlc3NpdmVseSUyMGdlbmVyYXRlcyUyMGZ1dHVyZSUyMHZhbHVlcyUwQW91dHB1dHMlMjAlM0QlMjBtb2RlbC5nZW5lcmF0ZSglMEElMjAlMjAlMjAlMjBwYXN0X3ZhbHVlcyUzRGJhdGNoJTVCJTIycGFzdF92YWx1ZXMlMjIlNUQlMkMlMEElMjAlMjAlMjAlMjBwYXN0X3RpbWVfZmVhdHVyZXMlM0RiYXRjaCU1QiUyMnBhc3RfdGltZV9mZWF0dXJlcyUyMiU1RCUyQyUwQSUyMCUyMCUyMCUyMHBhc3Rfb2JzZXJ2ZWRfbWFzayUzRGJhdGNoJTVCJTIycGFzdF9vYnNlcnZlZF9tYXNrJTIyJTVEJTJDJTBBJTIwJTIwJTIwJTIwc3RhdGljX2NhdGVnb3JpY2FsX2ZlYXR1cmVzJTNEYmF0Y2glNUIlMjJzdGF0aWNfY2F0ZWdvcmljYWxfZmVhdHVyZXMlMjIlNUQlMkMlMEElMjAlMjAlMjAlMjBmdXR1cmVfdGltZV9mZWF0dXJlcyUzRGJhdGNoJTVCJTIyZnV0dXJlX3RpbWVfZmVhdHVyZXMlMjIlNUQlMkMlMEEpJTBBJTBBbWVhbl9wcmVkaWN0aW9uJTIwJTNEJTIwb3V0cHV0cy5zZXF1ZW5jZXMubWVhbihkaW0lM0QxKQ==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> huggingface_hub <span class="hljs-keyword">import</span> hf_hub_download
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoformerForPrediction

<span class="hljs-meta">&gt;&gt;&gt; </span>file = hf_hub_download(
<span class="hljs-meta">... </span>    repo_id=<span class="hljs-string">&quot;hf-internal-testing/tourism-monthly-batch&quot;</span>, filename=<span class="hljs-string">&quot;train-batch.pt&quot;</span>, repo_type=<span class="hljs-string">&quot;dataset&quot;</span>
<span class="hljs-meta">... </span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>batch = torch.load(file)

<span class="hljs-meta">&gt;&gt;&gt; </span>model = AutoformerForPrediction.from_pretrained(<span class="hljs-string">&quot;huggingface/autoformer-tourism-monthly&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># during training, one provides both past and future values</span>
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># as well as possible additional features</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(
<span class="hljs-meta">... </span>    past_values=batch[<span class="hljs-string">&quot;past_values&quot;</span>],
<span class="hljs-meta">... </span>    past_time_features=batch[<span class="hljs-string">&quot;past_time_features&quot;</span>],
<span class="hljs-meta">... </span>    past_observed_mask=batch[<span class="hljs-string">&quot;past_observed_mask&quot;</span>],
<span class="hljs-meta">... </span>    static_categorical_features=batch[<span class="hljs-string">&quot;static_categorical_features&quot;</span>],
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
<span class="hljs-meta">... </span>    future_time_features=batch[<span class="hljs-string">&quot;future_time_features&quot;</span>],
<span class="hljs-meta">... </span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>mean_prediction = outputs.sequences.mean(dim=<span class="hljs-number">1</span>)`,wrap:!1}}),{c(){t=d("p"),t.textContent=u,s=l(),_(i.$$.fragment)},l(o){t=m(o,"P",{"data-svelte-h":!0}),f(t)!=="svelte-kvfsh7"&&(t.textContent=u),s=c(o),y(i.$$.fragment,o)},m(o,g){r(o,t,g),r(o,s,g),b(i,o,g),h=!0},p:be,i(o){h||(v(i.$$.fragment,o),h=!0)},o(o){M(i.$$.fragment,o),h=!1},d(o){o&&(a(t),a(s)),T(i,o)}}}function Bt(w){let t,u="is equal to 1), initialize the model and call as shown below:",s,i,h;return i=new Ne({props:{code:"ZnJvbSUyMGh1Z2dpbmdmYWNlX2h1YiUyMGltcG9ydCUyMGhmX2h1Yl9kb3dubG9hZCUwQWltcG9ydCUyMHRvcmNoJTBBZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9mb3JtZXJDb25maWclMkMlMjBBdXRvZm9ybWVyRm9yUHJlZGljdGlvbiUwQSUwQWZpbGUlMjAlM0QlMjBoZl9odWJfZG93bmxvYWQoJTBBJTIwJTIwJTIwJTIwcmVwb19pZCUzRCUyMmhmLWludGVybmFsLXRlc3RpbmclMkZ0b3VyaXNtLW1vbnRobHktYmF0Y2glMjIlMkMlMjBmaWxlbmFtZSUzRCUyMnRyYWluLWJhdGNoLnB0JTIyJTJDJTIwcmVwb190eXBlJTNEJTIyZGF0YXNldCUyMiUwQSklMEFiYXRjaCUyMCUzRCUyMHRvcmNoLmxvYWQoZmlsZSklMEElMEElMjMlMjBjaGVjayUyMG51bWJlciUyMG9mJTIwc3RhdGljJTIwcmVhbCUyMGZlYXR1cmVzJTBBbnVtX3N0YXRpY19yZWFsX2ZlYXR1cmVzJTIwJTNEJTIwYmF0Y2glNUIlMjJzdGF0aWNfcmVhbF9mZWF0dXJlcyUyMiU1RC5zaGFwZSU1Qi0xJTVEJTBBJTBBJTIzJTIwbG9hZCUyMGNvbmZpZ3VyYXRpb24lMjBvZiUyMHByZXRyYWluZWQlMjBtb2RlbCUyMGFuZCUyMG92ZXJyaWRlJTIwbnVtX3N0YXRpY19yZWFsX2ZlYXR1cmVzJTBBY29uZmlndXJhdGlvbiUyMCUzRCUyMEF1dG9mb3JtZXJDb25maWcuZnJvbV9wcmV0cmFpbmVkKCUwQSUyMCUyMCUyMCUyMCUyMmh1Z2dpbmdmYWNlJTJGYXV0b2Zvcm1lci10b3VyaXNtLW1vbnRobHklMjIlMkMlMEElMjAlMjAlMjAlMjBudW1fc3RhdGljX3JlYWxfZmVhdHVyZXMlM0RudW1fc3RhdGljX3JlYWxfZmVhdHVyZXMlMkMlMEEpJTBBJTIzJTIwd2UlMjBhbHNvJTIwbmVlZCUyMHRvJTIwdXBkYXRlJTIwZmVhdHVyZV9zaXplJTIwYXMlMjBpdCUyMGlzJTIwbm90JTIwcmVjYWxjdWxhdGVkJTBBY29uZmlndXJhdGlvbi5mZWF0dXJlX3NpemUlMjAlMkIlM0QlMjBudW1fc3RhdGljX3JlYWxfZmVhdHVyZXMlMEElMEFtb2RlbCUyMCUzRCUyMEF1dG9mb3JtZXJGb3JQcmVkaWN0aW9uKGNvbmZpZ3VyYXRpb24pJTBBJTBBb3V0cHV0cyUyMCUzRCUyMG1vZGVsKCUwQSUyMCUyMCUyMCUyMHBhc3RfdmFsdWVzJTNEYmF0Y2glNUIlMjJwYXN0X3ZhbHVlcyUyMiU1RCUyQyUwQSUyMCUyMCUyMCUyMHBhc3RfdGltZV9mZWF0dXJlcyUzRGJhdGNoJTVCJTIycGFzdF90aW1lX2ZlYXR1cmVzJTIyJTVEJTJDJTBBJTIwJTIwJTIwJTIwcGFzdF9vYnNlcnZlZF9tYXNrJTNEYmF0Y2glNUIlMjJwYXN0X29ic2VydmVkX21hc2slMjIlNUQlMkMlMEElMjAlMjAlMjAlMjBzdGF0aWNfY2F0ZWdvcmljYWxfZmVhdHVyZXMlM0RiYXRjaCU1QiUyMnN0YXRpY19jYXRlZ29yaWNhbF9mZWF0dXJlcyUyMiU1RCUyQyUwQSUyMCUyMCUyMCUyMHN0YXRpY19yZWFsX2ZlYXR1cmVzJTNEYmF0Y2glNUIlMjJzdGF0aWNfcmVhbF9mZWF0dXJlcyUyMiU1RCUyQyUwQSUyMCUyMCUyMCUyMGZ1dHVyZV92YWx1ZXMlM0RiYXRjaCU1QiUyMmZ1dHVyZV92YWx1ZXMlMjIlNUQlMkMlMEElMjAlMjAlMjAlMjBmdXR1cmVfdGltZV9mZWF0dXJlcyUzRGJhdGNoJTVCJTIyZnV0dXJlX3RpbWVfZmVhdHVyZXMlMjIlNUQlMkMlMEEp",highlighted:`<span class="hljs-meta prompt_">&gt;&gt;&gt;</span> <span class="language-python"><span class="hljs-keyword">from</span> huggingface_hub <span class="hljs-keyword">import</span> hf_hub_download</span>
<span class="hljs-meta prompt_">&gt;&gt;&gt;</span> <span class="language-python"><span class="hljs-keyword">import</span> torch</span>
<span class="hljs-meta prompt_">&gt;&gt;&gt;</span> <span class="language-python"><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoformerConfig, AutoformerForPrediction</span>

<span class="hljs-meta prompt_">&gt;&gt;&gt;</span> <span class="language-python">file = hf_hub_download(</span>
<span class="hljs-meta prompt_">...</span> <span class="language-python">    repo_id=<span class="hljs-string">&quot;hf-internal-testing/tourism-monthly-batch&quot;</span>, filename=<span class="hljs-string">&quot;train-batch.pt&quot;</span>, repo_type=<span class="hljs-string">&quot;dataset&quot;</span></span>
<span class="hljs-meta prompt_">...</span> <span class="language-python">)</span>
<span class="hljs-meta prompt_">&gt;&gt;&gt;</span> <span class="language-python">batch = torch.load(file)</span>

<span class="hljs-meta prompt_">&gt;&gt;&gt;</span> <span class="language-python"><span class="hljs-comment"># check number of static real features</span></span>
<span class="hljs-meta prompt_">&gt;&gt;&gt;</span> <span class="language-python">num_static_real_features = batch[<span class="hljs-string">&quot;static_real_features&quot;</span>].shape[-<span class="hljs-number">1</span>]</span>

<span class="hljs-meta prompt_">&gt;&gt;&gt;</span> <span class="language-python"><span class="hljs-comment"># load configuration of pretrained model and override num_static_real_features</span></span>
<span class="hljs-meta prompt_">&gt;&gt;&gt;</span> <span class="language-python">configuration = AutoformerConfig.from_pretrained(</span>
<span class="hljs-meta prompt_">...</span> <span class="language-python">    <span class="hljs-string">&quot;huggingface/autoformer-tourism-monthly&quot;</span>,</span>
<span class="hljs-meta prompt_">...</span> <span class="language-python">    num_static_real_features=num_static_real_features,</span>
<span class="hljs-meta prompt_">...</span> <span class="language-python">)</span>
<span class="hljs-meta prompt_">&gt;&gt;&gt;</span> <span class="language-python"><span class="hljs-comment"># we also need to update feature_size as it is not recalculated</span></span>
<span class="hljs-meta prompt_">&gt;&gt;&gt;</span> <span class="language-python">configuration.feature_size += num_static_real_features</span>

<span class="hljs-meta prompt_">&gt;&gt;&gt;</span> <span class="language-python">model = AutoformerForPrediction(configuration)</span>

<span class="hljs-meta prompt_">&gt;&gt;&gt;</span> <span class="language-python">outputs = model(</span>
<span class="hljs-meta prompt_">...</span> <span class="language-python">    past_values=batch[<span class="hljs-string">&quot;past_values&quot;</span>],</span>
<span class="hljs-meta prompt_">...</span> <span class="language-python">    past_time_features=batch[<span class="hljs-string">&quot;past_time_features&quot;</span>],</span>
<span class="hljs-meta prompt_">...</span> <span class="language-python">    past_observed_mask=batch[<span class="hljs-string">&quot;past_observed_mask&quot;</span>],</span>
<span class="hljs-meta prompt_">...</span> <span class="language-python">    static_categorical_features=batch[<span class="hljs-string">&quot;static_categorical_features&quot;</span>],</span>
<span class="hljs-meta prompt_">...</span> <span class="language-python">    static_real_features=batch[<span class="hljs-string">&quot;static_real_features&quot;</span>],</span>
<span class="hljs-meta prompt_">...</span> <span class="language-python">    future_values=batch[<span class="hljs-string">&quot;future_values&quot;</span>],</span>
<span class="hljs-meta prompt_">...</span> <span class="language-python">    future_time_features=batch[<span class="hljs-string">&quot;future_time_features&quot;</span>],</span>
<span class="hljs-meta prompt_">...</span> <span class="language-python">)</span>`,wrap:!1}}),{c(){t=d("p"),t.textContent=u,s=l(),_(i.$$.fragment)},l(o){t=m(o,"P",{"data-svelte-h":!0}),f(t)!=="svelte-1hzwtfu"&&(t.textContent=u),s=c(o),y(i.$$.fragment,o)},m(o,g){r(o,t,g),r(o,s,g),b(i,o,g),h=!0},p:be,i(o){h||(v(i.$$.fragment,o),h=!0)},o(o){M(i.$$.fragment,o),h=!1},d(o){o&&(a(t),a(s)),T(i,o)}}}function qt(w){let t,u=`The AutoformerForPrediction can also use static_real_features. To do so, set num_static_real_features in
AutoformerConfig based on number of such features in the dataset (in case of tourism_monthly dataset it`,s,i,h;return i=new Ge({props:{anchor:"transformers.AutoformerForPrediction.forward.example-2",$$slots:{default:[Bt]},$$scope:{ctx:w}}}),{c(){t=d("p"),t.textContent=u,s=l(),_(i.$$.fragment)},l(o){t=m(o,"P",{"data-svelte-h":!0}),f(t)!=="svelte-14i5hlv"&&(t.textContent=u),s=c(o),y(i.$$.fragment,o)},m(o,g){r(o,t,g),r(o,s,g),b(i,o,g),h=!0},p(o,g){const X={};g&2&&(X.$$scope={dirty:g,ctx:o}),i.$set(X)},i(o){h||(v(i.$$.fragment,o),h=!0)},o(o){M(i.$$.fragment,o),h=!1},d(o){o&&(a(t),a(s)),T(i,o)}}}function Rt(w){let t,u,s,i,h,o="<em>This model was released on 2021-06-24 and added to Hugging Face Transformers on 2023-05-30.</em>",g,X,Te,W,lt='<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/>',we,Y,je,I,ct='The Autoformer model was proposed in <a href="https://huggingface.co/papers/2106.13008" rel="nofollow">Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting</a> by Haixu Wu, Jiehui Xu, Jianmin Wang, Mingsheng Long.',Je,E,dt="This model augments the Transformer as a deep decomposition architecture, which can progressively decompose the trend and seasonal components during the forecasting process.",ke,H,mt="The abstract from the paper is the following:",Ce,S,pt="<em>Extending the forecasting time is a critical demand for real applications, such as extreme weather early warning and long-term energy consumption planning. This paper studies the long-term forecasting problem of time series. Prior Transformer-based models adopt various self-attention mechanisms to discover the long-range dependencies. However, intricate temporal patterns of the long-term future prohibit the model from finding reliable dependencies. Also, Transformers have to adopt the sparse versions of point-wise self-attentions for long series efficiency, resulting in the information utilization bottleneck. Going beyond Transformers, we design Autoformer as a novel decomposition architecture with an Auto-Correlation mechanism. We break with the pre-processing convention of series decomposition and renovate it as a basic inner block of deep models. This design empowers Autoformer with progressive decomposition capacities for complex time series. Further, inspired by the stochastic process theory, we design the Auto-Correlation mechanism based on the series periodicity, which conducts the dependencies discovery and representation aggregation at the sub-series level. Auto-Correlation outperforms self-attention in both efficiency and accuracy. In long-term forecasting, Autoformer yields state-of-the-art accuracy, with a 38% relative improvement on six benchmarks, covering five practical applications: energy, traffic, economics, weather and disease.</em>",xe,P,ut=`This model was contributed by <a href="https://huggingface.co/elisim" rel="nofollow">elisim</a> and <a href="https://huggingface.co/kashif" rel="nofollow">kashif</a>.
The original code can be found <a href="https://github.com/thuml/Autoformer" rel="nofollow">here</a>.`,Ue,Q,Ze,D,ht="A list of official Hugging Face and community (indicated by 🌎) resources to help you get started. If you’re interested in submitting a resource to be included here, please feel free to open a Pull Request and we’ll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.",$e,L,ft='<li>Check out the Autoformer blog-post in HuggingFace blog: <a href="https://huggingface.co/blog/autoformer" rel="nofollow">Yes, Transformers are Effective for Time Series Forecasting (+ Autoformer)</a></li>',Ae,O,Fe,C,K,Ye,le,gt=`This is the configuration class to store the configuration of an <a href="/docs/transformers/v4.56.2/en/model_doc/autoformer#transformers.AutoformerModel">AutoformerModel</a>. It is used to instantiate an
Autoformer model according to the specified arguments, defining the model architecture. Instantiating a
configuration with the defaults will yield a similar configuration to that of the Autoformer
<a href="https://huggingface.co/huggingface/autoformer-tourism-monthly" rel="nofollow">huggingface/autoformer-tourism-monthly</a>
architecture.`,Ie,ce,_t=`Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> can be used to control the model outputs. Read the
documentation from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> for more information.`,Ee,z,Xe,ee,We,j,te,He,de,yt="The bare Autoformer Model outputting raw hidden-states without any specific head on top.",Se,me,bt=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Pe,pe,vt=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Qe,$,oe,De,ue,Mt='The <a href="/docs/transformers/v4.56.2/en/model_doc/autoformer#transformers.AutoformerModel">AutoformerModel</a> forward method, overrides the <code>__call__</code> special method.',Le,V,Oe,B,ze,ne,Ve,J,ae,Ke,he,Tt="The Autoformer Model with a distribution head on top for time-series forecasting.",et,fe,wt=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,tt,ge,jt=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,ot,k,se,nt,_e,Jt='The <a href="/docs/transformers/v4.56.2/en/model_doc/autoformer#transformers.AutoformerForPrediction">AutoformerForPrediction</a> forward method, overrides the <code>__call__</code> special method.',at,q,st,R,rt,G,Be,re,qe,ve,Re;return X=new ye({props:{title:"Autoformer",local:"autoformer",headingTag:"h1"}}),Y=new ye({props:{title:"Overview",local:"overview",headingTag:"h2"}}),Q=new ye({props:{title:"Resources",local:"resources",headingTag:"h2"}}),O=new ye({props:{title:"AutoformerConfig",local:"transformers.AutoformerConfig",headingTag:"h2"}}),K=new Me({props:{name:"class transformers.AutoformerConfig",anchor:"transformers.AutoformerConfig",parameters:[{name:"prediction_length",val:": typing.Optional[int] = None"},{name:"context_length",val:": typing.Optional[int] = None"},{name:"distribution_output",val:": str = 'student_t'"},{name:"loss",val:": str = 'nll'"},{name:"input_size",val:": int = 1"},{name:"lags_sequence",val:": list = [1, 2, 3, 4, 5, 6, 7]"},{name:"scaling",val:": bool = True"},{name:"num_time_features",val:": int = 0"},{name:"num_dynamic_real_features",val:": int = 0"},{name:"num_static_categorical_features",val:": int = 0"},{name:"num_static_real_features",val:": int = 0"},{name:"cardinality",val:": typing.Optional[list[int]] = None"},{name:"embedding_dimension",val:": typing.Optional[list[int]] = None"},{name:"d_model",val:": int = 64"},{name:"encoder_attention_heads",val:": int = 2"},{name:"decoder_attention_heads",val:": int = 2"},{name:"encoder_layers",val:": int = 2"},{name:"decoder_layers",val:": int = 2"},{name:"encoder_ffn_dim",val:": int = 32"},{name:"decoder_ffn_dim",val:": int = 32"},{name:"activation_function",val:": str = 'gelu'"},{name:"dropout",val:": float = 0.1"},{name:"encoder_layerdrop",val:": float = 0.1"},{name:"decoder_layerdrop",val:": float = 0.1"},{name:"attention_dropout",val:": float = 0.1"},{name:"activation_dropout",val:": float = 0.1"},{name:"num_parallel_samples",val:": int = 100"},{name:"init_std",val:": float = 0.02"},{name:"use_cache",val:": bool = True"},{name:"is_encoder_decoder",val:" = True"},{name:"label_length",val:": int = 10"},{name:"moving_average",val:": int = 25"},{name:"autocorrelation_factor",val:": int = 3"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.AutoformerConfig.prediction_length",description:`<strong>prediction_length</strong> (<code>int</code>) &#x2014;
The prediction length for the decoder. In other words, the prediction horizon of the model.`,name:"prediction_length"},{anchor:"transformers.AutoformerConfig.context_length",description:`<strong>context_length</strong> (<code>int</code>, <em>optional</em>, defaults to <code>prediction_length</code>) &#x2014;
The context length for the encoder. If unset, the context length will be the same as the
<code>prediction_length</code>.`,name:"context_length"},{anchor:"transformers.AutoformerConfig.distribution_output",description:`<strong>distribution_output</strong> (<code>string</code>, <em>optional</em>, defaults to <code>&quot;student_t&quot;</code>) &#x2014;
The distribution emission head for the model. Could be either &#x201C;student_t&#x201D;, &#x201C;normal&#x201D; or &#x201C;negative_binomial&#x201D;.`,name:"distribution_output"},{anchor:"transformers.AutoformerConfig.loss",description:`<strong>loss</strong> (<code>string</code>, <em>optional</em>, defaults to <code>&quot;nll&quot;</code>) &#x2014;
The loss function for the model corresponding to the <code>distribution_output</code> head. For parametric
distributions it is the negative log likelihood (nll) - which currently is the only supported one.`,name:"loss"},{anchor:"transformers.AutoformerConfig.input_size",description:`<strong>input_size</strong> (<code>int</code>, <em>optional</em>, defaults to 1) &#x2014;
The size of the target variable which by default is 1 for univariate targets. Would be &gt; 1 in case of
multivariate targets.`,name:"input_size"},{anchor:"transformers.AutoformerConfig.lags_sequence",description:`<strong>lags_sequence</strong> (<code>list[int]</code>, <em>optional</em>, defaults to <code>[1, 2, 3, 4, 5, 6, 7]</code>) &#x2014;
The lags of the input time series as covariates often dictated by the frequency. Default is <code>[1, 2, 3, 4, 5, 6, 7]</code>.`,name:"lags_sequence"},{anchor:"transformers.AutoformerConfig.scaling",description:`<strong>scaling</strong> (<code>bool</code>, <em>optional</em> defaults to <code>True</code>) &#x2014;
Whether to scale the input targets.`,name:"scaling"},{anchor:"transformers.AutoformerConfig.num_time_features",description:`<strong>num_time_features</strong> (<code>int</code>, <em>optional</em>, defaults to 0) &#x2014;
The number of time features in the input time series.`,name:"num_time_features"},{anchor:"transformers.AutoformerConfig.num_dynamic_real_features",description:`<strong>num_dynamic_real_features</strong> (<code>int</code>, <em>optional</em>, defaults to 0) &#x2014;
The number of dynamic real valued features.`,name:"num_dynamic_real_features"},{anchor:"transformers.AutoformerConfig.num_static_categorical_features",description:`<strong>num_static_categorical_features</strong> (<code>int</code>, <em>optional</em>, defaults to 0) &#x2014;
The number of static categorical features.`,name:"num_static_categorical_features"},{anchor:"transformers.AutoformerConfig.num_static_real_features",description:`<strong>num_static_real_features</strong> (<code>int</code>, <em>optional</em>, defaults to 0) &#x2014;
The number of static real valued features.`,name:"num_static_real_features"},{anchor:"transformers.AutoformerConfig.cardinality",description:`<strong>cardinality</strong> (<code>list[int]</code>, <em>optional</em>) &#x2014;
The cardinality (number of different values) for each of the static categorical features. Should be a list
of integers, having the same length as <code>num_static_categorical_features</code>. Cannot be <code>None</code> if
<code>num_static_categorical_features</code> is &gt; 0.`,name:"cardinality"},{anchor:"transformers.AutoformerConfig.embedding_dimension",description:`<strong>embedding_dimension</strong> (<code>list[int]</code>, <em>optional</em>) &#x2014;
The dimension of the embedding for each of the static categorical features. Should be a list of integers,
having the same length as <code>num_static_categorical_features</code>. Cannot be <code>None</code> if
<code>num_static_categorical_features</code> is &gt; 0.`,name:"embedding_dimension"},{anchor:"transformers.AutoformerConfig.d_model",description:`<strong>d_model</strong> (<code>int</code>, <em>optional</em>, defaults to 64) &#x2014;
Dimensionality of the transformer layers.`,name:"d_model"},{anchor:"transformers.AutoformerConfig.encoder_layers",description:`<strong>encoder_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 2) &#x2014;
Number of encoder layers.`,name:"encoder_layers"},{anchor:"transformers.AutoformerConfig.decoder_layers",description:`<strong>decoder_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 2) &#x2014;
Number of decoder layers.`,name:"decoder_layers"},{anchor:"transformers.AutoformerConfig.encoder_attention_heads",description:`<strong>encoder_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 2) &#x2014;
Number of attention heads for each attention layer in the Transformer encoder.`,name:"encoder_attention_heads"},{anchor:"transformers.AutoformerConfig.decoder_attention_heads",description:`<strong>decoder_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 2) &#x2014;
Number of attention heads for each attention layer in the Transformer decoder.`,name:"decoder_attention_heads"},{anchor:"transformers.AutoformerConfig.encoder_ffn_dim",description:`<strong>encoder_ffn_dim</strong> (<code>int</code>, <em>optional</em>, defaults to 32) &#x2014;
Dimension of the &#x201C;intermediate&#x201D; (often named feed-forward) layer in encoder.`,name:"encoder_ffn_dim"},{anchor:"transformers.AutoformerConfig.decoder_ffn_dim",description:`<strong>decoder_ffn_dim</strong> (<code>int</code>, <em>optional</em>, defaults to 32) &#x2014;
Dimension of the &#x201C;intermediate&#x201D; (often named feed-forward) layer in decoder.`,name:"decoder_ffn_dim"},{anchor:"transformers.AutoformerConfig.activation_function",description:`<strong>activation_function</strong> (<code>str</code> or <code>function</code>, <em>optional</em>, defaults to <code>&quot;gelu&quot;</code>) &#x2014;
The non-linear activation function (function or string) in the encoder and decoder. If string, <code>&quot;gelu&quot;</code> and
<code>&quot;relu&quot;</code> are supported.`,name:"activation_function"},{anchor:"transformers.AutoformerConfig.dropout",description:`<strong>dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout probability for all fully connected layers in the encoder, and decoder.`,name:"dropout"},{anchor:"transformers.AutoformerConfig.encoder_layerdrop",description:`<strong>encoder_layerdrop</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout probability for the attention and fully connected layers for each encoder layer.`,name:"encoder_layerdrop"},{anchor:"transformers.AutoformerConfig.decoder_layerdrop",description:`<strong>decoder_layerdrop</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout probability for the attention and fully connected layers for each decoder layer.`,name:"decoder_layerdrop"},{anchor:"transformers.AutoformerConfig.attention_dropout",description:`<strong>attention_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout probability for the attention probabilities.`,name:"attention_dropout"},{anchor:"transformers.AutoformerConfig.activation_dropout",description:`<strong>activation_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout probability used between the two layers of the feed-forward networks.`,name:"activation_dropout"},{anchor:"transformers.AutoformerConfig.num_parallel_samples",description:`<strong>num_parallel_samples</strong> (<code>int</code>, <em>optional</em>, defaults to 100) &#x2014;
The number of samples to generate in parallel for each time step of inference.`,name:"num_parallel_samples"},{anchor:"transformers.AutoformerConfig.init_std",description:`<strong>init_std</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated normal weight initialization distribution.`,name:"init_std"},{anchor:"transformers.AutoformerConfig.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether to use the past key/values attentions (if applicable to the model) to speed up decoding.`,name:"use_cache"},{anchor:"transformers.AutoformerConfig.label_length",description:`<strong>label_length</strong> (<code>int</code>, <em>optional</em>, defaults to 10) &#x2014;
Start token length of the Autoformer decoder, which is used for direct multi-step prediction (i.e.
non-autoregressive generation).`,name:"label_length"},{anchor:"transformers.AutoformerConfig.moving_average",description:`<strong>moving_average</strong> (<code>int</code>, <em>optional</em>, defaults to 25) &#x2014;
The window size of the moving average. In practice, it&#x2019;s the kernel size in AvgPool1d of the Decomposition
Layer.`,name:"moving_average"},{anchor:"transformers.AutoformerConfig.autocorrelation_factor",description:`<strong>autocorrelation_factor</strong> (<code>int</code>, <em>optional</em>, defaults to 3) &#x2014;
&#x201C;Attention&#x201D; (i.e. AutoCorrelation mechanism) factor which is used to find top k autocorrelations delays.
It&#x2019;s recommended in the paper to set it to a number between 1 and 5.`,name:"autocorrelation_factor"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/autoformer/configuration_autoformer.py#L26"}}),z=new Ge({props:{anchor:"transformers.AutoformerConfig.example",$$slots:{default:[Ft]},$$scope:{ctx:w}}}),ee=new ye({props:{title:"AutoformerModel",local:"transformers.AutoformerModel",headingTag:"h2"}}),te=new Me({props:{name:"class transformers.AutoformerModel",anchor:"transformers.AutoformerModel",parameters:[{name:"config",val:": AutoformerConfig"}],parametersDescription:[{anchor:"transformers.AutoformerModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/autoformer#transformers.AutoformerConfig">AutoformerConfig</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/autoformer/modeling_autoformer.py#L1256"}}),oe=new Me({props:{name:"forward",anchor:"transformers.AutoformerModel.forward",parameters:[{name:"past_values",val:": Tensor"},{name:"past_time_features",val:": Tensor"},{name:"past_observed_mask",val:": Tensor"},{name:"static_categorical_features",val:": typing.Optional[torch.Tensor] = None"},{name:"static_real_features",val:": typing.Optional[torch.Tensor] = None"},{name:"future_values",val:": typing.Optional[torch.Tensor] = None"},{name:"future_time_features",val:": typing.Optional[torch.Tensor] = None"},{name:"decoder_attention_mask",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"decoder_head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"cross_attn_head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_outputs",val:": typing.Optional[list[torch.FloatTensor]] = None"},{name:"past_key_values",val:": typing.Optional[list[torch.FloatTensor]] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.Tensor] = None"}],parametersDescription:[{anchor:"transformers.AutoformerModel.forward.past_values",description:`<strong>past_values</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Past values of the time series, that serve as context in order to predict the future. These values may
contain lags, i.e. additional values from the past which are added in order to serve as &#x201C;extra context&#x201D;.
The <code>past_values</code> is what the Transformer encoder gets as input (with optional additional features, such as
<code>static_categorical_features</code>, <code>static_real_features</code>, <code>past_time_features</code>).</p>
<p>The sequence length here is equal to <code>context_length</code> + <code>max(config.lags_sequence)</code>.</p>
<p>Missing values need to be replaced with zeros.`,name:"past_values"},{anchor:"transformers.AutoformerModel.forward.past_time_features",description:`<strong>past_time_features</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, num_features)</code>, <em>optional</em>) &#x2014;
Optional time features, which the model internally will add to <code>past_values</code>. These could be things like
&#x201C;month of year&#x201D;, &#x201C;day of the month&#x201D;, etc. encoded as vectors (for instance as Fourier features). These
could also be so-called &#x201C;age&#x201D; features, which basically help the model know &#x201C;at which point in life&#x201D; a
time-series is. Age features have small values for distant past time steps and increase monotonically the
more we approach the current time step.</p>
<p>These features serve as the &#x201C;positional encodings&#x201D; of the inputs. So contrary to a model like BERT, where
the position encodings are learned from scratch internally as parameters of the model, the Time Series
Transformer requires to provide additional time features.</p>
<p>The Autoformer only learns additional embeddings for <code>static_categorical_features</code>.`,name:"past_time_features"},{anchor:"transformers.AutoformerModel.forward.past_observed_mask",description:`<strong>past_observed_mask</strong> (<code>torch.BoolTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Boolean mask to indicate which <code>past_values</code> were observed and which were missing. Mask values selected in
<code>[0, 1]</code>:</p>
<ul>
<li>1 for values that are <strong>observed</strong>,</li>
<li>0 for values that are <strong>missing</strong> (i.e. NaNs that were replaced by zeros).</li>
</ul>`,name:"past_observed_mask"},{anchor:"transformers.AutoformerModel.forward.static_categorical_features",description:`<strong>static_categorical_features</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, number of static categorical features)</code>, <em>optional</em>) &#x2014;
Optional static categorical features for which the model will learn an embedding, which it will add to the
values of the time series.</p>
<p>Static categorical features are features which have the same value for all time steps (static over time).</p>
<p>A typical example of a static categorical feature is a time series ID.`,name:"static_categorical_features"},{anchor:"transformers.AutoformerModel.forward.static_real_features",description:`<strong>static_real_features</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, number of static real features)</code>, <em>optional</em>) &#x2014;
Optional static real features which the model will add to the values of the time series.</p>
<p>Static real features are features which have the same value for all time steps (static over time).</p>
<p>A typical example of a static real feature is promotion information.`,name:"static_real_features"},{anchor:"transformers.AutoformerModel.forward.future_values",description:`<strong>future_values</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, prediction_length)</code>) &#x2014;
Future values of the time series, that serve as labels for the model. The <code>future_values</code> is what the
Transformer needs to learn to output, given the <code>past_values</code>.</p>
<p>See the demo notebook and code snippets for details.</p>
<p>Missing values need to be replaced with zeros.`,name:"future_values"},{anchor:"transformers.AutoformerModel.forward.future_time_features",description:`<strong>future_time_features</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, prediction_length, num_features)</code>, <em>optional</em>) &#x2014;
Optional time features, which the model internally will add to <code>future_values</code>. These could be things like
&#x201C;month of year&#x201D;, &#x201C;day of the month&#x201D;, etc. encoded as vectors (for instance as Fourier features). These
could also be so-called &#x201C;age&#x201D; features, which basically help the model know &#x201C;at which point in life&#x201D; a
time-series is. Age features have small values for distant past time steps and increase monotonically the
more we approach the current time step.</p>
<p>These features serve as the &#x201C;positional encodings&#x201D; of the inputs. So contrary to a model like BERT, where
the position encodings are learned from scratch internally as parameters of the model, the Time Series
Transformer requires to provide additional features.</p>
<p>The Autoformer only learns additional embeddings for <code>static_categorical_features</code>.`,name:"future_time_features"},{anchor:"transformers.AutoformerModel.forward.decoder_attention_mask",description:`<strong>decoder_attention_mask</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on certain token indices. By default, a causal mask will be used, to
make sure the model can only look at previous inputs in order to predict the future.`,name:"decoder_attention_mask"},{anchor:"transformers.AutoformerModel.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.AutoformerModel.forward.decoder_head_mask",description:`<strong>decoder_head_mask</strong> (<code>torch.Tensor</code> of shape <code>(decoder_layers, decoder_attention_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the attention modules in the decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"decoder_head_mask"},{anchor:"transformers.AutoformerModel.forward.cross_attn_head_mask",description:`<strong>cross_attn_head_mask</strong> (<code>torch.Tensor</code> of shape <code>(decoder_layers, decoder_attention_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the cross-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"cross_attn_head_mask"},{anchor:"transformers.AutoformerModel.forward.encoder_outputs",description:`<strong>encoder_outputs</strong> (<code>tuple(tuple(torch.FloatTensor)</code>, <em>optional</em>) &#x2014;
Tuple consists of <code>last_hidden_state</code>, <code>hidden_states</code> (<em>optional</em>) and <code>attentions</code> (<em>optional</em>)
<code>last_hidden_state</code> of shape <code>(batch_size, sequence_length, hidden_size)</code> (<em>optional</em>) is a sequence of
hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.`,name:"encoder_outputs"},{anchor:"transformers.AutoformerModel.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>list[torch.FloatTensor]</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.AutoformerModel.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.AutoformerModel.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.AutoformerModel.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.AutoformerModel.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.AutoformerModel.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.Tensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/autoformer/modeling_autoformer.py#L1427",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <code>transformers.models.autoformer.modeling_autoformer.AutoformerModelOutput</code> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/autoformer#transformers.AutoformerConfig"
>AutoformerConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>last_hidden_state</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>) — Sequence of hidden-states at the output of the last layer of the decoder of the model.</p>
<p>If <code>past_key_values</code> is used only the last hidden-state of the sequences of shape <code>(batch_size, 1, hidden_size)</code> is output.</p>
</li>
<li>
<p><strong>trend</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>) — Trend tensor for each time series.</p>
</li>
<li>
<p><strong>past_key_values</strong> (<code>tuple(tuple(torch.FloatTensor))</code>, <em>optional</em>, returned when <code>use_cache=True</code> is passed or when <code>config.use_cache=True</code>) — Tuple of <code>tuple(torch.FloatTensor)</code> of length <code>config.n_layers</code>, with each tuple having 2 tensors of shape
<code>(batch_size, num_heads, sequence_length, embed_size_per_head)</code>) and 2 additional tensors of shape
<code>(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)</code>.</p>
<p>Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used (see <code>past_key_values</code> input) to speed up sequential decoding.</p>
</li>
<li>
<p><strong>decoder_hidden_states</strong> (<code>tuple[torch.FloatTensor]</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings, if the model has an embedding layer, +
one for the output of each layer) of shape <code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.</p>
</li>
<li>
<p><strong>decoder_attentions</strong> (<code>tuple[torch.FloatTensor]</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
self-attention heads.</p>
</li>
<li>
<p><strong>cross_attentions</strong> (<code>tuple[torch.FloatTensor]</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the
weighted average in the cross-attention heads.</p>
</li>
<li>
<p><strong>encoder_last_hidden_state</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>, defaults to <code>None</code>) — Sequence of hidden-states at the output of the last layer of the encoder of the model.</p>
</li>
<li>
<p><strong>encoder_hidden_states</strong> (<code>tuple[torch.FloatTensor]</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings, if the model has an embedding layer, +
one for the output of each layer) of shape <code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.</p>
</li>
<li>
<p><strong>encoder_attentions</strong> (<code>tuple[torch.FloatTensor]</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
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
<p><strong>static_features:</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, feature size)</code>, <em>optional</em>) — Static features of each time series’ in a batch which are copied to the covariates at inference time.</p>
</li>
<li>
<p><strong>static_features</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, feature size)</code>, <em>optional</em>, defaults to <code>None</code>) — Static features of each time series’ in a batch which are copied to the covariates at inference time.</p>
</li>
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>transformers.models.autoformer.modeling_autoformer.AutoformerModelOutput</code> or <code>tuple(torch.FloatTensor)</code></p>
`}}),V=new it({props:{$$slots:{default:[Xt]},$$scope:{ctx:w}}}),B=new Ge({props:{anchor:"transformers.AutoformerModel.forward.example",$$slots:{default:[Wt]},$$scope:{ctx:w}}}),ne=new ye({props:{title:"AutoformerForPrediction",local:"transformers.AutoformerForPrediction",headingTag:"h2"}}),ae=new Me({props:{name:"class transformers.AutoformerForPrediction",anchor:"transformers.AutoformerForPrediction",parameters:[{name:"config",val:": AutoformerConfig"}],parametersDescription:[{anchor:"transformers.AutoformerForPrediction.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/autoformer#transformers.AutoformerConfig">AutoformerConfig</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/autoformer/modeling_autoformer.py#L1654"}}),se=new Me({props:{name:"forward",anchor:"transformers.AutoformerForPrediction.forward",parameters:[{name:"past_values",val:": Tensor"},{name:"past_time_features",val:": Tensor"},{name:"past_observed_mask",val:": Tensor"},{name:"static_categorical_features",val:": typing.Optional[torch.Tensor] = None"},{name:"static_real_features",val:": typing.Optional[torch.Tensor] = None"},{name:"future_values",val:": typing.Optional[torch.Tensor] = None"},{name:"future_time_features",val:": typing.Optional[torch.Tensor] = None"},{name:"future_observed_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"decoder_attention_mask",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"decoder_head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"cross_attn_head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_outputs",val:": typing.Optional[list[torch.FloatTensor]] = None"},{name:"past_key_values",val:": typing.Optional[list[torch.FloatTensor]] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.AutoformerForPrediction.forward.past_values",description:`<strong>past_values</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Past values of the time series, that serve as context in order to predict the future. These values may
contain lags, i.e. additional values from the past which are added in order to serve as &#x201C;extra context&#x201D;.
The <code>past_values</code> is what the Transformer encoder gets as input (with optional additional features, such as
<code>static_categorical_features</code>, <code>static_real_features</code>, <code>past_time_features</code>).</p>
<p>The sequence length here is equal to <code>context_length</code> + <code>max(config.lags_sequence)</code>.</p>
<p>Missing values need to be replaced with zeros.`,name:"past_values"},{anchor:"transformers.AutoformerForPrediction.forward.past_time_features",description:`<strong>past_time_features</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, num_features)</code>, <em>optional</em>) &#x2014;
Optional time features, which the model internally will add to <code>past_values</code>. These could be things like
&#x201C;month of year&#x201D;, &#x201C;day of the month&#x201D;, etc. encoded as vectors (for instance as Fourier features). These
could also be so-called &#x201C;age&#x201D; features, which basically help the model know &#x201C;at which point in life&#x201D; a
time-series is. Age features have small values for distant past time steps and increase monotonically the
more we approach the current time step.</p>
<p>These features serve as the &#x201C;positional encodings&#x201D; of the inputs. So contrary to a model like BERT, where
the position encodings are learned from scratch internally as parameters of the model, the Time Series
Transformer requires to provide additional time features.</p>
<p>The Autoformer only learns additional embeddings for <code>static_categorical_features</code>.`,name:"past_time_features"},{anchor:"transformers.AutoformerForPrediction.forward.past_observed_mask",description:`<strong>past_observed_mask</strong> (<code>torch.BoolTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Boolean mask to indicate which <code>past_values</code> were observed and which were missing. Mask values selected in
<code>[0, 1]</code>:</p>
<ul>
<li>1 for values that are <strong>observed</strong>,</li>
<li>0 for values that are <strong>missing</strong> (i.e. NaNs that were replaced by zeros).</li>
</ul>`,name:"past_observed_mask"},{anchor:"transformers.AutoformerForPrediction.forward.static_categorical_features",description:`<strong>static_categorical_features</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, number of static categorical features)</code>, <em>optional</em>) &#x2014;
Optional static categorical features for which the model will learn an embedding, which it will add to the
values of the time series.</p>
<p>Static categorical features are features which have the same value for all time steps (static over time).</p>
<p>A typical example of a static categorical feature is a time series ID.`,name:"static_categorical_features"},{anchor:"transformers.AutoformerForPrediction.forward.static_real_features",description:`<strong>static_real_features</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, number of static real features)</code>, <em>optional</em>) &#x2014;
Optional static real features which the model will add to the values of the time series.</p>
<p>Static real features are features which have the same value for all time steps (static over time).</p>
<p>A typical example of a static real feature is promotion information.`,name:"static_real_features"},{anchor:"transformers.AutoformerForPrediction.forward.future_values",description:`<strong>future_values</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, prediction_length)</code>) &#x2014;
Future values of the time series, that serve as labels for the model. The <code>future_values</code> is what the
Transformer needs to learn to output, given the <code>past_values</code>.</p>
<p>See the demo notebook and code snippets for details.</p>
<p>Missing values need to be replaced with zeros.`,name:"future_values"},{anchor:"transformers.AutoformerForPrediction.forward.future_time_features",description:`<strong>future_time_features</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, prediction_length, num_features)</code>, <em>optional</em>) &#x2014;
Optional time features, which the model internally will add to <code>future_values</code>. These could be things like
&#x201C;month of year&#x201D;, &#x201C;day of the month&#x201D;, etc. encoded as vectors (for instance as Fourier features). These
could also be so-called &#x201C;age&#x201D; features, which basically help the model know &#x201C;at which point in life&#x201D; a
time-series is. Age features have small values for distant past time steps and increase monotonically the
more we approach the current time step.</p>
<p>These features serve as the &#x201C;positional encodings&#x201D; of the inputs. So contrary to a model like BERT, where
the position encodings are learned from scratch internally as parameters of the model, the Time Series
Transformer requires to provide additional features.</p>
<p>The Autoformer only learns additional embeddings for <code>static_categorical_features</code>.`,name:"future_time_features"},{anchor:"transformers.AutoformerForPrediction.forward.future_observed_mask",description:`<strong>future_observed_mask</strong> (<code>torch.BoolTensor</code> of shape <code>(batch_size, sequence_length)</code> or <code>(batch_size, sequence_length, input_size)</code>, <em>optional</em>) &#x2014;
Boolean mask to indicate which <code>future_values</code> were observed and which were missing. Mask values selected
in <code>[0, 1]</code>:</p>
<ul>
<li>1 for values that are <strong>observed</strong>,</li>
<li>0 for values that are <strong>missing</strong> (i.e. NaNs that were replaced by zeros).</li>
</ul>
<p>This mask is used to filter out missing values for the final loss calculation.`,name:"future_observed_mask"},{anchor:"transformers.AutoformerForPrediction.forward.decoder_attention_mask",description:`<strong>decoder_attention_mask</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on certain token indices. By default, a causal mask will be used, to
make sure the model can only look at previous inputs in order to predict the future.`,name:"decoder_attention_mask"},{anchor:"transformers.AutoformerForPrediction.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.AutoformerForPrediction.forward.decoder_head_mask",description:`<strong>decoder_head_mask</strong> (<code>torch.Tensor</code> of shape <code>(decoder_layers, decoder_attention_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the attention modules in the decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"decoder_head_mask"},{anchor:"transformers.AutoformerForPrediction.forward.cross_attn_head_mask",description:`<strong>cross_attn_head_mask</strong> (<code>torch.Tensor</code> of shape <code>(decoder_layers, decoder_attention_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the cross-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"cross_attn_head_mask"},{anchor:"transformers.AutoformerForPrediction.forward.encoder_outputs",description:`<strong>encoder_outputs</strong> (<code>tuple(tuple(torch.FloatTensor)</code>, <em>optional</em>) &#x2014;
Tuple consists of <code>last_hidden_state</code>, <code>hidden_states</code> (<em>optional</em>) and <code>attentions</code> (<em>optional</em>)
<code>last_hidden_state</code> of shape <code>(batch_size, sequence_length, hidden_size)</code> (<em>optional</em>) is a sequence of
hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.`,name:"encoder_outputs"},{anchor:"transformers.AutoformerForPrediction.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>list[torch.FloatTensor]</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.AutoformerForPrediction.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.AutoformerForPrediction.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.AutoformerForPrediction.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.AutoformerForPrediction.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/autoformer/modeling_autoformer.py#L1694",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqTSPredictionOutput"
>transformers.modeling_outputs.Seq2SeqTSPredictionOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/autoformer#transformers.AutoformerConfig"
>AutoformerConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>loss</strong> (<code>torch.FloatTensor</code> of shape <code>(1,)</code>, <em>optional</em>, returned when a <code>future_values</code> is provided) — Distributional loss.</p>
</li>
<li>
<p><strong>params</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_samples, num_params)</code>) — Parameters of the chosen distribution.</p>
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
<p>Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.</p>
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
<p>Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.</p>
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
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqTSPredictionOutput"
>transformers.modeling_outputs.Seq2SeqTSPredictionOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),q=new it({props:{$$slots:{default:[zt]},$$scope:{ctx:w}}}),R=new Ge({props:{anchor:"transformers.AutoformerForPrediction.forward.example",$$slots:{default:[Vt]},$$scope:{ctx:w}}}),G=new it({props:{$$slots:{default:[qt]},$$scope:{ctx:w}}}),re=new At({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/autoformer.md"}}),{c(){t=d("meta"),u=l(),s=d("p"),i=l(),h=d("p"),h.innerHTML=o,g=l(),_(X.$$.fragment),Te=l(),W=d("div"),W.innerHTML=lt,we=l(),_(Y.$$.fragment),je=l(),I=d("p"),I.innerHTML=ct,Je=l(),E=d("p"),E.textContent=dt,ke=l(),H=d("p"),H.textContent=mt,Ce=l(),S=d("p"),S.innerHTML=pt,xe=l(),P=d("p"),P.innerHTML=ut,Ue=l(),_(Q.$$.fragment),Ze=l(),D=d("p"),D.textContent=ht,$e=l(),L=d("ul"),L.innerHTML=ft,Ae=l(),_(O.$$.fragment),Fe=l(),C=d("div"),_(K.$$.fragment),Ye=l(),le=d("p"),le.innerHTML=gt,Ie=l(),ce=d("p"),ce.innerHTML=_t,Ee=l(),_(z.$$.fragment),Xe=l(),_(ee.$$.fragment),We=l(),j=d("div"),_(te.$$.fragment),He=l(),de=d("p"),de.textContent=yt,Se=l(),me=d("p"),me.innerHTML=bt,Pe=l(),pe=d("p"),pe.innerHTML=vt,Qe=l(),$=d("div"),_(oe.$$.fragment),De=l(),ue=d("p"),ue.innerHTML=Mt,Le=l(),_(V.$$.fragment),Oe=l(),_(B.$$.fragment),ze=l(),_(ne.$$.fragment),Ve=l(),J=d("div"),_(ae.$$.fragment),Ke=l(),he=d("p"),he.textContent=Tt,et=l(),fe=d("p"),fe.innerHTML=wt,tt=l(),ge=d("p"),ge.innerHTML=jt,ot=l(),k=d("div"),_(se.$$.fragment),nt=l(),_e=d("p"),_e.innerHTML=Jt,at=l(),_(q.$$.fragment),st=l(),_(R.$$.fragment),rt=l(),_(G.$$.fragment),Be=l(),_(re.$$.fragment),qe=l(),ve=d("p"),this.h()},l(e){const n=$t("svelte-u9bgzb",document.head);t=m(n,"META",{name:!0,content:!0}),n.forEach(a),u=c(e),s=m(e,"P",{}),ie(s).forEach(a),i=c(e),h=m(e,"P",{"data-svelte-h":!0}),f(h)!=="svelte-1j8pye8"&&(h.innerHTML=o),g=c(e),y(X.$$.fragment,e),Te=c(e),W=m(e,"DIV",{class:!0,"data-svelte-h":!0}),f(W)!=="svelte-13t8s2t"&&(W.innerHTML=lt),we=c(e),y(Y.$$.fragment,e),je=c(e),I=m(e,"P",{"data-svelte-h":!0}),f(I)!=="svelte-7uqw5b"&&(I.innerHTML=ct),Je=c(e),E=m(e,"P",{"data-svelte-h":!0}),f(E)!=="svelte-1ki0fyb"&&(E.textContent=dt),ke=c(e),H=m(e,"P",{"data-svelte-h":!0}),f(H)!=="svelte-vfdo9a"&&(H.textContent=mt),Ce=c(e),S=m(e,"P",{"data-svelte-h":!0}),f(S)!=="svelte-o037y5"&&(S.innerHTML=pt),xe=c(e),P=m(e,"P",{"data-svelte-h":!0}),f(P)!=="svelte-1twzwdn"&&(P.innerHTML=ut),Ue=c(e),y(Q.$$.fragment,e),Ze=c(e),D=m(e,"P",{"data-svelte-h":!0}),f(D)!=="svelte-1e7xzkp"&&(D.textContent=ht),$e=c(e),L=m(e,"UL",{"data-svelte-h":!0}),f(L)!=="svelte-s0tqjc"&&(L.innerHTML=ft),Ae=c(e),y(O.$$.fragment,e),Fe=c(e),C=m(e,"DIV",{class:!0});var A=ie(C);y(K.$$.fragment,A),Ye=c(A),le=m(A,"P",{"data-svelte-h":!0}),f(le)!=="svelte-1agze89"&&(le.innerHTML=gt),Ie=c(A),ce=m(A,"P",{"data-svelte-h":!0}),f(ce)!=="svelte-1ynyot8"&&(ce.innerHTML=_t),Ee=c(A),y(z.$$.fragment,A),A.forEach(a),Xe=c(e),y(ee.$$.fragment,e),We=c(e),j=m(e,"DIV",{class:!0});var x=ie(j);y(te.$$.fragment,x),He=c(x),de=m(x,"P",{"data-svelte-h":!0}),f(de)!=="svelte-c8j2uu"&&(de.textContent=yt),Se=c(x),me=m(x,"P",{"data-svelte-h":!0}),f(me)!=="svelte-q52n56"&&(me.innerHTML=bt),Pe=c(x),pe=m(x,"P",{"data-svelte-h":!0}),f(pe)!=="svelte-hswkmf"&&(pe.innerHTML=vt),Qe=c(x),$=m(x,"DIV",{class:!0});var F=ie($);y(oe.$$.fragment,F),De=c(F),ue=m(F,"P",{"data-svelte-h":!0}),f(ue)!=="svelte-19c6i1l"&&(ue.innerHTML=Mt),Le=c(F),y(V.$$.fragment,F),Oe=c(F),y(B.$$.fragment,F),F.forEach(a),x.forEach(a),ze=c(e),y(ne.$$.fragment,e),Ve=c(e),J=m(e,"DIV",{class:!0});var U=ie(J);y(ae.$$.fragment,U),Ke=c(U),he=m(U,"P",{"data-svelte-h":!0}),f(he)!=="svelte-1gxg89a"&&(he.textContent=Tt),et=c(U),fe=m(U,"P",{"data-svelte-h":!0}),f(fe)!=="svelte-q52n56"&&(fe.innerHTML=wt),tt=c(U),ge=m(U,"P",{"data-svelte-h":!0}),f(ge)!=="svelte-hswkmf"&&(ge.innerHTML=jt),ot=c(U),k=m(U,"DIV",{class:!0});var Z=ie(k);y(se.$$.fragment,Z),nt=c(Z),_e=m(Z,"P",{"data-svelte-h":!0}),f(_e)!=="svelte-nnsxup"&&(_e.innerHTML=Jt),at=c(Z),y(q.$$.fragment,Z),st=c(Z),y(R.$$.fragment,Z),rt=c(Z),y(G.$$.fragment,Z),Z.forEach(a),U.forEach(a),Be=c(e),y(re.$$.fragment,e),qe=c(e),ve=m(e,"P",{}),ie(ve).forEach(a),this.h()},h(){N(t,"name","hf:doc:metadata"),N(t,"content",Gt),N(W,"class","flex flex-wrap space-x-1"),N(C,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),N($,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),N(j,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),N(k,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),N(J,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,n){p(document.head,t),r(e,u,n),r(e,s,n),r(e,i,n),r(e,h,n),r(e,g,n),b(X,e,n),r(e,Te,n),r(e,W,n),r(e,we,n),b(Y,e,n),r(e,je,n),r(e,I,n),r(e,Je,n),r(e,E,n),r(e,ke,n),r(e,H,n),r(e,Ce,n),r(e,S,n),r(e,xe,n),r(e,P,n),r(e,Ue,n),b(Q,e,n),r(e,Ze,n),r(e,D,n),r(e,$e,n),r(e,L,n),r(e,Ae,n),b(O,e,n),r(e,Fe,n),r(e,C,n),b(K,C,null),p(C,Ye),p(C,le),p(C,Ie),p(C,ce),p(C,Ee),b(z,C,null),r(e,Xe,n),b(ee,e,n),r(e,We,n),r(e,j,n),b(te,j,null),p(j,He),p(j,de),p(j,Se),p(j,me),p(j,Pe),p(j,pe),p(j,Qe),p(j,$),b(oe,$,null),p($,De),p($,ue),p($,Le),b(V,$,null),p($,Oe),b(B,$,null),r(e,ze,n),b(ne,e,n),r(e,Ve,n),r(e,J,n),b(ae,J,null),p(J,Ke),p(J,he),p(J,et),p(J,fe),p(J,tt),p(J,ge),p(J,ot),p(J,k),b(se,k,null),p(k,nt),p(k,_e),p(k,at),b(q,k,null),p(k,st),b(R,k,null),p(k,rt),b(G,k,null),r(e,Be,n),b(re,e,n),r(e,qe,n),r(e,ve,n),Re=!0},p(e,[n]){const A={};n&2&&(A.$$scope={dirty:n,ctx:e}),z.$set(A);const x={};n&2&&(x.$$scope={dirty:n,ctx:e}),V.$set(x);const F={};n&2&&(F.$$scope={dirty:n,ctx:e}),B.$set(F);const U={};n&2&&(U.$$scope={dirty:n,ctx:e}),q.$set(U);const Z={};n&2&&(Z.$$scope={dirty:n,ctx:e}),R.$set(Z);const kt={};n&2&&(kt.$$scope={dirty:n,ctx:e}),G.$set(kt)},i(e){Re||(v(X.$$.fragment,e),v(Y.$$.fragment,e),v(Q.$$.fragment,e),v(O.$$.fragment,e),v(K.$$.fragment,e),v(z.$$.fragment,e),v(ee.$$.fragment,e),v(te.$$.fragment,e),v(oe.$$.fragment,e),v(V.$$.fragment,e),v(B.$$.fragment,e),v(ne.$$.fragment,e),v(ae.$$.fragment,e),v(se.$$.fragment,e),v(q.$$.fragment,e),v(R.$$.fragment,e),v(G.$$.fragment,e),v(re.$$.fragment,e),Re=!0)},o(e){M(X.$$.fragment,e),M(Y.$$.fragment,e),M(Q.$$.fragment,e),M(O.$$.fragment,e),M(K.$$.fragment,e),M(z.$$.fragment,e),M(ee.$$.fragment,e),M(te.$$.fragment,e),M(oe.$$.fragment,e),M(V.$$.fragment,e),M(B.$$.fragment,e),M(ne.$$.fragment,e),M(ae.$$.fragment,e),M(se.$$.fragment,e),M(q.$$.fragment,e),M(R.$$.fragment,e),M(G.$$.fragment,e),M(re.$$.fragment,e),Re=!1},d(e){e&&(a(u),a(s),a(i),a(h),a(g),a(Te),a(W),a(we),a(je),a(I),a(Je),a(E),a(ke),a(H),a(Ce),a(S),a(xe),a(P),a(Ue),a(Ze),a(D),a($e),a(L),a(Ae),a(Fe),a(C),a(Xe),a(We),a(j),a(ze),a(Ve),a(J),a(Be),a(qe),a(ve)),a(t),T(X,e),T(Y,e),T(Q,e),T(O,e),T(K),T(z),T(ee,e),T(te),T(oe),T(V),T(B),T(ne,e),T(ae),T(se),T(q),T(R),T(G),T(re,e)}}}const Gt='{"title":"Autoformer","local":"autoformer","sections":[{"title":"Overview","local":"overview","sections":[],"depth":2},{"title":"Resources","local":"resources","sections":[],"depth":2},{"title":"AutoformerConfig","local":"transformers.AutoformerConfig","sections":[],"depth":2},{"title":"AutoformerModel","local":"transformers.AutoformerModel","sections":[],"depth":2},{"title":"AutoformerForPrediction","local":"transformers.AutoformerForPrediction","sections":[],"depth":2}],"depth":1}';function Nt(w){return xt(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class Dt extends Ut{constructor(t){super(),Zt(this,t,Nt,Rt,Ct,{})}}export{Dt as component};
