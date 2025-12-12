import{s as is,o as cs,n as E}from"../chunks/scheduler.18a86fab.js";import{S as ls,i as ps,g as i,s,r as m,A as hs,h as c,f as t,c as r,j as N,x as u,u as f,k as M,y as n,a as l,v as g,d as _,t as T,w as y}from"../chunks/index.98837b22.js";import{T as St}from"../chunks/Tip.77304350.js";import{D as z}from"../chunks/Docstring.a1ef7999.js";import{C as Bt}from"../chunks/CodeBlock.8d0c2e8a.js";import{E as Vt}from"../chunks/ExampleCodeBlock.8c3ee1f9.js";import{H as W,E as us}from"../chunks/getInferenceSnippets.06c2775f.js";function ms(w){let a,b=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){a=i("p"),a.innerHTML=b},l(p){a=c(p,"P",{"data-svelte-h":!0}),u(a)!=="svelte-fincs2"&&(a.innerHTML=b)},m(p,h){l(p,a,h)},p:E,d(p){p&&t(a)}}}function fs(w){let a,b="Example:",p,h,v;return h=new Bt({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBQcm9waGV0TmV0TW9kZWwlMEElMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJtaWNyb3NvZnQlMkZwcm9waGV0bmV0LWxhcmdlLXVuY2FzZWQlMjIpJTBBbW9kZWwlMjAlM0QlMjBQcm9waGV0TmV0TW9kZWwuZnJvbV9wcmV0cmFpbmVkKCUyMm1pY3Jvc29mdCUyRnByb3BoZXRuZXQtbGFyZ2UtdW5jYXNlZCUyMiklMEElMEFpbnB1dF9pZHMlMjAlM0QlMjB0b2tlbml6ZXIoJTBBJTIwJTIwJTIwJTIwJTIyU3R1ZGllcyUyMGhhdmUlMjBiZWVuJTIwc2hvd24lMjB0aGF0JTIwb3duaW5nJTIwYSUyMGRvZyUyMGlzJTIwZ29vZCUyMGZvciUyMHlvdSUyMiUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIlMEEpLmlucHV0X2lkcyUyMCUyMCUyMyUyMEJhdGNoJTIwc2l6ZSUyMDElMEFkZWNvZGVyX2lucHV0X2lkcyUyMCUzRCUyMHRva2VuaXplciglMjJTdHVkaWVzJTIwc2hvdyUyMHRoYXQlMjIlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKS5pbnB1dF9pZHMlMjAlMjAlMjMlMjBCYXRjaCUyMHNpemUlMjAxJTBBb3V0cHV0cyUyMCUzRCUyMG1vZGVsKGlucHV0X2lkcyUzRGlucHV0X2lkcyUyQyUyMGRlY29kZXJfaW5wdXRfaWRzJTNEZGVjb2Rlcl9pbnB1dF9pZHMpJTBBJTBBbGFzdF9oaWRkZW5fc3RhdGVzJTIwJTNEJTIwb3V0cHV0cy5sYXN0X2hpZGRlbl9zdGF0ZSUyMCUyMCUyMyUyMG1haW4lMjBzdHJlYW0lMjBoaWRkZW4lMjBzdGF0ZXMlMEFsYXN0X2hpZGRlbl9zdGF0ZXNfbmdyYW0lMjAlM0QlMjBvdXRwdXRzLmxhc3RfaGlkZGVuX3N0YXRlX25ncmFtJTIwJTIwJTIzJTIwcHJlZGljdCUyMGhpZGRlbiUyMHN0YXRlcw==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, ProphetNetModel

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;microsoft/prophetnet-large-uncased&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = ProphetNetModel.from_pretrained(<span class="hljs-string">&quot;microsoft/prophetnet-large-uncased&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>input_ids = tokenizer(
<span class="hljs-meta">... </span>    <span class="hljs-string">&quot;Studies have been shown that owning a dog is good for you&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>
<span class="hljs-meta">... </span>).input_ids  <span class="hljs-comment"># Batch size 1</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>decoder_input_ids = tokenizer(<span class="hljs-string">&quot;Studies show that&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>).input_ids  <span class="hljs-comment"># Batch size 1</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)

<span class="hljs-meta">&gt;&gt;&gt; </span>last_hidden_states = outputs.last_hidden_state  <span class="hljs-comment"># main stream hidden states</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>last_hidden_states_ngram = outputs.last_hidden_state_ngram  <span class="hljs-comment"># predict hidden states</span>`,wrap:!1}}),{c(){a=i("p"),a.textContent=b,p=s(),m(h.$$.fragment)},l(d){a=c(d,"P",{"data-svelte-h":!0}),u(a)!=="svelte-11lpom8"&&(a.textContent=b),p=r(d),f(h.$$.fragment,d)},m(d,k){l(d,a,k),l(d,p,k),g(h,d,k),v=!0},p:E,i(d){v||(_(h.$$.fragment,d),v=!0)},o(d){T(h.$$.fragment,d),v=!1},d(d){d&&(t(a),t(p)),y(h,d)}}}function gs(w){let a,b=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){a=i("p"),a.innerHTML=b},l(p){a=c(p,"P",{"data-svelte-h":!0}),u(a)!=="svelte-fincs2"&&(a.innerHTML=b)},m(p,h){l(p,a,h)},p:E,d(p){p&&t(a)}}}function _s(w){let a,b="Example:",p,h,v;return h=new Bt({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBQcm9waGV0TmV0RW5jb2RlciUwQWltcG9ydCUyMHRvcmNoJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIybWljcm9zb2Z0JTJGcHJvcGhldG5ldC1sYXJnZS11bmNhc2VkJTIyKSUwQW1vZGVsJTIwJTNEJTIwUHJvcGhldE5ldEVuY29kZXIuZnJvbV9wcmV0cmFpbmVkKCUyMnBhdHJpY2t2b25wbGF0ZW4lMkZwcm9waGV0bmV0LWxhcmdlLXVuY2FzZWQtc3RhbmRhbG9uZSUyMiklMEFpbnB1dHMlMjAlM0QlMjB0b2tlbml6ZXIoJTIySGVsbG8lMkMlMjBteSUyMGRvZyUyMGlzJTIwY3V0ZSUyMiUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpJTBBb3V0cHV0cyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzKSUwQSUwQWxhc3RfaGlkZGVuX3N0YXRlcyUyMCUzRCUyMG91dHB1dHMubGFzdF9oaWRkZW5fc3RhdGU=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, ProphetNetEncoder
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;microsoft/prophetnet-large-uncased&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = ProphetNetEncoder.from_pretrained(<span class="hljs-string">&quot;patrickvonplaten/prophetnet-large-uncased-standalone&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs)

<span class="hljs-meta">&gt;&gt;&gt; </span>last_hidden_states = outputs.last_hidden_state`,wrap:!1}}),{c(){a=i("p"),a.textContent=b,p=s(),m(h.$$.fragment)},l(d){a=c(d,"P",{"data-svelte-h":!0}),u(a)!=="svelte-11lpom8"&&(a.textContent=b),p=r(d),f(h.$$.fragment,d)},m(d,k){l(d,a,k),l(d,p,k),g(h,d,k),v=!0},p:E,i(d){v||(_(h.$$.fragment,d),v=!0)},o(d){T(h.$$.fragment,d),v=!1},d(d){d&&(t(a),t(p)),y(h,d)}}}function Ts(w){let a,b=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){a=i("p"),a.innerHTML=b},l(p){a=c(p,"P",{"data-svelte-h":!0}),u(a)!=="svelte-fincs2"&&(a.innerHTML=b)},m(p,h){l(p,a,h)},p:E,d(p){p&&t(a)}}}function ys(w){let a,b="Example:",p,h,v;return h=new Bt({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBQcm9waGV0TmV0RGVjb2RlciUwQWltcG9ydCUyMHRvcmNoJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIybWljcm9zb2Z0JTJGcHJvcGhldG5ldC1sYXJnZS11bmNhc2VkJTIyKSUwQW1vZGVsJTIwJTNEJTIwUHJvcGhldE5ldERlY29kZXIuZnJvbV9wcmV0cmFpbmVkKCUyMm1pY3Jvc29mdCUyRnByb3BoZXRuZXQtbGFyZ2UtdW5jYXNlZCUyMiUyQyUyMGFkZF9jcm9zc19hdHRlbnRpb24lM0RGYWxzZSklMEFpbnB1dHMlMjAlM0QlMjB0b2tlbml6ZXIoJTIySGVsbG8lMkMlMjBteSUyMGRvZyUyMGlzJTIwY3V0ZSUyMiUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpJTBBb3V0cHV0cyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzKSUwQSUwQWxhc3RfaGlkZGVuX3N0YXRlcyUyMCUzRCUyMG91dHB1dHMubGFzdF9oaWRkZW5fc3RhdGU=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, ProphetNetDecoder
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;microsoft/prophetnet-large-uncased&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = ProphetNetDecoder.from_pretrained(<span class="hljs-string">&quot;microsoft/prophetnet-large-uncased&quot;</span>, add_cross_attention=<span class="hljs-literal">False</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs)

<span class="hljs-meta">&gt;&gt;&gt; </span>last_hidden_states = outputs.last_hidden_state`,wrap:!1}}),{c(){a=i("p"),a.textContent=b,p=s(),m(h.$$.fragment)},l(d){a=c(d,"P",{"data-svelte-h":!0}),u(a)!=="svelte-11lpom8"&&(a.textContent=b),p=r(d),f(h.$$.fragment,d)},m(d,k){l(d,a,k),l(d,p,k),g(h,d,k),v=!0},p:E,i(d){v||(_(h.$$.fragment,d),v=!0)},o(d){T(h.$$.fragment,d),v=!1},d(d){d&&(t(a),t(p)),y(h,d)}}}function bs(w){let a,b=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){a=i("p"),a.innerHTML=b},l(p){a=c(p,"P",{"data-svelte-h":!0}),u(a)!=="svelte-fincs2"&&(a.innerHTML=b)},m(p,h){l(p,a,h)},p:E,d(p){p&&t(a)}}}function vs(w){let a,b="Example:",p,h,v;return h=new Bt({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBQcm9waGV0TmV0Rm9yQ29uZGl0aW9uYWxHZW5lcmF0aW9uJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIybWljcm9zb2Z0JTJGcHJvcGhldG5ldC1sYXJnZS11bmNhc2VkJTIyKSUwQW1vZGVsJTIwJTNEJTIwUHJvcGhldE5ldEZvckNvbmRpdGlvbmFsR2VuZXJhdGlvbi5mcm9tX3ByZXRyYWluZWQoJTIybWljcm9zb2Z0JTJGcHJvcGhldG5ldC1sYXJnZS11bmNhc2VkJTIyKSUwQSUwQWlucHV0X2lkcyUyMCUzRCUyMHRva2VuaXplciglMEElMjAlMjAlMjAlMjAlMjJTdHVkaWVzJTIwaGF2ZSUyMGJlZW4lMjBzaG93biUyMHRoYXQlMjBvd25pbmclMjBhJTIwZG9nJTIwaXMlMjBnb29kJTIwZm9yJTIweW91JTIyJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiUwQSkuaW5wdXRfaWRzJTIwJTIwJTIzJTIwQmF0Y2glMjBzaXplJTIwMSUwQWRlY29kZXJfaW5wdXRfaWRzJTIwJTNEJTIwdG9rZW5pemVyKCUyMlN0dWRpZXMlMjBzaG93JTIwdGhhdCUyMiUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpLmlucHV0X2lkcyUyMCUyMCUyMyUyMEJhdGNoJTIwc2l6ZSUyMDElMEFvdXRwdXRzJTIwJTNEJTIwbW9kZWwoaW5wdXRfaWRzJTNEaW5wdXRfaWRzJTJDJTIwZGVjb2Rlcl9pbnB1dF9pZHMlM0RkZWNvZGVyX2lucHV0X2lkcyklMEElMEFsb2dpdHNfbmV4dF90b2tlbiUyMCUzRCUyMG91dHB1dHMubG9naXRzJTIwJTIwJTIzJTIwbG9naXRzJTIwdG8lMjBwcmVkaWN0JTIwbmV4dCUyMHRva2VuJTIwYXMlMjB1c3VhbCUwQWxvZ2l0c19uZ3JhbV9uZXh0X3Rva2VucyUyMCUzRCUyMG91dHB1dHMubG9naXRzX25ncmFtJTIwJTIwJTIzJTIwbG9naXRzJTIwdG8lMjBwcmVkaWN0JTIwMm5kJTJDJTIwM3JkJTJDJTIwLi4uJTIwbmV4dCUyMHRva2Vucw==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, ProphetNetForConditionalGeneration

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;microsoft/prophetnet-large-uncased&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = ProphetNetForConditionalGeneration.from_pretrained(<span class="hljs-string">&quot;microsoft/prophetnet-large-uncased&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>input_ids = tokenizer(
<span class="hljs-meta">... </span>    <span class="hljs-string">&quot;Studies have been shown that owning a dog is good for you&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>
<span class="hljs-meta">... </span>).input_ids  <span class="hljs-comment"># Batch size 1</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>decoder_input_ids = tokenizer(<span class="hljs-string">&quot;Studies show that&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>).input_ids  <span class="hljs-comment"># Batch size 1</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)

<span class="hljs-meta">&gt;&gt;&gt; </span>logits_next_token = outputs.logits  <span class="hljs-comment"># logits to predict next token as usual</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>logits_ngram_next_tokens = outputs.logits_ngram  <span class="hljs-comment"># logits to predict 2nd, 3rd, ... next tokens</span>`,wrap:!1}}),{c(){a=i("p"),a.textContent=b,p=s(),m(h.$$.fragment)},l(d){a=c(d,"P",{"data-svelte-h":!0}),u(a)!=="svelte-11lpom8"&&(a.textContent=b),p=r(d),f(h.$$.fragment,d)},m(d,k){l(d,a,k),l(d,p,k),g(h,d,k),v=!0},p:E,i(d){v||(_(h.$$.fragment,d),v=!0)},o(d){T(h.$$.fragment,d),v=!1},d(d){d&&(t(a),t(p)),y(h,d)}}}function ks(w){let a,b=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){a=i("p"),a.innerHTML=b},l(p){a=c(p,"P",{"data-svelte-h":!0}),u(a)!=="svelte-fincs2"&&(a.innerHTML=b)},m(p,h){l(p,a,h)},p:E,d(p){p&&t(a)}}}function ws(w){let a,b="Example:",p,h,v;return h=new Bt({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBQcm9waGV0TmV0Rm9yQ2F1c2FsTE0lMEFpbXBvcnQlMjB0b3JjaCUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMm1pY3Jvc29mdCUyRnByb3BoZXRuZXQtbGFyZ2UtdW5jYXNlZCUyMiklMEFtb2RlbCUyMCUzRCUyMFByb3BoZXROZXRGb3JDYXVzYWxMTS5mcm9tX3ByZXRyYWluZWQoJTIybWljcm9zb2Z0JTJGcHJvcGhldG5ldC1sYXJnZS11bmNhc2VkJTIyKSUwQWFzc2VydCUyMG1vZGVsLmNvbmZpZy5pc19kZWNvZGVyJTJDJTIwZiUyMiU3Qm1vZGVsLl9fY2xhc3NfXyU3RCUyMGhhcyUyMHRvJTIwYmUlMjBjb25maWd1cmVkJTIwYXMlMjBhJTIwZGVjb2Rlci4lMjIlMEFpbnB1dHMlMjAlM0QlMjB0b2tlbml6ZXIoJTIySGVsbG8lMkMlMjBteSUyMGRvZyUyMGlzJTIwY3V0ZSUyMiUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpJTBBb3V0cHV0cyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzKSUwQSUwQWxvZ2l0cyUyMCUzRCUyMG91dHB1dHMubG9naXRzJTBBJTBBJTIzJTIwTW9kZWwlMjBjYW4lMjBhbHNvJTIwYmUlMjB1c2VkJTIwd2l0aCUyMEVuY29kZXJEZWNvZGVyJTIwZnJhbWV3b3JrJTBBZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEJlcnRUb2tlbml6ZXIlMkMlMjBFbmNvZGVyRGVjb2Rlck1vZGVsJTJDJTIwQXV0b1Rva2VuaXplciUwQWltcG9ydCUyMHRvcmNoJTBBJTBBdG9rZW5pemVyX2VuYyUyMCUzRCUyMEJlcnRUb2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMmdvb2dsZS1iZXJ0JTJGYmVydC1sYXJnZS11bmNhc2VkJTIyKSUwQXRva2VuaXplcl9kZWMlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJtaWNyb3NvZnQlMkZwcm9waGV0bmV0LWxhcmdlLXVuY2FzZWQlMjIpJTBBbW9kZWwlMjAlM0QlMjBFbmNvZGVyRGVjb2Rlck1vZGVsLmZyb21fZW5jb2Rlcl9kZWNvZGVyX3ByZXRyYWluZWQoJTBBJTIwJTIwJTIwJTIwJTIyZ29vZ2xlLWJlcnQlMkZiZXJ0LWxhcmdlLXVuY2FzZWQlMjIlMkMlMjAlMjJtaWNyb3NvZnQlMkZwcm9waGV0bmV0LWxhcmdlLXVuY2FzZWQlMjIlMEEpJTBBJTBBQVJUSUNMRSUyMCUzRCUyMCglMEElMjAlMjAlMjAlMjAlMjJ0aGUlMjB1cyUyMHN0YXRlJTIwZGVwYXJ0bWVudCUyMHNhaWQlMjB3ZWRuZXNkYXklMjBpdCUyMGhhZCUyMHJlY2VpdmVkJTIwbm8lMjAlMjIlMEElMjAlMjAlMjAlMjAlMjJmb3JtYWwlMjB3b3JkJTIwZnJvbSUyMGJvbGl2aWElMjB0aGF0JTIwaXQlMjB3YXMlMjBleHBlbGxpbmclMjB0aGUlMjB1cyUyMGFtYmFzc2Fkb3IlMjB0aGVyZSUyMCUyMiUwQSUyMCUyMCUyMCUyMCUyMmJ1dCUyMHNhaWQlMjB0aGUlMjBjaGFyZ2VzJTIwbWFkZSUyMGFnYWluc3QlMjBoaW0lMjBhcmUlMjAlNjAlNjAlMjBiYXNlbGVzcyUyMC4lMjIlMEEpJTBBaW5wdXRfaWRzJTIwJTNEJTIwdG9rZW5pemVyX2VuYyhBUlRJQ0xFJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMikuaW5wdXRfaWRzJTBBbGFiZWxzJTIwJTNEJTIwdG9rZW5pemVyX2RlYyglMEElMjAlMjAlMjAlMjAlMjJ1cyUyMHJlamVjdHMlMjBjaGFyZ2VzJTIwYWdhaW5zdCUyMGl0cyUyMGFtYmFzc2Fkb3IlMjBpbiUyMGJvbGl2aWElMjIlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyJTBBKS5pbnB1dF9pZHMlMEFvdXRwdXRzJTIwJTNEJTIwbW9kZWwoaW5wdXRfaWRzJTNEaW5wdXRfaWRzJTJDJTIwZGVjb2Rlcl9pbnB1dF9pZHMlM0RsYWJlbHMlNUIlM0ElMkMlMjAlM0EtMSU1RCUyQyUyMGxhYmVscyUzRGxhYmVscyU1QiUzQSUyQyUyMDElM0ElNUQpJTBBJTBBbG9zcyUyMCUzRCUyMG91dHB1dHMubG9zcw==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, ProphetNetForCausalLM
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;microsoft/prophetnet-large-uncased&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = ProphetNetForCausalLM.from_pretrained(<span class="hljs-string">&quot;microsoft/prophetnet-large-uncased&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">assert</span> model.config.is_decoder, <span class="hljs-string">f&quot;<span class="hljs-subst">{model.__class__}</span> has to be configured as a decoder.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs)

<span class="hljs-meta">&gt;&gt;&gt; </span>logits = outputs.logits

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Model can also be used with EncoderDecoder framework</span>
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> BertTokenizer, EncoderDecoderModel, AutoTokenizer
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer_enc = BertTokenizer.from_pretrained(<span class="hljs-string">&quot;google-bert/bert-large-uncased&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer_dec = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;microsoft/prophetnet-large-uncased&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = EncoderDecoderModel.from_encoder_decoder_pretrained(
<span class="hljs-meta">... </span>    <span class="hljs-string">&quot;google-bert/bert-large-uncased&quot;</span>, <span class="hljs-string">&quot;microsoft/prophetnet-large-uncased&quot;</span>
<span class="hljs-meta">... </span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>ARTICLE = (
<span class="hljs-meta">... </span>    <span class="hljs-string">&quot;the us state department said wednesday it had received no &quot;</span>
<span class="hljs-meta">... </span>    <span class="hljs-string">&quot;formal word from bolivia that it was expelling the us ambassador there &quot;</span>
<span class="hljs-meta">... </span>    <span class="hljs-string">&quot;but said the charges made against him are \`\` baseless .&quot;</span>
<span class="hljs-meta">... </span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>input_ids = tokenizer_enc(ARTICLE, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>).input_ids
<span class="hljs-meta">&gt;&gt;&gt; </span>labels = tokenizer_dec(
<span class="hljs-meta">... </span>    <span class="hljs-string">&quot;us rejects charges against its ambassador in bolivia&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>
<span class="hljs-meta">... </span>).input_ids
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(input_ids=input_ids, decoder_input_ids=labels[:, :-<span class="hljs-number">1</span>], labels=labels[:, <span class="hljs-number">1</span>:])

<span class="hljs-meta">&gt;&gt;&gt; </span>loss = outputs.loss`,wrap:!1}}),{c(){a=i("p"),a.textContent=b,p=s(),m(h.$$.fragment)},l(d){a=c(d,"P",{"data-svelte-h":!0}),u(a)!=="svelte-11lpom8"&&(a.textContent=b),p=r(d),f(h.$$.fragment,d)},m(d,k){l(d,a,k),l(d,p,k),g(h,d,k),v=!0},p:E,i(d){v||(_(h.$$.fragment,d),v=!0)},o(d){T(h.$$.fragment,d),v=!1},d(d){d&&(t(a),t(p)),y(h,d)}}}function Ms(w){let a,b,p,h,v,d="<em>This model was released on 2020-01-13 and added to Hugging Face Transformers on 2020-11-16.</em>",k,Te,Ht,ee,kn='<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/>',Rt,ye,Et,be,wn=`The ProphetNet model was proposed in <a href="https://huggingface.co/papers/2001.04063" rel="nofollow">ProphetNet: Predicting Future N-gram for Sequence-to-Sequence Pre-training,</a> by Yu Yan, Weizhen Qi, Yeyun Gong, Dayiheng Liu, Nan Duan, Jiusheng Chen, Ruofei
Zhang, Ming Zhou on 13 Jan, 2020.`,Dt,ve,Mn=`ProphetNet is an encoder-decoder model and can predict n-future tokens for “ngram” language modeling instead of just
the next token.`,Xt,ke,Nn="The abstract from the paper is the following:",At,we,zn=`<em>In this paper, we present a new sequence-to-sequence pretraining model called ProphetNet, which introduces a novel
self-supervised objective named future n-gram prediction and the proposed n-stream self-attention mechanism. Instead of
the optimization of one-step ahead prediction in traditional sequence-to-sequence model, the ProphetNet is optimized by
n-step ahead prediction which predicts the next n tokens simultaneously based on previous context tokens at each time
step. The future n-gram prediction explicitly encourages the model to plan for the future tokens and prevent
overfitting on strong local correlations. We pre-train ProphetNet using a base scale dataset (16GB) and a large scale
dataset (160GB) respectively. Then we conduct experiments on CNN/DailyMail, Gigaword, and SQuAD 1.1 benchmarks for
abstractive summarization and question generation tasks. Experimental results show that ProphetNet achieves new
state-of-the-art results on all these datasets compared to the models using the same scale pretraining corpus.</em>`,Qt,Me,Pn='The Authors’ code can be found <a href="https://github.com/microsoft/ProphetNet" rel="nofollow">here</a>.',Yt,Ne,Kt,ze,$n=`<li>ProphetNet is a model with absolute position embeddings so it’s usually advised to pad the inputs on the right rather than
the left.</li> <li>The model architecture is based on the original Transformer, but replaces the “standard” self-attention mechanism in the decoder by a main self-attention mechanism and a self and n-stream (predict) self-attention mechanism.</li>`,eo,Pe,to,$e,xn='<li><a href="../tasks/language_modeling">Causal language modeling task guide</a></li> <li><a href="../tasks/translation">Translation task guide</a></li> <li><a href="../tasks/summarization">Summarization task guide</a></li>',oo,xe,no,L,qe,No,ct,qn=`This is the configuration class to store the configuration of a <a href="/docs/transformers/v4.56.2/en/model_doc/prophetnet#transformers.ProphetNetModel">ProphetNetModel</a>. It is used to instantiate a
ProphetNet model according to the specified arguments, defining the model architecture. Instantiating a
configuration with the defaults will yield a similar configuration to that of the ProphetNet
<a href="https://huggingface.co/microsoft/prophetnet-large-uncased" rel="nofollow">microsoft/prophetnet-large-uncased</a> architecture.`,zo,lt,Cn=`Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> and can be used to control the model outputs. Read the
documentation from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> for more information.`,so,Ce,ro,P,Fe,Po,pt,Fn="Construct a ProphetNetTokenizer. Based on WordPiece.",$o,ht,Jn=`This tokenizer inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer">PreTrainedTokenizer</a> which contains most of the main methods. Users should refer to
this superclass for more information regarding those methods.`,xo,S,Je,qo,ut,jn=`Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
adding special tokens. A BERT sequence has the following format:`,Co,mt,Un="<li>single sequence: <code>[CLS] X [SEP]</code></li> <li>pair of sequences: <code>[CLS] A [SEP] B [SEP]</code></li>",Fo,te,je,Jo,ft,In="Converts a sequence of tokens (string) in a single string.",jo,oe,Ue,Uo,gt,On=`Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
special tokens using the tokenizer <code>prepare_for_model</code> method.`,ao,Ie,io,D,Oe,Io,_t,Gn="Base class for sequence-to-sequence language models outputs.",co,X,Ge,Oo,Tt,Zn=`Base class for model encoder’s outputs that also contains : pre-computed hidden states that can speed up sequential
decoding.`,lo,A,Ze,Go,yt,Wn="Base class for model’s outputs that may also contain a past key/values (to speed up sequential decoding).",po,Q,We,Zo,bt,Ln="Base class for model’s outputs that may also contain a past key/values (to speed up sequential decoding).",ho,Le,uo,$,Se,Wo,vt,Sn="The bare Prophetnet Model outputting raw hidden-states without any specific head on top.",Lo,kt,Vn=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,So,wt,Bn=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Vo,U,Ve,Bo,Mt,Hn='The <a href="/docs/transformers/v4.56.2/en/model_doc/prophetnet#transformers.ProphetNetModel">ProphetNetModel</a> forward method, overrides the <code>__call__</code> special method.',Ho,ne,Ro,se,mo,Be,fo,x,He,Eo,Nt,Rn="The standalone encoder part of the ProphetNetModel.",Do,zt,En=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Xo,Pt,Dn=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Ao,I,Re,Qo,$t,Xn='The <a href="/docs/transformers/v4.56.2/en/model_doc/prophetnet#transformers.ProphetNetEncoder">ProphetNetEncoder</a> forward method, overrides the <code>__call__</code> special method.',Yo,re,Ko,ae,go,Ee,_o,q,De,en,xt,An="The standalone decoder part of the ProphetNetModel.",tn,qt,Qn=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,on,Ct,Yn=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,nn,O,Xe,sn,Ft,Kn='The <a href="/docs/transformers/v4.56.2/en/model_doc/prophetnet#transformers.ProphetNetDecoder">ProphetNetDecoder</a> forward method, overrides the <code>__call__</code> special method.',rn,de,an,ie,To,Ae,yo,C,Qe,dn,Jt,es="The ProphetNet Model with a language modeling head. Can be used for sequence generation tasks.",cn,jt,ts=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,ln,Ut,os=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,pn,G,Ye,hn,It,ns='The <a href="/docs/transformers/v4.56.2/en/model_doc/prophetnet#transformers.ProphetNetForConditionalGeneration">ProphetNetForConditionalGeneration</a> forward method, overrides the <code>__call__</code> special method.',un,ce,mn,le,bo,Ke,vo,F,et,fn,Ot,ss="The standalone decoder part of the ProphetNetModel with a lm head on top. The model can be used for causal",gn,Gt,rs=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,_n,Zt,as=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Tn,Z,tt,yn,Wt,ds='The <a href="/docs/transformers/v4.56.2/en/model_doc/prophetnet#transformers.ProphetNetForCausalLM">ProphetNetForCausalLM</a> forward method, overrides the <code>__call__</code> special method.',bn,pe,vn,he,ko,ot,wo,Lt,Mo;return Te=new W({props:{title:"ProphetNet",local:"prophetnet",headingTag:"h1"}}),ye=new W({props:{title:"Overview",local:"overview",headingTag:"h2"}}),Ne=new W({props:{title:"Usage tips",local:"usage-tips",headingTag:"h2"}}),Pe=new W({props:{title:"Resources",local:"resources",headingTag:"h2"}}),xe=new W({props:{title:"ProphetNetConfig",local:"transformers.ProphetNetConfig",headingTag:"h2"}}),qe=new z({props:{name:"class transformers.ProphetNetConfig",anchor:"transformers.ProphetNetConfig",parameters:[{name:"activation_dropout",val:": typing.Optional[float] = 0.1"},{name:"activation_function",val:": typing.Union[str, typing.Callable, NoneType] = 'gelu'"},{name:"vocab_size",val:": typing.Optional[int] = 30522"},{name:"hidden_size",val:": typing.Optional[int] = 1024"},{name:"encoder_ffn_dim",val:": typing.Optional[int] = 4096"},{name:"num_encoder_layers",val:": typing.Optional[int] = 12"},{name:"num_encoder_attention_heads",val:": typing.Optional[int] = 16"},{name:"decoder_ffn_dim",val:": typing.Optional[int] = 4096"},{name:"num_decoder_layers",val:": typing.Optional[int] = 12"},{name:"num_decoder_attention_heads",val:": typing.Optional[int] = 16"},{name:"attention_dropout",val:": typing.Optional[float] = 0.1"},{name:"dropout",val:": typing.Optional[float] = 0.1"},{name:"max_position_embeddings",val:": typing.Optional[int] = 512"},{name:"init_std",val:": typing.Optional[float] = 0.02"},{name:"is_encoder_decoder",val:": typing.Optional[bool] = True"},{name:"add_cross_attention",val:": typing.Optional[bool] = True"},{name:"decoder_start_token_id",val:": typing.Optional[int] = 0"},{name:"ngram",val:": typing.Optional[int] = 2"},{name:"num_buckets",val:": typing.Optional[int] = 32"},{name:"relative_max_distance",val:": typing.Optional[int] = 128"},{name:"disable_ngram_loss",val:": typing.Optional[bool] = False"},{name:"eps",val:": typing.Optional[float] = 0.0"},{name:"use_cache",val:": typing.Optional[bool] = True"},{name:"pad_token_id",val:": typing.Optional[int] = 0"},{name:"bos_token_id",val:": typing.Optional[int] = 1"},{name:"eos_token_id",val:": typing.Optional[int] = 2"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.ProphetNetConfig.activation_dropout",description:`<strong>activation_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout ratio for activations inside the fully connected layer.`,name:"activation_dropout"},{anchor:"transformers.ProphetNetConfig.activation_function",description:`<strong>activation_function</strong> (<code>str</code> or <code>function</code>, <em>optional</em>, defaults to <code>&quot;gelu&quot;</code>) &#x2014;
The non-linear activation function (function or string) in the encoder and pooler. If string, <code>&quot;gelu&quot;</code>,
<code>&quot;relu&quot;</code>, <code>&quot;silu&quot;</code> and <code>&quot;gelu_new&quot;</code> are supported.`,name:"activation_function"},{anchor:"transformers.ProphetNetConfig.vocab_size",description:`<strong>vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 30522) &#x2014;
Vocabulary size of the ProphetNET model. Defines the number of different tokens that can be represented by
the <code>inputs_ids</code> passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/prophetnet#transformers.ProphetNetModel">ProphetNetModel</a>.`,name:"vocab_size"},{anchor:"transformers.ProphetNetConfig.hidden_size",description:`<strong>hidden_size</strong> (<code>int</code>, <em>optional</em>, defaults to 1024) &#x2014;
Dimensionality of the layers and the pooler layer.`,name:"hidden_size"},{anchor:"transformers.ProphetNetConfig.encoder_ffn_dim",description:`<strong>encoder_ffn_dim</strong> (<code>int</code>, <em>optional</em>, defaults to 4096) &#x2014;
Dimensionality of the &#x201C;intermediate&#x201D; (often named feed-forward) layer in decoder.`,name:"encoder_ffn_dim"},{anchor:"transformers.ProphetNetConfig.num_encoder_layers",description:`<strong>num_encoder_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 12) &#x2014;
Number of encoder layers.`,name:"num_encoder_layers"},{anchor:"transformers.ProphetNetConfig.num_encoder_attention_heads",description:`<strong>num_encoder_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 16) &#x2014;
Number of attention heads for each attention layer in the Transformer encoder.`,name:"num_encoder_attention_heads"},{anchor:"transformers.ProphetNetConfig.decoder_ffn_dim",description:`<strong>decoder_ffn_dim</strong> (<code>int</code>, <em>optional</em>, defaults to 4096) &#x2014;
Dimensionality of the <code>intermediate</code> (often named feed-forward) layer in decoder.`,name:"decoder_ffn_dim"},{anchor:"transformers.ProphetNetConfig.num_decoder_layers",description:`<strong>num_decoder_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 12) &#x2014;
Number of decoder layers.`,name:"num_decoder_layers"},{anchor:"transformers.ProphetNetConfig.num_decoder_attention_heads",description:`<strong>num_decoder_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 16) &#x2014;
Number of attention heads for each attention layer in the Transformer decoder.`,name:"num_decoder_attention_heads"},{anchor:"transformers.ProphetNetConfig.attention_dropout",description:`<strong>attention_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout ratio for the attention probabilities.`,name:"attention_dropout"},{anchor:"transformers.ProphetNetConfig.dropout",description:`<strong>dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.`,name:"dropout"},{anchor:"transformers.ProphetNetConfig.max_position_embeddings",description:`<strong>max_position_embeddings</strong> (<code>int</code>, <em>optional</em>, defaults to 512) &#x2014;
The maximum sequence length that this model might ever be used with. Typically set this to something large
just in case (e.g., 512 or 1024 or 2048).`,name:"max_position_embeddings"},{anchor:"transformers.ProphetNetConfig.init_std",description:`<strong>init_std</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.`,name:"init_std"},{anchor:"transformers.ProphetNetConfig.add_cross_attention",description:`<strong>add_cross_attention</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether cross-attention layers should be added to the model.`,name:"add_cross_attention"},{anchor:"transformers.ProphetNetConfig.is_encoder_decoder",description:`<strong>is_encoder_decoder</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether this is an encoder/decoder model.`,name:"is_encoder_decoder"},{anchor:"transformers.ProphetNetConfig.pad_token_id",description:`<strong>pad_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 1) &#x2014;
Padding token id.`,name:"pad_token_id"},{anchor:"transformers.ProphetNetConfig.bos_token_id",description:`<strong>bos_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 0) &#x2014;
Beginning of stream token id.`,name:"bos_token_id"},{anchor:"transformers.ProphetNetConfig.eos_token_id",description:`<strong>eos_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 2) &#x2014;
End of stream token id.`,name:"eos_token_id"},{anchor:"transformers.ProphetNetConfig.ngram",description:`<strong>ngram</strong> (<code>int</code>, <em>optional</em>, defaults to 2) &#x2014;
Number of future tokens to predict. Set to 1 to be same as traditional Language model to predict next first
token.`,name:"ngram"},{anchor:"transformers.ProphetNetConfig.num_buckets",description:`<strong>num_buckets</strong> (<code>int</code>, <em>optional</em>, defaults to 32) &#x2014;
The number of buckets to use for each attention layer. This is for relative position calculation. See the
[T5 paper](see <a href="https://huggingface.co/papers/1910.10683" rel="nofollow">https://huggingface.co/papers/1910.10683</a>) for more details.`,name:"num_buckets"},{anchor:"transformers.ProphetNetConfig.relative_max_distance",description:`<strong>relative_max_distance</strong> (<code>int</code>, <em>optional</em>, defaults to 128) &#x2014;
Relative distances greater than this number will be put into the last same bucket. This is for relative
position calculation. See the [T5 paper](see <a href="https://huggingface.co/papers/1910.10683" rel="nofollow">https://huggingface.co/papers/1910.10683</a>) for more details.`,name:"relative_max_distance"},{anchor:"transformers.ProphetNetConfig.disable_ngram_loss",description:`<strong>disable_ngram_loss</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether be trained predicting only the next first token.`,name:"disable_ngram_loss"},{anchor:"transformers.ProphetNetConfig.eps",description:`<strong>eps</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
Controls the <code>epsilon</code> parameter value for label smoothing in the loss calculation. If set to 0, no label
smoothing is performed.`,name:"eps"},{anchor:"transformers.ProphetNetConfig.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not the model should return the last key/values attentions (not used by all models).`,name:"use_cache"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/prophetnet/configuration_prophetnet.py#L26"}}),Ce=new W({props:{title:"ProphetNetTokenizer",local:"transformers.ProphetNetTokenizer",headingTag:"h2"}}),Fe=new z({props:{name:"class transformers.ProphetNetTokenizer",anchor:"transformers.ProphetNetTokenizer",parameters:[{name:"vocab_file",val:": str"},{name:"do_lower_case",val:": typing.Optional[bool] = True"},{name:"do_basic_tokenize",val:": typing.Optional[bool] = True"},{name:"never_split",val:": typing.Optional[collections.abc.Iterable] = None"},{name:"unk_token",val:": typing.Optional[str] = '[UNK]'"},{name:"sep_token",val:": typing.Optional[str] = '[SEP]'"},{name:"x_sep_token",val:": typing.Optional[str] = '[X_SEP]'"},{name:"pad_token",val:": typing.Optional[str] = '[PAD]'"},{name:"mask_token",val:": typing.Optional[str] = '[MASK]'"},{name:"tokenize_chinese_chars",val:": typing.Optional[bool] = True"},{name:"strip_accents",val:": typing.Optional[bool] = None"},{name:"clean_up_tokenization_spaces",val:": bool = True"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.ProphetNetTokenizer.vocab_file",description:`<strong>vocab_file</strong> (<code>str</code>) &#x2014;
File containing the vocabulary.`,name:"vocab_file"},{anchor:"transformers.ProphetNetTokenizer.do_lower_case",description:`<strong>do_lower_case</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to lowercase the input when tokenizing.`,name:"do_lower_case"},{anchor:"transformers.ProphetNetTokenizer.do_basic_tokenize",description:`<strong>do_basic_tokenize</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to do basic tokenization before WordPiece.`,name:"do_basic_tokenize"},{anchor:"transformers.ProphetNetTokenizer.never_split",description:`<strong>never_split</strong> (<code>Iterable</code>, <em>optional</em>) &#x2014;
Collection of tokens which will never be split during tokenization. Only has an effect when
<code>do_basic_tokenize=True</code>`,name:"never_split"},{anchor:"transformers.ProphetNetTokenizer.unk_token",description:`<strong>unk_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;[UNK]&quot;</code>) &#x2014;
The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
token instead.`,name:"unk_token"},{anchor:"transformers.ProphetNetTokenizer.sep_token",description:`<strong>sep_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;[SEP]&quot;</code>) &#x2014;
The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
sequence classification or for a text and a question for question answering. It is also used as the last
token of a sequence built with special tokens.`,name:"sep_token"},{anchor:"transformers.ProphetNetTokenizer.x_sep_token",description:`<strong>x_sep_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;[X_SEP]&quot;</code>) &#x2014;
Special second separator token, which can be generated by <a href="/docs/transformers/v4.56.2/en/model_doc/prophetnet#transformers.ProphetNetForConditionalGeneration">ProphetNetForConditionalGeneration</a>. It is
used to separate bullet-point like sentences in summarization, <em>e.g.</em>.`,name:"x_sep_token"},{anchor:"transformers.ProphetNetTokenizer.pad_token",description:`<strong>pad_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;[PAD]&quot;</code>) &#x2014;
The token used for padding, for example when batching sequences of different lengths.`,name:"pad_token"},{anchor:"transformers.ProphetNetTokenizer.mask_token",description:`<strong>mask_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;[MASK]&quot;</code>) &#x2014;
The token used for masking values. This is the token used when training this model with masked language
modeling. This is the token which the model will try to predict.`,name:"mask_token"},{anchor:"transformers.ProphetNetTokenizer.tokenize_chinese_chars",description:`<strong>tokenize_chinese_chars</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to tokenize Chinese characters.</p>
<p>This should likely be deactivated for Japanese (see this
<a href="https://github.com/huggingface/transformers/issues/328" rel="nofollow">issue</a>).`,name:"tokenize_chinese_chars"},{anchor:"transformers.ProphetNetTokenizer.strip_accents",description:`<strong>strip_accents</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to strip all accents. If this option is not specified, then it will be determined by the
value for <code>lowercase</code> (as in the original BERT).`,name:"strip_accents"},{anchor:"transformers.ProphetNetTokenizer.clean_up_tokenization_spaces",description:`<strong>clean_up_tokenization_spaces</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to cleanup spaces after decoding, cleanup consists in removing potential artifacts like
extra spaces.`,name:"clean_up_tokenization_spaces"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/prophetnet/tokenization_prophetnet.py#L272"}}),Je=new z({props:{name:"build_inputs_with_special_tokens",anchor:"transformers.ProphetNetTokenizer.build_inputs_with_special_tokens",parameters:[{name:"token_ids_0",val:": list"},{name:"token_ids_1",val:": typing.Optional[list[int]] = None"}],parametersDescription:[{anchor:"transformers.ProphetNetTokenizer.build_inputs_with_special_tokens.token_ids_0",description:`<strong>token_ids_0</strong> (<code>List[int]</code>) &#x2014;
List of IDs to which the special tokens will be added.`,name:"token_ids_0"},{anchor:"transformers.ProphetNetTokenizer.build_inputs_with_special_tokens.token_ids_1",description:`<strong>token_ids_1</strong> (<code>List[int]</code>, <em>optional</em>) &#x2014;
Optional second list of IDs for sequence pairs.`,name:"token_ids_1"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/prophetnet/tokenization_prophetnet.py#L455",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>List of <a href="../glossary#input-ids">input IDs</a> with the appropriate special tokens.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>List[int]</code></p>
`}}),je=new z({props:{name:"convert_tokens_to_string",anchor:"transformers.ProphetNetTokenizer.convert_tokens_to_string",parameters:[{name:"tokens",val:": str"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/prophetnet/tokenization_prophetnet.py#L400"}}),Ue=new z({props:{name:"get_special_tokens_mask",anchor:"transformers.ProphetNetTokenizer.get_special_tokens_mask",parameters:[{name:"token_ids_0",val:": list"},{name:"token_ids_1",val:": typing.Optional[list[int]] = None"},{name:"already_has_special_tokens",val:": typing.Optional[bool] = False"}],parametersDescription:[{anchor:"transformers.ProphetNetTokenizer.get_special_tokens_mask.token_ids_0",description:`<strong>token_ids_0</strong> (<code>List[int]</code>) &#x2014;
List of IDs.`,name:"token_ids_0"},{anchor:"transformers.ProphetNetTokenizer.get_special_tokens_mask.token_ids_1",description:`<strong>token_ids_1</strong> (<code>List[int]</code>, <em>optional</em>) &#x2014;
Optional second list of IDs for sequence pairs.`,name:"token_ids_1"},{anchor:"transformers.ProphetNetTokenizer.get_special_tokens_mask.already_has_special_tokens",description:`<strong>already_has_special_tokens</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not the token list is already formatted with special tokens for the model.`,name:"already_has_special_tokens"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/prophetnet/tokenization_prophetnet.py#L405",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>List[int]</code></p>
`}}),Ie=new W({props:{title:"ProphetNet specific outputs",local:"transformers.models.prophetnet.modeling_prophetnet.ProphetNetSeq2SeqLMOutput",headingTag:"h2"}}),Oe=new z({props:{name:"class transformers.models.prophetnet.modeling_prophetnet.ProphetNetSeq2SeqLMOutput",anchor:"transformers.models.prophetnet.modeling_prophetnet.ProphetNetSeq2SeqLMOutput",parameters:[{name:"loss",val:": typing.Optional[torch.FloatTensor] = None"},{name:"logits",val:": typing.Optional[torch.FloatTensor] = None"},{name:"logits_ngram",val:": typing.Optional[torch.FloatTensor] = None"},{name:"past_key_values",val:": typing.Optional[tuple[torch.FloatTensor]] = None"},{name:"decoder_hidden_states",val:": typing.Optional[tuple[torch.FloatTensor]] = None"},{name:"decoder_ngram_hidden_states",val:": typing.Optional[tuple[torch.FloatTensor]] = None"},{name:"decoder_attentions",val:": typing.Optional[tuple[torch.FloatTensor]] = None"},{name:"decoder_ngram_attentions",val:": typing.Optional[tuple[torch.FloatTensor]] = None"},{name:"cross_attentions",val:": typing.Optional[tuple[torch.FloatTensor]] = None"},{name:"encoder_last_hidden_state",val:": typing.Optional[torch.FloatTensor] = None"},{name:"encoder_hidden_states",val:": typing.Optional[tuple[torch.FloatTensor]] = None"},{name:"encoder_attentions",val:": typing.Optional[tuple[torch.FloatTensor]] = None"}],parametersDescription:[{anchor:"transformers.models.prophetnet.modeling_prophetnet.ProphetNetSeq2SeqLMOutput.loss",description:`<strong>loss</strong> (<code>torch.FloatTensor</code> of shape <code>(1,)</code>, <em>optional</em>, returned when <code>labels</code> is provided) &#x2014;
Language modeling loss.`,name:"loss"},{anchor:"transformers.models.prophetnet.modeling_prophetnet.ProphetNetSeq2SeqLMOutput.logits",description:`<strong>logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, decoder_sequence_length, config.vocab_size)</code>) &#x2014;
Prediction scores of the main stream language modeling head (scores for each vocabulary token before
SoftMax).`,name:"logits"},{anchor:"transformers.models.prophetnet.modeling_prophetnet.ProphetNetSeq2SeqLMOutput.logits_ngram",description:`<strong>logits_ngram</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, ngram * decoder_sequence_length, config.vocab_size)</code>) &#x2014;
Prediction scores of the predict stream language modeling head (scores for each vocabulary token before
SoftMax).`,name:"logits_ngram"},{anchor:"transformers.models.prophetnet.modeling_prophetnet.ProphetNetSeq2SeqLMOutput.past_key_values",description:`<strong>past_key_values</strong> (<code>list[torch.FloatTensor]</code>, <em>optional</em>, returned when <code>use_cache=True</code> is passed or when <code>config.use_cache=True</code>) &#x2014;
List of <code>torch.FloatTensor</code> of length <code>config.n_layers</code>, with each tensor of shape <code>(2, batch_size, num_attn_heads, decoder_sequence_length, embed_size_per_head)</code>).</p>
<p>Contains pre-computed hidden-states (key and values in the attention blocks) of the decoder that can be
used (see <code>past_key_values</code> input) to speed up sequential decoding.`,name:"past_key_values"},{anchor:"transformers.models.prophetnet.modeling_prophetnet.ProphetNetSeq2SeqLMOutput.decoder_hidden_states",description:`<strong>decoder_hidden_states</strong> (<code>tuple[torch.FloatTensor]</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) &#x2014;
Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings, if the model has an embedding layer, +
one for the output of each layer) of shape <code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.`,name:"decoder_hidden_states"},{anchor:"transformers.models.prophetnet.modeling_prophetnet.ProphetNetSeq2SeqLMOutput.decoder_ngram_hidden_states",description:`<strong>decoder_ngram_hidden_states</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) &#x2014;
Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings + one for the output of each layer) of
shape <code>(batch_size, ngram * decoder_sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the predict stream of the decoder at the output of each layer plus the initial embedding
outputs.`,name:"decoder_ngram_hidden_states"},{anchor:"transformers.models.prophetnet.modeling_prophetnet.ProphetNetSeq2SeqLMOutput.decoder_attentions",description:`<strong>decoder_attentions</strong> (<code>tuple[torch.FloatTensor]</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) &#x2014;
Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
self-attention heads.`,name:"decoder_attentions"},{anchor:"transformers.models.prophetnet.modeling_prophetnet.ProphetNetSeq2SeqLMOutput.decoder_ngram_attentions",description:`<strong>decoder_ngram_attentions</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) &#x2014;
Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_attn_heads, decoder_sequence_length, decoder_sequence_length)</code>.</p>
<p>Attentions weights of the predict stream of the decoder, after the attention softmax, used to compute the
weighted average in the self-attention heads.`,name:"decoder_ngram_attentions"},{anchor:"transformers.models.prophetnet.modeling_prophetnet.ProphetNetSeq2SeqLMOutput.cross_attentions",description:`<strong>cross_attentions</strong> (<code>tuple[torch.FloatTensor]</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) &#x2014;
Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights of the decoder&#x2019;s cross-attention layer, after the attention softmax, used to compute the
weighted average in the cross-attention heads.`,name:"cross_attentions"},{anchor:"transformers.models.prophetnet.modeling_prophetnet.ProphetNetSeq2SeqLMOutput.encoder_last_hidden_state",description:`<strong>encoder_last_hidden_state</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, encoder_sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Sequence of hidden-states at the output of the last layer of the encoder of the model.`,name:"encoder_last_hidden_state"},{anchor:"transformers.models.prophetnet.modeling_prophetnet.ProphetNetSeq2SeqLMOutput.encoder_hidden_states",description:`<strong>encoder_hidden_states</strong> (<code>tuple[torch.FloatTensor]</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) &#x2014;
Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings, if the model has an embedding layer, +
one for the output of each layer) of shape <code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.`,name:"encoder_hidden_states"},{anchor:"transformers.models.prophetnet.modeling_prophetnet.ProphetNetSeq2SeqLMOutput.encoder_attentions",description:`<strong>encoder_attentions</strong> (<code>tuple[torch.FloatTensor]</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) &#x2014;
Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
self-attention heads.`,name:"encoder_attentions"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/prophetnet/modeling_prophetnet.py#L122"}}),Ge=new z({props:{name:"class transformers.models.prophetnet.modeling_prophetnet.ProphetNetSeq2SeqModelOutput",anchor:"transformers.models.prophetnet.modeling_prophetnet.ProphetNetSeq2SeqModelOutput",parameters:[{name:"last_hidden_state",val:": FloatTensor"},{name:"last_hidden_state_ngram",val:": typing.Optional[torch.FloatTensor] = None"},{name:"past_key_values",val:": typing.Optional[tuple[torch.FloatTensor]] = None"},{name:"decoder_hidden_states",val:": typing.Optional[tuple[torch.FloatTensor]] = None"},{name:"decoder_ngram_hidden_states",val:": typing.Optional[tuple[torch.FloatTensor]] = None"},{name:"decoder_attentions",val:": typing.Optional[tuple[torch.FloatTensor]] = None"},{name:"decoder_ngram_attentions",val:": typing.Optional[tuple[torch.FloatTensor]] = None"},{name:"cross_attentions",val:": typing.Optional[tuple[torch.FloatTensor]] = None"},{name:"encoder_last_hidden_state",val:": typing.Optional[torch.FloatTensor] = None"},{name:"encoder_hidden_states",val:": typing.Optional[tuple[torch.FloatTensor]] = None"},{name:"encoder_attentions",val:": typing.Optional[tuple[torch.FloatTensor]] = None"}],parametersDescription:[{anchor:"transformers.models.prophetnet.modeling_prophetnet.ProphetNetSeq2SeqModelOutput.last_hidden_state",description:`<strong>last_hidden_state</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, decoder_sequence_length, hidden_size)</code>) &#x2014;
Sequence of main stream hidden-states at the output of the last layer of the decoder of the model.</p>
<p>If <code>past_key_values</code> is used only the last hidden-state of the sequences of shape <code>(batch_size, 1, hidden_size)</code> is output.`,name:"last_hidden_state"},{anchor:"transformers.models.prophetnet.modeling_prophetnet.ProphetNetSeq2SeqModelOutput.last_hidden_state_ngram",description:`<strong>last_hidden_state_ngram</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size,ngram * decoder_sequence_length, config.vocab_size)</code>, <em>optional</em>) &#x2014;
Sequence of predict stream hidden-states at the output of the last layer of the decoder of the model.`,name:"last_hidden_state_ngram"},{anchor:"transformers.models.prophetnet.modeling_prophetnet.ProphetNetSeq2SeqModelOutput.past_key_values",description:`<strong>past_key_values</strong> (<code>list[torch.FloatTensor]</code>, <em>optional</em>, returned when <code>use_cache=True</code> is passed or when <code>config.use_cache=True</code>) &#x2014;
List of <code>torch.FloatTensor</code> of length <code>config.n_layers</code>, with each tensor of shape <code>(2, batch_size, num_attn_heads, decoder_sequence_length, embed_size_per_head)</code>).</p>
<p>Contains pre-computed hidden-states (key and values in the attention blocks) of the decoder that can be
used (see <code>past_key_values</code> input) to speed up sequential decoding.`,name:"past_key_values"},{anchor:"transformers.models.prophetnet.modeling_prophetnet.ProphetNetSeq2SeqModelOutput.decoder_hidden_states",description:`<strong>decoder_hidden_states</strong> (<code>tuple[torch.FloatTensor]</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) &#x2014;
Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings, if the model has an embedding layer, +
one for the output of each layer) of shape <code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.`,name:"decoder_hidden_states"},{anchor:"transformers.models.prophetnet.modeling_prophetnet.ProphetNetSeq2SeqModelOutput.decoder_ngram_hidden_states",description:`<strong>decoder_ngram_hidden_states</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) &#x2014;
Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings + one for the output of each layer) of
shape <code>(batch_size, ngram * decoder_sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the predict stream of the decoder at the output of each layer plus the initial embedding
outputs.`,name:"decoder_ngram_hidden_states"},{anchor:"transformers.models.prophetnet.modeling_prophetnet.ProphetNetSeq2SeqModelOutput.decoder_attentions",description:`<strong>decoder_attentions</strong> (<code>tuple[torch.FloatTensor]</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) &#x2014;
Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
self-attention heads.`,name:"decoder_attentions"},{anchor:"transformers.models.prophetnet.modeling_prophetnet.ProphetNetSeq2SeqModelOutput.decoder_ngram_attentions",description:`<strong>decoder_ngram_attentions</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) &#x2014;
Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_attn_heads, decoder_sequence_length, decoder_sequence_length)</code>.</p>
<p>Attentions weights of the predict stream of the decoder, after the attention softmax, used to compute the
weighted average in the`,name:"decoder_ngram_attentions"},{anchor:"transformers.models.prophetnet.modeling_prophetnet.ProphetNetSeq2SeqModelOutput.cross_attentions",description:`<strong>cross_attentions</strong> (<code>tuple[torch.FloatTensor]</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) &#x2014;
Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights of the decoder&#x2019;s cross-attention layer, after the attention softmax, used to compute the
weighted average in the cross-attention heads.`,name:"cross_attentions"},{anchor:"transformers.models.prophetnet.modeling_prophetnet.ProphetNetSeq2SeqModelOutput.encoder_last_hidden_state",description:`<strong>encoder_last_hidden_state</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, encoder_sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Sequence of hidden-states at the output of the last layer of the encoder of the model.`,name:"encoder_last_hidden_state"},{anchor:"transformers.models.prophetnet.modeling_prophetnet.ProphetNetSeq2SeqModelOutput.encoder_hidden_states",description:`<strong>encoder_hidden_states</strong> (<code>tuple[torch.FloatTensor]</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) &#x2014;
Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings, if the model has an embedding layer, +
one for the output of each layer) of shape <code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.`,name:"encoder_hidden_states"},{anchor:"transformers.models.prophetnet.modeling_prophetnet.ProphetNetSeq2SeqModelOutput.encoder_attentions",description:`<strong>encoder_attentions</strong> (<code>tuple[torch.FloatTensor]</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) &#x2014;
Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
self-attention heads.`,name:"encoder_attentions"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/prophetnet/modeling_prophetnet.py#L184"}}),Ze=new z({props:{name:"class transformers.models.prophetnet.modeling_prophetnet.ProphetNetDecoderModelOutput",anchor:"transformers.models.prophetnet.modeling_prophetnet.ProphetNetDecoderModelOutput",parameters:[{name:"last_hidden_state",val:": FloatTensor"},{name:"last_hidden_state_ngram",val:": typing.Optional[torch.FloatTensor] = None"},{name:"past_key_values",val:": typing.Optional[tuple[torch.FloatTensor]] = None"},{name:"hidden_states",val:": typing.Optional[tuple[torch.FloatTensor]] = None"},{name:"hidden_states_ngram",val:": typing.Optional[tuple[torch.FloatTensor]] = None"},{name:"attentions",val:": typing.Optional[tuple[torch.FloatTensor]] = None"},{name:"ngram_attentions",val:": typing.Optional[tuple[torch.FloatTensor]] = None"},{name:"cross_attentions",val:": typing.Optional[tuple[torch.FloatTensor]] = None"}],parametersDescription:[{anchor:"transformers.models.prophetnet.modeling_prophetnet.ProphetNetDecoderModelOutput.last_hidden_state",description:`<strong>last_hidden_state</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, decoder_sequence_length, hidden_size)</code>) &#x2014;
Sequence of main stream hidden-states at the output of the last layer of the decoder of the model.</p>
<p>If <code>past_key_values</code> is used only the last hidden-state of the sequences of shape <code>(batch_size, 1, hidden_size)</code> is output.`,name:"last_hidden_state"},{anchor:"transformers.models.prophetnet.modeling_prophetnet.ProphetNetDecoderModelOutput.last_hidden_state_ngram",description:`<strong>last_hidden_state_ngram</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, ngram * decoder_sequence_length, config.vocab_size)</code>) &#x2014;
Sequence of predict stream hidden-states at the output of the last layer of the decoder of the model.`,name:"last_hidden_state_ngram"},{anchor:"transformers.models.prophetnet.modeling_prophetnet.ProphetNetDecoderModelOutput.past_key_values",description:`<strong>past_key_values</strong> (<code>list[torch.FloatTensor]</code>, <em>optional</em>, returned when <code>use_cache=True</code> is passed or when <code>config.use_cache=True</code>) &#x2014;
List of <code>torch.FloatTensor</code> of length <code>config.n_layers</code>, with each tensor of shape <code>(2, batch_size, num_attn_heads, decoder_sequence_length, embed_size_per_head)</code>).</p>
<p>Contains pre-computed hidden-states (key and values in the attention blocks) of the decoder that can be
used (see <code>past_key_values</code> input) to speed up sequential decoding.`,name:"past_key_values"},{anchor:"transformers.models.prophetnet.modeling_prophetnet.ProphetNetDecoderModelOutput.hidden_states",description:`<strong>hidden_states</strong> (<code>tuple[torch.FloatTensor]</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) &#x2014;
Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings, if the model has an embedding layer, +
one for the output of each layer) of shape <code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.`,name:"hidden_states"},{anchor:"transformers.models.prophetnet.modeling_prophetnet.ProphetNetDecoderModelOutput.hidden_states_ngram",description:`<strong>hidden_states_ngram</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) &#x2014;
Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings + one for the output of each layer) of
shape <code>(batch_size, ngram * decoder_sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the predict stream of the decoder at the output of each layer plus the initial embedding
outputs.`,name:"hidden_states_ngram"},{anchor:"transformers.models.prophetnet.modeling_prophetnet.ProphetNetDecoderModelOutput.attentions",description:`<strong>attentions</strong> (<code>tuple[torch.FloatTensor]</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) &#x2014;
Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
heads.`,name:"attentions"},{anchor:"transformers.models.prophetnet.modeling_prophetnet.ProphetNetDecoderModelOutput.ngram_attentions",description:`<strong>ngram_attentions</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) &#x2014;
Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_attn_heads, decoder_sequence_length, decoder_sequence_length)</code>.</p>
<p>Attentions weights of the predict stream of the decoder, after the attention softmax, used to compute the
weighted average in the`,name:"ngram_attentions"},{anchor:"transformers.models.prophetnet.modeling_prophetnet.ProphetNetDecoderModelOutput.cross_attentions",description:`<strong>cross_attentions</strong> (<code>tuple[torch.FloatTensor]</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) &#x2014;
Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights of the decoder&#x2019;s cross-attention layer, after the attention softmax, used to compute the
weighted average in the cross-attention heads.`,name:"cross_attentions"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/prophetnet/modeling_prophetnet.py#L243"}}),We=new z({props:{name:"class transformers.models.prophetnet.modeling_prophetnet.ProphetNetDecoderLMOutput",anchor:"transformers.models.prophetnet.modeling_prophetnet.ProphetNetDecoderLMOutput",parameters:[{name:"loss",val:": typing.Optional[torch.FloatTensor] = None"},{name:"logits",val:": typing.Optional[torch.FloatTensor] = None"},{name:"logits_ngram",val:": typing.Optional[torch.FloatTensor] = None"},{name:"past_key_values",val:": typing.Optional[tuple[torch.FloatTensor]] = None"},{name:"hidden_states",val:": typing.Optional[tuple[torch.FloatTensor]] = None"},{name:"hidden_states_ngram",val:": typing.Optional[tuple[torch.FloatTensor]] = None"},{name:"attentions",val:": typing.Optional[tuple[torch.FloatTensor]] = None"},{name:"ngram_attentions",val:": typing.Optional[tuple[torch.FloatTensor]] = None"},{name:"cross_attentions",val:": typing.Optional[tuple[torch.FloatTensor]] = None"}],parametersDescription:[{anchor:"transformers.models.prophetnet.modeling_prophetnet.ProphetNetDecoderLMOutput.loss",description:`<strong>loss</strong> (<code>torch.FloatTensor</code> of shape <code>(1,)</code>, <em>optional</em>, returned when <code>labels</code> is provided) &#x2014;
Language modeling loss.`,name:"loss"},{anchor:"transformers.models.prophetnet.modeling_prophetnet.ProphetNetDecoderLMOutput.logits",description:`<strong>logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, decoder_sequence_length, config.vocab_size)</code>) &#x2014;
Prediction scores of the main stream language modeling head (scores for each vocabulary token before
SoftMax).`,name:"logits"},{anchor:"transformers.models.prophetnet.modeling_prophetnet.ProphetNetDecoderLMOutput.logits_ngram",description:`<strong>logits_ngram</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, ngram * decoder_sequence_length, config.vocab_size)</code>) &#x2014;
Prediction scores of the predict stream language modeling head (scores for each vocabulary token before
SoftMax).`,name:"logits_ngram"},{anchor:"transformers.models.prophetnet.modeling_prophetnet.ProphetNetDecoderLMOutput.past_key_values",description:`<strong>past_key_values</strong> (<code>list[torch.FloatTensor]</code>, <em>optional</em>, returned when <code>use_cache=True</code> is passed or when <code>config.use_cache=True</code>) &#x2014;
List of <code>torch.FloatTensor</code> of length <code>config.n_layers</code>, with each tensor of shape <code>(2, batch_size, num_attn_heads, decoder_sequence_length, embed_size_per_head)</code>).</p>
<p>Contains pre-computed hidden-states (key and values in the attention blocks) of the decoder that can be
used (see <code>past_key_values</code> input) to speed up sequential decoding.`,name:"past_key_values"},{anchor:"transformers.models.prophetnet.modeling_prophetnet.ProphetNetDecoderLMOutput.hidden_states",description:`<strong>hidden_states</strong> (<code>tuple[torch.FloatTensor]</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) &#x2014;
Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings, if the model has an embedding layer, +
one for the output of each layer) of shape <code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.`,name:"hidden_states"},{anchor:"transformers.models.prophetnet.modeling_prophetnet.ProphetNetDecoderLMOutput.hidden_states_ngram",description:`<strong>hidden_states_ngram</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) &#x2014;
Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings + one for the output of each layer) of
shape <code>(batch_size, ngram * decoder_sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the predict stream of the decoder at the output of each layer plus the initial embedding
outputs.`,name:"hidden_states_ngram"},{anchor:"transformers.models.prophetnet.modeling_prophetnet.ProphetNetDecoderLMOutput.attentions",description:`<strong>attentions</strong> (<code>tuple[torch.FloatTensor]</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) &#x2014;
Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
heads.`,name:"attentions"},{anchor:"transformers.models.prophetnet.modeling_prophetnet.ProphetNetDecoderLMOutput.ngram_attentions",description:`<strong>ngram_attentions</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) &#x2014;
Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_attn_heads, decoder_sequence_length, decoder_sequence_length)</code>.</p>
<p>Attentions weights of the predict stream of the decoder, after the attention softmax, used to compute the
weighted average in the`,name:"ngram_attentions"},{anchor:"transformers.models.prophetnet.modeling_prophetnet.ProphetNetDecoderLMOutput.cross_attentions",description:`<strong>cross_attentions</strong> (<code>tuple[torch.FloatTensor]</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) &#x2014;
Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights of the decoder&#x2019;s cross-attention layer, after the attention softmax, used to compute the
weighted average in the cross-attention heads.`,name:"cross_attentions"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/prophetnet/modeling_prophetnet.py#L288"}}),Le=new W({props:{title:"ProphetNetModel",local:"transformers.ProphetNetModel",headingTag:"h2"}}),Se=new z({props:{name:"class transformers.ProphetNetModel",anchor:"transformers.ProphetNetModel",parameters:[{name:"config",val:": ProphetNetConfig"}],parametersDescription:[{anchor:"transformers.ProphetNetModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/prophetnet#transformers.ProphetNetConfig">ProphetNetConfig</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/prophetnet/modeling_prophetnet.py#L1477"}}),Ve=new z({props:{name:"forward",anchor:"transformers.ProphetNetModel.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"decoder_input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"decoder_attention_mask",val:": typing.Optional[torch.BoolTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"decoder_head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"cross_attn_head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_outputs",val:": typing.Optional[tuple] = None"},{name:"past_key_values",val:": typing.Optional[tuple[tuple[torch.Tensor]]] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"decoder_inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.Tensor] = None"}],parametersDescription:[{anchor:"transformers.ProphetNetModel.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.ProphetNetModel.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.ProphetNetModel.forward.decoder_input_ids",description:`<strong>decoder_input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of decoder input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#decoder-input-ids">What are decoder input IDs?</a></p>
<p>ProphetNet uses the <code>eos_token_id</code> as the starting token for <code>decoder_input_ids</code> generation. If
<code>past_key_values</code> is used, optionally only the last <code>decoder_input_ids</code> have to be input (see
<code>past_key_values</code>).`,name:"decoder_input_ids"},{anchor:"transformers.ProphetNetModel.forward.decoder_attention_mask",description:`<strong>decoder_attention_mask</strong> (<code>torch.BoolTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Default behavior: generate a tensor that ignores pad tokens in <code>decoder_input_ids</code>. Causal mask will also
be used by default.`,name:"decoder_attention_mask"},{anchor:"transformers.ProphetNetModel.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.ProphetNetModel.forward.decoder_head_mask",description:`<strong>decoder_head_mask</strong> (<code>torch.Tensor</code> of shape <code>(decoder_layers, decoder_attention_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the attention modules in the decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"decoder_head_mask"},{anchor:"transformers.ProphetNetModel.forward.cross_attn_head_mask",description:`<strong>cross_attn_head_mask</strong> (<code>torch.Tensor</code> of shape <code>(decoder_layers, decoder_attention_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the cross-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"cross_attn_head_mask"},{anchor:"transformers.ProphetNetModel.forward.encoder_outputs",description:`<strong>encoder_outputs</strong> (<code>tuple</code>, <em>optional</em>) &#x2014;
Tuple consists of (<code>last_hidden_state</code>, <em>optional</em>: <code>hidden_states</code>, <em>optional</em>: <code>attentions</code>)
<code>last_hidden_state</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) is a sequence of
hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.`,name:"encoder_outputs"},{anchor:"transformers.ProphetNetModel.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>tuple[tuple[torch.Tensor]]</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.ProphetNetModel.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.ProphetNetModel.forward.decoder_inputs_embeds",description:`<strong>decoder_inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, target_sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>decoder_input_ids</code> you can choose to directly pass an embedded
representation. If <code>past_key_values</code> is used, optionally only the last <code>decoder_inputs_embeds</code> have to be
input (see <code>past_key_values</code>). This is useful if you want more control over how to convert
<code>decoder_input_ids</code> indices into associated vectors than the model&#x2019;s internal embedding lookup matrix.</p>
<p>If <code>decoder_input_ids</code> and <code>decoder_inputs_embeds</code> are both unset, <code>decoder_inputs_embeds</code> takes the value
of <code>inputs_embeds</code>.`,name:"decoder_inputs_embeds"},{anchor:"transformers.ProphetNetModel.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.ProphetNetModel.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.ProphetNetModel.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.ProphetNetModel.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.ProphetNetModel.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.Tensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/prophetnet/modeling_prophetnet.py#L1513",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/model_doc/prophetnet#transformers.models.prophetnet.modeling_prophetnet.ProphetNetSeq2SeqModelOutput"
>transformers.models.prophetnet.modeling_prophetnet.ProphetNetSeq2SeqModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/prophetnet#transformers.ProphetNetConfig"
>ProphetNetConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>last_hidden_state</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, decoder_sequence_length, hidden_size)</code>) — Sequence of main stream hidden-states at the output of the last layer of the decoder of the model.</p>
<p>If <code>past_key_values</code> is used only the last hidden-state of the sequences of shape <code>(batch_size, 1, hidden_size)</code> is output.</p>
</li>
<li>
<p><strong>last_hidden_state_ngram</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size,ngram * decoder_sequence_length, config.vocab_size)</code>, <em>optional</em>) — Sequence of predict stream hidden-states at the output of the last layer of the decoder of the model.</p>
</li>
<li>
<p><strong>past_key_values</strong> (<code>list[torch.FloatTensor]</code>, <em>optional</em>, returned when <code>use_cache=True</code> is passed or when <code>config.use_cache=True</code>) — List of <code>torch.FloatTensor</code> of length <code>config.n_layers</code>, with each tensor of shape <code>(2, batch_size, num_attn_heads, decoder_sequence_length, embed_size_per_head)</code>).</p>
<p>Contains pre-computed hidden-states (key and values in the attention blocks) of the decoder that can be
used (see <code>past_key_values</code> input) to speed up sequential decoding.</p>
</li>
<li>
<p><strong>decoder_hidden_states</strong> (<code>tuple[torch.FloatTensor]</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings, if the model has an embedding layer, +
one for the output of each layer) of shape <code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.</p>
</li>
<li>
<p><strong>decoder_ngram_hidden_states</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings + one for the output of each layer) of
shape <code>(batch_size, ngram * decoder_sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the predict stream of the decoder at the output of each layer plus the initial embedding
outputs.</p>
</li>
<li>
<p><strong>decoder_attentions</strong> (<code>tuple[torch.FloatTensor]</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
self-attention heads.</p>
</li>
<li>
<p><strong>decoder_ngram_attentions</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_attn_heads, decoder_sequence_length, decoder_sequence_length)</code>.</p>
<p>Attentions weights of the predict stream of the decoder, after the attention softmax, used to compute the
weighted average in the</p>
</li>
<li>
<p><strong>cross_attentions</strong> (<code>tuple[torch.FloatTensor]</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the
weighted average in the cross-attention heads.</p>
</li>
<li>
<p><strong>encoder_last_hidden_state</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, encoder_sequence_length, hidden_size)</code>, <em>optional</em>) — Sequence of hidden-states at the output of the last layer of the encoder of the model.</p>
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
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><a
  href="/docs/transformers/v4.56.2/en/model_doc/prophetnet#transformers.models.prophetnet.modeling_prophetnet.ProphetNetSeq2SeqModelOutput"
>transformers.models.prophetnet.modeling_prophetnet.ProphetNetSeq2SeqModelOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),ne=new St({props:{$$slots:{default:[ms]},$$scope:{ctx:w}}}),se=new Vt({props:{anchor:"transformers.ProphetNetModel.forward.example",$$slots:{default:[fs]},$$scope:{ctx:w}}}),Be=new W({props:{title:"ProphetNetEncoder",local:"transformers.ProphetNetEncoder",headingTag:"h2"}}),He=new z({props:{name:"class transformers.ProphetNetEncoder",anchor:"transformers.ProphetNetEncoder",parameters:[{name:"config",val:": ProphetNetConfig"},{name:"word_embeddings",val:": Embedding = None"}],parametersDescription:[{anchor:"transformers.ProphetNetEncoder.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/prophetnet#transformers.ProphetNetConfig">ProphetNetConfig</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"},{anchor:"transformers.ProphetNetEncoder.word_embeddings",description:`<strong>word_embeddings</strong> (<code>torch.nn.Embeddings</code> of shape <code>(config.vocab_size, config.hidden_size)</code>, <em>optional</em>) &#x2014;
The word embedding parameters. This can be used to initialize <a href="/docs/transformers/v4.56.2/en/model_doc/prophetnet#transformers.ProphetNetEncoder">ProphetNetEncoder</a> with pre-defined word
embeddings instead of randomly initialized word embeddings.`,name:"word_embeddings"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/prophetnet/modeling_prophetnet.py#L1021"}}),Re=new z({props:{name:"forward",anchor:"transformers.ProphetNetEncoder.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.ProphetNetEncoder.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.ProphetNetEncoder.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.ProphetNetEncoder.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.ProphetNetEncoder.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.ProphetNetEncoder.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.ProphetNetEncoder.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.ProphetNetEncoder.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/prophetnet/modeling_prophetnet.py#L1050",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutput"
>transformers.modeling_outputs.BaseModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/prophetnet#transformers.ProphetNetConfig"
>ProphetNetConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>last_hidden_state</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>) — Sequence of hidden-states at the output of the last layer of the model.</p>
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
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutput"
>transformers.modeling_outputs.BaseModelOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),re=new St({props:{$$slots:{default:[gs]},$$scope:{ctx:w}}}),ae=new Vt({props:{anchor:"transformers.ProphetNetEncoder.forward.example",$$slots:{default:[_s]},$$scope:{ctx:w}}}),Ee=new W({props:{title:"ProphetNetDecoder",local:"transformers.ProphetNetDecoder",headingTag:"h2"}}),De=new z({props:{name:"class transformers.ProphetNetDecoder",anchor:"transformers.ProphetNetDecoder",parameters:[{name:"config",val:": ProphetNetConfig"},{name:"word_embeddings",val:": typing.Optional[torch.nn.modules.sparse.Embedding] = None"}],parametersDescription:[{anchor:"transformers.ProphetNetDecoder.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/prophetnet#transformers.ProphetNetConfig">ProphetNetConfig</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"},{anchor:"transformers.ProphetNetDecoder.word_embeddings",description:`<strong>word_embeddings</strong> (<code>torch.nn.Embeddings</code> of shape <code>(config.vocab_size, config.hidden_size)</code>, <em>optional</em>) &#x2014;
The word embedding parameters. This can be used to initialize <a href="/docs/transformers/v4.56.2/en/model_doc/prophetnet#transformers.ProphetNetEncoder">ProphetNetEncoder</a> with pre-defined word
embeddings instead of randomly initialized word embeddings.`,name:"word_embeddings"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/prophetnet/modeling_prophetnet.py#L1143"}}),Xe=new z({props:{name:"forward",anchor:"transformers.ProphetNetDecoder.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_hidden_states",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"cross_attn_head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"past_key_values",val:": typing.Optional[tuple[tuple[torch.Tensor]]] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.Tensor] = None"}],parametersDescription:[{anchor:"transformers.ProphetNetDecoder.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.ProphetNetDecoder.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.ProphetNetDecoder.forward.encoder_hidden_states",description:`<strong>encoder_hidden_states</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
if the model is configured as a decoder.`,name:"encoder_hidden_states"},{anchor:"transformers.ProphetNetDecoder.forward.encoder_attention_mask",description:`<strong>encoder_attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
the cross-attention if the model is configured as a decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>`,name:"encoder_attention_mask"},{anchor:"transformers.ProphetNetDecoder.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.ProphetNetDecoder.forward.cross_attn_head_mask",description:`<strong>cross_attn_head_mask</strong> (<code>torch.Tensor</code> of shape <code>(decoder_layers, decoder_attention_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the cross-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"cross_attn_head_mask"},{anchor:"transformers.ProphetNetDecoder.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>tuple[tuple[torch.Tensor]]</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.ProphetNetDecoder.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.ProphetNetDecoder.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.ProphetNetDecoder.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.ProphetNetDecoder.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.ProphetNetDecoder.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.ProphetNetDecoder.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.Tensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/prophetnet/modeling_prophetnet.py#L1181",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/model_doc/prophetnet#transformers.models.prophetnet.modeling_prophetnet.ProphetNetDecoderModelOutput"
>transformers.models.prophetnet.modeling_prophetnet.ProphetNetDecoderModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/prophetnet#transformers.ProphetNetConfig"
>ProphetNetConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>last_hidden_state</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, decoder_sequence_length, hidden_size)</code>) — Sequence of main stream hidden-states at the output of the last layer of the decoder of the model.</p>
<p>If <code>past_key_values</code> is used only the last hidden-state of the sequences of shape <code>(batch_size, 1, hidden_size)</code> is output.</p>
</li>
<li>
<p><strong>last_hidden_state_ngram</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, ngram * decoder_sequence_length, config.vocab_size)</code>) — Sequence of predict stream hidden-states at the output of the last layer of the decoder of the model.</p>
</li>
<li>
<p><strong>past_key_values</strong> (<code>list[torch.FloatTensor]</code>, <em>optional</em>, returned when <code>use_cache=True</code> is passed or when <code>config.use_cache=True</code>) — List of <code>torch.FloatTensor</code> of length <code>config.n_layers</code>, with each tensor of shape <code>(2, batch_size, num_attn_heads, decoder_sequence_length, embed_size_per_head)</code>).</p>
<p>Contains pre-computed hidden-states (key and values in the attention blocks) of the decoder that can be
used (see <code>past_key_values</code> input) to speed up sequential decoding.</p>
</li>
<li>
<p><strong>hidden_states</strong> (<code>tuple[torch.FloatTensor]</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings, if the model has an embedding layer, +
one for the output of each layer) of shape <code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.</p>
</li>
<li>
<p><strong>hidden_states_ngram</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings + one for the output of each layer) of
shape <code>(batch_size, ngram * decoder_sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the predict stream of the decoder at the output of each layer plus the initial embedding
outputs.</p>
</li>
<li>
<p><strong>attentions</strong> (<code>tuple[torch.FloatTensor]</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
heads.</p>
</li>
<li>
<p><strong>ngram_attentions</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_attn_heads, decoder_sequence_length, decoder_sequence_length)</code>.</p>
<p>Attentions weights of the predict stream of the decoder, after the attention softmax, used to compute the
weighted average in the</p>
</li>
<li>
<p><strong>cross_attentions</strong> (<code>tuple[torch.FloatTensor]</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the
weighted average in the cross-attention heads.</p>
</li>
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><a
  href="/docs/transformers/v4.56.2/en/model_doc/prophetnet#transformers.models.prophetnet.modeling_prophetnet.ProphetNetDecoderModelOutput"
>transformers.models.prophetnet.modeling_prophetnet.ProphetNetDecoderModelOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),de=new St({props:{$$slots:{default:[Ts]},$$scope:{ctx:w}}}),ie=new Vt({props:{anchor:"transformers.ProphetNetDecoder.forward.example",$$slots:{default:[ys]},$$scope:{ctx:w}}}),Ae=new W({props:{title:"ProphetNetForConditionalGeneration",local:"transformers.ProphetNetForConditionalGeneration",headingTag:"h2"}}),Qe=new z({props:{name:"class transformers.ProphetNetForConditionalGeneration",anchor:"transformers.ProphetNetForConditionalGeneration",parameters:[{name:"config",val:": ProphetNetConfig"}],parametersDescription:[{anchor:"transformers.ProphetNetForConditionalGeneration.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/prophetnet#transformers.ProphetNetConfig">ProphetNetConfig</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/prophetnet/modeling_prophetnet.py#L1628"}}),Ye=new z({props:{name:"forward",anchor:"transformers.ProphetNetForConditionalGeneration.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"decoder_input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"decoder_attention_mask",val:": typing.Optional[torch.BoolTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"decoder_head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"cross_attn_head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_outputs",val:": typing.Optional[torch.Tensor] = None"},{name:"past_key_values",val:": typing.Optional[tuple[tuple[torch.Tensor]]] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"decoder_inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"labels",val:": typing.Optional[torch.Tensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.Tensor] = None"}],parametersDescription:[{anchor:"transformers.ProphetNetForConditionalGeneration.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.ProphetNetForConditionalGeneration.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.ProphetNetForConditionalGeneration.forward.decoder_input_ids",description:`<strong>decoder_input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of decoder input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#decoder-input-ids">What are decoder input IDs?</a></p>
<p>ProphetNet uses the <code>eos_token_id</code> as the starting token for <code>decoder_input_ids</code> generation. If
<code>past_key_values</code> is used, optionally only the last <code>decoder_input_ids</code> have to be input (see
<code>past_key_values</code>).`,name:"decoder_input_ids"},{anchor:"transformers.ProphetNetForConditionalGeneration.forward.decoder_attention_mask",description:`<strong>decoder_attention_mask</strong> (<code>torch.BoolTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Default behavior: generate a tensor that ignores pad tokens in <code>decoder_input_ids</code>. Causal mask will also
be used by default.`,name:"decoder_attention_mask"},{anchor:"transformers.ProphetNetForConditionalGeneration.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.ProphetNetForConditionalGeneration.forward.decoder_head_mask",description:`<strong>decoder_head_mask</strong> (<code>torch.Tensor</code> of shape <code>(decoder_layers, decoder_attention_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the attention modules in the decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"decoder_head_mask"},{anchor:"transformers.ProphetNetForConditionalGeneration.forward.cross_attn_head_mask",description:`<strong>cross_attn_head_mask</strong> (<code>torch.Tensor</code> of shape <code>(decoder_layers, decoder_attention_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the cross-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"cross_attn_head_mask"},{anchor:"transformers.ProphetNetForConditionalGeneration.forward.encoder_outputs",description:`<strong>encoder_outputs</strong> (<code>torch.Tensor</code>, <em>optional</em>) &#x2014;
Tuple consists of (<code>last_hidden_state</code>, <em>optional</em>: <code>hidden_states</code>, <em>optional</em>: <code>attentions</code>)
<code>last_hidden_state</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) is a sequence of
hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.`,name:"encoder_outputs"},{anchor:"transformers.ProphetNetForConditionalGeneration.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>tuple[tuple[torch.Tensor]]</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.ProphetNetForConditionalGeneration.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.ProphetNetForConditionalGeneration.forward.decoder_inputs_embeds",description:`<strong>decoder_inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, target_sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>decoder_input_ids</code> you can choose to directly pass an embedded
representation. If <code>past_key_values</code> is used, optionally only the last <code>decoder_inputs_embeds</code> have to be
input (see <code>past_key_values</code>). This is useful if you want more control over how to convert
<code>decoder_input_ids</code> indices into associated vectors than the model&#x2019;s internal embedding lookup matrix.</p>
<p>If <code>decoder_input_ids</code> and <code>decoder_inputs_embeds</code> are both unset, <code>decoder_inputs_embeds</code> takes the value
of <code>inputs_embeds</code>.`,name:"decoder_inputs_embeds"},{anchor:"transformers.ProphetNetForConditionalGeneration.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the sequence classification/regression loss. Indices should be in <code>[-100, 0, ..., config.vocab_size - 1]</code>. All labels set to <code>-100</code> are ignored (masked), the loss is only computed for
labels in <code>[0, ..., config.vocab_size]</code>`,name:"labels"},{anchor:"transformers.ProphetNetForConditionalGeneration.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.ProphetNetForConditionalGeneration.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.ProphetNetForConditionalGeneration.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.ProphetNetForConditionalGeneration.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.ProphetNetForConditionalGeneration.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.Tensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/prophetnet/modeling_prophetnet.py#L1649",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/model_doc/prophetnet#transformers.models.prophetnet.modeling_prophetnet.ProphetNetSeq2SeqLMOutput"
>transformers.models.prophetnet.modeling_prophetnet.ProphetNetSeq2SeqLMOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/prophetnet#transformers.ProphetNetConfig"
>ProphetNetConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>loss</strong> (<code>torch.FloatTensor</code> of shape <code>(1,)</code>, <em>optional</em>, returned when <code>labels</code> is provided) — Language modeling loss.</p>
</li>
<li>
<p><strong>logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, decoder_sequence_length, config.vocab_size)</code>) — Prediction scores of the main stream language modeling head (scores for each vocabulary token before
SoftMax).</p>
</li>
<li>
<p><strong>logits_ngram</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, ngram * decoder_sequence_length, config.vocab_size)</code>) — Prediction scores of the predict stream language modeling head (scores for each vocabulary token before
SoftMax).</p>
</li>
<li>
<p><strong>past_key_values</strong> (<code>list[torch.FloatTensor]</code>, <em>optional</em>, returned when <code>use_cache=True</code> is passed or when <code>config.use_cache=True</code>) — List of <code>torch.FloatTensor</code> of length <code>config.n_layers</code>, with each tensor of shape <code>(2, batch_size, num_attn_heads, decoder_sequence_length, embed_size_per_head)</code>).</p>
<p>Contains pre-computed hidden-states (key and values in the attention blocks) of the decoder that can be
used (see <code>past_key_values</code> input) to speed up sequential decoding.</p>
</li>
<li>
<p><strong>decoder_hidden_states</strong> (<code>tuple[torch.FloatTensor]</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings, if the model has an embedding layer, +
one for the output of each layer) of shape <code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.</p>
</li>
<li>
<p><strong>decoder_ngram_hidden_states</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings + one for the output of each layer) of
shape <code>(batch_size, ngram * decoder_sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the predict stream of the decoder at the output of each layer plus the initial embedding
outputs.</p>
</li>
<li>
<p><strong>decoder_attentions</strong> (<code>tuple[torch.FloatTensor]</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
self-attention heads.</p>
</li>
<li>
<p><strong>decoder_ngram_attentions</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_attn_heads, decoder_sequence_length, decoder_sequence_length)</code>.</p>
<p>Attentions weights of the predict stream of the decoder, after the attention softmax, used to compute the
weighted average in the self-attention heads.</p>
</li>
<li>
<p><strong>cross_attentions</strong> (<code>tuple[torch.FloatTensor]</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the
weighted average in the cross-attention heads.</p>
</li>
<li>
<p><strong>encoder_last_hidden_state</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, encoder_sequence_length, hidden_size)</code>, <em>optional</em>) — Sequence of hidden-states at the output of the last layer of the encoder of the model.</p>
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
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><a
  href="/docs/transformers/v4.56.2/en/model_doc/prophetnet#transformers.models.prophetnet.modeling_prophetnet.ProphetNetSeq2SeqLMOutput"
>transformers.models.prophetnet.modeling_prophetnet.ProphetNetSeq2SeqLMOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),ce=new St({props:{$$slots:{default:[bs]},$$scope:{ctx:w}}}),le=new Vt({props:{anchor:"transformers.ProphetNetForConditionalGeneration.forward.example",$$slots:{default:[vs]},$$scope:{ctx:w}}}),Ke=new W({props:{title:"ProphetNetForCausalLM",local:"transformers.ProphetNetForCausalLM",headingTag:"h2"}}),et=new z({props:{name:"class transformers.ProphetNetForCausalLM",anchor:"transformers.ProphetNetForCausalLM",parameters:[{name:"config",val:": ProphetNetConfig"}],parametersDescription:[{anchor:"transformers.ProphetNetForCausalLM.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/prophetnet#transformers.ProphetNetConfig">ProphetNetConfig</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/prophetnet/modeling_prophetnet.py#L1816"}}),tt=new z({props:{name:"forward",anchor:"transformers.ProphetNetForCausalLM.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_hidden_states",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"cross_attn_head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"past_key_values",val:": typing.Optional[tuple[tuple[torch.Tensor]]] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"labels",val:": typing.Optional[torch.Tensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.ProphetNetForCausalLM.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.ProphetNetForCausalLM.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.ProphetNetForCausalLM.forward.encoder_hidden_states",description:`<strong>encoder_hidden_states</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
if the model is configured as a decoder.`,name:"encoder_hidden_states"},{anchor:"transformers.ProphetNetForCausalLM.forward.encoder_attention_mask",description:`<strong>encoder_attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
the cross-attention if the model is configured as a decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>`,name:"encoder_attention_mask"},{anchor:"transformers.ProphetNetForCausalLM.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.ProphetNetForCausalLM.forward.cross_attn_head_mask",description:`<strong>cross_attn_head_mask</strong> (<code>torch.Tensor</code> of shape <code>(decoder_layers, decoder_attention_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the cross-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"cross_attn_head_mask"},{anchor:"transformers.ProphetNetForCausalLM.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>tuple[tuple[torch.Tensor]]</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.ProphetNetForCausalLM.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.ProphetNetForCausalLM.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the left-to-right language modeling loss (next word prediction). Indices should be in
<code>[-100, 0, ..., config.vocab_size]</code> (see <code>input_ids</code> docstring) Tokens with indices set to <code>-100</code> are
ignored (masked), the loss is only computed for the tokens with labels n <code>[0, ..., config.vocab_size]</code>`,name:"labels"},{anchor:"transformers.ProphetNetForCausalLM.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.ProphetNetForCausalLM.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.ProphetNetForCausalLM.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.ProphetNetForCausalLM.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/prophetnet/modeling_prophetnet.py#L1855",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/model_doc/prophetnet#transformers.models.prophetnet.modeling_prophetnet.ProphetNetDecoderLMOutput"
>transformers.models.prophetnet.modeling_prophetnet.ProphetNetDecoderLMOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/prophetnet#transformers.ProphetNetConfig"
>ProphetNetConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>loss</strong> (<code>torch.FloatTensor</code> of shape <code>(1,)</code>, <em>optional</em>, returned when <code>labels</code> is provided) — Language modeling loss.</p>
</li>
<li>
<p><strong>logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, decoder_sequence_length, config.vocab_size)</code>) — Prediction scores of the main stream language modeling head (scores for each vocabulary token before
SoftMax).</p>
</li>
<li>
<p><strong>logits_ngram</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, ngram * decoder_sequence_length, config.vocab_size)</code>) — Prediction scores of the predict stream language modeling head (scores for each vocabulary token before
SoftMax).</p>
</li>
<li>
<p><strong>past_key_values</strong> (<code>list[torch.FloatTensor]</code>, <em>optional</em>, returned when <code>use_cache=True</code> is passed or when <code>config.use_cache=True</code>) — List of <code>torch.FloatTensor</code> of length <code>config.n_layers</code>, with each tensor of shape <code>(2, batch_size, num_attn_heads, decoder_sequence_length, embed_size_per_head)</code>).</p>
<p>Contains pre-computed hidden-states (key and values in the attention blocks) of the decoder that can be
used (see <code>past_key_values</code> input) to speed up sequential decoding.</p>
</li>
<li>
<p><strong>hidden_states</strong> (<code>tuple[torch.FloatTensor]</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings, if the model has an embedding layer, +
one for the output of each layer) of shape <code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.</p>
</li>
<li>
<p><strong>hidden_states_ngram</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings + one for the output of each layer) of
shape <code>(batch_size, ngram * decoder_sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the predict stream of the decoder at the output of each layer plus the initial embedding
outputs.</p>
</li>
<li>
<p><strong>attentions</strong> (<code>tuple[torch.FloatTensor]</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
heads.</p>
</li>
<li>
<p><strong>ngram_attentions</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_attn_heads, decoder_sequence_length, decoder_sequence_length)</code>.</p>
<p>Attentions weights of the predict stream of the decoder, after the attention softmax, used to compute the
weighted average in the</p>
</li>
<li>
<p><strong>cross_attentions</strong> (<code>tuple[torch.FloatTensor]</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the
weighted average in the cross-attention heads.</p>
</li>
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><a
  href="/docs/transformers/v4.56.2/en/model_doc/prophetnet#transformers.models.prophetnet.modeling_prophetnet.ProphetNetDecoderLMOutput"
>transformers.models.prophetnet.modeling_prophetnet.ProphetNetDecoderLMOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),pe=new St({props:{$$slots:{default:[ks]},$$scope:{ctx:w}}}),he=new Vt({props:{anchor:"transformers.ProphetNetForCausalLM.forward.example",$$slots:{default:[ws]},$$scope:{ctx:w}}}),ot=new us({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/prophetnet.md"}}),{c(){a=i("meta"),b=s(),p=i("p"),h=s(),v=i("p"),v.innerHTML=d,k=s(),m(Te.$$.fragment),Ht=s(),ee=i("div"),ee.innerHTML=kn,Rt=s(),m(ye.$$.fragment),Et=s(),be=i("p"),be.innerHTML=wn,Dt=s(),ve=i("p"),ve.textContent=Mn,Xt=s(),ke=i("p"),ke.textContent=Nn,At=s(),we=i("p"),we.innerHTML=zn,Qt=s(),Me=i("p"),Me.innerHTML=Pn,Yt=s(),m(Ne.$$.fragment),Kt=s(),ze=i("ul"),ze.innerHTML=$n,eo=s(),m(Pe.$$.fragment),to=s(),$e=i("ul"),$e.innerHTML=xn,oo=s(),m(xe.$$.fragment),no=s(),L=i("div"),m(qe.$$.fragment),No=s(),ct=i("p"),ct.innerHTML=qn,zo=s(),lt=i("p"),lt.innerHTML=Cn,so=s(),m(Ce.$$.fragment),ro=s(),P=i("div"),m(Fe.$$.fragment),Po=s(),pt=i("p"),pt.textContent=Fn,$o=s(),ht=i("p"),ht.innerHTML=Jn,xo=s(),S=i("div"),m(Je.$$.fragment),qo=s(),ut=i("p"),ut.textContent=jn,Co=s(),mt=i("ul"),mt.innerHTML=Un,Fo=s(),te=i("div"),m(je.$$.fragment),Jo=s(),ft=i("p"),ft.textContent=In,jo=s(),oe=i("div"),m(Ue.$$.fragment),Uo=s(),gt=i("p"),gt.innerHTML=On,ao=s(),m(Ie.$$.fragment),io=s(),D=i("div"),m(Oe.$$.fragment),Io=s(),_t=i("p"),_t.textContent=Gn,co=s(),X=i("div"),m(Ge.$$.fragment),Oo=s(),Tt=i("p"),Tt.textContent=Zn,lo=s(),A=i("div"),m(Ze.$$.fragment),Go=s(),yt=i("p"),yt.textContent=Wn,po=s(),Q=i("div"),m(We.$$.fragment),Zo=s(),bt=i("p"),bt.textContent=Ln,ho=s(),m(Le.$$.fragment),uo=s(),$=i("div"),m(Se.$$.fragment),Wo=s(),vt=i("p"),vt.textContent=Sn,Lo=s(),kt=i("p"),kt.innerHTML=Vn,So=s(),wt=i("p"),wt.innerHTML=Bn,Vo=s(),U=i("div"),m(Ve.$$.fragment),Bo=s(),Mt=i("p"),Mt.innerHTML=Hn,Ho=s(),m(ne.$$.fragment),Ro=s(),m(se.$$.fragment),mo=s(),m(Be.$$.fragment),fo=s(),x=i("div"),m(He.$$.fragment),Eo=s(),Nt=i("p"),Nt.textContent=Rn,Do=s(),zt=i("p"),zt.innerHTML=En,Xo=s(),Pt=i("p"),Pt.innerHTML=Dn,Ao=s(),I=i("div"),m(Re.$$.fragment),Qo=s(),$t=i("p"),$t.innerHTML=Xn,Yo=s(),m(re.$$.fragment),Ko=s(),m(ae.$$.fragment),go=s(),m(Ee.$$.fragment),_o=s(),q=i("div"),m(De.$$.fragment),en=s(),xt=i("p"),xt.textContent=An,tn=s(),qt=i("p"),qt.innerHTML=Qn,on=s(),Ct=i("p"),Ct.innerHTML=Yn,nn=s(),O=i("div"),m(Xe.$$.fragment),sn=s(),Ft=i("p"),Ft.innerHTML=Kn,rn=s(),m(de.$$.fragment),an=s(),m(ie.$$.fragment),To=s(),m(Ae.$$.fragment),yo=s(),C=i("div"),m(Qe.$$.fragment),dn=s(),Jt=i("p"),Jt.textContent=es,cn=s(),jt=i("p"),jt.innerHTML=ts,ln=s(),Ut=i("p"),Ut.innerHTML=os,pn=s(),G=i("div"),m(Ye.$$.fragment),hn=s(),It=i("p"),It.innerHTML=ns,un=s(),m(ce.$$.fragment),mn=s(),m(le.$$.fragment),bo=s(),m(Ke.$$.fragment),vo=s(),F=i("div"),m(et.$$.fragment),fn=s(),Ot=i("p"),Ot.textContent=ss,gn=s(),Gt=i("p"),Gt.innerHTML=rs,_n=s(),Zt=i("p"),Zt.innerHTML=as,Tn=s(),Z=i("div"),m(tt.$$.fragment),yn=s(),Wt=i("p"),Wt.innerHTML=ds,bn=s(),m(pe.$$.fragment),vn=s(),m(he.$$.fragment),ko=s(),m(ot.$$.fragment),wo=s(),Lt=i("p"),this.h()},l(e){const o=hs("svelte-u9bgzb",document.head);a=c(o,"META",{name:!0,content:!0}),o.forEach(t),b=r(e),p=c(e,"P",{}),N(p).forEach(t),h=r(e),v=c(e,"P",{"data-svelte-h":!0}),u(v)!=="svelte-jms5g0"&&(v.innerHTML=d),k=r(e),f(Te.$$.fragment,e),Ht=r(e),ee=c(e,"DIV",{class:!0,"data-svelte-h":!0}),u(ee)!=="svelte-13t8s2t"&&(ee.innerHTML=kn),Rt=r(e),f(ye.$$.fragment,e),Et=r(e),be=c(e,"P",{"data-svelte-h":!0}),u(be)!=="svelte-vg0xav"&&(be.innerHTML=wn),Dt=r(e),ve=c(e,"P",{"data-svelte-h":!0}),u(ve)!=="svelte-s2k052"&&(ve.textContent=Mn),Xt=r(e),ke=c(e,"P",{"data-svelte-h":!0}),u(ke)!=="svelte-vfdo9a"&&(ke.textContent=Nn),At=r(e),we=c(e,"P",{"data-svelte-h":!0}),u(we)!=="svelte-1jvtdli"&&(we.innerHTML=zn),Qt=r(e),Me=c(e,"P",{"data-svelte-h":!0}),u(Me)!=="svelte-mvxxnf"&&(Me.innerHTML=Pn),Yt=r(e),f(Ne.$$.fragment,e),Kt=r(e),ze=c(e,"UL",{"data-svelte-h":!0}),u(ze)!=="svelte-t2skrs"&&(ze.innerHTML=$n),eo=r(e),f(Pe.$$.fragment,e),to=r(e),$e=c(e,"UL",{"data-svelte-h":!0}),u($e)!=="svelte-jwyjs9"&&($e.innerHTML=xn),oo=r(e),f(xe.$$.fragment,e),no=r(e),L=c(e,"DIV",{class:!0});var Y=N(L);f(qe.$$.fragment,Y),No=r(Y),ct=c(Y,"P",{"data-svelte-h":!0}),u(ct)!=="svelte-1g1htfm"&&(ct.innerHTML=qn),zo=r(Y),lt=c(Y,"P",{"data-svelte-h":!0}),u(lt)!=="svelte-1ek1ss9"&&(lt.innerHTML=Cn),Y.forEach(t),so=r(e),f(Ce.$$.fragment,e),ro=r(e),P=c(e,"DIV",{class:!0});var J=N(P);f(Fe.$$.fragment,J),Po=r(J),pt=c(J,"P",{"data-svelte-h":!0}),u(pt)!=="svelte-rzl6zy"&&(pt.textContent=Fn),$o=r(J),ht=c(J,"P",{"data-svelte-h":!0}),u(ht)!=="svelte-ntrhio"&&(ht.innerHTML=Jn),xo=r(J),S=c(J,"DIV",{class:!0});var K=N(S);f(Je.$$.fragment,K),qo=r(K),ut=c(K,"P",{"data-svelte-h":!0}),u(ut)!=="svelte-t7qurq"&&(ut.textContent=jn),Co=r(K),mt=c(K,"UL",{"data-svelte-h":!0}),u(mt)!=="svelte-xi6653"&&(mt.innerHTML=Un),K.forEach(t),Fo=r(J),te=c(J,"DIV",{class:!0});var nt=N(te);f(je.$$.fragment,nt),Jo=r(nt),ft=c(nt,"P",{"data-svelte-h":!0}),u(ft)!=="svelte-b3k2yi"&&(ft.textContent=In),nt.forEach(t),jo=r(J),oe=c(J,"DIV",{class:!0});var st=N(oe);f(Ue.$$.fragment,st),Uo=r(st),gt=c(st,"P",{"data-svelte-h":!0}),u(gt)!=="svelte-1f4f5kp"&&(gt.innerHTML=On),st.forEach(t),J.forEach(t),ao=r(e),f(Ie.$$.fragment,e),io=r(e),D=c(e,"DIV",{class:!0});var rt=N(D);f(Oe.$$.fragment,rt),Io=r(rt),_t=c(rt,"P",{"data-svelte-h":!0}),u(_t)!=="svelte-1dobm33"&&(_t.textContent=Gn),rt.forEach(t),co=r(e),X=c(e,"DIV",{class:!0});var at=N(X);f(Ge.$$.fragment,at),Oo=r(at),Tt=c(at,"P",{"data-svelte-h":!0}),u(Tt)!=="svelte-k12cko"&&(Tt.textContent=Zn),at.forEach(t),lo=r(e),A=c(e,"DIV",{class:!0});var dt=N(A);f(Ze.$$.fragment,dt),Go=r(dt),yt=c(dt,"P",{"data-svelte-h":!0}),u(yt)!=="svelte-1kt4x95"&&(yt.textContent=Wn),dt.forEach(t),po=r(e),Q=c(e,"DIV",{class:!0});var it=N(Q);f(We.$$.fragment,it),Zo=r(it),bt=c(it,"P",{"data-svelte-h":!0}),u(bt)!=="svelte-1kt4x95"&&(bt.textContent=Ln),it.forEach(t),ho=r(e),f(Le.$$.fragment,e),uo=r(e),$=c(e,"DIV",{class:!0});var j=N($);f(Se.$$.fragment,j),Wo=r(j),vt=c(j,"P",{"data-svelte-h":!0}),u(vt)!=="svelte-1xaaqb9"&&(vt.textContent=Sn),Lo=r(j),kt=c(j,"P",{"data-svelte-h":!0}),u(kt)!=="svelte-q52n56"&&(kt.innerHTML=Vn),So=r(j),wt=c(j,"P",{"data-svelte-h":!0}),u(wt)!=="svelte-hswkmf"&&(wt.innerHTML=Bn),Vo=r(j),U=c(j,"DIV",{class:!0});var ue=N(U);f(Ve.$$.fragment,ue),Bo=r(ue),Mt=c(ue,"P",{"data-svelte-h":!0}),u(Mt)!=="svelte-onvfdy"&&(Mt.innerHTML=Hn),Ho=r(ue),f(ne.$$.fragment,ue),Ro=r(ue),f(se.$$.fragment,ue),ue.forEach(t),j.forEach(t),mo=r(e),f(Be.$$.fragment,e),fo=r(e),x=c(e,"DIV",{class:!0});var V=N(x);f(He.$$.fragment,V),Eo=r(V),Nt=c(V,"P",{"data-svelte-h":!0}),u(Nt)!=="svelte-g2u0mt"&&(Nt.textContent=Rn),Do=r(V),zt=c(V,"P",{"data-svelte-h":!0}),u(zt)!=="svelte-q52n56"&&(zt.innerHTML=En),Xo=r(V),Pt=c(V,"P",{"data-svelte-h":!0}),u(Pt)!=="svelte-hswkmf"&&(Pt.innerHTML=Dn),Ao=r(V),I=c(V,"DIV",{class:!0});var me=N(I);f(Re.$$.fragment,me),Qo=r(me),$t=c(me,"P",{"data-svelte-h":!0}),u($t)!=="svelte-256bn2"&&($t.innerHTML=Xn),Yo=r(me),f(re.$$.fragment,me),Ko=r(me),f(ae.$$.fragment,me),me.forEach(t),V.forEach(t),go=r(e),f(Ee.$$.fragment,e),_o=r(e),q=c(e,"DIV",{class:!0});var B=N(q);f(De.$$.fragment,B),en=r(B),xt=c(B,"P",{"data-svelte-h":!0}),u(xt)!=="svelte-u0j049"&&(xt.textContent=An),tn=r(B),qt=c(B,"P",{"data-svelte-h":!0}),u(qt)!=="svelte-q52n56"&&(qt.innerHTML=Qn),on=r(B),Ct=c(B,"P",{"data-svelte-h":!0}),u(Ct)!=="svelte-hswkmf"&&(Ct.innerHTML=Yn),nn=r(B),O=c(B,"DIV",{class:!0});var fe=N(O);f(Xe.$$.fragment,fe),sn=r(fe),Ft=c(fe,"P",{"data-svelte-h":!0}),u(Ft)!=="svelte-y6xyqi"&&(Ft.innerHTML=Kn),rn=r(fe),f(de.$$.fragment,fe),an=r(fe),f(ie.$$.fragment,fe),fe.forEach(t),B.forEach(t),To=r(e),f(Ae.$$.fragment,e),yo=r(e),C=c(e,"DIV",{class:!0});var H=N(C);f(Qe.$$.fragment,H),dn=r(H),Jt=c(H,"P",{"data-svelte-h":!0}),u(Jt)!=="svelte-6185sb"&&(Jt.textContent=es),cn=r(H),jt=c(H,"P",{"data-svelte-h":!0}),u(jt)!=="svelte-q52n56"&&(jt.innerHTML=ts),ln=r(H),Ut=c(H,"P",{"data-svelte-h":!0}),u(Ut)!=="svelte-hswkmf"&&(Ut.innerHTML=os),pn=r(H),G=c(H,"DIV",{class:!0});var ge=N(G);f(Ye.$$.fragment,ge),hn=r(ge),It=c(ge,"P",{"data-svelte-h":!0}),u(It)!=="svelte-1u7tii2"&&(It.innerHTML=ns),un=r(ge),f(ce.$$.fragment,ge),mn=r(ge),f(le.$$.fragment,ge),ge.forEach(t),H.forEach(t),bo=r(e),f(Ke.$$.fragment,e),vo=r(e),F=c(e,"DIV",{class:!0});var R=N(F);f(et.$$.fragment,R),fn=r(R),Ot=c(R,"P",{"data-svelte-h":!0}),u(Ot)!=="svelte-71w8gx"&&(Ot.textContent=ss),gn=r(R),Gt=c(R,"P",{"data-svelte-h":!0}),u(Gt)!=="svelte-q52n56"&&(Gt.innerHTML=rs),_n=r(R),Zt=c(R,"P",{"data-svelte-h":!0}),u(Zt)!=="svelte-hswkmf"&&(Zt.innerHTML=as),Tn=r(R),Z=c(R,"DIV",{class:!0});var _e=N(Z);f(tt.$$.fragment,_e),yn=r(_e),Wt=c(_e,"P",{"data-svelte-h":!0}),u(Wt)!=="svelte-1p6gwo6"&&(Wt.innerHTML=ds),bn=r(_e),f(pe.$$.fragment,_e),vn=r(_e),f(he.$$.fragment,_e),_e.forEach(t),R.forEach(t),ko=r(e),f(ot.$$.fragment,e),wo=r(e),Lt=c(e,"P",{}),N(Lt).forEach(t),this.h()},h(){M(a,"name","hf:doc:metadata"),M(a,"content",Ns),M(ee,"class","flex flex-wrap space-x-1"),M(L,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),M(S,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),M(te,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),M(oe,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),M(P,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),M(D,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),M(X,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),M(A,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),M(Q,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),M(U,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),M($,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),M(I,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),M(x,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),M(O,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),M(q,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),M(G,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),M(C,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),M(Z,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),M(F,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,o){n(document.head,a),l(e,b,o),l(e,p,o),l(e,h,o),l(e,v,o),l(e,k,o),g(Te,e,o),l(e,Ht,o),l(e,ee,o),l(e,Rt,o),g(ye,e,o),l(e,Et,o),l(e,be,o),l(e,Dt,o),l(e,ve,o),l(e,Xt,o),l(e,ke,o),l(e,At,o),l(e,we,o),l(e,Qt,o),l(e,Me,o),l(e,Yt,o),g(Ne,e,o),l(e,Kt,o),l(e,ze,o),l(e,eo,o),g(Pe,e,o),l(e,to,o),l(e,$e,o),l(e,oo,o),g(xe,e,o),l(e,no,o),l(e,L,o),g(qe,L,null),n(L,No),n(L,ct),n(L,zo),n(L,lt),l(e,so,o),g(Ce,e,o),l(e,ro,o),l(e,P,o),g(Fe,P,null),n(P,Po),n(P,pt),n(P,$o),n(P,ht),n(P,xo),n(P,S),g(Je,S,null),n(S,qo),n(S,ut),n(S,Co),n(S,mt),n(P,Fo),n(P,te),g(je,te,null),n(te,Jo),n(te,ft),n(P,jo),n(P,oe),g(Ue,oe,null),n(oe,Uo),n(oe,gt),l(e,ao,o),g(Ie,e,o),l(e,io,o),l(e,D,o),g(Oe,D,null),n(D,Io),n(D,_t),l(e,co,o),l(e,X,o),g(Ge,X,null),n(X,Oo),n(X,Tt),l(e,lo,o),l(e,A,o),g(Ze,A,null),n(A,Go),n(A,yt),l(e,po,o),l(e,Q,o),g(We,Q,null),n(Q,Zo),n(Q,bt),l(e,ho,o),g(Le,e,o),l(e,uo,o),l(e,$,o),g(Se,$,null),n($,Wo),n($,vt),n($,Lo),n($,kt),n($,So),n($,wt),n($,Vo),n($,U),g(Ve,U,null),n(U,Bo),n(U,Mt),n(U,Ho),g(ne,U,null),n(U,Ro),g(se,U,null),l(e,mo,o),g(Be,e,o),l(e,fo,o),l(e,x,o),g(He,x,null),n(x,Eo),n(x,Nt),n(x,Do),n(x,zt),n(x,Xo),n(x,Pt),n(x,Ao),n(x,I),g(Re,I,null),n(I,Qo),n(I,$t),n(I,Yo),g(re,I,null),n(I,Ko),g(ae,I,null),l(e,go,o),g(Ee,e,o),l(e,_o,o),l(e,q,o),g(De,q,null),n(q,en),n(q,xt),n(q,tn),n(q,qt),n(q,on),n(q,Ct),n(q,nn),n(q,O),g(Xe,O,null),n(O,sn),n(O,Ft),n(O,rn),g(de,O,null),n(O,an),g(ie,O,null),l(e,To,o),g(Ae,e,o),l(e,yo,o),l(e,C,o),g(Qe,C,null),n(C,dn),n(C,Jt),n(C,cn),n(C,jt),n(C,ln),n(C,Ut),n(C,pn),n(C,G),g(Ye,G,null),n(G,hn),n(G,It),n(G,un),g(ce,G,null),n(G,mn),g(le,G,null),l(e,bo,o),g(Ke,e,o),l(e,vo,o),l(e,F,o),g(et,F,null),n(F,fn),n(F,Ot),n(F,gn),n(F,Gt),n(F,_n),n(F,Zt),n(F,Tn),n(F,Z),g(tt,Z,null),n(Z,yn),n(Z,Wt),n(Z,bn),g(pe,Z,null),n(Z,vn),g(he,Z,null),l(e,ko,o),g(ot,e,o),l(e,wo,o),l(e,Lt,o),Mo=!0},p(e,[o]){const Y={};o&2&&(Y.$$scope={dirty:o,ctx:e}),ne.$set(Y);const J={};o&2&&(J.$$scope={dirty:o,ctx:e}),se.$set(J);const K={};o&2&&(K.$$scope={dirty:o,ctx:e}),re.$set(K);const nt={};o&2&&(nt.$$scope={dirty:o,ctx:e}),ae.$set(nt);const st={};o&2&&(st.$$scope={dirty:o,ctx:e}),de.$set(st);const rt={};o&2&&(rt.$$scope={dirty:o,ctx:e}),ie.$set(rt);const at={};o&2&&(at.$$scope={dirty:o,ctx:e}),ce.$set(at);const dt={};o&2&&(dt.$$scope={dirty:o,ctx:e}),le.$set(dt);const it={};o&2&&(it.$$scope={dirty:o,ctx:e}),pe.$set(it);const j={};o&2&&(j.$$scope={dirty:o,ctx:e}),he.$set(j)},i(e){Mo||(_(Te.$$.fragment,e),_(ye.$$.fragment,e),_(Ne.$$.fragment,e),_(Pe.$$.fragment,e),_(xe.$$.fragment,e),_(qe.$$.fragment,e),_(Ce.$$.fragment,e),_(Fe.$$.fragment,e),_(Je.$$.fragment,e),_(je.$$.fragment,e),_(Ue.$$.fragment,e),_(Ie.$$.fragment,e),_(Oe.$$.fragment,e),_(Ge.$$.fragment,e),_(Ze.$$.fragment,e),_(We.$$.fragment,e),_(Le.$$.fragment,e),_(Se.$$.fragment,e),_(Ve.$$.fragment,e),_(ne.$$.fragment,e),_(se.$$.fragment,e),_(Be.$$.fragment,e),_(He.$$.fragment,e),_(Re.$$.fragment,e),_(re.$$.fragment,e),_(ae.$$.fragment,e),_(Ee.$$.fragment,e),_(De.$$.fragment,e),_(Xe.$$.fragment,e),_(de.$$.fragment,e),_(ie.$$.fragment,e),_(Ae.$$.fragment,e),_(Qe.$$.fragment,e),_(Ye.$$.fragment,e),_(ce.$$.fragment,e),_(le.$$.fragment,e),_(Ke.$$.fragment,e),_(et.$$.fragment,e),_(tt.$$.fragment,e),_(pe.$$.fragment,e),_(he.$$.fragment,e),_(ot.$$.fragment,e),Mo=!0)},o(e){T(Te.$$.fragment,e),T(ye.$$.fragment,e),T(Ne.$$.fragment,e),T(Pe.$$.fragment,e),T(xe.$$.fragment,e),T(qe.$$.fragment,e),T(Ce.$$.fragment,e),T(Fe.$$.fragment,e),T(Je.$$.fragment,e),T(je.$$.fragment,e),T(Ue.$$.fragment,e),T(Ie.$$.fragment,e),T(Oe.$$.fragment,e),T(Ge.$$.fragment,e),T(Ze.$$.fragment,e),T(We.$$.fragment,e),T(Le.$$.fragment,e),T(Se.$$.fragment,e),T(Ve.$$.fragment,e),T(ne.$$.fragment,e),T(se.$$.fragment,e),T(Be.$$.fragment,e),T(He.$$.fragment,e),T(Re.$$.fragment,e),T(re.$$.fragment,e),T(ae.$$.fragment,e),T(Ee.$$.fragment,e),T(De.$$.fragment,e),T(Xe.$$.fragment,e),T(de.$$.fragment,e),T(ie.$$.fragment,e),T(Ae.$$.fragment,e),T(Qe.$$.fragment,e),T(Ye.$$.fragment,e),T(ce.$$.fragment,e),T(le.$$.fragment,e),T(Ke.$$.fragment,e),T(et.$$.fragment,e),T(tt.$$.fragment,e),T(pe.$$.fragment,e),T(he.$$.fragment,e),T(ot.$$.fragment,e),Mo=!1},d(e){e&&(t(b),t(p),t(h),t(v),t(k),t(Ht),t(ee),t(Rt),t(Et),t(be),t(Dt),t(ve),t(Xt),t(ke),t(At),t(we),t(Qt),t(Me),t(Yt),t(Kt),t(ze),t(eo),t(to),t($e),t(oo),t(no),t(L),t(so),t(ro),t(P),t(ao),t(io),t(D),t(co),t(X),t(lo),t(A),t(po),t(Q),t(ho),t(uo),t($),t(mo),t(fo),t(x),t(go),t(_o),t(q),t(To),t(yo),t(C),t(bo),t(vo),t(F),t(ko),t(wo),t(Lt)),t(a),y(Te,e),y(ye,e),y(Ne,e),y(Pe,e),y(xe,e),y(qe),y(Ce,e),y(Fe),y(Je),y(je),y(Ue),y(Ie,e),y(Oe),y(Ge),y(Ze),y(We),y(Le,e),y(Se),y(Ve),y(ne),y(se),y(Be,e),y(He),y(Re),y(re),y(ae),y(Ee,e),y(De),y(Xe),y(de),y(ie),y(Ae,e),y(Qe),y(Ye),y(ce),y(le),y(Ke,e),y(et),y(tt),y(pe),y(he),y(ot,e)}}}const Ns='{"title":"ProphetNet","local":"prophetnet","sections":[{"title":"Overview","local":"overview","sections":[],"depth":2},{"title":"Usage tips","local":"usage-tips","sections":[],"depth":2},{"title":"Resources","local":"resources","sections":[],"depth":2},{"title":"ProphetNetConfig","local":"transformers.ProphetNetConfig","sections":[],"depth":2},{"title":"ProphetNetTokenizer","local":"transformers.ProphetNetTokenizer","sections":[],"depth":2},{"title":"ProphetNet specific outputs","local":"transformers.models.prophetnet.modeling_prophetnet.ProphetNetSeq2SeqLMOutput","sections":[],"depth":2},{"title":"ProphetNetModel","local":"transformers.ProphetNetModel","sections":[],"depth":2},{"title":"ProphetNetEncoder","local":"transformers.ProphetNetEncoder","sections":[],"depth":2},{"title":"ProphetNetDecoder","local":"transformers.ProphetNetDecoder","sections":[],"depth":2},{"title":"ProphetNetForConditionalGeneration","local":"transformers.ProphetNetForConditionalGeneration","sections":[],"depth":2},{"title":"ProphetNetForCausalLM","local":"transformers.ProphetNetForCausalLM","sections":[],"depth":2}],"depth":1}';function zs(w){return cs(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class js extends ls{constructor(a){super(),ps(this,a,zs,Ms,is,{})}}export{js as component};
