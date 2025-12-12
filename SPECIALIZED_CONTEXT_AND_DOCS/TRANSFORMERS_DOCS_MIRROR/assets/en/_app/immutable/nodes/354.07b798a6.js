import{s as sa,o as aa,n as B}from"../chunks/scheduler.18a86fab.js";import{S as ra,i as ia,g as p,s as a,r as u,A as la,h as m,f as s,c as r,j,x as h,u as f,k as Q,y as l,a as c,v as g,d as _,t as b,w as M}from"../chunks/index.98837b22.js";import{T as pe}from"../chunks/Tip.77304350.js";import{D as J}from"../chunks/Docstring.a1ef7999.js";import{C as Y}from"../chunks/CodeBlock.8d0c2e8a.js";import{E as ce}from"../chunks/ExampleCodeBlock.8c3ee1f9.js";import{H as C,E as da}from"../chunks/getInferenceSnippets.06c2775f.js";function ca(k){let t,y=`This model is in maintenance mode only, we don’t accept any new PRs changing its code.
If you run into any issues running this model, please reinstall the last version that supported this model: v4.40.2.
You can do so by running the following command: <code>pip install -U transformers==4.40.2</code>.`;return{c(){t=p("p"),t.innerHTML=y},l(i){t=m(i,"P",{"data-svelte-h":!0}),h(t)!=="svelte-1sq0hrb"&&(t.innerHTML=y)},m(i,d){c(i,t,d)},p:B,d(i){i&&s(t)}}}function pa(k){let t,y="Examples:",i,d,T;return d=new Y({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMFFEUUJlcnRNb2RlbCUyQyUyMFFEUUJlcnRDb25maWclMEElMEElMjMlMjBJbml0aWFsaXppbmclMjBhJTIwUURRQkVSVCUyMGdvb2dsZS1iZXJ0JTJGYmVydC1iYXNlLXVuY2FzZWQlMjBzdHlsZSUyMGNvbmZpZ3VyYXRpb24lMEFjb25maWd1cmF0aW9uJTIwJTNEJTIwUURRQmVydENvbmZpZygpJTBBJTBBJTIzJTIwSW5pdGlhbGl6aW5nJTIwYSUyMG1vZGVsJTIwZnJvbSUyMHRoZSUyMGdvb2dsZS1iZXJ0JTJGYmVydC1iYXNlLXVuY2FzZWQlMjBzdHlsZSUyMGNvbmZpZ3VyYXRpb24lMEFtb2RlbCUyMCUzRCUyMFFEUUJlcnRNb2RlbChjb25maWd1cmF0aW9uKSUwQSUwQSUyMyUyMEFjY2Vzc2luZyUyMHRoZSUyMG1vZGVsJTIwY29uZmlndXJhdGlvbiUwQWNvbmZpZ3VyYXRpb24lMjAlM0QlMjBtb2RlbC5jb25maWc=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> QDQBertModel, QDQBertConfig

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a QDQBERT google-bert/bert-base-uncased style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = QDQBertConfig()

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a model from the google-bert/bert-base-uncased style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = QDQBertModel(configuration)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Accessing the model configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = model.config`,wrap:!1}}),{c(){t=p("p"),t.textContent=y,i=a(),u(d.$$.fragment)},l(n){t=m(n,"P",{"data-svelte-h":!0}),h(t)!=="svelte-kvfsh7"&&(t.textContent=y),i=r(n),f(d.$$.fragment,n)},m(n,w){c(n,t,w),c(n,i,w),g(d,n,w),T=!0},p:B,i(n){T||(_(d.$$.fragment,n),T=!0)},o(n){b(d.$$.fragment,n),T=!1},d(n){n&&(s(t),s(i)),M(d,n)}}}function ma(k){let t,y=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=p("p"),t.innerHTML=y},l(i){t=m(i,"P",{"data-svelte-h":!0}),h(t)!=="svelte-fincs2"&&(t.innerHTML=y)},m(i,d){c(i,t,d)},p:B,d(i){i&&s(t)}}}function ha(k){let t,y="Example:",i,d,T;return d=new Y({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBRRFFCZXJ0TW9kZWwlMEFpbXBvcnQlMjB0b3JjaCUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMmdvb2dsZS1iZXJ0JTJGYmVydC1iYXNlLXVuY2FzZWQlMjIpJTBBbW9kZWwlMjAlM0QlMjBRRFFCZXJ0TW9kZWwuZnJvbV9wcmV0cmFpbmVkKCUyMmdvb2dsZS1iZXJ0JTJGYmVydC1iYXNlLXVuY2FzZWQlMjIpJTBBJTBBaW5wdXRzJTIwJTNEJTIwdG9rZW5pemVyKCUyMkhlbGxvJTJDJTIwbXklMjBkb2clMjBpcyUyMGN1dGUlMjIlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQW91dHB1dHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyklMEElMEFsYXN0X2hpZGRlbl9zdGF0ZXMlMjAlM0QlMjBvdXRwdXRzLmxhc3RfaGlkZGVuX3N0YXRl",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, QDQBertModel
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;google-bert/bert-base-uncased&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = QDQBertModel.from_pretrained(<span class="hljs-string">&quot;google-bert/bert-base-uncased&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs)

<span class="hljs-meta">&gt;&gt;&gt; </span>last_hidden_states = outputs.last_hidden_state`,wrap:!1}}),{c(){t=p("p"),t.textContent=y,i=a(),u(d.$$.fragment)},l(n){t=m(n,"P",{"data-svelte-h":!0}),h(t)!=="svelte-11lpom8"&&(t.textContent=y),i=r(n),f(d.$$.fragment,n)},m(n,w){c(n,t,w),c(n,i,w),g(d,n,w),T=!0},p:B,i(n){T||(_(d.$$.fragment,n),T=!0)},o(n){b(d.$$.fragment,n),T=!1},d(n){n&&(s(t),s(i)),M(d,n)}}}function ua(k){let t,y=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=p("p"),t.innerHTML=y},l(i){t=m(i,"P",{"data-svelte-h":!0}),h(t)!=="svelte-fincs2"&&(t.innerHTML=y)},m(i,d){c(i,t,d)},p:B,d(i){i&&s(t)}}}function fa(k){let t,y="Example:",i,d,T;return d=new Y({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBRRFFCZXJ0TE1IZWFkTW9kZWwlMkMlMjBRRFFCZXJ0Q29uZmlnJTBBaW1wb3J0JTIwdG9yY2glMEElMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJnb29nbGUtYmVydCUyRmJlcnQtYmFzZS1jYXNlZCUyMiklMEFjb25maWclMjAlM0QlMjBRRFFCZXJ0Q29uZmlnLmZyb21fcHJldHJhaW5lZCglMjJnb29nbGUtYmVydCUyRmJlcnQtYmFzZS1jYXNlZCUyMiklMEFjb25maWcuaXNfZGVjb2RlciUyMCUzRCUyMFRydWUlMEFtb2RlbCUyMCUzRCUyMFFEUUJlcnRMTUhlYWRNb2RlbC5mcm9tX3ByZXRyYWluZWQoJTIyZ29vZ2xlLWJlcnQlMkZiZXJ0LWJhc2UtY2FzZWQlMjIlMkMlMjBjb25maWclM0Rjb25maWcpJTBBJTBBaW5wdXRzJTIwJTNEJTIwdG9rZW5pemVyKCUyMkhlbGxvJTJDJTIwbXklMjBkb2clMjBpcyUyMGN1dGUlMjIlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQW91dHB1dHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyklMEElMEFwcmVkaWN0aW9uX2xvZ2l0cyUyMCUzRCUyMG91dHB1dHMubG9naXRz",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, QDQBertLMHeadModel, QDQBertConfig
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;google-bert/bert-base-cased&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>config = QDQBertConfig.from_pretrained(<span class="hljs-string">&quot;google-bert/bert-base-cased&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>config.is_decoder = <span class="hljs-literal">True</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = QDQBertLMHeadModel.from_pretrained(<span class="hljs-string">&quot;google-bert/bert-base-cased&quot;</span>, config=config)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs)

<span class="hljs-meta">&gt;&gt;&gt; </span>prediction_logits = outputs.logits`,wrap:!1}}),{c(){t=p("p"),t.textContent=y,i=a(),u(d.$$.fragment)},l(n){t=m(n,"P",{"data-svelte-h":!0}),h(t)!=="svelte-11lpom8"&&(t.textContent=y),i=r(n),f(d.$$.fragment,n)},m(n,w){c(n,t,w),c(n,i,w),g(d,n,w),T=!0},p:B,i(n){T||(_(d.$$.fragment,n),T=!0)},o(n){b(d.$$.fragment,n),T=!1},d(n){n&&(s(t),s(i)),M(d,n)}}}function ga(k){let t,y=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=p("p"),t.innerHTML=y},l(i){t=m(i,"P",{"data-svelte-h":!0}),h(t)!=="svelte-fincs2"&&(t.innerHTML=y)},m(i,d){c(i,t,d)},p:B,d(i){i&&s(t)}}}function _a(k){let t,y="Example:",i,d,T;return d=new Y({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBRRFFCZXJ0Rm9yTWFza2VkTE0lMEFpbXBvcnQlMjB0b3JjaCUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMmdvb2dsZS1iZXJ0JTJGYmVydC1iYXNlLXVuY2FzZWQlMjIpJTBBbW9kZWwlMjAlM0QlMjBRRFFCZXJ0Rm9yTWFza2VkTE0uZnJvbV9wcmV0cmFpbmVkKCUyMmdvb2dsZS1iZXJ0JTJGYmVydC1iYXNlLXVuY2FzZWQlMjIpJTBBJTBBaW5wdXRzJTIwJTNEJTIwdG9rZW5pemVyKCUyMlRoZSUyMGNhcGl0YWwlMjBvZiUyMEZyYW5jZSUyMGlzJTIwJTVCTUFTSyU1RC4lMjIlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQSUwQXdpdGglMjB0b3JjaC5ub19ncmFkKCklM0ElMEElMjAlMjAlMjAlMjBsb2dpdHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cykubG9naXRzJTBBJTBBJTIzJTIwcmV0cmlldmUlMjBpbmRleCUyMG9mJTIwJTVCTUFTSyU1RCUwQW1hc2tfdG9rZW5faW5kZXglMjAlM0QlMjAoaW5wdXRzLmlucHV0X2lkcyUyMCUzRCUzRCUyMHRva2VuaXplci5tYXNrX3Rva2VuX2lkKSU1QjAlNUQubm9uemVybyhhc190dXBsZSUzRFRydWUpJTVCMCU1RCUwQSUwQXByZWRpY3RlZF90b2tlbl9pZCUyMCUzRCUyMGxvZ2l0cyU1QjAlMkMlMjBtYXNrX3Rva2VuX2luZGV4JTVELmFyZ21heChheGlzJTNELTEpJTBBJTBBbGFiZWxzJTIwJTNEJTIwdG9rZW5pemVyKCUyMlRoZSUyMGNhcGl0YWwlMjBvZiUyMEZyYW5jZSUyMGlzJTIwUGFyaXMuJTIyJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiklNUIlMjJpbnB1dF9pZHMlMjIlNUQlMEElMjMlMjBtYXNrJTIwbGFiZWxzJTIwb2YlMjBub24tJTVCTUFTSyU1RCUyMHRva2VucyUwQWxhYmVscyUyMCUzRCUyMHRvcmNoLndoZXJlKGlucHV0cy5pbnB1dF9pZHMlMjAlM0QlM0QlMjB0b2tlbml6ZXIubWFza190b2tlbl9pZCUyQyUyMGxhYmVscyUyQyUyMC0xMDApJTBBJTBBb3V0cHV0cyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzJTJDJTIwbGFiZWxzJTNEbGFiZWxzKQ==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, QDQBertForMaskedLM
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;google-bert/bert-base-uncased&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = QDQBertForMaskedLM.from_pretrained(<span class="hljs-string">&quot;google-bert/bert-base-uncased&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;The capital of France is [MASK].&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    logits = model(**inputs).logits

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># retrieve index of [MASK]</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[<span class="hljs-number">0</span>].nonzero(as_tuple=<span class="hljs-literal">True</span>)[<span class="hljs-number">0</span>]

<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_token_id = logits[<span class="hljs-number">0</span>, mask_token_index].argmax(axis=-<span class="hljs-number">1</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>labels = tokenizer(<span class="hljs-string">&quot;The capital of France is Paris.&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)[<span class="hljs-string">&quot;input_ids&quot;</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># mask labels of non-[MASK] tokens</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.where(inputs.input_ids == tokenizer.mask_token_id, labels, -<span class="hljs-number">100</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs, labels=labels)`,wrap:!1}}),{c(){t=p("p"),t.textContent=y,i=a(),u(d.$$.fragment)},l(n){t=m(n,"P",{"data-svelte-h":!0}),h(t)!=="svelte-11lpom8"&&(t.textContent=y),i=r(n),f(d.$$.fragment,n)},m(n,w){c(n,t,w),c(n,i,w),g(d,n,w),T=!0},p:B,i(n){T||(_(d.$$.fragment,n),T=!0)},o(n){b(d.$$.fragment,n),T=!1},d(n){n&&(s(t),s(i)),M(d,n)}}}function ba(k){let t,y=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=p("p"),t.innerHTML=y},l(i){t=m(i,"P",{"data-svelte-h":!0}),h(t)!=="svelte-fincs2"&&(t.innerHTML=y)},m(i,d){c(i,t,d)},p:B,d(i){i&&s(t)}}}function Ma(k){let t,y="Example of single-label classification:",i,d,T;return d=new Y({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Rva2VuaXplciUyQyUyMFFEUUJlcnRGb3JTZXF1ZW5jZUNsYXNzaWZpY2F0aW9uJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIyZ29vZ2xlLWJlcnQlMkZiZXJ0LWJhc2UtdW5jYXNlZCUyMiklMEFtb2RlbCUyMCUzRCUyMFFEUUJlcnRGb3JTZXF1ZW5jZUNsYXNzaWZpY2F0aW9uLmZyb21fcHJldHJhaW5lZCglMjJnb29nbGUtYmVydCUyRmJlcnQtYmFzZS11bmNhc2VkJTIyKSUwQSUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplciglMjJIZWxsbyUyQyUyMG15JTIwZG9nJTIwaXMlMjBjdXRlJTIyJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiklMEElMEF3aXRoJTIwdG9yY2gubm9fZ3JhZCgpJTNBJTBBJTIwJTIwJTIwJTIwbG9naXRzJTIwJTNEJTIwbW9kZWwoKippbnB1dHMpLmxvZ2l0cyUwQSUwQXByZWRpY3RlZF9jbGFzc19pZCUyMCUzRCUyMGxvZ2l0cy5hcmdtYXgoKS5pdGVtKCklMEElMEElMjMlMjBUbyUyMHRyYWluJTIwYSUyMG1vZGVsJTIwb24lMjAlNjBudW1fbGFiZWxzJTYwJTIwY2xhc3NlcyUyQyUyMHlvdSUyMGNhbiUyMHBhc3MlMjAlNjBudW1fbGFiZWxzJTNEbnVtX2xhYmVscyU2MCUyMHRvJTIwJTYwLmZyb21fcHJldHJhaW5lZCguLi4pJTYwJTBBbnVtX2xhYmVscyUyMCUzRCUyMGxlbihtb2RlbC5jb25maWcuaWQybGFiZWwpJTBBbW9kZWwlMjAlM0QlMjBRRFFCZXJ0Rm9yU2VxdWVuY2VDbGFzc2lmaWNhdGlvbi5mcm9tX3ByZXRyYWluZWQoJTIyZ29vZ2xlLWJlcnQlMkZiZXJ0LWJhc2UtdW5jYXNlZCUyMiUyQyUyMG51bV9sYWJlbHMlM0RudW1fbGFiZWxzKSUwQSUwQWxhYmVscyUyMCUzRCUyMHRvcmNoLnRlbnNvciglNUIxJTVEKSUwQWxvc3MlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyUyQyUyMGxhYmVscyUzRGxhYmVscykubG9zcw==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, QDQBertForSequenceClassification

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;google-bert/bert-base-uncased&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = QDQBertForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;google-bert/bert-base-uncased&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    logits = model(**inputs).logits

<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_class_id = logits.argmax().item()

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># To train a model on \`num_labels\` classes, you can pass \`num_labels=num_labels\` to \`.from_pretrained(...)\`</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>num_labels = <span class="hljs-built_in">len</span>(model.config.id2label)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = QDQBertForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;google-bert/bert-base-uncased&quot;</span>, num_labels=num_labels)

<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.tensor([<span class="hljs-number">1</span>])
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = model(**inputs, labels=labels).loss`,wrap:!1}}),{c(){t=p("p"),t.textContent=y,i=a(),u(d.$$.fragment)},l(n){t=m(n,"P",{"data-svelte-h":!0}),h(t)!=="svelte-ykxpe4"&&(t.textContent=y),i=r(n),f(d.$$.fragment,n)},m(n,w){c(n,t,w),c(n,i,w),g(d,n,w),T=!0},p:B,i(n){T||(_(d.$$.fragment,n),T=!0)},o(n){b(d.$$.fragment,n),T=!1},d(n){n&&(s(t),s(i)),M(d,n)}}}function ya(k){let t,y="Example of multi-label classification:",i,d,T;return d=new Y({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Rva2VuaXplciUyQyUyMFFEUUJlcnRGb3JTZXF1ZW5jZUNsYXNzaWZpY2F0aW9uJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIyZ29vZ2xlLWJlcnQlMkZiZXJ0LWJhc2UtdW5jYXNlZCUyMiklMEFtb2RlbCUyMCUzRCUyMFFEUUJlcnRGb3JTZXF1ZW5jZUNsYXNzaWZpY2F0aW9uLmZyb21fcHJldHJhaW5lZCglMjJnb29nbGUtYmVydCUyRmJlcnQtYmFzZS11bmNhc2VkJTIyJTJDJTIwcHJvYmxlbV90eXBlJTNEJTIybXVsdGlfbGFiZWxfY2xhc3NpZmljYXRpb24lMjIpJTBBJTBBaW5wdXRzJTIwJTNEJTIwdG9rZW5pemVyKCUyMkhlbGxvJTJDJTIwbXklMjBkb2clMjBpcyUyMGN1dGUlMjIlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQSUwQXdpdGglMjB0b3JjaC5ub19ncmFkKCklM0ElMEElMjAlMjAlMjAlMjBsb2dpdHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cykubG9naXRzJTBBJTBBcHJlZGljdGVkX2NsYXNzX2lkcyUyMCUzRCUyMHRvcmNoLmFyYW5nZSgwJTJDJTIwbG9naXRzLnNoYXBlJTVCLTElNUQpJTVCdG9yY2guc2lnbW9pZChsb2dpdHMpLnNxdWVlemUoZGltJTNEMCklMjAlM0UlMjAwLjUlNUQlMEElMEElMjMlMjBUbyUyMHRyYWluJTIwYSUyMG1vZGVsJTIwb24lMjAlNjBudW1fbGFiZWxzJTYwJTIwY2xhc3NlcyUyQyUyMHlvdSUyMGNhbiUyMHBhc3MlMjAlNjBudW1fbGFiZWxzJTNEbnVtX2xhYmVscyU2MCUyMHRvJTIwJTYwLmZyb21fcHJldHJhaW5lZCguLi4pJTYwJTBBbnVtX2xhYmVscyUyMCUzRCUyMGxlbihtb2RlbC5jb25maWcuaWQybGFiZWwpJTBBbW9kZWwlMjAlM0QlMjBRRFFCZXJ0Rm9yU2VxdWVuY2VDbGFzc2lmaWNhdGlvbi5mcm9tX3ByZXRyYWluZWQoJTBBJTIwJTIwJTIwJTIwJTIyZ29vZ2xlLWJlcnQlMkZiZXJ0LWJhc2UtdW5jYXNlZCUyMiUyQyUyMG51bV9sYWJlbHMlM0RudW1fbGFiZWxzJTJDJTIwcHJvYmxlbV90eXBlJTNEJTIybXVsdGlfbGFiZWxfY2xhc3NpZmljYXRpb24lMjIlMEEpJTBBJTBBbGFiZWxzJTIwJTNEJTIwdG9yY2guc3VtKCUwQSUyMCUyMCUyMCUyMHRvcmNoLm5uLmZ1bmN0aW9uYWwub25lX2hvdChwcmVkaWN0ZWRfY2xhc3NfaWRzJTVCTm9uZSUyQyUyMCUzQSU1RC5jbG9uZSgpJTJDJTIwbnVtX2NsYXNzZXMlM0RudW1fbGFiZWxzKSUyQyUyMGRpbSUzRDElMEEpLnRvKHRvcmNoLmZsb2F0KSUwQWxvc3MlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyUyQyUyMGxhYmVscyUzRGxhYmVscykubG9zcw==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, QDQBertForSequenceClassification

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;google-bert/bert-base-uncased&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = QDQBertForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;google-bert/bert-base-uncased&quot;</span>, problem_type=<span class="hljs-string">&quot;multi_label_classification&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    logits = model(**inputs).logits

<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_class_ids = torch.arange(<span class="hljs-number">0</span>, logits.shape[-<span class="hljs-number">1</span>])[torch.sigmoid(logits).squeeze(dim=<span class="hljs-number">0</span>) &gt; <span class="hljs-number">0.5</span>]

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># To train a model on \`num_labels\` classes, you can pass \`num_labels=num_labels\` to \`.from_pretrained(...)\`</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>num_labels = <span class="hljs-built_in">len</span>(model.config.id2label)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = QDQBertForSequenceClassification.from_pretrained(
<span class="hljs-meta">... </span>    <span class="hljs-string">&quot;google-bert/bert-base-uncased&quot;</span>, num_labels=num_labels, problem_type=<span class="hljs-string">&quot;multi_label_classification&quot;</span>
<span class="hljs-meta">... </span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.<span class="hljs-built_in">sum</span>(
<span class="hljs-meta">... </span>    torch.nn.functional.one_hot(predicted_class_ids[<span class="hljs-literal">None</span>, :].clone(), num_classes=num_labels), dim=<span class="hljs-number">1</span>
<span class="hljs-meta">... </span>).to(torch.<span class="hljs-built_in">float</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = model(**inputs, labels=labels).loss`,wrap:!1}}),{c(){t=p("p"),t.textContent=y,i=a(),u(d.$$.fragment)},l(n){t=m(n,"P",{"data-svelte-h":!0}),h(t)!=="svelte-1l8e32d"&&(t.textContent=y),i=r(n),f(d.$$.fragment,n)},m(n,w){c(n,t,w),c(n,i,w),g(d,n,w),T=!0},p:B,i(n){T||(_(d.$$.fragment,n),T=!0)},o(n){b(d.$$.fragment,n),T=!1},d(n){n&&(s(t),s(i)),M(d,n)}}}function Ta(k){let t,y=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=p("p"),t.innerHTML=y},l(i){t=m(i,"P",{"data-svelte-h":!0}),h(t)!=="svelte-fincs2"&&(t.innerHTML=y)},m(i,d){c(i,t,d)},p:B,d(i){i&&s(t)}}}function wa(k){let t,y="Example:",i,d,T;return d=new Y({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBRRFFCZXJ0Rm9yTmV4dFNlbnRlbmNlUHJlZGljdGlvbiUwQWltcG9ydCUyMHRvcmNoJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIyZ29vZ2xlLWJlcnQlMkZiZXJ0LWJhc2UtdW5jYXNlZCUyMiklMEFtb2RlbCUyMCUzRCUyMFFEUUJlcnRGb3JOZXh0U2VudGVuY2VQcmVkaWN0aW9uLmZyb21fcHJldHJhaW5lZCglMjJnb29nbGUtYmVydCUyRmJlcnQtYmFzZS11bmNhc2VkJTIyKSUwQSUwQXByb21wdCUyMCUzRCUyMCUyMkluJTIwSXRhbHklMkMlMjBwaXp6YSUyMHNlcnZlZCUyMGluJTIwZm9ybWFsJTIwc2V0dGluZ3MlMkMlMjBzdWNoJTIwYXMlMjBhdCUyMGElMjByZXN0YXVyYW50JTJDJTIwaXMlMjBwcmVzZW50ZWQlMjB1bnNsaWNlZC4lMjIlMEFuZXh0X3NlbnRlbmNlJTIwJTNEJTIwJTIyVGhlJTIwc2t5JTIwaXMlMjBibHVlJTIwZHVlJTIwdG8lMjB0aGUlMjBzaG9ydGVyJTIwd2F2ZWxlbmd0aCUyMG9mJTIwYmx1ZSUyMGxpZ2h0LiUyMiUwQWVuY29kaW5nJTIwJTNEJTIwdG9rZW5pemVyKHByb21wdCUyQyUyMG5leHRfc2VudGVuY2UlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQSUwQW91dHB1dHMlMjAlM0QlMjBtb2RlbCgqKmVuY29kaW5nJTJDJTIwbGFiZWxzJTNEdG9yY2guTG9uZ1RlbnNvciglNUIxJTVEKSklMEFsb2dpdHMlMjAlM0QlMjBvdXRwdXRzLmxvZ2l0cyUwQWFzc2VydCUyMGxvZ2l0cyU1QjAlMkMlMjAwJTVEJTIwJTNDJTIwbG9naXRzJTVCMCUyQyUyMDElNUQlMjAlMjAlMjMlMjBuZXh0JTIwc2VudGVuY2UlMjB3YXMlMjByYW5kb20=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, QDQBertForNextSentencePrediction
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;google-bert/bert-base-uncased&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = QDQBertForNextSentencePrediction.from_pretrained(<span class="hljs-string">&quot;google-bert/bert-base-uncased&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>prompt = <span class="hljs-string">&quot;In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>next_sentence = <span class="hljs-string">&quot;The sky is blue due to the shorter wavelength of blue light.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>encoding = tokenizer(prompt, next_sentence, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**encoding, labels=torch.LongTensor([<span class="hljs-number">1</span>]))
<span class="hljs-meta">&gt;&gt;&gt; </span>logits = outputs.logits
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">assert</span> logits[<span class="hljs-number">0</span>, <span class="hljs-number">0</span>] &lt; logits[<span class="hljs-number">0</span>, <span class="hljs-number">1</span>]  <span class="hljs-comment"># next sentence was random</span>`,wrap:!1}}),{c(){t=p("p"),t.textContent=y,i=a(),u(d.$$.fragment)},l(n){t=m(n,"P",{"data-svelte-h":!0}),h(t)!=="svelte-11lpom8"&&(t.textContent=y),i=r(n),f(d.$$.fragment,n)},m(n,w){c(n,t,w),c(n,i,w),g(d,n,w),T=!0},p:B,i(n){T||(_(d.$$.fragment,n),T=!0)},o(n){b(d.$$.fragment,n),T=!1},d(n){n&&(s(t),s(i)),M(d,n)}}}function ka(k){let t,y=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=p("p"),t.innerHTML=y},l(i){t=m(i,"P",{"data-svelte-h":!0}),h(t)!=="svelte-fincs2"&&(t.innerHTML=y)},m(i,d){c(i,t,d)},p:B,d(i){i&&s(t)}}}function Qa(k){let t,y="Example:",i,d,T;return d=new Y({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBRRFFCZXJ0Rm9yTXVsdGlwbGVDaG9pY2UlMEFpbXBvcnQlMjB0b3JjaCUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMmdvb2dsZS1iZXJ0JTJGYmVydC1iYXNlLXVuY2FzZWQlMjIpJTBBbW9kZWwlMjAlM0QlMjBRRFFCZXJ0Rm9yTXVsdGlwbGVDaG9pY2UuZnJvbV9wcmV0cmFpbmVkKCUyMmdvb2dsZS1iZXJ0JTJGYmVydC1iYXNlLXVuY2FzZWQlMjIpJTBBJTBBcHJvbXB0JTIwJTNEJTIwJTIySW4lMjBJdGFseSUyQyUyMHBpenphJTIwc2VydmVkJTIwaW4lMjBmb3JtYWwlMjBzZXR0aW5ncyUyQyUyMHN1Y2glMjBhcyUyMGF0JTIwYSUyMHJlc3RhdXJhbnQlMkMlMjBpcyUyMHByZXNlbnRlZCUyMHVuc2xpY2VkLiUyMiUwQWNob2ljZTAlMjAlM0QlMjAlMjJJdCUyMGlzJTIwZWF0ZW4lMjB3aXRoJTIwYSUyMGZvcmslMjBhbmQlMjBhJTIwa25pZmUuJTIyJTBBY2hvaWNlMSUyMCUzRCUyMCUyMkl0JTIwaXMlMjBlYXRlbiUyMHdoaWxlJTIwaGVsZCUyMGluJTIwdGhlJTIwaGFuZC4lMjIlMEFsYWJlbHMlMjAlM0QlMjB0b3JjaC50ZW5zb3IoMCkudW5zcXVlZXplKDApJTIwJTIwJTIzJTIwY2hvaWNlMCUyMGlzJTIwY29ycmVjdCUyMChhY2NvcmRpbmclMjB0byUyMFdpa2lwZWRpYSUyMCUzQikpJTJDJTIwYmF0Y2glMjBzaXplJTIwMSUwQSUwQWVuY29kaW5nJTIwJTNEJTIwdG9rZW5pemVyKCU1QnByb21wdCUyQyUyMHByb21wdCU1RCUyQyUyMCU1QmNob2ljZTAlMkMlMjBjaG9pY2UxJTVEJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiUyQyUyMHBhZGRpbmclM0RUcnVlKSUwQW91dHB1dHMlMjAlM0QlMjBtb2RlbCgqKiU3QmslM0ElMjB2LnVuc3F1ZWV6ZSgwKSUyMGZvciUyMGslMkMlMjB2JTIwaW4lMjBlbmNvZGluZy5pdGVtcygpJTdEJTJDJTIwbGFiZWxzJTNEbGFiZWxzKSUyMCUyMCUyMyUyMGJhdGNoJTIwc2l6ZSUyMGlzJTIwMSUwQSUwQSUyMyUyMHRoZSUyMGxpbmVhciUyMGNsYXNzaWZpZXIlMjBzdGlsbCUyMG5lZWRzJTIwdG8lMjBiZSUyMHRyYWluZWQlMEFsb3NzJTIwJTNEJTIwb3V0cHV0cy5sb3NzJTBBbG9naXRzJTIwJTNEJTIwb3V0cHV0cy5sb2dpdHM=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, QDQBertForMultipleChoice
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;google-bert/bert-base-uncased&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = QDQBertForMultipleChoice.from_pretrained(<span class="hljs-string">&quot;google-bert/bert-base-uncased&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>prompt = <span class="hljs-string">&quot;In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>choice0 = <span class="hljs-string">&quot;It is eaten with a fork and a knife.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>choice1 = <span class="hljs-string">&quot;It is eaten while held in the hand.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.tensor(<span class="hljs-number">0</span>).unsqueeze(<span class="hljs-number">0</span>)  <span class="hljs-comment"># choice0 is correct (according to Wikipedia ;)), batch size 1</span>

<span class="hljs-meta">&gt;&gt;&gt; </span>encoding = tokenizer([prompt, prompt], [choice0, choice1], return_tensors=<span class="hljs-string">&quot;pt&quot;</span>, padding=<span class="hljs-literal">True</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**{k: v.unsqueeze(<span class="hljs-number">0</span>) <span class="hljs-keyword">for</span> k, v <span class="hljs-keyword">in</span> encoding.items()}, labels=labels)  <span class="hljs-comment"># batch size is 1</span>

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># the linear classifier still needs to be trained</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = outputs.loss
<span class="hljs-meta">&gt;&gt;&gt; </span>logits = outputs.logits`,wrap:!1}}),{c(){t=p("p"),t.textContent=y,i=a(),u(d.$$.fragment)},l(n){t=m(n,"P",{"data-svelte-h":!0}),h(t)!=="svelte-11lpom8"&&(t.textContent=y),i=r(n),f(d.$$.fragment,n)},m(n,w){c(n,t,w),c(n,i,w),g(d,n,w),T=!0},p:B,i(n){T||(_(d.$$.fragment,n),T=!0)},o(n){b(d.$$.fragment,n),T=!1},d(n){n&&(s(t),s(i)),M(d,n)}}}function va(k){let t,y=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=p("p"),t.innerHTML=y},l(i){t=m(i,"P",{"data-svelte-h":!0}),h(t)!=="svelte-fincs2"&&(t.innerHTML=y)},m(i,d){c(i,t,d)},p:B,d(i){i&&s(t)}}}function ja(k){let t,y="Example:",i,d,T;return d=new Y({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBRRFFCZXJ0Rm9yVG9rZW5DbGFzc2lmaWNhdGlvbiUwQWltcG9ydCUyMHRvcmNoJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIyZ29vZ2xlLWJlcnQlMkZiZXJ0LWJhc2UtdW5jYXNlZCUyMiklMEFtb2RlbCUyMCUzRCUyMFFEUUJlcnRGb3JUb2tlbkNsYXNzaWZpY2F0aW9uLmZyb21fcHJldHJhaW5lZCglMjJnb29nbGUtYmVydCUyRmJlcnQtYmFzZS11bmNhc2VkJTIyKSUwQSUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplciglMEElMjAlMjAlMjAlMjAlMjJIdWdnaW5nRmFjZSUyMGlzJTIwYSUyMGNvbXBhbnklMjBiYXNlZCUyMGluJTIwUGFyaXMlMjBhbmQlMjBOZXclMjBZb3JrJTIyJTJDJTIwYWRkX3NwZWNpYWxfdG9rZW5zJTNERmFsc2UlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyJTBBKSUwQSUwQXdpdGglMjB0b3JjaC5ub19ncmFkKCklM0ElMEElMjAlMjAlMjAlMjBsb2dpdHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cykubG9naXRzJTBBJTBBcHJlZGljdGVkX3Rva2VuX2NsYXNzX2lkcyUyMCUzRCUyMGxvZ2l0cy5hcmdtYXgoLTEpJTBBJTBBJTIzJTIwTm90ZSUyMHRoYXQlMjB0b2tlbnMlMjBhcmUlMjBjbGFzc2lmaWVkJTIwcmF0aGVyJTIwdGhlbiUyMGlucHV0JTIwd29yZHMlMjB3aGljaCUyMG1lYW5zJTIwdGhhdCUwQSUyMyUyMHRoZXJlJTIwbWlnaHQlMjBiZSUyMG1vcmUlMjBwcmVkaWN0ZWQlMjB0b2tlbiUyMGNsYXNzZXMlMjB0aGFuJTIwd29yZHMuJTBBJTIzJTIwTXVsdGlwbGUlMjB0b2tlbiUyMGNsYXNzZXMlMjBtaWdodCUyMGFjY291bnQlMjBmb3IlMjB0aGUlMjBzYW1lJTIwd29yZCUwQXByZWRpY3RlZF90b2tlbnNfY2xhc3NlcyUyMCUzRCUyMCU1Qm1vZGVsLmNvbmZpZy5pZDJsYWJlbCU1QnQuaXRlbSgpJTVEJTIwZm9yJTIwdCUyMGluJTIwcHJlZGljdGVkX3Rva2VuX2NsYXNzX2lkcyU1QjAlNUQlNUQlMEElMEFsYWJlbHMlMjAlM0QlMjBwcmVkaWN0ZWRfdG9rZW5fY2xhc3NfaWRzJTBBbG9zcyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzJTJDJTIwbGFiZWxzJTNEbGFiZWxzKS5sb3Nz",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, QDQBertForTokenClassification
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;google-bert/bert-base-uncased&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = QDQBertForTokenClassification.from_pretrained(<span class="hljs-string">&quot;google-bert/bert-base-uncased&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(
<span class="hljs-meta">... </span>    <span class="hljs-string">&quot;HuggingFace is a company based in Paris and New York&quot;</span>, add_special_tokens=<span class="hljs-literal">False</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>
<span class="hljs-meta">... </span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    logits = model(**inputs).logits

<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_token_class_ids = logits.argmax(-<span class="hljs-number">1</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Note that tokens are classified rather then input words which means that</span>
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># there might be more predicted token classes than words.</span>
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Multiple token classes might account for the same word</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_tokens_classes = [model.config.id2label[t.item()] <span class="hljs-keyword">for</span> t <span class="hljs-keyword">in</span> predicted_token_class_ids[<span class="hljs-number">0</span>]]

<span class="hljs-meta">&gt;&gt;&gt; </span>labels = predicted_token_class_ids
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = model(**inputs, labels=labels).loss`,wrap:!1}}),{c(){t=p("p"),t.textContent=y,i=a(),u(d.$$.fragment)},l(n){t=m(n,"P",{"data-svelte-h":!0}),h(t)!=="svelte-11lpom8"&&(t.textContent=y),i=r(n),f(d.$$.fragment,n)},m(n,w){c(n,t,w),c(n,i,w),g(d,n,w),T=!0},p:B,i(n){T||(_(d.$$.fragment,n),T=!0)},o(n){b(d.$$.fragment,n),T=!1},d(n){n&&(s(t),s(i)),M(d,n)}}}function Ba(k){let t,y=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=p("p"),t.innerHTML=y},l(i){t=m(i,"P",{"data-svelte-h":!0}),h(t)!=="svelte-fincs2"&&(t.innerHTML=y)},m(i,d){c(i,t,d)},p:B,d(i){i&&s(t)}}}function $a(k){let t,y="Example:",i,d,T;return d=new Y({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBRRFFCZXJ0Rm9yUXVlc3Rpb25BbnN3ZXJpbmclMEFpbXBvcnQlMjB0b3JjaCUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMmdvb2dsZS1iZXJ0JTJGYmVydC1iYXNlLXVuY2FzZWQlMjIpJTBBbW9kZWwlMjAlM0QlMjBRRFFCZXJ0Rm9yUXVlc3Rpb25BbnN3ZXJpbmcuZnJvbV9wcmV0cmFpbmVkKCUyMmdvb2dsZS1iZXJ0JTJGYmVydC1iYXNlLXVuY2FzZWQlMjIpJTBBJTBBcXVlc3Rpb24lMkMlMjB0ZXh0JTIwJTNEJTIwJTIyV2hvJTIwd2FzJTIwSmltJTIwSGVuc29uJTNGJTIyJTJDJTIwJTIySmltJTIwSGVuc29uJTIwd2FzJTIwYSUyMG5pY2UlMjBwdXBwZXQlMjIlMEElMEFpbnB1dHMlMjAlM0QlMjB0b2tlbml6ZXIocXVlc3Rpb24lMkMlMjB0ZXh0JTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiklMEF3aXRoJTIwdG9yY2gubm9fZ3JhZCgpJTNBJTBBJTIwJTIwJTIwJTIwb3V0cHV0cyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzKSUwQSUwQWFuc3dlcl9zdGFydF9pbmRleCUyMCUzRCUyMG91dHB1dHMuc3RhcnRfbG9naXRzLmFyZ21heCgpJTBBYW5zd2VyX2VuZF9pbmRleCUyMCUzRCUyMG91dHB1dHMuZW5kX2xvZ2l0cy5hcmdtYXgoKSUwQSUwQXByZWRpY3RfYW5zd2VyX3Rva2VucyUyMCUzRCUyMGlucHV0cy5pbnB1dF9pZHMlNUIwJTJDJTIwYW5zd2VyX3N0YXJ0X2luZGV4JTIwJTNBJTIwYW5zd2VyX2VuZF9pbmRleCUyMCUyQiUyMDElNUQlMEElMEElMjMlMjB0YXJnZXQlMjBpcyUyMCUyMm5pY2UlMjBwdXBwZXQlMjIlMEF0YXJnZXRfc3RhcnRfaW5kZXglMjAlM0QlMjB0b3JjaC50ZW5zb3IoJTVCMTQlNUQpJTBBdGFyZ2V0X2VuZF9pbmRleCUyMCUzRCUyMHRvcmNoLnRlbnNvciglNUIxNSU1RCklMEElMEFvdXRwdXRzJTIwJTNEJTIwbW9kZWwoKippbnB1dHMlMkMlMjBzdGFydF9wb3NpdGlvbnMlM0R0YXJnZXRfc3RhcnRfaW5kZXglMkMlMjBlbmRfcG9zaXRpb25zJTNEdGFyZ2V0X2VuZF9pbmRleCklMEFsb3NzJTIwJTNEJTIwb3V0cHV0cy5sb3Nz",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, QDQBertForQuestionAnswering
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;google-bert/bert-base-uncased&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = QDQBertForQuestionAnswering.from_pretrained(<span class="hljs-string">&quot;google-bert/bert-base-uncased&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>question, text = <span class="hljs-string">&quot;Who was Jim Henson?&quot;</span>, <span class="hljs-string">&quot;Jim Henson was a nice puppet&quot;</span>

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(question, text, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    outputs = model(**inputs)

<span class="hljs-meta">&gt;&gt;&gt; </span>answer_start_index = outputs.start_logits.argmax()
<span class="hljs-meta">&gt;&gt;&gt; </span>answer_end_index = outputs.end_logits.argmax()

<span class="hljs-meta">&gt;&gt;&gt; </span>predict_answer_tokens = inputs.input_ids[<span class="hljs-number">0</span>, answer_start_index : answer_end_index + <span class="hljs-number">1</span>]

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># target is &quot;nice puppet&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>target_start_index = torch.tensor([<span class="hljs-number">14</span>])
<span class="hljs-meta">&gt;&gt;&gt; </span>target_end_index = torch.tensor([<span class="hljs-number">15</span>])

<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs, start_positions=target_start_index, end_positions=target_end_index)
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = outputs.loss`,wrap:!1}}),{c(){t=p("p"),t.textContent=y,i=a(),u(d.$$.fragment)},l(n){t=m(n,"P",{"data-svelte-h":!0}),h(t)!=="svelte-11lpom8"&&(t.textContent=y),i=r(n),f(d.$$.fragment,n)},m(n,w){c(n,t,w),c(n,i,w),g(d,n,w),T=!0},p:B,i(n){T||(_(d.$$.fragment,n),T=!0)},o(n){b(d.$$.fragment,n),T=!1},d(n){n&&(s(t),s(i)),M(d,n)}}}function Ja(k){let t,y,i,d,T,n="<em>This model was released on 2020-04-20 and added to Hugging Face Transformers on 2023-06-20.</em>",w,Fe,cn,me,ls='<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/>',pn,he,mn,ze,hn,xe,ds=`The QDQBERT model can be referenced in <a href="https://huggingface.co/papers/2004.09602" rel="nofollow">Integer Quantization for Deep Learning Inference: Principles and Empirical
Evaluation</a> by Hao Wu, Patrick Judd, Xiaojie Zhang, Mikhail Isaev and Paulius
Micikevicius.`,un,qe,cs="The abstract from the paper is the following:",fn,Ze,ps=`<em>Quantization techniques can reduce the size of Deep Neural Networks and improve inference latency and throughput by
taking advantage of high throughput integer instructions. In this paper we review the mathematical aspects of
quantization parameters and evaluate their choices on a wide range of neural network models for different application
domains, including vision, speech, and language. We focus on quantization techniques that are amenable to acceleration
by processors with high-throughput integer math pipelines. We also present a workflow for 8-bit quantization that is
able to maintain accuracy within 1% of the floating-point baseline on all networks studied, including models that are
more difficult to quantize, such as MobileNets and BERT-large.</em>`,gn,We,ms='This model was contributed by <a href="https://huggingface.co/shangz" rel="nofollow">shangz</a>.',_n,Ie,bn,De,hs=`<li>QDQBERT model adds fake quantization operations (pair of QuantizeLinear/DequantizeLinear ops) to (i) linear layer
inputs and weights, (ii) matmul inputs, (iii) residual add inputs, in BERT model.</li> <li>QDQBERT requires the dependency of <a href="https://github.com/NVIDIA/TensorRT/tree/master/tools/pytorch-quantization" rel="nofollow">Pytorch Quantization Toolkit</a>. To install <code>pip install pytorch-quantization --extra-index-url https://pypi.ngc.nvidia.com</code></li> <li>QDQBERT model can be loaded from any checkpoint of HuggingFace BERT model (for example <em>google-bert/bert-base-uncased</em>), and
perform Quantization Aware Training/Post Training Quantization.</li> <li>A complete example of using QDQBERT model to perform Quatization Aware Training and Post Training Quantization for
SQUAD task can be found at <a href="https://github.com/huggingface/transformers-research-projects/tree/main/quantization-qdqbert" rel="nofollow">https://github.com/huggingface/transformers-research-projects/tree/main/quantization-qdqbert</a>.</li>`,Mn,Re,yn,Xe,us=`QDQBERT model adds fake quantization operations (pair of QuantizeLinear/DequantizeLinear ops) to BERT by
<code>TensorQuantizer</code> in <a href="https://github.com/NVIDIA/TensorRT/tree/master/tools/pytorch-quantization" rel="nofollow">Pytorch Quantization Toolkit</a>. <code>TensorQuantizer</code> is the module
for quantizing tensors, with <code>QuantDescriptor</code> defining how the tensor should be quantized. Refer to <a href="https://docs.nvidia.com/deeplearning/tensorrt/pytorch-quantization-toolkit/docs/userguide.html" rel="nofollow">Pytorch
Quantization Toolkit userguide</a> for more details.`,Tn,Ne,fs="Before creating QDQBERT model, one has to set the default <code>QuantDescriptor</code> defining default tensor quantizers.",wn,Ge,gs="Example:",kn,Le,Qn,Ve,vn,He,_s=`Calibration is the terminology of passing data samples to the quantizer and deciding the best scaling factors for
tensors. After setting up the tensor quantizers, one can use the following example to calibrate the model:`,jn,Se,Bn,Ye,$n,Ee,bs=`The goal of exporting to ONNX is to deploy inference by <a href="https://developer.nvidia.com/tensorrt" rel="nofollow">TensorRT</a>. Fake
quantization will be broken into a pair of QuantizeLinear/DequantizeLinear ONNX ops. After setting static member of
TensorQuantizer to use Pytorch’s own fake quantization functions, fake quantized model can be exported to ONNX, follow
the instructions in <a href="https://pytorch.org/docs/stable/onnx.html" rel="nofollow">torch.onnx</a>. Example:`,Jn,Pe,Cn,Ae,Un,Oe,Ms='<li><a href="../tasks/sequence_classification">Text classification task guide</a></li> <li><a href="../tasks/token_classification">Token classification task guide</a></li> <li><a href="../tasks/question_answering">Question answering task guide</a></li> <li><a href="../tasks/language_modeling">Causal language modeling task guide</a></li> <li><a href="../tasks/masked_language_modeling">Masked language modeling task guide</a></li> <li><a href="../tasks/multiple_choice">Multiple choice task guide</a></li>',Fn,Ke,zn,D,et,Kn,jt,ys=`This is the configuration class to store the configuration of a <a href="/docs/transformers/v4.56.2/en/model_doc/qdqbert#transformers.QDQBertModel">QDQBertModel</a>. It is used to instantiate an
QDQBERT model according to the specified arguments, defining the model architecture. Instantiating a configuration
with the defaults will yield a similar configuration to that of the BERT
<a href="https://huggingface.co/google-bert/bert-base-uncased" rel="nofollow">google-bert/bert-base-uncased</a> architecture.`,eo,Bt,Ts=`Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> and can be used to control the model outputs. Read the
documentation from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> for more information.`,to,ue,xn,tt,qn,v,nt,no,$t,ws="The bare QDQBERT Model transformer outputting raw hidden-states without any specific head on top.",oo,Jt,ks=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,so,Ct,Qs=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,ao,Ut,vs=`The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
cross-attention is added between the self-attention layers, following the architecture described in <a href="https://huggingface.co/papers/1706.03762" rel="nofollow">Attention is
all you need</a> by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.`,ro,Ft,js=`To behave as an decoder the model needs to be initialized with the <code>is_decoder</code> argument of the configuration set
to <code>True</code>. To be used in a Seq2Seq model, the model needs to initialized with both <code>is_decoder</code> argument and
<code>add_cross_attention</code> set to <code>True</code>; an <code>encoder_hidden_states</code> is then expected as an input to the forward pass.`,io,E,ot,lo,zt,Bs='The <a href="/docs/transformers/v4.56.2/en/model_doc/qdqbert#transformers.QDQBertModel">QDQBertModel</a> forward method, overrides the <code>__call__</code> special method.',co,fe,po,ge,Zn,st,Wn,U,at,mo,xt,$s="QDQBERT Model with a <code>language modeling</code> head on top for CLM fine-tuning.",ho,qt,Js=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,uo,Zt,Cs=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,fo,P,rt,go,Wt,Us='The <a href="/docs/transformers/v4.56.2/en/model_doc/qdqbert#transformers.QDQBertLMHeadModel">QDQBertLMHeadModel</a> forward method, overrides the <code>__call__</code> special method.',_o,_e,bo,be,In,it,Dn,F,lt,Mo,It,Fs="QDQBERT Model with a <code>language modeling</code> head on top.",yo,Dt,zs=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,To,Rt,xs=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,wo,A,dt,ko,Xt,qs='The <a href="/docs/transformers/v4.56.2/en/model_doc/qdqbert#transformers.QDQBertForMaskedLM">QDQBertForMaskedLM</a> forward method, overrides the <code>__call__</code> special method.',Qo,Me,vo,ye,Rn,ct,Xn,z,pt,jo,Nt,Zs=`Bert Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled
output) e.g. for GLUE tasks.`,Bo,Gt,Ws=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,$o,Lt,Is=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Jo,I,mt,Co,Vt,Ds='The <a href="/docs/transformers/v4.56.2/en/model_doc/qdqbert#transformers.QDQBertForSequenceClassification">QDQBertForSequenceClassification</a> forward method, overrides the <code>__call__</code> special method.',Uo,Te,Fo,we,zo,ke,Nn,ht,Gn,x,ut,xo,Ht,Rs="Bert Model with a <code>next sentence prediction (classification)</code> head on top.",qo,St,Xs=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Zo,Yt,Ns=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Wo,O,ft,Io,Et,Gs='The <a href="/docs/transformers/v4.56.2/en/model_doc/qdqbert#transformers.QDQBertForNextSentencePrediction">QDQBertForNextSentencePrediction</a> forward method, overrides the <code>__call__</code> special method.',Do,Qe,Ro,ve,Ln,gt,Vn,q,_t,Xo,Pt,Ls=`Bert Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
softmax) e.g. for RocStories/SWAG tasks.`,No,At,Vs=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Go,Ot,Hs=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Lo,K,bt,Vo,Kt,Ss='The <a href="/docs/transformers/v4.56.2/en/model_doc/qdqbert#transformers.QDQBertForMultipleChoice">QDQBertForMultipleChoice</a> forward method, overrides the <code>__call__</code> special method.',Ho,je,So,Be,Hn,Mt,Sn,Z,yt,Yo,en,Ys=`QDQBERT Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
Named-Entity-Recognition (NER) tasks.`,Eo,tn,Es=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Po,nn,Ps=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Ao,ee,Tt,Oo,on,As='The <a href="/docs/transformers/v4.56.2/en/model_doc/qdqbert#transformers.QDQBertForTokenClassification">QDQBertForTokenClassification</a> forward method, overrides the <code>__call__</code> special method.',Ko,$e,es,Je,Yn,wt,En,W,kt,ts,sn,Os=`QDQBERT Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
layers on top of the hidden-states output to compute <code>span start logits</code> and <code>span end logits</code>).`,ns,an,Ks=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,os,rn,ea=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,ss,te,Qt,as,ln,ta='The <a href="/docs/transformers/v4.56.2/en/model_doc/qdqbert#transformers.QDQBertForQuestionAnswering">QDQBertForQuestionAnswering</a> forward method, overrides the <code>__call__</code> special method.',rs,Ce,is,Ue,Pn,vt,An,dn,On;return Fe=new C({props:{title:"QDQBERT",local:"qdqbert",headingTag:"h1"}}),he=new pe({props:{warning:!0,$$slots:{default:[ca]},$$scope:{ctx:k}}}),ze=new C({props:{title:"Overview",local:"overview",headingTag:"h2"}}),Ie=new C({props:{title:"Usage tips",local:"usage-tips",headingTag:"h2"}}),Re=new C({props:{title:"Set default quantizers",local:"set-default-quantizers",headingTag:"h3"}}),Le=new Y({props:{code:"aW1wb3J0JTIwcHl0b3JjaF9xdWFudGl6YXRpb24ubm4lMjBhcyUyMHF1YW50X25uJTBBZnJvbSUyMHB5dG9yY2hfcXVhbnRpemF0aW9uLnRlbnNvcl9xdWFudCUyMGltcG9ydCUyMFF1YW50RGVzY3JpcHRvciUwQSUwQSUyMyUyMFRoZSUyMGRlZmF1bHQlMjB0ZW5zb3IlMjBxdWFudGl6ZXIlMjBpcyUyMHNldCUyMHRvJTIwdXNlJTIwTWF4JTIwY2FsaWJyYXRpb24lMjBtZXRob2QlMEFpbnB1dF9kZXNjJTIwJTNEJTIwUXVhbnREZXNjcmlwdG9yKG51bV9iaXRzJTNEOCUyQyUyMGNhbGliX21ldGhvZCUzRCUyMm1heCUyMiklMEElMjMlMjBUaGUlMjBkZWZhdWx0JTIwdGVuc29yJTIwcXVhbnRpemVyJTIwaXMlMjBzZXQlMjB0byUyMGJlJTIwcGVyLWNoYW5uZWwlMjBxdWFudGl6YXRpb24lMjBmb3IlMjB3ZWlnaHRzJTBBd2VpZ2h0X2Rlc2MlMjAlM0QlMjBRdWFudERlc2NyaXB0b3IobnVtX2JpdHMlM0Q4JTJDJTIwYXhpcyUzRCgoMCUyQykpKSUwQXF1YW50X25uLlF1YW50TGluZWFyLnNldF9kZWZhdWx0X3F1YW50X2Rlc2NfaW5wdXQoaW5wdXRfZGVzYyklMEFxdWFudF9ubi5RdWFudExpbmVhci5zZXRfZGVmYXVsdF9xdWFudF9kZXNjX3dlaWdodCh3ZWlnaHRfZGVzYyk=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> pytorch_quantization.nn <span class="hljs-keyword">as</span> quant_nn
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> pytorch_quantization.tensor_quant <span class="hljs-keyword">import</span> QuantDescriptor

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># The default tensor quantizer is set to use Max calibration method</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>input_desc = QuantDescriptor(num_bits=<span class="hljs-number">8</span>, calib_method=<span class="hljs-string">&quot;max&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># The default tensor quantizer is set to be per-channel quantization for weights</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>weight_desc = QuantDescriptor(num_bits=<span class="hljs-number">8</span>, axis=((<span class="hljs-number">0</span>,)))
<span class="hljs-meta">&gt;&gt;&gt; </span>quant_nn.QuantLinear.set_default_quant_desc_input(input_desc)
<span class="hljs-meta">&gt;&gt;&gt; </span>quant_nn.QuantLinear.set_default_quant_desc_weight(weight_desc)`,wrap:!1}}),Ve=new C({props:{title:"Calibration",local:"calibration",headingTag:"h3"}}),Se=new Y({props:{code:"JTIzJTIwRmluZCUyMHRoZSUyMFRlbnNvclF1YW50aXplciUyMGFuZCUyMGVuYWJsZSUyMGNhbGlicmF0aW9uJTBBZm9yJTIwbmFtZSUyQyUyMG1vZHVsZSUyMGluJTIwbW9kZWwubmFtZWRfbW9kdWxlcygpJTNBJTBBJTIwJTIwJTIwJTIwaWYlMjBuYW1lLmVuZHN3aXRoKCUyMl9pbnB1dF9xdWFudGl6ZXIlMjIpJTNBJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwbW9kdWxlLmVuYWJsZV9jYWxpYigpJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwbW9kdWxlLmRpc2FibGVfcXVhbnQoKSUyMCUyMCUyMyUyMFVzZSUyMGZ1bGwlMjBwcmVjaXNpb24lMjBkYXRhJTIwdG8lMjBjYWxpYnJhdGUlMEElMEElMjMlMjBGZWVkaW5nJTIwZGF0YSUyMHNhbXBsZXMlMEFtb2RlbCh4KSUwQSUyMyUyMC4uLiUwQSUwQSUyMyUyMEZpbmFsaXplJTIwY2FsaWJyYXRpb24lMEFmb3IlMjBuYW1lJTJDJTIwbW9kdWxlJTIwaW4lMjBtb2RlbC5uYW1lZF9tb2R1bGVzKCklM0ElMEElMjAlMjAlMjAlMjBpZiUyMG5hbWUuZW5kc3dpdGgoJTIyX2lucHV0X3F1YW50aXplciUyMiklM0ElMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjBtb2R1bGUubG9hZF9jYWxpYl9hbWF4KCklMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjBtb2R1bGUuZW5hYmxlX3F1YW50KCklMEElMEElMjMlMjBJZiUyMHJ1bm5pbmclMjBvbiUyMGFjY2VsZXJhdG9yJTJDJTIwaXQlMjBuZWVkcyUyMHRvJTIwY2FsbCUyMCU2MC50byh4eCklNjAlMjBhZ2FpbiUyMGJlY2F1c2UlMjBuZXclMjB0ZW5zb3JzJTIwd2lsbCUyMGJlJTIwY3JlYXRlZCUyMGJ5JTIwY2FsaWJyYXRpb24lMjBwcm9jZXNzJTBBZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMGluZmVyX2RldmljZSUwQWRldmljZSUyMCUzRCUyMGluZmVyX2RldmljZSgpJTBBbW9kZWwudG8oZGV2aWNlKSUwQSUwQSUyMyUyMEtlZXAlMjBydW5uaW5nJTIwdGhlJTIwcXVhbnRpemVkJTIwbW9kZWwlMEElMjMlMjAuLi4=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Find the TensorQuantizer and enable calibration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">for</span> name, module <span class="hljs-keyword">in</span> model.named_modules():
<span class="hljs-meta">... </span>    <span class="hljs-keyword">if</span> name.endswith(<span class="hljs-string">&quot;_input_quantizer&quot;</span>):
<span class="hljs-meta">... </span>        module.enable_calib()
<span class="hljs-meta">... </span>        module.disable_quant()  <span class="hljs-comment"># Use full precision data to calibrate</span>

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Feeding data samples</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model(x)
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># ...</span>

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Finalize calibration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">for</span> name, module <span class="hljs-keyword">in</span> model.named_modules():
<span class="hljs-meta">... </span>    <span class="hljs-keyword">if</span> name.endswith(<span class="hljs-string">&quot;_input_quantizer&quot;</span>):
<span class="hljs-meta">... </span>        module.load_calib_amax()
<span class="hljs-meta">... </span>        module.enable_quant()

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># If running on accelerator, it needs to call \`.to(xx)\` again because new tensors will be created by calibration process</span>
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> infer_device
<span class="hljs-meta">&gt;&gt;&gt; </span>device = infer_device()
<span class="hljs-meta">&gt;&gt;&gt; </span>model.to(device)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Keep running the quantized model</span>
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># ...</span>`,wrap:!1}}),Ye=new C({props:{title:"Export to ONNX",local:"export-to-onnx",headingTag:"h3"}}),Pe=new Y({props:{code:"ZnJvbSUyMHB5dG9yY2hfcXVhbnRpemF0aW9uLm5uJTIwaW1wb3J0JTIwVGVuc29yUXVhbnRpemVyJTBBJTBBVGVuc29yUXVhbnRpemVyLnVzZV9mYl9mYWtlX3F1YW50JTIwJTNEJTIwVHJ1ZSUwQSUwQSUyMyUyMExvYWQlMjB0aGUlMjBjYWxpYnJhdGVkJTIwbW9kZWwlMEEuLi4lMEElMjMlMjBPTk5YJTIwZXhwb3J0JTBBdG9yY2gub25ueC5leHBvcnQoLi4uKQ==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> pytorch_quantization.nn <span class="hljs-keyword">import</span> TensorQuantizer

<span class="hljs-meta">&gt;&gt;&gt; </span>TensorQuantizer.use_fb_fake_quant = <span class="hljs-literal">True</span>

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Load the calibrated model</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>...
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># ONNX export</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>torch.onnx.export(...)`,wrap:!1}}),Ae=new C({props:{title:"Resources",local:"resources",headingTag:"h2"}}),Ke=new C({props:{title:"QDQBertConfig",local:"transformers.QDQBertConfig",headingTag:"h2"}}),et=new J({props:{name:"class transformers.QDQBertConfig",anchor:"transformers.QDQBertConfig",parameters:[{name:"vocab_size",val:" = 30522"},{name:"hidden_size",val:" = 768"},{name:"num_hidden_layers",val:" = 12"},{name:"num_attention_heads",val:" = 12"},{name:"intermediate_size",val:" = 3072"},{name:"hidden_act",val:" = 'gelu'"},{name:"hidden_dropout_prob",val:" = 0.1"},{name:"attention_probs_dropout_prob",val:" = 0.1"},{name:"max_position_embeddings",val:" = 512"},{name:"type_vocab_size",val:" = 2"},{name:"initializer_range",val:" = 0.02"},{name:"layer_norm_eps",val:" = 1e-12"},{name:"use_cache",val:" = True"},{name:"pad_token_id",val:" = 1"},{name:"bos_token_id",val:" = 0"},{name:"eos_token_id",val:" = 2"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.QDQBertConfig.vocab_size",description:`<strong>vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 30522) &#x2014;
Vocabulary size of the QDQBERT model. Defines the number of different tokens that can be represented by the
<code>inputs_ids</code> passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/qdqbert#transformers.QDQBertModel">QDQBertModel</a>.`,name:"vocab_size"},{anchor:"transformers.QDQBertConfig.hidden_size",description:`<strong>hidden_size</strong> (<code>int</code>, <em>optional</em>, defaults to 768) &#x2014;
Dimension of the encoder layers and the pooler layer.`,name:"hidden_size"},{anchor:"transformers.QDQBertConfig.num_hidden_layers",description:`<strong>num_hidden_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 12) &#x2014;
Number of hidden layers in the Transformer encoder.`,name:"num_hidden_layers"},{anchor:"transformers.QDQBertConfig.num_attention_heads",description:`<strong>num_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 12) &#x2014;
Number of attention heads for each attention layer in the Transformer encoder.`,name:"num_attention_heads"},{anchor:"transformers.QDQBertConfig.intermediate_size",description:`<strong>intermediate_size</strong> (<code>int</code>, <em>optional</em>, defaults to 3072) &#x2014;
Dimension of the &#x201C;intermediate&#x201D; (i.e., feed-forward) layer in the Transformer encoder.`,name:"intermediate_size"},{anchor:"transformers.QDQBertConfig.hidden_act",description:`<strong>hidden_act</strong> (<code>str</code> or <code>function</code>, <em>optional</em>, defaults to <code>&quot;gelu&quot;</code>) &#x2014;
The non-linear activation function (function or string) in the encoder and pooler. If string, <code>&quot;gelu&quot;</code>,
<code>&quot;relu&quot;</code>, <code>&quot;selu&quot;</code> and <code>&quot;gelu_new&quot;</code> are supported.`,name:"hidden_act"},{anchor:"transformers.QDQBertConfig.hidden_dropout_prob",description:`<strong>hidden_dropout_prob</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.`,name:"hidden_dropout_prob"},{anchor:"transformers.QDQBertConfig.attention_probs_dropout_prob",description:`<strong>attention_probs_dropout_prob</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout ratio for the attention probabilities.`,name:"attention_probs_dropout_prob"},{anchor:"transformers.QDQBertConfig.max_position_embeddings",description:`<strong>max_position_embeddings</strong> (<code>int</code>, <em>optional</em>, defaults to 512) &#x2014;
The maximum sequence length that this model might ever be used with. Typically set this to something large
just in case (e.g., 512 or 1024 or 2048).`,name:"max_position_embeddings"},{anchor:"transformers.QDQBertConfig.type_vocab_size",description:`<strong>type_vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 2) &#x2014;
The vocabulary size of the <code>token_type_ids</code> passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/qdqbert#transformers.QDQBertModel">QDQBertModel</a>.`,name:"type_vocab_size"},{anchor:"transformers.QDQBertConfig.initializer_range",description:`<strong>initializer_range</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.`,name:"initializer_range"},{anchor:"transformers.QDQBertConfig.layer_norm_eps",description:`<strong>layer_norm_eps</strong> (<code>float</code>, <em>optional</em>, defaults to 1e-12) &#x2014;
The epsilon used by the layer normalization layers.`,name:"layer_norm_eps"},{anchor:"transformers.QDQBertConfig.is_decoder",description:`<strong>is_decoder</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether the model is used as a decoder or not. If <code>False</code>, the model is used as an encoder.`,name:"is_decoder"},{anchor:"transformers.QDQBertConfig.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not the model should return the last key/values attentions (not used by all models). Only
relevant if <code>config.is_decoder=True</code>.`,name:"use_cache"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/qdqbert/configuration_qdqbert.py#L24"}}),ue=new ce({props:{anchor:"transformers.QDQBertConfig.example",$$slots:{default:[pa]},$$scope:{ctx:k}}}),tt=new C({props:{title:"QDQBertModel",local:"transformers.QDQBertModel",headingTag:"h2"}}),nt=new J({props:{name:"class transformers.QDQBertModel",anchor:"transformers.QDQBertModel",parameters:[{name:"config",val:""},{name:"add_pooling_layer",val:": bool = True"}],parametersDescription:[{anchor:"transformers.QDQBertModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/qdqbert#transformers.QDQBertConfig">QDQBertConfig</a>) &#x2014; Model configuration class with all the parameters of the model.
Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/qdqbert/modeling_qdqbert.py#L806"}}),ot=new J({props:{name:"forward",anchor:"transformers.QDQBertModel.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"encoder_hidden_states",val:": typing.Optional[torch.FloatTensor] = None"},{name:"encoder_attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"past_key_values",val:": typing.Optional[tuple[tuple[torch.FloatTensor]]] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.QDQBertModel.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.QDQBertModel.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.QDQBertModel.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.QDQBertModel.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.max_position_embeddings - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.QDQBertModel.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.QDQBertModel.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.QDQBertModel.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.QDQBertModel.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.QDQBertModel.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.QDQBertModel.forward.encoder_hidden_states",description:`<strong>encoder_hidden_states</strong>  (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
the model is configured as a decoder.`,name:"encoder_hidden_states"},{anchor:"transformers.QDQBertModel.forward.encoder_attention_mask",description:`<strong>encoder_attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
the cross-attention if the model is configured as a decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>`,name:"encoder_attention_mask"},{anchor:"transformers.QDQBertModel.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>tuple(tuple(torch.FloatTensor))</code> of length <code>config.n_layers</code> with each tuple having 4 tensors of shape <code>(batch_size, num_heads, sequence_length - 1, embed_size_per_head)</code>) &#x2014;
Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.</p>
<p>If <code>past_key_values</code> are used, the user can optionally input only the last <code>decoder_input_ids</code> (those that
don&#x2019;t have their past key value states given to this model) of shape <code>(batch_size, 1)</code> instead of all
<code>decoder_input_ids</code> of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.QDQBertModel.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/qdqbert/modeling_qdqbert.py#L846",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions"
>transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/qdqbert#transformers.QDQBertConfig"
>QDQBertConfig</a>) and inputs.</p>
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
<li>
<p><strong>cross_attentions</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> and <code>config.add_cross_attention=True</code> is passed or when <code>config.output_attentions=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the
weighted average in the cross-attention heads.</p>
</li>
<li>
<p><strong>past_key_values</strong> (<code>Cache</code>, <em>optional</em>, returned when <code>use_cache=True</code> is passed or when <code>config.use_cache=True</code>) — It is a <a
  href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache"
>Cache</a> instance. For more details, see our <a
  href="https://huggingface.co/docs/transformers/en/kv_cache"
  rel="nofollow"
>kv cache guide</a>.</p>
<p>Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
<code>config.is_encoder_decoder=True</code> in the cross-attention blocks) that can be used (see <code>past_key_values</code>
input) to speed up sequential decoding.</p>
</li>
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions"
>transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),fe=new pe({props:{$$slots:{default:[ma]},$$scope:{ctx:k}}}),ge=new ce({props:{anchor:"transformers.QDQBertModel.forward.example",$$slots:{default:[ha]},$$scope:{ctx:k}}}),st=new C({props:{title:"QDQBertLMHeadModel",local:"transformers.QDQBertLMHeadModel",headingTag:"h2"}}),at=new J({props:{name:"class transformers.QDQBertLMHeadModel",anchor:"transformers.QDQBertLMHeadModel",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.QDQBertLMHeadModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/qdqbert#transformers.QDQBertConfig">QDQBertConfig</a>) &#x2014; Model configuration class with all the parameters of the model.
Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/qdqbert/modeling_qdqbert.py#L987"}}),rt=new J({props:{name:"forward",anchor:"transformers.QDQBertLMHeadModel.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_hidden_states",val:": typing.Optional[torch.FloatTensor] = None"},{name:"encoder_attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[tuple[tuple[torch.LongTensor]]] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.QDQBertLMHeadModel.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.QDQBertLMHeadModel.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.QDQBertLMHeadModel.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.QDQBertLMHeadModel.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.max_position_embeddings - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.QDQBertLMHeadModel.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.QDQBertLMHeadModel.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.QDQBertLMHeadModel.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.QDQBertLMHeadModel.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.QDQBertLMHeadModel.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.QDQBertLMHeadModel.forward.encoder_hidden_states",description:`<strong>encoder_hidden_states</strong>  (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
the model is configured as a decoder.`,name:"encoder_hidden_states"},{anchor:"transformers.QDQBertLMHeadModel.forward.encoder_attention_mask",description:`<strong>encoder_attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
the cross-attention if the model is configured as a decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>`,name:"encoder_attention_mask"},{anchor:"transformers.QDQBertLMHeadModel.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the left-to-right language modeling loss (next word prediction). Indices should be in
<code>[-100, 0, ..., config.vocab_size]</code> (see <code>input_ids</code> docstring) Tokens with indices set to <code>-100</code> are
ignored (masked), the loss is only computed for the tokens with labels n <code>[0, ..., config.vocab_size]</code>`,name:"labels"},{anchor:"transformers.QDQBertLMHeadModel.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>tuple(tuple(torch.FloatTensor))</code> of length <code>config.n_layers</code> with each tuple having 4 tensors of shape <code>(batch_size, num_heads, sequence_length - 1, embed_size_per_head)</code>) &#x2014;
Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.</p>
<p>If <code>past_key_values</code> are used, the user can optionally input only the last <code>decoder_input_ids</code> (those that
don&#x2019;t have their past key value states given to this model) of shape <code>(batch_size, 1)</code> instead of all
<code>decoder_input_ids</code> of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.QDQBertLMHeadModel.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/qdqbert/modeling_qdqbert.py#L1009",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithCrossAttentions"
>transformers.modeling_outputs.CausalLMOutputWithCrossAttentions</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/qdqbert#transformers.QDQBertConfig"
>QDQBertConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>loss</strong> (<code>torch.FloatTensor</code> of shape <code>(1,)</code>, <em>optional</em>, returned when <code>labels</code> is provided) — Language modeling loss (for next-token prediction).</p>
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
<li>
<p><strong>cross_attentions</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Cross attentions weights after the attention softmax, used to compute the weighted average in the
cross-attention heads.</p>
</li>
<li>
<p><strong>past_key_values</strong> (<code>Cache</code>, <em>optional</em>, returned when <code>use_cache=True</code> is passed or when <code>config.use_cache=True</code>) — It is a <a
  href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache"
>Cache</a> instance. For more details, see our <a
  href="https://huggingface.co/docs/transformers/en/kv_cache"
  rel="nofollow"
>kv cache guide</a>.</p>
<p>Contains pre-computed hidden-states (key and values in the attention blocks) that can be used (see
<code>past_key_values</code> input) to speed up sequential decoding.</p>
</li>
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithCrossAttentions"
>transformers.modeling_outputs.CausalLMOutputWithCrossAttentions</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),_e=new pe({props:{$$slots:{default:[ua]},$$scope:{ctx:k}}}),be=new ce({props:{anchor:"transformers.QDQBertLMHeadModel.forward.example",$$slots:{default:[fa]},$$scope:{ctx:k}}}),it=new C({props:{title:"QDQBertForMaskedLM",local:"transformers.QDQBertForMaskedLM",headingTag:"h2"}}),lt=new J({props:{name:"class transformers.QDQBertForMaskedLM",anchor:"transformers.QDQBertForMaskedLM",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.QDQBertForMaskedLM.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/qdqbert#transformers.QDQBertConfig">QDQBertConfig</a>) &#x2014; Model configuration class with all the parameters of the model.
Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/qdqbert/modeling_qdqbert.py#L1151"}}),dt=new J({props:{name:"forward",anchor:"transformers.QDQBertForMaskedLM.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"encoder_hidden_states",val:": typing.Optional[torch.FloatTensor] = None"},{name:"encoder_attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.QDQBertForMaskedLM.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.QDQBertForMaskedLM.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.QDQBertForMaskedLM.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.QDQBertForMaskedLM.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.max_position_embeddings - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.QDQBertForMaskedLM.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.QDQBertForMaskedLM.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.QDQBertForMaskedLM.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.QDQBertForMaskedLM.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.QDQBertForMaskedLM.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.QDQBertForMaskedLM.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should be in <code>[-100, 0, ..., config.vocab_size]</code> (see <code>input_ids</code> docstring) Tokens with indices set to <code>-100</code> are ignored (masked), the
loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>`,name:"labels"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/qdqbert/modeling_qdqbert.py#L1176",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.MaskedLMOutput"
>transformers.modeling_outputs.MaskedLMOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/qdqbert#transformers.QDQBertConfig"
>QDQBertConfig</a>) and inputs.</p>
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
`}}),Me=new pe({props:{$$slots:{default:[ga]},$$scope:{ctx:k}}}),ye=new ce({props:{anchor:"transformers.QDQBertForMaskedLM.forward.example",$$slots:{default:[_a]},$$scope:{ctx:k}}}),ct=new C({props:{title:"QDQBertForSequenceClassification",local:"transformers.QDQBertForSequenceClassification",headingTag:"h2"}}),pt=new J({props:{name:"class transformers.QDQBertForSequenceClassification",anchor:"transformers.QDQBertForSequenceClassification",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.QDQBertForSequenceClassification.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/qdqbert#transformers.QDQBertConfig">QDQBertConfig</a>) &#x2014; Model configuration class with all the parameters of the model.
Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/qdqbert/modeling_qdqbert.py#L1366"}}),mt=new J({props:{name:"forward",anchor:"transformers.QDQBertForSequenceClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.QDQBertForSequenceClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.QDQBertForSequenceClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.QDQBertForSequenceClassification.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.QDQBertForSequenceClassification.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.max_position_embeddings - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.QDQBertForSequenceClassification.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.QDQBertForSequenceClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.QDQBertForSequenceClassification.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.QDQBertForSequenceClassification.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.QDQBertForSequenceClassification.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.QDQBertForSequenceClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the sequence classification/regression loss. Indices should be in <code>[0, ..., config.num_labels - 1]</code>. If <code>config.num_labels == 1</code> a regression loss is computed (Mean-Square loss), If
<code>config.num_labels &gt; 1</code> a classification loss is computed (Cross-Entropy).`,name:"labels"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/qdqbert/modeling_qdqbert.py#L1378",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput"
>transformers.modeling_outputs.SequenceClassifierOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/qdqbert#transformers.QDQBertConfig"
>QDQBertConfig</a>) and inputs.</p>
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
`}}),Te=new pe({props:{$$slots:{default:[ba]},$$scope:{ctx:k}}}),we=new ce({props:{anchor:"transformers.QDQBertForSequenceClassification.forward.example",$$slots:{default:[Ma]},$$scope:{ctx:k}}}),ke=new ce({props:{anchor:"transformers.QDQBertForSequenceClassification.forward.example-2",$$slots:{default:[ya]},$$scope:{ctx:k}}}),ht=new C({props:{title:"QDQBertForNextSentencePrediction",local:"transformers.QDQBertForNextSentencePrediction",headingTag:"h2"}}),ut=new J({props:{name:"class transformers.QDQBertForNextSentencePrediction",anchor:"transformers.QDQBertForNextSentencePrediction",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.QDQBertForNextSentencePrediction.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/qdqbert#transformers.QDQBertConfig">QDQBertConfig</a>) &#x2014; Model configuration class with all the parameters of the model.
Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/qdqbert/modeling_qdqbert.py#L1262"}}),ft=new J({props:{name:"forward",anchor:"transformers.QDQBertForNextSentencePrediction.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.QDQBertForNextSentencePrediction.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.QDQBertForNextSentencePrediction.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.QDQBertForNextSentencePrediction.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.QDQBertForNextSentencePrediction.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.max_position_embeddings - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.QDQBertForNextSentencePrediction.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.QDQBertForNextSentencePrediction.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.QDQBertForNextSentencePrediction.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.QDQBertForNextSentencePrediction.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.QDQBertForNextSentencePrediction.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.QDQBertForNextSentencePrediction.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the next sequence prediction (classification) loss. Input should be a sequence pair
(see <code>input_ids</code> docstring). Indices should be in <code>[0, 1]</code>:</p>
<ul>
<li>0 indicates sequence B is a continuation of sequence A,</li>
<li>1 indicates sequence B is a random sequence.</li>
</ul>`,name:"labels"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/qdqbert/modeling_qdqbert.py#L1272",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.NextSentencePredictorOutput"
>transformers.modeling_outputs.NextSentencePredictorOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/qdqbert#transformers.QDQBertConfig"
>QDQBertConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>loss</strong> (<code>torch.FloatTensor</code> of shape <code>(1,)</code>, <em>optional</em>, returned when <code>next_sentence_label</code> is provided) — Next sequence prediction (classification) loss.</p>
</li>
<li>
<p><strong>logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, 2)</code>) — Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation
before SoftMax).</p>
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
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.NextSentencePredictorOutput"
>transformers.modeling_outputs.NextSentencePredictorOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),Qe=new pe({props:{$$slots:{default:[Ta]},$$scope:{ctx:k}}}),ve=new ce({props:{anchor:"transformers.QDQBertForNextSentencePrediction.forward.example",$$slots:{default:[wa]},$$scope:{ctx:k}}}),gt=new C({props:{title:"QDQBertForMultipleChoice",local:"transformers.QDQBertForMultipleChoice",headingTag:"h2"}}),_t=new J({props:{name:"class transformers.QDQBertForMultipleChoice",anchor:"transformers.QDQBertForMultipleChoice",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.QDQBertForMultipleChoice.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/qdqbert#transformers.QDQBertConfig">QDQBertConfig</a>) &#x2014; Model configuration class with all the parameters of the model.
Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/qdqbert/modeling_qdqbert.py#L1463"}}),bt=new J({props:{name:"forward",anchor:"transformers.QDQBertForMultipleChoice.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.QDQBertForMultipleChoice.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, num_choices, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.QDQBertForMultipleChoice.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_choices, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.QDQBertForMultipleChoice.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, num_choices, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.QDQBertForMultipleChoice.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, num_choices, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.max_position_embeddings - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.QDQBertForMultipleChoice.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.QDQBertForMultipleChoice.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_choices, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.QDQBertForMultipleChoice.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.QDQBertForMultipleChoice.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.QDQBertForMultipleChoice.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.QDQBertForMultipleChoice.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the multiple choice classification loss. Indices should be in <code>[0, ..., num_choices-1]</code> where <code>num_choices</code> is the size of the second dimension of the input tensors. (See
<code>input_ids</code> above)`,name:"labels"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/qdqbert/modeling_qdqbert.py#L1474",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.MultipleChoiceModelOutput"
>transformers.modeling_outputs.MultipleChoiceModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/qdqbert#transformers.QDQBertConfig"
>QDQBertConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>loss</strong> (<code>torch.FloatTensor</code> of shape <em>(1,)</em>, <em>optional</em>, returned when <code>labels</code> is provided) — Classification loss.</p>
</li>
<li>
<p><strong>logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_choices)</code>) — <em>num_choices</em> is the second dimension of the input tensors. (see <em>input_ids</em> above).</p>
<p>Classification scores (before SoftMax).</p>
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
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.MultipleChoiceModelOutput"
>transformers.modeling_outputs.MultipleChoiceModelOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),je=new pe({props:{$$slots:{default:[ka]},$$scope:{ctx:k}}}),Be=new ce({props:{anchor:"transformers.QDQBertForMultipleChoice.forward.example",$$slots:{default:[Qa]},$$scope:{ctx:k}}}),Mt=new C({props:{title:"QDQBertForTokenClassification",local:"transformers.QDQBertForTokenClassification",headingTag:"h2"}}),yt=new J({props:{name:"class transformers.QDQBertForTokenClassification",anchor:"transformers.QDQBertForTokenClassification",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.QDQBertForTokenClassification.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/qdqbert#transformers.QDQBertConfig">QDQBertConfig</a>) &#x2014; Model configuration class with all the parameters of the model.
Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/qdqbert/modeling_qdqbert.py#L1554"}}),Tt=new J({props:{name:"forward",anchor:"transformers.QDQBertForTokenClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.QDQBertForTokenClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.QDQBertForTokenClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.QDQBertForTokenClassification.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.QDQBertForTokenClassification.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.max_position_embeddings - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.QDQBertForTokenClassification.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.QDQBertForTokenClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.QDQBertForTokenClassification.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.QDQBertForTokenClassification.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.QDQBertForTokenClassification.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.QDQBertForTokenClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the token classification loss. Indices should be in <code>[0, ..., config.num_labels - 1]</code>.`,name:"labels"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/qdqbert/modeling_qdqbert.py#L1566",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput"
>transformers.modeling_outputs.TokenClassifierOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/qdqbert#transformers.QDQBertConfig"
>QDQBertConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>loss</strong> (<code>torch.FloatTensor</code> of shape <code>(1,)</code>, <em>optional</em>, returned when <code>labels</code> is provided)  — Classification loss.</p>
</li>
<li>
<p><strong>logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, config.num_labels)</code>) — Classification scores (before SoftMax).</p>
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
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput"
>transformers.modeling_outputs.TokenClassifierOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),$e=new pe({props:{$$slots:{default:[va]},$$scope:{ctx:k}}}),Je=new ce({props:{anchor:"transformers.QDQBertForTokenClassification.forward.example",$$slots:{default:[ja]},$$scope:{ctx:k}}}),wt=new C({props:{title:"QDQBertForQuestionAnswering",local:"transformers.QDQBertForQuestionAnswering",headingTag:"h2"}}),kt=new J({props:{name:"class transformers.QDQBertForQuestionAnswering",anchor:"transformers.QDQBertForQuestionAnswering",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.QDQBertForQuestionAnswering.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/qdqbert#transformers.QDQBertConfig">QDQBertConfig</a>) &#x2014; Model configuration class with all the parameters of the model.
Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/qdqbert/modeling_qdqbert.py#L1632"}}),Qt=new J({props:{name:"forward",anchor:"transformers.QDQBertForQuestionAnswering.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"start_positions",val:": typing.Optional[torch.LongTensor] = None"},{name:"end_positions",val:": typing.Optional[torch.LongTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.QDQBertForQuestionAnswering.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.QDQBertForQuestionAnswering.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.QDQBertForQuestionAnswering.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.QDQBertForQuestionAnswering.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.max_position_embeddings - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.QDQBertForQuestionAnswering.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.QDQBertForQuestionAnswering.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.QDQBertForQuestionAnswering.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.QDQBertForQuestionAnswering.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.QDQBertForQuestionAnswering.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.QDQBertForQuestionAnswering.forward.start_positions",description:`<strong>start_positions</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for position (index) of the start of the labelled span for computing the token classification loss.
Positions are clamped to the length of the sequence (<code>sequence_length</code>). Position outside of the sequence
are not taken into account for computing the loss.`,name:"start_positions"},{anchor:"transformers.QDQBertForQuestionAnswering.forward.end_positions",description:`<strong>end_positions</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for position (index) of the end of the labelled span for computing the token classification loss.
Positions are clamped to the length of the sequence (<code>sequence_length</code>). Position outside of the sequence
are not taken into account for computing the loss.`,name:"end_positions"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/qdqbert/modeling_qdqbert.py#L1643",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.QuestionAnsweringModelOutput"
>transformers.modeling_outputs.QuestionAnsweringModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/qdqbert#transformers.QDQBertConfig"
>QDQBertConfig</a>) and inputs.</p>
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
`}}),Ce=new pe({props:{$$slots:{default:[Ba]},$$scope:{ctx:k}}}),Ue=new ce({props:{anchor:"transformers.QDQBertForQuestionAnswering.forward.example",$$slots:{default:[$a]},$$scope:{ctx:k}}}),vt=new da({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/qdqbert.md"}}),{c(){t=p("meta"),y=a(),i=p("p"),d=a(),T=p("p"),T.innerHTML=n,w=a(),u(Fe.$$.fragment),cn=a(),me=p("div"),me.innerHTML=ls,pn=a(),u(he.$$.fragment),mn=a(),u(ze.$$.fragment),hn=a(),xe=p("p"),xe.innerHTML=ds,un=a(),qe=p("p"),qe.textContent=cs,fn=a(),Ze=p("p"),Ze.innerHTML=ps,gn=a(),We=p("p"),We.innerHTML=ms,_n=a(),u(Ie.$$.fragment),bn=a(),De=p("ul"),De.innerHTML=hs,Mn=a(),u(Re.$$.fragment),yn=a(),Xe=p("p"),Xe.innerHTML=us,Tn=a(),Ne=p("p"),Ne.innerHTML=fs,wn=a(),Ge=p("p"),Ge.textContent=gs,kn=a(),u(Le.$$.fragment),Qn=a(),u(Ve.$$.fragment),vn=a(),He=p("p"),He.textContent=_s,jn=a(),u(Se.$$.fragment),Bn=a(),u(Ye.$$.fragment),$n=a(),Ee=p("p"),Ee.innerHTML=bs,Jn=a(),u(Pe.$$.fragment),Cn=a(),u(Ae.$$.fragment),Un=a(),Oe=p("ul"),Oe.innerHTML=Ms,Fn=a(),u(Ke.$$.fragment),zn=a(),D=p("div"),u(et.$$.fragment),Kn=a(),jt=p("p"),jt.innerHTML=ys,eo=a(),Bt=p("p"),Bt.innerHTML=Ts,to=a(),u(ue.$$.fragment),xn=a(),u(tt.$$.fragment),qn=a(),v=p("div"),u(nt.$$.fragment),no=a(),$t=p("p"),$t.textContent=ws,oo=a(),Jt=p("p"),Jt.innerHTML=ks,so=a(),Ct=p("p"),Ct.innerHTML=Qs,ao=a(),Ut=p("p"),Ut.innerHTML=vs,ro=a(),Ft=p("p"),Ft.innerHTML=js,io=a(),E=p("div"),u(ot.$$.fragment),lo=a(),zt=p("p"),zt.innerHTML=Bs,co=a(),u(fe.$$.fragment),po=a(),u(ge.$$.fragment),Zn=a(),u(st.$$.fragment),Wn=a(),U=p("div"),u(at.$$.fragment),mo=a(),xt=p("p"),xt.innerHTML=$s,ho=a(),qt=p("p"),qt.innerHTML=Js,uo=a(),Zt=p("p"),Zt.innerHTML=Cs,fo=a(),P=p("div"),u(rt.$$.fragment),go=a(),Wt=p("p"),Wt.innerHTML=Us,_o=a(),u(_e.$$.fragment),bo=a(),u(be.$$.fragment),In=a(),u(it.$$.fragment),Dn=a(),F=p("div"),u(lt.$$.fragment),Mo=a(),It=p("p"),It.innerHTML=Fs,yo=a(),Dt=p("p"),Dt.innerHTML=zs,To=a(),Rt=p("p"),Rt.innerHTML=xs,wo=a(),A=p("div"),u(dt.$$.fragment),ko=a(),Xt=p("p"),Xt.innerHTML=qs,Qo=a(),u(Me.$$.fragment),vo=a(),u(ye.$$.fragment),Rn=a(),u(ct.$$.fragment),Xn=a(),z=p("div"),u(pt.$$.fragment),jo=a(),Nt=p("p"),Nt.textContent=Zs,Bo=a(),Gt=p("p"),Gt.innerHTML=Ws,$o=a(),Lt=p("p"),Lt.innerHTML=Is,Jo=a(),I=p("div"),u(mt.$$.fragment),Co=a(),Vt=p("p"),Vt.innerHTML=Ds,Uo=a(),u(Te.$$.fragment),Fo=a(),u(we.$$.fragment),zo=a(),u(ke.$$.fragment),Nn=a(),u(ht.$$.fragment),Gn=a(),x=p("div"),u(ut.$$.fragment),xo=a(),Ht=p("p"),Ht.innerHTML=Rs,qo=a(),St=p("p"),St.innerHTML=Xs,Zo=a(),Yt=p("p"),Yt.innerHTML=Ns,Wo=a(),O=p("div"),u(ft.$$.fragment),Io=a(),Et=p("p"),Et.innerHTML=Gs,Do=a(),u(Qe.$$.fragment),Ro=a(),u(ve.$$.fragment),Ln=a(),u(gt.$$.fragment),Vn=a(),q=p("div"),u(_t.$$.fragment),Xo=a(),Pt=p("p"),Pt.textContent=Ls,No=a(),At=p("p"),At.innerHTML=Vs,Go=a(),Ot=p("p"),Ot.innerHTML=Hs,Lo=a(),K=p("div"),u(bt.$$.fragment),Vo=a(),Kt=p("p"),Kt.innerHTML=Ss,Ho=a(),u(je.$$.fragment),So=a(),u(Be.$$.fragment),Hn=a(),u(Mt.$$.fragment),Sn=a(),Z=p("div"),u(yt.$$.fragment),Yo=a(),en=p("p"),en.textContent=Ys,Eo=a(),tn=p("p"),tn.innerHTML=Es,Po=a(),nn=p("p"),nn.innerHTML=Ps,Ao=a(),ee=p("div"),u(Tt.$$.fragment),Oo=a(),on=p("p"),on.innerHTML=As,Ko=a(),u($e.$$.fragment),es=a(),u(Je.$$.fragment),Yn=a(),u(wt.$$.fragment),En=a(),W=p("div"),u(kt.$$.fragment),ts=a(),sn=p("p"),sn.innerHTML=Os,ns=a(),an=p("p"),an.innerHTML=Ks,os=a(),rn=p("p"),rn.innerHTML=ea,ss=a(),te=p("div"),u(Qt.$$.fragment),as=a(),ln=p("p"),ln.innerHTML=ta,rs=a(),u(Ce.$$.fragment),is=a(),u(Ue.$$.fragment),Pn=a(),u(vt.$$.fragment),An=a(),dn=p("p"),this.h()},l(e){const o=la("svelte-u9bgzb",document.head);t=m(o,"META",{name:!0,content:!0}),o.forEach(s),y=r(e),i=m(e,"P",{}),j(i).forEach(s),d=r(e),T=m(e,"P",{"data-svelte-h":!0}),h(T)!=="svelte-1x295qr"&&(T.innerHTML=n),w=r(e),f(Fe.$$.fragment,e),cn=r(e),me=m(e,"DIV",{class:!0,"data-svelte-h":!0}),h(me)!=="svelte-13t8s2t"&&(me.innerHTML=ls),pn=r(e),f(he.$$.fragment,e),mn=r(e),f(ze.$$.fragment,e),hn=r(e),xe=m(e,"P",{"data-svelte-h":!0}),h(xe)!=="svelte-1xdfjhg"&&(xe.innerHTML=ds),un=r(e),qe=m(e,"P",{"data-svelte-h":!0}),h(qe)!=="svelte-vfdo9a"&&(qe.textContent=cs),fn=r(e),Ze=m(e,"P",{"data-svelte-h":!0}),h(Ze)!=="svelte-nciigz"&&(Ze.innerHTML=ps),gn=r(e),We=m(e,"P",{"data-svelte-h":!0}),h(We)!=="svelte-14q60ph"&&(We.innerHTML=ms),_n=r(e),f(Ie.$$.fragment,e),bn=r(e),De=m(e,"UL",{"data-svelte-h":!0}),h(De)!=="svelte-1wat0qj"&&(De.innerHTML=hs),Mn=r(e),f(Re.$$.fragment,e),yn=r(e),Xe=m(e,"P",{"data-svelte-h":!0}),h(Xe)!=="svelte-1dko7r6"&&(Xe.innerHTML=us),Tn=r(e),Ne=m(e,"P",{"data-svelte-h":!0}),h(Ne)!=="svelte-1bset6k"&&(Ne.innerHTML=fs),wn=r(e),Ge=m(e,"P",{"data-svelte-h":!0}),h(Ge)!=="svelte-11lpom8"&&(Ge.textContent=gs),kn=r(e),f(Le.$$.fragment,e),Qn=r(e),f(Ve.$$.fragment,e),vn=r(e),He=m(e,"P",{"data-svelte-h":!0}),h(He)!=="svelte-1xk1npv"&&(He.textContent=_s),jn=r(e),f(Se.$$.fragment,e),Bn=r(e),f(Ye.$$.fragment,e),$n=r(e),Ee=m(e,"P",{"data-svelte-h":!0}),h(Ee)!=="svelte-8256ng"&&(Ee.innerHTML=bs),Jn=r(e),f(Pe.$$.fragment,e),Cn=r(e),f(Ae.$$.fragment,e),Un=r(e),Oe=m(e,"UL",{"data-svelte-h":!0}),h(Oe)!=="svelte-p1b16m"&&(Oe.innerHTML=Ms),Fn=r(e),f(Ke.$$.fragment,e),zn=r(e),D=m(e,"DIV",{class:!0});var ne=j(D);f(et.$$.fragment,ne),Kn=r(ne),jt=m(ne,"P",{"data-svelte-h":!0}),h(jt)!=="svelte-1van7js"&&(jt.innerHTML=ys),eo=r(ne),Bt=m(ne,"P",{"data-svelte-h":!0}),h(Bt)!=="svelte-1ek1ss9"&&(Bt.innerHTML=Ts),to=r(ne),f(ue.$$.fragment,ne),ne.forEach(s),xn=r(e),f(tt.$$.fragment,e),qn=r(e),v=m(e,"DIV",{class:!0});var $=j(v);f(nt.$$.fragment,$),no=r($),$t=m($,"P",{"data-svelte-h":!0}),h($t)!=="svelte-bx7qi6"&&($t.textContent=ws),oo=r($),Jt=m($,"P",{"data-svelte-h":!0}),h(Jt)!=="svelte-q52n56"&&(Jt.innerHTML=ks),so=r($),Ct=m($,"P",{"data-svelte-h":!0}),h(Ct)!=="svelte-hswkmf"&&(Ct.innerHTML=Qs),ao=r($),Ut=m($,"P",{"data-svelte-h":!0}),h(Ut)!=="svelte-1854dma"&&(Ut.innerHTML=vs),ro=r($),Ft=m($,"P",{"data-svelte-h":!0}),h(Ft)!=="svelte-174erte"&&(Ft.innerHTML=js),io=r($),E=m($,"DIV",{class:!0});var oe=j(E);f(ot.$$.fragment,oe),lo=r(oe),zt=m(oe,"P",{"data-svelte-h":!0}),h(zt)!=="svelte-1f4f4ia"&&(zt.innerHTML=Bs),co=r(oe),f(fe.$$.fragment,oe),po=r(oe),f(ge.$$.fragment,oe),oe.forEach(s),$.forEach(s),Zn=r(e),f(st.$$.fragment,e),Wn=r(e),U=m(e,"DIV",{class:!0});var R=j(U);f(at.$$.fragment,R),mo=r(R),xt=m(R,"P",{"data-svelte-h":!0}),h(xt)!=="svelte-noxvqu"&&(xt.innerHTML=$s),ho=r(R),qt=m(R,"P",{"data-svelte-h":!0}),h(qt)!=="svelte-q52n56"&&(qt.innerHTML=Js),uo=r(R),Zt=m(R,"P",{"data-svelte-h":!0}),h(Zt)!=="svelte-hswkmf"&&(Zt.innerHTML=Cs),fo=r(R),P=m(R,"DIV",{class:!0});var se=j(P);f(rt.$$.fragment,se),go=r(se),Wt=m(se,"P",{"data-svelte-h":!0}),h(Wt)!=="svelte-1m3kybo"&&(Wt.innerHTML=Us),_o=r(se),f(_e.$$.fragment,se),bo=r(se),f(be.$$.fragment,se),se.forEach(s),R.forEach(s),In=r(e),f(it.$$.fragment,e),Dn=r(e),F=m(e,"DIV",{class:!0});var X=j(F);f(lt.$$.fragment,X),Mo=r(X),It=m(X,"P",{"data-svelte-h":!0}),h(It)!=="svelte-1ulv96d"&&(It.innerHTML=Fs),yo=r(X),Dt=m(X,"P",{"data-svelte-h":!0}),h(Dt)!=="svelte-q52n56"&&(Dt.innerHTML=zs),To=r(X),Rt=m(X,"P",{"data-svelte-h":!0}),h(Rt)!=="svelte-hswkmf"&&(Rt.innerHTML=xs),wo=r(X),A=m(X,"DIV",{class:!0});var ae=j(A);f(dt.$$.fragment,ae),ko=r(ae),Xt=m(ae,"P",{"data-svelte-h":!0}),h(Xt)!=="svelte-zc2m8y"&&(Xt.innerHTML=qs),Qo=r(ae),f(Me.$$.fragment,ae),vo=r(ae),f(ye.$$.fragment,ae),ae.forEach(s),X.forEach(s),Rn=r(e),f(ct.$$.fragment,e),Xn=r(e),z=m(e,"DIV",{class:!0});var N=j(z);f(pt.$$.fragment,N),jo=r(N),Nt=m(N,"P",{"data-svelte-h":!0}),h(Nt)!=="svelte-mumz6f"&&(Nt.textContent=Zs),Bo=r(N),Gt=m(N,"P",{"data-svelte-h":!0}),h(Gt)!=="svelte-q52n56"&&(Gt.innerHTML=Ws),$o=r(N),Lt=m(N,"P",{"data-svelte-h":!0}),h(Lt)!=="svelte-hswkmf"&&(Lt.innerHTML=Is),Jo=r(N),I=m(N,"DIV",{class:!0});var G=j(I);f(mt.$$.fragment,G),Co=r(G),Vt=m(G,"P",{"data-svelte-h":!0}),h(Vt)!=="svelte-bhzwkg"&&(Vt.innerHTML=Ds),Uo=r(G),f(Te.$$.fragment,G),Fo=r(G),f(we.$$.fragment,G),zo=r(G),f(ke.$$.fragment,G),G.forEach(s),N.forEach(s),Nn=r(e),f(ht.$$.fragment,e),Gn=r(e),x=m(e,"DIV",{class:!0});var L=j(x);f(ut.$$.fragment,L),xo=r(L),Ht=m(L,"P",{"data-svelte-h":!0}),h(Ht)!=="svelte-1yvdzzs"&&(Ht.innerHTML=Rs),qo=r(L),St=m(L,"P",{"data-svelte-h":!0}),h(St)!=="svelte-q52n56"&&(St.innerHTML=Xs),Zo=r(L),Yt=m(L,"P",{"data-svelte-h":!0}),h(Yt)!=="svelte-hswkmf"&&(Yt.innerHTML=Ns),Wo=r(L),O=m(L,"DIV",{class:!0});var re=j(O);f(ft.$$.fragment,re),Io=r(re),Et=m(re,"P",{"data-svelte-h":!0}),h(Et)!=="svelte-a5op3c"&&(Et.innerHTML=Gs),Do=r(re),f(Qe.$$.fragment,re),Ro=r(re),f(ve.$$.fragment,re),re.forEach(s),L.forEach(s),Ln=r(e),f(gt.$$.fragment,e),Vn=r(e),q=m(e,"DIV",{class:!0});var V=j(q);f(_t.$$.fragment,V),Xo=r(V),Pt=m(V,"P",{"data-svelte-h":!0}),h(Pt)!=="svelte-4kts57"&&(Pt.textContent=Ls),No=r(V),At=m(V,"P",{"data-svelte-h":!0}),h(At)!=="svelte-q52n56"&&(At.innerHTML=Vs),Go=r(V),Ot=m(V,"P",{"data-svelte-h":!0}),h(Ot)!=="svelte-hswkmf"&&(Ot.innerHTML=Hs),Lo=r(V),K=m(V,"DIV",{class:!0});var ie=j(K);f(bt.$$.fragment,ie),Vo=r(ie),Kt=m(ie,"P",{"data-svelte-h":!0}),h(Kt)!=="svelte-83j1zc"&&(Kt.innerHTML=Ss),Ho=r(ie),f(je.$$.fragment,ie),So=r(ie),f(Be.$$.fragment,ie),ie.forEach(s),V.forEach(s),Hn=r(e),f(Mt.$$.fragment,e),Sn=r(e),Z=m(e,"DIV",{class:!0});var H=j(Z);f(yt.$$.fragment,H),Yo=r(H),en=m(H,"P",{"data-svelte-h":!0}),h(en)!=="svelte-1ynzopo"&&(en.textContent=Ys),Eo=r(H),tn=m(H,"P",{"data-svelte-h":!0}),h(tn)!=="svelte-q52n56"&&(tn.innerHTML=Es),Po=r(H),nn=m(H,"P",{"data-svelte-h":!0}),h(nn)!=="svelte-hswkmf"&&(nn.innerHTML=Ps),Ao=r(H),ee=m(H,"DIV",{class:!0});var le=j(ee);f(Tt.$$.fragment,le),Oo=r(le),on=m(le,"P",{"data-svelte-h":!0}),h(on)!=="svelte-efhtdm"&&(on.innerHTML=As),Ko=r(le),f($e.$$.fragment,le),es=r(le),f(Je.$$.fragment,le),le.forEach(s),H.forEach(s),Yn=r(e),f(wt.$$.fragment,e),En=r(e),W=m(e,"DIV",{class:!0});var S=j(W);f(kt.$$.fragment,S),ts=r(S),sn=m(S,"P",{"data-svelte-h":!0}),h(sn)!=="svelte-97if0a"&&(sn.innerHTML=Os),ns=r(S),an=m(S,"P",{"data-svelte-h":!0}),h(an)!=="svelte-q52n56"&&(an.innerHTML=Ks),os=r(S),rn=m(S,"P",{"data-svelte-h":!0}),h(rn)!=="svelte-hswkmf"&&(rn.innerHTML=ea),ss=r(S),te=m(S,"DIV",{class:!0});var de=j(te);f(Qt.$$.fragment,de),as=r(de),ln=m(de,"P",{"data-svelte-h":!0}),h(ln)!=="svelte-1fjdjve"&&(ln.innerHTML=ta),rs=r(de),f(Ce.$$.fragment,de),is=r(de),f(Ue.$$.fragment,de),de.forEach(s),S.forEach(s),Pn=r(e),f(vt.$$.fragment,e),An=r(e),dn=m(e,"P",{}),j(dn).forEach(s),this.h()},h(){Q(t,"name","hf:doc:metadata"),Q(t,"content",Ca),Q(me,"class","flex flex-wrap space-x-1"),Q(D,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),Q(E,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),Q(v,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),Q(P,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),Q(U,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),Q(A,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),Q(F,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),Q(I,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),Q(z,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),Q(O,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),Q(x,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),Q(K,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),Q(q,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),Q(ee,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),Q(Z,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),Q(te,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),Q(W,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,o){l(document.head,t),c(e,y,o),c(e,i,o),c(e,d,o),c(e,T,o),c(e,w,o),g(Fe,e,o),c(e,cn,o),c(e,me,o),c(e,pn,o),g(he,e,o),c(e,mn,o),g(ze,e,o),c(e,hn,o),c(e,xe,o),c(e,un,o),c(e,qe,o),c(e,fn,o),c(e,Ze,o),c(e,gn,o),c(e,We,o),c(e,_n,o),g(Ie,e,o),c(e,bn,o),c(e,De,o),c(e,Mn,o),g(Re,e,o),c(e,yn,o),c(e,Xe,o),c(e,Tn,o),c(e,Ne,o),c(e,wn,o),c(e,Ge,o),c(e,kn,o),g(Le,e,o),c(e,Qn,o),g(Ve,e,o),c(e,vn,o),c(e,He,o),c(e,jn,o),g(Se,e,o),c(e,Bn,o),g(Ye,e,o),c(e,$n,o),c(e,Ee,o),c(e,Jn,o),g(Pe,e,o),c(e,Cn,o),g(Ae,e,o),c(e,Un,o),c(e,Oe,o),c(e,Fn,o),g(Ke,e,o),c(e,zn,o),c(e,D,o),g(et,D,null),l(D,Kn),l(D,jt),l(D,eo),l(D,Bt),l(D,to),g(ue,D,null),c(e,xn,o),g(tt,e,o),c(e,qn,o),c(e,v,o),g(nt,v,null),l(v,no),l(v,$t),l(v,oo),l(v,Jt),l(v,so),l(v,Ct),l(v,ao),l(v,Ut),l(v,ro),l(v,Ft),l(v,io),l(v,E),g(ot,E,null),l(E,lo),l(E,zt),l(E,co),g(fe,E,null),l(E,po),g(ge,E,null),c(e,Zn,o),g(st,e,o),c(e,Wn,o),c(e,U,o),g(at,U,null),l(U,mo),l(U,xt),l(U,ho),l(U,qt),l(U,uo),l(U,Zt),l(U,fo),l(U,P),g(rt,P,null),l(P,go),l(P,Wt),l(P,_o),g(_e,P,null),l(P,bo),g(be,P,null),c(e,In,o),g(it,e,o),c(e,Dn,o),c(e,F,o),g(lt,F,null),l(F,Mo),l(F,It),l(F,yo),l(F,Dt),l(F,To),l(F,Rt),l(F,wo),l(F,A),g(dt,A,null),l(A,ko),l(A,Xt),l(A,Qo),g(Me,A,null),l(A,vo),g(ye,A,null),c(e,Rn,o),g(ct,e,o),c(e,Xn,o),c(e,z,o),g(pt,z,null),l(z,jo),l(z,Nt),l(z,Bo),l(z,Gt),l(z,$o),l(z,Lt),l(z,Jo),l(z,I),g(mt,I,null),l(I,Co),l(I,Vt),l(I,Uo),g(Te,I,null),l(I,Fo),g(we,I,null),l(I,zo),g(ke,I,null),c(e,Nn,o),g(ht,e,o),c(e,Gn,o),c(e,x,o),g(ut,x,null),l(x,xo),l(x,Ht),l(x,qo),l(x,St),l(x,Zo),l(x,Yt),l(x,Wo),l(x,O),g(ft,O,null),l(O,Io),l(O,Et),l(O,Do),g(Qe,O,null),l(O,Ro),g(ve,O,null),c(e,Ln,o),g(gt,e,o),c(e,Vn,o),c(e,q,o),g(_t,q,null),l(q,Xo),l(q,Pt),l(q,No),l(q,At),l(q,Go),l(q,Ot),l(q,Lo),l(q,K),g(bt,K,null),l(K,Vo),l(K,Kt),l(K,Ho),g(je,K,null),l(K,So),g(Be,K,null),c(e,Hn,o),g(Mt,e,o),c(e,Sn,o),c(e,Z,o),g(yt,Z,null),l(Z,Yo),l(Z,en),l(Z,Eo),l(Z,tn),l(Z,Po),l(Z,nn),l(Z,Ao),l(Z,ee),g(Tt,ee,null),l(ee,Oo),l(ee,on),l(ee,Ko),g($e,ee,null),l(ee,es),g(Je,ee,null),c(e,Yn,o),g(wt,e,o),c(e,En,o),c(e,W,o),g(kt,W,null),l(W,ts),l(W,sn),l(W,ns),l(W,an),l(W,os),l(W,rn),l(W,ss),l(W,te),g(Qt,te,null),l(te,as),l(te,ln),l(te,rs),g(Ce,te,null),l(te,is),g(Ue,te,null),c(e,Pn,o),g(vt,e,o),c(e,An,o),c(e,dn,o),On=!0},p(e,[o]){const ne={};o&2&&(ne.$$scope={dirty:o,ctx:e}),he.$set(ne);const $={};o&2&&($.$$scope={dirty:o,ctx:e}),ue.$set($);const oe={};o&2&&(oe.$$scope={dirty:o,ctx:e}),fe.$set(oe);const R={};o&2&&(R.$$scope={dirty:o,ctx:e}),ge.$set(R);const se={};o&2&&(se.$$scope={dirty:o,ctx:e}),_e.$set(se);const X={};o&2&&(X.$$scope={dirty:o,ctx:e}),be.$set(X);const ae={};o&2&&(ae.$$scope={dirty:o,ctx:e}),Me.$set(ae);const N={};o&2&&(N.$$scope={dirty:o,ctx:e}),ye.$set(N);const G={};o&2&&(G.$$scope={dirty:o,ctx:e}),Te.$set(G);const L={};o&2&&(L.$$scope={dirty:o,ctx:e}),we.$set(L);const re={};o&2&&(re.$$scope={dirty:o,ctx:e}),ke.$set(re);const V={};o&2&&(V.$$scope={dirty:o,ctx:e}),Qe.$set(V);const ie={};o&2&&(ie.$$scope={dirty:o,ctx:e}),ve.$set(ie);const H={};o&2&&(H.$$scope={dirty:o,ctx:e}),je.$set(H);const le={};o&2&&(le.$$scope={dirty:o,ctx:e}),Be.$set(le);const S={};o&2&&(S.$$scope={dirty:o,ctx:e}),$e.$set(S);const de={};o&2&&(de.$$scope={dirty:o,ctx:e}),Je.$set(de);const na={};o&2&&(na.$$scope={dirty:o,ctx:e}),Ce.$set(na);const oa={};o&2&&(oa.$$scope={dirty:o,ctx:e}),Ue.$set(oa)},i(e){On||(_(Fe.$$.fragment,e),_(he.$$.fragment,e),_(ze.$$.fragment,e),_(Ie.$$.fragment,e),_(Re.$$.fragment,e),_(Le.$$.fragment,e),_(Ve.$$.fragment,e),_(Se.$$.fragment,e),_(Ye.$$.fragment,e),_(Pe.$$.fragment,e),_(Ae.$$.fragment,e),_(Ke.$$.fragment,e),_(et.$$.fragment,e),_(ue.$$.fragment,e),_(tt.$$.fragment,e),_(nt.$$.fragment,e),_(ot.$$.fragment,e),_(fe.$$.fragment,e),_(ge.$$.fragment,e),_(st.$$.fragment,e),_(at.$$.fragment,e),_(rt.$$.fragment,e),_(_e.$$.fragment,e),_(be.$$.fragment,e),_(it.$$.fragment,e),_(lt.$$.fragment,e),_(dt.$$.fragment,e),_(Me.$$.fragment,e),_(ye.$$.fragment,e),_(ct.$$.fragment,e),_(pt.$$.fragment,e),_(mt.$$.fragment,e),_(Te.$$.fragment,e),_(we.$$.fragment,e),_(ke.$$.fragment,e),_(ht.$$.fragment,e),_(ut.$$.fragment,e),_(ft.$$.fragment,e),_(Qe.$$.fragment,e),_(ve.$$.fragment,e),_(gt.$$.fragment,e),_(_t.$$.fragment,e),_(bt.$$.fragment,e),_(je.$$.fragment,e),_(Be.$$.fragment,e),_(Mt.$$.fragment,e),_(yt.$$.fragment,e),_(Tt.$$.fragment,e),_($e.$$.fragment,e),_(Je.$$.fragment,e),_(wt.$$.fragment,e),_(kt.$$.fragment,e),_(Qt.$$.fragment,e),_(Ce.$$.fragment,e),_(Ue.$$.fragment,e),_(vt.$$.fragment,e),On=!0)},o(e){b(Fe.$$.fragment,e),b(he.$$.fragment,e),b(ze.$$.fragment,e),b(Ie.$$.fragment,e),b(Re.$$.fragment,e),b(Le.$$.fragment,e),b(Ve.$$.fragment,e),b(Se.$$.fragment,e),b(Ye.$$.fragment,e),b(Pe.$$.fragment,e),b(Ae.$$.fragment,e),b(Ke.$$.fragment,e),b(et.$$.fragment,e),b(ue.$$.fragment,e),b(tt.$$.fragment,e),b(nt.$$.fragment,e),b(ot.$$.fragment,e),b(fe.$$.fragment,e),b(ge.$$.fragment,e),b(st.$$.fragment,e),b(at.$$.fragment,e),b(rt.$$.fragment,e),b(_e.$$.fragment,e),b(be.$$.fragment,e),b(it.$$.fragment,e),b(lt.$$.fragment,e),b(dt.$$.fragment,e),b(Me.$$.fragment,e),b(ye.$$.fragment,e),b(ct.$$.fragment,e),b(pt.$$.fragment,e),b(mt.$$.fragment,e),b(Te.$$.fragment,e),b(we.$$.fragment,e),b(ke.$$.fragment,e),b(ht.$$.fragment,e),b(ut.$$.fragment,e),b(ft.$$.fragment,e),b(Qe.$$.fragment,e),b(ve.$$.fragment,e),b(gt.$$.fragment,e),b(_t.$$.fragment,e),b(bt.$$.fragment,e),b(je.$$.fragment,e),b(Be.$$.fragment,e),b(Mt.$$.fragment,e),b(yt.$$.fragment,e),b(Tt.$$.fragment,e),b($e.$$.fragment,e),b(Je.$$.fragment,e),b(wt.$$.fragment,e),b(kt.$$.fragment,e),b(Qt.$$.fragment,e),b(Ce.$$.fragment,e),b(Ue.$$.fragment,e),b(vt.$$.fragment,e),On=!1},d(e){e&&(s(y),s(i),s(d),s(T),s(w),s(cn),s(me),s(pn),s(mn),s(hn),s(xe),s(un),s(qe),s(fn),s(Ze),s(gn),s(We),s(_n),s(bn),s(De),s(Mn),s(yn),s(Xe),s(Tn),s(Ne),s(wn),s(Ge),s(kn),s(Qn),s(vn),s(He),s(jn),s(Bn),s($n),s(Ee),s(Jn),s(Cn),s(Un),s(Oe),s(Fn),s(zn),s(D),s(xn),s(qn),s(v),s(Zn),s(Wn),s(U),s(In),s(Dn),s(F),s(Rn),s(Xn),s(z),s(Nn),s(Gn),s(x),s(Ln),s(Vn),s(q),s(Hn),s(Sn),s(Z),s(Yn),s(En),s(W),s(Pn),s(An),s(dn)),s(t),M(Fe,e),M(he,e),M(ze,e),M(Ie,e),M(Re,e),M(Le,e),M(Ve,e),M(Se,e),M(Ye,e),M(Pe,e),M(Ae,e),M(Ke,e),M(et),M(ue),M(tt,e),M(nt),M(ot),M(fe),M(ge),M(st,e),M(at),M(rt),M(_e),M(be),M(it,e),M(lt),M(dt),M(Me),M(ye),M(ct,e),M(pt),M(mt),M(Te),M(we),M(ke),M(ht,e),M(ut),M(ft),M(Qe),M(ve),M(gt,e),M(_t),M(bt),M(je),M(Be),M(Mt,e),M(yt),M(Tt),M($e),M(Je),M(wt,e),M(kt),M(Qt),M(Ce),M(Ue),M(vt,e)}}}const Ca='{"title":"QDQBERT","local":"qdqbert","sections":[{"title":"Overview","local":"overview","sections":[],"depth":2},{"title":"Usage tips","local":"usage-tips","sections":[{"title":"Set default quantizers","local":"set-default-quantizers","sections":[],"depth":3},{"title":"Calibration","local":"calibration","sections":[],"depth":3},{"title":"Export to ONNX","local":"export-to-onnx","sections":[],"depth":3}],"depth":2},{"title":"Resources","local":"resources","sections":[],"depth":2},{"title":"QDQBertConfig","local":"transformers.QDQBertConfig","sections":[],"depth":2},{"title":"QDQBertModel","local":"transformers.QDQBertModel","sections":[],"depth":2},{"title":"QDQBertLMHeadModel","local":"transformers.QDQBertLMHeadModel","sections":[],"depth":2},{"title":"QDQBertForMaskedLM","local":"transformers.QDQBertForMaskedLM","sections":[],"depth":2},{"title":"QDQBertForSequenceClassification","local":"transformers.QDQBertForSequenceClassification","sections":[],"depth":2},{"title":"QDQBertForNextSentencePrediction","local":"transformers.QDQBertForNextSentencePrediction","sections":[],"depth":2},{"title":"QDQBertForMultipleChoice","local":"transformers.QDQBertForMultipleChoice","sections":[],"depth":2},{"title":"QDQBertForTokenClassification","local":"transformers.QDQBertForTokenClassification","sections":[],"depth":2},{"title":"QDQBertForQuestionAnswering","local":"transformers.QDQBertForQuestionAnswering","sections":[],"depth":2}],"depth":1}';function Ua(k){return aa(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class Da extends ra{constructor(t){super(),ia(this,t,Ua,Ja,sa,{})}}export{Da as component};
