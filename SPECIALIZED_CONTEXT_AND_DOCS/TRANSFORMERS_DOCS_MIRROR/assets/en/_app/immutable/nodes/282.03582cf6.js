import{s as ls,o as ds,n as z}from"../chunks/scheduler.18a86fab.js";import{S as cs,i as ps,g as c,s as r,r as g,A as ms,h as p,f as l,c as i,j,x as u,u as f,k as C,y as a,a as m,v as _,d as M,t as b,w as y}from"../chunks/index.98837b22.js";import{T as $e}from"../chunks/Tip.77304350.js";import{D as I}from"../chunks/Docstring.a1ef7999.js";import{C as re}from"../chunks/CodeBlock.8d0c2e8a.js";import{E as ae}from"../chunks/ExampleCodeBlock.8c3ee1f9.js";import{H as O,E as us}from"../chunks/getInferenceSnippets.06c2775f.js";function hs(v){let t,h=`This model is in maintenance mode only, we don’t accept any new PRs changing its code.
If you run into any issues running this model, please reinstall the last version that supported this model: v4.40.2.
You can do so by running the following command: <code>pip install -U transformers==4.40.2</code>.`;return{c(){t=c("p"),t.innerHTML=h},l(s){t=p(s,"P",{"data-svelte-h":!0}),u(t)!=="svelte-1sq0hrb"&&(t.innerHTML=h)},m(s,d){m(s,t,d)},p:z,d(s){s&&l(t)}}}function gs(v){let t,h="Examples:",s,d,T;return d=new re({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyME1lZ2FDb25maWclMkMlMjBNZWdhTW9kZWwlMEElMEElMjMlMjBJbml0aWFsaXppbmclMjBhJTIwTWVnYSUyMGNvbmZpZ3VyYXRpb24lMEFjb25maWd1cmF0aW9uJTIwJTNEJTIwTWVnYUNvbmZpZygpJTBBJTBBJTIzJTIwSW5pdGlhbGl6aW5nJTIwYSUyMG1vZGVsJTIwKHdpdGglMjByYW5kb20lMjB3ZWlnaHRzKSUyMGZyb20lMjB0aGUlMjBjb25maWd1cmF0aW9uJTBBbW9kZWwlMjAlM0QlMjBNZWdhTW9kZWwoY29uZmlndXJhdGlvbiklMEElMEElMjMlMjBBY2Nlc3NpbmclMjB0aGUlMjBtb2RlbCUyMGNvbmZpZ3VyYXRpb24lMEFjb25maWd1cmF0aW9uJTIwJTNEJTIwbW9kZWwuY29uZmln",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> MegaConfig, MegaModel

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a Mega configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = MegaConfig()

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a model (with random weights) from the configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = MegaModel(configuration)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Accessing the model configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = model.config`,wrap:!1}}),{c(){t=c("p"),t.textContent=h,s=r(),g(d.$$.fragment)},l(n){t=p(n,"P",{"data-svelte-h":!0}),u(t)!=="svelte-kvfsh7"&&(t.textContent=h),s=i(n),f(d.$$.fragment,n)},m(n,w){m(n,t,w),m(n,s,w),_(d,n,w),T=!0},p:z,i(n){T||(M(d.$$.fragment,n),T=!0)},o(n){b(d.$$.fragment,n),T=!1},d(n){n&&(l(t),l(s)),y(d,n)}}}function fs(v){let t,h=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=h},l(s){t=p(s,"P",{"data-svelte-h":!0}),u(t)!=="svelte-fincs2"&&(t.innerHTML=h)},m(s,d){m(s,t,d)},p:z,d(s){s&&l(t)}}}function _s(v){let t,h="Example:",s,d,T;return d=new re({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBNZWdhTW9kZWwlMEFpbXBvcnQlMjB0b3JjaCUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMm1uYXlsb3IlMkZtZWdhLWJhc2Utd2lraXRleHQlMjIpJTBBbW9kZWwlMjAlM0QlMjBNZWdhTW9kZWwuZnJvbV9wcmV0cmFpbmVkKCUyMm1uYXlsb3IlMkZtZWdhLWJhc2Utd2lraXRleHQlMjIpJTBBJTBBaW5wdXRzJTIwJTNEJTIwdG9rZW5pemVyKCUyMkhlbGxvJTJDJTIwbXklMjBkb2clMjBpcyUyMGN1dGUlMjIlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQW91dHB1dHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyklMEElMEFsYXN0X2hpZGRlbl9zdGF0ZXMlMjAlM0QlMjBvdXRwdXRzLmxhc3RfaGlkZGVuX3N0YXRl",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, MegaModel
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;mnaylor/mega-base-wikitext&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = MegaModel.from_pretrained(<span class="hljs-string">&quot;mnaylor/mega-base-wikitext&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs)

<span class="hljs-meta">&gt;&gt;&gt; </span>last_hidden_states = outputs.last_hidden_state`,wrap:!1}}),{c(){t=c("p"),t.textContent=h,s=r(),g(d.$$.fragment)},l(n){t=p(n,"P",{"data-svelte-h":!0}),u(t)!=="svelte-11lpom8"&&(t.textContent=h),s=i(n),f(d.$$.fragment,n)},m(n,w){m(n,t,w),m(n,s,w),_(d,n,w),T=!0},p:z,i(n){T||(M(d.$$.fragment,n),T=!0)},o(n){b(d.$$.fragment,n),T=!1},d(n){n&&(l(t),l(s)),y(d,n)}}}function Ms(v){let t,h=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=h},l(s){t=p(s,"P",{"data-svelte-h":!0}),u(t)!=="svelte-fincs2"&&(t.innerHTML=h)},m(s,d){m(s,t,d)},p:z,d(s){s&&l(t)}}}function bs(v){let t,h="Example:",s,d,T;return d=new re({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBNZWdhRm9yQ2F1c2FsTE0lMkMlMjBBdXRvQ29uZmlnJTBBaW1wb3J0JTIwdG9yY2glMEElMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJtbmF5bG9yJTJGbWVnYS1iYXNlLXdpa2l0ZXh0JTIyKSUwQWNvbmZpZyUyMCUzRCUyMEF1dG9Db25maWcuZnJvbV9wcmV0cmFpbmVkKCUyMm1uYXlsb3IlMkZtZWdhLWJhc2Utd2lraXRleHQlMjIpJTBBY29uZmlnLmlzX2RlY29kZXIlMjAlM0QlMjBUcnVlJTBBY29uZmlnLmJpZGlyZWN0aW9uYWwlMjAlM0QlMjBGYWxzZSUwQW1vZGVsJTIwJTNEJTIwTWVnYUZvckNhdXNhbExNLmZyb21fcHJldHJhaW5lZCglMEElMjAlMjAlMjAlMjAlMjJtbmF5bG9yJTJGbWVnYS1iYXNlLXdpa2l0ZXh0JTIyJTJDJTIwY29uZmlnJTNEY29uZmlnJTJDJTIwaWdub3JlX21pc21hdGNoZWRfc2l6ZXMlM0RUcnVlJTBBKSUwQSUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplciglMjJIZWxsbyUyQyUyMG15JTIwZG9nJTIwaXMlMjBjdXRlJTIyJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiklMEFvdXRwdXRzJTIwJTNEJTIwbW9kZWwoKippbnB1dHMpJTBBJTBBcHJlZGljdGlvbl9sb2dpdHMlMjAlM0QlMjBvdXRwdXRzLmxvZ2l0cw==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, MegaForCausalLM, AutoConfig
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;mnaylor/mega-base-wikitext&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>config = AutoConfig.from_pretrained(<span class="hljs-string">&quot;mnaylor/mega-base-wikitext&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>config.is_decoder = <span class="hljs-literal">True</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>config.bidirectional = <span class="hljs-literal">False</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = MegaForCausalLM.from_pretrained(
<span class="hljs-meta">... </span>    <span class="hljs-string">&quot;mnaylor/mega-base-wikitext&quot;</span>, config=config, ignore_mismatched_sizes=<span class="hljs-literal">True</span>
<span class="hljs-meta">... </span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs)

<span class="hljs-meta">&gt;&gt;&gt; </span>prediction_logits = outputs.logits`,wrap:!1}}),{c(){t=c("p"),t.textContent=h,s=r(),g(d.$$.fragment)},l(n){t=p(n,"P",{"data-svelte-h":!0}),u(t)!=="svelte-11lpom8"&&(t.textContent=h),s=i(n),f(d.$$.fragment,n)},m(n,w){m(n,t,w),m(n,s,w),_(d,n,w),T=!0},p:z,i(n){T||(M(d.$$.fragment,n),T=!0)},o(n){b(d.$$.fragment,n),T=!1},d(n){n&&(l(t),l(s)),y(d,n)}}}function ys(v){let t,h=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=h},l(s){t=p(s,"P",{"data-svelte-h":!0}),u(t)!=="svelte-fincs2"&&(t.innerHTML=h)},m(s,d){m(s,t,d)},p:z,d(s){s&&l(t)}}}function Ts(v){let t,h="Example:",s,d,T;return d=new re({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBNZWdhRm9yTWFza2VkTE0lMEFpbXBvcnQlMjB0b3JjaCUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMm1uYXlsb3IlMkZtZWdhLWJhc2Utd2lraXRleHQlMjIpJTBBbW9kZWwlMjAlM0QlMjBNZWdhRm9yTWFza2VkTE0uZnJvbV9wcmV0cmFpbmVkKCUyMm1uYXlsb3IlMkZtZWdhLWJhc2Utd2lraXRleHQlMjIpJTBBJTBBaW5wdXRzJTIwJTNEJTIwdG9rZW5pemVyKCUyMlRoZSUyMGNhcGl0YWwlMjBvZiUyMEZyYW5jZSUyMGlzJTIwJTNDbWFzayUzRS4lMjIlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQSUwQXdpdGglMjB0b3JjaC5ub19ncmFkKCklM0ElMEElMjAlMjAlMjAlMjBsb2dpdHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cykubG9naXRzJTBBJTBBJTIzJTIwcmV0cmlldmUlMjBpbmRleCUyMG9mJTIwJTNDbWFzayUzRSUwQW1hc2tfdG9rZW5faW5kZXglMjAlM0QlMjAoaW5wdXRzLmlucHV0X2lkcyUyMCUzRCUzRCUyMHRva2VuaXplci5tYXNrX3Rva2VuX2lkKSU1QjAlNUQubm9uemVybyhhc190dXBsZSUzRFRydWUpJTVCMCU1RCUwQSUwQXByZWRpY3RlZF90b2tlbl9pZCUyMCUzRCUyMGxvZ2l0cyU1QjAlMkMlMjBtYXNrX3Rva2VuX2luZGV4JTVELmFyZ21heChheGlzJTNELTEpJTBBdG9rZW5pemVyLmRlY29kZShwcmVkaWN0ZWRfdG9rZW5faWQpJTBBJTBBbGFiZWxzJTIwJTNEJTIwdG9rZW5pemVyKCUyMlRoZSUyMGNhcGl0YWwlMjBvZiUyMEZyYW5jZSUyMGlzJTIwUGFyaXMuJTIyJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiklNUIlMjJpbnB1dF9pZHMlMjIlNUQlMEElMjMlMjBtYXNrJTIwbGFiZWxzJTIwb2YlMjBub24tJTNDbWFzayUzRSUyMHRva2VucyUwQWxhYmVscyUyMCUzRCUyMHRvcmNoLndoZXJlKGlucHV0cy5pbnB1dF9pZHMlMjAlM0QlM0QlMjB0b2tlbml6ZXIubWFza190b2tlbl9pZCUyQyUyMGxhYmVscyUyQyUyMC0xMDApJTBBJTBBb3V0cHV0cyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzJTJDJTIwbGFiZWxzJTNEbGFiZWxzKSUwQXJvdW5kKG91dHB1dHMubG9zcy5pdGVtKCklMkMlMjAyKQ==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, MegaForMaskedLM
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;mnaylor/mega-base-wikitext&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = MegaForMaskedLM.from_pretrained(<span class="hljs-string">&quot;mnaylor/mega-base-wikitext&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;The capital of France is &lt;mask&gt;.&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    logits = model(**inputs).logits

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># retrieve index of &lt;mask&gt;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[<span class="hljs-number">0</span>].nonzero(as_tuple=<span class="hljs-literal">True</span>)[<span class="hljs-number">0</span>]

<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_token_id = logits[<span class="hljs-number">0</span>, mask_token_index].argmax(axis=-<span class="hljs-number">1</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer.decode(predicted_token_id)
<span class="hljs-string">&#x27; Paris&#x27;</span>

<span class="hljs-meta">&gt;&gt;&gt; </span>labels = tokenizer(<span class="hljs-string">&quot;The capital of France is Paris.&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)[<span class="hljs-string">&quot;input_ids&quot;</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># mask labels of non-&lt;mask&gt; tokens</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.where(inputs.input_ids == tokenizer.mask_token_id, labels, -<span class="hljs-number">100</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs, labels=labels)
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">round</span>(outputs.loss.item(), <span class="hljs-number">2</span>)
<span class="hljs-number">0.1</span>`,wrap:!1}}),{c(){t=c("p"),t.textContent=h,s=r(),g(d.$$.fragment)},l(n){t=p(n,"P",{"data-svelte-h":!0}),u(t)!=="svelte-11lpom8"&&(t.textContent=h),s=i(n),f(d.$$.fragment,n)},m(n,w){m(n,t,w),m(n,s,w),_(d,n,w),T=!0},p:z,i(n){T||(M(d.$$.fragment,n),T=!0)},o(n){b(d.$$.fragment,n),T=!1},d(n){n&&(l(t),l(s)),y(d,n)}}}function ws(v){let t,h=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=h},l(s){t=p(s,"P",{"data-svelte-h":!0}),u(t)!=="svelte-fincs2"&&(t.innerHTML=h)},m(s,d){m(s,t,d)},p:z,d(s){s&&l(t)}}}function vs(v){let t,h="Example of single-label classification:",s,d,T;return d=new re({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Rva2VuaXplciUyQyUyME1lZ2FGb3JTZXF1ZW5jZUNsYXNzaWZpY2F0aW9uJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIybW5heWxvciUyRm1lZ2EtYmFzZS13aWtpdGV4dCUyMiklMEFtb2RlbCUyMCUzRCUyME1lZ2FGb3JTZXF1ZW5jZUNsYXNzaWZpY2F0aW9uLmZyb21fcHJldHJhaW5lZCglMjJtbmF5bG9yJTJGbWVnYS1iYXNlLXdpa2l0ZXh0JTIyKSUwQSUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplciglMjJIZWxsbyUyQyUyMG15JTIwZG9nJTIwaXMlMjBjdXRlJTIyJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiklMEElMEF3aXRoJTIwdG9yY2gubm9fZ3JhZCgpJTNBJTBBJTIwJTIwJTIwJTIwbG9naXRzJTIwJTNEJTIwbW9kZWwoKippbnB1dHMpLmxvZ2l0cyUwQSUwQXByZWRpY3RlZF9jbGFzc19pZCUyMCUzRCUyMGxvZ2l0cy5hcmdtYXgoKS5pdGVtKCklMEElMEElMjMlMjBUbyUyMHRyYWluJTIwYSUyMG1vZGVsJTIwb24lMjAlNjBudW1fbGFiZWxzJTYwJTIwY2xhc3NlcyUyQyUyMHlvdSUyMGNhbiUyMHBhc3MlMjAlNjBudW1fbGFiZWxzJTNEbnVtX2xhYmVscyU2MCUyMHRvJTIwJTYwLmZyb21fcHJldHJhaW5lZCguLi4pJTYwJTBBbnVtX2xhYmVscyUyMCUzRCUyMGxlbihtb2RlbC5jb25maWcuaWQybGFiZWwpJTBBbW9kZWwlMjAlM0QlMjBNZWdhRm9yU2VxdWVuY2VDbGFzc2lmaWNhdGlvbi5mcm9tX3ByZXRyYWluZWQoJTIybW5heWxvciUyRm1lZ2EtYmFzZS13aWtpdGV4dCUyMiUyQyUyMG51bV9sYWJlbHMlM0RudW1fbGFiZWxzKSUwQSUwQWxhYmVscyUyMCUzRCUyMHRvcmNoLnRlbnNvciglNUIxJTVEKSUwQWxvc3MlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyUyQyUyMGxhYmVscyUzRGxhYmVscykubG9zcw==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, MegaForSequenceClassification

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;mnaylor/mega-base-wikitext&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = MegaForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;mnaylor/mega-base-wikitext&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    logits = model(**inputs).logits

<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_class_id = logits.argmax().item()

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># To train a model on \`num_labels\` classes, you can pass \`num_labels=num_labels\` to \`.from_pretrained(...)\`</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>num_labels = <span class="hljs-built_in">len</span>(model.config.id2label)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = MegaForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;mnaylor/mega-base-wikitext&quot;</span>, num_labels=num_labels)

<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.tensor([<span class="hljs-number">1</span>])
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = model(**inputs, labels=labels).loss`,wrap:!1}}),{c(){t=c("p"),t.textContent=h,s=r(),g(d.$$.fragment)},l(n){t=p(n,"P",{"data-svelte-h":!0}),u(t)!=="svelte-ykxpe4"&&(t.textContent=h),s=i(n),f(d.$$.fragment,n)},m(n,w){m(n,t,w),m(n,s,w),_(d,n,w),T=!0},p:z,i(n){T||(M(d.$$.fragment,n),T=!0)},o(n){b(d.$$.fragment,n),T=!1},d(n){n&&(l(t),l(s)),y(d,n)}}}function ks(v){let t,h="Example of multi-label classification:",s,d,T;return d=new re({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Rva2VuaXplciUyQyUyME1lZ2FGb3JTZXF1ZW5jZUNsYXNzaWZpY2F0aW9uJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIybW5heWxvciUyRm1lZ2EtYmFzZS13aWtpdGV4dCUyMiklMEFtb2RlbCUyMCUzRCUyME1lZ2FGb3JTZXF1ZW5jZUNsYXNzaWZpY2F0aW9uLmZyb21fcHJldHJhaW5lZCglMjJtbmF5bG9yJTJGbWVnYS1iYXNlLXdpa2l0ZXh0JTIyJTJDJTIwcHJvYmxlbV90eXBlJTNEJTIybXVsdGlfbGFiZWxfY2xhc3NpZmljYXRpb24lMjIpJTBBJTBBaW5wdXRzJTIwJTNEJTIwdG9rZW5pemVyKCUyMkhlbGxvJTJDJTIwbXklMjBkb2clMjBpcyUyMGN1dGUlMjIlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQSUwQXdpdGglMjB0b3JjaC5ub19ncmFkKCklM0ElMEElMjAlMjAlMjAlMjBsb2dpdHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cykubG9naXRzJTBBJTBBcHJlZGljdGVkX2NsYXNzX2lkcyUyMCUzRCUyMHRvcmNoLmFyYW5nZSgwJTJDJTIwbG9naXRzLnNoYXBlJTVCLTElNUQpJTVCdG9yY2guc2lnbW9pZChsb2dpdHMpLnNxdWVlemUoZGltJTNEMCklMjAlM0UlMjAwLjUlNUQlMEElMEElMjMlMjBUbyUyMHRyYWluJTIwYSUyMG1vZGVsJTIwb24lMjAlNjBudW1fbGFiZWxzJTYwJTIwY2xhc3NlcyUyQyUyMHlvdSUyMGNhbiUyMHBhc3MlMjAlNjBudW1fbGFiZWxzJTNEbnVtX2xhYmVscyU2MCUyMHRvJTIwJTYwLmZyb21fcHJldHJhaW5lZCguLi4pJTYwJTBBbnVtX2xhYmVscyUyMCUzRCUyMGxlbihtb2RlbC5jb25maWcuaWQybGFiZWwpJTBBbW9kZWwlMjAlM0QlMjBNZWdhRm9yU2VxdWVuY2VDbGFzc2lmaWNhdGlvbi5mcm9tX3ByZXRyYWluZWQoJTBBJTIwJTIwJTIwJTIwJTIybW5heWxvciUyRm1lZ2EtYmFzZS13aWtpdGV4dCUyMiUyQyUyMG51bV9sYWJlbHMlM0RudW1fbGFiZWxzJTJDJTIwcHJvYmxlbV90eXBlJTNEJTIybXVsdGlfbGFiZWxfY2xhc3NpZmljYXRpb24lMjIlMEEpJTBBJTBBbGFiZWxzJTIwJTNEJTIwdG9yY2guc3VtKCUwQSUyMCUyMCUyMCUyMHRvcmNoLm5uLmZ1bmN0aW9uYWwub25lX2hvdChwcmVkaWN0ZWRfY2xhc3NfaWRzJTVCTm9uZSUyQyUyMCUzQSU1RC5jbG9uZSgpJTJDJTIwbnVtX2NsYXNzZXMlM0RudW1fbGFiZWxzKSUyQyUyMGRpbSUzRDElMEEpLnRvKHRvcmNoLmZsb2F0KSUwQWxvc3MlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyUyQyUyMGxhYmVscyUzRGxhYmVscykubG9zcw==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, MegaForSequenceClassification

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;mnaylor/mega-base-wikitext&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = MegaForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;mnaylor/mega-base-wikitext&quot;</span>, problem_type=<span class="hljs-string">&quot;multi_label_classification&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    logits = model(**inputs).logits

<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_class_ids = torch.arange(<span class="hljs-number">0</span>, logits.shape[-<span class="hljs-number">1</span>])[torch.sigmoid(logits).squeeze(dim=<span class="hljs-number">0</span>) &gt; <span class="hljs-number">0.5</span>]

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># To train a model on \`num_labels\` classes, you can pass \`num_labels=num_labels\` to \`.from_pretrained(...)\`</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>num_labels = <span class="hljs-built_in">len</span>(model.config.id2label)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = MegaForSequenceClassification.from_pretrained(
<span class="hljs-meta">... </span>    <span class="hljs-string">&quot;mnaylor/mega-base-wikitext&quot;</span>, num_labels=num_labels, problem_type=<span class="hljs-string">&quot;multi_label_classification&quot;</span>
<span class="hljs-meta">... </span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.<span class="hljs-built_in">sum</span>(
<span class="hljs-meta">... </span>    torch.nn.functional.one_hot(predicted_class_ids[<span class="hljs-literal">None</span>, :].clone(), num_classes=num_labels), dim=<span class="hljs-number">1</span>
<span class="hljs-meta">... </span>).to(torch.<span class="hljs-built_in">float</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = model(**inputs, labels=labels).loss`,wrap:!1}}),{c(){t=c("p"),t.textContent=h,s=r(),g(d.$$.fragment)},l(n){t=p(n,"P",{"data-svelte-h":!0}),u(t)!=="svelte-1l8e32d"&&(t.textContent=h),s=i(n),f(d.$$.fragment,n)},m(n,w){m(n,t,w),m(n,s,w),_(d,n,w),T=!0},p:z,i(n){T||(M(d.$$.fragment,n),T=!0)},o(n){b(d.$$.fragment,n),T=!1},d(n){n&&(l(t),l(s)),y(d,n)}}}function $s(v){let t,h=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=h},l(s){t=p(s,"P",{"data-svelte-h":!0}),u(t)!=="svelte-fincs2"&&(t.innerHTML=h)},m(s,d){m(s,t,d)},p:z,d(s){s&&l(t)}}}function Cs(v){let t,h="Example:",s,d,T;return d=new re({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBNZWdhRm9yTXVsdGlwbGVDaG9pY2UlMEFpbXBvcnQlMjB0b3JjaCUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMm1uYXlsb3IlMkZtZWdhLWJhc2Utd2lraXRleHQlMjIpJTBBbW9kZWwlMjAlM0QlMjBNZWdhRm9yTXVsdGlwbGVDaG9pY2UuZnJvbV9wcmV0cmFpbmVkKCUyMm1uYXlsb3IlMkZtZWdhLWJhc2Utd2lraXRleHQlMjIpJTBBJTBBcHJvbXB0JTIwJTNEJTIwJTIySW4lMjBJdGFseSUyQyUyMHBpenphJTIwc2VydmVkJTIwaW4lMjBmb3JtYWwlMjBzZXR0aW5ncyUyQyUyMHN1Y2glMjBhcyUyMGF0JTIwYSUyMHJlc3RhdXJhbnQlMkMlMjBpcyUyMHByZXNlbnRlZCUyMHVuc2xpY2VkLiUyMiUwQWNob2ljZTAlMjAlM0QlMjAlMjJJdCUyMGlzJTIwZWF0ZW4lMjB3aXRoJTIwYSUyMGZvcmslMjBhbmQlMjBhJTIwa25pZmUuJTIyJTBBY2hvaWNlMSUyMCUzRCUyMCUyMkl0JTIwaXMlMjBlYXRlbiUyMHdoaWxlJTIwaGVsZCUyMGluJTIwdGhlJTIwaGFuZC4lMjIlMEFsYWJlbHMlMjAlM0QlMjB0b3JjaC50ZW5zb3IoMCkudW5zcXVlZXplKDApJTIwJTIwJTIzJTIwY2hvaWNlMCUyMGlzJTIwY29ycmVjdCUyMChhY2NvcmRpbmclMjB0byUyMFdpa2lwZWRpYSUyMCUzQikpJTJDJTIwYmF0Y2glMjBzaXplJTIwMSUwQSUwQWVuY29kaW5nJTIwJTNEJTIwdG9rZW5pemVyKCU1QnByb21wdCUyQyUyMHByb21wdCU1RCUyQyUyMCU1QmNob2ljZTAlMkMlMjBjaG9pY2UxJTVEJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiUyQyUyMHBhZGRpbmclM0RUcnVlKSUwQW91dHB1dHMlMjAlM0QlMjBtb2RlbCgqKiU3QmslM0ElMjB2LnVuc3F1ZWV6ZSgwKSUyMGZvciUyMGslMkMlMjB2JTIwaW4lMjBlbmNvZGluZy5pdGVtcygpJTdEJTJDJTIwbGFiZWxzJTNEbGFiZWxzKSUyMCUyMCUyMyUyMGJhdGNoJTIwc2l6ZSUyMGlzJTIwMSUwQSUwQSUyMyUyMHRoZSUyMGxpbmVhciUyMGNsYXNzaWZpZXIlMjBzdGlsbCUyMG5lZWRzJTIwdG8lMjBiZSUyMHRyYWluZWQlMEFsb3NzJTIwJTNEJTIwb3V0cHV0cy5sb3NzJTBBbG9naXRzJTIwJTNEJTIwb3V0cHV0cy5sb2dpdHM=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, MegaForMultipleChoice
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;mnaylor/mega-base-wikitext&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = MegaForMultipleChoice.from_pretrained(<span class="hljs-string">&quot;mnaylor/mega-base-wikitext&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>prompt = <span class="hljs-string">&quot;In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>choice0 = <span class="hljs-string">&quot;It is eaten with a fork and a knife.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>choice1 = <span class="hljs-string">&quot;It is eaten while held in the hand.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.tensor(<span class="hljs-number">0</span>).unsqueeze(<span class="hljs-number">0</span>)  <span class="hljs-comment"># choice0 is correct (according to Wikipedia ;)), batch size 1</span>

<span class="hljs-meta">&gt;&gt;&gt; </span>encoding = tokenizer([prompt, prompt], [choice0, choice1], return_tensors=<span class="hljs-string">&quot;pt&quot;</span>, padding=<span class="hljs-literal">True</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**{k: v.unsqueeze(<span class="hljs-number">0</span>) <span class="hljs-keyword">for</span> k, v <span class="hljs-keyword">in</span> encoding.items()}, labels=labels)  <span class="hljs-comment"># batch size is 1</span>

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># the linear classifier still needs to be trained</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = outputs.loss
<span class="hljs-meta">&gt;&gt;&gt; </span>logits = outputs.logits`,wrap:!1}}),{c(){t=c("p"),t.textContent=h,s=r(),g(d.$$.fragment)},l(n){t=p(n,"P",{"data-svelte-h":!0}),u(t)!=="svelte-11lpom8"&&(t.textContent=h),s=i(n),f(d.$$.fragment,n)},m(n,w){m(n,t,w),m(n,s,w),_(d,n,w),T=!0},p:z,i(n){T||(M(d.$$.fragment,n),T=!0)},o(n){b(d.$$.fragment,n),T=!1},d(n){n&&(l(t),l(s)),y(d,n)}}}function js(v){let t,h=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=h},l(s){t=p(s,"P",{"data-svelte-h":!0}),u(t)!=="svelte-fincs2"&&(t.innerHTML=h)},m(s,d){m(s,t,d)},p:z,d(s){s&&l(t)}}}function zs(v){let t,h="Example:",s,d,T;return d=new re({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBNZWdhRm9yVG9rZW5DbGFzc2lmaWNhdGlvbiUwQWltcG9ydCUyMHRvcmNoJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIybW5heWxvciUyRm1lZ2EtYmFzZS13aWtpdGV4dCUyMiklMEFtb2RlbCUyMCUzRCUyME1lZ2FGb3JUb2tlbkNsYXNzaWZpY2F0aW9uLmZyb21fcHJldHJhaW5lZCglMjJtbmF5bG9yJTJGbWVnYS1iYXNlLXdpa2l0ZXh0JTIyKSUwQSUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplciglMEElMjAlMjAlMjAlMjAlMjJIdWdnaW5nRmFjZSUyMGlzJTIwYSUyMGNvbXBhbnklMjBiYXNlZCUyMGluJTIwUGFyaXMlMjBhbmQlMjBOZXclMjBZb3JrJTIyJTJDJTIwYWRkX3NwZWNpYWxfdG9rZW5zJTNERmFsc2UlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyJTBBKSUwQSUwQXdpdGglMjB0b3JjaC5ub19ncmFkKCklM0ElMEElMjAlMjAlMjAlMjBsb2dpdHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cykubG9naXRzJTBBJTBBcHJlZGljdGVkX3Rva2VuX2NsYXNzX2lkcyUyMCUzRCUyMGxvZ2l0cy5hcmdtYXgoLTEpJTBBJTBBJTIzJTIwTm90ZSUyMHRoYXQlMjB0b2tlbnMlMjBhcmUlMjBjbGFzc2lmaWVkJTIwcmF0aGVyJTIwdGhlbiUyMGlucHV0JTIwd29yZHMlMjB3aGljaCUyMG1lYW5zJTIwdGhhdCUwQSUyMyUyMHRoZXJlJTIwbWlnaHQlMjBiZSUyMG1vcmUlMjBwcmVkaWN0ZWQlMjB0b2tlbiUyMGNsYXNzZXMlMjB0aGFuJTIwd29yZHMuJTBBJTIzJTIwTXVsdGlwbGUlMjB0b2tlbiUyMGNsYXNzZXMlMjBtaWdodCUyMGFjY291bnQlMjBmb3IlMjB0aGUlMjBzYW1lJTIwd29yZCUwQXByZWRpY3RlZF90b2tlbnNfY2xhc3NlcyUyMCUzRCUyMCU1Qm1vZGVsLmNvbmZpZy5pZDJsYWJlbCU1QnQuaXRlbSgpJTVEJTIwZm9yJTIwdCUyMGluJTIwcHJlZGljdGVkX3Rva2VuX2NsYXNzX2lkcyU1QjAlNUQlNUQlMEElMEFsYWJlbHMlMjAlM0QlMjBwcmVkaWN0ZWRfdG9rZW5fY2xhc3NfaWRzJTBBbG9zcyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzJTJDJTIwbGFiZWxzJTNEbGFiZWxzKS5sb3Nz",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, MegaForTokenClassification
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;mnaylor/mega-base-wikitext&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = MegaForTokenClassification.from_pretrained(<span class="hljs-string">&quot;mnaylor/mega-base-wikitext&quot;</span>)

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
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = model(**inputs, labels=labels).loss`,wrap:!1}}),{c(){t=c("p"),t.textContent=h,s=r(),g(d.$$.fragment)},l(n){t=p(n,"P",{"data-svelte-h":!0}),u(t)!=="svelte-11lpom8"&&(t.textContent=h),s=i(n),f(d.$$.fragment,n)},m(n,w){m(n,t,w),m(n,s,w),_(d,n,w),T=!0},p:z,i(n){T||(M(d.$$.fragment,n),T=!0)},o(n){b(d.$$.fragment,n),T=!1},d(n){n&&(l(t),l(s)),y(d,n)}}}function Js(v){let t,h=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=h},l(s){t=p(s,"P",{"data-svelte-h":!0}),u(t)!=="svelte-fincs2"&&(t.innerHTML=h)},m(s,d){m(s,t,d)},p:z,d(s){s&&l(t)}}}function xs(v){let t,h="Example:",s,d,T;return d=new re({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBNZWdhRm9yUXVlc3Rpb25BbnN3ZXJpbmclMEFpbXBvcnQlMjB0b3JjaCUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMm1uYXlsb3IlMkZtZWdhLWJhc2Utd2lraXRleHQlMjIpJTBBbW9kZWwlMjAlM0QlMjBNZWdhRm9yUXVlc3Rpb25BbnN3ZXJpbmcuZnJvbV9wcmV0cmFpbmVkKCUyMm1uYXlsb3IlMkZtZWdhLWJhc2Utd2lraXRleHQlMjIpJTBBJTBBcXVlc3Rpb24lMkMlMjB0ZXh0JTIwJTNEJTIwJTIyV2hvJTIwd2FzJTIwSmltJTIwSGVuc29uJTNGJTIyJTJDJTIwJTIySmltJTIwSGVuc29uJTIwd2FzJTIwYSUyMG5pY2UlMjBwdXBwZXQlMjIlMEElMEFpbnB1dHMlMjAlM0QlMjB0b2tlbml6ZXIocXVlc3Rpb24lMkMlMjB0ZXh0JTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiklMEF3aXRoJTIwdG9yY2gubm9fZ3JhZCgpJTNBJTBBJTIwJTIwJTIwJTIwb3V0cHV0cyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzKSUwQSUwQWFuc3dlcl9zdGFydF9pbmRleCUyMCUzRCUyMG91dHB1dHMuc3RhcnRfbG9naXRzLmFyZ21heCgpJTBBYW5zd2VyX2VuZF9pbmRleCUyMCUzRCUyMG91dHB1dHMuZW5kX2xvZ2l0cy5hcmdtYXgoKSUwQSUwQXByZWRpY3RfYW5zd2VyX3Rva2VucyUyMCUzRCUyMGlucHV0cy5pbnB1dF9pZHMlNUIwJTJDJTIwYW5zd2VyX3N0YXJ0X2luZGV4JTIwJTNBJTIwYW5zd2VyX2VuZF9pbmRleCUyMCUyQiUyMDElNUQlMEElMEElMjMlMjB0YXJnZXQlMjBpcyUyMCUyMm5pY2UlMjBwdXBwZXQlMjIlMEF0YXJnZXRfc3RhcnRfaW5kZXglMjAlM0QlMjB0b3JjaC50ZW5zb3IoJTVCMTQlNUQpJTBBdGFyZ2V0X2VuZF9pbmRleCUyMCUzRCUyMHRvcmNoLnRlbnNvciglNUIxNSU1RCklMEElMEFvdXRwdXRzJTIwJTNEJTIwbW9kZWwoKippbnB1dHMlMkMlMjBzdGFydF9wb3NpdGlvbnMlM0R0YXJnZXRfc3RhcnRfaW5kZXglMkMlMjBlbmRfcG9zaXRpb25zJTNEdGFyZ2V0X2VuZF9pbmRleCklMEFsb3NzJTIwJTNEJTIwb3V0cHV0cy5sb3Nz",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, MegaForQuestionAnswering
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;mnaylor/mega-base-wikitext&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = MegaForQuestionAnswering.from_pretrained(<span class="hljs-string">&quot;mnaylor/mega-base-wikitext&quot;</span>)

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
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = outputs.loss`,wrap:!1}}),{c(){t=c("p"),t.textContent=h,s=r(),g(d.$$.fragment)},l(n){t=p(n,"P",{"data-svelte-h":!0}),u(t)!=="svelte-11lpom8"&&(t.textContent=h),s=i(n),f(d.$$.fragment,n)},m(n,w){m(n,t,w),m(n,s,w),_(d,n,w),T=!0},p:z,i(n){T||(M(d.$$.fragment,n),T=!0)},o(n){b(d.$$.fragment,n),T=!1},d(n){n&&(l(t),l(s)),y(d,n)}}}function Fs(v){let t,h,s,d,T,n="<em>This model was released on 2022-09-21 and added to Hugging Face Transformers on 2023-06-20.</em>",w,Ce,Nt,ie,yo='<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/>',Xt,le,Rt,je,Vt,ze,To=`The MEGA model was proposed in <a href="https://huggingface.co/papers/2209.10655" rel="nofollow">Mega: Moving Average Equipped Gated Attention</a> by Xuezhe Ma, Chunting Zhou, Xiang Kong, Junxian He, Liangke Gui, Graham Neubig, Jonathan May, and Luke Zettlemoyer.
MEGA proposes a new approach to self-attention with each encoder layer having a multi-headed exponential moving average in addition to a single head of standard dot-product attention, giving the attention mechanism
stronger positional biases. This allows MEGA to perform competitively to Transformers on standard benchmarks including LRA
while also having significantly fewer parameters. MEGA’s compute efficiency allows it to scale to very long sequences, making it an
attractive option for long-document NLP tasks.`,Ht,Je,wo="The abstract from the paper is the following:",Et,xe,vo="<em>The design choices in the Transformer attention mechanism, including weak inductive bias and quadratic computational complexity, have limited its application for modeling long sequences. In this paper, we introduce Mega, a simple, theoretically grounded, single-head gated attention mechanism equipped with (exponential) moving average to incorporate inductive bias of position-aware local dependencies into the position-agnostic attention mechanism. We further propose a variant of Mega that offers linear time and space complexity yet yields only minimal quality loss, by efficiently splitting the whole sequence into multiple chunks with fixed length. Extensive experiments on a wide range of sequence modeling benchmarks, including the Long Range Arena, neural machine translation, auto-regressive language modeling, and image and speech classification, show that Mega achieves significant improvements over other sequence models, including variants of Transformers and recent state space models.</em>",At,Fe,ko=`This model was contributed by <a href="https://huggingface.co/mnaylor" rel="nofollow">mnaylor</a>.
The original code can be found <a href="https://github.com/facebookresearch/mega" rel="nofollow">here</a>.`,Qt,Ue,St,We,$o="<li>MEGA can perform quite well with relatively few parameters. See Appendix D in the MEGA paper for examples of architectural specs which perform well in various settings. If using MEGA as a decoder, be sure to set <code>bidirectional=False</code> to avoid errors with default bidirectional.</li> <li>Mega-chunk is a variant of mega that reduces time and spaces complexity from quadratic to linear. Utilize chunking with MegaConfig.use_chunking and control chunk size with MegaConfig.chunk_size</li>",Yt,Ze,Pt,Ie,Co="<li>The original implementation of MEGA had an inconsistent expectation of attention masks for padding and causal self-attention between the softmax attention and Laplace/squared ReLU method. This implementation addresses that inconsistency.</li> <li>The original implementation did not include token type embeddings; this implementation adds support for these, with the option controlled by MegaConfig.add_token_type_embeddings</li>",Ot,Be,Dt,q,qe,_n,rt,jo=`This is the configuration class to store the configuration of a <a href="/docs/transformers/v4.56.2/en/model_doc/mega#transformers.MegaModel">MegaModel</a>. It is used to instantiate a Mega
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the Mega
<a href="https://huggingface.co/mnaylor/mega-base-wikitext" rel="nofollow">mnaylor/mega-base-wikitext</a> architecture.`,Mn,it,zo=`Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> and can be used to control the model outputs. Read the
documentation from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> for more information.`,bn,de,Kt,Ge,en,k,Le,yn,lt,Jo="The bare MEGA Model transformer outputting raw hidden-states without any specific head on top.",Tn,dt,xo=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,wn,ct,Fo=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,vn,pt,Uo=`The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
cross-attention is added after self-attention, following the architecture described in <em>Mega: Moving Average
Equipped Gated Attention</em>_ by Xuezhe Ma, Chunting Zhou, Xiang Kong, Junxian He, Liangke Gui, Graham Neubig,
Jonathan May, and Luke Zettlemoyer`,kn,mt,Wo=`To behave as a decoder the model needs to be initialized with the <code>is_decoder</code> argument of the configuration set to
<code>True</code> and <code>bidirectional</code> set to <code>False</code>. To be used in a Seq2Seq model, the model needs to initialized with both
<code>is_decoder=True</code> and <code>bidirectional=False</code> argument as well as <code>add_cross_attention</code> set to <code>True</code>; an
<code>encoder_hidden_states</code> is then expected as an input to the forward pass.`,$n,ut,Zo='.. _<em>Mega: Moving Average Equipped Gated Attention</em>: <a href="https://huggingface.co/papers/2209.10655" rel="nofollow">https://huggingface.co/papers/2209.10655</a>',Cn,E,Ne,jn,ht,Io='The <a href="/docs/transformers/v4.56.2/en/model_doc/mega#transformers.MegaModel">MegaModel</a> forward method, overrides the <code>__call__</code> special method.',zn,ce,Jn,pe,tn,Xe,nn,J,Re,xn,gt,Bo="MEGA Model with a <code>language modeling</code> head on top for CLM fine-tuning.",Fn,ft,qo=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Un,_t,Go=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Wn,A,Ve,Zn,Mt,Lo='The <a href="/docs/transformers/v4.56.2/en/model_doc/mega#transformers.MegaForCausalLM">MegaForCausalLM</a> forward method, overrides the <code>__call__</code> special method.',In,me,Bn,ue,on,He,sn,x,Ee,qn,bt,No="MEGA Model with a <code>language modeling</code> head on top.",Gn,yt,Xo=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Ln,Tt,Ro=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Nn,Q,Ae,Xn,wt,Vo='The <a href="/docs/transformers/v4.56.2/en/model_doc/mega#transformers.MegaForMaskedLM">MegaForMaskedLM</a> forward method, overrides the <code>__call__</code> special method.',Rn,he,Vn,ge,an,Qe,rn,F,Se,Hn,vt,Ho=`MEGA Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled
output) e.g. for GLUE tasks.`,En,kt,Eo=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,An,$t,Ao=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Qn,B,Ye,Sn,Ct,Qo='The <a href="/docs/transformers/v4.56.2/en/model_doc/mega#transformers.MegaForSequenceClassification">MegaForSequenceClassification</a> forward method, overrides the <code>__call__</code> special method.',Yn,fe,Pn,_e,On,Me,ln,Pe,dn,U,Oe,Dn,jt,So=`MEGA Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
softmax) e.g. for RocStories/SWAG tasks.`,Kn,zt,Yo=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,eo,Jt,Po=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,to,S,De,no,xt,Oo='The <a href="/docs/transformers/v4.56.2/en/model_doc/mega#transformers.MegaForMultipleChoice">MegaForMultipleChoice</a> forward method, overrides the <code>__call__</code> special method.',oo,be,so,ye,cn,Ke,pn,W,et,ao,Ft,Do=`MEGA Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
Named-Entity-Recognition (NER) tasks.`,ro,Ut,Ko=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,io,Wt,es=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,lo,Y,tt,co,Zt,ts='The <a href="/docs/transformers/v4.56.2/en/model_doc/mega#transformers.MegaForTokenClassification">MegaForTokenClassification</a> forward method, overrides the <code>__call__</code> special method.',po,Te,mo,we,mn,nt,un,Z,ot,uo,It,ns=`MEGA Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
layers on top of the hidden-states output to compute <code>span start logits</code> and <code>span end logits</code>).`,ho,Bt,os=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,go,qt,ss=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,fo,P,st,_o,Gt,as='The <a href="/docs/transformers/v4.56.2/en/model_doc/mega#transformers.MegaForQuestionAnswering">MegaForQuestionAnswering</a> forward method, overrides the <code>__call__</code> special method.',Mo,ve,bo,ke,hn,at,gn,Lt,fn;return Ce=new O({props:{title:"MEGA",local:"mega",headingTag:"h1"}}),le=new $e({props:{warning:!0,$$slots:{default:[hs]},$$scope:{ctx:v}}}),je=new O({props:{title:"Overview",local:"overview",headingTag:"h2"}}),Ue=new O({props:{title:"Usage tips",local:"usage-tips",headingTag:"h2"}}),Ze=new O({props:{title:"Implementation Notes",local:"implementation-notes",headingTag:"h2"}}),Be=new O({props:{title:"MegaConfig",local:"transformers.MegaConfig",headingTag:"h2"}}),qe=new I({props:{name:"class transformers.MegaConfig",anchor:"transformers.MegaConfig",parameters:[{name:"vocab_size",val:" = 30522"},{name:"hidden_size",val:" = 128"},{name:"num_hidden_layers",val:" = 4"},{name:"intermediate_size",val:" = 256"},{name:"ema_projection_size",val:" = 16"},{name:"bidirectional",val:" = True"},{name:"shared_representation_size",val:" = 64"},{name:"use_chunking",val:" = False"},{name:"chunk_size",val:" = -1"},{name:"truncation",val:" = None"},{name:"normalize_before_mega",val:" = True"},{name:"normalization_type",val:" = 'scalenorm'"},{name:"norm_affine",val:" = True"},{name:"activation",val:" = 'silu'"},{name:"attention_activation",val:" = 'softmax'"},{name:"dropout_prob",val:" = 0.1"},{name:"hidden_dropout_prob",val:" = 0.1"},{name:"attention_probs_dropout_prob",val:" = 0.1"},{name:"use_feature_dropout",val:" = False"},{name:"use_normalized_ffn",val:" = True"},{name:"nffn_hidden_size",val:" = 256"},{name:"normalize_before_ffn",val:" = True"},{name:"nffn_activation_dropout_prob",val:" = 0.1"},{name:"max_positions",val:" = 2048"},{name:"add_token_type_embeddings",val:" = False"},{name:"type_vocab_size",val:" = 2"},{name:"initializer_range",val:" = 0.02"},{name:"ema_delta_alpha_range",val:" = 0.2"},{name:"ema_beta_range",val:" = 0.02"},{name:"ema_gamma_omega_range",val:" = 1.0"},{name:"pad_token_id",val:" = 1"},{name:"bos_token_id",val:" = 0"},{name:"eos_token_id",val:" = 2"},{name:"relative_positional_bias",val:" = 'rotary'"},{name:"classifier_dropout",val:" = None"},{name:"use_cache",val:" = True"},{name:"add_lm_hidden_dense_layer",val:" = True"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.MegaConfig.vocab_size",description:`<strong>vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 30522) &#x2014;
Vocabulary size of the Mega model. Defines the number of different tokens that can be represented by the
<code>inputs_ids</code> passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/mega#transformers.MegaModel">MegaModel</a>.`,name:"vocab_size"},{anchor:"transformers.MegaConfig.hidden_size",description:`<strong>hidden_size</strong> (<code>int</code>, <em>optional</em>, defaults to 128) &#x2014;
Dimensionality of the encoder layers and the pooler layer.`,name:"hidden_size"},{anchor:"transformers.MegaConfig.num_hidden_layers",description:`<strong>num_hidden_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 4) &#x2014;
Number of hidden layers in the Mega encoder.`,name:"num_hidden_layers"},{anchor:"transformers.MegaConfig.intermediate_size",description:`<strong>intermediate_size</strong> (<code>int</code>, <em>optional</em>, defaults to 256) &#x2014;
Dimensionality of the hidden size (self-attention value projection) within the Mega encoder`,name:"intermediate_size"},{anchor:"transformers.MegaConfig.ema_projection_size",description:`<strong>ema_projection_size</strong> (<code>int</code>, <em>optional</em>, defaults to 16) &#x2014;
Dimensionality of the MegaMultiDimensionDampedEma`,name:"ema_projection_size"},{anchor:"transformers.MegaConfig.bidirectional",description:`<strong>bidirectional</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether the MegaMultiDimensionDampedEma used in Mega&#x2019;s self-attention should work bidirectionally (<code>True</code>)
or unidirectionally (<code>False</code>). Bidirectional EMA is incompatible with causal decoding, so this should be
False if you intend to use the model as a decoder.`,name:"bidirectional"},{anchor:"transformers.MegaConfig.shared_representation_size",description:`<strong>shared_representation_size</strong> (<code>int</code>, <em>optional</em>, defaults to 64) &#x2014;
Dimensionality of the linear projection for shared representation of self-attention queries and keys`,name:"shared_representation_size"},{anchor:"transformers.MegaConfig.use_chunking",description:`<strong>use_chunking</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether to chunk inputs for linear self-attention complexity (described as Mega-chunk in the paper)`,name:"use_chunking"},{anchor:"transformers.MegaConfig.chunk_size",description:`<strong>chunk_size</strong> (<code>int</code>, <em>optional</em>, defaults to -1) &#x2014;
If <code>use_chunking</code> is set to <code>True</code>, determines the size of the chunks to apply to the input sequence. If
chunking is used, input sequences must be padded to a multiple of <code>chunk_size</code>`,name:"chunk_size"},{anchor:"transformers.MegaConfig.truncation",description:`<strong>truncation</strong> (<code>int</code>, <em>optional</em>) &#x2014;
If specified, the sequence length for which to truncate MegaMultiDimensionDampedEma`,name:"truncation"},{anchor:"transformers.MegaConfig.normalize_before_mega",description:`<strong>normalize_before_mega</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether to normalize before (<code>True</code>) or after (<code>False</code>) passing through Mega encoder blocks`,name:"normalize_before_mega"},{anchor:"transformers.MegaConfig.normalization_type",description:`<strong>normalization_type</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;scalenorm&quot;</code>) &#x2014;
Type of normalization to use in Mega encoder blocks. Choose one of <code>&quot;scalenorm&quot;</code>, <code>&quot;layernorm&quot;</code>,
<code>&quot;rmsnorm&quot;</code>, <code>&quot;batchnorm&quot;</code>, or <code>&quot;syncbatchnorm&quot;</code> (GPU required for syncbatchnorm)`,name:"normalization_type"},{anchor:"transformers.MegaConfig.norm_affine",description:`<strong>norm_affine</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
If <code>True</code>, applies a parameterized affine transformation to inputs during normalization`,name:"norm_affine"},{anchor:"transformers.MegaConfig.activation",description:`<strong>activation</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;silu&quot;</code>) &#x2014;
Activation function to apply within Mega encoder blocks. Choose one of <code>&quot;silu&quot;</code>, <code>&quot;relu&quot;</code>, <code>&quot;linear&quot;</code>,
<code>&quot;gelu&quot;</code>, or <code>&quot;gelu_accurate&quot;</code>`,name:"activation"},{anchor:"transformers.MegaConfig.attention_activation",description:`<strong>attention_activation</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;softmax&quot;</code>) &#x2014;
Activation function to apply for single-headed self-attention (a la Transformer). Choose one of
<code>&quot;softmax&quot;</code>, <code>&quot;laplace&quot;</code>, or <code>&quot;relu2&quot;</code>`,name:"attention_activation"},{anchor:"transformers.MegaConfig.dropout_prob",description:`<strong>dropout_prob</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout probability for EMA self-attention`,name:"dropout_prob"},{anchor:"transformers.MegaConfig.hidden_dropout_prob",description:`<strong>hidden_dropout_prob</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.`,name:"hidden_dropout_prob"},{anchor:"transformers.MegaConfig.attention_probs_dropout_prob",description:`<strong>attention_probs_dropout_prob</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout ratio for the attention probabilities.`,name:"attention_probs_dropout_prob"},{anchor:"transformers.MegaConfig.use_feature_dropout",description:`<strong>use_feature_dropout</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether to use feature-based (<code>True</code>) or standard dropout (<code>False</code>)`,name:"use_feature_dropout"},{anchor:"transformers.MegaConfig.use_normalized_ffn",description:`<strong>use_normalized_ffn</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether to use the normalized feed-forward sub-layer in Mega blocks (<code>True</code>) or pass Mega encoder output
as-is (<code>False</code>)`,name:"use_normalized_ffn"},{anchor:"transformers.MegaConfig.nffn_hidden_size",description:`<strong>nffn_hidden_size</strong> (<code>int</code>, <em>optional</em>, defaults to 256) &#x2014;
If using the normalized feed-forward network (NFFN) layer within Mega (<code>use_normalized_ffn = True</code>), this
is the hidden size of the NFFN`,name:"nffn_hidden_size"},{anchor:"transformers.MegaConfig.normalize_before_ffn",description:`<strong>normalize_before_ffn</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether to normalize before (<code>True</code>) or after (<code>False</code>) the feed-forward portion of NFFN`,name:"normalize_before_ffn"},{anchor:"transformers.MegaConfig.nffn_activation_dropout_prob",description:`<strong>nffn_activation_dropout_prob</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout ratio for the NFFN component.`,name:"nffn_activation_dropout_prob"},{anchor:"transformers.MegaConfig.max_positions",description:`<strong>max_positions</strong> (<code>int</code>, <em>optional</em>, defaults to 2048) &#x2014;
The maximum sequence length to use for positional representations. For <code>&quot;simple&quot;</code> relative positional bias,
this is a hard limit on input length; <code>&quot;rotary&quot;</code> relative positional bias will extrapolate to longer
sequences`,name:"max_positions"},{anchor:"transformers.MegaConfig.add_token_type_embeddings",description:`<strong>add_token_type_embeddings</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether to account for token types in embeddings. Left as optional to maintain compatibility with original
implementation while adding support for token types.`,name:"add_token_type_embeddings"},{anchor:"transformers.MegaConfig.type_vocab_size",description:`<strong>type_vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 2) &#x2014;
The vocabulary size of the <code>token_type_ids</code> passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/mega#transformers.MegaModel">MegaModel</a>. Only used if
<code>add_token_type_embeddings = True</code>`,name:"type_vocab_size"},{anchor:"transformers.MegaConfig.initializer_range",description:`<strong>initializer_range</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.`,name:"initializer_range"},{anchor:"transformers.MegaConfig.ema_delta_alpha_range",description:`<strong>ema_delta_alpha_range</strong> (<code>float</code>, <em>optional</em>, defaults to 0.2) &#x2014;
The standard deviation for initializing the delta (damping factor) and alpha (decay factor) parameters in
MegaMultiDimensionDampedEma.`,name:"ema_delta_alpha_range"},{anchor:"transformers.MegaConfig.ema_beta_range",description:`<strong>ema_beta_range</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation for initializing the beta parameter (expansion matrix) in
MegaMultiDimensionDampedEma.`,name:"ema_beta_range"},{anchor:"transformers.MegaConfig.ema_gamma_omega_range",description:`<strong>ema_gamma_omega_range</strong> (<code>float</code>, <em>optional</em>, defaults to 1.0) &#x2014;
The standard deviation for initializing the gamma (projection matrix) and omega (residual weight)
parameters in MultiDimensionEMA.`,name:"ema_gamma_omega_range"},{anchor:"transformers.MegaConfig.relative_positional_bias",description:`<strong>relative_positional_bias</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;rotary&quot;</code>) &#x2014;
Type of relative positional encoding. Choose one of <code>&quot;rotary&quot;</code> or <code>&quot;simple&quot;</code>. If <code>&quot;simple&quot;</code> is selected,
<code>max_positions</code> is used as a limit on input size, while <code>&quot;rotary&quot;</code> extrapolates beyond <code>max_positions</code>.`,name:"relative_positional_bias"},{anchor:"transformers.MegaConfig.is_decoder",description:`<strong>is_decoder</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether the model is used as a decoder or not. If <code>False</code>, the model is used as an encoder.`,name:"is_decoder"},{anchor:"transformers.MegaConfig.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not the model should return the last key/values attentions (not used by all models). Only
relevant if <code>config.is_decoder=True</code>.`,name:"use_cache"},{anchor:"transformers.MegaConfig.classifier_dropout",description:`<strong>classifier_dropout</strong> (<code>float</code>, <em>optional</em>) &#x2014;
The dropout ratio for the classification head.`,name:"classifier_dropout"},{anchor:"transformers.MegaConfig.add_lm_hidden_dense_layer",description:`<strong>add_lm_hidden_dense_layer</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether to include a hidden layer for projection between encoder outputs and LM heads (<code>True</code>) or pass
hidden states directly to LM head (<code>False</code>). Remains optional for compatibility with original
implementation`,name:"add_lm_hidden_dense_layer"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/mega/configuration_mega.py#L28"}}),de=new ae({props:{anchor:"transformers.MegaConfig.example",$$slots:{default:[gs]},$$scope:{ctx:v}}}),Ge=new O({props:{title:"MegaModel",local:"transformers.MegaModel",headingTag:"h2"}}),Le=new I({props:{name:"class transformers.MegaModel",anchor:"transformers.MegaModel",parameters:[{name:"config",val:": MegaConfig"},{name:"add_pooling_layer",val:" = True"}],parametersDescription:[{anchor:"transformers.MegaModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/mega#transformers.MegaConfig">MegaConfig</a>) &#x2014; Model configuration class with all the parameters of the
model. Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/mega/modeling_mega.py#L1444"}}),Ne=new I({props:{name:"forward",anchor:"transformers.MegaModel.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_hidden_states",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"past_key_values",val:": typing.Optional[list[torch.FloatTensor]] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.MegaModel.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.MegaModel.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.MegaModel.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0,1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.
This parameter can only be used when the model is initialized with <code>add_token_type_embeddings</code> parameter
set to <code>True</code>. All the value in this tensor should be always &lt; config.type_vocab_size.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.MegaModel.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.MegaModel.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.MegaModel.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.MegaModel.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.MegaModel.forward.encoder_hidden_states",description:`<strong>encoder_hidden_states</strong>  (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
the model is configured as a decoder.`,name:"encoder_hidden_states"},{anchor:"transformers.MegaModel.forward.encoder_attention_mask",description:`<strong>encoder_attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
the cross-attention if the model is configured as a decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>`,name:"encoder_attention_mask"},{anchor:"transformers.MegaModel.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>tuple(tuple(torch.FloatTensor))</code> of length <code>config.n_layers</code> with each tuple having 4 tensors of shape <code>(batch_size, num_heads, sequence_length - 1, embed_size_per_head)</code>) &#x2014;
Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.</p>
<p>If <code>past_key_values</code> are used, the user can optionally input only the last <code>decoder_input_ids</code> (those that
don&#x2019;t have their past key value states given to this model) of shape <code>(batch_size, 1)</code> instead of all
<code>decoder_input_ids</code> of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.MegaModel.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/mega/modeling_mega.py#L1479",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions"
>transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/mega#transformers.MegaConfig"
>MegaConfig</a>) and inputs.</p>
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
`}}),ce=new $e({props:{$$slots:{default:[fs]},$$scope:{ctx:v}}}),pe=new ae({props:{anchor:"transformers.MegaModel.forward.example",$$slots:{default:[_s]},$$scope:{ctx:v}}}),Xe=new O({props:{title:"MegaForCausalLM",local:"transformers.MegaForCausalLM",headingTag:"h2"}}),Re=new I({props:{name:"class transformers.MegaForCausalLM",anchor:"transformers.MegaForCausalLM",parameters:[{name:"config",val:": MegaConfig"}],parametersDescription:[{anchor:"transformers.MegaForCausalLM.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/mega#transformers.MegaConfig">MegaConfig</a>) &#x2014; Model configuration class with all the parameters of the
model. Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/mega/modeling_mega.py#L1644"}}),Ve=new I({props:{name:"forward",anchor:"transformers.MegaForCausalLM.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"encoder_hidden_states",val:": typing.Optional[torch.FloatTensor] = None"},{name:"encoder_attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[tuple[tuple[torch.FloatTensor]]] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.MegaForCausalLM.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.MegaForCausalLM.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.MegaForCausalLM.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0,1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.
This parameter can only be used when the model is initialized with <code>add_token_type_embeddings</code> parameter
set to <code>True</code>. All the value in this tensor should be always &lt; config.type_vocab_size.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.MegaForCausalLM.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.MegaForCausalLM.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.MegaForCausalLM.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.MegaForCausalLM.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.MegaForCausalLM.forward.encoder_hidden_states",description:`<strong>encoder_hidden_states</strong>  (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
the model is configured as a decoder.`,name:"encoder_hidden_states"},{anchor:"transformers.MegaForCausalLM.forward.encoder_attention_mask",description:`<strong>encoder_attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
the cross-attention if the model is configured as a decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>`,name:"encoder_attention_mask"},{anchor:"transformers.MegaForCausalLM.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the left-to-right language modeling loss (next word prediction). Indices should be in
<code>[-100, 0, ..., config.vocab_size]</code> (see <code>input_ids</code> docstring) Tokens with indices set to <code>-100</code> are
ignored (masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>`,name:"labels"},{anchor:"transformers.MegaForCausalLM.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>tuple(tuple(torch.FloatTensor))</code> of length <code>config.n_layers</code> with each tuple having 4 tensors of shape <code>(batch_size, num_heads, sequence_length - 1, embed_size_per_head)</code>) &#x2014;
Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.</p>
<p>If <code>past_key_values</code> are used, the user can optionally input only the last <code>decoder_input_ids</code> (those that
don&#x2019;t have their past key value states given to this model) of shape <code>(batch_size, 1)</code> instead of all
<code>decoder_input_ids</code> of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.MegaForCausalLM.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/mega/modeling_mega.py#L1667",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithCrossAttentions"
>transformers.modeling_outputs.CausalLMOutputWithCrossAttentions</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/mega#transformers.MegaConfig"
>MegaConfig</a>) and inputs.</p>
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
`}}),me=new $e({props:{$$slots:{default:[Ms]},$$scope:{ctx:v}}}),ue=new ae({props:{anchor:"transformers.MegaForCausalLM.forward.example",$$slots:{default:[bs]},$$scope:{ctx:v}}}),He=new O({props:{title:"MegaForMaskedLM",local:"transformers.MegaForMaskedLM",headingTag:"h2"}}),Ee=new I({props:{name:"class transformers.MegaForMaskedLM",anchor:"transformers.MegaForMaskedLM",parameters:[{name:"config",val:": MegaConfig"}],parametersDescription:[{anchor:"transformers.MegaForMaskedLM.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/mega#transformers.MegaConfig">MegaConfig</a>) &#x2014; Model configuration class with all the parameters of the
model. Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/mega/modeling_mega.py#L1799"}}),Ae=new I({props:{name:"forward",anchor:"transformers.MegaForMaskedLM.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"encoder_hidden_states",val:": typing.Optional[torch.FloatTensor] = None"},{name:"encoder_attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.MegaForMaskedLM.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.MegaForMaskedLM.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.MegaForMaskedLM.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0,1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.
This parameter can only be used when the model is initialized with <code>add_token_type_embeddings</code> parameter
set to <code>True</code>. All the value in this tensor should be always &lt; config.type_vocab_size.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.MegaForMaskedLM.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.MegaForMaskedLM.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.MegaForMaskedLM.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.MegaForMaskedLM.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.MegaForMaskedLM.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should be in <code>[-100, 0, ..., config.vocab_size]</code> (see <code>input_ids</code> docstring) Tokens with indices set to <code>-100</code> are ignored (masked), the
loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>`,name:"labels"},{anchor:"transformers.MegaForMaskedLM.forward.kwargs",description:`<strong>kwargs</strong> (<code>dict[str, any]</code>, optional, defaults to <em>{}</em>) &#x2014;
Used to hide legacy arguments that have been deprecated.`,name:"kwargs"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/mega/modeling_mega.py#L1830",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.MaskedLMOutput"
>transformers.modeling_outputs.MaskedLMOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/mega#transformers.MegaConfig"
>MegaConfig</a>) and inputs.</p>
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
`}}),he=new $e({props:{$$slots:{default:[ys]},$$scope:{ctx:v}}}),ge=new ae({props:{anchor:"transformers.MegaForMaskedLM.forward.example",$$slots:{default:[Ts]},$$scope:{ctx:v}}}),Qe=new O({props:{title:"MegaForSequenceClassification",local:"transformers.MegaForSequenceClassification",headingTag:"h2"}}),Se=new I({props:{name:"class transformers.MegaForSequenceClassification",anchor:"transformers.MegaForSequenceClassification",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.MegaForSequenceClassification.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/mega#transformers.MegaConfig">MegaConfig</a>) &#x2014; Model configuration class with all the parameters of the
model. Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/mega/modeling_mega.py#L1903"}}),Ye=new I({props:{name:"forward",anchor:"transformers.MegaForSequenceClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.MegaForSequenceClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.MegaForSequenceClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.MegaForSequenceClassification.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0,1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.
This parameter can only be used when the model is initialized with <code>add_token_type_embeddings</code> parameter
set to <code>True</code>. All the value in this tensor should be always &lt; config.type_vocab_size.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.MegaForSequenceClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.MegaForSequenceClassification.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.MegaForSequenceClassification.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.MegaForSequenceClassification.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.MegaForSequenceClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the sequence classification/regression loss. Indices should be in <code>[0, ..., config.num_labels - 1]</code>. If <code>config.num_labels == 1</code> a regression loss is computed (Mean-Square loss), If
<code>config.num_labels &gt; 1</code> a classification loss is computed (Cross-Entropy).`,name:"labels"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/mega/modeling_mega.py#L1915",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput"
>transformers.modeling_outputs.SequenceClassifierOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/mega#transformers.MegaConfig"
>MegaConfig</a>) and inputs.</p>
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
`}}),fe=new $e({props:{$$slots:{default:[ws]},$$scope:{ctx:v}}}),_e=new ae({props:{anchor:"transformers.MegaForSequenceClassification.forward.example",$$slots:{default:[vs]},$$scope:{ctx:v}}}),Me=new ae({props:{anchor:"transformers.MegaForSequenceClassification.forward.example-2",$$slots:{default:[ks]},$$scope:{ctx:v}}}),Pe=new O({props:{title:"MegaForMultipleChoice",local:"transformers.MegaForMultipleChoice",headingTag:"h2"}}),Oe=new I({props:{name:"class transformers.MegaForMultipleChoice",anchor:"transformers.MegaForMultipleChoice",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.MegaForMultipleChoice.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/mega#transformers.MegaConfig">MegaConfig</a>) &#x2014; Model configuration class with all the parameters of the
model. Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/mega/modeling_mega.py#L1994"}}),De=new I({props:{name:"forward",anchor:"transformers.MegaForMultipleChoice.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.MegaForMultipleChoice.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, num_choices, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.MegaForMultipleChoice.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_choices, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.MegaForMultipleChoice.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, num_choices, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0,1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.
This parameter can only be used when the model is initialized with <code>add_token_type_embeddings</code> parameter
set to <code>True</code>. All the value in this tensor should be always &lt; config.type_vocab_size.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.MegaForMultipleChoice.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_choices, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.MegaForMultipleChoice.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.MegaForMultipleChoice.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.MegaForMultipleChoice.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.MegaForMultipleChoice.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the multiple choice classification loss. Indices should be in <code>[0, ..., num_choices-1]</code> where <code>num_choices</code> is the size of the second dimension of the input tensors. (See
<code>input_ids</code> above)`,name:"labels"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/mega/modeling_mega.py#L2005",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.MultipleChoiceModelOutput"
>transformers.modeling_outputs.MultipleChoiceModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/mega#transformers.MegaConfig"
>MegaConfig</a>) and inputs.</p>
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
`}}),be=new $e({props:{$$slots:{default:[$s]},$$scope:{ctx:v}}}),ye=new ae({props:{anchor:"transformers.MegaForMultipleChoice.forward.example",$$slots:{default:[Cs]},$$scope:{ctx:v}}}),Ke=new O({props:{title:"MegaForTokenClassification",local:"transformers.MegaForTokenClassification",headingTag:"h2"}}),et=new I({props:{name:"class transformers.MegaForTokenClassification",anchor:"transformers.MegaForTokenClassification",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.MegaForTokenClassification.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/mega#transformers.MegaConfig">MegaConfig</a>) &#x2014; Model configuration class with all the parameters of the
model. Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/mega/modeling_mega.py#L2079"}}),tt=new I({props:{name:"forward",anchor:"transformers.MegaForTokenClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.MegaForTokenClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.MegaForTokenClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.MegaForTokenClassification.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0,1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.
This parameter can only be used when the model is initialized with <code>add_token_type_embeddings</code> parameter
set to <code>True</code>. All the value in this tensor should be always &lt; config.type_vocab_size.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.MegaForTokenClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.MegaForTokenClassification.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.MegaForTokenClassification.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.MegaForTokenClassification.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.MegaForTokenClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the token classification loss. Indices should be in <code>[0, ..., config.num_labels - 1]</code>.`,name:"labels"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/mega/modeling_mega.py#L2094",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput"
>transformers.modeling_outputs.TokenClassifierOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/mega#transformers.MegaConfig"
>MegaConfig</a>) and inputs.</p>
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
`}}),Te=new $e({props:{$$slots:{default:[js]},$$scope:{ctx:v}}}),we=new ae({props:{anchor:"transformers.MegaForTokenClassification.forward.example",$$slots:{default:[zs]},$$scope:{ctx:v}}}),nt=new O({props:{title:"MegaForQuestionAnswering",local:"transformers.MegaForQuestionAnswering",headingTag:"h2"}}),ot=new I({props:{name:"class transformers.MegaForQuestionAnswering",anchor:"transformers.MegaForQuestionAnswering",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.MegaForQuestionAnswering.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/mega#transformers.MegaConfig">MegaConfig</a>) &#x2014; Model configuration class with all the parameters of the
model. Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/mega/modeling_mega.py#L2179"}}),st=new I({props:{name:"forward",anchor:"transformers.MegaForQuestionAnswering.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"start_positions",val:": typing.Optional[torch.LongTensor] = None"},{name:"end_positions",val:": typing.Optional[torch.LongTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.MegaForQuestionAnswering.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.MegaForQuestionAnswering.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.MegaForQuestionAnswering.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0,1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.
This parameter can only be used when the model is initialized with <code>add_token_type_embeddings</code> parameter
set to <code>True</code>. All the value in this tensor should be always &lt; config.type_vocab_size.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.MegaForQuestionAnswering.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.MegaForQuestionAnswering.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.MegaForQuestionAnswering.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.MegaForQuestionAnswering.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.MegaForQuestionAnswering.forward.start_positions",description:`<strong>start_positions</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for position (index) of the start of the labelled span for computing the token classification loss.
Positions are clamped to the length of the sequence (<code>sequence_length</code>). Position outside of the sequence
are not taken into account for computing the loss.`,name:"start_positions"},{anchor:"transformers.MegaForQuestionAnswering.forward.end_positions",description:`<strong>end_positions</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for position (index) of the end of the labelled span for computing the token classification loss.
Positions are clamped to the length of the sequence (<code>sequence_length</code>). Position outside of the sequence
are not taken into account for computing the loss.`,name:"end_positions"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/mega/modeling_mega.py#L2190",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.QuestionAnsweringModelOutput"
>transformers.modeling_outputs.QuestionAnsweringModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/mega#transformers.MegaConfig"
>MegaConfig</a>) and inputs.</p>
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
`}}),ve=new $e({props:{$$slots:{default:[Js]},$$scope:{ctx:v}}}),ke=new ae({props:{anchor:"transformers.MegaForQuestionAnswering.forward.example",$$slots:{default:[xs]},$$scope:{ctx:v}}}),at=new us({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/mega.md"}}),{c(){t=c("meta"),h=r(),s=c("p"),d=r(),T=c("p"),T.innerHTML=n,w=r(),g(Ce.$$.fragment),Nt=r(),ie=c("div"),ie.innerHTML=yo,Xt=r(),g(le.$$.fragment),Rt=r(),g(je.$$.fragment),Vt=r(),ze=c("p"),ze.innerHTML=To,Ht=r(),Je=c("p"),Je.textContent=wo,Et=r(),xe=c("p"),xe.innerHTML=vo,At=r(),Fe=c("p"),Fe.innerHTML=ko,Qt=r(),g(Ue.$$.fragment),St=r(),We=c("ul"),We.innerHTML=$o,Yt=r(),g(Ze.$$.fragment),Pt=r(),Ie=c("ul"),Ie.innerHTML=Co,Ot=r(),g(Be.$$.fragment),Dt=r(),q=c("div"),g(qe.$$.fragment),_n=r(),rt=c("p"),rt.innerHTML=jo,Mn=r(),it=c("p"),it.innerHTML=zo,bn=r(),g(de.$$.fragment),Kt=r(),g(Ge.$$.fragment),en=r(),k=c("div"),g(Le.$$.fragment),yn=r(),lt=c("p"),lt.textContent=Jo,Tn=r(),dt=c("p"),dt.innerHTML=xo,wn=r(),ct=c("p"),ct.innerHTML=Fo,vn=r(),pt=c("p"),pt.innerHTML=Uo,kn=r(),mt=c("p"),mt.innerHTML=Wo,$n=r(),ut=c("p"),ut.innerHTML=Zo,Cn=r(),E=c("div"),g(Ne.$$.fragment),jn=r(),ht=c("p"),ht.innerHTML=Io,zn=r(),g(ce.$$.fragment),Jn=r(),g(pe.$$.fragment),tn=r(),g(Xe.$$.fragment),nn=r(),J=c("div"),g(Re.$$.fragment),xn=r(),gt=c("p"),gt.innerHTML=Bo,Fn=r(),ft=c("p"),ft.innerHTML=qo,Un=r(),_t=c("p"),_t.innerHTML=Go,Wn=r(),A=c("div"),g(Ve.$$.fragment),Zn=r(),Mt=c("p"),Mt.innerHTML=Lo,In=r(),g(me.$$.fragment),Bn=r(),g(ue.$$.fragment),on=r(),g(He.$$.fragment),sn=r(),x=c("div"),g(Ee.$$.fragment),qn=r(),bt=c("p"),bt.innerHTML=No,Gn=r(),yt=c("p"),yt.innerHTML=Xo,Ln=r(),Tt=c("p"),Tt.innerHTML=Ro,Nn=r(),Q=c("div"),g(Ae.$$.fragment),Xn=r(),wt=c("p"),wt.innerHTML=Vo,Rn=r(),g(he.$$.fragment),Vn=r(),g(ge.$$.fragment),an=r(),g(Qe.$$.fragment),rn=r(),F=c("div"),g(Se.$$.fragment),Hn=r(),vt=c("p"),vt.textContent=Ho,En=r(),kt=c("p"),kt.innerHTML=Eo,An=r(),$t=c("p"),$t.innerHTML=Ao,Qn=r(),B=c("div"),g(Ye.$$.fragment),Sn=r(),Ct=c("p"),Ct.innerHTML=Qo,Yn=r(),g(fe.$$.fragment),Pn=r(),g(_e.$$.fragment),On=r(),g(Me.$$.fragment),ln=r(),g(Pe.$$.fragment),dn=r(),U=c("div"),g(Oe.$$.fragment),Dn=r(),jt=c("p"),jt.textContent=So,Kn=r(),zt=c("p"),zt.innerHTML=Yo,eo=r(),Jt=c("p"),Jt.innerHTML=Po,to=r(),S=c("div"),g(De.$$.fragment),no=r(),xt=c("p"),xt.innerHTML=Oo,oo=r(),g(be.$$.fragment),so=r(),g(ye.$$.fragment),cn=r(),g(Ke.$$.fragment),pn=r(),W=c("div"),g(et.$$.fragment),ao=r(),Ft=c("p"),Ft.textContent=Do,ro=r(),Ut=c("p"),Ut.innerHTML=Ko,io=r(),Wt=c("p"),Wt.innerHTML=es,lo=r(),Y=c("div"),g(tt.$$.fragment),co=r(),Zt=c("p"),Zt.innerHTML=ts,po=r(),g(Te.$$.fragment),mo=r(),g(we.$$.fragment),mn=r(),g(nt.$$.fragment),un=r(),Z=c("div"),g(ot.$$.fragment),uo=r(),It=c("p"),It.innerHTML=ns,ho=r(),Bt=c("p"),Bt.innerHTML=os,go=r(),qt=c("p"),qt.innerHTML=ss,fo=r(),P=c("div"),g(st.$$.fragment),_o=r(),Gt=c("p"),Gt.innerHTML=as,Mo=r(),g(ve.$$.fragment),bo=r(),g(ke.$$.fragment),hn=r(),g(at.$$.fragment),gn=r(),Lt=c("p"),this.h()},l(e){const o=ms("svelte-u9bgzb",document.head);t=p(o,"META",{name:!0,content:!0}),o.forEach(l),h=i(e),s=p(e,"P",{}),j(s).forEach(l),d=i(e),T=p(e,"P",{"data-svelte-h":!0}),u(T)!=="svelte-6toczz"&&(T.innerHTML=n),w=i(e),f(Ce.$$.fragment,e),Nt=i(e),ie=p(e,"DIV",{class:!0,"data-svelte-h":!0}),u(ie)!=="svelte-13t8s2t"&&(ie.innerHTML=yo),Xt=i(e),f(le.$$.fragment,e),Rt=i(e),f(je.$$.fragment,e),Vt=i(e),ze=p(e,"P",{"data-svelte-h":!0}),u(ze)!=="svelte-1jfeswk"&&(ze.innerHTML=To),Ht=i(e),Je=p(e,"P",{"data-svelte-h":!0}),u(Je)!=="svelte-vfdo9a"&&(Je.textContent=wo),Et=i(e),xe=p(e,"P",{"data-svelte-h":!0}),u(xe)!=="svelte-usb8p6"&&(xe.innerHTML=vo),At=i(e),Fe=p(e,"P",{"data-svelte-h":!0}),u(Fe)!=="svelte-hu61i0"&&(Fe.innerHTML=ko),Qt=i(e),f(Ue.$$.fragment,e),St=i(e),We=p(e,"UL",{"data-svelte-h":!0}),u(We)!=="svelte-56llcu"&&(We.innerHTML=$o),Yt=i(e),f(Ze.$$.fragment,e),Pt=i(e),Ie=p(e,"UL",{"data-svelte-h":!0}),u(Ie)!=="svelte-12nl4xf"&&(Ie.innerHTML=Co),Ot=i(e),f(Be.$$.fragment,e),Dt=i(e),q=p(e,"DIV",{class:!0});var D=j(q);f(qe.$$.fragment,D),_n=i(D),rt=p(D,"P",{"data-svelte-h":!0}),u(rt)!=="svelte-7yo7n5"&&(rt.innerHTML=jo),Mn=i(D),it=p(D,"P",{"data-svelte-h":!0}),u(it)!=="svelte-1ek1ss9"&&(it.innerHTML=zo),bn=i(D),f(de.$$.fragment,D),D.forEach(l),Kt=i(e),f(Ge.$$.fragment,e),en=i(e),k=p(e,"DIV",{class:!0});var $=j(k);f(Le.$$.fragment,$),yn=i($),lt=p($,"P",{"data-svelte-h":!0}),u(lt)!=="svelte-wsjevx"&&(lt.textContent=Jo),Tn=i($),dt=p($,"P",{"data-svelte-h":!0}),u(dt)!=="svelte-q52n56"&&(dt.innerHTML=xo),wn=i($),ct=p($,"P",{"data-svelte-h":!0}),u(ct)!=="svelte-hswkmf"&&(ct.innerHTML=Fo),vn=i($),pt=p($,"P",{"data-svelte-h":!0}),u(pt)!=="svelte-1jfjp18"&&(pt.innerHTML=Uo),kn=i($),mt=p($,"P",{"data-svelte-h":!0}),u(mt)!=="svelte-1u2qb33"&&(mt.innerHTML=Wo),$n=i($),ut=p($,"P",{"data-svelte-h":!0}),u(ut)!=="svelte-1fes0vq"&&(ut.innerHTML=Zo),Cn=i($),E=p($,"DIV",{class:!0});var K=j(E);f(Ne.$$.fragment,K),jn=i(K),ht=p(K,"P",{"data-svelte-h":!0}),u(ht)!=="svelte-1nk162d"&&(ht.innerHTML=Io),zn=i(K),f(ce.$$.fragment,K),Jn=i(K),f(pe.$$.fragment,K),K.forEach(l),$.forEach(l),tn=i(e),f(Xe.$$.fragment,e),nn=i(e),J=p(e,"DIV",{class:!0});var G=j(J);f(Re.$$.fragment,G),xn=i(G),gt=p(G,"P",{"data-svelte-h":!0}),u(gt)!=="svelte-pogedt"&&(gt.innerHTML=Bo),Fn=i(G),ft=p(G,"P",{"data-svelte-h":!0}),u(ft)!=="svelte-q52n56"&&(ft.innerHTML=qo),Un=i(G),_t=p(G,"P",{"data-svelte-h":!0}),u(_t)!=="svelte-hswkmf"&&(_t.innerHTML=Go),Wn=i(G),A=p(G,"DIV",{class:!0});var ee=j(A);f(Ve.$$.fragment,ee),Zn=i(ee),Mt=p(ee,"P",{"data-svelte-h":!0}),u(Mt)!=="svelte-zjayy9"&&(Mt.innerHTML=Lo),In=i(ee),f(me.$$.fragment,ee),Bn=i(ee),f(ue.$$.fragment,ee),ee.forEach(l),G.forEach(l),on=i(e),f(He.$$.fragment,e),sn=i(e),x=p(e,"DIV",{class:!0});var L=j(x);f(Ee.$$.fragment,L),qn=i(L),bt=p(L,"P",{"data-svelte-h":!0}),u(bt)!=="svelte-vwo04g"&&(bt.innerHTML=No),Gn=i(L),yt=p(L,"P",{"data-svelte-h":!0}),u(yt)!=="svelte-q52n56"&&(yt.innerHTML=Xo),Ln=i(L),Tt=p(L,"P",{"data-svelte-h":!0}),u(Tt)!=="svelte-hswkmf"&&(Tt.innerHTML=Ro),Nn=i(L),Q=p(L,"DIV",{class:!0});var te=j(Q);f(Ae.$$.fragment,te),Xn=i(te),wt=p(te,"P",{"data-svelte-h":!0}),u(wt)!=="svelte-1igo5wp"&&(wt.innerHTML=Vo),Rn=i(te),f(he.$$.fragment,te),Vn=i(te),f(ge.$$.fragment,te),te.forEach(l),L.forEach(l),an=i(e),f(Qe.$$.fragment,e),rn=i(e),F=p(e,"DIV",{class:!0});var N=j(F);f(Se.$$.fragment,N),Hn=i(N),vt=p(N,"P",{"data-svelte-h":!0}),u(vt)!=="svelte-12iqzjk"&&(vt.textContent=Ho),En=i(N),kt=p(N,"P",{"data-svelte-h":!0}),u(kt)!=="svelte-q52n56"&&(kt.innerHTML=Eo),An=i(N),$t=p(N,"P",{"data-svelte-h":!0}),u($t)!=="svelte-hswkmf"&&($t.innerHTML=Ao),Qn=i(N),B=p(N,"DIV",{class:!0});var X=j(B);f(Ye.$$.fragment,X),Sn=i(X),Ct=p(X,"P",{"data-svelte-h":!0}),u(Ct)!=="svelte-12pmr8h"&&(Ct.innerHTML=Qo),Yn=i(X),f(fe.$$.fragment,X),Pn=i(X),f(_e.$$.fragment,X),On=i(X),f(Me.$$.fragment,X),X.forEach(l),N.forEach(l),ln=i(e),f(Pe.$$.fragment,e),dn=i(e),U=p(e,"DIV",{class:!0});var R=j(U);f(Oe.$$.fragment,R),Dn=i(R),jt=p(R,"P",{"data-svelte-h":!0}),u(jt)!=="svelte-370gjw"&&(jt.textContent=So),Kn=i(R),zt=p(R,"P",{"data-svelte-h":!0}),u(zt)!=="svelte-q52n56"&&(zt.innerHTML=Yo),eo=i(R),Jt=p(R,"P",{"data-svelte-h":!0}),u(Jt)!=="svelte-hswkmf"&&(Jt.innerHTML=Po),to=i(R),S=p(R,"DIV",{class:!0});var ne=j(S);f(De.$$.fragment,ne),no=i(ne),xt=p(ne,"P",{"data-svelte-h":!0}),u(xt)!=="svelte-cxky9t"&&(xt.innerHTML=Oo),oo=i(ne),f(be.$$.fragment,ne),so=i(ne),f(ye.$$.fragment,ne),ne.forEach(l),R.forEach(l),cn=i(e),f(Ke.$$.fragment,e),pn=i(e),W=p(e,"DIV",{class:!0});var V=j(W);f(et.$$.fragment,V),ao=i(V),Ft=p(V,"P",{"data-svelte-h":!0}),u(Ft)!=="svelte-1bzmarr"&&(Ft.textContent=Do),ro=i(V),Ut=p(V,"P",{"data-svelte-h":!0}),u(Ut)!=="svelte-q52n56"&&(Ut.innerHTML=Ko),io=i(V),Wt=p(V,"P",{"data-svelte-h":!0}),u(Wt)!=="svelte-hswkmf"&&(Wt.innerHTML=es),lo=i(V),Y=p(V,"DIV",{class:!0});var oe=j(Y);f(tt.$$.fragment,oe),co=i(oe),Zt=p(oe,"P",{"data-svelte-h":!0}),u(Zt)!=="svelte-1kebmwt"&&(Zt.innerHTML=ts),po=i(oe),f(Te.$$.fragment,oe),mo=i(oe),f(we.$$.fragment,oe),oe.forEach(l),V.forEach(l),mn=i(e),f(nt.$$.fragment,e),un=i(e),Z=p(e,"DIV",{class:!0});var H=j(Z);f(ot.$$.fragment,H),uo=i(H),It=p(H,"P",{"data-svelte-h":!0}),u(It)!=="svelte-8yqw99"&&(It.innerHTML=ns),ho=i(H),Bt=p(H,"P",{"data-svelte-h":!0}),u(Bt)!=="svelte-q52n56"&&(Bt.innerHTML=os),go=i(H),qt=p(H,"P",{"data-svelte-h":!0}),u(qt)!=="svelte-hswkmf"&&(qt.innerHTML=ss),fo=i(H),P=p(H,"DIV",{class:!0});var se=j(P);f(st.$$.fragment,se),_o=i(se),Gt=p(se,"P",{"data-svelte-h":!0}),u(Gt)!=="svelte-us7frz"&&(Gt.innerHTML=as),Mo=i(se),f(ve.$$.fragment,se),bo=i(se),f(ke.$$.fragment,se),se.forEach(l),H.forEach(l),hn=i(e),f(at.$$.fragment,e),gn=i(e),Lt=p(e,"P",{}),j(Lt).forEach(l),this.h()},h(){C(t,"name","hf:doc:metadata"),C(t,"content",Us),C(ie,"class","flex flex-wrap space-x-1"),C(q,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),C(E,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),C(k,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),C(A,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),C(J,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),C(Q,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),C(x,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),C(B,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),C(F,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),C(S,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),C(U,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),C(Y,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),C(W,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),C(P,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),C(Z,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,o){a(document.head,t),m(e,h,o),m(e,s,o),m(e,d,o),m(e,T,o),m(e,w,o),_(Ce,e,o),m(e,Nt,o),m(e,ie,o),m(e,Xt,o),_(le,e,o),m(e,Rt,o),_(je,e,o),m(e,Vt,o),m(e,ze,o),m(e,Ht,o),m(e,Je,o),m(e,Et,o),m(e,xe,o),m(e,At,o),m(e,Fe,o),m(e,Qt,o),_(Ue,e,o),m(e,St,o),m(e,We,o),m(e,Yt,o),_(Ze,e,o),m(e,Pt,o),m(e,Ie,o),m(e,Ot,o),_(Be,e,o),m(e,Dt,o),m(e,q,o),_(qe,q,null),a(q,_n),a(q,rt),a(q,Mn),a(q,it),a(q,bn),_(de,q,null),m(e,Kt,o),_(Ge,e,o),m(e,en,o),m(e,k,o),_(Le,k,null),a(k,yn),a(k,lt),a(k,Tn),a(k,dt),a(k,wn),a(k,ct),a(k,vn),a(k,pt),a(k,kn),a(k,mt),a(k,$n),a(k,ut),a(k,Cn),a(k,E),_(Ne,E,null),a(E,jn),a(E,ht),a(E,zn),_(ce,E,null),a(E,Jn),_(pe,E,null),m(e,tn,o),_(Xe,e,o),m(e,nn,o),m(e,J,o),_(Re,J,null),a(J,xn),a(J,gt),a(J,Fn),a(J,ft),a(J,Un),a(J,_t),a(J,Wn),a(J,A),_(Ve,A,null),a(A,Zn),a(A,Mt),a(A,In),_(me,A,null),a(A,Bn),_(ue,A,null),m(e,on,o),_(He,e,o),m(e,sn,o),m(e,x,o),_(Ee,x,null),a(x,qn),a(x,bt),a(x,Gn),a(x,yt),a(x,Ln),a(x,Tt),a(x,Nn),a(x,Q),_(Ae,Q,null),a(Q,Xn),a(Q,wt),a(Q,Rn),_(he,Q,null),a(Q,Vn),_(ge,Q,null),m(e,an,o),_(Qe,e,o),m(e,rn,o),m(e,F,o),_(Se,F,null),a(F,Hn),a(F,vt),a(F,En),a(F,kt),a(F,An),a(F,$t),a(F,Qn),a(F,B),_(Ye,B,null),a(B,Sn),a(B,Ct),a(B,Yn),_(fe,B,null),a(B,Pn),_(_e,B,null),a(B,On),_(Me,B,null),m(e,ln,o),_(Pe,e,o),m(e,dn,o),m(e,U,o),_(Oe,U,null),a(U,Dn),a(U,jt),a(U,Kn),a(U,zt),a(U,eo),a(U,Jt),a(U,to),a(U,S),_(De,S,null),a(S,no),a(S,xt),a(S,oo),_(be,S,null),a(S,so),_(ye,S,null),m(e,cn,o),_(Ke,e,o),m(e,pn,o),m(e,W,o),_(et,W,null),a(W,ao),a(W,Ft),a(W,ro),a(W,Ut),a(W,io),a(W,Wt),a(W,lo),a(W,Y),_(tt,Y,null),a(Y,co),a(Y,Zt),a(Y,po),_(Te,Y,null),a(Y,mo),_(we,Y,null),m(e,mn,o),_(nt,e,o),m(e,un,o),m(e,Z,o),_(ot,Z,null),a(Z,uo),a(Z,It),a(Z,ho),a(Z,Bt),a(Z,go),a(Z,qt),a(Z,fo),a(Z,P),_(st,P,null),a(P,_o),a(P,Gt),a(P,Mo),_(ve,P,null),a(P,bo),_(ke,P,null),m(e,hn,o),_(at,e,o),m(e,gn,o),m(e,Lt,o),fn=!0},p(e,[o]){const D={};o&2&&(D.$$scope={dirty:o,ctx:e}),le.$set(D);const $={};o&2&&($.$$scope={dirty:o,ctx:e}),de.$set($);const K={};o&2&&(K.$$scope={dirty:o,ctx:e}),ce.$set(K);const G={};o&2&&(G.$$scope={dirty:o,ctx:e}),pe.$set(G);const ee={};o&2&&(ee.$$scope={dirty:o,ctx:e}),me.$set(ee);const L={};o&2&&(L.$$scope={dirty:o,ctx:e}),ue.$set(L);const te={};o&2&&(te.$$scope={dirty:o,ctx:e}),he.$set(te);const N={};o&2&&(N.$$scope={dirty:o,ctx:e}),ge.$set(N);const X={};o&2&&(X.$$scope={dirty:o,ctx:e}),fe.$set(X);const R={};o&2&&(R.$$scope={dirty:o,ctx:e}),_e.$set(R);const ne={};o&2&&(ne.$$scope={dirty:o,ctx:e}),Me.$set(ne);const V={};o&2&&(V.$$scope={dirty:o,ctx:e}),be.$set(V);const oe={};o&2&&(oe.$$scope={dirty:o,ctx:e}),ye.$set(oe);const H={};o&2&&(H.$$scope={dirty:o,ctx:e}),Te.$set(H);const se={};o&2&&(se.$$scope={dirty:o,ctx:e}),we.$set(se);const rs={};o&2&&(rs.$$scope={dirty:o,ctx:e}),ve.$set(rs);const is={};o&2&&(is.$$scope={dirty:o,ctx:e}),ke.$set(is)},i(e){fn||(M(Ce.$$.fragment,e),M(le.$$.fragment,e),M(je.$$.fragment,e),M(Ue.$$.fragment,e),M(Ze.$$.fragment,e),M(Be.$$.fragment,e),M(qe.$$.fragment,e),M(de.$$.fragment,e),M(Ge.$$.fragment,e),M(Le.$$.fragment,e),M(Ne.$$.fragment,e),M(ce.$$.fragment,e),M(pe.$$.fragment,e),M(Xe.$$.fragment,e),M(Re.$$.fragment,e),M(Ve.$$.fragment,e),M(me.$$.fragment,e),M(ue.$$.fragment,e),M(He.$$.fragment,e),M(Ee.$$.fragment,e),M(Ae.$$.fragment,e),M(he.$$.fragment,e),M(ge.$$.fragment,e),M(Qe.$$.fragment,e),M(Se.$$.fragment,e),M(Ye.$$.fragment,e),M(fe.$$.fragment,e),M(_e.$$.fragment,e),M(Me.$$.fragment,e),M(Pe.$$.fragment,e),M(Oe.$$.fragment,e),M(De.$$.fragment,e),M(be.$$.fragment,e),M(ye.$$.fragment,e),M(Ke.$$.fragment,e),M(et.$$.fragment,e),M(tt.$$.fragment,e),M(Te.$$.fragment,e),M(we.$$.fragment,e),M(nt.$$.fragment,e),M(ot.$$.fragment,e),M(st.$$.fragment,e),M(ve.$$.fragment,e),M(ke.$$.fragment,e),M(at.$$.fragment,e),fn=!0)},o(e){b(Ce.$$.fragment,e),b(le.$$.fragment,e),b(je.$$.fragment,e),b(Ue.$$.fragment,e),b(Ze.$$.fragment,e),b(Be.$$.fragment,e),b(qe.$$.fragment,e),b(de.$$.fragment,e),b(Ge.$$.fragment,e),b(Le.$$.fragment,e),b(Ne.$$.fragment,e),b(ce.$$.fragment,e),b(pe.$$.fragment,e),b(Xe.$$.fragment,e),b(Re.$$.fragment,e),b(Ve.$$.fragment,e),b(me.$$.fragment,e),b(ue.$$.fragment,e),b(He.$$.fragment,e),b(Ee.$$.fragment,e),b(Ae.$$.fragment,e),b(he.$$.fragment,e),b(ge.$$.fragment,e),b(Qe.$$.fragment,e),b(Se.$$.fragment,e),b(Ye.$$.fragment,e),b(fe.$$.fragment,e),b(_e.$$.fragment,e),b(Me.$$.fragment,e),b(Pe.$$.fragment,e),b(Oe.$$.fragment,e),b(De.$$.fragment,e),b(be.$$.fragment,e),b(ye.$$.fragment,e),b(Ke.$$.fragment,e),b(et.$$.fragment,e),b(tt.$$.fragment,e),b(Te.$$.fragment,e),b(we.$$.fragment,e),b(nt.$$.fragment,e),b(ot.$$.fragment,e),b(st.$$.fragment,e),b(ve.$$.fragment,e),b(ke.$$.fragment,e),b(at.$$.fragment,e),fn=!1},d(e){e&&(l(h),l(s),l(d),l(T),l(w),l(Nt),l(ie),l(Xt),l(Rt),l(Vt),l(ze),l(Ht),l(Je),l(Et),l(xe),l(At),l(Fe),l(Qt),l(St),l(We),l(Yt),l(Pt),l(Ie),l(Ot),l(Dt),l(q),l(Kt),l(en),l(k),l(tn),l(nn),l(J),l(on),l(sn),l(x),l(an),l(rn),l(F),l(ln),l(dn),l(U),l(cn),l(pn),l(W),l(mn),l(un),l(Z),l(hn),l(gn),l(Lt)),l(t),y(Ce,e),y(le,e),y(je,e),y(Ue,e),y(Ze,e),y(Be,e),y(qe),y(de),y(Ge,e),y(Le),y(Ne),y(ce),y(pe),y(Xe,e),y(Re),y(Ve),y(me),y(ue),y(He,e),y(Ee),y(Ae),y(he),y(ge),y(Qe,e),y(Se),y(Ye),y(fe),y(_e),y(Me),y(Pe,e),y(Oe),y(De),y(be),y(ye),y(Ke,e),y(et),y(tt),y(Te),y(we),y(nt,e),y(ot),y(st),y(ve),y(ke),y(at,e)}}}const Us='{"title":"MEGA","local":"mega","sections":[{"title":"Overview","local":"overview","sections":[],"depth":2},{"title":"Usage tips","local":"usage-tips","sections":[],"depth":2},{"title":"Implementation Notes","local":"implementation-notes","sections":[],"depth":2},{"title":"MegaConfig","local":"transformers.MegaConfig","sections":[],"depth":2},{"title":"MegaModel","local":"transformers.MegaModel","sections":[],"depth":2},{"title":"MegaForCausalLM","local":"transformers.MegaForCausalLM","sections":[],"depth":2},{"title":"MegaForMaskedLM","local":"transformers.MegaForMaskedLM","sections":[],"depth":2},{"title":"MegaForSequenceClassification","local":"transformers.MegaForSequenceClassification","sections":[],"depth":2},{"title":"MegaForMultipleChoice","local":"transformers.MegaForMultipleChoice","sections":[],"depth":2},{"title":"MegaForTokenClassification","local":"transformers.MegaForTokenClassification","sections":[],"depth":2},{"title":"MegaForQuestionAnswering","local":"transformers.MegaForQuestionAnswering","sections":[],"depth":2}],"depth":1}';function Ws(v){return ds(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class Xs extends cs{constructor(t){super(),ps(this,t,Ws,Fs,ls,{})}}export{Xs as component};
