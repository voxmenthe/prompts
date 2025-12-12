import{s as _a,o as Ma,n as J}from"../chunks/scheduler.18a86fab.js";import{S as ba,i as ya,g as p,s as a,r as u,A as Ta,h as m,f as s,c as r,j as B,x as h,u as g,k as v,y as i,a as c,v as f,d as _,t as M,w as b}from"../chunks/index.98837b22.js";import{T as ge}from"../chunks/Tip.77304350.js";import{D as $}from"../chunks/Docstring.a1ef7999.js";import{C as q}from"../chunks/CodeBlock.8d0c2e8a.js";import{E as he}from"../chunks/ExampleCodeBlock.8c3ee1f9.js";import{H as L,E as ka}from"../chunks/getInferenceSnippets.06c2775f.js";function wa(w){let t,y="Examples:",l,d,T;return d=new q({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyME1lZ2F0cm9uQmVydENvbmZpZyUyQyUyME1lZ2F0cm9uQmVydE1vZGVsJTBBJTBBJTIzJTIwSW5pdGlhbGl6aW5nJTIwYSUyME1FR0FUUk9OX0JFUlQlMjBnb29nbGUtYmVydCUyRmJlcnQtYmFzZS11bmNhc2VkJTIwc3R5bGUlMjBjb25maWd1cmF0aW9uJTBBY29uZmlndXJhdGlvbiUyMCUzRCUyME1lZ2F0cm9uQmVydENvbmZpZygpJTBBJTBBJTIzJTIwSW5pdGlhbGl6aW5nJTIwYSUyMG1vZGVsJTIwKHdpdGglMjByYW5kb20lMjB3ZWlnaHRzKSUyMGZyb20lMjB0aGUlMjBnb29nbGUtYmVydCUyRmJlcnQtYmFzZS11bmNhc2VkJTIwc3R5bGUlMjBjb25maWd1cmF0aW9uJTBBbW9kZWwlMjAlM0QlMjBNZWdhdHJvbkJlcnRNb2RlbChjb25maWd1cmF0aW9uKSUwQSUwQSUyMyUyMEFjY2Vzc2luZyUyMHRoZSUyMG1vZGVsJTIwY29uZmlndXJhdGlvbiUwQWNvbmZpZ3VyYXRpb24lMjAlM0QlMjBtb2RlbC5jb25maWc=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> MegatronBertConfig, MegatronBertModel

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a MEGATRON_BERT google-bert/bert-base-uncased style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = MegatronBertConfig()

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a model (with random weights) from the google-bert/bert-base-uncased style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = MegatronBertModel(configuration)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Accessing the model configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = model.config`,wrap:!1}}),{c(){t=p("p"),t.textContent=y,l=a(),u(d.$$.fragment)},l(n){t=m(n,"P",{"data-svelte-h":!0}),h(t)!=="svelte-kvfsh7"&&(t.textContent=y),l=r(n),g(d.$$.fragment,n)},m(n,k){c(n,t,k),c(n,l,k),f(d,n,k),T=!0},p:J,i(n){T||(_(d.$$.fragment,n),T=!0)},o(n){M(d.$$.fragment,n),T=!1},d(n){n&&(s(t),s(l)),b(d,n)}}}function va(w){let t,y=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=p("p"),t.innerHTML=y},l(l){t=m(l,"P",{"data-svelte-h":!0}),h(t)!=="svelte-fincs2"&&(t.innerHTML=y)},m(l,d){c(l,t,d)},p:J,d(l){l&&s(t)}}}function Ba(w){let t,y=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=p("p"),t.innerHTML=y},l(l){t=m(l,"P",{"data-svelte-h":!0}),h(t)!=="svelte-fincs2"&&(t.innerHTML=y)},m(l,d){c(l,t,d)},p:J,d(l){l&&s(t)}}}function $a(w){let t,y="Example:",l,d,T;return d=new q({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBNZWdhdHJvbkJlcnRGb3JNYXNrZWRMTSUwQWltcG9ydCUyMHRvcmNoJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIybnZpZGlhJTJGbWVnYXRyb24tYmVydC11bmNhc2VkLTM0NW0lMjIpJTBBbW9kZWwlMjAlM0QlMjBNZWdhdHJvbkJlcnRGb3JNYXNrZWRMTS5mcm9tX3ByZXRyYWluZWQoJTIybnZpZGlhJTJGbWVnYXRyb24tYmVydC11bmNhc2VkLTM0NW0lMjIpJTBBJTBBaW5wdXRzJTIwJTNEJTIwdG9rZW5pemVyKCUyMlRoZSUyMGNhcGl0YWwlMjBvZiUyMEZyYW5jZSUyMGlzJTIwJTNDbWFzayUzRS4lMjIlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQSUwQXdpdGglMjB0b3JjaC5ub19ncmFkKCklM0ElMEElMjAlMjAlMjAlMjBsb2dpdHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cykubG9naXRzJTBBJTBBJTIzJTIwcmV0cmlldmUlMjBpbmRleCUyMG9mJTIwJTNDbWFzayUzRSUwQW1hc2tfdG9rZW5faW5kZXglMjAlM0QlMjAoaW5wdXRzLmlucHV0X2lkcyUyMCUzRCUzRCUyMHRva2VuaXplci5tYXNrX3Rva2VuX2lkKSU1QjAlNUQubm9uemVybyhhc190dXBsZSUzRFRydWUpJTVCMCU1RCUwQSUwQXByZWRpY3RlZF90b2tlbl9pZCUyMCUzRCUyMGxvZ2l0cyU1QjAlMkMlMjBtYXNrX3Rva2VuX2luZGV4JTVELmFyZ21heChheGlzJTNELTEpJTBBdG9rZW5pemVyLmRlY29kZShwcmVkaWN0ZWRfdG9rZW5faWQpJTBBJTBBbGFiZWxzJTIwJTNEJTIwdG9rZW5pemVyKCUyMlRoZSUyMGNhcGl0YWwlMjBvZiUyMEZyYW5jZSUyMGlzJTIwUGFyaXMuJTIyJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiklNUIlMjJpbnB1dF9pZHMlMjIlNUQlMEElMjMlMjBtYXNrJTIwbGFiZWxzJTIwb2YlMjBub24tJTNDbWFzayUzRSUyMHRva2VucyUwQWxhYmVscyUyMCUzRCUyMHRvcmNoLndoZXJlKGlucHV0cy5pbnB1dF9pZHMlMjAlM0QlM0QlMjB0b2tlbml6ZXIubWFza190b2tlbl9pZCUyQyUyMGxhYmVscyUyQyUyMC0xMDApJTBBJTBBb3V0cHV0cyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzJTJDJTIwbGFiZWxzJTNEbGFiZWxzKSUwQXJvdW5kKG91dHB1dHMubG9zcy5pdGVtKCklMkMlMjAyKQ==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, MegatronBertForMaskedLM
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;nvidia/megatron-bert-uncased-345m&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = MegatronBertForMaskedLM.from_pretrained(<span class="hljs-string">&quot;nvidia/megatron-bert-uncased-345m&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;The capital of France is &lt;mask&gt;.&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    logits = model(**inputs).logits

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># retrieve index of &lt;mask&gt;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[<span class="hljs-number">0</span>].nonzero(as_tuple=<span class="hljs-literal">True</span>)[<span class="hljs-number">0</span>]

<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_token_id = logits[<span class="hljs-number">0</span>, mask_token_index].argmax(axis=-<span class="hljs-number">1</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer.decode(predicted_token_id)
...

<span class="hljs-meta">&gt;&gt;&gt; </span>labels = tokenizer(<span class="hljs-string">&quot;The capital of France is Paris.&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)[<span class="hljs-string">&quot;input_ids&quot;</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># mask labels of non-&lt;mask&gt; tokens</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.where(inputs.input_ids == tokenizer.mask_token_id, labels, -<span class="hljs-number">100</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs, labels=labels)
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">round</span>(outputs.loss.item(), <span class="hljs-number">2</span>)
...`,wrap:!1}}),{c(){t=p("p"),t.textContent=y,l=a(),u(d.$$.fragment)},l(n){t=m(n,"P",{"data-svelte-h":!0}),h(t)!=="svelte-11lpom8"&&(t.textContent=y),l=r(n),g(d.$$.fragment,n)},m(n,k){c(n,t,k),c(n,l,k),f(d,n,k),T=!0},p:J,i(n){T||(_(d.$$.fragment,n),T=!0)},o(n){M(d.$$.fragment,n),T=!1},d(n){n&&(s(t),s(l)),b(d,n)}}}function Ja(w){let t,y=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=p("p"),t.innerHTML=y},l(l){t=m(l,"P",{"data-svelte-h":!0}),h(t)!=="svelte-fincs2"&&(t.innerHTML=y)},m(l,d){c(l,t,d)},p:J,d(l){l&&s(t)}}}function Ca(w){let t,y="Example:",l,d,T;return d=new q({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBNZWdhdHJvbkJlcnRGb3JDYXVzYWxMTSUyQyUyME1lZ2F0cm9uQmVydENvbmZpZyUwQWltcG9ydCUyMHRvcmNoJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIybnZpZGlhJTJGbWVnYXRyb24tYmVydC1jYXNlZC0zNDVtJTIyKSUwQW1vZGVsJTIwJTNEJTIwTWVnYXRyb25CZXJ0Rm9yQ2F1c2FsTE0uZnJvbV9wcmV0cmFpbmVkKCUyMm52aWRpYSUyRm1lZ2F0cm9uLWJlcnQtY2FzZWQtMzQ1bSUyMiUyQyUyMGlzX2RlY29kZXIlM0RUcnVlKSUwQSUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplciglMjJIZWxsbyUyQyUyMG15JTIwZG9nJTIwaXMlMjBjdXRlJTIyJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiklMEFvdXRwdXRzJTIwJTNEJTIwbW9kZWwoKippbnB1dHMpJTBBJTBBcHJlZGljdGlvbl9sb2dpdHMlMjAlM0QlMjBvdXRwdXRzLmxvZ2l0cw==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, MegatronBertForCausalLM, MegatronBertConfig
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;nvidia/megatron-bert-cased-345m&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = MegatronBertForCausalLM.from_pretrained(<span class="hljs-string">&quot;nvidia/megatron-bert-cased-345m&quot;</span>, is_decoder=<span class="hljs-literal">True</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs)

<span class="hljs-meta">&gt;&gt;&gt; </span>prediction_logits = outputs.logits`,wrap:!1}}),{c(){t=p("p"),t.textContent=y,l=a(),u(d.$$.fragment)},l(n){t=m(n,"P",{"data-svelte-h":!0}),h(t)!=="svelte-11lpom8"&&(t.textContent=y),l=r(n),g(d.$$.fragment,n)},m(n,k){c(n,t,k),c(n,l,k),f(d,n,k),T=!0},p:J,i(n){T||(_(d.$$.fragment,n),T=!0)},o(n){M(d.$$.fragment,n),T=!1},d(n){n&&(s(t),s(l)),b(d,n)}}}function ja(w){let t,y=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=p("p"),t.innerHTML=y},l(l){t=m(l,"P",{"data-svelte-h":!0}),h(t)!=="svelte-fincs2"&&(t.innerHTML=y)},m(l,d){c(l,t,d)},p:J,d(l){l&&s(t)}}}function Fa(w){let t,y="Example:",l,d,T;return d=new q({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBNZWdhdHJvbkJlcnRGb3JOZXh0U2VudGVuY2VQcmVkaWN0aW9uJTBBaW1wb3J0JTIwdG9yY2glMEElMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJudmlkaWElMkZtZWdhdHJvbi1iZXJ0LWNhc2VkLTM0NW0lMjIpJTBBbW9kZWwlMjAlM0QlMjBNZWdhdHJvbkJlcnRGb3JOZXh0U2VudGVuY2VQcmVkaWN0aW9uLmZyb21fcHJldHJhaW5lZCglMjJudmlkaWElMkZtZWdhdHJvbi1iZXJ0LWNhc2VkLTM0NW0lMjIpJTBBJTBBcHJvbXB0JTIwJTNEJTIwJTIySW4lMjBJdGFseSUyQyUyMHBpenphJTIwc2VydmVkJTIwaW4lMjBmb3JtYWwlMjBzZXR0aW5ncyUyQyUyMHN1Y2glMjBhcyUyMGF0JTIwYSUyMHJlc3RhdXJhbnQlMkMlMjBpcyUyMHByZXNlbnRlZCUyMHVuc2xpY2VkLiUyMiUwQW5leHRfc2VudGVuY2UlMjAlM0QlMjAlMjJUaGUlMjBza3klMjBpcyUyMGJsdWUlMjBkdWUlMjB0byUyMHRoZSUyMHNob3J0ZXIlMjB3YXZlbGVuZ3RoJTIwb2YlMjBibHVlJTIwbGlnaHQuJTIyJTBBZW5jb2RpbmclMjAlM0QlMjB0b2tlbml6ZXIocHJvbXB0JTJDJTIwbmV4dF9zZW50ZW5jZSUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpJTBBJTBBb3V0cHV0cyUyMCUzRCUyMG1vZGVsKCoqZW5jb2RpbmclMkMlMjBsYWJlbHMlM0R0b3JjaC5Mb25nVGVuc29yKCU1QjElNUQpKSUwQWxvZ2l0cyUyMCUzRCUyMG91dHB1dHMubG9naXRzJTBBYXNzZXJ0JTIwbG9naXRzJTVCMCUyQyUyMDAlNUQlMjAlM0MlMjBsb2dpdHMlNUIwJTJDJTIwMSU1RCUyMCUyMCUyMyUyMG5leHQlMjBzZW50ZW5jZSUyMHdhcyUyMHJhbmRvbQ==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, MegatronBertForNextSentencePrediction
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;nvidia/megatron-bert-cased-345m&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = MegatronBertForNextSentencePrediction.from_pretrained(<span class="hljs-string">&quot;nvidia/megatron-bert-cased-345m&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>prompt = <span class="hljs-string">&quot;In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>next_sentence = <span class="hljs-string">&quot;The sky is blue due to the shorter wavelength of blue light.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>encoding = tokenizer(prompt, next_sentence, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**encoding, labels=torch.LongTensor([<span class="hljs-number">1</span>]))
<span class="hljs-meta">&gt;&gt;&gt; </span>logits = outputs.logits
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">assert</span> logits[<span class="hljs-number">0</span>, <span class="hljs-number">0</span>] &lt; logits[<span class="hljs-number">0</span>, <span class="hljs-number">1</span>]  <span class="hljs-comment"># next sentence was random</span>`,wrap:!1}}),{c(){t=p("p"),t.textContent=y,l=a(),u(d.$$.fragment)},l(n){t=m(n,"P",{"data-svelte-h":!0}),h(t)!=="svelte-11lpom8"&&(t.textContent=y),l=r(n),g(d.$$.fragment,n)},m(n,k){c(n,t,k),c(n,l,k),f(d,n,k),T=!0},p:J,i(n){T||(_(d.$$.fragment,n),T=!0)},o(n){M(d.$$.fragment,n),T=!1},d(n){n&&(s(t),s(l)),b(d,n)}}}function xa(w){let t,y=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=p("p"),t.innerHTML=y},l(l){t=m(l,"P",{"data-svelte-h":!0}),h(t)!=="svelte-fincs2"&&(t.innerHTML=y)},m(l,d){c(l,t,d)},p:J,d(l){l&&s(t)}}}function za(w){let t,y="Example:",l,d,T;return d=new q({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBNZWdhdHJvbkJlcnRGb3JQcmVUcmFpbmluZyUwQWltcG9ydCUyMHRvcmNoJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIybnZpZGlhJTJGbWVnYXRyb24tYmVydC1jYXNlZC0zNDVtJTIyKSUwQW1vZGVsJTIwJTNEJTIwTWVnYXRyb25CZXJ0Rm9yUHJlVHJhaW5pbmcuZnJvbV9wcmV0cmFpbmVkKCUyMm52aWRpYSUyRm1lZ2F0cm9uLWJlcnQtY2FzZWQtMzQ1bSUyMiklMEElMEFpbnB1dHMlMjAlM0QlMjB0b2tlbml6ZXIoJTIySGVsbG8lMkMlMjBteSUyMGRvZyUyMGlzJTIwY3V0ZSUyMiUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpJTBBb3V0cHV0cyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzKSUwQSUwQXByZWRpY3Rpb25fbG9naXRzJTIwJTNEJTIwb3V0cHV0cy5wcmVkaWN0aW9uX2xvZ2l0cyUwQXNlcV9yZWxhdGlvbnNoaXBfbG9naXRzJTIwJTNEJTIwb3V0cHV0cy5zZXFfcmVsYXRpb25zaGlwX2xvZ2l0cw==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, MegatronBertForPreTraining
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;nvidia/megatron-bert-cased-345m&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = MegatronBertForPreTraining.from_pretrained(<span class="hljs-string">&quot;nvidia/megatron-bert-cased-345m&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs)

<span class="hljs-meta">&gt;&gt;&gt; </span>prediction_logits = outputs.prediction_logits
<span class="hljs-meta">&gt;&gt;&gt; </span>seq_relationship_logits = outputs.seq_relationship_logits`,wrap:!1}}),{c(){t=p("p"),t.textContent=y,l=a(),u(d.$$.fragment)},l(n){t=m(n,"P",{"data-svelte-h":!0}),h(t)!=="svelte-11lpom8"&&(t.textContent=y),l=r(n),g(d.$$.fragment,n)},m(n,k){c(n,t,k),c(n,l,k),f(d,n,k),T=!0},p:J,i(n){T||(_(d.$$.fragment,n),T=!0)},o(n){M(d.$$.fragment,n),T=!1},d(n){n&&(s(t),s(l)),b(d,n)}}}function Ua(w){let t,y=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=p("p"),t.innerHTML=y},l(l){t=m(l,"P",{"data-svelte-h":!0}),h(t)!=="svelte-fincs2"&&(t.innerHTML=y)},m(l,d){c(l,t,d)},p:J,d(l){l&&s(t)}}}function Wa(w){let t,y="Example of single-label classification:",l,d,T;return d=new q({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Rva2VuaXplciUyQyUyME1lZ2F0cm9uQmVydEZvclNlcXVlbmNlQ2xhc3NpZmljYXRpb24lMEElMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJudmlkaWElMkZtZWdhdHJvbi1iZXJ0LXVuY2FzZWQtMzQ1bSUyMiklMEFtb2RlbCUyMCUzRCUyME1lZ2F0cm9uQmVydEZvclNlcXVlbmNlQ2xhc3NpZmljYXRpb24uZnJvbV9wcmV0cmFpbmVkKCUyMm52aWRpYSUyRm1lZ2F0cm9uLWJlcnQtdW5jYXNlZC0zNDVtJTIyKSUwQSUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplciglMjJIZWxsbyUyQyUyMG15JTIwZG9nJTIwaXMlMjBjdXRlJTIyJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiklMEElMEF3aXRoJTIwdG9yY2gubm9fZ3JhZCgpJTNBJTBBJTIwJTIwJTIwJTIwbG9naXRzJTIwJTNEJTIwbW9kZWwoKippbnB1dHMpLmxvZ2l0cyUwQSUwQXByZWRpY3RlZF9jbGFzc19pZCUyMCUzRCUyMGxvZ2l0cy5hcmdtYXgoKS5pdGVtKCklMEFtb2RlbC5jb25maWcuaWQybGFiZWwlNUJwcmVkaWN0ZWRfY2xhc3NfaWQlNUQlMEElMEElMjMlMjBUbyUyMHRyYWluJTIwYSUyMG1vZGVsJTIwb24lMjAlNjBudW1fbGFiZWxzJTYwJTIwY2xhc3NlcyUyQyUyMHlvdSUyMGNhbiUyMHBhc3MlMjAlNjBudW1fbGFiZWxzJTNEbnVtX2xhYmVscyU2MCUyMHRvJTIwJTYwLmZyb21fcHJldHJhaW5lZCguLi4pJTYwJTBBbnVtX2xhYmVscyUyMCUzRCUyMGxlbihtb2RlbC5jb25maWcuaWQybGFiZWwpJTBBbW9kZWwlMjAlM0QlMjBNZWdhdHJvbkJlcnRGb3JTZXF1ZW5jZUNsYXNzaWZpY2F0aW9uLmZyb21fcHJldHJhaW5lZCglMjJudmlkaWElMkZtZWdhdHJvbi1iZXJ0LXVuY2FzZWQtMzQ1bSUyMiUyQyUyMG51bV9sYWJlbHMlM0RudW1fbGFiZWxzKSUwQSUwQWxhYmVscyUyMCUzRCUyMHRvcmNoLnRlbnNvciglNUIxJTVEKSUwQWxvc3MlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyUyQyUyMGxhYmVscyUzRGxhYmVscykubG9zcyUwQXJvdW5kKGxvc3MuaXRlbSgpJTJDJTIwMik=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, MegatronBertForSequenceClassification

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;nvidia/megatron-bert-uncased-345m&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = MegatronBertForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;nvidia/megatron-bert-uncased-345m&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    logits = model(**inputs).logits

<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_class_id = logits.argmax().item()
<span class="hljs-meta">&gt;&gt;&gt; </span>model.config.id2label[predicted_class_id]
...

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># To train a model on \`num_labels\` classes, you can pass \`num_labels=num_labels\` to \`.from_pretrained(...)\`</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>num_labels = <span class="hljs-built_in">len</span>(model.config.id2label)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = MegatronBertForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;nvidia/megatron-bert-uncased-345m&quot;</span>, num_labels=num_labels)

<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.tensor([<span class="hljs-number">1</span>])
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = model(**inputs, labels=labels).loss
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">round</span>(loss.item(), <span class="hljs-number">2</span>)
...`,wrap:!1}}),{c(){t=p("p"),t.textContent=y,l=a(),u(d.$$.fragment)},l(n){t=m(n,"P",{"data-svelte-h":!0}),h(t)!=="svelte-ykxpe4"&&(t.textContent=y),l=r(n),g(d.$$.fragment,n)},m(n,k){c(n,t,k),c(n,l,k),f(d,n,k),T=!0},p:J,i(n){T||(_(d.$$.fragment,n),T=!0)},o(n){M(d.$$.fragment,n),T=!1},d(n){n&&(s(t),s(l)),b(d,n)}}}function Za(w){let t,y="Example of multi-label classification:",l,d,T;return d=new q({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Rva2VuaXplciUyQyUyME1lZ2F0cm9uQmVydEZvclNlcXVlbmNlQ2xhc3NpZmljYXRpb24lMEElMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJudmlkaWElMkZtZWdhdHJvbi1iZXJ0LXVuY2FzZWQtMzQ1bSUyMiklMEFtb2RlbCUyMCUzRCUyME1lZ2F0cm9uQmVydEZvclNlcXVlbmNlQ2xhc3NpZmljYXRpb24uZnJvbV9wcmV0cmFpbmVkKCUyMm52aWRpYSUyRm1lZ2F0cm9uLWJlcnQtdW5jYXNlZC0zNDVtJTIyJTJDJTIwcHJvYmxlbV90eXBlJTNEJTIybXVsdGlfbGFiZWxfY2xhc3NpZmljYXRpb24lMjIpJTBBJTBBaW5wdXRzJTIwJTNEJTIwdG9rZW5pemVyKCUyMkhlbGxvJTJDJTIwbXklMjBkb2clMjBpcyUyMGN1dGUlMjIlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQSUwQXdpdGglMjB0b3JjaC5ub19ncmFkKCklM0ElMEElMjAlMjAlMjAlMjBsb2dpdHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cykubG9naXRzJTBBJTBBcHJlZGljdGVkX2NsYXNzX2lkcyUyMCUzRCUyMHRvcmNoLmFyYW5nZSgwJTJDJTIwbG9naXRzLnNoYXBlJTVCLTElNUQpJTVCdG9yY2guc2lnbW9pZChsb2dpdHMpLnNxdWVlemUoZGltJTNEMCklMjAlM0UlMjAwLjUlNUQlMEElMEElMjMlMjBUbyUyMHRyYWluJTIwYSUyMG1vZGVsJTIwb24lMjAlNjBudW1fbGFiZWxzJTYwJTIwY2xhc3NlcyUyQyUyMHlvdSUyMGNhbiUyMHBhc3MlMjAlNjBudW1fbGFiZWxzJTNEbnVtX2xhYmVscyU2MCUyMHRvJTIwJTYwLmZyb21fcHJldHJhaW5lZCguLi4pJTYwJTBBbnVtX2xhYmVscyUyMCUzRCUyMGxlbihtb2RlbC5jb25maWcuaWQybGFiZWwpJTBBbW9kZWwlMjAlM0QlMjBNZWdhdHJvbkJlcnRGb3JTZXF1ZW5jZUNsYXNzaWZpY2F0aW9uLmZyb21fcHJldHJhaW5lZCglMEElMjAlMjAlMjAlMjAlMjJudmlkaWElMkZtZWdhdHJvbi1iZXJ0LXVuY2FzZWQtMzQ1bSUyMiUyQyUyMG51bV9sYWJlbHMlM0RudW1fbGFiZWxzJTJDJTIwcHJvYmxlbV90eXBlJTNEJTIybXVsdGlfbGFiZWxfY2xhc3NpZmljYXRpb24lMjIlMEEpJTBBJTBBbGFiZWxzJTIwJTNEJTIwdG9yY2guc3VtKCUwQSUyMCUyMCUyMCUyMHRvcmNoLm5uLmZ1bmN0aW9uYWwub25lX2hvdChwcmVkaWN0ZWRfY2xhc3NfaWRzJTVCTm9uZSUyQyUyMCUzQSU1RC5jbG9uZSgpJTJDJTIwbnVtX2NsYXNzZXMlM0RudW1fbGFiZWxzKSUyQyUyMGRpbSUzRDElMEEpLnRvKHRvcmNoLmZsb2F0KSUwQWxvc3MlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyUyQyUyMGxhYmVscyUzRGxhYmVscykubG9zcw==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, MegatronBertForSequenceClassification

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;nvidia/megatron-bert-uncased-345m&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = MegatronBertForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;nvidia/megatron-bert-uncased-345m&quot;</span>, problem_type=<span class="hljs-string">&quot;multi_label_classification&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    logits = model(**inputs).logits

<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_class_ids = torch.arange(<span class="hljs-number">0</span>, logits.shape[-<span class="hljs-number">1</span>])[torch.sigmoid(logits).squeeze(dim=<span class="hljs-number">0</span>) &gt; <span class="hljs-number">0.5</span>]

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># To train a model on \`num_labels\` classes, you can pass \`num_labels=num_labels\` to \`.from_pretrained(...)\`</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>num_labels = <span class="hljs-built_in">len</span>(model.config.id2label)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = MegatronBertForSequenceClassification.from_pretrained(
<span class="hljs-meta">... </span>    <span class="hljs-string">&quot;nvidia/megatron-bert-uncased-345m&quot;</span>, num_labels=num_labels, problem_type=<span class="hljs-string">&quot;multi_label_classification&quot;</span>
<span class="hljs-meta">... </span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.<span class="hljs-built_in">sum</span>(
<span class="hljs-meta">... </span>    torch.nn.functional.one_hot(predicted_class_ids[<span class="hljs-literal">None</span>, :].clone(), num_classes=num_labels), dim=<span class="hljs-number">1</span>
<span class="hljs-meta">... </span>).to(torch.<span class="hljs-built_in">float</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = model(**inputs, labels=labels).loss`,wrap:!1}}),{c(){t=p("p"),t.textContent=y,l=a(),u(d.$$.fragment)},l(n){t=m(n,"P",{"data-svelte-h":!0}),h(t)!=="svelte-1l8e32d"&&(t.textContent=y),l=r(n),g(d.$$.fragment,n)},m(n,k){c(n,t,k),c(n,l,k),f(d,n,k),T=!0},p:J,i(n){T||(_(d.$$.fragment,n),T=!0)},o(n){M(d.$$.fragment,n),T=!1},d(n){n&&(s(t),s(l)),b(d,n)}}}function Na(w){let t,y=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=p("p"),t.innerHTML=y},l(l){t=m(l,"P",{"data-svelte-h":!0}),h(t)!=="svelte-fincs2"&&(t.innerHTML=y)},m(l,d){c(l,t,d)},p:J,d(l){l&&s(t)}}}function Ia(w){let t,y="Example:",l,d,T;return d=new q({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBNZWdhdHJvbkJlcnRGb3JNdWx0aXBsZUNob2ljZSUwQWltcG9ydCUyMHRvcmNoJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIybnZpZGlhJTJGbWVnYXRyb24tYmVydC11bmNhc2VkLTM0NW0lMjIpJTBBbW9kZWwlMjAlM0QlMjBNZWdhdHJvbkJlcnRGb3JNdWx0aXBsZUNob2ljZS5mcm9tX3ByZXRyYWluZWQoJTIybnZpZGlhJTJGbWVnYXRyb24tYmVydC11bmNhc2VkLTM0NW0lMjIpJTBBJTBBcHJvbXB0JTIwJTNEJTIwJTIySW4lMjBJdGFseSUyQyUyMHBpenphJTIwc2VydmVkJTIwaW4lMjBmb3JtYWwlMjBzZXR0aW5ncyUyQyUyMHN1Y2glMjBhcyUyMGF0JTIwYSUyMHJlc3RhdXJhbnQlMkMlMjBpcyUyMHByZXNlbnRlZCUyMHVuc2xpY2VkLiUyMiUwQWNob2ljZTAlMjAlM0QlMjAlMjJJdCUyMGlzJTIwZWF0ZW4lMjB3aXRoJTIwYSUyMGZvcmslMjBhbmQlMjBhJTIwa25pZmUuJTIyJTBBY2hvaWNlMSUyMCUzRCUyMCUyMkl0JTIwaXMlMjBlYXRlbiUyMHdoaWxlJTIwaGVsZCUyMGluJTIwdGhlJTIwaGFuZC4lMjIlMEFsYWJlbHMlMjAlM0QlMjB0b3JjaC50ZW5zb3IoMCkudW5zcXVlZXplKDApJTIwJTIwJTIzJTIwY2hvaWNlMCUyMGlzJTIwY29ycmVjdCUyMChhY2NvcmRpbmclMjB0byUyMFdpa2lwZWRpYSUyMCUzQikpJTJDJTIwYmF0Y2glMjBzaXplJTIwMSUwQSUwQWVuY29kaW5nJTIwJTNEJTIwdG9rZW5pemVyKCU1QnByb21wdCUyQyUyMHByb21wdCU1RCUyQyUyMCU1QmNob2ljZTAlMkMlMjBjaG9pY2UxJTVEJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiUyQyUyMHBhZGRpbmclM0RUcnVlKSUwQW91dHB1dHMlMjAlM0QlMjBtb2RlbCgqKiU3QmslM0ElMjB2LnVuc3F1ZWV6ZSgwKSUyMGZvciUyMGslMkMlMjB2JTIwaW4lMjBlbmNvZGluZy5pdGVtcygpJTdEJTJDJTIwbGFiZWxzJTNEbGFiZWxzKSUyMCUyMCUyMyUyMGJhdGNoJTIwc2l6ZSUyMGlzJTIwMSUwQSUwQSUyMyUyMHRoZSUyMGxpbmVhciUyMGNsYXNzaWZpZXIlMjBzdGlsbCUyMG5lZWRzJTIwdG8lMjBiZSUyMHRyYWluZWQlMEFsb3NzJTIwJTNEJTIwb3V0cHV0cy5sb3NzJTBBbG9naXRzJTIwJTNEJTIwb3V0cHV0cy5sb2dpdHM=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, MegatronBertForMultipleChoice
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;nvidia/megatron-bert-uncased-345m&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = MegatronBertForMultipleChoice.from_pretrained(<span class="hljs-string">&quot;nvidia/megatron-bert-uncased-345m&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>prompt = <span class="hljs-string">&quot;In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>choice0 = <span class="hljs-string">&quot;It is eaten with a fork and a knife.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>choice1 = <span class="hljs-string">&quot;It is eaten while held in the hand.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.tensor(<span class="hljs-number">0</span>).unsqueeze(<span class="hljs-number">0</span>)  <span class="hljs-comment"># choice0 is correct (according to Wikipedia ;)), batch size 1</span>

<span class="hljs-meta">&gt;&gt;&gt; </span>encoding = tokenizer([prompt, prompt], [choice0, choice1], return_tensors=<span class="hljs-string">&quot;pt&quot;</span>, padding=<span class="hljs-literal">True</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**{k: v.unsqueeze(<span class="hljs-number">0</span>) <span class="hljs-keyword">for</span> k, v <span class="hljs-keyword">in</span> encoding.items()}, labels=labels)  <span class="hljs-comment"># batch size is 1</span>

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># the linear classifier still needs to be trained</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = outputs.loss
<span class="hljs-meta">&gt;&gt;&gt; </span>logits = outputs.logits`,wrap:!1}}),{c(){t=p("p"),t.textContent=y,l=a(),u(d.$$.fragment)},l(n){t=m(n,"P",{"data-svelte-h":!0}),h(t)!=="svelte-11lpom8"&&(t.textContent=y),l=r(n),g(d.$$.fragment,n)},m(n,k){c(n,t,k),c(n,l,k),f(d,n,k),T=!0},p:J,i(n){T||(_(d.$$.fragment,n),T=!0)},o(n){M(d.$$.fragment,n),T=!1},d(n){n&&(s(t),s(l)),b(d,n)}}}function La(w){let t,y=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=p("p"),t.innerHTML=y},l(l){t=m(l,"P",{"data-svelte-h":!0}),h(t)!=="svelte-fincs2"&&(t.innerHTML=y)},m(l,d){c(l,t,d)},p:J,d(l){l&&s(t)}}}function qa(w){let t,y="Example:",l,d,T;return d=new q({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBNZWdhdHJvbkJlcnRGb3JUb2tlbkNsYXNzaWZpY2F0aW9uJTBBaW1wb3J0JTIwdG9yY2glMEElMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJudmlkaWElMkZtZWdhdHJvbi1iZXJ0LXVuY2FzZWQtMzQ1bSUyMiklMEFtb2RlbCUyMCUzRCUyME1lZ2F0cm9uQmVydEZvclRva2VuQ2xhc3NpZmljYXRpb24uZnJvbV9wcmV0cmFpbmVkKCUyMm52aWRpYSUyRm1lZ2F0cm9uLWJlcnQtdW5jYXNlZC0zNDVtJTIyKSUwQSUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplciglMEElMjAlMjAlMjAlMjAlMjJIdWdnaW5nRmFjZSUyMGlzJTIwYSUyMGNvbXBhbnklMjBiYXNlZCUyMGluJTIwUGFyaXMlMjBhbmQlMjBOZXclMjBZb3JrJTIyJTJDJTIwYWRkX3NwZWNpYWxfdG9rZW5zJTNERmFsc2UlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyJTBBKSUwQSUwQXdpdGglMjB0b3JjaC5ub19ncmFkKCklM0ElMEElMjAlMjAlMjAlMjBsb2dpdHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cykubG9naXRzJTBBJTBBcHJlZGljdGVkX3Rva2VuX2NsYXNzX2lkcyUyMCUzRCUyMGxvZ2l0cy5hcmdtYXgoLTEpJTBBJTBBJTIzJTIwTm90ZSUyMHRoYXQlMjB0b2tlbnMlMjBhcmUlMjBjbGFzc2lmaWVkJTIwcmF0aGVyJTIwdGhlbiUyMGlucHV0JTIwd29yZHMlMjB3aGljaCUyMG1lYW5zJTIwdGhhdCUwQSUyMyUyMHRoZXJlJTIwbWlnaHQlMjBiZSUyMG1vcmUlMjBwcmVkaWN0ZWQlMjB0b2tlbiUyMGNsYXNzZXMlMjB0aGFuJTIwd29yZHMuJTBBJTIzJTIwTXVsdGlwbGUlMjB0b2tlbiUyMGNsYXNzZXMlMjBtaWdodCUyMGFjY291bnQlMjBmb3IlMjB0aGUlMjBzYW1lJTIwd29yZCUwQXByZWRpY3RlZF90b2tlbnNfY2xhc3NlcyUyMCUzRCUyMCU1Qm1vZGVsLmNvbmZpZy5pZDJsYWJlbCU1QnQuaXRlbSgpJTVEJTIwZm9yJTIwdCUyMGluJTIwcHJlZGljdGVkX3Rva2VuX2NsYXNzX2lkcyU1QjAlNUQlNUQlMEFwcmVkaWN0ZWRfdG9rZW5zX2NsYXNzZXMlMEElMEFsYWJlbHMlMjAlM0QlMjBwcmVkaWN0ZWRfdG9rZW5fY2xhc3NfaWRzJTBBbG9zcyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzJTJDJTIwbGFiZWxzJTNEbGFiZWxzKS5sb3NzJTBBcm91bmQobG9zcy5pdGVtKCklMkMlMjAyKQ==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, MegatronBertForTokenClassification
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;nvidia/megatron-bert-uncased-345m&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = MegatronBertForTokenClassification.from_pretrained(<span class="hljs-string">&quot;nvidia/megatron-bert-uncased-345m&quot;</span>)

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
<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_tokens_classes
...

<span class="hljs-meta">&gt;&gt;&gt; </span>labels = predicted_token_class_ids
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = model(**inputs, labels=labels).loss
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">round</span>(loss.item(), <span class="hljs-number">2</span>)
...`,wrap:!1}}),{c(){t=p("p"),t.textContent=y,l=a(),u(d.$$.fragment)},l(n){t=m(n,"P",{"data-svelte-h":!0}),h(t)!=="svelte-11lpom8"&&(t.textContent=y),l=r(n),g(d.$$.fragment,n)},m(n,k){c(n,t,k),c(n,l,k),f(d,n,k),T=!0},p:J,i(n){T||(_(d.$$.fragment,n),T=!0)},o(n){M(d.$$.fragment,n),T=!1},d(n){n&&(s(t),s(l)),b(d,n)}}}function Ra(w){let t,y=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=p("p"),t.innerHTML=y},l(l){t=m(l,"P",{"data-svelte-h":!0}),h(t)!=="svelte-fincs2"&&(t.innerHTML=y)},m(l,d){c(l,t,d)},p:J,d(l){l&&s(t)}}}function Va(w){let t,y="Example:",l,d,T;return d=new q({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBNZWdhdHJvbkJlcnRGb3JRdWVzdGlvbkFuc3dlcmluZyUwQWltcG9ydCUyMHRvcmNoJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIybnZpZGlhJTJGbWVnYXRyb24tYmVydC11bmNhc2VkLTM0NW0lMjIpJTBBbW9kZWwlMjAlM0QlMjBNZWdhdHJvbkJlcnRGb3JRdWVzdGlvbkFuc3dlcmluZy5mcm9tX3ByZXRyYWluZWQoJTIybnZpZGlhJTJGbWVnYXRyb24tYmVydC11bmNhc2VkLTM0NW0lMjIpJTBBJTBBcXVlc3Rpb24lMkMlMjB0ZXh0JTIwJTNEJTIwJTIyV2hvJTIwd2FzJTIwSmltJTIwSGVuc29uJTNGJTIyJTJDJTIwJTIySmltJTIwSGVuc29uJTIwd2FzJTIwYSUyMG5pY2UlMjBwdXBwZXQlMjIlMEElMEFpbnB1dHMlMjAlM0QlMjB0b2tlbml6ZXIocXVlc3Rpb24lMkMlMjB0ZXh0JTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiklMEF3aXRoJTIwdG9yY2gubm9fZ3JhZCgpJTNBJTBBJTIwJTIwJTIwJTIwb3V0cHV0cyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzKSUwQSUwQWFuc3dlcl9zdGFydF9pbmRleCUyMCUzRCUyMG91dHB1dHMuc3RhcnRfbG9naXRzLmFyZ21heCgpJTBBYW5zd2VyX2VuZF9pbmRleCUyMCUzRCUyMG91dHB1dHMuZW5kX2xvZ2l0cy5hcmdtYXgoKSUwQSUwQXByZWRpY3RfYW5zd2VyX3Rva2VucyUyMCUzRCUyMGlucHV0cy5pbnB1dF9pZHMlNUIwJTJDJTIwYW5zd2VyX3N0YXJ0X2luZGV4JTIwJTNBJTIwYW5zd2VyX2VuZF9pbmRleCUyMCUyQiUyMDElNUQlMEF0b2tlbml6ZXIuZGVjb2RlKHByZWRpY3RfYW5zd2VyX3Rva2VucyUyQyUyMHNraXBfc3BlY2lhbF90b2tlbnMlM0RUcnVlKSUwQSUwQSUyMyUyMHRhcmdldCUyMGlzJTIwJTIybmljZSUyMHB1cHBldCUyMiUwQXRhcmdldF9zdGFydF9pbmRleCUyMCUzRCUyMHRvcmNoLnRlbnNvciglNUIxNCU1RCklMEF0YXJnZXRfZW5kX2luZGV4JTIwJTNEJTIwdG9yY2gudGVuc29yKCU1QjE1JTVEKSUwQSUwQW91dHB1dHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyUyQyUyMHN0YXJ0X3Bvc2l0aW9ucyUzRHRhcmdldF9zdGFydF9pbmRleCUyQyUyMGVuZF9wb3NpdGlvbnMlM0R0YXJnZXRfZW5kX2luZGV4KSUwQWxvc3MlMjAlM0QlMjBvdXRwdXRzLmxvc3MlMEFyb3VuZChsb3NzLml0ZW0oKSUyQyUyMDIp",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, MegatronBertForQuestionAnswering
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;nvidia/megatron-bert-uncased-345m&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = MegatronBertForQuestionAnswering.from_pretrained(<span class="hljs-string">&quot;nvidia/megatron-bert-uncased-345m&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>question, text = <span class="hljs-string">&quot;Who was Jim Henson?&quot;</span>, <span class="hljs-string">&quot;Jim Henson was a nice puppet&quot;</span>

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(question, text, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    outputs = model(**inputs)

<span class="hljs-meta">&gt;&gt;&gt; </span>answer_start_index = outputs.start_logits.argmax()
<span class="hljs-meta">&gt;&gt;&gt; </span>answer_end_index = outputs.end_logits.argmax()

<span class="hljs-meta">&gt;&gt;&gt; </span>predict_answer_tokens = inputs.input_ids[<span class="hljs-number">0</span>, answer_start_index : answer_end_index + <span class="hljs-number">1</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer.decode(predict_answer_tokens, skip_special_tokens=<span class="hljs-literal">True</span>)
...

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># target is &quot;nice puppet&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>target_start_index = torch.tensor([<span class="hljs-number">14</span>])
<span class="hljs-meta">&gt;&gt;&gt; </span>target_end_index = torch.tensor([<span class="hljs-number">15</span>])

<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs, start_positions=target_start_index, end_positions=target_end_index)
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = outputs.loss
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">round</span>(loss.item(), <span class="hljs-number">2</span>)
...`,wrap:!1}}),{c(){t=p("p"),t.textContent=y,l=a(),u(d.$$.fragment)},l(n){t=m(n,"P",{"data-svelte-h":!0}),h(t)!=="svelte-11lpom8"&&(t.textContent=y),l=r(n),g(d.$$.fragment,n)},m(n,k){c(n,t,k),c(n,l,k),f(d,n,k),T=!0},p:J,i(n){T||(_(d.$$.fragment,n),T=!0)},o(n){M(d.$$.fragment,n),T=!1},d(n){n&&(s(t),s(l)),b(d,n)}}}function Ga(w){let t,y,l,d,T,n="<em>This model was released on 2019-09-17 and added to Hugging Face Transformers on 2021-04-08.</em>",k,Ne,Mn,fe,ys='<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/>',bn,Ie,yn,Le,Ts=`The MegatronBERT model was proposed in <a href="https://huggingface.co/papers/1909.08053" rel="nofollow">Megatron-LM: Training Multi-Billion Parameter Language Models Using Model
Parallelism</a> by Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley,
Jared Casper and Bryan Catanzaro.`,Tn,qe,ks="The abstract from the paper is the following:",kn,Re,ws=`<em>Recent work in language modeling demonstrates that training large transformer models advances the state of the art in
Natural Language Processing applications. However, very large models can be quite difficult to train due to memory
constraints. In this work, we present our techniques for training very large transformer models and implement a simple,
efficient intra-layer model parallel approach that enables training transformer models with billions of parameters. Our
approach does not require a new compiler or library changes, is orthogonal and complimentary to pipeline model
parallelism, and can be fully implemented with the insertion of a few communication operations in native PyTorch. We
illustrate this approach by converging transformer based models up to 8.3 billion parameters using 512 GPUs. We sustain
15.1 PetaFLOPs across the entire application with 76% scaling efficiency when compared to a strong single GPU baseline
that sustains 39 TeraFLOPs, which is 30% of peak FLOPs. To demonstrate that large language models can further advance
the state of the art (SOTA), we train an 8.3 billion parameter transformer language model similar to GPT-2 and a 3.9
billion parameter model similar to BERT. We show that careful attention to the placement of layer normalization in
BERT-like models is critical to achieving increased performance as the model size grows. Using the GPT-2 model we
achieve SOTA results on the WikiText103 (10.8 compared to SOTA perplexity of 15.8) and LAMBADA (66.5% compared to SOTA
accuracy of 63.2%) datasets. Our BERT model achieves SOTA results on the RACE dataset (90.9% compared to SOTA accuracy
of 89.4%).</em>`,wn,Ve,vs=`This model was contributed by <a href="https://huggingface.co/jdemouth" rel="nofollow">jdemouth</a>. The original code can be found <a href="https://github.com/NVIDIA/Megatron-LM" rel="nofollow">here</a>.
That repository contains a multi-GPU and multi-node implementation of the Megatron Language models. In particular,
it contains a hybrid model parallel approach using “tensor parallel” and “pipeline parallel” techniques.`,vn,Ge,Bn,He,Bs=`We have provided pretrained <a href="https://ngc.nvidia.com/catalog/models/nvidia:megatron_bert_345m" rel="nofollow">BERT-345M</a> checkpoints
for use to evaluate or finetuning downstream tasks.`,$n,Xe,$s=`To access these checkpoints, first <a href="https://ngc.nvidia.com/signup" rel="nofollow">sign up</a> for and setup the NVIDIA GPU Cloud (NGC)
Registry CLI. Further documentation for downloading models can be found in the <a href="https://docs.nvidia.com/dgx/ngc-registry-cli-user-guide/index.html#topic_6_4_1" rel="nofollow">NGC documentation</a>.`,Jn,Se,Js="Alternatively, you can directly download the checkpoints using:",Cn,Pe,Cs="BERT-345M-uncased:",jn,Qe,Fn,Ye,js="BERT-345M-cased:",xn,Ee,zn,Ae,Fs=`Once you have obtained the checkpoints from NVIDIA GPU Cloud (NGC), you have to convert them to a format that will
easily be loaded by Hugging Face Transformers and our port of the BERT code.`,Un,Oe,xs=`The following commands allow you to do the conversion. We assume that the folder <code>models/megatron_bert</code> contains
<code>megatron_bert_345m_v0_1_{cased, uncased}.zip</code> and that the commands are run from inside that folder:`,Wn,De,Zn,Ke,Nn,et,In,tt,zs='<li><a href="../tasks/sequence_classification">Text classification task guide</a></li> <li><a href="../tasks/token_classification">Token classification task guide</a></li> <li><a href="../tasks/question_answering">Question answering task guide</a></li> <li><a href="../tasks/language_modeling">Causal language modeling task guide</a></li> <li><a href="../tasks/masked_language_modeling">Masked language modeling task guide</a></li> <li><a href="../tasks/multiple_choice">Multiple choice task guide</a></li>',Ln,nt,qn,R,ot,io,zt,Us=`This is the configuration class to store the configuration of a <a href="/docs/transformers/v4.56.2/en/model_doc/megatron-bert#transformers.MegatronBertModel">MegatronBertModel</a>. It is used to instantiate a
MEGATRON_BERT model according to the specified arguments, defining the model architecture. Instantiating a
configuration with the defaults will yield a similar configuration to that of the MEGATRON_BERT
<a href="https://huggingface.co/nvidia/megatron-bert-uncased-345m" rel="nofollow">nvidia/megatron-bert-uncased-345m</a> architecture.`,lo,Ut,Ws=`Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> and can be used to control the model outputs. Read the
documentation from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> for more information.`,co,_e,Rn,st,Vn,C,at,po,Wt,Zs="The bare Megatron Bert Model outputting raw hidden-states without any specific head on top.",mo,Zt,Ns=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,ho,Nt,Is=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,uo,me,rt,go,It,Ls='The <a href="/docs/transformers/v4.56.2/en/model_doc/megatron-bert#transformers.MegatronBertModel">MegatronBertModel</a> forward method, overrides the <code>__call__</code> special method.',fo,Me,Gn,it,Hn,j,lt,_o,Lt,qs="The Megatron Bert Model with a <code>language modeling</code> head on top.”",Mo,qt,Rs=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,bo,Rt,Vs=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,yo,O,dt,To,Vt,Gs='The <a href="/docs/transformers/v4.56.2/en/model_doc/megatron-bert#transformers.MegatronBertForMaskedLM">MegatronBertForMaskedLM</a> forward method, overrides the <code>__call__</code> special method.',ko,be,wo,ye,Xn,ct,Sn,F,pt,vo,Gt,Hs="MegatronBert Model with a <code>language modeling</code> head on top for CLM fine-tuning.",Bo,Ht,Xs=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,$o,Xt,Ss=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Jo,D,mt,Co,St,Ps='The <a href="/docs/transformers/v4.56.2/en/model_doc/megatron-bert#transformers.MegatronBertForCausalLM">MegatronBertForCausalLM</a> forward method, overrides the <code>__call__</code> special method.',jo,Te,Fo,ke,Pn,ht,Qn,x,ut,xo,Pt,Qs="MegatronBert Model with a <code>next sentence prediction (classification)</code> head on top.",zo,Qt,Ys=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Uo,Yt,Es=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Wo,K,gt,Zo,Et,As='The <a href="/docs/transformers/v4.56.2/en/model_doc/megatron-bert#transformers.MegatronBertForNextSentencePrediction">MegatronBertForNextSentencePrediction</a> forward method, overrides the <code>__call__</code> special method.',No,we,Io,ve,Yn,ft,En,z,_t,Lo,At,Os=`MegatronBert Model with two heads on top as done during the pretraining: a <code>masked language modeling</code> head and a
<code>next sentence prediction (classification)</code> head.`,qo,Ot,Ds=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Ro,Dt,Ks=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Vo,ee,Mt,Go,Kt,ea='The <a href="/docs/transformers/v4.56.2/en/model_doc/megatron-bert#transformers.MegatronBertForPreTraining">MegatronBertForPreTraining</a> forward method, overrides the <code>__call__</code> special method.',Ho,Be,Xo,$e,An,bt,On,U,yt,So,en,ta=`MegatronBert Model transformer with a sequence classification/regression head on top (a linear layer on top of the
pooled output) e.g. for GLUE tasks.`,Po,tn,na=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Qo,nn,oa=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Yo,I,Tt,Eo,on,sa='The <a href="/docs/transformers/v4.56.2/en/model_doc/megatron-bert#transformers.MegatronBertForSequenceClassification">MegatronBertForSequenceClassification</a> forward method, overrides the <code>__call__</code> special method.',Ao,Je,Oo,Ce,Do,je,Dn,kt,Kn,W,wt,Ko,sn,aa=`The Megatron Bert Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
softmax) e.g. for RocStories/SWAG tasks.`,es,an,ra=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,ts,rn,ia=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,ns,te,vt,os,ln,la='The <a href="/docs/transformers/v4.56.2/en/model_doc/megatron-bert#transformers.MegatronBertForMultipleChoice">MegatronBertForMultipleChoice</a> forward method, overrides the <code>__call__</code> special method.',ss,Fe,as,xe,eo,Bt,to,Z,$t,rs,dn,da=`The Megatron Bert transformer with a token classification head on top (a linear layer on top of the hidden-states
output) e.g. for Named-Entity-Recognition (NER) tasks.`,is,cn,ca=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,ls,pn,pa=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,ds,ne,Jt,cs,mn,ma='The <a href="/docs/transformers/v4.56.2/en/model_doc/megatron-bert#transformers.MegatronBertForTokenClassification">MegatronBertForTokenClassification</a> forward method, overrides the <code>__call__</code> special method.',ps,ze,ms,Ue,no,Ct,oo,N,jt,hs,hn,ha=`The Megatron Bert transformer with a span classification head on top for extractive question-answering tasks like
SQuAD (a linear layer on top of the hidden-states output to compute <code>span start logits</code> and <code>span end logits</code>).`,us,un,ua=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,gs,gn,ga=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,fs,oe,Ft,_s,fn,fa='The <a href="/docs/transformers/v4.56.2/en/model_doc/megatron-bert#transformers.MegatronBertForQuestionAnswering">MegatronBertForQuestionAnswering</a> forward method, overrides the <code>__call__</code> special method.',Ms,We,bs,Ze,so,xt,ao,_n,ro;return Ne=new L({props:{title:"MegatronBERT",local:"megatronbert",headingTag:"h1"}}),Ie=new L({props:{title:"Overview",local:"overview",headingTag:"h2"}}),Ge=new L({props:{title:"Usage tips",local:"usage-tips",headingTag:"h2"}}),Qe=new q({props:{code:"d2dldCUyMC0tY29udGVudC1kaXNwb3NpdGlvbiUyMGh0dHBzJTNBJTJGJTJGYXBpLm5nYy5udmlkaWEuY29tJTJGdjIlMkZtb2RlbHMlMkZudmlkaWElMkZtZWdhdHJvbl9iZXJ0XzM0NW0lMkZ2ZXJzaW9ucyUyRnYwLjFfdW5jYXNlZCUyRnppcCUwQS1PJTIwbWVnYXRyb25fYmVydF8zNDVtX3YwXzFfdW5jYXNlZC56aXA=",highlighted:`wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/megatron_bert_345m/versions/v0.1_uncased/zip
-O megatron_bert_345m_v0_1_uncased.zip`,wrap:!1}}),Ee=new q({props:{code:"d2dldCUyMC0tY29udGVudC1kaXNwb3NpdGlvbiUyMGh0dHBzJTNBJTJGJTJGYXBpLm5nYy5udmlkaWEuY29tJTJGdjIlMkZtb2RlbHMlMkZudmlkaWElMkZtZWdhdHJvbl9iZXJ0XzM0NW0lMkZ2ZXJzaW9ucyUyRnYwLjFfY2FzZWQlMkZ6aXAlMjAtTyUwQW1lZ2F0cm9uX2JlcnRfMzQ1bV92MF8xX2Nhc2VkLnppcA==",highlighted:`wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/megatron_bert_345m/versions/v0.1_cased/zip -O
megatron_bert_345m_v0_1_cased.zip`,wrap:!1}}),De=new q({props:{code:"cHl0aG9uMyUyMCUyNFBBVEhfVE9fVFJBTlNGT1JNRVJTJTJGbW9kZWxzJTJGbWVnYXRyb25fYmVydCUyRmNvbnZlcnRfbWVnYXRyb25fYmVydF9jaGVja3BvaW50LnB5JTIwbWVnYXRyb25fYmVydF8zNDVtX3YwXzFfdW5jYXNlZC56aXA=",highlighted:'python3 <span class="hljs-variable">$PATH_TO_TRANSFORMERS</span>/models/megatron_bert/convert_megatron_bert_checkpoint.py megatron_bert_345m_v0_1_uncased.zip',wrap:!1}}),Ke=new q({props:{code:"cHl0aG9uMyUyMCUyNFBBVEhfVE9fVFJBTlNGT1JNRVJTJTJGbW9kZWxzJTJGbWVnYXRyb25fYmVydCUyRmNvbnZlcnRfbWVnYXRyb25fYmVydF9jaGVja3BvaW50LnB5JTIwbWVnYXRyb25fYmVydF8zNDVtX3YwXzFfY2FzZWQuemlw",highlighted:'python3 <span class="hljs-variable">$PATH_TO_TRANSFORMERS</span>/models/megatron_bert/convert_megatron_bert_checkpoint.py megatron_bert_345m_v0_1_cased.zip',wrap:!1}}),et=new L({props:{title:"Resources",local:"resources",headingTag:"h2"}}),nt=new L({props:{title:"MegatronBertConfig",local:"transformers.MegatronBertConfig",headingTag:"h2"}}),ot=new $({props:{name:"class transformers.MegatronBertConfig",anchor:"transformers.MegatronBertConfig",parameters:[{name:"vocab_size",val:" = 29056"},{name:"hidden_size",val:" = 1024"},{name:"num_hidden_layers",val:" = 24"},{name:"num_attention_heads",val:" = 16"},{name:"intermediate_size",val:" = 4096"},{name:"hidden_act",val:" = 'gelu'"},{name:"hidden_dropout_prob",val:" = 0.1"},{name:"attention_probs_dropout_prob",val:" = 0.1"},{name:"max_position_embeddings",val:" = 512"},{name:"type_vocab_size",val:" = 2"},{name:"initializer_range",val:" = 0.02"},{name:"layer_norm_eps",val:" = 1e-12"},{name:"pad_token_id",val:" = 0"},{name:"position_embedding_type",val:" = 'absolute'"},{name:"use_cache",val:" = True"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.MegatronBertConfig.vocab_size",description:`<strong>vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 29056) &#x2014;
Vocabulary size of the MEGATRON_BERT model. Defines the number of different tokens that can be represented
by the <code>inputs_ids</code> passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/megatron-bert#transformers.MegatronBertModel">MegatronBertModel</a>.`,name:"vocab_size"},{anchor:"transformers.MegatronBertConfig.hidden_size",description:`<strong>hidden_size</strong> (<code>int</code>, <em>optional</em>, defaults to 1024) &#x2014;
Dimensionality of the encoder layers and the pooler layer.`,name:"hidden_size"},{anchor:"transformers.MegatronBertConfig.num_hidden_layers",description:`<strong>num_hidden_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 24) &#x2014;
Number of hidden layers in the Transformer encoder.`,name:"num_hidden_layers"},{anchor:"transformers.MegatronBertConfig.num_attention_heads",description:`<strong>num_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 16) &#x2014;
Number of attention heads for each attention layer in the Transformer encoder.`,name:"num_attention_heads"},{anchor:"transformers.MegatronBertConfig.intermediate_size",description:`<strong>intermediate_size</strong> (<code>int</code>, <em>optional</em>, defaults to 4096) &#x2014;
Dimensionality of the &#x201C;intermediate&#x201D; (often named feed-forward) layer in the Transformer encoder.`,name:"intermediate_size"},{anchor:"transformers.MegatronBertConfig.hidden_act",description:`<strong>hidden_act</strong> (<code>str</code> or <code>Callable</code>, <em>optional</em>, defaults to <code>&quot;gelu&quot;</code>) &#x2014;
The non-linear activation function (function or string) in the encoder and pooler. If string, <code>&quot;gelu&quot;</code>,
<code>&quot;relu&quot;</code>, <code>&quot;silu&quot;</code> and <code>&quot;gelu_new&quot;</code> are supported.`,name:"hidden_act"},{anchor:"transformers.MegatronBertConfig.hidden_dropout_prob",description:`<strong>hidden_dropout_prob</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.`,name:"hidden_dropout_prob"},{anchor:"transformers.MegatronBertConfig.attention_probs_dropout_prob",description:`<strong>attention_probs_dropout_prob</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout ratio for the attention probabilities.`,name:"attention_probs_dropout_prob"},{anchor:"transformers.MegatronBertConfig.max_position_embeddings",description:`<strong>max_position_embeddings</strong> (<code>int</code>, <em>optional</em>, defaults to 512) &#x2014;
The maximum sequence length that this model might ever be used with. Typically set this to something large
just in case (e.g., 512 or 1024 or 2048).`,name:"max_position_embeddings"},{anchor:"transformers.MegatronBertConfig.type_vocab_size",description:`<strong>type_vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 2) &#x2014;
The vocabulary size of the <code>token_type_ids</code> passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/megatron-bert#transformers.MegatronBertModel">MegatronBertModel</a>.`,name:"type_vocab_size"},{anchor:"transformers.MegatronBertConfig.initializer_range",description:`<strong>initializer_range</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.`,name:"initializer_range"},{anchor:"transformers.MegatronBertConfig.layer_norm_eps",description:`<strong>layer_norm_eps</strong> (<code>float</code>, <em>optional</em>, defaults to 1e-12) &#x2014;
The epsilon used by the layer normalization layers.`,name:"layer_norm_eps"},{anchor:"transformers.MegatronBertConfig.position_embedding_type",description:`<strong>position_embedding_type</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;absolute&quot;</code>) &#x2014;
Type of position embedding. Choose one of <code>&quot;absolute&quot;</code>, <code>&quot;relative_key&quot;</code>, <code>&quot;relative_key_query&quot;</code>. For
positional embeddings use <code>&quot;absolute&quot;</code>. For more information on <code>&quot;relative_key&quot;</code>, please refer to
<a href="https://huggingface.co/papers/1803.02155" rel="nofollow">Self-Attention with Relative Position Representations (Shaw et al.)</a>.
For more information on <code>&quot;relative_key_query&quot;</code>, please refer to <em>Method 4</em> in <a href="https://huggingface.co/papers/2009.13658" rel="nofollow">Improve Transformer Models
with Better Relative Position Embeddings (Huang et al.)</a>.`,name:"position_embedding_type"},{anchor:"transformers.MegatronBertConfig.is_decoder",description:`<strong>is_decoder</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether the model is used as a decoder or not. If <code>False</code>, the model is used as an encoder.`,name:"is_decoder"},{anchor:"transformers.MegatronBertConfig.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not the model should return the last key/values attentions (not used by all models). Only
relevant if <code>config.is_decoder=True</code>.`,name:"use_cache"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/megatron_bert/configuration_megatron_bert.py#L24"}}),_e=new he({props:{anchor:"transformers.MegatronBertConfig.example",$$slots:{default:[wa]},$$scope:{ctx:w}}}),st=new L({props:{title:"MegatronBertModel",local:"transformers.MegatronBertModel",headingTag:"h2"}}),at=new $({props:{name:"class transformers.MegatronBertModel",anchor:"transformers.MegatronBertModel",parameters:[{name:"config",val:""},{name:"add_pooling_layer",val:" = True"}],parametersDescription:[{anchor:"transformers.MegatronBertModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/megatron-bert#transformers.MegatronBertModel">MegatronBertModel</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"},{anchor:"transformers.MegatronBertModel.add_pooling_layer",description:`<strong>add_pooling_layer</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether to add a pooling layer`,name:"add_pooling_layer"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/megatron_bert/modeling_megatron_bert.py#L721"}}),rt=new $({props:{name:"forward",anchor:"transformers.MegatronBertModel.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"encoder_hidden_states",val:": typing.Optional[torch.FloatTensor] = None"},{name:"encoder_attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"past_key_values",val:": typing.Optional[tuple[tuple[torch.Tensor]]] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.Tensor] = None"}],parametersDescription:[{anchor:"transformers.MegatronBertModel.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.MegatronBertModel.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.MegatronBertModel.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.MegatronBertModel.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.MegatronBertModel.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.MegatronBertModel.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.MegatronBertModel.forward.encoder_hidden_states",description:`<strong>encoder_hidden_states</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
if the model is configured as a decoder.`,name:"encoder_hidden_states"},{anchor:"transformers.MegatronBertModel.forward.encoder_attention_mask",description:`<strong>encoder_attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
the cross-attention if the model is configured as a decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>`,name:"encoder_attention_mask"},{anchor:"transformers.MegatronBertModel.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>tuple[tuple[torch.Tensor]]</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.MegatronBertModel.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.MegatronBertModel.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.MegatronBertModel.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.MegatronBertModel.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.MegatronBertModel.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.Tensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/megatron_bert/modeling_megatron_bert.py#L764",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions"
>transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/megatron-bert#transformers.MegatronBertConfig"
>MegatronBertConfig</a>) and inputs.</p>
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
`}}),Me=new ge({props:{$$slots:{default:[va]},$$scope:{ctx:w}}}),it=new L({props:{title:"MegatronBertForMaskedLM",local:"transformers.MegatronBertForMaskedLM",headingTag:"h2"}}),lt=new $({props:{name:"class transformers.MegatronBertForMaskedLM",anchor:"transformers.MegatronBertForMaskedLM",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.MegatronBertForMaskedLM.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/megatron-bert#transformers.MegatronBertForMaskedLM">MegatronBertForMaskedLM</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/megatron_bert/modeling_megatron_bert.py#L1100"}}),dt=new $({props:{name:"forward",anchor:"transformers.MegatronBertForMaskedLM.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"encoder_hidden_states",val:": typing.Optional[torch.FloatTensor] = None"},{name:"encoder_attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.MegatronBertForMaskedLM.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.MegatronBertForMaskedLM.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.MegatronBertForMaskedLM.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.MegatronBertForMaskedLM.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.MegatronBertForMaskedLM.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.MegatronBertForMaskedLM.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.MegatronBertForMaskedLM.forward.encoder_hidden_states",description:`<strong>encoder_hidden_states</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
if the model is configured as a decoder.`,name:"encoder_hidden_states"},{anchor:"transformers.MegatronBertForMaskedLM.forward.encoder_attention_mask",description:`<strong>encoder_attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
the cross-attention if the model is configured as a decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>`,name:"encoder_attention_mask"},{anchor:"transformers.MegatronBertForMaskedLM.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should be in <code>[-100, 0, ..., config.vocab_size]</code> (see <code>input_ids</code> docstring) Tokens with indices set to <code>-100</code> are ignored (masked), the
loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>`,name:"labels"},{anchor:"transformers.MegatronBertForMaskedLM.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.MegatronBertForMaskedLM.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.MegatronBertForMaskedLM.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/megatron_bert/modeling_megatron_bert.py#L1125",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.MaskedLMOutput"
>transformers.modeling_outputs.MaskedLMOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/megatron-bert#transformers.MegatronBertConfig"
>MegatronBertConfig</a>) and inputs.</p>
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
`}}),be=new ge({props:{$$slots:{default:[Ba]},$$scope:{ctx:w}}}),ye=new he({props:{anchor:"transformers.MegatronBertForMaskedLM.forward.example",$$slots:{default:[$a]},$$scope:{ctx:w}}}),ct=new L({props:{title:"MegatronBertForCausalLM",local:"transformers.MegatronBertForCausalLM",headingTag:"h2"}}),pt=new $({props:{name:"class transformers.MegatronBertForCausalLM",anchor:"transformers.MegatronBertForCausalLM",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.MegatronBertForCausalLM.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/megatron-bert#transformers.MegatronBertForCausalLM">MegatronBertForCausalLM</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/megatron_bert/modeling_megatron_bert.py#L990"}}),mt=new $({props:{name:"forward",anchor:"transformers.MegatronBertForCausalLM.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"encoder_hidden_states",val:": typing.Optional[torch.FloatTensor] = None"},{name:"encoder_attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[tuple[tuple[torch.Tensor]]] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.Tensor] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.MegatronBertForCausalLM.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.MegatronBertForCausalLM.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.MegatronBertForCausalLM.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.MegatronBertForCausalLM.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.MegatronBertForCausalLM.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.MegatronBertForCausalLM.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.MegatronBertForCausalLM.forward.encoder_hidden_states",description:`<strong>encoder_hidden_states</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
if the model is configured as a decoder.`,name:"encoder_hidden_states"},{anchor:"transformers.MegatronBertForCausalLM.forward.encoder_attention_mask",description:`<strong>encoder_attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
the cross-attention if the model is configured as a decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>`,name:"encoder_attention_mask"},{anchor:"transformers.MegatronBertForCausalLM.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the left-to-right language modeling loss (next word prediction). Indices should be in
<code>[-100, 0, ..., config.vocab_size]</code> (see <code>input_ids</code> docstring) Tokens with indices set to <code>-100</code> are
ignored (masked), the loss is only computed for the tokens with labels n <code>[0, ..., config.vocab_size]</code>`,name:"labels"},{anchor:"transformers.MegatronBertForCausalLM.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>tuple[tuple[torch.Tensor]]</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.MegatronBertForCausalLM.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.MegatronBertForCausalLM.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.MegatronBertForCausalLM.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.MegatronBertForCausalLM.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.MegatronBertForCausalLM.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.Tensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/megatron_bert/modeling_megatron_bert.py#L1012",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithCrossAttentions"
>transformers.modeling_outputs.CausalLMOutputWithCrossAttentions</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/megatron-bert#transformers.MegatronBertConfig"
>MegatronBertConfig</a>) and inputs.</p>
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
`}}),Te=new ge({props:{$$slots:{default:[Ja]},$$scope:{ctx:w}}}),ke=new he({props:{anchor:"transformers.MegatronBertForCausalLM.forward.example",$$slots:{default:[Ca]},$$scope:{ctx:w}}}),ht=new L({props:{title:"MegatronBertForNextSentencePrediction",local:"transformers.MegatronBertForNextSentencePrediction",headingTag:"h2"}}),ut=new $({props:{name:"class transformers.MegatronBertForNextSentencePrediction",anchor:"transformers.MegatronBertForNextSentencePrediction",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.MegatronBertForNextSentencePrediction.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/megatron-bert#transformers.MegatronBertForNextSentencePrediction">MegatronBertForNextSentencePrediction</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/megatron_bert/modeling_megatron_bert.py#L1204"}}),gt=new $({props:{name:"forward",anchor:"transformers.MegatronBertForNextSentencePrediction.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.MegatronBertForNextSentencePrediction.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.MegatronBertForNextSentencePrediction.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.MegatronBertForNextSentencePrediction.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.MegatronBertForNextSentencePrediction.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.MegatronBertForNextSentencePrediction.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.MegatronBertForNextSentencePrediction.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.MegatronBertForNextSentencePrediction.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the next sequence prediction (classification) loss. Input should be a sequence pair
(see <code>input_ids</code> docstring). Indices should be in <code>[0, 1]</code>:</p>
<ul>
<li>0 indicates sequence B is a continuation of sequence A,</li>
<li>1 indicates sequence B is a random sequence.</li>
</ul>`,name:"labels"},{anchor:"transformers.MegatronBertForNextSentencePrediction.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.MegatronBertForNextSentencePrediction.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.MegatronBertForNextSentencePrediction.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/megatron_bert/modeling_megatron_bert.py#L1214",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.NextSentencePredictorOutput"
>transformers.modeling_outputs.NextSentencePredictorOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/megatron-bert#transformers.MegatronBertConfig"
>MegatronBertConfig</a>) and inputs.</p>
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
`}}),we=new ge({props:{$$slots:{default:[ja]},$$scope:{ctx:w}}}),ve=new he({props:{anchor:"transformers.MegatronBertForNextSentencePrediction.forward.example",$$slots:{default:[Fa]},$$scope:{ctx:w}}}),ft=new L({props:{title:"MegatronBertForPreTraining",local:"transformers.MegatronBertForPreTraining",headingTag:"h2"}}),_t=new $({props:{name:"class transformers.MegatronBertForPreTraining",anchor:"transformers.MegatronBertForPreTraining",parameters:[{name:"config",val:""},{name:"add_binary_head",val:" = True"}],parametersDescription:[{anchor:"transformers.MegatronBertForPreTraining.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/megatron-bert#transformers.MegatronBertForPreTraining">MegatronBertForPreTraining</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"},{anchor:"transformers.MegatronBertForPreTraining.add_binary_head",description:`<strong>add_binary_head</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to add a binary head.`,name:"add_binary_head"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/megatron_bert/modeling_megatron_bert.py#L883"}}),Mt=new $({props:{name:"forward",anchor:"transformers.MegatronBertForPreTraining.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"next_sentence_label",val:": typing.Optional[torch.LongTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.MegatronBertForPreTraining.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.MegatronBertForPreTraining.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.MegatronBertForPreTraining.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.MegatronBertForPreTraining.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.MegatronBertForPreTraining.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.MegatronBertForPreTraining.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.MegatronBertForPreTraining.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should be in <code>[-100, 0, ..., config.vocab_size]</code> (see <code>input_ids</code> docstring) Tokens with indices set to <code>-100</code> are ignored (masked), the
loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>`,name:"labels"},{anchor:"transformers.MegatronBertForPreTraining.forward.next_sentence_label",description:`<strong>next_sentence_label</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the next sequence prediction (classification) loss. Input should be a sequence pair
(see <code>input_ids</code> docstring) Indices should be in <code>[0, 1]</code>:</p>
<ul>
<li>0 indicates sequence B is a continuation of sequence A,</li>
<li>1 indicates sequence B is a random sequence.</li>
</ul>`,name:"next_sentence_label"},{anchor:"transformers.MegatronBertForPreTraining.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.MegatronBertForPreTraining.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.MegatronBertForPreTraining.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/megatron_bert/modeling_megatron_bert.py#L906",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <code>transformers.models.megatron_bert.modeling_megatron_bert.MegatronBertForPreTrainingOutput</code> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/megatron-bert#transformers.MegatronBertConfig"
>MegatronBertConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>loss</strong> (<code>*optional*</code>, returned when <code>labels</code> is provided, <code>torch.FloatTensor</code> of shape <code>(1,)</code>) — Total loss as the sum of the masked language modeling loss and the next sequence prediction
(classification) loss.</p>
</li>
<li>
<p><strong>prediction_logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, config.vocab_size)</code>) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).</p>
</li>
<li>
<p><strong>seq_relationship_logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, 2)</code>) — Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation
before SoftMax).</p>
</li>
<li>
<p><strong>hidden_states</strong> (<code>tuple[torch.FloatTensor]</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings, if the model has an embedding layer, +
one for the output of each layer) of shape <code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.</p>
</li>
<li>
<p><strong>attentions</strong> (<code>tuple[torch.FloatTensor]</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
heads.</p>
</li>
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>transformers.models.megatron_bert.modeling_megatron_bert.MegatronBertForPreTrainingOutput</code> or <code>tuple(torch.FloatTensor)</code></p>
`}}),Be=new ge({props:{$$slots:{default:[xa]},$$scope:{ctx:w}}}),$e=new he({props:{anchor:"transformers.MegatronBertForPreTraining.forward.example",$$slots:{default:[za]},$$scope:{ctx:w}}}),bt=new L({props:{title:"MegatronBertForSequenceClassification",local:"transformers.MegatronBertForSequenceClassification",headingTag:"h2"}}),yt=new $({props:{name:"class transformers.MegatronBertForSequenceClassification",anchor:"transformers.MegatronBertForSequenceClassification",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.MegatronBertForSequenceClassification.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/megatron-bert#transformers.MegatronBertForSequenceClassification">MegatronBertForSequenceClassification</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/megatron_bert/modeling_megatron_bert.py#L1304"}}),Tt=new $({props:{name:"forward",anchor:"transformers.MegatronBertForSequenceClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.MegatronBertForSequenceClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.MegatronBertForSequenceClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.MegatronBertForSequenceClassification.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.MegatronBertForSequenceClassification.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.MegatronBertForSequenceClassification.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.MegatronBertForSequenceClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.MegatronBertForSequenceClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the sequence classification/regression loss. Indices should be in <code>[0, ..., config.num_labels - 1]</code>. If <code>config.num_labels == 1</code> a regression loss is computed (Mean-Square loss), If
<code>config.num_labels &gt; 1</code> a classification loss is computed (Cross-Entropy).`,name:"labels"},{anchor:"transformers.MegatronBertForSequenceClassification.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.MegatronBertForSequenceClassification.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.MegatronBertForSequenceClassification.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/megatron_bert/modeling_megatron_bert.py#L1316",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput"
>transformers.modeling_outputs.SequenceClassifierOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/megatron-bert#transformers.MegatronBertConfig"
>MegatronBertConfig</a>) and inputs.</p>
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
`}}),Je=new ge({props:{$$slots:{default:[Ua]},$$scope:{ctx:w}}}),Ce=new he({props:{anchor:"transformers.MegatronBertForSequenceClassification.forward.example",$$slots:{default:[Wa]},$$scope:{ctx:w}}}),je=new he({props:{anchor:"transformers.MegatronBertForSequenceClassification.forward.example-2",$$slots:{default:[Za]},$$scope:{ctx:w}}}),kt=new L({props:{title:"MegatronBertForMultipleChoice",local:"transformers.MegatronBertForMultipleChoice",headingTag:"h2"}}),wt=new $({props:{name:"class transformers.MegatronBertForMultipleChoice",anchor:"transformers.MegatronBertForMultipleChoice",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.MegatronBertForMultipleChoice.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/megatron-bert#transformers.MegatronBertForMultipleChoice">MegatronBertForMultipleChoice</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/megatron_bert/modeling_megatron_bert.py#L1390"}}),vt=new $({props:{name:"forward",anchor:"transformers.MegatronBertForMultipleChoice.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.MegatronBertForMultipleChoice.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, num_choices, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.MegatronBertForMultipleChoice.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.MegatronBertForMultipleChoice.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, num_choices, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.MegatronBertForMultipleChoice.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, num_choices, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.max_position_embeddings - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.MegatronBertForMultipleChoice.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.MegatronBertForMultipleChoice.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_choices, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.MegatronBertForMultipleChoice.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the multiple choice classification loss. Indices should be in <code>[0, ..., num_choices-1]</code> where <code>num_choices</code> is the size of the second dimension of the input tensors. (See
<code>input_ids</code> above)`,name:"labels"},{anchor:"transformers.MegatronBertForMultipleChoice.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.MegatronBertForMultipleChoice.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.MegatronBertForMultipleChoice.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/megatron_bert/modeling_megatron_bert.py#L1401",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.MultipleChoiceModelOutput"
>transformers.modeling_outputs.MultipleChoiceModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/megatron-bert#transformers.MegatronBertConfig"
>MegatronBertConfig</a>) and inputs.</p>
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
`}}),Fe=new ge({props:{$$slots:{default:[Na]},$$scope:{ctx:w}}}),xe=new he({props:{anchor:"transformers.MegatronBertForMultipleChoice.forward.example",$$slots:{default:[Ia]},$$scope:{ctx:w}}}),Bt=new L({props:{title:"MegatronBertForTokenClassification",local:"transformers.MegatronBertForTokenClassification",headingTag:"h2"}}),$t=new $({props:{name:"class transformers.MegatronBertForTokenClassification",anchor:"transformers.MegatronBertForTokenClassification",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.MegatronBertForTokenClassification.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/megatron-bert#transformers.MegatronBertForTokenClassification">MegatronBertForTokenClassification</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/megatron_bert/modeling_megatron_bert.py#L1494"}}),Jt=new $({props:{name:"forward",anchor:"transformers.MegatronBertForTokenClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.MegatronBertForTokenClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.MegatronBertForTokenClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.MegatronBertForTokenClassification.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.MegatronBertForTokenClassification.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.MegatronBertForTokenClassification.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.MegatronBertForTokenClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.MegatronBertForTokenClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the token classification loss. Indices should be in <code>[0, ..., config.num_labels - 1]</code>.`,name:"labels"},{anchor:"transformers.MegatronBertForTokenClassification.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.MegatronBertForTokenClassification.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.MegatronBertForTokenClassification.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/megatron_bert/modeling_megatron_bert.py#L1506",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput"
>transformers.modeling_outputs.TokenClassifierOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/megatron-bert#transformers.MegatronBertConfig"
>MegatronBertConfig</a>) and inputs.</p>
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
`}}),ze=new ge({props:{$$slots:{default:[La]},$$scope:{ctx:w}}}),Ue=new he({props:{anchor:"transformers.MegatronBertForTokenClassification.forward.example",$$slots:{default:[qa]},$$scope:{ctx:w}}}),Ct=new L({props:{title:"MegatronBertForQuestionAnswering",local:"transformers.MegatronBertForQuestionAnswering",headingTag:"h2"}}),jt=new $({props:{name:"class transformers.MegatronBertForQuestionAnswering",anchor:"transformers.MegatronBertForQuestionAnswering",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.MegatronBertForQuestionAnswering.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/megatron-bert#transformers.MegatronBertForQuestionAnswering">MegatronBertForQuestionAnswering</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/megatron_bert/modeling_megatron_bert.py#L1561"}}),Ft=new $({props:{name:"forward",anchor:"transformers.MegatronBertForQuestionAnswering.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"start_positions",val:": typing.Optional[torch.LongTensor] = None"},{name:"end_positions",val:": typing.Optional[torch.LongTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.MegatronBertForQuestionAnswering.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.MegatronBertForQuestionAnswering.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.MegatronBertForQuestionAnswering.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.MegatronBertForQuestionAnswering.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.MegatronBertForQuestionAnswering.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.MegatronBertForQuestionAnswering.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.MegatronBertForQuestionAnswering.forward.start_positions",description:`<strong>start_positions</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for position (index) of the start of the labelled span for computing the token classification loss.
Positions are clamped to the length of the sequence (<code>sequence_length</code>). Position outside of the sequence
are not taken into account for computing the loss.`,name:"start_positions"},{anchor:"transformers.MegatronBertForQuestionAnswering.forward.end_positions",description:`<strong>end_positions</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for position (index) of the end of the labelled span for computing the token classification loss.
Positions are clamped to the length of the sequence (<code>sequence_length</code>). Position outside of the sequence
are not taken into account for computing the loss.`,name:"end_positions"},{anchor:"transformers.MegatronBertForQuestionAnswering.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.MegatronBertForQuestionAnswering.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.MegatronBertForQuestionAnswering.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/megatron_bert/modeling_megatron_bert.py#L1572",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.QuestionAnsweringModelOutput"
>transformers.modeling_outputs.QuestionAnsweringModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/megatron-bert#transformers.MegatronBertConfig"
>MegatronBertConfig</a>) and inputs.</p>
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
`}}),We=new ge({props:{$$slots:{default:[Ra]},$$scope:{ctx:w}}}),Ze=new he({props:{anchor:"transformers.MegatronBertForQuestionAnswering.forward.example",$$slots:{default:[Va]},$$scope:{ctx:w}}}),xt=new ka({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/megatron-bert.md"}}),{c(){t=p("meta"),y=a(),l=p("p"),d=a(),T=p("p"),T.innerHTML=n,k=a(),u(Ne.$$.fragment),Mn=a(),fe=p("div"),fe.innerHTML=ys,bn=a(),u(Ie.$$.fragment),yn=a(),Le=p("p"),Le.innerHTML=Ts,Tn=a(),qe=p("p"),qe.textContent=ks,kn=a(),Re=p("p"),Re.innerHTML=ws,wn=a(),Ve=p("p"),Ve.innerHTML=vs,vn=a(),u(Ge.$$.fragment),Bn=a(),He=p("p"),He.innerHTML=Bs,$n=a(),Xe=p("p"),Xe.innerHTML=$s,Jn=a(),Se=p("p"),Se.textContent=Js,Cn=a(),Pe=p("p"),Pe.textContent=Cs,jn=a(),u(Qe.$$.fragment),Fn=a(),Ye=p("p"),Ye.textContent=js,xn=a(),u(Ee.$$.fragment),zn=a(),Ae=p("p"),Ae.textContent=Fs,Un=a(),Oe=p("p"),Oe.innerHTML=xs,Wn=a(),u(De.$$.fragment),Zn=a(),u(Ke.$$.fragment),Nn=a(),u(et.$$.fragment),In=a(),tt=p("ul"),tt.innerHTML=zs,Ln=a(),u(nt.$$.fragment),qn=a(),R=p("div"),u(ot.$$.fragment),io=a(),zt=p("p"),zt.innerHTML=Us,lo=a(),Ut=p("p"),Ut.innerHTML=Ws,co=a(),u(_e.$$.fragment),Rn=a(),u(st.$$.fragment),Vn=a(),C=p("div"),u(at.$$.fragment),po=a(),Wt=p("p"),Wt.textContent=Zs,mo=a(),Zt=p("p"),Zt.innerHTML=Ns,ho=a(),Nt=p("p"),Nt.innerHTML=Is,uo=a(),me=p("div"),u(rt.$$.fragment),go=a(),It=p("p"),It.innerHTML=Ls,fo=a(),u(Me.$$.fragment),Gn=a(),u(it.$$.fragment),Hn=a(),j=p("div"),u(lt.$$.fragment),_o=a(),Lt=p("p"),Lt.innerHTML=qs,Mo=a(),qt=p("p"),qt.innerHTML=Rs,bo=a(),Rt=p("p"),Rt.innerHTML=Vs,yo=a(),O=p("div"),u(dt.$$.fragment),To=a(),Vt=p("p"),Vt.innerHTML=Gs,ko=a(),u(be.$$.fragment),wo=a(),u(ye.$$.fragment),Xn=a(),u(ct.$$.fragment),Sn=a(),F=p("div"),u(pt.$$.fragment),vo=a(),Gt=p("p"),Gt.innerHTML=Hs,Bo=a(),Ht=p("p"),Ht.innerHTML=Xs,$o=a(),Xt=p("p"),Xt.innerHTML=Ss,Jo=a(),D=p("div"),u(mt.$$.fragment),Co=a(),St=p("p"),St.innerHTML=Ps,jo=a(),u(Te.$$.fragment),Fo=a(),u(ke.$$.fragment),Pn=a(),u(ht.$$.fragment),Qn=a(),x=p("div"),u(ut.$$.fragment),xo=a(),Pt=p("p"),Pt.innerHTML=Qs,zo=a(),Qt=p("p"),Qt.innerHTML=Ys,Uo=a(),Yt=p("p"),Yt.innerHTML=Es,Wo=a(),K=p("div"),u(gt.$$.fragment),Zo=a(),Et=p("p"),Et.innerHTML=As,No=a(),u(we.$$.fragment),Io=a(),u(ve.$$.fragment),Yn=a(),u(ft.$$.fragment),En=a(),z=p("div"),u(_t.$$.fragment),Lo=a(),At=p("p"),At.innerHTML=Os,qo=a(),Ot=p("p"),Ot.innerHTML=Ds,Ro=a(),Dt=p("p"),Dt.innerHTML=Ks,Vo=a(),ee=p("div"),u(Mt.$$.fragment),Go=a(),Kt=p("p"),Kt.innerHTML=ea,Ho=a(),u(Be.$$.fragment),Xo=a(),u($e.$$.fragment),An=a(),u(bt.$$.fragment),On=a(),U=p("div"),u(yt.$$.fragment),So=a(),en=p("p"),en.textContent=ta,Po=a(),tn=p("p"),tn.innerHTML=na,Qo=a(),nn=p("p"),nn.innerHTML=oa,Yo=a(),I=p("div"),u(Tt.$$.fragment),Eo=a(),on=p("p"),on.innerHTML=sa,Ao=a(),u(Je.$$.fragment),Oo=a(),u(Ce.$$.fragment),Do=a(),u(je.$$.fragment),Dn=a(),u(kt.$$.fragment),Kn=a(),W=p("div"),u(wt.$$.fragment),Ko=a(),sn=p("p"),sn.textContent=aa,es=a(),an=p("p"),an.innerHTML=ra,ts=a(),rn=p("p"),rn.innerHTML=ia,ns=a(),te=p("div"),u(vt.$$.fragment),os=a(),ln=p("p"),ln.innerHTML=la,ss=a(),u(Fe.$$.fragment),as=a(),u(xe.$$.fragment),eo=a(),u(Bt.$$.fragment),to=a(),Z=p("div"),u($t.$$.fragment),rs=a(),dn=p("p"),dn.textContent=da,is=a(),cn=p("p"),cn.innerHTML=ca,ls=a(),pn=p("p"),pn.innerHTML=pa,ds=a(),ne=p("div"),u(Jt.$$.fragment),cs=a(),mn=p("p"),mn.innerHTML=ma,ps=a(),u(ze.$$.fragment),ms=a(),u(Ue.$$.fragment),no=a(),u(Ct.$$.fragment),oo=a(),N=p("div"),u(jt.$$.fragment),hs=a(),hn=p("p"),hn.innerHTML=ha,us=a(),un=p("p"),un.innerHTML=ua,gs=a(),gn=p("p"),gn.innerHTML=ga,fs=a(),oe=p("div"),u(Ft.$$.fragment),_s=a(),fn=p("p"),fn.innerHTML=fa,Ms=a(),u(We.$$.fragment),bs=a(),u(Ze.$$.fragment),so=a(),u(xt.$$.fragment),ao=a(),_n=p("p"),this.h()},l(e){const o=Ta("svelte-u9bgzb",document.head);t=m(o,"META",{name:!0,content:!0}),o.forEach(s),y=r(e),l=m(e,"P",{}),B(l).forEach(s),d=r(e),T=m(e,"P",{"data-svelte-h":!0}),h(T)!=="svelte-3ne5b6"&&(T.innerHTML=n),k=r(e),g(Ne.$$.fragment,e),Mn=r(e),fe=m(e,"DIV",{class:!0,"data-svelte-h":!0}),h(fe)!=="svelte-13t8s2t"&&(fe.innerHTML=ys),bn=r(e),g(Ie.$$.fragment,e),yn=r(e),Le=m(e,"P",{"data-svelte-h":!0}),h(Le)!=="svelte-u89tvj"&&(Le.innerHTML=Ts),Tn=r(e),qe=m(e,"P",{"data-svelte-h":!0}),h(qe)!=="svelte-vfdo9a"&&(qe.textContent=ks),kn=r(e),Re=m(e,"P",{"data-svelte-h":!0}),h(Re)!=="svelte-o6jluj"&&(Re.innerHTML=ws),wn=r(e),Ve=m(e,"P",{"data-svelte-h":!0}),h(Ve)!=="svelte-1sa9zwk"&&(Ve.innerHTML=vs),vn=r(e),g(Ge.$$.fragment,e),Bn=r(e),He=m(e,"P",{"data-svelte-h":!0}),h(He)!=="svelte-1bsd70l"&&(He.innerHTML=Bs),$n=r(e),Xe=m(e,"P",{"data-svelte-h":!0}),h(Xe)!=="svelte-1xz4v1w"&&(Xe.innerHTML=$s),Jn=r(e),Se=m(e,"P",{"data-svelte-h":!0}),h(Se)!=="svelte-115r3rj"&&(Se.textContent=Js),Cn=r(e),Pe=m(e,"P",{"data-svelte-h":!0}),h(Pe)!=="svelte-5kxi2p"&&(Pe.textContent=Cs),jn=r(e),g(Qe.$$.fragment,e),Fn=r(e),Ye=m(e,"P",{"data-svelte-h":!0}),h(Ye)!=="svelte-nv5zhs"&&(Ye.textContent=js),xn=r(e),g(Ee.$$.fragment,e),zn=r(e),Ae=m(e,"P",{"data-svelte-h":!0}),h(Ae)!=="svelte-1v6wmbu"&&(Ae.textContent=Fs),Un=r(e),Oe=m(e,"P",{"data-svelte-h":!0}),h(Oe)!=="svelte-1t6ms80"&&(Oe.innerHTML=xs),Wn=r(e),g(De.$$.fragment,e),Zn=r(e),g(Ke.$$.fragment,e),Nn=r(e),g(et.$$.fragment,e),In=r(e),tt=m(e,"UL",{"data-svelte-h":!0}),h(tt)!=="svelte-p1b16m"&&(tt.innerHTML=zs),Ln=r(e),g(nt.$$.fragment,e),qn=r(e),R=m(e,"DIV",{class:!0});var se=B(R);g(ot.$$.fragment,se),io=r(se),zt=m(se,"P",{"data-svelte-h":!0}),h(zt)!=="svelte-14a3jhg"&&(zt.innerHTML=Us),lo=r(se),Ut=m(se,"P",{"data-svelte-h":!0}),h(Ut)!=="svelte-1ek1ss9"&&(Ut.innerHTML=Ws),co=r(se),g(_e.$$.fragment,se),se.forEach(s),Rn=r(e),g(st.$$.fragment,e),Vn=r(e),C=m(e,"DIV",{class:!0});var V=B(C);g(at.$$.fragment,V),po=r(V),Wt=m(V,"P",{"data-svelte-h":!0}),h(Wt)!=="svelte-ly2wwa"&&(Wt.textContent=Zs),mo=r(V),Zt=m(V,"P",{"data-svelte-h":!0}),h(Zt)!=="svelte-q52n56"&&(Zt.innerHTML=Ns),ho=r(V),Nt=m(V,"P",{"data-svelte-h":!0}),h(Nt)!=="svelte-hswkmf"&&(Nt.innerHTML=Is),uo=r(V),me=m(V,"DIV",{class:!0});var ue=B(me);g(rt.$$.fragment,ue),go=r(ue),It=m(ue,"P",{"data-svelte-h":!0}),h(It)!=="svelte-5e4ii2"&&(It.innerHTML=Ls),fo=r(ue),g(Me.$$.fragment,ue),ue.forEach(s),V.forEach(s),Gn=r(e),g(it.$$.fragment,e),Hn=r(e),j=m(e,"DIV",{class:!0});var G=B(j);g(lt.$$.fragment,G),_o=r(G),Lt=m(G,"P",{"data-svelte-h":!0}),h(Lt)!=="svelte-crjizq"&&(Lt.innerHTML=qs),Mo=r(G),qt=m(G,"P",{"data-svelte-h":!0}),h(qt)!=="svelte-q52n56"&&(qt.innerHTML=Rs),bo=r(G),Rt=m(G,"P",{"data-svelte-h":!0}),h(Rt)!=="svelte-hswkmf"&&(Rt.innerHTML=Vs),yo=r(G),O=m(G,"DIV",{class:!0});var ae=B(O);g(dt.$$.fragment,ae),To=r(ae),Vt=m(ae,"P",{"data-svelte-h":!0}),h(Vt)!=="svelte-6we8e"&&(Vt.innerHTML=Gs),ko=r(ae),g(be.$$.fragment,ae),wo=r(ae),g(ye.$$.fragment,ae),ae.forEach(s),G.forEach(s),Xn=r(e),g(ct.$$.fragment,e),Sn=r(e),F=m(e,"DIV",{class:!0});var H=B(F);g(pt.$$.fragment,H),vo=r(H),Gt=m(H,"P",{"data-svelte-h":!0}),h(Gt)!=="svelte-fhxvft"&&(Gt.innerHTML=Hs),Bo=r(H),Ht=m(H,"P",{"data-svelte-h":!0}),h(Ht)!=="svelte-q52n56"&&(Ht.innerHTML=Xs),$o=r(H),Xt=m(H,"P",{"data-svelte-h":!0}),h(Xt)!=="svelte-hswkmf"&&(Xt.innerHTML=Ss),Jo=r(H),D=m(H,"DIV",{class:!0});var re=B(D);g(mt.$$.fragment,re),Co=r(re),St=m(re,"P",{"data-svelte-h":!0}),h(St)!=="svelte-hhmt2m"&&(St.innerHTML=Ps),jo=r(re),g(Te.$$.fragment,re),Fo=r(re),g(ke.$$.fragment,re),re.forEach(s),H.forEach(s),Pn=r(e),g(ht.$$.fragment,e),Qn=r(e),x=m(e,"DIV",{class:!0});var X=B(x);g(ut.$$.fragment,X),xo=r(X),Pt=m(X,"P",{"data-svelte-h":!0}),h(Pt)!=="svelte-1jlhqt3"&&(Pt.innerHTML=Qs),zo=r(X),Qt=m(X,"P",{"data-svelte-h":!0}),h(Qt)!=="svelte-q52n56"&&(Qt.innerHTML=Ys),Uo=r(X),Yt=m(X,"P",{"data-svelte-h":!0}),h(Yt)!=="svelte-hswkmf"&&(Yt.innerHTML=Es),Wo=r(X),K=m(X,"DIV",{class:!0});var ie=B(K);g(gt.$$.fragment,ie),Zo=r(ie),Et=m(ie,"P",{"data-svelte-h":!0}),h(Et)!=="svelte-633y5y"&&(Et.innerHTML=As),No=r(ie),g(we.$$.fragment,ie),Io=r(ie),g(ve.$$.fragment,ie),ie.forEach(s),X.forEach(s),Yn=r(e),g(ft.$$.fragment,e),En=r(e),z=m(e,"DIV",{class:!0});var S=B(z);g(_t.$$.fragment,S),Lo=r(S),At=m(S,"P",{"data-svelte-h":!0}),h(At)!=="svelte-d9kana"&&(At.innerHTML=Os),qo=r(S),Ot=m(S,"P",{"data-svelte-h":!0}),h(Ot)!=="svelte-q52n56"&&(Ot.innerHTML=Ds),Ro=r(S),Dt=m(S,"P",{"data-svelte-h":!0}),h(Dt)!=="svelte-hswkmf"&&(Dt.innerHTML=Ks),Vo=r(S),ee=m(S,"DIV",{class:!0});var le=B(ee);g(Mt.$$.fragment,le),Go=r(le),Kt=m(le,"P",{"data-svelte-h":!0}),h(Kt)!=="svelte-1yn18hi"&&(Kt.innerHTML=ea),Ho=r(le),g(Be.$$.fragment,le),Xo=r(le),g($e.$$.fragment,le),le.forEach(s),S.forEach(s),An=r(e),g(bt.$$.fragment,e),On=r(e),U=m(e,"DIV",{class:!0});var P=B(U);g(yt.$$.fragment,P),So=r(P),en=m(P,"P",{"data-svelte-h":!0}),h(en)!=="svelte-vpcmn0"&&(en.textContent=ta),Po=r(P),tn=m(P,"P",{"data-svelte-h":!0}),h(tn)!=="svelte-q52n56"&&(tn.innerHTML=na),Qo=r(P),nn=m(P,"P",{"data-svelte-h":!0}),h(nn)!=="svelte-hswkmf"&&(nn.innerHTML=oa),Yo=r(P),I=m(P,"DIV",{class:!0});var Q=B(I);g(Tt.$$.fragment,Q),Eo=r(Q),on=m(Q,"P",{"data-svelte-h":!0}),h(on)!=="svelte-5pahq2"&&(on.innerHTML=sa),Ao=r(Q),g(Je.$$.fragment,Q),Oo=r(Q),g(Ce.$$.fragment,Q),Do=r(Q),g(je.$$.fragment,Q),Q.forEach(s),P.forEach(s),Dn=r(e),g(kt.$$.fragment,e),Kn=r(e),W=m(e,"DIV",{class:!0});var Y=B(W);g(wt.$$.fragment,Y),Ko=r(Y),sn=m(Y,"P",{"data-svelte-h":!0}),h(sn)!=="svelte-18jrowp"&&(sn.textContent=aa),es=r(Y),an=m(Y,"P",{"data-svelte-h":!0}),h(an)!=="svelte-q52n56"&&(an.innerHTML=ra),ts=r(Y),rn=m(Y,"P",{"data-svelte-h":!0}),h(rn)!=="svelte-hswkmf"&&(rn.innerHTML=ia),ns=r(Y),te=m(Y,"DIV",{class:!0});var de=B(te);g(vt.$$.fragment,de),os=r(de),ln=m(de,"P",{"data-svelte-h":!0}),h(ln)!=="svelte-e6m9t2"&&(ln.innerHTML=la),ss=r(de),g(Fe.$$.fragment,de),as=r(de),g(xe.$$.fragment,de),de.forEach(s),Y.forEach(s),eo=r(e),g(Bt.$$.fragment,e),to=r(e),Z=m(e,"DIV",{class:!0});var E=B(Z);g($t.$$.fragment,E),rs=r(E),dn=m(E,"P",{"data-svelte-h":!0}),h(dn)!=="svelte-8e2xoi"&&(dn.textContent=da),is=r(E),cn=m(E,"P",{"data-svelte-h":!0}),h(cn)!=="svelte-q52n56"&&(cn.innerHTML=ca),ls=r(E),pn=m(E,"P",{"data-svelte-h":!0}),h(pn)!=="svelte-hswkmf"&&(pn.innerHTML=pa),ds=r(E),ne=m(E,"DIV",{class:!0});var ce=B(ne);g(Jt.$$.fragment,ce),cs=r(ce),mn=m(ce,"P",{"data-svelte-h":!0}),h(mn)!=="svelte-iovuym"&&(mn.innerHTML=ma),ps=r(ce),g(ze.$$.fragment,ce),ms=r(ce),g(Ue.$$.fragment,ce),ce.forEach(s),E.forEach(s),no=r(e),g(Ct.$$.fragment,e),oo=r(e),N=m(e,"DIV",{class:!0});var A=B(N);g(jt.$$.fragment,A),hs=r(A),hn=m(A,"P",{"data-svelte-h":!0}),h(hn)!=="svelte-46yest"&&(hn.innerHTML=ha),us=r(A),un=m(A,"P",{"data-svelte-h":!0}),h(un)!=="svelte-q52n56"&&(un.innerHTML=ua),gs=r(A),gn=m(A,"P",{"data-svelte-h":!0}),h(gn)!=="svelte-hswkmf"&&(gn.innerHTML=ga),fs=r(A),oe=m(A,"DIV",{class:!0});var pe=B(oe);g(Ft.$$.fragment,pe),_s=r(pe),fn=m(pe,"P",{"data-svelte-h":!0}),h(fn)!=="svelte-43310"&&(fn.innerHTML=fa),Ms=r(pe),g(We.$$.fragment,pe),bs=r(pe),g(Ze.$$.fragment,pe),pe.forEach(s),A.forEach(s),so=r(e),g(xt.$$.fragment,e),ao=r(e),_n=m(e,"P",{}),B(_n).forEach(s),this.h()},h(){v(t,"name","hf:doc:metadata"),v(t,"content",Ha),v(fe,"class","flex flex-wrap space-x-1"),v(R,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(me,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(C,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(O,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(j,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(D,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(F,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(K,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(x,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(ee,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(z,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(I,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(U,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(te,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(W,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(ne,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(Z,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(oe,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(N,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,o){i(document.head,t),c(e,y,o),c(e,l,o),c(e,d,o),c(e,T,o),c(e,k,o),f(Ne,e,o),c(e,Mn,o),c(e,fe,o),c(e,bn,o),f(Ie,e,o),c(e,yn,o),c(e,Le,o),c(e,Tn,o),c(e,qe,o),c(e,kn,o),c(e,Re,o),c(e,wn,o),c(e,Ve,o),c(e,vn,o),f(Ge,e,o),c(e,Bn,o),c(e,He,o),c(e,$n,o),c(e,Xe,o),c(e,Jn,o),c(e,Se,o),c(e,Cn,o),c(e,Pe,o),c(e,jn,o),f(Qe,e,o),c(e,Fn,o),c(e,Ye,o),c(e,xn,o),f(Ee,e,o),c(e,zn,o),c(e,Ae,o),c(e,Un,o),c(e,Oe,o),c(e,Wn,o),f(De,e,o),c(e,Zn,o),f(Ke,e,o),c(e,Nn,o),f(et,e,o),c(e,In,o),c(e,tt,o),c(e,Ln,o),f(nt,e,o),c(e,qn,o),c(e,R,o),f(ot,R,null),i(R,io),i(R,zt),i(R,lo),i(R,Ut),i(R,co),f(_e,R,null),c(e,Rn,o),f(st,e,o),c(e,Vn,o),c(e,C,o),f(at,C,null),i(C,po),i(C,Wt),i(C,mo),i(C,Zt),i(C,ho),i(C,Nt),i(C,uo),i(C,me),f(rt,me,null),i(me,go),i(me,It),i(me,fo),f(Me,me,null),c(e,Gn,o),f(it,e,o),c(e,Hn,o),c(e,j,o),f(lt,j,null),i(j,_o),i(j,Lt),i(j,Mo),i(j,qt),i(j,bo),i(j,Rt),i(j,yo),i(j,O),f(dt,O,null),i(O,To),i(O,Vt),i(O,ko),f(be,O,null),i(O,wo),f(ye,O,null),c(e,Xn,o),f(ct,e,o),c(e,Sn,o),c(e,F,o),f(pt,F,null),i(F,vo),i(F,Gt),i(F,Bo),i(F,Ht),i(F,$o),i(F,Xt),i(F,Jo),i(F,D),f(mt,D,null),i(D,Co),i(D,St),i(D,jo),f(Te,D,null),i(D,Fo),f(ke,D,null),c(e,Pn,o),f(ht,e,o),c(e,Qn,o),c(e,x,o),f(ut,x,null),i(x,xo),i(x,Pt),i(x,zo),i(x,Qt),i(x,Uo),i(x,Yt),i(x,Wo),i(x,K),f(gt,K,null),i(K,Zo),i(K,Et),i(K,No),f(we,K,null),i(K,Io),f(ve,K,null),c(e,Yn,o),f(ft,e,o),c(e,En,o),c(e,z,o),f(_t,z,null),i(z,Lo),i(z,At),i(z,qo),i(z,Ot),i(z,Ro),i(z,Dt),i(z,Vo),i(z,ee),f(Mt,ee,null),i(ee,Go),i(ee,Kt),i(ee,Ho),f(Be,ee,null),i(ee,Xo),f($e,ee,null),c(e,An,o),f(bt,e,o),c(e,On,o),c(e,U,o),f(yt,U,null),i(U,So),i(U,en),i(U,Po),i(U,tn),i(U,Qo),i(U,nn),i(U,Yo),i(U,I),f(Tt,I,null),i(I,Eo),i(I,on),i(I,Ao),f(Je,I,null),i(I,Oo),f(Ce,I,null),i(I,Do),f(je,I,null),c(e,Dn,o),f(kt,e,o),c(e,Kn,o),c(e,W,o),f(wt,W,null),i(W,Ko),i(W,sn),i(W,es),i(W,an),i(W,ts),i(W,rn),i(W,ns),i(W,te),f(vt,te,null),i(te,os),i(te,ln),i(te,ss),f(Fe,te,null),i(te,as),f(xe,te,null),c(e,eo,o),f(Bt,e,o),c(e,to,o),c(e,Z,o),f($t,Z,null),i(Z,rs),i(Z,dn),i(Z,is),i(Z,cn),i(Z,ls),i(Z,pn),i(Z,ds),i(Z,ne),f(Jt,ne,null),i(ne,cs),i(ne,mn),i(ne,ps),f(ze,ne,null),i(ne,ms),f(Ue,ne,null),c(e,no,o),f(Ct,e,o),c(e,oo,o),c(e,N,o),f(jt,N,null),i(N,hs),i(N,hn),i(N,us),i(N,un),i(N,gs),i(N,gn),i(N,fs),i(N,oe),f(Ft,oe,null),i(oe,_s),i(oe,fn),i(oe,Ms),f(We,oe,null),i(oe,bs),f(Ze,oe,null),c(e,so,o),f(xt,e,o),c(e,ao,o),c(e,_n,o),ro=!0},p(e,[o]){const se={};o&2&&(se.$$scope={dirty:o,ctx:e}),_e.$set(se);const V={};o&2&&(V.$$scope={dirty:o,ctx:e}),Me.$set(V);const ue={};o&2&&(ue.$$scope={dirty:o,ctx:e}),be.$set(ue);const G={};o&2&&(G.$$scope={dirty:o,ctx:e}),ye.$set(G);const ae={};o&2&&(ae.$$scope={dirty:o,ctx:e}),Te.$set(ae);const H={};o&2&&(H.$$scope={dirty:o,ctx:e}),ke.$set(H);const re={};o&2&&(re.$$scope={dirty:o,ctx:e}),we.$set(re);const X={};o&2&&(X.$$scope={dirty:o,ctx:e}),ve.$set(X);const ie={};o&2&&(ie.$$scope={dirty:o,ctx:e}),Be.$set(ie);const S={};o&2&&(S.$$scope={dirty:o,ctx:e}),$e.$set(S);const le={};o&2&&(le.$$scope={dirty:o,ctx:e}),Je.$set(le);const P={};o&2&&(P.$$scope={dirty:o,ctx:e}),Ce.$set(P);const Q={};o&2&&(Q.$$scope={dirty:o,ctx:e}),je.$set(Q);const Y={};o&2&&(Y.$$scope={dirty:o,ctx:e}),Fe.$set(Y);const de={};o&2&&(de.$$scope={dirty:o,ctx:e}),xe.$set(de);const E={};o&2&&(E.$$scope={dirty:o,ctx:e}),ze.$set(E);const ce={};o&2&&(ce.$$scope={dirty:o,ctx:e}),Ue.$set(ce);const A={};o&2&&(A.$$scope={dirty:o,ctx:e}),We.$set(A);const pe={};o&2&&(pe.$$scope={dirty:o,ctx:e}),Ze.$set(pe)},i(e){ro||(_(Ne.$$.fragment,e),_(Ie.$$.fragment,e),_(Ge.$$.fragment,e),_(Qe.$$.fragment,e),_(Ee.$$.fragment,e),_(De.$$.fragment,e),_(Ke.$$.fragment,e),_(et.$$.fragment,e),_(nt.$$.fragment,e),_(ot.$$.fragment,e),_(_e.$$.fragment,e),_(st.$$.fragment,e),_(at.$$.fragment,e),_(rt.$$.fragment,e),_(Me.$$.fragment,e),_(it.$$.fragment,e),_(lt.$$.fragment,e),_(dt.$$.fragment,e),_(be.$$.fragment,e),_(ye.$$.fragment,e),_(ct.$$.fragment,e),_(pt.$$.fragment,e),_(mt.$$.fragment,e),_(Te.$$.fragment,e),_(ke.$$.fragment,e),_(ht.$$.fragment,e),_(ut.$$.fragment,e),_(gt.$$.fragment,e),_(we.$$.fragment,e),_(ve.$$.fragment,e),_(ft.$$.fragment,e),_(_t.$$.fragment,e),_(Mt.$$.fragment,e),_(Be.$$.fragment,e),_($e.$$.fragment,e),_(bt.$$.fragment,e),_(yt.$$.fragment,e),_(Tt.$$.fragment,e),_(Je.$$.fragment,e),_(Ce.$$.fragment,e),_(je.$$.fragment,e),_(kt.$$.fragment,e),_(wt.$$.fragment,e),_(vt.$$.fragment,e),_(Fe.$$.fragment,e),_(xe.$$.fragment,e),_(Bt.$$.fragment,e),_($t.$$.fragment,e),_(Jt.$$.fragment,e),_(ze.$$.fragment,e),_(Ue.$$.fragment,e),_(Ct.$$.fragment,e),_(jt.$$.fragment,e),_(Ft.$$.fragment,e),_(We.$$.fragment,e),_(Ze.$$.fragment,e),_(xt.$$.fragment,e),ro=!0)},o(e){M(Ne.$$.fragment,e),M(Ie.$$.fragment,e),M(Ge.$$.fragment,e),M(Qe.$$.fragment,e),M(Ee.$$.fragment,e),M(De.$$.fragment,e),M(Ke.$$.fragment,e),M(et.$$.fragment,e),M(nt.$$.fragment,e),M(ot.$$.fragment,e),M(_e.$$.fragment,e),M(st.$$.fragment,e),M(at.$$.fragment,e),M(rt.$$.fragment,e),M(Me.$$.fragment,e),M(it.$$.fragment,e),M(lt.$$.fragment,e),M(dt.$$.fragment,e),M(be.$$.fragment,e),M(ye.$$.fragment,e),M(ct.$$.fragment,e),M(pt.$$.fragment,e),M(mt.$$.fragment,e),M(Te.$$.fragment,e),M(ke.$$.fragment,e),M(ht.$$.fragment,e),M(ut.$$.fragment,e),M(gt.$$.fragment,e),M(we.$$.fragment,e),M(ve.$$.fragment,e),M(ft.$$.fragment,e),M(_t.$$.fragment,e),M(Mt.$$.fragment,e),M(Be.$$.fragment,e),M($e.$$.fragment,e),M(bt.$$.fragment,e),M(yt.$$.fragment,e),M(Tt.$$.fragment,e),M(Je.$$.fragment,e),M(Ce.$$.fragment,e),M(je.$$.fragment,e),M(kt.$$.fragment,e),M(wt.$$.fragment,e),M(vt.$$.fragment,e),M(Fe.$$.fragment,e),M(xe.$$.fragment,e),M(Bt.$$.fragment,e),M($t.$$.fragment,e),M(Jt.$$.fragment,e),M(ze.$$.fragment,e),M(Ue.$$.fragment,e),M(Ct.$$.fragment,e),M(jt.$$.fragment,e),M(Ft.$$.fragment,e),M(We.$$.fragment,e),M(Ze.$$.fragment,e),M(xt.$$.fragment,e),ro=!1},d(e){e&&(s(y),s(l),s(d),s(T),s(k),s(Mn),s(fe),s(bn),s(yn),s(Le),s(Tn),s(qe),s(kn),s(Re),s(wn),s(Ve),s(vn),s(Bn),s(He),s($n),s(Xe),s(Jn),s(Se),s(Cn),s(Pe),s(jn),s(Fn),s(Ye),s(xn),s(zn),s(Ae),s(Un),s(Oe),s(Wn),s(Zn),s(Nn),s(In),s(tt),s(Ln),s(qn),s(R),s(Rn),s(Vn),s(C),s(Gn),s(Hn),s(j),s(Xn),s(Sn),s(F),s(Pn),s(Qn),s(x),s(Yn),s(En),s(z),s(An),s(On),s(U),s(Dn),s(Kn),s(W),s(eo),s(to),s(Z),s(no),s(oo),s(N),s(so),s(ao),s(_n)),s(t),b(Ne,e),b(Ie,e),b(Ge,e),b(Qe,e),b(Ee,e),b(De,e),b(Ke,e),b(et,e),b(nt,e),b(ot),b(_e),b(st,e),b(at),b(rt),b(Me),b(it,e),b(lt),b(dt),b(be),b(ye),b(ct,e),b(pt),b(mt),b(Te),b(ke),b(ht,e),b(ut),b(gt),b(we),b(ve),b(ft,e),b(_t),b(Mt),b(Be),b($e),b(bt,e),b(yt),b(Tt),b(Je),b(Ce),b(je),b(kt,e),b(wt),b(vt),b(Fe),b(xe),b(Bt,e),b($t),b(Jt),b(ze),b(Ue),b(Ct,e),b(jt),b(Ft),b(We),b(Ze),b(xt,e)}}}const Ha='{"title":"MegatronBERT","local":"megatronbert","sections":[{"title":"Overview","local":"overview","sections":[],"depth":2},{"title":"Usage tips","local":"usage-tips","sections":[],"depth":2},{"title":"Resources","local":"resources","sections":[],"depth":2},{"title":"MegatronBertConfig","local":"transformers.MegatronBertConfig","sections":[],"depth":2},{"title":"MegatronBertModel","local":"transformers.MegatronBertModel","sections":[],"depth":2},{"title":"MegatronBertForMaskedLM","local":"transformers.MegatronBertForMaskedLM","sections":[],"depth":2},{"title":"MegatronBertForCausalLM","local":"transformers.MegatronBertForCausalLM","sections":[],"depth":2},{"title":"MegatronBertForNextSentencePrediction","local":"transformers.MegatronBertForNextSentencePrediction","sections":[],"depth":2},{"title":"MegatronBertForPreTraining","local":"transformers.MegatronBertForPreTraining","sections":[],"depth":2},{"title":"MegatronBertForSequenceClassification","local":"transformers.MegatronBertForSequenceClassification","sections":[],"depth":2},{"title":"MegatronBertForMultipleChoice","local":"transformers.MegatronBertForMultipleChoice","sections":[],"depth":2},{"title":"MegatronBertForTokenClassification","local":"transformers.MegatronBertForTokenClassification","sections":[],"depth":2},{"title":"MegatronBertForQuestionAnswering","local":"transformers.MegatronBertForQuestionAnswering","sections":[],"depth":2}],"depth":1}';function Xa(w){return Ma(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class Da extends ba{constructor(t){super(),ya(this,t,Xa,Ga,_a,{})}}export{Da as component};
