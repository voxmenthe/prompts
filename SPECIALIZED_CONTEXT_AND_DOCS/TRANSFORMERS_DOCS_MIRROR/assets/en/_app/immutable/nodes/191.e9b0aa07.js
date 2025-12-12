import{s as Mr,o as wr,n as x}from"../chunks/scheduler.18a86fab.js";import{S as kr,i as vr,g as d,s,r as u,A as Fr,h as c,f as i,c as r,j as F,x as h,u as f,k as v,y as o,a as m,v as g,d as _,t as b,w as T}from"../chunks/index.98837b22.js";import{T as Xe}from"../chunks/Tip.77304350.js";import{D as $}from"../chunks/Docstring.a1ef7999.js";import{C as _e}from"../chunks/CodeBlock.8d0c2e8a.js";import{E as ge}from"../chunks/ExampleCodeBlock.8c3ee1f9.js";import{H as B,E as $r}from"../chunks/getInferenceSnippets.06c2775f.js";function Nr(k){let t,y="Example:",l,p,M;return p=new _e({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEZOZXRDb25maWclMkMlMjBGTmV0TW9kZWwlMEElMEElMjMlMjBJbml0aWFsaXppbmclMjBhJTIwRk5ldCUyMGZuZXQtYmFzZSUyMHN0eWxlJTIwY29uZmlndXJhdGlvbiUwQWNvbmZpZ3VyYXRpb24lMjAlM0QlMjBGTmV0Q29uZmlnKCklMEElMEElMjMlMjBJbml0aWFsaXppbmclMjBhJTIwbW9kZWwlMjAod2l0aCUyMHJhbmRvbSUyMHdlaWdodHMpJTIwZnJvbSUyMHRoZSUyMGZuZXQtYmFzZSUyMHN0eWxlJTIwY29uZmlndXJhdGlvbiUwQW1vZGVsJTIwJTNEJTIwRk5ldE1vZGVsKGNvbmZpZ3VyYXRpb24pJTBBJTBBJTIzJTIwQWNjZXNzaW5nJTIwdGhlJTIwbW9kZWwlMjBjb25maWd1cmF0aW9uJTBBY29uZmlndXJhdGlvbiUyMCUzRCUyMG1vZGVsLmNvbmZpZw==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> FNetConfig, FNetModel

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a FNet fnet-base style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = FNetConfig()

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a model (with random weights) from the fnet-base style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = FNetModel(configuration)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Accessing the model configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = model.config`,wrap:!1}}),{c(){t=d("p"),t.textContent=y,l=s(),u(p.$$.fragment)},l(n){t=c(n,"P",{"data-svelte-h":!0}),h(t)!=="svelte-11lpom8"&&(t.textContent=y),l=r(n),f(p.$$.fragment,n)},m(n,w){m(n,t,w),m(n,l,w),g(p,n,w),M=!0},p:x,i(n){M||(_(p.$$.fragment,n),M=!0)},o(n){b(p.$$.fragment,n),M=!1},d(n){n&&(i(t),i(l)),T(p,n)}}}function xr(k){let t,y=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=d("p"),t.innerHTML=y},l(l){t=c(l,"P",{"data-svelte-h":!0}),h(t)!=="svelte-fincs2"&&(t.innerHTML=y)},m(l,p){m(l,t,p)},p:x,d(l){l&&i(t)}}}function zr(k){let t,y=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=d("p"),t.innerHTML=y},l(l){t=c(l,"P",{"data-svelte-h":!0}),h(t)!=="svelte-fincs2"&&(t.innerHTML=y)},m(l,p){m(l,t,p)},p:x,d(l){l&&i(t)}}}function jr(k){let t,y="Example:",l,p,M;return p=new _e({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBGTmV0Rm9yUHJlVHJhaW5pbmclMEFpbXBvcnQlMjB0b3JjaCUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMmdvb2dsZSUyRmZuZXQtYmFzZSUyMiklMEFtb2RlbCUyMCUzRCUyMEZOZXRGb3JQcmVUcmFpbmluZy5mcm9tX3ByZXRyYWluZWQoJTIyZ29vZ2xlJTJGZm5ldC1iYXNlJTIyKSUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplciglMjJIZWxsbyUyQyUyMG15JTIwZG9nJTIwaXMlMjBjdXRlJTIyJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiklMEFvdXRwdXRzJTIwJTNEJTIwbW9kZWwoKippbnB1dHMpJTBBcHJlZGljdGlvbl9sb2dpdHMlMjAlM0QlMjBvdXRwdXRzLnByZWRpY3Rpb25fbG9naXRzJTBBc2VxX3JlbGF0aW9uc2hpcF9sb2dpdHMlMjAlM0QlMjBvdXRwdXRzLnNlcV9yZWxhdGlvbnNoaXBfbG9naXRz",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, FNetForPreTraining
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;google/fnet-base&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = FNetForPreTraining.from_pretrained(<span class="hljs-string">&quot;google/fnet-base&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs)
<span class="hljs-meta">&gt;&gt;&gt; </span>prediction_logits = outputs.prediction_logits
<span class="hljs-meta">&gt;&gt;&gt; </span>seq_relationship_logits = outputs.seq_relationship_logits`,wrap:!1}}),{c(){t=d("p"),t.textContent=y,l=s(),u(p.$$.fragment)},l(n){t=c(n,"P",{"data-svelte-h":!0}),h(t)!=="svelte-11lpom8"&&(t.textContent=y),l=r(n),f(p.$$.fragment,n)},m(n,w){m(n,t,w),m(n,l,w),g(p,n,w),M=!0},p:x,i(n){M||(_(p.$$.fragment,n),M=!0)},o(n){b(p.$$.fragment,n),M=!1},d(n){n&&(i(t),i(l)),T(p,n)}}}function Jr(k){let t,y=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=d("p"),t.innerHTML=y},l(l){t=c(l,"P",{"data-svelte-h":!0}),h(t)!=="svelte-fincs2"&&(t.innerHTML=y)},m(l,p){m(l,t,p)},p:x,d(l){l&&i(t)}}}function Ur(k){let t,y="Example:",l,p,M;return p=new _e({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBGTmV0Rm9yTWFza2VkTE0lMEFpbXBvcnQlMjB0b3JjaCUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMmdvb2dsZSUyRmZuZXQtYmFzZSUyMiklMEFtb2RlbCUyMCUzRCUyMEZOZXRGb3JNYXNrZWRMTS5mcm9tX3ByZXRyYWluZWQoJTIyZ29vZ2xlJTJGZm5ldC1iYXNlJTIyKSUwQSUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplciglMjJUaGUlMjBjYXBpdGFsJTIwb2YlMjBGcmFuY2UlMjBpcyUyMCUzQ21hc2slM0UuJTIyJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiklMEElMEF3aXRoJTIwdG9yY2gubm9fZ3JhZCgpJTNBJTBBJTIwJTIwJTIwJTIwbG9naXRzJTIwJTNEJTIwbW9kZWwoKippbnB1dHMpLmxvZ2l0cyUwQSUwQSUyMyUyMHJldHJpZXZlJTIwaW5kZXglMjBvZiUyMCUzQ21hc2slM0UlMEFtYXNrX3Rva2VuX2luZGV4JTIwJTNEJTIwKGlucHV0cy5pbnB1dF9pZHMlMjAlM0QlM0QlMjB0b2tlbml6ZXIubWFza190b2tlbl9pZCklNUIwJTVELm5vbnplcm8oYXNfdHVwbGUlM0RUcnVlKSU1QjAlNUQlMEElMEFwcmVkaWN0ZWRfdG9rZW5faWQlMjAlM0QlMjBsb2dpdHMlNUIwJTJDJTIwbWFza190b2tlbl9pbmRleCU1RC5hcmdtYXgoYXhpcyUzRC0xKSUwQXRva2VuaXplci5kZWNvZGUocHJlZGljdGVkX3Rva2VuX2lkKSUwQSUwQWxhYmVscyUyMCUzRCUyMHRva2VuaXplciglMjJUaGUlMjBjYXBpdGFsJTIwb2YlMjBGcmFuY2UlMjBpcyUyMFBhcmlzLiUyMiUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpJTVCJTIyaW5wdXRfaWRzJTIyJTVEJTBBJTIzJTIwbWFzayUyMGxhYmVscyUyMG9mJTIwbm9uLSUzQ21hc2slM0UlMjB0b2tlbnMlMEFsYWJlbHMlMjAlM0QlMjB0b3JjaC53aGVyZShpbnB1dHMuaW5wdXRfaWRzJTIwJTNEJTNEJTIwdG9rZW5pemVyLm1hc2tfdG9rZW5faWQlMkMlMjBsYWJlbHMlMkMlMjAtMTAwKSUwQSUwQW91dHB1dHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyUyQyUyMGxhYmVscyUzRGxhYmVscyklMEFyb3VuZChvdXRwdXRzLmxvc3MuaXRlbSgpJTJDJTIwMik=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, FNetForMaskedLM
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;google/fnet-base&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = FNetForMaskedLM.from_pretrained(<span class="hljs-string">&quot;google/fnet-base&quot;</span>)

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
...`,wrap:!1}}),{c(){t=d("p"),t.textContent=y,l=s(),u(p.$$.fragment)},l(n){t=c(n,"P",{"data-svelte-h":!0}),h(t)!=="svelte-11lpom8"&&(t.textContent=y),l=r(n),f(p.$$.fragment,n)},m(n,w){m(n,t,w),m(n,l,w),g(p,n,w),M=!0},p:x,i(n){M||(_(p.$$.fragment,n),M=!0)},o(n){b(p.$$.fragment,n),M=!1},d(n){n&&(i(t),i(l)),T(p,n)}}}function Cr(k){let t,y=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=d("p"),t.innerHTML=y},l(l){t=c(l,"P",{"data-svelte-h":!0}),h(t)!=="svelte-fincs2"&&(t.innerHTML=y)},m(l,p){m(l,t,p)},p:x,d(l){l&&i(t)}}}function Zr(k){let t,y="Example:",l,p,M;return p=new _e({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBGTmV0Rm9yTmV4dFNlbnRlbmNlUHJlZGljdGlvbiUwQWltcG9ydCUyMHRvcmNoJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIyZ29vZ2xlJTJGZm5ldC1iYXNlJTIyKSUwQW1vZGVsJTIwJTNEJTIwRk5ldEZvck5leHRTZW50ZW5jZVByZWRpY3Rpb24uZnJvbV9wcmV0cmFpbmVkKCUyMmdvb2dsZSUyRmZuZXQtYmFzZSUyMiklMEFwcm9tcHQlMjAlM0QlMjAlMjJJbiUyMEl0YWx5JTJDJTIwcGl6emElMjBzZXJ2ZWQlMjBpbiUyMGZvcm1hbCUyMHNldHRpbmdzJTJDJTIwc3VjaCUyMGFzJTIwYXQlMjBhJTIwcmVzdGF1cmFudCUyQyUyMGlzJTIwcHJlc2VudGVkJTIwdW5zbGljZWQuJTIyJTBBbmV4dF9zZW50ZW5jZSUyMCUzRCUyMCUyMlRoZSUyMHNreSUyMGlzJTIwYmx1ZSUyMGR1ZSUyMHRvJTIwdGhlJTIwc2hvcnRlciUyMHdhdmVsZW5ndGglMjBvZiUyMGJsdWUlMjBsaWdodC4lMjIlMEFlbmNvZGluZyUyMCUzRCUyMHRva2VuaXplcihwcm9tcHQlMkMlMjBuZXh0X3NlbnRlbmNlJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiklMEFvdXRwdXRzJTIwJTNEJTIwbW9kZWwoKiplbmNvZGluZyUyQyUyMGxhYmVscyUzRHRvcmNoLkxvbmdUZW5zb3IoJTVCMSU1RCkpJTBBbG9naXRzJTIwJTNEJTIwb3V0cHV0cy5sb2dpdHMlMEFhc3NlcnQlMjBsb2dpdHMlNUIwJTJDJTIwMCU1RCUyMCUzQyUyMGxvZ2l0cyU1QjAlMkMlMjAxJTVEJTIwJTIwJTIzJTIwbmV4dCUyMHNlbnRlbmNlJTIwd2FzJTIwcmFuZG9t",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, FNetForNextSentencePrediction
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;google/fnet-base&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = FNetForNextSentencePrediction.from_pretrained(<span class="hljs-string">&quot;google/fnet-base&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>prompt = <span class="hljs-string">&quot;In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>next_sentence = <span class="hljs-string">&quot;The sky is blue due to the shorter wavelength of blue light.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>encoding = tokenizer(prompt, next_sentence, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**encoding, labels=torch.LongTensor([<span class="hljs-number">1</span>]))
<span class="hljs-meta">&gt;&gt;&gt; </span>logits = outputs.logits
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">assert</span> logits[<span class="hljs-number">0</span>, <span class="hljs-number">0</span>] &lt; logits[<span class="hljs-number">0</span>, <span class="hljs-number">1</span>]  <span class="hljs-comment"># next sentence was random</span>`,wrap:!1}}),{c(){t=d("p"),t.textContent=y,l=s(),u(p.$$.fragment)},l(n){t=c(n,"P",{"data-svelte-h":!0}),h(t)!=="svelte-11lpom8"&&(t.textContent=y),l=r(n),f(p.$$.fragment,n)},m(n,w){m(n,t,w),m(n,l,w),g(p,n,w),M=!0},p:x,i(n){M||(_(p.$$.fragment,n),M=!0)},o(n){b(p.$$.fragment,n),M=!1},d(n){n&&(i(t),i(l)),T(p,n)}}}function Ir(k){let t,y=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=d("p"),t.innerHTML=y},l(l){t=c(l,"P",{"data-svelte-h":!0}),h(t)!=="svelte-fincs2"&&(t.innerHTML=y)},m(l,p){m(l,t,p)},p:x,d(l){l&&i(t)}}}function qr(k){let t,y="Example of single-label classification:",l,p,M;return p=new _e({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Rva2VuaXplciUyQyUyMEZOZXRGb3JTZXF1ZW5jZUNsYXNzaWZpY2F0aW9uJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIyZ29vZ2xlJTJGZm5ldC1iYXNlJTIyKSUwQW1vZGVsJTIwJTNEJTIwRk5ldEZvclNlcXVlbmNlQ2xhc3NpZmljYXRpb24uZnJvbV9wcmV0cmFpbmVkKCUyMmdvb2dsZSUyRmZuZXQtYmFzZSUyMiklMEElMEFpbnB1dHMlMjAlM0QlMjB0b2tlbml6ZXIoJTIySGVsbG8lMkMlMjBteSUyMGRvZyUyMGlzJTIwY3V0ZSUyMiUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpJTBBJTBBd2l0aCUyMHRvcmNoLm5vX2dyYWQoKSUzQSUwQSUyMCUyMCUyMCUyMGxvZ2l0cyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzKS5sb2dpdHMlMEElMEFwcmVkaWN0ZWRfY2xhc3NfaWQlMjAlM0QlMjBsb2dpdHMuYXJnbWF4KCkuaXRlbSgpJTBBbW9kZWwuY29uZmlnLmlkMmxhYmVsJTVCcHJlZGljdGVkX2NsYXNzX2lkJTVEJTBBJTBBJTIzJTIwVG8lMjB0cmFpbiUyMGElMjBtb2RlbCUyMG9uJTIwJTYwbnVtX2xhYmVscyU2MCUyMGNsYXNzZXMlMkMlMjB5b3UlMjBjYW4lMjBwYXNzJTIwJTYwbnVtX2xhYmVscyUzRG51bV9sYWJlbHMlNjAlMjB0byUyMCU2MC5mcm9tX3ByZXRyYWluZWQoLi4uKSU2MCUwQW51bV9sYWJlbHMlMjAlM0QlMjBsZW4obW9kZWwuY29uZmlnLmlkMmxhYmVsKSUwQW1vZGVsJTIwJTNEJTIwRk5ldEZvclNlcXVlbmNlQ2xhc3NpZmljYXRpb24uZnJvbV9wcmV0cmFpbmVkKCUyMmdvb2dsZSUyRmZuZXQtYmFzZSUyMiUyQyUyMG51bV9sYWJlbHMlM0RudW1fbGFiZWxzKSUwQSUwQWxhYmVscyUyMCUzRCUyMHRvcmNoLnRlbnNvciglNUIxJTVEKSUwQWxvc3MlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyUyQyUyMGxhYmVscyUzRGxhYmVscykubG9zcyUwQXJvdW5kKGxvc3MuaXRlbSgpJTJDJTIwMik=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, FNetForSequenceClassification

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;google/fnet-base&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = FNetForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;google/fnet-base&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    logits = model(**inputs).logits

<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_class_id = logits.argmax().item()
<span class="hljs-meta">&gt;&gt;&gt; </span>model.config.id2label[predicted_class_id]
...

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># To train a model on \`num_labels\` classes, you can pass \`num_labels=num_labels\` to \`.from_pretrained(...)\`</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>num_labels = <span class="hljs-built_in">len</span>(model.config.id2label)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = FNetForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;google/fnet-base&quot;</span>, num_labels=num_labels)

<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.tensor([<span class="hljs-number">1</span>])
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = model(**inputs, labels=labels).loss
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">round</span>(loss.item(), <span class="hljs-number">2</span>)
...`,wrap:!1}}),{c(){t=d("p"),t.textContent=y,l=s(),u(p.$$.fragment)},l(n){t=c(n,"P",{"data-svelte-h":!0}),h(t)!=="svelte-ykxpe4"&&(t.textContent=y),l=r(n),f(p.$$.fragment,n)},m(n,w){m(n,t,w),m(n,l,w),g(p,n,w),M=!0},p:x,i(n){M||(_(p.$$.fragment,n),M=!0)},o(n){b(p.$$.fragment,n),M=!1},d(n){n&&(i(t),i(l)),T(p,n)}}}function Wr(k){let t,y="Example of multi-label classification:",l,p,M;return p=new _e({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Rva2VuaXplciUyQyUyMEZOZXRGb3JTZXF1ZW5jZUNsYXNzaWZpY2F0aW9uJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIyZ29vZ2xlJTJGZm5ldC1iYXNlJTIyKSUwQW1vZGVsJTIwJTNEJTIwRk5ldEZvclNlcXVlbmNlQ2xhc3NpZmljYXRpb24uZnJvbV9wcmV0cmFpbmVkKCUyMmdvb2dsZSUyRmZuZXQtYmFzZSUyMiUyQyUyMHByb2JsZW1fdHlwZSUzRCUyMm11bHRpX2xhYmVsX2NsYXNzaWZpY2F0aW9uJTIyKSUwQSUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplciglMjJIZWxsbyUyQyUyMG15JTIwZG9nJTIwaXMlMjBjdXRlJTIyJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiklMEElMEF3aXRoJTIwdG9yY2gubm9fZ3JhZCgpJTNBJTBBJTIwJTIwJTIwJTIwbG9naXRzJTIwJTNEJTIwbW9kZWwoKippbnB1dHMpLmxvZ2l0cyUwQSUwQXByZWRpY3RlZF9jbGFzc19pZHMlMjAlM0QlMjB0b3JjaC5hcmFuZ2UoMCUyQyUyMGxvZ2l0cy5zaGFwZSU1Qi0xJTVEKSU1QnRvcmNoLnNpZ21vaWQobG9naXRzKS5zcXVlZXplKGRpbSUzRDApJTIwJTNFJTIwMC41JTVEJTBBJTBBJTIzJTIwVG8lMjB0cmFpbiUyMGElMjBtb2RlbCUyMG9uJTIwJTYwbnVtX2xhYmVscyU2MCUyMGNsYXNzZXMlMkMlMjB5b3UlMjBjYW4lMjBwYXNzJTIwJTYwbnVtX2xhYmVscyUzRG51bV9sYWJlbHMlNjAlMjB0byUyMCU2MC5mcm9tX3ByZXRyYWluZWQoLi4uKSU2MCUwQW51bV9sYWJlbHMlMjAlM0QlMjBsZW4obW9kZWwuY29uZmlnLmlkMmxhYmVsKSUwQW1vZGVsJTIwJTNEJTIwRk5ldEZvclNlcXVlbmNlQ2xhc3NpZmljYXRpb24uZnJvbV9wcmV0cmFpbmVkKCUwQSUyMCUyMCUyMCUyMCUyMmdvb2dsZSUyRmZuZXQtYmFzZSUyMiUyQyUyMG51bV9sYWJlbHMlM0RudW1fbGFiZWxzJTJDJTIwcHJvYmxlbV90eXBlJTNEJTIybXVsdGlfbGFiZWxfY2xhc3NpZmljYXRpb24lMjIlMEEpJTBBJTBBbGFiZWxzJTIwJTNEJTIwdG9yY2guc3VtKCUwQSUyMCUyMCUyMCUyMHRvcmNoLm5uLmZ1bmN0aW9uYWwub25lX2hvdChwcmVkaWN0ZWRfY2xhc3NfaWRzJTVCTm9uZSUyQyUyMCUzQSU1RC5jbG9uZSgpJTJDJTIwbnVtX2NsYXNzZXMlM0RudW1fbGFiZWxzKSUyQyUyMGRpbSUzRDElMEEpLnRvKHRvcmNoLmZsb2F0KSUwQWxvc3MlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyUyQyUyMGxhYmVscyUzRGxhYmVscykubG9zcw==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, FNetForSequenceClassification

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;google/fnet-base&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = FNetForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;google/fnet-base&quot;</span>, problem_type=<span class="hljs-string">&quot;multi_label_classification&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    logits = model(**inputs).logits

<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_class_ids = torch.arange(<span class="hljs-number">0</span>, logits.shape[-<span class="hljs-number">1</span>])[torch.sigmoid(logits).squeeze(dim=<span class="hljs-number">0</span>) &gt; <span class="hljs-number">0.5</span>]

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># To train a model on \`num_labels\` classes, you can pass \`num_labels=num_labels\` to \`.from_pretrained(...)\`</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>num_labels = <span class="hljs-built_in">len</span>(model.config.id2label)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = FNetForSequenceClassification.from_pretrained(
<span class="hljs-meta">... </span>    <span class="hljs-string">&quot;google/fnet-base&quot;</span>, num_labels=num_labels, problem_type=<span class="hljs-string">&quot;multi_label_classification&quot;</span>
<span class="hljs-meta">... </span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.<span class="hljs-built_in">sum</span>(
<span class="hljs-meta">... </span>    torch.nn.functional.one_hot(predicted_class_ids[<span class="hljs-literal">None</span>, :].clone(), num_classes=num_labels), dim=<span class="hljs-number">1</span>
<span class="hljs-meta">... </span>).to(torch.<span class="hljs-built_in">float</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = model(**inputs, labels=labels).loss`,wrap:!1}}),{c(){t=d("p"),t.textContent=y,l=s(),u(p.$$.fragment)},l(n){t=c(n,"P",{"data-svelte-h":!0}),h(t)!=="svelte-1l8e32d"&&(t.textContent=y),l=r(n),f(p.$$.fragment,n)},m(n,w){m(n,t,w),m(n,l,w),g(p,n,w),M=!0},p:x,i(n){M||(_(p.$$.fragment,n),M=!0)},o(n){b(p.$$.fragment,n),M=!1},d(n){n&&(i(t),i(l)),T(p,n)}}}function Br(k){let t,y=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=d("p"),t.innerHTML=y},l(l){t=c(l,"P",{"data-svelte-h":!0}),h(t)!=="svelte-fincs2"&&(t.innerHTML=y)},m(l,p){m(l,t,p)},p:x,d(l){l&&i(t)}}}function Rr(k){let t,y="Example:",l,p,M;return p=new _e({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBGTmV0Rm9yTXVsdGlwbGVDaG9pY2UlMEFpbXBvcnQlMjB0b3JjaCUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMmdvb2dsZSUyRmZuZXQtYmFzZSUyMiklMEFtb2RlbCUyMCUzRCUyMEZOZXRGb3JNdWx0aXBsZUNob2ljZS5mcm9tX3ByZXRyYWluZWQoJTIyZ29vZ2xlJTJGZm5ldC1iYXNlJTIyKSUwQSUwQXByb21wdCUyMCUzRCUyMCUyMkluJTIwSXRhbHklMkMlMjBwaXp6YSUyMHNlcnZlZCUyMGluJTIwZm9ybWFsJTIwc2V0dGluZ3MlMkMlMjBzdWNoJTIwYXMlMjBhdCUyMGElMjByZXN0YXVyYW50JTJDJTIwaXMlMjBwcmVzZW50ZWQlMjB1bnNsaWNlZC4lMjIlMEFjaG9pY2UwJTIwJTNEJTIwJTIySXQlMjBpcyUyMGVhdGVuJTIwd2l0aCUyMGElMjBmb3JrJTIwYW5kJTIwYSUyMGtuaWZlLiUyMiUwQWNob2ljZTElMjAlM0QlMjAlMjJJdCUyMGlzJTIwZWF0ZW4lMjB3aGlsZSUyMGhlbGQlMjBpbiUyMHRoZSUyMGhhbmQuJTIyJTBBbGFiZWxzJTIwJTNEJTIwdG9yY2gudGVuc29yKDApLnVuc3F1ZWV6ZSgwKSUyMCUyMCUyMyUyMGNob2ljZTAlMjBpcyUyMGNvcnJlY3QlMjAoYWNjb3JkaW5nJTIwdG8lMjBXaWtpcGVkaWElMjAlM0IpKSUyQyUyMGJhdGNoJTIwc2l6ZSUyMDElMEElMEFlbmNvZGluZyUyMCUzRCUyMHRva2VuaXplciglNUJwcm9tcHQlMkMlMjBwcm9tcHQlNUQlMkMlMjAlNUJjaG9pY2UwJTJDJTIwY2hvaWNlMSU1RCUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIlMkMlMjBwYWRkaW5nJTNEVHJ1ZSklMEFvdXRwdXRzJTIwJTNEJTIwbW9kZWwoKiolN0JrJTNBJTIwdi51bnNxdWVlemUoMCklMjBmb3IlMjBrJTJDJTIwdiUyMGluJTIwZW5jb2RpbmcuaXRlbXMoKSU3RCUyQyUyMGxhYmVscyUzRGxhYmVscyklMjAlMjAlMjMlMjBiYXRjaCUyMHNpemUlMjBpcyUyMDElMEElMEElMjMlMjB0aGUlMjBsaW5lYXIlMjBjbGFzc2lmaWVyJTIwc3RpbGwlMjBuZWVkcyUyMHRvJTIwYmUlMjB0cmFpbmVkJTBBbG9zcyUyMCUzRCUyMG91dHB1dHMubG9zcyUwQWxvZ2l0cyUyMCUzRCUyMG91dHB1dHMubG9naXRz",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, FNetForMultipleChoice
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;google/fnet-base&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = FNetForMultipleChoice.from_pretrained(<span class="hljs-string">&quot;google/fnet-base&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>prompt = <span class="hljs-string">&quot;In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>choice0 = <span class="hljs-string">&quot;It is eaten with a fork and a knife.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>choice1 = <span class="hljs-string">&quot;It is eaten while held in the hand.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.tensor(<span class="hljs-number">0</span>).unsqueeze(<span class="hljs-number">0</span>)  <span class="hljs-comment"># choice0 is correct (according to Wikipedia ;)), batch size 1</span>

<span class="hljs-meta">&gt;&gt;&gt; </span>encoding = tokenizer([prompt, prompt], [choice0, choice1], return_tensors=<span class="hljs-string">&quot;pt&quot;</span>, padding=<span class="hljs-literal">True</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**{k: v.unsqueeze(<span class="hljs-number">0</span>) <span class="hljs-keyword">for</span> k, v <span class="hljs-keyword">in</span> encoding.items()}, labels=labels)  <span class="hljs-comment"># batch size is 1</span>

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># the linear classifier still needs to be trained</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = outputs.loss
<span class="hljs-meta">&gt;&gt;&gt; </span>logits = outputs.logits`,wrap:!1}}),{c(){t=d("p"),t.textContent=y,l=s(),u(p.$$.fragment)},l(n){t=c(n,"P",{"data-svelte-h":!0}),h(t)!=="svelte-11lpom8"&&(t.textContent=y),l=r(n),f(p.$$.fragment,n)},m(n,w){m(n,t,w),m(n,l,w),g(p,n,w),M=!0},p:x,i(n){M||(_(p.$$.fragment,n),M=!0)},o(n){b(p.$$.fragment,n),M=!1},d(n){n&&(i(t),i(l)),T(p,n)}}}function Xr(k){let t,y=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=d("p"),t.innerHTML=y},l(l){t=c(l,"P",{"data-svelte-h":!0}),h(t)!=="svelte-fincs2"&&(t.innerHTML=y)},m(l,p){m(l,t,p)},p:x,d(l){l&&i(t)}}}function Sr(k){let t,y="Example:",l,p,M;return p=new _e({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBGTmV0Rm9yVG9rZW5DbGFzc2lmaWNhdGlvbiUwQWltcG9ydCUyMHRvcmNoJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIyZ29vZ2xlJTJGZm5ldC1iYXNlJTIyKSUwQW1vZGVsJTIwJTNEJTIwRk5ldEZvclRva2VuQ2xhc3NpZmljYXRpb24uZnJvbV9wcmV0cmFpbmVkKCUyMmdvb2dsZSUyRmZuZXQtYmFzZSUyMiklMEElMEFpbnB1dHMlMjAlM0QlMjB0b2tlbml6ZXIoJTBBJTIwJTIwJTIwJTIwJTIySHVnZ2luZ0ZhY2UlMjBpcyUyMGElMjBjb21wYW55JTIwYmFzZWQlMjBpbiUyMFBhcmlzJTIwYW5kJTIwTmV3JTIwWW9yayUyMiUyQyUyMGFkZF9zcGVjaWFsX3Rva2VucyUzREZhbHNlJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiUwQSklMEElMEF3aXRoJTIwdG9yY2gubm9fZ3JhZCgpJTNBJTBBJTIwJTIwJTIwJTIwbG9naXRzJTIwJTNEJTIwbW9kZWwoKippbnB1dHMpLmxvZ2l0cyUwQSUwQXByZWRpY3RlZF90b2tlbl9jbGFzc19pZHMlMjAlM0QlMjBsb2dpdHMuYXJnbWF4KC0xKSUwQSUwQSUyMyUyME5vdGUlMjB0aGF0JTIwdG9rZW5zJTIwYXJlJTIwY2xhc3NpZmllZCUyMHJhdGhlciUyMHRoZW4lMjBpbnB1dCUyMHdvcmRzJTIwd2hpY2glMjBtZWFucyUyMHRoYXQlMEElMjMlMjB0aGVyZSUyMG1pZ2h0JTIwYmUlMjBtb3JlJTIwcHJlZGljdGVkJTIwdG9rZW4lMjBjbGFzc2VzJTIwdGhhbiUyMHdvcmRzLiUwQSUyMyUyME11bHRpcGxlJTIwdG9rZW4lMjBjbGFzc2VzJTIwbWlnaHQlMjBhY2NvdW50JTIwZm9yJTIwdGhlJTIwc2FtZSUyMHdvcmQlMEFwcmVkaWN0ZWRfdG9rZW5zX2NsYXNzZXMlMjAlM0QlMjAlNUJtb2RlbC5jb25maWcuaWQybGFiZWwlNUJ0Lml0ZW0oKSU1RCUyMGZvciUyMHQlMjBpbiUyMHByZWRpY3RlZF90b2tlbl9jbGFzc19pZHMlNUIwJTVEJTVEJTBBcHJlZGljdGVkX3Rva2Vuc19jbGFzc2VzJTBBJTBBbGFiZWxzJTIwJTNEJTIwcHJlZGljdGVkX3Rva2VuX2NsYXNzX2lkcyUwQWxvc3MlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyUyQyUyMGxhYmVscyUzRGxhYmVscykubG9zcyUwQXJvdW5kKGxvc3MuaXRlbSgpJTJDJTIwMik=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, FNetForTokenClassification
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;google/fnet-base&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = FNetForTokenClassification.from_pretrained(<span class="hljs-string">&quot;google/fnet-base&quot;</span>)

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
...`,wrap:!1}}),{c(){t=d("p"),t.textContent=y,l=s(),u(p.$$.fragment)},l(n){t=c(n,"P",{"data-svelte-h":!0}),h(t)!=="svelte-11lpom8"&&(t.textContent=y),l=r(n),f(p.$$.fragment,n)},m(n,w){m(n,t,w),m(n,l,w),g(p,n,w),M=!0},p:x,i(n){M||(_(p.$$.fragment,n),M=!0)},o(n){b(p.$$.fragment,n),M=!1},d(n){n&&(i(t),i(l)),T(p,n)}}}function Gr(k){let t,y=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=d("p"),t.innerHTML=y},l(l){t=c(l,"P",{"data-svelte-h":!0}),h(t)!=="svelte-fincs2"&&(t.innerHTML=y)},m(l,p){m(l,t,p)},p:x,d(l){l&&i(t)}}}function Vr(k){let t,y="Example:",l,p,M;return p=new _e({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBGTmV0Rm9yUXVlc3Rpb25BbnN3ZXJpbmclMEFpbXBvcnQlMjB0b3JjaCUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMmdvb2dsZSUyRmZuZXQtYmFzZSUyMiklMEFtb2RlbCUyMCUzRCUyMEZOZXRGb3JRdWVzdGlvbkFuc3dlcmluZy5mcm9tX3ByZXRyYWluZWQoJTIyZ29vZ2xlJTJGZm5ldC1iYXNlJTIyKSUwQSUwQXF1ZXN0aW9uJTJDJTIwdGV4dCUyMCUzRCUyMCUyMldobyUyMHdhcyUyMEppbSUyMEhlbnNvbiUzRiUyMiUyQyUyMCUyMkppbSUyMEhlbnNvbiUyMHdhcyUyMGElMjBuaWNlJTIwcHVwcGV0JTIyJTBBJTBBaW5wdXRzJTIwJTNEJTIwdG9rZW5pemVyKHF1ZXN0aW9uJTJDJTIwdGV4dCUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpJTBBd2l0aCUyMHRvcmNoLm5vX2dyYWQoKSUzQSUwQSUyMCUyMCUyMCUyMG91dHB1dHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyklMEElMEFhbnN3ZXJfc3RhcnRfaW5kZXglMjAlM0QlMjBvdXRwdXRzLnN0YXJ0X2xvZ2l0cy5hcmdtYXgoKSUwQWFuc3dlcl9lbmRfaW5kZXglMjAlM0QlMjBvdXRwdXRzLmVuZF9sb2dpdHMuYXJnbWF4KCklMEElMEFwcmVkaWN0X2Fuc3dlcl90b2tlbnMlMjAlM0QlMjBpbnB1dHMuaW5wdXRfaWRzJTVCMCUyQyUyMGFuc3dlcl9zdGFydF9pbmRleCUyMCUzQSUyMGFuc3dlcl9lbmRfaW5kZXglMjAlMkIlMjAxJTVEJTBBdG9rZW5pemVyLmRlY29kZShwcmVkaWN0X2Fuc3dlcl90b2tlbnMlMkMlMjBza2lwX3NwZWNpYWxfdG9rZW5zJTNEVHJ1ZSklMEElMEElMjMlMjB0YXJnZXQlMjBpcyUyMCUyMm5pY2UlMjBwdXBwZXQlMjIlMEF0YXJnZXRfc3RhcnRfaW5kZXglMjAlM0QlMjB0b3JjaC50ZW5zb3IoJTVCMTQlNUQpJTBBdGFyZ2V0X2VuZF9pbmRleCUyMCUzRCUyMHRvcmNoLnRlbnNvciglNUIxNSU1RCklMEElMEFvdXRwdXRzJTIwJTNEJTIwbW9kZWwoKippbnB1dHMlMkMlMjBzdGFydF9wb3NpdGlvbnMlM0R0YXJnZXRfc3RhcnRfaW5kZXglMkMlMjBlbmRfcG9zaXRpb25zJTNEdGFyZ2V0X2VuZF9pbmRleCklMEFsb3NzJTIwJTNEJTIwb3V0cHV0cy5sb3NzJTBBcm91bmQobG9zcy5pdGVtKCklMkMlMjAyKQ==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, FNetForQuestionAnswering
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;google/fnet-base&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = FNetForQuestionAnswering.from_pretrained(<span class="hljs-string">&quot;google/fnet-base&quot;</span>)

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
...`,wrap:!1}}),{c(){t=d("p"),t.textContent=y,l=s(),u(p.$$.fragment)},l(n){t=c(n,"P",{"data-svelte-h":!0}),h(t)!=="svelte-11lpom8"&&(t.textContent=y),l=r(n),f(p.$$.fragment,n)},m(n,w){m(n,t,w),m(n,l,w),g(p,n,w),M=!0},p:x,i(n){M||(_(p.$$.fragment,n),M=!0)},o(n){b(p.$$.fragment,n),M=!1},d(n){n&&(i(t),i(l)),T(p,n)}}}function Hr(k){let t,y,l,p,M,n="<em>This model was released on 2021-05-09 and added to Hugging Face Transformers on 2021-09-20.</em>",w,Se,$n,be,$s='<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/>',Nn,Ge,xn,Ve,Ns=`The FNet model was proposed in <a href="https://huggingface.co/papers/2105.03824" rel="nofollow">FNet: Mixing Tokens with Fourier Transforms</a> by
James Lee-Thorp, Joshua Ainslie, Ilya Eckstein, Santiago Ontanon. The model replaces the self-attention layer in a BERT
model with a fourier transform which returns only the real parts of the transform. The model is significantly faster
than the BERT model because it has fewer parameters and is more memory efficient. The model achieves about 92-97%
accuracy of BERT counterparts on GLUE benchmark, and trains much faster than the BERT model. The abstract from the
paper is the following:`,zn,He,xs=`<em>We show that Transformer encoder architectures can be sped up, with limited accuracy costs, by replacing the
self-attention sublayers with simple linear transformations that “mix” input tokens. These linear mixers, along with
standard nonlinearities in feed-forward layers, prove competent at modeling semantic relationships in several text
classification tasks. Most surprisingly, we find that replacing the self-attention sublayer in a Transformer encoder
with a standard, unparameterized Fourier Transform achieves 92-97% of the accuracy of BERT counterparts on the GLUE
benchmark, but trains 80% faster on GPUs and 70% faster on TPUs at standard 512 input lengths. At longer input lengths,
our FNet model is significantly faster: when compared to the “efficient” Transformers on the Long Range Arena
benchmark, FNet matches the accuracy of the most accurate models, while outpacing the fastest models across all
sequence lengths on GPUs (and across relatively shorter lengths on TPUs). Finally, FNet has a light memory footprint
and is particularly efficient at smaller model sizes; for a fixed speed and accuracy budget, small FNet models
outperform Transformer counterparts.</em>`,jn,Pe,zs='This model was contributed by <a href="https://huggingface.co/gchhablani" rel="nofollow">gchhablani</a>. The original code can be found <a href="https://github.com/google-research/google-research/tree/master/f_net" rel="nofollow">here</a>.',Jn,Le,Un,Ee,js=`The model was trained without an attention mask as it is based on Fourier Transform. The model was trained with
maximum sequence length 512 which includes pad tokens. Hence, it is highly recommended to use the same maximum
sequence length for fine-tuning and inference.`,Cn,Qe,Zn,Ye,Js='<li><a href="../tasks/sequence_classification">Text classification task guide</a></li> <li><a href="../tasks/token_classification">Token classification task guide</a></li> <li><a href="../tasks/question_answering">Question answering task guide</a></li> <li><a href="../tasks/masked_language_modeling">Masked language modeling task guide</a></li> <li><a href="../tasks/multiple_choice">Multiple choice task guide</a></li>',In,Ae,qn,X,De,ao,Ct,Us=`This is the configuration class to store the configuration of a <a href="/docs/transformers/v4.56.2/en/model_doc/fnet#transformers.FNetModel">FNetModel</a>. It is used to instantiate an FNet
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the FNet
<a href="https://huggingface.co/google/fnet-base" rel="nofollow">google/fnet-base</a> architecture.`,io,Zt,Cs=`Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> and can be used to control the model outputs. Read the
documentation from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> for more information.`,lo,Te,Wn,Oe,Bn,N,Ke,co,It,Zs=`Construct an FNet tokenizer. Adapted from <code>AlbertTokenizer</code>. Based on
<a href="https://github.com/google/sentencepiece" rel="nofollow">SentencePiece</a>. This tokenizer inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer">PreTrainedTokenizer</a>
which contains most of the main methods. Users should refer to this superclass for more information regarding those
methods.`,po,oe,et,mo,qt,Is=`Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
adding special tokens. An FNet sequence has the following format:`,ho,Wt,qs="<li>single sequence: <code>[CLS] X [SEP]</code></li> <li>pair of sequences: <code>[CLS] A [SEP] B [SEP]</code></li>",uo,ye,tt,fo,Bt,Ws=`Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
special tokens using the tokenizer <code>prepare_for_model</code> method.`,go,se,nt,_o,Rt,Bs=`Create the token type IDs corresponding to the sequences passed. <a href="../glossary#token-type-ids">What are token type
IDs?</a>`,bo,Xt,Rs="Should be overridden in a subclass if the model has a special way of building those.",To,St,ot,Rn,st,Xn,O,rt,yo,Gt,Xs=`Construct a “fast” FNetTokenizer (backed by HuggingFace’s <em>tokenizers</em> library). Adapted from
<a href="/docs/transformers/v4.56.2/en/model_doc/albert#transformers.AlbertTokenizerFast">AlbertTokenizerFast</a>. Based on
<a href="https://huggingface.co/docs/tokenizers/python/latest/components.html?highlight=unigram#models" rel="nofollow">Unigram</a>. This
tokenizer inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast">PreTrainedTokenizerFast</a> which contains most of the main methods. Users should refer to
this superclass for more information regarding those methods`,Mo,re,at,wo,Vt,Ss=`Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
adding special tokens. An FNet sequence has the following format:`,ko,Ht,Gs="<li>single sequence: <code>[CLS] X [SEP]</code></li> <li>pair of sequences: <code>[CLS] A [SEP] B [SEP]</code></li>",Sn,it,Gn,z,lt,vo,Pt,Vs="The bare Fnet Model outputting raw hidden-states without any specific head on top.",Fo,Lt,Hs=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,$o,Et,Ps=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,No,ae,dt,xo,Qt,Ls='The <a href="/docs/transformers/v4.56.2/en/model_doc/fnet#transformers.FNetModel">FNetModel</a> forward method, overrides the <code>__call__</code> special method.',zo,Me,Vn,ct,Hn,j,pt,jo,Yt,Es="FNet Model with two heads on top as done during the pretraining: a <code>masked language modeling</code> head and a <code>next sentence prediction (classification)</code> head.",Jo,At,Qs=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Uo,Dt,Ys=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Co,L,mt,Zo,Ot,As='The <a href="/docs/transformers/v4.56.2/en/model_doc/fnet#transformers.FNetForPreTraining">FNetForPreTraining</a> forward method, overrides the <code>__call__</code> special method.',Io,we,qo,ke,Pn,ht,Ln,J,ut,Wo,Kt,Ds="The Fnet Model with a <code>language modeling</code> head on top.”",Bo,en,Os=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Ro,tn,Ks=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Xo,E,ft,So,nn,er='The <a href="/docs/transformers/v4.56.2/en/model_doc/fnet#transformers.FNetForMaskedLM">FNetForMaskedLM</a> forward method, overrides the <code>__call__</code> special method.',Go,ve,Vo,Fe,En,gt,Qn,U,_t,Ho,on,tr="FNet Model with a <code>next sentence prediction (classification)</code> head on top.",Po,sn,nr=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Lo,rn,or=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Eo,Q,bt,Qo,an,sr='The <a href="/docs/transformers/v4.56.2/en/model_doc/fnet#transformers.FNetForNextSentencePrediction">FNetForNextSentencePrediction</a> forward method, overrides the <code>__call__</code> special method.',Yo,$e,Ao,Ne,Yn,Tt,An,C,yt,Do,ln,rr=`FNet Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled
output) e.g. for GLUE tasks.`,Oo,dn,ar=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Ko,cn,ir=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,es,R,Mt,ts,pn,lr='The <a href="/docs/transformers/v4.56.2/en/model_doc/fnet#transformers.FNetForSequenceClassification">FNetForSequenceClassification</a> forward method, overrides the <code>__call__</code> special method.',ns,xe,os,ze,ss,je,Dn,wt,On,Z,kt,rs,mn,dr=`The Fnet Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
softmax) e.g. for RocStories/SWAG tasks.`,as,hn,cr=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,is,un,pr=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,ls,Y,vt,ds,fn,mr='The <a href="/docs/transformers/v4.56.2/en/model_doc/fnet#transformers.FNetForMultipleChoice">FNetForMultipleChoice</a> forward method, overrides the <code>__call__</code> special method.',cs,Je,ps,Ue,Kn,Ft,eo,I,$t,ms,gn,hr=`The Fnet transformer with a token classification head on top (a linear layer on top of the hidden-states
output) e.g. for Named-Entity-Recognition (NER) tasks.`,hs,_n,ur=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,us,bn,fr=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,fs,A,Nt,gs,Tn,gr='The <a href="/docs/transformers/v4.56.2/en/model_doc/fnet#transformers.FNetForTokenClassification">FNetForTokenClassification</a> forward method, overrides the <code>__call__</code> special method.',_s,Ce,bs,Ze,to,xt,no,q,zt,Ts,yn,_r=`The Fnet transformer with a span classification head on top for extractive question-answering tasks like
SQuAD (a linear layer on top of the hidden-states output to compute <code>span start logits</code> and <code>span end logits</code>).`,ys,Mn,br=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Ms,wn,Tr=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,ws,D,jt,ks,kn,yr='The <a href="/docs/transformers/v4.56.2/en/model_doc/fnet#transformers.FNetForQuestionAnswering">FNetForQuestionAnswering</a> forward method, overrides the <code>__call__</code> special method.',vs,Ie,Fs,qe,oo,Jt,so,vn,ro;return Se=new B({props:{title:"FNet",local:"fnet",headingTag:"h1"}}),Ge=new B({props:{title:"Overview",local:"overview",headingTag:"h2"}}),Le=new B({props:{title:"Usage tips",local:"usage-tips",headingTag:"h2"}}),Qe=new B({props:{title:"Resources",local:"resources",headingTag:"h2"}}),Ae=new B({props:{title:"FNetConfig",local:"transformers.FNetConfig",headingTag:"h2"}}),De=new $({props:{name:"class transformers.FNetConfig",anchor:"transformers.FNetConfig",parameters:[{name:"vocab_size",val:" = 32000"},{name:"hidden_size",val:" = 768"},{name:"num_hidden_layers",val:" = 12"},{name:"intermediate_size",val:" = 3072"},{name:"hidden_act",val:" = 'gelu_new'"},{name:"hidden_dropout_prob",val:" = 0.1"},{name:"max_position_embeddings",val:" = 512"},{name:"type_vocab_size",val:" = 4"},{name:"initializer_range",val:" = 0.02"},{name:"layer_norm_eps",val:" = 1e-12"},{name:"use_tpu_fourier_optimizations",val:" = False"},{name:"tpu_short_seq_length",val:" = 512"},{name:"pad_token_id",val:" = 3"},{name:"bos_token_id",val:" = 1"},{name:"eos_token_id",val:" = 2"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.FNetConfig.vocab_size",description:`<strong>vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 32000) &#x2014;
Vocabulary size of the FNet model. Defines the number of different tokens that can be represented by the
<code>inputs_ids</code> passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/fnet#transformers.FNetModel">FNetModel</a> or <code>TFFNetModel</code>.`,name:"vocab_size"},{anchor:"transformers.FNetConfig.hidden_size",description:`<strong>hidden_size</strong> (<code>int</code>, <em>optional</em>, defaults to 768) &#x2014;
Dimension of the encoder layers and the pooler layer.`,name:"hidden_size"},{anchor:"transformers.FNetConfig.num_hidden_layers",description:`<strong>num_hidden_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 12) &#x2014;
Number of hidden layers in the Transformer encoder.`,name:"num_hidden_layers"},{anchor:"transformers.FNetConfig.intermediate_size",description:`<strong>intermediate_size</strong> (<code>int</code>, <em>optional</em>, defaults to 3072) &#x2014;
Dimension of the &#x201C;intermediate&#x201D; (i.e., feed-forward) layer in the Transformer encoder.`,name:"intermediate_size"},{anchor:"transformers.FNetConfig.hidden_act",description:`<strong>hidden_act</strong> (<code>str</code> or <code>function</code>, <em>optional</em>, defaults to <code>&quot;gelu_new&quot;</code>) &#x2014;
The non-linear activation function (function or string) in the encoder and pooler. If string, <code>&quot;gelu&quot;</code>,
<code>&quot;relu&quot;</code>, <code>&quot;selu&quot;</code> and <code>&quot;gelu_new&quot;</code> are supported.`,name:"hidden_act"},{anchor:"transformers.FNetConfig.hidden_dropout_prob",description:`<strong>hidden_dropout_prob</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.`,name:"hidden_dropout_prob"},{anchor:"transformers.FNetConfig.max_position_embeddings",description:`<strong>max_position_embeddings</strong> (<code>int</code>, <em>optional</em>, defaults to 512) &#x2014;
The maximum sequence length that this model might ever be used with. Typically set this to something large
just in case (e.g., 512 or 1024 or 2048).`,name:"max_position_embeddings"},{anchor:"transformers.FNetConfig.type_vocab_size",description:`<strong>type_vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 4) &#x2014;
The vocabulary size of the <code>token_type_ids</code> passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/fnet#transformers.FNetModel">FNetModel</a> or <code>TFFNetModel</code>.`,name:"type_vocab_size"},{anchor:"transformers.FNetConfig.initializer_range",description:`<strong>initializer_range</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.`,name:"initializer_range"},{anchor:"transformers.FNetConfig.layer_norm_eps",description:`<strong>layer_norm_eps</strong> (<code>float</code>, <em>optional</em>, defaults to 1e-12) &#x2014;
The epsilon used by the layer normalization layers.`,name:"layer_norm_eps"},{anchor:"transformers.FNetConfig.use_tpu_fourier_optimizations",description:`<strong>use_tpu_fourier_optimizations</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Determines whether to use TPU optimized FFTs. If <code>True</code>, the model will favor axis-wise FFTs transforms.
Set to <code>False</code> for GPU/CPU hardware, in which case n-dimensional FFTs are used.`,name:"use_tpu_fourier_optimizations"},{anchor:"transformers.FNetConfig.tpu_short_seq_length",description:`<strong>tpu_short_seq_length</strong> (<code>int</code>, <em>optional</em>, defaults to 512) &#x2014;
The sequence length that is expected by the model when using TPUs. This will be used to initialize the DFT
matrix only when <em>use_tpu_fourier_optimizations</em> is set to <code>True</code> and the input sequence is shorter than or
equal to 4096 tokens.`,name:"tpu_short_seq_length"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/fnet/configuration_fnet.py#L24"}}),Te=new ge({props:{anchor:"transformers.FNetConfig.example",$$slots:{default:[Nr]},$$scope:{ctx:k}}}),Oe=new B({props:{title:"FNetTokenizer",local:"transformers.FNetTokenizer",headingTag:"h2"}}),Ke=new $({props:{name:"class transformers.FNetTokenizer",anchor:"transformers.FNetTokenizer",parameters:[{name:"vocab_file",val:""},{name:"do_lower_case",val:" = False"},{name:"remove_space",val:" = True"},{name:"keep_accents",val:" = True"},{name:"unk_token",val:" = '<unk>'"},{name:"sep_token",val:" = '[SEP]'"},{name:"pad_token",val:" = '<pad>'"},{name:"cls_token",val:" = '[CLS]'"},{name:"mask_token",val:" = '[MASK]'"},{name:"sp_model_kwargs",val:": typing.Optional[dict[str, typing.Any]] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.FNetTokenizer.vocab_file",description:`<strong>vocab_file</strong> (<code>str</code>) &#x2014;
<a href="https://github.com/google/sentencepiece" rel="nofollow">SentencePiece</a> file (generally has a <em>.spm</em> extension) that
contains the vocabulary necessary to instantiate a tokenizer.`,name:"vocab_file"},{anchor:"transformers.FNetTokenizer.do_lower_case",description:`<strong>do_lower_case</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to lowercase the input when tokenizing.`,name:"do_lower_case"},{anchor:"transformers.FNetTokenizer.remove_space",description:`<strong>remove_space</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to strip the text when tokenizing (removing excess spaces before and after the string).`,name:"remove_space"},{anchor:"transformers.FNetTokenizer.keep_accents",description:`<strong>keep_accents</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to keep accents when tokenizing.`,name:"keep_accents"},{anchor:"transformers.FNetTokenizer.unk_token",description:`<strong>unk_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;unk&gt;&quot;</code>) &#x2014;
The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
token instead.`,name:"unk_token"},{anchor:"transformers.FNetTokenizer.sep_token",description:`<strong>sep_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;[SEP]&quot;</code>) &#x2014;
The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
sequence classification or for a text and a question for question answering. It is also used as the last
token of a sequence built with special tokens.`,name:"sep_token"},{anchor:"transformers.FNetTokenizer.pad_token",description:`<strong>pad_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;pad&gt;&quot;</code>) &#x2014;
The token used for padding, for example when batching sequences of different lengths.`,name:"pad_token"},{anchor:"transformers.FNetTokenizer.cls_token",description:`<strong>cls_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;[CLS]&quot;</code>) &#x2014;
The classifier token which is used when doing sequence classification (classification of the whole sequence
instead of per-token classification). It is the first token of the sequence when built with special tokens.`,name:"cls_token"},{anchor:"transformers.FNetTokenizer.mask_token",description:`<strong>mask_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;[MASK]&quot;</code>) &#x2014;
The token used for masking values. This is the token used when training this model with masked language
modeling. This is the token which the model will try to predict.`,name:"mask_token"},{anchor:"transformers.FNetTokenizer.sp_model_kwargs",description:`<strong>sp_model_kwargs</strong> (<code>dict</code>, <em>optional</em>) &#x2014;
Will be passed to the <code>SentencePieceProcessor.__init__()</code> method. The <a href="https://github.com/google/sentencepiece/tree/master/python" rel="nofollow">Python wrapper for
SentencePiece</a> can be used, among other things,
to set:</p>
<ul>
<li>
<p><code>enable_sampling</code>: Enable subword regularization.</p>
</li>
<li>
<p><code>nbest_size</code>: Sampling parameters for unigram. Invalid for BPE-Dropout.</p>
<ul>
<li><code>nbest_size = {0,1}</code>: No sampling is performed.</li>
<li><code>nbest_size &gt; 1</code>: samples from the nbest_size results.</li>
<li><code>nbest_size &lt; 0</code>: assuming that nbest_size is infinite and samples from the all hypothesis (lattice)
using forward-filtering-and-backward-sampling algorithm.</li>
</ul>
</li>
<li>
<p><code>alpha</code>: Smoothing parameter for unigram sampling, and dropout probability of merge operations for
BPE-dropout.</p>
</li>
</ul>`,name:"sp_model_kwargs"},{anchor:"transformers.FNetTokenizer.sp_model",description:`<strong>sp_model</strong> (<code>SentencePieceProcessor</code>) &#x2014;
The <em>SentencePiece</em> processor that is used for every conversion (string, tokens and IDs).`,name:"sp_model"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/fnet/tokenization_fnet.py#L37"}}),et=new $({props:{name:"build_inputs_with_special_tokens",anchor:"transformers.FNetTokenizer.build_inputs_with_special_tokens",parameters:[{name:"token_ids_0",val:": list"},{name:"token_ids_1",val:": typing.Optional[list[int]] = None"}],parametersDescription:[{anchor:"transformers.FNetTokenizer.build_inputs_with_special_tokens.token_ids_0",description:`<strong>token_ids_0</strong> (<code>List[int]</code>) &#x2014;
List of IDs to which the special tokens will be added.`,name:"token_ids_0"},{anchor:"transformers.FNetTokenizer.build_inputs_with_special_tokens.token_ids_1",description:`<strong>token_ids_1</strong> (<code>List[int]</code>, <em>optional</em>) &#x2014;
Optional second list of IDs for sequence pairs.`,name:"token_ids_1"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/fnet/tokenization_fnet.py#L243",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>List of <a href="../glossary#input-ids">input IDs</a> with the appropriate special tokens.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>List[int]</code></p>
`}}),tt=new $({props:{name:"get_special_tokens_mask",anchor:"transformers.FNetTokenizer.get_special_tokens_mask",parameters:[{name:"token_ids_0",val:": list"},{name:"token_ids_1",val:": typing.Optional[list[int]] = None"},{name:"already_has_special_tokens",val:": bool = False"}],parametersDescription:[{anchor:"transformers.FNetTokenizer.get_special_tokens_mask.token_ids_0",description:`<strong>token_ids_0</strong> (<code>List[int]</code>) &#x2014;
List of IDs.`,name:"token_ids_0"},{anchor:"transformers.FNetTokenizer.get_special_tokens_mask.token_ids_1",description:`<strong>token_ids_1</strong> (<code>List[int]</code>, <em>optional</em>) &#x2014;
Optional second list of IDs for sequence pairs.`,name:"token_ids_1"},{anchor:"transformers.FNetTokenizer.get_special_tokens_mask.already_has_special_tokens",description:`<strong>already_has_special_tokens</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not the token list is already formatted with special tokens for the model.`,name:"already_has_special_tokens"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/fnet/tokenization_fnet.py#L268",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>List[int]</code></p>
`}}),nt=new $({props:{name:"create_token_type_ids_from_sequences",anchor:"transformers.FNetTokenizer.create_token_type_ids_from_sequences",parameters:[{name:"token_ids_0",val:": list"},{name:"token_ids_1",val:": typing.Optional[list[int]] = None"}],parametersDescription:[{anchor:"transformers.FNetTokenizer.create_token_type_ids_from_sequences.token_ids_0",description:"<strong>token_ids_0</strong> (<code>list[int]</code>) &#x2014; The first tokenized sequence.",name:"token_ids_0"},{anchor:"transformers.FNetTokenizer.create_token_type_ids_from_sequences.token_ids_1",description:"<strong>token_ids_1</strong> (<code>list[int]</code>, <em>optional</em>) &#x2014; The second tokenized sequence.",name:"token_ids_1"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L3432",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>The token type ids.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>list[int]</code></p>
`}}),ot=new $({props:{name:"save_vocabulary",anchor:"transformers.FNetTokenizer.save_vocabulary",parameters:[{name:"save_directory",val:": str"},{name:"filename_prefix",val:": typing.Optional[str] = None"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/fnet/tokenization_fnet.py#L296"}}),st=new B({props:{title:"FNetTokenizerFast",local:"transformers.FNetTokenizerFast",headingTag:"h2"}}),rt=new $({props:{name:"class transformers.FNetTokenizerFast",anchor:"transformers.FNetTokenizerFast",parameters:[{name:"vocab_file",val:" = None"},{name:"tokenizer_file",val:" = None"},{name:"do_lower_case",val:" = False"},{name:"remove_space",val:" = True"},{name:"keep_accents",val:" = True"},{name:"unk_token",val:" = '<unk>'"},{name:"sep_token",val:" = '[SEP]'"},{name:"pad_token",val:" = '<pad>'"},{name:"cls_token",val:" = '[CLS]'"},{name:"mask_token",val:" = '[MASK]'"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.FNetTokenizerFast.vocab_file",description:`<strong>vocab_file</strong> (<code>str</code>) &#x2014;
<a href="https://github.com/google/sentencepiece" rel="nofollow">SentencePiece</a> file (generally has a <em>.spm</em> extension) that
contains the vocabulary necessary to instantiate a tokenizer.`,name:"vocab_file"},{anchor:"transformers.FNetTokenizerFast.do_lower_case",description:`<strong>do_lower_case</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to lowercase the input when tokenizing.`,name:"do_lower_case"},{anchor:"transformers.FNetTokenizerFast.remove_space",description:`<strong>remove_space</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to strip the text when tokenizing (removing excess spaces before and after the string).`,name:"remove_space"},{anchor:"transformers.FNetTokenizerFast.keep_accents",description:`<strong>keep_accents</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to keep accents when tokenizing.`,name:"keep_accents"},{anchor:"transformers.FNetTokenizerFast.unk_token",description:`<strong>unk_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;unk&gt;&quot;</code>) &#x2014;
The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
token instead.`,name:"unk_token"},{anchor:"transformers.FNetTokenizerFast.sep_token",description:`<strong>sep_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;[SEP]&quot;</code>) &#x2014;
The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
sequence classification or for a text and a question for question answering. It is also used as the last
token of a sequence built with special tokens.`,name:"sep_token"},{anchor:"transformers.FNetTokenizerFast.pad_token",description:`<strong>pad_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;pad&gt;&quot;</code>) &#x2014;
The token used for padding, for example when batching sequences of different lengths.`,name:"pad_token"},{anchor:"transformers.FNetTokenizerFast.cls_token",description:`<strong>cls_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;[CLS]&quot;</code>) &#x2014;
The classifier token which is used when doing sequence classification (classification of the whole sequence
instead of per-token classification). It is the first token of the sequence when built with special tokens.`,name:"cls_token"},{anchor:"transformers.FNetTokenizerFast.mask_token",description:`<strong>mask_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;[MASK]&quot;</code>) &#x2014;
The token used for masking values. This is the token used when training this model with masked language
modeling. This is the token which the model will try to predict.`,name:"mask_token"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/fnet/tokenization_fnet_fast.py#L38"}}),at=new $({props:{name:"build_inputs_with_special_tokens",anchor:"transformers.FNetTokenizerFast.build_inputs_with_special_tokens",parameters:[{name:"token_ids_0",val:": list"},{name:"token_ids_1",val:": typing.Optional[list[int]] = None"}],parametersDescription:[{anchor:"transformers.FNetTokenizerFast.build_inputs_with_special_tokens.token_ids_0",description:`<strong>token_ids_0</strong> (<code>List[int]</code>) &#x2014;
List of IDs to which the special tokens will be added`,name:"token_ids_0"},{anchor:"transformers.FNetTokenizerFast.build_inputs_with_special_tokens.token_ids_1",description:`<strong>token_ids_1</strong> (<code>List[int]</code>, <em>optional</em>) &#x2014;
Optional second list of IDs for sequence pairs.`,name:"token_ids_1"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/fnet/tokenization_fnet_fast.py#L116",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>list of <a href="../glossary#input-ids">input IDs</a> with the appropriate special tokens.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>List[int]</code></p>
`}}),it=new B({props:{title:"FNetModel",local:"transformers.FNetModel",headingTag:"h2"}}),lt=new $({props:{name:"class transformers.FNetModel",anchor:"transformers.FNetModel",parameters:[{name:"config",val:""},{name:"add_pooling_layer",val:" = True"}],parametersDescription:[{anchor:"transformers.FNetModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/fnet#transformers.FNetModel">FNetModel</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"},{anchor:"transformers.FNetModel.add_pooling_layer",description:`<strong>add_pooling_layer</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether to add a pooling layer`,name:"add_pooling_layer"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/fnet/modeling_fnet.py#L436"}}),dt=new $({props:{name:"forward",anchor:"transformers.FNetModel.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.FNetModel.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.FNetModel.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.FNetModel.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.FNetModel.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.FNetModel.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.FNetModel.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/fnet/modeling_fnet.py#L466",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutput"
>transformers.modeling_outputs.BaseModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/fnet#transformers.FNetConfig"
>FNetConfig</a>) and inputs.</p>
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
`}}),Me=new Xe({props:{$$slots:{default:[xr]},$$scope:{ctx:k}}}),ct=new B({props:{title:"FNetForPreTraining",local:"transformers.FNetForPreTraining",headingTag:"h2"}}),pt=new $({props:{name:"class transformers.FNetForPreTraining",anchor:"transformers.FNetForPreTraining",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.FNetForPreTraining.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/fnet#transformers.FNetForPreTraining">FNetForPreTraining</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/fnet/modeling_fnet.py#L543"}}),mt=new $({props:{name:"forward",anchor:"transformers.FNetForPreTraining.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"labels",val:": typing.Optional[torch.Tensor] = None"},{name:"next_sentence_label",val:": typing.Optional[torch.Tensor] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.FNetForPreTraining.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.FNetForPreTraining.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.FNetForPreTraining.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.FNetForPreTraining.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.FNetForPreTraining.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should be in <code>[-100, 0, ..., config.vocab_size]</code> (see <code>input_ids</code> docstring) Tokens with indices set to <code>-100</code> are ignored (masked), the
loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>`,name:"labels"},{anchor:"transformers.FNetForPreTraining.forward.next_sentence_label",description:`<strong>next_sentence_label</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the next sequence prediction (classification) loss. Input should be a sequence pair
(see <code>input_ids</code> docstring) Indices should be in <code>[0, 1]</code>:</p>
<ul>
<li>0 indicates sequence B is a continuation of sequence A,</li>
<li>1 indicates sequence B is a random sequence.</li>
</ul>`,name:"next_sentence_label"},{anchor:"transformers.FNetForPreTraining.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.FNetForPreTraining.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/fnet/modeling_fnet.py#L562",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <code>transformers.models.fnet.modeling_fnet.FNetForPreTrainingOutput</code> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/fnet#transformers.FNetConfig"
>FNetConfig</a>) and inputs.</p>
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
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>transformers.models.fnet.modeling_fnet.FNetForPreTrainingOutput</code> or <code>tuple(torch.FloatTensor)</code></p>
`}}),we=new Xe({props:{$$slots:{default:[zr]},$$scope:{ctx:k}}}),ke=new ge({props:{anchor:"transformers.FNetForPreTraining.forward.example",$$slots:{default:[jr]},$$scope:{ctx:k}}}),ht=new B({props:{title:"FNetForMaskedLM",local:"transformers.FNetForMaskedLM",headingTag:"h2"}}),ut=new $({props:{name:"class transformers.FNetForMaskedLM",anchor:"transformers.FNetForMaskedLM",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.FNetForMaskedLM.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/fnet#transformers.FNetForMaskedLM">FNetForMaskedLM</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/fnet/modeling_fnet.py#L633"}}),ft=new $({props:{name:"forward",anchor:"transformers.FNetForMaskedLM.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"labels",val:": typing.Optional[torch.Tensor] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.FNetForMaskedLM.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.FNetForMaskedLM.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.FNetForMaskedLM.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.FNetForMaskedLM.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.FNetForMaskedLM.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should be in <code>[-100, 0, ..., config.vocab_size]</code> (see <code>input_ids</code> docstring) Tokens with indices set to <code>-100</code> are ignored (masked), the
loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.FNetForMaskedLM.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.FNetForMaskedLM.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/fnet/modeling_fnet.py#L652",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.MaskedLMOutput"
>transformers.modeling_outputs.MaskedLMOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/fnet#transformers.FNetConfig"
>FNetConfig</a>) and inputs.</p>
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
`}}),ve=new Xe({props:{$$slots:{default:[Jr]},$$scope:{ctx:k}}}),Fe=new ge({props:{anchor:"transformers.FNetForMaskedLM.forward.example",$$slots:{default:[Ur]},$$scope:{ctx:k}}}),gt=new B({props:{title:"FNetForNextSentencePrediction",local:"transformers.FNetForNextSentencePrediction",headingTag:"h2"}}),_t=new $({props:{name:"class transformers.FNetForNextSentencePrediction",anchor:"transformers.FNetForNextSentencePrediction",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.FNetForNextSentencePrediction.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/fnet#transformers.FNetForNextSentencePrediction">FNetForNextSentencePrediction</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/fnet/modeling_fnet.py#L700"}}),bt=new $({props:{name:"forward",anchor:"transformers.FNetForNextSentencePrediction.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"labels",val:": typing.Optional[torch.Tensor] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.FNetForNextSentencePrediction.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.FNetForNextSentencePrediction.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.FNetForNextSentencePrediction.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.FNetForNextSentencePrediction.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.FNetForNextSentencePrediction.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the next sequence prediction (classification) loss. Input should be a sequence pair
(see <code>input_ids</code> docstring). Indices should be in <code>[0, 1]</code>:</p>
<ul>
<li>0 indicates sequence B is a continuation of sequence A,</li>
<li>1 indicates sequence B is a random sequence.</li>
</ul>`,name:"labels"},{anchor:"transformers.FNetForNextSentencePrediction.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.FNetForNextSentencePrediction.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/fnet/modeling_fnet.py#L710",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.NextSentencePredictorOutput"
>transformers.modeling_outputs.NextSentencePredictorOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/fnet#transformers.FNetConfig"
>FNetConfig</a>) and inputs.</p>
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
`}}),$e=new Xe({props:{$$slots:{default:[Cr]},$$scope:{ctx:k}}}),Ne=new ge({props:{anchor:"transformers.FNetForNextSentencePrediction.forward.example",$$slots:{default:[Zr]},$$scope:{ctx:k}}}),Tt=new B({props:{title:"FNetForSequenceClassification",local:"transformers.FNetForSequenceClassification",headingTag:"h2"}}),yt=new $({props:{name:"class transformers.FNetForSequenceClassification",anchor:"transformers.FNetForSequenceClassification",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.FNetForSequenceClassification.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/fnet#transformers.FNetForSequenceClassification">FNetForSequenceClassification</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/fnet/modeling_fnet.py#L791"}}),Mt=new $({props:{name:"forward",anchor:"transformers.FNetForSequenceClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"labels",val:": typing.Optional[torch.Tensor] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.FNetForSequenceClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.FNetForSequenceClassification.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.FNetForSequenceClassification.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.FNetForSequenceClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.FNetForSequenceClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the sequence classification/regression loss. Indices should be in <code>[0, ..., config.num_labels - 1]</code>. If <code>config.num_labels == 1</code> a regression loss is computed (Mean-Square loss), If
<code>config.num_labels &gt; 1</code> a classification loss is computed (Cross-Entropy).`,name:"labels"},{anchor:"transformers.FNetForSequenceClassification.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.FNetForSequenceClassification.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/fnet/modeling_fnet.py#L803",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput"
>transformers.modeling_outputs.SequenceClassifierOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/fnet#transformers.FNetConfig"
>FNetConfig</a>) and inputs.</p>
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
`}}),xe=new Xe({props:{$$slots:{default:[Ir]},$$scope:{ctx:k}}}),ze=new ge({props:{anchor:"transformers.FNetForSequenceClassification.forward.example",$$slots:{default:[qr]},$$scope:{ctx:k}}}),je=new ge({props:{anchor:"transformers.FNetForSequenceClassification.forward.example-2",$$slots:{default:[Wr]},$$scope:{ctx:k}}}),wt=new B({props:{title:"FNetForMultipleChoice",local:"transformers.FNetForMultipleChoice",headingTag:"h2"}}),kt=new $({props:{name:"class transformers.FNetForMultipleChoice",anchor:"transformers.FNetForMultipleChoice",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.FNetForMultipleChoice.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/fnet#transformers.FNetForMultipleChoice">FNetForMultipleChoice</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/fnet/modeling_fnet.py#L865"}}),vt=new $({props:{name:"forward",anchor:"transformers.FNetForMultipleChoice.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"labels",val:": typing.Optional[torch.Tensor] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.FNetForMultipleChoice.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, num_choices, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.FNetForMultipleChoice.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, num_choices, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.FNetForMultipleChoice.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, num_choices, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.max_position_embeddings - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.FNetForMultipleChoice.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_choices, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <em>input_ids</em> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.FNetForMultipleChoice.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the multiple choice classification loss. Indices should be in <code>[0, ..., num_choices-1]</code> where <code>num_choices</code> is the size of the second dimension of the input tensors. (See
<code>input_ids</code> above)`,name:"labels"},{anchor:"transformers.FNetForMultipleChoice.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.FNetForMultipleChoice.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/fnet/modeling_fnet.py#L876",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.MultipleChoiceModelOutput"
>transformers.modeling_outputs.MultipleChoiceModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/fnet#transformers.FNetConfig"
>FNetConfig</a>) and inputs.</p>
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
`}}),Je=new Xe({props:{$$slots:{default:[Br]},$$scope:{ctx:k}}}),Ue=new ge({props:{anchor:"transformers.FNetForMultipleChoice.forward.example",$$slots:{default:[Rr]},$$scope:{ctx:k}}}),Ft=new B({props:{title:"FNetForTokenClassification",local:"transformers.FNetForTokenClassification",headingTag:"h2"}}),$t=new $({props:{name:"class transformers.FNetForTokenClassification",anchor:"transformers.FNetForTokenClassification",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.FNetForTokenClassification.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/fnet#transformers.FNetForTokenClassification">FNetForTokenClassification</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/fnet/modeling_fnet.py#L957"}}),Nt=new $({props:{name:"forward",anchor:"transformers.FNetForTokenClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"labels",val:": typing.Optional[torch.Tensor] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.FNetForTokenClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.FNetForTokenClassification.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.FNetForTokenClassification.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.FNetForTokenClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.FNetForTokenClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the token classification loss. Indices should be in <code>[0, ..., config.num_labels - 1]</code>.`,name:"labels"},{anchor:"transformers.FNetForTokenClassification.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.FNetForTokenClassification.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/fnet/modeling_fnet.py#L970",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput"
>transformers.modeling_outputs.TokenClassifierOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/fnet#transformers.FNetConfig"
>FNetConfig</a>) and inputs.</p>
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
`}}),Ce=new Xe({props:{$$slots:{default:[Xr]},$$scope:{ctx:k}}}),Ze=new ge({props:{anchor:"transformers.FNetForTokenClassification.forward.example",$$slots:{default:[Sr]},$$scope:{ctx:k}}}),xt=new B({props:{title:"FNetForQuestionAnswering",local:"transformers.FNetForQuestionAnswering",headingTag:"h2"}}),zt=new $({props:{name:"class transformers.FNetForQuestionAnswering",anchor:"transformers.FNetForQuestionAnswering",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.FNetForQuestionAnswering.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/fnet#transformers.FNetForQuestionAnswering">FNetForQuestionAnswering</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/fnet/modeling_fnet.py#L1015"}}),jt=new $({props:{name:"forward",anchor:"transformers.FNetForQuestionAnswering.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"start_positions",val:": typing.Optional[torch.Tensor] = None"},{name:"end_positions",val:": typing.Optional[torch.Tensor] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.FNetForQuestionAnswering.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.FNetForQuestionAnswering.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.FNetForQuestionAnswering.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.FNetForQuestionAnswering.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.FNetForQuestionAnswering.forward.start_positions",description:`<strong>start_positions</strong> (<code>torch.Tensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for position (index) of the start of the labelled span for computing the token classification loss.
Positions are clamped to the length of the sequence (<code>sequence_length</code>). Position outside of the sequence
are not taken into account for computing the loss.`,name:"start_positions"},{anchor:"transformers.FNetForQuestionAnswering.forward.end_positions",description:`<strong>end_positions</strong> (<code>torch.Tensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for position (index) of the end of the labelled span for computing the token classification loss.
Positions are clamped to the length of the sequence (<code>sequence_length</code>). Position outside of the sequence
are not taken into account for computing the loss.`,name:"end_positions"},{anchor:"transformers.FNetForQuestionAnswering.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.FNetForQuestionAnswering.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/fnet/modeling_fnet.py#L1027",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.QuestionAnsweringModelOutput"
>transformers.modeling_outputs.QuestionAnsweringModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/fnet#transformers.FNetConfig"
>FNetConfig</a>) and inputs.</p>
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
`}}),Ie=new Xe({props:{$$slots:{default:[Gr]},$$scope:{ctx:k}}}),qe=new ge({props:{anchor:"transformers.FNetForQuestionAnswering.forward.example",$$slots:{default:[Vr]},$$scope:{ctx:k}}}),Jt=new $r({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/fnet.md"}}),{c(){t=d("meta"),y=s(),l=d("p"),p=s(),M=d("p"),M.innerHTML=n,w=s(),u(Se.$$.fragment),$n=s(),be=d("div"),be.innerHTML=$s,Nn=s(),u(Ge.$$.fragment),xn=s(),Ve=d("p"),Ve.innerHTML=Ns,zn=s(),He=d("p"),He.innerHTML=xs,jn=s(),Pe=d("p"),Pe.innerHTML=zs,Jn=s(),u(Le.$$.fragment),Un=s(),Ee=d("p"),Ee.textContent=js,Cn=s(),u(Qe.$$.fragment),Zn=s(),Ye=d("ul"),Ye.innerHTML=Js,In=s(),u(Ae.$$.fragment),qn=s(),X=d("div"),u(De.$$.fragment),ao=s(),Ct=d("p"),Ct.innerHTML=Us,io=s(),Zt=d("p"),Zt.innerHTML=Cs,lo=s(),u(Te.$$.fragment),Wn=s(),u(Oe.$$.fragment),Bn=s(),N=d("div"),u(Ke.$$.fragment),co=s(),It=d("p"),It.innerHTML=Zs,po=s(),oe=d("div"),u(et.$$.fragment),mo=s(),qt=d("p"),qt.textContent=Is,ho=s(),Wt=d("ul"),Wt.innerHTML=qs,uo=s(),ye=d("div"),u(tt.$$.fragment),fo=s(),Bt=d("p"),Bt.innerHTML=Ws,go=s(),se=d("div"),u(nt.$$.fragment),_o=s(),Rt=d("p"),Rt.innerHTML=Bs,bo=s(),Xt=d("p"),Xt.textContent=Rs,To=s(),St=d("div"),u(ot.$$.fragment),Rn=s(),u(st.$$.fragment),Xn=s(),O=d("div"),u(rt.$$.fragment),yo=s(),Gt=d("p"),Gt.innerHTML=Xs,Mo=s(),re=d("div"),u(at.$$.fragment),wo=s(),Vt=d("p"),Vt.textContent=Ss,ko=s(),Ht=d("ul"),Ht.innerHTML=Gs,Sn=s(),u(it.$$.fragment),Gn=s(),z=d("div"),u(lt.$$.fragment),vo=s(),Pt=d("p"),Pt.textContent=Vs,Fo=s(),Lt=d("p"),Lt.innerHTML=Hs,$o=s(),Et=d("p"),Et.innerHTML=Ps,No=s(),ae=d("div"),u(dt.$$.fragment),xo=s(),Qt=d("p"),Qt.innerHTML=Ls,zo=s(),u(Me.$$.fragment),Vn=s(),u(ct.$$.fragment),Hn=s(),j=d("div"),u(pt.$$.fragment),jo=s(),Yt=d("p"),Yt.innerHTML=Es,Jo=s(),At=d("p"),At.innerHTML=Qs,Uo=s(),Dt=d("p"),Dt.innerHTML=Ys,Co=s(),L=d("div"),u(mt.$$.fragment),Zo=s(),Ot=d("p"),Ot.innerHTML=As,Io=s(),u(we.$$.fragment),qo=s(),u(ke.$$.fragment),Pn=s(),u(ht.$$.fragment),Ln=s(),J=d("div"),u(ut.$$.fragment),Wo=s(),Kt=d("p"),Kt.innerHTML=Ds,Bo=s(),en=d("p"),en.innerHTML=Os,Ro=s(),tn=d("p"),tn.innerHTML=Ks,Xo=s(),E=d("div"),u(ft.$$.fragment),So=s(),nn=d("p"),nn.innerHTML=er,Go=s(),u(ve.$$.fragment),Vo=s(),u(Fe.$$.fragment),En=s(),u(gt.$$.fragment),Qn=s(),U=d("div"),u(_t.$$.fragment),Ho=s(),on=d("p"),on.innerHTML=tr,Po=s(),sn=d("p"),sn.innerHTML=nr,Lo=s(),rn=d("p"),rn.innerHTML=or,Eo=s(),Q=d("div"),u(bt.$$.fragment),Qo=s(),an=d("p"),an.innerHTML=sr,Yo=s(),u($e.$$.fragment),Ao=s(),u(Ne.$$.fragment),Yn=s(),u(Tt.$$.fragment),An=s(),C=d("div"),u(yt.$$.fragment),Do=s(),ln=d("p"),ln.textContent=rr,Oo=s(),dn=d("p"),dn.innerHTML=ar,Ko=s(),cn=d("p"),cn.innerHTML=ir,es=s(),R=d("div"),u(Mt.$$.fragment),ts=s(),pn=d("p"),pn.innerHTML=lr,ns=s(),u(xe.$$.fragment),os=s(),u(ze.$$.fragment),ss=s(),u(je.$$.fragment),Dn=s(),u(wt.$$.fragment),On=s(),Z=d("div"),u(kt.$$.fragment),rs=s(),mn=d("p"),mn.textContent=dr,as=s(),hn=d("p"),hn.innerHTML=cr,is=s(),un=d("p"),un.innerHTML=pr,ls=s(),Y=d("div"),u(vt.$$.fragment),ds=s(),fn=d("p"),fn.innerHTML=mr,cs=s(),u(Je.$$.fragment),ps=s(),u(Ue.$$.fragment),Kn=s(),u(Ft.$$.fragment),eo=s(),I=d("div"),u($t.$$.fragment),ms=s(),gn=d("p"),gn.textContent=hr,hs=s(),_n=d("p"),_n.innerHTML=ur,us=s(),bn=d("p"),bn.innerHTML=fr,fs=s(),A=d("div"),u(Nt.$$.fragment),gs=s(),Tn=d("p"),Tn.innerHTML=gr,_s=s(),u(Ce.$$.fragment),bs=s(),u(Ze.$$.fragment),to=s(),u(xt.$$.fragment),no=s(),q=d("div"),u(zt.$$.fragment),Ts=s(),yn=d("p"),yn.innerHTML=_r,ys=s(),Mn=d("p"),Mn.innerHTML=br,Ms=s(),wn=d("p"),wn.innerHTML=Tr,ws=s(),D=d("div"),u(jt.$$.fragment),ks=s(),kn=d("p"),kn.innerHTML=yr,vs=s(),u(Ie.$$.fragment),Fs=s(),u(qe.$$.fragment),oo=s(),u(Jt.$$.fragment),so=s(),vn=d("p"),this.h()},l(e){const a=Fr("svelte-u9bgzb",document.head);t=c(a,"META",{name:!0,content:!0}),a.forEach(i),y=r(e),l=c(e,"P",{}),F(l).forEach(i),p=r(e),M=c(e,"P",{"data-svelte-h":!0}),h(M)!=="svelte-1l28wk3"&&(M.innerHTML=n),w=r(e),f(Se.$$.fragment,e),$n=r(e),be=c(e,"DIV",{class:!0,"data-svelte-h":!0}),h(be)!=="svelte-13t8s2t"&&(be.innerHTML=$s),Nn=r(e),f(Ge.$$.fragment,e),xn=r(e),Ve=c(e,"P",{"data-svelte-h":!0}),h(Ve)!=="svelte-fw1i7w"&&(Ve.innerHTML=Ns),zn=r(e),He=c(e,"P",{"data-svelte-h":!0}),h(He)!=="svelte-6esq7m"&&(He.innerHTML=xs),jn=r(e),Pe=c(e,"P",{"data-svelte-h":!0}),h(Pe)!=="svelte-li5hn8"&&(Pe.innerHTML=zs),Jn=r(e),f(Le.$$.fragment,e),Un=r(e),Ee=c(e,"P",{"data-svelte-h":!0}),h(Ee)!=="svelte-nw3sad"&&(Ee.textContent=js),Cn=r(e),f(Qe.$$.fragment,e),Zn=r(e),Ye=c(e,"UL",{"data-svelte-h":!0}),h(Ye)!=="svelte-mgusi3"&&(Ye.innerHTML=Js),In=r(e),f(Ae.$$.fragment,e),qn=r(e),X=c(e,"DIV",{class:!0});var K=F(X);f(De.$$.fragment,K),ao=r(K),Ct=c(K,"P",{"data-svelte-h":!0}),h(Ct)!=="svelte-1047tvo"&&(Ct.innerHTML=Us),io=r(K),Zt=c(K,"P",{"data-svelte-h":!0}),h(Zt)!=="svelte-1ek1ss9"&&(Zt.innerHTML=Cs),lo=r(K),f(Te.$$.fragment,K),K.forEach(i),Wn=r(e),f(Oe.$$.fragment,e),Bn=r(e),N=c(e,"DIV",{class:!0});var W=F(N);f(Ke.$$.fragment,W),co=r(W),It=c(W,"P",{"data-svelte-h":!0}),h(It)!=="svelte-167aia4"&&(It.innerHTML=Zs),po=r(W),oe=c(W,"DIV",{class:!0});var pe=F(oe);f(et.$$.fragment,pe),mo=r(pe),qt=c(pe,"P",{"data-svelte-h":!0}),h(qt)!=="svelte-lggsvw"&&(qt.textContent=Is),ho=r(pe),Wt=c(pe,"UL",{"data-svelte-h":!0}),h(Wt)!=="svelte-xi6653"&&(Wt.innerHTML=qs),pe.forEach(i),uo=r(W),ye=c(W,"DIV",{class:!0});var Ut=F(ye);f(tt.$$.fragment,Ut),fo=r(Ut),Bt=c(Ut,"P",{"data-svelte-h":!0}),h(Bt)!=="svelte-1f4f5kp"&&(Bt.innerHTML=Ws),Ut.forEach(i),go=r(W),se=c(W,"DIV",{class:!0});var me=F(se);f(nt.$$.fragment,me),_o=r(me),Rt=c(me,"P",{"data-svelte-h":!0}),h(Rt)!=="svelte-zj1vf1"&&(Rt.innerHTML=Bs),bo=r(me),Xt=c(me,"P",{"data-svelte-h":!0}),h(Xt)!=="svelte-9vptpw"&&(Xt.textContent=Rs),me.forEach(i),To=r(W),St=c(W,"DIV",{class:!0});var Fn=F(St);f(ot.$$.fragment,Fn),Fn.forEach(i),W.forEach(i),Rn=r(e),f(st.$$.fragment,e),Xn=r(e),O=c(e,"DIV",{class:!0});var he=F(O);f(rt.$$.fragment,he),yo=r(he),Gt=c(he,"P",{"data-svelte-h":!0}),h(Gt)!=="svelte-1vvtyp0"&&(Gt.innerHTML=Xs),Mo=r(he),re=c(he,"DIV",{class:!0});var ue=F(re);f(at.$$.fragment,ue),wo=r(ue),Vt=c(ue,"P",{"data-svelte-h":!0}),h(Vt)!=="svelte-lggsvw"&&(Vt.textContent=Ss),ko=r(ue),Ht=c(ue,"UL",{"data-svelte-h":!0}),h(Ht)!=="svelte-xi6653"&&(Ht.innerHTML=Gs),ue.forEach(i),he.forEach(i),Sn=r(e),f(it.$$.fragment,e),Gn=r(e),z=c(e,"DIV",{class:!0});var S=F(z);f(lt.$$.fragment,S),vo=r(S),Pt=c(S,"P",{"data-svelte-h":!0}),h(Pt)!=="svelte-5dd0m1"&&(Pt.textContent=Vs),Fo=r(S),Lt=c(S,"P",{"data-svelte-h":!0}),h(Lt)!=="svelte-q52n56"&&(Lt.innerHTML=Hs),$o=r(S),Et=c(S,"P",{"data-svelte-h":!0}),h(Et)!=="svelte-hswkmf"&&(Et.innerHTML=Ps),No=r(S),ae=c(S,"DIV",{class:!0});var fe=F(ae);f(dt.$$.fragment,fe),xo=r(fe),Qt=c(fe,"P",{"data-svelte-h":!0}),h(Qt)!=="svelte-i9lfba"&&(Qt.innerHTML=Ls),zo=r(fe),f(Me.$$.fragment,fe),fe.forEach(i),S.forEach(i),Vn=r(e),f(ct.$$.fragment,e),Hn=r(e),j=c(e,"DIV",{class:!0});var G=F(j);f(pt.$$.fragment,G),jo=r(G),Yt=c(G,"P",{"data-svelte-h":!0}),h(Yt)!=="svelte-wslq8d"&&(Yt.innerHTML=Es),Jo=r(G),At=c(G,"P",{"data-svelte-h":!0}),h(At)!=="svelte-q52n56"&&(At.innerHTML=Qs),Uo=r(G),Dt=c(G,"P",{"data-svelte-h":!0}),h(Dt)!=="svelte-hswkmf"&&(Dt.innerHTML=Ys),Co=r(G),L=c(G,"DIV",{class:!0});var ee=F(L);f(mt.$$.fragment,ee),Zo=r(ee),Ot=c(ee,"P",{"data-svelte-h":!0}),h(Ot)!=="svelte-1l59hlk"&&(Ot.innerHTML=As),Io=r(ee),f(we.$$.fragment,ee),qo=r(ee),f(ke.$$.fragment,ee),ee.forEach(i),G.forEach(i),Pn=r(e),f(ht.$$.fragment,e),Ln=r(e),J=c(e,"DIV",{class:!0});var V=F(J);f(ut.$$.fragment,V),Wo=r(V),Kt=c(V,"P",{"data-svelte-h":!0}),h(Kt)!=="svelte-10y5fp7"&&(Kt.innerHTML=Ds),Bo=r(V),en=c(V,"P",{"data-svelte-h":!0}),h(en)!=="svelte-q52n56"&&(en.innerHTML=Os),Ro=r(V),tn=c(V,"P",{"data-svelte-h":!0}),h(tn)!=="svelte-hswkmf"&&(tn.innerHTML=Ks),Xo=r(V),E=c(V,"DIV",{class:!0});var te=F(E);f(ft.$$.fragment,te),So=r(te),nn=c(te,"P",{"data-svelte-h":!0}),h(nn)!=="svelte-1vg6fee"&&(nn.innerHTML=er),Go=r(te),f(ve.$$.fragment,te),Vo=r(te),f(Fe.$$.fragment,te),te.forEach(i),V.forEach(i),En=r(e),f(gt.$$.fragment,e),Qn=r(e),U=c(e,"DIV",{class:!0});var H=F(U);f(_t.$$.fragment,H),Ho=r(H),on=c(H,"P",{"data-svelte-h":!0}),h(on)!=="svelte-1qlc2e6"&&(on.innerHTML=tr),Po=r(H),sn=c(H,"P",{"data-svelte-h":!0}),h(sn)!=="svelte-q52n56"&&(sn.innerHTML=nr),Lo=r(H),rn=c(H,"P",{"data-svelte-h":!0}),h(rn)!=="svelte-hswkmf"&&(rn.innerHTML=or),Eo=r(H),Q=c(H,"DIV",{class:!0});var ne=F(Q);f(bt.$$.fragment,ne),Qo=r(ne),an=c(ne,"P",{"data-svelte-h":!0}),h(an)!=="svelte-1vkn3rq"&&(an.innerHTML=sr),Yo=r(ne),f($e.$$.fragment,ne),Ao=r(ne),f(Ne.$$.fragment,ne),ne.forEach(i),H.forEach(i),Yn=r(e),f(Tt.$$.fragment,e),An=r(e),C=c(e,"DIV",{class:!0});var P=F(C);f(yt.$$.fragment,P),Do=r(P),ln=c(P,"P",{"data-svelte-h":!0}),h(ln)!=="svelte-ebnkul"&&(ln.textContent=rr),Oo=r(P),dn=c(P,"P",{"data-svelte-h":!0}),h(dn)!=="svelte-q52n56"&&(dn.innerHTML=ar),Ko=r(P),cn=c(P,"P",{"data-svelte-h":!0}),h(cn)!=="svelte-hswkmf"&&(cn.innerHTML=ir),es=r(P),R=c(P,"DIV",{class:!0});var ie=F(R);f(Mt.$$.fragment,ie),ts=r(ie),pn=c(ie,"P",{"data-svelte-h":!0}),h(pn)!=="svelte-9oelwa"&&(pn.innerHTML=lr),ns=r(ie),f(xe.$$.fragment,ie),os=r(ie),f(ze.$$.fragment,ie),ss=r(ie),f(je.$$.fragment,ie),ie.forEach(i),P.forEach(i),Dn=r(e),f(wt.$$.fragment,e),On=r(e),Z=c(e,"DIV",{class:!0});var le=F(Z);f(kt.$$.fragment,le),rs=r(le),mn=c(le,"P",{"data-svelte-h":!0}),h(mn)!=="svelte-16h2lzi"&&(mn.textContent=dr),as=r(le),hn=c(le,"P",{"data-svelte-h":!0}),h(hn)!=="svelte-q52n56"&&(hn.innerHTML=cr),is=r(le),un=c(le,"P",{"data-svelte-h":!0}),h(un)!=="svelte-hswkmf"&&(un.innerHTML=pr),ls=r(le),Y=c(le,"DIV",{class:!0});var We=F(Y);f(vt.$$.fragment,We),ds=r(We),fn=c(We,"P",{"data-svelte-h":!0}),h(fn)!=="svelte-c5vwnm"&&(fn.innerHTML=mr),cs=r(We),f(Je.$$.fragment,We),ps=r(We),f(Ue.$$.fragment,We),We.forEach(i),le.forEach(i),Kn=r(e),f(Ft.$$.fragment,e),eo=r(e),I=c(e,"DIV",{class:!0});var de=F(I);f($t.$$.fragment,de),ms=r(de),gn=c(de,"P",{"data-svelte-h":!0}),h(gn)!=="svelte-16c7q6n"&&(gn.textContent=hr),hs=r(de),_n=c(de,"P",{"data-svelte-h":!0}),h(_n)!=="svelte-q52n56"&&(_n.innerHTML=ur),us=r(de),bn=c(de,"P",{"data-svelte-h":!0}),h(bn)!=="svelte-hswkmf"&&(bn.innerHTML=fr),fs=r(de),A=c(de,"DIV",{class:!0});var Be=F(A);f(Nt.$$.fragment,Be),gs=r(Be),Tn=c(Be,"P",{"data-svelte-h":!0}),h(Tn)!=="svelte-h7abf4"&&(Tn.innerHTML=gr),_s=r(Be),f(Ce.$$.fragment,Be),bs=r(Be),f(Ze.$$.fragment,Be),Be.forEach(i),de.forEach(i),to=r(e),f(xt.$$.fragment,e),no=r(e),q=c(e,"DIV",{class:!0});var ce=F(q);f(zt.$$.fragment,ce),Ts=r(ce),yn=c(ce,"P",{"data-svelte-h":!0}),h(yn)!=="svelte-1dgn3je"&&(yn.innerHTML=_r),ys=r(ce),Mn=c(ce,"P",{"data-svelte-h":!0}),h(Mn)!=="svelte-q52n56"&&(Mn.innerHTML=br),Ms=r(ce),wn=c(ce,"P",{"data-svelte-h":!0}),h(wn)!=="svelte-hswkmf"&&(wn.innerHTML=Tr),ws=r(ce),D=c(ce,"DIV",{class:!0});var Re=F(D);f(jt.$$.fragment,Re),ks=r(Re),kn=c(Re,"P",{"data-svelte-h":!0}),h(kn)!=="svelte-bwzytu"&&(kn.innerHTML=yr),vs=r(Re),f(Ie.$$.fragment,Re),Fs=r(Re),f(qe.$$.fragment,Re),Re.forEach(i),ce.forEach(i),oo=r(e),f(Jt.$$.fragment,e),so=r(e),vn=c(e,"P",{}),F(vn).forEach(i),this.h()},h(){v(t,"name","hf:doc:metadata"),v(t,"content",Pr),v(be,"class","flex flex-wrap space-x-1"),v(X,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(oe,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(ye,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(se,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(St,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(N,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(re,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(O,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(ae,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(z,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(L,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(j,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(E,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(J,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(Q,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(U,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(R,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(C,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(Y,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(Z,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(A,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(I,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(D,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(q,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,a){o(document.head,t),m(e,y,a),m(e,l,a),m(e,p,a),m(e,M,a),m(e,w,a),g(Se,e,a),m(e,$n,a),m(e,be,a),m(e,Nn,a),g(Ge,e,a),m(e,xn,a),m(e,Ve,a),m(e,zn,a),m(e,He,a),m(e,jn,a),m(e,Pe,a),m(e,Jn,a),g(Le,e,a),m(e,Un,a),m(e,Ee,a),m(e,Cn,a),g(Qe,e,a),m(e,Zn,a),m(e,Ye,a),m(e,In,a),g(Ae,e,a),m(e,qn,a),m(e,X,a),g(De,X,null),o(X,ao),o(X,Ct),o(X,io),o(X,Zt),o(X,lo),g(Te,X,null),m(e,Wn,a),g(Oe,e,a),m(e,Bn,a),m(e,N,a),g(Ke,N,null),o(N,co),o(N,It),o(N,po),o(N,oe),g(et,oe,null),o(oe,mo),o(oe,qt),o(oe,ho),o(oe,Wt),o(N,uo),o(N,ye),g(tt,ye,null),o(ye,fo),o(ye,Bt),o(N,go),o(N,se),g(nt,se,null),o(se,_o),o(se,Rt),o(se,bo),o(se,Xt),o(N,To),o(N,St),g(ot,St,null),m(e,Rn,a),g(st,e,a),m(e,Xn,a),m(e,O,a),g(rt,O,null),o(O,yo),o(O,Gt),o(O,Mo),o(O,re),g(at,re,null),o(re,wo),o(re,Vt),o(re,ko),o(re,Ht),m(e,Sn,a),g(it,e,a),m(e,Gn,a),m(e,z,a),g(lt,z,null),o(z,vo),o(z,Pt),o(z,Fo),o(z,Lt),o(z,$o),o(z,Et),o(z,No),o(z,ae),g(dt,ae,null),o(ae,xo),o(ae,Qt),o(ae,zo),g(Me,ae,null),m(e,Vn,a),g(ct,e,a),m(e,Hn,a),m(e,j,a),g(pt,j,null),o(j,jo),o(j,Yt),o(j,Jo),o(j,At),o(j,Uo),o(j,Dt),o(j,Co),o(j,L),g(mt,L,null),o(L,Zo),o(L,Ot),o(L,Io),g(we,L,null),o(L,qo),g(ke,L,null),m(e,Pn,a),g(ht,e,a),m(e,Ln,a),m(e,J,a),g(ut,J,null),o(J,Wo),o(J,Kt),o(J,Bo),o(J,en),o(J,Ro),o(J,tn),o(J,Xo),o(J,E),g(ft,E,null),o(E,So),o(E,nn),o(E,Go),g(ve,E,null),o(E,Vo),g(Fe,E,null),m(e,En,a),g(gt,e,a),m(e,Qn,a),m(e,U,a),g(_t,U,null),o(U,Ho),o(U,on),o(U,Po),o(U,sn),o(U,Lo),o(U,rn),o(U,Eo),o(U,Q),g(bt,Q,null),o(Q,Qo),o(Q,an),o(Q,Yo),g($e,Q,null),o(Q,Ao),g(Ne,Q,null),m(e,Yn,a),g(Tt,e,a),m(e,An,a),m(e,C,a),g(yt,C,null),o(C,Do),o(C,ln),o(C,Oo),o(C,dn),o(C,Ko),o(C,cn),o(C,es),o(C,R),g(Mt,R,null),o(R,ts),o(R,pn),o(R,ns),g(xe,R,null),o(R,os),g(ze,R,null),o(R,ss),g(je,R,null),m(e,Dn,a),g(wt,e,a),m(e,On,a),m(e,Z,a),g(kt,Z,null),o(Z,rs),o(Z,mn),o(Z,as),o(Z,hn),o(Z,is),o(Z,un),o(Z,ls),o(Z,Y),g(vt,Y,null),o(Y,ds),o(Y,fn),o(Y,cs),g(Je,Y,null),o(Y,ps),g(Ue,Y,null),m(e,Kn,a),g(Ft,e,a),m(e,eo,a),m(e,I,a),g($t,I,null),o(I,ms),o(I,gn),o(I,hs),o(I,_n),o(I,us),o(I,bn),o(I,fs),o(I,A),g(Nt,A,null),o(A,gs),o(A,Tn),o(A,_s),g(Ce,A,null),o(A,bs),g(Ze,A,null),m(e,to,a),g(xt,e,a),m(e,no,a),m(e,q,a),g(zt,q,null),o(q,Ts),o(q,yn),o(q,ys),o(q,Mn),o(q,Ms),o(q,wn),o(q,ws),o(q,D),g(jt,D,null),o(D,ks),o(D,kn),o(D,vs),g(Ie,D,null),o(D,Fs),g(qe,D,null),m(e,oo,a),g(Jt,e,a),m(e,so,a),m(e,vn,a),ro=!0},p(e,[a]){const K={};a&2&&(K.$$scope={dirty:a,ctx:e}),Te.$set(K);const W={};a&2&&(W.$$scope={dirty:a,ctx:e}),Me.$set(W);const pe={};a&2&&(pe.$$scope={dirty:a,ctx:e}),we.$set(pe);const Ut={};a&2&&(Ut.$$scope={dirty:a,ctx:e}),ke.$set(Ut);const me={};a&2&&(me.$$scope={dirty:a,ctx:e}),ve.$set(me);const Fn={};a&2&&(Fn.$$scope={dirty:a,ctx:e}),Fe.$set(Fn);const he={};a&2&&(he.$$scope={dirty:a,ctx:e}),$e.$set(he);const ue={};a&2&&(ue.$$scope={dirty:a,ctx:e}),Ne.$set(ue);const S={};a&2&&(S.$$scope={dirty:a,ctx:e}),xe.$set(S);const fe={};a&2&&(fe.$$scope={dirty:a,ctx:e}),ze.$set(fe);const G={};a&2&&(G.$$scope={dirty:a,ctx:e}),je.$set(G);const ee={};a&2&&(ee.$$scope={dirty:a,ctx:e}),Je.$set(ee);const V={};a&2&&(V.$$scope={dirty:a,ctx:e}),Ue.$set(V);const te={};a&2&&(te.$$scope={dirty:a,ctx:e}),Ce.$set(te);const H={};a&2&&(H.$$scope={dirty:a,ctx:e}),Ze.$set(H);const ne={};a&2&&(ne.$$scope={dirty:a,ctx:e}),Ie.$set(ne);const P={};a&2&&(P.$$scope={dirty:a,ctx:e}),qe.$set(P)},i(e){ro||(_(Se.$$.fragment,e),_(Ge.$$.fragment,e),_(Le.$$.fragment,e),_(Qe.$$.fragment,e),_(Ae.$$.fragment,e),_(De.$$.fragment,e),_(Te.$$.fragment,e),_(Oe.$$.fragment,e),_(Ke.$$.fragment,e),_(et.$$.fragment,e),_(tt.$$.fragment,e),_(nt.$$.fragment,e),_(ot.$$.fragment,e),_(st.$$.fragment,e),_(rt.$$.fragment,e),_(at.$$.fragment,e),_(it.$$.fragment,e),_(lt.$$.fragment,e),_(dt.$$.fragment,e),_(Me.$$.fragment,e),_(ct.$$.fragment,e),_(pt.$$.fragment,e),_(mt.$$.fragment,e),_(we.$$.fragment,e),_(ke.$$.fragment,e),_(ht.$$.fragment,e),_(ut.$$.fragment,e),_(ft.$$.fragment,e),_(ve.$$.fragment,e),_(Fe.$$.fragment,e),_(gt.$$.fragment,e),_(_t.$$.fragment,e),_(bt.$$.fragment,e),_($e.$$.fragment,e),_(Ne.$$.fragment,e),_(Tt.$$.fragment,e),_(yt.$$.fragment,e),_(Mt.$$.fragment,e),_(xe.$$.fragment,e),_(ze.$$.fragment,e),_(je.$$.fragment,e),_(wt.$$.fragment,e),_(kt.$$.fragment,e),_(vt.$$.fragment,e),_(Je.$$.fragment,e),_(Ue.$$.fragment,e),_(Ft.$$.fragment,e),_($t.$$.fragment,e),_(Nt.$$.fragment,e),_(Ce.$$.fragment,e),_(Ze.$$.fragment,e),_(xt.$$.fragment,e),_(zt.$$.fragment,e),_(jt.$$.fragment,e),_(Ie.$$.fragment,e),_(qe.$$.fragment,e),_(Jt.$$.fragment,e),ro=!0)},o(e){b(Se.$$.fragment,e),b(Ge.$$.fragment,e),b(Le.$$.fragment,e),b(Qe.$$.fragment,e),b(Ae.$$.fragment,e),b(De.$$.fragment,e),b(Te.$$.fragment,e),b(Oe.$$.fragment,e),b(Ke.$$.fragment,e),b(et.$$.fragment,e),b(tt.$$.fragment,e),b(nt.$$.fragment,e),b(ot.$$.fragment,e),b(st.$$.fragment,e),b(rt.$$.fragment,e),b(at.$$.fragment,e),b(it.$$.fragment,e),b(lt.$$.fragment,e),b(dt.$$.fragment,e),b(Me.$$.fragment,e),b(ct.$$.fragment,e),b(pt.$$.fragment,e),b(mt.$$.fragment,e),b(we.$$.fragment,e),b(ke.$$.fragment,e),b(ht.$$.fragment,e),b(ut.$$.fragment,e),b(ft.$$.fragment,e),b(ve.$$.fragment,e),b(Fe.$$.fragment,e),b(gt.$$.fragment,e),b(_t.$$.fragment,e),b(bt.$$.fragment,e),b($e.$$.fragment,e),b(Ne.$$.fragment,e),b(Tt.$$.fragment,e),b(yt.$$.fragment,e),b(Mt.$$.fragment,e),b(xe.$$.fragment,e),b(ze.$$.fragment,e),b(je.$$.fragment,e),b(wt.$$.fragment,e),b(kt.$$.fragment,e),b(vt.$$.fragment,e),b(Je.$$.fragment,e),b(Ue.$$.fragment,e),b(Ft.$$.fragment,e),b($t.$$.fragment,e),b(Nt.$$.fragment,e),b(Ce.$$.fragment,e),b(Ze.$$.fragment,e),b(xt.$$.fragment,e),b(zt.$$.fragment,e),b(jt.$$.fragment,e),b(Ie.$$.fragment,e),b(qe.$$.fragment,e),b(Jt.$$.fragment,e),ro=!1},d(e){e&&(i(y),i(l),i(p),i(M),i(w),i($n),i(be),i(Nn),i(xn),i(Ve),i(zn),i(He),i(jn),i(Pe),i(Jn),i(Un),i(Ee),i(Cn),i(Zn),i(Ye),i(In),i(qn),i(X),i(Wn),i(Bn),i(N),i(Rn),i(Xn),i(O),i(Sn),i(Gn),i(z),i(Vn),i(Hn),i(j),i(Pn),i(Ln),i(J),i(En),i(Qn),i(U),i(Yn),i(An),i(C),i(Dn),i(On),i(Z),i(Kn),i(eo),i(I),i(to),i(no),i(q),i(oo),i(so),i(vn)),i(t),T(Se,e),T(Ge,e),T(Le,e),T(Qe,e),T(Ae,e),T(De),T(Te),T(Oe,e),T(Ke),T(et),T(tt),T(nt),T(ot),T(st,e),T(rt),T(at),T(it,e),T(lt),T(dt),T(Me),T(ct,e),T(pt),T(mt),T(we),T(ke),T(ht,e),T(ut),T(ft),T(ve),T(Fe),T(gt,e),T(_t),T(bt),T($e),T(Ne),T(Tt,e),T(yt),T(Mt),T(xe),T(ze),T(je),T(wt,e),T(kt),T(vt),T(Je),T(Ue),T(Ft,e),T($t),T(Nt),T(Ce),T(Ze),T(xt,e),T(zt),T(jt),T(Ie),T(qe),T(Jt,e)}}}const Pr='{"title":"FNet","local":"fnet","sections":[{"title":"Overview","local":"overview","sections":[],"depth":2},{"title":"Usage tips","local":"usage-tips","sections":[],"depth":2},{"title":"Resources","local":"resources","sections":[],"depth":2},{"title":"FNetConfig","local":"transformers.FNetConfig","sections":[],"depth":2},{"title":"FNetTokenizer","local":"transformers.FNetTokenizer","sections":[],"depth":2},{"title":"FNetTokenizerFast","local":"transformers.FNetTokenizerFast","sections":[],"depth":2},{"title":"FNetModel","local":"transformers.FNetModel","sections":[],"depth":2},{"title":"FNetForPreTraining","local":"transformers.FNetForPreTraining","sections":[],"depth":2},{"title":"FNetForMaskedLM","local":"transformers.FNetForMaskedLM","sections":[],"depth":2},{"title":"FNetForNextSentencePrediction","local":"transformers.FNetForNextSentencePrediction","sections":[],"depth":2},{"title":"FNetForSequenceClassification","local":"transformers.FNetForSequenceClassification","sections":[],"depth":2},{"title":"FNetForMultipleChoice","local":"transformers.FNetForMultipleChoice","sections":[],"depth":2},{"title":"FNetForTokenClassification","local":"transformers.FNetForTokenClassification","sections":[],"depth":2},{"title":"FNetForQuestionAnswering","local":"transformers.FNetForQuestionAnswering","sections":[],"depth":2}],"depth":1}';function Lr(k){return wr(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class ea extends kr{constructor(t){super(),vr(this,t,Lr,Hr,Mr,{})}}export{ea as component};
