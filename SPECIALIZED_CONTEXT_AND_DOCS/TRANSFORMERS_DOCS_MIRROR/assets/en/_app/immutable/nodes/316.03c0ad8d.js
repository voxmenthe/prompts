import{s as $s,o as Ns,n as N}from"../chunks/scheduler.18a86fab.js";import{S as js,i as Js,g as c,s as r,r as f,A as Us,h as p,f as d,c as i,j as $,x as m,u as g,k,y as s,a as h,v as _,d as b,t as M,w as T}from"../chunks/index.98837b22.js";import{T as pe}from"../chunks/Tip.77304350.js";import{D as J}from"../chunks/Docstring.a1ef7999.js";import{C as ce}from"../chunks/CodeBlock.8d0c2e8a.js";import{E as de}from"../chunks/ExampleCodeBlock.8c3ee1f9.js";import{H as ee,E as Cs}from"../chunks/getInferenceSnippets.06c2775f.js";function xs(z){let t,u=`This model is in maintenance mode only, we don’t accept any new PRs changing its code.
If you run into any issues running this model, please reinstall the last version that supported this model: v4.40.2.
You can do so by running the following command: <code>pip install -U transformers==4.40.2</code>.`;return{c(){t=c("p"),t.innerHTML=u},l(o){t=p(o,"P",{"data-svelte-h":!0}),m(t)!=="svelte-1sq0hrb"&&(t.innerHTML=u)},m(o,l){h(o,t,l)},p:N,d(o){o&&d(t)}}}function Fs(z){let t,u="Example:",o,l,y;return l=new ce({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyME5lemhhQ29uZmlnJTJDJTIwTmV6aGFNb2RlbCUwQSUwQSUyMyUyMEluaXRpYWxpemluZyUyMGFuJTIwTmV6aGElMjBjb25maWd1cmF0aW9uJTBBY29uZmlndXJhdGlvbiUyMCUzRCUyME5lemhhQ29uZmlnKCklMEElMEElMjMlMjBJbml0aWFsaXppbmclMjBhJTIwbW9kZWwlMjAod2l0aCUyMHJhbmRvbSUyMHdlaWdodHMpJTIwZnJvbSUyMHRoZSUyME5lemhhLWJhc2UlMjBzdHlsZSUyMGNvbmZpZ3VyYXRpb24lMjBtb2RlbCUwQW1vZGVsJTIwJTNEJTIwTmV6aGFNb2RlbChjb25maWd1cmF0aW9uKSUwQSUwQSUyMyUyMEFjY2Vzc2luZyUyMHRoZSUyMG1vZGVsJTIwY29uZmlndXJhdGlvbiUwQWNvbmZpZ3VyYXRpb24lMjAlM0QlMjBtb2RlbC5jb25maWc=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> NezhaConfig, NezhaModel

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing an Nezha configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = NezhaConfig()

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a model (with random weights) from the Nezha-base style configuration model</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = NezhaModel(configuration)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Accessing the model configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = model.config`,wrap:!1}}),{c(){t=c("p"),t.textContent=u,o=r(),f(l.$$.fragment)},l(n){t=p(n,"P",{"data-svelte-h":!0}),m(t)!=="svelte-11lpom8"&&(t.textContent=u),o=i(n),g(l.$$.fragment,n)},m(n,w){h(n,t,w),h(n,o,w),_(l,n,w),y=!0},p:N,i(n){y||(b(l.$$.fragment,n),y=!0)},o(n){M(l.$$.fragment,n),y=!1},d(n){n&&(d(t),d(o)),T(l,n)}}}function Zs(z){let t,u=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=u},l(o){t=p(o,"P",{"data-svelte-h":!0}),m(t)!=="svelte-fincs2"&&(t.innerHTML=u)},m(o,l){h(o,t,l)},p:N,d(o){o&&d(t)}}}function Ws(z){let t,u="Example:",o,l,y;return l=new ce({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBOZXpoYU1vZGVsJTBBaW1wb3J0JTIwdG9yY2glMEElMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJzaWp1bmhlJTJGbmV6aGEtY24tYmFzZSUyMiklMEFtb2RlbCUyMCUzRCUyME5lemhhTW9kZWwuZnJvbV9wcmV0cmFpbmVkKCUyMnNpanVuaGUlMkZuZXpoYS1jbi1iYXNlJTIyKSUwQSUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplciglMjJIZWxsbyUyQyUyMG15JTIwZG9nJTIwaXMlMjBjdXRlJTIyJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiklMEFvdXRwdXRzJTIwJTNEJTIwbW9kZWwoKippbnB1dHMpJTBBJTBBbGFzdF9oaWRkZW5fc3RhdGVzJTIwJTNEJTIwb3V0cHV0cy5sYXN0X2hpZGRlbl9zdGF0ZQ==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, NezhaModel
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;sijunhe/nezha-cn-base&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = NezhaModel.from_pretrained(<span class="hljs-string">&quot;sijunhe/nezha-cn-base&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs)

<span class="hljs-meta">&gt;&gt;&gt; </span>last_hidden_states = outputs.last_hidden_state`,wrap:!1}}),{c(){t=c("p"),t.textContent=u,o=r(),f(l.$$.fragment)},l(n){t=p(n,"P",{"data-svelte-h":!0}),m(t)!=="svelte-11lpom8"&&(t.textContent=u),o=i(n),g(l.$$.fragment,n)},m(n,w){h(n,t,w),h(n,o,w),_(l,n,w),y=!0},p:N,i(n){y||(b(l.$$.fragment,n),y=!0)},o(n){M(l.$$.fragment,n),y=!1},d(n){n&&(d(t),d(o)),T(l,n)}}}function Is(z){let t,u=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=u},l(o){t=p(o,"P",{"data-svelte-h":!0}),m(t)!=="svelte-fincs2"&&(t.innerHTML=u)},m(o,l){h(o,t,l)},p:N,d(o){o&&d(t)}}}function Gs(z){let t,u="Example:",o,l,y;return l=new ce({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBOZXpoYUZvclByZVRyYWluaW5nJTBBaW1wb3J0JTIwdG9yY2glMEElMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJzaWp1bmhlJTJGbmV6aGEtY24tYmFzZSUyMiklMEFtb2RlbCUyMCUzRCUyME5lemhhRm9yUHJlVHJhaW5pbmcuZnJvbV9wcmV0cmFpbmVkKCUyMnNpanVuaGUlMkZuZXpoYS1jbi1iYXNlJTIyKSUwQSUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplciglMjJIZWxsbyUyQyUyMG15JTIwZG9nJTIwaXMlMjBjdXRlJTIyJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiklMEFvdXRwdXRzJTIwJTNEJTIwbW9kZWwoKippbnB1dHMpJTBBJTBBcHJlZGljdGlvbl9sb2dpdHMlMjAlM0QlMjBvdXRwdXRzLnByZWRpY3Rpb25fbG9naXRzJTBBc2VxX3JlbGF0aW9uc2hpcF9sb2dpdHMlMjAlM0QlMjBvdXRwdXRzLnNlcV9yZWxhdGlvbnNoaXBfbG9naXRz",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, NezhaForPreTraining
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;sijunhe/nezha-cn-base&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = NezhaForPreTraining.from_pretrained(<span class="hljs-string">&quot;sijunhe/nezha-cn-base&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs)

<span class="hljs-meta">&gt;&gt;&gt; </span>prediction_logits = outputs.prediction_logits
<span class="hljs-meta">&gt;&gt;&gt; </span>seq_relationship_logits = outputs.seq_relationship_logits`,wrap:!1}}),{c(){t=c("p"),t.textContent=u,o=r(),f(l.$$.fragment)},l(n){t=p(n,"P",{"data-svelte-h":!0}),m(t)!=="svelte-11lpom8"&&(t.textContent=u),o=i(n),g(l.$$.fragment,n)},m(n,w){h(n,t,w),h(n,o,w),_(l,n,w),y=!0},p:N,i(n){y||(b(l.$$.fragment,n),y=!0)},o(n){M(l.$$.fragment,n),y=!1},d(n){n&&(d(t),d(o)),T(l,n)}}}function Hs(z){let t,u=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=u},l(o){t=p(o,"P",{"data-svelte-h":!0}),m(t)!=="svelte-fincs2"&&(t.innerHTML=u)},m(o,l){h(o,t,l)},p:N,d(o){o&&d(t)}}}function Vs(z){let t,u="Example:",o,l,y;return l=new ce({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBOZXpoYUZvck1hc2tlZExNJTBBaW1wb3J0JTIwdG9yY2glMEElMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJzaWp1bmhlJTJGbmV6aGEtY24tYmFzZSUyMiklMEFtb2RlbCUyMCUzRCUyME5lemhhRm9yTWFza2VkTE0uZnJvbV9wcmV0cmFpbmVkKCUyMnNpanVuaGUlMkZuZXpoYS1jbi1iYXNlJTIyKSUwQSUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplciglMjJUaGUlMjBjYXBpdGFsJTIwb2YlMjBGcmFuY2UlMjBpcyUyMCU1Qk1BU0slNUQuJTIyJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiklMEElMEF3aXRoJTIwdG9yY2gubm9fZ3JhZCgpJTNBJTBBJTIwJTIwJTIwJTIwbG9naXRzJTIwJTNEJTIwbW9kZWwoKippbnB1dHMpLmxvZ2l0cyUwQSUwQSUyMyUyMHJldHJpZXZlJTIwaW5kZXglMjBvZiUyMCU1Qk1BU0slNUQlMEFtYXNrX3Rva2VuX2luZGV4JTIwJTNEJTIwKGlucHV0cy5pbnB1dF9pZHMlMjAlM0QlM0QlMjB0b2tlbml6ZXIubWFza190b2tlbl9pZCklNUIwJTVELm5vbnplcm8oYXNfdHVwbGUlM0RUcnVlKSU1QjAlNUQlMEElMEFwcmVkaWN0ZWRfdG9rZW5faWQlMjAlM0QlMjBsb2dpdHMlNUIwJTJDJTIwbWFza190b2tlbl9pbmRleCU1RC5hcmdtYXgoYXhpcyUzRC0xKSUwQSUwQWxhYmVscyUyMCUzRCUyMHRva2VuaXplciglMjJUaGUlMjBjYXBpdGFsJTIwb2YlMjBGcmFuY2UlMjBpcyUyMFBhcmlzLiUyMiUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpJTVCJTIyaW5wdXRfaWRzJTIyJTVEJTBBJTIzJTIwbWFzayUyMGxhYmVscyUyMG9mJTIwbm9uLSU1Qk1BU0slNUQlMjB0b2tlbnMlMEFsYWJlbHMlMjAlM0QlMjB0b3JjaC53aGVyZShpbnB1dHMuaW5wdXRfaWRzJTIwJTNEJTNEJTIwdG9rZW5pemVyLm1hc2tfdG9rZW5faWQlMkMlMjBsYWJlbHMlMkMlMjAtMTAwKSUwQSUwQW91dHB1dHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyUyQyUyMGxhYmVscyUzRGxhYmVscyk=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, NezhaForMaskedLM
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;sijunhe/nezha-cn-base&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = NezhaForMaskedLM.from_pretrained(<span class="hljs-string">&quot;sijunhe/nezha-cn-base&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;The capital of France is [MASK].&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    logits = model(**inputs).logits

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># retrieve index of [MASK]</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[<span class="hljs-number">0</span>].nonzero(as_tuple=<span class="hljs-literal">True</span>)[<span class="hljs-number">0</span>]

<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_token_id = logits[<span class="hljs-number">0</span>, mask_token_index].argmax(axis=-<span class="hljs-number">1</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>labels = tokenizer(<span class="hljs-string">&quot;The capital of France is Paris.&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)[<span class="hljs-string">&quot;input_ids&quot;</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># mask labels of non-[MASK] tokens</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.where(inputs.input_ids == tokenizer.mask_token_id, labels, -<span class="hljs-number">100</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs, labels=labels)`,wrap:!1}}),{c(){t=c("p"),t.textContent=u,o=r(),f(l.$$.fragment)},l(n){t=p(n,"P",{"data-svelte-h":!0}),m(t)!=="svelte-11lpom8"&&(t.textContent=u),o=i(n),g(l.$$.fragment,n)},m(n,w){h(n,t,w),h(n,o,w),_(l,n,w),y=!0},p:N,i(n){y||(b(l.$$.fragment,n),y=!0)},o(n){M(l.$$.fragment,n),y=!1},d(n){n&&(d(t),d(o)),T(l,n)}}}function Bs(z){let t,u=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=u},l(o){t=p(o,"P",{"data-svelte-h":!0}),m(t)!=="svelte-fincs2"&&(t.innerHTML=u)},m(o,l){h(o,t,l)},p:N,d(o){o&&d(t)}}}function qs(z){let t,u="Example:",o,l,y;return l=new ce({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBOZXpoYUZvck5leHRTZW50ZW5jZVByZWRpY3Rpb24lMEFpbXBvcnQlMjB0b3JjaCUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMnNpanVuaGUlMkZuZXpoYS1jbi1iYXNlJTIyKSUwQW1vZGVsJTIwJTNEJTIwTmV6aGFGb3JOZXh0U2VudGVuY2VQcmVkaWN0aW9uLmZyb21fcHJldHJhaW5lZCglMjJzaWp1bmhlJTJGbmV6aGEtY24tYmFzZSUyMiklMEElMEFwcm9tcHQlMjAlM0QlMjAlMjJJbiUyMEl0YWx5JTJDJTIwcGl6emElMjBzZXJ2ZWQlMjBpbiUyMGZvcm1hbCUyMHNldHRpbmdzJTJDJTIwc3VjaCUyMGFzJTIwYXQlMjBhJTIwcmVzdGF1cmFudCUyQyUyMGlzJTIwcHJlc2VudGVkJTIwdW5zbGljZWQuJTIyJTBBbmV4dF9zZW50ZW5jZSUyMCUzRCUyMCUyMlRoZSUyMHNreSUyMGlzJTIwYmx1ZSUyMGR1ZSUyMHRvJTIwdGhlJTIwc2hvcnRlciUyMHdhdmVsZW5ndGglMjBvZiUyMGJsdWUlMjBsaWdodC4lMjIlMEFlbmNvZGluZyUyMCUzRCUyMHRva2VuaXplcihwcm9tcHQlMkMlMjBuZXh0X3NlbnRlbmNlJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiklMEElMEFvdXRwdXRzJTIwJTNEJTIwbW9kZWwoKiplbmNvZGluZyUyQyUyMGxhYmVscyUzRHRvcmNoLkxvbmdUZW5zb3IoJTVCMSU1RCkpJTBBbG9naXRzJTIwJTNEJTIwb3V0cHV0cy5sb2dpdHMlMEFhc3NlcnQlMjBsb2dpdHMlNUIwJTJDJTIwMCU1RCUyMCUzQyUyMGxvZ2l0cyU1QjAlMkMlMjAxJTVEJTIwJTIwJTIzJTIwbmV4dCUyMHNlbnRlbmNlJTIwd2FzJTIwcmFuZG9t",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, NezhaForNextSentencePrediction
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;sijunhe/nezha-cn-base&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = NezhaForNextSentencePrediction.from_pretrained(<span class="hljs-string">&quot;sijunhe/nezha-cn-base&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>prompt = <span class="hljs-string">&quot;In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>next_sentence = <span class="hljs-string">&quot;The sky is blue due to the shorter wavelength of blue light.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>encoding = tokenizer(prompt, next_sentence, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**encoding, labels=torch.LongTensor([<span class="hljs-number">1</span>]))
<span class="hljs-meta">&gt;&gt;&gt; </span>logits = outputs.logits
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">assert</span> logits[<span class="hljs-number">0</span>, <span class="hljs-number">0</span>] &lt; logits[<span class="hljs-number">0</span>, <span class="hljs-number">1</span>]  <span class="hljs-comment"># next sentence was random</span>`,wrap:!1}}),{c(){t=c("p"),t.textContent=u,o=r(),f(l.$$.fragment)},l(n){t=p(n,"P",{"data-svelte-h":!0}),m(t)!=="svelte-11lpom8"&&(t.textContent=u),o=i(n),g(l.$$.fragment,n)},m(n,w){h(n,t,w),h(n,o,w),_(l,n,w),y=!0},p:N,i(n){y||(b(l.$$.fragment,n),y=!0)},o(n){M(l.$$.fragment,n),y=!1},d(n){n&&(d(t),d(o)),T(l,n)}}}function Ss(z){let t,u=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=u},l(o){t=p(o,"P",{"data-svelte-h":!0}),m(t)!=="svelte-fincs2"&&(t.innerHTML=u)},m(o,l){h(o,t,l)},p:N,d(o){o&&d(t)}}}function Xs(z){let t,u="Example of single-label classification:",o,l,y;return l=new ce({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Rva2VuaXplciUyQyUyME5lemhhRm9yU2VxdWVuY2VDbGFzc2lmaWNhdGlvbiUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMnNpanVuaGUlMkZuZXpoYS1jbi1iYXNlJTIyKSUwQW1vZGVsJTIwJTNEJTIwTmV6aGFGb3JTZXF1ZW5jZUNsYXNzaWZpY2F0aW9uLmZyb21fcHJldHJhaW5lZCglMjJzaWp1bmhlJTJGbmV6aGEtY24tYmFzZSUyMiklMEElMEFpbnB1dHMlMjAlM0QlMjB0b2tlbml6ZXIoJTIySGVsbG8lMkMlMjBteSUyMGRvZyUyMGlzJTIwY3V0ZSUyMiUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpJTBBJTBBd2l0aCUyMHRvcmNoLm5vX2dyYWQoKSUzQSUwQSUyMCUyMCUyMCUyMGxvZ2l0cyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzKS5sb2dpdHMlMEElMEFwcmVkaWN0ZWRfY2xhc3NfaWQlMjAlM0QlMjBsb2dpdHMuYXJnbWF4KCkuaXRlbSgpJTBBJTBBJTIzJTIwVG8lMjB0cmFpbiUyMGElMjBtb2RlbCUyMG9uJTIwJTYwbnVtX2xhYmVscyU2MCUyMGNsYXNzZXMlMkMlMjB5b3UlMjBjYW4lMjBwYXNzJTIwJTYwbnVtX2xhYmVscyUzRG51bV9sYWJlbHMlNjAlMjB0byUyMCU2MC5mcm9tX3ByZXRyYWluZWQoLi4uKSU2MCUwQW51bV9sYWJlbHMlMjAlM0QlMjBsZW4obW9kZWwuY29uZmlnLmlkMmxhYmVsKSUwQW1vZGVsJTIwJTNEJTIwTmV6aGFGb3JTZXF1ZW5jZUNsYXNzaWZpY2F0aW9uLmZyb21fcHJldHJhaW5lZCglMjJzaWp1bmhlJTJGbmV6aGEtY24tYmFzZSUyMiUyQyUyMG51bV9sYWJlbHMlM0RudW1fbGFiZWxzKSUwQSUwQWxhYmVscyUyMCUzRCUyMHRvcmNoLnRlbnNvciglNUIxJTVEKSUwQWxvc3MlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyUyQyUyMGxhYmVscyUzRGxhYmVscykubG9zcw==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, NezhaForSequenceClassification

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;sijunhe/nezha-cn-base&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = NezhaForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;sijunhe/nezha-cn-base&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    logits = model(**inputs).logits

<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_class_id = logits.argmax().item()

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># To train a model on \`num_labels\` classes, you can pass \`num_labels=num_labels\` to \`.from_pretrained(...)\`</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>num_labels = <span class="hljs-built_in">len</span>(model.config.id2label)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = NezhaForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;sijunhe/nezha-cn-base&quot;</span>, num_labels=num_labels)

<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.tensor([<span class="hljs-number">1</span>])
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = model(**inputs, labels=labels).loss`,wrap:!1}}),{c(){t=c("p"),t.textContent=u,o=r(),f(l.$$.fragment)},l(n){t=p(n,"P",{"data-svelte-h":!0}),m(t)!=="svelte-ykxpe4"&&(t.textContent=u),o=i(n),g(l.$$.fragment,n)},m(n,w){h(n,t,w),h(n,o,w),_(l,n,w),y=!0},p:N,i(n){y||(b(l.$$.fragment,n),y=!0)},o(n){M(l.$$.fragment,n),y=!1},d(n){n&&(d(t),d(o)),T(l,n)}}}function Rs(z){let t,u="Example of multi-label classification:",o,l,y;return l=new ce({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Rva2VuaXplciUyQyUyME5lemhhRm9yU2VxdWVuY2VDbGFzc2lmaWNhdGlvbiUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMnNpanVuaGUlMkZuZXpoYS1jbi1iYXNlJTIyKSUwQW1vZGVsJTIwJTNEJTIwTmV6aGFGb3JTZXF1ZW5jZUNsYXNzaWZpY2F0aW9uLmZyb21fcHJldHJhaW5lZCglMjJzaWp1bmhlJTJGbmV6aGEtY24tYmFzZSUyMiUyQyUyMHByb2JsZW1fdHlwZSUzRCUyMm11bHRpX2xhYmVsX2NsYXNzaWZpY2F0aW9uJTIyKSUwQSUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplciglMjJIZWxsbyUyQyUyMG15JTIwZG9nJTIwaXMlMjBjdXRlJTIyJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiklMEElMEF3aXRoJTIwdG9yY2gubm9fZ3JhZCgpJTNBJTBBJTIwJTIwJTIwJTIwbG9naXRzJTIwJTNEJTIwbW9kZWwoKippbnB1dHMpLmxvZ2l0cyUwQSUwQXByZWRpY3RlZF9jbGFzc19pZHMlMjAlM0QlMjB0b3JjaC5hcmFuZ2UoMCUyQyUyMGxvZ2l0cy5zaGFwZSU1Qi0xJTVEKSU1QnRvcmNoLnNpZ21vaWQobG9naXRzKS5zcXVlZXplKGRpbSUzRDApJTIwJTNFJTIwMC41JTVEJTBBJTBBJTIzJTIwVG8lMjB0cmFpbiUyMGElMjBtb2RlbCUyMG9uJTIwJTYwbnVtX2xhYmVscyU2MCUyMGNsYXNzZXMlMkMlMjB5b3UlMjBjYW4lMjBwYXNzJTIwJTYwbnVtX2xhYmVscyUzRG51bV9sYWJlbHMlNjAlMjB0byUyMCU2MC5mcm9tX3ByZXRyYWluZWQoLi4uKSU2MCUwQW51bV9sYWJlbHMlMjAlM0QlMjBsZW4obW9kZWwuY29uZmlnLmlkMmxhYmVsKSUwQW1vZGVsJTIwJTNEJTIwTmV6aGFGb3JTZXF1ZW5jZUNsYXNzaWZpY2F0aW9uLmZyb21fcHJldHJhaW5lZCglMEElMjAlMjAlMjAlMjAlMjJzaWp1bmhlJTJGbmV6aGEtY24tYmFzZSUyMiUyQyUyMG51bV9sYWJlbHMlM0RudW1fbGFiZWxzJTJDJTIwcHJvYmxlbV90eXBlJTNEJTIybXVsdGlfbGFiZWxfY2xhc3NpZmljYXRpb24lMjIlMEEpJTBBJTBBbGFiZWxzJTIwJTNEJTIwdG9yY2guc3VtKCUwQSUyMCUyMCUyMCUyMHRvcmNoLm5uLmZ1bmN0aW9uYWwub25lX2hvdChwcmVkaWN0ZWRfY2xhc3NfaWRzJTVCTm9uZSUyQyUyMCUzQSU1RC5jbG9uZSgpJTJDJTIwbnVtX2NsYXNzZXMlM0RudW1fbGFiZWxzKSUyQyUyMGRpbSUzRDElMEEpLnRvKHRvcmNoLmZsb2F0KSUwQWxvc3MlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyUyQyUyMGxhYmVscyUzRGxhYmVscykubG9zcw==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, NezhaForSequenceClassification

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;sijunhe/nezha-cn-base&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = NezhaForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;sijunhe/nezha-cn-base&quot;</span>, problem_type=<span class="hljs-string">&quot;multi_label_classification&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    logits = model(**inputs).logits

<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_class_ids = torch.arange(<span class="hljs-number">0</span>, logits.shape[-<span class="hljs-number">1</span>])[torch.sigmoid(logits).squeeze(dim=<span class="hljs-number">0</span>) &gt; <span class="hljs-number">0.5</span>]

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># To train a model on \`num_labels\` classes, you can pass \`num_labels=num_labels\` to \`.from_pretrained(...)\`</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>num_labels = <span class="hljs-built_in">len</span>(model.config.id2label)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = NezhaForSequenceClassification.from_pretrained(
<span class="hljs-meta">... </span>    <span class="hljs-string">&quot;sijunhe/nezha-cn-base&quot;</span>, num_labels=num_labels, problem_type=<span class="hljs-string">&quot;multi_label_classification&quot;</span>
<span class="hljs-meta">... </span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.<span class="hljs-built_in">sum</span>(
<span class="hljs-meta">... </span>    torch.nn.functional.one_hot(predicted_class_ids[<span class="hljs-literal">None</span>, :].clone(), num_classes=num_labels), dim=<span class="hljs-number">1</span>
<span class="hljs-meta">... </span>).to(torch.<span class="hljs-built_in">float</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = model(**inputs, labels=labels).loss`,wrap:!1}}),{c(){t=c("p"),t.textContent=u,o=r(),f(l.$$.fragment)},l(n){t=p(n,"P",{"data-svelte-h":!0}),m(t)!=="svelte-1l8e32d"&&(t.textContent=u),o=i(n),g(l.$$.fragment,n)},m(n,w){h(n,t,w),h(n,o,w),_(l,n,w),y=!0},p:N,i(n){y||(b(l.$$.fragment,n),y=!0)},o(n){M(l.$$.fragment,n),y=!1},d(n){n&&(d(t),d(o)),T(l,n)}}}function Es(z){let t,u=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=u},l(o){t=p(o,"P",{"data-svelte-h":!0}),m(t)!=="svelte-fincs2"&&(t.innerHTML=u)},m(o,l){h(o,t,l)},p:N,d(o){o&&d(t)}}}function Ls(z){let t,u="Example:",o,l,y;return l=new ce({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBOZXpoYUZvck11bHRpcGxlQ2hvaWNlJTBBaW1wb3J0JTIwdG9yY2glMEElMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJzaWp1bmhlJTJGbmV6aGEtY24tYmFzZSUyMiklMEFtb2RlbCUyMCUzRCUyME5lemhhRm9yTXVsdGlwbGVDaG9pY2UuZnJvbV9wcmV0cmFpbmVkKCUyMnNpanVuaGUlMkZuZXpoYS1jbi1iYXNlJTIyKSUwQSUwQXByb21wdCUyMCUzRCUyMCUyMkluJTIwSXRhbHklMkMlMjBwaXp6YSUyMHNlcnZlZCUyMGluJTIwZm9ybWFsJTIwc2V0dGluZ3MlMkMlMjBzdWNoJTIwYXMlMjBhdCUyMGElMjByZXN0YXVyYW50JTJDJTIwaXMlMjBwcmVzZW50ZWQlMjB1bnNsaWNlZC4lMjIlMEFjaG9pY2UwJTIwJTNEJTIwJTIySXQlMjBpcyUyMGVhdGVuJTIwd2l0aCUyMGElMjBmb3JrJTIwYW5kJTIwYSUyMGtuaWZlLiUyMiUwQWNob2ljZTElMjAlM0QlMjAlMjJJdCUyMGlzJTIwZWF0ZW4lMjB3aGlsZSUyMGhlbGQlMjBpbiUyMHRoZSUyMGhhbmQuJTIyJTBBbGFiZWxzJTIwJTNEJTIwdG9yY2gudGVuc29yKDApLnVuc3F1ZWV6ZSgwKSUyMCUyMCUyMyUyMGNob2ljZTAlMjBpcyUyMGNvcnJlY3QlMjAoYWNjb3JkaW5nJTIwdG8lMjBXaWtpcGVkaWElMjAlM0IpKSUyQyUyMGJhdGNoJTIwc2l6ZSUyMDElMEElMEFlbmNvZGluZyUyMCUzRCUyMHRva2VuaXplciglNUJwcm9tcHQlMkMlMjBwcm9tcHQlNUQlMkMlMjAlNUJjaG9pY2UwJTJDJTIwY2hvaWNlMSU1RCUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIlMkMlMjBwYWRkaW5nJTNEVHJ1ZSklMEFvdXRwdXRzJTIwJTNEJTIwbW9kZWwoKiolN0JrJTNBJTIwdi51bnNxdWVlemUoMCklMjBmb3IlMjBrJTJDJTIwdiUyMGluJTIwZW5jb2RpbmcuaXRlbXMoKSU3RCUyQyUyMGxhYmVscyUzRGxhYmVscyklMjAlMjAlMjMlMjBiYXRjaCUyMHNpemUlMjBpcyUyMDElMEElMEElMjMlMjB0aGUlMjBsaW5lYXIlMjBjbGFzc2lmaWVyJTIwc3RpbGwlMjBuZWVkcyUyMHRvJTIwYmUlMjB0cmFpbmVkJTBBbG9zcyUyMCUzRCUyMG91dHB1dHMubG9zcyUwQWxvZ2l0cyUyMCUzRCUyMG91dHB1dHMubG9naXRz",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, NezhaForMultipleChoice
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;sijunhe/nezha-cn-base&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = NezhaForMultipleChoice.from_pretrained(<span class="hljs-string">&quot;sijunhe/nezha-cn-base&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>prompt = <span class="hljs-string">&quot;In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>choice0 = <span class="hljs-string">&quot;It is eaten with a fork and a knife.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>choice1 = <span class="hljs-string">&quot;It is eaten while held in the hand.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.tensor(<span class="hljs-number">0</span>).unsqueeze(<span class="hljs-number">0</span>)  <span class="hljs-comment"># choice0 is correct (according to Wikipedia ;)), batch size 1</span>

<span class="hljs-meta">&gt;&gt;&gt; </span>encoding = tokenizer([prompt, prompt], [choice0, choice1], return_tensors=<span class="hljs-string">&quot;pt&quot;</span>, padding=<span class="hljs-literal">True</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**{k: v.unsqueeze(<span class="hljs-number">0</span>) <span class="hljs-keyword">for</span> k, v <span class="hljs-keyword">in</span> encoding.items()}, labels=labels)  <span class="hljs-comment"># batch size is 1</span>

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># the linear classifier still needs to be trained</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = outputs.loss
<span class="hljs-meta">&gt;&gt;&gt; </span>logits = outputs.logits`,wrap:!1}}),{c(){t=c("p"),t.textContent=u,o=r(),f(l.$$.fragment)},l(n){t=p(n,"P",{"data-svelte-h":!0}),m(t)!=="svelte-11lpom8"&&(t.textContent=u),o=i(n),g(l.$$.fragment,n)},m(n,w){h(n,t,w),h(n,o,w),_(l,n,w),y=!0},p:N,i(n){y||(b(l.$$.fragment,n),y=!0)},o(n){M(l.$$.fragment,n),y=!1},d(n){n&&(d(t),d(o)),T(l,n)}}}function Ps(z){let t,u=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=u},l(o){t=p(o,"P",{"data-svelte-h":!0}),m(t)!=="svelte-fincs2"&&(t.innerHTML=u)},m(o,l){h(o,t,l)},p:N,d(o){o&&d(t)}}}function Qs(z){let t,u="Example:",o,l,y;return l=new ce({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBOZXpoYUZvclRva2VuQ2xhc3NpZmljYXRpb24lMEFpbXBvcnQlMjB0b3JjaCUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMnNpanVuaGUlMkZuZXpoYS1jbi1iYXNlJTIyKSUwQW1vZGVsJTIwJTNEJTIwTmV6aGFGb3JUb2tlbkNsYXNzaWZpY2F0aW9uLmZyb21fcHJldHJhaW5lZCglMjJzaWp1bmhlJTJGbmV6aGEtY24tYmFzZSUyMiklMEElMEFpbnB1dHMlMjAlM0QlMjB0b2tlbml6ZXIoJTBBJTIwJTIwJTIwJTIwJTIySHVnZ2luZ0ZhY2UlMjBpcyUyMGElMjBjb21wYW55JTIwYmFzZWQlMjBpbiUyMFBhcmlzJTIwYW5kJTIwTmV3JTIwWW9yayUyMiUyQyUyMGFkZF9zcGVjaWFsX3Rva2VucyUzREZhbHNlJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiUwQSklMEElMEF3aXRoJTIwdG9yY2gubm9fZ3JhZCgpJTNBJTBBJTIwJTIwJTIwJTIwbG9naXRzJTIwJTNEJTIwbW9kZWwoKippbnB1dHMpLmxvZ2l0cyUwQSUwQXByZWRpY3RlZF90b2tlbl9jbGFzc19pZHMlMjAlM0QlMjBsb2dpdHMuYXJnbWF4KC0xKSUwQSUwQSUyMyUyME5vdGUlMjB0aGF0JTIwdG9rZW5zJTIwYXJlJTIwY2xhc3NpZmllZCUyMHJhdGhlciUyMHRoZW4lMjBpbnB1dCUyMHdvcmRzJTIwd2hpY2glMjBtZWFucyUyMHRoYXQlMEElMjMlMjB0aGVyZSUyMG1pZ2h0JTIwYmUlMjBtb3JlJTIwcHJlZGljdGVkJTIwdG9rZW4lMjBjbGFzc2VzJTIwdGhhbiUyMHdvcmRzLiUwQSUyMyUyME11bHRpcGxlJTIwdG9rZW4lMjBjbGFzc2VzJTIwbWlnaHQlMjBhY2NvdW50JTIwZm9yJTIwdGhlJTIwc2FtZSUyMHdvcmQlMEFwcmVkaWN0ZWRfdG9rZW5zX2NsYXNzZXMlMjAlM0QlMjAlNUJtb2RlbC5jb25maWcuaWQybGFiZWwlNUJ0Lml0ZW0oKSU1RCUyMGZvciUyMHQlMjBpbiUyMHByZWRpY3RlZF90b2tlbl9jbGFzc19pZHMlNUIwJTVEJTVEJTBBJTBBbGFiZWxzJTIwJTNEJTIwcHJlZGljdGVkX3Rva2VuX2NsYXNzX2lkcyUwQWxvc3MlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyUyQyUyMGxhYmVscyUzRGxhYmVscykubG9zcw==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, NezhaForTokenClassification
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;sijunhe/nezha-cn-base&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = NezhaForTokenClassification.from_pretrained(<span class="hljs-string">&quot;sijunhe/nezha-cn-base&quot;</span>)

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
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = model(**inputs, labels=labels).loss`,wrap:!1}}),{c(){t=c("p"),t.textContent=u,o=r(),f(l.$$.fragment)},l(n){t=p(n,"P",{"data-svelte-h":!0}),m(t)!=="svelte-11lpom8"&&(t.textContent=u),o=i(n),g(l.$$.fragment,n)},m(n,w){h(n,t,w),h(n,o,w),_(l,n,w),y=!0},p:N,i(n){y||(b(l.$$.fragment,n),y=!0)},o(n){M(l.$$.fragment,n),y=!1},d(n){n&&(d(t),d(o)),T(l,n)}}}function Ys(z){let t,u=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=u},l(o){t=p(o,"P",{"data-svelte-h":!0}),m(t)!=="svelte-fincs2"&&(t.innerHTML=u)},m(o,l){h(o,t,l)},p:N,d(o){o&&d(t)}}}function As(z){let t,u="Example:",o,l,y;return l=new ce({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBOZXpoYUZvclF1ZXN0aW9uQW5zd2VyaW5nJTBBaW1wb3J0JTIwdG9yY2glMEElMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJzaWp1bmhlJTJGbmV6aGEtY24tYmFzZSUyMiklMEFtb2RlbCUyMCUzRCUyME5lemhhRm9yUXVlc3Rpb25BbnN3ZXJpbmcuZnJvbV9wcmV0cmFpbmVkKCUyMnNpanVuaGUlMkZuZXpoYS1jbi1iYXNlJTIyKSUwQSUwQXF1ZXN0aW9uJTJDJTIwdGV4dCUyMCUzRCUyMCUyMldobyUyMHdhcyUyMEppbSUyMEhlbnNvbiUzRiUyMiUyQyUyMCUyMkppbSUyMEhlbnNvbiUyMHdhcyUyMGElMjBuaWNlJTIwcHVwcGV0JTIyJTBBJTBBaW5wdXRzJTIwJTNEJTIwdG9rZW5pemVyKHF1ZXN0aW9uJTJDJTIwdGV4dCUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpJTBBd2l0aCUyMHRvcmNoLm5vX2dyYWQoKSUzQSUwQSUyMCUyMCUyMCUyMG91dHB1dHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyklMEElMEFhbnN3ZXJfc3RhcnRfaW5kZXglMjAlM0QlMjBvdXRwdXRzLnN0YXJ0X2xvZ2l0cy5hcmdtYXgoKSUwQWFuc3dlcl9lbmRfaW5kZXglMjAlM0QlMjBvdXRwdXRzLmVuZF9sb2dpdHMuYXJnbWF4KCklMEElMEFwcmVkaWN0X2Fuc3dlcl90b2tlbnMlMjAlM0QlMjBpbnB1dHMuaW5wdXRfaWRzJTVCMCUyQyUyMGFuc3dlcl9zdGFydF9pbmRleCUyMCUzQSUyMGFuc3dlcl9lbmRfaW5kZXglMjAlMkIlMjAxJTVEJTBBJTBBJTIzJTIwdGFyZ2V0JTIwaXMlMjAlMjJuaWNlJTIwcHVwcGV0JTIyJTBBdGFyZ2V0X3N0YXJ0X2luZGV4JTIwJTNEJTIwdG9yY2gudGVuc29yKCU1QjE0JTVEKSUwQXRhcmdldF9lbmRfaW5kZXglMjAlM0QlMjB0b3JjaC50ZW5zb3IoJTVCMTUlNUQpJTBBJTBBb3V0cHV0cyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzJTJDJTIwc3RhcnRfcG9zaXRpb25zJTNEdGFyZ2V0X3N0YXJ0X2luZGV4JTJDJTIwZW5kX3Bvc2l0aW9ucyUzRHRhcmdldF9lbmRfaW5kZXgpJTBBbG9zcyUyMCUzRCUyMG91dHB1dHMubG9zcw==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, NezhaForQuestionAnswering
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;sijunhe/nezha-cn-base&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = NezhaForQuestionAnswering.from_pretrained(<span class="hljs-string">&quot;sijunhe/nezha-cn-base&quot;</span>)

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
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = outputs.loss`,wrap:!1}}),{c(){t=c("p"),t.textContent=u,o=r(),f(l.$$.fragment)},l(n){t=p(n,"P",{"data-svelte-h":!0}),m(t)!=="svelte-11lpom8"&&(t.textContent=u),o=i(n),g(l.$$.fragment,n)},m(n,w){h(n,t,w),h(n,o,w),_(l,n,w),y=!0},p:N,i(n){y||(b(l.$$.fragment,n),y=!0)},o(n){M(l.$$.fragment,n),y=!1},d(n){n&&(d(t),d(o)),T(l,n)}}}function Os(z){let t,u,o,l,y,n="<em>This model was released on 2019-08-31 and added to Hugging Face Transformers on 2023-06-20.</em>",w,xe,At,he,Io='<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/>',Ot,me,Dt,Fe,Kt,Ze,Go='The Nezha model was proposed in <a href="https://huggingface.co/papers/1909.00204" rel="nofollow">NEZHA: Neural Contextualized Representation for Chinese Language Understanding</a> by Junqiu Wei et al.',en,We,Ho="The abstract from the paper is the following:",tn,Ie,Vo=`<em>The pre-trained language models have achieved great successes in various natural language understanding (NLU) tasks
due to its capacity to capture the deep contextualized information in text by pre-training on large-scale corpora.
In this technical report, we present our practice of pre-training language models named NEZHA (NEural contextualiZed
representation for CHinese lAnguage understanding) on Chinese corpora and finetuning for the Chinese NLU tasks.
The current version of NEZHA is based on BERT with a collection of proven improvements, which include Functional
Relative Positional Encoding as an effective positional encoding scheme, Whole Word Masking strategy,
Mixed Precision Training and the LAMB Optimizer in training the models. The experimental results show that NEZHA
achieves the state-of-the-art performances when finetuned on several representative Chinese tasks, including
named entity recognition (People’s Daily NER), sentence matching (LCQMC), Chinese sentiment classification (ChnSenti)
and natural language inference (XNLI).</em>`,nn,Ge,Bo='This model was contributed by <a href="https://huggingface.co/sijunhe" rel="nofollow">sijunhe</a>. The original code can be found <a href="https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/NEZHA-PyTorch" rel="nofollow">here</a>.',on,He,sn,Ve,qo='<li><a href="../tasks/sequence_classification">Text classification task guide</a></li> <li><a href="../tasks/token_classification">Token classification task guide</a></li> <li><a href="../tasks/question_answering">Question answering task guide</a></li> <li><a href="../tasks/masked_language_modeling">Masked language modeling task guide</a></li> <li><a href="../tasks/multiple_choice">Multiple choice task guide</a></li>',an,Be,rn,H,qe,Nn,mt,So=`This is the configuration class to store the configuration of an <a href="/docs/transformers/v4.56.2/en/model_doc/nezha#transformers.NezhaModel">NezhaModel</a>. It is used to instantiate an Nezha
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the Nezha
<a href="https://huggingface.co/sijunhe/nezha-cn-base" rel="nofollow">sijunhe/nezha-cn-base</a> architecture.`,jn,ut,Xo=`Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> and can be used to control the model outputs. Read the
documentation from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> for more information.`,Jn,ue,ln,Se,dn,v,Xe,Un,ft,Ro="The bare Nezha Model transformer outputting raw hidden-states without any specific head on top.",Cn,gt,Eo=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,xn,_t,Lo=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Fn,bt,Po=`The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
cross-attention is added between the self-attention layers, following the architecture described in <a href="https://huggingface.co/papers/1706.03762" rel="nofollow">Attention is
all you need</a> by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.`,Zn,Mt,Qo=`To behave as an decoder the model needs to be initialized with the <code>is_decoder</code> argument of the configuration set
to <code>True</code>. To be used in a Seq2Seq model, the model needs to initialized with both <code>is_decoder</code> argument and
<code>add_cross_attention</code> set to <code>True</code>; an <code>encoder_hidden_states</code> is then expected as an input to the forward pass.`,Wn,P,Re,In,Tt,Yo='The <a href="/docs/transformers/v4.56.2/en/model_doc/nezha#transformers.NezhaModel">NezhaModel</a> forward method, overrides the <code>__call__</code> special method.',Gn,fe,Hn,ge,cn,Ee,pn,U,Le,Vn,yt,Ao="Nezha Model with two heads on top as done during the pretraining: a <code>masked language modeling</code> head and a <code>next sentence prediction (classification)</code> head.",Bn,wt,Oo=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,qn,zt,Do=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Sn,Q,Pe,Xn,kt,Ko='The <a href="/docs/transformers/v4.56.2/en/model_doc/nezha#transformers.NezhaForPreTraining">NezhaForPreTraining</a> forward method, overrides the <code>__call__</code> special method.',Rn,_e,En,be,hn,Qe,mn,C,Ye,Ln,vt,es="Nezha Model with a <code>language modeling</code> head on top.",Pn,$t,ts=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Qn,Nt,ns=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Yn,Y,Ae,An,jt,os='The <a href="/docs/transformers/v4.56.2/en/model_doc/nezha#transformers.NezhaForMaskedLM">NezhaForMaskedLM</a> forward method, overrides the <code>__call__</code> special method.',On,Me,Dn,Te,un,Oe,fn,x,De,Kn,Jt,ss="Nezha Model with a <code>next sentence prediction (classification)</code> head on top.",eo,Ut,as=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,to,Ct,rs=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,no,A,Ke,oo,xt,is='The <a href="/docs/transformers/v4.56.2/en/model_doc/nezha#transformers.NezhaForNextSentencePrediction">NezhaForNextSentencePrediction</a> forward method, overrides the <code>__call__</code> special method.',so,ye,ao,we,gn,et,_n,F,tt,ro,Ft,ls=`Nezha Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled
output) e.g. for GLUE tasks.`,io,Zt,ds=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,lo,Wt,cs=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,co,G,nt,po,It,ps='The <a href="/docs/transformers/v4.56.2/en/model_doc/nezha#transformers.NezhaForSequenceClassification">NezhaForSequenceClassification</a> forward method, overrides the <code>__call__</code> special method.',ho,ze,mo,ke,uo,ve,bn,ot,Mn,Z,st,fo,Gt,hs=`Nezha Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
softmax) e.g. for RocStories/SWAG tasks.`,go,Ht,ms=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,_o,Vt,us=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,bo,O,at,Mo,Bt,fs='The <a href="/docs/transformers/v4.56.2/en/model_doc/nezha#transformers.NezhaForMultipleChoice">NezhaForMultipleChoice</a> forward method, overrides the <code>__call__</code> special method.',To,$e,yo,Ne,Tn,rt,yn,W,it,wo,qt,gs=`Nezha Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
Named-Entity-Recognition (NER) tasks.`,zo,St,_s=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,ko,Xt,bs=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,vo,D,lt,$o,Rt,Ms='The <a href="/docs/transformers/v4.56.2/en/model_doc/nezha#transformers.NezhaForTokenClassification">NezhaForTokenClassification</a> forward method, overrides the <code>__call__</code> special method.',No,je,jo,Je,wn,dt,zn,I,ct,Jo,Et,Ts=`Nezha Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
layers on top of the hidden-states output to compute <code>span start logits</code> and <code>span end logits</code>).`,Uo,Lt,ys=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Co,Pt,ws=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,xo,K,pt,Fo,Qt,zs='The <a href="/docs/transformers/v4.56.2/en/model_doc/nezha#transformers.NezhaForQuestionAnswering">NezhaForQuestionAnswering</a> forward method, overrides the <code>__call__</code> special method.',Zo,Ue,Wo,Ce,kn,ht,vn,Yt,$n;return xe=new ee({props:{title:"Nezha",local:"nezha",headingTag:"h1"}}),me=new pe({props:{warning:!0,$$slots:{default:[xs]},$$scope:{ctx:z}}}),Fe=new ee({props:{title:"Overview",local:"overview",headingTag:"h2"}}),He=new ee({props:{title:"Resources",local:"resources",headingTag:"h2"}}),Be=new ee({props:{title:"NezhaConfig",local:"transformers.NezhaConfig",headingTag:"h2"}}),qe=new J({props:{name:"class transformers.NezhaConfig",anchor:"transformers.NezhaConfig",parameters:[{name:"vocab_size",val:" = 21128"},{name:"hidden_size",val:" = 768"},{name:"num_hidden_layers",val:" = 12"},{name:"num_attention_heads",val:" = 12"},{name:"intermediate_size",val:" = 3072"},{name:"hidden_act",val:" = 'gelu'"},{name:"hidden_dropout_prob",val:" = 0.1"},{name:"attention_probs_dropout_prob",val:" = 0.1"},{name:"max_position_embeddings",val:" = 512"},{name:"max_relative_position",val:" = 64"},{name:"type_vocab_size",val:" = 2"},{name:"initializer_range",val:" = 0.02"},{name:"layer_norm_eps",val:" = 1e-12"},{name:"classifier_dropout",val:" = 0.1"},{name:"pad_token_id",val:" = 0"},{name:"bos_token_id",val:" = 2"},{name:"eos_token_id",val:" = 3"},{name:"use_cache",val:" = True"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.NezhaConfig.vocab_size",description:`<strong>vocab_size</strong> (<code>int</code>, optional, defaults to 21128) &#x2014;
Vocabulary size of the NEZHA model. Defines the different tokens that can be represented by the
<em>inputs_ids</em> passed to the forward method of <a href="/docs/transformers/v4.56.2/en/model_doc/nezha#transformers.NezhaModel">NezhaModel</a>.`,name:"vocab_size"},{anchor:"transformers.NezhaConfig.hidden_size",description:`<strong>hidden_size</strong> (<code>int</code>, optional, defaults to 768) &#x2014;
Dimensionality of the encoder layers and the pooler layer.`,name:"hidden_size"},{anchor:"transformers.NezhaConfig.num_hidden_layers",description:`<strong>num_hidden_layers</strong> (<code>int</code>, optional, defaults to 12) &#x2014;
Number of hidden layers in the Transformer encoder.`,name:"num_hidden_layers"},{anchor:"transformers.NezhaConfig.num_attention_heads",description:`<strong>num_attention_heads</strong> (<code>int</code>, optional, defaults to 12) &#x2014;
Number of attention heads for each attention layer in the Transformer encoder.`,name:"num_attention_heads"},{anchor:"transformers.NezhaConfig.intermediate_size",description:`<strong>intermediate_size</strong> (<code>int</code>, optional, defaults to 3072) &#x2014;
The dimensionality of the &#x201C;intermediate&#x201D; (i.e., feed-forward) layer in the Transformer encoder.`,name:"intermediate_size"},{anchor:"transformers.NezhaConfig.hidden_act",description:`<strong>hidden_act</strong> (<code>str</code> or <code>function</code>, optional, defaults to &#x201C;gelu&#x201D;) &#x2014;
The non-linear activation function (function or string) in the encoder and pooler.`,name:"hidden_act"},{anchor:"transformers.NezhaConfig.hidden_dropout_prob",description:`<strong>hidden_dropout_prob</strong> (<code>float</code>, optional, defaults to 0.1) &#x2014;
The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.`,name:"hidden_dropout_prob"},{anchor:"transformers.NezhaConfig.attention_probs_dropout_prob",description:`<strong>attention_probs_dropout_prob</strong> (<code>float</code>, optional, defaults to 0.1) &#x2014;
The dropout ratio for the attention probabilities.`,name:"attention_probs_dropout_prob"},{anchor:"transformers.NezhaConfig.max_position_embeddings",description:`<strong>max_position_embeddings</strong> (<code>int</code>, optional, defaults to 512) &#x2014;
The maximum sequence length that this model might ever be used with. Typically set this to something large
(e.g., 512 or 1024 or 2048).`,name:"max_position_embeddings"},{anchor:"transformers.NezhaConfig.type_vocab_size",description:`<strong>type_vocab_size</strong> (<code>int</code>, optional, defaults to 2) &#x2014;
The vocabulary size of the <em>token_type_ids</em> passed into <a href="/docs/transformers/v4.56.2/en/model_doc/nezha#transformers.NezhaModel">NezhaModel</a>.`,name:"type_vocab_size"},{anchor:"transformers.NezhaConfig.initializer_range",description:`<strong>initializer_range</strong> (<code>float</code>, optional, defaults to 0.02) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.`,name:"initializer_range"},{anchor:"transformers.NezhaConfig.layer_norm_eps",description:`<strong>layer_norm_eps</strong> (<code>float</code>, optional, defaults to 1e-12) &#x2014;
The epsilon used by the layer normalization layers.`,name:"layer_norm_eps"},{anchor:"transformers.NezhaConfig.classifier_dropout",description:`<strong>classifier_dropout</strong> (<code>float</code>, optional, defaults to 0.1) &#x2014;
The dropout ratio for attached classifiers.`,name:"classifier_dropout"},{anchor:"transformers.NezhaConfig.is_decoder",description:`<strong>is_decoder</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether the model is used as a decoder or not. If <code>False</code>, the model is used as an encoder.`,name:"is_decoder"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/nezha/configuration_nezha.py#L4"}}),ue=new de({props:{anchor:"transformers.NezhaConfig.example",$$slots:{default:[Fs]},$$scope:{ctx:z}}}),Se=new ee({props:{title:"NezhaModel",local:"transformers.NezhaModel",headingTag:"h2"}}),Xe=new J({props:{name:"class transformers.NezhaModel",anchor:"transformers.NezhaModel",parameters:[{name:"config",val:""},{name:"add_pooling_layer",val:" = True"}],parametersDescription:[{anchor:"transformers.NezhaModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/nezha#transformers.NezhaConfig">NezhaConfig</a>) &#x2014; Model configuration class with all the parameters of the model.
Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/nezha/modeling_nezha.py#L826"}}),Re=new J({props:{name:"forward",anchor:"transformers.NezhaModel.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_hidden_states",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"past_key_values",val:": typing.Optional[list[torch.FloatTensor]] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.NezhaModel.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.NezhaModel.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.NezhaModel.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.NezhaModel.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.NezhaModel.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.NezhaModel.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.NezhaModel.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.NezhaModel.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.NezhaModel.forward.encoder_hidden_states",description:`<strong>encoder_hidden_states</strong>  (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
the model is configured as a decoder.`,name:"encoder_hidden_states"},{anchor:"transformers.NezhaModel.forward.encoder_attention_mask",description:`<strong>encoder_attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
the cross-attention if the model is configured as a decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>`,name:"encoder_attention_mask"},{anchor:"transformers.NezhaModel.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>tuple(tuple(torch.FloatTensor))</code> of length <code>config.n_layers</code> with each tuple having 4 tensors of shape <code>(batch_size, num_heads, sequence_length - 1, embed_size_per_head)</code>) &#x2014;
Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.</p>
<p>If <code>past_key_values</code> are used, the user can optionally input only the last <code>decoder_input_ids</code> (those that
don&#x2019;t have their past key value states given to this model) of shape <code>(batch_size, 1)</code> instead of all
<code>decoder_input_ids</code> of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.NezhaModel.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/nezha/modeling_nezha.py#L865",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions"
>transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/nezha#transformers.NezhaConfig"
>NezhaConfig</a>) and inputs.</p>
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
`}}),fe=new pe({props:{$$slots:{default:[Zs]},$$scope:{ctx:z}}}),ge=new de({props:{anchor:"transformers.NezhaModel.forward.example",$$slots:{default:[Ws]},$$scope:{ctx:z}}}),Ee=new ee({props:{title:"NezhaForPreTraining",local:"transformers.NezhaForPreTraining",headingTag:"h2"}}),Le=new J({props:{name:"class transformers.NezhaForPreTraining",anchor:"transformers.NezhaForPreTraining",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.NezhaForPreTraining.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/nezha#transformers.NezhaConfig">NezhaConfig</a>) &#x2014; Model configuration class with all the parameters of the model.
Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/nezha/modeling_nezha.py#L1006"}}),Pe=new J({props:{name:"forward",anchor:"transformers.NezhaForPreTraining.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"labels",val:": typing.Optional[torch.Tensor] = None"},{name:"next_sentence_label",val:": typing.Optional[torch.Tensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.NezhaForPreTraining.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.NezhaForPreTraining.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.NezhaForPreTraining.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.NezhaForPreTraining.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.NezhaForPreTraining.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.NezhaForPreTraining.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.NezhaForPreTraining.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.NezhaForPreTraining.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.NezhaForPreTraining.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should be in <code>[-100, 0, ..., config.vocab_size]</code> (see <code>input_ids</code> docstring) Tokens with indices set to <code>-100</code> are ignored (masked),
the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>`,name:"labels"},{anchor:"transformers.NezhaForPreTraining.forward.next_sentence_label",description:`<strong>next_sentence_label</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the next sequence prediction (classification) loss. Input should be a sequence
pair (see <code>input_ids</code> docstring) Indices should be in <code>[0, 1]</code>:</p>
<ul>
<li>0 indicates sequence B is a continuation of sequence A,</li>
<li>1 indicates sequence B is a random sequence.</li>
</ul>`,name:"next_sentence_label"},{anchor:"transformers.NezhaForPreTraining.forward.kwargs",description:`<strong>kwargs</strong> (<code>dict[str, any]</code>, optional, defaults to <em>{}</em>) &#x2014;
Used to hide legacy arguments that have been deprecated.`,name:"kwargs"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/nezha/modeling_nezha.py#L1025",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <code>transformers.models.deprecated.nezha.modeling_nezha.NezhaForPreTrainingOutput</code> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/nezha#transformers.NezhaConfig"
>NezhaConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>loss</strong> (<em>optional</em>, returned when <code>labels</code> is provided, <code>torch.FloatTensor</code> of shape <code>(1,)</code>) — Total loss as the sum of the masked language modeling loss and the next sequence prediction
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
<p><strong>hidden_states</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings + one for the output of each layer) of
shape <code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the model at the output of each layer plus the initial embedding outputs.</p>
</li>
<li>
<p><strong>attentions</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
heads.</p>
</li>
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>transformers.models.deprecated.nezha.modeling_nezha.NezhaForPreTrainingOutput</code> or <code>tuple(torch.FloatTensor)</code></p>
`}}),_e=new pe({props:{$$slots:{default:[Is]},$$scope:{ctx:z}}}),be=new de({props:{anchor:"transformers.NezhaForPreTraining.forward.example",$$slots:{default:[Gs]},$$scope:{ctx:z}}}),Qe=new ee({props:{title:"NezhaForMaskedLM",local:"transformers.NezhaForMaskedLM",headingTag:"h2"}}),Ye=new J({props:{name:"class transformers.NezhaForMaskedLM",anchor:"transformers.NezhaForMaskedLM",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.NezhaForMaskedLM.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/nezha#transformers.NezhaConfig">NezhaConfig</a>) &#x2014; Model configuration class with all the parameters of the model.
Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/nezha/modeling_nezha.py#L1109"}}),Ae=new J({props:{name:"forward",anchor:"transformers.NezhaForMaskedLM.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_hidden_states",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"labels",val:": typing.Optional[torch.Tensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.NezhaForMaskedLM.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.NezhaForMaskedLM.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.NezhaForMaskedLM.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.NezhaForMaskedLM.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.NezhaForMaskedLM.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.NezhaForMaskedLM.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.NezhaForMaskedLM.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.NezhaForMaskedLM.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.NezhaForMaskedLM.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should be in <code>[-100, 0, ..., config.vocab_size]</code> (see <code>input_ids</code> docstring) Tokens with indices set to <code>-100</code> are ignored (masked), the
loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>`,name:"labels"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/nezha/modeling_nezha.py#L1134",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.MaskedLMOutput"
>transformers.modeling_outputs.MaskedLMOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/nezha#transformers.NezhaConfig"
>NezhaConfig</a>) and inputs.</p>
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
`}}),Me=new pe({props:{$$slots:{default:[Hs]},$$scope:{ctx:z}}}),Te=new de({props:{anchor:"transformers.NezhaForMaskedLM.forward.example",$$slots:{default:[Vs]},$$scope:{ctx:z}}}),Oe=new ee({props:{title:"NezhaForNextSentencePrediction",local:"transformers.NezhaForNextSentencePrediction",headingTag:"h2"}}),De=new J({props:{name:"class transformers.NezhaForNextSentencePrediction",anchor:"transformers.NezhaForNextSentencePrediction",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.NezhaForNextSentencePrediction.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/nezha#transformers.NezhaConfig">NezhaConfig</a>) &#x2014; Model configuration class with all the parameters of the model.
Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/nezha/modeling_nezha.py#L1216"}}),Ke=new J({props:{name:"forward",anchor:"transformers.NezhaForNextSentencePrediction.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"labels",val:": typing.Optional[torch.Tensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.NezhaForNextSentencePrediction.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.NezhaForNextSentencePrediction.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.NezhaForNextSentencePrediction.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.NezhaForNextSentencePrediction.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.NezhaForNextSentencePrediction.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.NezhaForNextSentencePrediction.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.NezhaForNextSentencePrediction.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.NezhaForNextSentencePrediction.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.NezhaForNextSentencePrediction.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the next sequence prediction (classification) loss. Input should be a sequence pair
(see <code>input_ids</code> docstring). Indices should be in <code>[0, 1]</code>:</p>
<ul>
<li>0 indicates sequence B is a continuation of sequence A,</li>
<li>1 indicates sequence B is a random sequence.</li>
</ul>`,name:"labels"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/nezha/modeling_nezha.py#L1226",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.NextSentencePredictorOutput"
>transformers.modeling_outputs.NextSentencePredictorOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/nezha#transformers.NezhaConfig"
>NezhaConfig</a>) and inputs.</p>
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
`}}),ye=new pe({props:{$$slots:{default:[Bs]},$$scope:{ctx:z}}}),we=new de({props:{anchor:"transformers.NezhaForNextSentencePrediction.forward.example",$$slots:{default:[qs]},$$scope:{ctx:z}}}),et=new ee({props:{title:"NezhaForSequenceClassification",local:"transformers.NezhaForSequenceClassification",headingTag:"h2"}}),tt=new J({props:{name:"class transformers.NezhaForSequenceClassification",anchor:"transformers.NezhaForSequenceClassification",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.NezhaForSequenceClassification.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/nezha#transformers.NezhaConfig">NezhaConfig</a>) &#x2014; Model configuration class with all the parameters of the model.
Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/nezha/modeling_nezha.py#L1319"}}),nt=new J({props:{name:"forward",anchor:"transformers.NezhaForSequenceClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"labels",val:": typing.Optional[torch.Tensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.NezhaForSequenceClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.NezhaForSequenceClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.NezhaForSequenceClassification.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.NezhaForSequenceClassification.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.NezhaForSequenceClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.NezhaForSequenceClassification.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.NezhaForSequenceClassification.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.NezhaForSequenceClassification.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.NezhaForSequenceClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the sequence classification/regression loss. Indices should be in <code>[0, ..., config.num_labels - 1]</code>. If <code>config.num_labels == 1</code> a regression loss is computed (Mean-Square loss), If
<code>config.num_labels &gt; 1</code> a classification loss is computed (Cross-Entropy).`,name:"labels"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/nezha/modeling_nezha.py#L1335",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput"
>transformers.modeling_outputs.SequenceClassifierOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/nezha#transformers.NezhaConfig"
>NezhaConfig</a>) and inputs.</p>
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
`}}),ze=new pe({props:{$$slots:{default:[Ss]},$$scope:{ctx:z}}}),ke=new de({props:{anchor:"transformers.NezhaForSequenceClassification.forward.example",$$slots:{default:[Xs]},$$scope:{ctx:z}}}),ve=new de({props:{anchor:"transformers.NezhaForSequenceClassification.forward.example-2",$$slots:{default:[Rs]},$$scope:{ctx:z}}}),ot=new ee({props:{title:"NezhaForMultipleChoice",local:"transformers.NezhaForMultipleChoice",headingTag:"h2"}}),st=new J({props:{name:"class transformers.NezhaForMultipleChoice",anchor:"transformers.NezhaForMultipleChoice",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.NezhaForMultipleChoice.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/nezha#transformers.NezhaConfig">NezhaConfig</a>) &#x2014; Model configuration class with all the parameters of the model.
Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/nezha/modeling_nezha.py#L1418"}}),at=new J({props:{name:"forward",anchor:"transformers.NezhaForMultipleChoice.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"labels",val:": typing.Optional[torch.Tensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.NezhaForMultipleChoice.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, num_choices, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.NezhaForMultipleChoice.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_choices, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.NezhaForMultipleChoice.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, num_choices, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.NezhaForMultipleChoice.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.NezhaForMultipleChoice.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_choices, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.NezhaForMultipleChoice.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.NezhaForMultipleChoice.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.NezhaForMultipleChoice.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.NezhaForMultipleChoice.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the multiple choice classification loss. Indices should be in <code>[0, ..., num_choices-1]</code> where <code>num_choices</code> is the size of the second dimension of the input tensors. (See
<code>input_ids</code> above)`,name:"labels"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/nezha/modeling_nezha.py#L1432",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.MultipleChoiceModelOutput"
>transformers.modeling_outputs.MultipleChoiceModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/nezha#transformers.NezhaConfig"
>NezhaConfig</a>) and inputs.</p>
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
`}}),$e=new pe({props:{$$slots:{default:[Es]},$$scope:{ctx:z}}}),Ne=new de({props:{anchor:"transformers.NezhaForMultipleChoice.forward.example",$$slots:{default:[Ls]},$$scope:{ctx:z}}}),rt=new ee({props:{title:"NezhaForTokenClassification",local:"transformers.NezhaForTokenClassification",headingTag:"h2"}}),it=new J({props:{name:"class transformers.NezhaForTokenClassification",anchor:"transformers.NezhaForTokenClassification",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.NezhaForTokenClassification.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/nezha#transformers.NezhaConfig">NezhaConfig</a>) &#x2014; Model configuration class with all the parameters of the model.
Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/nezha/modeling_nezha.py#L1510"}}),lt=new J({props:{name:"forward",anchor:"transformers.NezhaForTokenClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"labels",val:": typing.Optional[torch.Tensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.NezhaForTokenClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.NezhaForTokenClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.NezhaForTokenClassification.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.NezhaForTokenClassification.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.NezhaForTokenClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.NezhaForTokenClassification.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.NezhaForTokenClassification.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.NezhaForTokenClassification.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.NezhaForTokenClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the token classification loss. Indices should be in <code>[0, ..., config.num_labels - 1]</code>.`,name:"labels"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/nezha/modeling_nezha.py#L1525",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput"
>transformers.modeling_outputs.TokenClassifierOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/nezha#transformers.NezhaConfig"
>NezhaConfig</a>) and inputs.</p>
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
`}}),je=new pe({props:{$$slots:{default:[Ps]},$$scope:{ctx:z}}}),Je=new de({props:{anchor:"transformers.NezhaForTokenClassification.forward.example",$$slots:{default:[Qs]},$$scope:{ctx:z}}}),dt=new ee({props:{title:"NezhaForQuestionAnswering",local:"transformers.NezhaForQuestionAnswering",headingTag:"h2"}}),ct=new J({props:{name:"class transformers.NezhaForQuestionAnswering",anchor:"transformers.NezhaForQuestionAnswering",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.NezhaForQuestionAnswering.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/nezha#transformers.NezhaConfig">NezhaConfig</a>) &#x2014; Model configuration class with all the parameters of the model.
Initializing with a config file does not load the weights associated with the model, only the
configuration. Check out the <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/nezha/modeling_nezha.py#L1589"}}),pt=new J({props:{name:"forward",anchor:"transformers.NezhaForQuestionAnswering.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"start_positions",val:": typing.Optional[torch.Tensor] = None"},{name:"end_positions",val:": typing.Optional[torch.Tensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.NezhaForQuestionAnswering.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.NezhaForQuestionAnswering.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.NezhaForQuestionAnswering.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.NezhaForQuestionAnswering.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.NezhaForQuestionAnswering.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.NezhaForQuestionAnswering.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.NezhaForQuestionAnswering.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.NezhaForQuestionAnswering.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.NezhaForQuestionAnswering.forward.start_positions",description:`<strong>start_positions</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for position (index) of the start of the labelled span for computing the token classification loss.
Positions are clamped to the length of the sequence (<code>sequence_length</code>). Position outside of the sequence
are not taken into account for computing the loss.`,name:"start_positions"},{anchor:"transformers.NezhaForQuestionAnswering.forward.end_positions",description:`<strong>end_positions</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for position (index) of the end of the labelled span for computing the token classification loss.
Positions are clamped to the length of the sequence (<code>sequence_length</code>). Position outside of the sequence
are not taken into account for computing the loss.`,name:"end_positions"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deprecated/nezha/modeling_nezha.py#L1600",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.QuestionAnsweringModelOutput"
>transformers.modeling_outputs.QuestionAnsweringModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/nezha#transformers.NezhaConfig"
>NezhaConfig</a>) and inputs.</p>
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
`}}),Ue=new pe({props:{$$slots:{default:[Ys]},$$scope:{ctx:z}}}),Ce=new de({props:{anchor:"transformers.NezhaForQuestionAnswering.forward.example",$$slots:{default:[As]},$$scope:{ctx:z}}}),ht=new Cs({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/nezha.md"}}),{c(){t=c("meta"),u=r(),o=c("p"),l=r(),y=c("p"),y.innerHTML=n,w=r(),f(xe.$$.fragment),At=r(),he=c("div"),he.innerHTML=Io,Ot=r(),f(me.$$.fragment),Dt=r(),f(Fe.$$.fragment),Kt=r(),Ze=c("p"),Ze.innerHTML=Go,en=r(),We=c("p"),We.textContent=Ho,tn=r(),Ie=c("p"),Ie.innerHTML=Vo,nn=r(),Ge=c("p"),Ge.innerHTML=Bo,on=r(),f(He.$$.fragment),sn=r(),Ve=c("ul"),Ve.innerHTML=qo,an=r(),f(Be.$$.fragment),rn=r(),H=c("div"),f(qe.$$.fragment),Nn=r(),mt=c("p"),mt.innerHTML=So,jn=r(),ut=c("p"),ut.innerHTML=Xo,Jn=r(),f(ue.$$.fragment),ln=r(),f(Se.$$.fragment),dn=r(),v=c("div"),f(Xe.$$.fragment),Un=r(),ft=c("p"),ft.textContent=Ro,Cn=r(),gt=c("p"),gt.innerHTML=Eo,xn=r(),_t=c("p"),_t.innerHTML=Lo,Fn=r(),bt=c("p"),bt.innerHTML=Po,Zn=r(),Mt=c("p"),Mt.innerHTML=Qo,Wn=r(),P=c("div"),f(Re.$$.fragment),In=r(),Tt=c("p"),Tt.innerHTML=Yo,Gn=r(),f(fe.$$.fragment),Hn=r(),f(ge.$$.fragment),cn=r(),f(Ee.$$.fragment),pn=r(),U=c("div"),f(Le.$$.fragment),Vn=r(),yt=c("p"),yt.innerHTML=Ao,Bn=r(),wt=c("p"),wt.innerHTML=Oo,qn=r(),zt=c("p"),zt.innerHTML=Do,Sn=r(),Q=c("div"),f(Pe.$$.fragment),Xn=r(),kt=c("p"),kt.innerHTML=Ko,Rn=r(),f(_e.$$.fragment),En=r(),f(be.$$.fragment),hn=r(),f(Qe.$$.fragment),mn=r(),C=c("div"),f(Ye.$$.fragment),Ln=r(),vt=c("p"),vt.innerHTML=es,Pn=r(),$t=c("p"),$t.innerHTML=ts,Qn=r(),Nt=c("p"),Nt.innerHTML=ns,Yn=r(),Y=c("div"),f(Ae.$$.fragment),An=r(),jt=c("p"),jt.innerHTML=os,On=r(),f(Me.$$.fragment),Dn=r(),f(Te.$$.fragment),un=r(),f(Oe.$$.fragment),fn=r(),x=c("div"),f(De.$$.fragment),Kn=r(),Jt=c("p"),Jt.innerHTML=ss,eo=r(),Ut=c("p"),Ut.innerHTML=as,to=r(),Ct=c("p"),Ct.innerHTML=rs,no=r(),A=c("div"),f(Ke.$$.fragment),oo=r(),xt=c("p"),xt.innerHTML=is,so=r(),f(ye.$$.fragment),ao=r(),f(we.$$.fragment),gn=r(),f(et.$$.fragment),_n=r(),F=c("div"),f(tt.$$.fragment),ro=r(),Ft=c("p"),Ft.textContent=ls,io=r(),Zt=c("p"),Zt.innerHTML=ds,lo=r(),Wt=c("p"),Wt.innerHTML=cs,co=r(),G=c("div"),f(nt.$$.fragment),po=r(),It=c("p"),It.innerHTML=ps,ho=r(),f(ze.$$.fragment),mo=r(),f(ke.$$.fragment),uo=r(),f(ve.$$.fragment),bn=r(),f(ot.$$.fragment),Mn=r(),Z=c("div"),f(st.$$.fragment),fo=r(),Gt=c("p"),Gt.textContent=hs,go=r(),Ht=c("p"),Ht.innerHTML=ms,_o=r(),Vt=c("p"),Vt.innerHTML=us,bo=r(),O=c("div"),f(at.$$.fragment),Mo=r(),Bt=c("p"),Bt.innerHTML=fs,To=r(),f($e.$$.fragment),yo=r(),f(Ne.$$.fragment),Tn=r(),f(rt.$$.fragment),yn=r(),W=c("div"),f(it.$$.fragment),wo=r(),qt=c("p"),qt.textContent=gs,zo=r(),St=c("p"),St.innerHTML=_s,ko=r(),Xt=c("p"),Xt.innerHTML=bs,vo=r(),D=c("div"),f(lt.$$.fragment),$o=r(),Rt=c("p"),Rt.innerHTML=Ms,No=r(),f(je.$$.fragment),jo=r(),f(Je.$$.fragment),wn=r(),f(dt.$$.fragment),zn=r(),I=c("div"),f(ct.$$.fragment),Jo=r(),Et=c("p"),Et.innerHTML=Ts,Uo=r(),Lt=c("p"),Lt.innerHTML=ys,Co=r(),Pt=c("p"),Pt.innerHTML=ws,xo=r(),K=c("div"),f(pt.$$.fragment),Fo=r(),Qt=c("p"),Qt.innerHTML=zs,Zo=r(),f(Ue.$$.fragment),Wo=r(),f(Ce.$$.fragment),kn=r(),f(ht.$$.fragment),vn=r(),Yt=c("p"),this.h()},l(e){const a=Us("svelte-u9bgzb",document.head);t=p(a,"META",{name:!0,content:!0}),a.forEach(d),u=i(e),o=p(e,"P",{}),$(o).forEach(d),l=i(e),y=p(e,"P",{"data-svelte-h":!0}),m(y)!=="svelte-1ow9mu1"&&(y.innerHTML=n),w=i(e),g(xe.$$.fragment,e),At=i(e),he=p(e,"DIV",{class:!0,"data-svelte-h":!0}),m(he)!=="svelte-13t8s2t"&&(he.innerHTML=Io),Ot=i(e),g(me.$$.fragment,e),Dt=i(e),g(Fe.$$.fragment,e),Kt=i(e),Ze=p(e,"P",{"data-svelte-h":!0}),m(Ze)!=="svelte-7llj89"&&(Ze.innerHTML=Go),en=i(e),We=p(e,"P",{"data-svelte-h":!0}),m(We)!=="svelte-vfdo9a"&&(We.textContent=Ho),tn=i(e),Ie=p(e,"P",{"data-svelte-h":!0}),m(Ie)!=="svelte-abk8uh"&&(Ie.innerHTML=Vo),nn=i(e),Ge=p(e,"P",{"data-svelte-h":!0}),m(Ge)!=="svelte-gkcmpf"&&(Ge.innerHTML=Bo),on=i(e),g(He.$$.fragment,e),sn=i(e),Ve=p(e,"UL",{"data-svelte-h":!0}),m(Ve)!=="svelte-mgusi3"&&(Ve.innerHTML=qo),an=i(e),g(Be.$$.fragment,e),rn=i(e),H=p(e,"DIV",{class:!0});var te=$(H);g(qe.$$.fragment,te),Nn=i(te),mt=p(te,"P",{"data-svelte-h":!0}),m(mt)!=="svelte-orqh8x"&&(mt.innerHTML=So),jn=i(te),ut=p(te,"P",{"data-svelte-h":!0}),m(ut)!=="svelte-1ek1ss9"&&(ut.innerHTML=Xo),Jn=i(te),g(ue.$$.fragment,te),te.forEach(d),ln=i(e),g(Se.$$.fragment,e),dn=i(e),v=p(e,"DIV",{class:!0});var j=$(v);g(Xe.$$.fragment,j),Un=i(j),ft=p(j,"P",{"data-svelte-h":!0}),m(ft)!=="svelte-137xuhh"&&(ft.textContent=Ro),Cn=i(j),gt=p(j,"P",{"data-svelte-h":!0}),m(gt)!=="svelte-q52n56"&&(gt.innerHTML=Eo),xn=i(j),_t=p(j,"P",{"data-svelte-h":!0}),m(_t)!=="svelte-hswkmf"&&(_t.innerHTML=Lo),Fn=i(j),bt=p(j,"P",{"data-svelte-h":!0}),m(bt)!=="svelte-1854dma"&&(bt.innerHTML=Po),Zn=i(j),Mt=p(j,"P",{"data-svelte-h":!0}),m(Mt)!=="svelte-174erte"&&(Mt.innerHTML=Qo),Wn=i(j),P=p(j,"DIV",{class:!0});var ne=$(P);g(Re.$$.fragment,ne),In=i(ne),Tt=p(ne,"P",{"data-svelte-h":!0}),m(Tt)!=="svelte-1nft62j"&&(Tt.innerHTML=Yo),Gn=i(ne),g(fe.$$.fragment,ne),Hn=i(ne),g(ge.$$.fragment,ne),ne.forEach(d),j.forEach(d),cn=i(e),g(Ee.$$.fragment,e),pn=i(e),U=p(e,"DIV",{class:!0});var V=$(U);g(Le.$$.fragment,V),Vn=i(V),yt=p(V,"P",{"data-svelte-h":!0}),m(yt)!=="svelte-18yrhh4"&&(yt.innerHTML=Ao),Bn=i(V),wt=p(V,"P",{"data-svelte-h":!0}),m(wt)!=="svelte-q52n56"&&(wt.innerHTML=Oo),qn=i(V),zt=p(V,"P",{"data-svelte-h":!0}),m(zt)!=="svelte-hswkmf"&&(zt.innerHTML=Do),Sn=i(V),Q=p(V,"DIV",{class:!0});var oe=$(Q);g(Pe.$$.fragment,oe),Xn=i(oe),kt=p(oe,"P",{"data-svelte-h":!0}),m(kt)!=="svelte-1hxsoix"&&(kt.innerHTML=Ko),Rn=i(oe),g(_e.$$.fragment,oe),En=i(oe),g(be.$$.fragment,oe),oe.forEach(d),V.forEach(d),hn=i(e),g(Qe.$$.fragment,e),mn=i(e),C=p(e,"DIV",{class:!0});var B=$(C);g(Ye.$$.fragment,B),Ln=i(B),vt=p(B,"P",{"data-svelte-h":!0}),m(vt)!=="svelte-1lnby5k"&&(vt.innerHTML=es),Pn=i(B),$t=p(B,"P",{"data-svelte-h":!0}),m($t)!=="svelte-q52n56"&&($t.innerHTML=ts),Qn=i(B),Nt=p(B,"P",{"data-svelte-h":!0}),m(Nt)!=="svelte-hswkmf"&&(Nt.innerHTML=ns),Yn=i(B),Y=p(B,"DIV",{class:!0});var se=$(Y);g(Ae.$$.fragment,se),An=i(se),jt=p(se,"P",{"data-svelte-h":!0}),m(jt)!=="svelte-1ry7nxb"&&(jt.innerHTML=os),On=i(se),g(Me.$$.fragment,se),Dn=i(se),g(Te.$$.fragment,se),se.forEach(d),B.forEach(d),un=i(e),g(Oe.$$.fragment,e),fn=i(e),x=p(e,"DIV",{class:!0});var q=$(x);g(De.$$.fragment,q),Kn=i(q),Jt=p(q,"P",{"data-svelte-h":!0}),m(Jt)!=="svelte-1sv5e2l"&&(Jt.innerHTML=ss),eo=i(q),Ut=p(q,"P",{"data-svelte-h":!0}),m(Ut)!=="svelte-q52n56"&&(Ut.innerHTML=as),to=i(q),Ct=p(q,"P",{"data-svelte-h":!0}),m(Ct)!=="svelte-hswkmf"&&(Ct.innerHTML=rs),no=i(q),A=p(q,"DIV",{class:!0});var ae=$(A);g(Ke.$$.fragment,ae),oo=i(ae),xt=p(ae,"P",{"data-svelte-h":!0}),m(xt)!=="svelte-2yozmx"&&(xt.innerHTML=is),so=i(ae),g(ye.$$.fragment,ae),ao=i(ae),g(we.$$.fragment,ae),ae.forEach(d),q.forEach(d),gn=i(e),g(et.$$.fragment,e),_n=i(e),F=p(e,"DIV",{class:!0});var S=$(F);g(tt.$$.fragment,S),ro=i(S),Ft=p(S,"P",{"data-svelte-h":!0}),m(Ft)!=="svelte-10n5i7s"&&(Ft.textContent=ls),io=i(S),Zt=p(S,"P",{"data-svelte-h":!0}),m(Zt)!=="svelte-q52n56"&&(Zt.innerHTML=ds),lo=i(S),Wt=p(S,"P",{"data-svelte-h":!0}),m(Wt)!=="svelte-hswkmf"&&(Wt.innerHTML=cs),co=i(S),G=p(S,"DIV",{class:!0});var X=$(G);g(nt.$$.fragment,X),po=i(X),It=p(X,"P",{"data-svelte-h":!0}),m(It)!=="svelte-t5dlch"&&(It.innerHTML=ps),ho=i(X),g(ze.$$.fragment,X),mo=i(X),g(ke.$$.fragment,X),uo=i(X),g(ve.$$.fragment,X),X.forEach(d),S.forEach(d),bn=i(e),g(ot.$$.fragment,e),Mn=i(e),Z=p(e,"DIV",{class:!0});var R=$(Z);g(st.$$.fragment,R),fo=i(R),Gt=p(R,"P",{"data-svelte-h":!0}),m(Gt)!=="svelte-ksq0kk"&&(Gt.textContent=hs),go=i(R),Ht=p(R,"P",{"data-svelte-h":!0}),m(Ht)!=="svelte-q52n56"&&(Ht.innerHTML=ms),_o=i(R),Vt=p(R,"P",{"data-svelte-h":!0}),m(Vt)!=="svelte-hswkmf"&&(Vt.innerHTML=us),bo=i(R),O=p(R,"DIV",{class:!0});var re=$(O);g(at.$$.fragment,re),Mo=i(re),Bt=p(re,"P",{"data-svelte-h":!0}),m(Bt)!=="svelte-1j1k5ml"&&(Bt.innerHTML=fs),To=i(re),g($e.$$.fragment,re),yo=i(re),g(Ne.$$.fragment,re),re.forEach(d),R.forEach(d),Tn=i(e),g(rt.$$.fragment,e),yn=i(e),W=p(e,"DIV",{class:!0});var E=$(W);g(it.$$.fragment,E),wo=i(E),qt=p(E,"P",{"data-svelte-h":!0}),m(qt)!=="svelte-1hgqgbl"&&(qt.textContent=gs),zo=i(E),St=p(E,"P",{"data-svelte-h":!0}),m(St)!=="svelte-q52n56"&&(St.innerHTML=_s),ko=i(E),Xt=p(E,"P",{"data-svelte-h":!0}),m(Xt)!=="svelte-hswkmf"&&(Xt.innerHTML=bs),vo=i(E),D=p(E,"DIV",{class:!0});var ie=$(D);g(lt.$$.fragment,ie),$o=i(ie),Rt=p(ie,"P",{"data-svelte-h":!0}),m(Rt)!=="svelte-pw3b55"&&(Rt.innerHTML=Ms),No=i(ie),g(je.$$.fragment,ie),jo=i(ie),g(Je.$$.fragment,ie),ie.forEach(d),E.forEach(d),wn=i(e),g(dt.$$.fragment,e),zn=i(e),I=p(e,"DIV",{class:!0});var L=$(I);g(ct.$$.fragment,L),Jo=i(L),Et=p(L,"P",{"data-svelte-h":!0}),m(Et)!=="svelte-1ge7w0v"&&(Et.innerHTML=Ts),Uo=i(L),Lt=p(L,"P",{"data-svelte-h":!0}),m(Lt)!=="svelte-q52n56"&&(Lt.innerHTML=ys),Co=i(L),Pt=p(L,"P",{"data-svelte-h":!0}),m(Pt)!=="svelte-hswkmf"&&(Pt.innerHTML=ws),xo=i(L),K=p(L,"DIV",{class:!0});var le=$(K);g(pt.$$.fragment,le),Fo=i(le),Qt=p(le,"P",{"data-svelte-h":!0}),m(Qt)!=="svelte-jb4n3x"&&(Qt.innerHTML=zs),Zo=i(le),g(Ue.$$.fragment,le),Wo=i(le),g(Ce.$$.fragment,le),le.forEach(d),L.forEach(d),kn=i(e),g(ht.$$.fragment,e),vn=i(e),Yt=p(e,"P",{}),$(Yt).forEach(d),this.h()},h(){k(t,"name","hf:doc:metadata"),k(t,"content",Ds),k(he,"class","flex flex-wrap space-x-1"),k(H,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),k(P,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),k(v,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),k(Q,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),k(U,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),k(Y,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),k(C,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),k(A,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),k(x,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),k(G,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),k(F,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),k(O,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),k(Z,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),k(D,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),k(W,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),k(K,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),k(I,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,a){s(document.head,t),h(e,u,a),h(e,o,a),h(e,l,a),h(e,y,a),h(e,w,a),_(xe,e,a),h(e,At,a),h(e,he,a),h(e,Ot,a),_(me,e,a),h(e,Dt,a),_(Fe,e,a),h(e,Kt,a),h(e,Ze,a),h(e,en,a),h(e,We,a),h(e,tn,a),h(e,Ie,a),h(e,nn,a),h(e,Ge,a),h(e,on,a),_(He,e,a),h(e,sn,a),h(e,Ve,a),h(e,an,a),_(Be,e,a),h(e,rn,a),h(e,H,a),_(qe,H,null),s(H,Nn),s(H,mt),s(H,jn),s(H,ut),s(H,Jn),_(ue,H,null),h(e,ln,a),_(Se,e,a),h(e,dn,a),h(e,v,a),_(Xe,v,null),s(v,Un),s(v,ft),s(v,Cn),s(v,gt),s(v,xn),s(v,_t),s(v,Fn),s(v,bt),s(v,Zn),s(v,Mt),s(v,Wn),s(v,P),_(Re,P,null),s(P,In),s(P,Tt),s(P,Gn),_(fe,P,null),s(P,Hn),_(ge,P,null),h(e,cn,a),_(Ee,e,a),h(e,pn,a),h(e,U,a),_(Le,U,null),s(U,Vn),s(U,yt),s(U,Bn),s(U,wt),s(U,qn),s(U,zt),s(U,Sn),s(U,Q),_(Pe,Q,null),s(Q,Xn),s(Q,kt),s(Q,Rn),_(_e,Q,null),s(Q,En),_(be,Q,null),h(e,hn,a),_(Qe,e,a),h(e,mn,a),h(e,C,a),_(Ye,C,null),s(C,Ln),s(C,vt),s(C,Pn),s(C,$t),s(C,Qn),s(C,Nt),s(C,Yn),s(C,Y),_(Ae,Y,null),s(Y,An),s(Y,jt),s(Y,On),_(Me,Y,null),s(Y,Dn),_(Te,Y,null),h(e,un,a),_(Oe,e,a),h(e,fn,a),h(e,x,a),_(De,x,null),s(x,Kn),s(x,Jt),s(x,eo),s(x,Ut),s(x,to),s(x,Ct),s(x,no),s(x,A),_(Ke,A,null),s(A,oo),s(A,xt),s(A,so),_(ye,A,null),s(A,ao),_(we,A,null),h(e,gn,a),_(et,e,a),h(e,_n,a),h(e,F,a),_(tt,F,null),s(F,ro),s(F,Ft),s(F,io),s(F,Zt),s(F,lo),s(F,Wt),s(F,co),s(F,G),_(nt,G,null),s(G,po),s(G,It),s(G,ho),_(ze,G,null),s(G,mo),_(ke,G,null),s(G,uo),_(ve,G,null),h(e,bn,a),_(ot,e,a),h(e,Mn,a),h(e,Z,a),_(st,Z,null),s(Z,fo),s(Z,Gt),s(Z,go),s(Z,Ht),s(Z,_o),s(Z,Vt),s(Z,bo),s(Z,O),_(at,O,null),s(O,Mo),s(O,Bt),s(O,To),_($e,O,null),s(O,yo),_(Ne,O,null),h(e,Tn,a),_(rt,e,a),h(e,yn,a),h(e,W,a),_(it,W,null),s(W,wo),s(W,qt),s(W,zo),s(W,St),s(W,ko),s(W,Xt),s(W,vo),s(W,D),_(lt,D,null),s(D,$o),s(D,Rt),s(D,No),_(je,D,null),s(D,jo),_(Je,D,null),h(e,wn,a),_(dt,e,a),h(e,zn,a),h(e,I,a),_(ct,I,null),s(I,Jo),s(I,Et),s(I,Uo),s(I,Lt),s(I,Co),s(I,Pt),s(I,xo),s(I,K),_(pt,K,null),s(K,Fo),s(K,Qt),s(K,Zo),_(Ue,K,null),s(K,Wo),_(Ce,K,null),h(e,kn,a),_(ht,e,a),h(e,vn,a),h(e,Yt,a),$n=!0},p(e,[a]){const te={};a&2&&(te.$$scope={dirty:a,ctx:e}),me.$set(te);const j={};a&2&&(j.$$scope={dirty:a,ctx:e}),ue.$set(j);const ne={};a&2&&(ne.$$scope={dirty:a,ctx:e}),fe.$set(ne);const V={};a&2&&(V.$$scope={dirty:a,ctx:e}),ge.$set(V);const oe={};a&2&&(oe.$$scope={dirty:a,ctx:e}),_e.$set(oe);const B={};a&2&&(B.$$scope={dirty:a,ctx:e}),be.$set(B);const se={};a&2&&(se.$$scope={dirty:a,ctx:e}),Me.$set(se);const q={};a&2&&(q.$$scope={dirty:a,ctx:e}),Te.$set(q);const ae={};a&2&&(ae.$$scope={dirty:a,ctx:e}),ye.$set(ae);const S={};a&2&&(S.$$scope={dirty:a,ctx:e}),we.$set(S);const X={};a&2&&(X.$$scope={dirty:a,ctx:e}),ze.$set(X);const R={};a&2&&(R.$$scope={dirty:a,ctx:e}),ke.$set(R);const re={};a&2&&(re.$$scope={dirty:a,ctx:e}),ve.$set(re);const E={};a&2&&(E.$$scope={dirty:a,ctx:e}),$e.$set(E);const ie={};a&2&&(ie.$$scope={dirty:a,ctx:e}),Ne.$set(ie);const L={};a&2&&(L.$$scope={dirty:a,ctx:e}),je.$set(L);const le={};a&2&&(le.$$scope={dirty:a,ctx:e}),Je.$set(le);const ks={};a&2&&(ks.$$scope={dirty:a,ctx:e}),Ue.$set(ks);const vs={};a&2&&(vs.$$scope={dirty:a,ctx:e}),Ce.$set(vs)},i(e){$n||(b(xe.$$.fragment,e),b(me.$$.fragment,e),b(Fe.$$.fragment,e),b(He.$$.fragment,e),b(Be.$$.fragment,e),b(qe.$$.fragment,e),b(ue.$$.fragment,e),b(Se.$$.fragment,e),b(Xe.$$.fragment,e),b(Re.$$.fragment,e),b(fe.$$.fragment,e),b(ge.$$.fragment,e),b(Ee.$$.fragment,e),b(Le.$$.fragment,e),b(Pe.$$.fragment,e),b(_e.$$.fragment,e),b(be.$$.fragment,e),b(Qe.$$.fragment,e),b(Ye.$$.fragment,e),b(Ae.$$.fragment,e),b(Me.$$.fragment,e),b(Te.$$.fragment,e),b(Oe.$$.fragment,e),b(De.$$.fragment,e),b(Ke.$$.fragment,e),b(ye.$$.fragment,e),b(we.$$.fragment,e),b(et.$$.fragment,e),b(tt.$$.fragment,e),b(nt.$$.fragment,e),b(ze.$$.fragment,e),b(ke.$$.fragment,e),b(ve.$$.fragment,e),b(ot.$$.fragment,e),b(st.$$.fragment,e),b(at.$$.fragment,e),b($e.$$.fragment,e),b(Ne.$$.fragment,e),b(rt.$$.fragment,e),b(it.$$.fragment,e),b(lt.$$.fragment,e),b(je.$$.fragment,e),b(Je.$$.fragment,e),b(dt.$$.fragment,e),b(ct.$$.fragment,e),b(pt.$$.fragment,e),b(Ue.$$.fragment,e),b(Ce.$$.fragment,e),b(ht.$$.fragment,e),$n=!0)},o(e){M(xe.$$.fragment,e),M(me.$$.fragment,e),M(Fe.$$.fragment,e),M(He.$$.fragment,e),M(Be.$$.fragment,e),M(qe.$$.fragment,e),M(ue.$$.fragment,e),M(Se.$$.fragment,e),M(Xe.$$.fragment,e),M(Re.$$.fragment,e),M(fe.$$.fragment,e),M(ge.$$.fragment,e),M(Ee.$$.fragment,e),M(Le.$$.fragment,e),M(Pe.$$.fragment,e),M(_e.$$.fragment,e),M(be.$$.fragment,e),M(Qe.$$.fragment,e),M(Ye.$$.fragment,e),M(Ae.$$.fragment,e),M(Me.$$.fragment,e),M(Te.$$.fragment,e),M(Oe.$$.fragment,e),M(De.$$.fragment,e),M(Ke.$$.fragment,e),M(ye.$$.fragment,e),M(we.$$.fragment,e),M(et.$$.fragment,e),M(tt.$$.fragment,e),M(nt.$$.fragment,e),M(ze.$$.fragment,e),M(ke.$$.fragment,e),M(ve.$$.fragment,e),M(ot.$$.fragment,e),M(st.$$.fragment,e),M(at.$$.fragment,e),M($e.$$.fragment,e),M(Ne.$$.fragment,e),M(rt.$$.fragment,e),M(it.$$.fragment,e),M(lt.$$.fragment,e),M(je.$$.fragment,e),M(Je.$$.fragment,e),M(dt.$$.fragment,e),M(ct.$$.fragment,e),M(pt.$$.fragment,e),M(Ue.$$.fragment,e),M(Ce.$$.fragment,e),M(ht.$$.fragment,e),$n=!1},d(e){e&&(d(u),d(o),d(l),d(y),d(w),d(At),d(he),d(Ot),d(Dt),d(Kt),d(Ze),d(en),d(We),d(tn),d(Ie),d(nn),d(Ge),d(on),d(sn),d(Ve),d(an),d(rn),d(H),d(ln),d(dn),d(v),d(cn),d(pn),d(U),d(hn),d(mn),d(C),d(un),d(fn),d(x),d(gn),d(_n),d(F),d(bn),d(Mn),d(Z),d(Tn),d(yn),d(W),d(wn),d(zn),d(I),d(kn),d(vn),d(Yt)),d(t),T(xe,e),T(me,e),T(Fe,e),T(He,e),T(Be,e),T(qe),T(ue),T(Se,e),T(Xe),T(Re),T(fe),T(ge),T(Ee,e),T(Le),T(Pe),T(_e),T(be),T(Qe,e),T(Ye),T(Ae),T(Me),T(Te),T(Oe,e),T(De),T(Ke),T(ye),T(we),T(et,e),T(tt),T(nt),T(ze),T(ke),T(ve),T(ot,e),T(st),T(at),T($e),T(Ne),T(rt,e),T(it),T(lt),T(je),T(Je),T(dt,e),T(ct),T(pt),T(Ue),T(Ce),T(ht,e)}}}const Ds='{"title":"Nezha","local":"nezha","sections":[{"title":"Overview","local":"overview","sections":[],"depth":2},{"title":"Resources","local":"resources","sections":[],"depth":2},{"title":"NezhaConfig","local":"transformers.NezhaConfig","sections":[],"depth":2},{"title":"NezhaModel","local":"transformers.NezhaModel","sections":[],"depth":2},{"title":"NezhaForPreTraining","local":"transformers.NezhaForPreTraining","sections":[],"depth":2},{"title":"NezhaForMaskedLM","local":"transformers.NezhaForMaskedLM","sections":[],"depth":2},{"title":"NezhaForNextSentencePrediction","local":"transformers.NezhaForNextSentencePrediction","sections":[],"depth":2},{"title":"NezhaForSequenceClassification","local":"transformers.NezhaForSequenceClassification","sections":[],"depth":2},{"title":"NezhaForMultipleChoice","local":"transformers.NezhaForMultipleChoice","sections":[],"depth":2},{"title":"NezhaForTokenClassification","local":"transformers.NezhaForTokenClassification","sections":[],"depth":2},{"title":"NezhaForQuestionAnswering","local":"transformers.NezhaForQuestionAnswering","sections":[],"depth":2}],"depth":1}';function Ks(z){return Ns(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class ia extends js{constructor(t){super(),Js(this,t,Ks,Os,$s,{})}}export{ia as component};
