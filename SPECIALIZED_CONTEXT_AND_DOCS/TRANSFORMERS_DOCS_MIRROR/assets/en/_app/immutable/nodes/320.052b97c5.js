import{s as gn,o as yn,n as L}from"../chunks/scheduler.18a86fab.js";import{S as _n,i as bn,g as m,s as r,r as h,A as Mn,h as p,f as s,c as a,j as F,x as f,u,k as v,y as i,a as c,v as g,d as y,t as _,w as b}from"../chunks/index.98837b22.js";import{T as _t}from"../chunks/Tip.77304350.js";import{D as R}from"../chunks/Docstring.a1ef7999.js";import{C as Qe}from"../chunks/CodeBlock.8d0c2e8a.js";import{E as He}from"../chunks/ExampleCodeBlock.8c3ee1f9.js";import{H as K,E as Tn}from"../chunks/getInferenceSnippets.06c2775f.js";function wn(k){let t,M="Example:",l,d,T;return d=new Qe({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyME55c3Ryb21mb3JtZXJNb2RlbCUyQyUyME55c3Ryb21mb3JtZXJDb25maWclMEElMEElMjMlMjBJbml0aWFsaXppbmclMjBhJTIwTnlzdHJvbWZvcm1lciUyMHV3LW1hZGlzb24lMkZueXN0cm9tZm9ybWVyLTUxMiUyMHN0eWxlJTIwY29uZmlndXJhdGlvbiUwQWNvbmZpZ3VyYXRpb24lMjAlM0QlMjBOeXN0cm9tZm9ybWVyQ29uZmlnKCklMEElMEElMjMlMjBJbml0aWFsaXppbmclMjBhJTIwbW9kZWwlMjBmcm9tJTIwdGhlJTIwdXctbWFkaXNvbiUyRm55c3Ryb21mb3JtZXItNTEyJTIwc3R5bGUlMjBjb25maWd1cmF0aW9uJTBBbW9kZWwlMjAlM0QlMjBOeXN0cm9tZm9ybWVyTW9kZWwoY29uZmlndXJhdGlvbiklMEElMEElMjMlMjBBY2Nlc3NpbmclMjB0aGUlMjBtb2RlbCUyMGNvbmZpZ3VyYXRpb24lMEFjb25maWd1cmF0aW9uJTIwJTNEJTIwbW9kZWwuY29uZmln",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> NystromformerModel, NystromformerConfig

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a Nystromformer uw-madison/nystromformer-512 style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = NystromformerConfig()

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a model from the uw-madison/nystromformer-512 style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = NystromformerModel(configuration)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Accessing the model configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = model.config`,wrap:!1}}),{c(){t=m("p"),t.textContent=M,l=r(),h(d.$$.fragment)},l(o){t=p(o,"P",{"data-svelte-h":!0}),f(t)!=="svelte-11lpom8"&&(t.textContent=M),l=a(o),u(d.$$.fragment,o)},m(o,w){c(o,t,w),c(o,l,w),g(d,o,w),T=!0},p:L,i(o){T||(y(d.$$.fragment,o),T=!0)},o(o){_(d.$$.fragment,o),T=!1},d(o){o&&(s(t),s(l)),b(d,o)}}}function kn(k){let t,M=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=m("p"),t.innerHTML=M},l(l){t=p(l,"P",{"data-svelte-h":!0}),f(t)!=="svelte-fincs2"&&(t.innerHTML=M)},m(l,d){c(l,t,d)},p:L,d(l){l&&s(t)}}}function vn(k){let t,M=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=m("p"),t.innerHTML=M},l(l){t=p(l,"P",{"data-svelte-h":!0}),f(t)!=="svelte-fincs2"&&(t.innerHTML=M)},m(l,d){c(l,t,d)},p:L,d(l){l&&s(t)}}}function Nn(k){let t,M="Example:",l,d,T;return d=new Qe({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBOeXN0cm9tZm9ybWVyRm9yTWFza2VkTE0lMEFpbXBvcnQlMjB0b3JjaCUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMnV3LW1hZGlzb24lMkZueXN0cm9tZm9ybWVyLTUxMiUyMiklMEFtb2RlbCUyMCUzRCUyME55c3Ryb21mb3JtZXJGb3JNYXNrZWRMTS5mcm9tX3ByZXRyYWluZWQoJTIydXctbWFkaXNvbiUyRm55c3Ryb21mb3JtZXItNTEyJTIyKSUwQSUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplciglMjJUaGUlMjBjYXBpdGFsJTIwb2YlMjBGcmFuY2UlMjBpcyUyMCUzQ21hc2slM0UuJTIyJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiklMEElMEF3aXRoJTIwdG9yY2gubm9fZ3JhZCgpJTNBJTBBJTIwJTIwJTIwJTIwbG9naXRzJTIwJTNEJTIwbW9kZWwoKippbnB1dHMpLmxvZ2l0cyUwQSUwQSUyMyUyMHJldHJpZXZlJTIwaW5kZXglMjBvZiUyMCUzQ21hc2slM0UlMEFtYXNrX3Rva2VuX2luZGV4JTIwJTNEJTIwKGlucHV0cy5pbnB1dF9pZHMlMjAlM0QlM0QlMjB0b2tlbml6ZXIubWFza190b2tlbl9pZCklNUIwJTVELm5vbnplcm8oYXNfdHVwbGUlM0RUcnVlKSU1QjAlNUQlMEElMEFwcmVkaWN0ZWRfdG9rZW5faWQlMjAlM0QlMjBsb2dpdHMlNUIwJTJDJTIwbWFza190b2tlbl9pbmRleCU1RC5hcmdtYXgoYXhpcyUzRC0xKSUwQXRva2VuaXplci5kZWNvZGUocHJlZGljdGVkX3Rva2VuX2lkKSUwQSUwQWxhYmVscyUyMCUzRCUyMHRva2VuaXplciglMjJUaGUlMjBjYXBpdGFsJTIwb2YlMjBGcmFuY2UlMjBpcyUyMFBhcmlzLiUyMiUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpJTVCJTIyaW5wdXRfaWRzJTIyJTVEJTBBJTIzJTIwbWFzayUyMGxhYmVscyUyMG9mJTIwbm9uLSUzQ21hc2slM0UlMjB0b2tlbnMlMEFsYWJlbHMlMjAlM0QlMjB0b3JjaC53aGVyZShpbnB1dHMuaW5wdXRfaWRzJTIwJTNEJTNEJTIwdG9rZW5pemVyLm1hc2tfdG9rZW5faWQlMkMlMjBsYWJlbHMlMkMlMjAtMTAwKSUwQSUwQW91dHB1dHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyUyQyUyMGxhYmVscyUzRGxhYmVscyklMEFyb3VuZChvdXRwdXRzLmxvc3MuaXRlbSgpJTJDJTIwMik=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, NystromformerForMaskedLM
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;uw-madison/nystromformer-512&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = NystromformerForMaskedLM.from_pretrained(<span class="hljs-string">&quot;uw-madison/nystromformer-512&quot;</span>)

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
...`,wrap:!1}}),{c(){t=m("p"),t.textContent=M,l=r(),h(d.$$.fragment)},l(o){t=p(o,"P",{"data-svelte-h":!0}),f(t)!=="svelte-11lpom8"&&(t.textContent=M),l=a(o),u(d.$$.fragment,o)},m(o,w){c(o,t,w),c(o,l,w),g(d,o,w),T=!0},p:L,i(o){T||(y(d.$$.fragment,o),T=!0)},o(o){_(d.$$.fragment,o),T=!1},d(o){o&&(s(t),s(l)),b(d,o)}}}function $n(k){let t,M=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=m("p"),t.innerHTML=M},l(l){t=p(l,"P",{"data-svelte-h":!0}),f(t)!=="svelte-fincs2"&&(t.innerHTML=M)},m(l,d){c(l,t,d)},p:L,d(l){l&&s(t)}}}function jn(k){let t,M="Example of single-label classification:",l,d,T;return d=new Qe({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Rva2VuaXplciUyQyUyME55c3Ryb21mb3JtZXJGb3JTZXF1ZW5jZUNsYXNzaWZpY2F0aW9uJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIydXctbWFkaXNvbiUyRm55c3Ryb21mb3JtZXItNTEyJTIyKSUwQW1vZGVsJTIwJTNEJTIwTnlzdHJvbWZvcm1lckZvclNlcXVlbmNlQ2xhc3NpZmljYXRpb24uZnJvbV9wcmV0cmFpbmVkKCUyMnV3LW1hZGlzb24lMkZueXN0cm9tZm9ybWVyLTUxMiUyMiklMEElMEFpbnB1dHMlMjAlM0QlMjB0b2tlbml6ZXIoJTIySGVsbG8lMkMlMjBteSUyMGRvZyUyMGlzJTIwY3V0ZSUyMiUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpJTBBJTBBd2l0aCUyMHRvcmNoLm5vX2dyYWQoKSUzQSUwQSUyMCUyMCUyMCUyMGxvZ2l0cyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzKS5sb2dpdHMlMEElMEFwcmVkaWN0ZWRfY2xhc3NfaWQlMjAlM0QlMjBsb2dpdHMuYXJnbWF4KCkuaXRlbSgpJTBBbW9kZWwuY29uZmlnLmlkMmxhYmVsJTVCcHJlZGljdGVkX2NsYXNzX2lkJTVEJTBBJTBBJTIzJTIwVG8lMjB0cmFpbiUyMGElMjBtb2RlbCUyMG9uJTIwJTYwbnVtX2xhYmVscyU2MCUyMGNsYXNzZXMlMkMlMjB5b3UlMjBjYW4lMjBwYXNzJTIwJTYwbnVtX2xhYmVscyUzRG51bV9sYWJlbHMlNjAlMjB0byUyMCU2MC5mcm9tX3ByZXRyYWluZWQoLi4uKSU2MCUwQW51bV9sYWJlbHMlMjAlM0QlMjBsZW4obW9kZWwuY29uZmlnLmlkMmxhYmVsKSUwQW1vZGVsJTIwJTNEJTIwTnlzdHJvbWZvcm1lckZvclNlcXVlbmNlQ2xhc3NpZmljYXRpb24uZnJvbV9wcmV0cmFpbmVkKCUyMnV3LW1hZGlzb24lMkZueXN0cm9tZm9ybWVyLTUxMiUyMiUyQyUyMG51bV9sYWJlbHMlM0RudW1fbGFiZWxzKSUwQSUwQWxhYmVscyUyMCUzRCUyMHRvcmNoLnRlbnNvciglNUIxJTVEKSUwQWxvc3MlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyUyQyUyMGxhYmVscyUzRGxhYmVscykubG9zcyUwQXJvdW5kKGxvc3MuaXRlbSgpJTJDJTIwMik=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, NystromformerForSequenceClassification

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;uw-madison/nystromformer-512&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = NystromformerForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;uw-madison/nystromformer-512&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    logits = model(**inputs).logits

<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_class_id = logits.argmax().item()
<span class="hljs-meta">&gt;&gt;&gt; </span>model.config.id2label[predicted_class_id]
...

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># To train a model on \`num_labels\` classes, you can pass \`num_labels=num_labels\` to \`.from_pretrained(...)\`</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>num_labels = <span class="hljs-built_in">len</span>(model.config.id2label)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = NystromformerForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;uw-madison/nystromformer-512&quot;</span>, num_labels=num_labels)

<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.tensor([<span class="hljs-number">1</span>])
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = model(**inputs, labels=labels).loss
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">round</span>(loss.item(), <span class="hljs-number">2</span>)
...`,wrap:!1}}),{c(){t=m("p"),t.textContent=M,l=r(),h(d.$$.fragment)},l(o){t=p(o,"P",{"data-svelte-h":!0}),f(t)!=="svelte-ykxpe4"&&(t.textContent=M),l=a(o),u(d.$$.fragment,o)},m(o,w){c(o,t,w),c(o,l,w),g(d,o,w),T=!0},p:L,i(o){T||(y(d.$$.fragment,o),T=!0)},o(o){_(d.$$.fragment,o),T=!1},d(o){o&&(s(t),s(l)),b(d,o)}}}function Jn(k){let t,M="Example of multi-label classification:",l,d,T;return d=new Qe({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Rva2VuaXplciUyQyUyME55c3Ryb21mb3JtZXJGb3JTZXF1ZW5jZUNsYXNzaWZpY2F0aW9uJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIydXctbWFkaXNvbiUyRm55c3Ryb21mb3JtZXItNTEyJTIyKSUwQW1vZGVsJTIwJTNEJTIwTnlzdHJvbWZvcm1lckZvclNlcXVlbmNlQ2xhc3NpZmljYXRpb24uZnJvbV9wcmV0cmFpbmVkKCUyMnV3LW1hZGlzb24lMkZueXN0cm9tZm9ybWVyLTUxMiUyMiUyQyUyMHByb2JsZW1fdHlwZSUzRCUyMm11bHRpX2xhYmVsX2NsYXNzaWZpY2F0aW9uJTIyKSUwQSUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplciglMjJIZWxsbyUyQyUyMG15JTIwZG9nJTIwaXMlMjBjdXRlJTIyJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiklMEElMEF3aXRoJTIwdG9yY2gubm9fZ3JhZCgpJTNBJTBBJTIwJTIwJTIwJTIwbG9naXRzJTIwJTNEJTIwbW9kZWwoKippbnB1dHMpLmxvZ2l0cyUwQSUwQXByZWRpY3RlZF9jbGFzc19pZHMlMjAlM0QlMjB0b3JjaC5hcmFuZ2UoMCUyQyUyMGxvZ2l0cy5zaGFwZSU1Qi0xJTVEKSU1QnRvcmNoLnNpZ21vaWQobG9naXRzKS5zcXVlZXplKGRpbSUzRDApJTIwJTNFJTIwMC41JTVEJTBBJTBBJTIzJTIwVG8lMjB0cmFpbiUyMGElMjBtb2RlbCUyMG9uJTIwJTYwbnVtX2xhYmVscyU2MCUyMGNsYXNzZXMlMkMlMjB5b3UlMjBjYW4lMjBwYXNzJTIwJTYwbnVtX2xhYmVscyUzRG51bV9sYWJlbHMlNjAlMjB0byUyMCU2MC5mcm9tX3ByZXRyYWluZWQoLi4uKSU2MCUwQW51bV9sYWJlbHMlMjAlM0QlMjBsZW4obW9kZWwuY29uZmlnLmlkMmxhYmVsKSUwQW1vZGVsJTIwJTNEJTIwTnlzdHJvbWZvcm1lckZvclNlcXVlbmNlQ2xhc3NpZmljYXRpb24uZnJvbV9wcmV0cmFpbmVkKCUwQSUyMCUyMCUyMCUyMCUyMnV3LW1hZGlzb24lMkZueXN0cm9tZm9ybWVyLTUxMiUyMiUyQyUyMG51bV9sYWJlbHMlM0RudW1fbGFiZWxzJTJDJTIwcHJvYmxlbV90eXBlJTNEJTIybXVsdGlfbGFiZWxfY2xhc3NpZmljYXRpb24lMjIlMEEpJTBBJTBBbGFiZWxzJTIwJTNEJTIwdG9yY2guc3VtKCUwQSUyMCUyMCUyMCUyMHRvcmNoLm5uLmZ1bmN0aW9uYWwub25lX2hvdChwcmVkaWN0ZWRfY2xhc3NfaWRzJTVCTm9uZSUyQyUyMCUzQSU1RC5jbG9uZSgpJTJDJTIwbnVtX2NsYXNzZXMlM0RudW1fbGFiZWxzKSUyQyUyMGRpbSUzRDElMEEpLnRvKHRvcmNoLmZsb2F0KSUwQWxvc3MlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyUyQyUyMGxhYmVscyUzRGxhYmVscykubG9zcw==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, NystromformerForSequenceClassification

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;uw-madison/nystromformer-512&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = NystromformerForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;uw-madison/nystromformer-512&quot;</span>, problem_type=<span class="hljs-string">&quot;multi_label_classification&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    logits = model(**inputs).logits

<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_class_ids = torch.arange(<span class="hljs-number">0</span>, logits.shape[-<span class="hljs-number">1</span>])[torch.sigmoid(logits).squeeze(dim=<span class="hljs-number">0</span>) &gt; <span class="hljs-number">0.5</span>]

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># To train a model on \`num_labels\` classes, you can pass \`num_labels=num_labels\` to \`.from_pretrained(...)\`</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>num_labels = <span class="hljs-built_in">len</span>(model.config.id2label)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = NystromformerForSequenceClassification.from_pretrained(
<span class="hljs-meta">... </span>    <span class="hljs-string">&quot;uw-madison/nystromformer-512&quot;</span>, num_labels=num_labels, problem_type=<span class="hljs-string">&quot;multi_label_classification&quot;</span>
<span class="hljs-meta">... </span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.<span class="hljs-built_in">sum</span>(
<span class="hljs-meta">... </span>    torch.nn.functional.one_hot(predicted_class_ids[<span class="hljs-literal">None</span>, :].clone(), num_classes=num_labels), dim=<span class="hljs-number">1</span>
<span class="hljs-meta">... </span>).to(torch.<span class="hljs-built_in">float</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = model(**inputs, labels=labels).loss`,wrap:!1}}),{c(){t=m("p"),t.textContent=M,l=r(),h(d.$$.fragment)},l(o){t=p(o,"P",{"data-svelte-h":!0}),f(t)!=="svelte-1l8e32d"&&(t.textContent=M),l=a(o),u(d.$$.fragment,o)},m(o,w){c(o,t,w),c(o,l,w),g(d,o,w),T=!0},p:L,i(o){T||(y(d.$$.fragment,o),T=!0)},o(o){_(d.$$.fragment,o),T=!1},d(o){o&&(s(t),s(l)),b(d,o)}}}function Un(k){let t,M=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=m("p"),t.innerHTML=M},l(l){t=p(l,"P",{"data-svelte-h":!0}),f(t)!=="svelte-fincs2"&&(t.innerHTML=M)},m(l,d){c(l,t,d)},p:L,d(l){l&&s(t)}}}function Cn(k){let t,M="Example:",l,d,T;return d=new Qe({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBOeXN0cm9tZm9ybWVyRm9yTXVsdGlwbGVDaG9pY2UlMEFpbXBvcnQlMjB0b3JjaCUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMnV3LW1hZGlzb24lMkZueXN0cm9tZm9ybWVyLTUxMiUyMiklMEFtb2RlbCUyMCUzRCUyME55c3Ryb21mb3JtZXJGb3JNdWx0aXBsZUNob2ljZS5mcm9tX3ByZXRyYWluZWQoJTIydXctbWFkaXNvbiUyRm55c3Ryb21mb3JtZXItNTEyJTIyKSUwQSUwQXByb21wdCUyMCUzRCUyMCUyMkluJTIwSXRhbHklMkMlMjBwaXp6YSUyMHNlcnZlZCUyMGluJTIwZm9ybWFsJTIwc2V0dGluZ3MlMkMlMjBzdWNoJTIwYXMlMjBhdCUyMGElMjByZXN0YXVyYW50JTJDJTIwaXMlMjBwcmVzZW50ZWQlMjB1bnNsaWNlZC4lMjIlMEFjaG9pY2UwJTIwJTNEJTIwJTIySXQlMjBpcyUyMGVhdGVuJTIwd2l0aCUyMGElMjBmb3JrJTIwYW5kJTIwYSUyMGtuaWZlLiUyMiUwQWNob2ljZTElMjAlM0QlMjAlMjJJdCUyMGlzJTIwZWF0ZW4lMjB3aGlsZSUyMGhlbGQlMjBpbiUyMHRoZSUyMGhhbmQuJTIyJTBBbGFiZWxzJTIwJTNEJTIwdG9yY2gudGVuc29yKDApLnVuc3F1ZWV6ZSgwKSUyMCUyMCUyMyUyMGNob2ljZTAlMjBpcyUyMGNvcnJlY3QlMjAoYWNjb3JkaW5nJTIwdG8lMjBXaWtpcGVkaWElMjAlM0IpKSUyQyUyMGJhdGNoJTIwc2l6ZSUyMDElMEElMEFlbmNvZGluZyUyMCUzRCUyMHRva2VuaXplciglNUJwcm9tcHQlMkMlMjBwcm9tcHQlNUQlMkMlMjAlNUJjaG9pY2UwJTJDJTIwY2hvaWNlMSU1RCUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIlMkMlMjBwYWRkaW5nJTNEVHJ1ZSklMEFvdXRwdXRzJTIwJTNEJTIwbW9kZWwoKiolN0JrJTNBJTIwdi51bnNxdWVlemUoMCklMjBmb3IlMjBrJTJDJTIwdiUyMGluJTIwZW5jb2RpbmcuaXRlbXMoKSU3RCUyQyUyMGxhYmVscyUzRGxhYmVscyklMjAlMjAlMjMlMjBiYXRjaCUyMHNpemUlMjBpcyUyMDElMEElMEElMjMlMjB0aGUlMjBsaW5lYXIlMjBjbGFzc2lmaWVyJTIwc3RpbGwlMjBuZWVkcyUyMHRvJTIwYmUlMjB0cmFpbmVkJTBBbG9zcyUyMCUzRCUyMG91dHB1dHMubG9zcyUwQWxvZ2l0cyUyMCUzRCUyMG91dHB1dHMubG9naXRz",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, NystromformerForMultipleChoice
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;uw-madison/nystromformer-512&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = NystromformerForMultipleChoice.from_pretrained(<span class="hljs-string">&quot;uw-madison/nystromformer-512&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>prompt = <span class="hljs-string">&quot;In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>choice0 = <span class="hljs-string">&quot;It is eaten with a fork and a knife.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>choice1 = <span class="hljs-string">&quot;It is eaten while held in the hand.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.tensor(<span class="hljs-number">0</span>).unsqueeze(<span class="hljs-number">0</span>)  <span class="hljs-comment"># choice0 is correct (according to Wikipedia ;)), batch size 1</span>

<span class="hljs-meta">&gt;&gt;&gt; </span>encoding = tokenizer([prompt, prompt], [choice0, choice1], return_tensors=<span class="hljs-string">&quot;pt&quot;</span>, padding=<span class="hljs-literal">True</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**{k: v.unsqueeze(<span class="hljs-number">0</span>) <span class="hljs-keyword">for</span> k, v <span class="hljs-keyword">in</span> encoding.items()}, labels=labels)  <span class="hljs-comment"># batch size is 1</span>

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># the linear classifier still needs to be trained</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = outputs.loss
<span class="hljs-meta">&gt;&gt;&gt; </span>logits = outputs.logits`,wrap:!1}}),{c(){t=m("p"),t.textContent=M,l=r(),h(d.$$.fragment)},l(o){t=p(o,"P",{"data-svelte-h":!0}),f(t)!=="svelte-11lpom8"&&(t.textContent=M),l=a(o),u(d.$$.fragment,o)},m(o,w){c(o,t,w),c(o,l,w),g(d,o,w),T=!0},p:L,i(o){T||(y(d.$$.fragment,o),T=!0)},o(o){_(d.$$.fragment,o),T=!1},d(o){o&&(s(t),s(l)),b(d,o)}}}function Fn(k){let t,M=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=m("p"),t.innerHTML=M},l(l){t=p(l,"P",{"data-svelte-h":!0}),f(t)!=="svelte-fincs2"&&(t.innerHTML=M)},m(l,d){c(l,t,d)},p:L,d(l){l&&s(t)}}}function xn(k){let t,M="Example:",l,d,T;return d=new Qe({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBOeXN0cm9tZm9ybWVyRm9yVG9rZW5DbGFzc2lmaWNhdGlvbiUwQWltcG9ydCUyMHRvcmNoJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIydXctbWFkaXNvbiUyRm55c3Ryb21mb3JtZXItNTEyJTIyKSUwQW1vZGVsJTIwJTNEJTIwTnlzdHJvbWZvcm1lckZvclRva2VuQ2xhc3NpZmljYXRpb24uZnJvbV9wcmV0cmFpbmVkKCUyMnV3LW1hZGlzb24lMkZueXN0cm9tZm9ybWVyLTUxMiUyMiklMEElMEFpbnB1dHMlMjAlM0QlMjB0b2tlbml6ZXIoJTBBJTIwJTIwJTIwJTIwJTIySHVnZ2luZ0ZhY2UlMjBpcyUyMGElMjBjb21wYW55JTIwYmFzZWQlMjBpbiUyMFBhcmlzJTIwYW5kJTIwTmV3JTIwWW9yayUyMiUyQyUyMGFkZF9zcGVjaWFsX3Rva2VucyUzREZhbHNlJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiUwQSklMEElMEF3aXRoJTIwdG9yY2gubm9fZ3JhZCgpJTNBJTBBJTIwJTIwJTIwJTIwbG9naXRzJTIwJTNEJTIwbW9kZWwoKippbnB1dHMpLmxvZ2l0cyUwQSUwQXByZWRpY3RlZF90b2tlbl9jbGFzc19pZHMlMjAlM0QlMjBsb2dpdHMuYXJnbWF4KC0xKSUwQSUwQSUyMyUyME5vdGUlMjB0aGF0JTIwdG9rZW5zJTIwYXJlJTIwY2xhc3NpZmllZCUyMHJhdGhlciUyMHRoZW4lMjBpbnB1dCUyMHdvcmRzJTIwd2hpY2glMjBtZWFucyUyMHRoYXQlMEElMjMlMjB0aGVyZSUyMG1pZ2h0JTIwYmUlMjBtb3JlJTIwcHJlZGljdGVkJTIwdG9rZW4lMjBjbGFzc2VzJTIwdGhhbiUyMHdvcmRzLiUwQSUyMyUyME11bHRpcGxlJTIwdG9rZW4lMjBjbGFzc2VzJTIwbWlnaHQlMjBhY2NvdW50JTIwZm9yJTIwdGhlJTIwc2FtZSUyMHdvcmQlMEFwcmVkaWN0ZWRfdG9rZW5zX2NsYXNzZXMlMjAlM0QlMjAlNUJtb2RlbC5jb25maWcuaWQybGFiZWwlNUJ0Lml0ZW0oKSU1RCUyMGZvciUyMHQlMjBpbiUyMHByZWRpY3RlZF90b2tlbl9jbGFzc19pZHMlNUIwJTVEJTVEJTBBcHJlZGljdGVkX3Rva2Vuc19jbGFzc2VzJTBBJTBBbGFiZWxzJTIwJTNEJTIwcHJlZGljdGVkX3Rva2VuX2NsYXNzX2lkcyUwQWxvc3MlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyUyQyUyMGxhYmVscyUzRGxhYmVscykubG9zcyUwQXJvdW5kKGxvc3MuaXRlbSgpJTJDJTIwMik=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, NystromformerForTokenClassification
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;uw-madison/nystromformer-512&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = NystromformerForTokenClassification.from_pretrained(<span class="hljs-string">&quot;uw-madison/nystromformer-512&quot;</span>)

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
...`,wrap:!1}}),{c(){t=m("p"),t.textContent=M,l=r(),h(d.$$.fragment)},l(o){t=p(o,"P",{"data-svelte-h":!0}),f(t)!=="svelte-11lpom8"&&(t.textContent=M),l=a(o),u(d.$$.fragment,o)},m(o,w){c(o,t,w),c(o,l,w),g(d,o,w),T=!0},p:L,i(o){T||(y(d.$$.fragment,o),T=!0)},o(o){_(d.$$.fragment,o),T=!1},d(o){o&&(s(t),s(l)),b(d,o)}}}function zn(k){let t,M=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=m("p"),t.innerHTML=M},l(l){t=p(l,"P",{"data-svelte-h":!0}),f(t)!=="svelte-fincs2"&&(t.innerHTML=M)},m(l,d){c(l,t,d)},p:L,d(l){l&&s(t)}}}function Wn(k){let t,M="Example:",l,d,T;return d=new Qe({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBOeXN0cm9tZm9ybWVyRm9yUXVlc3Rpb25BbnN3ZXJpbmclMEFpbXBvcnQlMjB0b3JjaCUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMnV3LW1hZGlzb24lMkZueXN0cm9tZm9ybWVyLTUxMiUyMiklMEFtb2RlbCUyMCUzRCUyME55c3Ryb21mb3JtZXJGb3JRdWVzdGlvbkFuc3dlcmluZy5mcm9tX3ByZXRyYWluZWQoJTIydXctbWFkaXNvbiUyRm55c3Ryb21mb3JtZXItNTEyJTIyKSUwQSUwQXF1ZXN0aW9uJTJDJTIwdGV4dCUyMCUzRCUyMCUyMldobyUyMHdhcyUyMEppbSUyMEhlbnNvbiUzRiUyMiUyQyUyMCUyMkppbSUyMEhlbnNvbiUyMHdhcyUyMGElMjBuaWNlJTIwcHVwcGV0JTIyJTBBJTBBaW5wdXRzJTIwJTNEJTIwdG9rZW5pemVyKHF1ZXN0aW9uJTJDJTIwdGV4dCUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpJTBBd2l0aCUyMHRvcmNoLm5vX2dyYWQoKSUzQSUwQSUyMCUyMCUyMCUyMG91dHB1dHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyklMEElMEFhbnN3ZXJfc3RhcnRfaW5kZXglMjAlM0QlMjBvdXRwdXRzLnN0YXJ0X2xvZ2l0cy5hcmdtYXgoKSUwQWFuc3dlcl9lbmRfaW5kZXglMjAlM0QlMjBvdXRwdXRzLmVuZF9sb2dpdHMuYXJnbWF4KCklMEElMEFwcmVkaWN0X2Fuc3dlcl90b2tlbnMlMjAlM0QlMjBpbnB1dHMuaW5wdXRfaWRzJTVCMCUyQyUyMGFuc3dlcl9zdGFydF9pbmRleCUyMCUzQSUyMGFuc3dlcl9lbmRfaW5kZXglMjAlMkIlMjAxJTVEJTBBdG9rZW5pemVyLmRlY29kZShwcmVkaWN0X2Fuc3dlcl90b2tlbnMlMkMlMjBza2lwX3NwZWNpYWxfdG9rZW5zJTNEVHJ1ZSklMEElMEElMjMlMjB0YXJnZXQlMjBpcyUyMCUyMm5pY2UlMjBwdXBwZXQlMjIlMEF0YXJnZXRfc3RhcnRfaW5kZXglMjAlM0QlMjB0b3JjaC50ZW5zb3IoJTVCMTQlNUQpJTBBdGFyZ2V0X2VuZF9pbmRleCUyMCUzRCUyMHRvcmNoLnRlbnNvciglNUIxNSU1RCklMEElMEFvdXRwdXRzJTIwJTNEJTIwbW9kZWwoKippbnB1dHMlMkMlMjBzdGFydF9wb3NpdGlvbnMlM0R0YXJnZXRfc3RhcnRfaW5kZXglMkMlMjBlbmRfcG9zaXRpb25zJTNEdGFyZ2V0X2VuZF9pbmRleCklMEFsb3NzJTIwJTNEJTIwb3V0cHV0cy5sb3NzJTBBcm91bmQobG9zcy5pdGVtKCklMkMlMjAyKQ==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, NystromformerForQuestionAnswering
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;uw-madison/nystromformer-512&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = NystromformerForQuestionAnswering.from_pretrained(<span class="hljs-string">&quot;uw-madison/nystromformer-512&quot;</span>)

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
...`,wrap:!1}}),{c(){t=m("p"),t.textContent=M,l=r(),h(d.$$.fragment)},l(o){t=p(o,"P",{"data-svelte-h":!0}),f(t)!=="svelte-11lpom8"&&(t.textContent=M),l=a(o),u(d.$$.fragment,o)},m(o,w){c(o,t,w),c(o,l,w),g(d,o,w),T=!0},p:L,i(o){T||(y(d.$$.fragment,o),T=!0)},o(o){_(d.$$.fragment,o),T=!1},d(o){o&&(s(t),s(l)),b(d,o)}}}function Zn(k){let t,M,l,d,T,o="<em>This model was released on 2021-02-07 and added to Hugging Face Transformers on 2022-01-11.</em>",w,ue,Mt,te,Xo='<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/>',Tt,ge,wt,ye,Vo=`The Nyströmformer model was proposed in <a href="https://huggingface.co/papers/2102.03902" rel="nofollow"><em>Nyströmformer: A Nyström-Based Algorithm for Approximating Self-Attention</em></a> by Yunyang Xiong, Zhanpeng Zeng, Rudrasis Chakraborty, Mingxing Tan, Glenn
Fung, Yin Li, and Vikas Singh.`,kt,_e,Bo="The abstract from the paper is the following:",vt,be,qo=`<em>Transformers have emerged as a powerful tool for a broad range of natural language processing tasks. A key component
that drives the impressive performance of Transformers is the self-attention mechanism that encodes the influence or
dependence of other tokens on each specific token. While beneficial, the quadratic complexity of self-attention on the
input sequence length has limited its application to longer sequences — a topic being actively studied in the
community. To address this limitation, we propose Nyströmformer — a model that exhibits favorable scalability as a
function of sequence length. Our idea is based on adapting the Nyström method to approximate standard self-attention
with O(n) complexity. The scalability of Nyströmformer enables application to longer sequences with thousands of
tokens. We perform evaluations on multiple downstream tasks on the GLUE benchmark and IMDB reviews with standard
sequence length, and find that our Nyströmformer performs comparably, or in a few cases, even slightly better, than
standard self-attention. On longer sequence tasks in the Long Range Arena (LRA) benchmark, Nyströmformer performs
favorably relative to other efficient self-attention methods. Our code is available at this https URL.</em>`,Nt,Me,Ro='This model was contributed by <a href="https://huggingface.co/novice03" rel="nofollow">novice03</a>. The original code can be found <a href="https://github.com/mlpen/Nystromformer" rel="nofollow">here</a>.',$t,Te,jt,we,Lo='<li><a href="../tasks/sequence_classification">Text classification task guide</a></li> <li><a href="../tasks/token_classification">Token classification task guide</a></li> <li><a href="../tasks/question_answering">Question answering task guide</a></li> <li><a href="../tasks/masked_language_modeling">Masked language modeling task guide</a></li> <li><a href="../tasks/multiple_choice">Multiple choice task guide</a></li>',Jt,ke,Ut,z,ve,Qt,Ee,Go=`This is the configuration class to store the configuration of a <a href="/docs/transformers/v4.56.2/en/model_doc/nystromformer#transformers.NystromformerModel">NystromformerModel</a>. It is used to instantiate
an Nystromformer model according to the specified arguments, defining the model architecture. Instantiating a
configuration with the defaults will yield a similar configuration to that of the Nystromformer
<a href="https://huggingface.co/uw-madison/nystromformer-512" rel="nofollow">uw-madison/nystromformer-512</a> architecture.`,Et,Se,Ho=`Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> and can be used to control the model outputs. Read the
documentation from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> for more information.`,St,oe,Ct,Ne,Ft,N,$e,Yt,Ye,Qo="The bare Nystromformer Model outputting raw hidden-states without any specific head on top.",At,Ae,Eo=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Pt,Pe,So=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Ot,D,je,Dt,Oe,Yo='The <a href="/docs/transformers/v4.56.2/en/model_doc/nystromformer#transformers.NystromformerModel">NystromformerModel</a> forward method, overrides the <code>__call__</code> special method.',Kt,ne,xt,Je,zt,$,Ue,eo,De,Ao="The Nystromformer Model with a <code>language modeling</code> head on top.”",to,Ke,Po=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,oo,et,Oo=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,no,G,Ce,so,tt,Do='The <a href="/docs/transformers/v4.56.2/en/model_doc/nystromformer#transformers.NystromformerForMaskedLM">NystromformerForMaskedLM</a> forward method, overrides the <code>__call__</code> special method.',ro,se,ao,re,Wt,Fe,Zt,j,xe,io,ot,Ko=`Nyströmformer Model transformer with a sequence classification/regression head on top (a linear layer on top of the
pooled output) e.g. for GLUE tasks.`,lo,nt,en=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,co,st,tn=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,mo,x,ze,po,rt,on='The <a href="/docs/transformers/v4.56.2/en/model_doc/nystromformer#transformers.NystromformerForSequenceClassification">NystromformerForSequenceClassification</a> forward method, overrides the <code>__call__</code> special method.',fo,ae,ho,ie,uo,le,It,We,Xt,J,Ze,go,at,nn=`The Nystromformer Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
softmax) e.g. for RocStories/SWAG tasks.`,yo,it,sn=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,_o,lt,rn=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,bo,H,Ie,Mo,dt,an='The <a href="/docs/transformers/v4.56.2/en/model_doc/nystromformer#transformers.NystromformerForMultipleChoice">NystromformerForMultipleChoice</a> forward method, overrides the <code>__call__</code> special method.',To,de,wo,ce,Vt,Xe,Bt,U,Ve,ko,ct,ln=`The Nystromformer transformer with a token classification head on top (a linear layer on top of the hidden-states
output) e.g. for Named-Entity-Recognition (NER) tasks.`,vo,mt,dn=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,No,pt,cn=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,$o,Q,Be,jo,ft,mn='The <a href="/docs/transformers/v4.56.2/en/model_doc/nystromformer#transformers.NystromformerForTokenClassification">NystromformerForTokenClassification</a> forward method, overrides the <code>__call__</code> special method.',Jo,me,Uo,pe,qt,qe,Rt,C,Re,Co,ht,pn=`The Nystromformer transformer with a span classification head on top for extractive question-answering tasks like
SQuAD (a linear layer on top of the hidden-states output to compute <code>span start logits</code> and <code>span end logits</code>).`,Fo,ut,fn=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,xo,gt,hn=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,zo,E,Le,Wo,yt,un='The <a href="/docs/transformers/v4.56.2/en/model_doc/nystromformer#transformers.NystromformerForQuestionAnswering">NystromformerForQuestionAnswering</a> forward method, overrides the <code>__call__</code> special method.',Zo,fe,Io,he,Lt,Ge,Gt,bt,Ht;return ue=new K({props:{title:"Nyströmformer",local:"nyströmformer",headingTag:"h1"}}),ge=new K({props:{title:"Overview",local:"overview",headingTag:"h2"}}),Te=new K({props:{title:"Resources",local:"resources",headingTag:"h2"}}),ke=new K({props:{title:"NystromformerConfig",local:"transformers.NystromformerConfig",headingTag:"h2"}}),ve=new R({props:{name:"class transformers.NystromformerConfig",anchor:"transformers.NystromformerConfig",parameters:[{name:"vocab_size",val:" = 30000"},{name:"hidden_size",val:" = 768"},{name:"num_hidden_layers",val:" = 12"},{name:"num_attention_heads",val:" = 12"},{name:"intermediate_size",val:" = 3072"},{name:"hidden_act",val:" = 'gelu_new'"},{name:"hidden_dropout_prob",val:" = 0.1"},{name:"attention_probs_dropout_prob",val:" = 0.1"},{name:"max_position_embeddings",val:" = 510"},{name:"type_vocab_size",val:" = 2"},{name:"segment_means_seq_len",val:" = 64"},{name:"num_landmarks",val:" = 64"},{name:"conv_kernel_size",val:" = 65"},{name:"inv_coeff_init_option",val:" = False"},{name:"initializer_range",val:" = 0.02"},{name:"layer_norm_eps",val:" = 1e-05"},{name:"pad_token_id",val:" = 1"},{name:"bos_token_id",val:" = 0"},{name:"eos_token_id",val:" = 2"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.NystromformerConfig.vocab_size",description:`<strong>vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 30000) &#x2014;
Vocabulary size of the Nystromformer model. Defines the number of different tokens that can be represented
by the <code>inputs_ids</code> passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/nystromformer#transformers.NystromformerModel">NystromformerModel</a>.`,name:"vocab_size"},{anchor:"transformers.NystromformerConfig.hidden_size",description:`<strong>hidden_size</strong> (<code>int</code>, <em>optional</em>, defaults to 768) &#x2014;
Dimension of the encoder layers and the pooler layer.`,name:"hidden_size"},{anchor:"transformers.NystromformerConfig.num_hidden_layers",description:`<strong>num_hidden_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 12) &#x2014;
Number of hidden layers in the Transformer encoder.`,name:"num_hidden_layers"},{anchor:"transformers.NystromformerConfig.num_attention_heads",description:`<strong>num_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 12) &#x2014;
Number of attention heads for each attention layer in the Transformer encoder.`,name:"num_attention_heads"},{anchor:"transformers.NystromformerConfig.intermediate_size",description:`<strong>intermediate_size</strong> (<code>int</code>, <em>optional</em>, defaults to 3072) &#x2014;
Dimension of the &#x201C;intermediate&#x201D; (i.e., feed-forward) layer in the Transformer encoder.`,name:"intermediate_size"},{anchor:"transformers.NystromformerConfig.hidden_act",description:`<strong>hidden_act</strong> (<code>str</code> or <code>function</code>, <em>optional</em>, defaults to <code>&quot;gelu&quot;</code>) &#x2014;
The non-linear activation function (function or string) in the encoder and pooler. If string, <code>&quot;gelu&quot;</code>,
<code>&quot;relu&quot;</code>, <code>&quot;selu&quot;</code> and <code>&quot;gelu_new&quot;</code> are supported.`,name:"hidden_act"},{anchor:"transformers.NystromformerConfig.hidden_dropout_prob",description:`<strong>hidden_dropout_prob</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.`,name:"hidden_dropout_prob"},{anchor:"transformers.NystromformerConfig.attention_probs_dropout_prob",description:`<strong>attention_probs_dropout_prob</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout ratio for the attention probabilities.`,name:"attention_probs_dropout_prob"},{anchor:"transformers.NystromformerConfig.max_position_embeddings",description:`<strong>max_position_embeddings</strong> (<code>int</code>, <em>optional</em>, defaults to 512) &#x2014;
The maximum sequence length that this model might ever be used with. Typically set this to something large
just in case (e.g., 512 or 1024 or 2048).`,name:"max_position_embeddings"},{anchor:"transformers.NystromformerConfig.type_vocab_size",description:`<strong>type_vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 2) &#x2014;
The vocabulary size of the <code>token_type_ids</code> passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/nystromformer#transformers.NystromformerModel">NystromformerModel</a>.`,name:"type_vocab_size"},{anchor:"transformers.NystromformerConfig.segment_means_seq_len",description:`<strong>segment_means_seq_len</strong> (<code>int</code>, <em>optional</em>, defaults to 64) &#x2014;
Sequence length used in segment-means.`,name:"segment_means_seq_len"},{anchor:"transformers.NystromformerConfig.num_landmarks",description:`<strong>num_landmarks</strong> (<code>int</code>, <em>optional</em>, defaults to 64) &#x2014;
The number of landmark (or Nystrom) points to use in Nystrom approximation of the softmax self-attention
matrix.`,name:"num_landmarks"},{anchor:"transformers.NystromformerConfig.conv_kernel_size",description:`<strong>conv_kernel_size</strong> (<code>int</code>, <em>optional</em>, defaults to 65) &#x2014;
The kernel size of depthwise convolution used in Nystrom approximation.`,name:"conv_kernel_size"},{anchor:"transformers.NystromformerConfig.inv_coeff_init_option",description:`<strong>inv_coeff_init_option</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to use exact coefficient computation for the initial values for the iterative method of
calculating the Moore-Penrose inverse of a matrix.`,name:"inv_coeff_init_option"},{anchor:"transformers.NystromformerConfig.initializer_range",description:`<strong>initializer_range</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.`,name:"initializer_range"},{anchor:"transformers.NystromformerConfig.layer_norm_eps",description:`<strong>layer_norm_eps</strong> (<code>float</code>, <em>optional</em>, defaults to 1e-12) &#x2014;
The epsilon used by the layer normalization layers.`,name:"layer_norm_eps"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/nystromformer/configuration_nystromformer.py#L24"}}),oe=new He({props:{anchor:"transformers.NystromformerConfig.example",$$slots:{default:[wn]},$$scope:{ctx:k}}}),Ne=new K({props:{title:"NystromformerModel",local:"transformers.NystromformerModel",headingTag:"h2"}}),$e=new R({props:{name:"class transformers.NystromformerModel",anchor:"transformers.NystromformerModel",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.NystromformerModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/nystromformer#transformers.NystromformerModel">NystromformerModel</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/nystromformer/modeling_nystromformer.py#L468"}}),je=new R({props:{name:"forward",anchor:"transformers.NystromformerModel.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.NystromformerModel.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.NystromformerModel.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.NystromformerModel.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.NystromformerModel.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.NystromformerModel.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.NystromformerModel.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.NystromformerModel.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.NystromformerModel.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.NystromformerModel.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/nystromformer/modeling_nystromformer.py#L493",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPastAndCrossAttentions"
>transformers.modeling_outputs.BaseModelOutputWithPastAndCrossAttentions</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/nystromformer#transformers.NystromformerConfig"
>NystromformerConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>last_hidden_state</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>) — Sequence of hidden-states at the output of the last layer of the model.</p>
<p>If <code>past_key_values</code> is used only the last hidden-state of the sequences of shape <code>(batch_size, 1, hidden_size)</code> is output.</p>
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
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPastAndCrossAttentions"
>transformers.modeling_outputs.BaseModelOutputWithPastAndCrossAttentions</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),ne=new _t({props:{$$slots:{default:[kn]},$$scope:{ctx:k}}}),Je=new K({props:{title:"NystromformerForMaskedLM",local:"transformers.NystromformerForMaskedLM",headingTag:"h2"}}),Ue=new R({props:{name:"class transformers.NystromformerForMaskedLM",anchor:"transformers.NystromformerForMaskedLM",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.NystromformerForMaskedLM.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/nystromformer#transformers.NystromformerForMaskedLM">NystromformerForMaskedLM</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/nystromformer/modeling_nystromformer.py#L575"}}),Ce=new R({props:{name:"forward",anchor:"transformers.NystromformerForMaskedLM.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.NystromformerForMaskedLM.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.NystromformerForMaskedLM.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.NystromformerForMaskedLM.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.NystromformerForMaskedLM.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.NystromformerForMaskedLM.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.NystromformerForMaskedLM.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.NystromformerForMaskedLM.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should be in <code>[-100, 0, ..., config.vocab_size]</code> (see <code>input_ids</code> docstring) Tokens with indices set to <code>-100</code> are ignored (masked), the
loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.NystromformerForMaskedLM.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.NystromformerForMaskedLM.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.NystromformerForMaskedLM.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/nystromformer/modeling_nystromformer.py#L594",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.MaskedLMOutput"
>transformers.modeling_outputs.MaskedLMOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/nystromformer#transformers.NystromformerConfig"
>NystromformerConfig</a>) and inputs.</p>
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
`}}),se=new _t({props:{$$slots:{default:[vn]},$$scope:{ctx:k}}}),re=new He({props:{anchor:"transformers.NystromformerForMaskedLM.forward.example",$$slots:{default:[Nn]},$$scope:{ctx:k}}}),Fe=new K({props:{title:"NystromformerForSequenceClassification",local:"transformers.NystromformerForSequenceClassification",headingTag:"h2"}}),xe=new R({props:{name:"class transformers.NystromformerForSequenceClassification",anchor:"transformers.NystromformerForSequenceClassification",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.NystromformerForSequenceClassification.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/nystromformer#transformers.NystromformerForSequenceClassification">NystromformerForSequenceClassification</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/nystromformer/modeling_nystromformer.py#L675"}}),ze=new R({props:{name:"forward",anchor:"transformers.NystromformerForSequenceClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.NystromformerForSequenceClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.NystromformerForSequenceClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.NystromformerForSequenceClassification.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.NystromformerForSequenceClassification.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.NystromformerForSequenceClassification.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.NystromformerForSequenceClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.NystromformerForSequenceClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the sequence classification/regression loss. Indices should be in <code>[0, ..., config.num_labels - 1]</code>. If <code>config.num_labels == 1</code> a regression loss is computed (Mean-Square loss), If
<code>config.num_labels &gt; 1</code> a classification loss is computed (Cross-Entropy).`,name:"labels"},{anchor:"transformers.NystromformerForSequenceClassification.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.NystromformerForSequenceClassification.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.NystromformerForSequenceClassification.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/nystromformer/modeling_nystromformer.py#L685",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput"
>transformers.modeling_outputs.SequenceClassifierOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/nystromformer#transformers.NystromformerConfig"
>NystromformerConfig</a>) and inputs.</p>
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
`}}),ae=new _t({props:{$$slots:{default:[$n]},$$scope:{ctx:k}}}),ie=new He({props:{anchor:"transformers.NystromformerForSequenceClassification.forward.example",$$slots:{default:[jn]},$$scope:{ctx:k}}}),le=new He({props:{anchor:"transformers.NystromformerForSequenceClassification.forward.example-2",$$slots:{default:[Jn]},$$scope:{ctx:k}}}),We=new K({props:{title:"NystromformerForMultipleChoice",local:"transformers.NystromformerForMultipleChoice",headingTag:"h2"}}),Ze=new R({props:{name:"class transformers.NystromformerForMultipleChoice",anchor:"transformers.NystromformerForMultipleChoice",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.NystromformerForMultipleChoice.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/nystromformer#transformers.NystromformerForMultipleChoice">NystromformerForMultipleChoice</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/nystromformer/modeling_nystromformer.py#L757"}}),Ie=new R({props:{name:"forward",anchor:"transformers.NystromformerForMultipleChoice.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.NystromformerForMultipleChoice.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, num_choices, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.NystromformerForMultipleChoice.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.NystromformerForMultipleChoice.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, num_choices, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.NystromformerForMultipleChoice.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, num_choices, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.max_position_embeddings - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.NystromformerForMultipleChoice.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.NystromformerForMultipleChoice.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_choices, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <em>input_ids</em> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.NystromformerForMultipleChoice.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the multiple choice classification loss. Indices should be in <code>[0, ..., num_choices-1]</code> where <code>num_choices</code> is the size of the second dimension of the input tensors. (See
<code>input_ids</code> above)`,name:"labels"},{anchor:"transformers.NystromformerForMultipleChoice.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.NystromformerForMultipleChoice.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.NystromformerForMultipleChoice.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/nystromformer/modeling_nystromformer.py#L768",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.MultipleChoiceModelOutput"
>transformers.modeling_outputs.MultipleChoiceModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/nystromformer#transformers.NystromformerConfig"
>NystromformerConfig</a>) and inputs.</p>
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
`}}),de=new _t({props:{$$slots:{default:[Un]},$$scope:{ctx:k}}}),ce=new He({props:{anchor:"transformers.NystromformerForMultipleChoice.forward.example",$$slots:{default:[Cn]},$$scope:{ctx:k}}}),Xe=new K({props:{title:"NystromformerForTokenClassification",local:"transformers.NystromformerForTokenClassification",headingTag:"h2"}}),Ve=new R({props:{name:"class transformers.NystromformerForTokenClassification",anchor:"transformers.NystromformerForTokenClassification",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.NystromformerForTokenClassification.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/nystromformer#transformers.NystromformerForTokenClassification">NystromformerForTokenClassification</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/nystromformer/modeling_nystromformer.py#L863"}}),Be=new R({props:{name:"forward",anchor:"transformers.NystromformerForTokenClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.NystromformerForTokenClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.NystromformerForTokenClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.NystromformerForTokenClassification.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.NystromformerForTokenClassification.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.NystromformerForTokenClassification.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.NystromformerForTokenClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.NystromformerForTokenClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the token classification loss. Indices should be in <code>[0, ..., config.num_labels - 1]</code>.`,name:"labels"},{anchor:"transformers.NystromformerForTokenClassification.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.NystromformerForTokenClassification.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.NystromformerForTokenClassification.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/nystromformer/modeling_nystromformer.py#L875",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput"
>transformers.modeling_outputs.TokenClassifierOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/nystromformer#transformers.NystromformerConfig"
>NystromformerConfig</a>) and inputs.</p>
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
`}}),me=new _t({props:{$$slots:{default:[Fn]},$$scope:{ctx:k}}}),pe=new He({props:{anchor:"transformers.NystromformerForTokenClassification.forward.example",$$slots:{default:[xn]},$$scope:{ctx:k}}}),qe=new K({props:{title:"NystromformerForQuestionAnswering",local:"transformers.NystromformerForQuestionAnswering",headingTag:"h2"}}),Re=new R({props:{name:"class transformers.NystromformerForQuestionAnswering",anchor:"transformers.NystromformerForQuestionAnswering",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.NystromformerForQuestionAnswering.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/nystromformer#transformers.NystromformerForQuestionAnswering">NystromformerForQuestionAnswering</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/nystromformer/modeling_nystromformer.py#L930"}}),Le=new R({props:{name:"forward",anchor:"transformers.NystromformerForQuestionAnswering.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"start_positions",val:": typing.Optional[torch.LongTensor] = None"},{name:"end_positions",val:": typing.Optional[torch.LongTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.NystromformerForQuestionAnswering.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.NystromformerForQuestionAnswering.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.NystromformerForQuestionAnswering.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.NystromformerForQuestionAnswering.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.NystromformerForQuestionAnswering.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.NystromformerForQuestionAnswering.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.NystromformerForQuestionAnswering.forward.start_positions",description:`<strong>start_positions</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for position (index) of the start of the labelled span for computing the token classification loss.
Positions are clamped to the length of the sequence (<code>sequence_length</code>). Position outside of the sequence
are not taken into account for computing the loss.`,name:"start_positions"},{anchor:"transformers.NystromformerForQuestionAnswering.forward.end_positions",description:`<strong>end_positions</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for position (index) of the end of the labelled span for computing the token classification loss.
Positions are clamped to the length of the sequence (<code>sequence_length</code>). Position outside of the sequence
are not taken into account for computing the loss.`,name:"end_positions"},{anchor:"transformers.NystromformerForQuestionAnswering.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.NystromformerForQuestionAnswering.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.NystromformerForQuestionAnswering.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/nystromformer/modeling_nystromformer.py#L943",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.QuestionAnsweringModelOutput"
>transformers.modeling_outputs.QuestionAnsweringModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/nystromformer#transformers.NystromformerConfig"
>NystromformerConfig</a>) and inputs.</p>
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
`}}),fe=new _t({props:{$$slots:{default:[zn]},$$scope:{ctx:k}}}),he=new He({props:{anchor:"transformers.NystromformerForQuestionAnswering.forward.example",$$slots:{default:[Wn]},$$scope:{ctx:k}}}),Ge=new Tn({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/nystromformer.md"}}),{c(){t=m("meta"),M=r(),l=m("p"),d=r(),T=m("p"),T.innerHTML=o,w=r(),h(ue.$$.fragment),Mt=r(),te=m("div"),te.innerHTML=Xo,Tt=r(),h(ge.$$.fragment),wt=r(),ye=m("p"),ye.innerHTML=Vo,kt=r(),_e=m("p"),_e.textContent=Bo,vt=r(),be=m("p"),be.innerHTML=qo,Nt=r(),Me=m("p"),Me.innerHTML=Ro,$t=r(),h(Te.$$.fragment),jt=r(),we=m("ul"),we.innerHTML=Lo,Jt=r(),h(ke.$$.fragment),Ut=r(),z=m("div"),h(ve.$$.fragment),Qt=r(),Ee=m("p"),Ee.innerHTML=Go,Et=r(),Se=m("p"),Se.innerHTML=Ho,St=r(),h(oe.$$.fragment),Ct=r(),h(Ne.$$.fragment),Ft=r(),N=m("div"),h($e.$$.fragment),Yt=r(),Ye=m("p"),Ye.textContent=Qo,At=r(),Ae=m("p"),Ae.innerHTML=Eo,Pt=r(),Pe=m("p"),Pe.innerHTML=So,Ot=r(),D=m("div"),h(je.$$.fragment),Dt=r(),Oe=m("p"),Oe.innerHTML=Yo,Kt=r(),h(ne.$$.fragment),xt=r(),h(Je.$$.fragment),zt=r(),$=m("div"),h(Ue.$$.fragment),eo=r(),De=m("p"),De.innerHTML=Ao,to=r(),Ke=m("p"),Ke.innerHTML=Po,oo=r(),et=m("p"),et.innerHTML=Oo,no=r(),G=m("div"),h(Ce.$$.fragment),so=r(),tt=m("p"),tt.innerHTML=Do,ro=r(),h(se.$$.fragment),ao=r(),h(re.$$.fragment),Wt=r(),h(Fe.$$.fragment),Zt=r(),j=m("div"),h(xe.$$.fragment),io=r(),ot=m("p"),ot.textContent=Ko,lo=r(),nt=m("p"),nt.innerHTML=en,co=r(),st=m("p"),st.innerHTML=tn,mo=r(),x=m("div"),h(ze.$$.fragment),po=r(),rt=m("p"),rt.innerHTML=on,fo=r(),h(ae.$$.fragment),ho=r(),h(ie.$$.fragment),uo=r(),h(le.$$.fragment),It=r(),h(We.$$.fragment),Xt=r(),J=m("div"),h(Ze.$$.fragment),go=r(),at=m("p"),at.textContent=nn,yo=r(),it=m("p"),it.innerHTML=sn,_o=r(),lt=m("p"),lt.innerHTML=rn,bo=r(),H=m("div"),h(Ie.$$.fragment),Mo=r(),dt=m("p"),dt.innerHTML=an,To=r(),h(de.$$.fragment),wo=r(),h(ce.$$.fragment),Vt=r(),h(Xe.$$.fragment),Bt=r(),U=m("div"),h(Ve.$$.fragment),ko=r(),ct=m("p"),ct.textContent=ln,vo=r(),mt=m("p"),mt.innerHTML=dn,No=r(),pt=m("p"),pt.innerHTML=cn,$o=r(),Q=m("div"),h(Be.$$.fragment),jo=r(),ft=m("p"),ft.innerHTML=mn,Jo=r(),h(me.$$.fragment),Uo=r(),h(pe.$$.fragment),qt=r(),h(qe.$$.fragment),Rt=r(),C=m("div"),h(Re.$$.fragment),Co=r(),ht=m("p"),ht.innerHTML=pn,Fo=r(),ut=m("p"),ut.innerHTML=fn,xo=r(),gt=m("p"),gt.innerHTML=hn,zo=r(),E=m("div"),h(Le.$$.fragment),Wo=r(),yt=m("p"),yt.innerHTML=un,Zo=r(),h(fe.$$.fragment),Io=r(),h(he.$$.fragment),Lt=r(),h(Ge.$$.fragment),Gt=r(),bt=m("p"),this.h()},l(e){const n=Mn("svelte-u9bgzb",document.head);t=p(n,"META",{name:!0,content:!0}),n.forEach(s),M=a(e),l=p(e,"P",{}),F(l).forEach(s),d=a(e),T=p(e,"P",{"data-svelte-h":!0}),f(T)!=="svelte-1ipxp5p"&&(T.innerHTML=o),w=a(e),u(ue.$$.fragment,e),Mt=a(e),te=p(e,"DIV",{class:!0,"data-svelte-h":!0}),f(te)!=="svelte-13t8s2t"&&(te.innerHTML=Xo),Tt=a(e),u(ge.$$.fragment,e),wt=a(e),ye=p(e,"P",{"data-svelte-h":!0}),f(ye)!=="svelte-5fifyb"&&(ye.innerHTML=Vo),kt=a(e),_e=p(e,"P",{"data-svelte-h":!0}),f(_e)!=="svelte-vfdo9a"&&(_e.textContent=Bo),vt=a(e),be=p(e,"P",{"data-svelte-h":!0}),f(be)!=="svelte-151jd63"&&(be.innerHTML=qo),Nt=a(e),Me=p(e,"P",{"data-svelte-h":!0}),f(Me)!=="svelte-1ctybco"&&(Me.innerHTML=Ro),$t=a(e),u(Te.$$.fragment,e),jt=a(e),we=p(e,"UL",{"data-svelte-h":!0}),f(we)!=="svelte-mgusi3"&&(we.innerHTML=Lo),Jt=a(e),u(ke.$$.fragment,e),Ut=a(e),z=p(e,"DIV",{class:!0});var S=F(z);u(ve.$$.fragment,S),Qt=a(S),Ee=p(S,"P",{"data-svelte-h":!0}),f(Ee)!=="svelte-1bfvu9u"&&(Ee.innerHTML=Go),Et=a(S),Se=p(S,"P",{"data-svelte-h":!0}),f(Se)!=="svelte-1ek1ss9"&&(Se.innerHTML=Ho),St=a(S),u(oe.$$.fragment,S),S.forEach(s),Ct=a(e),u(Ne.$$.fragment,e),Ft=a(e),N=p(e,"DIV",{class:!0});var W=F(N);u($e.$$.fragment,W),Yt=a(W),Ye=p(W,"P",{"data-svelte-h":!0}),f(Ye)!=="svelte-1n7n33t"&&(Ye.textContent=Qo),At=a(W),Ae=p(W,"P",{"data-svelte-h":!0}),f(Ae)!=="svelte-q52n56"&&(Ae.innerHTML=Eo),Pt=a(W),Pe=p(W,"P",{"data-svelte-h":!0}),f(Pe)!=="svelte-hswkmf"&&(Pe.innerHTML=So),Ot=a(W),D=p(W,"DIV",{class:!0});var ee=F(D);u(je.$$.fragment,ee),Dt=a(ee),Oe=p(ee,"P",{"data-svelte-h":!0}),f(Oe)!=="svelte-1751i8w"&&(Oe.innerHTML=Yo),Kt=a(ee),u(ne.$$.fragment,ee),ee.forEach(s),W.forEach(s),xt=a(e),u(Je.$$.fragment,e),zt=a(e),$=p(e,"DIV",{class:!0});var Z=F($);u(Ue.$$.fragment,Z),eo=a(Z),De=p(Z,"P",{"data-svelte-h":!0}),f(De)!=="svelte-ozda9r"&&(De.innerHTML=Ao),to=a(Z),Ke=p(Z,"P",{"data-svelte-h":!0}),f(Ke)!=="svelte-q52n56"&&(Ke.innerHTML=Po),oo=a(Z),et=p(Z,"P",{"data-svelte-h":!0}),f(et)!=="svelte-hswkmf"&&(et.innerHTML=Oo),no=a(Z),G=p(Z,"DIV",{class:!0});var Y=F(G);u(Ce.$$.fragment,Y),so=a(Y),tt=p(Y,"P",{"data-svelte-h":!0}),f(tt)!=="svelte-qe0pjk"&&(tt.innerHTML=Do),ro=a(Y),u(se.$$.fragment,Y),ao=a(Y),u(re.$$.fragment,Y),Y.forEach(s),Z.forEach(s),Wt=a(e),u(Fe.$$.fragment,e),Zt=a(e),j=p(e,"DIV",{class:!0});var I=F(j);u(xe.$$.fragment,I),io=a(I),ot=p(I,"P",{"data-svelte-h":!0}),f(ot)!=="svelte-xud4sm"&&(ot.textContent=Ko),lo=a(I),nt=p(I,"P",{"data-svelte-h":!0}),f(nt)!=="svelte-q52n56"&&(nt.innerHTML=en),co=a(I),st=p(I,"P",{"data-svelte-h":!0}),f(st)!=="svelte-hswkmf"&&(st.innerHTML=tn),mo=a(I),x=p(I,"DIV",{class:!0});var X=F(x);u(ze.$$.fragment,X),po=a(X),rt=p(X,"P",{"data-svelte-h":!0}),f(rt)!=="svelte-1epuczy"&&(rt.innerHTML=on),fo=a(X),u(ae.$$.fragment,X),ho=a(X),u(ie.$$.fragment,X),uo=a(X),u(le.$$.fragment,X),X.forEach(s),I.forEach(s),It=a(e),u(We.$$.fragment,e),Xt=a(e),J=p(e,"DIV",{class:!0});var V=F(J);u(Ze.$$.fragment,V),go=a(V),at=p(V,"P",{"data-svelte-h":!0}),f(at)!=="svelte-18i177o"&&(at.textContent=nn),yo=a(V),it=p(V,"P",{"data-svelte-h":!0}),f(it)!=="svelte-q52n56"&&(it.innerHTML=sn),_o=a(V),lt=p(V,"P",{"data-svelte-h":!0}),f(lt)!=="svelte-hswkmf"&&(lt.innerHTML=rn),bo=a(V),H=p(V,"DIV",{class:!0});var A=F(H);u(Ie.$$.fragment,A),Mo=a(A),dt=p(A,"P",{"data-svelte-h":!0}),f(dt)!=="svelte-v6ahmq"&&(dt.innerHTML=an),To=a(A),u(de.$$.fragment,A),wo=a(A),u(ce.$$.fragment,A),A.forEach(s),V.forEach(s),Vt=a(e),u(Xe.$$.fragment,e),Bt=a(e),U=p(e,"DIV",{class:!0});var B=F(U);u(Ve.$$.fragment,B),ko=a(B),ct=p(B,"P",{"data-svelte-h":!0}),f(ct)!=="svelte-1ljsy3f"&&(ct.textContent=ln),vo=a(B),mt=p(B,"P",{"data-svelte-h":!0}),f(mt)!=="svelte-q52n56"&&(mt.innerHTML=dn),No=a(B),pt=p(B,"P",{"data-svelte-h":!0}),f(pt)!=="svelte-hswkmf"&&(pt.innerHTML=cn),$o=a(B),Q=p(B,"DIV",{class:!0});var P=F(Q);u(Be.$$.fragment,P),jo=a(P),ft=p(P,"P",{"data-svelte-h":!0}),f(ft)!=="svelte-1sspkng"&&(ft.innerHTML=mn),Jo=a(P),u(me.$$.fragment,P),Uo=a(P),u(pe.$$.fragment,P),P.forEach(s),B.forEach(s),qt=a(e),u(qe.$$.fragment,e),Rt=a(e),C=p(e,"DIV",{class:!0});var q=F(C);u(Re.$$.fragment,q),Co=a(q),ht=p(q,"P",{"data-svelte-h":!0}),f(ht)!=="svelte-1tt5760"&&(ht.innerHTML=pn),Fo=a(q),ut=p(q,"P",{"data-svelte-h":!0}),f(ut)!=="svelte-q52n56"&&(ut.innerHTML=fn),xo=a(q),gt=p(q,"P",{"data-svelte-h":!0}),f(gt)!=="svelte-hswkmf"&&(gt.innerHTML=hn),zo=a(q),E=p(q,"DIV",{class:!0});var O=F(E);u(Le.$$.fragment,O),Wo=a(O),yt=p(O,"P",{"data-svelte-h":!0}),f(yt)!=="svelte-s04iz4"&&(yt.innerHTML=un),Zo=a(O),u(fe.$$.fragment,O),Io=a(O),u(he.$$.fragment,O),O.forEach(s),q.forEach(s),Lt=a(e),u(Ge.$$.fragment,e),Gt=a(e),bt=p(e,"P",{}),F(bt).forEach(s),this.h()},h(){v(t,"name","hf:doc:metadata"),v(t,"content",In),v(te,"class","flex flex-wrap space-x-1"),v(z,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(D,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(N,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(G,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v($,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(x,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(j,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(H,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(J,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(Q,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(U,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(E,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(C,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,n){i(document.head,t),c(e,M,n),c(e,l,n),c(e,d,n),c(e,T,n),c(e,w,n),g(ue,e,n),c(e,Mt,n),c(e,te,n),c(e,Tt,n),g(ge,e,n),c(e,wt,n),c(e,ye,n),c(e,kt,n),c(e,_e,n),c(e,vt,n),c(e,be,n),c(e,Nt,n),c(e,Me,n),c(e,$t,n),g(Te,e,n),c(e,jt,n),c(e,we,n),c(e,Jt,n),g(ke,e,n),c(e,Ut,n),c(e,z,n),g(ve,z,null),i(z,Qt),i(z,Ee),i(z,Et),i(z,Se),i(z,St),g(oe,z,null),c(e,Ct,n),g(Ne,e,n),c(e,Ft,n),c(e,N,n),g($e,N,null),i(N,Yt),i(N,Ye),i(N,At),i(N,Ae),i(N,Pt),i(N,Pe),i(N,Ot),i(N,D),g(je,D,null),i(D,Dt),i(D,Oe),i(D,Kt),g(ne,D,null),c(e,xt,n),g(Je,e,n),c(e,zt,n),c(e,$,n),g(Ue,$,null),i($,eo),i($,De),i($,to),i($,Ke),i($,oo),i($,et),i($,no),i($,G),g(Ce,G,null),i(G,so),i(G,tt),i(G,ro),g(se,G,null),i(G,ao),g(re,G,null),c(e,Wt,n),g(Fe,e,n),c(e,Zt,n),c(e,j,n),g(xe,j,null),i(j,io),i(j,ot),i(j,lo),i(j,nt),i(j,co),i(j,st),i(j,mo),i(j,x),g(ze,x,null),i(x,po),i(x,rt),i(x,fo),g(ae,x,null),i(x,ho),g(ie,x,null),i(x,uo),g(le,x,null),c(e,It,n),g(We,e,n),c(e,Xt,n),c(e,J,n),g(Ze,J,null),i(J,go),i(J,at),i(J,yo),i(J,it),i(J,_o),i(J,lt),i(J,bo),i(J,H),g(Ie,H,null),i(H,Mo),i(H,dt),i(H,To),g(de,H,null),i(H,wo),g(ce,H,null),c(e,Vt,n),g(Xe,e,n),c(e,Bt,n),c(e,U,n),g(Ve,U,null),i(U,ko),i(U,ct),i(U,vo),i(U,mt),i(U,No),i(U,pt),i(U,$o),i(U,Q),g(Be,Q,null),i(Q,jo),i(Q,ft),i(Q,Jo),g(me,Q,null),i(Q,Uo),g(pe,Q,null),c(e,qt,n),g(qe,e,n),c(e,Rt,n),c(e,C,n),g(Re,C,null),i(C,Co),i(C,ht),i(C,Fo),i(C,ut),i(C,xo),i(C,gt),i(C,zo),i(C,E),g(Le,E,null),i(E,Wo),i(E,yt),i(E,Zo),g(fe,E,null),i(E,Io),g(he,E,null),c(e,Lt,n),g(Ge,e,n),c(e,Gt,n),c(e,bt,n),Ht=!0},p(e,[n]){const S={};n&2&&(S.$$scope={dirty:n,ctx:e}),oe.$set(S);const W={};n&2&&(W.$$scope={dirty:n,ctx:e}),ne.$set(W);const ee={};n&2&&(ee.$$scope={dirty:n,ctx:e}),se.$set(ee);const Z={};n&2&&(Z.$$scope={dirty:n,ctx:e}),re.$set(Z);const Y={};n&2&&(Y.$$scope={dirty:n,ctx:e}),ae.$set(Y);const I={};n&2&&(I.$$scope={dirty:n,ctx:e}),ie.$set(I);const X={};n&2&&(X.$$scope={dirty:n,ctx:e}),le.$set(X);const V={};n&2&&(V.$$scope={dirty:n,ctx:e}),de.$set(V);const A={};n&2&&(A.$$scope={dirty:n,ctx:e}),ce.$set(A);const B={};n&2&&(B.$$scope={dirty:n,ctx:e}),me.$set(B);const P={};n&2&&(P.$$scope={dirty:n,ctx:e}),pe.$set(P);const q={};n&2&&(q.$$scope={dirty:n,ctx:e}),fe.$set(q);const O={};n&2&&(O.$$scope={dirty:n,ctx:e}),he.$set(O)},i(e){Ht||(y(ue.$$.fragment,e),y(ge.$$.fragment,e),y(Te.$$.fragment,e),y(ke.$$.fragment,e),y(ve.$$.fragment,e),y(oe.$$.fragment,e),y(Ne.$$.fragment,e),y($e.$$.fragment,e),y(je.$$.fragment,e),y(ne.$$.fragment,e),y(Je.$$.fragment,e),y(Ue.$$.fragment,e),y(Ce.$$.fragment,e),y(se.$$.fragment,e),y(re.$$.fragment,e),y(Fe.$$.fragment,e),y(xe.$$.fragment,e),y(ze.$$.fragment,e),y(ae.$$.fragment,e),y(ie.$$.fragment,e),y(le.$$.fragment,e),y(We.$$.fragment,e),y(Ze.$$.fragment,e),y(Ie.$$.fragment,e),y(de.$$.fragment,e),y(ce.$$.fragment,e),y(Xe.$$.fragment,e),y(Ve.$$.fragment,e),y(Be.$$.fragment,e),y(me.$$.fragment,e),y(pe.$$.fragment,e),y(qe.$$.fragment,e),y(Re.$$.fragment,e),y(Le.$$.fragment,e),y(fe.$$.fragment,e),y(he.$$.fragment,e),y(Ge.$$.fragment,e),Ht=!0)},o(e){_(ue.$$.fragment,e),_(ge.$$.fragment,e),_(Te.$$.fragment,e),_(ke.$$.fragment,e),_(ve.$$.fragment,e),_(oe.$$.fragment,e),_(Ne.$$.fragment,e),_($e.$$.fragment,e),_(je.$$.fragment,e),_(ne.$$.fragment,e),_(Je.$$.fragment,e),_(Ue.$$.fragment,e),_(Ce.$$.fragment,e),_(se.$$.fragment,e),_(re.$$.fragment,e),_(Fe.$$.fragment,e),_(xe.$$.fragment,e),_(ze.$$.fragment,e),_(ae.$$.fragment,e),_(ie.$$.fragment,e),_(le.$$.fragment,e),_(We.$$.fragment,e),_(Ze.$$.fragment,e),_(Ie.$$.fragment,e),_(de.$$.fragment,e),_(ce.$$.fragment,e),_(Xe.$$.fragment,e),_(Ve.$$.fragment,e),_(Be.$$.fragment,e),_(me.$$.fragment,e),_(pe.$$.fragment,e),_(qe.$$.fragment,e),_(Re.$$.fragment,e),_(Le.$$.fragment,e),_(fe.$$.fragment,e),_(he.$$.fragment,e),_(Ge.$$.fragment,e),Ht=!1},d(e){e&&(s(M),s(l),s(d),s(T),s(w),s(Mt),s(te),s(Tt),s(wt),s(ye),s(kt),s(_e),s(vt),s(be),s(Nt),s(Me),s($t),s(jt),s(we),s(Jt),s(Ut),s(z),s(Ct),s(Ft),s(N),s(xt),s(zt),s($),s(Wt),s(Zt),s(j),s(It),s(Xt),s(J),s(Vt),s(Bt),s(U),s(qt),s(Rt),s(C),s(Lt),s(Gt),s(bt)),s(t),b(ue,e),b(ge,e),b(Te,e),b(ke,e),b(ve),b(oe),b(Ne,e),b($e),b(je),b(ne),b(Je,e),b(Ue),b(Ce),b(se),b(re),b(Fe,e),b(xe),b(ze),b(ae),b(ie),b(le),b(We,e),b(Ze),b(Ie),b(de),b(ce),b(Xe,e),b(Ve),b(Be),b(me),b(pe),b(qe,e),b(Re),b(Le),b(fe),b(he),b(Ge,e)}}}const In='{"title":"Nyströmformer","local":"nyströmformer","sections":[{"title":"Overview","local":"overview","sections":[],"depth":2},{"title":"Resources","local":"resources","sections":[],"depth":2},{"title":"NystromformerConfig","local":"transformers.NystromformerConfig","sections":[],"depth":2},{"title":"NystromformerModel","local":"transformers.NystromformerModel","sections":[],"depth":2},{"title":"NystromformerForMaskedLM","local":"transformers.NystromformerForMaskedLM","sections":[],"depth":2},{"title":"NystromformerForSequenceClassification","local":"transformers.NystromformerForSequenceClassification","sections":[],"depth":2},{"title":"NystromformerForMultipleChoice","local":"transformers.NystromformerForMultipleChoice","sections":[],"depth":2},{"title":"NystromformerForTokenClassification","local":"transformers.NystromformerForTokenClassification","sections":[],"depth":2},{"title":"NystromformerForQuestionAnswering","local":"transformers.NystromformerForQuestionAnswering","sections":[],"depth":2}],"depth":1}';function Xn(k){return yn(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class Qn extends _n{constructor(t){super(),bn(this,t,Xn,Zn,gn,{})}}export{Qn as component};
