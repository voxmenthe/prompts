import{s as Cn,z as Jn,o as Un,n as V}from"../chunks/scheduler.18a86fab.js";import{S as Yn,i as zn,g as p,s as a,r as u,A as xn,h as m,f as s,c as r,j as z,x as h,u as f,k as v,y as i,a as c,v as g,d as _,t as b,w as y}from"../chunks/index.98837b22.js";import{T as wt}from"../chunks/Tip.77304350.js";import{D as R}from"../chunks/Docstring.a1ef7999.js";import{C as Ee}from"../chunks/CodeBlock.8d0c2e8a.js";import{E as Ae}from"../chunks/ExampleCodeBlock.8c3ee1f9.js";import{H as D,E as Fn}from"../chunks/getInferenceSnippets.06c2775f.js";function Wn(k){let t,M="Example:",l,d,T;return d=new Ee({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMFlvc29Db25maWclMkMlMjBZb3NvTW9kZWwlMEElMEElMjMlMjBJbml0aWFsaXppbmclMjBhJTIwWU9TTyUyMHV3LW1hZGlzb24lMkZ5b3NvLTQwOTYlMjBzdHlsZSUyMGNvbmZpZ3VyYXRpb24lMEFjb25maWd1cmF0aW9uJTIwJTNEJTIwWW9zb0NvbmZpZygpJTBBJTBBJTIzJTIwSW5pdGlhbGl6aW5nJTIwYSUyMG1vZGVsJTIwKHdpdGglMjByYW5kb20lMjB3ZWlnaHRzKSUyMGZyb20lMjB0aGUlMjB1dy1tYWRpc29uJTJGeW9zby00MDk2JTIwc3R5bGUlMjBjb25maWd1cmF0aW9uJTBBbW9kZWwlMjAlM0QlMjBZb3NvTW9kZWwoY29uZmlndXJhdGlvbiklMEElMEElMjMlMjBBY2Nlc3NpbmclMjB0aGUlMjBtb2RlbCUyMGNvbmZpZ3VyYXRpb24lMEFjb25maWd1cmF0aW9uJTIwJTNEJTIwbW9kZWwuY29uZmln",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> YosoConfig, YosoModel

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a YOSO uw-madison/yoso-4096 style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = YosoConfig()

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a model (with random weights) from the uw-madison/yoso-4096 style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = YosoModel(configuration)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Accessing the model configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = model.config`,wrap:!1}}),{c(){t=p("p"),t.textContent=M,l=a(),u(d.$$.fragment)},l(o){t=m(o,"P",{"data-svelte-h":!0}),h(t)!=="svelte-11lpom8"&&(t.textContent=M),l=r(o),f(d.$$.fragment,o)},m(o,w){c(o,t,w),c(o,l,w),g(d,o,w),T=!0},p:V,i(o){T||(_(d.$$.fragment,o),T=!0)},o(o){b(d.$$.fragment,o),T=!1},d(o){o&&(s(t),s(l)),y(d,o)}}}function Zn(k){let t,M=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=p("p"),t.innerHTML=M},l(l){t=m(l,"P",{"data-svelte-h":!0}),h(t)!=="svelte-fincs2"&&(t.innerHTML=M)},m(l,d){c(l,t,d)},p:V,d(l){l&&s(t)}}}function Bn(k){let t,M=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=p("p"),t.innerHTML=M},l(l){t=m(l,"P",{"data-svelte-h":!0}),h(t)!=="svelte-fincs2"&&(t.innerHTML=M)},m(l,d){c(l,t,d)},p:V,d(l){l&&s(t)}}}function In(k){let t,M="Example:",l,d,T;return d=new Ee({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBZb3NvRm9yTWFza2VkTE0lMEFpbXBvcnQlMjB0b3JjaCUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMnV3LW1hZGlzb24lMkZ5b3NvLTQwOTYlMjIpJTBBbW9kZWwlMjAlM0QlMjBZb3NvRm9yTWFza2VkTE0uZnJvbV9wcmV0cmFpbmVkKCUyMnV3LW1hZGlzb24lMkZ5b3NvLTQwOTYlMjIpJTBBJTBBaW5wdXRzJTIwJTNEJTIwdG9rZW5pemVyKCUyMlRoZSUyMGNhcGl0YWwlMjBvZiUyMEZyYW5jZSUyMGlzJTIwJTNDbWFzayUzRS4lMjIlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQSUwQXdpdGglMjB0b3JjaC5ub19ncmFkKCklM0ElMEElMjAlMjAlMjAlMjBsb2dpdHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cykubG9naXRzJTBBJTBBJTIzJTIwcmV0cmlldmUlMjBpbmRleCUyMG9mJTIwJTNDbWFzayUzRSUwQW1hc2tfdG9rZW5faW5kZXglMjAlM0QlMjAoaW5wdXRzLmlucHV0X2lkcyUyMCUzRCUzRCUyMHRva2VuaXplci5tYXNrX3Rva2VuX2lkKSU1QjAlNUQubm9uemVybyhhc190dXBsZSUzRFRydWUpJTVCMCU1RCUwQSUwQXByZWRpY3RlZF90b2tlbl9pZCUyMCUzRCUyMGxvZ2l0cyU1QjAlMkMlMjBtYXNrX3Rva2VuX2luZGV4JTVELmFyZ21heChheGlzJTNELTEpJTBBdG9rZW5pemVyLmRlY29kZShwcmVkaWN0ZWRfdG9rZW5faWQpJTBBJTBBbGFiZWxzJTIwJTNEJTIwdG9rZW5pemVyKCUyMlRoZSUyMGNhcGl0YWwlMjBvZiUyMEZyYW5jZSUyMGlzJTIwUGFyaXMuJTIyJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiklNUIlMjJpbnB1dF9pZHMlMjIlNUQlMEElMjMlMjBtYXNrJTIwbGFiZWxzJTIwb2YlMjBub24tJTNDbWFzayUzRSUyMHRva2VucyUwQWxhYmVscyUyMCUzRCUyMHRvcmNoLndoZXJlKGlucHV0cy5pbnB1dF9pZHMlMjAlM0QlM0QlMjB0b2tlbml6ZXIubWFza190b2tlbl9pZCUyQyUyMGxhYmVscyUyQyUyMC0xMDApJTBBJTBBb3V0cHV0cyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzJTJDJTIwbGFiZWxzJTNEbGFiZWxzKSUwQXJvdW5kKG91dHB1dHMubG9zcy5pdGVtKCklMkMlMjAyKQ==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, YosoForMaskedLM
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;uw-madison/yoso-4096&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = YosoForMaskedLM.from_pretrained(<span class="hljs-string">&quot;uw-madison/yoso-4096&quot;</span>)

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
...`,wrap:!1}}),{c(){t=p("p"),t.textContent=M,l=a(),u(d.$$.fragment)},l(o){t=m(o,"P",{"data-svelte-h":!0}),h(t)!=="svelte-11lpom8"&&(t.textContent=M),l=r(o),f(d.$$.fragment,o)},m(o,w){c(o,t,w),c(o,l,w),g(d,o,w),T=!0},p:V,i(o){T||(_(d.$$.fragment,o),T=!0)},o(o){b(d.$$.fragment,o),T=!1},d(o){o&&(s(t),s(l)),y(d,o)}}}function Nn(k){let t,M=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=p("p"),t.innerHTML=M},l(l){t=m(l,"P",{"data-svelte-h":!0}),h(t)!=="svelte-fincs2"&&(t.innerHTML=M)},m(l,d){c(l,t,d)},p:V,d(l){l&&s(t)}}}function Gn(k){let t,M="Example of single-label classification:",l,d,T;return d=new Ee({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Rva2VuaXplciUyQyUyMFlvc29Gb3JTZXF1ZW5jZUNsYXNzaWZpY2F0aW9uJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIydXctbWFkaXNvbiUyRnlvc28tNDA5NiUyMiklMEFtb2RlbCUyMCUzRCUyMFlvc29Gb3JTZXF1ZW5jZUNsYXNzaWZpY2F0aW9uLmZyb21fcHJldHJhaW5lZCglMjJ1dy1tYWRpc29uJTJGeW9zby00MDk2JTIyKSUwQSUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplciglMjJIZWxsbyUyQyUyMG15JTIwZG9nJTIwaXMlMjBjdXRlJTIyJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiklMEElMEF3aXRoJTIwdG9yY2gubm9fZ3JhZCgpJTNBJTBBJTIwJTIwJTIwJTIwbG9naXRzJTIwJTNEJTIwbW9kZWwoKippbnB1dHMpLmxvZ2l0cyUwQSUwQXByZWRpY3RlZF9jbGFzc19pZCUyMCUzRCUyMGxvZ2l0cy5hcmdtYXgoKS5pdGVtKCklMEFtb2RlbC5jb25maWcuaWQybGFiZWwlNUJwcmVkaWN0ZWRfY2xhc3NfaWQlNUQlMEElMEElMjMlMjBUbyUyMHRyYWluJTIwYSUyMG1vZGVsJTIwb24lMjAlNjBudW1fbGFiZWxzJTYwJTIwY2xhc3NlcyUyQyUyMHlvdSUyMGNhbiUyMHBhc3MlMjAlNjBudW1fbGFiZWxzJTNEbnVtX2xhYmVscyU2MCUyMHRvJTIwJTYwLmZyb21fcHJldHJhaW5lZCguLi4pJTYwJTBBbnVtX2xhYmVscyUyMCUzRCUyMGxlbihtb2RlbC5jb25maWcuaWQybGFiZWwpJTBBbW9kZWwlMjAlM0QlMjBZb3NvRm9yU2VxdWVuY2VDbGFzc2lmaWNhdGlvbi5mcm9tX3ByZXRyYWluZWQoJTIydXctbWFkaXNvbiUyRnlvc28tNDA5NiUyMiUyQyUyMG51bV9sYWJlbHMlM0RudW1fbGFiZWxzKSUwQSUwQWxhYmVscyUyMCUzRCUyMHRvcmNoLnRlbnNvciglNUIxJTVEKSUwQWxvc3MlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyUyQyUyMGxhYmVscyUzRGxhYmVscykubG9zcyUwQXJvdW5kKGxvc3MuaXRlbSgpJTJDJTIwMik=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, YosoForSequenceClassification

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;uw-madison/yoso-4096&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = YosoForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;uw-madison/yoso-4096&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    logits = model(**inputs).logits

<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_class_id = logits.argmax().item()
<span class="hljs-meta">&gt;&gt;&gt; </span>model.config.id2label[predicted_class_id]
...

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># To train a model on \`num_labels\` classes, you can pass \`num_labels=num_labels\` to \`.from_pretrained(...)\`</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>num_labels = <span class="hljs-built_in">len</span>(model.config.id2label)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = YosoForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;uw-madison/yoso-4096&quot;</span>, num_labels=num_labels)

<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.tensor([<span class="hljs-number">1</span>])
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = model(**inputs, labels=labels).loss
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">round</span>(loss.item(), <span class="hljs-number">2</span>)
...`,wrap:!1}}),{c(){t=p("p"),t.textContent=M,l=a(),u(d.$$.fragment)},l(o){t=m(o,"P",{"data-svelte-h":!0}),h(t)!=="svelte-ykxpe4"&&(t.textContent=M),l=r(o),f(d.$$.fragment,o)},m(o,w){c(o,t,w),c(o,l,w),g(d,o,w),T=!0},p:V,i(o){T||(_(d.$$.fragment,o),T=!0)},o(o){b(d.$$.fragment,o),T=!1},d(o){o&&(s(t),s(l)),y(d,o)}}}function qn(k){let t,M="Example of multi-label classification:",l,d,T;return d=new Ee({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Rva2VuaXplciUyQyUyMFlvc29Gb3JTZXF1ZW5jZUNsYXNzaWZpY2F0aW9uJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIydXctbWFkaXNvbiUyRnlvc28tNDA5NiUyMiklMEFtb2RlbCUyMCUzRCUyMFlvc29Gb3JTZXF1ZW5jZUNsYXNzaWZpY2F0aW9uLmZyb21fcHJldHJhaW5lZCglMjJ1dy1tYWRpc29uJTJGeW9zby00MDk2JTIyJTJDJTIwcHJvYmxlbV90eXBlJTNEJTIybXVsdGlfbGFiZWxfY2xhc3NpZmljYXRpb24lMjIpJTBBJTBBaW5wdXRzJTIwJTNEJTIwdG9rZW5pemVyKCUyMkhlbGxvJTJDJTIwbXklMjBkb2clMjBpcyUyMGN1dGUlMjIlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQSUwQXdpdGglMjB0b3JjaC5ub19ncmFkKCklM0ElMEElMjAlMjAlMjAlMjBsb2dpdHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cykubG9naXRzJTBBJTBBcHJlZGljdGVkX2NsYXNzX2lkcyUyMCUzRCUyMHRvcmNoLmFyYW5nZSgwJTJDJTIwbG9naXRzLnNoYXBlJTVCLTElNUQpJTVCdG9yY2guc2lnbW9pZChsb2dpdHMpLnNxdWVlemUoZGltJTNEMCklMjAlM0UlMjAwLjUlNUQlMEElMEElMjMlMjBUbyUyMHRyYWluJTIwYSUyMG1vZGVsJTIwb24lMjAlNjBudW1fbGFiZWxzJTYwJTIwY2xhc3NlcyUyQyUyMHlvdSUyMGNhbiUyMHBhc3MlMjAlNjBudW1fbGFiZWxzJTNEbnVtX2xhYmVscyU2MCUyMHRvJTIwJTYwLmZyb21fcHJldHJhaW5lZCguLi4pJTYwJTBBbnVtX2xhYmVscyUyMCUzRCUyMGxlbihtb2RlbC5jb25maWcuaWQybGFiZWwpJTBBbW9kZWwlMjAlM0QlMjBZb3NvRm9yU2VxdWVuY2VDbGFzc2lmaWNhdGlvbi5mcm9tX3ByZXRyYWluZWQoJTBBJTIwJTIwJTIwJTIwJTIydXctbWFkaXNvbiUyRnlvc28tNDA5NiUyMiUyQyUyMG51bV9sYWJlbHMlM0RudW1fbGFiZWxzJTJDJTIwcHJvYmxlbV90eXBlJTNEJTIybXVsdGlfbGFiZWxfY2xhc3NpZmljYXRpb24lMjIlMEEpJTBBJTBBbGFiZWxzJTIwJTNEJTIwdG9yY2guc3VtKCUwQSUyMCUyMCUyMCUyMHRvcmNoLm5uLmZ1bmN0aW9uYWwub25lX2hvdChwcmVkaWN0ZWRfY2xhc3NfaWRzJTVCTm9uZSUyQyUyMCUzQSU1RC5jbG9uZSgpJTJDJTIwbnVtX2NsYXNzZXMlM0RudW1fbGFiZWxzKSUyQyUyMGRpbSUzRDElMEEpLnRvKHRvcmNoLmZsb2F0KSUwQWxvc3MlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyUyQyUyMGxhYmVscyUzRGxhYmVscykubG9zcw==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, YosoForSequenceClassification

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;uw-madison/yoso-4096&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = YosoForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;uw-madison/yoso-4096&quot;</span>, problem_type=<span class="hljs-string">&quot;multi_label_classification&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    logits = model(**inputs).logits

<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_class_ids = torch.arange(<span class="hljs-number">0</span>, logits.shape[-<span class="hljs-number">1</span>])[torch.sigmoid(logits).squeeze(dim=<span class="hljs-number">0</span>) &gt; <span class="hljs-number">0.5</span>]

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># To train a model on \`num_labels\` classes, you can pass \`num_labels=num_labels\` to \`.from_pretrained(...)\`</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>num_labels = <span class="hljs-built_in">len</span>(model.config.id2label)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = YosoForSequenceClassification.from_pretrained(
<span class="hljs-meta">... </span>    <span class="hljs-string">&quot;uw-madison/yoso-4096&quot;</span>, num_labels=num_labels, problem_type=<span class="hljs-string">&quot;multi_label_classification&quot;</span>
<span class="hljs-meta">... </span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.<span class="hljs-built_in">sum</span>(
<span class="hljs-meta">... </span>    torch.nn.functional.one_hot(predicted_class_ids[<span class="hljs-literal">None</span>, :].clone(), num_classes=num_labels), dim=<span class="hljs-number">1</span>
<span class="hljs-meta">... </span>).to(torch.<span class="hljs-built_in">float</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = model(**inputs, labels=labels).loss`,wrap:!1}}),{c(){t=p("p"),t.textContent=M,l=a(),u(d.$$.fragment)},l(o){t=m(o,"P",{"data-svelte-h":!0}),h(t)!=="svelte-1l8e32d"&&(t.textContent=M),l=r(o),f(d.$$.fragment,o)},m(o,w){c(o,t,w),c(o,l,w),g(d,o,w),T=!0},p:V,i(o){T||(_(d.$$.fragment,o),T=!0)},o(o){b(d.$$.fragment,o),T=!1},d(o){o&&(s(t),s(l)),y(d,o)}}}function Rn(k){let t,M=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=p("p"),t.innerHTML=M},l(l){t=m(l,"P",{"data-svelte-h":!0}),h(t)!=="svelte-fincs2"&&(t.innerHTML=M)},m(l,d){c(l,t,d)},p:V,d(l){l&&s(t)}}}function Vn(k){let t,M="Example:",l,d,T;return d=new Ee({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBZb3NvRm9yTXVsdGlwbGVDaG9pY2UlMEFpbXBvcnQlMjB0b3JjaCUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMnV3LW1hZGlzb24lMkZ5b3NvLTQwOTYlMjIpJTBBbW9kZWwlMjAlM0QlMjBZb3NvRm9yTXVsdGlwbGVDaG9pY2UuZnJvbV9wcmV0cmFpbmVkKCUyMnV3LW1hZGlzb24lMkZ5b3NvLTQwOTYlMjIpJTBBJTBBcHJvbXB0JTIwJTNEJTIwJTIySW4lMjBJdGFseSUyQyUyMHBpenphJTIwc2VydmVkJTIwaW4lMjBmb3JtYWwlMjBzZXR0aW5ncyUyQyUyMHN1Y2glMjBhcyUyMGF0JTIwYSUyMHJlc3RhdXJhbnQlMkMlMjBpcyUyMHByZXNlbnRlZCUyMHVuc2xpY2VkLiUyMiUwQWNob2ljZTAlMjAlM0QlMjAlMjJJdCUyMGlzJTIwZWF0ZW4lMjB3aXRoJTIwYSUyMGZvcmslMjBhbmQlMjBhJTIwa25pZmUuJTIyJTBBY2hvaWNlMSUyMCUzRCUyMCUyMkl0JTIwaXMlMjBlYXRlbiUyMHdoaWxlJTIwaGVsZCUyMGluJTIwdGhlJTIwaGFuZC4lMjIlMEFsYWJlbHMlMjAlM0QlMjB0b3JjaC50ZW5zb3IoMCkudW5zcXVlZXplKDApJTIwJTIwJTIzJTIwY2hvaWNlMCUyMGlzJTIwY29ycmVjdCUyMChhY2NvcmRpbmclMjB0byUyMFdpa2lwZWRpYSUyMCUzQikpJTJDJTIwYmF0Y2glMjBzaXplJTIwMSUwQSUwQWVuY29kaW5nJTIwJTNEJTIwdG9rZW5pemVyKCU1QnByb21wdCUyQyUyMHByb21wdCU1RCUyQyUyMCU1QmNob2ljZTAlMkMlMjBjaG9pY2UxJTVEJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiUyQyUyMHBhZGRpbmclM0RUcnVlKSUwQW91dHB1dHMlMjAlM0QlMjBtb2RlbCgqKiU3QmslM0ElMjB2LnVuc3F1ZWV6ZSgwKSUyMGZvciUyMGslMkMlMjB2JTIwaW4lMjBlbmNvZGluZy5pdGVtcygpJTdEJTJDJTIwbGFiZWxzJTNEbGFiZWxzKSUyMCUyMCUyMyUyMGJhdGNoJTIwc2l6ZSUyMGlzJTIwMSUwQSUwQSUyMyUyMHRoZSUyMGxpbmVhciUyMGNsYXNzaWZpZXIlMjBzdGlsbCUyMG5lZWRzJTIwdG8lMjBiZSUyMHRyYWluZWQlMEFsb3NzJTIwJTNEJTIwb3V0cHV0cy5sb3NzJTBBbG9naXRzJTIwJTNEJTIwb3V0cHV0cy5sb2dpdHM=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, YosoForMultipleChoice
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;uw-madison/yoso-4096&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = YosoForMultipleChoice.from_pretrained(<span class="hljs-string">&quot;uw-madison/yoso-4096&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>prompt = <span class="hljs-string">&quot;In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>choice0 = <span class="hljs-string">&quot;It is eaten with a fork and a knife.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>choice1 = <span class="hljs-string">&quot;It is eaten while held in the hand.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.tensor(<span class="hljs-number">0</span>).unsqueeze(<span class="hljs-number">0</span>)  <span class="hljs-comment"># choice0 is correct (according to Wikipedia ;)), batch size 1</span>

<span class="hljs-meta">&gt;&gt;&gt; </span>encoding = tokenizer([prompt, prompt], [choice0, choice1], return_tensors=<span class="hljs-string">&quot;pt&quot;</span>, padding=<span class="hljs-literal">True</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**{k: v.unsqueeze(<span class="hljs-number">0</span>) <span class="hljs-keyword">for</span> k, v <span class="hljs-keyword">in</span> encoding.items()}, labels=labels)  <span class="hljs-comment"># batch size is 1</span>

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># the linear classifier still needs to be trained</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = outputs.loss
<span class="hljs-meta">&gt;&gt;&gt; </span>logits = outputs.logits`,wrap:!1}}),{c(){t=p("p"),t.textContent=M,l=a(),u(d.$$.fragment)},l(o){t=m(o,"P",{"data-svelte-h":!0}),h(t)!=="svelte-11lpom8"&&(t.textContent=M),l=r(o),f(d.$$.fragment,o)},m(o,w){c(o,t,w),c(o,l,w),g(d,o,w),T=!0},p:V,i(o){T||(_(d.$$.fragment,o),T=!0)},o(o){b(d.$$.fragment,o),T=!1},d(o){o&&(s(t),s(l)),y(d,o)}}}function Xn(k){let t,M=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=p("p"),t.innerHTML=M},l(l){t=m(l,"P",{"data-svelte-h":!0}),h(t)!=="svelte-fincs2"&&(t.innerHTML=M)},m(l,d){c(l,t,d)},p:V,d(l){l&&s(t)}}}function Sn(k){let t,M="Example:",l,d,T;return d=new Ee({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBZb3NvRm9yVG9rZW5DbGFzc2lmaWNhdGlvbiUwQWltcG9ydCUyMHRvcmNoJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIydXctbWFkaXNvbiUyRnlvc28tNDA5NiUyMiklMEFtb2RlbCUyMCUzRCUyMFlvc29Gb3JUb2tlbkNsYXNzaWZpY2F0aW9uLmZyb21fcHJldHJhaW5lZCglMjJ1dy1tYWRpc29uJTJGeW9zby00MDk2JTIyKSUwQSUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplciglMEElMjAlMjAlMjAlMjAlMjJIdWdnaW5nRmFjZSUyMGlzJTIwYSUyMGNvbXBhbnklMjBiYXNlZCUyMGluJTIwUGFyaXMlMjBhbmQlMjBOZXclMjBZb3JrJTIyJTJDJTIwYWRkX3NwZWNpYWxfdG9rZW5zJTNERmFsc2UlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyJTBBKSUwQSUwQXdpdGglMjB0b3JjaC5ub19ncmFkKCklM0ElMEElMjAlMjAlMjAlMjBsb2dpdHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cykubG9naXRzJTBBJTBBcHJlZGljdGVkX3Rva2VuX2NsYXNzX2lkcyUyMCUzRCUyMGxvZ2l0cy5hcmdtYXgoLTEpJTBBJTBBJTIzJTIwTm90ZSUyMHRoYXQlMjB0b2tlbnMlMjBhcmUlMjBjbGFzc2lmaWVkJTIwcmF0aGVyJTIwdGhlbiUyMGlucHV0JTIwd29yZHMlMjB3aGljaCUyMG1lYW5zJTIwdGhhdCUwQSUyMyUyMHRoZXJlJTIwbWlnaHQlMjBiZSUyMG1vcmUlMjBwcmVkaWN0ZWQlMjB0b2tlbiUyMGNsYXNzZXMlMjB0aGFuJTIwd29yZHMuJTBBJTIzJTIwTXVsdGlwbGUlMjB0b2tlbiUyMGNsYXNzZXMlMjBtaWdodCUyMGFjY291bnQlMjBmb3IlMjB0aGUlMjBzYW1lJTIwd29yZCUwQXByZWRpY3RlZF90b2tlbnNfY2xhc3NlcyUyMCUzRCUyMCU1Qm1vZGVsLmNvbmZpZy5pZDJsYWJlbCU1QnQuaXRlbSgpJTVEJTIwZm9yJTIwdCUyMGluJTIwcHJlZGljdGVkX3Rva2VuX2NsYXNzX2lkcyU1QjAlNUQlNUQlMEFwcmVkaWN0ZWRfdG9rZW5zX2NsYXNzZXMlMEElMEFsYWJlbHMlMjAlM0QlMjBwcmVkaWN0ZWRfdG9rZW5fY2xhc3NfaWRzJTBBbG9zcyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzJTJDJTIwbGFiZWxzJTNEbGFiZWxzKS5sb3NzJTBBcm91bmQobG9zcy5pdGVtKCklMkMlMjAyKQ==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, YosoForTokenClassification
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;uw-madison/yoso-4096&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = YosoForTokenClassification.from_pretrained(<span class="hljs-string">&quot;uw-madison/yoso-4096&quot;</span>)

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
...`,wrap:!1}}),{c(){t=p("p"),t.textContent=M,l=a(),u(d.$$.fragment)},l(o){t=m(o,"P",{"data-svelte-h":!0}),h(t)!=="svelte-11lpom8"&&(t.textContent=M),l=r(o),f(d.$$.fragment,o)},m(o,w){c(o,t,w),c(o,l,w),g(d,o,w),T=!0},p:V,i(o){T||(_(d.$$.fragment,o),T=!0)},o(o){b(d.$$.fragment,o),T=!1},d(o){o&&(s(t),s(l)),y(d,o)}}}function Hn(k){let t,M=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=p("p"),t.innerHTML=M},l(l){t=m(l,"P",{"data-svelte-h":!0}),h(t)!=="svelte-fincs2"&&(t.innerHTML=M)},m(l,d){c(l,t,d)},p:V,d(l){l&&s(t)}}}function Ln(k){let t,M="Example:",l,d,T;return d=new Ee({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBZb3NvRm9yUXVlc3Rpb25BbnN3ZXJpbmclMEFpbXBvcnQlMjB0b3JjaCUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMnV3LW1hZGlzb24lMkZ5b3NvLTQwOTYlMjIpJTBBbW9kZWwlMjAlM0QlMjBZb3NvRm9yUXVlc3Rpb25BbnN3ZXJpbmcuZnJvbV9wcmV0cmFpbmVkKCUyMnV3LW1hZGlzb24lMkZ5b3NvLTQwOTYlMjIpJTBBJTBBcXVlc3Rpb24lMkMlMjB0ZXh0JTIwJTNEJTIwJTIyV2hvJTIwd2FzJTIwSmltJTIwSGVuc29uJTNGJTIyJTJDJTIwJTIySmltJTIwSGVuc29uJTIwd2FzJTIwYSUyMG5pY2UlMjBwdXBwZXQlMjIlMEElMEFpbnB1dHMlMjAlM0QlMjB0b2tlbml6ZXIocXVlc3Rpb24lMkMlMjB0ZXh0JTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiklMEF3aXRoJTIwdG9yY2gubm9fZ3JhZCgpJTNBJTBBJTIwJTIwJTIwJTIwb3V0cHV0cyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzKSUwQSUwQWFuc3dlcl9zdGFydF9pbmRleCUyMCUzRCUyMG91dHB1dHMuc3RhcnRfbG9naXRzLmFyZ21heCgpJTBBYW5zd2VyX2VuZF9pbmRleCUyMCUzRCUyMG91dHB1dHMuZW5kX2xvZ2l0cy5hcmdtYXgoKSUwQSUwQXByZWRpY3RfYW5zd2VyX3Rva2VucyUyMCUzRCUyMGlucHV0cy5pbnB1dF9pZHMlNUIwJTJDJTIwYW5zd2VyX3N0YXJ0X2luZGV4JTIwJTNBJTIwYW5zd2VyX2VuZF9pbmRleCUyMCUyQiUyMDElNUQlMEF0b2tlbml6ZXIuZGVjb2RlKHByZWRpY3RfYW5zd2VyX3Rva2VucyUyQyUyMHNraXBfc3BlY2lhbF90b2tlbnMlM0RUcnVlKSUwQSUwQSUyMyUyMHRhcmdldCUyMGlzJTIwJTIybmljZSUyMHB1cHBldCUyMiUwQXRhcmdldF9zdGFydF9pbmRleCUyMCUzRCUyMHRvcmNoLnRlbnNvciglNUIxNCU1RCklMEF0YXJnZXRfZW5kX2luZGV4JTIwJTNEJTIwdG9yY2gudGVuc29yKCU1QjE1JTVEKSUwQSUwQW91dHB1dHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyUyQyUyMHN0YXJ0X3Bvc2l0aW9ucyUzRHRhcmdldF9zdGFydF9pbmRleCUyQyUyMGVuZF9wb3NpdGlvbnMlM0R0YXJnZXRfZW5kX2luZGV4KSUwQWxvc3MlMjAlM0QlMjBvdXRwdXRzLmxvc3MlMEFyb3VuZChsb3NzLml0ZW0oKSUyQyUyMDIp",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, YosoForQuestionAnswering
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;uw-madison/yoso-4096&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = YosoForQuestionAnswering.from_pretrained(<span class="hljs-string">&quot;uw-madison/yoso-4096&quot;</span>)

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
...`,wrap:!1}}),{c(){t=p("p"),t.textContent=M,l=a(),u(d.$$.fragment)},l(o){t=m(o,"P",{"data-svelte-h":!0}),h(t)!=="svelte-11lpom8"&&(t.textContent=M),l=r(o),f(d.$$.fragment,o)},m(o,w){c(o,t,w),c(o,l,w),g(d,o,w),T=!0},p:V,i(o){T||(_(d.$$.fragment,o),T=!0)},o(o){b(d.$$.fragment,o),T=!1},d(o){o&&(s(t),s(l)),y(d,o)}}}function Qn(k){let t,M,l,d,T,o="<em>This model was released on 2021-11-18 and added to Hugging Face Transformers on 2022-01-26.</em>",w,ge,vt,te,Ho='<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/>',$t,_e,jt,be,Lo=`The YOSO model was proposed in <a href="https://huggingface.co/papers/2111.09714" rel="nofollow">You Only Sample (Almost) Once: Linear Cost Self-Attention Via Bernoulli Sampling</a><br/>
by Zhanpeng Zeng, Yunyang Xiong, Sathya N. Ravi, Shailesh Acharya, Glenn Fung, Vikas Singh. YOSO approximates standard softmax self-attention
via a Bernoulli sampling scheme based on Locality Sensitive Hashing (LSH). In principle, all the Bernoulli random variables can be sampled with
a single hash.`,Ct,ye,Qo="The abstract from the paper is the following:",Jt,Me,Ao=`<em>Transformer-based models are widely used in natural language processing (NLP). Central to the transformer model is
the self-attention mechanism, which captures the interactions of token pairs in the input sequences and depends quadratically
on the sequence length. Training such models on longer sequences is expensive. In this paper, we show that a Bernoulli sampling
attention mechanism based on Locality Sensitive Hashing (LSH), decreases the quadratic complexity of such models to linear.
We bypass the quadratic cost by considering self-attention as a sum of individual tokens associated with Bernoulli random
variables that can, in principle, be sampled at once by a single hash (although in practice, this number may be a small constant).
This leads to an efficient sampling scheme to estimate self-attention which relies on specific modifications of
LSH (to enable deployment on GPU architectures). We evaluate our algorithm on the GLUE benchmark with standard 512 sequence
length where we see favorable performance relative to a standard pretrained Transformer. On the Long Range Arena (LRA) benchmark,
for evaluating performance on long sequences, our method achieves results consistent with softmax self-attention but with sizable
speed-ups and memory savings and often outperforms other efficient self-attention methods. Our code is available at this https URL</em>`,Ut,Te,Eo='This model was contributed by <a href="https://huggingface.co/novice03" rel="nofollow">novice03</a>. The original code can be found <a href="https://github.com/mlpen/YOSO" rel="nofollow">here</a>.',Yt,we,zt,ke,Oo=`<li>The YOSO attention algorithm is implemented through custom CUDA kernels, functions written in CUDA C++ that can be executed multiple times
in parallel on a GPU.</li> <li>The kernels provide a <code>fast_hash</code> function, which approximates the random projections of the queries and keys using the Fast Hadamard Transform. Using these
hash codes, the <code>lsh_cumulation</code> function approximates self-attention via LSH-based Bernoulli sampling.</li> <li>To use the custom kernels, the user should set <code>config.use_expectation = False</code>. To ensure that the kernels are compiled successfully,
the user must install the correct version of PyTorch and cudatoolkit. By default, <code>config.use_expectation = True</code>, which uses YOSO-E and
does not require compiling CUDA kernels.</li>`,xt,oe,Po,Ft,ve,Do='YOSO Attention Algorithm. Taken from the <a href="https://huggingface.co/papers/2111.09714">original paper</a>.',Wt,$e,Zt,je,Ko='<li><a href="../tasks/sequence_classification">Text classification task guide</a></li> <li><a href="../tasks/token_classification">Token classification task guide</a></li> <li><a href="../tasks/question_answering">Question answering task guide</a></li> <li><a href="../tasks/masked_language_modeling">Masked language modeling task guide</a></li> <li><a href="../tasks/multiple_choice">Multiple choice task guide</a></li>',Bt,Ce,It,F,Je,Kt,Oe,en=`This is the configuration class to store the configuration of a <a href="/docs/transformers/v4.56.2/en/model_doc/yoso#transformers.YosoModel">YosoModel</a>. It is used to instantiate an YOSO
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the YOSO
<a href="https://huggingface.co/uw-madison/yoso-4096" rel="nofollow">uw-madison/yoso-4096</a> architecture.`,eo,Pe,tn=`Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> and can be used to control the model outputs. Read the
documentation from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> for more information.`,to,ne,Nt,Ue,Gt,$,Ye,oo,De,on="The bare Yoso Model outputting raw hidden-states without any specific head on top.",no,Ke,nn=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,so,et,sn=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,ao,K,ze,ro,tt,an='The <a href="/docs/transformers/v4.56.2/en/model_doc/yoso#transformers.YosoModel">YosoModel</a> forward method, overrides the <code>__call__</code> special method.',io,se,qt,xe,Rt,j,Fe,lo,ot,rn="The Yoso Model with a <code>language modeling</code> head on top.”",co,nt,ln=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,po,st,dn=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,mo,X,We,ho,at,cn='The <a href="/docs/transformers/v4.56.2/en/model_doc/yoso#transformers.YosoForMaskedLM">YosoForMaskedLM</a> forward method, overrides the <code>__call__</code> special method.',uo,ae,fo,re,Vt,Ze,Xt,C,Be,go,rt,pn=`YOSO Model transformer with a sequence classification/regression head on top (a linear layer on top of
the pooled output) e.g. for GLUE tasks.`,_o,it,mn=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,bo,lt,hn=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,yo,x,Ie,Mo,dt,un='The <a href="/docs/transformers/v4.56.2/en/model_doc/yoso#transformers.YosoForSequenceClassification">YosoForSequenceClassification</a> forward method, overrides the <code>__call__</code> special method.',To,ie,wo,le,ko,de,St,Ne,Ht,J,Ge,vo,ct,fn=`The Yoso Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
softmax) e.g. for RocStories/SWAG tasks.`,$o,pt,gn=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,jo,mt,_n=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Co,S,qe,Jo,ht,bn='The <a href="/docs/transformers/v4.56.2/en/model_doc/yoso#transformers.YosoForMultipleChoice">YosoForMultipleChoice</a> forward method, overrides the <code>__call__</code> special method.',Uo,ce,Yo,pe,Lt,Re,Qt,U,Ve,zo,ut,yn=`The Yoso transformer with a token classification head on top (a linear layer on top of the hidden-states
output) e.g. for Named-Entity-Recognition (NER) tasks.`,xo,ft,Mn=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Fo,gt,Tn=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Wo,H,Xe,Zo,_t,wn='The <a href="/docs/transformers/v4.56.2/en/model_doc/yoso#transformers.YosoForTokenClassification">YosoForTokenClassification</a> forward method, overrides the <code>__call__</code> special method.',Bo,me,Io,he,At,Se,Et,Y,He,No,bt,kn=`The Yoso transformer with a span classification head on top for extractive question-answering tasks like
SQuAD (a linear layer on top of the hidden-states output to compute <code>span start logits</code> and <code>span end logits</code>).`,Go,yt,vn=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,qo,Mt,$n=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Ro,L,Le,Vo,Tt,jn='The <a href="/docs/transformers/v4.56.2/en/model_doc/yoso#transformers.YosoForQuestionAnswering">YosoForQuestionAnswering</a> forward method, overrides the <code>__call__</code> special method.',Xo,ue,So,fe,Ot,Qe,Pt,kt,Dt;return ge=new D({props:{title:"YOSO",local:"yoso",headingTag:"h1"}}),_e=new D({props:{title:"Overview",local:"overview",headingTag:"h2"}}),we=new D({props:{title:"Usage tips",local:"usage-tips",headingTag:"h2"}}),$e=new D({props:{title:"Resources",local:"resources",headingTag:"h2"}}),Ce=new D({props:{title:"YosoConfig",local:"transformers.YosoConfig",headingTag:"h2"}}),Je=new R({props:{name:"class transformers.YosoConfig",anchor:"transformers.YosoConfig",parameters:[{name:"vocab_size",val:" = 50265"},{name:"hidden_size",val:" = 768"},{name:"num_hidden_layers",val:" = 12"},{name:"num_attention_heads",val:" = 12"},{name:"intermediate_size",val:" = 3072"},{name:"hidden_act",val:" = 'gelu'"},{name:"hidden_dropout_prob",val:" = 0.1"},{name:"attention_probs_dropout_prob",val:" = 0.1"},{name:"max_position_embeddings",val:" = 4096"},{name:"type_vocab_size",val:" = 1"},{name:"initializer_range",val:" = 0.02"},{name:"layer_norm_eps",val:" = 1e-12"},{name:"position_embedding_type",val:" = 'absolute'"},{name:"use_expectation",val:" = True"},{name:"hash_code_len",val:" = 9"},{name:"num_hash",val:" = 64"},{name:"conv_window",val:" = None"},{name:"use_fast_hash",val:" = True"},{name:"lsh_backward",val:" = True"},{name:"pad_token_id",val:" = 1"},{name:"bos_token_id",val:" = 0"},{name:"eos_token_id",val:" = 2"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.YosoConfig.vocab_size",description:`<strong>vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 50265) &#x2014;
Vocabulary size of the YOSO model. Defines the number of different tokens that can be represented by the
<code>inputs_ids</code> passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/yoso#transformers.YosoModel">YosoModel</a>.`,name:"vocab_size"},{anchor:"transformers.YosoConfig.hidden_size",description:`<strong>hidden_size</strong> (<code>int</code>, <em>optional</em>, defaults to 768) &#x2014;
Dimension of the encoder layers and the pooler layer.`,name:"hidden_size"},{anchor:"transformers.YosoConfig.num_hidden_layers",description:`<strong>num_hidden_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 12) &#x2014;
Number of hidden layers in the Transformer encoder.`,name:"num_hidden_layers"},{anchor:"transformers.YosoConfig.num_attention_heads",description:`<strong>num_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 12) &#x2014;
Number of attention heads for each attention layer in the Transformer encoder.`,name:"num_attention_heads"},{anchor:"transformers.YosoConfig.intermediate_size",description:`<strong>intermediate_size</strong> (<code>int</code>, <em>optional</em>, defaults to 3072) &#x2014;
Dimension of the &#x201C;intermediate&#x201D; (i.e., feed-forward) layer in the Transformer encoder.`,name:"intermediate_size"},{anchor:"transformers.YosoConfig.hidden_act",description:`<strong>hidden_act</strong> (<code>str</code> or <code>function</code>, <em>optional</em>, defaults to <code>&quot;gelu&quot;</code>) &#x2014;
The non-linear activation function (function or string) in the encoder and pooler. If string, <code>&quot;gelu&quot;</code>,
<code>&quot;relu&quot;</code>, <code>&quot;selu&quot;</code> and <code>&quot;gelu_new&quot;</code> are supported.`,name:"hidden_act"},{anchor:"transformers.YosoConfig.hidden_dropout_prob",description:`<strong>hidden_dropout_prob</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.`,name:"hidden_dropout_prob"},{anchor:"transformers.YosoConfig.attention_probs_dropout_prob",description:`<strong>attention_probs_dropout_prob</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout ratio for the attention probabilities.`,name:"attention_probs_dropout_prob"},{anchor:"transformers.YosoConfig.max_position_embeddings",description:`<strong>max_position_embeddings</strong> (<code>int</code>, <em>optional</em>, defaults to 512) &#x2014;
The maximum sequence length that this model might ever be used with. Typically set this to something large
just in case (e.g., 512 or 1024 or 2048).`,name:"max_position_embeddings"},{anchor:"transformers.YosoConfig.type_vocab_size",description:`<strong>type_vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 2) &#x2014;
The vocabulary size of the <code>token_type_ids</code> passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/yoso#transformers.YosoModel">YosoModel</a>.`,name:"type_vocab_size"},{anchor:"transformers.YosoConfig.initializer_range",description:`<strong>initializer_range</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.`,name:"initializer_range"},{anchor:"transformers.YosoConfig.layer_norm_eps",description:`<strong>layer_norm_eps</strong> (<code>float</code>, <em>optional</em>, defaults to 1e-12) &#x2014;
The epsilon used by the layer normalization layers.`,name:"layer_norm_eps"},{anchor:"transformers.YosoConfig.position_embedding_type",description:`<strong>position_embedding_type</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;absolute&quot;</code>) &#x2014;
Type of position embedding. Choose one of <code>&quot;absolute&quot;</code>, <code>&quot;relative_key&quot;</code>, <code>&quot;relative_key_query&quot;</code>.`,name:"position_embedding_type"},{anchor:"transformers.YosoConfig.use_expectation",description:`<strong>use_expectation</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to use YOSO Expectation. Overrides any effect of num_hash.`,name:"use_expectation"},{anchor:"transformers.YosoConfig.hash_code_len",description:`<strong>hash_code_len</strong> (<code>int</code>, <em>optional</em>, defaults to 9) &#x2014;
The length of hashes generated by the hash functions.`,name:"hash_code_len"},{anchor:"transformers.YosoConfig.num_hash",description:`<strong>num_hash</strong> (<code>int</code>, <em>optional</em>, defaults to 64) &#x2014;
Number of hash functions used in <code>YosoSelfAttention</code>.`,name:"num_hash"},{anchor:"transformers.YosoConfig.conv_window",description:`<strong>conv_window</strong> (<code>int</code>, <em>optional</em>) &#x2014;
Kernel size of depth-wise convolution.`,name:"conv_window"},{anchor:"transformers.YosoConfig.use_fast_hash",description:`<strong>use_fast_hash</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to use custom cuda kernels which perform fast random projection via hadamard transform.`,name:"use_fast_hash"},{anchor:"transformers.YosoConfig.lsh_backward",description:`<strong>lsh_backward</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to perform backpropagation using Locality Sensitive Hashing.`,name:"lsh_backward"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/yoso/configuration_yoso.py#L24"}}),ne=new Ae({props:{anchor:"transformers.YosoConfig.example",$$slots:{default:[Wn]},$$scope:{ctx:k}}}),Ue=new D({props:{title:"YosoModel",local:"transformers.YosoModel",headingTag:"h2"}}),Ye=new R({props:{name:"class transformers.YosoModel",anchor:"transformers.YosoModel",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.YosoModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/yoso#transformers.YosoModel">YosoModel</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/yoso/modeling_yoso.py#L667"}}),ze=new R({props:{name:"forward",anchor:"transformers.YosoModel.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.YosoModel.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.YosoModel.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.YosoModel.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.YosoModel.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.YosoModel.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.YosoModel.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.YosoModel.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.YosoModel.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.YosoModel.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/yoso/modeling_yoso.py#L692",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithCrossAttentions"
>transformers.modeling_outputs.BaseModelOutputWithCrossAttentions</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/yoso#transformers.YosoConfig"
>YosoConfig</a>) and inputs.</p>
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
<li>
<p><strong>cross_attentions</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> and <code>config.add_cross_attention=True</code> is passed or when <code>config.output_attentions=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the
weighted average in the cross-attention heads.</p>
</li>
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithCrossAttentions"
>transformers.modeling_outputs.BaseModelOutputWithCrossAttentions</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),se=new wt({props:{$$slots:{default:[Zn]},$$scope:{ctx:k}}}),xe=new D({props:{title:"YosoForMaskedLM",local:"transformers.YosoForMaskedLM",headingTag:"h2"}}),Fe=new R({props:{name:"class transformers.YosoForMaskedLM",anchor:"transformers.YosoForMaskedLM",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.YosoForMaskedLM.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/yoso#transformers.YosoForMaskedLM">YosoForMaskedLM</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/yoso/modeling_yoso.py#L770"}}),We=new R({props:{name:"forward",anchor:"transformers.YosoForMaskedLM.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"labels",val:": typing.Optional[torch.Tensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.YosoForMaskedLM.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.YosoForMaskedLM.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.YosoForMaskedLM.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.YosoForMaskedLM.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.YosoForMaskedLM.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.YosoForMaskedLM.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.YosoForMaskedLM.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should be in <code>[-100, 0, ..., config.vocab_size]</code> (see <code>input_ids</code> docstring) Tokens with indices set to <code>-100</code> are ignored (masked), the
loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.YosoForMaskedLM.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.YosoForMaskedLM.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.YosoForMaskedLM.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/yoso/modeling_yoso.py#L789",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.MaskedLMOutput"
>transformers.modeling_outputs.MaskedLMOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/yoso#transformers.YosoConfig"
>YosoConfig</a>) and inputs.</p>
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
`}}),ae=new wt({props:{$$slots:{default:[Bn]},$$scope:{ctx:k}}}),re=new Ae({props:{anchor:"transformers.YosoForMaskedLM.forward.example",$$slots:{default:[In]},$$scope:{ctx:k}}}),Ze=new D({props:{title:"YosoForSequenceClassification",local:"transformers.YosoForSequenceClassification",headingTag:"h2"}}),Be=new R({props:{name:"class transformers.YosoForSequenceClassification",anchor:"transformers.YosoForSequenceClassification",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.YosoForSequenceClassification.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/yoso#transformers.YosoForSequenceClassification">YosoForSequenceClassification</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/yoso/modeling_yoso.py#L870"}}),Ie=new R({props:{name:"forward",anchor:"transformers.YosoForSequenceClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"labels",val:": typing.Optional[torch.Tensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.YosoForSequenceClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.YosoForSequenceClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.YosoForSequenceClassification.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.YosoForSequenceClassification.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.YosoForSequenceClassification.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.YosoForSequenceClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.YosoForSequenceClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the sequence classification/regression loss. Indices should be in <code>[0, ..., config.num_labels - 1]</code>. If <code>config.num_labels == 1</code> a regression loss is computed (Mean-Square loss), If
<code>config.num_labels &gt; 1</code> a classification loss is computed (Cross-Entropy).`,name:"labels"},{anchor:"transformers.YosoForSequenceClassification.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.YosoForSequenceClassification.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.YosoForSequenceClassification.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/yoso/modeling_yoso.py#L880",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput"
>transformers.modeling_outputs.SequenceClassifierOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/yoso#transformers.YosoConfig"
>YosoConfig</a>) and inputs.</p>
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
`}}),ie=new wt({props:{$$slots:{default:[Nn]},$$scope:{ctx:k}}}),le=new Ae({props:{anchor:"transformers.YosoForSequenceClassification.forward.example",$$slots:{default:[Gn]},$$scope:{ctx:k}}}),de=new Ae({props:{anchor:"transformers.YosoForSequenceClassification.forward.example-2",$$slots:{default:[qn]},$$scope:{ctx:k}}}),Ne=new D({props:{title:"YosoForMultipleChoice",local:"transformers.YosoForMultipleChoice",headingTag:"h2"}}),Ge=new R({props:{name:"class transformers.YosoForMultipleChoice",anchor:"transformers.YosoForMultipleChoice",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.YosoForMultipleChoice.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/yoso#transformers.YosoForMultipleChoice">YosoForMultipleChoice</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/yoso/modeling_yoso.py#L952"}}),qe=new R({props:{name:"forward",anchor:"transformers.YosoForMultipleChoice.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"labels",val:": typing.Optional[torch.Tensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.YosoForMultipleChoice.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, num_choices, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.YosoForMultipleChoice.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.YosoForMultipleChoice.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, num_choices, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.YosoForMultipleChoice.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, num_choices, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.max_position_embeddings - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.YosoForMultipleChoice.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.YosoForMultipleChoice.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_choices, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <em>input_ids</em> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.YosoForMultipleChoice.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the multiple choice classification loss. Indices should be in <code>[0, ..., num_choices-1]</code> where <code>num_choices</code> is the size of the second dimension of the input tensors. (See
<code>input_ids</code> above)`,name:"labels"},{anchor:"transformers.YosoForMultipleChoice.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.YosoForMultipleChoice.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.YosoForMultipleChoice.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/yoso/modeling_yoso.py#L963",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.MultipleChoiceModelOutput"
>transformers.modeling_outputs.MultipleChoiceModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/yoso#transformers.YosoConfig"
>YosoConfig</a>) and inputs.</p>
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
`}}),ce=new wt({props:{$$slots:{default:[Rn]},$$scope:{ctx:k}}}),pe=new Ae({props:{anchor:"transformers.YosoForMultipleChoice.forward.example",$$slots:{default:[Vn]},$$scope:{ctx:k}}}),Re=new D({props:{title:"YosoForTokenClassification",local:"transformers.YosoForTokenClassification",headingTag:"h2"}}),Ve=new R({props:{name:"class transformers.YosoForTokenClassification",anchor:"transformers.YosoForTokenClassification",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.YosoForTokenClassification.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/yoso#transformers.YosoForTokenClassification">YosoForTokenClassification</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/yoso/modeling_yoso.py#L1058"}}),Xe=new R({props:{name:"forward",anchor:"transformers.YosoForTokenClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"labels",val:": typing.Optional[torch.Tensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.YosoForTokenClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.YosoForTokenClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.YosoForTokenClassification.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.YosoForTokenClassification.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.YosoForTokenClassification.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.YosoForTokenClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.YosoForTokenClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the token classification loss. Indices should be in <code>[0, ..., config.num_labels - 1]</code>.`,name:"labels"},{anchor:"transformers.YosoForTokenClassification.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.YosoForTokenClassification.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.YosoForTokenClassification.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/yoso/modeling_yoso.py#L1070",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput"
>transformers.modeling_outputs.TokenClassifierOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/yoso#transformers.YosoConfig"
>YosoConfig</a>) and inputs.</p>
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
`}}),me=new wt({props:{$$slots:{default:[Xn]},$$scope:{ctx:k}}}),he=new Ae({props:{anchor:"transformers.YosoForTokenClassification.forward.example",$$slots:{default:[Sn]},$$scope:{ctx:k}}}),Se=new D({props:{title:"YosoForQuestionAnswering",local:"transformers.YosoForQuestionAnswering",headingTag:"h2"}}),He=new R({props:{name:"class transformers.YosoForQuestionAnswering",anchor:"transformers.YosoForQuestionAnswering",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.YosoForQuestionAnswering.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/yoso#transformers.YosoForQuestionAnswering">YosoForQuestionAnswering</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/yoso/modeling_yoso.py#L1134"}}),Le=new R({props:{name:"forward",anchor:"transformers.YosoForQuestionAnswering.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"start_positions",val:": typing.Optional[torch.Tensor] = None"},{name:"end_positions",val:": typing.Optional[torch.Tensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.YosoForQuestionAnswering.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.YosoForQuestionAnswering.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.YosoForQuestionAnswering.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.YosoForQuestionAnswering.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.YosoForQuestionAnswering.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.YosoForQuestionAnswering.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.YosoForQuestionAnswering.forward.start_positions",description:`<strong>start_positions</strong> (<code>torch.Tensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for position (index) of the start of the labelled span for computing the token classification loss.
Positions are clamped to the length of the sequence (<code>sequence_length</code>). Position outside of the sequence
are not taken into account for computing the loss.`,name:"start_positions"},{anchor:"transformers.YosoForQuestionAnswering.forward.end_positions",description:`<strong>end_positions</strong> (<code>torch.Tensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for position (index) of the end of the labelled span for computing the token classification loss.
Positions are clamped to the length of the sequence (<code>sequence_length</code>). Position outside of the sequence
are not taken into account for computing the loss.`,name:"end_positions"},{anchor:"transformers.YosoForQuestionAnswering.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.YosoForQuestionAnswering.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.YosoForQuestionAnswering.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/yoso/modeling_yoso.py#L1147",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.QuestionAnsweringModelOutput"
>transformers.modeling_outputs.QuestionAnsweringModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/yoso#transformers.YosoConfig"
>YosoConfig</a>) and inputs.</p>
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
`}}),ue=new wt({props:{$$slots:{default:[Hn]},$$scope:{ctx:k}}}),fe=new Ae({props:{anchor:"transformers.YosoForQuestionAnswering.forward.example",$$slots:{default:[Ln]},$$scope:{ctx:k}}}),Qe=new Fn({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/yoso.md"}}),{c(){t=p("meta"),M=a(),l=p("p"),d=a(),T=p("p"),T.innerHTML=o,w=a(),u(ge.$$.fragment),vt=a(),te=p("div"),te.innerHTML=Ho,$t=a(),u(_e.$$.fragment),jt=a(),be=p("p"),be.innerHTML=Lo,Ct=a(),ye=p("p"),ye.textContent=Qo,Jt=a(),Me=p("p"),Me.innerHTML=Ao,Ut=a(),Te=p("p"),Te.innerHTML=Eo,Yt=a(),u(we.$$.fragment),zt=a(),ke=p("ul"),ke.innerHTML=Oo,xt=a(),oe=p("img"),Ft=a(),ve=p("small"),ve.innerHTML=Do,Wt=a(),u($e.$$.fragment),Zt=a(),je=p("ul"),je.innerHTML=Ko,Bt=a(),u(Ce.$$.fragment),It=a(),F=p("div"),u(Je.$$.fragment),Kt=a(),Oe=p("p"),Oe.innerHTML=en,eo=a(),Pe=p("p"),Pe.innerHTML=tn,to=a(),u(ne.$$.fragment),Nt=a(),u(Ue.$$.fragment),Gt=a(),$=p("div"),u(Ye.$$.fragment),oo=a(),De=p("p"),De.textContent=on,no=a(),Ke=p("p"),Ke.innerHTML=nn,so=a(),et=p("p"),et.innerHTML=sn,ao=a(),K=p("div"),u(ze.$$.fragment),ro=a(),tt=p("p"),tt.innerHTML=an,io=a(),u(se.$$.fragment),qt=a(),u(xe.$$.fragment),Rt=a(),j=p("div"),u(Fe.$$.fragment),lo=a(),ot=p("p"),ot.innerHTML=rn,co=a(),nt=p("p"),nt.innerHTML=ln,po=a(),st=p("p"),st.innerHTML=dn,mo=a(),X=p("div"),u(We.$$.fragment),ho=a(),at=p("p"),at.innerHTML=cn,uo=a(),u(ae.$$.fragment),fo=a(),u(re.$$.fragment),Vt=a(),u(Ze.$$.fragment),Xt=a(),C=p("div"),u(Be.$$.fragment),go=a(),rt=p("p"),rt.textContent=pn,_o=a(),it=p("p"),it.innerHTML=mn,bo=a(),lt=p("p"),lt.innerHTML=hn,yo=a(),x=p("div"),u(Ie.$$.fragment),Mo=a(),dt=p("p"),dt.innerHTML=un,To=a(),u(ie.$$.fragment),wo=a(),u(le.$$.fragment),ko=a(),u(de.$$.fragment),St=a(),u(Ne.$$.fragment),Ht=a(),J=p("div"),u(Ge.$$.fragment),vo=a(),ct=p("p"),ct.textContent=fn,$o=a(),pt=p("p"),pt.innerHTML=gn,jo=a(),mt=p("p"),mt.innerHTML=_n,Co=a(),S=p("div"),u(qe.$$.fragment),Jo=a(),ht=p("p"),ht.innerHTML=bn,Uo=a(),u(ce.$$.fragment),Yo=a(),u(pe.$$.fragment),Lt=a(),u(Re.$$.fragment),Qt=a(),U=p("div"),u(Ve.$$.fragment),zo=a(),ut=p("p"),ut.textContent=yn,xo=a(),ft=p("p"),ft.innerHTML=Mn,Fo=a(),gt=p("p"),gt.innerHTML=Tn,Wo=a(),H=p("div"),u(Xe.$$.fragment),Zo=a(),_t=p("p"),_t.innerHTML=wn,Bo=a(),u(me.$$.fragment),Io=a(),u(he.$$.fragment),At=a(),u(Se.$$.fragment),Et=a(),Y=p("div"),u(He.$$.fragment),No=a(),bt=p("p"),bt.innerHTML=kn,Go=a(),yt=p("p"),yt.innerHTML=vn,qo=a(),Mt=p("p"),Mt.innerHTML=$n,Ro=a(),L=p("div"),u(Le.$$.fragment),Vo=a(),Tt=p("p"),Tt.innerHTML=jn,Xo=a(),u(ue.$$.fragment),So=a(),u(fe.$$.fragment),Ot=a(),u(Qe.$$.fragment),Pt=a(),kt=p("p"),this.h()},l(e){const n=xn("svelte-u9bgzb",document.head);t=m(n,"META",{name:!0,content:!0}),n.forEach(s),M=r(e),l=m(e,"P",{}),z(l).forEach(s),d=r(e),T=m(e,"P",{"data-svelte-h":!0}),h(T)!=="svelte-1wchmsn"&&(T.innerHTML=o),w=r(e),f(ge.$$.fragment,e),vt=r(e),te=m(e,"DIV",{class:!0,"data-svelte-h":!0}),h(te)!=="svelte-13t8s2t"&&(te.innerHTML=Ho),$t=r(e),f(_e.$$.fragment,e),jt=r(e),be=m(e,"P",{"data-svelte-h":!0}),h(be)!=="svelte-hcrtc3"&&(be.innerHTML=Lo),Ct=r(e),ye=m(e,"P",{"data-svelte-h":!0}),h(ye)!=="svelte-vfdo9a"&&(ye.textContent=Qo),Jt=r(e),Me=m(e,"P",{"data-svelte-h":!0}),h(Me)!=="svelte-rozt8p"&&(Me.innerHTML=Ao),Ut=r(e),Te=m(e,"P",{"data-svelte-h":!0}),h(Te)!=="svelte-6fv3r5"&&(Te.innerHTML=Eo),Yt=r(e),f(we.$$.fragment,e),zt=r(e),ke=m(e,"UL",{"data-svelte-h":!0}),h(ke)!=="svelte-1uwfv15"&&(ke.innerHTML=Oo),xt=r(e),oe=m(e,"IMG",{src:!0,alt:!0,width:!0}),Ft=r(e),ve=m(e,"SMALL",{"data-svelte-h":!0}),h(ve)!=="svelte-pr16rt"&&(ve.innerHTML=Do),Wt=r(e),f($e.$$.fragment,e),Zt=r(e),je=m(e,"UL",{"data-svelte-h":!0}),h(je)!=="svelte-mgusi3"&&(je.innerHTML=Ko),Bt=r(e),f(Ce.$$.fragment,e),It=r(e),F=m(e,"DIV",{class:!0});var Q=z(F);f(Je.$$.fragment,Q),Kt=r(Q),Oe=m(Q,"P",{"data-svelte-h":!0}),h(Oe)!=="svelte-8ltjvr"&&(Oe.innerHTML=en),eo=r(Q),Pe=m(Q,"P",{"data-svelte-h":!0}),h(Pe)!=="svelte-1ek1ss9"&&(Pe.innerHTML=tn),to=r(Q),f(ne.$$.fragment,Q),Q.forEach(s),Nt=r(e),f(Ue.$$.fragment,e),Gt=r(e),$=m(e,"DIV",{class:!0});var W=z($);f(Ye.$$.fragment,W),oo=r(W),De=m(W,"P",{"data-svelte-h":!0}),h(De)!=="svelte-10xq65u"&&(De.textContent=on),no=r(W),Ke=m(W,"P",{"data-svelte-h":!0}),h(Ke)!=="svelte-q52n56"&&(Ke.innerHTML=nn),so=r(W),et=m(W,"P",{"data-svelte-h":!0}),h(et)!=="svelte-hswkmf"&&(et.innerHTML=sn),ao=r(W),K=m(W,"DIV",{class:!0});var ee=z(K);f(ze.$$.fragment,ee),ro=r(ee),tt=m(ee,"P",{"data-svelte-h":!0}),h(tt)!=="svelte-1gh7gtp"&&(tt.innerHTML=an),io=r(ee),f(se.$$.fragment,ee),ee.forEach(s),W.forEach(s),qt=r(e),f(xe.$$.fragment,e),Rt=r(e),j=m(e,"DIV",{class:!0});var Z=z(j);f(Fe.$$.fragment,Z),lo=r(Z),ot=m(Z,"P",{"data-svelte-h":!0}),h(ot)!=="svelte-1wm2sgm"&&(ot.innerHTML=rn),co=r(Z),nt=m(Z,"P",{"data-svelte-h":!0}),h(nt)!=="svelte-q52n56"&&(nt.innerHTML=ln),po=r(Z),st=m(Z,"P",{"data-svelte-h":!0}),h(st)!=="svelte-hswkmf"&&(st.innerHTML=dn),mo=r(Z),X=m(Z,"DIV",{class:!0});var A=z(X);f(We.$$.fragment,A),ho=r(A),at=m(A,"P",{"data-svelte-h":!0}),h(at)!=="svelte-109oz0x"&&(at.innerHTML=cn),uo=r(A),f(ae.$$.fragment,A),fo=r(A),f(re.$$.fragment,A),A.forEach(s),Z.forEach(s),Vt=r(e),f(Ze.$$.fragment,e),Xt=r(e),C=m(e,"DIV",{class:!0});var B=z(C);f(Be.$$.fragment,B),go=r(B),rt=m(B,"P",{"data-svelte-h":!0}),h(rt)!=="svelte-1fadnxc"&&(rt.textContent=pn),_o=r(B),it=m(B,"P",{"data-svelte-h":!0}),h(it)!=="svelte-q52n56"&&(it.innerHTML=mn),bo=r(B),lt=m(B,"P",{"data-svelte-h":!0}),h(lt)!=="svelte-hswkmf"&&(lt.innerHTML=hn),yo=r(B),x=m(B,"DIV",{class:!0});var I=z(x);f(Ie.$$.fragment,I),Mo=r(I),dt=m(I,"P",{"data-svelte-h":!0}),h(dt)!=="svelte-1lhemfl"&&(dt.innerHTML=un),To=r(I),f(ie.$$.fragment,I),wo=r(I),f(le.$$.fragment,I),ko=r(I),f(de.$$.fragment,I),I.forEach(s),B.forEach(s),St=r(e),f(Ne.$$.fragment,e),Ht=r(e),J=m(e,"DIV",{class:!0});var N=z(J);f(Ge.$$.fragment,N),vo=r(N),ct=m(N,"P",{"data-svelte-h":!0}),h(ct)!=="svelte-1uprcbb"&&(ct.textContent=fn),$o=r(N),pt=m(N,"P",{"data-svelte-h":!0}),h(pt)!=="svelte-q52n56"&&(pt.innerHTML=gn),jo=r(N),mt=m(N,"P",{"data-svelte-h":!0}),h(mt)!=="svelte-hswkmf"&&(mt.innerHTML=_n),Co=r(N),S=m(N,"DIV",{class:!0});var E=z(S);f(qe.$$.fragment,E),Jo=r(E),ht=m(E,"P",{"data-svelte-h":!0}),h(ht)!=="svelte-1sd29qh"&&(ht.innerHTML=bn),Uo=r(E),f(ce.$$.fragment,E),Yo=r(E),f(pe.$$.fragment,E),E.forEach(s),N.forEach(s),Lt=r(e),f(Re.$$.fragment,e),Qt=r(e),U=m(e,"DIV",{class:!0});var G=z(U);f(Ve.$$.fragment,G),zo=r(G),ut=m(G,"P",{"data-svelte-h":!0}),h(ut)!=="svelte-1ukhq62"&&(ut.textContent=yn),xo=r(G),ft=m(G,"P",{"data-svelte-h":!0}),h(ft)!=="svelte-q52n56"&&(ft.innerHTML=Mn),Fo=r(G),gt=m(G,"P",{"data-svelte-h":!0}),h(gt)!=="svelte-hswkmf"&&(gt.innerHTML=Tn),Wo=r(G),H=m(G,"DIV",{class:!0});var O=z(H);f(Xe.$$.fragment,O),Zo=r(O),_t=m(O,"P",{"data-svelte-h":!0}),h(_t)!=="svelte-19ie2g5"&&(_t.innerHTML=wn),Bo=r(O),f(me.$$.fragment,O),Io=r(O),f(he.$$.fragment,O),O.forEach(s),G.forEach(s),At=r(e),f(Se.$$.fragment,e),Et=r(e),Y=m(e,"DIV",{class:!0});var q=z(Y);f(He.$$.fragment,q),No=r(q),bt=m(q,"P",{"data-svelte-h":!0}),h(bt)!=="svelte-1siyn2r"&&(bt.innerHTML=kn),Go=r(q),yt=m(q,"P",{"data-svelte-h":!0}),h(yt)!=="svelte-q52n56"&&(yt.innerHTML=vn),qo=r(q),Mt=m(q,"P",{"data-svelte-h":!0}),h(Mt)!=="svelte-hswkmf"&&(Mt.innerHTML=$n),Ro=r(q),L=m(q,"DIV",{class:!0});var P=z(L);f(Le.$$.fragment,P),Vo=r(P),Tt=m(P,"P",{"data-svelte-h":!0}),h(Tt)!=="svelte-1jb3ebz"&&(Tt.innerHTML=jn),Xo=r(P),f(ue.$$.fragment,P),So=r(P),f(fe.$$.fragment,P),P.forEach(s),q.forEach(s),Ot=r(e),f(Qe.$$.fragment,e),Pt=r(e),kt=m(e,"P",{}),z(kt).forEach(s),this.h()},h(){v(t,"name","hf:doc:metadata"),v(t,"content",An),v(te,"class","flex flex-wrap space-x-1"),Jn(oe.src,Po="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/yoso_architecture.jpg")||v(oe,"src",Po),v(oe,"alt","drawing"),v(oe,"width","600"),v(F,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(K,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v($,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(X,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(j,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(x,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(C,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(S,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(J,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(H,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(U,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(L,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(Y,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,n){i(document.head,t),c(e,M,n),c(e,l,n),c(e,d,n),c(e,T,n),c(e,w,n),g(ge,e,n),c(e,vt,n),c(e,te,n),c(e,$t,n),g(_e,e,n),c(e,jt,n),c(e,be,n),c(e,Ct,n),c(e,ye,n),c(e,Jt,n),c(e,Me,n),c(e,Ut,n),c(e,Te,n),c(e,Yt,n),g(we,e,n),c(e,zt,n),c(e,ke,n),c(e,xt,n),c(e,oe,n),c(e,Ft,n),c(e,ve,n),c(e,Wt,n),g($e,e,n),c(e,Zt,n),c(e,je,n),c(e,Bt,n),g(Ce,e,n),c(e,It,n),c(e,F,n),g(Je,F,null),i(F,Kt),i(F,Oe),i(F,eo),i(F,Pe),i(F,to),g(ne,F,null),c(e,Nt,n),g(Ue,e,n),c(e,Gt,n),c(e,$,n),g(Ye,$,null),i($,oo),i($,De),i($,no),i($,Ke),i($,so),i($,et),i($,ao),i($,K),g(ze,K,null),i(K,ro),i(K,tt),i(K,io),g(se,K,null),c(e,qt,n),g(xe,e,n),c(e,Rt,n),c(e,j,n),g(Fe,j,null),i(j,lo),i(j,ot),i(j,co),i(j,nt),i(j,po),i(j,st),i(j,mo),i(j,X),g(We,X,null),i(X,ho),i(X,at),i(X,uo),g(ae,X,null),i(X,fo),g(re,X,null),c(e,Vt,n),g(Ze,e,n),c(e,Xt,n),c(e,C,n),g(Be,C,null),i(C,go),i(C,rt),i(C,_o),i(C,it),i(C,bo),i(C,lt),i(C,yo),i(C,x),g(Ie,x,null),i(x,Mo),i(x,dt),i(x,To),g(ie,x,null),i(x,wo),g(le,x,null),i(x,ko),g(de,x,null),c(e,St,n),g(Ne,e,n),c(e,Ht,n),c(e,J,n),g(Ge,J,null),i(J,vo),i(J,ct),i(J,$o),i(J,pt),i(J,jo),i(J,mt),i(J,Co),i(J,S),g(qe,S,null),i(S,Jo),i(S,ht),i(S,Uo),g(ce,S,null),i(S,Yo),g(pe,S,null),c(e,Lt,n),g(Re,e,n),c(e,Qt,n),c(e,U,n),g(Ve,U,null),i(U,zo),i(U,ut),i(U,xo),i(U,ft),i(U,Fo),i(U,gt),i(U,Wo),i(U,H),g(Xe,H,null),i(H,Zo),i(H,_t),i(H,Bo),g(me,H,null),i(H,Io),g(he,H,null),c(e,At,n),g(Se,e,n),c(e,Et,n),c(e,Y,n),g(He,Y,null),i(Y,No),i(Y,bt),i(Y,Go),i(Y,yt),i(Y,qo),i(Y,Mt),i(Y,Ro),i(Y,L),g(Le,L,null),i(L,Vo),i(L,Tt),i(L,Xo),g(ue,L,null),i(L,So),g(fe,L,null),c(e,Ot,n),g(Qe,e,n),c(e,Pt,n),c(e,kt,n),Dt=!0},p(e,[n]){const Q={};n&2&&(Q.$$scope={dirty:n,ctx:e}),ne.$set(Q);const W={};n&2&&(W.$$scope={dirty:n,ctx:e}),se.$set(W);const ee={};n&2&&(ee.$$scope={dirty:n,ctx:e}),ae.$set(ee);const Z={};n&2&&(Z.$$scope={dirty:n,ctx:e}),re.$set(Z);const A={};n&2&&(A.$$scope={dirty:n,ctx:e}),ie.$set(A);const B={};n&2&&(B.$$scope={dirty:n,ctx:e}),le.$set(B);const I={};n&2&&(I.$$scope={dirty:n,ctx:e}),de.$set(I);const N={};n&2&&(N.$$scope={dirty:n,ctx:e}),ce.$set(N);const E={};n&2&&(E.$$scope={dirty:n,ctx:e}),pe.$set(E);const G={};n&2&&(G.$$scope={dirty:n,ctx:e}),me.$set(G);const O={};n&2&&(O.$$scope={dirty:n,ctx:e}),he.$set(O);const q={};n&2&&(q.$$scope={dirty:n,ctx:e}),ue.$set(q);const P={};n&2&&(P.$$scope={dirty:n,ctx:e}),fe.$set(P)},i(e){Dt||(_(ge.$$.fragment,e),_(_e.$$.fragment,e),_(we.$$.fragment,e),_($e.$$.fragment,e),_(Ce.$$.fragment,e),_(Je.$$.fragment,e),_(ne.$$.fragment,e),_(Ue.$$.fragment,e),_(Ye.$$.fragment,e),_(ze.$$.fragment,e),_(se.$$.fragment,e),_(xe.$$.fragment,e),_(Fe.$$.fragment,e),_(We.$$.fragment,e),_(ae.$$.fragment,e),_(re.$$.fragment,e),_(Ze.$$.fragment,e),_(Be.$$.fragment,e),_(Ie.$$.fragment,e),_(ie.$$.fragment,e),_(le.$$.fragment,e),_(de.$$.fragment,e),_(Ne.$$.fragment,e),_(Ge.$$.fragment,e),_(qe.$$.fragment,e),_(ce.$$.fragment,e),_(pe.$$.fragment,e),_(Re.$$.fragment,e),_(Ve.$$.fragment,e),_(Xe.$$.fragment,e),_(me.$$.fragment,e),_(he.$$.fragment,e),_(Se.$$.fragment,e),_(He.$$.fragment,e),_(Le.$$.fragment,e),_(ue.$$.fragment,e),_(fe.$$.fragment,e),_(Qe.$$.fragment,e),Dt=!0)},o(e){b(ge.$$.fragment,e),b(_e.$$.fragment,e),b(we.$$.fragment,e),b($e.$$.fragment,e),b(Ce.$$.fragment,e),b(Je.$$.fragment,e),b(ne.$$.fragment,e),b(Ue.$$.fragment,e),b(Ye.$$.fragment,e),b(ze.$$.fragment,e),b(se.$$.fragment,e),b(xe.$$.fragment,e),b(Fe.$$.fragment,e),b(We.$$.fragment,e),b(ae.$$.fragment,e),b(re.$$.fragment,e),b(Ze.$$.fragment,e),b(Be.$$.fragment,e),b(Ie.$$.fragment,e),b(ie.$$.fragment,e),b(le.$$.fragment,e),b(de.$$.fragment,e),b(Ne.$$.fragment,e),b(Ge.$$.fragment,e),b(qe.$$.fragment,e),b(ce.$$.fragment,e),b(pe.$$.fragment,e),b(Re.$$.fragment,e),b(Ve.$$.fragment,e),b(Xe.$$.fragment,e),b(me.$$.fragment,e),b(he.$$.fragment,e),b(Se.$$.fragment,e),b(He.$$.fragment,e),b(Le.$$.fragment,e),b(ue.$$.fragment,e),b(fe.$$.fragment,e),b(Qe.$$.fragment,e),Dt=!1},d(e){e&&(s(M),s(l),s(d),s(T),s(w),s(vt),s(te),s($t),s(jt),s(be),s(Ct),s(ye),s(Jt),s(Me),s(Ut),s(Te),s(Yt),s(zt),s(ke),s(xt),s(oe),s(Ft),s(ve),s(Wt),s(Zt),s(je),s(Bt),s(It),s(F),s(Nt),s(Gt),s($),s(qt),s(Rt),s(j),s(Vt),s(Xt),s(C),s(St),s(Ht),s(J),s(Lt),s(Qt),s(U),s(At),s(Et),s(Y),s(Ot),s(Pt),s(kt)),s(t),y(ge,e),y(_e,e),y(we,e),y($e,e),y(Ce,e),y(Je),y(ne),y(Ue,e),y(Ye),y(ze),y(se),y(xe,e),y(Fe),y(We),y(ae),y(re),y(Ze,e),y(Be),y(Ie),y(ie),y(le),y(de),y(Ne,e),y(Ge),y(qe),y(ce),y(pe),y(Re,e),y(Ve),y(Xe),y(me),y(he),y(Se,e),y(He),y(Le),y(ue),y(fe),y(Qe,e)}}}const An='{"title":"YOSO","local":"yoso","sections":[{"title":"Overview","local":"overview","sections":[],"depth":2},{"title":"Usage tips","local":"usage-tips","sections":[],"depth":2},{"title":"Resources","local":"resources","sections":[],"depth":2},{"title":"YosoConfig","local":"transformers.YosoConfig","sections":[],"depth":2},{"title":"YosoModel","local":"transformers.YosoModel","sections":[],"depth":2},{"title":"YosoForMaskedLM","local":"transformers.YosoForMaskedLM","sections":[],"depth":2},{"title":"YosoForSequenceClassification","local":"transformers.YosoForSequenceClassification","sections":[],"depth":2},{"title":"YosoForMultipleChoice","local":"transformers.YosoForMultipleChoice","sections":[],"depth":2},{"title":"YosoForTokenClassification","local":"transformers.YosoForTokenClassification","sections":[],"depth":2},{"title":"YosoForQuestionAnswering","local":"transformers.YosoForQuestionAnswering","sections":[],"depth":2}],"depth":1}';function En(k){return Un(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class ns extends Yn{constructor(t){super(),zn(this,t,En,Qn,Cn,{})}}export{ns as component};
