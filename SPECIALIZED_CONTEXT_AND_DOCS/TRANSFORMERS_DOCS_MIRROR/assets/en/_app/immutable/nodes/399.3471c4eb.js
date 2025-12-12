import{s as js,o as Js,n as R}from"../chunks/scheduler.18a86fab.js";import{S as Us,i as Fs,g as l,s,r as h,A as Ws,h as d,f as i,c as r,j as v,x as m,u as f,k as w,y as n,a as u,v as g,d as _,t as b,w as y}from"../chunks/index.98837b22.js";import{T as Kt}from"../chunks/Tip.77304350.js";import{D as q}from"../chunks/Docstring.a1ef7999.js";import{C as ht}from"../chunks/CodeBlock.8d0c2e8a.js";import{E as mt}from"../chunks/ExampleCodeBlock.8c3ee1f9.js";import{H as X,E as Zs}from"../chunks/getInferenceSnippets.06c2775f.js";function Vs(z){let t,T="Examples:",c,p,k;return p=new ht({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMFNxdWVlemVCZXJ0Q29uZmlnJTJDJTIwU3F1ZWV6ZUJlcnRNb2RlbCUwQSUwQSUyMyUyMEluaXRpYWxpemluZyUyMGElMjBTcXVlZXplQkVSVCUyMGNvbmZpZ3VyYXRpb24lMEFjb25maWd1cmF0aW9uJTIwJTNEJTIwU3F1ZWV6ZUJlcnRDb25maWcoKSUwQSUwQSUyMyUyMEluaXRpYWxpemluZyUyMGElMjBtb2RlbCUyMCh3aXRoJTIwcmFuZG9tJTIwd2VpZ2h0cyklMjBmcm9tJTIwdGhlJTIwY29uZmlndXJhdGlvbiUyMGFib3ZlJTBBbW9kZWwlMjAlM0QlMjBTcXVlZXplQmVydE1vZGVsKGNvbmZpZ3VyYXRpb24pJTBBJTBBJTIzJTIwQWNjZXNzaW5nJTIwdGhlJTIwbW9kZWwlMjBjb25maWd1cmF0aW9uJTBBY29uZmlndXJhdGlvbiUyMCUzRCUyMG1vZGVsLmNvbmZpZw==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> SqueezeBertConfig, SqueezeBertModel

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a SqueezeBERT configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = SqueezeBertConfig()

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a model (with random weights) from the configuration above</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = SqueezeBertModel(configuration)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Accessing the model configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = model.config`,wrap:!1}}),{c(){t=l("p"),t.textContent=T,c=s(),h(p.$$.fragment)},l(o){t=d(o,"P",{"data-svelte-h":!0}),m(t)!=="svelte-kvfsh7"&&(t.textContent=T),c=r(o),f(p.$$.fragment,o)},m(o,M){u(o,t,M),u(o,c,M),g(p,o,M),k=!0},p:R,i(o){k||(_(p.$$.fragment,o),k=!0)},o(o){b(p.$$.fragment,o),k=!1},d(o){o&&(i(t),i(c)),y(p,o)}}}function Is(z){let t,T=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=l("p"),t.innerHTML=T},l(c){t=d(c,"P",{"data-svelte-h":!0}),m(t)!=="svelte-fincs2"&&(t.innerHTML=T)},m(c,p){u(c,t,p)},p:R,d(c){c&&i(t)}}}function Ns(z){let t,T=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=l("p"),t.innerHTML=T},l(c){t=d(c,"P",{"data-svelte-h":!0}),m(t)!=="svelte-fincs2"&&(t.innerHTML=T)},m(c,p){u(c,t,p)},p:R,d(c){c&&i(t)}}}function Xs(z){let t,T="Example:",c,p,k;return p=new ht({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBTcXVlZXplQmVydEZvck1hc2tlZExNJTBBaW1wb3J0JTIwdG9yY2glMEElMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJzcXVlZXplYmVydCUyRnNxdWVlemViZXJ0LXVuY2FzZWQlMjIpJTBBbW9kZWwlMjAlM0QlMjBTcXVlZXplQmVydEZvck1hc2tlZExNLmZyb21fcHJldHJhaW5lZCglMjJzcXVlZXplYmVydCUyRnNxdWVlemViZXJ0LXVuY2FzZWQlMjIpJTBBJTBBaW5wdXRzJTIwJTNEJTIwdG9rZW5pemVyKCUyMlRoZSUyMGNhcGl0YWwlMjBvZiUyMEZyYW5jZSUyMGlzJTIwJTNDbWFzayUzRS4lMjIlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQSUwQXdpdGglMjB0b3JjaC5ub19ncmFkKCklM0ElMEElMjAlMjAlMjAlMjBsb2dpdHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cykubG9naXRzJTBBJTBBJTIzJTIwcmV0cmlldmUlMjBpbmRleCUyMG9mJTIwJTNDbWFzayUzRSUwQW1hc2tfdG9rZW5faW5kZXglMjAlM0QlMjAoaW5wdXRzLmlucHV0X2lkcyUyMCUzRCUzRCUyMHRva2VuaXplci5tYXNrX3Rva2VuX2lkKSU1QjAlNUQubm9uemVybyhhc190dXBsZSUzRFRydWUpJTVCMCU1RCUwQSUwQXByZWRpY3RlZF90b2tlbl9pZCUyMCUzRCUyMGxvZ2l0cyU1QjAlMkMlMjBtYXNrX3Rva2VuX2luZGV4JTVELmFyZ21heChheGlzJTNELTEpJTBBdG9rZW5pemVyLmRlY29kZShwcmVkaWN0ZWRfdG9rZW5faWQpJTBBJTBBbGFiZWxzJTIwJTNEJTIwdG9rZW5pemVyKCUyMlRoZSUyMGNhcGl0YWwlMjBvZiUyMEZyYW5jZSUyMGlzJTIwUGFyaXMuJTIyJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiklNUIlMjJpbnB1dF9pZHMlMjIlNUQlMEElMjMlMjBtYXNrJTIwbGFiZWxzJTIwb2YlMjBub24tJTNDbWFzayUzRSUyMHRva2VucyUwQWxhYmVscyUyMCUzRCUyMHRvcmNoLndoZXJlKGlucHV0cy5pbnB1dF9pZHMlMjAlM0QlM0QlMjB0b2tlbml6ZXIubWFza190b2tlbl9pZCUyQyUyMGxhYmVscyUyQyUyMC0xMDApJTBBJTBBb3V0cHV0cyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzJTJDJTIwbGFiZWxzJTNEbGFiZWxzKSUwQXJvdW5kKG91dHB1dHMubG9zcy5pdGVtKCklMkMlMjAyKQ==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, SqueezeBertForMaskedLM
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;squeezebert/squeezebert-uncased&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = SqueezeBertForMaskedLM.from_pretrained(<span class="hljs-string">&quot;squeezebert/squeezebert-uncased&quot;</span>)

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
...`,wrap:!1}}),{c(){t=l("p"),t.textContent=T,c=s(),h(p.$$.fragment)},l(o){t=d(o,"P",{"data-svelte-h":!0}),m(t)!=="svelte-11lpom8"&&(t.textContent=T),c=r(o),f(p.$$.fragment,o)},m(o,M){u(o,t,M),u(o,c,M),g(p,o,M),k=!0},p:R,i(o){k||(_(p.$$.fragment,o),k=!0)},o(o){b(p.$$.fragment,o),k=!1},d(o){o&&(i(t),i(c)),y(p,o)}}}function Rs(z){let t,T=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=l("p"),t.innerHTML=T},l(c){t=d(c,"P",{"data-svelte-h":!0}),m(t)!=="svelte-fincs2"&&(t.innerHTML=T)},m(c,p){u(c,t,p)},p:R,d(c){c&&i(t)}}}function Gs(z){let t,T="Example of single-label classification:",c,p,k;return p=new ht({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Rva2VuaXplciUyQyUyMFNxdWVlemVCZXJ0Rm9yU2VxdWVuY2VDbGFzc2lmaWNhdGlvbiUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMnNxdWVlemViZXJ0JTJGc3F1ZWV6ZWJlcnQtdW5jYXNlZCUyMiklMEFtb2RlbCUyMCUzRCUyMFNxdWVlemVCZXJ0Rm9yU2VxdWVuY2VDbGFzc2lmaWNhdGlvbi5mcm9tX3ByZXRyYWluZWQoJTIyc3F1ZWV6ZWJlcnQlMkZzcXVlZXplYmVydC11bmNhc2VkJTIyKSUwQSUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplciglMjJIZWxsbyUyQyUyMG15JTIwZG9nJTIwaXMlMjBjdXRlJTIyJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiklMEElMEF3aXRoJTIwdG9yY2gubm9fZ3JhZCgpJTNBJTBBJTIwJTIwJTIwJTIwbG9naXRzJTIwJTNEJTIwbW9kZWwoKippbnB1dHMpLmxvZ2l0cyUwQSUwQXByZWRpY3RlZF9jbGFzc19pZCUyMCUzRCUyMGxvZ2l0cy5hcmdtYXgoKS5pdGVtKCklMEFtb2RlbC5jb25maWcuaWQybGFiZWwlNUJwcmVkaWN0ZWRfY2xhc3NfaWQlNUQlMEElMEElMjMlMjBUbyUyMHRyYWluJTIwYSUyMG1vZGVsJTIwb24lMjAlNjBudW1fbGFiZWxzJTYwJTIwY2xhc3NlcyUyQyUyMHlvdSUyMGNhbiUyMHBhc3MlMjAlNjBudW1fbGFiZWxzJTNEbnVtX2xhYmVscyU2MCUyMHRvJTIwJTYwLmZyb21fcHJldHJhaW5lZCguLi4pJTYwJTBBbnVtX2xhYmVscyUyMCUzRCUyMGxlbihtb2RlbC5jb25maWcuaWQybGFiZWwpJTBBbW9kZWwlMjAlM0QlMjBTcXVlZXplQmVydEZvclNlcXVlbmNlQ2xhc3NpZmljYXRpb24uZnJvbV9wcmV0cmFpbmVkKCUyMnNxdWVlemViZXJ0JTJGc3F1ZWV6ZWJlcnQtdW5jYXNlZCUyMiUyQyUyMG51bV9sYWJlbHMlM0RudW1fbGFiZWxzKSUwQSUwQWxhYmVscyUyMCUzRCUyMHRvcmNoLnRlbnNvciglNUIxJTVEKSUwQWxvc3MlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyUyQyUyMGxhYmVscyUzRGxhYmVscykubG9zcyUwQXJvdW5kKGxvc3MuaXRlbSgpJTJDJTIwMik=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, SqueezeBertForSequenceClassification

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;squeezebert/squeezebert-uncased&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = SqueezeBertForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;squeezebert/squeezebert-uncased&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    logits = model(**inputs).logits

<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_class_id = logits.argmax().item()
<span class="hljs-meta">&gt;&gt;&gt; </span>model.config.id2label[predicted_class_id]
...

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># To train a model on \`num_labels\` classes, you can pass \`num_labels=num_labels\` to \`.from_pretrained(...)\`</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>num_labels = <span class="hljs-built_in">len</span>(model.config.id2label)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = SqueezeBertForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;squeezebert/squeezebert-uncased&quot;</span>, num_labels=num_labels)

<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.tensor([<span class="hljs-number">1</span>])
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = model(**inputs, labels=labels).loss
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">round</span>(loss.item(), <span class="hljs-number">2</span>)
...`,wrap:!1}}),{c(){t=l("p"),t.textContent=T,c=s(),h(p.$$.fragment)},l(o){t=d(o,"P",{"data-svelte-h":!0}),m(t)!=="svelte-ykxpe4"&&(t.textContent=T),c=r(o),f(p.$$.fragment,o)},m(o,M){u(o,t,M),u(o,c,M),g(p,o,M),k=!0},p:R,i(o){k||(_(p.$$.fragment,o),k=!0)},o(o){b(p.$$.fragment,o),k=!1},d(o){o&&(i(t),i(c)),y(p,o)}}}function Ls(z){let t,T="Example of multi-label classification:",c,p,k;return p=new ht({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Rva2VuaXplciUyQyUyMFNxdWVlemVCZXJ0Rm9yU2VxdWVuY2VDbGFzc2lmaWNhdGlvbiUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMnNxdWVlemViZXJ0JTJGc3F1ZWV6ZWJlcnQtdW5jYXNlZCUyMiklMEFtb2RlbCUyMCUzRCUyMFNxdWVlemVCZXJ0Rm9yU2VxdWVuY2VDbGFzc2lmaWNhdGlvbi5mcm9tX3ByZXRyYWluZWQoJTIyc3F1ZWV6ZWJlcnQlMkZzcXVlZXplYmVydC11bmNhc2VkJTIyJTJDJTIwcHJvYmxlbV90eXBlJTNEJTIybXVsdGlfbGFiZWxfY2xhc3NpZmljYXRpb24lMjIpJTBBJTBBaW5wdXRzJTIwJTNEJTIwdG9rZW5pemVyKCUyMkhlbGxvJTJDJTIwbXklMjBkb2clMjBpcyUyMGN1dGUlMjIlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQSUwQXdpdGglMjB0b3JjaC5ub19ncmFkKCklM0ElMEElMjAlMjAlMjAlMjBsb2dpdHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cykubG9naXRzJTBBJTBBcHJlZGljdGVkX2NsYXNzX2lkcyUyMCUzRCUyMHRvcmNoLmFyYW5nZSgwJTJDJTIwbG9naXRzLnNoYXBlJTVCLTElNUQpJTVCdG9yY2guc2lnbW9pZChsb2dpdHMpLnNxdWVlemUoZGltJTNEMCklMjAlM0UlMjAwLjUlNUQlMEElMEElMjMlMjBUbyUyMHRyYWluJTIwYSUyMG1vZGVsJTIwb24lMjAlNjBudW1fbGFiZWxzJTYwJTIwY2xhc3NlcyUyQyUyMHlvdSUyMGNhbiUyMHBhc3MlMjAlNjBudW1fbGFiZWxzJTNEbnVtX2xhYmVscyU2MCUyMHRvJTIwJTYwLmZyb21fcHJldHJhaW5lZCguLi4pJTYwJTBBbnVtX2xhYmVscyUyMCUzRCUyMGxlbihtb2RlbC5jb25maWcuaWQybGFiZWwpJTBBbW9kZWwlMjAlM0QlMjBTcXVlZXplQmVydEZvclNlcXVlbmNlQ2xhc3NpZmljYXRpb24uZnJvbV9wcmV0cmFpbmVkKCUwQSUyMCUyMCUyMCUyMCUyMnNxdWVlemViZXJ0JTJGc3F1ZWV6ZWJlcnQtdW5jYXNlZCUyMiUyQyUyMG51bV9sYWJlbHMlM0RudW1fbGFiZWxzJTJDJTIwcHJvYmxlbV90eXBlJTNEJTIybXVsdGlfbGFiZWxfY2xhc3NpZmljYXRpb24lMjIlMEEpJTBBJTBBbGFiZWxzJTIwJTNEJTIwdG9yY2guc3VtKCUwQSUyMCUyMCUyMCUyMHRvcmNoLm5uLmZ1bmN0aW9uYWwub25lX2hvdChwcmVkaWN0ZWRfY2xhc3NfaWRzJTVCTm9uZSUyQyUyMCUzQSU1RC5jbG9uZSgpJTJDJTIwbnVtX2NsYXNzZXMlM0RudW1fbGFiZWxzKSUyQyUyMGRpbSUzRDElMEEpLnRvKHRvcmNoLmZsb2F0KSUwQWxvc3MlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyUyQyUyMGxhYmVscyUzRGxhYmVscykubG9zcw==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, SqueezeBertForSequenceClassification

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;squeezebert/squeezebert-uncased&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = SqueezeBertForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;squeezebert/squeezebert-uncased&quot;</span>, problem_type=<span class="hljs-string">&quot;multi_label_classification&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    logits = model(**inputs).logits

<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_class_ids = torch.arange(<span class="hljs-number">0</span>, logits.shape[-<span class="hljs-number">1</span>])[torch.sigmoid(logits).squeeze(dim=<span class="hljs-number">0</span>) &gt; <span class="hljs-number">0.5</span>]

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># To train a model on \`num_labels\` classes, you can pass \`num_labels=num_labels\` to \`.from_pretrained(...)\`</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>num_labels = <span class="hljs-built_in">len</span>(model.config.id2label)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = SqueezeBertForSequenceClassification.from_pretrained(
<span class="hljs-meta">... </span>    <span class="hljs-string">&quot;squeezebert/squeezebert-uncased&quot;</span>, num_labels=num_labels, problem_type=<span class="hljs-string">&quot;multi_label_classification&quot;</span>
<span class="hljs-meta">... </span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.<span class="hljs-built_in">sum</span>(
<span class="hljs-meta">... </span>    torch.nn.functional.one_hot(predicted_class_ids[<span class="hljs-literal">None</span>, :].clone(), num_classes=num_labels), dim=<span class="hljs-number">1</span>
<span class="hljs-meta">... </span>).to(torch.<span class="hljs-built_in">float</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = model(**inputs, labels=labels).loss`,wrap:!1}}),{c(){t=l("p"),t.textContent=T,c=s(),h(p.$$.fragment)},l(o){t=d(o,"P",{"data-svelte-h":!0}),m(t)!=="svelte-1l8e32d"&&(t.textContent=T),c=r(o),f(p.$$.fragment,o)},m(o,M){u(o,t,M),u(o,c,M),g(p,o,M),k=!0},p:R,i(o){k||(_(p.$$.fragment,o),k=!0)},o(o){b(p.$$.fragment,o),k=!1},d(o){o&&(i(t),i(c)),y(p,o)}}}function Qs(z){let t,T=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=l("p"),t.innerHTML=T},l(c){t=d(c,"P",{"data-svelte-h":!0}),m(t)!=="svelte-fincs2"&&(t.innerHTML=T)},m(c,p){u(c,t,p)},p:R,d(c){c&&i(t)}}}function Hs(z){let t,T="Example:",c,p,k;return p=new ht({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBTcXVlZXplQmVydEZvck11bHRpcGxlQ2hvaWNlJTBBaW1wb3J0JTIwdG9yY2glMEElMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJzcXVlZXplYmVydCUyRnNxdWVlemViZXJ0LXVuY2FzZWQlMjIpJTBBbW9kZWwlMjAlM0QlMjBTcXVlZXplQmVydEZvck11bHRpcGxlQ2hvaWNlLmZyb21fcHJldHJhaW5lZCglMjJzcXVlZXplYmVydCUyRnNxdWVlemViZXJ0LXVuY2FzZWQlMjIpJTBBJTBBcHJvbXB0JTIwJTNEJTIwJTIySW4lMjBJdGFseSUyQyUyMHBpenphJTIwc2VydmVkJTIwaW4lMjBmb3JtYWwlMjBzZXR0aW5ncyUyQyUyMHN1Y2glMjBhcyUyMGF0JTIwYSUyMHJlc3RhdXJhbnQlMkMlMjBpcyUyMHByZXNlbnRlZCUyMHVuc2xpY2VkLiUyMiUwQWNob2ljZTAlMjAlM0QlMjAlMjJJdCUyMGlzJTIwZWF0ZW4lMjB3aXRoJTIwYSUyMGZvcmslMjBhbmQlMjBhJTIwa25pZmUuJTIyJTBBY2hvaWNlMSUyMCUzRCUyMCUyMkl0JTIwaXMlMjBlYXRlbiUyMHdoaWxlJTIwaGVsZCUyMGluJTIwdGhlJTIwaGFuZC4lMjIlMEFsYWJlbHMlMjAlM0QlMjB0b3JjaC50ZW5zb3IoMCkudW5zcXVlZXplKDApJTIwJTIwJTIzJTIwY2hvaWNlMCUyMGlzJTIwY29ycmVjdCUyMChhY2NvcmRpbmclMjB0byUyMFdpa2lwZWRpYSUyMCUzQikpJTJDJTIwYmF0Y2glMjBzaXplJTIwMSUwQSUwQWVuY29kaW5nJTIwJTNEJTIwdG9rZW5pemVyKCU1QnByb21wdCUyQyUyMHByb21wdCU1RCUyQyUyMCU1QmNob2ljZTAlMkMlMjBjaG9pY2UxJTVEJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiUyQyUyMHBhZGRpbmclM0RUcnVlKSUwQW91dHB1dHMlMjAlM0QlMjBtb2RlbCgqKiU3QmslM0ElMjB2LnVuc3F1ZWV6ZSgwKSUyMGZvciUyMGslMkMlMjB2JTIwaW4lMjBlbmNvZGluZy5pdGVtcygpJTdEJTJDJTIwbGFiZWxzJTNEbGFiZWxzKSUyMCUyMCUyMyUyMGJhdGNoJTIwc2l6ZSUyMGlzJTIwMSUwQSUwQSUyMyUyMHRoZSUyMGxpbmVhciUyMGNsYXNzaWZpZXIlMjBzdGlsbCUyMG5lZWRzJTIwdG8lMjBiZSUyMHRyYWluZWQlMEFsb3NzJTIwJTNEJTIwb3V0cHV0cy5sb3NzJTBBbG9naXRzJTIwJTNEJTIwb3V0cHV0cy5sb2dpdHM=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, SqueezeBertForMultipleChoice
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;squeezebert/squeezebert-uncased&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = SqueezeBertForMultipleChoice.from_pretrained(<span class="hljs-string">&quot;squeezebert/squeezebert-uncased&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>prompt = <span class="hljs-string">&quot;In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>choice0 = <span class="hljs-string">&quot;It is eaten with a fork and a knife.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>choice1 = <span class="hljs-string">&quot;It is eaten while held in the hand.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.tensor(<span class="hljs-number">0</span>).unsqueeze(<span class="hljs-number">0</span>)  <span class="hljs-comment"># choice0 is correct (according to Wikipedia ;)), batch size 1</span>

<span class="hljs-meta">&gt;&gt;&gt; </span>encoding = tokenizer([prompt, prompt], [choice0, choice1], return_tensors=<span class="hljs-string">&quot;pt&quot;</span>, padding=<span class="hljs-literal">True</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**{k: v.unsqueeze(<span class="hljs-number">0</span>) <span class="hljs-keyword">for</span> k, v <span class="hljs-keyword">in</span> encoding.items()}, labels=labels)  <span class="hljs-comment"># batch size is 1</span>

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># the linear classifier still needs to be trained</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = outputs.loss
<span class="hljs-meta">&gt;&gt;&gt; </span>logits = outputs.logits`,wrap:!1}}),{c(){t=l("p"),t.textContent=T,c=s(),h(p.$$.fragment)},l(o){t=d(o,"P",{"data-svelte-h":!0}),m(t)!=="svelte-11lpom8"&&(t.textContent=T),c=r(o),f(p.$$.fragment,o)},m(o,M){u(o,t,M),u(o,c,M),g(p,o,M),k=!0},p:R,i(o){k||(_(p.$$.fragment,o),k=!0)},o(o){b(p.$$.fragment,o),k=!1},d(o){o&&(i(t),i(c)),y(p,o)}}}function Es(z){let t,T=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=l("p"),t.innerHTML=T},l(c){t=d(c,"P",{"data-svelte-h":!0}),m(t)!=="svelte-fincs2"&&(t.innerHTML=T)},m(c,p){u(c,t,p)},p:R,d(c){c&&i(t)}}}function Ps(z){let t,T="Example:",c,p,k;return p=new ht({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBTcXVlZXplQmVydEZvclRva2VuQ2xhc3NpZmljYXRpb24lMEFpbXBvcnQlMjB0b3JjaCUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMnNxdWVlemViZXJ0JTJGc3F1ZWV6ZWJlcnQtdW5jYXNlZCUyMiklMEFtb2RlbCUyMCUzRCUyMFNxdWVlemVCZXJ0Rm9yVG9rZW5DbGFzc2lmaWNhdGlvbi5mcm9tX3ByZXRyYWluZWQoJTIyc3F1ZWV6ZWJlcnQlMkZzcXVlZXplYmVydC11bmNhc2VkJTIyKSUwQSUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplciglMEElMjAlMjAlMjAlMjAlMjJIdWdnaW5nRmFjZSUyMGlzJTIwYSUyMGNvbXBhbnklMjBiYXNlZCUyMGluJTIwUGFyaXMlMjBhbmQlMjBOZXclMjBZb3JrJTIyJTJDJTIwYWRkX3NwZWNpYWxfdG9rZW5zJTNERmFsc2UlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyJTBBKSUwQSUwQXdpdGglMjB0b3JjaC5ub19ncmFkKCklM0ElMEElMjAlMjAlMjAlMjBsb2dpdHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cykubG9naXRzJTBBJTBBcHJlZGljdGVkX3Rva2VuX2NsYXNzX2lkcyUyMCUzRCUyMGxvZ2l0cy5hcmdtYXgoLTEpJTBBJTBBJTIzJTIwTm90ZSUyMHRoYXQlMjB0b2tlbnMlMjBhcmUlMjBjbGFzc2lmaWVkJTIwcmF0aGVyJTIwdGhlbiUyMGlucHV0JTIwd29yZHMlMjB3aGljaCUyMG1lYW5zJTIwdGhhdCUwQSUyMyUyMHRoZXJlJTIwbWlnaHQlMjBiZSUyMG1vcmUlMjBwcmVkaWN0ZWQlMjB0b2tlbiUyMGNsYXNzZXMlMjB0aGFuJTIwd29yZHMuJTBBJTIzJTIwTXVsdGlwbGUlMjB0b2tlbiUyMGNsYXNzZXMlMjBtaWdodCUyMGFjY291bnQlMjBmb3IlMjB0aGUlMjBzYW1lJTIwd29yZCUwQXByZWRpY3RlZF90b2tlbnNfY2xhc3NlcyUyMCUzRCUyMCU1Qm1vZGVsLmNvbmZpZy5pZDJsYWJlbCU1QnQuaXRlbSgpJTVEJTIwZm9yJTIwdCUyMGluJTIwcHJlZGljdGVkX3Rva2VuX2NsYXNzX2lkcyU1QjAlNUQlNUQlMEFwcmVkaWN0ZWRfdG9rZW5zX2NsYXNzZXMlMEElMEFsYWJlbHMlMjAlM0QlMjBwcmVkaWN0ZWRfdG9rZW5fY2xhc3NfaWRzJTBBbG9zcyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzJTJDJTIwbGFiZWxzJTNEbGFiZWxzKS5sb3NzJTBBcm91bmQobG9zcy5pdGVtKCklMkMlMjAyKQ==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, SqueezeBertForTokenClassification
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;squeezebert/squeezebert-uncased&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = SqueezeBertForTokenClassification.from_pretrained(<span class="hljs-string">&quot;squeezebert/squeezebert-uncased&quot;</span>)

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
...`,wrap:!1}}),{c(){t=l("p"),t.textContent=T,c=s(),h(p.$$.fragment)},l(o){t=d(o,"P",{"data-svelte-h":!0}),m(t)!=="svelte-11lpom8"&&(t.textContent=T),c=r(o),f(p.$$.fragment,o)},m(o,M){u(o,t,M),u(o,c,M),g(p,o,M),k=!0},p:R,i(o){k||(_(p.$$.fragment,o),k=!0)},o(o){b(p.$$.fragment,o),k=!1},d(o){o&&(i(t),i(c)),y(p,o)}}}function Ys(z){let t,T=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=l("p"),t.innerHTML=T},l(c){t=d(c,"P",{"data-svelte-h":!0}),m(t)!=="svelte-fincs2"&&(t.innerHTML=T)},m(c,p){u(c,t,p)},p:R,d(c){c&&i(t)}}}function As(z){let t,T="Example:",c,p,k;return p=new ht({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBTcXVlZXplQmVydEZvclF1ZXN0aW9uQW5zd2VyaW5nJTBBaW1wb3J0JTIwdG9yY2glMEElMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJzcXVlZXplYmVydCUyRnNxdWVlemViZXJ0LXVuY2FzZWQlMjIpJTBBbW9kZWwlMjAlM0QlMjBTcXVlZXplQmVydEZvclF1ZXN0aW9uQW5zd2VyaW5nLmZyb21fcHJldHJhaW5lZCglMjJzcXVlZXplYmVydCUyRnNxdWVlemViZXJ0LXVuY2FzZWQlMjIpJTBBJTBBcXVlc3Rpb24lMkMlMjB0ZXh0JTIwJTNEJTIwJTIyV2hvJTIwd2FzJTIwSmltJTIwSGVuc29uJTNGJTIyJTJDJTIwJTIySmltJTIwSGVuc29uJTIwd2FzJTIwYSUyMG5pY2UlMjBwdXBwZXQlMjIlMEElMEFpbnB1dHMlMjAlM0QlMjB0b2tlbml6ZXIocXVlc3Rpb24lMkMlMjB0ZXh0JTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiklMEF3aXRoJTIwdG9yY2gubm9fZ3JhZCgpJTNBJTBBJTIwJTIwJTIwJTIwb3V0cHV0cyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzKSUwQSUwQWFuc3dlcl9zdGFydF9pbmRleCUyMCUzRCUyMG91dHB1dHMuc3RhcnRfbG9naXRzLmFyZ21heCgpJTBBYW5zd2VyX2VuZF9pbmRleCUyMCUzRCUyMG91dHB1dHMuZW5kX2xvZ2l0cy5hcmdtYXgoKSUwQSUwQXByZWRpY3RfYW5zd2VyX3Rva2VucyUyMCUzRCUyMGlucHV0cy5pbnB1dF9pZHMlNUIwJTJDJTIwYW5zd2VyX3N0YXJ0X2luZGV4JTIwJTNBJTIwYW5zd2VyX2VuZF9pbmRleCUyMCUyQiUyMDElNUQlMEF0b2tlbml6ZXIuZGVjb2RlKHByZWRpY3RfYW5zd2VyX3Rva2VucyUyQyUyMHNraXBfc3BlY2lhbF90b2tlbnMlM0RUcnVlKSUwQSUwQSUyMyUyMHRhcmdldCUyMGlzJTIwJTIybmljZSUyMHB1cHBldCUyMiUwQXRhcmdldF9zdGFydF9pbmRleCUyMCUzRCUyMHRvcmNoLnRlbnNvciglNUIxNCU1RCklMEF0YXJnZXRfZW5kX2luZGV4JTIwJTNEJTIwdG9yY2gudGVuc29yKCU1QjE1JTVEKSUwQSUwQW91dHB1dHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyUyQyUyMHN0YXJ0X3Bvc2l0aW9ucyUzRHRhcmdldF9zdGFydF9pbmRleCUyQyUyMGVuZF9wb3NpdGlvbnMlM0R0YXJnZXRfZW5kX2luZGV4KSUwQWxvc3MlMjAlM0QlMjBvdXRwdXRzLmxvc3MlMEFyb3VuZChsb3NzLml0ZW0oKSUyQyUyMDIp",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, SqueezeBertForQuestionAnswering
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;squeezebert/squeezebert-uncased&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = SqueezeBertForQuestionAnswering.from_pretrained(<span class="hljs-string">&quot;squeezebert/squeezebert-uncased&quot;</span>)

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
...`,wrap:!1}}),{c(){t=l("p"),t.textContent=T,c=s(),h(p.$$.fragment)},l(o){t=d(o,"P",{"data-svelte-h":!0}),m(t)!=="svelte-11lpom8"&&(t.textContent=T),c=r(o),f(p.$$.fragment,o)},m(o,M){u(o,t,M),u(o,c,M),g(p,o,M),k=!0},p:R,i(o){k||(_(p.$$.fragment,o),k=!0)},o(o){b(p.$$.fragment,o),k=!1},d(o){o&&(i(t),i(c)),y(p,o)}}}function Ds(z){let t,T,c,p,k,o="<em>This model was released on 2020-06-19 and added to Hugging Face Transformers on 2020-11-16.</em>",M,qe,nn,le,Ro='<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/>',on,Be,sn,$e,Go=`The SqueezeBERT model was proposed in <a href="https://huggingface.co/papers/2006.11316" rel="nofollow">SqueezeBERT: What can computer vision teach NLP about efficient neural networks?</a> by Forrest N. Iandola, Albert E. Shaw, Ravi Krishna, Kurt W. Keutzer. It’s a
bidirectional transformer similar to the BERT model. The key difference between the BERT architecture and the
SqueezeBERT architecture is that SqueezeBERT uses <a href="https://blog.yani.io/filter-group-tutorial" rel="nofollow">grouped convolutions</a>
instead of fully-connected layers for the Q, K, V and FFN layers.`,rn,Se,Lo="The abstract from the paper is the following:",an,xe,Qo=`<em>Humans read and write hundreds of billions of messages every day. Further, due to the availability of large datasets,
large computing systems, and better neural network models, natural language processing (NLP) technology has made
significant strides in understanding, proofreading, and organizing these messages. Thus, there is a significant
opportunity to deploy NLP in myriad applications to help web users, social networks, and businesses. In particular, we
consider smartphones and other mobile devices as crucial platforms for deploying NLP models at scale. However, today’s
highly-accurate NLP neural network models such as BERT and RoBERTa are extremely computationally expensive, with
BERT-base taking 1.7 seconds to classify a text snippet on a Pixel 3 smartphone. In this work, we observe that methods
such as grouped convolutions have yielded significant speedups for computer vision networks, but many of these
techniques have not been adopted by NLP neural network designers. We demonstrate how to replace several operations in
self-attention layers with grouped convolutions, and we use this technique in a novel network architecture called
SqueezeBERT, which runs 4.3x faster than BERT-base on the Pixel 3 while achieving competitive accuracy on the GLUE test
set. The SqueezeBERT code will be released.</em>`,ln,Ce,Ho='This model was contributed by <a href="https://huggingface.co/forresti" rel="nofollow">forresti</a>.',dn,je,cn,Je,Eo=`<li>SqueezeBERT is a model with absolute position embeddings so it’s usually advised to pad the inputs on the right
rather than the left.</li> <li>SqueezeBERT is similar to BERT and therefore relies on the masked language modeling (MLM) objective. It is therefore
efficient at predicting masked tokens and at NLU in general, but is not optimal for text generation. Models trained
with a causal language modeling (CLM) objective are better in that regard.</li> <li>For best results when finetuning on sequence classification tasks, it is recommended to start with the
<em>squeezebert/squeezebert-mnli-headless</em> checkpoint.</li>`,pn,Ue,un,Fe,Po='<li><a href="../tasks/sequence_classification">Text classification task guide</a></li> <li><a href="../tasks/token_classification">Token classification task guide</a></li> <li><a href="../tasks/question_answering">Question answering task guide</a></li> <li><a href="../tasks/masked_language_modeling">Masked language modeling task guide</a></li> <li><a href="../tasks/multiple_choice">Multiple choice task guide</a></li>',mn,We,hn,W,Ze,Un,ft,Yo=`This is the configuration class to store the configuration of a <a href="/docs/transformers/v4.56.2/en/model_doc/squeezebert#transformers.SqueezeBertModel">SqueezeBertModel</a>. It is used to instantiate a
SqueezeBERT model according to the specified arguments, defining the model architecture. Instantiating a
configuration with the defaults will yield a similar configuration to that of the SqueezeBERT
<a href="https://huggingface.co/squeezebert/squeezebert-uncased" rel="nofollow">squeezebert/squeezebert-uncased</a> architecture.`,Fn,gt,Ao=`Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> and can be used to control the model outputs. Read the
documentation from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> for more information.`,Wn,de,fn,Ve,gn,B,Ie,Zn,_t,Do="Construct a SqueezeBERT tokenizer. Based on WordPiece.",Vn,bt,Oo=`This tokenizer inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer">PreTrainedTokenizer</a> which contains most of the main methods. Users should refer to
this superclass for more information regarding those methods.`,In,A,Ne,Nn,yt,Ko=`Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
adding special tokens. A SqueezeBERT sequence has the following format:`,Xn,Tt,es="<li>single sequence: <code>[CLS] X [SEP]</code></li> <li>pair of sequences: <code>[CLS] A [SEP] B [SEP]</code></li>",Rn,ce,Xe,Gn,kt,ts=`Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
special tokens using the tokenizer <code>prepare_for_model</code> method.`,Ln,D,Re,Qn,Mt,ns=`Create the token type IDs corresponding to the sequences passed. <a href="../glossary#token-type-ids">What are token type
IDs?</a>`,Hn,zt,os="Should be overridden in a subclass if the model has a special way of building those.",En,wt,Ge,_n,Le,bn,Z,Qe,Pn,vt,ss="Construct a “fast” SqueezeBERT tokenizer (backed by HuggingFace’s <em>tokenizers</em> library). Based on WordPiece.",Yn,qt,rs=`This tokenizer inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast">PreTrainedTokenizerFast</a> which contains most of the main methods. Users should
refer to this superclass for more information regarding those methods.`,An,O,He,Dn,Bt,as=`Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
adding special tokens. A SqueezeBERT sequence has the following format:`,On,$t,is="<li>single sequence: <code>[CLS] X [SEP]</code></li> <li>pair of sequences: <code>[CLS] A [SEP] B [SEP]</code></li>",yn,Ee,Tn,S,Pe,Kn,St,ls="The bare Squeezebert Model outputting raw hidden-states without any specific head on top.",eo,xt,ds=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,to,Ct,cs=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,no,K,Ye,oo,jt,ps='The <a href="/docs/transformers/v4.56.2/en/model_doc/squeezebert#transformers.SqueezeBertModel">SqueezeBertModel</a> forward method, overrides the <code>__call__</code> special method.',so,pe,kn,Ae,Mn,x,De,ro,Jt,us="The Squeezebert Model with a <code>language modeling</code> head on top.”",ao,Ut,ms=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,io,Ft,hs=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,lo,G,Oe,co,Wt,fs='The <a href="/docs/transformers/v4.56.2/en/model_doc/squeezebert#transformers.SqueezeBertForMaskedLM">SqueezeBertForMaskedLM</a> forward method, overrides the <code>__call__</code> special method.',po,ue,uo,me,zn,Ke,wn,C,et,mo,Zt,gs=`SqueezeBERT Model transformer with a sequence classification/regression head on top (a linear layer on top of the
pooled output) e.g. for GLUE tasks.`,ho,Vt,_s=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,fo,It,bs=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,go,F,tt,_o,Nt,ys='The <a href="/docs/transformers/v4.56.2/en/model_doc/squeezebert#transformers.SqueezeBertForSequenceClassification">SqueezeBertForSequenceClassification</a> forward method, overrides the <code>__call__</code> special method.',bo,he,yo,fe,To,ge,vn,nt,qn,j,ot,ko,Xt,Ts=`The Squeezebert Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
softmax) e.g. for RocStories/SWAG tasks.`,Mo,Rt,ks=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,zo,Gt,Ms=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,wo,L,st,vo,Lt,zs='The <a href="/docs/transformers/v4.56.2/en/model_doc/squeezebert#transformers.SqueezeBertForMultipleChoice">SqueezeBertForMultipleChoice</a> forward method, overrides the <code>__call__</code> special method.',qo,_e,Bo,be,Bn,rt,$n,J,at,$o,Qt,ws=`The Squeezebert transformer with a token classification head on top (a linear layer on top of the hidden-states
output) e.g. for Named-Entity-Recognition (NER) tasks.`,So,Ht,vs=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,xo,Et,qs=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Co,Q,it,jo,Pt,Bs='The <a href="/docs/transformers/v4.56.2/en/model_doc/squeezebert#transformers.SqueezeBertForTokenClassification">SqueezeBertForTokenClassification</a> forward method, overrides the <code>__call__</code> special method.',Jo,ye,Uo,Te,Sn,lt,xn,U,dt,Fo,Yt,$s=`The Squeezebert transformer with a span classification head on top for extractive question-answering tasks like
SQuAD (a linear layer on top of the hidden-states output to compute <code>span start logits</code> and <code>span end logits</code>).`,Wo,At,Ss=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Zo,Dt,xs=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Vo,H,ct,Io,Ot,Cs='The <a href="/docs/transformers/v4.56.2/en/model_doc/squeezebert#transformers.SqueezeBertForQuestionAnswering">SqueezeBertForQuestionAnswering</a> forward method, overrides the <code>__call__</code> special method.',No,ke,Xo,Me,Cn,pt,jn,en,Jn;return qe=new X({props:{title:"SqueezeBERT",local:"squeezebert",headingTag:"h1"}}),Be=new X({props:{title:"Overview",local:"overview",headingTag:"h2"}}),je=new X({props:{title:"Usage tips",local:"usage-tips",headingTag:"h2"}}),Ue=new X({props:{title:"Resources",local:"resources",headingTag:"h2"}}),We=new X({props:{title:"SqueezeBertConfig",local:"transformers.SqueezeBertConfig",headingTag:"h2"}}),Ze=new q({props:{name:"class transformers.SqueezeBertConfig",anchor:"transformers.SqueezeBertConfig",parameters:[{name:"vocab_size",val:" = 30522"},{name:"hidden_size",val:" = 768"},{name:"num_hidden_layers",val:" = 12"},{name:"num_attention_heads",val:" = 12"},{name:"intermediate_size",val:" = 3072"},{name:"hidden_act",val:" = 'gelu'"},{name:"hidden_dropout_prob",val:" = 0.1"},{name:"attention_probs_dropout_prob",val:" = 0.1"},{name:"max_position_embeddings",val:" = 512"},{name:"type_vocab_size",val:" = 2"},{name:"initializer_range",val:" = 0.02"},{name:"layer_norm_eps",val:" = 1e-12"},{name:"pad_token_id",val:" = 0"},{name:"embedding_size",val:" = 768"},{name:"q_groups",val:" = 4"},{name:"k_groups",val:" = 4"},{name:"v_groups",val:" = 4"},{name:"post_attention_groups",val:" = 1"},{name:"intermediate_groups",val:" = 4"},{name:"output_groups",val:" = 4"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.SqueezeBertConfig.vocab_size",description:`<strong>vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 30522) &#x2014;
Vocabulary size of the SqueezeBERT model. Defines the number of different tokens that can be represented by
the <code>inputs_ids</code> passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/squeezebert#transformers.SqueezeBertModel">SqueezeBertModel</a>.`,name:"vocab_size"},{anchor:"transformers.SqueezeBertConfig.hidden_size",description:`<strong>hidden_size</strong> (<code>int</code>, <em>optional</em>, defaults to 768) &#x2014;
Dimensionality of the encoder layers and the pooler layer.`,name:"hidden_size"},{anchor:"transformers.SqueezeBertConfig.num_hidden_layers",description:`<strong>num_hidden_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 12) &#x2014;
Number of hidden layers in the Transformer encoder.`,name:"num_hidden_layers"},{anchor:"transformers.SqueezeBertConfig.num_attention_heads",description:`<strong>num_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 12) &#x2014;
Number of attention heads for each attention layer in the Transformer encoder.`,name:"num_attention_heads"},{anchor:"transformers.SqueezeBertConfig.intermediate_size",description:`<strong>intermediate_size</strong> (<code>int</code>, <em>optional</em>, defaults to 3072) &#x2014;
Dimensionality of the &#x201C;intermediate&#x201D; (often named feed-forward) layer in the Transformer encoder.`,name:"intermediate_size"},{anchor:"transformers.SqueezeBertConfig.hidden_act",description:`<strong>hidden_act</strong> (<code>str</code> or <code>Callable</code>, <em>optional</em>, defaults to <code>&quot;gelu&quot;</code>) &#x2014;
The non-linear activation function (function or string) in the encoder and pooler. If string, <code>&quot;gelu&quot;</code>,
<code>&quot;relu&quot;</code>, <code>&quot;silu&quot;</code> and <code>&quot;gelu_new&quot;</code> are supported.`,name:"hidden_act"},{anchor:"transformers.SqueezeBertConfig.hidden_dropout_prob",description:`<strong>hidden_dropout_prob</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.`,name:"hidden_dropout_prob"},{anchor:"transformers.SqueezeBertConfig.attention_probs_dropout_prob",description:`<strong>attention_probs_dropout_prob</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout ratio for the attention probabilities.`,name:"attention_probs_dropout_prob"},{anchor:"transformers.SqueezeBertConfig.max_position_embeddings",description:`<strong>max_position_embeddings</strong> (<code>int</code>, <em>optional</em>, defaults to 512) &#x2014;
The maximum sequence length that this model might ever be used with. Typically set this to something large
just in case (e.g., 512 or 1024 or 2048).`,name:"max_position_embeddings"},{anchor:"transformers.SqueezeBertConfig.type_vocab_size",description:`<strong>type_vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 2) &#x2014;
The vocabulary size of the <code>token_type_ids</code> passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertModel">BertModel</a> or <code>TFBertModel</code>.`,name:"type_vocab_size"},{anchor:"transformers.SqueezeBertConfig.initializer_range",description:`<strong>initializer_range</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.`,name:"initializer_range"},{anchor:"transformers.SqueezeBertConfig.layer_norm_eps",description:"<strong>layer_norm_eps</strong> (<code>float</code>, <em>optional</em>, defaults to 1e-12) &#x2014;",name:"layer_norm_eps"},{anchor:"transformers.SqueezeBertConfig.pad_token_id",description:`<strong>pad_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 0) &#x2014;
The ID of the token in the word embedding to use as padding.`,name:"pad_token_id"},{anchor:"transformers.SqueezeBertConfig.embedding_size",description:`<strong>embedding_size</strong> (<code>int</code>, <em>optional</em>, defaults to 768) &#x2014;
The dimension of the word embedding vectors.`,name:"embedding_size"},{anchor:"transformers.SqueezeBertConfig.q_groups",description:`<strong>q_groups</strong> (<code>int</code>, <em>optional</em>, defaults to 4) &#x2014;
The number of groups in Q layer.`,name:"q_groups"},{anchor:"transformers.SqueezeBertConfig.k_groups",description:`<strong>k_groups</strong> (<code>int</code>, <em>optional</em>, defaults to 4) &#x2014;
The number of groups in K layer.`,name:"k_groups"},{anchor:"transformers.SqueezeBertConfig.v_groups",description:`<strong>v_groups</strong> (<code>int</code>, <em>optional</em>, defaults to 4) &#x2014;
The number of groups in V layer.`,name:"v_groups"},{anchor:"transformers.SqueezeBertConfig.post_attention_groups",description:`<strong>post_attention_groups</strong> (<code>int</code>, <em>optional</em>, defaults to 1) &#x2014;
The number of groups in the first feed forward network layer.`,name:"post_attention_groups"},{anchor:"transformers.SqueezeBertConfig.intermediate_groups",description:`<strong>intermediate_groups</strong> (<code>int</code>, <em>optional</em>, defaults to 4) &#x2014;
The number of groups in the second feed forward network layer.`,name:"intermediate_groups"},{anchor:"transformers.SqueezeBertConfig.output_groups",description:`<strong>output_groups</strong> (<code>int</code>, <em>optional</em>, defaults to 4) &#x2014;
The number of groups in the third feed forward network layer.`,name:"output_groups"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/squeezebert/configuration_squeezebert.py#L28"}}),de=new mt({props:{anchor:"transformers.SqueezeBertConfig.example",$$slots:{default:[Vs]},$$scope:{ctx:z}}}),Ve=new X({props:{title:"SqueezeBertTokenizer",local:"transformers.SqueezeBertTokenizer",headingTag:"h2"}}),Ie=new q({props:{name:"class transformers.SqueezeBertTokenizer",anchor:"transformers.SqueezeBertTokenizer",parameters:[{name:"vocab_file",val:""},{name:"do_lower_case",val:" = True"},{name:"do_basic_tokenize",val:" = True"},{name:"never_split",val:" = None"},{name:"unk_token",val:" = '[UNK]'"},{name:"sep_token",val:" = '[SEP]'"},{name:"pad_token",val:" = '[PAD]'"},{name:"cls_token",val:" = '[CLS]'"},{name:"mask_token",val:" = '[MASK]'"},{name:"tokenize_chinese_chars",val:" = True"},{name:"strip_accents",val:" = None"},{name:"clean_up_tokenization_spaces",val:" = True"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.SqueezeBertTokenizer.vocab_file",description:`<strong>vocab_file</strong> (<code>str</code>) &#x2014;
File containing the vocabulary.`,name:"vocab_file"},{anchor:"transformers.SqueezeBertTokenizer.do_lower_case",description:`<strong>do_lower_case</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to lowercase the input when tokenizing.`,name:"do_lower_case"},{anchor:"transformers.SqueezeBertTokenizer.do_basic_tokenize",description:`<strong>do_basic_tokenize</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to do basic tokenization before WordPiece.`,name:"do_basic_tokenize"},{anchor:"transformers.SqueezeBertTokenizer.never_split",description:`<strong>never_split</strong> (<code>Iterable</code>, <em>optional</em>) &#x2014;
Collection of tokens which will never be split during tokenization. Only has an effect when
<code>do_basic_tokenize=True</code>`,name:"never_split"},{anchor:"transformers.SqueezeBertTokenizer.unk_token",description:`<strong>unk_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;[UNK]&quot;</code>) &#x2014;
The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
token instead.`,name:"unk_token"},{anchor:"transformers.SqueezeBertTokenizer.sep_token",description:`<strong>sep_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;[SEP]&quot;</code>) &#x2014;
The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
sequence classification or for a text and a question for question answering. It is also used as the last
token of a sequence built with special tokens.`,name:"sep_token"},{anchor:"transformers.SqueezeBertTokenizer.pad_token",description:`<strong>pad_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;[PAD]&quot;</code>) &#x2014;
The token used for padding, for example when batching sequences of different lengths.`,name:"pad_token"},{anchor:"transformers.SqueezeBertTokenizer.cls_token",description:`<strong>cls_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;[CLS]&quot;</code>) &#x2014;
The classifier token which is used when doing sequence classification (classification of the whole sequence
instead of per-token classification). It is the first token of the sequence when built with special tokens.`,name:"cls_token"},{anchor:"transformers.SqueezeBertTokenizer.mask_token",description:`<strong>mask_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;[MASK]&quot;</code>) &#x2014;
The token used for masking values. This is the token used when training this model with masked language
modeling. This is the token which the model will try to predict.`,name:"mask_token"},{anchor:"transformers.SqueezeBertTokenizer.tokenize_chinese_chars",description:`<strong>tokenize_chinese_chars</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to tokenize Chinese characters.</p>
<p>This should likely be deactivated for Japanese (see this
<a href="https://github.com/huggingface/transformers/issues/328" rel="nofollow">issue</a>).`,name:"tokenize_chinese_chars"},{anchor:"transformers.SqueezeBertTokenizer.strip_accents",description:`<strong>strip_accents</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to strip all accents. If this option is not specified, then it will be determined by the
value for <code>lowercase</code> (as in the original SqueezeBERT).`,name:"strip_accents"},{anchor:"transformers.SqueezeBertTokenizer.clean_up_tokenization_spaces",description:`<strong>clean_up_tokenization_spaces</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to cleanup spaces after decoding, cleanup consists in removing potential artifacts like
extra spaces.`,name:"clean_up_tokenization_spaces"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/squeezebert/tokenization_squeezebert.py#L54"}}),Ne=new q({props:{name:"build_inputs_with_special_tokens",anchor:"transformers.SqueezeBertTokenizer.build_inputs_with_special_tokens",parameters:[{name:"token_ids_0",val:": list"},{name:"token_ids_1",val:": typing.Optional[list[int]] = None"}],parametersDescription:[{anchor:"transformers.SqueezeBertTokenizer.build_inputs_with_special_tokens.token_ids_0",description:`<strong>token_ids_0</strong> (<code>List[int]</code>) &#x2014;
List of IDs to which the special tokens will be added.`,name:"token_ids_0"},{anchor:"transformers.SqueezeBertTokenizer.build_inputs_with_special_tokens.token_ids_1",description:`<strong>token_ids_1</strong> (<code>List[int]</code>, <em>optional</em>) &#x2014;
Optional second list of IDs for sequence pairs.`,name:"token_ids_1"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/squeezebert/tokenization_squeezebert.py#L189",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>List of <a href="../glossary#input-ids">input IDs</a> with the appropriate special tokens.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>List[int]</code></p>
`}}),Xe=new q({props:{name:"get_special_tokens_mask",anchor:"transformers.SqueezeBertTokenizer.get_special_tokens_mask",parameters:[{name:"token_ids_0",val:": list"},{name:"token_ids_1",val:": typing.Optional[list[int]] = None"},{name:"already_has_special_tokens",val:": bool = False"}],parametersDescription:[{anchor:"transformers.SqueezeBertTokenizer.get_special_tokens_mask.token_ids_0",description:`<strong>token_ids_0</strong> (<code>List[int]</code>) &#x2014;
List of IDs.`,name:"token_ids_0"},{anchor:"transformers.SqueezeBertTokenizer.get_special_tokens_mask.token_ids_1",description:`<strong>token_ids_1</strong> (<code>List[int]</code>, <em>optional</em>) &#x2014;
Optional second list of IDs for sequence pairs.`,name:"token_ids_1"},{anchor:"transformers.SqueezeBertTokenizer.get_special_tokens_mask.already_has_special_tokens",description:`<strong>already_has_special_tokens</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not the token list is already formatted with special tokens for the model.`,name:"already_has_special_tokens"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/squeezebert/tokenization_squeezebert.py#L214",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>List[int]</code></p>
`}}),Re=new q({props:{name:"create_token_type_ids_from_sequences",anchor:"transformers.SqueezeBertTokenizer.create_token_type_ids_from_sequences",parameters:[{name:"token_ids_0",val:": list"},{name:"token_ids_1",val:": typing.Optional[list[int]] = None"}],parametersDescription:[{anchor:"transformers.SqueezeBertTokenizer.create_token_type_ids_from_sequences.token_ids_0",description:"<strong>token_ids_0</strong> (<code>list[int]</code>) &#x2014; The first tokenized sequence.",name:"token_ids_0"},{anchor:"transformers.SqueezeBertTokenizer.create_token_type_ids_from_sequences.token_ids_1",description:"<strong>token_ids_1</strong> (<code>list[int]</code>, <em>optional</em>) &#x2014; The second tokenized sequence.",name:"token_ids_1"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L3432",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>The token type ids.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>list[int]</code></p>
`}}),Ge=new q({props:{name:"save_vocabulary",anchor:"transformers.SqueezeBertTokenizer.save_vocabulary",parameters:[{name:"save_directory",val:": str"},{name:"filename_prefix",val:": typing.Optional[str] = None"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/squeezebert/tokenization_squeezebert.py#L242"}}),Le=new X({props:{title:"SqueezeBertTokenizerFast",local:"transformers.SqueezeBertTokenizerFast",headingTag:"h2"}}),Qe=new q({props:{name:"class transformers.SqueezeBertTokenizerFast",anchor:"transformers.SqueezeBertTokenizerFast",parameters:[{name:"vocab_file",val:" = None"},{name:"tokenizer_file",val:" = None"},{name:"do_lower_case",val:" = True"},{name:"unk_token",val:" = '[UNK]'"},{name:"sep_token",val:" = '[SEP]'"},{name:"pad_token",val:" = '[PAD]'"},{name:"cls_token",val:" = '[CLS]'"},{name:"mask_token",val:" = '[MASK]'"},{name:"tokenize_chinese_chars",val:" = True"},{name:"strip_accents",val:" = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.SqueezeBertTokenizerFast.vocab_file",description:`<strong>vocab_file</strong> (<code>str</code>) &#x2014;
File containing the vocabulary.`,name:"vocab_file"},{anchor:"transformers.SqueezeBertTokenizerFast.do_lower_case",description:`<strong>do_lower_case</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to lowercase the input when tokenizing.`,name:"do_lower_case"},{anchor:"transformers.SqueezeBertTokenizerFast.unk_token",description:`<strong>unk_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;[UNK]&quot;</code>) &#x2014;
The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
token instead.`,name:"unk_token"},{anchor:"transformers.SqueezeBertTokenizerFast.sep_token",description:`<strong>sep_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;[SEP]&quot;</code>) &#x2014;
The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
sequence classification or for a text and a question for question answering. It is also used as the last
token of a sequence built with special tokens.`,name:"sep_token"},{anchor:"transformers.SqueezeBertTokenizerFast.pad_token",description:`<strong>pad_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;[PAD]&quot;</code>) &#x2014;
The token used for padding, for example when batching sequences of different lengths.`,name:"pad_token"},{anchor:"transformers.SqueezeBertTokenizerFast.cls_token",description:`<strong>cls_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;[CLS]&quot;</code>) &#x2014;
The classifier token which is used when doing sequence classification (classification of the whole sequence
instead of per-token classification). It is the first token of the sequence when built with special tokens.`,name:"cls_token"},{anchor:"transformers.SqueezeBertTokenizerFast.mask_token",description:`<strong>mask_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;[MASK]&quot;</code>) &#x2014;
The token used for masking values. This is the token used when training this model with masked language
modeling. This is the token which the model will try to predict.`,name:"mask_token"},{anchor:"transformers.SqueezeBertTokenizerFast.clean_text",description:`<strong>clean_text</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to clean the text before tokenization by removing any control characters and replacing all
whitespaces by the classic one.`,name:"clean_text"},{anchor:"transformers.SqueezeBertTokenizerFast.tokenize_chinese_chars",description:`<strong>tokenize_chinese_chars</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to tokenize Chinese characters. This should likely be deactivated for Japanese (see <a href="https://github.com/huggingface/transformers/issues/328" rel="nofollow">this
issue</a>).`,name:"tokenize_chinese_chars"},{anchor:"transformers.SqueezeBertTokenizerFast.strip_accents",description:`<strong>strip_accents</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to strip all accents. If this option is not specified, then it will be determined by the
value for <code>lowercase</code> (as in the original SqueezeBERT).`,name:"strip_accents"},{anchor:"transformers.SqueezeBertTokenizerFast.wordpieces_prefix",description:`<strong>wordpieces_prefix</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;##&quot;</code>) &#x2014;
The prefix for subwords.`,name:"wordpieces_prefix"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/squeezebert/tokenization_squeezebert_fast.py#L33"}}),He=new q({props:{name:"build_inputs_with_special_tokens",anchor:"transformers.SqueezeBertTokenizerFast.build_inputs_with_special_tokens",parameters:[{name:"token_ids_0",val:""},{name:"token_ids_1",val:" = None"}],parametersDescription:[{anchor:"transformers.SqueezeBertTokenizerFast.build_inputs_with_special_tokens.token_ids_0",description:`<strong>token_ids_0</strong> (<code>List[int]</code>) &#x2014;
List of IDs to which the special tokens will be added.`,name:"token_ids_0"},{anchor:"transformers.SqueezeBertTokenizerFast.build_inputs_with_special_tokens.token_ids_1",description:`<strong>token_ids_1</strong> (<code>List[int]</code>, <em>optional</em>) &#x2014;
Optional second list of IDs for sequence pairs.`,name:"token_ids_1"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/squeezebert/tokenization_squeezebert_fast.py#L118",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>List of <a href="../glossary#input-ids">input IDs</a> with the appropriate special tokens.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>List[int]</code></p>
`}}),Ee=new X({props:{title:"SqueezeBertModel",local:"transformers.SqueezeBertModel",headingTag:"h2"}}),Pe=new q({props:{name:"class transformers.SqueezeBertModel",anchor:"transformers.SqueezeBertModel",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.SqueezeBertModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/squeezebert#transformers.SqueezeBertModel">SqueezeBertModel</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/squeezebert/modeling_squeezebert.py#L443"}}),Ye=new q({props:{name:"forward",anchor:"transformers.SqueezeBertModel.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.SqueezeBertModel.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.SqueezeBertModel.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.SqueezeBertModel.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.SqueezeBertModel.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.SqueezeBertModel.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.SqueezeBertModel.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.SqueezeBertModel.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.SqueezeBertModel.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.SqueezeBertModel.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/squeezebert/modeling_squeezebert.py#L468",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling"
>transformers.modeling_outputs.BaseModelOutputWithPooling</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/squeezebert#transformers.SqueezeBertConfig"
>SqueezeBertConfig</a>) and inputs.</p>
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
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling"
>transformers.modeling_outputs.BaseModelOutputWithPooling</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),pe=new Kt({props:{$$slots:{default:[Is]},$$scope:{ctx:z}}}),Ae=new X({props:{title:"SqueezeBertForMaskedLM",local:"transformers.SqueezeBertForMaskedLM",headingTag:"h2"}}),De=new q({props:{name:"class transformers.SqueezeBertForMaskedLM",anchor:"transformers.SqueezeBertForMaskedLM",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.SqueezeBertForMaskedLM.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/squeezebert#transformers.SqueezeBertForMaskedLM">SqueezeBertForMaskedLM</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/squeezebert/modeling_squeezebert.py#L538"}}),Oe=new q({props:{name:"forward",anchor:"transformers.SqueezeBertForMaskedLM.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"labels",val:": typing.Optional[torch.Tensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.SqueezeBertForMaskedLM.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.SqueezeBertForMaskedLM.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.SqueezeBertForMaskedLM.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.SqueezeBertForMaskedLM.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.SqueezeBertForMaskedLM.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.SqueezeBertForMaskedLM.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.SqueezeBertForMaskedLM.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should be in <code>[-100, 0, ..., config.vocab_size]</code> (see <code>input_ids</code> docstring) Tokens with indices set to <code>-100</code> are ignored (masked), the
loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>`,name:"labels"},{anchor:"transformers.SqueezeBertForMaskedLM.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.SqueezeBertForMaskedLM.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.SqueezeBertForMaskedLM.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/squeezebert/modeling_squeezebert.py#L557",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.MaskedLMOutput"
>transformers.modeling_outputs.MaskedLMOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/squeezebert#transformers.SqueezeBertConfig"
>SqueezeBertConfig</a>) and inputs.</p>
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
`}}),ue=new Kt({props:{$$slots:{default:[Ns]},$$scope:{ctx:z}}}),me=new mt({props:{anchor:"transformers.SqueezeBertForMaskedLM.forward.example",$$slots:{default:[Xs]},$$scope:{ctx:z}}}),Ke=new X({props:{title:"SqueezeBertForSequenceClassification",local:"transformers.SqueezeBertForSequenceClassification",headingTag:"h2"}}),et=new q({props:{name:"class transformers.SqueezeBertForSequenceClassification",anchor:"transformers.SqueezeBertForSequenceClassification",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.SqueezeBertForSequenceClassification.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/squeezebert#transformers.SqueezeBertForSequenceClassification">SqueezeBertForSequenceClassification</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/squeezebert/modeling_squeezebert.py#L617"}}),tt=new q({props:{name:"forward",anchor:"transformers.SqueezeBertForSequenceClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"labels",val:": typing.Optional[torch.Tensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.SqueezeBertForSequenceClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.SqueezeBertForSequenceClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.SqueezeBertForSequenceClassification.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.SqueezeBertForSequenceClassification.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.SqueezeBertForSequenceClassification.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.SqueezeBertForSequenceClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.SqueezeBertForSequenceClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the sequence classification/regression loss. Indices should be in <code>[0, ..., config.num_labels - 1]</code>. If <code>config.num_labels == 1</code> a regression loss is computed (Mean-Square loss), If
<code>config.num_labels &gt; 1</code> a classification loss is computed (Cross-Entropy).`,name:"labels"},{anchor:"transformers.SqueezeBertForSequenceClassification.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.SqueezeBertForSequenceClassification.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.SqueezeBertForSequenceClassification.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/squeezebert/modeling_squeezebert.py#L630",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput"
>transformers.modeling_outputs.SequenceClassifierOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/squeezebert#transformers.SqueezeBertConfig"
>SqueezeBertConfig</a>) and inputs.</p>
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
`}}),he=new Kt({props:{$$slots:{default:[Rs]},$$scope:{ctx:z}}}),fe=new mt({props:{anchor:"transformers.SqueezeBertForSequenceClassification.forward.example",$$slots:{default:[Gs]},$$scope:{ctx:z}}}),ge=new mt({props:{anchor:"transformers.SqueezeBertForSequenceClassification.forward.example-2",$$slots:{default:[Ls]},$$scope:{ctx:z}}}),nt=new X({props:{title:"SqueezeBertForMultipleChoice",local:"transformers.SqueezeBertForMultipleChoice",headingTag:"h2"}}),ot=new q({props:{name:"class transformers.SqueezeBertForMultipleChoice",anchor:"transformers.SqueezeBertForMultipleChoice",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.SqueezeBertForMultipleChoice.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/squeezebert#transformers.SqueezeBertForMultipleChoice">SqueezeBertForMultipleChoice</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/squeezebert/modeling_squeezebert.py#L705"}}),st=new q({props:{name:"forward",anchor:"transformers.SqueezeBertForMultipleChoice.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"labels",val:": typing.Optional[torch.Tensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.SqueezeBertForMultipleChoice.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, num_choices, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.SqueezeBertForMultipleChoice.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.SqueezeBertForMultipleChoice.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, num_choices, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.SqueezeBertForMultipleChoice.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, num_choices, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.max_position_embeddings - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.SqueezeBertForMultipleChoice.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.SqueezeBertForMultipleChoice.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_choices, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.SqueezeBertForMultipleChoice.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the multiple choice classification loss. Indices should be in <code>[0, ..., num_choices-1]</code> where <em>num_choices</em> is the size of the second dimension of the input tensors. (see
<em>input_ids</em> above)`,name:"labels"},{anchor:"transformers.SqueezeBertForMultipleChoice.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.SqueezeBertForMultipleChoice.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.SqueezeBertForMultipleChoice.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/squeezebert/modeling_squeezebert.py#L716",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.MultipleChoiceModelOutput"
>transformers.modeling_outputs.MultipleChoiceModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/squeezebert#transformers.SqueezeBertConfig"
>SqueezeBertConfig</a>) and inputs.</p>
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
`}}),_e=new Kt({props:{$$slots:{default:[Qs]},$$scope:{ctx:z}}}),be=new mt({props:{anchor:"transformers.SqueezeBertForMultipleChoice.forward.example",$$slots:{default:[Hs]},$$scope:{ctx:z}}}),rt=new X({props:{title:"SqueezeBertForTokenClassification",local:"transformers.SqueezeBertForTokenClassification",headingTag:"h2"}}),at=new q({props:{name:"class transformers.SqueezeBertForTokenClassification",anchor:"transformers.SqueezeBertForTokenClassification",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.SqueezeBertForTokenClassification.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/squeezebert#transformers.SqueezeBertForTokenClassification">SqueezeBertForTokenClassification</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/squeezebert/modeling_squeezebert.py#L809"}}),it=new q({props:{name:"forward",anchor:"transformers.SqueezeBertForTokenClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"labels",val:": typing.Optional[torch.Tensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.SqueezeBertForTokenClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.SqueezeBertForTokenClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.SqueezeBertForTokenClassification.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.SqueezeBertForTokenClassification.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.SqueezeBertForTokenClassification.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.SqueezeBertForTokenClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.SqueezeBertForTokenClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the token classification loss. Indices should be in <code>[0, ..., config.num_labels - 1]</code>.`,name:"labels"},{anchor:"transformers.SqueezeBertForTokenClassification.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.SqueezeBertForTokenClassification.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.SqueezeBertForTokenClassification.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/squeezebert/modeling_squeezebert.py#L821",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput"
>transformers.modeling_outputs.TokenClassifierOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/squeezebert#transformers.SqueezeBertConfig"
>SqueezeBertConfig</a>) and inputs.</p>
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
`}}),ye=new Kt({props:{$$slots:{default:[Es]},$$scope:{ctx:z}}}),Te=new mt({props:{anchor:"transformers.SqueezeBertForTokenClassification.forward.example",$$slots:{default:[Ps]},$$scope:{ctx:z}}}),lt=new X({props:{title:"SqueezeBertForQuestionAnswering",local:"transformers.SqueezeBertForQuestionAnswering",headingTag:"h2"}}),dt=new q({props:{name:"class transformers.SqueezeBertForQuestionAnswering",anchor:"transformers.SqueezeBertForQuestionAnswering",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.SqueezeBertForQuestionAnswering.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/squeezebert#transformers.SqueezeBertForQuestionAnswering">SqueezeBertForQuestionAnswering</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/squeezebert/modeling_squeezebert.py#L876"}}),ct=new q({props:{name:"forward",anchor:"transformers.SqueezeBertForQuestionAnswering.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"start_positions",val:": typing.Optional[torch.Tensor] = None"},{name:"end_positions",val:": typing.Optional[torch.Tensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.SqueezeBertForQuestionAnswering.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.SqueezeBertForQuestionAnswering.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.SqueezeBertForQuestionAnswering.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.SqueezeBertForQuestionAnswering.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.SqueezeBertForQuestionAnswering.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.SqueezeBertForQuestionAnswering.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.SqueezeBertForQuestionAnswering.forward.start_positions",description:`<strong>start_positions</strong> (<code>torch.Tensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for position (index) of the start of the labelled span for computing the token classification loss.
Positions are clamped to the length of the sequence (<code>sequence_length</code>). Position outside of the sequence
are not taken into account for computing the loss.`,name:"start_positions"},{anchor:"transformers.SqueezeBertForQuestionAnswering.forward.end_positions",description:`<strong>end_positions</strong> (<code>torch.Tensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for position (index) of the end of the labelled span for computing the token classification loss.
Positions are clamped to the length of the sequence (<code>sequence_length</code>). Position outside of the sequence
are not taken into account for computing the loss.`,name:"end_positions"},{anchor:"transformers.SqueezeBertForQuestionAnswering.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.SqueezeBertForQuestionAnswering.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.SqueezeBertForQuestionAnswering.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/squeezebert/modeling_squeezebert.py#L887",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.QuestionAnsweringModelOutput"
>transformers.modeling_outputs.QuestionAnsweringModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/squeezebert#transformers.SqueezeBertConfig"
>SqueezeBertConfig</a>) and inputs.</p>
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
`}}),ke=new Kt({props:{$$slots:{default:[Ys]},$$scope:{ctx:z}}}),Me=new mt({props:{anchor:"transformers.SqueezeBertForQuestionAnswering.forward.example",$$slots:{default:[As]},$$scope:{ctx:z}}}),pt=new Zs({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/squeezebert.md"}}),{c(){t=l("meta"),T=s(),c=l("p"),p=s(),k=l("p"),k.innerHTML=o,M=s(),h(qe.$$.fragment),nn=s(),le=l("div"),le.innerHTML=Ro,on=s(),h(Be.$$.fragment),sn=s(),$e=l("p"),$e.innerHTML=Go,rn=s(),Se=l("p"),Se.textContent=Lo,an=s(),xe=l("p"),xe.innerHTML=Qo,ln=s(),Ce=l("p"),Ce.innerHTML=Ho,dn=s(),h(je.$$.fragment),cn=s(),Je=l("ul"),Je.innerHTML=Eo,pn=s(),h(Ue.$$.fragment),un=s(),Fe=l("ul"),Fe.innerHTML=Po,mn=s(),h(We.$$.fragment),hn=s(),W=l("div"),h(Ze.$$.fragment),Un=s(),ft=l("p"),ft.innerHTML=Yo,Fn=s(),gt=l("p"),gt.innerHTML=Ao,Wn=s(),h(de.$$.fragment),fn=s(),h(Ve.$$.fragment),gn=s(),B=l("div"),h(Ie.$$.fragment),Zn=s(),_t=l("p"),_t.textContent=Do,Vn=s(),bt=l("p"),bt.innerHTML=Oo,In=s(),A=l("div"),h(Ne.$$.fragment),Nn=s(),yt=l("p"),yt.textContent=Ko,Xn=s(),Tt=l("ul"),Tt.innerHTML=es,Rn=s(),ce=l("div"),h(Xe.$$.fragment),Gn=s(),kt=l("p"),kt.innerHTML=ts,Ln=s(),D=l("div"),h(Re.$$.fragment),Qn=s(),Mt=l("p"),Mt.innerHTML=ns,Hn=s(),zt=l("p"),zt.textContent=os,En=s(),wt=l("div"),h(Ge.$$.fragment),_n=s(),h(Le.$$.fragment),bn=s(),Z=l("div"),h(Qe.$$.fragment),Pn=s(),vt=l("p"),vt.innerHTML=ss,Yn=s(),qt=l("p"),qt.innerHTML=rs,An=s(),O=l("div"),h(He.$$.fragment),Dn=s(),Bt=l("p"),Bt.textContent=as,On=s(),$t=l("ul"),$t.innerHTML=is,yn=s(),h(Ee.$$.fragment),Tn=s(),S=l("div"),h(Pe.$$.fragment),Kn=s(),St=l("p"),St.textContent=ls,eo=s(),xt=l("p"),xt.innerHTML=ds,to=s(),Ct=l("p"),Ct.innerHTML=cs,no=s(),K=l("div"),h(Ye.$$.fragment),oo=s(),jt=l("p"),jt.innerHTML=ps,so=s(),h(pe.$$.fragment),kn=s(),h(Ae.$$.fragment),Mn=s(),x=l("div"),h(De.$$.fragment),ro=s(),Jt=l("p"),Jt.innerHTML=us,ao=s(),Ut=l("p"),Ut.innerHTML=ms,io=s(),Ft=l("p"),Ft.innerHTML=hs,lo=s(),G=l("div"),h(Oe.$$.fragment),co=s(),Wt=l("p"),Wt.innerHTML=fs,po=s(),h(ue.$$.fragment),uo=s(),h(me.$$.fragment),zn=s(),h(Ke.$$.fragment),wn=s(),C=l("div"),h(et.$$.fragment),mo=s(),Zt=l("p"),Zt.textContent=gs,ho=s(),Vt=l("p"),Vt.innerHTML=_s,fo=s(),It=l("p"),It.innerHTML=bs,go=s(),F=l("div"),h(tt.$$.fragment),_o=s(),Nt=l("p"),Nt.innerHTML=ys,bo=s(),h(he.$$.fragment),yo=s(),h(fe.$$.fragment),To=s(),h(ge.$$.fragment),vn=s(),h(nt.$$.fragment),qn=s(),j=l("div"),h(ot.$$.fragment),ko=s(),Xt=l("p"),Xt.textContent=Ts,Mo=s(),Rt=l("p"),Rt.innerHTML=ks,zo=s(),Gt=l("p"),Gt.innerHTML=Ms,wo=s(),L=l("div"),h(st.$$.fragment),vo=s(),Lt=l("p"),Lt.innerHTML=zs,qo=s(),h(_e.$$.fragment),Bo=s(),h(be.$$.fragment),Bn=s(),h(rt.$$.fragment),$n=s(),J=l("div"),h(at.$$.fragment),$o=s(),Qt=l("p"),Qt.textContent=ws,So=s(),Ht=l("p"),Ht.innerHTML=vs,xo=s(),Et=l("p"),Et.innerHTML=qs,Co=s(),Q=l("div"),h(it.$$.fragment),jo=s(),Pt=l("p"),Pt.innerHTML=Bs,Jo=s(),h(ye.$$.fragment),Uo=s(),h(Te.$$.fragment),Sn=s(),h(lt.$$.fragment),xn=s(),U=l("div"),h(dt.$$.fragment),Fo=s(),Yt=l("p"),Yt.innerHTML=$s,Wo=s(),At=l("p"),At.innerHTML=Ss,Zo=s(),Dt=l("p"),Dt.innerHTML=xs,Vo=s(),H=l("div"),h(ct.$$.fragment),Io=s(),Ot=l("p"),Ot.innerHTML=Cs,No=s(),h(ke.$$.fragment),Xo=s(),h(Me.$$.fragment),Cn=s(),h(pt.$$.fragment),jn=s(),en=l("p"),this.h()},l(e){const a=Ws("svelte-u9bgzb",document.head);t=d(a,"META",{name:!0,content:!0}),a.forEach(i),T=r(e),c=d(e,"P",{}),v(c).forEach(i),p=r(e),k=d(e,"P",{"data-svelte-h":!0}),m(k)!=="svelte-12wa38d"&&(k.innerHTML=o),M=r(e),f(qe.$$.fragment,e),nn=r(e),le=d(e,"DIV",{class:!0,"data-svelte-h":!0}),m(le)!=="svelte-13t8s2t"&&(le.innerHTML=Ro),on=r(e),f(Be.$$.fragment,e),sn=r(e),$e=d(e,"P",{"data-svelte-h":!0}),m($e)!=="svelte-1yash9a"&&($e.innerHTML=Go),rn=r(e),Se=d(e,"P",{"data-svelte-h":!0}),m(Se)!=="svelte-vfdo9a"&&(Se.textContent=Lo),an=r(e),xe=d(e,"P",{"data-svelte-h":!0}),m(xe)!=="svelte-1emzktv"&&(xe.innerHTML=Qo),ln=r(e),Ce=d(e,"P",{"data-svelte-h":!0}),m(Ce)!=="svelte-1e4x21"&&(Ce.innerHTML=Ho),dn=r(e),f(je.$$.fragment,e),cn=r(e),Je=d(e,"UL",{"data-svelte-h":!0}),m(Je)!=="svelte-6f7ytb"&&(Je.innerHTML=Eo),pn=r(e),f(Ue.$$.fragment,e),un=r(e),Fe=d(e,"UL",{"data-svelte-h":!0}),m(Fe)!=="svelte-mgusi3"&&(Fe.innerHTML=Po),mn=r(e),f(We.$$.fragment,e),hn=r(e),W=d(e,"DIV",{class:!0});var E=v(W);f(Ze.$$.fragment,E),Un=r(E),ft=d(E,"P",{"data-svelte-h":!0}),m(ft)!=="svelte-9c1eiw"&&(ft.innerHTML=Yo),Fn=r(E),gt=d(E,"P",{"data-svelte-h":!0}),m(gt)!=="svelte-1ek1ss9"&&(gt.innerHTML=Ao),Wn=r(E),f(de.$$.fragment,E),E.forEach(i),fn=r(e),f(Ve.$$.fragment,e),gn=r(e),B=d(e,"DIV",{class:!0});var $=v(B);f(Ie.$$.fragment,$),Zn=r($),_t=d($,"P",{"data-svelte-h":!0}),m(_t)!=="svelte-1jixb9u"&&(_t.textContent=Do),Vn=r($),bt=d($,"P",{"data-svelte-h":!0}),m(bt)!=="svelte-ntrhio"&&(bt.innerHTML=Oo),In=r($),A=d($,"DIV",{class:!0});var se=v(A);f(Ne.$$.fragment,se),Nn=r(se),yt=d(se,"P",{"data-svelte-h":!0}),m(yt)!=="svelte-19k3keo"&&(yt.textContent=Ko),Xn=r(se),Tt=d(se,"UL",{"data-svelte-h":!0}),m(Tt)!=="svelte-xi6653"&&(Tt.innerHTML=es),se.forEach(i),Rn=r($),ce=d($,"DIV",{class:!0});var ut=v(ce);f(Xe.$$.fragment,ut),Gn=r(ut),kt=d(ut,"P",{"data-svelte-h":!0}),m(kt)!=="svelte-1f4f5kp"&&(kt.innerHTML=ts),ut.forEach(i),Ln=r($),D=d($,"DIV",{class:!0});var re=v(D);f(Re.$$.fragment,re),Qn=r(re),Mt=d(re,"P",{"data-svelte-h":!0}),m(Mt)!=="svelte-zj1vf1"&&(Mt.innerHTML=ns),Hn=r(re),zt=d(re,"P",{"data-svelte-h":!0}),m(zt)!=="svelte-9vptpw"&&(zt.textContent=os),re.forEach(i),En=r($),wt=d($,"DIV",{class:!0});var tn=v(wt);f(Ge.$$.fragment,tn),tn.forEach(i),$.forEach(i),_n=r(e),f(Le.$$.fragment,e),bn=r(e),Z=d(e,"DIV",{class:!0});var P=v(Z);f(Qe.$$.fragment,P),Pn=r(P),vt=d(P,"P",{"data-svelte-h":!0}),m(vt)!=="svelte-56e233"&&(vt.innerHTML=ss),Yn=r(P),qt=d(P,"P",{"data-svelte-h":!0}),m(qt)!=="svelte-gxzj9w"&&(qt.innerHTML=rs),An=r(P),O=d(P,"DIV",{class:!0});var ae=v(O);f(He.$$.fragment,ae),Dn=r(ae),Bt=d(ae,"P",{"data-svelte-h":!0}),m(Bt)!=="svelte-19k3keo"&&(Bt.textContent=as),On=r(ae),$t=d(ae,"UL",{"data-svelte-h":!0}),m($t)!=="svelte-xi6653"&&($t.innerHTML=is),ae.forEach(i),P.forEach(i),yn=r(e),f(Ee.$$.fragment,e),Tn=r(e),S=d(e,"DIV",{class:!0});var V=v(S);f(Pe.$$.fragment,V),Kn=r(V),St=d(V,"P",{"data-svelte-h":!0}),m(St)!=="svelte-wxutwf"&&(St.textContent=ls),eo=r(V),xt=d(V,"P",{"data-svelte-h":!0}),m(xt)!=="svelte-q52n56"&&(xt.innerHTML=ds),to=r(V),Ct=d(V,"P",{"data-svelte-h":!0}),m(Ct)!=="svelte-hswkmf"&&(Ct.innerHTML=cs),no=r(V),K=d(V,"DIV",{class:!0});var ie=v(K);f(Ye.$$.fragment,ie),oo=r(ie),jt=d(ie,"P",{"data-svelte-h":!0}),m(jt)!=="svelte-qcof12"&&(jt.innerHTML=ps),so=r(ie),f(pe.$$.fragment,ie),ie.forEach(i),V.forEach(i),kn=r(e),f(Ae.$$.fragment,e),Mn=r(e),x=d(e,"DIV",{class:!0});var I=v(x);f(De.$$.fragment,I),ro=r(I),Jt=d(I,"P",{"data-svelte-h":!0}),m(Jt)!=="svelte-xvgdrd"&&(Jt.innerHTML=us),ao=r(I),Ut=d(I,"P",{"data-svelte-h":!0}),m(Ut)!=="svelte-q52n56"&&(Ut.innerHTML=ms),io=r(I),Ft=d(I,"P",{"data-svelte-h":!0}),m(Ft)!=="svelte-hswkmf"&&(Ft.innerHTML=hs),lo=r(I),G=d(I,"DIV",{class:!0});var Y=v(G);f(Oe.$$.fragment,Y),co=r(Y),Wt=d(Y,"P",{"data-svelte-h":!0}),m(Wt)!=="svelte-19yijue"&&(Wt.innerHTML=fs),po=r(Y),f(ue.$$.fragment,Y),uo=r(Y),f(me.$$.fragment,Y),Y.forEach(i),I.forEach(i),zn=r(e),f(Ke.$$.fragment,e),wn=r(e),C=d(e,"DIV",{class:!0});var N=v(C);f(et.$$.fragment,N),mo=r(N),Zt=d(N,"P",{"data-svelte-h":!0}),m(Zt)!=="svelte-13ur6tp"&&(Zt.textContent=gs),ho=r(N),Vt=d(N,"P",{"data-svelte-h":!0}),m(Vt)!=="svelte-q52n56"&&(Vt.innerHTML=_s),fo=r(N),It=d(N,"P",{"data-svelte-h":!0}),m(It)!=="svelte-hswkmf"&&(It.innerHTML=bs),go=r(N),F=d(N,"DIV",{class:!0});var ee=v(F);f(tt.$$.fragment,ee),_o=r(ee),Nt=d(ee,"P",{"data-svelte-h":!0}),m(Nt)!=="svelte-1i4qtc"&&(Nt.innerHTML=ys),bo=r(ee),f(he.$$.fragment,ee),yo=r(ee),f(fe.$$.fragment,ee),To=r(ee),f(ge.$$.fragment,ee),ee.forEach(i),N.forEach(i),vn=r(e),f(nt.$$.fragment,e),qn=r(e),j=d(e,"DIV",{class:!0});var te=v(j);f(ot.$$.fragment,te),ko=r(te),Xt=d(te,"P",{"data-svelte-h":!0}),m(Xt)!=="svelte-1qjmtne"&&(Xt.textContent=Ts),Mo=r(te),Rt=d(te,"P",{"data-svelte-h":!0}),m(Rt)!=="svelte-q52n56"&&(Rt.innerHTML=ks),zo=r(te),Gt=d(te,"P",{"data-svelte-h":!0}),m(Gt)!=="svelte-hswkmf"&&(Gt.innerHTML=Ms),wo=r(te),L=d(te,"DIV",{class:!0});var ze=v(L);f(st.$$.fragment,ze),vo=r(ze),Lt=d(ze,"P",{"data-svelte-h":!0}),m(Lt)!=="svelte-r9abpc"&&(Lt.innerHTML=zs),qo=r(ze),f(_e.$$.fragment,ze),Bo=r(ze),f(be.$$.fragment,ze),ze.forEach(i),te.forEach(i),Bn=r(e),f(rt.$$.fragment,e),$n=r(e),J=d(e,"DIV",{class:!0});var ne=v(J);f(at.$$.fragment,ne),$o=r(ne),Qt=d(ne,"P",{"data-svelte-h":!0}),m(Qt)!=="svelte-15mfzhh"&&(Qt.textContent=ws),So=r(ne),Ht=d(ne,"P",{"data-svelte-h":!0}),m(Ht)!=="svelte-q52n56"&&(Ht.innerHTML=vs),xo=r(ne),Et=d(ne,"P",{"data-svelte-h":!0}),m(Et)!=="svelte-hswkmf"&&(Et.innerHTML=qs),Co=r(ne),Q=d(ne,"DIV",{class:!0});var we=v(Q);f(it.$$.fragment,we),jo=r(we),Pt=d(we,"P",{"data-svelte-h":!0}),m(Pt)!=="svelte-3np8wq"&&(Pt.innerHTML=Bs),Jo=r(we),f(ye.$$.fragment,we),Uo=r(we),f(Te.$$.fragment,we),we.forEach(i),ne.forEach(i),Sn=r(e),f(lt.$$.fragment,e),xn=r(e),U=d(e,"DIV",{class:!0});var oe=v(U);f(dt.$$.fragment,oe),Fo=r(oe),Yt=d(oe,"P",{"data-svelte-h":!0}),m(Yt)!=="svelte-10dp90u"&&(Yt.innerHTML=$s),Wo=r(oe),At=d(oe,"P",{"data-svelte-h":!0}),m(At)!=="svelte-q52n56"&&(At.innerHTML=Ss),Zo=r(oe),Dt=d(oe,"P",{"data-svelte-h":!0}),m(Dt)!=="svelte-hswkmf"&&(Dt.innerHTML=xs),Vo=r(oe),H=d(oe,"DIV",{class:!0});var ve=v(H);f(ct.$$.fragment,ve),Io=r(ve),Ot=d(ve,"P",{"data-svelte-h":!0}),m(Ot)!=="svelte-16qvjfa"&&(Ot.innerHTML=Cs),No=r(ve),f(ke.$$.fragment,ve),Xo=r(ve),f(Me.$$.fragment,ve),ve.forEach(i),oe.forEach(i),Cn=r(e),f(pt.$$.fragment,e),jn=r(e),en=d(e,"P",{}),v(en).forEach(i),this.h()},h(){w(t,"name","hf:doc:metadata"),w(t,"content",Os),w(le,"class","flex flex-wrap space-x-1"),w(W,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),w(A,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),w(ce,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),w(D,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),w(wt,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),w(B,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),w(O,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),w(Z,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),w(K,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),w(S,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),w(G,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),w(x,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),w(F,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),w(C,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),w(L,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),w(j,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),w(Q,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),w(J,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),w(H,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),w(U,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,a){n(document.head,t),u(e,T,a),u(e,c,a),u(e,p,a),u(e,k,a),u(e,M,a),g(qe,e,a),u(e,nn,a),u(e,le,a),u(e,on,a),g(Be,e,a),u(e,sn,a),u(e,$e,a),u(e,rn,a),u(e,Se,a),u(e,an,a),u(e,xe,a),u(e,ln,a),u(e,Ce,a),u(e,dn,a),g(je,e,a),u(e,cn,a),u(e,Je,a),u(e,pn,a),g(Ue,e,a),u(e,un,a),u(e,Fe,a),u(e,mn,a),g(We,e,a),u(e,hn,a),u(e,W,a),g(Ze,W,null),n(W,Un),n(W,ft),n(W,Fn),n(W,gt),n(W,Wn),g(de,W,null),u(e,fn,a),g(Ve,e,a),u(e,gn,a),u(e,B,a),g(Ie,B,null),n(B,Zn),n(B,_t),n(B,Vn),n(B,bt),n(B,In),n(B,A),g(Ne,A,null),n(A,Nn),n(A,yt),n(A,Xn),n(A,Tt),n(B,Rn),n(B,ce),g(Xe,ce,null),n(ce,Gn),n(ce,kt),n(B,Ln),n(B,D),g(Re,D,null),n(D,Qn),n(D,Mt),n(D,Hn),n(D,zt),n(B,En),n(B,wt),g(Ge,wt,null),u(e,_n,a),g(Le,e,a),u(e,bn,a),u(e,Z,a),g(Qe,Z,null),n(Z,Pn),n(Z,vt),n(Z,Yn),n(Z,qt),n(Z,An),n(Z,O),g(He,O,null),n(O,Dn),n(O,Bt),n(O,On),n(O,$t),u(e,yn,a),g(Ee,e,a),u(e,Tn,a),u(e,S,a),g(Pe,S,null),n(S,Kn),n(S,St),n(S,eo),n(S,xt),n(S,to),n(S,Ct),n(S,no),n(S,K),g(Ye,K,null),n(K,oo),n(K,jt),n(K,so),g(pe,K,null),u(e,kn,a),g(Ae,e,a),u(e,Mn,a),u(e,x,a),g(De,x,null),n(x,ro),n(x,Jt),n(x,ao),n(x,Ut),n(x,io),n(x,Ft),n(x,lo),n(x,G),g(Oe,G,null),n(G,co),n(G,Wt),n(G,po),g(ue,G,null),n(G,uo),g(me,G,null),u(e,zn,a),g(Ke,e,a),u(e,wn,a),u(e,C,a),g(et,C,null),n(C,mo),n(C,Zt),n(C,ho),n(C,Vt),n(C,fo),n(C,It),n(C,go),n(C,F),g(tt,F,null),n(F,_o),n(F,Nt),n(F,bo),g(he,F,null),n(F,yo),g(fe,F,null),n(F,To),g(ge,F,null),u(e,vn,a),g(nt,e,a),u(e,qn,a),u(e,j,a),g(ot,j,null),n(j,ko),n(j,Xt),n(j,Mo),n(j,Rt),n(j,zo),n(j,Gt),n(j,wo),n(j,L),g(st,L,null),n(L,vo),n(L,Lt),n(L,qo),g(_e,L,null),n(L,Bo),g(be,L,null),u(e,Bn,a),g(rt,e,a),u(e,$n,a),u(e,J,a),g(at,J,null),n(J,$o),n(J,Qt),n(J,So),n(J,Ht),n(J,xo),n(J,Et),n(J,Co),n(J,Q),g(it,Q,null),n(Q,jo),n(Q,Pt),n(Q,Jo),g(ye,Q,null),n(Q,Uo),g(Te,Q,null),u(e,Sn,a),g(lt,e,a),u(e,xn,a),u(e,U,a),g(dt,U,null),n(U,Fo),n(U,Yt),n(U,Wo),n(U,At),n(U,Zo),n(U,Dt),n(U,Vo),n(U,H),g(ct,H,null),n(H,Io),n(H,Ot),n(H,No),g(ke,H,null),n(H,Xo),g(Me,H,null),u(e,Cn,a),g(pt,e,a),u(e,jn,a),u(e,en,a),Jn=!0},p(e,[a]){const E={};a&2&&(E.$$scope={dirty:a,ctx:e}),de.$set(E);const $={};a&2&&($.$$scope={dirty:a,ctx:e}),pe.$set($);const se={};a&2&&(se.$$scope={dirty:a,ctx:e}),ue.$set(se);const ut={};a&2&&(ut.$$scope={dirty:a,ctx:e}),me.$set(ut);const re={};a&2&&(re.$$scope={dirty:a,ctx:e}),he.$set(re);const tn={};a&2&&(tn.$$scope={dirty:a,ctx:e}),fe.$set(tn);const P={};a&2&&(P.$$scope={dirty:a,ctx:e}),ge.$set(P);const ae={};a&2&&(ae.$$scope={dirty:a,ctx:e}),_e.$set(ae);const V={};a&2&&(V.$$scope={dirty:a,ctx:e}),be.$set(V);const ie={};a&2&&(ie.$$scope={dirty:a,ctx:e}),ye.$set(ie);const I={};a&2&&(I.$$scope={dirty:a,ctx:e}),Te.$set(I);const Y={};a&2&&(Y.$$scope={dirty:a,ctx:e}),ke.$set(Y);const N={};a&2&&(N.$$scope={dirty:a,ctx:e}),Me.$set(N)},i(e){Jn||(_(qe.$$.fragment,e),_(Be.$$.fragment,e),_(je.$$.fragment,e),_(Ue.$$.fragment,e),_(We.$$.fragment,e),_(Ze.$$.fragment,e),_(de.$$.fragment,e),_(Ve.$$.fragment,e),_(Ie.$$.fragment,e),_(Ne.$$.fragment,e),_(Xe.$$.fragment,e),_(Re.$$.fragment,e),_(Ge.$$.fragment,e),_(Le.$$.fragment,e),_(Qe.$$.fragment,e),_(He.$$.fragment,e),_(Ee.$$.fragment,e),_(Pe.$$.fragment,e),_(Ye.$$.fragment,e),_(pe.$$.fragment,e),_(Ae.$$.fragment,e),_(De.$$.fragment,e),_(Oe.$$.fragment,e),_(ue.$$.fragment,e),_(me.$$.fragment,e),_(Ke.$$.fragment,e),_(et.$$.fragment,e),_(tt.$$.fragment,e),_(he.$$.fragment,e),_(fe.$$.fragment,e),_(ge.$$.fragment,e),_(nt.$$.fragment,e),_(ot.$$.fragment,e),_(st.$$.fragment,e),_(_e.$$.fragment,e),_(be.$$.fragment,e),_(rt.$$.fragment,e),_(at.$$.fragment,e),_(it.$$.fragment,e),_(ye.$$.fragment,e),_(Te.$$.fragment,e),_(lt.$$.fragment,e),_(dt.$$.fragment,e),_(ct.$$.fragment,e),_(ke.$$.fragment,e),_(Me.$$.fragment,e),_(pt.$$.fragment,e),Jn=!0)},o(e){b(qe.$$.fragment,e),b(Be.$$.fragment,e),b(je.$$.fragment,e),b(Ue.$$.fragment,e),b(We.$$.fragment,e),b(Ze.$$.fragment,e),b(de.$$.fragment,e),b(Ve.$$.fragment,e),b(Ie.$$.fragment,e),b(Ne.$$.fragment,e),b(Xe.$$.fragment,e),b(Re.$$.fragment,e),b(Ge.$$.fragment,e),b(Le.$$.fragment,e),b(Qe.$$.fragment,e),b(He.$$.fragment,e),b(Ee.$$.fragment,e),b(Pe.$$.fragment,e),b(Ye.$$.fragment,e),b(pe.$$.fragment,e),b(Ae.$$.fragment,e),b(De.$$.fragment,e),b(Oe.$$.fragment,e),b(ue.$$.fragment,e),b(me.$$.fragment,e),b(Ke.$$.fragment,e),b(et.$$.fragment,e),b(tt.$$.fragment,e),b(he.$$.fragment,e),b(fe.$$.fragment,e),b(ge.$$.fragment,e),b(nt.$$.fragment,e),b(ot.$$.fragment,e),b(st.$$.fragment,e),b(_e.$$.fragment,e),b(be.$$.fragment,e),b(rt.$$.fragment,e),b(at.$$.fragment,e),b(it.$$.fragment,e),b(ye.$$.fragment,e),b(Te.$$.fragment,e),b(lt.$$.fragment,e),b(dt.$$.fragment,e),b(ct.$$.fragment,e),b(ke.$$.fragment,e),b(Me.$$.fragment,e),b(pt.$$.fragment,e),Jn=!1},d(e){e&&(i(T),i(c),i(p),i(k),i(M),i(nn),i(le),i(on),i(sn),i($e),i(rn),i(Se),i(an),i(xe),i(ln),i(Ce),i(dn),i(cn),i(Je),i(pn),i(un),i(Fe),i(mn),i(hn),i(W),i(fn),i(gn),i(B),i(_n),i(bn),i(Z),i(yn),i(Tn),i(S),i(kn),i(Mn),i(x),i(zn),i(wn),i(C),i(vn),i(qn),i(j),i(Bn),i($n),i(J),i(Sn),i(xn),i(U),i(Cn),i(jn),i(en)),i(t),y(qe,e),y(Be,e),y(je,e),y(Ue,e),y(We,e),y(Ze),y(de),y(Ve,e),y(Ie),y(Ne),y(Xe),y(Re),y(Ge),y(Le,e),y(Qe),y(He),y(Ee,e),y(Pe),y(Ye),y(pe),y(Ae,e),y(De),y(Oe),y(ue),y(me),y(Ke,e),y(et),y(tt),y(he),y(fe),y(ge),y(nt,e),y(ot),y(st),y(_e),y(be),y(rt,e),y(at),y(it),y(ye),y(Te),y(lt,e),y(dt),y(ct),y(ke),y(Me),y(pt,e)}}}const Os='{"title":"SqueezeBERT","local":"squeezebert","sections":[{"title":"Overview","local":"overview","sections":[],"depth":2},{"title":"Usage tips","local":"usage-tips","sections":[],"depth":2},{"title":"Resources","local":"resources","sections":[],"depth":2},{"title":"SqueezeBertConfig","local":"transformers.SqueezeBertConfig","sections":[],"depth":2},{"title":"SqueezeBertTokenizer","local":"transformers.SqueezeBertTokenizer","sections":[],"depth":2},{"title":"SqueezeBertTokenizerFast","local":"transformers.SqueezeBertTokenizerFast","sections":[],"depth":2},{"title":"SqueezeBertModel","local":"transformers.SqueezeBertModel","sections":[],"depth":2},{"title":"SqueezeBertForMaskedLM","local":"transformers.SqueezeBertForMaskedLM","sections":[],"depth":2},{"title":"SqueezeBertForSequenceClassification","local":"transformers.SqueezeBertForSequenceClassification","sections":[],"depth":2},{"title":"SqueezeBertForMultipleChoice","local":"transformers.SqueezeBertForMultipleChoice","sections":[],"depth":2},{"title":"SqueezeBertForTokenClassification","local":"transformers.SqueezeBertForTokenClassification","sections":[],"depth":2},{"title":"SqueezeBertForQuestionAnswering","local":"transformers.SqueezeBertForQuestionAnswering","sections":[],"depth":2}],"depth":1}';function Ks(z){return Js(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class ir extends Us{constructor(t){super(),Fs(this,t,Ks,Ds,js,{})}}export{ir as component};
