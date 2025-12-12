import{s as Do,o as Oo,n as H}from"../chunks/scheduler.18a86fab.js";import{S as Ko,i as es,g as d,s as a,r as f,A as ts,h as c,f as o,c as r,j as $,x as h,u,k as w,y as i,a as l,v as g,d as _,t as b,w as T}from"../chunks/index.98837b22.js";import{T as qt}from"../chunks/Tip.77304350.js";import{D as x}from"../chunks/Docstring.a1ef7999.js";import{C as Ut}from"../chunks/CodeBlock.8d0c2e8a.js";import{E as Jt}from"../chunks/ExampleCodeBlock.8c3ee1f9.js";import{H as V,E as ns}from"../chunks/getInferenceSnippets.06c2775f.js";function os(v){let t,y="Examples:",m,p,M;return p=new Ut({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEVzbU1vZGVsJTJDJTIwRXNtQ29uZmlnJTBBJTBBJTIzJTIwSW5pdGlhbGl6aW5nJTIwYSUyMEVTTSUyMGZhY2Vib29rJTJGZXNtLTFiJTIwc3R5bGUlMjBjb25maWd1cmF0aW9uJTBBY29uZmlndXJhdGlvbiUyMCUzRCUyMEVzbUNvbmZpZyh2b2NhYl9zaXplJTNEMzMpJTBBJTBBJTIzJTIwSW5pdGlhbGl6aW5nJTIwYSUyMG1vZGVsJTIwZnJvbSUyMHRoZSUyMGNvbmZpZ3VyYXRpb24lMEFtb2RlbCUyMCUzRCUyMEVzbU1vZGVsKGNvbmZpZ3VyYXRpb24pJTBBJTBBJTIzJTIwQWNjZXNzaW5nJTIwdGhlJTIwbW9kZWwlMjBjb25maWd1cmF0aW9uJTBBY29uZmlndXJhdGlvbiUyMCUzRCUyMG1vZGVsLmNvbmZpZw==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> EsmModel, EsmConfig

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a ESM facebook/esm-1b style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = EsmConfig(vocab_size=<span class="hljs-number">33</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a model from the configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = EsmModel(configuration)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Accessing the model configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = model.config`,wrap:!1}}),{c(){t=d("p"),t.textContent=y,m=a(),f(p.$$.fragment)},l(s){t=c(s,"P",{"data-svelte-h":!0}),h(t)!=="svelte-kvfsh7"&&(t.textContent=y),m=r(s),u(p.$$.fragment,s)},m(s,k){l(s,t,k),l(s,m,k),g(p,s,k),M=!0},p:H,i(s){M||(_(p.$$.fragment,s),M=!0)},o(s){b(p.$$.fragment,s),M=!1},d(s){s&&(o(t),o(m)),T(p,s)}}}function ss(v){let t,y=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=d("p"),t.innerHTML=y},l(m){t=c(m,"P",{"data-svelte-h":!0}),h(t)!=="svelte-fincs2"&&(t.innerHTML=y)},m(m,p){l(m,t,p)},p:H,d(m){m&&o(t)}}}function as(v){let t,y=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=d("p"),t.innerHTML=y},l(m){t=c(m,"P",{"data-svelte-h":!0}),h(t)!=="svelte-fincs2"&&(t.innerHTML=y)},m(m,p){l(m,t,p)},p:H,d(m){m&&o(t)}}}function rs(v){let t,y="Example:",m,p,M;return p=new Ut({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBFc21Gb3JNYXNrZWRMTSUwQWltcG9ydCUyMHRvcmNoJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIyZmFjZWJvb2slMkZlc20tMWIlMjIpJTBBbW9kZWwlMjAlM0QlMjBFc21Gb3JNYXNrZWRMTS5mcm9tX3ByZXRyYWluZWQoJTIyZmFjZWJvb2slMkZlc20tMWIlMjIpJTBBJTBBaW5wdXRzJTIwJTNEJTIwdG9rZW5pemVyKCUyMlRoZSUyMGNhcGl0YWwlMjBvZiUyMEZyYW5jZSUyMGlzJTIwJTNDbWFzayUzRS4lMjIlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQSUwQXdpdGglMjB0b3JjaC5ub19ncmFkKCklM0ElMEElMjAlMjAlMjAlMjBsb2dpdHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cykubG9naXRzJTBBJTBBJTIzJTIwcmV0cmlldmUlMjBpbmRleCUyMG9mJTIwJTNDbWFzayUzRSUwQW1hc2tfdG9rZW5faW5kZXglMjAlM0QlMjAoaW5wdXRzLmlucHV0X2lkcyUyMCUzRCUzRCUyMHRva2VuaXplci5tYXNrX3Rva2VuX2lkKSU1QjAlNUQubm9uemVybyhhc190dXBsZSUzRFRydWUpJTVCMCU1RCUwQSUwQXByZWRpY3RlZF90b2tlbl9pZCUyMCUzRCUyMGxvZ2l0cyU1QjAlMkMlMjBtYXNrX3Rva2VuX2luZGV4JTVELmFyZ21heChheGlzJTNELTEpJTBBdG9rZW5pemVyLmRlY29kZShwcmVkaWN0ZWRfdG9rZW5faWQpJTBBJTBBbGFiZWxzJTIwJTNEJTIwdG9rZW5pemVyKCUyMlRoZSUyMGNhcGl0YWwlMjBvZiUyMEZyYW5jZSUyMGlzJTIwUGFyaXMuJTIyJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiklNUIlMjJpbnB1dF9pZHMlMjIlNUQlMEElMjMlMjBtYXNrJTIwbGFiZWxzJTIwb2YlMjBub24tJTNDbWFzayUzRSUyMHRva2VucyUwQWxhYmVscyUyMCUzRCUyMHRvcmNoLndoZXJlKGlucHV0cy5pbnB1dF9pZHMlMjAlM0QlM0QlMjB0b2tlbml6ZXIubWFza190b2tlbl9pZCUyQyUyMGxhYmVscyUyQyUyMC0xMDApJTBBJTBBb3V0cHV0cyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzJTJDJTIwbGFiZWxzJTNEbGFiZWxzKSUwQXJvdW5kKG91dHB1dHMubG9zcy5pdGVtKCklMkMlMjAyKQ==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, EsmForMaskedLM
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;facebook/esm-1b&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = EsmForMaskedLM.from_pretrained(<span class="hljs-string">&quot;facebook/esm-1b&quot;</span>)

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
...`,wrap:!1}}),{c(){t=d("p"),t.textContent=y,m=a(),f(p.$$.fragment)},l(s){t=c(s,"P",{"data-svelte-h":!0}),h(t)!=="svelte-11lpom8"&&(t.textContent=y),m=r(s),u(p.$$.fragment,s)},m(s,k){l(s,t,k),l(s,m,k),g(p,s,k),M=!0},p:H,i(s){M||(_(p.$$.fragment,s),M=!0)},o(s){b(p.$$.fragment,s),M=!1},d(s){s&&(o(t),o(m)),T(p,s)}}}function is(v){let t,y=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=d("p"),t.innerHTML=y},l(m){t=c(m,"P",{"data-svelte-h":!0}),h(t)!=="svelte-fincs2"&&(t.innerHTML=y)},m(m,p){l(m,t,p)},p:H,d(m){m&&o(t)}}}function ls(v){let t,y="Example of single-label classification:",m,p,M;return p=new Ut({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Rva2VuaXplciUyQyUyMEVzbUZvclNlcXVlbmNlQ2xhc3NpZmljYXRpb24lMEElMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJmYWNlYm9vayUyRmVzbS0xYiUyMiklMEFtb2RlbCUyMCUzRCUyMEVzbUZvclNlcXVlbmNlQ2xhc3NpZmljYXRpb24uZnJvbV9wcmV0cmFpbmVkKCUyMmZhY2Vib29rJTJGZXNtLTFiJTIyKSUwQSUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplciglMjJIZWxsbyUyQyUyMG15JTIwZG9nJTIwaXMlMjBjdXRlJTIyJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiklMEElMEF3aXRoJTIwdG9yY2gubm9fZ3JhZCgpJTNBJTBBJTIwJTIwJTIwJTIwbG9naXRzJTIwJTNEJTIwbW9kZWwoKippbnB1dHMpLmxvZ2l0cyUwQSUwQXByZWRpY3RlZF9jbGFzc19pZCUyMCUzRCUyMGxvZ2l0cy5hcmdtYXgoKS5pdGVtKCklMEFtb2RlbC5jb25maWcuaWQybGFiZWwlNUJwcmVkaWN0ZWRfY2xhc3NfaWQlNUQlMEElMEElMjMlMjBUbyUyMHRyYWluJTIwYSUyMG1vZGVsJTIwb24lMjAlNjBudW1fbGFiZWxzJTYwJTIwY2xhc3NlcyUyQyUyMHlvdSUyMGNhbiUyMHBhc3MlMjAlNjBudW1fbGFiZWxzJTNEbnVtX2xhYmVscyU2MCUyMHRvJTIwJTYwLmZyb21fcHJldHJhaW5lZCguLi4pJTYwJTBBbnVtX2xhYmVscyUyMCUzRCUyMGxlbihtb2RlbC5jb25maWcuaWQybGFiZWwpJTBBbW9kZWwlMjAlM0QlMjBFc21Gb3JTZXF1ZW5jZUNsYXNzaWZpY2F0aW9uLmZyb21fcHJldHJhaW5lZCglMjJmYWNlYm9vayUyRmVzbS0xYiUyMiUyQyUyMG51bV9sYWJlbHMlM0RudW1fbGFiZWxzKSUwQSUwQWxhYmVscyUyMCUzRCUyMHRvcmNoLnRlbnNvciglNUIxJTVEKSUwQWxvc3MlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyUyQyUyMGxhYmVscyUzRGxhYmVscykubG9zcyUwQXJvdW5kKGxvc3MuaXRlbSgpJTJDJTIwMik=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, EsmForSequenceClassification

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;facebook/esm-1b&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = EsmForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;facebook/esm-1b&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    logits = model(**inputs).logits

<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_class_id = logits.argmax().item()
<span class="hljs-meta">&gt;&gt;&gt; </span>model.config.id2label[predicted_class_id]
...

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># To train a model on \`num_labels\` classes, you can pass \`num_labels=num_labels\` to \`.from_pretrained(...)\`</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>num_labels = <span class="hljs-built_in">len</span>(model.config.id2label)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = EsmForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;facebook/esm-1b&quot;</span>, num_labels=num_labels)

<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.tensor([<span class="hljs-number">1</span>])
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = model(**inputs, labels=labels).loss
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">round</span>(loss.item(), <span class="hljs-number">2</span>)
...`,wrap:!1}}),{c(){t=d("p"),t.textContent=y,m=a(),f(p.$$.fragment)},l(s){t=c(s,"P",{"data-svelte-h":!0}),h(t)!=="svelte-ykxpe4"&&(t.textContent=y),m=r(s),u(p.$$.fragment,s)},m(s,k){l(s,t,k),l(s,m,k),g(p,s,k),M=!0},p:H,i(s){M||(_(p.$$.fragment,s),M=!0)},o(s){b(p.$$.fragment,s),M=!1},d(s){s&&(o(t),o(m)),T(p,s)}}}function ds(v){let t,y="Example of multi-label classification:",m,p,M;return p=new Ut({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Rva2VuaXplciUyQyUyMEVzbUZvclNlcXVlbmNlQ2xhc3NpZmljYXRpb24lMEElMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJmYWNlYm9vayUyRmVzbS0xYiUyMiklMEFtb2RlbCUyMCUzRCUyMEVzbUZvclNlcXVlbmNlQ2xhc3NpZmljYXRpb24uZnJvbV9wcmV0cmFpbmVkKCUyMmZhY2Vib29rJTJGZXNtLTFiJTIyJTJDJTIwcHJvYmxlbV90eXBlJTNEJTIybXVsdGlfbGFiZWxfY2xhc3NpZmljYXRpb24lMjIpJTBBJTBBaW5wdXRzJTIwJTNEJTIwdG9rZW5pemVyKCUyMkhlbGxvJTJDJTIwbXklMjBkb2clMjBpcyUyMGN1dGUlMjIlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQSUwQXdpdGglMjB0b3JjaC5ub19ncmFkKCklM0ElMEElMjAlMjAlMjAlMjBsb2dpdHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cykubG9naXRzJTBBJTBBcHJlZGljdGVkX2NsYXNzX2lkcyUyMCUzRCUyMHRvcmNoLmFyYW5nZSgwJTJDJTIwbG9naXRzLnNoYXBlJTVCLTElNUQpJTVCdG9yY2guc2lnbW9pZChsb2dpdHMpLnNxdWVlemUoZGltJTNEMCklMjAlM0UlMjAwLjUlNUQlMEElMEElMjMlMjBUbyUyMHRyYWluJTIwYSUyMG1vZGVsJTIwb24lMjAlNjBudW1fbGFiZWxzJTYwJTIwY2xhc3NlcyUyQyUyMHlvdSUyMGNhbiUyMHBhc3MlMjAlNjBudW1fbGFiZWxzJTNEbnVtX2xhYmVscyU2MCUyMHRvJTIwJTYwLmZyb21fcHJldHJhaW5lZCguLi4pJTYwJTBBbnVtX2xhYmVscyUyMCUzRCUyMGxlbihtb2RlbC5jb25maWcuaWQybGFiZWwpJTBBbW9kZWwlMjAlM0QlMjBFc21Gb3JTZXF1ZW5jZUNsYXNzaWZpY2F0aW9uLmZyb21fcHJldHJhaW5lZCglMEElMjAlMjAlMjAlMjAlMjJmYWNlYm9vayUyRmVzbS0xYiUyMiUyQyUyMG51bV9sYWJlbHMlM0RudW1fbGFiZWxzJTJDJTIwcHJvYmxlbV90eXBlJTNEJTIybXVsdGlfbGFiZWxfY2xhc3NpZmljYXRpb24lMjIlMEEpJTBBJTBBbGFiZWxzJTIwJTNEJTIwdG9yY2guc3VtKCUwQSUyMCUyMCUyMCUyMHRvcmNoLm5uLmZ1bmN0aW9uYWwub25lX2hvdChwcmVkaWN0ZWRfY2xhc3NfaWRzJTVCTm9uZSUyQyUyMCUzQSU1RC5jbG9uZSgpJTJDJTIwbnVtX2NsYXNzZXMlM0RudW1fbGFiZWxzKSUyQyUyMGRpbSUzRDElMEEpLnRvKHRvcmNoLmZsb2F0KSUwQWxvc3MlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyUyQyUyMGxhYmVscyUzRGxhYmVscykubG9zcw==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, EsmForSequenceClassification

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;facebook/esm-1b&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = EsmForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;facebook/esm-1b&quot;</span>, problem_type=<span class="hljs-string">&quot;multi_label_classification&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    logits = model(**inputs).logits

<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_class_ids = torch.arange(<span class="hljs-number">0</span>, logits.shape[-<span class="hljs-number">1</span>])[torch.sigmoid(logits).squeeze(dim=<span class="hljs-number">0</span>) &gt; <span class="hljs-number">0.5</span>]

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># To train a model on \`num_labels\` classes, you can pass \`num_labels=num_labels\` to \`.from_pretrained(...)\`</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>num_labels = <span class="hljs-built_in">len</span>(model.config.id2label)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = EsmForSequenceClassification.from_pretrained(
<span class="hljs-meta">... </span>    <span class="hljs-string">&quot;facebook/esm-1b&quot;</span>, num_labels=num_labels, problem_type=<span class="hljs-string">&quot;multi_label_classification&quot;</span>
<span class="hljs-meta">... </span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.<span class="hljs-built_in">sum</span>(
<span class="hljs-meta">... </span>    torch.nn.functional.one_hot(predicted_class_ids[<span class="hljs-literal">None</span>, :].clone(), num_classes=num_labels), dim=<span class="hljs-number">1</span>
<span class="hljs-meta">... </span>).to(torch.<span class="hljs-built_in">float</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = model(**inputs, labels=labels).loss`,wrap:!1}}),{c(){t=d("p"),t.textContent=y,m=a(),f(p.$$.fragment)},l(s){t=c(s,"P",{"data-svelte-h":!0}),h(t)!=="svelte-1l8e32d"&&(t.textContent=y),m=r(s),u(p.$$.fragment,s)},m(s,k){l(s,t,k),l(s,m,k),g(p,s,k),M=!0},p:H,i(s){M||(_(p.$$.fragment,s),M=!0)},o(s){b(p.$$.fragment,s),M=!1},d(s){s&&(o(t),o(m)),T(p,s)}}}function cs(v){let t,y=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=d("p"),t.innerHTML=y},l(m){t=c(m,"P",{"data-svelte-h":!0}),h(t)!=="svelte-fincs2"&&(t.innerHTML=y)},m(m,p){l(m,t,p)},p:H,d(m){m&&o(t)}}}function ms(v){let t,y="Example:",m,p,M;return p=new Ut({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBFc21Gb3JUb2tlbkNsYXNzaWZpY2F0aW9uJTBBaW1wb3J0JTIwdG9yY2glMEElMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJmYWNlYm9vayUyRmVzbS0xYiUyMiklMEFtb2RlbCUyMCUzRCUyMEVzbUZvclRva2VuQ2xhc3NpZmljYXRpb24uZnJvbV9wcmV0cmFpbmVkKCUyMmZhY2Vib29rJTJGZXNtLTFiJTIyKSUwQSUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplciglMEElMjAlMjAlMjAlMjAlMjJIdWdnaW5nRmFjZSUyMGlzJTIwYSUyMGNvbXBhbnklMjBiYXNlZCUyMGluJTIwUGFyaXMlMjBhbmQlMjBOZXclMjBZb3JrJTIyJTJDJTIwYWRkX3NwZWNpYWxfdG9rZW5zJTNERmFsc2UlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyJTBBKSUwQSUwQXdpdGglMjB0b3JjaC5ub19ncmFkKCklM0ElMEElMjAlMjAlMjAlMjBsb2dpdHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cykubG9naXRzJTBBJTBBcHJlZGljdGVkX3Rva2VuX2NsYXNzX2lkcyUyMCUzRCUyMGxvZ2l0cy5hcmdtYXgoLTEpJTBBJTBBJTIzJTIwTm90ZSUyMHRoYXQlMjB0b2tlbnMlMjBhcmUlMjBjbGFzc2lmaWVkJTIwcmF0aGVyJTIwdGhlbiUyMGlucHV0JTIwd29yZHMlMjB3aGljaCUyMG1lYW5zJTIwdGhhdCUwQSUyMyUyMHRoZXJlJTIwbWlnaHQlMjBiZSUyMG1vcmUlMjBwcmVkaWN0ZWQlMjB0b2tlbiUyMGNsYXNzZXMlMjB0aGFuJTIwd29yZHMuJTBBJTIzJTIwTXVsdGlwbGUlMjB0b2tlbiUyMGNsYXNzZXMlMjBtaWdodCUyMGFjY291bnQlMjBmb3IlMjB0aGUlMjBzYW1lJTIwd29yZCUwQXByZWRpY3RlZF90b2tlbnNfY2xhc3NlcyUyMCUzRCUyMCU1Qm1vZGVsLmNvbmZpZy5pZDJsYWJlbCU1QnQuaXRlbSgpJTVEJTIwZm9yJTIwdCUyMGluJTIwcHJlZGljdGVkX3Rva2VuX2NsYXNzX2lkcyU1QjAlNUQlNUQlMEFwcmVkaWN0ZWRfdG9rZW5zX2NsYXNzZXMlMEElMEFsYWJlbHMlMjAlM0QlMjBwcmVkaWN0ZWRfdG9rZW5fY2xhc3NfaWRzJTBBbG9zcyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzJTJDJTIwbGFiZWxzJTNEbGFiZWxzKS5sb3NzJTBBcm91bmQobG9zcy5pdGVtKCklMkMlMjAyKQ==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, EsmForTokenClassification
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;facebook/esm-1b&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = EsmForTokenClassification.from_pretrained(<span class="hljs-string">&quot;facebook/esm-1b&quot;</span>)

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
...`,wrap:!1}}),{c(){t=d("p"),t.textContent=y,m=a(),f(p.$$.fragment)},l(s){t=c(s,"P",{"data-svelte-h":!0}),h(t)!=="svelte-11lpom8"&&(t.textContent=y),m=r(s),u(p.$$.fragment,s)},m(s,k){l(s,t,k),l(s,m,k),g(p,s,k),M=!0},p:H,i(s){M||(_(p.$$.fragment,s),M=!0)},o(s){b(p.$$.fragment,s),M=!1},d(s){s&&(o(t),o(m)),T(p,s)}}}function ps(v){let t,y=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=d("p"),t.innerHTML=y},l(m){t=c(m,"P",{"data-svelte-h":!0}),h(t)!=="svelte-fincs2"&&(t.innerHTML=y)},m(m,p){l(m,t,p)},p:H,d(m){m&&o(t)}}}function hs(v){let t,y="Example:",m,p,M;return p=new Ut({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBFc21Gb3JQcm90ZWluRm9sZGluZyUwQSUwQW1vZGVsJTIwJTNEJTIwRXNtRm9yUHJvdGVpbkZvbGRpbmcuZnJvbV9wcmV0cmFpbmVkKCUyMmZhY2Vib29rJTJGZXNtZm9sZF92MSUyMiklMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJmYWNlYm9vayUyRmVzbWZvbGRfdjElMjIpJTBBaW5wdXRzJTIwJTNEJTIwdG9rZW5pemVyKCU1QiUyMk1MS05WUVZRTFYlMjIlNUQlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyJTJDJTIwYWRkX3NwZWNpYWxfdG9rZW5zJTNERmFsc2UpJTIwJTIwJTIzJTIwQSUyMHRpbnklMjByYW5kb20lMjBwZXB0aWRlJTBBb3V0cHV0cyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzKSUwQWZvbGRlZF9wb3NpdGlvbnMlMjAlM0QlMjBvdXRwdXRzLnBvc2l0aW9ucw==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, EsmForProteinFolding

<span class="hljs-meta">&gt;&gt;&gt; </span>model = EsmForProteinFolding.from_pretrained(<span class="hljs-string">&quot;facebook/esmfold_v1&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;facebook/esmfold_v1&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer([<span class="hljs-string">&quot;MLKNVQVQLV&quot;</span>], return_tensors=<span class="hljs-string">&quot;pt&quot;</span>, add_special_tokens=<span class="hljs-literal">False</span>)  <span class="hljs-comment"># A tiny random peptide</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs)
<span class="hljs-meta">&gt;&gt;&gt; </span>folded_positions = outputs.positions`,wrap:!1}}),{c(){t=d("p"),t.textContent=y,m=a(),f(p.$$.fragment)},l(s){t=c(s,"P",{"data-svelte-h":!0}),h(t)!=="svelte-11lpom8"&&(t.textContent=y),m=r(s),u(p.$$.fragment,s)},m(s,k){l(s,t,k),l(s,m,k),g(p,s,k),M=!0},p:H,i(s){M||(_(p.$$.fragment,s),M=!0)},o(s){b(p.$$.fragment,s),M=!1},d(s){s&&(o(t),o(m)),T(p,s)}}}function fs(v){let t,y,m,p,M,s="<em>This model was released on 2019-04-19 and added to Hugging Face Transformers on 2022-09-30.</em>",k,ue,It,K,po='<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/>',Lt,ge,Bt,_e,ho=`This page provides code and pre-trained weights for Transformer protein language models from Meta AI’s Fundamental
AI Research Team, providing the state-of-the-art ESMFold and ESM-2, and the previously released ESM-1b and ESM-1v.
Transformer protein language models were introduced in the paper <a href="https://www.pnas.org/content/118/15/e2016239118" rel="nofollow">Biological structure and function emerge from scaling
unsupervised learning to 250 million protein sequences</a> by
Alexander Rives, Joshua Meier, Tom Sercu, Siddharth Goyal, Zeming Lin, Jason Liu, Demi Guo, Myle Ott,
C. Lawrence Zitnick, Jerry Ma, and Rob Fergus.
The first version of this paper was <a href="https://www.biorxiv.org/content/10.1101/622803v1?versioned=true" rel="nofollow">preprinted in 2019</a>.`,Rt,be,fo=`ESM-2 outperforms all tested single-sequence protein language models across a range of structure prediction tasks,
and enables atomic resolution structure prediction.
It was released with the paper <a href="https://doi.org/10.1101/2022.07.20.500902" rel="nofollow">Language models of protein sequences at the scale of evolution enable accurate
structure prediction</a> by Zeming Lin, Halil Akin, Roshan Rao, Brian Hie,
Zhongkai Zhu, Wenting Lu, Allan dos Santos Costa, Maryam Fazel-Zarandi, Tom Sercu, Sal Candido and Alexander Rives.`,Gt,Te,uo=`Also introduced in this paper was ESMFold. It uses an ESM-2 stem with a head that can predict folded protein
structures with state-of-the-art accuracy. Unlike <a href="https://www.nature.com/articles/s41586-021-03819-2" rel="nofollow">AlphaFold2</a>,
it relies on the token embeddings from the large pre-trained protein language model stem and does not perform a multiple
sequence alignment (MSA) step at inference time, which means that ESMFold checkpoints are fully “standalone” -
they do not require a database of known protein sequences and structures with associated external query tools
to make predictions, and are much faster as a result.`,Vt,ye,go=`The abstract from
“Biological structure and function emerge from scaling unsupervised learning to 250
million protein sequences” is`,Ht,Me,_o=`<em>In the field of artificial intelligence, a combination of scale in data and model capacity enabled by unsupervised
learning has led to major advances in representation learning and statistical generation. In the life sciences, the
anticipated growth of sequencing promises unprecedented data on natural sequence diversity. Protein language modeling
at the scale of evolution is a logical step toward predictive and generative artificial intelligence for biology. To
this end, we use unsupervised learning to train a deep contextual language model on 86 billion amino acids across 250
million protein sequences spanning evolutionary diversity. The resulting model contains information about biological
properties in its representations. The representations are learned from sequence data alone. The learned representation
space has a multiscale organization reflecting structure from the level of biochemical properties of amino acids to
remote homology of proteins. Information about secondary and tertiary structure is encoded in the representations and
can be identified by linear projections. Representation learning produces features that generalize across a range of
applications, enabling state-of-the-art supervised prediction of mutational effect and secondary structure and
improving state-of-the-art features for long-range contact prediction.</em>`,St,ke,bo=`The abstract from
“Language models of protein sequences at the scale of evolution enable accurate structure prediction” is`,Pt,ve,To=`<em>Large language models have recently been shown to develop emergent capabilities with scale, going beyond
simple pattern matching to perform higher level reasoning and generate lifelike images and text. While
language models trained on protein sequences have been studied at a smaller scale, little is known about
what they learn about biology as they are scaled up. In this work we train models up to 15 billion parameters,
the largest language models of proteins to be evaluated to date. We find that as models are scaled they learn
information enabling the prediction of the three-dimensional structure of a protein at the resolution of
individual atoms. We present ESMFold for high accuracy end-to-end atomic level structure prediction directly
from the individual sequence of a protein. ESMFold has similar accuracy to AlphaFold2 and RoseTTAFold for
sequences with low perplexity that are well understood by the language model. ESMFold inference is an
order of magnitude faster than AlphaFold2, enabling exploration of the structural space of metagenomic
proteins in practical timescales.</em>`,Xt,we,yo=`The original code can be found <a href="https://github.com/facebookresearch/esm" rel="nofollow">here</a> and was
was developed by the Fundamental AI Research team at Meta AI.
ESM-1b, ESM-1v and ESM-2 were contributed to huggingface by <a href="https://huggingface.co/jasonliu" rel="nofollow">jasonliu</a>
and <a href="https://huggingface.co/Rocketknight1" rel="nofollow">Matt</a>.`,Yt,$e,Mo=`ESMFold was contributed to huggingface by <a href="https://huggingface.co/Rocketknight1" rel="nofollow">Matt</a> and
<a href="https://huggingface.co/sgugger" rel="nofollow">Sylvain</a>, with a big thank you to Nikita Smetanin, Roshan Rao and Tom Sercu for their
help throughout the process!`,At,je,Qt,xe,ko='<li>ESM models are trained with a masked language modeling (MLM) objective.</li> <li>The HuggingFace port of ESMFold uses portions of the <a href="https://github.com/aqlaboratory/openfold" rel="nofollow">openfold</a> library. The <code>openfold</code> library is licensed under the Apache License 2.0.</li>',Dt,Ce,Ot,Ee,vo='<li><a href="../tasks/sequence_classification">Text classification task guide</a></li> <li><a href="../tasks/token_classification">Token classification task guide</a></li> <li><a href="../tasks/masked_language_modeling">Masked language modeling task guide</a></li>',Kt,Fe,en,C,ze,_n,ot,wo=`This is the configuration class to store the configuration of a <code>ESMModel</code>. It is used to instantiate a ESM model
according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the ESM
<a href="https://huggingface.co/facebook/esm-1b" rel="nofollow">facebook/esm-1b</a> architecture.`,bn,st,$o=`Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> and can be used to control the model outputs. Read the
documentation from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> for more information.`,Tn,ee,yn,te,Je,Mn,at,jo='Serializes this instance to a Python dictionary. Override the default <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig.to_dict">to_dict()</a>.',tn,Ue,nn,j,Ze,kn,rt,xo="Constructs an ESM tokenizer.",vn,it,We,wn,ne,Ne,$n,lt,Co=`Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
special tokens using the tokenizer <code>prepare_for_model</code> or <code>encode_plus</code> methods.`,jn,S,qe,xn,dt,Eo=`Create the token type IDs corresponding to the sequences passed. <a href="../glossary#token-type-ids">What are token type
IDs?</a>`,Cn,ct,Fo="Should be overridden in a subclass if the model has a special way of building those.",En,mt,Ie,on,Le,sn,E,Be,Fn,pt,zo="The bare Esm Model outputting raw hidden-states without any specific head on top.",zn,ht,Jo=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Jn,ft,Uo=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Un,P,Re,Zn,ut,Zo='The <a href="/docs/transformers/v4.56.2/en/model_doc/esm#transformers.EsmModel">EsmModel</a> forward method, overrides the <code>__call__</code> special method.',Wn,oe,an,Ge,rn,F,Ve,Nn,gt,Wo="The Esm Model with a <code>language modeling</code> head on top.”",qn,_t,No=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,In,bt,qo=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Ln,L,He,Bn,Tt,Io='The <a href="/docs/transformers/v4.56.2/en/model_doc/esm#transformers.EsmForMaskedLM">EsmForMaskedLM</a> forward method, overrides the <code>__call__</code> special method.',Rn,se,Gn,ae,ln,Se,dn,z,Pe,Vn,yt,Lo=`ESM Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled
output) e.g. for GLUE tasks.`,Hn,Mt,Bo=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Sn,kt,Ro=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Pn,W,Xe,Xn,vt,Go='The <a href="/docs/transformers/v4.56.2/en/model_doc/esm#transformers.EsmForSequenceClassification">EsmForSequenceClassification</a> forward method, overrides the <code>__call__</code> special method.',Yn,re,An,ie,Qn,le,cn,Ye,mn,J,Ae,Dn,wt,Vo=`The Esm transformer with a token classification head on top (a linear layer on top of the hidden-states
output) e.g. for Named-Entity-Recognition (NER) tasks.`,On,$t,Ho=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Kn,jt,So=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,eo,B,Qe,to,xt,Po='The <a href="/docs/transformers/v4.56.2/en/model_doc/esm#transformers.EsmForTokenClassification">EsmForTokenClassification</a> forward method, overrides the <code>__call__</code> special method.',no,de,oo,ce,pn,De,hn,U,Oe,so,Ct,Xo=`ESMForProteinFolding is the HuggingFace port of the original ESMFold model. It consists of an ESM-2 “stem” followed
by a protein folding “head”, although unlike most other output heads, this “head” is similar in size and runtime to
the rest of the model combined! It outputs a dictionary containing predicted structural information about the input
protein(s).`,ao,Et,Yo=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,ro,Ft,Ao=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,io,R,Ke,lo,zt,Qo='The <a href="/docs/transformers/v4.56.2/en/model_doc/esm#transformers.EsmForProteinFolding">EsmForProteinFolding</a> forward method, overrides the <code>__call__</code> special method.',co,me,mo,pe,fn,et,un,Zt,gn;return ue=new V({props:{title:"ESM",local:"esm",headingTag:"h1"}}),ge=new V({props:{title:"Overview",local:"overview",headingTag:"h2"}}),je=new V({props:{title:"Usage tips",local:"usage-tips",headingTag:"h2"}}),Ce=new V({props:{title:"Resources",local:"resources",headingTag:"h2"}}),Fe=new V({props:{title:"EsmConfig",local:"transformers.EsmConfig",headingTag:"h2"}}),ze=new x({props:{name:"class transformers.EsmConfig",anchor:"transformers.EsmConfig",parameters:[{name:"vocab_size",val:" = None"},{name:"mask_token_id",val:" = None"},{name:"pad_token_id",val:" = None"},{name:"hidden_size",val:" = 768"},{name:"num_hidden_layers",val:" = 12"},{name:"num_attention_heads",val:" = 12"},{name:"intermediate_size",val:" = 3072"},{name:"hidden_dropout_prob",val:" = 0.1"},{name:"attention_probs_dropout_prob",val:" = 0.1"},{name:"max_position_embeddings",val:" = 1026"},{name:"initializer_range",val:" = 0.02"},{name:"layer_norm_eps",val:" = 1e-12"},{name:"position_embedding_type",val:" = 'absolute'"},{name:"use_cache",val:" = True"},{name:"emb_layer_norm_before",val:" = None"},{name:"token_dropout",val:" = False"},{name:"is_folding_model",val:" = False"},{name:"esmfold_config",val:" = None"},{name:"vocab_list",val:" = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.EsmConfig.vocab_size",description:`<strong>vocab_size</strong> (<code>int</code>, <em>optional</em>) &#x2014;
Vocabulary size of the ESM model. Defines the number of different tokens that can be represented by the
<code>inputs_ids</code> passed when calling <code>ESMModel</code>.`,name:"vocab_size"},{anchor:"transformers.EsmConfig.mask_token_id",description:`<strong>mask_token_id</strong> (<code>int</code>, <em>optional</em>) &#x2014;
The index of the mask token in the vocabulary. This must be included in the config because of the
&#x201C;mask-dropout&#x201D; scaling trick, which will scale the inputs depending on the number of masked tokens.`,name:"mask_token_id"},{anchor:"transformers.EsmConfig.pad_token_id",description:`<strong>pad_token_id</strong> (<code>int</code>, <em>optional</em>) &#x2014;
The index of the padding token in the vocabulary. This must be included in the config because certain parts
of the ESM code use this instead of the attention mask.`,name:"pad_token_id"},{anchor:"transformers.EsmConfig.hidden_size",description:`<strong>hidden_size</strong> (<code>int</code>, <em>optional</em>, defaults to 768) &#x2014;
Dimensionality of the encoder layers and the pooler layer.`,name:"hidden_size"},{anchor:"transformers.EsmConfig.num_hidden_layers",description:`<strong>num_hidden_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 12) &#x2014;
Number of hidden layers in the Transformer encoder.`,name:"num_hidden_layers"},{anchor:"transformers.EsmConfig.num_attention_heads",description:`<strong>num_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 12) &#x2014;
Number of attention heads for each attention layer in the Transformer encoder.`,name:"num_attention_heads"},{anchor:"transformers.EsmConfig.intermediate_size",description:`<strong>intermediate_size</strong> (<code>int</code>, <em>optional</em>, defaults to 3072) &#x2014;
Dimensionality of the &#x201C;intermediate&#x201D; (often named feed-forward) layer in the Transformer encoder.`,name:"intermediate_size"},{anchor:"transformers.EsmConfig.hidden_dropout_prob",description:`<strong>hidden_dropout_prob</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.`,name:"hidden_dropout_prob"},{anchor:"transformers.EsmConfig.attention_probs_dropout_prob",description:`<strong>attention_probs_dropout_prob</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout ratio for the attention probabilities.`,name:"attention_probs_dropout_prob"},{anchor:"transformers.EsmConfig.max_position_embeddings",description:`<strong>max_position_embeddings</strong> (<code>int</code>, <em>optional</em>, defaults to 1026) &#x2014;
The maximum sequence length that this model might ever be used with. Typically set this to something large
just in case (e.g., 512 or 1024 or 2048).`,name:"max_position_embeddings"},{anchor:"transformers.EsmConfig.initializer_range",description:`<strong>initializer_range</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.`,name:"initializer_range"},{anchor:"transformers.EsmConfig.layer_norm_eps",description:`<strong>layer_norm_eps</strong> (<code>float</code>, <em>optional</em>, defaults to 1e-12) &#x2014;
The epsilon used by the layer normalization layers.`,name:"layer_norm_eps"},{anchor:"transformers.EsmConfig.position_embedding_type",description:`<strong>position_embedding_type</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;absolute&quot;</code>) &#x2014;
Type of position embedding. Choose one of <code>&quot;absolute&quot;</code>, <code>&quot;relative_key&quot;</code>, <code>&quot;relative_key_query&quot;, &quot;rotary&quot;</code>.
For positional embeddings use <code>&quot;absolute&quot;</code>. For more information on <code>&quot;relative_key&quot;</code>, please refer to
<a href="https://huggingface.co/papers/1803.02155" rel="nofollow">Self-Attention with Relative Position Representations (Shaw et al.)</a>.
For more information on <code>&quot;relative_key_query&quot;</code>, please refer to <em>Method 4</em> in <a href="https://huggingface.co/papers/2009.13658" rel="nofollow">Improve Transformer Models
with Better Relative Position Embeddings (Huang et al.)</a>.`,name:"position_embedding_type"},{anchor:"transformers.EsmConfig.is_decoder",description:`<strong>is_decoder</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether the model is used as a decoder or not. If <code>False</code>, the model is used as an encoder.`,name:"is_decoder"},{anchor:"transformers.EsmConfig.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not the model should return the last key/values attentions (not used by all models). Only
relevant if <code>config.is_decoder=True</code>.`,name:"use_cache"},{anchor:"transformers.EsmConfig.emb_layer_norm_before",description:`<strong>emb_layer_norm_before</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether to apply layer normalization after embeddings but before the main stem of the network.`,name:"emb_layer_norm_before"},{anchor:"transformers.EsmConfig.token_dropout",description:`<strong>token_dropout</strong> (<code>bool</code>, defaults to <code>False</code>) &#x2014;
When this is enabled, masked tokens are treated as if they had been dropped out by input dropout.`,name:"token_dropout"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/esm/configuration_esm.py#L29"}}),ee=new Jt({props:{anchor:"transformers.EsmConfig.example",$$slots:{default:[os]},$$scope:{ctx:v}}}),Je=new x({props:{name:"to_dict",anchor:"transformers.EsmConfig.to_dict",parameters:[],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/esm/configuration_esm.py#L160",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>Dictionary of all the attributes that make up this configuration instance,</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>dict[str, any]</code></p>
`}}),Ue=new V({props:{title:"EsmTokenizer",local:"transformers.EsmTokenizer",headingTag:"h2"}}),Ze=new x({props:{name:"class transformers.EsmTokenizer",anchor:"transformers.EsmTokenizer",parameters:[{name:"vocab_file",val:""},{name:"unk_token",val:" = '<unk>'"},{name:"cls_token",val:" = '<cls>'"},{name:"pad_token",val:" = '<pad>'"},{name:"mask_token",val:" = '<mask>'"},{name:"eos_token",val:" = '<eos>'"},{name:"**kwargs",val:""}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/esm/tokenization_esm.py#L35"}}),We=new x({props:{name:"build_inputs_with_special_tokens",anchor:"transformers.EsmTokenizer.build_inputs_with_special_tokens",parameters:[{name:"token_ids_0",val:": list"},{name:"token_ids_1",val:": typing.Optional[list[int]] = None"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/esm/tokenization_esm.py#L91"}}),Ne=new x({props:{name:"get_special_tokens_mask",anchor:"transformers.EsmTokenizer.get_special_tokens_mask",parameters:[{name:"token_ids_0",val:": list"},{name:"token_ids_1",val:": typing.Optional[list] = None"},{name:"already_has_special_tokens",val:": bool = False"}],parametersDescription:[{anchor:"transformers.EsmTokenizer.get_special_tokens_mask.token_ids_0",description:`<strong>token_ids_0</strong> (<code>list[int]</code>) &#x2014;
List of ids of the first sequence.`,name:"token_ids_0"},{anchor:"transformers.EsmTokenizer.get_special_tokens_mask.token_ids_1",description:`<strong>token_ids_1</strong> (<code>list[int]</code>, <em>optional</em>) &#x2014;
List of ids of the second sequence.`,name:"token_ids_1"},{anchor:"transformers.EsmTokenizer.get_special_tokens_mask.already_has_special_tokens",description:`<strong>already_has_special_tokens</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not the token list is already formatted with special tokens for the model.`,name:"already_has_special_tokens"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/esm/tokenization_esm.py#L105",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>1 for a special token, 0 for a sequence token.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A list of integers in the range [0, 1]</p>
`}}),qe=new x({props:{name:"create_token_type_ids_from_sequences",anchor:"transformers.EsmTokenizer.create_token_type_ids_from_sequences",parameters:[{name:"token_ids_0",val:": list"},{name:"token_ids_1",val:": typing.Optional[list[int]] = None"}],parametersDescription:[{anchor:"transformers.EsmTokenizer.create_token_type_ids_from_sequences.token_ids_0",description:"<strong>token_ids_0</strong> (<code>list[int]</code>) &#x2014; The first tokenized sequence.",name:"token_ids_0"},{anchor:"transformers.EsmTokenizer.create_token_type_ids_from_sequences.token_ids_1",description:"<strong>token_ids_1</strong> (<code>list[int]</code>, <em>optional</em>) &#x2014; The second tokenized sequence.",name:"token_ids_1"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L3432",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>The token type ids.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>list[int]</code></p>
`}}),Ie=new x({props:{name:"save_vocabulary",anchor:"transformers.EsmTokenizer.save_vocabulary",parameters:[{name:"save_directory",val:""},{name:"filename_prefix",val:""}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/esm/tokenization_esm.py#L136"}}),Le=new V({props:{title:"EsmModel",local:"transformers.EsmModel",headingTag:"h2"}}),Be=new x({props:{name:"class transformers.EsmModel",anchor:"transformers.EsmModel",parameters:[{name:"config",val:""},{name:"add_pooling_layer",val:" = True"}],parametersDescription:[{anchor:"transformers.EsmModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/esm#transformers.EsmModel">EsmModel</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"},{anchor:"transformers.EsmModel.add_pooling_layer",description:`<strong>add_pooling_layer</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether to add a pooling layer`,name:"add_pooling_layer"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/esm/modeling_esm.py#L635"}}),Re=new x({props:{name:"forward",anchor:"transformers.EsmModel.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_hidden_states",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.EsmModel.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>((batch_size, sequence_length))</code>) &#x2014;
Indices of input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.EsmModel.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.EsmModel.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>((batch_size, sequence_length))</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.max_position_embeddings - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.EsmModel.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.EsmModel.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>((batch_size, sequence_length), hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.EsmModel.forward.encoder_hidden_states",description:`<strong>encoder_hidden_states</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
if the model is configured as a decoder.`,name:"encoder_hidden_states"},{anchor:"transformers.EsmModel.forward.encoder_attention_mask",description:`<strong>encoder_attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
the cross-attention if the model is configured as a decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>`,name:"encoder_attention_mask"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/esm/modeling_esm.py#L682",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions"
>transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/esm#transformers.EsmConfig"
>EsmConfig</a>) and inputs.</p>
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
`}}),oe=new qt({props:{$$slots:{default:[ss]},$$scope:{ctx:v}}}),Ge=new V({props:{title:"EsmForMaskedLM",local:"transformers.EsmForMaskedLM",headingTag:"h2"}}),Ve=new x({props:{name:"class transformers.EsmForMaskedLM",anchor:"transformers.EsmForMaskedLM",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.EsmForMaskedLM.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/esm#transformers.EsmForMaskedLM">EsmForMaskedLM</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/esm/modeling_esm.py#L778"}}),He=new x({props:{name:"forward",anchor:"transformers.EsmForMaskedLM.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"encoder_hidden_states",val:": typing.Optional[torch.FloatTensor] = None"},{name:"encoder_attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.EsmForMaskedLM.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.EsmForMaskedLM.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.EsmForMaskedLM.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.EsmForMaskedLM.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.EsmForMaskedLM.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.EsmForMaskedLM.forward.encoder_hidden_states",description:`<strong>encoder_hidden_states</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
if the model is configured as a decoder.`,name:"encoder_hidden_states"},{anchor:"transformers.EsmForMaskedLM.forward.encoder_attention_mask",description:`<strong>encoder_attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
the cross-attention if the model is configured as a decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>`,name:"encoder_attention_mask"},{anchor:"transformers.EsmForMaskedLM.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should be in <code>[-100, 0, ..., config.vocab_size]</code> (see <code>input_ids</code> docstring) Tokens with indices set to <code>-100</code> are ignored (masked), the
loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>`,name:"labels"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/esm/modeling_esm.py#L803",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.MaskedLMOutput"
>transformers.modeling_outputs.MaskedLMOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/esm#transformers.EsmConfig"
>EsmConfig</a>) and inputs.</p>
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
`}}),se=new qt({props:{$$slots:{default:[as]},$$scope:{ctx:v}}}),ae=new Jt({props:{anchor:"transformers.EsmForMaskedLM.forward.example",$$slots:{default:[rs]},$$scope:{ctx:v}}}),Se=new V({props:{title:"EsmForSequenceClassification",local:"transformers.EsmForSequenceClassification",headingTag:"h2"}}),Pe=new x({props:{name:"class transformers.EsmForSequenceClassification",anchor:"transformers.EsmForSequenceClassification",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.EsmForSequenceClassification.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/esm#transformers.EsmForSequenceClassification">EsmForSequenceClassification</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/esm/modeling_esm.py#L882"}}),Xe=new x({props:{name:"forward",anchor:"transformers.EsmForSequenceClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.EsmForSequenceClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.EsmForSequenceClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.EsmForSequenceClassification.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.EsmForSequenceClassification.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.EsmForSequenceClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.EsmForSequenceClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the sequence classification/regression loss. Indices should be in <code>[0, ..., config.num_labels - 1]</code>. If <code>config.num_labels == 1</code> a regression loss is computed (Mean-Square loss), If
<code>config.num_labels &gt; 1</code> a classification loss is computed (Cross-Entropy).`,name:"labels"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/esm/modeling_esm.py#L895",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput"
>transformers.modeling_outputs.SequenceClassifierOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/esm#transformers.EsmConfig"
>EsmConfig</a>) and inputs.</p>
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
`}}),re=new qt({props:{$$slots:{default:[is]},$$scope:{ctx:v}}}),ie=new Jt({props:{anchor:"transformers.EsmForSequenceClassification.forward.example",$$slots:{default:[ls]},$$scope:{ctx:v}}}),le=new Jt({props:{anchor:"transformers.EsmForSequenceClassification.forward.example-2",$$slots:{default:[ds]},$$scope:{ctx:v}}}),Ye=new V({props:{title:"EsmForTokenClassification",local:"transformers.EsmForTokenClassification",headingTag:"h2"}}),Ae=new x({props:{name:"class transformers.EsmForTokenClassification",anchor:"transformers.EsmForTokenClassification",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.EsmForTokenClassification.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/esm#transformers.EsmForTokenClassification">EsmForTokenClassification</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/esm/modeling_esm.py#L959"}}),Qe=new x({props:{name:"forward",anchor:"transformers.EsmForTokenClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.EsmForTokenClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.EsmForTokenClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.EsmForTokenClassification.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.EsmForTokenClassification.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.EsmForTokenClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.EsmForTokenClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the token classification loss. Indices should be in <code>[0, ..., config.num_labels - 1]</code>.`,name:"labels"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/esm/modeling_esm.py#L972",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput"
>transformers.modeling_outputs.TokenClassifierOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/esm#transformers.EsmConfig"
>EsmConfig</a>) and inputs.</p>
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
`}}),de=new qt({props:{$$slots:{default:[cs]},$$scope:{ctx:v}}}),ce=new Jt({props:{anchor:"transformers.EsmForTokenClassification.forward.example",$$slots:{default:[ms]},$$scope:{ctx:v}}}),De=new V({props:{title:"EsmForProteinFolding",local:"transformers.EsmForProteinFolding",headingTag:"h2"}}),Oe=new x({props:{name:"class transformers.EsmForProteinFolding",anchor:"transformers.EsmForProteinFolding",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.EsmForProteinFolding.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/esm#transformers.EsmForProteinFolding">EsmForProteinFolding</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/esm/modeling_esmfold.py#L1991"}}),Ke=new x({props:{name:"forward",anchor:"transformers.EsmForProteinFolding.forward",parameters:[{name:"input_ids",val:": Tensor"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"masking_pattern",val:": typing.Optional[torch.Tensor] = None"},{name:"num_recycles",val:": typing.Optional[int] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = False"}],parametersDescription:[{anchor:"transformers.EsmForProteinFolding.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.EsmForProteinFolding.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.EsmForProteinFolding.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.EsmForProteinFolding.forward.masking_pattern",description:`<strong>masking_pattern</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Locations of tokens to mask during training as a form of regularization. Mask values selected in <code>[0, 1]</code>.`,name:"masking_pattern"},{anchor:"transformers.EsmForProteinFolding.forward.num_recycles",description:`<strong>num_recycles</strong> (<code>int</code>, <em>optional</em>, defaults to <code>None</code>) &#x2014;
Number of times to recycle the input sequence. If <code>None</code>, defaults to <code>config.num_recycles</code>. &#x201C;Recycling&#x201D;
consists of passing the output of the folding trunk back in as input to the trunk. During training, the
number of recycles should vary with each batch, to ensure that the model learns to output valid predictions
after each recycle. During inference, num_recycles should be set to the highest value that the model was
trained with for maximum accuracy. Accordingly, when this value is set to <code>None</code>, config.max_recycles is
used.`,name:"num_recycles"},{anchor:"transformers.EsmForProteinFolding.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/esm/modeling_esmfold.py#L2060",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <code>transformers.models.esm.modeling_esmfold.EsmForProteinFoldingOutput</code> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/esm#transformers.EsmConfig"
>EsmConfig</a>) and inputs.</p>
<ul>
<li><strong>frames</strong> (<code>torch.FloatTensor</code>, <em>optional</em>, defaults to <code>None</code>) — Output frames.</li>
<li><strong>sidechain_frames</strong> (<code>torch.FloatTensor</code>, <em>optional</em>, defaults to <code>None</code>) — Output sidechain frames.</li>
<li><strong>unnormalized_angles</strong> (<code>torch.FloatTensor</code>, <em>optional</em>, defaults to <code>None</code>) — Predicted unnormalized backbone and side chain torsion angles.</li>
<li><strong>angles</strong> (<code>torch.FloatTensor</code>, <em>optional</em>, defaults to <code>None</code>) — Predicted backbone and side chain torsion angles.</li>
<li><strong>positions</strong> (<code>torch.FloatTensor</code>, <em>optional</em>, defaults to <code>None</code>) — Predicted positions of the backbone and side chain atoms.</li>
<li><strong>states</strong> (<code>torch.FloatTensor</code>, <em>optional</em>, defaults to <code>None</code>) — Hidden states from the protein folding trunk.</li>
<li><strong>s_s</strong> (<code>torch.FloatTensor</code>, <em>optional</em>, defaults to <code>None</code>) — Per-residue embeddings derived by concatenating the hidden states of each layer of the ESM-2 LM stem.</li>
<li><strong>s_z</strong> (<code>torch.FloatTensor</code>, <em>optional</em>, defaults to <code>None</code>) — Pairwise residue embeddings.</li>
<li><strong>distogram_logits</strong> (<code>torch.FloatTensor</code>, <em>optional</em>, defaults to <code>None</code>) — Input logits to the distogram used to compute residue distances.</li>
<li><strong>lm_logits</strong> (<code>torch.FloatTensor</code>, <em>optional</em>, defaults to <code>None</code>) — Logits output by the ESM-2 protein language model stem.</li>
<li><strong>aatype</strong> (<code>torch.FloatTensor</code>, <em>optional</em>, defaults to <code>None</code>) — Input amino acids (AlphaFold2 indices).</li>
<li><strong>atom14_atom_exists</strong> (<code>torch.FloatTensor</code>, <em>optional</em>, defaults to <code>None</code>) — Whether each atom exists in the atom14 representation.</li>
<li><strong>residx_atom14_to_atom37</strong> (<code>torch.FloatTensor</code>, <em>optional</em>, defaults to <code>None</code>) — Mapping between atoms in the atom14 and atom37 representations.</li>
<li><strong>residx_atom37_to_atom14</strong> (<code>torch.FloatTensor</code>, <em>optional</em>, defaults to <code>None</code>) — Mapping between atoms in the atom37 and atom14 representations.</li>
<li><strong>atom37_atom_exists</strong> (<code>torch.FloatTensor</code>, <em>optional</em>, defaults to <code>None</code>) — Whether each atom exists in the atom37 representation.</li>
<li><strong>residue_index</strong> (<code>torch.FloatTensor</code>, <em>optional</em>, defaults to <code>None</code>) — The index of each residue in the protein chain. Unless internal padding tokens are used, this will just be
a sequence of integers from 0 to <code>sequence_length</code>.</li>
<li><strong>lddt_head</strong> (<code>torch.FloatTensor</code>, <em>optional</em>, defaults to <code>None</code>) — Raw outputs from the lddt head used to compute plddt.</li>
<li><strong>plddt</strong> (<code>torch.FloatTensor</code>, <em>optional</em>, defaults to <code>None</code>) — Per-residue confidence scores. Regions of low confidence may indicate areas where the model’s prediction is
uncertain, or where the protein structure is disordered.</li>
<li><strong>ptm_logits</strong> (<code>torch.FloatTensor</code>, <em>optional</em>, defaults to <code>None</code>) — Raw logits used for computing ptm.</li>
<li><strong>ptm</strong> (<code>torch.FloatTensor</code>, <em>optional</em>, defaults to <code>None</code>) — TM-score output representing the model’s high-level confidence in the overall structure.</li>
<li><strong>aligned_confidence_probs</strong> (<code>torch.FloatTensor</code>, <em>optional</em>, defaults to <code>None</code>) — Per-residue confidence scores for the aligned structure.</li>
<li><strong>predicted_aligned_error</strong> (<code>torch.FloatTensor</code>, <em>optional</em>, defaults to <code>None</code>) — Predicted error between the model’s prediction and the ground truth.</li>
<li><strong>max_predicted_aligned_error</strong> (<code>torch.FloatTensor</code>, <em>optional</em>, defaults to <code>None</code>) — Per-sample maximum predicted error.</li>
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>transformers.models.esm.modeling_esmfold.EsmForProteinFoldingOutput</code> or <code>tuple(torch.FloatTensor)</code></p>
`}}),me=new qt({props:{$$slots:{default:[ps]},$$scope:{ctx:v}}}),pe=new Jt({props:{anchor:"transformers.EsmForProteinFolding.forward.example",$$slots:{default:[hs]},$$scope:{ctx:v}}}),et=new ns({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/esm.md"}}),{c(){t=d("meta"),y=a(),m=d("p"),p=a(),M=d("p"),M.innerHTML=s,k=a(),f(ue.$$.fragment),It=a(),K=d("div"),K.innerHTML=po,Lt=a(),f(ge.$$.fragment),Bt=a(),_e=d("p"),_e.innerHTML=ho,Rt=a(),be=d("p"),be.innerHTML=fo,Gt=a(),Te=d("p"),Te.innerHTML=uo,Vt=a(),ye=d("p"),ye.textContent=go,Ht=a(),Me=d("p"),Me.innerHTML=_o,St=a(),ke=d("p"),ke.textContent=bo,Pt=a(),ve=d("p"),ve.innerHTML=To,Xt=a(),we=d("p"),we.innerHTML=yo,Yt=a(),$e=d("p"),$e.innerHTML=Mo,At=a(),f(je.$$.fragment),Qt=a(),xe=d("ul"),xe.innerHTML=ko,Dt=a(),f(Ce.$$.fragment),Ot=a(),Ee=d("ul"),Ee.innerHTML=vo,Kt=a(),f(Fe.$$.fragment),en=a(),C=d("div"),f(ze.$$.fragment),_n=a(),ot=d("p"),ot.innerHTML=wo,bn=a(),st=d("p"),st.innerHTML=$o,Tn=a(),f(ee.$$.fragment),yn=a(),te=d("div"),f(Je.$$.fragment),Mn=a(),at=d("p"),at.innerHTML=jo,tn=a(),f(Ue.$$.fragment),nn=a(),j=d("div"),f(Ze.$$.fragment),kn=a(),rt=d("p"),rt.textContent=xo,vn=a(),it=d("div"),f(We.$$.fragment),wn=a(),ne=d("div"),f(Ne.$$.fragment),$n=a(),lt=d("p"),lt.innerHTML=Co,jn=a(),S=d("div"),f(qe.$$.fragment),xn=a(),dt=d("p"),dt.innerHTML=Eo,Cn=a(),ct=d("p"),ct.textContent=Fo,En=a(),mt=d("div"),f(Ie.$$.fragment),on=a(),f(Le.$$.fragment),sn=a(),E=d("div"),f(Be.$$.fragment),Fn=a(),pt=d("p"),pt.textContent=zo,zn=a(),ht=d("p"),ht.innerHTML=Jo,Jn=a(),ft=d("p"),ft.innerHTML=Uo,Un=a(),P=d("div"),f(Re.$$.fragment),Zn=a(),ut=d("p"),ut.innerHTML=Zo,Wn=a(),f(oe.$$.fragment),an=a(),f(Ge.$$.fragment),rn=a(),F=d("div"),f(Ve.$$.fragment),Nn=a(),gt=d("p"),gt.innerHTML=Wo,qn=a(),_t=d("p"),_t.innerHTML=No,In=a(),bt=d("p"),bt.innerHTML=qo,Ln=a(),L=d("div"),f(He.$$.fragment),Bn=a(),Tt=d("p"),Tt.innerHTML=Io,Rn=a(),f(se.$$.fragment),Gn=a(),f(ae.$$.fragment),ln=a(),f(Se.$$.fragment),dn=a(),z=d("div"),f(Pe.$$.fragment),Vn=a(),yt=d("p"),yt.textContent=Lo,Hn=a(),Mt=d("p"),Mt.innerHTML=Bo,Sn=a(),kt=d("p"),kt.innerHTML=Ro,Pn=a(),W=d("div"),f(Xe.$$.fragment),Xn=a(),vt=d("p"),vt.innerHTML=Go,Yn=a(),f(re.$$.fragment),An=a(),f(ie.$$.fragment),Qn=a(),f(le.$$.fragment),cn=a(),f(Ye.$$.fragment),mn=a(),J=d("div"),f(Ae.$$.fragment),Dn=a(),wt=d("p"),wt.textContent=Vo,On=a(),$t=d("p"),$t.innerHTML=Ho,Kn=a(),jt=d("p"),jt.innerHTML=So,eo=a(),B=d("div"),f(Qe.$$.fragment),to=a(),xt=d("p"),xt.innerHTML=Po,no=a(),f(de.$$.fragment),oo=a(),f(ce.$$.fragment),pn=a(),f(De.$$.fragment),hn=a(),U=d("div"),f(Oe.$$.fragment),so=a(),Ct=d("p"),Ct.textContent=Xo,ao=a(),Et=d("p"),Et.innerHTML=Yo,ro=a(),Ft=d("p"),Ft.innerHTML=Ao,io=a(),R=d("div"),f(Ke.$$.fragment),lo=a(),zt=d("p"),zt.innerHTML=Qo,co=a(),f(me.$$.fragment),mo=a(),f(pe.$$.fragment),fn=a(),f(et.$$.fragment),un=a(),Zt=d("p"),this.h()},l(e){const n=ts("svelte-u9bgzb",document.head);t=c(n,"META",{name:!0,content:!0}),n.forEach(o),y=r(e),m=c(e,"P",{}),$(m).forEach(o),p=r(e),M=c(e,"P",{"data-svelte-h":!0}),h(M)!=="svelte-1vfh37g"&&(M.innerHTML=s),k=r(e),u(ue.$$.fragment,e),It=r(e),K=c(e,"DIV",{class:!0,"data-svelte-h":!0}),h(K)!=="svelte-13t8s2t"&&(K.innerHTML=po),Lt=r(e),u(ge.$$.fragment,e),Bt=r(e),_e=c(e,"P",{"data-svelte-h":!0}),h(_e)!=="svelte-1a7iz27"&&(_e.innerHTML=ho),Rt=r(e),be=c(e,"P",{"data-svelte-h":!0}),h(be)!=="svelte-xibyq9"&&(be.innerHTML=fo),Gt=r(e),Te=c(e,"P",{"data-svelte-h":!0}),h(Te)!=="svelte-1ab9wr9"&&(Te.innerHTML=uo),Vt=r(e),ye=c(e,"P",{"data-svelte-h":!0}),h(ye)!=="svelte-ck6x5b"&&(ye.textContent=go),Ht=r(e),Me=c(e,"P",{"data-svelte-h":!0}),h(Me)!=="svelte-5yzf50"&&(Me.innerHTML=_o),St=r(e),ke=c(e,"P",{"data-svelte-h":!0}),h(ke)!=="svelte-1vj6owx"&&(ke.textContent=bo),Pt=r(e),ve=c(e,"P",{"data-svelte-h":!0}),h(ve)!=="svelte-rbs8ft"&&(ve.innerHTML=To),Xt=r(e),we=c(e,"P",{"data-svelte-h":!0}),h(we)!=="svelte-1w2rws6"&&(we.innerHTML=yo),Yt=r(e),$e=c(e,"P",{"data-svelte-h":!0}),h($e)!=="svelte-17utf8f"&&($e.innerHTML=Mo),At=r(e),u(je.$$.fragment,e),Qt=r(e),xe=c(e,"UL",{"data-svelte-h":!0}),h(xe)!=="svelte-1ap38ix"&&(xe.innerHTML=ko),Dt=r(e),u(Ce.$$.fragment,e),Ot=r(e),Ee=c(e,"UL",{"data-svelte-h":!0}),h(Ee)!=="svelte-18pttsb"&&(Ee.innerHTML=vo),Kt=r(e),u(Fe.$$.fragment,e),en=r(e),C=c(e,"DIV",{class:!0});var N=$(C);u(ze.$$.fragment,N),_n=r(N),ot=c(N,"P",{"data-svelte-h":!0}),h(ot)!=="svelte-1n3e10h"&&(ot.innerHTML=wo),bn=r(N),st=c(N,"P",{"data-svelte-h":!0}),h(st)!=="svelte-1ek1ss9"&&(st.innerHTML=$o),Tn=r(N),u(ee.$$.fragment,N),yn=r(N),te=c(N,"DIV",{class:!0});var tt=$(te);u(Je.$$.fragment,tt),Mn=r(tt),at=c(tt,"P",{"data-svelte-h":!0}),h(at)!=="svelte-14z5e6y"&&(at.innerHTML=jo),tt.forEach(o),N.forEach(o),tn=r(e),u(Ue.$$.fragment,e),nn=r(e),j=c(e,"DIV",{class:!0});var Z=$(j);u(Ze.$$.fragment,Z),kn=r(Z),rt=c(Z,"P",{"data-svelte-h":!0}),h(rt)!=="svelte-nlc0vd"&&(rt.textContent=xo),vn=r(Z),it=c(Z,"DIV",{class:!0});var Wt=$(it);u(We.$$.fragment,Wt),Wt.forEach(o),wn=r(Z),ne=c(Z,"DIV",{class:!0});var nt=$(ne);u(Ne.$$.fragment,nt),$n=r(nt),lt=c(nt,"P",{"data-svelte-h":!0}),h(lt)!=="svelte-1wmjg8a"&&(lt.innerHTML=Co),nt.forEach(o),jn=r(Z),S=c(Z,"DIV",{class:!0});var D=$(S);u(qe.$$.fragment,D),xn=r(D),dt=c(D,"P",{"data-svelte-h":!0}),h(dt)!=="svelte-zj1vf1"&&(dt.innerHTML=Eo),Cn=r(D),ct=c(D,"P",{"data-svelte-h":!0}),h(ct)!=="svelte-9vptpw"&&(ct.textContent=Fo),D.forEach(o),En=r(Z),mt=c(Z,"DIV",{class:!0});var Nt=$(mt);u(Ie.$$.fragment,Nt),Nt.forEach(o),Z.forEach(o),on=r(e),u(Le.$$.fragment,e),sn=r(e),E=c(e,"DIV",{class:!0});var q=$(E);u(Be.$$.fragment,q),Fn=r(q),pt=c(q,"P",{"data-svelte-h":!0}),h(pt)!=="svelte-1bb4i9z"&&(pt.textContent=zo),zn=r(q),ht=c(q,"P",{"data-svelte-h":!0}),h(ht)!=="svelte-q52n56"&&(ht.innerHTML=Jo),Jn=r(q),ft=c(q,"P",{"data-svelte-h":!0}),h(ft)!=="svelte-hswkmf"&&(ft.innerHTML=Uo),Un=r(q),P=c(q,"DIV",{class:!0});var O=$(P);u(Re.$$.fragment,O),Zn=r(O),ut=c(O,"P",{"data-svelte-h":!0}),h(ut)!=="svelte-1khdle2"&&(ut.innerHTML=Zo),Wn=r(O),u(oe.$$.fragment,O),O.forEach(o),q.forEach(o),an=r(e),u(Ge.$$.fragment,e),rn=r(e),F=c(e,"DIV",{class:!0});var I=$(F);u(Ve.$$.fragment,I),Nn=r(I),gt=c(I,"P",{"data-svelte-h":!0}),h(gt)!=="svelte-1sebo11"&&(gt.innerHTML=Wo),qn=r(I),_t=c(I,"P",{"data-svelte-h":!0}),h(_t)!=="svelte-q52n56"&&(_t.innerHTML=No),In=r(I),bt=c(I,"P",{"data-svelte-h":!0}),h(bt)!=="svelte-hswkmf"&&(bt.innerHTML=qo),Ln=r(I),L=c(I,"DIV",{class:!0});var G=$(L);u(He.$$.fragment,G),Bn=r(G),Tt=c(G,"P",{"data-svelte-h":!0}),h(Tt)!=="svelte-pwwfxm"&&(Tt.innerHTML=Io),Rn=r(G),u(se.$$.fragment,G),Gn=r(G),u(ae.$$.fragment,G),G.forEach(o),I.forEach(o),ln=r(e),u(Se.$$.fragment,e),dn=r(e),z=c(e,"DIV",{class:!0});var X=$(z);u(Pe.$$.fragment,X),Vn=r(X),yt=c(X,"P",{"data-svelte-h":!0}),h(yt)!=="svelte-1l7pgh5"&&(yt.textContent=Lo),Hn=r(X),Mt=c(X,"P",{"data-svelte-h":!0}),h(Mt)!=="svelte-q52n56"&&(Mt.innerHTML=Bo),Sn=r(X),kt=c(X,"P",{"data-svelte-h":!0}),h(kt)!=="svelte-hswkmf"&&(kt.innerHTML=Ro),Pn=r(X),W=c(X,"DIV",{class:!0});var Y=$(W);u(Xe.$$.fragment,Y),Xn=r(Y),vt=c(Y,"P",{"data-svelte-h":!0}),h(vt)!=="svelte-udgsis"&&(vt.innerHTML=Go),Yn=r(Y),u(re.$$.fragment,Y),An=r(Y),u(ie.$$.fragment,Y),Qn=r(Y),u(le.$$.fragment,Y),Y.forEach(o),X.forEach(o),cn=r(e),u(Ye.$$.fragment,e),mn=r(e),J=c(e,"DIV",{class:!0});var A=$(J);u(Ae.$$.fragment,A),Dn=r(A),wt=c(A,"P",{"data-svelte-h":!0}),h(wt)!=="svelte-1rwdd3l"&&(wt.textContent=Vo),On=r(A),$t=c(A,"P",{"data-svelte-h":!0}),h($t)!=="svelte-q52n56"&&($t.innerHTML=Ho),Kn=r(A),jt=c(A,"P",{"data-svelte-h":!0}),h(jt)!=="svelte-hswkmf"&&(jt.innerHTML=So),eo=r(A),B=c(A,"DIV",{class:!0});var he=$(B);u(Qe.$$.fragment,he),to=r(he),xt=c(he,"P",{"data-svelte-h":!0}),h(xt)!=="svelte-1ugjjee"&&(xt.innerHTML=Po),no=r(he),u(de.$$.fragment,he),oo=r(he),u(ce.$$.fragment,he),he.forEach(o),A.forEach(o),pn=r(e),u(De.$$.fragment,e),hn=r(e),U=c(e,"DIV",{class:!0});var Q=$(U);u(Oe.$$.fragment,Q),so=r(Q),Ct=c(Q,"P",{"data-svelte-h":!0}),h(Ct)!=="svelte-lkinlf"&&(Ct.textContent=Xo),ao=r(Q),Et=c(Q,"P",{"data-svelte-h":!0}),h(Et)!=="svelte-q52n56"&&(Et.innerHTML=Yo),ro=r(Q),Ft=c(Q,"P",{"data-svelte-h":!0}),h(Ft)!=="svelte-hswkmf"&&(Ft.innerHTML=Ao),io=r(Q),R=c(Q,"DIV",{class:!0});var fe=$(R);u(Ke.$$.fragment,fe),lo=r(fe),zt=c(fe,"P",{"data-svelte-h":!0}),h(zt)!=="svelte-1poj9kq"&&(zt.innerHTML=Qo),co=r(fe),u(me.$$.fragment,fe),mo=r(fe),u(pe.$$.fragment,fe),fe.forEach(o),Q.forEach(o),fn=r(e),u(et.$$.fragment,e),un=r(e),Zt=c(e,"P",{}),$(Zt).forEach(o),this.h()},h(){w(t,"name","hf:doc:metadata"),w(t,"content",us),w(K,"class","flex flex-wrap space-x-1"),w(te,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),w(C,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),w(it,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),w(ne,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),w(S,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),w(mt,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),w(j,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),w(P,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),w(E,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),w(L,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),w(F,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),w(W,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),w(z,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),w(B,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),w(J,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),w(R,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),w(U,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,n){i(document.head,t),l(e,y,n),l(e,m,n),l(e,p,n),l(e,M,n),l(e,k,n),g(ue,e,n),l(e,It,n),l(e,K,n),l(e,Lt,n),g(ge,e,n),l(e,Bt,n),l(e,_e,n),l(e,Rt,n),l(e,be,n),l(e,Gt,n),l(e,Te,n),l(e,Vt,n),l(e,ye,n),l(e,Ht,n),l(e,Me,n),l(e,St,n),l(e,ke,n),l(e,Pt,n),l(e,ve,n),l(e,Xt,n),l(e,we,n),l(e,Yt,n),l(e,$e,n),l(e,At,n),g(je,e,n),l(e,Qt,n),l(e,xe,n),l(e,Dt,n),g(Ce,e,n),l(e,Ot,n),l(e,Ee,n),l(e,Kt,n),g(Fe,e,n),l(e,en,n),l(e,C,n),g(ze,C,null),i(C,_n),i(C,ot),i(C,bn),i(C,st),i(C,Tn),g(ee,C,null),i(C,yn),i(C,te),g(Je,te,null),i(te,Mn),i(te,at),l(e,tn,n),g(Ue,e,n),l(e,nn,n),l(e,j,n),g(Ze,j,null),i(j,kn),i(j,rt),i(j,vn),i(j,it),g(We,it,null),i(j,wn),i(j,ne),g(Ne,ne,null),i(ne,$n),i(ne,lt),i(j,jn),i(j,S),g(qe,S,null),i(S,xn),i(S,dt),i(S,Cn),i(S,ct),i(j,En),i(j,mt),g(Ie,mt,null),l(e,on,n),g(Le,e,n),l(e,sn,n),l(e,E,n),g(Be,E,null),i(E,Fn),i(E,pt),i(E,zn),i(E,ht),i(E,Jn),i(E,ft),i(E,Un),i(E,P),g(Re,P,null),i(P,Zn),i(P,ut),i(P,Wn),g(oe,P,null),l(e,an,n),g(Ge,e,n),l(e,rn,n),l(e,F,n),g(Ve,F,null),i(F,Nn),i(F,gt),i(F,qn),i(F,_t),i(F,In),i(F,bt),i(F,Ln),i(F,L),g(He,L,null),i(L,Bn),i(L,Tt),i(L,Rn),g(se,L,null),i(L,Gn),g(ae,L,null),l(e,ln,n),g(Se,e,n),l(e,dn,n),l(e,z,n),g(Pe,z,null),i(z,Vn),i(z,yt),i(z,Hn),i(z,Mt),i(z,Sn),i(z,kt),i(z,Pn),i(z,W),g(Xe,W,null),i(W,Xn),i(W,vt),i(W,Yn),g(re,W,null),i(W,An),g(ie,W,null),i(W,Qn),g(le,W,null),l(e,cn,n),g(Ye,e,n),l(e,mn,n),l(e,J,n),g(Ae,J,null),i(J,Dn),i(J,wt),i(J,On),i(J,$t),i(J,Kn),i(J,jt),i(J,eo),i(J,B),g(Qe,B,null),i(B,to),i(B,xt),i(B,no),g(de,B,null),i(B,oo),g(ce,B,null),l(e,pn,n),g(De,e,n),l(e,hn,n),l(e,U,n),g(Oe,U,null),i(U,so),i(U,Ct),i(U,ao),i(U,Et),i(U,ro),i(U,Ft),i(U,io),i(U,R),g(Ke,R,null),i(R,lo),i(R,zt),i(R,co),g(me,R,null),i(R,mo),g(pe,R,null),l(e,fn,n),g(et,e,n),l(e,un,n),l(e,Zt,n),gn=!0},p(e,[n]){const N={};n&2&&(N.$$scope={dirty:n,ctx:e}),ee.$set(N);const tt={};n&2&&(tt.$$scope={dirty:n,ctx:e}),oe.$set(tt);const Z={};n&2&&(Z.$$scope={dirty:n,ctx:e}),se.$set(Z);const Wt={};n&2&&(Wt.$$scope={dirty:n,ctx:e}),ae.$set(Wt);const nt={};n&2&&(nt.$$scope={dirty:n,ctx:e}),re.$set(nt);const D={};n&2&&(D.$$scope={dirty:n,ctx:e}),ie.$set(D);const Nt={};n&2&&(Nt.$$scope={dirty:n,ctx:e}),le.$set(Nt);const q={};n&2&&(q.$$scope={dirty:n,ctx:e}),de.$set(q);const O={};n&2&&(O.$$scope={dirty:n,ctx:e}),ce.$set(O);const I={};n&2&&(I.$$scope={dirty:n,ctx:e}),me.$set(I);const G={};n&2&&(G.$$scope={dirty:n,ctx:e}),pe.$set(G)},i(e){gn||(_(ue.$$.fragment,e),_(ge.$$.fragment,e),_(je.$$.fragment,e),_(Ce.$$.fragment,e),_(Fe.$$.fragment,e),_(ze.$$.fragment,e),_(ee.$$.fragment,e),_(Je.$$.fragment,e),_(Ue.$$.fragment,e),_(Ze.$$.fragment,e),_(We.$$.fragment,e),_(Ne.$$.fragment,e),_(qe.$$.fragment,e),_(Ie.$$.fragment,e),_(Le.$$.fragment,e),_(Be.$$.fragment,e),_(Re.$$.fragment,e),_(oe.$$.fragment,e),_(Ge.$$.fragment,e),_(Ve.$$.fragment,e),_(He.$$.fragment,e),_(se.$$.fragment,e),_(ae.$$.fragment,e),_(Se.$$.fragment,e),_(Pe.$$.fragment,e),_(Xe.$$.fragment,e),_(re.$$.fragment,e),_(ie.$$.fragment,e),_(le.$$.fragment,e),_(Ye.$$.fragment,e),_(Ae.$$.fragment,e),_(Qe.$$.fragment,e),_(de.$$.fragment,e),_(ce.$$.fragment,e),_(De.$$.fragment,e),_(Oe.$$.fragment,e),_(Ke.$$.fragment,e),_(me.$$.fragment,e),_(pe.$$.fragment,e),_(et.$$.fragment,e),gn=!0)},o(e){b(ue.$$.fragment,e),b(ge.$$.fragment,e),b(je.$$.fragment,e),b(Ce.$$.fragment,e),b(Fe.$$.fragment,e),b(ze.$$.fragment,e),b(ee.$$.fragment,e),b(Je.$$.fragment,e),b(Ue.$$.fragment,e),b(Ze.$$.fragment,e),b(We.$$.fragment,e),b(Ne.$$.fragment,e),b(qe.$$.fragment,e),b(Ie.$$.fragment,e),b(Le.$$.fragment,e),b(Be.$$.fragment,e),b(Re.$$.fragment,e),b(oe.$$.fragment,e),b(Ge.$$.fragment,e),b(Ve.$$.fragment,e),b(He.$$.fragment,e),b(se.$$.fragment,e),b(ae.$$.fragment,e),b(Se.$$.fragment,e),b(Pe.$$.fragment,e),b(Xe.$$.fragment,e),b(re.$$.fragment,e),b(ie.$$.fragment,e),b(le.$$.fragment,e),b(Ye.$$.fragment,e),b(Ae.$$.fragment,e),b(Qe.$$.fragment,e),b(de.$$.fragment,e),b(ce.$$.fragment,e),b(De.$$.fragment,e),b(Oe.$$.fragment,e),b(Ke.$$.fragment,e),b(me.$$.fragment,e),b(pe.$$.fragment,e),b(et.$$.fragment,e),gn=!1},d(e){e&&(o(y),o(m),o(p),o(M),o(k),o(It),o(K),o(Lt),o(Bt),o(_e),o(Rt),o(be),o(Gt),o(Te),o(Vt),o(ye),o(Ht),o(Me),o(St),o(ke),o(Pt),o(ve),o(Xt),o(we),o(Yt),o($e),o(At),o(Qt),o(xe),o(Dt),o(Ot),o(Ee),o(Kt),o(en),o(C),o(tn),o(nn),o(j),o(on),o(sn),o(E),o(an),o(rn),o(F),o(ln),o(dn),o(z),o(cn),o(mn),o(J),o(pn),o(hn),o(U),o(fn),o(un),o(Zt)),o(t),T(ue,e),T(ge,e),T(je,e),T(Ce,e),T(Fe,e),T(ze),T(ee),T(Je),T(Ue,e),T(Ze),T(We),T(Ne),T(qe),T(Ie),T(Le,e),T(Be),T(Re),T(oe),T(Ge,e),T(Ve),T(He),T(se),T(ae),T(Se,e),T(Pe),T(Xe),T(re),T(ie),T(le),T(Ye,e),T(Ae),T(Qe),T(de),T(ce),T(De,e),T(Oe),T(Ke),T(me),T(pe),T(et,e)}}}const us='{"title":"ESM","local":"esm","sections":[{"title":"Overview","local":"overview","sections":[],"depth":2},{"title":"Usage tips","local":"usage-tips","sections":[],"depth":2},{"title":"Resources","local":"resources","sections":[],"depth":2},{"title":"EsmConfig","local":"transformers.EsmConfig","sections":[],"depth":2},{"title":"EsmTokenizer","local":"transformers.EsmTokenizer","sections":[],"depth":2},{"title":"EsmModel","local":"transformers.EsmModel","sections":[],"depth":2},{"title":"EsmForMaskedLM","local":"transformers.EsmForMaskedLM","sections":[],"depth":2},{"title":"EsmForSequenceClassification","local":"transformers.EsmForSequenceClassification","sections":[],"depth":2},{"title":"EsmForTokenClassification","local":"transformers.EsmForTokenClassification","sections":[],"depth":2},{"title":"EsmForProteinFolding","local":"transformers.EsmForProteinFolding","sections":[],"depth":2}],"depth":1}';function gs(v){return Oo(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class ws extends Ko{constructor(t){super(),es(this,t,gs,fs,Do,{})}}export{ws as component};
