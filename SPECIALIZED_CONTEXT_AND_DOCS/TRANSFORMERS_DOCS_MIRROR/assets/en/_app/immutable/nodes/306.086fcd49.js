import{s as _s,o as Ms,n as G}from"../chunks/scheduler.18a86fab.js";import{S as bs,i as ks,g as d,s as r,r as u,A as Ts,h as c,f as a,c as i,j as $,x as b,u as h,k as v,y as s,a as p,v as f,d as g,t as _,w as M}from"../chunks/index.98837b22.js";import{T as St}from"../chunks/Tip.77304350.js";import{D as N}from"../chunks/Docstring.a1ef7999.js";import{C as ft}from"../chunks/CodeBlock.8d0c2e8a.js";import{E as ht}from"../chunks/ExampleCodeBlock.8c3ee1f9.js";import{H as B,E as ys}from"../chunks/getInferenceSnippets.06c2775f.js";function ws(w){let t,k="Examples:",l,m,T;return m=new ft({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyME1QTmV0TW9kZWwlMkMlMjBNUE5ldENvbmZpZyUwQSUwQSUyMyUyMEluaXRpYWxpemluZyUyMGElMjBNUE5ldCUyMG1wbmV0LWJhc2UlMjBzdHlsZSUyMGNvbmZpZ3VyYXRpb24lMEFjb25maWd1cmF0aW9uJTIwJTNEJTIwTVBOZXRDb25maWcoKSUwQSUwQSUyMyUyMEluaXRpYWxpemluZyUyMGElMjBtb2RlbCUyMGZyb20lMjB0aGUlMjBtcG5ldC1iYXNlJTIwc3R5bGUlMjBjb25maWd1cmF0aW9uJTBBbW9kZWwlMjAlM0QlMjBNUE5ldE1vZGVsKGNvbmZpZ3VyYXRpb24pJTBBJTBBJTIzJTIwQWNjZXNzaW5nJTIwdGhlJTIwbW9kZWwlMjBjb25maWd1cmF0aW9uJTBBY29uZmlndXJhdGlvbiUyMCUzRCUyMG1vZGVsLmNvbmZpZw==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> MPNetModel, MPNetConfig

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a MPNet mpnet-base style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = MPNetConfig()

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a model from the mpnet-base style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = MPNetModel(configuration)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Accessing the model configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = model.config`,wrap:!1}}),{c(){t=d("p"),t.textContent=k,l=r(),u(m.$$.fragment)},l(n){t=c(n,"P",{"data-svelte-h":!0}),b(t)!=="svelte-kvfsh7"&&(t.textContent=k),l=i(n),h(m.$$.fragment,n)},m(n,y){p(n,t,y),p(n,l,y),f(m,n,y),T=!0},p:G,i(n){T||(g(m.$$.fragment,n),T=!0)},o(n){_(m.$$.fragment,n),T=!1},d(n){n&&(a(t),a(l)),M(m,n)}}}function vs(w){let t,k=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=d("p"),t.innerHTML=k},l(l){t=c(l,"P",{"data-svelte-h":!0}),b(t)!=="svelte-fincs2"&&(t.innerHTML=k)},m(l,m){p(l,t,m)},p:G,d(l){l&&a(t)}}}function $s(w){let t,k=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=d("p"),t.innerHTML=k},l(l){t=c(l,"P",{"data-svelte-h":!0}),b(t)!=="svelte-fincs2"&&(t.innerHTML=k)},m(l,m){p(l,t,m)},p:G,d(l){l&&a(t)}}}function Ns(w){let t,k="Example:",l,m,T;return m=new ft({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBNUE5ldEZvck1hc2tlZExNJTBBaW1wb3J0JTIwdG9yY2glMEElMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJtaWNyb3NvZnQlMkZtcG5ldC1iYXNlJTIyKSUwQW1vZGVsJTIwJTNEJTIwTVBOZXRGb3JNYXNrZWRMTS5mcm9tX3ByZXRyYWluZWQoJTIybWljcm9zb2Z0JTJGbXBuZXQtYmFzZSUyMiklMEElMEFpbnB1dHMlMjAlM0QlMjB0b2tlbml6ZXIoJTIyVGhlJTIwY2FwaXRhbCUyMG9mJTIwRnJhbmNlJTIwaXMlMjAlM0NtYXNrJTNFLiUyMiUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpJTBBJTBBd2l0aCUyMHRvcmNoLm5vX2dyYWQoKSUzQSUwQSUyMCUyMCUyMCUyMGxvZ2l0cyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzKS5sb2dpdHMlMEElMEElMjMlMjByZXRyaWV2ZSUyMGluZGV4JTIwb2YlMjAlM0NtYXNrJTNFJTBBbWFza190b2tlbl9pbmRleCUyMCUzRCUyMChpbnB1dHMuaW5wdXRfaWRzJTIwJTNEJTNEJTIwdG9rZW5pemVyLm1hc2tfdG9rZW5faWQpJTVCMCU1RC5ub256ZXJvKGFzX3R1cGxlJTNEVHJ1ZSklNUIwJTVEJTBBJTBBcHJlZGljdGVkX3Rva2VuX2lkJTIwJTNEJTIwbG9naXRzJTVCMCUyQyUyMG1hc2tfdG9rZW5faW5kZXglNUQuYXJnbWF4KGF4aXMlM0QtMSklMEF0b2tlbml6ZXIuZGVjb2RlKHByZWRpY3RlZF90b2tlbl9pZCklMEElMEFsYWJlbHMlMjAlM0QlMjB0b2tlbml6ZXIoJTIyVGhlJTIwY2FwaXRhbCUyMG9mJTIwRnJhbmNlJTIwaXMlMjBQYXJpcy4lMjIlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSU1QiUyMmlucHV0X2lkcyUyMiU1RCUwQSUyMyUyMG1hc2slMjBsYWJlbHMlMjBvZiUyMG5vbi0lM0NtYXNrJTNFJTIwdG9rZW5zJTBBbGFiZWxzJTIwJTNEJTIwdG9yY2gud2hlcmUoaW5wdXRzLmlucHV0X2lkcyUyMCUzRCUzRCUyMHRva2VuaXplci5tYXNrX3Rva2VuX2lkJTJDJTIwbGFiZWxzJTJDJTIwLTEwMCklMEElMEFvdXRwdXRzJTIwJTNEJTIwbW9kZWwoKippbnB1dHMlMkMlMjBsYWJlbHMlM0RsYWJlbHMpJTBBcm91bmQob3V0cHV0cy5sb3NzLml0ZW0oKSUyQyUyMDIp",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, MPNetForMaskedLM
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;microsoft/mpnet-base&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = MPNetForMaskedLM.from_pretrained(<span class="hljs-string">&quot;microsoft/mpnet-base&quot;</span>)

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
...`,wrap:!1}}),{c(){t=d("p"),t.textContent=k,l=r(),u(m.$$.fragment)},l(n){t=c(n,"P",{"data-svelte-h":!0}),b(t)!=="svelte-11lpom8"&&(t.textContent=k),l=i(n),h(m.$$.fragment,n)},m(n,y){p(n,t,y),p(n,l,y),f(m,n,y),T=!0},p:G,i(n){T||(g(m.$$.fragment,n),T=!0)},o(n){_(m.$$.fragment,n),T=!1},d(n){n&&(a(t),a(l)),M(m,n)}}}function Js(w){let t,k=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=d("p"),t.innerHTML=k},l(l){t=c(l,"P",{"data-svelte-h":!0}),b(t)!=="svelte-fincs2"&&(t.innerHTML=k)},m(l,m){p(l,t,m)},p:G,d(l){l&&a(t)}}}function js(w){let t,k="Example of single-label classification:",l,m,T;return m=new ft({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Rva2VuaXplciUyQyUyME1QTmV0Rm9yU2VxdWVuY2VDbGFzc2lmaWNhdGlvbiUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMm1pY3Jvc29mdCUyRm1wbmV0LWJhc2UlMjIpJTBBbW9kZWwlMjAlM0QlMjBNUE5ldEZvclNlcXVlbmNlQ2xhc3NpZmljYXRpb24uZnJvbV9wcmV0cmFpbmVkKCUyMm1pY3Jvc29mdCUyRm1wbmV0LWJhc2UlMjIpJTBBJTBBaW5wdXRzJTIwJTNEJTIwdG9rZW5pemVyKCUyMkhlbGxvJTJDJTIwbXklMjBkb2clMjBpcyUyMGN1dGUlMjIlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQSUwQXdpdGglMjB0b3JjaC5ub19ncmFkKCklM0ElMEElMjAlMjAlMjAlMjBsb2dpdHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cykubG9naXRzJTBBJTBBcHJlZGljdGVkX2NsYXNzX2lkJTIwJTNEJTIwbG9naXRzLmFyZ21heCgpLml0ZW0oKSUwQW1vZGVsLmNvbmZpZy5pZDJsYWJlbCU1QnByZWRpY3RlZF9jbGFzc19pZCU1RCUwQSUwQSUyMyUyMFRvJTIwdHJhaW4lMjBhJTIwbW9kZWwlMjBvbiUyMCU2MG51bV9sYWJlbHMlNjAlMjBjbGFzc2VzJTJDJTIweW91JTIwY2FuJTIwcGFzcyUyMCU2MG51bV9sYWJlbHMlM0RudW1fbGFiZWxzJTYwJTIwdG8lMjAlNjAuZnJvbV9wcmV0cmFpbmVkKC4uLiklNjAlMEFudW1fbGFiZWxzJTIwJTNEJTIwbGVuKG1vZGVsLmNvbmZpZy5pZDJsYWJlbCklMEFtb2RlbCUyMCUzRCUyME1QTmV0Rm9yU2VxdWVuY2VDbGFzc2lmaWNhdGlvbi5mcm9tX3ByZXRyYWluZWQoJTIybWljcm9zb2Z0JTJGbXBuZXQtYmFzZSUyMiUyQyUyMG51bV9sYWJlbHMlM0RudW1fbGFiZWxzKSUwQSUwQWxhYmVscyUyMCUzRCUyMHRvcmNoLnRlbnNvciglNUIxJTVEKSUwQWxvc3MlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyUyQyUyMGxhYmVscyUzRGxhYmVscykubG9zcyUwQXJvdW5kKGxvc3MuaXRlbSgpJTJDJTIwMik=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, MPNetForSequenceClassification

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;microsoft/mpnet-base&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = MPNetForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;microsoft/mpnet-base&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    logits = model(**inputs).logits

<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_class_id = logits.argmax().item()
<span class="hljs-meta">&gt;&gt;&gt; </span>model.config.id2label[predicted_class_id]
...

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># To train a model on \`num_labels\` classes, you can pass \`num_labels=num_labels\` to \`.from_pretrained(...)\`</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>num_labels = <span class="hljs-built_in">len</span>(model.config.id2label)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = MPNetForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;microsoft/mpnet-base&quot;</span>, num_labels=num_labels)

<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.tensor([<span class="hljs-number">1</span>])
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = model(**inputs, labels=labels).loss
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">round</span>(loss.item(), <span class="hljs-number">2</span>)
...`,wrap:!1}}),{c(){t=d("p"),t.textContent=k,l=r(),u(m.$$.fragment)},l(n){t=c(n,"P",{"data-svelte-h":!0}),b(t)!=="svelte-ykxpe4"&&(t.textContent=k),l=i(n),h(m.$$.fragment,n)},m(n,y){p(n,t,y),p(n,l,y),f(m,n,y),T=!0},p:G,i(n){T||(g(m.$$.fragment,n),T=!0)},o(n){_(m.$$.fragment,n),T=!1},d(n){n&&(a(t),a(l)),M(m,n)}}}function zs(w){let t,k="Example of multi-label classification:",l,m,T;return m=new ft({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Rva2VuaXplciUyQyUyME1QTmV0Rm9yU2VxdWVuY2VDbGFzc2lmaWNhdGlvbiUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMm1pY3Jvc29mdCUyRm1wbmV0LWJhc2UlMjIpJTBBbW9kZWwlMjAlM0QlMjBNUE5ldEZvclNlcXVlbmNlQ2xhc3NpZmljYXRpb24uZnJvbV9wcmV0cmFpbmVkKCUyMm1pY3Jvc29mdCUyRm1wbmV0LWJhc2UlMjIlMkMlMjBwcm9ibGVtX3R5cGUlM0QlMjJtdWx0aV9sYWJlbF9jbGFzc2lmaWNhdGlvbiUyMiklMEElMEFpbnB1dHMlMjAlM0QlMjB0b2tlbml6ZXIoJTIySGVsbG8lMkMlMjBteSUyMGRvZyUyMGlzJTIwY3V0ZSUyMiUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpJTBBJTBBd2l0aCUyMHRvcmNoLm5vX2dyYWQoKSUzQSUwQSUyMCUyMCUyMCUyMGxvZ2l0cyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzKS5sb2dpdHMlMEElMEFwcmVkaWN0ZWRfY2xhc3NfaWRzJTIwJTNEJTIwdG9yY2guYXJhbmdlKDAlMkMlMjBsb2dpdHMuc2hhcGUlNUItMSU1RCklNUJ0b3JjaC5zaWdtb2lkKGxvZ2l0cykuc3F1ZWV6ZShkaW0lM0QwKSUyMCUzRSUyMDAuNSU1RCUwQSUwQSUyMyUyMFRvJTIwdHJhaW4lMjBhJTIwbW9kZWwlMjBvbiUyMCU2MG51bV9sYWJlbHMlNjAlMjBjbGFzc2VzJTJDJTIweW91JTIwY2FuJTIwcGFzcyUyMCU2MG51bV9sYWJlbHMlM0RudW1fbGFiZWxzJTYwJTIwdG8lMjAlNjAuZnJvbV9wcmV0cmFpbmVkKC4uLiklNjAlMEFudW1fbGFiZWxzJTIwJTNEJTIwbGVuKG1vZGVsLmNvbmZpZy5pZDJsYWJlbCklMEFtb2RlbCUyMCUzRCUyME1QTmV0Rm9yU2VxdWVuY2VDbGFzc2lmaWNhdGlvbi5mcm9tX3ByZXRyYWluZWQoJTBBJTIwJTIwJTIwJTIwJTIybWljcm9zb2Z0JTJGbXBuZXQtYmFzZSUyMiUyQyUyMG51bV9sYWJlbHMlM0RudW1fbGFiZWxzJTJDJTIwcHJvYmxlbV90eXBlJTNEJTIybXVsdGlfbGFiZWxfY2xhc3NpZmljYXRpb24lMjIlMEEpJTBBJTBBbGFiZWxzJTIwJTNEJTIwdG9yY2guc3VtKCUwQSUyMCUyMCUyMCUyMHRvcmNoLm5uLmZ1bmN0aW9uYWwub25lX2hvdChwcmVkaWN0ZWRfY2xhc3NfaWRzJTVCTm9uZSUyQyUyMCUzQSU1RC5jbG9uZSgpJTJDJTIwbnVtX2NsYXNzZXMlM0RudW1fbGFiZWxzKSUyQyUyMGRpbSUzRDElMEEpLnRvKHRvcmNoLmZsb2F0KSUwQWxvc3MlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyUyQyUyMGxhYmVscyUzRGxhYmVscykubG9zcw==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, MPNetForSequenceClassification

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;microsoft/mpnet-base&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = MPNetForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;microsoft/mpnet-base&quot;</span>, problem_type=<span class="hljs-string">&quot;multi_label_classification&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    logits = model(**inputs).logits

<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_class_ids = torch.arange(<span class="hljs-number">0</span>, logits.shape[-<span class="hljs-number">1</span>])[torch.sigmoid(logits).squeeze(dim=<span class="hljs-number">0</span>) &gt; <span class="hljs-number">0.5</span>]

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># To train a model on \`num_labels\` classes, you can pass \`num_labels=num_labels\` to \`.from_pretrained(...)\`</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>num_labels = <span class="hljs-built_in">len</span>(model.config.id2label)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = MPNetForSequenceClassification.from_pretrained(
<span class="hljs-meta">... </span>    <span class="hljs-string">&quot;microsoft/mpnet-base&quot;</span>, num_labels=num_labels, problem_type=<span class="hljs-string">&quot;multi_label_classification&quot;</span>
<span class="hljs-meta">... </span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.<span class="hljs-built_in">sum</span>(
<span class="hljs-meta">... </span>    torch.nn.functional.one_hot(predicted_class_ids[<span class="hljs-literal">None</span>, :].clone(), num_classes=num_labels), dim=<span class="hljs-number">1</span>
<span class="hljs-meta">... </span>).to(torch.<span class="hljs-built_in">float</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = model(**inputs, labels=labels).loss`,wrap:!1}}),{c(){t=d("p"),t.textContent=k,l=r(),u(m.$$.fragment)},l(n){t=c(n,"P",{"data-svelte-h":!0}),b(t)!=="svelte-1l8e32d"&&(t.textContent=k),l=i(n),h(m.$$.fragment,n)},m(n,y){p(n,t,y),p(n,l,y),f(m,n,y),T=!0},p:G,i(n){T||(g(m.$$.fragment,n),T=!0)},o(n){_(m.$$.fragment,n),T=!1},d(n){n&&(a(t),a(l)),M(m,n)}}}function xs(w){let t,k=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=d("p"),t.innerHTML=k},l(l){t=c(l,"P",{"data-svelte-h":!0}),b(t)!=="svelte-fincs2"&&(t.innerHTML=k)},m(l,m){p(l,t,m)},p:G,d(l){l&&a(t)}}}function Cs(w){let t,k="Example:",l,m,T;return m=new ft({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBNUE5ldEZvck11bHRpcGxlQ2hvaWNlJTBBaW1wb3J0JTIwdG9yY2glMEElMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJtaWNyb3NvZnQlMkZtcG5ldC1iYXNlJTIyKSUwQW1vZGVsJTIwJTNEJTIwTVBOZXRGb3JNdWx0aXBsZUNob2ljZS5mcm9tX3ByZXRyYWluZWQoJTIybWljcm9zb2Z0JTJGbXBuZXQtYmFzZSUyMiklMEElMEFwcm9tcHQlMjAlM0QlMjAlMjJJbiUyMEl0YWx5JTJDJTIwcGl6emElMjBzZXJ2ZWQlMjBpbiUyMGZvcm1hbCUyMHNldHRpbmdzJTJDJTIwc3VjaCUyMGFzJTIwYXQlMjBhJTIwcmVzdGF1cmFudCUyQyUyMGlzJTIwcHJlc2VudGVkJTIwdW5zbGljZWQuJTIyJTBBY2hvaWNlMCUyMCUzRCUyMCUyMkl0JTIwaXMlMjBlYXRlbiUyMHdpdGglMjBhJTIwZm9yayUyMGFuZCUyMGElMjBrbmlmZS4lMjIlMEFjaG9pY2UxJTIwJTNEJTIwJTIySXQlMjBpcyUyMGVhdGVuJTIwd2hpbGUlMjBoZWxkJTIwaW4lMjB0aGUlMjBoYW5kLiUyMiUwQWxhYmVscyUyMCUzRCUyMHRvcmNoLnRlbnNvcigwKS51bnNxdWVlemUoMCklMjAlMjAlMjMlMjBjaG9pY2UwJTIwaXMlMjBjb3JyZWN0JTIwKGFjY29yZGluZyUyMHRvJTIwV2lraXBlZGlhJTIwJTNCKSklMkMlMjBiYXRjaCUyMHNpemUlMjAxJTBBJTBBZW5jb2RpbmclMjAlM0QlMjB0b2tlbml6ZXIoJTVCcHJvbXB0JTJDJTIwcHJvbXB0JTVEJTJDJTIwJTVCY2hvaWNlMCUyQyUyMGNob2ljZTElNUQlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyJTJDJTIwcGFkZGluZyUzRFRydWUpJTBBb3V0cHV0cyUyMCUzRCUyMG1vZGVsKCoqJTdCayUzQSUyMHYudW5zcXVlZXplKDApJTIwZm9yJTIwayUyQyUyMHYlMjBpbiUyMGVuY29kaW5nLml0ZW1zKCklN0QlMkMlMjBsYWJlbHMlM0RsYWJlbHMpJTIwJTIwJTIzJTIwYmF0Y2glMjBzaXplJTIwaXMlMjAxJTBBJTBBJTIzJTIwdGhlJTIwbGluZWFyJTIwY2xhc3NpZmllciUyMHN0aWxsJTIwbmVlZHMlMjB0byUyMGJlJTIwdHJhaW5lZCUwQWxvc3MlMjAlM0QlMjBvdXRwdXRzLmxvc3MlMEFsb2dpdHMlMjAlM0QlMjBvdXRwdXRzLmxvZ2l0cw==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, MPNetForMultipleChoice
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;microsoft/mpnet-base&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = MPNetForMultipleChoice.from_pretrained(<span class="hljs-string">&quot;microsoft/mpnet-base&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>prompt = <span class="hljs-string">&quot;In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>choice0 = <span class="hljs-string">&quot;It is eaten with a fork and a knife.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>choice1 = <span class="hljs-string">&quot;It is eaten while held in the hand.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.tensor(<span class="hljs-number">0</span>).unsqueeze(<span class="hljs-number">0</span>)  <span class="hljs-comment"># choice0 is correct (according to Wikipedia ;)), batch size 1</span>

<span class="hljs-meta">&gt;&gt;&gt; </span>encoding = tokenizer([prompt, prompt], [choice0, choice1], return_tensors=<span class="hljs-string">&quot;pt&quot;</span>, padding=<span class="hljs-literal">True</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**{k: v.unsqueeze(<span class="hljs-number">0</span>) <span class="hljs-keyword">for</span> k, v <span class="hljs-keyword">in</span> encoding.items()}, labels=labels)  <span class="hljs-comment"># batch size is 1</span>

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># the linear classifier still needs to be trained</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = outputs.loss
<span class="hljs-meta">&gt;&gt;&gt; </span>logits = outputs.logits`,wrap:!1}}),{c(){t=d("p"),t.textContent=k,l=r(),u(m.$$.fragment)},l(n){t=c(n,"P",{"data-svelte-h":!0}),b(t)!=="svelte-11lpom8"&&(t.textContent=k),l=i(n),h(m.$$.fragment,n)},m(n,y){p(n,t,y),p(n,l,y),f(m,n,y),T=!0},p:G,i(n){T||(g(m.$$.fragment,n),T=!0)},o(n){_(m.$$.fragment,n),T=!1},d(n){n&&(a(t),a(l)),M(m,n)}}}function Ps(w){let t,k=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=d("p"),t.innerHTML=k},l(l){t=c(l,"P",{"data-svelte-h":!0}),b(t)!=="svelte-fincs2"&&(t.innerHTML=k)},m(l,m){p(l,t,m)},p:G,d(l){l&&a(t)}}}function Fs(w){let t,k="Example:",l,m,T;return m=new ft({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBNUE5ldEZvclRva2VuQ2xhc3NpZmljYXRpb24lMEFpbXBvcnQlMjB0b3JjaCUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMm1pY3Jvc29mdCUyRm1wbmV0LWJhc2UlMjIpJTBBbW9kZWwlMjAlM0QlMjBNUE5ldEZvclRva2VuQ2xhc3NpZmljYXRpb24uZnJvbV9wcmV0cmFpbmVkKCUyMm1pY3Jvc29mdCUyRm1wbmV0LWJhc2UlMjIpJTBBJTBBaW5wdXRzJTIwJTNEJTIwdG9rZW5pemVyKCUwQSUyMCUyMCUyMCUyMCUyMkh1Z2dpbmdGYWNlJTIwaXMlMjBhJTIwY29tcGFueSUyMGJhc2VkJTIwaW4lMjBQYXJpcyUyMGFuZCUyME5ldyUyMFlvcmslMjIlMkMlMjBhZGRfc3BlY2lhbF90b2tlbnMlM0RGYWxzZSUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIlMEEpJTBBJTBBd2l0aCUyMHRvcmNoLm5vX2dyYWQoKSUzQSUwQSUyMCUyMCUyMCUyMGxvZ2l0cyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzKS5sb2dpdHMlMEElMEFwcmVkaWN0ZWRfdG9rZW5fY2xhc3NfaWRzJTIwJTNEJTIwbG9naXRzLmFyZ21heCgtMSklMEElMEElMjMlMjBOb3RlJTIwdGhhdCUyMHRva2VucyUyMGFyZSUyMGNsYXNzaWZpZWQlMjByYXRoZXIlMjB0aGVuJTIwaW5wdXQlMjB3b3JkcyUyMHdoaWNoJTIwbWVhbnMlMjB0aGF0JTBBJTIzJTIwdGhlcmUlMjBtaWdodCUyMGJlJTIwbW9yZSUyMHByZWRpY3RlZCUyMHRva2VuJTIwY2xhc3NlcyUyMHRoYW4lMjB3b3Jkcy4lMEElMjMlMjBNdWx0aXBsZSUyMHRva2VuJTIwY2xhc3NlcyUyMG1pZ2h0JTIwYWNjb3VudCUyMGZvciUyMHRoZSUyMHNhbWUlMjB3b3JkJTBBcHJlZGljdGVkX3Rva2Vuc19jbGFzc2VzJTIwJTNEJTIwJTVCbW9kZWwuY29uZmlnLmlkMmxhYmVsJTVCdC5pdGVtKCklNUQlMjBmb3IlMjB0JTIwaW4lMjBwcmVkaWN0ZWRfdG9rZW5fY2xhc3NfaWRzJTVCMCU1RCU1RCUwQXByZWRpY3RlZF90b2tlbnNfY2xhc3NlcyUwQSUwQWxhYmVscyUyMCUzRCUyMHByZWRpY3RlZF90b2tlbl9jbGFzc19pZHMlMEFsb3NzJTIwJTNEJTIwbW9kZWwoKippbnB1dHMlMkMlMjBsYWJlbHMlM0RsYWJlbHMpLmxvc3MlMEFyb3VuZChsb3NzLml0ZW0oKSUyQyUyMDIp",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, MPNetForTokenClassification
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;microsoft/mpnet-base&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = MPNetForTokenClassification.from_pretrained(<span class="hljs-string">&quot;microsoft/mpnet-base&quot;</span>)

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
...`,wrap:!1}}),{c(){t=d("p"),t.textContent=k,l=r(),u(m.$$.fragment)},l(n){t=c(n,"P",{"data-svelte-h":!0}),b(t)!=="svelte-11lpom8"&&(t.textContent=k),l=i(n),h(m.$$.fragment,n)},m(n,y){p(n,t,y),p(n,l,y),f(m,n,y),T=!0},p:G,i(n){T||(g(m.$$.fragment,n),T=!0)},o(n){_(m.$$.fragment,n),T=!1},d(n){n&&(a(t),a(l)),M(m,n)}}}function Us(w){let t,k=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=d("p"),t.innerHTML=k},l(l){t=c(l,"P",{"data-svelte-h":!0}),b(t)!=="svelte-fincs2"&&(t.innerHTML=k)},m(l,m){p(l,t,m)},p:G,d(l){l&&a(t)}}}function Ws(w){let t,k="Example:",l,m,T;return m=new ft({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBNUE5ldEZvclF1ZXN0aW9uQW5zd2VyaW5nJTBBaW1wb3J0JTIwdG9yY2glMEElMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJtaWNyb3NvZnQlMkZtcG5ldC1iYXNlJTIyKSUwQW1vZGVsJTIwJTNEJTIwTVBOZXRGb3JRdWVzdGlvbkFuc3dlcmluZy5mcm9tX3ByZXRyYWluZWQoJTIybWljcm9zb2Z0JTJGbXBuZXQtYmFzZSUyMiklMEElMEFxdWVzdGlvbiUyQyUyMHRleHQlMjAlM0QlMjAlMjJXaG8lMjB3YXMlMjBKaW0lMjBIZW5zb24lM0YlMjIlMkMlMjAlMjJKaW0lMjBIZW5zb24lMjB3YXMlMjBhJTIwbmljZSUyMHB1cHBldCUyMiUwQSUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplcihxdWVzdGlvbiUyQyUyMHRleHQlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQXdpdGglMjB0b3JjaC5ub19ncmFkKCklM0ElMEElMjAlMjAlMjAlMjBvdXRwdXRzJTIwJTNEJTIwbW9kZWwoKippbnB1dHMpJTBBJTBBYW5zd2VyX3N0YXJ0X2luZGV4JTIwJTNEJTIwb3V0cHV0cy5zdGFydF9sb2dpdHMuYXJnbWF4KCklMEFhbnN3ZXJfZW5kX2luZGV4JTIwJTNEJTIwb3V0cHV0cy5lbmRfbG9naXRzLmFyZ21heCgpJTBBJTBBcHJlZGljdF9hbnN3ZXJfdG9rZW5zJTIwJTNEJTIwaW5wdXRzLmlucHV0X2lkcyU1QjAlMkMlMjBhbnN3ZXJfc3RhcnRfaW5kZXglMjAlM0ElMjBhbnN3ZXJfZW5kX2luZGV4JTIwJTJCJTIwMSU1RCUwQXRva2VuaXplci5kZWNvZGUocHJlZGljdF9hbnN3ZXJfdG9rZW5zJTJDJTIwc2tpcF9zcGVjaWFsX3Rva2VucyUzRFRydWUpJTBBJTBBJTIzJTIwdGFyZ2V0JTIwaXMlMjAlMjJuaWNlJTIwcHVwcGV0JTIyJTBBdGFyZ2V0X3N0YXJ0X2luZGV4JTIwJTNEJTIwdG9yY2gudGVuc29yKCU1QjE0JTVEKSUwQXRhcmdldF9lbmRfaW5kZXglMjAlM0QlMjB0b3JjaC50ZW5zb3IoJTVCMTUlNUQpJTBBJTBBb3V0cHV0cyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzJTJDJTIwc3RhcnRfcG9zaXRpb25zJTNEdGFyZ2V0X3N0YXJ0X2luZGV4JTJDJTIwZW5kX3Bvc2l0aW9ucyUzRHRhcmdldF9lbmRfaW5kZXgpJTBBbG9zcyUyMCUzRCUyMG91dHB1dHMubG9zcyUwQXJvdW5kKGxvc3MuaXRlbSgpJTJDJTIwMik=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, MPNetForQuestionAnswering
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;microsoft/mpnet-base&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = MPNetForQuestionAnswering.from_pretrained(<span class="hljs-string">&quot;microsoft/mpnet-base&quot;</span>)

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
...`,wrap:!1}}),{c(){t=d("p"),t.textContent=k,l=r(),u(m.$$.fragment)},l(n){t=c(n,"P",{"data-svelte-h":!0}),b(t)!=="svelte-11lpom8"&&(t.textContent=k),l=i(n),h(m.$$.fragment,n)},m(n,y){p(n,t,y),p(n,l,y),f(m,n,y),T=!0},p:G,i(n){T||(g(m.$$.fragment,n),T=!0)},o(n){_(m.$$.fragment,n),T=!1},d(n){n&&(a(t),a(l)),M(m,n)}}}function qs(w){let t,k,l,m,T,n="<em>This model was released on 2020-04-20 and added to Hugging Face Transformers on 2020-12-09.</em>",y,we,Ot,oe,Po='<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/>',Dt,ve,Kt,$e,Fo='The MPNet model was proposed in <a href="https://huggingface.co/papers/2004.09297" rel="nofollow">MPNet: Masked and Permuted Pre-training for Language Understanding</a> by Kaitao Song, Xu Tan, Tao Qin, Jianfeng Lu, Tie-Yan Liu.',en,Ne,Uo=`MPNet adopts a novel pre-training method, named masked and permuted language modeling, to inherit the advantages of
masked language modeling and permuted language modeling for natural language understanding.`,tn,Je,Wo="The abstract from the paper is the following:",nn,je,qo=`<em>BERT adopts masked language modeling (MLM) for pre-training and is one of the most successful pre-training models.
Since BERT neglects dependency among predicted tokens, XLNet introduces permuted language modeling (PLM) for
pre-training to address this problem. However, XLNet does not leverage the full position information of a sentence and
thus suffers from position discrepancy between pre-training and fine-tuning. In this paper, we propose MPNet, a novel
pre-training method that inherits the advantages of BERT and XLNet and avoids their limitations. MPNet leverages the
dependency among predicted tokens through permuted language modeling (vs. MLM in BERT), and takes auxiliary position
information as input to make the model see a full sentence and thus reducing the position discrepancy (vs. PLM in
XLNet). We pre-train MPNet on a large-scale dataset (over 160GB text corpora) and fine-tune on a variety of
down-streaming tasks (GLUE, SQuAD, etc). Experimental results show that MPNet outperforms MLM and PLM by a large
margin, and achieves better results on these tasks compared with previous state-of-the-art pre-trained methods (e.g.,
BERT, XLNet, RoBERTa) under the same model setting.</em>`,on,ze,Io='The original code can be found <a href="https://github.com/microsoft/MPNet" rel="nofollow">here</a>.',sn,xe,an,Ce,Zo=`MPNet doesn’t have <code>token_type_ids</code>, you don’t need to indicate which token belongs to which segment. Just
separate your segments with the separation token <code>tokenizer.sep_token</code> (or <code>[sep]</code>).`,rn,Pe,ln,Fe,Bo='<li><a href="../tasks/sequence_classification">Text classification task guide</a></li> <li><a href="../tasks/token_classification">Token classification task guide</a></li> <li><a href="../tasks/question_answering">Question answering task guide</a></li> <li><a href="../tasks/masked_language_modeling">Masked language modeling task guide</a></li> <li><a href="../tasks/multiple_choice">Multiple choice task guide</a></li>',dn,Ue,cn,W,We,xn,gt,Go=`This is the configuration class to store the configuration of a <a href="/docs/transformers/v4.56.2/en/model_doc/mpnet#transformers.MPNetModel">MPNetModel</a> or a <code>TFMPNetModel</code>. It is used to
instantiate a MPNet model according to the specified arguments, defining the model architecture. Instantiating a
configuration with the defaults will yield a similar configuration to that of the MPNet
<a href="https://huggingface.co/microsoft/mpnet-base" rel="nofollow">microsoft/mpnet-base</a> architecture.`,Cn,_t,Vo=`Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> and can be used to control the model outputs. Read the
documentation from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> for more information.`,Pn,se,pn,qe,mn,J,Ie,Fn,Mt,Lo=`This tokenizer inherits from <a href="/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertTokenizer">BertTokenizer</a> which contains most of the methods. Users should refer to the
superclass for more information regarding methods.`,Un,S,Ze,Wn,bt,Ro=`Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
adding special tokens. A MPNet sequence has the following format:`,qn,kt,Ho="<li>single sequence: <code>&lt;s&gt; X &lt;/s&gt;</code></li> <li>pair of sequences: <code>&lt;s&gt; A &lt;/s&gt;&lt;/s&gt; B &lt;/s&gt;</code></li>",In,ae,Be,Zn,Tt,Xo=`Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
special tokens using the tokenizer <code>prepare_for_model</code> methods.`,Bn,re,Ge,Gn,yt,Eo=`Creates a mask from the two sequences passed to be used in a sequence-pair classification task. MPNet does not
make use of token type ids, therefore a list of zeros is returned.`,Vn,wt,Ve,un,Le,hn,q,Re,Ln,vt,Qo="Construct a “fast” MPNet tokenizer (backed by HuggingFace’s <em>tokenizers</em> library). Based on WordPiece.",Rn,$t,So=`This tokenizer inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast">PreTrainedTokenizerFast</a> which contains most of the main methods. Users should
refer to this superclass for more information regarding those methods.`,Hn,ie,He,Xn,Nt,Yo=`Creates a mask from the two sequences passed to be used in a sequence-pair classification task. MPNet does not
make use of token type ids, therefore a list of zeros is returned`,fn,Xe,gn,j,Ee,En,Jt,Ao="The bare Mpnet Model outputting raw hidden-states without any specific head on top.",Qn,jt,Oo=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Sn,zt,Do=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Yn,Y,Qe,An,xt,Ko='The <a href="/docs/transformers/v4.56.2/en/model_doc/mpnet#transformers.MPNetModel">MPNetModel</a> forward method, overrides the <code>__call__</code> special method.',On,le,_n,Se,Mn,ee,Ye,Dn,V,Ae,Kn,Ct,es='The <a href="/docs/transformers/v4.56.2/en/model_doc/mpnet#transformers.MPNetForMaskedLM">MPNetForMaskedLM</a> forward method, overrides the <code>__call__</code> special method.',eo,de,to,ce,bn,Oe,kn,z,De,no,Pt,ts=`MPNet Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled
output) e.g. for GLUE tasks.`,oo,Ft,ns=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,so,Ut,os=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,ao,U,Ke,ro,Wt,ss='The <a href="/docs/transformers/v4.56.2/en/model_doc/mpnet#transformers.MPNetForSequenceClassification">MPNetForSequenceClassification</a> forward method, overrides the <code>__call__</code> special method.',io,pe,lo,me,co,ue,Tn,et,yn,x,tt,po,qt,as=`The Mpnet Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
softmax) e.g. for RocStories/SWAG tasks.`,mo,It,rs=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,uo,Zt,is=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,ho,L,nt,fo,Bt,ls='The <a href="/docs/transformers/v4.56.2/en/model_doc/mpnet#transformers.MPNetForMultipleChoice">MPNetForMultipleChoice</a> forward method, overrides the <code>__call__</code> special method.',go,he,_o,fe,wn,ot,vn,C,st,Mo,Gt,ds=`The Mpnet transformer with a token classification head on top (a linear layer on top of the hidden-states
output) e.g. for Named-Entity-Recognition (NER) tasks.`,bo,Vt,cs=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,ko,Lt,ps=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,To,R,at,yo,Rt,ms='The <a href="/docs/transformers/v4.56.2/en/model_doc/mpnet#transformers.MPNetForTokenClassification">MPNetForTokenClassification</a> forward method, overrides the <code>__call__</code> special method.',wo,ge,vo,_e,$n,rt,Nn,P,it,$o,Ht,us=`The Mpnet transformer with a span classification head on top for extractive question-answering tasks like
SQuAD (a linear layer on top of the hidden-states output to compute <code>span start logits</code> and <code>span end logits</code>).`,No,Xt,hs=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Jo,Et,fs=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,jo,H,lt,zo,Qt,gs='The <a href="/docs/transformers/v4.56.2/en/model_doc/mpnet#transformers.MPNetForQuestionAnswering">MPNetForQuestionAnswering</a> forward method, overrides the <code>__call__</code> special method.',xo,Me,Co,be,Jn,dt,jn,Yt,zn;return we=new B({props:{title:"MPNet",local:"mpnet",headingTag:"h1"}}),ve=new B({props:{title:"Overview",local:"overview",headingTag:"h2"}}),xe=new B({props:{title:"Usage tips",local:"usage-tips",headingTag:"h2"}}),Pe=new B({props:{title:"Resources",local:"resources",headingTag:"h2"}}),Ue=new B({props:{title:"MPNetConfig",local:"transformers.MPNetConfig",headingTag:"h2"}}),We=new N({props:{name:"class transformers.MPNetConfig",anchor:"transformers.MPNetConfig",parameters:[{name:"vocab_size",val:" = 30527"},{name:"hidden_size",val:" = 768"},{name:"num_hidden_layers",val:" = 12"},{name:"num_attention_heads",val:" = 12"},{name:"intermediate_size",val:" = 3072"},{name:"hidden_act",val:" = 'gelu'"},{name:"hidden_dropout_prob",val:" = 0.1"},{name:"attention_probs_dropout_prob",val:" = 0.1"},{name:"max_position_embeddings",val:" = 512"},{name:"initializer_range",val:" = 0.02"},{name:"layer_norm_eps",val:" = 1e-12"},{name:"relative_attention_num_buckets",val:" = 32"},{name:"pad_token_id",val:" = 1"},{name:"bos_token_id",val:" = 0"},{name:"eos_token_id",val:" = 2"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.MPNetConfig.vocab_size",description:`<strong>vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 30527) &#x2014;
Vocabulary size of the MPNet model. Defines the number of different tokens that can be represented by the
<code>inputs_ids</code> passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/mpnet#transformers.MPNetModel">MPNetModel</a> or <code>TFMPNetModel</code>.`,name:"vocab_size"},{anchor:"transformers.MPNetConfig.hidden_size",description:`<strong>hidden_size</strong> (<code>int</code>, <em>optional</em>, defaults to 768) &#x2014;
Dimensionality of the encoder layers and the pooler layer.`,name:"hidden_size"},{anchor:"transformers.MPNetConfig.num_hidden_layers",description:`<strong>num_hidden_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 12) &#x2014;
Number of hidden layers in the Transformer encoder.`,name:"num_hidden_layers"},{anchor:"transformers.MPNetConfig.num_attention_heads",description:`<strong>num_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 12) &#x2014;
Number of attention heads for each attention layer in the Transformer encoder.`,name:"num_attention_heads"},{anchor:"transformers.MPNetConfig.intermediate_size",description:`<strong>intermediate_size</strong> (<code>int</code>, <em>optional</em>, defaults to 3072) &#x2014;
Dimensionality of the &#x201C;intermediate&#x201D; (often named feed-forward) layer in the Transformer encoder.`,name:"intermediate_size"},{anchor:"transformers.MPNetConfig.hidden_act",description:`<strong>hidden_act</strong> (<code>str</code> or <code>Callable</code>, <em>optional</em>, defaults to <code>&quot;gelu&quot;</code>) &#x2014;
The non-linear activation function (function or string) in the encoder and pooler. If string, <code>&quot;gelu&quot;</code>,
<code>&quot;relu&quot;</code>, <code>&quot;silu&quot;</code> and <code>&quot;gelu_new&quot;</code> are supported.`,name:"hidden_act"},{anchor:"transformers.MPNetConfig.hidden_dropout_prob",description:`<strong>hidden_dropout_prob</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.`,name:"hidden_dropout_prob"},{anchor:"transformers.MPNetConfig.attention_probs_dropout_prob",description:`<strong>attention_probs_dropout_prob</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout ratio for the attention probabilities.`,name:"attention_probs_dropout_prob"},{anchor:"transformers.MPNetConfig.max_position_embeddings",description:`<strong>max_position_embeddings</strong> (<code>int</code>, <em>optional</em>, defaults to 512) &#x2014;
The maximum sequence length that this model might ever be used with. Typically set this to something large
just in case (e.g., 512 or 1024 or 2048).`,name:"max_position_embeddings"},{anchor:"transformers.MPNetConfig.initializer_range",description:`<strong>initializer_range</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.`,name:"initializer_range"},{anchor:"transformers.MPNetConfig.layer_norm_eps",description:`<strong>layer_norm_eps</strong> (<code>float</code>, <em>optional</em>, defaults to 1e-12) &#x2014;
The epsilon used by the layer normalization layers.`,name:"layer_norm_eps"},{anchor:"transformers.MPNetConfig.relative_attention_num_buckets",description:`<strong>relative_attention_num_buckets</strong> (<code>int</code>, <em>optional</em>, defaults to 32) &#x2014;
The number of buckets to use for each attention layer.`,name:"relative_attention_num_buckets"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mpnet/configuration_mpnet.py#L25"}}),se=new ht({props:{anchor:"transformers.MPNetConfig.example",$$slots:{default:[ws]},$$scope:{ctx:w}}}),qe=new B({props:{title:"MPNetTokenizer",local:"transformers.MPNetTokenizer",headingTag:"h2"}}),Ie=new N({props:{name:"class transformers.MPNetTokenizer",anchor:"transformers.MPNetTokenizer",parameters:[{name:"vocab_file",val:""},{name:"do_lower_case",val:" = True"},{name:"do_basic_tokenize",val:" = True"},{name:"never_split",val:" = None"},{name:"bos_token",val:" = '<s>'"},{name:"eos_token",val:" = '</s>'"},{name:"sep_token",val:" = '</s>'"},{name:"cls_token",val:" = '<s>'"},{name:"unk_token",val:" = '[UNK]'"},{name:"pad_token",val:" = '<pad>'"},{name:"mask_token",val:" = '<mask>'"},{name:"tokenize_chinese_chars",val:" = True"},{name:"strip_accents",val:" = None"},{name:"clean_up_tokenization_spaces",val:" = True"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.MPNetTokenizer.vocab_file",description:`<strong>vocab_file</strong> (<code>str</code>) &#x2014;
Path to the vocabulary file.`,name:"vocab_file"},{anchor:"transformers.MPNetTokenizer.do_lower_case",description:`<strong>do_lower_case</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to lowercase the input when tokenizing.`,name:"do_lower_case"},{anchor:"transformers.MPNetTokenizer.do_basic_tokenize",description:`<strong>do_basic_tokenize</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to do basic tokenization before WordPiece.`,name:"do_basic_tokenize"},{anchor:"transformers.MPNetTokenizer.never_split",description:`<strong>never_split</strong> (<code>Iterable</code>, <em>optional</em>) &#x2014;
Collection of tokens which will never be split during tokenization. Only has an effect when
<code>do_basic_tokenize=True</code>`,name:"never_split"},{anchor:"transformers.MPNetTokenizer.bos_token",description:`<strong>bos_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;s&gt;&quot;</code>) &#x2014;
The beginning of sequence token that was used during pre-training. Can be used a sequence classifier token.</p>
<div class="course-tip  bg-gradient-to-br dark:bg-gradient-to-r before:border-green-500 dark:before:border-green-800 from-green-50 dark:from-gray-900 to-white dark:to-gray-950 border border-green-50 text-green-700 dark:text-gray-400">
						
<p>When building a sequence using special tokens, this is not the token that is used for the beginning of
sequence. The token used is the <code>cls_token</code>.</p>

					</div>`,name:"bos_token"},{anchor:"transformers.MPNetTokenizer.eos_token",description:`<strong>eos_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;/s&gt;&quot;</code>) &#x2014;
The end of sequence token.</p>
<div class="course-tip  bg-gradient-to-br dark:bg-gradient-to-r before:border-green-500 dark:before:border-green-800 from-green-50 dark:from-gray-900 to-white dark:to-gray-950 border border-green-50 text-green-700 dark:text-gray-400">
						
<p>When building a sequence using special tokens, this is not the token that is used for the end of sequence.
The token used is the <code>sep_token</code>.</p>

					</div>`,name:"eos_token"},{anchor:"transformers.MPNetTokenizer.sep_token",description:`<strong>sep_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;/s&gt;&quot;</code>) &#x2014;
The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
sequence classification or for a text and a question for question answering. It is also used as the last
token of a sequence built with special tokens.`,name:"sep_token"},{anchor:"transformers.MPNetTokenizer.cls_token",description:`<strong>cls_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;s&gt;&quot;</code>) &#x2014;
The classifier token which is used when doing sequence classification (classification of the whole sequence
instead of per-token classification). It is the first token of the sequence when built with special tokens.`,name:"cls_token"},{anchor:"transformers.MPNetTokenizer.unk_token",description:`<strong>unk_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;[UNK]&quot;</code>) &#x2014;
The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
token instead.`,name:"unk_token"},{anchor:"transformers.MPNetTokenizer.pad_token",description:`<strong>pad_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;pad&gt;&quot;</code>) &#x2014;
The token used for padding, for example when batching sequences of different lengths.`,name:"pad_token"},{anchor:"transformers.MPNetTokenizer.mask_token",description:`<strong>mask_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;mask&gt;&quot;</code>) &#x2014;
The token used for masking values. This is the token used when training this model with masked language
modeling. This is the token which the model will try to predict.`,name:"mask_token"},{anchor:"transformers.MPNetTokenizer.tokenize_chinese_chars",description:`<strong>tokenize_chinese_chars</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to tokenize Chinese characters.</p>
<p>This should likely be deactivated for Japanese (see this
<a href="https://github.com/huggingface/transformers/issues/328" rel="nofollow">issue</a>).`,name:"tokenize_chinese_chars"},{anchor:"transformers.MPNetTokenizer.strip_accents",description:`<strong>strip_accents</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to strip all accents. If this option is not specified, then it will be determined by the
value for <code>lowercase</code> (as in the original BERT).`,name:"strip_accents"},{anchor:"transformers.MPNetTokenizer.clean_up_tokenization_spaces",description:`<strong>clean_up_tokenization_spaces</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to cleanup spaces after decoding, cleanup consists in removing potential artifacts like
extra spaces.`,name:"clean_up_tokenization_spaces"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mpnet/tokenization_mpnet.py#L52"}}),Ze=new N({props:{name:"build_inputs_with_special_tokens",anchor:"transformers.MPNetTokenizer.build_inputs_with_special_tokens",parameters:[{name:"token_ids_0",val:": list"},{name:"token_ids_1",val:": typing.Optional[list[int]] = None"}],parametersDescription:[{anchor:"transformers.MPNetTokenizer.build_inputs_with_special_tokens.token_ids_0",description:`<strong>token_ids_0</strong> (<code>list[int]</code>) &#x2014;
List of IDs to which the special tokens will be added`,name:"token_ids_0"},{anchor:"transformers.MPNetTokenizer.build_inputs_with_special_tokens.token_ids_1",description:`<strong>token_ids_1</strong> (<code>list[int]</code>, <em>optional</em>) &#x2014;
Optional second list of IDs for sequence pairs.`,name:"token_ids_1"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mpnet/tokenization_mpnet.py#L221",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>list of <a href="../glossary#input-ids">input IDs</a> with the appropriate special tokens.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>list[int]</code></p>
`}}),Be=new N({props:{name:"get_special_tokens_mask",anchor:"transformers.MPNetTokenizer.get_special_tokens_mask",parameters:[{name:"token_ids_0",val:": list"},{name:"token_ids_1",val:": typing.Optional[list[int]] = None"},{name:"already_has_special_tokens",val:": bool = False"}],parametersDescription:[{anchor:"transformers.MPNetTokenizer.get_special_tokens_mask.token_ids_0",description:`<strong>token_ids_0</strong> (<code>list[int]</code>) &#x2014;
List of ids.`,name:"token_ids_0"},{anchor:"transformers.MPNetTokenizer.get_special_tokens_mask.token_ids_1",description:`<strong>token_ids_1</strong> (<code>list[int]</code>, <em>optional</em>) &#x2014;
Optional second list of IDs for sequence pairs.`,name:"token_ids_1"},{anchor:"transformers.MPNetTokenizer.get_special_tokens_mask.already_has_special_tokens",description:`<strong>already_has_special_tokens</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Set to True if the token list is already formatted with special tokens for the model`,name:"already_has_special_tokens"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mpnet/tokenization_mpnet.py#L246",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>list[int]</code></p>
`}}),Ge=new N({props:{name:"create_token_type_ids_from_sequences",anchor:"transformers.MPNetTokenizer.create_token_type_ids_from_sequences",parameters:[{name:"token_ids_0",val:": list"},{name:"token_ids_1",val:": typing.Optional[list[int]] = None"}],parametersDescription:[{anchor:"transformers.MPNetTokenizer.create_token_type_ids_from_sequences.token_ids_0",description:`<strong>token_ids_0</strong> (<code>list[int]</code>) &#x2014;
List of ids.`,name:"token_ids_0"},{anchor:"transformers.MPNetTokenizer.create_token_type_ids_from_sequences.token_ids_1",description:`<strong>token_ids_1</strong> (<code>list[int]</code>, <em>optional</em>) &#x2014;
Optional second list of IDs for sequence pairs.`,name:"token_ids_1"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mpnet/tokenization_mpnet.py#L273",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>List of zeros.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>list[int]</code></p>
`}}),Ve=new N({props:{name:"save_vocabulary",anchor:"transformers.MPNetTokenizer.save_vocabulary",parameters:[{name:"save_directory",val:": str"},{name:"filename_prefix",val:": typing.Optional[str] = None"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mpnet/tokenization_mpnet.py#L296"}}),Le=new B({props:{title:"MPNetTokenizerFast",local:"transformers.MPNetTokenizerFast",headingTag:"h2"}}),Re=new N({props:{name:"class transformers.MPNetTokenizerFast",anchor:"transformers.MPNetTokenizerFast",parameters:[{name:"vocab_file",val:" = None"},{name:"tokenizer_file",val:" = None"},{name:"do_lower_case",val:" = True"},{name:"bos_token",val:" = '<s>'"},{name:"eos_token",val:" = '</s>'"},{name:"sep_token",val:" = '</s>'"},{name:"cls_token",val:" = '<s>'"},{name:"unk_token",val:" = '[UNK]'"},{name:"pad_token",val:" = '<pad>'"},{name:"mask_token",val:" = '<mask>'"},{name:"tokenize_chinese_chars",val:" = True"},{name:"strip_accents",val:" = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.MPNetTokenizerFast.vocab_file",description:`<strong>vocab_file</strong> (<code>str</code>) &#x2014;
File containing the vocabulary.`,name:"vocab_file"},{anchor:"transformers.MPNetTokenizerFast.do_lower_case",description:`<strong>do_lower_case</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to lowercase the input when tokenizing.`,name:"do_lower_case"},{anchor:"transformers.MPNetTokenizerFast.bos_token",description:`<strong>bos_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;s&gt;&quot;</code>) &#x2014;
The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.</p>
<div class="course-tip  bg-gradient-to-br dark:bg-gradient-to-r before:border-green-500 dark:before:border-green-800 from-green-50 dark:from-gray-900 to-white dark:to-gray-950 border border-green-50 text-green-700 dark:text-gray-400">
						
<p>When building a sequence using special tokens, this is not the token that is used for the beginning of
sequence. The token used is the <code>cls_token</code>.</p>

					</div>`,name:"bos_token"},{anchor:"transformers.MPNetTokenizerFast.eos_token",description:`<strong>eos_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;/s&gt;&quot;</code>) &#x2014;
The end of sequence token.</p>
<div class="course-tip  bg-gradient-to-br dark:bg-gradient-to-r before:border-green-500 dark:before:border-green-800 from-green-50 dark:from-gray-900 to-white dark:to-gray-950 border border-green-50 text-green-700 dark:text-gray-400">
						
<p>When building a sequence using special tokens, this is not the token that is used for the end of sequence.
The token used is the <code>sep_token</code>.</p>

					</div>`,name:"eos_token"},{anchor:"transformers.MPNetTokenizerFast.sep_token",description:`<strong>sep_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;/s&gt;&quot;</code>) &#x2014;
The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
sequence classification or for a text and a question for question answering. It is also used as the last
token of a sequence built with special tokens.`,name:"sep_token"},{anchor:"transformers.MPNetTokenizerFast.cls_token",description:`<strong>cls_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;s&gt;&quot;</code>) &#x2014;
The classifier token which is used when doing sequence classification (classification of the whole sequence
instead of per-token classification). It is the first token of the sequence when built with special tokens.`,name:"cls_token"},{anchor:"transformers.MPNetTokenizerFast.unk_token",description:`<strong>unk_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;[UNK]&quot;</code>) &#x2014;
The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
token instead.`,name:"unk_token"},{anchor:"transformers.MPNetTokenizerFast.pad_token",description:`<strong>pad_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;pad&gt;&quot;</code>) &#x2014;
The token used for padding, for example when batching sequences of different lengths.`,name:"pad_token"},{anchor:"transformers.MPNetTokenizerFast.mask_token",description:`<strong>mask_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;mask&gt;&quot;</code>) &#x2014;
The token used for masking values. This is the token used when training this model with masked language
modeling. This is the token which the model will try to predict.`,name:"mask_token"},{anchor:"transformers.MPNetTokenizerFast.tokenize_chinese_chars",description:`<strong>tokenize_chinese_chars</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to tokenize Chinese characters. This should likely be deactivated for Japanese (see <a href="https://github.com/huggingface/transformers/issues/328" rel="nofollow">this
issue</a>).`,name:"tokenize_chinese_chars"},{anchor:"transformers.MPNetTokenizerFast.strip_accents",description:`<strong>strip_accents</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to strip all accents. If this option is not specified, then it will be determined by the
value for <code>lowercase</code> (as in the original BERT).`,name:"strip_accents"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mpnet/tokenization_mpnet_fast.py#L34"}}),He=new N({props:{name:"create_token_type_ids_from_sequences",anchor:"transformers.MPNetTokenizerFast.create_token_type_ids_from_sequences",parameters:[{name:"token_ids_0",val:": list"},{name:"token_ids_1",val:": typing.Optional[list[int]] = None"}],parametersDescription:[{anchor:"transformers.MPNetTokenizerFast.create_token_type_ids_from_sequences.token_ids_0",description:`<strong>token_ids_0</strong> (<code>list[int]</code>) &#x2014;
List of ids.`,name:"token_ids_0"},{anchor:"transformers.MPNetTokenizerFast.create_token_type_ids_from_sequences.token_ids_1",description:`<strong>token_ids_1</strong> (<code>list[int]</code>, <em>optional</em>) &#x2014;
Optional second list of IDs for sequence pairs`,name:"token_ids_1"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mpnet/tokenization_mpnet_fast.py#L181",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>List of zeros.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>list[int]</code></p>
`}}),Xe=new B({props:{title:"MPNetModel",local:"transformers.MPNetModel",headingTag:"h2"}}),Ee=new N({props:{name:"class transformers.MPNetModel",anchor:"transformers.MPNetModel",parameters:[{name:"config",val:""},{name:"add_pooling_layer",val:" = True"}],parametersDescription:[{anchor:"transformers.MPNetModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/mpnet#transformers.MPNetModel">MPNetModel</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"},{anchor:"transformers.MPNetModel.add_pooling_layer",description:`<strong>add_pooling_layer</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether to add a pooling layer`,name:"add_pooling_layer"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mpnet/modeling_mpnet.py#L419"}}),Qe=new N({props:{name:"forward",anchor:"transformers.MPNetModel.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.MPNetModel.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.MPNetModel.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.MPNetModel.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.MPNetModel.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.MPNetModel.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.MPNetModel.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.MPNetModel.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.MPNetModel.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mpnet/modeling_mpnet.py#L449",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling"
>transformers.modeling_outputs.BaseModelOutputWithPooling</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/mpnet#transformers.MPNetConfig"
>MPNetConfig</a>) and inputs.</p>
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
`}}),le=new St({props:{$$slots:{default:[vs]},$$scope:{ctx:w}}}),Se=new B({props:{title:"MPNetForMaskedLM",local:"transformers.MPNetForMaskedLM",headingTag:"h2"}}),Ye=new N({props:{name:"class transformers.MPNetForMaskedLM",anchor:"transformers.MPNetForMaskedLM",parameters:[{name:"config",val:""}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mpnet/modeling_mpnet.py#L508"}}),Ae=new N({props:{name:"forward",anchor:"transformers.MPNetForMaskedLM.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.MPNetForMaskedLM.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.MPNetForMaskedLM.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.MPNetForMaskedLM.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.MPNetForMaskedLM.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.MPNetForMaskedLM.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.MPNetForMaskedLM.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should be in <code>[-100, 0, ..., config.vocab_size]</code> (see <code>input_ids</code> docstring) Tokens with indices set to <code>-100</code> are ignored (masked), the
loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>`,name:"labels"},{anchor:"transformers.MPNetForMaskedLM.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.MPNetForMaskedLM.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.MPNetForMaskedLM.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mpnet/modeling_mpnet.py#L527",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.MaskedLMOutput"
>transformers.modeling_outputs.MaskedLMOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/mpnet#transformers.MPNetConfig"
>MPNetConfig</a>) and inputs.</p>
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
`}}),de=new St({props:{$$slots:{default:[$s]},$$scope:{ctx:w}}}),ce=new ht({props:{anchor:"transformers.MPNetForMaskedLM.forward.example",$$slots:{default:[Ns]},$$scope:{ctx:w}}}),Oe=new B({props:{title:"MPNetForSequenceClassification",local:"transformers.MPNetForSequenceClassification",headingTag:"h2"}}),De=new N({props:{name:"class transformers.MPNetForSequenceClassification",anchor:"transformers.MPNetForSequenceClassification",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.MPNetForSequenceClassification.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/mpnet#transformers.MPNetForSequenceClassification">MPNetForSequenceClassification</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mpnet/modeling_mpnet.py#L613"}}),Ke=new N({props:{name:"forward",anchor:"transformers.MPNetForSequenceClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.MPNetForSequenceClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.MPNetForSequenceClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.MPNetForSequenceClassification.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.MPNetForSequenceClassification.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.MPNetForSequenceClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.MPNetForSequenceClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the sequence classification/regression loss. Indices should be in <code>[0, ..., config.num_labels - 1]</code>. If <code>config.num_labels == 1</code> a regression loss is computed (Mean-Square loss), If
<code>config.num_labels &gt; 1</code> a classification loss is computed (Cross-Entropy).`,name:"labels"},{anchor:"transformers.MPNetForSequenceClassification.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.MPNetForSequenceClassification.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.MPNetForSequenceClassification.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mpnet/modeling_mpnet.py#L624",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput"
>transformers.modeling_outputs.SequenceClassifierOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/mpnet#transformers.MPNetConfig"
>MPNetConfig</a>) and inputs.</p>
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
`}}),pe=new St({props:{$$slots:{default:[Js]},$$scope:{ctx:w}}}),me=new ht({props:{anchor:"transformers.MPNetForSequenceClassification.forward.example",$$slots:{default:[js]},$$scope:{ctx:w}}}),ue=new ht({props:{anchor:"transformers.MPNetForSequenceClassification.forward.example-2",$$slots:{default:[zs]},$$scope:{ctx:w}}}),et=new B({props:{title:"MPNetForMultipleChoice",local:"transformers.MPNetForMultipleChoice",headingTag:"h2"}}),tt=new N({props:{name:"class transformers.MPNetForMultipleChoice",anchor:"transformers.MPNetForMultipleChoice",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.MPNetForMultipleChoice.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/mpnet#transformers.MPNetForMultipleChoice">MPNetForMultipleChoice</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mpnet/modeling_mpnet.py#L694"}}),nt=new N({props:{name:"forward",anchor:"transformers.MPNetForMultipleChoice.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.MPNetForMultipleChoice.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, num_choices, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.MPNetForMultipleChoice.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.MPNetForMultipleChoice.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, num_choices, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.max_position_embeddings - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.MPNetForMultipleChoice.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.MPNetForMultipleChoice.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_choices, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <em>input_ids</em> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.MPNetForMultipleChoice.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the multiple choice classification loss. Indices should be in <code>[0, ..., num_choices-1]</code> where <code>num_choices</code> is the size of the second dimension of the input tensors. (See
<code>input_ids</code> above)`,name:"labels"},{anchor:"transformers.MPNetForMultipleChoice.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.MPNetForMultipleChoice.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.MPNetForMultipleChoice.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mpnet/modeling_mpnet.py#L705",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.MultipleChoiceModelOutput"
>transformers.modeling_outputs.MultipleChoiceModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/mpnet#transformers.MPNetConfig"
>MPNetConfig</a>) and inputs.</p>
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
`}}),he=new St({props:{$$slots:{default:[xs]},$$scope:{ctx:w}}}),fe=new ht({props:{anchor:"transformers.MPNetForMultipleChoice.forward.example",$$slots:{default:[Cs]},$$scope:{ctx:w}}}),ot=new B({props:{title:"MPNetForTokenClassification",local:"transformers.MPNetForTokenClassification",headingTag:"h2"}}),st=new N({props:{name:"class transformers.MPNetForTokenClassification",anchor:"transformers.MPNetForTokenClassification",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.MPNetForTokenClassification.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/mpnet#transformers.MPNetForTokenClassification">MPNetForTokenClassification</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mpnet/modeling_mpnet.py#L787"}}),at=new N({props:{name:"forward",anchor:"transformers.MPNetForTokenClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.MPNetForTokenClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.MPNetForTokenClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.MPNetForTokenClassification.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.MPNetForTokenClassification.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.MPNetForTokenClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.MPNetForTokenClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the token classification loss. Indices should be in <code>[0, ..., config.num_labels - 1]</code>.`,name:"labels"},{anchor:"transformers.MPNetForTokenClassification.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.MPNetForTokenClassification.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.MPNetForTokenClassification.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mpnet/modeling_mpnet.py#L799",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput"
>transformers.modeling_outputs.TokenClassifierOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/mpnet#transformers.MPNetConfig"
>MPNetConfig</a>) and inputs.</p>
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
`}}),ge=new St({props:{$$slots:{default:[Ps]},$$scope:{ctx:w}}}),_e=new ht({props:{anchor:"transformers.MPNetForTokenClassification.forward.example",$$slots:{default:[Fs]},$$scope:{ctx:w}}}),rt=new B({props:{title:"MPNetForQuestionAnswering",local:"transformers.MPNetForQuestionAnswering",headingTag:"h2"}}),it=new N({props:{name:"class transformers.MPNetForQuestionAnswering",anchor:"transformers.MPNetForQuestionAnswering",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.MPNetForQuestionAnswering.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/mpnet#transformers.MPNetForQuestionAnswering">MPNetForQuestionAnswering</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mpnet/modeling_mpnet.py#L872"}}),lt=new N({props:{name:"forward",anchor:"transformers.MPNetForQuestionAnswering.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"start_positions",val:": typing.Optional[torch.LongTensor] = None"},{name:"end_positions",val:": typing.Optional[torch.LongTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.MPNetForQuestionAnswering.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.MPNetForQuestionAnswering.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.MPNetForQuestionAnswering.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.MPNetForQuestionAnswering.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.MPNetForQuestionAnswering.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.MPNetForQuestionAnswering.forward.start_positions",description:`<strong>start_positions</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for position (index) of the start of the labelled span for computing the token classification loss.
Positions are clamped to the length of the sequence (<code>sequence_length</code>). Position outside of the sequence
are not taken into account for computing the loss.`,name:"start_positions"},{anchor:"transformers.MPNetForQuestionAnswering.forward.end_positions",description:`<strong>end_positions</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for position (index) of the end of the labelled span for computing the token classification loss.
Positions are clamped to the length of the sequence (<code>sequence_length</code>). Position outside of the sequence
are not taken into account for computing the loss.`,name:"end_positions"},{anchor:"transformers.MPNetForQuestionAnswering.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.MPNetForQuestionAnswering.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.MPNetForQuestionAnswering.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mpnet/modeling_mpnet.py#L883",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.QuestionAnsweringModelOutput"
>transformers.modeling_outputs.QuestionAnsweringModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/mpnet#transformers.MPNetConfig"
>MPNetConfig</a>) and inputs.</p>
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
`}}),Me=new St({props:{$$slots:{default:[Us]},$$scope:{ctx:w}}}),be=new ht({props:{anchor:"transformers.MPNetForQuestionAnswering.forward.example",$$slots:{default:[Ws]},$$scope:{ctx:w}}}),dt=new ys({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/mpnet.md"}}),{c(){t=d("meta"),k=r(),l=d("p"),m=r(),T=d("p"),T.innerHTML=n,y=r(),u(we.$$.fragment),Ot=r(),oe=d("div"),oe.innerHTML=Po,Dt=r(),u(ve.$$.fragment),Kt=r(),$e=d("p"),$e.innerHTML=Fo,en=r(),Ne=d("p"),Ne.textContent=Uo,tn=r(),Je=d("p"),Je.textContent=Wo,nn=r(),je=d("p"),je.innerHTML=qo,on=r(),ze=d("p"),ze.innerHTML=Io,sn=r(),u(xe.$$.fragment),an=r(),Ce=d("p"),Ce.innerHTML=Zo,rn=r(),u(Pe.$$.fragment),ln=r(),Fe=d("ul"),Fe.innerHTML=Bo,dn=r(),u(Ue.$$.fragment),cn=r(),W=d("div"),u(We.$$.fragment),xn=r(),gt=d("p"),gt.innerHTML=Go,Cn=r(),_t=d("p"),_t.innerHTML=Vo,Pn=r(),u(se.$$.fragment),pn=r(),u(qe.$$.fragment),mn=r(),J=d("div"),u(Ie.$$.fragment),Fn=r(),Mt=d("p"),Mt.innerHTML=Lo,Un=r(),S=d("div"),u(Ze.$$.fragment),Wn=r(),bt=d("p"),bt.textContent=Ro,qn=r(),kt=d("ul"),kt.innerHTML=Ho,In=r(),ae=d("div"),u(Be.$$.fragment),Zn=r(),Tt=d("p"),Tt.innerHTML=Xo,Bn=r(),re=d("div"),u(Ge.$$.fragment),Gn=r(),yt=d("p"),yt.textContent=Eo,Vn=r(),wt=d("div"),u(Ve.$$.fragment),un=r(),u(Le.$$.fragment),hn=r(),q=d("div"),u(Re.$$.fragment),Ln=r(),vt=d("p"),vt.innerHTML=Qo,Rn=r(),$t=d("p"),$t.innerHTML=So,Hn=r(),ie=d("div"),u(He.$$.fragment),Xn=r(),Nt=d("p"),Nt.textContent=Yo,fn=r(),u(Xe.$$.fragment),gn=r(),j=d("div"),u(Ee.$$.fragment),En=r(),Jt=d("p"),Jt.textContent=Ao,Qn=r(),jt=d("p"),jt.innerHTML=Oo,Sn=r(),zt=d("p"),zt.innerHTML=Do,Yn=r(),Y=d("div"),u(Qe.$$.fragment),An=r(),xt=d("p"),xt.innerHTML=Ko,On=r(),u(le.$$.fragment),_n=r(),u(Se.$$.fragment),Mn=r(),ee=d("div"),u(Ye.$$.fragment),Dn=r(),V=d("div"),u(Ae.$$.fragment),Kn=r(),Ct=d("p"),Ct.innerHTML=es,eo=r(),u(de.$$.fragment),to=r(),u(ce.$$.fragment),bn=r(),u(Oe.$$.fragment),kn=r(),z=d("div"),u(De.$$.fragment),no=r(),Pt=d("p"),Pt.textContent=ts,oo=r(),Ft=d("p"),Ft.innerHTML=ns,so=r(),Ut=d("p"),Ut.innerHTML=os,ao=r(),U=d("div"),u(Ke.$$.fragment),ro=r(),Wt=d("p"),Wt.innerHTML=ss,io=r(),u(pe.$$.fragment),lo=r(),u(me.$$.fragment),co=r(),u(ue.$$.fragment),Tn=r(),u(et.$$.fragment),yn=r(),x=d("div"),u(tt.$$.fragment),po=r(),qt=d("p"),qt.textContent=as,mo=r(),It=d("p"),It.innerHTML=rs,uo=r(),Zt=d("p"),Zt.innerHTML=is,ho=r(),L=d("div"),u(nt.$$.fragment),fo=r(),Bt=d("p"),Bt.innerHTML=ls,go=r(),u(he.$$.fragment),_o=r(),u(fe.$$.fragment),wn=r(),u(ot.$$.fragment),vn=r(),C=d("div"),u(st.$$.fragment),Mo=r(),Gt=d("p"),Gt.textContent=ds,bo=r(),Vt=d("p"),Vt.innerHTML=cs,ko=r(),Lt=d("p"),Lt.innerHTML=ps,To=r(),R=d("div"),u(at.$$.fragment),yo=r(),Rt=d("p"),Rt.innerHTML=ms,wo=r(),u(ge.$$.fragment),vo=r(),u(_e.$$.fragment),$n=r(),u(rt.$$.fragment),Nn=r(),P=d("div"),u(it.$$.fragment),$o=r(),Ht=d("p"),Ht.innerHTML=us,No=r(),Xt=d("p"),Xt.innerHTML=hs,Jo=r(),Et=d("p"),Et.innerHTML=fs,jo=r(),H=d("div"),u(lt.$$.fragment),zo=r(),Qt=d("p"),Qt.innerHTML=gs,xo=r(),u(Me.$$.fragment),Co=r(),u(be.$$.fragment),Jn=r(),u(dt.$$.fragment),jn=r(),Yt=d("p"),this.h()},l(e){const o=Ts("svelte-u9bgzb",document.head);t=c(o,"META",{name:!0,content:!0}),o.forEach(a),k=i(e),l=c(e,"P",{}),$(l).forEach(a),m=i(e),T=c(e,"P",{"data-svelte-h":!0}),b(T)!=="svelte-i4zuvm"&&(T.innerHTML=n),y=i(e),h(we.$$.fragment,e),Ot=i(e),oe=c(e,"DIV",{class:!0,"data-svelte-h":!0}),b(oe)!=="svelte-13t8s2t"&&(oe.innerHTML=Po),Dt=i(e),h(ve.$$.fragment,e),Kt=i(e),$e=c(e,"P",{"data-svelte-h":!0}),b($e)!=="svelte-12ug9vo"&&($e.innerHTML=Fo),en=i(e),Ne=c(e,"P",{"data-svelte-h":!0}),b(Ne)!=="svelte-nhc0f1"&&(Ne.textContent=Uo),tn=i(e),Je=c(e,"P",{"data-svelte-h":!0}),b(Je)!=="svelte-vfdo9a"&&(Je.textContent=Wo),nn=i(e),je=c(e,"P",{"data-svelte-h":!0}),b(je)!=="svelte-1jkkhrp"&&(je.innerHTML=qo),on=i(e),ze=c(e,"P",{"data-svelte-h":!0}),b(ze)!=="svelte-1xcjgnq"&&(ze.innerHTML=Io),sn=i(e),h(xe.$$.fragment,e),an=i(e),Ce=c(e,"P",{"data-svelte-h":!0}),b(Ce)!=="svelte-pcnnpr"&&(Ce.innerHTML=Zo),rn=i(e),h(Pe.$$.fragment,e),ln=i(e),Fe=c(e,"UL",{"data-svelte-h":!0}),b(Fe)!=="svelte-mgusi3"&&(Fe.innerHTML=Bo),dn=i(e),h(Ue.$$.fragment,e),cn=i(e),W=c(e,"DIV",{class:!0});var X=$(W);h(We.$$.fragment,X),xn=i(X),gt=c(X,"P",{"data-svelte-h":!0}),b(gt)!=="svelte-1ufo3p1"&&(gt.innerHTML=Go),Cn=i(X),_t=c(X,"P",{"data-svelte-h":!0}),b(_t)!=="svelte-1ek1ss9"&&(_t.innerHTML=Vo),Pn=i(X),h(se.$$.fragment,X),X.forEach(a),pn=i(e),h(qe.$$.fragment,e),mn=i(e),J=c(e,"DIV",{class:!0});var F=$(J);h(Ie.$$.fragment,F),Fn=i(F),Mt=c(F,"P",{"data-svelte-h":!0}),b(Mt)!=="svelte-2hsfhp"&&(Mt.innerHTML=Lo),Un=i(F),S=c(F,"DIV",{class:!0});var te=$(S);h(Ze.$$.fragment,te),Wn=i(te),bt=c(te,"P",{"data-svelte-h":!0}),b(bt)!=="svelte-e3idx9"&&(bt.textContent=Ro),qn=i(te),kt=c(te,"UL",{"data-svelte-h":!0}),b(kt)!=="svelte-rq8uot"&&(kt.innerHTML=Ho),te.forEach(a),In=i(F),ae=c(F,"DIV",{class:!0});var ct=$(ae);h(Be.$$.fragment,ct),Zn=i(ct),Tt=c(ct,"P",{"data-svelte-h":!0}),b(Tt)!=="svelte-rsgabl"&&(Tt.innerHTML=Xo),ct.forEach(a),Bn=i(F),re=c(F,"DIV",{class:!0});var pt=$(re);h(Ge.$$.fragment,pt),Gn=i(pt),yt=c(pt,"P",{"data-svelte-h":!0}),b(yt)!=="svelte-55gsom"&&(yt.textContent=Eo),pt.forEach(a),Vn=i(F),wt=c(F,"DIV",{class:!0});var At=$(wt);h(Ve.$$.fragment,At),At.forEach(a),F.forEach(a),un=i(e),h(Le.$$.fragment,e),hn=i(e),q=c(e,"DIV",{class:!0});var E=$(q);h(Re.$$.fragment,E),Ln=i(E),vt=c(E,"P",{"data-svelte-h":!0}),b(vt)!=="svelte-lg4gj2"&&(vt.innerHTML=Qo),Rn=i(E),$t=c(E,"P",{"data-svelte-h":!0}),b($t)!=="svelte-gxzj9w"&&($t.innerHTML=So),Hn=i(E),ie=c(E,"DIV",{class:!0});var mt=$(ie);h(He.$$.fragment,mt),Xn=i(mt),Nt=c(mt,"P",{"data-svelte-h":!0}),b(Nt)!=="svelte-442qh0"&&(Nt.textContent=Yo),mt.forEach(a),E.forEach(a),fn=i(e),h(Xe.$$.fragment,e),gn=i(e),j=c(e,"DIV",{class:!0});var I=$(j);h(Ee.$$.fragment,I),En=i(I),Jt=c(I,"P",{"data-svelte-h":!0}),b(Jt)!=="svelte-146w4va"&&(Jt.textContent=Ao),Qn=i(I),jt=c(I,"P",{"data-svelte-h":!0}),b(jt)!=="svelte-q52n56"&&(jt.innerHTML=Oo),Sn=i(I),zt=c(I,"P",{"data-svelte-h":!0}),b(zt)!=="svelte-hswkmf"&&(zt.innerHTML=Do),Yn=i(I),Y=c(I,"DIV",{class:!0});var ne=$(Y);h(Qe.$$.fragment,ne),An=i(ne),xt=c(ne,"P",{"data-svelte-h":!0}),b(xt)!=="svelte-jhzvrp"&&(xt.innerHTML=Ko),On=i(ne),h(le.$$.fragment,ne),ne.forEach(a),I.forEach(a),_n=i(e),h(Se.$$.fragment,e),Mn=i(e),ee=c(e,"DIV",{class:!0});var ut=$(ee);h(Ye.$$.fragment,ut),Dn=i(ut),V=c(ut,"DIV",{class:!0});var Q=$(V);h(Ae.$$.fragment,Q),Kn=i(Q),Ct=c(Q,"P",{"data-svelte-h":!0}),b(Ct)!=="svelte-3z1jmp"&&(Ct.innerHTML=es),eo=i(Q),h(de.$$.fragment,Q),to=i(Q),h(ce.$$.fragment,Q),Q.forEach(a),ut.forEach(a),bn=i(e),h(Oe.$$.fragment,e),kn=i(e),z=c(e,"DIV",{class:!0});var Z=$(z);h(De.$$.fragment,Z),no=i(Z),Pt=c(Z,"P",{"data-svelte-h":!0}),b(Pt)!=="svelte-i6k3hq"&&(Pt.textContent=ts),oo=i(Z),Ft=c(Z,"P",{"data-svelte-h":!0}),b(Ft)!=="svelte-q52n56"&&(Ft.innerHTML=ns),so=i(Z),Ut=c(Z,"P",{"data-svelte-h":!0}),b(Ut)!=="svelte-hswkmf"&&(Ut.innerHTML=os),ao=i(Z),U=c(Z,"DIV",{class:!0});var A=$(U);h(Ke.$$.fragment,A),ro=i(A),Wt=c(A,"P",{"data-svelte-h":!0}),b(Wt)!=="svelte-qps1q7"&&(Wt.innerHTML=ss),io=i(A),h(pe.$$.fragment,A),lo=i(A),h(me.$$.fragment,A),co=i(A),h(ue.$$.fragment,A),A.forEach(a),Z.forEach(a),Tn=i(e),h(et.$$.fragment,e),yn=i(e),x=c(e,"DIV",{class:!0});var O=$(x);h(tt.$$.fragment,O),po=i(O),qt=c(O,"P",{"data-svelte-h":!0}),b(qt)!=="svelte-14nyxr1"&&(qt.textContent=as),mo=i(O),It=c(O,"P",{"data-svelte-h":!0}),b(It)!=="svelte-q52n56"&&(It.innerHTML=rs),uo=i(O),Zt=c(O,"P",{"data-svelte-h":!0}),b(Zt)!=="svelte-hswkmf"&&(Zt.innerHTML=is),ho=i(O),L=c(O,"DIV",{class:!0});var ke=$(L);h(nt.$$.fragment,ke),fo=i(ke),Bt=c(ke,"P",{"data-svelte-h":!0}),b(Bt)!=="svelte-8o0tgr"&&(Bt.innerHTML=ls),go=i(ke),h(he.$$.fragment,ke),_o=i(ke),h(fe.$$.fragment,ke),ke.forEach(a),O.forEach(a),wn=i(e),h(ot.$$.fragment,e),vn=i(e),C=c(e,"DIV",{class:!0});var D=$(C);h(st.$$.fragment,D),Mo=i(D),Gt=c(D,"P",{"data-svelte-h":!0}),b(Gt)!=="svelte-1nu6zrm"&&(Gt.textContent=ds),bo=i(D),Vt=c(D,"P",{"data-svelte-h":!0}),b(Vt)!=="svelte-q52n56"&&(Vt.innerHTML=cs),ko=i(D),Lt=c(D,"P",{"data-svelte-h":!0}),b(Lt)!=="svelte-hswkmf"&&(Lt.innerHTML=ps),To=i(D),R=c(D,"DIV",{class:!0});var Te=$(R);h(at.$$.fragment,Te),yo=i(Te),Rt=c(Te,"P",{"data-svelte-h":!0}),b(Rt)!=="svelte-1pd8k5n"&&(Rt.innerHTML=ms),wo=i(Te),h(ge.$$.fragment,Te),vo=i(Te),h(_e.$$.fragment,Te),Te.forEach(a),D.forEach(a),$n=i(e),h(rt.$$.fragment,e),Nn=i(e),P=c(e,"DIV",{class:!0});var K=$(P);h(it.$$.fragment,K),$o=i(K),Ht=c(K,"P",{"data-svelte-h":!0}),b(Ht)!=="svelte-10b6f1d"&&(Ht.innerHTML=us),No=i(K),Xt=c(K,"P",{"data-svelte-h":!0}),b(Xt)!=="svelte-q52n56"&&(Xt.innerHTML=hs),Jo=i(K),Et=c(K,"P",{"data-svelte-h":!0}),b(Et)!=="svelte-hswkmf"&&(Et.innerHTML=fs),jo=i(K),H=c(K,"DIV",{class:!0});var ye=$(H);h(lt.$$.fragment,ye),zo=i(ye),Qt=c(ye,"P",{"data-svelte-h":!0}),b(Qt)!=="svelte-1gepfo3"&&(Qt.innerHTML=gs),xo=i(ye),h(Me.$$.fragment,ye),Co=i(ye),h(be.$$.fragment,ye),ye.forEach(a),K.forEach(a),Jn=i(e),h(dt.$$.fragment,e),jn=i(e),Yt=c(e,"P",{}),$(Yt).forEach(a),this.h()},h(){v(t,"name","hf:doc:metadata"),v(t,"content",Is),v(oe,"class","flex flex-wrap space-x-1"),v(W,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(S,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(ae,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(re,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(wt,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(J,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(ie,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(q,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(Y,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(j,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(V,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(ee,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(U,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(z,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(L,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(x,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(R,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(C,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(H,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(P,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,o){s(document.head,t),p(e,k,o),p(e,l,o),p(e,m,o),p(e,T,o),p(e,y,o),f(we,e,o),p(e,Ot,o),p(e,oe,o),p(e,Dt,o),f(ve,e,o),p(e,Kt,o),p(e,$e,o),p(e,en,o),p(e,Ne,o),p(e,tn,o),p(e,Je,o),p(e,nn,o),p(e,je,o),p(e,on,o),p(e,ze,o),p(e,sn,o),f(xe,e,o),p(e,an,o),p(e,Ce,o),p(e,rn,o),f(Pe,e,o),p(e,ln,o),p(e,Fe,o),p(e,dn,o),f(Ue,e,o),p(e,cn,o),p(e,W,o),f(We,W,null),s(W,xn),s(W,gt),s(W,Cn),s(W,_t),s(W,Pn),f(se,W,null),p(e,pn,o),f(qe,e,o),p(e,mn,o),p(e,J,o),f(Ie,J,null),s(J,Fn),s(J,Mt),s(J,Un),s(J,S),f(Ze,S,null),s(S,Wn),s(S,bt),s(S,qn),s(S,kt),s(J,In),s(J,ae),f(Be,ae,null),s(ae,Zn),s(ae,Tt),s(J,Bn),s(J,re),f(Ge,re,null),s(re,Gn),s(re,yt),s(J,Vn),s(J,wt),f(Ve,wt,null),p(e,un,o),f(Le,e,o),p(e,hn,o),p(e,q,o),f(Re,q,null),s(q,Ln),s(q,vt),s(q,Rn),s(q,$t),s(q,Hn),s(q,ie),f(He,ie,null),s(ie,Xn),s(ie,Nt),p(e,fn,o),f(Xe,e,o),p(e,gn,o),p(e,j,o),f(Ee,j,null),s(j,En),s(j,Jt),s(j,Qn),s(j,jt),s(j,Sn),s(j,zt),s(j,Yn),s(j,Y),f(Qe,Y,null),s(Y,An),s(Y,xt),s(Y,On),f(le,Y,null),p(e,_n,o),f(Se,e,o),p(e,Mn,o),p(e,ee,o),f(Ye,ee,null),s(ee,Dn),s(ee,V),f(Ae,V,null),s(V,Kn),s(V,Ct),s(V,eo),f(de,V,null),s(V,to),f(ce,V,null),p(e,bn,o),f(Oe,e,o),p(e,kn,o),p(e,z,o),f(De,z,null),s(z,no),s(z,Pt),s(z,oo),s(z,Ft),s(z,so),s(z,Ut),s(z,ao),s(z,U),f(Ke,U,null),s(U,ro),s(U,Wt),s(U,io),f(pe,U,null),s(U,lo),f(me,U,null),s(U,co),f(ue,U,null),p(e,Tn,o),f(et,e,o),p(e,yn,o),p(e,x,o),f(tt,x,null),s(x,po),s(x,qt),s(x,mo),s(x,It),s(x,uo),s(x,Zt),s(x,ho),s(x,L),f(nt,L,null),s(L,fo),s(L,Bt),s(L,go),f(he,L,null),s(L,_o),f(fe,L,null),p(e,wn,o),f(ot,e,o),p(e,vn,o),p(e,C,o),f(st,C,null),s(C,Mo),s(C,Gt),s(C,bo),s(C,Vt),s(C,ko),s(C,Lt),s(C,To),s(C,R),f(at,R,null),s(R,yo),s(R,Rt),s(R,wo),f(ge,R,null),s(R,vo),f(_e,R,null),p(e,$n,o),f(rt,e,o),p(e,Nn,o),p(e,P,o),f(it,P,null),s(P,$o),s(P,Ht),s(P,No),s(P,Xt),s(P,Jo),s(P,Et),s(P,jo),s(P,H),f(lt,H,null),s(H,zo),s(H,Qt),s(H,xo),f(Me,H,null),s(H,Co),f(be,H,null),p(e,Jn,o),f(dt,e,o),p(e,jn,o),p(e,Yt,o),zn=!0},p(e,[o]){const X={};o&2&&(X.$$scope={dirty:o,ctx:e}),se.$set(X);const F={};o&2&&(F.$$scope={dirty:o,ctx:e}),le.$set(F);const te={};o&2&&(te.$$scope={dirty:o,ctx:e}),de.$set(te);const ct={};o&2&&(ct.$$scope={dirty:o,ctx:e}),ce.$set(ct);const pt={};o&2&&(pt.$$scope={dirty:o,ctx:e}),pe.$set(pt);const At={};o&2&&(At.$$scope={dirty:o,ctx:e}),me.$set(At);const E={};o&2&&(E.$$scope={dirty:o,ctx:e}),ue.$set(E);const mt={};o&2&&(mt.$$scope={dirty:o,ctx:e}),he.$set(mt);const I={};o&2&&(I.$$scope={dirty:o,ctx:e}),fe.$set(I);const ne={};o&2&&(ne.$$scope={dirty:o,ctx:e}),ge.$set(ne);const ut={};o&2&&(ut.$$scope={dirty:o,ctx:e}),_e.$set(ut);const Q={};o&2&&(Q.$$scope={dirty:o,ctx:e}),Me.$set(Q);const Z={};o&2&&(Z.$$scope={dirty:o,ctx:e}),be.$set(Z)},i(e){zn||(g(we.$$.fragment,e),g(ve.$$.fragment,e),g(xe.$$.fragment,e),g(Pe.$$.fragment,e),g(Ue.$$.fragment,e),g(We.$$.fragment,e),g(se.$$.fragment,e),g(qe.$$.fragment,e),g(Ie.$$.fragment,e),g(Ze.$$.fragment,e),g(Be.$$.fragment,e),g(Ge.$$.fragment,e),g(Ve.$$.fragment,e),g(Le.$$.fragment,e),g(Re.$$.fragment,e),g(He.$$.fragment,e),g(Xe.$$.fragment,e),g(Ee.$$.fragment,e),g(Qe.$$.fragment,e),g(le.$$.fragment,e),g(Se.$$.fragment,e),g(Ye.$$.fragment,e),g(Ae.$$.fragment,e),g(de.$$.fragment,e),g(ce.$$.fragment,e),g(Oe.$$.fragment,e),g(De.$$.fragment,e),g(Ke.$$.fragment,e),g(pe.$$.fragment,e),g(me.$$.fragment,e),g(ue.$$.fragment,e),g(et.$$.fragment,e),g(tt.$$.fragment,e),g(nt.$$.fragment,e),g(he.$$.fragment,e),g(fe.$$.fragment,e),g(ot.$$.fragment,e),g(st.$$.fragment,e),g(at.$$.fragment,e),g(ge.$$.fragment,e),g(_e.$$.fragment,e),g(rt.$$.fragment,e),g(it.$$.fragment,e),g(lt.$$.fragment,e),g(Me.$$.fragment,e),g(be.$$.fragment,e),g(dt.$$.fragment,e),zn=!0)},o(e){_(we.$$.fragment,e),_(ve.$$.fragment,e),_(xe.$$.fragment,e),_(Pe.$$.fragment,e),_(Ue.$$.fragment,e),_(We.$$.fragment,e),_(se.$$.fragment,e),_(qe.$$.fragment,e),_(Ie.$$.fragment,e),_(Ze.$$.fragment,e),_(Be.$$.fragment,e),_(Ge.$$.fragment,e),_(Ve.$$.fragment,e),_(Le.$$.fragment,e),_(Re.$$.fragment,e),_(He.$$.fragment,e),_(Xe.$$.fragment,e),_(Ee.$$.fragment,e),_(Qe.$$.fragment,e),_(le.$$.fragment,e),_(Se.$$.fragment,e),_(Ye.$$.fragment,e),_(Ae.$$.fragment,e),_(de.$$.fragment,e),_(ce.$$.fragment,e),_(Oe.$$.fragment,e),_(De.$$.fragment,e),_(Ke.$$.fragment,e),_(pe.$$.fragment,e),_(me.$$.fragment,e),_(ue.$$.fragment,e),_(et.$$.fragment,e),_(tt.$$.fragment,e),_(nt.$$.fragment,e),_(he.$$.fragment,e),_(fe.$$.fragment,e),_(ot.$$.fragment,e),_(st.$$.fragment,e),_(at.$$.fragment,e),_(ge.$$.fragment,e),_(_e.$$.fragment,e),_(rt.$$.fragment,e),_(it.$$.fragment,e),_(lt.$$.fragment,e),_(Me.$$.fragment,e),_(be.$$.fragment,e),_(dt.$$.fragment,e),zn=!1},d(e){e&&(a(k),a(l),a(m),a(T),a(y),a(Ot),a(oe),a(Dt),a(Kt),a($e),a(en),a(Ne),a(tn),a(Je),a(nn),a(je),a(on),a(ze),a(sn),a(an),a(Ce),a(rn),a(ln),a(Fe),a(dn),a(cn),a(W),a(pn),a(mn),a(J),a(un),a(hn),a(q),a(fn),a(gn),a(j),a(_n),a(Mn),a(ee),a(bn),a(kn),a(z),a(Tn),a(yn),a(x),a(wn),a(vn),a(C),a($n),a(Nn),a(P),a(Jn),a(jn),a(Yt)),a(t),M(we,e),M(ve,e),M(xe,e),M(Pe,e),M(Ue,e),M(We),M(se),M(qe,e),M(Ie),M(Ze),M(Be),M(Ge),M(Ve),M(Le,e),M(Re),M(He),M(Xe,e),M(Ee),M(Qe),M(le),M(Se,e),M(Ye),M(Ae),M(de),M(ce),M(Oe,e),M(De),M(Ke),M(pe),M(me),M(ue),M(et,e),M(tt),M(nt),M(he),M(fe),M(ot,e),M(st),M(at),M(ge),M(_e),M(rt,e),M(it),M(lt),M(Me),M(be),M(dt,e)}}}const Is='{"title":"MPNet","local":"mpnet","sections":[{"title":"Overview","local":"overview","sections":[],"depth":2},{"title":"Usage tips","local":"usage-tips","sections":[],"depth":2},{"title":"Resources","local":"resources","sections":[],"depth":2},{"title":"MPNetConfig","local":"transformers.MPNetConfig","sections":[],"depth":2},{"title":"MPNetTokenizer","local":"transformers.MPNetTokenizer","sections":[],"depth":2},{"title":"MPNetTokenizerFast","local":"transformers.MPNetTokenizerFast","sections":[],"depth":2},{"title":"MPNetModel","local":"transformers.MPNetModel","sections":[],"depth":2},{"title":"MPNetForMaskedLM","local":"transformers.MPNetForMaskedLM","sections":[],"depth":2},{"title":"MPNetForSequenceClassification","local":"transformers.MPNetForSequenceClassification","sections":[],"depth":2},{"title":"MPNetForMultipleChoice","local":"transformers.MPNetForMultipleChoice","sections":[],"depth":2},{"title":"MPNetForTokenClassification","local":"transformers.MPNetForTokenClassification","sections":[],"depth":2},{"title":"MPNetForQuestionAnswering","local":"transformers.MPNetForQuestionAnswering","sections":[],"depth":2}],"depth":1}';function Zs(w){return Ms(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class Es extends bs{constructor(t){super(),ks(this,t,Zs,qs,_s,{})}}export{Es as component};
