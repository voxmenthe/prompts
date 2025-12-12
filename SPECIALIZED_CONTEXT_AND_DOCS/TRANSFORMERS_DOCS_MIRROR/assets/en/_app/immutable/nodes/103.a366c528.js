import{s as qn,o as Nn,n as R}from"../chunks/scheduler.18a86fab.js";import{S as Zn,i as Ln,g as p,s as a,r as h,A as Gn,h as m,f as s,c as r,j as W,x as u,u as f,k as F,y as l,a as c,v as g,d as _,t as b,w as y}from"../chunks/index.98837b22.js";import{T as Mt}from"../chunks/Tip.77304350.js";import{D as S}from"../chunks/Docstring.a1ef7999.js";import{C as Ee}from"../chunks/CodeBlock.8d0c2e8a.js";import{E as Pe}from"../chunks/ExampleCodeBlock.8c3ee1f9.js";import{P as Rn}from"../chunks/PipelineTag.7749150e.js";import{H as O,E as Vn}from"../chunks/getInferenceSnippets.06c2775f.js";function Hn(M){let t,T="Example:",i,d,v;return d=new Ee({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEJsb29tQ29uZmlnJTJDJTIwQmxvb21Nb2RlbCUwQSUwQSUyMyUyMEluaXRpYWxpemluZyUyMGElMjBCbG9vbSUyMGNvbmZpZ3VyYXRpb24lMEFjb25maWd1cmF0aW9uJTIwJTNEJTIwQmxvb21Db25maWcoKSUwQSUwQSUyMyUyMEluaXRpYWxpemluZyUyMGElMjBtb2RlbCUyMCh3aXRoJTIwcmFuZG9tJTIwd2VpZ2h0cyklMjBmcm9tJTIwdGhlJTIwY29uZmlndXJhdGlvbiUwQW1vZGVsJTIwJTNEJTIwQmxvb21Nb2RlbChjb25maWd1cmF0aW9uKSUwQSUwQSUyMyUyMEFjY2Vzc2luZyUyMHRoZSUyMG1vZGVsJTIwY29uZmlndXJhdGlvbiUwQWNvbmZpZ3VyYXRpb24lMjAlM0QlMjBtb2RlbC5jb25maWc=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> BloomConfig, BloomModel

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a Bloom configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = BloomConfig()

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a model (with random weights) from the configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = BloomModel(configuration)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Accessing the model configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = model.config`,wrap:!1}}),{c(){t=p("p"),t.textContent=T,i=a(),h(d.$$.fragment)},l(o){t=m(o,"P",{"data-svelte-h":!0}),u(t)!=="svelte-11lpom8"&&(t.textContent=T),i=r(o),f(d.$$.fragment,o)},m(o,w){c(o,t,w),c(o,i,w),g(d,o,w),v=!0},p:R,i(o){v||(_(d.$$.fragment,o),v=!0)},o(o){b(d.$$.fragment,o),v=!1},d(o){o&&(s(t),s(i)),y(d,o)}}}function Qn(M){let t,T="be encoded differently whether it is at the beginning of the sentence (without space) or not:",i,d,v;return d=new Ee({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEJsb29tVG9rZW5pemVyRmFzdCUwQSUwQXRva2VuaXplciUyMCUzRCUyMEJsb29tVG9rZW5pemVyRmFzdC5mcm9tX3ByZXRyYWluZWQoJTIyYmlnc2NpZW5jZSUyRmJsb29tJTIyKSUwQXRva2VuaXplciglMjJIZWxsbyUyMHdvcmxkJTIyKSU1QiUyMmlucHV0X2lkcyUyMiU1RCUwQSUwQXRva2VuaXplciglMjIlMjBIZWxsbyUyMHdvcmxkJTIyKSU1QiUyMmlucHV0X2lkcyUyMiU1RA==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> BloomTokenizerFast

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = BloomTokenizerFast.from_pretrained(<span class="hljs-string">&quot;bigscience/bloom&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer(<span class="hljs-string">&quot;Hello world&quot;</span>)[<span class="hljs-string">&quot;input_ids&quot;</span>]
[<span class="hljs-number">59414</span>, <span class="hljs-number">8876</span>]

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer(<span class="hljs-string">&quot; Hello world&quot;</span>)[<span class="hljs-string">&quot;input_ids&quot;</span>]
[<span class="hljs-number">86153</span>, <span class="hljs-number">8876</span>]`,wrap:!1}}),{c(){t=p("p"),t.textContent=T,i=a(),h(d.$$.fragment)},l(o){t=m(o,"P",{"data-svelte-h":!0}),u(t)!=="svelte-12atnao"&&(t.textContent=T),i=r(o),f(d.$$.fragment,o)},m(o,w){c(o,t,w),c(o,i,w),g(d,o,w),v=!0},p:R,i(o){v||(_(d.$$.fragment,o),v=!0)},o(o){b(d.$$.fragment,o),v=!1},d(o){o&&(s(t),s(i)),y(d,o)}}}function Sn(M){let t,T="When used with <code>is_split_into_words=True</code>, this tokenizer needs to be instantiated with <code>add_prefix_space=True</code>.";return{c(){t=p("p"),t.innerHTML=T},l(i){t=m(i,"P",{"data-svelte-h":!0}),u(t)!=="svelte-9gg91e"&&(t.innerHTML=T)},m(i,d){c(i,t,d)},p:R,d(i){i&&s(t)}}}function Xn(M){let t,T=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=p("p"),t.innerHTML=T},l(i){t=m(i,"P",{"data-svelte-h":!0}),u(t)!=="svelte-fincs2"&&(t.innerHTML=T)},m(i,d){c(i,t,d)},p:R,d(i){i&&s(t)}}}function Pn(M){let t,T=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=p("p"),t.innerHTML=T},l(i){t=m(i,"P",{"data-svelte-h":!0}),u(t)!=="svelte-fincs2"&&(t.innerHTML=T)},m(i,d){c(i,t,d)},p:R,d(i){i&&s(t)}}}function En(M){let t,T="Example:",i,d,v;return d=new Ee({props:{code:"",highlighted:"",wrap:!1}}),{c(){t=p("p"),t.textContent=T,i=a(),h(d.$$.fragment)},l(o){t=m(o,"P",{"data-svelte-h":!0}),u(t)!=="svelte-11lpom8"&&(t.textContent=T),i=r(o),f(d.$$.fragment,o)},m(o,w){c(o,t,w),c(o,i,w),g(d,o,w),v=!0},p:R,i(o){v||(_(d.$$.fragment,o),v=!0)},o(o){b(d.$$.fragment,o),v=!1},d(o){o&&(s(t),s(i)),y(d,o)}}}function An(M){let t,T=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=p("p"),t.innerHTML=T},l(i){t=m(i,"P",{"data-svelte-h":!0}),u(t)!=="svelte-fincs2"&&(t.innerHTML=T)},m(i,d){c(i,t,d)},p:R,d(i){i&&s(t)}}}function Yn(M){let t,T="Example of single-label classification:",i,d,v;return d=new Ee({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Rva2VuaXplciUyQyUyMEJsb29tRm9yU2VxdWVuY2VDbGFzc2lmaWNhdGlvbiUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMmJpZ3NjaWVuY2UlMkZibG9vbSUyMiklMEFtb2RlbCUyMCUzRCUyMEJsb29tRm9yU2VxdWVuY2VDbGFzc2lmaWNhdGlvbi5mcm9tX3ByZXRyYWluZWQoJTIyYmlnc2NpZW5jZSUyRmJsb29tJTIyKSUwQSUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplciglMjJIZWxsbyUyQyUyMG15JTIwZG9nJTIwaXMlMjBjdXRlJTIyJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiklMEElMEF3aXRoJTIwdG9yY2gubm9fZ3JhZCgpJTNBJTBBJTIwJTIwJTIwJTIwbG9naXRzJTIwJTNEJTIwbW9kZWwoKippbnB1dHMpLmxvZ2l0cyUwQSUwQXByZWRpY3RlZF9jbGFzc19pZCUyMCUzRCUyMGxvZ2l0cy5hcmdtYXgoKS5pdGVtKCklMEFtb2RlbC5jb25maWcuaWQybGFiZWwlNUJwcmVkaWN0ZWRfY2xhc3NfaWQlNUQlMEElMEElMjMlMjBUbyUyMHRyYWluJTIwYSUyMG1vZGVsJTIwb24lMjAlNjBudW1fbGFiZWxzJTYwJTIwY2xhc3NlcyUyQyUyMHlvdSUyMGNhbiUyMHBhc3MlMjAlNjBudW1fbGFiZWxzJTNEbnVtX2xhYmVscyU2MCUyMHRvJTIwJTYwLmZyb21fcHJldHJhaW5lZCguLi4pJTYwJTBBbnVtX2xhYmVscyUyMCUzRCUyMGxlbihtb2RlbC5jb25maWcuaWQybGFiZWwpJTBBbW9kZWwlMjAlM0QlMjBCbG9vbUZvclNlcXVlbmNlQ2xhc3NpZmljYXRpb24uZnJvbV9wcmV0cmFpbmVkKCUyMmJpZ3NjaWVuY2UlMkZibG9vbSUyMiUyQyUyMG51bV9sYWJlbHMlM0RudW1fbGFiZWxzKSUwQSUwQWxhYmVscyUyMCUzRCUyMHRvcmNoLnRlbnNvciglNUIxJTVEKSUwQWxvc3MlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyUyQyUyMGxhYmVscyUzRGxhYmVscykubG9zcyUwQXJvdW5kKGxvc3MuaXRlbSgpJTJDJTIwMik=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, BloomForSequenceClassification

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;bigscience/bloom&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = BloomForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;bigscience/bloom&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    logits = model(**inputs).logits

<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_class_id = logits.argmax().item()
<span class="hljs-meta">&gt;&gt;&gt; </span>model.config.id2label[predicted_class_id]
...

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># To train a model on \`num_labels\` classes, you can pass \`num_labels=num_labels\` to \`.from_pretrained(...)\`</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>num_labels = <span class="hljs-built_in">len</span>(model.config.id2label)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = BloomForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;bigscience/bloom&quot;</span>, num_labels=num_labels)

<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.tensor([<span class="hljs-number">1</span>])
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = model(**inputs, labels=labels).loss
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">round</span>(loss.item(), <span class="hljs-number">2</span>)
...`,wrap:!1}}),{c(){t=p("p"),t.textContent=T,i=a(),h(d.$$.fragment)},l(o){t=m(o,"P",{"data-svelte-h":!0}),u(t)!=="svelte-ykxpe4"&&(t.textContent=T),i=r(o),f(d.$$.fragment,o)},m(o,w){c(o,t,w),c(o,i,w),g(d,o,w),v=!0},p:R,i(o){v||(_(d.$$.fragment,o),v=!0)},o(o){b(d.$$.fragment,o),v=!1},d(o){o&&(s(t),s(i)),y(d,o)}}}function On(M){let t,T="Example of multi-label classification:",i,d,v;return d=new Ee({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Rva2VuaXplciUyQyUyMEJsb29tRm9yU2VxdWVuY2VDbGFzc2lmaWNhdGlvbiUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMmJpZ3NjaWVuY2UlMkZibG9vbSUyMiklMEFtb2RlbCUyMCUzRCUyMEJsb29tRm9yU2VxdWVuY2VDbGFzc2lmaWNhdGlvbi5mcm9tX3ByZXRyYWluZWQoJTIyYmlnc2NpZW5jZSUyRmJsb29tJTIyJTJDJTIwcHJvYmxlbV90eXBlJTNEJTIybXVsdGlfbGFiZWxfY2xhc3NpZmljYXRpb24lMjIpJTBBJTBBaW5wdXRzJTIwJTNEJTIwdG9rZW5pemVyKCUyMkhlbGxvJTJDJTIwbXklMjBkb2clMjBpcyUyMGN1dGUlMjIlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQSUwQXdpdGglMjB0b3JjaC5ub19ncmFkKCklM0ElMEElMjAlMjAlMjAlMjBsb2dpdHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cykubG9naXRzJTBBJTBBcHJlZGljdGVkX2NsYXNzX2lkcyUyMCUzRCUyMHRvcmNoLmFyYW5nZSgwJTJDJTIwbG9naXRzLnNoYXBlJTVCLTElNUQpJTVCdG9yY2guc2lnbW9pZChsb2dpdHMpLnNxdWVlemUoZGltJTNEMCklMjAlM0UlMjAwLjUlNUQlMEElMEElMjMlMjBUbyUyMHRyYWluJTIwYSUyMG1vZGVsJTIwb24lMjAlNjBudW1fbGFiZWxzJTYwJTIwY2xhc3NlcyUyQyUyMHlvdSUyMGNhbiUyMHBhc3MlMjAlNjBudW1fbGFiZWxzJTNEbnVtX2xhYmVscyU2MCUyMHRvJTIwJTYwLmZyb21fcHJldHJhaW5lZCguLi4pJTYwJTBBbnVtX2xhYmVscyUyMCUzRCUyMGxlbihtb2RlbC5jb25maWcuaWQybGFiZWwpJTBBbW9kZWwlMjAlM0QlMjBCbG9vbUZvclNlcXVlbmNlQ2xhc3NpZmljYXRpb24uZnJvbV9wcmV0cmFpbmVkKCUwQSUyMCUyMCUyMCUyMCUyMmJpZ3NjaWVuY2UlMkZibG9vbSUyMiUyQyUyMG51bV9sYWJlbHMlM0RudW1fbGFiZWxzJTJDJTIwcHJvYmxlbV90eXBlJTNEJTIybXVsdGlfbGFiZWxfY2xhc3NpZmljYXRpb24lMjIlMEEpJTBBJTBBbGFiZWxzJTIwJTNEJTIwdG9yY2guc3VtKCUwQSUyMCUyMCUyMCUyMHRvcmNoLm5uLmZ1bmN0aW9uYWwub25lX2hvdChwcmVkaWN0ZWRfY2xhc3NfaWRzJTVCTm9uZSUyQyUyMCUzQSU1RC5jbG9uZSgpJTJDJTIwbnVtX2NsYXNzZXMlM0RudW1fbGFiZWxzKSUyQyUyMGRpbSUzRDElMEEpLnRvKHRvcmNoLmZsb2F0KSUwQWxvc3MlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyUyQyUyMGxhYmVscyUzRGxhYmVscykubG9zcw==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, BloomForSequenceClassification

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;bigscience/bloom&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = BloomForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;bigscience/bloom&quot;</span>, problem_type=<span class="hljs-string">&quot;multi_label_classification&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    logits = model(**inputs).logits

<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_class_ids = torch.arange(<span class="hljs-number">0</span>, logits.shape[-<span class="hljs-number">1</span>])[torch.sigmoid(logits).squeeze(dim=<span class="hljs-number">0</span>) &gt; <span class="hljs-number">0.5</span>]

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># To train a model on \`num_labels\` classes, you can pass \`num_labels=num_labels\` to \`.from_pretrained(...)\`</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>num_labels = <span class="hljs-built_in">len</span>(model.config.id2label)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = BloomForSequenceClassification.from_pretrained(
<span class="hljs-meta">... </span>    <span class="hljs-string">&quot;bigscience/bloom&quot;</span>, num_labels=num_labels, problem_type=<span class="hljs-string">&quot;multi_label_classification&quot;</span>
<span class="hljs-meta">... </span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.<span class="hljs-built_in">sum</span>(
<span class="hljs-meta">... </span>    torch.nn.functional.one_hot(predicted_class_ids[<span class="hljs-literal">None</span>, :].clone(), num_classes=num_labels), dim=<span class="hljs-number">1</span>
<span class="hljs-meta">... </span>).to(torch.<span class="hljs-built_in">float</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = model(**inputs, labels=labels).loss`,wrap:!1}}),{c(){t=p("p"),t.textContent=T,i=a(),h(d.$$.fragment)},l(o){t=m(o,"P",{"data-svelte-h":!0}),u(t)!=="svelte-1l8e32d"&&(t.textContent=T),i=r(o),f(d.$$.fragment,o)},m(o,w){c(o,t,w),c(o,i,w),g(d,o,w),v=!0},p:R,i(o){v||(_(d.$$.fragment,o),v=!0)},o(o){b(d.$$.fragment,o),v=!1},d(o){o&&(s(t),s(i)),y(d,o)}}}function Dn(M){let t,T=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=p("p"),t.innerHTML=T},l(i){t=m(i,"P",{"data-svelte-h":!0}),u(t)!=="svelte-fincs2"&&(t.innerHTML=T)},m(i,d){c(i,t,d)},p:R,d(i){i&&s(t)}}}function Kn(M){let t,T="Example:",i,d,v;return d=new Ee({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBCbG9vbUZvclRva2VuQ2xhc3NpZmljYXRpb24lMEFpbXBvcnQlMjB0b3JjaCUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMmJpZ3NjaWVuY2UlMkZibG9vbSUyMiklMEFtb2RlbCUyMCUzRCUyMEJsb29tRm9yVG9rZW5DbGFzc2lmaWNhdGlvbi5mcm9tX3ByZXRyYWluZWQoJTIyYmlnc2NpZW5jZSUyRmJsb29tJTIyKSUwQSUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplciglMEElMjAlMjAlMjAlMjAlMjJIdWdnaW5nRmFjZSUyMGlzJTIwYSUyMGNvbXBhbnklMjBiYXNlZCUyMGluJTIwUGFyaXMlMjBhbmQlMjBOZXclMjBZb3JrJTIyJTJDJTIwYWRkX3NwZWNpYWxfdG9rZW5zJTNERmFsc2UlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyJTBBKSUwQSUwQXdpdGglMjB0b3JjaC5ub19ncmFkKCklM0ElMEElMjAlMjAlMjAlMjBsb2dpdHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cykubG9naXRzJTBBJTBBcHJlZGljdGVkX3Rva2VuX2NsYXNzX2lkcyUyMCUzRCUyMGxvZ2l0cy5hcmdtYXgoLTEpJTBBJTBBJTIzJTIwTm90ZSUyMHRoYXQlMjB0b2tlbnMlMjBhcmUlMjBjbGFzc2lmaWVkJTIwcmF0aGVyJTIwdGhlbiUyMGlucHV0JTIwd29yZHMlMjB3aGljaCUyMG1lYW5zJTIwdGhhdCUwQSUyMyUyMHRoZXJlJTIwbWlnaHQlMjBiZSUyMG1vcmUlMjBwcmVkaWN0ZWQlMjB0b2tlbiUyMGNsYXNzZXMlMjB0aGFuJTIwd29yZHMuJTBBJTIzJTIwTXVsdGlwbGUlMjB0b2tlbiUyMGNsYXNzZXMlMjBtaWdodCUyMGFjY291bnQlMjBmb3IlMjB0aGUlMjBzYW1lJTIwd29yZCUwQXByZWRpY3RlZF90b2tlbnNfY2xhc3NlcyUyMCUzRCUyMCU1Qm1vZGVsLmNvbmZpZy5pZDJsYWJlbCU1QnQuaXRlbSgpJTVEJTIwZm9yJTIwdCUyMGluJTIwcHJlZGljdGVkX3Rva2VuX2NsYXNzX2lkcyU1QjAlNUQlNUQlMEFwcmVkaWN0ZWRfdG9rZW5zX2NsYXNzZXMlMEElMEFsYWJlbHMlMjAlM0QlMjBwcmVkaWN0ZWRfdG9rZW5fY2xhc3NfaWRzJTBBbG9zcyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzJTJDJTIwbGFiZWxzJTNEbGFiZWxzKS5sb3NzJTBBcm91bmQobG9zcy5pdGVtKCklMkMlMjAyKQ==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, BloomForTokenClassification
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;bigscience/bloom&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = BloomForTokenClassification.from_pretrained(<span class="hljs-string">&quot;bigscience/bloom&quot;</span>)

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
...`,wrap:!1}}),{c(){t=p("p"),t.textContent=T,i=a(),h(d.$$.fragment)},l(o){t=m(o,"P",{"data-svelte-h":!0}),u(t)!=="svelte-11lpom8"&&(t.textContent=T),i=r(o),f(d.$$.fragment,o)},m(o,w){c(o,t,w),c(o,i,w),g(d,o,w),v=!0},p:R,i(o){v||(_(d.$$.fragment,o),v=!0)},o(o){b(d.$$.fragment,o),v=!1},d(o){o&&(s(t),s(i)),y(d,o)}}}function es(M){let t,T=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=p("p"),t.innerHTML=T},l(i){t=m(i,"P",{"data-svelte-h":!0}),u(t)!=="svelte-fincs2"&&(t.innerHTML=T)},m(i,d){c(i,t,d)},p:R,d(i){i&&s(t)}}}function ts(M){let t,T="Example:",i,d,v;return d=new Ee({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBCbG9vbUZvclF1ZXN0aW9uQW5zd2VyaW5nJTBBaW1wb3J0JTIwdG9yY2glMEElMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJiaWdzY2llbmNlJTJGYmxvb20lMjIpJTBBbW9kZWwlMjAlM0QlMjBCbG9vbUZvclF1ZXN0aW9uQW5zd2VyaW5nLmZyb21fcHJldHJhaW5lZCglMjJiaWdzY2llbmNlJTJGYmxvb20lMjIpJTBBJTBBcXVlc3Rpb24lMkMlMjB0ZXh0JTIwJTNEJTIwJTIyV2hvJTIwd2FzJTIwSmltJTIwSGVuc29uJTNGJTIyJTJDJTIwJTIySmltJTIwSGVuc29uJTIwd2FzJTIwYSUyMG5pY2UlMjBwdXBwZXQlMjIlMEElMEFpbnB1dHMlMjAlM0QlMjB0b2tlbml6ZXIocXVlc3Rpb24lMkMlMjB0ZXh0JTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiklMEF3aXRoJTIwdG9yY2gubm9fZ3JhZCgpJTNBJTBBJTIwJTIwJTIwJTIwb3V0cHV0cyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzKSUwQSUwQWFuc3dlcl9zdGFydF9pbmRleCUyMCUzRCUyMG91dHB1dHMuc3RhcnRfbG9naXRzLmFyZ21heCgpJTBBYW5zd2VyX2VuZF9pbmRleCUyMCUzRCUyMG91dHB1dHMuZW5kX2xvZ2l0cy5hcmdtYXgoKSUwQSUwQXByZWRpY3RfYW5zd2VyX3Rva2VucyUyMCUzRCUyMGlucHV0cy5pbnB1dF9pZHMlNUIwJTJDJTIwYW5zd2VyX3N0YXJ0X2luZGV4JTIwJTNBJTIwYW5zd2VyX2VuZF9pbmRleCUyMCUyQiUyMDElNUQlMEF0b2tlbml6ZXIuZGVjb2RlKHByZWRpY3RfYW5zd2VyX3Rva2VucyUyQyUyMHNraXBfc3BlY2lhbF90b2tlbnMlM0RUcnVlKSUwQSUwQSUyMyUyMHRhcmdldCUyMGlzJTIwJTIybmljZSUyMHB1cHBldCUyMiUwQXRhcmdldF9zdGFydF9pbmRleCUyMCUzRCUyMHRvcmNoLnRlbnNvciglNUIxNCU1RCklMEF0YXJnZXRfZW5kX2luZGV4JTIwJTNEJTIwdG9yY2gudGVuc29yKCU1QjE1JTVEKSUwQSUwQW91dHB1dHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyUyQyUyMHN0YXJ0X3Bvc2l0aW9ucyUzRHRhcmdldF9zdGFydF9pbmRleCUyQyUyMGVuZF9wb3NpdGlvbnMlM0R0YXJnZXRfZW5kX2luZGV4KSUwQWxvc3MlMjAlM0QlMjBvdXRwdXRzLmxvc3MlMEFyb3VuZChsb3NzLml0ZW0oKSUyQyUyMDIp",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, BloomForQuestionAnswering
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;bigscience/bloom&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = BloomForQuestionAnswering.from_pretrained(<span class="hljs-string">&quot;bigscience/bloom&quot;</span>)

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
...`,wrap:!1}}),{c(){t=p("p"),t.textContent=T,i=a(),h(d.$$.fragment)},l(o){t=m(o,"P",{"data-svelte-h":!0}),u(t)!=="svelte-11lpom8"&&(t.textContent=T),i=r(o),f(d.$$.fragment,o)},m(o,w){c(o,t,w),c(o,i,w),g(d,o,w),v=!0},p:R,i(o){v||(_(d.$$.fragment,o),v=!0)},o(o){b(d.$$.fragment,o),v=!1},d(o){o&&(s(t),s(i)),y(d,o)}}}function os(M){let t,T,i,d,v,o="<em>This model was released on 2022-11-09 and added to Hugging Face Transformers on 2022-06-09.</em>",w,ue,$t,K,Ao='<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/>',Ct,he,Bt,fe,Yo=`The <a href="https://huggingface.co/papers/2211.05100" rel="nofollow">BLOOM</a> model has been proposed with its various versions through the <a href="https://bigscience.huggingface.co/" rel="nofollow">BigScience Workshop</a>. BigScience is inspired by other open science initiatives where researchers have pooled their time and resources to collectively achieve a higher impact.
The architecture of BLOOM is essentially similar to GPT3 (auto-regressive model for next token prediction), but has been trained on 46 different languages and 13 programming languages.
Several smaller versions of the models have been trained on the same dataset. BLOOM is available in the following versions:`,xt,ge,Oo='<li><a href="https://huggingface.co/bigscience/bloom-560m" rel="nofollow">bloom-560m</a></li> <li><a href="https://huggingface.co/bigscience/bloom-1b1" rel="nofollow">bloom-1b1</a></li> <li><a href="https://huggingface.co/bigscience/bloom-1b7" rel="nofollow">bloom-1b7</a></li> <li><a href="https://huggingface.co/bigscience/bloom-3b" rel="nofollow">bloom-3b</a></li> <li><a href="https://huggingface.co/bigscience/bloom-7b1" rel="nofollow">bloom-7b1</a></li> <li><a href="https://huggingface.co/bigscience/bloom" rel="nofollow">bloom</a> (176B parameters)</li>',Ut,_e,jt,be,Do="A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with BLOOM. If you’re interested in submitting a resource to be included here, please feel free to open a Pull Request and we’ll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.",zt,ye,Ft,Te,Ko='<li><a href="/docs/transformers/v4.56.2/en/model_doc/bloom#transformers.BloomForCausalLM">BloomForCausalLM</a> is supported by this <a href="https://github.com/huggingface/transformers/tree/main/examples/pytorch/language-modeling#gpt-2gpt-and-causal-language-modeling" rel="nofollow">causal language modeling example script</a> and <a href="https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling.ipynb" rel="nofollow">notebook</a>.</li>',Jt,ve,en="See also:",Wt,we,tn='<li><a href="../tasks/language_modeling">Causal language modeling task guide</a></li> <li><a href="../tasks/sequence_classification">Text classification task guide</a></li> <li><a href="../tasks/token_classification">Token classification task guide</a></li> <li><a href="../tasks/question_answering">Question answering task guide</a></li>',It,Me,on="⚡️ Inference",qt,ke,nn='<li>A blog on <a href="https://huggingface.co/blog/bloom-inference-optimization" rel="nofollow">Optimization story: Bloom inference</a>.</li> <li>A blog on <a href="https://huggingface.co/blog/bloom-inference-pytorch-scripts" rel="nofollow">Incredibly Fast BLOOM Inference with DeepSpeed and Accelerate</a>.</li>',Nt,$e,sn="⚙️ Training",Zt,Ce,an='<li>A blog on <a href="https://huggingface.co/blog/bloom-megatron-deepspeed" rel="nofollow">The Technology Behind BLOOM Training</a>.</li>',Lt,Be,Gt,I,xe,oo,Ae,rn=`This is the configuration class to store the configuration of a <a href="/docs/transformers/v4.56.2/en/model_doc/bloom#transformers.BloomModel">BloomModel</a>. It is used to instantiate a Bloom
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to the Bloom architecture
<a href="https://huggingface.co/bigscience/bloom" rel="nofollow">bigscience/bloom</a>.`,no,Ye,ln=`Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> and can be used to control the model outputs. Read the
documentation from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> for more information.`,so,ee,Rt,Ue,Vt,k,je,ao,Oe,cn=`Construct a “fast” Bloom tokenizer (backed by HuggingFace’s <em>tokenizers</em> library). Based on byte-level
Byte-Pair-Encoding.`,ro,De,dn="This tokenizer has been trained to treat spaces like parts of the tokens (a bit like sentencepiece) so a word will",lo,te,io,Ke,pn=`You can get around that behavior by passing <code>add_prefix_space=True</code> when instantiating this tokenizer, but since
the model was not pretrained this way, it might yield a decrease in performance.`,co,oe,po,et,mn=`This tokenizer inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast">PreTrainedTokenizerFast</a> which contains most of the main methods. Users should
refer to this superclass for more information regarding those methods.`,Ht,ze,Qt,x,Fe,mo,tt,un="The bare Bloom Model outputting raw hidden-states without any specific head on top.",uo,ot,hn=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,ho,nt,fn=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,fo,Y,Je,go,st,gn='The <a href="/docs/transformers/v4.56.2/en/model_doc/bloom#transformers.BloomModel">BloomModel</a> forward method, overrides the <code>__call__</code> special method.',_o,ne,St,We,Xt,U,Ie,bo,at,_n=`The Bloom Model transformer with a language modeling head on top (linear layer with weights tied to the input
embeddings).`,yo,rt,bn=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,To,lt,yn=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,vo,V,qe,wo,it,Tn='The <a href="/docs/transformers/v4.56.2/en/model_doc/bloom#transformers.BloomForCausalLM">BloomForCausalLM</a> forward method, overrides the <code>__call__</code> special method.',Mo,se,ko,ae,Pt,Ne,Et,$,Ze,$o,ct,vn="The Bloom Model transformer with a sequence classification head on top (linear layer).",Co,dt,wn=`<a href="/docs/transformers/v4.56.2/en/model_doc/bloom#transformers.BloomForSequenceClassification">BloomForSequenceClassification</a> uses the last token in order to do the classification, as other causal models
(e.g. GPT-1) do.`,Bo,pt,Mn=`Since it does classification on the last token, it requires to know the position of the last token. If a
<code>pad_token_id</code> is defined in the configuration, it finds the last token that is not a padding token in each row. If
no <code>pad_token_id</code> is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
padding tokens when <code>inputs_embeds</code> are passed instead of <code>input_ids</code>, it does the same (take the last value in
each row of the batch).`,xo,mt,kn=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Uo,ut,$n=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,jo,J,Le,zo,ht,Cn='The <a href="/docs/transformers/v4.56.2/en/model_doc/bloom#transformers.BloomForSequenceClassification">BloomForSequenceClassification</a> forward method, overrides the <code>__call__</code> special method.',Fo,re,Jo,le,Wo,ie,At,Ge,Yt,j,Re,Io,ft,Bn=`The Bloom transformer with a token classification head on top (a linear layer on top of the hidden-states
output) e.g. for Named-Entity-Recognition (NER) tasks.`,qo,gt,xn=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,No,_t,Un=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Zo,H,Ve,Lo,bt,jn='The <a href="/docs/transformers/v4.56.2/en/model_doc/bloom#transformers.BloomForTokenClassification">BloomForTokenClassification</a> forward method, overrides the <code>__call__</code> special method.',Go,ce,Ro,de,Ot,He,Dt,z,Qe,Vo,yt,zn=`The Bloom transformer with a span classification head on top for extractive question-answering tasks like
SQuAD (a linear layer on top of the hidden-states output to compute <code>span start logits</code> and <code>span end logits</code>).`,Ho,Tt,Fn=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Qo,vt,Jn=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,So,Q,Se,Xo,wt,Wn='The <a href="/docs/transformers/v4.56.2/en/model_doc/bloom#transformers.BloomForQuestionAnswering">BloomForQuestionAnswering</a> forward method, overrides the <code>__call__</code> special method.',Po,pe,Eo,me,Kt,Xe,eo,kt,to;return ue=new O({props:{title:"BLOOM",local:"bloom",headingTag:"h1"}}),he=new O({props:{title:"Overview",local:"overview",headingTag:"h2"}}),_e=new O({props:{title:"Resources",local:"resources",headingTag:"h2"}}),ye=new Rn({props:{pipeline:"text-generation"}}),Be=new O({props:{title:"BloomConfig",local:"transformers.BloomConfig",headingTag:"h2"}}),xe=new S({props:{name:"class transformers.BloomConfig",anchor:"transformers.BloomConfig",parameters:[{name:"vocab_size",val:" = 250880"},{name:"hidden_size",val:" = 64"},{name:"n_layer",val:" = 2"},{name:"n_head",val:" = 8"},{name:"layer_norm_epsilon",val:" = 1e-05"},{name:"initializer_range",val:" = 0.02"},{name:"use_cache",val:" = True"},{name:"bos_token_id",val:" = 1"},{name:"eos_token_id",val:" = 2"},{name:"apply_residual_connection_post_layernorm",val:" = False"},{name:"hidden_dropout",val:" = 0.0"},{name:"attention_dropout",val:" = 0.0"},{name:"pretraining_tp",val:" = 1"},{name:"slow_but_exact",val:" = False"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.BloomConfig.vocab_size",description:`<strong>vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 250880) &#x2014;
Vocabulary size of the Bloom model. Defines the maximum number of different tokens that can be represented
by the <code>inputs_ids</code> passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/bloom#transformers.BloomModel">BloomModel</a>. Check <a href="https://huggingface.co/bigscience/bloom/discussions/120#633d28389addb8530b406c2a" rel="nofollow">this
discussion</a> on how the
<code>vocab_size</code> has been defined.`,name:"vocab_size"},{anchor:"transformers.BloomConfig.hidden_size",description:`<strong>hidden_size</strong> (<code>int</code>, <em>optional</em>, defaults to 64) &#x2014;
Dimensionality of the embeddings and hidden states.`,name:"hidden_size"},{anchor:"transformers.BloomConfig.n_layer",description:`<strong>n_layer</strong> (<code>int</code>, <em>optional</em>, defaults to 2) &#x2014;
Number of hidden layers in the Transformer encoder.`,name:"n_layer"},{anchor:"transformers.BloomConfig.n_head",description:`<strong>n_head</strong> (<code>int</code>, <em>optional</em>, defaults to 8) &#x2014;
Number of attention heads for each attention layer in the Transformer encoder.`,name:"n_head"},{anchor:"transformers.BloomConfig.layer_norm_epsilon",description:`<strong>layer_norm_epsilon</strong> (<code>float</code>, <em>optional</em>, defaults to 1e-5) &#x2014;
The epsilon to use in the layer normalization layers.`,name:"layer_norm_epsilon"},{anchor:"transformers.BloomConfig.initializer_range",description:`<strong>initializer_range</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.`,name:"initializer_range"},{anchor:"transformers.BloomConfig.apply_residual_connection_post_layernorm",description:`<strong>apply_residual_connection_post_layernorm</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
If enabled, use the layer norm of the hidden states as the residual in the transformer blocks`,name:"apply_residual_connection_post_layernorm"},{anchor:"transformers.BloomConfig.hidden_dropout",description:`<strong>hidden_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
Dropout rate of the dropout function on the bias dropout.`,name:"hidden_dropout"},{anchor:"transformers.BloomConfig.attention_dropout",description:`<strong>attention_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
Dropout rate applied to the attention probs`,name:"attention_dropout"},{anchor:"transformers.BloomConfig.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not the model should return the last key/values attentions (not used by all models).`,name:"use_cache"},{anchor:"transformers.BloomConfig.pretraining_tp",description:`<strong>pretraining_tp</strong> (<code>int</code>, <em>optional</em>, defaults to <code>1</code>) &#x2014;
Experimental feature. Tensor parallelism rank used during pretraining with Megatron. Please refer to <a href="https://huggingface.co/docs/transformers/parallelism" rel="nofollow">this
document</a> to understand more about it. This value is
necessary to ensure exact reproducibility of the pretraining results. Please refer to <a href="https://github.com/pytorch/pytorch/issues/76232" rel="nofollow">this
issue</a>. Note also that this is enabled only when
<code>slow_but_exact=True</code>.`,name:"pretraining_tp"},{anchor:"transformers.BloomConfig.slow_but_exact",description:`<strong>slow_but_exact</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Experimental feature. Whether to use slow but exact implementation of the attention mechanism. While
merging the TP rank tensors, due to slicing operations the results may be slightly different between the
model trained on Megatron and our model. Please refer to <a href="https://github.com/pytorch/pytorch/issues/76232" rel="nofollow">this
issue</a>. A solution to obtain more accurate results is to
enable this feature. Enabling this will hurt the computational time of the inference. Will be probably
resolved in the future once the main model has been fine-tuned with TP_rank=1.`,name:"slow_but_exact"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/bloom/configuration_bloom.py#L35"}}),ee=new Pe({props:{anchor:"transformers.BloomConfig.example",$$slots:{default:[Hn]},$$scope:{ctx:M}}}),Ue=new O({props:{title:"BloomTokenizerFast",local:"transformers.BloomTokenizerFast",headingTag:"h2"}}),je=new S({props:{name:"class transformers.BloomTokenizerFast",anchor:"transformers.BloomTokenizerFast",parameters:[{name:"vocab_file",val:" = None"},{name:"merges_file",val:" = None"},{name:"tokenizer_file",val:" = None"},{name:"unk_token",val:" = '<unk>'"},{name:"bos_token",val:" = '<s>'"},{name:"eos_token",val:" = '</s>'"},{name:"pad_token",val:" = '<pad>'"},{name:"add_prefix_space",val:" = False"},{name:"clean_up_tokenization_spaces",val:" = False"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.BloomTokenizerFast.vocab_file",description:`<strong>vocab_file</strong> (<code>str</code>) &#x2014;
Path to the vocabulary file.`,name:"vocab_file"},{anchor:"transformers.BloomTokenizerFast.merges_file",description:`<strong>merges_file</strong> (<code>str</code>) &#x2014;
Path to the merges file.`,name:"merges_file"},{anchor:"transformers.BloomTokenizerFast.errors",description:`<strong>errors</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;replace&quot;</code>) &#x2014;
Paradigm to follow when decoding bytes to UTF-8. See
<a href="https://docs.python.org/3/library/stdtypes.html#bytes.decode" rel="nofollow">bytes.decode</a> for more information.`,name:"errors"},{anchor:"transformers.BloomTokenizerFast.unk_token",description:`<strong>unk_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&lt;|endoftext|&gt;</code>) &#x2014;
The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
token instead.`,name:"unk_token"},{anchor:"transformers.BloomTokenizerFast.bos_token",description:`<strong>bos_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&lt;|endoftext|&gt;</code>) &#x2014;
The beginning of sequence token.`,name:"bos_token"},{anchor:"transformers.BloomTokenizerFast.eos_token",description:`<strong>eos_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&lt;|endoftext|&gt;</code>) &#x2014;
The end of sequence token.`,name:"eos_token"},{anchor:"transformers.BloomTokenizerFast.add_prefix_space",description:`<strong>add_prefix_space</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to add an initial space to the input. This allows to treat the leading word just as any
other word. (Bloom tokenizer detect beginning of words by the preceding space).`,name:"add_prefix_space"},{anchor:"transformers.BloomTokenizerFast.trim_offsets",description:`<strong>trim_offsets</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not the post-processing step should trim offsets to avoid including whitespaces.`,name:"trim_offsets"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/bloom/tokenization_bloom_fast.py#L30"}}),te=new Pe({props:{anchor:"transformers.BloomTokenizerFast.example",$$slots:{default:[Qn]},$$scope:{ctx:M}}}),oe=new Mt({props:{$$slots:{default:[Sn]},$$scope:{ctx:M}}}),ze=new O({props:{title:"BloomModel",local:"transformers.BloomModel",headingTag:"h2"}}),Fe=new S({props:{name:"class transformers.BloomModel",anchor:"transformers.BloomModel",parameters:[{name:"config",val:": BloomConfig"}],parametersDescription:[{anchor:"transformers.BloomModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/bloom#transformers.BloomConfig">BloomConfig</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/bloom/modeling_bloom.py#L460"}}),Je=new S({props:{name:"forward",anchor:"transformers.BloomModel.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Union[transformers.cache_utils.Cache, tuple[tuple[torch.Tensor, torch.Tensor], ...], NoneType] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.LongTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"},{name:"**deprecated_arguments",val:""}],parametersDescription:[{anchor:"transformers.BloomModel.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, input_ids_length)</code>) &#x2014;
<code>input_ids_length</code> = <code>sequence_length</code> if <code>past_key_values</code> is <code>None</code> else <code>past_key_values.get_seq_length()</code>
(<code>sequence_length</code> of input past key value states). Indices of input sequence tokens in the vocabulary.</p>
<p>If <code>past_key_values</code> is used, only <code>input_ids</code> that do not have their past calculated should be passed as
<code>input_ids</code>.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.BloomModel.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>Union[~cache_utils.Cache, tuple[tuple[torch.Tensor, torch.Tensor], ...], NoneType]</code>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.BloomModel.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.BloomModel.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.LongTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.BloomModel.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.BloomModel.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.BloomModel.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.BloomModel.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.BloomModel.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.BloomModel.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/bloom/modeling_bloom.py#L491",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPastAndCrossAttentions"
>transformers.modeling_outputs.BaseModelOutputWithPastAndCrossAttentions</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/bloom#transformers.BloomConfig"
>BloomConfig</a>) and inputs.</p>
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
`}}),ne=new Mt({props:{$$slots:{default:[Xn]},$$scope:{ctx:M}}}),We=new O({props:{title:"BloomForCausalLM",local:"transformers.BloomForCausalLM",headingTag:"h2"}}),Ie=new S({props:{name:"class transformers.BloomForCausalLM",anchor:"transformers.BloomForCausalLM",parameters:[{name:"config",val:": BloomConfig"}],parametersDescription:[{anchor:"transformers.BloomForCausalLM.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/bloom#transformers.BloomConfig">BloomConfig</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/bloom/modeling_bloom.py#L751"}}),qe=new S({props:{name:"forward",anchor:"transformers.BloomForCausalLM.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Union[transformers.cache_utils.Cache, tuple[tuple[torch.Tensor, torch.Tensor], ...], NoneType] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"labels",val:": typing.Optional[torch.Tensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"},{name:"**deprecated_arguments",val:""}],parametersDescription:[{anchor:"transformers.BloomForCausalLM.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, input_ids_length)</code>) &#x2014;
<code>input_ids_length</code> = <code>sequence_length</code> if <code>past_key_values</code> is <code>None</code> else <code>past_key_values.get_seq_length()</code>
(<code>sequence_length</code> of input past key value states). Indices of input sequence tokens in the vocabulary.</p>
<p>If <code>past_key_values</code> is used, only <code>input_ids</code> that do not have their past calculated should be passed as
<code>input_ids</code>.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.BloomForCausalLM.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>Union[~cache_utils.Cache, tuple[tuple[torch.Tensor, torch.Tensor], ...], NoneType]</code>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.BloomForCausalLM.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.BloomForCausalLM.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.BloomForCausalLM.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.BloomForCausalLM.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for language modeling. Note that the labels <strong>are shifted</strong> inside the model, i.e. you can set
<code>labels = input_ids</code> Indices are selected in <code>[-100, 0, ..., config.vocab_size]</code> All labels set to <code>-100</code>
are ignored (masked), the loss is only computed for labels in <code>[0, ..., config.vocab_size]</code>`,name:"labels"},{anchor:"transformers.BloomForCausalLM.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.BloomForCausalLM.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.BloomForCausalLM.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.BloomForCausalLM.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.BloomForCausalLM.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/bloom/modeling_bloom.py#L827",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithCrossAttentions"
>transformers.modeling_outputs.CausalLMOutputWithCrossAttentions</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/bloom#transformers.BloomConfig"
>BloomConfig</a>) and inputs.</p>
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
`}}),se=new Mt({props:{$$slots:{default:[Pn]},$$scope:{ctx:M}}}),ae=new Pe({props:{anchor:"transformers.BloomForCausalLM.forward.example",$$slots:{default:[En]},$$scope:{ctx:M}}}),Ne=new O({props:{title:"BloomForSequenceClassification",local:"transformers.BloomForSequenceClassification",headingTag:"h2"}}),Ze=new S({props:{name:"class transformers.BloomForSequenceClassification",anchor:"transformers.BloomForSequenceClassification",parameters:[{name:"config",val:": BloomConfig"}],parametersDescription:[{anchor:"transformers.BloomForSequenceClassification.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/bloom#transformers.BloomConfig">BloomConfig</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/bloom/modeling_bloom.py#L929"}}),Le=new S({props:{name:"forward",anchor:"transformers.BloomForSequenceClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Union[transformers.cache_utils.Cache, tuple[tuple[torch.Tensor, torch.Tensor], ...], NoneType] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"labels",val:": typing.Optional[torch.Tensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"**deprecated_arguments",val:""}],parametersDescription:[{anchor:"transformers.BloomForSequenceClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, input_ids_length)</code>) &#x2014;
<code>input_ids_length</code> = <code>sequence_length</code> if <code>past_key_values</code> is <code>None</code> else <code>past_key_values.get_seq_length()</code>
(<code>sequence_length</code> of input past key value states). Indices of input sequence tokens in the vocabulary.</p>
<p>If <code>past_key_values</code> is used, only <code>input_ids</code> that do not have their past calculated should be passed as
<code>input_ids</code>.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.BloomForSequenceClassification.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>Union[~cache_utils.Cache, tuple[tuple[torch.Tensor, torch.Tensor], ...], NoneType]</code>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.BloomForSequenceClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.BloomForSequenceClassification.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.BloomForSequenceClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.BloomForSequenceClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the sequence classification/regression loss. Indices should be in <code>[0, ..., config.num_labels - 1]</code>. If <code>config.num_labels == 1</code> a regression loss is computed (Mean-Square loss), If
<code>config.num_labels &gt; 1</code> a classification loss is computed (Cross-Entropy).`,name:"labels"},{anchor:"transformers.BloomForSequenceClassification.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.BloomForSequenceClassification.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.BloomForSequenceClassification.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.BloomForSequenceClassification.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/bloom/modeling_bloom.py#L939",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <code>transformers.modeling_outputs.SequenceClassifierOutputWithPast</code> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/bloom#transformers.BloomConfig"
>BloomConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>loss</strong> (<code>torch.FloatTensor</code> of shape <code>(1,)</code>, <em>optional</em>, returned when <code>labels</code> is provided) — Classification (or regression if config.num_labels==1) loss.</p>
</li>
<li>
<p><strong>logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, config.num_labels)</code>) — Classification (or regression if config.num_labels==1) scores (before SoftMax).</p>
</li>
<li>
<p><strong>past_key_values</strong> (<code>Cache</code>, <em>optional</em>, returned when <code>use_cache=True</code> is passed or when <code>config.use_cache=True</code>) — It is a <a
  href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache"
>Cache</a> instance. For more details, see our <a
  href="https://huggingface.co/docs/transformers/en/kv_cache"
  rel="nofollow"
>kv cache guide</a>.</p>
<p>Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
<code>past_key_values</code> input) to speed up sequential decoding.</p>
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


<p><code>transformers.modeling_outputs.SequenceClassifierOutputWithPast</code> or <code>tuple(torch.FloatTensor)</code></p>
`}}),re=new Mt({props:{$$slots:{default:[An]},$$scope:{ctx:M}}}),le=new Pe({props:{anchor:"transformers.BloomForSequenceClassification.forward.example",$$slots:{default:[Yn]},$$scope:{ctx:M}}}),ie=new Pe({props:{anchor:"transformers.BloomForSequenceClassification.forward.example-2",$$slots:{default:[On]},$$scope:{ctx:M}}}),Ge=new O({props:{title:"BloomForTokenClassification",local:"transformers.BloomForTokenClassification",headingTag:"h2"}}),Re=new S({props:{name:"class transformers.BloomForTokenClassification",anchor:"transformers.BloomForTokenClassification",parameters:[{name:"config",val:": BloomConfig"}],parametersDescription:[{anchor:"transformers.BloomForTokenClassification.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/bloom#transformers.BloomConfig">BloomConfig</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/bloom/modeling_bloom.py#L1057"}}),Ve=new S({props:{name:"forward",anchor:"transformers.BloomForTokenClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Union[transformers.cache_utils.Cache, tuple[tuple[torch.Tensor, torch.Tensor], ...], NoneType] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"labels",val:": typing.Optional[torch.Tensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"**deprecated_arguments",val:""}],parametersDescription:[{anchor:"transformers.BloomForTokenClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, input_ids_length)</code>) &#x2014;
<code>input_ids_length</code> = <code>sequence_length</code> if <code>past_key_values</code> is <code>None</code> else <code>past_key_values.get_seq_length()</code>
(<code>sequence_length</code> of input past key value states). Indices of input sequence tokens in the vocabulary.</p>
<p>If <code>past_key_values</code> is used, only <code>input_ids</code> that do not have their past calculated should be passed as
<code>input_ids</code>.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.BloomForTokenClassification.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>Union[~cache_utils.Cache, tuple[tuple[torch.Tensor, torch.Tensor], ...], NoneType]</code>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.BloomForTokenClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.BloomForTokenClassification.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.BloomForTokenClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.BloomForTokenClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the sequence classification/regression loss. Indices should be in <code>[0, ..., config.num_labels - 1]</code>. If <code>config.num_labels == 1</code> a regression loss is computed (Mean-Square loss), If
<code>config.num_labels &gt; 1</code> a classification loss is computed (Cross-Entropy).`,name:"labels"},{anchor:"transformers.BloomForTokenClassification.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.BloomForTokenClassification.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.BloomForTokenClassification.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.BloomForTokenClassification.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/bloom/modeling_bloom.py#L1075",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput"
>transformers.modeling_outputs.TokenClassifierOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/bloom#transformers.BloomConfig"
>BloomConfig</a>) and inputs.</p>
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
`}}),ce=new Mt({props:{$$slots:{default:[Dn]},$$scope:{ctx:M}}}),de=new Pe({props:{anchor:"transformers.BloomForTokenClassification.forward.example",$$slots:{default:[Kn]},$$scope:{ctx:M}}}),He=new O({props:{title:"BloomForQuestionAnswering",local:"transformers.BloomForQuestionAnswering",headingTag:"h2"}}),Qe=new S({props:{name:"class transformers.BloomForQuestionAnswering",anchor:"transformers.BloomForQuestionAnswering",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.BloomForQuestionAnswering.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/bloom#transformers.BloomForQuestionAnswering">BloomForQuestionAnswering</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/bloom/modeling_bloom.py#L1158"}}),Se=new S({props:{name:"forward",anchor:"transformers.BloomForQuestionAnswering.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"start_positions",val:": typing.Optional[torch.LongTensor] = None"},{name:"end_positions",val:": typing.Optional[torch.LongTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.BloomForQuestionAnswering.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, input_ids_length)</code>) &#x2014;
<code>input_ids_length</code> = <code>sequence_length</code> if <code>past_key_values</code> is <code>None</code> else <code>past_key_values.get_seq_length()</code>
(<code>sequence_length</code> of input past key value states). Indices of input sequence tokens in the vocabulary.</p>
<p>If <code>past_key_values</code> is used, only <code>input_ids</code> that do not have their past calculated should be passed as
<code>input_ids</code>.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.BloomForQuestionAnswering.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.BloomForQuestionAnswering.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.BloomForQuestionAnswering.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.BloomForQuestionAnswering.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.BloomForQuestionAnswering.forward.start_positions",description:`<strong>start_positions</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for position (index) of the start of the labelled span for computing the token classification loss.
Positions are clamped to the length of the sequence (<code>sequence_length</code>). Position outside of the sequence
are not taken into account for computing the loss.`,name:"start_positions"},{anchor:"transformers.BloomForQuestionAnswering.forward.end_positions",description:`<strong>end_positions</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for position (index) of the end of the labelled span for computing the token classification loss.
Positions are clamped to the length of the sequence (<code>sequence_length</code>). Position outside of the sequence
are not taken into account for computing the loss.`,name:"end_positions"},{anchor:"transformers.BloomForQuestionAnswering.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.BloomForQuestionAnswering.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.BloomForQuestionAnswering.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/bloom/modeling_bloom.py#L1167",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.QuestionAnsweringModelOutput"
>transformers.modeling_outputs.QuestionAnsweringModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/bloom#transformers.BloomConfig"
>BloomConfig</a>) and inputs.</p>
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
`}}),pe=new Mt({props:{$$slots:{default:[es]},$$scope:{ctx:M}}}),me=new Pe({props:{anchor:"transformers.BloomForQuestionAnswering.forward.example",$$slots:{default:[ts]},$$scope:{ctx:M}}}),Xe=new Vn({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/bloom.md"}}),{c(){t=p("meta"),T=a(),i=p("p"),d=a(),v=p("p"),v.innerHTML=o,w=a(),h(ue.$$.fragment),$t=a(),K=p("div"),K.innerHTML=Ao,Ct=a(),h(he.$$.fragment),Bt=a(),fe=p("p"),fe.innerHTML=Yo,xt=a(),ge=p("ul"),ge.innerHTML=Oo,Ut=a(),h(_e.$$.fragment),jt=a(),be=p("p"),be.textContent=Do,zt=a(),h(ye.$$.fragment),Ft=a(),Te=p("ul"),Te.innerHTML=Ko,Jt=a(),ve=p("p"),ve.textContent=en,Wt=a(),we=p("ul"),we.innerHTML=tn,It=a(),Me=p("p"),Me.textContent=on,qt=a(),ke=p("ul"),ke.innerHTML=nn,Nt=a(),$e=p("p"),$e.textContent=sn,Zt=a(),Ce=p("ul"),Ce.innerHTML=an,Lt=a(),h(Be.$$.fragment),Gt=a(),I=p("div"),h(xe.$$.fragment),oo=a(),Ae=p("p"),Ae.innerHTML=rn,no=a(),Ye=p("p"),Ye.innerHTML=ln,so=a(),h(ee.$$.fragment),Rt=a(),h(Ue.$$.fragment),Vt=a(),k=p("div"),h(je.$$.fragment),ao=a(),Oe=p("p"),Oe.innerHTML=cn,ro=a(),De=p("p"),De.textContent=dn,lo=a(),h(te.$$.fragment),io=a(),Ke=p("p"),Ke.innerHTML=pn,co=a(),h(oe.$$.fragment),po=a(),et=p("p"),et.innerHTML=mn,Ht=a(),h(ze.$$.fragment),Qt=a(),x=p("div"),h(Fe.$$.fragment),mo=a(),tt=p("p"),tt.textContent=un,uo=a(),ot=p("p"),ot.innerHTML=hn,ho=a(),nt=p("p"),nt.innerHTML=fn,fo=a(),Y=p("div"),h(Je.$$.fragment),go=a(),st=p("p"),st.innerHTML=gn,_o=a(),h(ne.$$.fragment),St=a(),h(We.$$.fragment),Xt=a(),U=p("div"),h(Ie.$$.fragment),bo=a(),at=p("p"),at.textContent=_n,yo=a(),rt=p("p"),rt.innerHTML=bn,To=a(),lt=p("p"),lt.innerHTML=yn,vo=a(),V=p("div"),h(qe.$$.fragment),wo=a(),it=p("p"),it.innerHTML=Tn,Mo=a(),h(se.$$.fragment),ko=a(),h(ae.$$.fragment),Pt=a(),h(Ne.$$.fragment),Et=a(),$=p("div"),h(Ze.$$.fragment),$o=a(),ct=p("p"),ct.textContent=vn,Co=a(),dt=p("p"),dt.innerHTML=wn,Bo=a(),pt=p("p"),pt.innerHTML=Mn,xo=a(),mt=p("p"),mt.innerHTML=kn,Uo=a(),ut=p("p"),ut.innerHTML=$n,jo=a(),J=p("div"),h(Le.$$.fragment),zo=a(),ht=p("p"),ht.innerHTML=Cn,Fo=a(),h(re.$$.fragment),Jo=a(),h(le.$$.fragment),Wo=a(),h(ie.$$.fragment),At=a(),h(Ge.$$.fragment),Yt=a(),j=p("div"),h(Re.$$.fragment),Io=a(),ft=p("p"),ft.textContent=Bn,qo=a(),gt=p("p"),gt.innerHTML=xn,No=a(),_t=p("p"),_t.innerHTML=Un,Zo=a(),H=p("div"),h(Ve.$$.fragment),Lo=a(),bt=p("p"),bt.innerHTML=jn,Go=a(),h(ce.$$.fragment),Ro=a(),h(de.$$.fragment),Ot=a(),h(He.$$.fragment),Dt=a(),z=p("div"),h(Qe.$$.fragment),Vo=a(),yt=p("p"),yt.innerHTML=zn,Ho=a(),Tt=p("p"),Tt.innerHTML=Fn,Qo=a(),vt=p("p"),vt.innerHTML=Jn,So=a(),Q=p("div"),h(Se.$$.fragment),Xo=a(),wt=p("p"),wt.innerHTML=Wn,Po=a(),h(pe.$$.fragment),Eo=a(),h(me.$$.fragment),Kt=a(),h(Xe.$$.fragment),eo=a(),kt=p("p"),this.h()},l(e){const n=Gn("svelte-u9bgzb",document.head);t=m(n,"META",{name:!0,content:!0}),n.forEach(s),T=r(e),i=m(e,"P",{}),W(i).forEach(s),d=r(e),v=m(e,"P",{"data-svelte-h":!0}),u(v)!=="svelte-1ok7zt0"&&(v.innerHTML=o),w=r(e),f(ue.$$.fragment,e),$t=r(e),K=m(e,"DIV",{class:!0,"data-svelte-h":!0}),u(K)!=="svelte-13t8s2t"&&(K.innerHTML=Ao),Ct=r(e),f(he.$$.fragment,e),Bt=r(e),fe=m(e,"P",{"data-svelte-h":!0}),u(fe)!=="svelte-1nps9yp"&&(fe.innerHTML=Yo),xt=r(e),ge=m(e,"UL",{"data-svelte-h":!0}),u(ge)!=="svelte-aurotn"&&(ge.innerHTML=Oo),Ut=r(e),f(_e.$$.fragment,e),jt=r(e),be=m(e,"P",{"data-svelte-h":!0}),u(be)!=="svelte-ijmgvk"&&(be.textContent=Do),zt=r(e),f(ye.$$.fragment,e),Ft=r(e),Te=m(e,"UL",{"data-svelte-h":!0}),u(Te)!=="svelte-1s9drdb"&&(Te.innerHTML=Ko),Jt=r(e),ve=m(e,"P",{"data-svelte-h":!0}),u(ve)!=="svelte-ee7uqc"&&(ve.textContent=en),Wt=r(e),we=m(e,"UL",{"data-svelte-h":!0}),u(we)!=="svelte-19ngtoj"&&(we.innerHTML=tn),It=r(e),Me=m(e,"P",{"data-svelte-h":!0}),u(Me)!=="svelte-1wntqpp"&&(Me.textContent=on),qt=r(e),ke=m(e,"UL",{"data-svelte-h":!0}),u(ke)!=="svelte-i91jqz"&&(ke.innerHTML=nn),Nt=r(e),$e=m(e,"P",{"data-svelte-h":!0}),u($e)!=="svelte-vhz7nm"&&($e.textContent=sn),Zt=r(e),Ce=m(e,"UL",{"data-svelte-h":!0}),u(Ce)!=="svelte-qfnfeg"&&(Ce.innerHTML=an),Lt=r(e),f(Be.$$.fragment,e),Gt=r(e),I=m(e,"DIV",{class:!0});var X=W(I);f(xe.$$.fragment,X),oo=r(X),Ae=m(X,"P",{"data-svelte-h":!0}),u(Ae)!=="svelte-gcxv90"&&(Ae.innerHTML=rn),no=r(X),Ye=m(X,"P",{"data-svelte-h":!0}),u(Ye)!=="svelte-1ek1ss9"&&(Ye.innerHTML=ln),so=r(X),f(ee.$$.fragment,X),X.forEach(s),Rt=r(e),f(Ue.$$.fragment,e),Vt=r(e),k=m(e,"DIV",{class:!0});var C=W(k);f(je.$$.fragment,C),ao=r(C),Oe=m(C,"P",{"data-svelte-h":!0}),u(Oe)!=="svelte-18ugbid"&&(Oe.innerHTML=cn),ro=r(C),De=m(C,"P",{"data-svelte-h":!0}),u(De)!=="svelte-1s077p3"&&(De.textContent=dn),lo=r(C),f(te.$$.fragment,C),io=r(C),Ke=m(C,"P",{"data-svelte-h":!0}),u(Ke)!=="svelte-1afeqmz"&&(Ke.innerHTML=pn),co=r(C),f(oe.$$.fragment,C),po=r(C),et=m(C,"P",{"data-svelte-h":!0}),u(et)!=="svelte-gxzj9w"&&(et.innerHTML=mn),C.forEach(s),Ht=r(e),f(ze.$$.fragment,e),Qt=r(e),x=m(e,"DIV",{class:!0});var q=W(x);f(Fe.$$.fragment,q),mo=r(q),tt=m(q,"P",{"data-svelte-h":!0}),u(tt)!=="svelte-koxq4n"&&(tt.textContent=un),uo=r(q),ot=m(q,"P",{"data-svelte-h":!0}),u(ot)!=="svelte-q52n56"&&(ot.innerHTML=hn),ho=r(q),nt=m(q,"P",{"data-svelte-h":!0}),u(nt)!=="svelte-hswkmf"&&(nt.innerHTML=fn),fo=r(q),Y=m(q,"DIV",{class:!0});var D=W(Y);f(Je.$$.fragment,D),go=r(D),st=m(D,"P",{"data-svelte-h":!0}),u(st)!=="svelte-u9elgy"&&(st.innerHTML=gn),_o=r(D),f(ne.$$.fragment,D),D.forEach(s),q.forEach(s),St=r(e),f(We.$$.fragment,e),Xt=r(e),U=m(e,"DIV",{class:!0});var N=W(U);f(Ie.$$.fragment,N),bo=r(N),at=m(N,"P",{"data-svelte-h":!0}),u(at)!=="svelte-ywufu4"&&(at.textContent=_n),yo=r(N),rt=m(N,"P",{"data-svelte-h":!0}),u(rt)!=="svelte-q52n56"&&(rt.innerHTML=bn),To=r(N),lt=m(N,"P",{"data-svelte-h":!0}),u(lt)!=="svelte-hswkmf"&&(lt.innerHTML=yn),vo=r(N),V=m(N,"DIV",{class:!0});var P=W(V);f(qe.$$.fragment,P),wo=r(P),it=m(P,"P",{"data-svelte-h":!0}),u(it)!=="svelte-1b485fe"&&(it.innerHTML=Tn),Mo=r(P),f(se.$$.fragment,P),ko=r(P),f(ae.$$.fragment,P),P.forEach(s),N.forEach(s),Pt=r(e),f(Ne.$$.fragment,e),Et=r(e),$=m(e,"DIV",{class:!0});var B=W($);f(Ze.$$.fragment,B),$o=r(B),ct=m(B,"P",{"data-svelte-h":!0}),u(ct)!=="svelte-5utzhf"&&(ct.textContent=vn),Co=r(B),dt=m(B,"P",{"data-svelte-h":!0}),u(dt)!=="svelte-12pz2sg"&&(dt.innerHTML=wn),Bo=r(B),pt=m(B,"P",{"data-svelte-h":!0}),u(pt)!=="svelte-10ugs3m"&&(pt.innerHTML=Mn),xo=r(B),mt=m(B,"P",{"data-svelte-h":!0}),u(mt)!=="svelte-q52n56"&&(mt.innerHTML=kn),Uo=r(B),ut=m(B,"P",{"data-svelte-h":!0}),u(ut)!=="svelte-hswkmf"&&(ut.innerHTML=$n),jo=r(B),J=m(B,"DIV",{class:!0});var Z=W(J);f(Le.$$.fragment,Z),zo=r(Z),ht=m(Z,"P",{"data-svelte-h":!0}),u(ht)!=="svelte-1xex564"&&(ht.innerHTML=Cn),Fo=r(Z),f(re.$$.fragment,Z),Jo=r(Z),f(le.$$.fragment,Z),Wo=r(Z),f(ie.$$.fragment,Z),Z.forEach(s),B.forEach(s),At=r(e),f(Ge.$$.fragment,e),Yt=r(e),j=m(e,"DIV",{class:!0});var L=W(j);f(Re.$$.fragment,L),Io=r(L),ft=m(L,"P",{"data-svelte-h":!0}),u(ft)!=="svelte-19sbvrd"&&(ft.textContent=Bn),qo=r(L),gt=m(L,"P",{"data-svelte-h":!0}),u(gt)!=="svelte-q52n56"&&(gt.innerHTML=xn),No=r(L),_t=m(L,"P",{"data-svelte-h":!0}),u(_t)!=="svelte-hswkmf"&&(_t.innerHTML=Un),Zo=r(L),H=m(L,"DIV",{class:!0});var E=W(H);f(Ve.$$.fragment,E),Lo=r(E),bt=m(E,"P",{"data-svelte-h":!0}),u(bt)!=="svelte-b5y3ou"&&(bt.innerHTML=jn),Go=r(E),f(ce.$$.fragment,E),Ro=r(E),f(de.$$.fragment,E),E.forEach(s),L.forEach(s),Ot=r(e),f(He.$$.fragment,e),Dt=r(e),z=m(e,"DIV",{class:!0});var G=W(z);f(Qe.$$.fragment,G),Vo=r(G),yt=m(G,"P",{"data-svelte-h":!0}),u(yt)!=="svelte-grteqe"&&(yt.innerHTML=zn),Ho=r(G),Tt=m(G,"P",{"data-svelte-h":!0}),u(Tt)!=="svelte-q52n56"&&(Tt.innerHTML=Fn),Qo=r(G),vt=m(G,"P",{"data-svelte-h":!0}),u(vt)!=="svelte-hswkmf"&&(vt.innerHTML=Jn),So=r(G),Q=m(G,"DIV",{class:!0});var A=W(Q);f(Se.$$.fragment,A),Xo=r(A),wt=m(A,"P",{"data-svelte-h":!0}),u(wt)!=="svelte-1ejwwyu"&&(wt.innerHTML=Wn),Po=r(A),f(pe.$$.fragment,A),Eo=r(A),f(me.$$.fragment,A),A.forEach(s),G.forEach(s),Kt=r(e),f(Xe.$$.fragment,e),eo=r(e),kt=m(e,"P",{}),W(kt).forEach(s),this.h()},h(){F(t,"name","hf:doc:metadata"),F(t,"content",ns),F(K,"class","flex flex-wrap space-x-1"),F(I,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),F(k,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),F(Y,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),F(x,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),F(V,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),F(U,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),F(J,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),F($,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),F(H,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),F(j,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),F(Q,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),F(z,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,n){l(document.head,t),c(e,T,n),c(e,i,n),c(e,d,n),c(e,v,n),c(e,w,n),g(ue,e,n),c(e,$t,n),c(e,K,n),c(e,Ct,n),g(he,e,n),c(e,Bt,n),c(e,fe,n),c(e,xt,n),c(e,ge,n),c(e,Ut,n),g(_e,e,n),c(e,jt,n),c(e,be,n),c(e,zt,n),g(ye,e,n),c(e,Ft,n),c(e,Te,n),c(e,Jt,n),c(e,ve,n),c(e,Wt,n),c(e,we,n),c(e,It,n),c(e,Me,n),c(e,qt,n),c(e,ke,n),c(e,Nt,n),c(e,$e,n),c(e,Zt,n),c(e,Ce,n),c(e,Lt,n),g(Be,e,n),c(e,Gt,n),c(e,I,n),g(xe,I,null),l(I,oo),l(I,Ae),l(I,no),l(I,Ye),l(I,so),g(ee,I,null),c(e,Rt,n),g(Ue,e,n),c(e,Vt,n),c(e,k,n),g(je,k,null),l(k,ao),l(k,Oe),l(k,ro),l(k,De),l(k,lo),g(te,k,null),l(k,io),l(k,Ke),l(k,co),g(oe,k,null),l(k,po),l(k,et),c(e,Ht,n),g(ze,e,n),c(e,Qt,n),c(e,x,n),g(Fe,x,null),l(x,mo),l(x,tt),l(x,uo),l(x,ot),l(x,ho),l(x,nt),l(x,fo),l(x,Y),g(Je,Y,null),l(Y,go),l(Y,st),l(Y,_o),g(ne,Y,null),c(e,St,n),g(We,e,n),c(e,Xt,n),c(e,U,n),g(Ie,U,null),l(U,bo),l(U,at),l(U,yo),l(U,rt),l(U,To),l(U,lt),l(U,vo),l(U,V),g(qe,V,null),l(V,wo),l(V,it),l(V,Mo),g(se,V,null),l(V,ko),g(ae,V,null),c(e,Pt,n),g(Ne,e,n),c(e,Et,n),c(e,$,n),g(Ze,$,null),l($,$o),l($,ct),l($,Co),l($,dt),l($,Bo),l($,pt),l($,xo),l($,mt),l($,Uo),l($,ut),l($,jo),l($,J),g(Le,J,null),l(J,zo),l(J,ht),l(J,Fo),g(re,J,null),l(J,Jo),g(le,J,null),l(J,Wo),g(ie,J,null),c(e,At,n),g(Ge,e,n),c(e,Yt,n),c(e,j,n),g(Re,j,null),l(j,Io),l(j,ft),l(j,qo),l(j,gt),l(j,No),l(j,_t),l(j,Zo),l(j,H),g(Ve,H,null),l(H,Lo),l(H,bt),l(H,Go),g(ce,H,null),l(H,Ro),g(de,H,null),c(e,Ot,n),g(He,e,n),c(e,Dt,n),c(e,z,n),g(Qe,z,null),l(z,Vo),l(z,yt),l(z,Ho),l(z,Tt),l(z,Qo),l(z,vt),l(z,So),l(z,Q),g(Se,Q,null),l(Q,Xo),l(Q,wt),l(Q,Po),g(pe,Q,null),l(Q,Eo),g(me,Q,null),c(e,Kt,n),g(Xe,e,n),c(e,eo,n),c(e,kt,n),to=!0},p(e,[n]){const X={};n&2&&(X.$$scope={dirty:n,ctx:e}),ee.$set(X);const C={};n&2&&(C.$$scope={dirty:n,ctx:e}),te.$set(C);const q={};n&2&&(q.$$scope={dirty:n,ctx:e}),oe.$set(q);const D={};n&2&&(D.$$scope={dirty:n,ctx:e}),ne.$set(D);const N={};n&2&&(N.$$scope={dirty:n,ctx:e}),se.$set(N);const P={};n&2&&(P.$$scope={dirty:n,ctx:e}),ae.$set(P);const B={};n&2&&(B.$$scope={dirty:n,ctx:e}),re.$set(B);const Z={};n&2&&(Z.$$scope={dirty:n,ctx:e}),le.$set(Z);const L={};n&2&&(L.$$scope={dirty:n,ctx:e}),ie.$set(L);const E={};n&2&&(E.$$scope={dirty:n,ctx:e}),ce.$set(E);const G={};n&2&&(G.$$scope={dirty:n,ctx:e}),de.$set(G);const A={};n&2&&(A.$$scope={dirty:n,ctx:e}),pe.$set(A);const In={};n&2&&(In.$$scope={dirty:n,ctx:e}),me.$set(In)},i(e){to||(_(ue.$$.fragment,e),_(he.$$.fragment,e),_(_e.$$.fragment,e),_(ye.$$.fragment,e),_(Be.$$.fragment,e),_(xe.$$.fragment,e),_(ee.$$.fragment,e),_(Ue.$$.fragment,e),_(je.$$.fragment,e),_(te.$$.fragment,e),_(oe.$$.fragment,e),_(ze.$$.fragment,e),_(Fe.$$.fragment,e),_(Je.$$.fragment,e),_(ne.$$.fragment,e),_(We.$$.fragment,e),_(Ie.$$.fragment,e),_(qe.$$.fragment,e),_(se.$$.fragment,e),_(ae.$$.fragment,e),_(Ne.$$.fragment,e),_(Ze.$$.fragment,e),_(Le.$$.fragment,e),_(re.$$.fragment,e),_(le.$$.fragment,e),_(ie.$$.fragment,e),_(Ge.$$.fragment,e),_(Re.$$.fragment,e),_(Ve.$$.fragment,e),_(ce.$$.fragment,e),_(de.$$.fragment,e),_(He.$$.fragment,e),_(Qe.$$.fragment,e),_(Se.$$.fragment,e),_(pe.$$.fragment,e),_(me.$$.fragment,e),_(Xe.$$.fragment,e),to=!0)},o(e){b(ue.$$.fragment,e),b(he.$$.fragment,e),b(_e.$$.fragment,e),b(ye.$$.fragment,e),b(Be.$$.fragment,e),b(xe.$$.fragment,e),b(ee.$$.fragment,e),b(Ue.$$.fragment,e),b(je.$$.fragment,e),b(te.$$.fragment,e),b(oe.$$.fragment,e),b(ze.$$.fragment,e),b(Fe.$$.fragment,e),b(Je.$$.fragment,e),b(ne.$$.fragment,e),b(We.$$.fragment,e),b(Ie.$$.fragment,e),b(qe.$$.fragment,e),b(se.$$.fragment,e),b(ae.$$.fragment,e),b(Ne.$$.fragment,e),b(Ze.$$.fragment,e),b(Le.$$.fragment,e),b(re.$$.fragment,e),b(le.$$.fragment,e),b(ie.$$.fragment,e),b(Ge.$$.fragment,e),b(Re.$$.fragment,e),b(Ve.$$.fragment,e),b(ce.$$.fragment,e),b(de.$$.fragment,e),b(He.$$.fragment,e),b(Qe.$$.fragment,e),b(Se.$$.fragment,e),b(pe.$$.fragment,e),b(me.$$.fragment,e),b(Xe.$$.fragment,e),to=!1},d(e){e&&(s(T),s(i),s(d),s(v),s(w),s($t),s(K),s(Ct),s(Bt),s(fe),s(xt),s(ge),s(Ut),s(jt),s(be),s(zt),s(Ft),s(Te),s(Jt),s(ve),s(Wt),s(we),s(It),s(Me),s(qt),s(ke),s(Nt),s($e),s(Zt),s(Ce),s(Lt),s(Gt),s(I),s(Rt),s(Vt),s(k),s(Ht),s(Qt),s(x),s(St),s(Xt),s(U),s(Pt),s(Et),s($),s(At),s(Yt),s(j),s(Ot),s(Dt),s(z),s(Kt),s(eo),s(kt)),s(t),y(ue,e),y(he,e),y(_e,e),y(ye,e),y(Be,e),y(xe),y(ee),y(Ue,e),y(je),y(te),y(oe),y(ze,e),y(Fe),y(Je),y(ne),y(We,e),y(Ie),y(qe),y(se),y(ae),y(Ne,e),y(Ze),y(Le),y(re),y(le),y(ie),y(Ge,e),y(Re),y(Ve),y(ce),y(de),y(He,e),y(Qe),y(Se),y(pe),y(me),y(Xe,e)}}}const ns='{"title":"BLOOM","local":"bloom","sections":[{"title":"Overview","local":"overview","sections":[],"depth":2},{"title":"Resources","local":"resources","sections":[],"depth":2},{"title":"BloomConfig","local":"transformers.BloomConfig","sections":[],"depth":2},{"title":"BloomTokenizerFast","local":"transformers.BloomTokenizerFast","sections":[],"depth":2},{"title":"BloomModel","local":"transformers.BloomModel","sections":[],"depth":2},{"title":"BloomForCausalLM","local":"transformers.BloomForCausalLM","sections":[],"depth":2},{"title":"BloomForSequenceClassification","local":"transformers.BloomForSequenceClassification","sections":[],"depth":2},{"title":"BloomForTokenClassification","local":"transformers.BloomForTokenClassification","sections":[],"depth":2},{"title":"BloomForQuestionAnswering","local":"transformers.BloomForQuestionAnswering","sections":[],"depth":2}],"depth":1}';function ss(M){return Nn(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class us extends Zn{constructor(t){super(),Ln(this,t,ss,os,qn,{})}}export{us as component};
