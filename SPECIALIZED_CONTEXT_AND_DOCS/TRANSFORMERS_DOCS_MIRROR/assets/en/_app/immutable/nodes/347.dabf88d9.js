import{s as gn,o as _n,n as Q}from"../chunks/scheduler.18a86fab.js";import{S as bn,i as yn,g as p,s as a,r as u,m as Tn,A as kn,h,f as s,c as r,j as G,x as m,u as f,n as vn,k as F,y as i,a as d,v as g,d as _,t as b,w as y}from"../chunks/index.98837b22.js";import{T as Qt}from"../chunks/Tip.77304350.js";import{D as H}from"../chunks/Docstring.a1ef7999.js";import{C as de}from"../chunks/CodeBlock.8d0c2e8a.js";import{E as ut}from"../chunks/ExampleCodeBlock.8c3ee1f9.js";import{H as X,E as Mn}from"../chunks/getInferenceSnippets.06c2775f.js";function wn(M){let n,k="Example:",c,l,T;return l=new de({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMFBMQmFydENvbmZpZyUyQyUyMFBMQmFydE1vZGVsJTBBJTBBJTIzJTIwSW5pdGlhbGl6aW5nJTIwYSUyMFBMQkFSVCUyMHVjbGFubHAlMkZwbGJhcnQtYmFzZSUyMHN0eWxlJTIwY29uZmlndXJhdGlvbiUwQWNvbmZpZ3VyYXRpb24lMjAlM0QlMjBQTEJhcnRDb25maWcoKSUwQSUwQSUyMyUyMEluaXRpYWxpemluZyUyMGElMjBtb2RlbCUyMCh3aXRoJTIwcmFuZG9tJTIwd2VpZ2h0cyklMjBmcm9tJTIwdGhlJTIwdWNsYW5scCUyRnBsYmFydC1iYXNlJTIwc3R5bGUlMjBjb25maWd1cmF0aW9uJTBBbW9kZWwlMjAlM0QlMjBQTEJhcnRNb2RlbChjb25maWd1cmF0aW9uKSUwQSUwQSUyMyUyMEFjY2Vzc2luZyUyMHRoZSUyMG1vZGVsJTIwY29uZmlndXJhdGlvbiUwQWNvbmZpZ3VyYXRpb24lMjAlM0QlMjBtb2RlbC5jb25maWc=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> PLBartConfig, PLBartModel

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a PLBART uclanlp/plbart-base style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = PLBartConfig()

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a model (with random weights) from the uclanlp/plbart-base style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = PLBartModel(configuration)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Accessing the model configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = model.config`,wrap:!1}}),{c(){n=p("p"),n.textContent=k,c=a(),u(l.$$.fragment)},l(t){n=h(t,"P",{"data-svelte-h":!0}),m(n)!=="svelte-11lpom8"&&(n.textContent=k),c=r(t),f(l.$$.fragment,t)},m(t,v){d(t,n,v),d(t,c,v),g(l,t,v),T=!0},p:Q,i(t){T||(_(l.$$.fragment,t),T=!0)},o(t){b(l.$$.fragment,t),T=!1},d(t){t&&(s(n),s(c)),y(l,t)}}}function Cn(M){let n,k="Examples:",c,l,T;return l=new de({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMFBMQmFydFRva2VuaXplciUwQSUwQXRva2VuaXplciUyMCUzRCUyMFBMQmFydFRva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIydWNsYW5scCUyRnBsYmFydC1weXRob24tZW5fWFglMjIlMkMlMjBzcmNfbGFuZyUzRCUyMnB5dGhvbiUyMiUyQyUyMHRndF9sYW5nJTNEJTIyZW5fWFglMjIpJTBBZXhhbXBsZV9weXRob25fcGhyYXNlJTIwJTNEJTIwJTIyZGVmJTIwbWF4aW11bShhJTJDYiUyQ2MpJTNBTkVXX0xJTkVfSU5ERU5UcmV0dXJuJTIwbWF4KCU1QmElMkNiJTJDYyU1RCklMjIlMEFleHBlY3RlZF90cmFuc2xhdGlvbl9lbmdsaXNoJTIwJTNEJTIwJTIyUmV0dXJucyUyMHRoZSUyMG1heGltdW0lMjB2YWx1ZSUyMG9mJTIwYSUyMGIlMjBjLiUyMiUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplcihleGFtcGxlX3B5dGhvbl9waHJhc2UlMkMlMjB0ZXh0X3RhcmdldCUzRGV4cGVjdGVkX3RyYW5zbGF0aW9uX2VuZ2xpc2glMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKQ==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> PLBartTokenizer

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = PLBartTokenizer.from_pretrained(<span class="hljs-string">&quot;uclanlp/plbart-python-en_XX&quot;</span>, src_lang=<span class="hljs-string">&quot;python&quot;</span>, tgt_lang=<span class="hljs-string">&quot;en_XX&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>example_python_phrase = <span class="hljs-string">&quot;def maximum(a,b,c):NEW_LINE_INDENTreturn max([a,b,c])&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>expected_translation_english = <span class="hljs-string">&quot;Returns the maximum value of a b c.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(example_python_phrase, text_target=expected_translation_english, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)`,wrap:!1}}),{c(){n=p("p"),n.textContent=k,c=a(),u(l.$$.fragment)},l(t){n=h(t,"P",{"data-svelte-h":!0}),m(n)!=="svelte-kvfsh7"&&(n.textContent=k),c=r(t),f(l.$$.fragment,t)},m(t,v){d(t,n,v),d(t,c,v),g(l,t,v),T=!0},p:Q,i(t){T||(_(l.$$.fragment,t),T=!0)},o(t){b(l.$$.fragment,t),T=!1},d(t){t&&(s(n),s(c)),y(l,t)}}}function Ln(M){let n,k=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){n=p("p"),n.innerHTML=k},l(c){n=h(c,"P",{"data-svelte-h":!0}),m(n)!=="svelte-fincs2"&&(n.innerHTML=k)},m(c,l){d(c,n,l)},p:Q,d(c){c&&s(n)}}}function Bn(M){let n,k=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){n=p("p"),n.innerHTML=k},l(c){n=h(c,"P",{"data-svelte-h":!0}),m(n)!=="svelte-fincs2"&&(n.innerHTML=k)},m(c,l){d(c,n,l)},p:Q,d(c){c&&s(n)}}}function xn(M){let n,k="Example Mask-filling:",c,l,T;return l=new de({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBQTEJhcnRGb3JDb25kaXRpb25hbEdlbmVyYXRpb24lMEElMEFtb2RlbCUyMCUzRCUyMFBMQmFydEZvckNvbmRpdGlvbmFsR2VuZXJhdGlvbi5mcm9tX3ByZXRyYWluZWQoJTIydWNsYW5scCUyRnBsYmFydC1iYXNlJTIyKSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMnVjbGFubHAlMkZwbGJhcnQtYmFzZSUyMiklMEElMEElMjMlMjBlbl9YWCUyMGlzJTIwdGhlJTIwbGFuZ3VhZ2UlMjBzeW1ib2wlMjBpZCUyMCUzQ0xJRCUzRSUyMGZvciUyMEVuZ2xpc2glMEFUWFQlMjAlM0QlMjAlMjIlM0NzJTNFJTIwSXMlMjAwJTIwdGhlJTIwJTNDbWFzayUzRSUyMEZpYm9uYWNjaSUyMG51bWJlciUyMCUzRiUyMCUzQyUyRnMlM0UlMjBlbl9YWCUyMiUwQWlucHV0X2lkcyUyMCUzRCUyMHRva2VuaXplciglNUJUWFQlNUQlMkMlMjBhZGRfc3BlY2lhbF90b2tlbnMlM0RGYWxzZSUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpLmlucHV0X2lkcyUwQSUwQWxvZ2l0cyUyMCUzRCUyMG1vZGVsKGlucHV0X2lkcykubG9naXRzJTBBbWFza2VkX2luZGV4JTIwJTNEJTIwKGlucHV0X2lkcyU1QjAlNUQlMjAlM0QlM0QlMjB0b2tlbml6ZXIubWFza190b2tlbl9pZCkubm9uemVybygpLml0ZW0oKSUwQXByb2JzJTIwJTNEJTIwbG9naXRzJTVCMCUyQyUyMG1hc2tlZF9pbmRleCU1RC5zb2Z0bWF4KGRpbSUzRDApJTBBdmFsdWVzJTJDJTIwcHJlZGljdGlvbnMlMjAlM0QlMjBwcm9icy50b3BrKDUpJTBBJTBBdG9rZW5pemVyLmRlY29kZShwcmVkaWN0aW9ucykuc3BsaXQoKQ==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, PLBartForConditionalGeneration

<span class="hljs-meta">&gt;&gt;&gt; </span>model = PLBartForConditionalGeneration.from_pretrained(<span class="hljs-string">&quot;uclanlp/plbart-base&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;uclanlp/plbart-base&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># en_XX is the language symbol id &lt;LID&gt; for English</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>TXT = <span class="hljs-string">&quot;&lt;s&gt; Is 0 the &lt;mask&gt; Fibonacci number ? &lt;/s&gt; en_XX&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>input_ids = tokenizer([TXT], add_special_tokens=<span class="hljs-literal">False</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>).input_ids

<span class="hljs-meta">&gt;&gt;&gt; </span>logits = model(input_ids).logits
<span class="hljs-meta">&gt;&gt;&gt; </span>masked_index = (input_ids[<span class="hljs-number">0</span>] == tokenizer.mask_token_id).nonzero().item()
<span class="hljs-meta">&gt;&gt;&gt; </span>probs = logits[<span class="hljs-number">0</span>, masked_index].softmax(dim=<span class="hljs-number">0</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>values, predictions = probs.topk(<span class="hljs-number">5</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer.decode(predictions).split()
[<span class="hljs-string">&#x27;first&#x27;</span>, <span class="hljs-string">&#x27;same&#x27;</span>, <span class="hljs-string">&#x27;highest&#x27;</span>, <span class="hljs-string">&#x27;result&#x27;</span>, <span class="hljs-string">&#x27;number&#x27;</span>]`,wrap:!1}}),{c(){n=p("p"),n.textContent=k,c=a(),u(l.$$.fragment)},l(t){n=h(t,"P",{"data-svelte-h":!0}),m(n)!=="svelte-kj1i6y"&&(n.textContent=k),c=r(t),f(l.$$.fragment,t)},m(t,v){d(t,n,v),d(t,c,v),g(l,t,v),T=!0},p:Q,i(t){T||(_(l.$$.fragment,t),T=!0)},o(t){b(l.$$.fragment,t),T=!1},d(t){t&&(s(n),s(c)),y(l,t)}}}function $n(M){let n,k=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){n=p("p"),n.innerHTML=k},l(c){n=h(c,"P",{"data-svelte-h":!0}),m(n)!=="svelte-fincs2"&&(n.innerHTML=k)},m(c,l){d(c,n,l)},p:Q,d(c){c&&s(n)}}}function zn(M){let n,k="Example of single-label classification:",c,l,T;return l=new de({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Rva2VuaXplciUyQyUyMFBMQmFydEZvclNlcXVlbmNlQ2xhc3NpZmljYXRpb24lMEElMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJ1Y2xhbmxwJTJGcGxiYXJ0LWJhc2UlMjIpJTBBbW9kZWwlMjAlM0QlMjBQTEJhcnRGb3JTZXF1ZW5jZUNsYXNzaWZpY2F0aW9uLmZyb21fcHJldHJhaW5lZCglMjJ1Y2xhbmxwJTJGcGxiYXJ0LWJhc2UlMjIpJTBBJTBBaW5wdXRzJTIwJTNEJTIwdG9rZW5pemVyKCUyMkhlbGxvJTJDJTIwbXklMjBkb2clMjBpcyUyMGN1dGUlMjIlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQSUwQXdpdGglMjB0b3JjaC5ub19ncmFkKCklM0ElMEElMjAlMjAlMjAlMjBsb2dpdHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cykubG9naXRzJTBBJTBBcHJlZGljdGVkX2NsYXNzX2lkJTIwJTNEJTIwbG9naXRzLmFyZ21heCgpLml0ZW0oKSUwQW1vZGVsLmNvbmZpZy5pZDJsYWJlbCU1QnByZWRpY3RlZF9jbGFzc19pZCU1RCUwQSUwQSUyMyUyMFRvJTIwdHJhaW4lMjBhJTIwbW9kZWwlMjBvbiUyMCU2MG51bV9sYWJlbHMlNjAlMjBjbGFzc2VzJTJDJTIweW91JTIwY2FuJTIwcGFzcyUyMCU2MG51bV9sYWJlbHMlM0RudW1fbGFiZWxzJTYwJTIwdG8lMjAlNjAuZnJvbV9wcmV0cmFpbmVkKC4uLiklNjAlMEFudW1fbGFiZWxzJTIwJTNEJTIwbGVuKG1vZGVsLmNvbmZpZy5pZDJsYWJlbCklMEFtb2RlbCUyMCUzRCUyMFBMQmFydEZvclNlcXVlbmNlQ2xhc3NpZmljYXRpb24uZnJvbV9wcmV0cmFpbmVkKCUyMnVjbGFubHAlMkZwbGJhcnQtYmFzZSUyMiUyQyUyMG51bV9sYWJlbHMlM0RudW1fbGFiZWxzKSUwQSUwQWxhYmVscyUyMCUzRCUyMHRvcmNoLnRlbnNvciglNUIxJTVEKSUwQWxvc3MlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyUyQyUyMGxhYmVscyUzRGxhYmVscykubG9zcyUwQXJvdW5kKGxvc3MuaXRlbSgpJTJDJTIwMik=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, PLBartForSequenceClassification

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;uclanlp/plbart-base&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = PLBartForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;uclanlp/plbart-base&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    logits = model(**inputs).logits

<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_class_id = logits.argmax().item()
<span class="hljs-meta">&gt;&gt;&gt; </span>model.config.id2label[predicted_class_id]
...

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># To train a model on \`num_labels\` classes, you can pass \`num_labels=num_labels\` to \`.from_pretrained(...)\`</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>num_labels = <span class="hljs-built_in">len</span>(model.config.id2label)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = PLBartForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;uclanlp/plbart-base&quot;</span>, num_labels=num_labels)

<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.tensor([<span class="hljs-number">1</span>])
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = model(**inputs, labels=labels).loss
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">round</span>(loss.item(), <span class="hljs-number">2</span>)
...`,wrap:!1}}),{c(){n=p("p"),n.textContent=k,c=a(),u(l.$$.fragment)},l(t){n=h(t,"P",{"data-svelte-h":!0}),m(n)!=="svelte-ykxpe4"&&(n.textContent=k),c=r(t),f(l.$$.fragment,t)},m(t,v){d(t,n,v),d(t,c,v),g(l,t,v),T=!0},p:Q,i(t){T||(_(l.$$.fragment,t),T=!0)},o(t){b(l.$$.fragment,t),T=!1},d(t){t&&(s(n),s(c)),y(l,t)}}}function Fn(M){let n,k="Example of multi-label classification:",c,l,T;return l=new de({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Rva2VuaXplciUyQyUyMFBMQmFydEZvclNlcXVlbmNlQ2xhc3NpZmljYXRpb24lMEElMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJ1Y2xhbmxwJTJGcGxiYXJ0LWJhc2UlMjIpJTBBbW9kZWwlMjAlM0QlMjBQTEJhcnRGb3JTZXF1ZW5jZUNsYXNzaWZpY2F0aW9uLmZyb21fcHJldHJhaW5lZCglMjJ1Y2xhbmxwJTJGcGxiYXJ0LWJhc2UlMjIlMkMlMjBwcm9ibGVtX3R5cGUlM0QlMjJtdWx0aV9sYWJlbF9jbGFzc2lmaWNhdGlvbiUyMiklMEElMEFpbnB1dHMlMjAlM0QlMjB0b2tlbml6ZXIoJTIySGVsbG8lMkMlMjBteSUyMGRvZyUyMGlzJTIwY3V0ZSUyMiUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpJTBBJTBBd2l0aCUyMHRvcmNoLm5vX2dyYWQoKSUzQSUwQSUyMCUyMCUyMCUyMGxvZ2l0cyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzKS5sb2dpdHMlMEElMEFwcmVkaWN0ZWRfY2xhc3NfaWRzJTIwJTNEJTIwdG9yY2guYXJhbmdlKDAlMkMlMjBsb2dpdHMuc2hhcGUlNUItMSU1RCklNUJ0b3JjaC5zaWdtb2lkKGxvZ2l0cykuc3F1ZWV6ZShkaW0lM0QwKSUyMCUzRSUyMDAuNSU1RCUwQSUwQSUyMyUyMFRvJTIwdHJhaW4lMjBhJTIwbW9kZWwlMjBvbiUyMCU2MG51bV9sYWJlbHMlNjAlMjBjbGFzc2VzJTJDJTIweW91JTIwY2FuJTIwcGFzcyUyMCU2MG51bV9sYWJlbHMlM0RudW1fbGFiZWxzJTYwJTIwdG8lMjAlNjAuZnJvbV9wcmV0cmFpbmVkKC4uLiklNjAlMEFudW1fbGFiZWxzJTIwJTNEJTIwbGVuKG1vZGVsLmNvbmZpZy5pZDJsYWJlbCklMEFtb2RlbCUyMCUzRCUyMFBMQmFydEZvclNlcXVlbmNlQ2xhc3NpZmljYXRpb24uZnJvbV9wcmV0cmFpbmVkKCUwQSUyMCUyMCUyMCUyMCUyMnVjbGFubHAlMkZwbGJhcnQtYmFzZSUyMiUyQyUyMG51bV9sYWJlbHMlM0RudW1fbGFiZWxzJTJDJTIwcHJvYmxlbV90eXBlJTNEJTIybXVsdGlfbGFiZWxfY2xhc3NpZmljYXRpb24lMjIlMEEpJTBBJTBBbGFiZWxzJTIwJTNEJTIwdG9yY2guc3VtKCUwQSUyMCUyMCUyMCUyMHRvcmNoLm5uLmZ1bmN0aW9uYWwub25lX2hvdChwcmVkaWN0ZWRfY2xhc3NfaWRzJTVCTm9uZSUyQyUyMCUzQSU1RC5jbG9uZSgpJTJDJTIwbnVtX2NsYXNzZXMlM0RudW1fbGFiZWxzKSUyQyUyMGRpbSUzRDElMEEpLnRvKHRvcmNoLmZsb2F0KSUwQWxvc3MlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyUyQyUyMGxhYmVscyUzRGxhYmVscykubG9zcw==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, PLBartForSequenceClassification

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;uclanlp/plbart-base&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = PLBartForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;uclanlp/plbart-base&quot;</span>, problem_type=<span class="hljs-string">&quot;multi_label_classification&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    logits = model(**inputs).logits

<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_class_ids = torch.arange(<span class="hljs-number">0</span>, logits.shape[-<span class="hljs-number">1</span>])[torch.sigmoid(logits).squeeze(dim=<span class="hljs-number">0</span>) &gt; <span class="hljs-number">0.5</span>]

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># To train a model on \`num_labels\` classes, you can pass \`num_labels=num_labels\` to \`.from_pretrained(...)\`</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>num_labels = <span class="hljs-built_in">len</span>(model.config.id2label)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = PLBartForSequenceClassification.from_pretrained(
<span class="hljs-meta">... </span>    <span class="hljs-string">&quot;uclanlp/plbart-base&quot;</span>, num_labels=num_labels, problem_type=<span class="hljs-string">&quot;multi_label_classification&quot;</span>
<span class="hljs-meta">... </span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.<span class="hljs-built_in">sum</span>(
<span class="hljs-meta">... </span>    torch.nn.functional.one_hot(predicted_class_ids[<span class="hljs-literal">None</span>, :].clone(), num_classes=num_labels), dim=<span class="hljs-number">1</span>
<span class="hljs-meta">... </span>).to(torch.<span class="hljs-built_in">float</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = model(**inputs, labels=labels).loss`,wrap:!1}}),{c(){n=p("p"),n.textContent=k,c=a(),u(l.$$.fragment)},l(t){n=h(t,"P",{"data-svelte-h":!0}),m(n)!=="svelte-1l8e32d"&&(n.textContent=k),c=r(t),f(l.$$.fragment,t)},m(t,v){d(t,n,v),d(t,c,v),g(l,t,v),T=!0},p:Q,i(t){T||(_(l.$$.fragment,t),T=!0)},o(t){b(l.$$.fragment,t),T=!1},d(t){t&&(s(n),s(c)),y(l,t)}}}function Pn(M){let n,k=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){n=p("p"),n.innerHTML=k},l(c){n=h(c,"P",{"data-svelte-h":!0}),m(n)!=="svelte-fincs2"&&(n.innerHTML=k)},m(c,l){d(c,n,l)},p:Q,d(c){c&&s(n)}}}function jn(M){let n,k="Example:",c,l,T;return l=new de({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBQTEJhcnRGb3JDYXVzYWxMTSUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMnVjbGFubHAlMkZwbGJhcnQtYmFzZSUyMiklMEFtb2RlbCUyMCUzRCUyMFBMQmFydEZvckNhdXNhbExNLmZyb21fcHJldHJhaW5lZCglMjJ1Y2xhbmxwJTJGcGxiYXJ0LWJhc2UlMjIlMkMlMjBhZGRfY3Jvc3NfYXR0ZW50aW9uJTNERmFsc2UpJTBBYXNzZXJ0JTIwbW9kZWwuY29uZmlnLmlzX2RlY29kZXIlMkMlMjBmJTIyJTdCbW9kZWwuX19jbGFzc19fJTdEJTIwaGFzJTIwdG8lMjBiZSUyMGNvbmZpZ3VyZWQlMjBhcyUyMGElMjBkZWNvZGVyLiUyMiUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplciglMjJIZWxsbyUyQyUyMG15JTIwZG9nJTIwaXMlMjBjdXRlJTIyJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiklMEFvdXRwdXRzJTIwJTNEJTIwbW9kZWwoKippbnB1dHMpJTBBJTBBbG9naXRzJTIwJTNEJTIwb3V0cHV0cy5sb2dpdHMlMEFleHBlY3RlZF9zaGFwZSUyMCUzRCUyMCU1QjElMkMlMjBpbnB1dHMuaW5wdXRfaWRzLnNoYXBlJTVCLTElNUQlMkMlMjBtb2RlbC5jb25maWcudm9jYWJfc2l6ZSU1RCUwQWxpc3QobG9naXRzLnNoYXBlKSUyMCUzRCUzRCUyMGV4cGVjdGVkX3NoYXBl",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, PLBartForCausalLM

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;uclanlp/plbart-base&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = PLBartForCausalLM.from_pretrained(<span class="hljs-string">&quot;uclanlp/plbart-base&quot;</span>, add_cross_attention=<span class="hljs-literal">False</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">assert</span> model.config.is_decoder, <span class="hljs-string">f&quot;<span class="hljs-subst">{model.__class__}</span> has to be configured as a decoder.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs)

<span class="hljs-meta">&gt;&gt;&gt; </span>logits = outputs.logits
<span class="hljs-meta">&gt;&gt;&gt; </span>expected_shape = [<span class="hljs-number">1</span>, inputs.input_ids.shape[-<span class="hljs-number">1</span>], model.config.vocab_size]
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">list</span>(logits.shape) == expected_shape
<span class="hljs-literal">True</span>`,wrap:!1}}),{c(){n=p("p"),n.textContent=k,c=a(),u(l.$$.fragment)},l(t){n=h(t,"P",{"data-svelte-h":!0}),m(n)!=="svelte-11lpom8"&&(n.textContent=k),c=r(t),f(l.$$.fragment,t)},m(t,v){d(t,n,v),d(t,c,v),g(l,t,v),T=!0},p:Q,i(t){T||(_(l.$$.fragment,t),T=!0)},o(t){b(l.$$.fragment,t),T=!1},d(t){t&&(s(n),s(c)),y(l,t)}}}function Un(M){let n,k,c,l,T,t="<em>This model was released on 2021-03-10 and added to Hugging Face Transformers on 2022-02-18.</em>",v,le,gt,A,Jo='<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/> <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat"/> <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/>',_t,ce,bt,pe,qo=`The PLBART model was proposed in <a href="https://huggingface.co/papers/2103.06333" rel="nofollow">Unified Pre-training for Program Understanding and Generation</a> by Wasi Uddin Ahmad, Saikat Chakraborty, Baishakhi Ray, Kai-Wei Chang.
This is a BART-like model which can be used to perform code-summarization, code-generation, and code-translation tasks. The pre-trained model <code>plbart-base</code> has been trained using multilingual denoising task
on Java, Python and English.`,yt,he,Wo="According to the abstract",Tt,me,Go=`<em>Code summarization and generation empower conversion between programming language (PL) and natural language (NL),
while code translation avails the migration of legacy code from one PL to another. This paper introduces PLBART,
a sequence-to-sequence model capable of performing a broad spectrum of program and language understanding and generation tasks.
PLBART is pre-trained on an extensive collection of Java and Python functions and associated NL text via denoising autoencoding.
Experiments on code summarization in the English language, code generation, and code translation in seven programming languages
show that PLBART outperforms or rivals state-of-the-art models. Moreover, experiments on discriminative tasks, e.g., program
repair, clone detection, and vulnerable code detection, demonstrate PLBART’s effectiveness in program understanding.
Furthermore, analysis reveals that PLBART learns program syntax, style (e.g., identifier naming convention), logical flow
(e.g., if block inside an else block is equivalent to else if block) that are crucial to program semantics and thus excels
even with limited annotations.</em>`,kt,ue,Io='This model was contributed by <a href="https://huggingface.co/gchhablani" rel="nofollow">gchhablani</a>. The Authors’ code can be found <a href="https://github.com/wasiahmad/PLBART" rel="nofollow">here</a>.',vt,fe,Mt,ge,Zo=`PLBart is a multilingual encoder-decoder (sequence-to-sequence) model primarily intended for code-to-text, text-to-code, code-to-code tasks. As the
model is multilingual it expects the sequences in a different format. A special language id token is added in both the
source and target text. The source text format is <code>X [eos, src_lang_code]</code> where <code>X</code> is the source text. The
target text format is <code>[tgt_lang_code] X [eos]</code>. <code>bos</code> is never used.`,wt,_e,Ro='However, for fine-tuning, in some cases no language token is provided in cases where a single language is used. Please refer to <a href="https://huggingface.co/papers/2103.06333" rel="nofollow">the paper</a> to learn more about this.',Ct,be,Xo=`In cases where the language code is needed, the regular <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__"><strong>call</strong>()</a> will encode source text format
when you pass texts as the first argument or with the keyword argument <code>text</code>, and will encode target text format if
it’s passed with the <code>text_target</code> keyword argument.`,Lt,ye,Bt,Te,xt,ke,$t,ve,No=`While generating the target text set the <code>decoder_start_token_id</code> to the target language id. The following
example shows how to translate Python to English using the <code>uclanlp/plbart-python-en_XX</code> model.`,zt,Me,Ft,we,Pt,Ce,So='<li><a href="../tasks/sequence_classification">Text classification task guide</a></li> <li><a href="../tasks/language_modeling">Causal language modeling task guide</a></li> <li><a href="../tasks/translation">Translation task guide</a></li> <li><a href="../tasks/summarization">Summarization task guide</a></li>',jt,Le,Ut,P,Be,Yt,Se,Vo=`This is the configuration class to store the configuration of a <a href="/docs/transformers/v4.56.2/en/model_doc/plbart#transformers.PLBartModel">PLBartModel</a>. It is used to instantiate an
PLBART model according to the specified arguments, defining the model architecture. Instantiating a configuration
with the defaults will yield a similar configuration to that of the PLBART
<a href="https://huggingface.co/uclanlp/plbart-base" rel="nofollow">uclanlp/plbart-base</a> architecture.`,At,Ve,Ho=`Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> and can be used to control the model outputs. Read the
documentation from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> for more information.`,Dt,D,Jt,xe,qt,w,$e,Ot,He,Eo="Construct an PLBART tokenizer.",Kt,Ee,Qo=`Adapted from <a href="/docs/transformers/v4.56.2/en/model_doc/roberta#transformers.RobertaTokenizer">RobertaTokenizer</a> and <a href="/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.XLNetTokenizer">XLNetTokenizer</a>. Based on
<a href="https://github.com/google/sentencepiece" rel="nofollow">SentencePiece</a>.`,eo,Qe,Yo="The tokenization method is <code>&lt;tokens&gt; &lt;eos&gt; &lt;language code&gt;</code> for source language documents, and `&lt;language code&gt;",to,O,oo,I,ze,no,Ye,Ao=`Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
adding special tokens. An PLBART sequence has the following format, where <code>X</code> represents the sequence:`,so,Ae,Do="<li><code>input_ids</code> (for encoder) <code>X [eos, src_lang_code]</code></li> <li><code>decoder_input_ids</code>: (for decoder) <code>X [eos, tgt_lang_code]</code></li>",ao,De,Oo=`BOS is never used. Pairs of sequences are not the expected use case, but they will be handled without a
separator.`,Wt,Fe,Gt,C,Pe,ro,Oe,Ko="The bare Plbart Model outputting raw hidden-states without any specific head on top.",io,Ke,en=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,lo,et,tn=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,co,E,je,po,tt,on='The <a href="/docs/transformers/v4.56.2/en/model_doc/plbart#transformers.PLBartModel">PLBartModel</a> forward method, overrides the <code>__call__</code> special method.',ho,K,It,Ue,Zt,L,Je,mo,ot,nn="The PLBART Model with a language modeling head. Can be used for code-to-text, text-to-code and code-to-code.",uo,nt,sn=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,fo,st,an=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,go,Z,qe,_o,at,rn='The <a href="/docs/transformers/v4.56.2/en/model_doc/plbart#transformers.PLBartForConditionalGeneration">PLBartForConditionalGeneration</a> forward method, overrides the <code>__call__</code> special method.',bo,ee,yo,te,Rt,We,Xt,B,Ge,To,rt,dn=`PLBart model with a sequence classification/head on top (a linear layer on top of the pooled output) e.g.
for GLUE tasks.`,ko,it,ln=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,vo,dt,cn=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Mo,z,Ie,wo,lt,pn='The <a href="/docs/transformers/v4.56.2/en/model_doc/plbart#transformers.PLBartForSequenceClassification">PLBartForSequenceClassification</a> forward method, overrides the <code>__call__</code> special method.',Co,oe,Lo,ne,Bo,se,Nt,Ze,St,x,Re,xo,ct,hn="PLBART decoder with a language modeling head on top (linear layer with weights tied to the input embeddings).",$o,pt,mn=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,zo,ht,un=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Fo,R,Xe,Po,mt,fn='The <a href="/docs/transformers/v4.56.2/en/model_doc/plbart#transformers.PLBartForCausalLM">PLBartForCausalLM</a> forward method, overrides the <code>__call__</code> special method.',jo,ae,Uo,re,Vt,Ne,Ht,ft,Et;return le=new X({props:{title:"PLBart",local:"plbart",headingTag:"h1"}}),ce=new X({props:{title:"Overview",local:"overview",headingTag:"h2"}}),fe=new X({props:{title:"Usage examples",local:"usage-examples",headingTag:"h2"}}),ye=new X({props:{title:"Supervised training",local:"supervised-training",headingTag:"h3"}}),Te=new de({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMFBMQmFydEZvckNvbmRpdGlvbmFsR2VuZXJhdGlvbiUyQyUyMFBMQmFydFRva2VuaXplciUwQSUwQXRva2VuaXplciUyMCUzRCUyMFBMQmFydFRva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIydWNsYW5scCUyRnBsYmFydC1iYXNlJTIyJTJDJTIwc3JjX2xhbmclM0QlMjJlbl9YWCUyMiUyQyUyMHRndF9sYW5nJTNEJTIycHl0aG9uJTIyKSUwQWV4YW1wbGVfcHl0aG9uX3BocmFzZSUyMCUzRCUyMCUyMmRlZiUyMG1heGltdW0oYSUyQ2IlMkNjKSUzQU5FV19MSU5FX0lOREVOVHJldHVybiUyMG1heCglNUJhJTJDYiUyQ2MlNUQpJTIyJTBBZXhwZWN0ZWRfdHJhbnNsYXRpb25fZW5nbGlzaCUyMCUzRCUyMCUyMlJldHVybnMlMjB0aGUlMjBtYXhpbXVtJTIwdmFsdWUlMjBvZiUyMGElMjBiJTIwYy4lMjIlMEFpbnB1dHMlMjAlM0QlMjB0b2tlbml6ZXIoZXhhbXBsZV9weXRob25fcGhyYXNlJTJDJTIwdGV4dF90YXJnZXQlM0RleHBlY3RlZF90cmFuc2xhdGlvbl9lbmdsaXNoJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiklMEFtb2RlbCgqKmlucHV0cyk=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> PLBartForConditionalGeneration, PLBartTokenizer

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = PLBartTokenizer.from_pretrained(<span class="hljs-string">&quot;uclanlp/plbart-base&quot;</span>, src_lang=<span class="hljs-string">&quot;en_XX&quot;</span>, tgt_lang=<span class="hljs-string">&quot;python&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>example_python_phrase = <span class="hljs-string">&quot;def maximum(a,b,c):NEW_LINE_INDENTreturn max([a,b,c])&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>expected_translation_english = <span class="hljs-string">&quot;Returns the maximum value of a b c.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(example_python_phrase, text_target=expected_translation_english, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model(**inputs)`,wrap:!1}}),ke=new X({props:{title:"Generation",local:"generation",headingTag:"h3"}}),Me=new de({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMFBMQmFydEZvckNvbmRpdGlvbmFsR2VuZXJhdGlvbiUyQyUyMFBMQmFydFRva2VuaXplciUwQSUwQXRva2VuaXplciUyMCUzRCUyMFBMQmFydFRva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIydWNsYW5scCUyRnBsYmFydC1weXRob24tZW5fWFglMjIlMkMlMjBzcmNfbGFuZyUzRCUyMnB5dGhvbiUyMiUyQyUyMHRndF9sYW5nJTNEJTIyZW5fWFglMjIpJTBBZXhhbXBsZV9weXRob25fcGhyYXNlJTIwJTNEJTIwJTIyZGVmJTIwbWF4aW11bShhJTJDYiUyQ2MpJTNBTkVXX0xJTkVfSU5ERU5UcmV0dXJuJTIwbWF4KCU1QmElMkNiJTJDYyU1RCklMjIlMEFpbnB1dHMlMjAlM0QlMjB0b2tlbml6ZXIoZXhhbXBsZV9weXRob25fcGhyYXNlJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiklMEFtb2RlbCUyMCUzRCUyMFBMQmFydEZvckNvbmRpdGlvbmFsR2VuZXJhdGlvbi5mcm9tX3ByZXRyYWluZWQoJTIydWNsYW5scCUyRnBsYmFydC1weXRob24tZW5fWFglMjIpJTBBdHJhbnNsYXRlZF90b2tlbnMlMjAlM0QlMjBtb2RlbC5nZW5lcmF0ZSgqKmlucHV0cyUyQyUyMGRlY29kZXJfc3RhcnRfdG9rZW5faWQlM0R0b2tlbml6ZXIubGFuZ19jb2RlX3RvX2lkJTVCJTIyZW5fWFglMjIlNUQpJTBBdG9rZW5pemVyLmJhdGNoX2RlY29kZSh0cmFuc2xhdGVkX3Rva2VucyUyQyUyMHNraXBfc3BlY2lhbF90b2tlbnMlM0RUcnVlKSU1QjAlNUQ=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> PLBartForConditionalGeneration, PLBartTokenizer

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = PLBartTokenizer.from_pretrained(<span class="hljs-string">&quot;uclanlp/plbart-python-en_XX&quot;</span>, src_lang=<span class="hljs-string">&quot;python&quot;</span>, tgt_lang=<span class="hljs-string">&quot;en_XX&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>example_python_phrase = <span class="hljs-string">&quot;def maximum(a,b,c):NEW_LINE_INDENTreturn max([a,b,c])&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(example_python_phrase, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = PLBartForConditionalGeneration.from_pretrained(<span class="hljs-string">&quot;uclanlp/plbart-python-en_XX&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>translated_tokens = model.generate(**inputs, decoder_start_token_id=tokenizer.lang_code_to_id[<span class="hljs-string">&quot;en_XX&quot;</span>])
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer.batch_decode(translated_tokens, skip_special_tokens=<span class="hljs-literal">True</span>)[<span class="hljs-number">0</span>]
<span class="hljs-string">&quot;Returns the maximum value of a b c.&quot;</span>`,wrap:!1}}),we=new X({props:{title:"Resources",local:"resources",headingTag:"h2"}}),Le=new X({props:{title:"PLBartConfig",local:"transformers.PLBartConfig",headingTag:"h2"}}),Be=new H({props:{name:"class transformers.PLBartConfig",anchor:"transformers.PLBartConfig",parameters:[{name:"vocab_size",val:" = 50005"},{name:"max_position_embeddings",val:" = 1024"},{name:"encoder_layers",val:" = 6"},{name:"encoder_ffn_dim",val:" = 3072"},{name:"encoder_attention_heads",val:" = 12"},{name:"decoder_layers",val:" = 6"},{name:"decoder_ffn_dim",val:" = 3072"},{name:"decoder_attention_heads",val:" = 12"},{name:"encoder_layerdrop",val:" = 0.0"},{name:"decoder_layerdrop",val:" = 0.0"},{name:"use_cache",val:" = True"},{name:"is_encoder_decoder",val:" = True"},{name:"activation_function",val:" = 'gelu'"},{name:"d_model",val:" = 768"},{name:"dropout",val:" = 0.1"},{name:"attention_dropout",val:" = 0.1"},{name:"activation_dropout",val:" = 0.0"},{name:"init_std",val:" = 0.02"},{name:"classifier_dropout",val:" = 0.0"},{name:"scale_embedding",val:" = True"},{name:"pad_token_id",val:" = 1"},{name:"bos_token_id",val:" = 0"},{name:"eos_token_id",val:" = 2"},{name:"forced_eos_token_id",val:" = 2"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.PLBartConfig.vocab_size",description:`<strong>vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 50005) &#x2014;
Vocabulary size of the PLBART model. Defines the number of different tokens that can be represented by the
<code>inputs_ids</code> passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/plbart#transformers.PLBartModel">PLBartModel</a>.`,name:"vocab_size"},{anchor:"transformers.PLBartConfig.d_model",description:`<strong>d_model</strong> (<code>int</code>, <em>optional</em>, defaults to 768) &#x2014;
Dimensionality of the layers and the pooler layer.`,name:"d_model"},{anchor:"transformers.PLBartConfig.encoder_layers",description:`<strong>encoder_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 6) &#x2014;
Number of encoder layers.`,name:"encoder_layers"},{anchor:"transformers.PLBartConfig.decoder_layers",description:`<strong>decoder_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 6) &#x2014;
Number of decoder layers.`,name:"decoder_layers"},{anchor:"transformers.PLBartConfig.encoder_attention_heads",description:`<strong>encoder_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 12) &#x2014;
Number of attention heads for each attention layer in the Transformer encoder.`,name:"encoder_attention_heads"},{anchor:"transformers.PLBartConfig.decoder_attention_heads",description:`<strong>decoder_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 12) &#x2014;
Number of attention heads for each attention layer in the Transformer decoder.`,name:"decoder_attention_heads"},{anchor:"transformers.PLBartConfig.decoder_ffn_dim",description:`<strong>decoder_ffn_dim</strong> (<code>int</code>, <em>optional</em>, defaults to 3072) &#x2014;
Dimensionality of the &#x201C;intermediate&#x201D; (often named feed-forward) layer in decoder.`,name:"decoder_ffn_dim"},{anchor:"transformers.PLBartConfig.encoder_ffn_dim",description:`<strong>encoder_ffn_dim</strong> (<code>int</code>, <em>optional</em>, defaults to 3072) &#x2014;
Dimensionality of the &#x201C;intermediate&#x201D; (often named feed-forward) layer in decoder.`,name:"encoder_ffn_dim"},{anchor:"transformers.PLBartConfig.activation_function",description:`<strong>activation_function</strong> (<code>str</code> or <code>function</code>, <em>optional</em>, defaults to <code>&quot;gelu&quot;</code>) &#x2014;
The non-linear activation function (function or string) in the encoder and pooler. If string, <code>&quot;gelu&quot;</code>,
<code>&quot;relu&quot;</code>, <code>&quot;silu&quot;</code> and <code>&quot;gelu_new&quot;</code> are supported.`,name:"activation_function"},{anchor:"transformers.PLBartConfig.dropout",description:`<strong>dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.`,name:"dropout"},{anchor:"transformers.PLBartConfig.attention_dropout",description:`<strong>attention_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout ratio for the attention probabilities.`,name:"attention_dropout"},{anchor:"transformers.PLBartConfig.activation_dropout",description:`<strong>activation_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The dropout ratio for activations inside the fully connected layer.`,name:"activation_dropout"},{anchor:"transformers.PLBartConfig.classifier_dropout",description:`<strong>classifier_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The dropout ratio for classifier.`,name:"classifier_dropout"},{anchor:"transformers.PLBartConfig.max_position_embeddings",description:`<strong>max_position_embeddings</strong> (<code>int</code>, <em>optional</em>, defaults to 1024) &#x2014;
The maximum sequence length that this model might ever be used with. Typically set this to something large
just in case (e.g., 512 or 1024 or 2048).`,name:"max_position_embeddings"},{anchor:"transformers.PLBartConfig.init_std",description:`<strong>init_std</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.`,name:"init_std"},{anchor:"transformers.PLBartConfig.encoder_layerdrop",description:`<strong>encoder_layerdrop</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The LayerDrop probability for the encoder. See the [LayerDrop paper](see <a href="https://huggingface.co/papers/1909.11556" rel="nofollow">https://huggingface.co/papers/1909.11556</a>)
for more details.`,name:"encoder_layerdrop"},{anchor:"transformers.PLBartConfig.decoder_layerdrop",description:`<strong>decoder_layerdrop</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The LayerDrop probability for the decoder. See the [LayerDrop paper](see <a href="https://huggingface.co/papers/1909.11556" rel="nofollow">https://huggingface.co/papers/1909.11556</a>)
for more details.`,name:"decoder_layerdrop"},{anchor:"transformers.PLBartConfig.scale_embedding",description:`<strong>scale_embedding</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Scale embeddings by diving by sqrt(d_model).`,name:"scale_embedding"},{anchor:"transformers.PLBartConfig.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not the model should return the last key/values attentions (not used by all models)`,name:"use_cache"},{anchor:"transformers.PLBartConfig.forced_eos_token_id",description:`<strong>forced_eos_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 2) &#x2014;
The id of the token to force as the last generated token when <code>max_length</code> is reached. Usually set to
<code>eos_token_id</code>.`,name:"forced_eos_token_id"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/plbart/configuration_plbart.py#L28"}}),D=new ut({props:{anchor:"transformers.PLBartConfig.example",$$slots:{default:[wn]},$$scope:{ctx:M}}}),xe=new X({props:{title:"PLBartTokenizer",local:"transformers.PLBartTokenizer",headingTag:"h2"}}),$e=new H({props:{name:"class transformers.PLBartTokenizer",anchor:"transformers.PLBartTokenizer",parameters:[{name:"vocab_file",val:""},{name:"bos_token",val:" = '<s>'"},{name:"eos_token",val:" = '</s>'"},{name:"sep_token",val:" = '</s>'"},{name:"cls_token",val:" = '<s>'"},{name:"unk_token",val:" = '<unk>'"},{name:"pad_token",val:" = '<pad>'"},{name:"mask_token",val:" = '<mask>'"},{name:"language_codes",val:" = 'base'"},{name:"tokenizer_file",val:" = None"},{name:"src_lang",val:" = None"},{name:"tgt_lang",val:" = None"},{name:"sp_model_kwargs",val:": typing.Optional[dict[str, typing.Any]] = None"},{name:"additional_special_tokens",val:" = None"},{name:"clean_up_tokenization_spaces",val:" = True"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.PLBartTokenizer.vocab_file",description:`<strong>vocab_file</strong> (<code>str</code>) &#x2014;
Path to the vocabulary file.`,name:"vocab_file"},{anchor:"transformers.PLBartTokenizer.src_lang",description:`<strong>src_lang</strong> (<code>str</code>, <em>optional</em>) &#x2014;
A string representing the source language.`,name:"src_lang"},{anchor:"transformers.PLBartTokenizer.tgt_lang",description:`<strong>tgt_lang</strong> (<code>str</code>, <em>optional</em>) &#x2014;
A string representing the target language.`,name:"tgt_lang"},{anchor:"transformers.PLBartTokenizer.bos_token",description:`<strong>bos_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;s&gt;&quot;</code>) &#x2014;
The start of sequence token.`,name:"bos_token"},{anchor:"transformers.PLBartTokenizer.eos_token",description:`<strong>eos_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;/s&gt;&quot;</code>) &#x2014;
The end of sequence token.`,name:"eos_token"},{anchor:"transformers.PLBartTokenizer.sep_token",description:`<strong>sep_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;/s&gt;&quot;</code>) &#x2014;
The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
sequence classification or for a text and a question for question answering. It is also used as the last
token of a sequence built with special tokens.`,name:"sep_token"},{anchor:"transformers.PLBartTokenizer.cls_token",description:`<strong>cls_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;s&gt;&quot;</code>) &#x2014;
The cls token, which is a special token used as the first token for all tasks.`,name:"cls_token"},{anchor:"transformers.PLBartTokenizer.unk_token",description:`<strong>unk_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;unk&gt;&quot;</code>) &#x2014;
The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
token instead.`,name:"unk_token"},{anchor:"transformers.PLBartTokenizer.pad_token",description:`<strong>pad_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;pad&gt;&quot;</code>) &#x2014;
The token used for padding, for example when batching sequences of different lengths.`,name:"pad_token"},{anchor:"transformers.PLBartTokenizer.mask_token(str,",description:`<strong>mask_token(<code>str</code>,</strong> <em>optional</em>, defaults to <code>&quot;&lt;mask&gt;&quot;</code>) &#x2014;
The token used for masking values. This is the token used when training this model with masking tasks. This
is only used in the <code>&quot;base&quot;</code> tokenizer type. For <code>&quot;multi&quot;</code> tokenizer, masking is never done for the
downstream tasks.`,name:"mask_token(str,"},{anchor:"transformers.PLBartTokenizer.language_codes",description:`<strong>language_codes</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;base&quot;</code>) &#x2014;
What language codes to use. Should be one of <code>&quot;base&quot;</code> or <code>&quot;multi&quot;</code>.`,name:"language_codes"},{anchor:"transformers.PLBartTokenizer.sp_model_kwargs",description:`<strong>sp_model_kwargs</strong> (<code>dict</code>, <em>optional</em>) &#x2014;
Will be passed to the <code>SentencePieceProcessor.__init__()</code> method. The <a href="https://github.com/google/sentencepiece/tree/master/python" rel="nofollow">Python wrapper for
SentencePiece</a> can be used, among other things,
to set:<ul>
<li><code>enable_sampling</code>: Enable subword regularization.</li>
<li><code>nbest_size</code>: Sampling parameters for unigram. Invalid for BPE-Dropout.<ul>
<li><code>nbest_size = {0,1}</code>: No sampling is performed.</li>
<li><code>nbest_size &gt; 1</code>: samples from the nbest_size results.</li>
<li><code>nbest_size &lt; 0</code>: assuming that nbest_size is infinite and samples from the all hypothesis (lattice)
using forward-filtering-and-backward-sampling algorithm.</li>
</ul></li>
<li><code>alpha</code>: Smoothing parameter for unigram sampling, and dropout probability of merge operations for
BPE-dropout.</li>
</ul>`,name:"sp_model_kwargs"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/plbart/tokenization_plbart.py#L51"}}),O=new ut({props:{anchor:"transformers.PLBartTokenizer.example",$$slots:{default:[Cn]},$$scope:{ctx:M}}}),ze=new H({props:{name:"build_inputs_with_special_tokens",anchor:"transformers.PLBartTokenizer.build_inputs_with_special_tokens",parameters:[{name:"token_ids_0",val:": list"},{name:"token_ids_1",val:": typing.Optional[list[int]] = None"}],parametersDescription:[{anchor:"transformers.PLBartTokenizer.build_inputs_with_special_tokens.token_ids_0",description:`<strong>token_ids_0</strong> (<code>list[int]</code>) &#x2014;
List of IDs to which the special tokens will be added.`,name:"token_ids_0"},{anchor:"transformers.PLBartTokenizer.build_inputs_with_special_tokens.token_ids_1",description:`<strong>token_ids_1</strong> (<code>list[int]</code>, <em>optional</em>) &#x2014;
Optional second list of IDs for sequence pairs.`,name:"token_ids_1"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/plbart/tokenization_plbart.py#L278",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>List of <a href="../glossary#input-ids">input IDs</a> with the appropriate special tokens.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>list[int]</code></p>
`}}),Fe=new X({props:{title:"PLBartModel",local:"transformers.PLBartModel",headingTag:"h2"}}),Pe=new H({props:{name:"class transformers.PLBartModel",anchor:"transformers.PLBartModel",parameters:[{name:"config",val:": PLBartConfig"}],parametersDescription:[{anchor:"transformers.PLBartModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/plbart#transformers.PLBartConfig">PLBartConfig</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/plbart/modeling_plbart.py#L1097"}}),je=new H({props:{name:"forward",anchor:"transformers.PLBartModel.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.LongTensor] = None"},{name:"decoder_input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"decoder_attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"decoder_head_mask",val:": typing.Optional[torch.LongTensor] = None"},{name:"cross_attn_head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_outputs",val:": typing.Optional[list[torch.FloatTensor]] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"decoder_inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"}],parametersDescription:[{anchor:"transformers.PLBartModel.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.PLBartModel.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.PLBartModel.forward.decoder_input_ids",description:`<strong>decoder_input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of decoder input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a> or <code>PLBartMultiTokenizer</code> depending on the checkpoint.
See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#decoder-input-ids">What are decoder input IDs?</a></p>
<p>PLBart uses a specific language id token as the starting token for <code>decoder_input_ids</code> generation that
varies according to source and target language, <em>e.g.</em> 50003 for <em>en_XX</em>, and 50001 for <em>java</em>. If
<code>past_key_values</code> is used, optionally only the last <code>decoder_input_ids</code> have to be input (see
<code>past_key_values</code>).</p>
<p>For translation and summarization training, <code>decoder_input_ids</code> should be provided. If no
<code>decoder_input_ids</code> is provided, the model will create this tensor by shifting the <code>input_ids</code> to the right
for denoising pre-training following the paper.`,name:"decoder_input_ids"},{anchor:"transformers.PLBartModel.forward.decoder_attention_mask",description:"<strong>decoder_attention_mask</strong> (` &#x2014;",name:"decoder_attention_mask"},{anchor:"transformers.PLBartModel.forward.`",description:`<strong>\`</strong>    obj &#x2014;<em>torch.LongTensor</em> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>):
Default behavior:
generate a tensor that ignores pad tokens in <code>decoder_input_ids</code>. Causal mask will also be used by default.`,name:"`"},{anchor:"transformers.PLBartModel.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.PLBartModel.forward.decoder_head_mask",description:`<strong>decoder_head_mask</strong> (<code>torch.LongTensor</code> of shape <code>(decoder_layers, decoder_attention_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the attention modules in the decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"decoder_head_mask"},{anchor:"transformers.PLBartModel.forward.cross_attn_head_mask",description:"<strong>cross_attn_head_mask</strong> (` &#x2014;",name:"cross_attn_head_mask"},{anchor:"transformers.PLBartModel.forward.`",description:`<strong>\`</strong>    obj &#x2014;<em>torch.Tensor</em> of shape <code>(decoder_layers, decoder_attention_heads)</code>, <em>optional</em>):
Mask to nullify
selected heads of the cross-attention modules in the decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"`"},{anchor:"transformers.PLBartModel.forward.encoder_outputs",description:`<strong>encoder_outputs</strong> (<code>list[torch.FloatTensor]</code>, <em>optional</em>) &#x2014;
Tuple consists of (<code>last_hidden_state</code>, <em>optional</em>: <code>hidden_states</code>, <em>optional</em>: <code>attentions</code>)
<code>last_hidden_state</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) is a sequence of
hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.`,name:"encoder_outputs"},{anchor:"transformers.PLBartModel.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.PLBartModel.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.PLBartModel.forward.decoder_inputs_embeds",description:`<strong>decoder_inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, target_sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>decoder_input_ids</code> you can choose to directly pass an embedded
representation. If <code>past_key_values</code> is used, optionally only the last <code>decoder_inputs_embeds</code> have to be
input (see <code>past_key_values</code>). This is useful if you want more control over how to convert
<code>decoder_input_ids</code> indices into associated vectors than the model&#x2019;s internal embedding lookup matrix.</p>
<p>If <code>decoder_input_ids</code> and <code>decoder_inputs_embeds</code> are both unset, <code>decoder_inputs_embeds</code> takes the value
of <code>inputs_embeds</code>.`,name:"decoder_inputs_embeds"},{anchor:"transformers.PLBartModel.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.PLBartModel.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.PLBartModel.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.PLBartModel.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.PLBartModel.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/plbart/modeling_plbart.py#L1128",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqModelOutput"
>transformers.modeling_outputs.Seq2SeqModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/plbart#transformers.PLBartConfig"
>PLBartConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>last_hidden_state</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>) — Sequence of hidden-states at the output of the last layer of the decoder of the model.</p>
<p>If <code>past_key_values</code> is used only the last hidden-state of the sequences of shape <code>(batch_size, 1, hidden_size)</code> is output.</p>
</li>
<li>
<p><strong>past_key_values</strong> (<code>EncoderDecoderCache</code>, <em>optional</em>, returned when <code>use_cache=True</code> is passed or when <code>config.use_cache=True</code>) — It is a <a
  href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.EncoderDecoderCache"
>EncoderDecoderCache</a> instance. For more details, see our <a
  href="https://huggingface.co/docs/transformers/en/kv_cache"
  rel="nofollow"
>kv cache guide</a>.</p>
<p>Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used (see <code>past_key_values</code> input) to speed up sequential decoding.</p>
</li>
<li>
<p><strong>decoder_hidden_states</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings, if the model has an embedding layer, +
one for the output of each layer) of shape <code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the decoder at the output of each layer plus the optional initial embedding outputs.</p>
</li>
<li>
<p><strong>decoder_attentions</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
self-attention heads.</p>
</li>
<li>
<p><strong>cross_attentions</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the
weighted average in the cross-attention heads.</p>
</li>
<li>
<p><strong>encoder_last_hidden_state</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) — Sequence of hidden-states at the output of the last layer of the encoder of the model.</p>
</li>
<li>
<p><strong>encoder_hidden_states</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings, if the model has an embedding layer, +
one for the output of each layer) of shape <code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the encoder at the output of each layer plus the optional initial embedding outputs.</p>
</li>
<li>
<p><strong>encoder_attentions</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
self-attention heads.</p>
</li>
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqModelOutput"
>transformers.modeling_outputs.Seq2SeqModelOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),K=new Qt({props:{$$slots:{default:[Ln]},$$scope:{ctx:M}}}),Ue=new X({props:{title:"PLBartForConditionalGeneration",local:"transformers.PLBartForConditionalGeneration",headingTag:"h2"}}),Je=new H({props:{name:"class transformers.PLBartForConditionalGeneration",anchor:"transformers.PLBartForConditionalGeneration",parameters:[{name:"config",val:": PLBartConfig"}],parametersDescription:[{anchor:"transformers.PLBartForConditionalGeneration.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/plbart#transformers.PLBartConfig">PLBartConfig</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/plbart/modeling_plbart.py#L1244"}}),qe=new H({props:{name:"forward",anchor:"transformers.PLBartForConditionalGeneration.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.LongTensor] = None"},{name:"decoder_input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"decoder_attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"decoder_head_mask",val:": typing.Optional[torch.LongTensor] = None"},{name:"cross_attn_head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_outputs",val:": typing.Optional[list[torch.FloatTensor]] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"decoder_inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.Tensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"}],parametersDescription:[{anchor:"transformers.PLBartForConditionalGeneration.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.PLBartForConditionalGeneration.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.PLBartForConditionalGeneration.forward.decoder_input_ids",description:`<strong>decoder_input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of decoder input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a> or <code>PLBartMultiTokenizer</code> depending on the checkpoint.
See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#decoder-input-ids">What are decoder input IDs?</a></p>
<p>PLBart uses a specific language id token as the starting token for <code>decoder_input_ids</code> generation that
varies according to source and target language, <em>e.g.</em> 50003 for <em>en_XX</em>, and 50001 for <em>java</em>. If
<code>past_key_values</code> is used, optionally only the last <code>decoder_input_ids</code> have to be input (see
<code>past_key_values</code>).</p>
<p>For translation and summarization training, <code>decoder_input_ids</code> should be provided. If no
<code>decoder_input_ids</code> is provided, the model will create this tensor by shifting the <code>input_ids</code> to the right
for denoising pre-training following the paper.`,name:"decoder_input_ids"},{anchor:"transformers.PLBartForConditionalGeneration.forward.decoder_attention_mask",description:"<strong>decoder_attention_mask</strong> (` &#x2014;",name:"decoder_attention_mask"},{anchor:"transformers.PLBartForConditionalGeneration.forward.`",description:`<strong>\`</strong>    obj &#x2014;<em>torch.LongTensor</em> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>):
Default behavior:
generate a tensor that ignores pad tokens in <code>decoder_input_ids</code>. Causal mask will also be used by default.`,name:"`"},{anchor:"transformers.PLBartForConditionalGeneration.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.PLBartForConditionalGeneration.forward.decoder_head_mask",description:`<strong>decoder_head_mask</strong> (<code>torch.LongTensor</code> of shape <code>(decoder_layers, decoder_attention_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the attention modules in the decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"decoder_head_mask"},{anchor:"transformers.PLBartForConditionalGeneration.forward.cross_attn_head_mask",description:"<strong>cross_attn_head_mask</strong> (` &#x2014;",name:"cross_attn_head_mask"},{anchor:"transformers.PLBartForConditionalGeneration.forward.`",description:`<strong>\`</strong>    obj &#x2014;<em>torch.Tensor</em> of shape <code>(decoder_layers, decoder_attention_heads)</code>, <em>optional</em>):
Mask to nullify
selected heads of the cross-attention modules in the decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"`"},{anchor:"transformers.PLBartForConditionalGeneration.forward.encoder_outputs",description:`<strong>encoder_outputs</strong> (<code>list[torch.FloatTensor]</code>, <em>optional</em>) &#x2014;
Tuple consists of (<code>last_hidden_state</code>, <em>optional</em>: <code>hidden_states</code>, <em>optional</em>: <code>attentions</code>)
<code>last_hidden_state</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) is a sequence of
hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.`,name:"encoder_outputs"},{anchor:"transformers.PLBartForConditionalGeneration.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.PLBartForConditionalGeneration.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.PLBartForConditionalGeneration.forward.decoder_inputs_embeds",description:`<strong>decoder_inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, target_sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>decoder_input_ids</code> you can choose to directly pass an embedded
representation. If <code>past_key_values</code> is used, optionally only the last <code>decoder_inputs_embeds</code> have to be
input (see <code>past_key_values</code>). This is useful if you want more control over how to convert
<code>decoder_input_ids</code> indices into associated vectors than the model&#x2019;s internal embedding lookup matrix.</p>
<p>If <code>decoder_input_ids</code> and <code>decoder_inputs_embeds</code> are both unset, <code>decoder_inputs_embeds</code> takes the value
of <code>inputs_embeds</code>.`,name:"decoder_inputs_embeds"},{anchor:"transformers.PLBartForConditionalGeneration.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.PLBartForConditionalGeneration.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.PLBartForConditionalGeneration.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.PLBartForConditionalGeneration.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.PLBartForConditionalGeneration.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.PLBartForConditionalGeneration.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/plbart/modeling_plbart.py#L1279",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqLMOutput"
>transformers.modeling_outputs.Seq2SeqLMOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/plbart#transformers.PLBartConfig"
>PLBartConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>loss</strong> (<code>torch.FloatTensor</code> of shape <code>(1,)</code>, <em>optional</em>, returned when <code>labels</code> is provided) — Language modeling loss.</p>
</li>
<li>
<p><strong>logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, config.vocab_size)</code>) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).</p>
</li>
<li>
<p><strong>past_key_values</strong> (<code>EncoderDecoderCache</code>, <em>optional</em>, returned when <code>use_cache=True</code> is passed or when <code>config.use_cache=True</code>) — It is a <a
  href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.EncoderDecoderCache"
>EncoderDecoderCache</a> instance. For more details, see our <a
  href="https://huggingface.co/docs/transformers/en/kv_cache"
  rel="nofollow"
>kv cache guide</a>.</p>
<p>Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used (see <code>past_key_values</code> input) to speed up sequential decoding.</p>
</li>
<li>
<p><strong>decoder_hidden_states</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings, if the model has an embedding layer, +
one for the output of each layer) of shape <code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.</p>
</li>
<li>
<p><strong>decoder_attentions</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
self-attention heads.</p>
</li>
<li>
<p><strong>cross_attentions</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the
weighted average in the cross-attention heads.</p>
</li>
<li>
<p><strong>encoder_last_hidden_state</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) — Sequence of hidden-states at the output of the last layer of the encoder of the model.</p>
</li>
<li>
<p><strong>encoder_hidden_states</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings, if the model has an embedding layer, +
one for the output of each layer) of shape <code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.</p>
</li>
<li>
<p><strong>encoder_attentions</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
self-attention heads.</p>
</li>
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqLMOutput"
>transformers.modeling_outputs.Seq2SeqLMOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),ee=new Qt({props:{$$slots:{default:[Bn]},$$scope:{ctx:M}}}),te=new ut({props:{anchor:"transformers.PLBartForConditionalGeneration.forward.example",$$slots:{default:[xn]},$$scope:{ctx:M}}}),We=new X({props:{title:"PLBartForSequenceClassification",local:"transformers.PLBartForSequenceClassification",headingTag:"h2"}}),Ge=new H({props:{name:"class transformers.PLBartForSequenceClassification",anchor:"transformers.PLBartForSequenceClassification",parameters:[{name:"config",val:": PLBartConfig"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.PLBartForSequenceClassification.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/plbart#transformers.PLBartConfig">PLBartConfig</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/plbart/modeling_plbart.py#L1436"}}),Ie=new H({props:{name:"forward",anchor:"transformers.PLBartForSequenceClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"decoder_input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"decoder_attention_mask",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"decoder_head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"cross_attn_head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_outputs",val:": typing.Optional[list[torch.FloatTensor]] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"decoder_inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"}],parametersDescription:[{anchor:"transformers.PLBartForSequenceClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.PLBartForSequenceClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.PLBartForSequenceClassification.forward.decoder_input_ids",description:`<strong>decoder_input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of decoder input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a> or <code>PLBartMultiTokenizer</code> depending on the checkpoint.
See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#decoder-input-ids">What are decoder input IDs?</a></p>
<p>PLBart uses a specific language id token as the starting token for <code>decoder_input_ids</code> generation that
varies according to source and target language, <em>e.g.</em> 50003 for <em>en_XX</em>, and 50001 for <em>java</em>. If
<code>past_key_values</code> is used, optionally only the last <code>decoder_input_ids</code> have to be input (see
<code>past_key_values</code>).</p>
<p>For translation and summarization training, <code>decoder_input_ids</code> should be provided. If no
<code>decoder_input_ids</code> is provided, the model will create this tensor by shifting the <code>input_ids</code> to the right
for denoising pre-training following the paper.`,name:"decoder_input_ids"},{anchor:"transformers.PLBartForSequenceClassification.forward.decoder_attention_mask",description:"<strong>decoder_attention_mask</strong> (` &#x2014;",name:"decoder_attention_mask"},{anchor:"transformers.PLBartForSequenceClassification.forward.`",description:`<strong>\`</strong>    obj &#x2014;<em>torch.LongTensor</em> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>):
Default behavior:
generate a tensor that ignores pad tokens in <code>decoder_input_ids</code>. Causal mask will also be used by default.`,name:"`"},{anchor:"transformers.PLBartForSequenceClassification.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.PLBartForSequenceClassification.forward.decoder_head_mask",description:`<strong>decoder_head_mask</strong> (<code>torch.Tensor</code> of shape <code>(decoder_layers, decoder_attention_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the attention modules in the decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"decoder_head_mask"},{anchor:"transformers.PLBartForSequenceClassification.forward.cross_attn_head_mask",description:"<strong>cross_attn_head_mask</strong> (` &#x2014;",name:"cross_attn_head_mask"},{anchor:"transformers.PLBartForSequenceClassification.forward.`",description:`<strong>\`</strong>    obj &#x2014;<em>torch.Tensor</em> of shape <code>(decoder_layers, decoder_attention_heads)</code>, <em>optional</em>):
Mask to nullify
selected heads of the cross-attention modules in the decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"`"},{anchor:"transformers.PLBartForSequenceClassification.forward.encoder_outputs",description:`<strong>encoder_outputs</strong> (<code>list[torch.FloatTensor]</code>, <em>optional</em>) &#x2014;
Tuple consists of (<code>last_hidden_state</code>, <em>optional</em>: <code>hidden_states</code>, <em>optional</em>: <code>attentions</code>)
<code>last_hidden_state</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) is a sequence of
hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.`,name:"encoder_outputs"},{anchor:"transformers.PLBartForSequenceClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.PLBartForSequenceClassification.forward.decoder_inputs_embeds",description:`<strong>decoder_inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, target_sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>decoder_input_ids</code> you can choose to directly pass an embedded
representation. If <code>past_key_values</code> is used, optionally only the last <code>decoder_inputs_embeds</code> have to be
input (see <code>past_key_values</code>). This is useful if you want more control over how to convert
<code>decoder_input_ids</code> indices into associated vectors than the model&#x2019;s internal embedding lookup matrix.</p>
<p>If <code>decoder_input_ids</code> and <code>decoder_inputs_embeds</code> are both unset, <code>decoder_inputs_embeds</code> takes the value
of <code>inputs_embeds</code>.`,name:"decoder_inputs_embeds"},{anchor:"transformers.PLBartForSequenceClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the sequence classification/regression loss. Indices should be in <code>[0, ..., config.num_labels - 1]</code>. If <code>config.num_labels &gt; 1</code> a classification loss is computed (Cross-Entropy).`,name:"labels"},{anchor:"transformers.PLBartForSequenceClassification.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.PLBartForSequenceClassification.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.PLBartForSequenceClassification.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.PLBartForSequenceClassification.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.PLBartForSequenceClassification.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/plbart/modeling_plbart.py#L1452",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqSequenceClassifierOutput"
>transformers.modeling_outputs.Seq2SeqSequenceClassifierOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/plbart#transformers.PLBartConfig"
>PLBartConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>loss</strong> (<code>torch.FloatTensor</code> of shape <code>(1,)</code>, <em>optional</em>, returned when <code>label</code> is provided) — Classification (or regression if config.num_labels==1) loss.</p>
</li>
<li>
<p><strong>logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, config.num_labels)</code>) — Classification (or regression if config.num_labels==1) scores (before SoftMax).</p>
</li>
<li>
<p><strong>past_key_values</strong> (<code>EncoderDecoderCache</code>, <em>optional</em>, returned when <code>use_cache=True</code> is passed or when <code>config.use_cache=True</code>) — It is a <a
  href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.EncoderDecoderCache"
>EncoderDecoderCache</a> instance. For more details, see our <a
  href="https://huggingface.co/docs/transformers/en/kv_cache"
  rel="nofollow"
>kv cache guide</a>.</p>
<p>Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used (see <code>past_key_values</code> input) to speed up sequential decoding.</p>
</li>
<li>
<p><strong>decoder_hidden_states</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings, if the model has an embedding layer, +
one for the output of each layer) of shape <code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.</p>
</li>
<li>
<p><strong>decoder_attentions</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
self-attention heads.</p>
</li>
<li>
<p><strong>cross_attentions</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the
weighted average in the cross-attention heads.</p>
</li>
<li>
<p><strong>encoder_last_hidden_state</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) — Sequence of hidden-states at the output of the last layer of the encoder of the model.</p>
</li>
<li>
<p><strong>encoder_hidden_states</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings, if the model has an embedding layer, +
one for the output of each layer) of shape <code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.</p>
</li>
<li>
<p><strong>encoder_attentions</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
self-attention heads.</p>
</li>
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqSequenceClassifierOutput"
>transformers.modeling_outputs.Seq2SeqSequenceClassifierOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),oe=new Qt({props:{$$slots:{default:[$n]},$$scope:{ctx:M}}}),ne=new ut({props:{anchor:"transformers.PLBartForSequenceClassification.forward.example",$$slots:{default:[zn]},$$scope:{ctx:M}}}),se=new ut({props:{anchor:"transformers.PLBartForSequenceClassification.forward.example-2",$$slots:{default:[Fn]},$$scope:{ctx:M}}}),Ze=new X({props:{title:"PLBartForCausalLM",local:"transformers.PLBartForCausalLM",headingTag:"h2"}}),Re=new H({props:{name:"class transformers.PLBartForCausalLM",anchor:"transformers.PLBartForCausalLM",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.PLBartForCausalLM.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/plbart#transformers.PLBartForCausalLM">PLBartForCausalLM</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/plbart/modeling_plbart.py#L1600"}}),Xe=new H({props:{name:"forward",anchor:"transformers.PLBartForCausalLM.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_hidden_states",val:": typing.Optional[torch.FloatTensor] = None"},{name:"encoder_attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"cross_attn_head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"}],parametersDescription:[{anchor:"transformers.PLBartForCausalLM.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.PLBartForCausalLM.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.PLBartForCausalLM.forward.encoder_hidden_states",description:`<strong>encoder_hidden_states</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
if the model is configured as a decoder.`,name:"encoder_hidden_states"},{anchor:"transformers.PLBartForCausalLM.forward.encoder_attention_mask",description:`<strong>encoder_attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
the cross-attention if the model is configured as a decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>`,name:"encoder_attention_mask"},{anchor:"transformers.PLBartForCausalLM.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.PLBartForCausalLM.forward.cross_attn_head_mask",description:`<strong>cross_attn_head_mask</strong> (<code>torch.Tensor</code> of shape <code>(decoder_layers, decoder_attention_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the cross-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"cross_attn_head_mask"},{anchor:"transformers.PLBartForCausalLM.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.PLBartForCausalLM.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.PLBartForCausalLM.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.PLBartForCausalLM.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.PLBartForCausalLM.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.PLBartForCausalLM.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.PLBartForCausalLM.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.PLBartForCausalLM.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/plbart/modeling_plbart.py#L1626",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithCrossAttentions"
>transformers.modeling_outputs.CausalLMOutputWithCrossAttentions</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/plbart#transformers.PLBartConfig"
>PLBartConfig</a>) and inputs.</p>
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
`}}),ae=new Qt({props:{$$slots:{default:[Pn]},$$scope:{ctx:M}}}),re=new ut({props:{anchor:"transformers.PLBartForCausalLM.forward.example",$$slots:{default:[jn]},$$scope:{ctx:M}}}),Ne=new Mn({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/plbart.md"}}),{c(){n=p("meta"),k=a(),c=p("p"),l=a(),T=p("p"),T.innerHTML=t,v=a(),u(le.$$.fragment),gt=a(),A=p("div"),A.innerHTML=Jo,_t=a(),u(ce.$$.fragment),bt=a(),pe=p("p"),pe.innerHTML=qo,yt=a(),he=p("p"),he.textContent=Wo,Tt=a(),me=p("p"),me.innerHTML=Go,kt=a(),ue=p("p"),ue.innerHTML=Io,vt=a(),u(fe.$$.fragment),Mt=a(),ge=p("p"),ge.innerHTML=Zo,wt=a(),_e=p("p"),_e.innerHTML=Ro,Ct=a(),be=p("p"),be.innerHTML=Xo,Lt=a(),u(ye.$$.fragment),Bt=a(),u(Te.$$.fragment),xt=a(),u(ke.$$.fragment),$t=a(),ve=p("p"),ve.innerHTML=No,zt=a(),u(Me.$$.fragment),Ft=a(),u(we.$$.fragment),Pt=a(),Ce=p("ul"),Ce.innerHTML=So,jt=a(),u(Le.$$.fragment),Ut=a(),P=p("div"),u(Be.$$.fragment),Yt=a(),Se=p("p"),Se.innerHTML=Vo,At=a(),Ve=p("p"),Ve.innerHTML=Ho,Dt=a(),u(D.$$.fragment),Jt=a(),u(xe.$$.fragment),qt=a(),w=p("div"),u($e.$$.fragment),Ot=a(),He=p("p"),He.textContent=Eo,Kt=a(),Ee=p("p"),Ee.innerHTML=Qo,eo=a(),Qe=p("p"),Qe.innerHTML=Yo,to=Tn(`
<tokens> <eos>\` for target language documents.
`),u(O.$$.fragment),oo=a(),I=p("div"),u(ze.$$.fragment),no=a(),Ye=p("p"),Ye.innerHTML=Ao,so=a(),Ae=p("ul"),Ae.innerHTML=Do,ao=a(),De=p("p"),De.textContent=Oo,Wt=a(),u(Fe.$$.fragment),Gt=a(),C=p("div"),u(Pe.$$.fragment),ro=a(),Oe=p("p"),Oe.textContent=Ko,io=a(),Ke=p("p"),Ke.innerHTML=en,lo=a(),et=p("p"),et.innerHTML=tn,co=a(),E=p("div"),u(je.$$.fragment),po=a(),tt=p("p"),tt.innerHTML=on,ho=a(),u(K.$$.fragment),It=a(),u(Ue.$$.fragment),Zt=a(),L=p("div"),u(Je.$$.fragment),mo=a(),ot=p("p"),ot.textContent=nn,uo=a(),nt=p("p"),nt.innerHTML=sn,fo=a(),st=p("p"),st.innerHTML=an,go=a(),Z=p("div"),u(qe.$$.fragment),_o=a(),at=p("p"),at.innerHTML=rn,bo=a(),u(ee.$$.fragment),yo=a(),u(te.$$.fragment),Rt=a(),u(We.$$.fragment),Xt=a(),B=p("div"),u(Ge.$$.fragment),To=a(),rt=p("p"),rt.textContent=dn,ko=a(),it=p("p"),it.innerHTML=ln,vo=a(),dt=p("p"),dt.innerHTML=cn,Mo=a(),z=p("div"),u(Ie.$$.fragment),wo=a(),lt=p("p"),lt.innerHTML=pn,Co=a(),u(oe.$$.fragment),Lo=a(),u(ne.$$.fragment),Bo=a(),u(se.$$.fragment),Nt=a(),u(Ze.$$.fragment),St=a(),x=p("div"),u(Re.$$.fragment),xo=a(),ct=p("p"),ct.textContent=hn,$o=a(),pt=p("p"),pt.innerHTML=mn,zo=a(),ht=p("p"),ht.innerHTML=un,Fo=a(),R=p("div"),u(Xe.$$.fragment),Po=a(),mt=p("p"),mt.innerHTML=fn,jo=a(),u(ae.$$.fragment),Uo=a(),u(re.$$.fragment),Vt=a(),u(Ne.$$.fragment),Ht=a(),ft=p("p"),this.h()},l(e){const o=kn("svelte-u9bgzb",document.head);n=h(o,"META",{name:!0,content:!0}),o.forEach(s),k=r(e),c=h(e,"P",{}),G(c).forEach(s),l=r(e),T=h(e,"P",{"data-svelte-h":!0}),m(T)!=="svelte-bzyei4"&&(T.innerHTML=t),v=r(e),f(le.$$.fragment,e),gt=r(e),A=h(e,"DIV",{class:!0,"data-svelte-h":!0}),m(A)!=="svelte-b95w5j"&&(A.innerHTML=Jo),_t=r(e),f(ce.$$.fragment,e),bt=r(e),pe=h(e,"P",{"data-svelte-h":!0}),m(pe)!=="svelte-154dlad"&&(pe.innerHTML=qo),yt=r(e),he=h(e,"P",{"data-svelte-h":!0}),m(he)!=="svelte-16emyqs"&&(he.textContent=Wo),Tt=r(e),me=h(e,"P",{"data-svelte-h":!0}),m(me)!=="svelte-dauvff"&&(me.innerHTML=Go),kt=r(e),ue=h(e,"P",{"data-svelte-h":!0}),m(ue)!=="svelte-1fckswv"&&(ue.innerHTML=Io),vt=r(e),f(fe.$$.fragment,e),Mt=r(e),ge=h(e,"P",{"data-svelte-h":!0}),m(ge)!=="svelte-cjf5hx"&&(ge.innerHTML=Zo),wt=r(e),_e=h(e,"P",{"data-svelte-h":!0}),m(_e)!=="svelte-uw04bg"&&(_e.innerHTML=Ro),Ct=r(e),be=h(e,"P",{"data-svelte-h":!0}),m(be)!=="svelte-xagr88"&&(be.innerHTML=Xo),Lt=r(e),f(ye.$$.fragment,e),Bt=r(e),f(Te.$$.fragment,e),xt=r(e),f(ke.$$.fragment,e),$t=r(e),ve=h(e,"P",{"data-svelte-h":!0}),m(ve)!=="svelte-1caunq5"&&(ve.innerHTML=No),zt=r(e),f(Me.$$.fragment,e),Ft=r(e),f(we.$$.fragment,e),Pt=r(e),Ce=h(e,"UL",{"data-svelte-h":!0}),m(Ce)!=="svelte-1wtorqd"&&(Ce.innerHTML=So),jt=r(e),f(Le.$$.fragment,e),Ut=r(e),P=h(e,"DIV",{class:!0});var N=G(P);f(Be.$$.fragment,N),Yt=r(N),Se=h(N,"P",{"data-svelte-h":!0}),m(Se)!=="svelte-1q8xf78"&&(Se.innerHTML=Vo),At=r(N),Ve=h(N,"P",{"data-svelte-h":!0}),m(Ve)!=="svelte-1ek1ss9"&&(Ve.innerHTML=Ho),Dt=r(N),f(D.$$.fragment,N),N.forEach(s),Jt=r(e),f(xe.$$.fragment,e),qt=r(e),w=h(e,"DIV",{class:!0});var $=G(w);f($e.$$.fragment,$),Ot=r($),He=h($,"P",{"data-svelte-h":!0}),m(He)!=="svelte-myrn70"&&(He.textContent=Eo),Kt=r($),Ee=h($,"P",{"data-svelte-h":!0}),m(Ee)!=="svelte-19vr0qz"&&(Ee.innerHTML=Qo),eo=r($),Qe=h($,"P",{"data-svelte-h":!0}),m(Qe)!=="svelte-1i8rh37"&&(Qe.innerHTML=Yo),to=vn($,`
<tokens> <eos>\` for target language documents.
`),f(O.$$.fragment,$),oo=r($),I=h($,"DIV",{class:!0});var S=G(I);f(ze.$$.fragment,S),no=r(S),Ye=h(S,"P",{"data-svelte-h":!0}),m(Ye)!=="svelte-1pxl1oh"&&(Ye.innerHTML=Ao),so=r(S),Ae=h(S,"UL",{"data-svelte-h":!0}),m(Ae)!=="svelte-mlrsks"&&(Ae.innerHTML=Do),ao=r(S),De=h(S,"P",{"data-svelte-h":!0}),m(De)!=="svelte-46aam0"&&(De.textContent=Oo),S.forEach(s),$.forEach(s),Wt=r(e),f(Fe.$$.fragment,e),Gt=r(e),C=h(e,"DIV",{class:!0});var j=G(C);f(Pe.$$.fragment,j),ro=r(j),Oe=h(j,"P",{"data-svelte-h":!0}),m(Oe)!=="svelte-1mtog9j"&&(Oe.textContent=Ko),io=r(j),Ke=h(j,"P",{"data-svelte-h":!0}),m(Ke)!=="svelte-q52n56"&&(Ke.innerHTML=en),lo=r(j),et=h(j,"P",{"data-svelte-h":!0}),m(et)!=="svelte-hswkmf"&&(et.innerHTML=tn),co=r(j),E=h(j,"DIV",{class:!0});var Y=G(E);f(je.$$.fragment,Y),po=r(Y),tt=h(Y,"P",{"data-svelte-h":!0}),m(tt)!=="svelte-1x0oyto"&&(tt.innerHTML=on),ho=r(Y),f(K.$$.fragment,Y),Y.forEach(s),j.forEach(s),It=r(e),f(Ue.$$.fragment,e),Zt=r(e),L=h(e,"DIV",{class:!0});var U=G(L);f(Je.$$.fragment,U),mo=r(U),ot=h(U,"P",{"data-svelte-h":!0}),m(ot)!=="svelte-1tgcz08"&&(ot.textContent=nn),uo=r(U),nt=h(U,"P",{"data-svelte-h":!0}),m(nt)!=="svelte-q52n56"&&(nt.innerHTML=sn),fo=r(U),st=h(U,"P",{"data-svelte-h":!0}),m(st)!=="svelte-hswkmf"&&(st.innerHTML=an),go=r(U),Z=h(U,"DIV",{class:!0});var V=G(Z);f(qe.$$.fragment,V),_o=r(V),at=h(V,"P",{"data-svelte-h":!0}),m(at)!=="svelte-12q7u0o"&&(at.innerHTML=rn),bo=r(V),f(ee.$$.fragment,V),yo=r(V),f(te.$$.fragment,V),V.forEach(s),U.forEach(s),Rt=r(e),f(We.$$.fragment,e),Xt=r(e),B=h(e,"DIV",{class:!0});var J=G(B);f(Ge.$$.fragment,J),To=r(J),rt=h(J,"P",{"data-svelte-h":!0}),m(rt)!=="svelte-17xm775"&&(rt.textContent=dn),ko=r(J),it=h(J,"P",{"data-svelte-h":!0}),m(it)!=="svelte-q52n56"&&(it.innerHTML=ln),vo=r(J),dt=h(J,"P",{"data-svelte-h":!0}),m(dt)!=="svelte-hswkmf"&&(dt.innerHTML=cn),Mo=r(J),z=h(J,"DIV",{class:!0});var q=G(z);f(Ie.$$.fragment,q),wo=r(q),lt=h(q,"P",{"data-svelte-h":!0}),m(lt)!=="svelte-7yga48"&&(lt.innerHTML=pn),Co=r(q),f(oe.$$.fragment,q),Lo=r(q),f(ne.$$.fragment,q),Bo=r(q),f(se.$$.fragment,q),q.forEach(s),J.forEach(s),Nt=r(e),f(Ze.$$.fragment,e),St=r(e),x=h(e,"DIV",{class:!0});var W=G(x);f(Re.$$.fragment,W),xo=r(W),ct=h(W,"P",{"data-svelte-h":!0}),m(ct)!=="svelte-gz6gmd"&&(ct.textContent=hn),$o=r(W),pt=h(W,"P",{"data-svelte-h":!0}),m(pt)!=="svelte-q52n56"&&(pt.innerHTML=mn),zo=r(W),ht=h(W,"P",{"data-svelte-h":!0}),m(ht)!=="svelte-hswkmf"&&(ht.innerHTML=un),Fo=r(W),R=h(W,"DIV",{class:!0});var ie=G(R);f(Xe.$$.fragment,ie),Po=r(ie),mt=h(ie,"P",{"data-svelte-h":!0}),m(mt)!=="svelte-1r1id7g"&&(mt.innerHTML=fn),jo=r(ie),f(ae.$$.fragment,ie),Uo=r(ie),f(re.$$.fragment,ie),ie.forEach(s),W.forEach(s),Vt=r(e),f(Ne.$$.fragment,e),Ht=r(e),ft=h(e,"P",{}),G(ft).forEach(s),this.h()},h(){F(n,"name","hf:doc:metadata"),F(n,"content",Jn),F(A,"class","flex flex-wrap space-x-1"),F(P,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),F(I,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),F(w,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),F(E,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),F(C,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),F(Z,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),F(L,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),F(z,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),F(B,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),F(R,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),F(x,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,o){i(document.head,n),d(e,k,o),d(e,c,o),d(e,l,o),d(e,T,o),d(e,v,o),g(le,e,o),d(e,gt,o),d(e,A,o),d(e,_t,o),g(ce,e,o),d(e,bt,o),d(e,pe,o),d(e,yt,o),d(e,he,o),d(e,Tt,o),d(e,me,o),d(e,kt,o),d(e,ue,o),d(e,vt,o),g(fe,e,o),d(e,Mt,o),d(e,ge,o),d(e,wt,o),d(e,_e,o),d(e,Ct,o),d(e,be,o),d(e,Lt,o),g(ye,e,o),d(e,Bt,o),g(Te,e,o),d(e,xt,o),g(ke,e,o),d(e,$t,o),d(e,ve,o),d(e,zt,o),g(Me,e,o),d(e,Ft,o),g(we,e,o),d(e,Pt,o),d(e,Ce,o),d(e,jt,o),g(Le,e,o),d(e,Ut,o),d(e,P,o),g(Be,P,null),i(P,Yt),i(P,Se),i(P,At),i(P,Ve),i(P,Dt),g(D,P,null),d(e,Jt,o),g(xe,e,o),d(e,qt,o),d(e,w,o),g($e,w,null),i(w,Ot),i(w,He),i(w,Kt),i(w,Ee),i(w,eo),i(w,Qe),i(w,to),g(O,w,null),i(w,oo),i(w,I),g(ze,I,null),i(I,no),i(I,Ye),i(I,so),i(I,Ae),i(I,ao),i(I,De),d(e,Wt,o),g(Fe,e,o),d(e,Gt,o),d(e,C,o),g(Pe,C,null),i(C,ro),i(C,Oe),i(C,io),i(C,Ke),i(C,lo),i(C,et),i(C,co),i(C,E),g(je,E,null),i(E,po),i(E,tt),i(E,ho),g(K,E,null),d(e,It,o),g(Ue,e,o),d(e,Zt,o),d(e,L,o),g(Je,L,null),i(L,mo),i(L,ot),i(L,uo),i(L,nt),i(L,fo),i(L,st),i(L,go),i(L,Z),g(qe,Z,null),i(Z,_o),i(Z,at),i(Z,bo),g(ee,Z,null),i(Z,yo),g(te,Z,null),d(e,Rt,o),g(We,e,o),d(e,Xt,o),d(e,B,o),g(Ge,B,null),i(B,To),i(B,rt),i(B,ko),i(B,it),i(B,vo),i(B,dt),i(B,Mo),i(B,z),g(Ie,z,null),i(z,wo),i(z,lt),i(z,Co),g(oe,z,null),i(z,Lo),g(ne,z,null),i(z,Bo),g(se,z,null),d(e,Nt,o),g(Ze,e,o),d(e,St,o),d(e,x,o),g(Re,x,null),i(x,xo),i(x,ct),i(x,$o),i(x,pt),i(x,zo),i(x,ht),i(x,Fo),i(x,R),g(Xe,R,null),i(R,Po),i(R,mt),i(R,jo),g(ae,R,null),i(R,Uo),g(re,R,null),d(e,Vt,o),g(Ne,e,o),d(e,Ht,o),d(e,ft,o),Et=!0},p(e,[o]){const N={};o&2&&(N.$$scope={dirty:o,ctx:e}),D.$set(N);const $={};o&2&&($.$$scope={dirty:o,ctx:e}),O.$set($);const S={};o&2&&(S.$$scope={dirty:o,ctx:e}),K.$set(S);const j={};o&2&&(j.$$scope={dirty:o,ctx:e}),ee.$set(j);const Y={};o&2&&(Y.$$scope={dirty:o,ctx:e}),te.$set(Y);const U={};o&2&&(U.$$scope={dirty:o,ctx:e}),oe.$set(U);const V={};o&2&&(V.$$scope={dirty:o,ctx:e}),ne.$set(V);const J={};o&2&&(J.$$scope={dirty:o,ctx:e}),se.$set(J);const q={};o&2&&(q.$$scope={dirty:o,ctx:e}),ae.$set(q);const W={};o&2&&(W.$$scope={dirty:o,ctx:e}),re.$set(W)},i(e){Et||(_(le.$$.fragment,e),_(ce.$$.fragment,e),_(fe.$$.fragment,e),_(ye.$$.fragment,e),_(Te.$$.fragment,e),_(ke.$$.fragment,e),_(Me.$$.fragment,e),_(we.$$.fragment,e),_(Le.$$.fragment,e),_(Be.$$.fragment,e),_(D.$$.fragment,e),_(xe.$$.fragment,e),_($e.$$.fragment,e),_(O.$$.fragment,e),_(ze.$$.fragment,e),_(Fe.$$.fragment,e),_(Pe.$$.fragment,e),_(je.$$.fragment,e),_(K.$$.fragment,e),_(Ue.$$.fragment,e),_(Je.$$.fragment,e),_(qe.$$.fragment,e),_(ee.$$.fragment,e),_(te.$$.fragment,e),_(We.$$.fragment,e),_(Ge.$$.fragment,e),_(Ie.$$.fragment,e),_(oe.$$.fragment,e),_(ne.$$.fragment,e),_(se.$$.fragment,e),_(Ze.$$.fragment,e),_(Re.$$.fragment,e),_(Xe.$$.fragment,e),_(ae.$$.fragment,e),_(re.$$.fragment,e),_(Ne.$$.fragment,e),Et=!0)},o(e){b(le.$$.fragment,e),b(ce.$$.fragment,e),b(fe.$$.fragment,e),b(ye.$$.fragment,e),b(Te.$$.fragment,e),b(ke.$$.fragment,e),b(Me.$$.fragment,e),b(we.$$.fragment,e),b(Le.$$.fragment,e),b(Be.$$.fragment,e),b(D.$$.fragment,e),b(xe.$$.fragment,e),b($e.$$.fragment,e),b(O.$$.fragment,e),b(ze.$$.fragment,e),b(Fe.$$.fragment,e),b(Pe.$$.fragment,e),b(je.$$.fragment,e),b(K.$$.fragment,e),b(Ue.$$.fragment,e),b(Je.$$.fragment,e),b(qe.$$.fragment,e),b(ee.$$.fragment,e),b(te.$$.fragment,e),b(We.$$.fragment,e),b(Ge.$$.fragment,e),b(Ie.$$.fragment,e),b(oe.$$.fragment,e),b(ne.$$.fragment,e),b(se.$$.fragment,e),b(Ze.$$.fragment,e),b(Re.$$.fragment,e),b(Xe.$$.fragment,e),b(ae.$$.fragment,e),b(re.$$.fragment,e),b(Ne.$$.fragment,e),Et=!1},d(e){e&&(s(k),s(c),s(l),s(T),s(v),s(gt),s(A),s(_t),s(bt),s(pe),s(yt),s(he),s(Tt),s(me),s(kt),s(ue),s(vt),s(Mt),s(ge),s(wt),s(_e),s(Ct),s(be),s(Lt),s(Bt),s(xt),s($t),s(ve),s(zt),s(Ft),s(Pt),s(Ce),s(jt),s(Ut),s(P),s(Jt),s(qt),s(w),s(Wt),s(Gt),s(C),s(It),s(Zt),s(L),s(Rt),s(Xt),s(B),s(Nt),s(St),s(x),s(Vt),s(Ht),s(ft)),s(n),y(le,e),y(ce,e),y(fe,e),y(ye,e),y(Te,e),y(ke,e),y(Me,e),y(we,e),y(Le,e),y(Be),y(D),y(xe,e),y($e),y(O),y(ze),y(Fe,e),y(Pe),y(je),y(K),y(Ue,e),y(Je),y(qe),y(ee),y(te),y(We,e),y(Ge),y(Ie),y(oe),y(ne),y(se),y(Ze,e),y(Re),y(Xe),y(ae),y(re),y(Ne,e)}}}const Jn='{"title":"PLBart","local":"plbart","sections":[{"title":"Overview","local":"overview","sections":[],"depth":2},{"title":"Usage examples","local":"usage-examples","sections":[{"title":"Supervised training","local":"supervised-training","sections":[],"depth":3},{"title":"Generation","local":"generation","sections":[],"depth":3}],"depth":2},{"title":"Resources","local":"resources","sections":[],"depth":2},{"title":"PLBartConfig","local":"transformers.PLBartConfig","sections":[],"depth":2},{"title":"PLBartTokenizer","local":"transformers.PLBartTokenizer","sections":[],"depth":2},{"title":"PLBartModel","local":"transformers.PLBartModel","sections":[],"depth":2},{"title":"PLBartForConditionalGeneration","local":"transformers.PLBartForConditionalGeneration","sections":[],"depth":2},{"title":"PLBartForSequenceClassification","local":"transformers.PLBartForSequenceClassification","sections":[],"depth":2},{"title":"PLBartForCausalLM","local":"transformers.PLBartForCausalLM","sections":[],"depth":2}],"depth":1}';function qn(M){return _n(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class Sn extends bn{constructor(n){super(),yn(this,n,qn,Un,gn,{})}}export{Sn as component};
