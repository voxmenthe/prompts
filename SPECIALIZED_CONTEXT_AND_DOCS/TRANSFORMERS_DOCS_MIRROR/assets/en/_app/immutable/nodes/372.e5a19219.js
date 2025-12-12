import{s as Kn,o as es,n as x}from"../chunks/scheduler.18a86fab.js";import{S as ts,i as os,g as c,s as r,r as u,A as ns,h as m,f as i,c as a,j as N,x as h,u as f,k as J,y as s,a as p,v as g,d as _,t as b,w as y}from"../chunks/index.98837b22.js";import{T as nt}from"../chunks/Tip.77304350.js";import{D as P}from"../chunks/Docstring.a1ef7999.js";import{C as ke}from"../chunks/CodeBlock.8d0c2e8a.js";import{E as Te}from"../chunks/ExampleCodeBlock.8c3ee1f9.js";import{H as Y,E as ss}from"../chunks/getInferenceSnippets.06c2775f.js";function rs(w){let t,M="Examples:",l,d,T;return d=new ke({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMFJvYmVydGFQcmVMYXllck5vcm1Db25maWclMkMlMjBSb2JlcnRhUHJlTGF5ZXJOb3JtTW9kZWwlMEElMEElMjMlMjBJbml0aWFsaXppbmclMjBhJTIwUm9CRVJUYS1QcmVMYXllck5vcm0lMjBjb25maWd1cmF0aW9uJTBBY29uZmlndXJhdGlvbiUyMCUzRCUyMFJvYmVydGFQcmVMYXllck5vcm1Db25maWcoKSUwQSUwQSUyMyUyMEluaXRpYWxpemluZyUyMGElMjBtb2RlbCUyMCh3aXRoJTIwcmFuZG9tJTIwd2VpZ2h0cyklMjBmcm9tJTIwdGhlJTIwY29uZmlndXJhdGlvbiUwQW1vZGVsJTIwJTNEJTIwUm9iZXJ0YVByZUxheWVyTm9ybU1vZGVsKGNvbmZpZ3VyYXRpb24pJTBBJTBBJTIzJTIwQWNjZXNzaW5nJTIwdGhlJTIwbW9kZWwlMjBjb25maWd1cmF0aW9uJTBBY29uZmlndXJhdGlvbiUyMCUzRCUyMG1vZGVsLmNvbmZpZw==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> RobertaPreLayerNormConfig, RobertaPreLayerNormModel

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a RoBERTa-PreLayerNorm configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = RobertaPreLayerNormConfig()

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a model (with random weights) from the configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = RobertaPreLayerNormModel(configuration)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Accessing the model configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = model.config`,wrap:!1}}),{c(){t=c("p"),t.textContent=M,l=r(),u(d.$$.fragment)},l(o){t=m(o,"P",{"data-svelte-h":!0}),h(t)!=="svelte-kvfsh7"&&(t.textContent=M),l=a(o),f(d.$$.fragment,o)},m(o,k){p(o,t,k),p(o,l,k),g(d,o,k),T=!0},p:x,i(o){T||(_(d.$$.fragment,o),T=!0)},o(o){b(d.$$.fragment,o),T=!1},d(o){o&&(i(t),i(l)),y(d,o)}}}function as(w){let t,M=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=M},l(l){t=m(l,"P",{"data-svelte-h":!0}),h(t)!=="svelte-fincs2"&&(t.innerHTML=M)},m(l,d){p(l,t,d)},p:x,d(l){l&&i(t)}}}function is(w){let t,M=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=M},l(l){t=m(l,"P",{"data-svelte-h":!0}),h(t)!=="svelte-fincs2"&&(t.innerHTML=M)},m(l,d){p(l,t,d)},p:x,d(l){l&&i(t)}}}function ls(w){let t,M="Example:",l,d,T;return d=new ke({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBSb2JlcnRhUHJlTGF5ZXJOb3JtRm9yQ2F1c2FsTE0lMkMlMjBBdXRvQ29uZmlnJTBBaW1wb3J0JTIwdG9yY2glMEElMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJhbmRyZWFzbWFkc2VuJTJGZWZmaWNpZW50X21sbV9tMC40MCUyMiklMEFjb25maWclMjAlM0QlMjBBdXRvQ29uZmlnLmZyb21fcHJldHJhaW5lZCglMjJhbmRyZWFzbWFkc2VuJTJGZWZmaWNpZW50X21sbV9tMC40MCUyMiklMEFjb25maWcuaXNfZGVjb2RlciUyMCUzRCUyMFRydWUlMEFtb2RlbCUyMCUzRCUyMFJvYmVydGFQcmVMYXllck5vcm1Gb3JDYXVzYWxMTS5mcm9tX3ByZXRyYWluZWQoJTIyYW5kcmVhc21hZHNlbiUyRmVmZmljaWVudF9tbG1fbTAuNDAlMjIlMkMlMjBjb25maWclM0Rjb25maWcpJTBBJTBBaW5wdXRzJTIwJTNEJTIwdG9rZW5pemVyKCUyMkhlbGxvJTJDJTIwbXklMjBkb2clMjBpcyUyMGN1dGUlMjIlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQW91dHB1dHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyklMEElMEFwcmVkaWN0aW9uX2xvZ2l0cyUyMCUzRCUyMG91dHB1dHMubG9naXRz",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, RobertaPreLayerNormForCausalLM, AutoConfig
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;andreasmadsen/efficient_mlm_m0.40&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>config = AutoConfig.from_pretrained(<span class="hljs-string">&quot;andreasmadsen/efficient_mlm_m0.40&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>config.is_decoder = <span class="hljs-literal">True</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = RobertaPreLayerNormForCausalLM.from_pretrained(<span class="hljs-string">&quot;andreasmadsen/efficient_mlm_m0.40&quot;</span>, config=config)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs)

<span class="hljs-meta">&gt;&gt;&gt; </span>prediction_logits = outputs.logits`,wrap:!1}}),{c(){t=c("p"),t.textContent=M,l=r(),u(d.$$.fragment)},l(o){t=m(o,"P",{"data-svelte-h":!0}),h(t)!=="svelte-11lpom8"&&(t.textContent=M),l=a(o),f(d.$$.fragment,o)},m(o,k){p(o,t,k),p(o,l,k),g(d,o,k),T=!0},p:x,i(o){T||(_(d.$$.fragment,o),T=!0)},o(o){b(d.$$.fragment,o),T=!1},d(o){o&&(i(t),i(l)),y(d,o)}}}function ds(w){let t,M=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=M},l(l){t=m(l,"P",{"data-svelte-h":!0}),h(t)!=="svelte-fincs2"&&(t.innerHTML=M)},m(l,d){p(l,t,d)},p:x,d(l){l&&i(t)}}}function cs(w){let t,M="Example:",l,d,T;return d=new ke({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBSb2JlcnRhUHJlTGF5ZXJOb3JtRm9yTWFza2VkTE0lMEFpbXBvcnQlMjB0b3JjaCUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMmFuZHJlYXNtYWRzZW4lMkZlZmZpY2llbnRfbWxtX20wLjQwJTIyKSUwQW1vZGVsJTIwJTNEJTIwUm9iZXJ0YVByZUxheWVyTm9ybUZvck1hc2tlZExNLmZyb21fcHJldHJhaW5lZCglMjJhbmRyZWFzbWFkc2VuJTJGZWZmaWNpZW50X21sbV9tMC40MCUyMiklMEElMEFpbnB1dHMlMjAlM0QlMjB0b2tlbml6ZXIoJTIyVGhlJTIwY2FwaXRhbCUyMG9mJTIwRnJhbmNlJTIwaXMlMjAlM0NtYXNrJTNFLiUyMiUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpJTBBJTBBd2l0aCUyMHRvcmNoLm5vX2dyYWQoKSUzQSUwQSUyMCUyMCUyMCUyMGxvZ2l0cyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzKS5sb2dpdHMlMEElMEElMjMlMjByZXRyaWV2ZSUyMGluZGV4JTIwb2YlMjAlM0NtYXNrJTNFJTBBbWFza190b2tlbl9pbmRleCUyMCUzRCUyMChpbnB1dHMuaW5wdXRfaWRzJTIwJTNEJTNEJTIwdG9rZW5pemVyLm1hc2tfdG9rZW5faWQpJTVCMCU1RC5ub256ZXJvKGFzX3R1cGxlJTNEVHJ1ZSklNUIwJTVEJTBBJTBBcHJlZGljdGVkX3Rva2VuX2lkJTIwJTNEJTIwbG9naXRzJTVCMCUyQyUyMG1hc2tfdG9rZW5faW5kZXglNUQuYXJnbWF4KGF4aXMlM0QtMSklMEF0b2tlbml6ZXIuZGVjb2RlKHByZWRpY3RlZF90b2tlbl9pZCklMEElMEFsYWJlbHMlMjAlM0QlMjB0b2tlbml6ZXIoJTIyVGhlJTIwY2FwaXRhbCUyMG9mJTIwRnJhbmNlJTIwaXMlMjBQYXJpcy4lMjIlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSU1QiUyMmlucHV0X2lkcyUyMiU1RCUwQSUyMyUyMG1hc2slMjBsYWJlbHMlMjBvZiUyMG5vbi0lM0NtYXNrJTNFJTIwdG9rZW5zJTBBbGFiZWxzJTIwJTNEJTIwdG9yY2gud2hlcmUoaW5wdXRzLmlucHV0X2lkcyUyMCUzRCUzRCUyMHRva2VuaXplci5tYXNrX3Rva2VuX2lkJTJDJTIwbGFiZWxzJTJDJTIwLTEwMCklMEElMEFvdXRwdXRzJTIwJTNEJTIwbW9kZWwoKippbnB1dHMlMkMlMjBsYWJlbHMlM0RsYWJlbHMpJTBBcm91bmQob3V0cHV0cy5sb3NzLml0ZW0oKSUyQyUyMDIp",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, RobertaPreLayerNormForMaskedLM
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;andreasmadsen/efficient_mlm_m0.40&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = RobertaPreLayerNormForMaskedLM.from_pretrained(<span class="hljs-string">&quot;andreasmadsen/efficient_mlm_m0.40&quot;</span>)

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
...`,wrap:!1}}),{c(){t=c("p"),t.textContent=M,l=r(),u(d.$$.fragment)},l(o){t=m(o,"P",{"data-svelte-h":!0}),h(t)!=="svelte-11lpom8"&&(t.textContent=M),l=a(o),f(d.$$.fragment,o)},m(o,k){p(o,t,k),p(o,l,k),g(d,o,k),T=!0},p:x,i(o){T||(_(d.$$.fragment,o),T=!0)},o(o){b(d.$$.fragment,o),T=!1},d(o){o&&(i(t),i(l)),y(d,o)}}}function ms(w){let t,M=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=M},l(l){t=m(l,"P",{"data-svelte-h":!0}),h(t)!=="svelte-fincs2"&&(t.innerHTML=M)},m(l,d){p(l,t,d)},p:x,d(l){l&&i(t)}}}function ps(w){let t,M="Example of single-label classification:",l,d,T;return d=new ke({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Rva2VuaXplciUyQyUyMFJvYmVydGFQcmVMYXllck5vcm1Gb3JTZXF1ZW5jZUNsYXNzaWZpY2F0aW9uJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIyYW5kcmVhc21hZHNlbiUyRmVmZmljaWVudF9tbG1fbTAuNDAlMjIpJTBBbW9kZWwlMjAlM0QlMjBSb2JlcnRhUHJlTGF5ZXJOb3JtRm9yU2VxdWVuY2VDbGFzc2lmaWNhdGlvbi5mcm9tX3ByZXRyYWluZWQoJTIyYW5kcmVhc21hZHNlbiUyRmVmZmljaWVudF9tbG1fbTAuNDAlMjIpJTBBJTBBaW5wdXRzJTIwJTNEJTIwdG9rZW5pemVyKCUyMkhlbGxvJTJDJTIwbXklMjBkb2clMjBpcyUyMGN1dGUlMjIlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQSUwQXdpdGglMjB0b3JjaC5ub19ncmFkKCklM0ElMEElMjAlMjAlMjAlMjBsb2dpdHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cykubG9naXRzJTBBJTBBcHJlZGljdGVkX2NsYXNzX2lkJTIwJTNEJTIwbG9naXRzLmFyZ21heCgpLml0ZW0oKSUwQW1vZGVsLmNvbmZpZy5pZDJsYWJlbCU1QnByZWRpY3RlZF9jbGFzc19pZCU1RCUwQSUwQSUyMyUyMFRvJTIwdHJhaW4lMjBhJTIwbW9kZWwlMjBvbiUyMCU2MG51bV9sYWJlbHMlNjAlMjBjbGFzc2VzJTJDJTIweW91JTIwY2FuJTIwcGFzcyUyMCU2MG51bV9sYWJlbHMlM0RudW1fbGFiZWxzJTYwJTIwdG8lMjAlNjAuZnJvbV9wcmV0cmFpbmVkKC4uLiklNjAlMEFudW1fbGFiZWxzJTIwJTNEJTIwbGVuKG1vZGVsLmNvbmZpZy5pZDJsYWJlbCklMEFtb2RlbCUyMCUzRCUyMFJvYmVydGFQcmVMYXllck5vcm1Gb3JTZXF1ZW5jZUNsYXNzaWZpY2F0aW9uLmZyb21fcHJldHJhaW5lZCglMjJhbmRyZWFzbWFkc2VuJTJGZWZmaWNpZW50X21sbV9tMC40MCUyMiUyQyUyMG51bV9sYWJlbHMlM0RudW1fbGFiZWxzKSUwQSUwQWxhYmVscyUyMCUzRCUyMHRvcmNoLnRlbnNvciglNUIxJTVEKSUwQWxvc3MlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyUyQyUyMGxhYmVscyUzRGxhYmVscykubG9zcyUwQXJvdW5kKGxvc3MuaXRlbSgpJTJDJTIwMik=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, RobertaPreLayerNormForSequenceClassification

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;andreasmadsen/efficient_mlm_m0.40&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = RobertaPreLayerNormForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;andreasmadsen/efficient_mlm_m0.40&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    logits = model(**inputs).logits

<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_class_id = logits.argmax().item()
<span class="hljs-meta">&gt;&gt;&gt; </span>model.config.id2label[predicted_class_id]
...

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># To train a model on \`num_labels\` classes, you can pass \`num_labels=num_labels\` to \`.from_pretrained(...)\`</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>num_labels = <span class="hljs-built_in">len</span>(model.config.id2label)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = RobertaPreLayerNormForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;andreasmadsen/efficient_mlm_m0.40&quot;</span>, num_labels=num_labels)

<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.tensor([<span class="hljs-number">1</span>])
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = model(**inputs, labels=labels).loss
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">round</span>(loss.item(), <span class="hljs-number">2</span>)
...`,wrap:!1}}),{c(){t=c("p"),t.textContent=M,l=r(),u(d.$$.fragment)},l(o){t=m(o,"P",{"data-svelte-h":!0}),h(t)!=="svelte-ykxpe4"&&(t.textContent=M),l=a(o),f(d.$$.fragment,o)},m(o,k){p(o,t,k),p(o,l,k),g(d,o,k),T=!0},p:x,i(o){T||(_(d.$$.fragment,o),T=!0)},o(o){b(d.$$.fragment,o),T=!1},d(o){o&&(i(t),i(l)),y(d,o)}}}function hs(w){let t,M="Example of multi-label classification:",l,d,T;return d=new ke({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Rva2VuaXplciUyQyUyMFJvYmVydGFQcmVMYXllck5vcm1Gb3JTZXF1ZW5jZUNsYXNzaWZpY2F0aW9uJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIyYW5kcmVhc21hZHNlbiUyRmVmZmljaWVudF9tbG1fbTAuNDAlMjIpJTBBbW9kZWwlMjAlM0QlMjBSb2JlcnRhUHJlTGF5ZXJOb3JtRm9yU2VxdWVuY2VDbGFzc2lmaWNhdGlvbi5mcm9tX3ByZXRyYWluZWQoJTIyYW5kcmVhc21hZHNlbiUyRmVmZmljaWVudF9tbG1fbTAuNDAlMjIlMkMlMjBwcm9ibGVtX3R5cGUlM0QlMjJtdWx0aV9sYWJlbF9jbGFzc2lmaWNhdGlvbiUyMiklMEElMEFpbnB1dHMlMjAlM0QlMjB0b2tlbml6ZXIoJTIySGVsbG8lMkMlMjBteSUyMGRvZyUyMGlzJTIwY3V0ZSUyMiUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpJTBBJTBBd2l0aCUyMHRvcmNoLm5vX2dyYWQoKSUzQSUwQSUyMCUyMCUyMCUyMGxvZ2l0cyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzKS5sb2dpdHMlMEElMEFwcmVkaWN0ZWRfY2xhc3NfaWRzJTIwJTNEJTIwdG9yY2guYXJhbmdlKDAlMkMlMjBsb2dpdHMuc2hhcGUlNUItMSU1RCklNUJ0b3JjaC5zaWdtb2lkKGxvZ2l0cykuc3F1ZWV6ZShkaW0lM0QwKSUyMCUzRSUyMDAuNSU1RCUwQSUwQSUyMyUyMFRvJTIwdHJhaW4lMjBhJTIwbW9kZWwlMjBvbiUyMCU2MG51bV9sYWJlbHMlNjAlMjBjbGFzc2VzJTJDJTIweW91JTIwY2FuJTIwcGFzcyUyMCU2MG51bV9sYWJlbHMlM0RudW1fbGFiZWxzJTYwJTIwdG8lMjAlNjAuZnJvbV9wcmV0cmFpbmVkKC4uLiklNjAlMEFudW1fbGFiZWxzJTIwJTNEJTIwbGVuKG1vZGVsLmNvbmZpZy5pZDJsYWJlbCklMEFtb2RlbCUyMCUzRCUyMFJvYmVydGFQcmVMYXllck5vcm1Gb3JTZXF1ZW5jZUNsYXNzaWZpY2F0aW9uLmZyb21fcHJldHJhaW5lZCglMEElMjAlMjAlMjAlMjAlMjJhbmRyZWFzbWFkc2VuJTJGZWZmaWNpZW50X21sbV9tMC40MCUyMiUyQyUyMG51bV9sYWJlbHMlM0RudW1fbGFiZWxzJTJDJTIwcHJvYmxlbV90eXBlJTNEJTIybXVsdGlfbGFiZWxfY2xhc3NpZmljYXRpb24lMjIlMEEpJTBBJTBBbGFiZWxzJTIwJTNEJTIwdG9yY2guc3VtKCUwQSUyMCUyMCUyMCUyMHRvcmNoLm5uLmZ1bmN0aW9uYWwub25lX2hvdChwcmVkaWN0ZWRfY2xhc3NfaWRzJTVCTm9uZSUyQyUyMCUzQSU1RC5jbG9uZSgpJTJDJTIwbnVtX2NsYXNzZXMlM0RudW1fbGFiZWxzKSUyQyUyMGRpbSUzRDElMEEpLnRvKHRvcmNoLmZsb2F0KSUwQWxvc3MlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyUyQyUyMGxhYmVscyUzRGxhYmVscykubG9zcw==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, RobertaPreLayerNormForSequenceClassification

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;andreasmadsen/efficient_mlm_m0.40&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = RobertaPreLayerNormForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;andreasmadsen/efficient_mlm_m0.40&quot;</span>, problem_type=<span class="hljs-string">&quot;multi_label_classification&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    logits = model(**inputs).logits

<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_class_ids = torch.arange(<span class="hljs-number">0</span>, logits.shape[-<span class="hljs-number">1</span>])[torch.sigmoid(logits).squeeze(dim=<span class="hljs-number">0</span>) &gt; <span class="hljs-number">0.5</span>]

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># To train a model on \`num_labels\` classes, you can pass \`num_labels=num_labels\` to \`.from_pretrained(...)\`</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>num_labels = <span class="hljs-built_in">len</span>(model.config.id2label)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = RobertaPreLayerNormForSequenceClassification.from_pretrained(
<span class="hljs-meta">... </span>    <span class="hljs-string">&quot;andreasmadsen/efficient_mlm_m0.40&quot;</span>, num_labels=num_labels, problem_type=<span class="hljs-string">&quot;multi_label_classification&quot;</span>
<span class="hljs-meta">... </span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.<span class="hljs-built_in">sum</span>(
<span class="hljs-meta">... </span>    torch.nn.functional.one_hot(predicted_class_ids[<span class="hljs-literal">None</span>, :].clone(), num_classes=num_labels), dim=<span class="hljs-number">1</span>
<span class="hljs-meta">... </span>).to(torch.<span class="hljs-built_in">float</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = model(**inputs, labels=labels).loss`,wrap:!1}}),{c(){t=c("p"),t.textContent=M,l=r(),u(d.$$.fragment)},l(o){t=m(o,"P",{"data-svelte-h":!0}),h(t)!=="svelte-1l8e32d"&&(t.textContent=M),l=a(o),f(d.$$.fragment,o)},m(o,k){p(o,t,k),p(o,l,k),g(d,o,k),T=!0},p:x,i(o){T||(_(d.$$.fragment,o),T=!0)},o(o){b(d.$$.fragment,o),T=!1},d(o){o&&(i(t),i(l)),y(d,o)}}}function us(w){let t,M=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=M},l(l){t=m(l,"P",{"data-svelte-h":!0}),h(t)!=="svelte-fincs2"&&(t.innerHTML=M)},m(l,d){p(l,t,d)},p:x,d(l){l&&i(t)}}}function fs(w){let t,M="Example:",l,d,T;return d=new ke({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBSb2JlcnRhUHJlTGF5ZXJOb3JtRm9yTXVsdGlwbGVDaG9pY2UlMEFpbXBvcnQlMjB0b3JjaCUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMmFuZHJlYXNtYWRzZW4lMkZlZmZpY2llbnRfbWxtX20wLjQwJTIyKSUwQW1vZGVsJTIwJTNEJTIwUm9iZXJ0YVByZUxheWVyTm9ybUZvck11bHRpcGxlQ2hvaWNlLmZyb21fcHJldHJhaW5lZCglMjJhbmRyZWFzbWFkc2VuJTJGZWZmaWNpZW50X21sbV9tMC40MCUyMiklMEElMEFwcm9tcHQlMjAlM0QlMjAlMjJJbiUyMEl0YWx5JTJDJTIwcGl6emElMjBzZXJ2ZWQlMjBpbiUyMGZvcm1hbCUyMHNldHRpbmdzJTJDJTIwc3VjaCUyMGFzJTIwYXQlMjBhJTIwcmVzdGF1cmFudCUyQyUyMGlzJTIwcHJlc2VudGVkJTIwdW5zbGljZWQuJTIyJTBBY2hvaWNlMCUyMCUzRCUyMCUyMkl0JTIwaXMlMjBlYXRlbiUyMHdpdGglMjBhJTIwZm9yayUyMGFuZCUyMGElMjBrbmlmZS4lMjIlMEFjaG9pY2UxJTIwJTNEJTIwJTIySXQlMjBpcyUyMGVhdGVuJTIwd2hpbGUlMjBoZWxkJTIwaW4lMjB0aGUlMjBoYW5kLiUyMiUwQWxhYmVscyUyMCUzRCUyMHRvcmNoLnRlbnNvcigwKS51bnNxdWVlemUoMCklMjAlMjAlMjMlMjBjaG9pY2UwJTIwaXMlMjBjb3JyZWN0JTIwKGFjY29yZGluZyUyMHRvJTIwV2lraXBlZGlhJTIwJTNCKSklMkMlMjBiYXRjaCUyMHNpemUlMjAxJTBBJTBBZW5jb2RpbmclMjAlM0QlMjB0b2tlbml6ZXIoJTVCcHJvbXB0JTJDJTIwcHJvbXB0JTVEJTJDJTIwJTVCY2hvaWNlMCUyQyUyMGNob2ljZTElNUQlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyJTJDJTIwcGFkZGluZyUzRFRydWUpJTBBb3V0cHV0cyUyMCUzRCUyMG1vZGVsKCoqJTdCayUzQSUyMHYudW5zcXVlZXplKDApJTIwZm9yJTIwayUyQyUyMHYlMjBpbiUyMGVuY29kaW5nLml0ZW1zKCklN0QlMkMlMjBsYWJlbHMlM0RsYWJlbHMpJTIwJTIwJTIzJTIwYmF0Y2glMjBzaXplJTIwaXMlMjAxJTBBJTBBJTIzJTIwdGhlJTIwbGluZWFyJTIwY2xhc3NpZmllciUyMHN0aWxsJTIwbmVlZHMlMjB0byUyMGJlJTIwdHJhaW5lZCUwQWxvc3MlMjAlM0QlMjBvdXRwdXRzLmxvc3MlMEFsb2dpdHMlMjAlM0QlMjBvdXRwdXRzLmxvZ2l0cw==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, RobertaPreLayerNormForMultipleChoice
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;andreasmadsen/efficient_mlm_m0.40&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = RobertaPreLayerNormForMultipleChoice.from_pretrained(<span class="hljs-string">&quot;andreasmadsen/efficient_mlm_m0.40&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>prompt = <span class="hljs-string">&quot;In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>choice0 = <span class="hljs-string">&quot;It is eaten with a fork and a knife.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>choice1 = <span class="hljs-string">&quot;It is eaten while held in the hand.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.tensor(<span class="hljs-number">0</span>).unsqueeze(<span class="hljs-number">0</span>)  <span class="hljs-comment"># choice0 is correct (according to Wikipedia ;)), batch size 1</span>

<span class="hljs-meta">&gt;&gt;&gt; </span>encoding = tokenizer([prompt, prompt], [choice0, choice1], return_tensors=<span class="hljs-string">&quot;pt&quot;</span>, padding=<span class="hljs-literal">True</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**{k: v.unsqueeze(<span class="hljs-number">0</span>) <span class="hljs-keyword">for</span> k, v <span class="hljs-keyword">in</span> encoding.items()}, labels=labels)  <span class="hljs-comment"># batch size is 1</span>

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># the linear classifier still needs to be trained</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = outputs.loss
<span class="hljs-meta">&gt;&gt;&gt; </span>logits = outputs.logits`,wrap:!1}}),{c(){t=c("p"),t.textContent=M,l=r(),u(d.$$.fragment)},l(o){t=m(o,"P",{"data-svelte-h":!0}),h(t)!=="svelte-11lpom8"&&(t.textContent=M),l=a(o),f(d.$$.fragment,o)},m(o,k){p(o,t,k),p(o,l,k),g(d,o,k),T=!0},p:x,i(o){T||(_(d.$$.fragment,o),T=!0)},o(o){b(d.$$.fragment,o),T=!1},d(o){o&&(i(t),i(l)),y(d,o)}}}function gs(w){let t,M=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=M},l(l){t=m(l,"P",{"data-svelte-h":!0}),h(t)!=="svelte-fincs2"&&(t.innerHTML=M)},m(l,d){p(l,t,d)},p:x,d(l){l&&i(t)}}}function _s(w){let t,M="Example:",l,d,T;return d=new ke({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBSb2JlcnRhUHJlTGF5ZXJOb3JtRm9yVG9rZW5DbGFzc2lmaWNhdGlvbiUwQWltcG9ydCUyMHRvcmNoJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIyYW5kcmVhc21hZHNlbiUyRmVmZmljaWVudF9tbG1fbTAuNDAlMjIpJTBBbW9kZWwlMjAlM0QlMjBSb2JlcnRhUHJlTGF5ZXJOb3JtRm9yVG9rZW5DbGFzc2lmaWNhdGlvbi5mcm9tX3ByZXRyYWluZWQoJTIyYW5kcmVhc21hZHNlbiUyRmVmZmljaWVudF9tbG1fbTAuNDAlMjIpJTBBJTBBaW5wdXRzJTIwJTNEJTIwdG9rZW5pemVyKCUwQSUyMCUyMCUyMCUyMCUyMkh1Z2dpbmdGYWNlJTIwaXMlMjBhJTIwY29tcGFueSUyMGJhc2VkJTIwaW4lMjBQYXJpcyUyMGFuZCUyME5ldyUyMFlvcmslMjIlMkMlMjBhZGRfc3BlY2lhbF90b2tlbnMlM0RGYWxzZSUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIlMEEpJTBBJTBBd2l0aCUyMHRvcmNoLm5vX2dyYWQoKSUzQSUwQSUyMCUyMCUyMCUyMGxvZ2l0cyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzKS5sb2dpdHMlMEElMEFwcmVkaWN0ZWRfdG9rZW5fY2xhc3NfaWRzJTIwJTNEJTIwbG9naXRzLmFyZ21heCgtMSklMEElMEElMjMlMjBOb3RlJTIwdGhhdCUyMHRva2VucyUyMGFyZSUyMGNsYXNzaWZpZWQlMjByYXRoZXIlMjB0aGVuJTIwaW5wdXQlMjB3b3JkcyUyMHdoaWNoJTIwbWVhbnMlMjB0aGF0JTBBJTIzJTIwdGhlcmUlMjBtaWdodCUyMGJlJTIwbW9yZSUyMHByZWRpY3RlZCUyMHRva2VuJTIwY2xhc3NlcyUyMHRoYW4lMjB3b3Jkcy4lMEElMjMlMjBNdWx0aXBsZSUyMHRva2VuJTIwY2xhc3NlcyUyMG1pZ2h0JTIwYWNjb3VudCUyMGZvciUyMHRoZSUyMHNhbWUlMjB3b3JkJTBBcHJlZGljdGVkX3Rva2Vuc19jbGFzc2VzJTIwJTNEJTIwJTVCbW9kZWwuY29uZmlnLmlkMmxhYmVsJTVCdC5pdGVtKCklNUQlMjBmb3IlMjB0JTIwaW4lMjBwcmVkaWN0ZWRfdG9rZW5fY2xhc3NfaWRzJTVCMCU1RCU1RCUwQXByZWRpY3RlZF90b2tlbnNfY2xhc3NlcyUwQSUwQWxhYmVscyUyMCUzRCUyMHByZWRpY3RlZF90b2tlbl9jbGFzc19pZHMlMEFsb3NzJTIwJTNEJTIwbW9kZWwoKippbnB1dHMlMkMlMjBsYWJlbHMlM0RsYWJlbHMpLmxvc3MlMEFyb3VuZChsb3NzLml0ZW0oKSUyQyUyMDIp",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, RobertaPreLayerNormForTokenClassification
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;andreasmadsen/efficient_mlm_m0.40&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = RobertaPreLayerNormForTokenClassification.from_pretrained(<span class="hljs-string">&quot;andreasmadsen/efficient_mlm_m0.40&quot;</span>)

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
...`,wrap:!1}}),{c(){t=c("p"),t.textContent=M,l=r(),u(d.$$.fragment)},l(o){t=m(o,"P",{"data-svelte-h":!0}),h(t)!=="svelte-11lpom8"&&(t.textContent=M),l=a(o),f(d.$$.fragment,o)},m(o,k){p(o,t,k),p(o,l,k),g(d,o,k),T=!0},p:x,i(o){T||(_(d.$$.fragment,o),T=!0)},o(o){b(d.$$.fragment,o),T=!1},d(o){o&&(i(t),i(l)),y(d,o)}}}function bs(w){let t,M=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=M},l(l){t=m(l,"P",{"data-svelte-h":!0}),h(t)!=="svelte-fincs2"&&(t.innerHTML=M)},m(l,d){p(l,t,d)},p:x,d(l){l&&i(t)}}}function ys(w){let t,M="Example:",l,d,T;return d=new ke({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBSb2JlcnRhUHJlTGF5ZXJOb3JtRm9yUXVlc3Rpb25BbnN3ZXJpbmclMEFpbXBvcnQlMjB0b3JjaCUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMmFuZHJlYXNtYWRzZW4lMkZlZmZpY2llbnRfbWxtX20wLjQwJTIyKSUwQW1vZGVsJTIwJTNEJTIwUm9iZXJ0YVByZUxheWVyTm9ybUZvclF1ZXN0aW9uQW5zd2VyaW5nLmZyb21fcHJldHJhaW5lZCglMjJhbmRyZWFzbWFkc2VuJTJGZWZmaWNpZW50X21sbV9tMC40MCUyMiklMEElMEFxdWVzdGlvbiUyQyUyMHRleHQlMjAlM0QlMjAlMjJXaG8lMjB3YXMlMjBKaW0lMjBIZW5zb24lM0YlMjIlMkMlMjAlMjJKaW0lMjBIZW5zb24lMjB3YXMlMjBhJTIwbmljZSUyMHB1cHBldCUyMiUwQSUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplcihxdWVzdGlvbiUyQyUyMHRleHQlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQXdpdGglMjB0b3JjaC5ub19ncmFkKCklM0ElMEElMjAlMjAlMjAlMjBvdXRwdXRzJTIwJTNEJTIwbW9kZWwoKippbnB1dHMpJTBBJTBBYW5zd2VyX3N0YXJ0X2luZGV4JTIwJTNEJTIwb3V0cHV0cy5zdGFydF9sb2dpdHMuYXJnbWF4KCklMEFhbnN3ZXJfZW5kX2luZGV4JTIwJTNEJTIwb3V0cHV0cy5lbmRfbG9naXRzLmFyZ21heCgpJTBBJTBBcHJlZGljdF9hbnN3ZXJfdG9rZW5zJTIwJTNEJTIwaW5wdXRzLmlucHV0X2lkcyU1QjAlMkMlMjBhbnN3ZXJfc3RhcnRfaW5kZXglMjAlM0ElMjBhbnN3ZXJfZW5kX2luZGV4JTIwJTJCJTIwMSU1RCUwQXRva2VuaXplci5kZWNvZGUocHJlZGljdF9hbnN3ZXJfdG9rZW5zJTJDJTIwc2tpcF9zcGVjaWFsX3Rva2VucyUzRFRydWUpJTBBJTBBJTIzJTIwdGFyZ2V0JTIwaXMlMjAlMjJuaWNlJTIwcHVwcGV0JTIyJTBBdGFyZ2V0X3N0YXJ0X2luZGV4JTIwJTNEJTIwdG9yY2gudGVuc29yKCU1QjE0JTVEKSUwQXRhcmdldF9lbmRfaW5kZXglMjAlM0QlMjB0b3JjaC50ZW5zb3IoJTVCMTUlNUQpJTBBJTBBb3V0cHV0cyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzJTJDJTIwc3RhcnRfcG9zaXRpb25zJTNEdGFyZ2V0X3N0YXJ0X2luZGV4JTJDJTIwZW5kX3Bvc2l0aW9ucyUzRHRhcmdldF9lbmRfaW5kZXgpJTBBbG9zcyUyMCUzRCUyMG91dHB1dHMubG9zcyUwQXJvdW5kKGxvc3MuaXRlbSgpJTJDJTIwMik=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, RobertaPreLayerNormForQuestionAnswering
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;andreasmadsen/efficient_mlm_m0.40&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = RobertaPreLayerNormForQuestionAnswering.from_pretrained(<span class="hljs-string">&quot;andreasmadsen/efficient_mlm_m0.40&quot;</span>)

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
...`,wrap:!1}}),{c(){t=c("p"),t.textContent=M,l=r(),u(d.$$.fragment)},l(o){t=m(o,"P",{"data-svelte-h":!0}),h(t)!=="svelte-11lpom8"&&(t.textContent=M),l=a(o),f(d.$$.fragment,o)},m(o,k){p(o,t,k),p(o,l,k),g(d,o,k),T=!0},p:x,i(o){T||(_(d.$$.fragment,o),T=!0)},o(o){b(d.$$.fragment,o),T=!1},d(o){o&&(i(t),i(l)),y(d,o)}}}function Ms(w){let t,M,l,d,T,o="<em>This model was released on 2019-04-01 and added to Hugging Face Transformers on 2022-12-19.</em>",k,we,Ut,re,un='<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/>',Zt,ve,It,Je,fn=`The RoBERTa-PreLayerNorm model was proposed in <a href="https://huggingface.co/papers/1904.01038" rel="nofollow">fairseq: A Fast, Extensible Toolkit for Sequence Modeling</a> by Myle Ott, Sergey Edunov, Alexei Baevski, Angela Fan, Sam Gross, Nathan Ng, David Grangier, Michael Auli.
It is identical to using the <code>--encoder-normalize-before</code> flag in <a href="https://fairseq.readthedocs.io/" rel="nofollow">fairseq</a>.`,qt,Le,gn="The abstract from the paper is the following:",Bt,Ne,_n="<em>fairseq is an open-source sequence modeling toolkit that allows researchers and developers to train custom models for translation, summarization, language modeling, and other text generation tasks. The toolkit is based on PyTorch and supports distributed training across multiple GPUs and machines. We also support fast mixed-precision training and inference on modern GPUs.</em>",Gt,$e,bn=`This model was contributed by <a href="https://huggingface.co/andreasmadsen" rel="nofollow">andreasmaden</a>.
The original code can be found <a href="https://github.com/princeton-nlp/DinkyTrain" rel="nofollow">here</a>.`,Vt,Re,Ht,Ce,yn='<li>The implementation is the same as <a href="roberta">Roberta</a> except instead of using <em>Add and Norm</em> it does <em>Norm and Add</em>. <em>Add</em> and <em>Norm</em> refers to the Addition and LayerNormalization as described in <a href="https://huggingface.co/papers/1706.03762" rel="nofollow">Attention Is All You Need</a>.</li> <li>This is identical to using the <code>--encoder-normalize-before</code> flag in <a href="https://fairseq.readthedocs.io/" rel="nofollow">fairseq</a>.</li>',Xt,Fe,At,je,Mn='<li><a href="../tasks/sequence_classification">Text classification task guide</a></li> <li><a href="../tasks/token_classification">Token classification task guide</a></li> <li><a href="../tasks/question_answering">Question answering task guide</a></li> <li><a href="../tasks/language_modeling">Causal language modeling task guide</a></li> <li><a href="../tasks/masked_language_modeling">Masked language modeling task guide</a></li> <li><a href="../tasks/multiple_choice">Multiple choice task guide</a></li>',St,ze,Et,U,Pe,ho,st,Tn=`This is the configuration class to store the configuration of a <a href="/docs/transformers/v4.56.2/en/model_doc/roberta-prelayernorm#transformers.RobertaPreLayerNormModel">RobertaPreLayerNormModel</a> or a <code>TFRobertaPreLayerNormModel</code>. It is
used to instantiate a RoBERTa-PreLayerNorm model according to the specified arguments, defining the model architecture.
Instantiating a configuration with the defaults will yield a similar configuration to that of the RoBERTa-PreLayerNorm
<a href="https://huggingface.co/andreasmadsen/efficient_mlm_m0.40" rel="nofollow">andreasmadsen/efficient_mlm_m0.40</a> architecture.`,uo,rt,kn=`Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> and can be used to control the model outputs. Read the
documentation from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> for more information.`,fo,ae,Qt,xe,Yt,v,We,go,at,wn=`The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
cross-attention is added between the self-attention layers, following the architecture described in <em>Attention is
all you need</em>_ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz
Kaiser and Illia Polosukhin.`,_o,it,vn=`To behave as an decoder the model needs to be initialized with the <code>is_decoder</code> argument of the configuration set
to <code>True</code>. To be used in a Seq2Seq model, the model needs to initialized with both <code>is_decoder</code> argument and
<code>add_cross_attention</code> set to <code>True</code>; an <code>encoder_hidden_states</code> is then expected as an input to the forward pass.`,bo,lt,Jn='.. _<em>Attention is all you need</em>: <a href="https://huggingface.co/papers/1706.03762" rel="nofollow">https://huggingface.co/papers/1706.03762</a>',yo,dt,Ln=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Mo,ct,Nn=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,To,ne,Ue,ko,mt,$n='The <a href="/docs/transformers/v4.56.2/en/model_doc/roberta-prelayernorm#transformers.RobertaPreLayerNormModel">RobertaPreLayerNormModel</a> forward method, overrides the <code>__call__</code> special method.',wo,ie,Ot,Ze,Dt,$,Ie,vo,pt,Rn="RoBERTa-PreLayerNorm Model with a <code>language modeling</code> head on top for CLM fine-tuning.",Jo,ht,Cn=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Lo,ut,Fn=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,No,X,qe,$o,ft,jn='The <a href="/docs/transformers/v4.56.2/en/model_doc/roberta-prelayernorm#transformers.RobertaPreLayerNormForCausalLM">RobertaPreLayerNormForCausalLM</a> forward method, overrides the <code>__call__</code> special method.',Ro,le,Co,de,Kt,Be,eo,R,Ge,Fo,gt,zn="RoBERTa-PreLayerNorm Model with a <code>language modeling</code> head on top.",jo,_t,Pn=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,zo,bt,xn=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Po,A,Ve,xo,yt,Wn='The <a href="/docs/transformers/v4.56.2/en/model_doc/roberta-prelayernorm#transformers.RobertaPreLayerNormForMaskedLM">RobertaPreLayerNormForMaskedLM</a> forward method, overrides the <code>__call__</code> special method.',Wo,ce,Uo,me,to,He,oo,C,Xe,Zo,Mt,Un=`RoBERTa-PreLayerNorm Model transformer with a sequence classification/regression head on top (a linear layer on top
of the pooled output) e.g. for GLUE tasks.`,Io,Tt,Zn=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,qo,kt,In=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Bo,W,Ae,Go,wt,qn='The <a href="/docs/transformers/v4.56.2/en/model_doc/roberta-prelayernorm#transformers.RobertaPreLayerNormForSequenceClassification">RobertaPreLayerNormForSequenceClassification</a> forward method, overrides the <code>__call__</code> special method.',Vo,pe,Ho,he,Xo,ue,no,Se,so,F,Ee,Ao,vt,Bn=`The Roberta Prelayernorm Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
softmax) e.g. for RocStories/SWAG tasks.`,So,Jt,Gn=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Eo,Lt,Vn=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Qo,S,Qe,Yo,Nt,Hn='The <a href="/docs/transformers/v4.56.2/en/model_doc/roberta-prelayernorm#transformers.RobertaPreLayerNormForMultipleChoice">RobertaPreLayerNormForMultipleChoice</a> forward method, overrides the <code>__call__</code> special method.',Oo,fe,Do,ge,ro,Ye,ao,j,Oe,Ko,$t,Xn=`The Roberta Prelayernorm transformer with a token classification head on top (a linear layer on top of the hidden-states
output) e.g. for Named-Entity-Recognition (NER) tasks.`,en,Rt,An=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,tn,Ct,Sn=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,on,E,De,nn,Ft,En='The <a href="/docs/transformers/v4.56.2/en/model_doc/roberta-prelayernorm#transformers.RobertaPreLayerNormForTokenClassification">RobertaPreLayerNormForTokenClassification</a> forward method, overrides the <code>__call__</code> special method.',sn,_e,rn,be,io,Ke,lo,z,et,an,jt,Qn=`The Roberta Prelayernorm transformer with a span classification head on top for extractive question-answering tasks like
SQuAD (a linear layer on top of the hidden-states output to compute <code>span start logits</code> and <code>span end logits</code>).`,ln,zt,Yn=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,dn,Pt,On=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,cn,Q,tt,mn,xt,Dn='The <a href="/docs/transformers/v4.56.2/en/model_doc/roberta-prelayernorm#transformers.RobertaPreLayerNormForQuestionAnswering">RobertaPreLayerNormForQuestionAnswering</a> forward method, overrides the <code>__call__</code> special method.',pn,ye,hn,Me,co,ot,mo,Wt,po;return we=new Y({props:{title:"RoBERTa-PreLayerNorm",local:"roberta-prelayernorm",headingTag:"h1"}}),ve=new Y({props:{title:"Overview",local:"overview",headingTag:"h2"}}),Re=new Y({props:{title:"Usage tips",local:"usage-tips",headingTag:"h2"}}),Fe=new Y({props:{title:"Resources",local:"resources",headingTag:"h2"}}),ze=new Y({props:{title:"RobertaPreLayerNormConfig",local:"transformers.RobertaPreLayerNormConfig",headingTag:"h2"}}),Pe=new P({props:{name:"class transformers.RobertaPreLayerNormConfig",anchor:"transformers.RobertaPreLayerNormConfig",parameters:[{name:"vocab_size",val:" = 50265"},{name:"hidden_size",val:" = 768"},{name:"num_hidden_layers",val:" = 12"},{name:"num_attention_heads",val:" = 12"},{name:"intermediate_size",val:" = 3072"},{name:"hidden_act",val:" = 'gelu'"},{name:"hidden_dropout_prob",val:" = 0.1"},{name:"attention_probs_dropout_prob",val:" = 0.1"},{name:"max_position_embeddings",val:" = 512"},{name:"type_vocab_size",val:" = 2"},{name:"initializer_range",val:" = 0.02"},{name:"layer_norm_eps",val:" = 1e-12"},{name:"pad_token_id",val:" = 1"},{name:"bos_token_id",val:" = 0"},{name:"eos_token_id",val:" = 2"},{name:"position_embedding_type",val:" = 'absolute'"},{name:"use_cache",val:" = True"},{name:"classifier_dropout",val:" = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.RobertaPreLayerNormConfig.vocab_size",description:`<strong>vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 50265) &#x2014;
Vocabulary size of the RoBERTa-PreLayerNorm model. Defines the number of different tokens that can be represented by the
<code>inputs_ids</code> passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/roberta-prelayernorm#transformers.RobertaPreLayerNormModel">RobertaPreLayerNormModel</a> or <code>TFRobertaPreLayerNormModel</code>.`,name:"vocab_size"},{anchor:"transformers.RobertaPreLayerNormConfig.hidden_size",description:`<strong>hidden_size</strong> (<code>int</code>, <em>optional</em>, defaults to 768) &#x2014;
Dimensionality of the encoder layers and the pooler layer.`,name:"hidden_size"},{anchor:"transformers.RobertaPreLayerNormConfig.num_hidden_layers",description:`<strong>num_hidden_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 12) &#x2014;
Number of hidden layers in the Transformer encoder.`,name:"num_hidden_layers"},{anchor:"transformers.RobertaPreLayerNormConfig.num_attention_heads",description:`<strong>num_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 12) &#x2014;
Number of attention heads for each attention layer in the Transformer encoder.`,name:"num_attention_heads"},{anchor:"transformers.RobertaPreLayerNormConfig.intermediate_size",description:`<strong>intermediate_size</strong> (<code>int</code>, <em>optional</em>, defaults to 3072) &#x2014;
Dimensionality of the &#x201C;intermediate&#x201D; (often named feed-forward) layer in the Transformer encoder.`,name:"intermediate_size"},{anchor:"transformers.RobertaPreLayerNormConfig.hidden_act",description:`<strong>hidden_act</strong> (<code>str</code> or <code>Callable</code>, <em>optional</em>, defaults to <code>&quot;gelu&quot;</code>) &#x2014;
The non-linear activation function (function or string) in the encoder and pooler. If string, <code>&quot;gelu&quot;</code>,
<code>&quot;relu&quot;</code>, <code>&quot;silu&quot;</code> and <code>&quot;gelu_new&quot;</code> are supported.`,name:"hidden_act"},{anchor:"transformers.RobertaPreLayerNormConfig.hidden_dropout_prob",description:`<strong>hidden_dropout_prob</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.`,name:"hidden_dropout_prob"},{anchor:"transformers.RobertaPreLayerNormConfig.attention_probs_dropout_prob",description:`<strong>attention_probs_dropout_prob</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout ratio for the attention probabilities.`,name:"attention_probs_dropout_prob"},{anchor:"transformers.RobertaPreLayerNormConfig.max_position_embeddings",description:`<strong>max_position_embeddings</strong> (<code>int</code>, <em>optional</em>, defaults to 512) &#x2014;
The maximum sequence length that this model might ever be used with. Typically set this to something large
just in case (e.g., 512 or 1024 or 2048).`,name:"max_position_embeddings"},{anchor:"transformers.RobertaPreLayerNormConfig.type_vocab_size",description:`<strong>type_vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 2) &#x2014;
The vocabulary size of the <code>token_type_ids</code> passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/roberta-prelayernorm#transformers.RobertaPreLayerNormModel">RobertaPreLayerNormModel</a> or <code>TFRobertaPreLayerNormModel</code>.`,name:"type_vocab_size"},{anchor:"transformers.RobertaPreLayerNormConfig.initializer_range",description:`<strong>initializer_range</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.`,name:"initializer_range"},{anchor:"transformers.RobertaPreLayerNormConfig.layer_norm_eps",description:`<strong>layer_norm_eps</strong> (<code>float</code>, <em>optional</em>, defaults to 1e-12) &#x2014;
The epsilon used by the layer normalization layers.`,name:"layer_norm_eps"},{anchor:"transformers.RobertaPreLayerNormConfig.position_embedding_type",description:`<strong>position_embedding_type</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;absolute&quot;</code>) &#x2014;
Type of position embedding. Choose one of <code>&quot;absolute&quot;</code>, <code>&quot;relative_key&quot;</code>, <code>&quot;relative_key_query&quot;</code>. For
positional embeddings use <code>&quot;absolute&quot;</code>. For more information on <code>&quot;relative_key&quot;</code>, please refer to
<a href="https://huggingface.co/papers/1803.02155" rel="nofollow">Self-Attention with Relative Position Representations (Shaw et al.)</a>.
For more information on <code>&quot;relative_key_query&quot;</code>, please refer to <em>Method 4</em> in <a href="https://huggingface.co/papers/2009.13658" rel="nofollow">Improve Transformer Models
with Better Relative Position Embeddings (Huang et al.)</a>.`,name:"position_embedding_type"},{anchor:"transformers.RobertaPreLayerNormConfig.is_decoder",description:`<strong>is_decoder</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether the model is used as a decoder or not. If <code>False</code>, the model is used as an encoder.`,name:"is_decoder"},{anchor:"transformers.RobertaPreLayerNormConfig.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not the model should return the last key/values attentions (not used by all models). Only
relevant if <code>config.is_decoder=True</code>.`,name:"use_cache"},{anchor:"transformers.RobertaPreLayerNormConfig.classifier_dropout",description:`<strong>classifier_dropout</strong> (<code>float</code>, <em>optional</em>) &#x2014;
The dropout ratio for the classification head.`,name:"classifier_dropout"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/roberta_prelayernorm/configuration_roberta_prelayernorm.py#L30"}}),ae=new Te({props:{anchor:"transformers.RobertaPreLayerNormConfig.example",$$slots:{default:[rs]},$$scope:{ctx:w}}}),xe=new Y({props:{title:"RobertaPreLayerNormModel",local:"transformers.RobertaPreLayerNormModel",headingTag:"h2"}}),We=new P({props:{name:"class transformers.RobertaPreLayerNormModel",anchor:"transformers.RobertaPreLayerNormModel",parameters:[{name:"config",val:""},{name:"add_pooling_layer",val:" = True"}],parametersDescription:[{anchor:"transformers.RobertaPreLayerNormModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/roberta-prelayernorm#transformers.RobertaPreLayerNormModel">RobertaPreLayerNormModel</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"},{anchor:"transformers.RobertaPreLayerNormModel.add_pooling_layer",description:`<strong>add_pooling_layer</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether to add a pooling layer`,name:"add_pooling_layer"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/roberta_prelayernorm/modeling_roberta_prelayernorm.py#L595"}}),Ue=new P({props:{name:"forward",anchor:"transformers.RobertaPreLayerNormModel.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_hidden_states",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"past_key_values",val:": typing.Optional[list[torch.FloatTensor]] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.RobertaPreLayerNormModel.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.RobertaPreLayerNormModel.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.RobertaPreLayerNormModel.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0,1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.
This parameter can only be used when the model is initialized with <code>type_vocab_size</code> parameter with value<blockquote>
<p>= 2. All the value in this tensor should be always &lt; type_vocab_size.</p>
</blockquote></li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.RobertaPreLayerNormModel.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.RobertaPreLayerNormModel.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.RobertaPreLayerNormModel.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.RobertaPreLayerNormModel.forward.encoder_hidden_states",description:`<strong>encoder_hidden_states</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
if the model is configured as a decoder.`,name:"encoder_hidden_states"},{anchor:"transformers.RobertaPreLayerNormModel.forward.encoder_attention_mask",description:`<strong>encoder_attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
the cross-attention if the model is configured as a decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>`,name:"encoder_attention_mask"},{anchor:"transformers.RobertaPreLayerNormModel.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>list[torch.FloatTensor]</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.RobertaPreLayerNormModel.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.RobertaPreLayerNormModel.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.RobertaPreLayerNormModel.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.RobertaPreLayerNormModel.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/roberta_prelayernorm/modeling_roberta_prelayernorm.py#L627",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions"
>transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/roberta-prelayernorm#transformers.RobertaPreLayerNormConfig"
>RobertaPreLayerNormConfig</a>) and inputs.</p>
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
`}}),ie=new nt({props:{$$slots:{default:[as]},$$scope:{ctx:w}}}),Ze=new Y({props:{title:"RobertaPreLayerNormForCausalLM",local:"transformers.RobertaPreLayerNormForCausalLM",headingTag:"h2"}}),Ie=new P({props:{name:"class transformers.RobertaPreLayerNormForCausalLM",anchor:"transformers.RobertaPreLayerNormForCausalLM",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.RobertaPreLayerNormForCausalLM.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/roberta-prelayernorm#transformers.RobertaPreLayerNormForCausalLM">RobertaPreLayerNormForCausalLM</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/roberta_prelayernorm/modeling_roberta_prelayernorm.py#L762"}}),qe=new P({props:{name:"forward",anchor:"transformers.RobertaPreLayerNormForCausalLM.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"encoder_hidden_states",val:": typing.Optional[torch.FloatTensor] = None"},{name:"encoder_attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[tuple[tuple[torch.FloatTensor]]] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.RobertaPreLayerNormForCausalLM.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.RobertaPreLayerNormForCausalLM.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.RobertaPreLayerNormForCausalLM.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0,1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.
This parameter can only be used when the model is initialized with <code>type_vocab_size</code> parameter with value<blockquote>
<p>= 2. All the value in this tensor should be always &lt; type_vocab_size.</p>
</blockquote></li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.RobertaPreLayerNormForCausalLM.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.RobertaPreLayerNormForCausalLM.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.RobertaPreLayerNormForCausalLM.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.RobertaPreLayerNormForCausalLM.forward.encoder_hidden_states",description:`<strong>encoder_hidden_states</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
if the model is configured as a decoder.`,name:"encoder_hidden_states"},{anchor:"transformers.RobertaPreLayerNormForCausalLM.forward.encoder_attention_mask",description:`<strong>encoder_attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
the cross-attention if the model is configured as a decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>`,name:"encoder_attention_mask"},{anchor:"transformers.RobertaPreLayerNormForCausalLM.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the left-to-right language modeling loss (next word prediction). Indices should be in
<code>[-100, 0, ..., config.vocab_size]</code> (see <code>input_ids</code> docstring) Tokens with indices set to <code>-100</code> are
ignored (masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>`,name:"labels"},{anchor:"transformers.RobertaPreLayerNormForCausalLM.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>tuple[tuple[torch.FloatTensor]]</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.RobertaPreLayerNormForCausalLM.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.RobertaPreLayerNormForCausalLM.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.RobertaPreLayerNormForCausalLM.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.RobertaPreLayerNormForCausalLM.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/roberta_prelayernorm/modeling_roberta_prelayernorm.py#L785",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithCrossAttentions"
>transformers.modeling_outputs.CausalLMOutputWithCrossAttentions</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/roberta-prelayernorm#transformers.RobertaPreLayerNormConfig"
>RobertaPreLayerNormConfig</a>) and inputs.</p>
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
`}}),le=new nt({props:{$$slots:{default:[is]},$$scope:{ctx:w}}}),de=new Te({props:{anchor:"transformers.RobertaPreLayerNormForCausalLM.forward.example",$$slots:{default:[ls]},$$scope:{ctx:w}}}),Be=new Y({props:{title:"RobertaPreLayerNormForMaskedLM",local:"transformers.RobertaPreLayerNormForMaskedLM",headingTag:"h2"}}),Ge=new P({props:{name:"class transformers.RobertaPreLayerNormForMaskedLM",anchor:"transformers.RobertaPreLayerNormForMaskedLM",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.RobertaPreLayerNormForMaskedLM.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/roberta-prelayernorm#transformers.RobertaPreLayerNormForMaskedLM">RobertaPreLayerNormForMaskedLM</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/roberta_prelayernorm/modeling_roberta_prelayernorm.py#L888"}}),Ve=new P({props:{name:"forward",anchor:"transformers.RobertaPreLayerNormForMaskedLM.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"encoder_hidden_states",val:": typing.Optional[torch.FloatTensor] = None"},{name:"encoder_attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.RobertaPreLayerNormForMaskedLM.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.RobertaPreLayerNormForMaskedLM.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.RobertaPreLayerNormForMaskedLM.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0,1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.
This parameter can only be used when the model is initialized with <code>type_vocab_size</code> parameter with value<blockquote>
<p>= 2. All the value in this tensor should be always &lt; type_vocab_size.</p>
</blockquote></li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.RobertaPreLayerNormForMaskedLM.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.RobertaPreLayerNormForMaskedLM.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.RobertaPreLayerNormForMaskedLM.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.RobertaPreLayerNormForMaskedLM.forward.encoder_hidden_states",description:`<strong>encoder_hidden_states</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
if the model is configured as a decoder.`,name:"encoder_hidden_states"},{anchor:"transformers.RobertaPreLayerNormForMaskedLM.forward.encoder_attention_mask",description:`<strong>encoder_attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
the cross-attention if the model is configured as a decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>`,name:"encoder_attention_mask"},{anchor:"transformers.RobertaPreLayerNormForMaskedLM.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should be in <code>[-100, 0, ..., config.vocab_size]</code> (see <code>input_ids</code> docstring) Tokens with indices set to <code>-100</code> are ignored (masked), the
loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>`,name:"labels"},{anchor:"transformers.RobertaPreLayerNormForMaskedLM.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.RobertaPreLayerNormForMaskedLM.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.RobertaPreLayerNormForMaskedLM.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/roberta_prelayernorm/modeling_roberta_prelayernorm.py#L913",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.MaskedLMOutput"
>transformers.modeling_outputs.MaskedLMOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/roberta-prelayernorm#transformers.RobertaPreLayerNormConfig"
>RobertaPreLayerNormConfig</a>) and inputs.</p>
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
`}}),ce=new nt({props:{$$slots:{default:[ds]},$$scope:{ctx:w}}}),me=new Te({props:{anchor:"transformers.RobertaPreLayerNormForMaskedLM.forward.example",$$slots:{default:[cs]},$$scope:{ctx:w}}}),He=new Y({props:{title:"RobertaPreLayerNormForSequenceClassification",local:"transformers.RobertaPreLayerNormForSequenceClassification",headingTag:"h2"}}),Xe=new P({props:{name:"class transformers.RobertaPreLayerNormForSequenceClassification",anchor:"transformers.RobertaPreLayerNormForSequenceClassification",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.RobertaPreLayerNormForSequenceClassification.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/roberta-prelayernorm#transformers.RobertaPreLayerNormForSequenceClassification">RobertaPreLayerNormForSequenceClassification</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/roberta_prelayernorm/modeling_roberta_prelayernorm.py#L1020"}}),Ae=new P({props:{name:"forward",anchor:"transformers.RobertaPreLayerNormForSequenceClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.RobertaPreLayerNormForSequenceClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.RobertaPreLayerNormForSequenceClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.RobertaPreLayerNormForSequenceClassification.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0,1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.
This parameter can only be used when the model is initialized with <code>type_vocab_size</code> parameter with value<blockquote>
<p>= 2. All the value in this tensor should be always &lt; type_vocab_size.</p>
</blockquote></li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.RobertaPreLayerNormForSequenceClassification.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.RobertaPreLayerNormForSequenceClassification.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.RobertaPreLayerNormForSequenceClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.RobertaPreLayerNormForSequenceClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the sequence classification/regression loss. Indices should be in <code>[0, ..., config.num_labels - 1]</code>. If <code>config.num_labels == 1</code> a regression loss is computed (Mean-Square loss), If
<code>config.num_labels &gt; 1</code> a classification loss is computed (Cross-Entropy).`,name:"labels"},{anchor:"transformers.RobertaPreLayerNormForSequenceClassification.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.RobertaPreLayerNormForSequenceClassification.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.RobertaPreLayerNormForSequenceClassification.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/roberta_prelayernorm/modeling_roberta_prelayernorm.py#L1032",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput"
>transformers.modeling_outputs.SequenceClassifierOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/roberta-prelayernorm#transformers.RobertaPreLayerNormConfig"
>RobertaPreLayerNormConfig</a>) and inputs.</p>
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
`}}),pe=new nt({props:{$$slots:{default:[ms]},$$scope:{ctx:w}}}),he=new Te({props:{anchor:"transformers.RobertaPreLayerNormForSequenceClassification.forward.example",$$slots:{default:[ps]},$$scope:{ctx:w}}}),ue=new Te({props:{anchor:"transformers.RobertaPreLayerNormForSequenceClassification.forward.example-2",$$slots:{default:[hs]},$$scope:{ctx:w}}}),Se=new Y({props:{title:"RobertaPreLayerNormForMultipleChoice",local:"transformers.RobertaPreLayerNormForMultipleChoice",headingTag:"h2"}}),Ee=new P({props:{name:"class transformers.RobertaPreLayerNormForMultipleChoice",anchor:"transformers.RobertaPreLayerNormForMultipleChoice",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.RobertaPreLayerNormForMultipleChoice.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/roberta-prelayernorm#transformers.RobertaPreLayerNormForMultipleChoice">RobertaPreLayerNormForMultipleChoice</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/roberta_prelayernorm/modeling_roberta_prelayernorm.py#L1117"}}),Qe=new P({props:{name:"forward",anchor:"transformers.RobertaPreLayerNormForMultipleChoice.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.RobertaPreLayerNormForMultipleChoice.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, num_choices, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.RobertaPreLayerNormForMultipleChoice.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, num_choices, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0,1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.
This parameter can only be used when the model is initialized with <code>type_vocab_size</code> parameter with value<blockquote>
<p>= 2. All the value in this tensor should be always &lt; type_vocab_size.</p>
</blockquote></li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.RobertaPreLayerNormForMultipleChoice.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.RobertaPreLayerNormForMultipleChoice.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the multiple choice classification loss. Indices should be in <code>[0, ..., num_choices-1]</code> where <code>num_choices</code> is the size of the second dimension of the input tensors. (See
<code>input_ids</code> above)`,name:"labels"},{anchor:"transformers.RobertaPreLayerNormForMultipleChoice.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, num_choices, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.max_position_embeddings - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.RobertaPreLayerNormForMultipleChoice.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.RobertaPreLayerNormForMultipleChoice.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_choices, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.RobertaPreLayerNormForMultipleChoice.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.RobertaPreLayerNormForMultipleChoice.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.RobertaPreLayerNormForMultipleChoice.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/roberta_prelayernorm/modeling_roberta_prelayernorm.py#L1128",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.MultipleChoiceModelOutput"
>transformers.modeling_outputs.MultipleChoiceModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/roberta-prelayernorm#transformers.RobertaPreLayerNormConfig"
>RobertaPreLayerNormConfig</a>) and inputs.</p>
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
`}}),fe=new nt({props:{$$slots:{default:[us]},$$scope:{ctx:w}}}),ge=new Te({props:{anchor:"transformers.RobertaPreLayerNormForMultipleChoice.forward.example",$$slots:{default:[fs]},$$scope:{ctx:w}}}),Ye=new Y({props:{title:"RobertaPreLayerNormForTokenClassification",local:"transformers.RobertaPreLayerNormForTokenClassification",headingTag:"h2"}}),Oe=new P({props:{name:"class transformers.RobertaPreLayerNormForTokenClassification",anchor:"transformers.RobertaPreLayerNormForTokenClassification",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.RobertaPreLayerNormForTokenClassification.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/roberta-prelayernorm#transformers.RobertaPreLayerNormForTokenClassification">RobertaPreLayerNormForTokenClassification</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/roberta_prelayernorm/modeling_roberta_prelayernorm.py#L1223"}}),De=new P({props:{name:"forward",anchor:"transformers.RobertaPreLayerNormForTokenClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.RobertaPreLayerNormForTokenClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.RobertaPreLayerNormForTokenClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.RobertaPreLayerNormForTokenClassification.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0,1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.
This parameter can only be used when the model is initialized with <code>type_vocab_size</code> parameter with value<blockquote>
<p>= 2. All the value in this tensor should be always &lt; type_vocab_size.</p>
</blockquote></li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.RobertaPreLayerNormForTokenClassification.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.RobertaPreLayerNormForTokenClassification.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.RobertaPreLayerNormForTokenClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.RobertaPreLayerNormForTokenClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the token classification loss. Indices should be in <code>[0, ..., config.num_labels - 1]</code>.`,name:"labels"},{anchor:"transformers.RobertaPreLayerNormForTokenClassification.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.RobertaPreLayerNormForTokenClassification.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.RobertaPreLayerNormForTokenClassification.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/roberta_prelayernorm/modeling_roberta_prelayernorm.py#L1238",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput"
>transformers.modeling_outputs.TokenClassifierOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/roberta-prelayernorm#transformers.RobertaPreLayerNormConfig"
>RobertaPreLayerNormConfig</a>) and inputs.</p>
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
`}}),_e=new nt({props:{$$slots:{default:[gs]},$$scope:{ctx:w}}}),be=new Te({props:{anchor:"transformers.RobertaPreLayerNormForTokenClassification.forward.example",$$slots:{default:[_s]},$$scope:{ctx:w}}}),Ke=new Y({props:{title:"RobertaPreLayerNormForQuestionAnswering",local:"transformers.RobertaPreLayerNormForQuestionAnswering",headingTag:"h2"}}),et=new P({props:{name:"class transformers.RobertaPreLayerNormForQuestionAnswering",anchor:"transformers.RobertaPreLayerNormForQuestionAnswering",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.RobertaPreLayerNormForQuestionAnswering.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/roberta-prelayernorm#transformers.RobertaPreLayerNormForQuestionAnswering">RobertaPreLayerNormForQuestionAnswering</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/roberta_prelayernorm/modeling_roberta_prelayernorm.py#L1328"}}),tt=new P({props:{name:"forward",anchor:"transformers.RobertaPreLayerNormForQuestionAnswering.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"start_positions",val:": typing.Optional[torch.LongTensor] = None"},{name:"end_positions",val:": typing.Optional[torch.LongTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.RobertaPreLayerNormForQuestionAnswering.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.RobertaPreLayerNormForQuestionAnswering.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.RobertaPreLayerNormForQuestionAnswering.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0,1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.
This parameter can only be used when the model is initialized with <code>type_vocab_size</code> parameter with value<blockquote>
<p>= 2. All the value in this tensor should be always &lt; type_vocab_size.</p>
</blockquote></li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.RobertaPreLayerNormForQuestionAnswering.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.RobertaPreLayerNormForQuestionAnswering.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.RobertaPreLayerNormForQuestionAnswering.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.RobertaPreLayerNormForQuestionAnswering.forward.start_positions",description:`<strong>start_positions</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for position (index) of the start of the labelled span for computing the token classification loss.
Positions are clamped to the length of the sequence (<code>sequence_length</code>). Position outside of the sequence
are not taken into account for computing the loss.`,name:"start_positions"},{anchor:"transformers.RobertaPreLayerNormForQuestionAnswering.forward.end_positions",description:`<strong>end_positions</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for position (index) of the end of the labelled span for computing the token classification loss.
Positions are clamped to the length of the sequence (<code>sequence_length</code>). Position outside of the sequence
are not taken into account for computing the loss.`,name:"end_positions"},{anchor:"transformers.RobertaPreLayerNormForQuestionAnswering.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.RobertaPreLayerNormForQuestionAnswering.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.RobertaPreLayerNormForQuestionAnswering.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/roberta_prelayernorm/modeling_roberta_prelayernorm.py#L1339",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.QuestionAnsweringModelOutput"
>transformers.modeling_outputs.QuestionAnsweringModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/roberta-prelayernorm#transformers.RobertaPreLayerNormConfig"
>RobertaPreLayerNormConfig</a>) and inputs.</p>
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
`}}),ye=new nt({props:{$$slots:{default:[bs]},$$scope:{ctx:w}}}),Me=new Te({props:{anchor:"transformers.RobertaPreLayerNormForQuestionAnswering.forward.example",$$slots:{default:[ys]},$$scope:{ctx:w}}}),ot=new ss({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/roberta-prelayernorm.md"}}),{c(){t=c("meta"),M=r(),l=c("p"),d=r(),T=c("p"),T.innerHTML=o,k=r(),u(we.$$.fragment),Ut=r(),re=c("div"),re.innerHTML=un,Zt=r(),u(ve.$$.fragment),It=r(),Je=c("p"),Je.innerHTML=fn,qt=r(),Le=c("p"),Le.textContent=gn,Bt=r(),Ne=c("p"),Ne.innerHTML=_n,Gt=r(),$e=c("p"),$e.innerHTML=bn,Vt=r(),u(Re.$$.fragment),Ht=r(),Ce=c("ul"),Ce.innerHTML=yn,Xt=r(),u(Fe.$$.fragment),At=r(),je=c("ul"),je.innerHTML=Mn,St=r(),u(ze.$$.fragment),Et=r(),U=c("div"),u(Pe.$$.fragment),ho=r(),st=c("p"),st.innerHTML=Tn,uo=r(),rt=c("p"),rt.innerHTML=kn,fo=r(),u(ae.$$.fragment),Qt=r(),u(xe.$$.fragment),Yt=r(),v=c("div"),u(We.$$.fragment),go=r(),at=c("p"),at.innerHTML=wn,_o=r(),it=c("p"),it.innerHTML=vn,bo=r(),lt=c("p"),lt.innerHTML=Jn,yo=r(),dt=c("p"),dt.innerHTML=Ln,Mo=r(),ct=c("p"),ct.innerHTML=Nn,To=r(),ne=c("div"),u(Ue.$$.fragment),ko=r(),mt=c("p"),mt.innerHTML=$n,wo=r(),u(ie.$$.fragment),Ot=r(),u(Ze.$$.fragment),Dt=r(),$=c("div"),u(Ie.$$.fragment),vo=r(),pt=c("p"),pt.innerHTML=Rn,Jo=r(),ht=c("p"),ht.innerHTML=Cn,Lo=r(),ut=c("p"),ut.innerHTML=Fn,No=r(),X=c("div"),u(qe.$$.fragment),$o=r(),ft=c("p"),ft.innerHTML=jn,Ro=r(),u(le.$$.fragment),Co=r(),u(de.$$.fragment),Kt=r(),u(Be.$$.fragment),eo=r(),R=c("div"),u(Ge.$$.fragment),Fo=r(),gt=c("p"),gt.innerHTML=zn,jo=r(),_t=c("p"),_t.innerHTML=Pn,zo=r(),bt=c("p"),bt.innerHTML=xn,Po=r(),A=c("div"),u(Ve.$$.fragment),xo=r(),yt=c("p"),yt.innerHTML=Wn,Wo=r(),u(ce.$$.fragment),Uo=r(),u(me.$$.fragment),to=r(),u(He.$$.fragment),oo=r(),C=c("div"),u(Xe.$$.fragment),Zo=r(),Mt=c("p"),Mt.textContent=Un,Io=r(),Tt=c("p"),Tt.innerHTML=Zn,qo=r(),kt=c("p"),kt.innerHTML=In,Bo=r(),W=c("div"),u(Ae.$$.fragment),Go=r(),wt=c("p"),wt.innerHTML=qn,Vo=r(),u(pe.$$.fragment),Ho=r(),u(he.$$.fragment),Xo=r(),u(ue.$$.fragment),no=r(),u(Se.$$.fragment),so=r(),F=c("div"),u(Ee.$$.fragment),Ao=r(),vt=c("p"),vt.textContent=Bn,So=r(),Jt=c("p"),Jt.innerHTML=Gn,Eo=r(),Lt=c("p"),Lt.innerHTML=Vn,Qo=r(),S=c("div"),u(Qe.$$.fragment),Yo=r(),Nt=c("p"),Nt.innerHTML=Hn,Oo=r(),u(fe.$$.fragment),Do=r(),u(ge.$$.fragment),ro=r(),u(Ye.$$.fragment),ao=r(),j=c("div"),u(Oe.$$.fragment),Ko=r(),$t=c("p"),$t.textContent=Xn,en=r(),Rt=c("p"),Rt.innerHTML=An,tn=r(),Ct=c("p"),Ct.innerHTML=Sn,on=r(),E=c("div"),u(De.$$.fragment),nn=r(),Ft=c("p"),Ft.innerHTML=En,sn=r(),u(_e.$$.fragment),rn=r(),u(be.$$.fragment),io=r(),u(Ke.$$.fragment),lo=r(),z=c("div"),u(et.$$.fragment),an=r(),jt=c("p"),jt.innerHTML=Qn,ln=r(),zt=c("p"),zt.innerHTML=Yn,dn=r(),Pt=c("p"),Pt.innerHTML=On,cn=r(),Q=c("div"),u(tt.$$.fragment),mn=r(),xt=c("p"),xt.innerHTML=Dn,pn=r(),u(ye.$$.fragment),hn=r(),u(Me.$$.fragment),co=r(),u(ot.$$.fragment),mo=r(),Wt=c("p"),this.h()},l(e){const n=ns("svelte-u9bgzb",document.head);t=m(n,"META",{name:!0,content:!0}),n.forEach(i),M=a(e),l=m(e,"P",{}),N(l).forEach(i),d=a(e),T=m(e,"P",{"data-svelte-h":!0}),h(T)!=="svelte-dubu0u"&&(T.innerHTML=o),k=a(e),f(we.$$.fragment,e),Ut=a(e),re=m(e,"DIV",{class:!0,"data-svelte-h":!0}),h(re)!=="svelte-13t8s2t"&&(re.innerHTML=un),Zt=a(e),f(ve.$$.fragment,e),It=a(e),Je=m(e,"P",{"data-svelte-h":!0}),h(Je)!=="svelte-1ty19k8"&&(Je.innerHTML=fn),qt=a(e),Le=m(e,"P",{"data-svelte-h":!0}),h(Le)!=="svelte-vfdo9a"&&(Le.textContent=gn),Bt=a(e),Ne=m(e,"P",{"data-svelte-h":!0}),h(Ne)!=="svelte-b3c4v1"&&(Ne.innerHTML=_n),Gt=a(e),$e=m(e,"P",{"data-svelte-h":!0}),h($e)!=="svelte-3d0kzc"&&($e.innerHTML=bn),Vt=a(e),f(Re.$$.fragment,e),Ht=a(e),Ce=m(e,"UL",{"data-svelte-h":!0}),h(Ce)!=="svelte-xarlqa"&&(Ce.innerHTML=yn),Xt=a(e),f(Fe.$$.fragment,e),At=a(e),je=m(e,"UL",{"data-svelte-h":!0}),h(je)!=="svelte-p1b16m"&&(je.innerHTML=Mn),St=a(e),f(ze.$$.fragment,e),Et=a(e),U=m(e,"DIV",{class:!0});var O=N(U);f(Pe.$$.fragment,O),ho=a(O),st=m(O,"P",{"data-svelte-h":!0}),h(st)!=="svelte-oahmwe"&&(st.innerHTML=Tn),uo=a(O),rt=m(O,"P",{"data-svelte-h":!0}),h(rt)!=="svelte-1ek1ss9"&&(rt.innerHTML=kn),fo=a(O),f(ae.$$.fragment,O),O.forEach(i),Qt=a(e),f(xe.$$.fragment,e),Yt=a(e),v=m(e,"DIV",{class:!0});var L=N(v);f(We.$$.fragment,L),go=a(L),at=m(L,"P",{"data-svelte-h":!0}),h(at)!=="svelte-rehfhh"&&(at.innerHTML=wn),_o=a(L),it=m(L,"P",{"data-svelte-h":!0}),h(it)!=="svelte-174erte"&&(it.innerHTML=vn),bo=a(L),lt=m(L,"P",{"data-svelte-h":!0}),h(lt)!=="svelte-joghtx"&&(lt.innerHTML=Jn),yo=a(L),dt=m(L,"P",{"data-svelte-h":!0}),h(dt)!=="svelte-q52n56"&&(dt.innerHTML=Ln),Mo=a(L),ct=m(L,"P",{"data-svelte-h":!0}),h(ct)!=="svelte-hswkmf"&&(ct.innerHTML=Nn),To=a(L),ne=m(L,"DIV",{class:!0});var se=N(ne);f(Ue.$$.fragment,se),ko=a(se),mt=m(se,"P",{"data-svelte-h":!0}),h(mt)!=="svelte-5pars1"&&(mt.innerHTML=$n),wo=a(se),f(ie.$$.fragment,se),se.forEach(i),L.forEach(i),Ot=a(e),f(Ze.$$.fragment,e),Dt=a(e),$=m(e,"DIV",{class:!0});var Z=N($);f(Ie.$$.fragment,Z),vo=a(Z),pt=m(Z,"P",{"data-svelte-h":!0}),h(pt)!=="svelte-1dggich"&&(pt.innerHTML=Rn),Jo=a(Z),ht=m(Z,"P",{"data-svelte-h":!0}),h(ht)!=="svelte-q52n56"&&(ht.innerHTML=Cn),Lo=a(Z),ut=m(Z,"P",{"data-svelte-h":!0}),h(ut)!=="svelte-hswkmf"&&(ut.innerHTML=Fn),No=a(Z),X=m(Z,"DIV",{class:!0});var D=N(X);f(qe.$$.fragment,D),$o=a(D),ft=m(D,"P",{"data-svelte-h":!0}),h(ft)!=="svelte-152g8kx"&&(ft.innerHTML=jn),Ro=a(D),f(le.$$.fragment,D),Co=a(D),f(de.$$.fragment,D),D.forEach(i),Z.forEach(i),Kt=a(e),f(Be.$$.fragment,e),eo=a(e),R=m(e,"DIV",{class:!0});var I=N(R);f(Ge.$$.fragment,I),Fo=a(I),gt=m(I,"P",{"data-svelte-h":!0}),h(gt)!=="svelte-8dtlt4"&&(gt.innerHTML=zn),jo=a(I),_t=m(I,"P",{"data-svelte-h":!0}),h(_t)!=="svelte-q52n56"&&(_t.innerHTML=Pn),zo=a(I),bt=m(I,"P",{"data-svelte-h":!0}),h(bt)!=="svelte-hswkmf"&&(bt.innerHTML=xn),Po=a(I),A=m(I,"DIV",{class:!0});var K=N(A);f(Ve.$$.fragment,K),xo=a(K),yt=m(K,"P",{"data-svelte-h":!0}),h(yt)!=="svelte-5su775"&&(yt.innerHTML=Wn),Wo=a(K),f(ce.$$.fragment,K),Uo=a(K),f(me.$$.fragment,K),K.forEach(i),I.forEach(i),to=a(e),f(He.$$.fragment,e),oo=a(e),C=m(e,"DIV",{class:!0});var q=N(C);f(Xe.$$.fragment,q),Zo=a(q),Mt=m(q,"P",{"data-svelte-h":!0}),h(Mt)!=="svelte-1bv1on4"&&(Mt.textContent=Un),Io=a(q),Tt=m(q,"P",{"data-svelte-h":!0}),h(Tt)!=="svelte-q52n56"&&(Tt.innerHTML=Zn),qo=a(q),kt=m(q,"P",{"data-svelte-h":!0}),h(kt)!=="svelte-hswkmf"&&(kt.innerHTML=In),Bo=a(q),W=m(q,"DIV",{class:!0});var B=N(W);f(Ae.$$.fragment,B),Go=a(B),wt=m(B,"P",{"data-svelte-h":!0}),h(wt)!=="svelte-z4slfz"&&(wt.innerHTML=qn),Vo=a(B),f(pe.$$.fragment,B),Ho=a(B),f(he.$$.fragment,B),Xo=a(B),f(ue.$$.fragment,B),B.forEach(i),q.forEach(i),no=a(e),f(Se.$$.fragment,e),so=a(e),F=m(e,"DIV",{class:!0});var G=N(F);f(Ee.$$.fragment,G),Ao=a(G),vt=m(G,"P",{"data-svelte-h":!0}),h(vt)!=="svelte-1oe4hy"&&(vt.textContent=Bn),So=a(G),Jt=m(G,"P",{"data-svelte-h":!0}),h(Jt)!=="svelte-q52n56"&&(Jt.innerHTML=Gn),Eo=a(G),Lt=m(G,"P",{"data-svelte-h":!0}),h(Lt)!=="svelte-hswkmf"&&(Lt.innerHTML=Vn),Qo=a(G),S=m(G,"DIV",{class:!0});var ee=N(S);f(Qe.$$.fragment,ee),Yo=a(ee),Nt=m(ee,"P",{"data-svelte-h":!0}),h(Nt)!=="svelte-1xjs3ab"&&(Nt.innerHTML=Hn),Oo=a(ee),f(fe.$$.fragment,ee),Do=a(ee),f(ge.$$.fragment,ee),ee.forEach(i),G.forEach(i),ro=a(e),f(Ye.$$.fragment,e),ao=a(e),j=m(e,"DIV",{class:!0});var V=N(j);f(Oe.$$.fragment,V),Ko=a(V),$t=m(V,"P",{"data-svelte-h":!0}),h($t)!=="svelte-1z0fcgb"&&($t.textContent=Xn),en=a(V),Rt=m(V,"P",{"data-svelte-h":!0}),h(Rt)!=="svelte-q52n56"&&(Rt.innerHTML=An),tn=a(V),Ct=m(V,"P",{"data-svelte-h":!0}),h(Ct)!=="svelte-hswkmf"&&(Ct.innerHTML=Sn),on=a(V),E=m(V,"DIV",{class:!0});var te=N(E);f(De.$$.fragment,te),nn=a(te),Ft=m(te,"P",{"data-svelte-h":!0}),h(Ft)!=="svelte-i0vh31"&&(Ft.innerHTML=En),sn=a(te),f(_e.$$.fragment,te),rn=a(te),f(be.$$.fragment,te),te.forEach(i),V.forEach(i),io=a(e),f(Ke.$$.fragment,e),lo=a(e),z=m(e,"DIV",{class:!0});var H=N(z);f(et.$$.fragment,H),an=a(H),jt=m(H,"P",{"data-svelte-h":!0}),h(jt)!=="svelte-y9ihbu"&&(jt.innerHTML=Qn),ln=a(H),zt=m(H,"P",{"data-svelte-h":!0}),h(zt)!=="svelte-q52n56"&&(zt.innerHTML=Yn),dn=a(H),Pt=m(H,"P",{"data-svelte-h":!0}),h(Pt)!=="svelte-hswkmf"&&(Pt.innerHTML=On),cn=a(H),Q=m(H,"DIV",{class:!0});var oe=N(Q);f(tt.$$.fragment,oe),mn=a(oe),xt=m(oe,"P",{"data-svelte-h":!0}),h(xt)!=="svelte-cbxrsl"&&(xt.innerHTML=Dn),pn=a(oe),f(ye.$$.fragment,oe),hn=a(oe),f(Me.$$.fragment,oe),oe.forEach(i),H.forEach(i),co=a(e),f(ot.$$.fragment,e),mo=a(e),Wt=m(e,"P",{}),N(Wt).forEach(i),this.h()},h(){J(t,"name","hf:doc:metadata"),J(t,"content",Ts),J(re,"class","flex flex-wrap space-x-1"),J(U,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(ne,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(v,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(X,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J($,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(A,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(R,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(W,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(C,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(S,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(F,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(E,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(j,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(Q,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(z,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,n){s(document.head,t),p(e,M,n),p(e,l,n),p(e,d,n),p(e,T,n),p(e,k,n),g(we,e,n),p(e,Ut,n),p(e,re,n),p(e,Zt,n),g(ve,e,n),p(e,It,n),p(e,Je,n),p(e,qt,n),p(e,Le,n),p(e,Bt,n),p(e,Ne,n),p(e,Gt,n),p(e,$e,n),p(e,Vt,n),g(Re,e,n),p(e,Ht,n),p(e,Ce,n),p(e,Xt,n),g(Fe,e,n),p(e,At,n),p(e,je,n),p(e,St,n),g(ze,e,n),p(e,Et,n),p(e,U,n),g(Pe,U,null),s(U,ho),s(U,st),s(U,uo),s(U,rt),s(U,fo),g(ae,U,null),p(e,Qt,n),g(xe,e,n),p(e,Yt,n),p(e,v,n),g(We,v,null),s(v,go),s(v,at),s(v,_o),s(v,it),s(v,bo),s(v,lt),s(v,yo),s(v,dt),s(v,Mo),s(v,ct),s(v,To),s(v,ne),g(Ue,ne,null),s(ne,ko),s(ne,mt),s(ne,wo),g(ie,ne,null),p(e,Ot,n),g(Ze,e,n),p(e,Dt,n),p(e,$,n),g(Ie,$,null),s($,vo),s($,pt),s($,Jo),s($,ht),s($,Lo),s($,ut),s($,No),s($,X),g(qe,X,null),s(X,$o),s(X,ft),s(X,Ro),g(le,X,null),s(X,Co),g(de,X,null),p(e,Kt,n),g(Be,e,n),p(e,eo,n),p(e,R,n),g(Ge,R,null),s(R,Fo),s(R,gt),s(R,jo),s(R,_t),s(R,zo),s(R,bt),s(R,Po),s(R,A),g(Ve,A,null),s(A,xo),s(A,yt),s(A,Wo),g(ce,A,null),s(A,Uo),g(me,A,null),p(e,to,n),g(He,e,n),p(e,oo,n),p(e,C,n),g(Xe,C,null),s(C,Zo),s(C,Mt),s(C,Io),s(C,Tt),s(C,qo),s(C,kt),s(C,Bo),s(C,W),g(Ae,W,null),s(W,Go),s(W,wt),s(W,Vo),g(pe,W,null),s(W,Ho),g(he,W,null),s(W,Xo),g(ue,W,null),p(e,no,n),g(Se,e,n),p(e,so,n),p(e,F,n),g(Ee,F,null),s(F,Ao),s(F,vt),s(F,So),s(F,Jt),s(F,Eo),s(F,Lt),s(F,Qo),s(F,S),g(Qe,S,null),s(S,Yo),s(S,Nt),s(S,Oo),g(fe,S,null),s(S,Do),g(ge,S,null),p(e,ro,n),g(Ye,e,n),p(e,ao,n),p(e,j,n),g(Oe,j,null),s(j,Ko),s(j,$t),s(j,en),s(j,Rt),s(j,tn),s(j,Ct),s(j,on),s(j,E),g(De,E,null),s(E,nn),s(E,Ft),s(E,sn),g(_e,E,null),s(E,rn),g(be,E,null),p(e,io,n),g(Ke,e,n),p(e,lo,n),p(e,z,n),g(et,z,null),s(z,an),s(z,jt),s(z,ln),s(z,zt),s(z,dn),s(z,Pt),s(z,cn),s(z,Q),g(tt,Q,null),s(Q,mn),s(Q,xt),s(Q,pn),g(ye,Q,null),s(Q,hn),g(Me,Q,null),p(e,co,n),g(ot,e,n),p(e,mo,n),p(e,Wt,n),po=!0},p(e,[n]){const O={};n&2&&(O.$$scope={dirty:n,ctx:e}),ae.$set(O);const L={};n&2&&(L.$$scope={dirty:n,ctx:e}),ie.$set(L);const se={};n&2&&(se.$$scope={dirty:n,ctx:e}),le.$set(se);const Z={};n&2&&(Z.$$scope={dirty:n,ctx:e}),de.$set(Z);const D={};n&2&&(D.$$scope={dirty:n,ctx:e}),ce.$set(D);const I={};n&2&&(I.$$scope={dirty:n,ctx:e}),me.$set(I);const K={};n&2&&(K.$$scope={dirty:n,ctx:e}),pe.$set(K);const q={};n&2&&(q.$$scope={dirty:n,ctx:e}),he.$set(q);const B={};n&2&&(B.$$scope={dirty:n,ctx:e}),ue.$set(B);const G={};n&2&&(G.$$scope={dirty:n,ctx:e}),fe.$set(G);const ee={};n&2&&(ee.$$scope={dirty:n,ctx:e}),ge.$set(ee);const V={};n&2&&(V.$$scope={dirty:n,ctx:e}),_e.$set(V);const te={};n&2&&(te.$$scope={dirty:n,ctx:e}),be.$set(te);const H={};n&2&&(H.$$scope={dirty:n,ctx:e}),ye.$set(H);const oe={};n&2&&(oe.$$scope={dirty:n,ctx:e}),Me.$set(oe)},i(e){po||(_(we.$$.fragment,e),_(ve.$$.fragment,e),_(Re.$$.fragment,e),_(Fe.$$.fragment,e),_(ze.$$.fragment,e),_(Pe.$$.fragment,e),_(ae.$$.fragment,e),_(xe.$$.fragment,e),_(We.$$.fragment,e),_(Ue.$$.fragment,e),_(ie.$$.fragment,e),_(Ze.$$.fragment,e),_(Ie.$$.fragment,e),_(qe.$$.fragment,e),_(le.$$.fragment,e),_(de.$$.fragment,e),_(Be.$$.fragment,e),_(Ge.$$.fragment,e),_(Ve.$$.fragment,e),_(ce.$$.fragment,e),_(me.$$.fragment,e),_(He.$$.fragment,e),_(Xe.$$.fragment,e),_(Ae.$$.fragment,e),_(pe.$$.fragment,e),_(he.$$.fragment,e),_(ue.$$.fragment,e),_(Se.$$.fragment,e),_(Ee.$$.fragment,e),_(Qe.$$.fragment,e),_(fe.$$.fragment,e),_(ge.$$.fragment,e),_(Ye.$$.fragment,e),_(Oe.$$.fragment,e),_(De.$$.fragment,e),_(_e.$$.fragment,e),_(be.$$.fragment,e),_(Ke.$$.fragment,e),_(et.$$.fragment,e),_(tt.$$.fragment,e),_(ye.$$.fragment,e),_(Me.$$.fragment,e),_(ot.$$.fragment,e),po=!0)},o(e){b(we.$$.fragment,e),b(ve.$$.fragment,e),b(Re.$$.fragment,e),b(Fe.$$.fragment,e),b(ze.$$.fragment,e),b(Pe.$$.fragment,e),b(ae.$$.fragment,e),b(xe.$$.fragment,e),b(We.$$.fragment,e),b(Ue.$$.fragment,e),b(ie.$$.fragment,e),b(Ze.$$.fragment,e),b(Ie.$$.fragment,e),b(qe.$$.fragment,e),b(le.$$.fragment,e),b(de.$$.fragment,e),b(Be.$$.fragment,e),b(Ge.$$.fragment,e),b(Ve.$$.fragment,e),b(ce.$$.fragment,e),b(me.$$.fragment,e),b(He.$$.fragment,e),b(Xe.$$.fragment,e),b(Ae.$$.fragment,e),b(pe.$$.fragment,e),b(he.$$.fragment,e),b(ue.$$.fragment,e),b(Se.$$.fragment,e),b(Ee.$$.fragment,e),b(Qe.$$.fragment,e),b(fe.$$.fragment,e),b(ge.$$.fragment,e),b(Ye.$$.fragment,e),b(Oe.$$.fragment,e),b(De.$$.fragment,e),b(_e.$$.fragment,e),b(be.$$.fragment,e),b(Ke.$$.fragment,e),b(et.$$.fragment,e),b(tt.$$.fragment,e),b(ye.$$.fragment,e),b(Me.$$.fragment,e),b(ot.$$.fragment,e),po=!1},d(e){e&&(i(M),i(l),i(d),i(T),i(k),i(Ut),i(re),i(Zt),i(It),i(Je),i(qt),i(Le),i(Bt),i(Ne),i(Gt),i($e),i(Vt),i(Ht),i(Ce),i(Xt),i(At),i(je),i(St),i(Et),i(U),i(Qt),i(Yt),i(v),i(Ot),i(Dt),i($),i(Kt),i(eo),i(R),i(to),i(oo),i(C),i(no),i(so),i(F),i(ro),i(ao),i(j),i(io),i(lo),i(z),i(co),i(mo),i(Wt)),i(t),y(we,e),y(ve,e),y(Re,e),y(Fe,e),y(ze,e),y(Pe),y(ae),y(xe,e),y(We),y(Ue),y(ie),y(Ze,e),y(Ie),y(qe),y(le),y(de),y(Be,e),y(Ge),y(Ve),y(ce),y(me),y(He,e),y(Xe),y(Ae),y(pe),y(he),y(ue),y(Se,e),y(Ee),y(Qe),y(fe),y(ge),y(Ye,e),y(Oe),y(De),y(_e),y(be),y(Ke,e),y(et),y(tt),y(ye),y(Me),y(ot,e)}}}const Ts='{"title":"RoBERTa-PreLayerNorm","local":"roberta-prelayernorm","sections":[{"title":"Overview","local":"overview","sections":[],"depth":2},{"title":"Usage tips","local":"usage-tips","sections":[],"depth":2},{"title":"Resources","local":"resources","sections":[],"depth":2},{"title":"RobertaPreLayerNormConfig","local":"transformers.RobertaPreLayerNormConfig","sections":[],"depth":2},{"title":"RobertaPreLayerNormModel","local":"transformers.RobertaPreLayerNormModel","sections":[],"depth":2},{"title":"RobertaPreLayerNormForCausalLM","local":"transformers.RobertaPreLayerNormForCausalLM","sections":[],"depth":2},{"title":"RobertaPreLayerNormForMaskedLM","local":"transformers.RobertaPreLayerNormForMaskedLM","sections":[],"depth":2},{"title":"RobertaPreLayerNormForSequenceClassification","local":"transformers.RobertaPreLayerNormForSequenceClassification","sections":[],"depth":2},{"title":"RobertaPreLayerNormForMultipleChoice","local":"transformers.RobertaPreLayerNormForMultipleChoice","sections":[],"depth":2},{"title":"RobertaPreLayerNormForTokenClassification","local":"transformers.RobertaPreLayerNormForTokenClassification","sections":[],"depth":2},{"title":"RobertaPreLayerNormForQuestionAnswering","local":"transformers.RobertaPreLayerNormForQuestionAnswering","sections":[],"depth":2}],"depth":1}';function ks(w){return es(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class Cs extends ts{constructor(t){super(),os(this,t,ks,Ms,Kn,{})}}export{Cs as component};
