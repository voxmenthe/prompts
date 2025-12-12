import{s as Us,o as Ws,n as Z}from"../chunks/scheduler.18a86fab.js";import{S as Is,i as Zs,g as p,s as a,r as u,A as Ls,h as m,f as s,c as r,j,x as h,u as f,k as J,y as i,a as l,v as g,d as _,t as b,w as M}from"../chunks/index.98837b22.js";import{T as _t}from"../chunks/Tip.77304350.js";import{D as I}from"../chunks/Docstring.a1ef7999.js";import{C as O}from"../chunks/CodeBlock.8d0c2e8a.js";import{E as ke}from"../chunks/ExampleCodeBlock.8c3ee1f9.js";import{H as C,E as qs}from"../chunks/getInferenceSnippets.06c2775f.js";function Bs(k){let t,y="Examples:",d,c,T;return c=new O({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMFhtb2RDb25maWclMkMlMjBYbW9kTW9kZWwlMEElMEElMjMlMjBJbml0aWFsaXppbmclMjBhbiUyMFgtTU9EJTIwZmFjZWJvb2slMkZ4bW9kLWJhc2UlMjBzdHlsZSUyMGNvbmZpZ3VyYXRpb24lMEFjb25maWd1cmF0aW9uJTIwJTNEJTIwWG1vZENvbmZpZygpJTBBJTBBJTIzJTIwSW5pdGlhbGl6aW5nJTIwYSUyMG1vZGVsJTIwKHdpdGglMjByYW5kb20lMjB3ZWlnaHRzKSUyMGZyb20lMjB0aGUlMjBmYWNlYm9vayUyRnhtb2QtYmFzZSUyMHN0eWxlJTIwY29uZmlndXJhdGlvbiUwQW1vZGVsJTIwJTNEJTIwWG1vZE1vZGVsKGNvbmZpZ3VyYXRpb24pJTBBJTBBJTIzJTIwQWNjZXNzaW5nJTIwdGhlJTIwbW9kZWwlMjBjb25maWd1cmF0aW9uJTBBY29uZmlndXJhdGlvbiUyMCUzRCUyMG1vZGVsLmNvbmZpZw==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> XmodConfig, XmodModel

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing an X-MOD facebook/xmod-base style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = XmodConfig()

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a model (with random weights) from the facebook/xmod-base style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = XmodModel(configuration)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Accessing the model configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = model.config`,wrap:!1}}),{c(){t=p("p"),t.textContent=y,d=a(),u(c.$$.fragment)},l(n){t=m(n,"P",{"data-svelte-h":!0}),h(t)!=="svelte-kvfsh7"&&(t.textContent=y),d=r(n),f(c.$$.fragment,n)},m(n,w){l(n,t,w),l(n,d,w),g(c,n,w),T=!0},p:Z,i(n){T||(_(c.$$.fragment,n),T=!0)},o(n){b(c.$$.fragment,n),T=!1},d(n){n&&(s(t),s(d)),M(c,n)}}}function Gs(k){let t,y=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=p("p"),t.innerHTML=y},l(d){t=m(d,"P",{"data-svelte-h":!0}),h(t)!=="svelte-fincs2"&&(t.innerHTML=y)},m(d,c){l(d,t,c)},p:Z,d(d){d&&s(t)}}}function Ns(k){let t,y=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=p("p"),t.innerHTML=y},l(d){t=m(d,"P",{"data-svelte-h":!0}),h(t)!=="svelte-fincs2"&&(t.innerHTML=y)},m(d,c){l(d,t,c)},p:Z,d(d){d&&s(t)}}}function Rs(k){let t,y="Example:",d,c,T;return c=new O({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBYbW9kRm9yQ2F1c2FsTE0lMkMlMjBBdXRvQ29uZmlnJTBBaW1wb3J0JTIwdG9yY2glMEElMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJGYWNlYm9va0FJJTJGeGxtLXJvYmVydGEtYmFzZSUyMiklMEFjb25maWclMjAlM0QlMjBBdXRvQ29uZmlnLmZyb21fcHJldHJhaW5lZCglMjJmYWNlYm9vayUyRnhtb2QtYmFzZSUyMiklMEFjb25maWcuaXNfZGVjb2RlciUyMCUzRCUyMFRydWUlMEFtb2RlbCUyMCUzRCUyMFhtb2RGb3JDYXVzYWxMTS5mcm9tX3ByZXRyYWluZWQoJTIyZmFjZWJvb2slMkZ4bW9kLWJhc2UlMjIlMkMlMjBjb25maWclM0Rjb25maWcpJTBBbW9kZWwuc2V0X2RlZmF1bHRfbGFuZ3VhZ2UoJTIyZW5fWFglMjIpJTBBJTBBaW5wdXRzJTIwJTNEJTIwdG9rZW5pemVyKCUyMkhlbGxvJTJDJTIwbXklMjBkb2clMjBpcyUyMGN1dGUlMjIlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQW91dHB1dHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyklMEElMEFwcmVkaWN0aW9uX2xvZ2l0cyUyMCUzRCUyMG91dHB1dHMubG9naXRz",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, XmodForCausalLM, AutoConfig
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;FacebookAI/xlm-roberta-base&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>config = AutoConfig.from_pretrained(<span class="hljs-string">&quot;facebook/xmod-base&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>config.is_decoder = <span class="hljs-literal">True</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = XmodForCausalLM.from_pretrained(<span class="hljs-string">&quot;facebook/xmod-base&quot;</span>, config=config)
<span class="hljs-meta">&gt;&gt;&gt; </span>model.set_default_language(<span class="hljs-string">&quot;en_XX&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs)

<span class="hljs-meta">&gt;&gt;&gt; </span>prediction_logits = outputs.logits`,wrap:!1}}),{c(){t=p("p"),t.textContent=y,d=a(),u(c.$$.fragment)},l(n){t=m(n,"P",{"data-svelte-h":!0}),h(t)!=="svelte-11lpom8"&&(t.textContent=y),d=r(n),f(c.$$.fragment,n)},m(n,w){l(n,t,w),l(n,d,w),g(c,n,w),T=!0},p:Z,i(n){T||(_(c.$$.fragment,n),T=!0)},o(n){b(c.$$.fragment,n),T=!1},d(n){n&&(s(t),s(d)),M(c,n)}}}function Hs(k){let t,y=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=p("p"),t.innerHTML=y},l(d){t=m(d,"P",{"data-svelte-h":!0}),h(t)!=="svelte-fincs2"&&(t.innerHTML=y)},m(d,c){l(d,t,c)},p:Z,d(d){d&&s(t)}}}function Vs(k){let t,y="Example:",d,c,T;return c=new O({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBYbW9kRm9yTWFza2VkTE0lMEFpbXBvcnQlMjB0b3JjaCUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMmZhY2Vib29rJTJGeG1vZC1iYXNlJTIyKSUwQW1vZGVsJTIwJTNEJTIwWG1vZEZvck1hc2tlZExNLmZyb21fcHJldHJhaW5lZCglMjJmYWNlYm9vayUyRnhtb2QtYmFzZSUyMiklMEElMEFpbnB1dHMlMjAlM0QlMjB0b2tlbml6ZXIoJTIyVGhlJTIwY2FwaXRhbCUyMG9mJTIwRnJhbmNlJTIwaXMlMjAlM0NtYXNrJTNFLiUyMiUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpJTBBJTBBd2l0aCUyMHRvcmNoLm5vX2dyYWQoKSUzQSUwQSUyMCUyMCUyMCUyMGxvZ2l0cyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzKS5sb2dpdHMlMEElMEElMjMlMjByZXRyaWV2ZSUyMGluZGV4JTIwb2YlMjAlM0NtYXNrJTNFJTBBbWFza190b2tlbl9pbmRleCUyMCUzRCUyMChpbnB1dHMuaW5wdXRfaWRzJTIwJTNEJTNEJTIwdG9rZW5pemVyLm1hc2tfdG9rZW5faWQpJTVCMCU1RC5ub256ZXJvKGFzX3R1cGxlJTNEVHJ1ZSklNUIwJTVEJTBBJTBBcHJlZGljdGVkX3Rva2VuX2lkJTIwJTNEJTIwbG9naXRzJTVCMCUyQyUyMG1hc2tfdG9rZW5faW5kZXglNUQuYXJnbWF4KGF4aXMlM0QtMSklMEF0b2tlbml6ZXIuZGVjb2RlKHByZWRpY3RlZF90b2tlbl9pZCklMEElMEFsYWJlbHMlMjAlM0QlMjB0b2tlbml6ZXIoJTIyVGhlJTIwY2FwaXRhbCUyMG9mJTIwRnJhbmNlJTIwaXMlMjBQYXJpcy4lMjIlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSU1QiUyMmlucHV0X2lkcyUyMiU1RCUwQSUyMyUyMG1hc2slMjBsYWJlbHMlMjBvZiUyMG5vbi0lM0NtYXNrJTNFJTIwdG9rZW5zJTBBbGFiZWxzJTIwJTNEJTIwdG9yY2gud2hlcmUoaW5wdXRzLmlucHV0X2lkcyUyMCUzRCUzRCUyMHRva2VuaXplci5tYXNrX3Rva2VuX2lkJTJDJTIwbGFiZWxzJTJDJTIwLTEwMCklMEElMEFvdXRwdXRzJTIwJTNEJTIwbW9kZWwoKippbnB1dHMlMkMlMjBsYWJlbHMlM0RsYWJlbHMpJTBBcm91bmQob3V0cHV0cy5sb3NzLml0ZW0oKSUyQyUyMDIp",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, XmodForMaskedLM
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;facebook/xmod-base&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = XmodForMaskedLM.from_pretrained(<span class="hljs-string">&quot;facebook/xmod-base&quot;</span>)

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
...`,wrap:!1}}),{c(){t=p("p"),t.textContent=y,d=a(),u(c.$$.fragment)},l(n){t=m(n,"P",{"data-svelte-h":!0}),h(t)!=="svelte-11lpom8"&&(t.textContent=y),d=r(n),f(c.$$.fragment,n)},m(n,w){l(n,t,w),l(n,d,w),g(c,n,w),T=!0},p:Z,i(n){T||(_(c.$$.fragment,n),T=!0)},o(n){b(c.$$.fragment,n),T=!1},d(n){n&&(s(t),s(d)),M(c,n)}}}function Es(k){let t,y=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=p("p"),t.innerHTML=y},l(d){t=m(d,"P",{"data-svelte-h":!0}),h(t)!=="svelte-fincs2"&&(t.innerHTML=y)},m(d,c){l(d,t,c)},p:Z,d(d){d&&s(t)}}}function As(k){let t,y="Example of single-label classification:",d,c,T;return c=new O({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Rva2VuaXplciUyQyUyMFhtb2RGb3JTZXF1ZW5jZUNsYXNzaWZpY2F0aW9uJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIyZmFjZWJvb2slMkZ4bW9kLWJhc2UlMjIpJTBBbW9kZWwlMjAlM0QlMjBYbW9kRm9yU2VxdWVuY2VDbGFzc2lmaWNhdGlvbi5mcm9tX3ByZXRyYWluZWQoJTIyZmFjZWJvb2slMkZ4bW9kLWJhc2UlMjIpJTBBJTBBaW5wdXRzJTIwJTNEJTIwdG9rZW5pemVyKCUyMkhlbGxvJTJDJTIwbXklMjBkb2clMjBpcyUyMGN1dGUlMjIlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQSUwQXdpdGglMjB0b3JjaC5ub19ncmFkKCklM0ElMEElMjAlMjAlMjAlMjBsb2dpdHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cykubG9naXRzJTBBJTBBcHJlZGljdGVkX2NsYXNzX2lkJTIwJTNEJTIwbG9naXRzLmFyZ21heCgpLml0ZW0oKSUwQW1vZGVsLmNvbmZpZy5pZDJsYWJlbCU1QnByZWRpY3RlZF9jbGFzc19pZCU1RCUwQSUwQSUyMyUyMFRvJTIwdHJhaW4lMjBhJTIwbW9kZWwlMjBvbiUyMCU2MG51bV9sYWJlbHMlNjAlMjBjbGFzc2VzJTJDJTIweW91JTIwY2FuJTIwcGFzcyUyMCU2MG51bV9sYWJlbHMlM0RudW1fbGFiZWxzJTYwJTIwdG8lMjAlNjAuZnJvbV9wcmV0cmFpbmVkKC4uLiklNjAlMEFudW1fbGFiZWxzJTIwJTNEJTIwbGVuKG1vZGVsLmNvbmZpZy5pZDJsYWJlbCklMEFtb2RlbCUyMCUzRCUyMFhtb2RGb3JTZXF1ZW5jZUNsYXNzaWZpY2F0aW9uLmZyb21fcHJldHJhaW5lZCglMjJmYWNlYm9vayUyRnhtb2QtYmFzZSUyMiUyQyUyMG51bV9sYWJlbHMlM0RudW1fbGFiZWxzKSUwQSUwQWxhYmVscyUyMCUzRCUyMHRvcmNoLnRlbnNvciglNUIxJTVEKSUwQWxvc3MlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyUyQyUyMGxhYmVscyUzRGxhYmVscykubG9zcyUwQXJvdW5kKGxvc3MuaXRlbSgpJTJDJTIwMik=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, XmodForSequenceClassification

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;facebook/xmod-base&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = XmodForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;facebook/xmod-base&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    logits = model(**inputs).logits

<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_class_id = logits.argmax().item()
<span class="hljs-meta">&gt;&gt;&gt; </span>model.config.id2label[predicted_class_id]
...

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># To train a model on \`num_labels\` classes, you can pass \`num_labels=num_labels\` to \`.from_pretrained(...)\`</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>num_labels = <span class="hljs-built_in">len</span>(model.config.id2label)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = XmodForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;facebook/xmod-base&quot;</span>, num_labels=num_labels)

<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.tensor([<span class="hljs-number">1</span>])
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = model(**inputs, labels=labels).loss
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">round</span>(loss.item(), <span class="hljs-number">2</span>)
...`,wrap:!1}}),{c(){t=p("p"),t.textContent=y,d=a(),u(c.$$.fragment)},l(n){t=m(n,"P",{"data-svelte-h":!0}),h(t)!=="svelte-ykxpe4"&&(t.textContent=y),d=r(n),f(c.$$.fragment,n)},m(n,w){l(n,t,w),l(n,d,w),g(c,n,w),T=!0},p:Z,i(n){T||(_(c.$$.fragment,n),T=!0)},o(n){b(c.$$.fragment,n),T=!1},d(n){n&&(s(t),s(d)),M(c,n)}}}function Ss(k){let t,y="Example of multi-label classification:",d,c,T;return c=new O({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Rva2VuaXplciUyQyUyMFhtb2RGb3JTZXF1ZW5jZUNsYXNzaWZpY2F0aW9uJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIyZmFjZWJvb2slMkZ4bW9kLWJhc2UlMjIpJTBBbW9kZWwlMjAlM0QlMjBYbW9kRm9yU2VxdWVuY2VDbGFzc2lmaWNhdGlvbi5mcm9tX3ByZXRyYWluZWQoJTIyZmFjZWJvb2slMkZ4bW9kLWJhc2UlMjIlMkMlMjBwcm9ibGVtX3R5cGUlM0QlMjJtdWx0aV9sYWJlbF9jbGFzc2lmaWNhdGlvbiUyMiklMEElMEFpbnB1dHMlMjAlM0QlMjB0b2tlbml6ZXIoJTIySGVsbG8lMkMlMjBteSUyMGRvZyUyMGlzJTIwY3V0ZSUyMiUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpJTBBJTBBd2l0aCUyMHRvcmNoLm5vX2dyYWQoKSUzQSUwQSUyMCUyMCUyMCUyMGxvZ2l0cyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzKS5sb2dpdHMlMEElMEFwcmVkaWN0ZWRfY2xhc3NfaWRzJTIwJTNEJTIwdG9yY2guYXJhbmdlKDAlMkMlMjBsb2dpdHMuc2hhcGUlNUItMSU1RCklNUJ0b3JjaC5zaWdtb2lkKGxvZ2l0cykuc3F1ZWV6ZShkaW0lM0QwKSUyMCUzRSUyMDAuNSU1RCUwQSUwQSUyMyUyMFRvJTIwdHJhaW4lMjBhJTIwbW9kZWwlMjBvbiUyMCU2MG51bV9sYWJlbHMlNjAlMjBjbGFzc2VzJTJDJTIweW91JTIwY2FuJTIwcGFzcyUyMCU2MG51bV9sYWJlbHMlM0RudW1fbGFiZWxzJTYwJTIwdG8lMjAlNjAuZnJvbV9wcmV0cmFpbmVkKC4uLiklNjAlMEFudW1fbGFiZWxzJTIwJTNEJTIwbGVuKG1vZGVsLmNvbmZpZy5pZDJsYWJlbCklMEFtb2RlbCUyMCUzRCUyMFhtb2RGb3JTZXF1ZW5jZUNsYXNzaWZpY2F0aW9uLmZyb21fcHJldHJhaW5lZCglMEElMjAlMjAlMjAlMjAlMjJmYWNlYm9vayUyRnhtb2QtYmFzZSUyMiUyQyUyMG51bV9sYWJlbHMlM0RudW1fbGFiZWxzJTJDJTIwcHJvYmxlbV90eXBlJTNEJTIybXVsdGlfbGFiZWxfY2xhc3NpZmljYXRpb24lMjIlMEEpJTBBJTBBbGFiZWxzJTIwJTNEJTIwdG9yY2guc3VtKCUwQSUyMCUyMCUyMCUyMHRvcmNoLm5uLmZ1bmN0aW9uYWwub25lX2hvdChwcmVkaWN0ZWRfY2xhc3NfaWRzJTVCTm9uZSUyQyUyMCUzQSU1RC5jbG9uZSgpJTJDJTIwbnVtX2NsYXNzZXMlM0RudW1fbGFiZWxzKSUyQyUyMGRpbSUzRDElMEEpLnRvKHRvcmNoLmZsb2F0KSUwQWxvc3MlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyUyQyUyMGxhYmVscyUzRGxhYmVscykubG9zcw==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, XmodForSequenceClassification

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;facebook/xmod-base&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = XmodForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;facebook/xmod-base&quot;</span>, problem_type=<span class="hljs-string">&quot;multi_label_classification&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    logits = model(**inputs).logits

<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_class_ids = torch.arange(<span class="hljs-number">0</span>, logits.shape[-<span class="hljs-number">1</span>])[torch.sigmoid(logits).squeeze(dim=<span class="hljs-number">0</span>) &gt; <span class="hljs-number">0.5</span>]

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># To train a model on \`num_labels\` classes, you can pass \`num_labels=num_labels\` to \`.from_pretrained(...)\`</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>num_labels = <span class="hljs-built_in">len</span>(model.config.id2label)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = XmodForSequenceClassification.from_pretrained(
<span class="hljs-meta">... </span>    <span class="hljs-string">&quot;facebook/xmod-base&quot;</span>, num_labels=num_labels, problem_type=<span class="hljs-string">&quot;multi_label_classification&quot;</span>
<span class="hljs-meta">... </span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.<span class="hljs-built_in">sum</span>(
<span class="hljs-meta">... </span>    torch.nn.functional.one_hot(predicted_class_ids[<span class="hljs-literal">None</span>, :].clone(), num_classes=num_labels), dim=<span class="hljs-number">1</span>
<span class="hljs-meta">... </span>).to(torch.<span class="hljs-built_in">float</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = model(**inputs, labels=labels).loss`,wrap:!1}}),{c(){t=p("p"),t.textContent=y,d=a(),u(c.$$.fragment)},l(n){t=m(n,"P",{"data-svelte-h":!0}),h(t)!=="svelte-1l8e32d"&&(t.textContent=y),d=r(n),f(c.$$.fragment,n)},m(n,w){l(n,t,w),l(n,d,w),g(c,n,w),T=!0},p:Z,i(n){T||(_(c.$$.fragment,n),T=!0)},o(n){b(c.$$.fragment,n),T=!1},d(n){n&&(s(t),s(d)),M(c,n)}}}function Qs(k){let t,y=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=p("p"),t.innerHTML=y},l(d){t=m(d,"P",{"data-svelte-h":!0}),h(t)!=="svelte-fincs2"&&(t.innerHTML=y)},m(d,c){l(d,t,c)},p:Z,d(d){d&&s(t)}}}function Ys(k){let t,y="Example:",d,c,T;return c=new O({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBYbW9kRm9yTXVsdGlwbGVDaG9pY2UlMEFpbXBvcnQlMjB0b3JjaCUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMmZhY2Vib29rJTJGeG1vZC1iYXNlJTIyKSUwQW1vZGVsJTIwJTNEJTIwWG1vZEZvck11bHRpcGxlQ2hvaWNlLmZyb21fcHJldHJhaW5lZCglMjJmYWNlYm9vayUyRnhtb2QtYmFzZSUyMiklMEElMEFwcm9tcHQlMjAlM0QlMjAlMjJJbiUyMEl0YWx5JTJDJTIwcGl6emElMjBzZXJ2ZWQlMjBpbiUyMGZvcm1hbCUyMHNldHRpbmdzJTJDJTIwc3VjaCUyMGFzJTIwYXQlMjBhJTIwcmVzdGF1cmFudCUyQyUyMGlzJTIwcHJlc2VudGVkJTIwdW5zbGljZWQuJTIyJTBBY2hvaWNlMCUyMCUzRCUyMCUyMkl0JTIwaXMlMjBlYXRlbiUyMHdpdGglMjBhJTIwZm9yayUyMGFuZCUyMGElMjBrbmlmZS4lMjIlMEFjaG9pY2UxJTIwJTNEJTIwJTIySXQlMjBpcyUyMGVhdGVuJTIwd2hpbGUlMjBoZWxkJTIwaW4lMjB0aGUlMjBoYW5kLiUyMiUwQWxhYmVscyUyMCUzRCUyMHRvcmNoLnRlbnNvcigwKS51bnNxdWVlemUoMCklMjAlMjAlMjMlMjBjaG9pY2UwJTIwaXMlMjBjb3JyZWN0JTIwKGFjY29yZGluZyUyMHRvJTIwV2lraXBlZGlhJTIwJTNCKSklMkMlMjBiYXRjaCUyMHNpemUlMjAxJTBBJTBBZW5jb2RpbmclMjAlM0QlMjB0b2tlbml6ZXIoJTVCcHJvbXB0JTJDJTIwcHJvbXB0JTVEJTJDJTIwJTVCY2hvaWNlMCUyQyUyMGNob2ljZTElNUQlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyJTJDJTIwcGFkZGluZyUzRFRydWUpJTBBb3V0cHV0cyUyMCUzRCUyMG1vZGVsKCoqJTdCayUzQSUyMHYudW5zcXVlZXplKDApJTIwZm9yJTIwayUyQyUyMHYlMjBpbiUyMGVuY29kaW5nLml0ZW1zKCklN0QlMkMlMjBsYWJlbHMlM0RsYWJlbHMpJTIwJTIwJTIzJTIwYmF0Y2glMjBzaXplJTIwaXMlMjAxJTBBJTBBJTIzJTIwdGhlJTIwbGluZWFyJTIwY2xhc3NpZmllciUyMHN0aWxsJTIwbmVlZHMlMjB0byUyMGJlJTIwdHJhaW5lZCUwQWxvc3MlMjAlM0QlMjBvdXRwdXRzLmxvc3MlMEFsb2dpdHMlMjAlM0QlMjBvdXRwdXRzLmxvZ2l0cw==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, XmodForMultipleChoice
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;facebook/xmod-base&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = XmodForMultipleChoice.from_pretrained(<span class="hljs-string">&quot;facebook/xmod-base&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>prompt = <span class="hljs-string">&quot;In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>choice0 = <span class="hljs-string">&quot;It is eaten with a fork and a knife.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>choice1 = <span class="hljs-string">&quot;It is eaten while held in the hand.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.tensor(<span class="hljs-number">0</span>).unsqueeze(<span class="hljs-number">0</span>)  <span class="hljs-comment"># choice0 is correct (according to Wikipedia ;)), batch size 1</span>

<span class="hljs-meta">&gt;&gt;&gt; </span>encoding = tokenizer([prompt, prompt], [choice0, choice1], return_tensors=<span class="hljs-string">&quot;pt&quot;</span>, padding=<span class="hljs-literal">True</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**{k: v.unsqueeze(<span class="hljs-number">0</span>) <span class="hljs-keyword">for</span> k, v <span class="hljs-keyword">in</span> encoding.items()}, labels=labels)  <span class="hljs-comment"># batch size is 1</span>

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># the linear classifier still needs to be trained</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = outputs.loss
<span class="hljs-meta">&gt;&gt;&gt; </span>logits = outputs.logits`,wrap:!1}}),{c(){t=p("p"),t.textContent=y,d=a(),u(c.$$.fragment)},l(n){t=m(n,"P",{"data-svelte-h":!0}),h(t)!=="svelte-11lpom8"&&(t.textContent=y),d=r(n),f(c.$$.fragment,n)},m(n,w){l(n,t,w),l(n,d,w),g(c,n,w),T=!0},p:Z,i(n){T||(_(c.$$.fragment,n),T=!0)},o(n){b(c.$$.fragment,n),T=!1},d(n){n&&(s(t),s(d)),M(c,n)}}}function Ps(k){let t,y=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=p("p"),t.innerHTML=y},l(d){t=m(d,"P",{"data-svelte-h":!0}),h(t)!=="svelte-fincs2"&&(t.innerHTML=y)},m(d,c){l(d,t,c)},p:Z,d(d){d&&s(t)}}}function Os(k){let t,y="Example:",d,c,T;return c=new O({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBYbW9kRm9yVG9rZW5DbGFzc2lmaWNhdGlvbiUwQWltcG9ydCUyMHRvcmNoJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIyZmFjZWJvb2slMkZ4bW9kLWJhc2UlMjIpJTBBbW9kZWwlMjAlM0QlMjBYbW9kRm9yVG9rZW5DbGFzc2lmaWNhdGlvbi5mcm9tX3ByZXRyYWluZWQoJTIyZmFjZWJvb2slMkZ4bW9kLWJhc2UlMjIpJTBBJTBBaW5wdXRzJTIwJTNEJTIwdG9rZW5pemVyKCUwQSUyMCUyMCUyMCUyMCUyMkh1Z2dpbmdGYWNlJTIwaXMlMjBhJTIwY29tcGFueSUyMGJhc2VkJTIwaW4lMjBQYXJpcyUyMGFuZCUyME5ldyUyMFlvcmslMjIlMkMlMjBhZGRfc3BlY2lhbF90b2tlbnMlM0RGYWxzZSUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIlMEEpJTBBJTBBd2l0aCUyMHRvcmNoLm5vX2dyYWQoKSUzQSUwQSUyMCUyMCUyMCUyMGxvZ2l0cyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzKS5sb2dpdHMlMEElMEFwcmVkaWN0ZWRfdG9rZW5fY2xhc3NfaWRzJTIwJTNEJTIwbG9naXRzLmFyZ21heCgtMSklMEElMEElMjMlMjBOb3RlJTIwdGhhdCUyMHRva2VucyUyMGFyZSUyMGNsYXNzaWZpZWQlMjByYXRoZXIlMjB0aGVuJTIwaW5wdXQlMjB3b3JkcyUyMHdoaWNoJTIwbWVhbnMlMjB0aGF0JTBBJTIzJTIwdGhlcmUlMjBtaWdodCUyMGJlJTIwbW9yZSUyMHByZWRpY3RlZCUyMHRva2VuJTIwY2xhc3NlcyUyMHRoYW4lMjB3b3Jkcy4lMEElMjMlMjBNdWx0aXBsZSUyMHRva2VuJTIwY2xhc3NlcyUyMG1pZ2h0JTIwYWNjb3VudCUyMGZvciUyMHRoZSUyMHNhbWUlMjB3b3JkJTBBcHJlZGljdGVkX3Rva2Vuc19jbGFzc2VzJTIwJTNEJTIwJTVCbW9kZWwuY29uZmlnLmlkMmxhYmVsJTVCdC5pdGVtKCklNUQlMjBmb3IlMjB0JTIwaW4lMjBwcmVkaWN0ZWRfdG9rZW5fY2xhc3NfaWRzJTVCMCU1RCU1RCUwQXByZWRpY3RlZF90b2tlbnNfY2xhc3NlcyUwQSUwQWxhYmVscyUyMCUzRCUyMHByZWRpY3RlZF90b2tlbl9jbGFzc19pZHMlMEFsb3NzJTIwJTNEJTIwbW9kZWwoKippbnB1dHMlMkMlMjBsYWJlbHMlM0RsYWJlbHMpLmxvc3MlMEFyb3VuZChsb3NzLml0ZW0oKSUyQyUyMDIp",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, XmodForTokenClassification
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;facebook/xmod-base&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = XmodForTokenClassification.from_pretrained(<span class="hljs-string">&quot;facebook/xmod-base&quot;</span>)

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
...`,wrap:!1}}),{c(){t=p("p"),t.textContent=y,d=a(),u(c.$$.fragment)},l(n){t=m(n,"P",{"data-svelte-h":!0}),h(t)!=="svelte-11lpom8"&&(t.textContent=y),d=r(n),f(c.$$.fragment,n)},m(n,w){l(n,t,w),l(n,d,w),g(c,n,w),T=!0},p:Z,i(n){T||(_(c.$$.fragment,n),T=!0)},o(n){b(c.$$.fragment,n),T=!1},d(n){n&&(s(t),s(d)),M(c,n)}}}function Ds(k){let t,y=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=p("p"),t.innerHTML=y},l(d){t=m(d,"P",{"data-svelte-h":!0}),h(t)!=="svelte-fincs2"&&(t.innerHTML=y)},m(d,c){l(d,t,c)},p:Z,d(d){d&&s(t)}}}function Ks(k){let t,y="Example:",d,c,T;return c=new O({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBYbW9kRm9yUXVlc3Rpb25BbnN3ZXJpbmclMEFpbXBvcnQlMjB0b3JjaCUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMmZhY2Vib29rJTJGeG1vZC1iYXNlJTIyKSUwQW1vZGVsJTIwJTNEJTIwWG1vZEZvclF1ZXN0aW9uQW5zd2VyaW5nLmZyb21fcHJldHJhaW5lZCglMjJmYWNlYm9vayUyRnhtb2QtYmFzZSUyMiklMEElMEFxdWVzdGlvbiUyQyUyMHRleHQlMjAlM0QlMjAlMjJXaG8lMjB3YXMlMjBKaW0lMjBIZW5zb24lM0YlMjIlMkMlMjAlMjJKaW0lMjBIZW5zb24lMjB3YXMlMjBhJTIwbmljZSUyMHB1cHBldCUyMiUwQSUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplcihxdWVzdGlvbiUyQyUyMHRleHQlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQXdpdGglMjB0b3JjaC5ub19ncmFkKCklM0ElMEElMjAlMjAlMjAlMjBvdXRwdXRzJTIwJTNEJTIwbW9kZWwoKippbnB1dHMpJTBBJTBBYW5zd2VyX3N0YXJ0X2luZGV4JTIwJTNEJTIwb3V0cHV0cy5zdGFydF9sb2dpdHMuYXJnbWF4KCklMEFhbnN3ZXJfZW5kX2luZGV4JTIwJTNEJTIwb3V0cHV0cy5lbmRfbG9naXRzLmFyZ21heCgpJTBBJTBBcHJlZGljdF9hbnN3ZXJfdG9rZW5zJTIwJTNEJTIwaW5wdXRzLmlucHV0X2lkcyU1QjAlMkMlMjBhbnN3ZXJfc3RhcnRfaW5kZXglMjAlM0ElMjBhbnN3ZXJfZW5kX2luZGV4JTIwJTJCJTIwMSU1RCUwQXRva2VuaXplci5kZWNvZGUocHJlZGljdF9hbnN3ZXJfdG9rZW5zJTJDJTIwc2tpcF9zcGVjaWFsX3Rva2VucyUzRFRydWUpJTBBJTBBJTIzJTIwdGFyZ2V0JTIwaXMlMjAlMjJuaWNlJTIwcHVwcGV0JTIyJTBBdGFyZ2V0X3N0YXJ0X2luZGV4JTIwJTNEJTIwdG9yY2gudGVuc29yKCU1QjE0JTVEKSUwQXRhcmdldF9lbmRfaW5kZXglMjAlM0QlMjB0b3JjaC50ZW5zb3IoJTVCMTUlNUQpJTBBJTBBb3V0cHV0cyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzJTJDJTIwc3RhcnRfcG9zaXRpb25zJTNEdGFyZ2V0X3N0YXJ0X2luZGV4JTJDJTIwZW5kX3Bvc2l0aW9ucyUzRHRhcmdldF9lbmRfaW5kZXgpJTBBbG9zcyUyMCUzRCUyMG91dHB1dHMubG9zcyUwQXJvdW5kKGxvc3MuaXRlbSgpJTJDJTIwMik=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, XmodForQuestionAnswering
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;facebook/xmod-base&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = XmodForQuestionAnswering.from_pretrained(<span class="hljs-string">&quot;facebook/xmod-base&quot;</span>)

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
...`,wrap:!1}}),{c(){t=p("p"),t.textContent=y,d=a(),u(c.$$.fragment)},l(n){t=m(n,"P",{"data-svelte-h":!0}),h(t)!=="svelte-11lpom8"&&(t.textContent=y),d=r(n),f(c.$$.fragment,n)},m(n,w){l(n,t,w),l(n,d,w),g(c,n,w),T=!0},p:Z,i(n){T||(_(c.$$.fragment,n),T=!0)},o(n){b(c.$$.fragment,n),T=!1},d(n){n&&(s(t),s(d)),M(c,n)}}}function ea(k){let t,y,d,c,T,n="<em>This model was released on 2022-05-12 and added to Hugging Face Transformers on 2023-02-10.</em>",w,ve,Ot,re,Rn='<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/>',Dt,Je,Kt,$e,Hn=`The X-MOD model was proposed in <a href="https://huggingface.co/papers/2205.06266" rel="nofollow">Lifting the Curse of Multilinguality by Pre-training Modular Transformers</a> by Jonas Pfeiffer, Naman Goyal, Xi Lin, Xian Li, James Cross, Sebastian Riedel, and Mikel Artetxe.
X-MOD extends multilingual masked language models like <a href="xlm-roberta">XLM-R</a> to include language-specific modular components (<em>language adapters</em>) during pre-training. For fine-tuning, the language adapters in each transformer layer are frozen.`,eo,je,Vn="The abstract from the paper is the following:",to,Ce,En="<em>Multilingual pre-trained models are known to suffer from the curse of multilinguality, which causes per-language performance to drop as they cover more languages. We address this issue by introducing language-specific modules, which allows us to grow the total capacity of the model, while keeping the total number of trainable parameters per language constant. In contrast with prior work that learns language-specific components post-hoc, we pre-train the modules of our Cross-lingual Modular (X-MOD) models from the start. Our experiments on natural language inference, named entity recognition and question answering show that our approach not only mitigates the negative interference between languages, but also enables positive transfer, resulting in improved monolingual and cross-lingual performance. Furthermore, our approach enables adding languages post-hoc with no measurable drop in performance, no longer limiting the model usage to the set of pre-trained languages.</em>",oo,xe,An=`This model was contributed by <a href="https://huggingface.co/jvamvas" rel="nofollow">jvamvas</a>.
The original code can be found <a href="https://github.com/facebookresearch/fairseq/tree/58cc6cca18f15e6d56e3f60c959fe4f878960a60/fairseq/models/xmod" rel="nofollow">here</a> and the original documentation is found <a href="https://github.com/facebookresearch/fairseq/tree/58cc6cca18f15e6d56e3f60c959fe4f878960a60/examples/xmod" rel="nofollow">here</a>.`,no,Xe,so,Fe,Sn="Tips:",ao,ze,Qn='<li>X-MOD is similar to <a href="xlm-roberta">XLM-R</a>, but a difference is that the input language needs to be specified so that the correct language adapter can be activated.</li> <li>The main models – base and large – have adapters for 81 languages.</li>',ro,Ue,io,We,lo,Ie,Yn="There are two ways to specify the input language:",co,Ze,Pn="<li>By setting a default language before using the model:</li>",po,Le,mo,ie,On="<li>By explicitly passing the index of the language adapter for each sample:</li>",ho,qe,uo,Be,fo,Ge,Dn="The paper recommends that the embedding layer and the language adapters are frozen during fine-tuning. A method for doing this is provided:",go,Ne,_o,Re,bo,He,Kn="After fine-tuning, zero-shot cross-lingual transfer can be tested by activating the language adapter of the target language:",Mo,Ve,yo,Ee,To,Ae,es='<li><a href="../tasks/sequence_classification">Text classification task guide</a></li> <li><a href="../tasks/token_classification">Token classification task guide</a></li> <li><a href="../tasks/question_answering">Question answering task guide</a></li> <li><a href="../tasks/language_modeling">Causal language modeling task guide</a></li> <li><a href="../tasks/masked_language_modeling">Masked language modeling task guide</a></li> <li><a href="../tasks/multiple_choice">Multiple choice task guide</a></li>',wo,Se,ko,q,Qe,No,bt,ts=`This is the configuration class to store the configuration of a <a href="/docs/transformers/v4.56.2/en/model_doc/xmod#transformers.XmodModel">XmodModel</a>. It is used to instantiate an X-MOD
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the
<a href="https://huggingface.co/facebook/xmod-base" rel="nofollow">facebook/xmod-base</a> architecture.`,Ro,Mt,os=`Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> and can be used to control the model outputs. Read the
documentation from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> for more information.`,Ho,le,vo,Ye,Jo,v,Pe,Vo,yt,ns=`The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
cross-attention is added between the self-attention layers, following the architecture described in <em>Attention is
all you need</em>_ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz
Kaiser and Illia Polosukhin.`,Eo,Tt,ss=`To behave as an decoder the model needs to be initialized with the <code>is_decoder</code> argument of the configuration set
to <code>True</code>. To be used in a Seq2Seq model, the model needs to initialized with both <code>is_decoder</code> argument and
<code>add_cross_attention</code> set to <code>True</code>; an <code>encoder_hidden_states</code> is then expected as an input to the forward pass.`,Ao,wt,as='.. _<em>Attention is all you need</em>: <a href="https://huggingface.co/papers/1706.03762" rel="nofollow">https://huggingface.co/papers/1706.03762</a>',So,kt,rs=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Qo,vt,is=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Yo,se,Oe,Po,Jt,ls='The <a href="/docs/transformers/v4.56.2/en/model_doc/xmod#transformers.XmodModel">XmodModel</a> forward method, overrides the <code>__call__</code> special method.',Oo,de,$o,De,jo,x,Ke,Do,$t,ds="X-MOD Model with a <code>language modeling</code> head on top for CLM fine-tuning.",Ko,jt,cs=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,en,Ct,ps=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,tn,A,et,on,xt,ms='The <a href="/docs/transformers/v4.56.2/en/model_doc/xmod#transformers.XmodForCausalLM">XmodForCausalLM</a> forward method, overrides the <code>__call__</code> special method.',nn,ce,sn,pe,Co,tt,xo,X,ot,an,Xt,hs="The Xmod Model with a <code>language modeling</code> head on top.”",rn,Ft,us=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,ln,zt,fs=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,dn,S,nt,cn,Ut,gs='The <a href="/docs/transformers/v4.56.2/en/model_doc/xmod#transformers.XmodForMaskedLM">XmodForMaskedLM</a> forward method, overrides the <code>__call__</code> special method.',pn,me,mn,he,Xo,st,Fo,F,at,hn,Wt,_s=`X-MOD Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled
output) e.g. for GLUE tasks.`,un,It,bs=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,fn,Zt,Ms=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,gn,L,rt,_n,Lt,ys='The <a href="/docs/transformers/v4.56.2/en/model_doc/xmod#transformers.XmodForSequenceClassification">XmodForSequenceClassification</a> forward method, overrides the <code>__call__</code> special method.',bn,ue,Mn,fe,yn,ge,zo,it,Uo,z,lt,Tn,qt,Ts=`The Xmod Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
softmax) e.g. for RocStories/SWAG tasks.`,wn,Bt,ws=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,kn,Gt,ks=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,vn,Q,dt,Jn,Nt,vs='The <a href="/docs/transformers/v4.56.2/en/model_doc/xmod#transformers.XmodForMultipleChoice">XmodForMultipleChoice</a> forward method, overrides the <code>__call__</code> special method.',$n,_e,jn,be,Wo,ct,Io,U,pt,Cn,Rt,Js=`The Xmod transformer with a token classification head on top (a linear layer on top of the hidden-states
output) e.g. for Named-Entity-Recognition (NER) tasks.`,xn,Ht,$s=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Xn,Vt,js=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Fn,Y,mt,zn,Et,Cs='The <a href="/docs/transformers/v4.56.2/en/model_doc/xmod#transformers.XmodForTokenClassification">XmodForTokenClassification</a> forward method, overrides the <code>__call__</code> special method.',Un,Me,Wn,ye,Zo,ht,Lo,W,ut,In,At,xs=`The Xmod transformer with a span classification head on top for extractive question-answering tasks like
SQuAD (a linear layer on top of the hidden-states output to compute <code>span start logits</code> and <code>span end logits</code>).`,Zn,St,Xs=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Ln,Qt,Fs=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,qn,P,ft,Bn,Yt,zs='The <a href="/docs/transformers/v4.56.2/en/model_doc/xmod#transformers.XmodForQuestionAnswering">XmodForQuestionAnswering</a> forward method, overrides the <code>__call__</code> special method.',Gn,Te,Nn,we,qo,gt,Bo,Pt,Go;return ve=new C({props:{title:"X-MOD",local:"x-mod",headingTag:"h1"}}),Je=new C({props:{title:"Overview",local:"overview",headingTag:"h2"}}),Xe=new C({props:{title:"Usage tips",local:"usage-tips",headingTag:"h2"}}),Ue=new C({props:{title:"Adapter Usage",local:"adapter-usage",headingTag:"h2"}}),We=new C({props:{title:"Input language",local:"input-language",headingTag:"h3"}}),Le=new O({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMFhtb2RNb2RlbCUwQSUwQW1vZGVsJTIwJTNEJTIwWG1vZE1vZGVsLmZyb21fcHJldHJhaW5lZCglMjJmYWNlYm9vayUyRnhtb2QtYmFzZSUyMiklMEFtb2RlbC5zZXRfZGVmYXVsdF9sYW5ndWFnZSglMjJlbl9YWCUyMik=",highlighted:`<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> XmodModel

model = XmodModel.from_pretrained(<span class="hljs-string">&quot;facebook/xmod-base&quot;</span>)
model.set_default_language(<span class="hljs-string">&quot;en_XX&quot;</span>)`,wrap:!1}}),qe=new O({props:{code:"aW1wb3J0JTIwdG9yY2glMEElMEFpbnB1dF9pZHMlMjAlM0QlMjB0b3JjaC50ZW5zb3IoJTBBJTIwJTIwJTIwJTIwJTVCJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTVCMCUyQyUyMDU4MSUyQyUyMDEwMjY5JTJDJTIwODMlMkMlMjA5OTk0MiUyQyUyMDEzNiUyQyUyMDYwNzQyJTJDJTIwMjMlMkMlMjA3MCUyQyUyMDgwNTgzJTJDJTIwMTgyNzYlMkMlMjAyJTVEJTJDJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTVCMCUyQyUyMDEzMTAlMkMlMjA0OTA4MyUyQyUyMDQ0MyUyQyUyMDI2OSUyQyUyMDcxJTJDJTIwNTQ4NiUyQyUyMDE2NSUyQyUyMDYwNDI5JTJDJTIwNjYwJTJDJTIwMjMlMkMlMjAyJTVEJTJDJTBBJTIwJTIwJTIwJTIwJTVEJTBBKSUwQWxhbmdfaWRzJTIwJTNEJTIwdG9yY2guTG9uZ1RlbnNvciglMEElMjAlMjAlMjAlMjAlNUIlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAwJTJDJTIwJTIwJTIzJTIwZW5fWFglMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjA4JTJDJTIwJTIwJTIzJTIwZGVfREUlMEElMjAlMjAlMjAlMjAlNUQlMEEpJTBBb3V0cHV0JTIwJTNEJTIwbW9kZWwoaW5wdXRfaWRzJTJDJTIwbGFuZ19pZHMlM0RsYW5nX2lkcyk=",highlighted:`<span class="hljs-keyword">import</span> torch

input_ids = torch.tensor(
    [
        [<span class="hljs-number">0</span>, <span class="hljs-number">581</span>, <span class="hljs-number">10269</span>, <span class="hljs-number">83</span>, <span class="hljs-number">99942</span>, <span class="hljs-number">136</span>, <span class="hljs-number">60742</span>, <span class="hljs-number">23</span>, <span class="hljs-number">70</span>, <span class="hljs-number">80583</span>, <span class="hljs-number">18276</span>, <span class="hljs-number">2</span>],
        [<span class="hljs-number">0</span>, <span class="hljs-number">1310</span>, <span class="hljs-number">49083</span>, <span class="hljs-number">443</span>, <span class="hljs-number">269</span>, <span class="hljs-number">71</span>, <span class="hljs-number">5486</span>, <span class="hljs-number">165</span>, <span class="hljs-number">60429</span>, <span class="hljs-number">660</span>, <span class="hljs-number">23</span>, <span class="hljs-number">2</span>],
    ]
)
lang_ids = torch.LongTensor(
    [
        <span class="hljs-number">0</span>,  <span class="hljs-comment"># en_XX</span>
        <span class="hljs-number">8</span>,  <span class="hljs-comment"># de_DE</span>
    ]
)
output = model(input_ids, lang_ids=lang_ids)`,wrap:!1}}),Be=new C({props:{title:"Fine-tuning",local:"fine-tuning",headingTag:"h3"}}),Ne=new O({props:{code:"bW9kZWwuZnJlZXplX2VtYmVkZGluZ3NfYW5kX2xhbmd1YWdlX2FkYXB0ZXJzKCklMEElMjMlMjBGaW5lLXR1bmUlMjB0aGUlMjBtb2RlbCUyMC4uLg==",highlighted:`model.freeze_embeddings_and_language_adapters()
<span class="hljs-comment"># Fine-tune the model ...</span>`,wrap:!1}}),Re=new C({props:{title:"Cross-lingual transfer",local:"cross-lingual-transfer",headingTag:"h3"}}),Ve=new O({props:{code:"bW9kZWwuc2V0X2RlZmF1bHRfbGFuZ3VhZ2UoJTIyZGVfREUlMjIpJTBBJTIzJTIwRXZhbHVhdGUlMjB0aGUlMjBtb2RlbCUyMG9uJTIwR2VybWFuJTIwZXhhbXBsZXMlMjAuLi4=",highlighted:`model.set_default_language(<span class="hljs-string">&quot;de_DE&quot;</span>)
<span class="hljs-comment"># Evaluate the model on German examples ...</span>`,wrap:!1}}),Ee=new C({props:{title:"Resources",local:"resources",headingTag:"h2"}}),Se=new C({props:{title:"XmodConfig",local:"transformers.XmodConfig",headingTag:"h2"}}),Qe=new I({props:{name:"class transformers.XmodConfig",anchor:"transformers.XmodConfig",parameters:[{name:"vocab_size",val:" = 30522"},{name:"hidden_size",val:" = 768"},{name:"num_hidden_layers",val:" = 12"},{name:"num_attention_heads",val:" = 12"},{name:"intermediate_size",val:" = 3072"},{name:"hidden_act",val:" = 'gelu'"},{name:"hidden_dropout_prob",val:" = 0.1"},{name:"attention_probs_dropout_prob",val:" = 0.1"},{name:"max_position_embeddings",val:" = 512"},{name:"type_vocab_size",val:" = 2"},{name:"initializer_range",val:" = 0.02"},{name:"layer_norm_eps",val:" = 1e-12"},{name:"pad_token_id",val:" = 1"},{name:"bos_token_id",val:" = 0"},{name:"eos_token_id",val:" = 2"},{name:"position_embedding_type",val:" = 'absolute'"},{name:"use_cache",val:" = True"},{name:"classifier_dropout",val:" = None"},{name:"pre_norm",val:" = False"},{name:"adapter_reduction_factor",val:" = 2"},{name:"adapter_layer_norm",val:" = False"},{name:"adapter_reuse_layer_norm",val:" = True"},{name:"ln_before_adapter",val:" = True"},{name:"languages",val:" = ('en_XX',)"},{name:"default_language",val:" = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.XmodConfig.vocab_size",description:`<strong>vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 30522) &#x2014;
Vocabulary size of the X-MOD model. Defines the number of different tokens that can be represented by the
<code>inputs_ids</code> passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/xmod#transformers.XmodModel">XmodModel</a>.`,name:"vocab_size"},{anchor:"transformers.XmodConfig.hidden_size",description:`<strong>hidden_size</strong> (<code>int</code>, <em>optional</em>, defaults to 768) &#x2014;
Dimensionality of the encoder layers and the pooler layer.`,name:"hidden_size"},{anchor:"transformers.XmodConfig.num_hidden_layers",description:`<strong>num_hidden_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 12) &#x2014;
Number of hidden layers in the Transformer encoder.`,name:"num_hidden_layers"},{anchor:"transformers.XmodConfig.num_attention_heads",description:`<strong>num_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 12) &#x2014;
Number of attention heads for each attention layer in the Transformer encoder.`,name:"num_attention_heads"},{anchor:"transformers.XmodConfig.intermediate_size",description:`<strong>intermediate_size</strong> (<code>int</code>, <em>optional</em>, defaults to 3072) &#x2014;
Dimensionality of the &#x201C;intermediate&#x201D; (often named feed-forward) layer in the Transformer encoder.`,name:"intermediate_size"},{anchor:"transformers.XmodConfig.hidden_act",description:`<strong>hidden_act</strong> (<code>str</code> or <code>Callable</code>, <em>optional</em>, defaults to <code>&quot;gelu&quot;</code>) &#x2014;
The non-linear activation function (function or string) in the encoder and pooler. If string, <code>&quot;gelu&quot;</code>,
<code>&quot;relu&quot;</code>, <code>&quot;silu&quot;</code> and <code>&quot;gelu_new&quot;</code> are supported.`,name:"hidden_act"},{anchor:"transformers.XmodConfig.hidden_dropout_prob",description:`<strong>hidden_dropout_prob</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.`,name:"hidden_dropout_prob"},{anchor:"transformers.XmodConfig.attention_probs_dropout_prob",description:`<strong>attention_probs_dropout_prob</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout ratio for the attention probabilities.`,name:"attention_probs_dropout_prob"},{anchor:"transformers.XmodConfig.max_position_embeddings",description:`<strong>max_position_embeddings</strong> (<code>int</code>, <em>optional</em>, defaults to 512) &#x2014;
The maximum sequence length that this model might ever be used with. Typically set this to something large
just in case (e.g., 512 or 1024 or 2048).`,name:"max_position_embeddings"},{anchor:"transformers.XmodConfig.type_vocab_size",description:`<strong>type_vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 2) &#x2014;
The vocabulary size of the <code>token_type_ids</code> passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/xmod#transformers.XmodModel">XmodModel</a>.`,name:"type_vocab_size"},{anchor:"transformers.XmodConfig.initializer_range",description:`<strong>initializer_range</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.`,name:"initializer_range"},{anchor:"transformers.XmodConfig.layer_norm_eps",description:`<strong>layer_norm_eps</strong> (<code>float</code>, <em>optional</em>, defaults to 1e-12) &#x2014;
The epsilon used by the layer normalization layers.`,name:"layer_norm_eps"},{anchor:"transformers.XmodConfig.position_embedding_type",description:`<strong>position_embedding_type</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;absolute&quot;</code>) &#x2014;
Type of position embedding. Choose one of <code>&quot;absolute&quot;</code>, <code>&quot;relative_key&quot;</code>, <code>&quot;relative_key_query&quot;</code>. For
positional embeddings use <code>&quot;absolute&quot;</code>. For more information on <code>&quot;relative_key&quot;</code>, please refer to
<a href="https://huggingface.co/papers/1803.02155" rel="nofollow">Self-Attention with Relative Position Representations (Shaw et al.)</a>.
For more information on <code>&quot;relative_key_query&quot;</code>, please refer to <em>Method 4</em> in <a href="https://huggingface.co/papers/2009.13658" rel="nofollow">Improve Transformer Models
with Better Relative Position Embeddings (Huang et al.)</a>.`,name:"position_embedding_type"},{anchor:"transformers.XmodConfig.is_decoder",description:`<strong>is_decoder</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether the model is used as a decoder or not. If <code>False</code>, the model is used as an encoder.`,name:"is_decoder"},{anchor:"transformers.XmodConfig.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not the model should return the last key/values attentions (not used by all models). Only
relevant if <code>config.is_decoder=True</code>.`,name:"use_cache"},{anchor:"transformers.XmodConfig.classifier_dropout",description:`<strong>classifier_dropout</strong> (<code>float</code>, <em>optional</em>) &#x2014;
The dropout ratio for the classification head.`,name:"classifier_dropout"},{anchor:"transformers.XmodConfig.pre_norm",description:`<strong>pre_norm</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether to apply layer normalization before each block.`,name:"pre_norm"},{anchor:"transformers.XmodConfig.adapter_reduction_factor",description:`<strong>adapter_reduction_factor</strong> (<code>int</code> or <code>float</code>, <em>optional</em>, defaults to 2) &#x2014;
The factor by which the dimensionality of the adapter is reduced relative to <code>hidden_size</code>.`,name:"adapter_reduction_factor"},{anchor:"transformers.XmodConfig.adapter_layer_norm",description:`<strong>adapter_layer_norm</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether to apply a new layer normalization before the adapter modules (shared across all adapters).`,name:"adapter_layer_norm"},{anchor:"transformers.XmodConfig.adapter_reuse_layer_norm",description:`<strong>adapter_reuse_layer_norm</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether to reuse the second layer normalization and apply it before the adapter modules as well.`,name:"adapter_reuse_layer_norm"},{anchor:"transformers.XmodConfig.ln_before_adapter",description:`<strong>ln_before_adapter</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether to apply the layer normalization before the residual connection around the adapter module.`,name:"ln_before_adapter"},{anchor:"transformers.XmodConfig.languages",description:`<strong>languages</strong> (<code>Iterable[str]</code>, <em>optional</em>, defaults to <code>[&quot;en_XX&quot;]</code>) &#x2014;
An iterable of language codes for which adapter modules should be initialized.`,name:"languages"},{anchor:"transformers.XmodConfig.default_language",description:`<strong>default_language</strong> (<code>str</code>, <em>optional</em>) &#x2014;
Language code of a default language. It will be assumed that the input is in this language if no language
codes are explicitly passed to the forward method.`,name:"default_language"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xmod/configuration_xmod.py#L29"}}),le=new ke({props:{anchor:"transformers.XmodConfig.example",$$slots:{default:[Bs]},$$scope:{ctx:k}}}),Ye=new C({props:{title:"XmodModel",local:"transformers.XmodModel",headingTag:"h2"}}),Pe=new I({props:{name:"class transformers.XmodModel",anchor:"transformers.XmodModel",parameters:[{name:"config",val:""},{name:"add_pooling_layer",val:" = True"}],parametersDescription:[{anchor:"transformers.XmodModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/xmod#transformers.XmodModel">XmodModel</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"},{anchor:"transformers.XmodModel.add_pooling_layer",description:`<strong>add_pooling_layer</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether to add a pooling layer`,name:"add_pooling_layer"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xmod/modeling_xmod.py#L687"}}),Oe=new I({props:{name:"forward",anchor:"transformers.XmodModel.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"lang_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_hidden_states",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"past_key_values",val:": typing.Optional[list[torch.FloatTensor]] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.Tensor] = None"}],parametersDescription:[{anchor:"transformers.XmodModel.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.XmodModel.forward.lang_ids",description:`<strong>lang_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of the language adapters that should be activated for each sample, respectively. Default: the index
that corresponds to <code>self.config.default_language</code>.`,name:"lang_ids"},{anchor:"transformers.XmodModel.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.XmodModel.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.XmodModel.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.XmodModel.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.XmodModel.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.XmodModel.forward.encoder_hidden_states",description:`<strong>encoder_hidden_states</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
if the model is configured as a decoder.`,name:"encoder_hidden_states"},{anchor:"transformers.XmodModel.forward.encoder_attention_mask",description:`<strong>encoder_attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
the cross-attention if the model is configured as a decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>`,name:"encoder_attention_mask"},{anchor:"transformers.XmodModel.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>list[torch.FloatTensor]</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.XmodModel.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.XmodModel.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.XmodModel.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.XmodModel.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.XmodModel.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.Tensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xmod/modeling_xmod.py#L722",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions"
>transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/xmod#transformers.XmodConfig"
>XmodConfig</a>) and inputs.</p>
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
`}}),de=new _t({props:{$$slots:{default:[Gs]},$$scope:{ctx:k}}}),De=new C({props:{title:"XmodForCausalLM",local:"transformers.XmodForCausalLM",headingTag:"h2"}}),Ke=new I({props:{name:"class transformers.XmodForCausalLM",anchor:"transformers.XmodForCausalLM",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.XmodForCausalLM.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/xmod#transformers.XmodForCausalLM">XmodForCausalLM</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xmod/modeling_xmod.py#L860"}}),et=new I({props:{name:"forward",anchor:"transformers.XmodForCausalLM.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"lang_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"encoder_hidden_states",val:": typing.Optional[torch.FloatTensor] = None"},{name:"encoder_attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[tuple[tuple[torch.FloatTensor]]] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.Tensor] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.XmodForCausalLM.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.XmodForCausalLM.forward.lang_ids",description:`<strong>lang_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of the language adapters that should be activated for each sample, respectively. Default: the index
that corresponds to <code>self.config.default_language</code>.`,name:"lang_ids"},{anchor:"transformers.XmodForCausalLM.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.XmodForCausalLM.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.XmodForCausalLM.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.XmodForCausalLM.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.XmodForCausalLM.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.XmodForCausalLM.forward.encoder_hidden_states",description:`<strong>encoder_hidden_states</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
if the model is configured as a decoder.`,name:"encoder_hidden_states"},{anchor:"transformers.XmodForCausalLM.forward.encoder_attention_mask",description:`<strong>encoder_attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
the cross-attention if the model is configured as a decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>`,name:"encoder_attention_mask"},{anchor:"transformers.XmodForCausalLM.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the left-to-right language modeling loss (next word prediction). Indices should be in
<code>[-100, 0, ..., config.vocab_size]</code> (see <code>input_ids</code> docstring) Tokens with indices set to <code>-100</code> are
ignored (masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>`,name:"labels"},{anchor:"transformers.XmodForCausalLM.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>tuple[tuple[torch.FloatTensor]]</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.XmodForCausalLM.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.XmodForCausalLM.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.XmodForCausalLM.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.XmodForCausalLM.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.XmodForCausalLM.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.Tensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xmod/modeling_xmod.py#L884",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithCrossAttentions"
>transformers.modeling_outputs.CausalLMOutputWithCrossAttentions</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/xmod#transformers.XmodConfig"
>XmodConfig</a>) and inputs.</p>
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
`}}),ce=new _t({props:{$$slots:{default:[Ns]},$$scope:{ctx:k}}}),pe=new ke({props:{anchor:"transformers.XmodForCausalLM.forward.example",$$slots:{default:[Rs]},$$scope:{ctx:k}}}),tt=new C({props:{title:"XmodForMaskedLM",local:"transformers.XmodForMaskedLM",headingTag:"h2"}}),ot=new I({props:{name:"class transformers.XmodForMaskedLM",anchor:"transformers.XmodForMaskedLM",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.XmodForMaskedLM.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/xmod#transformers.XmodForMaskedLM">XmodForMaskedLM</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xmod/modeling_xmod.py#L980"}}),nt=new I({props:{name:"forward",anchor:"transformers.XmodForMaskedLM.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"lang_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"encoder_hidden_states",val:": typing.Optional[torch.FloatTensor] = None"},{name:"encoder_attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.XmodForMaskedLM.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.XmodForMaskedLM.forward.lang_ids",description:`<strong>lang_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of the language adapters that should be activated for each sample, respectively. Default: the index
that corresponds to <code>self.config.default_language</code>.`,name:"lang_ids"},{anchor:"transformers.XmodForMaskedLM.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.XmodForMaskedLM.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.XmodForMaskedLM.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.XmodForMaskedLM.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.XmodForMaskedLM.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.XmodForMaskedLM.forward.encoder_hidden_states",description:`<strong>encoder_hidden_states</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
if the model is configured as a decoder.`,name:"encoder_hidden_states"},{anchor:"transformers.XmodForMaskedLM.forward.encoder_attention_mask",description:`<strong>encoder_attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
the cross-attention if the model is configured as a decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>`,name:"encoder_attention_mask"},{anchor:"transformers.XmodForMaskedLM.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should be in <code>[-100, 0, ..., config.vocab_size]</code> (see <code>input_ids</code> docstring) Tokens with indices set to <code>-100</code> are ignored (masked), the
loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>`,name:"labels"},{anchor:"transformers.XmodForMaskedLM.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.XmodForMaskedLM.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.XmodForMaskedLM.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xmod/modeling_xmod.py#L1007",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.MaskedLMOutput"
>transformers.modeling_outputs.MaskedLMOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/xmod#transformers.XmodConfig"
>XmodConfig</a>) and inputs.</p>
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
`}}),me=new _t({props:{$$slots:{default:[Hs]},$$scope:{ctx:k}}}),he=new ke({props:{anchor:"transformers.XmodForMaskedLM.forward.example",$$slots:{default:[Vs]},$$scope:{ctx:k}}}),st=new C({props:{title:"XmodForSequenceClassification",local:"transformers.XmodForSequenceClassification",headingTag:"h2"}}),at=new I({props:{name:"class transformers.XmodForSequenceClassification",anchor:"transformers.XmodForSequenceClassification",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.XmodForSequenceClassification.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/xmod#transformers.XmodForSequenceClassification">XmodForSequenceClassification</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xmod/modeling_xmod.py#L1107"}}),rt=new I({props:{name:"forward",anchor:"transformers.XmodForSequenceClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"lang_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.XmodForSequenceClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.XmodForSequenceClassification.forward.lang_ids",description:`<strong>lang_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of the language adapters that should be activated for each sample, respectively. Default: the index
that corresponds to <code>self.config.default_language</code>.`,name:"lang_ids"},{anchor:"transformers.XmodForSequenceClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.XmodForSequenceClassification.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.XmodForSequenceClassification.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.XmodForSequenceClassification.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.XmodForSequenceClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.XmodForSequenceClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the sequence classification/regression loss. Indices should be in <code>[0, ..., config.num_labels - 1]</code>. If <code>config.num_labels == 1</code> a regression loss is computed (Mean-Square loss), If
<code>config.num_labels &gt; 1</code> a classification loss is computed (Cross-Entropy).`,name:"labels"},{anchor:"transformers.XmodForSequenceClassification.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.XmodForSequenceClassification.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.XmodForSequenceClassification.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xmod/modeling_xmod.py#L1120",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput"
>transformers.modeling_outputs.SequenceClassifierOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/xmod#transformers.XmodConfig"
>XmodConfig</a>) and inputs.</p>
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
`}}),ue=new _t({props:{$$slots:{default:[Es]},$$scope:{ctx:k}}}),fe=new ke({props:{anchor:"transformers.XmodForSequenceClassification.forward.example",$$slots:{default:[As]},$$scope:{ctx:k}}}),ge=new ke({props:{anchor:"transformers.XmodForSequenceClassification.forward.example-2",$$slots:{default:[Ss]},$$scope:{ctx:k}}}),it=new C({props:{title:"XmodForMultipleChoice",local:"transformers.XmodForMultipleChoice",headingTag:"h2"}}),lt=new I({props:{name:"class transformers.XmodForMultipleChoice",anchor:"transformers.XmodForMultipleChoice",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.XmodForMultipleChoice.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/xmod#transformers.XmodForMultipleChoice">XmodForMultipleChoice</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xmod/modeling_xmod.py#L1197"}}),dt=new I({props:{name:"forward",anchor:"transformers.XmodForMultipleChoice.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"lang_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.XmodForMultipleChoice.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, num_choices, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.XmodForMultipleChoice.forward.lang_ids",description:`<strong>lang_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, num_choices, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of the language adapters that should be activated for each sample, respectively. Default: the index
that corresponds to <code>self.config.default_language</code>.`,name:"lang_ids"},{anchor:"transformers.XmodForMultipleChoice.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, num_choices, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.XmodForMultipleChoice.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.XmodForMultipleChoice.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the multiple choice classification loss. Indices should be in <code>[0, ..., num_choices-1]</code> where <code>num_choices</code> is the size of the second dimension of the input tensors. (See
<code>input_ids</code> above)`,name:"labels"},{anchor:"transformers.XmodForMultipleChoice.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, num_choices, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.max_position_embeddings - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.XmodForMultipleChoice.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.XmodForMultipleChoice.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_choices, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.XmodForMultipleChoice.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.XmodForMultipleChoice.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.XmodForMultipleChoice.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xmod/modeling_xmod.py#L1209",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.MultipleChoiceModelOutput"
>transformers.modeling_outputs.MultipleChoiceModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/xmod#transformers.XmodConfig"
>XmodConfig</a>) and inputs.</p>
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
`}}),_e=new _t({props:{$$slots:{default:[Qs]},$$scope:{ctx:k}}}),be=new ke({props:{anchor:"transformers.XmodForMultipleChoice.forward.example",$$slots:{default:[Ys]},$$scope:{ctx:k}}}),ct=new C({props:{title:"XmodForTokenClassification",local:"transformers.XmodForTokenClassification",headingTag:"h2"}}),pt=new I({props:{name:"class transformers.XmodForTokenClassification",anchor:"transformers.XmodForTokenClassification",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.XmodForTokenClassification.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/xmod#transformers.XmodForTokenClassification">XmodForTokenClassification</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xmod/modeling_xmod.py#L1307"}}),mt=new I({props:{name:"forward",anchor:"transformers.XmodForTokenClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"lang_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.XmodForTokenClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.XmodForTokenClassification.forward.lang_ids",description:`<strong>lang_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of the language adapters that should be activated for each sample, respectively. Default: the index
that corresponds to <code>self.config.default_language</code>.`,name:"lang_ids"},{anchor:"transformers.XmodForTokenClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.XmodForTokenClassification.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.XmodForTokenClassification.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.XmodForTokenClassification.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.XmodForTokenClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.XmodForTokenClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the token classification loss. Indices should be in <code>[0, ..., config.num_labels - 1]</code>.`,name:"labels"},{anchor:"transformers.XmodForTokenClassification.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.XmodForTokenClassification.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.XmodForTokenClassification.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xmod/modeling_xmod.py#L1323",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput"
>transformers.modeling_outputs.TokenClassifierOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/xmod#transformers.XmodConfig"
>XmodConfig</a>) and inputs.</p>
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
`}}),Me=new _t({props:{$$slots:{default:[Ps]},$$scope:{ctx:k}}}),ye=new ke({props:{anchor:"transformers.XmodForTokenClassification.forward.example",$$slots:{default:[Os]},$$scope:{ctx:k}}}),ht=new C({props:{title:"XmodForQuestionAnswering",local:"transformers.XmodForQuestionAnswering",headingTag:"h2"}}),ut=new I({props:{name:"class transformers.XmodForQuestionAnswering",anchor:"transformers.XmodForQuestionAnswering",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.XmodForQuestionAnswering.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/xmod#transformers.XmodForQuestionAnswering">XmodForQuestionAnswering</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xmod/modeling_xmod.py#L1406"}}),ft=new I({props:{name:"forward",anchor:"transformers.XmodForQuestionAnswering.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"lang_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"start_positions",val:": typing.Optional[torch.LongTensor] = None"},{name:"end_positions",val:": typing.Optional[torch.LongTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.XmodForQuestionAnswering.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.XmodForQuestionAnswering.forward.lang_ids",description:`<strong>lang_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of the language adapters that should be activated for each sample, respectively. Default: the index
that corresponds to <code>self.config.default_language</code>.`,name:"lang_ids"},{anchor:"transformers.XmodForQuestionAnswering.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.XmodForQuestionAnswering.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.XmodForQuestionAnswering.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.XmodForQuestionAnswering.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.XmodForQuestionAnswering.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.XmodForQuestionAnswering.forward.start_positions",description:`<strong>start_positions</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for position (index) of the start of the labelled span for computing the token classification loss.
Positions are clamped to the length of the sequence (<code>sequence_length</code>). Position outside of the sequence
are not taken into account for computing the loss.`,name:"start_positions"},{anchor:"transformers.XmodForQuestionAnswering.forward.end_positions",description:`<strong>end_positions</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for position (index) of the end of the labelled span for computing the token classification loss.
Positions are clamped to the length of the sequence (<code>sequence_length</code>). Position outside of the sequence
are not taken into account for computing the loss.`,name:"end_positions"},{anchor:"transformers.XmodForQuestionAnswering.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.XmodForQuestionAnswering.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.XmodForQuestionAnswering.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xmod/modeling_xmod.py#L1418",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.QuestionAnsweringModelOutput"
>transformers.modeling_outputs.QuestionAnsweringModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/xmod#transformers.XmodConfig"
>XmodConfig</a>) and inputs.</p>
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
`}}),Te=new _t({props:{$$slots:{default:[Ds]},$$scope:{ctx:k}}}),we=new ke({props:{anchor:"transformers.XmodForQuestionAnswering.forward.example",$$slots:{default:[Ks]},$$scope:{ctx:k}}}),gt=new qs({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/xmod.md"}}),{c(){t=p("meta"),y=a(),d=p("p"),c=a(),T=p("p"),T.innerHTML=n,w=a(),u(ve.$$.fragment),Ot=a(),re=p("div"),re.innerHTML=Rn,Dt=a(),u(Je.$$.fragment),Kt=a(),$e=p("p"),$e.innerHTML=Hn,eo=a(),je=p("p"),je.textContent=Vn,to=a(),Ce=p("p"),Ce.innerHTML=En,oo=a(),xe=p("p"),xe.innerHTML=An,no=a(),u(Xe.$$.fragment),so=a(),Fe=p("p"),Fe.textContent=Sn,ao=a(),ze=p("ul"),ze.innerHTML=Qn,ro=a(),u(Ue.$$.fragment),io=a(),u(We.$$.fragment),lo=a(),Ie=p("p"),Ie.textContent=Yn,co=a(),Ze=p("ol"),Ze.innerHTML=Pn,po=a(),u(Le.$$.fragment),mo=a(),ie=p("ol"),ie.innerHTML=On,ho=a(),u(qe.$$.fragment),uo=a(),u(Be.$$.fragment),fo=a(),Ge=p("p"),Ge.textContent=Dn,go=a(),u(Ne.$$.fragment),_o=a(),u(Re.$$.fragment),bo=a(),He=p("p"),He.textContent=Kn,Mo=a(),u(Ve.$$.fragment),yo=a(),u(Ee.$$.fragment),To=a(),Ae=p("ul"),Ae.innerHTML=es,wo=a(),u(Se.$$.fragment),ko=a(),q=p("div"),u(Qe.$$.fragment),No=a(),bt=p("p"),bt.innerHTML=ts,Ro=a(),Mt=p("p"),Mt.innerHTML=os,Ho=a(),u(le.$$.fragment),vo=a(),u(Ye.$$.fragment),Jo=a(),v=p("div"),u(Pe.$$.fragment),Vo=a(),yt=p("p"),yt.innerHTML=ns,Eo=a(),Tt=p("p"),Tt.innerHTML=ss,Ao=a(),wt=p("p"),wt.innerHTML=as,So=a(),kt=p("p"),kt.innerHTML=rs,Qo=a(),vt=p("p"),vt.innerHTML=is,Yo=a(),se=p("div"),u(Oe.$$.fragment),Po=a(),Jt=p("p"),Jt.innerHTML=ls,Oo=a(),u(de.$$.fragment),$o=a(),u(De.$$.fragment),jo=a(),x=p("div"),u(Ke.$$.fragment),Do=a(),$t=p("p"),$t.innerHTML=ds,Ko=a(),jt=p("p"),jt.innerHTML=cs,en=a(),Ct=p("p"),Ct.innerHTML=ps,tn=a(),A=p("div"),u(et.$$.fragment),on=a(),xt=p("p"),xt.innerHTML=ms,nn=a(),u(ce.$$.fragment),sn=a(),u(pe.$$.fragment),Co=a(),u(tt.$$.fragment),xo=a(),X=p("div"),u(ot.$$.fragment),an=a(),Xt=p("p"),Xt.innerHTML=hs,rn=a(),Ft=p("p"),Ft.innerHTML=us,ln=a(),zt=p("p"),zt.innerHTML=fs,dn=a(),S=p("div"),u(nt.$$.fragment),cn=a(),Ut=p("p"),Ut.innerHTML=gs,pn=a(),u(me.$$.fragment),mn=a(),u(he.$$.fragment),Xo=a(),u(st.$$.fragment),Fo=a(),F=p("div"),u(at.$$.fragment),hn=a(),Wt=p("p"),Wt.textContent=_s,un=a(),It=p("p"),It.innerHTML=bs,fn=a(),Zt=p("p"),Zt.innerHTML=Ms,gn=a(),L=p("div"),u(rt.$$.fragment),_n=a(),Lt=p("p"),Lt.innerHTML=ys,bn=a(),u(ue.$$.fragment),Mn=a(),u(fe.$$.fragment),yn=a(),u(ge.$$.fragment),zo=a(),u(it.$$.fragment),Uo=a(),z=p("div"),u(lt.$$.fragment),Tn=a(),qt=p("p"),qt.textContent=Ts,wn=a(),Bt=p("p"),Bt.innerHTML=ws,kn=a(),Gt=p("p"),Gt.innerHTML=ks,vn=a(),Q=p("div"),u(dt.$$.fragment),Jn=a(),Nt=p("p"),Nt.innerHTML=vs,$n=a(),u(_e.$$.fragment),jn=a(),u(be.$$.fragment),Wo=a(),u(ct.$$.fragment),Io=a(),U=p("div"),u(pt.$$.fragment),Cn=a(),Rt=p("p"),Rt.textContent=Js,xn=a(),Ht=p("p"),Ht.innerHTML=$s,Xn=a(),Vt=p("p"),Vt.innerHTML=js,Fn=a(),Y=p("div"),u(mt.$$.fragment),zn=a(),Et=p("p"),Et.innerHTML=Cs,Un=a(),u(Me.$$.fragment),Wn=a(),u(ye.$$.fragment),Zo=a(),u(ht.$$.fragment),Lo=a(),W=p("div"),u(ut.$$.fragment),In=a(),At=p("p"),At.innerHTML=xs,Zn=a(),St=p("p"),St.innerHTML=Xs,Ln=a(),Qt=p("p"),Qt.innerHTML=Fs,qn=a(),P=p("div"),u(ft.$$.fragment),Bn=a(),Yt=p("p"),Yt.innerHTML=zs,Gn=a(),u(Te.$$.fragment),Nn=a(),u(we.$$.fragment),qo=a(),u(gt.$$.fragment),Bo=a(),Pt=p("p"),this.h()},l(e){const o=Ls("svelte-u9bgzb",document.head);t=m(o,"META",{name:!0,content:!0}),o.forEach(s),y=r(e),d=m(e,"P",{}),j(d).forEach(s),c=r(e),T=m(e,"P",{"data-svelte-h":!0}),h(T)!=="svelte-11b9c2w"&&(T.innerHTML=n),w=r(e),f(ve.$$.fragment,e),Ot=r(e),re=m(e,"DIV",{class:!0,"data-svelte-h":!0}),h(re)!=="svelte-13t8s2t"&&(re.innerHTML=Rn),Dt=r(e),f(Je.$$.fragment,e),Kt=r(e),$e=m(e,"P",{"data-svelte-h":!0}),h($e)!=="svelte-1e5g9xq"&&($e.innerHTML=Hn),eo=r(e),je=m(e,"P",{"data-svelte-h":!0}),h(je)!=="svelte-vfdo9a"&&(je.textContent=Vn),to=r(e),Ce=m(e,"P",{"data-svelte-h":!0}),h(Ce)!=="svelte-1ce41kp"&&(Ce.innerHTML=En),oo=r(e),xe=m(e,"P",{"data-svelte-h":!0}),h(xe)!=="svelte-17wcokr"&&(xe.innerHTML=An),no=r(e),f(Xe.$$.fragment,e),so=r(e),Fe=m(e,"P",{"data-svelte-h":!0}),h(Fe)!=="svelte-axv494"&&(Fe.textContent=Sn),ao=r(e),ze=m(e,"UL",{"data-svelte-h":!0}),h(ze)!=="svelte-gqhdug"&&(ze.innerHTML=Qn),ro=r(e),f(Ue.$$.fragment,e),io=r(e),f(We.$$.fragment,e),lo=r(e),Ie=m(e,"P",{"data-svelte-h":!0}),h(Ie)!=="svelte-1bi3hxf"&&(Ie.textContent=Yn),co=r(e),Ze=m(e,"OL",{"data-svelte-h":!0}),h(Ze)!=="svelte-1nun6oj"&&(Ze.innerHTML=Pn),po=r(e),f(Le.$$.fragment,e),mo=r(e),ie=m(e,"OL",{start:!0,"data-svelte-h":!0}),h(ie)!=="svelte-1m276jh"&&(ie.innerHTML=On),ho=r(e),f(qe.$$.fragment,e),uo=r(e),f(Be.$$.fragment,e),fo=r(e),Ge=m(e,"P",{"data-svelte-h":!0}),h(Ge)!=="svelte-1p51yb4"&&(Ge.textContent=Dn),go=r(e),f(Ne.$$.fragment,e),_o=r(e),f(Re.$$.fragment,e),bo=r(e),He=m(e,"P",{"data-svelte-h":!0}),h(He)!=="svelte-1fqdbu1"&&(He.textContent=Kn),Mo=r(e),f(Ve.$$.fragment,e),yo=r(e),f(Ee.$$.fragment,e),To=r(e),Ae=m(e,"UL",{"data-svelte-h":!0}),h(Ae)!=="svelte-p1b16m"&&(Ae.innerHTML=es),wo=r(e),f(Se.$$.fragment,e),ko=r(e),q=m(e,"DIV",{class:!0});var D=j(q);f(Qe.$$.fragment,D),No=r(D),bt=m(D,"P",{"data-svelte-h":!0}),h(bt)!=="svelte-9140bu"&&(bt.innerHTML=ts),Ro=r(D),Mt=m(D,"P",{"data-svelte-h":!0}),h(Mt)!=="svelte-1ek1ss9"&&(Mt.innerHTML=os),Ho=r(D),f(le.$$.fragment,D),D.forEach(s),vo=r(e),f(Ye.$$.fragment,e),Jo=r(e),v=m(e,"DIV",{class:!0});var $=j(v);f(Pe.$$.fragment,$),Vo=r($),yt=m($,"P",{"data-svelte-h":!0}),h(yt)!=="svelte-rehfhh"&&(yt.innerHTML=ns),Eo=r($),Tt=m($,"P",{"data-svelte-h":!0}),h(Tt)!=="svelte-174erte"&&(Tt.innerHTML=ss),Ao=r($),wt=m($,"P",{"data-svelte-h":!0}),h(wt)!=="svelte-joghtx"&&(wt.innerHTML=as),So=r($),kt=m($,"P",{"data-svelte-h":!0}),h(kt)!=="svelte-q52n56"&&(kt.innerHTML=rs),Qo=r($),vt=m($,"P",{"data-svelte-h":!0}),h(vt)!=="svelte-hswkmf"&&(vt.innerHTML=is),Yo=r($),se=m($,"DIV",{class:!0});var ae=j(se);f(Oe.$$.fragment,ae),Po=r(ae),Jt=m(ae,"P",{"data-svelte-h":!0}),h(Jt)!=="svelte-y3utz7"&&(Jt.innerHTML=ls),Oo=r(ae),f(de.$$.fragment,ae),ae.forEach(s),$.forEach(s),$o=r(e),f(De.$$.fragment,e),jo=r(e),x=m(e,"DIV",{class:!0});var B=j(x);f(Ke.$$.fragment,B),Do=r(B),$t=m(B,"P",{"data-svelte-h":!0}),h($t)!=="svelte-dygsvo"&&($t.innerHTML=ds),Ko=r(B),jt=m(B,"P",{"data-svelte-h":!0}),h(jt)!=="svelte-q52n56"&&(jt.innerHTML=cs),en=r(B),Ct=m(B,"P",{"data-svelte-h":!0}),h(Ct)!=="svelte-hswkmf"&&(Ct.innerHTML=ps),tn=r(B),A=m(B,"DIV",{class:!0});var K=j(A);f(et.$$.fragment,K),on=r(K),xt=m(K,"P",{"data-svelte-h":!0}),h(xt)!=="svelte-6hh75j"&&(xt.innerHTML=ms),nn=r(K),f(ce.$$.fragment,K),sn=r(K),f(pe.$$.fragment,K),K.forEach(s),B.forEach(s),Co=r(e),f(tt.$$.fragment,e),xo=r(e),X=m(e,"DIV",{class:!0});var G=j(X);f(ot.$$.fragment,G),an=r(G),Xt=m(G,"P",{"data-svelte-h":!0}),h(Xt)!=="svelte-1moqo98"&&(Xt.innerHTML=hs),rn=r(G),Ft=m(G,"P",{"data-svelte-h":!0}),h(Ft)!=="svelte-q52n56"&&(Ft.innerHTML=us),ln=r(G),zt=m(G,"P",{"data-svelte-h":!0}),h(zt)!=="svelte-hswkmf"&&(zt.innerHTML=fs),dn=r(G),S=m(G,"DIV",{class:!0});var ee=j(S);f(nt.$$.fragment,ee),cn=r(ee),Ut=m(ee,"P",{"data-svelte-h":!0}),h(Ut)!=="svelte-fh9827"&&(Ut.innerHTML=gs),pn=r(ee),f(me.$$.fragment,ee),mn=r(ee),f(he.$$.fragment,ee),ee.forEach(s),G.forEach(s),Xo=r(e),f(st.$$.fragment,e),Fo=r(e),F=m(e,"DIV",{class:!0});var N=j(F);f(at.$$.fragment,N),hn=r(N),Wt=m(N,"P",{"data-svelte-h":!0}),h(Wt)!=="svelte-1vluxlz"&&(Wt.textContent=_s),un=r(N),It=m(N,"P",{"data-svelte-h":!0}),h(It)!=="svelte-q52n56"&&(It.innerHTML=bs),fn=r(N),Zt=m(N,"P",{"data-svelte-h":!0}),h(Zt)!=="svelte-hswkmf"&&(Zt.innerHTML=Ms),gn=r(N),L=m(N,"DIV",{class:!0});var R=j(L);f(rt.$$.fragment,R),_n=r(R),Lt=m(R,"P",{"data-svelte-h":!0}),h(Lt)!=="svelte-1ap5mpv"&&(Lt.innerHTML=ys),bn=r(R),f(ue.$$.fragment,R),Mn=r(R),f(fe.$$.fragment,R),yn=r(R),f(ge.$$.fragment,R),R.forEach(s),N.forEach(s),zo=r(e),f(it.$$.fragment,e),Uo=r(e),z=m(e,"DIV",{class:!0});var H=j(z);f(lt.$$.fragment,H),Tn=r(H),qt=m(H,"P",{"data-svelte-h":!0}),h(qt)!=="svelte-x6zovt"&&(qt.textContent=Ts),wn=r(H),Bt=m(H,"P",{"data-svelte-h":!0}),h(Bt)!=="svelte-q52n56"&&(Bt.innerHTML=ws),kn=r(H),Gt=m(H,"P",{"data-svelte-h":!0}),h(Gt)!=="svelte-hswkmf"&&(Gt.innerHTML=ks),vn=r(H),Q=m(H,"DIV",{class:!0});var te=j(Q);f(dt.$$.fragment,te),Jn=r(te),Nt=m(te,"P",{"data-svelte-h":!0}),h(Nt)!=="svelte-19lv29b"&&(Nt.innerHTML=vs),$n=r(te),f(_e.$$.fragment,te),jn=r(te),f(be.$$.fragment,te),te.forEach(s),H.forEach(s),Wo=r(e),f(ct.$$.fragment,e),Io=r(e),U=m(e,"DIV",{class:!0});var V=j(U);f(pt.$$.fragment,V),Cn=r(V),Rt=m(V,"P",{"data-svelte-h":!0}),h(Rt)!=="svelte-1cy9rew"&&(Rt.textContent=Js),xn=r(V),Ht=m(V,"P",{"data-svelte-h":!0}),h(Ht)!=="svelte-q52n56"&&(Ht.innerHTML=$s),Xn=r(V),Vt=m(V,"P",{"data-svelte-h":!0}),h(Vt)!=="svelte-hswkmf"&&(Vt.innerHTML=js),Fn=r(V),Y=m(V,"DIV",{class:!0});var oe=j(Y);f(mt.$$.fragment,oe),zn=r(oe),Et=m(oe,"P",{"data-svelte-h":!0}),h(Et)!=="svelte-1pd9pej"&&(Et.innerHTML=Cs),Un=r(oe),f(Me.$$.fragment,oe),Wn=r(oe),f(ye.$$.fragment,oe),oe.forEach(s),V.forEach(s),Zo=r(e),f(ht.$$.fragment,e),Lo=r(e),W=m(e,"DIV",{class:!0});var E=j(W);f(ut.$$.fragment,E),In=r(E),At=m(E,"P",{"data-svelte-h":!0}),h(At)!=="svelte-p7gk59"&&(At.innerHTML=xs),Zn=r(E),St=m(E,"P",{"data-svelte-h":!0}),h(St)!=="svelte-q52n56"&&(St.innerHTML=Xs),Ln=r(E),Qt=m(E,"P",{"data-svelte-h":!0}),h(Qt)!=="svelte-hswkmf"&&(Qt.innerHTML=Fs),qn=r(E),P=m(E,"DIV",{class:!0});var ne=j(P);f(ft.$$.fragment,ne),Bn=r(ne),Yt=m(ne,"P",{"data-svelte-h":!0}),h(Yt)!=="svelte-971bs5"&&(Yt.innerHTML=zs),Gn=r(ne),f(Te.$$.fragment,ne),Nn=r(ne),f(we.$$.fragment,ne),ne.forEach(s),E.forEach(s),qo=r(e),f(gt.$$.fragment,e),Bo=r(e),Pt=m(e,"P",{}),j(Pt).forEach(s),this.h()},h(){J(t,"name","hf:doc:metadata"),J(t,"content",ta),J(re,"class","flex flex-wrap space-x-1"),J(ie,"start","2"),J(q,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(se,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(v,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(A,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(x,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(S,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(X,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(L,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(F,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(Q,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(z,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(Y,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(U,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(P,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(W,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,o){i(document.head,t),l(e,y,o),l(e,d,o),l(e,c,o),l(e,T,o),l(e,w,o),g(ve,e,o),l(e,Ot,o),l(e,re,o),l(e,Dt,o),g(Je,e,o),l(e,Kt,o),l(e,$e,o),l(e,eo,o),l(e,je,o),l(e,to,o),l(e,Ce,o),l(e,oo,o),l(e,xe,o),l(e,no,o),g(Xe,e,o),l(e,so,o),l(e,Fe,o),l(e,ao,o),l(e,ze,o),l(e,ro,o),g(Ue,e,o),l(e,io,o),g(We,e,o),l(e,lo,o),l(e,Ie,o),l(e,co,o),l(e,Ze,o),l(e,po,o),g(Le,e,o),l(e,mo,o),l(e,ie,o),l(e,ho,o),g(qe,e,o),l(e,uo,o),g(Be,e,o),l(e,fo,o),l(e,Ge,o),l(e,go,o),g(Ne,e,o),l(e,_o,o),g(Re,e,o),l(e,bo,o),l(e,He,o),l(e,Mo,o),g(Ve,e,o),l(e,yo,o),g(Ee,e,o),l(e,To,o),l(e,Ae,o),l(e,wo,o),g(Se,e,o),l(e,ko,o),l(e,q,o),g(Qe,q,null),i(q,No),i(q,bt),i(q,Ro),i(q,Mt),i(q,Ho),g(le,q,null),l(e,vo,o),g(Ye,e,o),l(e,Jo,o),l(e,v,o),g(Pe,v,null),i(v,Vo),i(v,yt),i(v,Eo),i(v,Tt),i(v,Ao),i(v,wt),i(v,So),i(v,kt),i(v,Qo),i(v,vt),i(v,Yo),i(v,se),g(Oe,se,null),i(se,Po),i(se,Jt),i(se,Oo),g(de,se,null),l(e,$o,o),g(De,e,o),l(e,jo,o),l(e,x,o),g(Ke,x,null),i(x,Do),i(x,$t),i(x,Ko),i(x,jt),i(x,en),i(x,Ct),i(x,tn),i(x,A),g(et,A,null),i(A,on),i(A,xt),i(A,nn),g(ce,A,null),i(A,sn),g(pe,A,null),l(e,Co,o),g(tt,e,o),l(e,xo,o),l(e,X,o),g(ot,X,null),i(X,an),i(X,Xt),i(X,rn),i(X,Ft),i(X,ln),i(X,zt),i(X,dn),i(X,S),g(nt,S,null),i(S,cn),i(S,Ut),i(S,pn),g(me,S,null),i(S,mn),g(he,S,null),l(e,Xo,o),g(st,e,o),l(e,Fo,o),l(e,F,o),g(at,F,null),i(F,hn),i(F,Wt),i(F,un),i(F,It),i(F,fn),i(F,Zt),i(F,gn),i(F,L),g(rt,L,null),i(L,_n),i(L,Lt),i(L,bn),g(ue,L,null),i(L,Mn),g(fe,L,null),i(L,yn),g(ge,L,null),l(e,zo,o),g(it,e,o),l(e,Uo,o),l(e,z,o),g(lt,z,null),i(z,Tn),i(z,qt),i(z,wn),i(z,Bt),i(z,kn),i(z,Gt),i(z,vn),i(z,Q),g(dt,Q,null),i(Q,Jn),i(Q,Nt),i(Q,$n),g(_e,Q,null),i(Q,jn),g(be,Q,null),l(e,Wo,o),g(ct,e,o),l(e,Io,o),l(e,U,o),g(pt,U,null),i(U,Cn),i(U,Rt),i(U,xn),i(U,Ht),i(U,Xn),i(U,Vt),i(U,Fn),i(U,Y),g(mt,Y,null),i(Y,zn),i(Y,Et),i(Y,Un),g(Me,Y,null),i(Y,Wn),g(ye,Y,null),l(e,Zo,o),g(ht,e,o),l(e,Lo,o),l(e,W,o),g(ut,W,null),i(W,In),i(W,At),i(W,Zn),i(W,St),i(W,Ln),i(W,Qt),i(W,qn),i(W,P),g(ft,P,null),i(P,Bn),i(P,Yt),i(P,Gn),g(Te,P,null),i(P,Nn),g(we,P,null),l(e,qo,o),g(gt,e,o),l(e,Bo,o),l(e,Pt,o),Go=!0},p(e,[o]){const D={};o&2&&(D.$$scope={dirty:o,ctx:e}),le.$set(D);const $={};o&2&&($.$$scope={dirty:o,ctx:e}),de.$set($);const ae={};o&2&&(ae.$$scope={dirty:o,ctx:e}),ce.$set(ae);const B={};o&2&&(B.$$scope={dirty:o,ctx:e}),pe.$set(B);const K={};o&2&&(K.$$scope={dirty:o,ctx:e}),me.$set(K);const G={};o&2&&(G.$$scope={dirty:o,ctx:e}),he.$set(G);const ee={};o&2&&(ee.$$scope={dirty:o,ctx:e}),ue.$set(ee);const N={};o&2&&(N.$$scope={dirty:o,ctx:e}),fe.$set(N);const R={};o&2&&(R.$$scope={dirty:o,ctx:e}),ge.$set(R);const H={};o&2&&(H.$$scope={dirty:o,ctx:e}),_e.$set(H);const te={};o&2&&(te.$$scope={dirty:o,ctx:e}),be.$set(te);const V={};o&2&&(V.$$scope={dirty:o,ctx:e}),Me.$set(V);const oe={};o&2&&(oe.$$scope={dirty:o,ctx:e}),ye.$set(oe);const E={};o&2&&(E.$$scope={dirty:o,ctx:e}),Te.$set(E);const ne={};o&2&&(ne.$$scope={dirty:o,ctx:e}),we.$set(ne)},i(e){Go||(_(ve.$$.fragment,e),_(Je.$$.fragment,e),_(Xe.$$.fragment,e),_(Ue.$$.fragment,e),_(We.$$.fragment,e),_(Le.$$.fragment,e),_(qe.$$.fragment,e),_(Be.$$.fragment,e),_(Ne.$$.fragment,e),_(Re.$$.fragment,e),_(Ve.$$.fragment,e),_(Ee.$$.fragment,e),_(Se.$$.fragment,e),_(Qe.$$.fragment,e),_(le.$$.fragment,e),_(Ye.$$.fragment,e),_(Pe.$$.fragment,e),_(Oe.$$.fragment,e),_(de.$$.fragment,e),_(De.$$.fragment,e),_(Ke.$$.fragment,e),_(et.$$.fragment,e),_(ce.$$.fragment,e),_(pe.$$.fragment,e),_(tt.$$.fragment,e),_(ot.$$.fragment,e),_(nt.$$.fragment,e),_(me.$$.fragment,e),_(he.$$.fragment,e),_(st.$$.fragment,e),_(at.$$.fragment,e),_(rt.$$.fragment,e),_(ue.$$.fragment,e),_(fe.$$.fragment,e),_(ge.$$.fragment,e),_(it.$$.fragment,e),_(lt.$$.fragment,e),_(dt.$$.fragment,e),_(_e.$$.fragment,e),_(be.$$.fragment,e),_(ct.$$.fragment,e),_(pt.$$.fragment,e),_(mt.$$.fragment,e),_(Me.$$.fragment,e),_(ye.$$.fragment,e),_(ht.$$.fragment,e),_(ut.$$.fragment,e),_(ft.$$.fragment,e),_(Te.$$.fragment,e),_(we.$$.fragment,e),_(gt.$$.fragment,e),Go=!0)},o(e){b(ve.$$.fragment,e),b(Je.$$.fragment,e),b(Xe.$$.fragment,e),b(Ue.$$.fragment,e),b(We.$$.fragment,e),b(Le.$$.fragment,e),b(qe.$$.fragment,e),b(Be.$$.fragment,e),b(Ne.$$.fragment,e),b(Re.$$.fragment,e),b(Ve.$$.fragment,e),b(Ee.$$.fragment,e),b(Se.$$.fragment,e),b(Qe.$$.fragment,e),b(le.$$.fragment,e),b(Ye.$$.fragment,e),b(Pe.$$.fragment,e),b(Oe.$$.fragment,e),b(de.$$.fragment,e),b(De.$$.fragment,e),b(Ke.$$.fragment,e),b(et.$$.fragment,e),b(ce.$$.fragment,e),b(pe.$$.fragment,e),b(tt.$$.fragment,e),b(ot.$$.fragment,e),b(nt.$$.fragment,e),b(me.$$.fragment,e),b(he.$$.fragment,e),b(st.$$.fragment,e),b(at.$$.fragment,e),b(rt.$$.fragment,e),b(ue.$$.fragment,e),b(fe.$$.fragment,e),b(ge.$$.fragment,e),b(it.$$.fragment,e),b(lt.$$.fragment,e),b(dt.$$.fragment,e),b(_e.$$.fragment,e),b(be.$$.fragment,e),b(ct.$$.fragment,e),b(pt.$$.fragment,e),b(mt.$$.fragment,e),b(Me.$$.fragment,e),b(ye.$$.fragment,e),b(ht.$$.fragment,e),b(ut.$$.fragment,e),b(ft.$$.fragment,e),b(Te.$$.fragment,e),b(we.$$.fragment,e),b(gt.$$.fragment,e),Go=!1},d(e){e&&(s(y),s(d),s(c),s(T),s(w),s(Ot),s(re),s(Dt),s(Kt),s($e),s(eo),s(je),s(to),s(Ce),s(oo),s(xe),s(no),s(so),s(Fe),s(ao),s(ze),s(ro),s(io),s(lo),s(Ie),s(co),s(Ze),s(po),s(mo),s(ie),s(ho),s(uo),s(fo),s(Ge),s(go),s(_o),s(bo),s(He),s(Mo),s(yo),s(To),s(Ae),s(wo),s(ko),s(q),s(vo),s(Jo),s(v),s($o),s(jo),s(x),s(Co),s(xo),s(X),s(Xo),s(Fo),s(F),s(zo),s(Uo),s(z),s(Wo),s(Io),s(U),s(Zo),s(Lo),s(W),s(qo),s(Bo),s(Pt)),s(t),M(ve,e),M(Je,e),M(Xe,e),M(Ue,e),M(We,e),M(Le,e),M(qe,e),M(Be,e),M(Ne,e),M(Re,e),M(Ve,e),M(Ee,e),M(Se,e),M(Qe),M(le),M(Ye,e),M(Pe),M(Oe),M(de),M(De,e),M(Ke),M(et),M(ce),M(pe),M(tt,e),M(ot),M(nt),M(me),M(he),M(st,e),M(at),M(rt),M(ue),M(fe),M(ge),M(it,e),M(lt),M(dt),M(_e),M(be),M(ct,e),M(pt),M(mt),M(Me),M(ye),M(ht,e),M(ut),M(ft),M(Te),M(we),M(gt,e)}}}const ta='{"title":"X-MOD","local":"x-mod","sections":[{"title":"Overview","local":"overview","sections":[],"depth":2},{"title":"Usage tips","local":"usage-tips","sections":[],"depth":2},{"title":"Adapter Usage","local":"adapter-usage","sections":[{"title":"Input language","local":"input-language","sections":[],"depth":3},{"title":"Fine-tuning","local":"fine-tuning","sections":[],"depth":3},{"title":"Cross-lingual transfer","local":"cross-lingual-transfer","sections":[],"depth":3}],"depth":2},{"title":"Resources","local":"resources","sections":[],"depth":2},{"title":"XmodConfig","local":"transformers.XmodConfig","sections":[],"depth":2},{"title":"XmodModel","local":"transformers.XmodModel","sections":[],"depth":2},{"title":"XmodForCausalLM","local":"transformers.XmodForCausalLM","sections":[],"depth":2},{"title":"XmodForMaskedLM","local":"transformers.XmodForMaskedLM","sections":[],"depth":2},{"title":"XmodForSequenceClassification","local":"transformers.XmodForSequenceClassification","sections":[],"depth":2},{"title":"XmodForMultipleChoice","local":"transformers.XmodForMultipleChoice","sections":[],"depth":2},{"title":"XmodForTokenClassification","local":"transformers.XmodForTokenClassification","sections":[],"depth":2},{"title":"XmodForQuestionAnswering","local":"transformers.XmodForQuestionAnswering","sections":[],"depth":2}],"depth":1}';function oa(k){return Ws(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class ca extends Is{constructor(t){super(),Zs(this,t,oa,ea,Us,{})}}export{ca as component};
