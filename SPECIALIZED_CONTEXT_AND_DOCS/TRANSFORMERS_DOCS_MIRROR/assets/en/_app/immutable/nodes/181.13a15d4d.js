import{s as co,o as po,n as z}from"../chunks/scheduler.18a86fab.js";import{S as uo,i as mo,g as u,s as r,r as f,A as ho,h as m,f as a,c as l,j as U,x as w,u as g,k as R,l as fo,y as c,a as d,v as _,d as b,t as y,w as M}from"../chunks/index.98837b22.js";import{T as ht}from"../chunks/Tip.77304350.js";import{D as A}from"../chunks/Docstring.a1ef7999.js";import{C as Y}from"../chunks/CodeBlock.8d0c2e8a.js";import{E as ft}from"../chunks/ExampleCodeBlock.8c3ee1f9.js";import{H as ue,E as go}from"../chunks/getInferenceSnippets.06c2775f.js";import{H as _o,a as zn}from"../chunks/HfOption.6641485e.js";function bo(v){let t,p="Click on the Falcon models in the right sidebar for more examples of how to apply Falcon to different language tasks.";return{c(){t=u("p"),t.textContent=p},l(n){t=m(n,"P",{"data-svelte-h":!0}),w(t)!=="svelte-k5mb5t"&&(t.textContent=p)},m(n,i){d(n,t,i)},p:z,d(n){n&&a(t)}}}function yo(v){let t,p;return t=new Y({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwcGlwZWxpbmUlMEElMEFwaXBlbGluZSUyMCUzRCUyMHBpcGVsaW5lKCUwQSUyMCUyMCUyMCUyMHRhc2slM0QlMjJ0ZXh0LWdlbmVyYXRpb24lMjIlMkMlMEElMjAlMjAlMjAlMjBtb2RlbCUzRCUyMnRpaXVhZSUyRmZhbGNvbi03Yi1pbnN0cnVjdCUyMiUyQyUwQSUyMCUyMCUyMCUyMGR0eXBlJTNEdG9yY2guYmZsb2F0MTYlMkMlMEElMjAlMjAlMjAlMjBkZXZpY2UlM0QwJTBBKSUwQXBpcGVsaW5lKCUwQSUyMCUyMCUyMCUyMCUyMldyaXRlJTIwYSUyMHNob3J0JTIwcG9lbSUyMGFib3V0JTIwY29kaW5nJTIyJTJDJTBBJTIwJTIwJTIwJTIwbWF4X2xlbmd0aCUzRDEwMCUyQyUwQSUyMCUyMCUyMCUyMGRvX3NhbXBsZSUzRFRydWUlMkMlMEElMjAlMjAlMjAlMjB0ZW1wZXJhdHVyZSUzRDAuNyUwQSk=",highlighted:`<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> pipeline

pipeline = pipeline(
    task=<span class="hljs-string">&quot;text-generation&quot;</span>,
    model=<span class="hljs-string">&quot;tiiuae/falcon-7b-instruct&quot;</span>,
    dtype=torch.bfloat16,
    device=<span class="hljs-number">0</span>
)
pipeline(
    <span class="hljs-string">&quot;Write a short poem about coding&quot;</span>,
    max_length=<span class="hljs-number">100</span>,
    do_sample=<span class="hljs-literal">True</span>,
    temperature=<span class="hljs-number">0.7</span>
)`,wrap:!1}}),{c(){f(t.$$.fragment)},l(n){g(t.$$.fragment,n)},m(n,i){_(t,n,i),p=!0},p:z,i(n){p||(b(t.$$.fragment,n),p=!0)},o(n){y(t.$$.fragment,n),p=!1},d(n){M(t,n)}}}function Mo(v){let t,p;return t=new Y({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Rva2VuaXplciUyQyUyMEF1dG9Nb2RlbEZvckNhdXNhbExNJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIydGlpdWFlJTJGZmFsY29uLTdiLWluc3RydWN0JTIyKSUwQW1vZGVsJTIwJTNEJTIwQXV0b01vZGVsRm9yQ2F1c2FsTE0uZnJvbV9wcmV0cmFpbmVkKCUwQSUyMCUyMCUyMCUyMCUyMnRpaXVhZSUyRmZhbGNvbi03Yi1pbnN0cnVjdCUyMiUyQyUwQSUyMCUyMCUyMCUyMGR0eXBlJTNEdG9yY2guYmZsb2F0MTYlMkMlMEElMjAlMjAlMjAlMjBkZXZpY2VfbWFwJTNEJTIyYXV0byUyMiUyQyUwQSUyMCUyMCUyMCUyMGF0dG5faW1wbGVtZW50YXRpb24lM0QlMjJzZHBhJTIyJTJDJTBBKSUwQSUwQWlucHV0X2lkcyUyMCUzRCUyMHRva2VuaXplciglMjJXcml0ZSUyMGElMjBzaG9ydCUyMHBvZW0lMjBhYm91dCUyMGNvZGluZyUyMiUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpLnRvKG1vZGVsLmRldmljZSklMEElMEFvdXRwdXQlMjAlM0QlMjBtb2RlbC5nZW5lcmF0ZSgqKmlucHV0X2lkcyklMEFwcmludCh0b2tlbml6ZXIuZGVjb2RlKG91dHB1dCU1QjAlNUQlMkMlMjBza2lwX3NwZWNpYWxfdG9rZW5zJTNEVHJ1ZSkp",highlighted:`<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;tiiuae/falcon-7b-instruct&quot;</span>)
model = AutoModelForCausalLM.from_pretrained(
    <span class="hljs-string">&quot;tiiuae/falcon-7b-instruct&quot;</span>,
    dtype=torch.bfloat16,
    device_map=<span class="hljs-string">&quot;auto&quot;</span>,
    attn_implementation=<span class="hljs-string">&quot;sdpa&quot;</span>,
)

input_ids = tokenizer(<span class="hljs-string">&quot;Write a short poem about coding&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>).to(model.device)

output = model.generate(**input_ids)
<span class="hljs-built_in">print</span>(tokenizer.decode(output[<span class="hljs-number">0</span>], skip_special_tokens=<span class="hljs-literal">True</span>))`,wrap:!1}}),{c(){f(t.$$.fragment)},l(n){g(t.$$.fragment,n)},m(n,i){_(t,n,i),p=!0},p:z,i(n){p||(b(t.$$.fragment,n),p=!0)},o(n){y(t.$$.fragment,n),p=!1},d(n){M(t,n)}}}function To(v){let t,p;return t=new Y({props:{code:"JTIzJTIwcGlwJTIwaW5zdGFsbCUyMC1VJTIwZmxhc2gtYXR0biUyMC0tbm8tYnVpbGQtaXNvbGF0aW9uJTBBdHJhbnNmb3JtZXJzJTIwY2hhdCUyMHRpaXVhZSUyRmZhbGNvbi03Yi1pbnN0cnVjdCUyMC0tZHR5cGUlMjBhdXRvJTIwLS1hdHRuX2ltcGxlbWVudGF0aW9uJTIwZmxhc2hfYXR0ZW50aW9uXzIlMjAtLWRldmljZSUyMDA=",highlighted:`<span class="hljs-comment"># pip install -U flash-attn --no-build-isolation</span>
transformers chat tiiuae/falcon-7b-instruct --dtype auto --attn_implementation flash_attention_2 --device 0`,wrap:!1}}),{c(){f(t.$$.fragment)},l(n){g(t.$$.fragment,n)},m(n,i){_(t,n,i),p=!0},p:z,i(n){p||(b(t.$$.fragment,n),p=!0)},o(n){y(t.$$.fragment,n),p=!1},d(n){M(t,n)}}}function wo(v){let t,p,n,i,T,o;return t=new zn({props:{id:"usage",option:"Pipeline",$$slots:{default:[yo]},$$scope:{ctx:v}}}),n=new zn({props:{id:"usage",option:"AutoModel",$$slots:{default:[Mo]},$$scope:{ctx:v}}}),T=new zn({props:{id:"usage",option:"transformers CLI",$$slots:{default:[To]},$$scope:{ctx:v}}}),{c(){f(t.$$.fragment),p=r(),f(n.$$.fragment),i=r(),f(T.$$.fragment)},l(h){g(t.$$.fragment,h),p=l(h),g(n.$$.fragment,h),i=l(h),g(T.$$.fragment,h)},m(h,k){_(t,h,k),d(h,p,k),_(n,h,k),d(h,i,k),_(T,h,k),o=!0},p(h,k){const gt={};k&2&&(gt.$$scope={dirty:k,ctx:h}),t.$set(gt);const me={};k&2&&(me.$$scope={dirty:k,ctx:h}),n.$set(me);const P={};k&2&&(P.$$scope={dirty:k,ctx:h}),T.$set(P)},i(h){o||(b(t.$$.fragment,h),b(n.$$.fragment,h),b(T.$$.fragment,h),o=!0)},o(h){y(t.$$.fragment,h),y(n.$$.fragment,h),y(T.$$.fragment,h),o=!1},d(h){h&&(a(p),a(i)),M(t,h),M(n,h),M(T,h)}}}function vo(v){let t,p="Example:",n,i,T;return i=new Y({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEZhbGNvbk1vZGVsJTJDJTIwRmFsY29uQ29uZmlnJTBBJTBBJTIzJTIwSW5pdGlhbGl6aW5nJTIwYSUyMHNtYWxsJTIwKDItbGF5ZXIpJTIwRmFsY29uJTIwY29uZmlndXJhdGlvbiUwQWNvbmZpZ3VyYXRpb24lMjAlM0QlMjBGYWxjb25Db25maWcobnVtX2hpZGRlbl9sYXllcnMlM0QyKSUwQSUwQSUyMyUyMEluaXRpYWxpemluZyUyMGElMjBtb2RlbCUyMGZyb20lMjB0aGUlMjBzbWFsbCUyMGNvbmZpZ3VyYXRpb24lMEFtb2RlbCUyMCUzRCUyMEZhbGNvbk1vZGVsKGNvbmZpZ3VyYXRpb24pJTBBJTBBJTIzJTIwQWNjZXNzaW5nJTIwdGhlJTIwbW9kZWwlMjBjb25maWd1cmF0aW9uJTBBY29uZmlndXJhdGlvbiUyMCUzRCUyMG1vZGVsLmNvbmZpZw==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> FalconModel, FalconConfig

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a small (2-layer) Falcon configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = FalconConfig(num_hidden_layers=<span class="hljs-number">2</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a model from the small configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = FalconModel(configuration)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Accessing the model configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = model.config`,wrap:!1}}),{c(){t=u("p"),t.textContent=p,n=r(),f(i.$$.fragment)},l(o){t=m(o,"P",{"data-svelte-h":!0}),w(t)!=="svelte-11lpom8"&&(t.textContent=p),n=l(o),g(i.$$.fragment,o)},m(o,h){d(o,t,h),d(o,n,h),_(i,o,h),T=!0},p:z,i(o){T||(b(i.$$.fragment,o),T=!0)},o(o){y(i.$$.fragment,o),T=!1},d(o){o&&(a(t),a(n)),M(i,o)}}}function ko(v){let t,p=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=u("p"),t.innerHTML=p},l(n){t=m(n,"P",{"data-svelte-h":!0}),w(t)!=="svelte-fincs2"&&(t.innerHTML=p)},m(n,i){d(n,t,i)},p:z,d(n){n&&a(t)}}}function Fo(v){let t,p=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=u("p"),t.innerHTML=p},l(n){t=m(n,"P",{"data-svelte-h":!0}),w(t)!=="svelte-fincs2"&&(t.innerHTML=p)},m(n,i){d(n,t,i)},p:z,d(n){n&&a(t)}}}function $o(v){let t,p="Example:",n,i,T;return i=new Y({props:{code:"",highlighted:"",wrap:!1}}),{c(){t=u("p"),t.textContent=p,n=r(),f(i.$$.fragment)},l(o){t=m(o,"P",{"data-svelte-h":!0}),w(t)!=="svelte-11lpom8"&&(t.textContent=p),n=l(o),g(i.$$.fragment,o)},m(o,h){d(o,t,h),d(o,n,h),_(i,o,h),T=!0},p:z,i(o){T||(b(i.$$.fragment,o),T=!0)},o(o){y(i.$$.fragment,o),T=!1},d(o){o&&(a(t),a(n)),M(i,o)}}}function Co(v){let t,p=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=u("p"),t.innerHTML=p},l(n){t=m(n,"P",{"data-svelte-h":!0}),w(t)!=="svelte-fincs2"&&(t.innerHTML=p)},m(n,i){d(n,t,i)},p:z,d(n){n&&a(t)}}}function Jo(v){let t,p="Example of single-label classification:",n,i,T;return i=new Y({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Rva2VuaXplciUyQyUyMEZhbGNvbkZvclNlcXVlbmNlQ2xhc3NpZmljYXRpb24lMEElMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJ0aWl1YWUlMkZmYWxjb24tN2IlMjIpJTBBbW9kZWwlMjAlM0QlMjBGYWxjb25Gb3JTZXF1ZW5jZUNsYXNzaWZpY2F0aW9uLmZyb21fcHJldHJhaW5lZCglMjJ0aWl1YWUlMkZmYWxjb24tN2IlMjIpJTBBJTBBaW5wdXRzJTIwJTNEJTIwdG9rZW5pemVyKCUyMkhlbGxvJTJDJTIwbXklMjBkb2clMjBpcyUyMGN1dGUlMjIlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQSUwQXdpdGglMjB0b3JjaC5ub19ncmFkKCklM0ElMEElMjAlMjAlMjAlMjBsb2dpdHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cykubG9naXRzJTBBJTBBcHJlZGljdGVkX2NsYXNzX2lkJTIwJTNEJTIwbG9naXRzLmFyZ21heCgpLml0ZW0oKSUwQW1vZGVsLmNvbmZpZy5pZDJsYWJlbCU1QnByZWRpY3RlZF9jbGFzc19pZCU1RCUwQSUwQSUyMyUyMFRvJTIwdHJhaW4lMjBhJTIwbW9kZWwlMjBvbiUyMCU2MG51bV9sYWJlbHMlNjAlMjBjbGFzc2VzJTJDJTIweW91JTIwY2FuJTIwcGFzcyUyMCU2MG51bV9sYWJlbHMlM0RudW1fbGFiZWxzJTYwJTIwdG8lMjAlNjAuZnJvbV9wcmV0cmFpbmVkKC4uLiklNjAlMEFudW1fbGFiZWxzJTIwJTNEJTIwbGVuKG1vZGVsLmNvbmZpZy5pZDJsYWJlbCklMEFtb2RlbCUyMCUzRCUyMEZhbGNvbkZvclNlcXVlbmNlQ2xhc3NpZmljYXRpb24uZnJvbV9wcmV0cmFpbmVkKCUyMnRpaXVhZSUyRmZhbGNvbi03YiUyMiUyQyUyMG51bV9sYWJlbHMlM0RudW1fbGFiZWxzKSUwQSUwQWxhYmVscyUyMCUzRCUyMHRvcmNoLnRlbnNvciglNUIxJTVEKSUwQWxvc3MlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyUyQyUyMGxhYmVscyUzRGxhYmVscykubG9zcyUwQXJvdW5kKGxvc3MuaXRlbSgpJTJDJTIwMik=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, FalconForSequenceClassification

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;tiiuae/falcon-7b&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = FalconForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;tiiuae/falcon-7b&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    logits = model(**inputs).logits

<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_class_id = logits.argmax().item()
<span class="hljs-meta">&gt;&gt;&gt; </span>model.config.id2label[predicted_class_id]
...

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># To train a model on \`num_labels\` classes, you can pass \`num_labels=num_labels\` to \`.from_pretrained(...)\`</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>num_labels = <span class="hljs-built_in">len</span>(model.config.id2label)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = FalconForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;tiiuae/falcon-7b&quot;</span>, num_labels=num_labels)

<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.tensor([<span class="hljs-number">1</span>])
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = model(**inputs, labels=labels).loss
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">round</span>(loss.item(), <span class="hljs-number">2</span>)
...`,wrap:!1}}),{c(){t=u("p"),t.textContent=p,n=r(),f(i.$$.fragment)},l(o){t=m(o,"P",{"data-svelte-h":!0}),w(t)!=="svelte-ykxpe4"&&(t.textContent=p),n=l(o),g(i.$$.fragment,o)},m(o,h){d(o,t,h),d(o,n,h),_(i,o,h),T=!0},p:z,i(o){T||(b(i.$$.fragment,o),T=!0)},o(o){y(i.$$.fragment,o),T=!1},d(o){o&&(a(t),a(n)),M(i,o)}}}function jo(v){let t,p="Example of multi-label classification:",n,i,T;return i=new Y({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Rva2VuaXplciUyQyUyMEZhbGNvbkZvclNlcXVlbmNlQ2xhc3NpZmljYXRpb24lMEElMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJ0aWl1YWUlMkZmYWxjb24tN2IlMjIpJTBBbW9kZWwlMjAlM0QlMjBGYWxjb25Gb3JTZXF1ZW5jZUNsYXNzaWZpY2F0aW9uLmZyb21fcHJldHJhaW5lZCglMjJ0aWl1YWUlMkZmYWxjb24tN2IlMjIlMkMlMjBwcm9ibGVtX3R5cGUlM0QlMjJtdWx0aV9sYWJlbF9jbGFzc2lmaWNhdGlvbiUyMiklMEElMEFpbnB1dHMlMjAlM0QlMjB0b2tlbml6ZXIoJTIySGVsbG8lMkMlMjBteSUyMGRvZyUyMGlzJTIwY3V0ZSUyMiUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpJTBBJTBBd2l0aCUyMHRvcmNoLm5vX2dyYWQoKSUzQSUwQSUyMCUyMCUyMCUyMGxvZ2l0cyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzKS5sb2dpdHMlMEElMEFwcmVkaWN0ZWRfY2xhc3NfaWRzJTIwJTNEJTIwdG9yY2guYXJhbmdlKDAlMkMlMjBsb2dpdHMuc2hhcGUlNUItMSU1RCklNUJ0b3JjaC5zaWdtb2lkKGxvZ2l0cykuc3F1ZWV6ZShkaW0lM0QwKSUyMCUzRSUyMDAuNSU1RCUwQSUwQSUyMyUyMFRvJTIwdHJhaW4lMjBhJTIwbW9kZWwlMjBvbiUyMCU2MG51bV9sYWJlbHMlNjAlMjBjbGFzc2VzJTJDJTIweW91JTIwY2FuJTIwcGFzcyUyMCU2MG51bV9sYWJlbHMlM0RudW1fbGFiZWxzJTYwJTIwdG8lMjAlNjAuZnJvbV9wcmV0cmFpbmVkKC4uLiklNjAlMEFudW1fbGFiZWxzJTIwJTNEJTIwbGVuKG1vZGVsLmNvbmZpZy5pZDJsYWJlbCklMEFtb2RlbCUyMCUzRCUyMEZhbGNvbkZvclNlcXVlbmNlQ2xhc3NpZmljYXRpb24uZnJvbV9wcmV0cmFpbmVkKCUwQSUyMCUyMCUyMCUyMCUyMnRpaXVhZSUyRmZhbGNvbi03YiUyMiUyQyUyMG51bV9sYWJlbHMlM0RudW1fbGFiZWxzJTJDJTIwcHJvYmxlbV90eXBlJTNEJTIybXVsdGlfbGFiZWxfY2xhc3NpZmljYXRpb24lMjIlMEEpJTBBJTBBbGFiZWxzJTIwJTNEJTIwdG9yY2guc3VtKCUwQSUyMCUyMCUyMCUyMHRvcmNoLm5uLmZ1bmN0aW9uYWwub25lX2hvdChwcmVkaWN0ZWRfY2xhc3NfaWRzJTVCTm9uZSUyQyUyMCUzQSU1RC5jbG9uZSgpJTJDJTIwbnVtX2NsYXNzZXMlM0RudW1fbGFiZWxzKSUyQyUyMGRpbSUzRDElMEEpLnRvKHRvcmNoLmZsb2F0KSUwQWxvc3MlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyUyQyUyMGxhYmVscyUzRGxhYmVscykubG9zcw==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, FalconForSequenceClassification

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;tiiuae/falcon-7b&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = FalconForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;tiiuae/falcon-7b&quot;</span>, problem_type=<span class="hljs-string">&quot;multi_label_classification&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    logits = model(**inputs).logits

<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_class_ids = torch.arange(<span class="hljs-number">0</span>, logits.shape[-<span class="hljs-number">1</span>])[torch.sigmoid(logits).squeeze(dim=<span class="hljs-number">0</span>) &gt; <span class="hljs-number">0.5</span>]

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># To train a model on \`num_labels\` classes, you can pass \`num_labels=num_labels\` to \`.from_pretrained(...)\`</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>num_labels = <span class="hljs-built_in">len</span>(model.config.id2label)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = FalconForSequenceClassification.from_pretrained(
<span class="hljs-meta">... </span>    <span class="hljs-string">&quot;tiiuae/falcon-7b&quot;</span>, num_labels=num_labels, problem_type=<span class="hljs-string">&quot;multi_label_classification&quot;</span>
<span class="hljs-meta">... </span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.<span class="hljs-built_in">sum</span>(
<span class="hljs-meta">... </span>    torch.nn.functional.one_hot(predicted_class_ids[<span class="hljs-literal">None</span>, :].clone(), num_classes=num_labels), dim=<span class="hljs-number">1</span>
<span class="hljs-meta">... </span>).to(torch.<span class="hljs-built_in">float</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = model(**inputs, labels=labels).loss`,wrap:!1}}),{c(){t=u("p"),t.textContent=p,n=r(),f(i.$$.fragment)},l(o){t=m(o,"P",{"data-svelte-h":!0}),w(t)!=="svelte-1l8e32d"&&(t.textContent=p),n=l(o),g(i.$$.fragment,o)},m(o,h){d(o,t,h),d(o,n,h),_(i,o,h),T=!0},p:z,i(o){T||(b(i.$$.fragment,o),T=!0)},o(o){y(i.$$.fragment,o),T=!1},d(o){o&&(a(t),a(n)),M(i,o)}}}function xo(v){let t,p=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=u("p"),t.innerHTML=p},l(n){t=m(n,"P",{"data-svelte-h":!0}),w(t)!=="svelte-fincs2"&&(t.innerHTML=p)},m(n,i){d(n,t,i)},p:z,d(n){n&&a(t)}}}function Uo(v){let t,p="Example:",n,i,T;return i=new Y({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBGYWxjb25Gb3JUb2tlbkNsYXNzaWZpY2F0aW9uJTBBaW1wb3J0JTIwdG9yY2glMEElMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJ0aWl1YWUlMkZmYWxjb24tN2IlMjIpJTBBbW9kZWwlMjAlM0QlMjBGYWxjb25Gb3JUb2tlbkNsYXNzaWZpY2F0aW9uLmZyb21fcHJldHJhaW5lZCglMjJ0aWl1YWUlMkZmYWxjb24tN2IlMjIpJTBBJTBBaW5wdXRzJTIwJTNEJTIwdG9rZW5pemVyKCUwQSUyMCUyMCUyMCUyMCUyMkh1Z2dpbmdGYWNlJTIwaXMlMjBhJTIwY29tcGFueSUyMGJhc2VkJTIwaW4lMjBQYXJpcyUyMGFuZCUyME5ldyUyMFlvcmslMjIlMkMlMjBhZGRfc3BlY2lhbF90b2tlbnMlM0RGYWxzZSUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIlMEEpJTBBJTBBd2l0aCUyMHRvcmNoLm5vX2dyYWQoKSUzQSUwQSUyMCUyMCUyMCUyMGxvZ2l0cyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzKS5sb2dpdHMlMEElMEFwcmVkaWN0ZWRfdG9rZW5fY2xhc3NfaWRzJTIwJTNEJTIwbG9naXRzLmFyZ21heCgtMSklMEElMEElMjMlMjBOb3RlJTIwdGhhdCUyMHRva2VucyUyMGFyZSUyMGNsYXNzaWZpZWQlMjByYXRoZXIlMjB0aGVuJTIwaW5wdXQlMjB3b3JkcyUyMHdoaWNoJTIwbWVhbnMlMjB0aGF0JTBBJTIzJTIwdGhlcmUlMjBtaWdodCUyMGJlJTIwbW9yZSUyMHByZWRpY3RlZCUyMHRva2VuJTIwY2xhc3NlcyUyMHRoYW4lMjB3b3Jkcy4lMEElMjMlMjBNdWx0aXBsZSUyMHRva2VuJTIwY2xhc3NlcyUyMG1pZ2h0JTIwYWNjb3VudCUyMGZvciUyMHRoZSUyMHNhbWUlMjB3b3JkJTBBcHJlZGljdGVkX3Rva2Vuc19jbGFzc2VzJTIwJTNEJTIwJTVCbW9kZWwuY29uZmlnLmlkMmxhYmVsJTVCdC5pdGVtKCklNUQlMjBmb3IlMjB0JTIwaW4lMjBwcmVkaWN0ZWRfdG9rZW5fY2xhc3NfaWRzJTVCMCU1RCU1RCUwQXByZWRpY3RlZF90b2tlbnNfY2xhc3NlcyUwQSUwQWxhYmVscyUyMCUzRCUyMHByZWRpY3RlZF90b2tlbl9jbGFzc19pZHMlMEFsb3NzJTIwJTNEJTIwbW9kZWwoKippbnB1dHMlMkMlMjBsYWJlbHMlM0RsYWJlbHMpLmxvc3MlMEFyb3VuZChsb3NzLml0ZW0oKSUyQyUyMDIp",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, FalconForTokenClassification
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;tiiuae/falcon-7b&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = FalconForTokenClassification.from_pretrained(<span class="hljs-string">&quot;tiiuae/falcon-7b&quot;</span>)

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
...`,wrap:!1}}),{c(){t=u("p"),t.textContent=p,n=r(),f(i.$$.fragment)},l(o){t=m(o,"P",{"data-svelte-h":!0}),w(t)!=="svelte-11lpom8"&&(t.textContent=p),n=l(o),g(i.$$.fragment,o)},m(o,h){d(o,t,h),d(o,n,h),_(i,o,h),T=!0},p:z,i(o){T||(b(i.$$.fragment,o),T=!0)},o(o){y(i.$$.fragment,o),T=!1},d(o){o&&(a(t),a(n)),M(i,o)}}}function zo(v){let t,p=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=u("p"),t.innerHTML=p},l(n){t=m(n,"P",{"data-svelte-h":!0}),w(t)!=="svelte-fincs2"&&(t.innerHTML=p)},m(n,i){d(n,t,i)},p:z,d(n){n&&a(t)}}}function Wo(v){let t,p="Example:",n,i,T;return i=new Y({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBGYWxjb25Gb3JRdWVzdGlvbkFuc3dlcmluZyUwQWltcG9ydCUyMHRvcmNoJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIydGlpdWFlJTJGZmFsY29uLTdiJTIyKSUwQW1vZGVsJTIwJTNEJTIwRmFsY29uRm9yUXVlc3Rpb25BbnN3ZXJpbmcuZnJvbV9wcmV0cmFpbmVkKCUyMnRpaXVhZSUyRmZhbGNvbi03YiUyMiklMEElMEFxdWVzdGlvbiUyQyUyMHRleHQlMjAlM0QlMjAlMjJXaG8lMjB3YXMlMjBKaW0lMjBIZW5zb24lM0YlMjIlMkMlMjAlMjJKaW0lMjBIZW5zb24lMjB3YXMlMjBhJTIwbmljZSUyMHB1cHBldCUyMiUwQSUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplcihxdWVzdGlvbiUyQyUyMHRleHQlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQXdpdGglMjB0b3JjaC5ub19ncmFkKCklM0ElMEElMjAlMjAlMjAlMjBvdXRwdXRzJTIwJTNEJTIwbW9kZWwoKippbnB1dHMpJTBBJTBBYW5zd2VyX3N0YXJ0X2luZGV4JTIwJTNEJTIwb3V0cHV0cy5zdGFydF9sb2dpdHMuYXJnbWF4KCklMEFhbnN3ZXJfZW5kX2luZGV4JTIwJTNEJTIwb3V0cHV0cy5lbmRfbG9naXRzLmFyZ21heCgpJTBBJTBBcHJlZGljdF9hbnN3ZXJfdG9rZW5zJTIwJTNEJTIwaW5wdXRzLmlucHV0X2lkcyU1QjAlMkMlMjBhbnN3ZXJfc3RhcnRfaW5kZXglMjAlM0ElMjBhbnN3ZXJfZW5kX2luZGV4JTIwJTJCJTIwMSU1RCUwQXRva2VuaXplci5kZWNvZGUocHJlZGljdF9hbnN3ZXJfdG9rZW5zJTJDJTIwc2tpcF9zcGVjaWFsX3Rva2VucyUzRFRydWUpJTBBJTBBJTIzJTIwdGFyZ2V0JTIwaXMlMjAlMjJuaWNlJTIwcHVwcGV0JTIyJTBBdGFyZ2V0X3N0YXJ0X2luZGV4JTIwJTNEJTIwdG9yY2gudGVuc29yKCU1QjE0JTVEKSUwQXRhcmdldF9lbmRfaW5kZXglMjAlM0QlMjB0b3JjaC50ZW5zb3IoJTVCMTUlNUQpJTBBJTBBb3V0cHV0cyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzJTJDJTIwc3RhcnRfcG9zaXRpb25zJTNEdGFyZ2V0X3N0YXJ0X2luZGV4JTJDJTIwZW5kX3Bvc2l0aW9ucyUzRHRhcmdldF9lbmRfaW5kZXgpJTBBbG9zcyUyMCUzRCUyMG91dHB1dHMubG9zcyUwQXJvdW5kKGxvc3MuaXRlbSgpJTJDJTIwMik=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, FalconForQuestionAnswering
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;tiiuae/falcon-7b&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = FalconForQuestionAnswering.from_pretrained(<span class="hljs-string">&quot;tiiuae/falcon-7b&quot;</span>)

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
...`,wrap:!1}}),{c(){t=u("p"),t.textContent=p,n=r(),f(i.$$.fragment)},l(o){t=m(o,"P",{"data-svelte-h":!0}),w(t)!=="svelte-11lpom8"&&(t.textContent=p),n=l(o),g(i.$$.fragment,o)},m(o,h){d(o,t,h),d(o,n,h),_(i,o,h),T=!0},p:z,i(o){T||(b(i.$$.fragment,o),T=!0)},o(o){y(i.$$.fragment,o),T=!1},d(o){o&&(a(t),a(n)),M(i,o)}}}function Zo(v){let t,p,n,i,T,o="<em>This model was released on 2023-11-28 and added to Hugging Face Transformers on 2023-07-11.</em>",h,k,gt='<div class="flex flex-wrap space-x-1"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/> <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat"/> <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/></div>',me,P,yt,he,Wn='<a href="https://huggingface.co/papers/2311.16867" rel="nofollow">Falcon</a> is a family of large language models, available in 7B, 40B, and 180B parameters, as pretrained and instruction tuned variants. This model focuses on scaling pretraining over three categories, performance, data, and hardware. Falcon uses multigroup attention to significantly reduce inference memory requirements and rotary positional embeddings (RoPE). These models are pretrained on <a href="https://huggingface.co/datasets/tiiuae/falcon-refinedweb" rel="nofollow">RefinedWeb</a>, a high-quality and deduplicated 5T token dataset.',Mt,fe,Zn='You can find all the original Falcon checkpoints under the <a href="https://huggingface.co/collections/tiiuae/falcon-64fb432660017eeec9837b5a" rel="nofollow">Falcon</a> collection.',Tt,K,wt,ge,In='The example below demonstrates how to generate text with <a href="/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.Pipeline">Pipeline</a>, <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel">AutoModel</a>, and from the command line.',vt,ee,kt,_e,Nn='Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the <a href="../quantization/overview">Quantization</a> overview for more available quantization backends.',Ft,be,qn='The example below uses <a href="../quantization/bitsandbytes">bitsandbytes</a> to only quantize the weights to 4-bits.',$t,ye,Ct,Me,Jt,He,Te,Xe,Gn='If you’re upgrading from an older custom code checkpoint, remember to convert it to the official Transformers format for better stability and performance using the conversion script located in the <a href="https://github.com/huggingface/transformers/tree/main/src/transformers/models/falcon" rel="nofollow">Falcon model directory</a>.',Lt,we,jt,ve,xt,Z,ke,Et,Le,Bn=`This is the configuration class to store the configuration of a <a href="/docs/transformers/v4.56.2/en/model_doc/falcon#transformers.FalconModel">FalconModel</a>. It is used to instantiate a Falcon
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the
<a href="https://huggingface.co/tiiuae/falcon-7b" rel="nofollow">tiiuae/falcon-7b</a> architecture.`,St,Ee,Rn=`Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> and can be used to control the model outputs. Read the
documentation from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> for more information.`,Qt,te,Ut,Fe,zt,C,$e,At,Se,Vn="The bare Falcon Model outputting raw hidden-states without any specific head on top.",Yt,Qe,Hn=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Pt,Ae,Xn=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Ot,O,Ce,Dt,Ye,Ln='The <a href="/docs/transformers/v4.56.2/en/model_doc/falcon#transformers.FalconModel">FalconModel</a> forward method, overrides the <code>__call__</code> special method.',Kt,ne,Wt,Je,Zt,J,je,en,Pe,En="The Falcon Model transformer with a language modeling head on top (linear layer with weights tied to the input embeddings).",tn,Oe,Sn=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,nn,De,Qn=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,on,V,xe,sn,Ke,An='The <a href="/docs/transformers/v4.56.2/en/model_doc/falcon#transformers.FalconForCausalLM">FalconForCausalLM</a> forward method, overrides the <code>__call__</code> special method.',an,oe,rn,se,It,Ue,Nt,F,ze,ln,et,Yn="The Falcon Model transformer with a sequence classification head on top (linear layer).",cn,tt,Pn=`<a href="/docs/transformers/v4.56.2/en/model_doc/falcon#transformers.FalconForSequenceClassification">FalconForSequenceClassification</a> uses the last token in order to do the classification, as other causal models
(e.g. GPT-1) do.`,dn,nt,On=`Since it does classification on the last token, it requires to know the position of the last token. If a
<code>pad_token_id</code> is defined in the configuration, it finds the last token that is not a padding token in each row. If
no <code>pad_token_id</code> is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
padding tokens when <code>inputs_embeds</code> are passed instead of <code>input_ids</code>, it does the same (take the last value in
each row of the batch).`,pn,ot,Dn=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,un,st,Kn=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,mn,W,We,hn,at,eo='The <a href="/docs/transformers/v4.56.2/en/model_doc/falcon#transformers.FalconForSequenceClassification">FalconForSequenceClassification</a> forward method, overrides the <code>__call__</code> special method.',fn,ae,gn,re,_n,le,qt,Ze,Gt,j,Ie,bn,rt,to=`The Falcon transformer with a token classification head on top (a linear layer on top of the hidden-states
output) e.g. for Named-Entity-Recognition (NER) tasks.`,yn,lt,no=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Mn,it,oo=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Tn,H,Ne,wn,ct,so='The <a href="/docs/transformers/v4.56.2/en/model_doc/falcon#transformers.FalconForTokenClassification">FalconForTokenClassification</a> forward method, overrides the <code>__call__</code> special method.',vn,ie,kn,ce,Bt,qe,Rt,x,Ge,Fn,dt,ao=`The Falcon transformer with a span classification head on top for extractive question-answering tasks like
SQuAD (a linear layer on top of the hidden-states output to compute <code>span start logits</code> and <code>span end logits</code>).`,$n,pt,ro=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Cn,ut,lo=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Jn,X,Be,jn,mt,io='The <a href="/docs/transformers/v4.56.2/en/model_doc/falcon#transformers.FalconForQuestionAnswering">FalconForQuestionAnswering</a> forward method, overrides the <code>__call__</code> special method.',xn,de,Un,pe,Vt,Re,Ht,_t,Xt;return P=new ue({props:{title:"Falcon",local:"falcon",headingTag:"h1"}}),K=new ht({props:{warning:!1,$$slots:{default:[bo]},$$scope:{ctx:v}}}),ee=new _o({props:{id:"usage",options:["Pipeline","AutoModel","transformers CLI"],$$slots:{default:[wo]},$$scope:{ctx:v}}}),ye=new Y({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Rva2VuaXplciUyQyUyMEF1dG9Nb2RlbEZvckNhdXNhbExNJTJDJTIwQml0c0FuZEJ5dGVzQ29uZmlnJTBBJTBBcXVhbnRpemF0aW9uX2NvbmZpZyUyMCUzRCUyMEJpdHNBbmRCeXRlc0NvbmZpZyglMEElMjAlMjAlMjAlMjBsb2FkX2luXzRiaXQlM0RUcnVlJTJDJTBBJTIwJTIwJTIwJTIwYm5iXzRiaXRfY29tcHV0ZV9kdHlwZSUzRHRvcmNoLmJmbG9hdDE2JTJDJTBBJTIwJTIwJTIwJTIwYm5iXzRiaXRfcXVhbnRfdHlwZSUzRCUyMm5mNCUyMiUyQyUwQSUyMCUyMCUyMCUyMGJuYl80Yml0X3VzZV9kb3VibGVfcXVhbnQlM0RUcnVlJTJDJTBBKSUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMnRpaXVhZSUyRmZhbGNvbi03YiUyMiklMEFtb2RlbCUyMCUzRCUyMEF1dG9Nb2RlbEZvckNhdXNhbExNLmZyb21fcHJldHJhaW5lZCglMEElMjAlMjAlMjAlMjAlMjJ0aWl1YWUlMkZmYWxjb24tN2IlMjIlMkMlMEElMjAlMjAlMjAlMjBkdHlwZSUzRHRvcmNoLmJmbG9hdDE2JTJDJTBBJTIwJTIwJTIwJTIwZGV2aWNlX21hcCUzRCUyMmF1dG8lMjIlMkMlMEElMjAlMjAlMjAlMjBxdWFudGl6YXRpb25fY29uZmlnJTNEcXVhbnRpemF0aW9uX2NvbmZpZyUyQyUwQSklMEElMEFpbnB1dHMlMjAlM0QlMjB0b2tlbml6ZXIoJTIySW4lMjBxdWFudHVtJTIwcGh5c2ljcyUyQyUyMGVudGFuZ2xlbWVudCUyMG1lYW5zJTIyJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMikudG8obW9kZWwuZGV2aWNlKSUwQW91dHB1dHMlMjAlM0QlMjBtb2RlbC5nZW5lcmF0ZSgqKmlucHV0cyUyQyUyMG1heF9uZXdfdG9rZW5zJTNEMTAwKSUwQXByaW50KHRva2VuaXplci5kZWNvZGUob3V0cHV0cyU1QjAlNUQlMkMlMjBza2lwX3NwZWNpYWxfdG9rZW5zJTNEVHJ1ZSkp",highlighted:`<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=<span class="hljs-literal">True</span>,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type=<span class="hljs-string">&quot;nf4&quot;</span>,
    bnb_4bit_use_double_quant=<span class="hljs-literal">True</span>,
)

tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;tiiuae/falcon-7b&quot;</span>)
model = AutoModelForCausalLM.from_pretrained(
    <span class="hljs-string">&quot;tiiuae/falcon-7b&quot;</span>,
    dtype=torch.bfloat16,
    device_map=<span class="hljs-string">&quot;auto&quot;</span>,
    quantization_config=quantization_config,
)

inputs = tokenizer(<span class="hljs-string">&quot;In quantum physics, entanglement means&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>).to(model.device)
outputs = model.generate(**inputs, max_new_tokens=<span class="hljs-number">100</span>)
<span class="hljs-built_in">print</span>(tokenizer.decode(outputs[<span class="hljs-number">0</span>], skip_special_tokens=<span class="hljs-literal">True</span>))`,wrap:!1}}),Me=new ue({props:{title:"Notes",local:"notes",headingTag:"h2"}}),we=new Y({props:{code:"cHl0aG9uJTIwY29udmVydF9jdXN0b21fY29kZV9jaGVja3BvaW50LnB5JTIwLS1jaGVja3BvaW50X2RpciUyMG15X21vZGVs",highlighted:"python convert_custom_code_checkpoint.py --checkpoint_dir my_model",wrap:!1}}),ve=new ue({props:{title:"FalconConfig",local:"transformers.FalconConfig",headingTag:"h2"}}),ke=new A({props:{name:"class transformers.FalconConfig",anchor:"transformers.FalconConfig",parameters:[{name:"vocab_size",val:" = 65024"},{name:"hidden_size",val:" = 4544"},{name:"num_hidden_layers",val:" = 32"},{name:"num_attention_heads",val:" = 71"},{name:"num_ln_in_parallel_attn",val:" = None"},{name:"layer_norm_epsilon",val:" = 1e-05"},{name:"initializer_range",val:" = 0.02"},{name:"use_cache",val:" = True"},{name:"hidden_dropout",val:" = 0.0"},{name:"attention_dropout",val:" = 0.0"},{name:"num_kv_heads",val:" = None"},{name:"alibi",val:" = False"},{name:"new_decoder_architecture",val:" = False"},{name:"multi_query",val:" = True"},{name:"parallel_attn",val:" = True"},{name:"bias",val:" = False"},{name:"max_position_embeddings",val:" = 2048"},{name:"rope_theta",val:" = 10000.0"},{name:"rope_scaling",val:" = None"},{name:"bos_token_id",val:" = 11"},{name:"eos_token_id",val:" = 11"},{name:"ffn_hidden_size",val:" = None"},{name:"activation",val:" = 'gelu'"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.FalconConfig.vocab_size",description:`<strong>vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 65024) &#x2014;
Vocabulary size of the Falcon model. Defines the number of different tokens that can be represented by the
<code>inputs_ids</code> passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/falcon#transformers.FalconModel">FalconModel</a>`,name:"vocab_size"},{anchor:"transformers.FalconConfig.hidden_size",description:`<strong>hidden_size</strong> (<code>int</code>, <em>optional</em>, defaults to 4544) &#x2014;
Dimension of the hidden representations.`,name:"hidden_size"},{anchor:"transformers.FalconConfig.num_hidden_layers",description:`<strong>num_hidden_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 32) &#x2014;
Number of hidden layers in the Transformer decoder.`,name:"num_hidden_layers"},{anchor:"transformers.FalconConfig.num_attention_heads",description:`<strong>num_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 71) &#x2014;
Number of attention heads for each attention layer in the Transformer encoder.`,name:"num_attention_heads"},{anchor:"transformers.FalconConfig.num_ln_in_parallel_attn",description:`<strong>num_ln_in_parallel_attn</strong> (<code>int</code>, <em>optional</em>) &#x2014;
Set to 2 if separate layer norms are to be used for the MLP and the attention output when using parallel
attention, otherwise, 1.`,name:"num_ln_in_parallel_attn"},{anchor:"transformers.FalconConfig.layer_norm_epsilon",description:`<strong>layer_norm_epsilon</strong> (<code>float</code>, <em>optional</em>, defaults to 1e-05) &#x2014;
The epsilon used by the layer normalization layers.`,name:"layer_norm_epsilon"},{anchor:"transformers.FalconConfig.initializer_range",description:`<strong>initializer_range</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.`,name:"initializer_range"},{anchor:"transformers.FalconConfig.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether the model should return the last key/values attentions (not used by all models). Only relevant if
<code>config.is_decoder=True</code>.`,name:"use_cache"},{anchor:"transformers.FalconConfig.hidden_dropout",description:`<strong>hidden_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The dropout probability for MLP layers.`,name:"hidden_dropout"},{anchor:"transformers.FalconConfig.attention_dropout",description:`<strong>attention_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The dropout probability for attention layers.`,name:"attention_dropout"},{anchor:"transformers.FalconConfig.num_kv_heads",description:`<strong>num_kv_heads</strong> (<code>int</code>, <em>optional</em>) &#x2014;
Number of key-value heads to use per attention layer. If unset, defaults to the same value as
<code>num_attention_heads</code>.`,name:"num_kv_heads"},{anchor:"transformers.FalconConfig.alibi",description:`<strong>alibi</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether to use ALiBi positional biases during self-attention.`,name:"alibi"},{anchor:"transformers.FalconConfig.new_decoder_architecture",description:`<strong>new_decoder_architecture</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether to use the new (Falcon-40B) decoder architecture. If <code>True</code>, the <code>multi_query</code> and <code>parallel_attn</code>
arguments are ignored, as the new decoder always uses parallel attention.`,name:"new_decoder_architecture"},{anchor:"transformers.FalconConfig.multi_query",description:`<strong>multi_query</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether to use multi-query attention in the decoder. Ignored when <code>new_decoder_architecture</code> is <code>True</code>.`,name:"multi_query"},{anchor:"transformers.FalconConfig.parallel_attn",description:`<strong>parallel_attn</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether to compute attention in parallel with the feedforward layer. If False, they are consecutive
instead, as in the original Transformer architecture. Ignored when <code>new_decoder_architecture</code> is <code>True</code>.`,name:"parallel_attn"},{anchor:"transformers.FalconConfig.bias",description:`<strong>bias</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether to use bias on Linear layers.`,name:"bias"},{anchor:"transformers.FalconConfig.max_position_embeddings",description:`<strong>max_position_embeddings</strong> (<code>int</code>, <em>optional</em>, defaults to 2048) &#x2014;
The maximum sequence length that this model might ever be used with, when <code>alibi</code> is <code>False</code>. Pretrained
Falcon models with RoPE support up to 2048 tokens.`,name:"max_position_embeddings"},{anchor:"transformers.FalconConfig.rope_theta",description:`<strong>rope_theta</strong> (<code>float</code>, <em>optional</em>, defaults to 10000.0) &#x2014;
The base period of the RoPE embeddings.`,name:"rope_theta"},{anchor:"transformers.FalconConfig.rope_scaling",description:`<strong>rope_scaling</strong> (<code>Dict</code>, <em>optional</em>) &#x2014;
Dictionary containing the scaling configuration for the RoPE embeddings. NOTE: if you apply new rope type
and you expect the model to work on longer <code>max_position_embeddings</code>, we recommend you to update this value
accordingly.
Expected contents:
<code>rope_type</code> (<code>str</code>):
The sub-variant of RoPE to use. Can be one of [&#x2018;default&#x2019;, &#x2018;linear&#x2019;, &#x2018;dynamic&#x2019;, &#x2018;yarn&#x2019;, &#x2018;longrope&#x2019;,
&#x2018;llama3&#x2019;], with &#x2018;default&#x2019; being the original RoPE implementation.
<code>factor</code> (<code>float</code>, <em>optional</em>):
Used with all rope types except &#x2018;default&#x2019;. The scaling factor to apply to the RoPE embeddings. In
most scaling types, a <code>factor</code> of x will enable the model to handle sequences of length x <em>
original maximum pre-trained length.
<code>original_max_position_embeddings</code> (<code>int</code>, </em>optional<em>):
Used with &#x2018;dynamic&#x2019;, &#x2018;longrope&#x2019; and &#x2018;llama3&#x2019;. The original max position embeddings used during
pretraining.
<code>attention_factor</code> (<code>float</code>, </em>optional<em>):
Used with &#x2018;yarn&#x2019; and &#x2018;longrope&#x2019;. The scaling factor to be applied on the attention
computation. If unspecified, it defaults to value recommended by the implementation, using the
<code>factor</code> field to infer the suggested value.
<code>beta_fast</code> (<code>float</code>, </em>optional<em>):
Only used with &#x2018;yarn&#x2019;. Parameter to set the boundary for extrapolation (only) in the linear
ramp function. If unspecified, it defaults to 32.
<code>beta_slow</code> (<code>float</code>, </em>optional<em>):
Only used with &#x2018;yarn&#x2019;. Parameter to set the boundary for interpolation (only) in the linear
ramp function. If unspecified, it defaults to 1.
<code>short_factor</code> (<code>list[float]</code>, </em>optional<em>):
Only used with &#x2018;longrope&#x2019;. The scaling factor to be applied to short contexts (&lt;
<code>original_max_position_embeddings</code>). Must be a list of numbers with the same length as the hidden
size divided by the number of attention heads divided by 2
<code>long_factor</code> (<code>list[float]</code>, </em>optional<em>):
Only used with &#x2018;longrope&#x2019;. The scaling factor to be applied to long contexts (&lt;
<code>original_max_position_embeddings</code>). Must be a list of numbers with the same length as the hidden
size divided by the number of attention heads divided by 2
<code>low_freq_factor</code> (<code>float</code>, </em>optional<em>):
Only used with &#x2018;llama3&#x2019;. Scaling factor applied to low frequency components of the RoPE
<code>high_freq_factor</code> (<code>float</code>, </em>optional*):
Only used with &#x2018;llama3&#x2019;. Scaling factor applied to high frequency components of the RoPE`,name:"rope_scaling"},{anchor:"transformers.FalconConfig.bos_token_id",description:`<strong>bos_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 11) &#x2014;
The id of the &#x201C;beginning-of-sequence&#x201D; token.`,name:"bos_token_id"},{anchor:"transformers.FalconConfig.eos_token_id",description:`<strong>eos_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 11) &#x2014;
The id of the &#x201C;end-of-sequence&#x201D; token.`,name:"eos_token_id"},{anchor:"transformers.FalconConfig.ffn_hidden_size",description:`<strong>ffn_hidden_size</strong> (<code>int</code>, <em>optional</em>) &#x2014;
The hidden size of the feedforward layer in the Transformer decoder.
defaults to 4x hidden dim`,name:"ffn_hidden_size"},{anchor:"transformers.FalconConfig.activation",description:`<strong>activation</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;gelu&quot;</code>) &#x2014;
The activation function used in the feedforward layer.`,name:"activation"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/falcon/configuration_falcon.py#L24"}}),te=new ft({props:{anchor:"transformers.FalconConfig.example",$$slots:{default:[vo]},$$scope:{ctx:v}}}),Fe=new ue({props:{title:"FalconModel",local:"transformers.FalconModel",headingTag:"h2"}}),$e=new A({props:{name:"class transformers.FalconModel",anchor:"transformers.FalconModel",parameters:[{name:"config",val:": FalconConfig"}],parametersDescription:[{anchor:"transformers.FalconModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/falcon#transformers.FalconConfig">FalconConfig</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/falcon/modeling_falcon.py#L682"}}),Ce=new A({props:{name:"forward",anchor:"transformers.FalconModel.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Union[transformers.cache_utils.Cache, tuple[tuple[torch.Tensor, torch.Tensor], ...], NoneType] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.LongTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"}],parametersDescription:[{anchor:"transformers.FalconModel.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, input_ids_length)</code>) &#x2014;
<code>input_ids_length</code> = <code>sequence_length</code> if <code>past_key_values</code> is <code>None</code> else <code>past_key_values.get_seq_length()</code>
(<code>sequence_length</code> of input past key value states). Indices of input sequence tokens in the vocabulary.</p>
<p>If <code>past_key_values</code> is used, only <code>input_ids</code> that do not have their past calculated should be passed as
<code>input_ids</code>.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.FalconModel.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>Union[~cache_utils.Cache, tuple[tuple[torch.Tensor, torch.Tensor], ...], NoneType]</code>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.FalconModel.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.FalconModel.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.FalconModel.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.LongTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.FalconModel.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.FalconModel.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.FalconModel.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.FalconModel.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.FalconModel.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.FalconModel.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/falcon/modeling_falcon.py#L714",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPastAndCrossAttentions"
>transformers.modeling_outputs.BaseModelOutputWithPastAndCrossAttentions</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/falcon#transformers.FalconConfig"
>FalconConfig</a>) and inputs.</p>
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
`}}),ne=new ht({props:{$$slots:{default:[ko]},$$scope:{ctx:v}}}),Je=new ue({props:{title:"FalconForCausalLM",local:"transformers.FalconForCausalLM",headingTag:"h2"}}),je=new A({props:{name:"class transformers.FalconForCausalLM",anchor:"transformers.FalconForCausalLM",parameters:[{name:"config",val:": FalconConfig"}],parametersDescription:[{anchor:"transformers.FalconForCausalLM.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/falcon#transformers.FalconConfig">FalconConfig</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/falcon/modeling_falcon.py#L996"}}),xe=new A({props:{name:"forward",anchor:"transformers.FalconForCausalLM.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Union[transformers.cache_utils.Cache, tuple[tuple[torch.Tensor, torch.Tensor], ...], NoneType] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"labels",val:": typing.Optional[torch.Tensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"},{name:"logits_to_keep",val:": typing.Union[int, torch.Tensor] = 0"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.FalconForCausalLM.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, input_ids_length)</code>) &#x2014;
<code>input_ids_length</code> = <code>sequence_length</code> if <code>past_key_values</code> is <code>None</code> else <code>past_key_values.get_seq_length()</code>
(<code>sequence_length</code> of input past key value states). Indices of input sequence tokens in the vocabulary.</p>
<p>If <code>past_key_values</code> is used, only <code>input_ids</code> that do not have their past calculated should be passed as
<code>input_ids</code>.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.FalconForCausalLM.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>Union[~cache_utils.Cache, tuple[tuple[torch.Tensor, torch.Tensor], ...], NoneType]</code>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.FalconForCausalLM.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.FalconForCausalLM.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.FalconForCausalLM.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.FalconForCausalLM.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.FalconForCausalLM.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for language modeling. Note that the labels <strong>are shifted</strong> inside the model, i.e. you can set
<code>labels = input_ids</code> Indices are selected in <code>[-100, 0, ..., config.vocab_size]</code> All labels set to <code>-100</code>
are ignored (masked), the loss is only computed for labels in <code>[0, ..., config.vocab_size]</code>`,name:"labels"},{anchor:"transformers.FalconForCausalLM.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.FalconForCausalLM.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.FalconForCausalLM.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.FalconForCausalLM.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.FalconForCausalLM.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"},{anchor:"transformers.FalconForCausalLM.forward.logits_to_keep",description:`<strong>logits_to_keep</strong> (<code>Union[int, torch.Tensor]</code>, defaults to <code>0</code>) &#x2014;
If an <code>int</code>, compute logits for the last <code>logits_to_keep</code> tokens. If <code>0</code>, calculate logits for all
<code>input_ids</code> (special case). Only last token logits are needed for generation, and calculating them only for that
token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
If a <code>torch.Tensor</code>, must be 1D corresponding to the indices to keep in the sequence length dimension.
This is useful when using packed tensor format (single dimension for batch and sequence length).`,name:"logits_to_keep"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/falcon/modeling_falcon.py#L1010",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithCrossAttentions"
>transformers.modeling_outputs.CausalLMOutputWithCrossAttentions</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/falcon#transformers.FalconConfig"
>FalconConfig</a>) and inputs.</p>
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
`}}),oe=new ht({props:{$$slots:{default:[Fo]},$$scope:{ctx:v}}}),se=new ft({props:{anchor:"transformers.FalconForCausalLM.forward.example",$$slots:{default:[$o]},$$scope:{ctx:v}}}),Ue=new ue({props:{title:"FalconForSequenceClassification",local:"transformers.FalconForSequenceClassification",headingTag:"h2"}}),ze=new A({props:{name:"class transformers.FalconForSequenceClassification",anchor:"transformers.FalconForSequenceClassification",parameters:[{name:"config",val:": FalconConfig"}],parametersDescription:[{anchor:"transformers.FalconForSequenceClassification.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/falcon#transformers.FalconConfig">FalconConfig</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/falcon/modeling_falcon.py#L1102"}}),We=new A({props:{name:"forward",anchor:"transformers.FalconForSequenceClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[tuple[tuple[torch.Tensor, torch.Tensor], ...]] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"labels",val:": typing.Optional[torch.Tensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.FalconForSequenceClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, input_ids_length)</code>) &#x2014;
<code>input_ids_length</code> = <code>sequence_length</code> if <code>past_key_values</code> is <code>None</code> else <code>past_key_values.get_seq_length()</code>
(<code>sequence_length</code> of input past key value states). Indices of input sequence tokens in the vocabulary.</p>
<p>If <code>past_key_values</code> is used, only <code>input_ids</code> that do not have their past calculated should be passed as
<code>input_ids</code>.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.FalconForSequenceClassification.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>tuple[tuple[torch.Tensor, torch.Tensor, ...]]</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.FalconForSequenceClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.FalconForSequenceClassification.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.FalconForSequenceClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.FalconForSequenceClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the sequence classification/regression loss. Indices should be in <code>[0, ..., config.num_labels - 1]</code>. If <code>config.num_labels == 1</code> a regression loss is computed (Mean-Square loss), If
<code>config.num_labels &gt; 1</code> a classification loss is computed (Cross-Entropy).`,name:"labels"},{anchor:"transformers.FalconForSequenceClassification.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.FalconForSequenceClassification.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.FalconForSequenceClassification.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.FalconForSequenceClassification.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/falcon/modeling_falcon.py#L1112",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <code>transformers.modeling_outputs.SequenceClassifierOutputWithPast</code> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/falcon#transformers.FalconConfig"
>FalconConfig</a>) and inputs.</p>
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
`}}),ae=new ht({props:{$$slots:{default:[Co]},$$scope:{ctx:v}}}),re=new ft({props:{anchor:"transformers.FalconForSequenceClassification.forward.example",$$slots:{default:[Jo]},$$scope:{ctx:v}}}),le=new ft({props:{anchor:"transformers.FalconForSequenceClassification.forward.example-2",$$slots:{default:[jo]},$$scope:{ctx:v}}}),Ze=new ue({props:{title:"FalconForTokenClassification",local:"transformers.FalconForTokenClassification",headingTag:"h2"}}),Ie=new A({props:{name:"class transformers.FalconForTokenClassification",anchor:"transformers.FalconForTokenClassification",parameters:[{name:"config",val:": FalconConfig"}],parametersDescription:[{anchor:"transformers.FalconForTokenClassification.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/falcon#transformers.FalconConfig">FalconConfig</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/falcon/modeling_falcon.py#L1220"}}),Ne=new A({props:{name:"forward",anchor:"transformers.FalconForTokenClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[tuple[tuple[torch.Tensor, torch.Tensor], ...]] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"labels",val:": typing.Optional[torch.Tensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.FalconForTokenClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, input_ids_length)</code>) &#x2014;
<code>input_ids_length</code> = <code>sequence_length</code> if <code>past_key_values</code> is <code>None</code> else <code>past_key_values.get_seq_length()</code>
(<code>sequence_length</code> of input past key value states). Indices of input sequence tokens in the vocabulary.</p>
<p>If <code>past_key_values</code> is used, only <code>input_ids</code> that do not have their past calculated should be passed as
<code>input_ids</code>.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.FalconForTokenClassification.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>tuple[tuple[torch.Tensor, torch.Tensor, ...]]</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.FalconForTokenClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.FalconForTokenClassification.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.FalconForTokenClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.FalconForTokenClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the sequence classification/regression loss. Indices should be in <code>[0, ..., config.num_labels - 1]</code>. If <code>config.num_labels == 1</code> a regression loss is computed (Mean-Square loss), If
<code>config.num_labels &gt; 1</code> a classification loss is computed (Cross-Entropy).`,name:"labels"},{anchor:"transformers.FalconForTokenClassification.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.FalconForTokenClassification.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.FalconForTokenClassification.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.FalconForTokenClassification.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/falcon/modeling_falcon.py#L1238",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput"
>transformers.modeling_outputs.TokenClassifierOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/falcon#transformers.FalconConfig"
>FalconConfig</a>) and inputs.</p>
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
`}}),ie=new ht({props:{$$slots:{default:[xo]},$$scope:{ctx:v}}}),ce=new ft({props:{anchor:"transformers.FalconForTokenClassification.forward.example",$$slots:{default:[Uo]},$$scope:{ctx:v}}}),qe=new ue({props:{title:"FalconForQuestionAnswering",local:"transformers.FalconForQuestionAnswering",headingTag:"h2"}}),Ge=new A({props:{name:"class transformers.FalconForQuestionAnswering",anchor:"transformers.FalconForQuestionAnswering",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.FalconForQuestionAnswering.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/falcon#transformers.FalconForQuestionAnswering">FalconForQuestionAnswering</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/falcon/modeling_falcon.py#L1309"}}),Be=new A({props:{name:"forward",anchor:"transformers.FalconForQuestionAnswering.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"start_positions",val:": typing.Optional[torch.LongTensor] = None"},{name:"end_positions",val:": typing.Optional[torch.LongTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.FalconForQuestionAnswering.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, input_ids_length)</code>) &#x2014;
<code>input_ids_length</code> = <code>sequence_length</code> if <code>past_key_values</code> is <code>None</code> else <code>past_key_values.get_seq_length()</code>
(<code>sequence_length</code> of input past key value states). Indices of input sequence tokens in the vocabulary.</p>
<p>If <code>past_key_values</code> is used, only <code>input_ids</code> that do not have their past calculated should be passed as
<code>input_ids</code>.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.FalconForQuestionAnswering.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.FalconForQuestionAnswering.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.FalconForQuestionAnswering.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.FalconForQuestionAnswering.forward.start_positions",description:`<strong>start_positions</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for position (index) of the start of the labelled span for computing the token classification loss.
Positions are clamped to the length of the sequence (<code>sequence_length</code>). Position outside of the sequence
are not taken into account for computing the loss.`,name:"start_positions"},{anchor:"transformers.FalconForQuestionAnswering.forward.end_positions",description:`<strong>end_positions</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for position (index) of the end of the labelled span for computing the token classification loss.
Positions are clamped to the length of the sequence (<code>sequence_length</code>). Position outside of the sequence
are not taken into account for computing the loss.`,name:"end_positions"},{anchor:"transformers.FalconForQuestionAnswering.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.FalconForQuestionAnswering.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.FalconForQuestionAnswering.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/falcon/modeling_falcon.py#L1318",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.QuestionAnsweringModelOutput"
>transformers.modeling_outputs.QuestionAnsweringModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/falcon#transformers.FalconConfig"
>FalconConfig</a>) and inputs.</p>
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
`}}),de=new ht({props:{$$slots:{default:[zo]},$$scope:{ctx:v}}}),pe=new ft({props:{anchor:"transformers.FalconForQuestionAnswering.forward.example",$$slots:{default:[Wo]},$$scope:{ctx:v}}}),Re=new go({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/falcon.md"}}),{c(){t=u("meta"),p=r(),n=u("p"),i=r(),T=u("p"),T.innerHTML=o,h=r(),k=u("div"),k.innerHTML=gt,me=r(),f(P.$$.fragment),yt=r(),he=u("p"),he.innerHTML=Wn,Mt=r(),fe=u("p"),fe.innerHTML=Zn,Tt=r(),f(K.$$.fragment),wt=r(),ge=u("p"),ge.innerHTML=In,vt=r(),f(ee.$$.fragment),kt=r(),_e=u("p"),_e.innerHTML=Nn,Ft=r(),be=u("p"),be.innerHTML=qn,$t=r(),f(ye.$$.fragment),Ct=r(),f(Me.$$.fragment),Jt=r(),He=u("ul"),Te=u("li"),Xe=u("p"),Xe.innerHTML=Gn,Lt=r(),f(we.$$.fragment),jt=r(),f(ve.$$.fragment),xt=r(),Z=u("div"),f(ke.$$.fragment),Et=r(),Le=u("p"),Le.innerHTML=Bn,St=r(),Ee=u("p"),Ee.innerHTML=Rn,Qt=r(),f(te.$$.fragment),Ut=r(),f(Fe.$$.fragment),zt=r(),C=u("div"),f($e.$$.fragment),At=r(),Se=u("p"),Se.textContent=Vn,Yt=r(),Qe=u("p"),Qe.innerHTML=Hn,Pt=r(),Ae=u("p"),Ae.innerHTML=Xn,Ot=r(),O=u("div"),f(Ce.$$.fragment),Dt=r(),Ye=u("p"),Ye.innerHTML=Ln,Kt=r(),f(ne.$$.fragment),Wt=r(),f(Je.$$.fragment),Zt=r(),J=u("div"),f(je.$$.fragment),en=r(),Pe=u("p"),Pe.textContent=En,tn=r(),Oe=u("p"),Oe.innerHTML=Sn,nn=r(),De=u("p"),De.innerHTML=Qn,on=r(),V=u("div"),f(xe.$$.fragment),sn=r(),Ke=u("p"),Ke.innerHTML=An,an=r(),f(oe.$$.fragment),rn=r(),f(se.$$.fragment),It=r(),f(Ue.$$.fragment),Nt=r(),F=u("div"),f(ze.$$.fragment),ln=r(),et=u("p"),et.textContent=Yn,cn=r(),tt=u("p"),tt.innerHTML=Pn,dn=r(),nt=u("p"),nt.innerHTML=On,pn=r(),ot=u("p"),ot.innerHTML=Dn,un=r(),st=u("p"),st.innerHTML=Kn,mn=r(),W=u("div"),f(We.$$.fragment),hn=r(),at=u("p"),at.innerHTML=eo,fn=r(),f(ae.$$.fragment),gn=r(),f(re.$$.fragment),_n=r(),f(le.$$.fragment),qt=r(),f(Ze.$$.fragment),Gt=r(),j=u("div"),f(Ie.$$.fragment),bn=r(),rt=u("p"),rt.textContent=to,yn=r(),lt=u("p"),lt.innerHTML=no,Mn=r(),it=u("p"),it.innerHTML=oo,Tn=r(),H=u("div"),f(Ne.$$.fragment),wn=r(),ct=u("p"),ct.innerHTML=so,vn=r(),f(ie.$$.fragment),kn=r(),f(ce.$$.fragment),Bt=r(),f(qe.$$.fragment),Rt=r(),x=u("div"),f(Ge.$$.fragment),Fn=r(),dt=u("p"),dt.innerHTML=ao,$n=r(),pt=u("p"),pt.innerHTML=ro,Cn=r(),ut=u("p"),ut.innerHTML=lo,Jn=r(),X=u("div"),f(Be.$$.fragment),jn=r(),mt=u("p"),mt.innerHTML=io,xn=r(),f(de.$$.fragment),Un=r(),f(pe.$$.fragment),Vt=r(),f(Re.$$.fragment),Ht=r(),_t=u("p"),this.h()},l(e){const s=ho("svelte-u9bgzb",document.head);t=m(s,"META",{name:!0,content:!0}),s.forEach(a),p=l(e),n=m(e,"P",{}),U(n).forEach(a),i=l(e),T=m(e,"P",{"data-svelte-h":!0}),w(T)!=="svelte-x0xgkb"&&(T.innerHTML=o),h=l(e),k=m(e,"DIV",{style:!0,"data-svelte-h":!0}),w(k)!=="svelte-2m0t7r"&&(k.innerHTML=gt),me=l(e),g(P.$$.fragment,e),yt=l(e),he=m(e,"P",{"data-svelte-h":!0}),w(he)!=="svelte-a90509"&&(he.innerHTML=Wn),Mt=l(e),fe=m(e,"P",{"data-svelte-h":!0}),w(fe)!=="svelte-scuz6e"&&(fe.innerHTML=Zn),Tt=l(e),g(K.$$.fragment,e),wt=l(e),ge=m(e,"P",{"data-svelte-h":!0}),w(ge)!=="svelte-17pa8jt"&&(ge.innerHTML=In),vt=l(e),g(ee.$$.fragment,e),kt=l(e),_e=m(e,"P",{"data-svelte-h":!0}),w(_e)!=="svelte-nf5ooi"&&(_e.innerHTML=Nn),Ft=l(e),be=m(e,"P",{"data-svelte-h":!0}),w(be)!=="svelte-60nsd0"&&(be.innerHTML=qn),$t=l(e),g(ye.$$.fragment,e),Ct=l(e),g(Me.$$.fragment,e),Jt=l(e),He=m(e,"UL",{});var bt=U(He);Te=m(bt,"LI",{});var Ve=U(Te);Xe=m(Ve,"P",{"data-svelte-h":!0}),w(Xe)!=="svelte-m7g830"&&(Xe.innerHTML=Gn),Lt=l(Ve),g(we.$$.fragment,Ve),Ve.forEach(a),bt.forEach(a),jt=l(e),g(ve.$$.fragment,e),xt=l(e),Z=m(e,"DIV",{class:!0});var L=U(Z);g(ke.$$.fragment,L),Et=l(L),Le=m(L,"P",{"data-svelte-h":!0}),w(Le)!=="svelte-1on4nmp"&&(Le.innerHTML=Bn),St=l(L),Ee=m(L,"P",{"data-svelte-h":!0}),w(Ee)!=="svelte-1ek1ss9"&&(Ee.innerHTML=Rn),Qt=l(L),g(te.$$.fragment,L),L.forEach(a),Ut=l(e),g(Fe.$$.fragment,e),zt=l(e),C=m(e,"DIV",{class:!0});var I=U(C);g($e.$$.fragment,I),At=l(I),Se=m(I,"P",{"data-svelte-h":!0}),w(Se)!=="svelte-cqka21"&&(Se.textContent=Vn),Yt=l(I),Qe=m(I,"P",{"data-svelte-h":!0}),w(Qe)!=="svelte-q52n56"&&(Qe.innerHTML=Hn),Pt=l(I),Ae=m(I,"P",{"data-svelte-h":!0}),w(Ae)!=="svelte-hswkmf"&&(Ae.innerHTML=Xn),Ot=l(I),O=m(I,"DIV",{class:!0});var D=U(O);g(Ce.$$.fragment,D),Dt=l(D),Ye=m(D,"P",{"data-svelte-h":!0}),w(Ye)!=="svelte-fzgsru"&&(Ye.innerHTML=Ln),Kt=l(D),g(ne.$$.fragment,D),D.forEach(a),I.forEach(a),Wt=l(e),g(Je.$$.fragment,e),Zt=l(e),J=m(e,"DIV",{class:!0});var N=U(J);g(je.$$.fragment,N),en=l(N),Pe=m(N,"P",{"data-svelte-h":!0}),w(Pe)!=="svelte-88v1u6"&&(Pe.textContent=En),tn=l(N),Oe=m(N,"P",{"data-svelte-h":!0}),w(Oe)!=="svelte-q52n56"&&(Oe.innerHTML=Sn),nn=l(N),De=m(N,"P",{"data-svelte-h":!0}),w(De)!=="svelte-hswkmf"&&(De.innerHTML=Qn),on=l(N),V=m(N,"DIV",{class:!0});var E=U(V);g(xe.$$.fragment,E),sn=l(E),Ke=m(E,"P",{"data-svelte-h":!0}),w(Ke)!=="svelte-16f28lu"&&(Ke.innerHTML=An),an=l(E),g(oe.$$.fragment,E),rn=l(E),g(se.$$.fragment,E),E.forEach(a),N.forEach(a),It=l(e),g(Ue.$$.fragment,e),Nt=l(e),F=m(e,"DIV",{class:!0});var $=U(F);g(ze.$$.fragment,$),ln=l($),et=m($,"P",{"data-svelte-h":!0}),w(et)!=="svelte-n3dntd"&&(et.textContent=Yn),cn=l($),tt=m($,"P",{"data-svelte-h":!0}),w(tt)!=="svelte-ioflas"&&(tt.innerHTML=Pn),dn=l($),nt=m($,"P",{"data-svelte-h":!0}),w(nt)!=="svelte-10ugs3m"&&(nt.innerHTML=On),pn=l($),ot=m($,"P",{"data-svelte-h":!0}),w(ot)!=="svelte-q52n56"&&(ot.innerHTML=Dn),un=l($),st=m($,"P",{"data-svelte-h":!0}),w(st)!=="svelte-hswkmf"&&(st.innerHTML=Kn),mn=l($),W=m($,"DIV",{class:!0});var q=U(W);g(We.$$.fragment,q),hn=l(q),at=m(q,"P",{"data-svelte-h":!0}),w(at)!=="svelte-jbzeq2"&&(at.innerHTML=eo),fn=l(q),g(ae.$$.fragment,q),gn=l(q),g(re.$$.fragment,q),_n=l(q),g(le.$$.fragment,q),q.forEach(a),$.forEach(a),qt=l(e),g(Ze.$$.fragment,e),Gt=l(e),j=m(e,"DIV",{class:!0});var G=U(j);g(Ie.$$.fragment,G),bn=l(G),rt=m(G,"P",{"data-svelte-h":!0}),w(rt)!=="svelte-4smgpr"&&(rt.textContent=to),yn=l(G),lt=m(G,"P",{"data-svelte-h":!0}),w(lt)!=="svelte-q52n56"&&(lt.innerHTML=no),Mn=l(G),it=m(G,"P",{"data-svelte-h":!0}),w(it)!=="svelte-hswkmf"&&(it.innerHTML=oo),Tn=l(G),H=m(G,"DIV",{class:!0});var S=U(H);g(Ne.$$.fragment,S),wn=l(S),ct=m(S,"P",{"data-svelte-h":!0}),w(ct)!=="svelte-l6ehs0"&&(ct.innerHTML=so),vn=l(S),g(ie.$$.fragment,S),kn=l(S),g(ce.$$.fragment,S),S.forEach(a),G.forEach(a),Bt=l(e),g(qe.$$.fragment,e),Rt=l(e),x=m(e,"DIV",{class:!0});var B=U(x);g(Ge.$$.fragment,B),Fn=l(B),dt=m(B,"P",{"data-svelte-h":!0}),w(dt)!=="svelte-1pfjogu"&&(dt.innerHTML=ao),$n=l(B),pt=m(B,"P",{"data-svelte-h":!0}),w(pt)!=="svelte-q52n56"&&(pt.innerHTML=ro),Cn=l(B),ut=m(B,"P",{"data-svelte-h":!0}),w(ut)!=="svelte-hswkmf"&&(ut.innerHTML=lo),Jn=l(B),X=m(B,"DIV",{class:!0});var Q=U(X);g(Be.$$.fragment,Q),jn=l(Q),mt=m(Q,"P",{"data-svelte-h":!0}),w(mt)!=="svelte-1yz1lpy"&&(mt.innerHTML=io),xn=l(Q),g(de.$$.fragment,Q),Un=l(Q),g(pe.$$.fragment,Q),Q.forEach(a),B.forEach(a),Vt=l(e),g(Re.$$.fragment,e),Ht=l(e),_t=m(e,"P",{}),U(_t).forEach(a),this.h()},h(){R(t,"name","hf:doc:metadata"),R(t,"content",Io),fo(k,"float","right"),R(Z,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),R(O,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),R(C,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),R(V,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),R(J,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),R(W,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),R(F,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),R(H,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),R(j,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),R(X,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),R(x,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,s){c(document.head,t),d(e,p,s),d(e,n,s),d(e,i,s),d(e,T,s),d(e,h,s),d(e,k,s),d(e,me,s),_(P,e,s),d(e,yt,s),d(e,he,s),d(e,Mt,s),d(e,fe,s),d(e,Tt,s),_(K,e,s),d(e,wt,s),d(e,ge,s),d(e,vt,s),_(ee,e,s),d(e,kt,s),d(e,_e,s),d(e,Ft,s),d(e,be,s),d(e,$t,s),_(ye,e,s),d(e,Ct,s),_(Me,e,s),d(e,Jt,s),d(e,He,s),c(He,Te),c(Te,Xe),c(Te,Lt),_(we,Te,null),d(e,jt,s),_(ve,e,s),d(e,xt,s),d(e,Z,s),_(ke,Z,null),c(Z,Et),c(Z,Le),c(Z,St),c(Z,Ee),c(Z,Qt),_(te,Z,null),d(e,Ut,s),_(Fe,e,s),d(e,zt,s),d(e,C,s),_($e,C,null),c(C,At),c(C,Se),c(C,Yt),c(C,Qe),c(C,Pt),c(C,Ae),c(C,Ot),c(C,O),_(Ce,O,null),c(O,Dt),c(O,Ye),c(O,Kt),_(ne,O,null),d(e,Wt,s),_(Je,e,s),d(e,Zt,s),d(e,J,s),_(je,J,null),c(J,en),c(J,Pe),c(J,tn),c(J,Oe),c(J,nn),c(J,De),c(J,on),c(J,V),_(xe,V,null),c(V,sn),c(V,Ke),c(V,an),_(oe,V,null),c(V,rn),_(se,V,null),d(e,It,s),_(Ue,e,s),d(e,Nt,s),d(e,F,s),_(ze,F,null),c(F,ln),c(F,et),c(F,cn),c(F,tt),c(F,dn),c(F,nt),c(F,pn),c(F,ot),c(F,un),c(F,st),c(F,mn),c(F,W),_(We,W,null),c(W,hn),c(W,at),c(W,fn),_(ae,W,null),c(W,gn),_(re,W,null),c(W,_n),_(le,W,null),d(e,qt,s),_(Ze,e,s),d(e,Gt,s),d(e,j,s),_(Ie,j,null),c(j,bn),c(j,rt),c(j,yn),c(j,lt),c(j,Mn),c(j,it),c(j,Tn),c(j,H),_(Ne,H,null),c(H,wn),c(H,ct),c(H,vn),_(ie,H,null),c(H,kn),_(ce,H,null),d(e,Bt,s),_(qe,e,s),d(e,Rt,s),d(e,x,s),_(Ge,x,null),c(x,Fn),c(x,dt),c(x,$n),c(x,pt),c(x,Cn),c(x,ut),c(x,Jn),c(x,X),_(Be,X,null),c(X,jn),c(X,mt),c(X,xn),_(de,X,null),c(X,Un),_(pe,X,null),d(e,Vt,s),_(Re,e,s),d(e,Ht,s),d(e,_t,s),Xt=!0},p(e,[s]){const bt={};s&2&&(bt.$$scope={dirty:s,ctx:e}),K.$set(bt);const Ve={};s&2&&(Ve.$$scope={dirty:s,ctx:e}),ee.$set(Ve);const L={};s&2&&(L.$$scope={dirty:s,ctx:e}),te.$set(L);const I={};s&2&&(I.$$scope={dirty:s,ctx:e}),ne.$set(I);const D={};s&2&&(D.$$scope={dirty:s,ctx:e}),oe.$set(D);const N={};s&2&&(N.$$scope={dirty:s,ctx:e}),se.$set(N);const E={};s&2&&(E.$$scope={dirty:s,ctx:e}),ae.$set(E);const $={};s&2&&($.$$scope={dirty:s,ctx:e}),re.$set($);const q={};s&2&&(q.$$scope={dirty:s,ctx:e}),le.$set(q);const G={};s&2&&(G.$$scope={dirty:s,ctx:e}),ie.$set(G);const S={};s&2&&(S.$$scope={dirty:s,ctx:e}),ce.$set(S);const B={};s&2&&(B.$$scope={dirty:s,ctx:e}),de.$set(B);const Q={};s&2&&(Q.$$scope={dirty:s,ctx:e}),pe.$set(Q)},i(e){Xt||(b(P.$$.fragment,e),b(K.$$.fragment,e),b(ee.$$.fragment,e),b(ye.$$.fragment,e),b(Me.$$.fragment,e),b(we.$$.fragment,e),b(ve.$$.fragment,e),b(ke.$$.fragment,e),b(te.$$.fragment,e),b(Fe.$$.fragment,e),b($e.$$.fragment,e),b(Ce.$$.fragment,e),b(ne.$$.fragment,e),b(Je.$$.fragment,e),b(je.$$.fragment,e),b(xe.$$.fragment,e),b(oe.$$.fragment,e),b(se.$$.fragment,e),b(Ue.$$.fragment,e),b(ze.$$.fragment,e),b(We.$$.fragment,e),b(ae.$$.fragment,e),b(re.$$.fragment,e),b(le.$$.fragment,e),b(Ze.$$.fragment,e),b(Ie.$$.fragment,e),b(Ne.$$.fragment,e),b(ie.$$.fragment,e),b(ce.$$.fragment,e),b(qe.$$.fragment,e),b(Ge.$$.fragment,e),b(Be.$$.fragment,e),b(de.$$.fragment,e),b(pe.$$.fragment,e),b(Re.$$.fragment,e),Xt=!0)},o(e){y(P.$$.fragment,e),y(K.$$.fragment,e),y(ee.$$.fragment,e),y(ye.$$.fragment,e),y(Me.$$.fragment,e),y(we.$$.fragment,e),y(ve.$$.fragment,e),y(ke.$$.fragment,e),y(te.$$.fragment,e),y(Fe.$$.fragment,e),y($e.$$.fragment,e),y(Ce.$$.fragment,e),y(ne.$$.fragment,e),y(Je.$$.fragment,e),y(je.$$.fragment,e),y(xe.$$.fragment,e),y(oe.$$.fragment,e),y(se.$$.fragment,e),y(Ue.$$.fragment,e),y(ze.$$.fragment,e),y(We.$$.fragment,e),y(ae.$$.fragment,e),y(re.$$.fragment,e),y(le.$$.fragment,e),y(Ze.$$.fragment,e),y(Ie.$$.fragment,e),y(Ne.$$.fragment,e),y(ie.$$.fragment,e),y(ce.$$.fragment,e),y(qe.$$.fragment,e),y(Ge.$$.fragment,e),y(Be.$$.fragment,e),y(de.$$.fragment,e),y(pe.$$.fragment,e),y(Re.$$.fragment,e),Xt=!1},d(e){e&&(a(p),a(n),a(i),a(T),a(h),a(k),a(me),a(yt),a(he),a(Mt),a(fe),a(Tt),a(wt),a(ge),a(vt),a(kt),a(_e),a(Ft),a(be),a($t),a(Ct),a(Jt),a(He),a(jt),a(xt),a(Z),a(Ut),a(zt),a(C),a(Wt),a(Zt),a(J),a(It),a(Nt),a(F),a(qt),a(Gt),a(j),a(Bt),a(Rt),a(x),a(Vt),a(Ht),a(_t)),a(t),M(P,e),M(K,e),M(ee,e),M(ye,e),M(Me,e),M(we),M(ve,e),M(ke),M(te),M(Fe,e),M($e),M(Ce),M(ne),M(Je,e),M(je),M(xe),M(oe),M(se),M(Ue,e),M(ze),M(We),M(ae),M(re),M(le),M(Ze,e),M(Ie),M(Ne),M(ie),M(ce),M(qe,e),M(Ge),M(Be),M(de),M(pe),M(Re,e)}}}const Io='{"title":"Falcon","local":"falcon","sections":[{"title":"Notes","local":"notes","sections":[],"depth":2},{"title":"FalconConfig","local":"transformers.FalconConfig","sections":[],"depth":2},{"title":"FalconModel","local":"transformers.FalconModel","sections":[],"depth":2},{"title":"FalconForCausalLM","local":"transformers.FalconForCausalLM","sections":[],"depth":2},{"title":"FalconForSequenceClassification","local":"transformers.FalconForSequenceClassification","sections":[],"depth":2},{"title":"FalconForTokenClassification","local":"transformers.FalconForTokenClassification","sections":[],"depth":2},{"title":"FalconForQuestionAnswering","local":"transformers.FalconForQuestionAnswering","sections":[],"depth":2}],"depth":1}';function No(v){return po(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class Eo extends uo{constructor(t){super(),mo(this,t,No,Zo,co,{})}}export{Eo as component};
