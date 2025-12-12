import{s as an,o as rn,n as x}from"../chunks/scheduler.18a86fab.js";import{S as ln,i as dn,g as u,s as r,r as g,A as cn,h as m,f as a,c as i,j as B,x as w,u as _,k as q,l as pn,y as d,a as c,v as T,d as y,t as b,w as M}from"../chunks/index.98837b22.js";import{T as ct}from"../chunks/Tip.77304350.js";import{D as S}from"../chunks/Docstring.a1ef7999.js";import{C as O}from"../chunks/CodeBlock.8d0c2e8a.js";import{E as pt}from"../chunks/ExampleCodeBlock.8c3ee1f9.js";import{H as ue,E as un}from"../chunks/getInferenceSnippets.06c2775f.js";import{H as mn,a as Jo}from"../chunks/HfOption.6641485e.js";function hn(v){let t,p="Click on the GPT-Neo models in the right sidebar for more examples of how to apply GPT Neo to different language tasks.";return{c(){t=u("p"),t.textContent=p},l(o){t=m(o,"P",{"data-svelte-h":!0}),w(t)!=="svelte-jz6kw8"&&(t.textContent=p)},m(o,l){c(o,t,l)},p:x,d(o){o&&a(t)}}}function fn(v){let t,p;return t=new O({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwcGlwZWxpbmUlMEElMEFwaXBlbGluZSUyMCUzRCUyMHBpcGVsaW5lKHRhc2slM0QlMjJ0ZXh0LWdlbmVyYXRpb24lMjIlMkMlMjBtb2RlbCUzRCUyMkVsZXV0aGVyQUklMkZncHQtbmVvLTEuM0IlMjIlMkMlMjBkdHlwZSUzRHRvcmNoLmZsb2F0MTYlMkMlMjBkZXZpY2UlM0QwKSUwQXBpcGVsaW5lKCUyMkhlbGxvJTJDJTIwSSdtJTIwYSUyMGxhbmd1YWdlJTIwbW9kZWwlMjIp",highlighted:`<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> pipeline

pipeline = pipeline(task=<span class="hljs-string">&quot;text-generation&quot;</span>, model=<span class="hljs-string">&quot;EleutherAI/gpt-neo-1.3B&quot;</span>, dtype=torch.float16, device=<span class="hljs-number">0</span>)
pipeline(<span class="hljs-string">&quot;Hello, I&#x27;m a language model&quot;</span>)`,wrap:!1}}),{c(){g(t.$$.fragment)},l(o){_(t.$$.fragment,o)},m(o,l){T(t,o,l),p=!0},p:x,i(o){p||(y(t.$$.fragment,o),p=!0)},o(o){b(t.$$.fragment,o),p=!1},d(o){M(t,o)}}}function gn(v){let t,p;return t=new O({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b01vZGVsRm9yQ2F1c2FsTE0lMkMlMjBBdXRvVG9rZW5pemVyJTBBJTBBbW9kZWwlMjAlM0QlMjBBdXRvTW9kZWxGb3JDYXVzYWxMTS5mcm9tX3ByZXRyYWluZWQoJTIyRWxldXRoZXJBSSUyRmdwdC1uZW8tMS4zQiUyMiUyQyUyMGR0eXBlJTNEdG9yY2guZmxvYXQxNiUyQyUyMGRldmljZV9tYXAlM0QlMjJhdXRvJTIyJTJDJTIwYXR0bl9pbXBsZW1lbnRhdGlvbiUzRCUyMmZsYXNoX2F0dGVudGlvbl8yJTIyKSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMkVsZXV0aGVyQUklMkZncHQtbmVvLTEuM0IlMjIpJTBBJTBBaW5wdXRfaWRzJTIwJTNEJTIwdG9rZW5pemVyKCUyMkhlbGxvJTJDJTIwSSdtJTIwYSUyMGxhbmd1YWdlJTIwbW9kZWwlMjIlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKS50byhtb2RlbC5kZXZpY2UpJTBBJTBBb3V0cHV0JTIwJTNEJTIwbW9kZWwuZ2VuZXJhdGUoKippbnB1dF9pZHMpJTBBcHJpbnQodG9rZW5pemVyLmRlY29kZShvdXRwdXQlNUIwJTVEJTJDJTIwc2tpcF9zcGVjaWFsX3Rva2VucyUzRFRydWUpKQ==",highlighted:`<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(<span class="hljs-string">&quot;EleutherAI/gpt-neo-1.3B&quot;</span>, dtype=torch.float16, device_map=<span class="hljs-string">&quot;auto&quot;</span>, attn_implementation=<span class="hljs-string">&quot;flash_attention_2&quot;</span>)
tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;EleutherAI/gpt-neo-1.3B&quot;</span>)

input_ids = tokenizer(<span class="hljs-string">&quot;Hello, I&#x27;m a language model&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>).to(model.device)

output = model.generate(**input_ids)
<span class="hljs-built_in">print</span>(tokenizer.decode(output[<span class="hljs-number">0</span>], skip_special_tokens=<span class="hljs-literal">True</span>))`,wrap:!1}}),{c(){g(t.$$.fragment)},l(o){_(t.$$.fragment,o)},m(o,l){T(t,o,l),p=!0},p:x,i(o){p||(y(t.$$.fragment,o),p=!0)},o(o){b(t.$$.fragment,o),p=!1},d(o){M(t,o)}}}function _n(v){let t,p;return t=new O({props:{code:"ZWNobyUyMC1lJTIwJTIySGVsbG8lMkMlMjBJJ20lMjBhJTIwbGFuZ3VhZ2UlMjBtb2RlbCUyMiUyMCU3QyUyMHRyYW5zZm9ybWVycy1jbGklMjBydW4lMjAtLXRhc2slMjB0ZXh0LWdlbmVyYXRpb24lMjAtLW1vZGVsJTIwRWxldXRoZXJBSSUyRmdwdC1uZW8tMS4zQiUyMC0tZGV2aWNlJTIwMA==",highlighted:'<span class="hljs-built_in">echo</span> -e <span class="hljs-string">&quot;Hello, I&#x27;m a language model&quot;</span> | transformers-cli run --task text-generation --model EleutherAI/gpt-neo-1.3B --device 0',wrap:!1}}),{c(){g(t.$$.fragment)},l(o){_(t.$$.fragment,o)},m(o,l){T(t,o,l),p=!0},p:x,i(o){p||(y(t.$$.fragment,o),p=!0)},o(o){b(t.$$.fragment,o),p=!1},d(o){M(t,o)}}}function Tn(v){let t,p,o,l,f,n;return t=new Jo({props:{id:"usage",option:"Pipeline",$$slots:{default:[fn]},$$scope:{ctx:v}}}),o=new Jo({props:{id:"usage",option:"AutoModel",$$slots:{default:[gn]},$$scope:{ctx:v}}}),f=new Jo({props:{id:"usage",option:"transformers CLI",$$slots:{default:[_n]},$$scope:{ctx:v}}}),{c(){g(t.$$.fragment),p=r(),g(o.$$.fragment),l=r(),g(f.$$.fragment)},l(h){_(t.$$.fragment,h),p=i(h),_(o.$$.fragment,h),l=i(h),_(f.$$.fragment,h)},m(h,k){T(t,h,k),c(h,p,k),T(o,h,k),c(h,l,k),T(f,h,k),n=!0},p(h,k){const ut={};k&2&&(ut.$$scope={dirty:k,ctx:h}),t.$set(ut);const me={};k&2&&(me.$$scope={dirty:k,ctx:h}),o.$set(me);const A={};k&2&&(A.$$scope={dirty:k,ctx:h}),f.$set(A)},i(h){n||(y(t.$$.fragment,h),y(o.$$.fragment,h),y(f.$$.fragment,h),n=!0)},o(h){b(t.$$.fragment,h),b(o.$$.fragment,h),b(f.$$.fragment,h),n=!1},d(h){h&&(a(p),a(l)),M(t,h),M(o,h),M(f,h)}}}function yn(v){let t,p="Example:",o,l,f;return l=new O({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEdQVE5lb0NvbmZpZyUyQyUyMEdQVE5lb01vZGVsJTBBJTBBJTIzJTIwSW5pdGlhbGl6aW5nJTIwYSUyMEdQVE5lbyUyMEVsZXV0aGVyQUklMkZncHQtbmVvLTEuM0IlMjBzdHlsZSUyMGNvbmZpZ3VyYXRpb24lMEFjb25maWd1cmF0aW9uJTIwJTNEJTIwR1BUTmVvQ29uZmlnKCklMEElMEElMjMlMjBJbml0aWFsaXppbmclMjBhJTIwbW9kZWwlMjAod2l0aCUyMHJhbmRvbSUyMHdlaWdodHMpJTIwZnJvbSUyMHRoZSUyMEVsZXV0aGVyQUklMkZncHQtbmVvLTEuM0IlMjBzdHlsZSUyMGNvbmZpZ3VyYXRpb24lMEFtb2RlbCUyMCUzRCUyMEdQVE5lb01vZGVsKGNvbmZpZ3VyYXRpb24pJTBBJTBBJTIzJTIwQWNjZXNzaW5nJTIwdGhlJTIwbW9kZWwlMjBjb25maWd1cmF0aW9uJTBBY29uZmlndXJhdGlvbiUyMCUzRCUyMG1vZGVsLmNvbmZpZw==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> GPTNeoConfig, GPTNeoModel

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a GPTNeo EleutherAI/gpt-neo-1.3B style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = GPTNeoConfig()

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a model (with random weights) from the EleutherAI/gpt-neo-1.3B style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = GPTNeoModel(configuration)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Accessing the model configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = model.config`,wrap:!1}}),{c(){t=u("p"),t.textContent=p,o=r(),g(l.$$.fragment)},l(n){t=m(n,"P",{"data-svelte-h":!0}),w(t)!=="svelte-11lpom8"&&(t.textContent=p),o=i(n),_(l.$$.fragment,n)},m(n,h){c(n,t,h),c(n,o,h),T(l,n,h),f=!0},p:x,i(n){f||(y(l.$$.fragment,n),f=!0)},o(n){b(l.$$.fragment,n),f=!1},d(n){n&&(a(t),a(o)),M(l,n)}}}function bn(v){let t,p=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=u("p"),t.innerHTML=p},l(o){t=m(o,"P",{"data-svelte-h":!0}),w(t)!=="svelte-fincs2"&&(t.innerHTML=p)},m(o,l){c(o,t,l)},p:x,d(o){o&&a(t)}}}function Mn(v){let t,p=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=u("p"),t.innerHTML=p},l(o){t=m(o,"P",{"data-svelte-h":!0}),w(t)!=="svelte-fincs2"&&(t.innerHTML=p)},m(o,l){c(o,t,l)},p:x,d(o){o&&a(t)}}}function wn(v){let t,p="Example:",o,l,f;return l=new O({props:{code:"",highlighted:"",wrap:!1}}),{c(){t=u("p"),t.textContent=p,o=r(),g(l.$$.fragment)},l(n){t=m(n,"P",{"data-svelte-h":!0}),w(t)!=="svelte-11lpom8"&&(t.textContent=p),o=i(n),_(l.$$.fragment,n)},m(n,h){c(n,t,h),c(n,o,h),T(l,n,h),f=!0},p:x,i(n){f||(y(l.$$.fragment,n),f=!0)},o(n){b(l.$$.fragment,n),f=!1},d(n){n&&(a(t),a(o)),M(l,n)}}}function vn(v){let t,p=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=u("p"),t.innerHTML=p},l(o){t=m(o,"P",{"data-svelte-h":!0}),w(t)!=="svelte-fincs2"&&(t.innerHTML=p)},m(o,l){c(o,t,l)},p:x,d(o){o&&a(t)}}}function kn(v){let t,p="Example:",o,l,f;return l=new O({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBHUFROZW9Gb3JRdWVzdGlvbkFuc3dlcmluZyUwQWltcG9ydCUyMHRvcmNoJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIyRWxldXRoZXJBSSUyRmdwdC1uZW8tMS4zQiUyMiklMEFtb2RlbCUyMCUzRCUyMEdQVE5lb0ZvclF1ZXN0aW9uQW5zd2VyaW5nLmZyb21fcHJldHJhaW5lZCglMjJFbGV1dGhlckFJJTJGZ3B0LW5lby0xLjNCJTIyKSUwQSUwQXF1ZXN0aW9uJTJDJTIwdGV4dCUyMCUzRCUyMCUyMldobyUyMHdhcyUyMEppbSUyMEhlbnNvbiUzRiUyMiUyQyUyMCUyMkppbSUyMEhlbnNvbiUyMHdhcyUyMGElMjBuaWNlJTIwcHVwcGV0JTIyJTBBJTBBaW5wdXRzJTIwJTNEJTIwdG9rZW5pemVyKHF1ZXN0aW9uJTJDJTIwdGV4dCUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpJTBBd2l0aCUyMHRvcmNoLm5vX2dyYWQoKSUzQSUwQSUyMCUyMCUyMCUyMG91dHB1dHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyklMEElMEFhbnN3ZXJfc3RhcnRfaW5kZXglMjAlM0QlMjBvdXRwdXRzLnN0YXJ0X2xvZ2l0cy5hcmdtYXgoKSUwQWFuc3dlcl9lbmRfaW5kZXglMjAlM0QlMjBvdXRwdXRzLmVuZF9sb2dpdHMuYXJnbWF4KCklMEElMEFwcmVkaWN0X2Fuc3dlcl90b2tlbnMlMjAlM0QlMjBpbnB1dHMuaW5wdXRfaWRzJTVCMCUyQyUyMGFuc3dlcl9zdGFydF9pbmRleCUyMCUzQSUyMGFuc3dlcl9lbmRfaW5kZXglMjAlMkIlMjAxJTVEJTBBdG9rZW5pemVyLmRlY29kZShwcmVkaWN0X2Fuc3dlcl90b2tlbnMlMkMlMjBza2lwX3NwZWNpYWxfdG9rZW5zJTNEVHJ1ZSklMEElMEElMjMlMjB0YXJnZXQlMjBpcyUyMCUyMm5pY2UlMjBwdXBwZXQlMjIlMEF0YXJnZXRfc3RhcnRfaW5kZXglMjAlM0QlMjB0b3JjaC50ZW5zb3IoJTVCMTQlNUQpJTBBdGFyZ2V0X2VuZF9pbmRleCUyMCUzRCUyMHRvcmNoLnRlbnNvciglNUIxNSU1RCklMEElMEFvdXRwdXRzJTIwJTNEJTIwbW9kZWwoKippbnB1dHMlMkMlMjBzdGFydF9wb3NpdGlvbnMlM0R0YXJnZXRfc3RhcnRfaW5kZXglMkMlMjBlbmRfcG9zaXRpb25zJTNEdGFyZ2V0X2VuZF9pbmRleCklMEFsb3NzJTIwJTNEJTIwb3V0cHV0cy5sb3NzJTBBcm91bmQobG9zcy5pdGVtKCklMkMlMjAyKQ==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, GPTNeoForQuestionAnswering
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;EleutherAI/gpt-neo-1.3B&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = GPTNeoForQuestionAnswering.from_pretrained(<span class="hljs-string">&quot;EleutherAI/gpt-neo-1.3B&quot;</span>)

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
...`,wrap:!1}}),{c(){t=u("p"),t.textContent=p,o=r(),g(l.$$.fragment)},l(n){t=m(n,"P",{"data-svelte-h":!0}),w(t)!=="svelte-11lpom8"&&(t.textContent=p),o=i(n),_(l.$$.fragment,n)},m(n,h){c(n,t,h),c(n,o,h),T(l,n,h),f=!0},p:x,i(n){f||(y(l.$$.fragment,n),f=!0)},o(n){b(l.$$.fragment,n),f=!1},d(n){n&&(a(t),a(o)),M(l,n)}}}function $n(v){let t,p=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=u("p"),t.innerHTML=p},l(o){t=m(o,"P",{"data-svelte-h":!0}),w(t)!=="svelte-fincs2"&&(t.innerHTML=p)},m(o,l){c(o,t,l)},p:x,d(o){o&&a(t)}}}function Jn(v){let t,p="Example of single-label classification:",o,l,f;return l=new O({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Rva2VuaXplciUyQyUyMEdQVE5lb0ZvclNlcXVlbmNlQ2xhc3NpZmljYXRpb24lMEElMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJFbGV1dGhlckFJJTJGZ3B0LW5lby0xLjNCJTIyKSUwQW1vZGVsJTIwJTNEJTIwR1BUTmVvRm9yU2VxdWVuY2VDbGFzc2lmaWNhdGlvbi5mcm9tX3ByZXRyYWluZWQoJTIyRWxldXRoZXJBSSUyRmdwdC1uZW8tMS4zQiUyMiklMEElMEFpbnB1dHMlMjAlM0QlMjB0b2tlbml6ZXIoJTIySGVsbG8lMkMlMjBteSUyMGRvZyUyMGlzJTIwY3V0ZSUyMiUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpJTBBJTBBd2l0aCUyMHRvcmNoLm5vX2dyYWQoKSUzQSUwQSUyMCUyMCUyMCUyMGxvZ2l0cyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzKS5sb2dpdHMlMEElMEFwcmVkaWN0ZWRfY2xhc3NfaWQlMjAlM0QlMjBsb2dpdHMuYXJnbWF4KCkuaXRlbSgpJTBBbW9kZWwuY29uZmlnLmlkMmxhYmVsJTVCcHJlZGljdGVkX2NsYXNzX2lkJTVEJTBBJTBBJTIzJTIwVG8lMjB0cmFpbiUyMGElMjBtb2RlbCUyMG9uJTIwJTYwbnVtX2xhYmVscyU2MCUyMGNsYXNzZXMlMkMlMjB5b3UlMjBjYW4lMjBwYXNzJTIwJTYwbnVtX2xhYmVscyUzRG51bV9sYWJlbHMlNjAlMjB0byUyMCU2MC5mcm9tX3ByZXRyYWluZWQoLi4uKSU2MCUwQW51bV9sYWJlbHMlMjAlM0QlMjBsZW4obW9kZWwuY29uZmlnLmlkMmxhYmVsKSUwQW1vZGVsJTIwJTNEJTIwR1BUTmVvRm9yU2VxdWVuY2VDbGFzc2lmaWNhdGlvbi5mcm9tX3ByZXRyYWluZWQoJTIyRWxldXRoZXJBSSUyRmdwdC1uZW8tMS4zQiUyMiUyQyUyMG51bV9sYWJlbHMlM0RudW1fbGFiZWxzKSUwQSUwQWxhYmVscyUyMCUzRCUyMHRvcmNoLnRlbnNvciglNUIxJTVEKSUwQWxvc3MlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyUyQyUyMGxhYmVscyUzRGxhYmVscykubG9zcyUwQXJvdW5kKGxvc3MuaXRlbSgpJTJDJTIwMik=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, GPTNeoForSequenceClassification

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;EleutherAI/gpt-neo-1.3B&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = GPTNeoForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;EleutherAI/gpt-neo-1.3B&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    logits = model(**inputs).logits

<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_class_id = logits.argmax().item()
<span class="hljs-meta">&gt;&gt;&gt; </span>model.config.id2label[predicted_class_id]
...

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># To train a model on \`num_labels\` classes, you can pass \`num_labels=num_labels\` to \`.from_pretrained(...)\`</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>num_labels = <span class="hljs-built_in">len</span>(model.config.id2label)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = GPTNeoForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;EleutherAI/gpt-neo-1.3B&quot;</span>, num_labels=num_labels)

<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.tensor([<span class="hljs-number">1</span>])
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = model(**inputs, labels=labels).loss
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">round</span>(loss.item(), <span class="hljs-number">2</span>)
...`,wrap:!1}}),{c(){t=u("p"),t.textContent=p,o=r(),g(l.$$.fragment)},l(n){t=m(n,"P",{"data-svelte-h":!0}),w(t)!=="svelte-ykxpe4"&&(t.textContent=p),o=i(n),_(l.$$.fragment,n)},m(n,h){c(n,t,h),c(n,o,h),T(l,n,h),f=!0},p:x,i(n){f||(y(l.$$.fragment,n),f=!0)},o(n){b(l.$$.fragment,n),f=!1},d(n){n&&(a(t),a(o)),M(l,n)}}}function Cn(v){let t,p="Example of multi-label classification:",o,l,f;return l=new O({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Rva2VuaXplciUyQyUyMEdQVE5lb0ZvclNlcXVlbmNlQ2xhc3NpZmljYXRpb24lMEElMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJFbGV1dGhlckFJJTJGZ3B0LW5lby0xLjNCJTIyKSUwQW1vZGVsJTIwJTNEJTIwR1BUTmVvRm9yU2VxdWVuY2VDbGFzc2lmaWNhdGlvbi5mcm9tX3ByZXRyYWluZWQoJTIyRWxldXRoZXJBSSUyRmdwdC1uZW8tMS4zQiUyMiUyQyUyMHByb2JsZW1fdHlwZSUzRCUyMm11bHRpX2xhYmVsX2NsYXNzaWZpY2F0aW9uJTIyKSUwQSUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplciglMjJIZWxsbyUyQyUyMG15JTIwZG9nJTIwaXMlMjBjdXRlJTIyJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiklMEElMEF3aXRoJTIwdG9yY2gubm9fZ3JhZCgpJTNBJTBBJTIwJTIwJTIwJTIwbG9naXRzJTIwJTNEJTIwbW9kZWwoKippbnB1dHMpLmxvZ2l0cyUwQSUwQXByZWRpY3RlZF9jbGFzc19pZHMlMjAlM0QlMjB0b3JjaC5hcmFuZ2UoMCUyQyUyMGxvZ2l0cy5zaGFwZSU1Qi0xJTVEKSU1QnRvcmNoLnNpZ21vaWQobG9naXRzKS5zcXVlZXplKGRpbSUzRDApJTIwJTNFJTIwMC41JTVEJTBBJTBBJTIzJTIwVG8lMjB0cmFpbiUyMGElMjBtb2RlbCUyMG9uJTIwJTYwbnVtX2xhYmVscyU2MCUyMGNsYXNzZXMlMkMlMjB5b3UlMjBjYW4lMjBwYXNzJTIwJTYwbnVtX2xhYmVscyUzRG51bV9sYWJlbHMlNjAlMjB0byUyMCU2MC5mcm9tX3ByZXRyYWluZWQoLi4uKSU2MCUwQW51bV9sYWJlbHMlMjAlM0QlMjBsZW4obW9kZWwuY29uZmlnLmlkMmxhYmVsKSUwQW1vZGVsJTIwJTNEJTIwR1BUTmVvRm9yU2VxdWVuY2VDbGFzc2lmaWNhdGlvbi5mcm9tX3ByZXRyYWluZWQoJTBBJTIwJTIwJTIwJTIwJTIyRWxldXRoZXJBSSUyRmdwdC1uZW8tMS4zQiUyMiUyQyUyMG51bV9sYWJlbHMlM0RudW1fbGFiZWxzJTJDJTIwcHJvYmxlbV90eXBlJTNEJTIybXVsdGlfbGFiZWxfY2xhc3NpZmljYXRpb24lMjIlMEEpJTBBJTBBbGFiZWxzJTIwJTNEJTIwdG9yY2guc3VtKCUwQSUyMCUyMCUyMCUyMHRvcmNoLm5uLmZ1bmN0aW9uYWwub25lX2hvdChwcmVkaWN0ZWRfY2xhc3NfaWRzJTVCTm9uZSUyQyUyMCUzQSU1RC5jbG9uZSgpJTJDJTIwbnVtX2NsYXNzZXMlM0RudW1fbGFiZWxzKSUyQyUyMGRpbSUzRDElMEEpLnRvKHRvcmNoLmZsb2F0KSUwQWxvc3MlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyUyQyUyMGxhYmVscyUzRGxhYmVscykubG9zcw==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, GPTNeoForSequenceClassification

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;EleutherAI/gpt-neo-1.3B&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = GPTNeoForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;EleutherAI/gpt-neo-1.3B&quot;</span>, problem_type=<span class="hljs-string">&quot;multi_label_classification&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    logits = model(**inputs).logits

<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_class_ids = torch.arange(<span class="hljs-number">0</span>, logits.shape[-<span class="hljs-number">1</span>])[torch.sigmoid(logits).squeeze(dim=<span class="hljs-number">0</span>) &gt; <span class="hljs-number">0.5</span>]

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># To train a model on \`num_labels\` classes, you can pass \`num_labels=num_labels\` to \`.from_pretrained(...)\`</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>num_labels = <span class="hljs-built_in">len</span>(model.config.id2label)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = GPTNeoForSequenceClassification.from_pretrained(
<span class="hljs-meta">... </span>    <span class="hljs-string">&quot;EleutherAI/gpt-neo-1.3B&quot;</span>, num_labels=num_labels, problem_type=<span class="hljs-string">&quot;multi_label_classification&quot;</span>
<span class="hljs-meta">... </span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.<span class="hljs-built_in">sum</span>(
<span class="hljs-meta">... </span>    torch.nn.functional.one_hot(predicted_class_ids[<span class="hljs-literal">None</span>, :].clone(), num_classes=num_labels), dim=<span class="hljs-number">1</span>
<span class="hljs-meta">... </span>).to(torch.<span class="hljs-built_in">float</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = model(**inputs, labels=labels).loss`,wrap:!1}}),{c(){t=u("p"),t.textContent=p,o=r(),g(l.$$.fragment)},l(n){t=m(n,"P",{"data-svelte-h":!0}),w(t)!=="svelte-1l8e32d"&&(t.textContent=p),o=i(n),_(l.$$.fragment,n)},m(n,h){c(n,t,h),c(n,o,h),T(l,n,h),f=!0},p:x,i(n){f||(y(l.$$.fragment,n),f=!0)},o(n){b(l.$$.fragment,n),f=!1},d(n){n&&(a(t),a(o)),M(l,n)}}}function Gn(v){let t,p=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=u("p"),t.innerHTML=p},l(o){t=m(o,"P",{"data-svelte-h":!0}),w(t)!=="svelte-fincs2"&&(t.innerHTML=p)},m(o,l){c(o,t,l)},p:x,d(o){o&&a(t)}}}function Nn(v){let t,p="Example:",o,l,f;return l=new O({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBHUFROZW9Gb3JUb2tlbkNsYXNzaWZpY2F0aW9uJTBBaW1wb3J0JTIwdG9yY2glMEElMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJFbGV1dGhlckFJJTJGZ3B0LW5lby0xLjNCJTIyKSUwQW1vZGVsJTIwJTNEJTIwR1BUTmVvRm9yVG9rZW5DbGFzc2lmaWNhdGlvbi5mcm9tX3ByZXRyYWluZWQoJTIyRWxldXRoZXJBSSUyRmdwdC1uZW8tMS4zQiUyMiklMEElMEFpbnB1dHMlMjAlM0QlMjB0b2tlbml6ZXIoJTBBJTIwJTIwJTIwJTIwJTIySHVnZ2luZ0ZhY2UlMjBpcyUyMGElMjBjb21wYW55JTIwYmFzZWQlMjBpbiUyMFBhcmlzJTIwYW5kJTIwTmV3JTIwWW9yayUyMiUyQyUyMGFkZF9zcGVjaWFsX3Rva2VucyUzREZhbHNlJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiUwQSklMEElMEF3aXRoJTIwdG9yY2gubm9fZ3JhZCgpJTNBJTBBJTIwJTIwJTIwJTIwbG9naXRzJTIwJTNEJTIwbW9kZWwoKippbnB1dHMpLmxvZ2l0cyUwQSUwQXByZWRpY3RlZF90b2tlbl9jbGFzc19pZHMlMjAlM0QlMjBsb2dpdHMuYXJnbWF4KC0xKSUwQSUwQSUyMyUyME5vdGUlMjB0aGF0JTIwdG9rZW5zJTIwYXJlJTIwY2xhc3NpZmllZCUyMHJhdGhlciUyMHRoZW4lMjBpbnB1dCUyMHdvcmRzJTIwd2hpY2glMjBtZWFucyUyMHRoYXQlMEElMjMlMjB0aGVyZSUyMG1pZ2h0JTIwYmUlMjBtb3JlJTIwcHJlZGljdGVkJTIwdG9rZW4lMjBjbGFzc2VzJTIwdGhhbiUyMHdvcmRzLiUwQSUyMyUyME11bHRpcGxlJTIwdG9rZW4lMjBjbGFzc2VzJTIwbWlnaHQlMjBhY2NvdW50JTIwZm9yJTIwdGhlJTIwc2FtZSUyMHdvcmQlMEFwcmVkaWN0ZWRfdG9rZW5zX2NsYXNzZXMlMjAlM0QlMjAlNUJtb2RlbC5jb25maWcuaWQybGFiZWwlNUJ0Lml0ZW0oKSU1RCUyMGZvciUyMHQlMjBpbiUyMHByZWRpY3RlZF90b2tlbl9jbGFzc19pZHMlNUIwJTVEJTVEJTBBcHJlZGljdGVkX3Rva2Vuc19jbGFzc2VzJTBBJTBBbGFiZWxzJTIwJTNEJTIwcHJlZGljdGVkX3Rva2VuX2NsYXNzX2lkcyUwQWxvc3MlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyUyQyUyMGxhYmVscyUzRGxhYmVscykubG9zcyUwQXJvdW5kKGxvc3MuaXRlbSgpJTJDJTIwMik=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, GPTNeoForTokenClassification
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;EleutherAI/gpt-neo-1.3B&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = GPTNeoForTokenClassification.from_pretrained(<span class="hljs-string">&quot;EleutherAI/gpt-neo-1.3B&quot;</span>)

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
...`,wrap:!1}}),{c(){t=u("p"),t.textContent=p,o=r(),g(l.$$.fragment)},l(n){t=m(n,"P",{"data-svelte-h":!0}),w(t)!=="svelte-11lpom8"&&(t.textContent=p),o=i(n),_(l.$$.fragment,n)},m(n,h){c(n,t,h),c(n,o,h),T(l,n,h),f=!0},p:x,i(n){f||(y(l.$$.fragment,n),f=!0)},o(n){b(l.$$.fragment,n),f=!1},d(n){n&&(a(t),a(o)),M(l,n)}}}function jn(v){let t,p,o,l,f,n="<em>This model was released on 2021-03-21 and added to Hugging Face Transformers on 2021-03-30.</em>",h,k,ut='<div class="flex flex-wrap space-x-1"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/> <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat"/></div>',me,A,ht,he,Co='<a href="https://zenodo.org/records/5297715" rel="nofollow">GPT-Neo</a> is an open-source alternative to GPT-2 and GPT-3 models, built with Mesh TensorFlow for TPUs. GPT-Neo uses local attention in every other layer for more efficiency. It is trained on the <a href="https://huggingface.co/datasets/EleutherAI/pile" rel="nofollow">Pile</a>, a diverse dataset consisting of 22 smaller high-quality datasets. The original github repository can be found <a href="https://github.com/EleutherAI/gpt-neo/tree/v1.1" rel="nofollow">here</a>',ft,fe,Go='You can find all the original GPT-Neo checkpoints under the <a href="https://huggingface.co/EleutherAI?search_models=gpt-neo" rel="nofollow">EleutherAI</a> organization.',gt,K,_t,ge,No='The example below demonstrates how to generate text with <a href="/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.Pipeline">Pipeline</a> or the <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel">AutoModel</a>, and from the command line.',Tt,ee,yt,_e,jo='Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the <a href="../quantization/overview">Quantization</a> overview for more available quantization backends.',bt,Te,xo='The example below uses <a href="../quantization/bitsandbytes">bitsandbytes</a> to only quantize the weights to 4-bits.',Mt,ye,wt,be,vt,Me,Io="<li>Pad inputs on the right because GPT-Neo uses absolute position embeddings.</li>",kt,we,$t,z,ve,Bt,qe,zo=`This is the configuration class to store the configuration of a <a href="/docs/transformers/v4.56.2/en/model_doc/gpt_neo#transformers.GPTNeoModel">GPTNeoModel</a>. It is used to instantiate a GPT
Neo model according to the specified arguments, defining the model architecture. Instantiating a configuration with
the defaults will yield a similar configuration to that of the GPTNeo
<a href="https://huggingface.co/EleutherAI/gpt-neo-1.3B" rel="nofollow">EleutherAI/gpt-neo-1.3B</a> architecture.`,qt,Re,Uo=`Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> and can be used to control the model outputs. Read the
documentation from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> for more information.`,Rt,te,Jt,ke,Ct,C,$e,Vt,Ve,Fo="The bare Gpt Neo Model outputting raw hidden-states without any specific head on top.",Xt,Xe,Po=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Et,Ee,Wo=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Qt,Y,Je,Ht,Qe,Zo='The <a href="/docs/transformers/v4.56.2/en/model_doc/gpt_neo#transformers.GPTNeoModel">GPTNeoModel</a> forward method, overrides the <code>__call__</code> special method.',Lt,oe,Gt,Ce,Nt,G,Ge,St,He,Bo=`The GPT Neo Model transformer with a language modeling head on top (linear layer with weights tied to the input
embeddings).`,At,Le,qo=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Yt,Se,Ro=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Ot,R,Ne,Dt,Ae,Vo='The <a href="/docs/transformers/v4.56.2/en/model_doc/gpt_neo#transformers.GPTNeoForCausalLM">GPTNeoForCausalLM</a> forward method, overrides the <code>__call__</code> special method.',Kt,ne,eo,se,jt,je,xt,N,xe,to,Ye,Xo=`The Gpt Neo transformer with a span classification head on top for extractive question-answering tasks like
SQuAD (a linear layer on top of the hidden-states output to compute <code>span start logits</code> and <code>span end logits</code>).`,oo,Oe,Eo=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,no,De,Qo=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,so,V,Ie,ao,Ke,Ho='The <a href="/docs/transformers/v4.56.2/en/model_doc/gpt_neo#transformers.GPTNeoForQuestionAnswering">GPTNeoForQuestionAnswering</a> forward method, overrides the <code>__call__</code> special method.',ro,ae,io,re,It,ze,zt,$,Ue,lo,et,Lo="The GPTNeo Model transformer with a sequence classification head on top (linear layer).",co,tt,So=`<a href="/docs/transformers/v4.56.2/en/model_doc/gpt_neo#transformers.GPTNeoForSequenceClassification">GPTNeoForSequenceClassification</a> uses the last token in order to do the classification, as other causal models
(e.g. GPT-1) do.`,po,ot,Ao=`Since it does classification on the last token, it requires to know the position of the last token. If a
<code>pad_token_id</code> is defined in the configuration, it finds the last token that is not a padding token in each row. If
no <code>pad_token_id</code> is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
padding tokens when <code>inputs_embeds</code> are passed instead of <code>input_ids</code>, it does the same (take the last value in
each row of the batch).`,uo,nt,Yo=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,mo,st,Oo=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,ho,I,Fe,fo,at,Do='The <a href="/docs/transformers/v4.56.2/en/model_doc/gpt_neo#transformers.GPTNeoForSequenceClassification">GPTNeoForSequenceClassification</a> forward method, overrides the <code>__call__</code> special method.',go,ie,_o,le,To,de,Ut,Pe,Ft,j,We,yo,rt,Ko=`The Gpt Neo transformer with a token classification head on top (a linear layer on top of the hidden-states
output) e.g. for Named-Entity-Recognition (NER) tasks.`,bo,it,en=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Mo,lt,tn=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,wo,X,Ze,vo,dt,on='The <a href="/docs/transformers/v4.56.2/en/model_doc/gpt_neo#transformers.GPTNeoForTokenClassification">GPTNeoForTokenClassification</a> forward method, overrides the <code>__call__</code> special method.',ko,ce,$o,pe,Pt,Be,Wt,mt,Zt;return A=new ue({props:{title:"GPT-Neo",local:"gpt-neo",headingTag:"h2"}}),K=new ct({props:{warning:!1,$$slots:{default:[hn]},$$scope:{ctx:v}}}),ee=new mn({props:{id:"usage",options:["Pipeline","AutoModel","transformers CLI"],$$slots:{default:[Tn]},$$scope:{ctx:v}}}),ye=new O({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b01vZGVsRm9yQ2F1c2FsTE0lMkMlMjBBdXRvVG9rZW5pemVyJTJDJTIwQml0c0FuZEJ5dGVzQ29uZmlnJTBBJTBBcXVhbnRpemF0aW9uX2NvbmZpZyUyMCUzRCUyMEJpdHNBbmRCeXRlc0NvbmZpZyglMEElMjAlMjAlMjAlMjBsb2FkX2luXzRiaXQlM0RUcnVlJTJDJTBBJTIwJTIwJTIwJTIwYm5iXzRiaXRfcXVhbnRfdHlwZSUzRCUyMm5mNCUyMiUyQyUwQSUyMCUyMCUyMCUyMGJuYl80Yml0X2NvbXB1dGVfZHR5cGUlM0QlMjJmbG9hdDE2JTIyJTJDJTBBJTIwJTIwJTIwJTIwYm5iXzRiaXRfdXNlX2RvdWJsZV9xdWFudCUzRFRydWUlMEEpJTBBJTBBbW9kZWwlMjAlM0QlMjBBdXRvTW9kZWxGb3JDYXVzYWxMTS5mcm9tX3ByZXRyYWluZWQoJTBBJTIwJTIwJTIwJTIwJTIyRWxldXRoZXJBSSUyRmdwdC1uZW8tMi43QiUyMiUyQyUwQSUyMCUyMCUyMCUyMHF1YW50aXphdGlvbl9jb25maWclM0RxdWFudGl6YXRpb25fY29uZmlnJTJDJTBBJTIwJTIwJTIwJTIwZGV2aWNlX21hcCUzRCUyMmF1dG8lMjIlMEEpJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIyRWxldXRoZXJBSSUyRmdwdC1uZW8tMi43QiUyMiklMEFpbnB1dHMlMjAlM0QlMjB0b2tlbml6ZXIoJTIySGVsbG8lMkMlMjBJJ20lMjBhJTIwbGFuZ3VhZ2UlMjBtb2RlbCUyMiUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpLnRvKG1vZGVsLmRldmljZSklMEFvdXRwdXRzJTIwJTNEJTIwbW9kZWwuZ2VuZXJhdGUoKippbnB1dHMlMkMlMjBtYXhfbmV3X3Rva2VucyUzRDEwMCklMEFwcmludCh0b2tlbml6ZXIuZGVjb2RlKG91dHB1dHMlNUIwJTVEJTJDJTIwc2tpcF9zcGVjaWFsX3Rva2VucyUzRFRydWUpKQ==",highlighted:`<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=<span class="hljs-literal">True</span>,
    bnb_4bit_quant_type=<span class="hljs-string">&quot;nf4&quot;</span>,
    bnb_4bit_compute_dtype=<span class="hljs-string">&quot;float16&quot;</span>,
    bnb_4bit_use_double_quant=<span class="hljs-literal">True</span>
)

model = AutoModelForCausalLM.from_pretrained(
    <span class="hljs-string">&quot;EleutherAI/gpt-neo-2.7B&quot;</span>,
    quantization_config=quantization_config,
    device_map=<span class="hljs-string">&quot;auto&quot;</span>
)

tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;EleutherAI/gpt-neo-2.7B&quot;</span>)
inputs = tokenizer(<span class="hljs-string">&quot;Hello, I&#x27;m a language model&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>).to(model.device)
outputs = model.generate(**inputs, max_new_tokens=<span class="hljs-number">100</span>)
<span class="hljs-built_in">print</span>(tokenizer.decode(outputs[<span class="hljs-number">0</span>], skip_special_tokens=<span class="hljs-literal">True</span>))`,wrap:!1}}),be=new ue({props:{title:"Notes",local:"notes",headingTag:"h2"}}),we=new ue({props:{title:"GPTNeoConfig",local:"transformers.GPTNeoConfig",headingTag:"h2"}}),ve=new S({props:{name:"class transformers.GPTNeoConfig",anchor:"transformers.GPTNeoConfig",parameters:[{name:"vocab_size",val:" = 50257"},{name:"max_position_embeddings",val:" = 2048"},{name:"hidden_size",val:" = 2048"},{name:"num_layers",val:" = 24"},{name:"attention_types",val:" = [[['global', 'local'], 12]]"},{name:"num_heads",val:" = 16"},{name:"intermediate_size",val:" = None"},{name:"window_size",val:" = 256"},{name:"activation_function",val:" = 'gelu_new'"},{name:"resid_dropout",val:" = 0.0"},{name:"embed_dropout",val:" = 0.0"},{name:"attention_dropout",val:" = 0.0"},{name:"classifier_dropout",val:" = 0.1"},{name:"layer_norm_epsilon",val:" = 1e-05"},{name:"initializer_range",val:" = 0.02"},{name:"use_cache",val:" = True"},{name:"bos_token_id",val:" = 50256"},{name:"eos_token_id",val:" = 50256"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.GPTNeoConfig.vocab_size",description:`<strong>vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 50257) &#x2014;
Vocabulary size of the GPT Neo model. Defines the number of different tokens that can be represented by the
<code>inputs_ids</code> passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/gpt_neo#transformers.GPTNeoModel">GPTNeoModel</a>. Vocabulary size of the model. Defines the different
tokens that can be represented by the <em>inputs_ids</em> passed to the forward method of <a href="/docs/transformers/v4.56.2/en/model_doc/gpt_neo#transformers.GPTNeoModel">GPTNeoModel</a>.`,name:"vocab_size"},{anchor:"transformers.GPTNeoConfig.max_position_embeddings",description:`<strong>max_position_embeddings</strong> (<code>int</code>, <em>optional</em>, defaults to 2048) &#x2014;
The maximum sequence length that this model might ever be used with. Typically set this to something large
just in case (e.g., 512 or 1024 or 2048).`,name:"max_position_embeddings"},{anchor:"transformers.GPTNeoConfig.hidden_size",description:`<strong>hidden_size</strong> (<code>int</code>, <em>optional</em>, defaults to 2048) &#x2014;
Dimensionality of the encoder layers and the pooler layer.`,name:"hidden_size"},{anchor:"transformers.GPTNeoConfig.num_layers",description:`<strong>num_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 24) &#x2014;
Number of hidden layers in the Transformer encoder.`,name:"num_layers"},{anchor:"transformers.GPTNeoConfig.attention_types",description:`<strong>attention_types</strong> (<code>List</code>, <em>optional</em>, defaults to <code>[[[&apos;global&apos;, &apos;local&apos;], 12]]</code>) &#x2014;
The type of attention for each layer in a <code>List</code> of the following format <code>[[[&quot;attention_type&quot;], num_layerss]]</code> e.g. for a 24 layer model <code>[[[&quot;global&quot;], 24]]</code> or <code>[[[&quot;global&quot;, &quot;local&quot;], 12]]</code> Choose the
value of <code>attention_type</code> from <code>[&quot;global&quot;, &quot;local&quot;]</code>`,name:"attention_types"},{anchor:"transformers.GPTNeoConfig.num_heads",description:`<strong>num_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 16) &#x2014;
Number of attention heads for each attention layer in the Transformer encoder.`,name:"num_heads"},{anchor:"transformers.GPTNeoConfig.intermediate_size",description:`<strong>intermediate_size</strong> (<code>int</code>, <em>optional</em>, defaults to 8192) &#x2014;
Dimensionality of the &#x201C;intermediate&#x201D; (i.e., feed-forward) layer in the Transformer encoder.`,name:"intermediate_size"},{anchor:"transformers.GPTNeoConfig.window_size",description:`<strong>window_size</strong> (<code>int</code>, <em>optional</em>, defaults to 256) &#x2014;
The size of the sliding window for local attention.`,name:"window_size"},{anchor:"transformers.GPTNeoConfig.activation_function",description:`<strong>activation_function</strong> (<code>str</code> or <code>function</code>, <em>optional</em>, defaults to <code>&quot;gelu_new&quot;</code>) &#x2014;
The non-linear activation function (function or string) in the encoder and pooler. If string, <code>&quot;gelu&quot;</code>,
<code>&quot;relu&quot;</code>, <code>&quot;selu&quot;</code> and <code>&quot;gelu_new&quot;</code> are supported.`,name:"activation_function"},{anchor:"transformers.GPTNeoConfig.resid_dropout",description:`<strong>resid_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
Residual dropout used in the attention pattern.`,name:"resid_dropout"},{anchor:"transformers.GPTNeoConfig.embed_dropout",description:`<strong>embed_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.`,name:"embed_dropout"},{anchor:"transformers.GPTNeoConfig.attention_dropout",description:`<strong>attention_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The dropout ratio for the attention probabilities.`,name:"attention_dropout"},{anchor:"transformers.GPTNeoConfig.classifier_dropout",description:`<strong>classifier_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
Argument used when doing token classification, used in the model <a href="/docs/transformers/v4.56.2/en/model_doc/gpt_neo#transformers.GPTNeoForTokenClassification">GPTNeoForTokenClassification</a>. The
dropout ratio for the hidden layer.`,name:"classifier_dropout"},{anchor:"transformers.GPTNeoConfig.layer_norm_epsilon",description:`<strong>layer_norm_epsilon</strong> (<code>float</code>, <em>optional</em>, defaults to 1e-05) &#x2014;
The epsilon used by the layer normalization layers.`,name:"layer_norm_epsilon"},{anchor:"transformers.GPTNeoConfig.initializer_range",description:`<strong>initializer_range</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.`,name:"initializer_range"},{anchor:"transformers.GPTNeoConfig.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not the model should return the last key/values attentions (not used by all models). Only
relevant if <code>config.is_decoder=True</code>.`,name:"use_cache"},{anchor:"transformers.GPTNeoConfig.bos_token_id",description:`<strong>bos_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 50256) &#x2014;
The id of the beginning of sentence token in the vocabulary.`,name:"bos_token_id"},{anchor:"transformers.GPTNeoConfig.eos_token_id",description:`<strong>eos_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 50256) &#x2014;
The id of the end of sentence token in the vocabulary.`,name:"eos_token_id"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/gpt_neo/configuration_gpt_neo.py#L30"}}),te=new pt({props:{anchor:"transformers.GPTNeoConfig.example",$$slots:{default:[yn]},$$scope:{ctx:v}}}),ke=new ue({props:{title:"GPTNeoModel",local:"transformers.GPTNeoModel",headingTag:"h2"}}),$e=new S({props:{name:"class transformers.GPTNeoModel",anchor:"transformers.GPTNeoModel",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.GPTNeoModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/gpt_neo#transformers.GPTNeoModel">GPTNeoModel</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/gpt_neo/modeling_gpt_neo.py#L503"}}),Je=new S({props:{name:"forward",anchor:"transformers.GPTNeoModel.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"past_key_values",val:": typing.Union[transformers.cache_utils.Cache, tuple[torch.FloatTensor], NoneType] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"}],parametersDescription:[{anchor:"transformers.GPTNeoModel.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, input_ids_length)</code>) &#x2014;
<code>input_ids_length</code> = <code>sequence_length</code> if <code>past_key_values</code> is <code>None</code> else
<code>past_key_values.get_seq_length()</code> (<code>sequence_length</code> of input past key value states). Indices of input
sequence tokens in the vocabulary.</p>
<p>If <code>past_key_values</code> is used, only <code>input_ids</code> that do not have their past calculated should be passed as
<code>input_ids</code>.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.GPTNeoModel.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>Union[~cache_utils.Cache, tuple[torch.FloatTensor], NoneType]</code>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.GPTNeoModel.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.GPTNeoModel.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.GPTNeoModel.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.GPTNeoModel.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.GPTNeoModel.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.GPTNeoModel.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.GPTNeoModel.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.GPTNeoModel.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.GPTNeoModel.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.GPTNeoModel.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/gpt_neo/modeling_gpt_neo.py#L524",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPastAndCrossAttentions"
>transformers.modeling_outputs.BaseModelOutputWithPastAndCrossAttentions</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/gpt_neo#transformers.GPTNeoConfig"
>GPTNeoConfig</a>) and inputs.</p>
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
`}}),oe=new ct({props:{$$slots:{default:[bn]},$$scope:{ctx:v}}}),Ce=new ue({props:{title:"GPTNeoForCausalLM",local:"transformers.GPTNeoForCausalLM",headingTag:"h2"}}),Ge=new S({props:{name:"class transformers.GPTNeoForCausalLM",anchor:"transformers.GPTNeoForCausalLM",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.GPTNeoForCausalLM.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/gpt_neo#transformers.GPTNeoForCausalLM">GPTNeoForCausalLM</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/gpt_neo/modeling_gpt_neo.py#L780"}}),Ne=new S({props:{name:"forward",anchor:"transformers.GPTNeoForCausalLM.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"past_key_values",val:": typing.Union[transformers.cache_utils.Cache, tuple[torch.FloatTensor], NoneType] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"labels",val:": typing.Optional[torch.Tensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.GPTNeoForCausalLM.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, input_ids_length)</code>) &#x2014;
<code>input_ids_length</code> = <code>sequence_length</code> if <code>past_key_values</code> is <code>None</code> else
<code>past_key_values.get_seq_length()</code> (<code>sequence_length</code> of input past key value states). Indices of input
sequence tokens in the vocabulary.</p>
<p>If <code>past_key_values</code> is used, only <code>input_ids</code> that do not have their past calculated should be passed as
<code>input_ids</code>.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.GPTNeoForCausalLM.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>Union[~cache_utils.Cache, tuple[torch.FloatTensor], NoneType]</code>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.GPTNeoForCausalLM.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.GPTNeoForCausalLM.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.GPTNeoForCausalLM.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.GPTNeoForCausalLM.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.GPTNeoForCausalLM.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.GPTNeoForCausalLM.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, input_ids_length)</code>, <em>optional</em>) &#x2014;
Labels for language modeling. Note that the labels <strong>are shifted</strong> inside the model, i.e. you can set
<code>labels = input_ids</code> Indices are selected in <code>[-100, 0, ..., config.vocab_size]</code> All labels set to <code>-100</code>
are ignored (masked), the loss is only computed for labels in <code>[0, ..., config.vocab_size]</code>`,name:"labels"},{anchor:"transformers.GPTNeoForCausalLM.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.GPTNeoForCausalLM.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.GPTNeoForCausalLM.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.GPTNeoForCausalLM.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.GPTNeoForCausalLM.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/gpt_neo/modeling_gpt_neo.py#L791",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithCrossAttentions"
>transformers.modeling_outputs.CausalLMOutputWithCrossAttentions</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/gpt_neo#transformers.GPTNeoConfig"
>GPTNeoConfig</a>) and inputs.</p>
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
`}}),ne=new ct({props:{$$slots:{default:[Mn]},$$scope:{ctx:v}}}),se=new pt({props:{anchor:"transformers.GPTNeoForCausalLM.forward.example",$$slots:{default:[wn]},$$scope:{ctx:v}}}),je=new ue({props:{title:"GPTNeoForQuestionAnswering",local:"transformers.GPTNeoForQuestionAnswering",headingTag:"h2"}}),xe=new S({props:{name:"class transformers.GPTNeoForQuestionAnswering",anchor:"transformers.GPTNeoForQuestionAnswering",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.GPTNeoForQuestionAnswering.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/gpt_neo#transformers.GPTNeoForQuestionAnswering">GPTNeoForQuestionAnswering</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/gpt_neo/modeling_gpt_neo.py#L1099"}}),Ie=new S({props:{name:"forward",anchor:"transformers.GPTNeoForQuestionAnswering.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"start_positions",val:": typing.Optional[torch.LongTensor] = None"},{name:"end_positions",val:": typing.Optional[torch.LongTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.GPTNeoForQuestionAnswering.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, input_ids_length)</code>) &#x2014;
<code>input_ids_length</code> = <code>sequence_length</code> if <code>past_key_values</code> is <code>None</code> else
<code>past_key_values.get_seq_length()</code> (<code>sequence_length</code> of input past key value states). Indices of input
sequence tokens in the vocabulary.</p>
<p>If <code>past_key_values</code> is used, only <code>input_ids</code> that do not have their past calculated should be passed as
<code>input_ids</code>.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.GPTNeoForQuestionAnswering.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.GPTNeoForQuestionAnswering.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.GPTNeoForQuestionAnswering.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.GPTNeoForQuestionAnswering.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.GPTNeoForQuestionAnswering.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.GPTNeoForQuestionAnswering.forward.start_positions",description:`<strong>start_positions</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for position (index) of the start of the labelled span for computing the token classification loss.
Positions are clamped to the length of the sequence (<code>sequence_length</code>). Position outside of the sequence
are not taken into account for computing the loss.`,name:"start_positions"},{anchor:"transformers.GPTNeoForQuestionAnswering.forward.end_positions",description:`<strong>end_positions</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for position (index) of the end of the labelled span for computing the token classification loss.
Positions are clamped to the length of the sequence (<code>sequence_length</code>). Position outside of the sequence
are not taken into account for computing the loss.`,name:"end_positions"},{anchor:"transformers.GPTNeoForQuestionAnswering.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.GPTNeoForQuestionAnswering.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.GPTNeoForQuestionAnswering.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/gpt_neo/modeling_gpt_neo.py#L1109",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.QuestionAnsweringModelOutput"
>transformers.modeling_outputs.QuestionAnsweringModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/gpt_neo#transformers.GPTNeoConfig"
>GPTNeoConfig</a>) and inputs.</p>
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
`}}),ae=new ct({props:{$$slots:{default:[vn]},$$scope:{ctx:v}}}),re=new pt({props:{anchor:"transformers.GPTNeoForQuestionAnswering.forward.example",$$slots:{default:[kn]},$$scope:{ctx:v}}}),ze=new ue({props:{title:"GPTNeoForSequenceClassification",local:"transformers.GPTNeoForSequenceClassification",headingTag:"h2"}}),Ue=new S({props:{name:"class transformers.GPTNeoForSequenceClassification",anchor:"transformers.GPTNeoForSequenceClassification",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.GPTNeoForSequenceClassification.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/gpt_neo#transformers.GPTNeoForSequenceClassification">GPTNeoForSequenceClassification</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/gpt_neo/modeling_gpt_neo.py#L893"}}),Fe=new S({props:{name:"forward",anchor:"transformers.GPTNeoForSequenceClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"past_key_values",val:": typing.Union[transformers.cache_utils.Cache, tuple[torch.FloatTensor], NoneType] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"labels",val:": typing.Optional[torch.Tensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.GPTNeoForSequenceClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, input_ids_length)</code>) &#x2014;
<code>input_ids_length</code> = <code>sequence_length</code> if <code>past_key_values</code> is <code>None</code> else
<code>past_key_values.get_seq_length()</code> (<code>sequence_length</code> of input past key value states). Indices of input
sequence tokens in the vocabulary.</p>
<p>If <code>past_key_values</code> is used, only <code>input_ids</code> that do not have their past calculated should be passed as
<code>input_ids</code>.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.GPTNeoForSequenceClassification.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>Union[~cache_utils.Cache, tuple[torch.FloatTensor], NoneType]</code>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.GPTNeoForSequenceClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.GPTNeoForSequenceClassification.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.GPTNeoForSequenceClassification.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.GPTNeoForSequenceClassification.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.GPTNeoForSequenceClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.GPTNeoForSequenceClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the sequence classification/regression loss. Indices should be in <code>[0, ..., config.num_labels - 1]</code>. If <code>config.num_labels == 1</code> a regression loss is computed (Mean-Square loss), If
<code>config.num_labels &gt; 1</code> a classification loss is computed (Cross-Entropy).`,name:"labels"},{anchor:"transformers.GPTNeoForSequenceClassification.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.GPTNeoForSequenceClassification.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.GPTNeoForSequenceClassification.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.GPTNeoForSequenceClassification.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/gpt_neo/modeling_gpt_neo.py#L903",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <code>transformers.modeling_outputs.SequenceClassifierOutputWithPast</code> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/gpt_neo#transformers.GPTNeoConfig"
>GPTNeoConfig</a>) and inputs.</p>
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
`}}),ie=new ct({props:{$$slots:{default:[$n]},$$scope:{ctx:v}}}),le=new pt({props:{anchor:"transformers.GPTNeoForSequenceClassification.forward.example",$$slots:{default:[Jn]},$$scope:{ctx:v}}}),de=new pt({props:{anchor:"transformers.GPTNeoForSequenceClassification.forward.example-2",$$slots:{default:[Cn]},$$scope:{ctx:v}}}),Pe=new ue({props:{title:"GPTNeoForTokenClassification",local:"transformers.GPTNeoForTokenClassification",headingTag:"h2"}}),We=new S({props:{name:"class transformers.GPTNeoForTokenClassification",anchor:"transformers.GPTNeoForTokenClassification",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.GPTNeoForTokenClassification.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/gpt_neo#transformers.GPTNeoForTokenClassification">GPTNeoForTokenClassification</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/gpt_neo/modeling_gpt_neo.py#L1014"}}),Ze=new S({props:{name:"forward",anchor:"transformers.GPTNeoForTokenClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Union[transformers.cache_utils.Cache, tuple[tuple[torch.Tensor]], NoneType] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.GPTNeoForTokenClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, input_ids_length)</code>) &#x2014;
<code>input_ids_length</code> = <code>sequence_length</code> if <code>past_key_values</code> is <code>None</code> else
<code>past_key_values.get_seq_length()</code> (<code>sequence_length</code> of input past key value states). Indices of input
sequence tokens in the vocabulary.</p>
<p>If <code>past_key_values</code> is used, only <code>input_ids</code> that do not have their past calculated should be passed as
<code>input_ids</code>.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.GPTNeoForTokenClassification.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>Union[~cache_utils.Cache, tuple[tuple[torch.Tensor]], NoneType]</code>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.GPTNeoForTokenClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.GPTNeoForTokenClassification.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.GPTNeoForTokenClassification.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.GPTNeoForTokenClassification.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.GPTNeoForTokenClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.GPTNeoForTokenClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the sequence classification/regression loss. Indices should be in <code>[0, ..., config.num_labels - 1]</code>. If <code>config.num_labels == 1</code> a regression loss is computed (Mean-Square loss), If
<code>config.num_labels &gt; 1</code> a classification loss is computed (Cross-Entropy).`,name:"labels"},{anchor:"transformers.GPTNeoForTokenClassification.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.GPTNeoForTokenClassification.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.GPTNeoForTokenClassification.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.GPTNeoForTokenClassification.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/gpt_neo/modeling_gpt_neo.py#L1026",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput"
>transformers.modeling_outputs.TokenClassifierOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/gpt_neo#transformers.GPTNeoConfig"
>GPTNeoConfig</a>) and inputs.</p>
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
`}}),ce=new ct({props:{$$slots:{default:[Gn]},$$scope:{ctx:v}}}),pe=new pt({props:{anchor:"transformers.GPTNeoForTokenClassification.forward.example",$$slots:{default:[Nn]},$$scope:{ctx:v}}}),Be=new un({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/gpt_neo.md"}}),{c(){t=u("meta"),p=r(),o=u("p"),l=r(),f=u("p"),f.innerHTML=n,h=r(),k=u("div"),k.innerHTML=ut,me=r(),g(A.$$.fragment),ht=r(),he=u("p"),he.innerHTML=Co,ft=r(),fe=u("p"),fe.innerHTML=Go,gt=r(),g(K.$$.fragment),_t=r(),ge=u("p"),ge.innerHTML=No,Tt=r(),g(ee.$$.fragment),yt=r(),_e=u("p"),_e.innerHTML=jo,bt=r(),Te=u("p"),Te.innerHTML=xo,Mt=r(),g(ye.$$.fragment),wt=r(),g(be.$$.fragment),vt=r(),Me=u("ul"),Me.innerHTML=Io,kt=r(),g(we.$$.fragment),$t=r(),z=u("div"),g(ve.$$.fragment),Bt=r(),qe=u("p"),qe.innerHTML=zo,qt=r(),Re=u("p"),Re.innerHTML=Uo,Rt=r(),g(te.$$.fragment),Jt=r(),g(ke.$$.fragment),Ct=r(),C=u("div"),g($e.$$.fragment),Vt=r(),Ve=u("p"),Ve.textContent=Fo,Xt=r(),Xe=u("p"),Xe.innerHTML=Po,Et=r(),Ee=u("p"),Ee.innerHTML=Wo,Qt=r(),Y=u("div"),g(Je.$$.fragment),Ht=r(),Qe=u("p"),Qe.innerHTML=Zo,Lt=r(),g(oe.$$.fragment),Gt=r(),g(Ce.$$.fragment),Nt=r(),G=u("div"),g(Ge.$$.fragment),St=r(),He=u("p"),He.textContent=Bo,At=r(),Le=u("p"),Le.innerHTML=qo,Yt=r(),Se=u("p"),Se.innerHTML=Ro,Ot=r(),R=u("div"),g(Ne.$$.fragment),Dt=r(),Ae=u("p"),Ae.innerHTML=Vo,Kt=r(),g(ne.$$.fragment),eo=r(),g(se.$$.fragment),jt=r(),g(je.$$.fragment),xt=r(),N=u("div"),g(xe.$$.fragment),to=r(),Ye=u("p"),Ye.innerHTML=Xo,oo=r(),Oe=u("p"),Oe.innerHTML=Eo,no=r(),De=u("p"),De.innerHTML=Qo,so=r(),V=u("div"),g(Ie.$$.fragment),ao=r(),Ke=u("p"),Ke.innerHTML=Ho,ro=r(),g(ae.$$.fragment),io=r(),g(re.$$.fragment),It=r(),g(ze.$$.fragment),zt=r(),$=u("div"),g(Ue.$$.fragment),lo=r(),et=u("p"),et.textContent=Lo,co=r(),tt=u("p"),tt.innerHTML=So,po=r(),ot=u("p"),ot.innerHTML=Ao,uo=r(),nt=u("p"),nt.innerHTML=Yo,mo=r(),st=u("p"),st.innerHTML=Oo,ho=r(),I=u("div"),g(Fe.$$.fragment),fo=r(),at=u("p"),at.innerHTML=Do,go=r(),g(ie.$$.fragment),_o=r(),g(le.$$.fragment),To=r(),g(de.$$.fragment),Ut=r(),g(Pe.$$.fragment),Ft=r(),j=u("div"),g(We.$$.fragment),yo=r(),rt=u("p"),rt.textContent=Ko,bo=r(),it=u("p"),it.innerHTML=en,Mo=r(),lt=u("p"),lt.innerHTML=tn,wo=r(),X=u("div"),g(Ze.$$.fragment),vo=r(),dt=u("p"),dt.innerHTML=on,ko=r(),g(ce.$$.fragment),$o=r(),g(pe.$$.fragment),Pt=r(),g(Be.$$.fragment),Wt=r(),mt=u("p"),this.h()},l(e){const s=cn("svelte-u9bgzb",document.head);t=m(s,"META",{name:!0,content:!0}),s.forEach(a),p=i(e),o=m(e,"P",{}),B(o).forEach(a),l=i(e),f=m(e,"P",{"data-svelte-h":!0}),w(f)!=="svelte-1rynhwq"&&(f.innerHTML=n),h=i(e),k=m(e,"DIV",{style:!0,"data-svelte-h":!0}),w(k)!=="svelte-1wvb92v"&&(k.innerHTML=ut),me=i(e),_(A.$$.fragment,e),ht=i(e),he=m(e,"P",{"data-svelte-h":!0}),w(he)!=="svelte-r52f58"&&(he.innerHTML=Co),ft=i(e),fe=m(e,"P",{"data-svelte-h":!0}),w(fe)!=="svelte-1u1nqg6"&&(fe.innerHTML=Go),gt=i(e),_(K.$$.fragment,e),_t=i(e),ge=m(e,"P",{"data-svelte-h":!0}),w(ge)!=="svelte-x9rs6r"&&(ge.innerHTML=No),Tt=i(e),_(ee.$$.fragment,e),yt=i(e),_e=m(e,"P",{"data-svelte-h":!0}),w(_e)!=="svelte-nf5ooi"&&(_e.innerHTML=jo),bt=i(e),Te=m(e,"P",{"data-svelte-h":!0}),w(Te)!=="svelte-60nsd0"&&(Te.innerHTML=xo),Mt=i(e),_(ye.$$.fragment,e),wt=i(e),_(be.$$.fragment,e),vt=i(e),Me=m(e,"UL",{"data-svelte-h":!0}),w(Me)!=="svelte-r7n8xn"&&(Me.innerHTML=Io),kt=i(e),_(we.$$.fragment,e),$t=i(e),z=m(e,"DIV",{class:!0});var E=B(z);_(ve.$$.fragment,E),Bt=i(E),qe=m(E,"P",{"data-svelte-h":!0}),w(qe)!=="svelte-ld073b"&&(qe.innerHTML=zo),qt=i(E),Re=m(E,"P",{"data-svelte-h":!0}),w(Re)!=="svelte-1ek1ss9"&&(Re.innerHTML=Uo),Rt=i(E),_(te.$$.fragment,E),E.forEach(a),Jt=i(e),_(ke.$$.fragment,e),Ct=i(e),C=m(e,"DIV",{class:!0});var U=B(C);_($e.$$.fragment,U),Vt=i(U),Ve=m(U,"P",{"data-svelte-h":!0}),w(Ve)!=="svelte-1rkbf1j"&&(Ve.textContent=Fo),Xt=i(U),Xe=m(U,"P",{"data-svelte-h":!0}),w(Xe)!=="svelte-q52n56"&&(Xe.innerHTML=Po),Et=i(U),Ee=m(U,"P",{"data-svelte-h":!0}),w(Ee)!=="svelte-hswkmf"&&(Ee.innerHTML=Wo),Qt=i(U),Y=m(U,"DIV",{class:!0});var D=B(Y);_(Je.$$.fragment,D),Ht=i(D),Qe=m(D,"P",{"data-svelte-h":!0}),w(Qe)!=="svelte-1c7k6dr"&&(Qe.innerHTML=Zo),Lt=i(D),_(oe.$$.fragment,D),D.forEach(a),U.forEach(a),Gt=i(e),_(Ce.$$.fragment,e),Nt=i(e),G=m(e,"DIV",{class:!0});var F=B(G);_(Ge.$$.fragment,F),St=i(F),He=m(F,"P",{"data-svelte-h":!0}),w(He)!=="svelte-ygetm0"&&(He.textContent=Bo),At=i(F),Le=m(F,"P",{"data-svelte-h":!0}),w(Le)!=="svelte-q52n56"&&(Le.innerHTML=qo),Yt=i(F),Se=m(F,"P",{"data-svelte-h":!0}),w(Se)!=="svelte-hswkmf"&&(Se.innerHTML=Ro),Ot=i(F),R=m(F,"DIV",{class:!0});var Q=B(R);_(Ne.$$.fragment,Q),Dt=i(Q),Ae=m(Q,"P",{"data-svelte-h":!0}),w(Ae)!=="svelte-13l57lr"&&(Ae.innerHTML=Vo),Kt=i(Q),_(ne.$$.fragment,Q),eo=i(Q),_(se.$$.fragment,Q),Q.forEach(a),F.forEach(a),jt=i(e),_(je.$$.fragment,e),xt=i(e),N=m(e,"DIV",{class:!0});var P=B(N);_(xe.$$.fragment,P),to=i(P),Ye=m(P,"P",{"data-svelte-h":!0}),w(Ye)!=="svelte-kcocx6"&&(Ye.innerHTML=Xo),oo=i(P),Oe=m(P,"P",{"data-svelte-h":!0}),w(Oe)!=="svelte-q52n56"&&(Oe.innerHTML=Eo),no=i(P),De=m(P,"P",{"data-svelte-h":!0}),w(De)!=="svelte-hswkmf"&&(De.innerHTML=Qo),so=i(P),V=m(P,"DIV",{class:!0});var H=B(V);_(Ie.$$.fragment,H),ao=i(H),Ke=m(H,"P",{"data-svelte-h":!0}),w(Ke)!=="svelte-je4trr"&&(Ke.innerHTML=Ho),ro=i(H),_(ae.$$.fragment,H),io=i(H),_(re.$$.fragment,H),H.forEach(a),P.forEach(a),It=i(e),_(ze.$$.fragment,e),zt=i(e),$=m(e,"DIV",{class:!0});var J=B($);_(Ue.$$.fragment,J),lo=i(J),et=m(J,"P",{"data-svelte-h":!0}),w(et)!=="svelte-17hgce5"&&(et.textContent=Lo),co=i(J),tt=m(J,"P",{"data-svelte-h":!0}),w(tt)!=="svelte-fyfrzf"&&(tt.innerHTML=So),po=i(J),ot=m(J,"P",{"data-svelte-h":!0}),w(ot)!=="svelte-10ugs3m"&&(ot.innerHTML=Ao),uo=i(J),nt=m(J,"P",{"data-svelte-h":!0}),w(nt)!=="svelte-q52n56"&&(nt.innerHTML=Yo),mo=i(J),st=m(J,"P",{"data-svelte-h":!0}),w(st)!=="svelte-hswkmf"&&(st.innerHTML=Oo),ho=i(J),I=m(J,"DIV",{class:!0});var W=B(I);_(Fe.$$.fragment,W),fo=i(W),at=m(W,"P",{"data-svelte-h":!0}),w(at)!=="svelte-1k20ek3"&&(at.innerHTML=Do),go=i(W),_(ie.$$.fragment,W),_o=i(W),_(le.$$.fragment,W),To=i(W),_(de.$$.fragment,W),W.forEach(a),J.forEach(a),Ut=i(e),_(Pe.$$.fragment,e),Ft=i(e),j=m(e,"DIV",{class:!0});var Z=B(j);_(We.$$.fragment,Z),yo=i(Z),rt=m(Z,"P",{"data-svelte-h":!0}),w(rt)!=="svelte-i2hlyh"&&(rt.textContent=Ko),bo=i(Z),it=m(Z,"P",{"data-svelte-h":!0}),w(it)!=="svelte-q52n56"&&(it.innerHTML=en),Mo=i(Z),lt=m(Z,"P",{"data-svelte-h":!0}),w(lt)!=="svelte-hswkmf"&&(lt.innerHTML=tn),wo=i(Z),X=m(Z,"DIV",{class:!0});var L=B(X);_(Ze.$$.fragment,L),vo=i(L),dt=m(L,"P",{"data-svelte-h":!0}),w(dt)!=="svelte-ognm9t"&&(dt.innerHTML=on),ko=i(L),_(ce.$$.fragment,L),$o=i(L),_(pe.$$.fragment,L),L.forEach(a),Z.forEach(a),Pt=i(e),_(Be.$$.fragment,e),Wt=i(e),mt=m(e,"P",{}),B(mt).forEach(a),this.h()},h(){q(t,"name","hf:doc:metadata"),q(t,"content",xn),pn(k,"float","right"),q(z,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),q(Y,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),q(C,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),q(R,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),q(G,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),q(V,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),q(N,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),q(I,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),q($,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),q(X,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),q(j,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,s){d(document.head,t),c(e,p,s),c(e,o,s),c(e,l,s),c(e,f,s),c(e,h,s),c(e,k,s),c(e,me,s),T(A,e,s),c(e,ht,s),c(e,he,s),c(e,ft,s),c(e,fe,s),c(e,gt,s),T(K,e,s),c(e,_t,s),c(e,ge,s),c(e,Tt,s),T(ee,e,s),c(e,yt,s),c(e,_e,s),c(e,bt,s),c(e,Te,s),c(e,Mt,s),T(ye,e,s),c(e,wt,s),T(be,e,s),c(e,vt,s),c(e,Me,s),c(e,kt,s),T(we,e,s),c(e,$t,s),c(e,z,s),T(ve,z,null),d(z,Bt),d(z,qe),d(z,qt),d(z,Re),d(z,Rt),T(te,z,null),c(e,Jt,s),T(ke,e,s),c(e,Ct,s),c(e,C,s),T($e,C,null),d(C,Vt),d(C,Ve),d(C,Xt),d(C,Xe),d(C,Et),d(C,Ee),d(C,Qt),d(C,Y),T(Je,Y,null),d(Y,Ht),d(Y,Qe),d(Y,Lt),T(oe,Y,null),c(e,Gt,s),T(Ce,e,s),c(e,Nt,s),c(e,G,s),T(Ge,G,null),d(G,St),d(G,He),d(G,At),d(G,Le),d(G,Yt),d(G,Se),d(G,Ot),d(G,R),T(Ne,R,null),d(R,Dt),d(R,Ae),d(R,Kt),T(ne,R,null),d(R,eo),T(se,R,null),c(e,jt,s),T(je,e,s),c(e,xt,s),c(e,N,s),T(xe,N,null),d(N,to),d(N,Ye),d(N,oo),d(N,Oe),d(N,no),d(N,De),d(N,so),d(N,V),T(Ie,V,null),d(V,ao),d(V,Ke),d(V,ro),T(ae,V,null),d(V,io),T(re,V,null),c(e,It,s),T(ze,e,s),c(e,zt,s),c(e,$,s),T(Ue,$,null),d($,lo),d($,et),d($,co),d($,tt),d($,po),d($,ot),d($,uo),d($,nt),d($,mo),d($,st),d($,ho),d($,I),T(Fe,I,null),d(I,fo),d(I,at),d(I,go),T(ie,I,null),d(I,_o),T(le,I,null),d(I,To),T(de,I,null),c(e,Ut,s),T(Pe,e,s),c(e,Ft,s),c(e,j,s),T(We,j,null),d(j,yo),d(j,rt),d(j,bo),d(j,it),d(j,Mo),d(j,lt),d(j,wo),d(j,X),T(Ze,X,null),d(X,vo),d(X,dt),d(X,ko),T(ce,X,null),d(X,$o),T(pe,X,null),c(e,Pt,s),T(Be,e,s),c(e,Wt,s),c(e,mt,s),Zt=!0},p(e,[s]){const E={};s&2&&(E.$$scope={dirty:s,ctx:e}),K.$set(E);const U={};s&2&&(U.$$scope={dirty:s,ctx:e}),ee.$set(U);const D={};s&2&&(D.$$scope={dirty:s,ctx:e}),te.$set(D);const F={};s&2&&(F.$$scope={dirty:s,ctx:e}),oe.$set(F);const Q={};s&2&&(Q.$$scope={dirty:s,ctx:e}),ne.$set(Q);const P={};s&2&&(P.$$scope={dirty:s,ctx:e}),se.$set(P);const H={};s&2&&(H.$$scope={dirty:s,ctx:e}),ae.$set(H);const J={};s&2&&(J.$$scope={dirty:s,ctx:e}),re.$set(J);const W={};s&2&&(W.$$scope={dirty:s,ctx:e}),ie.$set(W);const Z={};s&2&&(Z.$$scope={dirty:s,ctx:e}),le.$set(Z);const L={};s&2&&(L.$$scope={dirty:s,ctx:e}),de.$set(L);const nn={};s&2&&(nn.$$scope={dirty:s,ctx:e}),ce.$set(nn);const sn={};s&2&&(sn.$$scope={dirty:s,ctx:e}),pe.$set(sn)},i(e){Zt||(y(A.$$.fragment,e),y(K.$$.fragment,e),y(ee.$$.fragment,e),y(ye.$$.fragment,e),y(be.$$.fragment,e),y(we.$$.fragment,e),y(ve.$$.fragment,e),y(te.$$.fragment,e),y(ke.$$.fragment,e),y($e.$$.fragment,e),y(Je.$$.fragment,e),y(oe.$$.fragment,e),y(Ce.$$.fragment,e),y(Ge.$$.fragment,e),y(Ne.$$.fragment,e),y(ne.$$.fragment,e),y(se.$$.fragment,e),y(je.$$.fragment,e),y(xe.$$.fragment,e),y(Ie.$$.fragment,e),y(ae.$$.fragment,e),y(re.$$.fragment,e),y(ze.$$.fragment,e),y(Ue.$$.fragment,e),y(Fe.$$.fragment,e),y(ie.$$.fragment,e),y(le.$$.fragment,e),y(de.$$.fragment,e),y(Pe.$$.fragment,e),y(We.$$.fragment,e),y(Ze.$$.fragment,e),y(ce.$$.fragment,e),y(pe.$$.fragment,e),y(Be.$$.fragment,e),Zt=!0)},o(e){b(A.$$.fragment,e),b(K.$$.fragment,e),b(ee.$$.fragment,e),b(ye.$$.fragment,e),b(be.$$.fragment,e),b(we.$$.fragment,e),b(ve.$$.fragment,e),b(te.$$.fragment,e),b(ke.$$.fragment,e),b($e.$$.fragment,e),b(Je.$$.fragment,e),b(oe.$$.fragment,e),b(Ce.$$.fragment,e),b(Ge.$$.fragment,e),b(Ne.$$.fragment,e),b(ne.$$.fragment,e),b(se.$$.fragment,e),b(je.$$.fragment,e),b(xe.$$.fragment,e),b(Ie.$$.fragment,e),b(ae.$$.fragment,e),b(re.$$.fragment,e),b(ze.$$.fragment,e),b(Ue.$$.fragment,e),b(Fe.$$.fragment,e),b(ie.$$.fragment,e),b(le.$$.fragment,e),b(de.$$.fragment,e),b(Pe.$$.fragment,e),b(We.$$.fragment,e),b(Ze.$$.fragment,e),b(ce.$$.fragment,e),b(pe.$$.fragment,e),b(Be.$$.fragment,e),Zt=!1},d(e){e&&(a(p),a(o),a(l),a(f),a(h),a(k),a(me),a(ht),a(he),a(ft),a(fe),a(gt),a(_t),a(ge),a(Tt),a(yt),a(_e),a(bt),a(Te),a(Mt),a(wt),a(vt),a(Me),a(kt),a($t),a(z),a(Jt),a(Ct),a(C),a(Gt),a(Nt),a(G),a(jt),a(xt),a(N),a(It),a(zt),a($),a(Ut),a(Ft),a(j),a(Pt),a(Wt),a(mt)),a(t),M(A,e),M(K,e),M(ee,e),M(ye,e),M(be,e),M(we,e),M(ve),M(te),M(ke,e),M($e),M(Je),M(oe),M(Ce,e),M(Ge),M(Ne),M(ne),M(se),M(je,e),M(xe),M(Ie),M(ae),M(re),M(ze,e),M(Ue),M(Fe),M(ie),M(le),M(de),M(Pe,e),M(We),M(Ze),M(ce),M(pe),M(Be,e)}}}const xn='{"title":"GPT-Neo","local":"gpt-neo","sections":[],"depth":2}';function In(v){return rn(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class Rn extends ln{constructor(t){super(),dn(this,t,In,jn,an,{})}}export{Rn as component};
