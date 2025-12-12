import{s as oa,o as na,n as W}from"../chunks/scheduler.18a86fab.js";import{S as sa,i as aa,g as l,s as n,r as u,A as ra,h as d,f as r,c as s,j as C,x as m,u as h,k as x,l as ia,y as t,a as c,v as f,d as g,t as _,w as y}from"../chunks/index.98837b22.js";import{T as co}from"../chunks/Tip.77304350.js";import{D as $}from"../chunks/Docstring.a1ef7999.js";import{C as ye}from"../chunks/CodeBlock.8d0c2e8a.js";import{E as as}from"../chunks/ExampleCodeBlock.8c3ee1f9.js";import{H as _e,E as la}from"../chunks/getInferenceSnippets.06c2775f.js";import{H as da,a as rs}from"../chunks/HfOption.6641485e.js";function ca(w){let o,b="Click on the Mistral models in the right sidebar for more examples of how to apply Mistral to different language tasks.";return{c(){o=l("p"),o.textContent=b},l(a){o=d(a,"P",{"data-svelte-h":!0}),m(o)!=="svelte-ceo0jh"&&(o.textContent=b)},m(a,v){c(a,o,v)},p:W,d(a){a&&r(o)}}}function ma(w){let o,b;return o=new ye({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwcGlwZWxpbmUlMEElMEFtZXNzYWdlcyUyMCUzRCUyMCU1QiUwQSUyMCUyMCUyMCUyMCU3QiUyMnJvbGUlMjIlM0ElMjAlMjJ1c2VyJTIyJTJDJTIwJTIyY29udGVudCUyMiUzQSUyMCUyMldoYXQlMjBpcyUyMHlvdXIlMjBmYXZvdXJpdGUlMjBjb25kaW1lbnQlM0YlMjIlN0QlMkMlMEElMjAlMjAlMjAlMjAlN0IlMjJyb2xlJTIyJTNBJTIwJTIyYXNzaXN0YW50JTIyJTJDJTIwJTIyY29udGVudCUyMiUzQSUyMCUyMldlbGwlMkMlMjBJJ20lMjBxdWl0ZSUyMHBhcnRpYWwlMjB0byUyMGElMjBnb29kJTIwc3F1ZWV6ZSUyMG9mJTIwZnJlc2glMjBsZW1vbiUyMGp1aWNlLiUyMEl0JTIwYWRkcyUyMGp1c3QlMjB0aGUlMjByaWdodCUyMGFtb3VudCUyMG9mJTIwemVzdHklMjBmbGF2b3VyJTIwdG8lMjB3aGF0ZXZlciUyMEknbSUyMGNvb2tpbmclMjB1cCUyMGluJTIwdGhlJTIwa2l0Y2hlbiElMjIlN0QlMkMlMEElMjAlMjAlMjAlMjAlN0IlMjJyb2xlJTIyJTNBJTIwJTIydXNlciUyMiUyQyUyMCUyMmNvbnRlbnQlMjIlM0ElMjAlMjJEbyUyMHlvdSUyMGhhdmUlMjBtYXlvbm5haXNlJTIwcmVjaXBlcyUzRiUyMiU3RCUwQSU1RCUwQSUwQWNoYXRib3QlMjAlM0QlMjBwaXBlbGluZSglMjJ0ZXh0LWdlbmVyYXRpb24lMjIlMkMlMjBtb2RlbCUzRCUyMm1pc3RyYWxhaSUyRk1pc3RyYWwtN0ItSW5zdHJ1Y3QtdjAuMyUyMiUyQyUyMGR0eXBlJTNEdG9yY2guYmZsb2F0MTYlMkMlMjBkZXZpY2UlM0QwKSUwQWNoYXRib3QobWVzc2FnZXMp",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> pipeline

<span class="hljs-meta">&gt;&gt;&gt; </span>messages = [
<span class="hljs-meta">... </span>    {<span class="hljs-string">&quot;role&quot;</span>: <span class="hljs-string">&quot;user&quot;</span>, <span class="hljs-string">&quot;content&quot;</span>: <span class="hljs-string">&quot;What is your favourite condiment?&quot;</span>},
<span class="hljs-meta">... </span>    {<span class="hljs-string">&quot;role&quot;</span>: <span class="hljs-string">&quot;assistant&quot;</span>, <span class="hljs-string">&quot;content&quot;</span>: <span class="hljs-string">&quot;Well, I&#x27;m quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I&#x27;m cooking up in the kitchen!&quot;</span>},
<span class="hljs-meta">... </span>    {<span class="hljs-string">&quot;role&quot;</span>: <span class="hljs-string">&quot;user&quot;</span>, <span class="hljs-string">&quot;content&quot;</span>: <span class="hljs-string">&quot;Do you have mayonnaise recipes?&quot;</span>}
<span class="hljs-meta">... </span>]

<span class="hljs-meta">&gt;&gt;&gt; </span>chatbot = pipeline(<span class="hljs-string">&quot;text-generation&quot;</span>, model=<span class="hljs-string">&quot;mistralai/Mistral-7B-Instruct-v0.3&quot;</span>, dtype=torch.bfloat16, device=<span class="hljs-number">0</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>chatbot(messages)`,wrap:!1}}),{c(){u(o.$$.fragment)},l(a){h(o.$$.fragment,a)},m(a,v){f(o,a,v),b=!0},p:W,i(a){b||(g(o.$$.fragment,a),b=!0)},o(a){_(o.$$.fragment,a),b=!1},d(a){y(o,a)}}}function pa(w){let o,b;return o=new ye({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b01vZGVsRm9yQ2F1c2FsTE0lMkMlMjBBdXRvVG9rZW5pemVyJTBBJTBBbW9kZWwlMjAlM0QlMjBBdXRvTW9kZWxGb3JDYXVzYWxMTS5mcm9tX3ByZXRyYWluZWQoJTIybWlzdHJhbGFpJTJGTWlzdHJhbC03Qi1JbnN0cnVjdC12MC4zJTIyJTJDJTIwZHR5cGUlM0R0b3JjaC5iZmxvYXQxNiUyQyUyMGF0dG5faW1wbGVtZW50YXRpb24lM0QlMjJzZHBhJTIyJTJDJTIwZGV2aWNlX21hcCUzRCUyMmF1dG8lMjIpJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIybWlzdHJhbGFpJTJGTWlzdHJhbC03Qi1JbnN0cnVjdC12MC4zJTIyKSUwQSUwQW1lc3NhZ2VzJTIwJTNEJTIwJTVCJTBBJTIwJTIwJTIwJTIwJTdCJTIycm9sZSUyMiUzQSUyMCUyMnVzZXIlMjIlMkMlMjAlMjJjb250ZW50JTIyJTNBJTIwJTIyV2hhdCUyMGlzJTIweW91ciUyMGZhdm91cml0ZSUyMGNvbmRpbWVudCUzRiUyMiU3RCUyQyUwQSUyMCUyMCUyMCUyMCU3QiUyMnJvbGUlMjIlM0ElMjAlMjJhc3Npc3RhbnQlMjIlMkMlMjAlMjJjb250ZW50JTIyJTNBJTIwJTIyV2VsbCUyQyUyMEknbSUyMHF1aXRlJTIwcGFydGlhbCUyMHRvJTIwYSUyMGdvb2QlMjBzcXVlZXplJTIwb2YlMjBmcmVzaCUyMGxlbW9uJTIwanVpY2UuJTIwSXQlMjBhZGRzJTIwanVzdCUyMHRoZSUyMHJpZ2h0JTIwYW1vdW50JTIwb2YlMjB6ZXN0eSUyMGZsYXZvdXIlMjB0byUyMHdoYXRldmVyJTIwSSdtJTIwY29va2luZyUyMHVwJTIwaW4lMjB0aGUlMjBraXRjaGVuISUyMiU3RCUyQyUwQSUyMCUyMCUyMCUyMCU3QiUyMnJvbGUlMjIlM0ElMjAlMjJ1c2VyJTIyJTJDJTIwJTIyY29udGVudCUyMiUzQSUyMCUyMkRvJTIweW91JTIwaGF2ZSUyMG1heW9ubmFpc2UlMjByZWNpcGVzJTNGJTIyJTdEJTBBJTVEJTBBJTBBbW9kZWxfaW5wdXRzJTIwJTNEJTIwdG9rZW5pemVyLmFwcGx5X2NoYXRfdGVtcGxhdGUobWVzc2FnZXMlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKS50byhtb2RlbC5kZXZpY2UpJTBBJTBBZ2VuZXJhdGVkX2lkcyUyMCUzRCUyMG1vZGVsLmdlbmVyYXRlKG1vZGVsX2lucHV0cyUyQyUyMG1heF9uZXdfdG9rZW5zJTNEMTAwJTJDJTIwZG9fc2FtcGxlJTNEVHJ1ZSklMEF0b2tlbml6ZXIuYmF0Y2hfZGVjb2RlKGdlbmVyYXRlZF9pZHMpJTVCMCU1RA==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoModelForCausalLM, AutoTokenizer

<span class="hljs-meta">&gt;&gt;&gt; </span>model = AutoModelForCausalLM.from_pretrained(<span class="hljs-string">&quot;mistralai/Mistral-7B-Instruct-v0.3&quot;</span>, dtype=torch.bfloat16, attn_implementation=<span class="hljs-string">&quot;sdpa&quot;</span>, device_map=<span class="hljs-string">&quot;auto&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;mistralai/Mistral-7B-Instruct-v0.3&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>messages = [
<span class="hljs-meta">... </span>    {<span class="hljs-string">&quot;role&quot;</span>: <span class="hljs-string">&quot;user&quot;</span>, <span class="hljs-string">&quot;content&quot;</span>: <span class="hljs-string">&quot;What is your favourite condiment?&quot;</span>},
<span class="hljs-meta">... </span>    {<span class="hljs-string">&quot;role&quot;</span>: <span class="hljs-string">&quot;assistant&quot;</span>, <span class="hljs-string">&quot;content&quot;</span>: <span class="hljs-string">&quot;Well, I&#x27;m quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I&#x27;m cooking up in the kitchen!&quot;</span>},
<span class="hljs-meta">... </span>    {<span class="hljs-string">&quot;role&quot;</span>: <span class="hljs-string">&quot;user&quot;</span>, <span class="hljs-string">&quot;content&quot;</span>: <span class="hljs-string">&quot;Do you have mayonnaise recipes?&quot;</span>}
<span class="hljs-meta">... </span>]

<span class="hljs-meta">&gt;&gt;&gt; </span>model_inputs = tokenizer.apply_chat_template(messages, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>).to(model.device)

<span class="hljs-meta">&gt;&gt;&gt; </span>generated_ids = model.generate(model_inputs, max_new_tokens=<span class="hljs-number">100</span>, do_sample=<span class="hljs-literal">True</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer.batch_decode(generated_ids)[<span class="hljs-number">0</span>]
<span class="hljs-string">&quot;Mayonnaise can be made as follows: (...)&quot;</span>`,wrap:!1}}),{c(){u(o.$$.fragment)},l(a){h(o.$$.fragment,a)},m(a,v){f(o,a,v),b=!0},p:W,i(a){b||(g(o.$$.fragment,a),b=!0)},o(a){_(o.$$.fragment,a),b=!1},d(a){y(o,a)}}}function ua(w){let o,b;return o=new ye({props:{code:"ZWNobyUyMC1lJTIwJTIyTXklMjBmYXZvcml0ZSUyMGNvbmRpbWVudCUyMGlzJTIyJTIwJTdDJTIwdHJhbnNmb3JtZXJzJTIwY2hhdCUyMG1pc3RyYWxhaSUyRk1pc3RyYWwtN0ItdjAuMyUyMC0tZHR5cGUlMjBhdXRvJTIwLS1kZXZpY2UlMjAwJTIwLS1hdHRuX2ltcGxlbWVudGF0aW9uJTIwZmxhc2hfYXR0ZW50aW9uXzI=",highlighted:'echo -e <span class="hljs-string">&quot;My favorite condiment is&quot;</span> | transformers chat mistralai/Mistral-7B-v0<span class="hljs-number">.3</span> --dtype auto --device <span class="hljs-number">0</span> --attn_implementation flash_attention_2',wrap:!1}}),{c(){u(o.$$.fragment)},l(a){h(o.$$.fragment,a)},m(a,v){f(o,a,v),b=!0},p:W,i(a){b||(g(o.$$.fragment,a),b=!0)},o(a){_(o.$$.fragment,a),b=!1},d(a){y(o,a)}}}function ha(w){let o,b,a,v,z,M;return o=new rs({props:{id:"usage",option:"Pipeline",$$slots:{default:[ma]},$$scope:{ctx:w}}}),a=new rs({props:{id:"usage",option:"AutoModel",$$slots:{default:[pa]},$$scope:{ctx:w}}}),z=new rs({props:{id:"usage",option:"transformers CLI",$$slots:{default:[ua]},$$scope:{ctx:w}}}),{c(){u(o.$$.fragment),b=n(),u(a.$$.fragment),v=n(),u(z.$$.fragment)},l(T){h(o.$$.fragment,T),b=s(T),h(a.$$.fragment,T),v=s(T),h(z.$$.fragment,T)},m(T,U){f(o,T,U),c(T,b,U),f(a,T,U),c(T,v,U),f(z,T,U),M=!0},p(T,U){const mo={};U&2&&(mo.$$scope={dirty:U,ctx:T}),o.$set(mo);const be={};U&2&&(be.$$scope={dirty:U,ctx:T}),a.$set(be);const N={};U&2&&(N.$$scope={dirty:U,ctx:T}),z.$set(N)},i(T){M||(g(o.$$.fragment,T),g(a.$$.fragment,T),g(z.$$.fragment,T),M=!0)},o(T){_(o.$$.fragment,T),_(a.$$.fragment,T),_(z.$$.fragment,T),M=!1},d(T){T&&(r(b),r(v)),y(o,T),y(a,T),y(z,T)}}}function fa(w){let o,b;return o=new ye({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyME1pc3RyYWxNb2RlbCUyQyUyME1pc3RyYWxDb25maWclMEElMEElMjMlMjBJbml0aWFsaXppbmclMjBhJTIwTWlzdHJhbCUyMDdCJTIwc3R5bGUlMjBjb25maWd1cmF0aW9uJTBBY29uZmlndXJhdGlvbiUyMCUzRCUyME1pc3RyYWxDb25maWcoKSUwQSUwQSUyMyUyMEluaXRpYWxpemluZyUyMGElMjBtb2RlbCUyMGZyb20lMjB0aGUlMjBNaXN0cmFsJTIwN0IlMjBzdHlsZSUyMGNvbmZpZ3VyYXRpb24lMEFtb2RlbCUyMCUzRCUyME1pc3RyYWxNb2RlbChjb25maWd1cmF0aW9uKSUwQSUwQSUyMyUyMEFjY2Vzc2luZyUyMHRoZSUyMG1vZGVsJTIwY29uZmlndXJhdGlvbiUwQWNvbmZpZ3VyYXRpb24lMjAlM0QlMjBtb2RlbC5jb25maWc=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> MistralModel, MistralConfig

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a Mistral 7B style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = MistralConfig()

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a model from the Mistral 7B style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = MistralModel(configuration)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Accessing the model configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = model.config`,wrap:!1}}),{c(){u(o.$$.fragment)},l(a){h(o.$$.fragment,a)},m(a,v){f(o,a,v),b=!0},p:W,i(a){b||(g(o.$$.fragment,a),b=!0)},o(a){_(o.$$.fragment,a),b=!1},d(a){y(o,a)}}}function ga(w){let o,b="<code>mistral-common</code> is the official tokenizer library for Mistral AI models. To use it, you need to install it with:",a,v,z;return v=new ye({props:{code:"cGlwJTIwaW5zdGFsbCUyMHRyYW5zZm9ybWVycyU1Qm1pc3RyYWwtY29tbW9uJTVE",highlighted:"pip install transformers[mistral-common]",wrap:!1}}),{c(){o=l("p"),o.innerHTML=b,a=n(),u(v.$$.fragment)},l(M){o=d(M,"P",{"data-svelte-h":!0}),m(o)!=="svelte-m7z88d"&&(o.innerHTML=b),a=s(M),h(v.$$.fragment,M)},m(M,T){c(M,o,T),c(M,a,T),f(v,M,T),z=!0},p:W,i(M){z||(g(v.$$.fragment,M),z=!0)},o(M){_(v.$$.fragment,M),z=!1},d(M){M&&(r(o),r(a)),y(v,M)}}}function _a(w){let o,b=`If the <code>encoded_inputs</code> passed are dictionary of numpy arrays, PyTorch tensors, the
result will use the same type unless you provide a different tensor type with <code>return_tensors</code>. In the case of
PyTorch tensors, you will lose the specific device of your tensors however.`;return{c(){o=l("p"),o.innerHTML=b},l(a){o=d(a,"P",{"data-svelte-h":!0}),m(o)!=="svelte-mer66"&&(o.innerHTML=b)},m(a,v){c(a,o,v)},p:W,d(a){a&&r(o)}}}function ya(w){let o,b=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){o=l("p"),o.innerHTML=b},l(a){o=d(a,"P",{"data-svelte-h":!0}),m(o)!=="svelte-fincs2"&&(o.innerHTML=b)},m(a,v){c(a,o,v)},p:W,d(a){a&&r(o)}}}function ba(w){let o,b=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){o=l("p"),o.innerHTML=b},l(a){o=d(a,"P",{"data-svelte-h":!0}),m(o)!=="svelte-fincs2"&&(o.innerHTML=b)},m(a,v){c(a,o,v)},p:W,d(a){a&&r(o)}}}function ka(w){let o,b="Example:",a,v,z;return v=new ye({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBNaXN0cmFsRm9yQ2F1c2FsTE0lMEElMEFtb2RlbCUyMCUzRCUyME1pc3RyYWxGb3JDYXVzYWxMTS5mcm9tX3ByZXRyYWluZWQoJTIybWV0YS1taXN0cmFsJTJGTWlzdHJhbC0yLTdiLWhmJTIyKSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMm1ldGEtbWlzdHJhbCUyRk1pc3RyYWwtMi03Yi1oZiUyMiklMEElMEFwcm9tcHQlMjAlM0QlMjAlMjJIZXklMkMlMjBhcmUlMjB5b3UlMjBjb25zY2lvdXMlM0YlMjBDYW4lMjB5b3UlMjB0YWxrJTIwdG8lMjBtZSUzRiUyMiUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplcihwcm9tcHQlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQSUwQSUyMyUyMEdlbmVyYXRlJTBBZ2VuZXJhdGVfaWRzJTIwJTNEJTIwbW9kZWwuZ2VuZXJhdGUoaW5wdXRzLmlucHV0X2lkcyUyQyUyMG1heF9sZW5ndGglM0QzMCklMEF0b2tlbml6ZXIuYmF0Y2hfZGVjb2RlKGdlbmVyYXRlX2lkcyUyQyUyMHNraXBfc3BlY2lhbF90b2tlbnMlM0RUcnVlJTJDJTIwY2xlYW5fdXBfdG9rZW5pemF0aW9uX3NwYWNlcyUzREZhbHNlKSU1QjAlNUQ=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, MistralForCausalLM

<span class="hljs-meta">&gt;&gt;&gt; </span>model = MistralForCausalLM.from_pretrained(<span class="hljs-string">&quot;meta-mistral/Mistral-2-7b-hf&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;meta-mistral/Mistral-2-7b-hf&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>prompt = <span class="hljs-string">&quot;Hey, are you conscious? Can you talk to me?&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(prompt, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Generate</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>generate_ids = model.generate(inputs.input_ids, max_length=<span class="hljs-number">30</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer.batch_decode(generate_ids, skip_special_tokens=<span class="hljs-literal">True</span>, clean_up_tokenization_spaces=<span class="hljs-literal">False</span>)[<span class="hljs-number">0</span>]
<span class="hljs-string">&quot;Hey, are you conscious? Can you talk to me?\\nI&#x27;m not conscious, but I can talk to you.&quot;</span>`,wrap:!1}}),{c(){o=l("p"),o.textContent=b,a=n(),u(v.$$.fragment)},l(M){o=d(M,"P",{"data-svelte-h":!0}),m(o)!=="svelte-11lpom8"&&(o.textContent=b),a=s(M),h(v.$$.fragment,M)},m(M,T){c(M,o,T),c(M,a,T),f(v,M,T),z=!0},p:W,i(M){z||(g(v.$$.fragment,M),z=!0)},o(M){_(v.$$.fragment,M),z=!1},d(M){M&&(r(o),r(a)),y(v,M)}}}function va(w){let o,b=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){o=l("p"),o.innerHTML=b},l(a){o=d(a,"P",{"data-svelte-h":!0}),m(o)!=="svelte-fincs2"&&(o.innerHTML=b)},m(a,v){c(a,o,v)},p:W,d(a){a&&r(o)}}}function Ta(w){let o,b=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){o=l("p"),o.innerHTML=b},l(a){o=d(a,"P",{"data-svelte-h":!0}),m(o)!=="svelte-fincs2"&&(o.innerHTML=b)},m(a,v){c(a,o,v)},p:W,d(a){a&&r(o)}}}function Ma(w){let o,b,a,v,z,M="<em>This model was released on 2023-10-10 and added to Hugging Face Transformers on 2023-09-27.</em>",T,U,mo='<div class="flex flex-wrap space-x-1"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/> <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat"/> <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/> <img alt="Tensor parallelism" src="https://img.shields.io/badge/Tensor%20parallelism-06b6d4?style=flat&amp;logoColor=white"/></div>',be,N,uo,ke,is=`<a href="https://huggingface.co/papers/2310.06825" rel="nofollow">Mistral</a> is a 7B parameter language model, available as a pretrained and instruction-tuned variant, focused on balancing
the scaling costs of large models with performance and efficient inference. This model uses sliding window attention (SWA) trained with a 8K context length and a fixed cache size to handle longer sequences more effectively. Grouped-query attention (GQA) speeds up inference and reduces memory requirements. Mistral also features a byte-fallback BPE tokenizer to improve token handling and efficiency by ensuring characters are never mapped to out-of-vocabulary tokens.`,ho,ve,ls='You can find all the original Mistral checkpoints under the <a href="https://huggingface.co/mistralai" rel="nofollow">Mistral AI_</a> organization.',fo,D,go,Te,ds='The example below demonstrates how to chat with <a href="/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.Pipeline">Pipeline</a> or the <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel">AutoModel</a>, and from the command line.',_o,A,yo,Me,cs='Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the <a href="../quantization/overview">Quantization</a> overview for more available quantization backends.',bo,we,ms='The example below uses <a href="../quantization/bitsandbytes">bitsandbytes</a> to only quantize the weights to 4-bits.',ko,xe,vo,Ce,ps='Use the <a href="https://github.com/huggingface/transformers/blob/beb9b5b02246b9b7ee81ddf938f93f44cfeaad19/src/transformers/utils/attention_visualizer.py#L139" rel="nofollow">AttentionMaskVisualizer</a> to better understand what tokens the model can and cannot attend to.',To,ze,Mo,Y,us='<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/mistral-attn-mask.png"/>',wo,$e,xo,I,Ue,So,_t,hs=`This is the configuration class to store the configuration of a <a href="/docs/transformers/v4.56.2/en/model_doc/mistral#transformers.MistralModel">MistralModel</a>. It is used to instantiate an
Mistral model according to the specified arguments, defining the model architecture. Instantiating a configuration
with the defaults will yield a similar configuration to that of the Mistral-7B-v0.1 or Mistral-7B-Instruct-v0.1.`,Eo,yt,fs='<a href="https://huggingface.co/mistralai/Mistral-7B-v0.1" rel="nofollow">mistralai/Mistral-7B-v0.1</a> <a href="https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1" rel="nofollow">mistralai/Mistral-7B-Instruct-v0.1</a>',Do,bt,gs=`Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> and can be used to control the model outputs. Read the
documentation from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> for more information.`,Ao,Q,Co,Ie,zo,p,je,Yo,kt,_s="Class to wrap <code>mistral-common</code> tokenizers.",Qo,O,Oo,vt,ys="Otherwise the tokenizer falls back to the Transformers implementation of the tokenizer.",Ko,Tt,bs='For more info on <code>mistral-common</code>, see <a href="https://github.com/mistralai/mistral-common" rel="nofollow">mistral-common</a>.',en,Mt,ks=`This class is a wrapper around a <code>mistral_common.tokens.tokenizers.mistral.MistralTokenizer</code>.
It provides a Hugging Face compatible interface to tokenize using the official mistral-common tokenizer.`,tn,wt,vs="Supports the following methods from the <code>PreTrainedTokenizerBase</code> class:",on,xt,Ts='<li><a href="/docs/transformers/v4.56.2/en/model_doc/pixtral#transformers.MistralCommonTokenizer.get_vocab">get_vocab()</a>: Returns the vocabulary as a dictionary of token to index.</li> <li><a href="/docs/transformers/v4.56.2/en/model_doc/pixtral#transformers.MistralCommonTokenizer.encode">encode()</a>: Encode a string to a list of integers.</li> <li><a href="/docs/transformers/v4.56.2/en/model_doc/pixtral#transformers.MistralCommonTokenizer.decode">decode()</a>: Decode a list of integers to a string.</li> <li><a href="/docs/transformers/v4.56.2/en/model_doc/pixtral#transformers.MistralCommonTokenizer.batch_decode">batch_decode()</a>: Decode a batch of list of integers to a list of strings.</li> <li><a href="/docs/transformers/v4.56.2/en/model_doc/pixtral#transformers.MistralCommonTokenizer.convert_tokens_to_ids">convert_tokens_to_ids()</a>: Convert a list of tokens to a list of integers.</li> <li><a href="/docs/transformers/v4.56.2/en/model_doc/pixtral#transformers.MistralCommonTokenizer.convert_ids_to_tokens">convert_ids_to_tokens()</a>: Convert a list of integers to a list of tokens.</li> <li><a href="/docs/transformers/v4.56.2/en/model_doc/pixtral#transformers.MistralCommonTokenizer.tokenize">tokenize()</a>: Tokenize a string.</li> <li><a href="/docs/transformers/v4.56.2/en/model_doc/pixtral#transformers.MistralCommonTokenizer.get_special_tokens_mask">get_special_tokens_mask()</a>: Get the special tokens mask for a list of tokens.</li> <li><a href="/docs/transformers/v4.56.2/en/model_doc/pixtral#transformers.MistralCommonTokenizer.prepare_for_model">prepare_for_model()</a>: Prepare a list of inputs for the model.</li> <li><a href="/docs/transformers/v4.56.2/en/model_doc/pixtral#transformers.MistralCommonTokenizer.pad">pad()</a>: Pad a list of inputs to the same length.</li> <li><a href="/docs/transformers/v4.56.2/en/model_doc/pixtral#transformers.MistralCommonTokenizer.truncate_sequences">truncate_sequences()</a>: Truncate a list of sequences to the same length.</li> <li><a href="/docs/transformers/v4.56.2/en/model_doc/pixtral#transformers.MistralCommonTokenizer.apply_chat_template">apply_chat_template()</a>: Apply a chat template to a list of messages.</li> <li><code>__call__()</code>: Tokenize a string or a list of strings.</li> <li><a href="/docs/transformers/v4.56.2/en/model_doc/pixtral#transformers.MistralCommonTokenizer.from_pretrained">from_pretrained()</a>: Download and cache a pretrained tokenizer from the Hugging Face model hub or local directory.</li> <li><a href="/docs/transformers/v4.56.2/en/model_doc/pixtral#transformers.MistralCommonTokenizer.save_pretrained">save_pretrained()</a>: Save a tokenizer to a directory, so it can be reloaded using the <code>from_pretrained</code> class method.</li> <li><a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.utils.PushToHubMixin.push_to_hub">push_to_hub()</a>: Upload tokenizer to the Hugging Face model hub.</li>',nn,Ct,Ms="Here are the key differences with the <code>PreTrainedTokenizerBase</code> class:",sn,zt,ws='<li>Pair of sequences are not supported. The signature have been kept for compatibility but all arguments related to pair of sequences are ignored. The return values of pairs are returned as <code>None</code>.</li> <li>The <code>is_split_into_words</code> argument is not supported.</li> <li>The <code>return_token_type_ids</code> argument is not supported.</li> <li>It is not possible to add new tokens to the tokenizer. Also the special tokens are handled differently from Transformers. In <code>mistral-common</code>, special tokens are never encoded directly. This means that: <code>tokenizer.encode(&quot;&lt;s&gt;&quot;)</code> will not return the ID of the <code>&lt;s&gt;</code> token. Instead, it will return a list of IDs corresponding to the tokenization of the string <code>&quot;&lt;s&gt;&quot;</code>. For more information, see the <a href="https://mistralai.github.io/mistral-common/usage/tokenizers/#special-tokens" rel="nofollow">mistral-common documentation</a>.</li>',an,$t,xs='If you have suggestions to improve this class, please open an issue on the <a href="https://github.com/mistralai/mistral-common/issues" rel="nofollow">mistral-common GitHub repository</a> if it is related to the tokenizer or on the <a href="https://github.com/huggingface/transformers/issues" rel="nofollow">Transformers GitHub repository</a> if it is related to the Hugging Face interface.',rn,K,Je,ln,Ut,Cs=`Converts a list of dictionaries with <code>&quot;role&quot;</code> and <code>&quot;content&quot;</code> keys to a list of token
ids.`,dn,ee,qe,cn,It,zs="Convert a list of lists of token ids into a list of strings by calling decode.",mn,te,Fe,pn,jt,$s=`Converts a single index or a sequence of indices in a token or a sequence of tokens, using the vocabulary and
added tokens.`,un,oe,Le,hn,Jt,Us=`Converts a token string (or a sequence of tokens) in a single integer id (or a sequence of ids), using the
vocabulary.`,fn,ne,We,gn,qt,Is=`Converts a sequence of ids in a string, using the tokenizer and vocabulary with options to remove special
tokens and clean up tokenization spaces.`,_n,se,Ne,yn,Ft,js="Converts a string to a sequence of ids (integer), using the tokenizer and vocabulary.",bn,ae,Be,kn,Lt,Js=`Instantiate a <code>MistralCommonTokenizer</code> from a predefined
tokenizer.`,vn,re,Ge,Tn,Wt,qs=`Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
special tokens using the tokenizer <code>prepare_for_model</code> or <code>encode_plus</code> methods.`,Mn,B,Pe,wn,Nt,Fs="Returns the vocabulary as a dictionary of token to index.",xn,Bt,Ls=`This is a lossy conversion. There may be multiple token ids that decode to the same
string due to partial UTF-8 byte sequences being converted to �.`,Cn,F,Ve,zn,Gt,Ws=`Pad a single encoded input or a batch of encoded inputs up to predefined length or to the max sequence length
in the batch.`,$n,Pt,Ns=`Padding side (left/right) padding token ids are defined at the tokenizer level (with <code>self.padding_side</code>,
<code>self.pad_token_id</code>).`,Un,ie,In,le,Ze,jn,Vt,Bs=`Prepares a sequence of input id so that it can be used by the model. It
adds special tokens, truncates sequences if overflowing while taking into account the special tokens and
manages a moving window (with user defined stride) for overflowing tokens.`,Jn,G,He,qn,Zt,Gs="Save the full tokenizer state.",Fn,Ht,Ps=`This method make sure the full tokenizer can then be re-loaded using the
<code>~MistralCommonTokenizer.tokenization_mistral_common.from_pretrained</code> class method.`,Ln,P,Re,Wn,Rt,Vs="Converts a string into a sequence of tokens, using the tokenizer.",Nn,Xt,Zs="Split in words for word-based vocabulary or sub-words for sub-word-based vocabularies.",Bn,de,Xe,Gn,St,Hs="Truncates a sequence pair in-place following the strategy.",$o,Se,Uo,j,Ee,Pn,Et,Rs="The bare Mistral Model outputting raw hidden-states without any specific head on top.",Vn,Dt,Xs=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Zn,At,Ss=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Hn,V,De,Rn,Yt,Es='The <a href="/docs/transformers/v4.56.2/en/model_doc/mistral#transformers.MistralModel">MistralModel</a> forward method, overrides the <code>__call__</code> special method.',Xn,ce,Io,Ae,jo,J,Ye,Sn,Qt,Ds="The Mistral Model for causal language modeling.",En,Ot,As=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Dn,Kt,Ys=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,An,L,Qe,Yn,eo,Qs='The <a href="/docs/transformers/v4.56.2/en/model_doc/mistral#transformers.MistralForCausalLM">MistralForCausalLM</a> forward method, overrides the <code>__call__</code> special method.',Qn,me,On,pe,Jo,Oe,qo,S,Ke,Kn,Z,et,es,to,Os="The <code>GenericForSequenceClassification</code> forward method, overrides the <code>__call__</code> special method.",ts,ue,Fo,tt,Lo,E,ot,os,H,nt,ns,oo,Ks="The <code>GenericForTokenClassification</code> forward method, overrides the <code>__call__</code> special method.",ss,he,Wo,st,No,at,rt,Bo,it,ea="<li>forward</li>",Go,lt,Po,po,Vo;return N=new _e({props:{title:"Mistral",local:"mistral",headingTag:"h1"}}),D=new co({props:{warning:!1,$$slots:{default:[ca]},$$scope:{ctx:w}}}),A=new da({props:{id:"usage",options:["Pipeline","AutoModel","transformers CLI"],$$slots:{default:[ha]},$$scope:{ctx:w}}}),xe=new ye({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b01vZGVsRm9yQ2F1c2FsTE0lMkMlMjBBdXRvVG9rZW5pemVyJTJDJTIwQml0c0FuZEJ5dGVzQ29uZmlnJTBBJTBBJTIzJTIwc3BlY2lmeSUyMGhvdyUyMHRvJTIwcXVhbnRpemUlMjB0aGUlMjBtb2RlbCUwQXF1YW50aXphdGlvbl9jb25maWclMjAlM0QlMjBCaXRzQW5kQnl0ZXNDb25maWcoJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwbG9hZF9pbl80Yml0JTNEVHJ1ZSUyQyUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMGJuYl80Yml0X3F1YW50X3R5cGUlM0QlMjJuZjQlMjIlMkMlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjBibmJfNGJpdF9jb21wdXRlX2R0eXBlJTNEJTIydG9yY2guZmxvYXQxNiUyMiUyQyUwQSklMEElMEFtb2RlbCUyMCUzRCUyMEF1dG9Nb2RlbEZvckNhdXNhbExNLmZyb21fcHJldHJhaW5lZCglMjJtaXN0cmFsYWklMkZNaXN0cmFsLTdCLUluc3RydWN0LXYwLjMlMjIlMkMlMjBxdWFudGl6YXRpb25fY29uZmlnJTNEVHJ1ZSUyQyUyMGR0eXBlJTNEdG9yY2guYmZsb2F0MTYlMkMlMjBkZXZpY2VfbWFwJTNEJTIyYXV0byUyMiklMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJtaXN0cmFsYWklMkZNaXN0cmFsLTdCLUluc3RydWN0LXYwLjMlMjIpJTBBJTBBcHJvbXB0JTIwJTNEJTIwJTIyTXklMjBmYXZvdXJpdGUlMjBjb25kaW1lbnQlMjBpcyUyMiUwQSUwQW1lc3NhZ2VzJTIwJTNEJTIwJTVCJTBBJTIwJTIwJTIwJTIwJTdCJTIycm9sZSUyMiUzQSUyMCUyMnVzZXIlMjIlMkMlMjAlMjJjb250ZW50JTIyJTNBJTIwJTIyV2hhdCUyMGlzJTIweW91ciUyMGZhdm91cml0ZSUyMGNvbmRpbWVudCUzRiUyMiU3RCUyQyUwQSUyMCUyMCUyMCUyMCU3QiUyMnJvbGUlMjIlM0ElMjAlMjJhc3Npc3RhbnQlMjIlMkMlMjAlMjJjb250ZW50JTIyJTNBJTIwJTIyV2VsbCUyQyUyMEknbSUyMHF1aXRlJTIwcGFydGlhbCUyMHRvJTIwYSUyMGdvb2QlMjBzcXVlZXplJTIwb2YlMjBmcmVzaCUyMGxlbW9uJTIwanVpY2UuJTIwSXQlMjBhZGRzJTIwanVzdCUyMHRoZSUyMHJpZ2h0JTIwYW1vdW50JTIwb2YlMjB6ZXN0eSUyMGZsYXZvdXIlMjB0byUyMHdoYXRldmVyJTIwSSdtJTIwY29va2luZyUyMHVwJTIwaW4lMjB0aGUlMjBraXRjaGVuISUyMiU3RCUyQyUwQSUyMCUyMCUyMCUyMCU3QiUyMnJvbGUlMjIlM0ElMjAlMjJ1c2VyJTIyJTJDJTIwJTIyY29udGVudCUyMiUzQSUyMCUyMkRvJTIweW91JTIwaGF2ZSUyMG1heW9ubmFpc2UlMjByZWNpcGVzJTNGJTIyJTdEJTBBJTVEJTBBJTBBbW9kZWxfaW5wdXRzJTIwJTNEJTIwdG9rZW5pemVyLmFwcGx5X2NoYXRfdGVtcGxhdGUobWVzc2FnZXMlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKS50byhtb2RlbC5kZXZpY2UpJTBBJTBBZ2VuZXJhdGVkX2lkcyUyMCUzRCUyMG1vZGVsLmdlbmVyYXRlKG1vZGVsX2lucHV0cyUyQyUyMG1heF9uZXdfdG9rZW5zJTNEMTAwJTJDJTIwZG9fc2FtcGxlJTNEVHJ1ZSklMEF0b2tlbml6ZXIuYmF0Y2hfZGVjb2RlKGdlbmVyYXRlZF9pZHMpJTVCMCU1RA==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># specify how to quantize the model</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>quantization_config = BitsAndBytesConfig(
<span class="hljs-meta">... </span>        load_in_4bit=<span class="hljs-literal">True</span>,
<span class="hljs-meta">... </span>        bnb_4bit_quant_type=<span class="hljs-string">&quot;nf4&quot;</span>,
<span class="hljs-meta">... </span>        bnb_4bit_compute_dtype=<span class="hljs-string">&quot;torch.float16&quot;</span>,
<span class="hljs-meta">... </span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>model = AutoModelForCausalLM.from_pretrained(<span class="hljs-string">&quot;mistralai/Mistral-7B-Instruct-v0.3&quot;</span>, quantization_config=<span class="hljs-literal">True</span>, dtype=torch.bfloat16, device_map=<span class="hljs-string">&quot;auto&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;mistralai/Mistral-7B-Instruct-v0.3&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>prompt = <span class="hljs-string">&quot;My favourite condiment is&quot;</span>

<span class="hljs-meta">&gt;&gt;&gt; </span>messages = [
<span class="hljs-meta">... </span>    {<span class="hljs-string">&quot;role&quot;</span>: <span class="hljs-string">&quot;user&quot;</span>, <span class="hljs-string">&quot;content&quot;</span>: <span class="hljs-string">&quot;What is your favourite condiment?&quot;</span>},
<span class="hljs-meta">... </span>    {<span class="hljs-string">&quot;role&quot;</span>: <span class="hljs-string">&quot;assistant&quot;</span>, <span class="hljs-string">&quot;content&quot;</span>: <span class="hljs-string">&quot;Well, I&#x27;m quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I&#x27;m cooking up in the kitchen!&quot;</span>},
<span class="hljs-meta">... </span>    {<span class="hljs-string">&quot;role&quot;</span>: <span class="hljs-string">&quot;user&quot;</span>, <span class="hljs-string">&quot;content&quot;</span>: <span class="hljs-string">&quot;Do you have mayonnaise recipes?&quot;</span>}
<span class="hljs-meta">... </span>]

<span class="hljs-meta">&gt;&gt;&gt; </span>model_inputs = tokenizer.apply_chat_template(messages, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>).to(model.device)

<span class="hljs-meta">&gt;&gt;&gt; </span>generated_ids = model.generate(model_inputs, max_new_tokens=<span class="hljs-number">100</span>, do_sample=<span class="hljs-literal">True</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer.batch_decode(generated_ids)[<span class="hljs-number">0</span>]
<span class="hljs-string">&quot;The expected output&quot;</span>`,wrap:!1}}),ze=new ye({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycy51dGlscy5hdHRlbnRpb25fdmlzdWFsaXplciUyMGltcG9ydCUyMEF0dGVudGlvbk1hc2tWaXN1YWxpemVyJTBBJTBBdmlzdWFsaXplciUyMCUzRCUyMEF0dGVudGlvbk1hc2tWaXN1YWxpemVyKCUyMm1pc3RyYWxhaSUyRk1pc3RyYWwtN0ItSW5zdHJ1Y3QtdjAuMyUyMiklMEF2aXN1YWxpemVyKCUyMkRvJTIweW91JTIwaGF2ZSUyMG1heW9ubmFpc2UlMjByZWNpcGVzJTNGJTIyKQ==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers.utils.attention_visualizer <span class="hljs-keyword">import</span> AttentionMaskVisualizer

<span class="hljs-meta">&gt;&gt;&gt; </span>visualizer = AttentionMaskVisualizer(<span class="hljs-string">&quot;mistralai/Mistral-7B-Instruct-v0.3&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>visualizer(<span class="hljs-string">&quot;Do you have mayonnaise recipes?&quot;</span>)`,wrap:!1}}),$e=new _e({props:{title:"MistralConfig",local:"transformers.MistralConfig",headingTag:"h2"}}),Ue=new $({props:{name:"class transformers.MistralConfig",anchor:"transformers.MistralConfig",parameters:[{name:"vocab_size",val:" = 32000"},{name:"hidden_size",val:" = 4096"},{name:"intermediate_size",val:" = 14336"},{name:"num_hidden_layers",val:" = 32"},{name:"num_attention_heads",val:" = 32"},{name:"num_key_value_heads",val:" = 8"},{name:"head_dim",val:" = None"},{name:"hidden_act",val:" = 'silu'"},{name:"max_position_embeddings",val:" = 131072"},{name:"initializer_range",val:" = 0.02"},{name:"rms_norm_eps",val:" = 1e-06"},{name:"use_cache",val:" = True"},{name:"pad_token_id",val:" = None"},{name:"bos_token_id",val:" = 1"},{name:"eos_token_id",val:" = 2"},{name:"tie_word_embeddings",val:" = False"},{name:"rope_theta",val:" = 10000.0"},{name:"sliding_window",val:" = 4096"},{name:"attention_dropout",val:" = 0.0"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.MistralConfig.vocab_size",description:`<strong>vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 32000) &#x2014;
Vocabulary size of the Mistral model. Defines the number of different tokens that can be represented by the
<code>inputs_ids</code> passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/mistral#transformers.MistralModel">MistralModel</a>`,name:"vocab_size"},{anchor:"transformers.MistralConfig.hidden_size",description:`<strong>hidden_size</strong> (<code>int</code>, <em>optional</em>, defaults to 4096) &#x2014;
Dimension of the hidden representations.`,name:"hidden_size"},{anchor:"transformers.MistralConfig.intermediate_size",description:`<strong>intermediate_size</strong> (<code>int</code>, <em>optional</em>, defaults to 14336) &#x2014;
Dimension of the MLP representations.`,name:"intermediate_size"},{anchor:"transformers.MistralConfig.num_hidden_layers",description:`<strong>num_hidden_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 32) &#x2014;
Number of hidden layers in the Transformer encoder.`,name:"num_hidden_layers"},{anchor:"transformers.MistralConfig.num_attention_heads",description:`<strong>num_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 32) &#x2014;
Number of attention heads for each attention layer in the Transformer encoder.`,name:"num_attention_heads"},{anchor:"transformers.MistralConfig.num_key_value_heads",description:`<strong>num_key_value_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 8) &#x2014;
This is the number of key_value heads that should be used to implement Grouped Query Attention. If
<code>num_key_value_heads=num_attention_heads</code>, the model will use Multi Head Attention (MHA), if
<code>num_key_value_heads=1</code> the model will use Multi Query Attention (MQA) otherwise GQA is used. When
converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
by meanpooling all the original heads within that group. For more details, check out <a href="https://huggingface.co/papers/2305.13245" rel="nofollow">this
paper</a>. If it is not specified, will default to <code>8</code>.`,name:"num_key_value_heads"},{anchor:"transformers.MistralConfig.head_dim",description:`<strong>head_dim</strong> (<code>int</code>, <em>optional</em>, defaults to <code>hidden_size // num_attention_heads</code>) &#x2014;
The attention head dimension.`,name:"head_dim"},{anchor:"transformers.MistralConfig.hidden_act",description:`<strong>hidden_act</strong> (<code>str</code> or <code>function</code>, <em>optional</em>, defaults to <code>&quot;silu&quot;</code>) &#x2014;
The non-linear activation function (function or string) in the decoder.`,name:"hidden_act"},{anchor:"transformers.MistralConfig.max_position_embeddings",description:`<strong>max_position_embeddings</strong> (<code>int</code>, <em>optional</em>, defaults to <code>4096*32</code>) &#x2014;
The maximum sequence length that this model might ever be used with. Mistral&#x2019;s sliding window attention
allows sequence of up to 4096*32 tokens.`,name:"max_position_embeddings"},{anchor:"transformers.MistralConfig.initializer_range",description:`<strong>initializer_range</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.`,name:"initializer_range"},{anchor:"transformers.MistralConfig.rms_norm_eps",description:`<strong>rms_norm_eps</strong> (<code>float</code>, <em>optional</em>, defaults to 1e-06) &#x2014;
The epsilon used by the rms normalization layers.`,name:"rms_norm_eps"},{anchor:"transformers.MistralConfig.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not the model should return the last key/values attentions (not used by all models). Only
relevant if <code>config.is_decoder=True</code>.`,name:"use_cache"},{anchor:"transformers.MistralConfig.pad_token_id",description:`<strong>pad_token_id</strong> (<code>int</code>, <em>optional</em>) &#x2014;
The id of the padding token.`,name:"pad_token_id"},{anchor:"transformers.MistralConfig.bos_token_id",description:`<strong>bos_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 1) &#x2014;
The id of the &#x201C;beginning-of-sequence&#x201D; token.`,name:"bos_token_id"},{anchor:"transformers.MistralConfig.eos_token_id",description:`<strong>eos_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 2) &#x2014;
The id of the &#x201C;end-of-sequence&#x201D; token.`,name:"eos_token_id"},{anchor:"transformers.MistralConfig.tie_word_embeddings",description:`<strong>tie_word_embeddings</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether the model&#x2019;s input and output word embeddings should be tied.`,name:"tie_word_embeddings"},{anchor:"transformers.MistralConfig.rope_theta",description:`<strong>rope_theta</strong> (<code>float</code>, <em>optional</em>, defaults to 10000.0) &#x2014;
The base period of the RoPE embeddings.`,name:"rope_theta"},{anchor:"transformers.MistralConfig.sliding_window",description:`<strong>sliding_window</strong> (<code>int</code>, <em>optional</em>, defaults to 4096) &#x2014;
Sliding window attention window size. If not specified, will default to <code>4096</code>.`,name:"sliding_window"},{anchor:"transformers.MistralConfig.attention_dropout",description:`<strong>attention_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The dropout ratio for the attention probabilities.`,name:"attention_dropout"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mistral/configuration_mistral.py#L24"}}),Q=new as({props:{anchor:"transformers.MistralConfig.example",$$slots:{default:[fa]},$$scope:{ctx:w}}}),Ie=new _e({props:{title:"MistralCommonTokenizer",local:"transformers.MistralCommonTokenizer",headingTag:"h2"}}),je=new $({props:{name:"class transformers.MistralCommonTokenizer",anchor:"transformers.MistralCommonTokenizer",parameters:[{name:"tokenizer_path",val:": typing.Union[str, os.PathLike, pathlib.Path]"},{name:"mode",val:": ValidationMode = <ValidationMode.test: 'test'>"},{name:"model_max_length",val:": int = 1000000000000000019884624838656"},{name:"padding_side",val:": str = 'left'"},{name:"truncation_side",val:": str = 'right'"},{name:"model_input_names",val:": typing.Optional[list[str]] = None"},{name:"clean_up_tokenization_spaces",val:": bool = False"},{name:"**kwargs",val:""}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_mistral_common.py#L158"}}),O=new as({props:{anchor:"transformers.MistralCommonTokenizer.example",$$slots:{default:[ga]},$$scope:{ctx:w}}}),Je=new $({props:{name:"apply_chat_template",anchor:"transformers.MistralCommonTokenizer.apply_chat_template",parameters:[{name:"conversation",val:": typing.Union[list[dict[str, str]], list[list[dict[str, str]]]]"},{name:"tools",val:": typing.Optional[list[typing.Union[dict, typing.Callable]]] = None"},{name:"continue_final_message",val:": bool = False"},{name:"tokenize",val:": bool = True"},{name:"padding",val:": typing.Union[bool, str, transformers.utils.generic.PaddingStrategy] = False"},{name:"truncation",val:": bool = False"},{name:"max_length",val:": typing.Optional[int] = None"},{name:"return_tensors",val:": typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None"},{name:"return_dict",val:": bool = False"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.MistralCommonTokenizer.apply_chat_template.conversation",description:`<strong>conversation</strong> (Union[List[Dict[str, str]], List[List[Dict[str, str]]]]) &#x2014; A list of dicts
with &#x201C;role&#x201D; and &#x201C;content&#x201D; keys, representing the chat history so far.`,name:"conversation"},{anchor:"transformers.MistralCommonTokenizer.apply_chat_template.tools",description:`<strong>tools</strong> (<code>List[Union[Dict, Callable]]</code>, <em>optional</em>) &#x2014;
A list of tools (callable functions) that will be accessible to the model. If the template does not
support function calling, this argument will have no effect. Each tool should be passed as a JSON Schema,
giving the name, description and argument types for the tool. See our
<a href="https://huggingface.co/docs/transformers/main/en/chat_templating#automated-function-conversion-for-tool-use" rel="nofollow">chat templating guide</a>
for more information.`,name:"tools"},{anchor:"transformers.MistralCommonTokenizer.apply_chat_template.continue_final_message",description:`<strong>continue_final_message</strong> (bool, <em>optional</em>) &#x2014;
If this is set, the chat will be formatted so that the final
message in the chat is open-ended, without any EOS tokens. The model will continue this message
rather than starting a new one. This allows you to &#x201C;prefill&#x201D; part of
the model&#x2019;s response for it. Cannot be used at the same time as <code>add_generation_prompt</code>.`,name:"continue_final_message"},{anchor:"transformers.MistralCommonTokenizer.apply_chat_template.tokenize",description:`<strong>tokenize</strong> (<code>bool</code>, defaults to <code>True</code>) &#x2014;
Whether to tokenize the output. If <code>False</code>, the output will be a string.`,name:"tokenize"},{anchor:"transformers.MistralCommonTokenizer.apply_chat_template.padding",description:`<strong>padding</strong> (<code>bool</code>, <code>str</code> or <a href="/docs/transformers/v4.56.2/en/internal/file_utils#transformers.utils.PaddingStrategy">PaddingStrategy</a>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Select a strategy to pad the returned sequences (according to the model&#x2019;s padding side and padding
index) among:</p>
<ul>
<li><code>True</code> or <code>&apos;longest&apos;</code>: Pad to the longest sequence in the batch (or no padding if only a single
sequence if provided).</li>
<li><code>&apos;max_length&apos;</code>: Pad to a maximum length specified with the argument <code>max_length</code> or to the maximum
acceptable input length for the model if that argument is not provided.</li>
<li><code>False</code> or <code>&apos;do_not_pad&apos;</code> (default): No padding (i.e., can output a batch with sequences of different
lengths).</li>
</ul>`,name:"padding"},{anchor:"transformers.MistralCommonTokenizer.apply_chat_template.truncation",description:`<strong>truncation</strong> (<code>bool</code>, defaults to <code>False</code>) &#x2014;
Whether to truncate sequences at the maximum length. Has no effect if tokenize is <code>False</code>.`,name:"truncation"},{anchor:"transformers.MistralCommonTokenizer.apply_chat_template.max_length",description:`<strong>max_length</strong> (<code>int</code>, <em>optional</em>) &#x2014;
Maximum length (in tokens) to use for padding or truncation. Has no effect if tokenize is <code>False</code>. If
not specified, the tokenizer&#x2019;s <code>max_length</code> attribute will be used as a default.`,name:"max_length"},{anchor:"transformers.MistralCommonTokenizer.apply_chat_template.return_tensors",description:`<strong>return_tensors</strong> (<code>str</code> or <a href="/docs/transformers/v4.56.2/en/internal/file_utils#transformers.TensorType">TensorType</a>, <em>optional</em>) &#x2014;
If set, will return tensors of a particular framework. Has no effect if tokenize is <code>False</code>. Acceptable
values are:</p>
<ul>
<li><code>&apos;pt&apos;</code>: Return PyTorch <code>torch.Tensor</code> objects.</li>
</ul>`,name:"return_tensors"},{anchor:"transformers.MistralCommonTokenizer.apply_chat_template.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, defaults to <code>False</code>) &#x2014;
Whether to return a dictionary with named outputs. Has no effect if tokenize is <code>False</code>.
If at least one conversation contains an image, its pixel values will be returned in the <code>pixel_values</code> key.`,name:"return_dict"},{anchor:"transformers.MistralCommonTokenizer.apply_chat_template.kwargs",description:`<strong>kwargs</strong> (additional keyword arguments, <em>optional</em>) &#x2014;
Not supported by <code>MistralCommonTokenizer.apply_chat_template</code>.
Will raise an error if used.`,name:"kwargs"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_mistral_common.py#L1368",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A list of token ids representing the tokenized chat so far, including control
tokens. This output is ready to pass to the model, either directly or via methods like <code>generate()</code>.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>Union[str, List[int], List[str], List[List[int]], BatchEncoding]</code></p>
`}}),qe=new $({props:{name:"batch_decode",anchor:"transformers.MistralCommonTokenizer.batch_decode",parameters:[{name:"sequences",val:": typing.Union[list[int], list[list[int]], ForwardRef('np.ndarray'), ForwardRef('torch.Tensor')]"},{name:"skip_special_tokens",val:": bool = False"},{name:"clean_up_tokenization_spaces",val:": typing.Optional[bool] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.MistralCommonTokenizer.batch_decode.sequences",description:`<strong>sequences</strong> (<code>Union[List[int], List[List[int]], np.ndarray, torch.Tensor]</code>) &#x2014;
List of tokenized input ids. Can be obtained using the <code>__call__</code> method.`,name:"sequences"},{anchor:"transformers.MistralCommonTokenizer.batch_decode.skip_special_tokens",description:`<strong>skip_special_tokens</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to remove special tokens in the decoding.`,name:"skip_special_tokens"},{anchor:"transformers.MistralCommonTokenizer.batch_decode.clean_up_tokenization_spaces",description:`<strong>clean_up_tokenization_spaces</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to clean up the tokenization spaces. If <code>None</code>, will default to
<code>self.clean_up_tokenization_spaces</code>.`,name:"clean_up_tokenization_spaces"},{anchor:"transformers.MistralCommonTokenizer.batch_decode.kwargs",description:`<strong>kwargs</strong> (additional keyword arguments, <em>optional</em>) &#x2014;
Not supported by <code>MistralCommonTokenizer.batch_decode</code>.
Will raise an error if used.`,name:"kwargs"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_mistral_common.py#L476",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>The list of decoded sentences.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>List[str]</code></p>
`}}),Fe=new $({props:{name:"convert_ids_to_tokens",anchor:"transformers.MistralCommonTokenizer.convert_ids_to_tokens",parameters:[{name:"ids",val:": typing.Union[int, list[int]]"},{name:"skip_special_tokens",val:": bool = False"}],parametersDescription:[{anchor:"transformers.MistralCommonTokenizer.convert_ids_to_tokens.ids",description:`<strong>ids</strong> (<code>int</code> or <code>List[int]</code>) &#x2014;
The token id (or token ids) to convert to tokens.`,name:"ids"},{anchor:"transformers.MistralCommonTokenizer.convert_ids_to_tokens.skip_special_tokens",description:`<strong>skip_special_tokens</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to remove special tokens in the decoding.`,name:"skip_special_tokens"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_mistral_common.py#L523",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>The decoded token(s).</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>str</code> or <code>List[str]</code></p>
`}}),Le=new $({props:{name:"convert_tokens_to_ids",anchor:"transformers.MistralCommonTokenizer.convert_tokens_to_ids",parameters:[{name:"tokens",val:": typing.Union[str, list[str]]"}],parametersDescription:[{anchor:"transformers.MistralCommonTokenizer.convert_tokens_to_ids.tokens",description:"<strong>tokens</strong> (<code>str</code> or <code>List[str]</code>) &#x2014; One or several token(s) to convert to token id(s).",name:"tokens"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_mistral_common.py#L571",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>The token id or list of token ids.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>int</code> or <code>List[int]</code></p>
`}}),We=new $({props:{name:"decode",anchor:"transformers.MistralCommonTokenizer.decode",parameters:[{name:"token_ids",val:": typing.Union[int, list[int], ForwardRef('np.ndarray'), ForwardRef('torch.Tensor')]"},{name:"skip_special_tokens",val:": bool = False"},{name:"clean_up_tokenization_spaces",val:": typing.Optional[bool] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.MistralCommonTokenizer.decode.token_ids",description:`<strong>token_ids</strong> (<code>Union[int, List[int], np.ndarray, torch.Tensor]</code>) &#x2014;
List of tokenized input ids. Can be obtained using the <code>__call__</code> method.`,name:"token_ids"},{anchor:"transformers.MistralCommonTokenizer.decode.skip_special_tokens",description:`<strong>skip_special_tokens</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to remove special tokens in the decoding.`,name:"skip_special_tokens"},{anchor:"transformers.MistralCommonTokenizer.decode.clean_up_tokenization_spaces",description:`<strong>clean_up_tokenization_spaces</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to clean up the tokenization spaces. If <code>None</code>, will default to
<code>self.clean_up_tokenization_spaces</code>.`,name:"clean_up_tokenization_spaces"},{anchor:"transformers.MistralCommonTokenizer.decode.kwargs",description:`<strong>kwargs</strong> (additional keyword arguments, <em>optional</em>) &#x2014;
Not supported by <code>MistralCommonTokenizer.decode</code>.
Will raise an error if used.`,name:"kwargs"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_mistral_common.py#L434",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>The decoded sentence.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>str</code></p>
`}}),Ne=new $({props:{name:"encode",anchor:"transformers.MistralCommonTokenizer.encode",parameters:[{name:"text",val:": typing.Union[str, list[int]]"},{name:"text_pair",val:": None = None"},{name:"add_special_tokens",val:": bool = True"},{name:"padding",val:": typing.Union[bool, str, transformers.utils.generic.PaddingStrategy] = False"},{name:"truncation",val:": typing.Union[bool, str, transformers.tokenization_utils_base.TruncationStrategy, NoneType] = None"},{name:"max_length",val:": typing.Optional[int] = None"},{name:"stride",val:": int = 0"},{name:"pad_to_multiple_of",val:": typing.Optional[int] = None"},{name:"padding_side",val:": typing.Optional[str] = None"},{name:"return_tensors",val:": typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None"},{name:"verbose",val:": bool = True"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.MistralCommonTokenizer.encode.text",description:`<strong>text</strong> (<code>str</code> or <code>List[int]</code>) &#x2014;
The first sequence to be encoded. This can be a string or a list of integers (tokenized string ids).`,name:"text"},{anchor:"transformers.MistralCommonTokenizer.encode.text_pair",description:`<strong>text_pair</strong> (<code>None</code>, <em>optional</em>) &#x2014;
Not supported by <code>MistralCommonTokenizer.encode</code>. Kept to match <code>PreTrainedTokenizerBase.encode</code> signature.`,name:"text_pair"},{anchor:"transformers.MistralCommonTokenizer.encode.add_special_tokens",description:`<strong>add_special_tokens</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to add special tokens when encoding the sequences. This will use the underlying
<code>PretrainedTokenizerBase.build_inputs_with_special_tokens</code> function, which defines which tokens are
automatically added to the input ids. This is useful if you want to add <code>bos</code> or <code>eos</code> tokens
automatically.`,name:"add_special_tokens"},{anchor:"transformers.MistralCommonTokenizer.encode.padding",description:`<strong>padding</strong> (<code>bool</code>, <code>str</code> or <a href="/docs/transformers/v4.56.2/en/internal/file_utils#transformers.utils.PaddingStrategy">PaddingStrategy</a>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Activates and controls padding. Accepts the following values:</p>
<ul>
<li><code>True</code> or <code>&apos;longest&apos;</code>: Pad to the longest sequence in the batch (or no padding if only a single
sequence is provided).</li>
<li><code>&apos;max_length&apos;</code>: Pad to a maximum length specified with the argument <code>max_length</code> or to the maximum
acceptable input length for the model if that argument is not provided.</li>
<li><code>False</code> or <code>&apos;do_not_pad&apos;</code> (default): No padding (i.e., can output a batch with sequences of different
lengths).</li>
</ul>`,name:"padding"},{anchor:"transformers.MistralCommonTokenizer.encode.truncation",description:`<strong>truncation</strong> (<code>bool</code>, <code>str</code> or <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.tokenization_utils_base.TruncationStrategy">TruncationStrategy</a>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Activates and controls truncation. Accepts the following values:</p>
<ul>
<li><code>True</code> or <code>&apos;longest_first&apos;</code>: Truncate to a maximum length specified with the argument <code>max_length</code> or
to the maximum acceptable input length for the model if that argument is not provided.</li>
<li><code>False</code> or <code>&apos;do_not_truncate&apos;</code> (default): No truncation (i.e., can output batch with sequence lengths
greater than the model maximum admissible input size).</li>
</ul>`,name:"truncation"},{anchor:"transformers.MistralCommonTokenizer.encode.max_length",description:`<strong>max_length</strong> (<code>int</code>, <em>optional</em>) &#x2014;
Controls the maximum length to use by one of the truncation/padding parameters.</p>
<p>If left unset or set to <code>None</code>, this will use the predefined model maximum length if a maximum length
is required by one of the truncation/padding parameters. If the model has no specific maximum input
length (like XLNet) truncation/padding to a maximum length will be deactivated.`,name:"max_length"},{anchor:"transformers.MistralCommonTokenizer.encode.stride",description:`<strong>stride</strong> (<code>int</code>, <em>optional</em>, defaults to 0) &#x2014;
If set to a number along with <code>max_length</code>, the overflowing tokens returned when
<code>return_overflowing_tokens=True</code> will contain some tokens from the end of the truncated sequence
returned to provide some overlap between truncated and overflowing sequences. The value of this
argument defines the number of overlapping tokens.`,name:"stride"},{anchor:"transformers.MistralCommonTokenizer.encode.pad_to_multiple_of",description:`<strong>pad_to_multiple_of</strong> (<code>int</code>, <em>optional</em>) &#x2014;
If set will pad the sequence to a multiple of the provided value. Requires <code>padding</code> to be activated.
This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
<code>&gt;= 7.5</code> (Volta).`,name:"pad_to_multiple_of"},{anchor:"transformers.MistralCommonTokenizer.encode.padding_side",description:`<strong>padding_side</strong> (<code>str</code>, <em>optional</em>) &#x2014;
The side on which the model should have padding applied. Should be selected between [&#x2018;right&#x2019;, &#x2018;left&#x2019;].
Default value is picked from the class attribute of the same name.`,name:"padding_side"},{anchor:"transformers.MistralCommonTokenizer.encode.return_tensors",description:`<strong>return_tensors</strong> (<code>str</code> or <a href="/docs/transformers/v4.56.2/en/internal/file_utils#transformers.TensorType">TensorType</a>, <em>optional</em>) &#x2014;
If set, will return tensors instead of list of python integers. Acceptable values are:</p>
<ul>
<li><code>&apos;pt&apos;</code>: Return PyTorch <code>torch.Tensor</code> objects.</li>
</ul>`,name:"return_tensors"},{anchor:"transformers.MistralCommonTokenizer.encode.*kwargs",description:`*<strong>*kwargs</strong> &#x2014; Not supported by <code>MistralCommonTokenizer.encode</code>.
Will raise an error if used.`,name:"*kwargs"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_mistral_common.py#L367",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>The tokenized ids of the text.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>List[int]</code>, <code>torch.Tensor</code></p>
`}}),Be=new $({props:{name:"from_pretrained",anchor:"transformers.MistralCommonTokenizer.from_pretrained",parameters:[{name:"pretrained_model_name_or_path",val:": typing.Union[str, os.PathLike]"},{name:"*init_inputs",val:""},{name:"mode",val:": ValidationMode = <ValidationMode.test: 'test'>"},{name:"cache_dir",val:": typing.Union[str, os.PathLike, NoneType] = None"},{name:"force_download",val:": bool = False"},{name:"local_files_only",val:": bool = False"},{name:"token",val:": typing.Union[bool, str, NoneType] = None"},{name:"revision",val:": str = 'main'"},{name:"model_max_length",val:": int = 1000000000000000019884624838656"},{name:"padding_side",val:": str = 'left'"},{name:"truncation_side",val:": str = 'right'"},{name:"model_input_names",val:": typing.Optional[list[str]] = None"},{name:"clean_up_tokenization_spaces",val:": bool = False"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.MistralCommonTokenizer.from_pretrained.pretrained_model_name_or_path",description:`<strong>pretrained_model_name_or_path</strong> (<code>str</code> or <code>os.PathLike</code>) &#x2014;
Can be either:</p>
<ul>
<li>A string, the <em>model id</em> of a predefined tokenizer hosted inside a model repo on huggingface.co.</li>
<li>A path to a <em>directory</em> containing the tokenizer config, for instance saved
using the <code>MistralCommonTokenizer.tokenization_mistral_common.save_pretrained</code> method, e.g.,
<code>./my_model_directory/</code>.</li>
</ul>`,name:"pretrained_model_name_or_path"},{anchor:"transformers.MistralCommonTokenizer.from_pretrained.mode",description:`<strong>mode</strong> (<code>ValidationMode</code>, <em>optional</em>, defaults to <code>ValidationMode.test</code>) &#x2014;
Validation mode for the <code>MistralTokenizer</code> tokenizer.`,name:"mode"},{anchor:"transformers.MistralCommonTokenizer.from_pretrained.cache_dir",description:`<strong>cache_dir</strong> (<code>str</code> or <code>os.PathLike</code>, <em>optional</em>) &#x2014;
Path to a directory in which a downloaded predefined tokenizer vocabulary files should be cached if the
standard cache should not be used.`,name:"cache_dir"},{anchor:"transformers.MistralCommonTokenizer.from_pretrained.force_download",description:`<strong>force_download</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to force the (re-)download the vocabulary files and override the cached versions if they
exist.`,name:"force_download"},{anchor:"transformers.MistralCommonTokenizer.from_pretrained.token",description:`<strong>token</strong> (<code>str</code> or <em>bool</em>, <em>optional</em>) &#x2014;
The token to use as HTTP bearer authorization for remote files. If <code>True</code>, will use the token generated
when running <code>hf auth login</code> (stored in <code>~/.huggingface</code>).`,name:"token"},{anchor:"transformers.MistralCommonTokenizer.from_pretrained.local_files_only",description:`<strong>local_files_only</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to only rely on local files and not to attempt to download any files.`,name:"local_files_only"},{anchor:"transformers.MistralCommonTokenizer.from_pretrained.revision",description:`<strong>revision</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;main&quot;</code>) &#x2014;
The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
git-based system for storing models and other artifacts on huggingface.co, so <code>revision</code> can be any
identifier allowed by git.`,name:"revision"},{anchor:"transformers.MistralCommonTokenizer.from_pretrained.max_length",description:`<strong>max_length</strong> (<code>int</code>, <em>optional</em>) &#x2014;
Controls the maximum length to use by one of the truncation/padding parameters.</p>
<p>If left unset or set to <code>None</code>, this will use the predefined model maximum length if a maximum length
is required by one of the truncation/padding parameters. If the model has no specific maximum input
length (like XLNet) truncation/padding to a maximum length will be deactivated.`,name:"max_length"},{anchor:"transformers.MistralCommonTokenizer.from_pretrained.padding_side",description:`<strong>padding_side</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;left&quot;</code>) &#x2014;
The side on which the model should have padding applied. Should be selected between [&#x2018;right&#x2019;, &#x2018;left&#x2019;].
Default value is picked from the class attribute of the same name.`,name:"padding_side"},{anchor:"transformers.MistralCommonTokenizer.from_pretrained.truncation_side",description:`<strong>truncation_side</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;right&quot;</code>) &#x2014;
The side on which the model should have truncation applied. Should be selected between [&#x2018;right&#x2019;, &#x2018;left&#x2019;].`,name:"truncation_side"},{anchor:"transformers.MistralCommonTokenizer.from_pretrained.model_input_names",description:`<strong>model_input_names</strong> (<code>List[string]</code>, <em>optional</em>) &#x2014;
The list of inputs accepted by the forward pass of the model (like <code>&quot;token_type_ids&quot;</code> or
<code>&quot;attention_mask&quot;</code>). Default value is picked from the class attribute of the same name.`,name:"model_input_names"},{anchor:"transformers.MistralCommonTokenizer.from_pretrained.clean_up_tokenization_spaces",description:`<strong>clean_up_tokenization_spaces</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not the model should cleanup the spaces that were added when splitting the input text during the
tokenization process.`,name:"clean_up_tokenization_spaces"},{anchor:"transformers.MistralCommonTokenizer.from_pretrained.kwargs",description:`<strong>kwargs</strong> (additional keyword arguments, <em>optional</em>) &#x2014;
Not supported by <code>MistralCommonTokenizer.from_pretrained</code>.
Will raise an error if used.`,name:"kwargs"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_mistral_common.py#L1689"}}),Ge=new $({props:{name:"get_special_tokens_mask",anchor:"transformers.MistralCommonTokenizer.get_special_tokens_mask",parameters:[{name:"token_ids_0",val:": list"},{name:"token_ids_1",val:": None = None"},{name:"already_has_special_tokens",val:": bool = False"}],parametersDescription:[{anchor:"transformers.MistralCommonTokenizer.get_special_tokens_mask.token_ids_0",description:`<strong>token_ids_0</strong> (<code>List[int]</code>) &#x2014;
List of ids of the sequence.`,name:"token_ids_0"},{anchor:"transformers.MistralCommonTokenizer.get_special_tokens_mask.token_ids_1",description:`<strong>token_ids_1</strong> (<code>List[int]</code>, <em>optional</em>) &#x2014;
Not supported by <code>MistralCommonTokenizer</code>. Kept to match the interface of <code>PreTrainedTokenizerBase</code>.`,name:"token_ids_1"},{anchor:"transformers.MistralCommonTokenizer.get_special_tokens_mask.already_has_special_tokens",description:`<strong>already_has_special_tokens</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not the token list is already formatted with special tokens for the model.`,name:"already_has_special_tokens"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_mistral_common.py#L746",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>1 for a special token, 0 for a sequence token.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A list of integers in the range [0, 1]</p>
`}}),Pe=new $({props:{name:"get_vocab",anchor:"transformers.MistralCommonTokenizer.get_vocab",parameters:[],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_mistral_common.py#L345",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>The vocabulary.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>Dict[str, int]</code></p>
`}}),Ve=new $({props:{name:"pad",anchor:"transformers.MistralCommonTokenizer.pad",parameters:[{name:"encoded_inputs",val:": typing.Union[transformers.tokenization_utils_base.BatchEncoding, list[transformers.tokenization_utils_base.BatchEncoding], dict[str, list[int]], dict[str, list[list[int]]], list[dict[str, list[int]]]]"},{name:"padding",val:": typing.Union[bool, str, transformers.utils.generic.PaddingStrategy] = True"},{name:"max_length",val:": typing.Optional[int] = None"},{name:"pad_to_multiple_of",val:": typing.Optional[int] = None"},{name:"padding_side",val:": typing.Optional[str] = None"},{name:"return_attention_mask",val:": typing.Optional[bool] = None"},{name:"return_tensors",val:": typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None"},{name:"verbose",val:": bool = True"}],parametersDescription:[{anchor:"transformers.MistralCommonTokenizer.pad.encoded_inputs",description:`<strong>encoded_inputs</strong> (<a href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.BatchEncoding">BatchEncoding</a>, list of <a href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.BatchEncoding">BatchEncoding</a>, <code>Dict[str, List[int]]</code>, <code>Dict[str, List[List[int]]</code> or <code>List[Dict[str, List[int]]]</code>) &#x2014;
Tokenized inputs. Can represent one input (<a href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.BatchEncoding">BatchEncoding</a> or <code>Dict[str, List[int]]</code>) or a batch of
tokenized inputs (list of <a href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.BatchEncoding">BatchEncoding</a>, <em>Dict[str, List[List[int]]]</em> or <em>List[Dict[str,
List[int]]]</em>) so you can use this method during preprocessing as well as in a PyTorch Dataloader
collate function.</p>
<p>Instead of <code>List[int]</code> you can have tensors (numpy arrays, PyTorch tensors), see
the note above for the return type.`,name:"encoded_inputs"},{anchor:"transformers.MistralCommonTokenizer.pad.padding",description:`<strong>padding</strong> (<code>bool</code>, <code>str</code> or <a href="/docs/transformers/v4.56.2/en/internal/file_utils#transformers.utils.PaddingStrategy">PaddingStrategy</a>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Select a strategy to pad the returned sequences (according to the model&#x2019;s padding side and padding
index) among:</p>
<ul>
<li><code>True</code> or <code>&apos;longest&apos;</code> (default): Pad to the longest sequence in the batch (or no padding if only a single
sequence if provided).</li>
<li><code>&apos;max_length&apos;</code>: Pad to a maximum length specified with the argument <code>max_length</code> or to the maximum
acceptable input length for the model if that argument is not provided.</li>
<li><code>False</code> or <code>&apos;do_not_pad&apos;</code>: No padding (i.e., can output a batch with sequences of different
lengths).</li>
</ul>`,name:"padding"},{anchor:"transformers.MistralCommonTokenizer.pad.max_length",description:`<strong>max_length</strong> (<code>int</code>, <em>optional</em>) &#x2014;
Maximum length of the returned list and optionally padding length (see above).`,name:"max_length"},{anchor:"transformers.MistralCommonTokenizer.pad.pad_to_multiple_of",description:`<strong>pad_to_multiple_of</strong> (<code>int</code>, <em>optional</em>) &#x2014;
If set will pad the sequence to a multiple of the provided value.</p>
<p>This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
<code>&gt;= 7.5</code> (Volta).`,name:"pad_to_multiple_of"},{anchor:"transformers.MistralCommonTokenizer.pad.padding_side",description:`<strong>padding_side</strong> (<code>str</code>, <em>optional</em>) &#x2014;
The side on which the model should have padding applied. Should be selected between [&#x2018;right&#x2019;, &#x2018;left&#x2019;].
Default value is picked from the class attribute of the same name.`,name:"padding_side"},{anchor:"transformers.MistralCommonTokenizer.pad.return_attention_mask",description:`<strong>return_attention_mask</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether to return the attention mask. If left to the default, will return the attention mask according
to the specific tokenizer&#x2019;s default, defined by the <code>return_outputs</code> attribute.</p>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"return_attention_mask"},{anchor:"transformers.MistralCommonTokenizer.pad.return_tensors",description:`<strong>return_tensors</strong> (<code>str</code> or <a href="/docs/transformers/v4.56.2/en/internal/file_utils#transformers.TensorType">TensorType</a>, <em>optional</em>) &#x2014;
If set, will return tensors instead of list of python integers. Acceptable values are:</p>
<ul>
<li><code>&apos;pt&apos;</code>: Return PyTorch <code>torch.Tensor</code> objects.</li>
<li><code>&apos;np&apos;</code>: Return Numpy <code>np.ndarray</code> objects.</li>
</ul>`,name:"return_tensors"},{anchor:"transformers.MistralCommonTokenizer.pad.verbose",description:`<strong>verbose</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to print more information and warnings.`,name:"verbose"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_mistral_common.py#L1130"}}),ie=new co({props:{$$slots:{default:[_a]},$$scope:{ctx:w}}}),Ze=new $({props:{name:"prepare_for_model",anchor:"transformers.MistralCommonTokenizer.prepare_for_model",parameters:[{name:"ids",val:": list"},{name:"pair_ids",val:": None = None"},{name:"add_special_tokens",val:": bool = True"},{name:"padding",val:": typing.Union[bool, str, transformers.utils.generic.PaddingStrategy] = False"},{name:"truncation",val:": typing.Union[bool, str, transformers.tokenization_utils_base.TruncationStrategy, NoneType] = None"},{name:"max_length",val:": typing.Optional[int] = None"},{name:"stride",val:": int = 0"},{name:"pad_to_multiple_of",val:": typing.Optional[int] = None"},{name:"padding_side",val:": typing.Optional[str] = None"},{name:"return_tensors",val:": typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None"},{name:"return_attention_mask",val:": typing.Optional[bool] = None"},{name:"return_overflowing_tokens",val:": bool = False"},{name:"return_special_tokens_mask",val:": bool = False"},{name:"return_length",val:": bool = False"},{name:"verbose",val:": bool = True"},{name:"prepend_batch_axis",val:": bool = False"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.MistralCommonTokenizer.prepare_for_model.ids",description:`<strong>ids</strong> (<code>List[int]</code>) &#x2014;
Tokenized input ids of the first sequence.`,name:"ids"},{anchor:"transformers.MistralCommonTokenizer.prepare_for_model.pair_ids",description:`<strong>pair_ids</strong> (<code>None</code>, <em>optional</em>) &#x2014;
Not supported by <code>MistralCommonTokenizer</code>. Kept to match the interface of <code>PreTrainedTokenizerBase</code>.`,name:"pair_ids"},{anchor:"transformers.MistralCommonTokenizer.prepare_for_model.add_special_tokens",description:`<strong>add_special_tokens</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to add special tokens when encoding the sequences. This will use the underlying
<code>PretrainedTokenizerBase.build_inputs_with_special_tokens</code> function, which defines which tokens are
automatically added to the input ids. This is useful if you want to add <code>bos</code> or <code>eos</code> tokens
automatically.`,name:"add_special_tokens"},{anchor:"transformers.MistralCommonTokenizer.prepare_for_model.padding",description:`<strong>padding</strong> (<code>bool</code>, <code>str</code> or <a href="/docs/transformers/v4.56.2/en/internal/file_utils#transformers.utils.PaddingStrategy">PaddingStrategy</a>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Activates and controls padding. Accepts the following values:</p>
<ul>
<li><code>True</code> or <code>&apos;longest&apos;</code>: Pad to the longest sequence in the batch (or no padding if only a single
sequence is provided).</li>
<li><code>&apos;max_length&apos;</code>: Pad to a maximum length specified with the argument <code>max_length</code> or to the maximum
acceptable input length for the model if that argument is not provided.</li>
<li><code>False</code> or <code>&apos;do_not_pad&apos;</code> (default): No padding (i.e., can output a batch with sequences of different
lengths).</li>
</ul>`,name:"padding"},{anchor:"transformers.MistralCommonTokenizer.prepare_for_model.truncation",description:`<strong>truncation</strong> (<code>bool</code>, <code>str</code> or <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.tokenization_utils_base.TruncationStrategy">TruncationStrategy</a>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Activates and controls truncation. Accepts the following values:</p>
<ul>
<li><code>True</code> or <code>&apos;longest_first&apos;</code>: Truncate to a maximum length specified with the argument <code>max_length</code> or
to the maximum acceptable input length for the model if that argument is not provided.</li>
<li><code>False</code> or <code>&apos;do_not_truncate&apos;</code> (default): No truncation (i.e., can output batch with sequence lengths
greater than the model maximum admissible input size).</li>
</ul>`,name:"truncation"},{anchor:"transformers.MistralCommonTokenizer.prepare_for_model.max_length",description:`<strong>max_length</strong> (<code>int</code>, <em>optional</em>) &#x2014;
Controls the maximum length to use by one of the truncation/padding parameters.</p>
<p>If left unset or set to <code>None</code>, this will use the predefined model maximum length if a maximum length
is required by one of the truncation/padding parameters. If the model has no specific maximum input
length (like XLNet) truncation/padding to a maximum length will be deactivated.`,name:"max_length"},{anchor:"transformers.MistralCommonTokenizer.prepare_for_model.stride",description:`<strong>stride</strong> (<code>int</code>, <em>optional</em>, defaults to 0) &#x2014;
If set to a number along with <code>max_length</code>, the overflowing tokens returned when
<code>return_overflowing_tokens=True</code> will contain some tokens from the end of the truncated sequence
returned to provide some overlap between truncated and overflowing sequences. The value of this
argument defines the number of overlapping tokens.`,name:"stride"},{anchor:"transformers.MistralCommonTokenizer.prepare_for_model.pad_to_multiple_of",description:`<strong>pad_to_multiple_of</strong> (<code>int</code>, <em>optional</em>) &#x2014;
If set will pad the sequence to a multiple of the provided value. Requires <code>padding</code> to be activated.
This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
<code>&gt;= 7.5</code> (Volta).`,name:"pad_to_multiple_of"},{anchor:"transformers.MistralCommonTokenizer.prepare_for_model.padding_side",description:`<strong>padding_side</strong> (<code>str</code>, <em>optional</em>) &#x2014;
The side on which the model should have padding applied. Should be selected between [&#x2018;right&#x2019;, &#x2018;left&#x2019;].
Default value is picked from the class attribute of the same name.`,name:"padding_side"},{anchor:"transformers.MistralCommonTokenizer.prepare_for_model.return_tensors",description:`<strong>return_tensors</strong> (<code>str</code> or <a href="/docs/transformers/v4.56.2/en/internal/file_utils#transformers.TensorType">TensorType</a>, <em>optional</em>) &#x2014;
If set, will return tensors instead of list of python integers. Acceptable values are:</p>
<ul>
<li><code>&apos;pt&apos;</code>: Return PyTorch <code>torch.Tensor</code> objects.</li>
</ul>`,name:"return_tensors"},{anchor:"transformers.MistralCommonTokenizer.prepare_for_model.return_attention_mask",description:`<strong>return_attention_mask</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether to return the attention mask. If left to the default, will return the attention mask according
to the specific tokenizer&#x2019;s default, defined by the <code>return_outputs</code> attribute.</p>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"return_attention_mask"},{anchor:"transformers.MistralCommonTokenizer.prepare_for_model.return_overflowing_tokens",description:`<strong>return_overflowing_tokens</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to return overflowing token sequences. If a pair of sequences of input ids (or a batch
of pairs) is provided with <code>truncation_strategy = longest_first</code> or <code>True</code>, an error is raised instead
of returning overflowing tokens.`,name:"return_overflowing_tokens"},{anchor:"transformers.MistralCommonTokenizer.prepare_for_model.return_special_tokens_mask",description:`<strong>return_special_tokens_mask</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to return special tokens mask information.`,name:"return_special_tokens_mask"},{anchor:"transformers.MistralCommonTokenizer.prepare_for_model.return_offsets_mapping",description:`<strong>return_offsets_mapping</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to return <code>(char_start, char_end)</code> for each token.</p>
<p>This is only available on fast tokenizers inheriting from <a href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast">PreTrainedTokenizerFast</a>, if using
Python&#x2019;s tokenizer, this method will raise <code>NotImplementedError</code>.`,name:"return_offsets_mapping"},{anchor:"transformers.MistralCommonTokenizer.prepare_for_model.return_length",description:`<strong>return_length</strong>  (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to return the lengths of the encoded inputs.`,name:"return_length"},{anchor:"transformers.MistralCommonTokenizer.prepare_for_model.verbose",description:`<strong>verbose</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to print more information and warnings.`,name:"verbose"},{anchor:"transformers.MistralCommonTokenizer.prepare_for_model.*kwargs",description:"*<strong>*kwargs</strong> &#x2014; passed to the <code>self.tokenize()</code> method",name:"*kwargs"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_mistral_common.py#L842",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.BatchEncoding"
>BatchEncoding</a> with the following fields:</p>
<ul>
<li>
<p><strong>input_ids</strong> — List of token ids to be fed to a model.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a></p>
</li>
<li>
<p><strong>attention_mask</strong> — List of indices specifying which tokens should be attended to by the model (when
<code>return_attention_mask=True</code> or if <em>“attention_mask”</em> is in <code>self.model_input_names</code>).</p>
<p><a href="../glossary#attention-mask">What are attention masks?</a></p>
</li>
<li>
<p><strong>overflowing_tokens</strong> — List of overflowing tokens sequences (when a <code>max_length</code> is specified and
<code>return_overflowing_tokens=True</code>).</p>
</li>
<li>
<p><strong>num_truncated_tokens</strong> — Number of tokens truncated (when a <code>max_length</code> is specified and
<code>return_overflowing_tokens=True</code>).</p>
</li>
<li>
<p><strong>special_tokens_mask</strong> — List of 0s and 1s, with 1 specifying added special tokens and 0 specifying
regular sequence tokens (when <code>add_special_tokens=True</code> and <code>return_special_tokens_mask=True</code>).</p>
</li>
<li>
<p><strong>length</strong> — The length of the inputs (when <code>return_length=True</code>)</p>
</li>
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><a
  href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.BatchEncoding"
>BatchEncoding</a></p>
`}}),He=new $({props:{name:"save_pretrained",anchor:"transformers.MistralCommonTokenizer.save_pretrained",parameters:[{name:"save_directory",val:": typing.Union[str, os.PathLike, pathlib.Path]"},{name:"push_to_hub",val:": bool = False"},{name:"token",val:": typing.Union[bool, str, NoneType] = None"},{name:"commit_message",val:": typing.Optional[str] = None"},{name:"repo_id",val:": typing.Optional[str] = None"},{name:"private",val:": typing.Optional[bool] = None"},{name:"repo_url",val:": typing.Optional[str] = None"},{name:"organization",val:": typing.Optional[str] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.MistralCommonTokenizer.save_pretrained.save_directory",description:"<strong>save_directory</strong> (<code>str</code> or <code>os.PathLike</code>) &#x2014; The path to a directory where the tokenizer will be saved.",name:"save_directory"},{anchor:"transformers.MistralCommonTokenizer.save_pretrained.push_to_hub",description:`<strong>push_to_hub</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to push your model to the Hugging Face model hub after saving it. You can specify the
repository you want to push to with <code>repo_id</code> (will default to the name of <code>save_directory</code> in your
namespace).`,name:"push_to_hub"},{anchor:"transformers.MistralCommonTokenizer.save_pretrained.token",description:`<strong>token</strong> (<code>str</code> or <em>bool</em>, <em>optional</em>, defaults to <code>None</code>) &#x2014;
The token to use to push to the model hub. If <code>True</code>, will use the token in the <code>HF_TOKEN</code> environment
variable.`,name:"token"},{anchor:"transformers.MistralCommonTokenizer.save_pretrained.commit_message",description:"<strong>commit_message</strong> (<code>str</code>, <em>optional</em>) &#x2014; The commit message to use when pushing to the hub.",name:"commit_message"},{anchor:"transformers.MistralCommonTokenizer.save_pretrained.repo_id",description:"<strong>repo_id</strong> (<code>str</code>, <em>optional</em>) &#x2014; The name of the repository to which push to the Hub.",name:"repo_id"},{anchor:"transformers.MistralCommonTokenizer.save_pretrained.private",description:"<strong>private</strong> (<code>bool</code>, <em>optional</em>) &#x2014; Whether the model repository is private or not.",name:"private"},{anchor:"transformers.MistralCommonTokenizer.save_pretrained.repo_url",description:"<strong>repo_url</strong> (<code>str</code>, <em>optional</em>) &#x2014; The URL to the Git repository to which push to the Hub.",name:"repo_url"},{anchor:"transformers.MistralCommonTokenizer.save_pretrained.organization",description:"<strong>organization</strong> (<code>str</code>, <em>optional</em>) &#x2014; The name of the organization in which you would like to push your model.",name:"organization"},{anchor:"transformers.MistralCommonTokenizer.save_pretrained.kwargs",description:`<strong>kwargs</strong> (<code>Dict[str, Any]</code>, <em>optional</em>) &#x2014;
Not supported by <code>MistralCommonTokenizer.save_pretrained</code>.
Will raise an error if used.`,name:"kwargs"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_mistral_common.py#L1816",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>The files saved.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A tuple of <code>str</code></p>
`}}),Re=new $({props:{name:"tokenize",anchor:"transformers.MistralCommonTokenizer.tokenize",parameters:[{name:"text",val:": str"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.MistralCommonTokenizer.tokenize.text",description:`<strong>text</strong> (<code>str</code>) &#x2014;
The sequence to be encoded.`,name:"text"},{anchor:"transformers.MistralCommonTokenizer.tokenize.*kwargs",description:`*<strong>*kwargs</strong> (additional keyword arguments) &#x2014;
Not supported by <code>MistralCommonTokenizer.tokenize</code>.
Will raise an error if used.`,name:"*kwargs"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_mistral_common.py#L606",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>The list of tokens.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>List[str]</code></p>
`}}),Xe=new $({props:{name:"truncate_sequences",anchor:"transformers.MistralCommonTokenizer.truncate_sequences",parameters:[{name:"ids",val:": list"},{name:"pair_ids",val:": None = None"},{name:"num_tokens_to_remove",val:": int = 0"},{name:"truncation_strategy",val:": typing.Union[str, transformers.tokenization_utils_base.TruncationStrategy] = 'longest_first'"},{name:"stride",val:": int = 0"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.MistralCommonTokenizer.truncate_sequences.ids",description:`<strong>ids</strong> (<code>List[int]</code>) &#x2014;
Tokenized input ids. Can be obtained from a string by chaining the <code>tokenize</code> and
<code>convert_tokens_to_ids</code> methods.`,name:"ids"},{anchor:"transformers.MistralCommonTokenizer.truncate_sequences.pair_ids",description:`<strong>pair_ids</strong> (<code>None</code>, <em>optional</em>) &#x2014;
Not supported by <code>MistralCommonTokenizer</code>. Kept to match the signature of <code>PreTrainedTokenizerBase.truncate_sequences</code>.`,name:"pair_ids"},{anchor:"transformers.MistralCommonTokenizer.truncate_sequences.num_tokens_to_remove",description:`<strong>num_tokens_to_remove</strong> (<code>int</code>, <em>optional</em>, defaults to 0) &#x2014;
Number of tokens to remove using the truncation strategy.`,name:"num_tokens_to_remove"},{anchor:"transformers.MistralCommonTokenizer.truncate_sequences.truncation_strategy",description:`<strong>truncation_strategy</strong> (<code>str</code> or <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.tokenization_utils_base.TruncationStrategy">TruncationStrategy</a>, <em>optional</em>, defaults to <code>&apos;longest_first&apos;</code>) &#x2014;
The strategy to follow for truncation. Can be:</p>
<ul>
<li><code>&apos;longest_first&apos;</code>: Truncate to a maximum length specified with the argument <code>max_length</code> or to the
maximum acceptable input length for the model if that argument is not provided.</li>
<li><code>&apos;do_not_truncate&apos;</code> (default): No truncation (i.e., can output batch with sequence lengths greater
than the model maximum admissible input size).</li>
</ul>`,name:"truncation_strategy"},{anchor:"transformers.MistralCommonTokenizer.truncate_sequences.stride",description:`<strong>stride</strong> (<code>int</code>, <em>optional</em>, defaults to 0) &#x2014;
If set to a positive number, the overflowing tokens returned will contain some tokens from the main
sequence returned. The value of this argument defines the number of additional tokens.`,name:"stride"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_mistral_common.py#L1293",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>The truncated <code>ids</code> and the list of
overflowing tokens. <code>None</code> is returned to match Transformers signature.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>Tuple[List[int], None, List[int]]</code></p>
`}}),Se=new _e({props:{title:"MistralModel",local:"transformers.MistralModel",headingTag:"h2"}}),Ee=new $({props:{name:"class transformers.MistralModel",anchor:"transformers.MistralModel",parameters:[{name:"config",val:": MistralConfig"}],parametersDescription:[{anchor:"transformers.MistralModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/mistral#transformers.MistralConfig">MistralConfig</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mistral/modeling_mistral.py#L307"}}),De=new $({props:{name:"forward",anchor:"transformers.MistralModel.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.MistralModel.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.MistralModel.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.MistralModel.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.MistralModel.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.MistralModel.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.MistralModel.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.MistralModel.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mistral/modeling_mistral.py#L324",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast"
>transformers.modeling_outputs.BaseModelOutputWithPast</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/mistral#transformers.MistralConfig"
>MistralConfig</a>) and inputs.</p>
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
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast"
>transformers.modeling_outputs.BaseModelOutputWithPast</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),ce=new co({props:{$$slots:{default:[ya]},$$scope:{ctx:w}}}),Ae=new _e({props:{title:"MistralForCausalLM",local:"transformers.MistralForCausalLM",headingTag:"h2"}}),Ye=new $({props:{name:"class transformers.MistralForCausalLM",anchor:"transformers.MistralForCausalLM",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.MistralForCausalLM.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/mistral#transformers.MistralForCausalLM">MistralForCausalLM</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mistral/modeling_mistral.py#L387"}}),Qe=new $({props:{name:"forward",anchor:"transformers.MistralForCausalLM.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"},{name:"logits_to_keep",val:": typing.Union[int, torch.Tensor] = 0"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.MistralForCausalLM.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.MistralForCausalLM.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.MistralForCausalLM.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.MistralForCausalLM.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.MistralForCausalLM.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.MistralForCausalLM.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.MistralForCausalLM.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.MistralForCausalLM.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"},{anchor:"transformers.MistralForCausalLM.forward.logits_to_keep",description:`<strong>logits_to_keep</strong> (<code>Union[int, torch.Tensor]</code>, defaults to <code>0</code>) &#x2014;
If an <code>int</code>, compute logits for the last <code>logits_to_keep</code> tokens. If <code>0</code>, calculate logits for all
<code>input_ids</code> (special case). Only last token logits are needed for generation, and calculating them only for that
token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
If a <code>torch.Tensor</code>, must be 1D corresponding to the indices to keep in the sequence length dimension.
This is useful when using packed tensor format (single dimension for batch and sequence length).`,name:"logits_to_keep"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mistral/modeling_mistral.py#L401",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast"
>transformers.modeling_outputs.CausalLMOutputWithPast</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/mistral#transformers.MistralConfig"
>MistralConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>loss</strong> (<code>torch.FloatTensor</code> of shape <code>(1,)</code>, <em>optional</em>, returned when <code>labels</code> is provided) — Language modeling loss (for next-token prediction).</p>
</li>
<li>
<p><strong>logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, config.vocab_size)</code>) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).</p>
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


<p><a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast"
>transformers.modeling_outputs.CausalLMOutputWithPast</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),me=new co({props:{$$slots:{default:[ba]},$$scope:{ctx:w}}}),pe=new as({props:{anchor:"transformers.MistralForCausalLM.forward.example",$$slots:{default:[ka]},$$scope:{ctx:w}}}),Oe=new _e({props:{title:"MistralForSequenceClassification",local:"transformers.MistralForSequenceClassification",headingTag:"h2"}}),Ke=new $({props:{name:"class transformers.MistralForSequenceClassification",anchor:"transformers.MistralForSequenceClassification",parameters:[{name:"config",val:""}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mistral/modeling_mistral.py#L466"}}),et=new $({props:{name:"forward",anchor:"transformers.MistralForSequenceClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.MistralForSequenceClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.MistralForSequenceClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.MistralForSequenceClassification.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.MistralForSequenceClassification.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.MistralForSequenceClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.MistralForSequenceClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.MistralForSequenceClassification.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/modeling_layers.py#L111",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <code>transformers.modeling_outputs.SequenceClassifierOutputWithPast</code> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<code>None</code>) and inputs.</p>
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
`}}),ue=new co({props:{$$slots:{default:[va]},$$scope:{ctx:w}}}),tt=new _e({props:{title:"MistralForTokenClassification",local:"transformers.MistralForTokenClassification",headingTag:"h2"}}),ot=new $({props:{name:"class transformers.MistralForTokenClassification",anchor:"transformers.MistralForTokenClassification",parameters:[{name:"config",val:""}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mistral/modeling_mistral.py#L462"}}),nt=new $({props:{name:"forward",anchor:"transformers.MistralForTokenClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.MistralForTokenClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.MistralForTokenClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.MistralForTokenClassification.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.MistralForTokenClassification.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.MistralForTokenClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.MistralForTokenClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.MistralForTokenClassification.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/modeling_layers.py#L254",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput"
>transformers.modeling_outputs.TokenClassifierOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<code>None</code>) and inputs.</p>
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
`}}),he=new co({props:{$$slots:{default:[Ta]},$$scope:{ctx:w}}}),st=new _e({props:{title:"MistralForQuestionAnswering",local:"transformers.MistralForQuestionAnswering",headingTag:"h2"}}),rt=new $({props:{name:"class transformers.MistralForQuestionAnswering",anchor:"transformers.MistralForQuestionAnswering",parameters:[{name:"config",val:""}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mistral/modeling_mistral.py#L470"}}),lt=new la({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/mistral.md"}}),{c(){o=l("meta"),b=n(),a=l("p"),v=n(),z=l("p"),z.innerHTML=M,T=n(),U=l("div"),U.innerHTML=mo,be=n(),u(N.$$.fragment),uo=n(),ke=l("p"),ke.innerHTML=is,ho=n(),ve=l("p"),ve.innerHTML=ls,fo=n(),u(D.$$.fragment),go=n(),Te=l("p"),Te.innerHTML=ds,_o=n(),u(A.$$.fragment),yo=n(),Me=l("p"),Me.innerHTML=cs,bo=n(),we=l("p"),we.innerHTML=ms,ko=n(),u(xe.$$.fragment),vo=n(),Ce=l("p"),Ce.innerHTML=ps,To=n(),u(ze.$$.fragment),Mo=n(),Y=l("div"),Y.innerHTML=us,wo=n(),u($e.$$.fragment),xo=n(),I=l("div"),u(Ue.$$.fragment),So=n(),_t=l("p"),_t.innerHTML=hs,Eo=n(),yt=l("p"),yt.innerHTML=fs,Do=n(),bt=l("p"),bt.innerHTML=gs,Ao=n(),u(Q.$$.fragment),Co=n(),u(Ie.$$.fragment),zo=n(),p=l("div"),u(je.$$.fragment),Yo=n(),kt=l("p"),kt.innerHTML=_s,Qo=n(),u(O.$$.fragment),Oo=n(),vt=l("p"),vt.textContent=ys,Ko=n(),Tt=l("p"),Tt.innerHTML=bs,en=n(),Mt=l("p"),Mt.innerHTML=ks,tn=n(),wt=l("p"),wt.innerHTML=vs,on=n(),xt=l("ul"),xt.innerHTML=Ts,nn=n(),Ct=l("p"),Ct.innerHTML=Ms,sn=n(),zt=l("ul"),zt.innerHTML=ws,an=n(),$t=l("p"),$t.innerHTML=xs,rn=n(),K=l("div"),u(Je.$$.fragment),ln=n(),Ut=l("p"),Ut.innerHTML=Cs,dn=n(),ee=l("div"),u(qe.$$.fragment),cn=n(),It=l("p"),It.textContent=zs,mn=n(),te=l("div"),u(Fe.$$.fragment),pn=n(),jt=l("p"),jt.textContent=$s,un=n(),oe=l("div"),u(Le.$$.fragment),hn=n(),Jt=l("p"),Jt.textContent=Us,fn=n(),ne=l("div"),u(We.$$.fragment),gn=n(),qt=l("p"),qt.textContent=Is,_n=n(),se=l("div"),u(Ne.$$.fragment),yn=n(),Ft=l("p"),Ft.textContent=js,bn=n(),ae=l("div"),u(Be.$$.fragment),kn=n(),Lt=l("p"),Lt.innerHTML=Js,vn=n(),re=l("div"),u(Ge.$$.fragment),Tn=n(),Wt=l("p"),Wt.innerHTML=qs,Mn=n(),B=l("div"),u(Pe.$$.fragment),wn=n(),Nt=l("p"),Nt.textContent=Fs,xn=n(),Bt=l("p"),Bt.textContent=Ls,Cn=n(),F=l("div"),u(Ve.$$.fragment),zn=n(),Gt=l("p"),Gt.textContent=Ws,$n=n(),Pt=l("p"),Pt.innerHTML=Ns,Un=n(),u(ie.$$.fragment),In=n(),le=l("div"),u(Ze.$$.fragment),jn=n(),Vt=l("p"),Vt.textContent=Bs,Jn=n(),G=l("div"),u(He.$$.fragment),qn=n(),Zt=l("p"),Zt.textContent=Gs,Fn=n(),Ht=l("p"),Ht.innerHTML=Ps,Ln=n(),P=l("div"),u(Re.$$.fragment),Wn=n(),Rt=l("p"),Rt.textContent=Vs,Nn=n(),Xt=l("p"),Xt.textContent=Zs,Bn=n(),de=l("div"),u(Xe.$$.fragment),Gn=n(),St=l("p"),St.textContent=Hs,$o=n(),u(Se.$$.fragment),Uo=n(),j=l("div"),u(Ee.$$.fragment),Pn=n(),Et=l("p"),Et.textContent=Rs,Vn=n(),Dt=l("p"),Dt.innerHTML=Xs,Zn=n(),At=l("p"),At.innerHTML=Ss,Hn=n(),V=l("div"),u(De.$$.fragment),Rn=n(),Yt=l("p"),Yt.innerHTML=Es,Xn=n(),u(ce.$$.fragment),Io=n(),u(Ae.$$.fragment),jo=n(),J=l("div"),u(Ye.$$.fragment),Sn=n(),Qt=l("p"),Qt.textContent=Ds,En=n(),Ot=l("p"),Ot.innerHTML=As,Dn=n(),Kt=l("p"),Kt.innerHTML=Ys,An=n(),L=l("div"),u(Qe.$$.fragment),Yn=n(),eo=l("p"),eo.innerHTML=Qs,Qn=n(),u(me.$$.fragment),On=n(),u(pe.$$.fragment),Jo=n(),u(Oe.$$.fragment),qo=n(),S=l("div"),u(Ke.$$.fragment),Kn=n(),Z=l("div"),u(et.$$.fragment),es=n(),to=l("p"),to.innerHTML=Os,ts=n(),u(ue.$$.fragment),Fo=n(),u(tt.$$.fragment),Lo=n(),E=l("div"),u(ot.$$.fragment),os=n(),H=l("div"),u(nt.$$.fragment),ns=n(),oo=l("p"),oo.innerHTML=Ks,ss=n(),u(he.$$.fragment),Wo=n(),u(st.$$.fragment),No=n(),at=l("div"),u(rt.$$.fragment),Bo=n(),it=l("ul"),it.innerHTML=ea,Go=n(),u(lt.$$.fragment),Po=n(),po=l("p"),this.h()},l(e){const i=ra("svelte-u9bgzb",document.head);o=d(i,"META",{name:!0,content:!0}),i.forEach(r),b=s(e),a=d(e,"P",{}),C(a).forEach(r),v=s(e),z=d(e,"P",{"data-svelte-h":!0}),m(z)!=="svelte-1k2f5s0"&&(z.innerHTML=M),T=s(e),U=d(e,"DIV",{style:!0,"data-svelte-h":!0}),m(U)!=="svelte-11gpmgv"&&(U.innerHTML=mo),be=s(e),h(N.$$.fragment,e),uo=s(e),ke=d(e,"P",{"data-svelte-h":!0}),m(ke)!=="svelte-nglnqg"&&(ke.innerHTML=is),ho=s(e),ve=d(e,"P",{"data-svelte-h":!0}),m(ve)!=="svelte-o8wgwi"&&(ve.innerHTML=ls),fo=s(e),h(D.$$.fragment,e),go=s(e),Te=d(e,"P",{"data-svelte-h":!0}),m(Te)!=="svelte-7xg9xl"&&(Te.innerHTML=ds),_o=s(e),h(A.$$.fragment,e),yo=s(e),Me=d(e,"P",{"data-svelte-h":!0}),m(Me)!=="svelte-nf5ooi"&&(Me.innerHTML=cs),bo=s(e),we=d(e,"P",{"data-svelte-h":!0}),m(we)!=="svelte-60nsd0"&&(we.innerHTML=ms),ko=s(e),h(xe.$$.fragment,e),vo=s(e),Ce=d(e,"P",{"data-svelte-h":!0}),m(Ce)!=="svelte-w3z5ks"&&(Ce.innerHTML=ps),To=s(e),h(ze.$$.fragment,e),Mo=s(e),Y=d(e,"DIV",{class:!0,"data-svelte-h":!0}),m(Y)!=="svelte-diufpa"&&(Y.innerHTML=us),wo=s(e),h($e.$$.fragment,e),xo=s(e),I=d(e,"DIV",{class:!0});var q=C(I);h(Ue.$$.fragment,q),So=s(q),_t=d(q,"P",{"data-svelte-h":!0}),m(_t)!=="svelte-1eo8bk9"&&(_t.innerHTML=hs),Eo=s(q),yt=d(q,"P",{"data-svelte-h":!0}),m(yt)!=="svelte-28p57e"&&(yt.innerHTML=fs),Do=s(q),bt=d(q,"P",{"data-svelte-h":!0}),m(bt)!=="svelte-1ek1ss9"&&(bt.innerHTML=gs),Ao=s(q),h(Q.$$.fragment,q),q.forEach(r),Co=s(e),h(Ie.$$.fragment,e),zo=s(e),p=d(e,"DIV",{class:!0});var k=C(p);h(je.$$.fragment,k),Yo=s(k),kt=d(k,"P",{"data-svelte-h":!0}),m(kt)!=="svelte-iuk2y8"&&(kt.innerHTML=_s),Qo=s(k),h(O.$$.fragment,k),Oo=s(k),vt=d(k,"P",{"data-svelte-h":!0}),m(vt)!=="svelte-kud278"&&(vt.textContent=ys),Ko=s(k),Tt=d(k,"P",{"data-svelte-h":!0}),m(Tt)!=="svelte-ifzpy9"&&(Tt.innerHTML=bs),en=s(k),Mt=d(k,"P",{"data-svelte-h":!0}),m(Mt)!=="svelte-ktmcb2"&&(Mt.innerHTML=ks),tn=s(k),wt=d(k,"P",{"data-svelte-h":!0}),m(wt)!=="svelte-mzof2m"&&(wt.innerHTML=vs),on=s(k),xt=d(k,"UL",{"data-svelte-h":!0}),m(xt)!=="svelte-1hlq74o"&&(xt.innerHTML=Ts),nn=s(k),Ct=d(k,"P",{"data-svelte-h":!0}),m(Ct)!=="svelte-k8piyc"&&(Ct.innerHTML=Ms),sn=s(k),zt=d(k,"UL",{"data-svelte-h":!0}),m(zt)!=="svelte-mjbefh"&&(zt.innerHTML=ws),an=s(k),$t=d(k,"P",{"data-svelte-h":!0}),m($t)!=="svelte-18hne1"&&($t.innerHTML=xs),rn=s(k),K=d(k,"DIV",{class:!0});var dt=C(K);h(Je.$$.fragment,dt),ln=s(dt),Ut=d(dt,"P",{"data-svelte-h":!0}),m(Ut)!=="svelte-sr2voc"&&(Ut.innerHTML=Cs),dt.forEach(r),dn=s(k),ee=d(k,"DIV",{class:!0});var ct=C(ee);h(qe.$$.fragment,ct),cn=s(ct),It=d(ct,"P",{"data-svelte-h":!0}),m(It)!=="svelte-1deng2j"&&(It.textContent=zs),ct.forEach(r),mn=s(k),te=d(k,"DIV",{class:!0});var mt=C(te);h(Fe.$$.fragment,mt),pn=s(mt),jt=d(mt,"P",{"data-svelte-h":!0}),m(jt)!=="svelte-cx157h"&&(jt.textContent=$s),mt.forEach(r),un=s(k),oe=d(k,"DIV",{class:!0});var pt=C(oe);h(Le.$$.fragment,pt),hn=s(pt),Jt=d(pt,"P",{"data-svelte-h":!0}),m(Jt)!=="svelte-1urz5jj"&&(Jt.textContent=Us),pt.forEach(r),fn=s(k),ne=d(k,"DIV",{class:!0});var ut=C(ne);h(We.$$.fragment,ut),gn=s(ut),qt=d(ut,"P",{"data-svelte-h":!0}),m(qt)!=="svelte-vbfkpu"&&(qt.textContent=Is),ut.forEach(r),_n=s(k),se=d(k,"DIV",{class:!0});var ht=C(se);h(Ne.$$.fragment,ht),yn=s(ht),Ft=d(ht,"P",{"data-svelte-h":!0}),m(Ft)!=="svelte-12b8hzo"&&(Ft.textContent=js),ht.forEach(r),bn=s(k),ae=d(k,"DIV",{class:!0});var ft=C(ae);h(Be.$$.fragment,ft),kn=s(ft),Lt=d(ft,"P",{"data-svelte-h":!0}),m(Lt)!=="svelte-5j01oy"&&(Lt.innerHTML=Js),ft.forEach(r),vn=s(k),re=d(k,"DIV",{class:!0});var gt=C(re);h(Ge.$$.fragment,gt),Tn=s(gt),Wt=d(gt,"P",{"data-svelte-h":!0}),m(Wt)!=="svelte-1wmjg8a"&&(Wt.innerHTML=qs),gt.forEach(r),Mn=s(k),B=d(k,"DIV",{class:!0});var no=C(B);h(Pe.$$.fragment,no),wn=s(no),Nt=d(no,"P",{"data-svelte-h":!0}),m(Nt)!=="svelte-1gbatu6"&&(Nt.textContent=Fs),xn=s(no),Bt=d(no,"P",{"data-svelte-h":!0}),m(Bt)!=="svelte-1d4v47d"&&(Bt.textContent=Ls),no.forEach(r),Cn=s(k),F=d(k,"DIV",{class:!0});var fe=C(F);h(Ve.$$.fragment,fe),zn=s(fe),Gt=d(fe,"P",{"data-svelte-h":!0}),m(Gt)!=="svelte-1n892mi"&&(Gt.textContent=Ws),$n=s(fe),Pt=d(fe,"P",{"data-svelte-h":!0}),m(Pt)!=="svelte-954lq4"&&(Pt.innerHTML=Ns),Un=s(fe),h(ie.$$.fragment,fe),fe.forEach(r),In=s(k),le=d(k,"DIV",{class:!0});var Zo=C(le);h(Ze.$$.fragment,Zo),jn=s(Zo),Vt=d(Zo,"P",{"data-svelte-h":!0}),m(Vt)!=="svelte-15kr77e"&&(Vt.textContent=Bs),Zo.forEach(r),Jn=s(k),G=d(k,"DIV",{class:!0});var so=C(G);h(He.$$.fragment,so),qn=s(so),Zt=d(so,"P",{"data-svelte-h":!0}),m(Zt)!=="svelte-u73u19"&&(Zt.textContent=Gs),Fn=s(so),Ht=d(so,"P",{"data-svelte-h":!0}),m(Ht)!=="svelte-oagoqu"&&(Ht.innerHTML=Ps),so.forEach(r),Ln=s(k),P=d(k,"DIV",{class:!0});var ao=C(P);h(Re.$$.fragment,ao),Wn=s(ao),Rt=d(ao,"P",{"data-svelte-h":!0}),m(Rt)!=="svelte-sso1qb"&&(Rt.textContent=Vs),Nn=s(ao),Xt=d(ao,"P",{"data-svelte-h":!0}),m(Xt)!=="svelte-46tdba"&&(Xt.textContent=Zs),ao.forEach(r),Bn=s(k),de=d(k,"DIV",{class:!0});var Ho=C(de);h(Xe.$$.fragment,Ho),Gn=s(Ho),St=d(Ho,"P",{"data-svelte-h":!0}),m(St)!=="svelte-fkofn"&&(St.textContent=Hs),Ho.forEach(r),k.forEach(r),$o=s(e),h(Se.$$.fragment,e),Uo=s(e),j=d(e,"DIV",{class:!0});var R=C(j);h(Ee.$$.fragment,R),Pn=s(R),Et=d(R,"P",{"data-svelte-h":!0}),m(Et)!=="svelte-1g3elk2"&&(Et.textContent=Rs),Vn=s(R),Dt=d(R,"P",{"data-svelte-h":!0}),m(Dt)!=="svelte-q52n56"&&(Dt.innerHTML=Xs),Zn=s(R),At=d(R,"P",{"data-svelte-h":!0}),m(At)!=="svelte-hswkmf"&&(At.innerHTML=Ss),Hn=s(R),V=d(R,"DIV",{class:!0});var ro=C(V);h(De.$$.fragment,ro),Rn=s(ro),Yt=d(ro,"P",{"data-svelte-h":!0}),m(Yt)!=="svelte-1m3m04p"&&(Yt.innerHTML=Es),Xn=s(ro),h(ce.$$.fragment,ro),ro.forEach(r),R.forEach(r),Io=s(e),h(Ae.$$.fragment,e),jo=s(e),J=d(e,"DIV",{class:!0});var X=C(J);h(Ye.$$.fragment,X),Sn=s(X),Qt=d(X,"P",{"data-svelte-h":!0}),m(Qt)!=="svelte-2fpz9j"&&(Qt.textContent=Ds),En=s(X),Ot=d(X,"P",{"data-svelte-h":!0}),m(Ot)!=="svelte-q52n56"&&(Ot.innerHTML=As),Dn=s(X),Kt=d(X,"P",{"data-svelte-h":!0}),m(Kt)!=="svelte-hswkmf"&&(Kt.innerHTML=Ys),An=s(X),L=d(X,"DIV",{class:!0});var ge=C(L);h(Qe.$$.fragment,ge),Yn=s(ge),eo=d(ge,"P",{"data-svelte-h":!0}),m(eo)!=="svelte-ols2n1"&&(eo.innerHTML=Qs),Qn=s(ge),h(me.$$.fragment,ge),On=s(ge),h(pe.$$.fragment,ge),ge.forEach(r),X.forEach(r),Jo=s(e),h(Oe.$$.fragment,e),qo=s(e),S=d(e,"DIV",{class:!0});var Ro=C(S);h(Ke.$$.fragment,Ro),Kn=s(Ro),Z=d(Ro,"DIV",{class:!0});var io=C(Z);h(et.$$.fragment,io),es=s(io),to=d(io,"P",{"data-svelte-h":!0}),m(to)!=="svelte-1sal4ui"&&(to.innerHTML=Os),ts=s(io),h(ue.$$.fragment,io),io.forEach(r),Ro.forEach(r),Fo=s(e),h(tt.$$.fragment,e),Lo=s(e),E=d(e,"DIV",{class:!0});var Xo=C(E);h(ot.$$.fragment,Xo),os=s(Xo),H=d(Xo,"DIV",{class:!0});var lo=C(H);h(nt.$$.fragment,lo),ns=s(lo),oo=d(lo,"P",{"data-svelte-h":!0}),m(oo)!=="svelte-1py4aay"&&(oo.innerHTML=Ks),ss=s(lo),h(he.$$.fragment,lo),lo.forEach(r),Xo.forEach(r),Wo=s(e),h(st.$$.fragment,e),No=s(e),at=d(e,"DIV",{class:!0});var ta=C(at);h(rt.$$.fragment,ta),ta.forEach(r),Bo=s(e),it=d(e,"UL",{"data-svelte-h":!0}),m(it)!=="svelte-n3ow4o"&&(it.innerHTML=ea),Go=s(e),h(lt.$$.fragment,e),Po=s(e),po=d(e,"P",{}),C(po).forEach(r),this.h()},h(){x(o,"name","hf:doc:metadata"),x(o,"content",wa),ia(U,"float","right"),x(Y,"class","flex justify-center"),x(I,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),x(K,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),x(ee,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),x(te,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),x(oe,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),x(ne,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),x(se,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),x(ae,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),x(re,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),x(B,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),x(F,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),x(le,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),x(G,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),x(P,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),x(de,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),x(p,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),x(V,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),x(j,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),x(L,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),x(J,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),x(Z,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),x(S,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),x(H,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),x(E,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),x(at,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,i){t(document.head,o),c(e,b,i),c(e,a,i),c(e,v,i),c(e,z,i),c(e,T,i),c(e,U,i),c(e,be,i),f(N,e,i),c(e,uo,i),c(e,ke,i),c(e,ho,i),c(e,ve,i),c(e,fo,i),f(D,e,i),c(e,go,i),c(e,Te,i),c(e,_o,i),f(A,e,i),c(e,yo,i),c(e,Me,i),c(e,bo,i),c(e,we,i),c(e,ko,i),f(xe,e,i),c(e,vo,i),c(e,Ce,i),c(e,To,i),f(ze,e,i),c(e,Mo,i),c(e,Y,i),c(e,wo,i),f($e,e,i),c(e,xo,i),c(e,I,i),f(Ue,I,null),t(I,So),t(I,_t),t(I,Eo),t(I,yt),t(I,Do),t(I,bt),t(I,Ao),f(Q,I,null),c(e,Co,i),f(Ie,e,i),c(e,zo,i),c(e,p,i),f(je,p,null),t(p,Yo),t(p,kt),t(p,Qo),f(O,p,null),t(p,Oo),t(p,vt),t(p,Ko),t(p,Tt),t(p,en),t(p,Mt),t(p,tn),t(p,wt),t(p,on),t(p,xt),t(p,nn),t(p,Ct),t(p,sn),t(p,zt),t(p,an),t(p,$t),t(p,rn),t(p,K),f(Je,K,null),t(K,ln),t(K,Ut),t(p,dn),t(p,ee),f(qe,ee,null),t(ee,cn),t(ee,It),t(p,mn),t(p,te),f(Fe,te,null),t(te,pn),t(te,jt),t(p,un),t(p,oe),f(Le,oe,null),t(oe,hn),t(oe,Jt),t(p,fn),t(p,ne),f(We,ne,null),t(ne,gn),t(ne,qt),t(p,_n),t(p,se),f(Ne,se,null),t(se,yn),t(se,Ft),t(p,bn),t(p,ae),f(Be,ae,null),t(ae,kn),t(ae,Lt),t(p,vn),t(p,re),f(Ge,re,null),t(re,Tn),t(re,Wt),t(p,Mn),t(p,B),f(Pe,B,null),t(B,wn),t(B,Nt),t(B,xn),t(B,Bt),t(p,Cn),t(p,F),f(Ve,F,null),t(F,zn),t(F,Gt),t(F,$n),t(F,Pt),t(F,Un),f(ie,F,null),t(p,In),t(p,le),f(Ze,le,null),t(le,jn),t(le,Vt),t(p,Jn),t(p,G),f(He,G,null),t(G,qn),t(G,Zt),t(G,Fn),t(G,Ht),t(p,Ln),t(p,P),f(Re,P,null),t(P,Wn),t(P,Rt),t(P,Nn),t(P,Xt),t(p,Bn),t(p,de),f(Xe,de,null),t(de,Gn),t(de,St),c(e,$o,i),f(Se,e,i),c(e,Uo,i),c(e,j,i),f(Ee,j,null),t(j,Pn),t(j,Et),t(j,Vn),t(j,Dt),t(j,Zn),t(j,At),t(j,Hn),t(j,V),f(De,V,null),t(V,Rn),t(V,Yt),t(V,Xn),f(ce,V,null),c(e,Io,i),f(Ae,e,i),c(e,jo,i),c(e,J,i),f(Ye,J,null),t(J,Sn),t(J,Qt),t(J,En),t(J,Ot),t(J,Dn),t(J,Kt),t(J,An),t(J,L),f(Qe,L,null),t(L,Yn),t(L,eo),t(L,Qn),f(me,L,null),t(L,On),f(pe,L,null),c(e,Jo,i),f(Oe,e,i),c(e,qo,i),c(e,S,i),f(Ke,S,null),t(S,Kn),t(S,Z),f(et,Z,null),t(Z,es),t(Z,to),t(Z,ts),f(ue,Z,null),c(e,Fo,i),f(tt,e,i),c(e,Lo,i),c(e,E,i),f(ot,E,null),t(E,os),t(E,H),f(nt,H,null),t(H,ns),t(H,oo),t(H,ss),f(he,H,null),c(e,Wo,i),f(st,e,i),c(e,No,i),c(e,at,i),f(rt,at,null),c(e,Bo,i),c(e,it,i),c(e,Go,i),f(lt,e,i),c(e,Po,i),c(e,po,i),Vo=!0},p(e,[i]){const q={};i&2&&(q.$$scope={dirty:i,ctx:e}),D.$set(q);const k={};i&2&&(k.$$scope={dirty:i,ctx:e}),A.$set(k);const dt={};i&2&&(dt.$$scope={dirty:i,ctx:e}),Q.$set(dt);const ct={};i&2&&(ct.$$scope={dirty:i,ctx:e}),O.$set(ct);const mt={};i&2&&(mt.$$scope={dirty:i,ctx:e}),ie.$set(mt);const pt={};i&2&&(pt.$$scope={dirty:i,ctx:e}),ce.$set(pt);const ut={};i&2&&(ut.$$scope={dirty:i,ctx:e}),me.$set(ut);const ht={};i&2&&(ht.$$scope={dirty:i,ctx:e}),pe.$set(ht);const ft={};i&2&&(ft.$$scope={dirty:i,ctx:e}),ue.$set(ft);const gt={};i&2&&(gt.$$scope={dirty:i,ctx:e}),he.$set(gt)},i(e){Vo||(g(N.$$.fragment,e),g(D.$$.fragment,e),g(A.$$.fragment,e),g(xe.$$.fragment,e),g(ze.$$.fragment,e),g($e.$$.fragment,e),g(Ue.$$.fragment,e),g(Q.$$.fragment,e),g(Ie.$$.fragment,e),g(je.$$.fragment,e),g(O.$$.fragment,e),g(Je.$$.fragment,e),g(qe.$$.fragment,e),g(Fe.$$.fragment,e),g(Le.$$.fragment,e),g(We.$$.fragment,e),g(Ne.$$.fragment,e),g(Be.$$.fragment,e),g(Ge.$$.fragment,e),g(Pe.$$.fragment,e),g(Ve.$$.fragment,e),g(ie.$$.fragment,e),g(Ze.$$.fragment,e),g(He.$$.fragment,e),g(Re.$$.fragment,e),g(Xe.$$.fragment,e),g(Se.$$.fragment,e),g(Ee.$$.fragment,e),g(De.$$.fragment,e),g(ce.$$.fragment,e),g(Ae.$$.fragment,e),g(Ye.$$.fragment,e),g(Qe.$$.fragment,e),g(me.$$.fragment,e),g(pe.$$.fragment,e),g(Oe.$$.fragment,e),g(Ke.$$.fragment,e),g(et.$$.fragment,e),g(ue.$$.fragment,e),g(tt.$$.fragment,e),g(ot.$$.fragment,e),g(nt.$$.fragment,e),g(he.$$.fragment,e),g(st.$$.fragment,e),g(rt.$$.fragment,e),g(lt.$$.fragment,e),Vo=!0)},o(e){_(N.$$.fragment,e),_(D.$$.fragment,e),_(A.$$.fragment,e),_(xe.$$.fragment,e),_(ze.$$.fragment,e),_($e.$$.fragment,e),_(Ue.$$.fragment,e),_(Q.$$.fragment,e),_(Ie.$$.fragment,e),_(je.$$.fragment,e),_(O.$$.fragment,e),_(Je.$$.fragment,e),_(qe.$$.fragment,e),_(Fe.$$.fragment,e),_(Le.$$.fragment,e),_(We.$$.fragment,e),_(Ne.$$.fragment,e),_(Be.$$.fragment,e),_(Ge.$$.fragment,e),_(Pe.$$.fragment,e),_(Ve.$$.fragment,e),_(ie.$$.fragment,e),_(Ze.$$.fragment,e),_(He.$$.fragment,e),_(Re.$$.fragment,e),_(Xe.$$.fragment,e),_(Se.$$.fragment,e),_(Ee.$$.fragment,e),_(De.$$.fragment,e),_(ce.$$.fragment,e),_(Ae.$$.fragment,e),_(Ye.$$.fragment,e),_(Qe.$$.fragment,e),_(me.$$.fragment,e),_(pe.$$.fragment,e),_(Oe.$$.fragment,e),_(Ke.$$.fragment,e),_(et.$$.fragment,e),_(ue.$$.fragment,e),_(tt.$$.fragment,e),_(ot.$$.fragment,e),_(nt.$$.fragment,e),_(he.$$.fragment,e),_(st.$$.fragment,e),_(rt.$$.fragment,e),_(lt.$$.fragment,e),Vo=!1},d(e){e&&(r(b),r(a),r(v),r(z),r(T),r(U),r(be),r(uo),r(ke),r(ho),r(ve),r(fo),r(go),r(Te),r(_o),r(yo),r(Me),r(bo),r(we),r(ko),r(vo),r(Ce),r(To),r(Mo),r(Y),r(wo),r(xo),r(I),r(Co),r(zo),r(p),r($o),r(Uo),r(j),r(Io),r(jo),r(J),r(Jo),r(qo),r(S),r(Fo),r(Lo),r(E),r(Wo),r(No),r(at),r(Bo),r(it),r(Go),r(Po),r(po)),r(o),y(N,e),y(D,e),y(A,e),y(xe,e),y(ze,e),y($e,e),y(Ue),y(Q),y(Ie,e),y(je),y(O),y(Je),y(qe),y(Fe),y(Le),y(We),y(Ne),y(Be),y(Ge),y(Pe),y(Ve),y(ie),y(Ze),y(He),y(Re),y(Xe),y(Se,e),y(Ee),y(De),y(ce),y(Ae,e),y(Ye),y(Qe),y(me),y(pe),y(Oe,e),y(Ke),y(et),y(ue),y(tt,e),y(ot),y(nt),y(he),y(st,e),y(rt),y(lt,e)}}}const wa='{"title":"Mistral","local":"mistral","sections":[{"title":"MistralConfig","local":"transformers.MistralConfig","sections":[],"depth":2},{"title":"MistralCommonTokenizer","local":"transformers.MistralCommonTokenizer","sections":[],"depth":2},{"title":"MistralModel","local":"transformers.MistralModel","sections":[],"depth":2},{"title":"MistralForCausalLM","local":"transformers.MistralForCausalLM","sections":[],"depth":2},{"title":"MistralForSequenceClassification","local":"transformers.MistralForSequenceClassification","sections":[],"depth":2},{"title":"MistralForTokenClassification","local":"transformers.MistralForTokenClassification","sections":[],"depth":2},{"title":"MistralForQuestionAnswering","local":"transformers.MistralForQuestionAnswering","sections":[],"depth":2}],"depth":1}';function xa(w){return na(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class Fa extends sa{constructor(o){super(),aa(this,o,xa,Ma,oa,{})}}export{Fa as component};
