import{s as Vt,o as Xt,n as x}from"../chunks/scheduler.18a86fab.js";import{S as Lt,i as Rt,g as p,s as r,r as f,m as St,A as Et,h as u,f as s,c as i,j as z,x as w,u as g,n as Ht,k as U,l as Yt,y as c,a as m,v as _,d as T,t as b,w as y}from"../chunks/index.98837b22.js";import{T as ao}from"../chunks/Tip.77304350.js";import{D as X}from"../chunks/Docstring.a1ef7999.js";import{C as K}from"../chunks/CodeBlock.8d0c2e8a.js";import{E as ro}from"../chunks/ExampleCodeBlock.8c3ee1f9.js";import{H as ue,E as Qt}from"../chunks/getInferenceSnippets.06c2775f.js";import{H as Pt,a as ft}from"../chunks/HfOption.6641485e.js";function At(v){let o,d="Click on the T5Gemma models in the right sidebar for more examples of how to apply T5Gemma to different language tasks.";return{c(){o=p("p"),o.textContent=d},l(t){o=u(t,"P",{"data-svelte-h":!0}),w(o)!=="svelte-1nu5ecx"&&(o.textContent=d)},m(t,l){m(t,o,l)},p:x,d(t){t&&s(o)}}}function Dt(v){let o,d;return o=new K({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwcGlwZWxpbmUlMEElMEFwaXBlJTIwJTNEJTIwcGlwZWxpbmUoJTBBJTIwJTIwJTIwJTIwJTIydGV4dDJ0ZXh0LWdlbmVyYXRpb24lMjIlMkMlMEElMjAlMjAlMjAlMjBtb2RlbCUzRCUyMmdvb2dsZSUyRnQ1Z2VtbWEtMmItMmItcHJlZml4bG0taXQlMjIlMkMlMEElMjAlMjAlMjAlMjBkdHlwZSUzRHRvcmNoLmJmbG9hdDE2JTJDJTBBJTIwJTIwJTIwJTIwZGV2aWNlX21hcCUzRCUyMmF1dG8lMjIlMkMlMEEpJTBBJTBBbWVzc2FnZXMlMjAlM0QlMjAlNUIlMEElMjAlMjAlMjAlMjAlN0IlMjJyb2xlJTIyJTNBJTIwJTIydXNlciUyMiUyQyUyMCUyMmNvbnRlbnQlMjIlM0ElMjAlMjJUZWxsJTIwbWUlMjBhbiUyMHVua25vd24lMjBpbnRlcmVzdGluZyUyMGJpb2xvZ3klMjBmYWN0JTIwYWJvdXQlMjB0aGUlMjBicmFpbi4lMjIlN0QlMkMlMEElNUQlMEFwcm9tcHQlMjAlM0QlMjBwaXBlLnRva2VuaXplci5hcHBseV9jaGF0X3RlbXBsYXRlKG1lc3NhZ2VzJTJDJTIwdG9rZW5pemUlM0RGYWxzZSUyQyUyMGFkZF9nZW5lcmF0aW9uX3Byb21wdCUzRFRydWUpJTBBJTBBcGlwZShwcm9tcHQlMkMlMjBtYXhfbmV3X3Rva2VucyUzRDMyKQ==",highlighted:`<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> pipeline

pipe = pipeline(
    <span class="hljs-string">&quot;text2text-generation&quot;</span>,
    model=<span class="hljs-string">&quot;google/t5gemma-2b-2b-prefixlm-it&quot;</span>,
    dtype=torch.bfloat16,
    device_map=<span class="hljs-string">&quot;auto&quot;</span>,
)

messages = [
    {<span class="hljs-string">&quot;role&quot;</span>: <span class="hljs-string">&quot;user&quot;</span>, <span class="hljs-string">&quot;content&quot;</span>: <span class="hljs-string">&quot;Tell me an unknown interesting biology fact about the brain.&quot;</span>},
]
prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=<span class="hljs-literal">False</span>, add_generation_prompt=<span class="hljs-literal">True</span>)

pipe(prompt, max_new_tokens=<span class="hljs-number">32</span>)`,wrap:!1}}),{c(){f(o.$$.fragment)},l(t){g(o.$$.fragment,t)},m(t,l){_(o,t,l),d=!0},p:x,i(t){d||(T(o.$$.fragment,t),d=!0)},o(t){b(o.$$.fragment,t),d=!1},d(t){y(o,t)}}}function Ot(v){let o,d;return o=new K({props:{code:"JTIzJTIwcGlwJTIwaW5zdGFsbCUyMGFjY2VsZXJhdGUlMEFpbXBvcnQlMjB0b3JjaCUwQWZyb20lMjB0cmFuc2Zvcm1lcnMlMjBpbXBvcnQlMjBBdXRvVG9rZW5pemVyJTJDJTIwQXV0b01vZGVsRm9yU2VxMlNlcUxNJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIyZ29vZ2xlJTJGdDVnZW1tYS0yYi0yYi1wcmVmaXhsbS1pdCUyMiklMEFtb2RlbCUyMCUzRCUyMEF1dG9Nb2RlbEZvclNlcTJTZXFMTS5mcm9tX3ByZXRyYWluZWQoJTBBJTIwJTIwJTIwJTIwJTIyZ29vZ2xlJTJGdDVnZW1tYS0yYi0yYi1wcmVmaXhsbS1pdCUyMiUyQyUwQSUyMCUyMCUyMCUyMGRldmljZV9tYXAlM0QlMjJhdXRvJTIyJTJDJTBBJTIwJTIwJTIwJTIwZHR5cGUlM0R0b3JjaC5iZmxvYXQxNiUyQyUwQSklMEElMEFtZXNzYWdlcyUyMCUzRCUyMCU1QiUwQSUyMCUyMCUyMCUyMCU3QiUyMnJvbGUlMjIlM0ElMjAlMjJ1c2VyJTIyJTJDJTIwJTIyY29udGVudCUyMiUzQSUyMCUyMlRlbGwlMjBtZSUyMGFuJTIwdW5rbm93biUyMGludGVyZXN0aW5nJTIwYmlvbG9neSUyMGZhY3QlMjBhYm91dCUyMHRoZSUyMGJyYWluLiUyMiU3RCUyQyUwQSU1RCUwQWlucHV0X2lkcyUyMCUzRCUyMHRva2VuaXplci5hcHBseV9jaGF0X3RlbXBsYXRlKG1lc3NhZ2VzJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiUyQyUyMHJldHVybl9kaWN0JTNEVHJ1ZSUyQyUyMGFkZF9nZW5lcmF0aW9uX3Byb21wdCUzRFRydWUpLnRvKG1vZGVsLmRldmljZSklMEElMEFvdXRwdXRzJTIwJTNEJTIwbW9kZWwuZ2VuZXJhdGUoKippbnB1dF9pZHMlMkMlMjBtYXhfbmV3X3Rva2VucyUzRDMyKSUwQXByaW50KHRva2VuaXplci5kZWNvZGUob3V0cHV0cyU1QjAlNUQpKQ==",highlighted:`<span class="hljs-comment"># pip install accelerate</span>
<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;google/t5gemma-2b-2b-prefixlm-it&quot;</span>)
model = AutoModelForSeq2SeqLM.from_pretrained(
    <span class="hljs-string">&quot;google/t5gemma-2b-2b-prefixlm-it&quot;</span>,
    device_map=<span class="hljs-string">&quot;auto&quot;</span>,
    dtype=torch.bfloat16,
)

messages = [
    {<span class="hljs-string">&quot;role&quot;</span>: <span class="hljs-string">&quot;user&quot;</span>, <span class="hljs-string">&quot;content&quot;</span>: <span class="hljs-string">&quot;Tell me an unknown interesting biology fact about the brain.&quot;</span>},
]
input_ids = tokenizer.apply_chat_template(messages, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>, return_dict=<span class="hljs-literal">True</span>, add_generation_prompt=<span class="hljs-literal">True</span>).to(model.device)

outputs = model.generate(**input_ids, max_new_tokens=<span class="hljs-number">32</span>)
<span class="hljs-built_in">print</span>(tokenizer.decode(outputs[<span class="hljs-number">0</span>]))`,wrap:!1}}),{c(){f(o.$$.fragment)},l(t){g(o.$$.fragment,t)},m(t,l){_(o,t,l),d=!0},p:x,i(t){d||(T(o.$$.fragment,t),d=!0)},o(t){b(o.$$.fragment,t),d=!1},d(t){y(o,t)}}}function Kt(v){let o,d;return o=new K({props:{code:"ZWNobyUyMC1lJTIwJTIyV3JpdGUlMjBtZSUyMGElMjBwb2VtJTIwYWJvdXQlMjBNYWNoaW5lJTIwTGVhcm5pbmcuJTIwQW5zd2VyJTNBJTIyJTIwJTdDJTIwdHJhbnNmb3JtZXJzJTIwcnVuJTIwLS10YXNrJTIwdGV4dDJ0ZXh0LWdlbmVyYXRpb24lMjAtLW1vZGVsJTIwZ29vZ2xlJTJGdDVnZW1tYS0yYi0yYi1wcmVmaXhsbSUyMC0tZGV2aWNlJTIwMA==",highlighted:'<span class="hljs-keyword">echo</span> -e <span class="hljs-string">&quot;Write me a poem about Machine Learning. Answer:&quot;</span> | transformers run <span class="hljs-params">--task</span> text2text-generation <span class="hljs-params">--model</span> google/t5gemma-2b-2b-prefixlm <span class="hljs-params">--device</span> 0',wrap:!1}}),{c(){f(o.$$.fragment)},l(t){g(o.$$.fragment,t)},m(t,l){_(o,t,l),d=!0},p:x,i(t){d||(T(o.$$.fragment,t),d=!0)},o(t){b(o.$$.fragment,t),d=!1},d(t){y(o,t)}}}function en(v){let o,d,t,l,M,a;return o=new ft({props:{id:"usage",option:"Pipeline",$$slots:{default:[Dt]},$$scope:{ctx:v}}}),t=new ft({props:{id:"usage",option:"AutoModel",$$slots:{default:[Ot]},$$scope:{ctx:v}}}),M=new ft({props:{id:"usage",option:"transformers CLI",$$slots:{default:[Kt]},$$scope:{ctx:v}}}),{c(){f(o.$$.fragment),d=r(),f(t.$$.fragment),l=r(),f(M.$$.fragment)},l(h){g(o.$$.fragment,h),d=i(h),g(t.$$.fragment,h),l=i(h),g(M.$$.fragment,h)},m(h,k){_(o,h,k),m(h,d,k),_(t,h,k),m(h,l,k),_(M,h,k),a=!0},p(h,k){const io={};k&2&&(io.$$scope={dirty:k,ctx:h}),o.$set(io);const he={};k&2&&(he.$$scope={dirty:k,ctx:h}),t.$set(he);const E={};k&2&&(E.$$scope={dirty:k,ctx:h}),M.$set(E)},i(h){a||(T(o.$$.fragment,h),T(t.$$.fragment,h),T(M.$$.fragment,h),a=!0)},o(h){b(o.$$.fragment,h),b(t.$$.fragment,h),b(M.$$.fragment,h),a=!1},d(h){h&&(s(d),s(l)),y(o,h),y(t,h),y(M,h)}}}function on(v){let o,d;return o=new K({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMFQ1R2VtbWFDb25maWclMkMlMjBUNUdlbW1hTW9kZWwlMEF0NWdlbW1hX2NvbmZpZyUyMCUzRCUyMFQ1R2VtbWFDb25maWcuZnJvbV9wcmV0cmFpbmVkKCUyMmdvb2dsZSUyRnQ1Z2VtbWEtMmItMmItcHJlZml4bG0taXQlMjIpJTBBbW9kZWwlMjAlM0QlMjBUNUdlbW1hTW9kZWwodDVnZW1tYV9jb25maWcp",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> T5GemmaConfig, T5GemmaModel
<span class="hljs-meta">&gt;&gt;&gt; </span>t5gemma_config = T5GemmaConfig.from_pretrained(<span class="hljs-string">&quot;google/t5gemma-2b-2b-prefixlm-it&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = T5GemmaModel(t5gemma_config)`,wrap:!1}}),{c(){f(o.$$.fragment)},l(t){g(o.$$.fragment,t)},m(t,l){_(o,t,l),d=!0},p:x,i(t){d||(T(o.$$.fragment,t),d=!0)},o(t){b(o.$$.fragment,t),d=!1},d(t){y(o,t)}}}function tn(v){let o,d;return o=new K({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMFQ1R2VtbWFNb2R1bGVNb2RlbCUyQyUyMFQ1R2VtbWFNb2R1bGVDb25maWclMEElMjMlMjBJbml0aWFsaXppbmclMjBhJTIwVDVHZW1tYU1vZHVsZSUyMHQ1X2dlbW1hX21vZHVsZS03YiUyMHN0eWxlJTIwY29uZmlndXJhdGlvbiUwQWNvbmZpZ3VyYXRpb24lMjAlM0QlMjBUNUdlbW1hTW9kdWxlQ29uZmlnKCklMEElMjMlMjBJbml0aWFsaXppbmclMjBhJTIwbW9kZWwlMjBmcm9tJTIwdGhlJTIwdDVfZ2VtbWFfbW9kdWxlLTdiJTIwc3R5bGUlMjBjb25maWd1cmF0aW9uJTBBbW9kZWwlMjAlM0QlMjBUNUdlbW1hTW9kdWxlTW9kZWwoY29uZmlndXJhdGlvbiklMEElMjMlMjBBY2Nlc3NpbmclMjB0aGUlMjBtb2RlbCUyMGNvbmZpZ3VyYXRpb24lMEFjb25maWd1cmF0aW9uJTIwJTNEJTIwbW9kZWwuY29uZmln",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> T5GemmaModuleModel, T5GemmaModuleConfig
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a T5GemmaModule t5_gemma_module-7b style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = T5GemmaModuleConfig()
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a model from the t5_gemma_module-7b style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = T5GemmaModuleModel(configuration)
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Accessing the model configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = model.config`,wrap:!1}}),{c(){f(o.$$.fragment)},l(t){g(o.$$.fragment,t)},m(t,l){_(o,t,l),d=!0},p:x,i(t){d||(T(o.$$.fragment,t),d=!0)},o(t){b(o.$$.fragment,t),d=!1},d(t){y(o,t)}}}function nn(v){let o,d=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){o=p("p"),o.innerHTML=d},l(t){o=u(t,"P",{"data-svelte-h":!0}),w(o)!=="svelte-fincs2"&&(o.innerHTML=d)},m(t,l){m(t,o,l)},p:x,d(t){t&&s(o)}}}function sn(v){let o,d=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){o=p("p"),o.innerHTML=d},l(t){o=u(t,"P",{"data-svelte-h":!0}),w(o)!=="svelte-fincs2"&&(o.innerHTML=d)},m(t,l){m(t,o,l)},p:x,d(t){t&&s(o)}}}function an(v){let o,d=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){o=p("p"),o.innerHTML=d},l(t){o=u(t,"P",{"data-svelte-h":!0}),w(o)!=="svelte-fincs2"&&(o.innerHTML=d)},m(t,l){m(t,o,l)},p:x,d(t){t&&s(o)}}}function rn(v){let o,d="Example:",t,l,M;return l=new K({props:{code:"",highlighted:"",wrap:!1}}),{c(){o=p("p"),o.textContent=d,t=r(),f(l.$$.fragment)},l(a){o=u(a,"P",{"data-svelte-h":!0}),w(o)!=="svelte-11lpom8"&&(o.textContent=d),t=i(a),g(l.$$.fragment,a)},m(a,h){m(a,o,h),m(a,t,h),_(l,a,h),M=!0},p:x,i(a){M||(T(l.$$.fragment,a),M=!0)},o(a){b(l.$$.fragment,a),M=!1},d(a){a&&(s(o),s(t)),y(l,a)}}}function dn(v){let o,d=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){o=p("p"),o.innerHTML=d},l(t){o=u(t,"P",{"data-svelte-h":!0}),w(o)!=="svelte-fincs2"&&(o.innerHTML=d)},m(t,l){m(t,o,l)},p:x,d(t){t&&s(o)}}}function ln(v){let o,d="Example of single-label classification:",t,l,M;return l=new K({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Rva2VuaXplciUyQyUyMFQ1R2VtbWFGb3JTZXF1ZW5jZUNsYXNzaWZpY2F0aW9uJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIyZ29vZ2xlJTJGdDVnZW1tYS0yYi0yYi1wcmVmaXhsbS1pdCUyMiklMEFtb2RlbCUyMCUzRCUyMFQ1R2VtbWFGb3JTZXF1ZW5jZUNsYXNzaWZpY2F0aW9uLmZyb21fcHJldHJhaW5lZCglMjJnb29nbGUlMkZ0NWdlbW1hLTJiLTJiLXByZWZpeGxtLWl0JTIyKSUwQSUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplciglMjJIZWxsbyUyQyUyMG15JTIwZG9nJTIwaXMlMjBjdXRlJTIyJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiklMEElMEF3aXRoJTIwdG9yY2gubm9fZ3JhZCgpJTNBJTBBJTIwJTIwJTIwJTIwbG9naXRzJTIwJTNEJTIwbW9kZWwoKippbnB1dHMpLmxvZ2l0cyUwQSUwQXByZWRpY3RlZF9jbGFzc19pZCUyMCUzRCUyMGxvZ2l0cy5hcmdtYXgoKS5pdGVtKCklMEFtb2RlbC5jb25maWcuaWQybGFiZWwlNUJwcmVkaWN0ZWRfY2xhc3NfaWQlNUQlMEElMEElMjMlMjBUbyUyMHRyYWluJTIwYSUyMG1vZGVsJTIwb24lMjAlNjBudW1fbGFiZWxzJTYwJTIwY2xhc3NlcyUyQyUyMHlvdSUyMGNhbiUyMHBhc3MlMjAlNjBudW1fbGFiZWxzJTNEbnVtX2xhYmVscyU2MCUyMHRvJTIwJTYwLmZyb21fcHJldHJhaW5lZCguLi4pJTYwJTBBbnVtX2xhYmVscyUyMCUzRCUyMGxlbihtb2RlbC5jb25maWcuaWQybGFiZWwpJTBBbW9kZWwlMjAlM0QlMjBUNUdlbW1hRm9yU2VxdWVuY2VDbGFzc2lmaWNhdGlvbi5mcm9tX3ByZXRyYWluZWQoJTIyZ29vZ2xlJTJGdDVnZW1tYS0yYi0yYi1wcmVmaXhsbS1pdCUyMiUyQyUyMG51bV9sYWJlbHMlM0RudW1fbGFiZWxzKSUwQSUwQWxhYmVscyUyMCUzRCUyMHRvcmNoLnRlbnNvciglNUIxJTVEKSUwQWxvc3MlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyUyQyUyMGxhYmVscyUzRGxhYmVscykubG9zcyUwQXJvdW5kKGxvc3MuaXRlbSgpJTJDJTIwMik=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, T5GemmaForSequenceClassification

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;google/t5gemma-2b-2b-prefixlm-it&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = T5GemmaForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;google/t5gemma-2b-2b-prefixlm-it&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    logits = model(**inputs).logits

<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_class_id = logits.argmax().item()
<span class="hljs-meta">&gt;&gt;&gt; </span>model.config.id2label[predicted_class_id]
...

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># To train a model on \`num_labels\` classes, you can pass \`num_labels=num_labels\` to \`.from_pretrained(...)\`</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>num_labels = <span class="hljs-built_in">len</span>(model.config.id2label)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = T5GemmaForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;google/t5gemma-2b-2b-prefixlm-it&quot;</span>, num_labels=num_labels)

<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.tensor([<span class="hljs-number">1</span>])
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = model(**inputs, labels=labels).loss
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">round</span>(loss.item(), <span class="hljs-number">2</span>)
...`,wrap:!1}}),{c(){o=p("p"),o.textContent=d,t=r(),f(l.$$.fragment)},l(a){o=u(a,"P",{"data-svelte-h":!0}),w(o)!=="svelte-ykxpe4"&&(o.textContent=d),t=i(a),g(l.$$.fragment,a)},m(a,h){m(a,o,h),m(a,t,h),_(l,a,h),M=!0},p:x,i(a){M||(T(l.$$.fragment,a),M=!0)},o(a){b(l.$$.fragment,a),M=!1},d(a){a&&(s(o),s(t)),y(l,a)}}}function cn(v){let o,d="Example of multi-label classification:",t,l,M;return l=new K({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Rva2VuaXplciUyQyUyMFQ1R2VtbWFGb3JTZXF1ZW5jZUNsYXNzaWZpY2F0aW9uJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIyZ29vZ2xlJTJGdDVnZW1tYS0yYi0yYi1wcmVmaXhsbS1pdCUyMiklMEFtb2RlbCUyMCUzRCUyMFQ1R2VtbWFGb3JTZXF1ZW5jZUNsYXNzaWZpY2F0aW9uLmZyb21fcHJldHJhaW5lZCglMjJnb29nbGUlMkZ0NWdlbW1hLTJiLTJiLXByZWZpeGxtLWl0JTIyJTJDJTIwcHJvYmxlbV90eXBlJTNEJTIybXVsdGlfbGFiZWxfY2xhc3NpZmljYXRpb24lMjIpJTBBJTBBaW5wdXRzJTIwJTNEJTIwdG9rZW5pemVyKCUyMkhlbGxvJTJDJTIwbXklMjBkb2clMjBpcyUyMGN1dGUlMjIlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQSUwQXdpdGglMjB0b3JjaC5ub19ncmFkKCklM0ElMEElMjAlMjAlMjAlMjBsb2dpdHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cykubG9naXRzJTBBJTBBcHJlZGljdGVkX2NsYXNzX2lkcyUyMCUzRCUyMHRvcmNoLmFyYW5nZSgwJTJDJTIwbG9naXRzLnNoYXBlJTVCLTElNUQpJTVCdG9yY2guc2lnbW9pZChsb2dpdHMpLnNxdWVlemUoZGltJTNEMCklMjAlM0UlMjAwLjUlNUQlMEElMEElMjMlMjBUbyUyMHRyYWluJTIwYSUyMG1vZGVsJTIwb24lMjAlNjBudW1fbGFiZWxzJTYwJTIwY2xhc3NlcyUyQyUyMHlvdSUyMGNhbiUyMHBhc3MlMjAlNjBudW1fbGFiZWxzJTNEbnVtX2xhYmVscyU2MCUyMHRvJTIwJTYwLmZyb21fcHJldHJhaW5lZCguLi4pJTYwJTBBbnVtX2xhYmVscyUyMCUzRCUyMGxlbihtb2RlbC5jb25maWcuaWQybGFiZWwpJTBBbW9kZWwlMjAlM0QlMjBUNUdlbW1hRm9yU2VxdWVuY2VDbGFzc2lmaWNhdGlvbi5mcm9tX3ByZXRyYWluZWQoJTBBJTIwJTIwJTIwJTIwJTIyZ29vZ2xlJTJGdDVnZW1tYS0yYi0yYi1wcmVmaXhsbS1pdCUyMiUyQyUyMG51bV9sYWJlbHMlM0RudW1fbGFiZWxzJTJDJTIwcHJvYmxlbV90eXBlJTNEJTIybXVsdGlfbGFiZWxfY2xhc3NpZmljYXRpb24lMjIlMEEpJTBBJTBBbGFiZWxzJTIwJTNEJTIwdG9yY2guc3VtKCUwQSUyMCUyMCUyMCUyMHRvcmNoLm5uLmZ1bmN0aW9uYWwub25lX2hvdChwcmVkaWN0ZWRfY2xhc3NfaWRzJTVCTm9uZSUyQyUyMCUzQSU1RC5jbG9uZSgpJTJDJTIwbnVtX2NsYXNzZXMlM0RudW1fbGFiZWxzKSUyQyUyMGRpbSUzRDElMEEpLnRvKHRvcmNoLmZsb2F0KSUwQWxvc3MlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyUyQyUyMGxhYmVscyUzRGxhYmVscykubG9zcw==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, T5GemmaForSequenceClassification

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;google/t5gemma-2b-2b-prefixlm-it&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = T5GemmaForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;google/t5gemma-2b-2b-prefixlm-it&quot;</span>, problem_type=<span class="hljs-string">&quot;multi_label_classification&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    logits = model(**inputs).logits

<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_class_ids = torch.arange(<span class="hljs-number">0</span>, logits.shape[-<span class="hljs-number">1</span>])[torch.sigmoid(logits).squeeze(dim=<span class="hljs-number">0</span>) &gt; <span class="hljs-number">0.5</span>]

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># To train a model on \`num_labels\` classes, you can pass \`num_labels=num_labels\` to \`.from_pretrained(...)\`</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>num_labels = <span class="hljs-built_in">len</span>(model.config.id2label)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = T5GemmaForSequenceClassification.from_pretrained(
<span class="hljs-meta">... </span>    <span class="hljs-string">&quot;google/t5gemma-2b-2b-prefixlm-it&quot;</span>, num_labels=num_labels, problem_type=<span class="hljs-string">&quot;multi_label_classification&quot;</span>
<span class="hljs-meta">... </span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.<span class="hljs-built_in">sum</span>(
<span class="hljs-meta">... </span>    torch.nn.functional.one_hot(predicted_class_ids[<span class="hljs-literal">None</span>, :].clone(), num_classes=num_labels), dim=<span class="hljs-number">1</span>
<span class="hljs-meta">... </span>).to(torch.<span class="hljs-built_in">float</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = model(**inputs, labels=labels).loss`,wrap:!1}}),{c(){o=p("p"),o.textContent=d,t=r(),f(l.$$.fragment)},l(a){o=u(a,"P",{"data-svelte-h":!0}),w(o)!=="svelte-1l8e32d"&&(o.textContent=d),t=i(a),g(l.$$.fragment,a)},m(a,h){m(a,o,h),m(a,t,h),_(l,a,h),M=!0},p:x,i(a){M||(T(l.$$.fragment,a),M=!0)},o(a){b(l.$$.fragment,a),M=!1},d(a){a&&(s(o),s(t)),y(l,a)}}}function mn(v){let o,d=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){o=p("p"),o.innerHTML=d},l(t){o=u(t,"P",{"data-svelte-h":!0}),w(o)!=="svelte-fincs2"&&(o.innerHTML=d)},m(t,l){m(t,o,l)},p:x,d(t){t&&s(o)}}}function pn(v){let o,d="Example:",t,l,M;return l=new K({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBUNUdlbW1hRm9yVG9rZW5DbGFzc2lmaWNhdGlvbiUwQWltcG9ydCUyMHRvcmNoJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIyZ29vZ2xlJTJGdDVnZW1tYS0yYi0yYi1wcmVmaXhsbS1pdCUyMiklMEFtb2RlbCUyMCUzRCUyMFQ1R2VtbWFGb3JUb2tlbkNsYXNzaWZpY2F0aW9uLmZyb21fcHJldHJhaW5lZCglMjJnb29nbGUlMkZ0NWdlbW1hLTJiLTJiLXByZWZpeGxtLWl0JTIyKSUwQSUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplciglMEElMjAlMjAlMjAlMjAlMjJIdWdnaW5nRmFjZSUyMGlzJTIwYSUyMGNvbXBhbnklMjBiYXNlZCUyMGluJTIwUGFyaXMlMjBhbmQlMjBOZXclMjBZb3JrJTIyJTJDJTIwYWRkX3NwZWNpYWxfdG9rZW5zJTNERmFsc2UlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyJTBBKSUwQSUwQXdpdGglMjB0b3JjaC5ub19ncmFkKCklM0ElMEElMjAlMjAlMjAlMjBsb2dpdHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cykubG9naXRzJTBBJTBBcHJlZGljdGVkX3Rva2VuX2NsYXNzX2lkcyUyMCUzRCUyMGxvZ2l0cy5hcmdtYXgoLTEpJTBBJTBBJTIzJTIwTm90ZSUyMHRoYXQlMjB0b2tlbnMlMjBhcmUlMjBjbGFzc2lmaWVkJTIwcmF0aGVyJTIwdGhlbiUyMGlucHV0JTIwd29yZHMlMjB3aGljaCUyMG1lYW5zJTIwdGhhdCUwQSUyMyUyMHRoZXJlJTIwbWlnaHQlMjBiZSUyMG1vcmUlMjBwcmVkaWN0ZWQlMjB0b2tlbiUyMGNsYXNzZXMlMjB0aGFuJTIwd29yZHMuJTBBJTIzJTIwTXVsdGlwbGUlMjB0b2tlbiUyMGNsYXNzZXMlMjBtaWdodCUyMGFjY291bnQlMjBmb3IlMjB0aGUlMjBzYW1lJTIwd29yZCUwQXByZWRpY3RlZF90b2tlbnNfY2xhc3NlcyUyMCUzRCUyMCU1Qm1vZGVsLmNvbmZpZy5pZDJsYWJlbCU1QnQuaXRlbSgpJTVEJTIwZm9yJTIwdCUyMGluJTIwcHJlZGljdGVkX3Rva2VuX2NsYXNzX2lkcyU1QjAlNUQlNUQlMEFwcmVkaWN0ZWRfdG9rZW5zX2NsYXNzZXMlMEElMEFsYWJlbHMlMjAlM0QlMjBwcmVkaWN0ZWRfdG9rZW5fY2xhc3NfaWRzJTBBbG9zcyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzJTJDJTIwbGFiZWxzJTNEbGFiZWxzKS5sb3NzJTBBcm91bmQobG9zcy5pdGVtKCklMkMlMjAyKQ==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, T5GemmaForTokenClassification
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;google/t5gemma-2b-2b-prefixlm-it&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = T5GemmaForTokenClassification.from_pretrained(<span class="hljs-string">&quot;google/t5gemma-2b-2b-prefixlm-it&quot;</span>)

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
...`,wrap:!1}}),{c(){o=p("p"),o.textContent=d,t=r(),f(l.$$.fragment)},l(a){o=u(a,"P",{"data-svelte-h":!0}),w(o)!=="svelte-11lpom8"&&(o.textContent=d),t=i(a),g(l.$$.fragment,a)},m(a,h){m(a,o,h),m(a,t,h),_(l,a,h),M=!0},p:x,i(a){M||(T(l.$$.fragment,a),M=!0)},o(a){b(l.$$.fragment,a),M=!1},d(a){a&&(s(o),s(t)),y(l,a)}}}function un(v){let o,d,t,l,M,a="<em>This model was released on 2025-04-08 and added to Hugging Face Transformers on 2025-06-25.</em>",h,k,io='<div class="flex flex-wrap space-x-1"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/> <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat"/> <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/></div>',he,E,co,fe,gt='T5Gemma (aka encoder-decoder Gemma) was proposed in a <a href="https://huggingface.co/papers/2504.06225" rel="nofollow">research paper</a> by Google. It is a family of encoder-decoder large language models, developed by adapting pretrained decoder-only models into encoder-decoder. T5Gemma includes pretrained and instruction-tuned variants. The architecture is based on transformer encoder-decoder design following T5, with improvements from Gemma 2: GQA, RoPE, GeGLU activation, RMSNorm, and interleaved local/global attention.',mo,ge,_t='T5Gemma has two groups of model sizes: 1) <a href="https://ai.google.dev/gemma/docs/core/model_card_2" rel="nofollow">Gemma 2</a> sizes (2B-2B, 9B-2B, and 9B-9B), which are based on the official Gemma 2 models (2B and 9B); and 2) <a href="https://huggingface.co/papers/1910.10683" rel="nofollow">T5</a> sizes (Small, Base, Large, and XL), where are pretrained under the Gemma 2 framework following T5 configuration. In addition, we also provide a model at ML size (medium large, ~2B in total), which is in-between T5 Large and T5 XL.',po,_e,Tt="The pretrained variants are trained with two objectives: prefix language modeling with knowledge distillation (PrefixLM) and UL2, separately. We release both variants for each model size. The instruction-turned variants was post-trained with supervised fine-tuning and reinforcement learning.",uo,ee,ho,Te,bt='The example below demonstrates how to chat with the model with <a href="/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.Pipeline">Pipeline</a> or the <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel">AutoModel</a> class, and from the command line.',fo,oe,go,be,_o,B,ye,Wo,Ve,yt=`This is the configuration class to store the configuration of a <a href="/docs/transformers/v4.56.2/en/model_doc/t5gemma#transformers.T5GemmaModel">T5GemmaModel</a>. It is used to instantiate an T5Gemma
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to a hypothetical balanced Gemma2 encoder-decoder model.
e.g. <a href="https://huggingface.co/google/t5gemma-2b-2b-prefixlm-it" rel="nofollow">google/t5gemma-2b-2b-prefixlm-it</a>`,Fo,te,Io,To,Me,bo,L,we,qo,Xe,Mt=`This is the configuration class to store the configuration of a <code>T5GemmaModuleModel</code>. It is used to instantiate an T5GemmaModule
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the T5GemmaModule-7B.
e.g. <a href="https://huggingface.co/google/t5_gemma_module-7b" rel="nofollow">google/t5_gemma_module-7b</a>
Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> and can be used to control the model outputs. Read the
documentation from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> for more information.`,Zo,ne,yo,ve,Mo,$,ke,Bo,Le,wt="The bare T5Gemma Model outputting raw hidden-states without any specific head on top.",No,Re,vt=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Vo,Se,kt=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Xo,H,$e,Lo,Ee,$t='The <a href="/docs/transformers/v4.56.2/en/model_doc/t5gemma#transformers.T5GemmaModel">T5GemmaModel</a> forward method, overrides the <code>__call__</code> special method.',Ro,se,wo,Ge,vo,G,Ce,So,He,Gt="The bare T5Gemma Model outputting raw hidden-states without any specific head on top.",Eo,Ye,Ct=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Ho,Qe,Jt=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Yo,Y,Je,Qo,Pe,xt='The <a href="/docs/transformers/v4.56.2/en/model_doc/t5gemma#transformers.T5GemmaEncoderModel">T5GemmaEncoderModel</a> forward method, overrides the <code>__call__</code> special method.',Po,ae,ko,xe,$o,P,je,Ao,N,ze,Do,Ae,jt='The <a href="/docs/transformers/v4.56.2/en/model_doc/t5gemma#transformers.T5GemmaForConditionalGeneration">T5GemmaForConditionalGeneration</a> forward method, overrides the <code>__call__</code> special method.',Oo,re,Ko,ie,Go,Ue,Co,C,We,et,De,zt="The T5Gemma Model with a sequence classification/regression head on top e.g. for GLUE tasks.",ot,Oe,Ut=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,tt,Ke,Wt=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,nt,j,Fe,st,eo,Ft='The <a href="/docs/transformers/v4.56.2/en/model_doc/t5gemma#transformers.T5GemmaForSequenceClassification">T5GemmaForSequenceClassification</a> forward method, overrides the <code>__call__</code> special method.',at,de,rt,le,it,ce,Jo,Ie,xo,J,qe,dt,oo,It=`The T5Gemma transformer with a token classification head on top (a linear layer on top of the hidden-states
output) e.g. for Named-Entity-Recognition (NER) tasks.`,lt,to,qt=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,ct,no,Zt=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,mt,V,Ze,pt,so,Bt='The <a href="/docs/transformers/v4.56.2/en/model_doc/t5gemma#transformers.T5GemmaForTokenClassification">T5GemmaForTokenClassification</a> forward method, overrides the <code>__call__</code> special method.',ut,me,ht,pe,jo,Be,zo,lo,Uo;return E=new ue({props:{title:"T5Gemma",local:"t5gemma",headingTag:"h1"}}),ee=new ao({props:{warning:!1,$$slots:{default:[At]},$$scope:{ctx:v}}}),oe=new Pt({props:{id:"usage",options:["Pipeline","AutoModel","transformers CLI"],$$slots:{default:[en]},$$scope:{ctx:v}}}),be=new ue({props:{title:"T5GemmaConfig",local:"transformers.T5GemmaConfig",headingTag:"h2"}}),ye=new X({props:{name:"class transformers.T5GemmaConfig",anchor:"transformers.T5GemmaConfig",parameters:[{name:"encoder",val:": typing.Union[transformers.models.t5gemma.configuration_t5gemma.T5GemmaModuleConfig, dict[typing.Any, typing.Any], NoneType] = None"},{name:"decoder",val:": typing.Union[transformers.models.t5gemma.configuration_t5gemma.T5GemmaModuleConfig, dict[typing.Any, typing.Any], NoneType] = None"},{name:"is_encoder_decoder",val:": bool = True"},{name:"dropout_rate",val:": float = 0.0"},{name:"classifier_dropout_rate",val:": float = 0.0"},{name:"attention_dropout",val:": float = 0.0"},{name:"tie_word_embeddings",val:": bool = True"},{name:"vocab_size",val:": int = 256000"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.T5GemmaConfig.encoder",description:`<strong>encoder</strong> (<code>Union[T5GemmaModuleConfig, dict]</code>, optional, <em>optional</em>) &#x2014;
Configuration for the encoder.`,name:"encoder"},{anchor:"transformers.T5GemmaConfig.decoder",description:`<strong>decoder</strong> (<code>Union[T5GemmaModuleConfig, dict]</code>, optional, <em>optional</em>) &#x2014;
Configuration for the decoder.`,name:"decoder"},{anchor:"transformers.T5GemmaConfig.is_encoder_decoder",description:`<strong>is_encoder_decoder</strong> (bool, optional, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether the model is used as an encoder/decoder or not.`,name:"is_encoder_decoder"},{anchor:"transformers.T5GemmaConfig.dropout_rate",description:`<strong>dropout_rate</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The ratio for all dropout layers (following T5).`,name:"dropout_rate"},{anchor:"transformers.T5GemmaConfig.classifier_dropout_rate",description:`<strong>classifier_dropout_rate</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The dropout ratio for classifier (following T5).`,name:"classifier_dropout_rate"},{anchor:"transformers.T5GemmaConfig.attention_dropout",description:`<strong>attention_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The dropout ratio for attention.`,name:"attention_dropout"},{anchor:"transformers.T5GemmaConfig.tie_word_embeddings",description:`<strong>tie_word_embeddings</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether tie input and output embeddings.`,name:"tie_word_embeddings"},{anchor:"transformers.T5GemmaConfig.vocab_size",description:`<strong>vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 256000) &#x2014;
Vocabulary size of the T5Gemma model (the same as Gemma 2).`,name:"vocab_size"},{anchor:"transformers.T5GemmaConfig.kwargs",description:`<strong>kwargs</strong> (additional keyword arguments, optional, <em>optional</em>) &#x2014;
Will be passed to the PretrainedConfig base class.`,name:"kwargs"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/t5gemma/configuration_t5gemma.py#L184"}}),te=new ro({props:{anchor:"transformers.T5GemmaConfig.example",$$slots:{default:[on]},$$scope:{ctx:v}}}),Me=new ue({props:{title:"T5GemmaModuleConfig",local:"transformers.T5GemmaModuleConfig",headingTag:"h2"}}),we=new X({props:{name:"class transformers.T5GemmaModuleConfig",anchor:"transformers.T5GemmaModuleConfig",parameters:[{name:"vocab_size",val:" = 256000"},{name:"hidden_size",val:" = 2304"},{name:"intermediate_size",val:" = 9216"},{name:"num_hidden_layers",val:" = 26"},{name:"num_attention_heads",val:" = 8"},{name:"num_key_value_heads",val:" = 4"},{name:"head_dim",val:" = 256"},{name:"hidden_activation",val:" = 'gelu_pytorch_tanh'"},{name:"max_position_embeddings",val:" = 8192"},{name:"initializer_range",val:" = 0.02"},{name:"rms_norm_eps",val:" = 1e-06"},{name:"use_cache",val:" = True"},{name:"pad_token_id",val:" = 0"},{name:"eos_token_id",val:" = 1"},{name:"bos_token_id",val:" = 2"},{name:"tie_word_embeddings",val:" = True"},{name:"rope_theta",val:" = 10000.0"},{name:"attention_bias",val:" = False"},{name:"attention_dropout",val:" = 0.0"},{name:"query_pre_attn_scalar",val:" = 256"},{name:"sliding_window",val:" = 4096"},{name:"layer_types",val:" = None"},{name:"final_logit_softcapping",val:" = 30.0"},{name:"attn_logit_softcapping",val:" = 50.0"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.T5GemmaModuleConfig.vocab_size",description:`<strong>vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 256000) &#x2014;
Vocabulary size of the T5GemmaModule model. Defines the number of different tokens that can be represented by the
<code>inputs_ids</code> passed when calling <code>T5GemmaModuleModel</code>`,name:"vocab_size"},{anchor:"transformers.T5GemmaModuleConfig.hidden_size",description:`<strong>hidden_size</strong> (<code>int</code>, <em>optional</em>, defaults to 2304) &#x2014;
Dimension of the hidden representations.`,name:"hidden_size"},{anchor:"transformers.T5GemmaModuleConfig.intermediate_size",description:`<strong>intermediate_size</strong> (<code>int</code>, <em>optional</em>, defaults to 9216) &#x2014;
Dimension of the MLP representations.`,name:"intermediate_size"},{anchor:"transformers.T5GemmaModuleConfig.num_hidden_layers",description:`<strong>num_hidden_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 26) &#x2014;
Number of hidden layers in the Transformer decoder.`,name:"num_hidden_layers"},{anchor:"transformers.T5GemmaModuleConfig.num_attention_heads",description:`<strong>num_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 8) &#x2014;
Number of attention heads for each attention layer in the Transformer decoder.`,name:"num_attention_heads"},{anchor:"transformers.T5GemmaModuleConfig.num_key_value_heads",description:`<strong>num_key_value_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 4) &#x2014;
This is the number of key_value heads that should be used to implement Grouped Query Attention. If
<code>num_key_value_heads=num_attention_heads</code>, the model will use Multi Head Attention (MHA), if
<code>num_key_value_heads=1</code> the model will use Multi Query Attention (MQA) otherwise GQA is used. When
converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
by meanpooling all the original heads within that group. For more details, check out <a href="https://huggingface.co/papers/2305.13245" rel="nofollow">this
paper</a>. If it is not specified, will default to
<code>num_attention_heads</code>.`,name:"num_key_value_heads"},{anchor:"transformers.T5GemmaModuleConfig.head_dim",description:`<strong>head_dim</strong> (<code>int</code>, <em>optional</em>, defaults to 256) &#x2014;
The attention head dimension.`,name:"head_dim"},{anchor:"transformers.T5GemmaModuleConfig.hidden_activation",description:`<strong>hidden_activation</strong> (<code>str</code> or <code>function</code>, <em>optional</em>, defaults to <code>&quot;gelu_pytorch_tanh&quot;</code>) &#x2014;
The non-linear activation function (function or string) in the decoder. Will default to <code>&quot;gelu_pytorch_tanh&quot;</code>
if not specified. <code>&quot;gelu_pytorch_tanh&quot;</code> uses an approximation of the <code>&quot;gelu&quot;</code> activation function.`,name:"hidden_activation"},{anchor:"transformers.T5GemmaModuleConfig.max_position_embeddings",description:`<strong>max_position_embeddings</strong> (<code>int</code>, <em>optional</em>, defaults to 8192) &#x2014;
The maximum sequence length that this model might ever be used with.`,name:"max_position_embeddings"},{anchor:"transformers.T5GemmaModuleConfig.initializer_range",description:`<strong>initializer_range</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.`,name:"initializer_range"},{anchor:"transformers.T5GemmaModuleConfig.rms_norm_eps",description:`<strong>rms_norm_eps</strong> (<code>float</code>, <em>optional</em>, defaults to 1e-06) &#x2014;
The epsilon used by the rms normalization layers.`,name:"rms_norm_eps"},{anchor:"transformers.T5GemmaModuleConfig.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not the model should return the last key/values attentions (not used by all models). Only
relevant if <code>config.is_decoder=True</code>.`,name:"use_cache"},{anchor:"transformers.T5GemmaModuleConfig.pad_token_id",description:`<strong>pad_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 0) &#x2014;
Padding token id.`,name:"pad_token_id"},{anchor:"transformers.T5GemmaModuleConfig.eos_token_id",description:`<strong>eos_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 1) &#x2014;
End of stream token id.`,name:"eos_token_id"},{anchor:"transformers.T5GemmaModuleConfig.bos_token_id",description:`<strong>bos_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 2) &#x2014;
Beginning of stream token id.`,name:"bos_token_id"},{anchor:"transformers.T5GemmaModuleConfig.tie_word_embeddings",description:`<strong>tie_word_embeddings</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether to tie weight embeddings`,name:"tie_word_embeddings"},{anchor:"transformers.T5GemmaModuleConfig.rope_theta",description:`<strong>rope_theta</strong> (<code>float</code>, <em>optional</em>, defaults to 10000.0) &#x2014;
The base period of the RoPE embeddings.`,name:"rope_theta"},{anchor:"transformers.T5GemmaModuleConfig.attention_bias",description:`<strong>attention_bias</strong> (<code>bool</code>, defaults to <code>False</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether to use a bias in the query, key, value and output projection layers during self-attention.`,name:"attention_bias"},{anchor:"transformers.T5GemmaModuleConfig.attention_dropout",description:`<strong>attention_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The dropout ratio for the attention probabilities.`,name:"attention_dropout"},{anchor:"transformers.T5GemmaModuleConfig.query_pre_attn_scalar",description:`<strong>query_pre_attn_scalar</strong> (<code>float</code>, <em>optional</em>, defaults to 256) &#x2014;
scaling factor used on the attention scores`,name:"query_pre_attn_scalar"},{anchor:"transformers.T5GemmaModuleConfig.sliding_window",description:`<strong>sliding_window</strong> (<code>int</code>, <em>optional</em>, defaults to 4096) &#x2014;
in T5GemmaModule, every other layer uses sliding window attention. This is the size of the sliding window.`,name:"sliding_window"},{anchor:"transformers.T5GemmaModuleConfig.layer_types",description:`<strong>layer_types</strong> (<code>list</code>, <em>optional</em>) &#x2014;
Attention pattern for each layer.`,name:"layer_types"},{anchor:"transformers.T5GemmaModuleConfig.final_logit_softcapping",description:`<strong>final_logit_softcapping</strong> (<code>float</code>, <em>optional</em>, defaults to 30.0) &#x2014;
scaling factor when applying tanh softcapping on the logits.`,name:"final_logit_softcapping"},{anchor:"transformers.T5GemmaModuleConfig.attn_logit_softcapping",description:`<strong>attn_logit_softcapping</strong> (<code>float</code>, <em>optional</em>, defaults to 50.0) &#x2014;
scaling factor when applying tanh softcapping on the attention scores.`,name:"attn_logit_softcapping"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/t5gemma/configuration_t5gemma.py#L27"}}),ne=new ro({props:{anchor:"transformers.T5GemmaModuleConfig.example",$$slots:{default:[tn]},$$scope:{ctx:v}}}),ve=new ue({props:{title:"T5GemmaModel",local:"transformers.T5GemmaModel",headingTag:"h2"}}),ke=new X({props:{name:"class transformers.T5GemmaModel",anchor:"transformers.T5GemmaModel",parameters:[{name:"config",val:": T5GemmaConfig"}],parametersDescription:[{anchor:"transformers.T5GemmaModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/t5gemma#transformers.T5GemmaConfig">T5GemmaConfig</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/t5gemma/modeling_t5gemma.py#L886"}}),$e=new X({props:{name:"forward",anchor:"transformers.T5GemmaModel.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"decoder_input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"decoder_attention_mask",val:": typing.Optional[torch.BoolTensor] = None"},{name:"decoder_position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"encoder_outputs",val:": typing.Optional[transformers.modeling_outputs.BaseModelOutput] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.EncoderDecoderCache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"decoder_inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.T5GemmaModel.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.T5GemmaModel.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.T5GemmaModel.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.T5GemmaModel.forward.decoder_input_ids",description:`<strong>decoder_input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of decoder input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#decoder-input-ids">What are decoder input IDs?</a>`,name:"decoder_input_ids"},{anchor:"transformers.T5GemmaModel.forward.decoder_attention_mask",description:`<strong>decoder_attention_mask</strong> (<code>torch.BoolTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on certain token indices. By default, a causal mask will be used, to
make sure the model can only look at previous inputs in order to predict the future.`,name:"decoder_attention_mask"},{anchor:"transformers.T5GemmaModel.forward.decoder_position_ids",description:`<strong>decoder_position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, decoder_sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each decoder input sequence tokens in the position embeddings. Selected in the range <code>[0, config.decoder.n_positions - 1]</code>. <a href="../glossary#position-ids">What are position IDs?</a>`,name:"decoder_position_ids"},{anchor:"transformers.T5GemmaModel.forward.encoder_outputs",description:`<strong>encoder_outputs</strong> (<code>~modeling_outputs.BaseModelOutput</code>, <em>optional</em>) &#x2014;
Tuple consists of (<code>last_hidden_state</code>, <em>optional</em>: <code>hidden_states</code>, <em>optional</em>: <code>attentions</code>)
<code>last_hidden_state</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) is a sequence of
hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.`,name:"encoder_outputs"},{anchor:"transformers.T5GemmaModel.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.EncoderDecoderCache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.T5GemmaModel.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.T5GemmaModel.forward.decoder_inputs_embeds",description:`<strong>decoder_inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, target_sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>decoder_input_ids</code> you can choose to directly pass an embedded
representation. If <code>past_key_values</code> is used, optionally only the last <code>decoder_inputs_embeds</code> have to be
input (see <code>past_key_values</code>). This is useful if you want more control over how to convert
<code>decoder_input_ids</code> indices into associated vectors than the model&#x2019;s internal embedding lookup matrix.</p>
<p>If <code>decoder_input_ids</code> and <code>decoder_inputs_embeds</code> are both unset, <code>decoder_inputs_embeds</code> takes the value
of <code>inputs_embeds</code>.`,name:"decoder_inputs_embeds"},{anchor:"transformers.T5GemmaModel.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.T5GemmaModel.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/t5gemma/modeling_t5gemma.py#L907",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqModelOutput"
>transformers.modeling_outputs.Seq2SeqModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/t5gemma#transformers.T5GemmaConfig"
>T5GemmaConfig</a>) and inputs.</p>
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
`}}),se=new ao({props:{$$slots:{default:[nn]},$$scope:{ctx:v}}}),Ge=new ue({props:{title:"T5GemmaEncoderModel",local:"transformers.T5GemmaEncoderModel",headingTag:"h2"}}),Ce=new X({props:{name:"class transformers.T5GemmaEncoderModel",anchor:"transformers.T5GemmaEncoderModel",parameters:[{name:"config",val:": T5GemmaConfig"}],parametersDescription:[{anchor:"transformers.T5GemmaEncoderModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/t5gemma#transformers.T5GemmaConfig">T5GemmaConfig</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/t5gemma/modeling_t5gemma.py#L969"}}),Je=new X({props:{name:"forward",anchor:"transformers.T5GemmaEncoderModel.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.T5GemmaEncoderModel.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.T5GemmaEncoderModel.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.T5GemmaEncoderModel.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.T5GemmaEncoderModel.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/t5gemma/modeling_t5gemma.py#L985",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutput"
>transformers.modeling_outputs.BaseModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/t5gemma#transformers.T5GemmaConfig"
>T5GemmaConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>last_hidden_state</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>) — Sequence of hidden-states at the output of the last layer of the model.</p>
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
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutput"
>transformers.modeling_outputs.BaseModelOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),ae=new ao({props:{$$slots:{default:[sn]},$$scope:{ctx:v}}}),xe=new ue({props:{title:"T5GemmaForConditionalGeneration",local:"transformers.T5GemmaForConditionalGeneration",headingTag:"h2"}}),je=new X({props:{name:"class transformers.T5GemmaForConditionalGeneration",anchor:"transformers.T5GemmaForConditionalGeneration",parameters:[{name:"config",val:": T5GemmaConfig"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/t5gemma/modeling_t5gemma.py#L1005"}}),ze=new X({props:{name:"forward",anchor:"transformers.T5GemmaForConditionalGeneration.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"decoder_input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"decoder_attention_mask",val:": typing.Optional[torch.BoolTensor] = None"},{name:"decoder_position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"encoder_outputs",val:": typing.Optional[transformers.modeling_outputs.BaseModelOutput] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.EncoderDecoderCache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"decoder_inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"},{name:"logits_to_keep",val:": typing.Union[int, torch.Tensor] = 0"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.T5GemmaForConditionalGeneration.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.T5GemmaForConditionalGeneration.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.T5GemmaForConditionalGeneration.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.T5GemmaForConditionalGeneration.forward.decoder_input_ids",description:`<strong>decoder_input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of decoder input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#decoder-input-ids">What are decoder input IDs?</a>`,name:"decoder_input_ids"},{anchor:"transformers.T5GemmaForConditionalGeneration.forward.decoder_attention_mask",description:`<strong>decoder_attention_mask</strong> (<code>torch.BoolTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on certain token indices. By default, a causal mask will be used, to
make sure the model can only look at previous inputs in order to predict the future.`,name:"decoder_attention_mask"},{anchor:"transformers.T5GemmaForConditionalGeneration.forward.decoder_position_ids",description:`<strong>decoder_position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, decoder_sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each decoder input sequence tokens in the position embeddings. Selected in the range <code>[0, config.decoder.n_positions - 1]</code>. <a href="../glossary#position-ids">What are position IDs?</a>`,name:"decoder_position_ids"},{anchor:"transformers.T5GemmaForConditionalGeneration.forward.encoder_outputs",description:`<strong>encoder_outputs</strong> (<code>~modeling_outputs.BaseModelOutput</code>, <em>optional</em>) &#x2014;
Tuple consists of (<code>last_hidden_state</code>, <em>optional</em>: <code>hidden_states</code>, <em>optional</em>: <code>attentions</code>)
<code>last_hidden_state</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) is a sequence of
hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.`,name:"encoder_outputs"},{anchor:"transformers.T5GemmaForConditionalGeneration.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.EncoderDecoderCache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.T5GemmaForConditionalGeneration.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.T5GemmaForConditionalGeneration.forward.decoder_inputs_embeds",description:`<strong>decoder_inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, target_sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>decoder_input_ids</code> you can choose to directly pass an embedded
representation. If <code>past_key_values</code> is used, optionally only the last <code>decoder_inputs_embeds</code> have to be
input (see <code>past_key_values</code>). This is useful if you want more control over how to convert
<code>decoder_input_ids</code> indices into associated vectors than the model&#x2019;s internal embedding lookup matrix.</p>
<p>If <code>decoder_input_ids</code> and <code>decoder_inputs_embeds</code> are both unset, <code>decoder_inputs_embeds</code> takes the value
of <code>inputs_embeds</code>.`,name:"decoder_inputs_embeds"},{anchor:"transformers.T5GemmaForConditionalGeneration.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.T5GemmaForConditionalGeneration.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.T5GemmaForConditionalGeneration.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"},{anchor:"transformers.T5GemmaForConditionalGeneration.forward.logits_to_keep",description:`<strong>logits_to_keep</strong> (<code>Union[int, torch.Tensor]</code>, defaults to <code>0</code>) &#x2014;
If an <code>int</code>, compute logits for the last <code>logits_to_keep</code> tokens. If <code>0</code>, calculate logits for all
<code>input_ids</code> (special case). Only last token logits are needed for generation, and calculating them only for that
token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
If a <code>torch.Tensor</code>, must be 1D corresponding to the indices to keep in the sequence length dimension.
This is useful when using packed tensor format (single dimension for batch and sequence length).`,name:"logits_to_keep"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/t5gemma/modeling_t5gemma.py#L1038",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqLMOutput"
>transformers.modeling_outputs.Seq2SeqLMOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/t5gemma#transformers.T5GemmaConfig"
>T5GemmaConfig</a>) and inputs.</p>
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
`}}),re=new ao({props:{$$slots:{default:[an]},$$scope:{ctx:v}}}),ie=new ro({props:{anchor:"transformers.T5GemmaForConditionalGeneration.forward.example",$$slots:{default:[rn]},$$scope:{ctx:v}}}),Ue=new ue({props:{title:"T5GemmaForSequenceClassification",local:"transformers.T5GemmaForSequenceClassification",headingTag:"h2"}}),We=new X({props:{name:"class transformers.T5GemmaForSequenceClassification",anchor:"transformers.T5GemmaForSequenceClassification",parameters:[{name:"config",val:": T5GemmaConfig"},{name:"is_encoder_decoder",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.T5GemmaForSequenceClassification.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/t5gemma#transformers.T5GemmaConfig">T5GemmaConfig</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"},{anchor:"transformers.T5GemmaForSequenceClassification.is_encoder_decoder",description:`<strong>is_encoder_decoder</strong> (<code>Optional</code>, <em>optional</em>) &#x2014;
Whether use encoder_decoder for sequence classification. When set to False, only encoder is used.`,name:"is_encoder_decoder"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/t5gemma/modeling_t5gemma.py#L1129"}}),Fe=new X({props:{name:"forward",anchor:"transformers.T5GemmaForSequenceClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"decoder_input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"decoder_attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"decoder_position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"encoder_outputs",val:": typing.Optional[transformers.modeling_outputs.BaseModelOutput] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"decoder_inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.T5GemmaForSequenceClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.T5GemmaForSequenceClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.T5GemmaForSequenceClassification.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.T5GemmaForSequenceClassification.forward.decoder_input_ids",description:`<strong>decoder_input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of decoder input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#decoder-input-ids">What are decoder input IDs?</a>`,name:"decoder_input_ids"},{anchor:"transformers.T5GemmaForSequenceClassification.forward.decoder_attention_mask",description:`<strong>decoder_attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on certain token indices. By default, a causal mask will be used, to
make sure the model can only look at previous inputs in order to predict the future.`,name:"decoder_attention_mask"},{anchor:"transformers.T5GemmaForSequenceClassification.forward.decoder_position_ids",description:`<strong>decoder_position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, decoder_sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each decoder input sequence tokens in the position embeddings. Selected in the range <code>[0, config.decoder.n_positions - 1]</code>. <a href="../glossary#position-ids">What are position IDs?</a>`,name:"decoder_position_ids"},{anchor:"transformers.T5GemmaForSequenceClassification.forward.encoder_outputs",description:`<strong>encoder_outputs</strong> (<code>~modeling_outputs.BaseModelOutput</code>, <em>optional</em>) &#x2014;
Tuple consists of (<code>last_hidden_state</code>, <em>optional</em>: <code>hidden_states</code>, <em>optional</em>: <code>attentions</code>)
<code>last_hidden_state</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) is a sequence of
hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.`,name:"encoder_outputs"},{anchor:"transformers.T5GemmaForSequenceClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.T5GemmaForSequenceClassification.forward.decoder_inputs_embeds",description:`<strong>decoder_inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, target_sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>decoder_input_ids</code> you can choose to directly pass an embedded
representation. If <code>past_key_values</code> is used, optionally only the last <code>decoder_inputs_embeds</code> have to be
input (see <code>past_key_values</code>). This is useful if you want more control over how to convert
<code>decoder_input_ids</code> indices into associated vectors than the model&#x2019;s internal embedding lookup matrix.</p>
<p>If <code>decoder_input_ids</code> and <code>decoder_inputs_embeds</code> are both unset, <code>decoder_inputs_embeds</code> takes the value
of <code>inputs_embeds</code>.`,name:"decoder_inputs_embeds"},{anchor:"transformers.T5GemmaForSequenceClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the sequence classification/regression loss. Indices should be in <code>[0, ..., config.num_labels - 1]</code>. If <code>config.num_labels == 1</code> a regression loss is computed (Mean-Square loss), If
<code>config.num_labels &gt; 1</code> a classification loss is computed (Cross-Entropy).`,name:"labels"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/t5gemma/modeling_t5gemma.py#L1159",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput"
>transformers.modeling_outputs.SequenceClassifierOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/t5gemma#transformers.T5GemmaConfig"
>T5GemmaConfig</a>) and inputs.</p>
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
`}}),de=new ao({props:{$$slots:{default:[dn]},$$scope:{ctx:v}}}),le=new ro({props:{anchor:"transformers.T5GemmaForSequenceClassification.forward.example",$$slots:{default:[ln]},$$scope:{ctx:v}}}),ce=new ro({props:{anchor:"transformers.T5GemmaForSequenceClassification.forward.example-2",$$slots:{default:[cn]},$$scope:{ctx:v}}}),Ie=new ue({props:{title:"T5GemmaForTokenClassification",local:"transformers.T5GemmaForTokenClassification",headingTag:"h2"}}),qe=new X({props:{name:"class transformers.T5GemmaForTokenClassification",anchor:"transformers.T5GemmaForTokenClassification",parameters:[{name:"config",val:": T5GemmaConfig"},{name:"is_encoder_decoder",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.T5GemmaForTokenClassification.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/t5gemma#transformers.T5GemmaConfig">T5GemmaConfig</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"},{anchor:"transformers.T5GemmaForTokenClassification.is_encoder_decoder",description:`<strong>is_encoder_decoder</strong> (<code>Optional</code>, <em>optional</em>) &#x2014;
Whether use encoder_decoder for token classification. When set to False, only encoder is used.`,name:"is_encoder_decoder"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/t5gemma/modeling_t5gemma.py#L1270"}}),Ze=new X({props:{name:"forward",anchor:"transformers.T5GemmaForTokenClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"decoder_input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"decoder_attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"decoder_position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"encoder_outputs",val:": typing.Optional[transformers.modeling_outputs.BaseModelOutput] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"decoder_inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.T5GemmaForTokenClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.T5GemmaForTokenClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.T5GemmaForTokenClassification.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.T5GemmaForTokenClassification.forward.decoder_input_ids",description:`<strong>decoder_input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of decoder input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#decoder-input-ids">What are decoder input IDs?</a>`,name:"decoder_input_ids"},{anchor:"transformers.T5GemmaForTokenClassification.forward.decoder_attention_mask",description:`<strong>decoder_attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on certain token indices. By default, a causal mask will be used, to
make sure the model can only look at previous inputs in order to predict the future.`,name:"decoder_attention_mask"},{anchor:"transformers.T5GemmaForTokenClassification.forward.decoder_position_ids",description:`<strong>decoder_position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, decoder_sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each decoder input sequence tokens in the position embeddings. Selected in the range <code>[0, config.decoder.n_positions - 1]</code>. <a href="../glossary#position-ids">What are position IDs?</a>`,name:"decoder_position_ids"},{anchor:"transformers.T5GemmaForTokenClassification.forward.encoder_outputs",description:`<strong>encoder_outputs</strong> (<code>~modeling_outputs.BaseModelOutput</code>, <em>optional</em>) &#x2014;
Tuple consists of (<code>last_hidden_state</code>, <em>optional</em>: <code>hidden_states</code>, <em>optional</em>: <code>attentions</code>)
<code>last_hidden_state</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) is a sequence of
hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.`,name:"encoder_outputs"},{anchor:"transformers.T5GemmaForTokenClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.T5GemmaForTokenClassification.forward.decoder_inputs_embeds",description:`<strong>decoder_inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, target_sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>decoder_input_ids</code> you can choose to directly pass an embedded
representation. If <code>past_key_values</code> is used, optionally only the last <code>decoder_inputs_embeds</code> have to be
input (see <code>past_key_values</code>). This is useful if you want more control over how to convert
<code>decoder_input_ids</code> indices into associated vectors than the model&#x2019;s internal embedding lookup matrix.</p>
<p>If <code>decoder_input_ids</code> and <code>decoder_inputs_embeds</code> are both unset, <code>decoder_inputs_embeds</code> takes the value
of <code>inputs_embeds</code>.`,name:"decoder_inputs_embeds"},{anchor:"transformers.T5GemmaForTokenClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the sequence classification/regression loss. Indices should be in <code>[0, ..., config.num_labels - 1]</code>. If <code>config.num_labels == 1</code> a regression loss is computed (Mean-Square loss), If
<code>config.num_labels &gt; 1</code> a classification loss is computed (Cross-Entropy).`,name:"labels"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/t5gemma/modeling_t5gemma.py#L1301",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput"
>transformers.modeling_outputs.TokenClassifierOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/t5gemma#transformers.T5GemmaConfig"
>T5GemmaConfig</a>) and inputs.</p>
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
`}}),me=new ao({props:{$$slots:{default:[mn]},$$scope:{ctx:v}}}),pe=new ro({props:{anchor:"transformers.T5GemmaForTokenClassification.forward.example",$$slots:{default:[pn]},$$scope:{ctx:v}}}),Be=new Qt({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/t5gemma.md"}}),{c(){o=p("meta"),d=r(),t=p("p"),l=r(),M=p("p"),M.innerHTML=a,h=r(),k=p("div"),k.innerHTML=io,he=r(),f(E.$$.fragment),co=r(),fe=p("p"),fe.innerHTML=gt,mo=r(),ge=p("p"),ge.innerHTML=_t,po=r(),_e=p("p"),_e.textContent=Tt,uo=r(),f(ee.$$.fragment),ho=r(),Te=p("p"),Te.innerHTML=bt,fo=r(),f(oe.$$.fragment),go=r(),f(be.$$.fragment),_o=r(),B=p("div"),f(ye.$$.fragment),Wo=r(),Ve=p("p"),Ve.innerHTML=yt,Fo=r(),f(te.$$.fragment),Io=St(`
Configuration objects inherit from [PretrainedConfig] and can be used to control the model outputs. Read the
documentation from [PretrainedConfig] for more information.`),To=r(),f(Me.$$.fragment),bo=r(),L=p("div"),f(we.$$.fragment),qo=r(),Xe=p("p"),Xe.innerHTML=Mt,Zo=r(),f(ne.$$.fragment),yo=r(),f(ve.$$.fragment),Mo=r(),$=p("div"),f(ke.$$.fragment),Bo=r(),Le=p("p"),Le.textContent=wt,No=r(),Re=p("p"),Re.innerHTML=vt,Vo=r(),Se=p("p"),Se.innerHTML=kt,Xo=r(),H=p("div"),f($e.$$.fragment),Lo=r(),Ee=p("p"),Ee.innerHTML=$t,Ro=r(),f(se.$$.fragment),wo=r(),f(Ge.$$.fragment),vo=r(),G=p("div"),f(Ce.$$.fragment),So=r(),He=p("p"),He.textContent=Gt,Eo=r(),Ye=p("p"),Ye.innerHTML=Ct,Ho=r(),Qe=p("p"),Qe.innerHTML=Jt,Yo=r(),Y=p("div"),f(Je.$$.fragment),Qo=r(),Pe=p("p"),Pe.innerHTML=xt,Po=r(),f(ae.$$.fragment),ko=r(),f(xe.$$.fragment),$o=r(),P=p("div"),f(je.$$.fragment),Ao=r(),N=p("div"),f(ze.$$.fragment),Do=r(),Ae=p("p"),Ae.innerHTML=jt,Oo=r(),f(re.$$.fragment),Ko=r(),f(ie.$$.fragment),Go=r(),f(Ue.$$.fragment),Co=r(),C=p("div"),f(We.$$.fragment),et=r(),De=p("p"),De.textContent=zt,ot=r(),Oe=p("p"),Oe.innerHTML=Ut,tt=r(),Ke=p("p"),Ke.innerHTML=Wt,nt=r(),j=p("div"),f(Fe.$$.fragment),st=r(),eo=p("p"),eo.innerHTML=Ft,at=r(),f(de.$$.fragment),rt=r(),f(le.$$.fragment),it=r(),f(ce.$$.fragment),Jo=r(),f(Ie.$$.fragment),xo=r(),J=p("div"),f(qe.$$.fragment),dt=r(),oo=p("p"),oo.textContent=It,lt=r(),to=p("p"),to.innerHTML=qt,ct=r(),no=p("p"),no.innerHTML=Zt,mt=r(),V=p("div"),f(Ze.$$.fragment),pt=r(),so=p("p"),so.innerHTML=Bt,ut=r(),f(me.$$.fragment),ht=r(),f(pe.$$.fragment),jo=r(),f(Be.$$.fragment),zo=r(),lo=p("p"),this.h()},l(e){const n=Et("svelte-u9bgzb",document.head);o=u(n,"META",{name:!0,content:!0}),n.forEach(s),d=i(e),t=u(e,"P",{}),z(t).forEach(s),l=i(e),M=u(e,"P",{"data-svelte-h":!0}),w(M)!=="svelte-r2aigt"&&(M.innerHTML=a),h=i(e),k=u(e,"DIV",{style:!0,"data-svelte-h":!0}),w(k)!=="svelte-2m0t7r"&&(k.innerHTML=io),he=i(e),g(E.$$.fragment,e),co=i(e),fe=u(e,"P",{"data-svelte-h":!0}),w(fe)!=="svelte-mtreyg"&&(fe.innerHTML=gt),mo=i(e),ge=u(e,"P",{"data-svelte-h":!0}),w(ge)!=="svelte-lop1e7"&&(ge.innerHTML=_t),po=i(e),_e=u(e,"P",{"data-svelte-h":!0}),w(_e)!=="svelte-t8jrrq"&&(_e.textContent=Tt),uo=i(e),g(ee.$$.fragment,e),ho=i(e),Te=u(e,"P",{"data-svelte-h":!0}),w(Te)!=="svelte-1eliowp"&&(Te.innerHTML=bt),fo=i(e),g(oe.$$.fragment,e),go=i(e),g(be.$$.fragment,e),_o=i(e),B=u(e,"DIV",{class:!0});var Q=z(B);g(ye.$$.fragment,Q),Wo=i(Q),Ve=u(Q,"P",{"data-svelte-h":!0}),w(Ve)!=="svelte-1af6rq1"&&(Ve.innerHTML=yt),Fo=i(Q),g(te.$$.fragment,Q),Io=Ht(Q,`
Configuration objects inherit from [PretrainedConfig] and can be used to control the model outputs. Read the
documentation from [PretrainedConfig] for more information.`),Q.forEach(s),To=i(e),g(Me.$$.fragment,e),bo=i(e),L=u(e,"DIV",{class:!0});var A=z(L);g(we.$$.fragment,A),qo=i(A),Xe=u(A,"P",{"data-svelte-h":!0}),w(Xe)!=="svelte-1tmdaa8"&&(Xe.innerHTML=Mt),Zo=i(A),g(ne.$$.fragment,A),A.forEach(s),yo=i(e),g(ve.$$.fragment,e),Mo=i(e),$=u(e,"DIV",{class:!0});var W=z($);g(ke.$$.fragment,W),Bo=i(W),Le=u(W,"P",{"data-svelte-h":!0}),w(Le)!=="svelte-26091u"&&(Le.textContent=wt),No=i(W),Re=u(W,"P",{"data-svelte-h":!0}),w(Re)!=="svelte-q52n56"&&(Re.innerHTML=vt),Vo=i(W),Se=u(W,"P",{"data-svelte-h":!0}),w(Se)!=="svelte-hswkmf"&&(Se.innerHTML=kt),Xo=i(W),H=u(W,"DIV",{class:!0});var D=z(H);g($e.$$.fragment,D),Lo=i(D),Ee=u(D,"P",{"data-svelte-h":!0}),w(Ee)!=="svelte-16b8owp"&&(Ee.innerHTML=$t),Ro=i(D),g(se.$$.fragment,D),D.forEach(s),W.forEach(s),wo=i(e),g(Ge.$$.fragment,e),vo=i(e),G=u(e,"DIV",{class:!0});var F=z(G);g(Ce.$$.fragment,F),So=i(F),He=u(F,"P",{"data-svelte-h":!0}),w(He)!=="svelte-26091u"&&(He.textContent=Gt),Eo=i(F),Ye=u(F,"P",{"data-svelte-h":!0}),w(Ye)!=="svelte-q52n56"&&(Ye.innerHTML=Ct),Ho=i(F),Qe=u(F,"P",{"data-svelte-h":!0}),w(Qe)!=="svelte-hswkmf"&&(Qe.innerHTML=Jt),Yo=i(F),Y=u(F,"DIV",{class:!0});var O=z(Y);g(Je.$$.fragment,O),Qo=i(O),Pe=u(O,"P",{"data-svelte-h":!0}),w(Pe)!=="svelte-uwmgrr"&&(Pe.innerHTML=xt),Po=i(O),g(ae.$$.fragment,O),O.forEach(s),F.forEach(s),ko=i(e),g(xe.$$.fragment,e),$o=i(e),P=u(e,"DIV",{class:!0});var Ne=z(P);g(je.$$.fragment,Ne),Ao=i(Ne),N=u(Ne,"DIV",{class:!0});var R=z(N);g(ze.$$.fragment,R),Do=i(R),Ae=u(R,"P",{"data-svelte-h":!0}),w(Ae)!=="svelte-skctnj"&&(Ae.innerHTML=jt),Oo=i(R),g(re.$$.fragment,R),Ko=i(R),g(ie.$$.fragment,R),R.forEach(s),Ne.forEach(s),Go=i(e),g(Ue.$$.fragment,e),Co=i(e),C=u(e,"DIV",{class:!0});var I=z(C);g(We.$$.fragment,I),et=i(I),De=u(I,"P",{"data-svelte-h":!0}),w(De)!=="svelte-17jn8x4"&&(De.textContent=zt),ot=i(I),Oe=u(I,"P",{"data-svelte-h":!0}),w(Oe)!=="svelte-q52n56"&&(Oe.innerHTML=Ut),tt=i(I),Ke=u(I,"P",{"data-svelte-h":!0}),w(Ke)!=="svelte-hswkmf"&&(Ke.innerHTML=Wt),nt=i(I),j=u(I,"DIV",{class:!0});var q=z(j);g(Fe.$$.fragment,q),st=i(q),eo=u(q,"P",{"data-svelte-h":!0}),w(eo)!=="svelte-1ezg36z"&&(eo.innerHTML=Ft),at=i(q),g(de.$$.fragment,q),rt=i(q),g(le.$$.fragment,q),it=i(q),g(ce.$$.fragment,q),q.forEach(s),I.forEach(s),Jo=i(e),g(Ie.$$.fragment,e),xo=i(e),J=u(e,"DIV",{class:!0});var Z=z(J);g(qe.$$.fragment,Z),dt=i(Z),oo=u(Z,"P",{"data-svelte-h":!0}),w(oo)!=="svelte-v9faxi"&&(oo.textContent=It),lt=i(Z),to=u(Z,"P",{"data-svelte-h":!0}),w(to)!=="svelte-q52n56"&&(to.innerHTML=qt),ct=i(Z),no=u(Z,"P",{"data-svelte-h":!0}),w(no)!=="svelte-hswkmf"&&(no.innerHTML=Zt),mt=i(Z),V=u(Z,"DIV",{class:!0});var S=z(V);g(Ze.$$.fragment,S),pt=i(S),so=u(S,"P",{"data-svelte-h":!0}),w(so)!=="svelte-1mgyx8n"&&(so.innerHTML=Bt),ut=i(S),g(me.$$.fragment,S),ht=i(S),g(pe.$$.fragment,S),S.forEach(s),Z.forEach(s),jo=i(e),g(Be.$$.fragment,e),zo=i(e),lo=u(e,"P",{}),z(lo).forEach(s),this.h()},h(){U(o,"name","hf:doc:metadata"),U(o,"content",hn),Yt(k,"float","right"),U(B,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),U(L,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),U(H,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),U($,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),U(Y,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),U(G,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),U(N,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),U(P,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),U(j,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),U(C,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),U(V,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),U(J,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,n){c(document.head,o),m(e,d,n),m(e,t,n),m(e,l,n),m(e,M,n),m(e,h,n),m(e,k,n),m(e,he,n),_(E,e,n),m(e,co,n),m(e,fe,n),m(e,mo,n),m(e,ge,n),m(e,po,n),m(e,_e,n),m(e,uo,n),_(ee,e,n),m(e,ho,n),m(e,Te,n),m(e,fo,n),_(oe,e,n),m(e,go,n),_(be,e,n),m(e,_o,n),m(e,B,n),_(ye,B,null),c(B,Wo),c(B,Ve),c(B,Fo),_(te,B,null),c(B,Io),m(e,To,n),_(Me,e,n),m(e,bo,n),m(e,L,n),_(we,L,null),c(L,qo),c(L,Xe),c(L,Zo),_(ne,L,null),m(e,yo,n),_(ve,e,n),m(e,Mo,n),m(e,$,n),_(ke,$,null),c($,Bo),c($,Le),c($,No),c($,Re),c($,Vo),c($,Se),c($,Xo),c($,H),_($e,H,null),c(H,Lo),c(H,Ee),c(H,Ro),_(se,H,null),m(e,wo,n),_(Ge,e,n),m(e,vo,n),m(e,G,n),_(Ce,G,null),c(G,So),c(G,He),c(G,Eo),c(G,Ye),c(G,Ho),c(G,Qe),c(G,Yo),c(G,Y),_(Je,Y,null),c(Y,Qo),c(Y,Pe),c(Y,Po),_(ae,Y,null),m(e,ko,n),_(xe,e,n),m(e,$o,n),m(e,P,n),_(je,P,null),c(P,Ao),c(P,N),_(ze,N,null),c(N,Do),c(N,Ae),c(N,Oo),_(re,N,null),c(N,Ko),_(ie,N,null),m(e,Go,n),_(Ue,e,n),m(e,Co,n),m(e,C,n),_(We,C,null),c(C,et),c(C,De),c(C,ot),c(C,Oe),c(C,tt),c(C,Ke),c(C,nt),c(C,j),_(Fe,j,null),c(j,st),c(j,eo),c(j,at),_(de,j,null),c(j,rt),_(le,j,null),c(j,it),_(ce,j,null),m(e,Jo,n),_(Ie,e,n),m(e,xo,n),m(e,J,n),_(qe,J,null),c(J,dt),c(J,oo),c(J,lt),c(J,to),c(J,ct),c(J,no),c(J,mt),c(J,V),_(Ze,V,null),c(V,pt),c(V,so),c(V,ut),_(me,V,null),c(V,ht),_(pe,V,null),m(e,jo,n),_(Be,e,n),m(e,zo,n),m(e,lo,n),Uo=!0},p(e,[n]){const Q={};n&2&&(Q.$$scope={dirty:n,ctx:e}),ee.$set(Q);const A={};n&2&&(A.$$scope={dirty:n,ctx:e}),oe.$set(A);const W={};n&2&&(W.$$scope={dirty:n,ctx:e}),te.$set(W);const D={};n&2&&(D.$$scope={dirty:n,ctx:e}),ne.$set(D);const F={};n&2&&(F.$$scope={dirty:n,ctx:e}),se.$set(F);const O={};n&2&&(O.$$scope={dirty:n,ctx:e}),ae.$set(O);const Ne={};n&2&&(Ne.$$scope={dirty:n,ctx:e}),re.$set(Ne);const R={};n&2&&(R.$$scope={dirty:n,ctx:e}),ie.$set(R);const I={};n&2&&(I.$$scope={dirty:n,ctx:e}),de.$set(I);const q={};n&2&&(q.$$scope={dirty:n,ctx:e}),le.$set(q);const Z={};n&2&&(Z.$$scope={dirty:n,ctx:e}),ce.$set(Z);const S={};n&2&&(S.$$scope={dirty:n,ctx:e}),me.$set(S);const Nt={};n&2&&(Nt.$$scope={dirty:n,ctx:e}),pe.$set(Nt)},i(e){Uo||(T(E.$$.fragment,e),T(ee.$$.fragment,e),T(oe.$$.fragment,e),T(be.$$.fragment,e),T(ye.$$.fragment,e),T(te.$$.fragment,e),T(Me.$$.fragment,e),T(we.$$.fragment,e),T(ne.$$.fragment,e),T(ve.$$.fragment,e),T(ke.$$.fragment,e),T($e.$$.fragment,e),T(se.$$.fragment,e),T(Ge.$$.fragment,e),T(Ce.$$.fragment,e),T(Je.$$.fragment,e),T(ae.$$.fragment,e),T(xe.$$.fragment,e),T(je.$$.fragment,e),T(ze.$$.fragment,e),T(re.$$.fragment,e),T(ie.$$.fragment,e),T(Ue.$$.fragment,e),T(We.$$.fragment,e),T(Fe.$$.fragment,e),T(de.$$.fragment,e),T(le.$$.fragment,e),T(ce.$$.fragment,e),T(Ie.$$.fragment,e),T(qe.$$.fragment,e),T(Ze.$$.fragment,e),T(me.$$.fragment,e),T(pe.$$.fragment,e),T(Be.$$.fragment,e),Uo=!0)},o(e){b(E.$$.fragment,e),b(ee.$$.fragment,e),b(oe.$$.fragment,e),b(be.$$.fragment,e),b(ye.$$.fragment,e),b(te.$$.fragment,e),b(Me.$$.fragment,e),b(we.$$.fragment,e),b(ne.$$.fragment,e),b(ve.$$.fragment,e),b(ke.$$.fragment,e),b($e.$$.fragment,e),b(se.$$.fragment,e),b(Ge.$$.fragment,e),b(Ce.$$.fragment,e),b(Je.$$.fragment,e),b(ae.$$.fragment,e),b(xe.$$.fragment,e),b(je.$$.fragment,e),b(ze.$$.fragment,e),b(re.$$.fragment,e),b(ie.$$.fragment,e),b(Ue.$$.fragment,e),b(We.$$.fragment,e),b(Fe.$$.fragment,e),b(de.$$.fragment,e),b(le.$$.fragment,e),b(ce.$$.fragment,e),b(Ie.$$.fragment,e),b(qe.$$.fragment,e),b(Ze.$$.fragment,e),b(me.$$.fragment,e),b(pe.$$.fragment,e),b(Be.$$.fragment,e),Uo=!1},d(e){e&&(s(d),s(t),s(l),s(M),s(h),s(k),s(he),s(co),s(fe),s(mo),s(ge),s(po),s(_e),s(uo),s(ho),s(Te),s(fo),s(go),s(_o),s(B),s(To),s(bo),s(L),s(yo),s(Mo),s($),s(wo),s(vo),s(G),s(ko),s($o),s(P),s(Go),s(Co),s(C),s(Jo),s(xo),s(J),s(jo),s(zo),s(lo)),s(o),y(E,e),y(ee,e),y(oe,e),y(be,e),y(ye),y(te),y(Me,e),y(we),y(ne),y(ve,e),y(ke),y($e),y(se),y(Ge,e),y(Ce),y(Je),y(ae),y(xe,e),y(je),y(ze),y(re),y(ie),y(Ue,e),y(We),y(Fe),y(de),y(le),y(ce),y(Ie,e),y(qe),y(Ze),y(me),y(pe),y(Be,e)}}}const hn='{"title":"T5Gemma","local":"t5gemma","sections":[{"title":"T5GemmaConfig","local":"transformers.T5GemmaConfig","sections":[],"depth":2},{"title":"T5GemmaModuleConfig","local":"transformers.T5GemmaModuleConfig","sections":[],"depth":2},{"title":"T5GemmaModel","local":"transformers.T5GemmaModel","sections":[],"depth":2},{"title":"T5GemmaEncoderModel","local":"transformers.T5GemmaEncoderModel","sections":[],"depth":2},{"title":"T5GemmaForConditionalGeneration","local":"transformers.T5GemmaForConditionalGeneration","sections":[],"depth":2},{"title":"T5GemmaForSequenceClassification","local":"transformers.T5GemmaForSequenceClassification","sections":[],"depth":2},{"title":"T5GemmaForTokenClassification","local":"transformers.T5GemmaForTokenClassification","sections":[],"depth":2}],"depth":1}';function fn(v){return Xt(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class kn extends Lt{constructor(o){super(),Rt(this,o,fn,un,Vt,{})}}export{kn as component};
