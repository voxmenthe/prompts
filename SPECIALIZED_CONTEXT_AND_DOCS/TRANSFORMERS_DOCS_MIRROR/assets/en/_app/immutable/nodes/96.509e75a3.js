import{s as an,o as rn,n as q}from"../chunks/scheduler.18a86fab.js";import{S as ln,i as dn,g as p,s as r,r as f,A as cn,h as m,f as a,c as i,j as G,x as T,u as g,k as I,l as pn,y as l,a as c,v as _,d as b,t as y,w as M}from"../chunks/index.98837b22.js";import{T as ft}from"../chunks/Tip.77304350.js";import{D as X}from"../chunks/Docstring.a1ef7999.js";import{C as A}from"../chunks/CodeBlock.8d0c2e8a.js";import{E as gt}from"../chunks/ExampleCodeBlock.8c3ee1f9.js";import{H as ce,E as mn}from"../chunks/getInferenceSnippets.06c2775f.js";import{H as un,a as Bo}from"../chunks/HfOption.6641485e.js";function hn(k){let t,u="Click on the BioGPT models in the right sidebar for more examples of how to apply BioGPT to different language tasks.";return{c(){t=p("p"),t.textContent=u},l(o){t=m(o,"P",{"data-svelte-h":!0}),T(t)!=="svelte-1tx6eox"&&(t.textContent=u)},m(o,d){c(o,t,d)},p:q,d(o){o&&a(t)}}}function fn(k){let t,u;return t=new A({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwcGlwZWxpbmUlMEElMEFnZW5lcmF0b3IlMjAlM0QlMjBwaXBlbGluZSglMEElMjAlMjAlMjAlMjB0YXNrJTNEJTIydGV4dC1nZW5lcmF0aW9uJTIyJTJDJTBBJTIwJTIwJTIwJTIwbW9kZWwlM0QlMjJtaWNyb3NvZnQlMkZiaW9ncHQlMjIlMkMlMEElMjAlMjAlMjAlMjBkdHlwZSUzRHRvcmNoLmZsb2F0MTYlMkMlMEElMjAlMjAlMjAlMjBkZXZpY2UlM0QwJTJDJTBBKSUwQXJlc3VsdCUyMCUzRCUyMGdlbmVyYXRvciglMjJJYnVwcm9mZW4lMjBpcyUyMGJlc3QlMjB1c2VkJTIwZm9yJTIyJTJDJTIwdHJ1bmNhdGlvbiUzRFRydWUlMkMlMjBtYXhfbGVuZ3RoJTNENTAlMkMlMjBkb19zYW1wbGUlM0RUcnVlKSU1QjAlNUQlNUIlMjJnZW5lcmF0ZWRfdGV4dCUyMiU1RCUwQXByaW50KHJlc3VsdCk=",highlighted:`<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> pipeline

generator = pipeline(
    task=<span class="hljs-string">&quot;text-generation&quot;</span>,
    model=<span class="hljs-string">&quot;microsoft/biogpt&quot;</span>,
    dtype=torch.float16,
    device=<span class="hljs-number">0</span>,
)
result = generator(<span class="hljs-string">&quot;Ibuprofen is best used for&quot;</span>, truncation=<span class="hljs-literal">True</span>, max_length=<span class="hljs-number">50</span>, do_sample=<span class="hljs-literal">True</span>)[<span class="hljs-number">0</span>][<span class="hljs-string">&quot;generated_text&quot;</span>]
<span class="hljs-built_in">print</span>(result)`,wrap:!1}}),{c(){f(t.$$.fragment)},l(o){g(t.$$.fragment,o)},m(o,d){_(t,o,d),u=!0},p:q,i(o){u||(b(t.$$.fragment,o),u=!0)},o(o){y(t.$$.fragment,o),u=!1},d(o){M(t,o)}}}function gn(k){let t,u;return t=new A({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b01vZGVsRm9yQ2F1c2FsTE0lMkMlMjBBdXRvVG9rZW5pemVyJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIybWljcm9zb2Z0JTJGYmlvZ3B0JTIyKSUwQW1vZGVsJTIwJTNEJTIwQXV0b01vZGVsRm9yQ2F1c2FsTE0uZnJvbV9wcmV0cmFpbmVkKCUwQSUyMCUyMCUyMCUyMCUyMm1pY3Jvc29mdCUyRmJpb2dwdCUyMiUyQyUwQSUyMCUyMCUyMCUyMGR0eXBlJTNEdG9yY2guZmxvYXQxNiUyQyUwQSUyMCUyMCUyMCUyMGRldmljZV9tYXAlM0QlMjJhdXRvJTIyJTJDJTBBJTIwJTIwJTIwJTIwYXR0bl9pbXBsZW1lbnRhdGlvbiUzRCUyMnNkcGElMjIlMEEpJTBBJTBBaW5wdXRfdGV4dCUyMCUzRCUyMCUyMklidXByb2ZlbiUyMGlzJTIwYmVzdCUyMHVzZWQlMjBmb3IlMjIlMEFpbnB1dHMlMjAlM0QlMjB0b2tlbml6ZXIoaW5wdXRfdGV4dCUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpLnRvKG1vZGVsLmRldmljZSklMEElMEF3aXRoJTIwdG9yY2gubm9fZ3JhZCgpJTNBJTBBJTIwJTIwJTIwJTIwZ2VuZXJhdGVkX2lkcyUyMCUzRCUyMG1vZGVsLmdlbmVyYXRlKCoqaW5wdXRzJTJDJTIwbWF4X2xlbmd0aCUzRDUwKSUwQSUyMCUyMCUyMCUyMCUwQW91dHB1dCUyMCUzRCUyMHRva2VuaXplci5kZWNvZGUoZ2VuZXJhdGVkX2lkcyU1QjAlNUQlMkMlMjBza2lwX3NwZWNpYWxfdG9rZW5zJTNEVHJ1ZSklMEFwcmludChvdXRwdXQp",highlighted:`<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;microsoft/biogpt&quot;</span>)
model = AutoModelForCausalLM.from_pretrained(
    <span class="hljs-string">&quot;microsoft/biogpt&quot;</span>,
    dtype=torch.float16,
    device_map=<span class="hljs-string">&quot;auto&quot;</span>,
    attn_implementation=<span class="hljs-string">&quot;sdpa&quot;</span>
)

input_text = <span class="hljs-string">&quot;Ibuprofen is best used for&quot;</span>
inputs = tokenizer(input_text, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>).to(model.device)

<span class="hljs-keyword">with</span> torch.no_grad():
    generated_ids = model.generate(**inputs, max_length=<span class="hljs-number">50</span>)
    
output = tokenizer.decode(generated_ids[<span class="hljs-number">0</span>], skip_special_tokens=<span class="hljs-literal">True</span>)
<span class="hljs-built_in">print</span>(output)`,wrap:!1}}),{c(){f(t.$$.fragment)},l(o){g(t.$$.fragment,o)},m(o,d){_(t,o,d),u=!0},p:q,i(o){u||(b(t.$$.fragment,o),u=!0)},o(o){y(t.$$.fragment,o),u=!1},d(o){M(t,o)}}}function _n(k){let t,u;return t=new A({props:{code:"ZWNobyUyMC1lJTIwJTIySWJ1cHJvZmVuJTIwaXMlMjBiZXN0JTIwdXNlZCUyMGZvciUyMiUyMCU3QyUyMHRyYW5zZm9ybWVycy1jbGklMjBydW4lMjAtLXRhc2slMjB0ZXh0LWdlbmVyYXRpb24lMjAtLW1vZGVsJTIwbWljcm9zb2Z0JTJGYmlvZ3B0JTIwLS1kZXZpY2UlMjAw",highlighted:'<span class="hljs-built_in">echo</span> -e <span class="hljs-string">&quot;Ibuprofen is best used for&quot;</span> | transformers-cli run --task text-generation --model microsoft/biogpt --device 0',wrap:!1}}),{c(){f(t.$$.fragment)},l(o){g(t.$$.fragment,o)},m(o,d){_(t,o,d),u=!0},p:q,i(o){u||(b(t.$$.fragment,o),u=!0)},o(o){y(t.$$.fragment,o),u=!1},d(o){M(t,o)}}}function bn(k){let t,u,o,d,v,n;return t=new Bo({props:{id:"usage",option:"Pipeline",$$slots:{default:[fn]},$$scope:{ctx:k}}}),o=new Bo({props:{id:"usage",option:"AutoModel",$$slots:{default:[gn]},$$scope:{ctx:k}}}),v=new Bo({props:{id:"usage",option:"transformers CLI",$$slots:{default:[_n]},$$scope:{ctx:k}}}),{c(){f(t.$$.fragment),u=r(),f(o.$$.fragment),d=r(),f(v.$$.fragment)},l(h){g(t.$$.fragment,h),u=i(h),g(o.$$.fragment,h),d=i(h),g(v.$$.fragment,h)},m(h,w){_(t,h,w),c(h,u,w),_(o,h,w),c(h,d,w),_(v,h,w),n=!0},p(h,w){const mt={};w&2&&(mt.$$scope={dirty:w,ctx:h}),t.$set(mt);const pe={};w&2&&(pe.$$scope={dirty:w,ctx:h}),o.$set(pe);const Q={};w&2&&(Q.$$scope={dirty:w,ctx:h}),v.$set(Q)},i(h){n||(b(t.$$.fragment,h),b(o.$$.fragment,h),b(v.$$.fragment,h),n=!0)},o(h){y(t.$$.fragment,h),y(o.$$.fragment,h),y(v.$$.fragment,h),n=!1},d(h){h&&(a(u),a(d)),M(t,h),M(o,h),M(v,h)}}}function yn(k){let t,u="Example:",o,d,v;return d=new A({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEJpb0dwdE1vZGVsJTJDJTIwQmlvR3B0Q29uZmlnJTBBJTBBJTIzJTIwSW5pdGlhbGl6aW5nJTIwYSUyMEJpb0dQVCUyMG1pY3Jvc29mdCUyRmJpb2dwdCUyMHN0eWxlJTIwY29uZmlndXJhdGlvbiUwQWNvbmZpZ3VyYXRpb24lMjAlM0QlMjBCaW9HcHRDb25maWcoKSUwQSUwQSUyMyUyMEluaXRpYWxpemluZyUyMGElMjBtb2RlbCUyMGZyb20lMjB0aGUlMjBtaWNyb3NvZnQlMkZiaW9ncHQlMjBzdHlsZSUyMGNvbmZpZ3VyYXRpb24lMEFtb2RlbCUyMCUzRCUyMEJpb0dwdE1vZGVsKGNvbmZpZ3VyYXRpb24pJTBBJTBBJTIzJTIwQWNjZXNzaW5nJTIwdGhlJTIwbW9kZWwlMjBjb25maWd1cmF0aW9uJTBBY29uZmlndXJhdGlvbiUyMCUzRCUyMG1vZGVsLmNvbmZpZw==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> BioGptModel, BioGptConfig

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a BioGPT microsoft/biogpt style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = BioGptConfig()

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a model from the microsoft/biogpt style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = BioGptModel(configuration)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Accessing the model configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = model.config`,wrap:!1}}),{c(){t=p("p"),t.textContent=u,o=r(),f(d.$$.fragment)},l(n){t=m(n,"P",{"data-svelte-h":!0}),T(t)!=="svelte-11lpom8"&&(t.textContent=u),o=i(n),g(d.$$.fragment,n)},m(n,h){c(n,t,h),c(n,o,h),_(d,n,h),v=!0},p:q,i(n){v||(b(d.$$.fragment,n),v=!0)},o(n){y(d.$$.fragment,n),v=!1},d(n){n&&(a(t),a(o)),M(d,n)}}}function Mn(k){let t,u=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=p("p"),t.innerHTML=u},l(o){t=m(o,"P",{"data-svelte-h":!0}),T(t)!=="svelte-fincs2"&&(t.innerHTML=u)},m(o,d){c(o,t,d)},p:q,d(o){o&&a(t)}}}function Tn(k){let t,u=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=p("p"),t.innerHTML=u},l(o){t=m(o,"P",{"data-svelte-h":!0}),T(t)!=="svelte-fincs2"&&(t.innerHTML=u)},m(o,d){c(o,t,d)},p:q,d(o){o&&a(t)}}}function vn(k){let t,u="Example:",o,d,v;return d=new A({props:{code:"",highlighted:"",wrap:!1}}),{c(){t=p("p"),t.textContent=u,o=r(),f(d.$$.fragment)},l(n){t=m(n,"P",{"data-svelte-h":!0}),T(t)!=="svelte-11lpom8"&&(t.textContent=u),o=i(n),g(d.$$.fragment,n)},m(n,h){c(n,t,h),c(n,o,h),_(d,n,h),v=!0},p:q,i(n){v||(b(d.$$.fragment,n),v=!0)},o(n){y(d.$$.fragment,n),v=!1},d(n){n&&(a(t),a(o)),M(d,n)}}}function kn(k){let t,u=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=p("p"),t.innerHTML=u},l(o){t=m(o,"P",{"data-svelte-h":!0}),T(t)!=="svelte-fincs2"&&(t.innerHTML=u)},m(o,d){c(o,t,d)},p:q,d(o){o&&a(t)}}}function wn(k){let t,u="Example:",o,d,v;return d=new A({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBCaW9HcHRGb3JUb2tlbkNsYXNzaWZpY2F0aW9uJTBBaW1wb3J0JTIwdG9yY2glMEElMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJtaWNyb3NvZnQlMkZiaW9ncHQlMjIpJTBBbW9kZWwlMjAlM0QlMjBCaW9HcHRGb3JUb2tlbkNsYXNzaWZpY2F0aW9uLmZyb21fcHJldHJhaW5lZCglMjJtaWNyb3NvZnQlMkZiaW9ncHQlMjIpJTBBJTBBaW5wdXRzJTIwJTNEJTIwdG9rZW5pemVyKCUwQSUyMCUyMCUyMCUyMCUyMkh1Z2dpbmdGYWNlJTIwaXMlMjBhJTIwY29tcGFueSUyMGJhc2VkJTIwaW4lMjBQYXJpcyUyMGFuZCUyME5ldyUyMFlvcmslMjIlMkMlMjBhZGRfc3BlY2lhbF90b2tlbnMlM0RGYWxzZSUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIlMEEpJTBBJTBBd2l0aCUyMHRvcmNoLm5vX2dyYWQoKSUzQSUwQSUyMCUyMCUyMCUyMGxvZ2l0cyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzKS5sb2dpdHMlMEElMEFwcmVkaWN0ZWRfdG9rZW5fY2xhc3NfaWRzJTIwJTNEJTIwbG9naXRzLmFyZ21heCgtMSklMEElMEElMjMlMjBOb3RlJTIwdGhhdCUyMHRva2VucyUyMGFyZSUyMGNsYXNzaWZpZWQlMjByYXRoZXIlMjB0aGVuJTIwaW5wdXQlMjB3b3JkcyUyMHdoaWNoJTIwbWVhbnMlMjB0aGF0JTBBJTIzJTIwdGhlcmUlMjBtaWdodCUyMGJlJTIwbW9yZSUyMHByZWRpY3RlZCUyMHRva2VuJTIwY2xhc3NlcyUyMHRoYW4lMjB3b3Jkcy4lMEElMjMlMjBNdWx0aXBsZSUyMHRva2VuJTIwY2xhc3NlcyUyMG1pZ2h0JTIwYWNjb3VudCUyMGZvciUyMHRoZSUyMHNhbWUlMjB3b3JkJTBBcHJlZGljdGVkX3Rva2Vuc19jbGFzc2VzJTIwJTNEJTIwJTVCbW9kZWwuY29uZmlnLmlkMmxhYmVsJTVCdC5pdGVtKCklNUQlMjBmb3IlMjB0JTIwaW4lMjBwcmVkaWN0ZWRfdG9rZW5fY2xhc3NfaWRzJTVCMCU1RCU1RCUwQXByZWRpY3RlZF90b2tlbnNfY2xhc3NlcyUwQSUwQWxhYmVscyUyMCUzRCUyMHByZWRpY3RlZF90b2tlbl9jbGFzc19pZHMlMEFsb3NzJTIwJTNEJTIwbW9kZWwoKippbnB1dHMlMkMlMjBsYWJlbHMlM0RsYWJlbHMpLmxvc3MlMEFyb3VuZChsb3NzLml0ZW0oKSUyQyUyMDIp",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, BioGptForTokenClassification
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;microsoft/biogpt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = BioGptForTokenClassification.from_pretrained(<span class="hljs-string">&quot;microsoft/biogpt&quot;</span>)

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
...`,wrap:!1}}),{c(){t=p("p"),t.textContent=u,o=r(),f(d.$$.fragment)},l(n){t=m(n,"P",{"data-svelte-h":!0}),T(t)!=="svelte-11lpom8"&&(t.textContent=u),o=i(n),g(d.$$.fragment,n)},m(n,h){c(n,t,h),c(n,o,h),_(d,n,h),v=!0},p:q,i(n){v||(b(d.$$.fragment,n),v=!0)},o(n){y(d.$$.fragment,n),v=!1},d(n){n&&(a(t),a(o)),M(d,n)}}}function Cn(k){let t,u=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=p("p"),t.innerHTML=u},l(o){t=m(o,"P",{"data-svelte-h":!0}),T(t)!=="svelte-fincs2"&&(t.innerHTML=u)},m(o,d){c(o,t,d)},p:q,d(o){o&&a(t)}}}function $n(k){let t,u="Example of single-label classification:",o,d,v;return d=new A({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Rva2VuaXplciUyQyUyMEJpb0dwdEZvclNlcXVlbmNlQ2xhc3NpZmljYXRpb24lMEElMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJtaWNyb3NvZnQlMkZiaW9ncHQlMjIpJTBBbW9kZWwlMjAlM0QlMjBCaW9HcHRGb3JTZXF1ZW5jZUNsYXNzaWZpY2F0aW9uLmZyb21fcHJldHJhaW5lZCglMjJtaWNyb3NvZnQlMkZiaW9ncHQlMjIpJTBBJTBBaW5wdXRzJTIwJTNEJTIwdG9rZW5pemVyKCUyMkhlbGxvJTJDJTIwbXklMjBkb2clMjBpcyUyMGN1dGUlMjIlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQSUwQXdpdGglMjB0b3JjaC5ub19ncmFkKCklM0ElMEElMjAlMjAlMjAlMjBsb2dpdHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cykubG9naXRzJTBBJTBBcHJlZGljdGVkX2NsYXNzX2lkJTIwJTNEJTIwbG9naXRzLmFyZ21heCgpLml0ZW0oKSUwQW1vZGVsLmNvbmZpZy5pZDJsYWJlbCU1QnByZWRpY3RlZF9jbGFzc19pZCU1RCUwQSUwQSUyMyUyMFRvJTIwdHJhaW4lMjBhJTIwbW9kZWwlMjBvbiUyMCU2MG51bV9sYWJlbHMlNjAlMjBjbGFzc2VzJTJDJTIweW91JTIwY2FuJTIwcGFzcyUyMCU2MG51bV9sYWJlbHMlM0RudW1fbGFiZWxzJTYwJTIwdG8lMjAlNjAuZnJvbV9wcmV0cmFpbmVkKC4uLiklNjAlMEFudW1fbGFiZWxzJTIwJTNEJTIwbGVuKG1vZGVsLmNvbmZpZy5pZDJsYWJlbCklMEFtb2RlbCUyMCUzRCUyMEJpb0dwdEZvclNlcXVlbmNlQ2xhc3NpZmljYXRpb24uZnJvbV9wcmV0cmFpbmVkKCUyMm1pY3Jvc29mdCUyRmJpb2dwdCUyMiUyQyUyMG51bV9sYWJlbHMlM0RudW1fbGFiZWxzKSUwQSUwQWxhYmVscyUyMCUzRCUyMHRvcmNoLnRlbnNvciglNUIxJTVEKSUwQWxvc3MlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyUyQyUyMGxhYmVscyUzRGxhYmVscykubG9zcyUwQXJvdW5kKGxvc3MuaXRlbSgpJTJDJTIwMik=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, BioGptForSequenceClassification

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;microsoft/biogpt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = BioGptForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;microsoft/biogpt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    logits = model(**inputs).logits

<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_class_id = logits.argmax().item()
<span class="hljs-meta">&gt;&gt;&gt; </span>model.config.id2label[predicted_class_id]
...

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># To train a model on \`num_labels\` classes, you can pass \`num_labels=num_labels\` to \`.from_pretrained(...)\`</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>num_labels = <span class="hljs-built_in">len</span>(model.config.id2label)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = BioGptForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;microsoft/biogpt&quot;</span>, num_labels=num_labels)

<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.tensor([<span class="hljs-number">1</span>])
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = model(**inputs, labels=labels).loss
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">round</span>(loss.item(), <span class="hljs-number">2</span>)
...`,wrap:!1}}),{c(){t=p("p"),t.textContent=u,o=r(),f(d.$$.fragment)},l(n){t=m(n,"P",{"data-svelte-h":!0}),T(t)!=="svelte-ykxpe4"&&(t.textContent=u),o=i(n),g(d.$$.fragment,n)},m(n,h){c(n,t,h),c(n,o,h),_(d,n,h),v=!0},p:q,i(n){v||(b(d.$$.fragment,n),v=!0)},o(n){y(d.$$.fragment,n),v=!1},d(n){n&&(a(t),a(o)),M(d,n)}}}function Jn(k){let t,u="Example of multi-label classification:",o,d,v;return d=new A({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Rva2VuaXplciUyQyUyMEJpb0dwdEZvclNlcXVlbmNlQ2xhc3NpZmljYXRpb24lMEElMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJtaWNyb3NvZnQlMkZiaW9ncHQlMjIpJTBBbW9kZWwlMjAlM0QlMjBCaW9HcHRGb3JTZXF1ZW5jZUNsYXNzaWZpY2F0aW9uLmZyb21fcHJldHJhaW5lZCglMjJtaWNyb3NvZnQlMkZiaW9ncHQlMjIlMkMlMjBwcm9ibGVtX3R5cGUlM0QlMjJtdWx0aV9sYWJlbF9jbGFzc2lmaWNhdGlvbiUyMiklMEElMEFpbnB1dHMlMjAlM0QlMjB0b2tlbml6ZXIoJTIySGVsbG8lMkMlMjBteSUyMGRvZyUyMGlzJTIwY3V0ZSUyMiUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpJTBBJTBBd2l0aCUyMHRvcmNoLm5vX2dyYWQoKSUzQSUwQSUyMCUyMCUyMCUyMGxvZ2l0cyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzKS5sb2dpdHMlMEElMEFwcmVkaWN0ZWRfY2xhc3NfaWRzJTIwJTNEJTIwdG9yY2guYXJhbmdlKDAlMkMlMjBsb2dpdHMuc2hhcGUlNUItMSU1RCklNUJ0b3JjaC5zaWdtb2lkKGxvZ2l0cykuc3F1ZWV6ZShkaW0lM0QwKSUyMCUzRSUyMDAuNSU1RCUwQSUwQSUyMyUyMFRvJTIwdHJhaW4lMjBhJTIwbW9kZWwlMjBvbiUyMCU2MG51bV9sYWJlbHMlNjAlMjBjbGFzc2VzJTJDJTIweW91JTIwY2FuJTIwcGFzcyUyMCU2MG51bV9sYWJlbHMlM0RudW1fbGFiZWxzJTYwJTIwdG8lMjAlNjAuZnJvbV9wcmV0cmFpbmVkKC4uLiklNjAlMEFudW1fbGFiZWxzJTIwJTNEJTIwbGVuKG1vZGVsLmNvbmZpZy5pZDJsYWJlbCklMEFtb2RlbCUyMCUzRCUyMEJpb0dwdEZvclNlcXVlbmNlQ2xhc3NpZmljYXRpb24uZnJvbV9wcmV0cmFpbmVkKCUwQSUyMCUyMCUyMCUyMCUyMm1pY3Jvc29mdCUyRmJpb2dwdCUyMiUyQyUyMG51bV9sYWJlbHMlM0RudW1fbGFiZWxzJTJDJTIwcHJvYmxlbV90eXBlJTNEJTIybXVsdGlfbGFiZWxfY2xhc3NpZmljYXRpb24lMjIlMEEpJTBBJTBBbGFiZWxzJTIwJTNEJTIwdG9yY2guc3VtKCUwQSUyMCUyMCUyMCUyMHRvcmNoLm5uLmZ1bmN0aW9uYWwub25lX2hvdChwcmVkaWN0ZWRfY2xhc3NfaWRzJTVCTm9uZSUyQyUyMCUzQSU1RC5jbG9uZSgpJTJDJTIwbnVtX2NsYXNzZXMlM0RudW1fbGFiZWxzKSUyQyUyMGRpbSUzRDElMEEpLnRvKHRvcmNoLmZsb2F0KSUwQWxvc3MlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyUyQyUyMGxhYmVscyUzRGxhYmVscykubG9zcw==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, BioGptForSequenceClassification

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;microsoft/biogpt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = BioGptForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;microsoft/biogpt&quot;</span>, problem_type=<span class="hljs-string">&quot;multi_label_classification&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    logits = model(**inputs).logits

<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_class_ids = torch.arange(<span class="hljs-number">0</span>, logits.shape[-<span class="hljs-number">1</span>])[torch.sigmoid(logits).squeeze(dim=<span class="hljs-number">0</span>) &gt; <span class="hljs-number">0.5</span>]

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># To train a model on \`num_labels\` classes, you can pass \`num_labels=num_labels\` to \`.from_pretrained(...)\`</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>num_labels = <span class="hljs-built_in">len</span>(model.config.id2label)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = BioGptForSequenceClassification.from_pretrained(
<span class="hljs-meta">... </span>    <span class="hljs-string">&quot;microsoft/biogpt&quot;</span>, num_labels=num_labels, problem_type=<span class="hljs-string">&quot;multi_label_classification&quot;</span>
<span class="hljs-meta">... </span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.<span class="hljs-built_in">sum</span>(
<span class="hljs-meta">... </span>    torch.nn.functional.one_hot(predicted_class_ids[<span class="hljs-literal">None</span>, :].clone(), num_classes=num_labels), dim=<span class="hljs-number">1</span>
<span class="hljs-meta">... </span>).to(torch.<span class="hljs-built_in">float</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = model(**inputs, labels=labels).loss`,wrap:!1}}),{c(){t=p("p"),t.textContent=u,o=r(),f(d.$$.fragment)},l(n){t=m(n,"P",{"data-svelte-h":!0}),T(t)!=="svelte-1l8e32d"&&(t.textContent=u),o=i(n),g(d.$$.fragment,n)},m(n,h){c(n,t,h),c(n,o,h),_(d,n,h),v=!0},p:q,i(n){v||(b(d.$$.fragment,n),v=!0)},o(n){y(d.$$.fragment,n),v=!1},d(n){n&&(a(t),a(o)),M(d,n)}}}function Bn(k){let t,u,o,d,v,n="<em>This model was released on 2022-10-19 and added to Hugging Face Transformers on 2022-12-05.</em>",h,w,mt='<div class="flex flex-wrap space-x-1"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/> <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat"/> <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/></div>',pe,Q,_t,me,Go='<a href="https://huggingface.co/papers/2210.10341" rel="nofollow">BioGPT</a> is a generative Transformer model based on <a href="./gpt2">GPT-2</a> and pretrained on 15 million PubMed abstracts. It is designed for biomedical language tasks.',bt,ue,jo='You can find all the original BioGPT checkpoints under the <a href="https://huggingface.co/microsoft?search_models=biogpt" rel="nofollow">Microsoft</a> organization.',yt,K,Mt,he,Uo='The example below demonstrates how to generate biomedical text with <a href="/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.Pipeline">Pipeline</a>, <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel">AutoModel</a>, and also from the command line.',Tt,ee,vt,fe,xo='Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the <a href="../quantization/overview">Quantization</a> overview for more available quantization backends.',kt,ge,zo='The example below uses <a href="../quantization/bitsandbytes">bitsandbytes</a> to only quantize the weights to 4-bit precision.',wt,_e,Ct,be,$t,S,Re,Zo="<p>Pad inputs on the right because BioGPT uses absolute position embeddings.</p>",Vt,He,Wo='<p>BioGPT can reuse previously computed key-value attention pairs. Access this feature with the <a href="https://huggingface.co/docs/transformers/main/en/model_doc/biogpt#transformers.BioGptModel.forward.past_key_values" rel="nofollow">past_key_values</a> parameter in <code>BioGPTModel.forward</code>.</p>',Lt,ye,Ve,Fo="The <code>head_mask</code> argument is ignored when using an attention implementation other than “eager”. If you want to use <code>head_mask</code>, make sure <code>attn_implementation=&quot;eager&quot;</code>).",Et,Me,Jt,Te,Bt,x,ve,Xt,Le,Io=`This is the configuration class to store the configuration of a <a href="/docs/transformers/v4.56.2/en/model_doc/biogpt#transformers.BioGptModel">BioGptModel</a>. It is used to instantiate an
BioGPT model according to the specified arguments, defining the model architecture. Instantiating a configuration
with the defaults will yield a similar configuration to that of the BioGPT
<a href="https://huggingface.co/microsoft/biogpt" rel="nofollow">microsoft/biogpt</a> architecture.`,Qt,Ee,qo=`Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> and can be used to control the model outputs. Read the
documentation from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> for more information.`,St,te,Gt,ke,jt,z,we,Pt,Xe,No="Construct an FAIRSEQ Transformer tokenizer. Moses tokenization followed by Byte-Pair Encoding.",Yt,Qe,Ro=`This tokenizer inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer">PreTrainedTokenizer</a> which contains most of the main methods. Users should refer to
this superclass for more information regarding those methods.`,At,Se,Ce,Ut,$e,xt,$,Je,Ot,Pe,Ho="The bare Biogpt Model outputting raw hidden-states without any specific head on top.",Dt,Ye,Vo=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Kt,Ae,Lo=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,eo,P,Be,to,Oe,Eo='The <a href="/docs/transformers/v4.56.2/en/model_doc/biogpt#transformers.BioGptModel">BioGptModel</a> forward method, overrides the <code>__call__</code> special method.',oo,oe,zt,Ge,Zt,J,je,no,De,Xo="BioGPT Model with a <code>language modeling</code> head on top for CLM fine-tuning.",so,Ke,Qo=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,ao,et,So=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,ro,N,Ue,io,tt,Po='The <a href="/docs/transformers/v4.56.2/en/model_doc/biogpt#transformers.BioGptForCausalLM">BioGptForCausalLM</a> forward method, overrides the <code>__call__</code> special method.',lo,ne,co,se,Wt,xe,Ft,B,ze,po,ot,Yo=`The Biogpt transformer with a token classification head on top (a linear layer on top of the hidden-states
output) e.g. for Named-Entity-Recognition (NER) tasks.`,mo,nt,Ao=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,uo,st,Oo=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,ho,R,Ze,fo,at,Do='The <a href="/docs/transformers/v4.56.2/en/model_doc/biogpt#transformers.BioGptForTokenClassification">BioGptForTokenClassification</a> forward method, overrides the <code>__call__</code> special method.',go,ae,_o,re,It,We,qt,C,Fe,bo,rt,Ko="The BioGpt Model transformer with a sequence classification head on top (linear layer).",yo,it,en=`<a href="/docs/transformers/v4.56.2/en/model_doc/biogpt#transformers.BioGptForSequenceClassification">BioGptForSequenceClassification</a> uses the last token in order to do the classification, as other causal models
(e.g. GPT-2) do.`,Mo,lt,tn=`Since it does classification on the last token, it is required to know the position of the last token. If a
<code>pad_token_id</code> is defined in the configuration, it finds the last token that is not a padding token in each row. If
no <code>pad_token_id</code> is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
padding tokens when <code>inputs_embeds</code> are passed instead of <code>input_ids</code>, it does the same (take the last value in
each row of the batch).`,To,dt,on=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,vo,ct,nn=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,ko,j,Ie,wo,pt,sn='The <a href="/docs/transformers/v4.56.2/en/model_doc/biogpt#transformers.BioGptForSequenceClassification">BioGptForSequenceClassification</a> forward method, overrides the <code>__call__</code> special method.',Co,ie,$o,le,Jo,de,Nt,qe,Rt,ut,Ht;return Q=new ce({props:{title:"BioGPT",local:"biogpt",headingTag:"h1"}}),K=new ft({props:{warning:!1,$$slots:{default:[hn]},$$scope:{ctx:k}}}),ee=new un({props:{id:"usage",options:["Pipeline","AutoModel","transformers CLI"],$$slots:{default:[bn]},$$scope:{ctx:k}}}),_e=new A({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Rva2VuaXplciUyQyUyMEF1dG9Nb2RlbEZvckNhdXNhbExNJTJDJTIwQml0c0FuZEJ5dGVzQ29uZmlnJTBBJTBBYm5iX2NvbmZpZyUyMCUzRCUyMEJpdHNBbmRCeXRlc0NvbmZpZyglMEElMjAlMjAlMjAlMjBsb2FkX2luXzRiaXQlM0RUcnVlJTJDJTBBJTIwJTIwJTIwJTIwYm5iXzRiaXRfcXVhbnRfdHlwZSUzRCUyMm5mNCUyMiUyQyUwQSUyMCUyMCUyMCUyMGJuYl80Yml0X2NvbXB1dGVfZHR5cGUlM0R0b3JjaC5iZmxvYXQxNiUyQyUwQSUyMCUyMCUyMCUyMGJuYl80Yml0X3VzZV9kb3VibGVfcXVhbnQlM0RUcnVlJTBBKSUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMm1pY3Jvc29mdCUyRkJpb0dQVC1MYXJnZSUyMiklMEFtb2RlbCUyMCUzRCUyMEF1dG9Nb2RlbEZvckNhdXNhbExNLmZyb21fcHJldHJhaW5lZCglMEElMjAlMjAlMjAlMjAlMjJtaWNyb3NvZnQlMkZCaW9HUFQtTGFyZ2UlMjIlMkMlMjAlMEElMjAlMjAlMjAlMjBxdWFudGl6YXRpb25fY29uZmlnJTNEYm5iX2NvbmZpZyUyQyUwQSUyMCUyMCUyMCUyMGR0eXBlJTNEdG9yY2guYmZsb2F0MTYlMkMlMEElMjAlMjAlMjAlMjBkZXZpY2VfbWFwJTNEJTIyYXV0byUyMiUwQSklMEElMEFpbnB1dF90ZXh0JTIwJTNEJTIwJTIySWJ1cHJvZmVuJTIwaXMlMjBiZXN0JTIwdXNlZCUyMGZvciUyMiUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplcihpbnB1dF90ZXh0JTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMikudG8obW9kZWwuZGV2aWNlKSUwQXdpdGglMjB0b3JjaC5ub19ncmFkKCklM0ElMEElMjAlMjAlMjAlMjBnZW5lcmF0ZWRfaWRzJTIwJTNEJTIwbW9kZWwuZ2VuZXJhdGUoKippbnB1dHMlMkMlMjBtYXhfbGVuZ3RoJTNENTApJTIwJTIwJTIwJTIwJTBBb3V0cHV0JTIwJTNEJTIwdG9rZW5pemVyLmRlY29kZShnZW5lcmF0ZWRfaWRzJTVCMCU1RCUyQyUyMHNraXBfc3BlY2lhbF90b2tlbnMlM0RUcnVlKSUwQXByaW50KG91dHB1dCk=",highlighted:`<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=<span class="hljs-literal">True</span>,
    bnb_4bit_quant_type=<span class="hljs-string">&quot;nf4&quot;</span>,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=<span class="hljs-literal">True</span>
)

tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;microsoft/BioGPT-Large&quot;</span>)
model = AutoModelForCausalLM.from_pretrained(
    <span class="hljs-string">&quot;microsoft/BioGPT-Large&quot;</span>, 
    quantization_config=bnb_config,
    dtype=torch.bfloat16,
    device_map=<span class="hljs-string">&quot;auto&quot;</span>
)

input_text = <span class="hljs-string">&quot;Ibuprofen is best used for&quot;</span>
inputs = tokenizer(input_text, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>).to(model.device)
<span class="hljs-keyword">with</span> torch.no_grad():
    generated_ids = model.generate(**inputs, max_length=<span class="hljs-number">50</span>)    
output = tokenizer.decode(generated_ids[<span class="hljs-number">0</span>], skip_special_tokens=<span class="hljs-literal">True</span>)
<span class="hljs-built_in">print</span>(output)`,wrap:!1}}),be=new ce({props:{title:"Notes",local:"notes",headingTag:"h2"}}),Me=new A({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Nb2RlbEZvckNhdXNhbExNJTBBJTBBbW9kZWwlMjAlM0QlMjBBdXRvTW9kZWxGb3JDYXVzYWxMTS5mcm9tX3ByZXRyYWluZWQoJTBBJTIwJTIwJTIwJTIybWljcm9zb2Z0JTJGYmlvZ3B0JTIyJTJDJTBBJTIwJTIwJTIwYXR0bl9pbXBsZW1lbnRhdGlvbiUzRCUyMmVhZ2VyJTIyJTBBKQ==",highlighted:`<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
   <span class="hljs-string">&quot;microsoft/biogpt&quot;</span>,
   attn_implementation=<span class="hljs-string">&quot;eager&quot;</span>
)`,wrap:!1}}),Te=new ce({props:{title:"BioGptConfig",local:"transformers.BioGptConfig",headingTag:"h2"}}),ve=new X({props:{name:"class transformers.BioGptConfig",anchor:"transformers.BioGptConfig",parameters:[{name:"vocab_size",val:" = 42384"},{name:"hidden_size",val:" = 1024"},{name:"num_hidden_layers",val:" = 24"},{name:"num_attention_heads",val:" = 16"},{name:"intermediate_size",val:" = 4096"},{name:"hidden_act",val:" = 'gelu'"},{name:"hidden_dropout_prob",val:" = 0.1"},{name:"attention_probs_dropout_prob",val:" = 0.1"},{name:"max_position_embeddings",val:" = 1024"},{name:"initializer_range",val:" = 0.02"},{name:"layer_norm_eps",val:" = 1e-12"},{name:"scale_embedding",val:" = True"},{name:"use_cache",val:" = True"},{name:"layerdrop",val:" = 0.0"},{name:"activation_dropout",val:" = 0.0"},{name:"pad_token_id",val:" = 1"},{name:"bos_token_id",val:" = 0"},{name:"eos_token_id",val:" = 2"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.BioGptConfig.vocab_size",description:`<strong>vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 42384) &#x2014;
Vocabulary size of the BioGPT model. Defines the number of different tokens that can be represented by the
<code>inputs_ids</code> passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/biogpt#transformers.BioGptModel">BioGptModel</a>.`,name:"vocab_size"},{anchor:"transformers.BioGptConfig.hidden_size",description:`<strong>hidden_size</strong> (<code>int</code>, <em>optional</em>, defaults to 1024) &#x2014;
Dimension of the encoder layers and the pooler layer.`,name:"hidden_size"},{anchor:"transformers.BioGptConfig.num_hidden_layers",description:`<strong>num_hidden_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 24) &#x2014;
Number of hidden layers in the Transformer encoder.`,name:"num_hidden_layers"},{anchor:"transformers.BioGptConfig.num_attention_heads",description:`<strong>num_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 16) &#x2014;
Number of attention heads for each attention layer in the Transformer encoder.`,name:"num_attention_heads"},{anchor:"transformers.BioGptConfig.intermediate_size",description:`<strong>intermediate_size</strong> (<code>int</code>, <em>optional</em>, defaults to 4096) &#x2014;
Dimension of the &#x201C;intermediate&#x201D; (i.e., feed-forward) layer in the Transformer encoder.`,name:"intermediate_size"},{anchor:"transformers.BioGptConfig.hidden_act",description:`<strong>hidden_act</strong> (<code>str</code> or <code>function</code>, <em>optional</em>, defaults to <code>&quot;gelu&quot;</code>) &#x2014;
The non-linear activation function (function or string) in the encoder and pooler. If string, <code>&quot;gelu&quot;</code>,
<code>&quot;relu&quot;</code>, <code>&quot;selu&quot;</code> and <code>&quot;gelu_new&quot;</code> are supported.`,name:"hidden_act"},{anchor:"transformers.BioGptConfig.hidden_dropout_prob",description:`<strong>hidden_dropout_prob</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.`,name:"hidden_dropout_prob"},{anchor:"transformers.BioGptConfig.attention_probs_dropout_prob",description:`<strong>attention_probs_dropout_prob</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout ratio for the attention probabilities.`,name:"attention_probs_dropout_prob"},{anchor:"transformers.BioGptConfig.max_position_embeddings",description:`<strong>max_position_embeddings</strong> (<code>int</code>, <em>optional</em>, defaults to 1024) &#x2014;
The maximum sequence length that this model might ever be used with. Typically set this to something large
just in case (e.g., 512 or 1024 or 2048).`,name:"max_position_embeddings"},{anchor:"transformers.BioGptConfig.initializer_range",description:`<strong>initializer_range</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.`,name:"initializer_range"},{anchor:"transformers.BioGptConfig.layer_norm_eps",description:`<strong>layer_norm_eps</strong> (<code>float</code>, <em>optional</em>, defaults to 1e-12) &#x2014;
The epsilon used by the layer normalization layers.`,name:"layer_norm_eps"},{anchor:"transformers.BioGptConfig.scale_embedding",description:`<strong>scale_embedding</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Scale embeddings by diving by sqrt(d_model).`,name:"scale_embedding"},{anchor:"transformers.BioGptConfig.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not the model should return the last key/values attentions (not used by all models). Only
relevant if <code>config.is_decoder=True</code>.`,name:"use_cache"},{anchor:"transformers.BioGptConfig.layerdrop",description:`<strong>layerdrop</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
Please refer to the paper about LayerDrop: <a href="https://huggingface.co/papers/1909.11556" rel="nofollow">https://huggingface.co/papers/1909.11556</a> for further details`,name:"layerdrop"},{anchor:"transformers.BioGptConfig.activation_dropout",description:`<strong>activation_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The dropout ratio for activations inside the fully connected layer.`,name:"activation_dropout"},{anchor:"transformers.BioGptConfig.pad_token_id",description:`<strong>pad_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 1) &#x2014;
Padding token id.`,name:"pad_token_id"},{anchor:"transformers.BioGptConfig.bos_token_id",description:`<strong>bos_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 0) &#x2014;
Beginning of stream token id.`,name:"bos_token_id"},{anchor:"transformers.BioGptConfig.eos_token_id",description:`<strong>eos_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 2) &#x2014;
End of stream token id.`,name:"eos_token_id"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/biogpt/configuration_biogpt.py#L24"}}),te=new gt({props:{anchor:"transformers.BioGptConfig.example",$$slots:{default:[yn]},$$scope:{ctx:k}}}),ke=new ce({props:{title:"BioGptTokenizer",local:"transformers.BioGptTokenizer",headingTag:"h2"}}),we=new X({props:{name:"class transformers.BioGptTokenizer",anchor:"transformers.BioGptTokenizer",parameters:[{name:"vocab_file",val:""},{name:"merges_file",val:""},{name:"unk_token",val:" = '<unk>'"},{name:"bos_token",val:" = '<s>'"},{name:"eos_token",val:" = '</s>'"},{name:"sep_token",val:" = '</s>'"},{name:"pad_token",val:" = '<pad>'"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.BioGptTokenizer.vocab_file",description:`<strong>vocab_file</strong> (<code>str</code>) &#x2014;
Path to the vocabulary file.`,name:"vocab_file"},{anchor:"transformers.BioGptTokenizer.merges_file",description:`<strong>merges_file</strong> (<code>str</code>) &#x2014;
Merges file.`,name:"merges_file"},{anchor:"transformers.BioGptTokenizer.unk_token",description:`<strong>unk_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;unk&gt;&quot;</code>) &#x2014;
The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
token instead.`,name:"unk_token"},{anchor:"transformers.BioGptTokenizer.bos_token",description:`<strong>bos_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;s&gt;&quot;</code>) &#x2014;
The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.</p>
<div class="course-tip  bg-gradient-to-br dark:bg-gradient-to-r before:border-green-500 dark:before:border-green-800 from-green-50 dark:from-gray-900 to-white dark:to-gray-950 border border-green-50 text-green-700 dark:text-gray-400">
						
<p>When building a sequence using special tokens, this is not the token that is used for the beginning of
sequence. The token used is the <code>cls_token</code>.</p>

					</div>`,name:"bos_token"},{anchor:"transformers.BioGptTokenizer.eos_token",description:`<strong>eos_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;/s&gt;&quot;</code>) &#x2014;
The end of sequence token.</p>
<div class="course-tip  bg-gradient-to-br dark:bg-gradient-to-r before:border-green-500 dark:before:border-green-800 from-green-50 dark:from-gray-900 to-white dark:to-gray-950 border border-green-50 text-green-700 dark:text-gray-400">
						
<p>When building a sequence using special tokens, this is not the token that is used for the end of sequence.
The token used is the <code>sep_token</code>.</p>

					</div>`,name:"eos_token"},{anchor:"transformers.BioGptTokenizer.sep_token",description:`<strong>sep_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;/s&gt;&quot;</code>) &#x2014;
The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
sequence classification or for a text and a question for question answering. It is also used as the last
token of a sequence built with special tokens.`,name:"sep_token"},{anchor:"transformers.BioGptTokenizer.pad_token",description:`<strong>pad_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;pad&gt;&quot;</code>) &#x2014;
The token used for padding, for example when batching sequences of different lengths.`,name:"pad_token"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/biogpt/tokenization_biogpt.py#L46"}}),Ce=new X({props:{name:"save_vocabulary",anchor:"transformers.BioGptTokenizer.save_vocabulary",parameters:[{name:"save_directory",val:": str"},{name:"filename_prefix",val:": typing.Optional[str] = None"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/biogpt/tokenization_biogpt.py#L284"}}),$e=new ce({props:{title:"BioGptModel",local:"transformers.BioGptModel",headingTag:"h2"}}),Je=new X({props:{name:"class transformers.BioGptModel",anchor:"transformers.BioGptModel",parameters:[{name:"config",val:": BioGptConfig"}],parametersDescription:[{anchor:"transformers.BioGptModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/biogpt#transformers.BioGptConfig">BioGptConfig</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/biogpt/modeling_biogpt.py#L490"}}),Be=new X({props:{name:"forward",anchor:"transformers.BioGptModel.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"past_key_values",val:": typing.Optional[tuple[tuple[torch.Tensor]]] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.Tensor] = None"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.BioGptModel.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.BioGptModel.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.BioGptModel.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.BioGptModel.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.BioGptModel.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>tuple[tuple[torch.Tensor]]</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.BioGptModel.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.BioGptModel.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.BioGptModel.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.BioGptModel.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.BioGptModel.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.BioGptModel.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.Tensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/biogpt/modeling_biogpt.py#L512",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPastAndCrossAttentions"
>transformers.modeling_outputs.BaseModelOutputWithPastAndCrossAttentions</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/biogpt#transformers.BioGptConfig"
>BioGptConfig</a>) and inputs.</p>
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
`}}),oe=new ft({props:{$$slots:{default:[Mn]},$$scope:{ctx:k}}}),Ge=new ce({props:{title:"BioGptForCausalLM",local:"transformers.BioGptForCausalLM",headingTag:"h2"}}),je=new X({props:{name:"class transformers.BioGptForCausalLM",anchor:"transformers.BioGptForCausalLM",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.BioGptForCausalLM.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/biogpt#transformers.BioGptForCausalLM">BioGptForCausalLM</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/biogpt/modeling_biogpt.py#L665"}}),Ue=new X({props:{name:"forward",anchor:"transformers.BioGptForCausalLM.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"past_key_values",val:": typing.Optional[tuple[tuple[torch.Tensor]]] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.Tensor] = None"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.BioGptForCausalLM.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.BioGptForCausalLM.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.BioGptForCausalLM.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.BioGptForCausalLM.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.BioGptForCausalLM.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>tuple[tuple[torch.Tensor]]</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.BioGptForCausalLM.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for language modeling. Note that the labels <strong>are shifted</strong> inside the model, i.e. you can set
<code>labels = input_ids</code> Indices are selected in <code>[-100, 0, ..., config.vocab_size]</code> All labels set to <code>-100</code>
are ignored (masked), the loss is only computed for labels in <code>[0, ..., config.vocab_size]</code>`,name:"labels"},{anchor:"transformers.BioGptForCausalLM.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.BioGptForCausalLM.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.BioGptForCausalLM.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.BioGptForCausalLM.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.BioGptForCausalLM.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.BioGptForCausalLM.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.Tensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/biogpt/modeling_biogpt.py#L683",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithCrossAttentions"
>transformers.modeling_outputs.CausalLMOutputWithCrossAttentions</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/biogpt#transformers.BioGptConfig"
>BioGptConfig</a>) and inputs.</p>
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
`}}),ne=new ft({props:{$$slots:{default:[Tn]},$$scope:{ctx:k}}}),se=new gt({props:{anchor:"transformers.BioGptForCausalLM.forward.example",$$slots:{default:[vn]},$$scope:{ctx:k}}}),xe=new ce({props:{title:"BioGptForTokenClassification",local:"transformers.BioGptForTokenClassification",headingTag:"h2"}}),ze=new X({props:{name:"class transformers.BioGptForTokenClassification",anchor:"transformers.BioGptForTokenClassification",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.BioGptForTokenClassification.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/biogpt#transformers.BioGptForTokenClassification">BioGptForTokenClassification</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/biogpt/modeling_biogpt.py#L750"}}),Ze=new X({props:{name:"forward",anchor:"transformers.BioGptForTokenClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"past_key_values",val:": typing.Optional[tuple[tuple[torch.Tensor]]] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.Tensor] = None"}],parametersDescription:[{anchor:"transformers.BioGptForTokenClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.BioGptForTokenClassification.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.BioGptForTokenClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.BioGptForTokenClassification.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.BioGptForTokenClassification.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>tuple[tuple[torch.Tensor]]</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.BioGptForTokenClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.BioGptForTokenClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the sequence classification/regression loss. Indices should be in <code>[0, ..., config.num_labels - 1]</code>. If <code>config.num_labels == 1</code> a regression loss is computed (Mean-Square loss), If
<code>config.num_labels &gt; 1</code> a classification loss is computed (Cross-Entropy).`,name:"labels"},{anchor:"transformers.BioGptForTokenClassification.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.BioGptForTokenClassification.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.BioGptForTokenClassification.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.BioGptForTokenClassification.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.BioGptForTokenClassification.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.BioGptForTokenClassification.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.Tensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/biogpt/modeling_biogpt.py#L765",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput"
>transformers.modeling_outputs.TokenClassifierOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/biogpt#transformers.BioGptConfig"
>BioGptConfig</a>) and inputs.</p>
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
`}}),ae=new ft({props:{$$slots:{default:[kn]},$$scope:{ctx:k}}}),re=new gt({props:{anchor:"transformers.BioGptForTokenClassification.forward.example",$$slots:{default:[wn]},$$scope:{ctx:k}}}),We=new ce({props:{title:"BioGptForSequenceClassification",local:"transformers.BioGptForSequenceClassification",headingTag:"h2"}}),Fe=new X({props:{name:"class transformers.BioGptForSequenceClassification",anchor:"transformers.BioGptForSequenceClassification",parameters:[{name:"config",val:": BioGptConfig"}],parametersDescription:[{anchor:"transformers.BioGptForSequenceClassification.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/biogpt#transformers.BioGptConfig">BioGptConfig</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/biogpt/modeling_biogpt.py#L848"}}),Ie=new X({props:{name:"forward",anchor:"transformers.BioGptForSequenceClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"past_key_values",val:": typing.Optional[tuple[tuple[torch.Tensor]]] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.Tensor] = None"}],parametersDescription:[{anchor:"transformers.BioGptForSequenceClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.BioGptForSequenceClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.BioGptForSequenceClassification.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.BioGptForSequenceClassification.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>tuple[tuple[torch.Tensor]]</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.BioGptForSequenceClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.BioGptForSequenceClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the sequence classification/regression loss. Indices should be in <code>[0, ..., config.num_labels - 1]</code>. If <code>config.num_labels == 1</code> a regression loss is computed (Mean-Square loss), If
<code>config.num_labels &gt; 1</code> a classification loss is computed (Cross-Entropy).`,name:"labels"},{anchor:"transformers.BioGptForSequenceClassification.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.BioGptForSequenceClassification.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.BioGptForSequenceClassification.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.BioGptForSequenceClassification.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.BioGptForSequenceClassification.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.BioGptForSequenceClassification.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.Tensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/biogpt/modeling_biogpt.py#L858",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <code>transformers.modeling_outputs.SequenceClassifierOutputWithPast</code> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/biogpt#transformers.BioGptConfig"
>BioGptConfig</a>) and inputs.</p>
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
`}}),ie=new ft({props:{$$slots:{default:[Cn]},$$scope:{ctx:k}}}),le=new gt({props:{anchor:"transformers.BioGptForSequenceClassification.forward.example",$$slots:{default:[$n]},$$scope:{ctx:k}}}),de=new gt({props:{anchor:"transformers.BioGptForSequenceClassification.forward.example-2",$$slots:{default:[Jn]},$$scope:{ctx:k}}}),qe=new mn({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/biogpt.md"}}),{c(){t=p("meta"),u=r(),o=p("p"),d=r(),v=p("p"),v.innerHTML=n,h=r(),w=p("div"),w.innerHTML=mt,pe=r(),f(Q.$$.fragment),_t=r(),me=p("p"),me.innerHTML=Go,bt=r(),ue=p("p"),ue.innerHTML=jo,yt=r(),f(K.$$.fragment),Mt=r(),he=p("p"),he.innerHTML=Uo,Tt=r(),f(ee.$$.fragment),vt=r(),fe=p("p"),fe.innerHTML=xo,kt=r(),ge=p("p"),ge.innerHTML=zo,wt=r(),f(_e.$$.fragment),Ct=r(),f(be.$$.fragment),$t=r(),S=p("ul"),Re=p("li"),Re.innerHTML=Zo,Vt=r(),He=p("li"),He.innerHTML=Wo,Lt=r(),ye=p("li"),Ve=p("p"),Ve.innerHTML=Fo,Et=r(),f(Me.$$.fragment),Jt=r(),f(Te.$$.fragment),Bt=r(),x=p("div"),f(ve.$$.fragment),Xt=r(),Le=p("p"),Le.innerHTML=Io,Qt=r(),Ee=p("p"),Ee.innerHTML=qo,St=r(),f(te.$$.fragment),Gt=r(),f(ke.$$.fragment),jt=r(),z=p("div"),f(we.$$.fragment),Pt=r(),Xe=p("p"),Xe.textContent=No,Yt=r(),Qe=p("p"),Qe.innerHTML=Ro,At=r(),Se=p("div"),f(Ce.$$.fragment),Ut=r(),f($e.$$.fragment),xt=r(),$=p("div"),f(Je.$$.fragment),Ot=r(),Pe=p("p"),Pe.textContent=Ho,Dt=r(),Ye=p("p"),Ye.innerHTML=Vo,Kt=r(),Ae=p("p"),Ae.innerHTML=Lo,eo=r(),P=p("div"),f(Be.$$.fragment),to=r(),Oe=p("p"),Oe.innerHTML=Eo,oo=r(),f(oe.$$.fragment),zt=r(),f(Ge.$$.fragment),Zt=r(),J=p("div"),f(je.$$.fragment),no=r(),De=p("p"),De.innerHTML=Xo,so=r(),Ke=p("p"),Ke.innerHTML=Qo,ao=r(),et=p("p"),et.innerHTML=So,ro=r(),N=p("div"),f(Ue.$$.fragment),io=r(),tt=p("p"),tt.innerHTML=Po,lo=r(),f(ne.$$.fragment),co=r(),f(se.$$.fragment),Wt=r(),f(xe.$$.fragment),Ft=r(),B=p("div"),f(ze.$$.fragment),po=r(),ot=p("p"),ot.textContent=Yo,mo=r(),nt=p("p"),nt.innerHTML=Ao,uo=r(),st=p("p"),st.innerHTML=Oo,ho=r(),R=p("div"),f(Ze.$$.fragment),fo=r(),at=p("p"),at.innerHTML=Do,go=r(),f(ae.$$.fragment),_o=r(),f(re.$$.fragment),It=r(),f(We.$$.fragment),qt=r(),C=p("div"),f(Fe.$$.fragment),bo=r(),rt=p("p"),rt.textContent=Ko,yo=r(),it=p("p"),it.innerHTML=en,Mo=r(),lt=p("p"),lt.innerHTML=tn,To=r(),dt=p("p"),dt.innerHTML=on,vo=r(),ct=p("p"),ct.innerHTML=nn,ko=r(),j=p("div"),f(Ie.$$.fragment),wo=r(),pt=p("p"),pt.innerHTML=sn,Co=r(),f(ie.$$.fragment),$o=r(),f(le.$$.fragment),Jo=r(),f(de.$$.fragment),Nt=r(),f(qe.$$.fragment),Rt=r(),ut=p("p"),this.h()},l(e){const s=cn("svelte-u9bgzb",document.head);t=m(s,"META",{name:!0,content:!0}),s.forEach(a),u=i(e),o=m(e,"P",{}),G(o).forEach(a),d=i(e),v=m(e,"P",{"data-svelte-h":!0}),T(v)!=="svelte-1xdjm4t"&&(v.innerHTML=n),h=i(e),w=m(e,"DIV",{style:!0,"data-svelte-h":!0}),T(w)!=="svelte-8nzrvr"&&(w.innerHTML=mt),pe=i(e),g(Q.$$.fragment,e),_t=i(e),me=m(e,"P",{"data-svelte-h":!0}),T(me)!=="svelte-1vngzva"&&(me.innerHTML=Go),bt=i(e),ue=m(e,"P",{"data-svelte-h":!0}),T(ue)!=="svelte-1pcfjhm"&&(ue.innerHTML=jo),yt=i(e),g(K.$$.fragment,e),Mt=i(e),he=m(e,"P",{"data-svelte-h":!0}),T(he)!=="svelte-yqvg95"&&(he.innerHTML=Uo),Tt=i(e),g(ee.$$.fragment,e),vt=i(e),fe=m(e,"P",{"data-svelte-h":!0}),T(fe)!=="svelte-nf5ooi"&&(fe.innerHTML=xo),kt=i(e),ge=m(e,"P",{"data-svelte-h":!0}),T(ge)!=="svelte-iz29bh"&&(ge.innerHTML=zo),wt=i(e),g(_e.$$.fragment,e),Ct=i(e),g(be.$$.fragment,e),$t=i(e),S=m(e,"UL",{});var O=G(S);Re=m(O,"LI",{"data-svelte-h":!0}),T(Re)!=="svelte-1l7abey"&&(Re.innerHTML=Zo),Vt=i(O),He=m(O,"LI",{"data-svelte-h":!0}),T(He)!=="svelte-kw2ffe"&&(He.innerHTML=Wo),Lt=i(O),ye=m(O,"LI",{});var Ne=G(ye);Ve=m(Ne,"P",{"data-svelte-h":!0}),T(Ve)!=="svelte-12n88bn"&&(Ve.innerHTML=Fo),Et=i(Ne),g(Me.$$.fragment,Ne),Ne.forEach(a),O.forEach(a),Jt=i(e),g(Te.$$.fragment,e),Bt=i(e),x=m(e,"DIV",{class:!0});var H=G(x);g(ve.$$.fragment,H),Xt=i(H),Le=m(H,"P",{"data-svelte-h":!0}),T(Le)!=="svelte-tvq66a"&&(Le.innerHTML=Io),Qt=i(H),Ee=m(H,"P",{"data-svelte-h":!0}),T(Ee)!=="svelte-1ek1ss9"&&(Ee.innerHTML=qo),St=i(H),g(te.$$.fragment,H),H.forEach(a),Gt=i(e),g(ke.$$.fragment,e),jt=i(e),z=m(e,"DIV",{class:!0});var V=G(z);g(we.$$.fragment,V),Pt=i(V),Xe=m(V,"P",{"data-svelte-h":!0}),T(Xe)!=="svelte-rkichk"&&(Xe.textContent=No),Yt=i(V),Qe=m(V,"P",{"data-svelte-h":!0}),T(Qe)!=="svelte-ntrhio"&&(Qe.innerHTML=Ro),At=i(V),Se=m(V,"DIV",{class:!0});var ht=G(Se);g(Ce.$$.fragment,ht),ht.forEach(a),V.forEach(a),Ut=i(e),g($e.$$.fragment,e),xt=i(e),$=m(e,"DIV",{class:!0});var Z=G($);g(Je.$$.fragment,Z),Ot=i(Z),Pe=m(Z,"P",{"data-svelte-h":!0}),T(Pe)!=="svelte-rvv9fr"&&(Pe.textContent=Ho),Dt=i(Z),Ye=m(Z,"P",{"data-svelte-h":!0}),T(Ye)!=="svelte-q52n56"&&(Ye.innerHTML=Vo),Kt=i(Z),Ae=m(Z,"P",{"data-svelte-h":!0}),T(Ae)!=="svelte-hswkmf"&&(Ae.innerHTML=Lo),eo=i(Z),P=m(Z,"DIV",{class:!0});var D=G(P);g(Be.$$.fragment,D),to=i(D),Oe=m(D,"P",{"data-svelte-h":!0}),T(Oe)!=="svelte-m18d18"&&(Oe.innerHTML=Eo),oo=i(D),g(oe.$$.fragment,D),D.forEach(a),Z.forEach(a),zt=i(e),g(Ge.$$.fragment,e),Zt=i(e),J=m(e,"DIV",{class:!0});var W=G(J);g(je.$$.fragment,W),no=i(W),De=m(W,"P",{"data-svelte-h":!0}),T(De)!=="svelte-3ek0bq"&&(De.innerHTML=Xo),so=i(W),Ke=m(W,"P",{"data-svelte-h":!0}),T(Ke)!=="svelte-q52n56"&&(Ke.innerHTML=Qo),ao=i(W),et=m(W,"P",{"data-svelte-h":!0}),T(et)!=="svelte-hswkmf"&&(et.innerHTML=So),ro=i(W),N=m(W,"DIV",{class:!0});var L=G(N);g(Ue.$$.fragment,L),io=i(L),tt=m(L,"P",{"data-svelte-h":!0}),T(tt)!=="svelte-1rb86bo"&&(tt.innerHTML=Po),lo=i(L),g(ne.$$.fragment,L),co=i(L),g(se.$$.fragment,L),L.forEach(a),W.forEach(a),Wt=i(e),g(xe.$$.fragment,e),Ft=i(e),B=m(e,"DIV",{class:!0});var F=G(B);g(ze.$$.fragment,F),po=i(F),ot=m(F,"P",{"data-svelte-h":!0}),T(ot)!=="svelte-1hhuutx"&&(ot.textContent=Yo),mo=i(F),nt=m(F,"P",{"data-svelte-h":!0}),T(nt)!=="svelte-q52n56"&&(nt.innerHTML=Ao),uo=i(F),st=m(F,"P",{"data-svelte-h":!0}),T(st)!=="svelte-hswkmf"&&(st.innerHTML=Oo),ho=i(F),R=m(F,"DIV",{class:!0});var E=G(R);g(Ze.$$.fragment,E),fo=i(E),at=m(E,"P",{"data-svelte-h":!0}),T(at)!=="svelte-e43b6a"&&(at.innerHTML=Do),go=i(E),g(ae.$$.fragment,E),_o=i(E),g(re.$$.fragment,E),E.forEach(a),F.forEach(a),It=i(e),g(We.$$.fragment,e),qt=i(e),C=m(e,"DIV",{class:!0});var U=G(C);g(Fe.$$.fragment,U),bo=i(U),rt=m(U,"P",{"data-svelte-h":!0}),T(rt)!=="svelte-1u6ikyr"&&(rt.textContent=Ko),yo=i(U),it=m(U,"P",{"data-svelte-h":!0}),T(it)!=="svelte-8rs2bd"&&(it.innerHTML=en),Mo=i(U),lt=m(U,"P",{"data-svelte-h":!0}),T(lt)!=="svelte-wtyfap"&&(lt.innerHTML=tn),To=i(U),dt=m(U,"P",{"data-svelte-h":!0}),T(dt)!=="svelte-q52n56"&&(dt.innerHTML=on),vo=i(U),ct=m(U,"P",{"data-svelte-h":!0}),T(ct)!=="svelte-hswkmf"&&(ct.innerHTML=nn),ko=i(U),j=m(U,"DIV",{class:!0});var Y=G(j);g(Ie.$$.fragment,Y),wo=i(Y),pt=m(Y,"P",{"data-svelte-h":!0}),T(pt)!=="svelte-s2t234"&&(pt.innerHTML=sn),Co=i(Y),g(ie.$$.fragment,Y),$o=i(Y),g(le.$$.fragment,Y),Jo=i(Y),g(de.$$.fragment,Y),Y.forEach(a),U.forEach(a),Nt=i(e),g(qe.$$.fragment,e),Rt=i(e),ut=m(e,"P",{}),G(ut).forEach(a),this.h()},h(){I(t,"name","hf:doc:metadata"),I(t,"content",Gn),pn(w,"float","right"),I(x,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),I(Se,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),I(z,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),I(P,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),I($,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),I(N,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),I(J,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),I(R,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),I(B,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),I(j,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),I(C,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,s){l(document.head,t),c(e,u,s),c(e,o,s),c(e,d,s),c(e,v,s),c(e,h,s),c(e,w,s),c(e,pe,s),_(Q,e,s),c(e,_t,s),c(e,me,s),c(e,bt,s),c(e,ue,s),c(e,yt,s),_(K,e,s),c(e,Mt,s),c(e,he,s),c(e,Tt,s),_(ee,e,s),c(e,vt,s),c(e,fe,s),c(e,kt,s),c(e,ge,s),c(e,wt,s),_(_e,e,s),c(e,Ct,s),_(be,e,s),c(e,$t,s),c(e,S,s),l(S,Re),l(S,Vt),l(S,He),l(S,Lt),l(S,ye),l(ye,Ve),l(ye,Et),_(Me,ye,null),c(e,Jt,s),_(Te,e,s),c(e,Bt,s),c(e,x,s),_(ve,x,null),l(x,Xt),l(x,Le),l(x,Qt),l(x,Ee),l(x,St),_(te,x,null),c(e,Gt,s),_(ke,e,s),c(e,jt,s),c(e,z,s),_(we,z,null),l(z,Pt),l(z,Xe),l(z,Yt),l(z,Qe),l(z,At),l(z,Se),_(Ce,Se,null),c(e,Ut,s),_($e,e,s),c(e,xt,s),c(e,$,s),_(Je,$,null),l($,Ot),l($,Pe),l($,Dt),l($,Ye),l($,Kt),l($,Ae),l($,eo),l($,P),_(Be,P,null),l(P,to),l(P,Oe),l(P,oo),_(oe,P,null),c(e,zt,s),_(Ge,e,s),c(e,Zt,s),c(e,J,s),_(je,J,null),l(J,no),l(J,De),l(J,so),l(J,Ke),l(J,ao),l(J,et),l(J,ro),l(J,N),_(Ue,N,null),l(N,io),l(N,tt),l(N,lo),_(ne,N,null),l(N,co),_(se,N,null),c(e,Wt,s),_(xe,e,s),c(e,Ft,s),c(e,B,s),_(ze,B,null),l(B,po),l(B,ot),l(B,mo),l(B,nt),l(B,uo),l(B,st),l(B,ho),l(B,R),_(Ze,R,null),l(R,fo),l(R,at),l(R,go),_(ae,R,null),l(R,_o),_(re,R,null),c(e,It,s),_(We,e,s),c(e,qt,s),c(e,C,s),_(Fe,C,null),l(C,bo),l(C,rt),l(C,yo),l(C,it),l(C,Mo),l(C,lt),l(C,To),l(C,dt),l(C,vo),l(C,ct),l(C,ko),l(C,j),_(Ie,j,null),l(j,wo),l(j,pt),l(j,Co),_(ie,j,null),l(j,$o),_(le,j,null),l(j,Jo),_(de,j,null),c(e,Nt,s),_(qe,e,s),c(e,Rt,s),c(e,ut,s),Ht=!0},p(e,[s]){const O={};s&2&&(O.$$scope={dirty:s,ctx:e}),K.$set(O);const Ne={};s&2&&(Ne.$$scope={dirty:s,ctx:e}),ee.$set(Ne);const H={};s&2&&(H.$$scope={dirty:s,ctx:e}),te.$set(H);const V={};s&2&&(V.$$scope={dirty:s,ctx:e}),oe.$set(V);const ht={};s&2&&(ht.$$scope={dirty:s,ctx:e}),ne.$set(ht);const Z={};s&2&&(Z.$$scope={dirty:s,ctx:e}),se.$set(Z);const D={};s&2&&(D.$$scope={dirty:s,ctx:e}),ae.$set(D);const W={};s&2&&(W.$$scope={dirty:s,ctx:e}),re.$set(W);const L={};s&2&&(L.$$scope={dirty:s,ctx:e}),ie.$set(L);const F={};s&2&&(F.$$scope={dirty:s,ctx:e}),le.$set(F);const E={};s&2&&(E.$$scope={dirty:s,ctx:e}),de.$set(E)},i(e){Ht||(b(Q.$$.fragment,e),b(K.$$.fragment,e),b(ee.$$.fragment,e),b(_e.$$.fragment,e),b(be.$$.fragment,e),b(Me.$$.fragment,e),b(Te.$$.fragment,e),b(ve.$$.fragment,e),b(te.$$.fragment,e),b(ke.$$.fragment,e),b(we.$$.fragment,e),b(Ce.$$.fragment,e),b($e.$$.fragment,e),b(Je.$$.fragment,e),b(Be.$$.fragment,e),b(oe.$$.fragment,e),b(Ge.$$.fragment,e),b(je.$$.fragment,e),b(Ue.$$.fragment,e),b(ne.$$.fragment,e),b(se.$$.fragment,e),b(xe.$$.fragment,e),b(ze.$$.fragment,e),b(Ze.$$.fragment,e),b(ae.$$.fragment,e),b(re.$$.fragment,e),b(We.$$.fragment,e),b(Fe.$$.fragment,e),b(Ie.$$.fragment,e),b(ie.$$.fragment,e),b(le.$$.fragment,e),b(de.$$.fragment,e),b(qe.$$.fragment,e),Ht=!0)},o(e){y(Q.$$.fragment,e),y(K.$$.fragment,e),y(ee.$$.fragment,e),y(_e.$$.fragment,e),y(be.$$.fragment,e),y(Me.$$.fragment,e),y(Te.$$.fragment,e),y(ve.$$.fragment,e),y(te.$$.fragment,e),y(ke.$$.fragment,e),y(we.$$.fragment,e),y(Ce.$$.fragment,e),y($e.$$.fragment,e),y(Je.$$.fragment,e),y(Be.$$.fragment,e),y(oe.$$.fragment,e),y(Ge.$$.fragment,e),y(je.$$.fragment,e),y(Ue.$$.fragment,e),y(ne.$$.fragment,e),y(se.$$.fragment,e),y(xe.$$.fragment,e),y(ze.$$.fragment,e),y(Ze.$$.fragment,e),y(ae.$$.fragment,e),y(re.$$.fragment,e),y(We.$$.fragment,e),y(Fe.$$.fragment,e),y(Ie.$$.fragment,e),y(ie.$$.fragment,e),y(le.$$.fragment,e),y(de.$$.fragment,e),y(qe.$$.fragment,e),Ht=!1},d(e){e&&(a(u),a(o),a(d),a(v),a(h),a(w),a(pe),a(_t),a(me),a(bt),a(ue),a(yt),a(Mt),a(he),a(Tt),a(vt),a(fe),a(kt),a(ge),a(wt),a(Ct),a($t),a(S),a(Jt),a(Bt),a(x),a(Gt),a(jt),a(z),a(Ut),a(xt),a($),a(zt),a(Zt),a(J),a(Wt),a(Ft),a(B),a(It),a(qt),a(C),a(Nt),a(Rt),a(ut)),a(t),M(Q,e),M(K,e),M(ee,e),M(_e,e),M(be,e),M(Me),M(Te,e),M(ve),M(te),M(ke,e),M(we),M(Ce),M($e,e),M(Je),M(Be),M(oe),M(Ge,e),M(je),M(Ue),M(ne),M(se),M(xe,e),M(ze),M(Ze),M(ae),M(re),M(We,e),M(Fe),M(Ie),M(ie),M(le),M(de),M(qe,e)}}}const Gn='{"title":"BioGPT","local":"biogpt","sections":[{"title":"Notes","local":"notes","sections":[],"depth":2},{"title":"BioGptConfig","local":"transformers.BioGptConfig","sections":[],"depth":2},{"title":"BioGptTokenizer","local":"transformers.BioGptTokenizer","sections":[],"depth":2},{"title":"BioGptModel","local":"transformers.BioGptModel","sections":[],"depth":2},{"title":"BioGptForCausalLM","local":"transformers.BioGptForCausalLM","sections":[],"depth":2},{"title":"BioGptForTokenClassification","local":"transformers.BioGptForTokenClassification","sections":[],"depth":2},{"title":"BioGptForSequenceClassification","local":"transformers.BioGptForSequenceClassification","sections":[],"depth":2}],"depth":1}';function jn(k){return rn(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class Nn extends ln{constructor(t){super(),dn(this,t,jn,Bn,an,{})}}export{Nn as component};
