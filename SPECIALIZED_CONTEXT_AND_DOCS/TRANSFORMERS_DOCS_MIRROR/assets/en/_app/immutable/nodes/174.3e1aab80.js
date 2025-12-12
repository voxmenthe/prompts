import{s as cr,o as pr,n as v}from"../chunks/scheduler.18a86fab.js";import{S as mr,i as hr,g as p,s as a,r as f,A as ur,h as m,f as l,c as i,j as $,x as M,u as g,k as J,l as fr,y as d,a as c,v as _,d as b,t as y,w as T}from"../chunks/index.98837b22.js";import{T as fe}from"../chunks/Tip.77304350.js";import{D as C}from"../chunks/Docstring.a1ef7999.js";import{C as Y}from"../chunks/CodeBlock.8d0c2e8a.js";import{E as ge}from"../chunks/ExampleCodeBlock.8c3ee1f9.js";import{H as V,E as gr}from"../chunks/getInferenceSnippets.06c2775f.js";import{H as _r,a as fs}from"../chunks/HfOption.6641485e.js";function br(w){let t,h='This model was contributed by <a href="https://huggingface.co/nghuyong" rel="nofollow">nghuyong</a>, and the official code can be found in <a href="https://github.com/PaddlePaddle/PaddleNLP" rel="nofollow">PaddleNLP</a> (in PaddlePaddle).',o,r,k="Click on the ERNIE models in the right sidebar for more examples of how to apply ERNIE to different language tasks.";return{c(){t=p("p"),t.innerHTML=h,o=a(),r=p("p"),r.textContent=k},l(n){t=m(n,"P",{"data-svelte-h":!0}),M(t)!=="svelte-1d19o21"&&(t.innerHTML=h),o=i(n),r=m(n,"P",{"data-svelte-h":!0}),M(r)!=="svelte-1hpph2l"&&(r.textContent=k)},m(n,u){c(n,t,u),c(n,o,u),c(n,r,u)},p:v,d(n){n&&(l(t),l(o),l(r))}}}function yr(w){let t,h;return t=new Y({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMHBpcGVsaW5lJTBBJTBBcGlwZWxpbmUlMjAlM0QlMjBwaXBlbGluZSglMEElMjAlMjAlMjAlMjB0YXNrJTNEJTIyZmlsbC1tYXNrJTIyJTJDJTBBJTIwJTIwJTIwJTIwbW9kZWwlM0QlMjJuZ2h1eW9uZyUyRmVybmllLTMuMC14YmFzZS16aCUyMiUwQSklMEElMEFwaXBlbGluZSglMjIlRTUlQjclQjQlRTklQkIlOEUlRTYlOTglQUYlNUJNQVNLJTVEJUU1JTlCJUJEJUU3JTlBJTg0JUU5JUE2JTk2JUU5JTgzJUJEJUUzJTgwJTgyJTIyKQ==",highlighted:`<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> pipeline

pipeline = pipeline(
    task=<span class="hljs-string">&quot;fill-mask&quot;</span>,
    model=<span class="hljs-string">&quot;nghuyong/ernie-3.0-xbase-zh&quot;</span>
)

pipeline(<span class="hljs-string">&quot;巴黎是[MASK]国的首都。&quot;</span>)`,wrap:!1}}),{c(){f(t.$$.fragment)},l(o){g(t.$$.fragment,o)},m(o,r){_(t,o,r),h=!0},p:v,i(o){h||(b(t.$$.fragment,o),h=!0)},o(o){y(t.$$.fragment,o),h=!1},d(o){T(t,o)}}}function Tr(w){let t,h;return t=new Y({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b01vZGVsRm9yTWFza2VkTE0lMkMlMjBBdXRvVG9rZW5pemVyJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTBBJTIwJTIwJTIwJTIwJTIybmdodXlvbmclMkZlcm5pZS0zLjAteGJhc2UtemglMjIlMkMlMEEpJTBBbW9kZWwlMjAlM0QlMjBBdXRvTW9kZWxGb3JNYXNrZWRMTS5mcm9tX3ByZXRyYWluZWQoJTBBJTIwJTIwJTIwJTIwJTIybmdodXlvbmclMkZlcm5pZS0zLjAteGJhc2UtemglMjIlMkMlMEElMjAlMjAlMjAlMjBkdHlwZSUzRHRvcmNoLmZsb2F0MTYlMkMlMEElMjAlMjAlMjAlMjBkZXZpY2VfbWFwJTNEJTIyYXV0byUyMiUwQSklMEFpbnB1dHMlMjAlM0QlMjB0b2tlbml6ZXIoJTIyJUU1JUI3JUI0JUU5JUJCJThFJUU2JTk4JUFGJTVCTUFTSyU1RCVFNSU5QiVCRCVFNyU5QSU4NCVFOSVBNiU5NiVFOSU4MyVCRCVFMyU4MCU4MiUyMiUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpLnRvKG1vZGVsLmRldmljZSklMEElMEF3aXRoJTIwdG9yY2gubm9fZ3JhZCgpJTNBJTBBJTIwJTIwJTIwJTIwb3V0cHV0cyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzKSUwQSUyMCUyMCUyMCUyMHByZWRpY3Rpb25zJTIwJTNEJTIwb3V0cHV0cy5sb2dpdHMlMEElMEFtYXNrZWRfaW5kZXglMjAlM0QlMjB0b3JjaC53aGVyZShpbnB1dHMlNUInaW5wdXRfaWRzJyU1RCUyMCUzRCUzRCUyMHRva2VuaXplci5tYXNrX3Rva2VuX2lkKSU1QjElNUQlMEFwcmVkaWN0ZWRfdG9rZW5faWQlMjAlM0QlMjBwcmVkaWN0aW9ucyU1QjAlMkMlMjBtYXNrZWRfaW5kZXglNUQuYXJnbWF4KGRpbSUzRC0xKSUwQXByZWRpY3RlZF90b2tlbiUyMCUzRCUyMHRva2VuaXplci5kZWNvZGUocHJlZGljdGVkX3Rva2VuX2lkKSUwQSUwQXByaW50KGYlMjJUaGUlMjBwcmVkaWN0ZWQlMjB0b2tlbiUyMGlzJTNBJTIwJTdCcHJlZGljdGVkX3Rva2VuJTdEJTIyKQ==",highlighted:`<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoModelForMaskedLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    <span class="hljs-string">&quot;nghuyong/ernie-3.0-xbase-zh&quot;</span>,
)
model = AutoModelForMaskedLM.from_pretrained(
    <span class="hljs-string">&quot;nghuyong/ernie-3.0-xbase-zh&quot;</span>,
    dtype=torch.float16,
    device_map=<span class="hljs-string">&quot;auto&quot;</span>
)
inputs = tokenizer(<span class="hljs-string">&quot;巴黎是[MASK]国的首都。&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>).to(model.device)

<span class="hljs-keyword">with</span> torch.no_grad():
    outputs = model(**inputs)
    predictions = outputs.logits

masked_index = torch.where(inputs[<span class="hljs-string">&#x27;input_ids&#x27;</span>] == tokenizer.mask_token_id)[<span class="hljs-number">1</span>]
predicted_token_id = predictions[<span class="hljs-number">0</span>, masked_index].argmax(dim=-<span class="hljs-number">1</span>)
predicted_token = tokenizer.decode(predicted_token_id)

<span class="hljs-built_in">print</span>(<span class="hljs-string">f&quot;The predicted token is: <span class="hljs-subst">{predicted_token}</span>&quot;</span>)`,wrap:!1}}),{c(){f(t.$$.fragment)},l(o){g(t.$$.fragment,o)},m(o,r){_(t,o,r),h=!0},p:v,i(o){h||(b(t.$$.fragment,o),h=!0)},o(o){y(t.$$.fragment,o),h=!1},d(o){T(t,o)}}}function Mr(w){let t,h;return t=new Y({props:{code:"ZWNobyUyMC1lJTIwJTIyJUU1JUI3JUI0JUU5JUJCJThFJUU2JTk4JUFGJTVCTUFTSyU1RCVFNSU5QiVCRCVFNyU5QSU4NCVFOSVBNiU5NiVFOSU4MyVCRCVFMyU4MCU4MiUyMiUyMCU3QyUyMHRyYW5zZm9ybWVycyUyMHJ1biUyMC0tdGFzayUyMGZpbGwtbWFzayUyMC0tbW9kZWwlMjBuZ2h1eW9uZyUyRmVybmllLTMuMC14YmFzZS16aCUyMC0tZGV2aWNlJTIwMA==",highlighted:'<span class="hljs-built_in">echo</span> -e <span class="hljs-string">&quot;巴黎是[MASK]国的首都。&quot;</span> | transformers run --task fill-mask --model nghuyong/ernie-3.0-xbase-zh --device 0',wrap:!1}}),{c(){f(t.$$.fragment)},l(o){g(t.$$.fragment,o)},m(o,r){_(t,o,r),h=!0},p:v,i(o){h||(b(t.$$.fragment,o),h=!0)},o(o){y(t.$$.fragment,o),h=!1},d(o){T(t,o)}}}function kr(w){let t,h,o,r,k,n;return t=new fs({props:{id:"usage",option:"Pipeline",$$slots:{default:[yr]},$$scope:{ctx:w}}}),o=new fs({props:{id:"usage",option:"AutoModel",$$slots:{default:[Tr]},$$scope:{ctx:w}}}),k=new fs({props:{id:"usage",option:"transformers CLI",$$slots:{default:[Mr]},$$scope:{ctx:w}}}),{c(){f(t.$$.fragment),h=a(),f(o.$$.fragment),r=a(),f(k.$$.fragment)},l(u){g(t.$$.fragment,u),h=i(u),g(o.$$.fragment,u),r=i(u),g(k.$$.fragment,u)},m(u,E){_(t,u,E),c(u,h,E),_(o,u,E),c(u,r,E),_(k,u,E),n=!0},p(u,E){const gn={};E&2&&(gn.$$scope={dirty:E,ctx:u}),t.$set(gn);const Ve={};E&2&&(Ve.$$scope={dirty:E,ctx:u}),o.$set(Ve);const he={};E&2&&(he.$$scope={dirty:E,ctx:u}),k.$set(he)},i(u){n||(b(t.$$.fragment,u),b(o.$$.fragment,u),b(k.$$.fragment,u),n=!0)},o(u){y(t.$$.fragment,u),y(o.$$.fragment,u),y(k.$$.fragment,u),n=!1},d(u){u&&(l(h),l(r)),T(t,u),T(o,u),T(k,u)}}}function wr(w){let t,h="Examples:",o,r,k;return r=new Y({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEVybmllQ29uZmlnJTJDJTIwRXJuaWVNb2RlbCUwQSUwQSUyMyUyMEluaXRpYWxpemluZyUyMGElMjBFUk5JRSUyMG5naHV5b25nJTJGZXJuaWUtMy4wLWJhc2UtemglMjBzdHlsZSUyMGNvbmZpZ3VyYXRpb24lMEFjb25maWd1cmF0aW9uJTIwJTNEJTIwRXJuaWVDb25maWcoKSUwQSUwQSUyMyUyMEluaXRpYWxpemluZyUyMGElMjBtb2RlbCUyMCh3aXRoJTIwcmFuZG9tJTIwd2VpZ2h0cyklMjBmcm9tJTIwdGhlJTIwbmdodXlvbmclMkZlcm5pZS0zLjAtYmFzZS16aCUyMHN0eWxlJTIwY29uZmlndXJhdGlvbiUwQW1vZGVsJTIwJTNEJTIwRXJuaWVNb2RlbChjb25maWd1cmF0aW9uKSUwQSUwQSUyMyUyMEFjY2Vzc2luZyUyMHRoZSUyMG1vZGVsJTIwY29uZmlndXJhdGlvbiUwQWNvbmZpZ3VyYXRpb24lMjAlM0QlMjBtb2RlbC5jb25maWc=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> ErnieConfig, ErnieModel

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a ERNIE nghuyong/ernie-3.0-base-zh style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = ErnieConfig()

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a model (with random weights) from the nghuyong/ernie-3.0-base-zh style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = ErnieModel(configuration)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Accessing the model configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = model.config`,wrap:!1}}),{c(){t=p("p"),t.textContent=h,o=a(),f(r.$$.fragment)},l(n){t=m(n,"P",{"data-svelte-h":!0}),M(t)!=="svelte-kvfsh7"&&(t.textContent=h),o=i(n),g(r.$$.fragment,n)},m(n,u){c(n,t,u),c(n,o,u),_(r,n,u),k=!0},p:v,i(n){k||(b(r.$$.fragment,n),k=!0)},o(n){y(r.$$.fragment,n),k=!1},d(n){n&&(l(t),l(o)),T(r,n)}}}function vr(w){let t,h=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=p("p"),t.innerHTML=h},l(o){t=m(o,"P",{"data-svelte-h":!0}),M(t)!=="svelte-fincs2"&&(t.innerHTML=h)},m(o,r){c(o,t,r)},p:v,d(o){o&&l(t)}}}function $r(w){let t,h=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=p("p"),t.innerHTML=h},l(o){t=m(o,"P",{"data-svelte-h":!0}),M(t)!=="svelte-fincs2"&&(t.innerHTML=h)},m(o,r){c(o,t,r)},p:v,d(o){o&&l(t)}}}function Jr(w){let t,h="Example:",o,r,k;return r=new Y({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBFcm5pZUZvclByZVRyYWluaW5nJTBBaW1wb3J0JTIwdG9yY2glMEElMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJuZ2h1eW9uZyUyRmVybmllLTEuMC1iYXNlLXpoJTIyKSUwQW1vZGVsJTIwJTNEJTIwRXJuaWVGb3JQcmVUcmFpbmluZy5mcm9tX3ByZXRyYWluZWQoJTIybmdodXlvbmclMkZlcm5pZS0xLjAtYmFzZS16aCUyMiklMEElMEFpbnB1dHMlMjAlM0QlMjB0b2tlbml6ZXIoJTIySGVsbG8lMkMlMjBteSUyMGRvZyUyMGlzJTIwY3V0ZSUyMiUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpJTBBb3V0cHV0cyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzKSUwQSUwQXByZWRpY3Rpb25fbG9naXRzJTIwJTNEJTIwb3V0cHV0cy5wcmVkaWN0aW9uX2xvZ2l0cyUwQXNlcV9yZWxhdGlvbnNoaXBfbG9naXRzJTIwJTNEJTIwb3V0cHV0cy5zZXFfcmVsYXRpb25zaGlwX2xvZ2l0cw==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, ErnieForPreTraining
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;nghuyong/ernie-1.0-base-zh&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = ErnieForPreTraining.from_pretrained(<span class="hljs-string">&quot;nghuyong/ernie-1.0-base-zh&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs)

<span class="hljs-meta">&gt;&gt;&gt; </span>prediction_logits = outputs.prediction_logits
<span class="hljs-meta">&gt;&gt;&gt; </span>seq_relationship_logits = outputs.seq_relationship_logits`,wrap:!1}}),{c(){t=p("p"),t.textContent=h,o=a(),f(r.$$.fragment)},l(n){t=m(n,"P",{"data-svelte-h":!0}),M(t)!=="svelte-11lpom8"&&(t.textContent=h),o=i(n),g(r.$$.fragment,n)},m(n,u){c(n,t,u),c(n,o,u),_(r,n,u),k=!0},p:v,i(n){k||(b(r.$$.fragment,n),k=!0)},o(n){y(r.$$.fragment,n),k=!1},d(n){n&&(l(t),l(o)),T(r,n)}}}function Er(w){let t,h=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=p("p"),t.innerHTML=h},l(o){t=m(o,"P",{"data-svelte-h":!0}),M(t)!=="svelte-fincs2"&&(t.innerHTML=h)},m(o,r){c(o,t,r)},p:v,d(o){o&&l(t)}}}function Cr(w){let t,h="Example:",o,r,k;return r=new Y({props:{code:"",highlighted:"",wrap:!1}}),{c(){t=p("p"),t.textContent=h,o=a(),f(r.$$.fragment)},l(n){t=m(n,"P",{"data-svelte-h":!0}),M(t)!=="svelte-11lpom8"&&(t.textContent=h),o=i(n),g(r.$$.fragment,n)},m(n,u){c(n,t,u),c(n,o,u),_(r,n,u),k=!0},p:v,i(n){k||(b(r.$$.fragment,n),k=!0)},o(n){y(r.$$.fragment,n),k=!1},d(n){n&&(l(t),l(o)),T(r,n)}}}function jr(w){let t,h=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=p("p"),t.innerHTML=h},l(o){t=m(o,"P",{"data-svelte-h":!0}),M(t)!=="svelte-fincs2"&&(t.innerHTML=h)},m(o,r){c(o,t,r)},p:v,d(o){o&&l(t)}}}function zr(w){let t,h="Example:",o,r,k;return r=new Y({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBFcm5pZUZvck1hc2tlZExNJTBBaW1wb3J0JTIwdG9yY2glMEElMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJuZ2h1eW9uZyUyRmVybmllLTMuMC1iYXNlLXpoJTIyKSUwQW1vZGVsJTIwJTNEJTIwRXJuaWVGb3JNYXNrZWRMTS5mcm9tX3ByZXRyYWluZWQoJTIybmdodXlvbmclMkZlcm5pZS0zLjAtYmFzZS16aCUyMiklMEElMEFpbnB1dHMlMjAlM0QlMjB0b2tlbml6ZXIoJTIyVGhlJTIwY2FwaXRhbCUyMG9mJTIwRnJhbmNlJTIwaXMlMjAlM0NtYXNrJTNFLiUyMiUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpJTBBJTBBd2l0aCUyMHRvcmNoLm5vX2dyYWQoKSUzQSUwQSUyMCUyMCUyMCUyMGxvZ2l0cyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzKS5sb2dpdHMlMEElMEElMjMlMjByZXRyaWV2ZSUyMGluZGV4JTIwb2YlMjAlM0NtYXNrJTNFJTBBbWFza190b2tlbl9pbmRleCUyMCUzRCUyMChpbnB1dHMuaW5wdXRfaWRzJTIwJTNEJTNEJTIwdG9rZW5pemVyLm1hc2tfdG9rZW5faWQpJTVCMCU1RC5ub256ZXJvKGFzX3R1cGxlJTNEVHJ1ZSklNUIwJTVEJTBBJTBBcHJlZGljdGVkX3Rva2VuX2lkJTIwJTNEJTIwbG9naXRzJTVCMCUyQyUyMG1hc2tfdG9rZW5faW5kZXglNUQuYXJnbWF4KGF4aXMlM0QtMSklMEF0b2tlbml6ZXIuZGVjb2RlKHByZWRpY3RlZF90b2tlbl9pZCklMEElMEFsYWJlbHMlMjAlM0QlMjB0b2tlbml6ZXIoJTIyVGhlJTIwY2FwaXRhbCUyMG9mJTIwRnJhbmNlJTIwaXMlMjBQYXJpcy4lMjIlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSU1QiUyMmlucHV0X2lkcyUyMiU1RCUwQSUyMyUyMG1hc2slMjBsYWJlbHMlMjBvZiUyMG5vbi0lM0NtYXNrJTNFJTIwdG9rZW5zJTBBbGFiZWxzJTIwJTNEJTIwdG9yY2gud2hlcmUoaW5wdXRzLmlucHV0X2lkcyUyMCUzRCUzRCUyMHRva2VuaXplci5tYXNrX3Rva2VuX2lkJTJDJTIwbGFiZWxzJTJDJTIwLTEwMCklMEElMEFvdXRwdXRzJTIwJTNEJTIwbW9kZWwoKippbnB1dHMlMkMlMjBsYWJlbHMlM0RsYWJlbHMpJTBBcm91bmQob3V0cHV0cy5sb3NzLml0ZW0oKSUyQyUyMDIp",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, ErnieForMaskedLM
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;nghuyong/ernie-3.0-base-zh&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = ErnieForMaskedLM.from_pretrained(<span class="hljs-string">&quot;nghuyong/ernie-3.0-base-zh&quot;</span>)

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
...`,wrap:!1}}),{c(){t=p("p"),t.textContent=h,o=a(),f(r.$$.fragment)},l(n){t=m(n,"P",{"data-svelte-h":!0}),M(t)!=="svelte-11lpom8"&&(t.textContent=h),o=i(n),g(r.$$.fragment,n)},m(n,u){c(n,t,u),c(n,o,u),_(r,n,u),k=!0},p:v,i(n){k||(b(r.$$.fragment,n),k=!0)},o(n){y(r.$$.fragment,n),k=!1},d(n){n&&(l(t),l(o)),T(r,n)}}}function Ur(w){let t,h=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=p("p"),t.innerHTML=h},l(o){t=m(o,"P",{"data-svelte-h":!0}),M(t)!=="svelte-fincs2"&&(t.innerHTML=h)},m(o,r){c(o,t,r)},p:v,d(o){o&&l(t)}}}function xr(w){let t,h="Example:",o,r,k;return r=new Y({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBFcm5pZUZvck5leHRTZW50ZW5jZVByZWRpY3Rpb24lMEFpbXBvcnQlMjB0b3JjaCUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMm5naHV5b25nJTJGZXJuaWUtMS4wLWJhc2UtemglMjIpJTBBbW9kZWwlMjAlM0QlMjBFcm5pZUZvck5leHRTZW50ZW5jZVByZWRpY3Rpb24uZnJvbV9wcmV0cmFpbmVkKCUyMm5naHV5b25nJTJGZXJuaWUtMS4wLWJhc2UtemglMjIpJTBBJTBBcHJvbXB0JTIwJTNEJTIwJTIySW4lMjBJdGFseSUyQyUyMHBpenphJTIwc2VydmVkJTIwaW4lMjBmb3JtYWwlMjBzZXR0aW5ncyUyQyUyMHN1Y2glMjBhcyUyMGF0JTIwYSUyMHJlc3RhdXJhbnQlMkMlMjBpcyUyMHByZXNlbnRlZCUyMHVuc2xpY2VkLiUyMiUwQW5leHRfc2VudGVuY2UlMjAlM0QlMjAlMjJUaGUlMjBza3klMjBpcyUyMGJsdWUlMjBkdWUlMjB0byUyMHRoZSUyMHNob3J0ZXIlMjB3YXZlbGVuZ3RoJTIwb2YlMjBibHVlJTIwbGlnaHQuJTIyJTBBZW5jb2RpbmclMjAlM0QlMjB0b2tlbml6ZXIocHJvbXB0JTJDJTIwbmV4dF9zZW50ZW5jZSUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpJTBBJTBBb3V0cHV0cyUyMCUzRCUyMG1vZGVsKCoqZW5jb2RpbmclMkMlMjBsYWJlbHMlM0R0b3JjaC5Mb25nVGVuc29yKCU1QjElNUQpKSUwQWxvZ2l0cyUyMCUzRCUyMG91dHB1dHMubG9naXRzJTBBYXNzZXJ0JTIwbG9naXRzJTVCMCUyQyUyMDAlNUQlMjAlM0MlMjBsb2dpdHMlNUIwJTJDJTIwMSU1RCUyMCUyMCUyMyUyMG5leHQlMjBzZW50ZW5jZSUyMHdhcyUyMHJhbmRvbQ==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, ErnieForNextSentencePrediction
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;nghuyong/ernie-1.0-base-zh&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = ErnieForNextSentencePrediction.from_pretrained(<span class="hljs-string">&quot;nghuyong/ernie-1.0-base-zh&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>prompt = <span class="hljs-string">&quot;In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>next_sentence = <span class="hljs-string">&quot;The sky is blue due to the shorter wavelength of blue light.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>encoding = tokenizer(prompt, next_sentence, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**encoding, labels=torch.LongTensor([<span class="hljs-number">1</span>]))
<span class="hljs-meta">&gt;&gt;&gt; </span>logits = outputs.logits
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">assert</span> logits[<span class="hljs-number">0</span>, <span class="hljs-number">0</span>] &lt; logits[<span class="hljs-number">0</span>, <span class="hljs-number">1</span>]  <span class="hljs-comment"># next sentence was random</span>`,wrap:!1}}),{c(){t=p("p"),t.textContent=h,o=a(),f(r.$$.fragment)},l(n){t=m(n,"P",{"data-svelte-h":!0}),M(t)!=="svelte-11lpom8"&&(t.textContent=h),o=i(n),g(r.$$.fragment,n)},m(n,u){c(n,t,u),c(n,o,u),_(r,n,u),k=!0},p:v,i(n){k||(b(r.$$.fragment,n),k=!0)},o(n){y(r.$$.fragment,n),k=!1},d(n){n&&(l(t),l(o)),T(r,n)}}}function Fr(w){let t,h=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=p("p"),t.innerHTML=h},l(o){t=m(o,"P",{"data-svelte-h":!0}),M(t)!=="svelte-fincs2"&&(t.innerHTML=h)},m(o,r){c(o,t,r)},p:v,d(o){o&&l(t)}}}function Ir(w){let t,h="Example of single-label classification:",o,r,k;return r=new Y({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Rva2VuaXplciUyQyUyMEVybmllRm9yU2VxdWVuY2VDbGFzc2lmaWNhdGlvbiUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMm5naHV5b25nJTJGZXJuaWUtMy4wLWJhc2UtemglMjIpJTBBbW9kZWwlMjAlM0QlMjBFcm5pZUZvclNlcXVlbmNlQ2xhc3NpZmljYXRpb24uZnJvbV9wcmV0cmFpbmVkKCUyMm5naHV5b25nJTJGZXJuaWUtMy4wLWJhc2UtemglMjIpJTBBJTBBaW5wdXRzJTIwJTNEJTIwdG9rZW5pemVyKCUyMkhlbGxvJTJDJTIwbXklMjBkb2clMjBpcyUyMGN1dGUlMjIlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQSUwQXdpdGglMjB0b3JjaC5ub19ncmFkKCklM0ElMEElMjAlMjAlMjAlMjBsb2dpdHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cykubG9naXRzJTBBJTBBcHJlZGljdGVkX2NsYXNzX2lkJTIwJTNEJTIwbG9naXRzLmFyZ21heCgpLml0ZW0oKSUwQW1vZGVsLmNvbmZpZy5pZDJsYWJlbCU1QnByZWRpY3RlZF9jbGFzc19pZCU1RCUwQSUwQSUyMyUyMFRvJTIwdHJhaW4lMjBhJTIwbW9kZWwlMjBvbiUyMCU2MG51bV9sYWJlbHMlNjAlMjBjbGFzc2VzJTJDJTIweW91JTIwY2FuJTIwcGFzcyUyMCU2MG51bV9sYWJlbHMlM0RudW1fbGFiZWxzJTYwJTIwdG8lMjAlNjAuZnJvbV9wcmV0cmFpbmVkKC4uLiklNjAlMEFudW1fbGFiZWxzJTIwJTNEJTIwbGVuKG1vZGVsLmNvbmZpZy5pZDJsYWJlbCklMEFtb2RlbCUyMCUzRCUyMEVybmllRm9yU2VxdWVuY2VDbGFzc2lmaWNhdGlvbi5mcm9tX3ByZXRyYWluZWQoJTIybmdodXlvbmclMkZlcm5pZS0zLjAtYmFzZS16aCUyMiUyQyUyMG51bV9sYWJlbHMlM0RudW1fbGFiZWxzKSUwQSUwQWxhYmVscyUyMCUzRCUyMHRvcmNoLnRlbnNvciglNUIxJTVEKSUwQWxvc3MlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyUyQyUyMGxhYmVscyUzRGxhYmVscykubG9zcyUwQXJvdW5kKGxvc3MuaXRlbSgpJTJDJTIwMik=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, ErnieForSequenceClassification

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;nghuyong/ernie-3.0-base-zh&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = ErnieForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;nghuyong/ernie-3.0-base-zh&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    logits = model(**inputs).logits

<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_class_id = logits.argmax().item()
<span class="hljs-meta">&gt;&gt;&gt; </span>model.config.id2label[predicted_class_id]
...

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># To train a model on \`num_labels\` classes, you can pass \`num_labels=num_labels\` to \`.from_pretrained(...)\`</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>num_labels = <span class="hljs-built_in">len</span>(model.config.id2label)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = ErnieForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;nghuyong/ernie-3.0-base-zh&quot;</span>, num_labels=num_labels)

<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.tensor([<span class="hljs-number">1</span>])
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = model(**inputs, labels=labels).loss
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">round</span>(loss.item(), <span class="hljs-number">2</span>)
...`,wrap:!1}}),{c(){t=p("p"),t.textContent=h,o=a(),f(r.$$.fragment)},l(n){t=m(n,"P",{"data-svelte-h":!0}),M(t)!=="svelte-ykxpe4"&&(t.textContent=h),o=i(n),g(r.$$.fragment,n)},m(n,u){c(n,t,u),c(n,o,u),_(r,n,u),k=!0},p:v,i(n){k||(b(r.$$.fragment,n),k=!0)},o(n){y(r.$$.fragment,n),k=!1},d(n){n&&(l(t),l(o)),T(r,n)}}}function Wr(w){let t,h="Example of multi-label classification:",o,r,k;return r=new Y({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Rva2VuaXplciUyQyUyMEVybmllRm9yU2VxdWVuY2VDbGFzc2lmaWNhdGlvbiUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMm5naHV5b25nJTJGZXJuaWUtMy4wLWJhc2UtemglMjIpJTBBbW9kZWwlMjAlM0QlMjBFcm5pZUZvclNlcXVlbmNlQ2xhc3NpZmljYXRpb24uZnJvbV9wcmV0cmFpbmVkKCUyMm5naHV5b25nJTJGZXJuaWUtMy4wLWJhc2UtemglMjIlMkMlMjBwcm9ibGVtX3R5cGUlM0QlMjJtdWx0aV9sYWJlbF9jbGFzc2lmaWNhdGlvbiUyMiklMEElMEFpbnB1dHMlMjAlM0QlMjB0b2tlbml6ZXIoJTIySGVsbG8lMkMlMjBteSUyMGRvZyUyMGlzJTIwY3V0ZSUyMiUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpJTBBJTBBd2l0aCUyMHRvcmNoLm5vX2dyYWQoKSUzQSUwQSUyMCUyMCUyMCUyMGxvZ2l0cyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzKS5sb2dpdHMlMEElMEFwcmVkaWN0ZWRfY2xhc3NfaWRzJTIwJTNEJTIwdG9yY2guYXJhbmdlKDAlMkMlMjBsb2dpdHMuc2hhcGUlNUItMSU1RCklNUJ0b3JjaC5zaWdtb2lkKGxvZ2l0cykuc3F1ZWV6ZShkaW0lM0QwKSUyMCUzRSUyMDAuNSU1RCUwQSUwQSUyMyUyMFRvJTIwdHJhaW4lMjBhJTIwbW9kZWwlMjBvbiUyMCU2MG51bV9sYWJlbHMlNjAlMjBjbGFzc2VzJTJDJTIweW91JTIwY2FuJTIwcGFzcyUyMCU2MG51bV9sYWJlbHMlM0RudW1fbGFiZWxzJTYwJTIwdG8lMjAlNjAuZnJvbV9wcmV0cmFpbmVkKC4uLiklNjAlMEFudW1fbGFiZWxzJTIwJTNEJTIwbGVuKG1vZGVsLmNvbmZpZy5pZDJsYWJlbCklMEFtb2RlbCUyMCUzRCUyMEVybmllRm9yU2VxdWVuY2VDbGFzc2lmaWNhdGlvbi5mcm9tX3ByZXRyYWluZWQoJTBBJTIwJTIwJTIwJTIwJTIybmdodXlvbmclMkZlcm5pZS0zLjAtYmFzZS16aCUyMiUyQyUyMG51bV9sYWJlbHMlM0RudW1fbGFiZWxzJTJDJTIwcHJvYmxlbV90eXBlJTNEJTIybXVsdGlfbGFiZWxfY2xhc3NpZmljYXRpb24lMjIlMEEpJTBBJTBBbGFiZWxzJTIwJTNEJTIwdG9yY2guc3VtKCUwQSUyMCUyMCUyMCUyMHRvcmNoLm5uLmZ1bmN0aW9uYWwub25lX2hvdChwcmVkaWN0ZWRfY2xhc3NfaWRzJTVCTm9uZSUyQyUyMCUzQSU1RC5jbG9uZSgpJTJDJTIwbnVtX2NsYXNzZXMlM0RudW1fbGFiZWxzKSUyQyUyMGRpbSUzRDElMEEpLnRvKHRvcmNoLmZsb2F0KSUwQWxvc3MlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyUyQyUyMGxhYmVscyUzRGxhYmVscykubG9zcw==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, ErnieForSequenceClassification

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;nghuyong/ernie-3.0-base-zh&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = ErnieForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;nghuyong/ernie-3.0-base-zh&quot;</span>, problem_type=<span class="hljs-string">&quot;multi_label_classification&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    logits = model(**inputs).logits

<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_class_ids = torch.arange(<span class="hljs-number">0</span>, logits.shape[-<span class="hljs-number">1</span>])[torch.sigmoid(logits).squeeze(dim=<span class="hljs-number">0</span>) &gt; <span class="hljs-number">0.5</span>]

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># To train a model on \`num_labels\` classes, you can pass \`num_labels=num_labels\` to \`.from_pretrained(...)\`</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>num_labels = <span class="hljs-built_in">len</span>(model.config.id2label)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = ErnieForSequenceClassification.from_pretrained(
<span class="hljs-meta">... </span>    <span class="hljs-string">&quot;nghuyong/ernie-3.0-base-zh&quot;</span>, num_labels=num_labels, problem_type=<span class="hljs-string">&quot;multi_label_classification&quot;</span>
<span class="hljs-meta">... </span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.<span class="hljs-built_in">sum</span>(
<span class="hljs-meta">... </span>    torch.nn.functional.one_hot(predicted_class_ids[<span class="hljs-literal">None</span>, :].clone(), num_classes=num_labels), dim=<span class="hljs-number">1</span>
<span class="hljs-meta">... </span>).to(torch.<span class="hljs-built_in">float</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = model(**inputs, labels=labels).loss`,wrap:!1}}),{c(){t=p("p"),t.textContent=h,o=a(),f(r.$$.fragment)},l(n){t=m(n,"P",{"data-svelte-h":!0}),M(t)!=="svelte-1l8e32d"&&(t.textContent=h),o=i(n),g(r.$$.fragment,n)},m(n,u){c(n,t,u),c(n,o,u),_(r,n,u),k=!0},p:v,i(n){k||(b(r.$$.fragment,n),k=!0)},o(n){y(r.$$.fragment,n),k=!1},d(n){n&&(l(t),l(o)),T(r,n)}}}function Zr(w){let t,h=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=p("p"),t.innerHTML=h},l(o){t=m(o,"P",{"data-svelte-h":!0}),M(t)!=="svelte-fincs2"&&(t.innerHTML=h)},m(o,r){c(o,t,r)},p:v,d(o){o&&l(t)}}}function Nr(w){let t,h="Example:",o,r,k;return r=new Y({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBFcm5pZUZvck11bHRpcGxlQ2hvaWNlJTBBaW1wb3J0JTIwdG9yY2glMEElMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJuZ2h1eW9uZyUyRmVybmllLTMuMC1iYXNlLXpoJTIyKSUwQW1vZGVsJTIwJTNEJTIwRXJuaWVGb3JNdWx0aXBsZUNob2ljZS5mcm9tX3ByZXRyYWluZWQoJTIybmdodXlvbmclMkZlcm5pZS0zLjAtYmFzZS16aCUyMiklMEElMEFwcm9tcHQlMjAlM0QlMjAlMjJJbiUyMEl0YWx5JTJDJTIwcGl6emElMjBzZXJ2ZWQlMjBpbiUyMGZvcm1hbCUyMHNldHRpbmdzJTJDJTIwc3VjaCUyMGFzJTIwYXQlMjBhJTIwcmVzdGF1cmFudCUyQyUyMGlzJTIwcHJlc2VudGVkJTIwdW5zbGljZWQuJTIyJTBBY2hvaWNlMCUyMCUzRCUyMCUyMkl0JTIwaXMlMjBlYXRlbiUyMHdpdGglMjBhJTIwZm9yayUyMGFuZCUyMGElMjBrbmlmZS4lMjIlMEFjaG9pY2UxJTIwJTNEJTIwJTIySXQlMjBpcyUyMGVhdGVuJTIwd2hpbGUlMjBoZWxkJTIwaW4lMjB0aGUlMjBoYW5kLiUyMiUwQWxhYmVscyUyMCUzRCUyMHRvcmNoLnRlbnNvcigwKS51bnNxdWVlemUoMCklMjAlMjAlMjMlMjBjaG9pY2UwJTIwaXMlMjBjb3JyZWN0JTIwKGFjY29yZGluZyUyMHRvJTIwV2lraXBlZGlhJTIwJTNCKSklMkMlMjBiYXRjaCUyMHNpemUlMjAxJTBBJTBBZW5jb2RpbmclMjAlM0QlMjB0b2tlbml6ZXIoJTVCcHJvbXB0JTJDJTIwcHJvbXB0JTVEJTJDJTIwJTVCY2hvaWNlMCUyQyUyMGNob2ljZTElNUQlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyJTJDJTIwcGFkZGluZyUzRFRydWUpJTBBb3V0cHV0cyUyMCUzRCUyMG1vZGVsKCoqJTdCayUzQSUyMHYudW5zcXVlZXplKDApJTIwZm9yJTIwayUyQyUyMHYlMjBpbiUyMGVuY29kaW5nLml0ZW1zKCklN0QlMkMlMjBsYWJlbHMlM0RsYWJlbHMpJTIwJTIwJTIzJTIwYmF0Y2glMjBzaXplJTIwaXMlMjAxJTBBJTBBJTIzJTIwdGhlJTIwbGluZWFyJTIwY2xhc3NpZmllciUyMHN0aWxsJTIwbmVlZHMlMjB0byUyMGJlJTIwdHJhaW5lZCUwQWxvc3MlMjAlM0QlMjBvdXRwdXRzLmxvc3MlMEFsb2dpdHMlMjAlM0QlMjBvdXRwdXRzLmxvZ2l0cw==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, ErnieForMultipleChoice
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;nghuyong/ernie-3.0-base-zh&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = ErnieForMultipleChoice.from_pretrained(<span class="hljs-string">&quot;nghuyong/ernie-3.0-base-zh&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>prompt = <span class="hljs-string">&quot;In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>choice0 = <span class="hljs-string">&quot;It is eaten with a fork and a knife.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>choice1 = <span class="hljs-string">&quot;It is eaten while held in the hand.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.tensor(<span class="hljs-number">0</span>).unsqueeze(<span class="hljs-number">0</span>)  <span class="hljs-comment"># choice0 is correct (according to Wikipedia ;)), batch size 1</span>

<span class="hljs-meta">&gt;&gt;&gt; </span>encoding = tokenizer([prompt, prompt], [choice0, choice1], return_tensors=<span class="hljs-string">&quot;pt&quot;</span>, padding=<span class="hljs-literal">True</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**{k: v.unsqueeze(<span class="hljs-number">0</span>) <span class="hljs-keyword">for</span> k, v <span class="hljs-keyword">in</span> encoding.items()}, labels=labels)  <span class="hljs-comment"># batch size is 1</span>

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># the linear classifier still needs to be trained</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = outputs.loss
<span class="hljs-meta">&gt;&gt;&gt; </span>logits = outputs.logits`,wrap:!1}}),{c(){t=p("p"),t.textContent=h,o=a(),f(r.$$.fragment)},l(n){t=m(n,"P",{"data-svelte-h":!0}),M(t)!=="svelte-11lpom8"&&(t.textContent=h),o=i(n),g(r.$$.fragment,n)},m(n,u){c(n,t,u),c(n,o,u),_(r,n,u),k=!0},p:v,i(n){k||(b(r.$$.fragment,n),k=!0)},o(n){y(r.$$.fragment,n),k=!1},d(n){n&&(l(t),l(o)),T(r,n)}}}function qr(w){let t,h=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=p("p"),t.innerHTML=h},l(o){t=m(o,"P",{"data-svelte-h":!0}),M(t)!=="svelte-fincs2"&&(t.innerHTML=h)},m(o,r){c(o,t,r)},p:v,d(o){o&&l(t)}}}function Br(w){let t,h="Example:",o,r,k;return r=new Y({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBFcm5pZUZvclRva2VuQ2xhc3NpZmljYXRpb24lMEFpbXBvcnQlMjB0b3JjaCUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMm5naHV5b25nJTJGZXJuaWUtMy4wLWJhc2UtemglMjIpJTBBbW9kZWwlMjAlM0QlMjBFcm5pZUZvclRva2VuQ2xhc3NpZmljYXRpb24uZnJvbV9wcmV0cmFpbmVkKCUyMm5naHV5b25nJTJGZXJuaWUtMy4wLWJhc2UtemglMjIpJTBBJTBBaW5wdXRzJTIwJTNEJTIwdG9rZW5pemVyKCUwQSUyMCUyMCUyMCUyMCUyMkh1Z2dpbmdGYWNlJTIwaXMlMjBhJTIwY29tcGFueSUyMGJhc2VkJTIwaW4lMjBQYXJpcyUyMGFuZCUyME5ldyUyMFlvcmslMjIlMkMlMjBhZGRfc3BlY2lhbF90b2tlbnMlM0RGYWxzZSUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIlMEEpJTBBJTBBd2l0aCUyMHRvcmNoLm5vX2dyYWQoKSUzQSUwQSUyMCUyMCUyMCUyMGxvZ2l0cyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzKS5sb2dpdHMlMEElMEFwcmVkaWN0ZWRfdG9rZW5fY2xhc3NfaWRzJTIwJTNEJTIwbG9naXRzLmFyZ21heCgtMSklMEElMEElMjMlMjBOb3RlJTIwdGhhdCUyMHRva2VucyUyMGFyZSUyMGNsYXNzaWZpZWQlMjByYXRoZXIlMjB0aGVuJTIwaW5wdXQlMjB3b3JkcyUyMHdoaWNoJTIwbWVhbnMlMjB0aGF0JTBBJTIzJTIwdGhlcmUlMjBtaWdodCUyMGJlJTIwbW9yZSUyMHByZWRpY3RlZCUyMHRva2VuJTIwY2xhc3NlcyUyMHRoYW4lMjB3b3Jkcy4lMEElMjMlMjBNdWx0aXBsZSUyMHRva2VuJTIwY2xhc3NlcyUyMG1pZ2h0JTIwYWNjb3VudCUyMGZvciUyMHRoZSUyMHNhbWUlMjB3b3JkJTBBcHJlZGljdGVkX3Rva2Vuc19jbGFzc2VzJTIwJTNEJTIwJTVCbW9kZWwuY29uZmlnLmlkMmxhYmVsJTVCdC5pdGVtKCklNUQlMjBmb3IlMjB0JTIwaW4lMjBwcmVkaWN0ZWRfdG9rZW5fY2xhc3NfaWRzJTVCMCU1RCU1RCUwQXByZWRpY3RlZF90b2tlbnNfY2xhc3NlcyUwQSUwQWxhYmVscyUyMCUzRCUyMHByZWRpY3RlZF90b2tlbl9jbGFzc19pZHMlMEFsb3NzJTIwJTNEJTIwbW9kZWwoKippbnB1dHMlMkMlMjBsYWJlbHMlM0RsYWJlbHMpLmxvc3MlMEFyb3VuZChsb3NzLml0ZW0oKSUyQyUyMDIp",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, ErnieForTokenClassification
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;nghuyong/ernie-3.0-base-zh&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = ErnieForTokenClassification.from_pretrained(<span class="hljs-string">&quot;nghuyong/ernie-3.0-base-zh&quot;</span>)

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
...`,wrap:!1}}),{c(){t=p("p"),t.textContent=h,o=a(),f(r.$$.fragment)},l(n){t=m(n,"P",{"data-svelte-h":!0}),M(t)!=="svelte-11lpom8"&&(t.textContent=h),o=i(n),g(r.$$.fragment,n)},m(n,u){c(n,t,u),c(n,o,u),_(r,n,u),k=!0},p:v,i(n){k||(b(r.$$.fragment,n),k=!0)},o(n){y(r.$$.fragment,n),k=!1},d(n){n&&(l(t),l(o)),T(r,n)}}}function Vr(w){let t,h=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=p("p"),t.innerHTML=h},l(o){t=m(o,"P",{"data-svelte-h":!0}),M(t)!=="svelte-fincs2"&&(t.innerHTML=h)},m(o,r){c(o,t,r)},p:v,d(o){o&&l(t)}}}function Hr(w){let t,h="Example:",o,r,k;return r=new Y({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBFcm5pZUZvclF1ZXN0aW9uQW5zd2VyaW5nJTBBaW1wb3J0JTIwdG9yY2glMEElMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJuZ2h1eW9uZyUyRmVybmllLTMuMC1iYXNlLXpoJTIyKSUwQW1vZGVsJTIwJTNEJTIwRXJuaWVGb3JRdWVzdGlvbkFuc3dlcmluZy5mcm9tX3ByZXRyYWluZWQoJTIybmdodXlvbmclMkZlcm5pZS0zLjAtYmFzZS16aCUyMiklMEElMEFxdWVzdGlvbiUyQyUyMHRleHQlMjAlM0QlMjAlMjJXaG8lMjB3YXMlMjBKaW0lMjBIZW5zb24lM0YlMjIlMkMlMjAlMjJKaW0lMjBIZW5zb24lMjB3YXMlMjBhJTIwbmljZSUyMHB1cHBldCUyMiUwQSUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplcihxdWVzdGlvbiUyQyUyMHRleHQlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQXdpdGglMjB0b3JjaC5ub19ncmFkKCklM0ElMEElMjAlMjAlMjAlMjBvdXRwdXRzJTIwJTNEJTIwbW9kZWwoKippbnB1dHMpJTBBJTBBYW5zd2VyX3N0YXJ0X2luZGV4JTIwJTNEJTIwb3V0cHV0cy5zdGFydF9sb2dpdHMuYXJnbWF4KCklMEFhbnN3ZXJfZW5kX2luZGV4JTIwJTNEJTIwb3V0cHV0cy5lbmRfbG9naXRzLmFyZ21heCgpJTBBJTBBcHJlZGljdF9hbnN3ZXJfdG9rZW5zJTIwJTNEJTIwaW5wdXRzLmlucHV0X2lkcyU1QjAlMkMlMjBhbnN3ZXJfc3RhcnRfaW5kZXglMjAlM0ElMjBhbnN3ZXJfZW5kX2luZGV4JTIwJTJCJTIwMSU1RCUwQXRva2VuaXplci5kZWNvZGUocHJlZGljdF9hbnN3ZXJfdG9rZW5zJTJDJTIwc2tpcF9zcGVjaWFsX3Rva2VucyUzRFRydWUpJTBBJTBBJTIzJTIwdGFyZ2V0JTIwaXMlMjAlMjJuaWNlJTIwcHVwcGV0JTIyJTBBdGFyZ2V0X3N0YXJ0X2luZGV4JTIwJTNEJTIwdG9yY2gudGVuc29yKCU1QjE0JTVEKSUwQXRhcmdldF9lbmRfaW5kZXglMjAlM0QlMjB0b3JjaC50ZW5zb3IoJTVCMTUlNUQpJTBBJTBBb3V0cHV0cyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzJTJDJTIwc3RhcnRfcG9zaXRpb25zJTNEdGFyZ2V0X3N0YXJ0X2luZGV4JTJDJTIwZW5kX3Bvc2l0aW9ucyUzRHRhcmdldF9lbmRfaW5kZXgpJTBBbG9zcyUyMCUzRCUyMG91dHB1dHMubG9zcyUwQXJvdW5kKGxvc3MuaXRlbSgpJTJDJTIwMik=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, ErnieForQuestionAnswering
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;nghuyong/ernie-3.0-base-zh&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = ErnieForQuestionAnswering.from_pretrained(<span class="hljs-string">&quot;nghuyong/ernie-3.0-base-zh&quot;</span>)

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
...`,wrap:!1}}),{c(){t=p("p"),t.textContent=h,o=a(),f(r.$$.fragment)},l(n){t=m(n,"P",{"data-svelte-h":!0}),M(t)!=="svelte-11lpom8"&&(t.textContent=h),o=i(n),g(r.$$.fragment,n)},m(n,u){c(n,t,u),c(n,o,u),_(r,n,u),k=!0},p:v,i(n){k||(b(r.$$.fragment,n),k=!0)},o(n){y(r.$$.fragment,n),k=!1},d(n){n&&(l(t),l(o)),T(r,n)}}}function Rr(w){let t,h,o,r,k,n="<em>This model was released on 2019-04-19 and added to Hugging Face Transformers on 2022-09-09.</em>",u,E,gn='<div class="flex flex-wrap space-x-1"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/></div>',Ve,he,bn,He,gs=`<a href="https://huggingface.co/papers/1904.09223" rel="nofollow">ERNIE1.0</a>, <a href="https://ojs.aaai.org/index.php/AAAI/article/view/6428" rel="nofollow">ERNIE2.0</a>,
<a href="https://huggingface.co/papers/2107.02137" rel="nofollow">ERNIE3.0</a>, <a href="https://huggingface.co/papers/2010.12148" rel="nofollow">ERNIE-Gram</a>, <a href="https://huggingface.co/papers/2110.07244" rel="nofollow">ERNIE-health</a> are a series of powerful models proposed by baidu, especially in Chinese tasks.`,yn,Re,_s="ERNIE (Enhanced Representation through kNowledge IntEgration) is designed to learn language representation enhanced by knowledge masking strategies, which includes entity-level masking and phrase-level masking.",Tn,Le,bs='Other ERNIE models released by baidu can be found at <a href="./ernie4_5">Ernie 4.5</a>, and <a href="./ernie4_5_moe">Ernie 4.5 MoE</a>.',Mn,ye,kn,Ge,ys='The example below demonstrates how to predict the <code>[MASK]</code> token with <a href="/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.Pipeline">Pipeline</a>, <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel">AutoModel</a>, and from the command line.',wn,Te,vn,Se,$n,Pe,Ts="Model variants are available in different sizes and languages.",Jn,Xe,Ms='<thead><tr><th align="center">Model Name</th> <th align="center">Language</th> <th align="center">Description</th></tr></thead> <tbody><tr><td align="center">ernie-1.0-base-zh</td> <td align="center">Chinese</td> <td align="center">Layer:12, Heads:12, Hidden:768</td></tr> <tr><td align="center">ernie-2.0-base-en</td> <td align="center">English</td> <td align="center">Layer:12, Heads:12, Hidden:768</td></tr> <tr><td align="center">ernie-2.0-large-en</td> <td align="center">English</td> <td align="center">Layer:24, Heads:16, Hidden:1024</td></tr> <tr><td align="center">ernie-3.0-base-zh</td> <td align="center">Chinese</td> <td align="center">Layer:12, Heads:12, Hidden:768</td></tr> <tr><td align="center">ernie-3.0-medium-zh</td> <td align="center">Chinese</td> <td align="center">Layer:6, Heads:12, Hidden:768</td></tr> <tr><td align="center">ernie-3.0-mini-zh</td> <td align="center">Chinese</td> <td align="center">Layer:6, Heads:12, Hidden:384</td></tr> <tr><td align="center">ernie-3.0-micro-zh</td> <td align="center">Chinese</td> <td align="center">Layer:4, Heads:12, Hidden:384</td></tr> <tr><td align="center">ernie-3.0-nano-zh</td> <td align="center">Chinese</td> <td align="center">Layer:4, Heads:12, Hidden:312</td></tr> <tr><td align="center">ernie-health-zh</td> <td align="center">Chinese</td> <td align="center">Layer:12, Heads:12, Hidden:768</td></tr> <tr><td align="center">ernie-gram-zh</td> <td align="center">Chinese</td> <td align="center">Layer:12, Heads:12, Hidden:768</td></tr></tbody>',En,Qe,Cn,Ae,ks=`You can find all the supported models from huggingface’s model hub: <a href="https://huggingface.co/nghuyong" rel="nofollow">huggingface.co/nghuyong</a>, and model details from paddle’s official
repo: <a href="https://paddlenlp.readthedocs.io/zh/latest/model_zoo/transformers/ERNIE/contents.html" rel="nofollow">PaddleNLP</a>
and <a href="https://github.com/PaddlePaddle/ERNIE/tree/legacy/develop" rel="nofollow">ERNIE’s legacy branch</a>.`,jn,Oe,zn,H,Ye,eo,jt,ws=`This is the configuration class to store the configuration of a <a href="/docs/transformers/v4.56.2/en/model_doc/ernie#transformers.ErnieModel">ErnieModel</a> or a <code>TFErnieModel</code>. It is used to
instantiate a ERNIE model according to the specified arguments, defining the model architecture. Instantiating a
configuration with the defaults will yield a similar configuration to that of the ERNIE
<a href="https://huggingface.co/nghuyong/ernie-3.0-base-zh" rel="nofollow">nghuyong/ernie-3.0-base-zh</a> architecture.`,to,zt,vs=`Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> and can be used to control the model outputs. Read the
documentation from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> for more information.`,no,Me,Un,De,xn,_e,Ke,oo,Ut,$s='Output type of <a href="/docs/transformers/v4.56.2/en/model_doc/ernie#transformers.ErnieForPreTraining">ErnieForPreTraining</a>.',Fn,et,In,j,tt,so,xt,Js=`The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
cross-attention is added between the self-attention layers, following the architecture described in <a href="https://huggingface.co/papers/1706.03762" rel="nofollow">Attention is
all you need</a> by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.`,ro,Ft,Es=`To behave as an decoder the model needs to be initialized with the <code>is_decoder</code> argument of the configuration set
to <code>True</code>. To be used in a Seq2Seq model, the model needs to initialized with both <code>is_decoder</code> argument and`,ao,It,Cs=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,io,Wt,js=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,lo,ue,nt,co,Zt,zs='The <a href="/docs/transformers/v4.56.2/en/model_doc/ernie#transformers.ErnieModel">ErnieModel</a> forward method, overrides the <code>__call__</code> special method.',po,ke,Wn,ot,Zn,z,st,mo,Nt,Us="Ernie Model with two heads on top as done during the pretraining: a <code>masked language modeling</code> head and a <code>next sentence prediction (classification)</code> head.",ho,qt,xs=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,uo,Bt,Fs=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,fo,D,rt,go,Vt,Is='The <a href="/docs/transformers/v4.56.2/en/model_doc/ernie#transformers.ErnieForPreTraining">ErnieForPreTraining</a> forward method, overrides the <code>__call__</code> special method.',_o,we,bo,ve,Nn,at,qn,U,it,yo,Ht,Ws="Ernie Model with a <code>language modeling</code> head on top for CLM fine-tuning.",To,Rt,Zs=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Mo,Lt,Ns=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,ko,K,lt,wo,Gt,qs='The <a href="/docs/transformers/v4.56.2/en/model_doc/ernie#transformers.ErnieForCausalLM">ErnieForCausalLM</a> forward method, overrides the <code>__call__</code> special method.',vo,$e,$o,Je,Bn,dt,Vn,x,ct,Jo,St,Bs="The Ernie Model with a <code>language modeling</code> head on top.”",Eo,Pt,Vs=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Co,Xt,Hs=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,jo,ee,pt,zo,Qt,Rs='The <a href="/docs/transformers/v4.56.2/en/model_doc/ernie#transformers.ErnieForMaskedLM">ErnieForMaskedLM</a> forward method, overrides the <code>__call__</code> special method.',Uo,Ee,xo,Ce,Hn,mt,Rn,F,ht,Fo,At,Ls="Ernie Model with a <code>next sentence prediction (classification)</code> head on top.",Io,Ot,Gs=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Wo,Yt,Ss=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Zo,te,ut,No,Dt,Ps='The <a href="/docs/transformers/v4.56.2/en/model_doc/ernie#transformers.ErnieForNextSentencePrediction">ErnieForNextSentencePrediction</a> forward method, overrides the <code>__call__</code> special method.',qo,je,Bo,ze,Ln,ft,Gn,I,gt,Vo,Kt,Xs=`Ernie Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled
output) e.g. for GLUE tasks.`,Ho,en,Qs=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Ro,tn,As=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Lo,B,_t,Go,nn,Os='The <a href="/docs/transformers/v4.56.2/en/model_doc/ernie#transformers.ErnieForSequenceClassification">ErnieForSequenceClassification</a> forward method, overrides the <code>__call__</code> special method.',So,Ue,Po,xe,Xo,Fe,Sn,bt,Pn,W,yt,Qo,on,Ys=`The Ernie Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
softmax) e.g. for RocStories/SWAG tasks.`,Ao,sn,Ds=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Oo,rn,Ks=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Yo,ne,Tt,Do,an,er='The <a href="/docs/transformers/v4.56.2/en/model_doc/ernie#transformers.ErnieForMultipleChoice">ErnieForMultipleChoice</a> forward method, overrides the <code>__call__</code> special method.',Ko,Ie,es,We,Xn,Mt,Qn,Z,kt,ts,ln,tr=`The Ernie transformer with a token classification head on top (a linear layer on top of the hidden-states
output) e.g. for Named-Entity-Recognition (NER) tasks.`,ns,dn,nr=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,os,cn,or=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,ss,oe,wt,rs,pn,sr='The <a href="/docs/transformers/v4.56.2/en/model_doc/ernie#transformers.ErnieForTokenClassification">ErnieForTokenClassification</a> forward method, overrides the <code>__call__</code> special method.',as,Ze,is,Ne,An,vt,On,N,$t,ls,mn,rr=`The Ernie transformer with a span classification head on top for extractive question-answering tasks like
SQuAD (a linear layer on top of the hidden-states output to compute <code>span start logits</code> and <code>span end logits</code>).`,ds,hn,ar=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,cs,un,ir=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,ps,se,Jt,ms,fn,lr='The <a href="/docs/transformers/v4.56.2/en/model_doc/ernie#transformers.ErnieForQuestionAnswering">ErnieForQuestionAnswering</a> forward method, overrides the <code>__call__</code> special method.',hs,qe,us,Be,Yn,Et,Dn,_n,Kn;return he=new V({props:{title:"ERNIE",local:"ernie",headingTag:"h1"}}),ye=new fe({props:{warning:!1,$$slots:{default:[br]},$$scope:{ctx:w}}}),Te=new _r({props:{id:"usage",options:["Pipeline","AutoModel","transformers CLI"],$$slots:{default:[kr]},$$scope:{ctx:w}}}),Se=new V({props:{title:"Notes",local:"notes",headingTag:"h2"}}),Qe=new V({props:{title:"Resources",local:"resources",headingTag:"h2"}}),Oe=new V({props:{title:"ErnieConfig",local:"transformers.ErnieConfig",headingTag:"h2"}}),Ye=new C({props:{name:"class transformers.ErnieConfig",anchor:"transformers.ErnieConfig",parameters:[{name:"vocab_size",val:" = 30522"},{name:"hidden_size",val:" = 768"},{name:"num_hidden_layers",val:" = 12"},{name:"num_attention_heads",val:" = 12"},{name:"intermediate_size",val:" = 3072"},{name:"hidden_act",val:" = 'gelu'"},{name:"hidden_dropout_prob",val:" = 0.1"},{name:"attention_probs_dropout_prob",val:" = 0.1"},{name:"max_position_embeddings",val:" = 512"},{name:"type_vocab_size",val:" = 2"},{name:"task_type_vocab_size",val:" = 3"},{name:"use_task_id",val:" = False"},{name:"initializer_range",val:" = 0.02"},{name:"layer_norm_eps",val:" = 1e-12"},{name:"pad_token_id",val:" = 0"},{name:"position_embedding_type",val:" = 'absolute'"},{name:"use_cache",val:" = True"},{name:"classifier_dropout",val:" = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.ErnieConfig.vocab_size",description:`<strong>vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 30522) &#x2014;
Vocabulary size of the ERNIE model. Defines the number of different tokens that can be represented by the
<code>inputs_ids</code> passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/ernie#transformers.ErnieModel">ErnieModel</a> or <code>TFErnieModel</code>.`,name:"vocab_size"},{anchor:"transformers.ErnieConfig.hidden_size",description:`<strong>hidden_size</strong> (<code>int</code>, <em>optional</em>, defaults to 768) &#x2014;
Dimensionality of the encoder layers and the pooler layer.`,name:"hidden_size"},{anchor:"transformers.ErnieConfig.num_hidden_layers",description:`<strong>num_hidden_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 12) &#x2014;
Number of hidden layers in the Transformer encoder.`,name:"num_hidden_layers"},{anchor:"transformers.ErnieConfig.num_attention_heads",description:`<strong>num_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 12) &#x2014;
Number of attention heads for each attention layer in the Transformer encoder.`,name:"num_attention_heads"},{anchor:"transformers.ErnieConfig.intermediate_size",description:`<strong>intermediate_size</strong> (<code>int</code>, <em>optional</em>, defaults to 3072) &#x2014;
Dimensionality of the &#x201C;intermediate&#x201D; (often named feed-forward) layer in the Transformer encoder.`,name:"intermediate_size"},{anchor:"transformers.ErnieConfig.hidden_act",description:`<strong>hidden_act</strong> (<code>str</code> or <code>Callable</code>, <em>optional</em>, defaults to <code>&quot;gelu&quot;</code>) &#x2014;
The non-linear activation function (function or string) in the encoder and pooler. If string, <code>&quot;gelu&quot;</code>,
<code>&quot;relu&quot;</code>, <code>&quot;silu&quot;</code> and <code>&quot;gelu_new&quot;</code> are supported.`,name:"hidden_act"},{anchor:"transformers.ErnieConfig.hidden_dropout_prob",description:`<strong>hidden_dropout_prob</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.`,name:"hidden_dropout_prob"},{anchor:"transformers.ErnieConfig.attention_probs_dropout_prob",description:`<strong>attention_probs_dropout_prob</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout ratio for the attention probabilities.`,name:"attention_probs_dropout_prob"},{anchor:"transformers.ErnieConfig.max_position_embeddings",description:`<strong>max_position_embeddings</strong> (<code>int</code>, <em>optional</em>, defaults to 512) &#x2014;
The maximum sequence length that this model might ever be used with. Typically set this to something large
just in case (e.g., 512 or 1024 or 2048).`,name:"max_position_embeddings"},{anchor:"transformers.ErnieConfig.type_vocab_size",description:`<strong>type_vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 2) &#x2014;
The vocabulary size of the <code>token_type_ids</code> passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/ernie#transformers.ErnieModel">ErnieModel</a> or <code>TFErnieModel</code>.`,name:"type_vocab_size"},{anchor:"transformers.ErnieConfig.task_type_vocab_size",description:`<strong>task_type_vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 3) &#x2014;
The vocabulary size of the <code>task_type_ids</code> for ERNIE2.0/ERNIE3.0 model`,name:"task_type_vocab_size"},{anchor:"transformers.ErnieConfig.use_task_id",description:`<strong>use_task_id</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not the model support <code>task_type_ids</code>`,name:"use_task_id"},{anchor:"transformers.ErnieConfig.initializer_range",description:`<strong>initializer_range</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.`,name:"initializer_range"},{anchor:"transformers.ErnieConfig.layer_norm_eps",description:`<strong>layer_norm_eps</strong> (<code>float</code>, <em>optional</em>, defaults to 1e-12) &#x2014;
The epsilon used by the layer normalization layers.`,name:"layer_norm_eps"},{anchor:"transformers.ErnieConfig.pad_token_id",description:`<strong>pad_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 0) &#x2014;
Padding token id.`,name:"pad_token_id"},{anchor:"transformers.ErnieConfig.position_embedding_type",description:`<strong>position_embedding_type</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;absolute&quot;</code>) &#x2014;
Type of position embedding. Choose one of <code>&quot;absolute&quot;</code>, <code>&quot;relative_key&quot;</code>, <code>&quot;relative_key_query&quot;</code>. For
positional embeddings use <code>&quot;absolute&quot;</code>. For more information on <code>&quot;relative_key&quot;</code>, please refer to
<a href="https://huggingface.co/papers/1803.02155" rel="nofollow">Self-Attention with Relative Position Representations (Shaw et al.)</a>.
For more information on <code>&quot;relative_key_query&quot;</code>, please refer to <em>Method 4</em> in <a href="https://huggingface.co/papers/2009.13658" rel="nofollow">Improve Transformer Models
with Better Relative Position Embeddings (Huang et al.)</a>.`,name:"position_embedding_type"},{anchor:"transformers.ErnieConfig.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not the model should return the last key/values attentions (not used by all models). Only
relevant if <code>config.is_decoder=True</code>.`,name:"use_cache"},{anchor:"transformers.ErnieConfig.classifier_dropout",description:`<strong>classifier_dropout</strong> (<code>float</code>, <em>optional</em>) &#x2014;
The dropout ratio for the classification head.`,name:"classifier_dropout"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/ernie/configuration_ernie.py#L29"}}),Me=new ge({props:{anchor:"transformers.ErnieConfig.example",$$slots:{default:[wr]},$$scope:{ctx:w}}}),De=new V({props:{title:"Ernie specific outputs",local:"transformers.models.ernie.modeling_ernie.ErnieForPreTrainingOutput",headingTag:"h2"}}),Ke=new C({props:{name:"class transformers.models.ernie.modeling_ernie.ErnieForPreTrainingOutput",anchor:"transformers.models.ernie.modeling_ernie.ErnieForPreTrainingOutput",parameters:[{name:"loss",val:": typing.Optional[torch.FloatTensor] = None"},{name:"prediction_logits",val:": typing.Optional[torch.FloatTensor] = None"},{name:"seq_relationship_logits",val:": typing.Optional[torch.FloatTensor] = None"},{name:"hidden_states",val:": typing.Optional[tuple[torch.FloatTensor]] = None"},{name:"attentions",val:": typing.Optional[tuple[torch.FloatTensor]] = None"}],parametersDescription:[{anchor:"transformers.models.ernie.modeling_ernie.ErnieForPreTrainingOutput.loss",description:`<strong>loss</strong> (<code>*optional*</code>, returned when <code>labels</code> is provided, <code>torch.FloatTensor</code> of shape <code>(1,)</code>) &#x2014;
Total loss as the sum of the masked language modeling loss and the next sequence prediction
(classification) loss.`,name:"loss"},{anchor:"transformers.models.ernie.modeling_ernie.ErnieForPreTrainingOutput.prediction_logits",description:`<strong>prediction_logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, config.vocab_size)</code>) &#x2014;
Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).`,name:"prediction_logits"},{anchor:"transformers.models.ernie.modeling_ernie.ErnieForPreTrainingOutput.seq_relationship_logits",description:`<strong>seq_relationship_logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, 2)</code>) &#x2014;
Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation
before SoftMax).`,name:"seq_relationship_logits"},{anchor:"transformers.models.ernie.modeling_ernie.ErnieForPreTrainingOutput.hidden_states",description:`<strong>hidden_states</strong> (<code>tuple[torch.FloatTensor]</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) &#x2014;
Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings, if the model has an embedding layer, +
one for the output of each layer) of shape <code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.`,name:"hidden_states"},{anchor:"transformers.models.ernie.modeling_ernie.ErnieForPreTrainingOutput.attentions",description:`<strong>attentions</strong> (<code>tuple[torch.FloatTensor]</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) &#x2014;
Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
heads.`,name:"attentions"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/ernie/modeling_ernie.py#L652"}}),et=new V({props:{title:"ErnieModel",local:"transformers.ErnieModel",headingTag:"h2"}}),tt=new C({props:{name:"class transformers.ErnieModel",anchor:"transformers.ErnieModel",parameters:[{name:"config",val:""},{name:"add_pooling_layer",val:" = True"}],parametersDescription:[{anchor:"transformers.ErnieModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/ernie#transformers.ErnieModel">ErnieModel</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"},{anchor:"transformers.ErnieModel.add_pooling_layer",description:`<strong>add_pooling_layer</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether to add a pooling layer`,name:"add_pooling_layer"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/ernie/modeling_ernie.py#L682"}}),nt=new C({props:{name:"forward",anchor:"transformers.ErnieModel.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"task_type_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_hidden_states",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"past_key_values",val:": typing.Optional[list[torch.FloatTensor]] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.ErnieModel.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.ErnieModel.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.ErnieModel.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.ErnieModel.forward.task_type_ids",description:`<strong>task_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Task type embedding is a special embedding to represent the characteristic of different tasks, such as
word-aware pre-training task, structure-aware pre-training task and semantic-aware pre-training task. We
assign a <code>task_type_id</code> to each task and the <code>task_type_id</code> is in the range \`[0,
config.task_type_vocab_size-1]`,name:"task_type_ids"},{anchor:"transformers.ErnieModel.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.ErnieModel.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.ErnieModel.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.ErnieModel.forward.encoder_hidden_states",description:`<strong>encoder_hidden_states</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
if the model is configured as a decoder.`,name:"encoder_hidden_states"},{anchor:"transformers.ErnieModel.forward.encoder_attention_mask",description:`<strong>encoder_attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
the cross-attention if the model is configured as a decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>`,name:"encoder_attention_mask"},{anchor:"transformers.ErnieModel.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>list[torch.FloatTensor]</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.ErnieModel.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.ErnieModel.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.ErnieModel.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.ErnieModel.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/ernie/modeling_ernie.py#L717",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions"
>transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/ernie#transformers.ErnieConfig"
>ErnieConfig</a>) and inputs.</p>
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
`}}),ke=new fe({props:{$$slots:{default:[vr]},$$scope:{ctx:w}}}),ot=new V({props:{title:"ErnieForPreTraining",local:"transformers.ErnieForPreTraining",headingTag:"h2"}}),st=new C({props:{name:"class transformers.ErnieForPreTraining",anchor:"transformers.ErnieForPreTraining",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.ErnieForPreTraining.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/ernie#transformers.ErnieForPreTraining">ErnieForPreTraining</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/ernie/modeling_ernie.py#L849"}}),rt=new C({props:{name:"forward",anchor:"transformers.ErnieForPreTraining.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"task_type_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"labels",val:": typing.Optional[torch.Tensor] = None"},{name:"next_sentence_label",val:": typing.Optional[torch.Tensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.ErnieForPreTraining.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.ErnieForPreTraining.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.ErnieForPreTraining.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.ErnieForPreTraining.forward.task_type_ids",description:`<strong>task_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Task type embedding is a special embedding to represent the characteristic of different tasks, such as
word-aware pre-training task, structure-aware pre-training task and semantic-aware pre-training task. We
assign a <code>task_type_id</code> to each task and the <code>task_type_id</code> is in the range \`[0,
config.task_type_vocab_size-1]`,name:"task_type_ids"},{anchor:"transformers.ErnieForPreTraining.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.ErnieForPreTraining.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.ErnieForPreTraining.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.ErnieForPreTraining.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should be in <code>[-100, 0, ..., config.vocab_size]</code> (see <code>input_ids</code> docstring) Tokens with indices set to <code>-100</code> are ignored (masked),
the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>`,name:"labels"},{anchor:"transformers.ErnieForPreTraining.forward.next_sentence_label",description:`<strong>next_sentence_label</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the next sequence prediction (classification) loss. Input should be a sequence
pair (see <code>input_ids</code> docstring) Indices should be in <code>[0, 1]</code>:</p>
<ul>
<li>0 indicates sequence B is a continuation of sequence A,</li>
<li>1 indicates sequence B is a random sequence.</li>
</ul>`,name:"next_sentence_label"},{anchor:"transformers.ErnieForPreTraining.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.ErnieForPreTraining.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.ErnieForPreTraining.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/ernie/modeling_ernie.py#L871",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/model_doc/ernie#transformers.models.ernie.modeling_ernie.ErnieForPreTrainingOutput"
>transformers.models.ernie.modeling_ernie.ErnieForPreTrainingOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/ernie#transformers.ErnieConfig"
>ErnieConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>loss</strong> (<code>*optional*</code>, returned when <code>labels</code> is provided, <code>torch.FloatTensor</code> of shape <code>(1,)</code>) — Total loss as the sum of the masked language modeling loss and the next sequence prediction
(classification) loss.</p>
</li>
<li>
<p><strong>prediction_logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, config.vocab_size)</code>) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).</p>
</li>
<li>
<p><strong>seq_relationship_logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, 2)</code>) — Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation
before SoftMax).</p>
</li>
<li>
<p><strong>hidden_states</strong> (<code>tuple[torch.FloatTensor]</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings, if the model has an embedding layer, +
one for the output of each layer) of shape <code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.</p>
</li>
<li>
<p><strong>attentions</strong> (<code>tuple[torch.FloatTensor]</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
heads.</p>
</li>
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><a
  href="/docs/transformers/v4.56.2/en/model_doc/ernie#transformers.models.ernie.modeling_ernie.ErnieForPreTrainingOutput"
>transformers.models.ernie.modeling_ernie.ErnieForPreTrainingOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),we=new fe({props:{$$slots:{default:[$r]},$$scope:{ctx:w}}}),ve=new ge({props:{anchor:"transformers.ErnieForPreTraining.forward.example",$$slots:{default:[Jr]},$$scope:{ctx:w}}}),at=new V({props:{title:"ErnieForCausalLM",local:"transformers.ErnieForCausalLM",headingTag:"h2"}}),it=new C({props:{name:"class transformers.ErnieForCausalLM",anchor:"transformers.ErnieForCausalLM",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.ErnieForCausalLM.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/ernie#transformers.ErnieForCausalLM">ErnieForCausalLM</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/ernie/modeling_ernie.py#L963"}}),lt=new C({props:{name:"forward",anchor:"transformers.ErnieForCausalLM.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"task_type_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_hidden_states",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"labels",val:": typing.Optional[torch.Tensor] = None"},{name:"past_key_values",val:": typing.Optional[list[torch.Tensor]] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.ErnieForCausalLM.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.ErnieForCausalLM.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.ErnieForCausalLM.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.ErnieForCausalLM.forward.task_type_ids",description:`<strong>task_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Task type embedding is a special embedding to represent the characteristic of different tasks, such as
word-aware pre-training task, structure-aware pre-training task and semantic-aware pre-training task. We
assign a <code>task_type_id</code> to each task and the <code>task_type_id</code> is in the range \`[0,
config.task_type_vocab_size-1]`,name:"task_type_ids"},{anchor:"transformers.ErnieForCausalLM.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.ErnieForCausalLM.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.ErnieForCausalLM.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.ErnieForCausalLM.forward.encoder_hidden_states",description:`<strong>encoder_hidden_states</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
if the model is configured as a decoder.`,name:"encoder_hidden_states"},{anchor:"transformers.ErnieForCausalLM.forward.encoder_attention_mask",description:`<strong>encoder_attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
the cross-attention if the model is configured as a decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>`,name:"encoder_attention_mask"},{anchor:"transformers.ErnieForCausalLM.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the left-to-right language modeling loss (next word prediction). Indices should be in
<code>[-100, 0, ..., config.vocab_size]</code> (see <code>input_ids</code> docstring) Tokens with indices set to <code>-100</code> are
ignored (masked), the loss is only computed for the tokens with labels n <code>[0, ..., config.vocab_size]</code>`,name:"labels"},{anchor:"transformers.ErnieForCausalLM.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>list[torch.Tensor]</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.ErnieForCausalLM.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.ErnieForCausalLM.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.ErnieForCausalLM.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.ErnieForCausalLM.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/ernie/modeling_ernie.py#L988",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithCrossAttentions"
>transformers.modeling_outputs.CausalLMOutputWithCrossAttentions</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/ernie#transformers.ErnieConfig"
>ErnieConfig</a>) and inputs.</p>
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
`}}),$e=new fe({props:{$$slots:{default:[Er]},$$scope:{ctx:w}}}),Je=new ge({props:{anchor:"transformers.ErnieForCausalLM.forward.example",$$slots:{default:[Cr]},$$scope:{ctx:w}}}),dt=new V({props:{title:"ErnieForMaskedLM",local:"transformers.ErnieForMaskedLM",headingTag:"h2"}}),ct=new C({props:{name:"class transformers.ErnieForMaskedLM",anchor:"transformers.ErnieForMaskedLM",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.ErnieForMaskedLM.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/ernie#transformers.ErnieForMaskedLM">ErnieForMaskedLM</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/ernie/modeling_ernie.py#L1067"}}),pt=new C({props:{name:"forward",anchor:"transformers.ErnieForMaskedLM.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"task_type_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_hidden_states",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"labels",val:": typing.Optional[torch.Tensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.ErnieForMaskedLM.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.ErnieForMaskedLM.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.ErnieForMaskedLM.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.ErnieForMaskedLM.forward.task_type_ids",description:`<strong>task_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Task type embedding is a special embedding to represent the characteristic of different tasks, such as
word-aware pre-training task, structure-aware pre-training task and semantic-aware pre-training task. We
assign a <code>task_type_id</code> to each task and the <code>task_type_id</code> is in the range \`[0,
config.task_type_vocab_size-1]`,name:"task_type_ids"},{anchor:"transformers.ErnieForMaskedLM.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.ErnieForMaskedLM.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.ErnieForMaskedLM.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.ErnieForMaskedLM.forward.encoder_hidden_states",description:`<strong>encoder_hidden_states</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
if the model is configured as a decoder.`,name:"encoder_hidden_states"},{anchor:"transformers.ErnieForMaskedLM.forward.encoder_attention_mask",description:`<strong>encoder_attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
the cross-attention if the model is configured as a decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>`,name:"encoder_attention_mask"},{anchor:"transformers.ErnieForMaskedLM.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should be in <code>[-100, 0, ..., config.vocab_size]</code> (see <code>input_ids</code> docstring) Tokens with indices set to <code>-100</code> are ignored (masked), the
loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>`,name:"labels"},{anchor:"transformers.ErnieForMaskedLM.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.ErnieForMaskedLM.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.ErnieForMaskedLM.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/ernie/modeling_ernie.py#L1095",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.MaskedLMOutput"
>transformers.modeling_outputs.MaskedLMOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/ernie#transformers.ErnieConfig"
>ErnieConfig</a>) and inputs.</p>
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
`}}),Ee=new fe({props:{$$slots:{default:[jr]},$$scope:{ctx:w}}}),Ce=new ge({props:{anchor:"transformers.ErnieForMaskedLM.forward.example",$$slots:{default:[zr]},$$scope:{ctx:w}}}),mt=new V({props:{title:"ErnieForNextSentencePrediction",local:"transformers.ErnieForNextSentencePrediction",headingTag:"h2"}}),ht=new C({props:{name:"class transformers.ErnieForNextSentencePrediction",anchor:"transformers.ErnieForNextSentencePrediction",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.ErnieForNextSentencePrediction.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/ernie#transformers.ErnieForNextSentencePrediction">ErnieForNextSentencePrediction</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/ernie/modeling_ernie.py#L1191"}}),ut=new C({props:{name:"forward",anchor:"transformers.ErnieForNextSentencePrediction.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"task_type_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"labels",val:": typing.Optional[torch.Tensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.ErnieForNextSentencePrediction.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.ErnieForNextSentencePrediction.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.ErnieForNextSentencePrediction.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.ErnieForNextSentencePrediction.forward.task_type_ids",description:`<strong>task_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Task type embedding is a special embedding to represent the characteristic of different tasks, such as
word-aware pre-training task, structure-aware pre-training task and semantic-aware pre-training task. We
assign a <code>task_type_id</code> to each task and the <code>task_type_id</code> is in the range \`[0,
config.task_type_vocab_size-1]`,name:"task_type_ids"},{anchor:"transformers.ErnieForNextSentencePrediction.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.ErnieForNextSentencePrediction.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.ErnieForNextSentencePrediction.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.ErnieForNextSentencePrediction.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the next sequence prediction (classification) loss. Input should be a sequence pair
(see <code>input_ids</code> docstring). Indices should be in <code>[0, 1]</code>:</p>
<ul>
<li>0 indicates sequence B is a continuation of sequence A,</li>
<li>1 indicates sequence B is a random sequence.</li>
</ul>`,name:"labels"},{anchor:"transformers.ErnieForNextSentencePrediction.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.ErnieForNextSentencePrediction.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.ErnieForNextSentencePrediction.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/ernie/modeling_ernie.py#L1202",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.NextSentencePredictorOutput"
>transformers.modeling_outputs.NextSentencePredictorOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/ernie#transformers.ErnieConfig"
>ErnieConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>loss</strong> (<code>torch.FloatTensor</code> of shape <code>(1,)</code>, <em>optional</em>, returned when <code>next_sentence_label</code> is provided) — Next sequence prediction (classification) loss.</p>
</li>
<li>
<p><strong>logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, 2)</code>) — Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation
before SoftMax).</p>
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
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.NextSentencePredictorOutput"
>transformers.modeling_outputs.NextSentencePredictorOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),je=new fe({props:{$$slots:{default:[Ur]},$$scope:{ctx:w}}}),ze=new ge({props:{anchor:"transformers.ErnieForNextSentencePrediction.forward.example",$$slots:{default:[xr]},$$scope:{ctx:w}}}),ft=new V({props:{title:"ErnieForSequenceClassification",local:"transformers.ErnieForSequenceClassification",headingTag:"h2"}}),gt=new C({props:{name:"class transformers.ErnieForSequenceClassification",anchor:"transformers.ErnieForSequenceClassification",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.ErnieForSequenceClassification.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/ernie#transformers.ErnieForSequenceClassification">ErnieForSequenceClassification</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/ernie/modeling_ernie.py#L1300"}}),_t=new C({props:{name:"forward",anchor:"transformers.ErnieForSequenceClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"task_type_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"labels",val:": typing.Optional[torch.Tensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.ErnieForSequenceClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.ErnieForSequenceClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.ErnieForSequenceClassification.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.ErnieForSequenceClassification.forward.task_type_ids",description:`<strong>task_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Task type embedding is a special embedding to represent the characteristic of different tasks, such as
word-aware pre-training task, structure-aware pre-training task and semantic-aware pre-training task. We
assign a <code>task_type_id</code> to each task and the <code>task_type_id</code> is in the range \`[0,
config.task_type_vocab_size-1]`,name:"task_type_ids"},{anchor:"transformers.ErnieForSequenceClassification.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.ErnieForSequenceClassification.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.ErnieForSequenceClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.ErnieForSequenceClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the sequence classification/regression loss. Indices should be in <code>[0, ..., config.num_labels - 1]</code>. If <code>config.num_labels == 1</code> a regression loss is computed (Mean-Square loss), If
<code>config.num_labels &gt; 1</code> a classification loss is computed (Cross-Entropy).`,name:"labels"},{anchor:"transformers.ErnieForSequenceClassification.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.ErnieForSequenceClassification.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.ErnieForSequenceClassification.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/ernie/modeling_ernie.py#L1317",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput"
>transformers.modeling_outputs.SequenceClassifierOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/ernie#transformers.ErnieConfig"
>ErnieConfig</a>) and inputs.</p>
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
`}}),Ue=new fe({props:{$$slots:{default:[Fr]},$$scope:{ctx:w}}}),xe=new ge({props:{anchor:"transformers.ErnieForSequenceClassification.forward.example",$$slots:{default:[Ir]},$$scope:{ctx:w}}}),Fe=new ge({props:{anchor:"transformers.ErnieForSequenceClassification.forward.example-2",$$slots:{default:[Wr]},$$scope:{ctx:w}}}),bt=new V({props:{title:"ErnieForMultipleChoice",local:"transformers.ErnieForMultipleChoice",headingTag:"h2"}}),yt=new C({props:{name:"class transformers.ErnieForMultipleChoice",anchor:"transformers.ErnieForMultipleChoice",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.ErnieForMultipleChoice.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/ernie#transformers.ErnieForMultipleChoice">ErnieForMultipleChoice</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/ernie/modeling_ernie.py#L1398"}}),Tt=new C({props:{name:"forward",anchor:"transformers.ErnieForMultipleChoice.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"task_type_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"labels",val:": typing.Optional[torch.Tensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.ErnieForMultipleChoice.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, num_choices, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.ErnieForMultipleChoice.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.ErnieForMultipleChoice.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, num_choices, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.ErnieForMultipleChoice.forward.task_type_ids",description:`<strong>task_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, num_choices, sequence_length)</code>, <em>optional</em>) &#x2014;
Task type embedding is a special embedding to represent the characteristic of different tasks, such as
word-aware pre-training task, structure-aware pre-training task and semantic-aware pre-training task. We
assign a <code>task_type_id</code> to each task and the <code>task_type_id</code> is in the range \`[0,
config.task_type_vocab_size-1]`,name:"task_type_ids"},{anchor:"transformers.ErnieForMultipleChoice.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, num_choices, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.max_position_embeddings - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.ErnieForMultipleChoice.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.ErnieForMultipleChoice.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_choices, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.ErnieForMultipleChoice.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the multiple choice classification loss. Indices should be in <code>[0, ..., num_choices-1]</code> where <code>num_choices</code> is the size of the second dimension of the input tensors. (See
<code>input_ids</code> above)`,name:"labels"},{anchor:"transformers.ErnieForMultipleChoice.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.ErnieForMultipleChoice.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.ErnieForMultipleChoice.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/ernie/modeling_ernie.py#L1413",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.MultipleChoiceModelOutput"
>transformers.modeling_outputs.MultipleChoiceModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/ernie#transformers.ErnieConfig"
>ErnieConfig</a>) and inputs.</p>
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
`}}),Ie=new fe({props:{$$slots:{default:[Zr]},$$scope:{ctx:w}}}),We=new ge({props:{anchor:"transformers.ErnieForMultipleChoice.forward.example",$$slots:{default:[Nr]},$$scope:{ctx:w}}}),Mt=new V({props:{title:"ErnieForTokenClassification",local:"transformers.ErnieForTokenClassification",headingTag:"h2"}}),kt=new C({props:{name:"class transformers.ErnieForTokenClassification",anchor:"transformers.ErnieForTokenClassification",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.ErnieForTokenClassification.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/ernie#transformers.ErnieForTokenClassification">ErnieForTokenClassification</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/ernie/modeling_ernie.py#L1513"}}),wt=new C({props:{name:"forward",anchor:"transformers.ErnieForTokenClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"task_type_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"labels",val:": typing.Optional[torch.Tensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.ErnieForTokenClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.ErnieForTokenClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.ErnieForTokenClassification.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.ErnieForTokenClassification.forward.task_type_ids",description:`<strong>task_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Task type embedding is a special embedding to represent the characteristic of different tasks, such as
word-aware pre-training task, structure-aware pre-training task and semantic-aware pre-training task. We
assign a <code>task_type_id</code> to each task and the <code>task_type_id</code> is in the range \`[0,
config.task_type_vocab_size-1]`,name:"task_type_ids"},{anchor:"transformers.ErnieForTokenClassification.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.ErnieForTokenClassification.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.ErnieForTokenClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.ErnieForTokenClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the token classification loss. Indices should be in <code>[0, ..., config.num_labels - 1]</code>.`,name:"labels"},{anchor:"transformers.ErnieForTokenClassification.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.ErnieForTokenClassification.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.ErnieForTokenClassification.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/ernie/modeling_ernie.py#L1529",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput"
>transformers.modeling_outputs.TokenClassifierOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/ernie#transformers.ErnieConfig"
>ErnieConfig</a>) and inputs.</p>
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
`}}),Ze=new fe({props:{$$slots:{default:[qr]},$$scope:{ctx:w}}}),Ne=new ge({props:{anchor:"transformers.ErnieForTokenClassification.forward.example",$$slots:{default:[Br]},$$scope:{ctx:w}}}),vt=new V({props:{title:"ErnieForQuestionAnswering",local:"transformers.ErnieForQuestionAnswering",headingTag:"h2"}}),$t=new C({props:{name:"class transformers.ErnieForQuestionAnswering",anchor:"transformers.ErnieForQuestionAnswering",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.ErnieForQuestionAnswering.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/ernie#transformers.ErnieForQuestionAnswering">ErnieForQuestionAnswering</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/ernie/modeling_ernie.py#L1591"}}),Jt=new C({props:{name:"forward",anchor:"transformers.ErnieForQuestionAnswering.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"task_type_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"start_positions",val:": typing.Optional[torch.Tensor] = None"},{name:"end_positions",val:": typing.Optional[torch.Tensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.ErnieForQuestionAnswering.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.ErnieForQuestionAnswering.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.ErnieForQuestionAnswering.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.ErnieForQuestionAnswering.forward.task_type_ids",description:`<strong>task_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Task type embedding is a special embedding to represent the characteristic of different tasks, such as
word-aware pre-training task, structure-aware pre-training task and semantic-aware pre-training task. We
assign a <code>task_type_id</code> to each task and the <code>task_type_id</code> is in the range \`[0,
config.task_type_vocab_size-1]`,name:"task_type_ids"},{anchor:"transformers.ErnieForQuestionAnswering.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.ErnieForQuestionAnswering.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.ErnieForQuestionAnswering.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.ErnieForQuestionAnswering.forward.start_positions",description:`<strong>start_positions</strong> (<code>torch.Tensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for position (index) of the start of the labelled span for computing the token classification loss.
Positions are clamped to the length of the sequence (<code>sequence_length</code>). Position outside of the sequence
are not taken into account for computing the loss.`,name:"start_positions"},{anchor:"transformers.ErnieForQuestionAnswering.forward.end_positions",description:`<strong>end_positions</strong> (<code>torch.Tensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for position (index) of the end of the labelled span for computing the token classification loss.
Positions are clamped to the length of the sequence (<code>sequence_length</code>). Position outside of the sequence
are not taken into account for computing the loss.`,name:"end_positions"},{anchor:"transformers.ErnieForQuestionAnswering.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.ErnieForQuestionAnswering.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.ErnieForQuestionAnswering.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/ernie/modeling_ernie.py#L1603",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.QuestionAnsweringModelOutput"
>transformers.modeling_outputs.QuestionAnsweringModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/ernie#transformers.ErnieConfig"
>ErnieConfig</a>) and inputs.</p>
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
`}}),qe=new fe({props:{$$slots:{default:[Vr]},$$scope:{ctx:w}}}),Be=new ge({props:{anchor:"transformers.ErnieForQuestionAnswering.forward.example",$$slots:{default:[Hr]},$$scope:{ctx:w}}}),Et=new gr({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/ernie.md"}}),{c(){t=p("meta"),h=a(),o=p("p"),r=a(),k=p("p"),k.innerHTML=n,u=a(),E=p("div"),E.innerHTML=gn,Ve=a(),f(he.$$.fragment),bn=a(),He=p("p"),He.innerHTML=gs,yn=a(),Re=p("p"),Re.textContent=_s,Tn=a(),Le=p("p"),Le.innerHTML=bs,Mn=a(),f(ye.$$.fragment),kn=a(),Ge=p("p"),Ge.innerHTML=ys,wn=a(),f(Te.$$.fragment),vn=a(),f(Se.$$.fragment),$n=a(),Pe=p("p"),Pe.textContent=Ts,Jn=a(),Xe=p("table"),Xe.innerHTML=Ms,En=a(),f(Qe.$$.fragment),Cn=a(),Ae=p("p"),Ae.innerHTML=ks,jn=a(),f(Oe.$$.fragment),zn=a(),H=p("div"),f(Ye.$$.fragment),eo=a(),jt=p("p"),jt.innerHTML=ws,to=a(),zt=p("p"),zt.innerHTML=vs,no=a(),f(Me.$$.fragment),Un=a(),f(De.$$.fragment),xn=a(),_e=p("div"),f(Ke.$$.fragment),oo=a(),Ut=p("p"),Ut.innerHTML=$s,Fn=a(),f(et.$$.fragment),In=a(),j=p("div"),f(tt.$$.fragment),so=a(),xt=p("p"),xt.innerHTML=Js,ro=a(),Ft=p("p"),Ft.innerHTML=Es,ao=a(),It=p("p"),It.innerHTML=Cs,io=a(),Wt=p("p"),Wt.innerHTML=js,lo=a(),ue=p("div"),f(nt.$$.fragment),co=a(),Zt=p("p"),Zt.innerHTML=zs,po=a(),f(ke.$$.fragment),Wn=a(),f(ot.$$.fragment),Zn=a(),z=p("div"),f(st.$$.fragment),mo=a(),Nt=p("p"),Nt.innerHTML=Us,ho=a(),qt=p("p"),qt.innerHTML=xs,uo=a(),Bt=p("p"),Bt.innerHTML=Fs,fo=a(),D=p("div"),f(rt.$$.fragment),go=a(),Vt=p("p"),Vt.innerHTML=Is,_o=a(),f(we.$$.fragment),bo=a(),f(ve.$$.fragment),Nn=a(),f(at.$$.fragment),qn=a(),U=p("div"),f(it.$$.fragment),yo=a(),Ht=p("p"),Ht.innerHTML=Ws,To=a(),Rt=p("p"),Rt.innerHTML=Zs,Mo=a(),Lt=p("p"),Lt.innerHTML=Ns,ko=a(),K=p("div"),f(lt.$$.fragment),wo=a(),Gt=p("p"),Gt.innerHTML=qs,vo=a(),f($e.$$.fragment),$o=a(),f(Je.$$.fragment),Bn=a(),f(dt.$$.fragment),Vn=a(),x=p("div"),f(ct.$$.fragment),Jo=a(),St=p("p"),St.innerHTML=Bs,Eo=a(),Pt=p("p"),Pt.innerHTML=Vs,Co=a(),Xt=p("p"),Xt.innerHTML=Hs,jo=a(),ee=p("div"),f(pt.$$.fragment),zo=a(),Qt=p("p"),Qt.innerHTML=Rs,Uo=a(),f(Ee.$$.fragment),xo=a(),f(Ce.$$.fragment),Hn=a(),f(mt.$$.fragment),Rn=a(),F=p("div"),f(ht.$$.fragment),Fo=a(),At=p("p"),At.innerHTML=Ls,Io=a(),Ot=p("p"),Ot.innerHTML=Gs,Wo=a(),Yt=p("p"),Yt.innerHTML=Ss,Zo=a(),te=p("div"),f(ut.$$.fragment),No=a(),Dt=p("p"),Dt.innerHTML=Ps,qo=a(),f(je.$$.fragment),Bo=a(),f(ze.$$.fragment),Ln=a(),f(ft.$$.fragment),Gn=a(),I=p("div"),f(gt.$$.fragment),Vo=a(),Kt=p("p"),Kt.textContent=Xs,Ho=a(),en=p("p"),en.innerHTML=Qs,Ro=a(),tn=p("p"),tn.innerHTML=As,Lo=a(),B=p("div"),f(_t.$$.fragment),Go=a(),nn=p("p"),nn.innerHTML=Os,So=a(),f(Ue.$$.fragment),Po=a(),f(xe.$$.fragment),Xo=a(),f(Fe.$$.fragment),Sn=a(),f(bt.$$.fragment),Pn=a(),W=p("div"),f(yt.$$.fragment),Qo=a(),on=p("p"),on.textContent=Ys,Ao=a(),sn=p("p"),sn.innerHTML=Ds,Oo=a(),rn=p("p"),rn.innerHTML=Ks,Yo=a(),ne=p("div"),f(Tt.$$.fragment),Do=a(),an=p("p"),an.innerHTML=er,Ko=a(),f(Ie.$$.fragment),es=a(),f(We.$$.fragment),Xn=a(),f(Mt.$$.fragment),Qn=a(),Z=p("div"),f(kt.$$.fragment),ts=a(),ln=p("p"),ln.textContent=tr,ns=a(),dn=p("p"),dn.innerHTML=nr,os=a(),cn=p("p"),cn.innerHTML=or,ss=a(),oe=p("div"),f(wt.$$.fragment),rs=a(),pn=p("p"),pn.innerHTML=sr,as=a(),f(Ze.$$.fragment),is=a(),f(Ne.$$.fragment),An=a(),f(vt.$$.fragment),On=a(),N=p("div"),f($t.$$.fragment),ls=a(),mn=p("p"),mn.innerHTML=rr,ds=a(),hn=p("p"),hn.innerHTML=ar,cs=a(),un=p("p"),un.innerHTML=ir,ps=a(),se=p("div"),f(Jt.$$.fragment),ms=a(),fn=p("p"),fn.innerHTML=lr,hs=a(),f(qe.$$.fragment),us=a(),f(Be.$$.fragment),Yn=a(),f(Et.$$.fragment),Dn=a(),_n=p("p"),this.h()},l(e){const s=ur("svelte-u9bgzb",document.head);t=m(s,"META",{name:!0,content:!0}),s.forEach(l),h=i(e),o=m(e,"P",{}),$(o).forEach(l),r=i(e),k=m(e,"P",{"data-svelte-h":!0}),M(k)!=="svelte-fydc9c"&&(k.innerHTML=n),u=i(e),E=m(e,"DIV",{style:!0,"data-svelte-h":!0}),M(E)!=="svelte-wa5t4p"&&(E.innerHTML=gn),Ve=i(e),g(he.$$.fragment,e),bn=i(e),He=m(e,"P",{"data-svelte-h":!0}),M(He)!=="svelte-u0ycqt"&&(He.innerHTML=gs),yn=i(e),Re=m(e,"P",{"data-svelte-h":!0}),M(Re)!=="svelte-j6x9xg"&&(Re.textContent=_s),Tn=i(e),Le=m(e,"P",{"data-svelte-h":!0}),M(Le)!=="svelte-phe9o0"&&(Le.innerHTML=bs),Mn=i(e),g(ye.$$.fragment,e),kn=i(e),Ge=m(e,"P",{"data-svelte-h":!0}),M(Ge)!=="svelte-lqa8w5"&&(Ge.innerHTML=ys),wn=i(e),g(Te.$$.fragment,e),vn=i(e),g(Se.$$.fragment,e),$n=i(e),Pe=m(e,"P",{"data-svelte-h":!0}),M(Pe)!=="svelte-13gdvs0"&&(Pe.textContent=Ts),Jn=i(e),Xe=m(e,"TABLE",{"data-svelte-h":!0}),M(Xe)!=="svelte-nh20hi"&&(Xe.innerHTML=Ms),En=i(e),g(Qe.$$.fragment,e),Cn=i(e),Ae=m(e,"P",{"data-svelte-h":!0}),M(Ae)!=="svelte-82tm21"&&(Ae.innerHTML=ks),jn=i(e),g(Oe.$$.fragment,e),zn=i(e),H=m(e,"DIV",{class:!0});var re=$(H);g(Ye.$$.fragment,re),eo=i(re),jt=m(re,"P",{"data-svelte-h":!0}),M(jt)!=="svelte-f9dbb9"&&(jt.innerHTML=ws),to=i(re),zt=m(re,"P",{"data-svelte-h":!0}),M(zt)!=="svelte-1ek1ss9"&&(zt.innerHTML=vs),no=i(re),g(Me.$$.fragment,re),re.forEach(l),Un=i(e),g(De.$$.fragment,e),xn=i(e),_e=m(e,"DIV",{class:!0});var Ct=$(_e);g(Ke.$$.fragment,Ct),oo=i(Ct),Ut=m(Ct,"P",{"data-svelte-h":!0}),M(Ut)!=="svelte-1rrxzb6"&&(Ut.innerHTML=$s),Ct.forEach(l),Fn=i(e),g(et.$$.fragment,e),In=i(e),j=m(e,"DIV",{class:!0});var q=$(j);g(tt.$$.fragment,q),so=i(q),xt=m(q,"P",{"data-svelte-h":!0}),M(xt)!=="svelte-1854dma"&&(xt.innerHTML=Js),ro=i(q),Ft=m(q,"P",{"data-svelte-h":!0}),M(Ft)!=="svelte-xh6zo5"&&(Ft.innerHTML=Es),ao=i(q),It=m(q,"P",{"data-svelte-h":!0}),M(It)!=="svelte-q52n56"&&(It.innerHTML=Cs),io=i(q),Wt=m(q,"P",{"data-svelte-h":!0}),M(Wt)!=="svelte-hswkmf"&&(Wt.innerHTML=js),lo=i(q),ue=m(q,"DIV",{class:!0});var be=$(ue);g(nt.$$.fragment,be),co=i(be),Zt=m(be,"P",{"data-svelte-h":!0}),M(Zt)!=="svelte-16yf1pw"&&(Zt.innerHTML=zs),po=i(be),g(ke.$$.fragment,be),be.forEach(l),q.forEach(l),Wn=i(e),g(ot.$$.fragment,e),Zn=i(e),z=m(e,"DIV",{class:!0});var R=$(z);g(st.$$.fragment,R),mo=i(R),Nt=m(R,"P",{"data-svelte-h":!0}),M(Nt)!=="svelte-ltv6uf"&&(Nt.innerHTML=Us),ho=i(R),qt=m(R,"P",{"data-svelte-h":!0}),M(qt)!=="svelte-q52n56"&&(qt.innerHTML=xs),uo=i(R),Bt=m(R,"P",{"data-svelte-h":!0}),M(Bt)!=="svelte-hswkmf"&&(Bt.innerHTML=Fs),fo=i(R),D=m(R,"DIV",{class:!0});var ae=$(D);g(rt.$$.fragment,ae),go=i(ae),Vt=m(ae,"P",{"data-svelte-h":!0}),M(Vt)!=="svelte-15s2c0s"&&(Vt.innerHTML=Is),_o=i(ae),g(we.$$.fragment,ae),bo=i(ae),g(ve.$$.fragment,ae),ae.forEach(l),R.forEach(l),Nn=i(e),g(at.$$.fragment,e),qn=i(e),U=m(e,"DIV",{class:!0});var L=$(U);g(it.$$.fragment,L),yo=i(L),Ht=m(L,"P",{"data-svelte-h":!0}),M(Ht)!=="svelte-ts4qhc"&&(Ht.innerHTML=Ws),To=i(L),Rt=m(L,"P",{"data-svelte-h":!0}),M(Rt)!=="svelte-q52n56"&&(Rt.innerHTML=Zs),Mo=i(L),Lt=m(L,"P",{"data-svelte-h":!0}),M(Lt)!=="svelte-hswkmf"&&(Lt.innerHTML=Ns),ko=i(L),K=m(L,"DIV",{class:!0});var ie=$(K);g(lt.$$.fragment,ie),wo=i(ie),Gt=m(ie,"P",{"data-svelte-h":!0}),M(Gt)!=="svelte-6dtozg"&&(Gt.innerHTML=qs),vo=i(ie),g($e.$$.fragment,ie),$o=i(ie),g(Je.$$.fragment,ie),ie.forEach(l),L.forEach(l),Bn=i(e),g(dt.$$.fragment,e),Vn=i(e),x=m(e,"DIV",{class:!0});var G=$(x);g(ct.$$.fragment,G),Jo=i(G),St=m(G,"P",{"data-svelte-h":!0}),M(St)!=="svelte-1ip00m3"&&(St.innerHTML=Bs),Eo=i(G),Pt=m(G,"P",{"data-svelte-h":!0}),M(Pt)!=="svelte-q52n56"&&(Pt.innerHTML=Vs),Co=i(G),Xt=m(G,"P",{"data-svelte-h":!0}),M(Xt)!=="svelte-hswkmf"&&(Xt.innerHTML=Hs),jo=i(G),ee=m(G,"DIV",{class:!0});var le=$(ee);g(pt.$$.fragment,le),zo=i(le),Qt=m(le,"P",{"data-svelte-h":!0}),M(Qt)!=="svelte-1cctwk4"&&(Qt.innerHTML=Rs),Uo=i(le),g(Ee.$$.fragment,le),xo=i(le),g(Ce.$$.fragment,le),le.forEach(l),G.forEach(l),Hn=i(e),g(mt.$$.fragment,e),Rn=i(e),F=m(e,"DIV",{class:!0});var S=$(F);g(ht.$$.fragment,S),Fo=i(S),At=m(S,"P",{"data-svelte-h":!0}),M(At)!=="svelte-139sre"&&(At.innerHTML=Ls),Io=i(S),Ot=m(S,"P",{"data-svelte-h":!0}),M(Ot)!=="svelte-q52n56"&&(Ot.innerHTML=Gs),Wo=i(S),Yt=m(S,"P",{"data-svelte-h":!0}),M(Yt)!=="svelte-hswkmf"&&(Yt.innerHTML=Ss),Zo=i(S),te=m(S,"DIV",{class:!0});var de=$(te);g(ut.$$.fragment,de),No=i(de),Dt=m(de,"P",{"data-svelte-h":!0}),M(Dt)!=="svelte-dheuti"&&(Dt.innerHTML=Ps),qo=i(de),g(je.$$.fragment,de),Bo=i(de),g(ze.$$.fragment,de),de.forEach(l),S.forEach(l),Ln=i(e),g(ft.$$.fragment,e),Gn=i(e),I=m(e,"DIV",{class:!0});var P=$(I);g(gt.$$.fragment,P),Vo=i(P),Kt=m(P,"P",{"data-svelte-h":!0}),M(Kt)!=="svelte-129pehz"&&(Kt.textContent=Xs),Ho=i(P),en=m(P,"P",{"data-svelte-h":!0}),M(en)!=="svelte-q52n56"&&(en.innerHTML=Qs),Ro=i(P),tn=m(P,"P",{"data-svelte-h":!0}),M(tn)!=="svelte-hswkmf"&&(tn.innerHTML=As),Lo=i(P),B=m(P,"DIV",{class:!0});var X=$(B);g(_t.$$.fragment,X),Go=i(X),nn=m(X,"P",{"data-svelte-h":!0}),M(nn)!=="svelte-26wlcq"&&(nn.innerHTML=Os),So=i(X),g(Ue.$$.fragment,X),Po=i(X),g(xe.$$.fragment,X),Xo=i(X),g(Fe.$$.fragment,X),X.forEach(l),P.forEach(l),Sn=i(e),g(bt.$$.fragment,e),Pn=i(e),W=m(e,"DIV",{class:!0});var Q=$(W);g(yt.$$.fragment,Q),Qo=i(Q),on=m(Q,"P",{"data-svelte-h":!0}),M(on)!=="svelte-lxm2j4"&&(on.textContent=Ys),Ao=i(Q),sn=m(Q,"P",{"data-svelte-h":!0}),M(sn)!=="svelte-q52n56"&&(sn.innerHTML=Ds),Oo=i(Q),rn=m(Q,"P",{"data-svelte-h":!0}),M(rn)!=="svelte-hswkmf"&&(rn.innerHTML=Ks),Yo=i(Q),ne=m(Q,"DIV",{class:!0});var ce=$(ne);g(Tt.$$.fragment,ce),Do=i(ce),an=m(ce,"P",{"data-svelte-h":!0}),M(an)!=="svelte-1rw1zqu"&&(an.innerHTML=er),Ko=i(ce),g(Ie.$$.fragment,ce),es=i(ce),g(We.$$.fragment,ce),ce.forEach(l),Q.forEach(l),Xn=i(e),g(Mt.$$.fragment,e),Qn=i(e),Z=m(e,"DIV",{class:!0});var A=$(Z);g(kt.$$.fragment,A),ts=i(A),ln=m(A,"P",{"data-svelte-h":!0}),M(ln)!=="svelte-3jxh3z"&&(ln.textContent=tr),ns=i(A),dn=m(A,"P",{"data-svelte-h":!0}),M(dn)!=="svelte-q52n56"&&(dn.innerHTML=nr),os=i(A),cn=m(A,"P",{"data-svelte-h":!0}),M(cn)!=="svelte-hswkmf"&&(cn.innerHTML=or),ss=i(A),oe=m(A,"DIV",{class:!0});var pe=$(oe);g(wt.$$.fragment,pe),rs=i(pe),pn=m(pe,"P",{"data-svelte-h":!0}),M(pn)!=="svelte-mac77k"&&(pn.innerHTML=sr),as=i(pe),g(Ze.$$.fragment,pe),is=i(pe),g(Ne.$$.fragment,pe),pe.forEach(l),A.forEach(l),An=i(e),g(vt.$$.fragment,e),On=i(e),N=m(e,"DIV",{class:!0});var O=$(N);g($t.$$.fragment,O),ls=i(O),mn=m(O,"P",{"data-svelte-h":!0}),M(mn)!=="svelte-1re1wto"&&(mn.innerHTML=rr),ds=i(O),hn=m(O,"P",{"data-svelte-h":!0}),M(hn)!=="svelte-q52n56"&&(hn.innerHTML=ar),cs=i(O),un=m(O,"P",{"data-svelte-h":!0}),M(un)!=="svelte-hswkmf"&&(un.innerHTML=ir),ps=i(O),se=m(O,"DIV",{class:!0});var me=$(se);g(Jt.$$.fragment,me),ms=i(me),fn=m(me,"P",{"data-svelte-h":!0}),M(fn)!=="svelte-13ypg1g"&&(fn.innerHTML=lr),hs=i(me),g(qe.$$.fragment,me),us=i(me),g(Be.$$.fragment,me),me.forEach(l),O.forEach(l),Yn=i(e),g(Et.$$.fragment,e),Dn=i(e),_n=m(e,"P",{}),$(_n).forEach(l),this.h()},h(){J(t,"name","hf:doc:metadata"),J(t,"content",Lr),fr(E,"float","right"),J(H,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(_e,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(ue,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(j,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(D,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(z,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(K,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(U,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(ee,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(x,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(te,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(F,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(B,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(I,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(ne,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(W,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(oe,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(Z,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(se,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(N,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,s){d(document.head,t),c(e,h,s),c(e,o,s),c(e,r,s),c(e,k,s),c(e,u,s),c(e,E,s),c(e,Ve,s),_(he,e,s),c(e,bn,s),c(e,He,s),c(e,yn,s),c(e,Re,s),c(e,Tn,s),c(e,Le,s),c(e,Mn,s),_(ye,e,s),c(e,kn,s),c(e,Ge,s),c(e,wn,s),_(Te,e,s),c(e,vn,s),_(Se,e,s),c(e,$n,s),c(e,Pe,s),c(e,Jn,s),c(e,Xe,s),c(e,En,s),_(Qe,e,s),c(e,Cn,s),c(e,Ae,s),c(e,jn,s),_(Oe,e,s),c(e,zn,s),c(e,H,s),_(Ye,H,null),d(H,eo),d(H,jt),d(H,to),d(H,zt),d(H,no),_(Me,H,null),c(e,Un,s),_(De,e,s),c(e,xn,s),c(e,_e,s),_(Ke,_e,null),d(_e,oo),d(_e,Ut),c(e,Fn,s),_(et,e,s),c(e,In,s),c(e,j,s),_(tt,j,null),d(j,so),d(j,xt),d(j,ro),d(j,Ft),d(j,ao),d(j,It),d(j,io),d(j,Wt),d(j,lo),d(j,ue),_(nt,ue,null),d(ue,co),d(ue,Zt),d(ue,po),_(ke,ue,null),c(e,Wn,s),_(ot,e,s),c(e,Zn,s),c(e,z,s),_(st,z,null),d(z,mo),d(z,Nt),d(z,ho),d(z,qt),d(z,uo),d(z,Bt),d(z,fo),d(z,D),_(rt,D,null),d(D,go),d(D,Vt),d(D,_o),_(we,D,null),d(D,bo),_(ve,D,null),c(e,Nn,s),_(at,e,s),c(e,qn,s),c(e,U,s),_(it,U,null),d(U,yo),d(U,Ht),d(U,To),d(U,Rt),d(U,Mo),d(U,Lt),d(U,ko),d(U,K),_(lt,K,null),d(K,wo),d(K,Gt),d(K,vo),_($e,K,null),d(K,$o),_(Je,K,null),c(e,Bn,s),_(dt,e,s),c(e,Vn,s),c(e,x,s),_(ct,x,null),d(x,Jo),d(x,St),d(x,Eo),d(x,Pt),d(x,Co),d(x,Xt),d(x,jo),d(x,ee),_(pt,ee,null),d(ee,zo),d(ee,Qt),d(ee,Uo),_(Ee,ee,null),d(ee,xo),_(Ce,ee,null),c(e,Hn,s),_(mt,e,s),c(e,Rn,s),c(e,F,s),_(ht,F,null),d(F,Fo),d(F,At),d(F,Io),d(F,Ot),d(F,Wo),d(F,Yt),d(F,Zo),d(F,te),_(ut,te,null),d(te,No),d(te,Dt),d(te,qo),_(je,te,null),d(te,Bo),_(ze,te,null),c(e,Ln,s),_(ft,e,s),c(e,Gn,s),c(e,I,s),_(gt,I,null),d(I,Vo),d(I,Kt),d(I,Ho),d(I,en),d(I,Ro),d(I,tn),d(I,Lo),d(I,B),_(_t,B,null),d(B,Go),d(B,nn),d(B,So),_(Ue,B,null),d(B,Po),_(xe,B,null),d(B,Xo),_(Fe,B,null),c(e,Sn,s),_(bt,e,s),c(e,Pn,s),c(e,W,s),_(yt,W,null),d(W,Qo),d(W,on),d(W,Ao),d(W,sn),d(W,Oo),d(W,rn),d(W,Yo),d(W,ne),_(Tt,ne,null),d(ne,Do),d(ne,an),d(ne,Ko),_(Ie,ne,null),d(ne,es),_(We,ne,null),c(e,Xn,s),_(Mt,e,s),c(e,Qn,s),c(e,Z,s),_(kt,Z,null),d(Z,ts),d(Z,ln),d(Z,ns),d(Z,dn),d(Z,os),d(Z,cn),d(Z,ss),d(Z,oe),_(wt,oe,null),d(oe,rs),d(oe,pn),d(oe,as),_(Ze,oe,null),d(oe,is),_(Ne,oe,null),c(e,An,s),_(vt,e,s),c(e,On,s),c(e,N,s),_($t,N,null),d(N,ls),d(N,mn),d(N,ds),d(N,hn),d(N,cs),d(N,un),d(N,ps),d(N,se),_(Jt,se,null),d(se,ms),d(se,fn),d(se,hs),_(qe,se,null),d(se,us),_(Be,se,null),c(e,Yn,s),_(Et,e,s),c(e,Dn,s),c(e,_n,s),Kn=!0},p(e,[s]){const re={};s&2&&(re.$$scope={dirty:s,ctx:e}),ye.$set(re);const Ct={};s&2&&(Ct.$$scope={dirty:s,ctx:e}),Te.$set(Ct);const q={};s&2&&(q.$$scope={dirty:s,ctx:e}),Me.$set(q);const be={};s&2&&(be.$$scope={dirty:s,ctx:e}),ke.$set(be);const R={};s&2&&(R.$$scope={dirty:s,ctx:e}),we.$set(R);const ae={};s&2&&(ae.$$scope={dirty:s,ctx:e}),ve.$set(ae);const L={};s&2&&(L.$$scope={dirty:s,ctx:e}),$e.$set(L);const ie={};s&2&&(ie.$$scope={dirty:s,ctx:e}),Je.$set(ie);const G={};s&2&&(G.$$scope={dirty:s,ctx:e}),Ee.$set(G);const le={};s&2&&(le.$$scope={dirty:s,ctx:e}),Ce.$set(le);const S={};s&2&&(S.$$scope={dirty:s,ctx:e}),je.$set(S);const de={};s&2&&(de.$$scope={dirty:s,ctx:e}),ze.$set(de);const P={};s&2&&(P.$$scope={dirty:s,ctx:e}),Ue.$set(P);const X={};s&2&&(X.$$scope={dirty:s,ctx:e}),xe.$set(X);const Q={};s&2&&(Q.$$scope={dirty:s,ctx:e}),Fe.$set(Q);const ce={};s&2&&(ce.$$scope={dirty:s,ctx:e}),Ie.$set(ce);const A={};s&2&&(A.$$scope={dirty:s,ctx:e}),We.$set(A);const pe={};s&2&&(pe.$$scope={dirty:s,ctx:e}),Ze.$set(pe);const O={};s&2&&(O.$$scope={dirty:s,ctx:e}),Ne.$set(O);const me={};s&2&&(me.$$scope={dirty:s,ctx:e}),qe.$set(me);const dr={};s&2&&(dr.$$scope={dirty:s,ctx:e}),Be.$set(dr)},i(e){Kn||(b(he.$$.fragment,e),b(ye.$$.fragment,e),b(Te.$$.fragment,e),b(Se.$$.fragment,e),b(Qe.$$.fragment,e),b(Oe.$$.fragment,e),b(Ye.$$.fragment,e),b(Me.$$.fragment,e),b(De.$$.fragment,e),b(Ke.$$.fragment,e),b(et.$$.fragment,e),b(tt.$$.fragment,e),b(nt.$$.fragment,e),b(ke.$$.fragment,e),b(ot.$$.fragment,e),b(st.$$.fragment,e),b(rt.$$.fragment,e),b(we.$$.fragment,e),b(ve.$$.fragment,e),b(at.$$.fragment,e),b(it.$$.fragment,e),b(lt.$$.fragment,e),b($e.$$.fragment,e),b(Je.$$.fragment,e),b(dt.$$.fragment,e),b(ct.$$.fragment,e),b(pt.$$.fragment,e),b(Ee.$$.fragment,e),b(Ce.$$.fragment,e),b(mt.$$.fragment,e),b(ht.$$.fragment,e),b(ut.$$.fragment,e),b(je.$$.fragment,e),b(ze.$$.fragment,e),b(ft.$$.fragment,e),b(gt.$$.fragment,e),b(_t.$$.fragment,e),b(Ue.$$.fragment,e),b(xe.$$.fragment,e),b(Fe.$$.fragment,e),b(bt.$$.fragment,e),b(yt.$$.fragment,e),b(Tt.$$.fragment,e),b(Ie.$$.fragment,e),b(We.$$.fragment,e),b(Mt.$$.fragment,e),b(kt.$$.fragment,e),b(wt.$$.fragment,e),b(Ze.$$.fragment,e),b(Ne.$$.fragment,e),b(vt.$$.fragment,e),b($t.$$.fragment,e),b(Jt.$$.fragment,e),b(qe.$$.fragment,e),b(Be.$$.fragment,e),b(Et.$$.fragment,e),Kn=!0)},o(e){y(he.$$.fragment,e),y(ye.$$.fragment,e),y(Te.$$.fragment,e),y(Se.$$.fragment,e),y(Qe.$$.fragment,e),y(Oe.$$.fragment,e),y(Ye.$$.fragment,e),y(Me.$$.fragment,e),y(De.$$.fragment,e),y(Ke.$$.fragment,e),y(et.$$.fragment,e),y(tt.$$.fragment,e),y(nt.$$.fragment,e),y(ke.$$.fragment,e),y(ot.$$.fragment,e),y(st.$$.fragment,e),y(rt.$$.fragment,e),y(we.$$.fragment,e),y(ve.$$.fragment,e),y(at.$$.fragment,e),y(it.$$.fragment,e),y(lt.$$.fragment,e),y($e.$$.fragment,e),y(Je.$$.fragment,e),y(dt.$$.fragment,e),y(ct.$$.fragment,e),y(pt.$$.fragment,e),y(Ee.$$.fragment,e),y(Ce.$$.fragment,e),y(mt.$$.fragment,e),y(ht.$$.fragment,e),y(ut.$$.fragment,e),y(je.$$.fragment,e),y(ze.$$.fragment,e),y(ft.$$.fragment,e),y(gt.$$.fragment,e),y(_t.$$.fragment,e),y(Ue.$$.fragment,e),y(xe.$$.fragment,e),y(Fe.$$.fragment,e),y(bt.$$.fragment,e),y(yt.$$.fragment,e),y(Tt.$$.fragment,e),y(Ie.$$.fragment,e),y(We.$$.fragment,e),y(Mt.$$.fragment,e),y(kt.$$.fragment,e),y(wt.$$.fragment,e),y(Ze.$$.fragment,e),y(Ne.$$.fragment,e),y(vt.$$.fragment,e),y($t.$$.fragment,e),y(Jt.$$.fragment,e),y(qe.$$.fragment,e),y(Be.$$.fragment,e),y(Et.$$.fragment,e),Kn=!1},d(e){e&&(l(h),l(o),l(r),l(k),l(u),l(E),l(Ve),l(bn),l(He),l(yn),l(Re),l(Tn),l(Le),l(Mn),l(kn),l(Ge),l(wn),l(vn),l($n),l(Pe),l(Jn),l(Xe),l(En),l(Cn),l(Ae),l(jn),l(zn),l(H),l(Un),l(xn),l(_e),l(Fn),l(In),l(j),l(Wn),l(Zn),l(z),l(Nn),l(qn),l(U),l(Bn),l(Vn),l(x),l(Hn),l(Rn),l(F),l(Ln),l(Gn),l(I),l(Sn),l(Pn),l(W),l(Xn),l(Qn),l(Z),l(An),l(On),l(N),l(Yn),l(Dn),l(_n)),l(t),T(he,e),T(ye,e),T(Te,e),T(Se,e),T(Qe,e),T(Oe,e),T(Ye),T(Me),T(De,e),T(Ke),T(et,e),T(tt),T(nt),T(ke),T(ot,e),T(st),T(rt),T(we),T(ve),T(at,e),T(it),T(lt),T($e),T(Je),T(dt,e),T(ct),T(pt),T(Ee),T(Ce),T(mt,e),T(ht),T(ut),T(je),T(ze),T(ft,e),T(gt),T(_t),T(Ue),T(xe),T(Fe),T(bt,e),T(yt),T(Tt),T(Ie),T(We),T(Mt,e),T(kt),T(wt),T(Ze),T(Ne),T(vt,e),T($t),T(Jt),T(qe),T(Be),T(Et,e)}}}const Lr='{"title":"ERNIE","local":"ernie","sections":[{"title":"Notes","local":"notes","sections":[],"depth":2},{"title":"Resources","local":"resources","sections":[],"depth":2},{"title":"ErnieConfig","local":"transformers.ErnieConfig","sections":[],"depth":2},{"title":"Ernie specific outputs","local":"transformers.models.ernie.modeling_ernie.ErnieForPreTrainingOutput","sections":[],"depth":2},{"title":"ErnieModel","local":"transformers.ErnieModel","sections":[],"depth":2},{"title":"ErnieForPreTraining","local":"transformers.ErnieForPreTraining","sections":[],"depth":2},{"title":"ErnieForCausalLM","local":"transformers.ErnieForCausalLM","sections":[],"depth":2},{"title":"ErnieForMaskedLM","local":"transformers.ErnieForMaskedLM","sections":[],"depth":2},{"title":"ErnieForNextSentencePrediction","local":"transformers.ErnieForNextSentencePrediction","sections":[],"depth":2},{"title":"ErnieForSequenceClassification","local":"transformers.ErnieForSequenceClassification","sections":[],"depth":2},{"title":"ErnieForMultipleChoice","local":"transformers.ErnieForMultipleChoice","sections":[],"depth":2},{"title":"ErnieForTokenClassification","local":"transformers.ErnieForTokenClassification","sections":[],"depth":2},{"title":"ErnieForQuestionAnswering","local":"transformers.ErnieForQuestionAnswering","sections":[],"depth":2}],"depth":1}';function Gr(w){return pr(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class Kr extends mr{constructor(t){super(),hr(this,t,Gr,Rr,cr,{})}}export{Kr as component};
