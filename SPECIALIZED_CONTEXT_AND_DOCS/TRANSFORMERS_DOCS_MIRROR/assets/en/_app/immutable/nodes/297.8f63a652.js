import{s as Jr,o as xr,n as x}from"../chunks/scheduler.18a86fab.js";import{S as Ur,i as jr,g as c,s as r,r as u,A as zr,h as p,f as l,c as a,j as v,x as y,u as f,k as $,l as Cr,y as s,a as m,v as g,d as b,t as _,w as M}from"../chunks/index.98837b22.js";import{T as _e}from"../chunks/Tip.77304350.js";import{D as B}from"../chunks/Docstring.a1ef7999.js";import{C as oe}from"../chunks/CodeBlock.8d0c2e8a.js";import{E as Me}from"../chunks/ExampleCodeBlock.8c3ee1f9.js";import{H as V,E as Fr}from"../chunks/getInferenceSnippets.06c2775f.js";import{H as Wr,a as zs}from"../chunks/HfOption.6641485e.js";function Zr(w){let t,h="Click on the MobileBERT models in the right sidebar for more examples of how to apply MobileBERT to different language tasks.";return{c(){t=c("p"),t.textContent=h},l(o){t=p(o,"P",{"data-svelte-h":!0}),y(t)!=="svelte-1o4gtlt"&&(t.textContent=h)},m(o,d){m(o,t,d)},p:x,d(o){o&&l(t)}}}function Ir(w){let t,h;return t=new oe({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwcGlwZWxpbmUlMEElMEFwaXBlbGluZSUyMCUzRCUyMHBpcGVsaW5lKCUwQSUyMCUyMCUyMCUyMHRhc2slM0QlMjJmaWxsLW1hc2slMjIlMkMlMEElMjAlMjAlMjAlMjBtb2RlbCUzRCUyMmdvb2dsZSUyRm1vYmlsZWJlcnQtdW5jYXNlZCUyMiUyQyUwQSUyMCUyMCUyMCUyMGR0eXBlJTNEdG9yY2guZmxvYXQxNiUyQyUwQSUyMCUyMCUyMCUyMGRldmljZSUzRDAlMEEpJTBBcGlwZWxpbmUoJTIyVGhlJTIwY2FwaXRhbCUyMG9mJTIwRnJhbmNlJTIwaXMlMjAlNUJNQVNLJTVELiUyMik=",highlighted:`<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> pipeline

pipeline = pipeline(
    task=<span class="hljs-string">&quot;fill-mask&quot;</span>,
    model=<span class="hljs-string">&quot;google/mobilebert-uncased&quot;</span>,
    dtype=torch.float16,
    device=<span class="hljs-number">0</span>
)
pipeline(<span class="hljs-string">&quot;The capital of France is [MASK].&quot;</span>)`,wrap:!1}}),{c(){u(t.$$.fragment)},l(o){f(t.$$.fragment,o)},m(o,d){g(t,o,d),h=!0},p:x,i(o){h||(b(t.$$.fragment,o),h=!0)},o(o){_(t.$$.fragment,o),h=!1},d(o){M(t,o)}}}function qr(w){let t,h;return t=new oe({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b01vZGVsRm9yTWFza2VkTE0lMkMlMjBBdXRvVG9rZW5pemVyJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTBBJTIwJTIwJTIwJTIwJTIyZ29vZ2xlJTJGbW9iaWxlYmVydC11bmNhc2VkJTIyJTJDJTBBKSUwQW1vZGVsJTIwJTNEJTIwQXV0b01vZGVsRm9yTWFza2VkTE0uZnJvbV9wcmV0cmFpbmVkKCUwQSUyMCUyMCUyMCUyMCUyMmdvb2dsZSUyRm1vYmlsZWJlcnQtdW5jYXNlZCUyMiUyQyUwQSUyMCUyMCUyMCUyMGR0eXBlJTNEdG9yY2guZmxvYXQxNiUyQyUwQSUyMCUyMCUyMCUyMGRldmljZV9tYXAlM0QlMjJhdXRvJTIyJTJDJTBBKSUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplciglMjJUaGUlMjBjYXBpdGFsJTIwb2YlMjBGcmFuY2UlMjBpcyUyMCU1Qk1BU0slNUQuJTIyJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMikudG8obW9kZWwuZGV2aWNlKSUwQSUwQXdpdGglMjB0b3JjaC5ub19ncmFkKCklM0ElMEElMjAlMjAlMjAlMjBvdXRwdXRzJTIwJTNEJTIwbW9kZWwoKippbnB1dHMpJTBBJTIwJTIwJTIwJTIwcHJlZGljdGlvbnMlMjAlM0QlMjBvdXRwdXRzLmxvZ2l0cyUwQSUwQW1hc2tlZF9pbmRleCUyMCUzRCUyMHRvcmNoLndoZXJlKGlucHV0cyU1QidpbnB1dF9pZHMnJTVEJTIwJTNEJTNEJTIwdG9rZW5pemVyLm1hc2tfdG9rZW5faWQpJTVCMSU1RCUwQXByZWRpY3RlZF90b2tlbl9pZCUyMCUzRCUyMHByZWRpY3Rpb25zJTVCMCUyQyUyMG1hc2tlZF9pbmRleCU1RC5hcmdtYXgoZGltJTNELTEpJTBBcHJlZGljdGVkX3Rva2VuJTIwJTNEJTIwdG9rZW5pemVyLmRlY29kZShwcmVkaWN0ZWRfdG9rZW5faWQpJTBBJTBBcHJpbnQoZiUyMlRoZSUyMHByZWRpY3RlZCUyMHRva2VuJTIwaXMlM0ElMjAlN0JwcmVkaWN0ZWRfdG9rZW4lN0QlMjIp",highlighted:`<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoModelForMaskedLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    <span class="hljs-string">&quot;google/mobilebert-uncased&quot;</span>,
)
model = AutoModelForMaskedLM.from_pretrained(
    <span class="hljs-string">&quot;google/mobilebert-uncased&quot;</span>,
    dtype=torch.float16,
    device_map=<span class="hljs-string">&quot;auto&quot;</span>,
)
inputs = tokenizer(<span class="hljs-string">&quot;The capital of France is [MASK].&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>).to(model.device)

<span class="hljs-keyword">with</span> torch.no_grad():
    outputs = model(**inputs)
    predictions = outputs.logits

masked_index = torch.where(inputs[<span class="hljs-string">&#x27;input_ids&#x27;</span>] == tokenizer.mask_token_id)[<span class="hljs-number">1</span>]
predicted_token_id = predictions[<span class="hljs-number">0</span>, masked_index].argmax(dim=-<span class="hljs-number">1</span>)
predicted_token = tokenizer.decode(predicted_token_id)

<span class="hljs-built_in">print</span>(<span class="hljs-string">f&quot;The predicted token is: <span class="hljs-subst">{predicted_token}</span>&quot;</span>)`,wrap:!1}}),{c(){u(t.$$.fragment)},l(o){f(t.$$.fragment,o)},m(o,d){g(t,o,d),h=!0},p:x,i(o){h||(b(t.$$.fragment,o),h=!0)},o(o){_(t.$$.fragment,o),h=!1},d(o){M(t,o)}}}function Nr(w){let t,h;return t=new oe({props:{code:"ZWNobyUyMC1lJTIwJTIyVGhlJTIwY2FwaXRhbCUyMG9mJTIwRnJhbmNlJTIwaXMlMjAlNUJNQVNLJTVELiUyMiUyMCU3QyUyMHRyYW5zZm9ybWVycyUyMHJ1biUyMC0tdGFzayUyMGZpbGwtbWFzayUyMC0tbW9kZWwlMjBnb29nbGUlMkZtb2JpbGViZXJ0LXVuY2FzZWQlMjAtLWRldmljZSUyMDA=",highlighted:'<span class="hljs-built_in">echo</span> -e <span class="hljs-string">&quot;The capital of France is [MASK].&quot;</span> | transformers run --task fill-mask --model google/mobilebert-uncased --device 0',wrap:!1}}),{c(){u(t.$$.fragment)},l(o){f(t.$$.fragment,o)},m(o,d){g(t,o,d),h=!0},p:x,i(o){h||(b(t.$$.fragment,o),h=!0)},o(o){_(t.$$.fragment,o),h=!1},d(o){M(t,o)}}}function Rr(w){let t,h,o,d,k,n;return t=new zs({props:{id:"usage",option:"Pipeline",$$slots:{default:[Ir]},$$scope:{ctx:w}}}),o=new zs({props:{id:"usage",option:"AutoModel",$$slots:{default:[qr]},$$scope:{ctx:w}}}),k=new zs({props:{id:"usage",option:"transformers CLI",$$slots:{default:[Nr]},$$scope:{ctx:w}}}),{c(){u(t.$$.fragment),h=r(),u(o.$$.fragment),d=r(),u(k.$$.fragment)},l(T){f(t.$$.fragment,T),h=a(T),f(o.$$.fragment,T),d=a(T),f(k.$$.fragment,T)},m(T,J){g(t,T,J),m(T,h,J),g(o,T,J),m(T,d,J),g(k,T,J),n=!0},p(T,J){const xo={};J&2&&(xo.$$scope={dirty:J,ctx:T}),t.$set(xo);const He={};J&2&&(He.$$scope={dirty:J,ctx:T}),o.$set(He);const le={};J&2&&(le.$$scope={dirty:J,ctx:T}),k.$set(le)},i(T){n||(b(t.$$.fragment,T),b(o.$$.fragment,T),b(k.$$.fragment,T),n=!0)},o(T){_(t.$$.fragment,T),_(o.$$.fragment,T),_(k.$$.fragment,T),n=!1},d(T){T&&(l(h),l(d)),M(t,T),M(o,T),M(k,T)}}}function Vr(w){let t,h="Examples:",o,d,k;return d=new oe({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyME1vYmlsZUJlcnRDb25maWclMkMlMjBNb2JpbGVCZXJ0TW9kZWwlMEElMEElMjMlMjBJbml0aWFsaXppbmclMjBhJTIwTW9iaWxlQkVSVCUyMGNvbmZpZ3VyYXRpb24lMEFjb25maWd1cmF0aW9uJTIwJTNEJTIwTW9iaWxlQmVydENvbmZpZygpJTBBJTBBJTIzJTIwSW5pdGlhbGl6aW5nJTIwYSUyMG1vZGVsJTIwKHdpdGglMjByYW5kb20lMjB3ZWlnaHRzKSUyMGZyb20lMjB0aGUlMjBjb25maWd1cmF0aW9uJTIwYWJvdmUlMEFtb2RlbCUyMCUzRCUyME1vYmlsZUJlcnRNb2RlbChjb25maWd1cmF0aW9uKSUwQSUwQSUyMyUyMEFjY2Vzc2luZyUyMHRoZSUyMG1vZGVsJTIwY29uZmlndXJhdGlvbiUwQWNvbmZpZ3VyYXRpb24lMjAlM0QlMjBtb2RlbC5jb25maWc=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> MobileBertConfig, MobileBertModel

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a MobileBERT configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = MobileBertConfig()

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a model (with random weights) from the configuration above</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = MobileBertModel(configuration)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Accessing the model configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = model.config`,wrap:!1}}),{c(){t=c("p"),t.textContent=h,o=r(),u(d.$$.fragment)},l(n){t=p(n,"P",{"data-svelte-h":!0}),y(t)!=="svelte-kvfsh7"&&(t.textContent=h),o=a(n),f(d.$$.fragment,n)},m(n,T){m(n,t,T),m(n,o,T),g(d,n,T),k=!0},p:x,i(n){k||(b(d.$$.fragment,n),k=!0)},o(n){_(d.$$.fragment,n),k=!1},d(n){n&&(l(t),l(o)),M(d,n)}}}function Gr(w){let t,h=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=h},l(o){t=p(o,"P",{"data-svelte-h":!0}),y(t)!=="svelte-fincs2"&&(t.innerHTML=h)},m(o,d){m(o,t,d)},p:x,d(o){o&&l(t)}}}function Sr(w){let t,h=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=h},l(o){t=p(o,"P",{"data-svelte-h":!0}),y(t)!=="svelte-fincs2"&&(t.innerHTML=h)},m(o,d){m(o,t,d)},p:x,d(o){o&&l(t)}}}function Xr(w){let t,h="Examples:",o,d,k;return d=new oe({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBNb2JpbGVCZXJ0Rm9yUHJlVHJhaW5pbmclMEFpbXBvcnQlMjB0b3JjaCUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMmdvb2dsZSUyRm1vYmlsZWJlcnQtdW5jYXNlZCUyMiklMEFtb2RlbCUyMCUzRCUyME1vYmlsZUJlcnRGb3JQcmVUcmFpbmluZy5mcm9tX3ByZXRyYWluZWQoJTIyZ29vZ2xlJTJGbW9iaWxlYmVydC11bmNhc2VkJTIyKSUwQSUwQWlucHV0X2lkcyUyMCUzRCUyMHRvcmNoLnRlbnNvcih0b2tlbml6ZXIuZW5jb2RlKCUyMkhlbGxvJTJDJTIwbXklMjBkb2clMjBpcyUyMGN1dGUlMjIlMkMlMjBhZGRfc3BlY2lhbF90b2tlbnMlM0RUcnVlKSkudW5zcXVlZXplKDApJTBBJTIzJTIwQmF0Y2glMjBzaXplJTIwMSUwQW91dHB1dHMlMjAlM0QlMjBtb2RlbChpbnB1dF9pZHMpJTBBJTBBcHJlZGljdGlvbl9sb2dpdHMlMjAlM0QlMjBvdXRwdXRzLnByZWRpY3Rpb25fbG9naXRzJTBBc2VxX3JlbGF0aW9uc2hpcF9sb2dpdHMlMjAlM0QlMjBvdXRwdXRzLnNlcV9yZWxhdGlvbnNoaXBfbG9naXRz",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, MobileBertForPreTraining
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;google/mobilebert-uncased&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = MobileBertForPreTraining.from_pretrained(<span class="hljs-string">&quot;google/mobilebert-uncased&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>input_ids = torch.tensor(tokenizer.encode(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, add_special_tokens=<span class="hljs-literal">True</span>)).unsqueeze(<span class="hljs-number">0</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Batch size 1</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(input_ids)

<span class="hljs-meta">&gt;&gt;&gt; </span>prediction_logits = outputs.prediction_logits
<span class="hljs-meta">&gt;&gt;&gt; </span>seq_relationship_logits = outputs.seq_relationship_logits`,wrap:!1}}),{c(){t=c("p"),t.textContent=h,o=r(),u(d.$$.fragment)},l(n){t=p(n,"P",{"data-svelte-h":!0}),y(t)!=="svelte-kvfsh7"&&(t.textContent=h),o=a(n),f(d.$$.fragment,n)},m(n,T){m(n,t,T),m(n,o,T),g(d,n,T),k=!0},p:x,i(n){k||(b(d.$$.fragment,n),k=!0)},o(n){_(d.$$.fragment,n),k=!1},d(n){n&&(l(t),l(o)),M(d,n)}}}function Hr(w){let t,h=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=h},l(o){t=p(o,"P",{"data-svelte-h":!0}),y(t)!=="svelte-fincs2"&&(t.innerHTML=h)},m(o,d){m(o,t,d)},p:x,d(o){o&&l(t)}}}function Lr(w){let t,h="Example:",o,d,k;return d=new oe({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBNb2JpbGVCZXJ0Rm9yTWFza2VkTE0lMEFpbXBvcnQlMjB0b3JjaCUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMmdvb2dsZSUyRm1vYmlsZWJlcnQtdW5jYXNlZCUyMiklMEFtb2RlbCUyMCUzRCUyME1vYmlsZUJlcnRGb3JNYXNrZWRMTS5mcm9tX3ByZXRyYWluZWQoJTIyZ29vZ2xlJTJGbW9iaWxlYmVydC11bmNhc2VkJTIyKSUwQSUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplciglMjJUaGUlMjBjYXBpdGFsJTIwb2YlMjBGcmFuY2UlMjBpcyUyMCUzQ21hc2slM0UuJTIyJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiklMEElMEF3aXRoJTIwdG9yY2gubm9fZ3JhZCgpJTNBJTBBJTIwJTIwJTIwJTIwbG9naXRzJTIwJTNEJTIwbW9kZWwoKippbnB1dHMpLmxvZ2l0cyUwQSUwQSUyMyUyMHJldHJpZXZlJTIwaW5kZXglMjBvZiUyMCUzQ21hc2slM0UlMEFtYXNrX3Rva2VuX2luZGV4JTIwJTNEJTIwKGlucHV0cy5pbnB1dF9pZHMlMjAlM0QlM0QlMjB0b2tlbml6ZXIubWFza190b2tlbl9pZCklNUIwJTVELm5vbnplcm8oYXNfdHVwbGUlM0RUcnVlKSU1QjAlNUQlMEElMEFwcmVkaWN0ZWRfdG9rZW5faWQlMjAlM0QlMjBsb2dpdHMlNUIwJTJDJTIwbWFza190b2tlbl9pbmRleCU1RC5hcmdtYXgoYXhpcyUzRC0xKSUwQXRva2VuaXplci5kZWNvZGUocHJlZGljdGVkX3Rva2VuX2lkKSUwQSUwQWxhYmVscyUyMCUzRCUyMHRva2VuaXplciglMjJUaGUlMjBjYXBpdGFsJTIwb2YlMjBGcmFuY2UlMjBpcyUyMFBhcmlzLiUyMiUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpJTVCJTIyaW5wdXRfaWRzJTIyJTVEJTBBJTIzJTIwbWFzayUyMGxhYmVscyUyMG9mJTIwbm9uLSUzQ21hc2slM0UlMjB0b2tlbnMlMEFsYWJlbHMlMjAlM0QlMjB0b3JjaC53aGVyZShpbnB1dHMuaW5wdXRfaWRzJTIwJTNEJTNEJTIwdG9rZW5pemVyLm1hc2tfdG9rZW5faWQlMkMlMjBsYWJlbHMlMkMlMjAtMTAwKSUwQSUwQW91dHB1dHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyUyQyUyMGxhYmVscyUzRGxhYmVscyklMEFyb3VuZChvdXRwdXRzLmxvc3MuaXRlbSgpJTJDJTIwMik=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, MobileBertForMaskedLM
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;google/mobilebert-uncased&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = MobileBertForMaskedLM.from_pretrained(<span class="hljs-string">&quot;google/mobilebert-uncased&quot;</span>)

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
...`,wrap:!1}}),{c(){t=c("p"),t.textContent=h,o=r(),u(d.$$.fragment)},l(n){t=p(n,"P",{"data-svelte-h":!0}),y(t)!=="svelte-11lpom8"&&(t.textContent=h),o=a(n),f(d.$$.fragment,n)},m(n,T){m(n,t,T),m(n,o,T),g(d,n,T),k=!0},p:x,i(n){k||(b(d.$$.fragment,n),k=!0)},o(n){_(d.$$.fragment,n),k=!1},d(n){n&&(l(t),l(o)),M(d,n)}}}function Qr(w){let t,h=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=h},l(o){t=p(o,"P",{"data-svelte-h":!0}),y(t)!=="svelte-fincs2"&&(t.innerHTML=h)},m(o,d){m(o,t,d)},p:x,d(o){o&&l(t)}}}function Er(w){let t,h="Examples:",o,d,k;return d=new oe({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBNb2JpbGVCZXJ0Rm9yTmV4dFNlbnRlbmNlUHJlZGljdGlvbiUwQWltcG9ydCUyMHRvcmNoJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIyZ29vZ2xlJTJGbW9iaWxlYmVydC11bmNhc2VkJTIyKSUwQW1vZGVsJTIwJTNEJTIwTW9iaWxlQmVydEZvck5leHRTZW50ZW5jZVByZWRpY3Rpb24uZnJvbV9wcmV0cmFpbmVkKCUyMmdvb2dsZSUyRm1vYmlsZWJlcnQtdW5jYXNlZCUyMiklMEElMEFwcm9tcHQlMjAlM0QlMjAlMjJJbiUyMEl0YWx5JTJDJTIwcGl6emElMjBzZXJ2ZWQlMjBpbiUyMGZvcm1hbCUyMHNldHRpbmdzJTJDJTIwc3VjaCUyMGFzJTIwYXQlMjBhJTIwcmVzdGF1cmFudCUyQyUyMGlzJTIwcHJlc2VudGVkJTIwdW5zbGljZWQuJTIyJTBBbmV4dF9zZW50ZW5jZSUyMCUzRCUyMCUyMlRoZSUyMHNreSUyMGlzJTIwYmx1ZSUyMGR1ZSUyMHRvJTIwdGhlJTIwc2hvcnRlciUyMHdhdmVsZW5ndGglMjBvZiUyMGJsdWUlMjBsaWdodC4lMjIlMEFlbmNvZGluZyUyMCUzRCUyMHRva2VuaXplcihwcm9tcHQlMkMlMjBuZXh0X3NlbnRlbmNlJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiklMEElMEFvdXRwdXRzJTIwJTNEJTIwbW9kZWwoKiplbmNvZGluZyUyQyUyMGxhYmVscyUzRHRvcmNoLkxvbmdUZW5zb3IoJTVCMSU1RCkpJTBBbG9zcyUyMCUzRCUyMG91dHB1dHMubG9zcyUwQWxvZ2l0cyUyMCUzRCUyMG91dHB1dHMubG9naXRz",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, MobileBertForNextSentencePrediction
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;google/mobilebert-uncased&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = MobileBertForNextSentencePrediction.from_pretrained(<span class="hljs-string">&quot;google/mobilebert-uncased&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>prompt = <span class="hljs-string">&quot;In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>next_sentence = <span class="hljs-string">&quot;The sky is blue due to the shorter wavelength of blue light.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>encoding = tokenizer(prompt, next_sentence, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**encoding, labels=torch.LongTensor([<span class="hljs-number">1</span>]))
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = outputs.loss
<span class="hljs-meta">&gt;&gt;&gt; </span>logits = outputs.logits`,wrap:!1}}),{c(){t=c("p"),t.textContent=h,o=r(),u(d.$$.fragment)},l(n){t=p(n,"P",{"data-svelte-h":!0}),y(t)!=="svelte-kvfsh7"&&(t.textContent=h),o=a(n),f(d.$$.fragment,n)},m(n,T){m(n,t,T),m(n,o,T),g(d,n,T),k=!0},p:x,i(n){k||(b(d.$$.fragment,n),k=!0)},o(n){_(d.$$.fragment,n),k=!1},d(n){n&&(l(t),l(o)),M(d,n)}}}function Pr(w){let t,h=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=h},l(o){t=p(o,"P",{"data-svelte-h":!0}),y(t)!=="svelte-fincs2"&&(t.innerHTML=h)},m(o,d){m(o,t,d)},p:x,d(o){o&&l(t)}}}function Yr(w){let t,h="Example of single-label classification:",o,d,k;return d=new oe({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Rva2VuaXplciUyQyUyME1vYmlsZUJlcnRGb3JTZXF1ZW5jZUNsYXNzaWZpY2F0aW9uJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIyZ29vZ2xlJTJGbW9iaWxlYmVydC11bmNhc2VkJTIyKSUwQW1vZGVsJTIwJTNEJTIwTW9iaWxlQmVydEZvclNlcXVlbmNlQ2xhc3NpZmljYXRpb24uZnJvbV9wcmV0cmFpbmVkKCUyMmdvb2dsZSUyRm1vYmlsZWJlcnQtdW5jYXNlZCUyMiklMEElMEFpbnB1dHMlMjAlM0QlMjB0b2tlbml6ZXIoJTIySGVsbG8lMkMlMjBteSUyMGRvZyUyMGlzJTIwY3V0ZSUyMiUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpJTBBJTBBd2l0aCUyMHRvcmNoLm5vX2dyYWQoKSUzQSUwQSUyMCUyMCUyMCUyMGxvZ2l0cyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzKS5sb2dpdHMlMEElMEFwcmVkaWN0ZWRfY2xhc3NfaWQlMjAlM0QlMjBsb2dpdHMuYXJnbWF4KCkuaXRlbSgpJTBBbW9kZWwuY29uZmlnLmlkMmxhYmVsJTVCcHJlZGljdGVkX2NsYXNzX2lkJTVEJTBBJTBBJTIzJTIwVG8lMjB0cmFpbiUyMGElMjBtb2RlbCUyMG9uJTIwJTYwbnVtX2xhYmVscyU2MCUyMGNsYXNzZXMlMkMlMjB5b3UlMjBjYW4lMjBwYXNzJTIwJTYwbnVtX2xhYmVscyUzRG51bV9sYWJlbHMlNjAlMjB0byUyMCU2MC5mcm9tX3ByZXRyYWluZWQoLi4uKSU2MCUwQW51bV9sYWJlbHMlMjAlM0QlMjBsZW4obW9kZWwuY29uZmlnLmlkMmxhYmVsKSUwQW1vZGVsJTIwJTNEJTIwTW9iaWxlQmVydEZvclNlcXVlbmNlQ2xhc3NpZmljYXRpb24uZnJvbV9wcmV0cmFpbmVkKCUyMmdvb2dsZSUyRm1vYmlsZWJlcnQtdW5jYXNlZCUyMiUyQyUyMG51bV9sYWJlbHMlM0RudW1fbGFiZWxzKSUwQSUwQWxhYmVscyUyMCUzRCUyMHRvcmNoLnRlbnNvciglNUIxJTVEKSUwQWxvc3MlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyUyQyUyMGxhYmVscyUzRGxhYmVscykubG9zcyUwQXJvdW5kKGxvc3MuaXRlbSgpJTJDJTIwMik=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, MobileBertForSequenceClassification

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;google/mobilebert-uncased&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = MobileBertForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;google/mobilebert-uncased&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    logits = model(**inputs).logits

<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_class_id = logits.argmax().item()
<span class="hljs-meta">&gt;&gt;&gt; </span>model.config.id2label[predicted_class_id]
...

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># To train a model on \`num_labels\` classes, you can pass \`num_labels=num_labels\` to \`.from_pretrained(...)\`</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>num_labels = <span class="hljs-built_in">len</span>(model.config.id2label)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = MobileBertForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;google/mobilebert-uncased&quot;</span>, num_labels=num_labels)

<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.tensor([<span class="hljs-number">1</span>])
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = model(**inputs, labels=labels).loss
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">round</span>(loss.item(), <span class="hljs-number">2</span>)
...`,wrap:!1}}),{c(){t=c("p"),t.textContent=h,o=r(),u(d.$$.fragment)},l(n){t=p(n,"P",{"data-svelte-h":!0}),y(t)!=="svelte-ykxpe4"&&(t.textContent=h),o=a(n),f(d.$$.fragment,n)},m(n,T){m(n,t,T),m(n,o,T),g(d,n,T),k=!0},p:x,i(n){k||(b(d.$$.fragment,n),k=!0)},o(n){_(d.$$.fragment,n),k=!1},d(n){n&&(l(t),l(o)),M(d,n)}}}function Ar(w){let t,h="Example of multi-label classification:",o,d,k;return d=new oe({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Rva2VuaXplciUyQyUyME1vYmlsZUJlcnRGb3JTZXF1ZW5jZUNsYXNzaWZpY2F0aW9uJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIyZ29vZ2xlJTJGbW9iaWxlYmVydC11bmNhc2VkJTIyKSUwQW1vZGVsJTIwJTNEJTIwTW9iaWxlQmVydEZvclNlcXVlbmNlQ2xhc3NpZmljYXRpb24uZnJvbV9wcmV0cmFpbmVkKCUyMmdvb2dsZSUyRm1vYmlsZWJlcnQtdW5jYXNlZCUyMiUyQyUyMHByb2JsZW1fdHlwZSUzRCUyMm11bHRpX2xhYmVsX2NsYXNzaWZpY2F0aW9uJTIyKSUwQSUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplciglMjJIZWxsbyUyQyUyMG15JTIwZG9nJTIwaXMlMjBjdXRlJTIyJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiklMEElMEF3aXRoJTIwdG9yY2gubm9fZ3JhZCgpJTNBJTBBJTIwJTIwJTIwJTIwbG9naXRzJTIwJTNEJTIwbW9kZWwoKippbnB1dHMpLmxvZ2l0cyUwQSUwQXByZWRpY3RlZF9jbGFzc19pZHMlMjAlM0QlMjB0b3JjaC5hcmFuZ2UoMCUyQyUyMGxvZ2l0cy5zaGFwZSU1Qi0xJTVEKSU1QnRvcmNoLnNpZ21vaWQobG9naXRzKS5zcXVlZXplKGRpbSUzRDApJTIwJTNFJTIwMC41JTVEJTBBJTBBJTIzJTIwVG8lMjB0cmFpbiUyMGElMjBtb2RlbCUyMG9uJTIwJTYwbnVtX2xhYmVscyU2MCUyMGNsYXNzZXMlMkMlMjB5b3UlMjBjYW4lMjBwYXNzJTIwJTYwbnVtX2xhYmVscyUzRG51bV9sYWJlbHMlNjAlMjB0byUyMCU2MC5mcm9tX3ByZXRyYWluZWQoLi4uKSU2MCUwQW51bV9sYWJlbHMlMjAlM0QlMjBsZW4obW9kZWwuY29uZmlnLmlkMmxhYmVsKSUwQW1vZGVsJTIwJTNEJTIwTW9iaWxlQmVydEZvclNlcXVlbmNlQ2xhc3NpZmljYXRpb24uZnJvbV9wcmV0cmFpbmVkKCUwQSUyMCUyMCUyMCUyMCUyMmdvb2dsZSUyRm1vYmlsZWJlcnQtdW5jYXNlZCUyMiUyQyUyMG51bV9sYWJlbHMlM0RudW1fbGFiZWxzJTJDJTIwcHJvYmxlbV90eXBlJTNEJTIybXVsdGlfbGFiZWxfY2xhc3NpZmljYXRpb24lMjIlMEEpJTBBJTBBbGFiZWxzJTIwJTNEJTIwdG9yY2guc3VtKCUwQSUyMCUyMCUyMCUyMHRvcmNoLm5uLmZ1bmN0aW9uYWwub25lX2hvdChwcmVkaWN0ZWRfY2xhc3NfaWRzJTVCTm9uZSUyQyUyMCUzQSU1RC5jbG9uZSgpJTJDJTIwbnVtX2NsYXNzZXMlM0RudW1fbGFiZWxzKSUyQyUyMGRpbSUzRDElMEEpLnRvKHRvcmNoLmZsb2F0KSUwQWxvc3MlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyUyQyUyMGxhYmVscyUzRGxhYmVscykubG9zcw==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, MobileBertForSequenceClassification

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;google/mobilebert-uncased&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = MobileBertForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;google/mobilebert-uncased&quot;</span>, problem_type=<span class="hljs-string">&quot;multi_label_classification&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    logits = model(**inputs).logits

<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_class_ids = torch.arange(<span class="hljs-number">0</span>, logits.shape[-<span class="hljs-number">1</span>])[torch.sigmoid(logits).squeeze(dim=<span class="hljs-number">0</span>) &gt; <span class="hljs-number">0.5</span>]

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># To train a model on \`num_labels\` classes, you can pass \`num_labels=num_labels\` to \`.from_pretrained(...)\`</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>num_labels = <span class="hljs-built_in">len</span>(model.config.id2label)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = MobileBertForSequenceClassification.from_pretrained(
<span class="hljs-meta">... </span>    <span class="hljs-string">&quot;google/mobilebert-uncased&quot;</span>, num_labels=num_labels, problem_type=<span class="hljs-string">&quot;multi_label_classification&quot;</span>
<span class="hljs-meta">... </span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.<span class="hljs-built_in">sum</span>(
<span class="hljs-meta">... </span>    torch.nn.functional.one_hot(predicted_class_ids[<span class="hljs-literal">None</span>, :].clone(), num_classes=num_labels), dim=<span class="hljs-number">1</span>
<span class="hljs-meta">... </span>).to(torch.<span class="hljs-built_in">float</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = model(**inputs, labels=labels).loss`,wrap:!1}}),{c(){t=c("p"),t.textContent=h,o=r(),u(d.$$.fragment)},l(n){t=p(n,"P",{"data-svelte-h":!0}),y(t)!=="svelte-1l8e32d"&&(t.textContent=h),o=a(n),f(d.$$.fragment,n)},m(n,T){m(n,t,T),m(n,o,T),g(d,n,T),k=!0},p:x,i(n){k||(b(d.$$.fragment,n),k=!0)},o(n){_(d.$$.fragment,n),k=!1},d(n){n&&(l(t),l(o)),M(d,n)}}}function Or(w){let t,h=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=h},l(o){t=p(o,"P",{"data-svelte-h":!0}),y(t)!=="svelte-fincs2"&&(t.innerHTML=h)},m(o,d){m(o,t,d)},p:x,d(o){o&&l(t)}}}function Dr(w){let t,h="Example:",o,d,k;return d=new oe({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBNb2JpbGVCZXJ0Rm9yTXVsdGlwbGVDaG9pY2UlMEFpbXBvcnQlMjB0b3JjaCUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMmdvb2dsZSUyRm1vYmlsZWJlcnQtdW5jYXNlZCUyMiklMEFtb2RlbCUyMCUzRCUyME1vYmlsZUJlcnRGb3JNdWx0aXBsZUNob2ljZS5mcm9tX3ByZXRyYWluZWQoJTIyZ29vZ2xlJTJGbW9iaWxlYmVydC11bmNhc2VkJTIyKSUwQSUwQXByb21wdCUyMCUzRCUyMCUyMkluJTIwSXRhbHklMkMlMjBwaXp6YSUyMHNlcnZlZCUyMGluJTIwZm9ybWFsJTIwc2V0dGluZ3MlMkMlMjBzdWNoJTIwYXMlMjBhdCUyMGElMjByZXN0YXVyYW50JTJDJTIwaXMlMjBwcmVzZW50ZWQlMjB1bnNsaWNlZC4lMjIlMEFjaG9pY2UwJTIwJTNEJTIwJTIySXQlMjBpcyUyMGVhdGVuJTIwd2l0aCUyMGElMjBmb3JrJTIwYW5kJTIwYSUyMGtuaWZlLiUyMiUwQWNob2ljZTElMjAlM0QlMjAlMjJJdCUyMGlzJTIwZWF0ZW4lMjB3aGlsZSUyMGhlbGQlMjBpbiUyMHRoZSUyMGhhbmQuJTIyJTBBbGFiZWxzJTIwJTNEJTIwdG9yY2gudGVuc29yKDApLnVuc3F1ZWV6ZSgwKSUyMCUyMCUyMyUyMGNob2ljZTAlMjBpcyUyMGNvcnJlY3QlMjAoYWNjb3JkaW5nJTIwdG8lMjBXaWtpcGVkaWElMjAlM0IpKSUyQyUyMGJhdGNoJTIwc2l6ZSUyMDElMEElMEFlbmNvZGluZyUyMCUzRCUyMHRva2VuaXplciglNUJwcm9tcHQlMkMlMjBwcm9tcHQlNUQlMkMlMjAlNUJjaG9pY2UwJTJDJTIwY2hvaWNlMSU1RCUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIlMkMlMjBwYWRkaW5nJTNEVHJ1ZSklMEFvdXRwdXRzJTIwJTNEJTIwbW9kZWwoKiolN0JrJTNBJTIwdi51bnNxdWVlemUoMCklMjBmb3IlMjBrJTJDJTIwdiUyMGluJTIwZW5jb2RpbmcuaXRlbXMoKSU3RCUyQyUyMGxhYmVscyUzRGxhYmVscyklMjAlMjAlMjMlMjBiYXRjaCUyMHNpemUlMjBpcyUyMDElMEElMEElMjMlMjB0aGUlMjBsaW5lYXIlMjBjbGFzc2lmaWVyJTIwc3RpbGwlMjBuZWVkcyUyMHRvJTIwYmUlMjB0cmFpbmVkJTBBbG9zcyUyMCUzRCUyMG91dHB1dHMubG9zcyUwQWxvZ2l0cyUyMCUzRCUyMG91dHB1dHMubG9naXRz",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, MobileBertForMultipleChoice
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;google/mobilebert-uncased&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = MobileBertForMultipleChoice.from_pretrained(<span class="hljs-string">&quot;google/mobilebert-uncased&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>prompt = <span class="hljs-string">&quot;In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>choice0 = <span class="hljs-string">&quot;It is eaten with a fork and a knife.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>choice1 = <span class="hljs-string">&quot;It is eaten while held in the hand.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.tensor(<span class="hljs-number">0</span>).unsqueeze(<span class="hljs-number">0</span>)  <span class="hljs-comment"># choice0 is correct (according to Wikipedia ;)), batch size 1</span>

<span class="hljs-meta">&gt;&gt;&gt; </span>encoding = tokenizer([prompt, prompt], [choice0, choice1], return_tensors=<span class="hljs-string">&quot;pt&quot;</span>, padding=<span class="hljs-literal">True</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**{k: v.unsqueeze(<span class="hljs-number">0</span>) <span class="hljs-keyword">for</span> k, v <span class="hljs-keyword">in</span> encoding.items()}, labels=labels)  <span class="hljs-comment"># batch size is 1</span>

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># the linear classifier still needs to be trained</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = outputs.loss
<span class="hljs-meta">&gt;&gt;&gt; </span>logits = outputs.logits`,wrap:!1}}),{c(){t=c("p"),t.textContent=h,o=r(),u(d.$$.fragment)},l(n){t=p(n,"P",{"data-svelte-h":!0}),y(t)!=="svelte-11lpom8"&&(t.textContent=h),o=a(n),f(d.$$.fragment,n)},m(n,T){m(n,t,T),m(n,o,T),g(d,n,T),k=!0},p:x,i(n){k||(b(d.$$.fragment,n),k=!0)},o(n){_(d.$$.fragment,n),k=!1},d(n){n&&(l(t),l(o)),M(d,n)}}}function Kr(w){let t,h=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=h},l(o){t=p(o,"P",{"data-svelte-h":!0}),y(t)!=="svelte-fincs2"&&(t.innerHTML=h)},m(o,d){m(o,t,d)},p:x,d(o){o&&l(t)}}}function ea(w){let t,h="Example:",o,d,k;return d=new oe({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBNb2JpbGVCZXJ0Rm9yVG9rZW5DbGFzc2lmaWNhdGlvbiUwQWltcG9ydCUyMHRvcmNoJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIyZ29vZ2xlJTJGbW9iaWxlYmVydC11bmNhc2VkJTIyKSUwQW1vZGVsJTIwJTNEJTIwTW9iaWxlQmVydEZvclRva2VuQ2xhc3NpZmljYXRpb24uZnJvbV9wcmV0cmFpbmVkKCUyMmdvb2dsZSUyRm1vYmlsZWJlcnQtdW5jYXNlZCUyMiklMEElMEFpbnB1dHMlMjAlM0QlMjB0b2tlbml6ZXIoJTBBJTIwJTIwJTIwJTIwJTIySHVnZ2luZ0ZhY2UlMjBpcyUyMGElMjBjb21wYW55JTIwYmFzZWQlMjBpbiUyMFBhcmlzJTIwYW5kJTIwTmV3JTIwWW9yayUyMiUyQyUyMGFkZF9zcGVjaWFsX3Rva2VucyUzREZhbHNlJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiUwQSklMEElMEF3aXRoJTIwdG9yY2gubm9fZ3JhZCgpJTNBJTBBJTIwJTIwJTIwJTIwbG9naXRzJTIwJTNEJTIwbW9kZWwoKippbnB1dHMpLmxvZ2l0cyUwQSUwQXByZWRpY3RlZF90b2tlbl9jbGFzc19pZHMlMjAlM0QlMjBsb2dpdHMuYXJnbWF4KC0xKSUwQSUwQSUyMyUyME5vdGUlMjB0aGF0JTIwdG9rZW5zJTIwYXJlJTIwY2xhc3NpZmllZCUyMHJhdGhlciUyMHRoZW4lMjBpbnB1dCUyMHdvcmRzJTIwd2hpY2glMjBtZWFucyUyMHRoYXQlMEElMjMlMjB0aGVyZSUyMG1pZ2h0JTIwYmUlMjBtb3JlJTIwcHJlZGljdGVkJTIwdG9rZW4lMjBjbGFzc2VzJTIwdGhhbiUyMHdvcmRzLiUwQSUyMyUyME11bHRpcGxlJTIwdG9rZW4lMjBjbGFzc2VzJTIwbWlnaHQlMjBhY2NvdW50JTIwZm9yJTIwdGhlJTIwc2FtZSUyMHdvcmQlMEFwcmVkaWN0ZWRfdG9rZW5zX2NsYXNzZXMlMjAlM0QlMjAlNUJtb2RlbC5jb25maWcuaWQybGFiZWwlNUJ0Lml0ZW0oKSU1RCUyMGZvciUyMHQlMjBpbiUyMHByZWRpY3RlZF90b2tlbl9jbGFzc19pZHMlNUIwJTVEJTVEJTBBcHJlZGljdGVkX3Rva2Vuc19jbGFzc2VzJTBBJTBBbGFiZWxzJTIwJTNEJTIwcHJlZGljdGVkX3Rva2VuX2NsYXNzX2lkcyUwQWxvc3MlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyUyQyUyMGxhYmVscyUzRGxhYmVscykubG9zcyUwQXJvdW5kKGxvc3MuaXRlbSgpJTJDJTIwMik=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, MobileBertForTokenClassification
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;google/mobilebert-uncased&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = MobileBertForTokenClassification.from_pretrained(<span class="hljs-string">&quot;google/mobilebert-uncased&quot;</span>)

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
...`,wrap:!1}}),{c(){t=c("p"),t.textContent=h,o=r(),u(d.$$.fragment)},l(n){t=p(n,"P",{"data-svelte-h":!0}),y(t)!=="svelte-11lpom8"&&(t.textContent=h),o=a(n),f(d.$$.fragment,n)},m(n,T){m(n,t,T),m(n,o,T),g(d,n,T),k=!0},p:x,i(n){k||(b(d.$$.fragment,n),k=!0)},o(n){_(d.$$.fragment,n),k=!1},d(n){n&&(l(t),l(o)),M(d,n)}}}function ta(w){let t,h=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=h},l(o){t=p(o,"P",{"data-svelte-h":!0}),y(t)!=="svelte-fincs2"&&(t.innerHTML=h)},m(o,d){m(o,t,d)},p:x,d(o){o&&l(t)}}}function oa(w){let t,h="Example:",o,d,k;return d=new oe({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBNb2JpbGVCZXJ0Rm9yUXVlc3Rpb25BbnN3ZXJpbmclMEFpbXBvcnQlMjB0b3JjaCUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMmdvb2dsZSUyRm1vYmlsZWJlcnQtdW5jYXNlZCUyMiklMEFtb2RlbCUyMCUzRCUyME1vYmlsZUJlcnRGb3JRdWVzdGlvbkFuc3dlcmluZy5mcm9tX3ByZXRyYWluZWQoJTIyZ29vZ2xlJTJGbW9iaWxlYmVydC11bmNhc2VkJTIyKSUwQSUwQXF1ZXN0aW9uJTJDJTIwdGV4dCUyMCUzRCUyMCUyMldobyUyMHdhcyUyMEppbSUyMEhlbnNvbiUzRiUyMiUyQyUyMCUyMkppbSUyMEhlbnNvbiUyMHdhcyUyMGElMjBuaWNlJTIwcHVwcGV0JTIyJTBBJTBBaW5wdXRzJTIwJTNEJTIwdG9rZW5pemVyKHF1ZXN0aW9uJTJDJTIwdGV4dCUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpJTBBd2l0aCUyMHRvcmNoLm5vX2dyYWQoKSUzQSUwQSUyMCUyMCUyMCUyMG91dHB1dHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyklMEElMEFhbnN3ZXJfc3RhcnRfaW5kZXglMjAlM0QlMjBvdXRwdXRzLnN0YXJ0X2xvZ2l0cy5hcmdtYXgoKSUwQWFuc3dlcl9lbmRfaW5kZXglMjAlM0QlMjBvdXRwdXRzLmVuZF9sb2dpdHMuYXJnbWF4KCklMEElMEFwcmVkaWN0X2Fuc3dlcl90b2tlbnMlMjAlM0QlMjBpbnB1dHMuaW5wdXRfaWRzJTVCMCUyQyUyMGFuc3dlcl9zdGFydF9pbmRleCUyMCUzQSUyMGFuc3dlcl9lbmRfaW5kZXglMjAlMkIlMjAxJTVEJTBBdG9rZW5pemVyLmRlY29kZShwcmVkaWN0X2Fuc3dlcl90b2tlbnMlMkMlMjBza2lwX3NwZWNpYWxfdG9rZW5zJTNEVHJ1ZSklMEElMEElMjMlMjB0YXJnZXQlMjBpcyUyMCUyMm5pY2UlMjBwdXBwZXQlMjIlMEF0YXJnZXRfc3RhcnRfaW5kZXglMjAlM0QlMjB0b3JjaC50ZW5zb3IoJTVCMTQlNUQpJTBBdGFyZ2V0X2VuZF9pbmRleCUyMCUzRCUyMHRvcmNoLnRlbnNvciglNUIxNSU1RCklMEElMEFvdXRwdXRzJTIwJTNEJTIwbW9kZWwoKippbnB1dHMlMkMlMjBzdGFydF9wb3NpdGlvbnMlM0R0YXJnZXRfc3RhcnRfaW5kZXglMkMlMjBlbmRfcG9zaXRpb25zJTNEdGFyZ2V0X2VuZF9pbmRleCklMEFsb3NzJTIwJTNEJTIwb3V0cHV0cy5sb3NzJTBBcm91bmQobG9zcy5pdGVtKCklMkMlMjAyKQ==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, MobileBertForQuestionAnswering
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;google/mobilebert-uncased&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = MobileBertForQuestionAnswering.from_pretrained(<span class="hljs-string">&quot;google/mobilebert-uncased&quot;</span>)

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
...`,wrap:!1}}),{c(){t=c("p"),t.textContent=h,o=r(),u(d.$$.fragment)},l(n){t=p(n,"P",{"data-svelte-h":!0}),y(t)!=="svelte-11lpom8"&&(t.textContent=h),o=a(n),f(d.$$.fragment,n)},m(n,T){m(n,t,T),m(n,o,T),g(d,n,T),k=!0},p:x,i(n){k||(b(d.$$.fragment,n),k=!0)},o(n){_(d.$$.fragment,n),k=!1},d(n){n&&(l(t),l(o)),M(d,n)}}}function na(w){let t,h,o,d,k,n="<em>This model was released on 2020-04-06 and added to Hugging Face Transformers on 2020-11-16.</em>",T,J,xo='<div class="flex flex-wrap space-x-1"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/></div>',He,le,jo,Le,Cs='<a href="https://huggingface.co/papers/2004.02984" rel="nofollow">MobileBERT</a> is a lightweight and efficient variant of BERT, specifically designed for resource-limited devices such as mobile phones. It retains BERT’s architecture but significantly reduces model size and inference latency while maintaining strong performance on NLP tasks. MobileBERT achieves this through a bottleneck structure and carefully balanced self-attention and feedforward networks. The model is trained by knowledge transfer from a large BERT model with an inverted bottleneck structure.',zo,Qe,Fs='You can find the original MobileBERT checkpoint under the <a href="https://huggingface.co/google/mobilebert-uncased" rel="nofollow">Google</a> organization.',Co,Te,Fo,Ee,Ws='The example below demonstrates how to predict the <code>[MASK]</code> token with <a href="/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.Pipeline">Pipeline</a>, <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel">AutoModel</a>, and from the command line.',Wo,ye,Zo,Pe,Io,Ye,Zs="<li>Inputs should be padded on the right because BERT uses absolute position embeddings.</li>",qo,Ae,No,G,Oe,pn,It,Is=`This is the configuration class to store the configuration of a <a href="/docs/transformers/v4.56.2/en/model_doc/mobilebert#transformers.MobileBertModel">MobileBertModel</a> or a <code>TFMobileBertModel</code>. It
is used to instantiate a MobileBERT model according to the specified arguments, defining the model architecture.
Instantiating a configuration with the defaults will yield a similar configuration to that of the MobileBERT
<a href="https://huggingface.co/google/mobilebert-uncased" rel="nofollow">google/mobilebert-uncased</a> architecture.`,mn,qt,qs=`Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> and can be used to control the model outputs. Read the
documentation from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> for more information.`,hn,ke,Ro,De,Vo,U,Ke,un,Nt,Ns="Construct a MobileBERT tokenizer. Based on WordPiece.",fn,Rt,Rs=`This tokenizer inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer">PreTrainedTokenizer</a> which contains most of the main methods. Users should refer to
this superclass for more information regarding those methods.`,gn,de,et,bn,Vt,Vs=`Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
adding special tokens. A MobileBERT sequence has the following format:`,_n,Gt,Gs="<li>single sequence: <code>[CLS] X [SEP]</code></li> <li>pair of sequences: <code>[CLS] A [SEP] B [SEP]</code></li>",Mn,we,tt,Tn,St,Ss="Converts a sequence of tokens (string) in a single string.",yn,ve,ot,kn,Xt,Xs=`Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
special tokens using the tokenizer <code>prepare_for_model</code> method.`,Go,nt,So,S,st,wn,Ht,Hs="Construct a “fast” MobileBERT tokenizer (backed by HuggingFace’s <em>tokenizers</em> library). Based on WordPiece.",vn,Lt,Ls=`This tokenizer inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast">PreTrainedTokenizerFast</a> which contains most of the main methods. Users should
refer to this superclass for more information regarding those methods.`,$n,ce,rt,Bn,Qt,Qs=`Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
adding special tokens. A MobileBERT sequence has the following format:`,Jn,Et,Es="<li>single sequence: <code>[CLS] X [SEP]</code></li> <li>pair of sequences: <code>[CLS] A [SEP] B [SEP]</code></li>",Xo,at,Ho,ue,it,xn,Pt,Ps='Output type of <a href="/docs/transformers/v4.56.2/en/model_doc/mobilebert#transformers.MobileBertForPreTraining">MobileBertForPreTraining</a>.',Lo,lt,Qo,j,dt,Un,Yt,Ys="The bare Mobilebert Model outputting raw hidden-states without any specific head on top.",jn,At,As=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,zn,Ot,Os=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Cn,pe,ct,Fn,Dt,Ds='The <a href="/docs/transformers/v4.56.2/en/model_doc/mobilebert#transformers.MobileBertModel">MobileBertModel</a> forward method, overrides the <code>__call__</code> special method.',Wn,$e,Eo,pt,Po,z,mt,Zn,Kt,Ks=`MobileBert Model with two heads on top as done during the pretraining: a <code>masked language modeling</code> head and a
<code>next sentence prediction (classification)</code> head.`,In,eo,er=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,qn,to,tr=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Nn,A,ht,Rn,oo,or='The <a href="/docs/transformers/v4.56.2/en/model_doc/mobilebert#transformers.MobileBertForPreTraining">MobileBertForPreTraining</a> forward method, overrides the <code>__call__</code> special method.',Vn,Be,Gn,Je,Yo,ut,Ao,C,ft,Sn,no,nr="The Mobilebert Model with a <code>language modeling</code> head on top.”",Xn,so,sr=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Hn,ro,rr=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Ln,O,gt,Qn,ao,ar='The <a href="/docs/transformers/v4.56.2/en/model_doc/mobilebert#transformers.MobileBertForMaskedLM">MobileBertForMaskedLM</a> forward method, overrides the <code>__call__</code> special method.',En,xe,Pn,Ue,Oo,bt,Do,F,_t,Yn,io,ir="MobileBert Model with a <code>next sentence prediction (classification)</code> head on top.",An,lo,lr=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,On,co,dr=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Dn,D,Mt,Kn,po,cr='The <a href="/docs/transformers/v4.56.2/en/model_doc/mobilebert#transformers.MobileBertForNextSentencePrediction">MobileBertForNextSentencePrediction</a> forward method, overrides the <code>__call__</code> special method.',es,je,ts,ze,Ko,Tt,en,W,yt,os,mo,pr=`MobileBert Model transformer with a sequence classification/regression head on top (a linear layer on top of the
pooled output) e.g. for GLUE tasks.`,ns,ho,mr=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,ss,uo,hr=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,rs,R,kt,as,fo,ur='The <a href="/docs/transformers/v4.56.2/en/model_doc/mobilebert#transformers.MobileBertForSequenceClassification">MobileBertForSequenceClassification</a> forward method, overrides the <code>__call__</code> special method.',is,Ce,ls,Fe,ds,We,tn,wt,on,Z,vt,cs,go,fr=`The Mobilebert Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
softmax) e.g. for RocStories/SWAG tasks.`,ps,bo,gr=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,ms,_o,br=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,hs,K,$t,us,Mo,_r='The <a href="/docs/transformers/v4.56.2/en/model_doc/mobilebert#transformers.MobileBertForMultipleChoice">MobileBertForMultipleChoice</a> forward method, overrides the <code>__call__</code> special method.',fs,Ze,gs,Ie,nn,Bt,sn,I,Jt,bs,To,Mr=`The Mobilebert transformer with a token classification head on top (a linear layer on top of the hidden-states
output) e.g. for Named-Entity-Recognition (NER) tasks.`,_s,yo,Tr=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Ms,ko,yr=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Ts,ee,xt,ys,wo,kr='The <a href="/docs/transformers/v4.56.2/en/model_doc/mobilebert#transformers.MobileBertForTokenClassification">MobileBertForTokenClassification</a> forward method, overrides the <code>__call__</code> special method.',ks,qe,ws,Ne,rn,Ut,an,q,jt,vs,vo,wr=`The Mobilebert transformer with a span classification head on top for extractive question-answering tasks like
SQuAD (a linear layer on top of the hidden-states output to compute <code>span start logits</code> and <code>span end logits</code>).`,$s,$o,vr=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Bs,Bo,$r=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Js,te,zt,xs,Jo,Br='The <a href="/docs/transformers/v4.56.2/en/model_doc/mobilebert#transformers.MobileBertForQuestionAnswering">MobileBertForQuestionAnswering</a> forward method, overrides the <code>__call__</code> special method.',Us,Re,js,Ve,ln,Ct,dn,Uo,cn;return le=new V({props:{title:"MobileBERT",local:"mobilebert",headingTag:"h1"}}),Te=new _e({props:{warning:!1,$$slots:{default:[Zr]},$$scope:{ctx:w}}}),ye=new Wr({props:{id:"usage",options:["Pipeline","AutoModel","transformers CLI"],$$slots:{default:[Rr]},$$scope:{ctx:w}}}),Pe=new V({props:{title:"Notes",local:"notes",headingTag:"h2"}}),Ae=new V({props:{title:"MobileBertConfig",local:"transformers.MobileBertConfig",headingTag:"h2"}}),Oe=new B({props:{name:"class transformers.MobileBertConfig",anchor:"transformers.MobileBertConfig",parameters:[{name:"vocab_size",val:" = 30522"},{name:"hidden_size",val:" = 512"},{name:"num_hidden_layers",val:" = 24"},{name:"num_attention_heads",val:" = 4"},{name:"intermediate_size",val:" = 512"},{name:"hidden_act",val:" = 'relu'"},{name:"hidden_dropout_prob",val:" = 0.0"},{name:"attention_probs_dropout_prob",val:" = 0.1"},{name:"max_position_embeddings",val:" = 512"},{name:"type_vocab_size",val:" = 2"},{name:"initializer_range",val:" = 0.02"},{name:"layer_norm_eps",val:" = 1e-12"},{name:"pad_token_id",val:" = 0"},{name:"embedding_size",val:" = 128"},{name:"trigram_input",val:" = True"},{name:"use_bottleneck",val:" = True"},{name:"intra_bottleneck_size",val:" = 128"},{name:"use_bottleneck_attention",val:" = False"},{name:"key_query_shared_bottleneck",val:" = True"},{name:"num_feedforward_networks",val:" = 4"},{name:"normalization_type",val:" = 'no_norm'"},{name:"classifier_activation",val:" = True"},{name:"classifier_dropout",val:" = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.MobileBertConfig.vocab_size",description:`<strong>vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 30522) &#x2014;
Vocabulary size of the MobileBERT model. Defines the number of different tokens that can be represented by
the <code>inputs_ids</code> passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/mobilebert#transformers.MobileBertModel">MobileBertModel</a> or <code>TFMobileBertModel</code>.`,name:"vocab_size"},{anchor:"transformers.MobileBertConfig.hidden_size",description:`<strong>hidden_size</strong> (<code>int</code>, <em>optional</em>, defaults to 512) &#x2014;
Dimensionality of the encoder layers and the pooler layer.`,name:"hidden_size"},{anchor:"transformers.MobileBertConfig.num_hidden_layers",description:`<strong>num_hidden_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 24) &#x2014;
Number of hidden layers in the Transformer encoder.`,name:"num_hidden_layers"},{anchor:"transformers.MobileBertConfig.num_attention_heads",description:`<strong>num_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 4) &#x2014;
Number of attention heads for each attention layer in the Transformer encoder.`,name:"num_attention_heads"},{anchor:"transformers.MobileBertConfig.intermediate_size",description:`<strong>intermediate_size</strong> (<code>int</code>, <em>optional</em>, defaults to 512) &#x2014;
Dimensionality of the &#x201C;intermediate&#x201D; (often named feed-forward) layer in the Transformer encoder.`,name:"intermediate_size"},{anchor:"transformers.MobileBertConfig.hidden_act",description:`<strong>hidden_act</strong> (<code>str</code> or <code>function</code>, <em>optional</em>, defaults to <code>&quot;relu&quot;</code>) &#x2014;
The non-linear activation function (function or string) in the encoder and pooler. If string, <code>&quot;gelu&quot;</code>,
<code>&quot;relu&quot;</code>, <code>&quot;silu&quot;</code> and <code>&quot;gelu_new&quot;</code> are supported.`,name:"hidden_act"},{anchor:"transformers.MobileBertConfig.hidden_dropout_prob",description:`<strong>hidden_dropout_prob</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.`,name:"hidden_dropout_prob"},{anchor:"transformers.MobileBertConfig.attention_probs_dropout_prob",description:`<strong>attention_probs_dropout_prob</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout ratio for the attention probabilities.`,name:"attention_probs_dropout_prob"},{anchor:"transformers.MobileBertConfig.max_position_embeddings",description:`<strong>max_position_embeddings</strong> (<code>int</code>, <em>optional</em>, defaults to 512) &#x2014;
The maximum sequence length that this model might ever be used with. Typically set this to something large
just in case (e.g., 512 or 1024 or 2048).`,name:"max_position_embeddings"},{anchor:"transformers.MobileBertConfig.type_vocab_size",description:`<strong>type_vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 2) &#x2014;
The vocabulary size of the <code>token_type_ids</code> passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/mobilebert#transformers.MobileBertModel">MobileBertModel</a> or
<code>TFMobileBertModel</code>.`,name:"type_vocab_size"},{anchor:"transformers.MobileBertConfig.initializer_range",description:`<strong>initializer_range</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.`,name:"initializer_range"},{anchor:"transformers.MobileBertConfig.layer_norm_eps",description:`<strong>layer_norm_eps</strong> (<code>float</code>, <em>optional</em>, defaults to 1e-12) &#x2014;
The epsilon used by the layer normalization layers.`,name:"layer_norm_eps"},{anchor:"transformers.MobileBertConfig.pad_token_id",description:`<strong>pad_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 0) &#x2014;
The ID of the token in the word embedding to use as padding.`,name:"pad_token_id"},{anchor:"transformers.MobileBertConfig.embedding_size",description:`<strong>embedding_size</strong> (<code>int</code>, <em>optional</em>, defaults to 128) &#x2014;
The dimension of the word embedding vectors.`,name:"embedding_size"},{anchor:"transformers.MobileBertConfig.trigram_input",description:`<strong>trigram_input</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Use a convolution of trigram as input.`,name:"trigram_input"},{anchor:"transformers.MobileBertConfig.use_bottleneck",description:`<strong>use_bottleneck</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether to use bottleneck in BERT.`,name:"use_bottleneck"},{anchor:"transformers.MobileBertConfig.intra_bottleneck_size",description:`<strong>intra_bottleneck_size</strong> (<code>int</code>, <em>optional</em>, defaults to 128) &#x2014;
Size of bottleneck layer output.`,name:"intra_bottleneck_size"},{anchor:"transformers.MobileBertConfig.use_bottleneck_attention",description:`<strong>use_bottleneck_attention</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether to use attention inputs from the bottleneck transformation.`,name:"use_bottleneck_attention"},{anchor:"transformers.MobileBertConfig.key_query_shared_bottleneck",description:`<strong>key_query_shared_bottleneck</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether to use the same linear transformation for query&amp;key in the bottleneck.`,name:"key_query_shared_bottleneck"},{anchor:"transformers.MobileBertConfig.num_feedforward_networks",description:`<strong>num_feedforward_networks</strong> (<code>int</code>, <em>optional</em>, defaults to 4) &#x2014;
Number of FFNs in a block.`,name:"num_feedforward_networks"},{anchor:"transformers.MobileBertConfig.normalization_type",description:`<strong>normalization_type</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;no_norm&quot;</code>) &#x2014;
The normalization type in MobileBERT.`,name:"normalization_type"},{anchor:"transformers.MobileBertConfig.classifier_dropout",description:`<strong>classifier_dropout</strong> (<code>float</code>, <em>optional</em>) &#x2014;
The dropout ratio for the classification head.`,name:"classifier_dropout"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mobilebert/configuration_mobilebert.py#L28"}}),ke=new Me({props:{anchor:"transformers.MobileBertConfig.example",$$slots:{default:[Vr]},$$scope:{ctx:w}}}),De=new V({props:{title:"MobileBertTokenizer",local:"transformers.MobileBertTokenizer",headingTag:"h2"}}),Ke=new B({props:{name:"class transformers.MobileBertTokenizer",anchor:"transformers.MobileBertTokenizer",parameters:[{name:"vocab_file",val:""},{name:"do_lower_case",val:" = True"},{name:"do_basic_tokenize",val:" = True"},{name:"never_split",val:" = None"},{name:"unk_token",val:" = '[UNK]'"},{name:"sep_token",val:" = '[SEP]'"},{name:"pad_token",val:" = '[PAD]'"},{name:"cls_token",val:" = '[CLS]'"},{name:"mask_token",val:" = '[MASK]'"},{name:"tokenize_chinese_chars",val:" = True"},{name:"strip_accents",val:" = None"},{name:"clean_up_tokenization_spaces",val:" = True"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.MobileBertTokenizer.vocab_file",description:`<strong>vocab_file</strong> (<code>str</code>) &#x2014;
File containing the vocabulary.`,name:"vocab_file"},{anchor:"transformers.MobileBertTokenizer.do_lower_case",description:`<strong>do_lower_case</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to lowercase the input when tokenizing.`,name:"do_lower_case"},{anchor:"transformers.MobileBertTokenizer.do_basic_tokenize",description:`<strong>do_basic_tokenize</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to do basic tokenization before WordPiece.`,name:"do_basic_tokenize"},{anchor:"transformers.MobileBertTokenizer.never_split",description:`<strong>never_split</strong> (<code>Iterable</code>, <em>optional</em>) &#x2014;
Collection of tokens which will never be split during tokenization. Only has an effect when
<code>do_basic_tokenize=True</code>`,name:"never_split"},{anchor:"transformers.MobileBertTokenizer.unk_token",description:`<strong>unk_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;[UNK]&quot;</code>) &#x2014;
The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
token instead.`,name:"unk_token"},{anchor:"transformers.MobileBertTokenizer.sep_token",description:`<strong>sep_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;[SEP]&quot;</code>) &#x2014;
The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
sequence classification or for a text and a question for question answering. It is also used as the last
token of a sequence built with special tokens.`,name:"sep_token"},{anchor:"transformers.MobileBertTokenizer.pad_token",description:`<strong>pad_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;[PAD]&quot;</code>) &#x2014;
The token used for padding, for example when batching sequences of different lengths.`,name:"pad_token"},{anchor:"transformers.MobileBertTokenizer.cls_token",description:`<strong>cls_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;[CLS]&quot;</code>) &#x2014;
The classifier token which is used when doing sequence classification (classification of the whole sequence
instead of per-token classification). It is the first token of the sequence when built with special tokens.`,name:"cls_token"},{anchor:"transformers.MobileBertTokenizer.mask_token",description:`<strong>mask_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;[MASK]&quot;</code>) &#x2014;
The token used for masking values. This is the token used when training this model with masked language
modeling. This is the token which the model will try to predict.`,name:"mask_token"},{anchor:"transformers.MobileBertTokenizer.tokenize_chinese_chars",description:`<strong>tokenize_chinese_chars</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to tokenize Chinese characters.</p>
<p>This should likely be deactivated for Japanese (see this
<a href="https://github.com/huggingface/transformers/issues/328" rel="nofollow">issue</a>).`,name:"tokenize_chinese_chars"},{anchor:"transformers.MobileBertTokenizer.strip_accents",description:`<strong>strip_accents</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to strip all accents. If this option is not specified, then it will be determined by the
value for <code>lowercase</code> (as in the original MobileBERT).`,name:"strip_accents"},{anchor:"transformers.MobileBertTokenizer.clean_up_tokenization_spaces",description:`<strong>clean_up_tokenization_spaces</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to cleanup spaces after decoding, cleanup consists in removing potential artifacts like
extra spaces.`,name:"clean_up_tokenization_spaces"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mobilebert/tokenization_mobilebert.py#L55"}}),et=new B({props:{name:"build_inputs_with_special_tokens",anchor:"transformers.MobileBertTokenizer.build_inputs_with_special_tokens",parameters:[{name:"token_ids_0",val:": list"},{name:"token_ids_1",val:": typing.Optional[list[int]] = None"}],parametersDescription:[{anchor:"transformers.MobileBertTokenizer.build_inputs_with_special_tokens.token_ids_0",description:`<strong>token_ids_0</strong> (<code>List[int]</code>) &#x2014;
List of IDs to which the special tokens will be added.`,name:"token_ids_0"},{anchor:"transformers.MobileBertTokenizer.build_inputs_with_special_tokens.token_ids_1",description:`<strong>token_ids_1</strong> (<code>List[int]</code>, <em>optional</em>) &#x2014;
Optional second list of IDs for sequence pairs.`,name:"token_ids_1"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mobilebert/tokenization_mobilebert.py#L190",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>List of <a href="../glossary#input-ids">input IDs</a> with the appropriate special tokens.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>List[int]</code></p>
`}}),tt=new B({props:{name:"convert_tokens_to_string",anchor:"transformers.MobileBertTokenizer.convert_tokens_to_string",parameters:[{name:"tokens",val:""}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mobilebert/tokenization_mobilebert.py#L185"}}),ot=new B({props:{name:"get_special_tokens_mask",anchor:"transformers.MobileBertTokenizer.get_special_tokens_mask",parameters:[{name:"token_ids_0",val:": list"},{name:"token_ids_1",val:": typing.Optional[list[int]] = None"},{name:"already_has_special_tokens",val:": bool = False"}],parametersDescription:[{anchor:"transformers.MobileBertTokenizer.get_special_tokens_mask.token_ids_0",description:`<strong>token_ids_0</strong> (<code>List[int]</code>) &#x2014;
List of IDs.`,name:"token_ids_0"},{anchor:"transformers.MobileBertTokenizer.get_special_tokens_mask.token_ids_1",description:`<strong>token_ids_1</strong> (<code>List[int]</code>, <em>optional</em>) &#x2014;
Optional second list of IDs for sequence pairs.`,name:"token_ids_1"},{anchor:"transformers.MobileBertTokenizer.get_special_tokens_mask.already_has_special_tokens",description:`<strong>already_has_special_tokens</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not the token list is already formatted with special tokens for the model.`,name:"already_has_special_tokens"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mobilebert/tokenization_mobilebert.py#L215",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>List[int]</code></p>
`}}),nt=new V({props:{title:"MobileBertTokenizerFast",local:"transformers.MobileBertTokenizerFast",headingTag:"h2"}}),st=new B({props:{name:"class transformers.MobileBertTokenizerFast",anchor:"transformers.MobileBertTokenizerFast",parameters:[{name:"vocab_file",val:" = None"},{name:"tokenizer_file",val:" = None"},{name:"do_lower_case",val:" = True"},{name:"unk_token",val:" = '[UNK]'"},{name:"sep_token",val:" = '[SEP]'"},{name:"pad_token",val:" = '[PAD]'"},{name:"cls_token",val:" = '[CLS]'"},{name:"mask_token",val:" = '[MASK]'"},{name:"tokenize_chinese_chars",val:" = True"},{name:"strip_accents",val:" = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.MobileBertTokenizerFast.vocab_file",description:`<strong>vocab_file</strong> (<code>str</code>) &#x2014;
File containing the vocabulary.`,name:"vocab_file"},{anchor:"transformers.MobileBertTokenizerFast.do_lower_case",description:`<strong>do_lower_case</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to lowercase the input when tokenizing.`,name:"do_lower_case"},{anchor:"transformers.MobileBertTokenizerFast.unk_token",description:`<strong>unk_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;[UNK]&quot;</code>) &#x2014;
The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
token instead.`,name:"unk_token"},{anchor:"transformers.MobileBertTokenizerFast.sep_token",description:`<strong>sep_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;[SEP]&quot;</code>) &#x2014;
The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
sequence classification or for a text and a question for question answering. It is also used as the last
token of a sequence built with special tokens.`,name:"sep_token"},{anchor:"transformers.MobileBertTokenizerFast.pad_token",description:`<strong>pad_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;[PAD]&quot;</code>) &#x2014;
The token used for padding, for example when batching sequences of different lengths.`,name:"pad_token"},{anchor:"transformers.MobileBertTokenizerFast.cls_token",description:`<strong>cls_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;[CLS]&quot;</code>) &#x2014;
The classifier token which is used when doing sequence classification (classification of the whole sequence
instead of per-token classification). It is the first token of the sequence when built with special tokens.`,name:"cls_token"},{anchor:"transformers.MobileBertTokenizerFast.mask_token",description:`<strong>mask_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;[MASK]&quot;</code>) &#x2014;
The token used for masking values. This is the token used when training this model with masked language
modeling. This is the token which the model will try to predict.`,name:"mask_token"},{anchor:"transformers.MobileBertTokenizerFast.clean_text",description:`<strong>clean_text</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to clean the text before tokenization by removing any control characters and replacing all
whitespaces by the classic one.`,name:"clean_text"},{anchor:"transformers.MobileBertTokenizerFast.tokenize_chinese_chars",description:`<strong>tokenize_chinese_chars</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to tokenize Chinese characters. This should likely be deactivated for Japanese (see <a href="https://github.com/huggingface/transformers/issues/328" rel="nofollow">this
issue</a>).`,name:"tokenize_chinese_chars"},{anchor:"transformers.MobileBertTokenizerFast.strip_accents",description:`<strong>strip_accents</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to strip all accents. If this option is not specified, then it will be determined by the
value for <code>lowercase</code> (as in the original MobileBERT).`,name:"strip_accents"},{anchor:"transformers.MobileBertTokenizerFast.wordpieces_prefix",description:`<strong>wordpieces_prefix</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;##&quot;</code>) &#x2014;
The prefix for subwords.`,name:"wordpieces_prefix"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mobilebert/tokenization_mobilebert_fast.py#L34"}}),rt=new B({props:{name:"build_inputs_with_special_tokens",anchor:"transformers.MobileBertTokenizerFast.build_inputs_with_special_tokens",parameters:[{name:"token_ids_0",val:""},{name:"token_ids_1",val:" = None"}],parametersDescription:[{anchor:"transformers.MobileBertTokenizerFast.build_inputs_with_special_tokens.token_ids_0",description:`<strong>token_ids_0</strong> (<code>List[int]</code>) &#x2014;
List of IDs to which the special tokens will be added.`,name:"token_ids_0"},{anchor:"transformers.MobileBertTokenizerFast.build_inputs_with_special_tokens.token_ids_1",description:`<strong>token_ids_1</strong> (<code>List[int]</code>, <em>optional</em>) &#x2014;
Optional second list of IDs for sequence pairs.`,name:"token_ids_1"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mobilebert/tokenization_mobilebert_fast.py#L119",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>List of <a href="../glossary#input-ids">input IDs</a> with the appropriate special tokens.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>List[int]</code></p>
`}}),at=new V({props:{title:"MobileBert specific outputs",local:"transformers.models.mobilebert.modeling_mobilebert.MobileBertForPreTrainingOutput",headingTag:"h2"}}),it=new B({props:{name:"class transformers.models.mobilebert.modeling_mobilebert.MobileBertForPreTrainingOutput",anchor:"transformers.models.mobilebert.modeling_mobilebert.MobileBertForPreTrainingOutput",parameters:[{name:"loss",val:": typing.Optional[torch.FloatTensor] = None"},{name:"prediction_logits",val:": typing.Optional[torch.FloatTensor] = None"},{name:"seq_relationship_logits",val:": typing.Optional[torch.FloatTensor] = None"},{name:"hidden_states",val:": typing.Optional[tuple[torch.FloatTensor]] = None"},{name:"attentions",val:": typing.Optional[tuple[torch.FloatTensor]] = None"}],parametersDescription:[{anchor:"transformers.models.mobilebert.modeling_mobilebert.MobileBertForPreTrainingOutput.loss",description:`<strong>loss</strong> (<code>*optional*</code>, returned when <code>labels</code> is provided, <code>torch.FloatTensor</code> of shape <code>(1,)</code>) &#x2014;
Total loss as the sum of the masked language modeling loss and the next sequence prediction
(classification) loss.`,name:"loss"},{anchor:"transformers.models.mobilebert.modeling_mobilebert.MobileBertForPreTrainingOutput.prediction_logits",description:`<strong>prediction_logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, config.vocab_size)</code>) &#x2014;
Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).`,name:"prediction_logits"},{anchor:"transformers.models.mobilebert.modeling_mobilebert.MobileBertForPreTrainingOutput.seq_relationship_logits",description:`<strong>seq_relationship_logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, 2)</code>) &#x2014;
Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation
before SoftMax).`,name:"seq_relationship_logits"},{anchor:"transformers.models.mobilebert.modeling_mobilebert.MobileBertForPreTrainingOutput.hidden_states",description:`<strong>hidden_states</strong> (<code>tuple[torch.FloatTensor]</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) &#x2014;
Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings, if the model has an embedding layer, +
one for the output of each layer) of shape <code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.`,name:"hidden_states"},{anchor:"transformers.models.mobilebert.modeling_mobilebert.MobileBertForPreTrainingOutput.attentions",description:`<strong>attentions</strong> (<code>tuple[torch.FloatTensor]</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) &#x2014;
Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
heads.`,name:"attentions"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mobilebert/modeling_mobilebert.py#L690"}}),lt=new V({props:{title:"MobileBertModel",local:"transformers.MobileBertModel",headingTag:"h2"}}),dt=new B({props:{name:"class transformers.MobileBertModel",anchor:"transformers.MobileBertModel",parameters:[{name:"config",val:""},{name:"add_pooling_layer",val:" = True"}],parametersDescription:[{anchor:"transformers.MobileBertModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/mobilebert#transformers.MobileBertModel">MobileBertModel</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"},{anchor:"transformers.MobileBertModel.add_pooling_layer",description:`<strong>add_pooling_layer</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether to add a pooling layer`,name:"add_pooling_layer"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mobilebert/modeling_mobilebert.py#L710"}}),ct=new B({props:{name:"forward",anchor:"transformers.MobileBertModel.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.MobileBertModel.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.MobileBertModel.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.MobileBertModel.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.MobileBertModel.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.MobileBertModel.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.MobileBertModel.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.MobileBertModel.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.MobileBertModel.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.MobileBertModel.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mobilebert/modeling_mobilebert.py#L744",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling"
>transformers.modeling_outputs.BaseModelOutputWithPooling</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/mobilebert#transformers.MobileBertConfig"
>MobileBertConfig</a>) and inputs.</p>
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
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling"
>transformers.modeling_outputs.BaseModelOutputWithPooling</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),$e=new _e({props:{$$slots:{default:[Gr]},$$scope:{ctx:w}}}),pt=new V({props:{title:"MobileBertForPreTraining",local:"transformers.MobileBertForPreTraining",headingTag:"h2"}}),mt=new B({props:{name:"class transformers.MobileBertForPreTraining",anchor:"transformers.MobileBertForPreTraining",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.MobileBertForPreTraining.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/mobilebert#transformers.MobileBertForPreTraining">MobileBertForPreTraining</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mobilebert/modeling_mobilebert.py#L822"}}),ht=new B({props:{name:"forward",anchor:"transformers.MobileBertForPreTraining.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"next_sentence_label",val:": typing.Optional[torch.LongTensor] = None"},{name:"output_attentions",val:": typing.Optional[torch.FloatTensor] = None"},{name:"output_hidden_states",val:": typing.Optional[torch.FloatTensor] = None"},{name:"return_dict",val:": typing.Optional[torch.FloatTensor] = None"}],parametersDescription:[{anchor:"transformers.MobileBertForPreTraining.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.MobileBertForPreTraining.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.MobileBertForPreTraining.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.MobileBertForPreTraining.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.MobileBertForPreTraining.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.MobileBertForPreTraining.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.MobileBertForPreTraining.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should be in <code>[-100, 0, ..., config.vocab_size]</code> (see <code>input_ids</code> docstring) Tokens with indices set to <code>-100</code> are ignored (masked), the
loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>`,name:"labels"},{anchor:"transformers.MobileBertForPreTraining.forward.next_sentence_label",description:`<strong>next_sentence_label</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the next sequence prediction (classification) loss. Input should be a sequence pair
(see <code>input_ids</code> docstring) Indices should be in <code>[0, 1]</code>:</p>
<ul>
<li>0 indicates sequence B is a continuation of sequence A,</li>
<li>1 indicates sequence B is a random sequence.</li>
</ul>`,name:"next_sentence_label"},{anchor:"transformers.MobileBertForPreTraining.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>torch.FloatTensor</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.MobileBertForPreTraining.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>torch.FloatTensor</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.MobileBertForPreTraining.forward.return_dict",description:`<strong>return_dict</strong> (<code>torch.FloatTensor</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mobilebert/modeling_mobilebert.py#L848",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/model_doc/mobilebert#transformers.models.mobilebert.modeling_mobilebert.MobileBertForPreTrainingOutput"
>transformers.models.mobilebert.modeling_mobilebert.MobileBertForPreTrainingOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/mobilebert#transformers.MobileBertConfig"
>MobileBertConfig</a>) and inputs.</p>
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
  href="/docs/transformers/v4.56.2/en/model_doc/mobilebert#transformers.models.mobilebert.modeling_mobilebert.MobileBertForPreTrainingOutput"
>transformers.models.mobilebert.modeling_mobilebert.MobileBertForPreTrainingOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),Be=new _e({props:{$$slots:{default:[Sr]},$$scope:{ctx:w}}}),Je=new Me({props:{anchor:"transformers.MobileBertForPreTraining.forward.example",$$slots:{default:[Xr]},$$scope:{ctx:w}}}),ut=new V({props:{title:"MobileBertForMaskedLM",local:"transformers.MobileBertForMaskedLM",headingTag:"h2"}}),ft=new B({props:{name:"class transformers.MobileBertForMaskedLM",anchor:"transformers.MobileBertForMaskedLM",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.MobileBertForMaskedLM.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/mobilebert#transformers.MobileBertForMaskedLM">MobileBertForMaskedLM</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mobilebert/modeling_mobilebert.py#L928"}}),gt=new B({props:{name:"forward",anchor:"transformers.MobileBertForMaskedLM.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.MobileBertForMaskedLM.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.MobileBertForMaskedLM.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.MobileBertForMaskedLM.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.MobileBertForMaskedLM.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.MobileBertForMaskedLM.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.MobileBertForMaskedLM.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.MobileBertForMaskedLM.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should be in <code>[-100, 0, ..., config.vocab_size]</code> (see <code>input_ids</code> docstring) Tokens with indices set to <code>-100</code> are ignored (masked), the
loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>`,name:"labels"},{anchor:"transformers.MobileBertForMaskedLM.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.MobileBertForMaskedLM.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.MobileBertForMaskedLM.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mobilebert/modeling_mobilebert.py#L954",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.MaskedLMOutput"
>transformers.modeling_outputs.MaskedLMOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/mobilebert#transformers.MobileBertConfig"
>MobileBertConfig</a>) and inputs.</p>
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
`}}),xe=new _e({props:{$$slots:{default:[Hr]},$$scope:{ctx:w}}}),Ue=new Me({props:{anchor:"transformers.MobileBertForMaskedLM.forward.example",$$slots:{default:[Lr]},$$scope:{ctx:w}}}),bt=new V({props:{title:"MobileBertForNextSentencePrediction",local:"transformers.MobileBertForNextSentencePrediction",headingTag:"h2"}}),_t=new B({props:{name:"class transformers.MobileBertForNextSentencePrediction",anchor:"transformers.MobileBertForNextSentencePrediction",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.MobileBertForNextSentencePrediction.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/mobilebert#transformers.MobileBertForNextSentencePrediction">MobileBertForNextSentencePrediction</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mobilebert/modeling_mobilebert.py#L1023"}}),Mt=new B({props:{name:"forward",anchor:"transformers.MobileBertForNextSentencePrediction.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.MobileBertForNextSentencePrediction.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.MobileBertForNextSentencePrediction.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.MobileBertForNextSentencePrediction.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.MobileBertForNextSentencePrediction.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.MobileBertForNextSentencePrediction.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.MobileBertForNextSentencePrediction.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.MobileBertForNextSentencePrediction.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the next sequence prediction (classification) loss. Input should be a sequence pair
(see <code>input_ids</code> docstring) Indices should be in <code>[0, 1]</code>.</p>
<ul>
<li>0 indicates sequence B is a continuation of sequence A,</li>
<li>1 indicates sequence B is a random sequence.</li>
</ul>`,name:"labels"},{anchor:"transformers.MobileBertForNextSentencePrediction.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.MobileBertForNextSentencePrediction.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.MobileBertForNextSentencePrediction.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mobilebert/modeling_mobilebert.py#L1033",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.NextSentencePredictorOutput"
>transformers.modeling_outputs.NextSentencePredictorOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/mobilebert#transformers.MobileBertConfig"
>MobileBertConfig</a>) and inputs.</p>
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
`}}),je=new _e({props:{$$slots:{default:[Qr]},$$scope:{ctx:w}}}),ze=new Me({props:{anchor:"transformers.MobileBertForNextSentencePrediction.forward.example",$$slots:{default:[Er]},$$scope:{ctx:w}}}),Tt=new V({props:{title:"MobileBertForSequenceClassification",local:"transformers.MobileBertForSequenceClassification",headingTag:"h2"}}),yt=new B({props:{name:"class transformers.MobileBertForSequenceClassification",anchor:"transformers.MobileBertForSequenceClassification",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.MobileBertForSequenceClassification.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/mobilebert#transformers.MobileBertForSequenceClassification">MobileBertForSequenceClassification</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mobilebert/modeling_mobilebert.py#L1123"}}),kt=new B({props:{name:"forward",anchor:"transformers.MobileBertForSequenceClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"labels",val:": typing.Optional[torch.Tensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.MobileBertForSequenceClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.MobileBertForSequenceClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.MobileBertForSequenceClassification.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.MobileBertForSequenceClassification.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.MobileBertForSequenceClassification.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.MobileBertForSequenceClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.MobileBertForSequenceClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the sequence classification/regression loss. Indices should be in <code>[0, ..., config.num_labels - 1]</code>. If <code>config.num_labels == 1</code> a regression loss is computed (Mean-Square loss), If
<code>config.num_labels &gt; 1</code> a classification loss is computed (Cross-Entropy).`,name:"labels"},{anchor:"transformers.MobileBertForSequenceClassification.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.MobileBertForSequenceClassification.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.MobileBertForSequenceClassification.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mobilebert/modeling_mobilebert.py#L1139",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput"
>transformers.modeling_outputs.SequenceClassifierOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/mobilebert#transformers.MobileBertConfig"
>MobileBertConfig</a>) and inputs.</p>
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
`}}),Ce=new _e({props:{$$slots:{default:[Pr]},$$scope:{ctx:w}}}),Fe=new Me({props:{anchor:"transformers.MobileBertForSequenceClassification.forward.example",$$slots:{default:[Yr]},$$scope:{ctx:w}}}),We=new Me({props:{anchor:"transformers.MobileBertForSequenceClassification.forward.example-2",$$slots:{default:[Ar]},$$scope:{ctx:w}}}),wt=new V({props:{title:"MobileBertForMultipleChoice",local:"transformers.MobileBertForMultipleChoice",headingTag:"h2"}}),vt=new B({props:{name:"class transformers.MobileBertForMultipleChoice",anchor:"transformers.MobileBertForMultipleChoice",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.MobileBertForMultipleChoice.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/mobilebert#transformers.MobileBertForMultipleChoice">MobileBertForMultipleChoice</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mobilebert/modeling_mobilebert.py#L1293"}}),$t=new B({props:{name:"forward",anchor:"transformers.MobileBertForMultipleChoice.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"labels",val:": typing.Optional[torch.Tensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.MobileBertForMultipleChoice.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, num_choices, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.MobileBertForMultipleChoice.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.MobileBertForMultipleChoice.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, num_choices, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.MobileBertForMultipleChoice.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, num_choices, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.max_position_embeddings - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.MobileBertForMultipleChoice.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.MobileBertForMultipleChoice.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_choices, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.MobileBertForMultipleChoice.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the multiple choice classification loss. Indices should be in <code>[0, ..., num_choices-1]</code> where <code>num_choices</code> is the size of the second dimension of the input tensors. (See
<code>input_ids</code> above)`,name:"labels"},{anchor:"transformers.MobileBertForMultipleChoice.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.MobileBertForMultipleChoice.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.MobileBertForMultipleChoice.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mobilebert/modeling_mobilebert.py#L1307",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.MultipleChoiceModelOutput"
>transformers.modeling_outputs.MultipleChoiceModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/mobilebert#transformers.MobileBertConfig"
>MobileBertConfig</a>) and inputs.</p>
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
`}}),Ze=new _e({props:{$$slots:{default:[Or]},$$scope:{ctx:w}}}),Ie=new Me({props:{anchor:"transformers.MobileBertForMultipleChoice.forward.example",$$slots:{default:[Dr]},$$scope:{ctx:w}}}),Bt=new V({props:{title:"MobileBertForTokenClassification",local:"transformers.MobileBertForTokenClassification",headingTag:"h2"}}),Jt=new B({props:{name:"class transformers.MobileBertForTokenClassification",anchor:"transformers.MobileBertForTokenClassification",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.MobileBertForTokenClassification.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/mobilebert#transformers.MobileBertForTokenClassification">MobileBertForTokenClassification</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mobilebert/modeling_mobilebert.py#L1401"}}),xt=new B({props:{name:"forward",anchor:"transformers.MobileBertForTokenClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"labels",val:": typing.Optional[torch.Tensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.MobileBertForTokenClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.MobileBertForTokenClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.MobileBertForTokenClassification.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.MobileBertForTokenClassification.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.MobileBertForTokenClassification.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.MobileBertForTokenClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.MobileBertForTokenClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the token classification loss. Indices should be in <code>[0, ..., config.num_labels - 1]</code>.`,name:"labels"},{anchor:"transformers.MobileBertForTokenClassification.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.MobileBertForTokenClassification.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.MobileBertForTokenClassification.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mobilebert/modeling_mobilebert.py#L1416",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput"
>transformers.modeling_outputs.TokenClassifierOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/mobilebert#transformers.MobileBertConfig"
>MobileBertConfig</a>) and inputs.</p>
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
`}}),qe=new _e({props:{$$slots:{default:[Kr]},$$scope:{ctx:w}}}),Ne=new Me({props:{anchor:"transformers.MobileBertForTokenClassification.forward.example",$$slots:{default:[ea]},$$scope:{ctx:w}}}),Ut=new V({props:{title:"MobileBertForQuestionAnswering",local:"transformers.MobileBertForQuestionAnswering",headingTag:"h2"}}),jt=new B({props:{name:"class transformers.MobileBertForQuestionAnswering",anchor:"transformers.MobileBertForQuestionAnswering",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.MobileBertForQuestionAnswering.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/mobilebert#transformers.MobileBertForQuestionAnswering">MobileBertForQuestionAnswering</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mobilebert/modeling_mobilebert.py#L1214"}}),zt=new B({props:{name:"forward",anchor:"transformers.MobileBertForQuestionAnswering.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"start_positions",val:": typing.Optional[torch.Tensor] = None"},{name:"end_positions",val:": typing.Optional[torch.Tensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.MobileBertForQuestionAnswering.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.MobileBertForQuestionAnswering.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.MobileBertForQuestionAnswering.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.MobileBertForQuestionAnswering.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.MobileBertForQuestionAnswering.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.MobileBertForQuestionAnswering.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.MobileBertForQuestionAnswering.forward.start_positions",description:`<strong>start_positions</strong> (<code>torch.Tensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for position (index) of the start of the labelled span for computing the token classification loss.
Positions are clamped to the length of the sequence (<code>sequence_length</code>). Position outside of the sequence
are not taken into account for computing the loss.`,name:"start_positions"},{anchor:"transformers.MobileBertForQuestionAnswering.forward.end_positions",description:`<strong>end_positions</strong> (<code>torch.Tensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for position (index) of the end of the labelled span for computing the token classification loss.
Positions are clamped to the length of the sequence (<code>sequence_length</code>). Position outside of the sequence
are not taken into account for computing the loss.`,name:"end_positions"},{anchor:"transformers.MobileBertForQuestionAnswering.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.MobileBertForQuestionAnswering.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.MobileBertForQuestionAnswering.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/mobilebert/modeling_mobilebert.py#L1225",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.QuestionAnsweringModelOutput"
>transformers.modeling_outputs.QuestionAnsweringModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/mobilebert#transformers.MobileBertConfig"
>MobileBertConfig</a>) and inputs.</p>
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
`}}),Re=new _e({props:{$$slots:{default:[ta]},$$scope:{ctx:w}}}),Ve=new Me({props:{anchor:"transformers.MobileBertForQuestionAnswering.forward.example",$$slots:{default:[oa]},$$scope:{ctx:w}}}),Ct=new Fr({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/mobilebert.md"}}),{c(){t=c("meta"),h=r(),o=c("p"),d=r(),k=c("p"),k.innerHTML=n,T=r(),J=c("div"),J.innerHTML=xo,He=r(),u(le.$$.fragment),jo=r(),Le=c("p"),Le.innerHTML=Cs,zo=r(),Qe=c("p"),Qe.innerHTML=Fs,Co=r(),u(Te.$$.fragment),Fo=r(),Ee=c("p"),Ee.innerHTML=Ws,Wo=r(),u(ye.$$.fragment),Zo=r(),u(Pe.$$.fragment),Io=r(),Ye=c("ul"),Ye.innerHTML=Zs,qo=r(),u(Ae.$$.fragment),No=r(),G=c("div"),u(Oe.$$.fragment),pn=r(),It=c("p"),It.innerHTML=Is,mn=r(),qt=c("p"),qt.innerHTML=qs,hn=r(),u(ke.$$.fragment),Ro=r(),u(De.$$.fragment),Vo=r(),U=c("div"),u(Ke.$$.fragment),un=r(),Nt=c("p"),Nt.textContent=Ns,fn=r(),Rt=c("p"),Rt.innerHTML=Rs,gn=r(),de=c("div"),u(et.$$.fragment),bn=r(),Vt=c("p"),Vt.textContent=Vs,_n=r(),Gt=c("ul"),Gt.innerHTML=Gs,Mn=r(),we=c("div"),u(tt.$$.fragment),Tn=r(),St=c("p"),St.textContent=Ss,yn=r(),ve=c("div"),u(ot.$$.fragment),kn=r(),Xt=c("p"),Xt.innerHTML=Xs,Go=r(),u(nt.$$.fragment),So=r(),S=c("div"),u(st.$$.fragment),wn=r(),Ht=c("p"),Ht.innerHTML=Hs,vn=r(),Lt=c("p"),Lt.innerHTML=Ls,$n=r(),ce=c("div"),u(rt.$$.fragment),Bn=r(),Qt=c("p"),Qt.textContent=Qs,Jn=r(),Et=c("ul"),Et.innerHTML=Es,Xo=r(),u(at.$$.fragment),Ho=r(),ue=c("div"),u(it.$$.fragment),xn=r(),Pt=c("p"),Pt.innerHTML=Ps,Lo=r(),u(lt.$$.fragment),Qo=r(),j=c("div"),u(dt.$$.fragment),Un=r(),Yt=c("p"),Yt.textContent=Ys,jn=r(),At=c("p"),At.innerHTML=As,zn=r(),Ot=c("p"),Ot.innerHTML=Os,Cn=r(),pe=c("div"),u(ct.$$.fragment),Fn=r(),Dt=c("p"),Dt.innerHTML=Ds,Wn=r(),u($e.$$.fragment),Eo=r(),u(pt.$$.fragment),Po=r(),z=c("div"),u(mt.$$.fragment),Zn=r(),Kt=c("p"),Kt.innerHTML=Ks,In=r(),eo=c("p"),eo.innerHTML=er,qn=r(),to=c("p"),to.innerHTML=tr,Nn=r(),A=c("div"),u(ht.$$.fragment),Rn=r(),oo=c("p"),oo.innerHTML=or,Vn=r(),u(Be.$$.fragment),Gn=r(),u(Je.$$.fragment),Yo=r(),u(ut.$$.fragment),Ao=r(),C=c("div"),u(ft.$$.fragment),Sn=r(),no=c("p"),no.innerHTML=nr,Xn=r(),so=c("p"),so.innerHTML=sr,Hn=r(),ro=c("p"),ro.innerHTML=rr,Ln=r(),O=c("div"),u(gt.$$.fragment),Qn=r(),ao=c("p"),ao.innerHTML=ar,En=r(),u(xe.$$.fragment),Pn=r(),u(Ue.$$.fragment),Oo=r(),u(bt.$$.fragment),Do=r(),F=c("div"),u(_t.$$.fragment),Yn=r(),io=c("p"),io.innerHTML=ir,An=r(),lo=c("p"),lo.innerHTML=lr,On=r(),co=c("p"),co.innerHTML=dr,Dn=r(),D=c("div"),u(Mt.$$.fragment),Kn=r(),po=c("p"),po.innerHTML=cr,es=r(),u(je.$$.fragment),ts=r(),u(ze.$$.fragment),Ko=r(),u(Tt.$$.fragment),en=r(),W=c("div"),u(yt.$$.fragment),os=r(),mo=c("p"),mo.textContent=pr,ns=r(),ho=c("p"),ho.innerHTML=mr,ss=r(),uo=c("p"),uo.innerHTML=hr,rs=r(),R=c("div"),u(kt.$$.fragment),as=r(),fo=c("p"),fo.innerHTML=ur,is=r(),u(Ce.$$.fragment),ls=r(),u(Fe.$$.fragment),ds=r(),u(We.$$.fragment),tn=r(),u(wt.$$.fragment),on=r(),Z=c("div"),u(vt.$$.fragment),cs=r(),go=c("p"),go.textContent=fr,ps=r(),bo=c("p"),bo.innerHTML=gr,ms=r(),_o=c("p"),_o.innerHTML=br,hs=r(),K=c("div"),u($t.$$.fragment),us=r(),Mo=c("p"),Mo.innerHTML=_r,fs=r(),u(Ze.$$.fragment),gs=r(),u(Ie.$$.fragment),nn=r(),u(Bt.$$.fragment),sn=r(),I=c("div"),u(Jt.$$.fragment),bs=r(),To=c("p"),To.textContent=Mr,_s=r(),yo=c("p"),yo.innerHTML=Tr,Ms=r(),ko=c("p"),ko.innerHTML=yr,Ts=r(),ee=c("div"),u(xt.$$.fragment),ys=r(),wo=c("p"),wo.innerHTML=kr,ks=r(),u(qe.$$.fragment),ws=r(),u(Ne.$$.fragment),rn=r(),u(Ut.$$.fragment),an=r(),q=c("div"),u(jt.$$.fragment),vs=r(),vo=c("p"),vo.innerHTML=wr,$s=r(),$o=c("p"),$o.innerHTML=vr,Bs=r(),Bo=c("p"),Bo.innerHTML=$r,Js=r(),te=c("div"),u(zt.$$.fragment),xs=r(),Jo=c("p"),Jo.innerHTML=Br,Us=r(),u(Re.$$.fragment),js=r(),u(Ve.$$.fragment),ln=r(),u(Ct.$$.fragment),dn=r(),Uo=c("p"),this.h()},l(e){const i=zr("svelte-u9bgzb",document.head);t=p(i,"META",{name:!0,content:!0}),i.forEach(l),h=a(e),o=p(e,"P",{}),v(o).forEach(l),d=a(e),k=p(e,"P",{"data-svelte-h":!0}),y(k)!=="svelte-cn2nh3"&&(k.innerHTML=n),T=a(e),J=p(e,"DIV",{style:!0,"data-svelte-h":!0}),y(J)!=="svelte-wa5t4p"&&(J.innerHTML=xo),He=a(e),f(le.$$.fragment,e),jo=a(e),Le=p(e,"P",{"data-svelte-h":!0}),y(Le)!=="svelte-1e56x0"&&(Le.innerHTML=Cs),zo=a(e),Qe=p(e,"P",{"data-svelte-h":!0}),y(Qe)!=="svelte-1lnv8b6"&&(Qe.innerHTML=Fs),Co=a(e),f(Te.$$.fragment,e),Fo=a(e),Ee=p(e,"P",{"data-svelte-h":!0}),y(Ee)!=="svelte-lqa8w5"&&(Ee.innerHTML=Ws),Wo=a(e),f(ye.$$.fragment,e),Zo=a(e),f(Pe.$$.fragment,e),Io=a(e),Ye=p(e,"UL",{"data-svelte-h":!0}),y(Ye)!=="svelte-19zhk15"&&(Ye.innerHTML=Zs),qo=a(e),f(Ae.$$.fragment,e),No=a(e),G=p(e,"DIV",{class:!0});var ne=v(G);f(Oe.$$.fragment,ne),pn=a(ne),It=p(ne,"P",{"data-svelte-h":!0}),y(It)!=="svelte-35a8jb"&&(It.innerHTML=Is),mn=a(ne),qt=p(ne,"P",{"data-svelte-h":!0}),y(qt)!=="svelte-1ek1ss9"&&(qt.innerHTML=qs),hn=a(ne),f(ke.$$.fragment,ne),ne.forEach(l),Ro=a(e),f(De.$$.fragment,e),Vo=a(e),U=p(e,"DIV",{class:!0});var N=v(U);f(Ke.$$.fragment,N),un=a(N),Nt=p(N,"P",{"data-svelte-h":!0}),y(Nt)!=="svelte-tnfkv8"&&(Nt.textContent=Ns),fn=a(N),Rt=p(N,"P",{"data-svelte-h":!0}),y(Rt)!=="svelte-ntrhio"&&(Rt.innerHTML=Rs),gn=a(N),de=p(N,"DIV",{class:!0});var fe=v(de);f(et.$$.fragment,fe),bn=a(fe),Vt=p(fe,"P",{"data-svelte-h":!0}),y(Vt)!=="svelte-1g0ue28"&&(Vt.textContent=Vs),_n=a(fe),Gt=p(fe,"UL",{"data-svelte-h":!0}),y(Gt)!=="svelte-xi6653"&&(Gt.innerHTML=Gs),fe.forEach(l),Mn=a(N),we=p(N,"DIV",{class:!0});var Ft=v(we);f(tt.$$.fragment,Ft),Tn=a(Ft),St=p(Ft,"P",{"data-svelte-h":!0}),y(St)!=="svelte-b3k2yi"&&(St.textContent=Ss),Ft.forEach(l),yn=a(N),ve=p(N,"DIV",{class:!0});var Wt=v(ve);f(ot.$$.fragment,Wt),kn=a(Wt),Xt=p(Wt,"P",{"data-svelte-h":!0}),y(Xt)!=="svelte-1f4f5kp"&&(Xt.innerHTML=Xs),Wt.forEach(l),N.forEach(l),Go=a(e),f(nt.$$.fragment,e),So=a(e),S=p(e,"DIV",{class:!0});var se=v(S);f(st.$$.fragment,se),wn=a(se),Ht=p(se,"P",{"data-svelte-h":!0}),y(Ht)!=="svelte-uqw469"&&(Ht.innerHTML=Hs),vn=a(se),Lt=p(se,"P",{"data-svelte-h":!0}),y(Lt)!=="svelte-gxzj9w"&&(Lt.innerHTML=Ls),$n=a(se),ce=p(se,"DIV",{class:!0});var ge=v(ce);f(rt.$$.fragment,ge),Bn=a(ge),Qt=p(ge,"P",{"data-svelte-h":!0}),y(Qt)!=="svelte-1g0ue28"&&(Qt.textContent=Qs),Jn=a(ge),Et=p(ge,"UL",{"data-svelte-h":!0}),y(Et)!=="svelte-xi6653"&&(Et.innerHTML=Es),ge.forEach(l),se.forEach(l),Xo=a(e),f(at.$$.fragment,e),Ho=a(e),ue=p(e,"DIV",{class:!0});var Zt=v(ue);f(it.$$.fragment,Zt),xn=a(Zt),Pt=p(Zt,"P",{"data-svelte-h":!0}),y(Pt)!=="svelte-3zvrhw"&&(Pt.innerHTML=Ps),Zt.forEach(l),Lo=a(e),f(lt.$$.fragment,e),Qo=a(e),j=p(e,"DIV",{class:!0});var X=v(j);f(dt.$$.fragment,X),Un=a(X),Yt=p(X,"P",{"data-svelte-h":!0}),y(Yt)!=="svelte-1td5s7x"&&(Yt.textContent=Ys),jn=a(X),At=p(X,"P",{"data-svelte-h":!0}),y(At)!=="svelte-q52n56"&&(At.innerHTML=As),zn=a(X),Ot=p(X,"P",{"data-svelte-h":!0}),y(Ot)!=="svelte-hswkmf"&&(Ot.innerHTML=Os),Cn=a(X),pe=p(X,"DIV",{class:!0});var be=v(pe);f(ct.$$.fragment,be),Fn=a(be),Dt=p(be,"P",{"data-svelte-h":!0}),y(Dt)!=="svelte-1wpfuqy"&&(Dt.innerHTML=Ds),Wn=a(be),f($e.$$.fragment,be),be.forEach(l),X.forEach(l),Eo=a(e),f(pt.$$.fragment,e),Po=a(e),z=p(e,"DIV",{class:!0});var H=v(z);f(mt.$$.fragment,H),Zn=a(H),Kt=p(H,"P",{"data-svelte-h":!0}),y(Kt)!=="svelte-13rs89z"&&(Kt.innerHTML=Ks),In=a(H),eo=p(H,"P",{"data-svelte-h":!0}),y(eo)!=="svelte-q52n56"&&(eo.innerHTML=er),qn=a(H),to=p(H,"P",{"data-svelte-h":!0}),y(to)!=="svelte-hswkmf"&&(to.innerHTML=tr),Nn=a(H),A=p(H,"DIV",{class:!0});var re=v(A);f(ht.$$.fragment,re),Rn=a(re),oo=p(re,"P",{"data-svelte-h":!0}),y(oo)!=="svelte-15ezwg4"&&(oo.innerHTML=or),Vn=a(re),f(Be.$$.fragment,re),Gn=a(re),f(Je.$$.fragment,re),re.forEach(l),H.forEach(l),Yo=a(e),f(ut.$$.fragment,e),Ao=a(e),C=p(e,"DIV",{class:!0});var L=v(C);f(ft.$$.fragment,L),Sn=a(L),no=p(L,"P",{"data-svelte-h":!0}),y(no)!=="svelte-27xk4v"&&(no.innerHTML=nr),Xn=a(L),so=p(L,"P",{"data-svelte-h":!0}),y(so)!=="svelte-q52n56"&&(so.innerHTML=sr),Hn=a(L),ro=p(L,"P",{"data-svelte-h":!0}),y(ro)!=="svelte-hswkmf"&&(ro.innerHTML=rr),Ln=a(L),O=p(L,"DIV",{class:!0});var ae=v(O);f(gt.$$.fragment,ae),Qn=a(ae),ao=p(ae,"P",{"data-svelte-h":!0}),y(ao)!=="svelte-1a8juy2"&&(ao.innerHTML=ar),En=a(ae),f(xe.$$.fragment,ae),Pn=a(ae),f(Ue.$$.fragment,ae),ae.forEach(l),L.forEach(l),Oo=a(e),f(bt.$$.fragment,e),Do=a(e),F=p(e,"DIV",{class:!0});var Q=v(F);f(_t.$$.fragment,Q),Yn=a(Q),io=p(Q,"P",{"data-svelte-h":!0}),y(io)!=="svelte-1kymcwy"&&(io.innerHTML=ir),An=a(Q),lo=p(Q,"P",{"data-svelte-h":!0}),y(lo)!=="svelte-q52n56"&&(lo.innerHTML=lr),On=a(Q),co=p(Q,"P",{"data-svelte-h":!0}),y(co)!=="svelte-hswkmf"&&(co.innerHTML=dr),Dn=a(Q),D=p(Q,"DIV",{class:!0});var ie=v(D);f(Mt.$$.fragment,ie),Kn=a(ie),po=p(ie,"P",{"data-svelte-h":!0}),y(po)!=="svelte-rk4tb6"&&(po.innerHTML=cr),es=a(ie),f(je.$$.fragment,ie),ts=a(ie),f(ze.$$.fragment,ie),ie.forEach(l),Q.forEach(l),Ko=a(e),f(Tt.$$.fragment,e),en=a(e),W=p(e,"DIV",{class:!0});var E=v(W);f(yt.$$.fragment,E),os=a(E),mo=p(E,"P",{"data-svelte-h":!0}),y(mo)!=="svelte-1nkv4mt"&&(mo.textContent=pr),ns=a(E),ho=p(E,"P",{"data-svelte-h":!0}),y(ho)!=="svelte-q52n56"&&(ho.innerHTML=mr),ss=a(E),uo=p(E,"P",{"data-svelte-h":!0}),y(uo)!=="svelte-hswkmf"&&(uo.innerHTML=hr),rs=a(E),R=p(E,"DIV",{class:!0});var P=v(R);f(kt.$$.fragment,P),as=a(P),fo=p(P,"P",{"data-svelte-h":!0}),y(fo)!=="svelte-1kf8yty"&&(fo.innerHTML=ur),is=a(P),f(Ce.$$.fragment,P),ls=a(P),f(Fe.$$.fragment,P),ds=a(P),f(We.$$.fragment,P),P.forEach(l),E.forEach(l),tn=a(e),f(wt.$$.fragment,e),on=a(e),Z=p(e,"DIV",{class:!0});var Y=v(Z);f(vt.$$.fragment,Y),cs=a(Y),go=p(Y,"P",{"data-svelte-h":!0}),y(go)!=="svelte-1y5gq42"&&(go.textContent=fr),ps=a(Y),bo=p(Y,"P",{"data-svelte-h":!0}),y(bo)!=="svelte-q52n56"&&(bo.innerHTML=gr),ms=a(Y),_o=p(Y,"P",{"data-svelte-h":!0}),y(_o)!=="svelte-hswkmf"&&(_o.innerHTML=br),hs=a(Y),K=p(Y,"DIV",{class:!0});var Ge=v(K);f($t.$$.fragment,Ge),us=a(Ge),Mo=p(Ge,"P",{"data-svelte-h":!0}),y(Mo)!=="svelte-1vsjzbi"&&(Mo.innerHTML=_r),fs=a(Ge),f(Ze.$$.fragment,Ge),gs=a(Ge),f(Ie.$$.fragment,Ge),Ge.forEach(l),Y.forEach(l),nn=a(e),f(Bt.$$.fragment,e),sn=a(e),I=p(e,"DIV",{class:!0});var me=v(I);f(Jt.$$.fragment,me),bs=a(me),To=p(me,"P",{"data-svelte-h":!0}),y(To)!=="svelte-c2yh8z"&&(To.textContent=Mr),_s=a(me),yo=p(me,"P",{"data-svelte-h":!0}),y(yo)!=="svelte-q52n56"&&(yo.innerHTML=Tr),Ms=a(me),ko=p(me,"P",{"data-svelte-h":!0}),y(ko)!=="svelte-hswkmf"&&(ko.innerHTML=yr),Ts=a(me),ee=p(me,"DIV",{class:!0});var Se=v(ee);f(xt.$$.fragment,Se),ys=a(Se),wo=p(Se,"P",{"data-svelte-h":!0}),y(wo)!=="svelte-o8faz0"&&(wo.innerHTML=kr),ks=a(Se),f(qe.$$.fragment,Se),ws=a(Se),f(Ne.$$.fragment,Se),Se.forEach(l),me.forEach(l),rn=a(e),f(Ut.$$.fragment,e),an=a(e),q=p(e,"DIV",{class:!0});var he=v(q);f(jt.$$.fragment,he),vs=a(he),vo=p(he,"P",{"data-svelte-h":!0}),y(vo)!=="svelte-17e2y6m"&&(vo.innerHTML=wr),$s=a(he),$o=p(he,"P",{"data-svelte-h":!0}),y($o)!=="svelte-q52n56"&&($o.innerHTML=vr),Bs=a(he),Bo=p(he,"P",{"data-svelte-h":!0}),y(Bo)!=="svelte-hswkmf"&&(Bo.innerHTML=$r),Js=a(he),te=p(he,"DIV",{class:!0});var Xe=v(te);f(zt.$$.fragment,Xe),xs=a(Xe),Jo=p(Xe,"P",{"data-svelte-h":!0}),y(Jo)!=="svelte-1wd2thi"&&(Jo.innerHTML=Br),Us=a(Xe),f(Re.$$.fragment,Xe),js=a(Xe),f(Ve.$$.fragment,Xe),Xe.forEach(l),he.forEach(l),ln=a(e),f(Ct.$$.fragment,e),dn=a(e),Uo=p(e,"P",{}),v(Uo).forEach(l),this.h()},h(){$(t,"name","hf:doc:metadata"),$(t,"content",sa),Cr(J,"float","right"),$(G,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(de,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(we,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(ve,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(U,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(ce,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(S,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(ue,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(pe,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(j,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(A,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(z,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(O,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(C,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(D,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(F,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(R,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(W,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(K,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(Z,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(ee,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(I,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(te,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(q,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,i){s(document.head,t),m(e,h,i),m(e,o,i),m(e,d,i),m(e,k,i),m(e,T,i),m(e,J,i),m(e,He,i),g(le,e,i),m(e,jo,i),m(e,Le,i),m(e,zo,i),m(e,Qe,i),m(e,Co,i),g(Te,e,i),m(e,Fo,i),m(e,Ee,i),m(e,Wo,i),g(ye,e,i),m(e,Zo,i),g(Pe,e,i),m(e,Io,i),m(e,Ye,i),m(e,qo,i),g(Ae,e,i),m(e,No,i),m(e,G,i),g(Oe,G,null),s(G,pn),s(G,It),s(G,mn),s(G,qt),s(G,hn),g(ke,G,null),m(e,Ro,i),g(De,e,i),m(e,Vo,i),m(e,U,i),g(Ke,U,null),s(U,un),s(U,Nt),s(U,fn),s(U,Rt),s(U,gn),s(U,de),g(et,de,null),s(de,bn),s(de,Vt),s(de,_n),s(de,Gt),s(U,Mn),s(U,we),g(tt,we,null),s(we,Tn),s(we,St),s(U,yn),s(U,ve),g(ot,ve,null),s(ve,kn),s(ve,Xt),m(e,Go,i),g(nt,e,i),m(e,So,i),m(e,S,i),g(st,S,null),s(S,wn),s(S,Ht),s(S,vn),s(S,Lt),s(S,$n),s(S,ce),g(rt,ce,null),s(ce,Bn),s(ce,Qt),s(ce,Jn),s(ce,Et),m(e,Xo,i),g(at,e,i),m(e,Ho,i),m(e,ue,i),g(it,ue,null),s(ue,xn),s(ue,Pt),m(e,Lo,i),g(lt,e,i),m(e,Qo,i),m(e,j,i),g(dt,j,null),s(j,Un),s(j,Yt),s(j,jn),s(j,At),s(j,zn),s(j,Ot),s(j,Cn),s(j,pe),g(ct,pe,null),s(pe,Fn),s(pe,Dt),s(pe,Wn),g($e,pe,null),m(e,Eo,i),g(pt,e,i),m(e,Po,i),m(e,z,i),g(mt,z,null),s(z,Zn),s(z,Kt),s(z,In),s(z,eo),s(z,qn),s(z,to),s(z,Nn),s(z,A),g(ht,A,null),s(A,Rn),s(A,oo),s(A,Vn),g(Be,A,null),s(A,Gn),g(Je,A,null),m(e,Yo,i),g(ut,e,i),m(e,Ao,i),m(e,C,i),g(ft,C,null),s(C,Sn),s(C,no),s(C,Xn),s(C,so),s(C,Hn),s(C,ro),s(C,Ln),s(C,O),g(gt,O,null),s(O,Qn),s(O,ao),s(O,En),g(xe,O,null),s(O,Pn),g(Ue,O,null),m(e,Oo,i),g(bt,e,i),m(e,Do,i),m(e,F,i),g(_t,F,null),s(F,Yn),s(F,io),s(F,An),s(F,lo),s(F,On),s(F,co),s(F,Dn),s(F,D),g(Mt,D,null),s(D,Kn),s(D,po),s(D,es),g(je,D,null),s(D,ts),g(ze,D,null),m(e,Ko,i),g(Tt,e,i),m(e,en,i),m(e,W,i),g(yt,W,null),s(W,os),s(W,mo),s(W,ns),s(W,ho),s(W,ss),s(W,uo),s(W,rs),s(W,R),g(kt,R,null),s(R,as),s(R,fo),s(R,is),g(Ce,R,null),s(R,ls),g(Fe,R,null),s(R,ds),g(We,R,null),m(e,tn,i),g(wt,e,i),m(e,on,i),m(e,Z,i),g(vt,Z,null),s(Z,cs),s(Z,go),s(Z,ps),s(Z,bo),s(Z,ms),s(Z,_o),s(Z,hs),s(Z,K),g($t,K,null),s(K,us),s(K,Mo),s(K,fs),g(Ze,K,null),s(K,gs),g(Ie,K,null),m(e,nn,i),g(Bt,e,i),m(e,sn,i),m(e,I,i),g(Jt,I,null),s(I,bs),s(I,To),s(I,_s),s(I,yo),s(I,Ms),s(I,ko),s(I,Ts),s(I,ee),g(xt,ee,null),s(ee,ys),s(ee,wo),s(ee,ks),g(qe,ee,null),s(ee,ws),g(Ne,ee,null),m(e,rn,i),g(Ut,e,i),m(e,an,i),m(e,q,i),g(jt,q,null),s(q,vs),s(q,vo),s(q,$s),s(q,$o),s(q,Bs),s(q,Bo),s(q,Js),s(q,te),g(zt,te,null),s(te,xs),s(te,Jo),s(te,Us),g(Re,te,null),s(te,js),g(Ve,te,null),m(e,ln,i),g(Ct,e,i),m(e,dn,i),m(e,Uo,i),cn=!0},p(e,[i]){const ne={};i&2&&(ne.$$scope={dirty:i,ctx:e}),Te.$set(ne);const N={};i&2&&(N.$$scope={dirty:i,ctx:e}),ye.$set(N);const fe={};i&2&&(fe.$$scope={dirty:i,ctx:e}),ke.$set(fe);const Ft={};i&2&&(Ft.$$scope={dirty:i,ctx:e}),$e.$set(Ft);const Wt={};i&2&&(Wt.$$scope={dirty:i,ctx:e}),Be.$set(Wt);const se={};i&2&&(se.$$scope={dirty:i,ctx:e}),Je.$set(se);const ge={};i&2&&(ge.$$scope={dirty:i,ctx:e}),xe.$set(ge);const Zt={};i&2&&(Zt.$$scope={dirty:i,ctx:e}),Ue.$set(Zt);const X={};i&2&&(X.$$scope={dirty:i,ctx:e}),je.$set(X);const be={};i&2&&(be.$$scope={dirty:i,ctx:e}),ze.$set(be);const H={};i&2&&(H.$$scope={dirty:i,ctx:e}),Ce.$set(H);const re={};i&2&&(re.$$scope={dirty:i,ctx:e}),Fe.$set(re);const L={};i&2&&(L.$$scope={dirty:i,ctx:e}),We.$set(L);const ae={};i&2&&(ae.$$scope={dirty:i,ctx:e}),Ze.$set(ae);const Q={};i&2&&(Q.$$scope={dirty:i,ctx:e}),Ie.$set(Q);const ie={};i&2&&(ie.$$scope={dirty:i,ctx:e}),qe.$set(ie);const E={};i&2&&(E.$$scope={dirty:i,ctx:e}),Ne.$set(E);const P={};i&2&&(P.$$scope={dirty:i,ctx:e}),Re.$set(P);const Y={};i&2&&(Y.$$scope={dirty:i,ctx:e}),Ve.$set(Y)},i(e){cn||(b(le.$$.fragment,e),b(Te.$$.fragment,e),b(ye.$$.fragment,e),b(Pe.$$.fragment,e),b(Ae.$$.fragment,e),b(Oe.$$.fragment,e),b(ke.$$.fragment,e),b(De.$$.fragment,e),b(Ke.$$.fragment,e),b(et.$$.fragment,e),b(tt.$$.fragment,e),b(ot.$$.fragment,e),b(nt.$$.fragment,e),b(st.$$.fragment,e),b(rt.$$.fragment,e),b(at.$$.fragment,e),b(it.$$.fragment,e),b(lt.$$.fragment,e),b(dt.$$.fragment,e),b(ct.$$.fragment,e),b($e.$$.fragment,e),b(pt.$$.fragment,e),b(mt.$$.fragment,e),b(ht.$$.fragment,e),b(Be.$$.fragment,e),b(Je.$$.fragment,e),b(ut.$$.fragment,e),b(ft.$$.fragment,e),b(gt.$$.fragment,e),b(xe.$$.fragment,e),b(Ue.$$.fragment,e),b(bt.$$.fragment,e),b(_t.$$.fragment,e),b(Mt.$$.fragment,e),b(je.$$.fragment,e),b(ze.$$.fragment,e),b(Tt.$$.fragment,e),b(yt.$$.fragment,e),b(kt.$$.fragment,e),b(Ce.$$.fragment,e),b(Fe.$$.fragment,e),b(We.$$.fragment,e),b(wt.$$.fragment,e),b(vt.$$.fragment,e),b($t.$$.fragment,e),b(Ze.$$.fragment,e),b(Ie.$$.fragment,e),b(Bt.$$.fragment,e),b(Jt.$$.fragment,e),b(xt.$$.fragment,e),b(qe.$$.fragment,e),b(Ne.$$.fragment,e),b(Ut.$$.fragment,e),b(jt.$$.fragment,e),b(zt.$$.fragment,e),b(Re.$$.fragment,e),b(Ve.$$.fragment,e),b(Ct.$$.fragment,e),cn=!0)},o(e){_(le.$$.fragment,e),_(Te.$$.fragment,e),_(ye.$$.fragment,e),_(Pe.$$.fragment,e),_(Ae.$$.fragment,e),_(Oe.$$.fragment,e),_(ke.$$.fragment,e),_(De.$$.fragment,e),_(Ke.$$.fragment,e),_(et.$$.fragment,e),_(tt.$$.fragment,e),_(ot.$$.fragment,e),_(nt.$$.fragment,e),_(st.$$.fragment,e),_(rt.$$.fragment,e),_(at.$$.fragment,e),_(it.$$.fragment,e),_(lt.$$.fragment,e),_(dt.$$.fragment,e),_(ct.$$.fragment,e),_($e.$$.fragment,e),_(pt.$$.fragment,e),_(mt.$$.fragment,e),_(ht.$$.fragment,e),_(Be.$$.fragment,e),_(Je.$$.fragment,e),_(ut.$$.fragment,e),_(ft.$$.fragment,e),_(gt.$$.fragment,e),_(xe.$$.fragment,e),_(Ue.$$.fragment,e),_(bt.$$.fragment,e),_(_t.$$.fragment,e),_(Mt.$$.fragment,e),_(je.$$.fragment,e),_(ze.$$.fragment,e),_(Tt.$$.fragment,e),_(yt.$$.fragment,e),_(kt.$$.fragment,e),_(Ce.$$.fragment,e),_(Fe.$$.fragment,e),_(We.$$.fragment,e),_(wt.$$.fragment,e),_(vt.$$.fragment,e),_($t.$$.fragment,e),_(Ze.$$.fragment,e),_(Ie.$$.fragment,e),_(Bt.$$.fragment,e),_(Jt.$$.fragment,e),_(xt.$$.fragment,e),_(qe.$$.fragment,e),_(Ne.$$.fragment,e),_(Ut.$$.fragment,e),_(jt.$$.fragment,e),_(zt.$$.fragment,e),_(Re.$$.fragment,e),_(Ve.$$.fragment,e),_(Ct.$$.fragment,e),cn=!1},d(e){e&&(l(h),l(o),l(d),l(k),l(T),l(J),l(He),l(jo),l(Le),l(zo),l(Qe),l(Co),l(Fo),l(Ee),l(Wo),l(Zo),l(Io),l(Ye),l(qo),l(No),l(G),l(Ro),l(Vo),l(U),l(Go),l(So),l(S),l(Xo),l(Ho),l(ue),l(Lo),l(Qo),l(j),l(Eo),l(Po),l(z),l(Yo),l(Ao),l(C),l(Oo),l(Do),l(F),l(Ko),l(en),l(W),l(tn),l(on),l(Z),l(nn),l(sn),l(I),l(rn),l(an),l(q),l(ln),l(dn),l(Uo)),l(t),M(le,e),M(Te,e),M(ye,e),M(Pe,e),M(Ae,e),M(Oe),M(ke),M(De,e),M(Ke),M(et),M(tt),M(ot),M(nt,e),M(st),M(rt),M(at,e),M(it),M(lt,e),M(dt),M(ct),M($e),M(pt,e),M(mt),M(ht),M(Be),M(Je),M(ut,e),M(ft),M(gt),M(xe),M(Ue),M(bt,e),M(_t),M(Mt),M(je),M(ze),M(Tt,e),M(yt),M(kt),M(Ce),M(Fe),M(We),M(wt,e),M(vt),M($t),M(Ze),M(Ie),M(Bt,e),M(Jt),M(xt),M(qe),M(Ne),M(Ut,e),M(jt),M(zt),M(Re),M(Ve),M(Ct,e)}}}const sa='{"title":"MobileBERT","local":"mobilebert","sections":[{"title":"Notes","local":"notes","sections":[],"depth":2},{"title":"MobileBertConfig","local":"transformers.MobileBertConfig","sections":[],"depth":2},{"title":"MobileBertTokenizer","local":"transformers.MobileBertTokenizer","sections":[],"depth":2},{"title":"MobileBertTokenizerFast","local":"transformers.MobileBertTokenizerFast","sections":[],"depth":2},{"title":"MobileBert specific outputs","local":"transformers.models.mobilebert.modeling_mobilebert.MobileBertForPreTrainingOutput","sections":[],"depth":2},{"title":"MobileBertModel","local":"transformers.MobileBertModel","sections":[],"depth":2},{"title":"MobileBertForPreTraining","local":"transformers.MobileBertForPreTraining","sections":[],"depth":2},{"title":"MobileBertForMaskedLM","local":"transformers.MobileBertForMaskedLM","sections":[],"depth":2},{"title":"MobileBertForNextSentencePrediction","local":"transformers.MobileBertForNextSentencePrediction","sections":[],"depth":2},{"title":"MobileBertForSequenceClassification","local":"transformers.MobileBertForSequenceClassification","sections":[],"depth":2},{"title":"MobileBertForMultipleChoice","local":"transformers.MobileBertForMultipleChoice","sections":[],"depth":2},{"title":"MobileBertForTokenClassification","local":"transformers.MobileBertForTokenClassification","sections":[],"depth":2},{"title":"MobileBertForQuestionAnswering","local":"transformers.MobileBertForQuestionAnswering","sections":[],"depth":2}],"depth":1}';function ra(w){return xr(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class ua extends Ur{constructor(t){super(),jr(this,t,ra,na,Jr,{})}}export{ua as component};
