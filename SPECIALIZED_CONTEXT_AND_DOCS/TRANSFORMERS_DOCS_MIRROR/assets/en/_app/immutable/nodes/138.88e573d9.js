import{s as sa,o as aa,n as V}from"../chunks/scheduler.18a86fab.js";import{S as ra,i as ia,g as c,s as a,r as f,A as la,h as p,f as i,c as r,j as v,x as T,u as g,k as $,l as da,y as o,a as m,v as b,d as _,t as y,w as M}from"../chunks/index.98837b22.js";import{T as qe}from"../chunks/Tip.77304350.js";import{D as J}from"../chunks/Docstring.a1ef7999.js";import{C as te}from"../chunks/CodeBlock.8d0c2e8a.js";import{E as Jt}from"../chunks/ExampleCodeBlock.8c3ee1f9.js";import{H as ee,E as ca}from"../chunks/getInferenceSnippets.06c2775f.js";import{H as pa,a as ps}from"../chunks/HfOption.6641485e.js";function ma(w){let t,u='This model was contributed by <a href="https://huggingface.co/DeBERTa" rel="nofollow">Pengcheng He</a>.',n,d,k="Click on the DeBERTa-v2 models in the right sidebar for more examples of how to apply DeBERTa-v2 to different language tasks.";return{c(){t=c("p"),t.innerHTML=u,n=a(),d=c("p"),d.textContent=k},l(s){t=p(s,"P",{"data-svelte-h":!0}),T(t)!=="svelte-cfdmd0"&&(t.innerHTML=u),n=r(s),d=p(s,"P",{"data-svelte-h":!0}),T(d)!=="svelte-rdf5cz"&&(d.textContent=k)},m(s,h){m(s,t,h),m(s,n,h),m(s,d,h)},p:V,d(s){s&&(i(t),i(n),i(d))}}}function ua(w){let t,u;return t=new te({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwcGlwZWxpbmUlMEElMEFwaXBlbGluZSUyMCUzRCUyMHBpcGVsaW5lKCUwQSUyMCUyMCUyMCUyMHRhc2slM0QlMjJ0ZXh0LWNsYXNzaWZpY2F0aW9uJTIyJTJDJTBBJTIwJTIwJTIwJTIwbW9kZWwlM0QlMjJtaWNyb3NvZnQlMkZkZWJlcnRhLXYyLXhsYXJnZS1tbmxpJTIyJTJDJTBBJTIwJTIwJTIwJTIwZGV2aWNlJTNEMCUyQyUwQSUyMCUyMCUyMCUyMGR0eXBlJTNEdG9yY2guZmxvYXQxNiUwQSklMEFyZXN1bHQlMjAlM0QlMjBwaXBlbGluZSglMjJEZUJFUlRhLXYyJTIwaXMlMjBncmVhdCUyMGF0JTIwdW5kZXJzdGFuZGluZyUyMGNvbnRleHQhJTIyKSUwQXByaW50KHJlc3VsdCk=",highlighted:`<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> pipeline

pipeline = pipeline(
    task=<span class="hljs-string">&quot;text-classification&quot;</span>,
    model=<span class="hljs-string">&quot;microsoft/deberta-v2-xlarge-mnli&quot;</span>,
    device=<span class="hljs-number">0</span>,
    dtype=torch.float16
)
result = pipeline(<span class="hljs-string">&quot;DeBERTa-v2 is great at understanding context!&quot;</span>)
<span class="hljs-built_in">print</span>(result)`,wrap:!1}}),{c(){f(t.$$.fragment)},l(n){g(t.$$.fragment,n)},m(n,d){b(t,n,d),u=!0},p:V,i(n){u||(_(t.$$.fragment,n),u=!0)},o(n){y(t.$$.fragment,n),u=!1},d(n){M(t,n)}}}function ha(w){let t,u;return t=new te({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Rva2VuaXplciUyQyUyMEF1dG9Nb2RlbEZvclNlcXVlbmNlQ2xhc3NpZmljYXRpb24lMEElMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMEElMjAlMjAlMjAlMjAlMjJtaWNyb3NvZnQlMkZkZWJlcnRhLXYyLXhsYXJnZS1tbmxpJTIyJTBBKSUwQW1vZGVsJTIwJTNEJTIwQXV0b01vZGVsRm9yU2VxdWVuY2VDbGFzc2lmaWNhdGlvbi5mcm9tX3ByZXRyYWluZWQoJTBBJTIwJTIwJTIwJTIwJTIybWljcm9zb2Z0JTJGZGViZXJ0YS12Mi14bGFyZ2UtbW5saSUyMiUyQyUwQSUyMCUyMCUyMCUyMGR0eXBlJTNEdG9yY2guZmxvYXQxNiUyQyUwQSUyMCUyMCUyMCUyMGRldmljZV9tYXAlM0QlMjJhdXRvJTIyJTBBKSUwQSUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplciglMjJEZUJFUlRhLXYyJTIwaXMlMjBncmVhdCUyMGF0JTIwdW5kZXJzdGFuZGluZyUyMGNvbnRleHQhJTIyJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMikudG8obW9kZWwuZGV2aWNlKSUwQW91dHB1dHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyklMEElMEFsb2dpdHMlMjAlM0QlMjBvdXRwdXRzLmxvZ2l0cyUwQXByZWRpY3RlZF9jbGFzc19pZCUyMCUzRCUyMGxvZ2l0cy5hcmdtYXgoKS5pdGVtKCklMEFwcmVkaWN0ZWRfbGFiZWwlMjAlM0QlMjBtb2RlbC5jb25maWcuaWQybGFiZWwlNUJwcmVkaWN0ZWRfY2xhc3NfaWQlNUQlMEFwcmludChmJTIyUHJlZGljdGVkJTIwbGFiZWwlM0ElMjAlN0JwcmVkaWN0ZWRfbGFiZWwlN0QlMjIpJTBB",highlighted:`<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained(
    <span class="hljs-string">&quot;microsoft/deberta-v2-xlarge-mnli&quot;</span>
)
model = AutoModelForSequenceClassification.from_pretrained(
    <span class="hljs-string">&quot;microsoft/deberta-v2-xlarge-mnli&quot;</span>,
    dtype=torch.float16,
    device_map=<span class="hljs-string">&quot;auto&quot;</span>
)

inputs = tokenizer(<span class="hljs-string">&quot;DeBERTa-v2 is great at understanding context!&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>).to(model.device)
outputs = model(**inputs)

logits = outputs.logits
predicted_class_id = logits.argmax().item()
predicted_label = model.config.id2label[predicted_class_id]
<span class="hljs-built_in">print</span>(<span class="hljs-string">f&quot;Predicted label: <span class="hljs-subst">{predicted_label}</span>&quot;</span>)
`,wrap:!1}}),{c(){f(t.$$.fragment)},l(n){g(t.$$.fragment,n)},m(n,d){b(t,n,d),u=!0},p:V,i(n){u||(_(t.$$.fragment,n),u=!0)},o(n){y(t.$$.fragment,n),u=!1},d(n){M(t,n)}}}function fa(w){let t,u;return t=new te({props:{code:"ZWNobyUyMC1lJTIwJTIyRGVCRVJUYS12MiUyMGlzJTIwZ3JlYXQlMjBhdCUyMHVuZGVyc3RhbmRpbmclMjBjb250ZXh0ISUyMiUyMCU3QyUyMHRyYW5zZm9ybWVycy1jbGklMjBydW4lMjAtLXRhc2slMjBmaWxsLW1hc2slMjAtLW1vZGVsJTIwbWljcm9zb2Z0JTJGZGViZXJ0YS12Mi14bGFyZ2UtbW5saSUyMC0tZGV2aWNlJTIwMA==",highlighted:'<span class="hljs-built_in">echo</span> -e <span class="hljs-string">&quot;DeBERTa-v2 is great at understanding context!&quot;</span> | transformers-cli run --task fill-mask --model microsoft/deberta-v2-xlarge-mnli --device 0',wrap:!1}}),{c(){f(t.$$.fragment)},l(n){g(t.$$.fragment,n)},m(n,d){b(t,n,d),u=!0},p:V,i(n){u||(_(t.$$.fragment,n),u=!0)},o(n){y(t.$$.fragment,n),u=!1},d(n){M(t,n)}}}function ga(w){let t,u,n,d,k,s;return t=new ps({props:{id:"usage",option:"Pipeline",$$slots:{default:[ua]},$$scope:{ctx:w}}}),n=new ps({props:{id:"usage",option:"AutoModel",$$slots:{default:[ha]},$$scope:{ctx:w}}}),k=new ps({props:{id:"usage",option:"transformers CLI",$$slots:{default:[fa]},$$scope:{ctx:w}}}),{c(){f(t.$$.fragment),u=a(),f(n.$$.fragment),d=a(),f(k.$$.fragment)},l(h){g(t.$$.fragment,h),u=r(h),g(n.$$.fragment,h),d=r(h),g(k.$$.fragment,h)},m(h,j){b(t,h,j),m(h,u,j),b(n,h,j),m(h,d,j),b(k,h,j),s=!0},p(h,j){const gn={};j&2&&(gn.$$scope={dirty:j,ctx:h}),t.$set(gn);const Ie={};j&2&&(Ie.$$scope={dirty:j,ctx:h}),n.$set(Ie);const ne={};j&2&&(ne.$$scope={dirty:j,ctx:h}),k.$set(ne)},i(h){s||(_(t.$$.fragment,h),_(n.$$.fragment,h),_(k.$$.fragment,h),s=!0)},o(h){y(t.$$.fragment,h),y(n.$$.fragment,h),y(k.$$.fragment,h),s=!1},d(h){h&&(i(u),i(d)),M(t,h),M(n,h),M(k,h)}}}function ba(w){let t,u="Example:",n,d,k;return d=new te({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMERlYmVydGFWMkNvbmZpZyUyQyUyMERlYmVydGFWMk1vZGVsJTBBJTBBJTIzJTIwSW5pdGlhbGl6aW5nJTIwYSUyMERlQkVSVGEtdjIlMjBtaWNyb3NvZnQlMkZkZWJlcnRhLXYyLXhsYXJnZSUyMHN0eWxlJTIwY29uZmlndXJhdGlvbiUwQWNvbmZpZ3VyYXRpb24lMjAlM0QlMjBEZWJlcnRhVjJDb25maWcoKSUwQSUwQSUyMyUyMEluaXRpYWxpemluZyUyMGElMjBtb2RlbCUyMCh3aXRoJTIwcmFuZG9tJTIwd2VpZ2h0cyklMjBmcm9tJTIwdGhlJTIwbWljcm9zb2Z0JTJGZGViZXJ0YS12Mi14bGFyZ2UlMjBzdHlsZSUyMGNvbmZpZ3VyYXRpb24lMEFtb2RlbCUyMCUzRCUyMERlYmVydGFWMk1vZGVsKGNvbmZpZ3VyYXRpb24pJTBBJTBBJTIzJTIwQWNjZXNzaW5nJTIwdGhlJTIwbW9kZWwlMjBjb25maWd1cmF0aW9uJTBBY29uZmlndXJhdGlvbiUyMCUzRCUyMG1vZGVsLmNvbmZpZw==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> DebertaV2Config, DebertaV2Model

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a DeBERTa-v2 microsoft/deberta-v2-xlarge style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = DebertaV2Config()

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a model (with random weights) from the microsoft/deberta-v2-xlarge style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = DebertaV2Model(configuration)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Accessing the model configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = model.config`,wrap:!1}}),{c(){t=c("p"),t.textContent=u,n=a(),f(d.$$.fragment)},l(s){t=p(s,"P",{"data-svelte-h":!0}),T(t)!=="svelte-11lpom8"&&(t.textContent=u),n=r(s),g(d.$$.fragment,s)},m(s,h){m(s,t,h),m(s,n,h),b(d,s,h),k=!0},p:V,i(s){k||(_(d.$$.fragment,s),k=!0)},o(s){y(d.$$.fragment,s),k=!1},d(s){s&&(i(t),i(n)),M(d,s)}}}function _a(w){let t,u=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=u},l(n){t=p(n,"P",{"data-svelte-h":!0}),T(t)!=="svelte-fincs2"&&(t.innerHTML=u)},m(n,d){m(n,t,d)},p:V,d(n){n&&i(t)}}}function ya(w){let t,u=`Although the recipe for forward pass needs to be defined within
this function, one should call the <code>Module</code> instance afterwards
instead of this since the former takes care of running the
registered hooks while the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=u},l(n){t=p(n,"P",{"data-svelte-h":!0}),T(t)!=="svelte-rqqap8"&&(t.innerHTML=u)},m(n,d){m(n,t,d)},p:V,d(n){n&&i(t)}}}function Ma(w){let t,u=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=u},l(n){t=p(n,"P",{"data-svelte-h":!0}),T(t)!=="svelte-fincs2"&&(t.innerHTML=u)},m(n,d){m(n,t,d)},p:V,d(n){n&&i(t)}}}function Ta(w){let t,u="Example:",n,d,k;return d=new te({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBEZWJlcnRhVjJGb3JNYXNrZWRMTSUwQWltcG9ydCUyMHRvcmNoJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIybWljcm9zb2Z0JTJGZGViZXJ0YS12Mi14bGFyZ2UlMjIpJTBBbW9kZWwlMjAlM0QlMjBEZWJlcnRhVjJGb3JNYXNrZWRMTS5mcm9tX3ByZXRyYWluZWQoJTIybWljcm9zb2Z0JTJGZGViZXJ0YS12Mi14bGFyZ2UlMjIpJTBBJTBBaW5wdXRzJTIwJTNEJTIwdG9rZW5pemVyKCUyMlRoZSUyMGNhcGl0YWwlMjBvZiUyMEZyYW5jZSUyMGlzJTIwJTNDbWFzayUzRS4lMjIlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQSUwQXdpdGglMjB0b3JjaC5ub19ncmFkKCklM0ElMEElMjAlMjAlMjAlMjBsb2dpdHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cykubG9naXRzJTBBJTBBJTIzJTIwcmV0cmlldmUlMjBpbmRleCUyMG9mJTIwJTNDbWFzayUzRSUwQW1hc2tfdG9rZW5faW5kZXglMjAlM0QlMjAoaW5wdXRzLmlucHV0X2lkcyUyMCUzRCUzRCUyMHRva2VuaXplci5tYXNrX3Rva2VuX2lkKSU1QjAlNUQubm9uemVybyhhc190dXBsZSUzRFRydWUpJTVCMCU1RCUwQSUwQXByZWRpY3RlZF90b2tlbl9pZCUyMCUzRCUyMGxvZ2l0cyU1QjAlMkMlMjBtYXNrX3Rva2VuX2luZGV4JTVELmFyZ21heChheGlzJTNELTEpJTBBdG9rZW5pemVyLmRlY29kZShwcmVkaWN0ZWRfdG9rZW5faWQpJTBBJTBBbGFiZWxzJTIwJTNEJTIwdG9rZW5pemVyKCUyMlRoZSUyMGNhcGl0YWwlMjBvZiUyMEZyYW5jZSUyMGlzJTIwUGFyaXMuJTIyJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiklNUIlMjJpbnB1dF9pZHMlMjIlNUQlMEElMjMlMjBtYXNrJTIwbGFiZWxzJTIwb2YlMjBub24tJTNDbWFzayUzRSUyMHRva2VucyUwQWxhYmVscyUyMCUzRCUyMHRvcmNoLndoZXJlKGlucHV0cy5pbnB1dF9pZHMlMjAlM0QlM0QlMjB0b2tlbml6ZXIubWFza190b2tlbl9pZCUyQyUyMGxhYmVscyUyQyUyMC0xMDApJTBBJTBBb3V0cHV0cyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzJTJDJTIwbGFiZWxzJTNEbGFiZWxzKSUwQXJvdW5kKG91dHB1dHMubG9zcy5pdGVtKCklMkMlMjAyKQ==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, DebertaV2ForMaskedLM
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;microsoft/deberta-v2-xlarge&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = DebertaV2ForMaskedLM.from_pretrained(<span class="hljs-string">&quot;microsoft/deberta-v2-xlarge&quot;</span>)

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
...`,wrap:!1}}),{c(){t=c("p"),t.textContent=u,n=a(),f(d.$$.fragment)},l(s){t=p(s,"P",{"data-svelte-h":!0}),T(t)!=="svelte-11lpom8"&&(t.textContent=u),n=r(s),g(d.$$.fragment,s)},m(s,h){m(s,t,h),m(s,n,h),b(d,s,h),k=!0},p:V,i(s){k||(_(d.$$.fragment,s),k=!0)},o(s){y(d.$$.fragment,s),k=!1},d(s){s&&(i(t),i(n)),M(d,s)}}}function ka(w){let t,u=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=u},l(n){t=p(n,"P",{"data-svelte-h":!0}),T(t)!=="svelte-fincs2"&&(t.innerHTML=u)},m(n,d){m(n,t,d)},p:V,d(n){n&&i(t)}}}function wa(w){let t,u="Example of single-label classification:",n,d,k;return d=new te({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Rva2VuaXplciUyQyUyMERlYmVydGFWMkZvclNlcXVlbmNlQ2xhc3NpZmljYXRpb24lMEElMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJtaWNyb3NvZnQlMkZkZWJlcnRhLXYyLXhsYXJnZSUyMiklMEFtb2RlbCUyMCUzRCUyMERlYmVydGFWMkZvclNlcXVlbmNlQ2xhc3NpZmljYXRpb24uZnJvbV9wcmV0cmFpbmVkKCUyMm1pY3Jvc29mdCUyRmRlYmVydGEtdjIteGxhcmdlJTIyKSUwQSUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplciglMjJIZWxsbyUyQyUyMG15JTIwZG9nJTIwaXMlMjBjdXRlJTIyJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiklMEElMEF3aXRoJTIwdG9yY2gubm9fZ3JhZCgpJTNBJTBBJTIwJTIwJTIwJTIwbG9naXRzJTIwJTNEJTIwbW9kZWwoKippbnB1dHMpLmxvZ2l0cyUwQSUwQXByZWRpY3RlZF9jbGFzc19pZCUyMCUzRCUyMGxvZ2l0cy5hcmdtYXgoKS5pdGVtKCklMEFtb2RlbC5jb25maWcuaWQybGFiZWwlNUJwcmVkaWN0ZWRfY2xhc3NfaWQlNUQlMEElMEElMjMlMjBUbyUyMHRyYWluJTIwYSUyMG1vZGVsJTIwb24lMjAlNjBudW1fbGFiZWxzJTYwJTIwY2xhc3NlcyUyQyUyMHlvdSUyMGNhbiUyMHBhc3MlMjAlNjBudW1fbGFiZWxzJTNEbnVtX2xhYmVscyU2MCUyMHRvJTIwJTYwLmZyb21fcHJldHJhaW5lZCguLi4pJTYwJTBBbnVtX2xhYmVscyUyMCUzRCUyMGxlbihtb2RlbC5jb25maWcuaWQybGFiZWwpJTBBbW9kZWwlMjAlM0QlMjBEZWJlcnRhVjJGb3JTZXF1ZW5jZUNsYXNzaWZpY2F0aW9uLmZyb21fcHJldHJhaW5lZCglMjJtaWNyb3NvZnQlMkZkZWJlcnRhLXYyLXhsYXJnZSUyMiUyQyUyMG51bV9sYWJlbHMlM0RudW1fbGFiZWxzKSUwQSUwQWxhYmVscyUyMCUzRCUyMHRvcmNoLnRlbnNvciglNUIxJTVEKSUwQWxvc3MlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyUyQyUyMGxhYmVscyUzRGxhYmVscykubG9zcyUwQXJvdW5kKGxvc3MuaXRlbSgpJTJDJTIwMik=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, DebertaV2ForSequenceClassification

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;microsoft/deberta-v2-xlarge&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = DebertaV2ForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;microsoft/deberta-v2-xlarge&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    logits = model(**inputs).logits

<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_class_id = logits.argmax().item()
<span class="hljs-meta">&gt;&gt;&gt; </span>model.config.id2label[predicted_class_id]
...

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># To train a model on \`num_labels\` classes, you can pass \`num_labels=num_labels\` to \`.from_pretrained(...)\`</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>num_labels = <span class="hljs-built_in">len</span>(model.config.id2label)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = DebertaV2ForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;microsoft/deberta-v2-xlarge&quot;</span>, num_labels=num_labels)

<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.tensor([<span class="hljs-number">1</span>])
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = model(**inputs, labels=labels).loss
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">round</span>(loss.item(), <span class="hljs-number">2</span>)
...`,wrap:!1}}),{c(){t=c("p"),t.textContent=u,n=a(),f(d.$$.fragment)},l(s){t=p(s,"P",{"data-svelte-h":!0}),T(t)!=="svelte-ykxpe4"&&(t.textContent=u),n=r(s),g(d.$$.fragment,s)},m(s,h){m(s,t,h),m(s,n,h),b(d,s,h),k=!0},p:V,i(s){k||(_(d.$$.fragment,s),k=!0)},o(s){y(d.$$.fragment,s),k=!1},d(s){s&&(i(t),i(n)),M(d,s)}}}function va(w){let t,u="Example of multi-label classification:",n,d,k;return d=new te({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Rva2VuaXplciUyQyUyMERlYmVydGFWMkZvclNlcXVlbmNlQ2xhc3NpZmljYXRpb24lMEElMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJtaWNyb3NvZnQlMkZkZWJlcnRhLXYyLXhsYXJnZSUyMiklMEFtb2RlbCUyMCUzRCUyMERlYmVydGFWMkZvclNlcXVlbmNlQ2xhc3NpZmljYXRpb24uZnJvbV9wcmV0cmFpbmVkKCUyMm1pY3Jvc29mdCUyRmRlYmVydGEtdjIteGxhcmdlJTIyJTJDJTIwcHJvYmxlbV90eXBlJTNEJTIybXVsdGlfbGFiZWxfY2xhc3NpZmljYXRpb24lMjIpJTBBJTBBaW5wdXRzJTIwJTNEJTIwdG9rZW5pemVyKCUyMkhlbGxvJTJDJTIwbXklMjBkb2clMjBpcyUyMGN1dGUlMjIlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQSUwQXdpdGglMjB0b3JjaC5ub19ncmFkKCklM0ElMEElMjAlMjAlMjAlMjBsb2dpdHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cykubG9naXRzJTBBJTBBcHJlZGljdGVkX2NsYXNzX2lkcyUyMCUzRCUyMHRvcmNoLmFyYW5nZSgwJTJDJTIwbG9naXRzLnNoYXBlJTVCLTElNUQpJTVCdG9yY2guc2lnbW9pZChsb2dpdHMpLnNxdWVlemUoZGltJTNEMCklMjAlM0UlMjAwLjUlNUQlMEElMEElMjMlMjBUbyUyMHRyYWluJTIwYSUyMG1vZGVsJTIwb24lMjAlNjBudW1fbGFiZWxzJTYwJTIwY2xhc3NlcyUyQyUyMHlvdSUyMGNhbiUyMHBhc3MlMjAlNjBudW1fbGFiZWxzJTNEbnVtX2xhYmVscyU2MCUyMHRvJTIwJTYwLmZyb21fcHJldHJhaW5lZCguLi4pJTYwJTBBbnVtX2xhYmVscyUyMCUzRCUyMGxlbihtb2RlbC5jb25maWcuaWQybGFiZWwpJTBBbW9kZWwlMjAlM0QlMjBEZWJlcnRhVjJGb3JTZXF1ZW5jZUNsYXNzaWZpY2F0aW9uLmZyb21fcHJldHJhaW5lZCglMEElMjAlMjAlMjAlMjAlMjJtaWNyb3NvZnQlMkZkZWJlcnRhLXYyLXhsYXJnZSUyMiUyQyUyMG51bV9sYWJlbHMlM0RudW1fbGFiZWxzJTJDJTIwcHJvYmxlbV90eXBlJTNEJTIybXVsdGlfbGFiZWxfY2xhc3NpZmljYXRpb24lMjIlMEEpJTBBJTBBbGFiZWxzJTIwJTNEJTIwdG9yY2guc3VtKCUwQSUyMCUyMCUyMCUyMHRvcmNoLm5uLmZ1bmN0aW9uYWwub25lX2hvdChwcmVkaWN0ZWRfY2xhc3NfaWRzJTVCTm9uZSUyQyUyMCUzQSU1RC5jbG9uZSgpJTJDJTIwbnVtX2NsYXNzZXMlM0RudW1fbGFiZWxzKSUyQyUyMGRpbSUzRDElMEEpLnRvKHRvcmNoLmZsb2F0KSUwQWxvc3MlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyUyQyUyMGxhYmVscyUzRGxhYmVscykubG9zcw==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, DebertaV2ForSequenceClassification

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;microsoft/deberta-v2-xlarge&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = DebertaV2ForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;microsoft/deberta-v2-xlarge&quot;</span>, problem_type=<span class="hljs-string">&quot;multi_label_classification&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    logits = model(**inputs).logits

<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_class_ids = torch.arange(<span class="hljs-number">0</span>, logits.shape[-<span class="hljs-number">1</span>])[torch.sigmoid(logits).squeeze(dim=<span class="hljs-number">0</span>) &gt; <span class="hljs-number">0.5</span>]

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># To train a model on \`num_labels\` classes, you can pass \`num_labels=num_labels\` to \`.from_pretrained(...)\`</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>num_labels = <span class="hljs-built_in">len</span>(model.config.id2label)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = DebertaV2ForSequenceClassification.from_pretrained(
<span class="hljs-meta">... </span>    <span class="hljs-string">&quot;microsoft/deberta-v2-xlarge&quot;</span>, num_labels=num_labels, problem_type=<span class="hljs-string">&quot;multi_label_classification&quot;</span>
<span class="hljs-meta">... </span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.<span class="hljs-built_in">sum</span>(
<span class="hljs-meta">... </span>    torch.nn.functional.one_hot(predicted_class_ids[<span class="hljs-literal">None</span>, :].clone(), num_classes=num_labels), dim=<span class="hljs-number">1</span>
<span class="hljs-meta">... </span>).to(torch.<span class="hljs-built_in">float</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = model(**inputs, labels=labels).loss`,wrap:!1}}),{c(){t=c("p"),t.textContent=u,n=a(),f(d.$$.fragment)},l(s){t=p(s,"P",{"data-svelte-h":!0}),T(t)!=="svelte-1l8e32d"&&(t.textContent=u),n=r(s),g(d.$$.fragment,s)},m(s,h){m(s,t,h),m(s,n,h),b(d,s,h),k=!0},p:V,i(s){k||(_(d.$$.fragment,s),k=!0)},o(s){y(d.$$.fragment,s),k=!1},d(s){s&&(i(t),i(n)),M(d,s)}}}function $a(w){let t,u=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=u},l(n){t=p(n,"P",{"data-svelte-h":!0}),T(t)!=="svelte-fincs2"&&(t.innerHTML=u)},m(n,d){m(n,t,d)},p:V,d(n){n&&i(t)}}}function Ja(w){let t,u="Example:",n,d,k;return d=new te({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBEZWJlcnRhVjJGb3JUb2tlbkNsYXNzaWZpY2F0aW9uJTBBaW1wb3J0JTIwdG9yY2glMEElMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJtaWNyb3NvZnQlMkZkZWJlcnRhLXYyLXhsYXJnZSUyMiklMEFtb2RlbCUyMCUzRCUyMERlYmVydGFWMkZvclRva2VuQ2xhc3NpZmljYXRpb24uZnJvbV9wcmV0cmFpbmVkKCUyMm1pY3Jvc29mdCUyRmRlYmVydGEtdjIteGxhcmdlJTIyKSUwQSUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplciglMEElMjAlMjAlMjAlMjAlMjJIdWdnaW5nRmFjZSUyMGlzJTIwYSUyMGNvbXBhbnklMjBiYXNlZCUyMGluJTIwUGFyaXMlMjBhbmQlMjBOZXclMjBZb3JrJTIyJTJDJTIwYWRkX3NwZWNpYWxfdG9rZW5zJTNERmFsc2UlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyJTBBKSUwQSUwQXdpdGglMjB0b3JjaC5ub19ncmFkKCklM0ElMEElMjAlMjAlMjAlMjBsb2dpdHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cykubG9naXRzJTBBJTBBcHJlZGljdGVkX3Rva2VuX2NsYXNzX2lkcyUyMCUzRCUyMGxvZ2l0cy5hcmdtYXgoLTEpJTBBJTBBJTIzJTIwTm90ZSUyMHRoYXQlMjB0b2tlbnMlMjBhcmUlMjBjbGFzc2lmaWVkJTIwcmF0aGVyJTIwdGhlbiUyMGlucHV0JTIwd29yZHMlMjB3aGljaCUyMG1lYW5zJTIwdGhhdCUwQSUyMyUyMHRoZXJlJTIwbWlnaHQlMjBiZSUyMG1vcmUlMjBwcmVkaWN0ZWQlMjB0b2tlbiUyMGNsYXNzZXMlMjB0aGFuJTIwd29yZHMuJTBBJTIzJTIwTXVsdGlwbGUlMjB0b2tlbiUyMGNsYXNzZXMlMjBtaWdodCUyMGFjY291bnQlMjBmb3IlMjB0aGUlMjBzYW1lJTIwd29yZCUwQXByZWRpY3RlZF90b2tlbnNfY2xhc3NlcyUyMCUzRCUyMCU1Qm1vZGVsLmNvbmZpZy5pZDJsYWJlbCU1QnQuaXRlbSgpJTVEJTIwZm9yJTIwdCUyMGluJTIwcHJlZGljdGVkX3Rva2VuX2NsYXNzX2lkcyU1QjAlNUQlNUQlMEFwcmVkaWN0ZWRfdG9rZW5zX2NsYXNzZXMlMEElMEFsYWJlbHMlMjAlM0QlMjBwcmVkaWN0ZWRfdG9rZW5fY2xhc3NfaWRzJTBBbG9zcyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzJTJDJTIwbGFiZWxzJTNEbGFiZWxzKS5sb3NzJTBBcm91bmQobG9zcy5pdGVtKCklMkMlMjAyKQ==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, DebertaV2ForTokenClassification
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;microsoft/deberta-v2-xlarge&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = DebertaV2ForTokenClassification.from_pretrained(<span class="hljs-string">&quot;microsoft/deberta-v2-xlarge&quot;</span>)

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
...`,wrap:!1}}),{c(){t=c("p"),t.textContent=u,n=a(),f(d.$$.fragment)},l(s){t=p(s,"P",{"data-svelte-h":!0}),T(t)!=="svelte-11lpom8"&&(t.textContent=u),n=r(s),g(d.$$.fragment,s)},m(s,h){m(s,t,h),m(s,n,h),b(d,s,h),k=!0},p:V,i(s){k||(_(d.$$.fragment,s),k=!0)},o(s){y(d.$$.fragment,s),k=!1},d(s){s&&(i(t),i(n)),M(d,s)}}}function ja(w){let t,u=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=u},l(n){t=p(n,"P",{"data-svelte-h":!0}),T(t)!=="svelte-fincs2"&&(t.innerHTML=u)},m(n,d){m(n,t,d)},p:V,d(n){n&&i(t)}}}function Va(w){let t,u="Example:",n,d,k;return d=new te({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBEZWJlcnRhVjJGb3JRdWVzdGlvbkFuc3dlcmluZyUwQWltcG9ydCUyMHRvcmNoJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIybWljcm9zb2Z0JTJGZGViZXJ0YS12Mi14bGFyZ2UlMjIpJTBBbW9kZWwlMjAlM0QlMjBEZWJlcnRhVjJGb3JRdWVzdGlvbkFuc3dlcmluZy5mcm9tX3ByZXRyYWluZWQoJTIybWljcm9zb2Z0JTJGZGViZXJ0YS12Mi14bGFyZ2UlMjIpJTBBJTBBcXVlc3Rpb24lMkMlMjB0ZXh0JTIwJTNEJTIwJTIyV2hvJTIwd2FzJTIwSmltJTIwSGVuc29uJTNGJTIyJTJDJTIwJTIySmltJTIwSGVuc29uJTIwd2FzJTIwYSUyMG5pY2UlMjBwdXBwZXQlMjIlMEElMEFpbnB1dHMlMjAlM0QlMjB0b2tlbml6ZXIocXVlc3Rpb24lMkMlMjB0ZXh0JTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiklMEF3aXRoJTIwdG9yY2gubm9fZ3JhZCgpJTNBJTBBJTIwJTIwJTIwJTIwb3V0cHV0cyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzKSUwQSUwQWFuc3dlcl9zdGFydF9pbmRleCUyMCUzRCUyMG91dHB1dHMuc3RhcnRfbG9naXRzLmFyZ21heCgpJTBBYW5zd2VyX2VuZF9pbmRleCUyMCUzRCUyMG91dHB1dHMuZW5kX2xvZ2l0cy5hcmdtYXgoKSUwQSUwQXByZWRpY3RfYW5zd2VyX3Rva2VucyUyMCUzRCUyMGlucHV0cy5pbnB1dF9pZHMlNUIwJTJDJTIwYW5zd2VyX3N0YXJ0X2luZGV4JTIwJTNBJTIwYW5zd2VyX2VuZF9pbmRleCUyMCUyQiUyMDElNUQlMEF0b2tlbml6ZXIuZGVjb2RlKHByZWRpY3RfYW5zd2VyX3Rva2VucyUyQyUyMHNraXBfc3BlY2lhbF90b2tlbnMlM0RUcnVlKSUwQSUwQSUyMyUyMHRhcmdldCUyMGlzJTIwJTIybmljZSUyMHB1cHBldCUyMiUwQXRhcmdldF9zdGFydF9pbmRleCUyMCUzRCUyMHRvcmNoLnRlbnNvciglNUIxNCU1RCklMEF0YXJnZXRfZW5kX2luZGV4JTIwJTNEJTIwdG9yY2gudGVuc29yKCU1QjE1JTVEKSUwQSUwQW91dHB1dHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyUyQyUyMHN0YXJ0X3Bvc2l0aW9ucyUzRHRhcmdldF9zdGFydF9pbmRleCUyQyUyMGVuZF9wb3NpdGlvbnMlM0R0YXJnZXRfZW5kX2luZGV4KSUwQWxvc3MlMjAlM0QlMjBvdXRwdXRzLmxvc3MlMEFyb3VuZChsb3NzLml0ZW0oKSUyQyUyMDIp",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, DebertaV2ForQuestionAnswering
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;microsoft/deberta-v2-xlarge&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = DebertaV2ForQuestionAnswering.from_pretrained(<span class="hljs-string">&quot;microsoft/deberta-v2-xlarge&quot;</span>)

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
...`,wrap:!1}}),{c(){t=c("p"),t.textContent=u,n=a(),f(d.$$.fragment)},l(s){t=p(s,"P",{"data-svelte-h":!0}),T(t)!=="svelte-11lpom8"&&(t.textContent=u),n=r(s),g(d.$$.fragment,s)},m(s,h){m(s,t,h),m(s,n,h),b(d,s,h),k=!0},p:V,i(s){k||(_(d.$$.fragment,s),k=!0)},o(s){y(d.$$.fragment,s),k=!1},d(s){s&&(i(t),i(n)),M(d,s)}}}function Ca(w){let t,u=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=u},l(n){t=p(n,"P",{"data-svelte-h":!0}),T(t)!=="svelte-fincs2"&&(t.innerHTML=u)},m(n,d){m(n,t,d)},p:V,d(n){n&&i(t)}}}function xa(w){let t,u="Example:",n,d,k;return d=new te({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBEZWJlcnRhVjJGb3JNdWx0aXBsZUNob2ljZSUwQWltcG9ydCUyMHRvcmNoJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIybWljcm9zb2Z0JTJGZGViZXJ0YS12Mi14bGFyZ2UlMjIpJTBBbW9kZWwlMjAlM0QlMjBEZWJlcnRhVjJGb3JNdWx0aXBsZUNob2ljZS5mcm9tX3ByZXRyYWluZWQoJTIybWljcm9zb2Z0JTJGZGViZXJ0YS12Mi14bGFyZ2UlMjIpJTBBJTBBcHJvbXB0JTIwJTNEJTIwJTIySW4lMjBJdGFseSUyQyUyMHBpenphJTIwc2VydmVkJTIwaW4lMjBmb3JtYWwlMjBzZXR0aW5ncyUyQyUyMHN1Y2glMjBhcyUyMGF0JTIwYSUyMHJlc3RhdXJhbnQlMkMlMjBpcyUyMHByZXNlbnRlZCUyMHVuc2xpY2VkLiUyMiUwQWNob2ljZTAlMjAlM0QlMjAlMjJJdCUyMGlzJTIwZWF0ZW4lMjB3aXRoJTIwYSUyMGZvcmslMjBhbmQlMjBhJTIwa25pZmUuJTIyJTBBY2hvaWNlMSUyMCUzRCUyMCUyMkl0JTIwaXMlMjBlYXRlbiUyMHdoaWxlJTIwaGVsZCUyMGluJTIwdGhlJTIwaGFuZC4lMjIlMEFsYWJlbHMlMjAlM0QlMjB0b3JjaC50ZW5zb3IoMCkudW5zcXVlZXplKDApJTIwJTIwJTIzJTIwY2hvaWNlMCUyMGlzJTIwY29ycmVjdCUyMChhY2NvcmRpbmclMjB0byUyMFdpa2lwZWRpYSUyMCUzQikpJTJDJTIwYmF0Y2glMjBzaXplJTIwMSUwQSUwQWVuY29kaW5nJTIwJTNEJTIwdG9rZW5pemVyKCU1QnByb21wdCUyQyUyMHByb21wdCU1RCUyQyUyMCU1QmNob2ljZTAlMkMlMjBjaG9pY2UxJTVEJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiUyQyUyMHBhZGRpbmclM0RUcnVlKSUwQW91dHB1dHMlMjAlM0QlMjBtb2RlbCgqKiU3QmslM0ElMjB2LnVuc3F1ZWV6ZSgwKSUyMGZvciUyMGslMkMlMjB2JTIwaW4lMjBlbmNvZGluZy5pdGVtcygpJTdEJTJDJTIwbGFiZWxzJTNEbGFiZWxzKSUyMCUyMCUyMyUyMGJhdGNoJTIwc2l6ZSUyMGlzJTIwMSUwQSUwQSUyMyUyMHRoZSUyMGxpbmVhciUyMGNsYXNzaWZpZXIlMjBzdGlsbCUyMG5lZWRzJTIwdG8lMjBiZSUyMHRyYWluZWQlMEFsb3NzJTIwJTNEJTIwb3V0cHV0cy5sb3NzJTBBbG9naXRzJTIwJTNEJTIwb3V0cHV0cy5sb2dpdHM=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, DebertaV2ForMultipleChoice
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;microsoft/deberta-v2-xlarge&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = DebertaV2ForMultipleChoice.from_pretrained(<span class="hljs-string">&quot;microsoft/deberta-v2-xlarge&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>prompt = <span class="hljs-string">&quot;In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>choice0 = <span class="hljs-string">&quot;It is eaten with a fork and a knife.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>choice1 = <span class="hljs-string">&quot;It is eaten while held in the hand.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.tensor(<span class="hljs-number">0</span>).unsqueeze(<span class="hljs-number">0</span>)  <span class="hljs-comment"># choice0 is correct (according to Wikipedia ;)), batch size 1</span>

<span class="hljs-meta">&gt;&gt;&gt; </span>encoding = tokenizer([prompt, prompt], [choice0, choice1], return_tensors=<span class="hljs-string">&quot;pt&quot;</span>, padding=<span class="hljs-literal">True</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**{k: v.unsqueeze(<span class="hljs-number">0</span>) <span class="hljs-keyword">for</span> k, v <span class="hljs-keyword">in</span> encoding.items()}, labels=labels)  <span class="hljs-comment"># batch size is 1</span>

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># the linear classifier still needs to be trained</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = outputs.loss
<span class="hljs-meta">&gt;&gt;&gt; </span>logits = outputs.logits`,wrap:!1}}),{c(){t=c("p"),t.textContent=u,n=a(),f(d.$$.fragment)},l(s){t=p(s,"P",{"data-svelte-h":!0}),T(t)!=="svelte-11lpom8"&&(t.textContent=u),n=r(s),g(d.$$.fragment,s)},m(s,h){m(s,t,h),m(s,n,h),b(d,s,h),k=!0},p:V,i(s){k||(_(d.$$.fragment,s),k=!0)},o(s){y(d.$$.fragment,s),k=!1},d(s){s&&(i(t),i(n)),M(d,s)}}}function Ua(w){let t,u,n,d,k,s="<em>This model was released on 2020-06-05 and added to Hugging Face Transformers on 2021-02-19.</em>",h,j,gn='<div class="flex flex-wrap space-x-1"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/></div>',Ie,ne,yn,Be,ms='<a href="https://huggingface.co/papers/2006.03654" rel="nofollow">DeBERTa-v2</a> improves on the original <a href="./deberta">DeBERTa</a> architecture by using a SentencePiece-based tokenizer and a new vocabulary size of 128K. It also adds an additional convolutional layer within the first transformer layer to better learn local dependencies of input tokens. Finally, the position projection and content projection matrices are shared in the attention layer to reduce the number of parameters.',Mn,Re,us='You can find all the original [DeBERTa-v2] checkpoints under the <a href="https://huggingface.co/microsoft?search_models=deberta-v2" rel="nofollow">Microsoft</a> organization.',Tn,be,kn,Ge,hs='The example below demonstrates how to classify text with <a href="/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.Pipeline">Pipeline</a> or the <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel">AutoModel</a> class.',wn,_e,vn,Ne,fs='Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the <a href="../quantization/overview">Quantization</a> overview for more available quantization backends.',$n,Xe,gs='The example below uses <a href="../quantization/bitsandbytes">bitsandbytes quantization</a> to only quantize the weights to 4-bit.',Jn,Se,jn,Qe,Vn,I,He,Pn,jt,bs=`This is the configuration class to store the configuration of a <a href="/docs/transformers/v4.56.2/en/model_doc/deberta-v2#transformers.DebertaV2Model">DebertaV2Model</a>. It is used to instantiate a
DeBERTa-v2 model according to the specified arguments, defining the model architecture. Instantiating a
configuration with the defaults will yield a similar configuration to that of the DeBERTa
<a href="https://huggingface.co/microsoft/deberta-v2-xlarge" rel="nofollow">microsoft/deberta-v2-xlarge</a> architecture.`,An,Vt,_s=`Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> and can be used to control the model outputs. Read the
documentation from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> for more information.`,On,ye,Cn,Ee,xn,C,Le,Kn,Ct,ys='Constructs a DeBERTa-v2 tokenizer. Based on <a href="https://github.com/google/sentencepiece" rel="nofollow">SentencePiece</a>.',eo,oe,Ye,to,xt,Ms=`Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
adding special tokens. A DeBERTa sequence has the following format:`,no,Ut,Ts="<li>single sequence: [CLS] X [SEP]</li> <li>pair of sequences: [CLS] A [SEP] B [SEP]</li>",oo,Me,Pe,so,zt,ks=`Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
special tokens using the tokenizer <code>prepare_for_model</code> or <code>encode_plus</code> methods.`,ao,se,Ae,ro,Zt,ws=`Create the token type IDs corresponding to the sequences passed. <a href="../glossary#token-type-ids">What are token type
IDs?</a>`,io,Wt,vs="Should be overridden in a subclass if the model has a special way of building those.",lo,Dt,Oe,Un,Ke,zn,B,et,co,Ft,$s='Constructs a DeBERTa-v2 fast tokenizer. Based on <a href="https://github.com/google/sentencepiece" rel="nofollow">SentencePiece</a>.',po,ae,tt,mo,qt,Js=`Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
adding special tokens. A DeBERTa sequence has the following format:`,uo,It,js="<li>single sequence: [CLS] X [SEP]</li> <li>pair of sequences: [CLS] A [SEP] B [SEP]</li>",ho,re,nt,fo,Bt,Vs=`Create the token type IDs corresponding to the sequences passed. <a href="../glossary#token-type-ids">What are token type
IDs?</a>`,go,Rt,Cs="Should be overridden in a subclass if the model has a special way of building those.",Zn,ot,Wn,x,st,bo,Gt,xs="The bare Deberta V2 Model outputting raw hidden-states without any specific head on top.",_o,Nt,Us=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,yo,Xt,zs=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Mo,ie,at,To,St,Zs='The <a href="/docs/transformers/v4.56.2/en/model_doc/deberta-v2#transformers.DebertaV2Model">DebertaV2Model</a> forward method, overrides the <code>__call__</code> special method.',ko,Te,Dn,rt,Fn,R,it,wo,Qt,Ws=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,vo,Ht,Ds=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,$o,S,lt,Jo,Et,Fs="Define the computation performed at every call.",jo,Lt,qs="Should be overridden by all subclasses.",Vo,ke,qn,dt,In,U,ct,Co,Yt,Is="The Deberta V2 Model with a <code>language modeling</code> head on top.”",xo,Pt,Bs=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Uo,At,Rs=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,zo,Q,pt,Zo,Ot,Gs='The <a href="/docs/transformers/v4.56.2/en/model_doc/deberta-v2#transformers.DebertaV2ForMaskedLM">DebertaV2ForMaskedLM</a> forward method, overrides the <code>__call__</code> special method.',Wo,we,Do,ve,Bn,mt,Rn,z,ut,Fo,Kt,Ns=`DeBERTa Model transformer with a sequence classification/regression head on top (a linear layer on top of the
pooled output) e.g. for GLUE tasks.`,qo,en,Xs=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Io,tn,Ss=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Bo,q,ht,Ro,nn,Qs='The <a href="/docs/transformers/v4.56.2/en/model_doc/deberta-v2#transformers.DebertaV2ForSequenceClassification">DebertaV2ForSequenceClassification</a> forward method, overrides the <code>__call__</code> special method.',Go,$e,No,Je,Xo,je,Gn,ft,Nn,Z,gt,So,on,Hs=`The Deberta V2 transformer with a token classification head on top (a linear layer on top of the hidden-states
output) e.g. for Named-Entity-Recognition (NER) tasks.`,Qo,sn,Es=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Ho,an,Ls=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Eo,H,bt,Lo,rn,Ys='The <a href="/docs/transformers/v4.56.2/en/model_doc/deberta-v2#transformers.DebertaV2ForTokenClassification">DebertaV2ForTokenClassification</a> forward method, overrides the <code>__call__</code> special method.',Yo,Ve,Po,Ce,Xn,_t,Sn,W,yt,Ao,ln,Ps=`The Deberta V2 transformer with a span classification head on top for extractive question-answering tasks like
SQuAD (a linear layer on top of the hidden-states output to compute <code>span start logits</code> and <code>span end logits</code>).`,Oo,dn,As=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Ko,cn,Os=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,es,E,Mt,ts,pn,Ks='The <a href="/docs/transformers/v4.56.2/en/model_doc/deberta-v2#transformers.DebertaV2ForQuestionAnswering">DebertaV2ForQuestionAnswering</a> forward method, overrides the <code>__call__</code> special method.',ns,xe,os,Ue,Qn,Tt,Hn,D,kt,ss,mn,ea=`The Deberta V2 Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
softmax) e.g. for RocStories/SWAG tasks.`,as,un,ta=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,rs,hn,na=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,is,L,wt,ls,fn,oa='The <a href="/docs/transformers/v4.56.2/en/model_doc/deberta-v2#transformers.DebertaV2ForMultipleChoice">DebertaV2ForMultipleChoice</a> forward method, overrides the <code>__call__</code> special method.',ds,ze,cs,Ze,En,vt,Ln,bn,Yn;return ne=new ee({props:{title:"DeBERTa-v2",local:"deberta-v2",headingTag:"h1"}}),be=new qe({props:{warning:!1,$$slots:{default:[ma]},$$scope:{ctx:w}}}),_e=new pa({props:{id:"usage",options:["Pipeline","AutoModel","transformers CLI"],$$slots:{default:[ga]},$$scope:{ctx:w}}}),Se=new te({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Nb2RlbEZvclNlcXVlbmNlQ2xhc3NpZmljYXRpb24lMkMlMjBBdXRvVG9rZW5pemVyJTJDJTIwQml0c0FuZEJ5dGVzQ29uZmlnJTBBJTBBbW9kZWxfaWQlMjAlM0QlMjAlMjJtaWNyb3NvZnQlMkZkZWJlcnRhLXYyLXhsYXJnZS1tbmxpJTIyJTBBcXVhbnRpemF0aW9uX2NvbmZpZyUyMCUzRCUyMEJpdHNBbmRCeXRlc0NvbmZpZyglMEElMjAlMjAlMjAlMjBsb2FkX2luXzRiaXQlM0RUcnVlJTJDJTBBJTIwJTIwJTIwJTIwYm5iXzRiaXRfcXVhbnRfdHlwZSUzRCUyMm5mNCUyMiUyQyUwQSUyMCUyMCUyMCUyMGJuYl80Yml0X2NvbXB1dGVfZHR5cGUlM0QlMjJmbG9hdDE2JTIyJTJDJTBBJTIwJTIwJTIwJTIwYm5iXzRiaXRfdXNlX2RvdWJsZV9xdWFudCUzRFRydWUlMkMlMEEpJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQobW9kZWxfaWQpJTBBbW9kZWwlMjAlM0QlMjBBdXRvTW9kZWxGb3JTZXF1ZW5jZUNsYXNzaWZpY2F0aW9uLmZyb21fcHJldHJhaW5lZCglMEElMjAlMjAlMjAlMjBtb2RlbF9pZCUyQyUwQSUyMCUyMCUyMCUyMHF1YW50aXphdGlvbl9jb25maWclM0RxdWFudGl6YXRpb25fY29uZmlnJTJDJTBBJTIwJTIwJTIwJTIwZHR5cGUlM0QlMjJmbG9hdDE2JTIyJTBBKSUwQSUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplciglMjJEZUJFUlRhLXYyJTIwaXMlMjBncmVhdCUyMGF0JTIwdW5kZXJzdGFuZGluZyUyMGNvbnRleHQhJTIyJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMikudG8obW9kZWwuZGV2aWNlKSUwQW91dHB1dHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyklMEFsb2dpdHMlMjAlM0QlMjBvdXRwdXRzLmxvZ2l0cyUwQXByZWRpY3RlZF9jbGFzc19pZCUyMCUzRCUyMGxvZ2l0cy5hcmdtYXgoKS5pdGVtKCklMEFwcmVkaWN0ZWRfbGFiZWwlMjAlM0QlMjBtb2RlbC5jb25maWcuaWQybGFiZWwlNUJwcmVkaWN0ZWRfY2xhc3NfaWQlNUQlMEFwcmludChmJTIyUHJlZGljdGVkJTIwbGFiZWwlM0ElMjAlN0JwcmVkaWN0ZWRfbGFiZWwlN0QlMjIpJTBB",highlighted:`<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig

model_id = <span class="hljs-string">&quot;microsoft/deberta-v2-xlarge-mnli&quot;</span>
quantization_config = BitsAndBytesConfig(
    load_in_4bit=<span class="hljs-literal">True</span>,
    bnb_4bit_quant_type=<span class="hljs-string">&quot;nf4&quot;</span>,
    bnb_4bit_compute_dtype=<span class="hljs-string">&quot;float16&quot;</span>,
    bnb_4bit_use_double_quant=<span class="hljs-literal">True</span>,
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    dtype=<span class="hljs-string">&quot;float16&quot;</span>
)

inputs = tokenizer(<span class="hljs-string">&quot;DeBERTa-v2 is great at understanding context!&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>).to(model.device)
outputs = model(**inputs)
logits = outputs.logits
predicted_class_id = logits.argmax().item()
predicted_label = model.config.id2label[predicted_class_id]
<span class="hljs-built_in">print</span>(<span class="hljs-string">f&quot;Predicted label: <span class="hljs-subst">{predicted_label}</span>&quot;</span>)
`,wrap:!1}}),Qe=new ee({props:{title:"DebertaV2Config",local:"transformers.DebertaV2Config",headingTag:"h2"}}),He=new J({props:{name:"class transformers.DebertaV2Config",anchor:"transformers.DebertaV2Config",parameters:[{name:"vocab_size",val:" = 128100"},{name:"hidden_size",val:" = 1536"},{name:"num_hidden_layers",val:" = 24"},{name:"num_attention_heads",val:" = 24"},{name:"intermediate_size",val:" = 6144"},{name:"hidden_act",val:" = 'gelu'"},{name:"hidden_dropout_prob",val:" = 0.1"},{name:"attention_probs_dropout_prob",val:" = 0.1"},{name:"max_position_embeddings",val:" = 512"},{name:"type_vocab_size",val:" = 0"},{name:"initializer_range",val:" = 0.02"},{name:"layer_norm_eps",val:" = 1e-07"},{name:"relative_attention",val:" = False"},{name:"max_relative_positions",val:" = -1"},{name:"pad_token_id",val:" = 0"},{name:"position_biased_input",val:" = True"},{name:"pos_att_type",val:" = None"},{name:"pooler_dropout",val:" = 0"},{name:"pooler_hidden_act",val:" = 'gelu'"},{name:"legacy",val:" = True"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.DebertaV2Config.vocab_size",description:`<strong>vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 128100) &#x2014;
Vocabulary size of the DeBERTa-v2 model. Defines the number of different tokens that can be represented by
the <code>inputs_ids</code> passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/deberta-v2#transformers.DebertaV2Model">DebertaV2Model</a>.`,name:"vocab_size"},{anchor:"transformers.DebertaV2Config.hidden_size",description:`<strong>hidden_size</strong> (<code>int</code>, <em>optional</em>, defaults to 1536) &#x2014;
Dimensionality of the encoder layers and the pooler layer.`,name:"hidden_size"},{anchor:"transformers.DebertaV2Config.num_hidden_layers",description:`<strong>num_hidden_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 24) &#x2014;
Number of hidden layers in the Transformer encoder.`,name:"num_hidden_layers"},{anchor:"transformers.DebertaV2Config.num_attention_heads",description:`<strong>num_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 24) &#x2014;
Number of attention heads for each attention layer in the Transformer encoder.`,name:"num_attention_heads"},{anchor:"transformers.DebertaV2Config.intermediate_size",description:`<strong>intermediate_size</strong> (<code>int</code>, <em>optional</em>, defaults to 6144) &#x2014;
Dimensionality of the &#x201C;intermediate&#x201D; (often named feed-forward) layer in the Transformer encoder.`,name:"intermediate_size"},{anchor:"transformers.DebertaV2Config.hidden_act",description:`<strong>hidden_act</strong> (<code>str</code> or <code>Callable</code>, <em>optional</em>, defaults to <code>&quot;gelu&quot;</code>) &#x2014;
The non-linear activation function (function or string) in the encoder and pooler. If string, <code>&quot;gelu&quot;</code>,
<code>&quot;relu&quot;</code>, <code>&quot;silu&quot;</code>, <code>&quot;gelu&quot;</code>, <code>&quot;tanh&quot;</code>, <code>&quot;gelu_fast&quot;</code>, <code>&quot;mish&quot;</code>, <code>&quot;linear&quot;</code>, <code>&quot;sigmoid&quot;</code> and <code>&quot;gelu_new&quot;</code>
are supported.`,name:"hidden_act"},{anchor:"transformers.DebertaV2Config.hidden_dropout_prob",description:`<strong>hidden_dropout_prob</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.`,name:"hidden_dropout_prob"},{anchor:"transformers.DebertaV2Config.attention_probs_dropout_prob",description:`<strong>attention_probs_dropout_prob</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout ratio for the attention probabilities.`,name:"attention_probs_dropout_prob"},{anchor:"transformers.DebertaV2Config.max_position_embeddings",description:`<strong>max_position_embeddings</strong> (<code>int</code>, <em>optional</em>, defaults to 512) &#x2014;
The maximum sequence length that this model might ever be used with. Typically set this to something large
just in case (e.g., 512 or 1024 or 2048).`,name:"max_position_embeddings"},{anchor:"transformers.DebertaV2Config.type_vocab_size",description:`<strong>type_vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 0) &#x2014;
The vocabulary size of the <code>token_type_ids</code> passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/deberta#transformers.DebertaModel">DebertaModel</a> or <code>TFDebertaModel</code>.`,name:"type_vocab_size"},{anchor:"transformers.DebertaV2Config.initializer_range",description:`<strong>initializer_range</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.`,name:"initializer_range"},{anchor:"transformers.DebertaV2Config.layer_norm_eps",description:`<strong>layer_norm_eps</strong> (<code>float</code>, <em>optional</em>, defaults to 1e-7) &#x2014;
The epsilon used by the layer normalization layers.`,name:"layer_norm_eps"},{anchor:"transformers.DebertaV2Config.relative_attention",description:`<strong>relative_attention</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether use relative position encoding.`,name:"relative_attention"},{anchor:"transformers.DebertaV2Config.max_relative_positions",description:`<strong>max_relative_positions</strong> (<code>int</code>, <em>optional</em>, defaults to -1) &#x2014;
The range of relative positions <code>[-max_position_embeddings, max_position_embeddings]</code>. Use the same value
as <code>max_position_embeddings</code>.`,name:"max_relative_positions"},{anchor:"transformers.DebertaV2Config.pad_token_id",description:`<strong>pad_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 0) &#x2014;
The value used to pad input_ids.`,name:"pad_token_id"},{anchor:"transformers.DebertaV2Config.position_biased_input",description:`<strong>position_biased_input</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether add absolute position embedding to content embedding.`,name:"position_biased_input"},{anchor:"transformers.DebertaV2Config.pos_att_type",description:`<strong>pos_att_type</strong> (<code>list[str]</code>, <em>optional</em>) &#x2014;
The type of relative position attention, it can be a combination of <code>[&quot;p2c&quot;, &quot;c2p&quot;]</code>, e.g. <code>[&quot;p2c&quot;]</code>,
<code>[&quot;p2c&quot;, &quot;c2p&quot;]</code>, <code>[&quot;p2c&quot;, &quot;c2p&quot;]</code>.`,name:"pos_att_type"},{anchor:"transformers.DebertaV2Config.layer_norm_eps",description:`<strong>layer_norm_eps</strong> (<code>float</code>, <em>optional</em>, defaults to 1e-12) &#x2014;
The epsilon used by the layer normalization layers.`,name:"layer_norm_eps"},{anchor:"transformers.DebertaV2Config.legacy",description:`<strong>legacy</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not the model should use the legacy <code>LegacyDebertaOnlyMLMHead</code>, which does not work properly
for mask infilling tasks.`,name:"legacy"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deberta_v2/configuration_deberta_v2.py#L33"}}),ye=new Jt({props:{anchor:"transformers.DebertaV2Config.example",$$slots:{default:[ba]},$$scope:{ctx:w}}}),Ee=new ee({props:{title:"DebertaV2Tokenizer",local:"transformers.DebertaV2Tokenizer",headingTag:"h2"}}),Le=new J({props:{name:"class transformers.DebertaV2Tokenizer",anchor:"transformers.DebertaV2Tokenizer",parameters:[{name:"vocab_file",val:""},{name:"do_lower_case",val:" = False"},{name:"split_by_punct",val:" = False"},{name:"bos_token",val:" = '[CLS]'"},{name:"eos_token",val:" = '[SEP]'"},{name:"unk_token",val:" = '[UNK]'"},{name:"sep_token",val:" = '[SEP]'"},{name:"pad_token",val:" = '[PAD]'"},{name:"cls_token",val:" = '[CLS]'"},{name:"mask_token",val:" = '[MASK]'"},{name:"sp_model_kwargs",val:": typing.Optional[dict[str, typing.Any]] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.DebertaV2Tokenizer.vocab_file",description:`<strong>vocab_file</strong> (<code>str</code>) &#x2014;
<a href="https://github.com/google/sentencepiece" rel="nofollow">SentencePiece</a> file (generally has a <em>.spm</em> extension) that
contains the vocabulary necessary to instantiate a tokenizer.`,name:"vocab_file"},{anchor:"transformers.DebertaV2Tokenizer.do_lower_case",description:`<strong>do_lower_case</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to lowercase the input when tokenizing.`,name:"do_lower_case"},{anchor:"transformers.DebertaV2Tokenizer.bos_token",description:`<strong>bos_token</strong> (<code>string</code>, <em>optional</em>, defaults to <code>&quot;[CLS]&quot;</code>) &#x2014;
The beginning of sequence token that was used during pre-training. Can be used a sequence classifier token.
When building a sequence using special tokens, this is not the token that is used for the beginning of
sequence. The token used is the <code>cls_token</code>.`,name:"bos_token"},{anchor:"transformers.DebertaV2Tokenizer.eos_token",description:`<strong>eos_token</strong> (<code>string</code>, <em>optional</em>, defaults to <code>&quot;[SEP]&quot;</code>) &#x2014;
The end of sequence token. When building a sequence using special tokens, this is not the token that is
used for the end of sequence. The token used is the <code>sep_token</code>.`,name:"eos_token"},{anchor:"transformers.DebertaV2Tokenizer.unk_token",description:`<strong>unk_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;[UNK]&quot;</code>) &#x2014;
The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
token instead.`,name:"unk_token"},{anchor:"transformers.DebertaV2Tokenizer.sep_token",description:`<strong>sep_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;[SEP]&quot;</code>) &#x2014;
The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
sequence classification or for a text and a question for question answering. It is also used as the last
token of a sequence built with special tokens.`,name:"sep_token"},{anchor:"transformers.DebertaV2Tokenizer.pad_token",description:`<strong>pad_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;[PAD]&quot;</code>) &#x2014;
The token used for padding, for example when batching sequences of different lengths.`,name:"pad_token"},{anchor:"transformers.DebertaV2Tokenizer.cls_token",description:`<strong>cls_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;[CLS]&quot;</code>) &#x2014;
The classifier token which is used when doing sequence classification (classification of the whole sequence
instead of per-token classification). It is the first token of the sequence when built with special tokens.`,name:"cls_token"},{anchor:"transformers.DebertaV2Tokenizer.mask_token",description:`<strong>mask_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;[MASK]&quot;</code>) &#x2014;
The token used for masking values. This is the token used when training this model with masked language
modeling. This is the token which the model will try to predict.`,name:"mask_token"},{anchor:"transformers.DebertaV2Tokenizer.sp_model_kwargs",description:`<strong>sp_model_kwargs</strong> (<code>dict</code>, <em>optional</em>) &#x2014;
Will be passed to the <code>SentencePieceProcessor.__init__()</code> method. The <a href="https://github.com/google/sentencepiece/tree/master/python" rel="nofollow">Python wrapper for
SentencePiece</a> can be used, among other things,
to set:</p>
<ul>
<li>
<p><code>enable_sampling</code>: Enable subword regularization.</p>
</li>
<li>
<p><code>nbest_size</code>: Sampling parameters for unigram. Invalid for BPE-Dropout.</p>
<ul>
<li><code>nbest_size = {0,1}</code>: No sampling is performed.</li>
<li><code>nbest_size &gt; 1</code>: samples from the nbest_size results.</li>
<li><code>nbest_size &lt; 0</code>: assuming that nbest_size is infinite and samples from the all hypothesis (lattice)
using forward-filtering-and-backward-sampling algorithm.</li>
</ul>
</li>
<li>
<p><code>alpha</code>: Smoothing parameter for unigram sampling, and dropout probability of merge operations for
BPE-dropout.</p>
</li>
</ul>`,name:"sp_model_kwargs"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deberta_v2/tokenization_deberta_v2.py#L35"}}),Ye=new J({props:{name:"build_inputs_with_special_tokens",anchor:"transformers.DebertaV2Tokenizer.build_inputs_with_special_tokens",parameters:[{name:"token_ids_0",val:""},{name:"token_ids_1",val:" = None"}],parametersDescription:[{anchor:"transformers.DebertaV2Tokenizer.build_inputs_with_special_tokens.token_ids_0",description:`<strong>token_ids_0</strong> (<code>List[int]</code>) &#x2014;
List of IDs to which the special tokens will be added.`,name:"token_ids_0"},{anchor:"transformers.DebertaV2Tokenizer.build_inputs_with_special_tokens.token_ids_1",description:`<strong>token_ids_1</strong> (<code>List[int]</code>, <em>optional</em>) &#x2014;
Optional second list of IDs for sequence pairs.`,name:"token_ids_1"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deberta_v2/tokenization_deberta_v2.py#L161",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>List of <a href="../glossary#input-ids">input IDs</a> with the appropriate special tokens.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>List[int]</code></p>
`}}),Pe=new J({props:{name:"get_special_tokens_mask",anchor:"transformers.DebertaV2Tokenizer.get_special_tokens_mask",parameters:[{name:"token_ids_0",val:""},{name:"token_ids_1",val:" = None"},{name:"already_has_special_tokens",val:" = False"}],parametersDescription:[{anchor:"transformers.DebertaV2Tokenizer.get_special_tokens_mask.token_ids_0",description:`<strong>token_ids_0</strong> (<code>List[int]</code>) &#x2014;
List of IDs.`,name:"token_ids_0"},{anchor:"transformers.DebertaV2Tokenizer.get_special_tokens_mask.token_ids_1",description:`<strong>token_ids_1</strong> (<code>List[int]</code>, <em>optional</em>) &#x2014;
Optional second list of IDs for sequence pairs.`,name:"token_ids_1"},{anchor:"transformers.DebertaV2Tokenizer.get_special_tokens_mask.already_has_special_tokens",description:`<strong>already_has_special_tokens</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not the token list is already formatted with special tokens for the model.`,name:"already_has_special_tokens"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deberta_v2/tokenization_deberta_v2.py#L185",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>List[int]</code></p>
`}}),Ae=new J({props:{name:"create_token_type_ids_from_sequences",anchor:"transformers.DebertaV2Tokenizer.create_token_type_ids_from_sequences",parameters:[{name:"token_ids_0",val:": list"},{name:"token_ids_1",val:": typing.Optional[list[int]] = None"}],parametersDescription:[{anchor:"transformers.DebertaV2Tokenizer.create_token_type_ids_from_sequences.token_ids_0",description:"<strong>token_ids_0</strong> (<code>list[int]</code>) &#x2014; The first tokenized sequence.",name:"token_ids_0"},{anchor:"transformers.DebertaV2Tokenizer.create_token_type_ids_from_sequences.token_ids_1",description:"<strong>token_ids_1</strong> (<code>list[int]</code>, <em>optional</em>) &#x2014; The second tokenized sequence.",name:"token_ids_1"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L3432",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>The token type ids.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>list[int]</code></p>
`}}),Oe=new J({props:{name:"save_vocabulary",anchor:"transformers.DebertaV2Tokenizer.save_vocabulary",parameters:[{name:"save_directory",val:": str"},{name:"filename_prefix",val:": typing.Optional[str] = None"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deberta_v2/tokenization_deberta_v2.py#L217"}}),Ke=new ee({props:{title:"DebertaV2TokenizerFast",local:"transformers.DebertaV2TokenizerFast",headingTag:"h2"}}),et=new J({props:{name:"class transformers.DebertaV2TokenizerFast",anchor:"transformers.DebertaV2TokenizerFast",parameters:[{name:"vocab_file",val:" = None"},{name:"tokenizer_file",val:" = None"},{name:"do_lower_case",val:" = False"},{name:"split_by_punct",val:" = False"},{name:"bos_token",val:" = '[CLS]'"},{name:"eos_token",val:" = '[SEP]'"},{name:"unk_token",val:" = '[UNK]'"},{name:"sep_token",val:" = '[SEP]'"},{name:"pad_token",val:" = '[PAD]'"},{name:"cls_token",val:" = '[CLS]'"},{name:"mask_token",val:" = '[MASK]'"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.DebertaV2TokenizerFast.vocab_file",description:`<strong>vocab_file</strong> (<code>str</code>) &#x2014;
<a href="https://github.com/google/sentencepiece" rel="nofollow">SentencePiece</a> file (generally has a <em>.spm</em> extension) that
contains the vocabulary necessary to instantiate a tokenizer.`,name:"vocab_file"},{anchor:"transformers.DebertaV2TokenizerFast.do_lower_case",description:`<strong>do_lower_case</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to lowercase the input when tokenizing.`,name:"do_lower_case"},{anchor:"transformers.DebertaV2TokenizerFast.bos_token",description:`<strong>bos_token</strong> (<code>string</code>, <em>optional</em>, defaults to <code>&quot;[CLS]&quot;</code>) &#x2014;
The beginning of sequence token that was used during pre-training. Can be used a sequence classifier token.
When building a sequence using special tokens, this is not the token that is used for the beginning of
sequence. The token used is the <code>cls_token</code>.`,name:"bos_token"},{anchor:"transformers.DebertaV2TokenizerFast.eos_token",description:`<strong>eos_token</strong> (<code>string</code>, <em>optional</em>, defaults to <code>&quot;[SEP]&quot;</code>) &#x2014;
The end of sequence token. When building a sequence using special tokens, this is not the token that is
used for the end of sequence. The token used is the <code>sep_token</code>.`,name:"eos_token"},{anchor:"transformers.DebertaV2TokenizerFast.unk_token",description:`<strong>unk_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;[UNK]&quot;</code>) &#x2014;
The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
token instead.`,name:"unk_token"},{anchor:"transformers.DebertaV2TokenizerFast.sep_token",description:`<strong>sep_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;[SEP]&quot;</code>) &#x2014;
The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
sequence classification or for a text and a question for question answering. It is also used as the last
token of a sequence built with special tokens.`,name:"sep_token"},{anchor:"transformers.DebertaV2TokenizerFast.pad_token",description:`<strong>pad_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;[PAD]&quot;</code>) &#x2014;
The token used for padding, for example when batching sequences of different lengths.`,name:"pad_token"},{anchor:"transformers.DebertaV2TokenizerFast.cls_token",description:`<strong>cls_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;[CLS]&quot;</code>) &#x2014;
The classifier token which is used when doing sequence classification (classification of the whole sequence
instead of per-token classification). It is the first token of the sequence when built with special tokens.`,name:"cls_token"},{anchor:"transformers.DebertaV2TokenizerFast.mask_token",description:`<strong>mask_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;[MASK]&quot;</code>) &#x2014;
The token used for masking values. This is the token used when training this model with masked language
modeling. This is the token which the model will try to predict.`,name:"mask_token"},{anchor:"transformers.DebertaV2TokenizerFast.sp_model_kwargs",description:`<strong>sp_model_kwargs</strong> (<code>dict</code>, <em>optional</em>) &#x2014;
Will be passed to the <code>SentencePieceProcessor.__init__()</code> method. The <a href="https://github.com/google/sentencepiece/tree/master/python" rel="nofollow">Python wrapper for
SentencePiece</a> can be used, among other things,
to set:</p>
<ul>
<li>
<p><code>enable_sampling</code>: Enable subword regularization.</p>
</li>
<li>
<p><code>nbest_size</code>: Sampling parameters for unigram. Invalid for BPE-Dropout.</p>
<ul>
<li><code>nbest_size = {0,1}</code>: No sampling is performed.</li>
<li><code>nbest_size &gt; 1</code>: samples from the nbest_size results.</li>
<li><code>nbest_size &lt; 0</code>: assuming that nbest_size is infinite and samples from the all hypothesis (lattice)
using forward-filtering-and-backward-sampling algorithm.</li>
</ul>
</li>
<li>
<p><code>alpha</code>: Smoothing parameter for unigram sampling, and dropout probability of merge operations for
BPE-dropout.</p>
</li>
</ul>`,name:"sp_model_kwargs"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deberta_v2/tokenization_deberta_v2_fast.py#L36"}}),tt=new J({props:{name:"build_inputs_with_special_tokens",anchor:"transformers.DebertaV2TokenizerFast.build_inputs_with_special_tokens",parameters:[{name:"token_ids_0",val:""},{name:"token_ids_1",val:" = None"}],parametersDescription:[{anchor:"transformers.DebertaV2TokenizerFast.build_inputs_with_special_tokens.token_ids_0",description:`<strong>token_ids_0</strong> (<code>List[int]</code>) &#x2014;
List of IDs to which the special tokens will be added.`,name:"token_ids_0"},{anchor:"transformers.DebertaV2TokenizerFast.build_inputs_with_special_tokens.token_ids_1",description:`<strong>token_ids_1</strong> (<code>List[int]</code>, <em>optional</em>) &#x2014;
Optional second list of IDs for sequence pairs.`,name:"token_ids_1"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deberta_v2/tokenization_deberta_v2_fast.py#L122",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>List of <a href="../glossary#input-ids">input IDs</a> with the appropriate special tokens.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>List[int]</code></p>
`}}),nt=new J({props:{name:"create_token_type_ids_from_sequences",anchor:"transformers.DebertaV2TokenizerFast.create_token_type_ids_from_sequences",parameters:[{name:"token_ids_0",val:": list"},{name:"token_ids_1",val:": typing.Optional[list[int]] = None"}],parametersDescription:[{anchor:"transformers.DebertaV2TokenizerFast.create_token_type_ids_from_sequences.token_ids_0",description:"<strong>token_ids_0</strong> (<code>list[int]</code>) &#x2014; The first tokenized sequence.",name:"token_ids_0"},{anchor:"transformers.DebertaV2TokenizerFast.create_token_type_ids_from_sequences.token_ids_1",description:"<strong>token_ids_1</strong> (<code>list[int]</code>, <em>optional</em>) &#x2014; The second tokenized sequence.",name:"token_ids_1"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L3432",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>The token type ids.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>list[int]</code></p>
`}}),ot=new ee({props:{title:"DebertaV2Model",local:"transformers.DebertaV2Model",headingTag:"h2"}}),st=new J({props:{name:"class transformers.DebertaV2Model",anchor:"transformers.DebertaV2Model",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.DebertaV2Model.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/deberta-v2#transformers.DebertaV2Model">DebertaV2Model</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deberta_v2/modeling_deberta_v2.py#L719"}}),at=new J({props:{name:"forward",anchor:"transformers.DebertaV2Model.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.DebertaV2Model.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.DebertaV2Model.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.DebertaV2Model.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.DebertaV2Model.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.DebertaV2Model.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.DebertaV2Model.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.DebertaV2Model.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.DebertaV2Model.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deberta_v2/modeling_deberta_v2.py#L743",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutput"
>transformers.modeling_outputs.BaseModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/deberta-v2#transformers.DebertaV2Config"
>DebertaV2Config</a>) and inputs.</p>
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
`}}),Te=new qe({props:{$$slots:{default:[_a]},$$scope:{ctx:w}}}),rt=new ee({props:{title:"DebertaV2PreTrainedModel",local:"transformers.DebertaV2PreTrainedModel",headingTag:"h2"}}),it=new J({props:{name:"class transformers.DebertaV2PreTrainedModel",anchor:"transformers.DebertaV2PreTrainedModel",parameters:[{name:"config",val:": PretrainedConfig"},{name:"*inputs",val:""},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.DebertaV2PreTrainedModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deberta_v2/modeling_deberta_v2.py#L692"}}),lt=new J({props:{name:"_forward_unimplemented",anchor:"transformers.DebertaV2PreTrainedModel.forward",parameters:[{name:"*input",val:": typing.Any"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/torch/nn/modules/module.py#L388"}}),ke=new qe({props:{$$slots:{default:[ya]},$$scope:{ctx:w}}}),dt=new ee({props:{title:"DebertaV2ForMaskedLM",local:"transformers.DebertaV2ForMaskedLM",headingTag:"h2"}}),ct=new J({props:{name:"class transformers.DebertaV2ForMaskedLM",anchor:"transformers.DebertaV2ForMaskedLM",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.DebertaV2ForMaskedLM.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/deberta-v2#transformers.DebertaV2ForMaskedLM">DebertaV2ForMaskedLM</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deberta_v2/modeling_deberta_v2.py#L916"}}),pt=new J({props:{name:"forward",anchor:"transformers.DebertaV2ForMaskedLM.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"labels",val:": typing.Optional[torch.Tensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.DebertaV2ForMaskedLM.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.DebertaV2ForMaskedLM.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.DebertaV2ForMaskedLM.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.DebertaV2ForMaskedLM.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.DebertaV2ForMaskedLM.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.DebertaV2ForMaskedLM.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should be in <code>[-100, 0, ..., config.vocab_size]</code> (see <code>input_ids</code> docstring) Tokens with indices set to <code>-100</code> are ignored (masked), the
loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>`,name:"labels"},{anchor:"transformers.DebertaV2ForMaskedLM.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.DebertaV2ForMaskedLM.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.DebertaV2ForMaskedLM.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deberta_v2/modeling_deberta_v2.py#L946",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.MaskedLMOutput"
>transformers.modeling_outputs.MaskedLMOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/deberta-v2#transformers.DebertaV2Config"
>DebertaV2Config</a>) and inputs.</p>
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
`}}),we=new qe({props:{$$slots:{default:[Ma]},$$scope:{ctx:w}}}),ve=new Jt({props:{anchor:"transformers.DebertaV2ForMaskedLM.forward.example",$$slots:{default:[Ta]},$$scope:{ctx:w}}}),mt=new ee({props:{title:"DebertaV2ForSequenceClassification",local:"transformers.DebertaV2ForSequenceClassification",headingTag:"h2"}}),ut=new J({props:{name:"class transformers.DebertaV2ForSequenceClassification",anchor:"transformers.DebertaV2ForSequenceClassification",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.DebertaV2ForSequenceClassification.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/deberta-v2#transformers.DebertaV2ForSequenceClassification">DebertaV2ForSequenceClassification</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deberta_v2/modeling_deberta_v2.py#L1032"}}),ht=new J({props:{name:"forward",anchor:"transformers.DebertaV2ForSequenceClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"labels",val:": typing.Optional[torch.Tensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.DebertaV2ForSequenceClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.DebertaV2ForSequenceClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.DebertaV2ForSequenceClassification.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.DebertaV2ForSequenceClassification.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.DebertaV2ForSequenceClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.DebertaV2ForSequenceClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the sequence classification/regression loss. Indices should be in <code>[0, ..., config.num_labels - 1]</code>. If <code>config.num_labels == 1</code> a regression loss is computed (Mean-Square loss), If
<code>config.num_labels &gt; 1</code> a classification loss is computed (Cross-Entropy).`,name:"labels"},{anchor:"transformers.DebertaV2ForSequenceClassification.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.DebertaV2ForSequenceClassification.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.DebertaV2ForSequenceClassification.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deberta_v2/modeling_deberta_v2.py#L1057",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput"
>transformers.modeling_outputs.SequenceClassifierOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/deberta-v2#transformers.DebertaV2Config"
>DebertaV2Config</a>) and inputs.</p>
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
`}}),$e=new qe({props:{$$slots:{default:[ka]},$$scope:{ctx:w}}}),Je=new Jt({props:{anchor:"transformers.DebertaV2ForSequenceClassification.forward.example",$$slots:{default:[wa]},$$scope:{ctx:w}}}),je=new Jt({props:{anchor:"transformers.DebertaV2ForSequenceClassification.forward.example-2",$$slots:{default:[va]},$$scope:{ctx:w}}}),ft=new ee({props:{title:"DebertaV2ForTokenClassification",local:"transformers.DebertaV2ForTokenClassification",headingTag:"h2"}}),gt=new J({props:{name:"class transformers.DebertaV2ForTokenClassification",anchor:"transformers.DebertaV2ForTokenClassification",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.DebertaV2ForTokenClassification.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/deberta-v2#transformers.DebertaV2ForTokenClassification">DebertaV2ForTokenClassification</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deberta_v2/modeling_deberta_v2.py#L1141"}}),bt=new J({props:{name:"forward",anchor:"transformers.DebertaV2ForTokenClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"labels",val:": typing.Optional[torch.Tensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.DebertaV2ForTokenClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.DebertaV2ForTokenClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.DebertaV2ForTokenClassification.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.DebertaV2ForTokenClassification.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.DebertaV2ForTokenClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.DebertaV2ForTokenClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the token classification loss. Indices should be in <code>[0, ..., config.num_labels - 1]</code>.`,name:"labels"},{anchor:"transformers.DebertaV2ForTokenClassification.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.DebertaV2ForTokenClassification.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.DebertaV2ForTokenClassification.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deberta_v2/modeling_deberta_v2.py#L1153",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput"
>transformers.modeling_outputs.TokenClassifierOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/deberta-v2#transformers.DebertaV2Config"
>DebertaV2Config</a>) and inputs.</p>
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
`}}),Ve=new qe({props:{$$slots:{default:[$a]},$$scope:{ctx:w}}}),Ce=new Jt({props:{anchor:"transformers.DebertaV2ForTokenClassification.forward.example",$$slots:{default:[Ja]},$$scope:{ctx:w}}}),_t=new ee({props:{title:"DebertaV2ForQuestionAnswering",local:"transformers.DebertaV2ForQuestionAnswering",headingTag:"h2"}}),yt=new J({props:{name:"class transformers.DebertaV2ForQuestionAnswering",anchor:"transformers.DebertaV2ForQuestionAnswering",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.DebertaV2ForQuestionAnswering.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/deberta-v2#transformers.DebertaV2ForQuestionAnswering">DebertaV2ForQuestionAnswering</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deberta_v2/modeling_deberta_v2.py#L1203"}}),Mt=new J({props:{name:"forward",anchor:"transformers.DebertaV2ForQuestionAnswering.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"start_positions",val:": typing.Optional[torch.Tensor] = None"},{name:"end_positions",val:": typing.Optional[torch.Tensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.DebertaV2ForQuestionAnswering.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.DebertaV2ForQuestionAnswering.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.DebertaV2ForQuestionAnswering.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.DebertaV2ForQuestionAnswering.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.DebertaV2ForQuestionAnswering.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.DebertaV2ForQuestionAnswering.forward.start_positions",description:`<strong>start_positions</strong> (<code>torch.Tensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for position (index) of the start of the labelled span for computing the token classification loss.
Positions are clamped to the length of the sequence (<code>sequence_length</code>). Position outside of the sequence
are not taken into account for computing the loss.`,name:"start_positions"},{anchor:"transformers.DebertaV2ForQuestionAnswering.forward.end_positions",description:`<strong>end_positions</strong> (<code>torch.Tensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for position (index) of the end of the labelled span for computing the token classification loss.
Positions are clamped to the length of the sequence (<code>sequence_length</code>). Position outside of the sequence
are not taken into account for computing the loss.`,name:"end_positions"},{anchor:"transformers.DebertaV2ForQuestionAnswering.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.DebertaV2ForQuestionAnswering.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.DebertaV2ForQuestionAnswering.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deberta_v2/modeling_deberta_v2.py#L1214",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.QuestionAnsweringModelOutput"
>transformers.modeling_outputs.QuestionAnsweringModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/deberta-v2#transformers.DebertaV2Config"
>DebertaV2Config</a>) and inputs.</p>
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
`}}),xe=new qe({props:{$$slots:{default:[ja]},$$scope:{ctx:w}}}),Ue=new Jt({props:{anchor:"transformers.DebertaV2ForQuestionAnswering.forward.example",$$slots:{default:[Va]},$$scope:{ctx:w}}}),Tt=new ee({props:{title:"DebertaV2ForMultipleChoice",local:"transformers.DebertaV2ForMultipleChoice",headingTag:"h2"}}),kt=new J({props:{name:"class transformers.DebertaV2ForMultipleChoice",anchor:"transformers.DebertaV2ForMultipleChoice",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.DebertaV2ForMultipleChoice.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/deberta-v2#transformers.DebertaV2ForMultipleChoice">DebertaV2ForMultipleChoice</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deberta_v2/modeling_deberta_v2.py#L1280"}}),wt=new J({props:{name:"forward",anchor:"transformers.DebertaV2ForMultipleChoice.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"labels",val:": typing.Optional[torch.Tensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.DebertaV2ForMultipleChoice.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.DebertaV2ForMultipleChoice.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.DebertaV2ForMultipleChoice.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.DebertaV2ForMultipleChoice.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.DebertaV2ForMultipleChoice.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.DebertaV2ForMultipleChoice.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the multiple choice classification loss. Indices should be in <code>[0, ..., num_choices-1]</code> where <code>num_choices</code> is the size of the second dimension of the input tensors. (See
<code>input_ids</code> above)`,name:"labels"},{anchor:"transformers.DebertaV2ForMultipleChoice.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.DebertaV2ForMultipleChoice.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.DebertaV2ForMultipleChoice.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deberta_v2/modeling_deberta_v2.py#L1304",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.MultipleChoiceModelOutput"
>transformers.modeling_outputs.MultipleChoiceModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/deberta-v2#transformers.DebertaV2Config"
>DebertaV2Config</a>) and inputs.</p>
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
`}}),ze=new qe({props:{$$slots:{default:[Ca]},$$scope:{ctx:w}}}),Ze=new Jt({props:{anchor:"transformers.DebertaV2ForMultipleChoice.forward.example",$$slots:{default:[xa]},$$scope:{ctx:w}}}),vt=new ca({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/deberta-v2.md"}}),{c(){t=c("meta"),u=a(),n=c("p"),d=a(),k=c("p"),k.innerHTML=s,h=a(),j=c("div"),j.innerHTML=gn,Ie=a(),f(ne.$$.fragment),yn=a(),Be=c("p"),Be.innerHTML=ms,Mn=a(),Re=c("p"),Re.innerHTML=us,Tn=a(),f(be.$$.fragment),kn=a(),Ge=c("p"),Ge.innerHTML=hs,wn=a(),f(_e.$$.fragment),vn=a(),Ne=c("p"),Ne.innerHTML=fs,$n=a(),Xe=c("p"),Xe.innerHTML=gs,Jn=a(),f(Se.$$.fragment),jn=a(),f(Qe.$$.fragment),Vn=a(),I=c("div"),f(He.$$.fragment),Pn=a(),jt=c("p"),jt.innerHTML=bs,An=a(),Vt=c("p"),Vt.innerHTML=_s,On=a(),f(ye.$$.fragment),Cn=a(),f(Ee.$$.fragment),xn=a(),C=c("div"),f(Le.$$.fragment),Kn=a(),Ct=c("p"),Ct.innerHTML=ys,eo=a(),oe=c("div"),f(Ye.$$.fragment),to=a(),xt=c("p"),xt.textContent=Ms,no=a(),Ut=c("ul"),Ut.innerHTML=Ts,oo=a(),Me=c("div"),f(Pe.$$.fragment),so=a(),zt=c("p"),zt.innerHTML=ks,ao=a(),se=c("div"),f(Ae.$$.fragment),ro=a(),Zt=c("p"),Zt.innerHTML=ws,io=a(),Wt=c("p"),Wt.textContent=vs,lo=a(),Dt=c("div"),f(Oe.$$.fragment),Un=a(),f(Ke.$$.fragment),zn=a(),B=c("div"),f(et.$$.fragment),co=a(),Ft=c("p"),Ft.innerHTML=$s,po=a(),ae=c("div"),f(tt.$$.fragment),mo=a(),qt=c("p"),qt.textContent=Js,uo=a(),It=c("ul"),It.innerHTML=js,ho=a(),re=c("div"),f(nt.$$.fragment),fo=a(),Bt=c("p"),Bt.innerHTML=Vs,go=a(),Rt=c("p"),Rt.textContent=Cs,Zn=a(),f(ot.$$.fragment),Wn=a(),x=c("div"),f(st.$$.fragment),bo=a(),Gt=c("p"),Gt.textContent=xs,_o=a(),Nt=c("p"),Nt.innerHTML=Us,yo=a(),Xt=c("p"),Xt.innerHTML=zs,Mo=a(),ie=c("div"),f(at.$$.fragment),To=a(),St=c("p"),St.innerHTML=Zs,ko=a(),f(Te.$$.fragment),Dn=a(),f(rt.$$.fragment),Fn=a(),R=c("div"),f(it.$$.fragment),wo=a(),Qt=c("p"),Qt.innerHTML=Ws,vo=a(),Ht=c("p"),Ht.innerHTML=Ds,$o=a(),S=c("div"),f(lt.$$.fragment),Jo=a(),Et=c("p"),Et.textContent=Fs,jo=a(),Lt=c("p"),Lt.textContent=qs,Vo=a(),f(ke.$$.fragment),qn=a(),f(dt.$$.fragment),In=a(),U=c("div"),f(ct.$$.fragment),Co=a(),Yt=c("p"),Yt.innerHTML=Is,xo=a(),Pt=c("p"),Pt.innerHTML=Bs,Uo=a(),At=c("p"),At.innerHTML=Rs,zo=a(),Q=c("div"),f(pt.$$.fragment),Zo=a(),Ot=c("p"),Ot.innerHTML=Gs,Wo=a(),f(we.$$.fragment),Do=a(),f(ve.$$.fragment),Bn=a(),f(mt.$$.fragment),Rn=a(),z=c("div"),f(ut.$$.fragment),Fo=a(),Kt=c("p"),Kt.textContent=Ns,qo=a(),en=c("p"),en.innerHTML=Xs,Io=a(),tn=c("p"),tn.innerHTML=Ss,Bo=a(),q=c("div"),f(ht.$$.fragment),Ro=a(),nn=c("p"),nn.innerHTML=Qs,Go=a(),f($e.$$.fragment),No=a(),f(Je.$$.fragment),Xo=a(),f(je.$$.fragment),Gn=a(),f(ft.$$.fragment),Nn=a(),Z=c("div"),f(gt.$$.fragment),So=a(),on=c("p"),on.textContent=Hs,Qo=a(),sn=c("p"),sn.innerHTML=Es,Ho=a(),an=c("p"),an.innerHTML=Ls,Eo=a(),H=c("div"),f(bt.$$.fragment),Lo=a(),rn=c("p"),rn.innerHTML=Ys,Yo=a(),f(Ve.$$.fragment),Po=a(),f(Ce.$$.fragment),Xn=a(),f(_t.$$.fragment),Sn=a(),W=c("div"),f(yt.$$.fragment),Ao=a(),ln=c("p"),ln.innerHTML=Ps,Oo=a(),dn=c("p"),dn.innerHTML=As,Ko=a(),cn=c("p"),cn.innerHTML=Os,es=a(),E=c("div"),f(Mt.$$.fragment),ts=a(),pn=c("p"),pn.innerHTML=Ks,ns=a(),f(xe.$$.fragment),os=a(),f(Ue.$$.fragment),Qn=a(),f(Tt.$$.fragment),Hn=a(),D=c("div"),f(kt.$$.fragment),ss=a(),mn=c("p"),mn.textContent=ea,as=a(),un=c("p"),un.innerHTML=ta,rs=a(),hn=c("p"),hn.innerHTML=na,is=a(),L=c("div"),f(wt.$$.fragment),ls=a(),fn=c("p"),fn.innerHTML=oa,ds=a(),f(ze.$$.fragment),cs=a(),f(Ze.$$.fragment),En=a(),f(vt.$$.fragment),Ln=a(),bn=c("p"),this.h()},l(e){const l=la("svelte-u9bgzb",document.head);t=p(l,"META",{name:!0,content:!0}),l.forEach(i),u=r(e),n=p(e,"P",{}),v(n).forEach(i),d=r(e),k=p(e,"P",{"data-svelte-h":!0}),T(k)!=="svelte-12ssjpu"&&(k.innerHTML=s),h=r(e),j=p(e,"DIV",{style:!0,"data-svelte-h":!0}),T(j)!=="svelte-383xsf"&&(j.innerHTML=gn),Ie=r(e),g(ne.$$.fragment,e),yn=r(e),Be=p(e,"P",{"data-svelte-h":!0}),T(Be)!=="svelte-1kzilp1"&&(Be.innerHTML=ms),Mn=r(e),Re=p(e,"P",{"data-svelte-h":!0}),T(Re)!=="svelte-17ejs6w"&&(Re.innerHTML=us),Tn=r(e),g(be.$$.fragment,e),kn=r(e),Ge=p(e,"P",{"data-svelte-h":!0}),T(Ge)!=="svelte-fba0sb"&&(Ge.innerHTML=hs),wn=r(e),g(_e.$$.fragment,e),vn=r(e),Ne=p(e,"P",{"data-svelte-h":!0}),T(Ne)!=="svelte-nf5ooi"&&(Ne.innerHTML=fs),$n=r(e),Xe=p(e,"P",{"data-svelte-h":!0}),T(Xe)!=="svelte-1y34bam"&&(Xe.innerHTML=gs),Jn=r(e),g(Se.$$.fragment,e),jn=r(e),g(Qe.$$.fragment,e),Vn=r(e),I=p(e,"DIV",{class:!0});var Y=v(I);g(He.$$.fragment,Y),Pn=r(Y),jt=p(Y,"P",{"data-svelte-h":!0}),T(jt)!=="svelte-17imqi6"&&(jt.innerHTML=bs),An=r(Y),Vt=p(Y,"P",{"data-svelte-h":!0}),T(Vt)!=="svelte-1ek1ss9"&&(Vt.innerHTML=_s),On=r(Y),g(ye.$$.fragment,Y),Y.forEach(i),Cn=r(e),g(Ee.$$.fragment,e),xn=r(e),C=p(e,"DIV",{class:!0});var F=v(C);g(Le.$$.fragment,F),Kn=r(F),Ct=p(F,"P",{"data-svelte-h":!0}),T(Ct)!=="svelte-1tn4ph1"&&(Ct.innerHTML=ys),eo=r(F),oe=p(F,"DIV",{class:!0});var me=v(oe);g(Ye.$$.fragment,me),to=r(me),xt=p(me,"P",{"data-svelte-h":!0}),T(xt)!=="svelte-vlm5xk"&&(xt.textContent=Ms),no=r(me),Ut=p(me,"UL",{"data-svelte-h":!0}),T(Ut)!=="svelte-1196rcj"&&(Ut.innerHTML=Ts),me.forEach(i),oo=r(F),Me=p(F,"DIV",{class:!0});var $t=v(Me);g(Pe.$$.fragment,$t),so=r($t),zt=p($t,"P",{"data-svelte-h":!0}),T(zt)!=="svelte-1wmjg8a"&&(zt.innerHTML=ks),$t.forEach(i),ao=r(F),se=p(F,"DIV",{class:!0});var ue=v(se);g(Ae.$$.fragment,ue),ro=r(ue),Zt=p(ue,"P",{"data-svelte-h":!0}),T(Zt)!=="svelte-zj1vf1"&&(Zt.innerHTML=ws),io=r(ue),Wt=p(ue,"P",{"data-svelte-h":!0}),T(Wt)!=="svelte-9vptpw"&&(Wt.textContent=vs),ue.forEach(i),lo=r(F),Dt=p(F,"DIV",{class:!0});var _n=v(Dt);g(Oe.$$.fragment,_n),_n.forEach(i),F.forEach(i),Un=r(e),g(Ke.$$.fragment,e),zn=r(e),B=p(e,"DIV",{class:!0});var P=v(B);g(et.$$.fragment,P),co=r(P),Ft=p(P,"P",{"data-svelte-h":!0}),T(Ft)!=="svelte-m60pk3"&&(Ft.innerHTML=$s),po=r(P),ae=p(P,"DIV",{class:!0});var he=v(ae);g(tt.$$.fragment,he),mo=r(he),qt=p(he,"P",{"data-svelte-h":!0}),T(qt)!=="svelte-vlm5xk"&&(qt.textContent=Js),uo=r(he),It=p(he,"UL",{"data-svelte-h":!0}),T(It)!=="svelte-1196rcj"&&(It.innerHTML=js),he.forEach(i),ho=r(P),re=p(P,"DIV",{class:!0});var fe=v(re);g(nt.$$.fragment,fe),fo=r(fe),Bt=p(fe,"P",{"data-svelte-h":!0}),T(Bt)!=="svelte-zj1vf1"&&(Bt.innerHTML=Vs),go=r(fe),Rt=p(fe,"P",{"data-svelte-h":!0}),T(Rt)!=="svelte-9vptpw"&&(Rt.textContent=Cs),fe.forEach(i),P.forEach(i),Zn=r(e),g(ot.$$.fragment,e),Wn=r(e),x=p(e,"DIV",{class:!0});var G=v(x);g(st.$$.fragment,G),bo=r(G),Gt=p(G,"P",{"data-svelte-h":!0}),T(Gt)!=="svelte-r7vcad"&&(Gt.textContent=xs),_o=r(G),Nt=p(G,"P",{"data-svelte-h":!0}),T(Nt)!=="svelte-q52n56"&&(Nt.innerHTML=Us),yo=r(G),Xt=p(G,"P",{"data-svelte-h":!0}),T(Xt)!=="svelte-hswkmf"&&(Xt.innerHTML=zs),Mo=r(G),ie=p(G,"DIV",{class:!0});var ge=v(ie);g(at.$$.fragment,ge),To=r(ge),St=p(ge,"P",{"data-svelte-h":!0}),T(St)!=="svelte-de0flp"&&(St.innerHTML=Zs),ko=r(ge),g(Te.$$.fragment,ge),ge.forEach(i),G.forEach(i),Dn=r(e),g(rt.$$.fragment,e),Fn=r(e),R=p(e,"DIV",{class:!0});var A=v(R);g(it.$$.fragment,A),wo=r(A),Qt=p(A,"P",{"data-svelte-h":!0}),T(Qt)!=="svelte-q52n56"&&(Qt.innerHTML=Ws),vo=r(A),Ht=p(A,"P",{"data-svelte-h":!0}),T(Ht)!=="svelte-hswkmf"&&(Ht.innerHTML=Ds),$o=r(A),S=p(A,"DIV",{class:!0});var O=v(S);g(lt.$$.fragment,O),Jo=r(O),Et=p(O,"P",{"data-svelte-h":!0}),T(Et)!=="svelte-1q5ym45"&&(Et.textContent=Fs),jo=r(O),Lt=p(O,"P",{"data-svelte-h":!0}),T(Lt)!=="svelte-w8wo9i"&&(Lt.textContent=qs),Vo=r(O),g(ke.$$.fragment,O),O.forEach(i),A.forEach(i),qn=r(e),g(dt.$$.fragment,e),In=r(e),U=p(e,"DIV",{class:!0});var N=v(U);g(ct.$$.fragment,N),Co=r(N),Yt=p(N,"P",{"data-svelte-h":!0}),T(Yt)!=="svelte-gof807"&&(Yt.innerHTML=Is),xo=r(N),Pt=p(N,"P",{"data-svelte-h":!0}),T(Pt)!=="svelte-q52n56"&&(Pt.innerHTML=Bs),Uo=r(N),At=p(N,"P",{"data-svelte-h":!0}),T(At)!=="svelte-hswkmf"&&(At.innerHTML=Rs),zo=r(N),Q=p(N,"DIV",{class:!0});var K=v(Q);g(pt.$$.fragment,K),Zo=r(K),Ot=p(K,"P",{"data-svelte-h":!0}),T(Ot)!=="svelte-15xt1kd"&&(Ot.innerHTML=Gs),Wo=r(K),g(we.$$.fragment,K),Do=r(K),g(ve.$$.fragment,K),K.forEach(i),N.forEach(i),Bn=r(e),g(mt.$$.fragment,e),Rn=r(e),z=p(e,"DIV",{class:!0});var X=v(z);g(ut.$$.fragment,X),Fo=r(X),Kt=p(X,"P",{"data-svelte-h":!0}),T(Kt)!=="svelte-1xhb56d"&&(Kt.textContent=Ns),qo=r(X),en=p(X,"P",{"data-svelte-h":!0}),T(en)!=="svelte-q52n56"&&(en.innerHTML=Xs),Io=r(X),tn=p(X,"P",{"data-svelte-h":!0}),T(tn)!=="svelte-hswkmf"&&(tn.innerHTML=Ss),Bo=r(X),q=p(X,"DIV",{class:!0});var le=v(q);g(ht.$$.fragment,le),Ro=r(le),nn=p(le,"P",{"data-svelte-h":!0}),T(nn)!=="svelte-u6q1wj"&&(nn.innerHTML=Qs),Go=r(le),g($e.$$.fragment,le),No=r(le),g(Je.$$.fragment,le),Xo=r(le),g(je.$$.fragment,le),le.forEach(i),X.forEach(i),Gn=r(e),g(ft.$$.fragment,e),Nn=r(e),Z=p(e,"DIV",{class:!0});var de=v(Z);g(gt.$$.fragment,de),So=r(de),on=p(de,"P",{"data-svelte-h":!0}),T(on)!=="svelte-b5q1f7"&&(on.textContent=Hs),Qo=r(de),sn=p(de,"P",{"data-svelte-h":!0}),T(sn)!=="svelte-q52n56"&&(sn.innerHTML=Es),Ho=r(de),an=p(de,"P",{"data-svelte-h":!0}),T(an)!=="svelte-hswkmf"&&(an.innerHTML=Ls),Eo=r(de),H=p(de,"DIV",{class:!0});var We=v(H);g(bt.$$.fragment,We),Lo=r(We),rn=p(We,"P",{"data-svelte-h":!0}),T(rn)!=="svelte-1oqti5l"&&(rn.innerHTML=Ys),Yo=r(We),g(Ve.$$.fragment,We),Po=r(We),g(Ce.$$.fragment,We),We.forEach(i),de.forEach(i),Xn=r(e),g(_t.$$.fragment,e),Sn=r(e),W=p(e,"DIV",{class:!0});var ce=v(W);g(yt.$$.fragment,ce),Ao=r(ce),ln=p(ce,"P",{"data-svelte-h":!0}),T(ln)!=="svelte-fs9f66"&&(ln.innerHTML=Ps),Oo=r(ce),dn=p(ce,"P",{"data-svelte-h":!0}),T(dn)!=="svelte-q52n56"&&(dn.innerHTML=As),Ko=r(ce),cn=p(ce,"P",{"data-svelte-h":!0}),T(cn)!=="svelte-hswkmf"&&(cn.innerHTML=Os),es=r(ce),E=p(ce,"DIV",{class:!0});var De=v(E);g(Mt.$$.fragment,De),ts=r(De),pn=p(De,"P",{"data-svelte-h":!0}),T(pn)!=="svelte-9l57vh"&&(pn.innerHTML=Ks),ns=r(De),g(xe.$$.fragment,De),os=r(De),g(Ue.$$.fragment,De),De.forEach(i),ce.forEach(i),Qn=r(e),g(Tt.$$.fragment,e),Hn=r(e),D=p(e,"DIV",{class:!0});var pe=v(D);g(kt.$$.fragment,pe),ss=r(pe),mn=p(pe,"P",{"data-svelte-h":!0}),T(mn)!=="svelte-zbj6iy"&&(mn.textContent=ea),as=r(pe),un=p(pe,"P",{"data-svelte-h":!0}),T(un)!=="svelte-q52n56"&&(un.innerHTML=ta),rs=r(pe),hn=p(pe,"P",{"data-svelte-h":!0}),T(hn)!=="svelte-hswkmf"&&(hn.innerHTML=na),is=r(pe),L=p(pe,"DIV",{class:!0});var Fe=v(L);g(wt.$$.fragment,Fe),ls=r(Fe),fn=p(Fe,"P",{"data-svelte-h":!0}),T(fn)!=="svelte-1l2enmb"&&(fn.innerHTML=oa),ds=r(Fe),g(ze.$$.fragment,Fe),cs=r(Fe),g(Ze.$$.fragment,Fe),Fe.forEach(i),pe.forEach(i),En=r(e),g(vt.$$.fragment,e),Ln=r(e),bn=p(e,"P",{}),v(bn).forEach(i),this.h()},h(){$(t,"name","hf:doc:metadata"),$(t,"content",za),da(j,"float","right"),$(I,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(oe,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(Me,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(se,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(Dt,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(C,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(ae,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(re,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(B,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(ie,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(x,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(S,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(R,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(Q,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(U,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(q,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(z,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(H,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(Z,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(E,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(W,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(L,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(D,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,l){o(document.head,t),m(e,u,l),m(e,n,l),m(e,d,l),m(e,k,l),m(e,h,l),m(e,j,l),m(e,Ie,l),b(ne,e,l),m(e,yn,l),m(e,Be,l),m(e,Mn,l),m(e,Re,l),m(e,Tn,l),b(be,e,l),m(e,kn,l),m(e,Ge,l),m(e,wn,l),b(_e,e,l),m(e,vn,l),m(e,Ne,l),m(e,$n,l),m(e,Xe,l),m(e,Jn,l),b(Se,e,l),m(e,jn,l),b(Qe,e,l),m(e,Vn,l),m(e,I,l),b(He,I,null),o(I,Pn),o(I,jt),o(I,An),o(I,Vt),o(I,On),b(ye,I,null),m(e,Cn,l),b(Ee,e,l),m(e,xn,l),m(e,C,l),b(Le,C,null),o(C,Kn),o(C,Ct),o(C,eo),o(C,oe),b(Ye,oe,null),o(oe,to),o(oe,xt),o(oe,no),o(oe,Ut),o(C,oo),o(C,Me),b(Pe,Me,null),o(Me,so),o(Me,zt),o(C,ao),o(C,se),b(Ae,se,null),o(se,ro),o(se,Zt),o(se,io),o(se,Wt),o(C,lo),o(C,Dt),b(Oe,Dt,null),m(e,Un,l),b(Ke,e,l),m(e,zn,l),m(e,B,l),b(et,B,null),o(B,co),o(B,Ft),o(B,po),o(B,ae),b(tt,ae,null),o(ae,mo),o(ae,qt),o(ae,uo),o(ae,It),o(B,ho),o(B,re),b(nt,re,null),o(re,fo),o(re,Bt),o(re,go),o(re,Rt),m(e,Zn,l),b(ot,e,l),m(e,Wn,l),m(e,x,l),b(st,x,null),o(x,bo),o(x,Gt),o(x,_o),o(x,Nt),o(x,yo),o(x,Xt),o(x,Mo),o(x,ie),b(at,ie,null),o(ie,To),o(ie,St),o(ie,ko),b(Te,ie,null),m(e,Dn,l),b(rt,e,l),m(e,Fn,l),m(e,R,l),b(it,R,null),o(R,wo),o(R,Qt),o(R,vo),o(R,Ht),o(R,$o),o(R,S),b(lt,S,null),o(S,Jo),o(S,Et),o(S,jo),o(S,Lt),o(S,Vo),b(ke,S,null),m(e,qn,l),b(dt,e,l),m(e,In,l),m(e,U,l),b(ct,U,null),o(U,Co),o(U,Yt),o(U,xo),o(U,Pt),o(U,Uo),o(U,At),o(U,zo),o(U,Q),b(pt,Q,null),o(Q,Zo),o(Q,Ot),o(Q,Wo),b(we,Q,null),o(Q,Do),b(ve,Q,null),m(e,Bn,l),b(mt,e,l),m(e,Rn,l),m(e,z,l),b(ut,z,null),o(z,Fo),o(z,Kt),o(z,qo),o(z,en),o(z,Io),o(z,tn),o(z,Bo),o(z,q),b(ht,q,null),o(q,Ro),o(q,nn),o(q,Go),b($e,q,null),o(q,No),b(Je,q,null),o(q,Xo),b(je,q,null),m(e,Gn,l),b(ft,e,l),m(e,Nn,l),m(e,Z,l),b(gt,Z,null),o(Z,So),o(Z,on),o(Z,Qo),o(Z,sn),o(Z,Ho),o(Z,an),o(Z,Eo),o(Z,H),b(bt,H,null),o(H,Lo),o(H,rn),o(H,Yo),b(Ve,H,null),o(H,Po),b(Ce,H,null),m(e,Xn,l),b(_t,e,l),m(e,Sn,l),m(e,W,l),b(yt,W,null),o(W,Ao),o(W,ln),o(W,Oo),o(W,dn),o(W,Ko),o(W,cn),o(W,es),o(W,E),b(Mt,E,null),o(E,ts),o(E,pn),o(E,ns),b(xe,E,null),o(E,os),b(Ue,E,null),m(e,Qn,l),b(Tt,e,l),m(e,Hn,l),m(e,D,l),b(kt,D,null),o(D,ss),o(D,mn),o(D,as),o(D,un),o(D,rs),o(D,hn),o(D,is),o(D,L),b(wt,L,null),o(L,ls),o(L,fn),o(L,ds),b(ze,L,null),o(L,cs),b(Ze,L,null),m(e,En,l),b(vt,e,l),m(e,Ln,l),m(e,bn,l),Yn=!0},p(e,[l]){const Y={};l&2&&(Y.$$scope={dirty:l,ctx:e}),be.$set(Y);const F={};l&2&&(F.$$scope={dirty:l,ctx:e}),_e.$set(F);const me={};l&2&&(me.$$scope={dirty:l,ctx:e}),ye.$set(me);const $t={};l&2&&($t.$$scope={dirty:l,ctx:e}),Te.$set($t);const ue={};l&2&&(ue.$$scope={dirty:l,ctx:e}),ke.$set(ue);const _n={};l&2&&(_n.$$scope={dirty:l,ctx:e}),we.$set(_n);const P={};l&2&&(P.$$scope={dirty:l,ctx:e}),ve.$set(P);const he={};l&2&&(he.$$scope={dirty:l,ctx:e}),$e.$set(he);const fe={};l&2&&(fe.$$scope={dirty:l,ctx:e}),Je.$set(fe);const G={};l&2&&(G.$$scope={dirty:l,ctx:e}),je.$set(G);const ge={};l&2&&(ge.$$scope={dirty:l,ctx:e}),Ve.$set(ge);const A={};l&2&&(A.$$scope={dirty:l,ctx:e}),Ce.$set(A);const O={};l&2&&(O.$$scope={dirty:l,ctx:e}),xe.$set(O);const N={};l&2&&(N.$$scope={dirty:l,ctx:e}),Ue.$set(N);const K={};l&2&&(K.$$scope={dirty:l,ctx:e}),ze.$set(K);const X={};l&2&&(X.$$scope={dirty:l,ctx:e}),Ze.$set(X)},i(e){Yn||(_(ne.$$.fragment,e),_(be.$$.fragment,e),_(_e.$$.fragment,e),_(Se.$$.fragment,e),_(Qe.$$.fragment,e),_(He.$$.fragment,e),_(ye.$$.fragment,e),_(Ee.$$.fragment,e),_(Le.$$.fragment,e),_(Ye.$$.fragment,e),_(Pe.$$.fragment,e),_(Ae.$$.fragment,e),_(Oe.$$.fragment,e),_(Ke.$$.fragment,e),_(et.$$.fragment,e),_(tt.$$.fragment,e),_(nt.$$.fragment,e),_(ot.$$.fragment,e),_(st.$$.fragment,e),_(at.$$.fragment,e),_(Te.$$.fragment,e),_(rt.$$.fragment,e),_(it.$$.fragment,e),_(lt.$$.fragment,e),_(ke.$$.fragment,e),_(dt.$$.fragment,e),_(ct.$$.fragment,e),_(pt.$$.fragment,e),_(we.$$.fragment,e),_(ve.$$.fragment,e),_(mt.$$.fragment,e),_(ut.$$.fragment,e),_(ht.$$.fragment,e),_($e.$$.fragment,e),_(Je.$$.fragment,e),_(je.$$.fragment,e),_(ft.$$.fragment,e),_(gt.$$.fragment,e),_(bt.$$.fragment,e),_(Ve.$$.fragment,e),_(Ce.$$.fragment,e),_(_t.$$.fragment,e),_(yt.$$.fragment,e),_(Mt.$$.fragment,e),_(xe.$$.fragment,e),_(Ue.$$.fragment,e),_(Tt.$$.fragment,e),_(kt.$$.fragment,e),_(wt.$$.fragment,e),_(ze.$$.fragment,e),_(Ze.$$.fragment,e),_(vt.$$.fragment,e),Yn=!0)},o(e){y(ne.$$.fragment,e),y(be.$$.fragment,e),y(_e.$$.fragment,e),y(Se.$$.fragment,e),y(Qe.$$.fragment,e),y(He.$$.fragment,e),y(ye.$$.fragment,e),y(Ee.$$.fragment,e),y(Le.$$.fragment,e),y(Ye.$$.fragment,e),y(Pe.$$.fragment,e),y(Ae.$$.fragment,e),y(Oe.$$.fragment,e),y(Ke.$$.fragment,e),y(et.$$.fragment,e),y(tt.$$.fragment,e),y(nt.$$.fragment,e),y(ot.$$.fragment,e),y(st.$$.fragment,e),y(at.$$.fragment,e),y(Te.$$.fragment,e),y(rt.$$.fragment,e),y(it.$$.fragment,e),y(lt.$$.fragment,e),y(ke.$$.fragment,e),y(dt.$$.fragment,e),y(ct.$$.fragment,e),y(pt.$$.fragment,e),y(we.$$.fragment,e),y(ve.$$.fragment,e),y(mt.$$.fragment,e),y(ut.$$.fragment,e),y(ht.$$.fragment,e),y($e.$$.fragment,e),y(Je.$$.fragment,e),y(je.$$.fragment,e),y(ft.$$.fragment,e),y(gt.$$.fragment,e),y(bt.$$.fragment,e),y(Ve.$$.fragment,e),y(Ce.$$.fragment,e),y(_t.$$.fragment,e),y(yt.$$.fragment,e),y(Mt.$$.fragment,e),y(xe.$$.fragment,e),y(Ue.$$.fragment,e),y(Tt.$$.fragment,e),y(kt.$$.fragment,e),y(wt.$$.fragment,e),y(ze.$$.fragment,e),y(Ze.$$.fragment,e),y(vt.$$.fragment,e),Yn=!1},d(e){e&&(i(u),i(n),i(d),i(k),i(h),i(j),i(Ie),i(yn),i(Be),i(Mn),i(Re),i(Tn),i(kn),i(Ge),i(wn),i(vn),i(Ne),i($n),i(Xe),i(Jn),i(jn),i(Vn),i(I),i(Cn),i(xn),i(C),i(Un),i(zn),i(B),i(Zn),i(Wn),i(x),i(Dn),i(Fn),i(R),i(qn),i(In),i(U),i(Bn),i(Rn),i(z),i(Gn),i(Nn),i(Z),i(Xn),i(Sn),i(W),i(Qn),i(Hn),i(D),i(En),i(Ln),i(bn)),i(t),M(ne,e),M(be,e),M(_e,e),M(Se,e),M(Qe,e),M(He),M(ye),M(Ee,e),M(Le),M(Ye),M(Pe),M(Ae),M(Oe),M(Ke,e),M(et),M(tt),M(nt),M(ot,e),M(st),M(at),M(Te),M(rt,e),M(it),M(lt),M(ke),M(dt,e),M(ct),M(pt),M(we),M(ve),M(mt,e),M(ut),M(ht),M($e),M(Je),M(je),M(ft,e),M(gt),M(bt),M(Ve),M(Ce),M(_t,e),M(yt),M(Mt),M(xe),M(Ue),M(Tt,e),M(kt),M(wt),M(ze),M(Ze),M(vt,e)}}}const za='{"title":"DeBERTa-v2","local":"deberta-v2","sections":[{"title":"DebertaV2Config","local":"transformers.DebertaV2Config","sections":[],"depth":2},{"title":"DebertaV2Tokenizer","local":"transformers.DebertaV2Tokenizer","sections":[],"depth":2},{"title":"DebertaV2TokenizerFast","local":"transformers.DebertaV2TokenizerFast","sections":[],"depth":2},{"title":"DebertaV2Model","local":"transformers.DebertaV2Model","sections":[],"depth":2},{"title":"DebertaV2PreTrainedModel","local":"transformers.DebertaV2PreTrainedModel","sections":[],"depth":2},{"title":"DebertaV2ForMaskedLM","local":"transformers.DebertaV2ForMaskedLM","sections":[],"depth":2},{"title":"DebertaV2ForSequenceClassification","local":"transformers.DebertaV2ForSequenceClassification","sections":[],"depth":2},{"title":"DebertaV2ForTokenClassification","local":"transformers.DebertaV2ForTokenClassification","sections":[],"depth":2},{"title":"DebertaV2ForQuestionAnswering","local":"transformers.DebertaV2ForQuestionAnswering","sections":[],"depth":2},{"title":"DebertaV2ForMultipleChoice","local":"transformers.DebertaV2ForMultipleChoice","sections":[],"depth":2}],"depth":1}';function Za(w){return aa(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class Na extends ra{constructor(t){super(),ia(this,t,Za,Ua,sa,{})}}export{Na as component};
