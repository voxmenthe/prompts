import{s as Zr,o as Nr,n as J}from"../chunks/scheduler.18a86fab.js";import{S as Gr,i as Rr,g as c,s as r,r as g,A as Vr,h as p,f as d,c as a,j as w,x as k,u,k as v,l as Hr,y as o,a as h,v as f,d as _,t as b,w as T}from"../chunks/index.98837b22.js";import{T as ye}from"../chunks/Tip.77304350.js";import{D as $}from"../chunks/Docstring.a1ef7999.js";import{C as V}from"../chunks/CodeBlock.8d0c2e8a.js";import{E as ae}from"../chunks/ExampleCodeBlock.8c3ee1f9.js";import{H as Z,E as Xr}from"../chunks/getInferenceSnippets.06c2775f.js";import{H as Er,a as Hs}from"../chunks/HfOption.6641485e.js";function Sr(B){let t,m="Click on the BigBird models in the right sidebar for more examples of how to apply BigBird to different language tasks.";return{c(){t=c("p"),t.textContent=m},l(n){t=p(n,"P",{"data-svelte-h":!0}),k(t)!=="svelte-6gcpht"&&(t.textContent=m)},m(n,l){h(n,t,l)},p:J,d(n){n&&d(t)}}}function Pr(B){let t,m;return t=new V({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwcGlwZWxpbmUlMEElMEFwaXBlbGluZSUyMCUzRCUyMHBpcGVsaW5lKCUwQSUyMCUyMCUyMCUyMHRhc2slM0QlMjJmaWxsLW1hc2slMjIlMkMlMEElMjAlMjAlMjAlMjBtb2RlbCUzRCUyMmdvb2dsZSUyRmJpZ2JpcmQtcm9iZXJ0YS1iYXNlJTIyJTJDJTBBJTIwJTIwJTIwJTIwZHR5cGUlM0R0b3JjaC5mbG9hdDE2JTJDJTBBJTIwJTIwJTIwJTIwZGV2aWNlJTNEMCUwQSklMEFwaXBlbGluZSglMjJQbGFudHMlMjBjcmVhdGUlMjAlNUJNQVNLJTVEJTIwdGhyb3VnaCUyMGElMjBwcm9jZXNzJTIwa25vd24lMjBhcyUyMHBob3Rvc3ludGhlc2lzLiUyMik=",highlighted:`<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> pipeline

pipeline = pipeline(
    task=<span class="hljs-string">&quot;fill-mask&quot;</span>,
    model=<span class="hljs-string">&quot;google/bigbird-roberta-base&quot;</span>,
    dtype=torch.float16,
    device=<span class="hljs-number">0</span>
)
pipeline(<span class="hljs-string">&quot;Plants create [MASK] through a process known as photosynthesis.&quot;</span>)`,wrap:!1}}),{c(){g(t.$$.fragment)},l(n){u(t.$$.fragment,n)},m(n,l){f(t,n,l),m=!0},p:J,i(n){m||(_(t.$$.fragment,n),m=!0)},o(n){b(t.$$.fragment,n),m=!1},d(n){T(t,n)}}}function Qr(B){let t,m;return t=new V({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b01vZGVsRm9yTWFza2VkTE0lMkMlMjBBdXRvVG9rZW5pemVyJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTBBJTIwJTIwJTIwJTIwJTIyZ29vZ2xlJTJGYmlnYmlyZC1yb2JlcnRhLWJhc2UlMjIlMkMlMEEpJTBBbW9kZWwlMjAlM0QlMjBBdXRvTW9kZWxGb3JNYXNrZWRMTS5mcm9tX3ByZXRyYWluZWQoJTBBJTIwJTIwJTIwJTIwJTIyZ29vZ2xlJTJGYmlnYmlyZC1yb2JlcnRhLWJhc2UlMjIlMkMlMEElMjAlMjAlMjAlMjBkdHlwZSUzRHRvcmNoLmZsb2F0MTYlMkMlMEElMjAlMjAlMjAlMjBkZXZpY2VfbWFwJTNEJTIyYXV0byUyMiUyQyUwQSklMEFpbnB1dHMlMjAlM0QlMjB0b2tlbml6ZXIoJTIyUGxhbnRzJTIwY3JlYXRlJTIwJTVCTUFTSyU1RCUyMHRocm91Z2glMjBhJTIwcHJvY2VzcyUyMGtub3duJTIwYXMlMjBwaG90b3N5bnRoZXNpcy4lMjIlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKS50byhtb2RlbC5kZXZpY2UpJTBBJTBBd2l0aCUyMHRvcmNoLm5vX2dyYWQoKSUzQSUwQSUyMCUyMCUyMCUyMG91dHB1dHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyklMEElMjAlMjAlMjAlMjBwcmVkaWN0aW9ucyUyMCUzRCUyMG91dHB1dHMubG9naXRzJTBBJTBBbWFza2VkX2luZGV4JTIwJTNEJTIwdG9yY2gud2hlcmUoaW5wdXRzJTVCJ2lucHV0X2lkcyclNUQlMjAlM0QlM0QlMjB0b2tlbml6ZXIubWFza190b2tlbl9pZCklNUIxJTVEJTBBcHJlZGljdGVkX3Rva2VuX2lkJTIwJTNEJTIwcHJlZGljdGlvbnMlNUIwJTJDJTIwbWFza2VkX2luZGV4JTVELmFyZ21heChkaW0lM0QtMSklMEFwcmVkaWN0ZWRfdG9rZW4lMjAlM0QlMjB0b2tlbml6ZXIuZGVjb2RlKHByZWRpY3RlZF90b2tlbl9pZCklMEElMEFwcmludChmJTIyVGhlJTIwcHJlZGljdGVkJTIwdG9rZW4lMjBpcyUzQSUyMCU3QnByZWRpY3RlZF90b2tlbiU3RCUyMik=",highlighted:`<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoModelForMaskedLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    <span class="hljs-string">&quot;google/bigbird-roberta-base&quot;</span>,
)
model = AutoModelForMaskedLM.from_pretrained(
    <span class="hljs-string">&quot;google/bigbird-roberta-base&quot;</span>,
    dtype=torch.float16,
    device_map=<span class="hljs-string">&quot;auto&quot;</span>,
)
inputs = tokenizer(<span class="hljs-string">&quot;Plants create [MASK] through a process known as photosynthesis.&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>).to(model.device)

<span class="hljs-keyword">with</span> torch.no_grad():
    outputs = model(**inputs)
    predictions = outputs.logits

masked_index = torch.where(inputs[<span class="hljs-string">&#x27;input_ids&#x27;</span>] == tokenizer.mask_token_id)[<span class="hljs-number">1</span>]
predicted_token_id = predictions[<span class="hljs-number">0</span>, masked_index].argmax(dim=-<span class="hljs-number">1</span>)
predicted_token = tokenizer.decode(predicted_token_id)

<span class="hljs-built_in">print</span>(<span class="hljs-string">f&quot;The predicted token is: <span class="hljs-subst">{predicted_token}</span>&quot;</span>)`,wrap:!1}}),{c(){g(t.$$.fragment)},l(n){u(t.$$.fragment,n)},m(n,l){f(t,n,l),m=!0},p:J,i(n){m||(_(t.$$.fragment,n),m=!0)},o(n){b(t.$$.fragment,n),m=!1},d(n){T(t,n)}}}function Or(B){let t,m;return t=new V({props:{code:"IWVjaG8lMjAtZSUyMCUyMlBsYW50cyUyMGNyZWF0ZSUyMCU1Qk1BU0slNUQlMjB0aHJvdWdoJTIwYSUyMHByb2Nlc3MlMjBrbm93biUyMGFzJTIwcGhvdG9zeW50aGVzaXMuJTIyJTIwJTdDJTIwdHJhbnNmb3JtZXJzLWNsaSUyMHJ1biUyMC0tdGFzayUyMGZpbGwtbWFzayUyMC0tbW9kZWwlMjBnb29nbGUlMkZiaWdiaXJkLXJvYmVydGEtYmFzZSUyMC0tZGV2aWNlJTIwMA==",highlighted:'!<span class="hljs-built_in">echo</span> -e <span class="hljs-string">&quot;Plants create [MASK] through a process known as photosynthesis.&quot;</span> | transformers-cli run --task fill-mask --model google/bigbird-roberta-base --device 0',wrap:!1}}),{c(){g(t.$$.fragment)},l(n){u(t.$$.fragment,n)},m(n,l){f(t,n,l),m=!0},p:J,i(n){m||(_(t.$$.fragment,n),m=!0)},o(n){b(t.$$.fragment,n),m=!1},d(n){T(t,n)}}}function Ar(B){let t,m,n,l,M,s;return t=new Hs({props:{id:"usage",option:"Pipeline",$$slots:{default:[Pr]},$$scope:{ctx:B}}}),n=new Hs({props:{id:"usage",option:"AutoModel",$$slots:{default:[Qr]},$$scope:{ctx:B}}}),M=new Hs({props:{id:"usage",option:"transformers CLI",$$slots:{default:[Or]},$$scope:{ctx:B}}}),{c(){g(t.$$.fragment),m=r(),g(n.$$.fragment),l=r(),g(M.$$.fragment)},l(y){u(t.$$.fragment,y),m=a(y),u(n.$$.fragment,y),l=a(y),u(M.$$.fragment,y)},m(y,C){f(t,y,C),h(y,m,C),f(n,y,C),h(y,l,C),f(M,y,C),s=!0},p(y,C){const qn={};C&2&&(qn.$$scope={dirty:C,ctx:y}),t.$set(qn);const Pe={};C&2&&(Pe.$$scope={dirty:C,ctx:y}),n.$set(Pe);const ie={};C&2&&(ie.$$scope={dirty:C,ctx:y}),M.$set(ie)},i(y){s||(_(t.$$.fragment,y),_(n.$$.fragment,y),_(M.$$.fragment,y),s=!0)},o(y){b(t.$$.fragment,y),b(n.$$.fragment,y),b(M.$$.fragment,y),s=!1},d(y){y&&(d(m),d(l)),T(t,y),T(n,y),T(M,y)}}}function Yr(B){let t,m="Example:",n,l,M;return l=new V({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEJpZ0JpcmRDb25maWclMkMlMjBCaWdCaXJkTW9kZWwlMEElMEElMjMlMjBJbml0aWFsaXppbmclMjBhJTIwQmlnQmlyZCUyMGdvb2dsZSUyRmJpZ2JpcmQtcm9iZXJ0YS1iYXNlJTIwc3R5bGUlMjBjb25maWd1cmF0aW9uJTBBY29uZmlndXJhdGlvbiUyMCUzRCUyMEJpZ0JpcmRDb25maWcoKSUwQSUwQSUyMyUyMEluaXRpYWxpemluZyUyMGElMjBtb2RlbCUyMCh3aXRoJTIwcmFuZG9tJTIwd2VpZ2h0cyklMjBmcm9tJTIwdGhlJTIwZ29vZ2xlJTJGYmlnYmlyZC1yb2JlcnRhLWJhc2UlMjBzdHlsZSUyMGNvbmZpZ3VyYXRpb24lMEFtb2RlbCUyMCUzRCUyMEJpZ0JpcmRNb2RlbChjb25maWd1cmF0aW9uKSUwQSUwQSUyMyUyMEFjY2Vzc2luZyUyMHRoZSUyMG1vZGVsJTIwY29uZmlndXJhdGlvbiUwQWNvbmZpZ3VyYXRpb24lMjAlM0QlMjBtb2RlbC5jb25maWc=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> BigBirdConfig, BigBirdModel

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a BigBird google/bigbird-roberta-base style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = BigBirdConfig()

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a model (with random weights) from the google/bigbird-roberta-base style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = BigBirdModel(configuration)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Accessing the model configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = model.config`,wrap:!1}}),{c(){t=c("p"),t.textContent=m,n=r(),g(l.$$.fragment)},l(s){t=p(s,"P",{"data-svelte-h":!0}),k(t)!=="svelte-11lpom8"&&(t.textContent=m),n=a(s),u(l.$$.fragment,s)},m(s,y){h(s,t,y),h(s,n,y),f(l,s,y),M=!0},p:J,i(s){M||(_(l.$$.fragment,s),M=!0)},o(s){b(l.$$.fragment,s),M=!1},d(s){s&&(d(t),d(n)),T(l,s)}}}function Dr(B){let t,m=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=m},l(n){t=p(n,"P",{"data-svelte-h":!0}),k(t)!=="svelte-fincs2"&&(t.innerHTML=m)},m(n,l){h(n,t,l)},p:J,d(n){n&&d(t)}}}function Kr(B){let t,m=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=m},l(n){t=p(n,"P",{"data-svelte-h":!0}),k(t)!=="svelte-fincs2"&&(t.innerHTML=m)},m(n,l){h(n,t,l)},p:J,d(n){n&&d(t)}}}function ea(B){let t,m="Example:",n,l,M;return l=new V({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBCaWdCaXJkRm9yUHJlVHJhaW5pbmclMEFpbXBvcnQlMjB0b3JjaCUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMmdvb2dsZSUyRmJpZ2JpcmQtcm9iZXJ0YS1iYXNlJTIyKSUwQW1vZGVsJTIwJTNEJTIwQmlnQmlyZEZvclByZVRyYWluaW5nLmZyb21fcHJldHJhaW5lZCglMjJnb29nbGUlMkZiaWdiaXJkLXJvYmVydGEtYmFzZSUyMiklMEElMEFpbnB1dHMlMjAlM0QlMjB0b2tlbml6ZXIoJTIySGVsbG8lMkMlMjBteSUyMGRvZyUyMGlzJTIwY3V0ZSUyMiUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpJTBBb3V0cHV0cyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzKSUwQSUwQXByZWRpY3Rpb25fbG9naXRzJTIwJTNEJTIwb3V0cHV0cy5wcmVkaWN0aW9uX2xvZ2l0cyUwQXNlcV9yZWxhdGlvbnNoaXBfbG9naXRzJTIwJTNEJTIwb3V0cHV0cy5zZXFfcmVsYXRpb25zaGlwX2xvZ2l0cw==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, BigBirdForPreTraining
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;google/bigbird-roberta-base&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = BigBirdForPreTraining.from_pretrained(<span class="hljs-string">&quot;google/bigbird-roberta-base&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs)

<span class="hljs-meta">&gt;&gt;&gt; </span>prediction_logits = outputs.prediction_logits
<span class="hljs-meta">&gt;&gt;&gt; </span>seq_relationship_logits = outputs.seq_relationship_logits`,wrap:!1}}),{c(){t=c("p"),t.textContent=m,n=r(),g(l.$$.fragment)},l(s){t=p(s,"P",{"data-svelte-h":!0}),k(t)!=="svelte-11lpom8"&&(t.textContent=m),n=a(s),u(l.$$.fragment,s)},m(s,y){h(s,t,y),h(s,n,y),f(l,s,y),M=!0},p:J,i(s){M||(_(l.$$.fragment,s),M=!0)},o(s){b(l.$$.fragment,s),M=!1},d(s){s&&(d(t),d(n)),T(l,s)}}}function ta(B){let t,m=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=m},l(n){t=p(n,"P",{"data-svelte-h":!0}),k(t)!=="svelte-fincs2"&&(t.innerHTML=m)},m(n,l){h(n,t,l)},p:J,d(n){n&&d(t)}}}function na(B){let t,m="Example:",n,l,M;return l=new V({props:{code:"",highlighted:"",wrap:!1}}),{c(){t=c("p"),t.textContent=m,n=r(),g(l.$$.fragment)},l(s){t=p(s,"P",{"data-svelte-h":!0}),k(t)!=="svelte-11lpom8"&&(t.textContent=m),n=a(s),u(l.$$.fragment,s)},m(s,y){h(s,t,y),h(s,n,y),f(l,s,y),M=!0},p:J,i(s){M||(_(l.$$.fragment,s),M=!0)},o(s){b(l.$$.fragment,s),M=!1},d(s){s&&(d(t),d(n)),T(l,s)}}}function oa(B){let t,m=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=m},l(n){t=p(n,"P",{"data-svelte-h":!0}),k(t)!=="svelte-fincs2"&&(t.innerHTML=m)},m(n,l){h(n,t,l)},p:J,d(n){n&&d(t)}}}function sa(B){let t,m="Example:",n,l,M;return l=new V({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Rva2VuaXplciUyQyUyMEJpZ0JpcmRGb3JNYXNrZWRMTSUwQWZyb20lMjBkYXRhc2V0cyUyMGltcG9ydCUyMGxvYWRfZGF0YXNldCUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMmdvb2dsZSUyRmJpZ2JpcmQtcm9iZXJ0YS1iYXNlJTIyKSUwQW1vZGVsJTIwJTNEJTIwQmlnQmlyZEZvck1hc2tlZExNLmZyb21fcHJldHJhaW5lZCglMjJnb29nbGUlMkZiaWdiaXJkLXJvYmVydGEtYmFzZSUyMiklMEFzcXVhZF9kcyUyMCUzRCUyMGxvYWRfZGF0YXNldCglMjJyYWpwdXJrYXIlMkZzcXVhZF92MiUyMiUyQyUyMHNwbGl0JTNEJTIydHJhaW4lMjIpJTBBJTIzJTIwc2VsZWN0JTIwcmFuZG9tJTIwbG9uZyUyMGFydGljbGUlMEFMT05HX0FSVElDTEVfVEFSR0VUJTIwJTNEJTIwc3F1YWRfZHMlNUI4MTUxNCU1RCU1QiUyMmNvbnRleHQlMjIlNUQlMEElMjMlMjBzZWxlY3QlMjByYW5kb20lMjBzZW50ZW5jZSUwQUxPTkdfQVJUSUNMRV9UQVJHRVQlNUIzMzIlM0EzOTglNUQlMEElMEElMjMlMjBhZGQlMjBtYXNrX3Rva2VuJTBBTE9OR19BUlRJQ0xFX1RPX01BU0slMjAlM0QlMjBMT05HX0FSVElDTEVfVEFSR0VULnJlcGxhY2UoJTIybWF4aW11bSUyMiUyQyUyMCUyMiU1Qk1BU0slNUQlMjIpJTBBaW5wdXRzJTIwJTNEJTIwdG9rZW5pemVyKExPTkdfQVJUSUNMRV9UT19NQVNLJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiklMEElMjMlMjBsb25nJTIwYXJ0aWNsZSUyMGlucHV0JTBBbGlzdChpbnB1dHMlNUIlMjJpbnB1dF9pZHMlMjIlNUQuc2hhcGUpJTBBJTBBd2l0aCUyMHRvcmNoLm5vX2dyYWQoKSUzQSUwQSUyMCUyMCUyMCUyMGxvZ2l0cyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzKS5sb2dpdHMlMEElMjMlMjByZXRyaWV2ZSUyMGluZGV4JTIwb2YlMjAlNUJNQVNLJTVEJTBBbWFza190b2tlbl9pbmRleCUyMCUzRCUyMChpbnB1dHMuaW5wdXRfaWRzJTIwJTNEJTNEJTIwdG9rZW5pemVyLm1hc2tfdG9rZW5faWQpJTVCMCU1RC5ub256ZXJvKGFzX3R1cGxlJTNEVHJ1ZSklNUIwJTVEJTBBcHJlZGljdGVkX3Rva2VuX2lkJTIwJTNEJTIwbG9naXRzJTVCMCUyQyUyMG1hc2tfdG9rZW5faW5kZXglNUQuYXJnbWF4KGF4aXMlM0QtMSklMEF0b2tlbml6ZXIuZGVjb2RlKHByZWRpY3RlZF90b2tlbl9pZCk=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, BigBirdForMaskedLM
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> datasets <span class="hljs-keyword">import</span> load_dataset

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;google/bigbird-roberta-base&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = BigBirdForMaskedLM.from_pretrained(<span class="hljs-string">&quot;google/bigbird-roberta-base&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>squad_ds = load_dataset(<span class="hljs-string">&quot;rajpurkar/squad_v2&quot;</span>, split=<span class="hljs-string">&quot;train&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># select random long article</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>LONG_ARTICLE_TARGET = squad_ds[<span class="hljs-number">81514</span>][<span class="hljs-string">&quot;context&quot;</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># select random sentence</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>LONG_ARTICLE_TARGET[<span class="hljs-number">332</span>:<span class="hljs-number">398</span>]
<span class="hljs-string">&#x27;the highest values are very close to the theoretical maximum value&#x27;</span>

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># add mask_token</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>LONG_ARTICLE_TO_MASK = LONG_ARTICLE_TARGET.replace(<span class="hljs-string">&quot;maximum&quot;</span>, <span class="hljs-string">&quot;[MASK]&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(LONG_ARTICLE_TO_MASK, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># long article input</span>
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">list</span>(inputs[<span class="hljs-string">&quot;input_ids&quot;</span>].shape)
[<span class="hljs-number">1</span>, <span class="hljs-number">919</span>]

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    logits = model(**inputs).logits
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># retrieve index of [MASK]</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[<span class="hljs-number">0</span>].nonzero(as_tuple=<span class="hljs-literal">True</span>)[<span class="hljs-number">0</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_token_id = logits[<span class="hljs-number">0</span>, mask_token_index].argmax(axis=-<span class="hljs-number">1</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer.decode(predicted_token_id)
<span class="hljs-string">&#x27;maximum&#x27;</span>`,wrap:!1}}),{c(){t=c("p"),t.textContent=m,n=r(),g(l.$$.fragment)},l(s){t=p(s,"P",{"data-svelte-h":!0}),k(t)!=="svelte-11lpom8"&&(t.textContent=m),n=a(s),u(l.$$.fragment,s)},m(s,y){h(s,t,y),h(s,n,y),f(l,s,y),M=!0},p:J,i(s){M||(_(l.$$.fragment,s),M=!0)},o(s){b(l.$$.fragment,s),M=!1},d(s){s&&(d(t),d(n)),T(l,s)}}}function ra(B){let t,m;return t=new V({props:{code:"bGFiZWxzJTIwJTNEJTIwdG9rZW5pemVyKExPTkdfQVJUSUNMRV9UQVJHRVQlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSU1QiUyMmlucHV0X2lkcyUyMiU1RCUwQWxhYmVscyUyMCUzRCUyMHRvcmNoLndoZXJlKGlucHV0cy5pbnB1dF9pZHMlMjAlM0QlM0QlMjB0b2tlbml6ZXIubWFza190b2tlbl9pZCUyQyUyMGxhYmVscyUyQyUyMC0xMDApJTBBb3V0cHV0cyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzJTJDJTIwbGFiZWxzJTNEbGFiZWxzKSUwQXJvdW5kKG91dHB1dHMubG9zcy5pdGVtKCklMkMlMjAyKQ==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span>labels = tokenizer(LONG_ARTICLE_TARGET, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)[<span class="hljs-string">&quot;input_ids&quot;</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.where(inputs.input_ids == tokenizer.mask_token_id, labels, -<span class="hljs-number">100</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs, labels=labels)
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">round</span>(outputs.loss.item(), <span class="hljs-number">2</span>)
<span class="hljs-number">1.99</span>`,wrap:!1}}),{c(){g(t.$$.fragment)},l(n){u(t.$$.fragment,n)},m(n,l){f(t,n,l),m=!0},p:J,i(n){m||(_(t.$$.fragment,n),m=!0)},o(n){b(t.$$.fragment,n),m=!1},d(n){T(t,n)}}}function aa(B){let t,m=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=m},l(n){t=p(n,"P",{"data-svelte-h":!0}),k(t)!=="svelte-fincs2"&&(t.innerHTML=m)},m(n,l){h(n,t,l)},p:J,d(n){n&&d(t)}}}function ia(B){let t,m="Example:",n,l,M;return l=new V({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Rva2VuaXplciUyQyUyMEJpZ0JpcmRGb3JTZXF1ZW5jZUNsYXNzaWZpY2F0aW9uJTBBZnJvbSUyMGRhdGFzZXRzJTIwaW1wb3J0JTIwbG9hZF9kYXRhc2V0JTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIybC15b2hhaSUyRmJpZ2JpcmQtcm9iZXJ0YS1iYXNlLW1ubGklMjIpJTBBbW9kZWwlMjAlM0QlMjBCaWdCaXJkRm9yU2VxdWVuY2VDbGFzc2lmaWNhdGlvbi5mcm9tX3ByZXRyYWluZWQoJTIybC15b2hhaSUyRmJpZ2JpcmQtcm9iZXJ0YS1iYXNlLW1ubGklMjIpJTBBc3F1YWRfZHMlMjAlM0QlMjBsb2FkX2RhdGFzZXQoJTIycmFqcHVya2FyJTJGc3F1YWRfdjIlMjIlMkMlMjBzcGxpdCUzRCUyMnRyYWluJTIyKSUwQUxPTkdfQVJUSUNMRSUyMCUzRCUyMHNxdWFkX2RzJTVCODE1MTQlNUQlNUIlMjJjb250ZXh0JTIyJTVEJTBBaW5wdXRzJTIwJTNEJTIwdG9rZW5pemVyKExPTkdfQVJUSUNMRSUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpJTBBJTIzJTIwbG9uZyUyMGlucHV0JTIwYXJ0aWNsZSUwQWxpc3QoaW5wdXRzJTVCJTIyaW5wdXRfaWRzJTIyJTVELnNoYXBlKSUwQSUwQXdpdGglMjB0b3JjaC5ub19ncmFkKCklM0ElMEElMjAlMjAlMjAlMjBsb2dpdHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cykubG9naXRzJTBBcHJlZGljdGVkX2NsYXNzX2lkJTIwJTNEJTIwbG9naXRzLmFyZ21heCgpLml0ZW0oKSUwQW1vZGVsLmNvbmZpZy5pZDJsYWJlbCU1QnByZWRpY3RlZF9jbGFzc19pZCU1RA==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, BigBirdForSequenceClassification
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> datasets <span class="hljs-keyword">import</span> load_dataset

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;l-yohai/bigbird-roberta-base-mnli&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = BigBirdForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;l-yohai/bigbird-roberta-base-mnli&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>squad_ds = load_dataset(<span class="hljs-string">&quot;rajpurkar/squad_v2&quot;</span>, split=<span class="hljs-string">&quot;train&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>LONG_ARTICLE = squad_ds[<span class="hljs-number">81514</span>][<span class="hljs-string">&quot;context&quot;</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(LONG_ARTICLE, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># long input article</span>
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">list</span>(inputs[<span class="hljs-string">&quot;input_ids&quot;</span>].shape)
[<span class="hljs-number">1</span>, <span class="hljs-number">919</span>]

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    logits = model(**inputs).logits
<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_class_id = logits.argmax().item()
<span class="hljs-meta">&gt;&gt;&gt; </span>model.config.id2label[predicted_class_id]
<span class="hljs-string">&#x27;LABEL_0&#x27;</span>`,wrap:!1}}),{c(){t=c("p"),t.textContent=m,n=r(),g(l.$$.fragment)},l(s){t=p(s,"P",{"data-svelte-h":!0}),k(t)!=="svelte-11lpom8"&&(t.textContent=m),n=a(s),u(l.$$.fragment,s)},m(s,y){h(s,t,y),h(s,n,y),f(l,s,y),M=!0},p:J,i(s){M||(_(l.$$.fragment,s),M=!0)},o(s){b(l.$$.fragment,s),M=!1},d(s){s&&(d(t),d(n)),T(l,s)}}}function da(B){let t,m;return t=new V({props:{code:"bnVtX2xhYmVscyUyMCUzRCUyMGxlbihtb2RlbC5jb25maWcuaWQybGFiZWwpJTBBbW9kZWwlMjAlM0QlMjBCaWdCaXJkRm9yU2VxdWVuY2VDbGFzc2lmaWNhdGlvbi5mcm9tX3ByZXRyYWluZWQoJTBBJTIwJTIwJTIwJTIwJTIybC15b2hhaSUyRmJpZ2JpcmQtcm9iZXJ0YS1iYXNlLW1ubGklMjIlMkMlMjBudW1fbGFiZWxzJTNEbnVtX2xhYmVscyUwQSklMEFsYWJlbHMlMjAlM0QlMjB0b3JjaC50ZW5zb3IoMSklMEFsb3NzJTIwJTNEJTIwbW9kZWwoKippbnB1dHMlMkMlMjBsYWJlbHMlM0RsYWJlbHMpLmxvc3MlMEFyb3VuZChsb3NzLml0ZW0oKSUyQyUyMDIp",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span>num_labels = <span class="hljs-built_in">len</span>(model.config.id2label)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = BigBirdForSequenceClassification.from_pretrained(
<span class="hljs-meta">... </span>    <span class="hljs-string">&quot;l-yohai/bigbird-roberta-base-mnli&quot;</span>, num_labels=num_labels
<span class="hljs-meta">... </span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.tensor(<span class="hljs-number">1</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = model(**inputs, labels=labels).loss
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">round</span>(loss.item(), <span class="hljs-number">2</span>)
<span class="hljs-number">1.13</span>`,wrap:!1}}),{c(){g(t.$$.fragment)},l(n){u(t.$$.fragment,n)},m(n,l){f(t,n,l),m=!0},p:J,i(n){m||(_(t.$$.fragment,n),m=!0)},o(n){b(t.$$.fragment,n),m=!1},d(n){T(t,n)}}}function la(B){let t,m=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=m},l(n){t=p(n,"P",{"data-svelte-h":!0}),k(t)!=="svelte-fincs2"&&(t.innerHTML=m)},m(n,l){h(n,t,l)},p:J,d(n){n&&d(t)}}}function ca(B){let t,m="Example:",n,l,M;return l=new V({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBCaWdCaXJkRm9yTXVsdGlwbGVDaG9pY2UlMEFpbXBvcnQlMjB0b3JjaCUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMmdvb2dsZSUyRmJpZ2JpcmQtcm9iZXJ0YS1iYXNlJTIyKSUwQW1vZGVsJTIwJTNEJTIwQmlnQmlyZEZvck11bHRpcGxlQ2hvaWNlLmZyb21fcHJldHJhaW5lZCglMjJnb29nbGUlMkZiaWdiaXJkLXJvYmVydGEtYmFzZSUyMiklMEElMEFwcm9tcHQlMjAlM0QlMjAlMjJJbiUyMEl0YWx5JTJDJTIwcGl6emElMjBzZXJ2ZWQlMjBpbiUyMGZvcm1hbCUyMHNldHRpbmdzJTJDJTIwc3VjaCUyMGFzJTIwYXQlMjBhJTIwcmVzdGF1cmFudCUyQyUyMGlzJTIwcHJlc2VudGVkJTIwdW5zbGljZWQuJTIyJTBBY2hvaWNlMCUyMCUzRCUyMCUyMkl0JTIwaXMlMjBlYXRlbiUyMHdpdGglMjBhJTIwZm9yayUyMGFuZCUyMGElMjBrbmlmZS4lMjIlMEFjaG9pY2UxJTIwJTNEJTIwJTIySXQlMjBpcyUyMGVhdGVuJTIwd2hpbGUlMjBoZWxkJTIwaW4lMjB0aGUlMjBoYW5kLiUyMiUwQWxhYmVscyUyMCUzRCUyMHRvcmNoLnRlbnNvcigwKS51bnNxdWVlemUoMCklMjAlMjAlMjMlMjBjaG9pY2UwJTIwaXMlMjBjb3JyZWN0JTIwKGFjY29yZGluZyUyMHRvJTIwV2lraXBlZGlhJTIwJTNCKSklMkMlMjBiYXRjaCUyMHNpemUlMjAxJTBBJTBBZW5jb2RpbmclMjAlM0QlMjB0b2tlbml6ZXIoJTVCcHJvbXB0JTJDJTIwcHJvbXB0JTVEJTJDJTIwJTVCY2hvaWNlMCUyQyUyMGNob2ljZTElNUQlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyJTJDJTIwcGFkZGluZyUzRFRydWUpJTBBb3V0cHV0cyUyMCUzRCUyMG1vZGVsKCoqJTdCayUzQSUyMHYudW5zcXVlZXplKDApJTIwZm9yJTIwayUyQyUyMHYlMjBpbiUyMGVuY29kaW5nLml0ZW1zKCklN0QlMkMlMjBsYWJlbHMlM0RsYWJlbHMpJTIwJTIwJTIzJTIwYmF0Y2glMjBzaXplJTIwaXMlMjAxJTBBJTBBJTIzJTIwdGhlJTIwbGluZWFyJTIwY2xhc3NpZmllciUyMHN0aWxsJTIwbmVlZHMlMjB0byUyMGJlJTIwdHJhaW5lZCUwQWxvc3MlMjAlM0QlMjBvdXRwdXRzLmxvc3MlMEFsb2dpdHMlMjAlM0QlMjBvdXRwdXRzLmxvZ2l0cw==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, BigBirdForMultipleChoice
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;google/bigbird-roberta-base&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = BigBirdForMultipleChoice.from_pretrained(<span class="hljs-string">&quot;google/bigbird-roberta-base&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>prompt = <span class="hljs-string">&quot;In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>choice0 = <span class="hljs-string">&quot;It is eaten with a fork and a knife.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>choice1 = <span class="hljs-string">&quot;It is eaten while held in the hand.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.tensor(<span class="hljs-number">0</span>).unsqueeze(<span class="hljs-number">0</span>)  <span class="hljs-comment"># choice0 is correct (according to Wikipedia ;)), batch size 1</span>

<span class="hljs-meta">&gt;&gt;&gt; </span>encoding = tokenizer([prompt, prompt], [choice0, choice1], return_tensors=<span class="hljs-string">&quot;pt&quot;</span>, padding=<span class="hljs-literal">True</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**{k: v.unsqueeze(<span class="hljs-number">0</span>) <span class="hljs-keyword">for</span> k, v <span class="hljs-keyword">in</span> encoding.items()}, labels=labels)  <span class="hljs-comment"># batch size is 1</span>

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># the linear classifier still needs to be trained</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = outputs.loss
<span class="hljs-meta">&gt;&gt;&gt; </span>logits = outputs.logits`,wrap:!1}}),{c(){t=c("p"),t.textContent=m,n=r(),g(l.$$.fragment)},l(s){t=p(s,"P",{"data-svelte-h":!0}),k(t)!=="svelte-11lpom8"&&(t.textContent=m),n=a(s),u(l.$$.fragment,s)},m(s,y){h(s,t,y),h(s,n,y),f(l,s,y),M=!0},p:J,i(s){M||(_(l.$$.fragment,s),M=!0)},o(s){b(l.$$.fragment,s),M=!1},d(s){s&&(d(t),d(n)),T(l,s)}}}function pa(B){let t,m=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=m},l(n){t=p(n,"P",{"data-svelte-h":!0}),k(t)!=="svelte-fincs2"&&(t.innerHTML=m)},m(n,l){h(n,t,l)},p:J,d(n){n&&d(t)}}}function ma(B){let t,m="Example:",n,l,M;return l=new V({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBCaWdCaXJkRm9yVG9rZW5DbGFzc2lmaWNhdGlvbiUwQWltcG9ydCUyMHRvcmNoJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIyZ29vZ2xlJTJGYmlnYmlyZC1yb2JlcnRhLWJhc2UlMjIpJTBBbW9kZWwlMjAlM0QlMjBCaWdCaXJkRm9yVG9rZW5DbGFzc2lmaWNhdGlvbi5mcm9tX3ByZXRyYWluZWQoJTIyZ29vZ2xlJTJGYmlnYmlyZC1yb2JlcnRhLWJhc2UlMjIpJTBBJTBBaW5wdXRzJTIwJTNEJTIwdG9rZW5pemVyKCUwQSUyMCUyMCUyMCUyMCUyMkh1Z2dpbmdGYWNlJTIwaXMlMjBhJTIwY29tcGFueSUyMGJhc2VkJTIwaW4lMjBQYXJpcyUyMGFuZCUyME5ldyUyMFlvcmslMjIlMkMlMjBhZGRfc3BlY2lhbF90b2tlbnMlM0RGYWxzZSUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIlMEEpJTBBJTBBd2l0aCUyMHRvcmNoLm5vX2dyYWQoKSUzQSUwQSUyMCUyMCUyMCUyMGxvZ2l0cyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzKS5sb2dpdHMlMEElMEFwcmVkaWN0ZWRfdG9rZW5fY2xhc3NfaWRzJTIwJTNEJTIwbG9naXRzLmFyZ21heCgtMSklMEElMEElMjMlMjBOb3RlJTIwdGhhdCUyMHRva2VucyUyMGFyZSUyMGNsYXNzaWZpZWQlMjByYXRoZXIlMjB0aGVuJTIwaW5wdXQlMjB3b3JkcyUyMHdoaWNoJTIwbWVhbnMlMjB0aGF0JTBBJTIzJTIwdGhlcmUlMjBtaWdodCUyMGJlJTIwbW9yZSUyMHByZWRpY3RlZCUyMHRva2VuJTIwY2xhc3NlcyUyMHRoYW4lMjB3b3Jkcy4lMEElMjMlMjBNdWx0aXBsZSUyMHRva2VuJTIwY2xhc3NlcyUyMG1pZ2h0JTIwYWNjb3VudCUyMGZvciUyMHRoZSUyMHNhbWUlMjB3b3JkJTBBcHJlZGljdGVkX3Rva2Vuc19jbGFzc2VzJTIwJTNEJTIwJTVCbW9kZWwuY29uZmlnLmlkMmxhYmVsJTVCdC5pdGVtKCklNUQlMjBmb3IlMjB0JTIwaW4lMjBwcmVkaWN0ZWRfdG9rZW5fY2xhc3NfaWRzJTVCMCU1RCU1RCUwQXByZWRpY3RlZF90b2tlbnNfY2xhc3NlcyUwQSUwQWxhYmVscyUyMCUzRCUyMHByZWRpY3RlZF90b2tlbl9jbGFzc19pZHMlMEFsb3NzJTIwJTNEJTIwbW9kZWwoKippbnB1dHMlMkMlMjBsYWJlbHMlM0RsYWJlbHMpLmxvc3MlMEFyb3VuZChsb3NzLml0ZW0oKSUyQyUyMDIp",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, BigBirdForTokenClassification
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;google/bigbird-roberta-base&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = BigBirdForTokenClassification.from_pretrained(<span class="hljs-string">&quot;google/bigbird-roberta-base&quot;</span>)

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
...`,wrap:!1}}),{c(){t=c("p"),t.textContent=m,n=r(),g(l.$$.fragment)},l(s){t=p(s,"P",{"data-svelte-h":!0}),k(t)!=="svelte-11lpom8"&&(t.textContent=m),n=a(s),u(l.$$.fragment,s)},m(s,y){h(s,t,y),h(s,n,y),f(l,s,y),M=!0},p:J,i(s){M||(_(l.$$.fragment,s),M=!0)},o(s){b(l.$$.fragment,s),M=!1},d(s){s&&(d(t),d(n)),T(l,s)}}}function ha(B){let t,m=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=m},l(n){t=p(n,"P",{"data-svelte-h":!0}),k(t)!=="svelte-fincs2"&&(t.innerHTML=m)},m(n,l){h(n,t,l)},p:J,d(n){n&&d(t)}}}function ga(B){let t,m="Example:",n,l,M;return l=new V({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Rva2VuaXplciUyQyUyMEJpZ0JpcmRGb3JRdWVzdGlvbkFuc3dlcmluZyUwQWZyb20lMjBkYXRhc2V0cyUyMGltcG9ydCUyMGxvYWRfZGF0YXNldCUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMmdvb2dsZSUyRmJpZ2JpcmQtcm9iZXJ0YS1iYXNlJTIyKSUwQW1vZGVsJTIwJTNEJTIwQmlnQmlyZEZvclF1ZXN0aW9uQW5zd2VyaW5nLmZyb21fcHJldHJhaW5lZCglMjJnb29nbGUlMkZiaWdiaXJkLXJvYmVydGEtYmFzZSUyMiklMEFzcXVhZF9kcyUyMCUzRCUyMGxvYWRfZGF0YXNldCglMjJyYWpwdXJrYXIlMkZzcXVhZF92MiUyMiUyQyUyMHNwbGl0JTNEJTIydHJhaW4lMjIpJTBBJTIzJTIwc2VsZWN0JTIwcmFuZG9tJTIwYXJ0aWNsZSUyMGFuZCUyMHF1ZXN0aW9uJTBBTE9OR19BUlRJQ0xFJTIwJTNEJTIwc3F1YWRfZHMlNUI4MTUxNCU1RCU1QiUyMmNvbnRleHQlMjIlNUQlMEFRVUVTVElPTiUyMCUzRCUyMHNxdWFkX2RzJTVCODE1MTQlNUQlNUIlMjJxdWVzdGlvbiUyMiU1RCUwQVFVRVNUSU9OJTBBJTBBaW5wdXRzJTIwJTNEJTIwdG9rZW5pemVyKFFVRVNUSU9OJTJDJTIwTE9OR19BUlRJQ0xFJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiklMEElMjMlMjBsb25nJTIwYXJ0aWNsZSUyMGFuZCUyMHF1ZXN0aW9uJTIwaW5wdXQlMEFsaXN0KGlucHV0cyU1QiUyMmlucHV0X2lkcyUyMiU1RC5zaGFwZSklMEElMEF3aXRoJTIwdG9yY2gubm9fZ3JhZCgpJTNBJTBBJTIwJTIwJTIwJTIwb3V0cHV0cyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzKSUwQSUwQWFuc3dlcl9zdGFydF9pbmRleCUyMCUzRCUyMG91dHB1dHMuc3RhcnRfbG9naXRzLmFyZ21heCgpJTBBYW5zd2VyX2VuZF9pbmRleCUyMCUzRCUyMG91dHB1dHMuZW5kX2xvZ2l0cy5hcmdtYXgoKSUwQXByZWRpY3RfYW5zd2VyX3Rva2VuX2lkcyUyMCUzRCUyMGlucHV0cy5pbnB1dF9pZHMlNUIwJTJDJTIwYW5zd2VyX3N0YXJ0X2luZGV4JTIwJTNBJTIwYW5zd2VyX2VuZF9pbmRleCUyMCUyQiUyMDElNUQlMEFwcmVkaWN0X2Fuc3dlcl90b2tlbiUyMCUzRCUyMHRva2VuaXplci5kZWNvZGUocHJlZGljdF9hbnN3ZXJfdG9rZW5faWRzKQ==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, BigBirdForQuestionAnswering
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> datasets <span class="hljs-keyword">import</span> load_dataset

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;google/bigbird-roberta-base&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = BigBirdForQuestionAnswering.from_pretrained(<span class="hljs-string">&quot;google/bigbird-roberta-base&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>squad_ds = load_dataset(<span class="hljs-string">&quot;rajpurkar/squad_v2&quot;</span>, split=<span class="hljs-string">&quot;train&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># select random article and question</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>LONG_ARTICLE = squad_ds[<span class="hljs-number">81514</span>][<span class="hljs-string">&quot;context&quot;</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span>QUESTION = squad_ds[<span class="hljs-number">81514</span>][<span class="hljs-string">&quot;question&quot;</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span>QUESTION
<span class="hljs-string">&#x27;During daytime how high can the temperatures reach?&#x27;</span>

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(QUESTION, LONG_ARTICLE, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># long article and question input</span>
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">list</span>(inputs[<span class="hljs-string">&quot;input_ids&quot;</span>].shape)
[<span class="hljs-number">1</span>, <span class="hljs-number">929</span>]

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    outputs = model(**inputs)

<span class="hljs-meta">&gt;&gt;&gt; </span>answer_start_index = outputs.start_logits.argmax()
<span class="hljs-meta">&gt;&gt;&gt; </span>answer_end_index = outputs.end_logits.argmax()
<span class="hljs-meta">&gt;&gt;&gt; </span>predict_answer_token_ids = inputs.input_ids[<span class="hljs-number">0</span>, answer_start_index : answer_end_index + <span class="hljs-number">1</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span>predict_answer_token = tokenizer.decode(predict_answer_token_ids)`,wrap:!1}}),{c(){t=c("p"),t.textContent=m,n=r(),g(l.$$.fragment)},l(s){t=p(s,"P",{"data-svelte-h":!0}),k(t)!=="svelte-11lpom8"&&(t.textContent=m),n=a(s),u(l.$$.fragment,s)},m(s,y){h(s,t,y),h(s,n,y),f(l,s,y),M=!0},p:J,i(s){M||(_(l.$$.fragment,s),M=!0)},o(s){b(l.$$.fragment,s),M=!1},d(s){s&&(d(t),d(n)),T(l,s)}}}function ua(B){let t,m;return t=new V({props:{code:"dGFyZ2V0X3N0YXJ0X2luZGV4JTJDJTIwdGFyZ2V0X2VuZF9pbmRleCUyMCUzRCUyMHRvcmNoLnRlbnNvciglNUIxMzAlNUQpJTJDJTIwdG9yY2gudGVuc29yKCU1QjEzMiU1RCklMEFvdXRwdXRzJTIwJTNEJTIwbW9kZWwoKippbnB1dHMlMkMlMjBzdGFydF9wb3NpdGlvbnMlM0R0YXJnZXRfc3RhcnRfaW5kZXglMkMlMjBlbmRfcG9zaXRpb25zJTNEdGFyZ2V0X2VuZF9pbmRleCklMEFsb3NzJTIwJTNEJTIwb3V0cHV0cy5sb3Nz",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span>target_start_index, target_end_index = torch.tensor([<span class="hljs-number">130</span>]), torch.tensor([<span class="hljs-number">132</span>])
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs, start_positions=target_start_index, end_positions=target_end_index)
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = outputs.loss`,wrap:!1}}),{c(){g(t.$$.fragment)},l(n){u(t.$$.fragment,n)},m(n,l){f(t,n,l),m=!0},p:J,i(n){m||(_(t.$$.fragment,n),m=!0)},o(n){b(t.$$.fragment,n),m=!1},d(n){T(t,n)}}}function fa(B){let t,m,n,l,M,s="<em>This model was released on 2020-07-28 and added to Hugging Face Transformers on 2021-03-30.</em>",y,C,qn='<div class="flex flex-wrap space-x-1"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/></div>',Pe,ie,Zn,Qe,Xs='<a href="https://huggingface.co/papers/2007.14062" rel="nofollow">BigBird</a> is a transformer model built to handle sequence lengths up to 4096 compared to 512 for <a href="./bert">BERT</a>. Traditional transformers struggle with long inputs because attention gets really expensive as the sequence length grows. BigBird fixes this by using a sparse attention mechanism, which means it doesn’t try to look at everything at once. Instead, it mixes in local attention, random attention, and a few global tokens to process the whole input. This combination gives it the best of both worlds. It keeps the computation efficient while still capturing enough of the sequence to understand it well. Because of this, BigBird is great at tasks involving long documents, like question answering, summarization, and genomic applications.',Nn,Oe,Es='You can find all the original BigBird checkpoints under the <a href="https://huggingface.co/google?search_models=bigbird" rel="nofollow">Google</a> organization.',Gn,Me,Rn,Ae,Ss='The example below demonstrates how to predict the <code>[MASK]</code> token with <a href="/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.Pipeline">Pipeline</a>, <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel">AutoModel</a>, and from the command line.',Vn,Be,Hn,Ye,Xn,De,Ps="<li>Inputs should be padded on the right because BigBird uses absolute position embeddings.</li> <li>BigBird supports <code>original_full</code> and <code>block_sparse</code> attention. If the input sequence length is less than 1024, it is recommended to use <code>original_full</code> since sparse patterns don’t offer much benefit for smaller inputs.</li> <li>The current implementation uses window size of 3 blocks and 2 global blocks, only supports the ITC-implementation, and doesn’t support <code>num_random_blocks=0</code>.</li> <li>The sequence length must be divisible by the block size.</li>",En,Ke,Sn,et,Qs='<li>Read the <a href="https://huggingface.co/blog/big-bird" rel="nofollow">BigBird</a> blog post for more details about how its attention works.</li>',Pn,tt,Qn,H,nt,yo,Xt,Os=`This is the configuration class to store the configuration of a <a href="/docs/transformers/v4.56.2/en/model_doc/big_bird#transformers.BigBirdModel">BigBirdModel</a>. It is used to instantiate an
BigBird model according to the specified arguments, defining the model architecture. Instantiating a configuration
with the defaults will yield a similar configuration to that of the BigBird
<a href="https://huggingface.co/google/bigbird-roberta-base" rel="nofollow">google/bigbird-roberta-base</a> architecture.`,Mo,Et,As=`Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> and can be used to control the model outputs. Read the
documentation from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> for more information.`,Bo,we,On,ot,An,F,st,wo,St,Ys='Construct a BigBird tokenizer. Based on <a href="https://github.com/google/sentencepiece" rel="nofollow">SentencePiece</a>.',vo,Pt,Ds=`This tokenizer inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer">PreTrainedTokenizer</a> which contains most of the main methods. Users should refer to
this superclass for more information regarding those methods.`,$o,de,rt,Jo,Qt,Ks=`Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
adding special tokens. A Big Bird sequence has the following format:`,Co,Ot,er="<li>single sequence: <code>[CLS] X [SEP]</code></li> <li>pair of sequences: <code>[CLS] A [SEP] B [SEP]</code></li>",Fo,ve,at,xo,At,tr=`Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
special tokens using the tokenizer <code>prepare_for_model</code> method.`,zo,le,it,jo,Yt,nr=`Create the token type IDs corresponding to the sequences passed. <a href="../glossary#token-type-ids">What are token type
IDs?</a>`,Uo,Dt,or="Should be overridden in a subclass if the model has a special way of building those.",Io,Kt,dt,Yn,lt,Dn,X,ct,qo,en,sr=`Construct a “fast” BigBird tokenizer (backed by HuggingFace’s <em>tokenizers</em> library). Based on
<a href="https://huggingface.co/docs/tokenizers/python/latest/components.html?highlight=unigram#models" rel="nofollow">Unigram</a>. This
tokenizer inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast">PreTrainedTokenizerFast</a> which contains most of the main methods. Users should refer to
this superclass for more information regarding those methods`,Wo,ce,pt,Lo,tn,rr=`Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
adding special tokens. An BigBird sequence has the following format:`,Zo,nn,ar="<li>single sequence: <code>[CLS] X [SEP]</code></li> <li>pair of sequences: <code>[CLS] A [SEP] B [SEP]</code></li>",No,$e,mt,Go,on,ir=`Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
special tokens using the tokenizer <code>prepare_for_model</code> method.`,Kn,ht,eo,ue,gt,Ro,sn,dr='Output type of <a href="/docs/transformers/v4.56.2/en/model_doc/big_bird#transformers.BigBirdForPreTraining">BigBirdForPreTraining</a>.',to,ut,no,z,ft,Vo,rn,lr="The bare Big Bird Model outputting raw hidden-states without any specific head on top.",Ho,an,cr=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Xo,dn,pr=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Eo,pe,_t,So,ln,mr='The <a href="/docs/transformers/v4.56.2/en/model_doc/big_bird#transformers.BigBirdModel">BigBirdModel</a> forward method, overrides the <code>__call__</code> special method.',Po,Je,oo,bt,so,fe,Tt,Qo,D,kt,Oo,cn,hr='The <a href="/docs/transformers/v4.56.2/en/model_doc/big_bird#transformers.BigBirdForPreTraining">BigBirdForPreTraining</a> forward method, overrides the <code>__call__</code> special method.',Ao,Ce,Yo,Fe,ro,yt,ao,j,Mt,Do,pn,gr="BigBird Model with a <code>language modeling</code> head on top for CLM fine-tuning.",Ko,mn,ur=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,es,hn,fr=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,ts,K,Bt,ns,gn,_r='The <a href="/docs/transformers/v4.56.2/en/model_doc/big_bird#transformers.BigBirdForCausalLM">BigBirdForCausalLM</a> forward method, overrides the <code>__call__</code> special method.',os,xe,ss,ze,io,wt,lo,U,vt,rs,un,br="The Big Bird Model with a <code>language modeling</code> head on top.”",as,fn,Tr=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,is,_n,kr=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,ds,N,$t,ls,bn,yr='The <a href="/docs/transformers/v4.56.2/en/model_doc/big_bird#transformers.BigBirdForMaskedLM">BigBirdForMaskedLM</a> forward method, overrides the <code>__call__</code> special method.',cs,je,ps,Ue,ms,Ie,co,Jt,po,I,Ct,hs,Tn,Mr=`BigBird Model transformer with a sequence classification/regression head on top (a linear layer on top of the
pooled output) e.g. for GLUE tasks.`,gs,kn,Br=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,us,yn,wr=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,fs,G,Ft,_s,Mn,vr='The <a href="/docs/transformers/v4.56.2/en/model_doc/big_bird#transformers.BigBirdForSequenceClassification">BigBirdForSequenceClassification</a> forward method, overrides the <code>__call__</code> special method.',bs,qe,Ts,We,ks,Le,mo,xt,ho,q,zt,ys,Bn,$r=`The Big Bird Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
softmax) e.g. for RocStories/SWAG tasks.`,Ms,wn,Jr=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Bs,vn,Cr=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,ws,ee,jt,vs,$n,Fr='The <a href="/docs/transformers/v4.56.2/en/model_doc/big_bird#transformers.BigBirdForMultipleChoice">BigBirdForMultipleChoice</a> forward method, overrides the <code>__call__</code> special method.',$s,Ze,Js,Ne,go,Ut,uo,W,It,Cs,Jn,xr=`The Big Bird transformer with a token classification head on top (a linear layer on top of the hidden-states
output) e.g. for Named-Entity-Recognition (NER) tasks.`,Fs,Cn,zr=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,xs,Fn,jr=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,zs,te,qt,js,xn,Ur='The <a href="/docs/transformers/v4.56.2/en/model_doc/big_bird#transformers.BigBirdForTokenClassification">BigBirdForTokenClassification</a> forward method, overrides the <code>__call__</code> special method.',Us,Ge,Is,Re,fo,Wt,_o,L,Lt,qs,zn,Ir=`The Big Bird transformer with a span classification head on top for extractive question-answering tasks like
SQuAD (a linear layer on top of the hidden-states output to compute <code>span start logits</code> and <code>span end logits</code>).`,Ws,jn,qr=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Ls,Un,Wr=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Zs,R,Zt,Ns,In,Lr='The <a href="/docs/transformers/v4.56.2/en/model_doc/big_bird#transformers.BigBirdForQuestionAnswering">BigBirdForQuestionAnswering</a> forward method, overrides the <code>__call__</code> special method.',Gs,Ve,Rs,He,Vs,Xe,bo,Nt,To,Wn,ko;return ie=new Z({props:{title:"BigBird",local:"bigbird",headingTag:"h1"}}),Me=new ye({props:{warning:!1,$$slots:{default:[Sr]},$$scope:{ctx:B}}}),Be=new Er({props:{id:"usage",options:["Pipeline","AutoModel","transformers CLI"],$$slots:{default:[Ar]},$$scope:{ctx:B}}}),Ye=new Z({props:{title:"Notes",local:"notes",headingTag:"h2"}}),Ke=new Z({props:{title:"Resources",local:"resources",headingTag:"h2"}}),tt=new Z({props:{title:"BigBirdConfig",local:"transformers.BigBirdConfig",headingTag:"h2"}}),nt=new $({props:{name:"class transformers.BigBirdConfig",anchor:"transformers.BigBirdConfig",parameters:[{name:"vocab_size",val:" = 50358"},{name:"hidden_size",val:" = 768"},{name:"num_hidden_layers",val:" = 12"},{name:"num_attention_heads",val:" = 12"},{name:"intermediate_size",val:" = 3072"},{name:"hidden_act",val:" = 'gelu_new'"},{name:"hidden_dropout_prob",val:" = 0.1"},{name:"attention_probs_dropout_prob",val:" = 0.1"},{name:"max_position_embeddings",val:" = 4096"},{name:"type_vocab_size",val:" = 2"},{name:"initializer_range",val:" = 0.02"},{name:"layer_norm_eps",val:" = 1e-12"},{name:"use_cache",val:" = True"},{name:"pad_token_id",val:" = 0"},{name:"bos_token_id",val:" = 1"},{name:"eos_token_id",val:" = 2"},{name:"sep_token_id",val:" = 66"},{name:"attention_type",val:" = 'block_sparse'"},{name:"use_bias",val:" = True"},{name:"rescale_embeddings",val:" = False"},{name:"block_size",val:" = 64"},{name:"num_random_blocks",val:" = 3"},{name:"classifier_dropout",val:" = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.BigBirdConfig.vocab_size",description:`<strong>vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 50358) &#x2014;
Vocabulary size of the BigBird model. Defines the number of different tokens that can be represented by the
<code>inputs_ids</code> passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/big_bird#transformers.BigBirdModel">BigBirdModel</a>.`,name:"vocab_size"},{anchor:"transformers.BigBirdConfig.hidden_size",description:`<strong>hidden_size</strong> (<code>int</code>, <em>optional</em>, defaults to 768) &#x2014;
Dimension of the encoder layers and the pooler layer.`,name:"hidden_size"},{anchor:"transformers.BigBirdConfig.num_hidden_layers",description:`<strong>num_hidden_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 12) &#x2014;
Number of hidden layers in the Transformer encoder.`,name:"num_hidden_layers"},{anchor:"transformers.BigBirdConfig.num_attention_heads",description:`<strong>num_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 12) &#x2014;
Number of attention heads for each attention layer in the Transformer encoder.`,name:"num_attention_heads"},{anchor:"transformers.BigBirdConfig.intermediate_size",description:`<strong>intermediate_size</strong> (<code>int</code>, <em>optional</em>, defaults to 3072) &#x2014;
Dimension of the &#x201C;intermediate&#x201D; (i.e., feed-forward) layer in the Transformer encoder.`,name:"intermediate_size"},{anchor:"transformers.BigBirdConfig.hidden_act",description:`<strong>hidden_act</strong> (<code>str</code> or <code>function</code>, <em>optional</em>, defaults to <code>&quot;gelu_new&quot;</code>) &#x2014;
The non-linear activation function (function or string) in the encoder and pooler. If string, <code>&quot;gelu&quot;</code>,
<code>&quot;relu&quot;</code>, <code>&quot;selu&quot;</code> and <code>&quot;gelu_new&quot;</code> are supported.`,name:"hidden_act"},{anchor:"transformers.BigBirdConfig.hidden_dropout_prob",description:`<strong>hidden_dropout_prob</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.`,name:"hidden_dropout_prob"},{anchor:"transformers.BigBirdConfig.attention_probs_dropout_prob",description:`<strong>attention_probs_dropout_prob</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout ratio for the attention probabilities.`,name:"attention_probs_dropout_prob"},{anchor:"transformers.BigBirdConfig.max_position_embeddings",description:`<strong>max_position_embeddings</strong> (<code>int</code>, <em>optional</em>, defaults to 4096) &#x2014;
The maximum sequence length that this model might ever be used with. Typically set this to something large
just in case (e.g., 1024 or 2048 or 4096).`,name:"max_position_embeddings"},{anchor:"transformers.BigBirdConfig.type_vocab_size",description:`<strong>type_vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 2) &#x2014;
The vocabulary size of the <code>token_type_ids</code> passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/big_bird#transformers.BigBirdModel">BigBirdModel</a>.`,name:"type_vocab_size"},{anchor:"transformers.BigBirdConfig.initializer_range",description:`<strong>initializer_range</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.`,name:"initializer_range"},{anchor:"transformers.BigBirdConfig.layer_norm_eps",description:`<strong>layer_norm_eps</strong> (<code>float</code>, <em>optional</em>, defaults to 1e-12) &#x2014;
The epsilon used by the layer normalization layers.`,name:"layer_norm_eps"},{anchor:"transformers.BigBirdConfig.is_decoder",description:`<strong>is_decoder</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether the model is used as a decoder or not. If <code>False</code>, the model is used as an encoder.`,name:"is_decoder"},{anchor:"transformers.BigBirdConfig.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not the model should return the last key/values attentions (not used by all models). Only
relevant if <code>config.is_decoder=True</code>.`,name:"use_cache"},{anchor:"transformers.BigBirdConfig.attention_type",description:`<strong>attention_type</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;block_sparse&quot;</code>) &#x2014;
Whether to use block sparse attention (with n complexity) as introduced in paper or original attention
layer (with n^2 complexity). Possible values are <code>&quot;original_full&quot;</code> and <code>&quot;block_sparse&quot;</code>.`,name:"attention_type"},{anchor:"transformers.BigBirdConfig.use_bias",description:`<strong>use_bias</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether to use bias in query, key, value.`,name:"use_bias"},{anchor:"transformers.BigBirdConfig.rescale_embeddings",description:`<strong>rescale_embeddings</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether to rescale embeddings with (hidden_size ** 0.5).`,name:"rescale_embeddings"},{anchor:"transformers.BigBirdConfig.block_size",description:`<strong>block_size</strong> (<code>int</code>, <em>optional</em>, defaults to 64) &#x2014;
Size of each block. Useful only when <code>attention_type == &quot;block_sparse&quot;</code>.`,name:"block_size"},{anchor:"transformers.BigBirdConfig.num_random_blocks",description:`<strong>num_random_blocks</strong> (<code>int</code>, <em>optional</em>, defaults to 3) &#x2014;
Each query is going to attend these many number of random blocks. Useful only when <code>attention_type == &quot;block_sparse&quot;</code>.`,name:"num_random_blocks"},{anchor:"transformers.BigBirdConfig.classifier_dropout",description:`<strong>classifier_dropout</strong> (<code>float</code>, <em>optional</em>) &#x2014;
The dropout ratio for the classification head.`,name:"classifier_dropout"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/big_bird/configuration_big_bird.py#L28"}}),we=new ae({props:{anchor:"transformers.BigBirdConfig.example",$$slots:{default:[Yr]},$$scope:{ctx:B}}}),ot=new Z({props:{title:"BigBirdTokenizer",local:"transformers.BigBirdTokenizer",headingTag:"h2"}}),st=new $({props:{name:"class transformers.BigBirdTokenizer",anchor:"transformers.BigBirdTokenizer",parameters:[{name:"vocab_file",val:""},{name:"unk_token",val:" = '<unk>'"},{name:"bos_token",val:" = '<s>'"},{name:"eos_token",val:" = '</s>'"},{name:"pad_token",val:" = '<pad>'"},{name:"sep_token",val:" = '[SEP]'"},{name:"mask_token",val:" = '[MASK]'"},{name:"cls_token",val:" = '[CLS]'"},{name:"sp_model_kwargs",val:": typing.Optional[dict[str, typing.Any]] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.BigBirdTokenizer.vocab_file",description:`<strong>vocab_file</strong> (<code>str</code>) &#x2014;
<a href="https://github.com/google/sentencepiece" rel="nofollow">SentencePiece</a> file (generally has a <em>.spm</em> extension) that
contains the vocabulary necessary to instantiate a tokenizer.`,name:"vocab_file"},{anchor:"transformers.BigBirdTokenizer.unk_token",description:`<strong>unk_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;unk&gt;&quot;</code>) &#x2014;
The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
token instead.`,name:"unk_token"},{anchor:"transformers.BigBirdTokenizer.bos_token",description:`<strong>bos_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;s&gt;&quot;</code>) &#x2014;
The begin of sequence token.`,name:"bos_token"},{anchor:"transformers.BigBirdTokenizer.eos_token",description:`<strong>eos_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;/s&gt;&quot;</code>) &#x2014;
The end of sequence token.`,name:"eos_token"},{anchor:"transformers.BigBirdTokenizer.pad_token",description:`<strong>pad_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;pad&gt;&quot;</code>) &#x2014;
The token used for padding, for example when batching sequences of different lengths.`,name:"pad_token"},{anchor:"transformers.BigBirdTokenizer.sep_token",description:`<strong>sep_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;[SEP]&quot;</code>) &#x2014;
The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
sequence classification or for a text and a question for question answering. It is also used as the last
token of a sequence built with special tokens.`,name:"sep_token"},{anchor:"transformers.BigBirdTokenizer.mask_token",description:`<strong>mask_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;[MASK]&quot;</code>) &#x2014;
The token used for masking values. This is the token used when training this model with masked language
modeling. This is the token which the model will try to predict.`,name:"mask_token"},{anchor:"transformers.BigBirdTokenizer.cls_token",description:`<strong>cls_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;[CLS]&quot;</code>) &#x2014;
The classifier token which is used when doing sequence classification (classification of the whole sequence
instead of per-token classification). It is the first token of the sequence when built with special tokens.`,name:"cls_token"},{anchor:"transformers.BigBirdTokenizer.sp_model_kwargs",description:`<strong>sp_model_kwargs</strong> (<code>dict</code>, <em>optional</em>) &#x2014;
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
</ul>`,name:"sp_model_kwargs"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/big_bird/tokenization_big_bird.py#L35"}}),rt=new $({props:{name:"build_inputs_with_special_tokens",anchor:"transformers.BigBirdTokenizer.build_inputs_with_special_tokens",parameters:[{name:"token_ids_0",val:": list"},{name:"token_ids_1",val:": typing.Optional[list[int]] = None"}],parametersDescription:[{anchor:"transformers.BigBirdTokenizer.build_inputs_with_special_tokens.token_ids_0",description:`<strong>token_ids_0</strong> (<code>List[int]</code>) &#x2014;
List of IDs to which the special tokens will be added.`,name:"token_ids_0"},{anchor:"transformers.BigBirdTokenizer.build_inputs_with_special_tokens.token_ids_1",description:`<strong>token_ids_1</strong> (<code>List[int]</code>, <em>optional</em>) &#x2014;
Optional second list of IDs for sequence pairs.`,name:"token_ids_1"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/big_bird/tokenization_big_bird.py#L250",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>List of <a href="../glossary#input-ids">input IDs</a> with the appropriate special tokens.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>List[int]</code></p>
`}}),at=new $({props:{name:"get_special_tokens_mask",anchor:"transformers.BigBirdTokenizer.get_special_tokens_mask",parameters:[{name:"token_ids_0",val:": list"},{name:"token_ids_1",val:": typing.Optional[list[int]] = None"},{name:"already_has_special_tokens",val:": bool = False"}],parametersDescription:[{anchor:"transformers.BigBirdTokenizer.get_special_tokens_mask.token_ids_0",description:`<strong>token_ids_0</strong> (<code>List[int]</code>) &#x2014;
List of IDs.`,name:"token_ids_0"},{anchor:"transformers.BigBirdTokenizer.get_special_tokens_mask.token_ids_1",description:`<strong>token_ids_1</strong> (<code>List[int]</code>, <em>optional</em>) &#x2014;
Optional second list of IDs for sequence pairs.`,name:"token_ids_1"},{anchor:"transformers.BigBirdTokenizer.get_special_tokens_mask.already_has_special_tokens",description:`<strong>already_has_special_tokens</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not the token list is already formatted with special tokens for the model.`,name:"already_has_special_tokens"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/big_bird/tokenization_big_bird.py#L275",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>List[int]</code></p>
`}}),it=new $({props:{name:"create_token_type_ids_from_sequences",anchor:"transformers.BigBirdTokenizer.create_token_type_ids_from_sequences",parameters:[{name:"token_ids_0",val:": list"},{name:"token_ids_1",val:": typing.Optional[list[int]] = None"}],parametersDescription:[{anchor:"transformers.BigBirdTokenizer.create_token_type_ids_from_sequences.token_ids_0",description:"<strong>token_ids_0</strong> (<code>list[int]</code>) &#x2014; The first tokenized sequence.",name:"token_ids_0"},{anchor:"transformers.BigBirdTokenizer.create_token_type_ids_from_sequences.token_ids_1",description:"<strong>token_ids_1</strong> (<code>list[int]</code>, <em>optional</em>) &#x2014; The second tokenized sequence.",name:"token_ids_1"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L3432",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>The token type ids.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>list[int]</code></p>
`}}),dt=new $({props:{name:"save_vocabulary",anchor:"transformers.BigBirdTokenizer.save_vocabulary",parameters:[{name:"save_directory",val:": str"},{name:"filename_prefix",val:": typing.Optional[str] = None"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/big_bird/tokenization_big_bird.py#L233"}}),lt=new Z({props:{title:"BigBirdTokenizerFast",local:"transformers.BigBirdTokenizerFast",headingTag:"h2"}}),ct=new $({props:{name:"class transformers.BigBirdTokenizerFast",anchor:"transformers.BigBirdTokenizerFast",parameters:[{name:"vocab_file",val:" = None"},{name:"tokenizer_file",val:" = None"},{name:"unk_token",val:" = '<unk>'"},{name:"bos_token",val:" = '<s>'"},{name:"eos_token",val:" = '</s>'"},{name:"pad_token",val:" = '<pad>'"},{name:"sep_token",val:" = '[SEP]'"},{name:"mask_token",val:" = '[MASK]'"},{name:"cls_token",val:" = '[CLS]'"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.BigBirdTokenizerFast.vocab_file",description:`<strong>vocab_file</strong> (<code>str</code>) &#x2014;
<a href="https://github.com/google/sentencepiece" rel="nofollow">SentencePiece</a> file (generally has a <em>.spm</em> extension) that
contains the vocabulary necessary to instantiate a tokenizer.`,name:"vocab_file"},{anchor:"transformers.BigBirdTokenizerFast.bos_token",description:`<strong>bos_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;s&gt;&quot;</code>) &#x2014;
The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.</p>
<div class="course-tip  bg-gradient-to-br dark:bg-gradient-to-r before:border-green-500 dark:before:border-green-800 from-green-50 dark:from-gray-900 to-white dark:to-gray-950 border border-green-50 text-green-700 dark:text-gray-400">
						
<p>When building a sequence using special tokens, this is not the token that is used for the beginning of
sequence. The token used is the <code>cls_token</code>.</p>

					</div>`,name:"bos_token"},{anchor:"transformers.BigBirdTokenizerFast.eos_token",description:`<strong>eos_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;/s&gt;&quot;</code>) &#x2014;
The end of sequence token. .. note:: When building a sequence using special tokens, this is not the token
that is used for the end of sequence. The token used is the <code>sep_token</code>.`,name:"eos_token"},{anchor:"transformers.BigBirdTokenizerFast.unk_token",description:`<strong>unk_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;unk&gt;&quot;</code>) &#x2014;
The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
token instead.`,name:"unk_token"},{anchor:"transformers.BigBirdTokenizerFast.sep_token",description:`<strong>sep_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;[SEP]&quot;</code>) &#x2014;
The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
sequence classification or for a text and a question for question answering. It is also used as the last
token of a sequence built with special tokens.`,name:"sep_token"},{anchor:"transformers.BigBirdTokenizerFast.pad_token",description:`<strong>pad_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;pad&gt;&quot;</code>) &#x2014;
The token used for padding, for example when batching sequences of different lengths.`,name:"pad_token"},{anchor:"transformers.BigBirdTokenizerFast.cls_token",description:`<strong>cls_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;[CLS]&quot;</code>) &#x2014;
The classifier token which is used when doing sequence classification (classification of the whole sequence
instead of per-token classification). It is the first token of the sequence when built with special tokens.`,name:"cls_token"},{anchor:"transformers.BigBirdTokenizerFast.mask_token",description:`<strong>mask_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;[MASK]&quot;</code>) &#x2014;
The token used for masking values. This is the token used when training this model with masked language
modeling. This is the token which the model will try to predict.`,name:"mask_token"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/big_bird/tokenization_big_bird_fast.py#L38"}}),pt=new $({props:{name:"build_inputs_with_special_tokens",anchor:"transformers.BigBirdTokenizerFast.build_inputs_with_special_tokens",parameters:[{name:"token_ids_0",val:": list"},{name:"token_ids_1",val:": typing.Optional[list[int]] = None"}],parametersDescription:[{anchor:"transformers.BigBirdTokenizerFast.build_inputs_with_special_tokens.token_ids_0",description:`<strong>token_ids_0</strong> (<code>List[int]</code>) &#x2014;
List of IDs to which the special tokens will be added`,name:"token_ids_0"},{anchor:"transformers.BigBirdTokenizerFast.build_inputs_with_special_tokens.token_ids_1",description:`<strong>token_ids_1</strong> (<code>List[int]</code>, <em>optional</em>) &#x2014;
Optional second list of IDs for sequence pairs.`,name:"token_ids_1"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/big_bird/tokenization_big_bird_fast.py#L122",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>list of <a href="../glossary#input-ids">input IDs</a> with the appropriate special tokens.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>List[int]</code></p>
`}}),mt=new $({props:{name:"get_special_tokens_mask",anchor:"transformers.BigBirdTokenizerFast.get_special_tokens_mask",parameters:[{name:"token_ids_0",val:": list"},{name:"token_ids_1",val:": typing.Optional[list[int]] = None"},{name:"already_has_special_tokens",val:": bool = False"}],parametersDescription:[{anchor:"transformers.BigBirdTokenizerFast.get_special_tokens_mask.token_ids_0",description:`<strong>token_ids_0</strong> (<code>List[int]</code>) &#x2014;
List of ids.`,name:"token_ids_0"},{anchor:"transformers.BigBirdTokenizerFast.get_special_tokens_mask.token_ids_1",description:`<strong>token_ids_1</strong> (<code>List[int]</code>, <em>optional</em>) &#x2014;
Optional second list of IDs for sequence pairs.`,name:"token_ids_1"},{anchor:"transformers.BigBirdTokenizerFast.get_special_tokens_mask.already_has_special_tokens",description:`<strong>already_has_special_tokens</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Set to True if the token list is already formatted with special tokens for the model`,name:"already_has_special_tokens"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/big_bird/tokenization_big_bird_fast.py#L147",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>List[int]</code></p>
`}}),ht=new Z({props:{title:"BigBird specific outputs",local:"transformers.models.big_bird.modeling_big_bird.BigBirdForPreTrainingOutput",headingTag:"h2"}}),gt=new $({props:{name:"class transformers.models.big_bird.modeling_big_bird.BigBirdForPreTrainingOutput",anchor:"transformers.models.big_bird.modeling_big_bird.BigBirdForPreTrainingOutput",parameters:[{name:"loss",val:": typing.Optional[torch.FloatTensor] = None"},{name:"prediction_logits",val:": typing.Optional[torch.FloatTensor] = None"},{name:"seq_relationship_logits",val:": typing.Optional[torch.FloatTensor] = None"},{name:"hidden_states",val:": typing.Optional[tuple[torch.FloatTensor]] = None"},{name:"attentions",val:": typing.Optional[tuple[torch.FloatTensor]] = None"}],parametersDescription:[{anchor:"transformers.models.big_bird.modeling_big_bird.BigBirdForPreTrainingOutput.loss",description:`<strong>loss</strong> (<code>*optional*</code>, returned when <code>labels</code> is provided, <code>torch.FloatTensor</code> of shape <code>(1,)</code>) &#x2014;
Total loss as the sum of the masked language modeling loss and the next sequence prediction
(classification) loss.`,name:"loss"},{anchor:"transformers.models.big_bird.modeling_big_bird.BigBirdForPreTrainingOutput.prediction_logits",description:`<strong>prediction_logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, config.vocab_size)</code>) &#x2014;
Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).`,name:"prediction_logits"},{anchor:"transformers.models.big_bird.modeling_big_bird.BigBirdForPreTrainingOutput.seq_relationship_logits",description:`<strong>seq_relationship_logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, 2)</code>) &#x2014;
Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation
before SoftMax).`,name:"seq_relationship_logits"},{anchor:"transformers.models.big_bird.modeling_big_bird.BigBirdForPreTrainingOutput.hidden_states",description:`<strong>hidden_states</strong> (<code>tuple[torch.FloatTensor]</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) &#x2014;
Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings, if the model has an embedding layer, +
one for the output of each layer) of shape <code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.`,name:"hidden_states"},{anchor:"transformers.models.big_bird.modeling_big_bird.BigBirdForPreTrainingOutput.attentions",description:`<strong>attentions</strong> (<code>tuple[torch.FloatTensor]</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) &#x2014;
Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
heads.`,name:"attentions"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/big_bird/modeling_big_bird.py#L1741"}}),ut=new Z({props:{title:"BigBirdModel",local:"transformers.BigBirdModel",headingTag:"h2"}}),ft=new $({props:{name:"class transformers.BigBirdModel",anchor:"transformers.BigBirdModel",parameters:[{name:"config",val:""},{name:"add_pooling_layer",val:" = True"}],parametersDescription:[{anchor:"transformers.BigBirdModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/big_bird#transformers.BigBirdModel">BigBirdModel</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"},{anchor:"transformers.BigBirdModel.add_pooling_layer",description:`<strong>add_pooling_layer</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether to add a pooling layer`,name:"add_pooling_layer"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/big_bird/modeling_big_bird.py#L1783"}}),_t=new $({props:{name:"forward",anchor:"transformers.BigBirdModel.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"encoder_hidden_states",val:": typing.Optional[torch.FloatTensor] = None"},{name:"encoder_attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"past_key_values",val:": typing.Optional[tuple[tuple[torch.FloatTensor]]] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.Tensor] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.BigBirdModel.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.BigBirdModel.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.BigBirdModel.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.BigBirdModel.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.BigBirdModel.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.BigBirdModel.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.BigBirdModel.forward.encoder_hidden_states",description:`<strong>encoder_hidden_states</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
if the model is configured as a decoder.`,name:"encoder_hidden_states"},{anchor:"transformers.BigBirdModel.forward.encoder_attention_mask",description:`<strong>encoder_attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
the cross-attention if the model is configured as a decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>`,name:"encoder_attention_mask"},{anchor:"transformers.BigBirdModel.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>tuple[tuple[torch.FloatTensor]]</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.BigBirdModel.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.BigBirdModel.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.BigBirdModel.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.BigBirdModel.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.BigBirdModel.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.Tensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/big_bird/modeling_big_bird.py#L1844",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions"
>transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/big_bird#transformers.BigBirdConfig"
>BigBirdConfig</a>) and inputs.</p>
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
`}}),Je=new ye({props:{$$slots:{default:[Dr]},$$scope:{ctx:B}}}),bt=new Z({props:{title:"BigBirdForPreTraining",local:"transformers.BigBirdForPreTraining",headingTag:"h2"}}),Tt=new $({props:{name:"class transformers.BigBirdForPreTraining",anchor:"transformers.BigBirdForPreTraining",parameters:[{name:"config",val:""}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/big_bird/modeling_big_bird.py#L2110"}}),kt=new $({props:{name:"forward",anchor:"transformers.BigBirdForPreTraining.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.FloatTensor] = None"},{name:"next_sentence_label",val:": typing.Optional[torch.LongTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.BigBirdForPreTraining.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.BigBirdForPreTraining.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.BigBirdForPreTraining.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.BigBirdForPreTraining.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.BigBirdForPreTraining.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.BigBirdForPreTraining.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.BigBirdForPreTraining.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should be in <code>[-100, 0, ..., config.vocab_size]</code> (see <code>input_ids</code> docstring) Tokens with indices set to <code>-100</code> are ignored (masked), the
loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>`,name:"labels"},{anchor:"transformers.BigBirdForPreTraining.forward.next_sentence_label",description:`<strong>next_sentence_label</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the next sequence prediction (classification) loss. If specified, nsp loss will be
added to masked_lm loss. Input should be a sequence pair (see <code>input_ids</code> docstring) Indices should be in
<code>[0, 1]</code>:</p>
<ul>
<li>0 indicates sequence B is a continuation of sequence A,</li>
<li>1 indicates sequence B is a random sequence.</li>
</ul>`,name:"next_sentence_label"},{anchor:"transformers.BigBirdForPreTraining.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.BigBirdForPreTraining.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.BigBirdForPreTraining.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/big_bird/modeling_big_bird.py#L2129",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/model_doc/big_bird#transformers.models.big_bird.modeling_big_bird.BigBirdForPreTrainingOutput"
>transformers.models.big_bird.modeling_big_bird.BigBirdForPreTrainingOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/big_bird#transformers.BigBirdConfig"
>BigBirdConfig</a>) and inputs.</p>
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
  href="/docs/transformers/v4.56.2/en/model_doc/big_bird#transformers.models.big_bird.modeling_big_bird.BigBirdForPreTrainingOutput"
>transformers.models.big_bird.modeling_big_bird.BigBirdForPreTrainingOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),Ce=new ye({props:{$$slots:{default:[Kr]},$$scope:{ctx:B}}}),Fe=new ae({props:{anchor:"transformers.BigBirdForPreTraining.forward.example",$$slots:{default:[ea]},$$scope:{ctx:B}}}),yt=new Z({props:{title:"BigBirdForCausalLM",local:"transformers.BigBirdForCausalLM",headingTag:"h2"}}),Mt=new $({props:{name:"class transformers.BigBirdForCausalLM",anchor:"transformers.BigBirdForCausalLM",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.BigBirdForCausalLM.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/big_bird#transformers.BigBirdForCausalLM">BigBirdForCausalLM</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/big_bird/modeling_big_bird.py#L2356"}}),Bt=new $({props:{name:"forward",anchor:"transformers.BigBirdForCausalLM.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"encoder_hidden_states",val:": typing.Optional[torch.FloatTensor] = None"},{name:"encoder_attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"past_key_values",val:": typing.Optional[tuple[tuple[torch.FloatTensor]]] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.Tensor] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.BigBirdForCausalLM.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.BigBirdForCausalLM.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.BigBirdForCausalLM.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.BigBirdForCausalLM.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.BigBirdForCausalLM.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.BigBirdForCausalLM.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.BigBirdForCausalLM.forward.encoder_hidden_states",description:`<strong>encoder_hidden_states</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
if the model is configured as a decoder.`,name:"encoder_hidden_states"},{anchor:"transformers.BigBirdForCausalLM.forward.encoder_attention_mask",description:`<strong>encoder_attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
the cross-attention if the model is configured as a decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>`,name:"encoder_attention_mask"},{anchor:"transformers.BigBirdForCausalLM.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>tuple[tuple[torch.FloatTensor]]</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.BigBirdForCausalLM.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the left-to-right language modeling loss (next word prediction). Indices should be in
<code>[-100, 0, ..., config.vocab_size]</code> (see <code>input_ids</code> docstring) Tokens with indices set to <code>-100</code> are
ignored (masked), the loss is only computed for the tokens with labels n <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.BigBirdForCausalLM.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.BigBirdForCausalLM.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.BigBirdForCausalLM.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.BigBirdForCausalLM.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.BigBirdForCausalLM.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.Tensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/big_bird/modeling_big_bird.py#L2378",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithCrossAttentions"
>transformers.modeling_outputs.CausalLMOutputWithCrossAttentions</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/big_bird#transformers.BigBirdConfig"
>BigBirdConfig</a>) and inputs.</p>
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
`}}),xe=new ye({props:{$$slots:{default:[ta]},$$scope:{ctx:B}}}),ze=new ae({props:{anchor:"transformers.BigBirdForCausalLM.forward.example",$$slots:{default:[na]},$$scope:{ctx:B}}}),wt=new Z({props:{title:"BigBirdForMaskedLM",local:"transformers.BigBirdForMaskedLM",headingTag:"h2"}}),vt=new $({props:{name:"class transformers.BigBirdForMaskedLM",anchor:"transformers.BigBirdForMaskedLM",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.BigBirdForMaskedLM.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/big_bird#transformers.BigBirdForMaskedLM">BigBirdForMaskedLM</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/big_bird/modeling_big_bird.py#L2212"}}),$t=new $({props:{name:"forward",anchor:"transformers.BigBirdForMaskedLM.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"encoder_hidden_states",val:": typing.Optional[torch.FloatTensor] = None"},{name:"encoder_attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.BigBirdForMaskedLM.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.BigBirdForMaskedLM.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.BigBirdForMaskedLM.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.BigBirdForMaskedLM.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.BigBirdForMaskedLM.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.BigBirdForMaskedLM.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.BigBirdForMaskedLM.forward.encoder_hidden_states",description:`<strong>encoder_hidden_states</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
if the model is configured as a decoder.`,name:"encoder_hidden_states"},{anchor:"transformers.BigBirdForMaskedLM.forward.encoder_attention_mask",description:`<strong>encoder_attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
the cross-attention if the model is configured as a decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>`,name:"encoder_attention_mask"},{anchor:"transformers.BigBirdForMaskedLM.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should be in <code>[-100, 0, ..., config.vocab_size]</code> (see <code>input_ids</code> docstring) Tokens with indices set to <code>-100</code> are ignored (masked), the
loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.BigBirdForMaskedLM.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.BigBirdForMaskedLM.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.BigBirdForMaskedLM.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/big_bird/modeling_big_bird.py#L2237",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.MaskedLMOutput"
>transformers.modeling_outputs.MaskedLMOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/big_bird#transformers.BigBirdConfig"
>BigBirdConfig</a>) and inputs.</p>
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
`}}),je=new ye({props:{$$slots:{default:[oa]},$$scope:{ctx:B}}}),Ue=new ae({props:{anchor:"transformers.BigBirdForMaskedLM.forward.example",$$slots:{default:[sa]},$$scope:{ctx:B}}}),Ie=new ae({props:{anchor:"transformers.BigBirdForMaskedLM.forward.example-2",$$slots:{default:[ra]},$$scope:{ctx:B}}}),Jt=new Z({props:{title:"BigBirdForSequenceClassification",local:"transformers.BigBirdForSequenceClassification",headingTag:"h2"}}),Ct=new $({props:{name:"class transformers.BigBirdForSequenceClassification",anchor:"transformers.BigBirdForSequenceClassification",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.BigBirdForSequenceClassification.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/big_bird#transformers.BigBirdForSequenceClassification">BigBirdForSequenceClassification</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/big_bird/modeling_big_bird.py#L2480"}}),Ft=new $({props:{name:"forward",anchor:"transformers.BigBirdForSequenceClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.BigBirdForSequenceClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.BigBirdForSequenceClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.BigBirdForSequenceClassification.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.BigBirdForSequenceClassification.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.BigBirdForSequenceClassification.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.BigBirdForSequenceClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.BigBirdForSequenceClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the sequence classification/regression loss. Indices should be in <code>[0, ..., config.num_labels - 1]</code>. If <code>config.num_labels == 1</code> a regression loss is computed (Mean-Square loss), If
<code>config.num_labels &gt; 1</code> a classification loss is computed (Cross-Entropy).`,name:"labels"},{anchor:"transformers.BigBirdForSequenceClassification.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.BigBirdForSequenceClassification.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.BigBirdForSequenceClassification.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/big_bird/modeling_big_bird.py#L2491",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput"
>transformers.modeling_outputs.SequenceClassifierOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/big_bird#transformers.BigBirdConfig"
>BigBirdConfig</a>) and inputs.</p>
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
`}}),qe=new ye({props:{$$slots:{default:[aa]},$$scope:{ctx:B}}}),We=new ae({props:{anchor:"transformers.BigBirdForSequenceClassification.forward.example",$$slots:{default:[ia]},$$scope:{ctx:B}}}),Le=new ae({props:{anchor:"transformers.BigBirdForSequenceClassification.forward.example-2",$$slots:{default:[da]},$$scope:{ctx:B}}}),xt=new Z({props:{title:"BigBirdForMultipleChoice",local:"transformers.BigBirdForMultipleChoice",headingTag:"h2"}}),zt=new $({props:{name:"class transformers.BigBirdForMultipleChoice",anchor:"transformers.BigBirdForMultipleChoice",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.BigBirdForMultipleChoice.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/big_bird#transformers.BigBirdForMultipleChoice">BigBirdForMultipleChoice</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/big_bird/modeling_big_bird.py#L2599"}}),jt=new $({props:{name:"forward",anchor:"transformers.BigBirdForMultipleChoice.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.BigBirdForMultipleChoice.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, num_choices, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.BigBirdForMultipleChoice.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.BigBirdForMultipleChoice.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, num_choices, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.BigBirdForMultipleChoice.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, num_choices, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.max_position_embeddings - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.BigBirdForMultipleChoice.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.BigBirdForMultipleChoice.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_choices, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <em>input_ids</em> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.BigBirdForMultipleChoice.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the multiple choice classification loss. Indices should be in <code>[0, ..., num_choices-1]</code> where <code>num_choices</code> is the size of the second dimension of the input tensors. (See
<code>input_ids</code> above)`,name:"labels"},{anchor:"transformers.BigBirdForMultipleChoice.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.BigBirdForMultipleChoice.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.BigBirdForMultipleChoice.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/big_bird/modeling_big_bird.py#L2610",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.MultipleChoiceModelOutput"
>transformers.modeling_outputs.MultipleChoiceModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/big_bird#transformers.BigBirdConfig"
>BigBirdConfig</a>) and inputs.</p>
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
`}}),Ze=new ye({props:{$$slots:{default:[la]},$$scope:{ctx:B}}}),Ne=new ae({props:{anchor:"transformers.BigBirdForMultipleChoice.forward.example",$$slots:{default:[ca]},$$scope:{ctx:B}}}),Ut=new Z({props:{title:"BigBirdForTokenClassification",local:"transformers.BigBirdForTokenClassification",headingTag:"h2"}}),It=new $({props:{name:"class transformers.BigBirdForTokenClassification",anchor:"transformers.BigBirdForTokenClassification",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.BigBirdForTokenClassification.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/big_bird#transformers.BigBirdForTokenClassification">BigBirdForTokenClassification</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/big_bird/modeling_big_bird.py#L2703"}}),qt=new $({props:{name:"forward",anchor:"transformers.BigBirdForTokenClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.BigBirdForTokenClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.BigBirdForTokenClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.BigBirdForTokenClassification.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.BigBirdForTokenClassification.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.BigBirdForTokenClassification.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.BigBirdForTokenClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.BigBirdForTokenClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the token classification loss. Indices should be in <code>[0, ..., config.num_labels - 1]</code>.`,name:"labels"},{anchor:"transformers.BigBirdForTokenClassification.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.BigBirdForTokenClassification.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.BigBirdForTokenClassification.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/big_bird/modeling_big_bird.py#L2718",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput"
>transformers.modeling_outputs.TokenClassifierOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/big_bird#transformers.BigBirdConfig"
>BigBirdConfig</a>) and inputs.</p>
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
`}}),Ge=new ye({props:{$$slots:{default:[pa]},$$scope:{ctx:B}}}),Re=new ae({props:{anchor:"transformers.BigBirdForTokenClassification.forward.example",$$slots:{default:[ma]},$$scope:{ctx:B}}}),Wt=new Z({props:{title:"BigBirdForQuestionAnswering",local:"transformers.BigBirdForQuestionAnswering",headingTag:"h2"}}),Lt=new $({props:{name:"class transformers.BigBirdForQuestionAnswering",anchor:"transformers.BigBirdForQuestionAnswering",parameters:[{name:"config",val:""},{name:"add_pooling_layer",val:" = False"}],parametersDescription:[{anchor:"transformers.BigBirdForQuestionAnswering.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/big_bird#transformers.BigBirdForQuestionAnswering">BigBirdForQuestionAnswering</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"},{anchor:"transformers.BigBirdForQuestionAnswering.add_pooling_layer",description:`<strong>add_pooling_layer</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether to add a pooling layer`,name:"add_pooling_layer"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/big_bird/modeling_big_bird.py#L2791"}}),Zt=new $({props:{name:"forward",anchor:"transformers.BigBirdForQuestionAnswering.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"question_lengths",val:": typing.Optional[torch.LongTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"start_positions",val:": typing.Optional[torch.LongTensor] = None"},{name:"end_positions",val:": typing.Optional[torch.LongTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.BigBirdForQuestionAnswering.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.BigBirdForQuestionAnswering.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.BigBirdForQuestionAnswering.forward.question_lengths",description:`<strong>question_lengths</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, 1)</code>, <em>optional</em>) &#x2014;
The lengths of the questions in the batch.`,name:"question_lengths"},{anchor:"transformers.BigBirdForQuestionAnswering.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.BigBirdForQuestionAnswering.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.BigBirdForQuestionAnswering.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.BigBirdForQuestionAnswering.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.BigBirdForQuestionAnswering.forward.start_positions",description:`<strong>start_positions</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for position (index) of the start of the labelled span for computing the token classification loss.
Positions are clamped to the length of the sequence (<code>sequence_length</code>). Position outside of the sequence
are not taken into account for computing the loss.`,name:"start_positions"},{anchor:"transformers.BigBirdForQuestionAnswering.forward.end_positions",description:`<strong>end_positions</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for position (index) of the end of the labelled span for computing the token classification loss.
Positions are clamped to the length of the sequence (<code>sequence_length</code>). Position outside of the sequence
are not taken into account for computing the loss.`,name:"end_positions"},{anchor:"transformers.BigBirdForQuestionAnswering.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.BigBirdForQuestionAnswering.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.BigBirdForQuestionAnswering.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/big_bird/modeling_big_bird.py#L2809",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <code>transformers.models.big_bird.modeling_big_bird.BigBirdForQuestionAnsweringModelOutput</code> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/big_bird#transformers.BigBirdConfig"
>BigBirdConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>loss</strong> (<code>torch.FloatTensor</code> of shape <code>(1,)</code>, <em>optional</em>, returned when <code>labels</code> is provided) — Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.</p>
</li>
<li>
<p><strong>start_logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>, defaults to <code>None</code>) — Span-start scores (before SoftMax).</p>
</li>
<li>
<p><strong>end_logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>, defaults to <code>None</code>) — Span-end scores (before SoftMax).</p>
</li>
<li>
<p><strong>pooler_output</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, 1)</code>) — pooler output from BigBigModel</p>
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


<p><code>transformers.models.big_bird.modeling_big_bird.BigBirdForQuestionAnsweringModelOutput</code> or <code>tuple(torch.FloatTensor)</code></p>
`}}),Ve=new ye({props:{$$slots:{default:[ha]},$$scope:{ctx:B}}}),He=new ae({props:{anchor:"transformers.BigBirdForQuestionAnswering.forward.example",$$slots:{default:[ga]},$$scope:{ctx:B}}}),Xe=new ae({props:{anchor:"transformers.BigBirdForQuestionAnswering.forward.example-2",$$slots:{default:[ua]},$$scope:{ctx:B}}}),Nt=new Xr({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/big_bird.md"}}),{c(){t=c("meta"),m=r(),n=c("p"),l=r(),M=c("p"),M.innerHTML=s,y=r(),C=c("div"),C.innerHTML=qn,Pe=r(),g(ie.$$.fragment),Zn=r(),Qe=c("p"),Qe.innerHTML=Xs,Nn=r(),Oe=c("p"),Oe.innerHTML=Es,Gn=r(),g(Me.$$.fragment),Rn=r(),Ae=c("p"),Ae.innerHTML=Ss,Vn=r(),g(Be.$$.fragment),Hn=r(),g(Ye.$$.fragment),Xn=r(),De=c("ul"),De.innerHTML=Ps,En=r(),g(Ke.$$.fragment),Sn=r(),et=c("ul"),et.innerHTML=Qs,Pn=r(),g(tt.$$.fragment),Qn=r(),H=c("div"),g(nt.$$.fragment),yo=r(),Xt=c("p"),Xt.innerHTML=Os,Mo=r(),Et=c("p"),Et.innerHTML=As,Bo=r(),g(we.$$.fragment),On=r(),g(ot.$$.fragment),An=r(),F=c("div"),g(st.$$.fragment),wo=r(),St=c("p"),St.innerHTML=Ys,vo=r(),Pt=c("p"),Pt.innerHTML=Ds,$o=r(),de=c("div"),g(rt.$$.fragment),Jo=r(),Qt=c("p"),Qt.textContent=Ks,Co=r(),Ot=c("ul"),Ot.innerHTML=er,Fo=r(),ve=c("div"),g(at.$$.fragment),xo=r(),At=c("p"),At.innerHTML=tr,zo=r(),le=c("div"),g(it.$$.fragment),jo=r(),Yt=c("p"),Yt.innerHTML=nr,Uo=r(),Dt=c("p"),Dt.textContent=or,Io=r(),Kt=c("div"),g(dt.$$.fragment),Yn=r(),g(lt.$$.fragment),Dn=r(),X=c("div"),g(ct.$$.fragment),qo=r(),en=c("p"),en.innerHTML=sr,Wo=r(),ce=c("div"),g(pt.$$.fragment),Lo=r(),tn=c("p"),tn.textContent=rr,Zo=r(),nn=c("ul"),nn.innerHTML=ar,No=r(),$e=c("div"),g(mt.$$.fragment),Go=r(),on=c("p"),on.innerHTML=ir,Kn=r(),g(ht.$$.fragment),eo=r(),ue=c("div"),g(gt.$$.fragment),Ro=r(),sn=c("p"),sn.innerHTML=dr,to=r(),g(ut.$$.fragment),no=r(),z=c("div"),g(ft.$$.fragment),Vo=r(),rn=c("p"),rn.textContent=lr,Ho=r(),an=c("p"),an.innerHTML=cr,Xo=r(),dn=c("p"),dn.innerHTML=pr,Eo=r(),pe=c("div"),g(_t.$$.fragment),So=r(),ln=c("p"),ln.innerHTML=mr,Po=r(),g(Je.$$.fragment),oo=r(),g(bt.$$.fragment),so=r(),fe=c("div"),g(Tt.$$.fragment),Qo=r(),D=c("div"),g(kt.$$.fragment),Oo=r(),cn=c("p"),cn.innerHTML=hr,Ao=r(),g(Ce.$$.fragment),Yo=r(),g(Fe.$$.fragment),ro=r(),g(yt.$$.fragment),ao=r(),j=c("div"),g(Mt.$$.fragment),Do=r(),pn=c("p"),pn.innerHTML=gr,Ko=r(),mn=c("p"),mn.innerHTML=ur,es=r(),hn=c("p"),hn.innerHTML=fr,ts=r(),K=c("div"),g(Bt.$$.fragment),ns=r(),gn=c("p"),gn.innerHTML=_r,os=r(),g(xe.$$.fragment),ss=r(),g(ze.$$.fragment),io=r(),g(wt.$$.fragment),lo=r(),U=c("div"),g(vt.$$.fragment),rs=r(),un=c("p"),un.innerHTML=br,as=r(),fn=c("p"),fn.innerHTML=Tr,is=r(),_n=c("p"),_n.innerHTML=kr,ds=r(),N=c("div"),g($t.$$.fragment),ls=r(),bn=c("p"),bn.innerHTML=yr,cs=r(),g(je.$$.fragment),ps=r(),g(Ue.$$.fragment),ms=r(),g(Ie.$$.fragment),co=r(),g(Jt.$$.fragment),po=r(),I=c("div"),g(Ct.$$.fragment),hs=r(),Tn=c("p"),Tn.textContent=Mr,gs=r(),kn=c("p"),kn.innerHTML=Br,us=r(),yn=c("p"),yn.innerHTML=wr,fs=r(),G=c("div"),g(Ft.$$.fragment),_s=r(),Mn=c("p"),Mn.innerHTML=vr,bs=r(),g(qe.$$.fragment),Ts=r(),g(We.$$.fragment),ks=r(),g(Le.$$.fragment),mo=r(),g(xt.$$.fragment),ho=r(),q=c("div"),g(zt.$$.fragment),ys=r(),Bn=c("p"),Bn.textContent=$r,Ms=r(),wn=c("p"),wn.innerHTML=Jr,Bs=r(),vn=c("p"),vn.innerHTML=Cr,ws=r(),ee=c("div"),g(jt.$$.fragment),vs=r(),$n=c("p"),$n.innerHTML=Fr,$s=r(),g(Ze.$$.fragment),Js=r(),g(Ne.$$.fragment),go=r(),g(Ut.$$.fragment),uo=r(),W=c("div"),g(It.$$.fragment),Cs=r(),Jn=c("p"),Jn.textContent=xr,Fs=r(),Cn=c("p"),Cn.innerHTML=zr,xs=r(),Fn=c("p"),Fn.innerHTML=jr,zs=r(),te=c("div"),g(qt.$$.fragment),js=r(),xn=c("p"),xn.innerHTML=Ur,Us=r(),g(Ge.$$.fragment),Is=r(),g(Re.$$.fragment),fo=r(),g(Wt.$$.fragment),_o=r(),L=c("div"),g(Lt.$$.fragment),qs=r(),zn=c("p"),zn.innerHTML=Ir,Ws=r(),jn=c("p"),jn.innerHTML=qr,Ls=r(),Un=c("p"),Un.innerHTML=Wr,Zs=r(),R=c("div"),g(Zt.$$.fragment),Ns=r(),In=c("p"),In.innerHTML=Lr,Gs=r(),g(Ve.$$.fragment),Rs=r(),g(He.$$.fragment),Vs=r(),g(Xe.$$.fragment),bo=r(),g(Nt.$$.fragment),To=r(),Wn=c("p"),this.h()},l(e){const i=Vr("svelte-u9bgzb",document.head);t=p(i,"META",{name:!0,content:!0}),i.forEach(d),m=a(e),n=p(e,"P",{}),w(n).forEach(d),l=a(e),M=p(e,"P",{"data-svelte-h":!0}),k(M)!=="svelte-j3fq5a"&&(M.innerHTML=s),y=a(e),C=p(e,"DIV",{style:!0,"data-svelte-h":!0}),k(C)!=="svelte-wa5t4p"&&(C.innerHTML=qn),Pe=a(e),u(ie.$$.fragment,e),Zn=a(e),Qe=p(e,"P",{"data-svelte-h":!0}),k(Qe)!=="svelte-1sjgcug"&&(Qe.innerHTML=Xs),Nn=a(e),Oe=p(e,"P",{"data-svelte-h":!0}),k(Oe)!=="svelte-150d6hg"&&(Oe.innerHTML=Es),Gn=a(e),u(Me.$$.fragment,e),Rn=a(e),Ae=p(e,"P",{"data-svelte-h":!0}),k(Ae)!=="svelte-lqa8w5"&&(Ae.innerHTML=Ss),Vn=a(e),u(Be.$$.fragment,e),Hn=a(e),u(Ye.$$.fragment,e),Xn=a(e),De=p(e,"UL",{"data-svelte-h":!0}),k(De)!=="svelte-n9w7hv"&&(De.innerHTML=Ps),En=a(e),u(Ke.$$.fragment,e),Sn=a(e),et=p(e,"UL",{"data-svelte-h":!0}),k(et)!=="svelte-14a2nwi"&&(et.innerHTML=Qs),Pn=a(e),u(tt.$$.fragment,e),Qn=a(e),H=p(e,"DIV",{class:!0});var ne=w(H);u(nt.$$.fragment,ne),yo=a(ne),Xt=p(ne,"P",{"data-svelte-h":!0}),k(Xt)!=="svelte-fur8g9"&&(Xt.innerHTML=Os),Mo=a(ne),Et=p(ne,"P",{"data-svelte-h":!0}),k(Et)!=="svelte-1ek1ss9"&&(Et.innerHTML=As),Bo=a(ne),u(we.$$.fragment,ne),ne.forEach(d),On=a(e),u(ot.$$.fragment,e),An=a(e),F=p(e,"DIV",{class:!0});var x=w(F);u(st.$$.fragment,x),wo=a(x),St=p(x,"P",{"data-svelte-h":!0}),k(St)!=="svelte-icdx2b"&&(St.innerHTML=Ys),vo=a(x),Pt=p(x,"P",{"data-svelte-h":!0}),k(Pt)!=="svelte-ntrhio"&&(Pt.innerHTML=Ds),$o=a(x),de=p(x,"DIV",{class:!0});var _e=w(de);u(rt.$$.fragment,_e),Jo=a(_e),Qt=p(_e,"P",{"data-svelte-h":!0}),k(Qt)!=="svelte-1wtdd6g"&&(Qt.textContent=Ks),Co=a(_e),Ot=p(_e,"UL",{"data-svelte-h":!0}),k(Ot)!=="svelte-xi6653"&&(Ot.innerHTML=er),_e.forEach(d),Fo=a(x),ve=p(x,"DIV",{class:!0});var Gt=w(ve);u(at.$$.fragment,Gt),xo=a(Gt),At=p(Gt,"P",{"data-svelte-h":!0}),k(At)!=="svelte-1f4f5kp"&&(At.innerHTML=tr),Gt.forEach(d),zo=a(x),le=p(x,"DIV",{class:!0});var be=w(le);u(it.$$.fragment,be),jo=a(be),Yt=p(be,"P",{"data-svelte-h":!0}),k(Yt)!=="svelte-zj1vf1"&&(Yt.innerHTML=nr),Uo=a(be),Dt=p(be,"P",{"data-svelte-h":!0}),k(Dt)!=="svelte-9vptpw"&&(Dt.textContent=or),be.forEach(d),Io=a(x),Kt=p(x,"DIV",{class:!0});var Ln=w(Kt);u(dt.$$.fragment,Ln),Ln.forEach(d),x.forEach(d),Yn=a(e),u(lt.$$.fragment,e),Dn=a(e),X=p(e,"DIV",{class:!0});var oe=w(X);u(ct.$$.fragment,oe),qo=a(oe),en=p(oe,"P",{"data-svelte-h":!0}),k(en)!=="svelte-1r6gfpg"&&(en.innerHTML=sr),Wo=a(oe),ce=p(oe,"DIV",{class:!0});var Te=w(ce);u(pt.$$.fragment,Te),Lo=a(Te),tn=p(Te,"P",{"data-svelte-h":!0}),k(tn)!=="svelte-1of3j8a"&&(tn.textContent=rr),Zo=a(Te),nn=p(Te,"UL",{"data-svelte-h":!0}),k(nn)!=="svelte-xi6653"&&(nn.innerHTML=ar),Te.forEach(d),No=a(oe),$e=p(oe,"DIV",{class:!0});var Rt=w($e);u(mt.$$.fragment,Rt),Go=a(Rt),on=p(Rt,"P",{"data-svelte-h":!0}),k(on)!=="svelte-1l6i2y2"&&(on.innerHTML=ir),Rt.forEach(d),oe.forEach(d),Kn=a(e),u(ht.$$.fragment,e),eo=a(e),ue=p(e,"DIV",{class:!0});var Vt=w(ue);u(gt.$$.fragment,Vt),Ro=a(Vt),sn=p(Vt,"P",{"data-svelte-h":!0}),k(sn)!=="svelte-zzjrll"&&(sn.innerHTML=dr),Vt.forEach(d),to=a(e),u(ut.$$.fragment,e),no=a(e),z=p(e,"DIV",{class:!0});var E=w(z);u(ft.$$.fragment,E),Vo=a(E),rn=p(E,"P",{"data-svelte-h":!0}),k(rn)!=="svelte-ez5gwh"&&(rn.textContent=lr),Ho=a(E),an=p(E,"P",{"data-svelte-h":!0}),k(an)!=="svelte-q52n56"&&(an.innerHTML=cr),Xo=a(E),dn=p(E,"P",{"data-svelte-h":!0}),k(dn)!=="svelte-hswkmf"&&(dn.innerHTML=pr),Eo=a(E),pe=p(E,"DIV",{class:!0});var ke=w(pe);u(_t.$$.fragment,ke),So=a(ke),ln=p(ke,"P",{"data-svelte-h":!0}),k(ln)!=="svelte-j3k353"&&(ln.innerHTML=mr),Po=a(ke),u(Je.$$.fragment,ke),ke.forEach(d),E.forEach(d),oo=a(e),u(bt.$$.fragment,e),so=a(e),fe=p(e,"DIV",{class:!0});var Ht=w(fe);u(Tt.$$.fragment,Ht),Qo=a(Ht),D=p(Ht,"DIV",{class:!0});var se=w(D);u(kt.$$.fragment,se),Oo=a(se),cn=p(se,"P",{"data-svelte-h":!0}),k(cn)!=="svelte-1a84y9v"&&(cn.innerHTML=hr),Ao=a(se),u(Ce.$$.fragment,se),Yo=a(se),u(Fe.$$.fragment,se),se.forEach(d),Ht.forEach(d),ro=a(e),u(yt.$$.fragment,e),ao=a(e),j=p(e,"DIV",{class:!0});var S=w(j);u(Mt.$$.fragment,S),Do=a(S),pn=p(S,"P",{"data-svelte-h":!0}),k(pn)!=="svelte-le54a0"&&(pn.innerHTML=gr),Ko=a(S),mn=p(S,"P",{"data-svelte-h":!0}),k(mn)!=="svelte-q52n56"&&(mn.innerHTML=ur),es=a(S),hn=p(S,"P",{"data-svelte-h":!0}),k(hn)!=="svelte-hswkmf"&&(hn.innerHTML=fr),ts=a(S),K=p(S,"DIV",{class:!0});var re=w(K);u(Bt.$$.fragment,re),ns=a(re),gn=p(re,"P",{"data-svelte-h":!0}),k(gn)!=="svelte-gays7"&&(gn.innerHTML=_r),os=a(re),u(xe.$$.fragment,re),ss=a(re),u(ze.$$.fragment,re),re.forEach(d),S.forEach(d),io=a(e),u(wt.$$.fragment,e),lo=a(e),U=p(e,"DIV",{class:!0});var P=w(U);u(vt.$$.fragment,P),rs=a(P),un=p(P,"P",{"data-svelte-h":!0}),k(un)!=="svelte-1jruyjf"&&(un.innerHTML=br),as=a(P),fn=p(P,"P",{"data-svelte-h":!0}),k(fn)!=="svelte-q52n56"&&(fn.innerHTML=Tr),is=a(P),_n=p(P,"P",{"data-svelte-h":!0}),k(_n)!=="svelte-hswkmf"&&(_n.innerHTML=kr),ds=a(P),N=p(P,"DIV",{class:!0});var Q=w(N);u($t.$$.fragment,Q),ls=a(Q),bn=p(Q,"P",{"data-svelte-h":!0}),k(bn)!=="svelte-afakhj"&&(bn.innerHTML=yr),cs=a(Q),u(je.$$.fragment,Q),ps=a(Q),u(Ue.$$.fragment,Q),ms=a(Q),u(Ie.$$.fragment,Q),Q.forEach(d),P.forEach(d),co=a(e),u(Jt.$$.fragment,e),po=a(e),I=p(e,"DIV",{class:!0});var O=w(I);u(Ct.$$.fragment,O),hs=a(O),Tn=p(O,"P",{"data-svelte-h":!0}),k(Tn)!=="svelte-ip8mlz"&&(Tn.textContent=Mr),gs=a(O),kn=p(O,"P",{"data-svelte-h":!0}),k(kn)!=="svelte-q52n56"&&(kn.innerHTML=Br),us=a(O),yn=p(O,"P",{"data-svelte-h":!0}),k(yn)!=="svelte-hswkmf"&&(yn.innerHTML=wr),fs=a(O),G=p(O,"DIV",{class:!0});var A=w(G);u(Ft.$$.fragment,A),_s=a(A),Mn=p(A,"P",{"data-svelte-h":!0}),k(Mn)!=="svelte-133jtnl"&&(Mn.innerHTML=vr),bs=a(A),u(qe.$$.fragment,A),Ts=a(A),u(We.$$.fragment,A),ks=a(A),u(Le.$$.fragment,A),A.forEach(d),O.forEach(d),mo=a(e),u(xt.$$.fragment,e),ho=a(e),q=p(e,"DIV",{class:!0});var Y=w(q);u(zt.$$.fragment,Y),ys=a(Y),Bn=p(Y,"P",{"data-svelte-h":!0}),k(Bn)!=="svelte-1wf4ogq"&&(Bn.textContent=$r),Ms=a(Y),wn=p(Y,"P",{"data-svelte-h":!0}),k(wn)!=="svelte-q52n56"&&(wn.innerHTML=Jr),Bs=a(Y),vn=p(Y,"P",{"data-svelte-h":!0}),k(vn)!=="svelte-hswkmf"&&(vn.innerHTML=Cr),ws=a(Y),ee=p(Y,"DIV",{class:!0});var Ee=w(ee);u(jt.$$.fragment,Ee),vs=a(Ee),$n=p(Ee,"P",{"data-svelte-h":!0}),k($n)!=="svelte-1beh311"&&($n.innerHTML=Fr),$s=a(Ee),u(Ze.$$.fragment,Ee),Js=a(Ee),u(Ne.$$.fragment,Ee),Ee.forEach(d),Y.forEach(d),go=a(e),u(Ut.$$.fragment,e),uo=a(e),W=p(e,"DIV",{class:!0});var me=w(W);u(It.$$.fragment,me),Cs=a(me),Jn=p(me,"P",{"data-svelte-h":!0}),k(Jn)!=="svelte-1p1avfz"&&(Jn.textContent=xr),Fs=a(me),Cn=p(me,"P",{"data-svelte-h":!0}),k(Cn)!=="svelte-q52n56"&&(Cn.innerHTML=zr),xs=a(me),Fn=p(me,"P",{"data-svelte-h":!0}),k(Fn)!=="svelte-hswkmf"&&(Fn.innerHTML=jr),zs=a(me),te=p(me,"DIV",{class:!0});var Se=w(te);u(qt.$$.fragment,Se),js=a(Se),xn=p(Se,"P",{"data-svelte-h":!0}),k(xn)!=="svelte-1oj62xr"&&(xn.innerHTML=Ur),Us=a(Se),u(Ge.$$.fragment,Se),Is=a(Se),u(Re.$$.fragment,Se),Se.forEach(d),me.forEach(d),fo=a(e),u(Wt.$$.fragment,e),_o=a(e),L=p(e,"DIV",{class:!0});var he=w(L);u(Lt.$$.fragment,he),qs=a(he),zn=p(he,"P",{"data-svelte-h":!0}),k(zn)!=="svelte-1eur2cm"&&(zn.innerHTML=Ir),Ws=a(he),jn=p(he,"P",{"data-svelte-h":!0}),k(jn)!=="svelte-q52n56"&&(jn.innerHTML=qr),Ls=a(he),Un=p(he,"P",{"data-svelte-h":!0}),k(Un)!=="svelte-hswkmf"&&(Un.innerHTML=Wr),Zs=a(he),R=p(he,"DIV",{class:!0});var ge=w(R);u(Zt.$$.fragment,ge),Ns=a(ge),In=p(ge,"P",{"data-svelte-h":!0}),k(In)!=="svelte-af9mpr"&&(In.innerHTML=Lr),Gs=a(ge),u(Ve.$$.fragment,ge),Rs=a(ge),u(He.$$.fragment,ge),Vs=a(ge),u(Xe.$$.fragment,ge),ge.forEach(d),he.forEach(d),bo=a(e),u(Nt.$$.fragment,e),To=a(e),Wn=p(e,"P",{}),w(Wn).forEach(d),this.h()},h(){v(t,"name","hf:doc:metadata"),v(t,"content",_a),Hr(C,"float","right"),v(H,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(de,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(ve,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(le,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(Kt,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(F,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(ce,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v($e,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(X,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(ue,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(pe,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(z,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(D,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(fe,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(K,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(j,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(N,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(U,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(G,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(I,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(ee,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(q,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(te,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(W,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(R,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(L,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,i){o(document.head,t),h(e,m,i),h(e,n,i),h(e,l,i),h(e,M,i),h(e,y,i),h(e,C,i),h(e,Pe,i),f(ie,e,i),h(e,Zn,i),h(e,Qe,i),h(e,Nn,i),h(e,Oe,i),h(e,Gn,i),f(Me,e,i),h(e,Rn,i),h(e,Ae,i),h(e,Vn,i),f(Be,e,i),h(e,Hn,i),f(Ye,e,i),h(e,Xn,i),h(e,De,i),h(e,En,i),f(Ke,e,i),h(e,Sn,i),h(e,et,i),h(e,Pn,i),f(tt,e,i),h(e,Qn,i),h(e,H,i),f(nt,H,null),o(H,yo),o(H,Xt),o(H,Mo),o(H,Et),o(H,Bo),f(we,H,null),h(e,On,i),f(ot,e,i),h(e,An,i),h(e,F,i),f(st,F,null),o(F,wo),o(F,St),o(F,vo),o(F,Pt),o(F,$o),o(F,de),f(rt,de,null),o(de,Jo),o(de,Qt),o(de,Co),o(de,Ot),o(F,Fo),o(F,ve),f(at,ve,null),o(ve,xo),o(ve,At),o(F,zo),o(F,le),f(it,le,null),o(le,jo),o(le,Yt),o(le,Uo),o(le,Dt),o(F,Io),o(F,Kt),f(dt,Kt,null),h(e,Yn,i),f(lt,e,i),h(e,Dn,i),h(e,X,i),f(ct,X,null),o(X,qo),o(X,en),o(X,Wo),o(X,ce),f(pt,ce,null),o(ce,Lo),o(ce,tn),o(ce,Zo),o(ce,nn),o(X,No),o(X,$e),f(mt,$e,null),o($e,Go),o($e,on),h(e,Kn,i),f(ht,e,i),h(e,eo,i),h(e,ue,i),f(gt,ue,null),o(ue,Ro),o(ue,sn),h(e,to,i),f(ut,e,i),h(e,no,i),h(e,z,i),f(ft,z,null),o(z,Vo),o(z,rn),o(z,Ho),o(z,an),o(z,Xo),o(z,dn),o(z,Eo),o(z,pe),f(_t,pe,null),o(pe,So),o(pe,ln),o(pe,Po),f(Je,pe,null),h(e,oo,i),f(bt,e,i),h(e,so,i),h(e,fe,i),f(Tt,fe,null),o(fe,Qo),o(fe,D),f(kt,D,null),o(D,Oo),o(D,cn),o(D,Ao),f(Ce,D,null),o(D,Yo),f(Fe,D,null),h(e,ro,i),f(yt,e,i),h(e,ao,i),h(e,j,i),f(Mt,j,null),o(j,Do),o(j,pn),o(j,Ko),o(j,mn),o(j,es),o(j,hn),o(j,ts),o(j,K),f(Bt,K,null),o(K,ns),o(K,gn),o(K,os),f(xe,K,null),o(K,ss),f(ze,K,null),h(e,io,i),f(wt,e,i),h(e,lo,i),h(e,U,i),f(vt,U,null),o(U,rs),o(U,un),o(U,as),o(U,fn),o(U,is),o(U,_n),o(U,ds),o(U,N),f($t,N,null),o(N,ls),o(N,bn),o(N,cs),f(je,N,null),o(N,ps),f(Ue,N,null),o(N,ms),f(Ie,N,null),h(e,co,i),f(Jt,e,i),h(e,po,i),h(e,I,i),f(Ct,I,null),o(I,hs),o(I,Tn),o(I,gs),o(I,kn),o(I,us),o(I,yn),o(I,fs),o(I,G),f(Ft,G,null),o(G,_s),o(G,Mn),o(G,bs),f(qe,G,null),o(G,Ts),f(We,G,null),o(G,ks),f(Le,G,null),h(e,mo,i),f(xt,e,i),h(e,ho,i),h(e,q,i),f(zt,q,null),o(q,ys),o(q,Bn),o(q,Ms),o(q,wn),o(q,Bs),o(q,vn),o(q,ws),o(q,ee),f(jt,ee,null),o(ee,vs),o(ee,$n),o(ee,$s),f(Ze,ee,null),o(ee,Js),f(Ne,ee,null),h(e,go,i),f(Ut,e,i),h(e,uo,i),h(e,W,i),f(It,W,null),o(W,Cs),o(W,Jn),o(W,Fs),o(W,Cn),o(W,xs),o(W,Fn),o(W,zs),o(W,te),f(qt,te,null),o(te,js),o(te,xn),o(te,Us),f(Ge,te,null),o(te,Is),f(Re,te,null),h(e,fo,i),f(Wt,e,i),h(e,_o,i),h(e,L,i),f(Lt,L,null),o(L,qs),o(L,zn),o(L,Ws),o(L,jn),o(L,Ls),o(L,Un),o(L,Zs),o(L,R),f(Zt,R,null),o(R,Ns),o(R,In),o(R,Gs),f(Ve,R,null),o(R,Rs),f(He,R,null),o(R,Vs),f(Xe,R,null),h(e,bo,i),f(Nt,e,i),h(e,To,i),h(e,Wn,i),ko=!0},p(e,[i]){const ne={};i&2&&(ne.$$scope={dirty:i,ctx:e}),Me.$set(ne);const x={};i&2&&(x.$$scope={dirty:i,ctx:e}),Be.$set(x);const _e={};i&2&&(_e.$$scope={dirty:i,ctx:e}),we.$set(_e);const Gt={};i&2&&(Gt.$$scope={dirty:i,ctx:e}),Je.$set(Gt);const be={};i&2&&(be.$$scope={dirty:i,ctx:e}),Ce.$set(be);const Ln={};i&2&&(Ln.$$scope={dirty:i,ctx:e}),Fe.$set(Ln);const oe={};i&2&&(oe.$$scope={dirty:i,ctx:e}),xe.$set(oe);const Te={};i&2&&(Te.$$scope={dirty:i,ctx:e}),ze.$set(Te);const Rt={};i&2&&(Rt.$$scope={dirty:i,ctx:e}),je.$set(Rt);const Vt={};i&2&&(Vt.$$scope={dirty:i,ctx:e}),Ue.$set(Vt);const E={};i&2&&(E.$$scope={dirty:i,ctx:e}),Ie.$set(E);const ke={};i&2&&(ke.$$scope={dirty:i,ctx:e}),qe.$set(ke);const Ht={};i&2&&(Ht.$$scope={dirty:i,ctx:e}),We.$set(Ht);const se={};i&2&&(se.$$scope={dirty:i,ctx:e}),Le.$set(se);const S={};i&2&&(S.$$scope={dirty:i,ctx:e}),Ze.$set(S);const re={};i&2&&(re.$$scope={dirty:i,ctx:e}),Ne.$set(re);const P={};i&2&&(P.$$scope={dirty:i,ctx:e}),Ge.$set(P);const Q={};i&2&&(Q.$$scope={dirty:i,ctx:e}),Re.$set(Q);const O={};i&2&&(O.$$scope={dirty:i,ctx:e}),Ve.$set(O);const A={};i&2&&(A.$$scope={dirty:i,ctx:e}),He.$set(A);const Y={};i&2&&(Y.$$scope={dirty:i,ctx:e}),Xe.$set(Y)},i(e){ko||(_(ie.$$.fragment,e),_(Me.$$.fragment,e),_(Be.$$.fragment,e),_(Ye.$$.fragment,e),_(Ke.$$.fragment,e),_(tt.$$.fragment,e),_(nt.$$.fragment,e),_(we.$$.fragment,e),_(ot.$$.fragment,e),_(st.$$.fragment,e),_(rt.$$.fragment,e),_(at.$$.fragment,e),_(it.$$.fragment,e),_(dt.$$.fragment,e),_(lt.$$.fragment,e),_(ct.$$.fragment,e),_(pt.$$.fragment,e),_(mt.$$.fragment,e),_(ht.$$.fragment,e),_(gt.$$.fragment,e),_(ut.$$.fragment,e),_(ft.$$.fragment,e),_(_t.$$.fragment,e),_(Je.$$.fragment,e),_(bt.$$.fragment,e),_(Tt.$$.fragment,e),_(kt.$$.fragment,e),_(Ce.$$.fragment,e),_(Fe.$$.fragment,e),_(yt.$$.fragment,e),_(Mt.$$.fragment,e),_(Bt.$$.fragment,e),_(xe.$$.fragment,e),_(ze.$$.fragment,e),_(wt.$$.fragment,e),_(vt.$$.fragment,e),_($t.$$.fragment,e),_(je.$$.fragment,e),_(Ue.$$.fragment,e),_(Ie.$$.fragment,e),_(Jt.$$.fragment,e),_(Ct.$$.fragment,e),_(Ft.$$.fragment,e),_(qe.$$.fragment,e),_(We.$$.fragment,e),_(Le.$$.fragment,e),_(xt.$$.fragment,e),_(zt.$$.fragment,e),_(jt.$$.fragment,e),_(Ze.$$.fragment,e),_(Ne.$$.fragment,e),_(Ut.$$.fragment,e),_(It.$$.fragment,e),_(qt.$$.fragment,e),_(Ge.$$.fragment,e),_(Re.$$.fragment,e),_(Wt.$$.fragment,e),_(Lt.$$.fragment,e),_(Zt.$$.fragment,e),_(Ve.$$.fragment,e),_(He.$$.fragment,e),_(Xe.$$.fragment,e),_(Nt.$$.fragment,e),ko=!0)},o(e){b(ie.$$.fragment,e),b(Me.$$.fragment,e),b(Be.$$.fragment,e),b(Ye.$$.fragment,e),b(Ke.$$.fragment,e),b(tt.$$.fragment,e),b(nt.$$.fragment,e),b(we.$$.fragment,e),b(ot.$$.fragment,e),b(st.$$.fragment,e),b(rt.$$.fragment,e),b(at.$$.fragment,e),b(it.$$.fragment,e),b(dt.$$.fragment,e),b(lt.$$.fragment,e),b(ct.$$.fragment,e),b(pt.$$.fragment,e),b(mt.$$.fragment,e),b(ht.$$.fragment,e),b(gt.$$.fragment,e),b(ut.$$.fragment,e),b(ft.$$.fragment,e),b(_t.$$.fragment,e),b(Je.$$.fragment,e),b(bt.$$.fragment,e),b(Tt.$$.fragment,e),b(kt.$$.fragment,e),b(Ce.$$.fragment,e),b(Fe.$$.fragment,e),b(yt.$$.fragment,e),b(Mt.$$.fragment,e),b(Bt.$$.fragment,e),b(xe.$$.fragment,e),b(ze.$$.fragment,e),b(wt.$$.fragment,e),b(vt.$$.fragment,e),b($t.$$.fragment,e),b(je.$$.fragment,e),b(Ue.$$.fragment,e),b(Ie.$$.fragment,e),b(Jt.$$.fragment,e),b(Ct.$$.fragment,e),b(Ft.$$.fragment,e),b(qe.$$.fragment,e),b(We.$$.fragment,e),b(Le.$$.fragment,e),b(xt.$$.fragment,e),b(zt.$$.fragment,e),b(jt.$$.fragment,e),b(Ze.$$.fragment,e),b(Ne.$$.fragment,e),b(Ut.$$.fragment,e),b(It.$$.fragment,e),b(qt.$$.fragment,e),b(Ge.$$.fragment,e),b(Re.$$.fragment,e),b(Wt.$$.fragment,e),b(Lt.$$.fragment,e),b(Zt.$$.fragment,e),b(Ve.$$.fragment,e),b(He.$$.fragment,e),b(Xe.$$.fragment,e),b(Nt.$$.fragment,e),ko=!1},d(e){e&&(d(m),d(n),d(l),d(M),d(y),d(C),d(Pe),d(Zn),d(Qe),d(Nn),d(Oe),d(Gn),d(Rn),d(Ae),d(Vn),d(Hn),d(Xn),d(De),d(En),d(Sn),d(et),d(Pn),d(Qn),d(H),d(On),d(An),d(F),d(Yn),d(Dn),d(X),d(Kn),d(eo),d(ue),d(to),d(no),d(z),d(oo),d(so),d(fe),d(ro),d(ao),d(j),d(io),d(lo),d(U),d(co),d(po),d(I),d(mo),d(ho),d(q),d(go),d(uo),d(W),d(fo),d(_o),d(L),d(bo),d(To),d(Wn)),d(t),T(ie,e),T(Me,e),T(Be,e),T(Ye,e),T(Ke,e),T(tt,e),T(nt),T(we),T(ot,e),T(st),T(rt),T(at),T(it),T(dt),T(lt,e),T(ct),T(pt),T(mt),T(ht,e),T(gt),T(ut,e),T(ft),T(_t),T(Je),T(bt,e),T(Tt),T(kt),T(Ce),T(Fe),T(yt,e),T(Mt),T(Bt),T(xe),T(ze),T(wt,e),T(vt),T($t),T(je),T(Ue),T(Ie),T(Jt,e),T(Ct),T(Ft),T(qe),T(We),T(Le),T(xt,e),T(zt),T(jt),T(Ze),T(Ne),T(Ut,e),T(It),T(qt),T(Ge),T(Re),T(Wt,e),T(Lt),T(Zt),T(Ve),T(He),T(Xe),T(Nt,e)}}}const _a='{"title":"BigBird","local":"bigbird","sections":[{"title":"Notes","local":"notes","sections":[],"depth":2},{"title":"Resources","local":"resources","sections":[],"depth":2},{"title":"BigBirdConfig","local":"transformers.BigBirdConfig","sections":[],"depth":2},{"title":"BigBirdTokenizer","local":"transformers.BigBirdTokenizer","sections":[],"depth":2},{"title":"BigBirdTokenizerFast","local":"transformers.BigBirdTokenizerFast","sections":[],"depth":2},{"title":"BigBird specific outputs","local":"transformers.models.big_bird.modeling_big_bird.BigBirdForPreTrainingOutput","sections":[],"depth":2},{"title":"BigBirdModel","local":"transformers.BigBirdModel","sections":[],"depth":2},{"title":"BigBirdForPreTraining","local":"transformers.BigBirdForPreTraining","sections":[],"depth":2},{"title":"BigBirdForCausalLM","local":"transformers.BigBirdForCausalLM","sections":[],"depth":2},{"title":"BigBirdForMaskedLM","local":"transformers.BigBirdForMaskedLM","sections":[],"depth":2},{"title":"BigBirdForSequenceClassification","local":"transformers.BigBirdForSequenceClassification","sections":[],"depth":2},{"title":"BigBirdForMultipleChoice","local":"transformers.BigBirdForMultipleChoice","sections":[],"depth":2},{"title":"BigBirdForTokenClassification","local":"transformers.BigBirdForTokenClassification","sections":[],"depth":2},{"title":"BigBirdForQuestionAnswering","local":"transformers.BigBirdForQuestionAnswering","sections":[],"depth":2}],"depth":1}';function ba(B){return Nr(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class Ja extends Gr{constructor(t){super(),Rr(this,t,ba,fa,Zr,{})}}export{Ja as component};
