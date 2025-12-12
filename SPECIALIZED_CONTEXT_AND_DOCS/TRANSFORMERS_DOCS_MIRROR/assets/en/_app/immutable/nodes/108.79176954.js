import{s as pa,o as ha,n as z}from"../chunks/scheduler.18a86fab.js";import{S as ua,i as fa,g as c,s as a,r as f,A as ga,h as m,f as i,c as r,j as v,x as T,u as g,k as C,l as ba,y as s,a as p,v as b,d as _,t as y,w as k}from"../chunks/index.98837b22.js";import{T as qe}from"../chunks/Tip.77304350.js";import{D as $}from"../chunks/Docstring.a1ef7999.js";import{C as D}from"../chunks/CodeBlock.8d0c2e8a.js";import{E as Be}from"../chunks/ExampleCodeBlock.8c3ee1f9.js";import{H as ne,E as _a}from"../chunks/getInferenceSnippets.06c2775f.js";import{H as ya,a as bs}from"../chunks/HfOption.6641485e.js";function ka(w){let t,h='This model was contributed by the <a href="https://huggingface.co/almanach" rel="nofollow">ALMAnaCH (Inria)</a> team.',o,d,M="Click on the CamemBERT models in the right sidebar for more examples of how to apply CamemBERT to different NLP tasks.";return{c(){t=c("p"),t.innerHTML=h,o=a(),d=c("p"),d.textContent=M},l(n){t=m(n,"P",{"data-svelte-h":!0}),T(t)!=="svelte-admls7"&&(t.innerHTML=h),o=r(n),d=m(n,"P",{"data-svelte-h":!0}),T(d)!=="svelte-10l9lmp"&&(d.textContent=M)},m(n,u){p(n,t,u),p(n,o,u),p(n,d,u)},p:z,d(n){n&&(i(t),i(o),i(d))}}}function Ta(w){let t,h;return t=new D({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwcGlwZWxpbmUlMEElMEFwaXBlbGluZSUyMCUzRCUyMHBpcGVsaW5lKCUyMmZpbGwtbWFzayUyMiUyQyUyMG1vZGVsJTNEJTIyY2FtZW1iZXJ0LWJhc2UlMjIlMkMlMjBkdHlwZSUzRHRvcmNoLmZsb2F0MTYlMkMlMjBkZXZpY2UlM0QwKSUwQXBpcGVsaW5lKCUyMkxlJTIwY2FtZW1iZXJ0JTIwZXN0JTIwdW4lMjBkJUMzJUE5bGljaWV1eCUyMGZyb21hZ2UlMjAlM0NtYXNrJTNFLiUyMik=",highlighted:`<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> pipeline

pipeline = pipeline(<span class="hljs-string">&quot;fill-mask&quot;</span>, model=<span class="hljs-string">&quot;camembert-base&quot;</span>, dtype=torch.float16, device=<span class="hljs-number">0</span>)
pipeline(<span class="hljs-string">&quot;Le camembert est un délicieux fromage &lt;mask&gt;.&quot;</span>)`,wrap:!1}}),{c(){f(t.$$.fragment)},l(o){g(t.$$.fragment,o)},m(o,d){b(t,o,d),h=!0},p:z,i(o){h||(_(t.$$.fragment,o),h=!0)},o(o){y(t.$$.fragment,o),h=!1},d(o){k(t,o)}}}function Ma(w){let t,h;return t=new D({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Rva2VuaXplciUyQyUyMEF1dG9Nb2RlbEZvck1hc2tlZExNJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIyY2FtZW1iZXJ0LWJhc2UlMjIpJTBBbW9kZWwlMjAlM0QlMjBBdXRvTW9kZWxGb3JNYXNrZWRMTS5mcm9tX3ByZXRyYWluZWQoJTIyY2FtZW1iZXJ0LWJhc2UlMjIlMkMlMjBkdHlwZSUzRCUyMmF1dG8lMjIlMkMlMjBkZXZpY2VfbWFwJTNEJTIyYXV0byUyMiUyQyUyMGF0dG5faW1wbGVtZW50YXRpb24lM0QlMjJzZHBhJTIyKSUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplciglMjJMZSUyMGNhbWVtYmVydCUyMGVzdCUyMHVuJTIwZCVDMyVBOWxpY2lldXglMjBmcm9tYWdlJTIwJTNDbWFzayUzRS4lMjIlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKS50byhtb2RlbC5kZXZpY2UpJTBBJTBBd2l0aCUyMHRvcmNoLm5vX2dyYWQoKSUzQSUwQSUyMCUyMCUyMCUyMG91dHB1dHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyklMEElMjAlMjAlMjAlMjBwcmVkaWN0aW9ucyUyMCUzRCUyMG91dHB1dHMubG9naXRzJTBBJTBBbWFza2VkX2luZGV4JTIwJTNEJTIwdG9yY2gud2hlcmUoaW5wdXRzJTVCJ2lucHV0X2lkcyclNUQlMjAlM0QlM0QlMjB0b2tlbml6ZXIubWFza190b2tlbl9pZCklNUIxJTVEJTBBcHJlZGljdGVkX3Rva2VuX2lkJTIwJTNEJTIwcHJlZGljdGlvbnMlNUIwJTJDJTIwbWFza2VkX2luZGV4JTVELmFyZ21heChkaW0lM0QtMSklMEFwcmVkaWN0ZWRfdG9rZW4lMjAlM0QlMjB0b2tlbml6ZXIuZGVjb2RlKHByZWRpY3RlZF90b2tlbl9pZCklMEElMEFwcmludChmJTIyVGhlJTIwcHJlZGljdGVkJTIwdG9rZW4lMjBpcyUzQSUyMCU3QnByZWRpY3RlZF90b2tlbiU3RCUyMik=",highlighted:`<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;camembert-base&quot;</span>)
model = AutoModelForMaskedLM.from_pretrained(<span class="hljs-string">&quot;camembert-base&quot;</span>, dtype=<span class="hljs-string">&quot;auto&quot;</span>, device_map=<span class="hljs-string">&quot;auto&quot;</span>, attn_implementation=<span class="hljs-string">&quot;sdpa&quot;</span>)
inputs = tokenizer(<span class="hljs-string">&quot;Le camembert est un délicieux fromage &lt;mask&gt;.&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>).to(model.device)

<span class="hljs-keyword">with</span> torch.no_grad():
    outputs = model(**inputs)
    predictions = outputs.logits

masked_index = torch.where(inputs[<span class="hljs-string">&#x27;input_ids&#x27;</span>] == tokenizer.mask_token_id)[<span class="hljs-number">1</span>]
predicted_token_id = predictions[<span class="hljs-number">0</span>, masked_index].argmax(dim=-<span class="hljs-number">1</span>)
predicted_token = tokenizer.decode(predicted_token_id)

<span class="hljs-built_in">print</span>(<span class="hljs-string">f&quot;The predicted token is: <span class="hljs-subst">{predicted_token}</span>&quot;</span>)`,wrap:!1}}),{c(){f(t.$$.fragment)},l(o){g(t.$$.fragment,o)},m(o,d){b(t,o,d),h=!0},p:z,i(o){h||(_(t.$$.fragment,o),h=!0)},o(o){y(t.$$.fragment,o),h=!1},d(o){k(t,o)}}}function wa(w){let t,h;return t=new D({props:{code:"ZWNobyUyMC1lJTIwJTIyTGUlMjBjYW1lbWJlcnQlMjBlc3QlMjB1biUyMGQlQzMlQTlsaWNpZXV4JTIwZnJvbWFnZSUyMCUzQ21hc2slM0UuJTIyJTIwJTdDJTIwdHJhbnNmb3JtZXJzJTIwcnVuJTIwLS10YXNrJTIwZmlsbC1tYXNrJTIwLS1tb2RlbCUyMGNhbWVtYmVydC1iYXNlJTIwLS1kZXZpY2UlMjAw",highlighted:'<span class="hljs-built_in">echo</span> -e <span class="hljs-string">&quot;Le camembert est un délicieux fromage &lt;mask&gt;.&quot;</span> | transformers run --task fill-mask --model camembert-base --device 0',wrap:!1}}),{c(){f(t.$$.fragment)},l(o){g(t.$$.fragment,o)},m(o,d){b(t,o,d),h=!0},p:z,i(o){h||(_(t.$$.fragment,o),h=!0)},o(o){y(t.$$.fragment,o),h=!1},d(o){k(t,o)}}}function va(w){let t,h,o,d,M,n;return t=new bs({props:{id:"usage",option:"Pipeline",$$slots:{default:[Ta]},$$scope:{ctx:w}}}),o=new bs({props:{id:"usage",option:"AutoModel",$$slots:{default:[Ma]},$$scope:{ctx:w}}}),M=new bs({props:{id:"usage",option:"transformers CLI",$$slots:{default:[wa]},$$scope:{ctx:w}}}),{c(){f(t.$$.fragment),h=a(),f(o.$$.fragment),d=a(),f(M.$$.fragment)},l(u){g(t.$$.fragment,u),h=r(u),g(o.$$.fragment,u),d=r(u),g(M.$$.fragment,u)},m(u,J){b(t,u,J),p(u,h,J),b(o,u,J),p(u,d,J),b(M,u,J),n=!0},p(u,J){const yn={};J&2&&(yn.$$scope={dirty:J,ctx:u}),t.$set(yn);const Ne={};J&2&&(Ne.$$scope={dirty:J,ctx:u}),o.$set(Ne);const oe={};J&2&&(oe.$$scope={dirty:J,ctx:u}),M.$set(oe)},i(u){n||(_(t.$$.fragment,u),_(o.$$.fragment,u),_(M.$$.fragment,u),n=!0)},o(u){y(t.$$.fragment,u),y(o.$$.fragment,u),y(M.$$.fragment,u),n=!1},d(u){u&&(i(h),i(d)),k(t,u),k(o,u),k(M,u)}}}function Ca(w){let t,h="Example:",o,d,M;return d=new D({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMENhbWVtYmVydENvbmZpZyUyQyUyMENhbWVtYmVydE1vZGVsJTBBJTBBJTIzJTIwSW5pdGlhbGl6aW5nJTIwYSUyMENhbWVtYmVydCUyMGFsbWFuYWNoJTJGY2FtZW1iZXJ0LWJhc2UlMjBzdHlsZSUyMGNvbmZpZ3VyYXRpb24lMEFjb25maWd1cmF0aW9uJTIwJTNEJTIwQ2FtZW1iZXJ0Q29uZmlnKCklMEElMEElMjMlMjBJbml0aWFsaXppbmclMjBhJTIwbW9kZWwlMjAod2l0aCUyMHJhbmRvbSUyMHdlaWdodHMpJTIwZnJvbSUyMHRoZSUyMGFsbWFuYWNoJTJGY2FtZW1iZXJ0LWJhc2UlMjBzdHlsZSUyMGNvbmZpZ3VyYXRpb24lMEFtb2RlbCUyMCUzRCUyMENhbWVtYmVydE1vZGVsKGNvbmZpZ3VyYXRpb24pJTBBJTBBJTIzJTIwQWNjZXNzaW5nJTIwdGhlJTIwbW9kZWwlMjBjb25maWd1cmF0aW9uJTBBY29uZmlndXJhdGlvbiUyMCUzRCUyMG1vZGVsLmNvbmZpZw==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> CamembertConfig, CamembertModel

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a Camembert almanach/camembert-base style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = CamembertConfig()

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a model (with random weights) from the almanach/camembert-base style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = CamembertModel(configuration)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Accessing the model configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = model.config`,wrap:!1}}),{c(){t=c("p"),t.textContent=h,o=a(),f(d.$$.fragment)},l(n){t=m(n,"P",{"data-svelte-h":!0}),T(t)!=="svelte-11lpom8"&&(t.textContent=h),o=r(n),g(d.$$.fragment,n)},m(n,u){p(n,t,u),p(n,o,u),b(d,n,u),M=!0},p:z,i(n){M||(_(d.$$.fragment,n),M=!0)},o(n){y(d.$$.fragment,n),M=!1},d(n){n&&(i(t),i(o)),k(d,n)}}}function $a(w){let t,h=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=h},l(o){t=m(o,"P",{"data-svelte-h":!0}),T(t)!=="svelte-fincs2"&&(t.innerHTML=h)},m(o,d){p(o,t,d)},p:z,d(o){o&&i(t)}}}function Ja(w){let t,h=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=h},l(o){t=m(o,"P",{"data-svelte-h":!0}),T(t)!=="svelte-fincs2"&&(t.innerHTML=h)},m(o,d){p(o,t,d)},p:z,d(o){o&&i(t)}}}function ja(w){let t,h="Example:",o,d,M;return d=new D({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBDYW1lbWJlcnRGb3JDYXVzYWxMTSUyQyUyMEF1dG9Db25maWclMEFpbXBvcnQlMjB0b3JjaCUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMmFsbWFuYWNoJTJGY2FtZW1iZXJ0LWJhc2UlMjIpJTBBY29uZmlnJTIwJTNEJTIwQXV0b0NvbmZpZy5mcm9tX3ByZXRyYWluZWQoJTIyYWxtYW5hY2glMkZjYW1lbWJlcnQtYmFzZSUyMiklMEFjb25maWcuaXNfZGVjb2RlciUyMCUzRCUyMFRydWUlMEFtb2RlbCUyMCUzRCUyMENhbWVtYmVydEZvckNhdXNhbExNLmZyb21fcHJldHJhaW5lZCglMjJhbG1hbmFjaCUyRmNhbWVtYmVydC1iYXNlJTIyJTJDJTIwY29uZmlnJTNEY29uZmlnKSUwQSUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplciglMjJIZWxsbyUyQyUyMG15JTIwZG9nJTIwaXMlMjBjdXRlJTIyJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiklMEFvdXRwdXRzJTIwJTNEJTIwbW9kZWwoKippbnB1dHMpJTBBJTBBcHJlZGljdGlvbl9sb2dpdHMlMjAlM0QlMjBvdXRwdXRzLmxvZ2l0cw==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, CamembertForCausalLM, AutoConfig
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;almanach/camembert-base&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>config = AutoConfig.from_pretrained(<span class="hljs-string">&quot;almanach/camembert-base&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>config.is_decoder = <span class="hljs-literal">True</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = CamembertForCausalLM.from_pretrained(<span class="hljs-string">&quot;almanach/camembert-base&quot;</span>, config=config)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs)

<span class="hljs-meta">&gt;&gt;&gt; </span>prediction_logits = outputs.logits`,wrap:!1}}),{c(){t=c("p"),t.textContent=h,o=a(),f(d.$$.fragment)},l(n){t=m(n,"P",{"data-svelte-h":!0}),T(t)!=="svelte-11lpom8"&&(t.textContent=h),o=r(n),g(d.$$.fragment,n)},m(n,u){p(n,t,u),p(n,o,u),b(d,n,u),M=!0},p:z,i(n){M||(_(d.$$.fragment,n),M=!0)},o(n){y(d.$$.fragment,n),M=!1},d(n){n&&(i(t),i(o)),k(d,n)}}}function za(w){let t,h=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=h},l(o){t=m(o,"P",{"data-svelte-h":!0}),T(t)!=="svelte-fincs2"&&(t.innerHTML=h)},m(o,d){p(o,t,d)},p:z,d(o){o&&i(t)}}}function xa(w){let t,h="Example:",o,d,M;return d=new D({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBDYW1lbWJlcnRGb3JNYXNrZWRMTSUwQWltcG9ydCUyMHRvcmNoJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIyYWxtYW5hY2glMkZjYW1lbWJlcnQtYmFzZSUyMiklMEFtb2RlbCUyMCUzRCUyMENhbWVtYmVydEZvck1hc2tlZExNLmZyb21fcHJldHJhaW5lZCglMjJhbG1hbmFjaCUyRmNhbWVtYmVydC1iYXNlJTIyKSUwQSUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplciglMjJUaGUlMjBjYXBpdGFsJTIwb2YlMjBGcmFuY2UlMjBpcyUyMCUzQ21hc2slM0UuJTIyJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiklMEElMEF3aXRoJTIwdG9yY2gubm9fZ3JhZCgpJTNBJTBBJTIwJTIwJTIwJTIwbG9naXRzJTIwJTNEJTIwbW9kZWwoKippbnB1dHMpLmxvZ2l0cyUwQSUwQSUyMyUyMHJldHJpZXZlJTIwaW5kZXglMjBvZiUyMCUzQ21hc2slM0UlMEFtYXNrX3Rva2VuX2luZGV4JTIwJTNEJTIwKGlucHV0cy5pbnB1dF9pZHMlMjAlM0QlM0QlMjB0b2tlbml6ZXIubWFza190b2tlbl9pZCklNUIwJTVELm5vbnplcm8oYXNfdHVwbGUlM0RUcnVlKSU1QjAlNUQlMEElMEFwcmVkaWN0ZWRfdG9rZW5faWQlMjAlM0QlMjBsb2dpdHMlNUIwJTJDJTIwbWFza190b2tlbl9pbmRleCU1RC5hcmdtYXgoYXhpcyUzRC0xKSUwQXRva2VuaXplci5kZWNvZGUocHJlZGljdGVkX3Rva2VuX2lkKSUwQSUwQWxhYmVscyUyMCUzRCUyMHRva2VuaXplciglMjJUaGUlMjBjYXBpdGFsJTIwb2YlMjBGcmFuY2UlMjBpcyUyMFBhcmlzLiUyMiUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpJTVCJTIyaW5wdXRfaWRzJTIyJTVEJTBBJTIzJTIwbWFzayUyMGxhYmVscyUyMG9mJTIwbm9uLSUzQ21hc2slM0UlMjB0b2tlbnMlMEFsYWJlbHMlMjAlM0QlMjB0b3JjaC53aGVyZShpbnB1dHMuaW5wdXRfaWRzJTIwJTNEJTNEJTIwdG9rZW5pemVyLm1hc2tfdG9rZW5faWQlMkMlMjBsYWJlbHMlMkMlMjAtMTAwKSUwQSUwQW91dHB1dHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyUyQyUyMGxhYmVscyUzRGxhYmVscyklMEFyb3VuZChvdXRwdXRzLmxvc3MuaXRlbSgpJTJDJTIwMik=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, CamembertForMaskedLM
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;almanach/camembert-base&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = CamembertForMaskedLM.from_pretrained(<span class="hljs-string">&quot;almanach/camembert-base&quot;</span>)

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
...`,wrap:!1}}),{c(){t=c("p"),t.textContent=h,o=a(),f(d.$$.fragment)},l(n){t=m(n,"P",{"data-svelte-h":!0}),T(t)!=="svelte-11lpom8"&&(t.textContent=h),o=r(n),g(d.$$.fragment,n)},m(n,u){p(n,t,u),p(n,o,u),b(d,n,u),M=!0},p:z,i(n){M||(_(d.$$.fragment,n),M=!0)},o(n){y(d.$$.fragment,n),M=!1},d(n){n&&(i(t),i(o)),k(d,n)}}}function Fa(w){let t,h=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=h},l(o){t=m(o,"P",{"data-svelte-h":!0}),T(t)!=="svelte-fincs2"&&(t.innerHTML=h)},m(o,d){p(o,t,d)},p:z,d(o){o&&i(t)}}}function Ua(w){let t,h="Example of single-label classification:",o,d,M;return d=new D({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Rva2VuaXplciUyQyUyMENhbWVtYmVydEZvclNlcXVlbmNlQ2xhc3NpZmljYXRpb24lMEElMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJhbG1hbmFjaCUyRmNhbWVtYmVydC1iYXNlJTIyKSUwQW1vZGVsJTIwJTNEJTIwQ2FtZW1iZXJ0Rm9yU2VxdWVuY2VDbGFzc2lmaWNhdGlvbi5mcm9tX3ByZXRyYWluZWQoJTIyYWxtYW5hY2glMkZjYW1lbWJlcnQtYmFzZSUyMiklMEElMEFpbnB1dHMlMjAlM0QlMjB0b2tlbml6ZXIoJTIySGVsbG8lMkMlMjBteSUyMGRvZyUyMGlzJTIwY3V0ZSUyMiUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpJTBBJTBBd2l0aCUyMHRvcmNoLm5vX2dyYWQoKSUzQSUwQSUyMCUyMCUyMCUyMGxvZ2l0cyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzKS5sb2dpdHMlMEElMEFwcmVkaWN0ZWRfY2xhc3NfaWQlMjAlM0QlMjBsb2dpdHMuYXJnbWF4KCkuaXRlbSgpJTBBbW9kZWwuY29uZmlnLmlkMmxhYmVsJTVCcHJlZGljdGVkX2NsYXNzX2lkJTVEJTBBJTBBJTIzJTIwVG8lMjB0cmFpbiUyMGElMjBtb2RlbCUyMG9uJTIwJTYwbnVtX2xhYmVscyU2MCUyMGNsYXNzZXMlMkMlMjB5b3UlMjBjYW4lMjBwYXNzJTIwJTYwbnVtX2xhYmVscyUzRG51bV9sYWJlbHMlNjAlMjB0byUyMCU2MC5mcm9tX3ByZXRyYWluZWQoLi4uKSU2MCUwQW51bV9sYWJlbHMlMjAlM0QlMjBsZW4obW9kZWwuY29uZmlnLmlkMmxhYmVsKSUwQW1vZGVsJTIwJTNEJTIwQ2FtZW1iZXJ0Rm9yU2VxdWVuY2VDbGFzc2lmaWNhdGlvbi5mcm9tX3ByZXRyYWluZWQoJTIyYWxtYW5hY2glMkZjYW1lbWJlcnQtYmFzZSUyMiUyQyUyMG51bV9sYWJlbHMlM0RudW1fbGFiZWxzKSUwQSUwQWxhYmVscyUyMCUzRCUyMHRvcmNoLnRlbnNvciglNUIxJTVEKSUwQWxvc3MlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyUyQyUyMGxhYmVscyUzRGxhYmVscykubG9zcyUwQXJvdW5kKGxvc3MuaXRlbSgpJTJDJTIwMik=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, CamembertForSequenceClassification

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;almanach/camembert-base&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = CamembertForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;almanach/camembert-base&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    logits = model(**inputs).logits

<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_class_id = logits.argmax().item()
<span class="hljs-meta">&gt;&gt;&gt; </span>model.config.id2label[predicted_class_id]
...

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># To train a model on \`num_labels\` classes, you can pass \`num_labels=num_labels\` to \`.from_pretrained(...)\`</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>num_labels = <span class="hljs-built_in">len</span>(model.config.id2label)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = CamembertForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;almanach/camembert-base&quot;</span>, num_labels=num_labels)

<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.tensor([<span class="hljs-number">1</span>])
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = model(**inputs, labels=labels).loss
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">round</span>(loss.item(), <span class="hljs-number">2</span>)
...`,wrap:!1}}),{c(){t=c("p"),t.textContent=h,o=a(),f(d.$$.fragment)},l(n){t=m(n,"P",{"data-svelte-h":!0}),T(t)!=="svelte-ykxpe4"&&(t.textContent=h),o=r(n),g(d.$$.fragment,n)},m(n,u){p(n,t,u),p(n,o,u),b(d,n,u),M=!0},p:z,i(n){M||(_(d.$$.fragment,n),M=!0)},o(n){y(d.$$.fragment,n),M=!1},d(n){n&&(i(t),i(o)),k(d,n)}}}function Wa(w){let t,h="Example of multi-label classification:",o,d,M;return d=new D({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Rva2VuaXplciUyQyUyMENhbWVtYmVydEZvclNlcXVlbmNlQ2xhc3NpZmljYXRpb24lMEElMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJhbG1hbmFjaCUyRmNhbWVtYmVydC1iYXNlJTIyKSUwQW1vZGVsJTIwJTNEJTIwQ2FtZW1iZXJ0Rm9yU2VxdWVuY2VDbGFzc2lmaWNhdGlvbi5mcm9tX3ByZXRyYWluZWQoJTIyYWxtYW5hY2glMkZjYW1lbWJlcnQtYmFzZSUyMiUyQyUyMHByb2JsZW1fdHlwZSUzRCUyMm11bHRpX2xhYmVsX2NsYXNzaWZpY2F0aW9uJTIyKSUwQSUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplciglMjJIZWxsbyUyQyUyMG15JTIwZG9nJTIwaXMlMjBjdXRlJTIyJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiklMEElMEF3aXRoJTIwdG9yY2gubm9fZ3JhZCgpJTNBJTBBJTIwJTIwJTIwJTIwbG9naXRzJTIwJTNEJTIwbW9kZWwoKippbnB1dHMpLmxvZ2l0cyUwQSUwQXByZWRpY3RlZF9jbGFzc19pZHMlMjAlM0QlMjB0b3JjaC5hcmFuZ2UoMCUyQyUyMGxvZ2l0cy5zaGFwZSU1Qi0xJTVEKSU1QnRvcmNoLnNpZ21vaWQobG9naXRzKS5zcXVlZXplKGRpbSUzRDApJTIwJTNFJTIwMC41JTVEJTBBJTBBJTIzJTIwVG8lMjB0cmFpbiUyMGElMjBtb2RlbCUyMG9uJTIwJTYwbnVtX2xhYmVscyU2MCUyMGNsYXNzZXMlMkMlMjB5b3UlMjBjYW4lMjBwYXNzJTIwJTYwbnVtX2xhYmVscyUzRG51bV9sYWJlbHMlNjAlMjB0byUyMCU2MC5mcm9tX3ByZXRyYWluZWQoLi4uKSU2MCUwQW51bV9sYWJlbHMlMjAlM0QlMjBsZW4obW9kZWwuY29uZmlnLmlkMmxhYmVsKSUwQW1vZGVsJTIwJTNEJTIwQ2FtZW1iZXJ0Rm9yU2VxdWVuY2VDbGFzc2lmaWNhdGlvbi5mcm9tX3ByZXRyYWluZWQoJTBBJTIwJTIwJTIwJTIwJTIyYWxtYW5hY2glMkZjYW1lbWJlcnQtYmFzZSUyMiUyQyUyMG51bV9sYWJlbHMlM0RudW1fbGFiZWxzJTJDJTIwcHJvYmxlbV90eXBlJTNEJTIybXVsdGlfbGFiZWxfY2xhc3NpZmljYXRpb24lMjIlMEEpJTBBJTBBbGFiZWxzJTIwJTNEJTIwdG9yY2guc3VtKCUwQSUyMCUyMCUyMCUyMHRvcmNoLm5uLmZ1bmN0aW9uYWwub25lX2hvdChwcmVkaWN0ZWRfY2xhc3NfaWRzJTVCTm9uZSUyQyUyMCUzQSU1RC5jbG9uZSgpJTJDJTIwbnVtX2NsYXNzZXMlM0RudW1fbGFiZWxzKSUyQyUyMGRpbSUzRDElMEEpLnRvKHRvcmNoLmZsb2F0KSUwQWxvc3MlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyUyQyUyMGxhYmVscyUzRGxhYmVscykubG9zcw==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, CamembertForSequenceClassification

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;almanach/camembert-base&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = CamembertForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;almanach/camembert-base&quot;</span>, problem_type=<span class="hljs-string">&quot;multi_label_classification&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    logits = model(**inputs).logits

<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_class_ids = torch.arange(<span class="hljs-number">0</span>, logits.shape[-<span class="hljs-number">1</span>])[torch.sigmoid(logits).squeeze(dim=<span class="hljs-number">0</span>) &gt; <span class="hljs-number">0.5</span>]

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># To train a model on \`num_labels\` classes, you can pass \`num_labels=num_labels\` to \`.from_pretrained(...)\`</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>num_labels = <span class="hljs-built_in">len</span>(model.config.id2label)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = CamembertForSequenceClassification.from_pretrained(
<span class="hljs-meta">... </span>    <span class="hljs-string">&quot;almanach/camembert-base&quot;</span>, num_labels=num_labels, problem_type=<span class="hljs-string">&quot;multi_label_classification&quot;</span>
<span class="hljs-meta">... </span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.<span class="hljs-built_in">sum</span>(
<span class="hljs-meta">... </span>    torch.nn.functional.one_hot(predicted_class_ids[<span class="hljs-literal">None</span>, :].clone(), num_classes=num_labels), dim=<span class="hljs-number">1</span>
<span class="hljs-meta">... </span>).to(torch.<span class="hljs-built_in">float</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = model(**inputs, labels=labels).loss`,wrap:!1}}),{c(){t=c("p"),t.textContent=h,o=a(),f(d.$$.fragment)},l(n){t=m(n,"P",{"data-svelte-h":!0}),T(t)!=="svelte-1l8e32d"&&(t.textContent=h),o=r(n),g(d.$$.fragment,n)},m(n,u){p(n,t,u),p(n,o,u),b(d,n,u),M=!0},p:z,i(n){M||(_(d.$$.fragment,n),M=!0)},o(n){y(d.$$.fragment,n),M=!1},d(n){n&&(i(t),i(o)),k(d,n)}}}function Ia(w){let t,h=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=h},l(o){t=m(o,"P",{"data-svelte-h":!0}),T(t)!=="svelte-fincs2"&&(t.innerHTML=h)},m(o,d){p(o,t,d)},p:z,d(o){o&&i(t)}}}function Za(w){let t,h="Example:",o,d,M;return d=new D({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBDYW1lbWJlcnRGb3JNdWx0aXBsZUNob2ljZSUwQWltcG9ydCUyMHRvcmNoJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIyYWxtYW5hY2glMkZjYW1lbWJlcnQtYmFzZSUyMiklMEFtb2RlbCUyMCUzRCUyMENhbWVtYmVydEZvck11bHRpcGxlQ2hvaWNlLmZyb21fcHJldHJhaW5lZCglMjJhbG1hbmFjaCUyRmNhbWVtYmVydC1iYXNlJTIyKSUwQSUwQXByb21wdCUyMCUzRCUyMCUyMkluJTIwSXRhbHklMkMlMjBwaXp6YSUyMHNlcnZlZCUyMGluJTIwZm9ybWFsJTIwc2V0dGluZ3MlMkMlMjBzdWNoJTIwYXMlMjBhdCUyMGElMjByZXN0YXVyYW50JTJDJTIwaXMlMjBwcmVzZW50ZWQlMjB1bnNsaWNlZC4lMjIlMEFjaG9pY2UwJTIwJTNEJTIwJTIySXQlMjBpcyUyMGVhdGVuJTIwd2l0aCUyMGElMjBmb3JrJTIwYW5kJTIwYSUyMGtuaWZlLiUyMiUwQWNob2ljZTElMjAlM0QlMjAlMjJJdCUyMGlzJTIwZWF0ZW4lMjB3aGlsZSUyMGhlbGQlMjBpbiUyMHRoZSUyMGhhbmQuJTIyJTBBbGFiZWxzJTIwJTNEJTIwdG9yY2gudGVuc29yKDApLnVuc3F1ZWV6ZSgwKSUyMCUyMCUyMyUyMGNob2ljZTAlMjBpcyUyMGNvcnJlY3QlMjAoYWNjb3JkaW5nJTIwdG8lMjBXaWtpcGVkaWElMjAlM0IpKSUyQyUyMGJhdGNoJTIwc2l6ZSUyMDElMEElMEFlbmNvZGluZyUyMCUzRCUyMHRva2VuaXplciglNUJwcm9tcHQlMkMlMjBwcm9tcHQlNUQlMkMlMjAlNUJjaG9pY2UwJTJDJTIwY2hvaWNlMSU1RCUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIlMkMlMjBwYWRkaW5nJTNEVHJ1ZSklMEFvdXRwdXRzJTIwJTNEJTIwbW9kZWwoKiolN0JrJTNBJTIwdi51bnNxdWVlemUoMCklMjBmb3IlMjBrJTJDJTIwdiUyMGluJTIwZW5jb2RpbmcuaXRlbXMoKSU3RCUyQyUyMGxhYmVscyUzRGxhYmVscyklMjAlMjAlMjMlMjBiYXRjaCUyMHNpemUlMjBpcyUyMDElMEElMEElMjMlMjB0aGUlMjBsaW5lYXIlMjBjbGFzc2lmaWVyJTIwc3RpbGwlMjBuZWVkcyUyMHRvJTIwYmUlMjB0cmFpbmVkJTBBbG9zcyUyMCUzRCUyMG91dHB1dHMubG9zcyUwQWxvZ2l0cyUyMCUzRCUyMG91dHB1dHMubG9naXRz",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, CamembertForMultipleChoice
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;almanach/camembert-base&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = CamembertForMultipleChoice.from_pretrained(<span class="hljs-string">&quot;almanach/camembert-base&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>prompt = <span class="hljs-string">&quot;In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>choice0 = <span class="hljs-string">&quot;It is eaten with a fork and a knife.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>choice1 = <span class="hljs-string">&quot;It is eaten while held in the hand.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.tensor(<span class="hljs-number">0</span>).unsqueeze(<span class="hljs-number">0</span>)  <span class="hljs-comment"># choice0 is correct (according to Wikipedia ;)), batch size 1</span>

<span class="hljs-meta">&gt;&gt;&gt; </span>encoding = tokenizer([prompt, prompt], [choice0, choice1], return_tensors=<span class="hljs-string">&quot;pt&quot;</span>, padding=<span class="hljs-literal">True</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**{k: v.unsqueeze(<span class="hljs-number">0</span>) <span class="hljs-keyword">for</span> k, v <span class="hljs-keyword">in</span> encoding.items()}, labels=labels)  <span class="hljs-comment"># batch size is 1</span>

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># the linear classifier still needs to be trained</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = outputs.loss
<span class="hljs-meta">&gt;&gt;&gt; </span>logits = outputs.logits`,wrap:!1}}),{c(){t=c("p"),t.textContent=h,o=a(),f(d.$$.fragment)},l(n){t=m(n,"P",{"data-svelte-h":!0}),T(t)!=="svelte-11lpom8"&&(t.textContent=h),o=r(n),g(d.$$.fragment,n)},m(n,u){p(n,t,u),p(n,o,u),b(d,n,u),M=!0},p:z,i(n){M||(_(d.$$.fragment,n),M=!0)},o(n){y(d.$$.fragment,n),M=!1},d(n){n&&(i(t),i(o)),k(d,n)}}}function qa(w){let t,h=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=h},l(o){t=m(o,"P",{"data-svelte-h":!0}),T(t)!=="svelte-fincs2"&&(t.innerHTML=h)},m(o,d){p(o,t,d)},p:z,d(o){o&&i(t)}}}function Ba(w){let t,h="Example:",o,d,M;return d=new D({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBDYW1lbWJlcnRGb3JUb2tlbkNsYXNzaWZpY2F0aW9uJTBBaW1wb3J0JTIwdG9yY2glMEElMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJhbG1hbmFjaCUyRmNhbWVtYmVydC1iYXNlJTIyKSUwQW1vZGVsJTIwJTNEJTIwQ2FtZW1iZXJ0Rm9yVG9rZW5DbGFzc2lmaWNhdGlvbi5mcm9tX3ByZXRyYWluZWQoJTIyYWxtYW5hY2glMkZjYW1lbWJlcnQtYmFzZSUyMiklMEElMEFpbnB1dHMlMjAlM0QlMjB0b2tlbml6ZXIoJTBBJTIwJTIwJTIwJTIwJTIySHVnZ2luZ0ZhY2UlMjBpcyUyMGElMjBjb21wYW55JTIwYmFzZWQlMjBpbiUyMFBhcmlzJTIwYW5kJTIwTmV3JTIwWW9yayUyMiUyQyUyMGFkZF9zcGVjaWFsX3Rva2VucyUzREZhbHNlJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiUwQSklMEElMEF3aXRoJTIwdG9yY2gubm9fZ3JhZCgpJTNBJTBBJTIwJTIwJTIwJTIwbG9naXRzJTIwJTNEJTIwbW9kZWwoKippbnB1dHMpLmxvZ2l0cyUwQSUwQXByZWRpY3RlZF90b2tlbl9jbGFzc19pZHMlMjAlM0QlMjBsb2dpdHMuYXJnbWF4KC0xKSUwQSUwQSUyMyUyME5vdGUlMjB0aGF0JTIwdG9rZW5zJTIwYXJlJTIwY2xhc3NpZmllZCUyMHJhdGhlciUyMHRoZW4lMjBpbnB1dCUyMHdvcmRzJTIwd2hpY2glMjBtZWFucyUyMHRoYXQlMEElMjMlMjB0aGVyZSUyMG1pZ2h0JTIwYmUlMjBtb3JlJTIwcHJlZGljdGVkJTIwdG9rZW4lMjBjbGFzc2VzJTIwdGhhbiUyMHdvcmRzLiUwQSUyMyUyME11bHRpcGxlJTIwdG9rZW4lMjBjbGFzc2VzJTIwbWlnaHQlMjBhY2NvdW50JTIwZm9yJTIwdGhlJTIwc2FtZSUyMHdvcmQlMEFwcmVkaWN0ZWRfdG9rZW5zX2NsYXNzZXMlMjAlM0QlMjAlNUJtb2RlbC5jb25maWcuaWQybGFiZWwlNUJ0Lml0ZW0oKSU1RCUyMGZvciUyMHQlMjBpbiUyMHByZWRpY3RlZF90b2tlbl9jbGFzc19pZHMlNUIwJTVEJTVEJTBBcHJlZGljdGVkX3Rva2Vuc19jbGFzc2VzJTBBJTBBbGFiZWxzJTIwJTNEJTIwcHJlZGljdGVkX3Rva2VuX2NsYXNzX2lkcyUwQWxvc3MlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyUyQyUyMGxhYmVscyUzRGxhYmVscykubG9zcyUwQXJvdW5kKGxvc3MuaXRlbSgpJTJDJTIwMik=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, CamembertForTokenClassification
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;almanach/camembert-base&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = CamembertForTokenClassification.from_pretrained(<span class="hljs-string">&quot;almanach/camembert-base&quot;</span>)

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
...`,wrap:!1}}),{c(){t=c("p"),t.textContent=h,o=a(),f(d.$$.fragment)},l(n){t=m(n,"P",{"data-svelte-h":!0}),T(t)!=="svelte-11lpom8"&&(t.textContent=h),o=r(n),g(d.$$.fragment,n)},m(n,u){p(n,t,u),p(n,o,u),b(d,n,u),M=!0},p:z,i(n){M||(_(d.$$.fragment,n),M=!0)},o(n){y(d.$$.fragment,n),M=!1},d(n){n&&(i(t),i(o)),k(d,n)}}}function Na(w){let t,h=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=h},l(o){t=m(o,"P",{"data-svelte-h":!0}),T(t)!=="svelte-fincs2"&&(t.innerHTML=h)},m(o,d){p(o,t,d)},p:z,d(o){o&&i(t)}}}function La(w){let t,h="Example:",o,d,M;return d=new D({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBDYW1lbWJlcnRGb3JRdWVzdGlvbkFuc3dlcmluZyUwQWltcG9ydCUyMHRvcmNoJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIyYWxtYW5hY2glMkZjYW1lbWJlcnQtYmFzZSUyMiklMEFtb2RlbCUyMCUzRCUyMENhbWVtYmVydEZvclF1ZXN0aW9uQW5zd2VyaW5nLmZyb21fcHJldHJhaW5lZCglMjJhbG1hbmFjaCUyRmNhbWVtYmVydC1iYXNlJTIyKSUwQSUwQXF1ZXN0aW9uJTJDJTIwdGV4dCUyMCUzRCUyMCUyMldobyUyMHdhcyUyMEppbSUyMEhlbnNvbiUzRiUyMiUyQyUyMCUyMkppbSUyMEhlbnNvbiUyMHdhcyUyMGElMjBuaWNlJTIwcHVwcGV0JTIyJTBBJTBBaW5wdXRzJTIwJTNEJTIwdG9rZW5pemVyKHF1ZXN0aW9uJTJDJTIwdGV4dCUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpJTBBd2l0aCUyMHRvcmNoLm5vX2dyYWQoKSUzQSUwQSUyMCUyMCUyMCUyMG91dHB1dHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyklMEElMEFhbnN3ZXJfc3RhcnRfaW5kZXglMjAlM0QlMjBvdXRwdXRzLnN0YXJ0X2xvZ2l0cy5hcmdtYXgoKSUwQWFuc3dlcl9lbmRfaW5kZXglMjAlM0QlMjBvdXRwdXRzLmVuZF9sb2dpdHMuYXJnbWF4KCklMEElMEFwcmVkaWN0X2Fuc3dlcl90b2tlbnMlMjAlM0QlMjBpbnB1dHMuaW5wdXRfaWRzJTVCMCUyQyUyMGFuc3dlcl9zdGFydF9pbmRleCUyMCUzQSUyMGFuc3dlcl9lbmRfaW5kZXglMjAlMkIlMjAxJTVEJTBBdG9rZW5pemVyLmRlY29kZShwcmVkaWN0X2Fuc3dlcl90b2tlbnMlMkMlMjBza2lwX3NwZWNpYWxfdG9rZW5zJTNEVHJ1ZSklMEElMEElMjMlMjB0YXJnZXQlMjBpcyUyMCUyMm5pY2UlMjBwdXBwZXQlMjIlMEF0YXJnZXRfc3RhcnRfaW5kZXglMjAlM0QlMjB0b3JjaC50ZW5zb3IoJTVCMTQlNUQpJTBBdGFyZ2V0X2VuZF9pbmRleCUyMCUzRCUyMHRvcmNoLnRlbnNvciglNUIxNSU1RCklMEElMEFvdXRwdXRzJTIwJTNEJTIwbW9kZWwoKippbnB1dHMlMkMlMjBzdGFydF9wb3NpdGlvbnMlM0R0YXJnZXRfc3RhcnRfaW5kZXglMkMlMjBlbmRfcG9zaXRpb25zJTNEdGFyZ2V0X2VuZF9pbmRleCklMEFsb3NzJTIwJTNEJTIwb3V0cHV0cy5sb3NzJTBBcm91bmQobG9zcy5pdGVtKCklMkMlMjAyKQ==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, CamembertForQuestionAnswering
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;almanach/camembert-base&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = CamembertForQuestionAnswering.from_pretrained(<span class="hljs-string">&quot;almanach/camembert-base&quot;</span>)

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
...`,wrap:!1}}),{c(){t=c("p"),t.textContent=h,o=a(),f(d.$$.fragment)},l(n){t=m(n,"P",{"data-svelte-h":!0}),T(t)!=="svelte-11lpom8"&&(t.textContent=h),o=r(n),g(d.$$.fragment,n)},m(n,u){p(n,t,u),p(n,o,u),b(d,n,u),M=!0},p:z,i(n){M||(_(d.$$.fragment,n),M=!0)},o(n){y(d.$$.fragment,n),M=!1},d(n){n&&(i(t),i(o)),k(d,n)}}}function Ra(w){let t,h,o,d,M,n="<em>This model was released on 2019-11-10 and added to Hugging Face Transformers on 2020-11-16.</em>",u,J,yn='<div class="flex flex-wrap space-x-1"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/> <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/></div>',Ne,oe,Mn,Le,_s='<a href="https://huggingface.co/papers/1911.03894" rel="nofollow">CamemBERT</a> is a language model based on <a href="./roberta">RoBERTa</a>, but trained specifically on French text from the OSCAR dataset, making it more effective for French language tasks.',wn,Re,ys="What sets CamemBERT apart is that it learned from a huge, high quality collection of French data, as opposed to mixing lots of languages. This helps it really understand French better than many multilingual models.",vn,Ve,ks="Common applications of CamemBERT include masked language modeling (Fill-mask prediction), text classification (sentiment analysis), token classification (entity recognition) and sentence pair classification (entailment tasks).",Cn,Ge,Ts='You can find all the original CamemBERT checkpoints under the <a href="https://huggingface.co/almanach/models?search=camembert" rel="nofollow">ALMAnaCH</a> organization.',$n,he,Jn,Xe,Ms='The examples below demonstrate how to predict the <code>&lt;mask&gt;</code> token with <a href="/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.Pipeline">Pipeline</a>, <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel">AutoModel</a>, and from the command line.',jn,ue,zn,Ee,ws='Quantization reduces the memory burden of large models by representing weights in lower precision. Refer to the <a href="../quantization/overview">Quantization</a> overview for available options.',xn,He,vs='The example below uses <a href="../quantization/bitsandbytes">bitsandbytes</a> quantization to quantize the weights to 8-bits.',Fn,Ye,Un,Qe,Wn,R,Se,to,xt,Cs=`This is the configuration class to store the configuration of a <a href="/docs/transformers/v4.56.2/en/model_doc/camembert#transformers.CamembertModel">CamembertModel</a> or a <code>TFCamembertModel</code>. It is
used to instantiate a Camembert model according to the specified arguments, defining the model architecture.
Instantiating a configuration with the defaults will yield a similar configuration to that of the Camembert
<a href="https://huggingface.co/almanach/camembert-base" rel="nofollow">almanach/camembert-base</a> architecture.`,no,Ft,$s=`Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> and can be used to control the model outputs. Read the
documentation from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> for more information.`,oo,fe,In,Ae,Zn,j,Pe,so,Ut,Js=`Adapted from <a href="/docs/transformers/v4.56.2/en/model_doc/roberta#transformers.RobertaTokenizer">RobertaTokenizer</a> and <a href="/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.XLNetTokenizer">XLNetTokenizer</a>. Construct a CamemBERT tokenizer. Based on
<a href="https://github.com/google/sentencepiece" rel="nofollow">SentencePiece</a>.`,ao,Wt,js=`This tokenizer inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer">PreTrainedTokenizer</a> which contains most of the main methods. Users should refer to
this superclass for more information regarding those methods.`,ro,se,Oe,io,It,zs=`Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
adding special tokens. An CamemBERT sequence has the following format:`,lo,Zt,xs="<li>single sequence: <code>&lt;s&gt; X &lt;/s&gt;</code></li> <li>pair of sequences: <code>&lt;s&gt; A &lt;/s&gt;&lt;/s&gt; B &lt;/s&gt;</code></li>",co,ge,De,mo,qt,Fs=`Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
special tokens using the tokenizer <code>prepare_for_model</code> method.`,po,be,Ke,ho,Bt,Us=`Create a mask from the two sequences passed to be used in a sequence-pair classification task. CamemBERT, like
RoBERTa, does not make use of token type ids, therefore a list of zeros is returned.`,uo,Nt,et,qn,tt,Bn,F,nt,fo,Lt,Ws=`Construct a “fast” CamemBERT tokenizer (backed by HuggingFace’s <em>tokenizers</em> library). Adapted from
<a href="/docs/transformers/v4.56.2/en/model_doc/roberta#transformers.RobertaTokenizer">RobertaTokenizer</a> and <a href="/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.XLNetTokenizer">XLNetTokenizer</a>. Based on
<a href="https://huggingface.co/docs/tokenizers/python/latest/components.html?highlight=BPE#models" rel="nofollow">BPE</a>.`,go,Rt,Is=`This tokenizer inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast">PreTrainedTokenizerFast</a> which contains most of the main methods. Users should
refer to this superclass for more information regarding those methods.`,bo,ae,ot,_o,Vt,Zs=`Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
adding special tokens. An CamemBERT sequence has the following format:`,yo,Gt,qs="<li>single sequence: <code>&lt;s&gt; X &lt;/s&gt;</code></li> <li>pair of sequences: <code>&lt;s&gt; A &lt;/s&gt;&lt;/s&gt; B &lt;/s&gt;</code></li>",ko,_e,st,To,Xt,Bs=`Create a mask from the two sequences passed to be used in a sequence-pair classification task. CamemBERT, like
RoBERTa, does not make use of token type ids, therefore a list of zeros is returned.`,Nn,at,Ln,U,rt,Mo,Et,Ns="The bare Camembert Model outputting raw hidden-states without any specific head on top.",wo,Ht,Ls=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,vo,Yt,Rs=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Co,re,it,$o,Qt,Vs='The <a href="/docs/transformers/v4.56.2/en/model_doc/camembert#transformers.CamembertModel">CamembertModel</a> forward method, overrides the <code>__call__</code> special method.',Jo,ye,Rn,lt,Vn,W,dt,jo,St,Gs="CamemBERT Model with a <code>language modeling</code> head on top for CLM fine-tuning.",zo,At,Xs=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,xo,Pt,Es=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Fo,Q,ct,Uo,Ot,Hs='The <a href="/docs/transformers/v4.56.2/en/model_doc/camembert#transformers.CamembertForCausalLM">CamembertForCausalLM</a> forward method, overrides the <code>__call__</code> special method.',Wo,ke,Io,Te,Gn,mt,Xn,I,pt,Zo,Dt,Ys="The Camembert Model with a <code>language modeling</code> head on top.”",qo,Kt,Qs=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Bo,en,Ss=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,No,S,ht,Lo,tn,As='The <a href="/docs/transformers/v4.56.2/en/model_doc/camembert#transformers.CamembertForMaskedLM">CamembertForMaskedLM</a> forward method, overrides the <code>__call__</code> special method.',Ro,Me,Vo,we,En,ut,Hn,Z,ft,Go,nn,Ps=`CamemBERT Model transformer with a sequence classification/regression head on top (a linear layer on top of the
pooled output) e.g. for GLUE tasks.`,Xo,on,Os=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Eo,sn,Ds=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Ho,L,gt,Yo,an,Ks='The <a href="/docs/transformers/v4.56.2/en/model_doc/camembert#transformers.CamembertForSequenceClassification">CamembertForSequenceClassification</a> forward method, overrides the <code>__call__</code> special method.',Qo,ve,So,Ce,Ao,$e,Yn,bt,Qn,q,_t,Po,rn,ea=`The Camembert Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
softmax) e.g. for RocStories/SWAG tasks.`,Oo,ln,ta=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Do,dn,na=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Ko,A,yt,es,cn,oa='The <a href="/docs/transformers/v4.56.2/en/model_doc/camembert#transformers.CamembertForMultipleChoice">CamembertForMultipleChoice</a> forward method, overrides the <code>__call__</code> special method.',ts,Je,ns,je,Sn,kt,An,B,Tt,os,mn,sa=`The Camembert transformer with a token classification head on top (a linear layer on top of the hidden-states
output) e.g. for Named-Entity-Recognition (NER) tasks.`,ss,pn,aa=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,as,hn,ra=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,rs,P,Mt,is,un,ia='The <a href="/docs/transformers/v4.56.2/en/model_doc/camembert#transformers.CamembertForTokenClassification">CamembertForTokenClassification</a> forward method, overrides the <code>__call__</code> special method.',ls,ze,ds,xe,Pn,wt,On,N,vt,cs,fn,la=`The Camembert transformer with a span classification head on top for extractive question-answering tasks like
SQuAD (a linear layer on top of the hidden-states output to compute <code>span start logits</code> and <code>span end logits</code>).`,ms,gn,da=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,ps,bn,ca=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,hs,O,Ct,us,_n,ma='The <a href="/docs/transformers/v4.56.2/en/model_doc/camembert#transformers.CamembertForQuestionAnswering">CamembertForQuestionAnswering</a> forward method, overrides the <code>__call__</code> special method.',fs,Fe,gs,Ue,Dn,$t,Kn,kn,eo;return oe=new ne({props:{title:"CamemBERT",local:"camembert",headingTag:"h1"}}),he=new qe({props:{warning:!1,$$slots:{default:[ka]},$$scope:{ctx:w}}}),ue=new ya({props:{id:"usage",options:["Pipeline","AutoModel","transformers CLI"],$$slots:{default:[va]},$$scope:{ctx:w}}}),Ye=new D({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBBdXRvTW9kZWxGb3JNYXNrZWRMTSUyQyUyMEJpdHNBbmRCeXRlc0NvbmZpZyUwQWltcG9ydCUyMHRvcmNoJTBBJTBBcXVhbnRfY29uZmlnJTIwJTNEJTIwQml0c0FuZEJ5dGVzQ29uZmlnKGxvYWRfaW5fOGJpdCUzRFRydWUpJTBBbW9kZWwlMjAlM0QlMjBBdXRvTW9kZWxGb3JNYXNrZWRMTS5mcm9tX3ByZXRyYWluZWQoJTBBJTIwJTIwJTIwJTIwJTIyYWxtYW5hY2glMkZjYW1lbWJlcnQtbGFyZ2UlMjIlMkMlMEElMjAlMjAlMjAlMjBxdWFudGl6YXRpb25fY29uZmlnJTNEcXVhbnRfY29uZmlnJTJDJTBBJTIwJTIwJTIwJTIwZGV2aWNlX21hcCUzRCUyMmF1dG8lMjIlMEEpJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIyYWxtYW5hY2glMkZjYW1lbWJlcnQtbGFyZ2UlMjIpJTBBJTBBaW5wdXRzJTIwJTNEJTIwdG9rZW5pemVyKCUyMkxlJTIwY2FtZW1iZXJ0JTIwZXN0JTIwdW4lMjBkJUMzJUE5bGljaWV1eCUyMGZyb21hZ2UlMjAlM0NtYXNrJTNFLiUyMiUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpLnRvKG1vZGVsLmRldmljZSklMEElMEF3aXRoJTIwdG9yY2gubm9fZ3JhZCgpJTNBJTBBJTIwJTIwJTIwJTIwb3V0cHV0cyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzKSUwQSUyMCUyMCUyMCUyMHByZWRpY3Rpb25zJTIwJTNEJTIwb3V0cHV0cy5sb2dpdHMlMEElMEFtYXNrZWRfaW5kZXglMjAlM0QlMjB0b3JjaC53aGVyZShpbnB1dHMlNUIlMjJpbnB1dF9pZHMlMjIlNUQlMjAlM0QlM0QlMjB0b2tlbml6ZXIubWFza190b2tlbl9pZCklNUIxJTVEJTBBcHJlZGljdGVkX3Rva2VuX2lkJTIwJTNEJTIwcHJlZGljdGlvbnMlNUIwJTJDJTIwbWFza2VkX2luZGV4JTVELmFyZ21heChkaW0lM0QtMSklMEFwcmVkaWN0ZWRfdG9rZW4lMjAlM0QlMjB0b2tlbml6ZXIuZGVjb2RlKHByZWRpY3RlZF90b2tlbl9pZCklMEElMEFwcmludChmJTIyVGhlJTIwcHJlZGljdGVkJTIwdG9rZW4lMjBpcyUzQSUyMCU3QnByZWRpY3RlZF90b2tlbiU3RCUyMik=",highlighted:`<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, AutoModelForMaskedLM, BitsAndBytesConfig
<span class="hljs-keyword">import</span> torch

quant_config = BitsAndBytesConfig(load_in_8bit=<span class="hljs-literal">True</span>)
model = AutoModelForMaskedLM.from_pretrained(
    <span class="hljs-string">&quot;almanach/camembert-large&quot;</span>,
    quantization_config=quant_config,
    device_map=<span class="hljs-string">&quot;auto&quot;</span>
)
tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;almanach/camembert-large&quot;</span>)

inputs = tokenizer(<span class="hljs-string">&quot;Le camembert est un délicieux fromage &lt;mask&gt;.&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>).to(model.device)

<span class="hljs-keyword">with</span> torch.no_grad():
    outputs = model(**inputs)
    predictions = outputs.logits

masked_index = torch.where(inputs[<span class="hljs-string">&quot;input_ids&quot;</span>] == tokenizer.mask_token_id)[<span class="hljs-number">1</span>]
predicted_token_id = predictions[<span class="hljs-number">0</span>, masked_index].argmax(dim=-<span class="hljs-number">1</span>)
predicted_token = tokenizer.decode(predicted_token_id)

<span class="hljs-built_in">print</span>(<span class="hljs-string">f&quot;The predicted token is: <span class="hljs-subst">{predicted_token}</span>&quot;</span>)`,wrap:!1}}),Qe=new ne({props:{title:"CamembertConfig",local:"transformers.CamembertConfig",headingTag:"h2"}}),Se=new $({props:{name:"class transformers.CamembertConfig",anchor:"transformers.CamembertConfig",parameters:[{name:"vocab_size",val:" = 30522"},{name:"hidden_size",val:" = 768"},{name:"num_hidden_layers",val:" = 12"},{name:"num_attention_heads",val:" = 12"},{name:"intermediate_size",val:" = 3072"},{name:"hidden_act",val:" = 'gelu'"},{name:"hidden_dropout_prob",val:" = 0.1"},{name:"attention_probs_dropout_prob",val:" = 0.1"},{name:"max_position_embeddings",val:" = 512"},{name:"type_vocab_size",val:" = 2"},{name:"initializer_range",val:" = 0.02"},{name:"layer_norm_eps",val:" = 1e-12"},{name:"pad_token_id",val:" = 1"},{name:"bos_token_id",val:" = 0"},{name:"eos_token_id",val:" = 2"},{name:"position_embedding_type",val:" = 'absolute'"},{name:"use_cache",val:" = True"},{name:"classifier_dropout",val:" = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.CamembertConfig.vocab_size",description:`<strong>vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 30522) &#x2014;
Vocabulary size of the BERT model. Defines the number of different tokens that can be represented by the
<code>inputs_ids</code> passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/camembert#transformers.CamembertModel">CamembertModel</a> or <code>TFCamembertModel</code>.`,name:"vocab_size"},{anchor:"transformers.CamembertConfig.hidden_size",description:`<strong>hidden_size</strong> (<code>int</code>, <em>optional</em>, defaults to 768) &#x2014;
Dimensionality of the encoder layers and the pooler layer.`,name:"hidden_size"},{anchor:"transformers.CamembertConfig.num_hidden_layers",description:`<strong>num_hidden_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 12) &#x2014;
Number of hidden layers in the Transformer encoder.`,name:"num_hidden_layers"},{anchor:"transformers.CamembertConfig.num_attention_heads",description:`<strong>num_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 12) &#x2014;
Number of attention heads for each attention layer in the Transformer encoder.`,name:"num_attention_heads"},{anchor:"transformers.CamembertConfig.intermediate_size",description:`<strong>intermediate_size</strong> (<code>int</code>, <em>optional</em>, defaults to 3072) &#x2014;
Dimensionality of the &#x201C;intermediate&#x201D; (often named feed-forward) layer in the Transformer encoder.`,name:"intermediate_size"},{anchor:"transformers.CamembertConfig.hidden_act",description:`<strong>hidden_act</strong> (<code>str</code> or <code>Callable</code>, <em>optional</em>, defaults to <code>&quot;gelu&quot;</code>) &#x2014;
The non-linear activation function (function or string) in the encoder and pooler. If string, <code>&quot;gelu&quot;</code>,
<code>&quot;relu&quot;</code>, <code>&quot;silu&quot;</code> and <code>&quot;gelu_new&quot;</code> are supported.`,name:"hidden_act"},{anchor:"transformers.CamembertConfig.hidden_dropout_prob",description:`<strong>hidden_dropout_prob</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.`,name:"hidden_dropout_prob"},{anchor:"transformers.CamembertConfig.attention_probs_dropout_prob",description:`<strong>attention_probs_dropout_prob</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout ratio for the attention probabilities.`,name:"attention_probs_dropout_prob"},{anchor:"transformers.CamembertConfig.max_position_embeddings",description:`<strong>max_position_embeddings</strong> (<code>int</code>, <em>optional</em>, defaults to 512) &#x2014;
The maximum sequence length that this model might ever be used with. Typically set this to something large
just in case (e.g., 512 or 1024 or 2048).`,name:"max_position_embeddings"},{anchor:"transformers.CamembertConfig.type_vocab_size",description:`<strong>type_vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 2) &#x2014;
The vocabulary size of the <code>token_type_ids</code> passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/camembert#transformers.CamembertModel">CamembertModel</a> or <code>TFCamembertModel</code>.`,name:"type_vocab_size"},{anchor:"transformers.CamembertConfig.initializer_range",description:`<strong>initializer_range</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.`,name:"initializer_range"},{anchor:"transformers.CamembertConfig.layer_norm_eps",description:`<strong>layer_norm_eps</strong> (<code>float</code>, <em>optional</em>, defaults to 1e-12) &#x2014;
The epsilon used by the layer normalization layers.`,name:"layer_norm_eps"},{anchor:"transformers.CamembertConfig.position_embedding_type",description:`<strong>position_embedding_type</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;absolute&quot;</code>) &#x2014;
Type of position embedding. Choose one of <code>&quot;absolute&quot;</code>, <code>&quot;relative_key&quot;</code>, <code>&quot;relative_key_query&quot;</code>. For
positional embeddings use <code>&quot;absolute&quot;</code>. For more information on <code>&quot;relative_key&quot;</code>, please refer to
<a href="https://huggingface.co/papers/1803.02155" rel="nofollow">Self-Attention with Relative Position Representations (Shaw et al.)</a>.
For more information on <code>&quot;relative_key_query&quot;</code>, please refer to <em>Method 4</em> in <a href="https://huggingface.co/papers/2009.13658" rel="nofollow">Improve Transformer Models
with Better Relative Position Embeddings (Huang et al.)</a>.`,name:"position_embedding_type"},{anchor:"transformers.CamembertConfig.is_decoder",description:`<strong>is_decoder</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether the model is used as a decoder or not. If <code>False</code>, the model is used as an encoder.`,name:"is_decoder"},{anchor:"transformers.CamembertConfig.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not the model should return the last key/values attentions (not used by all models). Only
relevant if <code>config.is_decoder=True</code>.`,name:"use_cache"},{anchor:"transformers.CamembertConfig.classifier_dropout",description:`<strong>classifier_dropout</strong> (<code>float</code>, <em>optional</em>) &#x2014;
The dropout ratio for the classification head.`,name:"classifier_dropout"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/camembert/configuration_camembert.py#L29"}}),fe=new Be({props:{anchor:"transformers.CamembertConfig.example",$$slots:{default:[Ca]},$$scope:{ctx:w}}}),Ae=new ne({props:{title:"CamembertTokenizer",local:"transformers.CamembertTokenizer",headingTag:"h2"}}),Pe=new $({props:{name:"class transformers.CamembertTokenizer",anchor:"transformers.CamembertTokenizer",parameters:[{name:"vocab_file",val:""},{name:"bos_token",val:" = '<s>'"},{name:"eos_token",val:" = '</s>'"},{name:"sep_token",val:" = '</s>'"},{name:"cls_token",val:" = '<s>'"},{name:"unk_token",val:" = '<unk>'"},{name:"pad_token",val:" = '<pad>'"},{name:"mask_token",val:" = '<mask>'"},{name:"additional_special_tokens",val:" = ['<s>NOTUSED', '</s>NOTUSED', '<unk>NOTUSED']"},{name:"sp_model_kwargs",val:": typing.Optional[dict[str, typing.Any]] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.CamembertTokenizer.vocab_file",description:`<strong>vocab_file</strong> (<code>str</code>) &#x2014;
<a href="https://github.com/google/sentencepiece" rel="nofollow">SentencePiece</a> file (generally has a <em>.spm</em> extension) that
contains the vocabulary necessary to instantiate a tokenizer.`,name:"vocab_file"},{anchor:"transformers.CamembertTokenizer.bos_token",description:`<strong>bos_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;s&gt;&quot;</code>) &#x2014;
The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.</p>
<div class="course-tip  bg-gradient-to-br dark:bg-gradient-to-r before:border-green-500 dark:before:border-green-800 from-green-50 dark:from-gray-900 to-white dark:to-gray-950 border border-green-50 text-green-700 dark:text-gray-400">
						
<p>When building a sequence using special tokens, this is not the token that is used for the beginning of
sequence. The token used is the <code>cls_token</code>.</p>

					</div>`,name:"bos_token"},{anchor:"transformers.CamembertTokenizer.eos_token",description:`<strong>eos_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;/s&gt;&quot;</code>) &#x2014;
The end of sequence token.</p>
<div class="course-tip  bg-gradient-to-br dark:bg-gradient-to-r before:border-green-500 dark:before:border-green-800 from-green-50 dark:from-gray-900 to-white dark:to-gray-950 border border-green-50 text-green-700 dark:text-gray-400">
						
<p>When building a sequence using special tokens, this is not the token that is used for the end of sequence.
The token used is the <code>sep_token</code>.</p>

					</div>`,name:"eos_token"},{anchor:"transformers.CamembertTokenizer.sep_token",description:`<strong>sep_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;/s&gt;&quot;</code>) &#x2014;
The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
sequence classification or for a text and a question for question answering. It is also used as the last
token of a sequence built with special tokens.`,name:"sep_token"},{anchor:"transformers.CamembertTokenizer.cls_token",description:`<strong>cls_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;s&gt;&quot;</code>) &#x2014;
The classifier token which is used when doing sequence classification (classification of the whole sequence
instead of per-token classification). It is the first token of the sequence when built with special tokens.`,name:"cls_token"},{anchor:"transformers.CamembertTokenizer.unk_token",description:`<strong>unk_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;unk&gt;&quot;</code>) &#x2014;
The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
token instead.`,name:"unk_token"},{anchor:"transformers.CamembertTokenizer.pad_token",description:`<strong>pad_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;pad&gt;&quot;</code>) &#x2014;
The token used for padding, for example when batching sequences of different lengths.`,name:"pad_token"},{anchor:"transformers.CamembertTokenizer.mask_token",description:`<strong>mask_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;mask&gt;&quot;</code>) &#x2014;
The token used for masking values. This is the token used when training this model with masked language
modeling. This is the token which the model will try to predict.`,name:"mask_token"},{anchor:"transformers.CamembertTokenizer.additional_special_tokens",description:`<strong>additional_special_tokens</strong> (<code>list[str]</code>, <em>optional</em>, defaults to <code>[&apos;&lt;s&gt;NOTUSED&apos;, &apos;&lt;/s&gt;NOTUSED&apos;, &apos;&lt;unk&gt;NOTUSED&apos;]</code>) &#x2014;
Additional special tokens used by the tokenizer.`,name:"additional_special_tokens"},{anchor:"transformers.CamembertTokenizer.sp_model_kwargs",description:`<strong>sp_model_kwargs</strong> (<code>dict</code>, <em>optional</em>) &#x2014;
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
</ul>`,name:"sp_model_kwargs"},{anchor:"transformers.CamembertTokenizer.sp_model",description:`<strong>sp_model</strong> (<code>SentencePieceProcessor</code>) &#x2014;
The <em>SentencePiece</em> processor that is used for every conversion (string, tokens and IDs).`,name:"sp_model"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/camembert/tokenization_camembert.py#L37"}}),Oe=new $({props:{name:"build_inputs_with_special_tokens",anchor:"transformers.CamembertTokenizer.build_inputs_with_special_tokens",parameters:[{name:"token_ids_0",val:": list"},{name:"token_ids_1",val:": typing.Optional[list[int]] = None"}],parametersDescription:[{anchor:"transformers.CamembertTokenizer.build_inputs_with_special_tokens.token_ids_0",description:`<strong>token_ids_0</strong> (<code>list[int]</code>) &#x2014;
List of IDs to which the special tokens will be added.`,name:"token_ids_0"},{anchor:"transformers.CamembertTokenizer.build_inputs_with_special_tokens.token_ids_1",description:`<strong>token_ids_1</strong> (<code>list[int]</code>, <em>optional</em>) &#x2014;
Optional second list of IDs for sequence pairs.`,name:"token_ids_1"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/camembert/tokenization_camembert.py#L246",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>List of <a href="../glossary#input-ids">input IDs</a> with the appropriate special tokens.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>list[int]</code></p>
`}}),De=new $({props:{name:"get_special_tokens_mask",anchor:"transformers.CamembertTokenizer.get_special_tokens_mask",parameters:[{name:"token_ids_0",val:": list"},{name:"token_ids_1",val:": typing.Optional[list[int]] = None"},{name:"already_has_special_tokens",val:": bool = False"}],parametersDescription:[{anchor:"transformers.CamembertTokenizer.get_special_tokens_mask.token_ids_0",description:`<strong>token_ids_0</strong> (<code>list[int]</code>) &#x2014;
List of IDs.`,name:"token_ids_0"},{anchor:"transformers.CamembertTokenizer.get_special_tokens_mask.token_ids_1",description:`<strong>token_ids_1</strong> (<code>list[int]</code>, <em>optional</em>) &#x2014;
Optional second list of IDs for sequence pairs.`,name:"token_ids_1"},{anchor:"transformers.CamembertTokenizer.get_special_tokens_mask.already_has_special_tokens",description:`<strong>already_has_special_tokens</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not the token list is already formatted with special tokens for the model.`,name:"already_has_special_tokens"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/camembert/tokenization_camembert.py#L272",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>list[int]</code></p>
`}}),Ke=new $({props:{name:"create_token_type_ids_from_sequences",anchor:"transformers.CamembertTokenizer.create_token_type_ids_from_sequences",parameters:[{name:"token_ids_0",val:": list"},{name:"token_ids_1",val:": typing.Optional[list[int]] = None"}],parametersDescription:[{anchor:"transformers.CamembertTokenizer.create_token_type_ids_from_sequences.token_ids_0",description:`<strong>token_ids_0</strong> (<code>list[int]</code>) &#x2014;
List of IDs.`,name:"token_ids_0"},{anchor:"transformers.CamembertTokenizer.create_token_type_ids_from_sequences.token_ids_1",description:`<strong>token_ids_1</strong> (<code>list[int]</code>, <em>optional</em>) &#x2014;
Optional second list of IDs for sequence pairs.`,name:"token_ids_1"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/camembert/tokenization_camembert.py#L299",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>List of zeros.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>list[int]</code></p>
`}}),et=new $({props:{name:"save_vocabulary",anchor:"transformers.CamembertTokenizer.save_vocabulary",parameters:[{name:"save_directory",val:": str"},{name:"filename_prefix",val:": typing.Optional[str] = None"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/camembert/tokenization_camembert.py#L229"}}),tt=new ne({props:{title:"CamembertTokenizerFast",local:"transformers.CamembertTokenizerFast",headingTag:"h2"}}),nt=new $({props:{name:"class transformers.CamembertTokenizerFast",anchor:"transformers.CamembertTokenizerFast",parameters:[{name:"vocab_file",val:" = None"},{name:"tokenizer_file",val:" = None"},{name:"bos_token",val:" = '<s>'"},{name:"eos_token",val:" = '</s>'"},{name:"sep_token",val:" = '</s>'"},{name:"cls_token",val:" = '<s>'"},{name:"unk_token",val:" = '<unk>'"},{name:"pad_token",val:" = '<pad>'"},{name:"mask_token",val:" = '<mask>'"},{name:"additional_special_tokens",val:" = ['<s>NOTUSED', '</s>NOTUSED', '<unk>NOTUSED']"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.CamembertTokenizerFast.vocab_file",description:`<strong>vocab_file</strong> (<code>str</code>) &#x2014;
<a href="https://github.com/google/sentencepiece" rel="nofollow">SentencePiece</a> file (generally has a <em>.spm</em> extension) that
contains the vocabulary necessary to instantiate a tokenizer.`,name:"vocab_file"},{anchor:"transformers.CamembertTokenizerFast.bos_token",description:`<strong>bos_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;s&gt;&quot;</code>) &#x2014;
The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.</p>
<div class="course-tip  bg-gradient-to-br dark:bg-gradient-to-r before:border-green-500 dark:before:border-green-800 from-green-50 dark:from-gray-900 to-white dark:to-gray-950 border border-green-50 text-green-700 dark:text-gray-400">
						
<p>When building a sequence using special tokens, this is not the token that is used for the beginning of
sequence. The token used is the <code>cls_token</code>.</p>

					</div>`,name:"bos_token"},{anchor:"transformers.CamembertTokenizerFast.eos_token",description:`<strong>eos_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;/s&gt;&quot;</code>) &#x2014;
The end of sequence token.</p>
<div class="course-tip  bg-gradient-to-br dark:bg-gradient-to-r before:border-green-500 dark:before:border-green-800 from-green-50 dark:from-gray-900 to-white dark:to-gray-950 border border-green-50 text-green-700 dark:text-gray-400">
						
<p>When building a sequence using special tokens, this is not the token that is used for the end of sequence.
The token used is the <code>sep_token</code>.</p>

					</div>`,name:"eos_token"},{anchor:"transformers.CamembertTokenizerFast.sep_token",description:`<strong>sep_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;/s&gt;&quot;</code>) &#x2014;
The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
sequence classification or for a text and a question for question answering. It is also used as the last
token of a sequence built with special tokens.`,name:"sep_token"},{anchor:"transformers.CamembertTokenizerFast.cls_token",description:`<strong>cls_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;s&gt;&quot;</code>) &#x2014;
The classifier token which is used when doing sequence classification (classification of the whole sequence
instead of per-token classification). It is the first token of the sequence when built with special tokens.`,name:"cls_token"},{anchor:"transformers.CamembertTokenizerFast.unk_token",description:`<strong>unk_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;unk&gt;&quot;</code>) &#x2014;
The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
token instead.`,name:"unk_token"},{anchor:"transformers.CamembertTokenizerFast.pad_token",description:`<strong>pad_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;pad&gt;&quot;</code>) &#x2014;
The token used for padding, for example when batching sequences of different lengths.`,name:"pad_token"},{anchor:"transformers.CamembertTokenizerFast.mask_token",description:`<strong>mask_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;mask&gt;&quot;</code>) &#x2014;
The token used for masking values. This is the token used when training this model with masked language
modeling. This is the token which the model will try to predict.`,name:"mask_token"},{anchor:"transformers.CamembertTokenizerFast.additional_special_tokens",description:`<strong>additional_special_tokens</strong> (<code>list[str]</code>, <em>optional</em>, defaults to <code>[&quot;&lt;s&gt;NOTUSED&quot;, &quot;&lt;/s&gt;NOTUSED&quot;]</code>) &#x2014;
Additional special tokens used by the tokenizer.`,name:"additional_special_tokens"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/camembert/tokenization_camembert_fast.py#L40"}}),ot=new $({props:{name:"build_inputs_with_special_tokens",anchor:"transformers.CamembertTokenizerFast.build_inputs_with_special_tokens",parameters:[{name:"token_ids_0",val:": list"},{name:"token_ids_1",val:": typing.Optional[list[int]] = None"}],parametersDescription:[{anchor:"transformers.CamembertTokenizerFast.build_inputs_with_special_tokens.token_ids_0",description:`<strong>token_ids_0</strong> (<code>list[int]</code>) &#x2014;
List of IDs to which the special tokens will be added.`,name:"token_ids_0"},{anchor:"transformers.CamembertTokenizerFast.build_inputs_with_special_tokens.token_ids_1",description:`<strong>token_ids_1</strong> (<code>list[int]</code>, <em>optional</em>) &#x2014;
Optional second list of IDs for sequence pairs.`,name:"token_ids_1"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/camembert/tokenization_camembert_fast.py#L128",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>List of <a href="../glossary#input-ids">input IDs</a> with the appropriate special tokens.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>list[int]</code></p>
`}}),st=new $({props:{name:"create_token_type_ids_from_sequences",anchor:"transformers.CamembertTokenizerFast.create_token_type_ids_from_sequences",parameters:[{name:"token_ids_0",val:": list"},{name:"token_ids_1",val:": typing.Optional[list[int]] = None"}],parametersDescription:[{anchor:"transformers.CamembertTokenizerFast.create_token_type_ids_from_sequences.token_ids_0",description:`<strong>token_ids_0</strong> (<code>list[int]</code>) &#x2014;
List of IDs.`,name:"token_ids_0"},{anchor:"transformers.CamembertTokenizerFast.create_token_type_ids_from_sequences.token_ids_1",description:`<strong>token_ids_1</strong> (<code>list[int]</code>, <em>optional</em>) &#x2014;
Optional second list of IDs for sequence pairs.`,name:"token_ids_1"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/camembert/tokenization_camembert_fast.py#L154",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>List of zeros.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>list[int]</code></p>
`}}),at=new ne({props:{title:"CamembertModel",local:"transformers.CamembertModel",headingTag:"h2"}}),rt=new $({props:{name:"class transformers.CamembertModel",anchor:"transformers.CamembertModel",parameters:[{name:"config",val:""},{name:"add_pooling_layer",val:" = True"}],parametersDescription:[{anchor:"transformers.CamembertModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/camembert#transformers.CamembertModel">CamembertModel</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"},{anchor:"transformers.CamembertModel.add_pooling_layer",description:`<strong>add_pooling_layer</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether to add a pooling layer`,name:"add_pooling_layer"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/camembert/modeling_camembert.py#L746"}}),it=new $({props:{name:"forward",anchor:"transformers.CamembertModel.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_hidden_states",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"past_key_values",val:": typing.Optional[list[torch.FloatTensor]] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.Tensor] = None"}],parametersDescription:[{anchor:"transformers.CamembertModel.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.CamembertModel.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.CamembertModel.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.CamembertModel.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.CamembertModel.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.CamembertModel.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.CamembertModel.forward.encoder_hidden_states",description:`<strong>encoder_hidden_states</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
if the model is configured as a decoder.`,name:"encoder_hidden_states"},{anchor:"transformers.CamembertModel.forward.encoder_attention_mask",description:`<strong>encoder_attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
the cross-attention if the model is configured as a decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>`,name:"encoder_attention_mask"},{anchor:"transformers.CamembertModel.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>list[torch.FloatTensor]</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.CamembertModel.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.CamembertModel.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.CamembertModel.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.CamembertModel.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.CamembertModel.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.Tensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/camembert/modeling_camembert.py#L798",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions"
>transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/camembert#transformers.CamembertConfig"
>CamembertConfig</a>) and inputs.</p>
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
`}}),ye=new qe({props:{$$slots:{default:[$a]},$$scope:{ctx:w}}}),lt=new ne({props:{title:"CamembertForCausalLM",local:"transformers.CamembertForCausalLM",headingTag:"h2"}}),dt=new $({props:{name:"class transformers.CamembertForCausalLM",anchor:"transformers.CamembertForCausalLM",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.CamembertForCausalLM.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/camembert#transformers.CamembertForCausalLM">CamembertForCausalLM</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/camembert/modeling_camembert.py#L1430"}}),ct=new $({props:{name:"forward",anchor:"transformers.CamembertForCausalLM.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"encoder_hidden_states",val:": typing.Optional[torch.FloatTensor] = None"},{name:"encoder_attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[tuple[tuple[torch.FloatTensor]]] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.CamembertForCausalLM.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.CamembertForCausalLM.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.CamembertForCausalLM.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0,1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.
This parameter can only be used when the model is initialized with <code>type_vocab_size</code> parameter with value<blockquote>
<p>= 2. All the value in this tensor should be always &lt; type_vocab_size.</p>
</blockquote></li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.CamembertForCausalLM.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.CamembertForCausalLM.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.CamembertForCausalLM.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.CamembertForCausalLM.forward.encoder_hidden_states",description:`<strong>encoder_hidden_states</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
if the model is configured as a decoder.`,name:"encoder_hidden_states"},{anchor:"transformers.CamembertForCausalLM.forward.encoder_attention_mask",description:`<strong>encoder_attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
the cross-attention if the model is configured as a decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>`,name:"encoder_attention_mask"},{anchor:"transformers.CamembertForCausalLM.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the left-to-right language modeling loss (next word prediction). Indices should be in
<code>[-100, 0, ..., config.vocab_size]</code> (see <code>input_ids</code> docstring) Tokens with indices set to <code>-100</code> are
ignored (masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>`,name:"labels"},{anchor:"transformers.CamembertForCausalLM.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>tuple[tuple[torch.FloatTensor]]</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.CamembertForCausalLM.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.CamembertForCausalLM.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.CamembertForCausalLM.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.CamembertForCausalLM.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/camembert/modeling_camembert.py#L1451",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithCrossAttentions"
>transformers.modeling_outputs.CausalLMOutputWithCrossAttentions</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/camembert#transformers.CamembertConfig"
>CamembertConfig</a>) and inputs.</p>
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
`}}),ke=new qe({props:{$$slots:{default:[Ja]},$$scope:{ctx:w}}}),Te=new Be({props:{anchor:"transformers.CamembertForCausalLM.forward.example",$$slots:{default:[ja]},$$scope:{ctx:w}}}),mt=new ne({props:{title:"CamembertForMaskedLM",local:"transformers.CamembertForMaskedLM",headingTag:"h2"}}),pt=new $({props:{name:"class transformers.CamembertForMaskedLM",anchor:"transformers.CamembertForMaskedLM",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.CamembertForMaskedLM.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/camembert#transformers.CamembertForMaskedLM">CamembertForMaskedLM</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/camembert/modeling_camembert.py#L952"}}),ht=new $({props:{name:"forward",anchor:"transformers.CamembertForMaskedLM.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"encoder_hidden_states",val:": typing.Optional[torch.FloatTensor] = None"},{name:"encoder_attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.CamembertForMaskedLM.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.CamembertForMaskedLM.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.CamembertForMaskedLM.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0,1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.
This parameter can only be used when the model is initialized with <code>type_vocab_size</code> parameter with value<blockquote>
<p>= 2. All the value in this tensor should be always &lt; type_vocab_size.</p>
</blockquote></li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.CamembertForMaskedLM.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.CamembertForMaskedLM.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.CamembertForMaskedLM.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.CamembertForMaskedLM.forward.encoder_hidden_states",description:`<strong>encoder_hidden_states</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
if the model is configured as a decoder.`,name:"encoder_hidden_states"},{anchor:"transformers.CamembertForMaskedLM.forward.encoder_attention_mask",description:`<strong>encoder_attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
the cross-attention if the model is configured as a decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>`,name:"encoder_attention_mask"},{anchor:"transformers.CamembertForMaskedLM.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should be in <code>[-100, 0, ..., config.vocab_size]</code> (see <code>input_ids</code> docstring) Tokens with indices set to <code>-100</code> are ignored (masked), the
loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>`,name:"labels"},{anchor:"transformers.CamembertForMaskedLM.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.CamembertForMaskedLM.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.CamembertForMaskedLM.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/camembert/modeling_camembert.py#L976",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.MaskedLMOutput"
>transformers.modeling_outputs.MaskedLMOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/camembert#transformers.CamembertConfig"
>CamembertConfig</a>) and inputs.</p>
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
`}}),Me=new qe({props:{$$slots:{default:[za]},$$scope:{ctx:w}}}),we=new Be({props:{anchor:"transformers.CamembertForMaskedLM.forward.example",$$slots:{default:[xa]},$$scope:{ctx:w}}}),ut=new ne({props:{title:"CamembertForSequenceClassification",local:"transformers.CamembertForSequenceClassification",headingTag:"h2"}}),ft=new $({props:{name:"class transformers.CamembertForSequenceClassification",anchor:"transformers.CamembertForSequenceClassification",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.CamembertForSequenceClassification.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/camembert#transformers.CamembertForSequenceClassification">CamembertForSequenceClassification</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/camembert/modeling_camembert.py#L1051"}}),gt=new $({props:{name:"forward",anchor:"transformers.CamembertForSequenceClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.CamembertForSequenceClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.CamembertForSequenceClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.CamembertForSequenceClassification.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0,1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.
This parameter can only be used when the model is initialized with <code>type_vocab_size</code> parameter with value<blockquote>
<p>= 2. All the value in this tensor should be always &lt; type_vocab_size.</p>
</blockquote></li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.CamembertForSequenceClassification.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.CamembertForSequenceClassification.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.CamembertForSequenceClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.CamembertForSequenceClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the sequence classification/regression loss. Indices should be in <code>[0, ..., config.num_labels - 1]</code>. If <code>config.num_labels == 1</code> a regression loss is computed (Mean-Square loss), If
<code>config.num_labels &gt; 1</code> a classification loss is computed (Cross-Entropy).`,name:"labels"},{anchor:"transformers.CamembertForSequenceClassification.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.CamembertForSequenceClassification.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.CamembertForSequenceClassification.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/camembert/modeling_camembert.py#L1063",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput"
>transformers.modeling_outputs.SequenceClassifierOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/camembert#transformers.CamembertConfig"
>CamembertConfig</a>) and inputs.</p>
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
`}}),ve=new qe({props:{$$slots:{default:[Fa]},$$scope:{ctx:w}}}),Ce=new Be({props:{anchor:"transformers.CamembertForSequenceClassification.forward.example",$$slots:{default:[Ua]},$$scope:{ctx:w}}}),$e=new Be({props:{anchor:"transformers.CamembertForSequenceClassification.forward.example-2",$$slots:{default:[Wa]},$$scope:{ctx:w}}}),bt=new ne({props:{title:"CamembertForMultipleChoice",local:"transformers.CamembertForMultipleChoice",headingTag:"h2"}}),_t=new $({props:{name:"class transformers.CamembertForMultipleChoice",anchor:"transformers.CamembertForMultipleChoice",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.CamembertForMultipleChoice.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/camembert#transformers.CamembertForMultipleChoice">CamembertForMultipleChoice</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/camembert/modeling_camembert.py#L1147"}}),yt=new $({props:{name:"forward",anchor:"transformers.CamembertForMultipleChoice.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.CamembertForMultipleChoice.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, num_choices, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.CamembertForMultipleChoice.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, num_choices, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0,1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.
This parameter can only be used when the model is initialized with <code>type_vocab_size</code> parameter with value<blockquote>
<p>= 2. All the value in this tensor should be always &lt; type_vocab_size.</p>
</blockquote></li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.CamembertForMultipleChoice.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.CamembertForMultipleChoice.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the multiple choice classification loss. Indices should be in <code>[0, ..., num_choices-1]</code> where <code>num_choices</code> is the size of the second dimension of the input tensors. (See
<code>input_ids</code> above)`,name:"labels"},{anchor:"transformers.CamembertForMultipleChoice.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, num_choices, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.max_position_embeddings - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.CamembertForMultipleChoice.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.CamembertForMultipleChoice.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_choices, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.CamembertForMultipleChoice.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.CamembertForMultipleChoice.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.CamembertForMultipleChoice.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/camembert/modeling_camembert.py#L1158",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.MultipleChoiceModelOutput"
>transformers.modeling_outputs.MultipleChoiceModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/camembert#transformers.CamembertConfig"
>CamembertConfig</a>) and inputs.</p>
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
`}}),Je=new qe({props:{$$slots:{default:[Ia]},$$scope:{ctx:w}}}),je=new Be({props:{anchor:"transformers.CamembertForMultipleChoice.forward.example",$$slots:{default:[Za]},$$scope:{ctx:w}}}),kt=new ne({props:{title:"CamembertForTokenClassification",local:"transformers.CamembertForTokenClassification",headingTag:"h2"}}),Tt=new $({props:{name:"class transformers.CamembertForTokenClassification",anchor:"transformers.CamembertForTokenClassification",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.CamembertForTokenClassification.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/camembert#transformers.CamembertForTokenClassification">CamembertForTokenClassification</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/camembert/modeling_camembert.py#L1254"}}),Mt=new $({props:{name:"forward",anchor:"transformers.CamembertForTokenClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.CamembertForTokenClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.CamembertForTokenClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.CamembertForTokenClassification.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0,1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.
This parameter can only be used when the model is initialized with <code>type_vocab_size</code> parameter with value<blockquote>
<p>= 2. All the value in this tensor should be always &lt; type_vocab_size.</p>
</blockquote></li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.CamembertForTokenClassification.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.CamembertForTokenClassification.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.CamembertForTokenClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.CamembertForTokenClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the token classification loss. Indices should be in <code>[0, ..., config.num_labels - 1]</code>.`,name:"labels"},{anchor:"transformers.CamembertForTokenClassification.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.CamembertForTokenClassification.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.CamembertForTokenClassification.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/camembert/modeling_camembert.py#L1269",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput"
>transformers.modeling_outputs.TokenClassifierOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/camembert#transformers.CamembertConfig"
>CamembertConfig</a>) and inputs.</p>
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
`}}),ze=new qe({props:{$$slots:{default:[qa]},$$scope:{ctx:w}}}),xe=new Be({props:{anchor:"transformers.CamembertForTokenClassification.forward.example",$$slots:{default:[Ba]},$$scope:{ctx:w}}}),wt=new ne({props:{title:"CamembertForQuestionAnswering",local:"transformers.CamembertForQuestionAnswering",headingTag:"h2"}}),vt=new $({props:{name:"class transformers.CamembertForQuestionAnswering",anchor:"transformers.CamembertForQuestionAnswering",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.CamembertForQuestionAnswering.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/camembert#transformers.CamembertForQuestionAnswering">CamembertForQuestionAnswering</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/camembert/modeling_camembert.py#L1336"}}),Ct=new $({props:{name:"forward",anchor:"transformers.CamembertForQuestionAnswering.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"start_positions",val:": typing.Optional[torch.LongTensor] = None"},{name:"end_positions",val:": typing.Optional[torch.LongTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.CamembertForQuestionAnswering.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.CamembertForQuestionAnswering.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.CamembertForQuestionAnswering.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0,1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.
This parameter can only be used when the model is initialized with <code>type_vocab_size</code> parameter with value<blockquote>
<p>= 2. All the value in this tensor should be always &lt; type_vocab_size.</p>
</blockquote></li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.CamembertForQuestionAnswering.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.CamembertForQuestionAnswering.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.CamembertForQuestionAnswering.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.CamembertForQuestionAnswering.forward.start_positions",description:`<strong>start_positions</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for position (index) of the start of the labelled span for computing the token classification loss.
Positions are clamped to the length of the sequence (<code>sequence_length</code>). Position outside of the sequence
are not taken into account for computing the loss.`,name:"start_positions"},{anchor:"transformers.CamembertForQuestionAnswering.forward.end_positions",description:`<strong>end_positions</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for position (index) of the end of the labelled span for computing the token classification loss.
Positions are clamped to the length of the sequence (<code>sequence_length</code>). Position outside of the sequence
are not taken into account for computing the loss.`,name:"end_positions"},{anchor:"transformers.CamembertForQuestionAnswering.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.CamembertForQuestionAnswering.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.CamembertForQuestionAnswering.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/camembert/modeling_camembert.py#L1347",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.QuestionAnsweringModelOutput"
>transformers.modeling_outputs.QuestionAnsweringModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/camembert#transformers.CamembertConfig"
>CamembertConfig</a>) and inputs.</p>
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
`}}),Fe=new qe({props:{$$slots:{default:[Na]},$$scope:{ctx:w}}}),Ue=new Be({props:{anchor:"transformers.CamembertForQuestionAnswering.forward.example",$$slots:{default:[La]},$$scope:{ctx:w}}}),$t=new _a({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/camembert.md"}}),{c(){t=c("meta"),h=a(),o=c("p"),d=a(),M=c("p"),M.innerHTML=n,u=a(),J=c("div"),J.innerHTML=yn,Ne=a(),f(oe.$$.fragment),Mn=a(),Le=c("p"),Le.innerHTML=_s,wn=a(),Re=c("p"),Re.textContent=ys,vn=a(),Ve=c("p"),Ve.textContent=ks,Cn=a(),Ge=c("p"),Ge.innerHTML=Ts,$n=a(),f(he.$$.fragment),Jn=a(),Xe=c("p"),Xe.innerHTML=Ms,jn=a(),f(ue.$$.fragment),zn=a(),Ee=c("p"),Ee.innerHTML=ws,xn=a(),He=c("p"),He.innerHTML=vs,Fn=a(),f(Ye.$$.fragment),Un=a(),f(Qe.$$.fragment),Wn=a(),R=c("div"),f(Se.$$.fragment),to=a(),xt=c("p"),xt.innerHTML=Cs,no=a(),Ft=c("p"),Ft.innerHTML=$s,oo=a(),f(fe.$$.fragment),In=a(),f(Ae.$$.fragment),Zn=a(),j=c("div"),f(Pe.$$.fragment),so=a(),Ut=c("p"),Ut.innerHTML=Js,ao=a(),Wt=c("p"),Wt.innerHTML=js,ro=a(),se=c("div"),f(Oe.$$.fragment),io=a(),It=c("p"),It.textContent=zs,lo=a(),Zt=c("ul"),Zt.innerHTML=xs,co=a(),ge=c("div"),f(De.$$.fragment),mo=a(),qt=c("p"),qt.innerHTML=Fs,po=a(),be=c("div"),f(Ke.$$.fragment),ho=a(),Bt=c("p"),Bt.textContent=Us,uo=a(),Nt=c("div"),f(et.$$.fragment),qn=a(),f(tt.$$.fragment),Bn=a(),F=c("div"),f(nt.$$.fragment),fo=a(),Lt=c("p"),Lt.innerHTML=Ws,go=a(),Rt=c("p"),Rt.innerHTML=Is,bo=a(),ae=c("div"),f(ot.$$.fragment),_o=a(),Vt=c("p"),Vt.textContent=Zs,yo=a(),Gt=c("ul"),Gt.innerHTML=qs,ko=a(),_e=c("div"),f(st.$$.fragment),To=a(),Xt=c("p"),Xt.textContent=Bs,Nn=a(),f(at.$$.fragment),Ln=a(),U=c("div"),f(rt.$$.fragment),Mo=a(),Et=c("p"),Et.textContent=Ns,wo=a(),Ht=c("p"),Ht.innerHTML=Ls,vo=a(),Yt=c("p"),Yt.innerHTML=Rs,Co=a(),re=c("div"),f(it.$$.fragment),$o=a(),Qt=c("p"),Qt.innerHTML=Vs,Jo=a(),f(ye.$$.fragment),Rn=a(),f(lt.$$.fragment),Vn=a(),W=c("div"),f(dt.$$.fragment),jo=a(),St=c("p"),St.innerHTML=Gs,zo=a(),At=c("p"),At.innerHTML=Xs,xo=a(),Pt=c("p"),Pt.innerHTML=Es,Fo=a(),Q=c("div"),f(ct.$$.fragment),Uo=a(),Ot=c("p"),Ot.innerHTML=Hs,Wo=a(),f(ke.$$.fragment),Io=a(),f(Te.$$.fragment),Gn=a(),f(mt.$$.fragment),Xn=a(),I=c("div"),f(pt.$$.fragment),Zo=a(),Dt=c("p"),Dt.innerHTML=Ys,qo=a(),Kt=c("p"),Kt.innerHTML=Qs,Bo=a(),en=c("p"),en.innerHTML=Ss,No=a(),S=c("div"),f(ht.$$.fragment),Lo=a(),tn=c("p"),tn.innerHTML=As,Ro=a(),f(Me.$$.fragment),Vo=a(),f(we.$$.fragment),En=a(),f(ut.$$.fragment),Hn=a(),Z=c("div"),f(ft.$$.fragment),Go=a(),nn=c("p"),nn.textContent=Ps,Xo=a(),on=c("p"),on.innerHTML=Os,Eo=a(),sn=c("p"),sn.innerHTML=Ds,Ho=a(),L=c("div"),f(gt.$$.fragment),Yo=a(),an=c("p"),an.innerHTML=Ks,Qo=a(),f(ve.$$.fragment),So=a(),f(Ce.$$.fragment),Ao=a(),f($e.$$.fragment),Yn=a(),f(bt.$$.fragment),Qn=a(),q=c("div"),f(_t.$$.fragment),Po=a(),rn=c("p"),rn.textContent=ea,Oo=a(),ln=c("p"),ln.innerHTML=ta,Do=a(),dn=c("p"),dn.innerHTML=na,Ko=a(),A=c("div"),f(yt.$$.fragment),es=a(),cn=c("p"),cn.innerHTML=oa,ts=a(),f(Je.$$.fragment),ns=a(),f(je.$$.fragment),Sn=a(),f(kt.$$.fragment),An=a(),B=c("div"),f(Tt.$$.fragment),os=a(),mn=c("p"),mn.textContent=sa,ss=a(),pn=c("p"),pn.innerHTML=aa,as=a(),hn=c("p"),hn.innerHTML=ra,rs=a(),P=c("div"),f(Mt.$$.fragment),is=a(),un=c("p"),un.innerHTML=ia,ls=a(),f(ze.$$.fragment),ds=a(),f(xe.$$.fragment),Pn=a(),f(wt.$$.fragment),On=a(),N=c("div"),f(vt.$$.fragment),cs=a(),fn=c("p"),fn.innerHTML=la,ms=a(),gn=c("p"),gn.innerHTML=da,ps=a(),bn=c("p"),bn.innerHTML=ca,hs=a(),O=c("div"),f(Ct.$$.fragment),us=a(),_n=c("p"),_n.innerHTML=ma,fs=a(),f(Fe.$$.fragment),gs=a(),f(Ue.$$.fragment),Dn=a(),f($t.$$.fragment),Kn=a(),kn=c("p"),this.h()},l(e){const l=ga("svelte-u9bgzb",document.head);t=m(l,"META",{name:!0,content:!0}),l.forEach(i),h=r(e),o=m(e,"P",{}),v(o).forEach(i),d=r(e),M=m(e,"P",{"data-svelte-h":!0}),T(M)!=="svelte-h5ugc2"&&(M.innerHTML=n),u=r(e),J=m(e,"DIV",{style:!0,"data-svelte-h":!0}),T(J)!=="svelte-vec1mn"&&(J.innerHTML=yn),Ne=r(e),g(oe.$$.fragment,e),Mn=r(e),Le=m(e,"P",{"data-svelte-h":!0}),T(Le)!=="svelte-1olh2s3"&&(Le.innerHTML=_s),wn=r(e),Re=m(e,"P",{"data-svelte-h":!0}),T(Re)!=="svelte-tcahta"&&(Re.textContent=ys),vn=r(e),Ve=m(e,"P",{"data-svelte-h":!0}),T(Ve)!=="svelte-wpordk"&&(Ve.textContent=ks),Cn=r(e),Ge=m(e,"P",{"data-svelte-h":!0}),T(Ge)!=="svelte-1kuvgaq"&&(Ge.innerHTML=Ts),$n=r(e),g(he.$$.fragment,e),Jn=r(e),Xe=m(e,"P",{"data-svelte-h":!0}),T(Xe)!=="svelte-4ucayk"&&(Xe.innerHTML=Ms),jn=r(e),g(ue.$$.fragment,e),zn=r(e),Ee=m(e,"P",{"data-svelte-h":!0}),T(Ee)!=="svelte-1hehvth"&&(Ee.innerHTML=ws),xn=r(e),He=m(e,"P",{"data-svelte-h":!0}),T(He)!=="svelte-8pvdld"&&(He.innerHTML=vs),Fn=r(e),g(Ye.$$.fragment,e),Un=r(e),g(Qe.$$.fragment,e),Wn=r(e),R=m(e,"DIV",{class:!0});var K=v(R);g(Se.$$.fragment,K),to=r(K),xt=m(K,"P",{"data-svelte-h":!0}),T(xt)!=="svelte-1pj6ou5"&&(xt.innerHTML=Cs),no=r(K),Ft=m(K,"P",{"data-svelte-h":!0}),T(Ft)!=="svelte-1ek1ss9"&&(Ft.innerHTML=$s),oo=r(K),g(fe.$$.fragment,K),K.forEach(i),In=r(e),g(Ae.$$.fragment,e),Zn=r(e),j=m(e,"DIV",{class:!0});var x=v(j);g(Pe.$$.fragment,x),so=r(x),Ut=m(x,"P",{"data-svelte-h":!0}),T(Ut)!=="svelte-1xouu4q"&&(Ut.innerHTML=Js),ao=r(x),Wt=m(x,"P",{"data-svelte-h":!0}),T(Wt)!=="svelte-ntrhio"&&(Wt.innerHTML=js),ro=r(x),se=m(x,"DIV",{class:!0});var ce=v(se);g(Oe.$$.fragment,ce),io=r(ce),It=m(ce,"P",{"data-svelte-h":!0}),T(It)!=="svelte-dvyskh"&&(It.textContent=zs),lo=r(ce),Zt=m(ce,"UL",{"data-svelte-h":!0}),T(Zt)!=="svelte-rq8uot"&&(Zt.innerHTML=xs),ce.forEach(i),co=r(x),ge=m(x,"DIV",{class:!0});var Jt=v(ge);g(De.$$.fragment,Jt),mo=r(Jt),qt=m(Jt,"P",{"data-svelte-h":!0}),T(qt)!=="svelte-1f4f5kp"&&(qt.innerHTML=Fs),Jt.forEach(i),po=r(x),be=m(x,"DIV",{class:!0});var jt=v(be);g(Ke.$$.fragment,jt),ho=r(jt),Bt=m(jt,"P",{"data-svelte-h":!0}),T(Bt)!=="svelte-tyio8x"&&(Bt.textContent=Us),jt.forEach(i),uo=r(x),Nt=m(x,"DIV",{class:!0});var Tn=v(Nt);g(et.$$.fragment,Tn),Tn.forEach(i),x.forEach(i),qn=r(e),g(tt.$$.fragment,e),Bn=r(e),F=m(e,"DIV",{class:!0});var V=v(F);g(nt.$$.fragment,V),fo=r(V),Lt=m(V,"P",{"data-svelte-h":!0}),T(Lt)!=="svelte-v64lnf"&&(Lt.innerHTML=Ws),go=r(V),Rt=m(V,"P",{"data-svelte-h":!0}),T(Rt)!=="svelte-gxzj9w"&&(Rt.innerHTML=Is),bo=r(V),ae=m(V,"DIV",{class:!0});var me=v(ae);g(ot.$$.fragment,me),_o=r(me),Vt=m(me,"P",{"data-svelte-h":!0}),T(Vt)!=="svelte-dvyskh"&&(Vt.textContent=Zs),yo=r(me),Gt=m(me,"UL",{"data-svelte-h":!0}),T(Gt)!=="svelte-rq8uot"&&(Gt.innerHTML=qs),me.forEach(i),ko=r(V),_e=m(V,"DIV",{class:!0});var zt=v(_e);g(st.$$.fragment,zt),To=r(zt),Xt=m(zt,"P",{"data-svelte-h":!0}),T(Xt)!=="svelte-tyio8x"&&(Xt.textContent=Bs),zt.forEach(i),V.forEach(i),Nn=r(e),g(at.$$.fragment,e),Ln=r(e),U=m(e,"DIV",{class:!0});var G=v(U);g(rt.$$.fragment,G),Mo=r(G),Et=m(G,"P",{"data-svelte-h":!0}),T(Et)!=="svelte-unhw2o"&&(Et.textContent=Ns),wo=r(G),Ht=m(G,"P",{"data-svelte-h":!0}),T(Ht)!=="svelte-q52n56"&&(Ht.innerHTML=Ls),vo=r(G),Yt=m(G,"P",{"data-svelte-h":!0}),T(Yt)!=="svelte-hswkmf"&&(Yt.innerHTML=Rs),Co=r(G),re=m(G,"DIV",{class:!0});var pe=v(re);g(it.$$.fragment,pe),$o=r(pe),Qt=m(pe,"P",{"data-svelte-h":!0}),T(Qt)!=="svelte-1q8yq1r"&&(Qt.innerHTML=Vs),Jo=r(pe),g(ye.$$.fragment,pe),pe.forEach(i),G.forEach(i),Rn=r(e),g(lt.$$.fragment,e),Vn=r(e),W=m(e,"DIV",{class:!0});var X=v(W);g(dt.$$.fragment,X),jo=r(X),St=m(X,"P",{"data-svelte-h":!0}),T(St)!=="svelte-jz3u0f"&&(St.innerHTML=Gs),zo=r(X),At=m(X,"P",{"data-svelte-h":!0}),T(At)!=="svelte-q52n56"&&(At.innerHTML=Xs),xo=r(X),Pt=m(X,"P",{"data-svelte-h":!0}),T(Pt)!=="svelte-hswkmf"&&(Pt.innerHTML=Es),Fo=r(X),Q=m(X,"DIV",{class:!0});var ee=v(Q);g(ct.$$.fragment,ee),Uo=r(ee),Ot=m(ee,"P",{"data-svelte-h":!0}),T(Ot)!=="svelte-hukx0z"&&(Ot.innerHTML=Hs),Wo=r(ee),g(ke.$$.fragment,ee),Io=r(ee),g(Te.$$.fragment,ee),ee.forEach(i),X.forEach(i),Gn=r(e),g(mt.$$.fragment,e),Xn=r(e),I=m(e,"DIV",{class:!0});var E=v(I);g(pt.$$.fragment,E),Zo=r(E),Dt=m(E,"P",{"data-svelte-h":!0}),T(Dt)!=="svelte-ifzhlw"&&(Dt.innerHTML=Ys),qo=r(E),Kt=m(E,"P",{"data-svelte-h":!0}),T(Kt)!=="svelte-q52n56"&&(Kt.innerHTML=Qs),Bo=r(E),en=m(E,"P",{"data-svelte-h":!0}),T(en)!=="svelte-hswkmf"&&(en.innerHTML=Ss),No=r(E),S=m(E,"DIV",{class:!0});var te=v(S);g(ht.$$.fragment,te),Lo=r(te),tn=m(te,"P",{"data-svelte-h":!0}),T(tn)!=="svelte-1efgjwz"&&(tn.innerHTML=As),Ro=r(te),g(Me.$$.fragment,te),Vo=r(te),g(we.$$.fragment,te),te.forEach(i),E.forEach(i),En=r(e),g(ut.$$.fragment,e),Hn=r(e),Z=m(e,"DIV",{class:!0});var H=v(Z);g(ft.$$.fragment,H),Go=r(H),nn=m(H,"P",{"data-svelte-h":!0}),T(nn)!=="svelte-1gwoj50"&&(nn.textContent=Ps),Xo=r(H),on=m(H,"P",{"data-svelte-h":!0}),T(on)!=="svelte-q52n56"&&(on.innerHTML=Os),Eo=r(H),sn=m(H,"P",{"data-svelte-h":!0}),T(sn)!=="svelte-hswkmf"&&(sn.innerHTML=Ds),Ho=r(H),L=m(H,"DIV",{class:!0});var Y=v(L);g(gt.$$.fragment,Y),Yo=r(Y),an=m(Y,"P",{"data-svelte-h":!0}),T(an)!=="svelte-4dmn9"&&(an.innerHTML=Ks),Qo=r(Y),g(ve.$$.fragment,Y),So=r(Y),g(Ce.$$.fragment,Y),Ao=r(Y),g($e.$$.fragment,Y),Y.forEach(i),H.forEach(i),Yn=r(e),g(bt.$$.fragment,e),Qn=r(e),q=m(e,"DIV",{class:!0});var ie=v(q);g(_t.$$.fragment,ie),Po=r(ie),rn=m(ie,"P",{"data-svelte-h":!0}),T(rn)!=="svelte-947z7j"&&(rn.textContent=ea),Oo=r(ie),ln=m(ie,"P",{"data-svelte-h":!0}),T(ln)!=="svelte-q52n56"&&(ln.innerHTML=ta),Do=r(ie),dn=m(ie,"P",{"data-svelte-h":!0}),T(dn)!=="svelte-hswkmf"&&(dn.innerHTML=na),Ko=r(ie),A=m(ie,"DIV",{class:!0});var We=v(A);g(yt.$$.fragment,We),es=r(We),cn=m(We,"P",{"data-svelte-h":!0}),T(cn)!=="svelte-j6uph9"&&(cn.innerHTML=oa),ts=r(We),g(Je.$$.fragment,We),ns=r(We),g(je.$$.fragment,We),We.forEach(i),ie.forEach(i),Sn=r(e),g(kt.$$.fragment,e),An=r(e),B=m(e,"DIV",{class:!0});var le=v(B);g(Tt.$$.fragment,le),os=r(le),mn=m(le,"P",{"data-svelte-h":!0}),T(mn)!=="svelte-3ubwug"&&(mn.textContent=sa),ss=r(le),pn=m(le,"P",{"data-svelte-h":!0}),T(pn)!=="svelte-q52n56"&&(pn.innerHTML=aa),as=r(le),hn=m(le,"P",{"data-svelte-h":!0}),T(hn)!=="svelte-hswkmf"&&(hn.innerHTML=ra),rs=r(le),P=m(le,"DIV",{class:!0});var Ie=v(P);g(Mt.$$.fragment,Ie),is=r(Ie),un=m(Ie,"P",{"data-svelte-h":!0}),T(un)!=="svelte-uoiv91"&&(un.innerHTML=ia),ls=r(Ie),g(ze.$$.fragment,Ie),ds=r(Ie),g(xe.$$.fragment,Ie),Ie.forEach(i),le.forEach(i),Pn=r(e),g(wt.$$.fragment,e),On=r(e),N=m(e,"DIV",{class:!0});var de=v(N);g(vt.$$.fragment,de),cs=r(de),fn=m(de,"P",{"data-svelte-h":!0}),T(fn)!=="svelte-qfptxn"&&(fn.innerHTML=la),ms=r(de),gn=m(de,"P",{"data-svelte-h":!0}),T(gn)!=="svelte-q52n56"&&(gn.innerHTML=da),ps=r(de),bn=m(de,"P",{"data-svelte-h":!0}),T(bn)!=="svelte-hswkmf"&&(bn.innerHTML=ca),hs=r(de),O=m(de,"DIV",{class:!0});var Ze=v(O);g(Ct.$$.fragment,Ze),us=r(Ze),_n=m(Ze,"P",{"data-svelte-h":!0}),T(_n)!=="svelte-173zf8d"&&(_n.innerHTML=ma),fs=r(Ze),g(Fe.$$.fragment,Ze),gs=r(Ze),g(Ue.$$.fragment,Ze),Ze.forEach(i),de.forEach(i),Dn=r(e),g($t.$$.fragment,e),Kn=r(e),kn=m(e,"P",{}),v(kn).forEach(i),this.h()},h(){C(t,"name","hf:doc:metadata"),C(t,"content",Va),ba(J,"float","right"),C(R,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),C(se,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),C(ge,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),C(be,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),C(Nt,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),C(j,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),C(ae,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),C(_e,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),C(F,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),C(re,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),C(U,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),C(Q,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),C(W,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),C(S,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),C(I,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),C(L,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),C(Z,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),C(A,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),C(q,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),C(P,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),C(B,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),C(O,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),C(N,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,l){s(document.head,t),p(e,h,l),p(e,o,l),p(e,d,l),p(e,M,l),p(e,u,l),p(e,J,l),p(e,Ne,l),b(oe,e,l),p(e,Mn,l),p(e,Le,l),p(e,wn,l),p(e,Re,l),p(e,vn,l),p(e,Ve,l),p(e,Cn,l),p(e,Ge,l),p(e,$n,l),b(he,e,l),p(e,Jn,l),p(e,Xe,l),p(e,jn,l),b(ue,e,l),p(e,zn,l),p(e,Ee,l),p(e,xn,l),p(e,He,l),p(e,Fn,l),b(Ye,e,l),p(e,Un,l),b(Qe,e,l),p(e,Wn,l),p(e,R,l),b(Se,R,null),s(R,to),s(R,xt),s(R,no),s(R,Ft),s(R,oo),b(fe,R,null),p(e,In,l),b(Ae,e,l),p(e,Zn,l),p(e,j,l),b(Pe,j,null),s(j,so),s(j,Ut),s(j,ao),s(j,Wt),s(j,ro),s(j,se),b(Oe,se,null),s(se,io),s(se,It),s(se,lo),s(se,Zt),s(j,co),s(j,ge),b(De,ge,null),s(ge,mo),s(ge,qt),s(j,po),s(j,be),b(Ke,be,null),s(be,ho),s(be,Bt),s(j,uo),s(j,Nt),b(et,Nt,null),p(e,qn,l),b(tt,e,l),p(e,Bn,l),p(e,F,l),b(nt,F,null),s(F,fo),s(F,Lt),s(F,go),s(F,Rt),s(F,bo),s(F,ae),b(ot,ae,null),s(ae,_o),s(ae,Vt),s(ae,yo),s(ae,Gt),s(F,ko),s(F,_e),b(st,_e,null),s(_e,To),s(_e,Xt),p(e,Nn,l),b(at,e,l),p(e,Ln,l),p(e,U,l),b(rt,U,null),s(U,Mo),s(U,Et),s(U,wo),s(U,Ht),s(U,vo),s(U,Yt),s(U,Co),s(U,re),b(it,re,null),s(re,$o),s(re,Qt),s(re,Jo),b(ye,re,null),p(e,Rn,l),b(lt,e,l),p(e,Vn,l),p(e,W,l),b(dt,W,null),s(W,jo),s(W,St),s(W,zo),s(W,At),s(W,xo),s(W,Pt),s(W,Fo),s(W,Q),b(ct,Q,null),s(Q,Uo),s(Q,Ot),s(Q,Wo),b(ke,Q,null),s(Q,Io),b(Te,Q,null),p(e,Gn,l),b(mt,e,l),p(e,Xn,l),p(e,I,l),b(pt,I,null),s(I,Zo),s(I,Dt),s(I,qo),s(I,Kt),s(I,Bo),s(I,en),s(I,No),s(I,S),b(ht,S,null),s(S,Lo),s(S,tn),s(S,Ro),b(Me,S,null),s(S,Vo),b(we,S,null),p(e,En,l),b(ut,e,l),p(e,Hn,l),p(e,Z,l),b(ft,Z,null),s(Z,Go),s(Z,nn),s(Z,Xo),s(Z,on),s(Z,Eo),s(Z,sn),s(Z,Ho),s(Z,L),b(gt,L,null),s(L,Yo),s(L,an),s(L,Qo),b(ve,L,null),s(L,So),b(Ce,L,null),s(L,Ao),b($e,L,null),p(e,Yn,l),b(bt,e,l),p(e,Qn,l),p(e,q,l),b(_t,q,null),s(q,Po),s(q,rn),s(q,Oo),s(q,ln),s(q,Do),s(q,dn),s(q,Ko),s(q,A),b(yt,A,null),s(A,es),s(A,cn),s(A,ts),b(Je,A,null),s(A,ns),b(je,A,null),p(e,Sn,l),b(kt,e,l),p(e,An,l),p(e,B,l),b(Tt,B,null),s(B,os),s(B,mn),s(B,ss),s(B,pn),s(B,as),s(B,hn),s(B,rs),s(B,P),b(Mt,P,null),s(P,is),s(P,un),s(P,ls),b(ze,P,null),s(P,ds),b(xe,P,null),p(e,Pn,l),b(wt,e,l),p(e,On,l),p(e,N,l),b(vt,N,null),s(N,cs),s(N,fn),s(N,ms),s(N,gn),s(N,ps),s(N,bn),s(N,hs),s(N,O),b(Ct,O,null),s(O,us),s(O,_n),s(O,fs),b(Fe,O,null),s(O,gs),b(Ue,O,null),p(e,Dn,l),b($t,e,l),p(e,Kn,l),p(e,kn,l),eo=!0},p(e,[l]){const K={};l&2&&(K.$$scope={dirty:l,ctx:e}),he.$set(K);const x={};l&2&&(x.$$scope={dirty:l,ctx:e}),ue.$set(x);const ce={};l&2&&(ce.$$scope={dirty:l,ctx:e}),fe.$set(ce);const Jt={};l&2&&(Jt.$$scope={dirty:l,ctx:e}),ye.$set(Jt);const jt={};l&2&&(jt.$$scope={dirty:l,ctx:e}),ke.$set(jt);const Tn={};l&2&&(Tn.$$scope={dirty:l,ctx:e}),Te.$set(Tn);const V={};l&2&&(V.$$scope={dirty:l,ctx:e}),Me.$set(V);const me={};l&2&&(me.$$scope={dirty:l,ctx:e}),we.$set(me);const zt={};l&2&&(zt.$$scope={dirty:l,ctx:e}),ve.$set(zt);const G={};l&2&&(G.$$scope={dirty:l,ctx:e}),Ce.$set(G);const pe={};l&2&&(pe.$$scope={dirty:l,ctx:e}),$e.$set(pe);const X={};l&2&&(X.$$scope={dirty:l,ctx:e}),Je.$set(X);const ee={};l&2&&(ee.$$scope={dirty:l,ctx:e}),je.$set(ee);const E={};l&2&&(E.$$scope={dirty:l,ctx:e}),ze.$set(E);const te={};l&2&&(te.$$scope={dirty:l,ctx:e}),xe.$set(te);const H={};l&2&&(H.$$scope={dirty:l,ctx:e}),Fe.$set(H);const Y={};l&2&&(Y.$$scope={dirty:l,ctx:e}),Ue.$set(Y)},i(e){eo||(_(oe.$$.fragment,e),_(he.$$.fragment,e),_(ue.$$.fragment,e),_(Ye.$$.fragment,e),_(Qe.$$.fragment,e),_(Se.$$.fragment,e),_(fe.$$.fragment,e),_(Ae.$$.fragment,e),_(Pe.$$.fragment,e),_(Oe.$$.fragment,e),_(De.$$.fragment,e),_(Ke.$$.fragment,e),_(et.$$.fragment,e),_(tt.$$.fragment,e),_(nt.$$.fragment,e),_(ot.$$.fragment,e),_(st.$$.fragment,e),_(at.$$.fragment,e),_(rt.$$.fragment,e),_(it.$$.fragment,e),_(ye.$$.fragment,e),_(lt.$$.fragment,e),_(dt.$$.fragment,e),_(ct.$$.fragment,e),_(ke.$$.fragment,e),_(Te.$$.fragment,e),_(mt.$$.fragment,e),_(pt.$$.fragment,e),_(ht.$$.fragment,e),_(Me.$$.fragment,e),_(we.$$.fragment,e),_(ut.$$.fragment,e),_(ft.$$.fragment,e),_(gt.$$.fragment,e),_(ve.$$.fragment,e),_(Ce.$$.fragment,e),_($e.$$.fragment,e),_(bt.$$.fragment,e),_(_t.$$.fragment,e),_(yt.$$.fragment,e),_(Je.$$.fragment,e),_(je.$$.fragment,e),_(kt.$$.fragment,e),_(Tt.$$.fragment,e),_(Mt.$$.fragment,e),_(ze.$$.fragment,e),_(xe.$$.fragment,e),_(wt.$$.fragment,e),_(vt.$$.fragment,e),_(Ct.$$.fragment,e),_(Fe.$$.fragment,e),_(Ue.$$.fragment,e),_($t.$$.fragment,e),eo=!0)},o(e){y(oe.$$.fragment,e),y(he.$$.fragment,e),y(ue.$$.fragment,e),y(Ye.$$.fragment,e),y(Qe.$$.fragment,e),y(Se.$$.fragment,e),y(fe.$$.fragment,e),y(Ae.$$.fragment,e),y(Pe.$$.fragment,e),y(Oe.$$.fragment,e),y(De.$$.fragment,e),y(Ke.$$.fragment,e),y(et.$$.fragment,e),y(tt.$$.fragment,e),y(nt.$$.fragment,e),y(ot.$$.fragment,e),y(st.$$.fragment,e),y(at.$$.fragment,e),y(rt.$$.fragment,e),y(it.$$.fragment,e),y(ye.$$.fragment,e),y(lt.$$.fragment,e),y(dt.$$.fragment,e),y(ct.$$.fragment,e),y(ke.$$.fragment,e),y(Te.$$.fragment,e),y(mt.$$.fragment,e),y(pt.$$.fragment,e),y(ht.$$.fragment,e),y(Me.$$.fragment,e),y(we.$$.fragment,e),y(ut.$$.fragment,e),y(ft.$$.fragment,e),y(gt.$$.fragment,e),y(ve.$$.fragment,e),y(Ce.$$.fragment,e),y($e.$$.fragment,e),y(bt.$$.fragment,e),y(_t.$$.fragment,e),y(yt.$$.fragment,e),y(Je.$$.fragment,e),y(je.$$.fragment,e),y(kt.$$.fragment,e),y(Tt.$$.fragment,e),y(Mt.$$.fragment,e),y(ze.$$.fragment,e),y(xe.$$.fragment,e),y(wt.$$.fragment,e),y(vt.$$.fragment,e),y(Ct.$$.fragment,e),y(Fe.$$.fragment,e),y(Ue.$$.fragment,e),y($t.$$.fragment,e),eo=!1},d(e){e&&(i(h),i(o),i(d),i(M),i(u),i(J),i(Ne),i(Mn),i(Le),i(wn),i(Re),i(vn),i(Ve),i(Cn),i(Ge),i($n),i(Jn),i(Xe),i(jn),i(zn),i(Ee),i(xn),i(He),i(Fn),i(Un),i(Wn),i(R),i(In),i(Zn),i(j),i(qn),i(Bn),i(F),i(Nn),i(Ln),i(U),i(Rn),i(Vn),i(W),i(Gn),i(Xn),i(I),i(En),i(Hn),i(Z),i(Yn),i(Qn),i(q),i(Sn),i(An),i(B),i(Pn),i(On),i(N),i(Dn),i(Kn),i(kn)),i(t),k(oe,e),k(he,e),k(ue,e),k(Ye,e),k(Qe,e),k(Se),k(fe),k(Ae,e),k(Pe),k(Oe),k(De),k(Ke),k(et),k(tt,e),k(nt),k(ot),k(st),k(at,e),k(rt),k(it),k(ye),k(lt,e),k(dt),k(ct),k(ke),k(Te),k(mt,e),k(pt),k(ht),k(Me),k(we),k(ut,e),k(ft),k(gt),k(ve),k(Ce),k($e),k(bt,e),k(_t),k(yt),k(Je),k(je),k(kt,e),k(Tt),k(Mt),k(ze),k(xe),k(wt,e),k(vt),k(Ct),k(Fe),k(Ue),k($t,e)}}}const Va='{"title":"CamemBERT","local":"camembert","sections":[{"title":"CamembertConfig","local":"transformers.CamembertConfig","sections":[],"depth":2},{"title":"CamembertTokenizer","local":"transformers.CamembertTokenizer","sections":[],"depth":2},{"title":"CamembertTokenizerFast","local":"transformers.CamembertTokenizerFast","sections":[],"depth":2},{"title":"CamembertModel","local":"transformers.CamembertModel","sections":[],"depth":2},{"title":"CamembertForCausalLM","local":"transformers.CamembertForCausalLM","sections":[],"depth":2},{"title":"CamembertForMaskedLM","local":"transformers.CamembertForMaskedLM","sections":[],"depth":2},{"title":"CamembertForSequenceClassification","local":"transformers.CamembertForSequenceClassification","sections":[],"depth":2},{"title":"CamembertForMultipleChoice","local":"transformers.CamembertForMultipleChoice","sections":[],"depth":2},{"title":"CamembertForTokenClassification","local":"transformers.CamembertForTokenClassification","sections":[],"depth":2},{"title":"CamembertForQuestionAnswering","local":"transformers.CamembertForQuestionAnswering","sections":[],"depth":2}],"depth":1}';function Ga(w){return ha(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class Oa extends ua{constructor(t){super(),fa(this,t,Ga,Ra,pa,{})}}export{Oa as component};
