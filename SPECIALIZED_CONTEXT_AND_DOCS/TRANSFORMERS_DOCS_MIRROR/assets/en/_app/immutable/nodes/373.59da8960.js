import{s as Os,o as Es,n as $}from"../chunks/scheduler.18a86fab.js";import{S as Ys,i as Ds,g as c,s as r,r as f,A as Ks,h as p,f as d,c as a,j as C,x as k,u as g,k as v,l as er,y as s,a as h,v as _,d as b,t as T,w as y}from"../chunks/index.98837b22.js";import{T as ge}from"../chunks/Tip.77304350.js";import{D as B}from"../chunks/Docstring.a1ef7999.js";import{C as te}from"../chunks/CodeBlock.8d0c2e8a.js";import{E as _e}from"../chunks/ExampleCodeBlock.8c3ee1f9.js";import{H as ie,E as tr}from"../chunks/getInferenceSnippets.06c2775f.js";import{H as or,a as rs}from"../chunks/HfOption.6641485e.js";function nr(w){let t,m='This model was contributed by <a href="https://huggingface.co/weiweishi" rel="nofollow">weiweishi</a>.',o,i,M="Click on the RoCBert models in the right sidebar for more examples of how to apply RoCBert to different Chinese language tasks.";return{c(){t=c("p"),t.innerHTML=m,o=r(),i=c("p"),i.textContent=M},l(n){t=p(n,"P",{"data-svelte-h":!0}),k(t)!=="svelte-61bhxp"&&(t.innerHTML=m),o=a(n),i=p(n,"P",{"data-svelte-h":!0}),k(i)!=="svelte-3psvli"&&(i.textContent=M)},m(n,u){h(n,t,u),h(n,o,u),h(n,i,u)},p:$,d(n){n&&(d(t),d(o),d(i))}}}function sr(w){let t,m;return t=new te({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwcGlwZWxpbmUlMEElMEFwaXBlbGluZSUyMCUzRCUyMHBpcGVsaW5lKCUwQSUyMCUyMCUyMHRhc2slM0QlMjJmaWxsLW1hc2slMjIlMkMlMEElMjAlMjAlMjBtb2RlbCUzRCUyMndlaXdlaXNoaSUyRnJvYy1iZXJ0LWJhc2UtemglMjIlMkMlMEElMjAlMjAlMjBkdHlwZSUzRHRvcmNoLmZsb2F0MTYlMkMlMEElMjAlMjAlMjBkZXZpY2UlM0QwJTBBKSUwQXBpcGVsaW5lKCUyMiVFOSU4MCU5OSVFNSVBRSVCNiVFOSVBNCU5MCVFNSVCQiVCMyVFNyU5QSU4NCVFNiU4QiU4OSVFOSVCQSVCNSVFNiU5OCVBRiVFNiU4OCU5MSU1Qk1BU0slNUQlRTklODElOEUlRTclOUElODQlRTYlOUMlODAlRTUlQTUlQkQlRTclOUElODQlRTYlOEIlODklRTklQkElQjUlRTQlQjklOEIlMjIp",highlighted:`<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> pipeline

pipeline = pipeline(
   task=<span class="hljs-string">&quot;fill-mask&quot;</span>,
   model=<span class="hljs-string">&quot;weiweishi/roc-bert-base-zh&quot;</span>,
   dtype=torch.float16,
   device=<span class="hljs-number">0</span>
)
pipeline(<span class="hljs-string">&quot;這家餐廳的拉麵是我[MASK]過的最好的拉麵之&quot;</span>)`,wrap:!1}}),{c(){f(t.$$.fragment)},l(o){g(t.$$.fragment,o)},m(o,i){_(t,o,i),m=!0},p:$,i(o){m||(b(t.$$.fragment,o),m=!0)},o(o){T(t.$$.fragment,o),m=!1},d(o){y(t,o)}}}function rr(w){let t,m;return t=new te({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b01vZGVsRm9yTWFza2VkTE0lMkMlMjBBdXRvVG9rZW5pemVyJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTBBJTIwJTIwJTIwJTIyd2Vpd2Vpc2hpJTJGcm9jLWJlcnQtYmFzZS16aCUyMiUyQyUwQSklMEFtb2RlbCUyMCUzRCUyMEF1dG9Nb2RlbEZvck1hc2tlZExNLmZyb21fcHJldHJhaW5lZCglMEElMjAlMjAlMjAlMjJ3ZWl3ZWlzaGklMkZyb2MtYmVydC1iYXNlLXpoJTIyJTJDJTBBJTIwJTIwJTIwZHR5cGUlM0R0b3JjaC5mbG9hdDE2JTJDJTBBJTIwJTIwJTIwZGV2aWNlX21hcCUzRCUyMmF1dG8lMjIlMkMlMEEpJTBBaW5wdXRzJTIwJTNEJTIwdG9rZW5pemVyKCUyMiVFOSU4MCU5OSVFNSVBRSVCNiVFOSVBNCU5MCVFNSVCQiVCMyVFNyU5QSU4NCVFNiU4QiU4OSVFOSVCQSVCNSVFNiU5OCVBRiVFNiU4OCU5MSU1Qk1BU0slNUQlRTklODElOEUlRTclOUElODQlRTYlOUMlODAlRTUlQTUlQkQlRTclOUElODQlRTYlOEIlODklRTklQkElQjUlRTQlQjklOEIlMjIlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKS50byhtb2RlbC5kZXZpY2UpJTBBJTBBd2l0aCUyMHRvcmNoLm5vX2dyYWQoKSUzQSUwQSUyMCUyMCUyMG91dHB1dHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyklMEElMjAlMjAlMjBwcmVkaWN0aW9ucyUyMCUzRCUyMG91dHB1dHMubG9naXRzJTBBJTBBbWFza2VkX2luZGV4JTIwJTNEJTIwdG9yY2gud2hlcmUoaW5wdXRzJTVCJ2lucHV0X2lkcyclNUQlMjAlM0QlM0QlMjB0b2tlbml6ZXIubWFza190b2tlbl9pZCklNUIxJTVEJTBBcHJlZGljdGVkX3Rva2VuX2lkJTIwJTNEJTIwcHJlZGljdGlvbnMlNUIwJTJDJTIwbWFza2VkX2luZGV4JTVELmFyZ21heChkaW0lM0QtMSklMEFwcmVkaWN0ZWRfdG9rZW4lMjAlM0QlMjB0b2tlbml6ZXIuZGVjb2RlKHByZWRpY3RlZF90b2tlbl9pZCklMEElMEFwcmludChmJTIyVGhlJTIwcHJlZGljdGVkJTIwdG9rZW4lMjBpcyUzQSUyMCU3QnByZWRpY3RlZF90b2tlbiU3RCUyMik=",highlighted:`<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoModelForMaskedLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
   <span class="hljs-string">&quot;weiweishi/roc-bert-base-zh&quot;</span>,
)
model = AutoModelForMaskedLM.from_pretrained(
   <span class="hljs-string">&quot;weiweishi/roc-bert-base-zh&quot;</span>,
   dtype=torch.float16,
   device_map=<span class="hljs-string">&quot;auto&quot;</span>,
)
inputs = tokenizer(<span class="hljs-string">&quot;這家餐廳的拉麵是我[MASK]過的最好的拉麵之&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>).to(model.device)

<span class="hljs-keyword">with</span> torch.no_grad():
   outputs = model(**inputs)
   predictions = outputs.logits

masked_index = torch.where(inputs[<span class="hljs-string">&#x27;input_ids&#x27;</span>] == tokenizer.mask_token_id)[<span class="hljs-number">1</span>]
predicted_token_id = predictions[<span class="hljs-number">0</span>, masked_index].argmax(dim=-<span class="hljs-number">1</span>)
predicted_token = tokenizer.decode(predicted_token_id)

<span class="hljs-built_in">print</span>(<span class="hljs-string">f&quot;The predicted token is: <span class="hljs-subst">{predicted_token}</span>&quot;</span>)`,wrap:!1}}),{c(){f(t.$$.fragment)},l(o){g(t.$$.fragment,o)},m(o,i){_(t,o,i),m=!0},p:$,i(o){m||(b(t.$$.fragment,o),m=!0)},o(o){T(t.$$.fragment,o),m=!1},d(o){y(t,o)}}}function ar(w){let t,m;return t=new te({props:{code:"ZWNobyUyMC1lJTIwJTIyJUU5JTgwJTk5JUU1JUFFJUI2JUU5JUE0JTkwJUU1JUJCJUIzJUU3JTlBJTg0JUU2JThCJTg5JUU5JUJBJUI1JUU2JTk4JUFGJUU2JTg4JTkxJTVCTUFTSyU1RCVFOSU4MSU4RSVFNyU5QSU4NCVFNiU5QyU4MCVFNSVBNSVCRCVFNyU5QSU4NCVFNiU4QiU4OSVFOSVCQSVCNSVFNCVCOSU4QiUyMiUyMCU3QyUyMHRyYW5zZm9ybWVycy1jbGklMjBydW4lMjAtLXRhc2slMjBmaWxsLW1hc2slMjAtLW1vZGVsJTIwd2Vpd2Vpc2hpJTJGcm9jLWJlcnQtYmFzZS16aCUyMC0tZGV2aWNlJTIwMA==",highlighted:'<span class="hljs-built_in">echo</span> -e <span class="hljs-string">&quot;這家餐廳的拉麵是我[MASK]過的最好的拉麵之&quot;</span> | transformers-cli run --task fill-mask --model weiweishi/roc-bert-base-zh --device 0',wrap:!1}}),{c(){f(t.$$.fragment)},l(o){g(t.$$.fragment,o)},m(o,i){_(t,o,i),m=!0},p:$,i(o){m||(b(t.$$.fragment,o),m=!0)},o(o){T(t.$$.fragment,o),m=!1},d(o){y(t,o)}}}function ir(w){let t,m,o,i,M,n;return t=new rs({props:{id:"usage",option:"Pipeline",$$slots:{default:[sr]},$$scope:{ctx:w}}}),o=new rs({props:{id:"usage",option:"AutoModel",$$slots:{default:[rr]},$$scope:{ctx:w}}}),M=new rs({props:{id:"usage",option:"transformers CLI",$$slots:{default:[ar]},$$scope:{ctx:w}}}),{c(){f(t.$$.fragment),m=r(),f(o.$$.fragment),i=r(),f(M.$$.fragment)},l(u){g(t.$$.fragment,u),m=a(u),g(o.$$.fragment,u),i=a(u),g(M.$$.fragment,u)},m(u,z){_(t,u,z),h(u,m,z),_(o,u,z),h(u,i,z),_(M,u,z),n=!0},p(u,z){const mo={};z&2&&(mo.$$scope={dirty:z,ctx:u}),t.$set(mo);const Ze={};z&2&&(Ze.$$scope={dirty:z,ctx:u}),o.$set(Ze);const le={};z&2&&(le.$$scope={dirty:z,ctx:u}),M.$set(le)},i(u){n||(b(t.$$.fragment,u),b(o.$$.fragment,u),b(M.$$.fragment,u),n=!0)},o(u){T(t.$$.fragment,u),T(o.$$.fragment,u),T(M.$$.fragment,u),n=!1},d(u){u&&(d(m),d(i)),y(t,u),y(o,u),y(M,u)}}}function lr(w){let t,m;return t=new te({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMFJvQ0JlcnRNb2RlbCUyQyUyMFJvQ0JlcnRDb25maWclMEElMEElMjMlMjBJbml0aWFsaXppbmclMjBhJTIwUm9DQmVydCUyMHdlaXdlaXNoaSUyRnJvYy1iZXJ0LWJhc2UtemglMjBzdHlsZSUyMGNvbmZpZ3VyYXRpb24lMEFjb25maWd1cmF0aW9uJTIwJTNEJTIwUm9DQmVydENvbmZpZygpJTBBJTBBJTIzJTIwSW5pdGlhbGl6aW5nJTIwYSUyMG1vZGVsJTIwZnJvbSUyMHRoZSUyMHdlaXdlaXNoaSUyRnJvYy1iZXJ0LWJhc2UtemglMjBzdHlsZSUyMGNvbmZpZ3VyYXRpb24lMEFtb2RlbCUyMCUzRCUyMFJvQ0JlcnRNb2RlbChjb25maWd1cmF0aW9uKSUwQSUwQSUyMyUyMEFjY2Vzc2luZyUyMHRoZSUyMG1vZGVsJTIwY29uZmlndXJhdGlvbiUwQWNvbmZpZ3VyYXRpb24lMjAlM0QlMjBtb2RlbC5jb25maWc=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> RoCBertModel, RoCBertConfig

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a RoCBert weiweishi/roc-bert-base-zh style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = RoCBertConfig()

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a model from the weiweishi/roc-bert-base-zh style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = RoCBertModel(configuration)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Accessing the model configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = model.config`,wrap:!1}}),{c(){f(t.$$.fragment)},l(o){g(t.$$.fragment,o)},m(o,i){_(t,o,i),m=!0},p:$,i(o){m||(b(t.$$.fragment,o),m=!0)},o(o){T(t.$$.fragment,o),m=!1},d(o){y(t,o)}}}function dr(w){let t,m=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=m},l(o){t=p(o,"P",{"data-svelte-h":!0}),k(t)!=="svelte-fincs2"&&(t.innerHTML=m)},m(o,i){h(o,t,i)},p:$,d(o){o&&d(t)}}}function cr(w){let t,m=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=m},l(o){t=p(o,"P",{"data-svelte-h":!0}),k(t)!=="svelte-fincs2"&&(t.innerHTML=m)},m(o,i){h(o,t,i)},p:$,d(o){o&&d(t)}}}function pr(w){let t,m="Example:",o,i,M;return i=new te({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBSb0NCZXJ0Rm9yUHJlVHJhaW5pbmclMEFpbXBvcnQlMjB0b3JjaCUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMndlaXdlaXNoaSUyRnJvYy1iZXJ0LWJhc2UtemglMjIpJTBBbW9kZWwlMjAlM0QlMjBSb0NCZXJ0Rm9yUHJlVHJhaW5pbmcuZnJvbV9wcmV0cmFpbmVkKCUyMndlaXdlaXNoaSUyRnJvYy1iZXJ0LWJhc2UtemglMjIpJTBBJTBBaW5wdXRzJTIwJTNEJTIwdG9rZW5pemVyKCUyMiVFNCVCRCVBMCVFNSVBNSVCRCVFRiVCQyU4QyVFNSVCRSU4OCVFOSVBQiU5OCVFNSU4NSVCNCVFOCVBRSVBNCVFOCVBRiU4NiVFNCVCRCVBMCUyMiUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpJTBBYXR0YWNrX2lucHV0cyUyMCUzRCUyMCU3QiU3RCUwQWZvciUyMGtleSUyMGluJTIwbGlzdChpbnB1dHMua2V5cygpKSUzQSUwQSUyMCUyMCUyMCUyMGF0dGFja19pbnB1dHMlNUJmJTIyYXR0YWNrXyU3QmtleSU3RCUyMiU1RCUyMCUzRCUyMGlucHV0cyU1QmtleSU1RCUwQWxhYmVsX2lucHV0cyUyMCUzRCUyMCU3QiU3RCUwQWZvciUyMGtleSUyMGluJTIwbGlzdChpbnB1dHMua2V5cygpKSUzQSUwQSUyMCUyMCUyMCUyMGxhYmVsX2lucHV0cyU1QmYlMjJsYWJlbHNfJTdCa2V5JTdEJTIyJTVEJTIwJTNEJTIwaW5wdXRzJTVCa2V5JTVEJTBBJTBBaW5wdXRzLnVwZGF0ZShsYWJlbF9pbnB1dHMpJTBBaW5wdXRzLnVwZGF0ZShhdHRhY2tfaW5wdXRzKSUwQW91dHB1dHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyklMEElMEFsb2dpdHMlMjAlM0QlMjBvdXRwdXRzLmxvZ2l0cyUwQWxvZ2l0cy5zaGFwZQ==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, RoCBertForPreTraining
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;weiweishi/roc-bert-base-zh&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = RoCBertForPreTraining.from_pretrained(<span class="hljs-string">&quot;weiweishi/roc-bert-base-zh&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;你好，很高兴认识你&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>attack_inputs = {}
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">for</span> key <span class="hljs-keyword">in</span> <span class="hljs-built_in">list</span>(inputs.keys()):
<span class="hljs-meta">... </span>    attack_inputs[<span class="hljs-string">f&quot;attack_<span class="hljs-subst">{key}</span>&quot;</span>] = inputs[key]
<span class="hljs-meta">&gt;&gt;&gt; </span>label_inputs = {}
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">for</span> key <span class="hljs-keyword">in</span> <span class="hljs-built_in">list</span>(inputs.keys()):
<span class="hljs-meta">... </span>    label_inputs[<span class="hljs-string">f&quot;labels_<span class="hljs-subst">{key}</span>&quot;</span>] = inputs[key]

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs.update(label_inputs)
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs.update(attack_inputs)
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs)

<span class="hljs-meta">&gt;&gt;&gt; </span>logits = outputs.logits
<span class="hljs-meta">&gt;&gt;&gt; </span>logits.shape
torch.Size([<span class="hljs-number">1</span>, <span class="hljs-number">11</span>, <span class="hljs-number">21128</span>])`,wrap:!1}}),{c(){t=c("p"),t.textContent=m,o=r(),f(i.$$.fragment)},l(n){t=p(n,"P",{"data-svelte-h":!0}),k(t)!=="svelte-11lpom8"&&(t.textContent=m),o=a(n),g(i.$$.fragment,n)},m(n,u){h(n,t,u),h(n,o,u),_(i,n,u),M=!0},p:$,i(n){M||(b(i.$$.fragment,n),M=!0)},o(n){T(i.$$.fragment,n),M=!1},d(n){n&&(d(t),d(o)),y(i,n)}}}function mr(w){let t,m=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=m},l(o){t=p(o,"P",{"data-svelte-h":!0}),k(t)!=="svelte-fincs2"&&(t.innerHTML=m)},m(o,i){h(o,t,i)},p:$,d(o){o&&d(t)}}}function hr(w){let t,m="Example:",o,i,M;return i=new te({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBSb0NCZXJ0Rm9yQ2F1c2FsTE0lMkMlMjBSb0NCZXJ0Q29uZmlnJTBBaW1wb3J0JTIwdG9yY2glMEElMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJ3ZWl3ZWlzaGklMkZyb2MtYmVydC1iYXNlLXpoJTIyKSUwQWNvbmZpZyUyMCUzRCUyMFJvQ0JlcnRDb25maWcuZnJvbV9wcmV0cmFpbmVkKCUyMndlaXdlaXNoaSUyRnJvYy1iZXJ0LWJhc2UtemglMjIpJTBBY29uZmlnLmlzX2RlY29kZXIlMjAlM0QlMjBUcnVlJTBBbW9kZWwlMjAlM0QlMjBSb0NCZXJ0Rm9yQ2F1c2FsTE0uZnJvbV9wcmV0cmFpbmVkKCUyMndlaXdlaXNoaSUyRnJvYy1iZXJ0LWJhc2UtemglMjIlMkMlMjBjb25maWclM0Rjb25maWcpJTBBJTBBaW5wdXRzJTIwJTNEJTIwdG9rZW5pemVyKCUyMiVFNCVCRCVBMCVFNSVBNSVCRCVFRiVCQyU4QyVFNSVCRSU4OCVFOSVBQiU5OCVFNSU4NSVCNCVFOCVBRSVBNCVFOCVBRiU4NiVFNCVCRCVBMCUyMiUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpJTBBb3V0cHV0cyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzKSUwQSUwQXByZWRpY3Rpb25fbG9naXRzJTIwJTNEJTIwb3V0cHV0cy5sb2dpdHM=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, RoCBertForCausalLM, RoCBertConfig
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;weiweishi/roc-bert-base-zh&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>config = RoCBertConfig.from_pretrained(<span class="hljs-string">&quot;weiweishi/roc-bert-base-zh&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>config.is_decoder = <span class="hljs-literal">True</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = RoCBertForCausalLM.from_pretrained(<span class="hljs-string">&quot;weiweishi/roc-bert-base-zh&quot;</span>, config=config)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;你好，很高兴认识你&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs)

<span class="hljs-meta">&gt;&gt;&gt; </span>prediction_logits = outputs.logits`,wrap:!1}}),{c(){t=c("p"),t.textContent=m,o=r(),f(i.$$.fragment)},l(n){t=p(n,"P",{"data-svelte-h":!0}),k(t)!=="svelte-11lpom8"&&(t.textContent=m),o=a(n),g(i.$$.fragment,n)},m(n,u){h(n,t,u),h(n,o,u),_(i,n,u),M=!0},p:$,i(n){M||(b(i.$$.fragment,n),M=!0)},o(n){T(i.$$.fragment,n),M=!1},d(n){n&&(d(t),d(o)),y(i,n)}}}function ur(w){let t,m=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=m},l(o){t=p(o,"P",{"data-svelte-h":!0}),k(t)!=="svelte-fincs2"&&(t.innerHTML=m)},m(o,i){h(o,t,i)},p:$,d(o){o&&d(t)}}}function fr(w){let t,m="Example:",o,i,M;return i=new te({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBSb0NCZXJ0Rm9yTWFza2VkTE0lMEFpbXBvcnQlMjB0b3JjaCUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMndlaXdlaXNoaSUyRnJvYy1iZXJ0LWJhc2UtemglMjIpJTBBbW9kZWwlMjAlM0QlMjBSb0NCZXJ0Rm9yTWFza2VkTE0uZnJvbV9wcmV0cmFpbmVkKCUyMndlaXdlaXNoaSUyRnJvYy1iZXJ0LWJhc2UtemglMjIpJTBBJTBBaW5wdXRzJTIwJTNEJTIwdG9rZW5pemVyKCUyMiVFNiVCMyU5NSVFNSU5QiVCRCVFNiU5OCVBRiVFOSVBNiU5NiVFOSU4MyVCRCU1Qk1BU0slNUQuJTIyJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiklMEElMEF3aXRoJTIwdG9yY2gubm9fZ3JhZCgpJTNBJTBBJTIwJTIwJTIwJTIwbG9naXRzJTIwJTNEJTIwbW9kZWwoKippbnB1dHMpLmxvZ2l0cyUwQSUwQSUyMyUyMHJldHJpZXZlJTIwaW5kZXglMjBvZiUyMCU3Qm1hc2slN0QlMEFtYXNrX3Rva2VuX2luZGV4JTIwJTNEJTIwKGlucHV0cy5pbnB1dF9pZHMlMjAlM0QlM0QlMjB0b2tlbml6ZXIubWFza190b2tlbl9pZCklNUIwJTVELm5vbnplcm8oYXNfdHVwbGUlM0RUcnVlKSU1QjAlNUQlMEElMEFwcmVkaWN0ZWRfdG9rZW5faWQlMjAlM0QlMjBsb2dpdHMlNUIwJTJDJTIwbWFza190b2tlbl9pbmRleCU1RC5hcmdtYXgoYXhpcyUzRC0xKSUwQXRva2VuaXplci5kZWNvZGUocHJlZGljdGVkX3Rva2VuX2lkKQ==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, RoCBertForMaskedLM
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;weiweishi/roc-bert-base-zh&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = RoCBertForMaskedLM.from_pretrained(<span class="hljs-string">&quot;weiweishi/roc-bert-base-zh&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;法国是首都[MASK].&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    logits = model(**inputs).logits

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># retrieve index of {mask}</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[<span class="hljs-number">0</span>].nonzero(as_tuple=<span class="hljs-literal">True</span>)[<span class="hljs-number">0</span>]

<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_token_id = logits[<span class="hljs-number">0</span>, mask_token_index].argmax(axis=-<span class="hljs-number">1</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer.decode(predicted_token_id)
<span class="hljs-string">&#x27;.&#x27;</span>`,wrap:!1}}),{c(){t=c("p"),t.textContent=m,o=r(),f(i.$$.fragment)},l(n){t=p(n,"P",{"data-svelte-h":!0}),k(t)!=="svelte-11lpom8"&&(t.textContent=m),o=a(n),g(i.$$.fragment,n)},m(n,u){h(n,t,u),h(n,o,u),_(i,n,u),M=!0},p:$,i(n){M||(b(i.$$.fragment,n),M=!0)},o(n){T(i.$$.fragment,n),M=!1},d(n){n&&(d(t),d(o)),y(i,n)}}}function gr(w){let t,m=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=m},l(o){t=p(o,"P",{"data-svelte-h":!0}),k(t)!=="svelte-fincs2"&&(t.innerHTML=m)},m(o,i){h(o,t,i)},p:$,d(o){o&&d(t)}}}function _r(w){let t,m="Example of single-label classification:",o,i,M;return i=new te({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Rva2VuaXplciUyQyUyMFJvQ0JlcnRGb3JTZXF1ZW5jZUNsYXNzaWZpY2F0aW9uJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIyd2Vpd2Vpc2hpJTJGcm9jLWJlcnQtYmFzZS16aCUyMiklMEFtb2RlbCUyMCUzRCUyMFJvQ0JlcnRGb3JTZXF1ZW5jZUNsYXNzaWZpY2F0aW9uLmZyb21fcHJldHJhaW5lZCglMjJ3ZWl3ZWlzaGklMkZyb2MtYmVydC1iYXNlLXpoJTIyKSUwQSUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplciglMjJIZWxsbyUyQyUyMG15JTIwZG9nJTIwaXMlMjBjdXRlJTIyJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiklMEElMEF3aXRoJTIwdG9yY2gubm9fZ3JhZCgpJTNBJTBBJTIwJTIwJTIwJTIwbG9naXRzJTIwJTNEJTIwbW9kZWwoKippbnB1dHMpLmxvZ2l0cyUwQSUwQXByZWRpY3RlZF9jbGFzc19pZCUyMCUzRCUyMGxvZ2l0cy5hcmdtYXgoKS5pdGVtKCklMEFtb2RlbC5jb25maWcuaWQybGFiZWwlNUJwcmVkaWN0ZWRfY2xhc3NfaWQlNUQlMEElMEElMjMlMjBUbyUyMHRyYWluJTIwYSUyMG1vZGVsJTIwb24lMjAlNjBudW1fbGFiZWxzJTYwJTIwY2xhc3NlcyUyQyUyMHlvdSUyMGNhbiUyMHBhc3MlMjAlNjBudW1fbGFiZWxzJTNEbnVtX2xhYmVscyU2MCUyMHRvJTIwJTYwLmZyb21fcHJldHJhaW5lZCguLi4pJTYwJTBBbnVtX2xhYmVscyUyMCUzRCUyMGxlbihtb2RlbC5jb25maWcuaWQybGFiZWwpJTBBbW9kZWwlMjAlM0QlMjBSb0NCZXJ0Rm9yU2VxdWVuY2VDbGFzc2lmaWNhdGlvbi5mcm9tX3ByZXRyYWluZWQoJTIyd2Vpd2Vpc2hpJTJGcm9jLWJlcnQtYmFzZS16aCUyMiUyQyUyMG51bV9sYWJlbHMlM0RudW1fbGFiZWxzKSUwQSUwQWxhYmVscyUyMCUzRCUyMHRvcmNoLnRlbnNvciglNUIxJTVEKSUwQWxvc3MlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyUyQyUyMGxhYmVscyUzRGxhYmVscykubG9zcyUwQXJvdW5kKGxvc3MuaXRlbSgpJTJDJTIwMik=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, RoCBertForSequenceClassification

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;weiweishi/roc-bert-base-zh&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = RoCBertForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;weiweishi/roc-bert-base-zh&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    logits = model(**inputs).logits

<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_class_id = logits.argmax().item()
<span class="hljs-meta">&gt;&gt;&gt; </span>model.config.id2label[predicted_class_id]
...

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># To train a model on \`num_labels\` classes, you can pass \`num_labels=num_labels\` to \`.from_pretrained(...)\`</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>num_labels = <span class="hljs-built_in">len</span>(model.config.id2label)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = RoCBertForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;weiweishi/roc-bert-base-zh&quot;</span>, num_labels=num_labels)

<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.tensor([<span class="hljs-number">1</span>])
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = model(**inputs, labels=labels).loss
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">round</span>(loss.item(), <span class="hljs-number">2</span>)
...`,wrap:!1}}),{c(){t=c("p"),t.textContent=m,o=r(),f(i.$$.fragment)},l(n){t=p(n,"P",{"data-svelte-h":!0}),k(t)!=="svelte-ykxpe4"&&(t.textContent=m),o=a(n),g(i.$$.fragment,n)},m(n,u){h(n,t,u),h(n,o,u),_(i,n,u),M=!0},p:$,i(n){M||(b(i.$$.fragment,n),M=!0)},o(n){T(i.$$.fragment,n),M=!1},d(n){n&&(d(t),d(o)),y(i,n)}}}function br(w){let t,m="Example of multi-label classification:",o,i,M;return i=new te({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Rva2VuaXplciUyQyUyMFJvQ0JlcnRGb3JTZXF1ZW5jZUNsYXNzaWZpY2F0aW9uJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIyd2Vpd2Vpc2hpJTJGcm9jLWJlcnQtYmFzZS16aCUyMiklMEFtb2RlbCUyMCUzRCUyMFJvQ0JlcnRGb3JTZXF1ZW5jZUNsYXNzaWZpY2F0aW9uLmZyb21fcHJldHJhaW5lZCglMjJ3ZWl3ZWlzaGklMkZyb2MtYmVydC1iYXNlLXpoJTIyJTJDJTIwcHJvYmxlbV90eXBlJTNEJTIybXVsdGlfbGFiZWxfY2xhc3NpZmljYXRpb24lMjIpJTBBJTBBaW5wdXRzJTIwJTNEJTIwdG9rZW5pemVyKCUyMkhlbGxvJTJDJTIwbXklMjBkb2clMjBpcyUyMGN1dGUlMjIlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQSUwQXdpdGglMjB0b3JjaC5ub19ncmFkKCklM0ElMEElMjAlMjAlMjAlMjBsb2dpdHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cykubG9naXRzJTBBJTBBcHJlZGljdGVkX2NsYXNzX2lkcyUyMCUzRCUyMHRvcmNoLmFyYW5nZSgwJTJDJTIwbG9naXRzLnNoYXBlJTVCLTElNUQpJTVCdG9yY2guc2lnbW9pZChsb2dpdHMpLnNxdWVlemUoZGltJTNEMCklMjAlM0UlMjAwLjUlNUQlMEElMEElMjMlMjBUbyUyMHRyYWluJTIwYSUyMG1vZGVsJTIwb24lMjAlNjBudW1fbGFiZWxzJTYwJTIwY2xhc3NlcyUyQyUyMHlvdSUyMGNhbiUyMHBhc3MlMjAlNjBudW1fbGFiZWxzJTNEbnVtX2xhYmVscyU2MCUyMHRvJTIwJTYwLmZyb21fcHJldHJhaW5lZCguLi4pJTYwJTBBbnVtX2xhYmVscyUyMCUzRCUyMGxlbihtb2RlbC5jb25maWcuaWQybGFiZWwpJTBBbW9kZWwlMjAlM0QlMjBSb0NCZXJ0Rm9yU2VxdWVuY2VDbGFzc2lmaWNhdGlvbi5mcm9tX3ByZXRyYWluZWQoJTBBJTIwJTIwJTIwJTIwJTIyd2Vpd2Vpc2hpJTJGcm9jLWJlcnQtYmFzZS16aCUyMiUyQyUyMG51bV9sYWJlbHMlM0RudW1fbGFiZWxzJTJDJTIwcHJvYmxlbV90eXBlJTNEJTIybXVsdGlfbGFiZWxfY2xhc3NpZmljYXRpb24lMjIlMEEpJTBBJTBBbGFiZWxzJTIwJTNEJTIwdG9yY2guc3VtKCUwQSUyMCUyMCUyMCUyMHRvcmNoLm5uLmZ1bmN0aW9uYWwub25lX2hvdChwcmVkaWN0ZWRfY2xhc3NfaWRzJTVCTm9uZSUyQyUyMCUzQSU1RC5jbG9uZSgpJTJDJTIwbnVtX2NsYXNzZXMlM0RudW1fbGFiZWxzKSUyQyUyMGRpbSUzRDElMEEpLnRvKHRvcmNoLmZsb2F0KSUwQWxvc3MlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyUyQyUyMGxhYmVscyUzRGxhYmVscykubG9zcw==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, RoCBertForSequenceClassification

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;weiweishi/roc-bert-base-zh&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = RoCBertForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;weiweishi/roc-bert-base-zh&quot;</span>, problem_type=<span class="hljs-string">&quot;multi_label_classification&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    logits = model(**inputs).logits

<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_class_ids = torch.arange(<span class="hljs-number">0</span>, logits.shape[-<span class="hljs-number">1</span>])[torch.sigmoid(logits).squeeze(dim=<span class="hljs-number">0</span>) &gt; <span class="hljs-number">0.5</span>]

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># To train a model on \`num_labels\` classes, you can pass \`num_labels=num_labels\` to \`.from_pretrained(...)\`</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>num_labels = <span class="hljs-built_in">len</span>(model.config.id2label)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = RoCBertForSequenceClassification.from_pretrained(
<span class="hljs-meta">... </span>    <span class="hljs-string">&quot;weiweishi/roc-bert-base-zh&quot;</span>, num_labels=num_labels, problem_type=<span class="hljs-string">&quot;multi_label_classification&quot;</span>
<span class="hljs-meta">... </span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.<span class="hljs-built_in">sum</span>(
<span class="hljs-meta">... </span>    torch.nn.functional.one_hot(predicted_class_ids[<span class="hljs-literal">None</span>, :].clone(), num_classes=num_labels), dim=<span class="hljs-number">1</span>
<span class="hljs-meta">... </span>).to(torch.<span class="hljs-built_in">float</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = model(**inputs, labels=labels).loss`,wrap:!1}}),{c(){t=c("p"),t.textContent=m,o=r(),f(i.$$.fragment)},l(n){t=p(n,"P",{"data-svelte-h":!0}),k(t)!=="svelte-1l8e32d"&&(t.textContent=m),o=a(n),g(i.$$.fragment,n)},m(n,u){h(n,t,u),h(n,o,u),_(i,n,u),M=!0},p:$,i(n){M||(b(i.$$.fragment,n),M=!0)},o(n){T(i.$$.fragment,n),M=!1},d(n){n&&(d(t),d(o)),y(i,n)}}}function Tr(w){let t,m=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=m},l(o){t=p(o,"P",{"data-svelte-h":!0}),k(t)!=="svelte-fincs2"&&(t.innerHTML=m)},m(o,i){h(o,t,i)},p:$,d(o){o&&d(t)}}}function yr(w){let t,m="Example:",o,i,M;return i=new te({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBSb0NCZXJ0Rm9yTXVsdGlwbGVDaG9pY2UlMEFpbXBvcnQlMjB0b3JjaCUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMndlaXdlaXNoaSUyRnJvYy1iZXJ0LWJhc2UtemglMjIpJTBBbW9kZWwlMjAlM0QlMjBSb0NCZXJ0Rm9yTXVsdGlwbGVDaG9pY2UuZnJvbV9wcmV0cmFpbmVkKCUyMndlaXdlaXNoaSUyRnJvYy1iZXJ0LWJhc2UtemglMjIpJTBBJTBBcHJvbXB0JTIwJTNEJTIwJTIySW4lMjBJdGFseSUyQyUyMHBpenphJTIwc2VydmVkJTIwaW4lMjBmb3JtYWwlMjBzZXR0aW5ncyUyQyUyMHN1Y2glMjBhcyUyMGF0JTIwYSUyMHJlc3RhdXJhbnQlMkMlMjBpcyUyMHByZXNlbnRlZCUyMHVuc2xpY2VkLiUyMiUwQWNob2ljZTAlMjAlM0QlMjAlMjJJdCUyMGlzJTIwZWF0ZW4lMjB3aXRoJTIwYSUyMGZvcmslMjBhbmQlMjBhJTIwa25pZmUuJTIyJTBBY2hvaWNlMSUyMCUzRCUyMCUyMkl0JTIwaXMlMjBlYXRlbiUyMHdoaWxlJTIwaGVsZCUyMGluJTIwdGhlJTIwaGFuZC4lMjIlMEFsYWJlbHMlMjAlM0QlMjB0b3JjaC50ZW5zb3IoMCkudW5zcXVlZXplKDApJTIwJTIwJTIzJTIwY2hvaWNlMCUyMGlzJTIwY29ycmVjdCUyMChhY2NvcmRpbmclMjB0byUyMFdpa2lwZWRpYSUyMCUzQikpJTJDJTIwYmF0Y2glMjBzaXplJTIwMSUwQSUwQWVuY29kaW5nJTIwJTNEJTIwdG9rZW5pemVyKCU1QnByb21wdCUyQyUyMHByb21wdCU1RCUyQyUyMCU1QmNob2ljZTAlMkMlMjBjaG9pY2UxJTVEJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiUyQyUyMHBhZGRpbmclM0RUcnVlKSUwQW91dHB1dHMlMjAlM0QlMjBtb2RlbCgqKiU3QmslM0ElMjB2LnVuc3F1ZWV6ZSgwKSUyMGZvciUyMGslMkMlMjB2JTIwaW4lMjBlbmNvZGluZy5pdGVtcygpJTdEJTJDJTIwbGFiZWxzJTNEbGFiZWxzKSUyMCUyMCUyMyUyMGJhdGNoJTIwc2l6ZSUyMGlzJTIwMSUwQSUwQSUyMyUyMHRoZSUyMGxpbmVhciUyMGNsYXNzaWZpZXIlMjBzdGlsbCUyMG5lZWRzJTIwdG8lMjBiZSUyMHRyYWluZWQlMEFsb3NzJTIwJTNEJTIwb3V0cHV0cy5sb3NzJTBBbG9naXRzJTIwJTNEJTIwb3V0cHV0cy5sb2dpdHM=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, RoCBertForMultipleChoice
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;weiweishi/roc-bert-base-zh&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = RoCBertForMultipleChoice.from_pretrained(<span class="hljs-string">&quot;weiweishi/roc-bert-base-zh&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>prompt = <span class="hljs-string">&quot;In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>choice0 = <span class="hljs-string">&quot;It is eaten with a fork and a knife.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>choice1 = <span class="hljs-string">&quot;It is eaten while held in the hand.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.tensor(<span class="hljs-number">0</span>).unsqueeze(<span class="hljs-number">0</span>)  <span class="hljs-comment"># choice0 is correct (according to Wikipedia ;)), batch size 1</span>

<span class="hljs-meta">&gt;&gt;&gt; </span>encoding = tokenizer([prompt, prompt], [choice0, choice1], return_tensors=<span class="hljs-string">&quot;pt&quot;</span>, padding=<span class="hljs-literal">True</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**{k: v.unsqueeze(<span class="hljs-number">0</span>) <span class="hljs-keyword">for</span> k, v <span class="hljs-keyword">in</span> encoding.items()}, labels=labels)  <span class="hljs-comment"># batch size is 1</span>

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># the linear classifier still needs to be trained</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = outputs.loss
<span class="hljs-meta">&gt;&gt;&gt; </span>logits = outputs.logits`,wrap:!1}}),{c(){t=c("p"),t.textContent=m,o=r(),f(i.$$.fragment)},l(n){t=p(n,"P",{"data-svelte-h":!0}),k(t)!=="svelte-11lpom8"&&(t.textContent=m),o=a(n),g(i.$$.fragment,n)},m(n,u){h(n,t,u),h(n,o,u),_(i,n,u),M=!0},p:$,i(n){M||(b(i.$$.fragment,n),M=!0)},o(n){T(i.$$.fragment,n),M=!1},d(n){n&&(d(t),d(o)),y(i,n)}}}function kr(w){let t,m=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=m},l(o){t=p(o,"P",{"data-svelte-h":!0}),k(t)!=="svelte-fincs2"&&(t.innerHTML=m)},m(o,i){h(o,t,i)},p:$,d(o){o&&d(t)}}}function Mr(w){let t,m="Example:",o,i,M;return i=new te({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBSb0NCZXJ0Rm9yVG9rZW5DbGFzc2lmaWNhdGlvbiUwQWltcG9ydCUyMHRvcmNoJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIyd2Vpd2Vpc2hpJTJGcm9jLWJlcnQtYmFzZS16aCUyMiklMEFtb2RlbCUyMCUzRCUyMFJvQ0JlcnRGb3JUb2tlbkNsYXNzaWZpY2F0aW9uLmZyb21fcHJldHJhaW5lZCglMjJ3ZWl3ZWlzaGklMkZyb2MtYmVydC1iYXNlLXpoJTIyKSUwQSUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplciglMEElMjAlMjAlMjAlMjAlMjJIdWdnaW5nRmFjZSUyMGlzJTIwYSUyMGNvbXBhbnklMjBiYXNlZCUyMGluJTIwUGFyaXMlMjBhbmQlMjBOZXclMjBZb3JrJTIyJTJDJTIwYWRkX3NwZWNpYWxfdG9rZW5zJTNERmFsc2UlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyJTBBKSUwQSUwQXdpdGglMjB0b3JjaC5ub19ncmFkKCklM0ElMEElMjAlMjAlMjAlMjBsb2dpdHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cykubG9naXRzJTBBJTBBcHJlZGljdGVkX3Rva2VuX2NsYXNzX2lkcyUyMCUzRCUyMGxvZ2l0cy5hcmdtYXgoLTEpJTBBJTBBJTIzJTIwTm90ZSUyMHRoYXQlMjB0b2tlbnMlMjBhcmUlMjBjbGFzc2lmaWVkJTIwcmF0aGVyJTIwdGhlbiUyMGlucHV0JTIwd29yZHMlMjB3aGljaCUyMG1lYW5zJTIwdGhhdCUwQSUyMyUyMHRoZXJlJTIwbWlnaHQlMjBiZSUyMG1vcmUlMjBwcmVkaWN0ZWQlMjB0b2tlbiUyMGNsYXNzZXMlMjB0aGFuJTIwd29yZHMuJTBBJTIzJTIwTXVsdGlwbGUlMjB0b2tlbiUyMGNsYXNzZXMlMjBtaWdodCUyMGFjY291bnQlMjBmb3IlMjB0aGUlMjBzYW1lJTIwd29yZCUwQXByZWRpY3RlZF90b2tlbnNfY2xhc3NlcyUyMCUzRCUyMCU1Qm1vZGVsLmNvbmZpZy5pZDJsYWJlbCU1QnQuaXRlbSgpJTVEJTIwZm9yJTIwdCUyMGluJTIwcHJlZGljdGVkX3Rva2VuX2NsYXNzX2lkcyU1QjAlNUQlNUQlMEFwcmVkaWN0ZWRfdG9rZW5zX2NsYXNzZXMlMEElMEFsYWJlbHMlMjAlM0QlMjBwcmVkaWN0ZWRfdG9rZW5fY2xhc3NfaWRzJTBBbG9zcyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzJTJDJTIwbGFiZWxzJTNEbGFiZWxzKS5sb3NzJTBBcm91bmQobG9zcy5pdGVtKCklMkMlMjAyKQ==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, RoCBertForTokenClassification
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;weiweishi/roc-bert-base-zh&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = RoCBertForTokenClassification.from_pretrained(<span class="hljs-string">&quot;weiweishi/roc-bert-base-zh&quot;</span>)

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
...`,wrap:!1}}),{c(){t=c("p"),t.textContent=m,o=r(),f(i.$$.fragment)},l(n){t=p(n,"P",{"data-svelte-h":!0}),k(t)!=="svelte-11lpom8"&&(t.textContent=m),o=a(n),g(i.$$.fragment,n)},m(n,u){h(n,t,u),h(n,o,u),_(i,n,u),M=!0},p:$,i(n){M||(b(i.$$.fragment,n),M=!0)},o(n){T(i.$$.fragment,n),M=!1},d(n){n&&(d(t),d(o)),y(i,n)}}}function wr(w){let t,m=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=m},l(o){t=p(o,"P",{"data-svelte-h":!0}),k(t)!=="svelte-fincs2"&&(t.innerHTML=m)},m(o,i){h(o,t,i)},p:$,d(o){o&&d(t)}}}function Cr(w){let t,m="Example:",o,i,M;return i=new te({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBSb0NCZXJ0Rm9yUXVlc3Rpb25BbnN3ZXJpbmclMEFpbXBvcnQlMjB0b3JjaCUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMndlaXdlaXNoaSUyRnJvYy1iZXJ0LWJhc2UtemglMjIpJTBBbW9kZWwlMjAlM0QlMjBSb0NCZXJ0Rm9yUXVlc3Rpb25BbnN3ZXJpbmcuZnJvbV9wcmV0cmFpbmVkKCUyMndlaXdlaXNoaSUyRnJvYy1iZXJ0LWJhc2UtemglMjIpJTBBJTBBcXVlc3Rpb24lMkMlMjB0ZXh0JTIwJTNEJTIwJTIyV2hvJTIwd2FzJTIwSmltJTIwSGVuc29uJTNGJTIyJTJDJTIwJTIySmltJTIwSGVuc29uJTIwd2FzJTIwYSUyMG5pY2UlMjBwdXBwZXQlMjIlMEElMEFpbnB1dHMlMjAlM0QlMjB0b2tlbml6ZXIocXVlc3Rpb24lMkMlMjB0ZXh0JTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiklMEF3aXRoJTIwdG9yY2gubm9fZ3JhZCgpJTNBJTBBJTIwJTIwJTIwJTIwb3V0cHV0cyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzKSUwQSUwQWFuc3dlcl9zdGFydF9pbmRleCUyMCUzRCUyMG91dHB1dHMuc3RhcnRfbG9naXRzLmFyZ21heCgpJTBBYW5zd2VyX2VuZF9pbmRleCUyMCUzRCUyMG91dHB1dHMuZW5kX2xvZ2l0cy5hcmdtYXgoKSUwQSUwQXByZWRpY3RfYW5zd2VyX3Rva2VucyUyMCUzRCUyMGlucHV0cy5pbnB1dF9pZHMlNUIwJTJDJTIwYW5zd2VyX3N0YXJ0X2luZGV4JTIwJTNBJTIwYW5zd2VyX2VuZF9pbmRleCUyMCUyQiUyMDElNUQlMEF0b2tlbml6ZXIuZGVjb2RlKHByZWRpY3RfYW5zd2VyX3Rva2VucyUyQyUyMHNraXBfc3BlY2lhbF90b2tlbnMlM0RUcnVlKSUwQSUwQSUyMyUyMHRhcmdldCUyMGlzJTIwJTIybmljZSUyMHB1cHBldCUyMiUwQXRhcmdldF9zdGFydF9pbmRleCUyMCUzRCUyMHRvcmNoLnRlbnNvciglNUIxNCU1RCklMEF0YXJnZXRfZW5kX2luZGV4JTIwJTNEJTIwdG9yY2gudGVuc29yKCU1QjE1JTVEKSUwQSUwQW91dHB1dHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyUyQyUyMHN0YXJ0X3Bvc2l0aW9ucyUzRHRhcmdldF9zdGFydF9pbmRleCUyQyUyMGVuZF9wb3NpdGlvbnMlM0R0YXJnZXRfZW5kX2luZGV4KSUwQWxvc3MlMjAlM0QlMjBvdXRwdXRzLmxvc3MlMEFyb3VuZChsb3NzLml0ZW0oKSUyQyUyMDIp",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, RoCBertForQuestionAnswering
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;weiweishi/roc-bert-base-zh&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = RoCBertForQuestionAnswering.from_pretrained(<span class="hljs-string">&quot;weiweishi/roc-bert-base-zh&quot;</span>)

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
...`,wrap:!1}}),{c(){t=c("p"),t.textContent=m,o=r(),f(i.$$.fragment)},l(n){t=p(n,"P",{"data-svelte-h":!0}),k(t)!=="svelte-11lpom8"&&(t.textContent=m),o=a(n),g(i.$$.fragment,n)},m(n,u){h(n,t,u),h(n,o,u),_(i,n,u),M=!0},p:$,i(n){M||(b(i.$$.fragment,n),M=!0)},o(n){T(i.$$.fragment,n),M=!1},d(n){n&&(d(t),d(o)),y(i,n)}}}function vr(w){let t,m,o,i,M,n="<em>This model was released on 2022-05-27 and added to Hugging Face Transformers on 2022-11-08.</em>",u,z,mo='<div class="flex flex-wrap space-x-1"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/></div>',Ze,le,fo,Se,as='<a href="https://aclanthology.org/2022.acl-long.65.pdf" rel="nofollow">RoCBert</a> is a pretrained Chinese <a href="./bert">BERT</a> model designed against adversarial attacks like typos and synonyms. It is pretrained with a contrastive learning objective to align normal and adversarial text examples. The examples include different semantic, phonetic, and visual features of Chinese. This makes RoCBert more robust against manipulation.',go,Le,is='You can find all the original RoCBert checkpoints under the <a href="https://huggingface.co/weiweishi" rel="nofollow">weiweishi</a> profile.',_o,be,bo,Xe,ls='The example below demonstrates how to predict the [MASK] token with <a href="/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.Pipeline">Pipeline</a>, <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel">AutoModel</a>, and from the command line.',To,Te,yo,Pe,ko,Z,Qe,Xo,Ct,ds=`This is the configuration class to store the configuration of a <a href="/docs/transformers/v4.56.2/en/model_doc/roc_bert#transformers.RoCBertModel">RoCBertModel</a>. It is used to instantiate a
RoCBert model according to the specified arguments, defining the model architecture. Instantiating a configuration
with the defaults will yield a similar configuration to that of the RoCBert
<a href="https://huggingface.co/weiweishi/roc-bert-base-zh" rel="nofollow">weiweishi/roc-bert-base-zh</a> architecture.`,Po,vt,cs=`Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> and can be used to control the model outputs. Read the
documentation from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> for more information.`,Qo,ye,Mo,Ge,wo,J,He,Go,de,Ae,Ho,Bt,ps=`Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
adding special tokens. A BERT sequence has the following format:`,Ao,zt,ms="<li>single sequence: <code>[CLS] X [SEP]</code></li> <li>pair of sequences: <code>[CLS] A [SEP] B [SEP]</code></li>",Oo,ke,Oe,Eo,$t,hs=`Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
special tokens using the tokenizer <code>prepare_for_model</code> method.`,Yo,ce,Ee,Do,Rt,us=`Create the token type IDs corresponding to the sequences passed. <a href="../glossary#token-type-ids">What are token type
IDs?</a>`,Ko,Jt,fs="Should be overridden in a subclass if the model has a special way of building those.",en,Ut,Ye,Co,De,vo,R,Ke,tn,jt,gs=`The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
cross-attention is added between the self-attention layers, following the architecture described in <a href="https://huggingface.co/papers/1706.03762" rel="nofollow">Attention is
all you need</a> by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.`,on,Ft,_s=`To behave as an decoder the model needs to be initialized with the <code>is_decoder</code> argument of the configuration set
to <code>True</code>. To be used in a Seq2Seq model, the model needs to be initialized with both <code>is_decoder</code> argument and
<code>add_cross_attention</code> set to <code>True</code>; an <code>encoder_hidden_states</code> is then expected as an input to the forward pass.`,nn,xt,bs=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,sn,It,Ts=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,rn,pe,et,an,Wt,ys='The <a href="/docs/transformers/v4.56.2/en/model_doc/roc_bert#transformers.RoCBertModel">RoCBertModel</a> forward method, overrides the <code>__call__</code> special method.',ln,Me,Bo,tt,zo,U,ot,dn,Vt,ks="RoCBert Model with contrastive loss and masked_lm_loss during the pretraining.",cn,Nt,Ms=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,pn,qt,ws=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,mn,O,nt,hn,Zt,Cs='The <a href="/docs/transformers/v4.56.2/en/model_doc/roc_bert#transformers.RoCBertForPreTraining">RoCBertForPreTraining</a> forward method, overrides the <code>__call__</code> special method.',un,we,fn,Ce,$o,st,Ro,j,rt,gn,St,vs="RoCBert Model with a <code>language modeling</code> head on top for CLM fine-tuning.",_n,Lt,Bs=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,bn,Xt,zs=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Tn,E,at,yn,Pt,$s='The <a href="/docs/transformers/v4.56.2/en/model_doc/roc_bert#transformers.RoCBertForCausalLM">RoCBertForCausalLM</a> forward method, overrides the <code>__call__</code> special method.',kn,ve,Mn,Be,Jo,it,Uo,F,lt,wn,Qt,Rs="The Roc Bert Model with a <code>language modeling</code> head on top.”",Cn,Gt,Js=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,vn,Ht,Us=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Bn,Y,dt,zn,At,js='The <a href="/docs/transformers/v4.56.2/en/model_doc/roc_bert#transformers.RoCBertForMaskedLM">RoCBertForMaskedLM</a> forward method, overrides the <code>__call__</code> special method.',$n,ze,Rn,$e,jo,ct,Fo,x,pt,Jn,Ot,Fs=`RoCBert Model transformer with a sequence classification/regression head on top (a linear layer on top of
the pooled output) e.g. for GLUE tasks.`,Un,Et,xs=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,jn,Yt,Is=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Fn,q,mt,xn,Dt,Ws='The <a href="/docs/transformers/v4.56.2/en/model_doc/roc_bert#transformers.RoCBertForSequenceClassification">RoCBertForSequenceClassification</a> forward method, overrides the <code>__call__</code> special method.',In,Re,Wn,Je,Vn,Ue,xo,ht,Io,I,ut,Nn,Kt,Vs=`The Roc Bert Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
softmax) e.g. for RocStories/SWAG tasks.`,qn,eo,Ns=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Zn,to,qs=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Sn,D,ft,Ln,oo,Zs='The <a href="/docs/transformers/v4.56.2/en/model_doc/roc_bert#transformers.RoCBertForMultipleChoice">RoCBertForMultipleChoice</a> forward method, overrides the <code>__call__</code> special method.',Xn,je,Pn,Fe,Wo,gt,Vo,W,_t,Qn,no,Ss=`The Roc Bert transformer with a token classification head on top (a linear layer on top of the hidden-states
output) e.g. for Named-Entity-Recognition (NER) tasks.`,Gn,so,Ls=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Hn,ro,Xs=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,An,K,bt,On,ao,Ps='The <a href="/docs/transformers/v4.56.2/en/model_doc/roc_bert#transformers.RoCBertForTokenClassification">RoCBertForTokenClassification</a> forward method, overrides the <code>__call__</code> special method.',En,xe,Yn,Ie,No,Tt,qo,V,yt,Dn,io,Qs=`The Roc Bert transformer with a span classification head on top for extractive question-answering tasks like
SQuAD (a linear layer on top of the hidden-states output to compute <code>span start logits</code> and <code>span end logits</code>).`,Kn,lo,Gs=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,es,co,Hs=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,ts,ee,kt,os,po,As='The <a href="/docs/transformers/v4.56.2/en/model_doc/roc_bert#transformers.RoCBertForQuestionAnswering">RoCBertForQuestionAnswering</a> forward method, overrides the <code>__call__</code> special method.',ns,We,ss,Ve,Zo,Mt,So,ho,Lo;return le=new ie({props:{title:"RoCBert",local:"rocbert",headingTag:"h1"}}),be=new ge({props:{warning:!1,$$slots:{default:[nr]},$$scope:{ctx:w}}}),Te=new or({props:{id:"usage",options:["Pipeline","AutoModel","transformers CLI"],$$slots:{default:[ir]},$$scope:{ctx:w}}}),Pe=new ie({props:{title:"RoCBertConfig",local:"transformers.RoCBertConfig",headingTag:"h2"}}),Qe=new B({props:{name:"class transformers.RoCBertConfig",anchor:"transformers.RoCBertConfig",parameters:[{name:"vocab_size",val:" = 30522"},{name:"hidden_size",val:" = 768"},{name:"num_hidden_layers",val:" = 12"},{name:"num_attention_heads",val:" = 12"},{name:"intermediate_size",val:" = 3072"},{name:"hidden_act",val:" = 'gelu'"},{name:"hidden_dropout_prob",val:" = 0.1"},{name:"attention_probs_dropout_prob",val:" = 0.1"},{name:"max_position_embeddings",val:" = 512"},{name:"type_vocab_size",val:" = 2"},{name:"initializer_range",val:" = 0.02"},{name:"layer_norm_eps",val:" = 1e-12"},{name:"use_cache",val:" = True"},{name:"pad_token_id",val:" = 0"},{name:"position_embedding_type",val:" = 'absolute'"},{name:"classifier_dropout",val:" = None"},{name:"enable_pronunciation",val:" = True"},{name:"enable_shape",val:" = True"},{name:"pronunciation_embed_dim",val:" = 768"},{name:"pronunciation_vocab_size",val:" = 910"},{name:"shape_embed_dim",val:" = 512"},{name:"shape_vocab_size",val:" = 24858"},{name:"concat_input",val:" = True"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.RoCBertConfig.vocab_size",description:`<strong>vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 30522) &#x2014;
Vocabulary size of the RoCBert model. Defines the number of different tokens that can be represented by the
<code>inputs_ids</code> passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/roc_bert#transformers.RoCBertModel">RoCBertModel</a>.`,name:"vocab_size"},{anchor:"transformers.RoCBertConfig.hidden_size",description:`<strong>hidden_size</strong> (<code>int</code>, <em>optional</em>, defaults to 768) &#x2014;
Dimension of the encoder layers and the pooler layer.`,name:"hidden_size"},{anchor:"transformers.RoCBertConfig.num_hidden_layers",description:`<strong>num_hidden_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 12) &#x2014;
Number of hidden layers in the Transformer encoder.`,name:"num_hidden_layers"},{anchor:"transformers.RoCBertConfig.num_attention_heads",description:`<strong>num_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 12) &#x2014;
Number of attention heads for each attention layer in the Transformer encoder.`,name:"num_attention_heads"},{anchor:"transformers.RoCBertConfig.intermediate_size",description:`<strong>intermediate_size</strong> (<code>int</code>, <em>optional</em>, defaults to 3072) &#x2014;
Dimension of the &#x201C;intermediate&#x201D; (i.e., feed-forward) layer in the Transformer encoder.`,name:"intermediate_size"},{anchor:"transformers.RoCBertConfig.hidden_act",description:`<strong>hidden_act</strong> (<code>str</code> or <code>function</code>, <em>optional</em>, defaults to <code>&quot;gelu&quot;</code>) &#x2014;
The non-linear activation function (function or string) in the encoder and pooler. If string, <code>&quot;gelu&quot;</code>,
<code>&quot;relu&quot;</code>, <code>&quot;selu&quot;</code> and <code>&quot;gelu_new&quot;</code> are supported.`,name:"hidden_act"},{anchor:"transformers.RoCBertConfig.hidden_dropout_prob",description:`<strong>hidden_dropout_prob</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.`,name:"hidden_dropout_prob"},{anchor:"transformers.RoCBertConfig.attention_probs_dropout_prob",description:`<strong>attention_probs_dropout_prob</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout ratio for the attention probabilities.`,name:"attention_probs_dropout_prob"},{anchor:"transformers.RoCBertConfig.max_position_embeddings",description:`<strong>max_position_embeddings</strong> (<code>int</code>, <em>optional</em>, defaults to 512) &#x2014;
The maximum sequence length that this model might ever be used with. Typically set this to something large
just in case (e.g., 512 or 1024 or 2048).`,name:"max_position_embeddings"},{anchor:"transformers.RoCBertConfig.type_vocab_size",description:`<strong>type_vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 2) &#x2014;
The vocabulary size of the <code>token_type_ids</code> passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/roc_bert#transformers.RoCBertModel">RoCBertModel</a>.`,name:"type_vocab_size"},{anchor:"transformers.RoCBertConfig.initializer_range",description:`<strong>initializer_range</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.`,name:"initializer_range"},{anchor:"transformers.RoCBertConfig.layer_norm_eps",description:`<strong>layer_norm_eps</strong> (<code>float</code>, <em>optional</em>, defaults to 1e-12) &#x2014;
The epsilon used by the layer normalization layers.`,name:"layer_norm_eps"},{anchor:"transformers.RoCBertConfig.is_decoder",description:`<strong>is_decoder</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether the model is used as a decoder or not. If <code>False</code>, the model is used as an encoder.`,name:"is_decoder"},{anchor:"transformers.RoCBertConfig.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not the model should return the last key/values attentions (not used by all models). Only
relevant if <code>config.is_decoder=True</code>.`,name:"use_cache"},{anchor:"transformers.RoCBertConfig.position_embedding_type",description:`<strong>position_embedding_type</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;absolute&quot;</code>) &#x2014;
Type of position embedding. Choose one of <code>&quot;absolute&quot;</code>, <code>&quot;relative_key&quot;</code>, <code>&quot;relative_key_query&quot;</code>. For
positional embeddings use <code>&quot;absolute&quot;</code>. For more information on <code>&quot;relative_key&quot;</code>, please refer to
<a href="https://huggingface.co/papers/1803.02155" rel="nofollow">Self-Attention with Relative Position Representations (Shaw et al.)</a>.
For more information on <code>&quot;relative_key_query&quot;</code>, please refer to <em>Method 4</em> in <a href="https://huggingface.co/papers/2009.13658" rel="nofollow">Improve Transformer Models
with Better Relative Position Embeddings (Huang et al.)</a>.`,name:"position_embedding_type"},{anchor:"transformers.RoCBertConfig.classifier_dropout",description:`<strong>classifier_dropout</strong> (<code>float</code>, <em>optional</em>) &#x2014;
The dropout ratio for the classification head.`,name:"classifier_dropout"},{anchor:"transformers.RoCBertConfig.enable_pronunciation",description:`<strong>enable_pronunciation</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not the model use pronunciation embed when training.`,name:"enable_pronunciation"},{anchor:"transformers.RoCBertConfig.enable_shape",description:`<strong>enable_shape</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not the model use shape embed when training.`,name:"enable_shape"},{anchor:"transformers.RoCBertConfig.pronunciation_embed_dim",description:`<strong>pronunciation_embed_dim</strong> (<code>int</code>, <em>optional</em>, defaults to 768) &#x2014;
Dimension of the pronunciation_embed.`,name:"pronunciation_embed_dim"},{anchor:"transformers.RoCBertConfig.pronunciation_vocab_size",description:`<strong>pronunciation_vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 910) &#x2014;
Pronunciation Vocabulary size of the RoCBert model. Defines the number of different tokens that can be
represented by the <code>input_pronunciation_ids</code> passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/roc_bert#transformers.RoCBertModel">RoCBertModel</a>.`,name:"pronunciation_vocab_size"},{anchor:"transformers.RoCBertConfig.shape_embed_dim",description:`<strong>shape_embed_dim</strong> (<code>int</code>, <em>optional</em>, defaults to 512) &#x2014;
Dimension of the shape_embed.`,name:"shape_embed_dim"},{anchor:"transformers.RoCBertConfig.shape_vocab_size",description:`<strong>shape_vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 24858) &#x2014;
Shape Vocabulary size of the RoCBert model. Defines the number of different tokens that can be represented
by the <code>input_shape_ids</code> passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/roc_bert#transformers.RoCBertModel">RoCBertModel</a>.`,name:"shape_vocab_size"},{anchor:"transformers.RoCBertConfig.concat_input",description:`<strong>concat_input</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Defines the way of merging the shape_embed, pronunciation_embed and word_embed, if the value is true,
output_embed = torch.cat((word_embed, shape_embed, pronunciation_embed), -1), else output_embed =
(word_embed + shape_embed + pronunciation_embed) / 3`,name:"concat_input"},{anchor:"transformers.RoCBertConfig.Example",description:"<strong>Example</strong> &#x2014;",name:"Example"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/roc_bert/configuration_roc_bert.py#L24"}}),ye=new _e({props:{anchor:"transformers.RoCBertConfig.example",$$slots:{default:[lr]},$$scope:{ctx:w}}}),Ge=new ie({props:{title:"RoCBertTokenizer",local:"transformers.RoCBertTokenizer",headingTag:"h2"}}),He=new B({props:{name:"class transformers.RoCBertTokenizer",anchor:"transformers.RoCBertTokenizer",parameters:[{name:"vocab_file",val:""},{name:"word_shape_file",val:""},{name:"word_pronunciation_file",val:""},{name:"do_lower_case",val:" = True"},{name:"do_basic_tokenize",val:" = True"},{name:"never_split",val:" = None"},{name:"unk_token",val:" = '[UNK]'"},{name:"sep_token",val:" = '[SEP]'"},{name:"pad_token",val:" = '[PAD]'"},{name:"cls_token",val:" = '[CLS]'"},{name:"mask_token",val:" = '[MASK]'"},{name:"tokenize_chinese_chars",val:" = True"},{name:"strip_accents",val:" = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.RoCBertTokenizer.Construct",description:'<strong>Construct</strong> a RoCBert tokenizer. Based on WordPiece. This tokenizer inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer">PreTrainedTokenizer</a> which &#x2014;',name:"Construct"},{anchor:"transformers.RoCBertTokenizer.contains",description:"<strong>contains</strong> most of the main methods. Users should refer to this superclass for more information regarding those &#x2014;",name:"contains"},{anchor:"transformers.RoCBertTokenizer.methods.",description:`<strong>methods.</strong> &#x2014;
vocab_file (<code>str</code>):
File containing the vocabulary.
word_shape_file (<code>str</code>):
File containing the word =&gt; shape info.
word_pronunciation_file (<code>str</code>):
File containing the word =&gt; pronunciation info.
do_lower_case (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>):
Whether or not to lowercase the input when tokenizing.
do_basic_tokenize (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>):
Whether or not to do basic tokenization before WordPiece.
never_split (<code>Iterable</code>, <em>optional</em>):
Collection of tokens which will never be split during tokenization. Only has an effect when
<code>do_basic_tokenize=True</code>
unk_token (<code>str</code>, <em>optional</em>, defaults to <code>&quot;[UNK]&quot;</code>):
The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
token instead.
sep_token (<code>str</code>, <em>optional</em>, defaults to <code>&quot;[SEP]&quot;</code>):
The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
sequence classification or for a text and a question for question answering. It is also used as the last
token of a sequence built with special tokens.
pad_token (<code>str</code>, <em>optional</em>, defaults to <code>&quot;[PAD]&quot;</code>):
The token used for padding, for example when batching sequences of different lengths.
cls_token (<code>str</code>, <em>optional</em>, defaults to <code>&quot;[CLS]&quot;</code>):
The classifier token which is used when doing sequence classification (classification of the whole sequence
instead of per-token classification). It is the first token of the sequence when built with special tokens.
mask_token (<code>str</code>, <em>optional</em>, defaults to <code>&quot;[MASK]&quot;</code>):
The token used for masking values. This is the token used when training this model with masked language
modeling. This is the token which the model will try to predict.
tokenize_chinese_chars (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>):
Whether or not to tokenize Chinese characters. This should likely be deactivated for Japanese (see this
<a href="https://github.com/huggingface/transformers/issues/328" rel="nofollow">issue</a>).
strip_accents (<code>bool</code>, <em>optional</em>):
Whether or not to strip all accents. If this option is not specified, then it will be determined by the
value for <code>lowercase</code> (as in the original BERT).`,name:"methods."}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/roc_bert/tokenization_roc_bert.py#L73"}}),Ae=new B({props:{name:"build_inputs_with_special_tokens",anchor:"transformers.RoCBertTokenizer.build_inputs_with_special_tokens",parameters:[{name:"token_ids_0",val:": list"},{name:"token_ids_1",val:": typing.Optional[list[int]] = None"},{name:"cls_token_id",val:": typing.Optional[int] = None"},{name:"sep_token_id",val:": typing.Optional[int] = None"}],parametersDescription:[{anchor:"transformers.RoCBertTokenizer.build_inputs_with_special_tokens.token_ids_0",description:`<strong>token_ids_0</strong> (<code>List[int]</code>) &#x2014;
List of IDs to which the special tokens will be added.`,name:"token_ids_0"},{anchor:"transformers.RoCBertTokenizer.build_inputs_with_special_tokens.token_ids_1",description:`<strong>token_ids_1</strong> (<code>List[int]</code>, <em>optional</em>) &#x2014;
Optional second list of IDs for sequence pairs.`,name:"token_ids_1"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/roc_bert/tokenization_roc_bert.py#L769",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>List of <a href="../glossary#input-ids">input IDs</a> with the appropriate special tokens.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>List[int]</code></p>
`}}),Oe=new B({props:{name:"get_special_tokens_mask",anchor:"transformers.RoCBertTokenizer.get_special_tokens_mask",parameters:[{name:"token_ids_0",val:": list"},{name:"token_ids_1",val:": typing.Optional[list[int]] = None"},{name:"already_has_special_tokens",val:": bool = False"}],parametersDescription:[{anchor:"transformers.RoCBertTokenizer.get_special_tokens_mask.token_ids_0",description:`<strong>token_ids_0</strong> (<code>List[int]</code>) &#x2014;
List of IDs.`,name:"token_ids_0"},{anchor:"transformers.RoCBertTokenizer.get_special_tokens_mask.token_ids_1",description:`<strong>token_ids_1</strong> (<code>List[int]</code>, <em>optional</em>) &#x2014;
Optional second list of IDs for sequence pairs.`,name:"token_ids_1"},{anchor:"transformers.RoCBertTokenizer.get_special_tokens_mask.already_has_special_tokens",description:`<strong>already_has_special_tokens</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not the token list is already formatted with special tokens for the model.`,name:"already_has_special_tokens"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/roc_bert/tokenization_roc_bert.py#L799",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>List[int]</code></p>
`}}),Ee=new B({props:{name:"create_token_type_ids_from_sequences",anchor:"transformers.RoCBertTokenizer.create_token_type_ids_from_sequences",parameters:[{name:"token_ids_0",val:": list"},{name:"token_ids_1",val:": typing.Optional[list[int]] = None"}],parametersDescription:[{anchor:"transformers.RoCBertTokenizer.create_token_type_ids_from_sequences.token_ids_0",description:"<strong>token_ids_0</strong> (<code>list[int]</code>) &#x2014; The first tokenized sequence.",name:"token_ids_0"},{anchor:"transformers.RoCBertTokenizer.create_token_type_ids_from_sequences.token_ids_1",description:"<strong>token_ids_1</strong> (<code>list[int]</code>, <em>optional</em>) &#x2014; The second tokenized sequence.",name:"token_ids_1"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L3432",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>The token type ids.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>list[int]</code></p>
`}}),Ye=new B({props:{name:"save_vocabulary",anchor:"transformers.RoCBertTokenizer.save_vocabulary",parameters:[{name:"save_directory",val:": str"},{name:"filename_prefix",val:": typing.Optional[str] = None"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/roc_bert/tokenization_roc_bert.py#L827"}}),De=new ie({props:{title:"RoCBertModel",local:"transformers.RoCBertModel",headingTag:"h2"}}),Ke=new B({props:{name:"class transformers.RoCBertModel",anchor:"transformers.RoCBertModel",parameters:[{name:"config",val:""},{name:"add_pooling_layer",val:" = True"}],parametersDescription:[{anchor:"transformers.RoCBertModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/roc_bert#transformers.RoCBertModel">RoCBertModel</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"},{anchor:"transformers.RoCBertModel.add_pooling_layer",description:`<strong>add_pooling_layer</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether to add a pooling layer`,name:"add_pooling_layer"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/roc_bert/modeling_roc_bert.py#L764"}}),et=new B({props:{name:"forward",anchor:"transformers.RoCBertModel.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"input_shape_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"input_pronunciation_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_hidden_states",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"past_key_values",val:": typing.Optional[list[torch.FloatTensor]] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.RoCBertModel.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.RoCBertModel.forward.input_shape_ids",description:`<strong>input_shape_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the shape vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input_shape_ids">What are input IDs?</a>`,name:"input_shape_ids"},{anchor:"transformers.RoCBertModel.forward.input_pronunciation_ids",description:`<strong>input_pronunciation_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the pronunciation vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input_pronunciation_ids">What are input IDs?</a>`,name:"input_pronunciation_ids"},{anchor:"transformers.RoCBertModel.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.RoCBertModel.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.RoCBertModel.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.RoCBertModel.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.RoCBertModel.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.RoCBertModel.forward.encoder_hidden_states",description:`<strong>encoder_hidden_states</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
if the model is configured as a decoder.`,name:"encoder_hidden_states"},{anchor:"transformers.RoCBertModel.forward.encoder_attention_mask",description:`<strong>encoder_attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
the cross-attention if the model is configured as a decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>`,name:"encoder_attention_mask"},{anchor:"transformers.RoCBertModel.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>list[torch.FloatTensor]</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.RoCBertModel.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.RoCBertModel.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.RoCBertModel.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.RoCBertModel.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/roc_bert/modeling_roc_bert.py#L811",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions"
>transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/roc_bert#transformers.RoCBertConfig"
>RoCBertConfig</a>) and inputs.</p>
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
`}}),Me=new ge({props:{$$slots:{default:[dr]},$$scope:{ctx:w}}}),tt=new ie({props:{title:"RoCBertForPreTraining",local:"transformers.RoCBertForPreTraining",headingTag:"h2"}}),ot=new B({props:{name:"class transformers.RoCBertForPreTraining",anchor:"transformers.RoCBertForPreTraining",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.RoCBertForPreTraining.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/roc_bert#transformers.RoCBertForPreTraining">RoCBertForPreTraining</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/roc_bert/modeling_roc_bert.py#L953"}}),nt=new B({props:{name:"forward",anchor:"transformers.RoCBertForPreTraining.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"input_shape_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"input_pronunciation_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attack_input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attack_input_shape_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attack_input_pronunciation_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attack_attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"attack_token_type_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"labels_input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"labels_input_shape_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"labels_input_pronunciation_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"labels_attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"labels_token_type_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.RoCBertForPreTraining.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.RoCBertForPreTraining.forward.input_shape_ids",description:`<strong>input_shape_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the shape vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input_shape_ids">What are input IDs?</a>`,name:"input_shape_ids"},{anchor:"transformers.RoCBertForPreTraining.forward.input_pronunciation_ids",description:`<strong>input_pronunciation_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the pronunciation vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input_pronunciation_ids">What are input IDs?</a>`,name:"input_pronunciation_ids"},{anchor:"transformers.RoCBertForPreTraining.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.RoCBertForPreTraining.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.RoCBertForPreTraining.forward.attack_input_ids",description:`<strong>attack_input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
attack sample ids for computing the contrastive loss. Indices should be in <code>[-100, 0, ..., config.vocab_size]</code> (see <code>input_ids</code> docstring) Tokens with indices set to <code>-100</code> are ignored (masked),
the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>`,name:"attack_input_ids"},{anchor:"transformers.RoCBertForPreTraining.forward.attack_input_shape_ids",description:`<strong>attack_input_shape_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
attack sample shape ids for computing the contrastive loss. Indices should be in <code>[-100, 0, ..., config.vocab_size]</code> (see <code>input_ids</code> docstring) Tokens with indices set to <code>-100</code> are ignored (masked),
the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>`,name:"attack_input_shape_ids"},{anchor:"transformers.RoCBertForPreTraining.forward.attack_input_pronunciation_ids",description:`<strong>attack_input_pronunciation_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
attack sample pronunciation ids for computing the contrastive loss. Indices should be in <code>[-100, 0, ..., config.vocab_size]</code> (see <code>input_ids</code> docstring) Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>`,name:"attack_input_pronunciation_ids"},{anchor:"transformers.RoCBertForPreTraining.forward.attack_attention_mask",description:`<strong>attack_attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices for the attack sample. Mask values selected in
<code>[0, 1]</code>: <code>1</code> for tokens that are NOT MASKED, <code>0</code> for MASKED tokens.`,name:"attack_attention_mask"},{anchor:"transformers.RoCBertForPreTraining.forward.attack_token_type_ids",description:`<strong>attack_token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate different portions of the attack inputs. Indices are selected in <code>[0, 1]</code>:
<code>0</code> corresponds to a sentence A token, <code>1</code> corresponds to a sentence B token.`,name:"attack_token_type_ids"},{anchor:"transformers.RoCBertForPreTraining.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.RoCBertForPreTraining.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.RoCBertForPreTraining.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.RoCBertForPreTraining.forward.labels_input_ids",description:`<strong>labels_input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
target ids for computing the contrastive loss and masked_lm_loss . Indices should be in <code>[-100, 0, ..., config.vocab_size]</code> (see <code>input_ids</code> docstring) Tokens with indices set to <code>-100</code> are ignored (masked),
the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>`,name:"labels_input_ids"},{anchor:"transformers.RoCBertForPreTraining.forward.labels_input_shape_ids",description:`<strong>labels_input_shape_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
target shape ids for computing the contrastive loss and masked_lm_loss . Indices should be in <code>[-100, 0, ..., config.vocab_size]</code> (see <code>input_ids</code> docstring) Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>`,name:"labels_input_shape_ids"},{anchor:"transformers.RoCBertForPreTraining.forward.labels_input_pronunciation_ids",description:`<strong>labels_input_pronunciation_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
target pronunciation ids for computing the contrastive loss and masked_lm_loss . Indices should be in
<code>[-100, 0, ..., config.vocab_size]</code> (see <code>input_ids</code> docstring) Tokens with indices set to <code>-100</code> are
ignored (masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>`,name:"labels_input_pronunciation_ids"},{anchor:"transformers.RoCBertForPreTraining.forward.labels_attention_mask",description:`<strong>labels_attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices for the label sample. Mask values selected in
<code>[0, 1]</code>: <code>1</code> for tokens that are NOT MASKED, <code>0</code> for MASKED tokens.`,name:"labels_attention_mask"},{anchor:"transformers.RoCBertForPreTraining.forward.labels_token_type_ids",description:`<strong>labels_token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate different portions of the label inputs. Indices are selected in <code>[0, 1]</code>:
<code>0</code> corresponds to a sentence A token, <code>1</code> corresponds to a sentence B token.`,name:"labels_token_type_ids"},{anchor:"transformers.RoCBertForPreTraining.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.RoCBertForPreTraining.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.RoCBertForPreTraining.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/roc_bert/modeling_roc_bert.py#L974",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.MaskedLMOutput"
>transformers.modeling_outputs.MaskedLMOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/roc_bert#transformers.RoCBertConfig"
>RoCBertConfig</a>) and inputs.</p>
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
`}}),we=new ge({props:{$$slots:{default:[cr]},$$scope:{ctx:w}}}),Ce=new _e({props:{anchor:"transformers.RoCBertForPreTraining.forward.example",$$slots:{default:[pr]},$$scope:{ctx:w}}}),st=new ie({props:{title:"RoCBertForCausalLM",local:"transformers.RoCBertForCausalLM",headingTag:"h2"}}),rt=new B({props:{name:"class transformers.RoCBertForCausalLM",anchor:"transformers.RoCBertForCausalLM",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.RoCBertForCausalLM.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/roc_bert#transformers.RoCBertForCausalLM">RoCBertForCausalLM</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/roc_bert/modeling_roc_bert.py#L1316"}}),at=new B({props:{name:"forward",anchor:"transformers.RoCBertForCausalLM.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"input_shape_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"input_pronunciation_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_hidden_states",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"past_key_values",val:": typing.Optional[list[torch.Tensor]] = None"},{name:"labels",val:": typing.Optional[torch.Tensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.RoCBertForCausalLM.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.RoCBertForCausalLM.forward.input_shape_ids",description:`<strong>input_shape_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the shape vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input_shape_ids">What are input IDs?</a>`,name:"input_shape_ids"},{anchor:"transformers.RoCBertForCausalLM.forward.input_pronunciation_ids",description:`<strong>input_pronunciation_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the pronunciation vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input_pronunciation_ids">What are input IDs?</a>`,name:"input_pronunciation_ids"},{anchor:"transformers.RoCBertForCausalLM.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.RoCBertForCausalLM.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.RoCBertForCausalLM.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.RoCBertForCausalLM.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.RoCBertForCausalLM.forward.encoder_hidden_states",description:`<strong>encoder_hidden_states</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
if the model is configured as a decoder.`,name:"encoder_hidden_states"},{anchor:"transformers.RoCBertForCausalLM.forward.encoder_attention_mask",description:`<strong>encoder_attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
the cross-attention if the model is configured as a decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>`,name:"encoder_attention_mask"},{anchor:"transformers.RoCBertForCausalLM.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.RoCBertForCausalLM.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>list[torch.Tensor]</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.RoCBertForCausalLM.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the left-to-right language modeling loss (next word prediction). Indices should be in
<code>[-100, 0, ..., config.vocab_size]</code> (see <code>input_ids</code> docstring) Tokens with indices set to <code>-100</code> are
ignored (masked), the loss is only computed for the tokens with labels n <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.RoCBertForCausalLM.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.RoCBertForCausalLM.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.RoCBertForCausalLM.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.RoCBertForCausalLM.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/roc_bert/modeling_roc_bert.py#L1341",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithCrossAttentions"
>transformers.modeling_outputs.CausalLMOutputWithCrossAttentions</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/roc_bert#transformers.RoCBertConfig"
>RoCBertConfig</a>) and inputs.</p>
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
`}}),ve=new ge({props:{$$slots:{default:[mr]},$$scope:{ctx:w}}}),Be=new _e({props:{anchor:"transformers.RoCBertForCausalLM.forward.example",$$slots:{default:[hr]},$$scope:{ctx:w}}}),it=new ie({props:{title:"RoCBertForMaskedLM",local:"transformers.RoCBertForMaskedLM",headingTag:"h2"}}),lt=new B({props:{name:"class transformers.RoCBertForMaskedLM",anchor:"transformers.RoCBertForMaskedLM",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.RoCBertForMaskedLM.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/roc_bert#transformers.RoCBertForMaskedLM">RoCBertForMaskedLM</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/roc_bert/modeling_roc_bert.py#L1159"}}),dt=new B({props:{name:"forward",anchor:"transformers.RoCBertForMaskedLM.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"input_shape_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"input_pronunciation_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_hidden_states",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"labels",val:": typing.Optional[torch.Tensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.RoCBertForMaskedLM.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.RoCBertForMaskedLM.forward.input_shape_ids",description:`<strong>input_shape_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the shape vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input_shape_ids">What are input IDs?</a>`,name:"input_shape_ids"},{anchor:"transformers.RoCBertForMaskedLM.forward.input_pronunciation_ids",description:`<strong>input_pronunciation_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the pronunciation vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input_pronunciation_ids">What are input IDs?</a>`,name:"input_pronunciation_ids"},{anchor:"transformers.RoCBertForMaskedLM.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.RoCBertForMaskedLM.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.RoCBertForMaskedLM.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.RoCBertForMaskedLM.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.RoCBertForMaskedLM.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.RoCBertForMaskedLM.forward.encoder_hidden_states",description:`<strong>encoder_hidden_states</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
if the model is configured as a decoder.`,name:"encoder_hidden_states"},{anchor:"transformers.RoCBertForMaskedLM.forward.encoder_attention_mask",description:`<strong>encoder_attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
the cross-attention if the model is configured as a decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>`,name:"encoder_attention_mask"},{anchor:"transformers.RoCBertForMaskedLM.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should be in <code>[-100, 0, ..., config.vocab_size]</code> (see <code>input_ids</code> docstring) Tokens with indices set to <code>-100</code> are ignored (masked), the
loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.RoCBertForMaskedLM.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.RoCBertForMaskedLM.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.RoCBertForMaskedLM.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/roc_bert/modeling_roc_bert.py#L1187",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.MaskedLMOutput"
>transformers.modeling_outputs.MaskedLMOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/roc_bert#transformers.RoCBertConfig"
>RoCBertConfig</a>) and inputs.</p>
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
`}}),ze=new ge({props:{$$slots:{default:[ur]},$$scope:{ctx:w}}}),$e=new _e({props:{anchor:"transformers.RoCBertForMaskedLM.forward.example",$$slots:{default:[fr]},$$scope:{ctx:w}}}),ct=new ie({props:{title:"RoCBertForSequenceClassification",local:"transformers.RoCBertForSequenceClassification",headingTag:"h2"}}),pt=new B({props:{name:"class transformers.RoCBertForSequenceClassification",anchor:"transformers.RoCBertForSequenceClassification",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.RoCBertForSequenceClassification.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/roc_bert#transformers.RoCBertForSequenceClassification">RoCBertForSequenceClassification</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/roc_bert/modeling_roc_bert.py#L1493"}}),mt=new B({props:{name:"forward",anchor:"transformers.RoCBertForSequenceClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"input_shape_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"input_pronunciation_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"labels",val:": typing.Optional[torch.Tensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.RoCBertForSequenceClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.RoCBertForSequenceClassification.forward.input_shape_ids",description:`<strong>input_shape_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the shape vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input_shape_ids">What are input IDs?</a>`,name:"input_shape_ids"},{anchor:"transformers.RoCBertForSequenceClassification.forward.input_pronunciation_ids",description:`<strong>input_pronunciation_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the pronunciation vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input_pronunciation_ids">What are input IDs?</a>`,name:"input_pronunciation_ids"},{anchor:"transformers.RoCBertForSequenceClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.RoCBertForSequenceClassification.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.RoCBertForSequenceClassification.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.RoCBertForSequenceClassification.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.RoCBertForSequenceClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.RoCBertForSequenceClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the sequence classification/regression loss. Indices should be in <code>[0, ..., config.num_labels - 1]</code>. If <code>config.num_labels == 1</code> a regression loss is computed (Mean-Square loss), If
<code>config.num_labels &gt; 1</code> a classification loss is computed (Cross-Entropy).`,name:"labels"},{anchor:"transformers.RoCBertForSequenceClassification.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.RoCBertForSequenceClassification.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.RoCBertForSequenceClassification.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/roc_bert/modeling_roc_bert.py#L1510",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput"
>transformers.modeling_outputs.SequenceClassifierOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/roc_bert#transformers.RoCBertConfig"
>RoCBertConfig</a>) and inputs.</p>
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
`}}),Re=new ge({props:{$$slots:{default:[gr]},$$scope:{ctx:w}}}),Je=new _e({props:{anchor:"transformers.RoCBertForSequenceClassification.forward.example",$$slots:{default:[_r]},$$scope:{ctx:w}}}),Ue=new _e({props:{anchor:"transformers.RoCBertForSequenceClassification.forward.example-2",$$slots:{default:[br]},$$scope:{ctx:w}}}),ht=new ie({props:{title:"RoCBertForMultipleChoice",local:"transformers.RoCBertForMultipleChoice",headingTag:"h2"}}),ut=new B({props:{name:"class transformers.RoCBertForMultipleChoice",anchor:"transformers.RoCBertForMultipleChoice",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.RoCBertForMultipleChoice.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/roc_bert#transformers.RoCBertForMultipleChoice">RoCBertForMultipleChoice</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/roc_bert/modeling_roc_bert.py#L1602"}}),ft=new B({props:{name:"forward",anchor:"transformers.RoCBertForMultipleChoice.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"input_shape_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"input_pronunciation_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"labels",val:": typing.Optional[torch.Tensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.RoCBertForMultipleChoice.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, num_choices, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.RoCBertForMultipleChoice.forward.input_shape_ids",description:`<strong>input_shape_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, num_choices, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the shape vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input_shape_ids">What are input IDs?</a>`,name:"input_shape_ids"},{anchor:"transformers.RoCBertForMultipleChoice.forward.input_pronunciation_ids",description:`<strong>input_pronunciation_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, num_choices, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the pronunciation vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input_pronunciation_ids">What are input IDs?</a>`,name:"input_pronunciation_ids"},{anchor:"transformers.RoCBertForMultipleChoice.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.RoCBertForMultipleChoice.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, num_choices, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.RoCBertForMultipleChoice.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, num_choices, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.max_position_embeddings - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.RoCBertForMultipleChoice.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.RoCBertForMultipleChoice.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_choices, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <em>input_ids</em> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.RoCBertForMultipleChoice.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the multiple choice classification loss. Indices should be in <code>[0, ..., num_choices-1]</code> where <code>num_choices</code> is the size of the second dimension of the input tensors. (See
<code>input_ids</code> above)`,name:"labels"},{anchor:"transformers.RoCBertForMultipleChoice.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.RoCBertForMultipleChoice.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.RoCBertForMultipleChoice.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/roc_bert/modeling_roc_bert.py#L1617",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.MultipleChoiceModelOutput"
>transformers.modeling_outputs.MultipleChoiceModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/roc_bert#transformers.RoCBertConfig"
>RoCBertConfig</a>) and inputs.</p>
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
`}}),je=new ge({props:{$$slots:{default:[Tr]},$$scope:{ctx:w}}}),Fe=new _e({props:{anchor:"transformers.RoCBertForMultipleChoice.forward.example",$$slots:{default:[yr]},$$scope:{ctx:w}}}),gt=new ie({props:{title:"RoCBertForTokenClassification",local:"transformers.RoCBertForTokenClassification",headingTag:"h2"}}),_t=new B({props:{name:"class transformers.RoCBertForTokenClassification",anchor:"transformers.RoCBertForTokenClassification",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.RoCBertForTokenClassification.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/roc_bert#transformers.RoCBertForTokenClassification">RoCBertForTokenClassification</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/roc_bert/modeling_roc_bert.py#L1734"}}),bt=new B({props:{name:"forward",anchor:"transformers.RoCBertForTokenClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"input_shape_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"input_pronunciation_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"labels",val:": typing.Optional[torch.Tensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.RoCBertForTokenClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.RoCBertForTokenClassification.forward.input_shape_ids",description:`<strong>input_shape_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the shape vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input_shape_ids">What are input IDs?</a>`,name:"input_shape_ids"},{anchor:"transformers.RoCBertForTokenClassification.forward.input_pronunciation_ids",description:`<strong>input_pronunciation_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the pronunciation vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input_pronunciation_ids">What are input IDs?</a>`,name:"input_pronunciation_ids"},{anchor:"transformers.RoCBertForTokenClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.RoCBertForTokenClassification.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.RoCBertForTokenClassification.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.RoCBertForTokenClassification.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.RoCBertForTokenClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.RoCBertForTokenClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the token classification loss. Indices should be in <code>[0, ..., config.num_labels - 1]</code>.`,name:"labels"},{anchor:"transformers.RoCBertForTokenClassification.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.RoCBertForTokenClassification.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.RoCBertForTokenClassification.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/roc_bert/modeling_roc_bert.py#L1750",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput"
>transformers.modeling_outputs.TokenClassifierOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/roc_bert#transformers.RoCBertConfig"
>RoCBertConfig</a>) and inputs.</p>
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
`}}),xe=new ge({props:{$$slots:{default:[kr]},$$scope:{ctx:w}}}),Ie=new _e({props:{anchor:"transformers.RoCBertForTokenClassification.forward.example",$$slots:{default:[Mr]},$$scope:{ctx:w}}}),Tt=new ie({props:{title:"RoCBertForQuestionAnswering",local:"transformers.RoCBertForQuestionAnswering",headingTag:"h2"}}),yt=new B({props:{name:"class transformers.RoCBertForQuestionAnswering",anchor:"transformers.RoCBertForQuestionAnswering",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.RoCBertForQuestionAnswering.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/roc_bert#transformers.RoCBertForQuestionAnswering">RoCBertForQuestionAnswering</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/roc_bert/modeling_roc_bert.py#L1823"}}),kt=new B({props:{name:"forward",anchor:"transformers.RoCBertForQuestionAnswering.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"input_shape_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"input_pronunciation_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"start_positions",val:": typing.Optional[torch.Tensor] = None"},{name:"end_positions",val:": typing.Optional[torch.Tensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.RoCBertForQuestionAnswering.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.RoCBertForQuestionAnswering.forward.input_shape_ids",description:`<strong>input_shape_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the shape vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input_shape_ids">What are input IDs?</a>`,name:"input_shape_ids"},{anchor:"transformers.RoCBertForQuestionAnswering.forward.input_pronunciation_ids",description:`<strong>input_pronunciation_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the pronunciation vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input_pronunciation_ids">What are input IDs?</a>`,name:"input_pronunciation_ids"},{anchor:"transformers.RoCBertForQuestionAnswering.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.RoCBertForQuestionAnswering.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.RoCBertForQuestionAnswering.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.RoCBertForQuestionAnswering.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.RoCBertForQuestionAnswering.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.RoCBertForQuestionAnswering.forward.start_positions",description:`<strong>start_positions</strong> (<code>torch.Tensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for position (index) of the start of the labelled span for computing the token classification loss.
Positions are clamped to the length of the sequence (<code>sequence_length</code>). Position outside of the sequence
are not taken into account for computing the loss.`,name:"start_positions"},{anchor:"transformers.RoCBertForQuestionAnswering.forward.end_positions",description:`<strong>end_positions</strong> (<code>torch.Tensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for position (index) of the end of the labelled span for computing the token classification loss.
Positions are clamped to the length of the sequence (<code>sequence_length</code>). Position outside of the sequence
are not taken into account for computing the loss.`,name:"end_positions"},{anchor:"transformers.RoCBertForQuestionAnswering.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.RoCBertForQuestionAnswering.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.RoCBertForQuestionAnswering.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/roc_bert/modeling_roc_bert.py#L1835",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.QuestionAnsweringModelOutput"
>transformers.modeling_outputs.QuestionAnsweringModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/roc_bert#transformers.RoCBertConfig"
>RoCBertConfig</a>) and inputs.</p>
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
`}}),We=new ge({props:{$$slots:{default:[wr]},$$scope:{ctx:w}}}),Ve=new _e({props:{anchor:"transformers.RoCBertForQuestionAnswering.forward.example",$$slots:{default:[Cr]},$$scope:{ctx:w}}}),Mt=new tr({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/roc_bert.md"}}),{c(){t=c("meta"),m=r(),o=c("p"),i=r(),M=c("p"),M.innerHTML=n,u=r(),z=c("div"),z.innerHTML=mo,Ze=r(),f(le.$$.fragment),fo=r(),Se=c("p"),Se.innerHTML=as,go=r(),Le=c("p"),Le.innerHTML=is,_o=r(),f(be.$$.fragment),bo=r(),Xe=c("p"),Xe.innerHTML=ls,To=r(),f(Te.$$.fragment),yo=r(),f(Pe.$$.fragment),ko=r(),Z=c("div"),f(Qe.$$.fragment),Xo=r(),Ct=c("p"),Ct.innerHTML=ds,Po=r(),vt=c("p"),vt.innerHTML=cs,Qo=r(),f(ye.$$.fragment),Mo=r(),f(Ge.$$.fragment),wo=r(),J=c("div"),f(He.$$.fragment),Go=r(),de=c("div"),f(Ae.$$.fragment),Ho=r(),Bt=c("p"),Bt.textContent=ps,Ao=r(),zt=c("ul"),zt.innerHTML=ms,Oo=r(),ke=c("div"),f(Oe.$$.fragment),Eo=r(),$t=c("p"),$t.innerHTML=hs,Yo=r(),ce=c("div"),f(Ee.$$.fragment),Do=r(),Rt=c("p"),Rt.innerHTML=us,Ko=r(),Jt=c("p"),Jt.textContent=fs,en=r(),Ut=c("div"),f(Ye.$$.fragment),Co=r(),f(De.$$.fragment),vo=r(),R=c("div"),f(Ke.$$.fragment),tn=r(),jt=c("p"),jt.innerHTML=gs,on=r(),Ft=c("p"),Ft.innerHTML=_s,nn=r(),xt=c("p"),xt.innerHTML=bs,sn=r(),It=c("p"),It.innerHTML=Ts,rn=r(),pe=c("div"),f(et.$$.fragment),an=r(),Wt=c("p"),Wt.innerHTML=ys,ln=r(),f(Me.$$.fragment),Bo=r(),f(tt.$$.fragment),zo=r(),U=c("div"),f(ot.$$.fragment),dn=r(),Vt=c("p"),Vt.textContent=ks,cn=r(),Nt=c("p"),Nt.innerHTML=Ms,pn=r(),qt=c("p"),qt.innerHTML=ws,mn=r(),O=c("div"),f(nt.$$.fragment),hn=r(),Zt=c("p"),Zt.innerHTML=Cs,un=r(),f(we.$$.fragment),fn=r(),f(Ce.$$.fragment),$o=r(),f(st.$$.fragment),Ro=r(),j=c("div"),f(rt.$$.fragment),gn=r(),St=c("p"),St.innerHTML=vs,_n=r(),Lt=c("p"),Lt.innerHTML=Bs,bn=r(),Xt=c("p"),Xt.innerHTML=zs,Tn=r(),E=c("div"),f(at.$$.fragment),yn=r(),Pt=c("p"),Pt.innerHTML=$s,kn=r(),f(ve.$$.fragment),Mn=r(),f(Be.$$.fragment),Jo=r(),f(it.$$.fragment),Uo=r(),F=c("div"),f(lt.$$.fragment),wn=r(),Qt=c("p"),Qt.innerHTML=Rs,Cn=r(),Gt=c("p"),Gt.innerHTML=Js,vn=r(),Ht=c("p"),Ht.innerHTML=Us,Bn=r(),Y=c("div"),f(dt.$$.fragment),zn=r(),At=c("p"),At.innerHTML=js,$n=r(),f(ze.$$.fragment),Rn=r(),f($e.$$.fragment),jo=r(),f(ct.$$.fragment),Fo=r(),x=c("div"),f(pt.$$.fragment),Jn=r(),Ot=c("p"),Ot.textContent=Fs,Un=r(),Et=c("p"),Et.innerHTML=xs,jn=r(),Yt=c("p"),Yt.innerHTML=Is,Fn=r(),q=c("div"),f(mt.$$.fragment),xn=r(),Dt=c("p"),Dt.innerHTML=Ws,In=r(),f(Re.$$.fragment),Wn=r(),f(Je.$$.fragment),Vn=r(),f(Ue.$$.fragment),xo=r(),f(ht.$$.fragment),Io=r(),I=c("div"),f(ut.$$.fragment),Nn=r(),Kt=c("p"),Kt.textContent=Vs,qn=r(),eo=c("p"),eo.innerHTML=Ns,Zn=r(),to=c("p"),to.innerHTML=qs,Sn=r(),D=c("div"),f(ft.$$.fragment),Ln=r(),oo=c("p"),oo.innerHTML=Zs,Xn=r(),f(je.$$.fragment),Pn=r(),f(Fe.$$.fragment),Wo=r(),f(gt.$$.fragment),Vo=r(),W=c("div"),f(_t.$$.fragment),Qn=r(),no=c("p"),no.textContent=Ss,Gn=r(),so=c("p"),so.innerHTML=Ls,Hn=r(),ro=c("p"),ro.innerHTML=Xs,An=r(),K=c("div"),f(bt.$$.fragment),On=r(),ao=c("p"),ao.innerHTML=Ps,En=r(),f(xe.$$.fragment),Yn=r(),f(Ie.$$.fragment),No=r(),f(Tt.$$.fragment),qo=r(),V=c("div"),f(yt.$$.fragment),Dn=r(),io=c("p"),io.innerHTML=Qs,Kn=r(),lo=c("p"),lo.innerHTML=Gs,es=r(),co=c("p"),co.innerHTML=Hs,ts=r(),ee=c("div"),f(kt.$$.fragment),os=r(),po=c("p"),po.innerHTML=As,ns=r(),f(We.$$.fragment),ss=r(),f(Ve.$$.fragment),Zo=r(),f(Mt.$$.fragment),So=r(),ho=c("p"),this.h()},l(e){const l=Ks("svelte-u9bgzb",document.head);t=p(l,"META",{name:!0,content:!0}),l.forEach(d),m=a(e),o=p(e,"P",{}),C(o).forEach(d),i=a(e),M=p(e,"P",{"data-svelte-h":!0}),k(M)!=="svelte-13t2bfi"&&(M.innerHTML=n),u=a(e),z=p(e,"DIV",{style:!0,"data-svelte-h":!0}),k(z)!=="svelte-1upb38l"&&(z.innerHTML=mo),Ze=a(e),g(le.$$.fragment,e),fo=a(e),Se=p(e,"P",{"data-svelte-h":!0}),k(Se)!=="svelte-1w9ds1y"&&(Se.innerHTML=as),go=a(e),Le=p(e,"P",{"data-svelte-h":!0}),k(Le)!=="svelte-1fbrrra"&&(Le.innerHTML=is),_o=a(e),g(be.$$.fragment,e),bo=a(e),Xe=p(e,"P",{"data-svelte-h":!0}),k(Xe)!=="svelte-l96m4o"&&(Xe.innerHTML=ls),To=a(e),g(Te.$$.fragment,e),yo=a(e),g(Pe.$$.fragment,e),ko=a(e),Z=p(e,"DIV",{class:!0});var oe=C(Z);g(Qe.$$.fragment,oe),Xo=a(oe),Ct=p(oe,"P",{"data-svelte-h":!0}),k(Ct)!=="svelte-18bojof"&&(Ct.innerHTML=ds),Po=a(oe),vt=p(oe,"P",{"data-svelte-h":!0}),k(vt)!=="svelte-1ek1ss9"&&(vt.innerHTML=cs),Qo=a(oe),g(ye.$$.fragment,oe),oe.forEach(d),Mo=a(e),g(Ge.$$.fragment,e),wo=a(e),J=p(e,"DIV",{class:!0});var S=C(J);g(He.$$.fragment,S),Go=a(S),de=p(S,"DIV",{class:!0});var he=C(de);g(Ae.$$.fragment,he),Ho=a(he),Bt=p(he,"P",{"data-svelte-h":!0}),k(Bt)!=="svelte-t7qurq"&&(Bt.textContent=ps),Ao=a(he),zt=p(he,"UL",{"data-svelte-h":!0}),k(zt)!=="svelte-xi6653"&&(zt.innerHTML=ms),he.forEach(d),Oo=a(S),ke=p(S,"DIV",{class:!0});var wt=C(ke);g(Oe.$$.fragment,wt),Eo=a(wt),$t=p(wt,"P",{"data-svelte-h":!0}),k($t)!=="svelte-1f4f5kp"&&($t.innerHTML=hs),wt.forEach(d),Yo=a(S),ce=p(S,"DIV",{class:!0});var ue=C(ce);g(Ee.$$.fragment,ue),Do=a(ue),Rt=p(ue,"P",{"data-svelte-h":!0}),k(Rt)!=="svelte-zj1vf1"&&(Rt.innerHTML=us),Ko=a(ue),Jt=p(ue,"P",{"data-svelte-h":!0}),k(Jt)!=="svelte-9vptpw"&&(Jt.textContent=fs),ue.forEach(d),en=a(S),Ut=p(S,"DIV",{class:!0});var uo=C(Ut);g(Ye.$$.fragment,uo),uo.forEach(d),S.forEach(d),Co=a(e),g(De.$$.fragment,e),vo=a(e),R=p(e,"DIV",{class:!0});var N=C(R);g(Ke.$$.fragment,N),tn=a(N),jt=p(N,"P",{"data-svelte-h":!0}),k(jt)!=="svelte-1854dma"&&(jt.innerHTML=gs),on=a(N),Ft=p(N,"P",{"data-svelte-h":!0}),k(Ft)!=="svelte-1c0aj2z"&&(Ft.innerHTML=_s),nn=a(N),xt=p(N,"P",{"data-svelte-h":!0}),k(xt)!=="svelte-q52n56"&&(xt.innerHTML=bs),sn=a(N),It=p(N,"P",{"data-svelte-h":!0}),k(It)!=="svelte-hswkmf"&&(It.innerHTML=Ts),rn=a(N),pe=p(N,"DIV",{class:!0});var fe=C(pe);g(et.$$.fragment,fe),an=a(fe),Wt=p(fe,"P",{"data-svelte-h":!0}),k(Wt)!=="svelte-1cwh831"&&(Wt.innerHTML=ys),ln=a(fe),g(Me.$$.fragment,fe),fe.forEach(d),N.forEach(d),Bo=a(e),g(tt.$$.fragment,e),zo=a(e),U=p(e,"DIV",{class:!0});var L=C(U);g(ot.$$.fragment,L),dn=a(L),Vt=p(L,"P",{"data-svelte-h":!0}),k(Vt)!=="svelte-1nv57qw"&&(Vt.textContent=ks),cn=a(L),Nt=p(L,"P",{"data-svelte-h":!0}),k(Nt)!=="svelte-q52n56"&&(Nt.innerHTML=Ms),pn=a(L),qt=p(L,"P",{"data-svelte-h":!0}),k(qt)!=="svelte-hswkmf"&&(qt.innerHTML=ws),mn=a(L),O=p(L,"DIV",{class:!0});var ne=C(O);g(nt.$$.fragment,ne),hn=a(ne),Zt=p(ne,"P",{"data-svelte-h":!0}),k(Zt)!=="svelte-10lbl8l"&&(Zt.innerHTML=Cs),un=a(ne),g(we.$$.fragment,ne),fn=a(ne),g(Ce.$$.fragment,ne),ne.forEach(d),L.forEach(d),$o=a(e),g(st.$$.fragment,e),Ro=a(e),j=p(e,"DIV",{class:!0});var X=C(j);g(rt.$$.fragment,X),gn=a(X),St=p(X,"P",{"data-svelte-h":!0}),k(St)!=="svelte-1fwo07i"&&(St.innerHTML=vs),_n=a(X),Lt=p(X,"P",{"data-svelte-h":!0}),k(Lt)!=="svelte-q52n56"&&(Lt.innerHTML=Bs),bn=a(X),Xt=p(X,"P",{"data-svelte-h":!0}),k(Xt)!=="svelte-hswkmf"&&(Xt.innerHTML=zs),Tn=a(X),E=p(X,"DIV",{class:!0});var se=C(E);g(at.$$.fragment,se),yn=a(se),Pt=p(se,"P",{"data-svelte-h":!0}),k(Pt)!=="svelte-1jvrwgd"&&(Pt.innerHTML=$s),kn=a(se),g(ve.$$.fragment,se),Mn=a(se),g(Be.$$.fragment,se),se.forEach(d),X.forEach(d),Jo=a(e),g(it.$$.fragment,e),Uo=a(e),F=p(e,"DIV",{class:!0});var P=C(F);g(lt.$$.fragment,P),wn=a(P),Qt=p(P,"P",{"data-svelte-h":!0}),k(Qt)!=="svelte-18n0pjl"&&(Qt.innerHTML=Rs),Cn=a(P),Gt=p(P,"P",{"data-svelte-h":!0}),k(Gt)!=="svelte-q52n56"&&(Gt.innerHTML=Js),vn=a(P),Ht=p(P,"P",{"data-svelte-h":!0}),k(Ht)!=="svelte-hswkmf"&&(Ht.innerHTML=Us),Bn=a(P),Y=p(P,"DIV",{class:!0});var re=C(Y);g(dt.$$.fragment,re),zn=a(re),At=p(re,"P",{"data-svelte-h":!0}),k(At)!=="svelte-h9apt9"&&(At.innerHTML=js),$n=a(re),g(ze.$$.fragment,re),Rn=a(re),g($e.$$.fragment,re),re.forEach(d),P.forEach(d),jo=a(e),g(ct.$$.fragment,e),Fo=a(e),x=p(e,"DIV",{class:!0});var Q=C(x);g(pt.$$.fragment,Q),Jn=a(Q),Ot=p(Q,"P",{"data-svelte-h":!0}),k(Ot)!=="svelte-plnhal"&&(Ot.textContent=Fs),Un=a(Q),Et=p(Q,"P",{"data-svelte-h":!0}),k(Et)!=="svelte-q52n56"&&(Et.innerHTML=xs),jn=a(Q),Yt=p(Q,"P",{"data-svelte-h":!0}),k(Yt)!=="svelte-hswkmf"&&(Yt.innerHTML=Is),Fn=a(Q),q=p(Q,"DIV",{class:!0});var G=C(q);g(mt.$$.fragment,G),xn=a(G),Dt=p(G,"P",{"data-svelte-h":!0}),k(Dt)!=="svelte-5tqlt3"&&(Dt.innerHTML=Ws),In=a(G),g(Re.$$.fragment,G),Wn=a(G),g(Je.$$.fragment,G),Vn=a(G),g(Ue.$$.fragment,G),G.forEach(d),Q.forEach(d),xo=a(e),g(ht.$$.fragment,e),Io=a(e),I=p(e,"DIV",{class:!0});var H=C(I);g(ut.$$.fragment,H),Nn=a(H),Kt=p(H,"P",{"data-svelte-h":!0}),k(Kt)!=="svelte-v0br3s"&&(Kt.textContent=Vs),qn=a(H),eo=p(H,"P",{"data-svelte-h":!0}),k(eo)!=="svelte-q52n56"&&(eo.innerHTML=Ns),Zn=a(H),to=p(H,"P",{"data-svelte-h":!0}),k(to)!=="svelte-hswkmf"&&(to.innerHTML=qs),Sn=a(H),D=p(H,"DIV",{class:!0});var ae=C(D);g(ft.$$.fragment,ae),Ln=a(ae),oo=p(ae,"P",{"data-svelte-h":!0}),k(oo)!=="svelte-x6ajlf"&&(oo.innerHTML=Zs),Xn=a(ae),g(je.$$.fragment,ae),Pn=a(ae),g(Fe.$$.fragment,ae),ae.forEach(d),H.forEach(d),Wo=a(e),g(gt.$$.fragment,e),Vo=a(e),W=p(e,"DIV",{class:!0});var A=C(W);g(_t.$$.fragment,A),Qn=a(A),no=p(A,"P",{"data-svelte-h":!0}),k(no)!=="svelte-1sr84od"&&(no.textContent=Ss),Gn=a(A),so=p(A,"P",{"data-svelte-h":!0}),k(so)!=="svelte-q52n56"&&(so.innerHTML=Ls),Hn=a(A),ro=p(A,"P",{"data-svelte-h":!0}),k(ro)!=="svelte-hswkmf"&&(ro.innerHTML=Xs),An=a(A),K=p(A,"DIV",{class:!0});var Ne=C(K);g(bt.$$.fragment,Ne),On=a(Ne),ao=p(Ne,"P",{"data-svelte-h":!0}),k(ao)!=="svelte-1iv7wa1"&&(ao.innerHTML=Ps),En=a(Ne),g(xe.$$.fragment,Ne),Yn=a(Ne),g(Ie.$$.fragment,Ne),Ne.forEach(d),A.forEach(d),No=a(e),g(Tt.$$.fragment,e),qo=a(e),V=p(e,"DIV",{class:!0});var me=C(V);g(yt.$$.fragment,me),Dn=a(me),io=p(me,"P",{"data-svelte-h":!0}),k(io)!=="svelte-im2b58"&&(io.innerHTML=Qs),Kn=a(me),lo=p(me,"P",{"data-svelte-h":!0}),k(lo)!=="svelte-q52n56"&&(lo.innerHTML=Gs),es=a(me),co=p(me,"P",{"data-svelte-h":!0}),k(co)!=="svelte-hswkmf"&&(co.innerHTML=Hs),ts=a(me),ee=p(me,"DIV",{class:!0});var qe=C(ee);g(kt.$$.fragment,qe),os=a(qe),po=p(qe,"P",{"data-svelte-h":!0}),k(po)!=="svelte-tcjfkx"&&(po.innerHTML=As),ns=a(qe),g(We.$$.fragment,qe),ss=a(qe),g(Ve.$$.fragment,qe),qe.forEach(d),me.forEach(d),Zo=a(e),g(Mt.$$.fragment,e),So=a(e),ho=p(e,"P",{}),C(ho).forEach(d),this.h()},h(){v(t,"name","hf:doc:metadata"),v(t,"content",Br),er(z,"float","right"),v(Z,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(de,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(ke,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(ce,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(Ut,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(J,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(pe,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(R,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(O,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(U,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(E,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(j,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(Y,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(F,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(q,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(x,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(D,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(I,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(K,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(W,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(ee,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),v(V,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,l){s(document.head,t),h(e,m,l),h(e,o,l),h(e,i,l),h(e,M,l),h(e,u,l),h(e,z,l),h(e,Ze,l),_(le,e,l),h(e,fo,l),h(e,Se,l),h(e,go,l),h(e,Le,l),h(e,_o,l),_(be,e,l),h(e,bo,l),h(e,Xe,l),h(e,To,l),_(Te,e,l),h(e,yo,l),_(Pe,e,l),h(e,ko,l),h(e,Z,l),_(Qe,Z,null),s(Z,Xo),s(Z,Ct),s(Z,Po),s(Z,vt),s(Z,Qo),_(ye,Z,null),h(e,Mo,l),_(Ge,e,l),h(e,wo,l),h(e,J,l),_(He,J,null),s(J,Go),s(J,de),_(Ae,de,null),s(de,Ho),s(de,Bt),s(de,Ao),s(de,zt),s(J,Oo),s(J,ke),_(Oe,ke,null),s(ke,Eo),s(ke,$t),s(J,Yo),s(J,ce),_(Ee,ce,null),s(ce,Do),s(ce,Rt),s(ce,Ko),s(ce,Jt),s(J,en),s(J,Ut),_(Ye,Ut,null),h(e,Co,l),_(De,e,l),h(e,vo,l),h(e,R,l),_(Ke,R,null),s(R,tn),s(R,jt),s(R,on),s(R,Ft),s(R,nn),s(R,xt),s(R,sn),s(R,It),s(R,rn),s(R,pe),_(et,pe,null),s(pe,an),s(pe,Wt),s(pe,ln),_(Me,pe,null),h(e,Bo,l),_(tt,e,l),h(e,zo,l),h(e,U,l),_(ot,U,null),s(U,dn),s(U,Vt),s(U,cn),s(U,Nt),s(U,pn),s(U,qt),s(U,mn),s(U,O),_(nt,O,null),s(O,hn),s(O,Zt),s(O,un),_(we,O,null),s(O,fn),_(Ce,O,null),h(e,$o,l),_(st,e,l),h(e,Ro,l),h(e,j,l),_(rt,j,null),s(j,gn),s(j,St),s(j,_n),s(j,Lt),s(j,bn),s(j,Xt),s(j,Tn),s(j,E),_(at,E,null),s(E,yn),s(E,Pt),s(E,kn),_(ve,E,null),s(E,Mn),_(Be,E,null),h(e,Jo,l),_(it,e,l),h(e,Uo,l),h(e,F,l),_(lt,F,null),s(F,wn),s(F,Qt),s(F,Cn),s(F,Gt),s(F,vn),s(F,Ht),s(F,Bn),s(F,Y),_(dt,Y,null),s(Y,zn),s(Y,At),s(Y,$n),_(ze,Y,null),s(Y,Rn),_($e,Y,null),h(e,jo,l),_(ct,e,l),h(e,Fo,l),h(e,x,l),_(pt,x,null),s(x,Jn),s(x,Ot),s(x,Un),s(x,Et),s(x,jn),s(x,Yt),s(x,Fn),s(x,q),_(mt,q,null),s(q,xn),s(q,Dt),s(q,In),_(Re,q,null),s(q,Wn),_(Je,q,null),s(q,Vn),_(Ue,q,null),h(e,xo,l),_(ht,e,l),h(e,Io,l),h(e,I,l),_(ut,I,null),s(I,Nn),s(I,Kt),s(I,qn),s(I,eo),s(I,Zn),s(I,to),s(I,Sn),s(I,D),_(ft,D,null),s(D,Ln),s(D,oo),s(D,Xn),_(je,D,null),s(D,Pn),_(Fe,D,null),h(e,Wo,l),_(gt,e,l),h(e,Vo,l),h(e,W,l),_(_t,W,null),s(W,Qn),s(W,no),s(W,Gn),s(W,so),s(W,Hn),s(W,ro),s(W,An),s(W,K),_(bt,K,null),s(K,On),s(K,ao),s(K,En),_(xe,K,null),s(K,Yn),_(Ie,K,null),h(e,No,l),_(Tt,e,l),h(e,qo,l),h(e,V,l),_(yt,V,null),s(V,Dn),s(V,io),s(V,Kn),s(V,lo),s(V,es),s(V,co),s(V,ts),s(V,ee),_(kt,ee,null),s(ee,os),s(ee,po),s(ee,ns),_(We,ee,null),s(ee,ss),_(Ve,ee,null),h(e,Zo,l),_(Mt,e,l),h(e,So,l),h(e,ho,l),Lo=!0},p(e,[l]){const oe={};l&2&&(oe.$$scope={dirty:l,ctx:e}),be.$set(oe);const S={};l&2&&(S.$$scope={dirty:l,ctx:e}),Te.$set(S);const he={};l&2&&(he.$$scope={dirty:l,ctx:e}),ye.$set(he);const wt={};l&2&&(wt.$$scope={dirty:l,ctx:e}),Me.$set(wt);const ue={};l&2&&(ue.$$scope={dirty:l,ctx:e}),we.$set(ue);const uo={};l&2&&(uo.$$scope={dirty:l,ctx:e}),Ce.$set(uo);const N={};l&2&&(N.$$scope={dirty:l,ctx:e}),ve.$set(N);const fe={};l&2&&(fe.$$scope={dirty:l,ctx:e}),Be.$set(fe);const L={};l&2&&(L.$$scope={dirty:l,ctx:e}),ze.$set(L);const ne={};l&2&&(ne.$$scope={dirty:l,ctx:e}),$e.$set(ne);const X={};l&2&&(X.$$scope={dirty:l,ctx:e}),Re.$set(X);const se={};l&2&&(se.$$scope={dirty:l,ctx:e}),Je.$set(se);const P={};l&2&&(P.$$scope={dirty:l,ctx:e}),Ue.$set(P);const re={};l&2&&(re.$$scope={dirty:l,ctx:e}),je.$set(re);const Q={};l&2&&(Q.$$scope={dirty:l,ctx:e}),Fe.$set(Q);const G={};l&2&&(G.$$scope={dirty:l,ctx:e}),xe.$set(G);const H={};l&2&&(H.$$scope={dirty:l,ctx:e}),Ie.$set(H);const ae={};l&2&&(ae.$$scope={dirty:l,ctx:e}),We.$set(ae);const A={};l&2&&(A.$$scope={dirty:l,ctx:e}),Ve.$set(A)},i(e){Lo||(b(le.$$.fragment,e),b(be.$$.fragment,e),b(Te.$$.fragment,e),b(Pe.$$.fragment,e),b(Qe.$$.fragment,e),b(ye.$$.fragment,e),b(Ge.$$.fragment,e),b(He.$$.fragment,e),b(Ae.$$.fragment,e),b(Oe.$$.fragment,e),b(Ee.$$.fragment,e),b(Ye.$$.fragment,e),b(De.$$.fragment,e),b(Ke.$$.fragment,e),b(et.$$.fragment,e),b(Me.$$.fragment,e),b(tt.$$.fragment,e),b(ot.$$.fragment,e),b(nt.$$.fragment,e),b(we.$$.fragment,e),b(Ce.$$.fragment,e),b(st.$$.fragment,e),b(rt.$$.fragment,e),b(at.$$.fragment,e),b(ve.$$.fragment,e),b(Be.$$.fragment,e),b(it.$$.fragment,e),b(lt.$$.fragment,e),b(dt.$$.fragment,e),b(ze.$$.fragment,e),b($e.$$.fragment,e),b(ct.$$.fragment,e),b(pt.$$.fragment,e),b(mt.$$.fragment,e),b(Re.$$.fragment,e),b(Je.$$.fragment,e),b(Ue.$$.fragment,e),b(ht.$$.fragment,e),b(ut.$$.fragment,e),b(ft.$$.fragment,e),b(je.$$.fragment,e),b(Fe.$$.fragment,e),b(gt.$$.fragment,e),b(_t.$$.fragment,e),b(bt.$$.fragment,e),b(xe.$$.fragment,e),b(Ie.$$.fragment,e),b(Tt.$$.fragment,e),b(yt.$$.fragment,e),b(kt.$$.fragment,e),b(We.$$.fragment,e),b(Ve.$$.fragment,e),b(Mt.$$.fragment,e),Lo=!0)},o(e){T(le.$$.fragment,e),T(be.$$.fragment,e),T(Te.$$.fragment,e),T(Pe.$$.fragment,e),T(Qe.$$.fragment,e),T(ye.$$.fragment,e),T(Ge.$$.fragment,e),T(He.$$.fragment,e),T(Ae.$$.fragment,e),T(Oe.$$.fragment,e),T(Ee.$$.fragment,e),T(Ye.$$.fragment,e),T(De.$$.fragment,e),T(Ke.$$.fragment,e),T(et.$$.fragment,e),T(Me.$$.fragment,e),T(tt.$$.fragment,e),T(ot.$$.fragment,e),T(nt.$$.fragment,e),T(we.$$.fragment,e),T(Ce.$$.fragment,e),T(st.$$.fragment,e),T(rt.$$.fragment,e),T(at.$$.fragment,e),T(ve.$$.fragment,e),T(Be.$$.fragment,e),T(it.$$.fragment,e),T(lt.$$.fragment,e),T(dt.$$.fragment,e),T(ze.$$.fragment,e),T($e.$$.fragment,e),T(ct.$$.fragment,e),T(pt.$$.fragment,e),T(mt.$$.fragment,e),T(Re.$$.fragment,e),T(Je.$$.fragment,e),T(Ue.$$.fragment,e),T(ht.$$.fragment,e),T(ut.$$.fragment,e),T(ft.$$.fragment,e),T(je.$$.fragment,e),T(Fe.$$.fragment,e),T(gt.$$.fragment,e),T(_t.$$.fragment,e),T(bt.$$.fragment,e),T(xe.$$.fragment,e),T(Ie.$$.fragment,e),T(Tt.$$.fragment,e),T(yt.$$.fragment,e),T(kt.$$.fragment,e),T(We.$$.fragment,e),T(Ve.$$.fragment,e),T(Mt.$$.fragment,e),Lo=!1},d(e){e&&(d(m),d(o),d(i),d(M),d(u),d(z),d(Ze),d(fo),d(Se),d(go),d(Le),d(_o),d(bo),d(Xe),d(To),d(yo),d(ko),d(Z),d(Mo),d(wo),d(J),d(Co),d(vo),d(R),d(Bo),d(zo),d(U),d($o),d(Ro),d(j),d(Jo),d(Uo),d(F),d(jo),d(Fo),d(x),d(xo),d(Io),d(I),d(Wo),d(Vo),d(W),d(No),d(qo),d(V),d(Zo),d(So),d(ho)),d(t),y(le,e),y(be,e),y(Te,e),y(Pe,e),y(Qe),y(ye),y(Ge,e),y(He),y(Ae),y(Oe),y(Ee),y(Ye),y(De,e),y(Ke),y(et),y(Me),y(tt,e),y(ot),y(nt),y(we),y(Ce),y(st,e),y(rt),y(at),y(ve),y(Be),y(it,e),y(lt),y(dt),y(ze),y($e),y(ct,e),y(pt),y(mt),y(Re),y(Je),y(Ue),y(ht,e),y(ut),y(ft),y(je),y(Fe),y(gt,e),y(_t),y(bt),y(xe),y(Ie),y(Tt,e),y(yt),y(kt),y(We),y(Ve),y(Mt,e)}}}const Br='{"title":"RoCBert","local":"rocbert","sections":[{"title":"RoCBertConfig","local":"transformers.RoCBertConfig","sections":[],"depth":2},{"title":"RoCBertTokenizer","local":"transformers.RoCBertTokenizer","sections":[],"depth":2},{"title":"RoCBertModel","local":"transformers.RoCBertModel","sections":[],"depth":2},{"title":"RoCBertForPreTraining","local":"transformers.RoCBertForPreTraining","sections":[],"depth":2},{"title":"RoCBertForCausalLM","local":"transformers.RoCBertForCausalLM","sections":[],"depth":2},{"title":"RoCBertForMaskedLM","local":"transformers.RoCBertForMaskedLM","sections":[],"depth":2},{"title":"RoCBertForSequenceClassification","local":"transformers.RoCBertForSequenceClassification","sections":[],"depth":2},{"title":"RoCBertForMultipleChoice","local":"transformers.RoCBertForMultipleChoice","sections":[],"depth":2},{"title":"RoCBertForTokenClassification","local":"transformers.RoCBertForTokenClassification","sections":[],"depth":2},{"title":"RoCBertForQuestionAnswering","local":"transformers.RoCBertForQuestionAnswering","sections":[],"depth":2}],"depth":1}';function zr(w){return Es(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class Wr extends Ys{constructor(t){super(),Ds(this,t,zr,vr,Os,{})}}export{Wr as component};
