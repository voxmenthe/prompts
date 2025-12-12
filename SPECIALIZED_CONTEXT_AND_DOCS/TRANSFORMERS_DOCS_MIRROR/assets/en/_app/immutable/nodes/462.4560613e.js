import{s as or,o as nr,n as L}from"../chunks/scheduler.18a86fab.js";import{S as sr,i as ar,g as c,s as a,r as f,A as rr,h as p,f as s,c as r,j as v,x as h,u as g,k as J,l as ir,y as l,a as d,v as b,d as _,t as M,w as k}from"../chunks/index.98837b22.js";import{T as he}from"../chunks/Tip.77304350.js";import{D as $}from"../chunks/Docstring.a1ef7999.js";import{C as ee}from"../chunks/CodeBlock.8d0c2e8a.js";import{E as We}from"../chunks/ExampleCodeBlock.8c3ee1f9.js";import{P as Zo}from"../chunks/PipelineTag.7749150e.js";import{H as Y,E as lr}from"../chunks/getInferenceSnippets.06c2775f.js";import{H as dr,a as Os}from"../chunks/HfOption.6641485e.js";function cr(w){let t,u="Click on the XLM-RoBERTa models in the right sidebar for more examples of how to apply XLM-RoBERTa to different cross-lingual tasks like classification, translation, and question answering.";return{c(){t=c("p"),t.textContent=u},l(n){t=p(n,"P",{"data-svelte-h":!0}),h(t)!=="svelte-3046aj"&&(t.textContent=u)},m(n,m){d(n,t,m)},p:L,d(n){n&&s(t)}}}function pr(w){let t,u;return t=new ee({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwcGlwZWxpbmUlMEElMEFwaXBlbGluZSUyMCUzRCUyMHBpcGVsaW5lKCUwQSUyMCUyMCUyMCUyMHRhc2slM0QlMjJmaWxsLW1hc2slMjIlMkMlMEElMjAlMjAlMjAlMjBtb2RlbCUzRCUyMkZhY2Vib29rQUklMkZ4bG0tcm9iZXJ0YS1iYXNlJTIyJTJDJTBBJTIwJTIwJTIwJTIwZHR5cGUlM0R0b3JjaC5mbG9hdDE2JTJDJTBBJTIwJTIwJTIwJTIwZGV2aWNlJTNEMCUwQSklMEElMjMlMjBFeGFtcGxlJTIwaW4lMjBGcmVuY2glMEFwaXBlbGluZSglMjJCb25qb3VyJTJDJTIwamUlMjBzdWlzJTIwdW4lMjBtb2QlQzMlQThsZSUyMCUzQ21hc2slM0UuJTIyKQ==",highlighted:`<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> pipeline

pipeline = pipeline(
    task=<span class="hljs-string">&quot;fill-mask&quot;</span>,
    model=<span class="hljs-string">&quot;FacebookAI/xlm-roberta-base&quot;</span>,
    dtype=torch.float16,
    device=<span class="hljs-number">0</span>
)
<span class="hljs-comment"># Example in French</span>
pipeline(<span class="hljs-string">&quot;Bonjour, je suis un modèle &lt;mask&gt;.&quot;</span>)`,wrap:!1}}),{c(){f(t.$$.fragment)},l(n){g(t.$$.fragment,n)},m(n,m){b(t,n,m),u=!0},p:L,i(n){u||(_(t.$$.fragment,n),u=!0)},o(n){M(t.$$.fragment,n),u=!1},d(n){k(t,n)}}}function mr(w){let t,u;return t=new ee({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Nb2RlbEZvck1hc2tlZExNJTJDJTIwQXV0b1Rva2VuaXplciUwQWltcG9ydCUyMHRvcmNoJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTBBJTIwJTIwJTIwJTIwJTIyRmFjZWJvb2tBSSUyRnhsbS1yb2JlcnRhLWJhc2UlMjIlMEEpJTBBbW9kZWwlMjAlM0QlMjBBdXRvTW9kZWxGb3JNYXNrZWRMTS5mcm9tX3ByZXRyYWluZWQoJTBBJTIwJTIwJTIwJTIwJTIyRmFjZWJvb2tBSSUyRnhsbS1yb2JlcnRhLWJhc2UlMjIlMkMlMEElMjAlMjAlMjAlMjBkdHlwZSUzRHRvcmNoLmZsb2F0MTYlMkMlMEElMjAlMjAlMjAlMjBkZXZpY2VfbWFwJTNEJTIyYXV0byUyMiUyQyUwQSUyMCUyMCUyMCUyMGF0dG5faW1wbGVtZW50YXRpb24lM0QlMjJzZHBhJTIyJTBBKSUwQSUwQSUyMyUyMFByZXBhcmUlMjBpbnB1dCUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplciglMjJCb25qb3VyJTJDJTIwamUlMjBzdWlzJTIwdW4lMjBtb2QlQzMlQThsZSUyMCUzQ21hc2slM0UuJTIyJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMikudG8obW9kZWwuZGV2aWNlKSUwQSUwQXdpdGglMjB0b3JjaC5ub19ncmFkKCklM0ElMEElMjAlMjAlMjAlMjBvdXRwdXRzJTIwJTNEJTIwbW9kZWwoKippbnB1dHMpJTBBJTIwJTIwJTIwJTIwcHJlZGljdGlvbnMlMjAlM0QlMjBvdXRwdXRzLmxvZ2l0cyUwQSUwQW1hc2tlZF9pbmRleCUyMCUzRCUyMHRvcmNoLndoZXJlKGlucHV0cyU1QidpbnB1dF9pZHMnJTVEJTIwJTNEJTNEJTIwdG9rZW5pemVyLm1hc2tfdG9rZW5faWQpJTVCMSU1RCUwQXByZWRpY3RlZF90b2tlbl9pZCUyMCUzRCUyMHByZWRpY3Rpb25zJTVCMCUyQyUyMG1hc2tlZF9pbmRleCU1RC5hcmdtYXgoZGltJTNELTEpJTBBcHJlZGljdGVkX3Rva2VuJTIwJTNEJTIwdG9rZW5pemVyLmRlY29kZShwcmVkaWN0ZWRfdG9rZW5faWQpJTBBJTBBcHJpbnQoZiUyMlRoZSUyMHByZWRpY3RlZCUyMHRva2VuJTIwaXMlM0ElMjAlN0JwcmVkaWN0ZWRfdG9rZW4lN0QlMjIp",highlighted:`<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoModelForMaskedLM, AutoTokenizer
<span class="hljs-keyword">import</span> torch

tokenizer = AutoTokenizer.from_pretrained(
    <span class="hljs-string">&quot;FacebookAI/xlm-roberta-base&quot;</span>
)
model = AutoModelForMaskedLM.from_pretrained(
    <span class="hljs-string">&quot;FacebookAI/xlm-roberta-base&quot;</span>,
    dtype=torch.float16,
    device_map=<span class="hljs-string">&quot;auto&quot;</span>,
    attn_implementation=<span class="hljs-string">&quot;sdpa&quot;</span>
)

<span class="hljs-comment"># Prepare input</span>
inputs = tokenizer(<span class="hljs-string">&quot;Bonjour, je suis un modèle &lt;mask&gt;.&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>).to(model.device)

<span class="hljs-keyword">with</span> torch.no_grad():
    outputs = model(**inputs)
    predictions = outputs.logits

masked_index = torch.where(inputs[<span class="hljs-string">&#x27;input_ids&#x27;</span>] == tokenizer.mask_token_id)[<span class="hljs-number">1</span>]
predicted_token_id = predictions[<span class="hljs-number">0</span>, masked_index].argmax(dim=-<span class="hljs-number">1</span>)
predicted_token = tokenizer.decode(predicted_token_id)

<span class="hljs-built_in">print</span>(<span class="hljs-string">f&quot;The predicted token is: <span class="hljs-subst">{predicted_token}</span>&quot;</span>)`,wrap:!1}}),{c(){f(t.$$.fragment)},l(n){g(t.$$.fragment,n)},m(n,m){b(t,n,m),u=!0},p:L,i(n){u||(_(t.$$.fragment,n),u=!0)},o(n){M(t.$$.fragment,n),u=!1},d(n){k(t,n)}}}function hr(w){let t,u;return t=new ee({props:{code:"ZWNobyUyMC1lJTIwJTIyUGxhbnRzJTIwY3JlYXRlJTIwJTNDbWFzayUzRSUyMHRocm91Z2glMjBhJTIwcHJvY2VzcyUyMGtub3duJTIwYXMlMjBwaG90b3N5bnRoZXNpcy4lMjIlMjAlN0MlMjB0cmFuc2Zvcm1lcnMtY2xpJTIwcnVuJTIwLS10YXNrJTIwZmlsbC1tYXNrJTIwLS1tb2RlbCUyMEZhY2Vib29rQUklMkZ4bG0tcm9iZXJ0YS1iYXNlJTIwLS1kZXZpY2UlMjAw",highlighted:'<span class="hljs-built_in">echo</span> -e <span class="hljs-string">&quot;Plants create &lt;mask&gt; through a process known as photosynthesis.&quot;</span> | transformers-cli run --task fill-mask --model FacebookAI/xlm-roberta-base --device 0',wrap:!1}}),{c(){f(t.$$.fragment)},l(n){g(t.$$.fragment,n)},m(n,m){b(t,n,m),u=!0},p:L,i(n){u||(_(t.$$.fragment,n),u=!0)},o(n){M(t.$$.fragment,n),u=!1},d(n){k(t,n)}}}function ur(w){let t,u,n,m,y,i;return t=new Os({props:{id:"usage",option:"Pipeline",$$slots:{default:[pr]},$$scope:{ctx:w}}}),n=new Os({props:{id:"usage",option:"AutoModel",$$slots:{default:[mr]},$$scope:{ctx:w}}}),y=new Os({props:{id:"usage",option:"transformers CLI",$$slots:{default:[hr]},$$scope:{ctx:w}}}),{c(){f(t.$$.fragment),u=a(),f(n.$$.fragment),m=a(),f(y.$$.fragment)},l(T){g(t.$$.fragment,T),u=r(T),g(n.$$.fragment,T),m=r(T),g(y.$$.fragment,T)},m(T,R){b(t,T,R),d(T,u,R),b(n,T,R),d(T,m,R),b(y,T,R),i=!0},p(T,R){const Io={};R&2&&(Io.$$scope={dirty:R,ctx:T}),t.$set(Io);const Ze={};R&2&&(Ze.$$scope={dirty:R,ctx:T}),n.$set(Ze);const se={};R&2&&(se.$$scope={dirty:R,ctx:T}),y.$set(se)},i(T){i||(_(t.$$.fragment,T),_(n.$$.fragment,T),_(y.$$.fragment,T),i=!0)},o(T){M(t.$$.fragment,T),M(n.$$.fragment,T),M(y.$$.fragment,T),i=!1},d(T){T&&(s(u),s(m)),k(t,T),k(n,T),k(y,T)}}}function fr(w){let t,u='This implementation is the same as RoBERTa. Refer to the <a href="roberta">documentation of RoBERTa</a> for usage examples as well as the information relative to the inputs and outputs.';return{c(){t=c("p"),t.innerHTML=u},l(n){t=p(n,"P",{"data-svelte-h":!0}),h(t)!=="svelte-huua3g"&&(t.innerHTML=u)},m(n,m){d(n,t,m)},p:L,d(n){n&&s(t)}}}function gr(w){let t,u="Examples:",n,m,y;return m=new ee({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMFhMTVJvYmVydGFDb25maWclMkMlMjBYTE1Sb2JlcnRhTW9kZWwlMEElMEElMjMlMjBJbml0aWFsaXppbmclMjBhJTIwWExNLVJvQkVSVGElMjBGYWNlYm9va0FJJTJGeGxtLXJvYmVydGEtYmFzZSUyMHN0eWxlJTIwY29uZmlndXJhdGlvbiUwQWNvbmZpZ3VyYXRpb24lMjAlM0QlMjBYTE1Sb2JlcnRhQ29uZmlnKCklMEElMEElMjMlMjBJbml0aWFsaXppbmclMjBhJTIwbW9kZWwlMjAod2l0aCUyMHJhbmRvbSUyMHdlaWdodHMpJTIwZnJvbSUyMHRoZSUyMEZhY2Vib29rQUklMkZ4bG0tcm9iZXJ0YS1iYXNlJTIwc3R5bGUlMjBjb25maWd1cmF0aW9uJTBBbW9kZWwlMjAlM0QlMjBYTE1Sb2JlcnRhTW9kZWwoY29uZmlndXJhdGlvbiklMEElMEElMjMlMjBBY2Nlc3NpbmclMjB0aGUlMjBtb2RlbCUyMGNvbmZpZ3VyYXRpb24lMEFjb25maWd1cmF0aW9uJTIwJTNEJTIwbW9kZWwuY29uZmln",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> XLMRobertaConfig, XLMRobertaModel

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a XLM-RoBERTa FacebookAI/xlm-roberta-base style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = XLMRobertaConfig()

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a model (with random weights) from the FacebookAI/xlm-roberta-base style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = XLMRobertaModel(configuration)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Accessing the model configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = model.config`,wrap:!1}}),{c(){t=c("p"),t.textContent=u,n=a(),f(m.$$.fragment)},l(i){t=p(i,"P",{"data-svelte-h":!0}),h(t)!=="svelte-kvfsh7"&&(t.textContent=u),n=r(i),g(m.$$.fragment,i)},m(i,T){d(i,t,T),d(i,n,T),b(m,i,T),y=!0},p:L,i(i){y||(_(m.$$.fragment,i),y=!0)},o(i){M(m.$$.fragment,i),y=!1},d(i){i&&(s(t),s(n)),k(m,i)}}}function br(w){let t,u=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=u},l(n){t=p(n,"P",{"data-svelte-h":!0}),h(t)!=="svelte-fincs2"&&(t.innerHTML=u)},m(n,m){d(n,t,m)},p:L,d(n){n&&s(t)}}}function _r(w){let t,u=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=u},l(n){t=p(n,"P",{"data-svelte-h":!0}),h(t)!=="svelte-fincs2"&&(t.innerHTML=u)},m(n,m){d(n,t,m)},p:L,d(n){n&&s(t)}}}function Mr(w){let t,u="Example:",n,m,y;return m=new ee({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBYTE1Sb2JlcnRhRm9yQ2F1c2FsTE0lMkMlMjBBdXRvQ29uZmlnJTBBaW1wb3J0JTIwdG9yY2glMEElMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJGYWNlYm9va0FJJTJGcm9iZXJ0YS1iYXNlJTIyKSUwQWNvbmZpZyUyMCUzRCUyMEF1dG9Db25maWcuZnJvbV9wcmV0cmFpbmVkKCUyMkZhY2Vib29rQUklMkZyb2JlcnRhLWJhc2UlMjIpJTBBY29uZmlnLmlzX2RlY29kZXIlMjAlM0QlMjBUcnVlJTBBbW9kZWwlMjAlM0QlMjBYTE1Sb2JlcnRhRm9yQ2F1c2FsTE0uZnJvbV9wcmV0cmFpbmVkKCUyMkZhY2Vib29rQUklMkZyb2JlcnRhLWJhc2UlMjIlMkMlMjBjb25maWclM0Rjb25maWcpJTBBJTBBaW5wdXRzJTIwJTNEJTIwdG9rZW5pemVyKCUyMkhlbGxvJTJDJTIwbXklMjBkb2clMjBpcyUyMGN1dGUlMjIlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQW91dHB1dHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyklMEElMEFwcmVkaWN0aW9uX2xvZ2l0cyUyMCUzRCUyMG91dHB1dHMubG9naXRz",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, XLMRobertaForCausalLM, AutoConfig
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;FacebookAI/roberta-base&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>config = AutoConfig.from_pretrained(<span class="hljs-string">&quot;FacebookAI/roberta-base&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>config.is_decoder = <span class="hljs-literal">True</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = XLMRobertaForCausalLM.from_pretrained(<span class="hljs-string">&quot;FacebookAI/roberta-base&quot;</span>, config=config)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs)

<span class="hljs-meta">&gt;&gt;&gt; </span>prediction_logits = outputs.logits`,wrap:!1}}),{c(){t=c("p"),t.textContent=u,n=a(),f(m.$$.fragment)},l(i){t=p(i,"P",{"data-svelte-h":!0}),h(t)!=="svelte-11lpom8"&&(t.textContent=u),n=r(i),g(m.$$.fragment,i)},m(i,T){d(i,t,T),d(i,n,T),b(m,i,T),y=!0},p:L,i(i){y||(_(m.$$.fragment,i),y=!0)},o(i){M(m.$$.fragment,i),y=!1},d(i){i&&(s(t),s(n)),k(m,i)}}}function kr(w){let t,u=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=u},l(n){t=p(n,"P",{"data-svelte-h":!0}),h(t)!=="svelte-fincs2"&&(t.innerHTML=u)},m(n,m){d(n,t,m)},p:L,d(n){n&&s(t)}}}function Tr(w){let t,u="Example:",n,m,y;return m=new ee({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBYTE1Sb2JlcnRhRm9yTWFza2VkTE0lMEFpbXBvcnQlMjB0b3JjaCUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMkZhY2Vib29rQUklMkZ4bG0tcm9iZXJ0YS1iYXNlJTIyKSUwQW1vZGVsJTIwJTNEJTIwWExNUm9iZXJ0YUZvck1hc2tlZExNLmZyb21fcHJldHJhaW5lZCglMjJGYWNlYm9va0FJJTJGeGxtLXJvYmVydGEtYmFzZSUyMiklMEElMEFpbnB1dHMlMjAlM0QlMjB0b2tlbml6ZXIoJTIyVGhlJTIwY2FwaXRhbCUyMG9mJTIwRnJhbmNlJTIwaXMlMjAlM0NtYXNrJTNFLiUyMiUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpJTBBJTBBd2l0aCUyMHRvcmNoLm5vX2dyYWQoKSUzQSUwQSUyMCUyMCUyMCUyMGxvZ2l0cyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzKS5sb2dpdHMlMEElMEElMjMlMjByZXRyaWV2ZSUyMGluZGV4JTIwb2YlMjAlM0NtYXNrJTNFJTBBbWFza190b2tlbl9pbmRleCUyMCUzRCUyMChpbnB1dHMuaW5wdXRfaWRzJTIwJTNEJTNEJTIwdG9rZW5pemVyLm1hc2tfdG9rZW5faWQpJTVCMCU1RC5ub256ZXJvKGFzX3R1cGxlJTNEVHJ1ZSklNUIwJTVEJTBBJTBBcHJlZGljdGVkX3Rva2VuX2lkJTIwJTNEJTIwbG9naXRzJTVCMCUyQyUyMG1hc2tfdG9rZW5faW5kZXglNUQuYXJnbWF4KGF4aXMlM0QtMSklMEF0b2tlbml6ZXIuZGVjb2RlKHByZWRpY3RlZF90b2tlbl9pZCklMEElMEFsYWJlbHMlMjAlM0QlMjB0b2tlbml6ZXIoJTIyVGhlJTIwY2FwaXRhbCUyMG9mJTIwRnJhbmNlJTIwaXMlMjBQYXJpcy4lMjIlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSU1QiUyMmlucHV0X2lkcyUyMiU1RCUwQSUyMyUyMG1hc2slMjBsYWJlbHMlMjBvZiUyMG5vbi0lM0NtYXNrJTNFJTIwdG9rZW5zJTBBbGFiZWxzJTIwJTNEJTIwdG9yY2gud2hlcmUoaW5wdXRzLmlucHV0X2lkcyUyMCUzRCUzRCUyMHRva2VuaXplci5tYXNrX3Rva2VuX2lkJTJDJTIwbGFiZWxzJTJDJTIwLTEwMCklMEElMEFvdXRwdXRzJTIwJTNEJTIwbW9kZWwoKippbnB1dHMlMkMlMjBsYWJlbHMlM0RsYWJlbHMpJTBBcm91bmQob3V0cHV0cy5sb3NzLml0ZW0oKSUyQyUyMDIp",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, XLMRobertaForMaskedLM
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;FacebookAI/xlm-roberta-base&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = XLMRobertaForMaskedLM.from_pretrained(<span class="hljs-string">&quot;FacebookAI/xlm-roberta-base&quot;</span>)

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
...`,wrap:!1}}),{c(){t=c("p"),t.textContent=u,n=a(),f(m.$$.fragment)},l(i){t=p(i,"P",{"data-svelte-h":!0}),h(t)!=="svelte-11lpom8"&&(t.textContent=u),n=r(i),g(m.$$.fragment,i)},m(i,T){d(i,t,T),d(i,n,T),b(m,i,T),y=!0},p:L,i(i){y||(_(m.$$.fragment,i),y=!0)},o(i){M(m.$$.fragment,i),y=!1},d(i){i&&(s(t),s(n)),k(m,i)}}}function yr(w){let t,u=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=u},l(n){t=p(n,"P",{"data-svelte-h":!0}),h(t)!=="svelte-fincs2"&&(t.innerHTML=u)},m(n,m){d(n,t,m)},p:L,d(n){n&&s(t)}}}function wr(w){let t,u="Example of single-label classification:",n,m,y;return m=new ee({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Rva2VuaXplciUyQyUyMFhMTVJvYmVydGFGb3JTZXF1ZW5jZUNsYXNzaWZpY2F0aW9uJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIyRmFjZWJvb2tBSSUyRnhsbS1yb2JlcnRhLWJhc2UlMjIpJTBBbW9kZWwlMjAlM0QlMjBYTE1Sb2JlcnRhRm9yU2VxdWVuY2VDbGFzc2lmaWNhdGlvbi5mcm9tX3ByZXRyYWluZWQoJTIyRmFjZWJvb2tBSSUyRnhsbS1yb2JlcnRhLWJhc2UlMjIpJTBBJTBBaW5wdXRzJTIwJTNEJTIwdG9rZW5pemVyKCUyMkhlbGxvJTJDJTIwbXklMjBkb2clMjBpcyUyMGN1dGUlMjIlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQSUwQXdpdGglMjB0b3JjaC5ub19ncmFkKCklM0ElMEElMjAlMjAlMjAlMjBsb2dpdHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cykubG9naXRzJTBBJTBBcHJlZGljdGVkX2NsYXNzX2lkJTIwJTNEJTIwbG9naXRzLmFyZ21heCgpLml0ZW0oKSUwQW1vZGVsLmNvbmZpZy5pZDJsYWJlbCU1QnByZWRpY3RlZF9jbGFzc19pZCU1RCUwQSUwQSUyMyUyMFRvJTIwdHJhaW4lMjBhJTIwbW9kZWwlMjBvbiUyMCU2MG51bV9sYWJlbHMlNjAlMjBjbGFzc2VzJTJDJTIweW91JTIwY2FuJTIwcGFzcyUyMCU2MG51bV9sYWJlbHMlM0RudW1fbGFiZWxzJTYwJTIwdG8lMjAlNjAuZnJvbV9wcmV0cmFpbmVkKC4uLiklNjAlMEFudW1fbGFiZWxzJTIwJTNEJTIwbGVuKG1vZGVsLmNvbmZpZy5pZDJsYWJlbCklMEFtb2RlbCUyMCUzRCUyMFhMTVJvYmVydGFGb3JTZXF1ZW5jZUNsYXNzaWZpY2F0aW9uLmZyb21fcHJldHJhaW5lZCglMjJGYWNlYm9va0FJJTJGeGxtLXJvYmVydGEtYmFzZSUyMiUyQyUyMG51bV9sYWJlbHMlM0RudW1fbGFiZWxzKSUwQSUwQWxhYmVscyUyMCUzRCUyMHRvcmNoLnRlbnNvciglNUIxJTVEKSUwQWxvc3MlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyUyQyUyMGxhYmVscyUzRGxhYmVscykubG9zcyUwQXJvdW5kKGxvc3MuaXRlbSgpJTJDJTIwMik=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, XLMRobertaForSequenceClassification

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;FacebookAI/xlm-roberta-base&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = XLMRobertaForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;FacebookAI/xlm-roberta-base&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    logits = model(**inputs).logits

<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_class_id = logits.argmax().item()
<span class="hljs-meta">&gt;&gt;&gt; </span>model.config.id2label[predicted_class_id]
...

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># To train a model on \`num_labels\` classes, you can pass \`num_labels=num_labels\` to \`.from_pretrained(...)\`</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>num_labels = <span class="hljs-built_in">len</span>(model.config.id2label)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = XLMRobertaForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;FacebookAI/xlm-roberta-base&quot;</span>, num_labels=num_labels)

<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.tensor([<span class="hljs-number">1</span>])
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = model(**inputs, labels=labels).loss
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">round</span>(loss.item(), <span class="hljs-number">2</span>)
...`,wrap:!1}}),{c(){t=c("p"),t.textContent=u,n=a(),f(m.$$.fragment)},l(i){t=p(i,"P",{"data-svelte-h":!0}),h(t)!=="svelte-ykxpe4"&&(t.textContent=u),n=r(i),g(m.$$.fragment,i)},m(i,T){d(i,t,T),d(i,n,T),b(m,i,T),y=!0},p:L,i(i){y||(_(m.$$.fragment,i),y=!0)},o(i){M(m.$$.fragment,i),y=!1},d(i){i&&(s(t),s(n)),k(m,i)}}}function vr(w){let t,u="Example of multi-label classification:",n,m,y;return m=new ee({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Rva2VuaXplciUyQyUyMFhMTVJvYmVydGFGb3JTZXF1ZW5jZUNsYXNzaWZpY2F0aW9uJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIyRmFjZWJvb2tBSSUyRnhsbS1yb2JlcnRhLWJhc2UlMjIpJTBBbW9kZWwlMjAlM0QlMjBYTE1Sb2JlcnRhRm9yU2VxdWVuY2VDbGFzc2lmaWNhdGlvbi5mcm9tX3ByZXRyYWluZWQoJTIyRmFjZWJvb2tBSSUyRnhsbS1yb2JlcnRhLWJhc2UlMjIlMkMlMjBwcm9ibGVtX3R5cGUlM0QlMjJtdWx0aV9sYWJlbF9jbGFzc2lmaWNhdGlvbiUyMiklMEElMEFpbnB1dHMlMjAlM0QlMjB0b2tlbml6ZXIoJTIySGVsbG8lMkMlMjBteSUyMGRvZyUyMGlzJTIwY3V0ZSUyMiUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpJTBBJTBBd2l0aCUyMHRvcmNoLm5vX2dyYWQoKSUzQSUwQSUyMCUyMCUyMCUyMGxvZ2l0cyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzKS5sb2dpdHMlMEElMEFwcmVkaWN0ZWRfY2xhc3NfaWRzJTIwJTNEJTIwdG9yY2guYXJhbmdlKDAlMkMlMjBsb2dpdHMuc2hhcGUlNUItMSU1RCklNUJ0b3JjaC5zaWdtb2lkKGxvZ2l0cykuc3F1ZWV6ZShkaW0lM0QwKSUyMCUzRSUyMDAuNSU1RCUwQSUwQSUyMyUyMFRvJTIwdHJhaW4lMjBhJTIwbW9kZWwlMjBvbiUyMCU2MG51bV9sYWJlbHMlNjAlMjBjbGFzc2VzJTJDJTIweW91JTIwY2FuJTIwcGFzcyUyMCU2MG51bV9sYWJlbHMlM0RudW1fbGFiZWxzJTYwJTIwdG8lMjAlNjAuZnJvbV9wcmV0cmFpbmVkKC4uLiklNjAlMEFudW1fbGFiZWxzJTIwJTNEJTIwbGVuKG1vZGVsLmNvbmZpZy5pZDJsYWJlbCklMEFtb2RlbCUyMCUzRCUyMFhMTVJvYmVydGFGb3JTZXF1ZW5jZUNsYXNzaWZpY2F0aW9uLmZyb21fcHJldHJhaW5lZCglMEElMjAlMjAlMjAlMjAlMjJGYWNlYm9va0FJJTJGeGxtLXJvYmVydGEtYmFzZSUyMiUyQyUyMG51bV9sYWJlbHMlM0RudW1fbGFiZWxzJTJDJTIwcHJvYmxlbV90eXBlJTNEJTIybXVsdGlfbGFiZWxfY2xhc3NpZmljYXRpb24lMjIlMEEpJTBBJTBBbGFiZWxzJTIwJTNEJTIwdG9yY2guc3VtKCUwQSUyMCUyMCUyMCUyMHRvcmNoLm5uLmZ1bmN0aW9uYWwub25lX2hvdChwcmVkaWN0ZWRfY2xhc3NfaWRzJTVCTm9uZSUyQyUyMCUzQSU1RC5jbG9uZSgpJTJDJTIwbnVtX2NsYXNzZXMlM0RudW1fbGFiZWxzKSUyQyUyMGRpbSUzRDElMEEpLnRvKHRvcmNoLmZsb2F0KSUwQWxvc3MlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyUyQyUyMGxhYmVscyUzRGxhYmVscykubG9zcw==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, XLMRobertaForSequenceClassification

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;FacebookAI/xlm-roberta-base&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = XLMRobertaForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;FacebookAI/xlm-roberta-base&quot;</span>, problem_type=<span class="hljs-string">&quot;multi_label_classification&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    logits = model(**inputs).logits

<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_class_ids = torch.arange(<span class="hljs-number">0</span>, logits.shape[-<span class="hljs-number">1</span>])[torch.sigmoid(logits).squeeze(dim=<span class="hljs-number">0</span>) &gt; <span class="hljs-number">0.5</span>]

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># To train a model on \`num_labels\` classes, you can pass \`num_labels=num_labels\` to \`.from_pretrained(...)\`</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>num_labels = <span class="hljs-built_in">len</span>(model.config.id2label)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = XLMRobertaForSequenceClassification.from_pretrained(
<span class="hljs-meta">... </span>    <span class="hljs-string">&quot;FacebookAI/xlm-roberta-base&quot;</span>, num_labels=num_labels, problem_type=<span class="hljs-string">&quot;multi_label_classification&quot;</span>
<span class="hljs-meta">... </span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.<span class="hljs-built_in">sum</span>(
<span class="hljs-meta">... </span>    torch.nn.functional.one_hot(predicted_class_ids[<span class="hljs-literal">None</span>, :].clone(), num_classes=num_labels), dim=<span class="hljs-number">1</span>
<span class="hljs-meta">... </span>).to(torch.<span class="hljs-built_in">float</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = model(**inputs, labels=labels).loss`,wrap:!1}}),{c(){t=c("p"),t.textContent=u,n=a(),f(m.$$.fragment)},l(i){t=p(i,"P",{"data-svelte-h":!0}),h(t)!=="svelte-1l8e32d"&&(t.textContent=u),n=r(i),g(m.$$.fragment,i)},m(i,T){d(i,t,T),d(i,n,T),b(m,i,T),y=!0},p:L,i(i){y||(_(m.$$.fragment,i),y=!0)},o(i){M(m.$$.fragment,i),y=!1},d(i){i&&(s(t),s(n)),k(m,i)}}}function Jr(w){let t,u=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=u},l(n){t=p(n,"P",{"data-svelte-h":!0}),h(t)!=="svelte-fincs2"&&(t.innerHTML=u)},m(n,m){d(n,t,m)},p:L,d(n){n&&s(t)}}}function $r(w){let t,u="Example:",n,m,y;return m=new ee({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBYTE1Sb2JlcnRhRm9yTXVsdGlwbGVDaG9pY2UlMEFpbXBvcnQlMjB0b3JjaCUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMkZhY2Vib29rQUklMkZ4bG0tcm9iZXJ0YS1iYXNlJTIyKSUwQW1vZGVsJTIwJTNEJTIwWExNUm9iZXJ0YUZvck11bHRpcGxlQ2hvaWNlLmZyb21fcHJldHJhaW5lZCglMjJGYWNlYm9va0FJJTJGeGxtLXJvYmVydGEtYmFzZSUyMiklMEElMEFwcm9tcHQlMjAlM0QlMjAlMjJJbiUyMEl0YWx5JTJDJTIwcGl6emElMjBzZXJ2ZWQlMjBpbiUyMGZvcm1hbCUyMHNldHRpbmdzJTJDJTIwc3VjaCUyMGFzJTIwYXQlMjBhJTIwcmVzdGF1cmFudCUyQyUyMGlzJTIwcHJlc2VudGVkJTIwdW5zbGljZWQuJTIyJTBBY2hvaWNlMCUyMCUzRCUyMCUyMkl0JTIwaXMlMjBlYXRlbiUyMHdpdGglMjBhJTIwZm9yayUyMGFuZCUyMGElMjBrbmlmZS4lMjIlMEFjaG9pY2UxJTIwJTNEJTIwJTIySXQlMjBpcyUyMGVhdGVuJTIwd2hpbGUlMjBoZWxkJTIwaW4lMjB0aGUlMjBoYW5kLiUyMiUwQWxhYmVscyUyMCUzRCUyMHRvcmNoLnRlbnNvcigwKS51bnNxdWVlemUoMCklMjAlMjAlMjMlMjBjaG9pY2UwJTIwaXMlMjBjb3JyZWN0JTIwKGFjY29yZGluZyUyMHRvJTIwV2lraXBlZGlhJTIwJTNCKSklMkMlMjBiYXRjaCUyMHNpemUlMjAxJTBBJTBBZW5jb2RpbmclMjAlM0QlMjB0b2tlbml6ZXIoJTVCcHJvbXB0JTJDJTIwcHJvbXB0JTVEJTJDJTIwJTVCY2hvaWNlMCUyQyUyMGNob2ljZTElNUQlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyJTJDJTIwcGFkZGluZyUzRFRydWUpJTBBb3V0cHV0cyUyMCUzRCUyMG1vZGVsKCoqJTdCayUzQSUyMHYudW5zcXVlZXplKDApJTIwZm9yJTIwayUyQyUyMHYlMjBpbiUyMGVuY29kaW5nLml0ZW1zKCklN0QlMkMlMjBsYWJlbHMlM0RsYWJlbHMpJTIwJTIwJTIzJTIwYmF0Y2glMjBzaXplJTIwaXMlMjAxJTBBJTBBJTIzJTIwdGhlJTIwbGluZWFyJTIwY2xhc3NpZmllciUyMHN0aWxsJTIwbmVlZHMlMjB0byUyMGJlJTIwdHJhaW5lZCUwQWxvc3MlMjAlM0QlMjBvdXRwdXRzLmxvc3MlMEFsb2dpdHMlMjAlM0QlMjBvdXRwdXRzLmxvZ2l0cw==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, XLMRobertaForMultipleChoice
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;FacebookAI/xlm-roberta-base&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = XLMRobertaForMultipleChoice.from_pretrained(<span class="hljs-string">&quot;FacebookAI/xlm-roberta-base&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>prompt = <span class="hljs-string">&quot;In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>choice0 = <span class="hljs-string">&quot;It is eaten with a fork and a knife.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>choice1 = <span class="hljs-string">&quot;It is eaten while held in the hand.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.tensor(<span class="hljs-number">0</span>).unsqueeze(<span class="hljs-number">0</span>)  <span class="hljs-comment"># choice0 is correct (according to Wikipedia ;)), batch size 1</span>

<span class="hljs-meta">&gt;&gt;&gt; </span>encoding = tokenizer([prompt, prompt], [choice0, choice1], return_tensors=<span class="hljs-string">&quot;pt&quot;</span>, padding=<span class="hljs-literal">True</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**{k: v.unsqueeze(<span class="hljs-number">0</span>) <span class="hljs-keyword">for</span> k, v <span class="hljs-keyword">in</span> encoding.items()}, labels=labels)  <span class="hljs-comment"># batch size is 1</span>

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># the linear classifier still needs to be trained</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = outputs.loss
<span class="hljs-meta">&gt;&gt;&gt; </span>logits = outputs.logits`,wrap:!1}}),{c(){t=c("p"),t.textContent=u,n=a(),f(m.$$.fragment)},l(i){t=p(i,"P",{"data-svelte-h":!0}),h(t)!=="svelte-11lpom8"&&(t.textContent=u),n=r(i),g(m.$$.fragment,i)},m(i,T){d(i,t,T),d(i,n,T),b(m,i,T),y=!0},p:L,i(i){y||(_(m.$$.fragment,i),y=!0)},o(i){M(m.$$.fragment,i),y=!1},d(i){i&&(s(t),s(n)),k(m,i)}}}function Rr(w){let t,u=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=u},l(n){t=p(n,"P",{"data-svelte-h":!0}),h(t)!=="svelte-fincs2"&&(t.innerHTML=u)},m(n,m){d(n,t,m)},p:L,d(n){n&&s(t)}}}function Lr(w){let t,u="Example:",n,m,y;return m=new ee({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBYTE1Sb2JlcnRhRm9yVG9rZW5DbGFzc2lmaWNhdGlvbiUwQWltcG9ydCUyMHRvcmNoJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIyRmFjZWJvb2tBSSUyRnhsbS1yb2JlcnRhLWJhc2UlMjIpJTBBbW9kZWwlMjAlM0QlMjBYTE1Sb2JlcnRhRm9yVG9rZW5DbGFzc2lmaWNhdGlvbi5mcm9tX3ByZXRyYWluZWQoJTIyRmFjZWJvb2tBSSUyRnhsbS1yb2JlcnRhLWJhc2UlMjIpJTBBJTBBaW5wdXRzJTIwJTNEJTIwdG9rZW5pemVyKCUwQSUyMCUyMCUyMCUyMCUyMkh1Z2dpbmdGYWNlJTIwaXMlMjBhJTIwY29tcGFueSUyMGJhc2VkJTIwaW4lMjBQYXJpcyUyMGFuZCUyME5ldyUyMFlvcmslMjIlMkMlMjBhZGRfc3BlY2lhbF90b2tlbnMlM0RGYWxzZSUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIlMEEpJTBBJTBBd2l0aCUyMHRvcmNoLm5vX2dyYWQoKSUzQSUwQSUyMCUyMCUyMCUyMGxvZ2l0cyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzKS5sb2dpdHMlMEElMEFwcmVkaWN0ZWRfdG9rZW5fY2xhc3NfaWRzJTIwJTNEJTIwbG9naXRzLmFyZ21heCgtMSklMEElMEElMjMlMjBOb3RlJTIwdGhhdCUyMHRva2VucyUyMGFyZSUyMGNsYXNzaWZpZWQlMjByYXRoZXIlMjB0aGVuJTIwaW5wdXQlMjB3b3JkcyUyMHdoaWNoJTIwbWVhbnMlMjB0aGF0JTBBJTIzJTIwdGhlcmUlMjBtaWdodCUyMGJlJTIwbW9yZSUyMHByZWRpY3RlZCUyMHRva2VuJTIwY2xhc3NlcyUyMHRoYW4lMjB3b3Jkcy4lMEElMjMlMjBNdWx0aXBsZSUyMHRva2VuJTIwY2xhc3NlcyUyMG1pZ2h0JTIwYWNjb3VudCUyMGZvciUyMHRoZSUyMHNhbWUlMjB3b3JkJTBBcHJlZGljdGVkX3Rva2Vuc19jbGFzc2VzJTIwJTNEJTIwJTVCbW9kZWwuY29uZmlnLmlkMmxhYmVsJTVCdC5pdGVtKCklNUQlMjBmb3IlMjB0JTIwaW4lMjBwcmVkaWN0ZWRfdG9rZW5fY2xhc3NfaWRzJTVCMCU1RCU1RCUwQXByZWRpY3RlZF90b2tlbnNfY2xhc3NlcyUwQSUwQWxhYmVscyUyMCUzRCUyMHByZWRpY3RlZF90b2tlbl9jbGFzc19pZHMlMEFsb3NzJTIwJTNEJTIwbW9kZWwoKippbnB1dHMlMkMlMjBsYWJlbHMlM0RsYWJlbHMpLmxvc3MlMEFyb3VuZChsb3NzLml0ZW0oKSUyQyUyMDIp",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, XLMRobertaForTokenClassification
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;FacebookAI/xlm-roberta-base&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = XLMRobertaForTokenClassification.from_pretrained(<span class="hljs-string">&quot;FacebookAI/xlm-roberta-base&quot;</span>)

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
...`,wrap:!1}}),{c(){t=c("p"),t.textContent=u,n=a(),f(m.$$.fragment)},l(i){t=p(i,"P",{"data-svelte-h":!0}),h(t)!=="svelte-11lpom8"&&(t.textContent=u),n=r(i),g(m.$$.fragment,i)},m(i,T){d(i,t,T),d(i,n,T),b(m,i,T),y=!0},p:L,i(i){y||(_(m.$$.fragment,i),y=!0)},o(i){M(m.$$.fragment,i),y=!1},d(i){i&&(s(t),s(n)),k(m,i)}}}function xr(w){let t,u=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=u},l(n){t=p(n,"P",{"data-svelte-h":!0}),h(t)!=="svelte-fincs2"&&(t.innerHTML=u)},m(n,m){d(n,t,m)},p:L,d(n){n&&s(t)}}}function Xr(w){let t,u="Example:",n,m,y;return m=new ee({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBYTE1Sb2JlcnRhRm9yUXVlc3Rpb25BbnN3ZXJpbmclMEFpbXBvcnQlMjB0b3JjaCUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMkZhY2Vib29rQUklMkZ4bG0tcm9iZXJ0YS1iYXNlJTIyKSUwQW1vZGVsJTIwJTNEJTIwWExNUm9iZXJ0YUZvclF1ZXN0aW9uQW5zd2VyaW5nLmZyb21fcHJldHJhaW5lZCglMjJGYWNlYm9va0FJJTJGeGxtLXJvYmVydGEtYmFzZSUyMiklMEElMEFxdWVzdGlvbiUyQyUyMHRleHQlMjAlM0QlMjAlMjJXaG8lMjB3YXMlMjBKaW0lMjBIZW5zb24lM0YlMjIlMkMlMjAlMjJKaW0lMjBIZW5zb24lMjB3YXMlMjBhJTIwbmljZSUyMHB1cHBldCUyMiUwQSUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplcihxdWVzdGlvbiUyQyUyMHRleHQlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQXdpdGglMjB0b3JjaC5ub19ncmFkKCklM0ElMEElMjAlMjAlMjAlMjBvdXRwdXRzJTIwJTNEJTIwbW9kZWwoKippbnB1dHMpJTBBJTBBYW5zd2VyX3N0YXJ0X2luZGV4JTIwJTNEJTIwb3V0cHV0cy5zdGFydF9sb2dpdHMuYXJnbWF4KCklMEFhbnN3ZXJfZW5kX2luZGV4JTIwJTNEJTIwb3V0cHV0cy5lbmRfbG9naXRzLmFyZ21heCgpJTBBJTBBcHJlZGljdF9hbnN3ZXJfdG9rZW5zJTIwJTNEJTIwaW5wdXRzLmlucHV0X2lkcyU1QjAlMkMlMjBhbnN3ZXJfc3RhcnRfaW5kZXglMjAlM0ElMjBhbnN3ZXJfZW5kX2luZGV4JTIwJTJCJTIwMSU1RCUwQXRva2VuaXplci5kZWNvZGUocHJlZGljdF9hbnN3ZXJfdG9rZW5zJTJDJTIwc2tpcF9zcGVjaWFsX3Rva2VucyUzRFRydWUpJTBBJTBBJTIzJTIwdGFyZ2V0JTIwaXMlMjAlMjJuaWNlJTIwcHVwcGV0JTIyJTBBdGFyZ2V0X3N0YXJ0X2luZGV4JTIwJTNEJTIwdG9yY2gudGVuc29yKCU1QjE0JTVEKSUwQXRhcmdldF9lbmRfaW5kZXglMjAlM0QlMjB0b3JjaC50ZW5zb3IoJTVCMTUlNUQpJTBBJTBBb3V0cHV0cyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzJTJDJTIwc3RhcnRfcG9zaXRpb25zJTNEdGFyZ2V0X3N0YXJ0X2luZGV4JTJDJTIwZW5kX3Bvc2l0aW9ucyUzRHRhcmdldF9lbmRfaW5kZXgpJTBBbG9zcyUyMCUzRCUyMG91dHB1dHMubG9zcyUwQXJvdW5kKGxvc3MuaXRlbSgpJTJDJTIwMik=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, XLMRobertaForQuestionAnswering
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;FacebookAI/xlm-roberta-base&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = XLMRobertaForQuestionAnswering.from_pretrained(<span class="hljs-string">&quot;FacebookAI/xlm-roberta-base&quot;</span>)

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
...`,wrap:!1}}),{c(){t=c("p"),t.textContent=u,n=a(),f(m.$$.fragment)},l(i){t=p(i,"P",{"data-svelte-h":!0}),h(t)!=="svelte-11lpom8"&&(t.textContent=u),n=r(i),g(m.$$.fragment,i)},m(i,T){d(i,t,T),d(i,n,T),b(m,i,T),y=!0},p:L,i(i){y||(_(m.$$.fragment,i),y=!0)},o(i){M(m.$$.fragment,i),y=!1},d(i){i&&(s(t),s(n)),k(m,i)}}}function jr(w){let t,u,n,m,y,i="<em>This model was released on 2019-11-05 and added to Hugging Face Transformers on 2020-11-16.</em>",T,R,Io='<div class="flex flex-wrap space-x-1"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/> <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/></div>',Ze,se,Bo,Be,Ks='<a href="https://huggingface.co/papers/1911.02116" rel="nofollow">XLM-RoBERTa</a> is a large multilingual masked language model trained on 2.5TB of filtered CommonCrawl data across 100 languages. It shows that scaling the model provides strong performance gains on high-resource and low-resource languages. The model uses the <a href="./roberta">RoBERTa</a> pretraining objectives on the <a href="./xlm">XLM</a> model.',Go,Ge,ea='You can find all the original XLM-RoBERTa checkpoints under the <a href="https://huggingface.co/FacebookAI" rel="nofollow">Facebook AI community</a> organization.',No,ue,Vo,Ne,ta='The example below demonstrates how to predict the <code>&lt;mask&gt;</code> token with <a href="/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.Pipeline">Pipeline</a>, <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel">AutoModel</a>, and from the command line.',Ho,fe,Eo,Ve,oa='Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the <a href="../quantization">quantization guide</a> overview for more available quantization backends.',So,He,na='The example below uses <a href="../quantization/bitsandbytes">bitsandbytes</a> the quantive the weights to 4 bits',Ao,Ee,Yo,Se,Qo,Ae,sa="<li>Unlike some XLM models, XLM-RoBERTa doesn’t require <code>lang</code> tensors to understand what language is being used. It automatically determines the language from the input IDs</li>",Po,Ye,Do,Qe,aa="A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with XLM-RoBERTa. If you’re interested in submitting a resource to be included here, please feel free to open a Pull Request and we’ll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.",Oo,Pe,Ko,De,ra='<li>A blog post on how to <a href="https://www.philschmid.de/habana-distributed-training" rel="nofollow">finetune XLM RoBERTa for multiclass classification with Habana Gaudi on AWS</a></li> <li><a href="/docs/transformers/v4.56.2/en/model_doc/xlm-roberta#transformers.XLMRobertaForSequenceClassification">XLMRobertaForSequenceClassification</a> is supported by this <a href="https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification" rel="nofollow">example script</a> and <a href="https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/text_classification.ipynb" rel="nofollow">notebook</a>..</li> <li><a href="https://huggingface.co/docs/transformers/tasks/sequence_classification" rel="nofollow">Text classification</a> chapter of the 🤗 Hugging Face Task Guides.</li> <li><a href="../tasks/sequence_classification">Text classification task guide</a></li>',en,Oe,tn,Ke,ia='<li><a href="/docs/transformers/v4.56.2/en/model_doc/xlm-roberta#transformers.XLMRobertaForTokenClassification">XLMRobertaForTokenClassification</a> is supported by this <a href="https://github.com/huggingface/transformers/tree/main/examples/pytorch/token-classification" rel="nofollow">example script</a> and <a href="https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/token_classification.ipynb" rel="nofollow">notebook</a>.</li> <li><a href="https://huggingface.co/course/chapter7/2?fw=pt" rel="nofollow">Token classification</a> chapter of the 🤗 Hugging Face Course.</li> <li><a href="../tasks/token_classification">Token classification task guide</a></li>',on,et,nn,tt,la='<li><a href="/docs/transformers/v4.56.2/en/model_doc/xlm-roberta#transformers.XLMRobertaForCausalLM">XLMRobertaForCausalLM</a> is supported by this <a href="https://github.com/huggingface/transformers/tree/main/examples/pytorch/language-modeling" rel="nofollow">example script</a> and <a href="https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling.ipynb" rel="nofollow">notebook</a>.</li> <li><a href="https://huggingface.co/docs/transformers/tasks/language_modeling" rel="nofollow">Causal language modeling</a> chapter of the 🤗 Hugging Face Task Guides.</li> <li><a href="../tasks/language_modeling">Causal language modeling task guide</a></li>',sn,ot,an,nt,da='<li><a href="/docs/transformers/v4.56.2/en/model_doc/xlm-roberta#transformers.XLMRobertaForMaskedLM">XLMRobertaForMaskedLM</a> is supported by this <a href="https://github.com/huggingface/transformers/tree/main/examples/pytorch/language-modeling#robertabertdistilbert-and-masked-language-modeling" rel="nofollow">example script</a> and <a href="https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling.ipynb" rel="nofollow">notebook</a>.</li> <li><a href="https://huggingface.co/course/chapter7/3?fw=pt" rel="nofollow">Masked language modeling</a> chapter of the 🤗 Hugging Face Course.</li> <li><a href="../tasks/masked_language_modeling">Masked language modeling</a></li>',rn,st,ln,at,ca='<li><a href="/docs/transformers/v4.56.2/en/model_doc/xlm-roberta#transformers.XLMRobertaForQuestionAnswering">XLMRobertaForQuestionAnswering</a> is supported by this <a href="https://github.com/huggingface/transformers/tree/main/examples/pytorch/question-answering" rel="nofollow">example script</a> and <a href="https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/question_answering.ipynb" rel="nofollow">notebook</a>.</li> <li><a href="https://huggingface.co/course/chapter7/7?fw=pt" rel="nofollow">Question answering</a> chapter of the 🤗 Hugging Face Course.</li> <li><a href="../tasks/question_answering">Question answering task guide</a></li>',dn,rt,pa="<strong>Multiple choice</strong>",cn,it,ma='<li><a href="/docs/transformers/v4.56.2/en/model_doc/xlm-roberta#transformers.XLMRobertaForMultipleChoice">XLMRobertaForMultipleChoice</a> is supported by this <a href="https://github.com/huggingface/transformers/tree/main/examples/pytorch/multiple-choice" rel="nofollow">example script</a> and <a href="https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/multiple_choice.ipynb" rel="nofollow">notebook</a>.</li> <li><a href="../tasks/multiple_choice">Multiple choice task guide</a></li>',pn,lt,ha="🚀 Deploy",mn,dt,ua='<li>A blog post on how to <a href="https://www.philschmid.de/multilingual-serverless-xlm-roberta-with-huggingface" rel="nofollow">Deploy Serverless XLM RoBERTa on AWS Lambda</a>.</li>',hn,ge,un,ct,fn,B,pt,qn,At,fa=`This is the configuration class to store the configuration of a <a href="/docs/transformers/v4.56.2/en/model_doc/xlm-roberta#transformers.XLMRobertaModel">XLMRobertaModel</a> or a <code>TFXLMRobertaModel</code>. It
is used to instantiate a XLM-RoBERTa model according to the specified arguments, defining the model architecture.
Instantiating a configuration with the defaults will yield a similar configuration to that of the XLMRoBERTa
<a href="https://huggingface.co/FacebookAI/xlm-roberta-base" rel="nofollow">FacebookAI/xlm-roberta-base</a> architecture.`,Wn,Yt,ga=`Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> and can be used to control the model outputs. Read the
documentation from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> for more information.`,Zn,be,gn,mt,bn,x,ht,Bn,Qt,ba=`Adapted from <a href="/docs/transformers/v4.56.2/en/model_doc/roberta#transformers.RobertaTokenizer">RobertaTokenizer</a> and <a href="/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.XLNetTokenizer">XLNetTokenizer</a>. Based on
<a href="https://github.com/google/sentencepiece" rel="nofollow">SentencePiece</a>.`,Gn,Pt,_a=`This tokenizer inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer">PreTrainedTokenizer</a> which contains most of the main methods. Users should refer to
this superclass for more information regarding those methods.`,Nn,ae,ut,Vn,Dt,Ma=`Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
adding special tokens. An XLM-RoBERTa sequence has the following format:`,Hn,Ot,ka="<li>single sequence: <code>&lt;s&gt; X &lt;/s&gt;</code></li> <li>pair of sequences: <code>&lt;s&gt; A &lt;/s&gt;&lt;/s&gt; B &lt;/s&gt;</code></li>",En,_e,ft,Sn,Kt,Ta=`Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
special tokens using the tokenizer <code>prepare_for_model</code> method.`,An,Me,gt,Yn,eo,ya=`Create a mask from the two sequences passed to be used in a sequence-pair classification task. XLM-RoBERTa does
not make use of token type ids, therefore a list of zeros is returned.`,Qn,to,bt,_n,_t,Mn,j,Mt,Pn,oo,wa=`Construct a “fast” XLM-RoBERTa tokenizer (backed by HuggingFace’s <em>tokenizers</em> library). Adapted from
<a href="/docs/transformers/v4.56.2/en/model_doc/roberta#transformers.RobertaTokenizer">RobertaTokenizer</a> and <a href="/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.XLNetTokenizer">XLNetTokenizer</a>. Based on
<a href="https://huggingface.co/docs/tokenizers/python/latest/components.html?highlight=BPE#models" rel="nofollow">BPE</a>.`,Dn,no,va=`This tokenizer inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast">PreTrainedTokenizerFast</a> which contains most of the main methods. Users should
refer to this superclass for more information regarding those methods.`,On,re,kt,Kn,so,Ja=`Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
adding special tokens. An XLM-RoBERTa sequence has the following format:`,es,ao,$a="<li>single sequence: <code>&lt;s&gt; X &lt;/s&gt;</code></li> <li>pair of sequences: <code>&lt;s&gt; A &lt;/s&gt;&lt;/s&gt; B &lt;/s&gt;</code></li>",ts,ke,Tt,os,ro,Ra=`Create a mask from the two sequences passed to be used in a sequence-pair classification task. XLM-RoBERTa does
not make use of token type ids, therefore a list of zeros is returned.`,kn,yt,Tn,C,wt,ns,io,La="The bare Xlm Roberta Model outputting raw hidden-states without any specific head on top.",ss,lo,xa=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,as,co,Xa=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,rs,ie,vt,is,po,ja='The <a href="/docs/transformers/v4.56.2/en/model_doc/xlm-roberta#transformers.XLMRobertaModel">XLMRobertaModel</a> forward method, overrides the <code>__call__</code> special method.',ls,Te,yn,Jt,wn,z,$t,ds,mo,Ca="XLM-RoBERTa Model with a <code>language modeling</code> head on top for CLM fine-tuning.",cs,ho,za=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,ps,uo,Fa=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,ms,Q,Rt,hs,fo,Ua='The <a href="/docs/transformers/v4.56.2/en/model_doc/xlm-roberta#transformers.XLMRobertaForCausalLM">XLMRobertaForCausalLM</a> forward method, overrides the <code>__call__</code> special method.',us,ye,fs,we,vn,Lt,Jn,F,xt,gs,go,Ia="The Xlm Roberta Model with a <code>language modeling</code> head on top.”",bs,bo,qa=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,_s,_o,Wa=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Ms,P,Xt,ks,Mo,Za='The <a href="/docs/transformers/v4.56.2/en/model_doc/xlm-roberta#transformers.XLMRobertaForMaskedLM">XLMRobertaForMaskedLM</a> forward method, overrides the <code>__call__</code> special method.',Ts,ve,ys,Je,$n,jt,Rn,U,Ct,ws,ko,Ba=`XLM-RoBERTa Model transformer with a sequence classification/regression head on top (a linear layer on top of the
pooled output) e.g. for GLUE tasks.`,vs,To,Ga=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Js,yo,Na=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,$s,Z,zt,Rs,wo,Va='The <a href="/docs/transformers/v4.56.2/en/model_doc/xlm-roberta#transformers.XLMRobertaForSequenceClassification">XLMRobertaForSequenceClassification</a> forward method, overrides the <code>__call__</code> special method.',Ls,$e,xs,Re,Xs,Le,Ln,Ft,xn,I,Ut,js,vo,Ha=`The Xlm Roberta Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
softmax) e.g. for RocStories/SWAG tasks.`,Cs,Jo,Ea=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,zs,$o,Sa=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Fs,D,It,Us,Ro,Aa='The <a href="/docs/transformers/v4.56.2/en/model_doc/xlm-roberta#transformers.XLMRobertaForMultipleChoice">XLMRobertaForMultipleChoice</a> forward method, overrides the <code>__call__</code> special method.',Is,xe,qs,Xe,Xn,qt,jn,q,Wt,Ws,Lo,Ya=`The Xlm Roberta transformer with a token classification head on top (a linear layer on top of the hidden-states
output) e.g. for Named-Entity-Recognition (NER) tasks.`,Zs,xo,Qa=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Bs,Xo,Pa=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Gs,O,Zt,Ns,jo,Da='The <a href="/docs/transformers/v4.56.2/en/model_doc/xlm-roberta#transformers.XLMRobertaForTokenClassification">XLMRobertaForTokenClassification</a> forward method, overrides the <code>__call__</code> special method.',Vs,je,Hs,Ce,Cn,Bt,zn,W,Gt,Es,Co,Oa=`The Xlm Roberta transformer with a span classification head on top for extractive question-answering tasks like
SQuAD (a linear layer on top of the hidden-states output to compute <code>span start logits</code> and <code>span end logits</code>).`,Ss,zo,Ka=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,As,Fo,er=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Ys,K,Nt,Qs,Uo,tr='The <a href="/docs/transformers/v4.56.2/en/model_doc/xlm-roberta#transformers.XLMRobertaForQuestionAnswering">XLMRobertaForQuestionAnswering</a> forward method, overrides the <code>__call__</code> special method.',Ps,ze,Ds,Fe,Fn,Vt,Un,qo,In;return se=new Y({props:{title:"XLM-RoBERTa",local:"xlm-roberta",headingTag:"h1"}}),ue=new he({props:{warning:!1,$$slots:{default:[cr]},$$scope:{ctx:w}}}),fe=new dr({props:{id:"usage",options:["Pipeline","AutoModel","transformers CLI"],$$slots:{default:[ur]},$$scope:{ctx:w}}}),Ee=new ee({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b01vZGVsRm9yTWFza2VkTE0lMkMlMjBBdXRvVG9rZW5pemVyJTJDJTIwQml0c0FuZEJ5dGVzQ29uZmlnJTBBJTBBcXVhbnRpemF0aW9uX2NvbmZpZyUyMCUzRCUyMEJpdHNBbmRCeXRlc0NvbmZpZyglMEElMjAlMjAlMjAlMjBsb2FkX2luXzRiaXQlM0RUcnVlJTJDJTBBJTIwJTIwJTIwJTIwYm5iXzRiaXRfY29tcHV0ZV9kdHlwZSUzRHRvcmNoLmJmbG9hdDE2JTBBJTIwJTIwJTIwJTIwYm5iXzRiaXRfcXVhbnRfdHlwZSUzRCUyMm5mNCUyMiUyQyUyMCUyMCUyMyUyMG9yJTIwJTIyZnA0JTIyJTIwZm9yJTIwZmxvYXQlMjA0LWJpdCUyMHF1YW50aXphdGlvbiUwQSUyMCUyMCUyMCUyMGJuYl80Yml0X3VzZV9kb3VibGVfcXVhbnQlM0RUcnVlJTJDJTIwJTIwJTIzJTIwdXNlJTIwZG91YmxlJTIwcXVhbnRpemF0aW9uJTIwZm9yJTIwYmV0dGVyJTIwcGVyZm9ybWFuY2UlMEEpJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIyZmFjZWJvb2slMkZ4bG0tcm9iZXJ0YS1sYXJnZSUyMiklMEFtb2RlbCUyMCUzRCUyMEF1dG9Nb2RlbEZvck1hc2tlZExNLmZyb21fcHJldHJhaW5lZCglMEElMjAlMjAlMjAlMjAlMjJmYWNlYm9vayUyRnhsbS1yb2JlcnRhLWxhcmdlJTIyJTJDJTBBJTIwJTIwJTIwJTIwZHR5cGUlM0R0b3JjaC5mbG9hdDE2JTJDJTBBJTIwJTIwJTIwJTIwZGV2aWNlX21hcCUzRCUyMmF1dG8lMjIlMkMlMEElMjAlMjAlMjAlMjBhdHRuX2ltcGxlbWVudGF0aW9uJTNEJTIyZmxhc2hfYXR0ZW50aW9uXzIlMjIlMkMlMEElMjAlMjAlMjAlMjBxdWFudGl6YXRpb25fY29uZmlnJTNEcXVhbnRpemF0aW9uX2NvbmZpZyUwQSklMEElMEFpbnB1dHMlMjAlM0QlMjB0b2tlbml6ZXIoJTIyQm9uam91ciUyQyUyMGplJTIwc3VpcyUyMHVuJTIwbW9kJUMzJUE4bGUlMjAlM0NtYXNrJTNFLiUyMiUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpLnRvKG1vZGVsLmRldmljZSklMEFvdXRwdXRzJTIwJTNEJTIwbW9kZWwuZ2VuZXJhdGUoKippbnB1dHMlMkMlMjBtYXhfbmV3X3Rva2VucyUzRDEwMCklMEFwcmludCh0b2tlbml6ZXIuZGVjb2RlKG91dHB1dHMlNUIwJTVEJTJDJTIwc2tpcF9zcGVjaWFsX3Rva2VucyUzRFRydWUpKQ==",highlighted:`<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoModelForMaskedLM, AutoTokenizer, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=<span class="hljs-literal">True</span>,
    bnb_4bit_compute_dtype=torch.bfloat16
    bnb_4bit_quant_type=<span class="hljs-string">&quot;nf4&quot;</span>,  <span class="hljs-comment"># or &quot;fp4&quot; for float 4-bit quantization</span>
    bnb_4bit_use_double_quant=<span class="hljs-literal">True</span>,  <span class="hljs-comment"># use double quantization for better performance</span>
)
tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;facebook/xlm-roberta-large&quot;</span>)
model = AutoModelForMaskedLM.from_pretrained(
    <span class="hljs-string">&quot;facebook/xlm-roberta-large&quot;</span>,
    dtype=torch.float16,
    device_map=<span class="hljs-string">&quot;auto&quot;</span>,
    attn_implementation=<span class="hljs-string">&quot;flash_attention_2&quot;</span>,
    quantization_config=quantization_config
)

inputs = tokenizer(<span class="hljs-string">&quot;Bonjour, je suis un modèle &lt;mask&gt;.&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>).to(model.device)
outputs = model.generate(**inputs, max_new_tokens=<span class="hljs-number">100</span>)
<span class="hljs-built_in">print</span>(tokenizer.decode(outputs[<span class="hljs-number">0</span>], skip_special_tokens=<span class="hljs-literal">True</span>))`,wrap:!1}}),Se=new Y({props:{title:"Notes",local:"notes",headingTag:"h2"}}),Ye=new Y({props:{title:"Resources",local:"resources",headingTag:"h2"}}),Pe=new Zo({props:{pipeline:"text-classification"}}),Oe=new Zo({props:{pipeline:"token-classification"}}),et=new Zo({props:{pipeline:"text-generation"}}),ot=new Zo({props:{pipeline:"fill-mask"}}),st=new Zo({props:{pipeline:"question-answering"}}),ge=new he({props:{$$slots:{default:[fr]},$$scope:{ctx:w}}}),ct=new Y({props:{title:"XLMRobertaConfig",local:"transformers.XLMRobertaConfig",headingTag:"h2"}}),pt=new $({props:{name:"class transformers.XLMRobertaConfig",anchor:"transformers.XLMRobertaConfig",parameters:[{name:"vocab_size",val:" = 30522"},{name:"hidden_size",val:" = 768"},{name:"num_hidden_layers",val:" = 12"},{name:"num_attention_heads",val:" = 12"},{name:"intermediate_size",val:" = 3072"},{name:"hidden_act",val:" = 'gelu'"},{name:"hidden_dropout_prob",val:" = 0.1"},{name:"attention_probs_dropout_prob",val:" = 0.1"},{name:"max_position_embeddings",val:" = 512"},{name:"type_vocab_size",val:" = 2"},{name:"initializer_range",val:" = 0.02"},{name:"layer_norm_eps",val:" = 1e-12"},{name:"pad_token_id",val:" = 1"},{name:"bos_token_id",val:" = 0"},{name:"eos_token_id",val:" = 2"},{name:"position_embedding_type",val:" = 'absolute'"},{name:"use_cache",val:" = True"},{name:"classifier_dropout",val:" = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.XLMRobertaConfig.vocab_size",description:`<strong>vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 30522) &#x2014;
Vocabulary size of the XLM-RoBERTa model. Defines the number of different tokens that can be represented by
the <code>inputs_ids</code> passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/xlm-roberta#transformers.XLMRobertaModel">XLMRobertaModel</a> or <code>TFXLMRobertaModel</code>.`,name:"vocab_size"},{anchor:"transformers.XLMRobertaConfig.hidden_size",description:`<strong>hidden_size</strong> (<code>int</code>, <em>optional</em>, defaults to 768) &#x2014;
Dimensionality of the encoder layers and the pooler layer.`,name:"hidden_size"},{anchor:"transformers.XLMRobertaConfig.num_hidden_layers",description:`<strong>num_hidden_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 12) &#x2014;
Number of hidden layers in the Transformer encoder.`,name:"num_hidden_layers"},{anchor:"transformers.XLMRobertaConfig.num_attention_heads",description:`<strong>num_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 12) &#x2014;
Number of attention heads for each attention layer in the Transformer encoder.`,name:"num_attention_heads"},{anchor:"transformers.XLMRobertaConfig.intermediate_size",description:`<strong>intermediate_size</strong> (<code>int</code>, <em>optional</em>, defaults to 3072) &#x2014;
Dimensionality of the &#x201C;intermediate&#x201D; (often named feed-forward) layer in the Transformer encoder.`,name:"intermediate_size"},{anchor:"transformers.XLMRobertaConfig.hidden_act",description:`<strong>hidden_act</strong> (<code>str</code> or <code>Callable</code>, <em>optional</em>, defaults to <code>&quot;gelu&quot;</code>) &#x2014;
The non-linear activation function (function or string) in the encoder and pooler. If string, <code>&quot;gelu&quot;</code>,
<code>&quot;relu&quot;</code>, <code>&quot;silu&quot;</code> and <code>&quot;gelu_new&quot;</code> are supported.`,name:"hidden_act"},{anchor:"transformers.XLMRobertaConfig.hidden_dropout_prob",description:`<strong>hidden_dropout_prob</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.`,name:"hidden_dropout_prob"},{anchor:"transformers.XLMRobertaConfig.attention_probs_dropout_prob",description:`<strong>attention_probs_dropout_prob</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout ratio for the attention probabilities.`,name:"attention_probs_dropout_prob"},{anchor:"transformers.XLMRobertaConfig.max_position_embeddings",description:`<strong>max_position_embeddings</strong> (<code>int</code>, <em>optional</em>, defaults to 512) &#x2014;
The maximum sequence length that this model might ever be used with. Typically set this to something large
just in case (e.g., 512 or 1024 or 2048).`,name:"max_position_embeddings"},{anchor:"transformers.XLMRobertaConfig.type_vocab_size",description:`<strong>type_vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 2) &#x2014;
The vocabulary size of the <code>token_type_ids</code> passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/xlm-roberta#transformers.XLMRobertaModel">XLMRobertaModel</a> or
<code>TFXLMRobertaModel</code>.`,name:"type_vocab_size"},{anchor:"transformers.XLMRobertaConfig.initializer_range",description:`<strong>initializer_range</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.`,name:"initializer_range"},{anchor:"transformers.XLMRobertaConfig.layer_norm_eps",description:`<strong>layer_norm_eps</strong> (<code>float</code>, <em>optional</em>, defaults to 1e-12) &#x2014;
The epsilon used by the layer normalization layers.`,name:"layer_norm_eps"},{anchor:"transformers.XLMRobertaConfig.position_embedding_type",description:`<strong>position_embedding_type</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;absolute&quot;</code>) &#x2014;
Type of position embedding. Choose one of <code>&quot;absolute&quot;</code>, <code>&quot;relative_key&quot;</code>, <code>&quot;relative_key_query&quot;</code>. For
positional embeddings use <code>&quot;absolute&quot;</code>. For more information on <code>&quot;relative_key&quot;</code>, please refer to
<a href="https://huggingface.co/papers/1803.02155" rel="nofollow">Self-Attention with Relative Position Representations (Shaw et al.)</a>.
For more information on <code>&quot;relative_key_query&quot;</code>, please refer to <em>Method 4</em> in <a href="https://huggingface.co/papers/2009.13658" rel="nofollow">Improve Transformer Models
with Better Relative Position Embeddings (Huang et al.)</a>.`,name:"position_embedding_type"},{anchor:"transformers.XLMRobertaConfig.is_decoder",description:`<strong>is_decoder</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether the model is used as a decoder or not. If <code>False</code>, the model is used as an encoder.`,name:"is_decoder"},{anchor:"transformers.XLMRobertaConfig.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not the model should return the last key/values attentions (not used by all models). Only
relevant if <code>config.is_decoder=True</code>.`,name:"use_cache"},{anchor:"transformers.XLMRobertaConfig.classifier_dropout",description:`<strong>classifier_dropout</strong> (<code>float</code>, <em>optional</em>) &#x2014;
The dropout ratio for the classification head.`,name:"classifier_dropout"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlm_roberta/configuration_xlm_roberta.py#L29"}}),be=new We({props:{anchor:"transformers.XLMRobertaConfig.example",$$slots:{default:[gr]},$$scope:{ctx:w}}}),mt=new Y({props:{title:"XLMRobertaTokenizer",local:"transformers.XLMRobertaTokenizer",headingTag:"h2"}}),ht=new $({props:{name:"class transformers.XLMRobertaTokenizer",anchor:"transformers.XLMRobertaTokenizer",parameters:[{name:"vocab_file",val:""},{name:"bos_token",val:" = '<s>'"},{name:"eos_token",val:" = '</s>'"},{name:"sep_token",val:" = '</s>'"},{name:"cls_token",val:" = '<s>'"},{name:"unk_token",val:" = '<unk>'"},{name:"pad_token",val:" = '<pad>'"},{name:"mask_token",val:" = '<mask>'"},{name:"sp_model_kwargs",val:": typing.Optional[dict[str, typing.Any]] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.XLMRobertaTokenizer.vocab_file",description:`<strong>vocab_file</strong> (<code>str</code>) &#x2014;
Path to the vocabulary file.`,name:"vocab_file"},{anchor:"transformers.XLMRobertaTokenizer.bos_token",description:`<strong>bos_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;s&gt;&quot;</code>) &#x2014;
The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.</p>
<div class="course-tip  bg-gradient-to-br dark:bg-gradient-to-r before:border-green-500 dark:before:border-green-800 from-green-50 dark:from-gray-900 to-white dark:to-gray-950 border border-green-50 text-green-700 dark:text-gray-400">
						
<p>When building a sequence using special tokens, this is not the token that is used for the beginning of
sequence. The token used is the <code>cls_token</code>.</p>

					</div>`,name:"bos_token"},{anchor:"transformers.XLMRobertaTokenizer.eos_token",description:`<strong>eos_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;/s&gt;&quot;</code>) &#x2014;
The end of sequence token.</p>
<div class="course-tip  bg-gradient-to-br dark:bg-gradient-to-r before:border-green-500 dark:before:border-green-800 from-green-50 dark:from-gray-900 to-white dark:to-gray-950 border border-green-50 text-green-700 dark:text-gray-400">
						
<p>When building a sequence using special tokens, this is not the token that is used for the end of sequence.
The token used is the <code>sep_token</code>.</p>

					</div>`,name:"eos_token"},{anchor:"transformers.XLMRobertaTokenizer.sep_token",description:`<strong>sep_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;/s&gt;&quot;</code>) &#x2014;
The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
sequence classification or for a text and a question for question answering. It is also used as the last
token of a sequence built with special tokens.`,name:"sep_token"},{anchor:"transformers.XLMRobertaTokenizer.cls_token",description:`<strong>cls_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;s&gt;&quot;</code>) &#x2014;
The classifier token which is used when doing sequence classification (classification of the whole sequence
instead of per-token classification). It is the first token of the sequence when built with special tokens.`,name:"cls_token"},{anchor:"transformers.XLMRobertaTokenizer.unk_token",description:`<strong>unk_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;unk&gt;&quot;</code>) &#x2014;
The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
token instead.`,name:"unk_token"},{anchor:"transformers.XLMRobertaTokenizer.pad_token",description:`<strong>pad_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;pad&gt;&quot;</code>) &#x2014;
The token used for padding, for example when batching sequences of different lengths.`,name:"pad_token"},{anchor:"transformers.XLMRobertaTokenizer.mask_token",description:`<strong>mask_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;mask&gt;&quot;</code>) &#x2014;
The token used for masking values. This is the token used when training this model with masked language
modeling. This is the token which the model will try to predict.`,name:"mask_token"},{anchor:"transformers.XLMRobertaTokenizer.sp_model_kwargs",description:`<strong>sp_model_kwargs</strong> (<code>dict</code>, <em>optional</em>) &#x2014;
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
</ul>`,name:"sp_model_kwargs"},{anchor:"transformers.XLMRobertaTokenizer.sp_model",description:`<strong>sp_model</strong> (<code>SentencePieceProcessor</code>) &#x2014;
The <em>SentencePiece</em> processor that is used for every conversion (string, tokens and IDs).`,name:"sp_model"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlm_roberta/tokenization_xlm_roberta.py#L36"}}),ut=new $({props:{name:"build_inputs_with_special_tokens",anchor:"transformers.XLMRobertaTokenizer.build_inputs_with_special_tokens",parameters:[{name:"token_ids_0",val:": list"},{name:"token_ids_1",val:": typing.Optional[list[int]] = None"}],parametersDescription:[{anchor:"transformers.XLMRobertaTokenizer.build_inputs_with_special_tokens.token_ids_0",description:`<strong>token_ids_0</strong> (<code>list[int]</code>) &#x2014;
List of IDs to which the special tokens will be added.`,name:"token_ids_0"},{anchor:"transformers.XLMRobertaTokenizer.build_inputs_with_special_tokens.token_ids_1",description:`<strong>token_ids_1</strong> (<code>list[int]</code>, <em>optional</em>) &#x2014;
Optional second list of IDs for sequence pairs.`,name:"token_ids_1"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlm_roberta/tokenization_xlm_roberta.py#L171",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>List of <a href="../glossary#input-ids">input IDs</a> with the appropriate special tokens.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>list[int]</code></p>
`}}),ft=new $({props:{name:"get_special_tokens_mask",anchor:"transformers.XLMRobertaTokenizer.get_special_tokens_mask",parameters:[{name:"token_ids_0",val:": list"},{name:"token_ids_1",val:": typing.Optional[list[int]] = None"},{name:"already_has_special_tokens",val:": bool = False"}],parametersDescription:[{anchor:"transformers.XLMRobertaTokenizer.get_special_tokens_mask.token_ids_0",description:`<strong>token_ids_0</strong> (<code>list[int]</code>) &#x2014;
List of IDs.`,name:"token_ids_0"},{anchor:"transformers.XLMRobertaTokenizer.get_special_tokens_mask.token_ids_1",description:`<strong>token_ids_1</strong> (<code>list[int]</code>, <em>optional</em>) &#x2014;
Optional second list of IDs for sequence pairs.`,name:"token_ids_1"},{anchor:"transformers.XLMRobertaTokenizer.get_special_tokens_mask.already_has_special_tokens",description:`<strong>already_has_special_tokens</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not the token list is already formatted with special tokens for the model.`,name:"already_has_special_tokens"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlm_roberta/tokenization_xlm_roberta.py#L197",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>list[int]</code></p>
`}}),gt=new $({props:{name:"create_token_type_ids_from_sequences",anchor:"transformers.XLMRobertaTokenizer.create_token_type_ids_from_sequences",parameters:[{name:"token_ids_0",val:": list"},{name:"token_ids_1",val:": typing.Optional[list[int]] = None"}],parametersDescription:[{anchor:"transformers.XLMRobertaTokenizer.create_token_type_ids_from_sequences.token_ids_0",description:`<strong>token_ids_0</strong> (<code>list[int]</code>) &#x2014;
List of IDs.`,name:"token_ids_0"},{anchor:"transformers.XLMRobertaTokenizer.create_token_type_ids_from_sequences.token_ids_1",description:`<strong>token_ids_1</strong> (<code>list[int]</code>, <em>optional</em>) &#x2014;
Optional second list of IDs for sequence pairs.`,name:"token_ids_1"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlm_roberta/tokenization_xlm_roberta.py#L225",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>List of zeros.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>list[int]</code></p>
`}}),bt=new $({props:{name:"save_vocabulary",anchor:"transformers.XLMRobertaTokenizer.save_vocabulary",parameters:[{name:"save_directory",val:": str"},{name:"filename_prefix",val:": typing.Optional[str] = None"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlm_roberta/tokenization_xlm_roberta.py#L283"}}),_t=new Y({props:{title:"XLMRobertaTokenizerFast",local:"transformers.XLMRobertaTokenizerFast",headingTag:"h2"}}),Mt=new $({props:{name:"class transformers.XLMRobertaTokenizerFast",anchor:"transformers.XLMRobertaTokenizerFast",parameters:[{name:"vocab_file",val:" = None"},{name:"tokenizer_file",val:" = None"},{name:"bos_token",val:" = '<s>'"},{name:"eos_token",val:" = '</s>'"},{name:"sep_token",val:" = '</s>'"},{name:"cls_token",val:" = '<s>'"},{name:"unk_token",val:" = '<unk>'"},{name:"pad_token",val:" = '<pad>'"},{name:"mask_token",val:" = '<mask>'"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.XLMRobertaTokenizerFast.vocab_file",description:`<strong>vocab_file</strong> (<code>str</code>) &#x2014;
Path to the vocabulary file.`,name:"vocab_file"},{anchor:"transformers.XLMRobertaTokenizerFast.bos_token",description:`<strong>bos_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;s&gt;&quot;</code>) &#x2014;
The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.</p>
<div class="course-tip  bg-gradient-to-br dark:bg-gradient-to-r before:border-green-500 dark:before:border-green-800 from-green-50 dark:from-gray-900 to-white dark:to-gray-950 border border-green-50 text-green-700 dark:text-gray-400">
						
<p>When building a sequence using special tokens, this is not the token that is used for the beginning of
sequence. The token used is the <code>cls_token</code>.</p>

					</div>`,name:"bos_token"},{anchor:"transformers.XLMRobertaTokenizerFast.eos_token",description:`<strong>eos_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;/s&gt;&quot;</code>) &#x2014;
The end of sequence token.</p>
<div class="course-tip  bg-gradient-to-br dark:bg-gradient-to-r before:border-green-500 dark:before:border-green-800 from-green-50 dark:from-gray-900 to-white dark:to-gray-950 border border-green-50 text-green-700 dark:text-gray-400">
						
<p>When building a sequence using special tokens, this is not the token that is used for the end of sequence.
The token used is the <code>sep_token</code>.</p>

					</div>`,name:"eos_token"},{anchor:"transformers.XLMRobertaTokenizerFast.sep_token",description:`<strong>sep_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;/s&gt;&quot;</code>) &#x2014;
The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
sequence classification or for a text and a question for question answering. It is also used as the last
token of a sequence built with special tokens.`,name:"sep_token"},{anchor:"transformers.XLMRobertaTokenizerFast.cls_token",description:`<strong>cls_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;s&gt;&quot;</code>) &#x2014;
The classifier token which is used when doing sequence classification (classification of the whole sequence
instead of per-token classification). It is the first token of the sequence when built with special tokens.`,name:"cls_token"},{anchor:"transformers.XLMRobertaTokenizerFast.unk_token",description:`<strong>unk_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;unk&gt;&quot;</code>) &#x2014;
The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
token instead.`,name:"unk_token"},{anchor:"transformers.XLMRobertaTokenizerFast.pad_token",description:`<strong>pad_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;pad&gt;&quot;</code>) &#x2014;
The token used for padding, for example when batching sequences of different lengths.`,name:"pad_token"},{anchor:"transformers.XLMRobertaTokenizerFast.mask_token",description:`<strong>mask_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;mask&gt;&quot;</code>) &#x2014;
The token used for masking values. This is the token used when training this model with masked language
modeling. This is the token which the model will try to predict.`,name:"mask_token"},{anchor:"transformers.XLMRobertaTokenizerFast.additional_special_tokens",description:`<strong>additional_special_tokens</strong> (<code>list[str]</code>, <em>optional</em>, defaults to <code>[&quot;&lt;s&gt;NOTUSED&quot;, &quot;&lt;/s&gt;NOTUSED&quot;]</code>) &#x2014;
Additional special tokens used by the tokenizer.`,name:"additional_special_tokens"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlm_roberta/tokenization_xlm_roberta_fast.py#L37"}}),kt=new $({props:{name:"build_inputs_with_special_tokens",anchor:"transformers.XLMRobertaTokenizerFast.build_inputs_with_special_tokens",parameters:[{name:"token_ids_0",val:": list"},{name:"token_ids_1",val:": typing.Optional[list[int]] = None"}],parametersDescription:[{anchor:"transformers.XLMRobertaTokenizerFast.build_inputs_with_special_tokens.token_ids_0",description:`<strong>token_ids_0</strong> (<code>list[int]</code>) &#x2014;
List of IDs to which the special tokens will be added.`,name:"token_ids_0"},{anchor:"transformers.XLMRobertaTokenizerFast.build_inputs_with_special_tokens.token_ids_1",description:`<strong>token_ids_1</strong> (<code>list[int]</code>, <em>optional</em>) &#x2014;
Optional second list of IDs for sequence pairs.`,name:"token_ids_1"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlm_roberta/tokenization_xlm_roberta_fast.py#L123",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>List of <a href="../glossary#input-ids">input IDs</a> with the appropriate special tokens.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>list[int]</code></p>
`}}),Tt=new $({props:{name:"create_token_type_ids_from_sequences",anchor:"transformers.XLMRobertaTokenizerFast.create_token_type_ids_from_sequences",parameters:[{name:"token_ids_0",val:": list"},{name:"token_ids_1",val:": typing.Optional[list[int]] = None"}],parametersDescription:[{anchor:"transformers.XLMRobertaTokenizerFast.create_token_type_ids_from_sequences.token_ids_0",description:`<strong>token_ids_0</strong> (<code>list[int]</code>) &#x2014;
List of IDs.`,name:"token_ids_0"},{anchor:"transformers.XLMRobertaTokenizerFast.create_token_type_ids_from_sequences.token_ids_1",description:`<strong>token_ids_1</strong> (<code>list[int]</code>, <em>optional</em>) &#x2014;
Optional second list of IDs for sequence pairs.`,name:"token_ids_1"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlm_roberta/tokenization_xlm_roberta_fast.py#L149",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>List of zeros.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>list[int]</code></p>
`}}),yt=new Y({props:{title:"XLMRobertaModel",local:"transformers.XLMRobertaModel",headingTag:"h2"}}),wt=new $({props:{name:"class transformers.XLMRobertaModel",anchor:"transformers.XLMRobertaModel",parameters:[{name:"config",val:""},{name:"add_pooling_layer",val:" = True"}],parametersDescription:[{anchor:"transformers.XLMRobertaModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/xlm-roberta#transformers.XLMRobertaModel">XLMRobertaModel</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"},{anchor:"transformers.XLMRobertaModel.add_pooling_layer",description:`<strong>add_pooling_layer</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether to add a pooling layer`,name:"add_pooling_layer"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlm_roberta/modeling_xlm_roberta.py#L694"}}),vt=new $({props:{name:"forward",anchor:"transformers.XLMRobertaModel.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_hidden_states",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"past_key_values",val:": typing.Optional[list[torch.FloatTensor]] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.Tensor] = None"}],parametersDescription:[{anchor:"transformers.XLMRobertaModel.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.XLMRobertaModel.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.XLMRobertaModel.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.XLMRobertaModel.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.XLMRobertaModel.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.XLMRobertaModel.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.XLMRobertaModel.forward.encoder_hidden_states",description:`<strong>encoder_hidden_states</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
if the model is configured as a decoder.`,name:"encoder_hidden_states"},{anchor:"transformers.XLMRobertaModel.forward.encoder_attention_mask",description:`<strong>encoder_attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
the cross-attention if the model is configured as a decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>`,name:"encoder_attention_mask"},{anchor:"transformers.XLMRobertaModel.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>list[torch.FloatTensor]</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.XLMRobertaModel.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.XLMRobertaModel.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.XLMRobertaModel.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.XLMRobertaModel.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.XLMRobertaModel.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.Tensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlm_roberta/modeling_xlm_roberta.py#L730",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions"
>transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/xlm-roberta#transformers.XLMRobertaConfig"
>XLMRobertaConfig</a>) and inputs.</p>
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
`}}),Te=new he({props:{$$slots:{default:[br]},$$scope:{ctx:w}}}),Jt=new Y({props:{title:"XLMRobertaForCausalLM",local:"transformers.XLMRobertaForCausalLM",headingTag:"h2"}}),$t=new $({props:{name:"class transformers.XLMRobertaForCausalLM",anchor:"transformers.XLMRobertaForCausalLM",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.XLMRobertaForCausalLM.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/xlm-roberta#transformers.XLMRobertaForCausalLM">XLMRobertaForCausalLM</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlm_roberta/modeling_xlm_roberta.py#L887"}}),Rt=new $({props:{name:"forward",anchor:"transformers.XLMRobertaForCausalLM.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"encoder_hidden_states",val:": typing.Optional[torch.FloatTensor] = None"},{name:"encoder_attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[tuple[tuple[torch.FloatTensor]]] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.XLMRobertaForCausalLM.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.XLMRobertaForCausalLM.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.XLMRobertaForCausalLM.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0,1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.
This parameter can only be used when the model is initialized with <code>type_vocab_size</code> parameter with value<blockquote>
<p>= 2. All the value in this tensor should be always &lt; type_vocab_size.</p>
</blockquote></li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.XLMRobertaForCausalLM.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.XLMRobertaForCausalLM.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.XLMRobertaForCausalLM.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.XLMRobertaForCausalLM.forward.encoder_hidden_states",description:`<strong>encoder_hidden_states</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
if the model is configured as a decoder.`,name:"encoder_hidden_states"},{anchor:"transformers.XLMRobertaForCausalLM.forward.encoder_attention_mask",description:`<strong>encoder_attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
the cross-attention if the model is configured as a decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>`,name:"encoder_attention_mask"},{anchor:"transformers.XLMRobertaForCausalLM.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the left-to-right language modeling loss (next word prediction). Indices should be in
<code>[-100, 0, ..., config.vocab_size]</code> (see <code>input_ids</code> docstring) Tokens with indices set to <code>-100</code> are
ignored (masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>`,name:"labels"},{anchor:"transformers.XLMRobertaForCausalLM.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>tuple[tuple[torch.FloatTensor]]</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.XLMRobertaForCausalLM.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.XLMRobertaForCausalLM.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.XLMRobertaForCausalLM.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.XLMRobertaForCausalLM.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlm_roberta/modeling_xlm_roberta.py#L908",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithCrossAttentions"
>transformers.modeling_outputs.CausalLMOutputWithCrossAttentions</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/xlm-roberta#transformers.XLMRobertaConfig"
>XLMRobertaConfig</a>) and inputs.</p>
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
`}}),ye=new he({props:{$$slots:{default:[_r]},$$scope:{ctx:w}}}),we=new We({props:{anchor:"transformers.XLMRobertaForCausalLM.forward.example",$$slots:{default:[Mr]},$$scope:{ctx:w}}}),Lt=new Y({props:{title:"XLMRobertaForMaskedLM",local:"transformers.XLMRobertaForMaskedLM",headingTag:"h2"}}),xt=new $({props:{name:"class transformers.XLMRobertaForMaskedLM",anchor:"transformers.XLMRobertaForMaskedLM",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.XLMRobertaForMaskedLM.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/xlm-roberta#transformers.XLMRobertaForMaskedLM">XLMRobertaForMaskedLM</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlm_roberta/modeling_xlm_roberta.py#L1008"}}),Xt=new $({props:{name:"forward",anchor:"transformers.XLMRobertaForMaskedLM.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"encoder_hidden_states",val:": typing.Optional[torch.FloatTensor] = None"},{name:"encoder_attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.XLMRobertaForMaskedLM.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.XLMRobertaForMaskedLM.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.XLMRobertaForMaskedLM.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0,1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.
This parameter can only be used when the model is initialized with <code>type_vocab_size</code> parameter with value<blockquote>
<p>= 2. All the value in this tensor should be always &lt; type_vocab_size.</p>
</blockquote></li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.XLMRobertaForMaskedLM.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.XLMRobertaForMaskedLM.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.XLMRobertaForMaskedLM.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.XLMRobertaForMaskedLM.forward.encoder_hidden_states",description:`<strong>encoder_hidden_states</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
if the model is configured as a decoder.`,name:"encoder_hidden_states"},{anchor:"transformers.XLMRobertaForMaskedLM.forward.encoder_attention_mask",description:`<strong>encoder_attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
the cross-attention if the model is configured as a decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>`,name:"encoder_attention_mask"},{anchor:"transformers.XLMRobertaForMaskedLM.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should be in <code>[-100, 0, ..., config.vocab_size]</code> (see <code>input_ids</code> docstring) Tokens with indices set to <code>-100</code> are ignored (masked), the
loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>`,name:"labels"},{anchor:"transformers.XLMRobertaForMaskedLM.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.XLMRobertaForMaskedLM.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.XLMRobertaForMaskedLM.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlm_roberta/modeling_xlm_roberta.py#L1032",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.MaskedLMOutput"
>transformers.modeling_outputs.MaskedLMOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/xlm-roberta#transformers.XLMRobertaConfig"
>XLMRobertaConfig</a>) and inputs.</p>
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
`}}),ve=new he({props:{$$slots:{default:[kr]},$$scope:{ctx:w}}}),Je=new We({props:{anchor:"transformers.XLMRobertaForMaskedLM.forward.example",$$slots:{default:[Tr]},$$scope:{ctx:w}}}),jt=new Y({props:{title:"XLMRobertaForSequenceClassification",local:"transformers.XLMRobertaForSequenceClassification",headingTag:"h2"}}),Ct=new $({props:{name:"class transformers.XLMRobertaForSequenceClassification",anchor:"transformers.XLMRobertaForSequenceClassification",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.XLMRobertaForSequenceClassification.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/xlm-roberta#transformers.XLMRobertaForSequenceClassification">XLMRobertaForSequenceClassification</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlm_roberta/modeling_xlm_roberta.py#L1139"}}),zt=new $({props:{name:"forward",anchor:"transformers.XLMRobertaForSequenceClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.XLMRobertaForSequenceClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.XLMRobertaForSequenceClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.XLMRobertaForSequenceClassification.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0,1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.
This parameter can only be used when the model is initialized with <code>type_vocab_size</code> parameter with value<blockquote>
<p>= 2. All the value in this tensor should be always &lt; type_vocab_size.</p>
</blockquote></li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.XLMRobertaForSequenceClassification.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.XLMRobertaForSequenceClassification.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.XLMRobertaForSequenceClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.XLMRobertaForSequenceClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the sequence classification/regression loss. Indices should be in <code>[0, ..., config.num_labels - 1]</code>. If <code>config.num_labels == 1</code> a regression loss is computed (Mean-Square loss), If
<code>config.num_labels &gt; 1</code> a classification loss is computed (Cross-Entropy).`,name:"labels"},{anchor:"transformers.XLMRobertaForSequenceClassification.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.XLMRobertaForSequenceClassification.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.XLMRobertaForSequenceClassification.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlm_roberta/modeling_xlm_roberta.py#L1151",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput"
>transformers.modeling_outputs.SequenceClassifierOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/xlm-roberta#transformers.XLMRobertaConfig"
>XLMRobertaConfig</a>) and inputs.</p>
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
`}}),$e=new he({props:{$$slots:{default:[yr]},$$scope:{ctx:w}}}),Re=new We({props:{anchor:"transformers.XLMRobertaForSequenceClassification.forward.example",$$slots:{default:[wr]},$$scope:{ctx:w}}}),Le=new We({props:{anchor:"transformers.XLMRobertaForSequenceClassification.forward.example-2",$$slots:{default:[vr]},$$scope:{ctx:w}}}),Ft=new Y({props:{title:"XLMRobertaForMultipleChoice",local:"transformers.XLMRobertaForMultipleChoice",headingTag:"h2"}}),Ut=new $({props:{name:"class transformers.XLMRobertaForMultipleChoice",anchor:"transformers.XLMRobertaForMultipleChoice",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.XLMRobertaForMultipleChoice.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/xlm-roberta#transformers.XLMRobertaForMultipleChoice">XLMRobertaForMultipleChoice</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlm_roberta/modeling_xlm_roberta.py#L1235"}}),It=new $({props:{name:"forward",anchor:"transformers.XLMRobertaForMultipleChoice.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.XLMRobertaForMultipleChoice.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, num_choices, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.XLMRobertaForMultipleChoice.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, num_choices, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0,1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.
This parameter can only be used when the model is initialized with <code>type_vocab_size</code> parameter with value<blockquote>
<p>= 2. All the value in this tensor should be always &lt; type_vocab_size.</p>
</blockquote></li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.XLMRobertaForMultipleChoice.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.XLMRobertaForMultipleChoice.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the multiple choice classification loss. Indices should be in <code>[0, ..., num_choices-1]</code> where <code>num_choices</code> is the size of the second dimension of the input tensors. (See
<code>input_ids</code> above)`,name:"labels"},{anchor:"transformers.XLMRobertaForMultipleChoice.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, num_choices, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.max_position_embeddings - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.XLMRobertaForMultipleChoice.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.XLMRobertaForMultipleChoice.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_choices, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.XLMRobertaForMultipleChoice.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.XLMRobertaForMultipleChoice.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.XLMRobertaForMultipleChoice.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlm_roberta/modeling_xlm_roberta.py#L1246",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.MultipleChoiceModelOutput"
>transformers.modeling_outputs.MultipleChoiceModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/xlm-roberta#transformers.XLMRobertaConfig"
>XLMRobertaConfig</a>) and inputs.</p>
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
`}}),xe=new he({props:{$$slots:{default:[Jr]},$$scope:{ctx:w}}}),Xe=new We({props:{anchor:"transformers.XLMRobertaForMultipleChoice.forward.example",$$slots:{default:[$r]},$$scope:{ctx:w}}}),qt=new Y({props:{title:"XLMRobertaForTokenClassification",local:"transformers.XLMRobertaForTokenClassification",headingTag:"h2"}}),Wt=new $({props:{name:"class transformers.XLMRobertaForTokenClassification",anchor:"transformers.XLMRobertaForTokenClassification",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.XLMRobertaForTokenClassification.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/xlm-roberta#transformers.XLMRobertaForTokenClassification">XLMRobertaForTokenClassification</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlm_roberta/modeling_xlm_roberta.py#L1342"}}),Zt=new $({props:{name:"forward",anchor:"transformers.XLMRobertaForTokenClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.XLMRobertaForTokenClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.XLMRobertaForTokenClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.XLMRobertaForTokenClassification.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0,1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.
This parameter can only be used when the model is initialized with <code>type_vocab_size</code> parameter with value<blockquote>
<p>= 2. All the value in this tensor should be always &lt; type_vocab_size.</p>
</blockquote></li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.XLMRobertaForTokenClassification.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.XLMRobertaForTokenClassification.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.XLMRobertaForTokenClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.XLMRobertaForTokenClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the token classification loss. Indices should be in <code>[0, ..., config.num_labels - 1]</code>.`,name:"labels"},{anchor:"transformers.XLMRobertaForTokenClassification.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.XLMRobertaForTokenClassification.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.XLMRobertaForTokenClassification.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlm_roberta/modeling_xlm_roberta.py#L1357",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput"
>transformers.modeling_outputs.TokenClassifierOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/xlm-roberta#transformers.XLMRobertaConfig"
>XLMRobertaConfig</a>) and inputs.</p>
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
`}}),je=new he({props:{$$slots:{default:[Rr]},$$scope:{ctx:w}}}),Ce=new We({props:{anchor:"transformers.XLMRobertaForTokenClassification.forward.example",$$slots:{default:[Lr]},$$scope:{ctx:w}}}),Bt=new Y({props:{title:"XLMRobertaForQuestionAnswering",local:"transformers.XLMRobertaForQuestionAnswering",headingTag:"h2"}}),Gt=new $({props:{name:"class transformers.XLMRobertaForQuestionAnswering",anchor:"transformers.XLMRobertaForQuestionAnswering",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.XLMRobertaForQuestionAnswering.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/xlm-roberta#transformers.XLMRobertaForQuestionAnswering">XLMRobertaForQuestionAnswering</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlm_roberta/modeling_xlm_roberta.py#L1447"}}),Nt=new $({props:{name:"forward",anchor:"transformers.XLMRobertaForQuestionAnswering.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"start_positions",val:": typing.Optional[torch.LongTensor] = None"},{name:"end_positions",val:": typing.Optional[torch.LongTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.XLMRobertaForQuestionAnswering.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.XLMRobertaForQuestionAnswering.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.XLMRobertaForQuestionAnswering.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0,1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.
This parameter can only be used when the model is initialized with <code>type_vocab_size</code> parameter with value<blockquote>
<p>= 2. All the value in this tensor should be always &lt; type_vocab_size.</p>
</blockquote></li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.XLMRobertaForQuestionAnswering.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.XLMRobertaForQuestionAnswering.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.XLMRobertaForQuestionAnswering.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.XLMRobertaForQuestionAnswering.forward.start_positions",description:`<strong>start_positions</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for position (index) of the start of the labelled span for computing the token classification loss.
Positions are clamped to the length of the sequence (<code>sequence_length</code>). Position outside of the sequence
are not taken into account for computing the loss.`,name:"start_positions"},{anchor:"transformers.XLMRobertaForQuestionAnswering.forward.end_positions",description:`<strong>end_positions</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for position (index) of the end of the labelled span for computing the token classification loss.
Positions are clamped to the length of the sequence (<code>sequence_length</code>). Position outside of the sequence
are not taken into account for computing the loss.`,name:"end_positions"},{anchor:"transformers.XLMRobertaForQuestionAnswering.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.XLMRobertaForQuestionAnswering.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.XLMRobertaForQuestionAnswering.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlm_roberta/modeling_xlm_roberta.py#L1458",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.QuestionAnsweringModelOutput"
>transformers.modeling_outputs.QuestionAnsweringModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/xlm-roberta#transformers.XLMRobertaConfig"
>XLMRobertaConfig</a>) and inputs.</p>
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
`}}),ze=new he({props:{$$slots:{default:[xr]},$$scope:{ctx:w}}}),Fe=new We({props:{anchor:"transformers.XLMRobertaForQuestionAnswering.forward.example",$$slots:{default:[Xr]},$$scope:{ctx:w}}}),Vt=new lr({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/xlm-roberta.md"}}),{c(){t=c("meta"),u=a(),n=c("p"),m=a(),y=c("p"),y.innerHTML=i,T=a(),R=c("div"),R.innerHTML=Io,Ze=a(),f(se.$$.fragment),Bo=a(),Be=c("p"),Be.innerHTML=Ks,Go=a(),Ge=c("p"),Ge.innerHTML=ea,No=a(),f(ue.$$.fragment),Vo=a(),Ne=c("p"),Ne.innerHTML=ta,Ho=a(),f(fe.$$.fragment),Eo=a(),Ve=c("p"),Ve.innerHTML=oa,So=a(),He=c("p"),He.innerHTML=na,Ao=a(),f(Ee.$$.fragment),Yo=a(),f(Se.$$.fragment),Qo=a(),Ae=c("ul"),Ae.innerHTML=sa,Po=a(),f(Ye.$$.fragment),Do=a(),Qe=c("p"),Qe.textContent=aa,Oo=a(),f(Pe.$$.fragment),Ko=a(),De=c("ul"),De.innerHTML=ra,en=a(),f(Oe.$$.fragment),tn=a(),Ke=c("ul"),Ke.innerHTML=ia,on=a(),f(et.$$.fragment),nn=a(),tt=c("ul"),tt.innerHTML=la,sn=a(),f(ot.$$.fragment),an=a(),nt=c("ul"),nt.innerHTML=da,rn=a(),f(st.$$.fragment),ln=a(),at=c("ul"),at.innerHTML=ca,dn=a(),rt=c("p"),rt.innerHTML=pa,cn=a(),it=c("ul"),it.innerHTML=ma,pn=a(),lt=c("p"),lt.textContent=ha,mn=a(),dt=c("ul"),dt.innerHTML=ua,hn=a(),f(ge.$$.fragment),un=a(),f(ct.$$.fragment),fn=a(),B=c("div"),f(pt.$$.fragment),qn=a(),At=c("p"),At.innerHTML=fa,Wn=a(),Yt=c("p"),Yt.innerHTML=ga,Zn=a(),f(be.$$.fragment),gn=a(),f(mt.$$.fragment),bn=a(),x=c("div"),f(ht.$$.fragment),Bn=a(),Qt=c("p"),Qt.innerHTML=ba,Gn=a(),Pt=c("p"),Pt.innerHTML=_a,Nn=a(),ae=c("div"),f(ut.$$.fragment),Vn=a(),Dt=c("p"),Dt.textContent=Ma,Hn=a(),Ot=c("ul"),Ot.innerHTML=ka,En=a(),_e=c("div"),f(ft.$$.fragment),Sn=a(),Kt=c("p"),Kt.innerHTML=Ta,An=a(),Me=c("div"),f(gt.$$.fragment),Yn=a(),eo=c("p"),eo.textContent=ya,Qn=a(),to=c("div"),f(bt.$$.fragment),_n=a(),f(_t.$$.fragment),Mn=a(),j=c("div"),f(Mt.$$.fragment),Pn=a(),oo=c("p"),oo.innerHTML=wa,Dn=a(),no=c("p"),no.innerHTML=va,On=a(),re=c("div"),f(kt.$$.fragment),Kn=a(),so=c("p"),so.textContent=Ja,es=a(),ao=c("ul"),ao.innerHTML=$a,ts=a(),ke=c("div"),f(Tt.$$.fragment),os=a(),ro=c("p"),ro.textContent=Ra,kn=a(),f(yt.$$.fragment),Tn=a(),C=c("div"),f(wt.$$.fragment),ns=a(),io=c("p"),io.textContent=La,ss=a(),lo=c("p"),lo.innerHTML=xa,as=a(),co=c("p"),co.innerHTML=Xa,rs=a(),ie=c("div"),f(vt.$$.fragment),is=a(),po=c("p"),po.innerHTML=ja,ls=a(),f(Te.$$.fragment),yn=a(),f(Jt.$$.fragment),wn=a(),z=c("div"),f($t.$$.fragment),ds=a(),mo=c("p"),mo.innerHTML=Ca,cs=a(),ho=c("p"),ho.innerHTML=za,ps=a(),uo=c("p"),uo.innerHTML=Fa,ms=a(),Q=c("div"),f(Rt.$$.fragment),hs=a(),fo=c("p"),fo.innerHTML=Ua,us=a(),f(ye.$$.fragment),fs=a(),f(we.$$.fragment),vn=a(),f(Lt.$$.fragment),Jn=a(),F=c("div"),f(xt.$$.fragment),gs=a(),go=c("p"),go.innerHTML=Ia,bs=a(),bo=c("p"),bo.innerHTML=qa,_s=a(),_o=c("p"),_o.innerHTML=Wa,Ms=a(),P=c("div"),f(Xt.$$.fragment),ks=a(),Mo=c("p"),Mo.innerHTML=Za,Ts=a(),f(ve.$$.fragment),ys=a(),f(Je.$$.fragment),$n=a(),f(jt.$$.fragment),Rn=a(),U=c("div"),f(Ct.$$.fragment),ws=a(),ko=c("p"),ko.textContent=Ba,vs=a(),To=c("p"),To.innerHTML=Ga,Js=a(),yo=c("p"),yo.innerHTML=Na,$s=a(),Z=c("div"),f(zt.$$.fragment),Rs=a(),wo=c("p"),wo.innerHTML=Va,Ls=a(),f($e.$$.fragment),xs=a(),f(Re.$$.fragment),Xs=a(),f(Le.$$.fragment),Ln=a(),f(Ft.$$.fragment),xn=a(),I=c("div"),f(Ut.$$.fragment),js=a(),vo=c("p"),vo.textContent=Ha,Cs=a(),Jo=c("p"),Jo.innerHTML=Ea,zs=a(),$o=c("p"),$o.innerHTML=Sa,Fs=a(),D=c("div"),f(It.$$.fragment),Us=a(),Ro=c("p"),Ro.innerHTML=Aa,Is=a(),f(xe.$$.fragment),qs=a(),f(Xe.$$.fragment),Xn=a(),f(qt.$$.fragment),jn=a(),q=c("div"),f(Wt.$$.fragment),Ws=a(),Lo=c("p"),Lo.textContent=Ya,Zs=a(),xo=c("p"),xo.innerHTML=Qa,Bs=a(),Xo=c("p"),Xo.innerHTML=Pa,Gs=a(),O=c("div"),f(Zt.$$.fragment),Ns=a(),jo=c("p"),jo.innerHTML=Da,Vs=a(),f(je.$$.fragment),Hs=a(),f(Ce.$$.fragment),Cn=a(),f(Bt.$$.fragment),zn=a(),W=c("div"),f(Gt.$$.fragment),Es=a(),Co=c("p"),Co.innerHTML=Oa,Ss=a(),zo=c("p"),zo.innerHTML=Ka,As=a(),Fo=c("p"),Fo.innerHTML=er,Ys=a(),K=c("div"),f(Nt.$$.fragment),Qs=a(),Uo=c("p"),Uo.innerHTML=tr,Ps=a(),f(ze.$$.fragment),Ds=a(),f(Fe.$$.fragment),Fn=a(),f(Vt.$$.fragment),Un=a(),qo=c("p"),this.h()},l(e){const o=rr("svelte-u9bgzb",document.head);t=p(o,"META",{name:!0,content:!0}),o.forEach(s),u=r(e),n=p(e,"P",{}),v(n).forEach(s),m=r(e),y=p(e,"P",{"data-svelte-h":!0}),h(y)!=="svelte-kpc6q4"&&(y.innerHTML=i),T=r(e),R=p(e,"DIV",{style:!0,"data-svelte-h":!0}),h(R)!=="svelte-ithiq1"&&(R.innerHTML=Io),Ze=r(e),g(se.$$.fragment,e),Bo=r(e),Be=p(e,"P",{"data-svelte-h":!0}),h(Be)!=="svelte-1kw26j5"&&(Be.innerHTML=Ks),Go=r(e),Ge=p(e,"P",{"data-svelte-h":!0}),h(Ge)!=="svelte-x0snbv"&&(Ge.innerHTML=ea),No=r(e),g(ue.$$.fragment,e),Vo=r(e),Ne=p(e,"P",{"data-svelte-h":!0}),h(Ne)!=="svelte-10lshn2"&&(Ne.innerHTML=ta),Ho=r(e),g(fe.$$.fragment,e),Eo=r(e),Ve=p(e,"P",{"data-svelte-h":!0}),h(Ve)!=="svelte-106h7za"&&(Ve.innerHTML=oa),So=r(e),He=p(e,"P",{"data-svelte-h":!0}),h(He)!=="svelte-ftl39v"&&(He.innerHTML=na),Ao=r(e),g(Ee.$$.fragment,e),Yo=r(e),g(Se.$$.fragment,e),Qo=r(e),Ae=p(e,"UL",{"data-svelte-h":!0}),h(Ae)!=="svelte-cyizke"&&(Ae.innerHTML=sa),Po=r(e),g(Ye.$$.fragment,e),Do=r(e),Qe=p(e,"P",{"data-svelte-h":!0}),h(Qe)!=="svelte-1ohr9zi"&&(Qe.textContent=aa),Oo=r(e),g(Pe.$$.fragment,e),Ko=r(e),De=p(e,"UL",{"data-svelte-h":!0}),h(De)!=="svelte-1mjp29w"&&(De.innerHTML=ra),en=r(e),g(Oe.$$.fragment,e),tn=r(e),Ke=p(e,"UL",{"data-svelte-h":!0}),h(Ke)!=="svelte-ix7xoi"&&(Ke.innerHTML=ia),on=r(e),g(et.$$.fragment,e),nn=r(e),tt=p(e,"UL",{"data-svelte-h":!0}),h(tt)!=="svelte-t4xz8h"&&(tt.innerHTML=la),sn=r(e),g(ot.$$.fragment,e),an=r(e),nt=p(e,"UL",{"data-svelte-h":!0}),h(nt)!=="svelte-1j5r39q"&&(nt.innerHTML=da),rn=r(e),g(st.$$.fragment,e),ln=r(e),at=p(e,"UL",{"data-svelte-h":!0}),h(at)!=="svelte-1u7bk0g"&&(at.innerHTML=ca),dn=r(e),rt=p(e,"P",{"data-svelte-h":!0}),h(rt)!=="svelte-cplu6u"&&(rt.innerHTML=pa),cn=r(e),it=p(e,"UL",{"data-svelte-h":!0}),h(it)!=="svelte-1cyvwmi"&&(it.innerHTML=ma),pn=r(e),lt=p(e,"P",{"data-svelte-h":!0}),h(lt)!=="svelte-lk14e4"&&(lt.textContent=ha),mn=r(e),dt=p(e,"UL",{"data-svelte-h":!0}),h(dt)!=="svelte-1eioy9o"&&(dt.innerHTML=ua),hn=r(e),g(ge.$$.fragment,e),un=r(e),g(ct.$$.fragment,e),fn=r(e),B=p(e,"DIV",{class:!0});var te=v(B);g(pt.$$.fragment,te),qn=r(te),At=p(te,"P",{"data-svelte-h":!0}),h(At)!=="svelte-l4ovip"&&(At.innerHTML=fa),Wn=r(te),Yt=p(te,"P",{"data-svelte-h":!0}),h(Yt)!=="svelte-1ek1ss9"&&(Yt.innerHTML=ga),Zn=r(te),g(be.$$.fragment,te),te.forEach(s),gn=r(e),g(mt.$$.fragment,e),bn=r(e),x=p(e,"DIV",{class:!0});var X=v(x);g(ht.$$.fragment,X),Bn=r(X),Qt=p(X,"P",{"data-svelte-h":!0}),h(Qt)!=="svelte-19vr0qz"&&(Qt.innerHTML=ba),Gn=r(X),Pt=p(X,"P",{"data-svelte-h":!0}),h(Pt)!=="svelte-ntrhio"&&(Pt.innerHTML=_a),Nn=r(X),ae=p(X,"DIV",{class:!0});var ce=v(ae);g(ut.$$.fragment,ce),Vn=r(ce),Dt=p(ce,"P",{"data-svelte-h":!0}),h(Dt)!=="svelte-1ooxl9e"&&(Dt.textContent=Ma),Hn=r(ce),Ot=p(ce,"UL",{"data-svelte-h":!0}),h(Ot)!=="svelte-rq8uot"&&(Ot.innerHTML=ka),ce.forEach(s),En=r(X),_e=p(X,"DIV",{class:!0});var Ht=v(_e);g(ft.$$.fragment,Ht),Sn=r(Ht),Kt=p(Ht,"P",{"data-svelte-h":!0}),h(Kt)!=="svelte-1f4f5kp"&&(Kt.innerHTML=Ta),Ht.forEach(s),An=r(X),Me=p(X,"DIV",{class:!0});var Et=v(Me);g(gt.$$.fragment,Et),Yn=r(Et),eo=p(Et,"P",{"data-svelte-h":!0}),h(eo)!=="svelte-bub0ru"&&(eo.textContent=ya),Et.forEach(s),Qn=r(X),to=p(X,"DIV",{class:!0});var Wo=v(to);g(bt.$$.fragment,Wo),Wo.forEach(s),X.forEach(s),_n=r(e),g(_t.$$.fragment,e),Mn=r(e),j=p(e,"DIV",{class:!0});var G=v(j);g(Mt.$$.fragment,G),Pn=r(G),oo=p(G,"P",{"data-svelte-h":!0}),h(oo)!=="svelte-c948km"&&(oo.innerHTML=wa),Dn=r(G),no=p(G,"P",{"data-svelte-h":!0}),h(no)!=="svelte-gxzj9w"&&(no.innerHTML=va),On=r(G),re=p(G,"DIV",{class:!0});var pe=v(re);g(kt.$$.fragment,pe),Kn=r(pe),so=p(pe,"P",{"data-svelte-h":!0}),h(so)!=="svelte-1ooxl9e"&&(so.textContent=Ja),es=r(pe),ao=p(pe,"UL",{"data-svelte-h":!0}),h(ao)!=="svelte-rq8uot"&&(ao.innerHTML=$a),pe.forEach(s),ts=r(G),ke=p(G,"DIV",{class:!0});var St=v(ke);g(Tt.$$.fragment,St),os=r(St),ro=p(St,"P",{"data-svelte-h":!0}),h(ro)!=="svelte-bub0ru"&&(ro.textContent=Ra),St.forEach(s),G.forEach(s),kn=r(e),g(yt.$$.fragment,e),Tn=r(e),C=p(e,"DIV",{class:!0});var N=v(C);g(wt.$$.fragment,N),ns=r(N),io=p(N,"P",{"data-svelte-h":!0}),h(io)!=="svelte-dtsn0e"&&(io.textContent=La),ss=r(N),lo=p(N,"P",{"data-svelte-h":!0}),h(lo)!=="svelte-q52n56"&&(lo.innerHTML=xa),as=r(N),co=p(N,"P",{"data-svelte-h":!0}),h(co)!=="svelte-hswkmf"&&(co.innerHTML=Xa),rs=r(N),ie=p(N,"DIV",{class:!0});var me=v(ie);g(vt.$$.fragment,me),is=r(me),po=p(me,"P",{"data-svelte-h":!0}),h(po)!=="svelte-1n2agz4"&&(po.innerHTML=ja),ls=r(me),g(Te.$$.fragment,me),me.forEach(s),N.forEach(s),yn=r(e),g(Jt.$$.fragment,e),wn=r(e),z=p(e,"DIV",{class:!0});var V=v(z);g($t.$$.fragment,V),ds=r(V),mo=p(V,"P",{"data-svelte-h":!0}),h(mo)!=="svelte-15apchw"&&(mo.innerHTML=Ca),cs=r(V),ho=p(V,"P",{"data-svelte-h":!0}),h(ho)!=="svelte-q52n56"&&(ho.innerHTML=za),ps=r(V),uo=p(V,"P",{"data-svelte-h":!0}),h(uo)!=="svelte-hswkmf"&&(uo.innerHTML=Fa),ms=r(V),Q=p(V,"DIV",{class:!0});var oe=v(Q);g(Rt.$$.fragment,oe),hs=r(oe),fo=p(oe,"P",{"data-svelte-h":!0}),h(fo)!=="svelte-qbarhw"&&(fo.innerHTML=Ua),us=r(oe),g(ye.$$.fragment,oe),fs=r(oe),g(we.$$.fragment,oe),oe.forEach(s),V.forEach(s),vn=r(e),g(Lt.$$.fragment,e),Jn=r(e),F=p(e,"DIV",{class:!0});var H=v(F);g(xt.$$.fragment,H),gs=r(H),go=p(H,"P",{"data-svelte-h":!0}),h(go)!=="svelte-1s9rwzm"&&(go.innerHTML=Ia),bs=r(H),bo=p(H,"P",{"data-svelte-h":!0}),h(bo)!=="svelte-q52n56"&&(bo.innerHTML=qa),_s=r(H),_o=p(H,"P",{"data-svelte-h":!0}),h(_o)!=="svelte-hswkmf"&&(_o.innerHTML=Wa),Ms=r(H),P=p(H,"DIV",{class:!0});var ne=v(P);g(Xt.$$.fragment,ne),ks=r(ne),Mo=p(ne,"P",{"data-svelte-h":!0}),h(Mo)!=="svelte-1j9rbys"&&(Mo.innerHTML=Za),Ts=r(ne),g(ve.$$.fragment,ne),ys=r(ne),g(Je.$$.fragment,ne),ne.forEach(s),H.forEach(s),$n=r(e),g(jt.$$.fragment,e),Rn=r(e),U=p(e,"DIV",{class:!0});var E=v(U);g(Ct.$$.fragment,E),ws=r(E),ko=p(E,"P",{"data-svelte-h":!0}),h(ko)!=="svelte-nc5ddr"&&(ko.textContent=Ba),vs=r(E),To=p(E,"P",{"data-svelte-h":!0}),h(To)!=="svelte-q52n56"&&(To.innerHTML=Ga),Js=r(E),yo=p(E,"P",{"data-svelte-h":!0}),h(yo)!=="svelte-hswkmf"&&(yo.innerHTML=Na),$s=r(E),Z=p(E,"DIV",{class:!0});var S=v(Z);g(zt.$$.fragment,S),Rs=r(S),wo=p(S,"P",{"data-svelte-h":!0}),h(wo)!=="svelte-1f1bz5w"&&(wo.innerHTML=Va),Ls=r(S),g($e.$$.fragment,S),xs=r(S),g(Re.$$.fragment,S),Xs=r(S),g(Le.$$.fragment,S),S.forEach(s),E.forEach(s),Ln=r(e),g(Ft.$$.fragment,e),xn=r(e),I=p(e,"DIV",{class:!0});var A=v(I);g(Ut.$$.fragment,A),js=r(A),vo=p(A,"P",{"data-svelte-h":!0}),h(vo)!=="svelte-1ht0ee9"&&(vo.textContent=Ha),Cs=r(A),Jo=p(A,"P",{"data-svelte-h":!0}),h(Jo)!=="svelte-q52n56"&&(Jo.innerHTML=Ea),zs=r(A),$o=p(A,"P",{"data-svelte-h":!0}),h($o)!=="svelte-hswkmf"&&($o.innerHTML=Sa),Fs=r(A),D=p(A,"DIV",{class:!0});var Ue=v(D);g(It.$$.fragment,Ue),Us=r(Ue),Ro=p(Ue,"P",{"data-svelte-h":!0}),h(Ro)!=="svelte-wujkqo"&&(Ro.innerHTML=Aa),Is=r(Ue),g(xe.$$.fragment,Ue),qs=r(Ue),g(Xe.$$.fragment,Ue),Ue.forEach(s),A.forEach(s),Xn=r(e),g(qt.$$.fragment,e),jn=r(e),q=p(e,"DIV",{class:!0});var le=v(q);g(Wt.$$.fragment,le),Ws=r(le),Lo=p(le,"P",{"data-svelte-h":!0}),h(Lo)!=="svelte-1yhdu46"&&(Lo.textContent=Ya),Zs=r(le),xo=p(le,"P",{"data-svelte-h":!0}),h(xo)!=="svelte-q52n56"&&(xo.innerHTML=Qa),Bs=r(le),Xo=p(le,"P",{"data-svelte-h":!0}),h(Xo)!=="svelte-hswkmf"&&(Xo.innerHTML=Pa),Gs=r(le),O=p(le,"DIV",{class:!0});var Ie=v(O);g(Zt.$$.fragment,Ie),Ns=r(Ie),jo=p(Ie,"P",{"data-svelte-h":!0}),h(jo)!=="svelte-fs2vmo"&&(jo.innerHTML=Da),Vs=r(Ie),g(je.$$.fragment,Ie),Hs=r(Ie),g(Ce.$$.fragment,Ie),Ie.forEach(s),le.forEach(s),Cn=r(e),g(Bt.$$.fragment,e),zn=r(e),W=p(e,"DIV",{class:!0});var de=v(W);g(Gt.$$.fragment,de),Es=r(de),Co=p(de,"P",{"data-svelte-h":!0}),h(Co)!=="svelte-1txzxod"&&(Co.innerHTML=Oa),Ss=r(de),zo=p(de,"P",{"data-svelte-h":!0}),h(zo)!=="svelte-q52n56"&&(zo.innerHTML=Ka),As=r(de),Fo=p(de,"P",{"data-svelte-h":!0}),h(Fo)!=="svelte-hswkmf"&&(Fo.innerHTML=er),Ys=r(de),K=p(de,"DIV",{class:!0});var qe=v(K);g(Nt.$$.fragment,qe),Qs=r(qe),Uo=p(qe,"P",{"data-svelte-h":!0}),h(Uo)!=="svelte-rlon3u"&&(Uo.innerHTML=tr),Ps=r(qe),g(ze.$$.fragment,qe),Ds=r(qe),g(Fe.$$.fragment,qe),qe.forEach(s),de.forEach(s),Fn=r(e),g(Vt.$$.fragment,e),Un=r(e),qo=p(e,"P",{}),v(qo).forEach(s),this.h()},h(){J(t,"name","hf:doc:metadata"),J(t,"content",Cr),ir(R,"float","right"),J(B,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(ae,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(_e,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(Me,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(to,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(x,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(re,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(ke,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(j,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(ie,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(C,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(Q,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(z,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(P,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(F,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(Z,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(U,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(D,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(I,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(O,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(q,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(K,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(W,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,o){l(document.head,t),d(e,u,o),d(e,n,o),d(e,m,o),d(e,y,o),d(e,T,o),d(e,R,o),d(e,Ze,o),b(se,e,o),d(e,Bo,o),d(e,Be,o),d(e,Go,o),d(e,Ge,o),d(e,No,o),b(ue,e,o),d(e,Vo,o),d(e,Ne,o),d(e,Ho,o),b(fe,e,o),d(e,Eo,o),d(e,Ve,o),d(e,So,o),d(e,He,o),d(e,Ao,o),b(Ee,e,o),d(e,Yo,o),b(Se,e,o),d(e,Qo,o),d(e,Ae,o),d(e,Po,o),b(Ye,e,o),d(e,Do,o),d(e,Qe,o),d(e,Oo,o),b(Pe,e,o),d(e,Ko,o),d(e,De,o),d(e,en,o),b(Oe,e,o),d(e,tn,o),d(e,Ke,o),d(e,on,o),b(et,e,o),d(e,nn,o),d(e,tt,o),d(e,sn,o),b(ot,e,o),d(e,an,o),d(e,nt,o),d(e,rn,o),b(st,e,o),d(e,ln,o),d(e,at,o),d(e,dn,o),d(e,rt,o),d(e,cn,o),d(e,it,o),d(e,pn,o),d(e,lt,o),d(e,mn,o),d(e,dt,o),d(e,hn,o),b(ge,e,o),d(e,un,o),b(ct,e,o),d(e,fn,o),d(e,B,o),b(pt,B,null),l(B,qn),l(B,At),l(B,Wn),l(B,Yt),l(B,Zn),b(be,B,null),d(e,gn,o),b(mt,e,o),d(e,bn,o),d(e,x,o),b(ht,x,null),l(x,Bn),l(x,Qt),l(x,Gn),l(x,Pt),l(x,Nn),l(x,ae),b(ut,ae,null),l(ae,Vn),l(ae,Dt),l(ae,Hn),l(ae,Ot),l(x,En),l(x,_e),b(ft,_e,null),l(_e,Sn),l(_e,Kt),l(x,An),l(x,Me),b(gt,Me,null),l(Me,Yn),l(Me,eo),l(x,Qn),l(x,to),b(bt,to,null),d(e,_n,o),b(_t,e,o),d(e,Mn,o),d(e,j,o),b(Mt,j,null),l(j,Pn),l(j,oo),l(j,Dn),l(j,no),l(j,On),l(j,re),b(kt,re,null),l(re,Kn),l(re,so),l(re,es),l(re,ao),l(j,ts),l(j,ke),b(Tt,ke,null),l(ke,os),l(ke,ro),d(e,kn,o),b(yt,e,o),d(e,Tn,o),d(e,C,o),b(wt,C,null),l(C,ns),l(C,io),l(C,ss),l(C,lo),l(C,as),l(C,co),l(C,rs),l(C,ie),b(vt,ie,null),l(ie,is),l(ie,po),l(ie,ls),b(Te,ie,null),d(e,yn,o),b(Jt,e,o),d(e,wn,o),d(e,z,o),b($t,z,null),l(z,ds),l(z,mo),l(z,cs),l(z,ho),l(z,ps),l(z,uo),l(z,ms),l(z,Q),b(Rt,Q,null),l(Q,hs),l(Q,fo),l(Q,us),b(ye,Q,null),l(Q,fs),b(we,Q,null),d(e,vn,o),b(Lt,e,o),d(e,Jn,o),d(e,F,o),b(xt,F,null),l(F,gs),l(F,go),l(F,bs),l(F,bo),l(F,_s),l(F,_o),l(F,Ms),l(F,P),b(Xt,P,null),l(P,ks),l(P,Mo),l(P,Ts),b(ve,P,null),l(P,ys),b(Je,P,null),d(e,$n,o),b(jt,e,o),d(e,Rn,o),d(e,U,o),b(Ct,U,null),l(U,ws),l(U,ko),l(U,vs),l(U,To),l(U,Js),l(U,yo),l(U,$s),l(U,Z),b(zt,Z,null),l(Z,Rs),l(Z,wo),l(Z,Ls),b($e,Z,null),l(Z,xs),b(Re,Z,null),l(Z,Xs),b(Le,Z,null),d(e,Ln,o),b(Ft,e,o),d(e,xn,o),d(e,I,o),b(Ut,I,null),l(I,js),l(I,vo),l(I,Cs),l(I,Jo),l(I,zs),l(I,$o),l(I,Fs),l(I,D),b(It,D,null),l(D,Us),l(D,Ro),l(D,Is),b(xe,D,null),l(D,qs),b(Xe,D,null),d(e,Xn,o),b(qt,e,o),d(e,jn,o),d(e,q,o),b(Wt,q,null),l(q,Ws),l(q,Lo),l(q,Zs),l(q,xo),l(q,Bs),l(q,Xo),l(q,Gs),l(q,O),b(Zt,O,null),l(O,Ns),l(O,jo),l(O,Vs),b(je,O,null),l(O,Hs),b(Ce,O,null),d(e,Cn,o),b(Bt,e,o),d(e,zn,o),d(e,W,o),b(Gt,W,null),l(W,Es),l(W,Co),l(W,Ss),l(W,zo),l(W,As),l(W,Fo),l(W,Ys),l(W,K),b(Nt,K,null),l(K,Qs),l(K,Uo),l(K,Ps),b(ze,K,null),l(K,Ds),b(Fe,K,null),d(e,Fn,o),b(Vt,e,o),d(e,Un,o),d(e,qo,o),In=!0},p(e,[o]){const te={};o&2&&(te.$$scope={dirty:o,ctx:e}),ue.$set(te);const X={};o&2&&(X.$$scope={dirty:o,ctx:e}),fe.$set(X);const ce={};o&2&&(ce.$$scope={dirty:o,ctx:e}),ge.$set(ce);const Ht={};o&2&&(Ht.$$scope={dirty:o,ctx:e}),be.$set(Ht);const Et={};o&2&&(Et.$$scope={dirty:o,ctx:e}),Te.$set(Et);const Wo={};o&2&&(Wo.$$scope={dirty:o,ctx:e}),ye.$set(Wo);const G={};o&2&&(G.$$scope={dirty:o,ctx:e}),we.$set(G);const pe={};o&2&&(pe.$$scope={dirty:o,ctx:e}),ve.$set(pe);const St={};o&2&&(St.$$scope={dirty:o,ctx:e}),Je.$set(St);const N={};o&2&&(N.$$scope={dirty:o,ctx:e}),$e.$set(N);const me={};o&2&&(me.$$scope={dirty:o,ctx:e}),Re.$set(me);const V={};o&2&&(V.$$scope={dirty:o,ctx:e}),Le.$set(V);const oe={};o&2&&(oe.$$scope={dirty:o,ctx:e}),xe.$set(oe);const H={};o&2&&(H.$$scope={dirty:o,ctx:e}),Xe.$set(H);const ne={};o&2&&(ne.$$scope={dirty:o,ctx:e}),je.$set(ne);const E={};o&2&&(E.$$scope={dirty:o,ctx:e}),Ce.$set(E);const S={};o&2&&(S.$$scope={dirty:o,ctx:e}),ze.$set(S);const A={};o&2&&(A.$$scope={dirty:o,ctx:e}),Fe.$set(A)},i(e){In||(_(se.$$.fragment,e),_(ue.$$.fragment,e),_(fe.$$.fragment,e),_(Ee.$$.fragment,e),_(Se.$$.fragment,e),_(Ye.$$.fragment,e),_(Pe.$$.fragment,e),_(Oe.$$.fragment,e),_(et.$$.fragment,e),_(ot.$$.fragment,e),_(st.$$.fragment,e),_(ge.$$.fragment,e),_(ct.$$.fragment,e),_(pt.$$.fragment,e),_(be.$$.fragment,e),_(mt.$$.fragment,e),_(ht.$$.fragment,e),_(ut.$$.fragment,e),_(ft.$$.fragment,e),_(gt.$$.fragment,e),_(bt.$$.fragment,e),_(_t.$$.fragment,e),_(Mt.$$.fragment,e),_(kt.$$.fragment,e),_(Tt.$$.fragment,e),_(yt.$$.fragment,e),_(wt.$$.fragment,e),_(vt.$$.fragment,e),_(Te.$$.fragment,e),_(Jt.$$.fragment,e),_($t.$$.fragment,e),_(Rt.$$.fragment,e),_(ye.$$.fragment,e),_(we.$$.fragment,e),_(Lt.$$.fragment,e),_(xt.$$.fragment,e),_(Xt.$$.fragment,e),_(ve.$$.fragment,e),_(Je.$$.fragment,e),_(jt.$$.fragment,e),_(Ct.$$.fragment,e),_(zt.$$.fragment,e),_($e.$$.fragment,e),_(Re.$$.fragment,e),_(Le.$$.fragment,e),_(Ft.$$.fragment,e),_(Ut.$$.fragment,e),_(It.$$.fragment,e),_(xe.$$.fragment,e),_(Xe.$$.fragment,e),_(qt.$$.fragment,e),_(Wt.$$.fragment,e),_(Zt.$$.fragment,e),_(je.$$.fragment,e),_(Ce.$$.fragment,e),_(Bt.$$.fragment,e),_(Gt.$$.fragment,e),_(Nt.$$.fragment,e),_(ze.$$.fragment,e),_(Fe.$$.fragment,e),_(Vt.$$.fragment,e),In=!0)},o(e){M(se.$$.fragment,e),M(ue.$$.fragment,e),M(fe.$$.fragment,e),M(Ee.$$.fragment,e),M(Se.$$.fragment,e),M(Ye.$$.fragment,e),M(Pe.$$.fragment,e),M(Oe.$$.fragment,e),M(et.$$.fragment,e),M(ot.$$.fragment,e),M(st.$$.fragment,e),M(ge.$$.fragment,e),M(ct.$$.fragment,e),M(pt.$$.fragment,e),M(be.$$.fragment,e),M(mt.$$.fragment,e),M(ht.$$.fragment,e),M(ut.$$.fragment,e),M(ft.$$.fragment,e),M(gt.$$.fragment,e),M(bt.$$.fragment,e),M(_t.$$.fragment,e),M(Mt.$$.fragment,e),M(kt.$$.fragment,e),M(Tt.$$.fragment,e),M(yt.$$.fragment,e),M(wt.$$.fragment,e),M(vt.$$.fragment,e),M(Te.$$.fragment,e),M(Jt.$$.fragment,e),M($t.$$.fragment,e),M(Rt.$$.fragment,e),M(ye.$$.fragment,e),M(we.$$.fragment,e),M(Lt.$$.fragment,e),M(xt.$$.fragment,e),M(Xt.$$.fragment,e),M(ve.$$.fragment,e),M(Je.$$.fragment,e),M(jt.$$.fragment,e),M(Ct.$$.fragment,e),M(zt.$$.fragment,e),M($e.$$.fragment,e),M(Re.$$.fragment,e),M(Le.$$.fragment,e),M(Ft.$$.fragment,e),M(Ut.$$.fragment,e),M(It.$$.fragment,e),M(xe.$$.fragment,e),M(Xe.$$.fragment,e),M(qt.$$.fragment,e),M(Wt.$$.fragment,e),M(Zt.$$.fragment,e),M(je.$$.fragment,e),M(Ce.$$.fragment,e),M(Bt.$$.fragment,e),M(Gt.$$.fragment,e),M(Nt.$$.fragment,e),M(ze.$$.fragment,e),M(Fe.$$.fragment,e),M(Vt.$$.fragment,e),In=!1},d(e){e&&(s(u),s(n),s(m),s(y),s(T),s(R),s(Ze),s(Bo),s(Be),s(Go),s(Ge),s(No),s(Vo),s(Ne),s(Ho),s(Eo),s(Ve),s(So),s(He),s(Ao),s(Yo),s(Qo),s(Ae),s(Po),s(Do),s(Qe),s(Oo),s(Ko),s(De),s(en),s(tn),s(Ke),s(on),s(nn),s(tt),s(sn),s(an),s(nt),s(rn),s(ln),s(at),s(dn),s(rt),s(cn),s(it),s(pn),s(lt),s(mn),s(dt),s(hn),s(un),s(fn),s(B),s(gn),s(bn),s(x),s(_n),s(Mn),s(j),s(kn),s(Tn),s(C),s(yn),s(wn),s(z),s(vn),s(Jn),s(F),s($n),s(Rn),s(U),s(Ln),s(xn),s(I),s(Xn),s(jn),s(q),s(Cn),s(zn),s(W),s(Fn),s(Un),s(qo)),s(t),k(se,e),k(ue,e),k(fe,e),k(Ee,e),k(Se,e),k(Ye,e),k(Pe,e),k(Oe,e),k(et,e),k(ot,e),k(st,e),k(ge,e),k(ct,e),k(pt),k(be),k(mt,e),k(ht),k(ut),k(ft),k(gt),k(bt),k(_t,e),k(Mt),k(kt),k(Tt),k(yt,e),k(wt),k(vt),k(Te),k(Jt,e),k($t),k(Rt),k(ye),k(we),k(Lt,e),k(xt),k(Xt),k(ve),k(Je),k(jt,e),k(Ct),k(zt),k($e),k(Re),k(Le),k(Ft,e),k(Ut),k(It),k(xe),k(Xe),k(qt,e),k(Wt),k(Zt),k(je),k(Ce),k(Bt,e),k(Gt),k(Nt),k(ze),k(Fe),k(Vt,e)}}}const Cr='{"title":"XLM-RoBERTa","local":"xlm-roberta","sections":[{"title":"Notes","local":"notes","sections":[],"depth":2},{"title":"Resources","local":"resources","sections":[],"depth":2},{"title":"XLMRobertaConfig","local":"transformers.XLMRobertaConfig","sections":[],"depth":2},{"title":"XLMRobertaTokenizer","local":"transformers.XLMRobertaTokenizer","sections":[],"depth":2},{"title":"XLMRobertaTokenizerFast","local":"transformers.XLMRobertaTokenizerFast","sections":[],"depth":2},{"title":"XLMRobertaModel","local":"transformers.XLMRobertaModel","sections":[],"depth":2},{"title":"XLMRobertaForCausalLM","local":"transformers.XLMRobertaForCausalLM","sections":[],"depth":2},{"title":"XLMRobertaForMaskedLM","local":"transformers.XLMRobertaForMaskedLM","sections":[],"depth":2},{"title":"XLMRobertaForSequenceClassification","local":"transformers.XLMRobertaForSequenceClassification","sections":[],"depth":2},{"title":"XLMRobertaForMultipleChoice","local":"transformers.XLMRobertaForMultipleChoice","sections":[],"depth":2},{"title":"XLMRobertaForTokenClassification","local":"transformers.XLMRobertaForTokenClassification","sections":[],"depth":2},{"title":"XLMRobertaForQuestionAnswering","local":"transformers.XLMRobertaForQuestionAnswering","sections":[],"depth":2}],"depth":1}';function zr(w){return nr(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class Vr extends sr{constructor(t){super(),ar(this,t,zr,jr,or,{})}}export{Vr as component};
