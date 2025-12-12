import{s as Dn,o as Kn,n as L}from"../chunks/scheduler.18a86fab.js";import{S as es,i as ts,g as m,s as l,r as f,A as os,h,f as r,c as i,j as v,x as w,u as g,k as J,l as ns,y as d,a as c,v as M,d as b,t as _,w as y}from"../chunks/index.98837b22.js";import{T as Le}from"../chunks/Tip.77304350.js";import{D as z}from"../chunks/Docstring.a1ef7999.js";import{C as O}from"../chunks/CodeBlock.8d0c2e8a.js";import{E as ve}from"../chunks/ExampleCodeBlock.8c3ee1f9.js";import{H as re,E as ss}from"../chunks/getInferenceSnippets.06c2775f.js";import{H as as,a as hn}from"../chunks/HfOption.6641485e.js";function rs(k){let t,p="Click on the XLM-RoBERTa-XL models in the right sidebar for more examples of how to apply XLM-RoBERTa-XL to different cross-lingual tasks like classification, translation, and question answering.";return{c(){t=m("p"),t.textContent=p},l(o){t=h(o,"P",{"data-svelte-h":!0}),w(t)!=="svelte-jine3d"&&(t.textContent=p)},m(o,a){c(o,t,a)},p:L,d(o){o&&r(t)}}}function ls(k){let t,p;return t=new O({props:{code:"aW1wb3J0JTIwdG9yY2glMjAlMjAlMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwcGlwZWxpbmUlMjAlMjAlMEElMEFwaXBlbGluZSUyMCUzRCUyMHBpcGVsaW5lKCUyMCUyMCUwQSUyMCUyMCUyMCUyMHRhc2slM0QlMjJmaWxsLW1hc2slMjIlMkMlMjAlMjAlMEElMjAlMjAlMjAlMjBtb2RlbCUzRCUyMmZhY2Vib29rJTJGeGxtLXJvYmVydGEteGwlMjIlMkMlMjAlMjAlMEElMjAlMjAlMjAlMjBkdHlwZSUzRHRvcmNoLmZsb2F0MTYlMkMlMjAlMjAlMEElMjAlMjAlMjAlMjBkZXZpY2UlM0QwJTIwJTIwJTBBKSUyMCUyMCUwQXBpcGVsaW5lKCUyMkJvbmpvdXIlMkMlMjBqZSUyMHN1aXMlMjB1biUyMG1vZCVDMyVBOGxlJTIwJTNDbWFzayUzRS4lMjIpJTIwJTIw",highlighted:`<span class="hljs-keyword">import</span> torch  
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> pipeline  

pipeline = pipeline(  
    task=<span class="hljs-string">&quot;fill-mask&quot;</span>,  
    model=<span class="hljs-string">&quot;facebook/xlm-roberta-xl&quot;</span>,  
    dtype=torch.float16,  
    device=<span class="hljs-number">0</span>  
)  
pipeline(<span class="hljs-string">&quot;Bonjour, je suis un modèle &lt;mask&gt;.&quot;</span>)  `,wrap:!1}}),{c(){f(t.$$.fragment)},l(o){g(t.$$.fragment,o)},m(o,a){M(t,o,a),p=!0},p:L,i(o){p||(b(t.$$.fragment,o),p=!0)},o(o){_(t.$$.fragment,o),p=!1},d(o){y(t,o)}}}function is(k){let t,p;return t=new O({props:{code:"aW1wb3J0JTIwdG9yY2glMjAlMjAlMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b01vZGVsRm9yTWFza2VkTE0lMkMlMjBBdXRvVG9rZW5pemVyJTIwJTIwJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIwJTIwJTBBJTIwJTIwJTIwJTIwJTIyZmFjZWJvb2slMkZ4bG0tcm9iZXJ0YS14bCUyMiUyQyUyMCUyMCUwQSklMjAlMjAlMEFtb2RlbCUyMCUzRCUyMEF1dG9Nb2RlbEZvck1hc2tlZExNLmZyb21fcHJldHJhaW5lZCglMjAlMjAlMEElMjAlMjAlMjAlMjAlMjJmYWNlYm9vayUyRnhsbS1yb2JlcnRhLXhsJTIyJTJDJTIwJTIwJTBBJTIwJTIwJTIwJTIwZHR5cGUlM0R0b3JjaC5mbG9hdDE2JTJDJTIwJTIwJTBBJTIwJTIwJTIwJTIwZGV2aWNlX21hcCUzRCUyMmF1dG8lMjIlMkMlMjAlMjAlMEElMjAlMjAlMjAlMjBhdHRuX2ltcGxlbWVudGF0aW9uJTNEJTIyc2RwYSUyMiUyMCUyMCUwQSklMjAlMjAlMEFpbnB1dHMlMjAlM0QlMjB0b2tlbml6ZXIoJTIyQm9uam91ciUyQyUyMGplJTIwc3VpcyUyMHVuJTIwbW9kJUMzJUE4bGUlMjAlM0NtYXNrJTNFLiUyMiUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpLnRvKG1vZGVsLmRldmljZSklMjAlMjAlMEElMEF3aXRoJTIwdG9yY2gubm9fZ3JhZCgpJTNBJTIwJTIwJTBBJTIwJTIwJTIwJTIwb3V0cHV0cyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzKSUyMCUyMCUwQSUyMCUyMCUyMCUyMHByZWRpY3Rpb25zJTIwJTNEJTIwb3V0cHV0cy5sb2dpdHMlMjAlMjAlMEElMEFtYXNrZWRfaW5kZXglMjAlM0QlMjB0b3JjaC53aGVyZShpbnB1dHMlNUInaW5wdXRfaWRzJyU1RCUyMCUzRCUzRCUyMHRva2VuaXplci5tYXNrX3Rva2VuX2lkKSU1QjElNUQlMjAlMjAlMEFwcmVkaWN0ZWRfdG9rZW5faWQlMjAlM0QlMjBwcmVkaWN0aW9ucyU1QjAlMkMlMjBtYXNrZWRfaW5kZXglNUQuYXJnbWF4KGRpbSUzRC0xKSUyMCUyMCUwQXByZWRpY3RlZF90b2tlbiUyMCUzRCUyMHRva2VuaXplci5kZWNvZGUocHJlZGljdGVkX3Rva2VuX2lkKSUyMCUyMCUwQSUwQXByaW50KGYlMjJUaGUlMjBwcmVkaWN0ZWQlMjB0b2tlbiUyMGlzJTNBJTIwJTdCcHJlZGljdGVkX3Rva2VuJTdEJTIyKQ==",highlighted:`<span class="hljs-keyword">import</span> torch  
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoModelForMaskedLM, AutoTokenizer  

tokenizer = AutoTokenizer.from_pretrained(  
    <span class="hljs-string">&quot;facebook/xlm-roberta-xl&quot;</span>,  
)  
model = AutoModelForMaskedLM.from_pretrained(  
    <span class="hljs-string">&quot;facebook/xlm-roberta-xl&quot;</span>,  
    dtype=torch.float16,  
    device_map=<span class="hljs-string">&quot;auto&quot;</span>,  
    attn_implementation=<span class="hljs-string">&quot;sdpa&quot;</span>  
)  
inputs = tokenizer(<span class="hljs-string">&quot;Bonjour, je suis un modèle &lt;mask&gt;.&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>).to(model.device)  

<span class="hljs-keyword">with</span> torch.no_grad():  
    outputs = model(**inputs)  
    predictions = outputs.logits  

masked_index = torch.where(inputs[<span class="hljs-string">&#x27;input_ids&#x27;</span>] == tokenizer.mask_token_id)[<span class="hljs-number">1</span>]  
predicted_token_id = predictions[<span class="hljs-number">0</span>, masked_index].argmax(dim=-<span class="hljs-number">1</span>)  
predicted_token = tokenizer.decode(predicted_token_id)  

<span class="hljs-built_in">print</span>(<span class="hljs-string">f&quot;The predicted token is: <span class="hljs-subst">{predicted_token}</span>&quot;</span>)`,wrap:!1}}),{c(){f(t.$$.fragment)},l(o){g(t.$$.fragment,o)},m(o,a){M(t,o,a),p=!0},p:L,i(o){p||(b(t.$$.fragment,o),p=!0)},o(o){_(t.$$.fragment,o),p=!1},d(o){y(t,o)}}}function ds(k){let t,p;return t=new O({props:{code:"ZWNobyUyMC1lJTIwJTIyUGxhbnRzJTIwY3JlYXRlJTIwJTNDbWFzayUzRSUyMHRocm91Z2glMjBhJTIwcHJvY2VzcyUyMGtub3duJTIwYXMlMjBwaG90b3N5bnRoZXNpcy4lMjIlMjAlN0MlMjB0cmFuc2Zvcm1lcnMtY2xpJTIwcnVuJTIwLS10YXNrJTIwZmlsbC1tYXNrJTIwLS1tb2RlbCUyMGZhY2Vib29rJTJGeGxtLXJvYmVydGEteGwlMjAtLWRldmljZSUyMDA=",highlighted:'<span class="hljs-built_in">echo</span> -e <span class="hljs-string">&quot;Plants create &lt;mask&gt; through a process known as photosynthesis.&quot;</span> | transformers-cli run --task fill-mask --model facebook/xlm-roberta-xl --device 0',wrap:!1}}),{c(){f(t.$$.fragment)},l(o){g(t.$$.fragment,o)},m(o,a){M(t,o,a),p=!0},p:L,i(o){p||(b(t.$$.fragment,o),p=!0)},o(o){_(t.$$.fragment,o),p=!1},d(o){y(t,o)}}}function cs(k){let t,p,o,a,T,n;return t=new hn({props:{id:"usage",option:"Pipeline",$$slots:{default:[ls]},$$scope:{ctx:k}}}),o=new hn({props:{id:"usage",option:"AutoModel",$$slots:{default:[is]},$$scope:{ctx:k}}}),T=new hn({props:{id:"usage",option:"transformers CLI",$$slots:{default:[ds]},$$scope:{ctx:k}}}),{c(){f(t.$$.fragment),p=l(),f(o.$$.fragment),a=l(),f(T.$$.fragment)},l(u){g(t.$$.fragment,u),p=i(u),g(o.$$.fragment,u),a=i(u),g(T.$$.fragment,u)},m(u,X){M(t,u,X),c(u,p,X),M(o,u,X),c(u,a,X),M(T,u,X),n=!0},p(u,X){const Wt={};X&2&&(Wt.$$scope={dirty:X,ctx:u}),t.$set(Wt);const Je={};X&2&&(Je.$$scope={dirty:X,ctx:u}),o.$set(Je);const se={};X&2&&(se.$$scope={dirty:X,ctx:u}),T.$set(se)},i(u){n||(b(t.$$.fragment,u),b(o.$$.fragment,u),b(T.$$.fragment,u),n=!0)},o(u){_(t.$$.fragment,u),_(o.$$.fragment,u),_(T.$$.fragment,u),n=!1},d(u){u&&(r(p),r(a)),y(t,u),y(o,u),y(T,u)}}}function ps(k){let t,p="Examples:",o,a,T;return a=new O({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMFhMTVJvYmVydGFYTENvbmZpZyUyQyUyMFhMTVJvYmVydGFYTE1vZGVsJTBBJTBBJTIzJTIwSW5pdGlhbGl6aW5nJTIwYSUyMFhMTV9ST0JFUlRBX1hMJTIwZ29vZ2xlLWJlcnQlMkZiZXJ0LWJhc2UtdW5jYXNlZCUyMHN0eWxlJTIwY29uZmlndXJhdGlvbiUwQWNvbmZpZ3VyYXRpb24lMjAlM0QlMjBYTE1Sb2JlcnRhWExDb25maWcoKSUwQSUwQSUyMyUyMEluaXRpYWxpemluZyUyMGElMjBtb2RlbCUyMCh3aXRoJTIwcmFuZG9tJTIwd2VpZ2h0cyklMjBmcm9tJTIwdGhlJTIwZ29vZ2xlLWJlcnQlMkZiZXJ0LWJhc2UtdW5jYXNlZCUyMHN0eWxlJTIwY29uZmlndXJhdGlvbiUwQW1vZGVsJTIwJTNEJTIwWExNUm9iZXJ0YVhMTW9kZWwoY29uZmlndXJhdGlvbiklMEElMEElMjMlMjBBY2Nlc3NpbmclMjB0aGUlMjBtb2RlbCUyMGNvbmZpZ3VyYXRpb24lMEFjb25maWd1cmF0aW9uJTIwJTNEJTIwbW9kZWwuY29uZmln",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> XLMRobertaXLConfig, XLMRobertaXLModel

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a XLM_ROBERTA_XL google-bert/bert-base-uncased style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = XLMRobertaXLConfig()

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a model (with random weights) from the google-bert/bert-base-uncased style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = XLMRobertaXLModel(configuration)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Accessing the model configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = model.config`,wrap:!1}}),{c(){t=m("p"),t.textContent=p,o=l(),f(a.$$.fragment)},l(n){t=h(n,"P",{"data-svelte-h":!0}),w(t)!=="svelte-kvfsh7"&&(t.textContent=p),o=i(n),g(a.$$.fragment,n)},m(n,u){c(n,t,u),c(n,o,u),M(a,n,u),T=!0},p:L,i(n){T||(b(a.$$.fragment,n),T=!0)},o(n){_(a.$$.fragment,n),T=!1},d(n){n&&(r(t),r(o)),y(a,n)}}}function ms(k){let t,p=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=m("p"),t.innerHTML=p},l(o){t=h(o,"P",{"data-svelte-h":!0}),w(t)!=="svelte-fincs2"&&(t.innerHTML=p)},m(o,a){c(o,t,a)},p:L,d(o){o&&r(t)}}}function hs(k){let t,p=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=m("p"),t.innerHTML=p},l(o){t=h(o,"P",{"data-svelte-h":!0}),w(t)!=="svelte-fincs2"&&(t.innerHTML=p)},m(o,a){c(o,t,a)},p:L,d(o){o&&r(t)}}}function us(k){let t,p="Example:",o,a,T;return a=new O({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBSb2JlcnRhRm9yQ2F1c2FsTE0lMkMlMjBSb2JlcnRhQ29uZmlnJTBBaW1wb3J0JTIwdG9yY2glMEElMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJGYWNlYm9va0FJJTJGcm9iZXJ0YS1iYXNlJTIyKSUwQWNvbmZpZyUyMCUzRCUyMFJvYmVydGFDb25maWcuZnJvbV9wcmV0cmFpbmVkKCUyMkZhY2Vib29rQUklMkZyb2JlcnRhLWJhc2UlMjIpJTBBY29uZmlnLmlzX2RlY29kZXIlMjAlM0QlMjBUcnVlJTBBbW9kZWwlMjAlM0QlMjBSb2JlcnRhRm9yQ2F1c2FsTE0uZnJvbV9wcmV0cmFpbmVkKCUyMkZhY2Vib29rQUklMkZyb2JlcnRhLWJhc2UlMjIlMkMlMjBjb25maWclM0Rjb25maWcpJTBBaW5wdXRzJTIwJTNEJTIwdG9rZW5pemVyKCUyMkhlbGxvJTJDJTIwbXklMjBkb2clMjBpcyUyMGN1dGUlMjIlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQW91dHB1dHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyklMEFwcmVkaWN0aW9uX2xvZ2l0cyUyMCUzRCUyMG91dHB1dHMubG9naXRz",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, RobertaForCausalLM, RobertaConfig
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;FacebookAI/roberta-base&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>config = RobertaConfig.from_pretrained(<span class="hljs-string">&quot;FacebookAI/roberta-base&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>config.is_decoder = <span class="hljs-literal">True</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = RobertaForCausalLM.from_pretrained(<span class="hljs-string">&quot;FacebookAI/roberta-base&quot;</span>, config=config)
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs)
<span class="hljs-meta">&gt;&gt;&gt; </span>prediction_logits = outputs.logits`,wrap:!1}}),{c(){t=m("p"),t.textContent=p,o=l(),f(a.$$.fragment)},l(n){t=h(n,"P",{"data-svelte-h":!0}),w(t)!=="svelte-11lpom8"&&(t.textContent=p),o=i(n),g(a.$$.fragment,n)},m(n,u){c(n,t,u),c(n,o,u),M(a,n,u),T=!0},p:L,i(n){T||(b(a.$$.fragment,n),T=!0)},o(n){_(a.$$.fragment,n),T=!1},d(n){n&&(r(t),r(o)),y(a,n)}}}function fs(k){let t,p=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=m("p"),t.innerHTML=p},l(o){t=h(o,"P",{"data-svelte-h":!0}),w(t)!=="svelte-fincs2"&&(t.innerHTML=p)},m(o,a){c(o,t,a)},p:L,d(o){o&&r(t)}}}function gs(k){let t,p="Example:",o,a,T;return a=new O({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBYTE1Sb2JlcnRhWExGb3JNYXNrZWRMTSUwQWltcG9ydCUyMHRvcmNoJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIyZmFjZWJvb2slMkZ4bG0tcm9iZXJ0YS14bCUyMiklMEFtb2RlbCUyMCUzRCUyMFhMTVJvYmVydGFYTEZvck1hc2tlZExNLmZyb21fcHJldHJhaW5lZCglMjJmYWNlYm9vayUyRnhsbS1yb2JlcnRhLXhsJTIyKSUwQSUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplciglMjJUaGUlMjBjYXBpdGFsJTIwb2YlMjBGcmFuY2UlMjBpcyUyMCUzQ21hc2slM0UuJTIyJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiklMEElMEF3aXRoJTIwdG9yY2gubm9fZ3JhZCgpJTNBJTBBJTIwJTIwJTIwJTIwbG9naXRzJTIwJTNEJTIwbW9kZWwoKippbnB1dHMpLmxvZ2l0cyUwQSUwQSUyMyUyMHJldHJpZXZlJTIwaW5kZXglMjBvZiUyMCUzQ21hc2slM0UlMEFtYXNrX3Rva2VuX2luZGV4JTIwJTNEJTIwKGlucHV0cy5pbnB1dF9pZHMlMjAlM0QlM0QlMjB0b2tlbml6ZXIubWFza190b2tlbl9pZCklNUIwJTVELm5vbnplcm8oYXNfdHVwbGUlM0RUcnVlKSU1QjAlNUQlMEElMEFwcmVkaWN0ZWRfdG9rZW5faWQlMjAlM0QlMjBsb2dpdHMlNUIwJTJDJTIwbWFza190b2tlbl9pbmRleCU1RC5hcmdtYXgoYXhpcyUzRC0xKSUwQXRva2VuaXplci5kZWNvZGUocHJlZGljdGVkX3Rva2VuX2lkKSUwQSUwQWxhYmVscyUyMCUzRCUyMHRva2VuaXplciglMjJUaGUlMjBjYXBpdGFsJTIwb2YlMjBGcmFuY2UlMjBpcyUyMFBhcmlzLiUyMiUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpJTVCJTIyaW5wdXRfaWRzJTIyJTVEJTBBJTIzJTIwbWFzayUyMGxhYmVscyUyMG9mJTIwbm9uLSUzQ21hc2slM0UlMjB0b2tlbnMlMEFsYWJlbHMlMjAlM0QlMjB0b3JjaC53aGVyZShpbnB1dHMuaW5wdXRfaWRzJTIwJTNEJTNEJTIwdG9rZW5pemVyLm1hc2tfdG9rZW5faWQlMkMlMjBsYWJlbHMlMkMlMjAtMTAwKSUwQSUwQW91dHB1dHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyUyQyUyMGxhYmVscyUzRGxhYmVscyklMEFyb3VuZChvdXRwdXRzLmxvc3MuaXRlbSgpJTJDJTIwMik=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, XLMRobertaXLForMaskedLM
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;facebook/xlm-roberta-xl&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = XLMRobertaXLForMaskedLM.from_pretrained(<span class="hljs-string">&quot;facebook/xlm-roberta-xl&quot;</span>)

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
...`,wrap:!1}}),{c(){t=m("p"),t.textContent=p,o=l(),f(a.$$.fragment)},l(n){t=h(n,"P",{"data-svelte-h":!0}),w(t)!=="svelte-11lpom8"&&(t.textContent=p),o=i(n),g(a.$$.fragment,n)},m(n,u){c(n,t,u),c(n,o,u),M(a,n,u),T=!0},p:L,i(n){T||(b(a.$$.fragment,n),T=!0)},o(n){_(a.$$.fragment,n),T=!1},d(n){n&&(r(t),r(o)),y(a,n)}}}function Ms(k){let t,p=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=m("p"),t.innerHTML=p},l(o){t=h(o,"P",{"data-svelte-h":!0}),w(t)!=="svelte-fincs2"&&(t.innerHTML=p)},m(o,a){c(o,t,a)},p:L,d(o){o&&r(t)}}}function bs(k){let t,p="Example of single-label classification:",o,a,T;return a=new O({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Rva2VuaXplciUyQyUyMFhMTVJvYmVydGFYTEZvclNlcXVlbmNlQ2xhc3NpZmljYXRpb24lMEElMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJmYWNlYm9vayUyRnhsbS1yb2JlcnRhLXhsJTIyKSUwQW1vZGVsJTIwJTNEJTIwWExNUm9iZXJ0YVhMRm9yU2VxdWVuY2VDbGFzc2lmaWNhdGlvbi5mcm9tX3ByZXRyYWluZWQoJTIyZmFjZWJvb2slMkZ4bG0tcm9iZXJ0YS14bCUyMiklMEElMEFpbnB1dHMlMjAlM0QlMjB0b2tlbml6ZXIoJTIySGVsbG8lMkMlMjBteSUyMGRvZyUyMGlzJTIwY3V0ZSUyMiUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpJTBBJTBBd2l0aCUyMHRvcmNoLm5vX2dyYWQoKSUzQSUwQSUyMCUyMCUyMCUyMGxvZ2l0cyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzKS5sb2dpdHMlMEElMEFwcmVkaWN0ZWRfY2xhc3NfaWQlMjAlM0QlMjBsb2dpdHMuYXJnbWF4KCkuaXRlbSgpJTBBbW9kZWwuY29uZmlnLmlkMmxhYmVsJTVCcHJlZGljdGVkX2NsYXNzX2lkJTVEJTBBJTBBJTIzJTIwVG8lMjB0cmFpbiUyMGElMjBtb2RlbCUyMG9uJTIwJTYwbnVtX2xhYmVscyU2MCUyMGNsYXNzZXMlMkMlMjB5b3UlMjBjYW4lMjBwYXNzJTIwJTYwbnVtX2xhYmVscyUzRG51bV9sYWJlbHMlNjAlMjB0byUyMCU2MC5mcm9tX3ByZXRyYWluZWQoLi4uKSU2MCUwQW51bV9sYWJlbHMlMjAlM0QlMjBsZW4obW9kZWwuY29uZmlnLmlkMmxhYmVsKSUwQW1vZGVsJTIwJTNEJTIwWExNUm9iZXJ0YVhMRm9yU2VxdWVuY2VDbGFzc2lmaWNhdGlvbi5mcm9tX3ByZXRyYWluZWQoJTIyZmFjZWJvb2slMkZ4bG0tcm9iZXJ0YS14bCUyMiUyQyUyMG51bV9sYWJlbHMlM0RudW1fbGFiZWxzKSUwQSUwQWxhYmVscyUyMCUzRCUyMHRvcmNoLnRlbnNvciglNUIxJTVEKSUwQWxvc3MlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyUyQyUyMGxhYmVscyUzRGxhYmVscykubG9zcyUwQXJvdW5kKGxvc3MuaXRlbSgpJTJDJTIwMik=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, XLMRobertaXLForSequenceClassification

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;facebook/xlm-roberta-xl&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = XLMRobertaXLForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;facebook/xlm-roberta-xl&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    logits = model(**inputs).logits

<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_class_id = logits.argmax().item()
<span class="hljs-meta">&gt;&gt;&gt; </span>model.config.id2label[predicted_class_id]
...

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># To train a model on \`num_labels\` classes, you can pass \`num_labels=num_labels\` to \`.from_pretrained(...)\`</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>num_labels = <span class="hljs-built_in">len</span>(model.config.id2label)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = XLMRobertaXLForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;facebook/xlm-roberta-xl&quot;</span>, num_labels=num_labels)

<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.tensor([<span class="hljs-number">1</span>])
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = model(**inputs, labels=labels).loss
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">round</span>(loss.item(), <span class="hljs-number">2</span>)
...`,wrap:!1}}),{c(){t=m("p"),t.textContent=p,o=l(),f(a.$$.fragment)},l(n){t=h(n,"P",{"data-svelte-h":!0}),w(t)!=="svelte-ykxpe4"&&(t.textContent=p),o=i(n),g(a.$$.fragment,n)},m(n,u){c(n,t,u),c(n,o,u),M(a,n,u),T=!0},p:L,i(n){T||(b(a.$$.fragment,n),T=!0)},o(n){_(a.$$.fragment,n),T=!1},d(n){n&&(r(t),r(o)),y(a,n)}}}function _s(k){let t,p="Example of multi-label classification:",o,a,T;return a=new O({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Rva2VuaXplciUyQyUyMFhMTVJvYmVydGFYTEZvclNlcXVlbmNlQ2xhc3NpZmljYXRpb24lMEElMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJmYWNlYm9vayUyRnhsbS1yb2JlcnRhLXhsJTIyKSUwQW1vZGVsJTIwJTNEJTIwWExNUm9iZXJ0YVhMRm9yU2VxdWVuY2VDbGFzc2lmaWNhdGlvbi5mcm9tX3ByZXRyYWluZWQoJTIyZmFjZWJvb2slMkZ4bG0tcm9iZXJ0YS14bCUyMiUyQyUyMHByb2JsZW1fdHlwZSUzRCUyMm11bHRpX2xhYmVsX2NsYXNzaWZpY2F0aW9uJTIyKSUwQSUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplciglMjJIZWxsbyUyQyUyMG15JTIwZG9nJTIwaXMlMjBjdXRlJTIyJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiklMEElMEF3aXRoJTIwdG9yY2gubm9fZ3JhZCgpJTNBJTBBJTIwJTIwJTIwJTIwbG9naXRzJTIwJTNEJTIwbW9kZWwoKippbnB1dHMpLmxvZ2l0cyUwQSUwQXByZWRpY3RlZF9jbGFzc19pZHMlMjAlM0QlMjB0b3JjaC5hcmFuZ2UoMCUyQyUyMGxvZ2l0cy5zaGFwZSU1Qi0xJTVEKSU1QnRvcmNoLnNpZ21vaWQobG9naXRzKS5zcXVlZXplKGRpbSUzRDApJTIwJTNFJTIwMC41JTVEJTBBJTBBJTIzJTIwVG8lMjB0cmFpbiUyMGElMjBtb2RlbCUyMG9uJTIwJTYwbnVtX2xhYmVscyU2MCUyMGNsYXNzZXMlMkMlMjB5b3UlMjBjYW4lMjBwYXNzJTIwJTYwbnVtX2xhYmVscyUzRG51bV9sYWJlbHMlNjAlMjB0byUyMCU2MC5mcm9tX3ByZXRyYWluZWQoLi4uKSU2MCUwQW51bV9sYWJlbHMlMjAlM0QlMjBsZW4obW9kZWwuY29uZmlnLmlkMmxhYmVsKSUwQW1vZGVsJTIwJTNEJTIwWExNUm9iZXJ0YVhMRm9yU2VxdWVuY2VDbGFzc2lmaWNhdGlvbi5mcm9tX3ByZXRyYWluZWQoJTBBJTIwJTIwJTIwJTIwJTIyZmFjZWJvb2slMkZ4bG0tcm9iZXJ0YS14bCUyMiUyQyUyMG51bV9sYWJlbHMlM0RudW1fbGFiZWxzJTJDJTIwcHJvYmxlbV90eXBlJTNEJTIybXVsdGlfbGFiZWxfY2xhc3NpZmljYXRpb24lMjIlMEEpJTBBJTBBbGFiZWxzJTIwJTNEJTIwdG9yY2guc3VtKCUwQSUyMCUyMCUyMCUyMHRvcmNoLm5uLmZ1bmN0aW9uYWwub25lX2hvdChwcmVkaWN0ZWRfY2xhc3NfaWRzJTVCTm9uZSUyQyUyMCUzQSU1RC5jbG9uZSgpJTJDJTIwbnVtX2NsYXNzZXMlM0RudW1fbGFiZWxzKSUyQyUyMGRpbSUzRDElMEEpLnRvKHRvcmNoLmZsb2F0KSUwQWxvc3MlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyUyQyUyMGxhYmVscyUzRGxhYmVscykubG9zcw==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, XLMRobertaXLForSequenceClassification

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;facebook/xlm-roberta-xl&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = XLMRobertaXLForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;facebook/xlm-roberta-xl&quot;</span>, problem_type=<span class="hljs-string">&quot;multi_label_classification&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    logits = model(**inputs).logits

<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_class_ids = torch.arange(<span class="hljs-number">0</span>, logits.shape[-<span class="hljs-number">1</span>])[torch.sigmoid(logits).squeeze(dim=<span class="hljs-number">0</span>) &gt; <span class="hljs-number">0.5</span>]

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># To train a model on \`num_labels\` classes, you can pass \`num_labels=num_labels\` to \`.from_pretrained(...)\`</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>num_labels = <span class="hljs-built_in">len</span>(model.config.id2label)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = XLMRobertaXLForSequenceClassification.from_pretrained(
<span class="hljs-meta">... </span>    <span class="hljs-string">&quot;facebook/xlm-roberta-xl&quot;</span>, num_labels=num_labels, problem_type=<span class="hljs-string">&quot;multi_label_classification&quot;</span>
<span class="hljs-meta">... </span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.<span class="hljs-built_in">sum</span>(
<span class="hljs-meta">... </span>    torch.nn.functional.one_hot(predicted_class_ids[<span class="hljs-literal">None</span>, :].clone(), num_classes=num_labels), dim=<span class="hljs-number">1</span>
<span class="hljs-meta">... </span>).to(torch.<span class="hljs-built_in">float</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = model(**inputs, labels=labels).loss`,wrap:!1}}),{c(){t=m("p"),t.textContent=p,o=l(),f(a.$$.fragment)},l(n){t=h(n,"P",{"data-svelte-h":!0}),w(t)!=="svelte-1l8e32d"&&(t.textContent=p),o=i(n),g(a.$$.fragment,n)},m(n,u){c(n,t,u),c(n,o,u),M(a,n,u),T=!0},p:L,i(n){T||(b(a.$$.fragment,n),T=!0)},o(n){_(a.$$.fragment,n),T=!1},d(n){n&&(r(t),r(o)),y(a,n)}}}function ys(k){let t,p=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=m("p"),t.innerHTML=p},l(o){t=h(o,"P",{"data-svelte-h":!0}),w(t)!=="svelte-fincs2"&&(t.innerHTML=p)},m(o,a){c(o,t,a)},p:L,d(o){o&&r(t)}}}function Ts(k){let t,p="Example:",o,a,T;return a=new O({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBYTE1Sb2JlcnRhWExGb3JNdWx0aXBsZUNob2ljZSUwQWltcG9ydCUyMHRvcmNoJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIyZmFjZWJvb2slMkZ4bG0tcm9iZXJ0YS14bCUyMiklMEFtb2RlbCUyMCUzRCUyMFhMTVJvYmVydGFYTEZvck11bHRpcGxlQ2hvaWNlLmZyb21fcHJldHJhaW5lZCglMjJmYWNlYm9vayUyRnhsbS1yb2JlcnRhLXhsJTIyKSUwQSUwQXByb21wdCUyMCUzRCUyMCUyMkluJTIwSXRhbHklMkMlMjBwaXp6YSUyMHNlcnZlZCUyMGluJTIwZm9ybWFsJTIwc2V0dGluZ3MlMkMlMjBzdWNoJTIwYXMlMjBhdCUyMGElMjByZXN0YXVyYW50JTJDJTIwaXMlMjBwcmVzZW50ZWQlMjB1bnNsaWNlZC4lMjIlMEFjaG9pY2UwJTIwJTNEJTIwJTIySXQlMjBpcyUyMGVhdGVuJTIwd2l0aCUyMGElMjBmb3JrJTIwYW5kJTIwYSUyMGtuaWZlLiUyMiUwQWNob2ljZTElMjAlM0QlMjAlMjJJdCUyMGlzJTIwZWF0ZW4lMjB3aGlsZSUyMGhlbGQlMjBpbiUyMHRoZSUyMGhhbmQuJTIyJTBBbGFiZWxzJTIwJTNEJTIwdG9yY2gudGVuc29yKDApLnVuc3F1ZWV6ZSgwKSUyMCUyMCUyMyUyMGNob2ljZTAlMjBpcyUyMGNvcnJlY3QlMjAoYWNjb3JkaW5nJTIwdG8lMjBXaWtpcGVkaWElMjAlM0IpKSUyQyUyMGJhdGNoJTIwc2l6ZSUyMDElMEElMEFlbmNvZGluZyUyMCUzRCUyMHRva2VuaXplciglNUJwcm9tcHQlMkMlMjBwcm9tcHQlNUQlMkMlMjAlNUJjaG9pY2UwJTJDJTIwY2hvaWNlMSU1RCUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIlMkMlMjBwYWRkaW5nJTNEVHJ1ZSklMEFvdXRwdXRzJTIwJTNEJTIwbW9kZWwoKiolN0JrJTNBJTIwdi51bnNxdWVlemUoMCklMjBmb3IlMjBrJTJDJTIwdiUyMGluJTIwZW5jb2RpbmcuaXRlbXMoKSU3RCUyQyUyMGxhYmVscyUzRGxhYmVscyklMjAlMjAlMjMlMjBiYXRjaCUyMHNpemUlMjBpcyUyMDElMEElMEElMjMlMjB0aGUlMjBsaW5lYXIlMjBjbGFzc2lmaWVyJTIwc3RpbGwlMjBuZWVkcyUyMHRvJTIwYmUlMjB0cmFpbmVkJTBBbG9zcyUyMCUzRCUyMG91dHB1dHMubG9zcyUwQWxvZ2l0cyUyMCUzRCUyMG91dHB1dHMubG9naXRz",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, XLMRobertaXLForMultipleChoice
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;facebook/xlm-roberta-xl&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = XLMRobertaXLForMultipleChoice.from_pretrained(<span class="hljs-string">&quot;facebook/xlm-roberta-xl&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>prompt = <span class="hljs-string">&quot;In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>choice0 = <span class="hljs-string">&quot;It is eaten with a fork and a knife.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>choice1 = <span class="hljs-string">&quot;It is eaten while held in the hand.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.tensor(<span class="hljs-number">0</span>).unsqueeze(<span class="hljs-number">0</span>)  <span class="hljs-comment"># choice0 is correct (according to Wikipedia ;)), batch size 1</span>

<span class="hljs-meta">&gt;&gt;&gt; </span>encoding = tokenizer([prompt, prompt], [choice0, choice1], return_tensors=<span class="hljs-string">&quot;pt&quot;</span>, padding=<span class="hljs-literal">True</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**{k: v.unsqueeze(<span class="hljs-number">0</span>) <span class="hljs-keyword">for</span> k, v <span class="hljs-keyword">in</span> encoding.items()}, labels=labels)  <span class="hljs-comment"># batch size is 1</span>

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># the linear classifier still needs to be trained</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = outputs.loss
<span class="hljs-meta">&gt;&gt;&gt; </span>logits = outputs.logits`,wrap:!1}}),{c(){t=m("p"),t.textContent=p,o=l(),f(a.$$.fragment)},l(n){t=h(n,"P",{"data-svelte-h":!0}),w(t)!=="svelte-11lpom8"&&(t.textContent=p),o=i(n),g(a.$$.fragment,n)},m(n,u){c(n,t,u),c(n,o,u),M(a,n,u),T=!0},p:L,i(n){T||(b(a.$$.fragment,n),T=!0)},o(n){_(a.$$.fragment,n),T=!1},d(n){n&&(r(t),r(o)),y(a,n)}}}function ws(k){let t,p=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=m("p"),t.innerHTML=p},l(o){t=h(o,"P",{"data-svelte-h":!0}),w(t)!=="svelte-fincs2"&&(t.innerHTML=p)},m(o,a){c(o,t,a)},p:L,d(o){o&&r(t)}}}function ks(k){let t,p="Example:",o,a,T;return a=new O({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBYTE1Sb2JlcnRhWExGb3JUb2tlbkNsYXNzaWZpY2F0aW9uJTBBaW1wb3J0JTIwdG9yY2glMEElMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJmYWNlYm9vayUyRnhsbS1yb2JlcnRhLXhsJTIyKSUwQW1vZGVsJTIwJTNEJTIwWExNUm9iZXJ0YVhMRm9yVG9rZW5DbGFzc2lmaWNhdGlvbi5mcm9tX3ByZXRyYWluZWQoJTIyZmFjZWJvb2slMkZ4bG0tcm9iZXJ0YS14bCUyMiklMEElMEFpbnB1dHMlMjAlM0QlMjB0b2tlbml6ZXIoJTBBJTIwJTIwJTIwJTIwJTIySHVnZ2luZ0ZhY2UlMjBpcyUyMGElMjBjb21wYW55JTIwYmFzZWQlMjBpbiUyMFBhcmlzJTIwYW5kJTIwTmV3JTIwWW9yayUyMiUyQyUyMGFkZF9zcGVjaWFsX3Rva2VucyUzREZhbHNlJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiUwQSklMEElMEF3aXRoJTIwdG9yY2gubm9fZ3JhZCgpJTNBJTBBJTIwJTIwJTIwJTIwbG9naXRzJTIwJTNEJTIwbW9kZWwoKippbnB1dHMpLmxvZ2l0cyUwQSUwQXByZWRpY3RlZF90b2tlbl9jbGFzc19pZHMlMjAlM0QlMjBsb2dpdHMuYXJnbWF4KC0xKSUwQSUwQSUyMyUyME5vdGUlMjB0aGF0JTIwdG9rZW5zJTIwYXJlJTIwY2xhc3NpZmllZCUyMHJhdGhlciUyMHRoZW4lMjBpbnB1dCUyMHdvcmRzJTIwd2hpY2glMjBtZWFucyUyMHRoYXQlMEElMjMlMjB0aGVyZSUyMG1pZ2h0JTIwYmUlMjBtb3JlJTIwcHJlZGljdGVkJTIwdG9rZW4lMjBjbGFzc2VzJTIwdGhhbiUyMHdvcmRzLiUwQSUyMyUyME11bHRpcGxlJTIwdG9rZW4lMjBjbGFzc2VzJTIwbWlnaHQlMjBhY2NvdW50JTIwZm9yJTIwdGhlJTIwc2FtZSUyMHdvcmQlMEFwcmVkaWN0ZWRfdG9rZW5zX2NsYXNzZXMlMjAlM0QlMjAlNUJtb2RlbC5jb25maWcuaWQybGFiZWwlNUJ0Lml0ZW0oKSU1RCUyMGZvciUyMHQlMjBpbiUyMHByZWRpY3RlZF90b2tlbl9jbGFzc19pZHMlNUIwJTVEJTVEJTBBcHJlZGljdGVkX3Rva2Vuc19jbGFzc2VzJTBBJTBBbGFiZWxzJTIwJTNEJTIwcHJlZGljdGVkX3Rva2VuX2NsYXNzX2lkcyUwQWxvc3MlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyUyQyUyMGxhYmVscyUzRGxhYmVscykubG9zcyUwQXJvdW5kKGxvc3MuaXRlbSgpJTJDJTIwMik=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, XLMRobertaXLForTokenClassification
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;facebook/xlm-roberta-xl&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = XLMRobertaXLForTokenClassification.from_pretrained(<span class="hljs-string">&quot;facebook/xlm-roberta-xl&quot;</span>)

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
...`,wrap:!1}}),{c(){t=m("p"),t.textContent=p,o=l(),f(a.$$.fragment)},l(n){t=h(n,"P",{"data-svelte-h":!0}),w(t)!=="svelte-11lpom8"&&(t.textContent=p),o=i(n),g(a.$$.fragment,n)},m(n,u){c(n,t,u),c(n,o,u),M(a,n,u),T=!0},p:L,i(n){T||(b(a.$$.fragment,n),T=!0)},o(n){_(a.$$.fragment,n),T=!1},d(n){n&&(r(t),r(o)),y(a,n)}}}function Xs(k){let t,p=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=m("p"),t.innerHTML=p},l(o){t=h(o,"P",{"data-svelte-h":!0}),w(t)!=="svelte-fincs2"&&(t.innerHTML=p)},m(o,a){c(o,t,a)},p:L,d(o){o&&r(t)}}}function Ls(k){let t,p="Example:",o,a,T;return a=new O({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBYTE1Sb2JlcnRhWExGb3JRdWVzdGlvbkFuc3dlcmluZyUwQWltcG9ydCUyMHRvcmNoJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIyZmFjZWJvb2slMkZ4bG0tcm9iZXJ0YS14bCUyMiklMEFtb2RlbCUyMCUzRCUyMFhMTVJvYmVydGFYTEZvclF1ZXN0aW9uQW5zd2VyaW5nLmZyb21fcHJldHJhaW5lZCglMjJmYWNlYm9vayUyRnhsbS1yb2JlcnRhLXhsJTIyKSUwQSUwQXF1ZXN0aW9uJTJDJTIwdGV4dCUyMCUzRCUyMCUyMldobyUyMHdhcyUyMEppbSUyMEhlbnNvbiUzRiUyMiUyQyUyMCUyMkppbSUyMEhlbnNvbiUyMHdhcyUyMGElMjBuaWNlJTIwcHVwcGV0JTIyJTBBJTBBaW5wdXRzJTIwJTNEJTIwdG9rZW5pemVyKHF1ZXN0aW9uJTJDJTIwdGV4dCUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpJTBBd2l0aCUyMHRvcmNoLm5vX2dyYWQoKSUzQSUwQSUyMCUyMCUyMCUyMG91dHB1dHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyklMEElMEFhbnN3ZXJfc3RhcnRfaW5kZXglMjAlM0QlMjBvdXRwdXRzLnN0YXJ0X2xvZ2l0cy5hcmdtYXgoKSUwQWFuc3dlcl9lbmRfaW5kZXglMjAlM0QlMjBvdXRwdXRzLmVuZF9sb2dpdHMuYXJnbWF4KCklMEElMEFwcmVkaWN0X2Fuc3dlcl90b2tlbnMlMjAlM0QlMjBpbnB1dHMuaW5wdXRfaWRzJTVCMCUyQyUyMGFuc3dlcl9zdGFydF9pbmRleCUyMCUzQSUyMGFuc3dlcl9lbmRfaW5kZXglMjAlMkIlMjAxJTVEJTBBdG9rZW5pemVyLmRlY29kZShwcmVkaWN0X2Fuc3dlcl90b2tlbnMlMkMlMjBza2lwX3NwZWNpYWxfdG9rZW5zJTNEVHJ1ZSklMEElMEElMjMlMjB0YXJnZXQlMjBpcyUyMCUyMm5pY2UlMjBwdXBwZXQlMjIlMEF0YXJnZXRfc3RhcnRfaW5kZXglMjAlM0QlMjB0b3JjaC50ZW5zb3IoJTVCMTQlNUQpJTBBdGFyZ2V0X2VuZF9pbmRleCUyMCUzRCUyMHRvcmNoLnRlbnNvciglNUIxNSU1RCklMEElMEFvdXRwdXRzJTIwJTNEJTIwbW9kZWwoKippbnB1dHMlMkMlMjBzdGFydF9wb3NpdGlvbnMlM0R0YXJnZXRfc3RhcnRfaW5kZXglMkMlMjBlbmRfcG9zaXRpb25zJTNEdGFyZ2V0X2VuZF9pbmRleCklMEFsb3NzJTIwJTNEJTIwb3V0cHV0cy5sb3NzJTBBcm91bmQobG9zcy5pdGVtKCklMkMlMjAyKQ==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, XLMRobertaXLForQuestionAnswering
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;facebook/xlm-roberta-xl&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = XLMRobertaXLForQuestionAnswering.from_pretrained(<span class="hljs-string">&quot;facebook/xlm-roberta-xl&quot;</span>)

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
...`,wrap:!1}}),{c(){t=m("p"),t.textContent=p,o=l(),f(a.$$.fragment)},l(n){t=h(n,"P",{"data-svelte-h":!0}),w(t)!=="svelte-11lpom8"&&(t.textContent=p),o=i(n),g(a.$$.fragment,n)},m(n,u){c(n,t,u),c(n,o,u),M(a,n,u),T=!0},p:L,i(n){T||(b(a.$$.fragment,n),T=!0)},o(n){_(a.$$.fragment,n),T=!1},d(n){n&&(r(t),r(o)),y(a,n)}}}function vs(k){let t,p,o,a,T,n="<em>This model was released on 2021-05-02 and added to Hugging Face Transformers on 2022-01-29.</em>",u,X,Wt='<div class="flex flex-wrap space-x-1"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/> <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/></div>',Je,se,Zt,je,un='<a href="https://huggingface.co/papers/2105.00572" rel="nofollow">XLM-RoBERTa-XL</a> is a 3.5B parameter multilingual masked language model pretrained on 100 languages. It shows that by scaling model capacity, multilingual models demonstrates strong performance on high-resource languages and can even zero-shot low-resource languages.',Bt,$e,fn='You can find all the original XLM-RoBERTa-XL checkpoints under the <a href="https://huggingface.co/facebook?search_models=xlm" rel="nofollow">AI at Meta</a> organization.',Gt,ie,qt,Re,gn='The example below demonstrates how to predict the <code>&lt;mask&gt;</code> token with <a href="/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.Pipeline">Pipeline</a>, <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel">AutoModel</a>, and from the command line.',Nt,de,Vt,xe,Mn='Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the <a href="../quantization/overview">Quantization</a> overview for more available quantization backends.',Et,Ue,bn='The example below uses <a href="../quantization/torchao">torchao</a> to only quantize the weights to int4.',Ht,Ce,Qt,Fe,St,ze,_n="<li>Unlike some XLM models, XLM-RoBERTa-XL doesn’t require <code>lang</code> tensors to understand which language is used. It automatically determines the language from the input ids.</li>",Yt,We,At,I,Ie,uo,rt,yn=`This is the configuration class to store the configuration of a <a href="/docs/transformers/v4.56.2/en/model_doc/xlm-roberta-xl#transformers.XLMRobertaXLModel">XLMRobertaXLModel</a> or a <code>TFXLMRobertaXLModel</code>.
It is used to instantiate a XLM_ROBERTA_XL model according to the specified arguments, defining the model
architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the
XLM_ROBERTA_XL <a href="https://huggingface.co/facebook/xlm-roberta-xl" rel="nofollow">facebook/xlm-roberta-xl</a> architecture.`,fo,lt,Tn=`Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> and can be used to control the model outputs. Read the
documentation from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> for more information.`,go,ce,Pt,Ze,Ot,j,Be,Mo,it,wn="The bare Xlm Roberta Xl Model outputting raw hidden-states without any specific head on top.",bo,dt,kn=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,_o,ct,Xn=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,yo,ae,Ge,To,pt,Ln='The <a href="/docs/transformers/v4.56.2/en/model_doc/xlm-roberta-xl#transformers.XLMRobertaXLModel">XLMRobertaXLModel</a> forward method, overrides the <code>__call__</code> special method.',wo,pe,Dt,qe,Kt,$,Ne,ko,mt,vn="XLM-RoBERTa-XL Model with a <code>language modeling</code> head on top for CLM fine-tuning.",Xo,ht,Jn=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Lo,ut,jn=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,vo,Q,Ve,Jo,ft,$n='The <a href="/docs/transformers/v4.56.2/en/model_doc/xlm-roberta-xl#transformers.XLMRobertaXLForCausalLM">XLMRobertaXLForCausalLM</a> forward method, overrides the <code>__call__</code> special method.',jo,me,$o,he,eo,Ee,to,R,He,Ro,gt,Rn="The Xlm Roberta Xl Model with a <code>language modeling</code> head on top.”",xo,Mt,xn=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Uo,bt,Un=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Co,S,Qe,Fo,_t,Cn='The <a href="/docs/transformers/v4.56.2/en/model_doc/xlm-roberta-xl#transformers.XLMRobertaXLForMaskedLM">XLMRobertaXLForMaskedLM</a> forward method, overrides the <code>__call__</code> special method.',zo,ue,Wo,fe,oo,Se,no,x,Ye,Io,yt,Fn=`XLM-RoBERTa-XL Model transformer with a sequence classification/regression head on top (a linear layer on top
of the pooled output) e.g. for GLUE tasks.`,Zo,Tt,zn=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Bo,wt,Wn=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Go,W,Ae,qo,kt,In='The <a href="/docs/transformers/v4.56.2/en/model_doc/xlm-roberta-xl#transformers.XLMRobertaXLForSequenceClassification">XLMRobertaXLForSequenceClassification</a> forward method, overrides the <code>__call__</code> special method.',No,ge,Vo,Me,Eo,be,so,Pe,ao,U,Oe,Ho,Xt,Zn=`The Xlm Roberta Xl Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
softmax) e.g. for RocStories/SWAG tasks.`,Qo,Lt,Bn=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,So,vt,Gn=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Yo,Y,De,Ao,Jt,qn='The <a href="/docs/transformers/v4.56.2/en/model_doc/xlm-roberta-xl#transformers.XLMRobertaXLForMultipleChoice">XLMRobertaXLForMultipleChoice</a> forward method, overrides the <code>__call__</code> special method.',Po,_e,Oo,ye,ro,Ke,lo,C,et,Do,jt,Nn=`The Xlm Roberta Xl transformer with a token classification head on top (a linear layer on top of the hidden-states
output) e.g. for Named-Entity-Recognition (NER) tasks.`,Ko,$t,Vn=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,en,Rt,En=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,tn,A,tt,on,xt,Hn='The <a href="/docs/transformers/v4.56.2/en/model_doc/xlm-roberta-xl#transformers.XLMRobertaXLForTokenClassification">XLMRobertaXLForTokenClassification</a> forward method, overrides the <code>__call__</code> special method.',nn,Te,sn,we,io,ot,co,F,nt,an,Ut,Qn=`The Xlm Roberta Xl transformer with a span classification head on top for extractive question-answering tasks like
SQuAD (a linear layer on top of the hidden-states output to compute <code>span start logits</code> and <code>span end logits</code>).`,rn,Ct,Sn=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,ln,Ft,Yn=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,dn,P,st,cn,zt,An='The <a href="/docs/transformers/v4.56.2/en/model_doc/xlm-roberta-xl#transformers.XLMRobertaXLForQuestionAnswering">XLMRobertaXLForQuestionAnswering</a> forward method, overrides the <code>__call__</code> special method.',pn,ke,mn,Xe,po,at,mo,It,ho;return se=new re({props:{title:"XLM-RoBERTa-XL",local:"xlm-roberta-xl",headingTag:"h1"}}),ie=new Le({props:{warning:!1,$$slots:{default:[rs]},$$scope:{ctx:k}}}),de=new as({props:{id:"usage",options:["Pipeline","AutoModel","transformers CLI"],$$slots:{default:[cs]},$$scope:{ctx:k}}}),Ce=new O({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b01vZGVsRm9yTWFza2VkTE0lMkMlMjBBdXRvVG9rZW5pemVyJTJDJTIwVG9yY2hBb0NvbmZpZyUwQSUwQXF1YW50aXphdGlvbl9jb25maWclMjAlM0QlMjBUb3JjaEFvQ29uZmlnKCUyMmludDRfd2VpZ2h0X29ubHklMjIlMkMlMjBncm91cF9zaXplJTNEMTI4KSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUwQSUyMCUyMCUyMCUyMCUyMmZhY2Vib29rJTJGeGxtLXJvYmVydGEteGwlMjIlMkMlMEEpJTBBbW9kZWwlMjAlM0QlMjBBdXRvTW9kZWxGb3JNYXNrZWRMTS5mcm9tX3ByZXRyYWluZWQoJTBBJTIwJTIwJTIwJTIwJTIyZmFjZWJvb2slMkZ4bG0tcm9iZXJ0YS14bCUyMiUyQyUwQSUyMCUyMCUyMCUyMGR0eXBlJTNEdG9yY2guZmxvYXQxNiUyQyUwQSUyMCUyMCUyMCUyMGRldmljZV9tYXAlM0QlMjJhdXRvJTIyJTJDJTBBJTIwJTIwJTIwJTIwYXR0bl9pbXBsZW1lbnRhdGlvbiUzRCUyMnNkcGElMjIlMkMlMEElMjAlMjAlMjAlMjBxdWFudGl6YXRpb25fY29uZmlnJTNEcXVhbnRpemF0aW9uX2NvbmZpZyUwQSklMEFpbnB1dHMlMjAlM0QlMjB0b2tlbml6ZXIoJTIyQm9uam91ciUyQyUyMGplJTIwc3VpcyUyMHVuJTIwbW9kJUMzJUE4bGUlMjAlM0NtYXNrJTNFLiUyMiUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpLnRvKG1vZGVsLmRldmljZSklMEElMEF3aXRoJTIwdG9yY2gubm9fZ3JhZCgpJTNBJTBBJTIwJTIwJTIwJTIwb3V0cHV0cyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzKSUwQSUyMCUyMCUyMCUyMHByZWRpY3Rpb25zJTIwJTNEJTIwb3V0cHV0cy5sb2dpdHMlMEElMEFtYXNrZWRfaW5kZXglMjAlM0QlMjB0b3JjaC53aGVyZShpbnB1dHMlNUInaW5wdXRfaWRzJyU1RCUyMCUzRCUzRCUyMHRva2VuaXplci5tYXNrX3Rva2VuX2lkKSU1QjElNUQlMEFwcmVkaWN0ZWRfdG9rZW5faWQlMjAlM0QlMjBwcmVkaWN0aW9ucyU1QjAlMkMlMjBtYXNrZWRfaW5kZXglNUQuYXJnbWF4KGRpbSUzRC0xKSUwQXByZWRpY3RlZF90b2tlbiUyMCUzRCUyMHRva2VuaXplci5kZWNvZGUocHJlZGljdGVkX3Rva2VuX2lkKSUwQSUwQXByaW50KGYlMjJUaGUlMjBwcmVkaWN0ZWQlMjB0b2tlbiUyMGlzJTNBJTIwJTdCcHJlZGljdGVkX3Rva2VuJTdEJTIyKQ==",highlighted:`<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoModelForMaskedLM, AutoTokenizer, TorchAoConfig

quantization_config = TorchAoConfig(<span class="hljs-string">&quot;int4_weight_only&quot;</span>, group_size=<span class="hljs-number">128</span>)
tokenizer = AutoTokenizer.from_pretrained(
    <span class="hljs-string">&quot;facebook/xlm-roberta-xl&quot;</span>,
)
model = AutoModelForMaskedLM.from_pretrained(
    <span class="hljs-string">&quot;facebook/xlm-roberta-xl&quot;</span>,
    dtype=torch.float16,
    device_map=<span class="hljs-string">&quot;auto&quot;</span>,
    attn_implementation=<span class="hljs-string">&quot;sdpa&quot;</span>,
    quantization_config=quantization_config
)
inputs = tokenizer(<span class="hljs-string">&quot;Bonjour, je suis un modèle &lt;mask&gt;.&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>).to(model.device)

<span class="hljs-keyword">with</span> torch.no_grad():
    outputs = model(**inputs)
    predictions = outputs.logits

masked_index = torch.where(inputs[<span class="hljs-string">&#x27;input_ids&#x27;</span>] == tokenizer.mask_token_id)[<span class="hljs-number">1</span>]
predicted_token_id = predictions[<span class="hljs-number">0</span>, masked_index].argmax(dim=-<span class="hljs-number">1</span>)
predicted_token = tokenizer.decode(predicted_token_id)

<span class="hljs-built_in">print</span>(<span class="hljs-string">f&quot;The predicted token is: <span class="hljs-subst">{predicted_token}</span>&quot;</span>)`,wrap:!1}}),Fe=new re({props:{title:"Notes",local:"notes",headingTag:"h2"}}),We=new re({props:{title:"XLMRobertaXLConfig",local:"transformers.XLMRobertaXLConfig",headingTag:"h2"}}),Ie=new z({props:{name:"class transformers.XLMRobertaXLConfig",anchor:"transformers.XLMRobertaXLConfig",parameters:[{name:"vocab_size",val:" = 250880"},{name:"hidden_size",val:" = 2560"},{name:"num_hidden_layers",val:" = 36"},{name:"num_attention_heads",val:" = 32"},{name:"intermediate_size",val:" = 10240"},{name:"hidden_act",val:" = 'gelu'"},{name:"hidden_dropout_prob",val:" = 0.1"},{name:"attention_probs_dropout_prob",val:" = 0.1"},{name:"max_position_embeddings",val:" = 514"},{name:"type_vocab_size",val:" = 1"},{name:"initializer_range",val:" = 0.02"},{name:"layer_norm_eps",val:" = 1e-05"},{name:"pad_token_id",val:" = 1"},{name:"bos_token_id",val:" = 0"},{name:"eos_token_id",val:" = 2"},{name:"position_embedding_type",val:" = 'absolute'"},{name:"use_cache",val:" = True"},{name:"classifier_dropout",val:" = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.XLMRobertaXLConfig.vocab_size",description:`<strong>vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 250880) &#x2014;
Vocabulary size of the XLM_ROBERTA_XL model. Defines the number of different tokens that can be represented
by the <code>inputs_ids</code> passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/xlm-roberta-xl#transformers.XLMRobertaXLModel">XLMRobertaXLModel</a>.`,name:"vocab_size"},{anchor:"transformers.XLMRobertaXLConfig.hidden_size",description:`<strong>hidden_size</strong> (<code>int</code>, <em>optional</em>, defaults to 2560) &#x2014;
Dimensionality of the encoder layers and the pooler layer.`,name:"hidden_size"},{anchor:"transformers.XLMRobertaXLConfig.num_hidden_layers",description:`<strong>num_hidden_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 36) &#x2014;
Number of hidden layers in the Transformer encoder.`,name:"num_hidden_layers"},{anchor:"transformers.XLMRobertaXLConfig.num_attention_heads",description:`<strong>num_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 32) &#x2014;
Number of attention heads for each attention layer in the Transformer encoder.`,name:"num_attention_heads"},{anchor:"transformers.XLMRobertaXLConfig.intermediate_size",description:`<strong>intermediate_size</strong> (<code>int</code>, <em>optional</em>, defaults to 10240) &#x2014;
Dimensionality of the &#x201C;intermediate&#x201D; (often named feed-forward) layer in the Transformer encoder.`,name:"intermediate_size"},{anchor:"transformers.XLMRobertaXLConfig.hidden_act",description:`<strong>hidden_act</strong> (<code>str</code> or <code>Callable</code>, <em>optional</em>, defaults to <code>&quot;gelu&quot;</code>) &#x2014;
The non-linear activation function (function or string) in the encoder and pooler. If string, <code>&quot;gelu&quot;</code>,
<code>&quot;relu&quot;</code>, <code>&quot;silu&quot;</code> and <code>&quot;gelu_new&quot;</code> are supported.`,name:"hidden_act"},{anchor:"transformers.XLMRobertaXLConfig.hidden_dropout_prob",description:`<strong>hidden_dropout_prob</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.`,name:"hidden_dropout_prob"},{anchor:"transformers.XLMRobertaXLConfig.attention_probs_dropout_prob",description:`<strong>attention_probs_dropout_prob</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout ratio for the attention probabilities.`,name:"attention_probs_dropout_prob"},{anchor:"transformers.XLMRobertaXLConfig.max_position_embeddings",description:`<strong>max_position_embeddings</strong> (<code>int</code>, <em>optional</em>, defaults to 514) &#x2014;
The maximum sequence length that this model might ever be used with. Typically set this to something large
just in case (e.g., 512 or 1024 or 2048).`,name:"max_position_embeddings"},{anchor:"transformers.XLMRobertaXLConfig.type_vocab_size",description:`<strong>type_vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 1) &#x2014;
The vocabulary size of the <code>token_type_ids</code> passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/xlm-roberta-xl#transformers.XLMRobertaXLModel">XLMRobertaXLModel</a> or
<code>TFXLMRobertaXLModel</code>.`,name:"type_vocab_size"},{anchor:"transformers.XLMRobertaXLConfig.initializer_range",description:`<strong>initializer_range</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.`,name:"initializer_range"},{anchor:"transformers.XLMRobertaXLConfig.layer_norm_eps",description:`<strong>layer_norm_eps</strong> (<code>float</code>, <em>optional</em>, defaults to 1e-5) &#x2014;
The epsilon used by the layer normalization layers.`,name:"layer_norm_eps"},{anchor:"transformers.XLMRobertaXLConfig.position_embedding_type",description:`<strong>position_embedding_type</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;absolute&quot;</code>) &#x2014;
Type of position embedding. Choose one of <code>&quot;absolute&quot;</code>, <code>&quot;relative_key&quot;</code>, <code>&quot;relative_key_query&quot;</code>. For
positional embeddings use <code>&quot;absolute&quot;</code>. For more information on <code>&quot;relative_key&quot;</code>, please refer to
<a href="https://huggingface.co/papers/1803.02155" rel="nofollow">Self-Attention with Relative Position Representations (Shaw et al.)</a>.
For more information on <code>&quot;relative_key_query&quot;</code>, please refer to <em>Method 4</em> in <a href="https://huggingface.co/papers/2009.13658" rel="nofollow">Improve Transformer Models
with Better Relative Position Embeddings (Huang et al.)</a>.`,name:"position_embedding_type"},{anchor:"transformers.XLMRobertaXLConfig.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not the model should return the last key/values attentions (not used by all models). Only
relevant if <code>config.is_decoder=True</code>.`,name:"use_cache"},{anchor:"transformers.XLMRobertaXLConfig.classifier_dropout",description:`<strong>classifier_dropout</strong> (<code>float</code>, <em>optional</em>) &#x2014;
The dropout ratio for the classification head.`,name:"classifier_dropout"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlm_roberta_xl/configuration_xlm_roberta_xl.py#L28"}}),ce=new ve({props:{anchor:"transformers.XLMRobertaXLConfig.example",$$slots:{default:[ps]},$$scope:{ctx:k}}}),Ze=new re({props:{title:"XLMRobertaXLModel",local:"transformers.XLMRobertaXLModel",headingTag:"h2"}}),Be=new z({props:{name:"class transformers.XLMRobertaXLModel",anchor:"transformers.XLMRobertaXLModel",parameters:[{name:"config",val:""},{name:"add_pooling_layer",val:" = True"}],parametersDescription:[{anchor:"transformers.XLMRobertaXLModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/xlm-roberta-xl#transformers.XLMRobertaXLModel">XLMRobertaXLModel</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"},{anchor:"transformers.XLMRobertaXLModel.add_pooling_layer",description:`<strong>add_pooling_layer</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether to add a pooling layer`,name:"add_pooling_layer"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlm_roberta_xl/modeling_xlm_roberta_xl.py#L686"}}),Ge=new z({props:{name:"forward",anchor:"transformers.XLMRobertaXLModel.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_hidden_states",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"past_key_values",val:": typing.Optional[list[torch.FloatTensor]] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.Tensor] = None"}],parametersDescription:[{anchor:"transformers.XLMRobertaXLModel.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.XLMRobertaXLModel.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.XLMRobertaXLModel.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.XLMRobertaXLModel.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.XLMRobertaXLModel.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.XLMRobertaXLModel.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.XLMRobertaXLModel.forward.encoder_hidden_states",description:`<strong>encoder_hidden_states</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
if the model is configured as a decoder.`,name:"encoder_hidden_states"},{anchor:"transformers.XLMRobertaXLModel.forward.encoder_attention_mask",description:`<strong>encoder_attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
the cross-attention if the model is configured as a decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>`,name:"encoder_attention_mask"},{anchor:"transformers.XLMRobertaXLModel.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>list[torch.FloatTensor]</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.XLMRobertaXLModel.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.XLMRobertaXLModel.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.XLMRobertaXLModel.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.XLMRobertaXLModel.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.XLMRobertaXLModel.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.Tensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlm_roberta_xl/modeling_xlm_roberta_xl.py#L722",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions"
>transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/xlm-roberta-xl#transformers.XLMRobertaXLConfig"
>XLMRobertaXLConfig</a>) and inputs.</p>
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
`}}),pe=new Le({props:{$$slots:{default:[ms]},$$scope:{ctx:k}}}),qe=new re({props:{title:"XLMRobertaXLForCausalLM",local:"transformers.XLMRobertaXLForCausalLM",headingTag:"h2"}}),Ne=new z({props:{name:"class transformers.XLMRobertaXLForCausalLM",anchor:"transformers.XLMRobertaXLForCausalLM",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.XLMRobertaXLForCausalLM.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/xlm-roberta-xl#transformers.XLMRobertaXLForCausalLM">XLMRobertaXLForCausalLM</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlm_roberta_xl/modeling_xlm_roberta_xl.py#L878"}}),Ve=new z({props:{name:"forward",anchor:"transformers.XLMRobertaXLForCausalLM.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"encoder_hidden_states",val:": typing.Optional[torch.FloatTensor] = None"},{name:"encoder_attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[tuple[tuple[torch.FloatTensor]]] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.XLMRobertaXLForCausalLM.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.XLMRobertaXLForCausalLM.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.XLMRobertaXLForCausalLM.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.XLMRobertaXLForCausalLM.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.XLMRobertaXLForCausalLM.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.XLMRobertaXLForCausalLM.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.XLMRobertaXLForCausalLM.forward.encoder_hidden_states",description:`<strong>encoder_hidden_states</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
if the model is configured as a decoder.`,name:"encoder_hidden_states"},{anchor:"transformers.XLMRobertaXLForCausalLM.forward.encoder_attention_mask",description:`<strong>encoder_attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
the cross-attention if the model is configured as a decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>`,name:"encoder_attention_mask"},{anchor:"transformers.XLMRobertaXLForCausalLM.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the left-to-right language modeling loss (next word prediction). Indices should be in
<code>[-100, 0, ..., config.vocab_size]</code> (see <code>input_ids</code> docstring) Tokens with indices set to <code>-100</code> are
ignored (masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>`,name:"labels"},{anchor:"transformers.XLMRobertaXLForCausalLM.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>tuple[tuple[torch.FloatTensor]]</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.XLMRobertaXLForCausalLM.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.XLMRobertaXLForCausalLM.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.XLMRobertaXLForCausalLM.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.XLMRobertaXLForCausalLM.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlm_roberta_xl/modeling_xlm_roberta_xl.py#L899",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithCrossAttentions"
>transformers.modeling_outputs.CausalLMOutputWithCrossAttentions</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/xlm-roberta-xl#transformers.XLMRobertaXLConfig"
>XLMRobertaXLConfig</a>) and inputs.</p>
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
`}}),me=new Le({props:{$$slots:{default:[hs]},$$scope:{ctx:k}}}),he=new ve({props:{anchor:"transformers.XLMRobertaXLForCausalLM.forward.example",$$slots:{default:[us]},$$scope:{ctx:k}}}),Ee=new re({props:{title:"XLMRobertaXLForMaskedLM",local:"transformers.XLMRobertaXLForMaskedLM",headingTag:"h2"}}),He=new z({props:{name:"class transformers.XLMRobertaXLForMaskedLM",anchor:"transformers.XLMRobertaXLForMaskedLM",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.XLMRobertaXLForMaskedLM.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/xlm-roberta-xl#transformers.XLMRobertaXLForMaskedLM">XLMRobertaXLForMaskedLM</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlm_roberta_xl/modeling_xlm_roberta_xl.py#L1023"}}),Qe=new z({props:{name:"forward",anchor:"transformers.XLMRobertaXLForMaskedLM.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"encoder_hidden_states",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.XLMRobertaXLForMaskedLM.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.XLMRobertaXLForMaskedLM.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.XLMRobertaXLForMaskedLM.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.XLMRobertaXLForMaskedLM.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.XLMRobertaXLForMaskedLM.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.XLMRobertaXLForMaskedLM.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.XLMRobertaXLForMaskedLM.forward.encoder_hidden_states",description:`<strong>encoder_hidden_states</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
if the model is configured as a decoder.`,name:"encoder_hidden_states"},{anchor:"transformers.XLMRobertaXLForMaskedLM.forward.encoder_attention_mask",description:`<strong>encoder_attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
the cross-attention if the model is configured as a decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>`,name:"encoder_attention_mask"},{anchor:"transformers.XLMRobertaXLForMaskedLM.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should be in <code>[-100, 0, ..., config.vocab_size]</code> (see <code>input_ids</code> docstring) Tokens with indices set to <code>-100</code> are ignored (masked), the
loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>`,name:"labels"},{anchor:"transformers.XLMRobertaXLForMaskedLM.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.XLMRobertaXLForMaskedLM.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.XLMRobertaXLForMaskedLM.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlm_roberta_xl/modeling_xlm_roberta_xl.py#L1047",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.MaskedLMOutput"
>transformers.modeling_outputs.MaskedLMOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/xlm-roberta-xl#transformers.XLMRobertaXLConfig"
>XLMRobertaXLConfig</a>) and inputs.</p>
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
`}}),ue=new Le({props:{$$slots:{default:[fs]},$$scope:{ctx:k}}}),fe=new ve({props:{anchor:"transformers.XLMRobertaXLForMaskedLM.forward.example",$$slots:{default:[gs]},$$scope:{ctx:k}}}),Se=new re({props:{title:"XLMRobertaXLForSequenceClassification",local:"transformers.XLMRobertaXLForSequenceClassification",headingTag:"h2"}}),Ye=new z({props:{name:"class transformers.XLMRobertaXLForSequenceClassification",anchor:"transformers.XLMRobertaXLForSequenceClassification",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.XLMRobertaXLForSequenceClassification.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/xlm-roberta-xl#transformers.XLMRobertaXLForSequenceClassification">XLMRobertaXLForSequenceClassification</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlm_roberta_xl/modeling_xlm_roberta_xl.py#L1141"}}),Ae=new z({props:{name:"forward",anchor:"transformers.XLMRobertaXLForSequenceClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.XLMRobertaXLForSequenceClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.XLMRobertaXLForSequenceClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.XLMRobertaXLForSequenceClassification.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.XLMRobertaXLForSequenceClassification.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.XLMRobertaXLForSequenceClassification.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.XLMRobertaXLForSequenceClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.XLMRobertaXLForSequenceClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the sequence classification/regression loss. Indices should be in <code>[0, ..., config.num_labels - 1]</code>. If <code>config.num_labels == 1</code> a regression loss is computed (Mean-Square loss), If
<code>config.num_labels &gt; 1</code> a classification loss is computed (Cross-Entropy).`,name:"labels"},{anchor:"transformers.XLMRobertaXLForSequenceClassification.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.XLMRobertaXLForSequenceClassification.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.XLMRobertaXLForSequenceClassification.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlm_roberta_xl/modeling_xlm_roberta_xl.py#L1152",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput"
>transformers.modeling_outputs.SequenceClassifierOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/xlm-roberta-xl#transformers.XLMRobertaXLConfig"
>XLMRobertaXLConfig</a>) and inputs.</p>
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
`}}),ge=new Le({props:{$$slots:{default:[Ms]},$$scope:{ctx:k}}}),Me=new ve({props:{anchor:"transformers.XLMRobertaXLForSequenceClassification.forward.example",$$slots:{default:[bs]},$$scope:{ctx:k}}}),be=new ve({props:{anchor:"transformers.XLMRobertaXLForSequenceClassification.forward.example-2",$$slots:{default:[_s]},$$scope:{ctx:k}}}),Pe=new re({props:{title:"XLMRobertaXLForMultipleChoice",local:"transformers.XLMRobertaXLForMultipleChoice",headingTag:"h2"}}),Oe=new z({props:{name:"class transformers.XLMRobertaXLForMultipleChoice",anchor:"transformers.XLMRobertaXLForMultipleChoice",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.XLMRobertaXLForMultipleChoice.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/xlm-roberta-xl#transformers.XLMRobertaXLForMultipleChoice">XLMRobertaXLForMultipleChoice</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlm_roberta_xl/modeling_xlm_roberta_xl.py#L1224"}}),De=new z({props:{name:"forward",anchor:"transformers.XLMRobertaXLForMultipleChoice.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.XLMRobertaXLForMultipleChoice.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, num_choices, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary. Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details. <a href="../glossary#input-ids">What are input
IDs?</a>`,name:"input_ids"},{anchor:"transformers.XLMRobertaXLForMultipleChoice.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, num_choices, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.
<a href="../glossary#token-type-ids">What are token type IDs?</a></li>
</ul>`,name:"token_type_ids"},{anchor:"transformers.XLMRobertaXLForMultipleChoice.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.XLMRobertaXLForMultipleChoice.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the multiple choice classification loss. Indices should be in <code>[0, ..., num_choices-1]</code> where <code>num_choices</code> is the size of the second dimension of the input tensors. (See
<code>input_ids</code> above)`,name:"labels"},{anchor:"transformers.XLMRobertaXLForMultipleChoice.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, num_choices, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.max_position_embeddings - 1]</code>. <a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.XLMRobertaXLForMultipleChoice.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.XLMRobertaXLForMultipleChoice.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_choices, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.XLMRobertaXLForMultipleChoice.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.XLMRobertaXLForMultipleChoice.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.XLMRobertaXLForMultipleChoice.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlm_roberta_xl/modeling_xlm_roberta_xl.py#L1234",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.MultipleChoiceModelOutput"
>transformers.modeling_outputs.MultipleChoiceModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/xlm-roberta-xl#transformers.XLMRobertaXLConfig"
>XLMRobertaXLConfig</a>) and inputs.</p>
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
`}}),_e=new Le({props:{$$slots:{default:[ys]},$$scope:{ctx:k}}}),ye=new ve({props:{anchor:"transformers.XLMRobertaXLForMultipleChoice.forward.example",$$slots:{default:[Ts]},$$scope:{ctx:k}}}),Ke=new re({props:{title:"XLMRobertaXLForTokenClassification",local:"transformers.XLMRobertaXLForTokenClassification",headingTag:"h2"}}),et=new z({props:{name:"class transformers.XLMRobertaXLForTokenClassification",anchor:"transformers.XLMRobertaXLForTokenClassification",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.XLMRobertaXLForTokenClassification.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/xlm-roberta-xl#transformers.XLMRobertaXLForTokenClassification">XLMRobertaXLForTokenClassification</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlm_roberta_xl/modeling_xlm_roberta_xl.py#L1320"}}),tt=new z({props:{name:"forward",anchor:"transformers.XLMRobertaXLForTokenClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.XLMRobertaXLForTokenClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.XLMRobertaXLForTokenClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.XLMRobertaXLForTokenClassification.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.XLMRobertaXLForTokenClassification.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.XLMRobertaXLForTokenClassification.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.XLMRobertaXLForTokenClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.XLMRobertaXLForTokenClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the token classification loss. Indices should be in <code>[0, ..., config.num_labels - 1]</code>.`,name:"labels"},{anchor:"transformers.XLMRobertaXLForTokenClassification.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.XLMRobertaXLForTokenClassification.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.XLMRobertaXLForTokenClassification.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlm_roberta_xl/modeling_xlm_roberta_xl.py#L1334",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput"
>transformers.modeling_outputs.TokenClassifierOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/xlm-roberta-xl#transformers.XLMRobertaXLConfig"
>XLMRobertaXLConfig</a>) and inputs.</p>
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
`}}),Te=new Le({props:{$$slots:{default:[ws]},$$scope:{ctx:k}}}),we=new ve({props:{anchor:"transformers.XLMRobertaXLForTokenClassification.forward.example",$$slots:{default:[ks]},$$scope:{ctx:k}}}),ot=new re({props:{title:"XLMRobertaXLForQuestionAnswering",local:"transformers.XLMRobertaXLForQuestionAnswering",headingTag:"h2"}}),nt=new z({props:{name:"class transformers.XLMRobertaXLForQuestionAnswering",anchor:"transformers.XLMRobertaXLForQuestionAnswering",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.XLMRobertaXLForQuestionAnswering.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/xlm-roberta-xl#transformers.XLMRobertaXLForQuestionAnswering">XLMRobertaXLForQuestionAnswering</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlm_roberta_xl/modeling_xlm_roberta_xl.py#L1420"}}),st=new z({props:{name:"forward",anchor:"transformers.XLMRobertaXLForQuestionAnswering.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"start_positions",val:": typing.Optional[torch.LongTensor] = None"},{name:"end_positions",val:": typing.Optional[torch.LongTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.XLMRobertaXLForQuestionAnswering.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.XLMRobertaXLForQuestionAnswering.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.XLMRobertaXLForQuestionAnswering.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.XLMRobertaXLForQuestionAnswering.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.XLMRobertaXLForQuestionAnswering.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.XLMRobertaXLForQuestionAnswering.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.XLMRobertaXLForQuestionAnswering.forward.start_positions",description:`<strong>start_positions</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for position (index) of the start of the labelled span for computing the token classification loss.
Positions are clamped to the length of the sequence (<code>sequence_length</code>). Position outside of the sequence
are not taken into account for computing the loss.`,name:"start_positions"},{anchor:"transformers.XLMRobertaXLForQuestionAnswering.forward.end_positions",description:`<strong>end_positions</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for position (index) of the end of the labelled span for computing the token classification loss.
Positions are clamped to the length of the sequence (<code>sequence_length</code>). Position outside of the sequence
are not taken into account for computing the loss.`,name:"end_positions"},{anchor:"transformers.XLMRobertaXLForQuestionAnswering.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.XLMRobertaXLForQuestionAnswering.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.XLMRobertaXLForQuestionAnswering.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xlm_roberta_xl/modeling_xlm_roberta_xl.py#L1430",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.QuestionAnsweringModelOutput"
>transformers.modeling_outputs.QuestionAnsweringModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/xlm-roberta-xl#transformers.XLMRobertaXLConfig"
>XLMRobertaXLConfig</a>) and inputs.</p>
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
`}}),ke=new Le({props:{$$slots:{default:[Xs]},$$scope:{ctx:k}}}),Xe=new ve({props:{anchor:"transformers.XLMRobertaXLForQuestionAnswering.forward.example",$$slots:{default:[Ls]},$$scope:{ctx:k}}}),at=new ss({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/xlm-roberta-xl.md"}}),{c(){t=m("meta"),p=l(),o=m("p"),a=l(),T=m("p"),T.innerHTML=n,u=l(),X=m("div"),X.innerHTML=Wt,Je=l(),f(se.$$.fragment),Zt=l(),je=m("p"),je.innerHTML=un,Bt=l(),$e=m("p"),$e.innerHTML=fn,Gt=l(),f(ie.$$.fragment),qt=l(),Re=m("p"),Re.innerHTML=gn,Nt=l(),f(de.$$.fragment),Vt=l(),xe=m("p"),xe.innerHTML=Mn,Et=l(),Ue=m("p"),Ue.innerHTML=bn,Ht=l(),f(Ce.$$.fragment),Qt=l(),f(Fe.$$.fragment),St=l(),ze=m("ul"),ze.innerHTML=_n,Yt=l(),f(We.$$.fragment),At=l(),I=m("div"),f(Ie.$$.fragment),uo=l(),rt=m("p"),rt.innerHTML=yn,fo=l(),lt=m("p"),lt.innerHTML=Tn,go=l(),f(ce.$$.fragment),Pt=l(),f(Ze.$$.fragment),Ot=l(),j=m("div"),f(Be.$$.fragment),Mo=l(),it=m("p"),it.textContent=wn,bo=l(),dt=m("p"),dt.innerHTML=kn,_o=l(),ct=m("p"),ct.innerHTML=Xn,yo=l(),ae=m("div"),f(Ge.$$.fragment),To=l(),pt=m("p"),pt.innerHTML=Ln,wo=l(),f(pe.$$.fragment),Dt=l(),f(qe.$$.fragment),Kt=l(),$=m("div"),f(Ne.$$.fragment),ko=l(),mt=m("p"),mt.innerHTML=vn,Xo=l(),ht=m("p"),ht.innerHTML=Jn,Lo=l(),ut=m("p"),ut.innerHTML=jn,vo=l(),Q=m("div"),f(Ve.$$.fragment),Jo=l(),ft=m("p"),ft.innerHTML=$n,jo=l(),f(me.$$.fragment),$o=l(),f(he.$$.fragment),eo=l(),f(Ee.$$.fragment),to=l(),R=m("div"),f(He.$$.fragment),Ro=l(),gt=m("p"),gt.innerHTML=Rn,xo=l(),Mt=m("p"),Mt.innerHTML=xn,Uo=l(),bt=m("p"),bt.innerHTML=Un,Co=l(),S=m("div"),f(Qe.$$.fragment),Fo=l(),_t=m("p"),_t.innerHTML=Cn,zo=l(),f(ue.$$.fragment),Wo=l(),f(fe.$$.fragment),oo=l(),f(Se.$$.fragment),no=l(),x=m("div"),f(Ye.$$.fragment),Io=l(),yt=m("p"),yt.textContent=Fn,Zo=l(),Tt=m("p"),Tt.innerHTML=zn,Bo=l(),wt=m("p"),wt.innerHTML=Wn,Go=l(),W=m("div"),f(Ae.$$.fragment),qo=l(),kt=m("p"),kt.innerHTML=In,No=l(),f(ge.$$.fragment),Vo=l(),f(Me.$$.fragment),Eo=l(),f(be.$$.fragment),so=l(),f(Pe.$$.fragment),ao=l(),U=m("div"),f(Oe.$$.fragment),Ho=l(),Xt=m("p"),Xt.textContent=Zn,Qo=l(),Lt=m("p"),Lt.innerHTML=Bn,So=l(),vt=m("p"),vt.innerHTML=Gn,Yo=l(),Y=m("div"),f(De.$$.fragment),Ao=l(),Jt=m("p"),Jt.innerHTML=qn,Po=l(),f(_e.$$.fragment),Oo=l(),f(ye.$$.fragment),ro=l(),f(Ke.$$.fragment),lo=l(),C=m("div"),f(et.$$.fragment),Do=l(),jt=m("p"),jt.textContent=Nn,Ko=l(),$t=m("p"),$t.innerHTML=Vn,en=l(),Rt=m("p"),Rt.innerHTML=En,tn=l(),A=m("div"),f(tt.$$.fragment),on=l(),xt=m("p"),xt.innerHTML=Hn,nn=l(),f(Te.$$.fragment),sn=l(),f(we.$$.fragment),io=l(),f(ot.$$.fragment),co=l(),F=m("div"),f(nt.$$.fragment),an=l(),Ut=m("p"),Ut.innerHTML=Qn,rn=l(),Ct=m("p"),Ct.innerHTML=Sn,ln=l(),Ft=m("p"),Ft.innerHTML=Yn,dn=l(),P=m("div"),f(st.$$.fragment),cn=l(),zt=m("p"),zt.innerHTML=An,pn=l(),f(ke.$$.fragment),mn=l(),f(Xe.$$.fragment),po=l(),f(at.$$.fragment),mo=l(),It=m("p"),this.h()},l(e){const s=os("svelte-u9bgzb",document.head);t=h(s,"META",{name:!0,content:!0}),s.forEach(r),p=i(e),o=h(e,"P",{}),v(o).forEach(r),a=i(e),T=h(e,"P",{"data-svelte-h":!0}),w(T)!=="svelte-5m6y2k"&&(T.innerHTML=n),u=i(e),X=h(e,"DIV",{style:!0,"data-svelte-h":!0}),w(X)!=="svelte-ithiq1"&&(X.innerHTML=Wt),Je=i(e),g(se.$$.fragment,e),Zt=i(e),je=h(e,"P",{"data-svelte-h":!0}),w(je)!=="svelte-u0qc8w"&&(je.innerHTML=un),Bt=i(e),$e=h(e,"P",{"data-svelte-h":!0}),w($e)!=="svelte-qmwz83"&&($e.innerHTML=fn),Gt=i(e),g(ie.$$.fragment,e),qt=i(e),Re=h(e,"P",{"data-svelte-h":!0}),w(Re)!=="svelte-10lshn2"&&(Re.innerHTML=gn),Nt=i(e),g(de.$$.fragment,e),Vt=i(e),xe=h(e,"P",{"data-svelte-h":!0}),w(xe)!=="svelte-nf5ooi"&&(xe.innerHTML=Mn),Et=i(e),Ue=h(e,"P",{"data-svelte-h":!0}),w(Ue)!=="svelte-w36i1c"&&(Ue.innerHTML=bn),Ht=i(e),g(Ce.$$.fragment,e),Qt=i(e),g(Fe.$$.fragment,e),St=i(e),ze=h(e,"UL",{"data-svelte-h":!0}),w(ze)!=="svelte-17o1h29"&&(ze.innerHTML=_n),Yt=i(e),g(We.$$.fragment,e),At=i(e),I=h(e,"DIV",{class:!0});var D=v(I);g(Ie.$$.fragment,D),uo=i(D),rt=h(D,"P",{"data-svelte-h":!0}),w(rt)!=="svelte-3a7wgr"&&(rt.innerHTML=yn),fo=i(D),lt=h(D,"P",{"data-svelte-h":!0}),w(lt)!=="svelte-1ek1ss9"&&(lt.innerHTML=Tn),go=i(D),g(ce.$$.fragment,D),D.forEach(r),Pt=i(e),g(Ze.$$.fragment,e),Ot=i(e),j=h(e,"DIV",{class:!0});var Z=v(j);g(Be.$$.fragment,Z),Mo=i(Z),it=h(Z,"P",{"data-svelte-h":!0}),w(it)!=="svelte-1w77jls"&&(it.textContent=wn),bo=i(Z),dt=h(Z,"P",{"data-svelte-h":!0}),w(dt)!=="svelte-q52n56"&&(dt.innerHTML=kn),_o=i(Z),ct=h(Z,"P",{"data-svelte-h":!0}),w(ct)!=="svelte-hswkmf"&&(ct.innerHTML=Xn),yo=i(Z),ae=h(Z,"DIV",{class:!0});var le=v(ae);g(Ge.$$.fragment,le),To=i(le),pt=h(le,"P",{"data-svelte-h":!0}),w(pt)!=="svelte-15ia9y3"&&(pt.innerHTML=Ln),wo=i(le),g(pe.$$.fragment,le),le.forEach(r),Z.forEach(r),Dt=i(e),g(qe.$$.fragment,e),Kt=i(e),$=h(e,"DIV",{class:!0});var B=v($);g(Ne.$$.fragment,B),ko=i(B),mt=h(B,"P",{"data-svelte-h":!0}),w(mt)!=="svelte-18zyi1r"&&(mt.innerHTML=vn),Xo=i(B),ht=h(B,"P",{"data-svelte-h":!0}),w(ht)!=="svelte-q52n56"&&(ht.innerHTML=Jn),Lo=i(B),ut=h(B,"P",{"data-svelte-h":!0}),w(ut)!=="svelte-hswkmf"&&(ut.innerHTML=jn),vo=i(B),Q=h(B,"DIV",{class:!0});var K=v(Q);g(Ve.$$.fragment,K),Jo=i(K),ft=h(K,"P",{"data-svelte-h":!0}),w(ft)!=="svelte-riqzbr"&&(ft.innerHTML=$n),jo=i(K),g(me.$$.fragment,K),$o=i(K),g(he.$$.fragment,K),K.forEach(r),B.forEach(r),eo=i(e),g(Ee.$$.fragment,e),to=i(e),R=h(e,"DIV",{class:!0});var G=v(R);g(He.$$.fragment,G),Ro=i(G),gt=h(G,"P",{"data-svelte-h":!0}),w(gt)!=="svelte-a00t64"&&(gt.innerHTML=Rn),xo=i(G),Mt=h(G,"P",{"data-svelte-h":!0}),w(Mt)!=="svelte-q52n56"&&(Mt.innerHTML=xn),Uo=i(G),bt=h(G,"P",{"data-svelte-h":!0}),w(bt)!=="svelte-hswkmf"&&(bt.innerHTML=Un),Co=i(G),S=h(G,"DIV",{class:!0});var ee=v(S);g(Qe.$$.fragment,ee),Fo=i(ee),_t=h(ee,"P",{"data-svelte-h":!0}),w(_t)!=="svelte-15hdy0v"&&(_t.innerHTML=Cn),zo=i(ee),g(ue.$$.fragment,ee),Wo=i(ee),g(fe.$$.fragment,ee),ee.forEach(r),G.forEach(r),oo=i(e),g(Se.$$.fragment,e),no=i(e),x=h(e,"DIV",{class:!0});var q=v(x);g(Ye.$$.fragment,q),Io=i(q),yt=h(q,"P",{"data-svelte-h":!0}),w(yt)!=="svelte-19228m"&&(yt.textContent=Fn),Zo=i(q),Tt=h(q,"P",{"data-svelte-h":!0}),w(Tt)!=="svelte-q52n56"&&(Tt.innerHTML=zn),Bo=i(q),wt=h(q,"P",{"data-svelte-h":!0}),w(wt)!=="svelte-hswkmf"&&(wt.innerHTML=Wn),Go=i(q),W=h(q,"DIV",{class:!0});var N=v(W);g(Ae.$$.fragment,N),qo=i(N),kt=h(N,"P",{"data-svelte-h":!0}),w(kt)!=="svelte-va5mo7"&&(kt.innerHTML=In),No=i(N),g(ge.$$.fragment,N),Vo=i(N),g(Me.$$.fragment,N),Eo=i(N),g(be.$$.fragment,N),N.forEach(r),q.forEach(r),so=i(e),g(Pe.$$.fragment,e),ao=i(e),U=h(e,"DIV",{class:!0});var V=v(U);g(Oe.$$.fragment,V),Ho=i(V),Xt=h(V,"P",{"data-svelte-h":!0}),w(Xt)!=="svelte-12m07dp"&&(Xt.textContent=Zn),Qo=i(V),Lt=h(V,"P",{"data-svelte-h":!0}),w(Lt)!=="svelte-q52n56"&&(Lt.innerHTML=Bn),So=i(V),vt=h(V,"P",{"data-svelte-h":!0}),w(vt)!=="svelte-hswkmf"&&(vt.innerHTML=Gn),Yo=i(V),Y=h(V,"DIV",{class:!0});var te=v(Y);g(De.$$.fragment,te),Ao=i(te),Jt=h(te,"P",{"data-svelte-h":!0}),w(Jt)!=="svelte-1oqo917"&&(Jt.innerHTML=qn),Po=i(te),g(_e.$$.fragment,te),Oo=i(te),g(ye.$$.fragment,te),te.forEach(r),V.forEach(r),ro=i(e),g(Ke.$$.fragment,e),lo=i(e),C=h(e,"DIV",{class:!0});var E=v(C);g(et.$$.fragment,E),Do=i(E),jt=h(E,"P",{"data-svelte-h":!0}),w(jt)!=="svelte-swrfk"&&(jt.textContent=Nn),Ko=i(E),$t=h(E,"P",{"data-svelte-h":!0}),w($t)!=="svelte-q52n56"&&($t.innerHTML=Vn),en=i(E),Rt=h(E,"P",{"data-svelte-h":!0}),w(Rt)!=="svelte-hswkmf"&&(Rt.innerHTML=En),tn=i(E),A=h(E,"DIV",{class:!0});var oe=v(A);g(tt.$$.fragment,oe),on=i(oe),xt=h(oe,"P",{"data-svelte-h":!0}),w(xt)!=="svelte-qcyh1n"&&(xt.innerHTML=Hn),nn=i(oe),g(Te.$$.fragment,oe),sn=i(oe),g(we.$$.fragment,oe),oe.forEach(r),E.forEach(r),io=i(e),g(ot.$$.fragment,e),co=i(e),F=h(e,"DIV",{class:!0});var H=v(F);g(nt.$$.fragment,H),an=i(H),Ut=h(H,"P",{"data-svelte-h":!0}),w(Ut)!=="svelte-1spa9pl"&&(Ut.innerHTML=Qn),rn=i(H),Ct=h(H,"P",{"data-svelte-h":!0}),w(Ct)!=="svelte-q52n56"&&(Ct.innerHTML=Sn),ln=i(H),Ft=h(H,"P",{"data-svelte-h":!0}),w(Ft)!=="svelte-hswkmf"&&(Ft.innerHTML=Yn),dn=i(H),P=h(H,"DIV",{class:!0});var ne=v(P);g(st.$$.fragment,ne),cn=i(ne),zt=h(ne,"P",{"data-svelte-h":!0}),w(zt)!=="svelte-7ldept"&&(zt.innerHTML=An),pn=i(ne),g(ke.$$.fragment,ne),mn=i(ne),g(Xe.$$.fragment,ne),ne.forEach(r),H.forEach(r),po=i(e),g(at.$$.fragment,e),mo=i(e),It=h(e,"P",{}),v(It).forEach(r),this.h()},h(){J(t,"name","hf:doc:metadata"),J(t,"content",Js),ns(X,"float","right"),J(I,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(ae,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(j,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(Q,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J($,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(S,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(R,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(W,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(x,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(Y,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(U,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(A,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(C,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(P,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(F,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,s){d(document.head,t),c(e,p,s),c(e,o,s),c(e,a,s),c(e,T,s),c(e,u,s),c(e,X,s),c(e,Je,s),M(se,e,s),c(e,Zt,s),c(e,je,s),c(e,Bt,s),c(e,$e,s),c(e,Gt,s),M(ie,e,s),c(e,qt,s),c(e,Re,s),c(e,Nt,s),M(de,e,s),c(e,Vt,s),c(e,xe,s),c(e,Et,s),c(e,Ue,s),c(e,Ht,s),M(Ce,e,s),c(e,Qt,s),M(Fe,e,s),c(e,St,s),c(e,ze,s),c(e,Yt,s),M(We,e,s),c(e,At,s),c(e,I,s),M(Ie,I,null),d(I,uo),d(I,rt),d(I,fo),d(I,lt),d(I,go),M(ce,I,null),c(e,Pt,s),M(Ze,e,s),c(e,Ot,s),c(e,j,s),M(Be,j,null),d(j,Mo),d(j,it),d(j,bo),d(j,dt),d(j,_o),d(j,ct),d(j,yo),d(j,ae),M(Ge,ae,null),d(ae,To),d(ae,pt),d(ae,wo),M(pe,ae,null),c(e,Dt,s),M(qe,e,s),c(e,Kt,s),c(e,$,s),M(Ne,$,null),d($,ko),d($,mt),d($,Xo),d($,ht),d($,Lo),d($,ut),d($,vo),d($,Q),M(Ve,Q,null),d(Q,Jo),d(Q,ft),d(Q,jo),M(me,Q,null),d(Q,$o),M(he,Q,null),c(e,eo,s),M(Ee,e,s),c(e,to,s),c(e,R,s),M(He,R,null),d(R,Ro),d(R,gt),d(R,xo),d(R,Mt),d(R,Uo),d(R,bt),d(R,Co),d(R,S),M(Qe,S,null),d(S,Fo),d(S,_t),d(S,zo),M(ue,S,null),d(S,Wo),M(fe,S,null),c(e,oo,s),M(Se,e,s),c(e,no,s),c(e,x,s),M(Ye,x,null),d(x,Io),d(x,yt),d(x,Zo),d(x,Tt),d(x,Bo),d(x,wt),d(x,Go),d(x,W),M(Ae,W,null),d(W,qo),d(W,kt),d(W,No),M(ge,W,null),d(W,Vo),M(Me,W,null),d(W,Eo),M(be,W,null),c(e,so,s),M(Pe,e,s),c(e,ao,s),c(e,U,s),M(Oe,U,null),d(U,Ho),d(U,Xt),d(U,Qo),d(U,Lt),d(U,So),d(U,vt),d(U,Yo),d(U,Y),M(De,Y,null),d(Y,Ao),d(Y,Jt),d(Y,Po),M(_e,Y,null),d(Y,Oo),M(ye,Y,null),c(e,ro,s),M(Ke,e,s),c(e,lo,s),c(e,C,s),M(et,C,null),d(C,Do),d(C,jt),d(C,Ko),d(C,$t),d(C,en),d(C,Rt),d(C,tn),d(C,A),M(tt,A,null),d(A,on),d(A,xt),d(A,nn),M(Te,A,null),d(A,sn),M(we,A,null),c(e,io,s),M(ot,e,s),c(e,co,s),c(e,F,s),M(nt,F,null),d(F,an),d(F,Ut),d(F,rn),d(F,Ct),d(F,ln),d(F,Ft),d(F,dn),d(F,P),M(st,P,null),d(P,cn),d(P,zt),d(P,pn),M(ke,P,null),d(P,mn),M(Xe,P,null),c(e,po,s),M(at,e,s),c(e,mo,s),c(e,It,s),ho=!0},p(e,[s]){const D={};s&2&&(D.$$scope={dirty:s,ctx:e}),ie.$set(D);const Z={};s&2&&(Z.$$scope={dirty:s,ctx:e}),de.$set(Z);const le={};s&2&&(le.$$scope={dirty:s,ctx:e}),ce.$set(le);const B={};s&2&&(B.$$scope={dirty:s,ctx:e}),pe.$set(B);const K={};s&2&&(K.$$scope={dirty:s,ctx:e}),me.$set(K);const G={};s&2&&(G.$$scope={dirty:s,ctx:e}),he.$set(G);const ee={};s&2&&(ee.$$scope={dirty:s,ctx:e}),ue.$set(ee);const q={};s&2&&(q.$$scope={dirty:s,ctx:e}),fe.$set(q);const N={};s&2&&(N.$$scope={dirty:s,ctx:e}),ge.$set(N);const V={};s&2&&(V.$$scope={dirty:s,ctx:e}),Me.$set(V);const te={};s&2&&(te.$$scope={dirty:s,ctx:e}),be.$set(te);const E={};s&2&&(E.$$scope={dirty:s,ctx:e}),_e.$set(E);const oe={};s&2&&(oe.$$scope={dirty:s,ctx:e}),ye.$set(oe);const H={};s&2&&(H.$$scope={dirty:s,ctx:e}),Te.$set(H);const ne={};s&2&&(ne.$$scope={dirty:s,ctx:e}),we.$set(ne);const Pn={};s&2&&(Pn.$$scope={dirty:s,ctx:e}),ke.$set(Pn);const On={};s&2&&(On.$$scope={dirty:s,ctx:e}),Xe.$set(On)},i(e){ho||(b(se.$$.fragment,e),b(ie.$$.fragment,e),b(de.$$.fragment,e),b(Ce.$$.fragment,e),b(Fe.$$.fragment,e),b(We.$$.fragment,e),b(Ie.$$.fragment,e),b(ce.$$.fragment,e),b(Ze.$$.fragment,e),b(Be.$$.fragment,e),b(Ge.$$.fragment,e),b(pe.$$.fragment,e),b(qe.$$.fragment,e),b(Ne.$$.fragment,e),b(Ve.$$.fragment,e),b(me.$$.fragment,e),b(he.$$.fragment,e),b(Ee.$$.fragment,e),b(He.$$.fragment,e),b(Qe.$$.fragment,e),b(ue.$$.fragment,e),b(fe.$$.fragment,e),b(Se.$$.fragment,e),b(Ye.$$.fragment,e),b(Ae.$$.fragment,e),b(ge.$$.fragment,e),b(Me.$$.fragment,e),b(be.$$.fragment,e),b(Pe.$$.fragment,e),b(Oe.$$.fragment,e),b(De.$$.fragment,e),b(_e.$$.fragment,e),b(ye.$$.fragment,e),b(Ke.$$.fragment,e),b(et.$$.fragment,e),b(tt.$$.fragment,e),b(Te.$$.fragment,e),b(we.$$.fragment,e),b(ot.$$.fragment,e),b(nt.$$.fragment,e),b(st.$$.fragment,e),b(ke.$$.fragment,e),b(Xe.$$.fragment,e),b(at.$$.fragment,e),ho=!0)},o(e){_(se.$$.fragment,e),_(ie.$$.fragment,e),_(de.$$.fragment,e),_(Ce.$$.fragment,e),_(Fe.$$.fragment,e),_(We.$$.fragment,e),_(Ie.$$.fragment,e),_(ce.$$.fragment,e),_(Ze.$$.fragment,e),_(Be.$$.fragment,e),_(Ge.$$.fragment,e),_(pe.$$.fragment,e),_(qe.$$.fragment,e),_(Ne.$$.fragment,e),_(Ve.$$.fragment,e),_(me.$$.fragment,e),_(he.$$.fragment,e),_(Ee.$$.fragment,e),_(He.$$.fragment,e),_(Qe.$$.fragment,e),_(ue.$$.fragment,e),_(fe.$$.fragment,e),_(Se.$$.fragment,e),_(Ye.$$.fragment,e),_(Ae.$$.fragment,e),_(ge.$$.fragment,e),_(Me.$$.fragment,e),_(be.$$.fragment,e),_(Pe.$$.fragment,e),_(Oe.$$.fragment,e),_(De.$$.fragment,e),_(_e.$$.fragment,e),_(ye.$$.fragment,e),_(Ke.$$.fragment,e),_(et.$$.fragment,e),_(tt.$$.fragment,e),_(Te.$$.fragment,e),_(we.$$.fragment,e),_(ot.$$.fragment,e),_(nt.$$.fragment,e),_(st.$$.fragment,e),_(ke.$$.fragment,e),_(Xe.$$.fragment,e),_(at.$$.fragment,e),ho=!1},d(e){e&&(r(p),r(o),r(a),r(T),r(u),r(X),r(Je),r(Zt),r(je),r(Bt),r($e),r(Gt),r(qt),r(Re),r(Nt),r(Vt),r(xe),r(Et),r(Ue),r(Ht),r(Qt),r(St),r(ze),r(Yt),r(At),r(I),r(Pt),r(Ot),r(j),r(Dt),r(Kt),r($),r(eo),r(to),r(R),r(oo),r(no),r(x),r(so),r(ao),r(U),r(ro),r(lo),r(C),r(io),r(co),r(F),r(po),r(mo),r(It)),r(t),y(se,e),y(ie,e),y(de,e),y(Ce,e),y(Fe,e),y(We,e),y(Ie),y(ce),y(Ze,e),y(Be),y(Ge),y(pe),y(qe,e),y(Ne),y(Ve),y(me),y(he),y(Ee,e),y(He),y(Qe),y(ue),y(fe),y(Se,e),y(Ye),y(Ae),y(ge),y(Me),y(be),y(Pe,e),y(Oe),y(De),y(_e),y(ye),y(Ke,e),y(et),y(tt),y(Te),y(we),y(ot,e),y(nt),y(st),y(ke),y(Xe),y(at,e)}}}const Js='{"title":"XLM-RoBERTa-XL","local":"xlm-roberta-xl","sections":[{"title":"Notes","local":"notes","sections":[],"depth":2},{"title":"XLMRobertaXLConfig","local":"transformers.XLMRobertaXLConfig","sections":[],"depth":2},{"title":"XLMRobertaXLModel","local":"transformers.XLMRobertaXLModel","sections":[],"depth":2},{"title":"XLMRobertaXLForCausalLM","local":"transformers.XLMRobertaXLForCausalLM","sections":[],"depth":2},{"title":"XLMRobertaXLForMaskedLM","local":"transformers.XLMRobertaXLForMaskedLM","sections":[],"depth":2},{"title":"XLMRobertaXLForSequenceClassification","local":"transformers.XLMRobertaXLForSequenceClassification","sections":[],"depth":2},{"title":"XLMRobertaXLForMultipleChoice","local":"transformers.XLMRobertaXLForMultipleChoice","sections":[],"depth":2},{"title":"XLMRobertaXLForTokenClassification","local":"transformers.XLMRobertaXLForTokenClassification","sections":[],"depth":2},{"title":"XLMRobertaXLForQuestionAnswering","local":"transformers.XLMRobertaXLForQuestionAnswering","sections":[],"depth":2}],"depth":1}';function js(k){return Kn(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class Is extends es{constructor(t){super(),ts(this,t,js,vs,Dn,{})}}export{Is as component};
