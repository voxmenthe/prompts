import{s as da,o as ca,n as J}from"../chunks/scheduler.18a86fab.js";import{S as pa,i as ma,g as c,s as r,r as u,A as ha,h as p,f as l,c as a,j as v,x as y,u as f,k as $,l as ua,y as s,a as m,v as g,d as _,t as b,w as T}from"../chunks/index.98837b22.js";import{T as Te}from"../chunks/Tip.77304350.js";import{D as B}from"../chunks/Docstring.a1ef7999.js";import{C as D}from"../chunks/CodeBlock.8d0c2e8a.js";import{E as ye}from"../chunks/ExampleCodeBlock.8c3ee1f9.js";import{H,E as fa}from"../chunks/getInferenceSnippets.06c2775f.js";import{H as ga,a as ir}from"../chunks/HfOption.6641485e.js";function _a(w){let t,h="Click on the BERT models in the right sidebar for more examples of how to apply BERT to different language tasks.";return{c(){t=c("p"),t.textContent=h},l(o){t=p(o,"P",{"data-svelte-h":!0}),y(t)!=="svelte-1kq8hf9"&&(t.textContent=h)},m(o,d){m(o,t,d)},p:J,d(o){o&&l(t)}}}function ba(w){let t,h;return t=new D({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwcGlwZWxpbmUlMEElMEFwaXBlbGluZSUyMCUzRCUyMHBpcGVsaW5lKCUwQSUyMCUyMCUyMCUyMHRhc2slM0QlMjJmaWxsLW1hc2slMjIlMkMlMEElMjAlMjAlMjAlMjBtb2RlbCUzRCUyMmdvb2dsZS1iZXJ0JTJGYmVydC1iYXNlLXVuY2FzZWQlMjIlMkMlMEElMjAlMjAlMjAlMjBkdHlwZSUzRHRvcmNoLmZsb2F0MTYlMkMlMEElMjAlMjAlMjAlMjBkZXZpY2UlM0QwJTBBKSUwQXBpcGVsaW5lKCUyMlBsYW50cyUyMGNyZWF0ZSUyMCU1Qk1BU0slNUQlMjB0aHJvdWdoJTIwYSUyMHByb2Nlc3MlMjBrbm93biUyMGFzJTIwcGhvdG9zeW50aGVzaXMuJTIyKQ==",highlighted:`<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> pipeline

pipeline = pipeline(
    task=<span class="hljs-string">&quot;fill-mask&quot;</span>,
    model=<span class="hljs-string">&quot;google-bert/bert-base-uncased&quot;</span>,
    dtype=torch.float16,
    device=<span class="hljs-number">0</span>
)
pipeline(<span class="hljs-string">&quot;Plants create [MASK] through a process known as photosynthesis.&quot;</span>)`,wrap:!1}}),{c(){u(t.$$.fragment)},l(o){f(t.$$.fragment,o)},m(o,d){g(t,o,d),h=!0},p:J,i(o){h||(_(t.$$.fragment,o),h=!0)},o(o){b(t.$$.fragment,o),h=!1},d(o){T(t,o)}}}function Ta(w){let t,h;return t=new D({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b01vZGVsRm9yTWFza2VkTE0lMkMlMjBBdXRvVG9rZW5pemVyJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTBBJTIwJTIwJTIwJTIwJTIyZ29vZ2xlLWJlcnQlMkZiZXJ0LWJhc2UtdW5jYXNlZCUyMiUyQyUwQSklMEFtb2RlbCUyMCUzRCUyMEF1dG9Nb2RlbEZvck1hc2tlZExNLmZyb21fcHJldHJhaW5lZCglMEElMjAlMjAlMjAlMjAlMjJnb29nbGUtYmVydCUyRmJlcnQtYmFzZS11bmNhc2VkJTIyJTJDJTBBJTIwJTIwJTIwJTIwZHR5cGUlM0R0b3JjaC5mbG9hdDE2JTJDJTBBJTIwJTIwJTIwJTIwZGV2aWNlX21hcCUzRCUyMmF1dG8lMjIlMkMlMEElMjAlMjAlMjAlMjBhdHRuX2ltcGxlbWVudGF0aW9uJTNEJTIyc2RwYSUyMiUwQSklMEFpbnB1dHMlMjAlM0QlMjB0b2tlbml6ZXIoJTIyUGxhbnRzJTIwY3JlYXRlJTIwJTVCTUFTSyU1RCUyMHRocm91Z2glMjBhJTIwcHJvY2VzcyUyMGtub3duJTIwYXMlMjBwaG90b3N5bnRoZXNpcy4lMjIlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKS50byhtb2RlbC5kZXZpY2UpJTBBJTBBd2l0aCUyMHRvcmNoLm5vX2dyYWQoKSUzQSUwQSUyMCUyMCUyMCUyMG91dHB1dHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyklMEElMjAlMjAlMjAlMjBwcmVkaWN0aW9ucyUyMCUzRCUyMG91dHB1dHMubG9naXRzJTBBJTBBbWFza2VkX2luZGV4JTIwJTNEJTIwdG9yY2gud2hlcmUoaW5wdXRzJTVCJ2lucHV0X2lkcyclNUQlMjAlM0QlM0QlMjB0b2tlbml6ZXIubWFza190b2tlbl9pZCklNUIxJTVEJTBBcHJlZGljdGVkX3Rva2VuX2lkJTIwJTNEJTIwcHJlZGljdGlvbnMlNUIwJTJDJTIwbWFza2VkX2luZGV4JTVELmFyZ21heChkaW0lM0QtMSklMEFwcmVkaWN0ZWRfdG9rZW4lMjAlM0QlMjB0b2tlbml6ZXIuZGVjb2RlKHByZWRpY3RlZF90b2tlbl9pZCklMEElMEFwcmludChmJTIyVGhlJTIwcHJlZGljdGVkJTIwdG9rZW4lMjBpcyUzQSUyMCU3QnByZWRpY3RlZF90b2tlbiU3RCUyMik=",highlighted:`<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoModelForMaskedLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    <span class="hljs-string">&quot;google-bert/bert-base-uncased&quot;</span>,
)
model = AutoModelForMaskedLM.from_pretrained(
    <span class="hljs-string">&quot;google-bert/bert-base-uncased&quot;</span>,
    dtype=torch.float16,
    device_map=<span class="hljs-string">&quot;auto&quot;</span>,
    attn_implementation=<span class="hljs-string">&quot;sdpa&quot;</span>
)
inputs = tokenizer(<span class="hljs-string">&quot;Plants create [MASK] through a process known as photosynthesis.&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>).to(model.device)

<span class="hljs-keyword">with</span> torch.no_grad():
    outputs = model(**inputs)
    predictions = outputs.logits

masked_index = torch.where(inputs[<span class="hljs-string">&#x27;input_ids&#x27;</span>] == tokenizer.mask_token_id)[<span class="hljs-number">1</span>]
predicted_token_id = predictions[<span class="hljs-number">0</span>, masked_index].argmax(dim=-<span class="hljs-number">1</span>)
predicted_token = tokenizer.decode(predicted_token_id)

<span class="hljs-built_in">print</span>(<span class="hljs-string">f&quot;The predicted token is: <span class="hljs-subst">{predicted_token}</span>&quot;</span>)`,wrap:!1}}),{c(){u(t.$$.fragment)},l(o){f(t.$$.fragment,o)},m(o,d){g(t,o,d),h=!0},p:J,i(o){h||(_(t.$$.fragment,o),h=!0)},o(o){b(t.$$.fragment,o),h=!1},d(o){T(t,o)}}}function ya(w){let t,h;return t=new D({props:{code:"ZWNobyUyMC1lJTIwJTIyUGxhbnRzJTIwY3JlYXRlJTIwJTVCTUFTSyU1RCUyMHRocm91Z2glMjBhJTIwcHJvY2VzcyUyMGtub3duJTIwYXMlMjBwaG90b3N5bnRoZXNpcy4lMjIlMjAlN0MlMjB0cmFuc2Zvcm1lcnMlMjBydW4lMjAtLXRhc2slMjBmaWxsLW1hc2slMjAtLW1vZGVsJTIwZ29vZ2xlLWJlcnQlMkZiZXJ0LWJhc2UtdW5jYXNlZCUyMC0tZGV2aWNlJTIwMA==",highlighted:'<span class="hljs-built_in">echo</span> -e <span class="hljs-string">&quot;Plants create [MASK] through a process known as photosynthesis.&quot;</span> | transformers run --task fill-mask --model google-bert/bert-base-uncased --device 0',wrap:!1}}),{c(){u(t.$$.fragment)},l(o){f(t.$$.fragment,o)},m(o,d){g(t,o,d),h=!0},p:J,i(o){h||(_(t.$$.fragment,o),h=!0)},o(o){b(t.$$.fragment,o),h=!1},d(o){T(t,o)}}}function Ma(w){let t,h,o,d,k,n;return t=new ir({props:{id:"usage",option:"Pipeline",$$slots:{default:[ba]},$$scope:{ctx:w}}}),o=new ir({props:{id:"usage",option:"AutoModel",$$slots:{default:[Ta]},$$scope:{ctx:w}}}),k=new ir({props:{id:"usage",option:"transformers CLI",$$slots:{default:[ya]},$$scope:{ctx:w}}}),{c(){u(t.$$.fragment),h=r(),u(o.$$.fragment),d=r(),u(k.$$.fragment)},l(M){f(t.$$.fragment,M),h=a(M),f(o.$$.fragment,M),d=a(M),f(k.$$.fragment,M)},m(M,j){g(t,M,j),m(M,h,j),g(o,M,j),m(M,d,j),g(k,M,j),n=!0},p(M,j){const Rn={};j&2&&(Rn.$$scope={dirty:j,ctx:M}),t.$set(Rn);const Oe={};j&2&&(Oe.$$scope={dirty:j,ctx:M}),o.$set(Oe);const me={};j&2&&(me.$$scope={dirty:j,ctx:M}),k.$set(me)},i(M){n||(_(t.$$.fragment,M),_(o.$$.fragment,M),_(k.$$.fragment,M),n=!0)},o(M){b(t.$$.fragment,M),b(o.$$.fragment,M),b(k.$$.fragment,M),n=!1},d(M){M&&(l(h),l(d)),T(t,M),T(o,M),T(k,M)}}}function ka(w){let t,h="Examples:",o,d,k;return d=new D({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEJlcnRDb25maWclMkMlMjBCZXJ0TW9kZWwlMEElMEElMjMlMjBJbml0aWFsaXppbmclMjBhJTIwQkVSVCUyMGdvb2dsZS1iZXJ0JTJGYmVydC1iYXNlLXVuY2FzZWQlMjBzdHlsZSUyMGNvbmZpZ3VyYXRpb24lMEFjb25maWd1cmF0aW9uJTIwJTNEJTIwQmVydENvbmZpZygpJTBBJTBBJTIzJTIwSW5pdGlhbGl6aW5nJTIwYSUyMG1vZGVsJTIwKHdpdGglMjByYW5kb20lMjB3ZWlnaHRzKSUyMGZyb20lMjB0aGUlMjBnb29nbGUtYmVydCUyRmJlcnQtYmFzZS11bmNhc2VkJTIwc3R5bGUlMjBjb25maWd1cmF0aW9uJTBBbW9kZWwlMjAlM0QlMjBCZXJ0TW9kZWwoY29uZmlndXJhdGlvbiklMEElMEElMjMlMjBBY2Nlc3NpbmclMjB0aGUlMjBtb2RlbCUyMGNvbmZpZ3VyYXRpb24lMEFjb25maWd1cmF0aW9uJTIwJTNEJTIwbW9kZWwuY29uZmln",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> BertConfig, BertModel

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a BERT google-bert/bert-base-uncased style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = BertConfig()

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a model (with random weights) from the google-bert/bert-base-uncased style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = BertModel(configuration)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Accessing the model configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = model.config`,wrap:!1}}),{c(){t=c("p"),t.textContent=h,o=r(),u(d.$$.fragment)},l(n){t=p(n,"P",{"data-svelte-h":!0}),y(t)!=="svelte-kvfsh7"&&(t.textContent=h),o=a(n),f(d.$$.fragment,n)},m(n,M){m(n,t,M),m(n,o,M),g(d,n,M),k=!0},p:J,i(n){k||(_(d.$$.fragment,n),k=!0)},o(n){b(d.$$.fragment,n),k=!1},d(n){n&&(l(t),l(o)),T(d,n)}}}function wa(w){let t,h=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=h},l(o){t=p(o,"P",{"data-svelte-h":!0}),y(t)!=="svelte-fincs2"&&(t.innerHTML=h)},m(o,d){m(o,t,d)},p:J,d(o){o&&l(t)}}}function va(w){let t,h=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=h},l(o){t=p(o,"P",{"data-svelte-h":!0}),y(t)!=="svelte-fincs2"&&(t.innerHTML=h)},m(o,d){m(o,t,d)},p:J,d(o){o&&l(t)}}}function $a(w){let t,h="Example:",o,d,k;return d=new D({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBCZXJ0Rm9yUHJlVHJhaW5pbmclMEFpbXBvcnQlMjB0b3JjaCUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMmdvb2dsZS1iZXJ0JTJGYmVydC1iYXNlLXVuY2FzZWQlMjIpJTBBbW9kZWwlMjAlM0QlMjBCZXJ0Rm9yUHJlVHJhaW5pbmcuZnJvbV9wcmV0cmFpbmVkKCUyMmdvb2dsZS1iZXJ0JTJGYmVydC1iYXNlLXVuY2FzZWQlMjIpJTBBJTBBaW5wdXRzJTIwJTNEJTIwdG9rZW5pemVyKCUyMkhlbGxvJTJDJTIwbXklMjBkb2clMjBpcyUyMGN1dGUlMjIlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQW91dHB1dHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyklMEElMEFwcmVkaWN0aW9uX2xvZ2l0cyUyMCUzRCUyMG91dHB1dHMucHJlZGljdGlvbl9sb2dpdHMlMEFzZXFfcmVsYXRpb25zaGlwX2xvZ2l0cyUyMCUzRCUyMG91dHB1dHMuc2VxX3JlbGF0aW9uc2hpcF9sb2dpdHM=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, BertForPreTraining
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;google-bert/bert-base-uncased&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = BertForPreTraining.from_pretrained(<span class="hljs-string">&quot;google-bert/bert-base-uncased&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs)

<span class="hljs-meta">&gt;&gt;&gt; </span>prediction_logits = outputs.prediction_logits
<span class="hljs-meta">&gt;&gt;&gt; </span>seq_relationship_logits = outputs.seq_relationship_logits`,wrap:!1}}),{c(){t=c("p"),t.textContent=h,o=r(),u(d.$$.fragment)},l(n){t=p(n,"P",{"data-svelte-h":!0}),y(t)!=="svelte-11lpom8"&&(t.textContent=h),o=a(n),f(d.$$.fragment,n)},m(n,M){m(n,t,M),m(n,o,M),g(d,n,M),k=!0},p:J,i(n){k||(_(d.$$.fragment,n),k=!0)},o(n){b(d.$$.fragment,n),k=!1},d(n){n&&(l(t),l(o)),T(d,n)}}}function Ba(w){let t,h=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=h},l(o){t=p(o,"P",{"data-svelte-h":!0}),y(t)!=="svelte-fincs2"&&(t.innerHTML=h)},m(o,d){m(o,t,d)},p:J,d(o){o&&l(t)}}}function Ja(w){let t,h="Example:",o,d,k;return d=new D({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Rva2VuaXplciUyQyUyMEJlcnRMTUhlYWRNb2RlbCUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMmdvb2dsZS1iZXJ0JTJGYmVydC1iYXNlLXVuY2FzZWQlMjIpJTBBbW9kZWwlMjAlM0QlMjBCZXJ0TE1IZWFkTW9kZWwuZnJvbV9wcmV0cmFpbmVkKCUyMmdvb2dsZS1iZXJ0JTJGYmVydC1iYXNlLXVuY2FzZWQlMjIpJTBBJTBBaW5wdXRzJTIwJTNEJTIwdG9rZW5pemVyKCUyMkhlbGxvJTJDJTIwbXklMjBkb2clMjBpcyUyMGN1dGUlMjIlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQW91dHB1dHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyUyQyUyMGxhYmVscyUzRGlucHV0cyU1QiUyMmlucHV0X2lkcyUyMiU1RCklMEFsb3NzJTIwJTNEJTIwb3V0cHV0cy5sb3NzJTBBbG9naXRzJTIwJTNEJTIwb3V0cHV0cy5sb2dpdHM=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, BertLMHeadModel

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;google-bert/bert-base-uncased&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = BertLMHeadModel.from_pretrained(<span class="hljs-string">&quot;google-bert/bert-base-uncased&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs, labels=inputs[<span class="hljs-string">&quot;input_ids&quot;</span>])
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = outputs.loss
<span class="hljs-meta">&gt;&gt;&gt; </span>logits = outputs.logits`,wrap:!1}}),{c(){t=c("p"),t.textContent=h,o=r(),u(d.$$.fragment)},l(n){t=p(n,"P",{"data-svelte-h":!0}),y(t)!=="svelte-11lpom8"&&(t.textContent=h),o=a(n),f(d.$$.fragment,n)},m(n,M){m(n,t,M),m(n,o,M),g(d,n,M),k=!0},p:J,i(n){k||(_(d.$$.fragment,n),k=!0)},o(n){b(d.$$.fragment,n),k=!1},d(n){n&&(l(t),l(o)),T(d,n)}}}function ja(w){let t,h=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=h},l(o){t=p(o,"P",{"data-svelte-h":!0}),y(t)!=="svelte-fincs2"&&(t.innerHTML=h)},m(o,d){m(o,t,d)},p:J,d(o){o&&l(t)}}}function za(w){let t,h="Example:",o,d,k;return d=new D({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBCZXJ0Rm9yTWFza2VkTE0lMEFpbXBvcnQlMjB0b3JjaCUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMmdvb2dsZS1iZXJ0JTJGYmVydC1iYXNlLXVuY2FzZWQlMjIpJTBBbW9kZWwlMjAlM0QlMjBCZXJ0Rm9yTWFza2VkTE0uZnJvbV9wcmV0cmFpbmVkKCUyMmdvb2dsZS1iZXJ0JTJGYmVydC1iYXNlLXVuY2FzZWQlMjIpJTBBJTBBaW5wdXRzJTIwJTNEJTIwdG9rZW5pemVyKCUyMlRoZSUyMGNhcGl0YWwlMjBvZiUyMEZyYW5jZSUyMGlzJTIwJTNDbWFzayUzRS4lMjIlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQSUwQXdpdGglMjB0b3JjaC5ub19ncmFkKCklM0ElMEElMjAlMjAlMjAlMjBsb2dpdHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cykubG9naXRzJTBBJTBBJTIzJTIwcmV0cmlldmUlMjBpbmRleCUyMG9mJTIwJTNDbWFzayUzRSUwQW1hc2tfdG9rZW5faW5kZXglMjAlM0QlMjAoaW5wdXRzLmlucHV0X2lkcyUyMCUzRCUzRCUyMHRva2VuaXplci5tYXNrX3Rva2VuX2lkKSU1QjAlNUQubm9uemVybyhhc190dXBsZSUzRFRydWUpJTVCMCU1RCUwQSUwQXByZWRpY3RlZF90b2tlbl9pZCUyMCUzRCUyMGxvZ2l0cyU1QjAlMkMlMjBtYXNrX3Rva2VuX2luZGV4JTVELmFyZ21heChheGlzJTNELTEpJTBBdG9rZW5pemVyLmRlY29kZShwcmVkaWN0ZWRfdG9rZW5faWQpJTBBJTBBbGFiZWxzJTIwJTNEJTIwdG9rZW5pemVyKCUyMlRoZSUyMGNhcGl0YWwlMjBvZiUyMEZyYW5jZSUyMGlzJTIwUGFyaXMuJTIyJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiklNUIlMjJpbnB1dF9pZHMlMjIlNUQlMEElMjMlMjBtYXNrJTIwbGFiZWxzJTIwb2YlMjBub24tJTNDbWFzayUzRSUyMHRva2VucyUwQWxhYmVscyUyMCUzRCUyMHRvcmNoLndoZXJlKGlucHV0cy5pbnB1dF9pZHMlMjAlM0QlM0QlMjB0b2tlbml6ZXIubWFza190b2tlbl9pZCUyQyUyMGxhYmVscyUyQyUyMC0xMDApJTBBJTBBb3V0cHV0cyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzJTJDJTIwbGFiZWxzJTNEbGFiZWxzKSUwQXJvdW5kKG91dHB1dHMubG9zcy5pdGVtKCklMkMlMjAyKQ==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, BertForMaskedLM
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;google-bert/bert-base-uncased&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = BertForMaskedLM.from_pretrained(<span class="hljs-string">&quot;google-bert/bert-base-uncased&quot;</span>)

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
...`,wrap:!1}}),{c(){t=c("p"),t.textContent=h,o=r(),u(d.$$.fragment)},l(n){t=p(n,"P",{"data-svelte-h":!0}),y(t)!=="svelte-11lpom8"&&(t.textContent=h),o=a(n),f(d.$$.fragment,n)},m(n,M){m(n,t,M),m(n,o,M),g(d,n,M),k=!0},p:J,i(n){k||(_(d.$$.fragment,n),k=!0)},o(n){b(d.$$.fragment,n),k=!1},d(n){n&&(l(t),l(o)),T(d,n)}}}function xa(w){let t,h=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=h},l(o){t=p(o,"P",{"data-svelte-h":!0}),y(t)!=="svelte-fincs2"&&(t.innerHTML=h)},m(o,d){m(o,t,d)},p:J,d(o){o&&l(t)}}}function Ca(w){let t,h="Example:",o,d,k;return d=new D({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBCZXJ0Rm9yTmV4dFNlbnRlbmNlUHJlZGljdGlvbiUwQWltcG9ydCUyMHRvcmNoJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIyZ29vZ2xlLWJlcnQlMkZiZXJ0LWJhc2UtdW5jYXNlZCUyMiklMEFtb2RlbCUyMCUzRCUyMEJlcnRGb3JOZXh0U2VudGVuY2VQcmVkaWN0aW9uLmZyb21fcHJldHJhaW5lZCglMjJnb29nbGUtYmVydCUyRmJlcnQtYmFzZS11bmNhc2VkJTIyKSUwQSUwQXByb21wdCUyMCUzRCUyMCUyMkluJTIwSXRhbHklMkMlMjBwaXp6YSUyMHNlcnZlZCUyMGluJTIwZm9ybWFsJTIwc2V0dGluZ3MlMkMlMjBzdWNoJTIwYXMlMjBhdCUyMGElMjByZXN0YXVyYW50JTJDJTIwaXMlMjBwcmVzZW50ZWQlMjB1bnNsaWNlZC4lMjIlMEFuZXh0X3NlbnRlbmNlJTIwJTNEJTIwJTIyVGhlJTIwc2t5JTIwaXMlMjBibHVlJTIwZHVlJTIwdG8lMjB0aGUlMjBzaG9ydGVyJTIwd2F2ZWxlbmd0aCUyMG9mJTIwYmx1ZSUyMGxpZ2h0LiUyMiUwQWVuY29kaW5nJTIwJTNEJTIwdG9rZW5pemVyKHByb21wdCUyQyUyMG5leHRfc2VudGVuY2UlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQSUwQW91dHB1dHMlMjAlM0QlMjBtb2RlbCgqKmVuY29kaW5nJTJDJTIwbGFiZWxzJTNEdG9yY2guTG9uZ1RlbnNvciglNUIxJTVEKSklMEFsb2dpdHMlMjAlM0QlMjBvdXRwdXRzLmxvZ2l0cyUwQWFzc2VydCUyMGxvZ2l0cyU1QjAlMkMlMjAwJTVEJTIwJTNDJTIwbG9naXRzJTVCMCUyQyUyMDElNUQlMjAlMjAlMjMlMjBuZXh0JTIwc2VudGVuY2UlMjB3YXMlMjByYW5kb20=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, BertForNextSentencePrediction
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;google-bert/bert-base-uncased&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = BertForNextSentencePrediction.from_pretrained(<span class="hljs-string">&quot;google-bert/bert-base-uncased&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>prompt = <span class="hljs-string">&quot;In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>next_sentence = <span class="hljs-string">&quot;The sky is blue due to the shorter wavelength of blue light.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>encoding = tokenizer(prompt, next_sentence, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**encoding, labels=torch.LongTensor([<span class="hljs-number">1</span>]))
<span class="hljs-meta">&gt;&gt;&gt; </span>logits = outputs.logits
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">assert</span> logits[<span class="hljs-number">0</span>, <span class="hljs-number">0</span>] &lt; logits[<span class="hljs-number">0</span>, <span class="hljs-number">1</span>]  <span class="hljs-comment"># next sentence was random</span>`,wrap:!1}}),{c(){t=c("p"),t.textContent=h,o=r(),u(d.$$.fragment)},l(n){t=p(n,"P",{"data-svelte-h":!0}),y(t)!=="svelte-11lpom8"&&(t.textContent=h),o=a(n),f(d.$$.fragment,n)},m(n,M){m(n,t,M),m(n,o,M),g(d,n,M),k=!0},p:J,i(n){k||(_(d.$$.fragment,n),k=!0)},o(n){b(d.$$.fragment,n),k=!1},d(n){n&&(l(t),l(o)),T(d,n)}}}function Ua(w){let t,h=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=h},l(o){t=p(o,"P",{"data-svelte-h":!0}),y(t)!=="svelte-fincs2"&&(t.innerHTML=h)},m(o,d){m(o,t,d)},p:J,d(o){o&&l(t)}}}function Fa(w){let t,h="Example of single-label classification:",o,d,k;return d=new D({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Rva2VuaXplciUyQyUyMEJlcnRGb3JTZXF1ZW5jZUNsYXNzaWZpY2F0aW9uJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIyZ29vZ2xlLWJlcnQlMkZiZXJ0LWJhc2UtdW5jYXNlZCUyMiklMEFtb2RlbCUyMCUzRCUyMEJlcnRGb3JTZXF1ZW5jZUNsYXNzaWZpY2F0aW9uLmZyb21fcHJldHJhaW5lZCglMjJnb29nbGUtYmVydCUyRmJlcnQtYmFzZS11bmNhc2VkJTIyKSUwQSUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplciglMjJIZWxsbyUyQyUyMG15JTIwZG9nJTIwaXMlMjBjdXRlJTIyJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiklMEElMEF3aXRoJTIwdG9yY2gubm9fZ3JhZCgpJTNBJTBBJTIwJTIwJTIwJTIwbG9naXRzJTIwJTNEJTIwbW9kZWwoKippbnB1dHMpLmxvZ2l0cyUwQSUwQXByZWRpY3RlZF9jbGFzc19pZCUyMCUzRCUyMGxvZ2l0cy5hcmdtYXgoKS5pdGVtKCklMEFtb2RlbC5jb25maWcuaWQybGFiZWwlNUJwcmVkaWN0ZWRfY2xhc3NfaWQlNUQlMEElMEElMjMlMjBUbyUyMHRyYWluJTIwYSUyMG1vZGVsJTIwb24lMjAlNjBudW1fbGFiZWxzJTYwJTIwY2xhc3NlcyUyQyUyMHlvdSUyMGNhbiUyMHBhc3MlMjAlNjBudW1fbGFiZWxzJTNEbnVtX2xhYmVscyU2MCUyMHRvJTIwJTYwLmZyb21fcHJldHJhaW5lZCguLi4pJTYwJTBBbnVtX2xhYmVscyUyMCUzRCUyMGxlbihtb2RlbC5jb25maWcuaWQybGFiZWwpJTBBbW9kZWwlMjAlM0QlMjBCZXJ0Rm9yU2VxdWVuY2VDbGFzc2lmaWNhdGlvbi5mcm9tX3ByZXRyYWluZWQoJTIyZ29vZ2xlLWJlcnQlMkZiZXJ0LWJhc2UtdW5jYXNlZCUyMiUyQyUyMG51bV9sYWJlbHMlM0RudW1fbGFiZWxzKSUwQSUwQWxhYmVscyUyMCUzRCUyMHRvcmNoLnRlbnNvciglNUIxJTVEKSUwQWxvc3MlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyUyQyUyMGxhYmVscyUzRGxhYmVscykubG9zcyUwQXJvdW5kKGxvc3MuaXRlbSgpJTJDJTIwMik=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, BertForSequenceClassification

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;google-bert/bert-base-uncased&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = BertForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;google-bert/bert-base-uncased&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    logits = model(**inputs).logits

<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_class_id = logits.argmax().item()
<span class="hljs-meta">&gt;&gt;&gt; </span>model.config.id2label[predicted_class_id]
...

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># To train a model on \`num_labels\` classes, you can pass \`num_labels=num_labels\` to \`.from_pretrained(...)\`</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>num_labels = <span class="hljs-built_in">len</span>(model.config.id2label)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = BertForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;google-bert/bert-base-uncased&quot;</span>, num_labels=num_labels)

<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.tensor([<span class="hljs-number">1</span>])
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = model(**inputs, labels=labels).loss
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">round</span>(loss.item(), <span class="hljs-number">2</span>)
...`,wrap:!1}}),{c(){t=c("p"),t.textContent=h,o=r(),u(d.$$.fragment)},l(n){t=p(n,"P",{"data-svelte-h":!0}),y(t)!=="svelte-ykxpe4"&&(t.textContent=h),o=a(n),f(d.$$.fragment,n)},m(n,M){m(n,t,M),m(n,o,M),g(d,n,M),k=!0},p:J,i(n){k||(_(d.$$.fragment,n),k=!0)},o(n){b(d.$$.fragment,n),k=!1},d(n){n&&(l(t),l(o)),T(d,n)}}}function Ia(w){let t,h="Example of multi-label classification:",o,d,k;return d=new D({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Rva2VuaXplciUyQyUyMEJlcnRGb3JTZXF1ZW5jZUNsYXNzaWZpY2F0aW9uJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIyZ29vZ2xlLWJlcnQlMkZiZXJ0LWJhc2UtdW5jYXNlZCUyMiklMEFtb2RlbCUyMCUzRCUyMEJlcnRGb3JTZXF1ZW5jZUNsYXNzaWZpY2F0aW9uLmZyb21fcHJldHJhaW5lZCglMjJnb29nbGUtYmVydCUyRmJlcnQtYmFzZS11bmNhc2VkJTIyJTJDJTIwcHJvYmxlbV90eXBlJTNEJTIybXVsdGlfbGFiZWxfY2xhc3NpZmljYXRpb24lMjIpJTBBJTBBaW5wdXRzJTIwJTNEJTIwdG9rZW5pemVyKCUyMkhlbGxvJTJDJTIwbXklMjBkb2clMjBpcyUyMGN1dGUlMjIlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQSUwQXdpdGglMjB0b3JjaC5ub19ncmFkKCklM0ElMEElMjAlMjAlMjAlMjBsb2dpdHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cykubG9naXRzJTBBJTBBcHJlZGljdGVkX2NsYXNzX2lkcyUyMCUzRCUyMHRvcmNoLmFyYW5nZSgwJTJDJTIwbG9naXRzLnNoYXBlJTVCLTElNUQpJTVCdG9yY2guc2lnbW9pZChsb2dpdHMpLnNxdWVlemUoZGltJTNEMCklMjAlM0UlMjAwLjUlNUQlMEElMEElMjMlMjBUbyUyMHRyYWluJTIwYSUyMG1vZGVsJTIwb24lMjAlNjBudW1fbGFiZWxzJTYwJTIwY2xhc3NlcyUyQyUyMHlvdSUyMGNhbiUyMHBhc3MlMjAlNjBudW1fbGFiZWxzJTNEbnVtX2xhYmVscyU2MCUyMHRvJTIwJTYwLmZyb21fcHJldHJhaW5lZCguLi4pJTYwJTBBbnVtX2xhYmVscyUyMCUzRCUyMGxlbihtb2RlbC5jb25maWcuaWQybGFiZWwpJTBBbW9kZWwlMjAlM0QlMjBCZXJ0Rm9yU2VxdWVuY2VDbGFzc2lmaWNhdGlvbi5mcm9tX3ByZXRyYWluZWQoJTBBJTIwJTIwJTIwJTIwJTIyZ29vZ2xlLWJlcnQlMkZiZXJ0LWJhc2UtdW5jYXNlZCUyMiUyQyUyMG51bV9sYWJlbHMlM0RudW1fbGFiZWxzJTJDJTIwcHJvYmxlbV90eXBlJTNEJTIybXVsdGlfbGFiZWxfY2xhc3NpZmljYXRpb24lMjIlMEEpJTBBJTBBbGFiZWxzJTIwJTNEJTIwdG9yY2guc3VtKCUwQSUyMCUyMCUyMCUyMHRvcmNoLm5uLmZ1bmN0aW9uYWwub25lX2hvdChwcmVkaWN0ZWRfY2xhc3NfaWRzJTVCTm9uZSUyQyUyMCUzQSU1RC5jbG9uZSgpJTJDJTIwbnVtX2NsYXNzZXMlM0RudW1fbGFiZWxzKSUyQyUyMGRpbSUzRDElMEEpLnRvKHRvcmNoLmZsb2F0KSUwQWxvc3MlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyUyQyUyMGxhYmVscyUzRGxhYmVscykubG9zcw==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, BertForSequenceClassification

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;google-bert/bert-base-uncased&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = BertForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;google-bert/bert-base-uncased&quot;</span>, problem_type=<span class="hljs-string">&quot;multi_label_classification&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    logits = model(**inputs).logits

<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_class_ids = torch.arange(<span class="hljs-number">0</span>, logits.shape[-<span class="hljs-number">1</span>])[torch.sigmoid(logits).squeeze(dim=<span class="hljs-number">0</span>) &gt; <span class="hljs-number">0.5</span>]

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># To train a model on \`num_labels\` classes, you can pass \`num_labels=num_labels\` to \`.from_pretrained(...)\`</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>num_labels = <span class="hljs-built_in">len</span>(model.config.id2label)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = BertForSequenceClassification.from_pretrained(
<span class="hljs-meta">... </span>    <span class="hljs-string">&quot;google-bert/bert-base-uncased&quot;</span>, num_labels=num_labels, problem_type=<span class="hljs-string">&quot;multi_label_classification&quot;</span>
<span class="hljs-meta">... </span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.<span class="hljs-built_in">sum</span>(
<span class="hljs-meta">... </span>    torch.nn.functional.one_hot(predicted_class_ids[<span class="hljs-literal">None</span>, :].clone(), num_classes=num_labels), dim=<span class="hljs-number">1</span>
<span class="hljs-meta">... </span>).to(torch.<span class="hljs-built_in">float</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = model(**inputs, labels=labels).loss`,wrap:!1}}),{c(){t=c("p"),t.textContent=h,o=r(),u(d.$$.fragment)},l(n){t=p(n,"P",{"data-svelte-h":!0}),y(t)!=="svelte-1l8e32d"&&(t.textContent=h),o=a(n),f(d.$$.fragment,n)},m(n,M){m(n,t,M),m(n,o,M),g(d,n,M),k=!0},p:J,i(n){k||(_(d.$$.fragment,n),k=!0)},o(n){b(d.$$.fragment,n),k=!1},d(n){n&&(l(t),l(o)),T(d,n)}}}function qa(w){let t,h=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=h},l(o){t=p(o,"P",{"data-svelte-h":!0}),y(t)!=="svelte-fincs2"&&(t.innerHTML=h)},m(o,d){m(o,t,d)},p:J,d(o){o&&l(t)}}}function Za(w){let t,h="Example:",o,d,k;return d=new D({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBCZXJ0Rm9yTXVsdGlwbGVDaG9pY2UlMEFpbXBvcnQlMjB0b3JjaCUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMmdvb2dsZS1iZXJ0JTJGYmVydC1iYXNlLXVuY2FzZWQlMjIpJTBBbW9kZWwlMjAlM0QlMjBCZXJ0Rm9yTXVsdGlwbGVDaG9pY2UuZnJvbV9wcmV0cmFpbmVkKCUyMmdvb2dsZS1iZXJ0JTJGYmVydC1iYXNlLXVuY2FzZWQlMjIpJTBBJTBBcHJvbXB0JTIwJTNEJTIwJTIySW4lMjBJdGFseSUyQyUyMHBpenphJTIwc2VydmVkJTIwaW4lMjBmb3JtYWwlMjBzZXR0aW5ncyUyQyUyMHN1Y2glMjBhcyUyMGF0JTIwYSUyMHJlc3RhdXJhbnQlMkMlMjBpcyUyMHByZXNlbnRlZCUyMHVuc2xpY2VkLiUyMiUwQWNob2ljZTAlMjAlM0QlMjAlMjJJdCUyMGlzJTIwZWF0ZW4lMjB3aXRoJTIwYSUyMGZvcmslMjBhbmQlMjBhJTIwa25pZmUuJTIyJTBBY2hvaWNlMSUyMCUzRCUyMCUyMkl0JTIwaXMlMjBlYXRlbiUyMHdoaWxlJTIwaGVsZCUyMGluJTIwdGhlJTIwaGFuZC4lMjIlMEFsYWJlbHMlMjAlM0QlMjB0b3JjaC50ZW5zb3IoMCkudW5zcXVlZXplKDApJTIwJTIwJTIzJTIwY2hvaWNlMCUyMGlzJTIwY29ycmVjdCUyMChhY2NvcmRpbmclMjB0byUyMFdpa2lwZWRpYSUyMCUzQikpJTJDJTIwYmF0Y2glMjBzaXplJTIwMSUwQSUwQWVuY29kaW5nJTIwJTNEJTIwdG9rZW5pemVyKCU1QnByb21wdCUyQyUyMHByb21wdCU1RCUyQyUyMCU1QmNob2ljZTAlMkMlMjBjaG9pY2UxJTVEJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiUyQyUyMHBhZGRpbmclM0RUcnVlKSUwQW91dHB1dHMlMjAlM0QlMjBtb2RlbCgqKiU3QmslM0ElMjB2LnVuc3F1ZWV6ZSgwKSUyMGZvciUyMGslMkMlMjB2JTIwaW4lMjBlbmNvZGluZy5pdGVtcygpJTdEJTJDJTIwbGFiZWxzJTNEbGFiZWxzKSUyMCUyMCUyMyUyMGJhdGNoJTIwc2l6ZSUyMGlzJTIwMSUwQSUwQSUyMyUyMHRoZSUyMGxpbmVhciUyMGNsYXNzaWZpZXIlMjBzdGlsbCUyMG5lZWRzJTIwdG8lMjBiZSUyMHRyYWluZWQlMEFsb3NzJTIwJTNEJTIwb3V0cHV0cy5sb3NzJTBBbG9naXRzJTIwJTNEJTIwb3V0cHV0cy5sb2dpdHM=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, BertForMultipleChoice
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;google-bert/bert-base-uncased&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = BertForMultipleChoice.from_pretrained(<span class="hljs-string">&quot;google-bert/bert-base-uncased&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>prompt = <span class="hljs-string">&quot;In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>choice0 = <span class="hljs-string">&quot;It is eaten with a fork and a knife.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>choice1 = <span class="hljs-string">&quot;It is eaten while held in the hand.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.tensor(<span class="hljs-number">0</span>).unsqueeze(<span class="hljs-number">0</span>)  <span class="hljs-comment"># choice0 is correct (according to Wikipedia ;)), batch size 1</span>

<span class="hljs-meta">&gt;&gt;&gt; </span>encoding = tokenizer([prompt, prompt], [choice0, choice1], return_tensors=<span class="hljs-string">&quot;pt&quot;</span>, padding=<span class="hljs-literal">True</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**{k: v.unsqueeze(<span class="hljs-number">0</span>) <span class="hljs-keyword">for</span> k, v <span class="hljs-keyword">in</span> encoding.items()}, labels=labels)  <span class="hljs-comment"># batch size is 1</span>

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># the linear classifier still needs to be trained</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = outputs.loss
<span class="hljs-meta">&gt;&gt;&gt; </span>logits = outputs.logits`,wrap:!1}}),{c(){t=c("p"),t.textContent=h,o=r(),u(d.$$.fragment)},l(n){t=p(n,"P",{"data-svelte-h":!0}),y(t)!=="svelte-11lpom8"&&(t.textContent=h),o=a(n),f(d.$$.fragment,n)},m(n,M){m(n,t,M),m(n,o,M),g(d,n,M),k=!0},p:J,i(n){k||(_(d.$$.fragment,n),k=!0)},o(n){b(d.$$.fragment,n),k=!1},d(n){n&&(l(t),l(o)),T(d,n)}}}function Wa(w){let t,h=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=h},l(o){t=p(o,"P",{"data-svelte-h":!0}),y(t)!=="svelte-fincs2"&&(t.innerHTML=h)},m(o,d){m(o,t,d)},p:J,d(o){o&&l(t)}}}function Na(w){let t,h="Example:",o,d,k;return d=new D({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBCZXJ0Rm9yVG9rZW5DbGFzc2lmaWNhdGlvbiUwQWltcG9ydCUyMHRvcmNoJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIyZ29vZ2xlLWJlcnQlMkZiZXJ0LWJhc2UtdW5jYXNlZCUyMiklMEFtb2RlbCUyMCUzRCUyMEJlcnRGb3JUb2tlbkNsYXNzaWZpY2F0aW9uLmZyb21fcHJldHJhaW5lZCglMjJnb29nbGUtYmVydCUyRmJlcnQtYmFzZS11bmNhc2VkJTIyKSUwQSUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplciglMEElMjAlMjAlMjAlMjAlMjJIdWdnaW5nRmFjZSUyMGlzJTIwYSUyMGNvbXBhbnklMjBiYXNlZCUyMGluJTIwUGFyaXMlMjBhbmQlMjBOZXclMjBZb3JrJTIyJTJDJTIwYWRkX3NwZWNpYWxfdG9rZW5zJTNERmFsc2UlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyJTBBKSUwQSUwQXdpdGglMjB0b3JjaC5ub19ncmFkKCklM0ElMEElMjAlMjAlMjAlMjBsb2dpdHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cykubG9naXRzJTBBJTBBcHJlZGljdGVkX3Rva2VuX2NsYXNzX2lkcyUyMCUzRCUyMGxvZ2l0cy5hcmdtYXgoLTEpJTBBJTBBJTIzJTIwTm90ZSUyMHRoYXQlMjB0b2tlbnMlMjBhcmUlMjBjbGFzc2lmaWVkJTIwcmF0aGVyJTIwdGhlbiUyMGlucHV0JTIwd29yZHMlMjB3aGljaCUyMG1lYW5zJTIwdGhhdCUwQSUyMyUyMHRoZXJlJTIwbWlnaHQlMjBiZSUyMG1vcmUlMjBwcmVkaWN0ZWQlMjB0b2tlbiUyMGNsYXNzZXMlMjB0aGFuJTIwd29yZHMuJTBBJTIzJTIwTXVsdGlwbGUlMjB0b2tlbiUyMGNsYXNzZXMlMjBtaWdodCUyMGFjY291bnQlMjBmb3IlMjB0aGUlMjBzYW1lJTIwd29yZCUwQXByZWRpY3RlZF90b2tlbnNfY2xhc3NlcyUyMCUzRCUyMCU1Qm1vZGVsLmNvbmZpZy5pZDJsYWJlbCU1QnQuaXRlbSgpJTVEJTIwZm9yJTIwdCUyMGluJTIwcHJlZGljdGVkX3Rva2VuX2NsYXNzX2lkcyU1QjAlNUQlNUQlMEFwcmVkaWN0ZWRfdG9rZW5zX2NsYXNzZXMlMEElMEFsYWJlbHMlMjAlM0QlMjBwcmVkaWN0ZWRfdG9rZW5fY2xhc3NfaWRzJTBBbG9zcyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzJTJDJTIwbGFiZWxzJTNEbGFiZWxzKS5sb3NzJTBBcm91bmQobG9zcy5pdGVtKCklMkMlMjAyKQ==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, BertForTokenClassification
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;google-bert/bert-base-uncased&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = BertForTokenClassification.from_pretrained(<span class="hljs-string">&quot;google-bert/bert-base-uncased&quot;</span>)

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
...`,wrap:!1}}),{c(){t=c("p"),t.textContent=h,o=r(),u(d.$$.fragment)},l(n){t=p(n,"P",{"data-svelte-h":!0}),y(t)!=="svelte-11lpom8"&&(t.textContent=h),o=a(n),f(d.$$.fragment,n)},m(n,M){m(n,t,M),m(n,o,M),g(d,n,M),k=!0},p:J,i(n){k||(_(d.$$.fragment,n),k=!0)},o(n){b(d.$$.fragment,n),k=!1},d(n){n&&(l(t),l(o)),T(d,n)}}}function Va(w){let t,h=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=h},l(o){t=p(o,"P",{"data-svelte-h":!0}),y(t)!=="svelte-fincs2"&&(t.innerHTML=h)},m(o,d){m(o,t,d)},p:J,d(o){o&&l(t)}}}function Ga(w){let t,h="Example:",o,d,k;return d=new D({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBCZXJ0Rm9yUXVlc3Rpb25BbnN3ZXJpbmclMEFpbXBvcnQlMjB0b3JjaCUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMmdvb2dsZS1iZXJ0JTJGYmVydC1iYXNlLXVuY2FzZWQlMjIpJTBBbW9kZWwlMjAlM0QlMjBCZXJ0Rm9yUXVlc3Rpb25BbnN3ZXJpbmcuZnJvbV9wcmV0cmFpbmVkKCUyMmdvb2dsZS1iZXJ0JTJGYmVydC1iYXNlLXVuY2FzZWQlMjIpJTBBJTBBcXVlc3Rpb24lMkMlMjB0ZXh0JTIwJTNEJTIwJTIyV2hvJTIwd2FzJTIwSmltJTIwSGVuc29uJTNGJTIyJTJDJTIwJTIySmltJTIwSGVuc29uJTIwd2FzJTIwYSUyMG5pY2UlMjBwdXBwZXQlMjIlMEElMEFpbnB1dHMlMjAlM0QlMjB0b2tlbml6ZXIocXVlc3Rpb24lMkMlMjB0ZXh0JTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiklMEF3aXRoJTIwdG9yY2gubm9fZ3JhZCgpJTNBJTBBJTIwJTIwJTIwJTIwb3V0cHV0cyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzKSUwQSUwQWFuc3dlcl9zdGFydF9pbmRleCUyMCUzRCUyMG91dHB1dHMuc3RhcnRfbG9naXRzLmFyZ21heCgpJTBBYW5zd2VyX2VuZF9pbmRleCUyMCUzRCUyMG91dHB1dHMuZW5kX2xvZ2l0cy5hcmdtYXgoKSUwQSUwQXByZWRpY3RfYW5zd2VyX3Rva2VucyUyMCUzRCUyMGlucHV0cy5pbnB1dF9pZHMlNUIwJTJDJTIwYW5zd2VyX3N0YXJ0X2luZGV4JTIwJTNBJTIwYW5zd2VyX2VuZF9pbmRleCUyMCUyQiUyMDElNUQlMEF0b2tlbml6ZXIuZGVjb2RlKHByZWRpY3RfYW5zd2VyX3Rva2VucyUyQyUyMHNraXBfc3BlY2lhbF90b2tlbnMlM0RUcnVlKSUwQSUwQSUyMyUyMHRhcmdldCUyMGlzJTIwJTIybmljZSUyMHB1cHBldCUyMiUwQXRhcmdldF9zdGFydF9pbmRleCUyMCUzRCUyMHRvcmNoLnRlbnNvciglNUIxNCU1RCklMEF0YXJnZXRfZW5kX2luZGV4JTIwJTNEJTIwdG9yY2gudGVuc29yKCU1QjE1JTVEKSUwQSUwQW91dHB1dHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyUyQyUyMHN0YXJ0X3Bvc2l0aW9ucyUzRHRhcmdldF9zdGFydF9pbmRleCUyQyUyMGVuZF9wb3NpdGlvbnMlM0R0YXJnZXRfZW5kX2luZGV4KSUwQWxvc3MlMjAlM0QlMjBvdXRwdXRzLmxvc3MlMEFyb3VuZChsb3NzLml0ZW0oKSUyQyUyMDIp",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, BertForQuestionAnswering
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;google-bert/bert-base-uncased&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = BertForQuestionAnswering.from_pretrained(<span class="hljs-string">&quot;google-bert/bert-base-uncased&quot;</span>)

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
...`,wrap:!1}}),{c(){t=c("p"),t.textContent=h,o=r(),u(d.$$.fragment)},l(n){t=p(n,"P",{"data-svelte-h":!0}),y(t)!=="svelte-11lpom8"&&(t.textContent=h),o=a(n),f(d.$$.fragment,n)},m(n,M){m(n,t,M),m(n,o,M),g(d,n,M),k=!0},p:J,i(n){k||(_(d.$$.fragment,n),k=!0)},o(n){b(d.$$.fragment,n),k=!1},d(n){n&&(l(t),l(o)),T(d,n)}}}function Ha(w){let t,h,o,d,k,n="<em>This model was released on 2018-10-11 and added to Hugging Face Transformers on 2020-11-16.</em>",M,j,Rn='<div class="flex flex-wrap space-x-1"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/> <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/></div>',Oe,me,En,De,lr='<a href="https://huggingface.co/papers/1810.04805" rel="nofollow">BERT</a> is a bidirectional transformer pretrained on unlabeled text to predict masked tokens in a sentence and to predict whether one sentence follows another. The main idea is that by randomly masking some tokens, the model can train on text to the left and right, giving it a more thorough understanding. BERT is also very versatile because its learned language representations can be adapted for other NLP tasks by fine-tuning an additional layer or head.',Qn,Ke,dr='You can find all the original BERT checkpoints under the <a href="https://huggingface.co/collections/google/bert-release-64ff5e7a4be99045d1896dbc" rel="nofollow">BERT</a> collection.',An,Be,Yn,et,cr='The example below demonstrates how to predict the <code>[MASK]</code> token with <a href="/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.Pipeline">Pipeline</a>, <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel">AutoModel</a>, and from the command line.',On,Je,Dn,tt,Kn,nt,pr="<li>Inputs should be padded on the right because BERT uses absolute position embeddings.</li>",eo,ot,to,L,st,Co,St,mr=`This is the configuration class to store the configuration of a <a href="/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertModel">BertModel</a> or a <code>TFBertModel</code>. It is used to
instantiate a BERT model according to the specified arguments, defining the model architecture. Instantiating a
configuration with the defaults will yield a similar configuration to that of the BERT
<a href="https://huggingface.co/google-bert/bert-base-uncased" rel="nofollow">google-bert/bert-base-uncased</a> architecture.`,Uo,Pt,hr=`Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> and can be used to control the model outputs. Read the
documentation from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> for more information.`,Fo,je,no,rt,oo,z,at,Io,Et,ur="Construct a BERT tokenizer. Based on WordPiece.",qo,Qt,fr=`This tokenizer inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer">PreTrainedTokenizer</a> which contains most of the main methods. Users should refer to
this superclass for more information regarding those methods.`,Zo,he,it,Wo,At,gr=`Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
adding special tokens. A BERT sequence has the following format:`,No,Yt,_r="<li>single sequence: <code>[CLS] X [SEP]</code></li> <li>pair of sequences: <code>[CLS] A [SEP] B [SEP]</code></li>",Vo,ze,lt,Go,Ot,br=`Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
special tokens using the tokenizer <code>prepare_for_model</code> method.`,Ho,ue,dt,Xo,Dt,Tr=`Create the token type IDs corresponding to the sequences passed. <a href="../glossary#token-type-ids">What are token type
IDs?</a>`,Lo,Kt,yr="Should be overridden in a subclass if the model has a special way of building those.",Ro,en,ct,so,pt,ro,R,mt,So,tn,Mr="Construct a “fast” BERT tokenizer (backed by HuggingFace’s <em>tokenizers</em> library). Based on WordPiece.",Po,nn,kr=`This tokenizer inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast">PreTrainedTokenizerFast</a> which contains most of the main methods. Users should
refer to this superclass for more information regarding those methods.`,Eo,fe,ht,Qo,on,wr=`Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
adding special tokens. A BERT sequence has the following format:`,Ao,sn,vr="<li>single sequence: <code>[CLS] X [SEP]</code></li> <li>pair of sequences: <code>[CLS] A [SEP] B [SEP]</code></li>",ao,ut,io,x,ft,Yo,rn,$r=`The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
cross-attention is added between the self-attention layers, following the architecture described in <a href="https://huggingface.co/papers/1706.03762" rel="nofollow">Attention is
all you need</a> by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.`,Oo,an,Br=`To behave as an decoder the model needs to be initialized with the <code>is_decoder</code> argument of the configuration set
to <code>True</code>. To be used in a Seq2Seq model, the model needs to initialized with both <code>is_decoder</code> argument and
<code>add_cross_attention</code> set to <code>True</code>; an <code>encoder_hidden_states</code> is then expected as an input to the forward pass.`,Do,ln,Jr=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Ko,dn,jr=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,es,ge,gt,ts,cn,zr='The <a href="/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertModel">BertModel</a> forward method, overrides the <code>__call__</code> special method.',ns,xe,lo,_t,co,U,bt,os,pn,xr="Bert Model with two heads on top as done during the pretraining: a <code>masked language modeling</code> head and a <code>next sentence prediction (classification)</code> head.",ss,mn,Cr=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,rs,hn,Ur=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,as,K,Tt,is,un,Fr='The <a href="/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertForPreTraining">BertForPreTraining</a> forward method, overrides the <code>__call__</code> special method.',ls,Ce,ds,Ue,po,yt,mo,F,Mt,cs,fn,Ir="Bert Model with a <code>language modeling</code> head on top for CLM fine-tuning.",ps,gn,qr=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,ms,_n,Zr=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,hs,ee,kt,us,bn,Wr='The <a href="/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertLMHeadModel">BertLMHeadModel</a> forward method, overrides the <code>__call__</code> special method.',fs,Fe,gs,Ie,ho,wt,uo,I,vt,_s,Tn,Nr="The Bert Model with a <code>language modeling</code> head on top.”",bs,yn,Vr=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Ts,Mn,Gr=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,ys,te,$t,Ms,kn,Hr='The <a href="/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertForMaskedLM">BertForMaskedLM</a> forward method, overrides the <code>__call__</code> special method.',ks,qe,ws,Ze,fo,Bt,go,q,Jt,vs,wn,Xr="Bert Model with a <code>next sentence prediction (classification)</code> head on top.",$s,vn,Lr=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Bs,$n,Rr=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Js,ne,jt,js,Bn,Sr='The <a href="/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertForNextSentencePrediction">BertForNextSentencePrediction</a> forward method, overrides the <code>__call__</code> special method.',zs,We,xs,Ne,_o,zt,bo,Z,xt,Cs,Jn,Pr=`Bert Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled
output) e.g. for GLUE tasks.`,Us,jn,Er=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Fs,zn,Qr=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Is,X,Ct,qs,xn,Ar='The <a href="/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertForSequenceClassification">BertForSequenceClassification</a> forward method, overrides the <code>__call__</code> special method.',Zs,Ve,Ws,Ge,Ns,He,To,Ut,yo,W,Ft,Vs,Cn,Yr=`The Bert Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
softmax) e.g. for RocStories/SWAG tasks.`,Gs,Un,Or=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Hs,Fn,Dr=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Xs,oe,It,Ls,In,Kr='The <a href="/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertForMultipleChoice">BertForMultipleChoice</a> forward method, overrides the <code>__call__</code> special method.',Rs,Xe,Ss,Le,Mo,qt,ko,N,Zt,Ps,qn,ea=`The Bert transformer with a token classification head on top (a linear layer on top of the hidden-states
output) e.g. for Named-Entity-Recognition (NER) tasks.`,Es,Zn,ta=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Qs,Wn,na=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,As,se,Wt,Ys,Nn,oa='The <a href="/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertForTokenClassification">BertForTokenClassification</a> forward method, overrides the <code>__call__</code> special method.',Os,Re,Ds,Se,wo,Nt,vo,V,Vt,Ks,Vn,sa=`The Bert transformer with a span classification head on top for extractive question-answering tasks like
SQuAD (a linear layer on top of the hidden-states output to compute <code>span start logits</code> and <code>span end logits</code>).`,er,Gn,ra=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,tr,Hn,aa=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,nr,re,Gt,or,Xn,ia='The <a href="/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertForQuestionAnswering">BertForQuestionAnswering</a> forward method, overrides the <code>__call__</code> special method.',sr,Pe,rr,Ee,$o,Ht,Bo,Me,Xt,ar,Ln,la='Output type of <a href="/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertForPreTraining">BertForPreTraining</a>.',Jo,Lt,jo,Sn,zo;return me=new H({props:{title:"BERT",local:"bert",headingTag:"h1"}}),Be=new Te({props:{warning:!1,$$slots:{default:[_a]},$$scope:{ctx:w}}}),Je=new ga({props:{id:"usage",options:["Pipeline","AutoModel","transformers CLI"],$$slots:{default:[Ma]},$$scope:{ctx:w}}}),tt=new H({props:{title:"Notes",local:"notes",headingTag:"h2"}}),ot=new H({props:{title:"BertConfig",local:"transformers.BertConfig",headingTag:"h2"}}),st=new B({props:{name:"class transformers.BertConfig",anchor:"transformers.BertConfig",parameters:[{name:"vocab_size",val:" = 30522"},{name:"hidden_size",val:" = 768"},{name:"num_hidden_layers",val:" = 12"},{name:"num_attention_heads",val:" = 12"},{name:"intermediate_size",val:" = 3072"},{name:"hidden_act",val:" = 'gelu'"},{name:"hidden_dropout_prob",val:" = 0.1"},{name:"attention_probs_dropout_prob",val:" = 0.1"},{name:"max_position_embeddings",val:" = 512"},{name:"type_vocab_size",val:" = 2"},{name:"initializer_range",val:" = 0.02"},{name:"layer_norm_eps",val:" = 1e-12"},{name:"pad_token_id",val:" = 0"},{name:"position_embedding_type",val:" = 'absolute'"},{name:"use_cache",val:" = True"},{name:"classifier_dropout",val:" = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.BertConfig.vocab_size",description:`<strong>vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 30522) &#x2014;
Vocabulary size of the BERT model. Defines the number of different tokens that can be represented by the
<code>inputs_ids</code> passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertModel">BertModel</a> or <code>TFBertModel</code>.`,name:"vocab_size"},{anchor:"transformers.BertConfig.hidden_size",description:`<strong>hidden_size</strong> (<code>int</code>, <em>optional</em>, defaults to 768) &#x2014;
Dimensionality of the encoder layers and the pooler layer.`,name:"hidden_size"},{anchor:"transformers.BertConfig.num_hidden_layers",description:`<strong>num_hidden_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 12) &#x2014;
Number of hidden layers in the Transformer encoder.`,name:"num_hidden_layers"},{anchor:"transformers.BertConfig.num_attention_heads",description:`<strong>num_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 12) &#x2014;
Number of attention heads for each attention layer in the Transformer encoder.`,name:"num_attention_heads"},{anchor:"transformers.BertConfig.intermediate_size",description:`<strong>intermediate_size</strong> (<code>int</code>, <em>optional</em>, defaults to 3072) &#x2014;
Dimensionality of the &#x201C;intermediate&#x201D; (often named feed-forward) layer in the Transformer encoder.`,name:"intermediate_size"},{anchor:"transformers.BertConfig.hidden_act",description:`<strong>hidden_act</strong> (<code>str</code> or <code>Callable</code>, <em>optional</em>, defaults to <code>&quot;gelu&quot;</code>) &#x2014;
The non-linear activation function (function or string) in the encoder and pooler. If string, <code>&quot;gelu&quot;</code>,
<code>&quot;relu&quot;</code>, <code>&quot;silu&quot;</code> and <code>&quot;gelu_new&quot;</code> are supported.`,name:"hidden_act"},{anchor:"transformers.BertConfig.hidden_dropout_prob",description:`<strong>hidden_dropout_prob</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.`,name:"hidden_dropout_prob"},{anchor:"transformers.BertConfig.attention_probs_dropout_prob",description:`<strong>attention_probs_dropout_prob</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout ratio for the attention probabilities.`,name:"attention_probs_dropout_prob"},{anchor:"transformers.BertConfig.max_position_embeddings",description:`<strong>max_position_embeddings</strong> (<code>int</code>, <em>optional</em>, defaults to 512) &#x2014;
The maximum sequence length that this model might ever be used with. Typically set this to something large
just in case (e.g., 512 or 1024 or 2048).`,name:"max_position_embeddings"},{anchor:"transformers.BertConfig.type_vocab_size",description:`<strong>type_vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 2) &#x2014;
The vocabulary size of the <code>token_type_ids</code> passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertModel">BertModel</a> or <code>TFBertModel</code>.`,name:"type_vocab_size"},{anchor:"transformers.BertConfig.initializer_range",description:`<strong>initializer_range</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.`,name:"initializer_range"},{anchor:"transformers.BertConfig.layer_norm_eps",description:`<strong>layer_norm_eps</strong> (<code>float</code>, <em>optional</em>, defaults to 1e-12) &#x2014;
The epsilon used by the layer normalization layers.`,name:"layer_norm_eps"},{anchor:"transformers.BertConfig.position_embedding_type",description:`<strong>position_embedding_type</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;absolute&quot;</code>) &#x2014;
Type of position embedding. Choose one of <code>&quot;absolute&quot;</code>, <code>&quot;relative_key&quot;</code>, <code>&quot;relative_key_query&quot;</code>. For
positional embeddings use <code>&quot;absolute&quot;</code>. For more information on <code>&quot;relative_key&quot;</code>, please refer to
<a href="https://huggingface.co/papers/1803.02155" rel="nofollow">Self-Attention with Relative Position Representations (Shaw et al.)</a>.
For more information on <code>&quot;relative_key_query&quot;</code>, please refer to <em>Method 4</em> in <a href="https://huggingface.co/papers/2009.13658" rel="nofollow">Improve Transformer Models
with Better Relative Position Embeddings (Huang et al.)</a>.`,name:"position_embedding_type"},{anchor:"transformers.BertConfig.is_decoder",description:`<strong>is_decoder</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether the model is used as a decoder or not. If <code>False</code>, the model is used as an encoder.`,name:"is_decoder"},{anchor:"transformers.BertConfig.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not the model should return the last key/values attentions (not used by all models). Only
relevant if <code>config.is_decoder=True</code>.`,name:"use_cache"},{anchor:"transformers.BertConfig.classifier_dropout",description:`<strong>classifier_dropout</strong> (<code>float</code>, <em>optional</em>) &#x2014;
The dropout ratio for the classification head.`,name:"classifier_dropout"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/bert/configuration_bert.py#L29"}}),je=new ye({props:{anchor:"transformers.BertConfig.example",$$slots:{default:[ka]},$$scope:{ctx:w}}}),rt=new H({props:{title:"BertTokenizer",local:"transformers.BertTokenizer",headingTag:"h2"}}),at=new B({props:{name:"class transformers.BertTokenizer",anchor:"transformers.BertTokenizer",parameters:[{name:"vocab_file",val:""},{name:"do_lower_case",val:" = True"},{name:"do_basic_tokenize",val:" = True"},{name:"never_split",val:" = None"},{name:"unk_token",val:" = '[UNK]'"},{name:"sep_token",val:" = '[SEP]'"},{name:"pad_token",val:" = '[PAD]'"},{name:"cls_token",val:" = '[CLS]'"},{name:"mask_token",val:" = '[MASK]'"},{name:"tokenize_chinese_chars",val:" = True"},{name:"strip_accents",val:" = None"},{name:"clean_up_tokenization_spaces",val:" = True"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.BertTokenizer.vocab_file",description:`<strong>vocab_file</strong> (<code>str</code>) &#x2014;
File containing the vocabulary.`,name:"vocab_file"},{anchor:"transformers.BertTokenizer.do_lower_case",description:`<strong>do_lower_case</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to lowercase the input when tokenizing.`,name:"do_lower_case"},{anchor:"transformers.BertTokenizer.do_basic_tokenize",description:`<strong>do_basic_tokenize</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to do basic tokenization before WordPiece.`,name:"do_basic_tokenize"},{anchor:"transformers.BertTokenizer.never_split",description:`<strong>never_split</strong> (<code>Iterable</code>, <em>optional</em>) &#x2014;
Collection of tokens which will never be split during tokenization. Only has an effect when
<code>do_basic_tokenize=True</code>`,name:"never_split"},{anchor:"transformers.BertTokenizer.unk_token",description:`<strong>unk_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;[UNK]&quot;</code>) &#x2014;
The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
token instead.`,name:"unk_token"},{anchor:"transformers.BertTokenizer.sep_token",description:`<strong>sep_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;[SEP]&quot;</code>) &#x2014;
The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
sequence classification or for a text and a question for question answering. It is also used as the last
token of a sequence built with special tokens.`,name:"sep_token"},{anchor:"transformers.BertTokenizer.pad_token",description:`<strong>pad_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;[PAD]&quot;</code>) &#x2014;
The token used for padding, for example when batching sequences of different lengths.`,name:"pad_token"},{anchor:"transformers.BertTokenizer.cls_token",description:`<strong>cls_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;[CLS]&quot;</code>) &#x2014;
The classifier token which is used when doing sequence classification (classification of the whole sequence
instead of per-token classification). It is the first token of the sequence when built with special tokens.`,name:"cls_token"},{anchor:"transformers.BertTokenizer.mask_token",description:`<strong>mask_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;[MASK]&quot;</code>) &#x2014;
The token used for masking values. This is the token used when training this model with masked language
modeling. This is the token which the model will try to predict.`,name:"mask_token"},{anchor:"transformers.BertTokenizer.tokenize_chinese_chars",description:`<strong>tokenize_chinese_chars</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to tokenize Chinese characters.</p>
<p>This should likely be deactivated for Japanese (see this
<a href="https://github.com/huggingface/transformers/issues/328" rel="nofollow">issue</a>).`,name:"tokenize_chinese_chars"},{anchor:"transformers.BertTokenizer.strip_accents",description:`<strong>strip_accents</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to strip all accents. If this option is not specified, then it will be determined by the
value for <code>lowercase</code> (as in the original BERT).`,name:"strip_accents"},{anchor:"transformers.BertTokenizer.clean_up_tokenization_spaces",description:`<strong>clean_up_tokenization_spaces</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to cleanup spaces after decoding, cleanup consists in removing potential artifacts like
extra spaces.`,name:"clean_up_tokenization_spaces"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/bert/tokenization_bert.py#L51"}}),it=new B({props:{name:"build_inputs_with_special_tokens",anchor:"transformers.BertTokenizer.build_inputs_with_special_tokens",parameters:[{name:"token_ids_0",val:": list"},{name:"token_ids_1",val:": typing.Optional[list[int]] = None"}],parametersDescription:[{anchor:"transformers.BertTokenizer.build_inputs_with_special_tokens.token_ids_0",description:`<strong>token_ids_0</strong> (<code>List[int]</code>) &#x2014;
List of IDs to which the special tokens will be added.`,name:"token_ids_0"},{anchor:"transformers.BertTokenizer.build_inputs_with_special_tokens.token_ids_1",description:`<strong>token_ids_1</strong> (<code>List[int]</code>, <em>optional</em>) &#x2014;
Optional second list of IDs for sequence pairs.`,name:"token_ids_1"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/bert/tokenization_bert.py#L186",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>List of <a href="../glossary#input-ids">input IDs</a> with the appropriate special tokens.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>List[int]</code></p>
`}}),lt=new B({props:{name:"get_special_tokens_mask",anchor:"transformers.BertTokenizer.get_special_tokens_mask",parameters:[{name:"token_ids_0",val:": list"},{name:"token_ids_1",val:": typing.Optional[list[int]] = None"},{name:"already_has_special_tokens",val:": bool = False"}],parametersDescription:[{anchor:"transformers.BertTokenizer.get_special_tokens_mask.token_ids_0",description:`<strong>token_ids_0</strong> (<code>List[int]</code>) &#x2014;
List of IDs.`,name:"token_ids_0"},{anchor:"transformers.BertTokenizer.get_special_tokens_mask.token_ids_1",description:`<strong>token_ids_1</strong> (<code>List[int]</code>, <em>optional</em>) &#x2014;
Optional second list of IDs for sequence pairs.`,name:"token_ids_1"},{anchor:"transformers.BertTokenizer.get_special_tokens_mask.already_has_special_tokens",description:`<strong>already_has_special_tokens</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not the token list is already formatted with special tokens for the model.`,name:"already_has_special_tokens"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/bert/tokenization_bert.py#L211",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>List[int]</code></p>
`}}),dt=new B({props:{name:"create_token_type_ids_from_sequences",anchor:"transformers.BertTokenizer.create_token_type_ids_from_sequences",parameters:[{name:"token_ids_0",val:": list"},{name:"token_ids_1",val:": typing.Optional[list[int]] = None"}],parametersDescription:[{anchor:"transformers.BertTokenizer.create_token_type_ids_from_sequences.token_ids_0",description:"<strong>token_ids_0</strong> (<code>list[int]</code>) &#x2014; The first tokenized sequence.",name:"token_ids_0"},{anchor:"transformers.BertTokenizer.create_token_type_ids_from_sequences.token_ids_1",description:"<strong>token_ids_1</strong> (<code>list[int]</code>, <em>optional</em>) &#x2014; The second tokenized sequence.",name:"token_ids_1"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L3432",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>The token type ids.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>list[int]</code></p>
`}}),ct=new B({props:{name:"save_vocabulary",anchor:"transformers.BertTokenizer.save_vocabulary",parameters:[{name:"save_directory",val:": str"},{name:"filename_prefix",val:": typing.Optional[str] = None"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/bert/tokenization_bert.py#L239"}}),pt=new H({props:{title:"BertTokenizerFast",local:"transformers.BertTokenizerFast",headingTag:"h2"}}),mt=new B({props:{name:"class transformers.BertTokenizerFast",anchor:"transformers.BertTokenizerFast",parameters:[{name:"vocab_file",val:" = None"},{name:"tokenizer_file",val:" = None"},{name:"do_lower_case",val:" = True"},{name:"unk_token",val:" = '[UNK]'"},{name:"sep_token",val:" = '[SEP]'"},{name:"pad_token",val:" = '[PAD]'"},{name:"cls_token",val:" = '[CLS]'"},{name:"mask_token",val:" = '[MASK]'"},{name:"tokenize_chinese_chars",val:" = True"},{name:"strip_accents",val:" = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.BertTokenizerFast.vocab_file",description:`<strong>vocab_file</strong> (<code>str</code>) &#x2014;
File containing the vocabulary.`,name:"vocab_file"},{anchor:"transformers.BertTokenizerFast.do_lower_case",description:`<strong>do_lower_case</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to lowercase the input when tokenizing.`,name:"do_lower_case"},{anchor:"transformers.BertTokenizerFast.unk_token",description:`<strong>unk_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;[UNK]&quot;</code>) &#x2014;
The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
token instead.`,name:"unk_token"},{anchor:"transformers.BertTokenizerFast.sep_token",description:`<strong>sep_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;[SEP]&quot;</code>) &#x2014;
The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
sequence classification or for a text and a question for question answering. It is also used as the last
token of a sequence built with special tokens.`,name:"sep_token"},{anchor:"transformers.BertTokenizerFast.pad_token",description:`<strong>pad_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;[PAD]&quot;</code>) &#x2014;
The token used for padding, for example when batching sequences of different lengths.`,name:"pad_token"},{anchor:"transformers.BertTokenizerFast.cls_token",description:`<strong>cls_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;[CLS]&quot;</code>) &#x2014;
The classifier token which is used when doing sequence classification (classification of the whole sequence
instead of per-token classification). It is the first token of the sequence when built with special tokens.`,name:"cls_token"},{anchor:"transformers.BertTokenizerFast.mask_token",description:`<strong>mask_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;[MASK]&quot;</code>) &#x2014;
The token used for masking values. This is the token used when training this model with masked language
modeling. This is the token which the model will try to predict.`,name:"mask_token"},{anchor:"transformers.BertTokenizerFast.clean_text",description:`<strong>clean_text</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to clean the text before tokenization by removing any control characters and replacing all
whitespaces by the classic one.`,name:"clean_text"},{anchor:"transformers.BertTokenizerFast.tokenize_chinese_chars",description:`<strong>tokenize_chinese_chars</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to tokenize Chinese characters. This should likely be deactivated for Japanese (see <a href="https://github.com/huggingface/transformers/issues/328" rel="nofollow">this
issue</a>).`,name:"tokenize_chinese_chars"},{anchor:"transformers.BertTokenizerFast.strip_accents",description:`<strong>strip_accents</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to strip all accents. If this option is not specified, then it will be determined by the
value for <code>lowercase</code> (as in the original BERT).`,name:"strip_accents"},{anchor:"transformers.BertTokenizerFast.wordpieces_prefix",description:`<strong>wordpieces_prefix</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;##&quot;</code>) &#x2014;
The prefix for subwords.`,name:"wordpieces_prefix"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/bert/tokenization_bert_fast.py#L32"}}),ht=new B({props:{name:"build_inputs_with_special_tokens",anchor:"transformers.BertTokenizerFast.build_inputs_with_special_tokens",parameters:[{name:"token_ids_0",val:""},{name:"token_ids_1",val:" = None"}],parametersDescription:[{anchor:"transformers.BertTokenizerFast.build_inputs_with_special_tokens.token_ids_0",description:`<strong>token_ids_0</strong> (<code>List[int]</code>) &#x2014;
List of IDs to which the special tokens will be added.`,name:"token_ids_0"},{anchor:"transformers.BertTokenizerFast.build_inputs_with_special_tokens.token_ids_1",description:`<strong>token_ids_1</strong> (<code>List[int]</code>, <em>optional</em>) &#x2014;
Optional second list of IDs for sequence pairs.`,name:"token_ids_1"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/bert/tokenization_bert_fast.py#L117",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>List of <a href="../glossary#input-ids">input IDs</a> with the appropriate special tokens.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>List[int]</code></p>
`}}),ut=new H({props:{title:"BertModel",local:"transformers.BertModel",headingTag:"h2"}}),ft=new B({props:{name:"class transformers.BertModel",anchor:"transformers.BertModel",parameters:[{name:"config",val:""},{name:"add_pooling_layer",val:" = True"}],parametersDescription:[{anchor:"transformers.BertModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertModel">BertModel</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"},{anchor:"transformers.BertModel.add_pooling_layer",description:`<strong>add_pooling_layer</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether to add a pooling layer`,name:"add_pooling_layer"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/bert/modeling_bert.py#L841"}}),gt=new B({props:{name:"forward",anchor:"transformers.BertModel.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_hidden_states",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"past_key_values",val:": typing.Optional[list[torch.FloatTensor]] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.Tensor] = None"}],parametersDescription:[{anchor:"transformers.BertModel.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.BertModel.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.BertModel.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.BertModel.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.BertModel.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.BertModel.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.BertModel.forward.encoder_hidden_states",description:`<strong>encoder_hidden_states</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
if the model is configured as a decoder.`,name:"encoder_hidden_states"},{anchor:"transformers.BertModel.forward.encoder_attention_mask",description:`<strong>encoder_attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
the cross-attention if the model is configured as a decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>`,name:"encoder_attention_mask"},{anchor:"transformers.BertModel.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>list[torch.FloatTensor]</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.BertModel.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.BertModel.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.BertModel.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.BertModel.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.BertModel.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.Tensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/bert/modeling_bert.py#L877",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions"
>transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertConfig"
>BertConfig</a>) and inputs.</p>
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
`}}),xe=new Te({props:{$$slots:{default:[wa]},$$scope:{ctx:w}}}),_t=new H({props:{title:"BertForPreTraining",local:"transformers.BertForPreTraining",headingTag:"h2"}}),bt=new B({props:{name:"class transformers.BertForPreTraining",anchor:"transformers.BertForPreTraining",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.BertForPreTraining.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertForPreTraining">BertForPreTraining</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/bert/modeling_bert.py#L1034"}}),Tt=new B({props:{name:"forward",anchor:"transformers.BertForPreTraining.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"labels",val:": typing.Optional[torch.Tensor] = None"},{name:"next_sentence_label",val:": typing.Optional[torch.Tensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.BertForPreTraining.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.BertForPreTraining.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.BertForPreTraining.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.BertForPreTraining.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.BertForPreTraining.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.BertForPreTraining.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.BertForPreTraining.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should be in <code>[-100, 0, ..., config.vocab_size]</code> (see <code>input_ids</code> docstring) Tokens with indices set to <code>-100</code> are ignored (masked),
the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>`,name:"labels"},{anchor:"transformers.BertForPreTraining.forward.next_sentence_label",description:`<strong>next_sentence_label</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the next sequence prediction (classification) loss. Input should be a sequence
pair (see <code>input_ids</code> docstring) Indices should be in <code>[0, 1]</code>:</p>
<ul>
<li>0 indicates sequence B is a continuation of sequence A,</li>
<li>1 indicates sequence B is a random sequence.</li>
</ul>`,name:"next_sentence_label"},{anchor:"transformers.BertForPreTraining.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.BertForPreTraining.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.BertForPreTraining.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/bert/modeling_bert.py#L1053",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/model_doc/bert#transformers.models.bert.modeling_bert.BertForPreTrainingOutput"
>transformers.models.bert.modeling_bert.BertForPreTrainingOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertConfig"
>BertConfig</a>) and inputs.</p>
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
  href="/docs/transformers/v4.56.2/en/model_doc/bert#transformers.models.bert.modeling_bert.BertForPreTrainingOutput"
>transformers.models.bert.modeling_bert.BertForPreTrainingOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),Ce=new Te({props:{$$slots:{default:[va]},$$scope:{ctx:w}}}),Ue=new ye({props:{anchor:"transformers.BertForPreTraining.forward.example",$$slots:{default:[$a]},$$scope:{ctx:w}}}),yt=new H({props:{title:"BertLMHeadModel",local:"transformers.BertLMHeadModel",headingTag:"h2"}}),Mt=new B({props:{name:"class transformers.BertLMHeadModel",anchor:"transformers.BertLMHeadModel",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.BertLMHeadModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertLMHeadModel">BertLMHeadModel</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/bert/modeling_bert.py#L1138"}}),kt=new B({props:{name:"forward",anchor:"transformers.BertLMHeadModel.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_hidden_states",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"labels",val:": typing.Optional[torch.Tensor] = None"},{name:"past_key_values",val:": typing.Optional[list[torch.Tensor]] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.Tensor] = None"},{name:"**loss_kwargs",val:""}],parametersDescription:[{anchor:"transformers.BertLMHeadModel.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.BertLMHeadModel.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.BertLMHeadModel.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.BertLMHeadModel.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.BertLMHeadModel.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.BertLMHeadModel.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.BertLMHeadModel.forward.encoder_hidden_states",description:`<strong>encoder_hidden_states</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
if the model is configured as a decoder.`,name:"encoder_hidden_states"},{anchor:"transformers.BertLMHeadModel.forward.encoder_attention_mask",description:`<strong>encoder_attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
the cross-attention if the model is configured as a decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>`,name:"encoder_attention_mask"},{anchor:"transformers.BertLMHeadModel.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the left-to-right language modeling loss (next word prediction). Indices should be in
<code>[-100, 0, ..., config.vocab_size]</code> (see <code>input_ids</code> docstring) Tokens with indices set to <code>-100</code> are
ignored (masked), the loss is only computed for the tokens with labels n <code>[0, ..., config.vocab_size]</code>`,name:"labels"},{anchor:"transformers.BertLMHeadModel.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>list[torch.Tensor]</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.BertLMHeadModel.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.BertLMHeadModel.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.BertLMHeadModel.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.BertLMHeadModel.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.BertLMHeadModel.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.Tensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/bert/modeling_bert.py#L1160",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithCrossAttentions"
>transformers.modeling_outputs.CausalLMOutputWithCrossAttentions</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertConfig"
>BertConfig</a>) and inputs.</p>
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
`}}),Fe=new Te({props:{$$slots:{default:[Ba]},$$scope:{ctx:w}}}),Ie=new ye({props:{anchor:"transformers.BertLMHeadModel.forward.example",$$slots:{default:[Ja]},$$scope:{ctx:w}}}),wt=new H({props:{title:"BertForMaskedLM",local:"transformers.BertForMaskedLM",headingTag:"h2"}}),vt=new B({props:{name:"class transformers.BertForMaskedLM",anchor:"transformers.BertForMaskedLM",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.BertForMaskedLM.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertForMaskedLM">BertForMaskedLM</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/bert/modeling_bert.py#L1229"}}),$t=new B({props:{name:"forward",anchor:"transformers.BertForMaskedLM.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_hidden_states",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"labels",val:": typing.Optional[torch.Tensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.BertForMaskedLM.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.BertForMaskedLM.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.BertForMaskedLM.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.BertForMaskedLM.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.BertForMaskedLM.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.BertForMaskedLM.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.BertForMaskedLM.forward.encoder_hidden_states",description:`<strong>encoder_hidden_states</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
if the model is configured as a decoder.`,name:"encoder_hidden_states"},{anchor:"transformers.BertForMaskedLM.forward.encoder_attention_mask",description:`<strong>encoder_attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
the cross-attention if the model is configured as a decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>`,name:"encoder_attention_mask"},{anchor:"transformers.BertForMaskedLM.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should be in <code>[-100, 0, ..., config.vocab_size]</code> (see <code>input_ids</code> docstring) Tokens with indices set to <code>-100</code> are ignored (masked), the
loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>`,name:"labels"},{anchor:"transformers.BertForMaskedLM.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.BertForMaskedLM.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.BertForMaskedLM.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/bert/modeling_bert.py#L1254",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.MaskedLMOutput"
>transformers.modeling_outputs.MaskedLMOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertConfig"
>BertConfig</a>) and inputs.</p>
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
`}}),qe=new Te({props:{$$slots:{default:[ja]},$$scope:{ctx:w}}}),Ze=new ye({props:{anchor:"transformers.BertForMaskedLM.forward.example",$$slots:{default:[za]},$$scope:{ctx:w}}}),Bt=new H({props:{title:"BertForNextSentencePrediction",local:"transformers.BertForNextSentencePrediction",headingTag:"h2"}}),Jt=new B({props:{name:"class transformers.BertForNextSentencePrediction",anchor:"transformers.BertForNextSentencePrediction",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.BertForNextSentencePrediction.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertForNextSentencePrediction">BertForNextSentencePrediction</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/bert/modeling_bert.py#L1342"}}),jt=new B({props:{name:"forward",anchor:"transformers.BertForNextSentencePrediction.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"labels",val:": typing.Optional[torch.Tensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.BertForNextSentencePrediction.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.BertForNextSentencePrediction.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.BertForNextSentencePrediction.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.BertForNextSentencePrediction.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.BertForNextSentencePrediction.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.BertForNextSentencePrediction.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.BertForNextSentencePrediction.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the next sequence prediction (classification) loss. Input should be a sequence pair
(see <code>input_ids</code> docstring). Indices should be in <code>[0, 1]</code>:</p>
<ul>
<li>0 indicates sequence B is a continuation of sequence A,</li>
<li>1 indicates sequence B is a random sequence.</li>
</ul>`,name:"labels"},{anchor:"transformers.BertForNextSentencePrediction.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.BertForNextSentencePrediction.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.BertForNextSentencePrediction.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/bert/modeling_bert.py#L1352",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.NextSentencePredictorOutput"
>transformers.modeling_outputs.NextSentencePredictorOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertConfig"
>BertConfig</a>) and inputs.</p>
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
`}}),We=new Te({props:{$$slots:{default:[xa]},$$scope:{ctx:w}}}),Ne=new ye({props:{anchor:"transformers.BertForNextSentencePrediction.forward.example",$$slots:{default:[Ca]},$$scope:{ctx:w}}}),zt=new H({props:{title:"BertForSequenceClassification",local:"transformers.BertForSequenceClassification",headingTag:"h2"}}),xt=new B({props:{name:"class transformers.BertForSequenceClassification",anchor:"transformers.BertForSequenceClassification",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.BertForSequenceClassification.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertForSequenceClassification">BertForSequenceClassification</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/bert/modeling_bert.py#L1443"}}),Ct=new B({props:{name:"forward",anchor:"transformers.BertForSequenceClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"labels",val:": typing.Optional[torch.Tensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.BertForSequenceClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.BertForSequenceClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.BertForSequenceClassification.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.BertForSequenceClassification.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.BertForSequenceClassification.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.BertForSequenceClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.BertForSequenceClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the sequence classification/regression loss. Indices should be in <code>[0, ..., config.num_labels - 1]</code>. If <code>config.num_labels == 1</code> a regression loss is computed (Mean-Square loss), If
<code>config.num_labels &gt; 1</code> a classification loss is computed (Cross-Entropy).`,name:"labels"},{anchor:"transformers.BertForSequenceClassification.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.BertForSequenceClassification.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.BertForSequenceClassification.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/bert/modeling_bert.py#L1459",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput"
>transformers.modeling_outputs.SequenceClassifierOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertConfig"
>BertConfig</a>) and inputs.</p>
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
`}}),Ve=new Te({props:{$$slots:{default:[Ua]},$$scope:{ctx:w}}}),Ge=new ye({props:{anchor:"transformers.BertForSequenceClassification.forward.example",$$slots:{default:[Fa]},$$scope:{ctx:w}}}),He=new ye({props:{anchor:"transformers.BertForSequenceClassification.forward.example-2",$$slots:{default:[Ia]},$$scope:{ctx:w}}}),Ut=new H({props:{title:"BertForMultipleChoice",local:"transformers.BertForMultipleChoice",headingTag:"h2"}}),Ft=new B({props:{name:"class transformers.BertForMultipleChoice",anchor:"transformers.BertForMultipleChoice",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.BertForMultipleChoice.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertForMultipleChoice">BertForMultipleChoice</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/bert/modeling_bert.py#L1533"}}),It=new B({props:{name:"forward",anchor:"transformers.BertForMultipleChoice.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"labels",val:": typing.Optional[torch.Tensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.BertForMultipleChoice.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, num_choices, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.BertForMultipleChoice.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.BertForMultipleChoice.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, num_choices, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.BertForMultipleChoice.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, num_choices, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.max_position_embeddings - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.BertForMultipleChoice.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.BertForMultipleChoice.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_choices, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.BertForMultipleChoice.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the multiple choice classification loss. Indices should be in <code>[0, ..., num_choices-1]</code> where <code>num_choices</code> is the size of the second dimension of the input tensors. (See
<code>input_ids</code> above)`,name:"labels"},{anchor:"transformers.BertForMultipleChoice.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.BertForMultipleChoice.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.BertForMultipleChoice.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/bert/modeling_bert.py#L1547",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.MultipleChoiceModelOutput"
>transformers.modeling_outputs.MultipleChoiceModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertConfig"
>BertConfig</a>) and inputs.</p>
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
`}}),Xe=new Te({props:{$$slots:{default:[qa]},$$scope:{ctx:w}}}),Le=new ye({props:{anchor:"transformers.BertForMultipleChoice.forward.example",$$slots:{default:[Za]},$$scope:{ctx:w}}}),qt=new H({props:{title:"BertForTokenClassification",local:"transformers.BertForTokenClassification",headingTag:"h2"}}),Zt=new B({props:{name:"class transformers.BertForTokenClassification",anchor:"transformers.BertForTokenClassification",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.BertForTokenClassification.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertForTokenClassification">BertForTokenClassification</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/bert/modeling_bert.py#L1640"}}),Wt=new B({props:{name:"forward",anchor:"transformers.BertForTokenClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"labels",val:": typing.Optional[torch.Tensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.BertForTokenClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.BertForTokenClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.BertForTokenClassification.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.BertForTokenClassification.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.BertForTokenClassification.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.BertForTokenClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.BertForTokenClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the token classification loss. Indices should be in <code>[0, ..., config.num_labels - 1]</code>.`,name:"labels"},{anchor:"transformers.BertForTokenClassification.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.BertForTokenClassification.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.BertForTokenClassification.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/bert/modeling_bert.py#L1655",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput"
>transformers.modeling_outputs.TokenClassifierOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertConfig"
>BertConfig</a>) and inputs.</p>
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
`}}),Re=new Te({props:{$$slots:{default:[Wa]},$$scope:{ctx:w}}}),Se=new ye({props:{anchor:"transformers.BertForTokenClassification.forward.example",$$slots:{default:[Na]},$$scope:{ctx:w}}}),Nt=new H({props:{title:"BertForQuestionAnswering",local:"transformers.BertForQuestionAnswering",headingTag:"h2"}}),Vt=new B({props:{name:"class transformers.BertForQuestionAnswering",anchor:"transformers.BertForQuestionAnswering",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.BertForQuestionAnswering.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertForQuestionAnswering">BertForQuestionAnswering</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/bert/modeling_bert.py#L1710"}}),Gt=new B({props:{name:"forward",anchor:"transformers.BertForQuestionAnswering.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"start_positions",val:": typing.Optional[torch.Tensor] = None"},{name:"end_positions",val:": typing.Optional[torch.Tensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.BertForQuestionAnswering.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.BertForQuestionAnswering.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.BertForQuestionAnswering.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.BertForQuestionAnswering.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.BertForQuestionAnswering.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.BertForQuestionAnswering.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.BertForQuestionAnswering.forward.start_positions",description:`<strong>start_positions</strong> (<code>torch.Tensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for position (index) of the start of the labelled span for computing the token classification loss.
Positions are clamped to the length of the sequence (<code>sequence_length</code>). Position outside of the sequence
are not taken into account for computing the loss.`,name:"start_positions"},{anchor:"transformers.BertForQuestionAnswering.forward.end_positions",description:`<strong>end_positions</strong> (<code>torch.Tensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for position (index) of the end of the labelled span for computing the token classification loss.
Positions are clamped to the length of the sequence (<code>sequence_length</code>). Position outside of the sequence
are not taken into account for computing the loss.`,name:"end_positions"},{anchor:"transformers.BertForQuestionAnswering.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.BertForQuestionAnswering.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.BertForQuestionAnswering.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/bert/modeling_bert.py#L1721",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.QuestionAnsweringModelOutput"
>transformers.modeling_outputs.QuestionAnsweringModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertConfig"
>BertConfig</a>) and inputs.</p>
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
`}}),Pe=new Te({props:{$$slots:{default:[Va]},$$scope:{ctx:w}}}),Ee=new ye({props:{anchor:"transformers.BertForQuestionAnswering.forward.example",$$slots:{default:[Ga]},$$scope:{ctx:w}}}),Ht=new H({props:{title:"Bert specific outputs",local:"transformers.models.bert.modeling_bert.BertForPreTrainingOutput",headingTag:"h2"}}),Xt=new B({props:{name:"class transformers.models.bert.modeling_bert.BertForPreTrainingOutput",anchor:"transformers.models.bert.modeling_bert.BertForPreTrainingOutput",parameters:[{name:"loss",val:": typing.Optional[torch.FloatTensor] = None"},{name:"prediction_logits",val:": typing.Optional[torch.FloatTensor] = None"},{name:"seq_relationship_logits",val:": typing.Optional[torch.FloatTensor] = None"},{name:"hidden_states",val:": typing.Optional[tuple[torch.FloatTensor]] = None"},{name:"attentions",val:": typing.Optional[tuple[torch.FloatTensor]] = None"}],parametersDescription:[{anchor:"transformers.models.bert.modeling_bert.BertForPreTrainingOutput.loss",description:`<strong>loss</strong> (<code>*optional*</code>, returned when <code>labels</code> is provided, <code>torch.FloatTensor</code> of shape <code>(1,)</code>) &#x2014;
Total loss as the sum of the masked language modeling loss and the next sequence prediction
(classification) loss.`,name:"loss"},{anchor:"transformers.models.bert.modeling_bert.BertForPreTrainingOutput.prediction_logits",description:`<strong>prediction_logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, config.vocab_size)</code>) &#x2014;
Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).`,name:"prediction_logits"},{anchor:"transformers.models.bert.modeling_bert.BertForPreTrainingOutput.seq_relationship_logits",description:`<strong>seq_relationship_logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, 2)</code>) &#x2014;
Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation
before SoftMax).`,name:"seq_relationship_logits"},{anchor:"transformers.models.bert.modeling_bert.BertForPreTrainingOutput.hidden_states",description:`<strong>hidden_states</strong> (<code>tuple[torch.FloatTensor]</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) &#x2014;
Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings, if the model has an embedding layer, +
one for the output of each layer) of shape <code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.`,name:"hidden_states"},{anchor:"transformers.models.bert.modeling_bert.BertForPreTrainingOutput.attentions",description:`<strong>attentions</strong> (<code>tuple[torch.FloatTensor]</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) &#x2014;
Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
heads.`,name:"attentions"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/bert/modeling_bert.py#L810"}}),Lt=new fa({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/bert.md"}}),{c(){t=c("meta"),h=r(),o=c("p"),d=r(),k=c("p"),k.innerHTML=n,M=r(),j=c("div"),j.innerHTML=Rn,Oe=r(),u(me.$$.fragment),En=r(),De=c("p"),De.innerHTML=lr,Qn=r(),Ke=c("p"),Ke.innerHTML=dr,An=r(),u(Be.$$.fragment),Yn=r(),et=c("p"),et.innerHTML=cr,On=r(),u(Je.$$.fragment),Dn=r(),u(tt.$$.fragment),Kn=r(),nt=c("ul"),nt.innerHTML=pr,eo=r(),u(ot.$$.fragment),to=r(),L=c("div"),u(st.$$.fragment),Co=r(),St=c("p"),St.innerHTML=mr,Uo=r(),Pt=c("p"),Pt.innerHTML=hr,Fo=r(),u(je.$$.fragment),no=r(),u(rt.$$.fragment),oo=r(),z=c("div"),u(at.$$.fragment),Io=r(),Et=c("p"),Et.textContent=ur,qo=r(),Qt=c("p"),Qt.innerHTML=fr,Zo=r(),he=c("div"),u(it.$$.fragment),Wo=r(),At=c("p"),At.textContent=gr,No=r(),Yt=c("ul"),Yt.innerHTML=_r,Vo=r(),ze=c("div"),u(lt.$$.fragment),Go=r(),Ot=c("p"),Ot.innerHTML=br,Ho=r(),ue=c("div"),u(dt.$$.fragment),Xo=r(),Dt=c("p"),Dt.innerHTML=Tr,Lo=r(),Kt=c("p"),Kt.textContent=yr,Ro=r(),en=c("div"),u(ct.$$.fragment),so=r(),u(pt.$$.fragment),ro=r(),R=c("div"),u(mt.$$.fragment),So=r(),tn=c("p"),tn.innerHTML=Mr,Po=r(),nn=c("p"),nn.innerHTML=kr,Eo=r(),fe=c("div"),u(ht.$$.fragment),Qo=r(),on=c("p"),on.textContent=wr,Ao=r(),sn=c("ul"),sn.innerHTML=vr,ao=r(),u(ut.$$.fragment),io=r(),x=c("div"),u(ft.$$.fragment),Yo=r(),rn=c("p"),rn.innerHTML=$r,Oo=r(),an=c("p"),an.innerHTML=Br,Do=r(),ln=c("p"),ln.innerHTML=Jr,Ko=r(),dn=c("p"),dn.innerHTML=jr,es=r(),ge=c("div"),u(gt.$$.fragment),ts=r(),cn=c("p"),cn.innerHTML=zr,ns=r(),u(xe.$$.fragment),lo=r(),u(_t.$$.fragment),co=r(),U=c("div"),u(bt.$$.fragment),os=r(),pn=c("p"),pn.innerHTML=xr,ss=r(),mn=c("p"),mn.innerHTML=Cr,rs=r(),hn=c("p"),hn.innerHTML=Ur,as=r(),K=c("div"),u(Tt.$$.fragment),is=r(),un=c("p"),un.innerHTML=Fr,ls=r(),u(Ce.$$.fragment),ds=r(),u(Ue.$$.fragment),po=r(),u(yt.$$.fragment),mo=r(),F=c("div"),u(Mt.$$.fragment),cs=r(),fn=c("p"),fn.innerHTML=Ir,ps=r(),gn=c("p"),gn.innerHTML=qr,ms=r(),_n=c("p"),_n.innerHTML=Zr,hs=r(),ee=c("div"),u(kt.$$.fragment),us=r(),bn=c("p"),bn.innerHTML=Wr,fs=r(),u(Fe.$$.fragment),gs=r(),u(Ie.$$.fragment),ho=r(),u(wt.$$.fragment),uo=r(),I=c("div"),u(vt.$$.fragment),_s=r(),Tn=c("p"),Tn.innerHTML=Nr,bs=r(),yn=c("p"),yn.innerHTML=Vr,Ts=r(),Mn=c("p"),Mn.innerHTML=Gr,ys=r(),te=c("div"),u($t.$$.fragment),Ms=r(),kn=c("p"),kn.innerHTML=Hr,ks=r(),u(qe.$$.fragment),ws=r(),u(Ze.$$.fragment),fo=r(),u(Bt.$$.fragment),go=r(),q=c("div"),u(Jt.$$.fragment),vs=r(),wn=c("p"),wn.innerHTML=Xr,$s=r(),vn=c("p"),vn.innerHTML=Lr,Bs=r(),$n=c("p"),$n.innerHTML=Rr,Js=r(),ne=c("div"),u(jt.$$.fragment),js=r(),Bn=c("p"),Bn.innerHTML=Sr,zs=r(),u(We.$$.fragment),xs=r(),u(Ne.$$.fragment),_o=r(),u(zt.$$.fragment),bo=r(),Z=c("div"),u(xt.$$.fragment),Cs=r(),Jn=c("p"),Jn.textContent=Pr,Us=r(),jn=c("p"),jn.innerHTML=Er,Fs=r(),zn=c("p"),zn.innerHTML=Qr,Is=r(),X=c("div"),u(Ct.$$.fragment),qs=r(),xn=c("p"),xn.innerHTML=Ar,Zs=r(),u(Ve.$$.fragment),Ws=r(),u(Ge.$$.fragment),Ns=r(),u(He.$$.fragment),To=r(),u(Ut.$$.fragment),yo=r(),W=c("div"),u(Ft.$$.fragment),Vs=r(),Cn=c("p"),Cn.textContent=Yr,Gs=r(),Un=c("p"),Un.innerHTML=Or,Hs=r(),Fn=c("p"),Fn.innerHTML=Dr,Xs=r(),oe=c("div"),u(It.$$.fragment),Ls=r(),In=c("p"),In.innerHTML=Kr,Rs=r(),u(Xe.$$.fragment),Ss=r(),u(Le.$$.fragment),Mo=r(),u(qt.$$.fragment),ko=r(),N=c("div"),u(Zt.$$.fragment),Ps=r(),qn=c("p"),qn.textContent=ea,Es=r(),Zn=c("p"),Zn.innerHTML=ta,Qs=r(),Wn=c("p"),Wn.innerHTML=na,As=r(),se=c("div"),u(Wt.$$.fragment),Ys=r(),Nn=c("p"),Nn.innerHTML=oa,Os=r(),u(Re.$$.fragment),Ds=r(),u(Se.$$.fragment),wo=r(),u(Nt.$$.fragment),vo=r(),V=c("div"),u(Vt.$$.fragment),Ks=r(),Vn=c("p"),Vn.innerHTML=sa,er=r(),Gn=c("p"),Gn.innerHTML=ra,tr=r(),Hn=c("p"),Hn.innerHTML=aa,nr=r(),re=c("div"),u(Gt.$$.fragment),or=r(),Xn=c("p"),Xn.innerHTML=ia,sr=r(),u(Pe.$$.fragment),rr=r(),u(Ee.$$.fragment),$o=r(),u(Ht.$$.fragment),Bo=r(),Me=c("div"),u(Xt.$$.fragment),ar=r(),Ln=c("p"),Ln.innerHTML=la,Jo=r(),u(Lt.$$.fragment),jo=r(),Sn=c("p"),this.h()},l(e){const i=ha("svelte-u9bgzb",document.head);t=p(i,"META",{name:!0,content:!0}),i.forEach(l),h=a(e),o=p(e,"P",{}),v(o).forEach(l),d=a(e),k=p(e,"P",{"data-svelte-h":!0}),y(k)!=="svelte-1hdwlrf"&&(k.innerHTML=n),M=a(e),j=p(e,"DIV",{style:!0,"data-svelte-h":!0}),y(j)!=="svelte-ithiq1"&&(j.innerHTML=Rn),Oe=a(e),f(me.$$.fragment,e),En=a(e),De=p(e,"P",{"data-svelte-h":!0}),y(De)!=="svelte-16kuata"&&(De.innerHTML=lr),Qn=a(e),Ke=p(e,"P",{"data-svelte-h":!0}),y(Ke)!=="svelte-z34ajl"&&(Ke.innerHTML=dr),An=a(e),f(Be.$$.fragment,e),Yn=a(e),et=p(e,"P",{"data-svelte-h":!0}),y(et)!=="svelte-lqa8w5"&&(et.innerHTML=cr),On=a(e),f(Je.$$.fragment,e),Dn=a(e),f(tt.$$.fragment,e),Kn=a(e),nt=p(e,"UL",{"data-svelte-h":!0}),y(nt)!=="svelte-19zhk15"&&(nt.innerHTML=pr),eo=a(e),f(ot.$$.fragment,e),to=a(e),L=p(e,"DIV",{class:!0});var ae=v(L);f(st.$$.fragment,ae),Co=a(ae),St=p(ae,"P",{"data-svelte-h":!0}),y(St)!=="svelte-1p040p3"&&(St.innerHTML=mr),Uo=a(ae),Pt=p(ae,"P",{"data-svelte-h":!0}),y(Pt)!=="svelte-1ek1ss9"&&(Pt.innerHTML=hr),Fo=a(ae),f(je.$$.fragment,ae),ae.forEach(l),no=a(e),f(rt.$$.fragment,e),oo=a(e),z=p(e,"DIV",{class:!0});var C=v(z);f(at.$$.fragment,C),Io=a(C),Et=p(C,"P",{"data-svelte-h":!0}),y(Et)!=="svelte-1aju482"&&(Et.textContent=ur),qo=a(C),Qt=p(C,"P",{"data-svelte-h":!0}),y(Qt)!=="svelte-ntrhio"&&(Qt.innerHTML=fr),Zo=a(C),he=p(C,"DIV",{class:!0});var ke=v(he);f(it.$$.fragment,ke),Wo=a(ke),At=p(ke,"P",{"data-svelte-h":!0}),y(At)!=="svelte-t7qurq"&&(At.textContent=gr),No=a(ke),Yt=p(ke,"UL",{"data-svelte-h":!0}),y(Yt)!=="svelte-xi6653"&&(Yt.innerHTML=_r),ke.forEach(l),Vo=a(C),ze=p(C,"DIV",{class:!0});var Rt=v(ze);f(lt.$$.fragment,Rt),Go=a(Rt),Ot=p(Rt,"P",{"data-svelte-h":!0}),y(Ot)!=="svelte-1f4f5kp"&&(Ot.innerHTML=br),Rt.forEach(l),Ho=a(C),ue=p(C,"DIV",{class:!0});var we=v(ue);f(dt.$$.fragment,we),Xo=a(we),Dt=p(we,"P",{"data-svelte-h":!0}),y(Dt)!=="svelte-zj1vf1"&&(Dt.innerHTML=Tr),Lo=a(we),Kt=p(we,"P",{"data-svelte-h":!0}),y(Kt)!=="svelte-9vptpw"&&(Kt.textContent=yr),we.forEach(l),Ro=a(C),en=p(C,"DIV",{class:!0});var Pn=v(en);f(ct.$$.fragment,Pn),Pn.forEach(l),C.forEach(l),so=a(e),f(pt.$$.fragment,e),ro=a(e),R=p(e,"DIV",{class:!0});var ie=v(R);f(mt.$$.fragment,ie),So=a(ie),tn=p(ie,"P",{"data-svelte-h":!0}),y(tn)!=="svelte-9i6ter"&&(tn.innerHTML=Mr),Po=a(ie),nn=p(ie,"P",{"data-svelte-h":!0}),y(nn)!=="svelte-gxzj9w"&&(nn.innerHTML=kr),Eo=a(ie),fe=p(ie,"DIV",{class:!0});var ve=v(fe);f(ht.$$.fragment,ve),Qo=a(ve),on=p(ve,"P",{"data-svelte-h":!0}),y(on)!=="svelte-t7qurq"&&(on.textContent=wr),Ao=a(ve),sn=p(ve,"UL",{"data-svelte-h":!0}),y(sn)!=="svelte-xi6653"&&(sn.innerHTML=vr),ve.forEach(l),ie.forEach(l),ao=a(e),f(ut.$$.fragment,e),io=a(e),x=p(e,"DIV",{class:!0});var G=v(x);f(ft.$$.fragment,G),Yo=a(G),rn=p(G,"P",{"data-svelte-h":!0}),y(rn)!=="svelte-1854dma"&&(rn.innerHTML=$r),Oo=a(G),an=p(G,"P",{"data-svelte-h":!0}),y(an)!=="svelte-174erte"&&(an.innerHTML=Br),Do=a(G),ln=p(G,"P",{"data-svelte-h":!0}),y(ln)!=="svelte-q52n56"&&(ln.innerHTML=Jr),Ko=a(G),dn=p(G,"P",{"data-svelte-h":!0}),y(dn)!=="svelte-hswkmf"&&(dn.innerHTML=jr),es=a(G),ge=p(G,"DIV",{class:!0});var $e=v(ge);f(gt.$$.fragment,$e),ts=a($e),cn=p($e,"P",{"data-svelte-h":!0}),y(cn)!=="svelte-lx7ksk"&&(cn.innerHTML=zr),ns=a($e),f(xe.$$.fragment,$e),$e.forEach(l),G.forEach(l),lo=a(e),f(_t.$$.fragment,e),co=a(e),U=p(e,"DIV",{class:!0});var S=v(U);f(bt.$$.fragment,S),os=a(S),pn=p(S,"P",{"data-svelte-h":!0}),y(pn)!=="svelte-198cggn"&&(pn.innerHTML=xr),ss=a(S),mn=p(S,"P",{"data-svelte-h":!0}),y(mn)!=="svelte-q52n56"&&(mn.innerHTML=Cr),rs=a(S),hn=p(S,"P",{"data-svelte-h":!0}),y(hn)!=="svelte-hswkmf"&&(hn.innerHTML=Ur),as=a(S),K=p(S,"DIV",{class:!0});var le=v(K);f(Tt.$$.fragment,le),is=a(le),un=p(le,"P",{"data-svelte-h":!0}),y(un)!=="svelte-1bxd93y"&&(un.innerHTML=Fr),ls=a(le),f(Ce.$$.fragment,le),ds=a(le),f(Ue.$$.fragment,le),le.forEach(l),S.forEach(l),po=a(e),f(yt.$$.fragment,e),mo=a(e),F=p(e,"DIV",{class:!0});var P=v(F);f(Mt.$$.fragment,P),cs=a(P),fn=p(P,"P",{"data-svelte-h":!0}),y(fn)!=="svelte-2mjbai"&&(fn.innerHTML=Ir),ps=a(P),gn=p(P,"P",{"data-svelte-h":!0}),y(gn)!=="svelte-q52n56"&&(gn.innerHTML=qr),ms=a(P),_n=p(P,"P",{"data-svelte-h":!0}),y(_n)!=="svelte-hswkmf"&&(_n.innerHTML=Zr),hs=a(P),ee=p(P,"DIV",{class:!0});var de=v(ee);f(kt.$$.fragment,de),us=a(de),bn=p(de,"P",{"data-svelte-h":!0}),y(bn)!=="svelte-1ng8cmc"&&(bn.innerHTML=Wr),fs=a(de),f(Fe.$$.fragment,de),gs=a(de),f(Ie.$$.fragment,de),de.forEach(l),P.forEach(l),ho=a(e),f(wt.$$.fragment,e),uo=a(e),I=p(e,"DIV",{class:!0});var E=v(I);f(vt.$$.fragment,E),_s=a(E),Tn=p(E,"P",{"data-svelte-h":!0}),y(Tn)!=="svelte-p1zbd9"&&(Tn.innerHTML=Nr),bs=a(E),yn=p(E,"P",{"data-svelte-h":!0}),y(yn)!=="svelte-q52n56"&&(yn.innerHTML=Vr),Ts=a(E),Mn=p(E,"P",{"data-svelte-h":!0}),y(Mn)!=="svelte-hswkmf"&&(Mn.innerHTML=Gr),ys=a(E),te=p(E,"DIV",{class:!0});var ce=v(te);f($t.$$.fragment,ce),Ms=a(ce),kn=p(ce,"P",{"data-svelte-h":!0}),y(kn)!=="svelte-mfjaqs"&&(kn.innerHTML=Hr),ks=a(ce),f(qe.$$.fragment,ce),ws=a(ce),f(Ze.$$.fragment,ce),ce.forEach(l),E.forEach(l),fo=a(e),f(Bt.$$.fragment,e),go=a(e),q=p(e,"DIV",{class:!0});var Q=v(q);f(Jt.$$.fragment,Q),vs=a(Q),wn=p(Q,"P",{"data-svelte-h":!0}),y(wn)!=="svelte-1yvdzzs"&&(wn.innerHTML=Xr),$s=a(Q),vn=p(Q,"P",{"data-svelte-h":!0}),y(vn)!=="svelte-q52n56"&&(vn.innerHTML=Lr),Bs=a(Q),$n=p(Q,"P",{"data-svelte-h":!0}),y($n)!=="svelte-hswkmf"&&($n.innerHTML=Rr),Js=a(Q),ne=p(Q,"DIV",{class:!0});var pe=v(ne);f(jt.$$.fragment,pe),js=a(pe),Bn=p(pe,"P",{"data-svelte-h":!0}),y(Bn)!=="svelte-1p9hp4g"&&(Bn.innerHTML=Sr),zs=a(pe),f(We.$$.fragment,pe),xs=a(pe),f(Ne.$$.fragment,pe),pe.forEach(l),Q.forEach(l),_o=a(e),f(zt.$$.fragment,e),bo=a(e),Z=p(e,"DIV",{class:!0});var A=v(Z);f(xt.$$.fragment,A),Cs=a(A),Jn=p(A,"P",{"data-svelte-h":!0}),y(Jn)!=="svelte-mumz6f"&&(Jn.textContent=Pr),Us=a(A),jn=p(A,"P",{"data-svelte-h":!0}),y(jn)!=="svelte-q52n56"&&(jn.innerHTML=Er),Fs=a(A),zn=p(A,"P",{"data-svelte-h":!0}),y(zn)!=="svelte-hswkmf"&&(zn.innerHTML=Qr),Is=a(A),X=p(A,"DIV",{class:!0});var Y=v(X);f(Ct.$$.fragment,Y),qs=a(Y),xn=p(Y,"P",{"data-svelte-h":!0}),y(xn)!=="svelte-lv3x4"&&(xn.innerHTML=Ar),Zs=a(Y),f(Ve.$$.fragment,Y),Ws=a(Y),f(Ge.$$.fragment,Y),Ns=a(Y),f(He.$$.fragment,Y),Y.forEach(l),A.forEach(l),To=a(e),f(Ut.$$.fragment,e),yo=a(e),W=p(e,"DIV",{class:!0});var O=v(W);f(Ft.$$.fragment,O),Vs=a(O),Cn=p(O,"P",{"data-svelte-h":!0}),y(Cn)!=="svelte-ullxr4"&&(Cn.textContent=Yr),Gs=a(O),Un=p(O,"P",{"data-svelte-h":!0}),y(Un)!=="svelte-q52n56"&&(Un.innerHTML=Or),Hs=a(O),Fn=p(O,"P",{"data-svelte-h":!0}),y(Fn)!=="svelte-hswkmf"&&(Fn.innerHTML=Dr),Xs=a(O),oe=p(O,"DIV",{class:!0});var Qe=v(oe);f(It.$$.fragment,Qe),Ls=a(Qe),In=p(Qe,"P",{"data-svelte-h":!0}),y(In)!=="svelte-pg0f4c"&&(In.innerHTML=Kr),Rs=a(Qe),f(Xe.$$.fragment,Qe),Ss=a(Qe),f(Le.$$.fragment,Qe),Qe.forEach(l),O.forEach(l),Mo=a(e),f(qt.$$.fragment,e),ko=a(e),N=p(e,"DIV",{class:!0});var _e=v(N);f(Zt.$$.fragment,_e),Ps=a(_e),qn=p(_e,"P",{"data-svelte-h":!0}),y(qn)!=="svelte-10pjdkh"&&(qn.textContent=ea),Es=a(_e),Zn=p(_e,"P",{"data-svelte-h":!0}),y(Zn)!=="svelte-q52n56"&&(Zn.innerHTML=ta),Qs=a(_e),Wn=p(_e,"P",{"data-svelte-h":!0}),y(Wn)!=="svelte-hswkmf"&&(Wn.innerHTML=na),As=a(_e),se=p(_e,"DIV",{class:!0});var Ae=v(se);f(Wt.$$.fragment,Ae),Ys=a(Ae),Nn=p(Ae,"P",{"data-svelte-h":!0}),y(Nn)!=="svelte-11js5vu"&&(Nn.innerHTML=oa),Os=a(Ae),f(Re.$$.fragment,Ae),Ds=a(Ae),f(Se.$$.fragment,Ae),Ae.forEach(l),_e.forEach(l),wo=a(e),f(Nt.$$.fragment,e),vo=a(e),V=p(e,"DIV",{class:!0});var be=v(V);f(Vt.$$.fragment,be),Ks=a(be),Vn=p(be,"P",{"data-svelte-h":!0}),y(Vn)!=="svelte-10qzh8c"&&(Vn.innerHTML=sa),er=a(be),Gn=p(be,"P",{"data-svelte-h":!0}),y(Gn)!=="svelte-q52n56"&&(Gn.innerHTML=ra),tr=a(be),Hn=p(be,"P",{"data-svelte-h":!0}),y(Hn)!=="svelte-hswkmf"&&(Hn.innerHTML=aa),nr=a(be),re=p(be,"DIV",{class:!0});var Ye=v(re);f(Gt.$$.fragment,Ye),or=a(Ye),Xn=p(Ye,"P",{"data-svelte-h":!0}),y(Xn)!=="svelte-3wz0rk"&&(Xn.innerHTML=ia),sr=a(Ye),f(Pe.$$.fragment,Ye),rr=a(Ye),f(Ee.$$.fragment,Ye),Ye.forEach(l),be.forEach(l),$o=a(e),f(Ht.$$.fragment,e),Bo=a(e),Me=p(e,"DIV",{class:!0});var xo=v(Me);f(Xt.$$.fragment,xo),ar=a(xo),Ln=p(xo,"P",{"data-svelte-h":!0}),y(Ln)!=="svelte-75l13y"&&(Ln.innerHTML=la),xo.forEach(l),Jo=a(e),f(Lt.$$.fragment,e),jo=a(e),Sn=p(e,"P",{}),v(Sn).forEach(l),this.h()},h(){$(t,"name","hf:doc:metadata"),$(t,"content",Xa),ua(j,"float","right"),$(L,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(he,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(ze,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(ue,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(en,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(z,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(fe,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(R,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(ge,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(x,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(K,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(U,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(ee,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(F,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(te,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(I,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(ne,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(q,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(X,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(Z,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(oe,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(W,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(se,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(N,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(re,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(V,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(Me,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,i){s(document.head,t),m(e,h,i),m(e,o,i),m(e,d,i),m(e,k,i),m(e,M,i),m(e,j,i),m(e,Oe,i),g(me,e,i),m(e,En,i),m(e,De,i),m(e,Qn,i),m(e,Ke,i),m(e,An,i),g(Be,e,i),m(e,Yn,i),m(e,et,i),m(e,On,i),g(Je,e,i),m(e,Dn,i),g(tt,e,i),m(e,Kn,i),m(e,nt,i),m(e,eo,i),g(ot,e,i),m(e,to,i),m(e,L,i),g(st,L,null),s(L,Co),s(L,St),s(L,Uo),s(L,Pt),s(L,Fo),g(je,L,null),m(e,no,i),g(rt,e,i),m(e,oo,i),m(e,z,i),g(at,z,null),s(z,Io),s(z,Et),s(z,qo),s(z,Qt),s(z,Zo),s(z,he),g(it,he,null),s(he,Wo),s(he,At),s(he,No),s(he,Yt),s(z,Vo),s(z,ze),g(lt,ze,null),s(ze,Go),s(ze,Ot),s(z,Ho),s(z,ue),g(dt,ue,null),s(ue,Xo),s(ue,Dt),s(ue,Lo),s(ue,Kt),s(z,Ro),s(z,en),g(ct,en,null),m(e,so,i),g(pt,e,i),m(e,ro,i),m(e,R,i),g(mt,R,null),s(R,So),s(R,tn),s(R,Po),s(R,nn),s(R,Eo),s(R,fe),g(ht,fe,null),s(fe,Qo),s(fe,on),s(fe,Ao),s(fe,sn),m(e,ao,i),g(ut,e,i),m(e,io,i),m(e,x,i),g(ft,x,null),s(x,Yo),s(x,rn),s(x,Oo),s(x,an),s(x,Do),s(x,ln),s(x,Ko),s(x,dn),s(x,es),s(x,ge),g(gt,ge,null),s(ge,ts),s(ge,cn),s(ge,ns),g(xe,ge,null),m(e,lo,i),g(_t,e,i),m(e,co,i),m(e,U,i),g(bt,U,null),s(U,os),s(U,pn),s(U,ss),s(U,mn),s(U,rs),s(U,hn),s(U,as),s(U,K),g(Tt,K,null),s(K,is),s(K,un),s(K,ls),g(Ce,K,null),s(K,ds),g(Ue,K,null),m(e,po,i),g(yt,e,i),m(e,mo,i),m(e,F,i),g(Mt,F,null),s(F,cs),s(F,fn),s(F,ps),s(F,gn),s(F,ms),s(F,_n),s(F,hs),s(F,ee),g(kt,ee,null),s(ee,us),s(ee,bn),s(ee,fs),g(Fe,ee,null),s(ee,gs),g(Ie,ee,null),m(e,ho,i),g(wt,e,i),m(e,uo,i),m(e,I,i),g(vt,I,null),s(I,_s),s(I,Tn),s(I,bs),s(I,yn),s(I,Ts),s(I,Mn),s(I,ys),s(I,te),g($t,te,null),s(te,Ms),s(te,kn),s(te,ks),g(qe,te,null),s(te,ws),g(Ze,te,null),m(e,fo,i),g(Bt,e,i),m(e,go,i),m(e,q,i),g(Jt,q,null),s(q,vs),s(q,wn),s(q,$s),s(q,vn),s(q,Bs),s(q,$n),s(q,Js),s(q,ne),g(jt,ne,null),s(ne,js),s(ne,Bn),s(ne,zs),g(We,ne,null),s(ne,xs),g(Ne,ne,null),m(e,_o,i),g(zt,e,i),m(e,bo,i),m(e,Z,i),g(xt,Z,null),s(Z,Cs),s(Z,Jn),s(Z,Us),s(Z,jn),s(Z,Fs),s(Z,zn),s(Z,Is),s(Z,X),g(Ct,X,null),s(X,qs),s(X,xn),s(X,Zs),g(Ve,X,null),s(X,Ws),g(Ge,X,null),s(X,Ns),g(He,X,null),m(e,To,i),g(Ut,e,i),m(e,yo,i),m(e,W,i),g(Ft,W,null),s(W,Vs),s(W,Cn),s(W,Gs),s(W,Un),s(W,Hs),s(W,Fn),s(W,Xs),s(W,oe),g(It,oe,null),s(oe,Ls),s(oe,In),s(oe,Rs),g(Xe,oe,null),s(oe,Ss),g(Le,oe,null),m(e,Mo,i),g(qt,e,i),m(e,ko,i),m(e,N,i),g(Zt,N,null),s(N,Ps),s(N,qn),s(N,Es),s(N,Zn),s(N,Qs),s(N,Wn),s(N,As),s(N,se),g(Wt,se,null),s(se,Ys),s(se,Nn),s(se,Os),g(Re,se,null),s(se,Ds),g(Se,se,null),m(e,wo,i),g(Nt,e,i),m(e,vo,i),m(e,V,i),g(Vt,V,null),s(V,Ks),s(V,Vn),s(V,er),s(V,Gn),s(V,tr),s(V,Hn),s(V,nr),s(V,re),g(Gt,re,null),s(re,or),s(re,Xn),s(re,sr),g(Pe,re,null),s(re,rr),g(Ee,re,null),m(e,$o,i),g(Ht,e,i),m(e,Bo,i),m(e,Me,i),g(Xt,Me,null),s(Me,ar),s(Me,Ln),m(e,Jo,i),g(Lt,e,i),m(e,jo,i),m(e,Sn,i),zo=!0},p(e,[i]){const ae={};i&2&&(ae.$$scope={dirty:i,ctx:e}),Be.$set(ae);const C={};i&2&&(C.$$scope={dirty:i,ctx:e}),Je.$set(C);const ke={};i&2&&(ke.$$scope={dirty:i,ctx:e}),je.$set(ke);const Rt={};i&2&&(Rt.$$scope={dirty:i,ctx:e}),xe.$set(Rt);const we={};i&2&&(we.$$scope={dirty:i,ctx:e}),Ce.$set(we);const Pn={};i&2&&(Pn.$$scope={dirty:i,ctx:e}),Ue.$set(Pn);const ie={};i&2&&(ie.$$scope={dirty:i,ctx:e}),Fe.$set(ie);const ve={};i&2&&(ve.$$scope={dirty:i,ctx:e}),Ie.$set(ve);const G={};i&2&&(G.$$scope={dirty:i,ctx:e}),qe.$set(G);const $e={};i&2&&($e.$$scope={dirty:i,ctx:e}),Ze.$set($e);const S={};i&2&&(S.$$scope={dirty:i,ctx:e}),We.$set(S);const le={};i&2&&(le.$$scope={dirty:i,ctx:e}),Ne.$set(le);const P={};i&2&&(P.$$scope={dirty:i,ctx:e}),Ve.$set(P);const de={};i&2&&(de.$$scope={dirty:i,ctx:e}),Ge.$set(de);const E={};i&2&&(E.$$scope={dirty:i,ctx:e}),He.$set(E);const ce={};i&2&&(ce.$$scope={dirty:i,ctx:e}),Xe.$set(ce);const Q={};i&2&&(Q.$$scope={dirty:i,ctx:e}),Le.$set(Q);const pe={};i&2&&(pe.$$scope={dirty:i,ctx:e}),Re.$set(pe);const A={};i&2&&(A.$$scope={dirty:i,ctx:e}),Se.$set(A);const Y={};i&2&&(Y.$$scope={dirty:i,ctx:e}),Pe.$set(Y);const O={};i&2&&(O.$$scope={dirty:i,ctx:e}),Ee.$set(O)},i(e){zo||(_(me.$$.fragment,e),_(Be.$$.fragment,e),_(Je.$$.fragment,e),_(tt.$$.fragment,e),_(ot.$$.fragment,e),_(st.$$.fragment,e),_(je.$$.fragment,e),_(rt.$$.fragment,e),_(at.$$.fragment,e),_(it.$$.fragment,e),_(lt.$$.fragment,e),_(dt.$$.fragment,e),_(ct.$$.fragment,e),_(pt.$$.fragment,e),_(mt.$$.fragment,e),_(ht.$$.fragment,e),_(ut.$$.fragment,e),_(ft.$$.fragment,e),_(gt.$$.fragment,e),_(xe.$$.fragment,e),_(_t.$$.fragment,e),_(bt.$$.fragment,e),_(Tt.$$.fragment,e),_(Ce.$$.fragment,e),_(Ue.$$.fragment,e),_(yt.$$.fragment,e),_(Mt.$$.fragment,e),_(kt.$$.fragment,e),_(Fe.$$.fragment,e),_(Ie.$$.fragment,e),_(wt.$$.fragment,e),_(vt.$$.fragment,e),_($t.$$.fragment,e),_(qe.$$.fragment,e),_(Ze.$$.fragment,e),_(Bt.$$.fragment,e),_(Jt.$$.fragment,e),_(jt.$$.fragment,e),_(We.$$.fragment,e),_(Ne.$$.fragment,e),_(zt.$$.fragment,e),_(xt.$$.fragment,e),_(Ct.$$.fragment,e),_(Ve.$$.fragment,e),_(Ge.$$.fragment,e),_(He.$$.fragment,e),_(Ut.$$.fragment,e),_(Ft.$$.fragment,e),_(It.$$.fragment,e),_(Xe.$$.fragment,e),_(Le.$$.fragment,e),_(qt.$$.fragment,e),_(Zt.$$.fragment,e),_(Wt.$$.fragment,e),_(Re.$$.fragment,e),_(Se.$$.fragment,e),_(Nt.$$.fragment,e),_(Vt.$$.fragment,e),_(Gt.$$.fragment,e),_(Pe.$$.fragment,e),_(Ee.$$.fragment,e),_(Ht.$$.fragment,e),_(Xt.$$.fragment,e),_(Lt.$$.fragment,e),zo=!0)},o(e){b(me.$$.fragment,e),b(Be.$$.fragment,e),b(Je.$$.fragment,e),b(tt.$$.fragment,e),b(ot.$$.fragment,e),b(st.$$.fragment,e),b(je.$$.fragment,e),b(rt.$$.fragment,e),b(at.$$.fragment,e),b(it.$$.fragment,e),b(lt.$$.fragment,e),b(dt.$$.fragment,e),b(ct.$$.fragment,e),b(pt.$$.fragment,e),b(mt.$$.fragment,e),b(ht.$$.fragment,e),b(ut.$$.fragment,e),b(ft.$$.fragment,e),b(gt.$$.fragment,e),b(xe.$$.fragment,e),b(_t.$$.fragment,e),b(bt.$$.fragment,e),b(Tt.$$.fragment,e),b(Ce.$$.fragment,e),b(Ue.$$.fragment,e),b(yt.$$.fragment,e),b(Mt.$$.fragment,e),b(kt.$$.fragment,e),b(Fe.$$.fragment,e),b(Ie.$$.fragment,e),b(wt.$$.fragment,e),b(vt.$$.fragment,e),b($t.$$.fragment,e),b(qe.$$.fragment,e),b(Ze.$$.fragment,e),b(Bt.$$.fragment,e),b(Jt.$$.fragment,e),b(jt.$$.fragment,e),b(We.$$.fragment,e),b(Ne.$$.fragment,e),b(zt.$$.fragment,e),b(xt.$$.fragment,e),b(Ct.$$.fragment,e),b(Ve.$$.fragment,e),b(Ge.$$.fragment,e),b(He.$$.fragment,e),b(Ut.$$.fragment,e),b(Ft.$$.fragment,e),b(It.$$.fragment,e),b(Xe.$$.fragment,e),b(Le.$$.fragment,e),b(qt.$$.fragment,e),b(Zt.$$.fragment,e),b(Wt.$$.fragment,e),b(Re.$$.fragment,e),b(Se.$$.fragment,e),b(Nt.$$.fragment,e),b(Vt.$$.fragment,e),b(Gt.$$.fragment,e),b(Pe.$$.fragment,e),b(Ee.$$.fragment,e),b(Ht.$$.fragment,e),b(Xt.$$.fragment,e),b(Lt.$$.fragment,e),zo=!1},d(e){e&&(l(h),l(o),l(d),l(k),l(M),l(j),l(Oe),l(En),l(De),l(Qn),l(Ke),l(An),l(Yn),l(et),l(On),l(Dn),l(Kn),l(nt),l(eo),l(to),l(L),l(no),l(oo),l(z),l(so),l(ro),l(R),l(ao),l(io),l(x),l(lo),l(co),l(U),l(po),l(mo),l(F),l(ho),l(uo),l(I),l(fo),l(go),l(q),l(_o),l(bo),l(Z),l(To),l(yo),l(W),l(Mo),l(ko),l(N),l(wo),l(vo),l(V),l($o),l(Bo),l(Me),l(Jo),l(jo),l(Sn)),l(t),T(me,e),T(Be,e),T(Je,e),T(tt,e),T(ot,e),T(st),T(je),T(rt,e),T(at),T(it),T(lt),T(dt),T(ct),T(pt,e),T(mt),T(ht),T(ut,e),T(ft),T(gt),T(xe),T(_t,e),T(bt),T(Tt),T(Ce),T(Ue),T(yt,e),T(Mt),T(kt),T(Fe),T(Ie),T(wt,e),T(vt),T($t),T(qe),T(Ze),T(Bt,e),T(Jt),T(jt),T(We),T(Ne),T(zt,e),T(xt),T(Ct),T(Ve),T(Ge),T(He),T(Ut,e),T(Ft),T(It),T(Xe),T(Le),T(qt,e),T(Zt),T(Wt),T(Re),T(Se),T(Nt,e),T(Vt),T(Gt),T(Pe),T(Ee),T(Ht,e),T(Xt),T(Lt,e)}}}const Xa='{"title":"BERT","local":"bert","sections":[{"title":"Notes","local":"notes","sections":[],"depth":2},{"title":"BertConfig","local":"transformers.BertConfig","sections":[],"depth":2},{"title":"BertTokenizer","local":"transformers.BertTokenizer","sections":[],"depth":2},{"title":"BertTokenizerFast","local":"transformers.BertTokenizerFast","sections":[],"depth":2},{"title":"BertModel","local":"transformers.BertModel","sections":[],"depth":2},{"title":"BertForPreTraining","local":"transformers.BertForPreTraining","sections":[],"depth":2},{"title":"BertLMHeadModel","local":"transformers.BertLMHeadModel","sections":[],"depth":2},{"title":"BertForMaskedLM","local":"transformers.BertForMaskedLM","sections":[],"depth":2},{"title":"BertForNextSentencePrediction","local":"transformers.BertForNextSentencePrediction","sections":[],"depth":2},{"title":"BertForSequenceClassification","local":"transformers.BertForSequenceClassification","sections":[],"depth":2},{"title":"BertForMultipleChoice","local":"transformers.BertForMultipleChoice","sections":[],"depth":2},{"title":"BertForTokenClassification","local":"transformers.BertForTokenClassification","sections":[],"depth":2},{"title":"BertForQuestionAnswering","local":"transformers.BertForQuestionAnswering","sections":[],"depth":2},{"title":"Bert specific outputs","local":"transformers.models.bert.modeling_bert.BertForPreTrainingOutput","sections":[],"depth":2}],"depth":1}';function La(w){return ca(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class Da extends pa{constructor(t){super(),ma(this,t,La,Ha,da,{})}}export{Da as component};
