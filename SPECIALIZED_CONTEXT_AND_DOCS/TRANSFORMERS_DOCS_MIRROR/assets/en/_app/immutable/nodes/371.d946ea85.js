import{s as ha,o as ua,n as j}from"../chunks/scheduler.18a86fab.js";import{S as fa,i as ga,g as c,s as a,r as f,A as ba,h as p,f as d,c as r,j as R,x as T,u as g,k as J,l as _a,y as s,a as h,v as b,d as _,t as y,w as k}from"../chunks/index.98837b22.js";import{T as pe}from"../chunks/Tip.77304350.js";import{D as C}from"../chunks/Docstring.a1ef7999.js";import{C as P}from"../chunks/CodeBlock.8d0c2e8a.js";import{E as me}from"../chunks/ExampleCodeBlock.8c3ee1f9.js";import{H as oe,E as ya}from"../chunks/getInferenceSnippets.06c2775f.js";import{H as ka,a as ys}from"../chunks/HfOption.6641485e.js";function Ta(w){let t,m="Click on the RoBERTa models in the right sidebar for more examples of how to apply RoBERTa to different language tasks.";return{c(){t=c("p"),t.textContent=m},l(n){t=p(n,"P",{"data-svelte-h":!0}),T(t)!=="svelte-1tbzw7l"&&(t.textContent=m)},m(n,i){h(n,t,i)},p:j,d(n){n&&d(t)}}}function Ma(w){let t,m;return t=new P({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwcGlwZWxpbmUlMEElMEFwaXBlbGluZSUyMCUzRCUyMHBpcGVsaW5lKCUwQSUyMCUyMCUyMCUyMHRhc2slM0QlMjJmaWxsLW1hc2slMjIlMkMlMEElMjAlMjAlMjAlMjBtb2RlbCUzRCUyMkZhY2Vib29rQUklMkZyb2JlcnRhLWJhc2UlMjIlMkMlMEElMjAlMjAlMjAlMjBkdHlwZSUzRHRvcmNoLmZsb2F0MTYlMkMlMEElMjAlMjAlMjAlMjBkZXZpY2UlM0QwJTBBKSUwQXBpcGVsaW5lKCUyMlBsYW50cyUyMGNyZWF0ZSUyMCUzQ21hc2slM0UlMjB0aHJvdWdoJTIwYSUyMHByb2Nlc3MlMjBrbm93biUyMGFzJTIwcGhvdG9zeW50aGVzaXMuJTIyKQ==",highlighted:`<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> pipeline

pipeline = pipeline(
    task=<span class="hljs-string">&quot;fill-mask&quot;</span>,
    model=<span class="hljs-string">&quot;FacebookAI/roberta-base&quot;</span>,
    dtype=torch.float16,
    device=<span class="hljs-number">0</span>
)
pipeline(<span class="hljs-string">&quot;Plants create &lt;mask&gt; through a process known as photosynthesis.&quot;</span>)`,wrap:!1}}),{c(){f(t.$$.fragment)},l(n){g(t.$$.fragment,n)},m(n,i){b(t,n,i),m=!0},p:j,i(n){m||(_(t.$$.fragment,n),m=!0)},o(n){y(t.$$.fragment,n),m=!1},d(n){k(t,n)}}}function wa(w){let t,m;return t=new P({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b01vZGVsRm9yTWFza2VkTE0lMkMlMjBBdXRvVG9rZW5pemVyJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTBBJTIwJTIwJTIwJTIwJTIyRmFjZWJvb2tBSSUyRnJvYmVydGEtYmFzZSUyMiUyQyUwQSklMEFtb2RlbCUyMCUzRCUyMEF1dG9Nb2RlbEZvck1hc2tlZExNLmZyb21fcHJldHJhaW5lZCglMEElMjAlMjAlMjAlMjAlMjJGYWNlYm9va0FJJTJGcm9iZXJ0YS1iYXNlJTIyJTJDJTBBJTIwJTIwJTIwJTIwZHR5cGUlM0R0b3JjaC5mbG9hdDE2JTJDJTBBJTIwJTIwJTIwJTIwZGV2aWNlX21hcCUzRCUyMmF1dG8lMjIlMkMlMEElMjAlMjAlMjAlMjBhdHRuX2ltcGxlbWVudGF0aW9uJTNEJTIyc2RwYSUyMiUwQSklMEFpbnB1dHMlMjAlM0QlMjB0b2tlbml6ZXIoJTIyUGxhbnRzJTIwY3JlYXRlJTIwJTNDbWFzayUzRSUyMHRocm91Z2glMjBhJTIwcHJvY2VzcyUyMGtub3duJTIwYXMlMjBwaG90b3N5bnRoZXNpcy4lMjIlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKS50byhtb2RlbC5kZXZpY2UpJTBBJTBBd2l0aCUyMHRvcmNoLm5vX2dyYWQoKSUzQSUwQSUyMCUyMCUyMCUyMG91dHB1dHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyklMEElMjAlMjAlMjAlMjBwcmVkaWN0aW9ucyUyMCUzRCUyMG91dHB1dHMubG9naXRzJTBBJTBBbWFza2VkX2luZGV4JTIwJTNEJTIwdG9yY2gud2hlcmUoaW5wdXRzJTVCJ2lucHV0X2lkcyclNUQlMjAlM0QlM0QlMjB0b2tlbml6ZXIubWFza190b2tlbl9pZCklNUIxJTVEJTBBcHJlZGljdGVkX3Rva2VuX2lkJTIwJTNEJTIwcHJlZGljdGlvbnMlNUIwJTJDJTIwbWFza2VkX2luZGV4JTVELmFyZ21heChkaW0lM0QtMSklMEFwcmVkaWN0ZWRfdG9rZW4lMjAlM0QlMjB0b2tlbml6ZXIuZGVjb2RlKHByZWRpY3RlZF90b2tlbl9pZCklMEElMEFwcmludChmJTIyVGhlJTIwcHJlZGljdGVkJTIwdG9rZW4lMjBpcyUzQSUyMCU3QnByZWRpY3RlZF90b2tlbiU3RCUyMik=",highlighted:`<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoModelForMaskedLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    <span class="hljs-string">&quot;FacebookAI/roberta-base&quot;</span>,
)
model = AutoModelForMaskedLM.from_pretrained(
    <span class="hljs-string">&quot;FacebookAI/roberta-base&quot;</span>,
    dtype=torch.float16,
    device_map=<span class="hljs-string">&quot;auto&quot;</span>,
    attn_implementation=<span class="hljs-string">&quot;sdpa&quot;</span>
)
inputs = tokenizer(<span class="hljs-string">&quot;Plants create &lt;mask&gt; through a process known as photosynthesis.&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>).to(model.device)

<span class="hljs-keyword">with</span> torch.no_grad():
    outputs = model(**inputs)
    predictions = outputs.logits

masked_index = torch.where(inputs[<span class="hljs-string">&#x27;input_ids&#x27;</span>] == tokenizer.mask_token_id)[<span class="hljs-number">1</span>]
predicted_token_id = predictions[<span class="hljs-number">0</span>, masked_index].argmax(dim=-<span class="hljs-number">1</span>)
predicted_token = tokenizer.decode(predicted_token_id)

<span class="hljs-built_in">print</span>(<span class="hljs-string">f&quot;The predicted token is: <span class="hljs-subst">{predicted_token}</span>&quot;</span>)`,wrap:!1}}),{c(){f(t.$$.fragment)},l(n){g(t.$$.fragment,n)},m(n,i){b(t,n,i),m=!0},p:j,i(n){m||(_(t.$$.fragment,n),m=!0)},o(n){y(t.$$.fragment,n),m=!1},d(n){k(t,n)}}}function va(w){let t,m;return t=new P({props:{code:"ZWNobyUyMC1lJTIwJTIyUGxhbnRzJTIwY3JlYXRlJTIwJTNDbWFzayUzRSUyMHRocm91Z2glMjBhJTIwcHJvY2VzcyUyMGtub3duJTIwYXMlMjBwaG90b3N5bnRoZXNpcy4lMjIlMjAlN0MlMjB0cmFuc2Zvcm1lcnMtY2xpJTIwcnVuJTIwLS10YXNrJTIwZmlsbC1tYXNrJTIwLS1tb2RlbCUyMEZhY2Vib29rQUklMkZyb2JlcnRhLWJhc2UlMjAtLWRldmljZSUyMDA=",highlighted:'<span class="hljs-built_in">echo</span> -e <span class="hljs-string">&quot;Plants create &lt;mask&gt; through a process known as photosynthesis.&quot;</span> | transformers-cli run --task fill-mask --model FacebookAI/roberta-base --device 0',wrap:!1}}),{c(){f(t.$$.fragment)},l(n){g(t.$$.fragment,n)},m(n,i){b(t,n,i),m=!0},p:j,i(n){m||(_(t.$$.fragment,n),m=!0)},o(n){y(t.$$.fragment,n),m=!1},d(n){k(t,n)}}}function $a(w){let t,m,n,i,M,o;return t=new ys({props:{id:"usage",option:"Pipeline",$$slots:{default:[Ma]},$$scope:{ctx:w}}}),n=new ys({props:{id:"usage",option:"AutoModel",$$slots:{default:[wa]},$$scope:{ctx:w}}}),M=new ys({props:{id:"usage",option:"transformers CLI",$$slots:{default:[va]},$$scope:{ctx:w}}}),{c(){f(t.$$.fragment),m=a(),f(n.$$.fragment),i=a(),f(M.$$.fragment)},l(u){g(t.$$.fragment,u),m=r(u),g(n.$$.fragment,u),i=r(u),g(M.$$.fragment,u)},m(u,z){b(t,u,z),h(u,m,z),b(n,u,z),h(u,i,z),b(M,u,z),o=!0},p(u,z){const _o={};z&2&&(_o.$$scope={dirty:z,ctx:u}),t.$set(_o);const Ge={};z&2&&(Ge.$$scope={dirty:z,ctx:u}),n.$set(Ge);const le={};z&2&&(le.$$scope={dirty:z,ctx:u}),M.$set(le)},i(u){o||(_(t.$$.fragment,u),_(n.$$.fragment,u),_(M.$$.fragment,u),o=!0)},o(u){y(t.$$.fragment,u),y(n.$$.fragment,u),y(M.$$.fragment,u),o=!1},d(u){u&&(d(m),d(i)),k(t,u),k(n,u),k(M,u)}}}function Ra(w){let t,m="Examples:",n,i,M;return i=new P({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMFJvYmVydGFDb25maWclMkMlMjBSb2JlcnRhTW9kZWwlMEElMEElMjMlMjBJbml0aWFsaXppbmclMjBhJTIwUm9CRVJUYSUyMGNvbmZpZ3VyYXRpb24lMEFjb25maWd1cmF0aW9uJTIwJTNEJTIwUm9iZXJ0YUNvbmZpZygpJTBBJTBBJTIzJTIwSW5pdGlhbGl6aW5nJTIwYSUyMG1vZGVsJTIwKHdpdGglMjByYW5kb20lMjB3ZWlnaHRzKSUyMGZyb20lMjB0aGUlMjBjb25maWd1cmF0aW9uJTBBbW9kZWwlMjAlM0QlMjBSb2JlcnRhTW9kZWwoY29uZmlndXJhdGlvbiklMEElMEElMjMlMjBBY2Nlc3NpbmclMjB0aGUlMjBtb2RlbCUyMGNvbmZpZ3VyYXRpb24lMEFjb25maWd1cmF0aW9uJTIwJTNEJTIwbW9kZWwuY29uZmln",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> RobertaConfig, RobertaModel

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a RoBERTa configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = RobertaConfig()

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a model (with random weights) from the configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = RobertaModel(configuration)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Accessing the model configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = model.config`,wrap:!1}}),{c(){t=c("p"),t.textContent=m,n=a(),f(i.$$.fragment)},l(o){t=p(o,"P",{"data-svelte-h":!0}),T(t)!=="svelte-kvfsh7"&&(t.textContent=m),n=r(o),g(i.$$.fragment,o)},m(o,u){h(o,t,u),h(o,n,u),b(i,o,u),M=!0},p:j,i(o){M||(_(i.$$.fragment,o),M=!0)},o(o){y(i.$$.fragment,o),M=!1},d(o){o&&(d(t),d(n)),k(i,o)}}}function Ja(w){let t,m="be encoded differently whether it is at the beginning of the sentence (without space) or not:",n,i,M;return i=new P({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMFJvYmVydGFUb2tlbml6ZXIlMEElMEF0b2tlbml6ZXIlMjAlM0QlMjBSb2JlcnRhVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJGYWNlYm9va0FJJTJGcm9iZXJ0YS1iYXNlJTIyKSUwQXRva2VuaXplciglMjJIZWxsbyUyMHdvcmxkJTIyKSU1QiUyMmlucHV0X2lkcyUyMiU1RCUwQSUwQXRva2VuaXplciglMjIlMjBIZWxsbyUyMHdvcmxkJTIyKSU1QiUyMmlucHV0X2lkcyUyMiU1RA==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> RobertaTokenizer

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = RobertaTokenizer.from_pretrained(<span class="hljs-string">&quot;FacebookAI/roberta-base&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer(<span class="hljs-string">&quot;Hello world&quot;</span>)[<span class="hljs-string">&quot;input_ids&quot;</span>]
[<span class="hljs-number">0</span>, <span class="hljs-number">31414</span>, <span class="hljs-number">232</span>, <span class="hljs-number">2</span>]

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer(<span class="hljs-string">&quot; Hello world&quot;</span>)[<span class="hljs-string">&quot;input_ids&quot;</span>]
[<span class="hljs-number">0</span>, <span class="hljs-number">20920</span>, <span class="hljs-number">232</span>, <span class="hljs-number">2</span>]`,wrap:!1}}),{c(){t=c("p"),t.textContent=m,n=a(),f(i.$$.fragment)},l(o){t=p(o,"P",{"data-svelte-h":!0}),T(t)!=="svelte-12atnao"&&(t.textContent=m),n=r(o),g(i.$$.fragment,o)},m(o,u){h(o,t,u),h(o,n,u),b(i,o,u),M=!0},p:j,i(o){M||(_(i.$$.fragment,o),M=!0)},o(o){y(i.$$.fragment,o),M=!1},d(o){o&&(d(t),d(n)),k(i,o)}}}function ja(w){let t,m="When used with <code>is_split_into_words=True</code>, this tokenizer will add a space before each word (even the first one).";return{c(){t=c("p"),t.innerHTML=m},l(n){t=p(n,"P",{"data-svelte-h":!0}),T(t)!=="svelte-jhmxzm"&&(t.innerHTML=m)},m(n,i){h(n,t,i)},p:j,d(n){n&&d(t)}}}function Ca(w){let t,m="be encoded differently whether it is at the beginning of the sentence (without space) or not:",n,i,M;return i=new P({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMFJvYmVydGFUb2tlbml6ZXJGYXN0JTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwUm9iZXJ0YVRva2VuaXplckZhc3QuZnJvbV9wcmV0cmFpbmVkKCUyMkZhY2Vib29rQUklMkZyb2JlcnRhLWJhc2UlMjIpJTBBdG9rZW5pemVyKCUyMkhlbGxvJTIwd29ybGQlMjIpJTVCJTIyaW5wdXRfaWRzJTIyJTVEJTBBJTBBdG9rZW5pemVyKCUyMiUyMEhlbGxvJTIwd29ybGQlMjIpJTVCJTIyaW5wdXRfaWRzJTIyJTVE",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> RobertaTokenizerFast

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = RobertaTokenizerFast.from_pretrained(<span class="hljs-string">&quot;FacebookAI/roberta-base&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer(<span class="hljs-string">&quot;Hello world&quot;</span>)[<span class="hljs-string">&quot;input_ids&quot;</span>]
[<span class="hljs-number">0</span>, <span class="hljs-number">31414</span>, <span class="hljs-number">232</span>, <span class="hljs-number">2</span>]

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer(<span class="hljs-string">&quot; Hello world&quot;</span>)[<span class="hljs-string">&quot;input_ids&quot;</span>]
[<span class="hljs-number">0</span>, <span class="hljs-number">20920</span>, <span class="hljs-number">232</span>, <span class="hljs-number">2</span>]`,wrap:!1}}),{c(){t=c("p"),t.textContent=m,n=a(),f(i.$$.fragment)},l(o){t=p(o,"P",{"data-svelte-h":!0}),T(t)!=="svelte-12atnao"&&(t.textContent=m),n=r(o),g(i.$$.fragment,o)},m(o,u){h(o,t,u),h(o,n,u),b(i,o,u),M=!0},p:j,i(o){M||(_(i.$$.fragment,o),M=!0)},o(o){y(i.$$.fragment,o),M=!1},d(o){o&&(d(t),d(n)),k(i,o)}}}function za(w){let t,m="When used with <code>is_split_into_words=True</code>, this tokenizer needs to be instantiated with <code>add_prefix_space=True</code>.";return{c(){t=c("p"),t.innerHTML=m},l(n){t=p(n,"P",{"data-svelte-h":!0}),T(t)!=="svelte-9gg91e"&&(t.innerHTML=m)},m(n,i){h(n,t,i)},p:j,d(n){n&&d(t)}}}function Fa(w){let t,m=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=m},l(n){t=p(n,"P",{"data-svelte-h":!0}),T(t)!=="svelte-fincs2"&&(t.innerHTML=m)},m(n,i){h(n,t,i)},p:j,d(n){n&&d(t)}}}function Ua(w){let t,m=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=m},l(n){t=p(n,"P",{"data-svelte-h":!0}),T(t)!=="svelte-fincs2"&&(t.innerHTML=m)},m(n,i){h(n,t,i)},p:j,d(n){n&&d(t)}}}function xa(w){let t,m="Example:",n,i,M;return i=new P({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBSb2JlcnRhRm9yQ2F1c2FsTE0lMkMlMjBBdXRvQ29uZmlnJTBBaW1wb3J0JTIwdG9yY2glMEElMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJGYWNlYm9va0FJJTJGcm9iZXJ0YS1iYXNlJTIyKSUwQWNvbmZpZyUyMCUzRCUyMEF1dG9Db25maWcuZnJvbV9wcmV0cmFpbmVkKCUyMkZhY2Vib29rQUklMkZyb2JlcnRhLWJhc2UlMjIpJTBBY29uZmlnLmlzX2RlY29kZXIlMjAlM0QlMjBUcnVlJTBBbW9kZWwlMjAlM0QlMjBSb2JlcnRhRm9yQ2F1c2FsTE0uZnJvbV9wcmV0cmFpbmVkKCUyMkZhY2Vib29rQUklMkZyb2JlcnRhLWJhc2UlMjIlMkMlMjBjb25maWclM0Rjb25maWcpJTBBJTBBaW5wdXRzJTIwJTNEJTIwdG9rZW5pemVyKCUyMkhlbGxvJTJDJTIwbXklMjBkb2clMjBpcyUyMGN1dGUlMjIlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQW91dHB1dHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyklMEElMEFwcmVkaWN0aW9uX2xvZ2l0cyUyMCUzRCUyMG91dHB1dHMubG9naXRz",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, RobertaForCausalLM, AutoConfig
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;FacebookAI/roberta-base&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>config = AutoConfig.from_pretrained(<span class="hljs-string">&quot;FacebookAI/roberta-base&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>config.is_decoder = <span class="hljs-literal">True</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = RobertaForCausalLM.from_pretrained(<span class="hljs-string">&quot;FacebookAI/roberta-base&quot;</span>, config=config)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs)

<span class="hljs-meta">&gt;&gt;&gt; </span>prediction_logits = outputs.logits`,wrap:!1}}),{c(){t=c("p"),t.textContent=m,n=a(),f(i.$$.fragment)},l(o){t=p(o,"P",{"data-svelte-h":!0}),T(t)!=="svelte-11lpom8"&&(t.textContent=m),n=r(o),g(i.$$.fragment,o)},m(o,u){h(o,t,u),h(o,n,u),b(i,o,u),M=!0},p:j,i(o){M||(_(i.$$.fragment,o),M=!0)},o(o){y(i.$$.fragment,o),M=!1},d(o){o&&(d(t),d(n)),k(i,o)}}}function Ia(w){let t,m=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=m},l(n){t=p(n,"P",{"data-svelte-h":!0}),T(t)!=="svelte-fincs2"&&(t.innerHTML=m)},m(n,i){h(n,t,i)},p:j,d(n){n&&d(t)}}}function qa(w){let t,m="Example:",n,i,M;return i=new P({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBSb2JlcnRhRm9yTWFza2VkTE0lMEFpbXBvcnQlMjB0b3JjaCUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMkZhY2Vib29rQUklMkZyb2JlcnRhLWJhc2UlMjIpJTBBbW9kZWwlMjAlM0QlMjBSb2JlcnRhRm9yTWFza2VkTE0uZnJvbV9wcmV0cmFpbmVkKCUyMkZhY2Vib29rQUklMkZyb2JlcnRhLWJhc2UlMjIpJTBBJTBBaW5wdXRzJTIwJTNEJTIwdG9rZW5pemVyKCUyMlRoZSUyMGNhcGl0YWwlMjBvZiUyMEZyYW5jZSUyMGlzJTIwJTNDbWFzayUzRS4lMjIlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQSUwQXdpdGglMjB0b3JjaC5ub19ncmFkKCklM0ElMEElMjAlMjAlMjAlMjBsb2dpdHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cykubG9naXRzJTBBJTBBJTIzJTIwcmV0cmlldmUlMjBpbmRleCUyMG9mJTIwJTNDbWFzayUzRSUwQW1hc2tfdG9rZW5faW5kZXglMjAlM0QlMjAoaW5wdXRzLmlucHV0X2lkcyUyMCUzRCUzRCUyMHRva2VuaXplci5tYXNrX3Rva2VuX2lkKSU1QjAlNUQubm9uemVybyhhc190dXBsZSUzRFRydWUpJTVCMCU1RCUwQSUwQXByZWRpY3RlZF90b2tlbl9pZCUyMCUzRCUyMGxvZ2l0cyU1QjAlMkMlMjBtYXNrX3Rva2VuX2luZGV4JTVELmFyZ21heChheGlzJTNELTEpJTBBdG9rZW5pemVyLmRlY29kZShwcmVkaWN0ZWRfdG9rZW5faWQpJTBBJTBBbGFiZWxzJTIwJTNEJTIwdG9rZW5pemVyKCUyMlRoZSUyMGNhcGl0YWwlMjBvZiUyMEZyYW5jZSUyMGlzJTIwUGFyaXMuJTIyJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiklNUIlMjJpbnB1dF9pZHMlMjIlNUQlMEElMjMlMjBtYXNrJTIwbGFiZWxzJTIwb2YlMjBub24tJTNDbWFzayUzRSUyMHRva2VucyUwQWxhYmVscyUyMCUzRCUyMHRvcmNoLndoZXJlKGlucHV0cy5pbnB1dF9pZHMlMjAlM0QlM0QlMjB0b2tlbml6ZXIubWFza190b2tlbl9pZCUyQyUyMGxhYmVscyUyQyUyMC0xMDApJTBBJTBBb3V0cHV0cyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzJTJDJTIwbGFiZWxzJTNEbGFiZWxzKSUwQXJvdW5kKG91dHB1dHMubG9zcy5pdGVtKCklMkMlMjAyKQ==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, RobertaForMaskedLM
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;FacebookAI/roberta-base&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = RobertaForMaskedLM.from_pretrained(<span class="hljs-string">&quot;FacebookAI/roberta-base&quot;</span>)

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
...`,wrap:!1}}),{c(){t=c("p"),t.textContent=m,n=a(),f(i.$$.fragment)},l(o){t=p(o,"P",{"data-svelte-h":!0}),T(t)!=="svelte-11lpom8"&&(t.textContent=m),n=r(o),g(i.$$.fragment,o)},m(o,u){h(o,t,u),h(o,n,u),b(i,o,u),M=!0},p:j,i(o){M||(_(i.$$.fragment,o),M=!0)},o(o){y(i.$$.fragment,o),M=!1},d(o){o&&(d(t),d(n)),k(i,o)}}}function Wa(w){let t,m=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=m},l(n){t=p(n,"P",{"data-svelte-h":!0}),T(t)!=="svelte-fincs2"&&(t.innerHTML=m)},m(n,i){h(n,t,i)},p:j,d(n){n&&d(t)}}}function Ba(w){let t,m="Example of single-label classification:",n,i,M;return i=new P({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Rva2VuaXplciUyQyUyMFJvYmVydGFGb3JTZXF1ZW5jZUNsYXNzaWZpY2F0aW9uJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIyRmFjZWJvb2tBSSUyRnJvYmVydGEtYmFzZSUyMiklMEFtb2RlbCUyMCUzRCUyMFJvYmVydGFGb3JTZXF1ZW5jZUNsYXNzaWZpY2F0aW9uLmZyb21fcHJldHJhaW5lZCglMjJGYWNlYm9va0FJJTJGcm9iZXJ0YS1iYXNlJTIyKSUwQSUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplciglMjJIZWxsbyUyQyUyMG15JTIwZG9nJTIwaXMlMjBjdXRlJTIyJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiklMEElMEF3aXRoJTIwdG9yY2gubm9fZ3JhZCgpJTNBJTBBJTIwJTIwJTIwJTIwbG9naXRzJTIwJTNEJTIwbW9kZWwoKippbnB1dHMpLmxvZ2l0cyUwQSUwQXByZWRpY3RlZF9jbGFzc19pZCUyMCUzRCUyMGxvZ2l0cy5hcmdtYXgoKS5pdGVtKCklMEFtb2RlbC5jb25maWcuaWQybGFiZWwlNUJwcmVkaWN0ZWRfY2xhc3NfaWQlNUQlMEElMEElMjMlMjBUbyUyMHRyYWluJTIwYSUyMG1vZGVsJTIwb24lMjAlNjBudW1fbGFiZWxzJTYwJTIwY2xhc3NlcyUyQyUyMHlvdSUyMGNhbiUyMHBhc3MlMjAlNjBudW1fbGFiZWxzJTNEbnVtX2xhYmVscyU2MCUyMHRvJTIwJTYwLmZyb21fcHJldHJhaW5lZCguLi4pJTYwJTBBbnVtX2xhYmVscyUyMCUzRCUyMGxlbihtb2RlbC5jb25maWcuaWQybGFiZWwpJTBBbW9kZWwlMjAlM0QlMjBSb2JlcnRhRm9yU2VxdWVuY2VDbGFzc2lmaWNhdGlvbi5mcm9tX3ByZXRyYWluZWQoJTIyRmFjZWJvb2tBSSUyRnJvYmVydGEtYmFzZSUyMiUyQyUyMG51bV9sYWJlbHMlM0RudW1fbGFiZWxzKSUwQSUwQWxhYmVscyUyMCUzRCUyMHRvcmNoLnRlbnNvciglNUIxJTVEKSUwQWxvc3MlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyUyQyUyMGxhYmVscyUzRGxhYmVscykubG9zcyUwQXJvdW5kKGxvc3MuaXRlbSgpJTJDJTIwMik=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, RobertaForSequenceClassification

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;FacebookAI/roberta-base&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = RobertaForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;FacebookAI/roberta-base&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    logits = model(**inputs).logits

<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_class_id = logits.argmax().item()
<span class="hljs-meta">&gt;&gt;&gt; </span>model.config.id2label[predicted_class_id]
...

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># To train a model on \`num_labels\` classes, you can pass \`num_labels=num_labels\` to \`.from_pretrained(...)\`</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>num_labels = <span class="hljs-built_in">len</span>(model.config.id2label)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = RobertaForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;FacebookAI/roberta-base&quot;</span>, num_labels=num_labels)

<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.tensor([<span class="hljs-number">1</span>])
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = model(**inputs, labels=labels).loss
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">round</span>(loss.item(), <span class="hljs-number">2</span>)
...`,wrap:!1}}),{c(){t=c("p"),t.textContent=m,n=a(),f(i.$$.fragment)},l(o){t=p(o,"P",{"data-svelte-h":!0}),T(t)!=="svelte-ykxpe4"&&(t.textContent=m),n=r(o),g(i.$$.fragment,o)},m(o,u){h(o,t,u),h(o,n,u),b(i,o,u),M=!0},p:j,i(o){M||(_(i.$$.fragment,o),M=!0)},o(o){y(i.$$.fragment,o),M=!1},d(o){o&&(d(t),d(n)),k(i,o)}}}function Za(w){let t,m="Example of multi-label classification:",n,i,M;return i=new P({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Rva2VuaXplciUyQyUyMFJvYmVydGFGb3JTZXF1ZW5jZUNsYXNzaWZpY2F0aW9uJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIyRmFjZWJvb2tBSSUyRnJvYmVydGEtYmFzZSUyMiklMEFtb2RlbCUyMCUzRCUyMFJvYmVydGFGb3JTZXF1ZW5jZUNsYXNzaWZpY2F0aW9uLmZyb21fcHJldHJhaW5lZCglMjJGYWNlYm9va0FJJTJGcm9iZXJ0YS1iYXNlJTIyJTJDJTIwcHJvYmxlbV90eXBlJTNEJTIybXVsdGlfbGFiZWxfY2xhc3NpZmljYXRpb24lMjIpJTBBJTBBaW5wdXRzJTIwJTNEJTIwdG9rZW5pemVyKCUyMkhlbGxvJTJDJTIwbXklMjBkb2clMjBpcyUyMGN1dGUlMjIlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQSUwQXdpdGglMjB0b3JjaC5ub19ncmFkKCklM0ElMEElMjAlMjAlMjAlMjBsb2dpdHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cykubG9naXRzJTBBJTBBcHJlZGljdGVkX2NsYXNzX2lkcyUyMCUzRCUyMHRvcmNoLmFyYW5nZSgwJTJDJTIwbG9naXRzLnNoYXBlJTVCLTElNUQpJTVCdG9yY2guc2lnbW9pZChsb2dpdHMpLnNxdWVlemUoZGltJTNEMCklMjAlM0UlMjAwLjUlNUQlMEElMEElMjMlMjBUbyUyMHRyYWluJTIwYSUyMG1vZGVsJTIwb24lMjAlNjBudW1fbGFiZWxzJTYwJTIwY2xhc3NlcyUyQyUyMHlvdSUyMGNhbiUyMHBhc3MlMjAlNjBudW1fbGFiZWxzJTNEbnVtX2xhYmVscyU2MCUyMHRvJTIwJTYwLmZyb21fcHJldHJhaW5lZCguLi4pJTYwJTBBbnVtX2xhYmVscyUyMCUzRCUyMGxlbihtb2RlbC5jb25maWcuaWQybGFiZWwpJTBBbW9kZWwlMjAlM0QlMjBSb2JlcnRhRm9yU2VxdWVuY2VDbGFzc2lmaWNhdGlvbi5mcm9tX3ByZXRyYWluZWQoJTBBJTIwJTIwJTIwJTIwJTIyRmFjZWJvb2tBSSUyRnJvYmVydGEtYmFzZSUyMiUyQyUyMG51bV9sYWJlbHMlM0RudW1fbGFiZWxzJTJDJTIwcHJvYmxlbV90eXBlJTNEJTIybXVsdGlfbGFiZWxfY2xhc3NpZmljYXRpb24lMjIlMEEpJTBBJTBBbGFiZWxzJTIwJTNEJTIwdG9yY2guc3VtKCUwQSUyMCUyMCUyMCUyMHRvcmNoLm5uLmZ1bmN0aW9uYWwub25lX2hvdChwcmVkaWN0ZWRfY2xhc3NfaWRzJTVCTm9uZSUyQyUyMCUzQSU1RC5jbG9uZSgpJTJDJTIwbnVtX2NsYXNzZXMlM0RudW1fbGFiZWxzKSUyQyUyMGRpbSUzRDElMEEpLnRvKHRvcmNoLmZsb2F0KSUwQWxvc3MlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyUyQyUyMGxhYmVscyUzRGxhYmVscykubG9zcw==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, RobertaForSequenceClassification

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;FacebookAI/roberta-base&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = RobertaForSequenceClassification.from_pretrained(<span class="hljs-string">&quot;FacebookAI/roberta-base&quot;</span>, problem_type=<span class="hljs-string">&quot;multi_label_classification&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">with</span> torch.no_grad():
<span class="hljs-meta">... </span>    logits = model(**inputs).logits

<span class="hljs-meta">&gt;&gt;&gt; </span>predicted_class_ids = torch.arange(<span class="hljs-number">0</span>, logits.shape[-<span class="hljs-number">1</span>])[torch.sigmoid(logits).squeeze(dim=<span class="hljs-number">0</span>) &gt; <span class="hljs-number">0.5</span>]

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># To train a model on \`num_labels\` classes, you can pass \`num_labels=num_labels\` to \`.from_pretrained(...)\`</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>num_labels = <span class="hljs-built_in">len</span>(model.config.id2label)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = RobertaForSequenceClassification.from_pretrained(
<span class="hljs-meta">... </span>    <span class="hljs-string">&quot;FacebookAI/roberta-base&quot;</span>, num_labels=num_labels, problem_type=<span class="hljs-string">&quot;multi_label_classification&quot;</span>
<span class="hljs-meta">... </span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.<span class="hljs-built_in">sum</span>(
<span class="hljs-meta">... </span>    torch.nn.functional.one_hot(predicted_class_ids[<span class="hljs-literal">None</span>, :].clone(), num_classes=num_labels), dim=<span class="hljs-number">1</span>
<span class="hljs-meta">... </span>).to(torch.<span class="hljs-built_in">float</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = model(**inputs, labels=labels).loss`,wrap:!1}}),{c(){t=c("p"),t.textContent=m,n=a(),f(i.$$.fragment)},l(o){t=p(o,"P",{"data-svelte-h":!0}),T(t)!=="svelte-1l8e32d"&&(t.textContent=m),n=r(o),g(i.$$.fragment,o)},m(o,u){h(o,t,u),h(o,n,u),b(i,o,u),M=!0},p:j,i(o){M||(_(i.$$.fragment,o),M=!0)},o(o){y(i.$$.fragment,o),M=!1},d(o){o&&(d(t),d(n)),k(i,o)}}}function Ga(w){let t,m=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=m},l(n){t=p(n,"P",{"data-svelte-h":!0}),T(t)!=="svelte-fincs2"&&(t.innerHTML=m)},m(n,i){h(n,t,i)},p:j,d(n){n&&d(t)}}}function La(w){let t,m="Example:",n,i,M;return i=new P({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBSb2JlcnRhRm9yTXVsdGlwbGVDaG9pY2UlMEFpbXBvcnQlMjB0b3JjaCUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMkZhY2Vib29rQUklMkZyb2JlcnRhLWJhc2UlMjIpJTBBbW9kZWwlMjAlM0QlMjBSb2JlcnRhRm9yTXVsdGlwbGVDaG9pY2UuZnJvbV9wcmV0cmFpbmVkKCUyMkZhY2Vib29rQUklMkZyb2JlcnRhLWJhc2UlMjIpJTBBJTBBcHJvbXB0JTIwJTNEJTIwJTIySW4lMjBJdGFseSUyQyUyMHBpenphJTIwc2VydmVkJTIwaW4lMjBmb3JtYWwlMjBzZXR0aW5ncyUyQyUyMHN1Y2glMjBhcyUyMGF0JTIwYSUyMHJlc3RhdXJhbnQlMkMlMjBpcyUyMHByZXNlbnRlZCUyMHVuc2xpY2VkLiUyMiUwQWNob2ljZTAlMjAlM0QlMjAlMjJJdCUyMGlzJTIwZWF0ZW4lMjB3aXRoJTIwYSUyMGZvcmslMjBhbmQlMjBhJTIwa25pZmUuJTIyJTBBY2hvaWNlMSUyMCUzRCUyMCUyMkl0JTIwaXMlMjBlYXRlbiUyMHdoaWxlJTIwaGVsZCUyMGluJTIwdGhlJTIwaGFuZC4lMjIlMEFsYWJlbHMlMjAlM0QlMjB0b3JjaC50ZW5zb3IoMCkudW5zcXVlZXplKDApJTIwJTIwJTIzJTIwY2hvaWNlMCUyMGlzJTIwY29ycmVjdCUyMChhY2NvcmRpbmclMjB0byUyMFdpa2lwZWRpYSUyMCUzQikpJTJDJTIwYmF0Y2glMjBzaXplJTIwMSUwQSUwQWVuY29kaW5nJTIwJTNEJTIwdG9rZW5pemVyKCU1QnByb21wdCUyQyUyMHByb21wdCU1RCUyQyUyMCU1QmNob2ljZTAlMkMlMjBjaG9pY2UxJTVEJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiUyQyUyMHBhZGRpbmclM0RUcnVlKSUwQW91dHB1dHMlMjAlM0QlMjBtb2RlbCgqKiU3QmslM0ElMjB2LnVuc3F1ZWV6ZSgwKSUyMGZvciUyMGslMkMlMjB2JTIwaW4lMjBlbmNvZGluZy5pdGVtcygpJTdEJTJDJTIwbGFiZWxzJTNEbGFiZWxzKSUyMCUyMCUyMyUyMGJhdGNoJTIwc2l6ZSUyMGlzJTIwMSUwQSUwQSUyMyUyMHRoZSUyMGxpbmVhciUyMGNsYXNzaWZpZXIlMjBzdGlsbCUyMG5lZWRzJTIwdG8lMjBiZSUyMHRyYWluZWQlMEFsb3NzJTIwJTNEJTIwb3V0cHV0cy5sb3NzJTBBbG9naXRzJTIwJTNEJTIwb3V0cHV0cy5sb2dpdHM=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, RobertaForMultipleChoice
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;FacebookAI/roberta-base&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = RobertaForMultipleChoice.from_pretrained(<span class="hljs-string">&quot;FacebookAI/roberta-base&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>prompt = <span class="hljs-string">&quot;In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>choice0 = <span class="hljs-string">&quot;It is eaten with a fork and a knife.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>choice1 = <span class="hljs-string">&quot;It is eaten while held in the hand.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>labels = torch.tensor(<span class="hljs-number">0</span>).unsqueeze(<span class="hljs-number">0</span>)  <span class="hljs-comment"># choice0 is correct (according to Wikipedia ;)), batch size 1</span>

<span class="hljs-meta">&gt;&gt;&gt; </span>encoding = tokenizer([prompt, prompt], [choice0, choice1], return_tensors=<span class="hljs-string">&quot;pt&quot;</span>, padding=<span class="hljs-literal">True</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**{k: v.unsqueeze(<span class="hljs-number">0</span>) <span class="hljs-keyword">for</span> k, v <span class="hljs-keyword">in</span> encoding.items()}, labels=labels)  <span class="hljs-comment"># batch size is 1</span>

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># the linear classifier still needs to be trained</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>loss = outputs.loss
<span class="hljs-meta">&gt;&gt;&gt; </span>logits = outputs.logits`,wrap:!1}}),{c(){t=c("p"),t.textContent=m,n=a(),f(i.$$.fragment)},l(o){t=p(o,"P",{"data-svelte-h":!0}),T(t)!=="svelte-11lpom8"&&(t.textContent=m),n=r(o),g(i.$$.fragment,o)},m(o,u){h(o,t,u),h(o,n,u),b(i,o,u),M=!0},p:j,i(o){M||(_(i.$$.fragment,o),M=!0)},o(o){y(i.$$.fragment,o),M=!1},d(o){o&&(d(t),d(n)),k(i,o)}}}function Va(w){let t,m=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=m},l(n){t=p(n,"P",{"data-svelte-h":!0}),T(t)!=="svelte-fincs2"&&(t.innerHTML=m)},m(n,i){h(n,t,i)},p:j,d(n){n&&d(t)}}}function Na(w){let t,m="Example:",n,i,M;return i=new P({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBSb2JlcnRhRm9yVG9rZW5DbGFzc2lmaWNhdGlvbiUwQWltcG9ydCUyMHRvcmNoJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIyRmFjZWJvb2tBSSUyRnJvYmVydGEtYmFzZSUyMiklMEFtb2RlbCUyMCUzRCUyMFJvYmVydGFGb3JUb2tlbkNsYXNzaWZpY2F0aW9uLmZyb21fcHJldHJhaW5lZCglMjJGYWNlYm9va0FJJTJGcm9iZXJ0YS1iYXNlJTIyKSUwQSUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplciglMEElMjAlMjAlMjAlMjAlMjJIdWdnaW5nRmFjZSUyMGlzJTIwYSUyMGNvbXBhbnklMjBiYXNlZCUyMGluJTIwUGFyaXMlMjBhbmQlMjBOZXclMjBZb3JrJTIyJTJDJTIwYWRkX3NwZWNpYWxfdG9rZW5zJTNERmFsc2UlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyJTBBKSUwQSUwQXdpdGglMjB0b3JjaC5ub19ncmFkKCklM0ElMEElMjAlMjAlMjAlMjBsb2dpdHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cykubG9naXRzJTBBJTBBcHJlZGljdGVkX3Rva2VuX2NsYXNzX2lkcyUyMCUzRCUyMGxvZ2l0cy5hcmdtYXgoLTEpJTBBJTBBJTIzJTIwTm90ZSUyMHRoYXQlMjB0b2tlbnMlMjBhcmUlMjBjbGFzc2lmaWVkJTIwcmF0aGVyJTIwdGhlbiUyMGlucHV0JTIwd29yZHMlMjB3aGljaCUyMG1lYW5zJTIwdGhhdCUwQSUyMyUyMHRoZXJlJTIwbWlnaHQlMjBiZSUyMG1vcmUlMjBwcmVkaWN0ZWQlMjB0b2tlbiUyMGNsYXNzZXMlMjB0aGFuJTIwd29yZHMuJTBBJTIzJTIwTXVsdGlwbGUlMjB0b2tlbiUyMGNsYXNzZXMlMjBtaWdodCUyMGFjY291bnQlMjBmb3IlMjB0aGUlMjBzYW1lJTIwd29yZCUwQXByZWRpY3RlZF90b2tlbnNfY2xhc3NlcyUyMCUzRCUyMCU1Qm1vZGVsLmNvbmZpZy5pZDJsYWJlbCU1QnQuaXRlbSgpJTVEJTIwZm9yJTIwdCUyMGluJTIwcHJlZGljdGVkX3Rva2VuX2NsYXNzX2lkcyU1QjAlNUQlNUQlMEFwcmVkaWN0ZWRfdG9rZW5zX2NsYXNzZXMlMEElMEFsYWJlbHMlMjAlM0QlMjBwcmVkaWN0ZWRfdG9rZW5fY2xhc3NfaWRzJTBBbG9zcyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzJTJDJTIwbGFiZWxzJTNEbGFiZWxzKS5sb3NzJTBBcm91bmQobG9zcy5pdGVtKCklMkMlMjAyKQ==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, RobertaForTokenClassification
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;FacebookAI/roberta-base&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = RobertaForTokenClassification.from_pretrained(<span class="hljs-string">&quot;FacebookAI/roberta-base&quot;</span>)

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
...`,wrap:!1}}),{c(){t=c("p"),t.textContent=m,n=a(),f(i.$$.fragment)},l(o){t=p(o,"P",{"data-svelte-h":!0}),T(t)!=="svelte-11lpom8"&&(t.textContent=m),n=r(o),g(i.$$.fragment,o)},m(o,u){h(o,t,u),h(o,n,u),b(i,o,u),M=!0},p:j,i(o){M||(_(i.$$.fragment,o),M=!0)},o(o){y(i.$$.fragment,o),M=!1},d(o){o&&(d(t),d(n)),k(i,o)}}}function Ha(w){let t,m=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=m},l(n){t=p(n,"P",{"data-svelte-h":!0}),T(t)!=="svelte-fincs2"&&(t.innerHTML=m)},m(n,i){h(n,t,i)},p:j,d(n){n&&d(t)}}}function Xa(w){let t,m="Example:",n,i,M;return i=new P({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBSb2JlcnRhRm9yUXVlc3Rpb25BbnN3ZXJpbmclMEFpbXBvcnQlMjB0b3JjaCUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMkZhY2Vib29rQUklMkZyb2JlcnRhLWJhc2UlMjIpJTBBbW9kZWwlMjAlM0QlMjBSb2JlcnRhRm9yUXVlc3Rpb25BbnN3ZXJpbmcuZnJvbV9wcmV0cmFpbmVkKCUyMkZhY2Vib29rQUklMkZyb2JlcnRhLWJhc2UlMjIpJTBBJTBBcXVlc3Rpb24lMkMlMjB0ZXh0JTIwJTNEJTIwJTIyV2hvJTIwd2FzJTIwSmltJTIwSGVuc29uJTNGJTIyJTJDJTIwJTIySmltJTIwSGVuc29uJTIwd2FzJTIwYSUyMG5pY2UlMjBwdXBwZXQlMjIlMEElMEFpbnB1dHMlMjAlM0QlMjB0b2tlbml6ZXIocXVlc3Rpb24lMkMlMjB0ZXh0JTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiklMEF3aXRoJTIwdG9yY2gubm9fZ3JhZCgpJTNBJTBBJTIwJTIwJTIwJTIwb3V0cHV0cyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzKSUwQSUwQWFuc3dlcl9zdGFydF9pbmRleCUyMCUzRCUyMG91dHB1dHMuc3RhcnRfbG9naXRzLmFyZ21heCgpJTBBYW5zd2VyX2VuZF9pbmRleCUyMCUzRCUyMG91dHB1dHMuZW5kX2xvZ2l0cy5hcmdtYXgoKSUwQSUwQXByZWRpY3RfYW5zd2VyX3Rva2VucyUyMCUzRCUyMGlucHV0cy5pbnB1dF9pZHMlNUIwJTJDJTIwYW5zd2VyX3N0YXJ0X2luZGV4JTIwJTNBJTIwYW5zd2VyX2VuZF9pbmRleCUyMCUyQiUyMDElNUQlMEF0b2tlbml6ZXIuZGVjb2RlKHByZWRpY3RfYW5zd2VyX3Rva2VucyUyQyUyMHNraXBfc3BlY2lhbF90b2tlbnMlM0RUcnVlKSUwQSUwQSUyMyUyMHRhcmdldCUyMGlzJTIwJTIybmljZSUyMHB1cHBldCUyMiUwQXRhcmdldF9zdGFydF9pbmRleCUyMCUzRCUyMHRvcmNoLnRlbnNvciglNUIxNCU1RCklMEF0YXJnZXRfZW5kX2luZGV4JTIwJTNEJTIwdG9yY2gudGVuc29yKCU1QjE1JTVEKSUwQSUwQW91dHB1dHMlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyUyQyUyMHN0YXJ0X3Bvc2l0aW9ucyUzRHRhcmdldF9zdGFydF9pbmRleCUyQyUyMGVuZF9wb3NpdGlvbnMlM0R0YXJnZXRfZW5kX2luZGV4KSUwQWxvc3MlMjAlM0QlMjBvdXRwdXRzLmxvc3MlMEFyb3VuZChsb3NzLml0ZW0oKSUyQyUyMDIp",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, RobertaForQuestionAnswering
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;FacebookAI/roberta-base&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = RobertaForQuestionAnswering.from_pretrained(<span class="hljs-string">&quot;FacebookAI/roberta-base&quot;</span>)

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
...`,wrap:!1}}),{c(){t=c("p"),t.textContent=m,n=a(),f(i.$$.fragment)},l(o){t=p(o,"P",{"data-svelte-h":!0}),T(t)!=="svelte-11lpom8"&&(t.textContent=m),n=r(o),g(i.$$.fragment,o)},m(o,u){h(o,t,u),h(o,n,u),b(i,o,u),M=!0},p:j,i(o){M||(_(i.$$.fragment,o),M=!0)},o(o){y(i.$$.fragment,o),M=!1},d(o){o&&(d(t),d(n)),k(i,o)}}}function Sa(w){let t,m,n,i,M,o="<em>This model was released on 2019-07-26 and added to Hugging Face Transformers on 2020-11-16.</em>",u,z,_o='<div class="flex flex-wrap space-x-1"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/> <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/></div>',Ge,le,Mo,Le,ks='<a href="https://huggingface.co/papers/1907.11692" rel="nofollow">RoBERTa</a> improves BERT with new pretraining objectives, demonstrating <a href="./bert">BERT</a> was undertrained and training design is important. The pretraining objectives include dynamic masking, sentence packing, larger batches and a byte-level BPE tokenizer.',wo,Ve,Ts='You can find all the original RoBERTa checkpoints under the <a href="https://huggingface.co/FacebookAI" rel="nofollow">Facebook AI</a> organization.',vo,fe,$o,Ne,Ms='The example below demonstrates how to predict the <code>&lt;mask&gt;</code> token with <a href="/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.Pipeline">Pipeline</a>, <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel">AutoModel</a>, and from the command line.',Ro,ge,Jo,He,jo,Xe,ws="<li>RoBERTa doesn’t have <code>token_type_ids</code> so you don’t need to indicate which token belongs to which segment. Separate your segments with the separation token <code>tokenizer.sep_token</code> or <code>&lt;/s&gt;</code>.</li>",Co,Se,zo,N,Ee,Do,Rt,vs=`This is the configuration class to store the configuration of a <a href="/docs/transformers/v4.56.2/en/model_doc/roberta#transformers.RobertaModel">RobertaModel</a> or a <code>TFRobertaModel</code>. It is
used to instantiate a RoBERTa model according to the specified arguments, defining the model architecture.
Instantiating a configuration with the defaults will yield a similar configuration to that of the RoBERTa
<a href="https://huggingface.co/FacebookAI/roberta-base" rel="nofollow">FacebookAI/roberta-base</a> architecture.`,Ko,Jt,$s=`Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> and can be used to control the model outputs. Read the
documentation from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> for more information.`,en,be,Fo,Ae,Uo,v,Ye,tn,jt,Rs="Constructs a RoBERTa tokenizer, derived from the GPT-2 tokenizer, using byte-level Byte-Pair-Encoding.",on,Ct,Js="This tokenizer has been trained to treat spaces like parts of the tokens (a bit like sentencepiece) so a word will",nn,_e,sn,zt,js=`You can get around that behavior by passing <code>add_prefix_space=True</code> when instantiating this tokenizer or when you
call it on some text, but since the model was not pretrained this way, it might yield a decrease in performance.`,an,ye,rn,Ft,Cs=`This tokenizer inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer">PreTrainedTokenizer</a> which contains most of the main methods. Users should refer to
this superclass for more information regarding those methods.`,ln,de,Qe,dn,Ut,zs=`Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
adding special tokens. A RoBERTa sequence has the following format:`,cn,xt,Fs="<li>single sequence: <code>&lt;s&gt; X &lt;/s&gt;</code></li> <li>pair of sequences: <code>&lt;s&gt; A &lt;/s&gt;&lt;/s&gt; B &lt;/s&gt;</code></li>",pn,ke,Pe,mn,It,Us=`Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
special tokens using the tokenizer <code>prepare_for_model</code> method.`,hn,Te,Oe,un,qt,xs=`Create a mask from the two sequences passed to be used in a sequence-pair classification task. RoBERTa does not
make use of token type ids, therefore a list of zeros is returned.`,fn,Wt,De,xo,Ke,Io,F,et,gn,Bt,Is=`Construct a “fast” RoBERTa tokenizer (backed by HuggingFace’s <em>tokenizers</em> library), derived from the GPT-2
tokenizer, using byte-level Byte-Pair-Encoding.`,bn,Zt,qs="This tokenizer has been trained to treat spaces like parts of the tokens (a bit like sentencepiece) so a word will",_n,Me,yn,Gt,Ws=`You can get around that behavior by passing <code>add_prefix_space=True</code> when instantiating this tokenizer or when you
call it on some text, but since the model was not pretrained this way, it might yield a decrease in performance.`,kn,we,Tn,Lt,Bs=`This tokenizer inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast">PreTrainedTokenizerFast</a> which contains most of the main methods. Users should
refer to this superclass for more information regarding those methods.`,Mn,Vt,tt,qo,ot,Wo,x,nt,wn,Nt,Zs=`The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
cross-attention is added between the self-attention layers, following the architecture described in <a href="https://huggingface.co/papers/1706.03762" rel="nofollow">Attention is
all you need</a> by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.`,vn,Ht,Gs=`To behave as an decoder the model needs to be initialized with the <code>is_decoder</code> argument of the configuration set
to <code>True</code>. To be used in a Seq2Seq model, the model needs to initialized with both <code>is_decoder</code> argument and
<code>add_cross_attention</code> set to <code>True</code>; an <code>encoder_hidden_states</code> is then expected as an input to the forward pass.`,$n,Xt,Ls=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Rn,St,Vs=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Jn,ce,st,jn,Et,Ns='The <a href="/docs/transformers/v4.56.2/en/model_doc/roberta#transformers.RobertaModel">RobertaModel</a> forward method, overrides the <code>__call__</code> special method.',Cn,ve,Bo,at,Zo,I,rt,zn,At,Hs="RoBERTa Model with a <code>language modeling</code> head on top for CLM fine-tuning.",Fn,Yt,Xs=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Un,Qt,Ss=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,xn,O,it,In,Pt,Es='The <a href="/docs/transformers/v4.56.2/en/model_doc/roberta#transformers.RobertaForCausalLM">RobertaForCausalLM</a> forward method, overrides the <code>__call__</code> special method.',qn,$e,Wn,Re,Go,lt,Lo,q,dt,Bn,Ot,As="The Roberta Model with a <code>language modeling</code> head on top.”",Zn,Dt,Ys=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Gn,Kt,Qs=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Ln,D,ct,Vn,eo,Ps='The <a href="/docs/transformers/v4.56.2/en/model_doc/roberta#transformers.RobertaForMaskedLM">RobertaForMaskedLM</a> forward method, overrides the <code>__call__</code> special method.',Nn,Je,Hn,je,Vo,pt,No,W,mt,Xn,to,Os=`RoBERTa Model transformer with a sequence classification/regression head on top (a linear layer on top of the
pooled output) e.g. for GLUE tasks.`,Sn,oo,Ds=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,En,no,Ks=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,An,V,ht,Yn,so,ea='The <a href="/docs/transformers/v4.56.2/en/model_doc/roberta#transformers.RobertaForSequenceClassification">RobertaForSequenceClassification</a> forward method, overrides the <code>__call__</code> special method.',Qn,Ce,Pn,ze,On,Fe,Ho,ut,Xo,B,ft,Dn,ao,ta=`The Roberta Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
softmax) e.g. for RocStories/SWAG tasks.`,Kn,ro,oa=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,es,io,na=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,ts,K,gt,os,lo,sa='The <a href="/docs/transformers/v4.56.2/en/model_doc/roberta#transformers.RobertaForMultipleChoice">RobertaForMultipleChoice</a> forward method, overrides the <code>__call__</code> special method.',ns,Ue,ss,xe,So,bt,Eo,Z,_t,as,co,aa=`The Roberta transformer with a token classification head on top (a linear layer on top of the hidden-states
output) e.g. for Named-Entity-Recognition (NER) tasks.`,rs,po,ra=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,is,mo,ia=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,ls,ee,yt,ds,ho,la='The <a href="/docs/transformers/v4.56.2/en/model_doc/roberta#transformers.RobertaForTokenClassification">RobertaForTokenClassification</a> forward method, overrides the <code>__call__</code> special method.',cs,Ie,ps,qe,Ao,kt,Yo,G,Tt,ms,uo,da=`The Roberta transformer with a span classification head on top for extractive question-answering tasks like
SQuAD (a linear layer on top of the hidden-states output to compute <code>span start logits</code> and <code>span end logits</code>).`,hs,fo,ca=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,us,go,pa=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,fs,te,Mt,gs,bo,ma='The <a href="/docs/transformers/v4.56.2/en/model_doc/roberta#transformers.RobertaForQuestionAnswering">RobertaForQuestionAnswering</a> forward method, overrides the <code>__call__</code> special method.',bs,We,_s,Be,Qo,wt,Po,yo,Oo;return le=new oe({props:{title:"RoBERTa",local:"roberta",headingTag:"h1"}}),fe=new pe({props:{warning:!1,$$slots:{default:[Ta]},$$scope:{ctx:w}}}),ge=new ka({props:{id:"usage",options:["Pipeline","AutoModel","transformers CLI"],$$slots:{default:[$a]},$$scope:{ctx:w}}}),He=new oe({props:{title:"Notes",local:"notes",headingTag:"h2"}}),Se=new oe({props:{title:"RobertaConfig",local:"transformers.RobertaConfig",headingTag:"h2"}}),Ee=new C({props:{name:"class transformers.RobertaConfig",anchor:"transformers.RobertaConfig",parameters:[{name:"vocab_size",val:" = 50265"},{name:"hidden_size",val:" = 768"},{name:"num_hidden_layers",val:" = 12"},{name:"num_attention_heads",val:" = 12"},{name:"intermediate_size",val:" = 3072"},{name:"hidden_act",val:" = 'gelu'"},{name:"hidden_dropout_prob",val:" = 0.1"},{name:"attention_probs_dropout_prob",val:" = 0.1"},{name:"max_position_embeddings",val:" = 512"},{name:"type_vocab_size",val:" = 2"},{name:"initializer_range",val:" = 0.02"},{name:"layer_norm_eps",val:" = 1e-12"},{name:"pad_token_id",val:" = 1"},{name:"bos_token_id",val:" = 0"},{name:"eos_token_id",val:" = 2"},{name:"position_embedding_type",val:" = 'absolute'"},{name:"use_cache",val:" = True"},{name:"classifier_dropout",val:" = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.RobertaConfig.vocab_size",description:`<strong>vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 50265) &#x2014;
Vocabulary size of the RoBERTa model. Defines the number of different tokens that can be represented by the
<code>inputs_ids</code> passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/roberta#transformers.RobertaModel">RobertaModel</a> or <code>TFRobertaModel</code>.`,name:"vocab_size"},{anchor:"transformers.RobertaConfig.hidden_size",description:`<strong>hidden_size</strong> (<code>int</code>, <em>optional</em>, defaults to 768) &#x2014;
Dimensionality of the encoder layers and the pooler layer.`,name:"hidden_size"},{anchor:"transformers.RobertaConfig.num_hidden_layers",description:`<strong>num_hidden_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 12) &#x2014;
Number of hidden layers in the Transformer encoder.`,name:"num_hidden_layers"},{anchor:"transformers.RobertaConfig.num_attention_heads",description:`<strong>num_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 12) &#x2014;
Number of attention heads for each attention layer in the Transformer encoder.`,name:"num_attention_heads"},{anchor:"transformers.RobertaConfig.intermediate_size",description:`<strong>intermediate_size</strong> (<code>int</code>, <em>optional</em>, defaults to 3072) &#x2014;
Dimensionality of the &#x201C;intermediate&#x201D; (often named feed-forward) layer in the Transformer encoder.`,name:"intermediate_size"},{anchor:"transformers.RobertaConfig.hidden_act",description:`<strong>hidden_act</strong> (<code>str</code> or <code>Callable</code>, <em>optional</em>, defaults to <code>&quot;gelu&quot;</code>) &#x2014;
The non-linear activation function (function or string) in the encoder and pooler. If string, <code>&quot;gelu&quot;</code>,
<code>&quot;relu&quot;</code>, <code>&quot;silu&quot;</code> and <code>&quot;gelu_new&quot;</code> are supported.`,name:"hidden_act"},{anchor:"transformers.RobertaConfig.hidden_dropout_prob",description:`<strong>hidden_dropout_prob</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.`,name:"hidden_dropout_prob"},{anchor:"transformers.RobertaConfig.attention_probs_dropout_prob",description:`<strong>attention_probs_dropout_prob</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout ratio for the attention probabilities.`,name:"attention_probs_dropout_prob"},{anchor:"transformers.RobertaConfig.max_position_embeddings",description:`<strong>max_position_embeddings</strong> (<code>int</code>, <em>optional</em>, defaults to 512) &#x2014;
The maximum sequence length that this model might ever be used with. Typically set this to something large
just in case (e.g., 512 or 1024 or 2048).`,name:"max_position_embeddings"},{anchor:"transformers.RobertaConfig.type_vocab_size",description:`<strong>type_vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 2) &#x2014;
The vocabulary size of the <code>token_type_ids</code> passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/roberta#transformers.RobertaModel">RobertaModel</a> or <code>TFRobertaModel</code>.`,name:"type_vocab_size"},{anchor:"transformers.RobertaConfig.initializer_range",description:`<strong>initializer_range</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.`,name:"initializer_range"},{anchor:"transformers.RobertaConfig.layer_norm_eps",description:`<strong>layer_norm_eps</strong> (<code>float</code>, <em>optional</em>, defaults to 1e-12) &#x2014;
The epsilon used by the layer normalization layers.`,name:"layer_norm_eps"},{anchor:"transformers.RobertaConfig.position_embedding_type",description:`<strong>position_embedding_type</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;absolute&quot;</code>) &#x2014;
Type of position embedding. Choose one of <code>&quot;absolute&quot;</code>, <code>&quot;relative_key&quot;</code>, <code>&quot;relative_key_query&quot;</code>. For
positional embeddings use <code>&quot;absolute&quot;</code>. For more information on <code>&quot;relative_key&quot;</code>, please refer to
<a href="https://huggingface.co/papers/1803.02155" rel="nofollow">Self-Attention with Relative Position Representations (Shaw et al.)</a>.
For more information on <code>&quot;relative_key_query&quot;</code>, please refer to <em>Method 4</em> in <a href="https://huggingface.co/papers/2009.13658" rel="nofollow">Improve Transformer Models
with Better Relative Position Embeddings (Huang et al.)</a>.`,name:"position_embedding_type"},{anchor:"transformers.RobertaConfig.is_decoder",description:`<strong>is_decoder</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether the model is used as a decoder or not. If <code>False</code>, the model is used as an encoder.`,name:"is_decoder"},{anchor:"transformers.RobertaConfig.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not the model should return the last key/values attentions (not used by all models). Only
relevant if <code>config.is_decoder=True</code>.`,name:"use_cache"},{anchor:"transformers.RobertaConfig.classifier_dropout",description:`<strong>classifier_dropout</strong> (<code>float</code>, <em>optional</em>) &#x2014;
The dropout ratio for the classification head.`,name:"classifier_dropout"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/roberta/configuration_roberta.py#L29"}}),be=new me({props:{anchor:"transformers.RobertaConfig.example",$$slots:{default:[Ra]},$$scope:{ctx:w}}}),Ae=new oe({props:{title:"RobertaTokenizer",local:"transformers.RobertaTokenizer",headingTag:"h2"}}),Ye=new C({props:{name:"class transformers.RobertaTokenizer",anchor:"transformers.RobertaTokenizer",parameters:[{name:"vocab_file",val:""},{name:"merges_file",val:""},{name:"errors",val:" = 'replace'"},{name:"bos_token",val:" = '<s>'"},{name:"eos_token",val:" = '</s>'"},{name:"sep_token",val:" = '</s>'"},{name:"cls_token",val:" = '<s>'"},{name:"unk_token",val:" = '<unk>'"},{name:"pad_token",val:" = '<pad>'"},{name:"mask_token",val:" = '<mask>'"},{name:"add_prefix_space",val:" = False"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.RobertaTokenizer.vocab_file",description:`<strong>vocab_file</strong> (<code>str</code>) &#x2014;
Path to the vocabulary file.`,name:"vocab_file"},{anchor:"transformers.RobertaTokenizer.merges_file",description:`<strong>merges_file</strong> (<code>str</code>) &#x2014;
Path to the merges file.`,name:"merges_file"},{anchor:"transformers.RobertaTokenizer.errors",description:`<strong>errors</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;replace&quot;</code>) &#x2014;
Paradigm to follow when decoding bytes to UTF-8. See
<a href="https://docs.python.org/3/library/stdtypes.html#bytes.decode" rel="nofollow">bytes.decode</a> for more information.`,name:"errors"},{anchor:"transformers.RobertaTokenizer.bos_token",description:`<strong>bos_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;s&gt;&quot;</code>) &#x2014;
The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.</p>
<div class="course-tip  bg-gradient-to-br dark:bg-gradient-to-r before:border-green-500 dark:before:border-green-800 from-green-50 dark:from-gray-900 to-white dark:to-gray-950 border border-green-50 text-green-700 dark:text-gray-400">
						
<p>When building a sequence using special tokens, this is not the token that is used for the beginning of
sequence. The token used is the <code>cls_token</code>.</p>

					</div>`,name:"bos_token"},{anchor:"transformers.RobertaTokenizer.eos_token",description:`<strong>eos_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;/s&gt;&quot;</code>) &#x2014;
The end of sequence token.</p>
<div class="course-tip  bg-gradient-to-br dark:bg-gradient-to-r before:border-green-500 dark:before:border-green-800 from-green-50 dark:from-gray-900 to-white dark:to-gray-950 border border-green-50 text-green-700 dark:text-gray-400">
						
<p>When building a sequence using special tokens, this is not the token that is used for the end of sequence.
The token used is the <code>sep_token</code>.</p>

					</div>`,name:"eos_token"},{anchor:"transformers.RobertaTokenizer.sep_token",description:`<strong>sep_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;/s&gt;&quot;</code>) &#x2014;
The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
sequence classification or for a text and a question for question answering. It is also used as the last
token of a sequence built with special tokens.`,name:"sep_token"},{anchor:"transformers.RobertaTokenizer.cls_token",description:`<strong>cls_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;s&gt;&quot;</code>) &#x2014;
The classifier token which is used when doing sequence classification (classification of the whole sequence
instead of per-token classification). It is the first token of the sequence when built with special tokens.`,name:"cls_token"},{anchor:"transformers.RobertaTokenizer.unk_token",description:`<strong>unk_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;unk&gt;&quot;</code>) &#x2014;
The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
token instead.`,name:"unk_token"},{anchor:"transformers.RobertaTokenizer.pad_token",description:`<strong>pad_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;pad&gt;&quot;</code>) &#x2014;
The token used for padding, for example when batching sequences of different lengths.`,name:"pad_token"},{anchor:"transformers.RobertaTokenizer.mask_token",description:`<strong>mask_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;mask&gt;&quot;</code>) &#x2014;
The token used for masking values. This is the token used when training this model with masked language
modeling. This is the token which the model will try to predict.`,name:"mask_token"},{anchor:"transformers.RobertaTokenizer.add_prefix_space",description:`<strong>add_prefix_space</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to add an initial space to the input. This allows to treat the leading word just as any
other word. (RoBERTa tokenizer detect beginning of words by the preceding space).`,name:"add_prefix_space"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/roberta/tokenization_roberta.py#L75"}}),_e=new me({props:{anchor:"transformers.RobertaTokenizer.example",$$slots:{default:[Ja]},$$scope:{ctx:w}}}),ye=new pe({props:{$$slots:{default:[ja]},$$scope:{ctx:w}}}),Qe=new C({props:{name:"build_inputs_with_special_tokens",anchor:"transformers.RobertaTokenizer.build_inputs_with_special_tokens",parameters:[{name:"token_ids_0",val:": list"},{name:"token_ids_1",val:": typing.Optional[list[int]] = None"}],parametersDescription:[{anchor:"transformers.RobertaTokenizer.build_inputs_with_special_tokens.token_ids_0",description:`<strong>token_ids_0</strong> (<code>list[int]</code>) &#x2014;
List of IDs to which the special tokens will be added.`,name:"token_ids_0"},{anchor:"transformers.RobertaTokenizer.build_inputs_with_special_tokens.token_ids_1",description:`<strong>token_ids_1</strong> (<code>list[int]</code>, <em>optional</em>) &#x2014;
Optional second list of IDs for sequence pairs.`,name:"token_ids_1"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/roberta/tokenization_roberta.py#L320",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>List of <a href="../glossary#input-ids">input IDs</a> with the appropriate special tokens.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>list[int]</code></p>
`}}),Pe=new C({props:{name:"get_special_tokens_mask",anchor:"transformers.RobertaTokenizer.get_special_tokens_mask",parameters:[{name:"token_ids_0",val:": list"},{name:"token_ids_1",val:": typing.Optional[list[int]] = None"},{name:"already_has_special_tokens",val:": bool = False"}],parametersDescription:[{anchor:"transformers.RobertaTokenizer.get_special_tokens_mask.token_ids_0",description:`<strong>token_ids_0</strong> (<code>list[int]</code>) &#x2014;
List of IDs.`,name:"token_ids_0"},{anchor:"transformers.RobertaTokenizer.get_special_tokens_mask.token_ids_1",description:`<strong>token_ids_1</strong> (<code>list[int]</code>, <em>optional</em>) &#x2014;
Optional second list of IDs for sequence pairs.`,name:"token_ids_1"},{anchor:"transformers.RobertaTokenizer.get_special_tokens_mask.already_has_special_tokens",description:`<strong>already_has_special_tokens</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not the token list is already formatted with special tokens for the model.`,name:"already_has_special_tokens"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/roberta/tokenization_roberta.py#L345",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>list[int]</code></p>
`}}),Oe=new C({props:{name:"create_token_type_ids_from_sequences",anchor:"transformers.RobertaTokenizer.create_token_type_ids_from_sequences",parameters:[{name:"token_ids_0",val:": list"},{name:"token_ids_1",val:": typing.Optional[list[int]] = None"}],parametersDescription:[{anchor:"transformers.RobertaTokenizer.create_token_type_ids_from_sequences.token_ids_0",description:`<strong>token_ids_0</strong> (<code>list[int]</code>) &#x2014;
List of IDs.`,name:"token_ids_0"},{anchor:"transformers.RobertaTokenizer.create_token_type_ids_from_sequences.token_ids_1",description:`<strong>token_ids_1</strong> (<code>list[int]</code>, <em>optional</em>) &#x2014;
Optional second list of IDs for sequence pairs.`,name:"token_ids_1"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/roberta/tokenization_roberta.py#L372",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>List of zeros.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>list[int]</code></p>
`}}),De=new C({props:{name:"save_vocabulary",anchor:"transformers.RobertaTokenizer.save_vocabulary",parameters:[{name:"save_directory",val:": str"},{name:"filename_prefix",val:": typing.Optional[str] = None"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/roberta/tokenization_roberta.py#L291"}}),Ke=new oe({props:{title:"RobertaTokenizerFast",local:"transformers.RobertaTokenizerFast",headingTag:"h2"}}),et=new C({props:{name:"class transformers.RobertaTokenizerFast",anchor:"transformers.RobertaTokenizerFast",parameters:[{name:"vocab_file",val:" = None"},{name:"merges_file",val:" = None"},{name:"tokenizer_file",val:" = None"},{name:"errors",val:" = 'replace'"},{name:"bos_token",val:" = '<s>'"},{name:"eos_token",val:" = '</s>'"},{name:"sep_token",val:" = '</s>'"},{name:"cls_token",val:" = '<s>'"},{name:"unk_token",val:" = '<unk>'"},{name:"pad_token",val:" = '<pad>'"},{name:"mask_token",val:" = '<mask>'"},{name:"add_prefix_space",val:" = False"},{name:"trim_offsets",val:" = True"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.RobertaTokenizerFast.vocab_file",description:`<strong>vocab_file</strong> (<code>str</code>) &#x2014;
Path to the vocabulary file.`,name:"vocab_file"},{anchor:"transformers.RobertaTokenizerFast.merges_file",description:`<strong>merges_file</strong> (<code>str</code>) &#x2014;
Path to the merges file.`,name:"merges_file"},{anchor:"transformers.RobertaTokenizerFast.errors",description:`<strong>errors</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;replace&quot;</code>) &#x2014;
Paradigm to follow when decoding bytes to UTF-8. See
<a href="https://docs.python.org/3/library/stdtypes.html#bytes.decode" rel="nofollow">bytes.decode</a> for more information.`,name:"errors"},{anchor:"transformers.RobertaTokenizerFast.bos_token",description:`<strong>bos_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;s&gt;&quot;</code>) &#x2014;
The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.</p>
<div class="course-tip  bg-gradient-to-br dark:bg-gradient-to-r before:border-green-500 dark:before:border-green-800 from-green-50 dark:from-gray-900 to-white dark:to-gray-950 border border-green-50 text-green-700 dark:text-gray-400">
						
<p>When building a sequence using special tokens, this is not the token that is used for the beginning of
sequence. The token used is the <code>cls_token</code>.</p>

					</div>`,name:"bos_token"},{anchor:"transformers.RobertaTokenizerFast.eos_token",description:`<strong>eos_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;/s&gt;&quot;</code>) &#x2014;
The end of sequence token.</p>
<div class="course-tip  bg-gradient-to-br dark:bg-gradient-to-r before:border-green-500 dark:before:border-green-800 from-green-50 dark:from-gray-900 to-white dark:to-gray-950 border border-green-50 text-green-700 dark:text-gray-400">
						
<p>When building a sequence using special tokens, this is not the token that is used for the end of sequence.
The token used is the <code>sep_token</code>.</p>

					</div>`,name:"eos_token"},{anchor:"transformers.RobertaTokenizerFast.sep_token",description:`<strong>sep_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;/s&gt;&quot;</code>) &#x2014;
The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
sequence classification or for a text and a question for question answering. It is also used as the last
token of a sequence built with special tokens.`,name:"sep_token"},{anchor:"transformers.RobertaTokenizerFast.cls_token",description:`<strong>cls_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;s&gt;&quot;</code>) &#x2014;
The classifier token which is used when doing sequence classification (classification of the whole sequence
instead of per-token classification). It is the first token of the sequence when built with special tokens.`,name:"cls_token"},{anchor:"transformers.RobertaTokenizerFast.unk_token",description:`<strong>unk_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;unk&gt;&quot;</code>) &#x2014;
The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
token instead.`,name:"unk_token"},{anchor:"transformers.RobertaTokenizerFast.pad_token",description:`<strong>pad_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;pad&gt;&quot;</code>) &#x2014;
The token used for padding, for example when batching sequences of different lengths.`,name:"pad_token"},{anchor:"transformers.RobertaTokenizerFast.mask_token",description:`<strong>mask_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;mask&gt;&quot;</code>) &#x2014;
The token used for masking values. This is the token used when training this model with masked language
modeling. This is the token which the model will try to predict.`,name:"mask_token"},{anchor:"transformers.RobertaTokenizerFast.add_prefix_space",description:`<strong>add_prefix_space</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to add an initial space to the input. This allows to treat the leading word just as any
other word. (RoBERTa tokenizer detect beginning of words by the preceding space).`,name:"add_prefix_space"},{anchor:"transformers.RobertaTokenizerFast.trim_offsets",description:`<strong>trim_offsets</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether the post processing step should trim offsets to avoid including whitespaces.`,name:"trim_offsets"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/roberta/tokenization_roberta_fast.py#L33"}}),Me=new me({props:{anchor:"transformers.RobertaTokenizerFast.example",$$slots:{default:[Ca]},$$scope:{ctx:w}}}),we=new pe({props:{$$slots:{default:[za]},$$scope:{ctx:w}}}),tt=new C({props:{name:"build_inputs_with_special_tokens",anchor:"transformers.RobertaTokenizerFast.build_inputs_with_special_tokens",parameters:[{name:"token_ids_0",val:""},{name:"token_ids_1",val:" = None"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/roberta/tokenization_roberta_fast.py#L233"}}),ot=new oe({props:{title:"RobertaModel",local:"transformers.RobertaModel",headingTag:"h2"}}),nt=new C({props:{name:"class transformers.RobertaModel",anchor:"transformers.RobertaModel",parameters:[{name:"config",val:""},{name:"add_pooling_layer",val:" = True"}],parametersDescription:[{anchor:"transformers.RobertaModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/roberta#transformers.RobertaModel">RobertaModel</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"},{anchor:"transformers.RobertaModel.add_pooling_layer",description:`<strong>add_pooling_layer</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether to add a pooling layer`,name:"add_pooling_layer"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/roberta/modeling_roberta.py#L703"}}),st=new C({props:{name:"forward",anchor:"transformers.RobertaModel.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_hidden_states",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"past_key_values",val:": typing.Optional[list[torch.FloatTensor]] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.Tensor] = None"}],parametersDescription:[{anchor:"transformers.RobertaModel.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.RobertaModel.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.RobertaModel.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0, 1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.</li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.RobertaModel.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.RobertaModel.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.RobertaModel.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.RobertaModel.forward.encoder_hidden_states",description:`<strong>encoder_hidden_states</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
if the model is configured as a decoder.`,name:"encoder_hidden_states"},{anchor:"transformers.RobertaModel.forward.encoder_attention_mask",description:`<strong>encoder_attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
the cross-attention if the model is configured as a decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>`,name:"encoder_attention_mask"},{anchor:"transformers.RobertaModel.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>list[torch.FloatTensor]</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.RobertaModel.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.RobertaModel.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.RobertaModel.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.RobertaModel.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.RobertaModel.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.Tensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/roberta/modeling_roberta.py#L739",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions"
>transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/roberta#transformers.RobertaConfig"
>RobertaConfig</a>) and inputs.</p>
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
`}}),ve=new pe({props:{$$slots:{default:[Fa]},$$scope:{ctx:w}}}),at=new oe({props:{title:"RobertaForCausalLM",local:"transformers.RobertaForCausalLM",headingTag:"h2"}}),rt=new C({props:{name:"class transformers.RobertaForCausalLM",anchor:"transformers.RobertaForCausalLM",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.RobertaForCausalLM.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/roberta#transformers.RobertaForCausalLM">RobertaForCausalLM</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/roberta/modeling_roberta.py#L895"}}),it=new C({props:{name:"forward",anchor:"transformers.RobertaForCausalLM.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"encoder_hidden_states",val:": typing.Optional[torch.FloatTensor] = None"},{name:"encoder_attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[tuple[tuple[torch.FloatTensor]]] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.RobertaForCausalLM.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.RobertaForCausalLM.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.RobertaForCausalLM.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0,1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.
This parameter can only be used when the model is initialized with <code>type_vocab_size</code> parameter with value<blockquote>
<p>= 2. All the value in this tensor should be always &lt; type_vocab_size.</p>
</blockquote></li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.RobertaForCausalLM.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.RobertaForCausalLM.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.RobertaForCausalLM.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.RobertaForCausalLM.forward.encoder_hidden_states",description:`<strong>encoder_hidden_states</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
if the model is configured as a decoder.`,name:"encoder_hidden_states"},{anchor:"transformers.RobertaForCausalLM.forward.encoder_attention_mask",description:`<strong>encoder_attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
the cross-attention if the model is configured as a decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>`,name:"encoder_attention_mask"},{anchor:"transformers.RobertaForCausalLM.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the left-to-right language modeling loss (next word prediction). Indices should be in
<code>[-100, 0, ..., config.vocab_size]</code> (see <code>input_ids</code> docstring) Tokens with indices set to <code>-100</code> are
ignored (masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>`,name:"labels"},{anchor:"transformers.RobertaForCausalLM.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>tuple[tuple[torch.FloatTensor]]</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.RobertaForCausalLM.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.RobertaForCausalLM.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.RobertaForCausalLM.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.RobertaForCausalLM.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/roberta/modeling_roberta.py#L916",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithCrossAttentions"
>transformers.modeling_outputs.CausalLMOutputWithCrossAttentions</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/roberta#transformers.RobertaConfig"
>RobertaConfig</a>) and inputs.</p>
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
`}}),$e=new pe({props:{$$slots:{default:[Ua]},$$scope:{ctx:w}}}),Re=new me({props:{anchor:"transformers.RobertaForCausalLM.forward.example",$$slots:{default:[xa]},$$scope:{ctx:w}}}),lt=new oe({props:{title:"RobertaForMaskedLM",local:"transformers.RobertaForMaskedLM",headingTag:"h2"}}),dt=new C({props:{name:"class transformers.RobertaForMaskedLM",anchor:"transformers.RobertaForMaskedLM",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.RobertaForMaskedLM.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/roberta#transformers.RobertaForMaskedLM">RobertaForMaskedLM</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/roberta/modeling_roberta.py#L1015"}}),ct=new C({props:{name:"forward",anchor:"transformers.RobertaForMaskedLM.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"encoder_hidden_states",val:": typing.Optional[torch.FloatTensor] = None"},{name:"encoder_attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.RobertaForMaskedLM.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.RobertaForMaskedLM.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.RobertaForMaskedLM.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0,1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.
This parameter can only be used when the model is initialized with <code>type_vocab_size</code> parameter with value<blockquote>
<p>= 2. All the value in this tensor should be always &lt; type_vocab_size.</p>
</blockquote></li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.RobertaForMaskedLM.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.RobertaForMaskedLM.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.RobertaForMaskedLM.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.RobertaForMaskedLM.forward.encoder_hidden_states",description:`<strong>encoder_hidden_states</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
if the model is configured as a decoder.`,name:"encoder_hidden_states"},{anchor:"transformers.RobertaForMaskedLM.forward.encoder_attention_mask",description:`<strong>encoder_attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
the cross-attention if the model is configured as a decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>`,name:"encoder_attention_mask"},{anchor:"transformers.RobertaForMaskedLM.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should be in <code>[-100, 0, ..., config.vocab_size]</code> (see <code>input_ids</code> docstring) Tokens with indices set to <code>-100</code> are ignored (masked), the
loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>`,name:"labels"},{anchor:"transformers.RobertaForMaskedLM.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.RobertaForMaskedLM.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.RobertaForMaskedLM.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/roberta/modeling_roberta.py#L1039",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.MaskedLMOutput"
>transformers.modeling_outputs.MaskedLMOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/roberta#transformers.RobertaConfig"
>RobertaConfig</a>) and inputs.</p>
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
`}}),Je=new pe({props:{$$slots:{default:[Ia]},$$scope:{ctx:w}}}),je=new me({props:{anchor:"transformers.RobertaForMaskedLM.forward.example",$$slots:{default:[qa]},$$scope:{ctx:w}}}),pt=new oe({props:{title:"RobertaForSequenceClassification",local:"transformers.RobertaForSequenceClassification",headingTag:"h2"}}),mt=new C({props:{name:"class transformers.RobertaForSequenceClassification",anchor:"transformers.RobertaForSequenceClassification",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.RobertaForSequenceClassification.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/roberta#transformers.RobertaForSequenceClassification">RobertaForSequenceClassification</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/roberta/modeling_roberta.py#L1144"}}),ht=new C({props:{name:"forward",anchor:"transformers.RobertaForSequenceClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.RobertaForSequenceClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.RobertaForSequenceClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.RobertaForSequenceClassification.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0,1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.
This parameter can only be used when the model is initialized with <code>type_vocab_size</code> parameter with value<blockquote>
<p>= 2. All the value in this tensor should be always &lt; type_vocab_size.</p>
</blockquote></li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.RobertaForSequenceClassification.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.RobertaForSequenceClassification.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.RobertaForSequenceClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.RobertaForSequenceClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the sequence classification/regression loss. Indices should be in <code>[0, ..., config.num_labels - 1]</code>. If <code>config.num_labels == 1</code> a regression loss is computed (Mean-Square loss), If
<code>config.num_labels &gt; 1</code> a classification loss is computed (Cross-Entropy).`,name:"labels"},{anchor:"transformers.RobertaForSequenceClassification.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.RobertaForSequenceClassification.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.RobertaForSequenceClassification.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/roberta/modeling_roberta.py#L1156",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput"
>transformers.modeling_outputs.SequenceClassifierOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/roberta#transformers.RobertaConfig"
>RobertaConfig</a>) and inputs.</p>
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
`}}),Ce=new pe({props:{$$slots:{default:[Wa]},$$scope:{ctx:w}}}),ze=new me({props:{anchor:"transformers.RobertaForSequenceClassification.forward.example",$$slots:{default:[Ba]},$$scope:{ctx:w}}}),Fe=new me({props:{anchor:"transformers.RobertaForSequenceClassification.forward.example-2",$$slots:{default:[Za]},$$scope:{ctx:w}}}),ut=new oe({props:{title:"RobertaForMultipleChoice",local:"transformers.RobertaForMultipleChoice",headingTag:"h2"}}),ft=new C({props:{name:"class transformers.RobertaForMultipleChoice",anchor:"transformers.RobertaForMultipleChoice",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.RobertaForMultipleChoice.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/roberta#transformers.RobertaForMultipleChoice">RobertaForMultipleChoice</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/roberta/modeling_roberta.py#L1239"}}),gt=new C({props:{name:"forward",anchor:"transformers.RobertaForMultipleChoice.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.RobertaForMultipleChoice.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, num_choices, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.RobertaForMultipleChoice.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, num_choices, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0,1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.
This parameter can only be used when the model is initialized with <code>type_vocab_size</code> parameter with value<blockquote>
<p>= 2. All the value in this tensor should be always &lt; type_vocab_size.</p>
</blockquote></li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.RobertaForMultipleChoice.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.RobertaForMultipleChoice.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for computing the multiple choice classification loss. Indices should be in <code>[0, ..., num_choices-1]</code> where <code>num_choices</code> is the size of the second dimension of the input tensors. (See
<code>input_ids</code> above)`,name:"labels"},{anchor:"transformers.RobertaForMultipleChoice.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, num_choices, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.max_position_embeddings - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.RobertaForMultipleChoice.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.RobertaForMultipleChoice.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_choices, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.RobertaForMultipleChoice.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.RobertaForMultipleChoice.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.RobertaForMultipleChoice.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/roberta/modeling_roberta.py#L1250",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.MultipleChoiceModelOutput"
>transformers.modeling_outputs.MultipleChoiceModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/roberta#transformers.RobertaConfig"
>RobertaConfig</a>) and inputs.</p>
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
`}}),Ue=new pe({props:{$$slots:{default:[Ga]},$$scope:{ctx:w}}}),xe=new me({props:{anchor:"transformers.RobertaForMultipleChoice.forward.example",$$slots:{default:[La]},$$scope:{ctx:w}}}),bt=new oe({props:{title:"RobertaForTokenClassification",local:"transformers.RobertaForTokenClassification",headingTag:"h2"}}),_t=new C({props:{name:"class transformers.RobertaForTokenClassification",anchor:"transformers.RobertaForTokenClassification",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.RobertaForTokenClassification.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/roberta#transformers.RobertaForTokenClassification">RobertaForTokenClassification</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/roberta/modeling_roberta.py#L1345"}}),yt=new C({props:{name:"forward",anchor:"transformers.RobertaForTokenClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.RobertaForTokenClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.RobertaForTokenClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.RobertaForTokenClassification.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0,1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.
This parameter can only be used when the model is initialized with <code>type_vocab_size</code> parameter with value<blockquote>
<p>= 2. All the value in this tensor should be always &lt; type_vocab_size.</p>
</blockquote></li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.RobertaForTokenClassification.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.RobertaForTokenClassification.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.RobertaForTokenClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.RobertaForTokenClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the token classification loss. Indices should be in <code>[0, ..., config.num_labels - 1]</code>.`,name:"labels"},{anchor:"transformers.RobertaForTokenClassification.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.RobertaForTokenClassification.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.RobertaForTokenClassification.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/roberta/modeling_roberta.py#L1360",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput"
>transformers.modeling_outputs.TokenClassifierOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/roberta#transformers.RobertaConfig"
>RobertaConfig</a>) and inputs.</p>
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
`}}),Ie=new pe({props:{$$slots:{default:[Va]},$$scope:{ctx:w}}}),qe=new me({props:{anchor:"transformers.RobertaForTokenClassification.forward.example",$$slots:{default:[Na]},$$scope:{ctx:w}}}),kt=new oe({props:{title:"RobertaForQuestionAnswering",local:"transformers.RobertaForQuestionAnswering",headingTag:"h2"}}),Tt=new C({props:{name:"class transformers.RobertaForQuestionAnswering",anchor:"transformers.RobertaForQuestionAnswering",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.RobertaForQuestionAnswering.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/roberta#transformers.RobertaForQuestionAnswering">RobertaForQuestionAnswering</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/roberta/modeling_roberta.py#L1448"}}),Mt=new C({props:{name:"forward",anchor:"transformers.RobertaForQuestionAnswering.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"token_type_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"start_positions",val:": typing.Optional[torch.LongTensor] = None"},{name:"end_positions",val:": typing.Optional[torch.LongTensor] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"transformers.RobertaForQuestionAnswering.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.RobertaForQuestionAnswering.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.RobertaForQuestionAnswering.forward.token_type_ids",description:`<strong>token_type_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Segment token indices to indicate first and second portions of the inputs. Indices are selected in <code>[0,1]</code>:</p>
<ul>
<li>0 corresponds to a <em>sentence A</em> token,</li>
<li>1 corresponds to a <em>sentence B</em> token.
This parameter can only be used when the model is initialized with <code>type_vocab_size</code> parameter with value<blockquote>
<p>= 2. All the value in this tensor should be always &lt; type_vocab_size.</p>
</blockquote></li>
</ul>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"token_type_ids"},{anchor:"transformers.RobertaForQuestionAnswering.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.RobertaForQuestionAnswering.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.RobertaForQuestionAnswering.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.RobertaForQuestionAnswering.forward.start_positions",description:`<strong>start_positions</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for position (index) of the start of the labelled span for computing the token classification loss.
Positions are clamped to the length of the sequence (<code>sequence_length</code>). Position outside of the sequence
are not taken into account for computing the loss.`,name:"start_positions"},{anchor:"transformers.RobertaForQuestionAnswering.forward.end_positions",description:`<strong>end_positions</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for position (index) of the end of the labelled span for computing the token classification loss.
Positions are clamped to the length of the sequence (<code>sequence_length</code>). Position outside of the sequence
are not taken into account for computing the loss.`,name:"end_positions"},{anchor:"transformers.RobertaForQuestionAnswering.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.RobertaForQuestionAnswering.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.RobertaForQuestionAnswering.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/roberta/modeling_roberta.py#L1459",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.QuestionAnsweringModelOutput"
>transformers.modeling_outputs.QuestionAnsweringModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/roberta#transformers.RobertaConfig"
>RobertaConfig</a>) and inputs.</p>
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
`}}),We=new pe({props:{$$slots:{default:[Ha]},$$scope:{ctx:w}}}),Be=new me({props:{anchor:"transformers.RobertaForQuestionAnswering.forward.example",$$slots:{default:[Xa]},$$scope:{ctx:w}}}),wt=new ya({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/roberta.md"}}),{c(){t=c("meta"),m=a(),n=c("p"),i=a(),M=c("p"),M.innerHTML=o,u=a(),z=c("div"),z.innerHTML=_o,Ge=a(),f(le.$$.fragment),Mo=a(),Le=c("p"),Le.innerHTML=ks,wo=a(),Ve=c("p"),Ve.innerHTML=Ts,vo=a(),f(fe.$$.fragment),$o=a(),Ne=c("p"),Ne.innerHTML=Ms,Ro=a(),f(ge.$$.fragment),Jo=a(),f(He.$$.fragment),jo=a(),Xe=c("ul"),Xe.innerHTML=ws,Co=a(),f(Se.$$.fragment),zo=a(),N=c("div"),f(Ee.$$.fragment),Do=a(),Rt=c("p"),Rt.innerHTML=vs,Ko=a(),Jt=c("p"),Jt.innerHTML=$s,en=a(),f(be.$$.fragment),Fo=a(),f(Ae.$$.fragment),Uo=a(),v=c("div"),f(Ye.$$.fragment),tn=a(),jt=c("p"),jt.textContent=Rs,on=a(),Ct=c("p"),Ct.textContent=Js,nn=a(),f(_e.$$.fragment),sn=a(),zt=c("p"),zt.innerHTML=js,an=a(),f(ye.$$.fragment),rn=a(),Ft=c("p"),Ft.innerHTML=Cs,ln=a(),de=c("div"),f(Qe.$$.fragment),dn=a(),Ut=c("p"),Ut.textContent=zs,cn=a(),xt=c("ul"),xt.innerHTML=Fs,pn=a(),ke=c("div"),f(Pe.$$.fragment),mn=a(),It=c("p"),It.innerHTML=Us,hn=a(),Te=c("div"),f(Oe.$$.fragment),un=a(),qt=c("p"),qt.textContent=xs,fn=a(),Wt=c("div"),f(De.$$.fragment),xo=a(),f(Ke.$$.fragment),Io=a(),F=c("div"),f(et.$$.fragment),gn=a(),Bt=c("p"),Bt.innerHTML=Is,bn=a(),Zt=c("p"),Zt.textContent=qs,_n=a(),f(Me.$$.fragment),yn=a(),Gt=c("p"),Gt.innerHTML=Ws,kn=a(),f(we.$$.fragment),Tn=a(),Lt=c("p"),Lt.innerHTML=Bs,Mn=a(),Vt=c("div"),f(tt.$$.fragment),qo=a(),f(ot.$$.fragment),Wo=a(),x=c("div"),f(nt.$$.fragment),wn=a(),Nt=c("p"),Nt.innerHTML=Zs,vn=a(),Ht=c("p"),Ht.innerHTML=Gs,$n=a(),Xt=c("p"),Xt.innerHTML=Ls,Rn=a(),St=c("p"),St.innerHTML=Vs,Jn=a(),ce=c("div"),f(st.$$.fragment),jn=a(),Et=c("p"),Et.innerHTML=Ns,Cn=a(),f(ve.$$.fragment),Bo=a(),f(at.$$.fragment),Zo=a(),I=c("div"),f(rt.$$.fragment),zn=a(),At=c("p"),At.innerHTML=Hs,Fn=a(),Yt=c("p"),Yt.innerHTML=Xs,Un=a(),Qt=c("p"),Qt.innerHTML=Ss,xn=a(),O=c("div"),f(it.$$.fragment),In=a(),Pt=c("p"),Pt.innerHTML=Es,qn=a(),f($e.$$.fragment),Wn=a(),f(Re.$$.fragment),Go=a(),f(lt.$$.fragment),Lo=a(),q=c("div"),f(dt.$$.fragment),Bn=a(),Ot=c("p"),Ot.innerHTML=As,Zn=a(),Dt=c("p"),Dt.innerHTML=Ys,Gn=a(),Kt=c("p"),Kt.innerHTML=Qs,Ln=a(),D=c("div"),f(ct.$$.fragment),Vn=a(),eo=c("p"),eo.innerHTML=Ps,Nn=a(),f(Je.$$.fragment),Hn=a(),f(je.$$.fragment),Vo=a(),f(pt.$$.fragment),No=a(),W=c("div"),f(mt.$$.fragment),Xn=a(),to=c("p"),to.textContent=Os,Sn=a(),oo=c("p"),oo.innerHTML=Ds,En=a(),no=c("p"),no.innerHTML=Ks,An=a(),V=c("div"),f(ht.$$.fragment),Yn=a(),so=c("p"),so.innerHTML=ea,Qn=a(),f(Ce.$$.fragment),Pn=a(),f(ze.$$.fragment),On=a(),f(Fe.$$.fragment),Ho=a(),f(ut.$$.fragment),Xo=a(),B=c("div"),f(ft.$$.fragment),Dn=a(),ao=c("p"),ao.textContent=ta,Kn=a(),ro=c("p"),ro.innerHTML=oa,es=a(),io=c("p"),io.innerHTML=na,ts=a(),K=c("div"),f(gt.$$.fragment),os=a(),lo=c("p"),lo.innerHTML=sa,ns=a(),f(Ue.$$.fragment),ss=a(),f(xe.$$.fragment),So=a(),f(bt.$$.fragment),Eo=a(),Z=c("div"),f(_t.$$.fragment),as=a(),co=c("p"),co.textContent=aa,rs=a(),po=c("p"),po.innerHTML=ra,is=a(),mo=c("p"),mo.innerHTML=ia,ls=a(),ee=c("div"),f(yt.$$.fragment),ds=a(),ho=c("p"),ho.innerHTML=la,cs=a(),f(Ie.$$.fragment),ps=a(),f(qe.$$.fragment),Ao=a(),f(kt.$$.fragment),Yo=a(),G=c("div"),f(Tt.$$.fragment),ms=a(),uo=c("p"),uo.innerHTML=da,hs=a(),fo=c("p"),fo.innerHTML=ca,us=a(),go=c("p"),go.innerHTML=pa,fs=a(),te=c("div"),f(Mt.$$.fragment),gs=a(),bo=c("p"),bo.innerHTML=ma,bs=a(),f(We.$$.fragment),_s=a(),f(Be.$$.fragment),Qo=a(),f(wt.$$.fragment),Po=a(),yo=c("p"),this.h()},l(e){const l=ba("svelte-u9bgzb",document.head);t=p(l,"META",{name:!0,content:!0}),l.forEach(d),m=r(e),n=p(e,"P",{}),R(n).forEach(d),i=r(e),M=p(e,"P",{"data-svelte-h":!0}),T(M)!=="svelte-jr9vna"&&(M.innerHTML=o),u=r(e),z=p(e,"DIV",{style:!0,"data-svelte-h":!0}),T(z)!=="svelte-ithiq1"&&(z.innerHTML=_o),Ge=r(e),g(le.$$.fragment,e),Mo=r(e),Le=p(e,"P",{"data-svelte-h":!0}),T(Le)!=="svelte-7w6jto"&&(Le.innerHTML=ks),wo=r(e),Ve=p(e,"P",{"data-svelte-h":!0}),T(Ve)!=="svelte-127vl7w"&&(Ve.innerHTML=Ts),vo=r(e),g(fe.$$.fragment,e),$o=r(e),Ne=p(e,"P",{"data-svelte-h":!0}),T(Ne)!=="svelte-10lshn2"&&(Ne.innerHTML=Ms),Ro=r(e),g(ge.$$.fragment,e),Jo=r(e),g(He.$$.fragment,e),jo=r(e),Xe=p(e,"UL",{"data-svelte-h":!0}),T(Xe)!=="svelte-170i0wl"&&(Xe.innerHTML=ws),Co=r(e),g(Se.$$.fragment,e),zo=r(e),N=p(e,"DIV",{class:!0});var ne=R(N);g(Ee.$$.fragment,ne),Do=r(ne),Rt=p(ne,"P",{"data-svelte-h":!0}),T(Rt)!=="svelte-i2u28l"&&(Rt.innerHTML=vs),Ko=r(ne),Jt=p(ne,"P",{"data-svelte-h":!0}),T(Jt)!=="svelte-1ek1ss9"&&(Jt.innerHTML=$s),en=r(ne),g(be.$$.fragment,ne),ne.forEach(d),Fo=r(e),g(Ae.$$.fragment,e),Uo=r(e),v=p(e,"DIV",{class:!0});var $=R(v);g(Ye.$$.fragment,$),tn=r($),jt=p($,"P",{"data-svelte-h":!0}),T(jt)!=="svelte-1ax9a0k"&&(jt.textContent=Rs),on=r($),Ct=p($,"P",{"data-svelte-h":!0}),T(Ct)!=="svelte-1s077p3"&&(Ct.textContent=Js),nn=r($),g(_e.$$.fragment,$),sn=r($),zt=p($,"P",{"data-svelte-h":!0}),T(zt)!=="svelte-1jfcabo"&&(zt.innerHTML=js),an=r($),g(ye.$$.fragment,$),rn=r($),Ft=p($,"P",{"data-svelte-h":!0}),T(Ft)!=="svelte-ntrhio"&&(Ft.innerHTML=Cs),ln=r($),de=p($,"DIV",{class:!0});var he=R(de);g(Qe.$$.fragment,he),dn=r(he),Ut=p(he,"P",{"data-svelte-h":!0}),T(Ut)!=="svelte-og4clw"&&(Ut.textContent=zs),cn=r(he),xt=p(he,"UL",{"data-svelte-h":!0}),T(xt)!=="svelte-rq8uot"&&(xt.innerHTML=Fs),he.forEach(d),pn=r($),ke=p($,"DIV",{class:!0});var vt=R(ke);g(Pe.$$.fragment,vt),mn=r(vt),It=p(vt,"P",{"data-svelte-h":!0}),T(It)!=="svelte-1f4f5kp"&&(It.innerHTML=Us),vt.forEach(d),hn=r($),Te=p($,"DIV",{class:!0});var $t=R(Te);g(Oe.$$.fragment,$t),un=r($t),qt=p($t,"P",{"data-svelte-h":!0}),T(qt)!=="svelte-wwxeoo"&&(qt.textContent=xs),$t.forEach(d),fn=r($),Wt=p($,"DIV",{class:!0});var ko=R(Wt);g(De.$$.fragment,ko),ko.forEach(d),$.forEach(d),xo=r(e),g(Ke.$$.fragment,e),Io=r(e),F=p(e,"DIV",{class:!0});var U=R(F);g(et.$$.fragment,U),gn=r(U),Bt=p(U,"P",{"data-svelte-h":!0}),T(Bt)!=="svelte-171du1m"&&(Bt.innerHTML=Is),bn=r(U),Zt=p(U,"P",{"data-svelte-h":!0}),T(Zt)!=="svelte-1s077p3"&&(Zt.textContent=qs),_n=r(U),g(Me.$$.fragment,U),yn=r(U),Gt=p(U,"P",{"data-svelte-h":!0}),T(Gt)!=="svelte-1jfcabo"&&(Gt.innerHTML=Ws),kn=r(U),g(we.$$.fragment,U),Tn=r(U),Lt=p(U,"P",{"data-svelte-h":!0}),T(Lt)!=="svelte-gxzj9w"&&(Lt.innerHTML=Bs),Mn=r(U),Vt=p(U,"DIV",{class:!0});var To=R(Vt);g(tt.$$.fragment,To),To.forEach(d),U.forEach(d),qo=r(e),g(ot.$$.fragment,e),Wo=r(e),x=p(e,"DIV",{class:!0});var L=R(x);g(nt.$$.fragment,L),wn=r(L),Nt=p(L,"P",{"data-svelte-h":!0}),T(Nt)!=="svelte-1854dma"&&(Nt.innerHTML=Zs),vn=r(L),Ht=p(L,"P",{"data-svelte-h":!0}),T(Ht)!=="svelte-174erte"&&(Ht.innerHTML=Gs),$n=r(L),Xt=p(L,"P",{"data-svelte-h":!0}),T(Xt)!=="svelte-q52n56"&&(Xt.innerHTML=Ls),Rn=r(L),St=p(L,"P",{"data-svelte-h":!0}),T(St)!=="svelte-hswkmf"&&(St.innerHTML=Vs),Jn=r(L),ce=p(L,"DIV",{class:!0});var ue=R(ce);g(st.$$.fragment,ue),jn=r(ue),Et=p(ue,"P",{"data-svelte-h":!0}),T(Et)!=="svelte-sc29nu"&&(Et.innerHTML=Ns),Cn=r(ue),g(ve.$$.fragment,ue),ue.forEach(d),L.forEach(d),Bo=r(e),g(at.$$.fragment,e),Zo=r(e),I=p(e,"DIV",{class:!0});var H=R(I);g(rt.$$.fragment,H),zn=r(H),At=p(H,"P",{"data-svelte-h":!0}),T(At)!=="svelte-aerwia"&&(At.innerHTML=Hs),Fn=r(H),Yt=p(H,"P",{"data-svelte-h":!0}),T(Yt)!=="svelte-q52n56"&&(Yt.innerHTML=Xs),Un=r(H),Qt=p(H,"P",{"data-svelte-h":!0}),T(Qt)!=="svelte-hswkmf"&&(Qt.innerHTML=Ss),xn=r(H),O=p(H,"DIV",{class:!0});var se=R(O);g(it.$$.fragment,se),In=r(se),Pt=p(se,"P",{"data-svelte-h":!0}),T(Pt)!=="svelte-auvoya"&&(Pt.innerHTML=Es),qn=r(se),g($e.$$.fragment,se),Wn=r(se),g(Re.$$.fragment,se),se.forEach(d),H.forEach(d),Go=r(e),g(lt.$$.fragment,e),Lo=r(e),q=p(e,"DIV",{class:!0});var X=R(q);g(dt.$$.fragment,X),Bn=r(X),Ot=p(X,"P",{"data-svelte-h":!0}),T(Ot)!=="svelte-1kspo1x"&&(Ot.innerHTML=As),Zn=r(X),Dt=p(X,"P",{"data-svelte-h":!0}),T(Dt)!=="svelte-q52n56"&&(Dt.innerHTML=Ys),Gn=r(X),Kt=p(X,"P",{"data-svelte-h":!0}),T(Kt)!=="svelte-hswkmf"&&(Kt.innerHTML=Qs),Ln=r(X),D=p(X,"DIV",{class:!0});var ae=R(D);g(ct.$$.fragment,ae),Vn=r(ae),eo=p(ae,"P",{"data-svelte-h":!0}),T(eo)!=="svelte-1nnddpm"&&(eo.innerHTML=Ps),Nn=r(ae),g(Je.$$.fragment,ae),Hn=r(ae),g(je.$$.fragment,ae),ae.forEach(d),X.forEach(d),Vo=r(e),g(pt.$$.fragment,e),No=r(e),W=p(e,"DIV",{class:!0});var S=R(W);g(mt.$$.fragment,S),Xn=r(S),to=p(S,"P",{"data-svelte-h":!0}),T(to)!=="svelte-1xmlxux"&&(to.textContent=Os),Sn=r(S),oo=p(S,"P",{"data-svelte-h":!0}),T(oo)!=="svelte-q52n56"&&(oo.innerHTML=Ds),En=r(S),no=p(S,"P",{"data-svelte-h":!0}),T(no)!=="svelte-hswkmf"&&(no.innerHTML=Ks),An=r(S),V=p(S,"DIV",{class:!0});var E=R(V);g(ht.$$.fragment,E),Yn=r(E),so=p(E,"P",{"data-svelte-h":!0}),T(so)!=="svelte-j9dh3o"&&(so.innerHTML=ea),Qn=r(E),g(Ce.$$.fragment,E),Pn=r(E),g(ze.$$.fragment,E),On=r(E),g(Fe.$$.fragment,E),E.forEach(d),S.forEach(d),Ho=r(e),g(ut.$$.fragment,e),Xo=r(e),B=p(e,"DIV",{class:!0});var A=R(B);g(ft.$$.fragment,A),Dn=r(A),ao=p(A,"P",{"data-svelte-h":!0}),T(ao)!=="svelte-m2bybq"&&(ao.textContent=ta),Kn=r(A),ro=p(A,"P",{"data-svelte-h":!0}),T(ro)!=="svelte-q52n56"&&(ro.innerHTML=oa),es=r(A),io=p(A,"P",{"data-svelte-h":!0}),T(io)!=="svelte-hswkmf"&&(io.innerHTML=na),ts=r(A),K=p(A,"DIV",{class:!0});var re=R(K);g(gt.$$.fragment,re),os=r(re),lo=p(re,"P",{"data-svelte-h":!0}),T(lo)!=="svelte-5imbm4"&&(lo.innerHTML=sa),ns=r(re),g(Ue.$$.fragment,re),ss=r(re),g(xe.$$.fragment,re),re.forEach(d),A.forEach(d),So=r(e),g(bt.$$.fragment,e),Eo=r(e),Z=p(e,"DIV",{class:!0});var Y=R(Z);g(_t.$$.fragment,Y),as=r(Y),co=p(Y,"P",{"data-svelte-h":!0}),T(co)!=="svelte-1dpvp2x"&&(co.textContent=aa),rs=r(Y),po=p(Y,"P",{"data-svelte-h":!0}),T(po)!=="svelte-q52n56"&&(po.innerHTML=ra),is=r(Y),mo=p(Y,"P",{"data-svelte-h":!0}),T(mo)!=="svelte-hswkmf"&&(mo.innerHTML=ia),ls=r(Y),ee=p(Y,"DIV",{class:!0});var ie=R(ee);g(yt.$$.fragment,ie),ds=r(ie),ho=p(ie,"P",{"data-svelte-h":!0}),T(ho)!=="svelte-le1j2m"&&(ho.innerHTML=la),cs=r(ie),g(Ie.$$.fragment,ie),ps=r(ie),g(qe.$$.fragment,ie),ie.forEach(d),Y.forEach(d),Ao=r(e),g(kt.$$.fragment,e),Yo=r(e),G=p(e,"DIV",{class:!0});var Q=R(G);g(Tt.$$.fragment,Q),ms=r(Q),uo=p(Q,"P",{"data-svelte-h":!0}),T(uo)!=="svelte-avi9b6"&&(uo.innerHTML=da),hs=r(Q),fo=p(Q,"P",{"data-svelte-h":!0}),T(fo)!=="svelte-q52n56"&&(fo.innerHTML=ca),us=r(Q),go=p(Q,"P",{"data-svelte-h":!0}),T(go)!=="svelte-hswkmf"&&(go.innerHTML=pa),fs=r(Q),te=p(Q,"DIV",{class:!0});var Ze=R(te);g(Mt.$$.fragment,Ze),gs=r(Ze),bo=p(Ze,"P",{"data-svelte-h":!0}),T(bo)!=="svelte-1iqg67e"&&(bo.innerHTML=ma),bs=r(Ze),g(We.$$.fragment,Ze),_s=r(Ze),g(Be.$$.fragment,Ze),Ze.forEach(d),Q.forEach(d),Qo=r(e),g(wt.$$.fragment,e),Po=r(e),yo=p(e,"P",{}),R(yo).forEach(d),this.h()},h(){J(t,"name","hf:doc:metadata"),J(t,"content",Ea),_a(z,"float","right"),J(N,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(de,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(ke,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(Te,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(Wt,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(v,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(Vt,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(F,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(ce,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(x,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(O,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(I,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(D,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(q,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(V,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(W,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(K,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(B,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(ee,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(Z,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(te,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(G,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,l){s(document.head,t),h(e,m,l),h(e,n,l),h(e,i,l),h(e,M,l),h(e,u,l),h(e,z,l),h(e,Ge,l),b(le,e,l),h(e,Mo,l),h(e,Le,l),h(e,wo,l),h(e,Ve,l),h(e,vo,l),b(fe,e,l),h(e,$o,l),h(e,Ne,l),h(e,Ro,l),b(ge,e,l),h(e,Jo,l),b(He,e,l),h(e,jo,l),h(e,Xe,l),h(e,Co,l),b(Se,e,l),h(e,zo,l),h(e,N,l),b(Ee,N,null),s(N,Do),s(N,Rt),s(N,Ko),s(N,Jt),s(N,en),b(be,N,null),h(e,Fo,l),b(Ae,e,l),h(e,Uo,l),h(e,v,l),b(Ye,v,null),s(v,tn),s(v,jt),s(v,on),s(v,Ct),s(v,nn),b(_e,v,null),s(v,sn),s(v,zt),s(v,an),b(ye,v,null),s(v,rn),s(v,Ft),s(v,ln),s(v,de),b(Qe,de,null),s(de,dn),s(de,Ut),s(de,cn),s(de,xt),s(v,pn),s(v,ke),b(Pe,ke,null),s(ke,mn),s(ke,It),s(v,hn),s(v,Te),b(Oe,Te,null),s(Te,un),s(Te,qt),s(v,fn),s(v,Wt),b(De,Wt,null),h(e,xo,l),b(Ke,e,l),h(e,Io,l),h(e,F,l),b(et,F,null),s(F,gn),s(F,Bt),s(F,bn),s(F,Zt),s(F,_n),b(Me,F,null),s(F,yn),s(F,Gt),s(F,kn),b(we,F,null),s(F,Tn),s(F,Lt),s(F,Mn),s(F,Vt),b(tt,Vt,null),h(e,qo,l),b(ot,e,l),h(e,Wo,l),h(e,x,l),b(nt,x,null),s(x,wn),s(x,Nt),s(x,vn),s(x,Ht),s(x,$n),s(x,Xt),s(x,Rn),s(x,St),s(x,Jn),s(x,ce),b(st,ce,null),s(ce,jn),s(ce,Et),s(ce,Cn),b(ve,ce,null),h(e,Bo,l),b(at,e,l),h(e,Zo,l),h(e,I,l),b(rt,I,null),s(I,zn),s(I,At),s(I,Fn),s(I,Yt),s(I,Un),s(I,Qt),s(I,xn),s(I,O),b(it,O,null),s(O,In),s(O,Pt),s(O,qn),b($e,O,null),s(O,Wn),b(Re,O,null),h(e,Go,l),b(lt,e,l),h(e,Lo,l),h(e,q,l),b(dt,q,null),s(q,Bn),s(q,Ot),s(q,Zn),s(q,Dt),s(q,Gn),s(q,Kt),s(q,Ln),s(q,D),b(ct,D,null),s(D,Vn),s(D,eo),s(D,Nn),b(Je,D,null),s(D,Hn),b(je,D,null),h(e,Vo,l),b(pt,e,l),h(e,No,l),h(e,W,l),b(mt,W,null),s(W,Xn),s(W,to),s(W,Sn),s(W,oo),s(W,En),s(W,no),s(W,An),s(W,V),b(ht,V,null),s(V,Yn),s(V,so),s(V,Qn),b(Ce,V,null),s(V,Pn),b(ze,V,null),s(V,On),b(Fe,V,null),h(e,Ho,l),b(ut,e,l),h(e,Xo,l),h(e,B,l),b(ft,B,null),s(B,Dn),s(B,ao),s(B,Kn),s(B,ro),s(B,es),s(B,io),s(B,ts),s(B,K),b(gt,K,null),s(K,os),s(K,lo),s(K,ns),b(Ue,K,null),s(K,ss),b(xe,K,null),h(e,So,l),b(bt,e,l),h(e,Eo,l),h(e,Z,l),b(_t,Z,null),s(Z,as),s(Z,co),s(Z,rs),s(Z,po),s(Z,is),s(Z,mo),s(Z,ls),s(Z,ee),b(yt,ee,null),s(ee,ds),s(ee,ho),s(ee,cs),b(Ie,ee,null),s(ee,ps),b(qe,ee,null),h(e,Ao,l),b(kt,e,l),h(e,Yo,l),h(e,G,l),b(Tt,G,null),s(G,ms),s(G,uo),s(G,hs),s(G,fo),s(G,us),s(G,go),s(G,fs),s(G,te),b(Mt,te,null),s(te,gs),s(te,bo),s(te,bs),b(We,te,null),s(te,_s),b(Be,te,null),h(e,Qo,l),b(wt,e,l),h(e,Po,l),h(e,yo,l),Oo=!0},p(e,[l]){const ne={};l&2&&(ne.$$scope={dirty:l,ctx:e}),fe.$set(ne);const $={};l&2&&($.$$scope={dirty:l,ctx:e}),ge.$set($);const he={};l&2&&(he.$$scope={dirty:l,ctx:e}),be.$set(he);const vt={};l&2&&(vt.$$scope={dirty:l,ctx:e}),_e.$set(vt);const $t={};l&2&&($t.$$scope={dirty:l,ctx:e}),ye.$set($t);const ko={};l&2&&(ko.$$scope={dirty:l,ctx:e}),Me.$set(ko);const U={};l&2&&(U.$$scope={dirty:l,ctx:e}),we.$set(U);const To={};l&2&&(To.$$scope={dirty:l,ctx:e}),ve.$set(To);const L={};l&2&&(L.$$scope={dirty:l,ctx:e}),$e.$set(L);const ue={};l&2&&(ue.$$scope={dirty:l,ctx:e}),Re.$set(ue);const H={};l&2&&(H.$$scope={dirty:l,ctx:e}),Je.$set(H);const se={};l&2&&(se.$$scope={dirty:l,ctx:e}),je.$set(se);const X={};l&2&&(X.$$scope={dirty:l,ctx:e}),Ce.$set(X);const ae={};l&2&&(ae.$$scope={dirty:l,ctx:e}),ze.$set(ae);const S={};l&2&&(S.$$scope={dirty:l,ctx:e}),Fe.$set(S);const E={};l&2&&(E.$$scope={dirty:l,ctx:e}),Ue.$set(E);const A={};l&2&&(A.$$scope={dirty:l,ctx:e}),xe.$set(A);const re={};l&2&&(re.$$scope={dirty:l,ctx:e}),Ie.$set(re);const Y={};l&2&&(Y.$$scope={dirty:l,ctx:e}),qe.$set(Y);const ie={};l&2&&(ie.$$scope={dirty:l,ctx:e}),We.$set(ie);const Q={};l&2&&(Q.$$scope={dirty:l,ctx:e}),Be.$set(Q)},i(e){Oo||(_(le.$$.fragment,e),_(fe.$$.fragment,e),_(ge.$$.fragment,e),_(He.$$.fragment,e),_(Se.$$.fragment,e),_(Ee.$$.fragment,e),_(be.$$.fragment,e),_(Ae.$$.fragment,e),_(Ye.$$.fragment,e),_(_e.$$.fragment,e),_(ye.$$.fragment,e),_(Qe.$$.fragment,e),_(Pe.$$.fragment,e),_(Oe.$$.fragment,e),_(De.$$.fragment,e),_(Ke.$$.fragment,e),_(et.$$.fragment,e),_(Me.$$.fragment,e),_(we.$$.fragment,e),_(tt.$$.fragment,e),_(ot.$$.fragment,e),_(nt.$$.fragment,e),_(st.$$.fragment,e),_(ve.$$.fragment,e),_(at.$$.fragment,e),_(rt.$$.fragment,e),_(it.$$.fragment,e),_($e.$$.fragment,e),_(Re.$$.fragment,e),_(lt.$$.fragment,e),_(dt.$$.fragment,e),_(ct.$$.fragment,e),_(Je.$$.fragment,e),_(je.$$.fragment,e),_(pt.$$.fragment,e),_(mt.$$.fragment,e),_(ht.$$.fragment,e),_(Ce.$$.fragment,e),_(ze.$$.fragment,e),_(Fe.$$.fragment,e),_(ut.$$.fragment,e),_(ft.$$.fragment,e),_(gt.$$.fragment,e),_(Ue.$$.fragment,e),_(xe.$$.fragment,e),_(bt.$$.fragment,e),_(_t.$$.fragment,e),_(yt.$$.fragment,e),_(Ie.$$.fragment,e),_(qe.$$.fragment,e),_(kt.$$.fragment,e),_(Tt.$$.fragment,e),_(Mt.$$.fragment,e),_(We.$$.fragment,e),_(Be.$$.fragment,e),_(wt.$$.fragment,e),Oo=!0)},o(e){y(le.$$.fragment,e),y(fe.$$.fragment,e),y(ge.$$.fragment,e),y(He.$$.fragment,e),y(Se.$$.fragment,e),y(Ee.$$.fragment,e),y(be.$$.fragment,e),y(Ae.$$.fragment,e),y(Ye.$$.fragment,e),y(_e.$$.fragment,e),y(ye.$$.fragment,e),y(Qe.$$.fragment,e),y(Pe.$$.fragment,e),y(Oe.$$.fragment,e),y(De.$$.fragment,e),y(Ke.$$.fragment,e),y(et.$$.fragment,e),y(Me.$$.fragment,e),y(we.$$.fragment,e),y(tt.$$.fragment,e),y(ot.$$.fragment,e),y(nt.$$.fragment,e),y(st.$$.fragment,e),y(ve.$$.fragment,e),y(at.$$.fragment,e),y(rt.$$.fragment,e),y(it.$$.fragment,e),y($e.$$.fragment,e),y(Re.$$.fragment,e),y(lt.$$.fragment,e),y(dt.$$.fragment,e),y(ct.$$.fragment,e),y(Je.$$.fragment,e),y(je.$$.fragment,e),y(pt.$$.fragment,e),y(mt.$$.fragment,e),y(ht.$$.fragment,e),y(Ce.$$.fragment,e),y(ze.$$.fragment,e),y(Fe.$$.fragment,e),y(ut.$$.fragment,e),y(ft.$$.fragment,e),y(gt.$$.fragment,e),y(Ue.$$.fragment,e),y(xe.$$.fragment,e),y(bt.$$.fragment,e),y(_t.$$.fragment,e),y(yt.$$.fragment,e),y(Ie.$$.fragment,e),y(qe.$$.fragment,e),y(kt.$$.fragment,e),y(Tt.$$.fragment,e),y(Mt.$$.fragment,e),y(We.$$.fragment,e),y(Be.$$.fragment,e),y(wt.$$.fragment,e),Oo=!1},d(e){e&&(d(m),d(n),d(i),d(M),d(u),d(z),d(Ge),d(Mo),d(Le),d(wo),d(Ve),d(vo),d($o),d(Ne),d(Ro),d(Jo),d(jo),d(Xe),d(Co),d(zo),d(N),d(Fo),d(Uo),d(v),d(xo),d(Io),d(F),d(qo),d(Wo),d(x),d(Bo),d(Zo),d(I),d(Go),d(Lo),d(q),d(Vo),d(No),d(W),d(Ho),d(Xo),d(B),d(So),d(Eo),d(Z),d(Ao),d(Yo),d(G),d(Qo),d(Po),d(yo)),d(t),k(le,e),k(fe,e),k(ge,e),k(He,e),k(Se,e),k(Ee),k(be),k(Ae,e),k(Ye),k(_e),k(ye),k(Qe),k(Pe),k(Oe),k(De),k(Ke,e),k(et),k(Me),k(we),k(tt),k(ot,e),k(nt),k(st),k(ve),k(at,e),k(rt),k(it),k($e),k(Re),k(lt,e),k(dt),k(ct),k(Je),k(je),k(pt,e),k(mt),k(ht),k(Ce),k(ze),k(Fe),k(ut,e),k(ft),k(gt),k(Ue),k(xe),k(bt,e),k(_t),k(yt),k(Ie),k(qe),k(kt,e),k(Tt),k(Mt),k(We),k(Be),k(wt,e)}}}const Ea='{"title":"RoBERTa","local":"roberta","sections":[{"title":"Notes","local":"notes","sections":[],"depth":2},{"title":"RobertaConfig","local":"transformers.RobertaConfig","sections":[],"depth":2},{"title":"RobertaTokenizer","local":"transformers.RobertaTokenizer","sections":[],"depth":2},{"title":"RobertaTokenizerFast","local":"transformers.RobertaTokenizerFast","sections":[],"depth":2},{"title":"RobertaModel","local":"transformers.RobertaModel","sections":[],"depth":2},{"title":"RobertaForCausalLM","local":"transformers.RobertaForCausalLM","sections":[],"depth":2},{"title":"RobertaForMaskedLM","local":"transformers.RobertaForMaskedLM","sections":[],"depth":2},{"title":"RobertaForSequenceClassification","local":"transformers.RobertaForSequenceClassification","sections":[],"depth":2},{"title":"RobertaForMultipleChoice","local":"transformers.RobertaForMultipleChoice","sections":[],"depth":2},{"title":"RobertaForTokenClassification","local":"transformers.RobertaForTokenClassification","sections":[],"depth":2},{"title":"RobertaForQuestionAnswering","local":"transformers.RobertaForQuestionAnswering","sections":[],"depth":2}],"depth":1}';function Aa(w){return ua(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class or extends fa{constructor(t){super(),ga(this,t,Aa,Sa,ha,{})}}export{or as component};
