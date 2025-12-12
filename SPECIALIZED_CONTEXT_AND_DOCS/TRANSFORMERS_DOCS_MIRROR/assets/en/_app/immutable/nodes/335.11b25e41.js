import{s as gn,o as fn,n as V}from"../chunks/scheduler.18a86fab.js";import{S as _n,i as bn,g as c,s as a,r as g,A as yn,h as p,f as s,c as r,j as z,x as M,u as f,k as j,l as kn,y as i,a as l,v as _,d as b,t as y,w as k}from"../chunks/index.98837b22.js";import{T as Rt}from"../chunks/Tip.77304350.js";import{D as I}from"../chunks/Docstring.a1ef7999.js";import{C as pe}from"../chunks/CodeBlock.8d0c2e8a.js";import{E as At}from"../chunks/ExampleCodeBlock.8c3ee1f9.js";import{H as ce,E as Mn}from"../chunks/getInferenceSnippets.06c2775f.js";import{H as Tn,a as Fo}from"../chunks/HfOption.6641485e.js";function vn(v){let t,h="Click on the Pegasus models in the right sidebar for more examples of how to apply Pegasus to different language tasks.";return{c(){t=c("p"),t.textContent=h},l(o){t=p(o,"P",{"data-svelte-h":!0}),M(t)!=="svelte-esqijt"&&(t.textContent=h)},m(o,u){l(o,t,u)},p:V,d(o){o&&s(t)}}}function wn(v){let t,h;return t=new pe({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwcGlwZWxpbmUlMEElMEFwaXBlbGluZSUyMCUzRCUyMHBpcGVsaW5lKCUwQSUyMCUyMCUyMCUyMHRhc2slM0QlMjJzdW1tYXJpemF0aW9uJTIyJTJDJTBBJTIwJTIwJTIwJTIwbW9kZWwlM0QlMjJnb29nbGUlMkZwZWdhc3VzLXhzdW0lMjIlMkMlMEElMjAlMjAlMjAlMjBkdHlwZSUzRHRvcmNoLmZsb2F0MTYlMkMlMEElMjAlMjAlMjAlMjBkZXZpY2UlM0QwJTBBKSUwQXBpcGVsaW5lKCUyMiUyMiUyMlBsYW50cyUyMGFyZSUyMHJlbWFya2FibGUlMjBvcmdhbmlzbXMlMjB0aGF0JTIwcHJvZHVjZSUyMHRoZWlyJTIwb3duJTIwZm9vZCUyMHVzaW5nJTIwYSUyMG1ldGhvZCUyMGNhbGxlZCUyMHBob3Rvc3ludGhlc2lzLiUwQVRoaXMlMjBwcm9jZXNzJTIwaW52b2x2ZXMlMjBjb252ZXJ0aW5nJTIwc3VubGlnaHQlMkMlMjBjYXJib24lMjBkaW94aWRlJTJDJTIwYW5kJTIwd2F0ZXIlMjBpbnRvJTIwZ2x1Y29zZSUyQyUyMHdoaWNoJTIwcHJvdmlkZXMlMjBlbmVyZ3klMjBmb3IlMjBncm93dGguJTBBUGxhbnRzJTIwcGxheSUyMGElMjBjcnVjaWFsJTIwcm9sZSUyMGluJTIwc3VzdGFpbmluZyUyMGxpZmUlMjBvbiUyMEVhcnRoJTIwYnklMjBnZW5lcmF0aW5nJTIwb3h5Z2VuJTIwYW5kJTIwc2VydmluZyUyMGFzJTIwdGhlJTIwZm91bmRhdGlvbiUyMG9mJTIwbW9zdCUyMGVjb3N5c3RlbXMuJTIyJTIyJTIyKQ==",highlighted:`<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> pipeline

pipeline = pipeline(
    task=<span class="hljs-string">&quot;summarization&quot;</span>,
    model=<span class="hljs-string">&quot;google/pegasus-xsum&quot;</span>,
    dtype=torch.float16,
    device=<span class="hljs-number">0</span>
)
pipeline(<span class="hljs-string">&quot;&quot;&quot;Plants are remarkable organisms that produce their own food using a method called photosynthesis.
This process involves converting sunlight, carbon dioxide, and water into glucose, which provides energy for growth.
Plants play a crucial role in sustaining life on Earth by generating oxygen and serving as the foundation of most ecosystems.&quot;&quot;&quot;</span>)`,wrap:!1}}),{c(){g(t.$$.fragment)},l(o){f(t.$$.fragment,o)},m(o,u){_(t,o,u),h=!0},p:V,i(o){h||(b(t.$$.fragment,o),h=!0)},o(o){y(t.$$.fragment,o),h=!1},d(o){k(t,o)}}}function $n(v){let t,h;return t=new pe({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b01vZGVsRm9yU2VxMlNlcUxNJTJDJTIwQXV0b1Rva2VuaXplciUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUwQSUyMCUyMCUyMCUyMCUyMmdvb2dsZSUyRnBlZ2FzdXMteHN1bSUyMiUwQSklMEFtb2RlbCUyMCUzRCUyMEF1dG9Nb2RlbEZvclNlcTJTZXFMTS5mcm9tX3ByZXRyYWluZWQoJTBBJTIwJTIwJTIwJTIwJTIyZ29vZ2xlJTJGcGVnYXN1cy14c3VtJTIyJTJDJTBBJTIwJTIwJTIwJTIwZHR5cGUlM0R0b3JjaC5mbG9hdDE2JTJDJTBBJTIwJTIwJTIwJTIwZGV2aWNlX21hcCUzRCUyMmF1dG8lMjIlMkMlMEElMjAlMjAlMjAlMjBhdHRuX2ltcGxlbWVudGF0aW9uJTNEJTIyc2RwYSUyMiUwQSklMEElMEFpbnB1dF90ZXh0JTIwJTNEJTIwJTIyJTIyJTIyUGxhbnRzJTIwYXJlJTIwcmVtYXJrYWJsZSUyMG9yZ2FuaXNtcyUyMHRoYXQlMjBwcm9kdWNlJTIwdGhlaXIlMjBvd24lMjBmb29kJTIwdXNpbmclMjBhJTIwbWV0aG9kJTIwY2FsbGVkJTIwcGhvdG9zeW50aGVzaXMuJTBBVGhpcyUyMHByb2Nlc3MlMjBpbnZvbHZlcyUyMGNvbnZlcnRpbmclMjBzdW5saWdodCUyQyUyMGNhcmJvbiUyMGRpb3hpZGUlMkMlMjBhbmQlMjB3YXRlciUyMGludG8lMjBnbHVjb3NlJTJDJTIwd2hpY2glMjBwcm92aWRlcyUyMGVuZXJneSUyMGZvciUyMGdyb3d0aC4lMEFQbGFudHMlMjBwbGF5JTIwYSUyMGNydWNpYWwlMjByb2xlJTIwaW4lMjBzdXN0YWluaW5nJTIwbGlmZSUyMG9uJTIwRWFydGglMjBieSUyMGdlbmVyYXRpbmclMjBveHlnZW4lMjBhbmQlMjBzZXJ2aW5nJTIwYXMlMjB0aGUlMjBmb3VuZGF0aW9uJTIwb2YlMjBtb3N0JTIwZWNvc3lzdGVtcy4lMjIlMjIlMjIlMEFpbnB1dF9pZHMlMjAlM0QlMjB0b2tlbml6ZXIoaW5wdXRfdGV4dCUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpLnRvKG1vZGVsLmRldmljZSklMEElMEFvdXRwdXQlMjAlM0QlMjBtb2RlbC5nZW5lcmF0ZSgqKmlucHV0X2lkcyUyQyUyMGNhY2hlX2ltcGxlbWVudGF0aW9uJTNEJTIyc3RhdGljJTIyKSUwQXByaW50KHRva2VuaXplci5kZWNvZGUob3V0cHV0JTVCMCU1RCUyQyUyMHNraXBfc3BlY2lhbF90b2tlbnMlM0RUcnVlKSk=",highlighted:`<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoModelForSeq2SeqLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    <span class="hljs-string">&quot;google/pegasus-xsum&quot;</span>
)
model = AutoModelForSeq2SeqLM.from_pretrained(
    <span class="hljs-string">&quot;google/pegasus-xsum&quot;</span>,
    dtype=torch.float16,
    device_map=<span class="hljs-string">&quot;auto&quot;</span>,
    attn_implementation=<span class="hljs-string">&quot;sdpa&quot;</span>
)

input_text = <span class="hljs-string">&quot;&quot;&quot;Plants are remarkable organisms that produce their own food using a method called photosynthesis.
This process involves converting sunlight, carbon dioxide, and water into glucose, which provides energy for growth.
Plants play a crucial role in sustaining life on Earth by generating oxygen and serving as the foundation of most ecosystems.&quot;&quot;&quot;</span>
input_ids = tokenizer(input_text, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>).to(model.device)

output = model.generate(**input_ids, cache_implementation=<span class="hljs-string">&quot;static&quot;</span>)
<span class="hljs-built_in">print</span>(tokenizer.decode(output[<span class="hljs-number">0</span>], skip_special_tokens=<span class="hljs-literal">True</span>))`,wrap:!1}}),{c(){g(t.$$.fragment)},l(o){f(t.$$.fragment,o)},m(o,u){_(t,o,u),h=!0},p:V,i(o){h||(b(t.$$.fragment,o),h=!0)},o(o){y(t.$$.fragment,o),h=!1},d(o){k(t,o)}}}function Jn(v){let t,h;return t=new pe({props:{code:"ZWNobyUyMC1lJTIwJTIyUGxhbnRzJTIwYXJlJTIwcmVtYXJrYWJsZSUyMG9yZ2FuaXNtcyUyMHRoYXQlMjBwcm9kdWNlJTIwdGhlaXIlMjBvd24lMjBmb29kJTIwdXNpbmclMjBhJTIwbWV0aG9kJTIwY2FsbGVkJTIwcGhvdG9zeW50aGVzaXMuJTIwVGhpcyUyMHByb2Nlc3MlMjBpbnZvbHZlcyUyMGNvbnZlcnRpbmclMjBzdW5saWdodCUyQyUyMGNhcmJvbiUyMGRpb3hpZGUlMkMlMjBhbmQlMjB3YXRlciUyMGludG8lMjBnbHVjb3NlJTJDJTIwd2hpY2glMjBwcm92aWRlcyUyMGVuZXJneSUyMGZvciUyMGdyb3d0aC4lMjBQbGFudHMlMjBwbGF5JTIwYSUyMGNydWNpYWwlMjByb2xlJTIwaW4lMjBzdXN0YWluaW5nJTIwbGlmZSUyMG9uJTIwRWFydGglMjBieSUyMGdlbmVyYXRpbmclMjBveHlnZW4lMjBhbmQlMjBzZXJ2aW5nJTIwYXMlMjB0aGUlMjBmb3VuZGF0aW9uJTIwb2YlMjBtb3N0JTIwZWNvc3lzdGVtcy4lMjIlMjAlN0MlMjB0cmFuc2Zvcm1lcnMtY2xpJTIwcnVuJTIwLS10YXNrJTIwc3VtbWFyaXphdGlvbiUyMC0tbW9kZWwlMjBnb29nbGUlMkZwZWdhc3VzLXhzdW0lMjAtLWRldmljZSUyMDA=",highlighted:'<span class="hljs-built_in">echo</span> -e <span class="hljs-string">&quot;Plants are remarkable organisms that produce their own food using a method called photosynthesis. This process involves converting sunlight, carbon dioxide, and water into glucose, which provides energy for growth. Plants play a crucial role in sustaining life on Earth by generating oxygen and serving as the foundation of most ecosystems.&quot;</span> | transformers-cli run --task summarization --model google/pegasus-xsum --device 0',wrap:!1}}),{c(){g(t.$$.fragment)},l(o){f(t.$$.fragment,o)},m(o,u){_(t,o,u),h=!0},p:V,i(o){h||(b(t.$$.fragment,o),h=!0)},o(o){y(t.$$.fragment,o),h=!1},d(o){k(t,o)}}}function zn(v){let t,h,o,u,T,d;return t=new Fo({props:{id:"usage",option:"Pipeline",$$slots:{default:[wn]},$$scope:{ctx:v}}}),o=new Fo({props:{id:"usage",option:"AutoModel",$$slots:{default:[$n]},$$scope:{ctx:v}}}),T=new Fo({props:{id:"usage",option:"transformers CLI",$$slots:{default:[Jn]},$$scope:{ctx:v}}}),{c(){g(t.$$.fragment),h=a(),g(o.$$.fragment),u=a(),g(T.$$.fragment)},l(m){f(t.$$.fragment,m),h=r(m),f(o.$$.fragment,m),u=r(m),f(T.$$.fragment,m)},m(m,w){_(t,m,w),l(m,h,w),_(o,m,w),l(m,u,w),_(T,m,w),d=!0},p(m,w){const bt={};w&2&&(bt.$$scope={dirty:w,ctx:m}),t.$set(bt);const ue={};w&2&&(ue.$$scope={dirty:w,ctx:m}),o.$set(ue);const N={};w&2&&(N.$$scope={dirty:w,ctx:m}),T.$set(N)},i(m){d||(b(t.$$.fragment,m),b(o.$$.fragment,m),b(T.$$.fragment,m),d=!0)},o(m){y(t.$$.fragment,m),y(o.$$.fragment,m),y(T.$$.fragment,m),d=!1},d(m){m&&(s(h),s(u)),k(t,m),k(o,m),k(T,m)}}}function jn(v){let t,h="Example:",o,u,T;return u=new pe({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMFBlZ2FzdXNDb25maWclMkMlMjBQZWdhc3VzTW9kZWwlMEElMEElMjMlMjBJbml0aWFsaXppbmclMjBhJTIwUEVHQVNVUyUyMGdvb2dsZSUyRnBlZ2FzdXMtbGFyZ2UlMjBzdHlsZSUyMGNvbmZpZ3VyYXRpb24lMEFjb25maWd1cmF0aW9uJTIwJTNEJTIwUGVnYXN1c0NvbmZpZygpJTBBJTBBJTIzJTIwSW5pdGlhbGl6aW5nJTIwYSUyMG1vZGVsJTIwKHdpdGglMjByYW5kb20lMjB3ZWlnaHRzKSUyMGZyb20lMjB0aGUlMjBnb29nbGUlMkZwZWdhc3VzLWxhcmdlJTIwc3R5bGUlMjBjb25maWd1cmF0aW9uJTBBbW9kZWwlMjAlM0QlMjBQZWdhc3VzTW9kZWwoY29uZmlndXJhdGlvbiklMEElMEElMjMlMjBBY2Nlc3NpbmclMjB0aGUlMjBtb2RlbCUyMGNvbmZpZ3VyYXRpb24lMEFjb25maWd1cmF0aW9uJTIwJTNEJTIwbW9kZWwuY29uZmln",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> PegasusConfig, PegasusModel

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a PEGASUS google/pegasus-large style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = PegasusConfig()

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a model (with random weights) from the google/pegasus-large style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = PegasusModel(configuration)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Accessing the model configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = model.config`,wrap:!1}}),{c(){t=c("p"),t.textContent=h,o=a(),g(u.$$.fragment)},l(d){t=p(d,"P",{"data-svelte-h":!0}),M(t)!=="svelte-11lpom8"&&(t.textContent=h),o=r(d),f(u.$$.fragment,d)},m(d,m){l(d,t,m),l(d,o,m),_(u,d,m),T=!0},p:V,i(d){T||(b(u.$$.fragment,d),T=!0)},o(d){y(u.$$.fragment,d),T=!1},d(d){d&&(s(t),s(o)),k(u,d)}}}function Un(v){let t,h=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=h},l(o){t=p(o,"P",{"data-svelte-h":!0}),M(t)!=="svelte-fincs2"&&(t.innerHTML=h)},m(o,u){l(o,t,u)},p:V,d(o){o&&s(t)}}}function xn(v){let t,h="Example:",o,u,T;return u=new pe({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBQZWdhc3VzTW9kZWwlMEElMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJnb29nbGUlMkZwZWdhc3VzLWxhcmdlJTIyKSUwQW1vZGVsJTIwJTNEJTIwUGVnYXN1c01vZGVsLmZyb21fcHJldHJhaW5lZCglMjJnb29nbGUlMkZwZWdhc3VzLWxhcmdlJTIyKSUwQSUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplciglMjJTdHVkaWVzJTIwaGF2ZSUyMGJlZW4lMjBzaG93biUyMHRoYXQlMjBvd25pbmclMjBhJTIwZG9nJTIwaXMlMjBnb29kJTIwZm9yJTIweW91JTIyJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiklMEFkZWNvZGVyX2lucHV0cyUyMCUzRCUyMHRva2VuaXplciglMjJTdHVkaWVzJTIwc2hvdyUyMHRoYXQlMjIlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQW91dHB1dHMlMjAlM0QlMjBtb2RlbChpbnB1dF9pZHMlM0RpbnB1dHMuaW5wdXRfaWRzJTJDJTIwZGVjb2Rlcl9pbnB1dF9pZHMlM0RkZWNvZGVyX2lucHV0cy5pbnB1dF9pZHMpJTBBJTBBbGFzdF9oaWRkZW5fc3RhdGVzJTIwJTNEJTIwb3V0cHV0cy5sYXN0X2hpZGRlbl9zdGF0ZSUwQWxpc3QobGFzdF9oaWRkZW5fc3RhdGVzLnNoYXBlKQ==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, PegasusModel

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;google/pegasus-large&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = PegasusModel.from_pretrained(<span class="hljs-string">&quot;google/pegasus-large&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Studies have been shown that owning a dog is good for you&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>decoder_inputs = tokenizer(<span class="hljs-string">&quot;Studies show that&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(input_ids=inputs.input_ids, decoder_input_ids=decoder_inputs.input_ids)

<span class="hljs-meta">&gt;&gt;&gt; </span>last_hidden_states = outputs.last_hidden_state
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">list</span>(last_hidden_states.shape)
[<span class="hljs-number">1</span>, <span class="hljs-number">4</span>, <span class="hljs-number">1024</span>]`,wrap:!1}}),{c(){t=c("p"),t.textContent=h,o=a(),g(u.$$.fragment)},l(d){t=p(d,"P",{"data-svelte-h":!0}),M(t)!=="svelte-11lpom8"&&(t.textContent=h),o=r(d),f(u.$$.fragment,d)},m(d,m){l(d,t,m),l(d,o,m),_(u,d,m),T=!0},p:V,i(d){T||(b(u.$$.fragment,d),T=!0)},o(d){y(u.$$.fragment,d),T=!1},d(d){d&&(s(t),s(o)),k(u,d)}}}function Cn(v){let t,h=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=h},l(o){t=p(o,"P",{"data-svelte-h":!0}),M(t)!=="svelte-fincs2"&&(t.innerHTML=h)},m(o,u){l(o,t,u)},p:V,d(o){o&&s(t)}}}function In(v){let t,h="Example Summarization:",o,u,T;return u=new pe({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBQZWdhc3VzRm9yQ29uZGl0aW9uYWxHZW5lcmF0aW9uJTBBJTBBbW9kZWwlMjAlM0QlMjBQZWdhc3VzRm9yQ29uZGl0aW9uYWxHZW5lcmF0aW9uLmZyb21fcHJldHJhaW5lZCglMjJnb29nbGUlMkZwZWdhc3VzLXhzdW0lMjIpJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIyZ29vZ2xlJTJGcGVnYXN1cy14c3VtJTIyKSUwQSUwQUFSVElDTEVfVE9fU1VNTUFSSVpFJTIwJTNEJTIwKCUwQSUyMCUyMCUyMCUyMCUyMlBHJTI2RSUyMHN0YXRlZCUyMGl0JTIwc2NoZWR1bGVkJTIwdGhlJTIwYmxhY2tvdXRzJTIwaW4lMjByZXNwb25zZSUyMHRvJTIwZm9yZWNhc3RzJTIwZm9yJTIwaGlnaCUyMHdpbmRzJTIwJTIyJTBBJTIwJTIwJTIwJTIwJTIyYW1pZCUyMGRyeSUyMGNvbmRpdGlvbnMuJTIwVGhlJTIwYWltJTIwaXMlMjB0byUyMHJlZHVjZSUyMHRoZSUyMHJpc2slMjBvZiUyMHdpbGRmaXJlcy4lMjBOZWFybHklMjA4MDAlMjB0aG91c2FuZCUyMGN1c3RvbWVycyUyMHdlcmUlMjAlMjIlMEElMjAlMjAlMjAlMjAlMjJzY2hlZHVsZWQlMjB0byUyMGJlJTIwYWZmZWN0ZWQlMjBieSUyMHRoZSUyMHNodXRvZmZzJTIwd2hpY2glMjB3ZXJlJTIwZXhwZWN0ZWQlMjB0byUyMGxhc3QlMjB0aHJvdWdoJTIwYXQlMjBsZWFzdCUyMG1pZGRheSUyMHRvbW9ycm93LiUyMiUwQSklMEFpbnB1dHMlMjAlM0QlMjB0b2tlbml6ZXIoQVJUSUNMRV9UT19TVU1NQVJJWkUlMkMlMjBtYXhfbGVuZ3RoJTNEMTAyNCUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpJTBBJTBBJTIzJTIwR2VuZXJhdGUlMjBTdW1tYXJ5JTBBc3VtbWFyeV9pZHMlMjAlM0QlMjBtb2RlbC5nZW5lcmF0ZShpbnB1dHMlNUIlMjJpbnB1dF9pZHMlMjIlNUQpJTBBdG9rZW5pemVyLmJhdGNoX2RlY29kZShzdW1tYXJ5X2lkcyUyQyUyMHNraXBfc3BlY2lhbF90b2tlbnMlM0RUcnVlJTJDJTIwY2xlYW5fdXBfdG9rZW5pemF0aW9uX3NwYWNlcyUzREZhbHNlKSU1QjAlNUQ=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, PegasusForConditionalGeneration

<span class="hljs-meta">&gt;&gt;&gt; </span>model = PegasusForConditionalGeneration.from_pretrained(<span class="hljs-string">&quot;google/pegasus-xsum&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;google/pegasus-xsum&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>ARTICLE_TO_SUMMARIZE = (
<span class="hljs-meta">... </span>    <span class="hljs-string">&quot;PG&amp;E stated it scheduled the blackouts in response to forecasts for high winds &quot;</span>
<span class="hljs-meta">... </span>    <span class="hljs-string">&quot;amid dry conditions. The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were &quot;</span>
<span class="hljs-meta">... </span>    <span class="hljs-string">&quot;scheduled to be affected by the shutoffs which were expected to last through at least midday tomorrow.&quot;</span>
<span class="hljs-meta">... </span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(ARTICLE_TO_SUMMARIZE, max_length=<span class="hljs-number">1024</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Generate Summary</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>summary_ids = model.generate(inputs[<span class="hljs-string">&quot;input_ids&quot;</span>])
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer.batch_decode(summary_ids, skip_special_tokens=<span class="hljs-literal">True</span>, clean_up_tokenization_spaces=<span class="hljs-literal">False</span>)[<span class="hljs-number">0</span>]
<span class="hljs-string">&quot;California&#x27;s largest electricity provider has turned off power to hundreds of thousands of customers.&quot;</span>`,wrap:!1}}),{c(){t=c("p"),t.textContent=h,o=a(),g(u.$$.fragment)},l(d){t=p(d,"P",{"data-svelte-h":!0}),M(t)!=="svelte-50wxjj"&&(t.textContent=h),o=r(d),f(u.$$.fragment,d)},m(d,m){l(d,t,m),l(d,o,m),_(u,d,m),T=!0},p:V,i(d){T||(b(u.$$.fragment,d),T=!0)},o(d){y(u.$$.fragment,d),T=!1},d(d){d&&(s(t),s(o)),k(u,d)}}}function Gn(v){let t,h=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=h},l(o){t=p(o,"P",{"data-svelte-h":!0}),M(t)!=="svelte-fincs2"&&(t.innerHTML=h)},m(o,u){l(o,t,u)},p:V,d(o){o&&s(t)}}}function Pn(v){let t,h="Example:",o,u,T;return u=new pe({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBQZWdhc3VzRm9yQ2F1c2FsTE0lMEElMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJnb29nbGUlMkZwZWdhc3VzLWxhcmdlJTIyKSUwQW1vZGVsJTIwJTNEJTIwUGVnYXN1c0ZvckNhdXNhbExNLmZyb21fcHJldHJhaW5lZCglMjJnb29nbGUlMkZwZWdhc3VzLWxhcmdlJTIyJTJDJTIwYWRkX2Nyb3NzX2F0dGVudGlvbiUzREZhbHNlKSUwQWFzc2VydCUyMG1vZGVsLmNvbmZpZy5pc19kZWNvZGVyJTJDJTIwZiUyMiU3Qm1vZGVsLl9fY2xhc3NfXyU3RCUyMGhhcyUyMHRvJTIwYmUlMjBjb25maWd1cmVkJTIwYXMlMjBhJTIwZGVjb2Rlci4lMjIlMEFpbnB1dHMlMjAlM0QlMjB0b2tlbml6ZXIoJTIySGVsbG8lMkMlMjBteSUyMGRvZyUyMGlzJTIwY3V0ZSUyMiUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpJTBBb3V0cHV0cyUyMCUzRCUyMG1vZGVsKCoqaW5wdXRzKSUwQSUwQWxvZ2l0cyUyMCUzRCUyMG91dHB1dHMubG9naXRzJTBBZXhwZWN0ZWRfc2hhcGUlMjAlM0QlMjAlNUIxJTJDJTIwaW5wdXRzLmlucHV0X2lkcy5zaGFwZSU1Qi0xJTVEJTJDJTIwbW9kZWwuY29uZmlnLnZvY2FiX3NpemUlNUQlMEFsaXN0KGxvZ2l0cy5zaGFwZSklMjAlM0QlM0QlMjBleHBlY3RlZF9zaGFwZQ==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, PegasusForCausalLM

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;google/pegasus-large&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = PegasusForCausalLM.from_pretrained(<span class="hljs-string">&quot;google/pegasus-large&quot;</span>, add_cross_attention=<span class="hljs-literal">False</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">assert</span> model.config.is_decoder, <span class="hljs-string">f&quot;<span class="hljs-subst">{model.__class__}</span> has to be configured as a decoder.&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs)

<span class="hljs-meta">&gt;&gt;&gt; </span>logits = outputs.logits
<span class="hljs-meta">&gt;&gt;&gt; </span>expected_shape = [<span class="hljs-number">1</span>, inputs.input_ids.shape[-<span class="hljs-number">1</span>], model.config.vocab_size]
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">list</span>(logits.shape) == expected_shape
<span class="hljs-literal">True</span>`,wrap:!1}}),{c(){t=c("p"),t.textContent=h,o=a(),g(u.$$.fragment)},l(d){t=p(d,"P",{"data-svelte-h":!0}),M(t)!=="svelte-11lpom8"&&(t.textContent=h),o=r(d),f(u.$$.fragment,d)},m(d,m){l(d,t,m),l(d,o,m),_(u,d,m),T=!0},p:V,i(d){T||(b(u.$$.fragment,d),T=!0)},o(d){y(u.$$.fragment,d),T=!1},d(d){d&&(s(t),s(o)),k(u,d)}}}function Zn(v){let t,h,o,u,T,d="<em>This model was released on 2019-12-18 and added to Hugging Face Transformers on 2020-11-16.</em>",m,w,bt='<div class="flex flex-wrap space-x-1"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/> <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat"/> <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/></div>',ue,N,kt,me,Wo='<a href="https://huggingface.co/papers/1912.08777" rel="nofollow">Pegasus</a> is an encoder-decoder (sequence-to-sequence) transformer model pretrained on unlabeled text to perform abstractive summarization. Pegasus is trained jointly on two self-supervised objective functions, masked language modeling (MLM) and gap sentence generation (GSG). Whole sentences are masked and the model has to fill in the gaps in the document. It can be fine-tuned with good performance even on small datasets with only 1000 examples.',Mt,he,qo='You can find all the original Pegasus checkpoints under the <a href="https://huggingface.co/google?search_models=pegasus" rel="nofollow">Google</a> organization.',Tt,A,vt,ge,Bo='The example below demonstrates how to summarize text with <a href="/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.Pipeline">Pipeline</a>, <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel">AutoModel</a>, and from the command line.',wt,Q,$t,fe,So='Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the <a href="../quantization/overview">Quantization</a> overview for more available quantization backends.',Jt,_e,Vo='The example below uses <a href="../quantization/bitsandbytes">bitsandbytes</a> to only quantize the weights to int4.',zt,be,jt,ye,Ut,ke,No='<li><code>AdaFactor</code> is the recommended optimizer for fine-tuning Pegasus.</li> <li>This implementation of Pegasus inherits from <a href="/docs/transformers/v4.56.2/en/model_doc/bart#transformers.BartForConditionalGeneration">BartForConditionalGeneration</a> but it uses static/sinusoidal positional embeddings instead. Pegasus also starts generating with <code>pad_token_id</code> as the prefix and uses <code>num_beams=8</code>.</li>',xt,Me,Ct,G,Te,Qt,Ae,Ho=`This is the configuration class to store the configuration of a <a href="/docs/transformers/v4.56.2/en/model_doc/pegasus#transformers.PegasusModel">PegasusModel</a>. It is used to instantiate an
PEGASUS model according to the specified arguments, defining the model architecture. Instantiating a configuration
with the defaults will yield a similar configuration to that of the PEGASUS
<a href="https://huggingface.co/google/pegasus-large" rel="nofollow">google/pegasus-large</a> architecture.`,Yt,Qe,Xo=`Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> and can be used to control the model outputs. Read the
documentation from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> for more information.`,Dt,Y,It,ve,Gt,we,Lo="warning: <code>add_tokens</code> does not work at the moment.",Pt,$,$e,Ot,Ye,Eo='Construct a PEGASUS tokenizer. Based on <a href="https://github.com/google/sentencepiece" rel="nofollow">SentencePiece</a>.',Kt,De,Ro=`This tokenizer inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer">PreTrainedTokenizer</a> which contains most of the main methods. Users should refer to
this superclass for more information regarding those methods.`,eo,Z,Je,to,Oe,Ao=`Build model inputs from a sequence or a pair of sequences for sequence classification tasks by concatenating
and adding special tokens. A PEGASUS sequence has the following format, where <code>X</code> represents the sequence:`,oo,Ke,Qo="<li>single sequence: <code>X &lt;/s&gt;</code></li> <li>pair of sequences: <code>A B &lt;/s&gt;</code> (not intended use)</li>",no,et,Yo=`BOS is never used. Pairs of sequences are not the expected use case, but they will be handled without a
separator.`,so,D,ze,ao,tt,Do="Converts a sequence of tokens (string) in a single string.",ro,O,je,io,ot,Oo="Get list where entries are [1] if a token is [eos] or [pad] else 0.",lo,K,Ue,co,nt,Ko="Just EOS",Zt,xe,Ft,U,Ce,po,st,en=`Construct a “fast” PEGASUS tokenizer (backed by HuggingFace’s <em>tokenizers</em> library). Based on
<a href="https://huggingface.co/docs/tokenizers/python/latest/components.html?highlight=unigram#models" rel="nofollow">Unigram</a>.`,uo,at,tn=`This tokenizer inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast">PreTrainedTokenizerFast</a> which contains most of the main methods. Users should
refer to this superclass for more information regarding those methods.`,mo,H,Ie,ho,rt,on="Build model inputs from a sequence by adding eos to the end. no bos token is added to the front.",go,it,nn="<li>single sequence: <code>X &lt;/s&gt;</code></li> <li>pair of sequences: <code>A B &lt;/s&gt;</code> (not intended use)</li>",fo,ee,Ge,_o,dt,sn="Get list where entries are [1] if a token is [eos] or [pad] else 0.",Wt,Pe,qt,x,Ze,bo,lt,an="The bare Pegasus Model outputting raw hidden-states without any specific head on top.",yo,ct,rn=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,ko,pt,dn=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Mo,F,Fe,To,ut,ln='The <a href="/docs/transformers/v4.56.2/en/model_doc/pegasus#transformers.PegasusModel">PegasusModel</a> forward method, overrides the <code>__call__</code> special method.',vo,te,wo,oe,Bt,We,St,C,qe,$o,mt,cn="The PEGASUS Model with a language modeling head. Can be used for summarization.",Jo,ht,pn=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,zo,gt,un=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,jo,W,Be,Uo,ft,mn='The <a href="/docs/transformers/v4.56.2/en/model_doc/pegasus#transformers.PegasusForConditionalGeneration">PegasusForConditionalGeneration</a> forward method, overrides the <code>__call__</code> special method.',xo,ne,Co,se,Vt,Se,Nt,E,Ve,Io,q,Ne,Go,_t,hn='The <a href="/docs/transformers/v4.56.2/en/model_doc/pegasus#transformers.PegasusForCausalLM">PegasusForCausalLM</a> forward method, overrides the <code>__call__</code> special method.',Po,ae,Zo,re,Ht,He,Xt,yt,Lt;return N=new ce({props:{title:"Pegasus",local:"pegasus",headingTag:"h1"}}),A=new Rt({props:{warning:!1,$$slots:{default:[vn]},$$scope:{ctx:v}}}),Q=new Tn({props:{id:"usage",options:["Pipeline","AutoModel","transformers CLI"],$$slots:{default:[zn]},$$scope:{ctx:v}}}),be=new pe({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQml0c0FuZEJ5dGVzQ29uZmlnJTJDJTIwQXV0b01vZGVsRm9yU2VxMlNlcUxNJTJDJTIwQXV0b1Rva2VuaXplciUwQSUwQXF1YW50aXphdGlvbl9jb25maWclMjAlM0QlMjBCaXRzQW5kQnl0ZXNDb25maWcoJTBBJTIwJTIwJTIwJTIwbG9hZF9pbl80Yml0JTNEVHJ1ZSUyQyUwQSUyMCUyMCUyMCUyMGJuYl80Yml0X2NvbXB1dGVfZHR5cGUlM0R0b3JjaC5iZmxvYXQxNiUyQyUwQSUyMCUyMCUyMCUyMGJuYl80Yml0X3F1YW50X3R5cGUlM0QlMjJuZjQlMjIlMEEpJTBBbW9kZWwlMjAlM0QlMjBBdXRvTW9kZWxGb3JTZXEyU2VxTE0uZnJvbV9wcmV0cmFpbmVkKCUwQSUyMCUyMCUyMCUyMCUyMmdvb2dsZSUyRnBlZ2FzdXMteHN1bSUyMiUyQyUwQSUyMCUyMCUyMCUyMGR0eXBlJTNEdG9yY2guYmZsb2F0MTYlMkMlMEElMjAlMjAlMjAlMjBkZXZpY2VfbWFwJTNEJTIyYXV0byUyMiUyQyUwQSUyMCUyMCUyMCUyMHF1YW50aXphdGlvbl9jb25maWclM0RxdWFudGl6YXRpb25fY29uZmlnJTBBKSUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUwQSUyMCUyMCUyMCUyMCUyMmdvb2dsZSUyRnBlZ2FzdXMteHN1bSUyMiUwQSklMEFpbnB1dF90ZXh0JTIwJTNEJTIwJTIyJTIyJTIyUGxhbnRzJTIwYXJlJTIwcmVtYXJrYWJsZSUyMG9yZ2FuaXNtcyUyMHRoYXQlMjBwcm9kdWNlJTIwdGhlaXIlMjBvd24lMjBmb29kJTIwdXNpbmclMjBhJTIwbWV0aG9kJTIwY2FsbGVkJTIwcGhvdG9zeW50aGVzaXMuJTBBVGhpcyUyMHByb2Nlc3MlMjBpbnZvbHZlcyUyMGNvbnZlcnRpbmclMjBzdW5saWdodCUyQyUyMGNhcmJvbiUyMGRpb3hpZGUlMkMlMjBhbmQlMjB3YXRlciUyMGludG8lMjBnbHVjb3NlJTJDJTIwd2hpY2glMjBwcm92aWRlcyUyMGVuZXJneSUyMGZvciUyMGdyb3d0aC4lMEFQbGFudHMlMjBwbGF5JTIwYSUyMGNydWNpYWwlMjByb2xlJTIwaW4lMjBzdXN0YWluaW5nJTIwbGlmZSUyMG9uJTIwRWFydGglMjBieSUyMGdlbmVyYXRpbmclMjBveHlnZW4lMjBhbmQlMjBzZXJ2aW5nJTIwYXMlMjB0aGUlMjBmb3VuZGF0aW9uJTIwb2YlMjBtb3N0JTIwZWNvc3lzdGVtcy4lMjIlMjIlMjIlMEFpbnB1dF9pZHMlMjAlM0QlMjB0b2tlbml6ZXIoaW5wdXRfdGV4dCUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpLnRvKG1vZGVsLmRldmljZSklMEElMEFvdXRwdXQlMjAlM0QlMjBtb2RlbC5nZW5lcmF0ZSgqKmlucHV0X2lkcyUyQyUyMGNhY2hlX2ltcGxlbWVudGF0aW9uJTNEJTIyc3RhdGljJTIyKSUwQXByaW50KHRva2VuaXplci5kZWNvZGUob3V0cHV0JTVCMCU1RCUyQyUyMHNraXBfc3BlY2lhbF90b2tlbnMlM0RUcnVlKSk=",highlighted:`<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> BitsAndBytesConfig, AutoModelForSeq2SeqLM, AutoTokenizer

quantization_config = BitsAndBytesConfig(
    load_in_4bit=<span class="hljs-literal">True</span>,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type=<span class="hljs-string">&quot;nf4&quot;</span>
)
model = AutoModelForSeq2SeqLM.from_pretrained(
    <span class="hljs-string">&quot;google/pegasus-xsum&quot;</span>,
    dtype=torch.bfloat16,
    device_map=<span class="hljs-string">&quot;auto&quot;</span>,
    quantization_config=quantization_config
)

tokenizer = AutoTokenizer.from_pretrained(
    <span class="hljs-string">&quot;google/pegasus-xsum&quot;</span>
)
input_text = <span class="hljs-string">&quot;&quot;&quot;Plants are remarkable organisms that produce their own food using a method called photosynthesis.
This process involves converting sunlight, carbon dioxide, and water into glucose, which provides energy for growth.
Plants play a crucial role in sustaining life on Earth by generating oxygen and serving as the foundation of most ecosystems.&quot;&quot;&quot;</span>
input_ids = tokenizer(input_text, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>).to(model.device)

output = model.generate(**input_ids, cache_implementation=<span class="hljs-string">&quot;static&quot;</span>)
<span class="hljs-built_in">print</span>(tokenizer.decode(output[<span class="hljs-number">0</span>], skip_special_tokens=<span class="hljs-literal">True</span>))`,wrap:!1}}),ye=new ce({props:{title:"Notes",local:"notes",headingTag:"h2"}}),Me=new ce({props:{title:"PegasusConfig",local:"transformers.PegasusConfig",headingTag:"h2"}}),Te=new I({props:{name:"class transformers.PegasusConfig",anchor:"transformers.PegasusConfig",parameters:[{name:"vocab_size",val:" = 50265"},{name:"max_position_embeddings",val:" = 1024"},{name:"encoder_layers",val:" = 12"},{name:"encoder_ffn_dim",val:" = 4096"},{name:"encoder_attention_heads",val:" = 16"},{name:"decoder_layers",val:" = 12"},{name:"decoder_ffn_dim",val:" = 4096"},{name:"decoder_attention_heads",val:" = 16"},{name:"encoder_layerdrop",val:" = 0.0"},{name:"decoder_layerdrop",val:" = 0.0"},{name:"use_cache",val:" = True"},{name:"is_encoder_decoder",val:" = True"},{name:"activation_function",val:" = 'gelu'"},{name:"d_model",val:" = 1024"},{name:"dropout",val:" = 0.1"},{name:"attention_dropout",val:" = 0.0"},{name:"activation_dropout",val:" = 0.0"},{name:"init_std",val:" = 0.02"},{name:"decoder_start_token_id",val:" = 0"},{name:"scale_embedding",val:" = False"},{name:"pad_token_id",val:" = 0"},{name:"eos_token_id",val:" = 1"},{name:"forced_eos_token_id",val:" = 1"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.PegasusConfig.vocab_size",description:`<strong>vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 50265) &#x2014;
Vocabulary size of the PEGASUS model. Defines the number of different tokens that can be represented by the
<code>inputs_ids</code> passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/pegasus#transformers.PegasusModel">PegasusModel</a> or <code>TFPegasusModel</code>.`,name:"vocab_size"},{anchor:"transformers.PegasusConfig.d_model",description:`<strong>d_model</strong> (<code>int</code>, <em>optional</em>, defaults to 1024) &#x2014;
Dimensionality of the layers and the pooler layer.`,name:"d_model"},{anchor:"transformers.PegasusConfig.encoder_layers",description:`<strong>encoder_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 12) &#x2014;
Number of encoder layers.`,name:"encoder_layers"},{anchor:"transformers.PegasusConfig.decoder_layers",description:`<strong>decoder_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 12) &#x2014;
Number of decoder layers.`,name:"decoder_layers"},{anchor:"transformers.PegasusConfig.encoder_attention_heads",description:`<strong>encoder_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 16) &#x2014;
Number of attention heads for each attention layer in the Transformer encoder.`,name:"encoder_attention_heads"},{anchor:"transformers.PegasusConfig.decoder_attention_heads",description:`<strong>decoder_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 16) &#x2014;
Number of attention heads for each attention layer in the Transformer decoder.`,name:"decoder_attention_heads"},{anchor:"transformers.PegasusConfig.decoder_ffn_dim",description:`<strong>decoder_ffn_dim</strong> (<code>int</code>, <em>optional</em>, defaults to 4096) &#x2014;
Dimensionality of the &#x201C;intermediate&#x201D; (often named feed-forward) layer in decoder.`,name:"decoder_ffn_dim"},{anchor:"transformers.PegasusConfig.encoder_ffn_dim",description:`<strong>encoder_ffn_dim</strong> (<code>int</code>, <em>optional</em>, defaults to 4096) &#x2014;
Dimensionality of the &#x201C;intermediate&#x201D; (often named feed-forward) layer in decoder.`,name:"encoder_ffn_dim"},{anchor:"transformers.PegasusConfig.activation_function",description:`<strong>activation_function</strong> (<code>str</code> or <code>function</code>, <em>optional</em>, defaults to <code>&quot;gelu&quot;</code>) &#x2014;
The non-linear activation function (function or string) in the encoder and pooler. If string, <code>&quot;gelu&quot;</code>,
<code>&quot;relu&quot;</code>, <code>&quot;silu&quot;</code> and <code>&quot;gelu_new&quot;</code> are supported.`,name:"activation_function"},{anchor:"transformers.PegasusConfig.dropout",description:`<strong>dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.`,name:"dropout"},{anchor:"transformers.PegasusConfig.attention_dropout",description:`<strong>attention_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The dropout ratio for the attention probabilities.`,name:"attention_dropout"},{anchor:"transformers.PegasusConfig.activation_dropout",description:`<strong>activation_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The dropout ratio for activations inside the fully connected layer.`,name:"activation_dropout"},{anchor:"transformers.PegasusConfig.max_position_embeddings",description:`<strong>max_position_embeddings</strong> (<code>int</code>, <em>optional</em>, defaults to 1024) &#x2014;
The maximum sequence length that this model might ever be used with. Typically set this to something large
just in case (e.g., 512 or 1024 or 2048).`,name:"max_position_embeddings"},{anchor:"transformers.PegasusConfig.init_std",description:`<strong>init_std</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.`,name:"init_std"},{anchor:"transformers.PegasusConfig.encoder_layerdrop",description:`<strong>encoder_layerdrop</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The LayerDrop probability for the encoder. See the [LayerDrop paper](see <a href="https://huggingface.co/papers/1909.11556" rel="nofollow">https://huggingface.co/papers/1909.11556</a>)
for more details.`,name:"encoder_layerdrop"},{anchor:"transformers.PegasusConfig.decoder_layerdrop",description:`<strong>decoder_layerdrop</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The LayerDrop probability for the decoder. See the [LayerDrop paper](see <a href="https://huggingface.co/papers/1909.11556" rel="nofollow">https://huggingface.co/papers/1909.11556</a>)
for more details.`,name:"decoder_layerdrop"},{anchor:"transformers.PegasusConfig.scale_embedding",description:`<strong>scale_embedding</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Scale embeddings by diving by sqrt(d_model).`,name:"scale_embedding"},{anchor:"transformers.PegasusConfig.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not the model should return the last key/values attentions (not used by all models)`,name:"use_cache"},{anchor:"transformers.PegasusConfig.forced_eos_token_id",description:`<strong>forced_eos_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 1) &#x2014;
The id of the token to force as the last generated token when <code>max_length</code> is reached. Usually set to
<code>eos_token_id</code>.`,name:"forced_eos_token_id"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/pegasus/configuration_pegasus.py#L24"}}),Y=new At({props:{anchor:"transformers.PegasusConfig.example",$$slots:{default:[jn]},$$scope:{ctx:v}}}),ve=new ce({props:{title:"PegasusTokenizer",local:"transformers.PegasusTokenizer",headingTag:"h2"}}),$e=new I({props:{name:"class transformers.PegasusTokenizer",anchor:"transformers.PegasusTokenizer",parameters:[{name:"vocab_file",val:""},{name:"pad_token",val:" = '<pad>'"},{name:"eos_token",val:" = '</s>'"},{name:"unk_token",val:" = '<unk>'"},{name:"mask_token",val:" = '<mask_2>'"},{name:"mask_token_sent",val:" = '<mask_1>'"},{name:"additional_special_tokens",val:" = None"},{name:"offset",val:" = 103"},{name:"sp_model_kwargs",val:": typing.Optional[dict[str, typing.Any]] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.PegasusTokenizer.vocab_file",description:`<strong>vocab_file</strong> (<code>str</code>) &#x2014;
<a href="https://github.com/google/sentencepiece" rel="nofollow">SentencePiece</a> file (generally has a <em>.spm</em> extension) that
contains the vocabulary necessary to instantiate a tokenizer.`,name:"vocab_file"},{anchor:"transformers.PegasusTokenizer.pad_token",description:`<strong>pad_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;pad&gt;&quot;</code>) &#x2014;
The token used for padding, for example when batching sequences of different lengths.`,name:"pad_token"},{anchor:"transformers.PegasusTokenizer.eos_token",description:`<strong>eos_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;/s&gt;&quot;</code>) &#x2014;
The end of sequence token.</p>
<div class="course-tip  bg-gradient-to-br dark:bg-gradient-to-r before:border-green-500 dark:before:border-green-800 from-green-50 dark:from-gray-900 to-white dark:to-gray-950 border border-green-50 text-green-700 dark:text-gray-400">
						
<p>When building a sequence using special tokens, this is not the token that is used for the end of sequence.
The token used is the <code>sep_token</code>.</p>

					</div>`,name:"eos_token"},{anchor:"transformers.PegasusTokenizer.unk_token",description:`<strong>unk_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;unk&gt;&quot;</code>) &#x2014;
The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
token instead.`,name:"unk_token"},{anchor:"transformers.PegasusTokenizer.mask_token",description:`<strong>mask_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;mask_2&gt;&quot;</code>) &#x2014;
The token used for masking single token values. This is the token used when training this model with masked
language modeling (MLM). This is the token that the PEGASUS encoder will try to predict during pretraining.
It corresponds to <em>[MASK2]</em> in <a href="https://huggingface.co/papers/1912.08777" rel="nofollow">PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive
Summarization</a>.`,name:"mask_token"},{anchor:"transformers.PegasusTokenizer.mask_token_sent",description:`<strong>mask_token_sent</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;mask_1&gt;&quot;</code>) &#x2014;
The token used for masking whole target sentences. This is the token used when training this model with gap
sentences generation (GSG). This is the sentence that the PEGASUS decoder will try to predict during
pretraining. It corresponds to <em>[MASK1]</em> in <a href="https://huggingface.co/papers/1912.08777" rel="nofollow">PEGASUS: Pre-training with Extracted Gap-sentences for
Abstractive Summarization</a>.`,name:"mask_token_sent"},{anchor:"transformers.PegasusTokenizer.additional_special_tokens",description:`<strong>additional_special_tokens</strong> (<code>List[str]</code>, <em>optional</em>) &#x2014;
Additional special tokens used by the tokenizer. If no additional_special_tokens are provided <mask_2> and
<unk_2, …, unk_102> are used as additional special tokens corresponding to the <a href="https://github.com/google-research/pegasus/blob/939830367bcf411193d2b5eca2f2f90f3f9260ca/pegasus/ops/pretrain_parsing_ops.cc#L66" rel="nofollow">original PEGASUS
tokenizer</a>
that uses the tokens 2 - 104 only for pretraining</unk_2,></mask_2>`,name:"additional_special_tokens"},{anchor:"transformers.PegasusTokenizer.sp_model_kwargs",description:`<strong>sp_model_kwargs</strong> (<code>dict</code>, <em>optional</em>) &#x2014;
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
</ul>`,name:"sp_model_kwargs"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/pegasus/tokenization_pegasus.py#L38"}}),Je=new I({props:{name:"build_inputs_with_special_tokens",anchor:"transformers.PegasusTokenizer.build_inputs_with_special_tokens",parameters:[{name:"token_ids_0",val:""},{name:"token_ids_1",val:" = None"}],parametersDescription:[{anchor:"transformers.PegasusTokenizer.build_inputs_with_special_tokens.token_ids_0",description:`<strong>token_ids_0</strong> (<code>List[int]</code>) &#x2014;
List of IDs to which the special tokens will be added.`,name:"token_ids_0"},{anchor:"transformers.PegasusTokenizer.build_inputs_with_special_tokens.token_ids_1",description:`<strong>token_ids_1</strong> (<code>List[int]</code>, <em>optional</em>) &#x2014;
Optional second list of IDs for sequence pairs.`,name:"token_ids_1"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/pegasus/tokenization_pegasus.py#L249",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>List of <a href="../glossary#input-ids">input IDs</a> with the appropriate special tokens.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>List[int]</code></p>
`}}),ze=new I({props:{name:"convert_tokens_to_string",anchor:"transformers.PegasusTokenizer.convert_tokens_to_string",parameters:[{name:"tokens",val:""}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/pegasus/tokenization_pegasus.py#L214"}}),je=new I({props:{name:"get_special_tokens_mask",anchor:"transformers.PegasusTokenizer.get_special_tokens_mask",parameters:[{name:"token_ids_0",val:": list"},{name:"token_ids_1",val:": typing.Optional[list] = None"},{name:"already_has_special_tokens",val:": bool = False"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/pegasus/tokenization_pegasus.py#L238"}}),Ue=new I({props:{name:"num_special_tokens_to_add",anchor:"transformers.PegasusTokenizer.num_special_tokens_to_add",parameters:[{name:"pair",val:" = False"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/pegasus/tokenization_pegasus.py#L228"}}),xe=new ce({props:{title:"PegasusTokenizerFast",local:"transformers.PegasusTokenizerFast",headingTag:"h2"}}),Ce=new I({props:{name:"class transformers.PegasusTokenizerFast",anchor:"transformers.PegasusTokenizerFast",parameters:[{name:"vocab_file",val:" = None"},{name:"tokenizer_file",val:" = None"},{name:"pad_token",val:" = '<pad>'"},{name:"eos_token",val:" = '</s>'"},{name:"unk_token",val:" = '<unk>'"},{name:"mask_token",val:" = '<mask_2>'"},{name:"mask_token_sent",val:" = '<mask_1>'"},{name:"additional_special_tokens",val:" = None"},{name:"offset",val:" = 103"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.PegasusTokenizerFast.vocab_file",description:`<strong>vocab_file</strong> (<code>str</code>) &#x2014;
<a href="https://github.com/google/sentencepiece" rel="nofollow">SentencePiece</a> file (generally has a <em>.spm</em> extension) that
contains the vocabulary necessary to instantiate a tokenizer.`,name:"vocab_file"},{anchor:"transformers.PegasusTokenizerFast.pad_token",description:`<strong>pad_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;pad&gt;&quot;</code>) &#x2014;
The token used for padding, for example when batching sequences of different lengths.`,name:"pad_token"},{anchor:"transformers.PegasusTokenizerFast.eos_token",description:`<strong>eos_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;/s&gt;&quot;</code>) &#x2014;
The end of sequence token.</p>
<div class="course-tip  bg-gradient-to-br dark:bg-gradient-to-r before:border-green-500 dark:before:border-green-800 from-green-50 dark:from-gray-900 to-white dark:to-gray-950 border border-green-50 text-green-700 dark:text-gray-400">
						
<p>When building a sequence using special tokens, this is not the token that is used for the end of sequence.
The token used is the <code>sep_token</code>.</p>

					</div>`,name:"eos_token"},{anchor:"transformers.PegasusTokenizerFast.unk_token",description:`<strong>unk_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;unk&gt;&quot;</code>) &#x2014;
The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
token instead.`,name:"unk_token"},{anchor:"transformers.PegasusTokenizerFast.mask_token",description:`<strong>mask_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;mask_2&gt;&quot;</code>) &#x2014;
The token used for masking single token values. This is the token used when training this model with masked
language modeling (MLM). This is the token that the PEGASUS encoder will try to predict during pretraining.
It corresponds to <em>[MASK2]</em> in <a href="https://huggingface.co/papers/1912.08777" rel="nofollow">PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive
Summarization</a>.`,name:"mask_token"},{anchor:"transformers.PegasusTokenizerFast.mask_token_sent",description:`<strong>mask_token_sent</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;mask_1&gt;&quot;</code>) &#x2014;
The token used for masking whole target sentences. This is the token used when training this model with gap
sentences generation (GSG). This is the sentence that the PEGASUS decoder will try to predict during
pretraining. It corresponds to <em>[MASK1]</em> in <a href="https://huggingface.co/papers/1912.08777" rel="nofollow">PEGASUS: Pre-training with Extracted Gap-sentences for
Abstractive Summarization</a>.`,name:"mask_token_sent"},{anchor:"transformers.PegasusTokenizerFast.additional_special_tokens",description:`<strong>additional_special_tokens</strong> (<code>List[str]</code>, <em>optional</em>) &#x2014;
Additional special tokens used by the tokenizer. If no additional_special_tokens are provided <mask_2> and
<unk_2, …, unk_102> are used as additional special tokens corresponding to the <a href="https://github.com/google-research/pegasus/blob/939830367bcf411193d2b5eca2f2f90f3f9260ca/pegasus/ops/pretrain_parsing_ops.cc#L66" rel="nofollow">original PEGASUS
tokenizer</a>
that uses the tokens 2 - 104 only for pretraining</unk_2,></mask_2>`,name:"additional_special_tokens"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/pegasus/tokenization_pegasus_fast.py#L39"}}),Ie=new I({props:{name:"build_inputs_with_special_tokens",anchor:"transformers.PegasusTokenizerFast.build_inputs_with_special_tokens",parameters:[{name:"token_ids_0",val:""},{name:"token_ids_1",val:" = None"}],parametersDescription:[{anchor:"transformers.PegasusTokenizerFast.build_inputs_with_special_tokens.token_ids_0",description:`<strong>token_ids_0</strong> (<code>List[int]</code>) &#x2014;
List of IDs to which the special tokens will be added`,name:"token_ids_0"},{anchor:"transformers.PegasusTokenizerFast.build_inputs_with_special_tokens.token_ids_1",description:`<strong>token_ids_1</strong> (<code>List[int]</code>, <em>optional</em>) &#x2014;
Optional second list of IDs for sequence pairs.`,name:"token_ids_1"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/pegasus/tokenization_pegasus_fast.py#L174",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>list of <a href="../glossary#input-ids">input IDs</a> with the appropriate special tokens.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>List[int]</code></p>
`}}),Ge=new I({props:{name:"get_special_tokens_mask",anchor:"transformers.PegasusTokenizerFast.get_special_tokens_mask",parameters:[{name:"token_ids_0",val:": list"},{name:"token_ids_1",val:": typing.Optional[list] = None"},{name:"already_has_special_tokens",val:": bool = False"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/pegasus/tokenization_pegasus_fast.py#L163"}}),Pe=new ce({props:{title:"PegasusModel",local:"transformers.PegasusModel",headingTag:"h2"}}),Ze=new I({props:{name:"class transformers.PegasusModel",anchor:"transformers.PegasusModel",parameters:[{name:"config",val:": PegasusConfig"}],parametersDescription:[{anchor:"transformers.PegasusModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/pegasus#transformers.PegasusConfig">PegasusConfig</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/pegasus/modeling_pegasus.py#L1164"}}),Fe=new I({props:{name:"forward",anchor:"transformers.PegasusModel.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"decoder_input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"decoder_attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"decoder_head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"cross_attn_head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_outputs",val:": typing.Optional[tuple[torch.FloatTensor]] = None"},{name:"past_key_values",val:": typing.Optional[tuple[torch.FloatTensor]] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"decoder_inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.Tensor] = None"}],parametersDescription:[{anchor:"transformers.PegasusModel.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.PegasusModel.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.PegasusModel.forward.decoder_input_ids",description:`<strong>decoder_input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of decoder input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#decoder-input-ids">What are decoder input IDs?</a></p>
<p>Pegasus uses the <code>pad_token_id</code> as the starting token for <code>decoder_input_ids</code> generation. If
<code>past_key_values</code> is used, optionally only the last <code>decoder_input_ids</code> have to be input (see
<code>past_key_values</code>).`,name:"decoder_input_ids"},{anchor:"transformers.PegasusModel.forward.decoder_attention_mask",description:`<strong>decoder_attention_mask</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Default behavior: generate a tensor that ignores pad tokens in <code>decoder_input_ids</code>. Causal mask will also
be used by default.`,name:"decoder_attention_mask"},{anchor:"transformers.PegasusModel.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.PegasusModel.forward.decoder_head_mask",description:`<strong>decoder_head_mask</strong> (<code>torch.Tensor</code> of shape <code>(decoder_layers, decoder_attention_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the attention modules in the decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"decoder_head_mask"},{anchor:"transformers.PegasusModel.forward.cross_attn_head_mask",description:`<strong>cross_attn_head_mask</strong> (<code>torch.Tensor</code> of shape <code>(decoder_layers, decoder_attention_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the cross-attention modules in the decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"cross_attn_head_mask"},{anchor:"transformers.PegasusModel.forward.encoder_outputs",description:`<strong>encoder_outputs</strong> (<code>tuple[torch.FloatTensor]</code>, <em>optional</em>) &#x2014;
Tuple consists of (<code>last_hidden_state</code>, <em>optional</em>: <code>hidden_states</code>, <em>optional</em>: <code>attentions</code>)
<code>last_hidden_state</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) is a sequence of
hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.`,name:"encoder_outputs"},{anchor:"transformers.PegasusModel.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>tuple[torch.FloatTensor]</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.PegasusModel.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.PegasusModel.forward.decoder_inputs_embeds",description:`<strong>decoder_inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, target_sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>decoder_input_ids</code> you can choose to directly pass an embedded
representation. If <code>past_key_values</code> is used, optionally only the last <code>decoder_inputs_embeds</code> have to be
input (see <code>past_key_values</code>). This is useful if you want more control over how to convert
<code>decoder_input_ids</code> indices into associated vectors than the model&#x2019;s internal embedding lookup matrix.</p>
<p>If <code>decoder_input_ids</code> and <code>decoder_inputs_embeds</code> are both unset, <code>decoder_inputs_embeds</code> takes the value
of <code>inputs_embeds</code>.`,name:"decoder_inputs_embeds"},{anchor:"transformers.PegasusModel.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.PegasusModel.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.PegasusModel.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.PegasusModel.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.PegasusModel.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.Tensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/pegasus/modeling_pegasus.py#L1213",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqModelOutput"
>transformers.modeling_outputs.Seq2SeqModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/pegasus#transformers.PegasusConfig"
>PegasusConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>last_hidden_state</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>) — Sequence of hidden-states at the output of the last layer of the decoder of the model.</p>
<p>If <code>past_key_values</code> is used only the last hidden-state of the sequences of shape <code>(batch_size, 1, hidden_size)</code> is output.</p>
</li>
<li>
<p><strong>past_key_values</strong> (<code>EncoderDecoderCache</code>, <em>optional</em>, returned when <code>use_cache=True</code> is passed or when <code>config.use_cache=True</code>) — It is a <a
  href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.EncoderDecoderCache"
>EncoderDecoderCache</a> instance. For more details, see our <a
  href="https://huggingface.co/docs/transformers/en/kv_cache"
  rel="nofollow"
>kv cache guide</a>.</p>
<p>Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used (see <code>past_key_values</code> input) to speed up sequential decoding.</p>
</li>
<li>
<p><strong>decoder_hidden_states</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings, if the model has an embedding layer, +
one for the output of each layer) of shape <code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the decoder at the output of each layer plus the optional initial embedding outputs.</p>
</li>
<li>
<p><strong>decoder_attentions</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
self-attention heads.</p>
</li>
<li>
<p><strong>cross_attentions</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the
weighted average in the cross-attention heads.</p>
</li>
<li>
<p><strong>encoder_last_hidden_state</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) — Sequence of hidden-states at the output of the last layer of the encoder of the model.</p>
</li>
<li>
<p><strong>encoder_hidden_states</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings, if the model has an embedding layer, +
one for the output of each layer) of shape <code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the encoder at the output of each layer plus the optional initial embedding outputs.</p>
</li>
<li>
<p><strong>encoder_attentions</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
self-attention heads.</p>
</li>
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqModelOutput"
>transformers.modeling_outputs.Seq2SeqModelOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),te=new Rt({props:{$$slots:{default:[Un]},$$scope:{ctx:v}}}),oe=new At({props:{anchor:"transformers.PegasusModel.forward.example",$$slots:{default:[xn]},$$scope:{ctx:v}}}),We=new ce({props:{title:"PegasusForConditionalGeneration",local:"transformers.PegasusForConditionalGeneration",headingTag:"h2"}}),qe=new I({props:{name:"class transformers.PegasusForConditionalGeneration",anchor:"transformers.PegasusForConditionalGeneration",parameters:[{name:"config",val:": PegasusConfig"}],parametersDescription:[{anchor:"transformers.PegasusForConditionalGeneration.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/pegasus#transformers.PegasusConfig">PegasusConfig</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/pegasus/modeling_pegasus.py#L1334"}}),Be=new I({props:{name:"forward",anchor:"transformers.PegasusForConditionalGeneration.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"decoder_input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"decoder_attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"decoder_head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"cross_attn_head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_outputs",val:": typing.Optional[tuple[torch.FloatTensor]] = None"},{name:"past_key_values",val:": typing.Optional[tuple[torch.FloatTensor]] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"decoder_inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"labels",val:": typing.Optional[torch.Tensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.Tensor] = None"}],parametersDescription:[{anchor:"transformers.PegasusForConditionalGeneration.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.PegasusForConditionalGeneration.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.PegasusForConditionalGeneration.forward.decoder_input_ids",description:`<strong>decoder_input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of decoder input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#decoder-input-ids">What are decoder input IDs?</a></p>
<p>Pegasus uses the <code>pad_token_id</code> as the starting token for <code>decoder_input_ids</code> generation. If
<code>past_key_values</code> is used, optionally only the last <code>decoder_input_ids</code> have to be input (see
<code>past_key_values</code>).`,name:"decoder_input_ids"},{anchor:"transformers.PegasusForConditionalGeneration.forward.decoder_attention_mask",description:`<strong>decoder_attention_mask</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Default behavior: generate a tensor that ignores pad tokens in <code>decoder_input_ids</code>. Causal mask will also
be used by default.`,name:"decoder_attention_mask"},{anchor:"transformers.PegasusForConditionalGeneration.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.PegasusForConditionalGeneration.forward.decoder_head_mask",description:`<strong>decoder_head_mask</strong> (<code>torch.Tensor</code> of shape <code>(decoder_layers, decoder_attention_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the attention modules in the decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"decoder_head_mask"},{anchor:"transformers.PegasusForConditionalGeneration.forward.cross_attn_head_mask",description:`<strong>cross_attn_head_mask</strong> (<code>torch.Tensor</code> of shape <code>(decoder_layers, decoder_attention_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the cross-attention modules in the decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"cross_attn_head_mask"},{anchor:"transformers.PegasusForConditionalGeneration.forward.encoder_outputs",description:`<strong>encoder_outputs</strong> (<code>tuple[torch.FloatTensor]</code>, <em>optional</em>) &#x2014;
Tuple consists of (<code>last_hidden_state</code>, <em>optional</em>: <code>hidden_states</code>, <em>optional</em>: <code>attentions</code>)
<code>last_hidden_state</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) is a sequence of
hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.`,name:"encoder_outputs"},{anchor:"transformers.PegasusForConditionalGeneration.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>tuple[torch.FloatTensor]</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.PegasusForConditionalGeneration.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.PegasusForConditionalGeneration.forward.decoder_inputs_embeds",description:`<strong>decoder_inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, target_sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>decoder_input_ids</code> you can choose to directly pass an embedded
representation. If <code>past_key_values</code> is used, optionally only the last <code>decoder_inputs_embeds</code> have to be
input (see <code>past_key_values</code>). This is useful if you want more control over how to convert
<code>decoder_input_ids</code> indices into associated vectors than the model&#x2019;s internal embedding lookup matrix.</p>
<p>If <code>decoder_input_ids</code> and <code>decoder_inputs_embeds</code> are both unset, <code>decoder_inputs_embeds</code> takes the value
of <code>inputs_embeds</code>.`,name:"decoder_inputs_embeds"},{anchor:"transformers.PegasusForConditionalGeneration.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.PegasusForConditionalGeneration.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.PegasusForConditionalGeneration.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.PegasusForConditionalGeneration.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.PegasusForConditionalGeneration.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.PegasusForConditionalGeneration.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.Tensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/pegasus/modeling_pegasus.py#L1393",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqLMOutput"
>transformers.modeling_outputs.Seq2SeqLMOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/pegasus#transformers.PegasusConfig"
>PegasusConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>loss</strong> (<code>torch.FloatTensor</code> of shape <code>(1,)</code>, <em>optional</em>, returned when <code>labels</code> is provided) — Language modeling loss.</p>
</li>
<li>
<p><strong>logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, config.vocab_size)</code>) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).</p>
</li>
<li>
<p><strong>past_key_values</strong> (<code>EncoderDecoderCache</code>, <em>optional</em>, returned when <code>use_cache=True</code> is passed or when <code>config.use_cache=True</code>) — It is a <a
  href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.EncoderDecoderCache"
>EncoderDecoderCache</a> instance. For more details, see our <a
  href="https://huggingface.co/docs/transformers/en/kv_cache"
  rel="nofollow"
>kv cache guide</a>.</p>
<p>Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used (see <code>past_key_values</code> input) to speed up sequential decoding.</p>
</li>
<li>
<p><strong>decoder_hidden_states</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings, if the model has an embedding layer, +
one for the output of each layer) of shape <code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.</p>
</li>
<li>
<p><strong>decoder_attentions</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
self-attention heads.</p>
</li>
<li>
<p><strong>cross_attentions</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the
weighted average in the cross-attention heads.</p>
</li>
<li>
<p><strong>encoder_last_hidden_state</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) — Sequence of hidden-states at the output of the last layer of the encoder of the model.</p>
</li>
<li>
<p><strong>encoder_hidden_states</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings, if the model has an embedding layer, +
one for the output of each layer) of shape <code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.</p>
</li>
<li>
<p><strong>encoder_attentions</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
self-attention heads.</p>
</li>
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqLMOutput"
>transformers.modeling_outputs.Seq2SeqLMOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),ne=new Rt({props:{$$slots:{default:[Cn]},$$scope:{ctx:v}}}),se=new At({props:{anchor:"transformers.PegasusForConditionalGeneration.forward.example",$$slots:{default:[In]},$$scope:{ctx:v}}}),Se=new ce({props:{title:"PegasusForCausalLM",local:"transformers.PegasusForCausalLM",headingTag:"h2"}}),Ve=new I({props:{name:"class transformers.PegasusForCausalLM",anchor:"transformers.PegasusForCausalLM",parameters:[{name:"config",val:""}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/pegasus/modeling_pegasus.py#L1532"}}),Ne=new I({props:{name:"forward",anchor:"transformers.PegasusForCausalLM.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_hidden_states",val:": typing.Optional[torch.FloatTensor] = None"},{name:"encoder_attention_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"cross_attn_head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"}],parametersDescription:[{anchor:"transformers.PegasusForCausalLM.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.PegasusForCausalLM.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.PegasusForCausalLM.forward.encoder_hidden_states",description:`<strong>encoder_hidden_states</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
if the model is configured as a decoder.`,name:"encoder_hidden_states"},{anchor:"transformers.PegasusForCausalLM.forward.encoder_attention_mask",description:`<strong>encoder_attention_mask</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
the cross-attention if the model is configured as a decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>`,name:"encoder_attention_mask"},{anchor:"transformers.PegasusForCausalLM.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.PegasusForCausalLM.forward.cross_attn_head_mask",description:`<strong>cross_attn_head_mask</strong> (<code>torch.Tensor</code> of shape <code>(decoder_layers, decoder_attention_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the cross-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"cross_attn_head_mask"},{anchor:"transformers.PegasusForCausalLM.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.PegasusForCausalLM.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.PegasusForCausalLM.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.PegasusForCausalLM.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.PegasusForCausalLM.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.PegasusForCausalLM.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.PegasusForCausalLM.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.PegasusForCausalLM.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/pegasus/modeling_pegasus.py#L1581",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithCrossAttentions"
>transformers.modeling_outputs.CausalLMOutputWithCrossAttentions</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/pegasus#transformers.PegasusConfig"
>PegasusConfig</a>) and inputs.</p>
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
`}}),ae=new Rt({props:{$$slots:{default:[Gn]},$$scope:{ctx:v}}}),re=new At({props:{anchor:"transformers.PegasusForCausalLM.forward.example",$$slots:{default:[Pn]},$$scope:{ctx:v}}}),He=new Mn({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/pegasus.md"}}),{c(){t=c("meta"),h=a(),o=c("p"),u=a(),T=c("p"),T.innerHTML=d,m=a(),w=c("div"),w.innerHTML=bt,ue=a(),g(N.$$.fragment),kt=a(),me=c("p"),me.innerHTML=Wo,Mt=a(),he=c("p"),he.innerHTML=qo,Tt=a(),g(A.$$.fragment),vt=a(),ge=c("p"),ge.innerHTML=Bo,wt=a(),g(Q.$$.fragment),$t=a(),fe=c("p"),fe.innerHTML=So,Jt=a(),_e=c("p"),_e.innerHTML=Vo,zt=a(),g(be.$$.fragment),jt=a(),g(ye.$$.fragment),Ut=a(),ke=c("ul"),ke.innerHTML=No,xt=a(),g(Me.$$.fragment),Ct=a(),G=c("div"),g(Te.$$.fragment),Qt=a(),Ae=c("p"),Ae.innerHTML=Ho,Yt=a(),Qe=c("p"),Qe.innerHTML=Xo,Dt=a(),g(Y.$$.fragment),It=a(),g(ve.$$.fragment),Gt=a(),we=c("p"),we.innerHTML=Lo,Pt=a(),$=c("div"),g($e.$$.fragment),Ot=a(),Ye=c("p"),Ye.innerHTML=Eo,Kt=a(),De=c("p"),De.innerHTML=Ro,eo=a(),Z=c("div"),g(Je.$$.fragment),to=a(),Oe=c("p"),Oe.innerHTML=Ao,oo=a(),Ke=c("ul"),Ke.innerHTML=Qo,no=a(),et=c("p"),et.textContent=Yo,so=a(),D=c("div"),g(ze.$$.fragment),ao=a(),tt=c("p"),tt.textContent=Do,ro=a(),O=c("div"),g(je.$$.fragment),io=a(),ot=c("p"),ot.textContent=Oo,lo=a(),K=c("div"),g(Ue.$$.fragment),co=a(),nt=c("p"),nt.textContent=Ko,Zt=a(),g(xe.$$.fragment),Ft=a(),U=c("div"),g(Ce.$$.fragment),po=a(),st=c("p"),st.innerHTML=en,uo=a(),at=c("p"),at.innerHTML=tn,mo=a(),H=c("div"),g(Ie.$$.fragment),ho=a(),rt=c("p"),rt.textContent=on,go=a(),it=c("ul"),it.innerHTML=nn,fo=a(),ee=c("div"),g(Ge.$$.fragment),_o=a(),dt=c("p"),dt.textContent=sn,Wt=a(),g(Pe.$$.fragment),qt=a(),x=c("div"),g(Ze.$$.fragment),bo=a(),lt=c("p"),lt.textContent=an,yo=a(),ct=c("p"),ct.innerHTML=rn,ko=a(),pt=c("p"),pt.innerHTML=dn,Mo=a(),F=c("div"),g(Fe.$$.fragment),To=a(),ut=c("p"),ut.innerHTML=ln,vo=a(),g(te.$$.fragment),wo=a(),g(oe.$$.fragment),Bt=a(),g(We.$$.fragment),St=a(),C=c("div"),g(qe.$$.fragment),$o=a(),mt=c("p"),mt.textContent=cn,Jo=a(),ht=c("p"),ht.innerHTML=pn,zo=a(),gt=c("p"),gt.innerHTML=un,jo=a(),W=c("div"),g(Be.$$.fragment),Uo=a(),ft=c("p"),ft.innerHTML=mn,xo=a(),g(ne.$$.fragment),Co=a(),g(se.$$.fragment),Vt=a(),g(Se.$$.fragment),Nt=a(),E=c("div"),g(Ve.$$.fragment),Io=a(),q=c("div"),g(Ne.$$.fragment),Go=a(),_t=c("p"),_t.innerHTML=hn,Po=a(),g(ae.$$.fragment),Zo=a(),g(re.$$.fragment),Ht=a(),g(He.$$.fragment),Xt=a(),yt=c("p"),this.h()},l(e){const n=yn("svelte-u9bgzb",document.head);t=p(n,"META",{name:!0,content:!0}),n.forEach(s),h=r(e),o=p(e,"P",{}),z(o).forEach(s),u=r(e),T=p(e,"P",{"data-svelte-h":!0}),M(T)!=="svelte-19pii71"&&(T.innerHTML=d),m=r(e),w=p(e,"DIV",{style:!0,"data-svelte-h":!0}),M(w)!=="svelte-2m0t7r"&&(w.innerHTML=bt),ue=r(e),f(N.$$.fragment,e),kt=r(e),me=p(e,"P",{"data-svelte-h":!0}),M(me)!=="svelte-1f6ndxx"&&(me.innerHTML=Wo),Mt=r(e),he=p(e,"P",{"data-svelte-h":!0}),M(he)!=="svelte-2qo7ly"&&(he.innerHTML=qo),Tt=r(e),f(A.$$.fragment,e),vt=r(e),ge=p(e,"P",{"data-svelte-h":!0}),M(ge)!=="svelte-1q65a0t"&&(ge.innerHTML=Bo),wt=r(e),f(Q.$$.fragment,e),$t=r(e),fe=p(e,"P",{"data-svelte-h":!0}),M(fe)!=="svelte-nf5ooi"&&(fe.innerHTML=So),Jt=r(e),_e=p(e,"P",{"data-svelte-h":!0}),M(_e)!=="svelte-11sw8fc"&&(_e.innerHTML=Vo),zt=r(e),f(be.$$.fragment,e),jt=r(e),f(ye.$$.fragment,e),Ut=r(e),ke=p(e,"UL",{"data-svelte-h":!0}),M(ke)!=="svelte-1bpibbg"&&(ke.innerHTML=No),xt=r(e),f(Me.$$.fragment,e),Ct=r(e),G=p(e,"DIV",{class:!0});var B=z(G);f(Te.$$.fragment,B),Qt=r(B),Ae=p(B,"P",{"data-svelte-h":!0}),M(Ae)!=="svelte-gdgd2t"&&(Ae.innerHTML=Ho),Yt=r(B),Qe=p(B,"P",{"data-svelte-h":!0}),M(Qe)!=="svelte-1ek1ss9"&&(Qe.innerHTML=Xo),Dt=r(B),f(Y.$$.fragment,B),B.forEach(s),It=r(e),f(ve.$$.fragment,e),Gt=r(e),we=p(e,"P",{"data-svelte-h":!0}),M(we)!=="svelte-1w4c0l4"&&(we.innerHTML=Lo),Pt=r(e),$=p(e,"DIV",{class:!0});var J=z($);f($e.$$.fragment,J),Ot=r(J),Ye=p(J,"P",{"data-svelte-h":!0}),M(Ye)!=="svelte-174ofpm"&&(Ye.innerHTML=Eo),Kt=r(J),De=p(J,"P",{"data-svelte-h":!0}),M(De)!=="svelte-ntrhio"&&(De.innerHTML=Ro),eo=r(J),Z=p(J,"DIV",{class:!0});var S=z(Z);f(Je.$$.fragment,S),to=r(S),Oe=p(S,"P",{"data-svelte-h":!0}),M(Oe)!=="svelte-14d8kgv"&&(Oe.innerHTML=Ao),oo=r(S),Ke=p(S,"UL",{"data-svelte-h":!0}),M(Ke)!=="svelte-1xl4xih"&&(Ke.innerHTML=Qo),no=r(S),et=p(S,"P",{"data-svelte-h":!0}),M(et)!=="svelte-46aam0"&&(et.textContent=Yo),S.forEach(s),so=r(J),D=p(J,"DIV",{class:!0});var Xe=z(D);f(ze.$$.fragment,Xe),ao=r(Xe),tt=p(Xe,"P",{"data-svelte-h":!0}),M(tt)!=="svelte-b3k2yi"&&(tt.textContent=Do),Xe.forEach(s),ro=r(J),O=p(J,"DIV",{class:!0});var Le=z(O);f(je.$$.fragment,Le),io=r(Le),ot=p(Le,"P",{"data-svelte-h":!0}),M(ot)!=="svelte-1tattgh"&&(ot.textContent=Oo),Le.forEach(s),lo=r(J),K=p(J,"DIV",{class:!0});var Ee=z(K);f(Ue.$$.fragment,Ee),co=r(Ee),nt=p(Ee,"P",{"data-svelte-h":!0}),M(nt)!=="svelte-zt8o99"&&(nt.textContent=Ko),Ee.forEach(s),J.forEach(s),Zt=r(e),f(xe.$$.fragment,e),Ft=r(e),U=p(e,"DIV",{class:!0});var P=z(U);f(Ce.$$.fragment,P),po=r(P),st=p(P,"P",{"data-svelte-h":!0}),M(st)!=="svelte-11n2akr"&&(st.innerHTML=en),uo=r(P),at=p(P,"P",{"data-svelte-h":!0}),M(at)!=="svelte-gxzj9w"&&(at.innerHTML=tn),mo=r(P),H=p(P,"DIV",{class:!0});var R=z(H);f(Ie.$$.fragment,R),ho=r(R),rt=p(R,"P",{"data-svelte-h":!0}),M(rt)!=="svelte-ptaowd"&&(rt.textContent=on),go=r(R),it=p(R,"UL",{"data-svelte-h":!0}),M(it)!=="svelte-1xl4xih"&&(it.innerHTML=nn),R.forEach(s),fo=r(P),ee=p(P,"DIV",{class:!0});var Re=z(ee);f(Ge.$$.fragment,Re),_o=r(Re),dt=p(Re,"P",{"data-svelte-h":!0}),M(dt)!=="svelte-1tattgh"&&(dt.textContent=sn),Re.forEach(s),P.forEach(s),Wt=r(e),f(Pe.$$.fragment,e),qt=r(e),x=p(e,"DIV",{class:!0});var X=z(x);f(Ze.$$.fragment,X),bo=r(X),lt=p(X,"P",{"data-svelte-h":!0}),M(lt)!=="svelte-ijuydm"&&(lt.textContent=an),yo=r(X),ct=p(X,"P",{"data-svelte-h":!0}),M(ct)!=="svelte-q52n56"&&(ct.innerHTML=rn),ko=r(X),pt=p(X,"P",{"data-svelte-h":!0}),M(pt)!=="svelte-hswkmf"&&(pt.innerHTML=dn),Mo=r(X),F=p(X,"DIV",{class:!0});var ie=z(F);f(Fe.$$.fragment,ie),To=r(ie),ut=p(ie,"P",{"data-svelte-h":!0}),M(ut)!=="svelte-1exb53t"&&(ut.innerHTML=ln),vo=r(ie),f(te.$$.fragment,ie),wo=r(ie),f(oe.$$.fragment,ie),ie.forEach(s),X.forEach(s),Bt=r(e),f(We.$$.fragment,e),St=r(e),C=p(e,"DIV",{class:!0});var L=z(C);f(qe.$$.fragment,L),$o=r(L),mt=p(L,"P",{"data-svelte-h":!0}),M(mt)!=="svelte-1rlkdfa"&&(mt.textContent=cn),Jo=r(L),ht=p(L,"P",{"data-svelte-h":!0}),M(ht)!=="svelte-q52n56"&&(ht.innerHTML=pn),zo=r(L),gt=p(L,"P",{"data-svelte-h":!0}),M(gt)!=="svelte-hswkmf"&&(gt.innerHTML=un),jo=r(L),W=p(L,"DIV",{class:!0});var de=z(W);f(Be.$$.fragment,de),Uo=r(de),ft=p(de,"P",{"data-svelte-h":!0}),M(ft)!=="svelte-1kiqr5z"&&(ft.innerHTML=mn),xo=r(de),f(ne.$$.fragment,de),Co=r(de),f(se.$$.fragment,de),de.forEach(s),L.forEach(s),Vt=r(e),f(Se.$$.fragment,e),Nt=r(e),E=p(e,"DIV",{class:!0});var Et=z(E);f(Ve.$$.fragment,Et),Io=r(Et),q=p(Et,"DIV",{class:!0});var le=z(q);f(Ne.$$.fragment,le),Go=r(le),_t=p(le,"P",{"data-svelte-h":!0}),M(_t)!=="svelte-1mx3i51"&&(_t.innerHTML=hn),Po=r(le),f(ae.$$.fragment,le),Zo=r(le),f(re.$$.fragment,le),le.forEach(s),Et.forEach(s),Ht=r(e),f(He.$$.fragment,e),Xt=r(e),yt=p(e,"P",{}),z(yt).forEach(s),this.h()},h(){j(t,"name","hf:doc:metadata"),j(t,"content",Fn),kn(w,"float","right"),j(G,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),j(Z,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),j(D,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),j(O,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),j(K,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),j($,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),j(H,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),j(ee,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),j(U,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),j(F,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),j(x,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),j(W,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),j(C,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),j(q,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),j(E,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,n){i(document.head,t),l(e,h,n),l(e,o,n),l(e,u,n),l(e,T,n),l(e,m,n),l(e,w,n),l(e,ue,n),_(N,e,n),l(e,kt,n),l(e,me,n),l(e,Mt,n),l(e,he,n),l(e,Tt,n),_(A,e,n),l(e,vt,n),l(e,ge,n),l(e,wt,n),_(Q,e,n),l(e,$t,n),l(e,fe,n),l(e,Jt,n),l(e,_e,n),l(e,zt,n),_(be,e,n),l(e,jt,n),_(ye,e,n),l(e,Ut,n),l(e,ke,n),l(e,xt,n),_(Me,e,n),l(e,Ct,n),l(e,G,n),_(Te,G,null),i(G,Qt),i(G,Ae),i(G,Yt),i(G,Qe),i(G,Dt),_(Y,G,null),l(e,It,n),_(ve,e,n),l(e,Gt,n),l(e,we,n),l(e,Pt,n),l(e,$,n),_($e,$,null),i($,Ot),i($,Ye),i($,Kt),i($,De),i($,eo),i($,Z),_(Je,Z,null),i(Z,to),i(Z,Oe),i(Z,oo),i(Z,Ke),i(Z,no),i(Z,et),i($,so),i($,D),_(ze,D,null),i(D,ao),i(D,tt),i($,ro),i($,O),_(je,O,null),i(O,io),i(O,ot),i($,lo),i($,K),_(Ue,K,null),i(K,co),i(K,nt),l(e,Zt,n),_(xe,e,n),l(e,Ft,n),l(e,U,n),_(Ce,U,null),i(U,po),i(U,st),i(U,uo),i(U,at),i(U,mo),i(U,H),_(Ie,H,null),i(H,ho),i(H,rt),i(H,go),i(H,it),i(U,fo),i(U,ee),_(Ge,ee,null),i(ee,_o),i(ee,dt),l(e,Wt,n),_(Pe,e,n),l(e,qt,n),l(e,x,n),_(Ze,x,null),i(x,bo),i(x,lt),i(x,yo),i(x,ct),i(x,ko),i(x,pt),i(x,Mo),i(x,F),_(Fe,F,null),i(F,To),i(F,ut),i(F,vo),_(te,F,null),i(F,wo),_(oe,F,null),l(e,Bt,n),_(We,e,n),l(e,St,n),l(e,C,n),_(qe,C,null),i(C,$o),i(C,mt),i(C,Jo),i(C,ht),i(C,zo),i(C,gt),i(C,jo),i(C,W),_(Be,W,null),i(W,Uo),i(W,ft),i(W,xo),_(ne,W,null),i(W,Co),_(se,W,null),l(e,Vt,n),_(Se,e,n),l(e,Nt,n),l(e,E,n),_(Ve,E,null),i(E,Io),i(E,q),_(Ne,q,null),i(q,Go),i(q,_t),i(q,Po),_(ae,q,null),i(q,Zo),_(re,q,null),l(e,Ht,n),_(He,e,n),l(e,Xt,n),l(e,yt,n),Lt=!0},p(e,[n]){const B={};n&2&&(B.$$scope={dirty:n,ctx:e}),A.$set(B);const J={};n&2&&(J.$$scope={dirty:n,ctx:e}),Q.$set(J);const S={};n&2&&(S.$$scope={dirty:n,ctx:e}),Y.$set(S);const Xe={};n&2&&(Xe.$$scope={dirty:n,ctx:e}),te.$set(Xe);const Le={};n&2&&(Le.$$scope={dirty:n,ctx:e}),oe.$set(Le);const Ee={};n&2&&(Ee.$$scope={dirty:n,ctx:e}),ne.$set(Ee);const P={};n&2&&(P.$$scope={dirty:n,ctx:e}),se.$set(P);const R={};n&2&&(R.$$scope={dirty:n,ctx:e}),ae.$set(R);const Re={};n&2&&(Re.$$scope={dirty:n,ctx:e}),re.$set(Re)},i(e){Lt||(b(N.$$.fragment,e),b(A.$$.fragment,e),b(Q.$$.fragment,e),b(be.$$.fragment,e),b(ye.$$.fragment,e),b(Me.$$.fragment,e),b(Te.$$.fragment,e),b(Y.$$.fragment,e),b(ve.$$.fragment,e),b($e.$$.fragment,e),b(Je.$$.fragment,e),b(ze.$$.fragment,e),b(je.$$.fragment,e),b(Ue.$$.fragment,e),b(xe.$$.fragment,e),b(Ce.$$.fragment,e),b(Ie.$$.fragment,e),b(Ge.$$.fragment,e),b(Pe.$$.fragment,e),b(Ze.$$.fragment,e),b(Fe.$$.fragment,e),b(te.$$.fragment,e),b(oe.$$.fragment,e),b(We.$$.fragment,e),b(qe.$$.fragment,e),b(Be.$$.fragment,e),b(ne.$$.fragment,e),b(se.$$.fragment,e),b(Se.$$.fragment,e),b(Ve.$$.fragment,e),b(Ne.$$.fragment,e),b(ae.$$.fragment,e),b(re.$$.fragment,e),b(He.$$.fragment,e),Lt=!0)},o(e){y(N.$$.fragment,e),y(A.$$.fragment,e),y(Q.$$.fragment,e),y(be.$$.fragment,e),y(ye.$$.fragment,e),y(Me.$$.fragment,e),y(Te.$$.fragment,e),y(Y.$$.fragment,e),y(ve.$$.fragment,e),y($e.$$.fragment,e),y(Je.$$.fragment,e),y(ze.$$.fragment,e),y(je.$$.fragment,e),y(Ue.$$.fragment,e),y(xe.$$.fragment,e),y(Ce.$$.fragment,e),y(Ie.$$.fragment,e),y(Ge.$$.fragment,e),y(Pe.$$.fragment,e),y(Ze.$$.fragment,e),y(Fe.$$.fragment,e),y(te.$$.fragment,e),y(oe.$$.fragment,e),y(We.$$.fragment,e),y(qe.$$.fragment,e),y(Be.$$.fragment,e),y(ne.$$.fragment,e),y(se.$$.fragment,e),y(Se.$$.fragment,e),y(Ve.$$.fragment,e),y(Ne.$$.fragment,e),y(ae.$$.fragment,e),y(re.$$.fragment,e),y(He.$$.fragment,e),Lt=!1},d(e){e&&(s(h),s(o),s(u),s(T),s(m),s(w),s(ue),s(kt),s(me),s(Mt),s(he),s(Tt),s(vt),s(ge),s(wt),s($t),s(fe),s(Jt),s(_e),s(zt),s(jt),s(Ut),s(ke),s(xt),s(Ct),s(G),s(It),s(Gt),s(we),s(Pt),s($),s(Zt),s(Ft),s(U),s(Wt),s(qt),s(x),s(Bt),s(St),s(C),s(Vt),s(Nt),s(E),s(Ht),s(Xt),s(yt)),s(t),k(N,e),k(A,e),k(Q,e),k(be,e),k(ye,e),k(Me,e),k(Te),k(Y),k(ve,e),k($e),k(Je),k(ze),k(je),k(Ue),k(xe,e),k(Ce),k(Ie),k(Ge),k(Pe,e),k(Ze),k(Fe),k(te),k(oe),k(We,e),k(qe),k(Be),k(ne),k(se),k(Se,e),k(Ve),k(Ne),k(ae),k(re),k(He,e)}}}const Fn='{"title":"Pegasus","local":"pegasus","sections":[{"title":"Notes","local":"notes","sections":[],"depth":2},{"title":"PegasusConfig","local":"transformers.PegasusConfig","sections":[],"depth":2},{"title":"PegasusTokenizer","local":"transformers.PegasusTokenizer","sections":[],"depth":2},{"title":"PegasusTokenizerFast","local":"transformers.PegasusTokenizerFast","sections":[],"depth":2},{"title":"PegasusModel","local":"transformers.PegasusModel","sections":[],"depth":2},{"title":"PegasusForConditionalGeneration","local":"transformers.PegasusForConditionalGeneration","sections":[],"depth":2},{"title":"PegasusForCausalLM","local":"transformers.PegasusForCausalLM","sections":[],"depth":2}],"depth":1}';function Wn(v){return fn(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class En extends _n{constructor(t){super(),bn(this,t,Wn,Zn,gn,{})}}export{En as component};
