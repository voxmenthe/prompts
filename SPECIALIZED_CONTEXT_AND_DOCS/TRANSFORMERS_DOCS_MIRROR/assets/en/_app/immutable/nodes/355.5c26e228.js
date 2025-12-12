import{s as go,o as _o,n as W}from"../chunks/scheduler.18a86fab.js";import{S as wo,i as bo,g as c,s as a,r as h,m as ho,A as yo,h as p,f as s,c as r,j as x,x as T,u as f,n as fo,k as C,l as To,y as l,a as i,v as g,d as _,t as w,w as b}from"../chunks/index.98837b22.js";import{T as wt}from"../chunks/Tip.77304350.js";import{D as z}from"../chunks/Docstring.a1ef7999.js";import{C as de}from"../chunks/CodeBlock.8d0c2e8a.js";import{E as nn}from"../chunks/ExampleCodeBlock.8c3ee1f9.js";import{H as L,E as ko}from"../chunks/getInferenceSnippets.06c2775f.js";import{H as Mo,a as Nn}from"../chunks/HfOption.6641485e.js";function vo(M){let t,d="Click on the Qwen2 models in the right sidebar for more examples of how to apply Qwen2 to different language tasks.";return{c(){t=c("p"),t.textContent=d},l(n){t=p(n,"P",{"data-svelte-h":!0}),T(t)!=="svelte-1vpo7y9"&&(t.textContent=d)},m(n,m){i(n,t,m)},p:W,d(n){n&&s(t)}}}function $o(M){let t,d;return t=new de({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwcGlwZWxpbmUlMEElMEFwaXBlJTIwJTNEJTIwcGlwZWxpbmUoJTBBJTIwJTIwJTIwJTIwdGFzayUzRCUyMnRleHQtZ2VuZXJhdGlvbiUyMiUyQyUwQSUyMCUyMCUyMCUyMG1vZGVsJTNEJTIyUXdlbiUyRlF3ZW4yLTEuNUItSW5zdHJ1Y3QlMjIlMkMlMEElMjAlMjAlMjAlMjBkdHlwZSUzRHRvcmNoLmJmbG9hdDE2JTJDJTBBJTIwJTIwJTIwJTIwZGV2aWNlX21hcCUzRDAlMEEpJTBBJTBBbWVzc2FnZXMlMjAlM0QlMjAlNUIlMEElMjAlMjAlMjAlMjAlN0IlMjJyb2xlJTIyJTNBJTIwJTIyc3lzdGVtJTIyJTJDJTIwJTIyY29udGVudCUyMiUzQSUyMCUyMllvdSUyMGFyZSUyMGElMjBoZWxwZnVsJTIwYXNzaXN0YW50LiUyMiU3RCUyQyUwQSUyMCUyMCUyMCUyMCU3QiUyMnJvbGUlMjIlM0ElMjAlMjJ1c2VyJTIyJTJDJTIwJTIyY29udGVudCUyMiUzQSUyMCUyMlRlbGwlMjBtZSUyMGFib3V0JTIwdGhlJTIwUXdlbjIlMjBtb2RlbCUyMGZhbWlseS4lMjIlN0QlMkMlMEElNUQlMEFvdXRwdXRzJTIwJTNEJTIwcGlwZShtZXNzYWdlcyUyQyUyMG1heF9uZXdfdG9rZW5zJTNEMjU2JTJDJTIwZG9fc2FtcGxlJTNEVHJ1ZSUyQyUyMHRlbXBlcmF0dXJlJTNEMC43JTJDJTIwdG9wX2slM0Q1MCUyQyUyMHRvcF9wJTNEMC45NSklMEFwcmludChvdXRwdXRzJTVCMCU1RCU1QiUyMmdlbmVyYXRlZF90ZXh0JTIyJTVEJTVCLTElNUQlNUInY29udGVudCclNUQp",highlighted:`<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> pipeline

pipe = pipeline(
    task=<span class="hljs-string">&quot;text-generation&quot;</span>,
    model=<span class="hljs-string">&quot;Qwen/Qwen2-1.5B-Instruct&quot;</span>,
    dtype=torch.bfloat16,
    device_map=<span class="hljs-number">0</span>
)

messages = [
    {<span class="hljs-string">&quot;role&quot;</span>: <span class="hljs-string">&quot;system&quot;</span>, <span class="hljs-string">&quot;content&quot;</span>: <span class="hljs-string">&quot;You are a helpful assistant.&quot;</span>},
    {<span class="hljs-string">&quot;role&quot;</span>: <span class="hljs-string">&quot;user&quot;</span>, <span class="hljs-string">&quot;content&quot;</span>: <span class="hljs-string">&quot;Tell me about the Qwen2 model family.&quot;</span>},
]
outputs = pipe(messages, max_new_tokens=<span class="hljs-number">256</span>, do_sample=<span class="hljs-literal">True</span>, temperature=<span class="hljs-number">0.7</span>, top_k=<span class="hljs-number">50</span>, top_p=<span class="hljs-number">0.95</span>)
<span class="hljs-built_in">print</span>(outputs[<span class="hljs-number">0</span>][<span class="hljs-string">&quot;generated_text&quot;</span>][-<span class="hljs-number">1</span>][<span class="hljs-string">&#x27;content&#x27;</span>])`,wrap:!1}}),{c(){h(t.$$.fragment)},l(n){f(t.$$.fragment,n)},m(n,m){g(t,n,m),d=!0},p:W,i(n){d||(_(t.$$.fragment,n),d=!0)},o(n){w(t.$$.fragment,n),d=!1},d(n){b(t,n)}}}function xo(M){let t,d;return t=new de({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b01vZGVsRm9yQ2F1c2FsTE0lMkMlMjBBdXRvVG9rZW5pemVyJTBBJTBBbW9kZWwlMjAlM0QlMjBBdXRvTW9kZWxGb3JDYXVzYWxMTS5mcm9tX3ByZXRyYWluZWQoJTBBJTIwJTIwJTIwJTIwJTIyUXdlbiUyRlF3ZW4yLTEuNUItSW5zdHJ1Y3QlMjIlMkMlMEElMjAlMjAlMjAlMjBkdHlwZSUzRHRvcmNoLmJmbG9hdDE2JTJDJTBBJTIwJTIwJTIwJTIwZGV2aWNlX21hcCUzRCUyMmF1dG8lMjIlMkMlMEElMjAlMjAlMjAlMjBhdHRuX2ltcGxlbWVudGF0aW9uJTNEJTIyc2RwYSUyMiUwQSklMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJRd2VuJTJGUXdlbjItMS41Qi1JbnN0cnVjdCUyMiklMEElMEFwcm9tcHQlMjAlM0QlMjAlMjJHaXZlJTIwbWUlMjBhJTIwc2hvcnQlMjBpbnRyb2R1Y3Rpb24lMjB0byUyMGxhcmdlJTIwbGFuZ3VhZ2UlMjBtb2RlbHMuJTIyJTBBbWVzc2FnZXMlMjAlM0QlMjAlNUIlMEElMjAlMjAlMjAlMjAlN0IlMjJyb2xlJTIyJTNBJTIwJTIyc3lzdGVtJTIyJTJDJTIwJTIyY29udGVudCUyMiUzQSUyMCUyMllvdSUyMGFyZSUyMGElMjBoZWxwZnVsJTIwYXNzaXN0YW50LiUyMiU3RCUyQyUwQSUyMCUyMCUyMCUyMCU3QiUyMnJvbGUlMjIlM0ElMjAlMjJ1c2VyJTIyJTJDJTIwJTIyY29udGVudCUyMiUzQSUyMHByb21wdCU3RCUwQSU1RCUwQXRleHQlMjAlM0QlMjB0b2tlbml6ZXIuYXBwbHlfY2hhdF90ZW1wbGF0ZSglMEElMjAlMjAlMjAlMjBtZXNzYWdlcyUyQyUwQSUyMCUyMCUyMCUyMHRva2VuaXplJTNERmFsc2UlMkMlMEElMjAlMjAlMjAlMjBhZGRfZ2VuZXJhdGlvbl9wcm9tcHQlM0RUcnVlJTBBKSUwQW1vZGVsX2lucHV0cyUyMCUzRCUyMHRva2VuaXplciglNUJ0ZXh0JTVEJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMikudG8obW9kZWwuZGV2aWNlKSUwQSUwQWdlbmVyYXRlZF9pZHMlMjAlM0QlMjBtb2RlbC5nZW5lcmF0ZSglMEElMjAlMjAlMjAlMjBtb2RlbF9pbnB1dHMuaW5wdXRfaWRzJTJDJTBBJTIwJTIwJTIwJTIwY2FjaGVfaW1wbGVtZW50YXRpb24lM0QlMjJzdGF0aWMlMjIlMkMlMEElMjAlMjAlMjAlMjBtYXhfbmV3X3Rva2VucyUzRDUxMiUyQyUwQSUyMCUyMCUyMCUyMGRvX3NhbXBsZSUzRFRydWUlMkMlMEElMjAlMjAlMjAlMjB0ZW1wZXJhdHVyZSUzRDAuNyUyQyUwQSUyMCUyMCUyMCUyMHRvcF9rJTNENTAlMkMlMEElMjAlMjAlMjAlMjB0b3BfcCUzRDAuOTUlMEEpJTBBZ2VuZXJhdGVkX2lkcyUyMCUzRCUyMCU1QiUwQSUyMCUyMCUyMCUyMG91dHB1dF9pZHMlNUJsZW4oaW5wdXRfaWRzKSUzQSU1RCUyMGZvciUyMGlucHV0X2lkcyUyQyUyMG91dHB1dF9pZHMlMjBpbiUyMHppcChtb2RlbF9pbnB1dHMuaW5wdXRfaWRzJTJDJTIwZ2VuZXJhdGVkX2lkcyklMEElNUQlMEElMEFyZXNwb25zZSUyMCUzRCUyMHRva2VuaXplci5iYXRjaF9kZWNvZGUoZ2VuZXJhdGVkX2lkcyUyQyUyMHNraXBfc3BlY2lhbF90b2tlbnMlM0RUcnVlKSU1QjAlNUQlMEFwcmludChyZXNwb25zZSk=",highlighted:`<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    <span class="hljs-string">&quot;Qwen/Qwen2-1.5B-Instruct&quot;</span>,
    dtype=torch.bfloat16,
    device_map=<span class="hljs-string">&quot;auto&quot;</span>,
    attn_implementation=<span class="hljs-string">&quot;sdpa&quot;</span>
)
tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;Qwen/Qwen2-1.5B-Instruct&quot;</span>)

prompt = <span class="hljs-string">&quot;Give me a short introduction to large language models.&quot;</span>
messages = [
    {<span class="hljs-string">&quot;role&quot;</span>: <span class="hljs-string">&quot;system&quot;</span>, <span class="hljs-string">&quot;content&quot;</span>: <span class="hljs-string">&quot;You are a helpful assistant.&quot;</span>},
    {<span class="hljs-string">&quot;role&quot;</span>: <span class="hljs-string">&quot;user&quot;</span>, <span class="hljs-string">&quot;content&quot;</span>: prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=<span class="hljs-literal">False</span>,
    add_generation_prompt=<span class="hljs-literal">True</span>
)
model_inputs = tokenizer([text], return_tensors=<span class="hljs-string">&quot;pt&quot;</span>).to(model.device)

generated_ids = model.generate(
    model_inputs.input_ids,
    cache_implementation=<span class="hljs-string">&quot;static&quot;</span>,
    max_new_tokens=<span class="hljs-number">512</span>,
    do_sample=<span class="hljs-literal">True</span>,
    temperature=<span class="hljs-number">0.7</span>,
    top_k=<span class="hljs-number">50</span>,
    top_p=<span class="hljs-number">0.95</span>
)
generated_ids = [
    output_ids[<span class="hljs-built_in">len</span>(input_ids):] <span class="hljs-keyword">for</span> input_ids, output_ids <span class="hljs-keyword">in</span> <span class="hljs-built_in">zip</span>(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=<span class="hljs-literal">True</span>)[<span class="hljs-number">0</span>]
<span class="hljs-built_in">print</span>(response)`,wrap:!1}}),{c(){h(t.$$.fragment)},l(n){f(t.$$.fragment,n)},m(n,m){g(t,n,m),d=!0},p:W,i(n){d||(_(t.$$.fragment,n),d=!0)},o(n){w(t.$$.fragment,n),d=!1},d(n){b(t,n)}}}function Co(M){let t,d;return t=new de({props:{code:"JTIzJTIwcGlwJTIwaW5zdGFsbCUyMC1VJTIwZmxhc2gtYXR0biUyMC0tbm8tYnVpbGQtaXNvbGF0aW9uJTBBdHJhbnNmb3JtZXJzJTIwY2hhdCUyMFF3ZW4lMkZRd2VuMi03Qi1JbnN0cnVjdCUyMC0tZHR5cGUlMjBhdXRvJTIwLS1hdHRuX2ltcGxlbWVudGF0aW9uJTIwZmxhc2hfYXR0ZW50aW9uXzIlMjAtLWRldmljZSUyMDA=",highlighted:`<span class="hljs-comment"># pip install -U flash-attn --no-build-isolation</span>
transformers chat Qwen/Qwen2-7B-Instruct --dtype auto --attn_implementation flash_attention_2 --device 0`,wrap:!1}}),{c(){h(t.$$.fragment)},l(n){f(t.$$.fragment,n)},m(n,m){g(t,n,m),d=!0},p:W,i(n){d||(_(t.$$.fragment,n),d=!0)},o(n){w(t.$$.fragment,n),d=!1},d(n){b(t,n)}}}function Qo(M){let t,d,n,m,k,u;return t=new Nn({props:{id:"usage",option:"Pipeline",$$slots:{default:[$o]},$$scope:{ctx:M}}}),n=new Nn({props:{id:"usage",option:"AutoModel",$$slots:{default:[xo]},$$scope:{ctx:M}}}),k=new Nn({props:{id:"usage",option:"transformers CLI",$$slots:{default:[Co]},$$scope:{ctx:M}}}),{c(){h(t.$$.fragment),d=a(),h(n.$$.fragment),m=a(),h(k.$$.fragment)},l(y){f(t.$$.fragment,y),d=r(y),f(n.$$.fragment,y),m=r(y),f(k.$$.fragment,y)},m(y,v){g(t,y,v),i(y,d,v),g(n,y,v),i(y,m,v),g(k,y,v),u=!0},p(y,v){const bt={};v&2&&(bt.$$scope={dirty:v,ctx:y}),t.$set(bt);const ce={};v&2&&(ce.$$scope={dirty:v,ctx:y}),n.$set(ce);const A={};v&2&&(A.$$scope={dirty:v,ctx:y}),k.$set(A)},i(y){u||(_(t.$$.fragment,y),_(n.$$.fragment,y),_(k.$$.fragment,y),u=!0)},o(y){w(t.$$.fragment,y),w(n.$$.fragment,y),w(k.$$.fragment,y),u=!1},d(y){y&&(s(d),s(m)),b(t,y),b(n,y),b(k,y)}}}function zo(M){let t,d;return t=new de({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMFF3ZW4yTW9kZWwlMkMlMjBRd2VuMkNvbmZpZyUwQSUwQSUyMyUyMEluaXRpYWxpemluZyUyMGElMjBRd2VuMiUyMHN0eWxlJTIwY29uZmlndXJhdGlvbiUwQWNvbmZpZ3VyYXRpb24lMjAlM0QlMjBRd2VuMkNvbmZpZygpJTBBJTBBJTIzJTIwSW5pdGlhbGl6aW5nJTIwYSUyMG1vZGVsJTIwZnJvbSUyMHRoZSUyMFF3ZW4yLTdCJTIwc3R5bGUlMjBjb25maWd1cmF0aW9uJTBBbW9kZWwlMjAlM0QlMjBRd2VuMk1vZGVsKGNvbmZpZ3VyYXRpb24pJTBBJTBBJTIzJTIwQWNjZXNzaW5nJTIwdGhlJTIwbW9kZWwlMjBjb25maWd1cmF0aW9uJTBBY29uZmlndXJhdGlvbiUyMCUzRCUyMG1vZGVsLmNvbmZpZw==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> Qwen2Model, Qwen2Config

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a Qwen2 style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = Qwen2Config()

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a model from the Qwen2-7B style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = Qwen2Model(configuration)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Accessing the model configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = model.config`,wrap:!1}}),{c(){h(t.$$.fragment)},l(n){f(t.$$.fragment,n)},m(n,m){g(t,n,m),d=!0},p:W,i(n){d||(_(t.$$.fragment,n),d=!0)},o(n){w(t.$$.fragment,n),d=!1},d(n){b(t,n)}}}function Uo(M){let t,d="be encoded differently whether it is at the beginning of the sentence (without space) or not:",n,m,k;return m=new de({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMFF3ZW4yVG9rZW5pemVyJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwUXdlbjJUb2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMlF3ZW4lMkZRd2VuLXRva2VuaXplciUyMiklMEF0b2tlbml6ZXIoJTIySGVsbG8lMjB3b3JsZCUyMiklNUIlMjJpbnB1dF9pZHMlMjIlNUQlMEElMEF0b2tlbml6ZXIoJTIyJTIwSGVsbG8lMjB3b3JsZCUyMiklNUIlMjJpbnB1dF9pZHMlMjIlNUQ=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> Qwen2Tokenizer

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = Qwen2Tokenizer.from_pretrained(<span class="hljs-string">&quot;Qwen/Qwen-tokenizer&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer(<span class="hljs-string">&quot;Hello world&quot;</span>)[<span class="hljs-string">&quot;input_ids&quot;</span>]
[<span class="hljs-number">9707</span>, <span class="hljs-number">1879</span>]

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer(<span class="hljs-string">&quot; Hello world&quot;</span>)[<span class="hljs-string">&quot;input_ids&quot;</span>]
[<span class="hljs-number">21927</span>, <span class="hljs-number">1879</span>]`,wrap:!1}}),{c(){t=c("p"),t.textContent=d,n=a(),h(m.$$.fragment)},l(u){t=p(u,"P",{"data-svelte-h":!0}),T(t)!=="svelte-12atnao"&&(t.textContent=d),n=r(u),f(m.$$.fragment,u)},m(u,y){i(u,t,y),i(u,n,y),g(m,u,y),k=!0},p:W,i(u){k||(_(m.$$.fragment,u),k=!0)},o(u){w(m.$$.fragment,u),k=!1},d(u){u&&(s(t),s(n)),b(m,u)}}}function jo(M){let t,d="be encoded differently whether it is at the beginning of the sentence (without space) or not:",n,m,k;return m=new de({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMFF3ZW4yVG9rZW5pemVyRmFzdCUwQSUwQXRva2VuaXplciUyMCUzRCUyMFF3ZW4yVG9rZW5pemVyRmFzdC5mcm9tX3ByZXRyYWluZWQoJTIyUXdlbiUyRlF3ZW4tdG9rZW5pemVyJTIyKSUwQXRva2VuaXplciglMjJIZWxsbyUyMHdvcmxkJTIyKSU1QiUyMmlucHV0X2lkcyUyMiU1RCUwQSUwQXRva2VuaXplciglMjIlMjBIZWxsbyUyMHdvcmxkJTIyKSU1QiUyMmlucHV0X2lkcyUyMiU1RA==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> Qwen2TokenizerFast

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = Qwen2TokenizerFast.from_pretrained(<span class="hljs-string">&quot;Qwen/Qwen-tokenizer&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer(<span class="hljs-string">&quot;Hello world&quot;</span>)[<span class="hljs-string">&quot;input_ids&quot;</span>]
[<span class="hljs-number">9707</span>, <span class="hljs-number">1879</span>]

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer(<span class="hljs-string">&quot; Hello world&quot;</span>)[<span class="hljs-string">&quot;input_ids&quot;</span>]
[<span class="hljs-number">21927</span>, <span class="hljs-number">1879</span>]`,wrap:!1}}),{c(){t=c("p"),t.textContent=d,n=a(),h(m.$$.fragment)},l(u){t=p(u,"P",{"data-svelte-h":!0}),T(t)!=="svelte-12atnao"&&(t.textContent=d),n=r(u),f(m.$$.fragment,u)},m(u,y){i(u,t,y),i(u,n,y),g(m,u,y),k=!0},p:W,i(u){k||(_(m.$$.fragment,u),k=!0)},o(u){w(m.$$.fragment,u),k=!1},d(u){u&&(s(t),s(n)),b(m,u)}}}function Jo(M){let t,d=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=d},l(n){t=p(n,"P",{"data-svelte-h":!0}),T(t)!=="svelte-fincs2"&&(t.innerHTML=d)},m(n,m){i(n,t,m)},p:W,d(n){n&&s(t)}}}function qo(M){let t,d=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=d},l(n){t=p(n,"P",{"data-svelte-h":!0}),T(t)!=="svelte-fincs2"&&(t.innerHTML=d)},m(n,m){i(n,t,m)},p:W,d(n){n&&s(t)}}}function Fo(M){let t,d="Example:",n,m,k;return m=new de({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBRd2VuMkZvckNhdXNhbExNJTBBJTBBbW9kZWwlMjAlM0QlMjBRd2VuMkZvckNhdXNhbExNLmZyb21fcHJldHJhaW5lZCglMjJtZXRhLXF3ZW4yJTJGUXdlbjItMi03Yi1oZiUyMiklMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJtZXRhLXF3ZW4yJTJGUXdlbjItMi03Yi1oZiUyMiklMEElMEFwcm9tcHQlMjAlM0QlMjAlMjJIZXklMkMlMjBhcmUlMjB5b3UlMjBjb25zY2lvdXMlM0YlMjBDYW4lMjB5b3UlMjB0YWxrJTIwdG8lMjBtZSUzRiUyMiUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplcihwcm9tcHQlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQSUwQSUyMyUyMEdlbmVyYXRlJTBBZ2VuZXJhdGVfaWRzJTIwJTNEJTIwbW9kZWwuZ2VuZXJhdGUoaW5wdXRzLmlucHV0X2lkcyUyQyUyMG1heF9sZW5ndGglM0QzMCklMEF0b2tlbml6ZXIuYmF0Y2hfZGVjb2RlKGdlbmVyYXRlX2lkcyUyQyUyMHNraXBfc3BlY2lhbF90b2tlbnMlM0RUcnVlJTJDJTIwY2xlYW5fdXBfdG9rZW5pemF0aW9uX3NwYWNlcyUzREZhbHNlKSU1QjAlNUQ=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, Qwen2ForCausalLM

<span class="hljs-meta">&gt;&gt;&gt; </span>model = Qwen2ForCausalLM.from_pretrained(<span class="hljs-string">&quot;meta-qwen2/Qwen2-2-7b-hf&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;meta-qwen2/Qwen2-2-7b-hf&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>prompt = <span class="hljs-string">&quot;Hey, are you conscious? Can you talk to me?&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(prompt, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Generate</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>generate_ids = model.generate(inputs.input_ids, max_length=<span class="hljs-number">30</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer.batch_decode(generate_ids, skip_special_tokens=<span class="hljs-literal">True</span>, clean_up_tokenization_spaces=<span class="hljs-literal">False</span>)[<span class="hljs-number">0</span>]
<span class="hljs-string">&quot;Hey, are you conscious? Can you talk to me?\\nI&#x27;m not conscious, but I can talk to you.&quot;</span>`,wrap:!1}}),{c(){t=c("p"),t.textContent=d,n=a(),h(m.$$.fragment)},l(u){t=p(u,"P",{"data-svelte-h":!0}),T(t)!=="svelte-11lpom8"&&(t.textContent=d),n=r(u),f(m.$$.fragment,u)},m(u,y){i(u,t,y),i(u,n,y),g(m,u,y),k=!0},p:W,i(u){k||(_(m.$$.fragment,u),k=!0)},o(u){w(m.$$.fragment,u),k=!1},d(u){u&&(s(t),s(n)),b(m,u)}}}function Io(M){let t,d=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=d},l(n){t=p(n,"P",{"data-svelte-h":!0}),T(t)!=="svelte-fincs2"&&(t.innerHTML=d)},m(n,m){i(n,t,m)},p:W,d(n){n&&s(t)}}}function Zo(M){let t,d=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=d},l(n){t=p(n,"P",{"data-svelte-h":!0}),T(t)!=="svelte-fincs2"&&(t.innerHTML=d)},m(n,m){i(n,t,m)},p:W,d(n){n&&s(t)}}}function Wo(M){let t,d=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=d},l(n){t=p(n,"P",{"data-svelte-h":!0}),T(t)!=="svelte-fincs2"&&(t.innerHTML=d)},m(n,m){i(n,t,m)},p:W,d(n){n&&s(t)}}}function Bo(M){let t,d,n,m,k,u="<em>This model was released on 2024-07-15 and added to Hugging Face Transformers on 2024-01-17.</em>",y,v,bt='<div class="flex flex-wrap space-x-1"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/> <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat"/> <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/> <img alt="Tensor parallelism" src="https://img.shields.io/badge/Tensor%20parallelism-06b6d4?style=flat&amp;logoColor=white"/></div>',ce,A,Mt,pe,Ln='<a href="https://huggingface.co/papers/2407.10671" rel="nofollow">Qwen2</a> is a family of large language models (pretrained, instruction-tuned and mixture-of-experts) available in sizes from 0.5B to 72B parameters. The models are built on the Transformer architecture featuring enhancements like group query attention (GQA), rotary positional embeddings (RoPE), a mix of sliding window and full attention, and dual chunk attention with YARN for training stability. Qwen2 models support multiple languages and context lengths up to 131,072 tokens.',vt,me,An='You can find all the official Qwen2 checkpoints under the <a href="https://huggingface.co/collections/Qwen/qwen2-6659360b33528ced941e557f" rel="nofollow">Qwen2</a> collection.',$t,Y,xt,ue,Xn='The example below demonstrates how to generate text with <a href="/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.Pipeline">Pipeline</a>, <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel">AutoModel</a>, and from the command line using the instruction-tuned models.',Ct,K,Qt,he,Vn='Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the <a href="../quantization/overview">Quantization</a> overview for more available quantization backends.',zt,fe,En='The example below uses <a href="../quantization/bitsandbytes">bitsandbytes</a> to quantize the weights to 4-bits.',Ut,ge,jt,_e,Jt,we,Gn="<li>Ensure your Transformers library version is up-to-date. Qwen2 requires Transformers&gt;=4.37.0 for full support.</li>",qt,be,Ft,q,ye,on,He,Hn=`This is the configuration class to store the configuration of a <a href="/docs/transformers/v4.56.2/en/model_doc/qwen2#transformers.Qwen2Model">Qwen2Model</a>. It is used to instantiate a
Qwen2 model according to the specified arguments, defining the model architecture. Instantiating a configuration
with the defaults will yield a similar configuration to that of
Qwen2-7B-beta <a href="https://huggingface.co/Qwen/Qwen2-7B-beta" rel="nofollow">Qwen/Qwen2-7B-beta</a>.`,sn,Pe,Pn=`Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> and can be used to control the model outputs. Read the
documentation from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> for more information.`,an,ee,It,Te,Zt,$,ke,rn,Se,Sn="Construct a Qwen2 tokenizer. Based on byte-level Byte-Pair-Encoding.",ln,De,Dn="Same with GPT2Tokenizer, this tokenizer has been trained to treat spaces like parts of the tokens so a word will",dn,te,cn,Oe,On="You should not use GPT2Tokenizer instead, because of the different pretokenization rules.",pn,Ye,Yn=`This tokenizer inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer">PreTrainedTokenizer</a> which contains most of the main methods. Users should refer to
this superclass for more information regarding those methods.`,mn,Ke,Me,Wt,ve,Bt,U,$e,un,et,Kn=`Construct a “fast” Qwen2 tokenizer (backed by HuggingFace’s <em>tokenizers</em> library). Based on byte-level
Byte-Pair-Encoding.`,hn,tt,eo="Same with GPT2Tokenizer, this tokenizer has been trained to treat spaces like parts of the tokens so a word will",fn,ne,gn,nt,to=`This tokenizer inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast">PreTrainedTokenizerFast</a> which contains most of the main methods. Users should
refer to this superclass for more information regarding those methods.`,Rt,xe,Nt,H,Ce,_n,ot,Qe,Lt,ze,At,j,Ue,wn,st,no="The bare Qwen2 Model outputting raw hidden-states without any specific head on top.",bn,at,oo=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,yn,rt,so=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Tn,X,je,kn,it,ao='The <a href="/docs/transformers/v4.56.2/en/model_doc/qwen2#transformers.Qwen2Model">Qwen2Model</a> forward method, overrides the <code>__call__</code> special method.',Mn,oe,Xt,Je,Vt,J,qe,vn,lt,ro="The Qwen2 Model for causal language modeling.",$n,dt,io=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,xn,ct,lo=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Cn,B,Fe,Qn,pt,co='The <a href="/docs/transformers/v4.56.2/en/model_doc/qwen2#transformers.Qwen2ForCausalLM">Qwen2ForCausalLM</a> forward method, overrides the <code>__call__</code> special method.',zn,se,Un,ae,Et,Ie,Gt,P,Ze,jn,V,We,Jn,mt,po="The <code>GenericForSequenceClassification</code> forward method, overrides the <code>__call__</code> special method.",qn,re,Ht,Be,Pt,S,Re,Fn,E,Ne,In,ut,mo="The <code>GenericForTokenClassification</code> forward method, overrides the <code>__call__</code> special method.",Zn,ie,St,Le,Dt,D,Ae,Wn,G,Xe,Bn,ht,uo="The <code>GenericForQuestionAnswering</code> forward method, overrides the <code>__call__</code> special method.",Rn,le,Ot,Ve,Yt,yt,Kt;return A=new L({props:{title:"Qwen2",local:"qwen2",headingTag:"h1"}}),Y=new wt({props:{warning:!1,$$slots:{default:[vo]},$$scope:{ctx:M}}}),K=new Mo({props:{id:"usage",options:["Pipeline","AutoModel","transformers CLI"],$$slots:{default:[Qo]},$$scope:{ctx:M}}}),ge=new de({props:{code:"JTIzJTIwcGlwJTIwaW5zdGFsbCUyMC1VJTIwZmxhc2gtYXR0biUyMC0tbm8tYnVpbGQtaXNvbGF0aW9uJTBBaW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Rva2VuaXplciUyQyUyMEF1dG9Nb2RlbEZvckNhdXNhbExNJTJDJTIwQml0c0FuZEJ5dGVzQ29uZmlnJTBBJTBBcXVhbnRpemF0aW9uX2NvbmZpZyUyMCUzRCUyMEJpdHNBbmRCeXRlc0NvbmZpZyglMEElMjAlMjAlMjAlMjBsb2FkX2luXzRiaXQlM0RUcnVlJTJDJTBBJTIwJTIwJTIwJTIwYm5iXzRiaXRfY29tcHV0ZV9kdHlwZSUzRHRvcmNoLmJmbG9hdDE2JTJDJTBBJTIwJTIwJTIwJTIwYm5iXzRiaXRfcXVhbnRfdHlwZSUzRCUyMm5mNCUyMiUyQyUwQSUyMCUyMCUyMCUyMGJuYl80Yml0X3VzZV9kb3VibGVfcXVhbnQlM0RUcnVlJTJDJTBBKSUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMlF3ZW4lMkZRd2VuMi03QiUyMiklMEFtb2RlbCUyMCUzRCUyMEF1dG9Nb2RlbEZvckNhdXNhbExNLmZyb21fcHJldHJhaW5lZCglMEElMjAlMjAlMjAlMjAlMjJRd2VuJTJGUXdlbjItN0IlMjIlMkMlMEElMjAlMjAlMjAlMjBkdHlwZSUzRHRvcmNoLmJmbG9hdDE2JTJDJTBBJTIwJTIwJTIwJTIwZGV2aWNlX21hcCUzRCUyMmF1dG8lMjIlMkMlMEElMjAlMjAlMjAlMjBxdWFudGl6YXRpb25fY29uZmlnJTNEcXVhbnRpemF0aW9uX2NvbmZpZyUyQyUwQSUyMCUyMCUyMCUyMGF0dG5faW1wbGVtZW50YXRpb24lM0QlMjJmbGFzaF9hdHRlbnRpb25fMiUyMiUwQSklMEElMEFpbnB1dHMlMjAlM0QlMjB0b2tlbml6ZXIoJTIyVGhlJTIwUXdlbjIlMjBtb2RlbCUyMGZhbWlseSUyMGlzJTIyJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMikudG8obW9kZWwuZGV2aWNlKSUwQW91dHB1dHMlMjAlM0QlMjBtb2RlbC5nZW5lcmF0ZSgqKmlucHV0cyUyQyUyMG1heF9uZXdfdG9rZW5zJTNEMTAwKSUwQXByaW50KHRva2VuaXplci5kZWNvZGUob3V0cHV0cyU1QjAlNUQlMkMlMjBza2lwX3NwZWNpYWxfdG9rZW5zJTNEVHJ1ZSkp",highlighted:`<span class="hljs-comment"># pip install -U flash-attn --no-build-isolation</span>
<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=<span class="hljs-literal">True</span>,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type=<span class="hljs-string">&quot;nf4&quot;</span>,
    bnb_4bit_use_double_quant=<span class="hljs-literal">True</span>,
)

tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;Qwen/Qwen2-7B&quot;</span>)
model = AutoModelForCausalLM.from_pretrained(
    <span class="hljs-string">&quot;Qwen/Qwen2-7B&quot;</span>,
    dtype=torch.bfloat16,
    device_map=<span class="hljs-string">&quot;auto&quot;</span>,
    quantization_config=quantization_config,
    attn_implementation=<span class="hljs-string">&quot;flash_attention_2&quot;</span>
)

inputs = tokenizer(<span class="hljs-string">&quot;The Qwen2 model family is&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>).to(model.device)
outputs = model.generate(**inputs, max_new_tokens=<span class="hljs-number">100</span>)
<span class="hljs-built_in">print</span>(tokenizer.decode(outputs[<span class="hljs-number">0</span>], skip_special_tokens=<span class="hljs-literal">True</span>))`,wrap:!1}}),_e=new L({props:{title:"Notes",local:"notes",headingTag:"h2"}}),be=new L({props:{title:"Qwen2Config",local:"transformers.Qwen2Config",headingTag:"h2"}}),ye=new z({props:{name:"class transformers.Qwen2Config",anchor:"transformers.Qwen2Config",parameters:[{name:"vocab_size",val:" = 151936"},{name:"hidden_size",val:" = 4096"},{name:"intermediate_size",val:" = 22016"},{name:"num_hidden_layers",val:" = 32"},{name:"num_attention_heads",val:" = 32"},{name:"num_key_value_heads",val:" = 32"},{name:"hidden_act",val:" = 'silu'"},{name:"max_position_embeddings",val:" = 32768"},{name:"initializer_range",val:" = 0.02"},{name:"rms_norm_eps",val:" = 1e-06"},{name:"use_cache",val:" = True"},{name:"tie_word_embeddings",val:" = False"},{name:"rope_theta",val:" = 10000.0"},{name:"rope_scaling",val:" = None"},{name:"use_sliding_window",val:" = False"},{name:"sliding_window",val:" = 4096"},{name:"max_window_layers",val:" = 28"},{name:"layer_types",val:" = None"},{name:"attention_dropout",val:" = 0.0"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.Qwen2Config.vocab_size",description:`<strong>vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 151936) &#x2014;
Vocabulary size of the Qwen2 model. Defines the number of different tokens that can be represented by the
<code>inputs_ids</code> passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/qwen2#transformers.Qwen2Model">Qwen2Model</a>`,name:"vocab_size"},{anchor:"transformers.Qwen2Config.hidden_size",description:`<strong>hidden_size</strong> (<code>int</code>, <em>optional</em>, defaults to 4096) &#x2014;
Dimension of the hidden representations.`,name:"hidden_size"},{anchor:"transformers.Qwen2Config.intermediate_size",description:`<strong>intermediate_size</strong> (<code>int</code>, <em>optional</em>, defaults to 22016) &#x2014;
Dimension of the MLP representations.`,name:"intermediate_size"},{anchor:"transformers.Qwen2Config.num_hidden_layers",description:`<strong>num_hidden_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 32) &#x2014;
Number of hidden layers in the Transformer encoder.`,name:"num_hidden_layers"},{anchor:"transformers.Qwen2Config.num_attention_heads",description:`<strong>num_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 32) &#x2014;
Number of attention heads for each attention layer in the Transformer encoder.`,name:"num_attention_heads"},{anchor:"transformers.Qwen2Config.num_key_value_heads",description:`<strong>num_key_value_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 32) &#x2014;
This is the number of key_value heads that should be used to implement Grouped Query Attention. If
<code>num_key_value_heads=num_attention_heads</code>, the model will use Multi Head Attention (MHA), if
<code>num_key_value_heads=1</code> the model will use Multi Query Attention (MQA) otherwise GQA is used. When
converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
by meanpooling all the original heads within that group. For more details, check out <a href="https://huggingface.co/papers/2305.13245" rel="nofollow">this
paper</a>. If it is not specified, will default to <code>32</code>.`,name:"num_key_value_heads"},{anchor:"transformers.Qwen2Config.hidden_act",description:`<strong>hidden_act</strong> (<code>str</code> or <code>function</code>, <em>optional</em>, defaults to <code>&quot;silu&quot;</code>) &#x2014;
The non-linear activation function (function or string) in the decoder.`,name:"hidden_act"},{anchor:"transformers.Qwen2Config.max_position_embeddings",description:`<strong>max_position_embeddings</strong> (<code>int</code>, <em>optional</em>, defaults to 32768) &#x2014;
The maximum sequence length that this model might ever be used with.`,name:"max_position_embeddings"},{anchor:"transformers.Qwen2Config.initializer_range",description:`<strong>initializer_range</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.`,name:"initializer_range"},{anchor:"transformers.Qwen2Config.rms_norm_eps",description:`<strong>rms_norm_eps</strong> (<code>float</code>, <em>optional</em>, defaults to 1e-06) &#x2014;
The epsilon used by the rms normalization layers.`,name:"rms_norm_eps"},{anchor:"transformers.Qwen2Config.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not the model should return the last key/values attentions (not used by all models). Only
relevant if <code>config.is_decoder=True</code>.`,name:"use_cache"},{anchor:"transformers.Qwen2Config.tie_word_embeddings",description:`<strong>tie_word_embeddings</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether the model&#x2019;s input and output word embeddings should be tied.`,name:"tie_word_embeddings"},{anchor:"transformers.Qwen2Config.rope_theta",description:`<strong>rope_theta</strong> (<code>float</code>, <em>optional</em>, defaults to 10000.0) &#x2014;
The base period of the RoPE embeddings.`,name:"rope_theta"},{anchor:"transformers.Qwen2Config.rope_scaling",description:`<strong>rope_scaling</strong> (<code>Dict</code>, <em>optional</em>) &#x2014;
Dictionary containing the scaling configuration for the RoPE embeddings. NOTE: if you apply new rope type
and you expect the model to work on longer <code>max_position_embeddings</code>, we recommend you to update this value
accordingly.
Expected contents:
<code>rope_type</code> (<code>str</code>):
The sub-variant of RoPE to use. Can be one of [&#x2018;default&#x2019;, &#x2018;linear&#x2019;, &#x2018;dynamic&#x2019;, &#x2018;yarn&#x2019;, &#x2018;longrope&#x2019;,
&#x2018;llama3&#x2019;], with &#x2018;default&#x2019; being the original RoPE implementation.
<code>factor</code> (<code>float</code>, <em>optional</em>):
Used with all rope types except &#x2018;default&#x2019;. The scaling factor to apply to the RoPE embeddings. In
most scaling types, a <code>factor</code> of x will enable the model to handle sequences of length x <em>
original maximum pre-trained length.
<code>original_max_position_embeddings</code> (<code>int</code>, </em>optional<em>):
Used with &#x2018;dynamic&#x2019;, &#x2018;longrope&#x2019; and &#x2018;llama3&#x2019;. The original max position embeddings used during
pretraining.
<code>attention_factor</code> (<code>float</code>, </em>optional<em>):
Used with &#x2018;yarn&#x2019; and &#x2018;longrope&#x2019;. The scaling factor to be applied on the attention
computation. If unspecified, it defaults to value recommended by the implementation, using the
<code>factor</code> field to infer the suggested value.
<code>beta_fast</code> (<code>float</code>, </em>optional<em>):
Only used with &#x2018;yarn&#x2019;. Parameter to set the boundary for extrapolation (only) in the linear
ramp function. If unspecified, it defaults to 32.
<code>beta_slow</code> (<code>float</code>, </em>optional<em>):
Only used with &#x2018;yarn&#x2019;. Parameter to set the boundary for interpolation (only) in the linear
ramp function. If unspecified, it defaults to 1.
<code>short_factor</code> (<code>list[float]</code>, </em>optional<em>):
Only used with &#x2018;longrope&#x2019;. The scaling factor to be applied to short contexts (&lt;
<code>original_max_position_embeddings</code>). Must be a list of numbers with the same length as the hidden
size divided by the number of attention heads divided by 2
<code>long_factor</code> (<code>list[float]</code>, </em>optional<em>):
Only used with &#x2018;longrope&#x2019;. The scaling factor to be applied to long contexts (&lt;
<code>original_max_position_embeddings</code>). Must be a list of numbers with the same length as the hidden
size divided by the number of attention heads divided by 2
<code>low_freq_factor</code> (<code>float</code>, </em>optional<em>):
Only used with &#x2018;llama3&#x2019;. Scaling factor applied to low frequency components of the RoPE
<code>high_freq_factor</code> (<code>float</code>, </em>optional*):
Only used with &#x2018;llama3&#x2019;. Scaling factor applied to high frequency components of the RoPE`,name:"rope_scaling"},{anchor:"transformers.Qwen2Config.use_sliding_window",description:`<strong>use_sliding_window</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether to use sliding window attention.`,name:"use_sliding_window"},{anchor:"transformers.Qwen2Config.sliding_window",description:`<strong>sliding_window</strong> (<code>int</code>, <em>optional</em>, defaults to 4096) &#x2014;
Sliding window attention (SWA) window size. If not specified, will default to <code>4096</code>.`,name:"sliding_window"},{anchor:"transformers.Qwen2Config.max_window_layers",description:`<strong>max_window_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 28) &#x2014;
The number of layers using full attention. The first <code>max_window_layers</code> layers will use full attention, while any
additional layer afterwards will use SWA (Sliding Window Attention).`,name:"max_window_layers"},{anchor:"transformers.Qwen2Config.layer_types",description:`<strong>layer_types</strong> (<code>list</code>, <em>optional</em>) &#x2014;
Attention pattern for each layer.`,name:"layer_types"},{anchor:"transformers.Qwen2Config.attention_dropout",description:`<strong>attention_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The dropout ratio for the attention probabilities.`,name:"attention_dropout"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/qwen2/configuration_qwen2.py#L25"}}),ee=new nn({props:{anchor:"transformers.Qwen2Config.example",$$slots:{default:[zo]},$$scope:{ctx:M}}}),Te=new L({props:{title:"Qwen2Tokenizer",local:"transformers.Qwen2Tokenizer",headingTag:"h2"}}),ke=new z({props:{name:"class transformers.Qwen2Tokenizer",anchor:"transformers.Qwen2Tokenizer",parameters:[{name:"vocab_file",val:""},{name:"merges_file",val:""},{name:"errors",val:" = 'replace'"},{name:"unk_token",val:" = '<|endoftext|>'"},{name:"bos_token",val:" = None"},{name:"eos_token",val:" = '<|endoftext|>'"},{name:"pad_token",val:" = '<|endoftext|>'"},{name:"clean_up_tokenization_spaces",val:" = False"},{name:"split_special_tokens",val:" = False"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.Qwen2Tokenizer.vocab_file",description:`<strong>vocab_file</strong> (<code>str</code>) &#x2014;
Path to the vocabulary file.`,name:"vocab_file"},{anchor:"transformers.Qwen2Tokenizer.merges_file",description:`<strong>merges_file</strong> (<code>str</code>) &#x2014;
Path to the merges file.`,name:"merges_file"},{anchor:"transformers.Qwen2Tokenizer.errors",description:`<strong>errors</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;replace&quot;</code>) &#x2014;
Paradigm to follow when decoding bytes to UTF-8. See
<a href="https://docs.python.org/3/library/stdtypes.html#bytes.decode" rel="nofollow">bytes.decode</a> for more information.`,name:"errors"},{anchor:"transformers.Qwen2Tokenizer.unk_token",description:`<strong>unk_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;|endoftext|&gt;&quot;</code>) &#x2014;
The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
token instead.`,name:"unk_token"},{anchor:"transformers.Qwen2Tokenizer.bos_token",description:`<strong>bos_token</strong> (<code>str</code>, <em>optional</em>) &#x2014;
The beginning of sequence token. Not applicable for this tokenizer.`,name:"bos_token"},{anchor:"transformers.Qwen2Tokenizer.eos_token",description:`<strong>eos_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;|endoftext|&gt;&quot;</code>) &#x2014;
The end of sequence token.`,name:"eos_token"},{anchor:"transformers.Qwen2Tokenizer.pad_token",description:`<strong>pad_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;|endoftext|&gt;&quot;</code>) &#x2014;
The token used for padding, for example when batching sequences of different lengths.`,name:"pad_token"},{anchor:"transformers.Qwen2Tokenizer.clean_up_tokenization_spaces",description:`<strong>clean_up_tokenization_spaces</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not the model should cleanup the spaces that were added when splitting the input text during the
tokenization process. Not applicable to this tokenizer, since tokenization does not add spaces.`,name:"clean_up_tokenization_spaces"},{anchor:"transformers.Qwen2Tokenizer.split_special_tokens",description:`<strong>split_special_tokens</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not the special tokens should be split during the tokenization process. The default behavior is
to not split special tokens. This means that if <code>&lt;|endoftext|&gt;</code> is the <code>eos_token</code>, then <code>tokenizer.tokenize(&quot;&lt;|endoftext|&gt;&quot;) = [&apos;&lt;|endoftext|&gt;</code>]. Otherwise, if <code>split_special_tokens=True</code>, then <code>tokenizer.tokenize(&quot;&lt;|endoftext|&gt;&quot;)</code> will be give <code>[&apos;&lt;&apos;, &apos;|&apos;, &apos;endo&apos;, &apos;ft&apos;, &apos;ext&apos;, &apos;|&apos;, &apos;&gt;&apos;]</code>. This argument is only supported for <code>slow</code> tokenizers for the moment.`,name:"split_special_tokens"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/qwen2/tokenization_qwen2.py#L83"}}),te=new nn({props:{anchor:"transformers.Qwen2Tokenizer.example",$$slots:{default:[Uo]},$$scope:{ctx:M}}}),Me=new z({props:{name:"save_vocabulary",anchor:"transformers.Qwen2Tokenizer.save_vocabulary",parameters:[{name:"save_directory",val:": str"},{name:"filename_prefix",val:": typing.Optional[str] = None"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/qwen2/tokenization_qwen2.py#L308"}}),ve=new L({props:{title:"Qwen2TokenizerFast",local:"transformers.Qwen2TokenizerFast",headingTag:"h2"}}),$e=new z({props:{name:"class transformers.Qwen2TokenizerFast",anchor:"transformers.Qwen2TokenizerFast",parameters:[{name:"vocab_file",val:" = None"},{name:"merges_file",val:" = None"},{name:"tokenizer_file",val:" = None"},{name:"unk_token",val:" = '<|endoftext|>'"},{name:"bos_token",val:" = None"},{name:"eos_token",val:" = '<|endoftext|>'"},{name:"pad_token",val:" = '<|endoftext|>'"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.Qwen2TokenizerFast.vocab_file",description:`<strong>vocab_file</strong> (<code>str</code>, <em>optional</em>) &#x2014;
Path to the vocabulary file.`,name:"vocab_file"},{anchor:"transformers.Qwen2TokenizerFast.merges_file",description:`<strong>merges_file</strong> (<code>str</code>, <em>optional</em>) &#x2014;
Path to the merges file.`,name:"merges_file"},{anchor:"transformers.Qwen2TokenizerFast.tokenizer_file",description:`<strong>tokenizer_file</strong> (<code>str</code>, <em>optional</em>) &#x2014;
Path to <a href="https://github.com/huggingface/tokenizers" rel="nofollow">tokenizers</a> file (generally has a .json extension) that
contains everything needed to load the tokenizer.`,name:"tokenizer_file"},{anchor:"transformers.Qwen2TokenizerFast.unk_token",description:`<strong>unk_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;|endoftext|&gt;&quot;</code>) &#x2014;
The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
token instead. Not applicable to this tokenizer.`,name:"unk_token"},{anchor:"transformers.Qwen2TokenizerFast.bos_token",description:`<strong>bos_token</strong> (<code>str</code>, <em>optional</em>) &#x2014;
The beginning of sequence token. Not applicable for this tokenizer.`,name:"bos_token"},{anchor:"transformers.Qwen2TokenizerFast.eos_token",description:`<strong>eos_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;|endoftext|&gt;&quot;</code>) &#x2014;
The end of sequence token.`,name:"eos_token"},{anchor:"transformers.Qwen2TokenizerFast.pad_token",description:`<strong>pad_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;|endoftext|&gt;&quot;</code>) &#x2014;
The token used for padding, for example when batching sequences of different lengths.`,name:"pad_token"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/qwen2/tokenization_qwen2_fast.py#L37"}}),ne=new nn({props:{anchor:"transformers.Qwen2TokenizerFast.example",$$slots:{default:[jo]},$$scope:{ctx:M}}}),xe=new L({props:{title:"Qwen2RMSNorm",local:"transformers.Qwen2RMSNorm",headingTag:"h2"}}),Ce=new z({props:{name:"class transformers.Qwen2RMSNorm",anchor:"transformers.Qwen2RMSNorm",parameters:[{name:"hidden_size",val:""},{name:"eps",val:": float = 1e-06"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/qwen2/modeling_qwen2.py#L187"}}),Qe=new z({props:{name:"forward",anchor:"transformers.Qwen2RMSNorm.forward",parameters:[{name:"hidden_states",val:": Tensor"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/qwen2/modeling_qwen2.py#L196"}}),ze=new L({props:{title:"Qwen2Model",local:"transformers.Qwen2Model",headingTag:"h2"}}),Ue=new z({props:{name:"class transformers.Qwen2Model",anchor:"transformers.Qwen2Model",parameters:[{name:"config",val:": Qwen2Config"}],parametersDescription:[{anchor:"transformers.Qwen2Model.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/qwen2#transformers.Qwen2Config">Qwen2Config</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/qwen2/modeling_qwen2.py#L310"}}),je=new z({props:{name:"forward",anchor:"transformers.Qwen2Model.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.Qwen2Model.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.Qwen2Model.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.Qwen2Model.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.Qwen2Model.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.Qwen2Model.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.Qwen2Model.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.Qwen2Model.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/qwen2/modeling_qwen2.py#L328",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast"
>transformers.modeling_outputs.BaseModelOutputWithPast</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/qwen2#transformers.Qwen2Config"
>Qwen2Config</a>) and inputs.</p>
<ul>
<li>
<p><strong>last_hidden_state</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>) — Sequence of hidden-states at the output of the last layer of the model.</p>
<p>If <code>past_key_values</code> is used only the last hidden-state of the sequences of shape <code>(batch_size, 1, hidden_size)</code> is output.</p>
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
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast"
>transformers.modeling_outputs.BaseModelOutputWithPast</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),oe=new wt({props:{$$slots:{default:[Jo]},$$scope:{ctx:M}}}),Je=new L({props:{title:"Qwen2ForCausalLM",local:"transformers.Qwen2ForCausalLM",headingTag:"h2"}}),qe=new z({props:{name:"class transformers.Qwen2ForCausalLM",anchor:"transformers.Qwen2ForCausalLM",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.Qwen2ForCausalLM.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/qwen2#transformers.Qwen2ForCausalLM">Qwen2ForCausalLM</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/qwen2/modeling_qwen2.py#L403"}}),Fe=new z({props:{name:"forward",anchor:"transformers.Qwen2ForCausalLM.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"},{name:"logits_to_keep",val:": typing.Union[int, torch.Tensor] = 0"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.Qwen2ForCausalLM.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.Qwen2ForCausalLM.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.Qwen2ForCausalLM.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.Qwen2ForCausalLM.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.Qwen2ForCausalLM.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.Qwen2ForCausalLM.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.Qwen2ForCausalLM.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.Qwen2ForCausalLM.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"},{anchor:"transformers.Qwen2ForCausalLM.forward.logits_to_keep",description:`<strong>logits_to_keep</strong> (<code>Union[int, torch.Tensor]</code>, defaults to <code>0</code>) &#x2014;
If an <code>int</code>, compute logits for the last <code>logits_to_keep</code> tokens. If <code>0</code>, calculate logits for all
<code>input_ids</code> (special case). Only last token logits are needed for generation, and calculating them only for that
token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
If a <code>torch.Tensor</code>, must be 1D corresponding to the indices to keep in the sequence length dimension.
This is useful when using packed tensor format (single dimension for batch and sequence length).`,name:"logits_to_keep"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/qwen2/modeling_qwen2.py#L417",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast"
>transformers.modeling_outputs.CausalLMOutputWithPast</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/qwen2#transformers.Qwen2Config"
>Qwen2Config</a>) and inputs.</p>
<ul>
<li>
<p><strong>loss</strong> (<code>torch.FloatTensor</code> of shape <code>(1,)</code>, <em>optional</em>, returned when <code>labels</code> is provided) — Language modeling loss (for next-token prediction).</p>
</li>
<li>
<p><strong>logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, config.vocab_size)</code>) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).</p>
</li>
<li>
<p><strong>past_key_values</strong> (<code>Cache</code>, <em>optional</em>, returned when <code>use_cache=True</code> is passed or when <code>config.use_cache=True</code>) — It is a <a
  href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache"
>Cache</a> instance. For more details, see our <a
  href="https://huggingface.co/docs/transformers/en/kv_cache"
  rel="nofollow"
>kv cache guide</a>.</p>
<p>Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
<code>past_key_values</code> input) to speed up sequential decoding.</p>
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
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast"
>transformers.modeling_outputs.CausalLMOutputWithPast</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),se=new wt({props:{$$slots:{default:[qo]},$$scope:{ctx:M}}}),ae=new nn({props:{anchor:"transformers.Qwen2ForCausalLM.forward.example",$$slots:{default:[Fo]},$$scope:{ctx:M}}}),Ie=new L({props:{title:"Qwen2ForSequenceClassification",local:"transformers.Qwen2ForSequenceClassification",headingTag:"h2"}}),Ze=new z({props:{name:"class transformers.Qwen2ForSequenceClassification",anchor:"transformers.Qwen2ForSequenceClassification",parameters:[{name:"config",val:""}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/qwen2/modeling_qwen2.py#L478"}}),We=new z({props:{name:"forward",anchor:"transformers.Qwen2ForSequenceClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.Qwen2ForSequenceClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.Qwen2ForSequenceClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.Qwen2ForSequenceClassification.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.Qwen2ForSequenceClassification.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.Qwen2ForSequenceClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.Qwen2ForSequenceClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.Qwen2ForSequenceClassification.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/modeling_layers.py#L111",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <code>transformers.modeling_outputs.SequenceClassifierOutputWithPast</code> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<code>None</code>) and inputs.</p>
<ul>
<li>
<p><strong>loss</strong> (<code>torch.FloatTensor</code> of shape <code>(1,)</code>, <em>optional</em>, returned when <code>labels</code> is provided) — Classification (or regression if config.num_labels==1) loss.</p>
</li>
<li>
<p><strong>logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, config.num_labels)</code>) — Classification (or regression if config.num_labels==1) scores (before SoftMax).</p>
</li>
<li>
<p><strong>past_key_values</strong> (<code>Cache</code>, <em>optional</em>, returned when <code>use_cache=True</code> is passed or when <code>config.use_cache=True</code>) — It is a <a
  href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache"
>Cache</a> instance. For more details, see our <a
  href="https://huggingface.co/docs/transformers/en/kv_cache"
  rel="nofollow"
>kv cache guide</a>.</p>
<p>Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
<code>past_key_values</code> input) to speed up sequential decoding.</p>
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


<p><code>transformers.modeling_outputs.SequenceClassifierOutputWithPast</code> or <code>tuple(torch.FloatTensor)</code></p>
`}}),re=new wt({props:{$$slots:{default:[Io]},$$scope:{ctx:M}}}),Be=new L({props:{title:"Qwen2ForTokenClassification",local:"transformers.Qwen2ForTokenClassification",headingTag:"h2"}}),Re=new z({props:{name:"class transformers.Qwen2ForTokenClassification",anchor:"transformers.Qwen2ForTokenClassification",parameters:[{name:"config",val:""}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/qwen2/modeling_qwen2.py#L482"}}),Ne=new z({props:{name:"forward",anchor:"transformers.Qwen2ForTokenClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.Qwen2ForTokenClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.Qwen2ForTokenClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.Qwen2ForTokenClassification.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.Qwen2ForTokenClassification.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.Qwen2ForTokenClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.Qwen2ForTokenClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.Qwen2ForTokenClassification.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/modeling_layers.py#L254",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput"
>transformers.modeling_outputs.TokenClassifierOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<code>None</code>) and inputs.</p>
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
`}}),ie=new wt({props:{$$slots:{default:[Zo]},$$scope:{ctx:M}}}),Le=new L({props:{title:"Qwen2ForQuestionAnswering",local:"transformers.Qwen2ForQuestionAnswering",headingTag:"h2"}}),Ae=new z({props:{name:"class transformers.Qwen2ForQuestionAnswering",anchor:"transformers.Qwen2ForQuestionAnswering",parameters:[{name:"config",val:""}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/qwen2/modeling_qwen2.py#L486"}}),Xe=new z({props:{name:"forward",anchor:"transformers.Qwen2ForQuestionAnswering.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"start_positions",val:": typing.Optional[torch.LongTensor] = None"},{name:"end_positions",val:": typing.Optional[torch.LongTensor] = None"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.Qwen2ForQuestionAnswering.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.Qwen2ForQuestionAnswering.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.Qwen2ForQuestionAnswering.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.Qwen2ForQuestionAnswering.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.Qwen2ForQuestionAnswering.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.Qwen2ForQuestionAnswering.forward.start_positions",description:`<strong>start_positions</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for position (index) of the start of the labelled span for computing the token classification loss.
Positions are clamped to the length of the sequence (<code>sequence_length</code>). Position outside of the sequence
are not taken into account for computing the loss.`,name:"start_positions"},{anchor:"transformers.Qwen2ForQuestionAnswering.forward.end_positions",description:`<strong>end_positions</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for position (index) of the end of the labelled span for computing the token classification loss.
Positions are clamped to the length of the sequence (<code>sequence_length</code>). Position outside of the sequence
are not taken into account for computing the loss.`,name:"end_positions"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/modeling_layers.py#L191",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.QuestionAnsweringModelOutput"
>transformers.modeling_outputs.QuestionAnsweringModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<code>None</code>) and inputs.</p>
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
`}}),le=new wt({props:{$$slots:{default:[Wo]},$$scope:{ctx:M}}}),Ve=new ko({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/qwen2.md"}}),{c(){t=c("meta"),d=a(),n=c("p"),m=a(),k=c("p"),k.innerHTML=u,y=a(),v=c("div"),v.innerHTML=bt,ce=a(),h(A.$$.fragment),Mt=a(),pe=c("p"),pe.innerHTML=Ln,vt=a(),me=c("p"),me.innerHTML=An,$t=a(),h(Y.$$.fragment),xt=a(),ue=c("p"),ue.innerHTML=Xn,Ct=a(),h(K.$$.fragment),Qt=a(),he=c("p"),he.innerHTML=Vn,zt=a(),fe=c("p"),fe.innerHTML=En,Ut=a(),h(ge.$$.fragment),jt=a(),h(_e.$$.fragment),Jt=a(),we=c("ul"),we.innerHTML=Gn,qt=a(),h(be.$$.fragment),Ft=a(),q=c("div"),h(ye.$$.fragment),on=a(),He=c("p"),He.innerHTML=Hn,sn=a(),Pe=c("p"),Pe.innerHTML=Pn,an=a(),h(ee.$$.fragment),It=a(),h(Te.$$.fragment),Zt=a(),$=c("div"),h(ke.$$.fragment),rn=a(),Se=c("p"),Se.textContent=Sn,ln=a(),De=c("p"),De.textContent=Dn,dn=a(),h(te.$$.fragment),cn=ho(`
This is expected.
`),Oe=c("p"),Oe.textContent=On,pn=a(),Ye=c("p"),Ye.innerHTML=Yn,mn=a(),Ke=c("div"),h(Me.$$.fragment),Wt=a(),h(ve.$$.fragment),Bt=a(),U=c("div"),h($e.$$.fragment),un=a(),et=c("p"),et.innerHTML=Kn,hn=a(),tt=c("p"),tt.textContent=eo,fn=a(),h(ne.$$.fragment),gn=ho(`
This is expected.
`),nt=c("p"),nt.innerHTML=to,Rt=a(),h(xe.$$.fragment),Nt=a(),H=c("div"),h(Ce.$$.fragment),_n=a(),ot=c("div"),h(Qe.$$.fragment),Lt=a(),h(ze.$$.fragment),At=a(),j=c("div"),h(Ue.$$.fragment),wn=a(),st=c("p"),st.textContent=no,bn=a(),at=c("p"),at.innerHTML=oo,yn=a(),rt=c("p"),rt.innerHTML=so,Tn=a(),X=c("div"),h(je.$$.fragment),kn=a(),it=c("p"),it.innerHTML=ao,Mn=a(),h(oe.$$.fragment),Xt=a(),h(Je.$$.fragment),Vt=a(),J=c("div"),h(qe.$$.fragment),vn=a(),lt=c("p"),lt.textContent=ro,$n=a(),dt=c("p"),dt.innerHTML=io,xn=a(),ct=c("p"),ct.innerHTML=lo,Cn=a(),B=c("div"),h(Fe.$$.fragment),Qn=a(),pt=c("p"),pt.innerHTML=co,zn=a(),h(se.$$.fragment),Un=a(),h(ae.$$.fragment),Et=a(),h(Ie.$$.fragment),Gt=a(),P=c("div"),h(Ze.$$.fragment),jn=a(),V=c("div"),h(We.$$.fragment),Jn=a(),mt=c("p"),mt.innerHTML=po,qn=a(),h(re.$$.fragment),Ht=a(),h(Be.$$.fragment),Pt=a(),S=c("div"),h(Re.$$.fragment),Fn=a(),E=c("div"),h(Ne.$$.fragment),In=a(),ut=c("p"),ut.innerHTML=mo,Zn=a(),h(ie.$$.fragment),St=a(),h(Le.$$.fragment),Dt=a(),D=c("div"),h(Ae.$$.fragment),Wn=a(),G=c("div"),h(Xe.$$.fragment),Bn=a(),ht=c("p"),ht.innerHTML=uo,Rn=a(),h(le.$$.fragment),Ot=a(),h(Ve.$$.fragment),Yt=a(),yt=c("p"),this.h()},l(e){const o=yo("svelte-u9bgzb",document.head);t=p(o,"META",{name:!0,content:!0}),o.forEach(s),d=r(e),n=p(e,"P",{}),x(n).forEach(s),m=r(e),k=p(e,"P",{"data-svelte-h":!0}),T(k)!=="svelte-1jbfciw"&&(k.innerHTML=u),y=r(e),v=p(e,"DIV",{style:!0,"data-svelte-h":!0}),T(v)!=="svelte-11gpmgv"&&(v.innerHTML=bt),ce=r(e),f(A.$$.fragment,e),Mt=r(e),pe=p(e,"P",{"data-svelte-h":!0}),T(pe)!=="svelte-1liz77w"&&(pe.innerHTML=Ln),vt=r(e),me=p(e,"P",{"data-svelte-h":!0}),T(me)!=="svelte-jnpfni"&&(me.innerHTML=An),$t=r(e),f(Y.$$.fragment,e),xt=r(e),ue=p(e,"P",{"data-svelte-h":!0}),T(ue)!=="svelte-1v4bhcb"&&(ue.innerHTML=Xn),Ct=r(e),f(K.$$.fragment,e),Qt=r(e),he=p(e,"P",{"data-svelte-h":!0}),T(he)!=="svelte-nf5ooi"&&(he.innerHTML=Vn),zt=r(e),fe=p(e,"P",{"data-svelte-h":!0}),T(fe)!=="svelte-1ca5nhg"&&(fe.innerHTML=En),Ut=r(e),f(ge.$$.fragment,e),jt=r(e),f(_e.$$.fragment,e),Jt=r(e),we=p(e,"UL",{"data-svelte-h":!0}),T(we)!=="svelte-1bli25w"&&(we.innerHTML=Gn),qt=r(e),f(be.$$.fragment,e),Ft=r(e),q=p(e,"DIV",{class:!0});var R=x(q);f(ye.$$.fragment,R),on=r(R),He=p(R,"P",{"data-svelte-h":!0}),T(He)!=="svelte-17rjjut"&&(He.innerHTML=Hn),sn=r(R),Pe=p(R,"P",{"data-svelte-h":!0}),T(Pe)!=="svelte-1ek1ss9"&&(Pe.innerHTML=Pn),an=r(R),f(ee.$$.fragment,R),R.forEach(s),It=r(e),f(Te.$$.fragment,e),Zt=r(e),$=p(e,"DIV",{class:!0});var Q=x($);f(ke.$$.fragment,Q),rn=r(Q),Se=p(Q,"P",{"data-svelte-h":!0}),T(Se)!=="svelte-owj012"&&(Se.textContent=Sn),ln=r(Q),De=p(Q,"P",{"data-svelte-h":!0}),T(De)!=="svelte-ei9kk8"&&(De.textContent=Dn),dn=r(Q),f(te.$$.fragment,Q),cn=fo(Q,`
This is expected.
`),Oe=p(Q,"P",{"data-svelte-h":!0}),T(Oe)!=="svelte-187588w"&&(Oe.textContent=On),pn=r(Q),Ye=p(Q,"P",{"data-svelte-h":!0}),T(Ye)!=="svelte-ntrhio"&&(Ye.innerHTML=Yn),mn=r(Q),Ke=p(Q,"DIV",{class:!0});var Tt=x(Ke);f(Me.$$.fragment,Tt),Tt.forEach(s),Q.forEach(s),Wt=r(e),f(ve.$$.fragment,e),Bt=r(e),U=p(e,"DIV",{class:!0});var F=x(U);f($e.$$.fragment,F),un=r(F),et=p(F,"P",{"data-svelte-h":!0}),T(et)!=="svelte-v6yiax"&&(et.innerHTML=Kn),hn=r(F),tt=p(F,"P",{"data-svelte-h":!0}),T(tt)!=="svelte-ei9kk8"&&(tt.textContent=eo),fn=r(F),f(ne.$$.fragment,F),gn=fo(F,`
This is expected.
`),nt=p(F,"P",{"data-svelte-h":!0}),T(nt)!=="svelte-gxzj9w"&&(nt.innerHTML=to),F.forEach(s),Rt=r(e),f(xe.$$.fragment,e),Nt=r(e),H=p(e,"DIV",{class:!0});var Ee=x(H);f(Ce.$$.fragment,Ee),_n=r(Ee),ot=p(Ee,"DIV",{class:!0});var kt=x(ot);f(Qe.$$.fragment,kt),kt.forEach(s),Ee.forEach(s),Lt=r(e),f(ze.$$.fragment,e),At=r(e),j=p(e,"DIV",{class:!0});var I=x(j);f(Ue.$$.fragment,I),wn=r(I),st=p(I,"P",{"data-svelte-h":!0}),T(st)!=="svelte-1tbuuqj"&&(st.textContent=no),bn=r(I),at=p(I,"P",{"data-svelte-h":!0}),T(at)!=="svelte-q52n56"&&(at.innerHTML=oo),yn=r(I),rt=p(I,"P",{"data-svelte-h":!0}),T(rt)!=="svelte-hswkmf"&&(rt.innerHTML=so),Tn=r(I),X=p(I,"DIV",{class:!0});var O=x(X);f(je.$$.fragment,O),kn=r(O),it=p(O,"P",{"data-svelte-h":!0}),T(it)!=="svelte-ffrw5m"&&(it.innerHTML=ao),Mn=r(O),f(oe.$$.fragment,O),O.forEach(s),I.forEach(s),Xt=r(e),f(Je.$$.fragment,e),Vt=r(e),J=p(e,"DIV",{class:!0});var Z=x(J);f(qe.$$.fragment,Z),vn=r(Z),lt=p(Z,"P",{"data-svelte-h":!0}),T(lt)!=="svelte-xhccus"&&(lt.textContent=ro),$n=r(Z),dt=p(Z,"P",{"data-svelte-h":!0}),T(dt)!=="svelte-q52n56"&&(dt.innerHTML=io),xn=r(Z),ct=p(Z,"P",{"data-svelte-h":!0}),T(ct)!=="svelte-hswkmf"&&(ct.innerHTML=lo),Cn=r(Z),B=p(Z,"DIV",{class:!0});var N=x(B);f(Fe.$$.fragment,N),Qn=r(N),pt=p(N,"P",{"data-svelte-h":!0}),T(pt)!=="svelte-y89lci"&&(pt.innerHTML=co),zn=r(N),f(se.$$.fragment,N),Un=r(N),f(ae.$$.fragment,N),N.forEach(s),Z.forEach(s),Et=r(e),f(Ie.$$.fragment,e),Gt=r(e),P=p(e,"DIV",{class:!0});var Ge=x(P);f(Ze.$$.fragment,Ge),jn=r(Ge),V=p(Ge,"DIV",{class:!0});var ft=x(V);f(We.$$.fragment,ft),Jn=r(ft),mt=p(ft,"P",{"data-svelte-h":!0}),T(mt)!=="svelte-1sal4ui"&&(mt.innerHTML=po),qn=r(ft),f(re.$$.fragment,ft),ft.forEach(s),Ge.forEach(s),Ht=r(e),f(Be.$$.fragment,e),Pt=r(e),S=p(e,"DIV",{class:!0});var en=x(S);f(Re.$$.fragment,en),Fn=r(en),E=p(en,"DIV",{class:!0});var gt=x(E);f(Ne.$$.fragment,gt),In=r(gt),ut=p(gt,"P",{"data-svelte-h":!0}),T(ut)!=="svelte-1py4aay"&&(ut.innerHTML=mo),Zn=r(gt),f(ie.$$.fragment,gt),gt.forEach(s),en.forEach(s),St=r(e),f(Le.$$.fragment,e),Dt=r(e),D=p(e,"DIV",{class:!0});var tn=x(D);f(Ae.$$.fragment,tn),Wn=r(tn),G=p(tn,"DIV",{class:!0});var _t=x(G);f(Xe.$$.fragment,_t),Bn=r(_t),ht=p(_t,"P",{"data-svelte-h":!0}),T(ht)!=="svelte-dyrov9"&&(ht.innerHTML=uo),Rn=r(_t),f(le.$$.fragment,_t),_t.forEach(s),tn.forEach(s),Ot=r(e),f(Ve.$$.fragment,e),Yt=r(e),yt=p(e,"P",{}),x(yt).forEach(s),this.h()},h(){C(t,"name","hf:doc:metadata"),C(t,"content",Ro),To(v,"float","right"),C(q,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),C(Ke,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),C($,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),C(U,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),C(ot,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),C(H,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),C(X,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),C(j,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),C(B,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),C(J,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),C(V,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),C(P,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),C(E,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),C(S,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),C(G,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),C(D,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,o){l(document.head,t),i(e,d,o),i(e,n,o),i(e,m,o),i(e,k,o),i(e,y,o),i(e,v,o),i(e,ce,o),g(A,e,o),i(e,Mt,o),i(e,pe,o),i(e,vt,o),i(e,me,o),i(e,$t,o),g(Y,e,o),i(e,xt,o),i(e,ue,o),i(e,Ct,o),g(K,e,o),i(e,Qt,o),i(e,he,o),i(e,zt,o),i(e,fe,o),i(e,Ut,o),g(ge,e,o),i(e,jt,o),g(_e,e,o),i(e,Jt,o),i(e,we,o),i(e,qt,o),g(be,e,o),i(e,Ft,o),i(e,q,o),g(ye,q,null),l(q,on),l(q,He),l(q,sn),l(q,Pe),l(q,an),g(ee,q,null),i(e,It,o),g(Te,e,o),i(e,Zt,o),i(e,$,o),g(ke,$,null),l($,rn),l($,Se),l($,ln),l($,De),l($,dn),g(te,$,null),l($,cn),l($,Oe),l($,pn),l($,Ye),l($,mn),l($,Ke),g(Me,Ke,null),i(e,Wt,o),g(ve,e,o),i(e,Bt,o),i(e,U,o),g($e,U,null),l(U,un),l(U,et),l(U,hn),l(U,tt),l(U,fn),g(ne,U,null),l(U,gn),l(U,nt),i(e,Rt,o),g(xe,e,o),i(e,Nt,o),i(e,H,o),g(Ce,H,null),l(H,_n),l(H,ot),g(Qe,ot,null),i(e,Lt,o),g(ze,e,o),i(e,At,o),i(e,j,o),g(Ue,j,null),l(j,wn),l(j,st),l(j,bn),l(j,at),l(j,yn),l(j,rt),l(j,Tn),l(j,X),g(je,X,null),l(X,kn),l(X,it),l(X,Mn),g(oe,X,null),i(e,Xt,o),g(Je,e,o),i(e,Vt,o),i(e,J,o),g(qe,J,null),l(J,vn),l(J,lt),l(J,$n),l(J,dt),l(J,xn),l(J,ct),l(J,Cn),l(J,B),g(Fe,B,null),l(B,Qn),l(B,pt),l(B,zn),g(se,B,null),l(B,Un),g(ae,B,null),i(e,Et,o),g(Ie,e,o),i(e,Gt,o),i(e,P,o),g(Ze,P,null),l(P,jn),l(P,V),g(We,V,null),l(V,Jn),l(V,mt),l(V,qn),g(re,V,null),i(e,Ht,o),g(Be,e,o),i(e,Pt,o),i(e,S,o),g(Re,S,null),l(S,Fn),l(S,E),g(Ne,E,null),l(E,In),l(E,ut),l(E,Zn),g(ie,E,null),i(e,St,o),g(Le,e,o),i(e,Dt,o),i(e,D,o),g(Ae,D,null),l(D,Wn),l(D,G),g(Xe,G,null),l(G,Bn),l(G,ht),l(G,Rn),g(le,G,null),i(e,Ot,o),g(Ve,e,o),i(e,Yt,o),i(e,yt,o),Kt=!0},p(e,[o]){const R={};o&2&&(R.$$scope={dirty:o,ctx:e}),Y.$set(R);const Q={};o&2&&(Q.$$scope={dirty:o,ctx:e}),K.$set(Q);const Tt={};o&2&&(Tt.$$scope={dirty:o,ctx:e}),ee.$set(Tt);const F={};o&2&&(F.$$scope={dirty:o,ctx:e}),te.$set(F);const Ee={};o&2&&(Ee.$$scope={dirty:o,ctx:e}),ne.$set(Ee);const kt={};o&2&&(kt.$$scope={dirty:o,ctx:e}),oe.$set(kt);const I={};o&2&&(I.$$scope={dirty:o,ctx:e}),se.$set(I);const O={};o&2&&(O.$$scope={dirty:o,ctx:e}),ae.$set(O);const Z={};o&2&&(Z.$$scope={dirty:o,ctx:e}),re.$set(Z);const N={};o&2&&(N.$$scope={dirty:o,ctx:e}),ie.$set(N);const Ge={};o&2&&(Ge.$$scope={dirty:o,ctx:e}),le.$set(Ge)},i(e){Kt||(_(A.$$.fragment,e),_(Y.$$.fragment,e),_(K.$$.fragment,e),_(ge.$$.fragment,e),_(_e.$$.fragment,e),_(be.$$.fragment,e),_(ye.$$.fragment,e),_(ee.$$.fragment,e),_(Te.$$.fragment,e),_(ke.$$.fragment,e),_(te.$$.fragment,e),_(Me.$$.fragment,e),_(ve.$$.fragment,e),_($e.$$.fragment,e),_(ne.$$.fragment,e),_(xe.$$.fragment,e),_(Ce.$$.fragment,e),_(Qe.$$.fragment,e),_(ze.$$.fragment,e),_(Ue.$$.fragment,e),_(je.$$.fragment,e),_(oe.$$.fragment,e),_(Je.$$.fragment,e),_(qe.$$.fragment,e),_(Fe.$$.fragment,e),_(se.$$.fragment,e),_(ae.$$.fragment,e),_(Ie.$$.fragment,e),_(Ze.$$.fragment,e),_(We.$$.fragment,e),_(re.$$.fragment,e),_(Be.$$.fragment,e),_(Re.$$.fragment,e),_(Ne.$$.fragment,e),_(ie.$$.fragment,e),_(Le.$$.fragment,e),_(Ae.$$.fragment,e),_(Xe.$$.fragment,e),_(le.$$.fragment,e),_(Ve.$$.fragment,e),Kt=!0)},o(e){w(A.$$.fragment,e),w(Y.$$.fragment,e),w(K.$$.fragment,e),w(ge.$$.fragment,e),w(_e.$$.fragment,e),w(be.$$.fragment,e),w(ye.$$.fragment,e),w(ee.$$.fragment,e),w(Te.$$.fragment,e),w(ke.$$.fragment,e),w(te.$$.fragment,e),w(Me.$$.fragment,e),w(ve.$$.fragment,e),w($e.$$.fragment,e),w(ne.$$.fragment,e),w(xe.$$.fragment,e),w(Ce.$$.fragment,e),w(Qe.$$.fragment,e),w(ze.$$.fragment,e),w(Ue.$$.fragment,e),w(je.$$.fragment,e),w(oe.$$.fragment,e),w(Je.$$.fragment,e),w(qe.$$.fragment,e),w(Fe.$$.fragment,e),w(se.$$.fragment,e),w(ae.$$.fragment,e),w(Ie.$$.fragment,e),w(Ze.$$.fragment,e),w(We.$$.fragment,e),w(re.$$.fragment,e),w(Be.$$.fragment,e),w(Re.$$.fragment,e),w(Ne.$$.fragment,e),w(ie.$$.fragment,e),w(Le.$$.fragment,e),w(Ae.$$.fragment,e),w(Xe.$$.fragment,e),w(le.$$.fragment,e),w(Ve.$$.fragment,e),Kt=!1},d(e){e&&(s(d),s(n),s(m),s(k),s(y),s(v),s(ce),s(Mt),s(pe),s(vt),s(me),s($t),s(xt),s(ue),s(Ct),s(Qt),s(he),s(zt),s(fe),s(Ut),s(jt),s(Jt),s(we),s(qt),s(Ft),s(q),s(It),s(Zt),s($),s(Wt),s(Bt),s(U),s(Rt),s(Nt),s(H),s(Lt),s(At),s(j),s(Xt),s(Vt),s(J),s(Et),s(Gt),s(P),s(Ht),s(Pt),s(S),s(St),s(Dt),s(D),s(Ot),s(Yt),s(yt)),s(t),b(A,e),b(Y,e),b(K,e),b(ge,e),b(_e,e),b(be,e),b(ye),b(ee),b(Te,e),b(ke),b(te),b(Me),b(ve,e),b($e),b(ne),b(xe,e),b(Ce),b(Qe),b(ze,e),b(Ue),b(je),b(oe),b(Je,e),b(qe),b(Fe),b(se),b(ae),b(Ie,e),b(Ze),b(We),b(re),b(Be,e),b(Re),b(Ne),b(ie),b(Le,e),b(Ae),b(Xe),b(le),b(Ve,e)}}}const Ro='{"title":"Qwen2","local":"qwen2","sections":[{"title":"Notes","local":"notes","sections":[],"depth":2},{"title":"Qwen2Config","local":"transformers.Qwen2Config","sections":[],"depth":2},{"title":"Qwen2Tokenizer","local":"transformers.Qwen2Tokenizer","sections":[],"depth":2},{"title":"Qwen2TokenizerFast","local":"transformers.Qwen2TokenizerFast","sections":[],"depth":2},{"title":"Qwen2RMSNorm","local":"transformers.Qwen2RMSNorm","sections":[],"depth":2},{"title":"Qwen2Model","local":"transformers.Qwen2Model","sections":[],"depth":2},{"title":"Qwen2ForCausalLM","local":"transformers.Qwen2ForCausalLM","sections":[],"depth":2},{"title":"Qwen2ForSequenceClassification","local":"transformers.Qwen2ForSequenceClassification","sections":[],"depth":2},{"title":"Qwen2ForTokenClassification","local":"transformers.Qwen2ForTokenClassification","sections":[],"depth":2},{"title":"Qwen2ForQuestionAnswering","local":"transformers.Qwen2ForQuestionAnswering","sections":[],"depth":2}],"depth":1}';function No(M){return _o(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class So extends wo{constructor(t){super(),bo(this,t,No,Bo,go,{})}}export{So as component};
