import{s as at,o as rt,n as I}from"../chunks/scheduler.18a86fab.js";import{S as it,i as lt,g as d,s as r,r as h,A as dt,h as c,f as s,c as i,j as Q,x as b,u as f,k as U,l as ct,y as p,a,v as g,d as _,t as w,w as M}from"../chunks/index.98837b22.js";import{T as Ve}from"../chunks/Tip.77304350.js";import{D as j}from"../chunks/Docstring.a1ef7999.js";import{C as Ne}from"../chunks/CodeBlock.8d0c2e8a.js";import{E as st}from"../chunks/ExampleCodeBlock.8c3ee1f9.js";import{H as ze,E as pt}from"../chunks/getInferenceSnippets.06c2775f.js";import{H as ut,a as Ao}from"../chunks/HfOption.6641485e.js";function mt(y){let o,l="Click on the Qwen2MoE models in the right sidebar for more examples of how to apply Qwen2MoE to different language tasks.";return{c(){o=d("p"),o.textContent=l},l(t){o=c(t,"P",{"data-svelte-h":!0}),b(o)!=="svelte-1r0tg7v"&&(o.textContent=l)},m(t,u){a(t,o,u)},p:I,d(t){t&&s(o)}}}function ht(y){let o,l;return o=new Ne({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwcGlwZWxpbmUlMEElMEFwaXBlJTIwJTNEJTIwcGlwZWxpbmUoJTBBJTIwJTIwJTIwJTIwdGFzayUzRCUyMnRleHQtZ2VuZXJhdGlvbiUyMiUyQyUwQSUyMCUyMCUyMCUyMG1vZGVsJTNEJTIyUXdlbiUyRlF3ZW4xLjUtTW9FLUEyLjdCJTIyJTJDJTBBJTIwJTIwJTIwJTIwZHR5cGUlM0R0b3JjaC5iZmxvYXQxNiUyQyUwQSUyMCUyMCUyMCUyMGRldmljZV9tYXAlM0QwJTBBKSUwQSUwQW1lc3NhZ2VzJTIwJTNEJTIwJTVCJTBBJTIwJTIwJTIwJTIwJTdCJTIycm9sZSUyMiUzQSUyMCUyMnN5c3RlbSUyMiUyQyUyMCUyMmNvbnRlbnQlMjIlM0ElMjAlMjJZb3UlMjBhcmUlMjBhJTIwaGVscGZ1bCUyMGFzc2lzdGFudC4lMjIlN0QlMkMlMEElMjAlMjAlMjAlMjAlN0IlMjJyb2xlJTIyJTNBJTIwJTIydXNlciUyMiUyQyUyMCUyMmNvbnRlbnQlMjIlM0ElMjAlMjJUZWxsJTIwbWUlMjBhYm91dCUyMHRoZSUyMFF3ZW4yJTIwbW9kZWwlMjBmYW1pbHkuJTIyJTdEJTJDJTBBJTVEJTBBb3V0cHV0cyUyMCUzRCUyMHBpcGUobWVzc2FnZXMlMkMlMjBtYXhfbmV3X3Rva2VucyUzRDI1NiUyQyUyMGRvX3NhbXBsZSUzRFRydWUlMkMlMjB0ZW1wZXJhdHVyZSUzRDAuNyUyQyUyMHRvcF9rJTNENTAlMkMlMjB0b3BfcCUzRDAuOTUpJTBBcHJpbnQob3V0cHV0cyU1QjAlNUQlNUIlMjJnZW5lcmF0ZWRfdGV4dCUyMiU1RCU1Qi0xJTVEJTVCJ2NvbnRlbnQnJTVEKQ==",highlighted:`<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> pipeline

pipe = pipeline(
    task=<span class="hljs-string">&quot;text-generation&quot;</span>,
    model=<span class="hljs-string">&quot;Qwen/Qwen1.5-MoE-A2.7B&quot;</span>,
    dtype=torch.bfloat16,
    device_map=<span class="hljs-number">0</span>
)

messages = [
    {<span class="hljs-string">&quot;role&quot;</span>: <span class="hljs-string">&quot;system&quot;</span>, <span class="hljs-string">&quot;content&quot;</span>: <span class="hljs-string">&quot;You are a helpful assistant.&quot;</span>},
    {<span class="hljs-string">&quot;role&quot;</span>: <span class="hljs-string">&quot;user&quot;</span>, <span class="hljs-string">&quot;content&quot;</span>: <span class="hljs-string">&quot;Tell me about the Qwen2 model family.&quot;</span>},
]
outputs = pipe(messages, max_new_tokens=<span class="hljs-number">256</span>, do_sample=<span class="hljs-literal">True</span>, temperature=<span class="hljs-number">0.7</span>, top_k=<span class="hljs-number">50</span>, top_p=<span class="hljs-number">0.95</span>)
<span class="hljs-built_in">print</span>(outputs[<span class="hljs-number">0</span>][<span class="hljs-string">&quot;generated_text&quot;</span>][-<span class="hljs-number">1</span>][<span class="hljs-string">&#x27;content&#x27;</span>])`,wrap:!1}}),{c(){h(o.$$.fragment)},l(t){f(o.$$.fragment,t)},m(t,u){g(o,t,u),l=!0},p:I,i(t){l||(_(o.$$.fragment,t),l=!0)},o(t){w(o.$$.fragment,t),l=!1},d(t){M(o,t)}}}function ft(y){let o,l;return o=new Ne({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b01vZGVsRm9yQ2F1c2FsTE0lMkMlMjBBdXRvVG9rZW5pemVyJTBBJTBBbW9kZWwlMjAlM0QlMjBBdXRvTW9kZWxGb3JDYXVzYWxMTS5mcm9tX3ByZXRyYWluZWQoJTBBJTIwJTIwJTIwJTIwJTIyUXdlbiUyRlF3ZW4xLjUtTW9FLUEyLjdCLUNoYXQlMjIlMkMlMEElMjAlMjAlMjAlMjBkdHlwZSUzRHRvcmNoLmJmbG9hdDE2JTJDJTBBJTIwJTIwJTIwJTIwZGV2aWNlX21hcCUzRCUyMmF1dG8lMjIlMkMlMEElMjAlMjAlMjAlMjBhdHRuX2ltcGxlbWVudGF0aW9uJTNEJTIyc2RwYSUyMiUwQSklMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJRd2VuJTJGUXdlbjEuNS1Nb0UtQTIuN0ItQ2hhdCUyMiklMEElMEFwcm9tcHQlMjAlM0QlMjAlMjJHaXZlJTIwbWUlMjBhJTIwc2hvcnQlMjBpbnRyb2R1Y3Rpb24lMjB0byUyMGxhcmdlJTIwbGFuZ3VhZ2UlMjBtb2RlbHMuJTIyJTBBbWVzc2FnZXMlMjAlM0QlMjAlNUIlMEElMjAlMjAlMjAlMjAlN0IlMjJyb2xlJTIyJTNBJTIwJTIyc3lzdGVtJTIyJTJDJTIwJTIyY29udGVudCUyMiUzQSUyMCUyMllvdSUyMGFyZSUyMGElMjBoZWxwZnVsJTIwYXNzaXN0YW50LiUyMiU3RCUyQyUwQSUyMCUyMCUyMCUyMCU3QiUyMnJvbGUlMjIlM0ElMjAlMjJ1c2VyJTIyJTJDJTIwJTIyY29udGVudCUyMiUzQSUyMHByb21wdCU3RCUwQSU1RCUwQXRleHQlMjAlM0QlMjB0b2tlbml6ZXIuYXBwbHlfY2hhdF90ZW1wbGF0ZSglMEElMjAlMjAlMjAlMjBtZXNzYWdlcyUyQyUwQSUyMCUyMCUyMCUyMHRva2VuaXplJTNERmFsc2UlMkMlMEElMjAlMjAlMjAlMjBhZGRfZ2VuZXJhdGlvbl9wcm9tcHQlM0RUcnVlJTBBKSUwQW1vZGVsX2lucHV0cyUyMCUzRCUyMHRva2VuaXplciglNUJ0ZXh0JTVEJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMikudG8obW9kZWwuZGV2aWNlKSUwQSUwQWdlbmVyYXRlZF9pZHMlMjAlM0QlMjBtb2RlbC5nZW5lcmF0ZSglMEElMjAlMjAlMjAlMjBtb2RlbF9pbnB1dHMuaW5wdXRfaWRzJTJDJTBBJTIwJTIwJTIwJTIwY2FjaGVfaW1wbGVtZW50YXRpb24lM0QlMjJzdGF0aWMlMjIlMkMlMEElMjAlMjAlMjAlMjBtYXhfbmV3X3Rva2VucyUzRDUxMiUyQyUwQSUyMCUyMCUyMCUyMGRvX3NhbXBsZSUzRFRydWUlMkMlMEElMjAlMjAlMjAlMjB0ZW1wZXJhdHVyZSUzRDAuNyUyQyUwQSUyMCUyMCUyMCUyMHRvcF9rJTNENTAlMkMlMEElMjAlMjAlMjAlMjB0b3BfcCUzRDAuOTUlMEEpJTBBZ2VuZXJhdGVkX2lkcyUyMCUzRCUyMCU1QiUwQSUyMCUyMCUyMCUyMG91dHB1dF9pZHMlNUJsZW4oaW5wdXRfaWRzKSUzQSU1RCUyMGZvciUyMGlucHV0X2lkcyUyQyUyMG91dHB1dF9pZHMlMjBpbiUyMHppcChtb2RlbF9pbnB1dHMuaW5wdXRfaWRzJTJDJTIwZ2VuZXJhdGVkX2lkcyklMEElNUQlMEElMEFyZXNwb25zZSUyMCUzRCUyMHRva2VuaXplci5iYXRjaF9kZWNvZGUoZ2VuZXJhdGVkX2lkcyUyQyUyMHNraXBfc3BlY2lhbF90b2tlbnMlM0RUcnVlKSU1QjAlNUQlMEFwcmludChyZXNwb25zZSk=",highlighted:`<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    <span class="hljs-string">&quot;Qwen/Qwen1.5-MoE-A2.7B-Chat&quot;</span>,
    dtype=torch.bfloat16,
    device_map=<span class="hljs-string">&quot;auto&quot;</span>,
    attn_implementation=<span class="hljs-string">&quot;sdpa&quot;</span>
)
tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;Qwen/Qwen1.5-MoE-A2.7B-Chat&quot;</span>)

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
<span class="hljs-built_in">print</span>(response)`,wrap:!1}}),{c(){h(o.$$.fragment)},l(t){f(o.$$.fragment,t)},m(t,u){g(o,t,u),l=!0},p:I,i(t){l||(_(o.$$.fragment,t),l=!0)},o(t){w(o.$$.fragment,t),l=!1},d(t){M(o,t)}}}function gt(y){let o,l;return o=new Ne({props:{code:"dHJhbnNmb3JtZXJzJTIwY2hhdCUyMFF3ZW4lMkZRd2VuMS41LU1vRS1BMi43Qi1DaGF0JTIwLS1kdHlwZSUyMGF1dG8lMjAtLWF0dG5faW1wbGVtZW50YXRpb24lMjBmbGFzaF9hdHRlbnRpb25fMg==",highlighted:"transformers chat Qwen/Qwen1.5-MoE-A2.7B-Chat --dtype auto --attn_implementation flash_attention_2",wrap:!1}}),{c(){h(o.$$.fragment)},l(t){f(o.$$.fragment,t)},m(t,u){g(o,t,u),l=!0},p:I,i(t){l||(_(o.$$.fragment,t),l=!0)},o(t){w(o.$$.fragment,t),l=!1},d(t){M(o,t)}}}function _t(y){let o,l,t,u,v,T;return o=new Ao({props:{id:"usage",option:"Pipeline",$$slots:{default:[ht]},$$scope:{ctx:y}}}),t=new Ao({props:{id:"usage",option:"AutoModel",$$slots:{default:[ft]},$$scope:{ctx:y}}}),v=new Ao({props:{id:"usage",option:"transformers CLI",$$slots:{default:[gt]},$$scope:{ctx:y}}}),{c(){h(o.$$.fragment),l=r(),h(t.$$.fragment),u=r(),h(v.$$.fragment)},l(m){f(o.$$.fragment,m),l=i(m),f(t.$$.fragment,m),u=i(m),f(v.$$.fragment,m)},m(m,k){g(o,m,k),a(m,l,k),g(t,m,k),a(m,u,k),g(v,m,k),T=!0},p(m,k){const Xe={};k&2&&(Xe.$$scope={dirty:k,ctx:m}),o.$set(Xe);const te={};k&2&&(te.$$scope={dirty:k,ctx:m}),t.$set(te);const q={};k&2&&(q.$$scope={dirty:k,ctx:m}),v.$set(q)},i(m){T||(_(o.$$.fragment,m),_(t.$$.fragment,m),_(v.$$.fragment,m),T=!0)},o(m){w(o.$$.fragment,m),w(t.$$.fragment,m),w(v.$$.fragment,m),T=!1},d(m){m&&(s(l),s(u)),M(o,m),M(t,m),M(v,m)}}}function wt(y){let o,l;return o=new Ne({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMFF3ZW4yTW9lTW9kZWwlMkMlMjBRd2VuMk1vZUNvbmZpZyUwQSUwQSUyMyUyMEluaXRpYWxpemluZyUyMGElMjBRd2VuMk1vRSUyMHN0eWxlJTIwY29uZmlndXJhdGlvbiUwQWNvbmZpZ3VyYXRpb24lMjAlM0QlMjBRd2VuMk1vZUNvbmZpZygpJTBBJTBBJTIzJTIwSW5pdGlhbGl6aW5nJTIwYSUyMG1vZGVsJTIwZnJvbSUyMHRoZSUyMFF3ZW4xLjUtTW9FLUEyLjdCJTIyJTIwc3R5bGUlMjBjb25maWd1cmF0aW9uJTBBbW9kZWwlMjAlM0QlMjBRd2VuMk1vZU1vZGVsKGNvbmZpZ3VyYXRpb24pJTBBJTBBJTIzJTIwQWNjZXNzaW5nJTIwdGhlJTIwbW9kZWwlMjBjb25maWd1cmF0aW9uJTBBY29uZmlndXJhdGlvbiUyMCUzRCUyMG1vZGVsLmNvbmZpZw==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> Qwen2MoeModel, Qwen2MoeConfig

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a Qwen2MoE style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = Qwen2MoeConfig()

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a model from the Qwen1.5-MoE-A2.7B&quot; style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = Qwen2MoeModel(configuration)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Accessing the model configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = model.config`,wrap:!1}}),{c(){h(o.$$.fragment)},l(t){f(o.$$.fragment,t)},m(t,u){g(o,t,u),l=!0},p:I,i(t){l||(_(o.$$.fragment,t),l=!0)},o(t){w(o.$$.fragment,t),l=!1},d(t){M(o,t)}}}function Mt(y){let o,l=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){o=d("p"),o.innerHTML=l},l(t){o=c(t,"P",{"data-svelte-h":!0}),b(o)!=="svelte-fincs2"&&(o.innerHTML=l)},m(t,u){a(t,o,u)},p:I,d(t){t&&s(o)}}}function yt(y){let o,l=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){o=d("p"),o.innerHTML=l},l(t){o=c(t,"P",{"data-svelte-h":!0}),b(o)!=="svelte-fincs2"&&(o.innerHTML=l)},m(t,u){a(t,o,u)},p:I,d(t){t&&s(o)}}}function bt(y){let o,l="Example:",t,u,v;return u=new Ne({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBRd2VuMk1vZUZvckNhdXNhbExNJTBBJTBBbW9kZWwlMjAlM0QlMjBRd2VuMk1vZUZvckNhdXNhbExNLmZyb21fcHJldHJhaW5lZChQQVRIX1RPX0NPTlZFUlRFRF9XRUlHSFRTKSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKFBBVEhfVE9fQ09OVkVSVEVEX1RPS0VOSVpFUiklMEElMEFwcm9tcHQlMjAlM0QlMjAlMjJIZXklMkMlMjBhcmUlMjB5b3UlMjBjb25zY2lvdXMlM0YlMjBDYW4lMjB5b3UlMjB0YWxrJTIwdG8lMjBtZSUzRiUyMiUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplcihwcm9tcHQlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQSUwQSUyMyUyMEdlbmVyYXRlJTBBZ2VuZXJhdGVfaWRzJTIwJTNEJTIwbW9kZWwuZ2VuZXJhdGUoaW5wdXRzLmlucHV0X2lkcyUyQyUyMG1heF9sZW5ndGglM0QzMCklMEF0b2tlbml6ZXIuYmF0Y2hfZGVjb2RlKGdlbmVyYXRlX2lkcyUyQyUyMHNraXBfc3BlY2lhbF90b2tlbnMlM0RUcnVlJTJDJTIwY2xlYW5fdXBfdG9rZW5pemF0aW9uX3NwYWNlcyUzREZhbHNlKSU1QjAlNUQ=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, Qwen2MoeForCausalLM

<span class="hljs-meta">&gt;&gt;&gt; </span>model = Qwen2MoeForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

<span class="hljs-meta">&gt;&gt;&gt; </span>prompt = <span class="hljs-string">&quot;Hey, are you conscious? Can you talk to me?&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(prompt, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Generate</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>generate_ids = model.generate(inputs.input_ids, max_length=<span class="hljs-number">30</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer.batch_decode(generate_ids, skip_special_tokens=<span class="hljs-literal">True</span>, clean_up_tokenization_spaces=<span class="hljs-literal">False</span>)[<span class="hljs-number">0</span>]
<span class="hljs-string">&quot;Hey, are you conscious? Can you talk to me?\\nI&#x27;m not conscious, but I can talk to you.&quot;</span>`,wrap:!1}}),{c(){o=d("p"),o.textContent=l,t=r(),h(u.$$.fragment)},l(T){o=c(T,"P",{"data-svelte-h":!0}),b(o)!=="svelte-11lpom8"&&(o.textContent=l),t=i(T),f(u.$$.fragment,T)},m(T,m){a(T,o,m),a(T,t,m),g(u,T,m),v=!0},p:I,i(T){v||(_(u.$$.fragment,T),v=!0)},o(T){w(u.$$.fragment,T),v=!1},d(T){T&&(s(o),s(t)),M(u,T)}}}function Tt(y){let o,l=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){o=d("p"),o.innerHTML=l},l(t){o=c(t,"P",{"data-svelte-h":!0}),b(o)!=="svelte-fincs2"&&(o.innerHTML=l)},m(t,u){a(t,o,u)},p:I,d(t){t&&s(o)}}}function vt(y){let o,l=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){o=d("p"),o.innerHTML=l},l(t){o=c(t,"P",{"data-svelte-h":!0}),b(o)!=="svelte-fincs2"&&(o.innerHTML=l)},m(t,u){a(t,o,u)},p:I,d(t){t&&s(o)}}}function kt(y){let o,l=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){o=d("p"),o.innerHTML=l},l(t){o=c(t,"P",{"data-svelte-h":!0}),b(o)!=="svelte-fincs2"&&(o.innerHTML=l)},m(t,u){a(t,o,u)},p:I,d(t){t&&s(o)}}}function $t(y){let o,l,t,u,v,T="<em>This model was released on 2024-07-15 and added to Hugging Face Transformers on 2024-03-27.</em>",m,k,Xe='<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/> <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat"/> <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/> <img alt="Tensor parallelism" src="https://img.shields.io/badge/Tensor%20parallelism-06b6d4?style=flat&amp;logoColor=white"/>',te,q,Se,ne,Lo='<a href="https://huggingface.co/papers/2407.10671" rel="nofollow">Qwen2MoE</a> is a Mixture-of-Experts (MoE) variant of <a href="./qwen2">Qwen2</a>, available as a base model and an aligned chat model. It uses SwiGLU activation, group query attention and a mixture of sliding window attention and full attention. The tokenizer can also be adapted to multiple languages and codes.',Ge,se,Vo="The MoE architecture uses upcyled models from the dense language models. For example, Qwen1.5-MoE-A2.7B is upcycled from Qwen-1.8B. It has 14.3B parameters but only 2.7B parameters are activated during runtime.",Pe,ae,No='You can find all the original checkpoints in the <a href="https://huggingface.co/collections/Qwen/qwen15-65c0a2f577b1ecb76d786524" rel="nofollow">Qwen1.5</a> collection.',Oe,S,De,re,Xo='The example below demonstrates how to generate text with <a href="/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.Pipeline">Pipeline</a>, <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel">AutoModel</a>, and from the command line.',Ye,G,Ke,ie,Ho='Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the <a href="../quantization/overview">Quantization</a> overview for more available quantization backends.',eo,le,So='The example below uses <a href="../quantization/bitsandbytes">bitsandbytes</a> to quantize the weights to 8-bits.',oo,de,to,ce,no,x,pe,Mo,Fe,Go=`This is the configuration class to store the configuration of a <a href="/docs/transformers/v4.56.2/en/model_doc/qwen2_moe#transformers.Qwen2MoeModel">Qwen2MoeModel</a>. It is used to instantiate a
Qwen2MoE model according to the specified arguments, defining the model architecture. Instantiating a configuration
with the defaults will yield a similar configuration to that of <a href="https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B" rel="nofollow">Qwen/Qwen1.5-MoE-A2.7B</a>.`,yo,je,Po=`Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> and can be used to control the model outputs. Read the
documentation from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> for more information.`,bo,P,so,ue,ao,$,me,To,Ie,Oo="The bare Qwen2 Moe Model outputting raw hidden-states without any specific head on top.",vo,qe,Do=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,ko,Ze,Yo=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,$o,Z,he,xo,Be,Ko='The <a href="/docs/transformers/v4.56.2/en/model_doc/qwen2_moe#transformers.Qwen2MoeModel">Qwen2MoeModel</a> forward method, overrides the <code>__call__</code> special method.',Co,O,ro,fe,io,R,ge,Qo,J,_e,Uo,Ee,et='The <a href="/docs/transformers/v4.56.2/en/model_doc/qwen2_moe#transformers.Qwen2MoeForCausalLM">Qwen2MoeForCausalLM</a> forward method, overrides the <code>__call__</code> special method.',Jo,D,zo,Y,lo,we,co,A,Me,Fo,B,ye,jo,We,ot="The <code>GenericForSequenceClassification</code> forward method, overrides the <code>__call__</code> special method.",Io,K,po,be,uo,L,Te,qo,E,ve,Zo,Re,tt="The <code>GenericForTokenClassification</code> forward method, overrides the <code>__call__</code> special method.",Bo,ee,mo,ke,ho,V,$e,Eo,W,xe,Wo,Ae,nt="The <code>GenericForQuestionAnswering</code> forward method, overrides the <code>__call__</code> special method.",Ro,oe,fo,Ce,go,He,_o;return q=new ze({props:{title:"Qwen2MoE",local:"qwen2moe",headingTag:"h1"}}),S=new Ve({props:{warning:!1,$$slots:{default:[mt]},$$scope:{ctx:y}}}),G=new ut({props:{id:"usage",options:["Pipeline","AutoModel","transformers CLI"],$$slots:{default:[_t]},$$scope:{ctx:y}}}),de=new Ne({props:{code:"JTIzJTIwcGlwJTIwaW5zdGFsbCUyMC1VJTIwZmxhc2gtYXR0biUyMC0tbm8tYnVpbGQtaXNvbGF0aW9uJTBBaW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Rva2VuaXplciUyQyUyMEF1dG9Nb2RlbEZvckNhdXNhbExNJTJDJTIwQml0c0FuZEJ5dGVzQ29uZmlnJTBBJTBBcXVhbnRpemF0aW9uX2NvbmZpZyUyMCUzRCUyMEJpdHNBbmRCeXRlc0NvbmZpZyglMEElMjAlMjAlMjAlMjBsb2FkX2luXzhiaXQlM0RUcnVlJTBBKSUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMlF3ZW4lMkZRd2VuMS41LU1vRS1BMi43Qi1DaGF0JTIyKSUwQW1vZGVsJTIwJTNEJTIwQXV0b01vZGVsRm9yQ2F1c2FsTE0uZnJvbV9wcmV0cmFpbmVkKCUwQSUyMCUyMCUyMCUyMCUyMlF3ZW4lMkZRd2VuMS41LU1vRS1BMi43Qi1DaGF0JTIyJTJDJTBBJTIwJTIwJTIwJTIwZHR5cGUlM0R0b3JjaC5iZmxvYXQxNiUyQyUwQSUyMCUyMCUyMCUyMGRldmljZV9tYXAlM0QlMjJhdXRvJTIyJTJDJTBBJTIwJTIwJTIwJTIwcXVhbnRpemF0aW9uX2NvbmZpZyUzRHF1YW50aXphdGlvbl9jb25maWclMkMlMEElMjAlMjAlMjAlMjBhdHRuX2ltcGxlbWVudGF0aW9uJTNEJTIyZmxhc2hfYXR0ZW50aW9uXzIlMjIlMEEpJTBBJTBBaW5wdXRzJTIwJTNEJTIwdG9rZW5pemVyKCUyMlRoZSUyMFF3ZW4yJTIwbW9kZWwlMjBmYW1pbHklMjBpcyUyMiUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpLnRvKG1vZGVsLmRldmljZSklMEFvdXRwdXRzJTIwJTNEJTIwbW9kZWwuZ2VuZXJhdGUoKippbnB1dHMlMkMlMjBtYXhfbmV3X3Rva2VucyUzRDEwMCklMEFwcmludCh0b2tlbml6ZXIuZGVjb2RlKG91dHB1dHMlNUIwJTVEJTJDJTIwc2tpcF9zcGVjaWFsX3Rva2VucyUzRFRydWUpKQ==",highlighted:`<span class="hljs-comment"># pip install -U flash-attn --no-build-isolation</span>
<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_8bit=<span class="hljs-literal">True</span>
)

tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;Qwen/Qwen1.5-MoE-A2.7B-Chat&quot;</span>)
model = AutoModelForCausalLM.from_pretrained(
    <span class="hljs-string">&quot;Qwen/Qwen1.5-MoE-A2.7B-Chat&quot;</span>,
    dtype=torch.bfloat16,
    device_map=<span class="hljs-string">&quot;auto&quot;</span>,
    quantization_config=quantization_config,
    attn_implementation=<span class="hljs-string">&quot;flash_attention_2&quot;</span>
)

inputs = tokenizer(<span class="hljs-string">&quot;The Qwen2 model family is&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>).to(model.device)
outputs = model.generate(**inputs, max_new_tokens=<span class="hljs-number">100</span>)
<span class="hljs-built_in">print</span>(tokenizer.decode(outputs[<span class="hljs-number">0</span>], skip_special_tokens=<span class="hljs-literal">True</span>))`,wrap:!1}}),ce=new ze({props:{title:"Qwen2MoeConfig",local:"transformers.Qwen2MoeConfig",headingTag:"h2"}}),pe=new j({props:{name:"class transformers.Qwen2MoeConfig",anchor:"transformers.Qwen2MoeConfig",parameters:[{name:"vocab_size",val:" = 151936"},{name:"hidden_size",val:" = 2048"},{name:"intermediate_size",val:" = 5632"},{name:"num_hidden_layers",val:" = 24"},{name:"num_attention_heads",val:" = 16"},{name:"num_key_value_heads",val:" = 16"},{name:"hidden_act",val:" = 'silu'"},{name:"max_position_embeddings",val:" = 32768"},{name:"initializer_range",val:" = 0.02"},{name:"rms_norm_eps",val:" = 1e-06"},{name:"use_cache",val:" = True"},{name:"tie_word_embeddings",val:" = False"},{name:"rope_theta",val:" = 10000.0"},{name:"rope_scaling",val:" = None"},{name:"use_sliding_window",val:" = False"},{name:"sliding_window",val:" = 4096"},{name:"max_window_layers",val:" = 28"},{name:"attention_dropout",val:" = 0.0"},{name:"decoder_sparse_step",val:" = 1"},{name:"moe_intermediate_size",val:" = 1408"},{name:"shared_expert_intermediate_size",val:" = 5632"},{name:"num_experts_per_tok",val:" = 4"},{name:"num_experts",val:" = 60"},{name:"norm_topk_prob",val:" = False"},{name:"output_router_logits",val:" = False"},{name:"router_aux_loss_coef",val:" = 0.001"},{name:"mlp_only_layers",val:" = None"},{name:"qkv_bias",val:" = True"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.Qwen2MoeConfig.vocab_size",description:`<strong>vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 151936) &#x2014;
Vocabulary size of the Qwen2MoE model. Defines the number of different tokens that can be represented by the
<code>inputs_ids</code> passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/qwen2_moe#transformers.Qwen2MoeModel">Qwen2MoeModel</a>`,name:"vocab_size"},{anchor:"transformers.Qwen2MoeConfig.hidden_size",description:`<strong>hidden_size</strong> (<code>int</code>, <em>optional</em>, defaults to 2048) &#x2014;
Dimension of the hidden representations.`,name:"hidden_size"},{anchor:"transformers.Qwen2MoeConfig.intermediate_size",description:`<strong>intermediate_size</strong> (<code>int</code>, <em>optional</em>, defaults to 5632) &#x2014;
Dimension of the MLP representations.`,name:"intermediate_size"},{anchor:"transformers.Qwen2MoeConfig.num_hidden_layers",description:`<strong>num_hidden_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 24) &#x2014;
Number of hidden layers in the Transformer encoder.`,name:"num_hidden_layers"},{anchor:"transformers.Qwen2MoeConfig.num_attention_heads",description:`<strong>num_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 16) &#x2014;
Number of attention heads for each attention layer in the Transformer encoder.`,name:"num_attention_heads"},{anchor:"transformers.Qwen2MoeConfig.num_key_value_heads",description:`<strong>num_key_value_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 16) &#x2014;
This is the number of key_value heads that should be used to implement Grouped Query Attention. If
<code>num_key_value_heads=num_attention_heads</code>, the model will use Multi Head Attention (MHA), if
<code>num_key_value_heads=1</code> the model will use Multi Query Attention (MQA) otherwise GQA is used. When
converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
by meanpooling all the original heads within that group. For more details, check out <a href="https://huggingface.co/papers/2305.13245" rel="nofollow">this
paper</a>. If it is not specified, will default to <code>32</code>.`,name:"num_key_value_heads"},{anchor:"transformers.Qwen2MoeConfig.hidden_act",description:`<strong>hidden_act</strong> (<code>str</code> or <code>function</code>, <em>optional</em>, defaults to <code>&quot;silu&quot;</code>) &#x2014;
The non-linear activation function (function or string) in the decoder.`,name:"hidden_act"},{anchor:"transformers.Qwen2MoeConfig.max_position_embeddings",description:`<strong>max_position_embeddings</strong> (<code>int</code>, <em>optional</em>, defaults to 32768) &#x2014;
The maximum sequence length that this model might ever be used with.`,name:"max_position_embeddings"},{anchor:"transformers.Qwen2MoeConfig.initializer_range",description:`<strong>initializer_range</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.`,name:"initializer_range"},{anchor:"transformers.Qwen2MoeConfig.rms_norm_eps",description:`<strong>rms_norm_eps</strong> (<code>float</code>, <em>optional</em>, defaults to 1e-06) &#x2014;
The epsilon used by the rms normalization layers.`,name:"rms_norm_eps"},{anchor:"transformers.Qwen2MoeConfig.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not the model should return the last key/values attentions (not used by all models). Only
relevant if <code>config.is_decoder=True</code>.`,name:"use_cache"},{anchor:"transformers.Qwen2MoeConfig.tie_word_embeddings",description:`<strong>tie_word_embeddings</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether the model&#x2019;s input and output word embeddings should be tied.`,name:"tie_word_embeddings"},{anchor:"transformers.Qwen2MoeConfig.rope_theta",description:`<strong>rope_theta</strong> (<code>float</code>, <em>optional</em>, defaults to 10000.0) &#x2014;
The base period of the RoPE embeddings.`,name:"rope_theta"},{anchor:"transformers.Qwen2MoeConfig.rope_scaling",description:`<strong>rope_scaling</strong> (<code>Dict</code>, <em>optional</em>) &#x2014;
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
Only used with &#x2018;llama3&#x2019;. Scaling factor applied to high frequency components of the RoPE`,name:"rope_scaling"},{anchor:"transformers.Qwen2MoeConfig.use_sliding_window",description:`<strong>use_sliding_window</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether to use sliding window attention.`,name:"use_sliding_window"},{anchor:"transformers.Qwen2MoeConfig.sliding_window",description:`<strong>sliding_window</strong> (<code>int</code>, <em>optional</em>, defaults to 4096) &#x2014;
Sliding window attention (SWA) window size. If not specified, will default to <code>4096</code>.`,name:"sliding_window"},{anchor:"transformers.Qwen2MoeConfig.max_window_layers",description:`<strong>max_window_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 28) &#x2014;
The number of layers using full attention. The first <code>max_window_layers</code> layers will use full attention, while any
additional layer afterwards will use SWA (Sliding Window Attention).`,name:"max_window_layers"},{anchor:"transformers.Qwen2MoeConfig.attention_dropout",description:`<strong>attention_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The dropout ratio for the attention probabilities.`,name:"attention_dropout"},{anchor:"transformers.Qwen2MoeConfig.decoder_sparse_step",description:`<strong>decoder_sparse_step</strong> (<code>int</code>, <em>optional</em>, defaults to 1) &#x2014;
The frequency of the MoE layer.`,name:"decoder_sparse_step"},{anchor:"transformers.Qwen2MoeConfig.moe_intermediate_size",description:`<strong>moe_intermediate_size</strong> (<code>int</code>, <em>optional</em>, defaults to 1408) &#x2014;
Intermediate size of the routed expert.`,name:"moe_intermediate_size"},{anchor:"transformers.Qwen2MoeConfig.shared_expert_intermediate_size",description:`<strong>shared_expert_intermediate_size</strong> (<code>int</code>, <em>optional</em>, defaults to 5632) &#x2014;
Intermediate size of the shared expert.`,name:"shared_expert_intermediate_size"},{anchor:"transformers.Qwen2MoeConfig.num_experts_per_tok",description:`<strong>num_experts_per_tok</strong> (<code>int</code>, <em>optional</em>, defaults to 4) &#x2014;
Number of selected experts.`,name:"num_experts_per_tok"},{anchor:"transformers.Qwen2MoeConfig.num_experts",description:`<strong>num_experts</strong> (<code>int</code>, <em>optional</em>, defaults to 60) &#x2014;
Number of routed experts.`,name:"num_experts"},{anchor:"transformers.Qwen2MoeConfig.norm_topk_prob",description:`<strong>norm_topk_prob</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether to normalize the topk probabilities.`,name:"norm_topk_prob"},{anchor:"transformers.Qwen2MoeConfig.output_router_logits",description:`<strong>output_router_logits</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not the router logits should be returned by the model. Enabling this will also
allow the model to output the auxiliary loss, including load balancing loss and router z-loss.`,name:"output_router_logits"},{anchor:"transformers.Qwen2MoeConfig.router_aux_loss_coef",description:`<strong>router_aux_loss_coef</strong> (<code>float</code>, <em>optional</em>, defaults to 0.001) &#x2014;
The aux loss factor for the total loss.`,name:"router_aux_loss_coef"},{anchor:"transformers.Qwen2MoeConfig.mlp_only_layers",description:`<strong>mlp_only_layers</strong> (<code>list[int]</code>, <em>optional</em>, defaults to <code>[]</code>) &#x2014;
Indicate which layers use Qwen2MoeMLP rather than Qwen2MoeSparseMoeBlock
The list contains layer index, from 0 to num_layers-1 if we have num_layers layers
If <code>mlp_only_layers</code> is empty, <code>decoder_sparse_step</code> is used to determine the sparsity.`,name:"mlp_only_layers"},{anchor:"transformers.Qwen2MoeConfig.qkv_bias",description:`<strong>qkv_bias</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether to add a bias to the queries, keys and values.`,name:"qkv_bias"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/qwen2_moe/configuration_qwen2_moe.py#L25"}}),P=new st({props:{anchor:"transformers.Qwen2MoeConfig.example",$$slots:{default:[wt]},$$scope:{ctx:y}}}),ue=new ze({props:{title:"Qwen2MoeModel",local:"transformers.Qwen2MoeModel",headingTag:"h2"}}),me=new j({props:{name:"class transformers.Qwen2MoeModel",anchor:"transformers.Qwen2MoeModel",parameters:[{name:"config",val:": Qwen2MoeConfig"}],parametersDescription:[{anchor:"transformers.Qwen2MoeModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/qwen2_moe#transformers.Qwen2MoeConfig">Qwen2MoeConfig</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/qwen2_moe/modeling_qwen2_moe.py#L774"}}),he=new j({props:{name:"forward",anchor:"transformers.Qwen2MoeModel.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"output_router_logits",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"}],parametersDescription:[{anchor:"transformers.Qwen2MoeModel.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.Qwen2MoeModel.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.Qwen2MoeModel.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.Qwen2MoeModel.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.Qwen2MoeModel.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.Qwen2MoeModel.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.Qwen2MoeModel.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.Qwen2MoeModel.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.Qwen2MoeModel.forward.output_router_logits",description:`<strong>output_router_logits</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the logits of all the routers. They are useful for computing the router loss, and
should not be returned during inference.`,name:"output_router_logits"},{anchor:"transformers.Qwen2MoeModel.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/qwen2_moe/modeling_qwen2_moe.py#L792",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <code>transformers.modeling_outputs.MoeModelOutputWithPast</code> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/qwen2_moe#transformers.Qwen2MoeConfig"
>Qwen2MoeConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>last_hidden_state</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>) — Sequence of hidden-states at the output of the last layer of the model.</p>
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
<li>
<p><strong>router_logits</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_router_probs=True</code> and <code>config.add_router_probs=True</code> is passed or when <code>config.output_router_probs=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, sequence_length, num_experts)</code>.</p>
<p>Raw router logtis (post-softmax) that are computed by MoE routers, these terms are used to compute the auxiliary
loss for Mixture of Experts models.</p>
</li>
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>transformers.modeling_outputs.MoeModelOutputWithPast</code> or <code>tuple(torch.FloatTensor)</code></p>
`}}),O=new Ve({props:{$$slots:{default:[Mt]},$$scope:{ctx:y}}}),fe=new ze({props:{title:"Qwen2MoeForCausalLM",local:"transformers.Qwen2MoeForCausalLM",headingTag:"h2"}}),ge=new j({props:{name:"class transformers.Qwen2MoeForCausalLM",anchor:"transformers.Qwen2MoeForCausalLM",parameters:[{name:"config",val:""}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/qwen2_moe/modeling_qwen2_moe.py#L1050"}}),_e=new j({props:{name:"forward",anchor:"transformers.Qwen2MoeForCausalLM.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"output_router_logits",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"},{name:"logits_to_keep",val:": typing.Union[int, torch.Tensor] = 0"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.Qwen2MoeForCausalLM.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.Qwen2MoeForCausalLM.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.Qwen2MoeForCausalLM.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.Qwen2MoeForCausalLM.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.Qwen2MoeForCausalLM.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.Qwen2MoeForCausalLM.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.Qwen2MoeForCausalLM.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.Qwen2MoeForCausalLM.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.Qwen2MoeForCausalLM.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.Qwen2MoeForCausalLM.forward.output_router_logits",description:`<strong>output_router_logits</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the logits of all the routers. They are useful for computing the router loss, and
should not be returned during inference.`,name:"output_router_logits"},{anchor:"transformers.Qwen2MoeForCausalLM.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"},{anchor:"transformers.Qwen2MoeForCausalLM.forward.logits_to_keep",description:`<strong>logits_to_keep</strong> (<code>Union[int, torch.Tensor]</code>, defaults to <code>0</code>) &#x2014;
If an <code>int</code>, compute logits for the last <code>logits_to_keep</code> tokens. If <code>0</code>, calculate logits for all
<code>input_ids</code> (special case). Only last token logits are needed for generation, and calculating them only for that
token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
If a <code>torch.Tensor</code>, must be 1D corresponding to the indices to keep in the sequence length dimension.
This is useful when using packed tensor format (single dimension for batch and sequence length).`,name:"logits_to_keep"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/qwen2_moe/modeling_qwen2_moe.py#L1067",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <code>transformers.modeling_outputs.MoeCausalLMOutputWithPast</code> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/qwen2_moe#transformers.Qwen2MoeConfig"
>Qwen2MoeConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>loss</strong> (<code>torch.FloatTensor</code> of shape <code>(1,)</code>, <em>optional</em>, returned when <code>labels</code> is provided) — Language modeling loss (for next-token prediction).</p>
</li>
<li>
<p><strong>logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, config.vocab_size)</code>) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).</p>
</li>
<li>
<p><strong>aux_loss</strong> (<code>torch.FloatTensor</code>, <em>optional</em>, returned when <code>labels</code> is provided) — aux_loss for the sparse modules.</p>
</li>
<li>
<p><strong>router_logits</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_router_probs=True</code> and <code>config.add_router_probs=True</code> is passed or when <code>config.output_router_probs=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, sequence_length, num_experts)</code>.</p>
<p>Raw router logtis (post-softmax) that are computed by MoE routers, these terms are used to compute the auxiliary
loss for Mixture of Experts models.</p>
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


<p><code>transformers.modeling_outputs.MoeCausalLMOutputWithPast</code> or <code>tuple(torch.FloatTensor)</code></p>
`}}),D=new Ve({props:{$$slots:{default:[yt]},$$scope:{ctx:y}}}),Y=new st({props:{anchor:"transformers.Qwen2MoeForCausalLM.forward.example",$$slots:{default:[bt]},$$scope:{ctx:y}}}),we=new ze({props:{title:"Qwen2MoeForSequenceClassification",local:"transformers.Qwen2MoeForSequenceClassification",headingTag:"h2"}}),Me=new j({props:{name:"class transformers.Qwen2MoeForSequenceClassification",anchor:"transformers.Qwen2MoeForSequenceClassification",parameters:[{name:"config",val:""}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/qwen2_moe/modeling_qwen2_moe.py#L1161"}}),ye=new j({props:{name:"forward",anchor:"transformers.Qwen2MoeForSequenceClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.Qwen2MoeForSequenceClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.Qwen2MoeForSequenceClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.Qwen2MoeForSequenceClassification.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.Qwen2MoeForSequenceClassification.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.Qwen2MoeForSequenceClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.Qwen2MoeForSequenceClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.Qwen2MoeForSequenceClassification.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
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
`}}),K=new Ve({props:{$$slots:{default:[Tt]},$$scope:{ctx:y}}}),be=new ze({props:{title:"Qwen2MoeForTokenClassification",local:"transformers.Qwen2MoeForTokenClassification",headingTag:"h2"}}),Te=new j({props:{name:"class transformers.Qwen2MoeForTokenClassification",anchor:"transformers.Qwen2MoeForTokenClassification",parameters:[{name:"config",val:""}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/qwen2_moe/modeling_qwen2_moe.py#L1164"}}),ve=new j({props:{name:"forward",anchor:"transformers.Qwen2MoeForTokenClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.Qwen2MoeForTokenClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.Qwen2MoeForTokenClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.Qwen2MoeForTokenClassification.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.Qwen2MoeForTokenClassification.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.Qwen2MoeForTokenClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.Qwen2MoeForTokenClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.Qwen2MoeForTokenClassification.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
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
`}}),ee=new Ve({props:{$$slots:{default:[vt]},$$scope:{ctx:y}}}),ke=new ze({props:{title:"Qwen2MoeForQuestionAnswering",local:"transformers.Qwen2MoeForQuestionAnswering",headingTag:"h2"}}),$e=new j({props:{name:"class transformers.Qwen2MoeForQuestionAnswering",anchor:"transformers.Qwen2MoeForQuestionAnswering",parameters:[{name:"config",val:""}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/qwen2_moe/modeling_qwen2_moe.py#L1167"}}),xe=new j({props:{name:"forward",anchor:"transformers.Qwen2MoeForQuestionAnswering.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"start_positions",val:": typing.Optional[torch.LongTensor] = None"},{name:"end_positions",val:": typing.Optional[torch.LongTensor] = None"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.Qwen2MoeForQuestionAnswering.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.Qwen2MoeForQuestionAnswering.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.Qwen2MoeForQuestionAnswering.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.Qwen2MoeForQuestionAnswering.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.Qwen2MoeForQuestionAnswering.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.Qwen2MoeForQuestionAnswering.forward.start_positions",description:`<strong>start_positions</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for position (index) of the start of the labelled span for computing the token classification loss.
Positions are clamped to the length of the sequence (<code>sequence_length</code>). Position outside of the sequence
are not taken into account for computing the loss.`,name:"start_positions"},{anchor:"transformers.Qwen2MoeForQuestionAnswering.forward.end_positions",description:`<strong>end_positions</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
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
`}}),oe=new Ve({props:{$$slots:{default:[kt]},$$scope:{ctx:y}}}),Ce=new pt({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/qwen2_moe.md"}}),{c(){o=d("meta"),l=r(),t=d("p"),u=r(),v=d("p"),v.innerHTML=T,m=r(),k=d("div"),k.innerHTML=Xe,te=r(),h(q.$$.fragment),Se=r(),ne=d("p"),ne.innerHTML=Lo,Ge=r(),se=d("p"),se.textContent=Vo,Pe=r(),ae=d("p"),ae.innerHTML=No,Oe=r(),h(S.$$.fragment),De=r(),re=d("p"),re.innerHTML=Xo,Ye=r(),h(G.$$.fragment),Ke=r(),ie=d("p"),ie.innerHTML=Ho,eo=r(),le=d("p"),le.innerHTML=So,oo=r(),h(de.$$.fragment),to=r(),h(ce.$$.fragment),no=r(),x=d("div"),h(pe.$$.fragment),Mo=r(),Fe=d("p"),Fe.innerHTML=Go,yo=r(),je=d("p"),je.innerHTML=Po,bo=r(),h(P.$$.fragment),so=r(),h(ue.$$.fragment),ao=r(),$=d("div"),h(me.$$.fragment),To=r(),Ie=d("p"),Ie.textContent=Oo,vo=r(),qe=d("p"),qe.innerHTML=Do,ko=r(),Ze=d("p"),Ze.innerHTML=Yo,$o=r(),Z=d("div"),h(he.$$.fragment),xo=r(),Be=d("p"),Be.innerHTML=Ko,Co=r(),h(O.$$.fragment),ro=r(),h(fe.$$.fragment),io=r(),R=d("div"),h(ge.$$.fragment),Qo=r(),J=d("div"),h(_e.$$.fragment),Uo=r(),Ee=d("p"),Ee.innerHTML=et,Jo=r(),h(D.$$.fragment),zo=r(),h(Y.$$.fragment),lo=r(),h(we.$$.fragment),co=r(),A=d("div"),h(Me.$$.fragment),Fo=r(),B=d("div"),h(ye.$$.fragment),jo=r(),We=d("p"),We.innerHTML=ot,Io=r(),h(K.$$.fragment),po=r(),h(be.$$.fragment),uo=r(),L=d("div"),h(Te.$$.fragment),qo=r(),E=d("div"),h(ve.$$.fragment),Zo=r(),Re=d("p"),Re.innerHTML=tt,Bo=r(),h(ee.$$.fragment),mo=r(),h(ke.$$.fragment),ho=r(),V=d("div"),h($e.$$.fragment),Eo=r(),W=d("div"),h(xe.$$.fragment),Wo=r(),Ae=d("p"),Ae.innerHTML=nt,Ro=r(),h(oe.$$.fragment),fo=r(),h(Ce.$$.fragment),go=r(),He=d("p"),this.h()},l(e){const n=dt("svelte-u9bgzb",document.head);o=c(n,"META",{name:!0,content:!0}),n.forEach(s),l=i(e),t=c(e,"P",{}),Q(t).forEach(s),u=i(e),v=c(e,"P",{"data-svelte-h":!0}),b(v)!=="svelte-omfmcr"&&(v.innerHTML=T),m=i(e),k=c(e,"DIV",{style:!0,"data-svelte-h":!0}),b(k)!=="svelte-1x9sw1l"&&(k.innerHTML=Xe),te=i(e),f(q.$$.fragment,e),Se=i(e),ne=c(e,"P",{"data-svelte-h":!0}),b(ne)!=="svelte-1ata8tf"&&(ne.innerHTML=Lo),Ge=i(e),se=c(e,"P",{"data-svelte-h":!0}),b(se)!=="svelte-1ikegs1"&&(se.textContent=Vo),Pe=i(e),ae=c(e,"P",{"data-svelte-h":!0}),b(ae)!=="svelte-zksxg"&&(ae.innerHTML=No),Oe=i(e),f(S.$$.fragment,e),De=i(e),re=c(e,"P",{"data-svelte-h":!0}),b(re)!=="svelte-17pa8jt"&&(re.innerHTML=Xo),Ye=i(e),f(G.$$.fragment,e),Ke=i(e),ie=c(e,"P",{"data-svelte-h":!0}),b(ie)!=="svelte-nf5ooi"&&(ie.innerHTML=Ho),eo=i(e),le=c(e,"P",{"data-svelte-h":!0}),b(le)!=="svelte-u672qo"&&(le.innerHTML=So),oo=i(e),f(de.$$.fragment,e),to=i(e),f(ce.$$.fragment,e),no=i(e),x=c(e,"DIV",{class:!0});var z=Q(x);f(pe.$$.fragment,z),Mo=i(z),Fe=c(z,"P",{"data-svelte-h":!0}),b(Fe)!=="svelte-146lutg"&&(Fe.innerHTML=Go),yo=i(z),je=c(z,"P",{"data-svelte-h":!0}),b(je)!=="svelte-1ek1ss9"&&(je.innerHTML=Po),bo=i(z),f(P.$$.fragment,z),z.forEach(s),so=i(e),f(ue.$$.fragment,e),ao=i(e),$=c(e,"DIV",{class:!0});var C=Q($);f(me.$$.fragment,C),To=i(C),Ie=c(C,"P",{"data-svelte-h":!0}),b(Ie)!=="svelte-g186hq"&&(Ie.textContent=Oo),vo=i(C),qe=c(C,"P",{"data-svelte-h":!0}),b(qe)!=="svelte-q52n56"&&(qe.innerHTML=Do),ko=i(C),Ze=c(C,"P",{"data-svelte-h":!0}),b(Ze)!=="svelte-hswkmf"&&(Ze.innerHTML=Yo),$o=i(C),Z=c(C,"DIV",{class:!0});var N=Q(Z);f(he.$$.fragment,N),xo=i(N),Be=c(N,"P",{"data-svelte-h":!0}),b(Be)!=="svelte-1pdgd6u"&&(Be.innerHTML=Ko),Co=i(N),f(O.$$.fragment,N),N.forEach(s),C.forEach(s),ro=i(e),f(fe.$$.fragment,e),io=i(e),R=c(e,"DIV",{class:!0});var Qe=Q(R);f(ge.$$.fragment,Qe),Qo=i(Qe),J=c(Qe,"DIV",{class:!0});var F=Q(J);f(_e.$$.fragment,F),Uo=i(F),Ee=c(F,"P",{"data-svelte-h":!0}),b(Ee)!=="svelte-nkhtwi"&&(Ee.innerHTML=et),Jo=i(F),f(D.$$.fragment,F),zo=i(F),f(Y.$$.fragment,F),F.forEach(s),Qe.forEach(s),lo=i(e),f(we.$$.fragment,e),co=i(e),A=c(e,"DIV",{class:!0});var Ue=Q(A);f(Me.$$.fragment,Ue),Fo=i(Ue),B=c(Ue,"DIV",{class:!0});var X=Q(B);f(ye.$$.fragment,X),jo=i(X),We=c(X,"P",{"data-svelte-h":!0}),b(We)!=="svelte-1sal4ui"&&(We.innerHTML=ot),Io=i(X),f(K.$$.fragment,X),X.forEach(s),Ue.forEach(s),po=i(e),f(be.$$.fragment,e),uo=i(e),L=c(e,"DIV",{class:!0});var Je=Q(L);f(Te.$$.fragment,Je),qo=i(Je),E=c(Je,"DIV",{class:!0});var H=Q(E);f(ve.$$.fragment,H),Zo=i(H),Re=c(H,"P",{"data-svelte-h":!0}),b(Re)!=="svelte-1py4aay"&&(Re.innerHTML=tt),Bo=i(H),f(ee.$$.fragment,H),H.forEach(s),Je.forEach(s),mo=i(e),f(ke.$$.fragment,e),ho=i(e),V=c(e,"DIV",{class:!0});var wo=Q(V);f($e.$$.fragment,wo),Eo=i(wo),W=c(wo,"DIV",{class:!0});var Le=Q(W);f(xe.$$.fragment,Le),Wo=i(Le),Ae=c(Le,"P",{"data-svelte-h":!0}),b(Ae)!=="svelte-dyrov9"&&(Ae.innerHTML=nt),Ro=i(Le),f(oe.$$.fragment,Le),Le.forEach(s),wo.forEach(s),fo=i(e),f(Ce.$$.fragment,e),go=i(e),He=c(e,"P",{}),Q(He).forEach(s),this.h()},h(){U(o,"name","hf:doc:metadata"),U(o,"content",xt),ct(k,"float","right"),U(x,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),U(Z,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),U($,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),U(J,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),U(R,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),U(B,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),U(A,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),U(E,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),U(L,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),U(W,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),U(V,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,n){p(document.head,o),a(e,l,n),a(e,t,n),a(e,u,n),a(e,v,n),a(e,m,n),a(e,k,n),a(e,te,n),g(q,e,n),a(e,Se,n),a(e,ne,n),a(e,Ge,n),a(e,se,n),a(e,Pe,n),a(e,ae,n),a(e,Oe,n),g(S,e,n),a(e,De,n),a(e,re,n),a(e,Ye,n),g(G,e,n),a(e,Ke,n),a(e,ie,n),a(e,eo,n),a(e,le,n),a(e,oo,n),g(de,e,n),a(e,to,n),g(ce,e,n),a(e,no,n),a(e,x,n),g(pe,x,null),p(x,Mo),p(x,Fe),p(x,yo),p(x,je),p(x,bo),g(P,x,null),a(e,so,n),g(ue,e,n),a(e,ao,n),a(e,$,n),g(me,$,null),p($,To),p($,Ie),p($,vo),p($,qe),p($,ko),p($,Ze),p($,$o),p($,Z),g(he,Z,null),p(Z,xo),p(Z,Be),p(Z,Co),g(O,Z,null),a(e,ro,n),g(fe,e,n),a(e,io,n),a(e,R,n),g(ge,R,null),p(R,Qo),p(R,J),g(_e,J,null),p(J,Uo),p(J,Ee),p(J,Jo),g(D,J,null),p(J,zo),g(Y,J,null),a(e,lo,n),g(we,e,n),a(e,co,n),a(e,A,n),g(Me,A,null),p(A,Fo),p(A,B),g(ye,B,null),p(B,jo),p(B,We),p(B,Io),g(K,B,null),a(e,po,n),g(be,e,n),a(e,uo,n),a(e,L,n),g(Te,L,null),p(L,qo),p(L,E),g(ve,E,null),p(E,Zo),p(E,Re),p(E,Bo),g(ee,E,null),a(e,mo,n),g(ke,e,n),a(e,ho,n),a(e,V,n),g($e,V,null),p(V,Eo),p(V,W),g(xe,W,null),p(W,Wo),p(W,Ae),p(W,Ro),g(oe,W,null),a(e,fo,n),g(Ce,e,n),a(e,go,n),a(e,He,n),_o=!0},p(e,[n]){const z={};n&2&&(z.$$scope={dirty:n,ctx:e}),S.$set(z);const C={};n&2&&(C.$$scope={dirty:n,ctx:e}),G.$set(C);const N={};n&2&&(N.$$scope={dirty:n,ctx:e}),P.$set(N);const Qe={};n&2&&(Qe.$$scope={dirty:n,ctx:e}),O.$set(Qe);const F={};n&2&&(F.$$scope={dirty:n,ctx:e}),D.$set(F);const Ue={};n&2&&(Ue.$$scope={dirty:n,ctx:e}),Y.$set(Ue);const X={};n&2&&(X.$$scope={dirty:n,ctx:e}),K.$set(X);const Je={};n&2&&(Je.$$scope={dirty:n,ctx:e}),ee.$set(Je);const H={};n&2&&(H.$$scope={dirty:n,ctx:e}),oe.$set(H)},i(e){_o||(_(q.$$.fragment,e),_(S.$$.fragment,e),_(G.$$.fragment,e),_(de.$$.fragment,e),_(ce.$$.fragment,e),_(pe.$$.fragment,e),_(P.$$.fragment,e),_(ue.$$.fragment,e),_(me.$$.fragment,e),_(he.$$.fragment,e),_(O.$$.fragment,e),_(fe.$$.fragment,e),_(ge.$$.fragment,e),_(_e.$$.fragment,e),_(D.$$.fragment,e),_(Y.$$.fragment,e),_(we.$$.fragment,e),_(Me.$$.fragment,e),_(ye.$$.fragment,e),_(K.$$.fragment,e),_(be.$$.fragment,e),_(Te.$$.fragment,e),_(ve.$$.fragment,e),_(ee.$$.fragment,e),_(ke.$$.fragment,e),_($e.$$.fragment,e),_(xe.$$.fragment,e),_(oe.$$.fragment,e),_(Ce.$$.fragment,e),_o=!0)},o(e){w(q.$$.fragment,e),w(S.$$.fragment,e),w(G.$$.fragment,e),w(de.$$.fragment,e),w(ce.$$.fragment,e),w(pe.$$.fragment,e),w(P.$$.fragment,e),w(ue.$$.fragment,e),w(me.$$.fragment,e),w(he.$$.fragment,e),w(O.$$.fragment,e),w(fe.$$.fragment,e),w(ge.$$.fragment,e),w(_e.$$.fragment,e),w(D.$$.fragment,e),w(Y.$$.fragment,e),w(we.$$.fragment,e),w(Me.$$.fragment,e),w(ye.$$.fragment,e),w(K.$$.fragment,e),w(be.$$.fragment,e),w(Te.$$.fragment,e),w(ve.$$.fragment,e),w(ee.$$.fragment,e),w(ke.$$.fragment,e),w($e.$$.fragment,e),w(xe.$$.fragment,e),w(oe.$$.fragment,e),w(Ce.$$.fragment,e),_o=!1},d(e){e&&(s(l),s(t),s(u),s(v),s(m),s(k),s(te),s(Se),s(ne),s(Ge),s(se),s(Pe),s(ae),s(Oe),s(De),s(re),s(Ye),s(Ke),s(ie),s(eo),s(le),s(oo),s(to),s(no),s(x),s(so),s(ao),s($),s(ro),s(io),s(R),s(lo),s(co),s(A),s(po),s(uo),s(L),s(mo),s(ho),s(V),s(fo),s(go),s(He)),s(o),M(q,e),M(S,e),M(G,e),M(de,e),M(ce,e),M(pe),M(P),M(ue,e),M(me),M(he),M(O),M(fe,e),M(ge),M(_e),M(D),M(Y),M(we,e),M(Me),M(ye),M(K),M(be,e),M(Te),M(ve),M(ee),M(ke,e),M($e),M(xe),M(oe),M(Ce,e)}}}const xt='{"title":"Qwen2MoE","local":"qwen2moe","sections":[{"title":"Qwen2MoeConfig","local":"transformers.Qwen2MoeConfig","sections":[],"depth":2},{"title":"Qwen2MoeModel","local":"transformers.Qwen2MoeModel","sections":[],"depth":2},{"title":"Qwen2MoeForCausalLM","local":"transformers.Qwen2MoeForCausalLM","sections":[],"depth":2},{"title":"Qwen2MoeForSequenceClassification","local":"transformers.Qwen2MoeForSequenceClassification","sections":[],"depth":2},{"title":"Qwen2MoeForTokenClassification","local":"transformers.Qwen2MoeForTokenClassification","sections":[],"depth":2},{"title":"Qwen2MoeForQuestionAnswering","local":"transformers.Qwen2MoeForQuestionAnswering","sections":[],"depth":2}],"depth":1}';function Ct(y){return rt(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class Zt extends it{constructor(o){super(),lt(this,o,Ct,$t,at,{})}}export{Zt as component};
