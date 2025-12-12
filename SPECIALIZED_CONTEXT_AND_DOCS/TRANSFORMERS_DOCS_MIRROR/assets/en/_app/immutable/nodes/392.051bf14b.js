import{s as ut,o as ht,n as q}from"../chunks/scheduler.18a86fab.js";import{S as ft,i as gt,g as c,s as a,r as m,A as _t,h as p,f as s,c as r,j as J,x as b,u,k as z,l as Mt,y as d,a as i,v as h,d as f,t as g,w as _}from"../chunks/index.98837b22.js";import{T as Ge}from"../chunks/Tip.77304350.js";import{D as I}from"../chunks/Docstring.a1ef7999.js";import{C as Qe}from"../chunks/CodeBlock.8d0c2e8a.js";import{E as mt}from"../chunks/ExampleCodeBlock.8c3ee1f9.js";import{H as ne,E as yt}from"../chunks/getInferenceSnippets.06c2775f.js";import{H as bt,a as Qo}from"../chunks/HfOption.6641485e.js";function Tt(T){let o,l="Click on the SmolLM3 models in the right sidebar for more examples of how to apply SmolLM3 to different language tasks.";return{c(){o=c("p"),o.textContent=l},l(t){o=p(t,"P",{"data-svelte-h":!0}),b(o)!=="svelte-c55jhp"&&(o.textContent=l)},m(t,M){i(t,o,M)},p:q,d(t){t&&s(o)}}}function wt(T){let o,l;return o=new Qe({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwcGlwZWxpbmUlMEElMEFwaXBlJTIwJTNEJTIwcGlwZWxpbmUoJTBBJTIwJTIwJTIwJTIwdGFzayUzRCUyMnRleHQtZ2VuZXJhdGlvbiUyMiUyQyUwQSUyMCUyMCUyMCUyMG1vZGVsJTNEJTIySHVnZ2luZ0ZhY2VUQiUyRlNtb2xMTTMtM0IlMjIlMkMlMEElMjAlMjAlMjAlMjBkdHlwZSUzRHRvcmNoLmJmbG9hdDE2JTJDJTBBJTIwJTIwJTIwJTIwZGV2aWNlX21hcCUzRDAlMEEpJTBBJTBBbWVzc2FnZXMlMjAlM0QlMjAlNUIlMEElMjAlMjAlMjAlMjAlN0IlMjJyb2xlJTIyJTNBJTIwJTIyc3lzdGVtJTIyJTJDJTIwJTIyY29udGVudCUyMiUzQSUyMCUyMllvdSUyMGFyZSUyMGElMjBoZWxwZnVsJTIwYXNzaXN0YW50LiUyMiU3RCUyQyUwQSUyMCUyMCUyMCUyMCU3QiUyMnJvbGUlMjIlM0ElMjAlMjJ1c2VyJTIyJTJDJTIwJTIyY29udGVudCUyMiUzQSUyMCUyMlRlbGwlMjBtZSUyMGFib3V0JTIweW91cnNlbGYuJTIyJTdEJTJDJTBBJTVEJTBBb3V0cHV0cyUyMCUzRCUyMHBpcGUobWVzc2FnZXMlMkMlMjBtYXhfbmV3X3Rva2VucyUzRDI1NiUyQyUyMGRvX3NhbXBsZSUzRFRydWUlMkMlMjB0ZW1wZXJhdHVyZSUzRDAuNyUyQyUyMHRvcF9rJTNENTAlMkMlMjB0b3BfcCUzRDAuOTUpJTBBcHJpbnQob3V0cHV0cyU1QjAlNUQlNUIlMjJnZW5lcmF0ZWRfdGV4dCUyMiU1RCU1Qi0xJTVEJTVCJ2NvbnRlbnQnJTVEKQ==",highlighted:`<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> pipeline

pipe = pipeline(
    task=<span class="hljs-string">&quot;text-generation&quot;</span>,
    model=<span class="hljs-string">&quot;HuggingFaceTB/SmolLM3-3B&quot;</span>,
    dtype=torch.bfloat16,
    device_map=<span class="hljs-number">0</span>
)

messages = [
    {<span class="hljs-string">&quot;role&quot;</span>: <span class="hljs-string">&quot;system&quot;</span>, <span class="hljs-string">&quot;content&quot;</span>: <span class="hljs-string">&quot;You are a helpful assistant.&quot;</span>},
    {<span class="hljs-string">&quot;role&quot;</span>: <span class="hljs-string">&quot;user&quot;</span>, <span class="hljs-string">&quot;content&quot;</span>: <span class="hljs-string">&quot;Tell me about yourself.&quot;</span>},
]
outputs = pipe(messages, max_new_tokens=<span class="hljs-number">256</span>, do_sample=<span class="hljs-literal">True</span>, temperature=<span class="hljs-number">0.7</span>, top_k=<span class="hljs-number">50</span>, top_p=<span class="hljs-number">0.95</span>)
<span class="hljs-built_in">print</span>(outputs[<span class="hljs-number">0</span>][<span class="hljs-string">&quot;generated_text&quot;</span>][-<span class="hljs-number">1</span>][<span class="hljs-string">&#x27;content&#x27;</span>])`,wrap:!1}}),{c(){m(o.$$.fragment)},l(t){u(o.$$.fragment,t)},m(t,M){h(o,t,M),l=!0},p:q,i(t){l||(f(o.$$.fragment,t),l=!0)},o(t){g(o.$$.fragment,t),l=!1},d(t){_(o,t)}}}function vt(T){let o,l;return o=new Qe({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b01vZGVsRm9yQ2F1c2FsTE0lMkMlMjBBdXRvVG9rZW5pemVyJTBBJTBBbW9kZWwlMjAlM0QlMjBBdXRvTW9kZWxGb3JDYXVzYWxMTS5mcm9tX3ByZXRyYWluZWQoJTBBJTIwJTIwJTIwJTIwJTIySHVnZ2luZ0ZhY2VUQiUyRlNtb2xMTTMtM0IlMjIlMkMlMEElMjAlMjAlMjAlMjBkdHlwZSUzRHRvcmNoLmJmbG9hdDE2JTJDJTBBJTIwJTIwJTIwJTIwZGV2aWNlX21hcCUzRCUyMmF1dG8lMjIlMkMlMEElMjAlMjAlMjAlMjBhdHRuX2ltcGxlbWVudGF0aW9uJTNEJTIyc2RwYSUyMiUwQSklMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJIdWdnaW5nRmFjZVRCJTJGU21vbExNMy0zQiUyMiklMEElMEFwcm9tcHQlMjAlM0QlMjAlMjJHaXZlJTIwbWUlMjBhJTIwc2hvcnQlMjBpbnRyb2R1Y3Rpb24lMjB0byUyMGxhcmdlJTIwbGFuZ3VhZ2UlMjBtb2RlbHMuJTIyJTBBbWVzc2FnZXMlMjAlM0QlMjAlNUIlMEElMjAlMjAlMjAlMjAlN0IlMjJyb2xlJTIyJTNBJTIwJTIyc3lzdGVtJTIyJTJDJTIwJTIyY29udGVudCUyMiUzQSUyMCUyMllvdSUyMGFyZSUyMGElMjBoZWxwZnVsJTIwYXNzaXN0YW50LiUyMiU3RCUyQyUwQSUyMCUyMCUyMCUyMCU3QiUyMnJvbGUlMjIlM0ElMjAlMjJ1c2VyJTIyJTJDJTIwJTIyY29udGVudCUyMiUzQSUyMHByb21wdCU3RCUwQSU1RCUwQXRleHQlMjAlM0QlMjB0b2tlbml6ZXIuYXBwbHlfY2hhdF90ZW1wbGF0ZSglMEElMjAlMjAlMjAlMjBtZXNzYWdlcyUyQyUwQSUyMCUyMCUyMCUyMHRva2VuaXplJTNERmFsc2UlMkMlMEElMjAlMjAlMjAlMjBhZGRfZ2VuZXJhdGlvbl9wcm9tcHQlM0RUcnVlJTBBKSUwQW1vZGVsX2lucHV0cyUyMCUzRCUyMHRva2VuaXplciglNUJ0ZXh0JTVEJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMikudG8obW9kZWwuZGV2aWNlKSUwQSUwQWdlbmVyYXRlZF9pZHMlMjAlM0QlMjBtb2RlbC5nZW5lcmF0ZSglMEElMjAlMjAlMjAlMjBtb2RlbF9pbnB1dHMuaW5wdXRfaWRzJTJDJTBBJTIwJTIwJTIwJTIwY2FjaGVfaW1wbGVtZW50YXRpb24lM0QlMjJzdGF0aWMlMjIlMkMlMEElMjAlMjAlMjAlMjBtYXhfbmV3X3Rva2VucyUzRDUxMiUyQyUwQSUyMCUyMCUyMCUyMGRvX3NhbXBsZSUzRFRydWUlMkMlMEElMjAlMjAlMjAlMjB0ZW1wZXJhdHVyZSUzRDAuNyUyQyUwQSUyMCUyMCUyMCUyMHRvcF9rJTNENTAlMkMlMEElMjAlMjAlMjAlMjB0b3BfcCUzRDAuOTUlMEEpJTBBZ2VuZXJhdGVkX2lkcyUyMCUzRCUyMCU1QiUwQSUyMCUyMCUyMCUyMG91dHB1dF9pZHMlNUJsZW4oaW5wdXRfaWRzKSUzQSU1RCUyMGZvciUyMGlucHV0X2lkcyUyQyUyMG91dHB1dF9pZHMlMjBpbiUyMHppcChtb2RlbF9pbnB1dHMuaW5wdXRfaWRzJTJDJTIwZ2VuZXJhdGVkX2lkcyklMEElNUQlMEElMEFyZXNwb25zZSUyMCUzRCUyMHRva2VuaXplci5iYXRjaF9kZWNvZGUoZ2VuZXJhdGVkX2lkcyUyQyUyMHNraXBfc3BlY2lhbF90b2tlbnMlM0RUcnVlKSU1QjAlNUQlMEFwcmludChyZXNwb25zZSk=",highlighted:`<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    <span class="hljs-string">&quot;HuggingFaceTB/SmolLM3-3B&quot;</span>,
    dtype=torch.bfloat16,
    device_map=<span class="hljs-string">&quot;auto&quot;</span>,
    attn_implementation=<span class="hljs-string">&quot;sdpa&quot;</span>
)
tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;HuggingFaceTB/SmolLM3-3B&quot;</span>)

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
<span class="hljs-built_in">print</span>(response)`,wrap:!1}}),{c(){m(o.$$.fragment)},l(t){u(o.$$.fragment,t)},m(t,M){h(o,t,M),l=!0},p:q,i(t){l||(f(o.$$.fragment,t),l=!0)},o(t){g(o.$$.fragment,t),l=!1},d(t){_(o,t)}}}function kt(T){let o,l;return o=new Qe({props:{code:"JTIzJTIwcGlwJTIwaW5zdGFsbCUyMC1VJTIwZmxhc2gtYXR0biUyMC0tbm8tYnVpbGQtaXNvbGF0aW9uJTBBdHJhbnNmb3JtZXJzJTIwY2hhdCUyMEh1Z2dpbmdGYWNlVEIlMkZTbW9sTE0zLTNCJTIwLS1kdHlwZSUyMGF1dG8lMjAtLWF0dG5faW1wbGVtZW50YXRpb24lMjBmbGFzaF9hdHRlbnRpb25fMiUyMC0tZGV2aWNlJTIwMA==",highlighted:`<span class="hljs-comment"># pip install -U flash-attn --no-build-isolation</span>
transformers chat HuggingFaceTB/SmolLM3-3B --dtype auto --attn_implementation flash_attention_2 --device 0`,wrap:!1}}),{c(){m(o.$$.fragment)},l(t){u(o.$$.fragment,t)},m(t,M){h(o,t,M),l=!0},p:q,i(t){l||(f(o.$$.fragment,t),l=!0)},o(t){g(o.$$.fragment,t),l=!1},d(t){_(o,t)}}}function $t(T){let o,l,t,M,v,w;return o=new Qo({props:{id:"usage",option:"Pipeline",$$slots:{default:[wt]},$$scope:{ctx:T}}}),t=new Qo({props:{id:"usage",option:"AutoModel",$$slots:{default:[vt]},$$scope:{ctx:T}}}),v=new Qo({props:{id:"usage",option:"transformers CLI",$$slots:{default:[kt]},$$scope:{ctx:T}}}),{c(){m(o.$$.fragment),l=a(),m(t.$$.fragment),M=a(),m(v.$$.fragment)},l(y){u(o.$$.fragment,y),l=r(y),u(t.$$.fragment,y),M=r(y),u(v.$$.fragment,y)},m(y,k){h(o,y,k),i(y,l,k),h(t,y,k),i(y,M,k),h(v,y,k),w=!0},p(y,k){const Xe={};k&2&&(Xe.$$scope={dirty:k,ctx:y}),o.$set(Xe);const se={};k&2&&(se.$$scope={dirty:k,ctx:y}),t.$set(se);const B={};k&2&&(B.$$scope={dirty:k,ctx:y}),v.$set(B)},i(y){w||(f(o.$$.fragment,y),f(t.$$.fragment,y),f(v.$$.fragment,y),w=!0)},o(y){g(o.$$.fragment,y),g(t.$$.fragment,y),g(v.$$.fragment,y),w=!1},d(y){y&&(s(l),s(M)),_(o,y),_(t,y),_(v,y)}}}function Ct(T){let o,l;return o=new Qe({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMFNtb2xMTTNNb2RlbCUyQyUyMFNtb2xMTTNDb25maWclMEElMEElMjMlMjBJbml0aWFsaXppbmclMjBhJTIwU21vbExNMyUyMHN0eWxlJTIwY29uZmlndXJhdGlvbiUwQWNvbmZpZ3VyYXRpb24lMjAlM0QlMjBTbW9sTE0zQ29uZmlnKCklMEElMEElMjMlMjBJbml0aWFsaXppbmclMjBhJTIwbW9kZWwlMjBmcm9tJTIwdGhlJTIwU21vbExNMyUyMHN0eWxlJTIwY29uZmlndXJhdGlvbiUwQW1vZGVsJTIwJTNEJTIwU21vbExNM01vZGVsKGNvbmZpZ3VyYXRpb24pJTBBJTBBJTIzJTIwQWNjZXNzaW5nJTIwdGhlJTIwbW9kZWwlMjBjb25maWd1cmF0aW9uJTBBY29uZmlndXJhdGlvbiUyMCUzRCUyMG1vZGVsLmNvbmZpZw==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> SmolLM3Model, SmolLM3Config

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a SmolLM3 style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = SmolLM3Config()

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a model from the SmolLM3 style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = SmolLM3Model(configuration)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Accessing the model configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = model.config`,wrap:!1}}),{c(){m(o.$$.fragment)},l(t){u(o.$$.fragment,t)},m(t,M){h(o,t,M),l=!0},p:q,i(t){l||(f(o.$$.fragment,t),l=!0)},o(t){g(o.$$.fragment,t),l=!1},d(t){_(o,t)}}}function Lt(T){let o,l=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){o=c("p"),o.innerHTML=l},l(t){o=p(t,"P",{"data-svelte-h":!0}),b(o)!=="svelte-fincs2"&&(o.innerHTML=l)},m(t,M){i(t,o,M)},p:q,d(t){t&&s(o)}}}function xt(T){let o,l=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){o=c("p"),o.innerHTML=l},l(t){o=p(t,"P",{"data-svelte-h":!0}),b(o)!=="svelte-fincs2"&&(o.innerHTML=l)},m(t,M){i(t,o,M)},p:q,d(t){t&&s(o)}}}function Ut(T){let o,l="Example:",t,M,v;return M=new Qe({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBTbW9sTE0zRm9yQ2F1c2FsTE0lMEElMEFtb2RlbCUyMCUzRCUyMFNtb2xMTTNGb3JDYXVzYWxMTS5mcm9tX3ByZXRyYWluZWQoJTIybWV0YS1zbW9sbG0zJTJGU21vbExNMy0yLTdiLWhmJTIyKSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMm1ldGEtc21vbGxtMyUyRlNtb2xMTTMtMi03Yi1oZiUyMiklMEElMEFwcm9tcHQlMjAlM0QlMjAlMjJIZXklMkMlMjBhcmUlMjB5b3UlMjBjb25zY2lvdXMlM0YlMjBDYW4lMjB5b3UlMjB0YWxrJTIwdG8lMjBtZSUzRiUyMiUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplcihwcm9tcHQlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQSUwQSUyMyUyMEdlbmVyYXRlJTBBZ2VuZXJhdGVfaWRzJTIwJTNEJTIwbW9kZWwuZ2VuZXJhdGUoaW5wdXRzLmlucHV0X2lkcyUyQyUyMG1heF9sZW5ndGglM0QzMCklMEF0b2tlbml6ZXIuYmF0Y2hfZGVjb2RlKGdlbmVyYXRlX2lkcyUyQyUyMHNraXBfc3BlY2lhbF90b2tlbnMlM0RUcnVlJTJDJTIwY2xlYW5fdXBfdG9rZW5pemF0aW9uX3NwYWNlcyUzREZhbHNlKSU1QjAlNUQ=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, SmolLM3ForCausalLM

<span class="hljs-meta">&gt;&gt;&gt; </span>model = SmolLM3ForCausalLM.from_pretrained(<span class="hljs-string">&quot;meta-smollm3/SmolLM3-2-7b-hf&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;meta-smollm3/SmolLM3-2-7b-hf&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>prompt = <span class="hljs-string">&quot;Hey, are you conscious? Can you talk to me?&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(prompt, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Generate</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>generate_ids = model.generate(inputs.input_ids, max_length=<span class="hljs-number">30</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer.batch_decode(generate_ids, skip_special_tokens=<span class="hljs-literal">True</span>, clean_up_tokenization_spaces=<span class="hljs-literal">False</span>)[<span class="hljs-number">0</span>]
<span class="hljs-string">&quot;Hey, are you conscious? Can you talk to me?\\nI&#x27;m not conscious, but I can talk to you.&quot;</span>`,wrap:!1}}),{c(){o=c("p"),o.textContent=l,t=a(),m(M.$$.fragment)},l(w){o=p(w,"P",{"data-svelte-h":!0}),b(o)!=="svelte-11lpom8"&&(o.textContent=l),t=r(w),u(M.$$.fragment,w)},m(w,y){i(w,o,y),i(w,t,y),h(M,w,y),v=!0},p:q,i(w){v||(f(M.$$.fragment,w),v=!0)},o(w){g(M.$$.fragment,w),v=!1},d(w){w&&(s(o),s(t)),_(M,w)}}}function Jt(T){let o,l=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){o=c("p"),o.innerHTML=l},l(t){o=p(t,"P",{"data-svelte-h":!0}),b(o)!=="svelte-fincs2"&&(o.innerHTML=l)},m(t,M){i(t,o,M)},p:q,d(t){t&&s(o)}}}function zt(T){let o,l=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){o=c("p"),o.innerHTML=l},l(t){o=p(t,"P",{"data-svelte-h":!0}),b(o)!=="svelte-fincs2"&&(o.innerHTML=l)},m(t,M){i(t,o,M)},p:q,d(t){t&&s(o)}}}function St(T){let o,l=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){o=c("p"),o.innerHTML=l},l(t){o=p(t,"P",{"data-svelte-h":!0}),b(o)!=="svelte-fincs2"&&(o.innerHTML=l)},m(t,M){i(t,o,M)},p:q,d(t){t&&s(o)}}}function jt(T){let o,l,t,M,v,w="<em>This model was released on 2025-07-08 and added to Hugging Face Transformers on 2025-06-25.</em>",y,k,Xe='<div class="flex flex-wrap space-x-1"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/> <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat"/> <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/></div>',se,B,De,ae,Xo='<a href="https://huggingface.co/blog/smollm3" rel="nofollow">SmolLM3</a> is a fully open, compact language model designed for efficient deployment while maintaining strong performance. It uses a Transformer decoder architecture with Grouped Query Attention (GQA) to reduce the kv cache, and no RoPE, enabling improved performance on long-context tasks. It is trained using a multi-stage training approach on high-quality public datasets across web, code, and math domains. The model is multilingual and supports very large context lengths. The instruct variant is optimized for reasoning and tool use.',Oe,X,Ye,re,Po='The example below demonstrates how to generate text with <a href="/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.Pipeline">Pipeline</a>, <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel">AutoModel</a>, and from the command line using the instruction-tuned models.',Ke,P,eo,ie,Do='Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the <a href="../quantization/overview">Quantization</a> overview for more available quantization backends.',oo,le,Oo='The example below uses <a href="../quantization/bitsandbytes">bitsandbytes</a> to quantize the weights to 4-bits.',to,de,no,ce,so,pe,Yo="<li>Ensure your Transformers library version is up-to-date. SmolLM3 requires Transformers&gt;=4.53.0 for full support.</li>",ao,me,ro,L,ue,wo,Se,Ko=`This is the configuration class to store the configuration of a <a href="/docs/transformers/v4.56.2/en/model_doc/smollm3#transformers.SmolLM3Model">SmolLM3Model</a>. It is used to instantiate a
SmolLM3 model according to the specified arguments, defining the model architecture. Instantiating a configuration
with the defaults will yield a similar configuration to that of the SmolLM3 3B.
e.g. <a href="https://huggingface.co/HuggingFaceTB/SmolLM3-3B" rel="nofollow">HuggingFaceTB/SmolLM3-3B</a>`,vo,je,et=`Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> and can be used to control the model outputs. Read the
documentation from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> for more information.`,ko,D,io,he,lo,$,fe,$o,Fe,ot="The bare Smollm3 Model outputting raw hidden-states without any specific head on top.",Co,Ie,tt=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Lo,qe,nt=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,xo,Z,ge,Uo,Be,st='The <a href="/docs/transformers/v4.56.2/en/model_doc/smollm3#transformers.SmolLM3Model">SmolLM3Model</a> forward method, overrides the <code>__call__</code> special method.',Jo,O,co,_e,po,C,Me,zo,Ze,at="The Smollm3 Model for causal language modeling.",So,We,rt=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,jo,Ne,it=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Fo,S,ye,Io,Ee,lt='The <a href="/docs/transformers/v4.56.2/en/model_doc/smollm3#transformers.SmolLM3ForCausalLM">SmolLM3ForCausalLM</a> forward method, overrides the <code>__call__</code> special method.',qo,Y,Bo,K,mo,be,uo,A,Te,Zo,W,we,Wo,Ae,dt="The <code>GenericForSequenceClassification</code> forward method, overrides the <code>__call__</code> special method.",No,ee,ho,ve,fo,V,ke,Eo,N,$e,Ao,Ve,ct="The <code>GenericForTokenClassification</code> forward method, overrides the <code>__call__</code> special method.",Vo,oe,go,Ce,_o,R,Le,Ro,E,xe,Ho,Re,pt="The <code>GenericForQuestionAnswering</code> forward method, overrides the <code>__call__</code> special method.",Go,te,Mo,Ue,yo,Pe,bo;return B=new ne({props:{title:"SmolLM3",local:"smollm3",headingTag:"h1"}}),X=new Ge({props:{warning:!1,$$slots:{default:[Tt]},$$scope:{ctx:T}}}),P=new bt({props:{id:"usage",options:["Pipeline","AutoModel","transformers CLI"],$$slots:{default:[$t]},$$scope:{ctx:T}}}),de=new Qe({props:{code:"JTIzJTIwcGlwJTIwaW5zdGFsbCUyMC1VJTIwZmxhc2gtYXR0biUyMC0tbm8tYnVpbGQtaXNvbGF0aW9uJTBBaW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Rva2VuaXplciUyQyUyMEF1dG9Nb2RlbEZvckNhdXNhbExNJTJDJTIwQml0c0FuZEJ5dGVzQ29uZmlnJTBBJTBBcXVhbnRpemF0aW9uX2NvbmZpZyUyMCUzRCUyMEJpdHNBbmRCeXRlc0NvbmZpZyglMEElMjAlMjAlMjAlMjBsb2FkX2luXzRiaXQlM0RUcnVlJTJDJTBBJTIwJTIwJTIwJTIwYm5iXzRiaXRfY29tcHV0ZV9kdHlwZSUzRHRvcmNoLmJmbG9hdDE2JTJDJTBBJTIwJTIwJTIwJTIwYm5iXzRiaXRfcXVhbnRfdHlwZSUzRCUyMm5mNCUyMiUyQyUwQSUyMCUyMCUyMCUyMGJuYl80Yml0X3VzZV9kb3VibGVfcXVhbnQlM0RUcnVlJTJDJTBBKSUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMkh1Z2dpbmdGYWNlVEIlMkZTbW9sTE0zLTNCJTIyKSUwQW1vZGVsJTIwJTNEJTIwQXV0b01vZGVsRm9yQ2F1c2FsTE0uZnJvbV9wcmV0cmFpbmVkKCUwQSUyMCUyMCUyMCUyMCUyMkh1Z2dpbmdGYWNlVEIlMkZTbW9sTE0zLTNCJTIyJTJDJTBBJTIwJTIwJTIwJTIwZHR5cGUlM0R0b3JjaC5iZmxvYXQxNiUyQyUwQSUyMCUyMCUyMCUyMGRldmljZV9tYXAlM0QlMjJhdXRvJTIyJTJDJTBBJTIwJTIwJTIwJTIwcXVhbnRpemF0aW9uX2NvbmZpZyUzRHF1YW50aXphdGlvbl9jb25maWclMkMlMEElMjAlMjAlMjAlMjBhdHRuX2ltcGxlbWVudGF0aW9uJTNEJTIyZmxhc2hfYXR0ZW50aW9uXzIlMjIlMEEpJTBBJTBBaW5wdXRzJTIwJTNEJTIwdG9rZW5pemVyKCUyMkdyYXZpdHklMjBpcyUyMHRoZSUyMGZvcmNlJTIyJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMikudG8obW9kZWwuZGV2aWNlKSUwQW91dHB1dHMlMjAlM0QlMjBtb2RlbC5nZW5lcmF0ZSgqKmlucHV0cyUyQyUyMG1heF9uZXdfdG9rZW5zJTNEMTAwKSUwQXByaW50KHRva2VuaXplci5kZWNvZGUob3V0cHV0cyU1QjAlNUQlMkMlMjBza2lwX3NwZWNpYWxfdG9rZW5zJTNEVHJ1ZSkp",highlighted:`<span class="hljs-comment"># pip install -U flash-attn --no-build-isolation</span>
<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=<span class="hljs-literal">True</span>,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type=<span class="hljs-string">&quot;nf4&quot;</span>,
    bnb_4bit_use_double_quant=<span class="hljs-literal">True</span>,
)

tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;HuggingFaceTB/SmolLM3-3B&quot;</span>)
model = AutoModelForCausalLM.from_pretrained(
    <span class="hljs-string">&quot;HuggingFaceTB/SmolLM3-3B&quot;</span>,
    dtype=torch.bfloat16,
    device_map=<span class="hljs-string">&quot;auto&quot;</span>,
    quantization_config=quantization_config,
    attn_implementation=<span class="hljs-string">&quot;flash_attention_2&quot;</span>
)

inputs = tokenizer(<span class="hljs-string">&quot;Gravity is the force&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>).to(model.device)
outputs = model.generate(**inputs, max_new_tokens=<span class="hljs-number">100</span>)
<span class="hljs-built_in">print</span>(tokenizer.decode(outputs[<span class="hljs-number">0</span>], skip_special_tokens=<span class="hljs-literal">True</span>))`,wrap:!1}}),ce=new ne({props:{title:"Notes",local:"notes",headingTag:"h2"}}),me=new ne({props:{title:"SmolLM3Config",local:"transformers.SmolLM3Config",headingTag:"h2"}}),ue=new I({props:{name:"class transformers.SmolLM3Config",anchor:"transformers.SmolLM3Config",parameters:[{name:"vocab_size",val:" = 128256"},{name:"hidden_size",val:" = 2048"},{name:"intermediate_size",val:" = 11008"},{name:"num_hidden_layers",val:" = 36"},{name:"num_attention_heads",val:" = 16"},{name:"num_key_value_heads",val:" = 4"},{name:"hidden_act",val:" = 'silu'"},{name:"max_position_embeddings",val:" = 32768"},{name:"initializer_range",val:" = 0.02"},{name:"rms_norm_eps",val:" = 1e-06"},{name:"use_cache",val:" = True"},{name:"pad_token_id",val:" = 128004"},{name:"bos_token_id",val:" = 128000"},{name:"eos_token_id",val:" = 128001"},{name:"rope_theta",val:" = 2000000.0"},{name:"rope_scaling",val:" = None"},{name:"use_sliding_window",val:" = False"},{name:"sliding_window",val:" = None"},{name:"no_rope_layers",val:" = None"},{name:"no_rope_layer_interval",val:" = 4"},{name:"layer_types",val:" = None"},{name:"attention_bias",val:" = False"},{name:"attention_dropout",val:" = 0.0"},{name:"mlp_bias",val:" = False"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.SmolLM3Config.vocab_size",description:`<strong>vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 128256) &#x2014;
Vocabulary size of the SmolLM3 model. Defines the number of different tokens that can be represented by the
<code>inputs_ids</code> passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/smollm3#transformers.SmolLM3Model">SmolLM3Model</a>`,name:"vocab_size"},{anchor:"transformers.SmolLM3Config.hidden_size",description:`<strong>hidden_size</strong> (<code>int</code>, <em>optional</em>, defaults to 2048) &#x2014;
Dimension of the hidden representations.`,name:"hidden_size"},{anchor:"transformers.SmolLM3Config.intermediate_size",description:`<strong>intermediate_size</strong> (<code>int</code>, <em>optional</em>, defaults to 11008) &#x2014;
Dimension of the MLP representations.`,name:"intermediate_size"},{anchor:"transformers.SmolLM3Config.num_hidden_layers",description:`<strong>num_hidden_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 36) &#x2014;
Number of hidden layers in the Transformer encoder.`,name:"num_hidden_layers"},{anchor:"transformers.SmolLM3Config.num_attention_heads",description:`<strong>num_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 16) &#x2014;
Number of attention heads for each attention layer in the Transformer encoder.`,name:"num_attention_heads"},{anchor:"transformers.SmolLM3Config.num_key_value_heads",description:`<strong>num_key_value_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 4) &#x2014;
This is the number of key_value heads that should be used to implement Grouped Query Attention. If
<code>num_key_value_heads=num_attention_heads</code>, the model will use Multi Head Attention (MHA), if
<code>num_key_value_heads=1</code> the model will use Multi Query Attention (MQA) otherwise GQA is used. When
converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
by meanpooling all the original heads within that group. For more details checkout <a href="https://huggingface.co/papers/2305.13245" rel="nofollow">this
paper</a>. If it is not specified, will default to <code>16</code>.`,name:"num_key_value_heads"},{anchor:"transformers.SmolLM3Config.hidden_act",description:`<strong>hidden_act</strong> (<code>str</code> or <code>function</code>, <em>optional</em>, defaults to <code>&quot;silu&quot;</code>) &#x2014;
The non-linear activation function (function or string) in the decoder.`,name:"hidden_act"},{anchor:"transformers.SmolLM3Config.max_position_embeddings",description:`<strong>max_position_embeddings</strong> (<code>int</code>, <em>optional</em>, defaults to 32768) &#x2014;
The maximum sequence length that this model might ever be used with.`,name:"max_position_embeddings"},{anchor:"transformers.SmolLM3Config.initializer_range",description:`<strong>initializer_range</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.`,name:"initializer_range"},{anchor:"transformers.SmolLM3Config.rms_norm_eps",description:`<strong>rms_norm_eps</strong> (<code>float</code>, <em>optional</em>, defaults to 1e-06) &#x2014;
The epsilon used by the rms normalization layers.`,name:"rms_norm_eps"},{anchor:"transformers.SmolLM3Config.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not the model should return the last key/values attentions (not used by all models). Only
relevant if <code>config.is_decoder=True</code>.`,name:"use_cache"},{anchor:"transformers.SmolLM3Config.pad_token_id",description:`<strong>pad_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 128004) &#x2014;
The id of the padding token.`,name:"pad_token_id"},{anchor:"transformers.SmolLM3Config.bos_token_id",description:`<strong>bos_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 128000) &#x2014;
The id of the beginning of sentence token.`,name:"bos_token_id"},{anchor:"transformers.SmolLM3Config.eos_token_id",description:`<strong>eos_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 128001) &#x2014;
The id of the end of sentence token.`,name:"eos_token_id"},{anchor:"transformers.SmolLM3Config.rope_theta",description:`<strong>rope_theta</strong> (<code>float</code>, <em>optional</em>, defaults to 2000000.0) &#x2014;
The base period of the RoPE embeddings.`,name:"rope_theta"},{anchor:"transformers.SmolLM3Config.rope_scaling",description:`<strong>rope_scaling</strong> (<code>Dict</code>, <em>optional</em>) &#x2014;
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
<code>short_factor</code> (<code>List[float]</code>, </em>optional<em>):
Only used with &#x2018;longrope&#x2019;. The scaling factor to be applied to short contexts (&lt;
<code>original_max_position_embeddings</code>). Must be a list of numbers with the same length as the hidden
size divided by the number of attention heads divided by 2
<code>long_factor</code> (<code>List[float]</code>, </em>optional<em>):
Only used with &#x2018;longrope&#x2019;. The scaling factor to be applied to long contexts (&lt;
<code>original_max_position_embeddings</code>). Must be a list of numbers with the same length as the hidden
size divided by the number of attention heads divided by 2
<code>low_freq_factor</code> (<code>float</code>, </em>optional<em>):
Only used with &#x2018;llama3&#x2019;. Scaling factor applied to low frequency components of the RoPE
<code>high_freq_factor</code> (<code>float</code>, </em>optional*):
Only used with &#x2018;llama3&#x2019;. Scaling factor applied to high frequency components of the RoPE`,name:"rope_scaling"},{anchor:"transformers.SmolLM3Config.use_sliding_window",description:`<strong>use_sliding_window</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether to use sliding window attention.`,name:"use_sliding_window"},{anchor:"transformers.SmolLM3Config.sliding_window",description:`<strong>sliding_window</strong> (<code>int</code>, <em>optional</em>) &#x2014;
Sliding window attention (SWA) window size. If not specified, will default to <code>None</code>.`,name:"sliding_window"},{anchor:"transformers.SmolLM3Config.no_rope_layers",description:`<strong>no_rope_layers</strong> (<code>List[int]</code>, <em>optional</em>) &#x2014;
List with at least the same length as the number of layers in the model.
A <code>1</code> at an index position indicates that the corresponding layer will use RoPE,
while a <code>0</code> indicates that it&#x2019;s a NoPE layer.`,name:"no_rope_layers"},{anchor:"transformers.SmolLM3Config.no_rope_layer_interval",description:`<strong>no_rope_layer_interval</strong> (<code>int</code>, <em>optional</em>, defaults to 4) &#x2014;
If <code>no_rope_layers</code> is <code>None</code>, it will be created using a NoPE layer every
<code>no_rope_layer_interval</code> layers.`,name:"no_rope_layer_interval"},{anchor:"transformers.SmolLM3Config.layer_types",description:`<strong>layer_types</strong> (<code>list</code>, <em>optional</em>) &#x2014;
Attention pattern for each layer. Automatically computed based on sliding window and NoPE settings.`,name:"layer_types"},{anchor:"transformers.SmolLM3Config.attention_bias",description:`<strong>attention_bias</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether to use a bias in the query, key, value and output projection layers during self-attention.`,name:"attention_bias"},{anchor:"transformers.SmolLM3Config.attention_dropout",description:`<strong>attention_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The dropout ratio for the attention probabilities.`,name:"attention_dropout"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/smollm3/configuration_smollm3.py#L26"}}),D=new mt({props:{anchor:"transformers.SmolLM3Config.example",$$slots:{default:[Ct]},$$scope:{ctx:T}}}),he=new ne({props:{title:"SmolLM3Model",local:"transformers.SmolLM3Model",headingTag:"h2"}}),fe=new I({props:{name:"class transformers.SmolLM3Model",anchor:"transformers.SmolLM3Model",parameters:[{name:"config",val:": SmolLM3Config"}],parametersDescription:[{anchor:"transformers.SmolLM3Model.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/smollm3#transformers.SmolLM3Config">SmolLM3Config</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/smollm3/modeling_smollm3.py#L340"}}),ge=new I({props:{name:"forward",anchor:"transformers.SmolLM3Model.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.SmolLM3Model.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.SmolLM3Model.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.SmolLM3Model.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.SmolLM3Model.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.SmolLM3Model.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.SmolLM3Model.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.SmolLM3Model.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/smollm3/modeling_smollm3.py#L358",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast"
>transformers.modeling_outputs.BaseModelOutputWithPast</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/smollm3#transformers.SmolLM3Config"
>SmolLM3Config</a>) and inputs.</p>
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
`}}),O=new Ge({props:{$$slots:{default:[Lt]},$$scope:{ctx:T}}}),_e=new ne({props:{title:"SmolLM3ForCausalLM",local:"transformers.SmolLM3ForCausalLM",headingTag:"h2"}}),Me=new I({props:{name:"class transformers.SmolLM3ForCausalLM",anchor:"transformers.SmolLM3ForCausalLM",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.SmolLM3ForCausalLM.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/smollm3#transformers.SmolLM3ForCausalLM">SmolLM3ForCausalLM</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/smollm3/modeling_smollm3.py#L433"}}),ye=new I({props:{name:"forward",anchor:"transformers.SmolLM3ForCausalLM.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"},{name:"logits_to_keep",val:": typing.Union[int, torch.Tensor] = 0"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.SmolLM3ForCausalLM.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.SmolLM3ForCausalLM.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.SmolLM3ForCausalLM.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.SmolLM3ForCausalLM.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.SmolLM3ForCausalLM.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.SmolLM3ForCausalLM.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.SmolLM3ForCausalLM.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.SmolLM3ForCausalLM.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"},{anchor:"transformers.SmolLM3ForCausalLM.forward.logits_to_keep",description:`<strong>logits_to_keep</strong> (<code>Union[int, torch.Tensor]</code>, defaults to <code>0</code>) &#x2014;
If an <code>int</code>, compute logits for the last <code>logits_to_keep</code> tokens. If <code>0</code>, calculate logits for all
<code>input_ids</code> (special case). Only last token logits are needed for generation, and calculating them only for that
token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
If a <code>torch.Tensor</code>, must be 1D corresponding to the indices to keep in the sequence length dimension.
This is useful when using packed tensor format (single dimension for batch and sequence length).`,name:"logits_to_keep"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/smollm3/modeling_smollm3.py#L447",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast"
>transformers.modeling_outputs.CausalLMOutputWithPast</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/smollm3#transformers.SmolLM3Config"
>SmolLM3Config</a>) and inputs.</p>
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
`}}),Y=new Ge({props:{$$slots:{default:[xt]},$$scope:{ctx:T}}}),K=new mt({props:{anchor:"transformers.SmolLM3ForCausalLM.forward.example",$$slots:{default:[Ut]},$$scope:{ctx:T}}}),be=new ne({props:{title:"SmolLM3ForSequenceClassification",local:"transformers.SmolLM3ForSequenceClassification",headingTag:"h2"}}),Te=new I({props:{name:"class transformers.SmolLM3ForSequenceClassification",anchor:"transformers.SmolLM3ForSequenceClassification",parameters:[{name:"config",val:""}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/smollm3/modeling_smollm3.py#L508"}}),we=new I({props:{name:"forward",anchor:"transformers.SmolLM3ForSequenceClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.SmolLM3ForSequenceClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.SmolLM3ForSequenceClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.SmolLM3ForSequenceClassification.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.SmolLM3ForSequenceClassification.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.SmolLM3ForSequenceClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.SmolLM3ForSequenceClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.SmolLM3ForSequenceClassification.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
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
`}}),ee=new Ge({props:{$$slots:{default:[Jt]},$$scope:{ctx:T}}}),ve=new ne({props:{title:"SmolLM3ForTokenClassification",local:"transformers.SmolLM3ForTokenClassification",headingTag:"h2"}}),ke=new I({props:{name:"class transformers.SmolLM3ForTokenClassification",anchor:"transformers.SmolLM3ForTokenClassification",parameters:[{name:"config",val:""}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/smollm3/modeling_smollm3.py#L512"}}),$e=new I({props:{name:"forward",anchor:"transformers.SmolLM3ForTokenClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.SmolLM3ForTokenClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.SmolLM3ForTokenClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.SmolLM3ForTokenClassification.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.SmolLM3ForTokenClassification.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.SmolLM3ForTokenClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.SmolLM3ForTokenClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.SmolLM3ForTokenClassification.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
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
`}}),oe=new Ge({props:{$$slots:{default:[zt]},$$scope:{ctx:T}}}),Ce=new ne({props:{title:"SmolLM3ForQuestionAnswering",local:"transformers.SmolLM3ForQuestionAnswering",headingTag:"h2"}}),Le=new I({props:{name:"class transformers.SmolLM3ForQuestionAnswering",anchor:"transformers.SmolLM3ForQuestionAnswering",parameters:[{name:"config",val:""}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/smollm3/modeling_smollm3.py#L516"}}),xe=new I({props:{name:"forward",anchor:"transformers.SmolLM3ForQuestionAnswering.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"start_positions",val:": typing.Optional[torch.LongTensor] = None"},{name:"end_positions",val:": typing.Optional[torch.LongTensor] = None"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.SmolLM3ForQuestionAnswering.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.SmolLM3ForQuestionAnswering.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.SmolLM3ForQuestionAnswering.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.SmolLM3ForQuestionAnswering.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.SmolLM3ForQuestionAnswering.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.SmolLM3ForQuestionAnswering.forward.start_positions",description:`<strong>start_positions</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for position (index) of the start of the labelled span for computing the token classification loss.
Positions are clamped to the length of the sequence (<code>sequence_length</code>). Position outside of the sequence
are not taken into account for computing the loss.`,name:"start_positions"},{anchor:"transformers.SmolLM3ForQuestionAnswering.forward.end_positions",description:`<strong>end_positions</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
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
`}}),te=new Ge({props:{$$slots:{default:[St]},$$scope:{ctx:T}}}),Ue=new yt({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/smollm3.md"}}),{c(){o=c("meta"),l=a(),t=c("p"),M=a(),v=c("p"),v.innerHTML=w,y=a(),k=c("div"),k.innerHTML=Xe,se=a(),m(B.$$.fragment),De=a(),ae=c("p"),ae.innerHTML=Xo,Oe=a(),m(X.$$.fragment),Ye=a(),re=c("p"),re.innerHTML=Po,Ke=a(),m(P.$$.fragment),eo=a(),ie=c("p"),ie.innerHTML=Do,oo=a(),le=c("p"),le.innerHTML=Oo,to=a(),m(de.$$.fragment),no=a(),m(ce.$$.fragment),so=a(),pe=c("ul"),pe.innerHTML=Yo,ao=a(),m(me.$$.fragment),ro=a(),L=c("div"),m(ue.$$.fragment),wo=a(),Se=c("p"),Se.innerHTML=Ko,vo=a(),je=c("p"),je.innerHTML=et,ko=a(),m(D.$$.fragment),io=a(),m(he.$$.fragment),lo=a(),$=c("div"),m(fe.$$.fragment),$o=a(),Fe=c("p"),Fe.textContent=ot,Co=a(),Ie=c("p"),Ie.innerHTML=tt,Lo=a(),qe=c("p"),qe.innerHTML=nt,xo=a(),Z=c("div"),m(ge.$$.fragment),Uo=a(),Be=c("p"),Be.innerHTML=st,Jo=a(),m(O.$$.fragment),co=a(),m(_e.$$.fragment),po=a(),C=c("div"),m(Me.$$.fragment),zo=a(),Ze=c("p"),Ze.textContent=at,So=a(),We=c("p"),We.innerHTML=rt,jo=a(),Ne=c("p"),Ne.innerHTML=it,Fo=a(),S=c("div"),m(ye.$$.fragment),Io=a(),Ee=c("p"),Ee.innerHTML=lt,qo=a(),m(Y.$$.fragment),Bo=a(),m(K.$$.fragment),mo=a(),m(be.$$.fragment),uo=a(),A=c("div"),m(Te.$$.fragment),Zo=a(),W=c("div"),m(we.$$.fragment),Wo=a(),Ae=c("p"),Ae.innerHTML=dt,No=a(),m(ee.$$.fragment),ho=a(),m(ve.$$.fragment),fo=a(),V=c("div"),m(ke.$$.fragment),Eo=a(),N=c("div"),m($e.$$.fragment),Ao=a(),Ve=c("p"),Ve.innerHTML=ct,Vo=a(),m(oe.$$.fragment),go=a(),m(Ce.$$.fragment),_o=a(),R=c("div"),m(Le.$$.fragment),Ro=a(),E=c("div"),m(xe.$$.fragment),Ho=a(),Re=c("p"),Re.innerHTML=pt,Go=a(),m(te.$$.fragment),Mo=a(),m(Ue.$$.fragment),yo=a(),Pe=c("p"),this.h()},l(e){const n=_t("svelte-u9bgzb",document.head);o=p(n,"META",{name:!0,content:!0}),n.forEach(s),l=r(e),t=p(e,"P",{}),J(t).forEach(s),M=r(e),v=p(e,"P",{"data-svelte-h":!0}),b(v)!=="svelte-1qhoe36"&&(v.innerHTML=w),y=r(e),k=p(e,"DIV",{style:!0,"data-svelte-h":!0}),b(k)!=="svelte-2m0t7r"&&(k.innerHTML=Xe),se=r(e),u(B.$$.fragment,e),De=r(e),ae=p(e,"P",{"data-svelte-h":!0}),b(ae)!=="svelte-183yfc3"&&(ae.innerHTML=Xo),Oe=r(e),u(X.$$.fragment,e),Ye=r(e),re=p(e,"P",{"data-svelte-h":!0}),b(re)!=="svelte-1v4bhcb"&&(re.innerHTML=Po),Ke=r(e),u(P.$$.fragment,e),eo=r(e),ie=p(e,"P",{"data-svelte-h":!0}),b(ie)!=="svelte-nf5ooi"&&(ie.innerHTML=Do),oo=r(e),le=p(e,"P",{"data-svelte-h":!0}),b(le)!=="svelte-1ca5nhg"&&(le.innerHTML=Oo),to=r(e),u(de.$$.fragment,e),no=r(e),u(ce.$$.fragment,e),so=r(e),pe=p(e,"UL",{"data-svelte-h":!0}),b(pe)!=="svelte-tx6kkw"&&(pe.innerHTML=Yo),ao=r(e),u(me.$$.fragment,e),ro=r(e),L=p(e,"DIV",{class:!0});var j=J(L);u(ue.$$.fragment,j),wo=r(j),Se=p(j,"P",{"data-svelte-h":!0}),b(Se)!=="svelte-1w8axo"&&(Se.innerHTML=Ko),vo=r(j),je=p(j,"P",{"data-svelte-h":!0}),b(je)!=="svelte-1ek1ss9"&&(je.innerHTML=et),ko=r(j),u(D.$$.fragment,j),j.forEach(s),io=r(e),u(he.$$.fragment,e),lo=r(e),$=p(e,"DIV",{class:!0});var x=J($);u(fe.$$.fragment,x),$o=r(x),Fe=p(x,"P",{"data-svelte-h":!0}),b(Fe)!=="svelte-ag7r6t"&&(Fe.textContent=ot),Co=r(x),Ie=p(x,"P",{"data-svelte-h":!0}),b(Ie)!=="svelte-q52n56"&&(Ie.innerHTML=tt),Lo=r(x),qe=p(x,"P",{"data-svelte-h":!0}),b(qe)!=="svelte-hswkmf"&&(qe.innerHTML=nt),xo=r(x),Z=p(x,"DIV",{class:!0});var H=J(Z);u(ge.$$.fragment,H),Uo=r(H),Be=p(H,"P",{"data-svelte-h":!0}),b(Be)!=="svelte-j7l5v4"&&(Be.innerHTML=st),Jo=r(H),u(O.$$.fragment,H),H.forEach(s),x.forEach(s),co=r(e),u(_e.$$.fragment,e),po=r(e),C=p(e,"DIV",{class:!0});var U=J(C);u(Me.$$.fragment,U),zo=r(U),Ze=p(U,"P",{"data-svelte-h":!0}),b(Ze)!=="svelte-143j3se"&&(Ze.textContent=at),So=r(U),We=p(U,"P",{"data-svelte-h":!0}),b(We)!=="svelte-q52n56"&&(We.innerHTML=rt),jo=r(U),Ne=p(U,"P",{"data-svelte-h":!0}),b(Ne)!=="svelte-hswkmf"&&(Ne.innerHTML=it),Fo=r(U),S=p(U,"DIV",{class:!0});var F=J(S);u(ye.$$.fragment,F),Io=r(F),Ee=p(F,"P",{"data-svelte-h":!0}),b(Ee)!=="svelte-uac5fs"&&(Ee.innerHTML=lt),qo=r(F),u(Y.$$.fragment,F),Bo=r(F),u(K.$$.fragment,F),F.forEach(s),U.forEach(s),mo=r(e),u(be.$$.fragment,e),uo=r(e),A=p(e,"DIV",{class:!0});var Je=J(A);u(Te.$$.fragment,Je),Zo=r(Je),W=p(Je,"DIV",{class:!0});var G=J(W);u(we.$$.fragment,G),Wo=r(G),Ae=p(G,"P",{"data-svelte-h":!0}),b(Ae)!=="svelte-1sal4ui"&&(Ae.innerHTML=dt),No=r(G),u(ee.$$.fragment,G),G.forEach(s),Je.forEach(s),ho=r(e),u(ve.$$.fragment,e),fo=r(e),V=p(e,"DIV",{class:!0});var ze=J(V);u(ke.$$.fragment,ze),Eo=r(ze),N=p(ze,"DIV",{class:!0});var Q=J(N);u($e.$$.fragment,Q),Ao=r(Q),Ve=p(Q,"P",{"data-svelte-h":!0}),b(Ve)!=="svelte-1py4aay"&&(Ve.innerHTML=ct),Vo=r(Q),u(oe.$$.fragment,Q),Q.forEach(s),ze.forEach(s),go=r(e),u(Ce.$$.fragment,e),_o=r(e),R=p(e,"DIV",{class:!0});var To=J(R);u(Le.$$.fragment,To),Ro=r(To),E=p(To,"DIV",{class:!0});var He=J(E);u(xe.$$.fragment,He),Ho=r(He),Re=p(He,"P",{"data-svelte-h":!0}),b(Re)!=="svelte-dyrov9"&&(Re.innerHTML=pt),Go=r(He),u(te.$$.fragment,He),He.forEach(s),To.forEach(s),Mo=r(e),u(Ue.$$.fragment,e),yo=r(e),Pe=p(e,"P",{}),J(Pe).forEach(s),this.h()},h(){z(o,"name","hf:doc:metadata"),z(o,"content",Ft),Mt(k,"float","right"),z(L,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),z(Z,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),z($,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),z(S,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),z(C,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),z(W,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),z(A,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),z(N,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),z(V,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),z(E,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),z(R,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,n){d(document.head,o),i(e,l,n),i(e,t,n),i(e,M,n),i(e,v,n),i(e,y,n),i(e,k,n),i(e,se,n),h(B,e,n),i(e,De,n),i(e,ae,n),i(e,Oe,n),h(X,e,n),i(e,Ye,n),i(e,re,n),i(e,Ke,n),h(P,e,n),i(e,eo,n),i(e,ie,n),i(e,oo,n),i(e,le,n),i(e,to,n),h(de,e,n),i(e,no,n),h(ce,e,n),i(e,so,n),i(e,pe,n),i(e,ao,n),h(me,e,n),i(e,ro,n),i(e,L,n),h(ue,L,null),d(L,wo),d(L,Se),d(L,vo),d(L,je),d(L,ko),h(D,L,null),i(e,io,n),h(he,e,n),i(e,lo,n),i(e,$,n),h(fe,$,null),d($,$o),d($,Fe),d($,Co),d($,Ie),d($,Lo),d($,qe),d($,xo),d($,Z),h(ge,Z,null),d(Z,Uo),d(Z,Be),d(Z,Jo),h(O,Z,null),i(e,co,n),h(_e,e,n),i(e,po,n),i(e,C,n),h(Me,C,null),d(C,zo),d(C,Ze),d(C,So),d(C,We),d(C,jo),d(C,Ne),d(C,Fo),d(C,S),h(ye,S,null),d(S,Io),d(S,Ee),d(S,qo),h(Y,S,null),d(S,Bo),h(K,S,null),i(e,mo,n),h(be,e,n),i(e,uo,n),i(e,A,n),h(Te,A,null),d(A,Zo),d(A,W),h(we,W,null),d(W,Wo),d(W,Ae),d(W,No),h(ee,W,null),i(e,ho,n),h(ve,e,n),i(e,fo,n),i(e,V,n),h(ke,V,null),d(V,Eo),d(V,N),h($e,N,null),d(N,Ao),d(N,Ve),d(N,Vo),h(oe,N,null),i(e,go,n),h(Ce,e,n),i(e,_o,n),i(e,R,n),h(Le,R,null),d(R,Ro),d(R,E),h(xe,E,null),d(E,Ho),d(E,Re),d(E,Go),h(te,E,null),i(e,Mo,n),h(Ue,e,n),i(e,yo,n),i(e,Pe,n),bo=!0},p(e,[n]){const j={};n&2&&(j.$$scope={dirty:n,ctx:e}),X.$set(j);const x={};n&2&&(x.$$scope={dirty:n,ctx:e}),P.$set(x);const H={};n&2&&(H.$$scope={dirty:n,ctx:e}),D.$set(H);const U={};n&2&&(U.$$scope={dirty:n,ctx:e}),O.$set(U);const F={};n&2&&(F.$$scope={dirty:n,ctx:e}),Y.$set(F);const Je={};n&2&&(Je.$$scope={dirty:n,ctx:e}),K.$set(Je);const G={};n&2&&(G.$$scope={dirty:n,ctx:e}),ee.$set(G);const ze={};n&2&&(ze.$$scope={dirty:n,ctx:e}),oe.$set(ze);const Q={};n&2&&(Q.$$scope={dirty:n,ctx:e}),te.$set(Q)},i(e){bo||(f(B.$$.fragment,e),f(X.$$.fragment,e),f(P.$$.fragment,e),f(de.$$.fragment,e),f(ce.$$.fragment,e),f(me.$$.fragment,e),f(ue.$$.fragment,e),f(D.$$.fragment,e),f(he.$$.fragment,e),f(fe.$$.fragment,e),f(ge.$$.fragment,e),f(O.$$.fragment,e),f(_e.$$.fragment,e),f(Me.$$.fragment,e),f(ye.$$.fragment,e),f(Y.$$.fragment,e),f(K.$$.fragment,e),f(be.$$.fragment,e),f(Te.$$.fragment,e),f(we.$$.fragment,e),f(ee.$$.fragment,e),f(ve.$$.fragment,e),f(ke.$$.fragment,e),f($e.$$.fragment,e),f(oe.$$.fragment,e),f(Ce.$$.fragment,e),f(Le.$$.fragment,e),f(xe.$$.fragment,e),f(te.$$.fragment,e),f(Ue.$$.fragment,e),bo=!0)},o(e){g(B.$$.fragment,e),g(X.$$.fragment,e),g(P.$$.fragment,e),g(de.$$.fragment,e),g(ce.$$.fragment,e),g(me.$$.fragment,e),g(ue.$$.fragment,e),g(D.$$.fragment,e),g(he.$$.fragment,e),g(fe.$$.fragment,e),g(ge.$$.fragment,e),g(O.$$.fragment,e),g(_e.$$.fragment,e),g(Me.$$.fragment,e),g(ye.$$.fragment,e),g(Y.$$.fragment,e),g(K.$$.fragment,e),g(be.$$.fragment,e),g(Te.$$.fragment,e),g(we.$$.fragment,e),g(ee.$$.fragment,e),g(ve.$$.fragment,e),g(ke.$$.fragment,e),g($e.$$.fragment,e),g(oe.$$.fragment,e),g(Ce.$$.fragment,e),g(Le.$$.fragment,e),g(xe.$$.fragment,e),g(te.$$.fragment,e),g(Ue.$$.fragment,e),bo=!1},d(e){e&&(s(l),s(t),s(M),s(v),s(y),s(k),s(se),s(De),s(ae),s(Oe),s(Ye),s(re),s(Ke),s(eo),s(ie),s(oo),s(le),s(to),s(no),s(so),s(pe),s(ao),s(ro),s(L),s(io),s(lo),s($),s(co),s(po),s(C),s(mo),s(uo),s(A),s(ho),s(fo),s(V),s(go),s(_o),s(R),s(Mo),s(yo),s(Pe)),s(o),_(B,e),_(X,e),_(P,e),_(de,e),_(ce,e),_(me,e),_(ue),_(D),_(he,e),_(fe),_(ge),_(O),_(_e,e),_(Me),_(ye),_(Y),_(K),_(be,e),_(Te),_(we),_(ee),_(ve,e),_(ke),_($e),_(oe),_(Ce,e),_(Le),_(xe),_(te),_(Ue,e)}}}const Ft='{"title":"SmolLM3","local":"smollm3","sections":[{"title":"Notes","local":"notes","sections":[],"depth":2},{"title":"SmolLM3Config","local":"transformers.SmolLM3Config","sections":[],"depth":2},{"title":"SmolLM3Model","local":"transformers.SmolLM3Model","sections":[],"depth":2},{"title":"SmolLM3ForCausalLM","local":"transformers.SmolLM3ForCausalLM","sections":[],"depth":2},{"title":"SmolLM3ForSequenceClassification","local":"transformers.SmolLM3ForSequenceClassification","sections":[],"depth":2},{"title":"SmolLM3ForTokenClassification","local":"transformers.SmolLM3ForTokenClassification","sections":[],"depth":2},{"title":"SmolLM3ForQuestionAnswering","local":"transformers.SmolLM3ForQuestionAnswering","sections":[],"depth":2}],"depth":1}';function It(T){return ht(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class Rt extends ft{constructor(o){super(),gt(this,o,It,jt,ut,{})}}export{Rt as component};
