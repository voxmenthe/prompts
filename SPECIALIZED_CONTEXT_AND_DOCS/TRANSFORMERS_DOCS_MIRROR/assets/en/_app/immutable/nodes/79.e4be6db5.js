import{s as zn,o as qn,n as H}from"../chunks/scheduler.18a86fab.js";import{S as Fn,i as Bn,g as l,s,r as m,A as Nn,h as d,f as a,c as r,j as w,x as _,u as p,k as j,l as Wn,y as o,a as c,v as h,d as g,t as u,w as f}from"../chunks/index.98837b22.js";import{T as kt}from"../chunks/Tip.77304350.js";import{D as A}from"../chunks/Docstring.a1ef7999.js";import{C as Ct}from"../chunks/CodeBlock.8d0c2e8a.js";import{E as Jn}from"../chunks/ExampleCodeBlock.8c3ee1f9.js";import{H as X,E as Pn}from"../chunks/getInferenceSnippets.06c2775f.js";import{H as Zn,a as $n}from"../chunks/HfOption.6641485e.js";function Rn(v){let t,y="Click on the Aria models in the right sidebar for more examples of how to apply Aria to different multimodal tasks.";return{c(){t=l("p"),t.textContent=y},l(i){t=d(i,"P",{"data-svelte-h":!0}),_(t)!=="svelte-1pi0dt1"&&(t.textContent=y)},m(i,T){c(i,t,T)},p:H,d(i){i&&a(t)}}}function En(v){let t,y;return t=new Ct({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwcGlwZWxpbmUlMEElMEFwaXBlbGluZSUyMCUzRCUyMHBpcGVsaW5lKCUwQSUyMCUyMCUyMCUyMCUyMmltYWdlLXRvLXRleHQlMjIlMkMlMEElMjAlMjAlMjAlMjBtb2RlbCUzRCUyMnJoeW1lcy1haSUyRkFyaWElMjIlMkMlMEElMjAlMjAlMjAlMjBkZXZpY2UlM0QwJTJDJTBBJTIwJTIwJTIwJTIwZHR5cGUlM0R0b3JjaC5iZmxvYXQxNiUwQSklMEFwaXBlbGluZSglMEElMjAlMjAlMjAlMjAlMjJodHRwcyUzQSUyRiUyRmh1Z2dpbmdmYWNlLmNvJTJGZGF0YXNldHMlMkZodWdnaW5nZmFjZSUyRmRvY3VtZW50YXRpb24taW1hZ2VzJTJGcmVzb2x2ZSUyRm1haW4lMkZwaXBlbGluZS1jYXQtY2hvbmsuanBlZyUyMiUyQyUwQSUyMCUyMCUyMCUyMHRleHQlM0QlMjJXaGF0JTIwaXMlMjBzaG93biUyMGluJTIwdGhpcyUyMGltYWdlJTNGJTIyJTBBKQ==",highlighted:`<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> pipeline

pipeline = pipeline(
    <span class="hljs-string">&quot;image-to-text&quot;</span>,
    model=<span class="hljs-string">&quot;rhymes-ai/Aria&quot;</span>,
    device=<span class="hljs-number">0</span>,
    dtype=torch.bfloat16
)
pipeline(
    <span class="hljs-string">&quot;https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg&quot;</span>,
    text=<span class="hljs-string">&quot;What is shown in this image?&quot;</span>
)`,wrap:!1}}),{c(){m(t.$$.fragment)},l(i){p(t.$$.fragment,i)},m(i,T){h(t,i,T),y=!0},p:H,i(i){y||(g(t.$$.fragment,i),y=!0)},o(i){u(t.$$.fragment,i),y=!1},d(i){f(t,i)}}}function Ln(v){let t,y;return t=new Ct({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b01vZGVsRm9yQ2F1c2FsTE0lMkMlMjBBdXRvUHJvY2Vzc29yJTBBJTBBbW9kZWwlMjAlM0QlMjBBdXRvTW9kZWxGb3JDYXVzYWxMTS5mcm9tX3ByZXRyYWluZWQoJTBBJTIwJTIwJTIwJTIwJTIycmh5bWVzLWFpJTJGQXJpYSUyMiUyQyUwQSUyMCUyMCUyMCUyMGRldmljZV9tYXAlM0QlMjJhdXRvJTIyJTJDJTBBJTIwJTIwJTIwJTIwZHR5cGUlM0R0b3JjaC5iZmxvYXQxNiUyQyUwQSUyMCUyMCUyMCUyMGF0dG5faW1wbGVtZW50YXRpb24lM0QlMjJzZHBhJTIyJTBBKSUwQSUwQXByb2Nlc3NvciUyMCUzRCUyMEF1dG9Qcm9jZXNzb3IuZnJvbV9wcmV0cmFpbmVkKCUyMnJoeW1lcy1haSUyRkFyaWElMjIpJTBBJTBBbWVzc2FnZXMlMjAlM0QlMjAlNUIlMEElMjAlMjAlMjAlMjAlN0IlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjJyb2xlJTIyJTNBJTIwJTIydXNlciUyMiUyQyUyMCUyMmNvbnRlbnQlMjIlM0ElMjAlNUIlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlN0IlMjJ0eXBlJTIyJTNBJTIwJTIyaW1hZ2UlMjIlMkMlMjAlMjJ1cmwlMjIlM0ElMjAlMjJodHRwcyUzQSUyRiUyRmh1Z2dpbmdmYWNlLmNvJTJGZGF0YXNldHMlMkZodWdnaW5nZmFjZSUyRmRvY3VtZW50YXRpb24taW1hZ2VzJTJGcmVzb2x2ZSUyRm1haW4lMkZwaXBlbGluZS1jYXQtY2hvbmsuanBlZyUyMiU3RCUyQyUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCU3QiUyMnR5cGUlMjIlM0ElMjAlMjJ0ZXh0JTIyJTJDJTIwJTIydGV4dCUyMiUzQSUyMCUyMldoYXQlMjBpcyUyMHNob3duJTIwaW4lMjB0aGlzJTIwaW1hZ2UlM0YlMjIlN0QlMkMlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlNUQlMEElMjAlMjAlMjAlMjAlN0QlMkMlMEElNUQlMEElMEFpbnB1dHMlMjAlM0QlMjBwcm9jZXNzb3IuYXBwbHlfY2hhdF90ZW1wbGF0ZShtZXNzYWdlcyUyQyUyMGFkZF9nZW5lcmF0aW9uX3Byb21wdCUzRFRydWUlMkMlMjB0b2tlbml6ZSUzRFRydWUlMkMlMjByZXR1cm5fZGljdCUzRFRydWUlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQWlwbnV0cyUyMCUzRCUyMGlucHV0cy50byhtb2RlbC5kZXZpY2UlMkMlMjB0b3JjaC5iZmxvYXQxNiklMEElMEFvdXRwdXQlMjAlM0QlMjBtb2RlbC5nZW5lcmF0ZSglMEElMjAlMjAlMjAlMjAqKmlucHV0cyUyQyUwQSUyMCUyMCUyMCUyMG1heF9uZXdfdG9rZW5zJTNEMTUlMkMlMEElMjAlMjAlMjAlMjBzdG9wX3N0cmluZ3MlM0QlNUIlMjIlM0MlN0NpbV9lbmQlN0MlM0UlMjIlNUQlMkMlMEElMjAlMjAlMjAlMjB0b2tlbml6ZXIlM0Rwcm9jZXNzb3IudG9rZW5pemVyJTJDJTBBJTIwJTIwJTIwJTIwZG9fc2FtcGxlJTNEVHJ1ZSUyQyUwQSUyMCUyMCUyMCUyMHRlbXBlcmF0dXJlJTNEMC45JTJDJTBBKSUwQW91dHB1dF9pZHMlMjAlM0QlMjBvdXRwdXQlNUIwJTVEJTVCaW5wdXRzJTVCJTIyaW5wdXRfaWRzJTIyJTVELnNoYXBlJTVCMSU1RCUzQSU1RCUwQXJlc3BvbnNlJTIwJTNEJTIwcHJvY2Vzc29yLmRlY29kZShvdXRwdXRfaWRzJTJDJTIwc2tpcF9zcGVjaWFsX3Rva2VucyUzRFRydWUpJTBBcHJpbnQocmVzcG9uc2Up",highlighted:`<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoModelForCausalLM, AutoProcessor

model = AutoModelForCausalLM.from_pretrained(
    <span class="hljs-string">&quot;rhymes-ai/Aria&quot;</span>,
    device_map=<span class="hljs-string">&quot;auto&quot;</span>,
    dtype=torch.bfloat16,
    attn_implementation=<span class="hljs-string">&quot;sdpa&quot;</span>
)

processor = AutoProcessor.from_pretrained(<span class="hljs-string">&quot;rhymes-ai/Aria&quot;</span>)

messages = [
    {
        <span class="hljs-string">&quot;role&quot;</span>: <span class="hljs-string">&quot;user&quot;</span>, <span class="hljs-string">&quot;content&quot;</span>: [
            {<span class="hljs-string">&quot;type&quot;</span>: <span class="hljs-string">&quot;image&quot;</span>, <span class="hljs-string">&quot;url&quot;</span>: <span class="hljs-string">&quot;https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg&quot;</span>},
            {<span class="hljs-string">&quot;type&quot;</span>: <span class="hljs-string">&quot;text&quot;</span>, <span class="hljs-string">&quot;text&quot;</span>: <span class="hljs-string">&quot;What is shown in this image?&quot;</span>},
        ]
    },
]

inputs = processor.apply_chat_template(messages, add_generation_prompt=<span class="hljs-literal">True</span>, tokenize=<span class="hljs-literal">True</span>, return_dict=<span class="hljs-literal">True</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
ipnuts = inputs.to(model.device, torch.bfloat16)

output = model.generate(
    **inputs,
    max_new_tokens=<span class="hljs-number">15</span>,
    stop_strings=[<span class="hljs-string">&quot;&lt;|im_end|&gt;&quot;</span>],
    tokenizer=processor.tokenizer,
    do_sample=<span class="hljs-literal">True</span>,
    temperature=<span class="hljs-number">0.9</span>,
)
output_ids = output[<span class="hljs-number">0</span>][inputs[<span class="hljs-string">&quot;input_ids&quot;</span>].shape[<span class="hljs-number">1</span>]:]
response = processor.decode(output_ids, skip_special_tokens=<span class="hljs-literal">True</span>)
<span class="hljs-built_in">print</span>(response)`,wrap:!1}}),{c(){m(t.$$.fragment)},l(i){p(t.$$.fragment,i)},m(i,T){h(t,i,T),y=!0},p:H,i(i){y||(g(t.$$.fragment,i),y=!0)},o(i){u(t.$$.fragment,i),y=!1},d(i){f(t,i)}}}function Qn(v){let t,y,i,T;return t=new $n({props:{id:"usage",option:"Pipeline",$$slots:{default:[En]},$$scope:{ctx:v}}}),i=new $n({props:{id:"usage",option:"AutoModel",$$slots:{default:[Ln]},$$scope:{ctx:v}}}),{c(){m(t.$$.fragment),y=s(),m(i.$$.fragment)},l(b){p(t.$$.fragment,b),y=r(b),p(i.$$.fragment,b)},m(b,M){h(t,b,M),c(b,y,M),h(i,b,M),T=!0},p(b,M){const U={};M&2&&(U.$$scope={dirty:M,ctx:b}),t.$set(U);const N={};M&2&&(N.$$scope={dirty:M,ctx:b}),i.$set(N)},i(b){T||(g(t.$$.fragment,b),g(i.$$.fragment,b),T=!0)},o(b){u(t.$$.fragment,b),u(i.$$.fragment,b),T=!1},d(b){b&&a(y),f(t,b),f(i,b)}}}function Gn(v){let t,y=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=l("p"),t.innerHTML=y},l(i){t=d(i,"P",{"data-svelte-h":!0}),_(t)!=="svelte-fincs2"&&(t.innerHTML=y)},m(i,T){c(i,t,T)},p:H,d(i){i&&a(t)}}}function Xn(v){let t,y=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=l("p"),t.innerHTML=y},l(i){t=d(i,"P",{"data-svelte-h":!0}),_(t)!=="svelte-fincs2"&&(t.innerHTML=y)},m(i,T){c(i,t,T)},p:H,d(i){i&&a(t)}}}function Hn(v){let t,y=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=l("p"),t.innerHTML=y},l(i){t=d(i,"P",{"data-svelte-h":!0}),_(t)!=="svelte-fincs2"&&(t.innerHTML=y)},m(i,T){c(i,t,T)},p:H,d(i){i&&a(t)}}}function Vn(v){let t,y="Example:",i,T,b;return T=new Ct({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBBcmlhVGV4dEZvckNhdXNhbExNJTBBJTBBbW9kZWwlMjAlM0QlMjBBcmlhVGV4dEZvckNhdXNhbExNLmZyb21fcHJldHJhaW5lZCglMjJtZXRhLWFyaWFfdGV4dCUyRkFyaWFUZXh0LTItN2ItaGYlMjIpJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIybWV0YS1hcmlhX3RleHQlMkZBcmlhVGV4dC0yLTdiLWhmJTIyKSUwQSUwQXByb21wdCUyMCUzRCUyMCUyMkhleSUyQyUyMGFyZSUyMHlvdSUyMGNvbnNjaW91cyUzRiUyMENhbiUyMHlvdSUyMHRhbGslMjB0byUyMG1lJTNGJTIyJTBBaW5wdXRzJTIwJTNEJTIwdG9rZW5pemVyKHByb21wdCUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpJTBBJTBBJTIzJTIwR2VuZXJhdGUlMEFnZW5lcmF0ZV9pZHMlMjAlM0QlMjBtb2RlbC5nZW5lcmF0ZShpbnB1dHMuaW5wdXRfaWRzJTJDJTIwbWF4X2xlbmd0aCUzRDMwKSUwQXRva2VuaXplci5iYXRjaF9kZWNvZGUoZ2VuZXJhdGVfaWRzJTJDJTIwc2tpcF9zcGVjaWFsX3Rva2VucyUzRFRydWUlMkMlMjBjbGVhbl91cF90b2tlbml6YXRpb25fc3BhY2VzJTNERmFsc2UpJTVCMCU1RA==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, AriaTextForCausalLM

<span class="hljs-meta">&gt;&gt;&gt; </span>model = AriaTextForCausalLM.from_pretrained(<span class="hljs-string">&quot;meta-aria_text/AriaText-2-7b-hf&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;meta-aria_text/AriaText-2-7b-hf&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>prompt = <span class="hljs-string">&quot;Hey, are you conscious? Can you talk to me?&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(prompt, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Generate</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>generate_ids = model.generate(inputs.input_ids, max_length=<span class="hljs-number">30</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer.batch_decode(generate_ids, skip_special_tokens=<span class="hljs-literal">True</span>, clean_up_tokenization_spaces=<span class="hljs-literal">False</span>)[<span class="hljs-number">0</span>]
<span class="hljs-string">&quot;Hey, are you conscious? Can you talk to me?\\nI&#x27;m not conscious, but I can talk to you.&quot;</span>`,wrap:!1}}),{c(){t=l("p"),t.textContent=y,i=s(),m(T.$$.fragment)},l(M){t=d(M,"P",{"data-svelte-h":!0}),_(t)!=="svelte-11lpom8"&&(t.textContent=y),i=r(M),p(T.$$.fragment,M)},m(M,U){c(M,t,U),c(M,i,U),h(T,M,U),b=!0},p:H,i(M){b||(g(T.$$.fragment,M),b=!0)},o(M){u(T.$$.fragment,M),b=!1},d(M){M&&(a(t),a(i)),f(T,M)}}}function Sn(v){let t,y=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=l("p"),t.innerHTML=y},l(i){t=d(i,"P",{"data-svelte-h":!0}),_(t)!=="svelte-fincs2"&&(t.innerHTML=y)},m(i,T){c(i,t,T)},p:H,d(i){i&&a(t)}}}function Dn(v){let t,y="Example:",i,T,b;return T=new Ct({props:{code:"aW1wb3J0JTIwcmVxdWVzdHMlMEFpbXBvcnQlMjB0b3JjaCUwQWZyb20lMjBQSUwlMjBpbXBvcnQlMjBJbWFnZSUwQWZyb20lMjBpbyUyMGltcG9ydCUyMEJ5dGVzSU8lMEElMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Byb2Nlc3NvciUyQyUyMEF1dG9Nb2RlbCUwQWZyb20lMjB0cmFuc2Zvcm1lcnMuaW1hZ2VfdXRpbHMlMjBpbXBvcnQlMjBsb2FkX2ltYWdlJTBBJTBBJTIzJTIwTm90ZSUyMHRoYXQlMjBwYXNzaW5nJTIwdGhlJTIwaW1hZ2UlMjB1cmxzJTIwKGluc3RlYWQlMjBvZiUyMHRoZSUyMGFjdHVhbCUyMHBpbCUyMGltYWdlcyklMjB0byUyMHRoZSUyMHByb2Nlc3NvciUyMGlzJTIwYWxzbyUyMHBvc3NpYmxlJTBBaW1hZ2UxJTIwJTNEJTIwbG9hZF9pbWFnZSglMjJodHRwcyUzQSUyRiUyRmNkbi5icml0YW5uaWNhLmNvbSUyRjYxJTJGOTMwNjEtMDUwLTk5MTQ3RENFJTJGU3RhdHVlLW9mLUxpYmVydHktSXNsYW5kLU5ldy1Zb3JrLUJheS5qcGclMjIpJTBBaW1hZ2UyJTIwJTNEJTIwbG9hZF9pbWFnZSglMjJodHRwcyUzQSUyRiUyRmNkbi5icml0YW5uaWNhLmNvbSUyRjU5JTJGOTQ0NTktMDUwLURCQTQyNDY3JTJGU2t5bGluZS1DaGljYWdvLmpwZyUyMiklMEFpbWFnZTMlMjAlM0QlMjBsb2FkX2ltYWdlKCUyMmh0dHBzJTNBJTJGJTJGY2RuLmJyaXRhbm5pY2EuY29tJTJGNjglMkYxNzA4NjgtMDUwLThEREU4MjYzJTJGR29sZGVuLUdhdGUtQnJpZGdlLVNhbi1GcmFuY2lzY28uanBnJTIyKSUwQSUwQXByb2Nlc3NvciUyMCUzRCUyMEF1dG9Qcm9jZXNzb3IuZnJvbV9wcmV0cmFpbmVkKCUyMlJoeW1lcy1BSSUyRkFyaWElMjIpJTBBbW9kZWwlMjAlM0QlMjBBdXRvTW9kZWwuZnJvbV9wcmV0cmFpbmVkKCUyMlJoeW1lcy1BSSUyRkFyaWElMjIlMkMlMjBkdHlwZSUzRHRvcmNoLmJmbG9hdDE2JTJDJTIwZGV2aWNlX21hcCUzRCUyMmF1dG8lMjIpJTBBJTBBJTIzJTIwQ3JlYXRlJTIwaW5wdXRzJTBBbWVzc2FnZXMlMjAlM0QlMjAlNUIlMEElMjAlMjAlMjAlMjAlN0IlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjJyb2xlJTIyJTNBJTIwJTIydXNlciUyMiUyQyUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMmNvbnRlbnQlMjIlM0ElMjAlNUIlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlN0IlMjJ0eXBlJTIyJTNBJTIwJTIyaW1hZ2UlMjIlN0QlMkMlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlN0IlMjJ0eXBlJTIyJTNBJTIwJTIydGV4dCUyMiUyQyUyMCUyMnRleHQlMjIlM0ElMjAlMjJJbiUyMHRoaXMlMjBpbWFnZSUyQyUyMHdlJTIwY2FuJTIwc2VlJTIwdGhlJTIwY2l0eSUyMG9mJTIwTmV3JTIwWW9yayUyQyUyMGFuZCUyMG1vcmUlMjBzcGVjaWZpY2FsbHklMjB0aGUlMjBTdGF0dWUlMjBvZiUyMExpYmVydHkuJTIyJTdEJTJDJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTdCJTIydHlwZSUyMiUzQSUyMCUyMmltYWdlJTIyJTdEJTJDJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTdCJTIydHlwZSUyMiUzQSUyMCUyMnRleHQlMjIlMkMlMjAlMjJ0ZXh0JTIyJTNBJTIwJTIyV2hhdCUyMGNhbiUyMHdlJTIwc2VlJTIwaW4lMjB0aGlzJTIwaW1hZ2UlM0YlMjIlN0QlMkMlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlNUQlMEElMjAlMjAlMjAlMjAlN0QlMkMlMEElMjAlMjAlMjAlMjAlN0IlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjJyb2xlJTIyJTNBJTIwJTIydXNlciUyMiUyQyUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMmNvbnRlbnQlMjIlM0ElMjAlNUIlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlN0IlMjJ0eXBlJTIyJTNBJTIwJTIyaW1hZ2UlMjIlN0QlMkMlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlN0IlMjJ0eXBlJTIyJTNBJTIwJTIydGV4dCUyMiUyQyUyMCUyMnRleHQlMjIlM0ElMjAlMjJJbiUyMHdoaWNoJTIwY2l0eSUyMGlzJTIwdGhhdCUyMGJyaWRnZSUyMGxvY2F0ZWQlM0YlMjIlN0QlMkMlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlNUQlMEElMjAlMjAlMjAlMjAlN0QlMEElNUQlMEElMEFwcm9tcHRzJTIwJTNEJTIwJTVCcHJvY2Vzc29yLmFwcGx5X2NoYXRfdGVtcGxhdGUoJTVCbWVzc2FnZSU1RCUyQyUyMGFkZF9nZW5lcmF0aW9uX3Byb21wdCUzRFRydWUpJTIwZm9yJTIwbWVzc2FnZSUyMGluJTIwbWVzc2FnZXMlNUQlMEFpbWFnZXMlMjAlM0QlMjAlNUIlNUJpbWFnZTElMkMlMjBpbWFnZTIlNUQlMkMlMjAlNUJpbWFnZTMlNUQlNUQlMEFpbnB1dHMlMjAlM0QlMjBwcm9jZXNzb3IodGV4dCUzRHByb21wdHMlMkMlMjBpbWFnZXMlM0RpbWFnZXMlMkMlMjBwYWRkaW5nJTNEVHJ1ZSUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpLnRvKG1vZGVsLmRldmljZSklMEElMEElMjMlMjBHZW5lcmF0ZSUwQWdlbmVyYXRlZF9pZHMlMjAlM0QlMjBtb2RlbC5nZW5lcmF0ZSgqKmlucHV0cyUyQyUyMG1heF9uZXdfdG9rZW5zJTNEMjU2KSUwQWdlbmVyYXRlZF90ZXh0cyUyMCUzRCUyMHByb2Nlc3Nvci5iYXRjaF9kZWNvZGUoZ2VuZXJhdGVkX2lkcyUyQyUyMHNraXBfc3BlY2lhbF90b2tlbnMlM0RUcnVlKSUwQSUwQXByaW50KGdlbmVyYXRlZF90ZXh0cyU1QjAlNUQpJTBBJTBBcHJpbnQoZ2VuZXJhdGVkX3RleHRzJTVCMSU1RCk=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> requests
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> PIL <span class="hljs-keyword">import</span> Image
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> io <span class="hljs-keyword">import</span> BytesIO

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoProcessor, AutoModel
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers.image_utils <span class="hljs-keyword">import</span> load_image

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Note that passing the image urls (instead of the actual pil images) to the processor is also possible</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>image1 = load_image(<span class="hljs-string">&quot;https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>image2 = load_image(<span class="hljs-string">&quot;https://cdn.britannica.com/59/94459-050-DBA42467/Skyline-Chicago.jpg&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>image3 = load_image(<span class="hljs-string">&quot;https://cdn.britannica.com/68/170868-050-8DDE8263/Golden-Gate-Bridge-San-Francisco.jpg&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>processor = AutoProcessor.from_pretrained(<span class="hljs-string">&quot;Rhymes-AI/Aria&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = AutoModel.from_pretrained(<span class="hljs-string">&quot;Rhymes-AI/Aria&quot;</span>, dtype=torch.bfloat16, device_map=<span class="hljs-string">&quot;auto&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Create inputs</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>messages = [
<span class="hljs-meta">... </span>    {
<span class="hljs-meta">... </span>        <span class="hljs-string">&quot;role&quot;</span>: <span class="hljs-string">&quot;user&quot;</span>,
<span class="hljs-meta">... </span>        <span class="hljs-string">&quot;content&quot;</span>: [
<span class="hljs-meta">... </span>            {<span class="hljs-string">&quot;type&quot;</span>: <span class="hljs-string">&quot;image&quot;</span>},
<span class="hljs-meta">... </span>            {<span class="hljs-string">&quot;type&quot;</span>: <span class="hljs-string">&quot;text&quot;</span>, <span class="hljs-string">&quot;text&quot;</span>: <span class="hljs-string">&quot;In this image, we can see the city of New York, and more specifically the Statue of Liberty.&quot;</span>},
<span class="hljs-meta">... </span>            {<span class="hljs-string">&quot;type&quot;</span>: <span class="hljs-string">&quot;image&quot;</span>},
<span class="hljs-meta">... </span>            {<span class="hljs-string">&quot;type&quot;</span>: <span class="hljs-string">&quot;text&quot;</span>, <span class="hljs-string">&quot;text&quot;</span>: <span class="hljs-string">&quot;What can we see in this image?&quot;</span>},
<span class="hljs-meta">... </span>        ]
<span class="hljs-meta">... </span>    },
<span class="hljs-meta">... </span>    {
<span class="hljs-meta">... </span>        <span class="hljs-string">&quot;role&quot;</span>: <span class="hljs-string">&quot;user&quot;</span>,
<span class="hljs-meta">... </span>        <span class="hljs-string">&quot;content&quot;</span>: [
<span class="hljs-meta">... </span>            {<span class="hljs-string">&quot;type&quot;</span>: <span class="hljs-string">&quot;image&quot;</span>},
<span class="hljs-meta">... </span>            {<span class="hljs-string">&quot;type&quot;</span>: <span class="hljs-string">&quot;text&quot;</span>, <span class="hljs-string">&quot;text&quot;</span>: <span class="hljs-string">&quot;In which city is that bridge located?&quot;</span>},
<span class="hljs-meta">... </span>        ]
<span class="hljs-meta">... </span>    }
<span class="hljs-meta">... </span>]

<span class="hljs-meta">&gt;&gt;&gt; </span>prompts = [processor.apply_chat_template([message], add_generation_prompt=<span class="hljs-literal">True</span>) <span class="hljs-keyword">for</span> message <span class="hljs-keyword">in</span> messages]
<span class="hljs-meta">&gt;&gt;&gt; </span>images = [[image1, image2], [image3]]
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = processor(text=prompts, images=images, padding=<span class="hljs-literal">True</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>).to(model.device)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Generate</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>generated_ids = model.generate(**inputs, max_new_tokens=<span class="hljs-number">256</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=<span class="hljs-literal">True</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">print</span>(generated_texts[<span class="hljs-number">0</span>])
Assistant: There are buildings, trees, lights, <span class="hljs-keyword">and</span> water visible <span class="hljs-keyword">in</span> this image.

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">print</span>(generated_texts[<span class="hljs-number">1</span>])
Assistant: The bridge <span class="hljs-keyword">is</span> <span class="hljs-keyword">in</span> San Francisco.`,wrap:!1}}),{c(){t=l("p"),t.textContent=y,i=s(),m(T.$$.fragment)},l(M){t=d(M,"P",{"data-svelte-h":!0}),_(t)!=="svelte-11lpom8"&&(t.textContent=y),i=r(M),p(T.$$.fragment,M)},m(M,U){c(M,t,U),c(M,i,U),h(T,M,U),b=!0},p:H,i(M){b||(g(T.$$.fragment,M),b=!0)},o(M){u(T.$$.fragment,M),b=!1},d(M){M&&(a(t),a(i)),f(T,M)}}}function Yn(v){let t,y,i,T,b,M="<em>This model was released on 2024-10-08 and added to Hugging Face Transformers on 2024-12-06.</em>",U,N,Do='<div class="flex flex-wrap space-x-1"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/> <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat"/> <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/></div>',Ut,ce,It,me,Yo='<a href="https://huggingface.co/papers/2410.05993" rel="nofollow">Aria</a> is a multimodal mixture-of-experts (MoE) model. The goal of this model is to open-source a training recipe for creating a multimodal native model from scratch. Aria has 3.9B and 3.5B activated parameters per visual and text token respectively. Text is handled by a MoE decoder and visual inputs are handled by a lightweight visual encoder. It is trained in 4 stages, language pretraining, multimodal pretraining, multimodal long-context pretraining, and multimodal post-training.',Jt,pe,Oo='You can find all the original Aria checkpoints under the <a href="https://huggingface.co/rhymes-ai?search_models=aria" rel="nofollow">Aria</a> organization.',$t,V,zt,he,Ko='The example below demonstrates how to generate text based on an image with <a href="/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.Pipeline">Pipeline</a> or the <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel">AutoModel</a> class.',qt,S,Ft,ge,en='Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the <a href="../quantization/overview">Quantization</a> overview for more available quantization backends.',Bt,ue,tn='The example below uses <a href="../quantization/torchao">torchao</a> to only quantize the weights to int4 and the <a href="https://huggingface.co/rhymes-ai/Aria-sequential_mlp" rel="nofollow">rhymes-ai/Aria-sequential_mlp</a> checkpoint. This checkpoint replaces grouped GEMM with <code>torch.nn.Linear</code> layers for easier quantization.',Nt,fe,Wt,_e,Pt,k,Me,so,De,on=`A vision processor for the Aria model that handles image preprocessing.
Initialize the AriaImageProcessor.`,ro,D,ye,io,Ye,nn="Process an image with variable resolutions by dividing it into patches.",lo,Y,Te,co,Oe,an="A utility that returns number of image patches for a given image size.",mo,O,be,po,Ke,sn=`Pads the <code>image</code> with the specified <code>padding</code> and <code>mode</code>. Padding can be in the (<code>height</code>, <code>width</code>)
dimension of in the (<code>num_patches</code>) dimension. In the second case an iterable if tuples is expected
as input.`,ho,K,ve,go,et,rn="Process a list of images.",Zt,we,Rt,L,je,uo,tt,ln="AriaProcessor is a processor for the Aria model which wraps the Aria image preprocessor and the LLama slow tokenizer.",Et,xe,Lt,Q,Ae,fo,ot,dn=`This class handles the configuration for the text component of the Aria model.
Instantiating a configuration with the defaults will yield a similar configuration to that of the model of the Aria
<a href="https://huggingface.co/rhymes-ai/Aria" rel="nofollow">rhymes-ai/Aria</a> architecture.
This class extends the LlamaConfig to include additional parameters specific to the Mixture of Experts (MoE) architecture.`,Qt,ke,Gt,W,Ce,_o,nt,cn=`This class handles the configuration for both vision and text components of the Aria model,
as well as additional parameters for image token handling and projector mapping.
Instantiating a configuration with the defaults will yield a similar configuration to that of the model of the Aria
<a href="https://huggingface.co/rhymes-ai/Aria" rel="nofollow">rhymes-ai/Aria</a> architecture.`,Mo,at,mn=`Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> and can be used to control the model outputs. Read the
documentation from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> for more information.`,Xt,Ue,Ht,I,Ie,yo,st,pn="The bare Aria Text Model outputting raw hidden-states without any specific head on to.",To,rt,hn=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,bo,it,gn=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,vo,P,Je,wo,lt,un='The <a href="/docs/transformers/v4.56.2/en/model_doc/aria#transformers.AriaTextModel">AriaTextModel</a> forward method, overrides the <code>__call__</code> special method.',jo,ee,Vt,$e,St,x,ze,xo,dt,fn="The Aria model which consists of a vision backbone and a language model, without a language modeling head.",Ao,ct,_n=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,ko,mt,Mn=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Co,Z,qe,Uo,pt,yn='The <a href="/docs/transformers/v4.56.2/en/model_doc/aria#transformers.AriaModel">AriaModel</a> forward method, overrides the <code>__call__</code> special method.',Io,te,Jo,oe,Fe,$o,ht,Tn="Obtains image last hidden states from the vision tower and apply multimodal projection.",zo,ne,Be,qo,gt,bn=`Obtains multimodal placeholder mask from <code>input_ids</code> or <code>inputs_embeds</code>, and checks that the placeholder token count is
equal to the length of multimodal features. If the lengths are different, an error is raised.`,Dt,Ne,Yt,J,We,Fo,ut,vn="The Aria Model for causal language modeling.",Bo,ft,wn=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,No,_t,jn=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Wo,q,Pe,Po,Mt,xn='The <a href="/docs/transformers/v4.56.2/en/model_doc/aria#transformers.AriaTextForCausalLM">AriaTextForCausalLM</a> forward method, overrides the <code>__call__</code> special method.',Zo,ae,Ro,se,Ot,Ze,Kt,C,Re,Eo,yt,An="Aria model for conditional generation tasks.",Lo,Tt,kn=`This model combines a vision tower, a multi-modal projector, and a language model
to perform tasks that involve both image and text inputs.`,Qo,bt,Cn=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Go,vt,Un=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Xo,F,Ee,Ho,wt,In='The <a href="/docs/transformers/v4.56.2/en/model_doc/aria#transformers.AriaForConditionalGeneration">AriaForConditionalGeneration</a> forward method, overrides the <code>__call__</code> special method.',Vo,re,So,ie,eo,Le,to,At,oo;return ce=new X({props:{title:"Aria",local:"aria",headingTag:"h1"}}),V=new kt({props:{warning:!1,$$slots:{default:[Rn]},$$scope:{ctx:v}}}),S=new Zn({props:{id:"usage",options:["Pipeline","AutoModel"],$$slots:{default:[Qn]},$$scope:{ctx:v}}}),fe=new Ct({props:{code:"JTIzJTIwcGlwJTIwaW5zdGFsbCUyMHRvcmNoYW8lMEFpbXBvcnQlMjB0b3JjaCUwQWZyb20lMjB0cmFuc2Zvcm1lcnMlMjBpbXBvcnQlMjBUb3JjaEFvQ29uZmlnJTJDJTIwQXV0b01vZGVsRm9yQ2F1c2FsTE0lMkMlMjBBdXRvUHJvY2Vzc29yJTBBJTBBcXVhbnRpemF0aW9uX2NvbmZpZyUyMCUzRCUyMFRvcmNoQW9Db25maWcoJTIyaW50NF93ZWlnaHRfb25seSUyMiUyQyUyMGdyb3VwX3NpemUlM0QxMjgpJTBBbW9kZWwlMjAlM0QlMjBBdXRvTW9kZWxGb3JDYXVzYWxMTS5mcm9tX3ByZXRyYWluZWQoJTBBJTIwJTIwJTIwJTIwJTIycmh5bWVzLWFpJTJGQXJpYS1zZXF1ZW50aWFsX21scCUyMiUyQyUwQSUyMCUyMCUyMCUyMGR0eXBlJTNEdG9yY2guYmZsb2F0MTYlMkMlMEElMjAlMjAlMjAlMjBkZXZpY2VfbWFwJTNEJTIyYXV0byUyMiUyQyUwQSUyMCUyMCUyMCUyMHF1YW50aXphdGlvbl9jb25maWclM0RxdWFudGl6YXRpb25fY29uZmlnJTBBKSUwQXByb2Nlc3NvciUyMCUzRCUyMEF1dG9Qcm9jZXNzb3IuZnJvbV9wcmV0cmFpbmVkKCUwQSUyMCUyMCUyMCUyMCUyMnJoeW1lcy1haSUyRkFyaWEtc2VxdWVudGlhbF9tbHAlMjIlMkMlMEEpJTBBJTBBbWVzc2FnZXMlMjAlM0QlMjAlNUIlMEElMjAlMjAlMjAlMjAlN0IlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjJyb2xlJTIyJTNBJTIwJTIydXNlciUyMiUyQyUyMCUyMmNvbnRlbnQlMjIlM0ElMjAlNUIlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlN0IlMjJ0eXBlJTIyJTNBJTIwJTIyaW1hZ2UlMjIlMkMlMjAlMjJ1cmwlMjIlM0ElMjAlMjJodHRwcyUzQSUyRiUyRmh1Z2dpbmdmYWNlLmNvJTJGZGF0YXNldHMlMkZodWdnaW5nZmFjZSUyRmRvY3VtZW50YXRpb24taW1hZ2VzJTJGcmVzb2x2ZSUyRm1haW4lMkZwaXBlbGluZS1jYXQtY2hvbmsuanBlZyUyMiU3RCUyQyUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCU3QiUyMnR5cGUlMjIlM0ElMjAlMjJ0ZXh0JTIyJTJDJTIwJTIydGV4dCUyMiUzQSUyMCUyMldoYXQlMjBpcyUyMHNob3duJTIwaW4lMjB0aGlzJTIwaW1hZ2UlM0YlMjIlN0QlMkMlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlNUQlMEElMjAlMjAlMjAlMjAlN0QlMkMlMEElNUQlMEElMEFpbnB1dHMlMjAlM0QlMjBwcm9jZXNzb3IuYXBwbHlfY2hhdF90ZW1wbGF0ZShtZXNzYWdlcyUyQyUyMGFkZF9nZW5lcmF0aW9uX3Byb21wdCUzRFRydWUlMkMlMjB0b2tlbml6ZSUzRFRydWUlMkMlMjByZXR1cm5fZGljdCUzRFRydWUlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQWlucHV0cyUyMCUzRCUyMGlucHV0cy50byhtb2RlbC5kZXZpY2UlMkMlMjB0b3JjaC5iZmxvYXQxNiklMEElMEFvdXRwdXQlMjAlM0QlMjBtb2RlbC5nZW5lcmF0ZSglMEElMjAlMjAlMjAlMjAqKmlucHV0cyUyQyUwQSUyMCUyMCUyMCUyMG1heF9uZXdfdG9rZW5zJTNEMTUlMkMlMEElMjAlMjAlMjAlMjBzdG9wX3N0cmluZ3MlM0QlNUIlMjIlM0MlN0NpbV9lbmQlN0MlM0UlMjIlNUQlMkMlMEElMjAlMjAlMjAlMjB0b2tlbml6ZXIlM0Rwcm9jZXNzb3IudG9rZW5pemVyJTJDJTBBJTIwJTIwJTIwJTIwZG9fc2FtcGxlJTNEVHJ1ZSUyQyUwQSUyMCUyMCUyMCUyMHRlbXBlcmF0dXJlJTNEMC45JTJDJTBBKSUwQW91dHB1dF9pZHMlMjAlM0QlMjBvdXRwdXQlNUIwJTVEJTVCaW5wdXRzJTVCJTIyaW5wdXRfaWRzJTIyJTVELnNoYXBlJTVCMSU1RCUzQSU1RCUwQXJlc3BvbnNlJTIwJTNEJTIwcHJvY2Vzc29yLmRlY29kZShvdXRwdXRfaWRzJTJDJTIwc2tpcF9zcGVjaWFsX3Rva2VucyUzRFRydWUpJTBBcHJpbnQocmVzcG9uc2Up",highlighted:`<span class="hljs-comment"># pip install torchao</span>
<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> TorchAoConfig, AutoModelForCausalLM, AutoProcessor

quantization_config = TorchAoConfig(<span class="hljs-string">&quot;int4_weight_only&quot;</span>, group_size=<span class="hljs-number">128</span>)
model = AutoModelForCausalLM.from_pretrained(
    <span class="hljs-string">&quot;rhymes-ai/Aria-sequential_mlp&quot;</span>,
    dtype=torch.bfloat16,
    device_map=<span class="hljs-string">&quot;auto&quot;</span>,
    quantization_config=quantization_config
)
processor = AutoProcessor.from_pretrained(
    <span class="hljs-string">&quot;rhymes-ai/Aria-sequential_mlp&quot;</span>,
)

messages = [
    {
        <span class="hljs-string">&quot;role&quot;</span>: <span class="hljs-string">&quot;user&quot;</span>, <span class="hljs-string">&quot;content&quot;</span>: [
            {<span class="hljs-string">&quot;type&quot;</span>: <span class="hljs-string">&quot;image&quot;</span>, <span class="hljs-string">&quot;url&quot;</span>: <span class="hljs-string">&quot;https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg&quot;</span>},
            {<span class="hljs-string">&quot;type&quot;</span>: <span class="hljs-string">&quot;text&quot;</span>, <span class="hljs-string">&quot;text&quot;</span>: <span class="hljs-string">&quot;What is shown in this image?&quot;</span>},
        ]
    },
]

inputs = processor.apply_chat_template(messages, add_generation_prompt=<span class="hljs-literal">True</span>, tokenize=<span class="hljs-literal">True</span>, return_dict=<span class="hljs-literal">True</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
inputs = inputs.to(model.device, torch.bfloat16)

output = model.generate(
    **inputs,
    max_new_tokens=<span class="hljs-number">15</span>,
    stop_strings=[<span class="hljs-string">&quot;&lt;|im_end|&gt;&quot;</span>],
    tokenizer=processor.tokenizer,
    do_sample=<span class="hljs-literal">True</span>,
    temperature=<span class="hljs-number">0.9</span>,
)
output_ids = output[<span class="hljs-number">0</span>][inputs[<span class="hljs-string">&quot;input_ids&quot;</span>].shape[<span class="hljs-number">1</span>]:]
response = processor.decode(output_ids, skip_special_tokens=<span class="hljs-literal">True</span>)
<span class="hljs-built_in">print</span>(response)`,wrap:!1}}),_e=new X({props:{title:"AriaImageProcessor",local:"transformers.AriaImageProcessor",headingTag:"h2"}}),Me=new A({props:{name:"class transformers.AriaImageProcessor",anchor:"transformers.AriaImageProcessor",parameters:[{name:"image_mean",val:": typing.Optional[list[float]] = None"},{name:"image_std",val:": typing.Optional[list[float]] = None"},{name:"max_image_size",val:": int = 980"},{name:"min_image_size",val:": int = 336"},{name:"split_resolutions",val:": typing.Optional[list[tuple[int, int]]] = None"},{name:"split_image",val:": typing.Optional[bool] = False"},{name:"do_convert_rgb",val:": typing.Optional[bool] = True"},{name:"do_rescale",val:": bool = True"},{name:"rescale_factor",val:": typing.Union[int, float] = 0.00392156862745098"},{name:"do_normalize",val:": typing.Optional[bool] = True"},{name:"resample",val:": Resampling = <Resampling.BICUBIC: 3>"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.AriaImageProcessor.image_mean",description:`<strong>image_mean</strong> (<code>list</code>, <em>optional</em>, defaults to [0.5, 0.5, 0.5]) &#x2014;
Mean values for normalization.`,name:"image_mean"},{anchor:"transformers.AriaImageProcessor.image_std",description:`<strong>image_std</strong> (<code>list</code>, <em>optional</em>, defaults to [0.5, 0.5, 0.5]) &#x2014;
Standard deviation values for normalization.`,name:"image_std"},{anchor:"transformers.AriaImageProcessor.max_image_size",description:`<strong>max_image_size</strong> (<code>int</code>, <em>optional</em>, defaults to 980) &#x2014;
Maximum image size.`,name:"max_image_size"},{anchor:"transformers.AriaImageProcessor.min_image_size",description:`<strong>min_image_size</strong> (<code>int</code>, <em>optional</em>, defaults to 336) &#x2014;
Minimum image size.`,name:"min_image_size"},{anchor:"transformers.AriaImageProcessor.split_resolutions",description:`<strong>split_resolutions</strong> (<code>list</code>, <em>optional</em>, defaults to a list of optimal,resolutions as tuples) &#x2014;
The optimal resolutions for splitting the image.`,name:"split_resolutions"},{anchor:"transformers.AriaImageProcessor.split_image",description:`<strong>split_image</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether to split the image.`,name:"split_image"},{anchor:"transformers.AriaImageProcessor.do_convert_rgb",description:`<strong>do_convert_rgb</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether to convert the image to RGB.`,name:"do_convert_rgb"},{anchor:"transformers.AriaImageProcessor.do_rescale",description:`<strong>do_rescale</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether to rescale the image by the specified scale <code>rescale_factor</code>. Can be overridden by <code>do_rescale</code> in
the <code>preprocess</code> method.`,name:"do_rescale"},{anchor:"transformers.AriaImageProcessor.rescale_factor",description:`<strong>rescale_factor</strong> (<code>int</code> or <code>float</code>, <em>optional</em>, defaults to <code>1/255</code>) &#x2014;
Scale factor to use if rescaling the image. Can be overridden by <code>rescale_factor</code> in the <code>preprocess</code>
method.`,name:"rescale_factor"},{anchor:"transformers.AriaImageProcessor.do_normalize",description:`<strong>do_normalize</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether to normalize the image.`,name:"do_normalize"},{anchor:"transformers.AriaImageProcessor.resample",description:`<strong>resample</strong> (PILImageResampling, <em>optional</em>, defaults to <code>BICUBIC</code>) &#x2014;
The resampling filter to use if resizing the image.`,name:"resample"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/aria/image_processing_aria.py#L74"}}),ye=new A({props:{name:"get_image_patches",anchor:"transformers.AriaImageProcessor.get_image_patches",parameters:[{name:"image",val:": <built-in function array>"},{name:"grid_pinpoints",val:": list"},{name:"patch_size",val:": int"},{name:"resample",val:": Resampling"},{name:"data_format",val:": ChannelDimension"},{name:"input_data_format",val:": ChannelDimension"}],parametersDescription:[{anchor:"transformers.AriaImageProcessor.get_image_patches.image",description:`<strong>image</strong> (<code>np.array</code>) &#x2014;
The input image to be processed.`,name:"image"},{anchor:"transformers.AriaImageProcessor.get_image_patches.grid_pinpoints",description:`<strong>grid_pinpoints</strong> (list[tuple[int, int]]) &#x2014;
A list of possible resolutions as tuples.`,name:"grid_pinpoints"},{anchor:"transformers.AriaImageProcessor.get_image_patches.patch_size",description:`<strong>patch_size</strong> (<code>int</code>) &#x2014;
Size of the patches to divide the image into.`,name:"patch_size"},{anchor:"transformers.AriaImageProcessor.get_image_patches.resample",description:`<strong>resample</strong> (<code>PILImageResampling</code>) &#x2014;
Resampling filter to use if resizing the image.`,name:"resample"},{anchor:"transformers.AriaImageProcessor.get_image_patches.data_format",description:`<strong>data_format</strong> (<code>ChannelDimension</code> or <code>str</code>) &#x2014;
The channel dimension format for the output image.`,name:"data_format"},{anchor:"transformers.AriaImageProcessor.get_image_patches.input_data_format",description:`<strong>input_data_format</strong> (<code>ChannelDimension</code> or <code>str</code>) &#x2014;
The channel dimension format of the input image.`,name:"input_data_format"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/aria/image_processing_aria.py#L455",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A list of NumPy arrays containing the processed image patches.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>list[np.array]</code></p>
`}}),Te=new A({props:{name:"get_number_of_image_patches",anchor:"transformers.AriaImageProcessor.get_number_of_image_patches",parameters:[{name:"height",val:": int"},{name:"width",val:": int"},{name:"images_kwargs",val:" = None"}],parametersDescription:[{anchor:"transformers.AriaImageProcessor.get_number_of_image_patches.height",description:`<strong>height</strong> (<code>int</code>) &#x2014;
Height of the input image.`,name:"height"},{anchor:"transformers.AriaImageProcessor.get_number_of_image_patches.width",description:`<strong>width</strong> (<code>int</code>) &#x2014;
Width of the input image.`,name:"width"},{anchor:"transformers.AriaImageProcessor.get_number_of_image_patches.images_kwargs",description:`<strong>images_kwargs</strong> (<code>dict</code>, <em>optional</em>) &#x2014;
Any kwargs to override defaults of the image processor.`,name:"images_kwargs"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/aria/image_processing_aria.py#L505",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>Number of patches per image.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>int</code></p>
`}}),be=new A({props:{name:"pad",anchor:"transformers.AriaImageProcessor.pad",parameters:[{name:"image",val:": ndarray"},{name:"padding",val:": typing.Union[int, tuple[int, int], collections.abc.Iterable[tuple[int, int]]]"},{name:"mode",val:": PaddingMode = <PaddingMode.CONSTANT: 'constant'>"},{name:"constant_values",val:": typing.Union[float, collections.abc.Iterable[float]] = 0.0"},{name:"data_format",val:": typing.Union[str, transformers.image_utils.ChannelDimension, NoneType] = None"},{name:"input_data_format",val:": typing.Union[str, transformers.image_utils.ChannelDimension, NoneType] = None"}],parametersDescription:[{anchor:"transformers.AriaImageProcessor.pad.image",description:`<strong>image</strong> (<code>np.ndarray</code>) &#x2014;
The image to pad.`,name:"image"},{anchor:"transformers.AriaImageProcessor.pad.padding",description:`<strong>padding</strong> (<code>int</code> or <code>tuple[int, int]</code> or <code>Iterable[tuple[int, int]]</code>) &#x2014;
Padding to apply to the edges of the height, width axes. Can be one of three formats:<ul>
<li><code>((before_height, after_height), (before_width, after_width))</code> unique pad widths for each axis.</li>
<li><code>((before, after),)</code> yields same before and after pad for height and width.</li>
<li><code>(pad,)</code> or int is a shortcut for before = after = pad width for all axes.</li>
</ul>`,name:"padding"},{anchor:"transformers.AriaImageProcessor.pad.mode",description:`<strong>mode</strong> (<code>PaddingMode</code>) &#x2014;
The padding mode to use. Can be one of:<ul>
<li><code>&quot;constant&quot;</code>: pads with a constant value.</li>
<li><code>&quot;reflect&quot;</code>: pads with the reflection of the vector mirrored on the first and last values of the
vector along each axis.</li>
<li><code>&quot;replicate&quot;</code>: pads with the replication of the last value on the edge of the array along each axis.</li>
<li><code>&quot;symmetric&quot;</code>: pads with the reflection of the vector mirrored along the edge of the array.</li>
</ul>`,name:"mode"},{anchor:"transformers.AriaImageProcessor.pad.constant_values",description:`<strong>constant_values</strong> (<code>float</code> or <code>Iterable[float]</code>, <em>optional</em>) &#x2014;
The value to use for the padding if <code>mode</code> is <code>&quot;constant&quot;</code>.`,name:"constant_values"},{anchor:"transformers.AriaImageProcessor.pad.data_format",description:`<strong>data_format</strong> (<code>str</code> or <code>ChannelDimension</code>, <em>optional</em>) &#x2014;
The channel dimension format for the output image. Can be one of:<ul>
<li><code>&quot;channels_first&quot;</code> or <code>ChannelDimension.FIRST</code>: image in (num_channels, height, width) format.</li>
<li><code>&quot;channels_last&quot;</code> or <code>ChannelDimension.LAST</code>: image in (height, width, num_channels) format.
If unset, will use same as the input image.</li>
</ul>`,name:"data_format"},{anchor:"transformers.AriaImageProcessor.pad.input_data_format",description:`<strong>input_data_format</strong> (<code>str</code> or <code>ChannelDimension</code>, <em>optional</em>) &#x2014;
The channel dimension format for the input image. Can be one of:<ul>
<li><code>&quot;channels_first&quot;</code> or <code>ChannelDimension.FIRST</code>: image in (num_channels, height, width) format.</li>
<li><code>&quot;channels_last&quot;</code> or <code>ChannelDimension.LAST</code>: image in (height, width, num_channels) format.
If unset, will use the inferred format of the input image.</li>
</ul>`,name:"input_data_format"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/aria/image_processing_aria.py#L389",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>The padded image.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>np.ndarray</code></p>
`}}),ve=new A({props:{name:"preprocess",anchor:"transformers.AriaImageProcessor.preprocess",parameters:[{name:"images",val:": typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor'], list[typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']]]]"},{name:"image_mean",val:": typing.Union[float, list[float], NoneType] = None"},{name:"image_std",val:": typing.Union[float, list[float], NoneType] = None"},{name:"max_image_size",val:": typing.Optional[int] = None"},{name:"min_image_size",val:": typing.Optional[int] = None"},{name:"split_image",val:": typing.Optional[bool] = None"},{name:"do_convert_rgb",val:": typing.Optional[bool] = None"},{name:"do_rescale",val:": typing.Optional[bool] = None"},{name:"rescale_factor",val:": typing.Optional[float] = None"},{name:"do_normalize",val:": typing.Optional[bool] = None"},{name:"resample",val:": Resampling = None"},{name:"return_tensors",val:": typing.Union[str, transformers.utils.generic.TensorType, NoneType] = 'pt'"},{name:"data_format",val:": typing.Optional[transformers.image_utils.ChannelDimension] = <ChannelDimension.FIRST: 'channels_first'>"},{name:"input_data_format",val:": typing.Union[str, transformers.image_utils.ChannelDimension, NoneType] = None"}],parametersDescription:[{anchor:"transformers.AriaImageProcessor.preprocess.images",description:`<strong>images</strong> (ImageInput or list of ImageInput) &#x2014;
The input image or a list of images.`,name:"images"},{anchor:"transformers.AriaImageProcessor.preprocess.image_mean",description:`<strong>image_mean</strong> (<code>list</code>, <em>optional</em>, defaults to [0.5, 0.5, 0.5]) &#x2014;
Mean values for normalization.`,name:"image_mean"},{anchor:"transformers.AriaImageProcessor.preprocess.image_std",description:`<strong>image_std</strong> (<code>list</code>, <em>optional</em>, defaults to [0.5, 0.5, 0.5]) &#x2014;
Standard deviation values for normalization.`,name:"image_std"},{anchor:"transformers.AriaImageProcessor.preprocess.max_image_size",description:`<strong>max_image_size</strong> (<code>int</code>, <em>optional</em>, defaults to <code>self.max_image_size</code> (980)) &#x2014;
Maximum image size.`,name:"max_image_size"},{anchor:"transformers.AriaImageProcessor.preprocess.min_image_size",description:`<strong>min_image_size</strong> (<code>int</code>, <em>optional</em>, defaults to <code>self.min_image_size</code> (336)) &#x2014;
Minimum image size.`,name:"min_image_size"},{anchor:"transformers.AriaImageProcessor.preprocess.split_image",description:`<strong>split_image</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>self.split_image</code> (False)) &#x2014;
Whether to split the image.`,name:"split_image"},{anchor:"transformers.AriaImageProcessor.preprocess.do_convert_rgb",description:`<strong>do_convert_rgb</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>self.do_convert_rgb</code> (True)) &#x2014;
Whether to convert the image to RGB.`,name:"do_convert_rgb"},{anchor:"transformers.AriaImageProcessor.preprocess.do_rescale",description:`<strong>do_rescale</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>self.do_rescale</code>) &#x2014;
Whether to rescale the image.`,name:"do_rescale"},{anchor:"transformers.AriaImageProcessor.preprocess.rescale_factor",description:`<strong>rescale_factor</strong> (<code>float</code>, <em>optional</em>, defaults to <code>self.rescale_factor</code>) &#x2014;
Rescale factor to rescale the image by if <code>do_rescale</code> is set to <code>True</code>.`,name:"rescale_factor"},{anchor:"transformers.AriaImageProcessor.preprocess.do_normalize",description:`<strong>do_normalize</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>self.do_normalize</code> (True)) &#x2014;
Whether to normalize the image.`,name:"do_normalize"},{anchor:"transformers.AriaImageProcessor.preprocess.resample",description:`<strong>resample</strong> (PILImageResampling, <em>optional</em>, defaults to <code>self.resample</code> (BICUBIC)) &#x2014;
The resampling filter to use if resizing the image.`,name:"resample"},{anchor:"transformers.AriaImageProcessor.preprocess.return_tensors",description:`<strong>return_tensors</strong> (<code>str</code> or <code>TensorType</code>, <em>optional</em>, defaults to &#x201C;pt&#x201D;) &#x2014;
The type of tensor to return.`,name:"return_tensors"},{anchor:"transformers.AriaImageProcessor.preprocess.data_format",description:`<strong>data_format</strong> (<code>str</code> or <code>ChannelDimension</code>, <em>optional</em>) &#x2014;
The channel dimension format for the output image. Can be one of:<ul>
<li><code>&quot;channels_first&quot;</code> or <code>ChannelDimension.FIRST</code>:
image in (num_channels, height, width) format.</li>
<li><code>&quot;channels_last&quot;</code> or <code>ChannelDimension.LAST</code>:
image in (height, width, num_channels) format.
If unset, will use same as the input image.</li>
</ul>`,name:"data_format"},{anchor:"transformers.AriaImageProcessor.preprocess.input_data_format",description:`<strong>input_data_format</strong> (<code>str</code> or <code>ChannelDimension</code>, <em>optional</em>) &#x2014;
The channel dimension format for the input image. Can be one of:<ul>
<li><code>&quot;channels_first&quot;</code> or <code>ChannelDimension.FIRST</code>:
image in (num_channels, height, width) format.</li>
<li><code>&quot;channels_last&quot;</code> or <code>ChannelDimension.LAST</code>:
image in (height, width, num_channels) format.
If unset, will use the inferred format of the input image.</li>
</ul>`,name:"input_data_format"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/aria/image_processing_aria.py#L144",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A BatchFeature object containing:</p>
<ul>
<li>‘pixel_values’:
Tensor of processed image pixel values.</li>
<li>‘pixel_mask’:
Boolean pixel mask. This mask is a 2D tensor of shape (max_image_size, max_image_size) where:<ul>
<li>True (1) values indicate pixels that belong to the original resized image.</li>
<li>False (0) values indicate pixels that are part of the padding.
The mask helps distinguish between actual image content and padded areas in subsequent processing steps.</li>
</ul></li>
<li>‘num_crops’:
The maximum number of crops across all images.</li>
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p>BatchFeature</p>
`}}),we=new X({props:{title:"AriaProcessor",local:"transformers.AriaProcessor",headingTag:"h2"}}),je=new A({props:{name:"class transformers.AriaProcessor",anchor:"transformers.AriaProcessor",parameters:[{name:"image_processor",val:" = None"},{name:"tokenizer",val:": typing.Union[transformers.models.auto.tokenization_auto.AutoTokenizer, str] = None"},{name:"chat_template",val:": typing.Optional[str] = None"},{name:"size_conversion",val:": typing.Optional[dict[typing.Union[float, int], int]] = None"}],parametersDescription:[{anchor:"transformers.AriaProcessor.image_processor",description:`<strong>image_processor</strong> (<code>AriaImageProcessor</code>, <em>optional</em>) &#x2014;
The AriaImageProcessor to use for image preprocessing.`,name:"image_processor"},{anchor:"transformers.AriaProcessor.tokenizer",description:`<strong>tokenizer</strong> (<code>PreTrainedTokenizerBase</code>, <em>optional</em>) &#x2014;
An instance of <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase">PreTrainedTokenizerBase</a>. This should correspond with the model&#x2019;s text model. The tokenizer is a required input.`,name:"tokenizer"},{anchor:"transformers.AriaProcessor.chat_template",description:`<strong>chat_template</strong> (<code>str</code>, <em>optional</em>) &#x2014;
A Jinja template which will be used to convert lists of messages in a chat into a tokenizable string.`,name:"chat_template"},{anchor:"transformers.AriaProcessor.size_conversion",description:`<strong>size_conversion</strong> (<code>Dict</code>, <em>optional</em>) &#x2014;
A dictionary indicating size conversions for images.`,name:"size_conversion"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/aria/processing_aria.py#L47"}}),xe=new X({props:{title:"AriaTextConfig",local:"transformers.AriaTextConfig",headingTag:"h2"}}),Ae=new A({props:{name:"class transformers.AriaTextConfig",anchor:"transformers.AriaTextConfig",parameters:[{name:"vocab_size",val:" = 32000"},{name:"hidden_size",val:" = 4096"},{name:"intermediate_size",val:": int = 4096"},{name:"num_hidden_layers",val:" = 32"},{name:"num_attention_heads",val:" = 32"},{name:"num_key_value_heads",val:" = None"},{name:"hidden_act",val:" = 'silu'"},{name:"max_position_embeddings",val:" = 2048"},{name:"initializer_range",val:" = 0.02"},{name:"rms_norm_eps",val:" = 1e-06"},{name:"use_cache",val:" = True"},{name:"pad_token_id",val:" = 2"},{name:"bos_token_id",val:" = 1"},{name:"eos_token_id",val:" = 2"},{name:"pretraining_tp",val:" = 1"},{name:"tie_word_embeddings",val:" = False"},{name:"rope_theta",val:" = 10000.0"},{name:"rope_scaling",val:" = None"},{name:"attention_bias",val:" = False"},{name:"attention_dropout",val:" = 0.0"},{name:"mlp_bias",val:" = False"},{name:"head_dim",val:" = None"},{name:"moe_num_experts",val:": int = 8"},{name:"moe_topk",val:": int = 2"},{name:"moe_num_shared_experts",val:": int = 2"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.AriaTextConfig.vocab_size",description:`<strong>vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 32000) &#x2014;
Vocabulary size of the LLaMA model. Defines the number of different tokens that can be represented by the
<code>inputs_ids</code> passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaModel">LlamaModel</a>`,name:"vocab_size"},{anchor:"transformers.AriaTextConfig.hidden_size",description:`<strong>hidden_size</strong> (<code>int</code>, <em>optional</em>, defaults to 4096) &#x2014;
Dimension of the hidden representations.`,name:"hidden_size"},{anchor:"transformers.AriaTextConfig.intermediate_size",description:`<strong>intermediate_size</strong> (<code>int</code>, <em>optional</em>, defaults to 4096) &#x2014;
The size of the MLP representations.`,name:"intermediate_size"},{anchor:"transformers.AriaTextConfig.num_hidden_layers",description:`<strong>num_hidden_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 32) &#x2014;
Number of hidden layers in the Transformer decoder.`,name:"num_hidden_layers"},{anchor:"transformers.AriaTextConfig.num_attention_heads",description:`<strong>num_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 32) &#x2014;
Number of attention heads for each attention layer in the Transformer decoder.`,name:"num_attention_heads"},{anchor:"transformers.AriaTextConfig.num_key_value_heads",description:`<strong>num_key_value_heads</strong> (<code>int</code>, <em>optional</em>) &#x2014;
This is the number of key_value heads that should be used to implement Grouped Query Attention. If
<code>num_key_value_heads=num_attention_heads</code>, the model will use Multi Head Attention (MHA), if
<code>num_key_value_heads=1</code> the model will use Multi Query Attention (MQA) otherwise GQA is used. When
converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
by meanpooling all the original heads within that group. For more details, check out <a href="https://huggingface.co/papers/2305.13245" rel="nofollow">this
paper</a>. If it is not specified, will default to
<code>num_attention_heads</code>.`,name:"num_key_value_heads"},{anchor:"transformers.AriaTextConfig.hidden_act",description:`<strong>hidden_act</strong> (<code>str</code> or <code>function</code>, <em>optional</em>, defaults to <code>&quot;silu&quot;</code>) &#x2014;
The non-linear activation function (function or string) in the decoder.`,name:"hidden_act"},{anchor:"transformers.AriaTextConfig.max_position_embeddings",description:`<strong>max_position_embeddings</strong> (<code>int</code>, <em>optional</em>, defaults to 2048) &#x2014;
The maximum sequence length that this model might ever be used with. Llama 1 supports up to 2048 tokens,
Llama 2 up to 4096, CodeLlama up to 16384.`,name:"max_position_embeddings"},{anchor:"transformers.AriaTextConfig.initializer_range",description:`<strong>initializer_range</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.`,name:"initializer_range"},{anchor:"transformers.AriaTextConfig.rms_norm_eps",description:`<strong>rms_norm_eps</strong> (<code>float</code>, <em>optional</em>, defaults to 1e-06) &#x2014;
The epsilon used by the rms normalization layers.`,name:"rms_norm_eps"},{anchor:"transformers.AriaTextConfig.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not the model should return the last key/values attentions (not used by all models). Only
relevant if <code>config.is_decoder=True</code>.`,name:"use_cache"},{anchor:"transformers.AriaTextConfig.pad_token_id",description:`<strong>pad_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 2) &#x2014;
Padding token id.`,name:"pad_token_id"},{anchor:"transformers.AriaTextConfig.bos_token_id",description:`<strong>bos_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 1) &#x2014;
Beginning of stream token id.`,name:"bos_token_id"},{anchor:"transformers.AriaTextConfig.eos_token_id",description:`<strong>eos_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 2) &#x2014;
End of stream token id.`,name:"eos_token_id"},{anchor:"transformers.AriaTextConfig.pretraining_tp",description:`<strong>pretraining_tp</strong> (<code>int</code>, <em>optional</em>, defaults to 1) &#x2014;
Experimental feature. Tensor parallelism rank used during pretraining. Please refer to <a href="https://huggingface.co/docs/transformers/main/perf_train_gpu_many#tensor-parallelism" rel="nofollow">this
document</a> to
understand more about it. This value is necessary to ensure exact reproducibility of the pretraining
results. Please refer to <a href="https://github.com/pytorch/pytorch/issues/76232" rel="nofollow">this issue</a>.`,name:"pretraining_tp"},{anchor:"transformers.AriaTextConfig.tie_word_embeddings",description:`<strong>tie_word_embeddings</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether to tie weight embeddings`,name:"tie_word_embeddings"},{anchor:"transformers.AriaTextConfig.rope_theta",description:`<strong>rope_theta</strong> (<code>float</code>, <em>optional</em>, defaults to 10000.0) &#x2014;
The base period of the RoPE embeddings.`,name:"rope_theta"},{anchor:"transformers.AriaTextConfig.rope_scaling",description:`<strong>rope_scaling</strong> (<code>Dict</code>, <em>optional</em>) &#x2014;
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
Only used with &#x2018;llama3&#x2019;. Scaling factor applied to high frequency components of the RoPE`,name:"rope_scaling"},{anchor:"transformers.AriaTextConfig.attention_bias",description:`<strong>attention_bias</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether to use a bias in the query, key, value and output projection layers during self-attention.`,name:"attention_bias"},{anchor:"transformers.AriaTextConfig.attention_dropout",description:`<strong>attention_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The dropout ratio for the attention probabilities.`,name:"attention_dropout"},{anchor:"transformers.AriaTextConfig.mlp_bias",description:`<strong>mlp_bias</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether to use a bias in up_proj, down_proj and gate_proj layers in the MLP layers.`,name:"mlp_bias"},{anchor:"transformers.AriaTextConfig.head_dim",description:`<strong>head_dim</strong> (<code>int</code>, <em>optional</em>) &#x2014;
The attention head dimension. If None, it will default to hidden_size // num_heads`,name:"head_dim"},{anchor:"transformers.AriaTextConfig.moe_num_experts",description:`<strong>moe_num_experts</strong> (<code>int</code>, <em>optional</em>, defaults to 8) &#x2014;
The number of experts in the MoE layer.`,name:"moe_num_experts"},{anchor:"transformers.AriaTextConfig.moe_topk",description:`<strong>moe_topk</strong> (<code>int</code>, <em>optional</em>, defaults to 2) &#x2014;
The number of top experts to route to for each token.`,name:"moe_topk"},{anchor:"transformers.AriaTextConfig.moe_num_shared_experts",description:`<strong>moe_num_shared_experts</strong> (<code>int</code>, <em>optional</em>, defaults to 2) &#x2014;
The number of shared experts.`,name:"moe_num_shared_experts"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/aria/configuration_aria.py#L28"}}),ke=new X({props:{title:"AriaConfig",local:"transformers.AriaConfig",headingTag:"h2"}}),Ce=new A({props:{name:"class transformers.AriaConfig",anchor:"transformers.AriaConfig",parameters:[{name:"vision_config",val:" = None"},{name:"vision_feature_layer",val:": int = -1"},{name:"text_config",val:": AriaTextConfig = None"},{name:"projector_patch_to_query_dict",val:": typing.Optional[dict] = None"},{name:"image_token_index",val:": int = 9"},{name:"initializer_range",val:": float = 0.02"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.AriaConfig.vision_config",description:`<strong>vision_config</strong> (<code>AriaVisionConfig</code> or <code>dict</code>, <em>optional</em>) &#x2014;
Configuration for the vision component.`,name:"vision_config"},{anchor:"transformers.AriaConfig.vision_feature_layer",description:`<strong>vision_feature_layer</strong> (<code>int</code>, <em>optional</em>, defaults to -1) &#x2014;
The index of the layer to select the vision feature.`,name:"vision_feature_layer"},{anchor:"transformers.AriaConfig.text_config",description:`<strong>text_config</strong> (<code>AriaTextConfig</code> or <code>dict</code>, <em>optional</em>) &#x2014;
Configuration for the text component.`,name:"text_config"},{anchor:"transformers.AriaConfig.projector_patch_to_query_dict",description:`<strong>projector_patch_to_query_dict</strong> (<code>dict</code>, <em>optional</em>) &#x2014;
Mapping of patch sizes to query dimensions.`,name:"projector_patch_to_query_dict"},{anchor:"transformers.AriaConfig.image_token_index",description:`<strong>image_token_index</strong> (<code>int</code>, <em>optional</em>, defaults to 9) &#x2014;
Index used to represent image tokens.`,name:"image_token_index"},{anchor:"transformers.AriaConfig.initializer_range",description:`<strong>initializer_range</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated normal initializer for initializing all weight matrices.`,name:"initializer_range"},{anchor:"transformers.AriaConfig.model_type",description:`<strong>model_type</strong> (<code>str</code>) &#x2014;
Type of the model, set to <code>&quot;aria&quot;</code>.`,name:"model_type"},{anchor:"transformers.AriaConfig.image_token_index",description:`<strong>image_token_index</strong> (<code>int</code>) &#x2014;
Index used to represent image tokens.`,name:"image_token_index"},{anchor:"transformers.AriaConfig.projector_patch_to_query_dict",description:`<strong>projector_patch_to_query_dict</strong> (<code>dict</code>) &#x2014;
Mapping of patch sizes to query dimensions.`,name:"projector_patch_to_query_dict"},{anchor:"transformers.AriaConfig.vision_config",description:`<strong>vision_config</strong> (<code>AriaVisionConfig</code>) &#x2014;
Configuration for the vision component.`,name:"vision_config"},{anchor:"transformers.AriaConfig.text_config",description:`<strong>text_config</strong> (<code>AriaTextConfig</code>) &#x2014;
Configuration for the text component.`,name:"text_config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/aria/configuration_aria.py#L223"}}),Ue=new X({props:{title:"AriaTextModel",local:"transformers.AriaTextModel",headingTag:"h2"}}),Ie=new A({props:{name:"class transformers.AriaTextModel",anchor:"transformers.AriaTextModel",parameters:[{name:"config",val:": AriaTextConfig"}],parametersDescription:[{anchor:"transformers.AriaTextModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/aria#transformers.AriaTextConfig">AriaTextConfig</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/aria/modeling_aria.py#L710"}}),Je=new A({props:{name:"forward",anchor:"transformers.AriaTextModel.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.AriaTextModel.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.AriaTextModel.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.AriaTextModel.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.AriaTextModel.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.AriaTextModel.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.AriaTextModel.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"},{anchor:"transformers.AriaTextModel.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/aria/modeling_aria.py#L727",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast"
>transformers.modeling_outputs.BaseModelOutputWithPast</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/aria#transformers.AriaConfig"
>AriaConfig</a>) and inputs.</p>
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
`}}),ee=new kt({props:{$$slots:{default:[Gn]},$$scope:{ctx:v}}}),$e=new X({props:{title:"AriaModel",local:"transformers.AriaModel",headingTag:"h2"}}),ze=new A({props:{name:"class transformers.AriaModel",anchor:"transformers.AriaModel",parameters:[{name:"config",val:": AriaConfig"}],parametersDescription:[{anchor:"transformers.AriaModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/aria#transformers.AriaConfig">AriaConfig</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/aria/modeling_aria.py#L921"}}),qe=new A({props:{name:"forward",anchor:"transformers.AriaModel.forward",parameters:[{name:"input_ids",val:": LongTensor = None"},{name:"pixel_values",val:": FloatTensor = None"},{name:"pixel_mask",val:": LongTensor = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.modeling_flash_attention_utils.FlashAttentionKwargs]"}],parametersDescription:[{anchor:"transformers.AriaModel.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.AriaModel.forward.pixel_values",description:`<strong>pixel_values</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_channels, image_size, image_size)</code>) &#x2014;
The tensors corresponding to the input images. Pixel values can be obtained using
<a href="/docs/transformers/v4.56.2/en/model_doc/aria#transformers.AriaImageProcessor">AriaImageProcessor</a>. See <a href="/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__">AriaImageProcessor.<strong>call</strong>()</a> for details (<a href="/docs/transformers/v4.56.2/en/model_doc/aria#transformers.AriaProcessor">AriaProcessor</a> uses
<a href="/docs/transformers/v4.56.2/en/model_doc/aria#transformers.AriaImageProcessor">AriaImageProcessor</a> for processing images).`,name:"pixel_values"},{anchor:"transformers.AriaModel.forward.pixel_mask",description:`<strong>pixel_mask</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, height, width)</code>) &#x2014;
Mask to avoid performing attention on padding pixel values. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for pixels that are real (i.e. <strong>not masked</strong>),</li>
<li>0 for pixels that are padding (i.e. <strong>masked</strong>).</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"pixel_mask"},{anchor:"transformers.AriaModel.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.AriaModel.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.AriaModel.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.AriaModel.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.AriaModel.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.AriaModel.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/aria/modeling_aria.py#L1004",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <code>transformers.models.aria.modeling_aria.AriaModelOutputWithPast</code> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/aria#transformers.AriaConfig"
>AriaConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>last_hidden_state</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) — Sequence of hidden-states at the output of the last layer of the model.</p>
</li>
<li>
<p><strong>past_key_values</strong> (<code>Cache</code>, <em>optional</em>, returned when <code>use_cache=True</code> is passed or when <code>config.use_cache=True</code>) — Tuple of <code>tuple(torch.FloatTensor)</code> of length <code>config.n_layers</code>, with each tuple having 2 tensors of shape
<code>(batch_size, num_heads, sequence_length, embed_size_per_head)</code>)</p>
<p>Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
<code>past_key_values</code> input) to speed up sequential decoding.</p>
</li>
<li>
<p><strong>hidden_states</strong> (<code>tuple[torch.FloatTensor, ...]</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings, if the model has an embedding layer, +
one for the output of each layer) of shape <code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.</p>
</li>
<li>
<p><strong>attentions</strong> (<code>tuple[torch.FloatTensor, ...]</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
heads.</p>
</li>
<li>
<p><strong>image_hidden_states</strong> (<code>torch.FloatTensor</code>, <em>optional</em>) — A <code>torch.FloatTensor</code> of size <code>(batch_size, num_images, sequence_length, hidden_size)</code>.
image_hidden_states of the model produced by the vision encoder and after projecting the last hidden state.</p>
</li>
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>transformers.models.aria.modeling_aria.AriaModelOutputWithPast</code> or <code>tuple(torch.FloatTensor)</code></p>
`}}),te=new kt({props:{$$slots:{default:[Xn]},$$scope:{ctx:v}}}),Fe=new A({props:{name:"get_image_features",anchor:"transformers.AriaModel.get_image_features",parameters:[{name:"pixel_values",val:": FloatTensor"},{name:"pixel_mask",val:": typing.Optional[torch.FloatTensor] = None"},{name:"vision_feature_layer",val:": int = -1"}],parametersDescription:[{anchor:"transformers.AriaModel.get_image_features.pixel_values",description:`<strong>pixel_values</strong> (<code>torch.FloatTensor]</code> of shape <code>(batch_size, channels, height, width)</code>) &#x2014;
The tensors corresponding to the input images.`,name:"pixel_values"},{anchor:"transformers.AriaModel.get_image_features.pixel_mask",description:`<strong>pixel_mask</strong> (<code>torch.FloatTensor]</code>, <em>optional</em>) &#x2014;
The tensors corresponding to the input image mask.`,name:"pixel_mask"},{anchor:"transformers.AriaModel.get_image_features.vision_feature_layer",description:`<strong>vision_feature_layer</strong> (<code>Union[int, list[int]]</code>, <em>optional</em>) &#x2014;
The index of the layer to select the vision feature. If multiple indices are provided,
the vision feature of the corresponding indices will be concatenated to form the
vision features.`,name:"vision_feature_layer"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/aria/modeling_aria.py#L943",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>Image feature tensor of shape <code>(num_images, image_length, embed_dim)</code>).</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p>image_features (<code>torch.Tensor</code>)</p>
`}}),Be=new A({props:{name:"get_placeholder_mask",anchor:"transformers.AriaModel.get_placeholder_mask",parameters:[{name:"input_ids",val:": LongTensor"},{name:"inputs_embeds",val:": FloatTensor"},{name:"image_features",val:": FloatTensor"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/aria/modeling_aria.py#L980"}}),Ne=new X({props:{title:"AriaTextForCausalLM",local:"transformers.AriaTextForCausalLM",headingTag:"h2"}}),We=new A({props:{name:"class transformers.AriaTextForCausalLM",anchor:"transformers.AriaTextForCausalLM",parameters:[{name:"config",val:": AriaTextConfig"}],parametersDescription:[{anchor:"transformers.AriaTextForCausalLM.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/aria#transformers.AriaTextConfig">AriaTextConfig</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/aria/modeling_aria.py#L789"}}),Pe=new A({props:{name:"forward",anchor:"transformers.AriaTextForCausalLM.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"},{name:"logits_to_keep",val:": typing.Union[int, torch.Tensor] = 0"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.AriaTextForCausalLM.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.AriaTextForCausalLM.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.AriaTextForCausalLM.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.AriaTextForCausalLM.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.AriaTextForCausalLM.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.AriaTextForCausalLM.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.AriaTextForCausalLM.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.AriaTextForCausalLM.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"},{anchor:"transformers.AriaTextForCausalLM.forward.logits_to_keep",description:`<strong>logits_to_keep</strong> (<code>Union[int, torch.Tensor]</code>, defaults to <code>0</code>) &#x2014;
If an <code>int</code>, compute logits for the last <code>logits_to_keep</code> tokens. If <code>0</code>, calculate logits for all
<code>input_ids</code> (special case). Only last token logits are needed for generation, and calculating them only for that
token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
If a <code>torch.Tensor</code>, must be 1D corresponding to the indices to keep in the sequence length dimension.
This is useful when using packed tensor format (single dimension for batch and sequence length).`,name:"logits_to_keep"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/aria/modeling_aria.py#L803",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast"
>transformers.modeling_outputs.CausalLMOutputWithPast</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/aria#transformers.AriaConfig"
>AriaConfig</a>) and inputs.</p>
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
`}}),ae=new kt({props:{$$slots:{default:[Hn]},$$scope:{ctx:v}}}),se=new Jn({props:{anchor:"transformers.AriaTextForCausalLM.forward.example",$$slots:{default:[Vn]},$$scope:{ctx:v}}}),Ze=new X({props:{title:"AriaForConditionalGeneration",local:"transformers.AriaForConditionalGeneration",headingTag:"h2"}}),Re=new A({props:{name:"class transformers.AriaForConditionalGeneration",anchor:"transformers.AriaForConditionalGeneration",parameters:[{name:"config",val:": AriaConfig"}],parametersDescription:[{anchor:"transformers.AriaForConditionalGeneration.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/aria#transformers.AriaConfig">AriaConfig</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/aria/modeling_aria.py#L1078"}}),Ee=new A({props:{name:"forward",anchor:"transformers.AriaForConditionalGeneration.forward",parameters:[{name:"input_ids",val:": LongTensor = None"},{name:"pixel_values",val:": FloatTensor = None"},{name:"pixel_mask",val:": LongTensor = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"logits_to_keep",val:": typing.Union[int, torch.Tensor] = 0"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.AriaForConditionalGeneration.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.AriaForConditionalGeneration.forward.pixel_values",description:`<strong>pixel_values</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, num_channels, image_size, image_size)</code>) &#x2014;
The tensors corresponding to the input images. Pixel values can be obtained using
<a href="/docs/transformers/v4.56.2/en/model_doc/aria#transformers.AriaImageProcessor">AriaImageProcessor</a>. See <a href="/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__">AriaImageProcessor.<strong>call</strong>()</a> for details (<a href="/docs/transformers/v4.56.2/en/model_doc/aria#transformers.AriaProcessor">AriaProcessor</a> uses
<a href="/docs/transformers/v4.56.2/en/model_doc/aria#transformers.AriaImageProcessor">AriaImageProcessor</a> for processing images).`,name:"pixel_values"},{anchor:"transformers.AriaForConditionalGeneration.forward.pixel_mask",description:`<strong>pixel_mask</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, height, width)</code>) &#x2014;
Mask to avoid performing attention on padding pixel values. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for pixels that are real (i.e. <strong>not masked</strong>),</li>
<li>0 for pixels that are padding (i.e. <strong>masked</strong>).</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"pixel_mask"},{anchor:"transformers.AriaForConditionalGeneration.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.AriaForConditionalGeneration.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.AriaForConditionalGeneration.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.AriaForConditionalGeneration.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.AriaForConditionalGeneration.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or <code>model.image_token_id</code> (where <code>model</code> is your instance of <code>AriaForConditionalGeneration</code>).
Tokens with indices set to <code>model.image_token_id</code> are ignored (masked), the loss is only
computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.AriaForConditionalGeneration.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.AriaForConditionalGeneration.forward.logits_to_keep",description:`<strong>logits_to_keep</strong> (<code>Union[int, torch.Tensor]</code>, defaults to <code>0</code>) &#x2014;
If an <code>int</code>, compute logits for the last <code>logits_to_keep</code> tokens. If <code>0</code>, calculate logits for all
<code>input_ids</code> (special case). Only last token logits are needed for generation, and calculating them only for that
token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
If a <code>torch.Tensor</code>, must be 1D corresponding to the indices to keep in the sequence length dimension.
This is useful when using packed tensor format (single dimension for batch and sequence length).`,name:"logits_to_keep"},{anchor:"transformers.AriaForConditionalGeneration.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/aria/modeling_aria.py#L1133",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <code>transformers.models.aria.modeling_aria.AriaCausalLMOutputWithPast</code> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/aria#transformers.AriaConfig"
>AriaConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>loss</strong> (<code>torch.FloatTensor</code> of shape <code>(1,)</code>, <em>optional</em>, returned when <code>labels</code> is provided) — Language modeling loss (for next-token prediction).</p>
</li>
<li>
<p><strong>logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, config.vocab_size)</code>) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).</p>
</li>
<li>
<p><strong>past_key_values</strong> (<code>Cache</code>, <em>optional</em>, returned when <code>use_cache=True</code> is passed or when <code>config.use_cache=True</code>) — Tuple of <code>tuple(torch.FloatTensor)</code> of length <code>config.n_layers</code>, with each tuple having 2 tensors of shape
<code>(batch_size, num_heads, sequence_length, embed_size_per_head)</code>)</p>
<p>Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
<code>past_key_values</code> input) to speed up sequential decoding.</p>
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
<li>
<p><strong>image_hidden_states</strong> (<code>torch.FloatTensor</code>, <em>optional</em>) — A <code>torch.FloatTensor</code> of size <code>(batch_size, num_images, sequence_length, hidden_size)</code>.
image_hidden_states of the model produced by the vision encoder and after projecting the last hidden state.</p>
</li>
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>transformers.models.aria.modeling_aria.AriaCausalLMOutputWithPast</code> or <code>tuple(torch.FloatTensor)</code></p>
`}}),re=new kt({props:{$$slots:{default:[Sn]},$$scope:{ctx:v}}}),ie=new Jn({props:{anchor:"transformers.AriaForConditionalGeneration.forward.example",$$slots:{default:[Dn]},$$scope:{ctx:v}}}),Le=new Pn({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/aria.md"}}),{c(){t=l("meta"),y=s(),i=l("p"),T=s(),b=l("p"),b.innerHTML=M,U=s(),N=l("div"),N.innerHTML=Do,Ut=s(),m(ce.$$.fragment),It=s(),me=l("p"),me.innerHTML=Yo,Jt=s(),pe=l("p"),pe.innerHTML=Oo,$t=s(),m(V.$$.fragment),zt=s(),he=l("p"),he.innerHTML=Ko,qt=s(),m(S.$$.fragment),Ft=s(),ge=l("p"),ge.innerHTML=en,Bt=s(),ue=l("p"),ue.innerHTML=tn,Nt=s(),m(fe.$$.fragment),Wt=s(),m(_e.$$.fragment),Pt=s(),k=l("div"),m(Me.$$.fragment),so=s(),De=l("p"),De.textContent=on,ro=s(),D=l("div"),m(ye.$$.fragment),io=s(),Ye=l("p"),Ye.textContent=nn,lo=s(),Y=l("div"),m(Te.$$.fragment),co=s(),Oe=l("p"),Oe.textContent=an,mo=s(),O=l("div"),m(be.$$.fragment),po=s(),Ke=l("p"),Ke.innerHTML=sn,ho=s(),K=l("div"),m(ve.$$.fragment),go=s(),et=l("p"),et.textContent=rn,Zt=s(),m(we.$$.fragment),Rt=s(),L=l("div"),m(je.$$.fragment),uo=s(),tt=l("p"),tt.textContent=ln,Et=s(),m(xe.$$.fragment),Lt=s(),Q=l("div"),m(Ae.$$.fragment),fo=s(),ot=l("p"),ot.innerHTML=dn,Qt=s(),m(ke.$$.fragment),Gt=s(),W=l("div"),m(Ce.$$.fragment),_o=s(),nt=l("p"),nt.innerHTML=cn,Mo=s(),at=l("p"),at.innerHTML=mn,Xt=s(),m(Ue.$$.fragment),Ht=s(),I=l("div"),m(Ie.$$.fragment),yo=s(),st=l("p"),st.textContent=pn,To=s(),rt=l("p"),rt.innerHTML=hn,bo=s(),it=l("p"),it.innerHTML=gn,vo=s(),P=l("div"),m(Je.$$.fragment),wo=s(),lt=l("p"),lt.innerHTML=un,jo=s(),m(ee.$$.fragment),Vt=s(),m($e.$$.fragment),St=s(),x=l("div"),m(ze.$$.fragment),xo=s(),dt=l("p"),dt.textContent=fn,Ao=s(),ct=l("p"),ct.innerHTML=_n,ko=s(),mt=l("p"),mt.innerHTML=Mn,Co=s(),Z=l("div"),m(qe.$$.fragment),Uo=s(),pt=l("p"),pt.innerHTML=yn,Io=s(),m(te.$$.fragment),Jo=s(),oe=l("div"),m(Fe.$$.fragment),$o=s(),ht=l("p"),ht.textContent=Tn,zo=s(),ne=l("div"),m(Be.$$.fragment),qo=s(),gt=l("p"),gt.innerHTML=bn,Dt=s(),m(Ne.$$.fragment),Yt=s(),J=l("div"),m(We.$$.fragment),Fo=s(),ut=l("p"),ut.textContent=vn,Bo=s(),ft=l("p"),ft.innerHTML=wn,No=s(),_t=l("p"),_t.innerHTML=jn,Wo=s(),q=l("div"),m(Pe.$$.fragment),Po=s(),Mt=l("p"),Mt.innerHTML=xn,Zo=s(),m(ae.$$.fragment),Ro=s(),m(se.$$.fragment),Ot=s(),m(Ze.$$.fragment),Kt=s(),C=l("div"),m(Re.$$.fragment),Eo=s(),yt=l("p"),yt.textContent=An,Lo=s(),Tt=l("p"),Tt.textContent=kn,Qo=s(),bt=l("p"),bt.innerHTML=Cn,Go=s(),vt=l("p"),vt.innerHTML=Un,Xo=s(),F=l("div"),m(Ee.$$.fragment),Ho=s(),wt=l("p"),wt.innerHTML=In,Vo=s(),m(re.$$.fragment),So=s(),m(ie.$$.fragment),eo=s(),m(Le.$$.fragment),to=s(),At=l("p"),this.h()},l(e){const n=Nn("svelte-u9bgzb",document.head);t=d(n,"META",{name:!0,content:!0}),n.forEach(a),y=r(e),i=d(e,"P",{}),w(i).forEach(a),T=r(e),b=d(e,"P",{"data-svelte-h":!0}),_(b)!=="svelte-114f654"&&(b.innerHTML=M),U=r(e),N=d(e,"DIV",{style:!0,"data-svelte-h":!0}),_(N)!=="svelte-2m0t7r"&&(N.innerHTML=Do),Ut=r(e),p(ce.$$.fragment,e),It=r(e),me=d(e,"P",{"data-svelte-h":!0}),_(me)!=="svelte-agimpv"&&(me.innerHTML=Yo),Jt=r(e),pe=d(e,"P",{"data-svelte-h":!0}),_(pe)!=="svelte-10i9fzm"&&(pe.innerHTML=Oo),$t=r(e),p(V.$$.fragment,e),zt=r(e),he=d(e,"P",{"data-svelte-h":!0}),_(he)!=="svelte-2n7mbe"&&(he.innerHTML=Ko),qt=r(e),p(S.$$.fragment,e),Ft=r(e),ge=d(e,"P",{"data-svelte-h":!0}),_(ge)!=="svelte-nf5ooi"&&(ge.innerHTML=en),Bt=r(e),ue=d(e,"P",{"data-svelte-h":!0}),_(ue)!=="svelte-1nmsv5v"&&(ue.innerHTML=tn),Nt=r(e),p(fe.$$.fragment,e),Wt=r(e),p(_e.$$.fragment,e),Pt=r(e),k=d(e,"DIV",{class:!0});var $=w(k);p(Me.$$.fragment,$),so=r($),De=d($,"P",{"data-svelte-h":!0}),_(De)!=="svelte-xu4frv"&&(De.textContent=on),ro=r($),D=d($,"DIV",{class:!0});var Qe=w(D);p(ye.$$.fragment,Qe),io=r(Qe),Ye=d(Qe,"P",{"data-svelte-h":!0}),_(Ye)!=="svelte-1ycjrv2"&&(Ye.textContent=nn),Qe.forEach(a),lo=r($),Y=d($,"DIV",{class:!0});var Ge=w(Y);p(Te.$$.fragment,Ge),co=r(Ge),Oe=d(Ge,"P",{"data-svelte-h":!0}),_(Oe)!=="svelte-16g0mbp"&&(Oe.textContent=an),Ge.forEach(a),mo=r($),O=d($,"DIV",{class:!0});var Xe=w(O);p(be.$$.fragment,Xe),po=r(Xe),Ke=d(Xe,"P",{"data-svelte-h":!0}),_(Ke)!=="svelte-1wucq2q"&&(Ke.innerHTML=sn),Xe.forEach(a),ho=r($),K=d($,"DIV",{class:!0});var He=w(K);p(ve.$$.fragment,He),go=r(He),et=d(He,"P",{"data-svelte-h":!0}),_(et)!=="svelte-1bzutl7"&&(et.textContent=rn),He.forEach(a),$.forEach(a),Zt=r(e),p(we.$$.fragment,e),Rt=r(e),L=d(e,"DIV",{class:!0});var Ve=w(L);p(je.$$.fragment,Ve),uo=r(Ve),tt=d(Ve,"P",{"data-svelte-h":!0}),_(tt)!=="svelte-1nclzmj"&&(tt.textContent=ln),Ve.forEach(a),Et=r(e),p(xe.$$.fragment,e),Lt=r(e),Q=d(e,"DIV",{class:!0});var Se=w(Q);p(Ae.$$.fragment,Se),fo=r(Se),ot=d(Se,"P",{"data-svelte-h":!0}),_(ot)!=="svelte-1e52xps"&&(ot.innerHTML=dn),Se.forEach(a),Qt=r(e),p(ke.$$.fragment,e),Gt=r(e),W=d(e,"DIV",{class:!0});var G=w(W);p(Ce.$$.fragment,G),_o=r(G),nt=d(G,"P",{"data-svelte-h":!0}),_(nt)!=="svelte-1yio4ls"&&(nt.innerHTML=cn),Mo=r(G),at=d(G,"P",{"data-svelte-h":!0}),_(at)!=="svelte-1ek1ss9"&&(at.innerHTML=mn),G.forEach(a),Xt=r(e),p(Ue.$$.fragment,e),Ht=r(e),I=d(e,"DIV",{class:!0});var R=w(I);p(Ie.$$.fragment,R),yo=r(R),st=d(R,"P",{"data-svelte-h":!0}),_(st)!=="svelte-28faaa"&&(st.textContent=pn),To=r(R),rt=d(R,"P",{"data-svelte-h":!0}),_(rt)!=="svelte-q52n56"&&(rt.innerHTML=hn),bo=r(R),it=d(R,"P",{"data-svelte-h":!0}),_(it)!=="svelte-hswkmf"&&(it.innerHTML=gn),vo=r(R),P=d(R,"DIV",{class:!0});var jt=w(P);p(Je.$$.fragment,jt),wo=r(jt),lt=d(jt,"P",{"data-svelte-h":!0}),_(lt)!=="svelte-1mri2uq"&&(lt.innerHTML=un),jo=r(jt),p(ee.$$.fragment,jt),jt.forEach(a),R.forEach(a),Vt=r(e),p($e.$$.fragment,e),St=r(e),x=d(e,"DIV",{class:!0});var z=w(x);p(ze.$$.fragment,z),xo=r(z),dt=d(z,"P",{"data-svelte-h":!0}),_(dt)!=="svelte-1wj231m"&&(dt.textContent=fn),Ao=r(z),ct=d(z,"P",{"data-svelte-h":!0}),_(ct)!=="svelte-q52n56"&&(ct.innerHTML=_n),ko=r(z),mt=d(z,"P",{"data-svelte-h":!0}),_(mt)!=="svelte-hswkmf"&&(mt.innerHTML=Mn),Co=r(z),Z=d(z,"DIV",{class:!0});var xt=w(Z);p(qe.$$.fragment,xt),Uo=r(xt),pt=d(xt,"P",{"data-svelte-h":!0}),_(pt)!=="svelte-1wckfm2"&&(pt.innerHTML=yn),Io=r(xt),p(te.$$.fragment,xt),xt.forEach(a),Jo=r(z),oe=d(z,"DIV",{class:!0});var no=w(oe);p(Fe.$$.fragment,no),$o=r(no),ht=d(no,"P",{"data-svelte-h":!0}),_(ht)!=="svelte-1vzo9k5"&&(ht.textContent=Tn),no.forEach(a),zo=r(z),ne=d(z,"DIV",{class:!0});var ao=w(ne);p(Be.$$.fragment,ao),qo=r(ao),gt=d(ao,"P",{"data-svelte-h":!0}),_(gt)!=="svelte-3ue1dv"&&(gt.innerHTML=bn),ao.forEach(a),z.forEach(a),Dt=r(e),p(Ne.$$.fragment,e),Yt=r(e),J=d(e,"DIV",{class:!0});var E=w(J);p(We.$$.fragment,E),Fo=r(E),ut=d(E,"P",{"data-svelte-h":!0}),_(ut)!=="svelte-1mpj2tw"&&(ut.textContent=vn),Bo=r(E),ft=d(E,"P",{"data-svelte-h":!0}),_(ft)!=="svelte-q52n56"&&(ft.innerHTML=wn),No=r(E),_t=d(E,"P",{"data-svelte-h":!0}),_(_t)!=="svelte-hswkmf"&&(_t.innerHTML=jn),Wo=r(E),q=d(E,"DIV",{class:!0});var le=w(q);p(Pe.$$.fragment,le),Po=r(le),Mt=d(le,"P",{"data-svelte-h":!0}),_(Mt)!=="svelte-1nlnuba"&&(Mt.innerHTML=xn),Zo=r(le),p(ae.$$.fragment,le),Ro=r(le),p(se.$$.fragment,le),le.forEach(a),E.forEach(a),Ot=r(e),p(Ze.$$.fragment,e),Kt=r(e),C=d(e,"DIV",{class:!0});var B=w(C);p(Re.$$.fragment,B),Eo=r(B),yt=d(B,"P",{"data-svelte-h":!0}),_(yt)!=="svelte-1yqt7kz"&&(yt.textContent=An),Lo=r(B),Tt=d(B,"P",{"data-svelte-h":!0}),_(Tt)!=="svelte-10hcme9"&&(Tt.textContent=kn),Qo=r(B),bt=d(B,"P",{"data-svelte-h":!0}),_(bt)!=="svelte-q52n56"&&(bt.innerHTML=Cn),Go=r(B),vt=d(B,"P",{"data-svelte-h":!0}),_(vt)!=="svelte-hswkmf"&&(vt.innerHTML=Un),Xo=r(B),F=d(B,"DIV",{class:!0});var de=w(F);p(Ee.$$.fragment,de),Ho=r(de),wt=d(de,"P",{"data-svelte-h":!0}),_(wt)!=="svelte-3u09oe"&&(wt.innerHTML=In),Vo=r(de),p(re.$$.fragment,de),So=r(de),p(ie.$$.fragment,de),de.forEach(a),B.forEach(a),eo=r(e),p(Le.$$.fragment,e),to=r(e),At=d(e,"P",{}),w(At).forEach(a),this.h()},h(){j(t,"name","hf:doc:metadata"),j(t,"content",On),Wn(N,"float","right"),j(D,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),j(Y,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),j(O,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),j(K,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),j(k,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),j(L,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),j(Q,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),j(W,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),j(P,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),j(I,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),j(Z,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),j(oe,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),j(ne,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),j(x,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),j(q,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),j(J,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),j(F,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),j(C,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,n){o(document.head,t),c(e,y,n),c(e,i,n),c(e,T,n),c(e,b,n),c(e,U,n),c(e,N,n),c(e,Ut,n),h(ce,e,n),c(e,It,n),c(e,me,n),c(e,Jt,n),c(e,pe,n),c(e,$t,n),h(V,e,n),c(e,zt,n),c(e,he,n),c(e,qt,n),h(S,e,n),c(e,Ft,n),c(e,ge,n),c(e,Bt,n),c(e,ue,n),c(e,Nt,n),h(fe,e,n),c(e,Wt,n),h(_e,e,n),c(e,Pt,n),c(e,k,n),h(Me,k,null),o(k,so),o(k,De),o(k,ro),o(k,D),h(ye,D,null),o(D,io),o(D,Ye),o(k,lo),o(k,Y),h(Te,Y,null),o(Y,co),o(Y,Oe),o(k,mo),o(k,O),h(be,O,null),o(O,po),o(O,Ke),o(k,ho),o(k,K),h(ve,K,null),o(K,go),o(K,et),c(e,Zt,n),h(we,e,n),c(e,Rt,n),c(e,L,n),h(je,L,null),o(L,uo),o(L,tt),c(e,Et,n),h(xe,e,n),c(e,Lt,n),c(e,Q,n),h(Ae,Q,null),o(Q,fo),o(Q,ot),c(e,Qt,n),h(ke,e,n),c(e,Gt,n),c(e,W,n),h(Ce,W,null),o(W,_o),o(W,nt),o(W,Mo),o(W,at),c(e,Xt,n),h(Ue,e,n),c(e,Ht,n),c(e,I,n),h(Ie,I,null),o(I,yo),o(I,st),o(I,To),o(I,rt),o(I,bo),o(I,it),o(I,vo),o(I,P),h(Je,P,null),o(P,wo),o(P,lt),o(P,jo),h(ee,P,null),c(e,Vt,n),h($e,e,n),c(e,St,n),c(e,x,n),h(ze,x,null),o(x,xo),o(x,dt),o(x,Ao),o(x,ct),o(x,ko),o(x,mt),o(x,Co),o(x,Z),h(qe,Z,null),o(Z,Uo),o(Z,pt),o(Z,Io),h(te,Z,null),o(x,Jo),o(x,oe),h(Fe,oe,null),o(oe,$o),o(oe,ht),o(x,zo),o(x,ne),h(Be,ne,null),o(ne,qo),o(ne,gt),c(e,Dt,n),h(Ne,e,n),c(e,Yt,n),c(e,J,n),h(We,J,null),o(J,Fo),o(J,ut),o(J,Bo),o(J,ft),o(J,No),o(J,_t),o(J,Wo),o(J,q),h(Pe,q,null),o(q,Po),o(q,Mt),o(q,Zo),h(ae,q,null),o(q,Ro),h(se,q,null),c(e,Ot,n),h(Ze,e,n),c(e,Kt,n),c(e,C,n),h(Re,C,null),o(C,Eo),o(C,yt),o(C,Lo),o(C,Tt),o(C,Qo),o(C,bt),o(C,Go),o(C,vt),o(C,Xo),o(C,F),h(Ee,F,null),o(F,Ho),o(F,wt),o(F,Vo),h(re,F,null),o(F,So),h(ie,F,null),c(e,eo,n),h(Le,e,n),c(e,to,n),c(e,At,n),oo=!0},p(e,[n]){const $={};n&2&&($.$$scope={dirty:n,ctx:e}),V.$set($);const Qe={};n&2&&(Qe.$$scope={dirty:n,ctx:e}),S.$set(Qe);const Ge={};n&2&&(Ge.$$scope={dirty:n,ctx:e}),ee.$set(Ge);const Xe={};n&2&&(Xe.$$scope={dirty:n,ctx:e}),te.$set(Xe);const He={};n&2&&(He.$$scope={dirty:n,ctx:e}),ae.$set(He);const Ve={};n&2&&(Ve.$$scope={dirty:n,ctx:e}),se.$set(Ve);const Se={};n&2&&(Se.$$scope={dirty:n,ctx:e}),re.$set(Se);const G={};n&2&&(G.$$scope={dirty:n,ctx:e}),ie.$set(G)},i(e){oo||(g(ce.$$.fragment,e),g(V.$$.fragment,e),g(S.$$.fragment,e),g(fe.$$.fragment,e),g(_e.$$.fragment,e),g(Me.$$.fragment,e),g(ye.$$.fragment,e),g(Te.$$.fragment,e),g(be.$$.fragment,e),g(ve.$$.fragment,e),g(we.$$.fragment,e),g(je.$$.fragment,e),g(xe.$$.fragment,e),g(Ae.$$.fragment,e),g(ke.$$.fragment,e),g(Ce.$$.fragment,e),g(Ue.$$.fragment,e),g(Ie.$$.fragment,e),g(Je.$$.fragment,e),g(ee.$$.fragment,e),g($e.$$.fragment,e),g(ze.$$.fragment,e),g(qe.$$.fragment,e),g(te.$$.fragment,e),g(Fe.$$.fragment,e),g(Be.$$.fragment,e),g(Ne.$$.fragment,e),g(We.$$.fragment,e),g(Pe.$$.fragment,e),g(ae.$$.fragment,e),g(se.$$.fragment,e),g(Ze.$$.fragment,e),g(Re.$$.fragment,e),g(Ee.$$.fragment,e),g(re.$$.fragment,e),g(ie.$$.fragment,e),g(Le.$$.fragment,e),oo=!0)},o(e){u(ce.$$.fragment,e),u(V.$$.fragment,e),u(S.$$.fragment,e),u(fe.$$.fragment,e),u(_e.$$.fragment,e),u(Me.$$.fragment,e),u(ye.$$.fragment,e),u(Te.$$.fragment,e),u(be.$$.fragment,e),u(ve.$$.fragment,e),u(we.$$.fragment,e),u(je.$$.fragment,e),u(xe.$$.fragment,e),u(Ae.$$.fragment,e),u(ke.$$.fragment,e),u(Ce.$$.fragment,e),u(Ue.$$.fragment,e),u(Ie.$$.fragment,e),u(Je.$$.fragment,e),u(ee.$$.fragment,e),u($e.$$.fragment,e),u(ze.$$.fragment,e),u(qe.$$.fragment,e),u(te.$$.fragment,e),u(Fe.$$.fragment,e),u(Be.$$.fragment,e),u(Ne.$$.fragment,e),u(We.$$.fragment,e),u(Pe.$$.fragment,e),u(ae.$$.fragment,e),u(se.$$.fragment,e),u(Ze.$$.fragment,e),u(Re.$$.fragment,e),u(Ee.$$.fragment,e),u(re.$$.fragment,e),u(ie.$$.fragment,e),u(Le.$$.fragment,e),oo=!1},d(e){e&&(a(y),a(i),a(T),a(b),a(U),a(N),a(Ut),a(It),a(me),a(Jt),a(pe),a($t),a(zt),a(he),a(qt),a(Ft),a(ge),a(Bt),a(ue),a(Nt),a(Wt),a(Pt),a(k),a(Zt),a(Rt),a(L),a(Et),a(Lt),a(Q),a(Qt),a(Gt),a(W),a(Xt),a(Ht),a(I),a(Vt),a(St),a(x),a(Dt),a(Yt),a(J),a(Ot),a(Kt),a(C),a(eo),a(to),a(At)),a(t),f(ce,e),f(V,e),f(S,e),f(fe,e),f(_e,e),f(Me),f(ye),f(Te),f(be),f(ve),f(we,e),f(je),f(xe,e),f(Ae),f(ke,e),f(Ce),f(Ue,e),f(Ie),f(Je),f(ee),f($e,e),f(ze),f(qe),f(te),f(Fe),f(Be),f(Ne,e),f(We),f(Pe),f(ae),f(se),f(Ze,e),f(Re),f(Ee),f(re),f(ie),f(Le,e)}}}const On='{"title":"Aria","local":"aria","sections":[{"title":"AriaImageProcessor","local":"transformers.AriaImageProcessor","sections":[],"depth":2},{"title":"AriaProcessor","local":"transformers.AriaProcessor","sections":[],"depth":2},{"title":"AriaTextConfig","local":"transformers.AriaTextConfig","sections":[],"depth":2},{"title":"AriaConfig","local":"transformers.AriaConfig","sections":[],"depth":2},{"title":"AriaTextModel","local":"transformers.AriaTextModel","sections":[],"depth":2},{"title":"AriaModel","local":"transformers.AriaModel","sections":[],"depth":2},{"title":"AriaTextForCausalLM","local":"transformers.AriaTextForCausalLM","sections":[],"depth":2},{"title":"AriaForConditionalGeneration","local":"transformers.AriaForConditionalGeneration","sections":[],"depth":2}],"depth":1}';function Kn(v){return qn(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class la extends Fn{constructor(t){super(),Bn(this,t,Kn,Yn,zn,{})}}export{la as component};
