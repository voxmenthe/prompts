import{s as wt,o as vt,n as q}from"../chunks/scheduler.18a86fab.js";import{S as kt,i as Ut,g as p,s as i,r as u,A as $t,h,f as s,c as l,j as re,x as T,u as f,k as ie,l as jt,y as m,a as r,v as g,d as _,t as M,w as y}from"../chunks/index.98837b22.js";import{T as nt}from"../chunks/Tip.77304350.js";import{D as Te}from"../chunks/Docstring.a1ef7999.js";import{C as Me}from"../chunks/CodeBlock.8d0c2e8a.js";import{E as Ct}from"../chunks/ExampleCodeBlock.8c3ee1f9.js";import{H as Ge,E as Jt}from"../chunks/getInferenceSnippets.06c2775f.js";import{H as xt,a as st}from"../chunks/HfOption.6641485e.js";function It(v){let t,a="Click on the Cohere models in the right sidebar for more examples of how to apply Cohere to different language tasks.";return{c(){t=p("p"),t.textContent=a},l(o){t=h(o,"P",{"data-svelte-h":!0}),T(t)!=="svelte-1p3ael3"&&(t.textContent=a)},m(o,c){r(o,t,c)},p:q,d(o){o&&s(t)}}}function zt(v){let t,a;return t=new Me({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwcGlwZWxpbmUlMEElMEFwaXBlbGluZSUyMCUzRCUyMHBpcGVsaW5lKCUwQSUyMCUyMCUyMCUyMHRhc2slM0QlMjJ0ZXh0LWdlbmVyYXRpb24lMjIlMkMlMjAlMEElMjAlMjAlMjAlMjBtb2RlbCUzRCUyMkNvaGVyZUxhYnMlMkZjNGFpLWNvbW1hbmQtcjdiLTEyLTIwMjQlMjIlMkMlMEElMjAlMjAlMjAlMjBkdHlwZSUzRHRvcmNoLmZsb2F0MTYlMkMlMEElMjAlMjAlMjAlMjBkZXZpY2VfbWFwJTNEMCUwQSklMEElMEFtZXNzYWdlcyUyMCUzRCUyMCU1QiUwQSUyMCUyMCUyMCUyMCU3QiUyMnJvbGUlMjIlM0ElMjAlMjJ1c2VyJTIyJTJDJTIwJTIyY29udGVudCUyMiUzQSUyMCUyMkhlbGxvJTJDJTIwY2FuJTIweW91JTIwcGxlYXNlJTIwaGVscCUyMG1lJTIwYm9vayUyMGElMjBob3RlbCUyMGluJTIwSmFwYW4lM0YlMjIlN0QlMkMlMEElNUQlMEFwaXBlbGluZShtZXNzYWdlcyk=",highlighted:`<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> pipeline

pipeline = pipeline(
    task=<span class="hljs-string">&quot;text-generation&quot;</span>, 
    model=<span class="hljs-string">&quot;CohereLabs/c4ai-command-r7b-12-2024&quot;</span>,
    dtype=torch.float16,
    device_map=<span class="hljs-number">0</span>
)

messages = [
    {<span class="hljs-string">&quot;role&quot;</span>: <span class="hljs-string">&quot;user&quot;</span>, <span class="hljs-string">&quot;content&quot;</span>: <span class="hljs-string">&quot;Hello, can you please help me book a hotel in Japan?&quot;</span>},
]
pipeline(messages)`,wrap:!1}}),{c(){u(t.$$.fragment)},l(o){f(t.$$.fragment,o)},m(o,c){g(t,o,c),a=!0},p:q,i(o){a||(_(t.$$.fragment,o),a=!0)},o(o){M(t.$$.fragment,o),a=!1},d(o){y(t,o)}}}function Rt(v){let t,a;return t=new Me({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Rva2VuaXplciUyQyUyMEF1dG9Nb2RlbEZvckNhdXNhbExNJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIyQ29oZXJlTGFicyUyRmM0YWktY29tbWFuZC1yN2ItMTItMjAyNCUyMiklMEFtb2RlbCUyMCUzRCUyMEF1dG9Nb2RlbEZvckNhdXNhbExNLmZyb21fcHJldHJhaW5lZCglMEElMjAlMjAlMjAlMjAlMjJDb2hlcmVMYWJzJTJGYzRhaS1jb21tYW5kLXI3Yi0xMi0yMDI0JTIyJTJDJTIwJTBBJTIwJTIwJTIwJTIwZHR5cGUlM0R0b3JjaC5mbG9hdDE2JTJDJTIwJTBBJTIwJTIwJTIwJTIwZGV2aWNlX21hcCUzRCUyMmF1dG8lMjIlMkMlMjAlMEElMjAlMjAlMjAlMjBhdHRuX2ltcGxlbWVudGF0aW9uJTNEJTIyc2RwYSUyMiUwQSklMEElMEElMjMlMjBmb3JtYXQlMjBtZXNzYWdlJTIwd2l0aCUyMHRoZSUyMENvbW1hbmQtUiUyMGNoYXQlMjB0ZW1wbGF0ZSUwQW1lc3NhZ2VzJTIwJTNEJTIwJTVCJTdCJTIycm9sZSUyMiUzQSUyMCUyMnVzZXIlMjIlMkMlMjAlMjJjb250ZW50JTIyJTNBJTIwJTIySGVsbG8lMkMlMjBjYW4lMjB5b3UlMjBwbGVhc2UlMjBoZWxwJTIwbWUlMjBib29rJTIwYSUyMGhvdGVsJTIwaW4lMjBKYXBhbiUzRiUyMiU3RCU1RCUwQWlucHV0X2lkcyUyMCUzRCUyMHRva2VuaXplci5hcHBseV9jaGF0X3RlbXBsYXRlKG1lc3NhZ2VzJTJDJTIwdG9rZW5pemUlM0RUcnVlJTJDJTIwYWRkX2dlbmVyYXRpb25fcHJvbXB0JTNEVHJ1ZSUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpLnRvKG1vZGVsLmRldmljZSklMEFvdXRwdXQlMjAlM0QlMjBtb2RlbC5nZW5lcmF0ZSglMEElMjAlMjAlMjAlMjBpbnB1dF9pZHMlMkMlMEElMjAlMjAlMjAlMjBtYXhfbmV3X3Rva2VucyUzRDEwMCUyQyUwQSUyMCUyMCUyMCUyMGRvX3NhbXBsZSUzRFRydWUlMkMlMEElMjAlMjAlMjAlMjB0ZW1wZXJhdHVyZSUzRDAuMyUyQyUwQSUyMCUyMCUyMCUyMGNhY2hlX2ltcGxlbWVudGF0aW9uJTNEJTIyc3RhdGljJTIyJTJDJTBBKSUwQXByaW50KHRva2VuaXplci5kZWNvZGUob3V0cHV0JTVCMCU1RCUyQyUyMHNraXBfc3BlY2lhbF90b2tlbnMlM0RUcnVlKSk=",highlighted:`<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;CohereLabs/c4ai-command-r7b-12-2024&quot;</span>)
model = AutoModelForCausalLM.from_pretrained(
    <span class="hljs-string">&quot;CohereLabs/c4ai-command-r7b-12-2024&quot;</span>, 
    dtype=torch.float16, 
    device_map=<span class="hljs-string">&quot;auto&quot;</span>, 
    attn_implementation=<span class="hljs-string">&quot;sdpa&quot;</span>
)

<span class="hljs-comment"># format message with the Command-R chat template</span>
messages = [{<span class="hljs-string">&quot;role&quot;</span>: <span class="hljs-string">&quot;user&quot;</span>, <span class="hljs-string">&quot;content&quot;</span>: <span class="hljs-string">&quot;Hello, can you please help me book a hotel in Japan?&quot;</span>}]
input_ids = tokenizer.apply_chat_template(messages, tokenize=<span class="hljs-literal">True</span>, add_generation_prompt=<span class="hljs-literal">True</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>).to(model.device)
output = model.generate(
    input_ids,
    max_new_tokens=<span class="hljs-number">100</span>,
    do_sample=<span class="hljs-literal">True</span>,
    temperature=<span class="hljs-number">0.3</span>,
    cache_implementation=<span class="hljs-string">&quot;static&quot;</span>,
)
<span class="hljs-built_in">print</span>(tokenizer.decode(output[<span class="hljs-number">0</span>], skip_special_tokens=<span class="hljs-literal">True</span>))`,wrap:!1}}),{c(){u(t.$$.fragment)},l(o){f(t.$$.fragment,o)},m(o,c){g(t,o,c),a=!0},p:q,i(o){a||(_(t.$$.fragment,o),a=!0)},o(o){M(t.$$.fragment,o),a=!1},d(o){y(t,o)}}}function Zt(v){let t,a;return t=new Me({props:{code:"JTIzJTIwcGlwJTIwaW5zdGFsbCUyMC1VJTIwZmxhc2gtYXR0biUyMC0tbm8tYnVpbGQtaXNvbGF0aW9uJTBBdHJhbnNmb3JtZXJzLWNsaSUyMGNoYXQlMjBDb2hlcmVMYWJzJTJGYzRhaS1jb21tYW5kLXI3Yi0xMi0yMDI0JTIwLS1kdHlwZSUyMGF1dG8lMjAtLWF0dG5faW1wbGVtZW50YXRpb24lMjBmbGFzaF9hdHRlbnRpb25fMg==",highlighted:`<span class="hljs-comment"># pip install -U flash-attn --no-build-isolation</span>
transformers-cli chat CohereLabs/c4ai-command-r7b-12-2024 --dtype auto --attn_implementation flash_attention_2`,wrap:!1}}),{c(){u(t.$$.fragment)},l(o){f(t.$$.fragment,o)},m(o,c){g(t,o,c),a=!0},p:q,i(o){a||(_(t.$$.fragment,o),a=!0)},o(o){M(t.$$.fragment,o),a=!1},d(o){y(t,o)}}}function Wt(v){let t,a,o,c,C,b;return t=new st({props:{id:"usage",option:"Pipeline",$$slots:{default:[zt]},$$scope:{ctx:v}}}),o=new st({props:{id:"usage",option:"AutoModel",$$slots:{default:[Rt]},$$scope:{ctx:v}}}),C=new st({props:{id:"usage",option:"transformers CLI",$$slots:{default:[Zt]},$$scope:{ctx:v}}}),{c(){u(t.$$.fragment),a=i(),u(o.$$.fragment),c=i(),u(C.$$.fragment)},l(d){f(t.$$.fragment,d),a=l(d),f(o.$$.fragment,d),c=l(d),f(C.$$.fragment,d)},m(d,w){g(t,d,w),r(d,a,w),g(o,d,w),r(d,c,w),g(C,d,w),b=!0},p(d,w){const ye={};w&2&&(ye.$$scope={dirty:w,ctx:d}),t.$set(ye);const N={};w&2&&(N.$$scope={dirty:w,ctx:d}),o.$set(N);const R={};w&2&&(R.$$scope={dirty:w,ctx:d}),C.$set(R)},i(d){b||(_(t.$$.fragment,d),_(o.$$.fragment,d),_(C.$$.fragment,d),b=!0)},o(d){M(t.$$.fragment,d),M(o.$$.fragment,d),M(C.$$.fragment,d),b=!1},d(d){d&&(s(a),s(c)),y(t,d),y(o,d),y(C,d)}}}function Bt(v){let t,a;return t=new Me({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMENvaGVyZTJNb2RlbCUyQyUyMENvaGVyZTJDb25maWclMEElMEElMjMlMjBJbml0aWFsaXppbmclMjBhJTIwQ29oZXJlJTIwTmV4dG1vZGVsJTIwY29uZmlndXJhdGlvbiUwQWNvbmZpZ3VyYXRpb24lMjAlM0QlMjBDb2hlcmUyQ29uZmlnKCklMEElMEElMjMlMjBJbml0aWFsaXppbmclMjBhJTIwbW9kZWwlMjBmcm9tJTIwdGhlJTIwQ29oZXJlMiUyMGNvbmZpZ3VyYXRpb24lMEFtb2RlbCUyMCUzRCUyMENvaGVyZTJNb2RlbChjb25maWd1cmF0aW9uKSUwQSUyMyUyMEFjY2Vzc2luZyUyMHRoZSUyMG1vZGVsJTIwY29uZmlndXJhdGlvbiUwQWNvbmZpZ3VyYXRpb24lMjAlM0QlMjBtb2RlbC5jb25maWc=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> Cohere2Model, Cohere2Config

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a Cohere Nextmodel configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = Cohere2Config()

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a model from the Cohere2 configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = Cohere2Model(configuration)
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Accessing the model configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = model.config`,wrap:!1}}),{c(){u(t.$$.fragment)},l(o){f(t.$$.fragment,o)},m(o,c){g(t,o,c),a=!0},p:q,i(o){a||(_(t.$$.fragment,o),a=!0)},o(o){M(t.$$.fragment,o),a=!1},d(o){y(t,o)}}}function Et(v){let t,a=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=p("p"),t.innerHTML=a},l(o){t=h(o,"P",{"data-svelte-h":!0}),T(t)!=="svelte-fincs2"&&(t.innerHTML=a)},m(o,c){r(o,t,c)},p:q,d(o){o&&s(t)}}}function Ft(v){let t,a=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=p("p"),t.innerHTML=a},l(o){t=h(o,"P",{"data-svelte-h":!0}),T(t)!=="svelte-fincs2"&&(t.innerHTML=a)},m(o,c){r(o,t,c)},p:q,d(o){o&&s(t)}}}function Xt(v){let t,a="Example:",o,c,C;return c=new Me({props:{code:"JTNFJTNFJTIwZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBDb2hlcmUyRm9yQ2F1c2FsTE0lMEElMEElM0UlM0UlMjBtb2RlbCUyMCUzRCUyMENvaGVyZTJGb3JDYXVzYWxMTS5mcm9tX3ByZXRyYWluZWQoJTIyQ29oZXJlMkZvckFJJTJGYzRhaS1jb21tYW5kLXItdjAxJTIyKSUwQSUzRSUzRSUyMHRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMkNvaGVyZTJGb3JBSSUyRmM0YWktY29tbWFuZC1yLXYwMSUyMiklMEElMEElM0UlM0UlMjBwcm9tcHQlMjAlM0QlMjAlMjJIZXklMkMlMjBhcmUlMjB5b3UlMjBjb25zY2lvdXMlM0YlMjBDYW4lMjB5b3UlMjB0YWxrJTIwdG8lMjBtZSUzRiUyMiUwQSUzRSUzRSUyMGlucHV0cyUyMCUzRCUyMHRva2VuaXplcihwcm9tcHQlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQSUwQSUzRSUzRSUyMCUyMyUyMEdlbmVyYXRlJTBBJTNFJTNFJTIwZ2VuZXJhdGVfaWRzJTIwJTNEJTIwbW9kZWwuZ2VuZXJhdGUoaW5wdXRzLmlucHV0X2lkcyUyQyUyMG1heF9sZW5ndGglM0QzMCklMEElM0UlM0UlMjB0b2tlbml6ZXIuYmF0Y2hfZGVjb2RlKGdlbmVyYXRlX2lkcyUyQyUyMHNraXBfc3BlY2lhbF90b2tlbnMlM0RUcnVlJTJDJTIwY2xlYW5fdXBfdG9rZW5pemF0aW9uX3NwYWNlcyUzREZhbHNlKSU1QjAlNUQlMEElMjJIZXklMkMlMjBhcmUlMjB5b3UlMjBjb25zY2lvdXMlM0YlMjBDYW4lMjB5b3UlMjB0YWxrJTIwdG8lMjBtZSUzRiU1Q25JJ20lMjBub3QlMjBjb25zY2lvdXMlMkMlMjBidXQlMjBJJTIwY2FuJTIwdGFsayUyMHRvJTIweW91LiUyMg==",highlighted:`&gt;&gt; <span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, Cohere2ForCausalLM

&gt;&gt; model = Cohere2ForCausalLM.from_pretrained(<span class="hljs-string">&quot;Cohere2ForAI/c4ai-command-r-v01&quot;</span>)
&gt;&gt; tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;Cohere2ForAI/c4ai-command-r-v01&quot;</span>)

&gt;&gt; prompt = <span class="hljs-string">&quot;Hey, are you conscious? Can you talk to me?&quot;</span>
&gt;&gt; inputs = tokenizer(prompt, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

&gt;&gt; <span class="hljs-comment"># Generate</span>
&gt;&gt; generate_ids = model.generate(inputs.input_ids, max_length=<span class="hljs-number">30</span>)
&gt;&gt; tokenizer.batch_decode(generate_ids, skip_special_tokens=<span class="hljs-literal">True</span>, clean_up_tokenization_spaces=<span class="hljs-literal">False</span>)[<span class="hljs-number">0</span>]
<span class="hljs-string">&quot;Hey, are you conscious? Can you talk to me?\\nI&#x27;m not conscious, but I can talk to you.&quot;</span>`,wrap:!1}}),{c(){t=p("p"),t.textContent=a,o=i(),u(c.$$.fragment)},l(b){t=h(b,"P",{"data-svelte-h":!0}),T(t)!=="svelte-11lpom8"&&(t.textContent=a),o=l(b),f(c.$$.fragment,b)},m(b,d){r(b,t,d),r(b,o,d),g(c,b,d),C=!0},p:q,i(b){C||(_(c.$$.fragment,b),C=!0)},o(b){M(c.$$.fragment,b),C=!1},d(b){b&&(s(t),s(o)),y(c,b)}}}function Gt(v){let t,a,o,c,C,b="<em>This model was released on 2024-12-13 and added to Hugging Face Transformers on 2024-12-13.</em>",d,w,ye='<div class="flex flex-wrap space-x-1"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/> <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat"/> <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/> <img alt="Tensor parallelism" src="https://img.shields.io/badge/Tensor%20parallelism-06b6d4?style=flat&amp;logoColor=white"/></div>',N,R,Ce,V,at='<a href="https://cohere.com/blog/command-r7b" rel="nofollow">Cohere Command R7B</a> is an open weights research release of a 7B billion parameter model. It is a multilingual model trained on 23 languages and has a context window of 128k. The model features three layers with sliding window attention and ROPE for efficient local context modeling and relative positional encoding. A fourth layer uses global attention without positional embeddings, enabling unrestricted token interactions across the entire sequence.',we,A,rt="This model is optimized for speed, cost-performance, and compute resources.",ve,H,it='You can find all the original Command-R checkpoints under the <a href="https://huggingface.co/collections/CohereForAI/command-models-67652b401665205e17b192ad" rel="nofollow">Command Models</a> collection.',ke,B,Ue,Q,lt='The example below demonstrates how to generate text with <a href="/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.Pipeline">Pipeline</a> or the <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel">AutoModel</a> class, and from the command line.',$e,E,je,Y,dt='Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the <a href="../quantization/overview.md">Quantization</a> overview for more available quantization backends.',Je,S,ct='The example below uses <a href="../quantization/bitsandbytes.md">bitsandbytes</a> to quantize the weights to 4-bits.',xe,P,Ie,D,ze,$,O,Le,le,pt=`This is the configuration class to store the configuration of a <a href="/docs/transformers/v4.56.2/en/model_doc/cohere#transformers.CohereModel">CohereModel</a>. It is used to instantiate an Cohere
model according to the specified arguments, defining the model architecture.`,qe,de,ht=`Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> and can be used to control the model outputs. Read the
documentation from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> for more information. Instantiating a configuration
with the defaults will yield a similar configuration to that of the <a href="https://huggingface.co/CohereForAI/c4ai-command-r-v01" rel="nofollow">CohereForAI/c4ai-command-r-v01</a> model.`,Ne,F,Re,K,Ze,k,ee,Ve,ce,mt="The bare Cohere2 Model outputting raw hidden-states without any specific head on top.",Ae,pe,ut=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,He,he,ft=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Qe,Z,te,Ye,me,gt='The <a href="/docs/transformers/v4.56.2/en/model_doc/cohere2#transformers.Cohere2Model">Cohere2Model</a> forward method, overrides the <code>__call__</code> special method.',Se,X,We,oe,Be,U,ne,Pe,ue,_t="The Cohere2 Model for causal language modeling.",De,fe,Mt=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Oe,ge,yt=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Ke,x,se,et,_e,bt='The <a href="/docs/transformers/v4.56.2/en/model_doc/cohere2#transformers.Cohere2ForCausalLM">Cohere2ForCausalLM</a> forward method, overrides the <code>__call__</code> special method.',tt,G,ot,L,Ee,ae,Fe,be,Xe;return R=new Ge({props:{title:"Cohere 2",local:"cohere-2",headingTag:"h1"}}),B=new nt({props:{warning:!1,$$slots:{default:[It]},$$scope:{ctx:v}}}),E=new xt({props:{id:"usage",options:["Pipeline","AutoModel","transformers CLI"],$$slots:{default:[Wt]},$$scope:{ctx:v}}}),P=new Me({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQml0c0FuZEJ5dGVzQ29uZmlnJTJDJTIwQXV0b1Rva2VuaXplciUyQyUyMEF1dG9Nb2RlbEZvckNhdXNhbExNJTBBJTBBYm5iX2NvbmZpZyUyMCUzRCUyMEJpdHNBbmRCeXRlc0NvbmZpZyhsb2FkX2luXzRiaXQlM0RUcnVlKSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMkNvaGVyZUxhYnMlMkZjNGFpLWNvbW1hbmQtcjdiLTEyLTIwMjQlMjIpJTBBbW9kZWwlMjAlM0QlMjBBdXRvTW9kZWxGb3JDYXVzYWxMTS5mcm9tX3ByZXRyYWluZWQoJTBBJTIwJTIwJTIwJTIwJTIyQ29oZXJlTGFicyUyRmM0YWktY29tbWFuZC1yN2ItMTItMjAyNCUyMiUyQyUyMCUwQSUyMCUyMCUyMCUyMGR0eXBlJTNEdG9yY2guZmxvYXQxNiUyQyUyMCUwQSUyMCUyMCUyMCUyMGRldmljZV9tYXAlM0QlMjJhdXRvJTIyJTJDJTIwJTBBJTIwJTIwJTIwJTIwcXVhbnRpemF0aW9uX2NvbmZpZyUzRGJuYl9jb25maWclMkMlMjAlMEElMjAlMjAlMjAlMjBhdHRuX2ltcGxlbWVudGF0aW9uJTNEJTIyc2RwYSUyMiUwQSklMEElMEElMjMlMjBmb3JtYXQlMjBtZXNzYWdlJTIwd2l0aCUyMHRoZSUyMENvbW1hbmQtUiUyMGNoYXQlMjB0ZW1wbGF0ZSUwQW1lc3NhZ2VzJTIwJTNEJTIwJTVCJTdCJTIycm9sZSUyMiUzQSUyMCUyMnVzZXIlMjIlMkMlMjAlMjJjb250ZW50JTIyJTNBJTIwJTIySGVsbG8lMkMlMjBjYW4lMjB5b3UlMjBwbGVhc2UlMjBoZWxwJTIwbWUlMjBib29rJTIwYSUyMGhvdGVsJTIwaW4lMjBKYXBhbiUzRiUyMiU3RCU1RCUwQWlucHV0X2lkcyUyMCUzRCUyMHRva2VuaXplci5hcHBseV9jaGF0X3RlbXBsYXRlKG1lc3NhZ2VzJTJDJTIwdG9rZW5pemUlM0RUcnVlJTJDJTIwYWRkX2dlbmVyYXRpb25fcHJvbXB0JTNEVHJ1ZSUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpLnRvKG1vZGVsLmRldmljZSklMEFvdXRwdXQlMjAlM0QlMjBtb2RlbC5nZW5lcmF0ZSglMEElMjAlMjAlMjAlMjBpbnB1dF9pZHMlMkMlMEElMjAlMjAlMjAlMjBtYXhfbmV3X3Rva2VucyUzRDEwMCUyQyUwQSUyMCUyMCUyMCUyMGRvX3NhbXBsZSUzRFRydWUlMkMlMEElMjAlMjAlMjAlMjB0ZW1wZXJhdHVyZSUzRDAuMyUyQyUwQSUyMCUyMCUyMCUyMGNhY2hlX2ltcGxlbWVudGF0aW9uJTNEJTIyc3RhdGljJTIyJTJDJTBBKSUwQXByaW50KHRva2VuaXplci5kZWNvZGUob3V0cHV0JTVCMCU1RCUyQyUyMHNraXBfc3BlY2lhbF90b2tlbnMlM0RUcnVlKSk=",highlighted:`<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM

bnb_config = BitsAndBytesConfig(load_in_4bit=<span class="hljs-literal">True</span>)
tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;CohereLabs/c4ai-command-r7b-12-2024&quot;</span>)
model = AutoModelForCausalLM.from_pretrained(
    <span class="hljs-string">&quot;CohereLabs/c4ai-command-r7b-12-2024&quot;</span>, 
    dtype=torch.float16, 
    device_map=<span class="hljs-string">&quot;auto&quot;</span>, 
    quantization_config=bnb_config, 
    attn_implementation=<span class="hljs-string">&quot;sdpa&quot;</span>
)

<span class="hljs-comment"># format message with the Command-R chat template</span>
messages = [{<span class="hljs-string">&quot;role&quot;</span>: <span class="hljs-string">&quot;user&quot;</span>, <span class="hljs-string">&quot;content&quot;</span>: <span class="hljs-string">&quot;Hello, can you please help me book a hotel in Japan?&quot;</span>}]
input_ids = tokenizer.apply_chat_template(messages, tokenize=<span class="hljs-literal">True</span>, add_generation_prompt=<span class="hljs-literal">True</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>).to(model.device)
output = model.generate(
    input_ids,
    max_new_tokens=<span class="hljs-number">100</span>,
    do_sample=<span class="hljs-literal">True</span>,
    temperature=<span class="hljs-number">0.3</span>,
    cache_implementation=<span class="hljs-string">&quot;static&quot;</span>,
)
<span class="hljs-built_in">print</span>(tokenizer.decode(output[<span class="hljs-number">0</span>], skip_special_tokens=<span class="hljs-literal">True</span>))`,wrap:!1}}),D=new Ge({props:{title:"Cohere2Config",local:"transformers.Cohere2Config",headingTag:"h2"}}),O=new Te({props:{name:"class transformers.Cohere2Config",anchor:"transformers.Cohere2Config",parameters:[{name:"vocab_size",val:" = 256000"},{name:"hidden_size",val:" = 8192"},{name:"intermediate_size",val:" = 22528"},{name:"logit_scale",val:" = 0.0625"},{name:"num_hidden_layers",val:" = 40"},{name:"num_attention_heads",val:" = 64"},{name:"num_key_value_heads",val:" = None"},{name:"hidden_act",val:" = 'silu'"},{name:"max_position_embeddings",val:" = 8192"},{name:"initializer_range",val:" = 0.02"},{name:"layer_norm_eps",val:" = 1e-05"},{name:"use_cache",val:" = True"},{name:"pad_token_id",val:" = 0"},{name:"bos_token_id",val:" = 5"},{name:"eos_token_id",val:" = 255001"},{name:"tie_word_embeddings",val:" = True"},{name:"rope_theta",val:" = 10000.0"},{name:"rope_scaling",val:" = None"},{name:"attention_bias",val:" = False"},{name:"attention_dropout",val:" = 0.0"},{name:"sliding_window",val:" = 4096"},{name:"layer_types",val:" = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.Cohere2Config.vocab_size",description:`<strong>vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 256000) &#x2014;
Vocabulary size of the Cohere model. Defines the number of different tokens that can be represented by the
<code>inputs_ids</code> passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/cohere#transformers.CohereModel">CohereModel</a>`,name:"vocab_size"},{anchor:"transformers.Cohere2Config.hidden_size",description:`<strong>hidden_size</strong> (<code>int</code>, <em>optional</em>, defaults to 8192) &#x2014;
Dimension of the hidden representations.`,name:"hidden_size"},{anchor:"transformers.Cohere2Config.intermediate_size",description:`<strong>intermediate_size</strong> (<code>int</code>, <em>optional</em>, defaults to 22528) &#x2014;
Dimension of the MLP representations.`,name:"intermediate_size"},{anchor:"transformers.Cohere2Config.logit_scale",description:`<strong>logit_scale</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0625) &#x2014;
The scaling factor for the output logits.`,name:"logit_scale"},{anchor:"transformers.Cohere2Config.num_hidden_layers",description:`<strong>num_hidden_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 40) &#x2014;
Number of hidden layers in the Transformer decoder.`,name:"num_hidden_layers"},{anchor:"transformers.Cohere2Config.num_attention_heads",description:`<strong>num_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 64) &#x2014;
Number of attention heads for each attention layer in the Transformer decoder.`,name:"num_attention_heads"},{anchor:"transformers.Cohere2Config.num_key_value_heads",description:`<strong>num_key_value_heads</strong> (<code>int</code>, <em>optional</em>) &#x2014;
This is the number of key_value heads that should be used to implement Grouped Query Attention. If
<code>num_key_value_heads=num_attention_heads</code>, the model will use Multi Head Attention (MHA), if
<code>num_key_value_heads=1</code> the model will use Multi Query Attention (MQA) otherwise GQA is used. When
converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
by meanpooling all the original heads within that group. For more details, check out <a href="https://huggingface.co/papers/2305.13245" rel="nofollow">this
paper</a>. If it is not specified, will default to
<code>num_attention_heads</code>.`,name:"num_key_value_heads"},{anchor:"transformers.Cohere2Config.hidden_act",description:`<strong>hidden_act</strong> (<code>str</code> or <code>function</code>, <em>optional</em>, defaults to <code>&quot;silu&quot;</code>) &#x2014;
The non-linear activation function (function or string) in the decoder.`,name:"hidden_act"},{anchor:"transformers.Cohere2Config.max_position_embeddings",description:`<strong>max_position_embeddings</strong> (<code>int</code>, <em>optional</em>, defaults to 8192) &#x2014;
The maximum sequence length that this model might ever be used with.`,name:"max_position_embeddings"},{anchor:"transformers.Cohere2Config.initializer_range",description:`<strong>initializer_range</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.`,name:"initializer_range"},{anchor:"transformers.Cohere2Config.layer_norm_eps",description:`<strong>layer_norm_eps</strong> (<code>float</code>, <em>optional</em>, defaults to 1e-05) &#x2014;
The epsilon used by the layer normalization.`,name:"layer_norm_eps"},{anchor:"transformers.Cohere2Config.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not the model should return the last key/values attentions (not used by all models). Only
relevant if <code>config.is_decoder=True</code>.`,name:"use_cache"},{anchor:"transformers.Cohere2Config.pad_token_id",description:`<strong>pad_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 0) &#x2014;
Padding token id.`,name:"pad_token_id"},{anchor:"transformers.Cohere2Config.bos_token_id",description:`<strong>bos_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 5) &#x2014;
Beginning of stream token id.`,name:"bos_token_id"},{anchor:"transformers.Cohere2Config.eos_token_id",description:`<strong>eos_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 255001) &#x2014;
End of stream token id.`,name:"eos_token_id"},{anchor:"transformers.Cohere2Config.tie_word_embeddings",description:`<strong>tie_word_embeddings</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether to tie weight embeddings`,name:"tie_word_embeddings"},{anchor:"transformers.Cohere2Config.rope_theta",description:`<strong>rope_theta</strong> (<code>float</code>, <em>optional</em>, defaults to 10000.0) &#x2014;
The base period of the RoPE embeddings.`,name:"rope_theta"},{anchor:"transformers.Cohere2Config.rope_scaling",description:`<strong>rope_scaling</strong> (<code>Dict</code>, <em>optional</em>) &#x2014;
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
Only used with &#x2018;llama3&#x2019;. Scaling factor applied to high frequency components of the RoPE`,name:"rope_scaling"},{anchor:"transformers.Cohere2Config.attention_bias",description:`<strong>attention_bias</strong> (<code>bool</code>, defaults to <code>False</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether to use a bias in the query, key, value and output projection layers during self-attention.`,name:"attention_bias"},{anchor:"transformers.Cohere2Config.attention_dropout",description:`<strong>attention_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The dropout ratio for the attention probabilities.`,name:"attention_dropout"},{anchor:"transformers.Cohere2Config.sliding_window",description:`<strong>sliding_window</strong> (<code>int</code>, <em>optional</em>, defaults to 4096) &#x2014;
Size of the sliding window attention context.`,name:"sliding_window"},{anchor:"transformers.Cohere2Config.layer_types",description:`<strong>layer_types</strong> (<code>list</code>, <em>optional</em>) &#x2014;
Attention pattern for each layer.`,name:"layer_types"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/cohere2/configuration_cohere2.py#L28"}}),F=new Ct({props:{anchor:"transformers.Cohere2Config.example",$$slots:{default:[Bt]},$$scope:{ctx:v}}}),K=new Ge({props:{title:"Cohere2Model",local:"transformers.Cohere2Model",headingTag:"h2"}}),ee=new Te({props:{name:"class transformers.Cohere2Model",anchor:"transformers.Cohere2Model",parameters:[{name:"config",val:": Cohere2Config"}],parametersDescription:[{anchor:"transformers.Cohere2Model.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/cohere2#transformers.Cohere2Config">Cohere2Config</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/cohere2/modeling_cohere2.py#L337"}}),te=new Te({props:{name:"forward",anchor:"transformers.Cohere2Model.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.Cohere2Model.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.Cohere2Model.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.Cohere2Model.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.Cohere2Model.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.Cohere2Model.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.Cohere2Model.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.Cohere2Model.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/cohere2/modeling_cohere2.py#L354",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast"
>transformers.modeling_outputs.BaseModelOutputWithPast</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/cohere2#transformers.Cohere2Config"
>Cohere2Config</a>) and inputs.</p>
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
`}}),X=new nt({props:{$$slots:{default:[Et]},$$scope:{ctx:v}}}),oe=new Ge({props:{title:"Cohere2ForCausalLM",local:"transformers.Cohere2ForCausalLM",headingTag:"h2"}}),ne=new Te({props:{name:"class transformers.Cohere2ForCausalLM",anchor:"transformers.Cohere2ForCausalLM",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.Cohere2ForCausalLM.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/cohere2#transformers.Cohere2ForCausalLM">Cohere2ForCausalLM</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/cohere2/modeling_cohere2.py#L420"}}),se=new Te({props:{name:"forward",anchor:"transformers.Cohere2ForCausalLM.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Union[transformers.cache_utils.Cache, list[torch.FloatTensor], NoneType] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"},{name:"logits_to_keep",val:": typing.Union[int, torch.Tensor] = 0"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.Cohere2ForCausalLM.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.Cohere2ForCausalLM.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.Cohere2ForCausalLM.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.Cohere2ForCausalLM.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>Union[~cache_utils.Cache, list[torch.FloatTensor], NoneType]</code>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.Cohere2ForCausalLM.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.Cohere2ForCausalLM.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.Cohere2ForCausalLM.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.Cohere2ForCausalLM.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.Cohere2ForCausalLM.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.Cohere2ForCausalLM.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"},{anchor:"transformers.Cohere2ForCausalLM.forward.logits_to_keep",description:`<strong>logits_to_keep</strong> (<code>Union[int, torch.Tensor]</code>, defaults to <code>0</code>) &#x2014;
If an <code>int</code>, compute logits for the last <code>logits_to_keep</code> tokens. If <code>0</code>, calculate logits for all
<code>input_ids</code> (special case). Only last token logits are needed for generation, and calculating them only for that
token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
If a <code>torch.Tensor</code>, must be 1D corresponding to the indices to keep in the sequence length dimension.
This is useful when using packed tensor format (single dimension for batch and sequence length).`,name:"logits_to_keep"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/cohere2/modeling_cohere2.py#L436",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast"
>transformers.modeling_outputs.CausalLMOutputWithPast</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/cohere2#transformers.Cohere2Config"
>Cohere2Config</a>) and inputs.</p>
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
`}}),G=new nt({props:{$$slots:{default:[Ft]},$$scope:{ctx:v}}}),L=new Ct({props:{anchor:"transformers.Cohere2ForCausalLM.forward.example",$$slots:{default:[Xt]},$$scope:{ctx:v}}}),ae=new Jt({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/cohere2.md"}}),{c(){t=p("meta"),a=i(),o=p("p"),c=i(),C=p("p"),C.innerHTML=b,d=i(),w=p("div"),w.innerHTML=ye,N=i(),u(R.$$.fragment),Ce=i(),V=p("p"),V.innerHTML=at,we=i(),A=p("p"),A.textContent=rt,ve=i(),H=p("p"),H.innerHTML=it,ke=i(),u(B.$$.fragment),Ue=i(),Q=p("p"),Q.innerHTML=lt,$e=i(),u(E.$$.fragment),je=i(),Y=p("p"),Y.innerHTML=dt,Je=i(),S=p("p"),S.innerHTML=ct,xe=i(),u(P.$$.fragment),Ie=i(),u(D.$$.fragment),ze=i(),$=p("div"),u(O.$$.fragment),Le=i(),le=p("p"),le.innerHTML=pt,qe=i(),de=p("p"),de.innerHTML=ht,Ne=i(),u(F.$$.fragment),Re=i(),u(K.$$.fragment),Ze=i(),k=p("div"),u(ee.$$.fragment),Ve=i(),ce=p("p"),ce.textContent=mt,Ae=i(),pe=p("p"),pe.innerHTML=ut,He=i(),he=p("p"),he.innerHTML=ft,Qe=i(),Z=p("div"),u(te.$$.fragment),Ye=i(),me=p("p"),me.innerHTML=gt,Se=i(),u(X.$$.fragment),We=i(),u(oe.$$.fragment),Be=i(),U=p("div"),u(ne.$$.fragment),Pe=i(),ue=p("p"),ue.textContent=_t,De=i(),fe=p("p"),fe.innerHTML=Mt,Oe=i(),ge=p("p"),ge.innerHTML=yt,Ke=i(),x=p("div"),u(se.$$.fragment),et=i(),_e=p("p"),_e.innerHTML=bt,tt=i(),u(G.$$.fragment),ot=i(),u(L.$$.fragment),Ee=i(),u(ae.$$.fragment),Fe=i(),be=p("p"),this.h()},l(e){const n=$t("svelte-u9bgzb",document.head);t=h(n,"META",{name:!0,content:!0}),n.forEach(s),a=l(e),o=h(e,"P",{}),re(o).forEach(s),c=l(e),C=h(e,"P",{"data-svelte-h":!0}),T(C)!=="svelte-1yg019k"&&(C.innerHTML=b),d=l(e),w=h(e,"DIV",{style:!0,"data-svelte-h":!0}),T(w)!=="svelte-11gpmgv"&&(w.innerHTML=ye),N=l(e),f(R.$$.fragment,e),Ce=l(e),V=h(e,"P",{"data-svelte-h":!0}),T(V)!=="svelte-18frc9y"&&(V.innerHTML=at),we=l(e),A=h(e,"P",{"data-svelte-h":!0}),T(A)!=="svelte-1i3ab8t"&&(A.textContent=rt),ve=l(e),H=h(e,"P",{"data-svelte-h":!0}),T(H)!=="svelte-1rlhpz"&&(H.innerHTML=it),ke=l(e),f(B.$$.fragment,e),Ue=l(e),Q=h(e,"P",{"data-svelte-h":!0}),T(Q)!=="svelte-15pjspz"&&(Q.innerHTML=lt),$e=l(e),f(E.$$.fragment,e),je=l(e),Y=h(e,"P",{"data-svelte-h":!0}),T(Y)!=="svelte-wdgc1l"&&(Y.innerHTML=dt),Je=l(e),S=h(e,"P",{"data-svelte-h":!0}),T(S)!=="svelte-1lm9bbj"&&(S.innerHTML=ct),xe=l(e),f(P.$$.fragment,e),Ie=l(e),f(D.$$.fragment,e),ze=l(e),$=h(e,"DIV",{class:!0});var I=re($);f(O.$$.fragment,I),Le=l(I),le=h(I,"P",{"data-svelte-h":!0}),T(le)!=="svelte-w69f57"&&(le.innerHTML=pt),qe=l(I),de=h(I,"P",{"data-svelte-h":!0}),T(de)!=="svelte-zksmgt"&&(de.innerHTML=ht),Ne=l(I),f(F.$$.fragment,I),I.forEach(s),Re=l(e),f(K.$$.fragment,e),Ze=l(e),k=h(e,"DIV",{class:!0});var j=re(k);f(ee.$$.fragment,j),Ve=l(j),ce=h(j,"P",{"data-svelte-h":!0}),T(ce)!=="svelte-12njgno"&&(ce.textContent=mt),Ae=l(j),pe=h(j,"P",{"data-svelte-h":!0}),T(pe)!=="svelte-q52n56"&&(pe.innerHTML=ut),He=l(j),he=h(j,"P",{"data-svelte-h":!0}),T(he)!=="svelte-hswkmf"&&(he.innerHTML=ft),Qe=l(j),Z=h(j,"DIV",{class:!0});var W=re(Z);f(te.$$.fragment,W),Ye=l(W),me=h(W,"P",{"data-svelte-h":!0}),T(me)!=="svelte-h1vgt3"&&(me.innerHTML=gt),Se=l(W),f(X.$$.fragment,W),W.forEach(s),j.forEach(s),We=l(e),f(oe.$$.fragment,e),Be=l(e),U=h(e,"DIV",{class:!0});var J=re(U);f(ne.$$.fragment,J),Pe=l(J),ue=h(J,"P",{"data-svelte-h":!0}),T(ue)!=="svelte-fnxhv1"&&(ue.textContent=_t),De=l(J),fe=h(J,"P",{"data-svelte-h":!0}),T(fe)!=="svelte-q52n56"&&(fe.innerHTML=Mt),Oe=l(J),ge=h(J,"P",{"data-svelte-h":!0}),T(ge)!=="svelte-hswkmf"&&(ge.innerHTML=yt),Ke=l(J),x=h(J,"DIV",{class:!0});var z=re(x);f(se.$$.fragment,z),et=l(z),_e=h(z,"P",{"data-svelte-h":!0}),T(_e)!=="svelte-1pldkib"&&(_e.innerHTML=bt),tt=l(z),f(G.$$.fragment,z),ot=l(z),f(L.$$.fragment,z),z.forEach(s),J.forEach(s),Ee=l(e),f(ae.$$.fragment,e),Fe=l(e),be=h(e,"P",{}),re(be).forEach(s),this.h()},h(){ie(t,"name","hf:doc:metadata"),ie(t,"content",Lt),jt(w,"float","right"),ie($,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),ie(Z,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),ie(k,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),ie(x,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),ie(U,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,n){m(document.head,t),r(e,a,n),r(e,o,n),r(e,c,n),r(e,C,n),r(e,d,n),r(e,w,n),r(e,N,n),g(R,e,n),r(e,Ce,n),r(e,V,n),r(e,we,n),r(e,A,n),r(e,ve,n),r(e,H,n),r(e,ke,n),g(B,e,n),r(e,Ue,n),r(e,Q,n),r(e,$e,n),g(E,e,n),r(e,je,n),r(e,Y,n),r(e,Je,n),r(e,S,n),r(e,xe,n),g(P,e,n),r(e,Ie,n),g(D,e,n),r(e,ze,n),r(e,$,n),g(O,$,null),m($,Le),m($,le),m($,qe),m($,de),m($,Ne),g(F,$,null),r(e,Re,n),g(K,e,n),r(e,Ze,n),r(e,k,n),g(ee,k,null),m(k,Ve),m(k,ce),m(k,Ae),m(k,pe),m(k,He),m(k,he),m(k,Qe),m(k,Z),g(te,Z,null),m(Z,Ye),m(Z,me),m(Z,Se),g(X,Z,null),r(e,We,n),g(oe,e,n),r(e,Be,n),r(e,U,n),g(ne,U,null),m(U,Pe),m(U,ue),m(U,De),m(U,fe),m(U,Oe),m(U,ge),m(U,Ke),m(U,x),g(se,x,null),m(x,et),m(x,_e),m(x,tt),g(G,x,null),m(x,ot),g(L,x,null),r(e,Ee,n),g(ae,e,n),r(e,Fe,n),r(e,be,n),Xe=!0},p(e,[n]){const I={};n&2&&(I.$$scope={dirty:n,ctx:e}),B.$set(I);const j={};n&2&&(j.$$scope={dirty:n,ctx:e}),E.$set(j);const W={};n&2&&(W.$$scope={dirty:n,ctx:e}),F.$set(W);const J={};n&2&&(J.$$scope={dirty:n,ctx:e}),X.$set(J);const z={};n&2&&(z.$$scope={dirty:n,ctx:e}),G.$set(z);const Tt={};n&2&&(Tt.$$scope={dirty:n,ctx:e}),L.$set(Tt)},i(e){Xe||(_(R.$$.fragment,e),_(B.$$.fragment,e),_(E.$$.fragment,e),_(P.$$.fragment,e),_(D.$$.fragment,e),_(O.$$.fragment,e),_(F.$$.fragment,e),_(K.$$.fragment,e),_(ee.$$.fragment,e),_(te.$$.fragment,e),_(X.$$.fragment,e),_(oe.$$.fragment,e),_(ne.$$.fragment,e),_(se.$$.fragment,e),_(G.$$.fragment,e),_(L.$$.fragment,e),_(ae.$$.fragment,e),Xe=!0)},o(e){M(R.$$.fragment,e),M(B.$$.fragment,e),M(E.$$.fragment,e),M(P.$$.fragment,e),M(D.$$.fragment,e),M(O.$$.fragment,e),M(F.$$.fragment,e),M(K.$$.fragment,e),M(ee.$$.fragment,e),M(te.$$.fragment,e),M(X.$$.fragment,e),M(oe.$$.fragment,e),M(ne.$$.fragment,e),M(se.$$.fragment,e),M(G.$$.fragment,e),M(L.$$.fragment,e),M(ae.$$.fragment,e),Xe=!1},d(e){e&&(s(a),s(o),s(c),s(C),s(d),s(w),s(N),s(Ce),s(V),s(we),s(A),s(ve),s(H),s(ke),s(Ue),s(Q),s($e),s(je),s(Y),s(Je),s(S),s(xe),s(Ie),s(ze),s($),s(Re),s(Ze),s(k),s(We),s(Be),s(U),s(Ee),s(Fe),s(be)),s(t),y(R,e),y(B,e),y(E,e),y(P,e),y(D,e),y(O),y(F),y(K,e),y(ee),y(te),y(X),y(oe,e),y(ne),y(se),y(G),y(L),y(ae,e)}}}const Lt='{"title":"Cohere 2","local":"cohere-2","sections":[{"title":"Cohere2Config","local":"transformers.Cohere2Config","sections":[],"depth":2},{"title":"Cohere2Model","local":"transformers.Cohere2Model","sections":[],"depth":2},{"title":"Cohere2ForCausalLM","local":"transformers.Cohere2ForCausalLM","sections":[],"depth":2}],"depth":1}';function qt(v){return vt(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class Dt extends kt{constructor(t){super(),Ut(this,t,qt,Gt,wt,{})}}export{Dt as component};
