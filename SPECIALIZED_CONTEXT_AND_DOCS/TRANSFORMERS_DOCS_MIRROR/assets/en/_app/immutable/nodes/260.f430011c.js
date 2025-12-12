import{s as oa,o as aa,n as R}from"../chunks/scheduler.18a86fab.js";import{S as sa,i as ra,g as l,s as r,r as p,A as ia,h as d,f as s,c as i,j as $,x as b,u as h,k as z,l as la,y as o,a as c,v as u,d as f,t as g,w as _}from"../chunks/index.98837b22.js";import{T as kn}from"../chunks/Tip.77304350.js";import{D as C}from"../chunks/Docstring.a1ef7999.js";import{C as H}from"../chunks/CodeBlock.8d0c2e8a.js";import{E as yn}from"../chunks/ExampleCodeBlock.8c3ee1f9.js";import{H as me,E as da}from"../chunks/getInferenceSnippets.06c2775f.js";import{H as ca,a as _o}from"../chunks/HfOption.6641485e.js";function ma(w){let t,m="Click on the Llama 2 models in the right sidebar for more examples of how to apply Llama to different language tasks.";return{c(){t=l("p"),t.textContent=m},l(n){t=d(n,"P",{"data-svelte-h":!0}),b(t)!=="svelte-holg3z"&&(t.textContent=m)},m(n,k){c(n,t,k)},p:R,d(n){n&&s(t)}}}function pa(w){let t,m;return t=new H({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwcGlwZWxpbmUlMEElMEFwaXBlbGluZSUyMCUzRCUyMHBpcGVsaW5lKCUwQSUyMCUyMCUyMCUyMHRhc2slM0QlMjJ0ZXh0LWdlbmVyYXRpb24lMjIlMkMlMEElMjAlMjAlMjAlMjBtb2RlbCUzRCUyMm1ldGEtbGxhbWElMkZMbGFtYS0yLTdiLWhmJTIyJTJDJTBBJTIwJTIwJTIwJTIwZHR5cGUlM0R0b3JjaC5mbG9hdDE2JTJDJTBBJTIwJTIwJTIwJTIwZGV2aWNlJTNEMCUwQSklMEFwaXBlbGluZSglMjJQbGFudHMlMjBjcmVhdGUlMjBlbmVyZ3klMjB0aHJvdWdoJTIwYSUyMHByb2Nlc3MlMjBrbm93biUyMGFzJTIyKQ==",highlighted:`<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> pipeline

pipeline = pipeline(
    task=<span class="hljs-string">&quot;text-generation&quot;</span>,
    model=<span class="hljs-string">&quot;meta-llama/Llama-2-7b-hf&quot;</span>,
    dtype=torch.float16,
    device=<span class="hljs-number">0</span>
)
pipeline(<span class="hljs-string">&quot;Plants create energy through a process known as&quot;</span>)`,wrap:!1}}),{c(){p(t.$$.fragment)},l(n){h(t.$$.fragment,n)},m(n,k){u(t,n,k),m=!0},p:R,i(n){m||(f(t.$$.fragment,n),m=!0)},o(n){g(t.$$.fragment,n),m=!1},d(n){_(t,n)}}}function ha(w){let t,m;return t=new H({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b01vZGVsRm9yQ2F1c2FsTE0lMkMlMjBBdXRvVG9rZW5pemVyJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTBBJTIwJTIwJTIwJTIwJTIybWV0YS1sbGFtYSUyRkxsYW1hLTItN2ItaGYlMjIlMkMlMEEpJTBBbW9kZWwlMjAlM0QlMjBBdXRvTW9kZWxGb3JDYXVzYWxMTS5mcm9tX3ByZXRyYWluZWQoJTBBJTIwJTIwJTIwJTIwJTIybWV0YS1sbGFtYSUyRkxsYW1hLTItN2ItaGYlMjIlMkMlMEElMjAlMjAlMjAlMjBkdHlwZSUzRHRvcmNoLmZsb2F0MTYlMkMlMEElMjAlMjAlMjAlMjBkZXZpY2VfbWFwJTNEJTIyYXV0byUyMiUyQyUwQSUyMCUyMCUyMCUyMGF0dG5faW1wbGVtZW50YXRpb24lM0QlMjJzZHBhJTIyJTBBKSUwQWlucHV0X2lkcyUyMCUzRCUyMHRva2VuaXplciglMjJQbGFudHMlMjBjcmVhdGUlMjBlbmVyZ3klMjB0aHJvdWdoJTIwYSUyMHByb2Nlc3MlMjBrbm93biUyMGFzJTIyJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMikudG8obW9kZWwuZGV2aWNlKSUwQSUwQW91dHB1dCUyMCUzRCUyMG1vZGVsLmdlbmVyYXRlKCoqaW5wdXRfaWRzJTJDJTIwY2FjaGVfaW1wbGVtZW50YXRpb24lM0QlMjJzdGF0aWMlMjIpJTBBcHJpbnQodG9rZW5pemVyLmRlY29kZShvdXRwdXQlNUIwJTVEJTJDJTIwc2tpcF9zcGVjaWFsX3Rva2VucyUzRFRydWUpKQ==",highlighted:`<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    <span class="hljs-string">&quot;meta-llama/Llama-2-7b-hf&quot;</span>,
)
model = AutoModelForCausalLM.from_pretrained(
    <span class="hljs-string">&quot;meta-llama/Llama-2-7b-hf&quot;</span>,
    dtype=torch.float16,
    device_map=<span class="hljs-string">&quot;auto&quot;</span>,
    attn_implementation=<span class="hljs-string">&quot;sdpa&quot;</span>
)
input_ids = tokenizer(<span class="hljs-string">&quot;Plants create energy through a process known as&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>).to(model.device)

output = model.generate(**input_ids, cache_implementation=<span class="hljs-string">&quot;static&quot;</span>)
<span class="hljs-built_in">print</span>(tokenizer.decode(output[<span class="hljs-number">0</span>], skip_special_tokens=<span class="hljs-literal">True</span>))`,wrap:!1}}),{c(){p(t.$$.fragment)},l(n){h(t.$$.fragment,n)},m(n,k){u(t,n,k),m=!0},p:R,i(n){m||(f(t.$$.fragment,n),m=!0)},o(n){g(t.$$.fragment,n),m=!1},d(n){_(t,n)}}}function ua(w){let t,m;return t=new H({props:{code:"dHJhbnNmb3JtZXJzJTIwY2hhdCUyMG1ldGEtbGxhbWElMkZMbGFtYS0yLTdiLWNoYXQtaGYlMjAtLWR0eXBlJTIwYXV0byUyMC0tYXR0bl9pbXBsZW1lbnRhdGlvbiUyMGZsYXNoX2F0dGVudGlvbl8y",highlighted:"transformers chat meta-llama/Llama-2-7b-chat-hf --dtype auto --attn_implementation flash_attention_2",wrap:!1}}),{c(){p(t.$$.fragment)},l(n){h(t.$$.fragment,n)},m(n,k){u(t,n,k),m=!0},p:R,i(n){m||(f(t.$$.fragment,n),m=!0)},o(n){g(t.$$.fragment,n),m=!1},d(n){_(t,n)}}}function fa(w){let t,m,n,k,M,v;return t=new _o({props:{id:"usage",option:"Pipeline",$$slots:{default:[pa]},$$scope:{ctx:w}}}),n=new _o({props:{id:"usage",option:"AutoModel",$$slots:{default:[ha]},$$scope:{ctx:w}}}),M=new _o({props:{id:"usage",option:"transformers CLI",$$slots:{default:[ua]},$$scope:{ctx:w}}}),{c(){p(t.$$.fragment),m=r(),p(n.$$.fragment),k=r(),p(M.$$.fragment)},l(y){h(t.$$.fragment,y),m=i(y),h(n.$$.fragment,y),k=i(y),h(M.$$.fragment,y)},m(y,x){u(t,y,x),c(y,m,x),u(n,y,x),c(y,k,x),u(M,y,x),v=!0},p(y,x){const Bt={};x&2&&(Bt.$$scope={dirty:x,ctx:y}),t.$set(Bt);const pe={};x&2&&(pe.$$scope={dirty:x,ctx:y}),n.$set(pe);const V={};x&2&&(V.$$scope={dirty:x,ctx:y}),M.$set(V)},i(y){v||(f(t.$$.fragment,y),f(n.$$.fragment,y),f(M.$$.fragment,y),v=!0)},o(y){g(t.$$.fragment,y),g(n.$$.fragment,y),g(M.$$.fragment,y),v=!1},d(y){y&&(s(m),s(k)),_(t,y),_(n,y),_(M,y)}}}function ga(w){let t,m;return t=new H({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMExsYW1hTW9kZWwlMkMlMjBMbGFtYUNvbmZpZyUwQSUwQSUyMyUyMEluaXRpYWxpemluZyUyMGElMjBMTGFNQSUyMGxsYW1hLTdiJTIwc3R5bGUlMjBjb25maWd1cmF0aW9uJTBBY29uZmlndXJhdGlvbiUyMCUzRCUyMExsYW1hQ29uZmlnKCklMEElMEElMjMlMjBJbml0aWFsaXppbmclMjBhJTIwbW9kZWwlMjBmcm9tJTIwdGhlJTIwbGxhbWEtN2IlMjBzdHlsZSUyMGNvbmZpZ3VyYXRpb24lMEFtb2RlbCUyMCUzRCUyMExsYW1hTW9kZWwoY29uZmlndXJhdGlvbiklMEElMEElMjMlMjBBY2Nlc3NpbmclMjB0aGUlMjBtb2RlbCUyMGNvbmZpZ3VyYXRpb24lMEFjb25maWd1cmF0aW9uJTIwJTNEJTIwbW9kZWwuY29uZmln",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> LlamaModel, LlamaConfig

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a LLaMA llama-7b style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = LlamaConfig()

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a model from the llama-7b style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = LlamaModel(configuration)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Accessing the model configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = model.config`,wrap:!1}}),{c(){p(t.$$.fragment)},l(n){h(t.$$.fragment,n)},m(n,k){u(t,n,k),m=!0},p:R,i(n){m||(f(t.$$.fragment,n),m=!0)},o(n){g(t.$$.fragment,n),m=!1},d(n){_(t,n)}}}function _a(w){let t,m="sequence pair mask has the following format:",n,k,M;return k=new H({props:{code:"MCUyMDAlMjAwJTIwMCUyMDAlMjAwJTIwMCUyMDAlMjAwJTIwMCUyMDAlMjAxJTIwMSUyMDElMjAxJTIwMSUyMDElMjAxJTIwMSUyMDElMEElN0MlMjBmaXJzdCUyMHNlcXVlbmNlJTIwJTIwJTIwJTIwJTdDJTIwc2Vjb25kJTIwc2VxdWVuY2UlMjAlN0M=",highlighted:`0<span class="hljs-number"> 0 </span>0<span class="hljs-number"> 0 </span>0<span class="hljs-number"> 0 </span>0<span class="hljs-number"> 0 </span>0<span class="hljs-number"> 0 </span>0<span class="hljs-number"> 1 </span>1<span class="hljs-number"> 1 </span>1<span class="hljs-number"> 1 </span>1<span class="hljs-number"> 1 </span>1 1
| first sequence    | second sequence |`,wrap:!1}}),{c(){t=l("p"),t.textContent=m,n=r(),p(k.$$.fragment)},l(v){t=d(v,"P",{"data-svelte-h":!0}),b(t)!=="svelte-16klr56"&&(t.textContent=m),n=i(v),h(k.$$.fragment,v)},m(v,y){c(v,t,y),c(v,n,y),u(k,v,y),M=!0},p:R,i(v){M||(f(k.$$.fragment,v),M=!0)},o(v){g(k.$$.fragment,v),M=!1},d(v){v&&(s(t),s(n)),_(k,v)}}}function ba(w){let t,m;return t=new H({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMExsYW1hVG9rZW5pemVyRmFzdCUwQSUwQXRva2VuaXplciUyMCUzRCUyMExsYW1hVG9rZW5pemVyRmFzdC5mcm9tX3ByZXRyYWluZWQoJTIyaGYtaW50ZXJuYWwtdGVzdGluZyUyRmxsYW1hLXRva2VuaXplciUyMiklMEF0b2tlbml6ZXIuZW5jb2RlKCUyMkhlbGxvJTIwdGhpcyUyMGlzJTIwYSUyMHRlc3QlMjIp",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> LlamaTokenizerFast

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = LlamaTokenizerFast.from_pretrained(<span class="hljs-string">&quot;hf-internal-testing/llama-tokenizer&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer.encode(<span class="hljs-string">&quot;Hello this is a test&quot;</span>)
[<span class="hljs-number">1</span>, <span class="hljs-number">15043</span>, <span class="hljs-number">445</span>, <span class="hljs-number">338</span>, <span class="hljs-number">263</span>, <span class="hljs-number">1243</span>]`,wrap:!1}}),{c(){p(t.$$.fragment)},l(n){h(t.$$.fragment,n)},m(n,k){u(t,n,k),m=!0},p:R,i(n){m||(f(t.$$.fragment,n),m=!0)},o(n){g(t.$$.fragment,n),m=!1},d(n){_(t,n)}}}function ka(w){let t,m=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=l("p"),t.innerHTML=m},l(n){t=d(n,"P",{"data-svelte-h":!0}),b(t)!=="svelte-fincs2"&&(t.innerHTML=m)},m(n,k){c(n,t,k)},p:R,d(n){n&&s(t)}}}function ya(w){let t,m=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=l("p"),t.innerHTML=m},l(n){t=d(n,"P",{"data-svelte-h":!0}),b(t)!=="svelte-fincs2"&&(t.innerHTML=m)},m(n,k){c(n,t,k)},p:R,d(n){n&&s(t)}}}function va(w){let t,m="Example:",n,k,M;return k=new H({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBMbGFtYUZvckNhdXNhbExNJTBBJTBBbW9kZWwlMjAlM0QlMjBMbGFtYUZvckNhdXNhbExNLmZyb21fcHJldHJhaW5lZCglMjJtZXRhLWxsYW1hJTJGTGxhbWEtMi03Yi1oZiUyMiklMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJtZXRhLWxsYW1hJTJGTGxhbWEtMi03Yi1oZiUyMiklMEElMEFwcm9tcHQlMjAlM0QlMjAlMjJIZXklMkMlMjBhcmUlMjB5b3UlMjBjb25zY2lvdXMlM0YlMjBDYW4lMjB5b3UlMjB0YWxrJTIwdG8lMjBtZSUzRiUyMiUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplcihwcm9tcHQlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQSUwQSUyMyUyMEdlbmVyYXRlJTBBZ2VuZXJhdGVfaWRzJTIwJTNEJTIwbW9kZWwuZ2VuZXJhdGUoaW5wdXRzLmlucHV0X2lkcyUyQyUyMG1heF9sZW5ndGglM0QzMCklMEF0b2tlbml6ZXIuYmF0Y2hfZGVjb2RlKGdlbmVyYXRlX2lkcyUyQyUyMHNraXBfc3BlY2lhbF90b2tlbnMlM0RUcnVlJTJDJTIwY2xlYW5fdXBfdG9rZW5pemF0aW9uX3NwYWNlcyUzREZhbHNlKSU1QjAlNUQ=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, LlamaForCausalLM

<span class="hljs-meta">&gt;&gt;&gt; </span>model = LlamaForCausalLM.from_pretrained(<span class="hljs-string">&quot;meta-llama/Llama-2-7b-hf&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;meta-llama/Llama-2-7b-hf&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>prompt = <span class="hljs-string">&quot;Hey, are you conscious? Can you talk to me?&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(prompt, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Generate</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>generate_ids = model.generate(inputs.input_ids, max_length=<span class="hljs-number">30</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer.batch_decode(generate_ids, skip_special_tokens=<span class="hljs-literal">True</span>, clean_up_tokenization_spaces=<span class="hljs-literal">False</span>)[<span class="hljs-number">0</span>]
<span class="hljs-string">&quot;Hey, are you conscious? Can you talk to me?\\nI&#x27;m not conscious, but I can talk to you.&quot;</span>`,wrap:!1}}),{c(){t=l("p"),t.textContent=m,n=r(),p(k.$$.fragment)},l(v){t=d(v,"P",{"data-svelte-h":!0}),b(t)!=="svelte-11lpom8"&&(t.textContent=m),n=i(v),h(k.$$.fragment,v)},m(v,y){c(v,t,y),c(v,n,y),u(k,v,y),M=!0},p:R,i(v){M||(f(k.$$.fragment,v),M=!0)},o(v){g(k.$$.fragment,v),M=!1},d(v){v&&(s(t),s(n)),_(k,v)}}}function Ta(w){let t,m=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=l("p"),t.innerHTML=m},l(n){t=d(n,"P",{"data-svelte-h":!0}),b(t)!=="svelte-fincs2"&&(t.innerHTML=m)},m(n,k){c(n,t,k)},p:R,d(n){n&&s(t)}}}function wa(w){let t,m,n,k,M,v="<em>This model was released on 2023-07-18 and added to Hugging Face Transformers on 2023-07-18.</em>",y,x,Bt='<div class="flex flex-wrap space-x-1"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/> <img alt="Tensor parallelism" src="https://img.shields.io/badge/Tensor%20parallelism-06b6d4?style=flat&amp;logoColor=white"/></div>',pe,V,Pt,he,bo='<a href="https://huggingface.co/papers/2307.09288" rel="nofollow">Llama 2</a> is a family of large language models, Llama 2 and Llama 2-Chat, available in 7B, 13B, and 70B parameters. The Llama 2 model mostly keeps the same architecture as <a href="./llama">Llama</a>, but it is pretrained on more tokens, doubles the context length, and uses grouped-query attention (GQA) in the 70B model to improve inference.',Ht,ue,ko="Llama 2-Chat is trained with supervised fine-tuning (SFT), and reinforcement learning with human feedback (RLHF) - rejection sampling and proximal policy optimization (PPO) - is applied to the fine-tuned model to align the chat model with human preferences.",Rt,fe,yo='You can find all the original Llama 2 checkpoints under the <a href="https://huggingface.co/collections/meta-llama/llama-2-family-661da1f90a9d678b6f55773b" rel="nofollow">Llama 2 Family</a> collection.',Vt,S,Et,ge,vo='The example below demonstrates how to generate text with <a href="/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.Pipeline">Pipeline</a>, <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel">AutoModel</a>, and how to chat with Llama 2-Chat from the command line.',Xt,Q,Nt,_e,To='Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the <a href="../quantization/overview">Quantization</a> overview for more available quantization backends.',Yt,be,wo='The example below uses <a href="../quantization/torchao">torchao</a> to only quantize the weights to int4.',At,ke,Dt,ye,Mo='Use the <a href="https://github.com/huggingface/transformers/blob/beb9b5b02246b9b7ee81ddf938f93f44cfeaad19/src/transformers/utils/attention_visualizer.py#L139" rel="nofollow">AttentionMaskVisualizer</a> to better understand what tokens the model can and cannot attend to.',St,ve,Qt,O,$o='<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/llama-2-attn-mask.png"/>',Ot,Te,Kt,W,nt,Lo="<p>Setting <code>config.pretraining_tp</code> to a value besides <code>1</code> activates a more accurate but slower computation of the linear layers. This matches the original logits better.</p>",vn,we,ot,xo="The original model uses <code>pad_id = -1</code> to indicate a padding token. The Transformers implementation requires adding a padding token and resizing the token embedding accordingly.",Tn,Me,wn,$e,at,zo="It is recommended to initialize the <code>embed_tokens</code> layer with the following code to ensure encoding the padding token outputs zeros.",Mn,Le,$n,st,Co='<p>The tokenizer is a byte-pair encoding model based on <a href="https://github.com/google/sentencepiece" rel="nofollow">SentencePiece</a>. During decoding, if the first token is the start of the word (for example, “Banana”), the tokenizer doesn’t prepend the prefix space to the string.</p>',Ln,rt,jo='<p>Don’t use the <code>dtype</code> parameter in <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel.from_pretrained">from_pretrained()</a> if you’re using FlashAttention-2 because it only supports fp16 or bf16. You should use <a href="https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html" rel="nofollow">Automatic Mixed Precision</a>, set fp16 or bf16 to <code>True</code> if using <a href="/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.Trainer">Trainer</a>, or use <a href="https://pytorch.org/docs/stable/amp.html#torch.autocast" rel="nofollow">torch.autocast</a>.</p>',en,xe,tn,q,ze,xn,it,Jo=`This is the configuration class to store the configuration of a <a href="/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaModel">LlamaModel</a>. It is used to instantiate an LLaMA
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the LLaMA-7B.
e.g. <a href="https://huggingface.co/meta-llama/Llama-2-7b-hf" rel="nofollow">meta-llama/Llama-2-7b-hf</a>`,zn,lt,Fo=`Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> and can be used to control the model outputs. Read the
documentation from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> for more information.`,Cn,K,nn,Ce,on,j,je,jn,dt,Io=`Construct a Llama tokenizer. Based on byte-level Byte-Pair-Encoding. The default padding token is unset as there is
no padding token in the original model.`,Jn,ct,Je,Fn,ee,Fe,In,mt,Wo=`Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
special tokens using the tokenizer <code>prepare_for_model</code> method.`,Wn,B,Ie,qn,pt,qo="Creates a mask from the two sequences passed to be used in a sequence-pair classification task. An ALBERT",Un,te,Bn,ht,Uo="if token_ids_1 is None, only returns the first portion of the mask (0s).",Zn,ne,We,Gn,ut,Bo="Save the vocabulary and special tokens file to a directory.",an,qe,sn,T,Ue,Pn,ft,Zo="Construct a Llama tokenizer. Based on byte-level Byte-Pair-Encoding.",Hn,gt,Go="This uses notably ByteFallback and no normalization.",Rn,oe,Vn,_t,Po=`If you want to change the <code>bos_token</code> or the <code>eos_token</code>, make sure to specify them when initializing the model, or
call <code>tokenizer.update_post_processor()</code> to make sure that the post-processing is correctly done (otherwise the
values of the first token and final token of an encoded sequence will not be correct). For more details, checkout
[post-processors] (<a href="https://huggingface.co/docs/tokenizers/api/post-processors" rel="nofollow">https://huggingface.co/docs/tokenizers/api/post-processors</a>) documentation.`,En,bt,Ho=`This tokenizer inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast">PreTrainedTokenizerFast</a> which contains most of the main methods. Users should
refer to this superclass for more information regarding those methods.`,Xn,kt,Be,Nn,ae,Ze,Yn,yt,Ro=`Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
special tokens using the tokenizer <code>prepare_for_model</code> or <code>encode_plus</code> methods.`,An,E,Ge,Dn,vt,Vo=`Create the token type IDs corresponding to the sequences passed. <a href="../glossary#token-type-ids">What are token type
IDs?</a>`,Sn,Tt,Eo="Should be overridden in a subclass if the model has a special way of building those.",Qn,se,Pe,On,wt,Xo="Updates the underlying post processor with the current <code>bos_token</code> and <code>eos_token</code>.",Kn,Mt,He,rn,Re,ln,J,Ve,eo,$t,No="The bare Llama Model outputting raw hidden-states without any specific head on top.",to,Lt,Yo=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,no,xt,Ao=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,oo,X,Ee,ao,zt,Do='The <a href="/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaModel">LlamaModel</a> forward method, overrides the <code>__call__</code> special method.',so,re,dn,Xe,cn,F,Ne,ro,Ct,So="The Llama Model for causal language modeling.",io,jt,Qo=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,lo,Jt,Oo=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,co,Z,Ye,mo,Ft,Ko='The <a href="/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaForCausalLM">LlamaForCausalLM</a> forward method, overrides the <code>__call__</code> special method.',po,ie,ho,le,mn,Ae,pn,D,De,uo,N,Se,fo,It,ea="The <code>GenericForSequenceClassification</code> forward method, overrides the <code>__call__</code> special method.",go,de,hn,Qe,un,Zt,fn;return V=new me({props:{title:"Llama 2",local:"llama-2",headingTag:"h1"}}),S=new kn({props:{warning:!1,$$slots:{default:[ma]},$$scope:{ctx:w}}}),Q=new ca({props:{id:"usage",options:["Pipeline","AutoModel","transformers CLI"],$$slots:{default:[fa]},$$scope:{ctx:w}}}),ke=new H({props:{code:"JTIzJTIwcGlwJTIwaW5zdGFsbCUyMHRvcmNoYW8lMEFpbXBvcnQlMjB0b3JjaCUwQWZyb20lMjB0cmFuc2Zvcm1lcnMlMjBpbXBvcnQlMjBUb3JjaEFvQ29uZmlnJTJDJTIwQXV0b01vZGVsRm9yQ2F1c2FsTE0lMkMlMjBBdXRvVG9rZW5pemVyJTBBJTBBcXVhbnRpemF0aW9uX2NvbmZpZyUyMCUzRCUyMFRvcmNoQW9Db25maWcoJTIyaW50NF93ZWlnaHRfb25seSUyMiUyQyUyMGdyb3VwX3NpemUlM0QxMjgpJTBBbW9kZWwlMjAlM0QlMjBBdXRvTW9kZWxGb3JDYXVzYWxMTS5mcm9tX3ByZXRyYWluZWQoJTBBJTIwJTIwJTIwJTIwJTIybWV0YS1sbGFtYSUyRkxsYW1hLTItMTNiLWhmJTIyJTJDJTBBJTIwJTIwJTIwJTIwZHR5cGUlM0R0b3JjaC5iZmxvYXQxNiUyQyUwQSUyMCUyMCUyMCUyMGRldmljZV9tYXAlM0QlMjJhdXRvJTIyJTJDJTBBJTIwJTIwJTIwJTIwcXVhbnRpemF0aW9uX2NvbmZpZyUzRHF1YW50aXphdGlvbl9jb25maWclMEEpJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIybWV0YS1sbGFtYSUyRkxsYW1hLTItMTNiLWhmJTIyKSUwQWlucHV0X2lkcyUyMCUzRCUyMHRva2VuaXplciglMjJQbGFudHMlMjBjcmVhdGUlMjBlbmVyZ3klMjB0aHJvdWdoJTIwYSUyMHByb2Nlc3MlMjBrbm93biUyMGFzJTIyJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMikudG8obW9kZWwuZGV2aWNlKSUwQSUwQW91dHB1dCUyMCUzRCUyMG1vZGVsLmdlbmVyYXRlKCoqaW5wdXRfaWRzJTJDJTIwY2FjaGVfaW1wbGVtZW50YXRpb24lM0QlMjJzdGF0aWMlMjIpJTBBcHJpbnQodG9rZW5pemVyLmRlY29kZShvdXRwdXQlNUIwJTVEJTJDJTIwc2tpcF9zcGVjaWFsX3Rva2VucyUzRFRydWUpKQ==",highlighted:`<span class="hljs-comment"># pip install torchao</span>
<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> TorchAoConfig, AutoModelForCausalLM, AutoTokenizer

quantization_config = TorchAoConfig(<span class="hljs-string">&quot;int4_weight_only&quot;</span>, group_size=<span class="hljs-number">128</span>)
model = AutoModelForCausalLM.from_pretrained(
    <span class="hljs-string">&quot;meta-llama/Llama-2-13b-hf&quot;</span>,
    dtype=torch.bfloat16,
    device_map=<span class="hljs-string">&quot;auto&quot;</span>,
    quantization_config=quantization_config
)

tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;meta-llama/Llama-2-13b-hf&quot;</span>)
input_ids = tokenizer(<span class="hljs-string">&quot;Plants create energy through a process known as&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>).to(model.device)

output = model.generate(**input_ids, cache_implementation=<span class="hljs-string">&quot;static&quot;</span>)
<span class="hljs-built_in">print</span>(tokenizer.decode(output[<span class="hljs-number">0</span>], skip_special_tokens=<span class="hljs-literal">True</span>))`,wrap:!1}}),ve=new H({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycy51dGlscy5hdHRlbnRpb25fdmlzdWFsaXplciUyMGltcG9ydCUyMEF0dGVudGlvbk1hc2tWaXN1YWxpemVyJTBBJTBBdmlzdWFsaXplciUyMCUzRCUyMEF0dGVudGlvbk1hc2tWaXN1YWxpemVyKCUyMm1ldGEtbGxhbWElMkZMbGFtYS0yLTdiLWhmJTIyKSUwQXZpc3VhbGl6ZXIoJTIyUGxhbnRzJTIwY3JlYXRlJTIwZW5lcmd5JTIwdGhyb3VnaCUyMGElMjBwcm9jZXNzJTIwa25vd24lMjBhcyUyMik=",highlighted:`<span class="hljs-keyword">from</span> transformers.utils.attention_visualizer <span class="hljs-keyword">import</span> AttentionMaskVisualizer

visualizer = AttentionMaskVisualizer(<span class="hljs-string">&quot;meta-llama/Llama-2-7b-hf&quot;</span>)
visualizer(<span class="hljs-string">&quot;Plants create energy through a process known as&quot;</span>)`,wrap:!1}}),Te=new me({props:{title:"Notes",local:"notes",headingTag:"h2"}}),Me=new H({props:{code:"dG9rZW5pemVyLmFkZF9zcGVjaWFsX3Rva2VucyglN0IlMjJwYWRfdG9rZW4lMjIlM0ElMjIlM0NwYWQlM0UlMjIlN0QpJTBBJTIzJTIwdXBkYXRlJTIwbW9kZWwlMjBjb25maWclMjB3aXRoJTIwcGFkZGluZyUyMHRva2VuJTBBbW9kZWwuY29uZmlnLnBhZF90b2tlbl9pZA==",highlighted:`tokenizer.add_special_tokens({<span class="hljs-string">&quot;pad_token&quot;</span>:<span class="hljs-string">&quot;&lt;pad&gt;&quot;</span>})
<span class="hljs-comment"># update model config with padding token</span>
model.config.pad_token_id`,wrap:!1}}),Le=new H({props:{code:"c2VsZi5lbWJlZF90b2tlbnMlMjAlM0QlMjBubi5FbWJlZGRpbmcoY29uZmlnLnZvY2FiX3NpemUlMkMlMjBjb25maWcuaGlkZGVuX3NpemUlMkMlMjBzZWxmLmNvbmZpZy5wYWRkaW5nX2lkeCk=",highlighted:"self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.config.padding_idx)",wrap:!1}}),xe=new me({props:{title:"LlamaConfig",local:"transformers.LlamaConfig",headingTag:"h2"}}),ze=new C({props:{name:"class transformers.LlamaConfig",anchor:"transformers.LlamaConfig",parameters:[{name:"vocab_size",val:" = 32000"},{name:"hidden_size",val:" = 4096"},{name:"intermediate_size",val:" = 11008"},{name:"num_hidden_layers",val:" = 32"},{name:"num_attention_heads",val:" = 32"},{name:"num_key_value_heads",val:" = None"},{name:"hidden_act",val:" = 'silu'"},{name:"max_position_embeddings",val:" = 2048"},{name:"initializer_range",val:" = 0.02"},{name:"rms_norm_eps",val:" = 1e-06"},{name:"use_cache",val:" = True"},{name:"pad_token_id",val:" = None"},{name:"bos_token_id",val:" = 1"},{name:"eos_token_id",val:" = 2"},{name:"pretraining_tp",val:" = 1"},{name:"tie_word_embeddings",val:" = False"},{name:"rope_theta",val:" = 10000.0"},{name:"rope_scaling",val:" = None"},{name:"attention_bias",val:" = False"},{name:"attention_dropout",val:" = 0.0"},{name:"mlp_bias",val:" = False"},{name:"head_dim",val:" = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.LlamaConfig.vocab_size",description:`<strong>vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 32000) &#x2014;
Vocabulary size of the LLaMA model. Defines the number of different tokens that can be represented by the
<code>inputs_ids</code> passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaModel">LlamaModel</a>`,name:"vocab_size"},{anchor:"transformers.LlamaConfig.hidden_size",description:`<strong>hidden_size</strong> (<code>int</code>, <em>optional</em>, defaults to 4096) &#x2014;
Dimension of the hidden representations.`,name:"hidden_size"},{anchor:"transformers.LlamaConfig.intermediate_size",description:`<strong>intermediate_size</strong> (<code>int</code>, <em>optional</em>, defaults to 11008) &#x2014;
Dimension of the MLP representations.`,name:"intermediate_size"},{anchor:"transformers.LlamaConfig.num_hidden_layers",description:`<strong>num_hidden_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 32) &#x2014;
Number of hidden layers in the Transformer decoder.`,name:"num_hidden_layers"},{anchor:"transformers.LlamaConfig.num_attention_heads",description:`<strong>num_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 32) &#x2014;
Number of attention heads for each attention layer in the Transformer decoder.`,name:"num_attention_heads"},{anchor:"transformers.LlamaConfig.num_key_value_heads",description:`<strong>num_key_value_heads</strong> (<code>int</code>, <em>optional</em>) &#x2014;
This is the number of key_value heads that should be used to implement Grouped Query Attention. If
<code>num_key_value_heads=num_attention_heads</code>, the model will use Multi Head Attention (MHA), if
<code>num_key_value_heads=1</code> the model will use Multi Query Attention (MQA) otherwise GQA is used. When
converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
by meanpooling all the original heads within that group. For more details, check out <a href="https://huggingface.co/papers/2305.13245" rel="nofollow">this
paper</a>. If it is not specified, will default to
<code>num_attention_heads</code>.`,name:"num_key_value_heads"},{anchor:"transformers.LlamaConfig.hidden_act",description:`<strong>hidden_act</strong> (<code>str</code> or <code>function</code>, <em>optional</em>, defaults to <code>&quot;silu&quot;</code>) &#x2014;
The non-linear activation function (function or string) in the decoder.`,name:"hidden_act"},{anchor:"transformers.LlamaConfig.max_position_embeddings",description:`<strong>max_position_embeddings</strong> (<code>int</code>, <em>optional</em>, defaults to 2048) &#x2014;
The maximum sequence length that this model might ever be used with. Llama 1 supports up to 2048 tokens,
Llama 2 up to 4096, CodeLlama up to 16384.`,name:"max_position_embeddings"},{anchor:"transformers.LlamaConfig.initializer_range",description:`<strong>initializer_range</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.`,name:"initializer_range"},{anchor:"transformers.LlamaConfig.rms_norm_eps",description:`<strong>rms_norm_eps</strong> (<code>float</code>, <em>optional</em>, defaults to 1e-06) &#x2014;
The epsilon used by the rms normalization layers.`,name:"rms_norm_eps"},{anchor:"transformers.LlamaConfig.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not the model should return the last key/values attentions (not used by all models). Only
relevant if <code>config.is_decoder=True</code>.`,name:"use_cache"},{anchor:"transformers.LlamaConfig.pad_token_id",description:`<strong>pad_token_id</strong> (<code>int</code>, <em>optional</em>) &#x2014;
Padding token id.`,name:"pad_token_id"},{anchor:"transformers.LlamaConfig.bos_token_id",description:`<strong>bos_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 1) &#x2014;
Beginning of stream token id.`,name:"bos_token_id"},{anchor:"transformers.LlamaConfig.eos_token_id",description:`<strong>eos_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 2) &#x2014;
End of stream token id.`,name:"eos_token_id"},{anchor:"transformers.LlamaConfig.pretraining_tp",description:`<strong>pretraining_tp</strong> (<code>int</code>, <em>optional</em>, defaults to 1) &#x2014;
Experimental feature. Tensor parallelism rank used during pretraining. Please refer to <a href="https://huggingface.co/docs/transformers/main/perf_train_gpu_many#tensor-parallelism" rel="nofollow">this
document</a> to
understand more about it. This value is necessary to ensure exact reproducibility of the pretraining
results. Please refer to <a href="https://github.com/pytorch/pytorch/issues/76232" rel="nofollow">this issue</a>.`,name:"pretraining_tp"},{anchor:"transformers.LlamaConfig.tie_word_embeddings",description:`<strong>tie_word_embeddings</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether to tie weight embeddings`,name:"tie_word_embeddings"},{anchor:"transformers.LlamaConfig.rope_theta",description:`<strong>rope_theta</strong> (<code>float</code>, <em>optional</em>, defaults to 10000.0) &#x2014;
The base period of the RoPE embeddings.`,name:"rope_theta"},{anchor:"transformers.LlamaConfig.rope_scaling",description:`<strong>rope_scaling</strong> (<code>Dict</code>, <em>optional</em>) &#x2014;
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
Only used with &#x2018;llama3&#x2019;. Scaling factor applied to high frequency components of the RoPE`,name:"rope_scaling"},{anchor:"transformers.LlamaConfig.attention_bias",description:`<strong>attention_bias</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether to use a bias in the query, key, value and output projection layers during self-attention.`,name:"attention_bias"},{anchor:"transformers.LlamaConfig.attention_dropout",description:`<strong>attention_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The dropout ratio for the attention probabilities.`,name:"attention_dropout"},{anchor:"transformers.LlamaConfig.mlp_bias",description:`<strong>mlp_bias</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether to use a bias in up_proj, down_proj and gate_proj layers in the MLP layers.`,name:"mlp_bias"},{anchor:"transformers.LlamaConfig.head_dim",description:`<strong>head_dim</strong> (<code>int</code>, <em>optional</em>) &#x2014;
The attention head dimension. If None, it will default to hidden_size // num_attention_heads`,name:"head_dim"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/llama/configuration_llama.py#L26"}}),K=new yn({props:{anchor:"transformers.LlamaConfig.example",$$slots:{default:[ga]},$$scope:{ctx:w}}}),Ce=new me({props:{title:"LlamaTokenizer",local:"transformers.LlamaTokenizer",headingTag:"h2"}}),je=new C({props:{name:"class transformers.LlamaTokenizer",anchor:"transformers.LlamaTokenizer",parameters:[{name:"vocab_file",val:""},{name:"unk_token",val:" = '<unk>'"},{name:"bos_token",val:" = '<s>'"},{name:"eos_token",val:" = '</s>'"},{name:"pad_token",val:" = None"},{name:"sp_model_kwargs",val:": typing.Optional[dict[str, typing.Any]] = None"},{name:"add_bos_token",val:" = True"},{name:"add_eos_token",val:" = False"},{name:"clean_up_tokenization_spaces",val:" = False"},{name:"use_default_system_prompt",val:" = False"},{name:"spaces_between_special_tokens",val:" = False"},{name:"legacy",val:" = None"},{name:"add_prefix_space",val:" = True"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.LlamaTokenizer.vocab_file",description:`<strong>vocab_file</strong> (<code>str</code>) &#x2014;
Path to the vocabulary file.`,name:"vocab_file"},{anchor:"transformers.LlamaTokenizer.unk_token",description:`<strong>unk_token</strong> (<code>str</code> or <code>tokenizers.AddedToken</code>, <em>optional</em>, defaults to <code>&quot;&lt;unk&gt;&quot;</code>) &#x2014;
The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
token instead.`,name:"unk_token"},{anchor:"transformers.LlamaTokenizer.bos_token",description:`<strong>bos_token</strong> (<code>str</code> or <code>tokenizers.AddedToken</code>, <em>optional</em>, defaults to <code>&quot;&lt;s&gt;&quot;</code>) &#x2014;
The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.`,name:"bos_token"},{anchor:"transformers.LlamaTokenizer.eos_token",description:`<strong>eos_token</strong> (<code>str</code> or <code>tokenizers.AddedToken</code>, <em>optional</em>, defaults to <code>&quot;&lt;/s&gt;&quot;</code>) &#x2014;
The end of sequence token.`,name:"eos_token"},{anchor:"transformers.LlamaTokenizer.pad_token",description:`<strong>pad_token</strong> (<code>str</code> or <code>tokenizers.AddedToken</code>, <em>optional</em>) &#x2014;
A special token used to make arrays of tokens the same size for batching purpose. Will then be ignored by
attention mechanisms or loss computation.`,name:"pad_token"},{anchor:"transformers.LlamaTokenizer.sp_model_kwargs",description:`<strong>sp_model_kwargs</strong> (<code>dict[str, Any]</code>, <code>Optional</code>, <em>optional</em>) &#x2014;
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
</ul>`,name:"sp_model_kwargs"},{anchor:"transformers.LlamaTokenizer.add_bos_token",description:`<strong>add_bos_token</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to add an <code>bos_token</code> at the start of sequences.`,name:"add_bos_token"},{anchor:"transformers.LlamaTokenizer.add_eos_token",description:`<strong>add_eos_token</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to add an <code>eos_token</code> at the end of sequences.`,name:"add_eos_token"},{anchor:"transformers.LlamaTokenizer.clean_up_tokenization_spaces",description:`<strong>clean_up_tokenization_spaces</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to cleanup spaces after decoding, cleanup consists in removing potential artifacts like
extra spaces.`,name:"clean_up_tokenization_spaces"},{anchor:"transformers.LlamaTokenizer.use_default_system_prompt",description:`<strong>use_default_system_prompt</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not the default system prompt for Llama should be used.`,name:"use_default_system_prompt"},{anchor:"transformers.LlamaTokenizer.spaces_between_special_tokens",description:`<strong>spaces_between_special_tokens</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to add spaces between special tokens.`,name:"spaces_between_special_tokens"},{anchor:"transformers.LlamaTokenizer.legacy",description:`<strong>legacy</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not the <code>legacy</code> behavior of the tokenizer should be used. Legacy is before the merge of #24622
and #25224 which includes fixes to properly handle tokens that appear after special tokens.
Make sure to also set <code>from_slow</code> to <code>True</code>.
A simple example:</p>
<ul>
<li><code>legacy=True</code>:</li>
</ul>`,name:"legacy"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/llama/tokenization_llama.py#L56"}}),Je=new C({props:{name:"build_inputs_with_special_tokens",anchor:"transformers.LlamaTokenizer.build_inputs_with_special_tokens",parameters:[{name:"token_ids_0",val:""},{name:"token_ids_1",val:" = None"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/llama/tokenization_llama.py#L333"}}),Fe=new C({props:{name:"get_special_tokens_mask",anchor:"transformers.LlamaTokenizer.get_special_tokens_mask",parameters:[{name:"token_ids_0",val:": list"},{name:"token_ids_1",val:": typing.Optional[list[int]] = None"},{name:"already_has_special_tokens",val:": bool = False"}],parametersDescription:[{anchor:"transformers.LlamaTokenizer.get_special_tokens_mask.token_ids_0",description:`<strong>token_ids_0</strong> (<code>list[int]</code>) &#x2014;
List of IDs.`,name:"token_ids_0"},{anchor:"transformers.LlamaTokenizer.get_special_tokens_mask.token_ids_1",description:`<strong>token_ids_1</strong> (<code>list[int]</code>, <em>optional</em>) &#x2014;
Optional second list of IDs for sequence pairs.`,name:"token_ids_1"},{anchor:"transformers.LlamaTokenizer.get_special_tokens_mask.already_has_special_tokens",description:`<strong>already_has_special_tokens</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not the token list is already formatted with special tokens for the model.`,name:"already_has_special_tokens"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/llama/tokenization_llama.py#L344",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>list[int]</code></p>
`}}),Ie=new C({props:{name:"create_token_type_ids_from_sequences",anchor:"transformers.LlamaTokenizer.create_token_type_ids_from_sequences",parameters:[{name:"token_ids_0",val:": list"},{name:"token_ids_1",val:": typing.Optional[list[int]] = None"}],parametersDescription:[{anchor:"transformers.LlamaTokenizer.create_token_type_ids_from_sequences.token_ids_0",description:`<strong>token_ids_0</strong> (<code>list[int]</code>) &#x2014;
List of ids.`,name:"token_ids_0"},{anchor:"transformers.LlamaTokenizer.create_token_type_ids_from_sequences.token_ids_1",description:`<strong>token_ids_1</strong> (<code>list[int]</code>, <em>optional</em>) &#x2014;
Optional second list of IDs for sequence pairs.`,name:"token_ids_1"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/llama/tokenization_llama.py#L381",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>List of <a href="../glossary#token-type-ids">token type IDs</a> according to the given sequence(s).</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>list[int]</code></p>
`}}),te=new yn({props:{anchor:"transformers.LlamaTokenizer.create_token_type_ids_from_sequences.example",$$slots:{default:[_a]},$$scope:{ctx:w}}}),We=new C({props:{name:"save_vocabulary",anchor:"transformers.LlamaTokenizer.save_vocabulary",parameters:[{name:"save_directory",val:""},{name:"filename_prefix",val:": typing.Optional[str] = None"}],parametersDescription:[{anchor:"transformers.LlamaTokenizer.save_vocabulary.save_directory",description:`<strong>save_directory</strong> (<code>str</code>) &#x2014;
The directory in which to save the vocabulary.`,name:"save_directory"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/llama/tokenization_llama.py#L306",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>Paths to the files saved.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>Tuple(str)</code></p>
`}}),qe=new me({props:{title:"LlamaTokenizerFast",local:"transformers.LlamaTokenizerFast",headingTag:"h2"}}),Ue=new C({props:{name:"class transformers.LlamaTokenizerFast",anchor:"transformers.LlamaTokenizerFast",parameters:[{name:"vocab_file",val:" = None"},{name:"tokenizer_file",val:" = None"},{name:"clean_up_tokenization_spaces",val:" = False"},{name:"unk_token",val:" = '<unk>'"},{name:"bos_token",val:" = '<s>'"},{name:"eos_token",val:" = '</s>'"},{name:"add_bos_token",val:" = True"},{name:"add_eos_token",val:" = False"},{name:"use_default_system_prompt",val:" = False"},{name:"legacy",val:" = None"},{name:"add_prefix_space",val:" = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.LlamaTokenizerFast.vocab_file",description:`<strong>vocab_file</strong> (<code>str</code>, <em>optional</em>) &#x2014;
<a href="https://github.com/google/sentencepiece" rel="nofollow">SentencePiece</a> file (generally has a .model extension) that
contains the vocabulary necessary to instantiate a tokenizer.`,name:"vocab_file"},{anchor:"transformers.LlamaTokenizerFast.tokenizer_file",description:`<strong>tokenizer_file</strong> (<code>str</code>, <em>optional</em>) &#x2014;
<a href="https://github.com/huggingface/tokenizers" rel="nofollow">tokenizers</a> file (generally has a .json extension) that
contains everything needed to load the tokenizer.`,name:"tokenizer_file"},{anchor:"transformers.LlamaTokenizerFast.clean_up_tokenization_spaces",description:`<strong>clean_up_tokenization_spaces</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to cleanup spaces after decoding, cleanup consists in removing potential artifacts like
extra spaces.`,name:"clean_up_tokenization_spaces"},{anchor:"transformers.LlamaTokenizerFast.unk_token",description:`<strong>unk_token</strong> (<code>str</code> or <code>tokenizers.AddedToken</code>, <em>optional</em>, defaults to <code>&quot;&lt;unk&gt;&quot;</code>) &#x2014;
The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
token instead.`,name:"unk_token"},{anchor:"transformers.LlamaTokenizerFast.bos_token",description:`<strong>bos_token</strong> (<code>str</code> or <code>tokenizers.AddedToken</code>, <em>optional</em>, defaults to <code>&quot;&lt;s&gt;&quot;</code>) &#x2014;
The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.`,name:"bos_token"},{anchor:"transformers.LlamaTokenizerFast.eos_token",description:`<strong>eos_token</strong> (<code>str</code> or <code>tokenizers.AddedToken</code>, <em>optional</em>, defaults to <code>&quot;&lt;/s&gt;&quot;</code>) &#x2014;
The end of sequence token.`,name:"eos_token"},{anchor:"transformers.LlamaTokenizerFast.add_bos_token",description:`<strong>add_bos_token</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to add an <code>bos_token</code> at the start of sequences.`,name:"add_bos_token"},{anchor:"transformers.LlamaTokenizerFast.add_eos_token",description:`<strong>add_eos_token</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to add an <code>eos_token</code> at the end of sequences.`,name:"add_eos_token"},{anchor:"transformers.LlamaTokenizerFast.use_default_system_prompt",description:`<strong>use_default_system_prompt</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not the default system prompt for Llama should be used`,name:"use_default_system_prompt"},{anchor:"transformers.LlamaTokenizerFast.legacy",description:`<strong>legacy</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not the <code>legacy</code> behavior of the tokenizer should be used. Legacy is before the merge of #24622
and #25224 which includes fixes to properly handle tokens that appear after special tokens.
Make sure to also set <code>from_slow</code> to <code>True</code>.
A simple example:</p>
<ul>
<li><code>legacy=True</code>:</li>
</ul>`,name:"legacy"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/llama/tokenization_llama_fast.py#L46"}}),oe=new yn({props:{anchor:"transformers.LlamaTokenizerFast.example",$$slots:{default:[ba]},$$scope:{ctx:w}}}),Be=new C({props:{name:"build_inputs_with_special_tokens",anchor:"transformers.LlamaTokenizerFast.build_inputs_with_special_tokens",parameters:[{name:"token_ids_0",val:""},{name:"token_ids_1",val:" = None"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/llama/tokenization_llama_fast.py#L239"}}),Ze=new C({props:{name:"get_special_tokens_mask",anchor:"transformers.LlamaTokenizerFast.get_special_tokens_mask",parameters:[{name:"token_ids_0",val:": list"},{name:"token_ids_1",val:": typing.Optional[list[int]] = None"},{name:"already_has_special_tokens",val:": bool = False"}],parametersDescription:[{anchor:"transformers.LlamaTokenizerFast.get_special_tokens_mask.token_ids_0",description:`<strong>token_ids_0</strong> (<code>list[int]</code>) &#x2014;
List of ids of the first sequence.`,name:"token_ids_0"},{anchor:"transformers.LlamaTokenizerFast.get_special_tokens_mask.token_ids_1",description:`<strong>token_ids_1</strong> (<code>list[int]</code>, <em>optional</em>) &#x2014;
List of ids of the second sequence.`,name:"token_ids_1"},{anchor:"transformers.LlamaTokenizerFast.get_special_tokens_mask.already_has_special_tokens",description:`<strong>already_has_special_tokens</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not the token list is already formatted with special tokens for the model.`,name:"already_has_special_tokens"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L3913",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>1 for a special token, 0 for a sequence token.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A list of integers in the range [0, 1]</p>
`}}),Ge=new C({props:{name:"create_token_type_ids_from_sequences",anchor:"transformers.LlamaTokenizerFast.create_token_type_ids_from_sequences",parameters:[{name:"token_ids_0",val:": list"},{name:"token_ids_1",val:": typing.Optional[list[int]] = None"}],parametersDescription:[{anchor:"transformers.LlamaTokenizerFast.create_token_type_ids_from_sequences.token_ids_0",description:"<strong>token_ids_0</strong> (<code>list[int]</code>) &#x2014; The first tokenized sequence.",name:"token_ids_0"},{anchor:"transformers.LlamaTokenizerFast.create_token_type_ids_from_sequences.token_ids_1",description:"<strong>token_ids_1</strong> (<code>list[int]</code>, <em>optional</em>) &#x2014; The second tokenized sequence.",name:"token_ids_1"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L3432",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>The token type ids.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>list[int]</code></p>
`}}),Pe=new C({props:{name:"update_post_processor",anchor:"transformers.LlamaTokenizerFast.update_post_processor",parameters:[],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/llama/tokenization_llama_fast.py#L174"}}),He=new C({props:{name:"save_vocabulary",anchor:"transformers.LlamaTokenizerFast.save_vocabulary",parameters:[{name:"save_directory",val:": str"},{name:"filename_prefix",val:": typing.Optional[str] = None"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/llama/tokenization_llama_fast.py#L218"}}),Re=new me({props:{title:"LlamaModel",local:"transformers.LlamaModel",headingTag:"h2"}}),Ve=new C({props:{name:"class transformers.LlamaModel",anchor:"transformers.LlamaModel",parameters:[{name:"config",val:": LlamaConfig"}],parametersDescription:[{anchor:"transformers.LlamaModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaConfig">LlamaConfig</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/llama/modeling_llama.py#L334"}}),Ee=new C({props:{name:"forward",anchor:"transformers.LlamaModel.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.LlamaModel.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.LlamaModel.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.LlamaModel.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.LlamaModel.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.LlamaModel.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.LlamaModel.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"},{anchor:"transformers.LlamaModel.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/llama/modeling_llama.py#L351",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast"
>transformers.modeling_outputs.BaseModelOutputWithPast</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaConfig"
>LlamaConfig</a>) and inputs.</p>
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
`}}),re=new kn({props:{$$slots:{default:[ka]},$$scope:{ctx:w}}}),Xe=new me({props:{title:"LlamaForCausalLM",local:"transformers.LlamaForCausalLM",headingTag:"h2"}}),Ne=new C({props:{name:"class transformers.LlamaForCausalLM",anchor:"transformers.LlamaForCausalLM",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.LlamaForCausalLM.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaForCausalLM">LlamaForCausalLM</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/llama/modeling_llama.py#L413"}}),Ye=new C({props:{name:"forward",anchor:"transformers.LlamaForCausalLM.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"},{name:"logits_to_keep",val:": typing.Union[int, torch.Tensor] = 0"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.LlamaForCausalLM.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.LlamaForCausalLM.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.LlamaForCausalLM.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.LlamaForCausalLM.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.LlamaForCausalLM.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.LlamaForCausalLM.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.LlamaForCausalLM.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.LlamaForCausalLM.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"},{anchor:"transformers.LlamaForCausalLM.forward.logits_to_keep",description:`<strong>logits_to_keep</strong> (<code>Union[int, torch.Tensor]</code>, defaults to <code>0</code>) &#x2014;
If an <code>int</code>, compute logits for the last <code>logits_to_keep</code> tokens. If <code>0</code>, calculate logits for all
<code>input_ids</code> (special case). Only last token logits are needed for generation, and calculating them only for that
token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
If a <code>torch.Tensor</code>, must be 1D corresponding to the indices to keep in the sequence length dimension.
This is useful when using packed tensor format (single dimension for batch and sequence length).`,name:"logits_to_keep"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/llama/modeling_llama.py#L427",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast"
>transformers.modeling_outputs.CausalLMOutputWithPast</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaConfig"
>LlamaConfig</a>) and inputs.</p>
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
`}}),ie=new kn({props:{$$slots:{default:[ya]},$$scope:{ctx:w}}}),le=new yn({props:{anchor:"transformers.LlamaForCausalLM.forward.example",$$slots:{default:[va]},$$scope:{ctx:w}}}),Ae=new me({props:{title:"LlamaForSequenceClassification",local:"transformers.LlamaForSequenceClassification",headingTag:"h2"}}),De=new C({props:{name:"class transformers.LlamaForSequenceClassification",anchor:"transformers.LlamaForSequenceClassification",parameters:[{name:"config",val:""}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/llama/modeling_llama.py#L488"}}),Se=new C({props:{name:"forward",anchor:"transformers.LlamaForSequenceClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.LlamaForSequenceClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.LlamaForSequenceClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.LlamaForSequenceClassification.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.LlamaForSequenceClassification.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.LlamaForSequenceClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.LlamaForSequenceClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.LlamaForSequenceClassification.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
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
`}}),de=new kn({props:{$$slots:{default:[Ta]},$$scope:{ctx:w}}}),Qe=new da({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/llama2.md"}}),{c(){t=l("meta"),m=r(),n=l("p"),k=r(),M=l("p"),M.innerHTML=v,y=r(),x=l("div"),x.innerHTML=Bt,pe=r(),p(V.$$.fragment),Pt=r(),he=l("p"),he.innerHTML=bo,Ht=r(),ue=l("p"),ue.textContent=ko,Rt=r(),fe=l("p"),fe.innerHTML=yo,Vt=r(),p(S.$$.fragment),Et=r(),ge=l("p"),ge.innerHTML=vo,Xt=r(),p(Q.$$.fragment),Nt=r(),_e=l("p"),_e.innerHTML=To,Yt=r(),be=l("p"),be.innerHTML=wo,At=r(),p(ke.$$.fragment),Dt=r(),ye=l("p"),ye.innerHTML=Mo,St=r(),p(ve.$$.fragment),Qt=r(),O=l("div"),O.innerHTML=$o,Ot=r(),p(Te.$$.fragment),Kt=r(),W=l("ul"),nt=l("li"),nt.innerHTML=Lo,vn=r(),we=l("li"),ot=l("p"),ot.innerHTML=xo,Tn=r(),p(Me.$$.fragment),wn=r(),$e=l("li"),at=l("p"),at.innerHTML=zo,Mn=r(),p(Le.$$.fragment),$n=r(),st=l("li"),st.innerHTML=Co,Ln=r(),rt=l("li"),rt.innerHTML=jo,en=r(),p(xe.$$.fragment),tn=r(),q=l("div"),p(ze.$$.fragment),xn=r(),it=l("p"),it.innerHTML=Jo,zn=r(),lt=l("p"),lt.innerHTML=Fo,Cn=r(),p(K.$$.fragment),nn=r(),p(Ce.$$.fragment),on=r(),j=l("div"),p(je.$$.fragment),jn=r(),dt=l("p"),dt.textContent=Io,Jn=r(),ct=l("div"),p(Je.$$.fragment),Fn=r(),ee=l("div"),p(Fe.$$.fragment),In=r(),mt=l("p"),mt.innerHTML=Wo,Wn=r(),B=l("div"),p(Ie.$$.fragment),qn=r(),pt=l("p"),pt.textContent=qo,Un=r(),p(te.$$.fragment),Bn=r(),ht=l("p"),ht.textContent=Uo,Zn=r(),ne=l("div"),p(We.$$.fragment),Gn=r(),ut=l("p"),ut.textContent=Bo,an=r(),p(qe.$$.fragment),sn=r(),T=l("div"),p(Ue.$$.fragment),Pn=r(),ft=l("p"),ft.textContent=Zo,Hn=r(),gt=l("p"),gt.textContent=Go,Rn=r(),p(oe.$$.fragment),Vn=r(),_t=l("p"),_t.innerHTML=Po,En=r(),bt=l("p"),bt.innerHTML=Ho,Xn=r(),kt=l("div"),p(Be.$$.fragment),Nn=r(),ae=l("div"),p(Ze.$$.fragment),Yn=r(),yt=l("p"),yt.innerHTML=Ro,An=r(),E=l("div"),p(Ge.$$.fragment),Dn=r(),vt=l("p"),vt.innerHTML=Vo,Sn=r(),Tt=l("p"),Tt.textContent=Eo,Qn=r(),se=l("div"),p(Pe.$$.fragment),On=r(),wt=l("p"),wt.innerHTML=Xo,Kn=r(),Mt=l("div"),p(He.$$.fragment),rn=r(),p(Re.$$.fragment),ln=r(),J=l("div"),p(Ve.$$.fragment),eo=r(),$t=l("p"),$t.textContent=No,to=r(),Lt=l("p"),Lt.innerHTML=Yo,no=r(),xt=l("p"),xt.innerHTML=Ao,oo=r(),X=l("div"),p(Ee.$$.fragment),ao=r(),zt=l("p"),zt.innerHTML=Do,so=r(),p(re.$$.fragment),dn=r(),p(Xe.$$.fragment),cn=r(),F=l("div"),p(Ne.$$.fragment),ro=r(),Ct=l("p"),Ct.textContent=So,io=r(),jt=l("p"),jt.innerHTML=Qo,lo=r(),Jt=l("p"),Jt.innerHTML=Oo,co=r(),Z=l("div"),p(Ye.$$.fragment),mo=r(),Ft=l("p"),Ft.innerHTML=Ko,po=r(),p(ie.$$.fragment),ho=r(),p(le.$$.fragment),mn=r(),p(Ae.$$.fragment),pn=r(),D=l("div"),p(De.$$.fragment),uo=r(),N=l("div"),p(Se.$$.fragment),fo=r(),It=l("p"),It.innerHTML=ea,go=r(),p(de.$$.fragment),hn=r(),p(Qe.$$.fragment),un=r(),Zt=l("p"),this.h()},l(e){const a=ia("svelte-u9bgzb",document.head);t=d(a,"META",{name:!0,content:!0}),a.forEach(s),m=i(e),n=d(e,"P",{}),$(n).forEach(s),k=i(e),M=d(e,"P",{"data-svelte-h":!0}),b(M)!=="svelte-dx7og"&&(M.innerHTML=v),y=i(e),x=d(e,"DIV",{style:!0,"data-svelte-h":!0}),b(x)!=="svelte-gspis1"&&(x.innerHTML=Bt),pe=i(e),h(V.$$.fragment,e),Pt=i(e),he=d(e,"P",{"data-svelte-h":!0}),b(he)!=="svelte-fsxopc"&&(he.innerHTML=bo),Ht=i(e),ue=d(e,"P",{"data-svelte-h":!0}),b(ue)!=="svelte-8adcds"&&(ue.textContent=ko),Rt=i(e),fe=d(e,"P",{"data-svelte-h":!0}),b(fe)!=="svelte-1syr1he"&&(fe.innerHTML=yo),Vt=i(e),h(S.$$.fragment,e),Et=i(e),ge=d(e,"P",{"data-svelte-h":!0}),b(ge)!=="svelte-s57n5g"&&(ge.innerHTML=vo),Xt=i(e),h(Q.$$.fragment,e),Nt=i(e),_e=d(e,"P",{"data-svelte-h":!0}),b(_e)!=="svelte-nf5ooi"&&(_e.innerHTML=To),Yt=i(e),be=d(e,"P",{"data-svelte-h":!0}),b(be)!=="svelte-w36i1c"&&(be.innerHTML=wo),At=i(e),h(ke.$$.fragment,e),Dt=i(e),ye=d(e,"P",{"data-svelte-h":!0}),b(ye)!=="svelte-w3z5ks"&&(ye.innerHTML=Mo),St=i(e),h(ve.$$.fragment,e),Qt=i(e),O=d(e,"DIV",{class:!0,"data-svelte-h":!0}),b(O)!=="svelte-zernbs"&&(O.innerHTML=$o),Ot=i(e),h(Te.$$.fragment,e),Kt=i(e),W=d(e,"UL",{});var U=$(W);nt=d(U,"LI",{"data-svelte-h":!0}),b(nt)!=="svelte-22xnzg"&&(nt.innerHTML=Lo),vn=i(U),we=d(U,"LI",{});var Oe=$(we);ot=d(Oe,"P",{"data-svelte-h":!0}),b(ot)!=="svelte-r4616o"&&(ot.innerHTML=xo),Tn=i(Oe),h(Me.$$.fragment,Oe),Oe.forEach(s),wn=i(U),$e=d(U,"LI",{});var Ke=$($e);at=d(Ke,"P",{"data-svelte-h":!0}),b(at)!=="svelte-1yui325"&&(at.innerHTML=zo),Mn=i(Ke),h(Le.$$.fragment,Ke),Ke.forEach(s),$n=i(U),st=d(U,"LI",{"data-svelte-h":!0}),b(st)!=="svelte-4gdjt6"&&(st.innerHTML=Co),Ln=i(U),rt=d(U,"LI",{"data-svelte-h":!0}),b(rt)!=="svelte-1u7c7nn"&&(rt.innerHTML=jo),U.forEach(s),en=i(e),h(xe.$$.fragment,e),tn=i(e),q=d(e,"DIV",{class:!0});var G=$(q);h(ze.$$.fragment,G),xn=i(G),it=d(G,"P",{"data-svelte-h":!0}),b(it)!=="svelte-7fhu95"&&(it.innerHTML=Jo),zn=i(G),lt=d(G,"P",{"data-svelte-h":!0}),b(lt)!=="svelte-1ek1ss9"&&(lt.innerHTML=Fo),Cn=i(G),h(K.$$.fragment,G),G.forEach(s),nn=i(e),h(Ce.$$.fragment,e),on=i(e),j=d(e,"DIV",{class:!0});var I=$(j);h(je.$$.fragment,I),jn=i(I),dt=d(I,"P",{"data-svelte-h":!0}),b(dt)!=="svelte-qfiu5a"&&(dt.textContent=Io),Jn=i(I),ct=d(I,"DIV",{class:!0});var Gt=$(ct);h(Je.$$.fragment,Gt),Gt.forEach(s),Fn=i(I),ee=d(I,"DIV",{class:!0});var et=$(ee);h(Fe.$$.fragment,et),In=i(et),mt=d(et,"P",{"data-svelte-h":!0}),b(mt)!=="svelte-1f4f5kp"&&(mt.innerHTML=Wo),et.forEach(s),Wn=i(I),B=d(I,"DIV",{class:!0});var P=$(B);h(Ie.$$.fragment,P),qn=i(P),pt=d(P,"P",{"data-svelte-h":!0}),b(pt)!=="svelte-13bfd60"&&(pt.textContent=qo),Un=i(P),h(te.$$.fragment,P),Bn=i(P),ht=d(P,"P",{"data-svelte-h":!0}),b(ht)!=="svelte-wtrslu"&&(ht.textContent=Uo),P.forEach(s),Zn=i(I),ne=d(I,"DIV",{class:!0});var tt=$(ne);h(We.$$.fragment,tt),Gn=i(tt),ut=d(tt,"P",{"data-svelte-h":!0}),b(ut)!=="svelte-1slb66l"&&(ut.textContent=Bo),tt.forEach(s),I.forEach(s),an=i(e),h(qe.$$.fragment,e),sn=i(e),T=d(e,"DIV",{class:!0});var L=$(T);h(Ue.$$.fragment,L),Pn=i(L),ft=d(L,"P",{"data-svelte-h":!0}),b(ft)!=="svelte-15tdcz8"&&(ft.textContent=Zo),Hn=i(L),gt=d(L,"P",{"data-svelte-h":!0}),b(gt)!=="svelte-llhmpa"&&(gt.textContent=Go),Rn=i(L),h(oe.$$.fragment,L),Vn=i(L),_t=d(L,"P",{"data-svelte-h":!0}),b(_t)!=="svelte-cnb6q1"&&(_t.innerHTML=Po),En=i(L),bt=d(L,"P",{"data-svelte-h":!0}),b(bt)!=="svelte-gxzj9w"&&(bt.innerHTML=Ho),Xn=i(L),kt=d(L,"DIV",{class:!0});var ta=$(kt);h(Be.$$.fragment,ta),ta.forEach(s),Nn=i(L),ae=d(L,"DIV",{class:!0});var gn=$(ae);h(Ze.$$.fragment,gn),Yn=i(gn),yt=d(gn,"P",{"data-svelte-h":!0}),b(yt)!=="svelte-1wmjg8a"&&(yt.innerHTML=Ro),gn.forEach(s),An=i(L),E=d(L,"DIV",{class:!0});var Wt=$(E);h(Ge.$$.fragment,Wt),Dn=i(Wt),vt=d(Wt,"P",{"data-svelte-h":!0}),b(vt)!=="svelte-zj1vf1"&&(vt.innerHTML=Vo),Sn=i(Wt),Tt=d(Wt,"P",{"data-svelte-h":!0}),b(Tt)!=="svelte-9vptpw"&&(Tt.textContent=Eo),Wt.forEach(s),Qn=i(L),se=d(L,"DIV",{class:!0});var _n=$(se);h(Pe.$$.fragment,_n),On=i(_n),wt=d(_n,"P",{"data-svelte-h":!0}),b(wt)!=="svelte-nfci2w"&&(wt.innerHTML=Xo),_n.forEach(s),Kn=i(L),Mt=d(L,"DIV",{class:!0});var na=$(Mt);h(He.$$.fragment,na),na.forEach(s),L.forEach(s),rn=i(e),h(Re.$$.fragment,e),ln=i(e),J=d(e,"DIV",{class:!0});var Y=$(J);h(Ve.$$.fragment,Y),eo=i(Y),$t=d(Y,"P",{"data-svelte-h":!0}),b($t)!=="svelte-ahmvbp"&&($t.textContent=No),to=i(Y),Lt=d(Y,"P",{"data-svelte-h":!0}),b(Lt)!=="svelte-q52n56"&&(Lt.innerHTML=Yo),no=i(Y),xt=d(Y,"P",{"data-svelte-h":!0}),b(xt)!=="svelte-hswkmf"&&(xt.innerHTML=Ao),oo=i(Y),X=d(Y,"DIV",{class:!0});var qt=$(X);h(Ee.$$.fragment,qt),ao=i(qt),zt=d(qt,"P",{"data-svelte-h":!0}),b(zt)!=="svelte-1wrnj28"&&(zt.innerHTML=Do),so=i(qt),h(re.$$.fragment,qt),qt.forEach(s),Y.forEach(s),dn=i(e),h(Xe.$$.fragment,e),cn=i(e),F=d(e,"DIV",{class:!0});var A=$(F);h(Ne.$$.fragment,A),ro=i(A),Ct=d(A,"P",{"data-svelte-h":!0}),b(Ct)!=="svelte-a2k4ga"&&(Ct.textContent=So),io=i(A),jt=d(A,"P",{"data-svelte-h":!0}),b(jt)!=="svelte-q52n56"&&(jt.innerHTML=Qo),lo=i(A),Jt=d(A,"P",{"data-svelte-h":!0}),b(Jt)!=="svelte-hswkmf"&&(Jt.innerHTML=Oo),co=i(A),Z=d(A,"DIV",{class:!0});var ce=$(Z);h(Ye.$$.fragment,ce),mo=i(ce),Ft=d(ce,"P",{"data-svelte-h":!0}),b(Ft)!=="svelte-1p7qkf4"&&(Ft.innerHTML=Ko),po=i(ce),h(ie.$$.fragment,ce),ho=i(ce),h(le.$$.fragment,ce),ce.forEach(s),A.forEach(s),mn=i(e),h(Ae.$$.fragment,e),pn=i(e),D=d(e,"DIV",{class:!0});var bn=$(D);h(De.$$.fragment,bn),uo=i(bn),N=d(bn,"DIV",{class:!0});var Ut=$(N);h(Se.$$.fragment,Ut),fo=i(Ut),It=d(Ut,"P",{"data-svelte-h":!0}),b(It)!=="svelte-1sal4ui"&&(It.innerHTML=ea),go=i(Ut),h(de.$$.fragment,Ut),Ut.forEach(s),bn.forEach(s),hn=i(e),h(Qe.$$.fragment,e),un=i(e),Zt=d(e,"P",{}),$(Zt).forEach(s),this.h()},h(){z(t,"name","hf:doc:metadata"),z(t,"content",Ma),la(x,"float","right"),z(O,"class","flex justify-center"),z(q,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),z(ct,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),z(ee,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),z(B,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),z(ne,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),z(j,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),z(kt,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),z(ae,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),z(E,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),z(se,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),z(Mt,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),z(T,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),z(X,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),z(J,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),z(Z,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),z(F,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),z(N,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),z(D,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,a){o(document.head,t),c(e,m,a),c(e,n,a),c(e,k,a),c(e,M,a),c(e,y,a),c(e,x,a),c(e,pe,a),u(V,e,a),c(e,Pt,a),c(e,he,a),c(e,Ht,a),c(e,ue,a),c(e,Rt,a),c(e,fe,a),c(e,Vt,a),u(S,e,a),c(e,Et,a),c(e,ge,a),c(e,Xt,a),u(Q,e,a),c(e,Nt,a),c(e,_e,a),c(e,Yt,a),c(e,be,a),c(e,At,a),u(ke,e,a),c(e,Dt,a),c(e,ye,a),c(e,St,a),u(ve,e,a),c(e,Qt,a),c(e,O,a),c(e,Ot,a),u(Te,e,a),c(e,Kt,a),c(e,W,a),o(W,nt),o(W,vn),o(W,we),o(we,ot),o(we,Tn),u(Me,we,null),o(W,wn),o(W,$e),o($e,at),o($e,Mn),u(Le,$e,null),o(W,$n),o(W,st),o(W,Ln),o(W,rt),c(e,en,a),u(xe,e,a),c(e,tn,a),c(e,q,a),u(ze,q,null),o(q,xn),o(q,it),o(q,zn),o(q,lt),o(q,Cn),u(K,q,null),c(e,nn,a),u(Ce,e,a),c(e,on,a),c(e,j,a),u(je,j,null),o(j,jn),o(j,dt),o(j,Jn),o(j,ct),u(Je,ct,null),o(j,Fn),o(j,ee),u(Fe,ee,null),o(ee,In),o(ee,mt),o(j,Wn),o(j,B),u(Ie,B,null),o(B,qn),o(B,pt),o(B,Un),u(te,B,null),o(B,Bn),o(B,ht),o(j,Zn),o(j,ne),u(We,ne,null),o(ne,Gn),o(ne,ut),c(e,an,a),u(qe,e,a),c(e,sn,a),c(e,T,a),u(Ue,T,null),o(T,Pn),o(T,ft),o(T,Hn),o(T,gt),o(T,Rn),u(oe,T,null),o(T,Vn),o(T,_t),o(T,En),o(T,bt),o(T,Xn),o(T,kt),u(Be,kt,null),o(T,Nn),o(T,ae),u(Ze,ae,null),o(ae,Yn),o(ae,yt),o(T,An),o(T,E),u(Ge,E,null),o(E,Dn),o(E,vt),o(E,Sn),o(E,Tt),o(T,Qn),o(T,se),u(Pe,se,null),o(se,On),o(se,wt),o(T,Kn),o(T,Mt),u(He,Mt,null),c(e,rn,a),u(Re,e,a),c(e,ln,a),c(e,J,a),u(Ve,J,null),o(J,eo),o(J,$t),o(J,to),o(J,Lt),o(J,no),o(J,xt),o(J,oo),o(J,X),u(Ee,X,null),o(X,ao),o(X,zt),o(X,so),u(re,X,null),c(e,dn,a),u(Xe,e,a),c(e,cn,a),c(e,F,a),u(Ne,F,null),o(F,ro),o(F,Ct),o(F,io),o(F,jt),o(F,lo),o(F,Jt),o(F,co),o(F,Z),u(Ye,Z,null),o(Z,mo),o(Z,Ft),o(Z,po),u(ie,Z,null),o(Z,ho),u(le,Z,null),c(e,mn,a),u(Ae,e,a),c(e,pn,a),c(e,D,a),u(De,D,null),o(D,uo),o(D,N),u(Se,N,null),o(N,fo),o(N,It),o(N,go),u(de,N,null),c(e,hn,a),u(Qe,e,a),c(e,un,a),c(e,Zt,a),fn=!0},p(e,[a]){const U={};a&2&&(U.$$scope={dirty:a,ctx:e}),S.$set(U);const Oe={};a&2&&(Oe.$$scope={dirty:a,ctx:e}),Q.$set(Oe);const Ke={};a&2&&(Ke.$$scope={dirty:a,ctx:e}),K.$set(Ke);const G={};a&2&&(G.$$scope={dirty:a,ctx:e}),te.$set(G);const I={};a&2&&(I.$$scope={dirty:a,ctx:e}),oe.$set(I);const Gt={};a&2&&(Gt.$$scope={dirty:a,ctx:e}),re.$set(Gt);const et={};a&2&&(et.$$scope={dirty:a,ctx:e}),ie.$set(et);const P={};a&2&&(P.$$scope={dirty:a,ctx:e}),le.$set(P);const tt={};a&2&&(tt.$$scope={dirty:a,ctx:e}),de.$set(tt)},i(e){fn||(f(V.$$.fragment,e),f(S.$$.fragment,e),f(Q.$$.fragment,e),f(ke.$$.fragment,e),f(ve.$$.fragment,e),f(Te.$$.fragment,e),f(Me.$$.fragment,e),f(Le.$$.fragment,e),f(xe.$$.fragment,e),f(ze.$$.fragment,e),f(K.$$.fragment,e),f(Ce.$$.fragment,e),f(je.$$.fragment,e),f(Je.$$.fragment,e),f(Fe.$$.fragment,e),f(Ie.$$.fragment,e),f(te.$$.fragment,e),f(We.$$.fragment,e),f(qe.$$.fragment,e),f(Ue.$$.fragment,e),f(oe.$$.fragment,e),f(Be.$$.fragment,e),f(Ze.$$.fragment,e),f(Ge.$$.fragment,e),f(Pe.$$.fragment,e),f(He.$$.fragment,e),f(Re.$$.fragment,e),f(Ve.$$.fragment,e),f(Ee.$$.fragment,e),f(re.$$.fragment,e),f(Xe.$$.fragment,e),f(Ne.$$.fragment,e),f(Ye.$$.fragment,e),f(ie.$$.fragment,e),f(le.$$.fragment,e),f(Ae.$$.fragment,e),f(De.$$.fragment,e),f(Se.$$.fragment,e),f(de.$$.fragment,e),f(Qe.$$.fragment,e),fn=!0)},o(e){g(V.$$.fragment,e),g(S.$$.fragment,e),g(Q.$$.fragment,e),g(ke.$$.fragment,e),g(ve.$$.fragment,e),g(Te.$$.fragment,e),g(Me.$$.fragment,e),g(Le.$$.fragment,e),g(xe.$$.fragment,e),g(ze.$$.fragment,e),g(K.$$.fragment,e),g(Ce.$$.fragment,e),g(je.$$.fragment,e),g(Je.$$.fragment,e),g(Fe.$$.fragment,e),g(Ie.$$.fragment,e),g(te.$$.fragment,e),g(We.$$.fragment,e),g(qe.$$.fragment,e),g(Ue.$$.fragment,e),g(oe.$$.fragment,e),g(Be.$$.fragment,e),g(Ze.$$.fragment,e),g(Ge.$$.fragment,e),g(Pe.$$.fragment,e),g(He.$$.fragment,e),g(Re.$$.fragment,e),g(Ve.$$.fragment,e),g(Ee.$$.fragment,e),g(re.$$.fragment,e),g(Xe.$$.fragment,e),g(Ne.$$.fragment,e),g(Ye.$$.fragment,e),g(ie.$$.fragment,e),g(le.$$.fragment,e),g(Ae.$$.fragment,e),g(De.$$.fragment,e),g(Se.$$.fragment,e),g(de.$$.fragment,e),g(Qe.$$.fragment,e),fn=!1},d(e){e&&(s(m),s(n),s(k),s(M),s(y),s(x),s(pe),s(Pt),s(he),s(Ht),s(ue),s(Rt),s(fe),s(Vt),s(Et),s(ge),s(Xt),s(Nt),s(_e),s(Yt),s(be),s(At),s(Dt),s(ye),s(St),s(Qt),s(O),s(Ot),s(Kt),s(W),s(en),s(tn),s(q),s(nn),s(on),s(j),s(an),s(sn),s(T),s(rn),s(ln),s(J),s(dn),s(cn),s(F),s(mn),s(pn),s(D),s(hn),s(un),s(Zt)),s(t),_(V,e),_(S,e),_(Q,e),_(ke,e),_(ve,e),_(Te,e),_(Me),_(Le),_(xe,e),_(ze),_(K),_(Ce,e),_(je),_(Je),_(Fe),_(Ie),_(te),_(We),_(qe,e),_(Ue),_(oe),_(Be),_(Ze),_(Ge),_(Pe),_(He),_(Re,e),_(Ve),_(Ee),_(re),_(Xe,e),_(Ne),_(Ye),_(ie),_(le),_(Ae,e),_(De),_(Se),_(de),_(Qe,e)}}}const Ma='{"title":"Llama 2","local":"llama-2","sections":[{"title":"Notes","local":"notes","sections":[],"depth":2},{"title":"LlamaConfig","local":"transformers.LlamaConfig","sections":[],"depth":2},{"title":"LlamaTokenizer","local":"transformers.LlamaTokenizer","sections":[],"depth":2},{"title":"LlamaTokenizerFast","local":"transformers.LlamaTokenizerFast","sections":[],"depth":2},{"title":"LlamaModel","local":"transformers.LlamaModel","sections":[],"depth":2},{"title":"LlamaForCausalLM","local":"transformers.LlamaForCausalLM","sections":[],"depth":2},{"title":"LlamaForSequenceClassification","local":"transformers.LlamaForSequenceClassification","sections":[],"depth":2}],"depth":1}';function $a(w){return aa(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class Wa extends sa{constructor(t){super(),ra(this,t,$a,wa,oa,{})}}export{Wa as component};
