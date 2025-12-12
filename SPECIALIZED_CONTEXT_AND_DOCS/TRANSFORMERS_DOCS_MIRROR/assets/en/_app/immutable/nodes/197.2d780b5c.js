import{s as on,o as sn,n as Z}from"../chunks/scheduler.18a86fab.js";import{S as an,i as rn,g as d,s as r,r as h,A as ln,h as c,f as s,c as i,j as U,x as v,u as f,k as J,l as dn,y as p,a,v as g,d as _,t as b,w as y}from"../chunks/index.98837b22.js";import{T as Xe}from"../chunks/Tip.77304350.js";import{D as E}from"../chunks/Docstring.a1ef7999.js";import{C as Ge}from"../chunks/CodeBlock.8d0c2e8a.js";import{E as nn}from"../chunks/ExampleCodeBlock.8c3ee1f9.js";import{H as Be,E as cn}from"../chunks/getInferenceSnippets.06c2775f.js";import{H as mn,a as Wt}from"../chunks/HfOption.6641485e.js";function pn(T){let t,l="Click on the Gemma 2 models in the right sidebar for more examples of how to apply Gemma to different language tasks.";return{c(){t=d("p"),t.textContent=l},l(n){t=c(n,"P",{"data-svelte-h":!0}),v(t)!=="svelte-qk0k3z"&&(t.textContent=l)},m(n,u){a(n,t,u)},p:Z,d(n){n&&s(t)}}}function un(T){let t,l;return t=new Ge({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwcGlwZWxpbmUlMEElMEFwaXBlJTIwJTNEJTIwcGlwZWxpbmUoJTBBJTIwJTIwJTIwJTIwdGFzayUzRCUyMnRleHQtZ2VuZXJhdGlvbiUyMiUyQyUwQSUyMCUyMCUyMCUyMG1vZGVsJTNEJTIyZ29vZ2xlJTJGZ2VtbWEtMi05YiUyMiUyQyUwQSUyMCUyMCUyMCUyMGR0eXBlJTNEdG9yY2guYmZsb2F0MTYlMkMlMEElMjAlMjAlMjAlMjBkZXZpY2VfbWFwJTNEJTIyYXV0byUyMiUyQyUwQSklMEElMEFwaXBlKCUyMkV4cGxhaW4lMjBxdWFudHVtJTIwY29tcHV0aW5nJTIwc2ltcGx5LiUyMCUyMiUyQyUyMG1heF9uZXdfdG9rZW5zJTNENTAp",highlighted:`<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> pipeline

pipe = pipeline(
    task=<span class="hljs-string">&quot;text-generation&quot;</span>,
    model=<span class="hljs-string">&quot;google/gemma-2-9b&quot;</span>,
    dtype=torch.bfloat16,
    device_map=<span class="hljs-string">&quot;auto&quot;</span>,
)

pipe(<span class="hljs-string">&quot;Explain quantum computing simply. &quot;</span>, max_new_tokens=<span class="hljs-number">50</span>)`,wrap:!1}}),{c(){h(t.$$.fragment)},l(n){f(t.$$.fragment,n)},m(n,u){g(t,n,u),l=!0},p:Z,i(n){l||(_(t.$$.fragment,n),l=!0)},o(n){b(t.$$.fragment,n),l=!1},d(n){y(t,n)}}}function hn(T){let t,l;return t=new Ge({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Rva2VuaXplciUyQyUyMEF1dG9Nb2RlbEZvckNhdXNhbExNJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIyZ29vZ2xlJTJGZ2VtbWEtMi05YiUyMiklMEFtb2RlbCUyMCUzRCUyMEF1dG9Nb2RlbEZvckNhdXNhbExNLmZyb21fcHJldHJhaW5lZCglMEElMjAlMjAlMjAlMjAlMjJnb29nbGUlMkZnZW1tYS0yLTliJTIyJTJDJTBBJTIwJTIwJTIwJTIwZHR5cGUlM0R0b3JjaC5iZmxvYXQxNiUyQyUwQSUyMCUyMCUyMCUyMGRldmljZV9tYXAlM0QlMjJhdXRvJTIyJTJDJTBBJTIwJTIwJTIwJTIwYXR0bl9pbXBsZW1lbnRhdGlvbiUzRCUyMnNkcGElMjIlMEEpJTBBJTBBaW5wdXRfdGV4dCUyMCUzRCUyMCUyMkV4cGxhaW4lMjBxdWFudHVtJTIwY29tcHV0aW5nJTIwc2ltcGx5LiUyMiUwQWlucHV0X2lkcyUyMCUzRCUyMHRva2VuaXplcihpbnB1dF90ZXh0JTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMikudG8obW9kZWwuZGV2aWNlKSUwQSUwQW91dHB1dHMlMjAlM0QlMjBtb2RlbC5nZW5lcmF0ZSgqKmlucHV0X2lkcyUyQyUyMG1heF9uZXdfdG9rZW5zJTNEMzIlMkMlMjBjYWNoZV9pbXBsZW1lbnRhdGlvbiUzRCUyMnN0YXRpYyUyMiklMEFwcmludCh0b2tlbml6ZXIuZGVjb2RlKG91dHB1dHMlNUIwJTVEJTJDJTIwc2tpcF9zcGVjaWFsX3Rva2VucyUzRFRydWUpKSUwQQ==",highlighted:`<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;google/gemma-2-9b&quot;</span>)
model = AutoModelForCausalLM.from_pretrained(
    <span class="hljs-string">&quot;google/gemma-2-9b&quot;</span>,
    dtype=torch.bfloat16,
    device_map=<span class="hljs-string">&quot;auto&quot;</span>,
    attn_implementation=<span class="hljs-string">&quot;sdpa&quot;</span>
)

input_text = <span class="hljs-string">&quot;Explain quantum computing simply.&quot;</span>
input_ids = tokenizer(input_text, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>).to(model.device)

outputs = model.generate(**input_ids, max_new_tokens=<span class="hljs-number">32</span>, cache_implementation=<span class="hljs-string">&quot;static&quot;</span>)
<span class="hljs-built_in">print</span>(tokenizer.decode(outputs[<span class="hljs-number">0</span>], skip_special_tokens=<span class="hljs-literal">True</span>))
`,wrap:!1}}),{c(){h(t.$$.fragment)},l(n){f(t.$$.fragment,n)},m(n,u){g(t,n,u),l=!0},p:Z,i(n){l||(_(t.$$.fragment,n),l=!0)},o(n){b(t.$$.fragment,n),l=!1},d(n){y(t,n)}}}function fn(T){let t,l;return t=new Ge({props:{code:"ZWNobyUyMC1lJTIwJTIyRXhwbGFpbiUyMHF1YW50dW0lMjBjb21wdXRpbmclMjBzaW1wbHkuJTIyJTIwJTdDJTIwdHJhbnNmb3JtZXJzJTIwcnVuJTIwLS10YXNrJTIwdGV4dC1nZW5lcmF0aW9uJTIwLS1tb2RlbCUyMGdvb2dsZSUyRmdlbW1hLTItMmIlMjAtLWRldmljZSUyMDA=",highlighted:'<span class="hljs-keyword">echo</span> -e <span class="hljs-string">&quot;Explain quantum computing simply.&quot;</span> | transformers run <span class="hljs-params">--task</span> text-generation <span class="hljs-params">--model</span> google/gemma-2-2b <span class="hljs-params">--device</span> 0',wrap:!1}}),{c(){h(t.$$.fragment)},l(n){f(t.$$.fragment,n)},m(n,u){g(t,n,u),l=!0},p:Z,i(n){l||(_(t.$$.fragment,n),l=!0)},o(n){b(t.$$.fragment,n),l=!1},d(n){y(t,n)}}}function gn(T){let t,l,n,u,k,w;return t=new Wt({props:{id:"usage",option:"Pipeline",$$slots:{default:[un]},$$scope:{ctx:T}}}),n=new Wt({props:{id:"usage",option:"AutoModel",$$slots:{default:[hn]},$$scope:{ctx:T}}}),k=new Wt({props:{id:"usage",option:"transformers CLI",$$slots:{default:[fn]},$$scope:{ctx:T}}}),{c(){h(t.$$.fragment),l=r(),h(n.$$.fragment),u=r(),h(k.$$.fragment)},l(m){f(t.$$.fragment,m),l=i(m),f(n.$$.fragment,m),u=i(m),f(k.$$.fragment,m)},m(m,M){g(t,m,M),a(m,l,M),g(n,m,M),a(m,u,M),g(k,m,M),w=!0},p(m,M){const He={};M&2&&(He.$$scope={dirty:M,ctx:m}),t.$set(He);const ee={};M&2&&(ee.$$scope={dirty:M,ctx:m}),n.$set(ee);const q={};M&2&&(q.$$scope={dirty:M,ctx:m}),k.$set(q)},i(m){w||(_(t.$$.fragment,m),_(n.$$.fragment,m),_(k.$$.fragment,m),w=!0)},o(m){b(t.$$.fragment,m),b(n.$$.fragment,m),b(k.$$.fragment,m),w=!1},d(m){m&&(s(l),s(u)),y(t,m),y(n,m),y(k,m)}}}function _n(T){let t,l;return t=new Ge({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEdlbW1hMk1vZGVsJTJDJTIwR2VtbWEyQ29uZmlnJTBBJTIzJTIwSW5pdGlhbGl6aW5nJTIwYSUyMEdlbW1hMiUyMGdlbW1hMi03YiUyMHN0eWxlJTIwY29uZmlndXJhdGlvbiUwQWNvbmZpZ3VyYXRpb24lMjAlM0QlMjBHZW1tYTJDb25maWcoKSUwQSUyMyUyMEluaXRpYWxpemluZyUyMGElMjBtb2RlbCUyMGZyb20lMjB0aGUlMjBnZW1tYTItN2IlMjBzdHlsZSUyMGNvbmZpZ3VyYXRpb24lMEFtb2RlbCUyMCUzRCUyMEdlbW1hMk1vZGVsKGNvbmZpZ3VyYXRpb24pJTBBJTIzJTIwQWNjZXNzaW5nJTIwdGhlJTIwbW9kZWwlMjBjb25maWd1cmF0aW9uJTBBY29uZmlndXJhdGlvbiUyMCUzRCUyMG1vZGVsLmNvbmZpZw==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> Gemma2Model, Gemma2Config
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a Gemma2 gemma2-7b style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = Gemma2Config()
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a model from the gemma2-7b style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = Gemma2Model(configuration)
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Accessing the model configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = model.config`,wrap:!1}}),{c(){h(t.$$.fragment)},l(n){f(t.$$.fragment,n)},m(n,u){g(t,n,u),l=!0},p:Z,i(n){l||(_(t.$$.fragment,n),l=!0)},o(n){b(t.$$.fragment,n),l=!1},d(n){y(t,n)}}}function bn(T){let t,l=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=d("p"),t.innerHTML=l},l(n){t=c(n,"P",{"data-svelte-h":!0}),v(t)!=="svelte-fincs2"&&(t.innerHTML=l)},m(n,u){a(n,t,u)},p:Z,d(n){n&&s(t)}}}function yn(T){let t,l=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=d("p"),t.innerHTML=l},l(n){t=c(n,"P",{"data-svelte-h":!0}),v(t)!=="svelte-fincs2"&&(t.innerHTML=l)},m(n,u){a(n,t,u)},p:Z,d(n){n&&s(t)}}}function vn(T){let t,l="Example:",n,u,k;return u=new Ge({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBHZW1tYTJGb3JDYXVzYWxMTSUwQSUwQW1vZGVsJTIwJTNEJTIwR2VtbWEyRm9yQ2F1c2FsTE0uZnJvbV9wcmV0cmFpbmVkKCUyMmdvb2dsZSUyRmdlbW1hLTItOWIlMjIpJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIyZ29vZ2xlJTJGZ2VtbWEtMi05YiUyMiklMEElMEFwcm9tcHQlMjAlM0QlMjAlMjJXaGF0JTIwaXMlMjB5b3VyJTIwZmF2b3JpdGUlMjBjb25kaW1lbnQlM0YlMjIlMEFpbnB1dHMlMjAlM0QlMjB0b2tlbml6ZXIocHJvbXB0JTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiklMEElMEElMjMlMjBHZW5lcmF0ZSUwQWdlbmVyYXRlX2lkcyUyMCUzRCUyMG1vZGVsLmdlbmVyYXRlKGlucHV0cy5pbnB1dF9pZHMlMkMlMjBtYXhfbGVuZ3RoJTNEMzApJTBBdG9rZW5pemVyLmJhdGNoX2RlY29kZShnZW5lcmF0ZV9pZHMlMkMlMjBza2lwX3NwZWNpYWxfdG9rZW5zJTNEVHJ1ZSUyQyUyMGNsZWFuX3VwX3Rva2VuaXphdGlvbl9zcGFjZXMlM0RGYWxzZSklNUIwJTVE",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, Gemma2ForCausalLM

<span class="hljs-meta">&gt;&gt;&gt; </span>model = Gemma2ForCausalLM.from_pretrained(<span class="hljs-string">&quot;google/gemma-2-9b&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;google/gemma-2-9b&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>prompt = <span class="hljs-string">&quot;What is your favorite condiment?&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(prompt, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Generate</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>generate_ids = model.generate(inputs.input_ids, max_length=<span class="hljs-number">30</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer.batch_decode(generate_ids, skip_special_tokens=<span class="hljs-literal">True</span>, clean_up_tokenization_spaces=<span class="hljs-literal">False</span>)[<span class="hljs-number">0</span>]
<span class="hljs-string">&quot;What is your favorite condiment?&quot;</span>`,wrap:!1}}),{c(){t=d("p"),t.textContent=l,n=r(),h(u.$$.fragment)},l(w){t=c(w,"P",{"data-svelte-h":!0}),v(t)!=="svelte-11lpom8"&&(t.textContent=l),n=i(w),f(u.$$.fragment,w)},m(w,m){a(w,t,m),a(w,n,m),g(u,w,m),k=!0},p:Z,i(w){k||(_(u.$$.fragment,w),k=!0)},o(w){b(u.$$.fragment,w),k=!1},d(w){w&&(s(t),s(n)),y(u,w)}}}function Tn(T){let t,l=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=d("p"),t.innerHTML=l},l(n){t=c(n,"P",{"data-svelte-h":!0}),v(t)!=="svelte-fincs2"&&(t.innerHTML=l)},m(n,u){a(n,t,u)},p:Z,d(n){n&&s(t)}}}function wn(T){let t,l=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=d("p"),t.innerHTML=l},l(n){t=c(n,"P",{"data-svelte-h":!0}),v(t)!=="svelte-fincs2"&&(t.innerHTML=l)},m(n,u){a(n,t,u)},p:Z,d(n){n&&s(t)}}}function kn(T){let t,l,n,u,k,w="<em>This model was released on 2024-07-31 and added to Hugging Face Transformers on 2024-06-27.</em>",m,M,He='<div class="flex flex-wrap space-x-1"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/> <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat"/> <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/> <img alt="Tensor parallelism" src="https://img.shields.io/badge/Tensor%20parallelism-06b6d4?style=flat&amp;logoColor=white"/></div>',ee,q,Ee,te,Lt='<a href="https://huggingface.co/papers/2408.00118" rel="nofollow">Gemma 2</a> is a family of language models with pretrained and instruction-tuned variants, available in 2B, 9B, 27B parameters. The architecture is similar to the previous Gemma, except it features interleaved local attention (4096 tokens) and global attention (8192 tokens) and grouped-query attention (GQA) to increase inference performance.',Re,ne,Zt="The 2B and 9B models are trained with knowledge distillation, and the instruction-tuned variant was post-trained with supervised fine-tuning and reinforcement learning.",Pe,oe,Vt='You can find all the original Gemma 2 checkpoints under the <a href="https://huggingface.co/collections/google/gemma-2-release-667d6600fd5220e7b967f315" rel="nofollow">Gemma 2</a> collection.',Se,R,Ae,se,Bt='The example below demonstrates how to chat with the model with <a href="/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.Pipeline">Pipeline</a> or the <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel">AutoModel</a> class, and from the command line.',Qe,P,De,ae,Ht='Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the <a href="../quantization/overview">Quantization</a> overview for more available quantization backends.',Oe,re,Nt='The example below uses <a href="../quantization/bitsandbytes">bitsandbytes</a> to only quantize the weights to int4.',Ye,ie,Ke,le,Xt='Use the <a href="https://github.com/huggingface/transformers/blob/beb9b5b02246b9b7ee81ddf938f93f44cfeaad19/src/transformers/utils/attention_visualizer.py#L139" rel="nofollow">AttentionMaskVisualizer</a> to better understand what tokens the model can and cannot attend to.',et,de,tt,S,Et='<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/gemma-2-attn-mask.png"/>',nt,ce,ot,F,me,ft,xe,Rt=`This is the configuration class to store the configuration of a <a href="/docs/transformers/v4.56.2/en/model_doc/gemma2#transformers.Gemma2Model">Gemma2Model</a>. It is used to instantiate an Gemma2
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the Gemma2-7B.
e.g. <a href="https://huggingface.co/google/gemma2-7b" rel="nofollow">google/gemma2-7b</a>
Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> and can be used to control the model outputs. Read the
documentation from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> for more information.`,gt,A,st,pe,at,$,ue,_t,ze,Pt="The bare Gemma2 Model outputting raw hidden-states without any specific head on top.",bt,Je,St=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,yt,Fe,At=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,vt,j,he,Tt,Ie,Qt='The <a href="/docs/transformers/v4.56.2/en/model_doc/gemma2#transformers.Gemma2Model">Gemma2Model</a> forward method, overrides the <code>__call__</code> special method.',wt,Q,rt,fe,it,C,ge,kt,Ue,Dt="The Gemma2 Model for causal language modeling.",Mt,qe,Ot=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,$t,je,Yt=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Ct,z,_e,Gt,We,Kt='The <a href="/docs/transformers/v4.56.2/en/model_doc/gemma2#transformers.Gemma2ForCausalLM">Gemma2ForCausalLM</a> forward method, overrides the <code>__call__</code> special method.',xt,D,zt,O,lt,be,dt,V,ye,Jt,W,ve,Ft,Le,en="The <code>GenericForSequenceClassification</code> forward method, overrides the <code>__call__</code> special method.",It,Y,ct,Te,mt,B,we,Ut,L,ke,qt,Ze,tn="The <code>GenericForTokenClassification</code> forward method, overrides the <code>__call__</code> special method.",jt,K,pt,Me,ut,Ne,ht;return q=new Be({props:{title:"Gemma2",local:"gemma2",headingTag:"h1"}}),R=new Xe({props:{warning:!1,$$slots:{default:[pn]},$$scope:{ctx:T}}}),P=new mn({props:{id:"usage",options:["Pipeline","AutoModel","transformers CLI"],$$slots:{default:[gn]},$$scope:{ctx:T}}}),ie=new Ge({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Rva2VuaXplciUyQyUyMEF1dG9Nb2RlbEZvckNhdXNhbExNJTJDJTIwQml0c0FuZEJ5dGVzQ29uZmlnJTBBJTBBcXVhbnRpemF0aW9uX2NvbmZpZyUyMCUzRCUyMEJpdHNBbmRCeXRlc0NvbmZpZyhsb2FkX2luXzRiaXQlM0RUcnVlKSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMmdvb2dsZSUyRmdlbW1hLTItMjdiJTIyKSUwQW1vZGVsJTIwJTNEJTIwQXV0b01vZGVsRm9yQ2F1c2FsTE0uZnJvbV9wcmV0cmFpbmVkKCUwQSUyMCUyMCUyMCUyMCUyMmdvb2dsZSUyRmdlbW1hLTItMjdiJTIyJTJDJTBBJTIwJTIwJTIwJTIwZHR5cGUlM0R0b3JjaC5iZmxvYXQxNiUyQyUwQSUyMCUyMCUyMCUyMGRldmljZV9tYXAlM0QlMjJhdXRvJTIyJTJDJTBBJTIwJTIwJTIwJTIwYXR0bl9pbXBsZW1lbnRhdGlvbiUzRCUyMnNkcGElMjIlMEEpJTBBJTBBaW5wdXRfdGV4dCUyMCUzRCUyMCUyMkV4cGxhaW4lMjBxdWFudHVtJTIwY29tcHV0aW5nJTIwc2ltcGx5LiUyMiUwQWlucHV0X2lkcyUyMCUzRCUyMHRva2VuaXplcihpbnB1dF90ZXh0JTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMikudG8obW9kZWwuZGV2aWNlKSUwQSUwQW91dHB1dHMlMjAlM0QlMjBtb2RlbC5nZW5lcmF0ZSgqKmlucHV0X2lkcyUyQyUyMG1heF9uZXdfdG9rZW5zJTNEMzIlMkMlMjBjYWNoZV9pbXBsZW1lbnRhdGlvbiUzRCUyMnN0YXRpYyUyMiklMEFwcmludCh0b2tlbml6ZXIuZGVjb2RlKG91dHB1dHMlNUIwJTVEJTJDJTIwc2tpcF9zcGVjaWFsX3Rva2VucyUzRFRydWUpKQ==",highlighted:`<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(load_in_4bit=<span class="hljs-literal">True</span>)
tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;google/gemma-2-27b&quot;</span>)
model = AutoModelForCausalLM.from_pretrained(
    <span class="hljs-string">&quot;google/gemma-2-27b&quot;</span>,
    dtype=torch.bfloat16,
    device_map=<span class="hljs-string">&quot;auto&quot;</span>,
    attn_implementation=<span class="hljs-string">&quot;sdpa&quot;</span>
)

input_text = <span class="hljs-string">&quot;Explain quantum computing simply.&quot;</span>
input_ids = tokenizer(input_text, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>).to(model.device)

outputs = model.generate(**input_ids, max_new_tokens=<span class="hljs-number">32</span>, cache_implementation=<span class="hljs-string">&quot;static&quot;</span>)
<span class="hljs-built_in">print</span>(tokenizer.decode(outputs[<span class="hljs-number">0</span>], skip_special_tokens=<span class="hljs-literal">True</span>))`,wrap:!1}}),de=new Ge({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycy51dGlscy5hdHRlbnRpb25fdmlzdWFsaXplciUyMGltcG9ydCUyMEF0dGVudGlvbk1hc2tWaXN1YWxpemVyJTBBdmlzdWFsaXplciUyMCUzRCUyMEF0dGVudGlvbk1hc2tWaXN1YWxpemVyKCUyMmdvb2dsZSUyRmdlbW1hLTJiJTIyKSUwQXZpc3VhbGl6ZXIoJTIyWW91JTIwYXJlJTIwYW4lMjBhc3Npc3RhbnQuJTIwTWFrZSUyMHN1cmUlMjB5b3UlMjBwcmludCUyMG1lJTIyKQ==",highlighted:`<span class="hljs-keyword">from</span> transformers.utils.attention_visualizer <span class="hljs-keyword">import</span> AttentionMaskVisualizer
visualizer = AttentionMaskVisualizer(<span class="hljs-string">&quot;google/gemma-2b&quot;</span>)
visualizer(<span class="hljs-string">&quot;You are an assistant. Make sure you print me&quot;</span>)`,wrap:!1}}),ce=new Be({props:{title:"Gemma2Config",local:"transformers.Gemma2Config",headingTag:"h2"}}),me=new E({props:{name:"class transformers.Gemma2Config",anchor:"transformers.Gemma2Config",parameters:[{name:"vocab_size",val:" = 256000"},{name:"hidden_size",val:" = 2304"},{name:"intermediate_size",val:" = 9216"},{name:"num_hidden_layers",val:" = 26"},{name:"num_attention_heads",val:" = 8"},{name:"num_key_value_heads",val:" = 4"},{name:"head_dim",val:" = 256"},{name:"hidden_activation",val:" = 'gelu_pytorch_tanh'"},{name:"max_position_embeddings",val:" = 8192"},{name:"initializer_range",val:" = 0.02"},{name:"rms_norm_eps",val:" = 1e-06"},{name:"use_cache",val:" = True"},{name:"pad_token_id",val:" = 0"},{name:"eos_token_id",val:" = 1"},{name:"bos_token_id",val:" = 2"},{name:"tie_word_embeddings",val:" = True"},{name:"rope_theta",val:" = 10000.0"},{name:"attention_bias",val:" = False"},{name:"attention_dropout",val:" = 0.0"},{name:"query_pre_attn_scalar",val:" = 256"},{name:"sliding_window",val:" = 4096"},{name:"layer_types",val:" = None"},{name:"final_logit_softcapping",val:" = 30.0"},{name:"attn_logit_softcapping",val:" = 50.0"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.Gemma2Config.vocab_size",description:`<strong>vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 256000) &#x2014;
Vocabulary size of the Gemma2 model. Defines the number of different tokens that can be represented by the
<code>inputs_ids</code> passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/gemma2#transformers.Gemma2Model">Gemma2Model</a>`,name:"vocab_size"},{anchor:"transformers.Gemma2Config.hidden_size",description:`<strong>hidden_size</strong> (<code>int</code>, <em>optional</em>, defaults to 2304) &#x2014;
Dimension of the hidden representations.`,name:"hidden_size"},{anchor:"transformers.Gemma2Config.intermediate_size",description:`<strong>intermediate_size</strong> (<code>int</code>, <em>optional</em>, defaults to 9216) &#x2014;
Dimension of the MLP representations.`,name:"intermediate_size"},{anchor:"transformers.Gemma2Config.num_hidden_layers",description:`<strong>num_hidden_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 26) &#x2014;
Number of hidden layers in the Transformer decoder.`,name:"num_hidden_layers"},{anchor:"transformers.Gemma2Config.num_attention_heads",description:`<strong>num_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 8) &#x2014;
Number of attention heads for each attention layer in the Transformer decoder.`,name:"num_attention_heads"},{anchor:"transformers.Gemma2Config.num_key_value_heads",description:`<strong>num_key_value_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 4) &#x2014;
This is the number of key_value heads that should be used to implement Grouped Query Attention. If
<code>num_key_value_heads=num_attention_heads</code>, the model will use Multi Head Attention (MHA), if
<code>num_key_value_heads=1</code> the model will use Multi Query Attention (MQA) otherwise GQA is used. When
converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
by meanpooling all the original heads within that group. For more details, check out <a href="https://huggingface.co/papers/2305.13245" rel="nofollow">this
paper</a>. If it is not specified, will default to
<code>num_attention_heads</code>.`,name:"num_key_value_heads"},{anchor:"transformers.Gemma2Config.head_dim",description:`<strong>head_dim</strong> (<code>int</code>, <em>optional</em>, defaults to 256) &#x2014;
The attention head dimension.`,name:"head_dim"},{anchor:"transformers.Gemma2Config.hidden_activation",description:`<strong>hidden_activation</strong> (<code>str</code> or <code>function</code>, <em>optional</em>, defaults to <code>&quot;gelu_pytorch_tanh&quot;</code>) &#x2014;
The non-linear activation function (function or string) in the decoder. Will default to <code>&quot;gelu_pytorch_tanh&quot;</code>
if not specified. <code>&quot;gelu_pytorch_tanh&quot;</code> uses an approximation of the <code>&quot;gelu&quot;</code> activation function.`,name:"hidden_activation"},{anchor:"transformers.Gemma2Config.max_position_embeddings",description:`<strong>max_position_embeddings</strong> (<code>int</code>, <em>optional</em>, defaults to 8192) &#x2014;
The maximum sequence length that this model might ever be used with.`,name:"max_position_embeddings"},{anchor:"transformers.Gemma2Config.initializer_range",description:`<strong>initializer_range</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.`,name:"initializer_range"},{anchor:"transformers.Gemma2Config.rms_norm_eps",description:`<strong>rms_norm_eps</strong> (<code>float</code>, <em>optional</em>, defaults to 1e-06) &#x2014;
The epsilon used by the rms normalization layers.`,name:"rms_norm_eps"},{anchor:"transformers.Gemma2Config.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not the model should return the last key/values attentions (not used by all models). Only
relevant if <code>config.is_decoder=True</code>.`,name:"use_cache"},{anchor:"transformers.Gemma2Config.pad_token_id",description:`<strong>pad_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 0) &#x2014;
Padding token id.`,name:"pad_token_id"},{anchor:"transformers.Gemma2Config.eos_token_id",description:`<strong>eos_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 1) &#x2014;
End of stream token id.`,name:"eos_token_id"},{anchor:"transformers.Gemma2Config.bos_token_id",description:`<strong>bos_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 2) &#x2014;
Beginning of stream token id.`,name:"bos_token_id"},{anchor:"transformers.Gemma2Config.tie_word_embeddings",description:`<strong>tie_word_embeddings</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether to tie weight embeddings`,name:"tie_word_embeddings"},{anchor:"transformers.Gemma2Config.rope_theta",description:`<strong>rope_theta</strong> (<code>float</code>, <em>optional</em>, defaults to 10000.0) &#x2014;
The base period of the RoPE embeddings.`,name:"rope_theta"},{anchor:"transformers.Gemma2Config.attention_bias",description:`<strong>attention_bias</strong> (<code>bool</code>, defaults to <code>False</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether to use a bias in the query, key, value and output projection layers during self-attention.`,name:"attention_bias"},{anchor:"transformers.Gemma2Config.attention_dropout",description:`<strong>attention_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The dropout ratio for the attention probabilities.`,name:"attention_dropout"},{anchor:"transformers.Gemma2Config.query_pre_attn_scalar",description:`<strong>query_pre_attn_scalar</strong> (<code>float</code>, <em>optional</em>, defaults to 256) &#x2014;
scaling factor used on the attention scores`,name:"query_pre_attn_scalar"},{anchor:"transformers.Gemma2Config.sliding_window",description:`<strong>sliding_window</strong> (<code>int</code>, <em>optional</em>, defaults to 4096) &#x2014;
in Gemma2, every other layer uses sliding window attention. This is the size of the sliding window.`,name:"sliding_window"},{anchor:"transformers.Gemma2Config.layer_types",description:`<strong>layer_types</strong> (<code>list</code>, <em>optional</em>) &#x2014;
Attention pattern for each layer.`,name:"layer_types"},{anchor:"transformers.Gemma2Config.final_logit_softcapping",description:`<strong>final_logit_softcapping</strong> (<code>float</code>, <em>optional</em>, defaults to 30.0) &#x2014;
scaling factor when applying tanh softcapping on the logits.`,name:"final_logit_softcapping"},{anchor:"transformers.Gemma2Config.attn_logit_softcapping",description:`<strong>attn_logit_softcapping</strong> (<code>float</code>, <em>optional</em>, defaults to 50.0) &#x2014;
scaling factor when applying tanh softcapping on the attention scores.`,name:"attn_logit_softcapping"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/gemma2/configuration_gemma2.py#L25"}}),A=new nn({props:{anchor:"transformers.Gemma2Config.example",$$slots:{default:[_n]},$$scope:{ctx:T}}}),pe=new Be({props:{title:"Gemma2Model",local:"transformers.Gemma2Model",headingTag:"h2"}}),ue=new E({props:{name:"class transformers.Gemma2Model",anchor:"transformers.Gemma2Model",parameters:[{name:"config",val:": Gemma2Config"}],parametersDescription:[{anchor:"transformers.Gemma2Model.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/gemma2#transformers.Gemma2Config">Gemma2Config</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/gemma2/modeling_gemma2.py#L358"}}),he=new E({props:{name:"forward",anchor:"transformers.Gemma2Model.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.Gemma2Model.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.Gemma2Model.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.Gemma2Model.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.Gemma2Model.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.Gemma2Model.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.Gemma2Model.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.Gemma2Model.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.Gemma2Model.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.Gemma2Model.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/gemma2/modeling_gemma2.py#L375",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast"
>transformers.modeling_outputs.BaseModelOutputWithPast</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/gemma2#transformers.Gemma2Config"
>Gemma2Config</a>) and inputs.</p>
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
`}}),Q=new Xe({props:{$$slots:{default:[bn]},$$scope:{ctx:T}}}),fe=new Be({props:{title:"Gemma2ForCausalLM",local:"transformers.Gemma2ForCausalLM",headingTag:"h2"}}),ge=new E({props:{name:"class transformers.Gemma2ForCausalLM",anchor:"transformers.Gemma2ForCausalLM",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.Gemma2ForCausalLM.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/gemma2#transformers.Gemma2ForCausalLM">Gemma2ForCausalLM</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/gemma2/modeling_gemma2.py#L488"}}),_e=new E({props:{name:"forward",anchor:"transformers.Gemma2ForCausalLM.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"},{name:"logits_to_keep",val:": typing.Union[int, torch.Tensor] = 0"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.Gemma2ForCausalLM.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.Gemma2ForCausalLM.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.Gemma2ForCausalLM.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.Gemma2ForCausalLM.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.Gemma2ForCausalLM.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.Gemma2ForCausalLM.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.Gemma2ForCausalLM.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.Gemma2ForCausalLM.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.Gemma2ForCausalLM.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.Gemma2ForCausalLM.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"},{anchor:"transformers.Gemma2ForCausalLM.forward.logits_to_keep",description:`<strong>logits_to_keep</strong> (<code>Union[int, torch.Tensor]</code>, defaults to <code>0</code>) &#x2014;
If an <code>int</code>, compute logits for the last <code>logits_to_keep</code> tokens. If <code>0</code>, calculate logits for all
<code>input_ids</code> (special case). Only last token logits are needed for generation, and calculating them only for that
token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
If a <code>torch.Tensor</code>, must be 1D corresponding to the indices to keep in the sequence length dimension.
This is useful when using packed tensor format (single dimension for batch and sequence length).`,name:"logits_to_keep"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/gemma2/modeling_gemma2.py#L502",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast"
>transformers.modeling_outputs.CausalLMOutputWithPast</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/gemma2#transformers.Gemma2Config"
>Gemma2Config</a>) and inputs.</p>
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
`}}),D=new Xe({props:{$$slots:{default:[yn]},$$scope:{ctx:T}}}),O=new nn({props:{anchor:"transformers.Gemma2ForCausalLM.forward.example",$$slots:{default:[vn]},$$scope:{ctx:T}}}),be=new Be({props:{title:"Gemma2ForSequenceClassification",local:"transformers.Gemma2ForSequenceClassification",headingTag:"h2"}}),ye=new E({props:{name:"class transformers.Gemma2ForSequenceClassification",anchor:"transformers.Gemma2ForSequenceClassification",parameters:[{name:"config",val:""}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/gemma2/modeling_gemma2.py#L582"}}),ve=new E({props:{name:"forward",anchor:"transformers.Gemma2ForSequenceClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.Gemma2ForSequenceClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.Gemma2ForSequenceClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.Gemma2ForSequenceClassification.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.Gemma2ForSequenceClassification.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.Gemma2ForSequenceClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.Gemma2ForSequenceClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.Gemma2ForSequenceClassification.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
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
`}}),Y=new Xe({props:{$$slots:{default:[Tn]},$$scope:{ctx:T}}}),Te=new Be({props:{title:"Gemma2ForTokenClassification",local:"transformers.Gemma2ForTokenClassification",headingTag:"h2"}}),we=new E({props:{name:"class transformers.Gemma2ForTokenClassification",anchor:"transformers.Gemma2ForTokenClassification",parameters:[{name:"config",val:""}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/gemma2/modeling_gemma2.py#L586"}}),ke=new E({props:{name:"forward",anchor:"transformers.Gemma2ForTokenClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.Gemma2ForTokenClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.Gemma2ForTokenClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.Gemma2ForTokenClassification.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.Gemma2ForTokenClassification.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.Gemma2ForTokenClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.Gemma2ForTokenClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.Gemma2ForTokenClassification.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
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
`}}),K=new Xe({props:{$$slots:{default:[wn]},$$scope:{ctx:T}}}),Me=new cn({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/gemma2.md"}}),{c(){t=d("meta"),l=r(),n=d("p"),u=r(),k=d("p"),k.innerHTML=w,m=r(),M=d("div"),M.innerHTML=He,ee=r(),h(q.$$.fragment),Ee=r(),te=d("p"),te.innerHTML=Lt,Re=r(),ne=d("p"),ne.textContent=Zt,Pe=r(),oe=d("p"),oe.innerHTML=Vt,Se=r(),h(R.$$.fragment),Ae=r(),se=d("p"),se.innerHTML=Bt,Qe=r(),h(P.$$.fragment),De=r(),ae=d("p"),ae.innerHTML=Ht,Oe=r(),re=d("p"),re.innerHTML=Nt,Ye=r(),h(ie.$$.fragment),Ke=r(),le=d("p"),le.innerHTML=Xt,et=r(),h(de.$$.fragment),tt=r(),S=d("div"),S.innerHTML=Et,nt=r(),h(ce.$$.fragment),ot=r(),F=d("div"),h(me.$$.fragment),ft=r(),xe=d("p"),xe.innerHTML=Rt,gt=r(),h(A.$$.fragment),st=r(),h(pe.$$.fragment),at=r(),$=d("div"),h(ue.$$.fragment),_t=r(),ze=d("p"),ze.textContent=Pt,bt=r(),Je=d("p"),Je.innerHTML=St,yt=r(),Fe=d("p"),Fe.innerHTML=At,vt=r(),j=d("div"),h(he.$$.fragment),Tt=r(),Ie=d("p"),Ie.innerHTML=Qt,wt=r(),h(Q.$$.fragment),rt=r(),h(fe.$$.fragment),it=r(),C=d("div"),h(ge.$$.fragment),kt=r(),Ue=d("p"),Ue.textContent=Dt,Mt=r(),qe=d("p"),qe.innerHTML=Ot,$t=r(),je=d("p"),je.innerHTML=Yt,Ct=r(),z=d("div"),h(_e.$$.fragment),Gt=r(),We=d("p"),We.innerHTML=Kt,xt=r(),h(D.$$.fragment),zt=r(),h(O.$$.fragment),lt=r(),h(be.$$.fragment),dt=r(),V=d("div"),h(ye.$$.fragment),Jt=r(),W=d("div"),h(ve.$$.fragment),Ft=r(),Le=d("p"),Le.innerHTML=en,It=r(),h(Y.$$.fragment),ct=r(),h(Te.$$.fragment),mt=r(),B=d("div"),h(we.$$.fragment),Ut=r(),L=d("div"),h(ke.$$.fragment),qt=r(),Ze=d("p"),Ze.innerHTML=tn,jt=r(),h(K.$$.fragment),pt=r(),h(Me.$$.fragment),ut=r(),Ne=d("p"),this.h()},l(e){const o=ln("svelte-u9bgzb",document.head);t=c(o,"META",{name:!0,content:!0}),o.forEach(s),l=i(e),n=c(e,"P",{}),U(n).forEach(s),u=i(e),k=c(e,"P",{"data-svelte-h":!0}),v(k)!=="svelte-1jsic5e"&&(k.innerHTML=w),m=i(e),M=c(e,"DIV",{style:!0,"data-svelte-h":!0}),v(M)!=="svelte-11gpmgv"&&(M.innerHTML=He),ee=i(e),f(q.$$.fragment,e),Ee=i(e),te=c(e,"P",{"data-svelte-h":!0}),v(te)!=="svelte-1371qud"&&(te.innerHTML=Lt),Re=i(e),ne=c(e,"P",{"data-svelte-h":!0}),v(ne)!=="svelte-pfgil3"&&(ne.textContent=Zt),Pe=i(e),oe=c(e,"P",{"data-svelte-h":!0}),v(oe)!=="svelte-o0pjqs"&&(oe.innerHTML=Vt),Se=i(e),f(R.$$.fragment,e),Ae=i(e),se=c(e,"P",{"data-svelte-h":!0}),v(se)!=="svelte-1eliowp"&&(se.innerHTML=Bt),Qe=i(e),f(P.$$.fragment,e),De=i(e),ae=c(e,"P",{"data-svelte-h":!0}),v(ae)!=="svelte-nf5ooi"&&(ae.innerHTML=Ht),Oe=i(e),re=c(e,"P",{"data-svelte-h":!0}),v(re)!=="svelte-11sw8fc"&&(re.innerHTML=Nt),Ye=i(e),f(ie.$$.fragment,e),Ke=i(e),le=c(e,"P",{"data-svelte-h":!0}),v(le)!=="svelte-w3z5ks"&&(le.innerHTML=Xt),et=i(e),f(de.$$.fragment,e),tt=i(e),S=c(e,"DIV",{class:!0,"data-svelte-h":!0}),v(S)!=="svelte-3g7rv8"&&(S.innerHTML=Et),nt=i(e),f(ce.$$.fragment,e),ot=i(e),F=c(e,"DIV",{class:!0});var H=U(F);f(me.$$.fragment,H),ft=i(H),xe=c(H,"P",{"data-svelte-h":!0}),v(xe)!=="svelte-z4p9ii"&&(xe.innerHTML=Rt),gt=i(H),f(A.$$.fragment,H),H.forEach(s),st=i(e),f(pe.$$.fragment,e),at=i(e),$=c(e,"DIV",{class:!0});var G=U($);f(ue.$$.fragment,G),_t=i(G),ze=c(G,"P",{"data-svelte-h":!0}),v(ze)!=="svelte-19gzhzl"&&(ze.textContent=Pt),bt=i(G),Je=c(G,"P",{"data-svelte-h":!0}),v(Je)!=="svelte-q52n56"&&(Je.innerHTML=St),yt=i(G),Fe=c(G,"P",{"data-svelte-h":!0}),v(Fe)!=="svelte-hswkmf"&&(Fe.innerHTML=At),vt=i(G),j=c(G,"DIV",{class:!0});var N=U(j);f(he.$$.fragment,N),Tt=i(N),Ie=c(N,"P",{"data-svelte-h":!0}),v(Ie)!=="svelte-1dpslb2"&&(Ie.innerHTML=Qt),wt=i(N),f(Q.$$.fragment,N),N.forEach(s),G.forEach(s),rt=i(e),f(fe.$$.fragment,e),it=i(e),C=c(e,"DIV",{class:!0});var x=U(C);f(ge.$$.fragment,x),kt=i(x),Ue=c(x,"P",{"data-svelte-h":!0}),v(Ue)!=="svelte-157oieg"&&(Ue.textContent=Dt),Mt=i(x),qe=c(x,"P",{"data-svelte-h":!0}),v(qe)!=="svelte-q52n56"&&(qe.innerHTML=Ot),$t=i(x),je=c(x,"P",{"data-svelte-h":!0}),v(je)!=="svelte-hswkmf"&&(je.innerHTML=Yt),Ct=i(x),z=c(x,"DIV",{class:!0});var I=U(z);f(_e.$$.fragment,I),Gt=i(I),We=c(I,"P",{"data-svelte-h":!0}),v(We)!=="svelte-15hvvm6"&&(We.innerHTML=Kt),xt=i(I),f(D.$$.fragment,I),zt=i(I),f(O.$$.fragment,I),I.forEach(s),x.forEach(s),lt=i(e),f(be.$$.fragment,e),dt=i(e),V=c(e,"DIV",{class:!0});var $e=U(V);f(ye.$$.fragment,$e),Jt=i($e),W=c($e,"DIV",{class:!0});var X=U(W);f(ve.$$.fragment,X),Ft=i(X),Le=c(X,"P",{"data-svelte-h":!0}),v(Le)!=="svelte-1sal4ui"&&(Le.innerHTML=en),It=i(X),f(Y.$$.fragment,X),X.forEach(s),$e.forEach(s),ct=i(e),f(Te.$$.fragment,e),mt=i(e),B=c(e,"DIV",{class:!0});var Ce=U(B);f(we.$$.fragment,Ce),Ut=i(Ce),L=c(Ce,"DIV",{class:!0});var Ve=U(L);f(ke.$$.fragment,Ve),qt=i(Ve),Ze=c(Ve,"P",{"data-svelte-h":!0}),v(Ze)!=="svelte-1py4aay"&&(Ze.innerHTML=tn),jt=i(Ve),f(K.$$.fragment,Ve),Ve.forEach(s),Ce.forEach(s),pt=i(e),f(Me.$$.fragment,e),ut=i(e),Ne=c(e,"P",{}),U(Ne).forEach(s),this.h()},h(){J(t,"name","hf:doc:metadata"),J(t,"content",Mn),dn(M,"float","right"),J(S,"class","flex justify-center"),J(F,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(j,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J($,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(z,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(C,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(W,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(V,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(L,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),J(B,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,o){p(document.head,t),a(e,l,o),a(e,n,o),a(e,u,o),a(e,k,o),a(e,m,o),a(e,M,o),a(e,ee,o),g(q,e,o),a(e,Ee,o),a(e,te,o),a(e,Re,o),a(e,ne,o),a(e,Pe,o),a(e,oe,o),a(e,Se,o),g(R,e,o),a(e,Ae,o),a(e,se,o),a(e,Qe,o),g(P,e,o),a(e,De,o),a(e,ae,o),a(e,Oe,o),a(e,re,o),a(e,Ye,o),g(ie,e,o),a(e,Ke,o),a(e,le,o),a(e,et,o),g(de,e,o),a(e,tt,o),a(e,S,o),a(e,nt,o),g(ce,e,o),a(e,ot,o),a(e,F,o),g(me,F,null),p(F,ft),p(F,xe),p(F,gt),g(A,F,null),a(e,st,o),g(pe,e,o),a(e,at,o),a(e,$,o),g(ue,$,null),p($,_t),p($,ze),p($,bt),p($,Je),p($,yt),p($,Fe),p($,vt),p($,j),g(he,j,null),p(j,Tt),p(j,Ie),p(j,wt),g(Q,j,null),a(e,rt,o),g(fe,e,o),a(e,it,o),a(e,C,o),g(ge,C,null),p(C,kt),p(C,Ue),p(C,Mt),p(C,qe),p(C,$t),p(C,je),p(C,Ct),p(C,z),g(_e,z,null),p(z,Gt),p(z,We),p(z,xt),g(D,z,null),p(z,zt),g(O,z,null),a(e,lt,o),g(be,e,o),a(e,dt,o),a(e,V,o),g(ye,V,null),p(V,Jt),p(V,W),g(ve,W,null),p(W,Ft),p(W,Le),p(W,It),g(Y,W,null),a(e,ct,o),g(Te,e,o),a(e,mt,o),a(e,B,o),g(we,B,null),p(B,Ut),p(B,L),g(ke,L,null),p(L,qt),p(L,Ze),p(L,jt),g(K,L,null),a(e,pt,o),g(Me,e,o),a(e,ut,o),a(e,Ne,o),ht=!0},p(e,[o]){const H={};o&2&&(H.$$scope={dirty:o,ctx:e}),R.$set(H);const G={};o&2&&(G.$$scope={dirty:o,ctx:e}),P.$set(G);const N={};o&2&&(N.$$scope={dirty:o,ctx:e}),A.$set(N);const x={};o&2&&(x.$$scope={dirty:o,ctx:e}),Q.$set(x);const I={};o&2&&(I.$$scope={dirty:o,ctx:e}),D.$set(I);const $e={};o&2&&($e.$$scope={dirty:o,ctx:e}),O.$set($e);const X={};o&2&&(X.$$scope={dirty:o,ctx:e}),Y.$set(X);const Ce={};o&2&&(Ce.$$scope={dirty:o,ctx:e}),K.$set(Ce)},i(e){ht||(_(q.$$.fragment,e),_(R.$$.fragment,e),_(P.$$.fragment,e),_(ie.$$.fragment,e),_(de.$$.fragment,e),_(ce.$$.fragment,e),_(me.$$.fragment,e),_(A.$$.fragment,e),_(pe.$$.fragment,e),_(ue.$$.fragment,e),_(he.$$.fragment,e),_(Q.$$.fragment,e),_(fe.$$.fragment,e),_(ge.$$.fragment,e),_(_e.$$.fragment,e),_(D.$$.fragment,e),_(O.$$.fragment,e),_(be.$$.fragment,e),_(ye.$$.fragment,e),_(ve.$$.fragment,e),_(Y.$$.fragment,e),_(Te.$$.fragment,e),_(we.$$.fragment,e),_(ke.$$.fragment,e),_(K.$$.fragment,e),_(Me.$$.fragment,e),ht=!0)},o(e){b(q.$$.fragment,e),b(R.$$.fragment,e),b(P.$$.fragment,e),b(ie.$$.fragment,e),b(de.$$.fragment,e),b(ce.$$.fragment,e),b(me.$$.fragment,e),b(A.$$.fragment,e),b(pe.$$.fragment,e),b(ue.$$.fragment,e),b(he.$$.fragment,e),b(Q.$$.fragment,e),b(fe.$$.fragment,e),b(ge.$$.fragment,e),b(_e.$$.fragment,e),b(D.$$.fragment,e),b(O.$$.fragment,e),b(be.$$.fragment,e),b(ye.$$.fragment,e),b(ve.$$.fragment,e),b(Y.$$.fragment,e),b(Te.$$.fragment,e),b(we.$$.fragment,e),b(ke.$$.fragment,e),b(K.$$.fragment,e),b(Me.$$.fragment,e),ht=!1},d(e){e&&(s(l),s(n),s(u),s(k),s(m),s(M),s(ee),s(Ee),s(te),s(Re),s(ne),s(Pe),s(oe),s(Se),s(Ae),s(se),s(Qe),s(De),s(ae),s(Oe),s(re),s(Ye),s(Ke),s(le),s(et),s(tt),s(S),s(nt),s(ot),s(F),s(st),s(at),s($),s(rt),s(it),s(C),s(lt),s(dt),s(V),s(ct),s(mt),s(B),s(pt),s(ut),s(Ne)),s(t),y(q,e),y(R,e),y(P,e),y(ie,e),y(de,e),y(ce,e),y(me),y(A),y(pe,e),y(ue),y(he),y(Q),y(fe,e),y(ge),y(_e),y(D),y(O),y(be,e),y(ye),y(ve),y(Y),y(Te,e),y(we),y(ke),y(K),y(Me,e)}}}const Mn='{"title":"Gemma2","local":"gemma2","sections":[{"title":"Gemma2Config","local":"transformers.Gemma2Config","sections":[],"depth":2},{"title":"Gemma2Model","local":"transformers.Gemma2Model","sections":[],"depth":2},{"title":"Gemma2ForCausalLM","local":"transformers.Gemma2ForCausalLM","sections":[],"depth":2},{"title":"Gemma2ForSequenceClassification","local":"transformers.Gemma2ForSequenceClassification","sections":[],"depth":2},{"title":"Gemma2ForTokenClassification","local":"transformers.Gemma2ForTokenClassification","sections":[],"depth":2}],"depth":1}';function $n(T){return sn(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class qn extends an{constructor(t){super(),rn(this,t,$n,kn,on,{})}}export{qn as component};
