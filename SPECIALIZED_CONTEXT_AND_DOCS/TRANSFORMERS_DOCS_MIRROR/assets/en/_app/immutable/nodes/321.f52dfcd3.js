import{s as bt,o as Tt,n as X}from"../chunks/scheduler.18a86fab.js";import{S as wt,i as vt,g as p,s as l,r as f,A as kt,h as u,f as s,c as d,j as ae,x as w,u as g,k as re,l as $t,y as h,a as i,v as _,d as y,t as M,w as b}from"../chunks/index.98837b22.js";import{T as tt}from"../chunks/Tip.77304350.js";import{D as be}from"../chunks/Docstring.a1ef7999.js";import{C as _e}from"../chunks/CodeBlock.8d0c2e8a.js";import{E as Mt}from"../chunks/ExampleCodeBlock.8c3ee1f9.js";import{H as qe,E as Ct}from"../chunks/getInferenceSnippets.06c2775f.js";import{H as Jt,a as ot}from"../chunks/HfOption.6641485e.js";function Ut(k){let t,a='This model was contributed by <a href="https://huggingface.co/shanearora" rel="nofollow">shanearora</a>.',o,c,T="Click on the OLMo models in the right sidebar for more examples of how to apply OLMo to different language tasks.";return{c(){t=p("p"),t.innerHTML=a,o=l(),c=p("p"),c.textContent=T},l(m){t=u(m,"P",{"data-svelte-h":!0}),w(t)!=="svelte-ebupj1"&&(t.innerHTML=a),o=d(m),c=u(m,"P",{"data-svelte-h":!0}),w(c)!=="svelte-k9cqg1"&&(c.textContent=T)},m(m,r){i(m,t,r),i(m,o,r),i(m,c,r)},p:X,d(m){m&&(s(t),s(o),s(c))}}}function jt(k){let t,a;return t=new _e({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwcGlwZWxpbmUlMEElMEFwaXBlJTIwJTNEJTIwcGlwZWxpbmUoJTBBJTIwJTIwJTIwJTIwdGFzayUzRCUyMnRleHQtZ2VuZXJhdGlvbiUyMiUyQyUwQSUyMCUyMCUyMCUyMG1vZGVsJTNEJTIyYWxsZW5haSUyRk9MTW8tN0ItaGYlMjIlMkMlMEElMjAlMjAlMjAlMjBkdHlwZSUzRHRvcmNoLmZsb2F0MTYlMkMlMEElMjAlMjAlMjAlMjBkZXZpY2UlM0QwJTJDJTBBKSUwQSUwQXJlc3VsdCUyMCUzRCUyMHBpcGUoJTIyUGxhbnRzJTIwY3JlYXRlJTIwZW5lcmd5JTIwdGhyb3VnaCUyMGElMjBwcm9jZXNzJTIwa25vd24lMjBhcyUyMiklMEFwcmludChyZXN1bHQp",highlighted:`<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> pipeline

pipe = pipeline(
    task=<span class="hljs-string">&quot;text-generation&quot;</span>,
    model=<span class="hljs-string">&quot;allenai/OLMo-7B-hf&quot;</span>,
    dtype=torch.float16,
    device=<span class="hljs-number">0</span>,
)

result = pipe(<span class="hljs-string">&quot;Plants create energy through a process known as&quot;</span>)
<span class="hljs-built_in">print</span>(result)`,wrap:!1}}),{c(){f(t.$$.fragment)},l(o){g(t.$$.fragment,o)},m(o,c){_(t,o,c),a=!0},p:X,i(o){a||(y(t.$$.fragment,o),a=!0)},o(o){M(t.$$.fragment,o),a=!1},d(o){b(t,o)}}}function zt(k){let t,a;return t=new _e({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b01vZGVsRm9yQ2F1c2FsTE0lMkMlMjBBdXRvVG9rZW5pemVyJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTBBJTIwJTIwJTIwJTIwJTIyYWxsZW5haSUyRk9MTW8tN0ItaGYlMjIlMEEpJTBBJTBBbW9kZWwlMjAlM0QlMjBBdXRvTW9kZWxGb3JDYXVzYWxMTS5mcm9tX3ByZXRyYWluZWQoJTBBJTIwJTIwJTIwJTIwJTIyYWxsZW5haSUyRk9MTW8tN0ItaGYlMjIlMkMlMEElMjAlMjAlMjAlMjBkdHlwZSUzRHRvcmNoLmZsb2F0MTYlMkMlMEElMjAlMjAlMjAlMjBkZXZpY2VfbWFwJTNEJTIyYXV0byUyMiUyQyUwQSUyMCUyMCUyMCUyMGF0dG5faW1wbGVtZW50YXRpb24lM0QlMjJzZHBhJTIyJTBBKSUwQWlucHV0X2lkcyUyMCUzRCUyMHRva2VuaXplciglMjJQbGFudHMlMjBjcmVhdGUlMjBlbmVyZ3klMjB0aHJvdWdoJTIwYSUyMHByb2Nlc3MlMjBrbm93biUyMGFzJTIyJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMikudG8obW9kZWwuZGV2aWNlKSUwQSUwQW91dHB1dCUyMCUzRCUyMG1vZGVsLmdlbmVyYXRlKCoqaW5wdXRfaWRzJTJDJTIwbWF4X2xlbmd0aCUzRDUwJTJDJTIwY2FjaGVfaW1wbGVtZW50YXRpb24lM0QlMjJzdGF0aWMlMjIpJTBBcHJpbnQodG9rZW5pemVyLmRlY29kZShvdXRwdXQlNUIwJTVEJTJDJTIwc2tpcF9zcGVjaWFsX3Rva2VucyUzRFRydWUpKQ==",highlighted:`<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    <span class="hljs-string">&quot;allenai/OLMo-7B-hf&quot;</span>
)

model = AutoModelForCausalLM.from_pretrained(
    <span class="hljs-string">&quot;allenai/OLMo-7B-hf&quot;</span>,
    dtype=torch.float16,
    device_map=<span class="hljs-string">&quot;auto&quot;</span>,
    attn_implementation=<span class="hljs-string">&quot;sdpa&quot;</span>
)
input_ids = tokenizer(<span class="hljs-string">&quot;Plants create energy through a process known as&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>).to(model.device)

output = model.generate(**input_ids, max_length=<span class="hljs-number">50</span>, cache_implementation=<span class="hljs-string">&quot;static&quot;</span>)
<span class="hljs-built_in">print</span>(tokenizer.decode(output[<span class="hljs-number">0</span>], skip_special_tokens=<span class="hljs-literal">True</span>))`,wrap:!1}}),{c(){f(t.$$.fragment)},l(o){g(t.$$.fragment,o)},m(o,c){_(t,o,c),a=!0},p:X,i(o){a||(y(t.$$.fragment,o),a=!0)},o(o){M(t.$$.fragment,o),a=!1},d(o){b(t,o)}}}function It(k){let t,a;return t=new _e({props:{code:"ZWNobyUyMC1lJTIwJTIyUGxhbnRzJTIwY3JlYXRlJTIwZW5lcmd5JTIwdGhyb3VnaCUyMGElMjBwcm9jZXNzJTIwa25vd24lMjBhcyUyMiUyMCU3QyUyMHRyYW5zZm9ybWVycyUyMHJ1biUyMC0tdGFzayUyMHRleHQtZ2VuZXJhdGlvbiUyMC0tbW9kZWwlMjBhbGxlbmFpJTJGT0xNby03Qi1oZiUyMC0tZGV2aWNlJTIwMA==",highlighted:'<span class="hljs-built_in">echo</span> -e <span class="hljs-string">&quot;Plants create energy through a process known as&quot;</span> | transformers run --task text-generation --model allenai/OLMo-7B-hf --device 0',wrap:!1}}),{c(){f(t.$$.fragment)},l(o){g(t.$$.fragment,o)},m(o,c){_(t,o,c),a=!0},p:X,i(o){a||(y(t.$$.fragment,o),a=!0)},o(o){M(t.$$.fragment,o),a=!1},d(o){b(t,o)}}}function xt(k){let t,a,o,c,T,m;return t=new ot({props:{id:"usage",option:"Pipeline",$$slots:{default:[jt]},$$scope:{ctx:k}}}),o=new ot({props:{id:"usage",option:"AutoModel",$$slots:{default:[zt]},$$scope:{ctx:k}}}),T=new ot({props:{id:"usage",option:"transformers CLI",$$slots:{default:[It]},$$scope:{ctx:k}}}),{c(){f(t.$$.fragment),a=l(),f(o.$$.fragment),c=l(),f(T.$$.fragment)},l(r){g(t.$$.fragment,r),a=d(r),g(o.$$.fragment,r),c=d(r),g(T.$$.fragment,r)},m(r,v){_(t,r,v),i(r,a,v),_(o,r,v),i(r,c,v),_(T,r,v),m=!0},p(r,v){const ye={};v&2&&(ye.$$scope={dirty:v,ctx:r}),t.$set(ye);const E={};v&2&&(E.$$scope={dirty:v,ctx:r}),o.$set(E);const W={};v&2&&(W.$$scope={dirty:v,ctx:r}),T.$set(W)},i(r){m||(y(t.$$.fragment,r),y(o.$$.fragment,r),y(T.$$.fragment,r),m=!0)},o(r){M(t.$$.fragment,r),M(o.$$.fragment,r),M(T.$$.fragment,r),m=!1},d(r){r&&(s(a),s(c)),b(t,r),b(o,r),b(T,r)}}}function Wt(k){let t,a;return t=new _e({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyME9sbW9Nb2RlbCUyQyUyME9sbW9Db25maWclMEElMEElMjMlMjBJbml0aWFsaXppbmclMjBhJTIwT0xNbyUyMDdCJTIwc3R5bGUlMjBjb25maWd1cmF0aW9uJTBBY29uZmlndXJhdGlvbiUyMCUzRCUyME9sbW9Db25maWcoKSUwQSUwQSUyMyUyMEluaXRpYWxpemluZyUyMGElMjBtb2RlbCUyMGZyb20lMjB0aGUlMjBPTE1vJTIwN0IlMjBzdHlsZSUyMGNvbmZpZ3VyYXRpb24lMEFtb2RlbCUyMCUzRCUyME9sbW9Nb2RlbChjb25maWd1cmF0aW9uKSUwQSUwQSUyMyUyMEFjY2Vzc2luZyUyMHRoZSUyMG1vZGVsJTIwY29uZmlndXJhdGlvbiUwQWNvbmZpZ3VyYXRpb24lMjAlM0QlMjBtb2RlbC5jb25maWc=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> OlmoModel, OlmoConfig

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a OLMo 7B style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = OlmoConfig()

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a model from the OLMo 7B style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = OlmoModel(configuration)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Accessing the model configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = model.config`,wrap:!1}}),{c(){f(t.$$.fragment)},l(o){g(t.$$.fragment,o)},m(o,c){_(t,o,c),a=!0},p:X,i(o){a||(y(t.$$.fragment,o),a=!0)},o(o){M(t.$$.fragment,o),a=!1},d(o){b(t,o)}}}function Bt(k){let t,a=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=p("p"),t.innerHTML=a},l(o){t=u(o,"P",{"data-svelte-h":!0}),w(t)!=="svelte-fincs2"&&(t.innerHTML=a)},m(o,c){i(o,t,c)},p:X,d(o){o&&s(t)}}}function Ot(k){let t,a=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=p("p"),t.innerHTML=a},l(o){t=u(o,"P",{"data-svelte-h":!0}),w(t)!=="svelte-fincs2"&&(t.innerHTML=a)},m(o,c){i(o,t,c)},p:X,d(o){o&&s(t)}}}function Lt(k){let t,a="Example:",o,c,T;return c=new _e({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBPbG1vRm9yQ2F1c2FsTE0lMEElMEFtb2RlbCUyMCUzRCUyME9sbW9Gb3JDYXVzYWxMTS5mcm9tX3ByZXRyYWluZWQoJTIybWV0YS1vbG1vJTJGT2xtby0yLTdiLWhmJTIyKSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMm1ldGEtb2xtbyUyRk9sbW8tMi03Yi1oZiUyMiklMEElMEFwcm9tcHQlMjAlM0QlMjAlMjJIZXklMkMlMjBhcmUlMjB5b3UlMjBjb25zY2lvdXMlM0YlMjBDYW4lMjB5b3UlMjB0YWxrJTIwdG8lMjBtZSUzRiUyMiUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplcihwcm9tcHQlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQSUwQSUyMyUyMEdlbmVyYXRlJTBBZ2VuZXJhdGVfaWRzJTIwJTNEJTIwbW9kZWwuZ2VuZXJhdGUoaW5wdXRzLmlucHV0X2lkcyUyQyUyMG1heF9sZW5ndGglM0QzMCklMEF0b2tlbml6ZXIuYmF0Y2hfZGVjb2RlKGdlbmVyYXRlX2lkcyUyQyUyMHNraXBfc3BlY2lhbF90b2tlbnMlM0RUcnVlJTJDJTIwY2xlYW5fdXBfdG9rZW5pemF0aW9uX3NwYWNlcyUzREZhbHNlKSU1QjAlNUQ=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, OlmoForCausalLM

<span class="hljs-meta">&gt;&gt;&gt; </span>model = OlmoForCausalLM.from_pretrained(<span class="hljs-string">&quot;meta-olmo/Olmo-2-7b-hf&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;meta-olmo/Olmo-2-7b-hf&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>prompt = <span class="hljs-string">&quot;Hey, are you conscious? Can you talk to me?&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(prompt, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Generate</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>generate_ids = model.generate(inputs.input_ids, max_length=<span class="hljs-number">30</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer.batch_decode(generate_ids, skip_special_tokens=<span class="hljs-literal">True</span>, clean_up_tokenization_spaces=<span class="hljs-literal">False</span>)[<span class="hljs-number">0</span>]
<span class="hljs-string">&quot;Hey, are you conscious? Can you talk to me?\\nI&#x27;m not conscious, but I can talk to you.&quot;</span>`,wrap:!1}}),{c(){t=p("p"),t.textContent=a,o=l(),f(c.$$.fragment)},l(m){t=u(m,"P",{"data-svelte-h":!0}),w(t)!=="svelte-11lpom8"&&(t.textContent=a),o=d(m),g(c.$$.fragment,m)},m(m,r){i(m,t,r),i(m,o,r),_(c,m,r),T=!0},p:X,i(m){T||(y(c.$$.fragment,m),T=!0)},o(m){M(c.$$.fragment,m),T=!1},d(m){m&&(s(t),s(o)),b(c,m)}}}function Zt(k){let t,a,o,c,T,m="<em>This model was released on 2024-02-01 and added to Hugging Face Transformers on 2024-04-17.</em>",r,v,ye='<div class="flex flex-wrap space-x-1"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/> <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat"/> <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/> <img alt="Tensor parallelism" src="https://img.shields.io/badge/Tensor%20parallelism-06b6d4?style=flat&amp;logoColor=white"/></div>',E,W,Te,H,nt='<a href="https://huggingface.co/papers/2402.00838" rel="nofollow">OLMo</a> is a 7B-parameter dense language model. It uses SwiGLU activations, non-parametric layer normalization, rotary positional embeddings, and a BPE tokenizer that masks personally identifiable information. It is pretrained on <a href="https://huggingface.co/datasets/allenai/dolma" rel="nofollow">Dolma</a>, a 3T-token dataset. OLMo was released to provide complete transparency of not just the model weights but the training data, training code, and evaluation code to enable more research on language models.',we,V,st='You can find all the original OLMo checkpoints under the <a href="https://huggingface.co/collections/allenai/olmo-suite-65aeaae8fe5b6b2122b46778" rel="nofollow">OLMo</a> collection.',ve,L,ke,P,at='The example below demonstrates how to generate text with <a href="/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.Pipeline">Pipeline</a> or the <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel">AutoModel</a> class.',$e,Z,Ce,Q,rt='Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the <a href="../quantization/overview">Quantization</a> overview for more available quantization backends.',Je,Y,it='The example below uses <a href="../quantization/bitsandbytes">bitsandbytes</a> to only quantize the weights to 4-bits.',Ue,N,je,A,ze,J,S,Fe,ie,lt=`This is the configuration class to store the configuration of a <a href="/docs/transformers/v4.56.2/en/model_doc/olmo#transformers.OlmoModel">OlmoModel</a>. It is used to instantiate an OLMo
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the <a href="https://huggingface.co/allenai/OLMo-7B-hf" rel="nofollow">allenai/OLMo-7B-hf</a>.`,Re,le,dt=`Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> and can be used to control the model outputs. Read the
documentation from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> for more information.`,Ge,q,Ie,D,xe,$,K,Xe,de,ct="The bare Olmo Model outputting raw hidden-states without any specific head on top.",Ee,ce,mt=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,He,me,pt=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Ve,B,ee,Pe,pe,ut='The <a href="/docs/transformers/v4.56.2/en/model_doc/olmo#transformers.OlmoModel">OlmoModel</a> forward method, overrides the <code>__call__</code> special method.',Qe,F,We,te,Be,C,oe,Ye,ue,ht="The Olmo Model for causal language modeling.",Ne,he,ft=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Ae,fe,gt=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Se,z,ne,De,ge,_t='The <a href="/docs/transformers/v4.56.2/en/model_doc/olmo#transformers.OlmoForCausalLM">OlmoForCausalLM</a> forward method, overrides the <code>__call__</code> special method.',Ke,R,et,G,Oe,se,Le,Me,Ze;return W=new qe({props:{title:"OLMo",local:"olmo",headingTag:"h1"}}),L=new tt({props:{warning:!1,$$slots:{default:[Ut]},$$scope:{ctx:k}}}),Z=new Jt({props:{id:"usage",options:["Pipeline","AutoModel","transformers CLI"],$$slots:{default:[xt]},$$scope:{ctx:k}}}),N=new _e({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b01vZGVsRm9yQ2F1c2FsTE0lMkMlMjBBdXRvVG9rZW5pemVyJTJDJTIwQml0c0FuZEJ5dGVzQ29uZmlnJTBBJTBBcXVhbnRpemF0aW9uX2NvbmZpZyUyMCUzRCUyMEJpdHNBbmRCeXRlc0NvbmZpZyglMEElMjAlMjAlMjAlMjBsb2FkX2luXzRiaXQlM0RUcnVlJTJDJTBBJTIwJTIwJTIwJTIwYm5iXzRiaXRfY29tcHV0ZV9kdHlwZSUzRHRvcmNoLmZsb2F0MTYlMkMlMEElMjAlMjAlMjAlMjBibmJfNGJpdF91c2VfZG91YmxlX3F1YW50JTNEVHJ1ZSUyQyUwQSUyMCUyMCUyMCUyMGJuYl80Yml0X3F1YW50X3R5cGUlM0QlMjJuZjQlMjIlMEEpJTBBJTBBbW9kZWwlMjAlM0QlMjBBdXRvTW9kZWxGb3JDYXVzYWxMTS5mcm9tX3ByZXRyYWluZWQoJTBBJTIwJTIwJTIwJTIwJTIyYWxsZW5haSUyRk9MTW8tN0ItaGYlMjIlMkMlMEElMjAlMjAlMjAlMjBhdHRuX2ltcGxlbWVudGF0aW9uJTNEJTIyc2RwYSUyMiUyQyUwQSUyMCUyMCUyMCUyMGR0eXBlJTNEdG9yY2guZmxvYXQxNiUyQyUwQSUyMCUyMCUyMCUyMGRldmljZV9tYXAlM0QlMjJhdXRvJTIyJTJDJTBBJTIwJTIwJTIwJTIwcXVhbnRpemF0aW9uX2NvbmZpZyUzRHF1YW50aXphdGlvbl9jb25maWclMEEpJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIyYWxsZW5haSUyRk9MTW8tN0ItaGYlMjIpJTBBJTBBaW5wdXRzJTIwJTNEJTIwdG9rZW5pemVyKCUyMkJpdGNvaW4lMjBpcyUyMiUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpJTBBaW5wdXRzJTIwJTNEJTIwJTdCayUzQSUyMHYudG8obW9kZWwuZGV2aWNlKSUyMGZvciUyMGslMkMlMjB2JTIwaW4lMjBpbnB1dHMuaXRlbXMoKSU3RCUwQSUwQW91dHB1dCUyMCUzRCUyMG1vZGVsLmdlbmVyYXRlKCoqaW5wdXRzJTJDJTIwbWF4X2xlbmd0aCUzRDY0KSUwQSUwQXByaW50KHRva2VuaXplci5kZWNvZGUob3V0cHV0JTVCMCU1RCkp",highlighted:`<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=<span class="hljs-literal">True</span>,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=<span class="hljs-literal">True</span>,
    bnb_4bit_quant_type=<span class="hljs-string">&quot;nf4&quot;</span>
)

model = AutoModelForCausalLM.from_pretrained(
    <span class="hljs-string">&quot;allenai/OLMo-7B-hf&quot;</span>,
    attn_implementation=<span class="hljs-string">&quot;sdpa&quot;</span>,
    dtype=torch.float16,
    device_map=<span class="hljs-string">&quot;auto&quot;</span>,
    quantization_config=quantization_config
)

tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;allenai/OLMo-7B-hf&quot;</span>)

inputs = tokenizer(<span class="hljs-string">&quot;Bitcoin is&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
inputs = {k: v.to(model.device) <span class="hljs-keyword">for</span> k, v <span class="hljs-keyword">in</span> inputs.items()}

output = model.generate(**inputs, max_length=<span class="hljs-number">64</span>)

<span class="hljs-built_in">print</span>(tokenizer.decode(output[<span class="hljs-number">0</span>]))`,wrap:!1}}),A=new qe({props:{title:"OlmoConfig",local:"transformers.OlmoConfig",headingTag:"h2"}}),S=new be({props:{name:"class transformers.OlmoConfig",anchor:"transformers.OlmoConfig",parameters:[{name:"vocab_size",val:" = 50304"},{name:"hidden_size",val:" = 4096"},{name:"intermediate_size",val:" = 11008"},{name:"num_hidden_layers",val:" = 32"},{name:"num_attention_heads",val:" = 32"},{name:"num_key_value_heads",val:" = None"},{name:"hidden_act",val:" = 'silu'"},{name:"max_position_embeddings",val:" = 2048"},{name:"initializer_range",val:" = 0.02"},{name:"use_cache",val:" = True"},{name:"pad_token_id",val:" = 1"},{name:"bos_token_id",val:" = None"},{name:"eos_token_id",val:" = 50279"},{name:"tie_word_embeddings",val:" = False"},{name:"rope_theta",val:" = 10000.0"},{name:"rope_scaling",val:" = None"},{name:"attention_bias",val:" = False"},{name:"attention_dropout",val:" = 0.0"},{name:"clip_qkv",val:" = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.OlmoConfig.vocab_size",description:`<strong>vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 50304) &#x2014;
Vocabulary size of the OLMo model. Defines the number of different tokens that can be represented by the
<code>inputs_ids</code> passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/olmo#transformers.OlmoModel">OlmoModel</a>`,name:"vocab_size"},{anchor:"transformers.OlmoConfig.hidden_size",description:`<strong>hidden_size</strong> (<code>int</code>, <em>optional</em>, defaults to 4096) &#x2014;
Dimension of the hidden representations.`,name:"hidden_size"},{anchor:"transformers.OlmoConfig.intermediate_size",description:`<strong>intermediate_size</strong> (<code>int</code>, <em>optional</em>, defaults to 11008) &#x2014;
Dimension of the MLP representations.`,name:"intermediate_size"},{anchor:"transformers.OlmoConfig.num_hidden_layers",description:`<strong>num_hidden_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 32) &#x2014;
Number of hidden layers in the Transformer decoder.`,name:"num_hidden_layers"},{anchor:"transformers.OlmoConfig.num_attention_heads",description:`<strong>num_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 32) &#x2014;
Number of attention heads for each attention layer in the Transformer decoder.`,name:"num_attention_heads"},{anchor:"transformers.OlmoConfig.num_key_value_heads",description:`<strong>num_key_value_heads</strong> (<code>int</code>, <em>optional</em>) &#x2014;
This is the number of key_value heads that should be used to implement Grouped Query Attention. If
<code>num_key_value_heads=num_attention_heads</code>, the model will use Multi Head Attention (MHA), if
<code>num_key_value_heads=1</code> the model will use Multi Query Attention (MQA) otherwise GQA is used. When
converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
by meanpooling all the original heads within that group. For more details, check out <a href="https://huggingface.co/papers/2305.13245" rel="nofollow">this
paper</a>. If it is not specified, will default to
<code>num_attention_heads</code>.`,name:"num_key_value_heads"},{anchor:"transformers.OlmoConfig.hidden_act",description:`<strong>hidden_act</strong> (<code>str</code> or <code>function</code>, <em>optional</em>, defaults to <code>&quot;silu&quot;</code>) &#x2014;
The non-linear activation function (function or string) in the decoder.`,name:"hidden_act"},{anchor:"transformers.OlmoConfig.max_position_embeddings",description:`<strong>max_position_embeddings</strong> (<code>int</code>, <em>optional</em>, defaults to 2048) &#x2014;
The maximum sequence length that this model might ever be used with.`,name:"max_position_embeddings"},{anchor:"transformers.OlmoConfig.initializer_range",description:`<strong>initializer_range</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.`,name:"initializer_range"},{anchor:"transformers.OlmoConfig.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not the model should return the last key/values attentions (not used by all models). Only
relevant if <code>config.is_decoder=True</code>.`,name:"use_cache"},{anchor:"transformers.OlmoConfig.pad_token_id",description:`<strong>pad_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 1) &#x2014;
Padding token id.`,name:"pad_token_id"},{anchor:"transformers.OlmoConfig.bos_token_id",description:`<strong>bos_token_id</strong> (<code>int</code>, <em>optional</em>) &#x2014;
Beginning of stream token id.`,name:"bos_token_id"},{anchor:"transformers.OlmoConfig.eos_token_id",description:`<strong>eos_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 50279) &#x2014;
End of stream token id.`,name:"eos_token_id"},{anchor:"transformers.OlmoConfig.tie_word_embeddings",description:`<strong>tie_word_embeddings</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether to tie weight embeddings`,name:"tie_word_embeddings"},{anchor:"transformers.OlmoConfig.rope_theta",description:`<strong>rope_theta</strong> (<code>float</code>, <em>optional</em>, defaults to 10000.0) &#x2014;
The base period of the RoPE embeddings.`,name:"rope_theta"},{anchor:"transformers.OlmoConfig.rope_scaling",description:`<strong>rope_scaling</strong> (<code>Dict</code>, <em>optional</em>) &#x2014;
Dictionary containing the scaling configuration for the RoPE embeddings. Currently supports two scaling
strategies: linear and dynamic. Their scaling factor must be a float greater than 1. The expected format is
<code>{&quot;type&quot;: strategy name, &quot;factor&quot;: scaling factor}</code>. When using this flag, don&#x2019;t update
<code>max_position_embeddings</code> to the expected new maximum. See the following thread for more information on how
these scaling strategies behave:
<a href="https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically_scaled_rope_further_increases/" rel="nofollow">https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically_scaled_rope_further_increases/</a>. This is an
experimental feature, subject to breaking API changes in future versions.`,name:"rope_scaling"},{anchor:"transformers.OlmoConfig.attention_bias",description:`<strong>attention_bias</strong> (<code>bool</code>, defaults to <code>False</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether to use a bias in the query, key, value and output projection layers during self-attention.`,name:"attention_bias"},{anchor:"transformers.OlmoConfig.attention_dropout",description:`<strong>attention_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The dropout ratio for the attention probabilities.`,name:"attention_dropout"},{anchor:"transformers.OlmoConfig.clip_qkv",description:`<strong>clip_qkv</strong> (<code>float</code>, <em>optional</em>) &#x2014;
If not <code>None</code>, elements of query, key and value attention states are clipped so that their
absolute value does not exceed this value.`,name:"clip_qkv"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/olmo/configuration_olmo.py#L29"}}),q=new Mt({props:{anchor:"transformers.OlmoConfig.example",$$slots:{default:[Wt]},$$scope:{ctx:k}}}),D=new qe({props:{title:"OlmoModel",local:"transformers.OlmoModel",headingTag:"h2"}}),K=new be({props:{name:"class transformers.OlmoModel",anchor:"transformers.OlmoModel",parameters:[{name:"config",val:": OlmoConfig"}],parametersDescription:[{anchor:"transformers.OlmoModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/olmo#transformers.OlmoConfig">OlmoConfig</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/olmo/modeling_olmo.py#L311"}}),ee=new be({props:{name:"forward",anchor:"transformers.OlmoModel.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.OlmoModel.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.OlmoModel.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.OlmoModel.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.OlmoModel.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.OlmoModel.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.OlmoModel.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"},{anchor:"transformers.OlmoModel.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/olmo/modeling_olmo.py#L328",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast"
>transformers.modeling_outputs.BaseModelOutputWithPast</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/olmo#transformers.OlmoConfig"
>OlmoConfig</a>) and inputs.</p>
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
`}}),F=new tt({props:{$$slots:{default:[Bt]},$$scope:{ctx:k}}}),te=new qe({props:{title:"OlmoForCausalLM",local:"transformers.OlmoForCausalLM",headingTag:"h2"}}),oe=new be({props:{name:"class transformers.OlmoForCausalLM",anchor:"transformers.OlmoForCausalLM",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.OlmoForCausalLM.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/olmo#transformers.OlmoForCausalLM">OlmoForCausalLM</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/olmo/modeling_olmo.py#L390"}}),ne=new be({props:{name:"forward",anchor:"transformers.OlmoForCausalLM.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"},{name:"logits_to_keep",val:": typing.Union[int, torch.Tensor] = 0"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.OlmoForCausalLM.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.OlmoForCausalLM.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.OlmoForCausalLM.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.OlmoForCausalLM.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.OlmoForCausalLM.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.OlmoForCausalLM.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.OlmoForCausalLM.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.OlmoForCausalLM.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"},{anchor:"transformers.OlmoForCausalLM.forward.logits_to_keep",description:`<strong>logits_to_keep</strong> (<code>Union[int, torch.Tensor]</code>, defaults to <code>0</code>) &#x2014;
If an <code>int</code>, compute logits for the last <code>logits_to_keep</code> tokens. If <code>0</code>, calculate logits for all
<code>input_ids</code> (special case). Only last token logits are needed for generation, and calculating them only for that
token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
If a <code>torch.Tensor</code>, must be 1D corresponding to the indices to keep in the sequence length dimension.
This is useful when using packed tensor format (single dimension for batch and sequence length).`,name:"logits_to_keep"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/olmo/modeling_olmo.py#L404",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast"
>transformers.modeling_outputs.CausalLMOutputWithPast</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/olmo#transformers.OlmoConfig"
>OlmoConfig</a>) and inputs.</p>
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
`}}),R=new tt({props:{$$slots:{default:[Ot]},$$scope:{ctx:k}}}),G=new Mt({props:{anchor:"transformers.OlmoForCausalLM.forward.example",$$slots:{default:[Lt]},$$scope:{ctx:k}}}),se=new Ct({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/olmo.md"}}),{c(){t=p("meta"),a=l(),o=p("p"),c=l(),T=p("p"),T.innerHTML=m,r=l(),v=p("div"),v.innerHTML=ye,E=l(),f(W.$$.fragment),Te=l(),H=p("p"),H.innerHTML=nt,we=l(),V=p("p"),V.innerHTML=st,ve=l(),f(L.$$.fragment),ke=l(),P=p("p"),P.innerHTML=at,$e=l(),f(Z.$$.fragment),Ce=l(),Q=p("p"),Q.innerHTML=rt,Je=l(),Y=p("p"),Y.innerHTML=it,Ue=l(),f(N.$$.fragment),je=l(),f(A.$$.fragment),ze=l(),J=p("div"),f(S.$$.fragment),Fe=l(),ie=p("p"),ie.innerHTML=lt,Re=l(),le=p("p"),le.innerHTML=dt,Ge=l(),f(q.$$.fragment),Ie=l(),f(D.$$.fragment),xe=l(),$=p("div"),f(K.$$.fragment),Xe=l(),de=p("p"),de.textContent=ct,Ee=l(),ce=p("p"),ce.innerHTML=mt,He=l(),me=p("p"),me.innerHTML=pt,Ve=l(),B=p("div"),f(ee.$$.fragment),Pe=l(),pe=p("p"),pe.innerHTML=ut,Qe=l(),f(F.$$.fragment),We=l(),f(te.$$.fragment),Be=l(),C=p("div"),f(oe.$$.fragment),Ye=l(),ue=p("p"),ue.textContent=ht,Ne=l(),he=p("p"),he.innerHTML=ft,Ae=l(),fe=p("p"),fe.innerHTML=gt,Se=l(),z=p("div"),f(ne.$$.fragment),De=l(),ge=p("p"),ge.innerHTML=_t,Ke=l(),f(R.$$.fragment),et=l(),f(G.$$.fragment),Oe=l(),f(se.$$.fragment),Le=l(),Me=p("p"),this.h()},l(e){const n=kt("svelte-u9bgzb",document.head);t=u(n,"META",{name:!0,content:!0}),n.forEach(s),a=d(e),o=u(e,"P",{}),ae(o).forEach(s),c=d(e),T=u(e,"P",{"data-svelte-h":!0}),w(T)!=="svelte-i3wkaj"&&(T.innerHTML=m),r=d(e),v=u(e,"DIV",{style:!0,"data-svelte-h":!0}),w(v)!=="svelte-11gpmgv"&&(v.innerHTML=ye),E=d(e),g(W.$$.fragment,e),Te=d(e),H=u(e,"P",{"data-svelte-h":!0}),w(H)!=="svelte-1skjqw3"&&(H.innerHTML=nt),we=d(e),V=u(e,"P",{"data-svelte-h":!0}),w(V)!=="svelte-1pkgehx"&&(V.innerHTML=st),ve=d(e),g(L.$$.fragment,e),ke=d(e),P=u(e,"P",{"data-svelte-h":!0}),w(P)!=="svelte-c361bk"&&(P.innerHTML=at),$e=d(e),g(Z.$$.fragment,e),Ce=d(e),Q=u(e,"P",{"data-svelte-h":!0}),w(Q)!=="svelte-nf5ooi"&&(Q.innerHTML=rt),Je=d(e),Y=u(e,"P",{"data-svelte-h":!0}),w(Y)!=="svelte-60nsd0"&&(Y.innerHTML=it),Ue=d(e),g(N.$$.fragment,e),je=d(e),g(A.$$.fragment,e),ze=d(e),J=u(e,"DIV",{class:!0});var I=ae(J);g(S.$$.fragment,I),Fe=d(I),ie=u(I,"P",{"data-svelte-h":!0}),w(ie)!=="svelte-zyn0fq"&&(ie.innerHTML=lt),Re=d(I),le=u(I,"P",{"data-svelte-h":!0}),w(le)!=="svelte-1ek1ss9"&&(le.innerHTML=dt),Ge=d(I),g(q.$$.fragment,I),I.forEach(s),Ie=d(e),g(D.$$.fragment,e),xe=d(e),$=u(e,"DIV",{class:!0});var U=ae($);g(K.$$.fragment,U),Xe=d(U),de=u(U,"P",{"data-svelte-h":!0}),w(de)!=="svelte-vkg0hb"&&(de.textContent=ct),Ee=d(U),ce=u(U,"P",{"data-svelte-h":!0}),w(ce)!=="svelte-q52n56"&&(ce.innerHTML=mt),He=d(U),me=u(U,"P",{"data-svelte-h":!0}),w(me)!=="svelte-hswkmf"&&(me.innerHTML=pt),Ve=d(U),B=u(U,"DIV",{class:!0});var O=ae(B);g(ee.$$.fragment,O),Pe=d(O),pe=u(O,"P",{"data-svelte-h":!0}),w(pe)!=="svelte-1ihypw8"&&(pe.innerHTML=ut),Qe=d(O),g(F.$$.fragment,O),O.forEach(s),U.forEach(s),We=d(e),g(te.$$.fragment,e),Be=d(e),C=u(e,"DIV",{class:!0});var j=ae(C);g(oe.$$.fragment,j),Ye=d(j),ue=u(j,"P",{"data-svelte-h":!0}),w(ue)!=="svelte-a11r7i"&&(ue.textContent=ht),Ne=d(j),he=u(j,"P",{"data-svelte-h":!0}),w(he)!=="svelte-q52n56"&&(he.innerHTML=ft),Ae=d(j),fe=u(j,"P",{"data-svelte-h":!0}),w(fe)!=="svelte-hswkmf"&&(fe.innerHTML=gt),Se=d(j),z=u(j,"DIV",{class:!0});var x=ae(z);g(ne.$$.fragment,x),De=d(x),ge=u(x,"P",{"data-svelte-h":!0}),w(ge)!=="svelte-5e1uuo"&&(ge.innerHTML=_t),Ke=d(x),g(R.$$.fragment,x),et=d(x),g(G.$$.fragment,x),x.forEach(s),j.forEach(s),Oe=d(e),g(se.$$.fragment,e),Le=d(e),Me=u(e,"P",{}),ae(Me).forEach(s),this.h()},h(){re(t,"name","hf:doc:metadata"),re(t,"content",qt),$t(v,"float","right"),re(J,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),re(B,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),re($,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),re(z,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),re(C,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,n){h(document.head,t),i(e,a,n),i(e,o,n),i(e,c,n),i(e,T,n),i(e,r,n),i(e,v,n),i(e,E,n),_(W,e,n),i(e,Te,n),i(e,H,n),i(e,we,n),i(e,V,n),i(e,ve,n),_(L,e,n),i(e,ke,n),i(e,P,n),i(e,$e,n),_(Z,e,n),i(e,Ce,n),i(e,Q,n),i(e,Je,n),i(e,Y,n),i(e,Ue,n),_(N,e,n),i(e,je,n),_(A,e,n),i(e,ze,n),i(e,J,n),_(S,J,null),h(J,Fe),h(J,ie),h(J,Re),h(J,le),h(J,Ge),_(q,J,null),i(e,Ie,n),_(D,e,n),i(e,xe,n),i(e,$,n),_(K,$,null),h($,Xe),h($,de),h($,Ee),h($,ce),h($,He),h($,me),h($,Ve),h($,B),_(ee,B,null),h(B,Pe),h(B,pe),h(B,Qe),_(F,B,null),i(e,We,n),_(te,e,n),i(e,Be,n),i(e,C,n),_(oe,C,null),h(C,Ye),h(C,ue),h(C,Ne),h(C,he),h(C,Ae),h(C,fe),h(C,Se),h(C,z),_(ne,z,null),h(z,De),h(z,ge),h(z,Ke),_(R,z,null),h(z,et),_(G,z,null),i(e,Oe,n),_(se,e,n),i(e,Le,n),i(e,Me,n),Ze=!0},p(e,[n]){const I={};n&2&&(I.$$scope={dirty:n,ctx:e}),L.$set(I);const U={};n&2&&(U.$$scope={dirty:n,ctx:e}),Z.$set(U);const O={};n&2&&(O.$$scope={dirty:n,ctx:e}),q.$set(O);const j={};n&2&&(j.$$scope={dirty:n,ctx:e}),F.$set(j);const x={};n&2&&(x.$$scope={dirty:n,ctx:e}),R.$set(x);const yt={};n&2&&(yt.$$scope={dirty:n,ctx:e}),G.$set(yt)},i(e){Ze||(y(W.$$.fragment,e),y(L.$$.fragment,e),y(Z.$$.fragment,e),y(N.$$.fragment,e),y(A.$$.fragment,e),y(S.$$.fragment,e),y(q.$$.fragment,e),y(D.$$.fragment,e),y(K.$$.fragment,e),y(ee.$$.fragment,e),y(F.$$.fragment,e),y(te.$$.fragment,e),y(oe.$$.fragment,e),y(ne.$$.fragment,e),y(R.$$.fragment,e),y(G.$$.fragment,e),y(se.$$.fragment,e),Ze=!0)},o(e){M(W.$$.fragment,e),M(L.$$.fragment,e),M(Z.$$.fragment,e),M(N.$$.fragment,e),M(A.$$.fragment,e),M(S.$$.fragment,e),M(q.$$.fragment,e),M(D.$$.fragment,e),M(K.$$.fragment,e),M(ee.$$.fragment,e),M(F.$$.fragment,e),M(te.$$.fragment,e),M(oe.$$.fragment,e),M(ne.$$.fragment,e),M(R.$$.fragment,e),M(G.$$.fragment,e),M(se.$$.fragment,e),Ze=!1},d(e){e&&(s(a),s(o),s(c),s(T),s(r),s(v),s(E),s(Te),s(H),s(we),s(V),s(ve),s(ke),s(P),s($e),s(Ce),s(Q),s(Je),s(Y),s(Ue),s(je),s(ze),s(J),s(Ie),s(xe),s($),s(We),s(Be),s(C),s(Oe),s(Le),s(Me)),s(t),b(W,e),b(L,e),b(Z,e),b(N,e),b(A,e),b(S),b(q),b(D,e),b(K),b(ee),b(F),b(te,e),b(oe),b(ne),b(R),b(G),b(se,e)}}}const qt='{"title":"OLMo","local":"olmo","sections":[{"title":"OlmoConfig","local":"transformers.OlmoConfig","sections":[],"depth":2},{"title":"OlmoModel","local":"transformers.OlmoModel","sections":[],"depth":2},{"title":"OlmoForCausalLM","local":"transformers.OlmoForCausalLM","sections":[],"depth":2}],"depth":1}';function Ft(k){return Tt(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class Yt extends wt{constructor(t){super(),vt(this,t,Ft,Zt,bt,{})}}export{Yt as component};
