import{s as Zt,o as Wt,n as N}from"../chunks/scheduler.18a86fab.js";import{S as Ft,i as Et,g as m,s as a,r as u,A as qt,h as p,f as s,c as r,j as W,x as b,u as f,k as pe,l as Gt,y as c,a as i,v as g,d as _,t as M,w as y}from"../chunks/index.98837b22.js";import{T as ft}from"../chunks/Tip.77304350.js";import{D as Ue}from"../chunks/Docstring.a1ef7999.js";import{C as he}from"../chunks/CodeBlock.8d0c2e8a.js";import{E as Ot}from"../chunks/ExampleCodeBlock.8c3ee1f9.js";import{H as xe,E as Ht}from"../chunks/getInferenceSnippets.06c2775f.js";import{H as Rt,a as gt}from"../chunks/HfOption.6641485e.js";function Xt(k){let t,l="Click on the OLMo2 models in the right sidebar for more examples of how to apply OLMo2 to different language tasks.";return{c(){t=m("p"),t.textContent=l},l(o){t=p(o,"P",{"data-svelte-h":!0}),b(t)!=="svelte-ub048x"&&(t.textContent=l)},m(o,h){i(o,t,h)},p:N,d(o){o&&s(t)}}}function Nt(k){let t,l;return t=new he({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwcGlwZWxpbmUlMEElMEFwaXBlJTIwJTNEJTIwcGlwZWxpbmUoJTBBJTIwJTIwJTIwJTIwdGFzayUzRCUyMnRleHQtZ2VuZXJhdGlvbiUyMiUyQyUwQSUyMCUyMCUyMCUyMG1vZGVsJTNEJTIyYWxsZW5haSUyRk9MTW8tMi0wNDI1LTFCJTIyJTJDJTBBJTIwJTIwJTIwJTIwZHR5cGUlM0R0b3JjaC5mbG9hdDE2JTJDJTBBJTIwJTIwJTIwJTIwZGV2aWNlJTNEMCUyQyUwQSklMEElMjAlMjAlMjAlMjAlMEFyZXN1bHQlMjAlM0QlMjBwaXBlKCUyMlBsYW50cyUyMGNyZWF0ZSUyMGVuZXJneSUyMHRocm91Z2glMjBhJTIwcHJvY2VzcyUyMGtub3duJTIwYXMlMjIpJTBBcHJpbnQocmVzdWx0KQ==",highlighted:`<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> pipeline

pipe = pipeline(
    task=<span class="hljs-string">&quot;text-generation&quot;</span>,
    model=<span class="hljs-string">&quot;allenai/OLMo-2-0425-1B&quot;</span>,
    dtype=torch.float16,
    device=<span class="hljs-number">0</span>,
)
    
result = pipe(<span class="hljs-string">&quot;Plants create energy through a process known as&quot;</span>)
<span class="hljs-built_in">print</span>(result)`,wrap:!1}}),{c(){u(t.$$.fragment)},l(o){f(t.$$.fragment,o)},m(o,h){g(t,o,h),l=!0},p:N,i(o){l||(_(t.$$.fragment,o),l=!0)},o(o){M(t.$$.fragment,o),l=!1},d(o){y(t,o)}}}function Pt(k){let t,l;return t=new he({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b01vZGVsRm9yQ2F1c2FsTE0lMkMlMjBBdXRvVG9rZW5pemVyJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTBBJTIwJTIwJTIwJTIwJTIyYWxsZW5haSUyRk9MTW8tMi0wNDI1LTFCJTIyJTBBKSUwQSUwQW1vZGVsJTIwJTNEJTIwQXV0b01vZGVsRm9yQ2F1c2FsTE0uZnJvbV9wcmV0cmFpbmVkKCUwQSUyMCUyMCUyMCUyMCUyMmFsbGVuYWklMkZPTE1vLTItMDQyNS0xQiUyMiUyQyUwQSUyMCUyMCUyMCUyMGR0eXBlJTNEdG9yY2guZmxvYXQxNiUyQyUwQSUyMCUyMCUyMCUyMGRldmljZV9tYXAlM0QlMjJhdXRvJTIyJTJDJTBBJTIwJTIwJTIwJTIwYXR0bl9pbXBsZW1lbnRhdGlvbiUzRCUyMnNkcGElMjIlMEEpJTBBaW5wdXRfaWRzJTIwJTNEJTIwdG9rZW5pemVyKCUyMlBsYW50cyUyMGNyZWF0ZSUyMGVuZXJneSUyMHRocm91Z2glMjBhJTIwcHJvY2VzcyUyMGtub3duJTIwYXMlMjIlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKS50byhtb2RlbC5kZXZpY2UpJTBBJTBBb3V0cHV0JTIwJTNEJTIwbW9kZWwuZ2VuZXJhdGUoKippbnB1dF9pZHMlMkMlMjBtYXhfbGVuZ3RoJTNENTAlMkMlMjBjYWNoZV9pbXBsZW1lbnRhdGlvbiUzRCUyMnN0YXRpYyUyMiklMEFwcmludCh0b2tlbml6ZXIuZGVjb2RlKG91dHB1dCU1QjAlNUQlMkMlMjBza2lwX3NwZWNpYWxfdG9rZW5zJTNEVHJ1ZSkp",highlighted:`<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    <span class="hljs-string">&quot;allenai/OLMo-2-0425-1B&quot;</span>
)

model = AutoModelForCausalLM.from_pretrained(
    <span class="hljs-string">&quot;allenai/OLMo-2-0425-1B&quot;</span>,
    dtype=torch.float16,
    device_map=<span class="hljs-string">&quot;auto&quot;</span>,
    attn_implementation=<span class="hljs-string">&quot;sdpa&quot;</span>
)
input_ids = tokenizer(<span class="hljs-string">&quot;Plants create energy through a process known as&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>).to(model.device)

output = model.generate(**input_ids, max_length=<span class="hljs-number">50</span>, cache_implementation=<span class="hljs-string">&quot;static&quot;</span>)
<span class="hljs-built_in">print</span>(tokenizer.decode(output[<span class="hljs-number">0</span>], skip_special_tokens=<span class="hljs-literal">True</span>))`,wrap:!1}}),{c(){u(t.$$.fragment)},l(o){f(t.$$.fragment,o)},m(o,h){g(t,o,h),l=!0},p:N,i(o){l||(_(t.$$.fragment,o),l=!0)},o(o){M(t.$$.fragment,o),l=!1},d(o){y(t,o)}}}function At(k){let t,l;return t=new he({props:{code:"ZWNobyUyMC1lJTIwJTIyUGxhbnRzJTIwY3JlYXRlJTIwZW5lcmd5JTIwdGhyb3VnaCUyMGElMjBwcm9jZXNzJTIwa25vd24lMjBhcyUyMiUyMCU3QyUyMHRyYW5zZm9ybWVycy1jbGklMjBydW4lMjAtLXRhc2slMjB0ZXh0LWdlbmVyYXRpb24lMjAtLW1vZGVsJTIwYWxsZW5haSUyRk9MTW8tMi0wNDI1LTFCJTIwLS1kZXZpY2UlMjAw",highlighted:'<span class="hljs-built_in">echo</span> -e <span class="hljs-string">&quot;Plants create energy through a process known as&quot;</span> | transformers-cli run --task text-generation --model allenai/OLMo-2-0425-1B --device 0',wrap:!1}}),{c(){u(t.$$.fragment)},l(o){f(t.$$.fragment,o)},m(o,h){g(t,o,h),l=!0},p:N,i(o){l||(_(t.$$.fragment,o),l=!0)},o(o){M(t.$$.fragment,o),l=!1},d(o){y(t,o)}}}function Qt(k){let t,l,o,h,w,T;return t=new gt({props:{id:"usage",option:"Pipeline",$$slots:{default:[Nt]},$$scope:{ctx:k}}}),o=new gt({props:{id:"usage",option:"AutoModel",$$slots:{default:[Pt]},$$scope:{ctx:k}}}),w=new gt({props:{id:"usage",option:"transformers CLI",$$slots:{default:[At]},$$scope:{ctx:k}}}),{c(){u(t.$$.fragment),l=a(),u(o.$$.fragment),h=a(),u(w.$$.fragment)},l(d){f(t.$$.fragment,d),l=r(d),f(o.$$.fragment,d),h=r(d),f(w.$$.fragment,d)},m(d,v){g(t,d,v),i(d,l,v),g(o,d,v),i(d,h,v),g(w,d,v),T=!0},p(d,v){const je={};v&2&&(je.$$scope={dirty:v,ctx:d}),t.$set(je);const P={};v&2&&(P.$$scope={dirty:v,ctx:d}),o.$set(P);const z={};v&2&&(z.$$scope={dirty:v,ctx:d}),w.$set(z)},i(d){T||(_(t.$$.fragment,d),_(o.$$.fragment,d),_(w.$$.fragment,d),T=!0)},o(d){M(t.$$.fragment,d),M(o.$$.fragment,d),M(w.$$.fragment,d),T=!1},d(d){d&&(s(l),s(h)),y(t,d),y(o,d),y(w,d)}}}function Vt(k){let t,l;return t=new he({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyME9sbW8yTW9kZWwlMkMlMjBPbG1vMkNvbmZpZyUwQSUwQSUyMyUyMEluaXRpYWxpemluZyUyMGElMjBPbG1vMiUyMDdCJTIwc3R5bGUlMjBjb25maWd1cmF0aW9uJTBBY29uZmlndXJhdGlvbiUyMCUzRCUyME9sbW8yQ29uZmlnKCklMEElMEElMjMlMjBJbml0aWFsaXppbmclMjBhJTIwbW9kZWwlMjBmcm9tJTIwdGhlJTIwT2xtbzIlMjA3QiUyMHN0eWxlJTIwY29uZmlndXJhdGlvbiUwQW1vZGVsJTIwJTNEJTIwT2xtbzJNb2RlbChjb25maWd1cmF0aW9uKSUwQSUwQSUyMyUyMEFjY2Vzc2luZyUyMHRoZSUyMG1vZGVsJTIwY29uZmlndXJhdGlvbiUwQWNvbmZpZ3VyYXRpb24lMjAlM0QlMjBtb2RlbC5jb25maWc=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> Olmo2Model, Olmo2Config

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a Olmo2 7B style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = Olmo2Config()

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a model from the Olmo2 7B style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = Olmo2Model(configuration)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Accessing the model configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = model.config`,wrap:!1}}),{c(){u(t.$$.fragment)},l(o){f(t.$$.fragment,o)},m(o,h){g(t,o,h),l=!0},p:N,i(o){l||(_(t.$$.fragment,o),l=!0)},o(o){M(t.$$.fragment,o),l=!1},d(o){y(t,o)}}}function Yt(k){let t,l=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=m("p"),t.innerHTML=l},l(o){t=p(o,"P",{"data-svelte-h":!0}),b(t)!=="svelte-fincs2"&&(t.innerHTML=l)},m(o,h){i(o,t,h)},p:N,d(o){o&&s(t)}}}function St(k){let t,l=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=m("p"),t.innerHTML=l},l(o){t=p(o,"P",{"data-svelte-h":!0}),b(t)!=="svelte-fincs2"&&(t.innerHTML=l)},m(o,h){i(o,t,h)},p:N,d(o){o&&s(t)}}}function Dt(k){let t,l="Example:",o,h,w;return h=new he({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBPbG1vMkZvckNhdXNhbExNJTBBJTBBbW9kZWwlMjAlM0QlMjBPbG1vMkZvckNhdXNhbExNLmZyb21fcHJldHJhaW5lZCglMjJtZXRhLW9sbW8yJTJGT2xtbzItMi03Yi1oZiUyMiklMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJtZXRhLW9sbW8yJTJGT2xtbzItMi03Yi1oZiUyMiklMEElMEFwcm9tcHQlMjAlM0QlMjAlMjJIZXklMkMlMjBhcmUlMjB5b3UlMjBjb25zY2lvdXMlM0YlMjBDYW4lMjB5b3UlMjB0YWxrJTIwdG8lMjBtZSUzRiUyMiUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplcihwcm9tcHQlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQSUwQSUyMyUyMEdlbmVyYXRlJTBBZ2VuZXJhdGVfaWRzJTIwJTNEJTIwbW9kZWwuZ2VuZXJhdGUoaW5wdXRzLmlucHV0X2lkcyUyQyUyMG1heF9sZW5ndGglM0QzMCklMEF0b2tlbml6ZXIuYmF0Y2hfZGVjb2RlKGdlbmVyYXRlX2lkcyUyQyUyMHNraXBfc3BlY2lhbF90b2tlbnMlM0RUcnVlJTJDJTIwY2xlYW5fdXBfdG9rZW5pemF0aW9uX3NwYWNlcyUzREZhbHNlKSU1QjAlNUQ=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, Olmo2ForCausalLM

<span class="hljs-meta">&gt;&gt;&gt; </span>model = Olmo2ForCausalLM.from_pretrained(<span class="hljs-string">&quot;meta-olmo2/Olmo2-2-7b-hf&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;meta-olmo2/Olmo2-2-7b-hf&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>prompt = <span class="hljs-string">&quot;Hey, are you conscious? Can you talk to me?&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(prompt, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Generate</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>generate_ids = model.generate(inputs.input_ids, max_length=<span class="hljs-number">30</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer.batch_decode(generate_ids, skip_special_tokens=<span class="hljs-literal">True</span>, clean_up_tokenization_spaces=<span class="hljs-literal">False</span>)[<span class="hljs-number">0</span>]
<span class="hljs-string">&quot;Hey, are you conscious? Can you talk to me?\\nI&#x27;m not conscious, but I can talk to you.&quot;</span>`,wrap:!1}}),{c(){t=m("p"),t.textContent=l,o=a(),u(h.$$.fragment)},l(T){t=p(T,"P",{"data-svelte-h":!0}),b(t)!=="svelte-11lpom8"&&(t.textContent=l),o=r(T),f(h.$$.fragment,T)},m(T,d){i(T,t,d),i(T,o,d),g(h,T,d),w=!0},p:N,i(T){w||(_(h.$$.fragment,T),w=!0)},o(T){M(h.$$.fragment,T),w=!1},d(T){T&&(s(t),s(o)),y(h,T)}}}function Kt(k){let t,l,o,h,w,T="<em>This model was released on 2024-12-31 and added to Hugging Face Transformers on 2024-11-25.</em>",d,v,je='<div class="flex flex-wrap space-x-1"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/> <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat"/> <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/></div>',P,z,Ie,A,_t='<a href="https://huggingface.co/papers/2501.00656" rel="nofollow">OLMo2</a> improves on <a href="./olmo">OLMo</a> by changing the architecture and training recipes of the original models. This includes excluding all biases to improve training stability, non-parametric layer norm, SwiGLU activation function, rotary positional embeddings, and a modified BPE-based tokenizer that masks personal identifiable information. It is pretrained on <a href="https://huggingface.co/datasets/allenai/dolma" rel="nofollow">Dolma</a>, a dataset of 3T tokens.',ze,Q,Mt='You can find all the original OLMo2 checkpoints under the <a href="https://huggingface.co/collections/allenai/olmo-2-674117b93ab84e98afc72edc" rel="nofollow">OLMo2</a> collection.',Le,F,Be,V,yt='The example below demonstrates how to generate text with <a href="/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.Pipeline">Pipeline</a>, <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel">AutoModel</a> and from the command line.',Oe,E,Ze,Y,bt='Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the <a href="../quantization/overview">Quantization</a> overview for more available quantization backends.',We,S,Tt='The example below uses <a href="../quantization/torchao">torchao</a> to only quantize the weights to 4-bits.',Fe,D,Ee,K,qe,L,ue,wt="<p>OLMo2 uses RMSNorm instead of standard layer norm. The RMSNorm is applied to attention queries and keys, and it is applied after the attention and feedforward layers rather than before.</p>",Ye,fe,vt="<p>OLMo2 requires Transformers v4.48 or higher.</p>",Se,ee,ge,kt='Load specific intermediate checkpoints by adding the <code>revision</code> parameter to <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a>.',De,te,Ge,oe,He,j,ne,Ke,_e,$t=`This is the configuration class to store the configuration of a <a href="/docs/transformers/v4.56.2/en/model_doc/olmo2#transformers.Olmo2Model">Olmo2Model</a>. It is used to instantiate an OLMo2
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the <a href="https://huggingface.co/allenai/Olmo2-7B-1124-hf" rel="nofollow">allenai/Olmo2-7B-1124-hf</a>.`,et,Me,Ct=`Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> and can be used to control the model outputs. Read the
documentation from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> for more information.`,tt,q,Re,se,Xe,$,ae,ot,ye,jt="The bare Olmo2 Model outputting raw hidden-states without any specific head on top.",nt,be,Jt=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,st,Te,Ut=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,at,B,re,rt,we,xt='The <a href="/docs/transformers/v4.56.2/en/model_doc/olmo2#transformers.Olmo2Model">Olmo2Model</a> forward method, overrides the <code>__call__</code> special method.',it,G,Ne,ie,Pe,C,le,lt,ve,It="The Olmo2 Model for causal language modeling.",dt,ke,zt=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,ct,$e,Lt=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,mt,x,de,pt,Ce,Bt='The <a href="/docs/transformers/v4.56.2/en/model_doc/olmo2#transformers.Olmo2ForCausalLM">Olmo2ForCausalLM</a> forward method, overrides the <code>__call__</code> special method.',ht,H,ut,R,Ae,ce,Qe,Je,Ve;return z=new xe({props:{title:"OLMo2",local:"olmo2",headingTag:"h1"}}),F=new ft({props:{warning:!1,$$slots:{default:[Xt]},$$scope:{ctx:k}}}),E=new Rt({props:{id:"usage",options:["Pipeline","AutoModel","transformers CLI"],$$slots:{default:[Qt]},$$scope:{ctx:k}}}),D=new he({props:{code:"JTBBJTIzcGlwJTIwaW5zdGFsbCUyMHRvcmNoYW8lMEFpbXBvcnQlMjB0b3JjaCUwQWZyb20lMjB0cmFuc2Zvcm1lcnMlMjBpbXBvcnQlMjBBdXRvTW9kZWxGb3JDYXVzYWxMTSUyQyUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBUb3JjaEFvQ29uZmlnJTBBJTBBdG9yY2hhb19jb25maWclMjAlM0QlMjBUb3JjaEFvQ29uZmlnKCUwQSUyMCUyMCUyMCUyMCUyMmludDRfd2VpZ2h0X29ubHklMjIlMkMlMEElMjAlMjAlMjAlMjBncm91cF9zaXplJTNEMTI4JTBBKSUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUwQSUyMCUyMCUyMCUyMCUyMmFsbGVuYWklMkZPTE1vLTItMDQyNS0xQiUyMiUwQSklMEElMEFtb2RlbCUyMCUzRCUyMEF1dG9Nb2RlbEZvckNhdXNhbExNLmZyb21fcHJldHJhaW5lZCglMEElMjAlMjAlMjAlMjAlMjJhbGxlbmFpJTJGT0xNby0yLTA0MjUtMUIlMjIlMkMlMEElMjAlMjAlMjAlMjBxdWFudGl6YXRpb25fY29uZmlnJTNEdG9yY2hhb19jb25maWclMkMlMEElMjAlMjAlMjAlMjBkdHlwZSUzRHRvcmNoLmJmbG9hdDE2JTJDJTBBJTIwJTIwJTIwJTIwZGV2aWNlX21hcCUzRCUyMmF1dG8lMjIlMkMlMEElMjAlMjAlMjAlMjBhdHRuX2ltcGxlbWVudGF0aW9uJTNEJTIyc2RwYSUyMiUwQSklMEFpbnB1dF9pZHMlMjAlM0QlMjB0b2tlbml6ZXIoJTIyUGxhbnRzJTIwY3JlYXRlJTIwZW5lcmd5JTIwdGhyb3VnaCUyMGElMjBwcm9jZXNzJTIwa25vd24lMjBhcyUyMiUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpLnRvKG1vZGVsLmRldmljZSklMEElMEFvdXRwdXQlMjAlM0QlMjBtb2RlbC5nZW5lcmF0ZSgqKmlucHV0X2lkcyUyQyUyMG1heF9sZW5ndGglM0Q1MCUyQyUyMGNhY2hlX2ltcGxlbWVudGF0aW9uJTNEJTIyc3RhdGljJTIyKSUwQXByaW50KHRva2VuaXplci5kZWNvZGUob3V0cHV0JTVCMCU1RCUyQyUyMHNraXBfc3BlY2lhbF90b2tlbnMlM0RUcnVlKSklMEE=",highlighted:`
<span class="hljs-comment">#pip install torchao</span>
<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoModelForCausalLM, AutoTokenizer, TorchAoConfig

torchao_config = TorchAoConfig(
    <span class="hljs-string">&quot;int4_weight_only&quot;</span>,
    group_size=<span class="hljs-number">128</span>
)

tokenizer = AutoTokenizer.from_pretrained(
    <span class="hljs-string">&quot;allenai/OLMo-2-0425-1B&quot;</span>
)

model = AutoModelForCausalLM.from_pretrained(
    <span class="hljs-string">&quot;allenai/OLMo-2-0425-1B&quot;</span>,
    quantization_config=torchao_config,
    dtype=torch.bfloat16,
    device_map=<span class="hljs-string">&quot;auto&quot;</span>,
    attn_implementation=<span class="hljs-string">&quot;sdpa&quot;</span>
)
input_ids = tokenizer(<span class="hljs-string">&quot;Plants create energy through a process known as&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>).to(model.device)

output = model.generate(**input_ids, max_length=<span class="hljs-number">50</span>, cache_implementation=<span class="hljs-string">&quot;static&quot;</span>)
<span class="hljs-built_in">print</span>(tokenizer.decode(output[<span class="hljs-number">0</span>], skip_special_tokens=<span class="hljs-literal">True</span>))
`,wrap:!1}}),K=new xe({props:{title:"Notes",local:"notes",headingTag:"h2"}}),te=new he({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Nb2RlbEZvckNhdXNhbExNJTBBJTBBbW9kZWwlMjAlM0QlMjBBdXRvTW9kZWxGb3JDYXVzYWxMTS5mcm9tX3ByZXRyYWluZWQoJTIyYWxsZW5haSUyRk9MTW8tMi0wNDI1LTFCJTIyJTJDJTIwcmV2aXNpb24lM0QlMjJzdGFnZTEtc3RlcDE0MDAwMC10b2tlbnMyOTRCJTIyKQ==",highlighted:`<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(<span class="hljs-string">&quot;allenai/OLMo-2-0425-1B&quot;</span>, revision=<span class="hljs-string">&quot;stage1-step140000-tokens294B&quot;</span>)`,wrap:!1}}),oe=new xe({props:{title:"Olmo2Config",local:"transformers.Olmo2Config",headingTag:"h2"}}),ne=new Ue({props:{name:"class transformers.Olmo2Config",anchor:"transformers.Olmo2Config",parameters:[{name:"vocab_size",val:" = 50304"},{name:"hidden_size",val:" = 4096"},{name:"intermediate_size",val:" = 11008"},{name:"num_hidden_layers",val:" = 32"},{name:"num_attention_heads",val:" = 32"},{name:"num_key_value_heads",val:" = None"},{name:"hidden_act",val:" = 'silu'"},{name:"max_position_embeddings",val:" = 2048"},{name:"initializer_range",val:" = 0.02"},{name:"use_cache",val:" = True"},{name:"pad_token_id",val:" = 1"},{name:"bos_token_id",val:" = None"},{name:"eos_token_id",val:" = 50279"},{name:"tie_word_embeddings",val:" = False"},{name:"rope_theta",val:" = 10000.0"},{name:"rope_scaling",val:" = None"},{name:"attention_bias",val:" = False"},{name:"attention_dropout",val:" = 0.0"},{name:"rms_norm_eps",val:" = 1e-05"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.Olmo2Config.vocab_size",description:`<strong>vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 50304) &#x2014;
Vocabulary size of the Olmo2 model. Defines the number of different tokens that can be represented by the
<code>inputs_ids</code> passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/olmo2#transformers.Olmo2Model">Olmo2Model</a>`,name:"vocab_size"},{anchor:"transformers.Olmo2Config.hidden_size",description:`<strong>hidden_size</strong> (<code>int</code>, <em>optional</em>, defaults to 4096) &#x2014;
Dimension of the hidden representations.`,name:"hidden_size"},{anchor:"transformers.Olmo2Config.intermediate_size",description:`<strong>intermediate_size</strong> (<code>int</code>, <em>optional</em>, defaults to 11008) &#x2014;
Dimension of the MLP representations.`,name:"intermediate_size"},{anchor:"transformers.Olmo2Config.num_hidden_layers",description:`<strong>num_hidden_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 32) &#x2014;
Number of hidden layers in the Transformer decoder.`,name:"num_hidden_layers"},{anchor:"transformers.Olmo2Config.num_attention_heads",description:`<strong>num_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 32) &#x2014;
Number of attention heads for each attention layer in the Transformer decoder.`,name:"num_attention_heads"},{anchor:"transformers.Olmo2Config.num_key_value_heads",description:`<strong>num_key_value_heads</strong> (<code>int</code>, <em>optional</em>) &#x2014;
This is the number of key_value heads that should be used to implement Grouped Query Attention. If
<code>num_key_value_heads=num_attention_heads</code>, the model will use Multi Head Attention (MHA), if
<code>num_key_value_heads=1</code> the model will use Multi Query Attention (MQA) otherwise GQA is used. When
converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
by meanpooling all the original heads within that group. For more details, check out <a href="https://huggingface.co/papers/2305.13245" rel="nofollow">this
paper</a>. If it is not specified, will default to
<code>num_attention_heads</code>.`,name:"num_key_value_heads"},{anchor:"transformers.Olmo2Config.hidden_act",description:`<strong>hidden_act</strong> (<code>str</code> or <code>function</code>, <em>optional</em>, defaults to <code>&quot;silu&quot;</code>) &#x2014;
The non-linear activation function (function or string) in the decoder.`,name:"hidden_act"},{anchor:"transformers.Olmo2Config.max_position_embeddings",description:`<strong>max_position_embeddings</strong> (<code>int</code>, <em>optional</em>, defaults to 2048) &#x2014;
The maximum sequence length that this model might ever be used with.`,name:"max_position_embeddings"},{anchor:"transformers.Olmo2Config.initializer_range",description:`<strong>initializer_range</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.`,name:"initializer_range"},{anchor:"transformers.Olmo2Config.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not the model should return the last key/values attentions (not used by all models). Only
relevant if <code>config.is_decoder=True</code>.`,name:"use_cache"},{anchor:"transformers.Olmo2Config.pad_token_id",description:`<strong>pad_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 1) &#x2014;
Padding token id.`,name:"pad_token_id"},{anchor:"transformers.Olmo2Config.bos_token_id",description:`<strong>bos_token_id</strong> (<code>int</code>, <em>optional</em>) &#x2014;
Beginning of stream token id.`,name:"bos_token_id"},{anchor:"transformers.Olmo2Config.eos_token_id",description:`<strong>eos_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 50279) &#x2014;
End of stream token id.`,name:"eos_token_id"},{anchor:"transformers.Olmo2Config.tie_word_embeddings",description:`<strong>tie_word_embeddings</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether to tie weight embeddings`,name:"tie_word_embeddings"},{anchor:"transformers.Olmo2Config.rope_theta",description:`<strong>rope_theta</strong> (<code>float</code>, <em>optional</em>, defaults to 10000.0) &#x2014;
The base period of the RoPE embeddings.`,name:"rope_theta"},{anchor:"transformers.Olmo2Config.rope_scaling",description:`<strong>rope_scaling</strong> (<code>Dict</code>, <em>optional</em>) &#x2014;
Dictionary containing the scaling configuration for the RoPE embeddings. Currently supports two scaling
strategies: linear and dynamic. Their scaling factor must be a float greater than 1. The expected format is
<code>{&quot;type&quot;: strategy name, &quot;factor&quot;: scaling factor}</code>. When using this flag, don&#x2019;t update
<code>max_position_embeddings</code> to the expected new maximum. See the following thread for more information on how
these scaling strategies behave:
<a href="https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically_scaled_rope_further_increases/" rel="nofollow">https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically_scaled_rope_further_increases/</a>. This is an
experimental feature, subject to breaking API changes in future versions.`,name:"rope_scaling"},{anchor:"transformers.Olmo2Config.attention_bias",description:`<strong>attention_bias</strong> (<code>bool</code>, defaults to <code>False</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether to use a bias in the query, key, value and output projection layers during self-attention.`,name:"attention_bias"},{anchor:"transformers.Olmo2Config.attention_dropout",description:`<strong>attention_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The dropout ratio for the attention probabilities.`,name:"attention_dropout"},{anchor:"transformers.Olmo2Config.rms_norm_eps",description:`<strong>rms_norm_eps</strong> (<code>float</code>, <em>optional</em>, defaults to 1e-05) &#x2014;
The epsilon used by the rms normalization layers.`,name:"rms_norm_eps"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/olmo2/configuration_olmo2.py#L11"}}),q=new Ot({props:{anchor:"transformers.Olmo2Config.example",$$slots:{default:[Vt]},$$scope:{ctx:k}}}),se=new xe({props:{title:"Olmo2Model",local:"transformers.Olmo2Model",headingTag:"h2"}}),ae=new Ue({props:{name:"class transformers.Olmo2Model",anchor:"transformers.Olmo2Model",parameters:[{name:"config",val:": Olmo2Config"}],parametersDescription:[{anchor:"transformers.Olmo2Model.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/olmo2#transformers.Olmo2Config">Olmo2Config</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/olmo2/modeling_olmo2.py#L316"}}),re=new Ue({props:{name:"forward",anchor:"transformers.Olmo2Model.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.Olmo2Model.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.Olmo2Model.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.Olmo2Model.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.Olmo2Model.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.Olmo2Model.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.Olmo2Model.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"},{anchor:"transformers.Olmo2Model.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/olmo2/modeling_olmo2.py#L333",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast"
>transformers.modeling_outputs.BaseModelOutputWithPast</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/olmo2#transformers.Olmo2Config"
>Olmo2Config</a>) and inputs.</p>
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
`}}),G=new ft({props:{$$slots:{default:[Yt]},$$scope:{ctx:k}}}),ie=new xe({props:{title:"Olmo2ForCausalLM",local:"transformers.Olmo2ForCausalLM",headingTag:"h2"}}),le=new Ue({props:{name:"class transformers.Olmo2ForCausalLM",anchor:"transformers.Olmo2ForCausalLM",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.Olmo2ForCausalLM.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/olmo2#transformers.Olmo2ForCausalLM">Olmo2ForCausalLM</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/olmo2/modeling_olmo2.py#L395"}}),de=new Ue({props:{name:"forward",anchor:"transformers.Olmo2ForCausalLM.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"},{name:"logits_to_keep",val:": typing.Union[int, torch.Tensor] = 0"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.Olmo2ForCausalLM.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.Olmo2ForCausalLM.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.Olmo2ForCausalLM.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.Olmo2ForCausalLM.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.Olmo2ForCausalLM.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.Olmo2ForCausalLM.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.Olmo2ForCausalLM.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.Olmo2ForCausalLM.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"},{anchor:"transformers.Olmo2ForCausalLM.forward.logits_to_keep",description:`<strong>logits_to_keep</strong> (<code>Union[int, torch.Tensor]</code>, defaults to <code>0</code>) &#x2014;
If an <code>int</code>, compute logits for the last <code>logits_to_keep</code> tokens. If <code>0</code>, calculate logits for all
<code>input_ids</code> (special case). Only last token logits are needed for generation, and calculating them only for that
token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
If a <code>torch.Tensor</code>, must be 1D corresponding to the indices to keep in the sequence length dimension.
This is useful when using packed tensor format (single dimension for batch and sequence length).`,name:"logits_to_keep"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/olmo2/modeling_olmo2.py#L409",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast"
>transformers.modeling_outputs.CausalLMOutputWithPast</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/olmo2#transformers.Olmo2Config"
>Olmo2Config</a>) and inputs.</p>
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
`}}),H=new ft({props:{$$slots:{default:[St]},$$scope:{ctx:k}}}),R=new Ot({props:{anchor:"transformers.Olmo2ForCausalLM.forward.example",$$slots:{default:[Dt]},$$scope:{ctx:k}}}),ce=new Ht({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/olmo2.md"}}),{c(){t=m("meta"),l=a(),o=m("p"),h=a(),w=m("p"),w.innerHTML=T,d=a(),v=m("div"),v.innerHTML=je,P=a(),u(z.$$.fragment),Ie=a(),A=m("p"),A.innerHTML=_t,ze=a(),Q=m("p"),Q.innerHTML=Mt,Le=a(),u(F.$$.fragment),Be=a(),V=m("p"),V.innerHTML=yt,Oe=a(),u(E.$$.fragment),Ze=a(),Y=m("p"),Y.innerHTML=bt,We=a(),S=m("p"),S.innerHTML=Tt,Fe=a(),u(D.$$.fragment),Ee=a(),u(K.$$.fragment),qe=a(),L=m("ul"),ue=m("li"),ue.innerHTML=wt,Ye=a(),fe=m("li"),fe.innerHTML=vt,Se=a(),ee=m("li"),ge=m("p"),ge.innerHTML=kt,De=a(),u(te.$$.fragment),Ge=a(),u(oe.$$.fragment),He=a(),j=m("div"),u(ne.$$.fragment),Ke=a(),_e=m("p"),_e.innerHTML=$t,et=a(),Me=m("p"),Me.innerHTML=Ct,tt=a(),u(q.$$.fragment),Re=a(),u(se.$$.fragment),Xe=a(),$=m("div"),u(ae.$$.fragment),ot=a(),ye=m("p"),ye.textContent=jt,nt=a(),be=m("p"),be.innerHTML=Jt,st=a(),Te=m("p"),Te.innerHTML=Ut,at=a(),B=m("div"),u(re.$$.fragment),rt=a(),we=m("p"),we.innerHTML=xt,it=a(),u(G.$$.fragment),Ne=a(),u(ie.$$.fragment),Pe=a(),C=m("div"),u(le.$$.fragment),lt=a(),ve=m("p"),ve.textContent=It,dt=a(),ke=m("p"),ke.innerHTML=zt,ct=a(),$e=m("p"),$e.innerHTML=Lt,mt=a(),x=m("div"),u(de.$$.fragment),pt=a(),Ce=m("p"),Ce.innerHTML=Bt,ht=a(),u(H.$$.fragment),ut=a(),u(R.$$.fragment),Ae=a(),u(ce.$$.fragment),Qe=a(),Je=m("p"),this.h()},l(e){const n=qt("svelte-u9bgzb",document.head);t=p(n,"META",{name:!0,content:!0}),n.forEach(s),l=r(e),o=p(e,"P",{}),W(o).forEach(s),h=r(e),w=p(e,"P",{"data-svelte-h":!0}),b(w)!=="svelte-x3hkwy"&&(w.innerHTML=T),d=r(e),v=p(e,"DIV",{style:!0,"data-svelte-h":!0}),b(v)!=="svelte-2m0t7r"&&(v.innerHTML=je),P=r(e),f(z.$$.fragment,e),Ie=r(e),A=p(e,"P",{"data-svelte-h":!0}),b(A)!=="svelte-1dmwccd"&&(A.innerHTML=_t),ze=r(e),Q=p(e,"P",{"data-svelte-h":!0}),b(Q)!=="svelte-1hzirno"&&(Q.innerHTML=Mt),Le=r(e),f(F.$$.fragment,e),Be=r(e),V=p(e,"P",{"data-svelte-h":!0}),b(V)!=="svelte-s90gxn"&&(V.innerHTML=yt),Oe=r(e),f(E.$$.fragment,e),Ze=r(e),Y=p(e,"P",{"data-svelte-h":!0}),b(Y)!=="svelte-nf5ooi"&&(Y.innerHTML=bt),We=r(e),S=p(e,"P",{"data-svelte-h":!0}),b(S)!=="svelte-1m04524"&&(S.innerHTML=Tt),Fe=r(e),f(D.$$.fragment,e),Ee=r(e),f(K.$$.fragment,e),qe=r(e),L=p(e,"UL",{});var O=W(L);ue=p(O,"LI",{"data-svelte-h":!0}),b(ue)!=="svelte-iposz9"&&(ue.innerHTML=wt),Ye=r(O),fe=p(O,"LI",{"data-svelte-h":!0}),b(fe)!=="svelte-33kh5s"&&(fe.innerHTML=vt),Se=r(O),ee=p(O,"LI",{});var me=W(ee);ge=p(me,"P",{"data-svelte-h":!0}),b(ge)!=="svelte-1fwrp49"&&(ge.innerHTML=kt),De=r(me),f(te.$$.fragment,me),me.forEach(s),O.forEach(s),Ge=r(e),f(oe.$$.fragment,e),He=r(e),j=p(e,"DIV",{class:!0});var I=W(j);f(ne.$$.fragment,I),Ke=r(I),_e=p(I,"P",{"data-svelte-h":!0}),b(_e)!=="svelte-sj5kqe"&&(_e.innerHTML=$t),et=r(I),Me=p(I,"P",{"data-svelte-h":!0}),b(Me)!=="svelte-1ek1ss9"&&(Me.innerHTML=Ct),tt=r(I),f(q.$$.fragment,I),I.forEach(s),Re=r(e),f(se.$$.fragment,e),Xe=r(e),$=p(e,"DIV",{class:!0});var J=W($);f(ae.$$.fragment,J),ot=r(J),ye=p(J,"P",{"data-svelte-h":!0}),b(ye)!=="svelte-1iafjtf"&&(ye.textContent=jt),nt=r(J),be=p(J,"P",{"data-svelte-h":!0}),b(be)!=="svelte-q52n56"&&(be.innerHTML=Jt),st=r(J),Te=p(J,"P",{"data-svelte-h":!0}),b(Te)!=="svelte-hswkmf"&&(Te.innerHTML=Ut),at=r(J),B=p(J,"DIV",{class:!0});var Z=W(B);f(re.$$.fragment,Z),rt=r(Z),we=p(Z,"P",{"data-svelte-h":!0}),b(we)!=="svelte-b8t752"&&(we.innerHTML=xt),it=r(Z),f(G.$$.fragment,Z),Z.forEach(s),J.forEach(s),Ne=r(e),f(ie.$$.fragment,e),Pe=r(e),C=p(e,"DIV",{class:!0});var U=W(C);f(le.$$.fragment,U),lt=r(U),ve=p(U,"P",{"data-svelte-h":!0}),b(ve)!=="svelte-1r3dw9c"&&(ve.textContent=It),dt=r(U),ke=p(U,"P",{"data-svelte-h":!0}),b(ke)!=="svelte-q52n56"&&(ke.innerHTML=zt),ct=r(U),$e=p(U,"P",{"data-svelte-h":!0}),b($e)!=="svelte-hswkmf"&&($e.innerHTML=Lt),mt=r(U),x=p(U,"DIV",{class:!0});var X=W(x);f(de.$$.fragment,X),pt=r(X),Ce=p(X,"P",{"data-svelte-h":!0}),b(Ce)!=="svelte-1jbrwti"&&(Ce.innerHTML=Bt),ht=r(X),f(H.$$.fragment,X),ut=r(X),f(R.$$.fragment,X),X.forEach(s),U.forEach(s),Ae=r(e),f(ce.$$.fragment,e),Qe=r(e),Je=p(e,"P",{}),W(Je).forEach(s),this.h()},h(){pe(t,"name","hf:doc:metadata"),pe(t,"content",eo),Gt(v,"float","right"),pe(j,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),pe(B,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),pe($,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),pe(x,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),pe(C,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,n){c(document.head,t),i(e,l,n),i(e,o,n),i(e,h,n),i(e,w,n),i(e,d,n),i(e,v,n),i(e,P,n),g(z,e,n),i(e,Ie,n),i(e,A,n),i(e,ze,n),i(e,Q,n),i(e,Le,n),g(F,e,n),i(e,Be,n),i(e,V,n),i(e,Oe,n),g(E,e,n),i(e,Ze,n),i(e,Y,n),i(e,We,n),i(e,S,n),i(e,Fe,n),g(D,e,n),i(e,Ee,n),g(K,e,n),i(e,qe,n),i(e,L,n),c(L,ue),c(L,Ye),c(L,fe),c(L,Se),c(L,ee),c(ee,ge),c(ee,De),g(te,ee,null),i(e,Ge,n),g(oe,e,n),i(e,He,n),i(e,j,n),g(ne,j,null),c(j,Ke),c(j,_e),c(j,et),c(j,Me),c(j,tt),g(q,j,null),i(e,Re,n),g(se,e,n),i(e,Xe,n),i(e,$,n),g(ae,$,null),c($,ot),c($,ye),c($,nt),c($,be),c($,st),c($,Te),c($,at),c($,B),g(re,B,null),c(B,rt),c(B,we),c(B,it),g(G,B,null),i(e,Ne,n),g(ie,e,n),i(e,Pe,n),i(e,C,n),g(le,C,null),c(C,lt),c(C,ve),c(C,dt),c(C,ke),c(C,ct),c(C,$e),c(C,mt),c(C,x),g(de,x,null),c(x,pt),c(x,Ce),c(x,ht),g(H,x,null),c(x,ut),g(R,x,null),i(e,Ae,n),g(ce,e,n),i(e,Qe,n),i(e,Je,n),Ve=!0},p(e,[n]){const O={};n&2&&(O.$$scope={dirty:n,ctx:e}),F.$set(O);const me={};n&2&&(me.$$scope={dirty:n,ctx:e}),E.$set(me);const I={};n&2&&(I.$$scope={dirty:n,ctx:e}),q.$set(I);const J={};n&2&&(J.$$scope={dirty:n,ctx:e}),G.$set(J);const Z={};n&2&&(Z.$$scope={dirty:n,ctx:e}),H.$set(Z);const U={};n&2&&(U.$$scope={dirty:n,ctx:e}),R.$set(U)},i(e){Ve||(_(z.$$.fragment,e),_(F.$$.fragment,e),_(E.$$.fragment,e),_(D.$$.fragment,e),_(K.$$.fragment,e),_(te.$$.fragment,e),_(oe.$$.fragment,e),_(ne.$$.fragment,e),_(q.$$.fragment,e),_(se.$$.fragment,e),_(ae.$$.fragment,e),_(re.$$.fragment,e),_(G.$$.fragment,e),_(ie.$$.fragment,e),_(le.$$.fragment,e),_(de.$$.fragment,e),_(H.$$.fragment,e),_(R.$$.fragment,e),_(ce.$$.fragment,e),Ve=!0)},o(e){M(z.$$.fragment,e),M(F.$$.fragment,e),M(E.$$.fragment,e),M(D.$$.fragment,e),M(K.$$.fragment,e),M(te.$$.fragment,e),M(oe.$$.fragment,e),M(ne.$$.fragment,e),M(q.$$.fragment,e),M(se.$$.fragment,e),M(ae.$$.fragment,e),M(re.$$.fragment,e),M(G.$$.fragment,e),M(ie.$$.fragment,e),M(le.$$.fragment,e),M(de.$$.fragment,e),M(H.$$.fragment,e),M(R.$$.fragment,e),M(ce.$$.fragment,e),Ve=!1},d(e){e&&(s(l),s(o),s(h),s(w),s(d),s(v),s(P),s(Ie),s(A),s(ze),s(Q),s(Le),s(Be),s(V),s(Oe),s(Ze),s(Y),s(We),s(S),s(Fe),s(Ee),s(qe),s(L),s(Ge),s(He),s(j),s(Re),s(Xe),s($),s(Ne),s(Pe),s(C),s(Ae),s(Qe),s(Je)),s(t),y(z,e),y(F,e),y(E,e),y(D,e),y(K,e),y(te),y(oe,e),y(ne),y(q),y(se,e),y(ae),y(re),y(G),y(ie,e),y(le),y(de),y(H),y(R),y(ce,e)}}}const eo='{"title":"OLMo2","local":"olmo2","sections":[{"title":"Notes","local":"notes","sections":[],"depth":2},{"title":"Olmo2Config","local":"transformers.Olmo2Config","sections":[],"depth":2},{"title":"Olmo2Model","local":"transformers.Olmo2Model","sections":[],"depth":2},{"title":"Olmo2ForCausalLM","local":"transformers.Olmo2ForCausalLM","sections":[],"depth":2}],"depth":1}';function to(k){return Wt(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class mo extends Ft{constructor(t){super(),Et(this,t,to,Kt,Zt,{})}}export{mo as component};
