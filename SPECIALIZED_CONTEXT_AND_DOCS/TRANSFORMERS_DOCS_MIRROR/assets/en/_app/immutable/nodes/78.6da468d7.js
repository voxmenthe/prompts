import{s as Po,o as Io,n as O}from"../chunks/scheduler.18a86fab.js";import{S as Uo,i as jo,g as c,s as a,r as u,A as Jo,h as d,f as i,c as r,j as I,x as m,u as f,k as U,l as Ho,y as s,a as p,v as g,d as _,t as b,w as y}from"../chunks/index.98837b22.js";import{T as Ye}from"../chunks/Tip.77304350.js";import{D as B}from"../chunks/Docstring.a1ef7999.js";import{C as Tt}from"../chunks/CodeBlock.8d0c2e8a.js";import{E as Lo}from"../chunks/ExampleCodeBlock.8c3ee1f9.js";import{H as Ae,E as Wo}from"../chunks/getInferenceSnippets.06c2775f.js";import{H as Bo,a as qo}from"../chunks/HfOption.6641485e.js";function Zo(w){let t,l="The Arcee model supports extended context with RoPE scaling and all standard transformers features including Flash Attention 2, SDPA, gradient checkpointing, and quantization support.";return{c(){t=c("p"),t.textContent=l},l(o){t=d(o,"P",{"data-svelte-h":!0}),m(t)!=="svelte-aymze3"&&(t.textContent=l)},m(o,h){p(o,t,h)},p:O,d(o){o&&i(t)}}}function No(w){let t,l;return t=new Tt({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwcGlwZWxpbmUlMEElMEFwaXBlbGluZSUyMCUzRCUyMHBpcGVsaW5lKCUwQSUyMCUyMCUyMCUyMHRhc2slM0QlMjJ0ZXh0LWdlbmVyYXRpb24lMjIlMkMlMEElMjAlMjAlMjAlMjBtb2RlbCUzRCUyMmFyY2VlLWFpJTJGQUZNLTQuNUIlMjIlMkMlMEElMjAlMjAlMjAlMjBkdHlwZSUzRHRvcmNoLmZsb2F0MTYlMkMlMEElMjAlMjAlMjAlMjBkZXZpY2UlM0QwJTBBKSUwQSUwQW91dHB1dCUyMCUzRCUyMHBpcGVsaW5lKCUyMlRoZSUyMGtleSUyMGlubm92YXRpb24lMjBpbiUyMEFyY2VlJTIwaXMlMjIpJTBBcHJpbnQob3V0cHV0JTVCMCU1RCU1QiUyMmdlbmVyYXRlZF90ZXh0JTIyJTVEKQ==",highlighted:`<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> pipeline

pipeline = pipeline(
    task=<span class="hljs-string">&quot;text-generation&quot;</span>,
    model=<span class="hljs-string">&quot;arcee-ai/AFM-4.5B&quot;</span>,
    dtype=torch.float16,
    device=<span class="hljs-number">0</span>
)

output = pipeline(<span class="hljs-string">&quot;The key innovation in Arcee is&quot;</span>)
<span class="hljs-built_in">print</span>(output[<span class="hljs-number">0</span>][<span class="hljs-string">&quot;generated_text&quot;</span>])`,wrap:!1}}),{c(){u(t.$$.fragment)},l(o){f(t.$$.fragment,o)},m(o,h){g(t,o,h),l=!0},p:O,i(o){l||(_(t.$$.fragment,o),l=!0)},o(o){b(t.$$.fragment,o),l=!1},d(o){y(t,o)}}}function Qo(w){let t,l;return t=new Tt({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Rva2VuaXplciUyQyUyMEFyY2VlRm9yQ2F1c2FsTE0lMEElMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJhcmNlZS1haSUyRkFGTS00LjVCJTIyKSUwQW1vZGVsJTIwJTNEJTIwQXJjZWVGb3JDYXVzYWxMTS5mcm9tX3ByZXRyYWluZWQoJTBBJTIwJTIwJTIwJTIwJTIyYXJjZWUtYWklMkZBRk0tNC41QiUyMiUyQyUwQSUyMCUyMCUyMCUyMGR0eXBlJTNEdG9yY2guZmxvYXQxNiUyQyUwQSUyMCUyMCUyMCUyMGRldmljZV9tYXAlM0QlMjJhdXRvJTIyJTBBKSUwQSUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplciglMjJUaGUlMjBrZXklMjBpbm5vdmF0aW9uJTIwaW4lMjBBcmNlZSUyMGlzJTIyJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiklMEF3aXRoJTIwdG9yY2gubm9fZ3JhZCgpJTNBJTBBJTIwJTIwJTIwJTIwb3V0cHV0cyUyMCUzRCUyMG1vZGVsLmdlbmVyYXRlKCoqaW5wdXRzJTJDJTIwbWF4X25ld190b2tlbnMlM0Q1MCklMEFwcmludCh0b2tlbml6ZXIuZGVjb2RlKG91dHB1dHMlNUIwJTVEJTJDJTIwc2tpcF9zcGVjaWFsX3Rva2VucyUzRFRydWUpKQ==",highlighted:`<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, ArceeForCausalLM

tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;arcee-ai/AFM-4.5B&quot;</span>)
model = ArceeForCausalLM.from_pretrained(
    <span class="hljs-string">&quot;arcee-ai/AFM-4.5B&quot;</span>,
    dtype=torch.float16,
    device_map=<span class="hljs-string">&quot;auto&quot;</span>
)

inputs = tokenizer(<span class="hljs-string">&quot;The key innovation in Arcee is&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-keyword">with</span> torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=<span class="hljs-number">50</span>)
<span class="hljs-built_in">print</span>(tokenizer.decode(outputs[<span class="hljs-number">0</span>], skip_special_tokens=<span class="hljs-literal">True</span>))`,wrap:!1}}),{c(){u(t.$$.fragment)},l(o){f(t.$$.fragment,o)},m(o,h){g(t,o,h),l=!0},p:O,i(o){l||(_(t.$$.fragment,o),l=!0)},o(o){b(t.$$.fragment,o),l=!1},d(o){y(t,o)}}}function Eo(w){let t,l,o,h;return t=new qo({props:{id:"usage",option:"Pipeline",$$slots:{default:[No]},$$scope:{ctx:w}}}),o=new qo({props:{id:"usage",option:"AutoModel",$$slots:{default:[Qo]},$$scope:{ctx:w}}}),{c(){u(t.$$.fragment),l=a(),u(o.$$.fragment)},l(v){f(t.$$.fragment,v),l=r(v),f(o.$$.fragment,v)},m(v,T){g(t,v,T),p(v,l,T),g(o,v,T),h=!0},p(v,T){const J={};T&2&&(J.$$scope={dirty:T,ctx:v}),t.$set(J);const H={};T&2&&(H.$$scope={dirty:T,ctx:v}),o.$set(H)},i(v){h||(_(t.$$.fragment,v),_(o.$$.fragment,v),h=!0)},o(v){b(t.$$.fragment,v),b(o.$$.fragment,v),h=!1},d(v){v&&i(l),y(t,v),y(o,v)}}}function So(w){let t,l;return t=new Tt({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEFyY2VlTW9kZWwlMkMlMjBBcmNlZUNvbmZpZyUwQSUwQSUyMyUyMEluaXRpYWxpemluZyUyMGFuJTIwQXJjZWUlMjBBRk0tNC41Qi1CYXNlJTIwc3R5bGUlMjBjb25maWd1cmF0aW9uJTBBY29uZmlndXJhdGlvbiUyMCUzRCUyMEFyY2VlQ29uZmlnKCklMEElMEElMjMlMjBJbml0aWFsaXppbmclMjBhJTIwbW9kZWwlMjBmcm9tJTIwdGhlJTIwQUZNLTQuNUItQmFzZSUyMHN0eWxlJTIwY29uZmlndXJhdGlvbiUwQW1vZGVsJTIwJTNEJTIwQXJjZWVNb2RlbChjb25maWd1cmF0aW9uKSUwQSUwQSUyMyUyMEFjY2Vzc2luZyUyMHRoZSUyMG1vZGVsJTIwY29uZmlndXJhdGlvbiUwQWNvbmZpZ3VyYXRpb24lMjAlM0QlMjBtb2RlbC5jb25maWc=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> ArceeModel, ArceeConfig

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing an Arcee AFM-4.5B-Base style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = ArceeConfig()

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a model from the AFM-4.5B-Base style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = ArceeModel(configuration)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Accessing the model configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = model.config`,wrap:!1}}),{c(){u(t.$$.fragment)},l(o){f(t.$$.fragment,o)},m(o,h){g(t,o,h),l=!0},p:O,i(o){l||(_(t.$$.fragment,o),l=!0)},o(o){b(t.$$.fragment,o),l=!1},d(o){y(t,o)}}}function Oo(w){let t,l=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=l},l(o){t=d(o,"P",{"data-svelte-h":!0}),m(t)!=="svelte-fincs2"&&(t.innerHTML=l)},m(o,h){p(o,t,h)},p:O,d(o){o&&i(t)}}}function Ro(w){let t,l=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=l},l(o){t=d(o,"P",{"data-svelte-h":!0}),m(t)!=="svelte-fincs2"&&(t.innerHTML=l)},m(o,h){p(o,t,h)},p:O,d(o){o&&i(t)}}}function Do(w){let t,l="Example:",o,h,v;return h=new Tt({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBBcmNlZUZvckNhdXNhbExNJTBBJTBBbW9kZWwlMjAlM0QlMjBBcmNlZUZvckNhdXNhbExNLmZyb21fcHJldHJhaW5lZCglMjJtZXRhLWFyY2VlJTJGQXJjZWUtMi03Yi1oZiUyMiklMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJtZXRhLWFyY2VlJTJGQXJjZWUtMi03Yi1oZiUyMiklMEElMEFwcm9tcHQlMjAlM0QlMjAlMjJIZXklMkMlMjBhcmUlMjB5b3UlMjBjb25zY2lvdXMlM0YlMjBDYW4lMjB5b3UlMjB0YWxrJTIwdG8lMjBtZSUzRiUyMiUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplcihwcm9tcHQlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQSUwQSUyMyUyMEdlbmVyYXRlJTBBZ2VuZXJhdGVfaWRzJTIwJTNEJTIwbW9kZWwuZ2VuZXJhdGUoaW5wdXRzLmlucHV0X2lkcyUyQyUyMG1heF9sZW5ndGglM0QzMCklMEF0b2tlbml6ZXIuYmF0Y2hfZGVjb2RlKGdlbmVyYXRlX2lkcyUyQyUyMHNraXBfc3BlY2lhbF90b2tlbnMlM0RUcnVlJTJDJTIwY2xlYW5fdXBfdG9rZW5pemF0aW9uX3NwYWNlcyUzREZhbHNlKSU1QjAlNUQ=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, ArceeForCausalLM

<span class="hljs-meta">&gt;&gt;&gt; </span>model = ArceeForCausalLM.from_pretrained(<span class="hljs-string">&quot;meta-arcee/Arcee-2-7b-hf&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;meta-arcee/Arcee-2-7b-hf&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>prompt = <span class="hljs-string">&quot;Hey, are you conscious? Can you talk to me?&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(prompt, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Generate</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>generate_ids = model.generate(inputs.input_ids, max_length=<span class="hljs-number">30</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer.batch_decode(generate_ids, skip_special_tokens=<span class="hljs-literal">True</span>, clean_up_tokenization_spaces=<span class="hljs-literal">False</span>)[<span class="hljs-number">0</span>]
<span class="hljs-string">&quot;Hey, are you conscious? Can you talk to me?\\nI&#x27;m not conscious, but I can talk to you.&quot;</span>`,wrap:!1}}),{c(){t=c("p"),t.textContent=l,o=a(),u(h.$$.fragment)},l(T){t=d(T,"P",{"data-svelte-h":!0}),m(t)!=="svelte-11lpom8"&&(t.textContent=l),o=r(T),f(h.$$.fragment,T)},m(T,J){p(T,t,J),p(T,o,J),g(h,T,J),v=!0},p:O,i(T){v||(_(h.$$.fragment,T),v=!0)},o(T){b(h.$$.fragment,T),v=!1},d(T){T&&(i(t),i(o)),y(h,T)}}}function Vo(w){let t,l=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=l},l(o){t=d(o,"P",{"data-svelte-h":!0}),m(t)!=="svelte-fincs2"&&(t.innerHTML=l)},m(o,h){p(o,t,h)},p:O,d(o){o&&i(t)}}}function Xo(w){let t,l=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=l},l(o){t=d(o,"P",{"data-svelte-h":!0}),m(t)!=="svelte-fincs2"&&(t.innerHTML=l)},m(o,h){p(o,t,h)},p:O,d(o){o&&i(t)}}}function Go(w){let t,l=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=l},l(o){t=d(o,"P",{"data-svelte-h":!0}),m(t)!=="svelte-fincs2"&&(t.innerHTML=l)},m(o,h){p(o,t,h)},p:O,d(o){o&&i(t)}}}function Yo(w){let t,l,o,h,v,T="<em>This model was released on 2025-06-18 and added to Hugging Face Transformers on 2025-06-24.</em>",J,H,no='<div class="flex flex-wrap space-x-1"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/> <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat"/> <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/></div>',et,ae,tt,re,so='<a href="https://www.arcee.ai/blog/deep-dive-afm-4-5b-the-first-arcee-foundational-model" rel="nofollow">Arcee</a> is a decoder-only transformer model based on the Llama architecture with a key modification: it uses ReLU² (ReLU-squared) activation in the MLP blocks instead of SiLU, following recent research showing improved training efficiency with squared activations. This architecture is designed for efficient training and inference while maintaining the proven stability of the Llama design.',ot,ie,ao="The Arcee model is architecturally similar to Llama but uses <code>x * relu(x)</code> in MLP layers for improved gradient flow and is optimized for efficiency in both training and inference scenarios.",nt,X,st,ce,ro='The example below demonstrates how to generate text with Arcee using <a href="/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.Pipeline">Pipeline</a> or the <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel">AutoModel</a>.',at,G,rt,de,it,k,le,wt,xe,io=`This is the configuration class to store the configuration of a <a href="/docs/transformers/v4.56.2/en/model_doc/arcee#transformers.ArceeModel">ArceeModel</a>. It is used to instantiate an Arcee
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the AFM-4.5B-Base.`,kt,Fe,co=`Pre-trained weights are available at
<a href="https://huggingface.co/arcee-ai/AFM-4.5B" rel="nofollow">arcee-ai/AFM-4.5B</a>
and were used to build the examples below.`,Mt,ze,lo=`Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> and can be used to control the model outputs. Read the
documentation from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> for more information.`,$t,Y,ct,pe,dt,M,he,Ct,Le,po="The bare Arcee Model outputting raw hidden-states without any specific head on top.",At,qe,ho=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,xt,Pe,mo=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Ft,Z,me,zt,Ie,uo='The <a href="/docs/transformers/v4.56.2/en/model_doc/arcee#transformers.ArceeModel">ArceeModel</a> forward method, overrides the <code>__call__</code> special method.',Lt,K,lt,ue,pt,$,fe,qt,Ue,fo="The Arcee Model for causal language modeling.",Pt,je,go=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,It,Je,_o=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Ut,j,ge,jt,He,bo='The <a href="/docs/transformers/v4.56.2/en/model_doc/arcee#transformers.ArceeForCausalLM">ArceeForCausalLM</a> forward method, overrides the <code>__call__</code> special method.',Jt,ee,Ht,te,ht,_e,mt,C,be,Wt,We,yo="The Arcee Model with a sequence classification/regression head on top e.g. for GLUE tasks.",Bt,Be,vo=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Zt,Ze,To=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Nt,N,ye,Qt,Ne,wo="The <code>GenericForSequenceClassification</code> forward method, overrides the <code>__call__</code> special method.",Et,oe,ut,ve,ft,A,Te,St,Qe,ko=`The Arcee transformer with a span classification head on top for extractive question-answering tasks like
SQuAD (a linear layer on top of the hidden-states output to compute <code>span start logits</code> and <code>span end logits</code>).`,Ot,Ee,Mo=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Rt,Se,$o=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Dt,Q,we,Vt,Oe,Co="The <code>GenericForQuestionAnswering</code> forward method, overrides the <code>__call__</code> special method.",Xt,ne,gt,ke,_t,x,Me,Gt,Re,Ao=`The Arcee transformer with a token classification head on top (a linear layer on top of the hidden-states
output) e.g. for Named-Entity-Recognition (NER) tasks.`,Yt,De,xo=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Kt,Ve,Fo=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,eo,E,$e,to,Xe,zo="The <code>GenericForTokenClassification</code> forward method, overrides the <code>__call__</code> special method.",oo,se,bt,Ce,yt,Ke,vt;return ae=new Ae({props:{title:"Arcee",local:"arcee",headingTag:"h1"}}),X=new Ye({props:{warning:!1,$$slots:{default:[Zo]},$$scope:{ctx:w}}}),G=new Bo({props:{id:"usage",options:["Pipeline","AutoModel"],$$slots:{default:[Eo]},$$scope:{ctx:w}}}),de=new Ae({props:{title:"ArceeConfig",local:"transformers.ArceeConfig",headingTag:"h2"}}),le=new B({props:{name:"class transformers.ArceeConfig",anchor:"transformers.ArceeConfig",parameters:[{name:"vocab_size",val:" = 32000"},{name:"hidden_size",val:" = 2560"},{name:"intermediate_size",val:" = 18432"},{name:"num_hidden_layers",val:" = 32"},{name:"num_attention_heads",val:" = 32"},{name:"num_key_value_heads",val:" = None"},{name:"hidden_act",val:" = 'relu2'"},{name:"max_position_embeddings",val:" = 4096"},{name:"initializer_range",val:" = 0.02"},{name:"rms_norm_eps",val:" = 1e-05"},{name:"use_cache",val:" = True"},{name:"pad_token_id",val:" = None"},{name:"bos_token_id",val:" = 128000"},{name:"eos_token_id",val:" = 128001"},{name:"tie_word_embeddings",val:" = False"},{name:"rope_theta",val:" = 10000.0"},{name:"rope_scaling",val:" = None"},{name:"attention_bias",val:" = False"},{name:"attention_dropout",val:" = 0.0"},{name:"mlp_bias",val:" = False"},{name:"head_dim",val:" = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.ArceeConfig.vocab_size",description:`<strong>vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 32000) &#x2014;
Vocabulary size of the Arcee model. Defines the number of different tokens that can be represented by the
<code>inputs_ids</code> passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/arcee#transformers.ArceeModel">ArceeModel</a>`,name:"vocab_size"},{anchor:"transformers.ArceeConfig.hidden_size",description:`<strong>hidden_size</strong> (<code>int</code>, <em>optional</em>, defaults to 2560) &#x2014;
Dimension of the hidden representations.`,name:"hidden_size"},{anchor:"transformers.ArceeConfig.intermediate_size",description:`<strong>intermediate_size</strong> (<code>int</code>, <em>optional</em>, defaults to 18432) &#x2014;
Dimension of the MLP representations.`,name:"intermediate_size"},{anchor:"transformers.ArceeConfig.num_hidden_layers",description:`<strong>num_hidden_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 32) &#x2014;
Number of hidden layers in the Transformer decoder.`,name:"num_hidden_layers"},{anchor:"transformers.ArceeConfig.num_attention_heads",description:`<strong>num_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 32) &#x2014;
Number of attention heads for each attention layer in the Transformer decoder.`,name:"num_attention_heads"},{anchor:"transformers.ArceeConfig.num_key_value_heads",description:`<strong>num_key_value_heads</strong> (<code>int</code>, <em>optional</em>) &#x2014;
This is the number of key_value heads that should be used to implement Grouped Query Attention. If
<code>num_key_value_heads=num_attention_heads</code>, the model will use Multi Head Attention (MHA), if
<code>num_key_value_heads=1</code> the model will use Multi Query Attention (MQA) otherwise GQA is used. When
converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
by meanpooling all the original heads within that group. For more details checkout <a href="https://huggingface.co/papers/2305.13245" rel="nofollow">this
paper</a>. If it is not specified, will default to
<code>num_attention_heads</code>.`,name:"num_key_value_heads"},{anchor:"transformers.ArceeConfig.hidden_act",description:`<strong>hidden_act</strong> (<code>str</code> or <code>function</code>, <em>optional</em>, defaults to <code>&quot;relu2&quot;</code>) &#x2014;
The non-linear activation function (function or string) in the decoder.`,name:"hidden_act"},{anchor:"transformers.ArceeConfig.max_position_embeddings",description:`<strong>max_position_embeddings</strong> (<code>int</code>, <em>optional</em>, defaults to 4096) &#x2014;
The maximum sequence length that this model might ever be used with. AFM-4.5B-Base supports up to 16384 tokens.`,name:"max_position_embeddings"},{anchor:"transformers.ArceeConfig.initializer_range",description:`<strong>initializer_range</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.`,name:"initializer_range"},{anchor:"transformers.ArceeConfig.rms_norm_eps",description:`<strong>rms_norm_eps</strong> (<code>float</code>, <em>optional</em>, defaults to 1e-05) &#x2014;
The epsilon used by the rms normalization layers.`,name:"rms_norm_eps"},{anchor:"transformers.ArceeConfig.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not the model should return the last key/values attentions (not used by all models). Only
relevant if <code>config.is_decoder=True</code>.`,name:"use_cache"},{anchor:"transformers.ArceeConfig.pad_token_id",description:`<strong>pad_token_id</strong> (<code>int</code>, <em>optional</em>) &#x2014;
Padding token id.`,name:"pad_token_id"},{anchor:"transformers.ArceeConfig.bos_token_id",description:`<strong>bos_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 128000) &#x2014;
Beginning of stream token id.`,name:"bos_token_id"},{anchor:"transformers.ArceeConfig.eos_token_id",description:`<strong>eos_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 128001) &#x2014;
End of stream token id.`,name:"eos_token_id"},{anchor:"transformers.ArceeConfig.tie_word_embeddings",description:`<strong>tie_word_embeddings</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether to tie weight embeddings`,name:"tie_word_embeddings"},{anchor:"transformers.ArceeConfig.rope_theta",description:`<strong>rope_theta</strong> (<code>float</code>, <em>optional</em>, defaults to 10000.0) &#x2014;
The base period of the RoPE embeddings.`,name:"rope_theta"},{anchor:"transformers.ArceeConfig.rope_scaling",description:`<strong>rope_scaling</strong> (<code>Dict</code>, <em>optional</em>) &#x2014;
Dictionary containing the scaling configuration for the RoPE embeddings. NOTE: if you apply new rope type
and you expect the model to work on longer <code>max_position_embeddings</code>, we recommend you to update this value
accordingly.
Expected contents:
<code>rope_type</code> (<code>str</code>):
The sub-variant of RoPE to use. Can be one of [&#x2018;default&#x2019;, &#x2018;yarn&#x2019;], with &#x2018;default&#x2019; being the original RoPE implementation.
<code>factor</code> (<code>float</code>, <em>optional</em>):
Used with all rope types except &#x2018;default&#x2019;. The scaling factor to apply to the RoPE embeddings. In
most scaling types, a <code>factor</code> of x will enable the model to handle sequences of length x <em>
original maximum pre-trained length.
<code>original_max_position_embeddings</code> (<code>int</code>, </em>optional<em>):
Used with &#x2018;yarn&#x2019;. The original max position embeddings used during pretraining.
<code>attention_factor</code> (<code>float</code>, </em>optional<em>):
Used with &#x2018;yarn&#x2019;. The scaling factor to be applied on the attention computation. If unspecified,
it defaults to value recommended by the implementation, using the <code>factor</code> field to infer the suggested value.
<code>beta_fast</code> (<code>float</code>, </em>optional<em>):
Only used with &#x2018;yarn&#x2019;. Parameter to set the boundary for extrapolation (only) in the linear
ramp function. If unspecified, it defaults to 32.
<code>beta_slow</code> (<code>float</code>, </em>optional*):
Only used with &#x2018;yarn&#x2019;. Parameter to set the boundary for interpolation (only) in the linear
ramp function. If unspecified, it defaults to 1.`,name:"rope_scaling"},{anchor:"transformers.ArceeConfig.attention_bias",description:`<strong>attention_bias</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether to use a bias in the query, key, value and output projection layers during self-attention.`,name:"attention_bias"},{anchor:"transformers.ArceeConfig.attention_dropout",description:`<strong>attention_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The dropout ratio for the attention probabilities.`,name:"attention_dropout"},{anchor:"transformers.ArceeConfig.mlp_bias",description:`<strong>mlp_bias</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether to use a bias in up_proj, down_proj and gate_proj layers in the MLP layers.`,name:"mlp_bias"},{anchor:"transformers.ArceeConfig.head_dim",description:`<strong>head_dim</strong> (<code>int</code>, <em>optional</em>) &#x2014;
The attention head dimension. If None, it will default to hidden_size // num_attention_heads`,name:"head_dim"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/arcee/configuration_arcee.py#L26"}}),Y=new Lo({props:{anchor:"transformers.ArceeConfig.example",$$slots:{default:[So]},$$scope:{ctx:w}}}),pe=new Ae({props:{title:"ArceeModel",local:"transformers.ArceeModel",headingTag:"h2"}}),he=new B({props:{name:"class transformers.ArceeModel",anchor:"transformers.ArceeModel",parameters:[{name:"config",val:": ArceeConfig"}],parametersDescription:[{anchor:"transformers.ArceeModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/arcee#transformers.ArceeConfig">ArceeConfig</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/arcee/modeling_arcee.py#L330"}}),me=new B({props:{name:"forward",anchor:"transformers.ArceeModel.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.ArceeModel.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.ArceeModel.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.ArceeModel.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.ArceeModel.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.ArceeModel.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.ArceeModel.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"},{anchor:"transformers.ArceeModel.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/arcee/modeling_arcee.py#L347",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast"
>transformers.modeling_outputs.BaseModelOutputWithPast</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/arcee#transformers.ArceeConfig"
>ArceeConfig</a>) and inputs.</p>
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
`}}),K=new Ye({props:{$$slots:{default:[Oo]},$$scope:{ctx:w}}}),ue=new Ae({props:{title:"ArceeForCausalLM",local:"transformers.ArceeForCausalLM",headingTag:"h2"}}),fe=new B({props:{name:"class transformers.ArceeForCausalLM",anchor:"transformers.ArceeForCausalLM",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.ArceeForCausalLM.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/arcee#transformers.ArceeForCausalLM">ArceeForCausalLM</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/arcee/modeling_arcee.py#L409"}}),ge=new B({props:{name:"forward",anchor:"transformers.ArceeForCausalLM.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"},{name:"logits_to_keep",val:": typing.Union[int, torch.Tensor] = 0"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.ArceeForCausalLM.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.ArceeForCausalLM.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.ArceeForCausalLM.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.ArceeForCausalLM.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.ArceeForCausalLM.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.ArceeForCausalLM.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.ArceeForCausalLM.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.ArceeForCausalLM.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"},{anchor:"transformers.ArceeForCausalLM.forward.logits_to_keep",description:`<strong>logits_to_keep</strong> (<code>Union[int, torch.Tensor]</code>, defaults to <code>0</code>) &#x2014;
If an <code>int</code>, compute logits for the last <code>logits_to_keep</code> tokens. If <code>0</code>, calculate logits for all
<code>input_ids</code> (special case). Only last token logits are needed for generation, and calculating them only for that
token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
If a <code>torch.Tensor</code>, must be 1D corresponding to the indices to keep in the sequence length dimension.
This is useful when using packed tensor format (single dimension for batch and sequence length).`,name:"logits_to_keep"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/arcee/modeling_arcee.py#L423",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast"
>transformers.modeling_outputs.CausalLMOutputWithPast</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/arcee#transformers.ArceeConfig"
>ArceeConfig</a>) and inputs.</p>
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
`}}),ee=new Ye({props:{$$slots:{default:[Ro]},$$scope:{ctx:w}}}),te=new Lo({props:{anchor:"transformers.ArceeForCausalLM.forward.example",$$slots:{default:[Do]},$$scope:{ctx:w}}}),_e=new Ae({props:{title:"ArceeForSequenceClassification",local:"transformers.ArceeForSequenceClassification",headingTag:"h2"}}),be=new B({props:{name:"class transformers.ArceeForSequenceClassification",anchor:"transformers.ArceeForSequenceClassification",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.ArceeForSequenceClassification.config",description:`<strong>config</strong> (<code>GenericForSequenceClassification</code>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/arcee/modeling_arcee.py#L485"}}),ye=new B({props:{name:"forward",anchor:"transformers.ArceeForSequenceClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.ArceeForSequenceClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.ArceeForSequenceClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.ArceeForSequenceClassification.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.ArceeForSequenceClassification.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.ArceeForSequenceClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.ArceeForSequenceClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.ArceeForSequenceClassification.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
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
`}}),oe=new Ye({props:{$$slots:{default:[Vo]},$$scope:{ctx:w}}}),ve=new Ae({props:{title:"ArceeForQuestionAnswering",local:"transformers.ArceeForQuestionAnswering",headingTag:"h2"}}),Te=new B({props:{name:"class transformers.ArceeForQuestionAnswering",anchor:"transformers.ArceeForQuestionAnswering",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.ArceeForQuestionAnswering.config",description:`<strong>config</strong> (<code>GenericForQuestionAnswering</code>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/arcee/modeling_arcee.py#L490"}}),we=new B({props:{name:"forward",anchor:"transformers.ArceeForQuestionAnswering.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"start_positions",val:": typing.Optional[torch.LongTensor] = None"},{name:"end_positions",val:": typing.Optional[torch.LongTensor] = None"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.ArceeForQuestionAnswering.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.ArceeForQuestionAnswering.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.ArceeForQuestionAnswering.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.ArceeForQuestionAnswering.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.ArceeForQuestionAnswering.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.ArceeForQuestionAnswering.forward.start_positions",description:`<strong>start_positions</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for position (index) of the start of the labelled span for computing the token classification loss.
Positions are clamped to the length of the sequence (<code>sequence_length</code>). Position outside of the sequence
are not taken into account for computing the loss.`,name:"start_positions"},{anchor:"transformers.ArceeForQuestionAnswering.forward.end_positions",description:`<strong>end_positions</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
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
`}}),ne=new Ye({props:{$$slots:{default:[Xo]},$$scope:{ctx:w}}}),ke=new Ae({props:{title:"ArceeForTokenClassification",local:"transformers.ArceeForTokenClassification",headingTag:"h2"}}),Me=new B({props:{name:"class transformers.ArceeForTokenClassification",anchor:"transformers.ArceeForTokenClassification",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.ArceeForTokenClassification.config",description:`<strong>config</strong> (<code>GenericForTokenClassification</code>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/arcee/modeling_arcee.py#L495"}}),$e=new B({props:{name:"forward",anchor:"transformers.ArceeForTokenClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.ArceeForTokenClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.ArceeForTokenClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.ArceeForTokenClassification.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.ArceeForTokenClassification.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.ArceeForTokenClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.ArceeForTokenClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.ArceeForTokenClassification.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
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
`}}),se=new Ye({props:{$$slots:{default:[Go]},$$scope:{ctx:w}}}),Ce=new Wo({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/arcee.md"}}),{c(){t=c("meta"),l=a(),o=c("p"),h=a(),v=c("p"),v.innerHTML=T,J=a(),H=c("div"),H.innerHTML=no,et=a(),u(ae.$$.fragment),tt=a(),re=c("p"),re.innerHTML=so,ot=a(),ie=c("p"),ie.innerHTML=ao,nt=a(),u(X.$$.fragment),st=a(),ce=c("p"),ce.innerHTML=ro,at=a(),u(G.$$.fragment),rt=a(),u(de.$$.fragment),it=a(),k=c("div"),u(le.$$.fragment),wt=a(),xe=c("p"),xe.innerHTML=io,kt=a(),Fe=c("p"),Fe.innerHTML=co,Mt=a(),ze=c("p"),ze.innerHTML=lo,$t=a(),u(Y.$$.fragment),ct=a(),u(pe.$$.fragment),dt=a(),M=c("div"),u(he.$$.fragment),Ct=a(),Le=c("p"),Le.textContent=po,At=a(),qe=c("p"),qe.innerHTML=ho,xt=a(),Pe=c("p"),Pe.innerHTML=mo,Ft=a(),Z=c("div"),u(me.$$.fragment),zt=a(),Ie=c("p"),Ie.innerHTML=uo,Lt=a(),u(K.$$.fragment),lt=a(),u(ue.$$.fragment),pt=a(),$=c("div"),u(fe.$$.fragment),qt=a(),Ue=c("p"),Ue.textContent=fo,Pt=a(),je=c("p"),je.innerHTML=go,It=a(),Je=c("p"),Je.innerHTML=_o,Ut=a(),j=c("div"),u(ge.$$.fragment),jt=a(),He=c("p"),He.innerHTML=bo,Jt=a(),u(ee.$$.fragment),Ht=a(),u(te.$$.fragment),ht=a(),u(_e.$$.fragment),mt=a(),C=c("div"),u(be.$$.fragment),Wt=a(),We=c("p"),We.textContent=yo,Bt=a(),Be=c("p"),Be.innerHTML=vo,Zt=a(),Ze=c("p"),Ze.innerHTML=To,Nt=a(),N=c("div"),u(ye.$$.fragment),Qt=a(),Ne=c("p"),Ne.innerHTML=wo,Et=a(),u(oe.$$.fragment),ut=a(),u(ve.$$.fragment),ft=a(),A=c("div"),u(Te.$$.fragment),St=a(),Qe=c("p"),Qe.innerHTML=ko,Ot=a(),Ee=c("p"),Ee.innerHTML=Mo,Rt=a(),Se=c("p"),Se.innerHTML=$o,Dt=a(),Q=c("div"),u(we.$$.fragment),Vt=a(),Oe=c("p"),Oe.innerHTML=Co,Xt=a(),u(ne.$$.fragment),gt=a(),u(ke.$$.fragment),_t=a(),x=c("div"),u(Me.$$.fragment),Gt=a(),Re=c("p"),Re.textContent=Ao,Yt=a(),De=c("p"),De.innerHTML=xo,Kt=a(),Ve=c("p"),Ve.innerHTML=Fo,eo=a(),E=c("div"),u($e.$$.fragment),to=a(),Xe=c("p"),Xe.innerHTML=zo,oo=a(),u(se.$$.fragment),bt=a(),u(Ce.$$.fragment),yt=a(),Ke=c("p"),this.h()},l(e){const n=Jo("svelte-u9bgzb",document.head);t=d(n,"META",{name:!0,content:!0}),n.forEach(i),l=r(e),o=d(e,"P",{}),I(o).forEach(i),h=r(e),v=d(e,"P",{"data-svelte-h":!0}),m(v)!=="svelte-13xybor"&&(v.innerHTML=T),J=r(e),H=d(e,"DIV",{style:!0,"data-svelte-h":!0}),m(H)!=="svelte-2m0t7r"&&(H.innerHTML=no),et=r(e),f(ae.$$.fragment,e),tt=r(e),re=d(e,"P",{"data-svelte-h":!0}),m(re)!=="svelte-1na4ejj"&&(re.innerHTML=so),ot=r(e),ie=d(e,"P",{"data-svelte-h":!0}),m(ie)!=="svelte-rjl0b4"&&(ie.innerHTML=ao),nt=r(e),f(X.$$.fragment,e),st=r(e),ce=d(e,"P",{"data-svelte-h":!0}),m(ce)!=="svelte-1bcss8a"&&(ce.innerHTML=ro),at=r(e),f(G.$$.fragment,e),rt=r(e),f(de.$$.fragment,e),it=r(e),k=d(e,"DIV",{class:!0});var F=I(k);f(le.$$.fragment,F),wt=r(F),xe=d(F,"P",{"data-svelte-h":!0}),m(xe)!=="svelte-16uwxb"&&(xe.innerHTML=io),kt=r(F),Fe=d(F,"P",{"data-svelte-h":!0}),m(Fe)!=="svelte-wwir5u"&&(Fe.innerHTML=co),Mt=r(F),ze=d(F,"P",{"data-svelte-h":!0}),m(ze)!=="svelte-1ek1ss9"&&(ze.innerHTML=lo),$t=r(F),f(Y.$$.fragment,F),F.forEach(i),ct=r(e),f(pe.$$.fragment,e),dt=r(e),M=d(e,"DIV",{class:!0});var z=I(M);f(he.$$.fragment,z),Ct=r(z),Le=d(z,"P",{"data-svelte-h":!0}),m(Le)!=="svelte-obf9c8"&&(Le.textContent=po),At=r(z),qe=d(z,"P",{"data-svelte-h":!0}),m(qe)!=="svelte-q52n56"&&(qe.innerHTML=ho),xt=r(z),Pe=d(z,"P",{"data-svelte-h":!0}),m(Pe)!=="svelte-hswkmf"&&(Pe.innerHTML=mo),Ft=r(z),Z=d(z,"DIV",{class:!0});var R=I(Z);f(me.$$.fragment,R),zt=r(R),Ie=d(R,"P",{"data-svelte-h":!0}),m(Ie)!=="svelte-bqqsyz"&&(Ie.innerHTML=uo),Lt=r(R),f(K.$$.fragment,R),R.forEach(i),z.forEach(i),lt=r(e),f(ue.$$.fragment,e),pt=r(e),$=d(e,"DIV",{class:!0});var L=I($);f(fe.$$.fragment,L),qt=r(L),Ue=d(L,"P",{"data-svelte-h":!0}),m(Ue)!=="svelte-204rs1"&&(Ue.textContent=fo),Pt=r(L),je=d(L,"P",{"data-svelte-h":!0}),m(je)!=="svelte-q52n56"&&(je.innerHTML=go),It=r(L),Je=d(L,"P",{"data-svelte-h":!0}),m(Je)!=="svelte-hswkmf"&&(Je.innerHTML=_o),Ut=r(L),j=d(L,"DIV",{class:!0});var W=I(j);f(ge.$$.fragment,W),jt=r(W),He=d(W,"P",{"data-svelte-h":!0}),m(He)!=="svelte-mg6pof"&&(He.innerHTML=bo),Jt=r(W),f(ee.$$.fragment,W),Ht=r(W),f(te.$$.fragment,W),W.forEach(i),L.forEach(i),ht=r(e),f(_e.$$.fragment,e),mt=r(e),C=d(e,"DIV",{class:!0});var q=I(C);f(be.$$.fragment,q),Wt=r(q),We=d(q,"P",{"data-svelte-h":!0}),m(We)!=="svelte-9uh1zm"&&(We.textContent=yo),Bt=r(q),Be=d(q,"P",{"data-svelte-h":!0}),m(Be)!=="svelte-q52n56"&&(Be.innerHTML=vo),Zt=r(q),Ze=d(q,"P",{"data-svelte-h":!0}),m(Ze)!=="svelte-hswkmf"&&(Ze.innerHTML=To),Nt=r(q),N=d(q,"DIV",{class:!0});var D=I(N);f(ye.$$.fragment,D),Qt=r(D),Ne=d(D,"P",{"data-svelte-h":!0}),m(Ne)!=="svelte-1sal4ui"&&(Ne.innerHTML=wo),Et=r(D),f(oe.$$.fragment,D),D.forEach(i),q.forEach(i),ut=r(e),f(ve.$$.fragment,e),ft=r(e),A=d(e,"DIV",{class:!0});var P=I(A);f(Te.$$.fragment,P),St=r(P),Qe=d(P,"P",{"data-svelte-h":!0}),m(Qe)!=="svelte-wngnw7"&&(Qe.innerHTML=ko),Ot=r(P),Ee=d(P,"P",{"data-svelte-h":!0}),m(Ee)!=="svelte-q52n56"&&(Ee.innerHTML=Mo),Rt=r(P),Se=d(P,"P",{"data-svelte-h":!0}),m(Se)!=="svelte-hswkmf"&&(Se.innerHTML=$o),Dt=r(P),Q=d(P,"DIV",{class:!0});var V=I(Q);f(we.$$.fragment,V),Vt=r(V),Oe=d(V,"P",{"data-svelte-h":!0}),m(Oe)!=="svelte-dyrov9"&&(Oe.innerHTML=Co),Xt=r(V),f(ne.$$.fragment,V),V.forEach(i),P.forEach(i),gt=r(e),f(ke.$$.fragment,e),_t=r(e),x=d(e,"DIV",{class:!0});var S=I(x);f(Me.$$.fragment,S),Gt=r(S),Re=d(S,"P",{"data-svelte-h":!0}),m(Re)!=="svelte-1ecsnks"&&(Re.textContent=Ao),Yt=r(S),De=d(S,"P",{"data-svelte-h":!0}),m(De)!=="svelte-q52n56"&&(De.innerHTML=xo),Kt=r(S),Ve=d(S,"P",{"data-svelte-h":!0}),m(Ve)!=="svelte-hswkmf"&&(Ve.innerHTML=Fo),eo=r(S),E=d(S,"DIV",{class:!0});var Ge=I(E);f($e.$$.fragment,Ge),to=r(Ge),Xe=d(Ge,"P",{"data-svelte-h":!0}),m(Xe)!=="svelte-1py4aay"&&(Xe.innerHTML=zo),oo=r(Ge),f(se.$$.fragment,Ge),Ge.forEach(i),S.forEach(i),bt=r(e),f(Ce.$$.fragment,e),yt=r(e),Ke=d(e,"P",{}),I(Ke).forEach(i),this.h()},h(){U(t,"name","hf:doc:metadata"),U(t,"content",Ko),Ho(H,"float","right"),U(k,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),U(Z,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),U(M,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),U(j,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),U($,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),U(N,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),U(C,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),U(Q,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),U(A,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),U(E,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),U(x,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,n){s(document.head,t),p(e,l,n),p(e,o,n),p(e,h,n),p(e,v,n),p(e,J,n),p(e,H,n),p(e,et,n),g(ae,e,n),p(e,tt,n),p(e,re,n),p(e,ot,n),p(e,ie,n),p(e,nt,n),g(X,e,n),p(e,st,n),p(e,ce,n),p(e,at,n),g(G,e,n),p(e,rt,n),g(de,e,n),p(e,it,n),p(e,k,n),g(le,k,null),s(k,wt),s(k,xe),s(k,kt),s(k,Fe),s(k,Mt),s(k,ze),s(k,$t),g(Y,k,null),p(e,ct,n),g(pe,e,n),p(e,dt,n),p(e,M,n),g(he,M,null),s(M,Ct),s(M,Le),s(M,At),s(M,qe),s(M,xt),s(M,Pe),s(M,Ft),s(M,Z),g(me,Z,null),s(Z,zt),s(Z,Ie),s(Z,Lt),g(K,Z,null),p(e,lt,n),g(ue,e,n),p(e,pt,n),p(e,$,n),g(fe,$,null),s($,qt),s($,Ue),s($,Pt),s($,je),s($,It),s($,Je),s($,Ut),s($,j),g(ge,j,null),s(j,jt),s(j,He),s(j,Jt),g(ee,j,null),s(j,Ht),g(te,j,null),p(e,ht,n),g(_e,e,n),p(e,mt,n),p(e,C,n),g(be,C,null),s(C,Wt),s(C,We),s(C,Bt),s(C,Be),s(C,Zt),s(C,Ze),s(C,Nt),s(C,N),g(ye,N,null),s(N,Qt),s(N,Ne),s(N,Et),g(oe,N,null),p(e,ut,n),g(ve,e,n),p(e,ft,n),p(e,A,n),g(Te,A,null),s(A,St),s(A,Qe),s(A,Ot),s(A,Ee),s(A,Rt),s(A,Se),s(A,Dt),s(A,Q),g(we,Q,null),s(Q,Vt),s(Q,Oe),s(Q,Xt),g(ne,Q,null),p(e,gt,n),g(ke,e,n),p(e,_t,n),p(e,x,n),g(Me,x,null),s(x,Gt),s(x,Re),s(x,Yt),s(x,De),s(x,Kt),s(x,Ve),s(x,eo),s(x,E),g($e,E,null),s(E,to),s(E,Xe),s(E,oo),g(se,E,null),p(e,bt,n),g(Ce,e,n),p(e,yt,n),p(e,Ke,n),vt=!0},p(e,[n]){const F={};n&2&&(F.$$scope={dirty:n,ctx:e}),X.$set(F);const z={};n&2&&(z.$$scope={dirty:n,ctx:e}),G.$set(z);const R={};n&2&&(R.$$scope={dirty:n,ctx:e}),Y.$set(R);const L={};n&2&&(L.$$scope={dirty:n,ctx:e}),K.$set(L);const W={};n&2&&(W.$$scope={dirty:n,ctx:e}),ee.$set(W);const q={};n&2&&(q.$$scope={dirty:n,ctx:e}),te.$set(q);const D={};n&2&&(D.$$scope={dirty:n,ctx:e}),oe.$set(D);const P={};n&2&&(P.$$scope={dirty:n,ctx:e}),ne.$set(P);const V={};n&2&&(V.$$scope={dirty:n,ctx:e}),se.$set(V)},i(e){vt||(_(ae.$$.fragment,e),_(X.$$.fragment,e),_(G.$$.fragment,e),_(de.$$.fragment,e),_(le.$$.fragment,e),_(Y.$$.fragment,e),_(pe.$$.fragment,e),_(he.$$.fragment,e),_(me.$$.fragment,e),_(K.$$.fragment,e),_(ue.$$.fragment,e),_(fe.$$.fragment,e),_(ge.$$.fragment,e),_(ee.$$.fragment,e),_(te.$$.fragment,e),_(_e.$$.fragment,e),_(be.$$.fragment,e),_(ye.$$.fragment,e),_(oe.$$.fragment,e),_(ve.$$.fragment,e),_(Te.$$.fragment,e),_(we.$$.fragment,e),_(ne.$$.fragment,e),_(ke.$$.fragment,e),_(Me.$$.fragment,e),_($e.$$.fragment,e),_(se.$$.fragment,e),_(Ce.$$.fragment,e),vt=!0)},o(e){b(ae.$$.fragment,e),b(X.$$.fragment,e),b(G.$$.fragment,e),b(de.$$.fragment,e),b(le.$$.fragment,e),b(Y.$$.fragment,e),b(pe.$$.fragment,e),b(he.$$.fragment,e),b(me.$$.fragment,e),b(K.$$.fragment,e),b(ue.$$.fragment,e),b(fe.$$.fragment,e),b(ge.$$.fragment,e),b(ee.$$.fragment,e),b(te.$$.fragment,e),b(_e.$$.fragment,e),b(be.$$.fragment,e),b(ye.$$.fragment,e),b(oe.$$.fragment,e),b(ve.$$.fragment,e),b(Te.$$.fragment,e),b(we.$$.fragment,e),b(ne.$$.fragment,e),b(ke.$$.fragment,e),b(Me.$$.fragment,e),b($e.$$.fragment,e),b(se.$$.fragment,e),b(Ce.$$.fragment,e),vt=!1},d(e){e&&(i(l),i(o),i(h),i(v),i(J),i(H),i(et),i(tt),i(re),i(ot),i(ie),i(nt),i(st),i(ce),i(at),i(rt),i(it),i(k),i(ct),i(dt),i(M),i(lt),i(pt),i($),i(ht),i(mt),i(C),i(ut),i(ft),i(A),i(gt),i(_t),i(x),i(bt),i(yt),i(Ke)),i(t),y(ae,e),y(X,e),y(G,e),y(de,e),y(le),y(Y),y(pe,e),y(he),y(me),y(K),y(ue,e),y(fe),y(ge),y(ee),y(te),y(_e,e),y(be),y(ye),y(oe),y(ve,e),y(Te),y(we),y(ne),y(ke,e),y(Me),y($e),y(se),y(Ce,e)}}}const Ko='{"title":"Arcee","local":"arcee","sections":[{"title":"ArceeConfig","local":"transformers.ArceeConfig","sections":[],"depth":2},{"title":"ArceeModel","local":"transformers.ArceeModel","sections":[],"depth":2},{"title":"ArceeForCausalLM","local":"transformers.ArceeForCausalLM","sections":[],"depth":2},{"title":"ArceeForSequenceClassification","local":"transformers.ArceeForSequenceClassification","sections":[],"depth":2},{"title":"ArceeForQuestionAnswering","local":"transformers.ArceeForQuestionAnswering","sections":[],"depth":2},{"title":"ArceeForTokenClassification","local":"transformers.ArceeForTokenClassification","sections":[],"depth":2}],"depth":1}';function en(w){return Io(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class ln extends Uo{constructor(t){super(),jo(this,t,en,Yo,Po,{})}}export{ln as component};
