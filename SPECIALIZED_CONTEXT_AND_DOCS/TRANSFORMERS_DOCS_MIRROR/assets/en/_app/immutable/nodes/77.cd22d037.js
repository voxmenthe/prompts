import{s as vt,o as Tt,n as E}from"../chunks/scheduler.18a86fab.js";import{S as wt,i as Mt,g as u,s as r,r as h,A as kt,h as m,f as s,c as i,j as P,x as k,u as f,k as Z,l as $t,y as d,a as c,v as g,d as _,t as y,w as b}from"../chunks/index.98837b22.js";import{T as Ze}from"../chunks/Tip.77304350.js";import{D as ce}from"../chunks/Docstring.a1ef7999.js";import{C as $e}from"../chunks/CodeBlock.8d0c2e8a.js";import{E as bt}from"../chunks/ExampleCodeBlock.8c3ee1f9.js";import{H as ke,E as Ct}from"../chunks/getInferenceSnippets.06c2775f.js";import{H as xt,a as rt}from"../chunks/HfOption.6641485e.js";function At(v){let t,a="Coming soon";return{c(){t=u("p"),t.textContent=a},l(o){t=m(o,"P",{"data-svelte-h":!0}),k(t)!=="svelte-3ot190"&&(t.textContent=a)},m(o,p){c(o,t,p)},p:E,d(o){o&&s(t)}}}function zt(v){let t,a;return t=new $e({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwcGlwZWxpbmUlMEElMEFwaXBlbGluZSUyMCUzRCUyMHBpcGVsaW5lKCUwQSUyMCUyMCUyMCUyMHRhc2slM0QlMjJ0ZXh0LWdlbmVyYXRpb24lMjIlMkMlMEElMjAlMjAlMjAlMjBtb2RlbCUzRCUyMnN3aXNzLWFpJTJGQXBlcnR1cy04QiUyMiUyQyUwQSUyMCUyMCUyMCUyMGR0eXBlJTNEdG9yY2guYmZsb2F0MTYlMkMlMEElMjAlMjAlMjAlMjBkZXZpY2UlM0QwJTBBKSUwQXBpcGVsaW5lKCUyMlBsYW50cyUyMGNyZWF0ZSUyMGVuZXJneSUyMHRocm91Z2glMjBhJTIwcHJvY2VzcyUyMGtub3duJTIwYXMlMjIp",highlighted:`<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> pipeline

pipeline = pipeline(
    task=<span class="hljs-string">&quot;text-generation&quot;</span>,
    model=<span class="hljs-string">&quot;swiss-ai/Apertus-8B&quot;</span>,
    dtype=torch.bfloat16,
    device=<span class="hljs-number">0</span>
)
pipeline(<span class="hljs-string">&quot;Plants create energy through a process known as&quot;</span>)`,wrap:!1}}),{c(){h(t.$$.fragment)},l(o){f(t.$$.fragment,o)},m(o,p){g(t,o,p),a=!0},p:E,i(o){a||(_(t.$$.fragment,o),a=!0)},o(o){y(t.$$.fragment,o),a=!1},d(o){b(t,o)}}}function Ut(v){let t,a;return t=new $e({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b01vZGVsRm9yQ2F1c2FsTE0lMkMlMjBBdXRvVG9rZW5pemVyJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTBBJTIwJTIwJTIwJTIwJTIyc3dpc3MtYWklMkZBcGVydHVzLThCJTIyJTJDJTBBKSUwQW1vZGVsJTIwJTNEJTIwQXV0b01vZGVsRm9yQ2F1c2FsTE0uZnJvbV9wcmV0cmFpbmVkKCUwQSUyMCUyMCUyMCUyMCUyMnN3aXNzLWFpJTJGQXBlcnR1cy04QiUyMiUyQyUwQSUyMCUyMCUyMCUyMGR0eXBlJTNEdG9yY2guYmZsb2F0MTYlMkMlMEElMjAlMjAlMjAlMjBkZXZpY2VfbWFwJTNEJTIyYXV0byUyMiUyQyUwQSUyMCUyMCUyMCUyMGF0dG5faW1wbGVtZW50YXRpb24lM0QlMjJzZHBhJTIyJTBBKSUwQWlucHV0X2lkcyUyMCUzRCUyMHRva2VuaXplciglMjJQbGFudHMlMjBjcmVhdGUlMjBlbmVyZ3klMjB0aHJvdWdoJTIwYSUyMHByb2Nlc3MlMjBrbm93biUyMGFzJTIyJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMikudG8oJTIyY3VkYSUyMiklMEElMEFvdXRwdXQlMjAlM0QlMjBtb2RlbC5nZW5lcmF0ZSgqKmlucHV0X2lkcyklMEFwcmludCh0b2tlbml6ZXIuZGVjb2RlKG91dHB1dCU1QjAlNUQlMkMlMjBza2lwX3NwZWNpYWxfdG9rZW5zJTNEVHJ1ZSkp",highlighted:`<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    <span class="hljs-string">&quot;swiss-ai/Apertus-8B&quot;</span>,
)
model = AutoModelForCausalLM.from_pretrained(
    <span class="hljs-string">&quot;swiss-ai/Apertus-8B&quot;</span>,
    dtype=torch.bfloat16,
    device_map=<span class="hljs-string">&quot;auto&quot;</span>,
    attn_implementation=<span class="hljs-string">&quot;sdpa&quot;</span>
)
input_ids = tokenizer(<span class="hljs-string">&quot;Plants create energy through a process known as&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>).to(<span class="hljs-string">&quot;cuda&quot;</span>)

output = model.generate(**input_ids)
<span class="hljs-built_in">print</span>(tokenizer.decode(output[<span class="hljs-number">0</span>], skip_special_tokens=<span class="hljs-literal">True</span>))`,wrap:!1}}),{c(){h(t.$$.fragment)},l(o){f(t.$$.fragment,o)},m(o,p){g(t,o,p),a=!0},p:E,i(o){a||(_(t.$$.fragment,o),a=!0)},o(o){y(t.$$.fragment,o),a=!1},d(o){b(t,o)}}}function Ft(v){let t,a;return t=new $e({props:{code:"ZWNobyUyMC1lJTIwJTIyUGxhbnRzJTIwY3JlYXRlJTIwZW5lcmd5JTIwdGhyb3VnaCUyMGElMjBwcm9jZXNzJTIwa25vd24lMjBhcyUyMiUyMCU3QyUyMHRyYW5zZm9ybWVycyUyMHJ1biUyMC0tdGFzayUyMHRleHQtZ2VuZXJhdGlvbiUyMC0tbW9kZWwlMjBzd2lzcy1haSUyRkFwZXJ0dXMtOEIlMjAtLWRldmljZSUyMDA=",highlighted:'<span class="hljs-built_in">echo</span> -e <span class="hljs-string">&quot;Plants create energy through a process known as&quot;</span> | transformers run --task text-generation --model swiss-ai/Apertus-8B --device 0',wrap:!1}}),{c(){h(t.$$.fragment)},l(o){f(t.$$.fragment,o)},m(o,p){g(t,o,p),a=!0},p:E,i(o){a||(_(t.$$.fragment,o),a=!0)},o(o){y(t.$$.fragment,o),a=!1},d(o){b(t,o)}}}function It(v){let t,a,o,p,T,w;return t=new rt({props:{id:"usage",option:"Pipeline",$$slots:{default:[zt]},$$scope:{ctx:v}}}),o=new rt({props:{id:"usage",option:"AutoModel",$$slots:{default:[Ut]},$$scope:{ctx:v}}}),T=new rt({props:{id:"usage",option:"transformers CLI",$$slots:{default:[Ft]},$$scope:{ctx:v}}}),{c(){h(t.$$.fragment),a=r(),h(o.$$.fragment),p=r(),h(T.$$.fragment)},l(l){f(t.$$.fragment,l),a=i(l),f(o.$$.fragment,l),p=i(l),f(T.$$.fragment,l)},m(l,M){g(t,l,M),c(l,a,M),g(o,l,M),c(l,p,M),g(T,l,M),w=!0},p(l,M){const O={};M&2&&(O.$$scope={dirty:M,ctx:l}),t.$set(O);const j={};M&2&&(j.$$scope={dirty:M,ctx:l}),o.$set(j);const we={};M&2&&(we.$$scope={dirty:M,ctx:l}),T.$set(we)},i(l){w||(_(t.$$.fragment,l),_(o.$$.fragment,l),_(T.$$.fragment,l),w=!0)},o(l){y(t.$$.fragment,l),y(o.$$.fragment,l),y(T.$$.fragment,l),w=!1},d(l){l&&(s(a),s(p)),b(t,l),b(o,l),b(T,l)}}}function jt(v){let t,a;return t=new $e({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEFwZXJ0dXNNb2RlbCUyQyUyMEFwZXJ0dXNDb25maWclMEElMEElMjMlMjBJbml0aWFsaXppbmclMjBhJTIwQXBlcnR1cy04QiUyMHN0eWxlJTIwY29uZmlndXJhdGlvbiUwQWNvbmZpZ3VyYXRpb24lMjAlM0QlMjBBcGVydHVzQ29uZmlnKCklMEElMEElMjMlMjBJbml0aWFsaXppbmclMjBhJTIwbW9kZWwlMjBmcm9tJTIwdGhlJTIwQXBlcnR1cy04QiUyMHN0eWxlJTIwY29uZmlndXJhdGlvbiUwQW1vZGVsJTIwJTNEJTIwQXBlcnR1c01vZGVsKGNvbmZpZ3VyYXRpb24pJTBBJTBBJTIzJTIwQWNjZXNzaW5nJTIwdGhlJTIwbW9kZWwlMjBjb25maWd1cmF0aW9uJTBBY29uZmlndXJhdGlvbiUyMCUzRCUyMG1vZGVsLmNvbmZpZw==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> ApertusModel, ApertusConfig

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a Apertus-8B style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = ApertusConfig()

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a model from the Apertus-8B style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = ApertusModel(configuration)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Accessing the model configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = model.config`,wrap:!1}}),{c(){h(t.$$.fragment)},l(o){f(t.$$.fragment,o)},m(o,p){g(t,o,p),a=!0},p:E,i(o){a||(_(t.$$.fragment,o),a=!0)},o(o){y(t.$$.fragment,o),a=!1},d(o){b(t,o)}}}function Jt(v){let t,a=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=u("p"),t.innerHTML=a},l(o){t=m(o,"P",{"data-svelte-h":!0}),k(t)!=="svelte-fincs2"&&(t.innerHTML=a)},m(o,p){c(o,t,p)},p:E,d(o){o&&s(t)}}}function Lt(v){let t,a=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=u("p"),t.innerHTML=a},l(o){t=m(o,"P",{"data-svelte-h":!0}),k(t)!=="svelte-fincs2"&&(t.innerHTML=a)},m(o,p){c(o,t,p)},p:E,d(o){o&&s(t)}}}function Bt(v){let t,a="Example:",o,p,T;return p=new $e({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBBcGVydHVzRm9yQ2F1c2FsTE0lMEElMEFtb2RlbCUyMCUzRCUyMEFwZXJ0dXNGb3JDYXVzYWxMTS5mcm9tX3ByZXRyYWluZWQoJTIyc3dpc3MtYWklMkZBcGVydHVzLThCJTIyKSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMnN3aXNzLWFpJTJGQXBlcnR1cy04QiUyMiklMEElMEFwcm9tcHQlMjAlM0QlMjAlMjJIZXklMkMlMjBhcmUlMjB5b3UlMjBjb25zY2lvdXMlM0YlMjBDYW4lMjB5b3UlMjB0YWxrJTIwdG8lMjBtZSUzRiUyMiUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplcihwcm9tcHQlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQSUwQSUyMyUyMEdlbmVyYXRlJTBBZ2VuZXJhdGVfaWRzJTIwJTNEJTIwbW9kZWwuZ2VuZXJhdGUoaW5wdXRzLmlucHV0X2lkcyUyQyUyMG1heF9sZW5ndGglM0QzMCklMEF0b2tlbml6ZXIuYmF0Y2hfZGVjb2RlKGdlbmVyYXRlX2lkcyUyQyUyMHNraXBfc3BlY2lhbF90b2tlbnMlM0RUcnVlJTJDJTIwY2xlYW5fdXBfdG9rZW5pemF0aW9uX3NwYWNlcyUzREZhbHNlKSU1QjAlNUQ=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, ApertusForCausalLM

<span class="hljs-meta">&gt;&gt;&gt; </span>model = ApertusForCausalLM.from_pretrained(<span class="hljs-string">&quot;swiss-ai/Apertus-8B&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;swiss-ai/Apertus-8B&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>prompt = <span class="hljs-string">&quot;Hey, are you conscious? Can you talk to me?&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(prompt, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Generate</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>generate_ids = model.generate(inputs.input_ids, max_length=<span class="hljs-number">30</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer.batch_decode(generate_ids, skip_special_tokens=<span class="hljs-literal">True</span>, clean_up_tokenization_spaces=<span class="hljs-literal">False</span>)[<span class="hljs-number">0</span>]
<span class="hljs-string">&quot;Hey, are you conscious? Can you talk to me?\\nI&#x27;m not conscious, but I can talk to you.&quot;</span>`,wrap:!1}}),{c(){t=u("p"),t.textContent=a,o=r(),h(p.$$.fragment)},l(w){t=m(w,"P",{"data-svelte-h":!0}),k(t)!=="svelte-11lpom8"&&(t.textContent=a),o=i(w),f(p.$$.fragment,w)},m(w,l){c(w,t,l),c(w,o,l),g(p,w,l),T=!0},p:E,i(w){T||(_(p.$$.fragment,w),T=!0)},o(w){y(p.$$.fragment,w),T=!1},d(w){w&&(s(t),s(o)),b(p,w)}}}function qt(v){let t,a=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=u("p"),t.innerHTML=a},l(o){t=m(o,"P",{"data-svelte-h":!0}),k(t)!=="svelte-fincs2"&&(t.innerHTML=a)},m(o,p){c(o,t,p)},p:E,d(o){o&&s(t)}}}function Wt(v){let t,a,o,p,T,w='<div class="flex flex-wrap space-x-1"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/> <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat"/> <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/> <img alt="Tensor parallelism" src="https://img.shields.io/badge/Tensor%20parallelism-06b6d4?style=flat&amp;logoColor=white"/></div>',l,M,O,j,we='<a href="https://www.swiss-ai.org" rel="nofollow">Apertus</a> is a family of large language models from the Swiss AI Initiative.',Ce,X,xe,D,it='The example below demonstrates how to generate text with <a href="/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.Pipeline">Pipeline</a> or the <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel">AutoModel</a>, and from the command line.',Ae,H,ze,Y,Ue,x,S,Ee,pe,lt=`This is the configuration class to store the configuration of a <a href="/docs/transformers/v4.56.2/en/model_doc/apertus#transformers.ApertusModel">ApertusModel</a>. It is used to instantiate a Apertus
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the Apertus-8B.
e.g. <a href="https://huggingface.co/swiss-ai/Apertus-8B" rel="nofollow">swiss-ai/Apertus-8B</a>`,Xe,ue,dt=`Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> and can be used to control the model outputs. Read the
documentation from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> for more information.`,He,V,Fe,K,Ie,$,ee,Ve,me,ct="The bare Apertus Model outputting raw hidden-states without any specific head on top.",Ge,he,pt=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Ne,fe,ut=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Re,J,te,Qe,ge,mt='The <a href="/docs/transformers/v4.56.2/en/model_doc/apertus#transformers.ApertusModel">ApertusModel</a> forward method, overrides the <code>__call__</code> special method.',Oe,G,je,oe,Je,C,ne,De,_e,ht="The Apertus Model for causal language modeling.",Ye,ye,ft=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Se,be,gt=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Ke,U,se,et,ve,_t='The <a href="/docs/transformers/v4.56.2/en/model_doc/apertus#transformers.ApertusForCausalLM">ApertusForCausalLM</a> forward method, overrides the <code>__call__</code> special method.',tt,N,ot,R,Le,ae,Be,B,re,nt,L,ie,st,Te,yt="The <code>GenericForTokenClassification</code> forward method, overrides the <code>__call__</code> special method.",at,Q,qe,le,We,Me,Pe;return M=new ke({props:{title:"Apertus",local:"apertus",headingTag:"h1"}}),X=new Ze({props:{warning:!1,$$slots:{default:[At]},$$scope:{ctx:v}}}),H=new xt({props:{id:"usage",options:["Pipeline","AutoModel","transformers CLI"],$$slots:{default:[It]},$$scope:{ctx:v}}}),Y=new ke({props:{title:"ApertusConfig",local:"transformers.ApertusConfig",headingTag:"h2"}}),S=new ce({props:{name:"class transformers.ApertusConfig",anchor:"transformers.ApertusConfig",parameters:[{name:"vocab_size",val:" = 131072"},{name:"hidden_size",val:" = 4096"},{name:"intermediate_size",val:" = 14336"},{name:"num_hidden_layers",val:" = 32"},{name:"num_attention_heads",val:" = 32"},{name:"num_key_value_heads",val:" = None"},{name:"hidden_act",val:" = 'xielu'"},{name:"max_position_embeddings",val:" = 65536"},{name:"initializer_range",val:" = 0.02"},{name:"rms_norm_eps",val:" = 1e-05"},{name:"use_cache",val:" = True"},{name:"pad_token_id",val:" = 3"},{name:"bos_token_id",val:" = 1"},{name:"eos_token_id",val:" = 2"},{name:"tie_word_embeddings",val:" = False"},{name:"rope_theta",val:" = 12000000.0"},{name:"rope_scaling",val:" = {'rope_type': 'llama3', 'factor': 8.0, 'original_max_position_embeddings': 8192, 'low_freq_factor': 1.0, 'high_freq_factor': 4.0}"},{name:"attention_bias",val:" = False"},{name:"attention_dropout",val:" = 0.0"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.ApertusConfig.vocab_size",description:`<strong>vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 131072) &#x2014;
Vocabulary size of the Apertus model. Defines the number of different tokens that can be represented by the
<code>inputs_ids</code> passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/apertus#transformers.ApertusModel">ApertusModel</a>`,name:"vocab_size"},{anchor:"transformers.ApertusConfig.hidden_size",description:`<strong>hidden_size</strong> (<code>int</code>, <em>optional</em>, defaults to 4096) &#x2014;
Dimension of the hidden representations.`,name:"hidden_size"},{anchor:"transformers.ApertusConfig.intermediate_size",description:`<strong>intermediate_size</strong> (<code>int</code>, <em>optional</em>, defaults to 14336) &#x2014;
Dimension of the MLP representations.`,name:"intermediate_size"},{anchor:"transformers.ApertusConfig.num_hidden_layers",description:`<strong>num_hidden_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 32) &#x2014;
Number of hidden layers in the Transformer decoder.`,name:"num_hidden_layers"},{anchor:"transformers.ApertusConfig.num_attention_heads",description:`<strong>num_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 32) &#x2014;
Number of attention heads for each attention layer in the Transformer decoder.`,name:"num_attention_heads"},{anchor:"transformers.ApertusConfig.num_key_value_heads",description:`<strong>num_key_value_heads</strong> (<code>int</code>, <em>optional</em>) &#x2014;
This is the number of key_value heads that should be used to implement Grouped Query Attention. If
<code>num_key_value_heads=num_attention_heads</code>, the model will use Multi Head Attention (MHA), if
<code>num_key_value_heads=1</code> the model will use Multi Query Attention (MQA) otherwise GQA is used. When
converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
by meanpooling all the original heads within that group. For more details, check out <a href="https://huggingface.co/papers/2305.13245" rel="nofollow">this
paper</a>. If it is not specified, will default to
<code>num_attention_heads</code>.`,name:"num_key_value_heads"},{anchor:"transformers.ApertusConfig.hidden_act",description:`<strong>hidden_act</strong> (<code>str</code> or <code>function</code>, <em>optional</em>, defaults to <code>&quot;xielu&quot;</code>) &#x2014;
The non-linear activation function (function or string) in the decoder.`,name:"hidden_act"},{anchor:"transformers.ApertusConfig.max_position_embeddings",description:`<strong>max_position_embeddings</strong> (<code>int</code>, <em>optional</em>, defaults to 65536) &#x2014;
The maximum sequence length that this model might ever be used with. Apertus supports up to 65536 tokens.`,name:"max_position_embeddings"},{anchor:"transformers.ApertusConfig.initializer_range",description:`<strong>initializer_range</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.`,name:"initializer_range"},{anchor:"transformers.ApertusConfig.rms_norm_eps",description:`<strong>rms_norm_eps</strong> (<code>float</code>, <em>optional</em>, defaults to 1e-05) &#x2014;
The epsilon used by the rms normalization layers.`,name:"rms_norm_eps"},{anchor:"transformers.ApertusConfig.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not the model should return the last key/values attentions (not used by all models). Only
relevant if <code>config.is_decoder=True</code>.`,name:"use_cache"},{anchor:"transformers.ApertusConfig.pad_token_id",description:`<strong>pad_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 3) &#x2014;
Padding token id.`,name:"pad_token_id"},{anchor:"transformers.ApertusConfig.bos_token_id",description:`<strong>bos_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 1) &#x2014;
Beginning of stream token id.`,name:"bos_token_id"},{anchor:"transformers.ApertusConfig.eos_token_id",description:`<strong>eos_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 2) &#x2014;
End of stream token id.`,name:"eos_token_id"},{anchor:"transformers.ApertusConfig.tie_word_embeddings",description:`<strong>tie_word_embeddings</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether to tie weight embeddings`,name:"tie_word_embeddings"},{anchor:"transformers.ApertusConfig.rope_theta",description:`<strong>rope_theta</strong> (<code>float</code>, <em>optional</em>, defaults to 12000000.0) &#x2014;
The base period of the RoPE embeddings.`,name:"rope_theta"},{anchor:"transformers.ApertusConfig.rope_scaling",description:`<strong>rope_scaling</strong> (<code>Dict</code>, <em>optional</em>) &#x2014;
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
Only used with &#x2018;llama3&#x2019;. Scaling factor applied to high frequency components of the RoPE`,name:"rope_scaling"},{anchor:"transformers.ApertusConfig.attention_bias",description:`<strong>attention_bias</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether to use a bias in the query, key, value and output projection layers during self-attention.`,name:"attention_bias"},{anchor:"transformers.ApertusConfig.attention_dropout",description:`<strong>attention_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The dropout ratio for the attention probabilities.`,name:"attention_dropout"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/apertus/configuration_apertus.py#L27"}}),V=new bt({props:{anchor:"transformers.ApertusConfig.example",$$slots:{default:[jt]},$$scope:{ctx:v}}}),K=new ke({props:{title:"ApertusModel",local:"transformers.ApertusModel",headingTag:"h2"}}),ee=new ce({props:{name:"class transformers.ApertusModel",anchor:"transformers.ApertusModel",parameters:[{name:"config",val:": ApertusConfig"}],parametersDescription:[{anchor:"transformers.ApertusModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/apertus#transformers.ApertusConfig">ApertusConfig</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/apertus/modeling_apertus.py#L325"}}),te=new ce({props:{name:"forward",anchor:"transformers.ApertusModel.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.ApertusModel.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.ApertusModel.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.ApertusModel.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.ApertusModel.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.ApertusModel.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.ApertusModel.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"},{anchor:"transformers.ApertusModel.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/apertus/modeling_apertus.py#L342",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast"
>transformers.modeling_outputs.BaseModelOutputWithPast</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/apertus#transformers.ApertusConfig"
>ApertusConfig</a>) and inputs.</p>
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
`}}),G=new Ze({props:{$$slots:{default:[Jt]},$$scope:{ctx:v}}}),oe=new ke({props:{title:"ApertusForCausalLM",local:"transformers.ApertusForCausalLM",headingTag:"h2"}}),ne=new ce({props:{name:"class transformers.ApertusForCausalLM",anchor:"transformers.ApertusForCausalLM",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.ApertusForCausalLM.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/apertus#transformers.ApertusForCausalLM">ApertusForCausalLM</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/apertus/modeling_apertus.py#L404"}}),se=new ce({props:{name:"forward",anchor:"transformers.ApertusForCausalLM.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"},{name:"logits_to_keep",val:": typing.Union[int, torch.Tensor] = 0"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.ApertusForCausalLM.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.ApertusForCausalLM.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.ApertusForCausalLM.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.ApertusForCausalLM.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.ApertusForCausalLM.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.ApertusForCausalLM.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.ApertusForCausalLM.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.ApertusForCausalLM.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"},{anchor:"transformers.ApertusForCausalLM.forward.logits_to_keep",description:`<strong>logits_to_keep</strong> (<code>Union[int, torch.Tensor]</code>, defaults to <code>0</code>) &#x2014;
If an <code>int</code>, compute logits for the last <code>logits_to_keep</code> tokens. If <code>0</code>, calculate logits for all
<code>input_ids</code> (special case). Only last token logits are needed for generation, and calculating them only for that
token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
If a <code>torch.Tensor</code>, must be 1D corresponding to the indices to keep in the sequence length dimension.
This is useful when using packed tensor format (single dimension for batch and sequence length).`,name:"logits_to_keep"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/apertus/modeling_apertus.py#L418",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast"
>transformers.modeling_outputs.CausalLMOutputWithPast</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/apertus#transformers.ApertusConfig"
>ApertusConfig</a>) and inputs.</p>
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
`}}),N=new Ze({props:{$$slots:{default:[Lt]},$$scope:{ctx:v}}}),R=new bt({props:{anchor:"transformers.ApertusForCausalLM.forward.example",$$slots:{default:[Bt]},$$scope:{ctx:v}}}),ae=new ke({props:{title:"ApertusForTokenClassification",local:"transformers.ApertusForTokenClassification",headingTag:"h2"}}),re=new ce({props:{name:"class transformers.ApertusForTokenClassification",anchor:"transformers.ApertusForTokenClassification",parameters:[{name:"config",val:""}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/apertus/modeling_apertus.py#L484"}}),ie=new ce({props:{name:"forward",anchor:"transformers.ApertusForTokenClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.ApertusForTokenClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.ApertusForTokenClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.ApertusForTokenClassification.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.ApertusForTokenClassification.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.ApertusForTokenClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.ApertusForTokenClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.ApertusForTokenClassification.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
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
`}}),Q=new Ze({props:{$$slots:{default:[qt]},$$scope:{ctx:v}}}),le=new Ct({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/apertus.md"}}),{c(){t=u("meta"),a=r(),o=u("p"),p=r(),T=u("div"),T.innerHTML=w,l=r(),h(M.$$.fragment),O=r(),j=u("p"),j.innerHTML=we,Ce=r(),h(X.$$.fragment),xe=r(),D=u("p"),D.innerHTML=it,Ae=r(),h(H.$$.fragment),ze=r(),h(Y.$$.fragment),Ue=r(),x=u("div"),h(S.$$.fragment),Ee=r(),pe=u("p"),pe.innerHTML=lt,Xe=r(),ue=u("p"),ue.innerHTML=dt,He=r(),h(V.$$.fragment),Fe=r(),h(K.$$.fragment),Ie=r(),$=u("div"),h(ee.$$.fragment),Ve=r(),me=u("p"),me.textContent=ct,Ge=r(),he=u("p"),he.innerHTML=pt,Ne=r(),fe=u("p"),fe.innerHTML=ut,Re=r(),J=u("div"),h(te.$$.fragment),Qe=r(),ge=u("p"),ge.innerHTML=mt,Oe=r(),h(G.$$.fragment),je=r(),h(oe.$$.fragment),Je=r(),C=u("div"),h(ne.$$.fragment),De=r(),_e=u("p"),_e.textContent=ht,Ye=r(),ye=u("p"),ye.innerHTML=ft,Se=r(),be=u("p"),be.innerHTML=gt,Ke=r(),U=u("div"),h(se.$$.fragment),et=r(),ve=u("p"),ve.innerHTML=_t,tt=r(),h(N.$$.fragment),ot=r(),h(R.$$.fragment),Le=r(),h(ae.$$.fragment),Be=r(),B=u("div"),h(re.$$.fragment),nt=r(),L=u("div"),h(ie.$$.fragment),st=r(),Te=u("p"),Te.innerHTML=yt,at=r(),h(Q.$$.fragment),qe=r(),h(le.$$.fragment),We=r(),Me=u("p"),this.h()},l(e){const n=kt("svelte-u9bgzb",document.head);t=m(n,"META",{name:!0,content:!0}),n.forEach(s),a=i(e),o=m(e,"P",{}),P(o).forEach(s),p=i(e),T=m(e,"DIV",{style:!0,"data-svelte-h":!0}),k(T)!=="svelte-11gpmgv"&&(T.innerHTML=w),l=i(e),f(M.$$.fragment,e),O=i(e),j=m(e,"P",{"data-svelte-h":!0}),k(j)!=="svelte-tp6xkm"&&(j.innerHTML=we),Ce=i(e),f(X.$$.fragment,e),xe=i(e),D=m(e,"P",{"data-svelte-h":!0}),k(D)!=="svelte-x9rs6r"&&(D.innerHTML=it),Ae=i(e),f(H.$$.fragment,e),ze=i(e),f(Y.$$.fragment,e),Ue=i(e),x=m(e,"DIV",{class:!0});var F=P(x);f(S.$$.fragment,F),Ee=i(F),pe=m(F,"P",{"data-svelte-h":!0}),k(pe)!=="svelte-146d8sd"&&(pe.innerHTML=lt),Xe=i(F),ue=m(F,"P",{"data-svelte-h":!0}),k(ue)!=="svelte-1ek1ss9"&&(ue.innerHTML=dt),He=i(F),f(V.$$.fragment,F),F.forEach(s),Fe=i(e),f(K.$$.fragment,e),Ie=i(e),$=m(e,"DIV",{class:!0});var A=P($);f(ee.$$.fragment,A),Ve=i(A),me=m(A,"P",{"data-svelte-h":!0}),k(me)!=="svelte-5gca9o"&&(me.textContent=ct),Ge=i(A),he=m(A,"P",{"data-svelte-h":!0}),k(he)!=="svelte-q52n56"&&(he.innerHTML=pt),Ne=i(A),fe=m(A,"P",{"data-svelte-h":!0}),k(fe)!=="svelte-hswkmf"&&(fe.innerHTML=ut),Re=i(A),J=m(A,"DIV",{class:!0});var q=P(J);f(te.$$.fragment,q),Qe=i(q),ge=m(q,"P",{"data-svelte-h":!0}),k(ge)!=="svelte-1on34an"&&(ge.innerHTML=mt),Oe=i(q),f(G.$$.fragment,q),q.forEach(s),A.forEach(s),je=i(e),f(oe.$$.fragment,e),Je=i(e),C=m(e,"DIV",{class:!0});var z=P(C);f(ne.$$.fragment,z),De=i(z),_e=m(z,"P",{"data-svelte-h":!0}),k(_e)!=="svelte-gtanxh"&&(_e.textContent=ht),Ye=i(z),ye=m(z,"P",{"data-svelte-h":!0}),k(ye)!=="svelte-q52n56"&&(ye.innerHTML=ft),Se=i(z),be=m(z,"P",{"data-svelte-h":!0}),k(be)!=="svelte-hswkmf"&&(be.innerHTML=gt),Ke=i(z),U=m(z,"DIV",{class:!0});var I=P(U);f(se.$$.fragment,I),et=i(I),ve=m(I,"P",{"data-svelte-h":!0}),k(ve)!=="svelte-itrvbv"&&(ve.innerHTML=_t),tt=i(I),f(N.$$.fragment,I),ot=i(I),f(R.$$.fragment,I),I.forEach(s),z.forEach(s),Le=i(e),f(ae.$$.fragment,e),Be=i(e),B=m(e,"DIV",{class:!0});var de=P(B);f(re.$$.fragment,de),nt=i(de),L=m(de,"DIV",{class:!0});var W=P(L);f(ie.$$.fragment,W),st=i(W),Te=m(W,"P",{"data-svelte-h":!0}),k(Te)!=="svelte-1py4aay"&&(Te.innerHTML=yt),at=i(W),f(Q.$$.fragment,W),W.forEach(s),de.forEach(s),qe=i(e),f(le.$$.fragment,e),We=i(e),Me=m(e,"P",{}),P(Me).forEach(s),this.h()},h(){Z(t,"name","hf:doc:metadata"),Z(t,"content",Pt),$t(T,"float","right"),Z(x,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),Z(J,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),Z($,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),Z(U,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),Z(C,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),Z(L,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),Z(B,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,n){d(document.head,t),c(e,a,n),c(e,o,n),c(e,p,n),c(e,T,n),c(e,l,n),g(M,e,n),c(e,O,n),c(e,j,n),c(e,Ce,n),g(X,e,n),c(e,xe,n),c(e,D,n),c(e,Ae,n),g(H,e,n),c(e,ze,n),g(Y,e,n),c(e,Ue,n),c(e,x,n),g(S,x,null),d(x,Ee),d(x,pe),d(x,Xe),d(x,ue),d(x,He),g(V,x,null),c(e,Fe,n),g(K,e,n),c(e,Ie,n),c(e,$,n),g(ee,$,null),d($,Ve),d($,me),d($,Ge),d($,he),d($,Ne),d($,fe),d($,Re),d($,J),g(te,J,null),d(J,Qe),d(J,ge),d(J,Oe),g(G,J,null),c(e,je,n),g(oe,e,n),c(e,Je,n),c(e,C,n),g(ne,C,null),d(C,De),d(C,_e),d(C,Ye),d(C,ye),d(C,Se),d(C,be),d(C,Ke),d(C,U),g(se,U,null),d(U,et),d(U,ve),d(U,tt),g(N,U,null),d(U,ot),g(R,U,null),c(e,Le,n),g(ae,e,n),c(e,Be,n),c(e,B,n),g(re,B,null),d(B,nt),d(B,L),g(ie,L,null),d(L,st),d(L,Te),d(L,at),g(Q,L,null),c(e,qe,n),g(le,e,n),c(e,We,n),c(e,Me,n),Pe=!0},p(e,[n]){const F={};n&2&&(F.$$scope={dirty:n,ctx:e}),X.$set(F);const A={};n&2&&(A.$$scope={dirty:n,ctx:e}),H.$set(A);const q={};n&2&&(q.$$scope={dirty:n,ctx:e}),V.$set(q);const z={};n&2&&(z.$$scope={dirty:n,ctx:e}),G.$set(z);const I={};n&2&&(I.$$scope={dirty:n,ctx:e}),N.$set(I);const de={};n&2&&(de.$$scope={dirty:n,ctx:e}),R.$set(de);const W={};n&2&&(W.$$scope={dirty:n,ctx:e}),Q.$set(W)},i(e){Pe||(_(M.$$.fragment,e),_(X.$$.fragment,e),_(H.$$.fragment,e),_(Y.$$.fragment,e),_(S.$$.fragment,e),_(V.$$.fragment,e),_(K.$$.fragment,e),_(ee.$$.fragment,e),_(te.$$.fragment,e),_(G.$$.fragment,e),_(oe.$$.fragment,e),_(ne.$$.fragment,e),_(se.$$.fragment,e),_(N.$$.fragment,e),_(R.$$.fragment,e),_(ae.$$.fragment,e),_(re.$$.fragment,e),_(ie.$$.fragment,e),_(Q.$$.fragment,e),_(le.$$.fragment,e),Pe=!0)},o(e){y(M.$$.fragment,e),y(X.$$.fragment,e),y(H.$$.fragment,e),y(Y.$$.fragment,e),y(S.$$.fragment,e),y(V.$$.fragment,e),y(K.$$.fragment,e),y(ee.$$.fragment,e),y(te.$$.fragment,e),y(G.$$.fragment,e),y(oe.$$.fragment,e),y(ne.$$.fragment,e),y(se.$$.fragment,e),y(N.$$.fragment,e),y(R.$$.fragment,e),y(ae.$$.fragment,e),y(re.$$.fragment,e),y(ie.$$.fragment,e),y(Q.$$.fragment,e),y(le.$$.fragment,e),Pe=!1},d(e){e&&(s(a),s(o),s(p),s(T),s(l),s(O),s(j),s(Ce),s(xe),s(D),s(Ae),s(ze),s(Ue),s(x),s(Fe),s(Ie),s($),s(je),s(Je),s(C),s(Le),s(Be),s(B),s(qe),s(We),s(Me)),s(t),b(M,e),b(X,e),b(H,e),b(Y,e),b(S),b(V),b(K,e),b(ee),b(te),b(G),b(oe,e),b(ne),b(se),b(N),b(R),b(ae,e),b(re),b(ie),b(Q),b(le,e)}}}const Pt='{"title":"Apertus","local":"apertus","sections":[{"title":"ApertusConfig","local":"transformers.ApertusConfig","sections":[],"depth":2},{"title":"ApertusModel","local":"transformers.ApertusModel","sections":[],"depth":2},{"title":"ApertusForCausalLM","local":"transformers.ApertusForCausalLM","sections":[],"depth":2},{"title":"ApertusForTokenClassification","local":"transformers.ApertusForTokenClassification","sections":[],"depth":2}],"depth":1}';function Zt(v){return Tt(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class Ot extends wt{constructor(t){super(),Mt(this,t,Zt,Wt,vt,{})}}export{Ot as component};
