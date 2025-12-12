import{s as _t,o as yt,n as Ze}from"../chunks/scheduler.18a86fab.js";import{S as bt,i as vt,g as d,s,r as m,A as Mt,h as c,f as n,c as a,j as se,x as _,u as p,k as G,l as Tt,y as l,a as r,v as h,d as u,t as f,w as g}from"../chunks/index.98837b22.js";import{T as ft}from"../chunks/Tip.77304350.js";import{D as _e}from"../chunks/Docstring.a1ef7999.js";import{C as Ke}from"../chunks/CodeBlock.8d0c2e8a.js";import{E as gt}from"../chunks/ExampleCodeBlock.8c3ee1f9.js";import{H as ae,E as wt}from"../chunks/getInferenceSnippets.06c2775f.js";function kt(U){let o,y;return o=new Ke({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEVybmllNF81TW9kZWwlMkMlMjBFcm5pZTRfNUNvbmZpZyUwQSUwQSUyMyUyMEluaXRpYWxpemluZyUyMGElMjBFcm5pZTRfNSUyMDAuM0IlMjBzdHlsZSUyMGNvbmZpZ3VyYXRpb24lMEFjb25maWd1cmF0aW9uJTIwJTNEJTIwRXJuaWU0XzVDb25maWcoKSUwQSUwQSUyMyUyMEluaXRpYWxpemluZyUyMGElMjBtb2RlbCUyMGZyb20lMjB0aGUlMjAwLjNCJTIwc3R5bGUlMjBjb25maWd1cmF0aW9uJTBBbW9kZWwlMjAlM0QlMjBFcm5pZTRfNU1vZGVsKGNvbmZpZ3VyYXRpb24pJTBBJTBBJTIzJTIwQWNjZXNzaW5nJTIwdGhlJTIwbW9kZWwlMjBjb25maWd1cmF0aW9uJTBBY29uZmlndXJhdGlvbiUyMCUzRCUyMG1vZGVsLmNvbmZpZw==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> Ernie4_5Model, Ernie4_5Config

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a Ernie4_5 0.3B style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = Ernie4_5Config()

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a model from the 0.3B style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = Ernie4_5Model(configuration)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Accessing the model configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = model.config`,wrap:!1}}),{c(){m(o.$$.fragment)},l(i){p(o.$$.fragment,i)},m(i,b){h(o,i,b),y=!0},p:Ze,i(i){y||(u(o.$$.fragment,i),y=!0)},o(i){f(o.$$.fragment,i),y=!1},d(i){g(o,i)}}}function $t(U){let o,y=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){o=d("p"),o.innerHTML=y},l(i){o=c(i,"P",{"data-svelte-h":!0}),_(o)!=="svelte-fincs2"&&(o.innerHTML=y)},m(i,b){r(i,o,b)},p:Ze,d(i){i&&n(o)}}}function xt(U){let o,y=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){o=d("p"),o.innerHTML=y},l(i){o=c(i,"P",{"data-svelte-h":!0}),_(o)!=="svelte-fincs2"&&(o.innerHTML=y)},m(i,b){r(i,o,b)},p:Ze,d(i){i&&n(o)}}}function Et(U){let o,y="Example:",i,b,x;return b=new Ke({props:{code:"",highlighted:"",wrap:!1}}),{c(){o=d("p"),o.textContent=y,i=s(),m(b.$$.fragment)},l(v){o=c(v,"P",{"data-svelte-h":!0}),_(o)!=="svelte-11lpom8"&&(o.textContent=y),i=a(v),p(b.$$.fragment,v)},m(v,j){r(v,o,j),r(v,i,j),h(b,v,j),x=!0},p:Ze,i(v){x||(u(b.$$.fragment,v),x=!0)},o(v){f(b.$$.fragment,v),x=!1},d(v){v&&(n(o),n(i)),g(b,v)}}}function Ct(U){let o,y,i,b,x,v="<em>This model was released on 2025-06-30 and added to Hugging Face Transformers on 2025-07-21.</em>",j,J,et='<div class="flex flex-wrap space-x-1"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/> <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat"/> <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/> <img alt="Tensor parallelism" src="https://img.shields.io/badge/Tensor%20parallelism-06b6d4?style=flat&amp;logoColor=white"/></div>',ye,q,be,H,ve,N,tt=`The Ernie 4.5 model was released in the <a href="https://ernie.baidu.com/blog/posts/ernie4.5/" rel="nofollow">Ernie 4.5 Model Family</a> release by baidu.
This family of models contains multiple different architectures and model sizes. This model in specific targets the base text
model without mixture of experts (moe) with 0.3B parameters in total. It uses the standard <a href="./llama">Llama</a> at its core.`,Me,V,nt='Other models from the family can be found at <a href="./ernie4_5_moe">Ernie 4.5 Moe</a>.',Te,F,ot='<img src="https://ernie.baidu.com/blog/posts/ernie4.5/overview.png"/>',we,R,ke,A,$e,Q,xe,S,st=`This model was contributed by <a href="https://huggingface.co/AntonV" rel="nofollow">Anton Vlasjuk</a>.
The original code can be found <a href="https://github.com/PaddlePaddle/ERNIE" rel="nofollow">here</a>.`,Ee,X,Ce,w,O,Pe,re,at=`This is the configuration class to store the configuration of a <a href="/docs/transformers/v4.56.2/en/model_doc/ernie4_5#transformers.Ernie4_5Model">Ernie4_5Model</a>. It is used to instantiate an Ernie 4.5
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the Ernie 4.5 0.3B.
e.g. <a href="https://huggingface.co/baidu/ERNIE-4.5-0.3B-PT" rel="nofollow">baidu/ERNIE-4.5-0.3B-PT</a>`,Be,ie,rt=`Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> and can be used to control the model outputs. Read the
documentation from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> for more information.`,We,L,Ue,D,ze,M,Y,Ge,le,it="The bare Ernie4 5 Model outputting raw hidden-states without any specific head on top.",qe,de,lt=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,He,ce,dt=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Ne,z,K,Ve,me,ct='The <a href="/docs/transformers/v4.56.2/en/model_doc/ernie4_5#transformers.Ernie4_5Model">Ernie4_5Model</a> forward method, overrides the <code>__call__</code> special method.',Re,Z,Ie,ee,je,T,te,Ae,pe,mt="The Ernie4 5 Model for causal language modeling.",Qe,he,pt=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Se,ue,ht=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Xe,E,ne,Oe,fe,ut='The <a href="/docs/transformers/v4.56.2/en/model_doc/ernie4_5#transformers.Ernie4_5ForCausalLM">Ernie4_5ForCausalLM</a> forward method, overrides the <code>__call__</code> special method.',De,P,Ye,B,Je,oe,Fe,ge,Le;return q=new ae({props:{title:"Ernie 4.5",local:"ernie-45",headingTag:"h1"}}),H=new ae({props:{title:"Overview",local:"overview",headingTag:"h2"}}),R=new ae({props:{title:"Usage Tips",local:"usage-tips",headingTag:"h2"}}),A=new ae({props:{title:"Generate text",local:"generate-text",headingTag:"h3"}}),Q=new Ke({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b01vZGVsRm9yQ2F1c2FsTE0lMkMlMjBBdXRvVG9rZW5pemVyJTBBJTBBbW9kZWxfbmFtZSUyMCUzRCUyMCUyMmJhaWR1JTJGRVJOSUUtNC41LTAuM0ItUFQlMjIlMEElMEElMjMlMjBsb2FkJTIwdGhlJTIwdG9rZW5pemVyJTIwYW5kJTIwdGhlJTIwbW9kZWwlMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZChtb2RlbF9uYW1lKSUwQW1vZGVsJTIwJTNEJTIwQXV0b01vZGVsRm9yQ2F1c2FsTE0uZnJvbV9wcmV0cmFpbmVkKCUwQSUyMCUyMCUyMCUyMG1vZGVsX25hbWUlMkMlMEElMjAlMjAlMjAlMjBkZXZpY2VfbWFwJTNEJTIyYXV0byUyMiUyQyUwQSUyMCUyMCUyMCUyMGR0eXBlJTNEdG9yY2guYmZsb2F0MTYlMkMlMEEpJTBBJTBBJTIzJTIwcHJlcGFyZSUyMHRoZSUyMG1vZGVsJTIwaW5wdXQlMEFpbnB1dHMlMjAlM0QlMjB0b2tlbml6ZXIoJTIySGV5JTJDJTIwYXJlJTIweW91JTIwY29uc2Npb3VzJTNGJTIwQ2FuJTIweW91JTIwdGFsayUyMHRvJTIwbWUlM0YlMjIlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQXByb21wdCUyMCUzRCUyMCUyMkhleSUyQyUyMGFyZSUyMHlvdSUyMGNvbnNjaW91cyUzRiUyMENhbiUyMHlvdSUyMHRhbGslMjB0byUyMG1lJTNGJTIyJTBBbWVzc2FnZXMlMjAlM0QlMjAlNUIlMEElMjAlMjAlMjAlMjAlN0IlMjJyb2xlJTIyJTNBJTIwJTIydXNlciUyMiUyQyUyMCUyMmNvbnRlbnQlMjIlM0ElMjBwcm9tcHQlN0QlMEElNUQlMEF0ZXh0JTIwJTNEJTIwdG9rZW5pemVyLmFwcGx5X2NoYXRfdGVtcGxhdGUoJTBBJTIwJTIwJTIwJTIwbWVzc2FnZXMlMkMlMEElMjAlMjAlMjAlMjB0b2tlbml6ZSUzREZhbHNlJTJDJTBBJTIwJTIwJTIwJTIwYWRkX2dlbmVyYXRpb25fcHJvbXB0JTNEVHJ1ZSUwQSklMEFtb2RlbF9pbnB1dHMlMjAlM0QlMjB0b2tlbml6ZXIoJTVCdGV4dCU1RCUyQyUyMGFkZF9zcGVjaWFsX3Rva2VucyUzREZhbHNlJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMikudG8obW9kZWwuZGV2aWNlKSUwQSUwQSUyMyUyMGNvbmR1Y3QlMjB0ZXh0JTIwY29tcGxldGlvbiUwQWdlbmVyYXRlZF9pZHMlMjAlM0QlMjBtb2RlbC5nZW5lcmF0ZSglMEElMjAlMjAlMjAlMjAqKm1vZGVsX2lucHV0cyUyQyUwQSUyMCUyMCUyMCUyMG1heF9uZXdfdG9rZW5zJTNEMzIlMkMlMEEpJTBBb3V0cHV0X2lkcyUyMCUzRCUyMGdlbmVyYXRlZF9pZHMlNUIwJTVEJTVCbGVuKG1vZGVsX2lucHV0cy5pbnB1dF9pZHMlNUIwJTVEKSUzQSU1RC50b2xpc3QoKSUwQSUwQSUyMyUyMGRlY29kZSUyMHRoZSUyMGdlbmVyYXRlZCUyMGlkcyUwQWdlbmVyYXRlX3RleHQlMjAlM0QlMjB0b2tlbml6ZXIuZGVjb2RlKG91dHB1dF9pZHMlMkMlMjBza2lwX3NwZWNpYWxfdG9rZW5zJTNEVHJ1ZSk=",highlighted:`<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoModelForCausalLM, AutoTokenizer

model_name = <span class="hljs-string">&quot;baidu/ERNIE-4.5-0.3B-PT&quot;</span>

<span class="hljs-comment"># load the tokenizer and the model</span>
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map=<span class="hljs-string">&quot;auto&quot;</span>,
    dtype=torch.bfloat16,
)

<span class="hljs-comment"># prepare the model input</span>
inputs = tokenizer(<span class="hljs-string">&quot;Hey, are you conscious? Can you talk to me?&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
prompt = <span class="hljs-string">&quot;Hey, are you conscious? Can you talk to me?&quot;</span>
messages = [
    {<span class="hljs-string">&quot;role&quot;</span>: <span class="hljs-string">&quot;user&quot;</span>, <span class="hljs-string">&quot;content&quot;</span>: prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=<span class="hljs-literal">False</span>,
    add_generation_prompt=<span class="hljs-literal">True</span>
)
model_inputs = tokenizer([text], add_special_tokens=<span class="hljs-literal">False</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>).to(model.device)

<span class="hljs-comment"># conduct text completion</span>
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=<span class="hljs-number">32</span>,
)
output_ids = generated_ids[<span class="hljs-number">0</span>][<span class="hljs-built_in">len</span>(model_inputs.input_ids[<span class="hljs-number">0</span>]):].tolist()

<span class="hljs-comment"># decode the generated ids</span>
generate_text = tokenizer.decode(output_ids, skip_special_tokens=<span class="hljs-literal">True</span>)`,wrap:!1}}),X=new ae({props:{title:"Ernie4_5Config",local:"transformers.Ernie4_5Config",headingTag:"h2"}}),O=new _e({props:{name:"class transformers.Ernie4_5Config",anchor:"transformers.Ernie4_5Config",parameters:[{name:"vocab_size",val:" = 103424"},{name:"hidden_size",val:" = 1024"},{name:"intermediate_size",val:" = 3072"},{name:"num_hidden_layers",val:" = 18"},{name:"num_attention_heads",val:" = 16"},{name:"num_key_value_heads",val:" = 2"},{name:"hidden_act",val:" = 'silu'"},{name:"max_position_embeddings",val:" = 131072"},{name:"initializer_range",val:" = 0.02"},{name:"rms_norm_eps",val:" = 1e-05"},{name:"use_cache",val:" = True"},{name:"pad_token_id",val:" = 0"},{name:"bos_token_id",val:" = 1"},{name:"eos_token_id",val:" = 2"},{name:"tie_word_embeddings",val:" = True"},{name:"rope_theta",val:" = 500000.0"},{name:"rope_scaling",val:" = None"},{name:"use_bias",val:" = False"},{name:"head_dim",val:" = 128"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.Ernie4_5Config.vocab_size",description:`<strong>vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 103424) &#x2014;
Vocabulary size of the Ernie 4.5 model. Defines the number of different tokens that can be represented by the
<code>inputs_ids</code> passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/ernie4_5#transformers.Ernie4_5Model">Ernie4_5Model</a>`,name:"vocab_size"},{anchor:"transformers.Ernie4_5Config.hidden_size",description:`<strong>hidden_size</strong> (<code>int</code>, <em>optional</em>, defaults to 1024) &#x2014;
Dimension of the hidden representations.`,name:"hidden_size"},{anchor:"transformers.Ernie4_5Config.intermediate_size",description:`<strong>intermediate_size</strong> (<code>int</code>, <em>optional</em>, defaults to 3072) &#x2014;
Dimension of the MLP representations.`,name:"intermediate_size"},{anchor:"transformers.Ernie4_5Config.num_hidden_layers",description:`<strong>num_hidden_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 18) &#x2014;
Number of hidden layers in the Transformer decoder.`,name:"num_hidden_layers"},{anchor:"transformers.Ernie4_5Config.num_attention_heads",description:`<strong>num_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 16) &#x2014;
Number of attention heads for each attention layer in the Transformer decoder.`,name:"num_attention_heads"},{anchor:"transformers.Ernie4_5Config.num_key_value_heads",description:`<strong>num_key_value_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 2) &#x2014;
This is the number of key_value heads that should be used to implement Grouped Query Attention. If
<code>num_key_value_heads=num_attention_heads</code>, the model will use Multi Head Attention (MHA), if
<code>num_key_value_heads=1</code> the model will use Multi Query Attention (MQA) otherwise GQA is used. When
converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
by meanpooling all the original heads within that group. For more details, check out <a href="https://huggingface.co/papers/2305.13245" rel="nofollow">this
paper</a>. If it is not specified, will default to
<code>num_attention_heads</code>.`,name:"num_key_value_heads"},{anchor:"transformers.Ernie4_5Config.hidden_act",description:`<strong>hidden_act</strong> (<code>str</code> or <code>function</code>, <em>optional</em>, defaults to <code>&quot;silu&quot;</code>) &#x2014;
The non-linear activation function (function or string) in the decoder.`,name:"hidden_act"},{anchor:"transformers.Ernie4_5Config.max_position_embeddings",description:`<strong>max_position_embeddings</strong> (<code>int</code>, <em>optional</em>, defaults to 131072) &#x2014;
The maximum sequence length that this model might ever be used with.`,name:"max_position_embeddings"},{anchor:"transformers.Ernie4_5Config.initializer_range",description:`<strong>initializer_range</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.`,name:"initializer_range"},{anchor:"transformers.Ernie4_5Config.rms_norm_eps",description:`<strong>rms_norm_eps</strong> (<code>float</code>, <em>optional</em>, defaults to 1e-05) &#x2014;
The epsilon used by the rms normalization layers.`,name:"rms_norm_eps"},{anchor:"transformers.Ernie4_5Config.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not the model should return the last key/values attentions.`,name:"use_cache"},{anchor:"transformers.Ernie4_5Config.pad_token_id",description:`<strong>pad_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 0) &#x2014;
Padding token id.`,name:"pad_token_id"},{anchor:"transformers.Ernie4_5Config.bos_token_id",description:`<strong>bos_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 1) &#x2014;
Beginning of stream token id.`,name:"bos_token_id"},{anchor:"transformers.Ernie4_5Config.eos_token_id",description:`<strong>eos_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 2) &#x2014;
End of stream token id.`,name:"eos_token_id"},{anchor:"transformers.Ernie4_5Config.tie_word_embeddings",description:`<strong>tie_word_embeddings</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether to tie weight embeddings`,name:"tie_word_embeddings"},{anchor:"transformers.Ernie4_5Config.rope_theta",description:`<strong>rope_theta</strong> (<code>float</code>, <em>optional</em>, defaults to 500000.0) &#x2014;
The base period of the RoPE embeddings.`,name:"rope_theta"},{anchor:"transformers.Ernie4_5Config.rope_scaling",description:`<strong>rope_scaling</strong> (<code>Dict</code>, <em>optional</em>) &#x2014;
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
Only used with &#x2018;llama3&#x2019;. Scaling factor applied to high frequency components of the RoPE`,name:"rope_scaling"},{anchor:"transformers.Ernie4_5Config.use_bias",description:`<strong>use_bias</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether to use a bias in any of the projections including mlp and attention for example.`,name:"use_bias"},{anchor:"transformers.Ernie4_5Config.head_dim",description:`<strong>head_dim</strong> (<code>int</code>, <em>optional</em>, defaults to 128) &#x2014;
The attention head dimension. If None, it will default to hidden_size // num_attention_heads`,name:"head_dim"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/ernie4_5/configuration_ernie4_5.py#L20"}}),L=new gt({props:{anchor:"transformers.Ernie4_5Config.example",$$slots:{default:[kt]},$$scope:{ctx:U}}}),D=new ae({props:{title:"Ernie4_5Model",local:"transformers.Ernie4_5Model",headingTag:"h2"}}),Y=new _e({props:{name:"class transformers.Ernie4_5Model",anchor:"transformers.Ernie4_5Model",parameters:[{name:"config",val:": Ernie4_5Config"}],parametersDescription:[{anchor:"transformers.Ernie4_5Model.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/ernie4_5#transformers.Ernie4_5Config">Ernie4_5Config</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/ernie4_5/modeling_ernie4_5.py#L328"}}),K=new _e({props:{name:"forward",anchor:"transformers.Ernie4_5Model.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.Ernie4_5Model.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.Ernie4_5Model.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.Ernie4_5Model.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.Ernie4_5Model.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.Ernie4_5Model.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.Ernie4_5Model.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"},{anchor:"transformers.Ernie4_5Model.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/ernie4_5/modeling_ernie4_5.py#L345",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast"
>transformers.modeling_outputs.BaseModelOutputWithPast</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/ernie4_5#transformers.Ernie4_5Config"
>Ernie4_5Config</a>) and inputs.</p>
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
`}}),Z=new ft({props:{$$slots:{default:[$t]},$$scope:{ctx:U}}}),ee=new ae({props:{title:"Ernie4_5ForCausalLM",local:"transformers.Ernie4_5ForCausalLM",headingTag:"h2"}}),te=new _e({props:{name:"class transformers.Ernie4_5ForCausalLM",anchor:"transformers.Ernie4_5ForCausalLM",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.Ernie4_5ForCausalLM.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/ernie4_5#transformers.Ernie4_5ForCausalLM">Ernie4_5ForCausalLM</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/ernie4_5/modeling_ernie4_5.py#L407"}}),ne=new _e({props:{name:"forward",anchor:"transformers.Ernie4_5ForCausalLM.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"},{name:"logits_to_keep",val:": typing.Union[int, torch.Tensor] = 0"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.Ernie4_5ForCausalLM.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.Ernie4_5ForCausalLM.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.Ernie4_5ForCausalLM.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.Ernie4_5ForCausalLM.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.Ernie4_5ForCausalLM.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.Ernie4_5ForCausalLM.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.Ernie4_5ForCausalLM.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.Ernie4_5ForCausalLM.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"},{anchor:"transformers.Ernie4_5ForCausalLM.forward.logits_to_keep",description:`<strong>logits_to_keep</strong> (<code>Union[int, torch.Tensor]</code>, defaults to <code>0</code>) &#x2014;
If an <code>int</code>, compute logits for the last <code>logits_to_keep</code> tokens. If <code>0</code>, calculate logits for all
<code>input_ids</code> (special case). Only last token logits are needed for generation, and calculating them only for that
token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
If a <code>torch.Tensor</code>, must be 1D corresponding to the indices to keep in the sequence length dimension.
This is useful when using packed tensor format (single dimension for batch and sequence length).`,name:"logits_to_keep"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/ernie4_5/modeling_ernie4_5.py#L421",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast"
>transformers.modeling_outputs.CausalLMOutputWithPast</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/ernie4_5#transformers.Ernie4_5Config"
>Ernie4_5Config</a>) and inputs.</p>
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
`}}),P=new ft({props:{$$slots:{default:[xt]},$$scope:{ctx:U}}}),B=new gt({props:{anchor:"transformers.Ernie4_5ForCausalLM.forward.example",$$slots:{default:[Et]},$$scope:{ctx:U}}}),oe=new wt({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/ernie4_5.md"}}),{c(){o=d("meta"),y=s(),i=d("p"),b=s(),x=d("p"),x.innerHTML=v,j=s(),J=d("div"),J.innerHTML=et,ye=s(),m(q.$$.fragment),be=s(),m(H.$$.fragment),ve=s(),N=d("p"),N.innerHTML=tt,Me=s(),V=d("p"),V.innerHTML=nt,Te=s(),F=d("div"),F.innerHTML=ot,we=s(),m(R.$$.fragment),ke=s(),m(A.$$.fragment),$e=s(),m(Q.$$.fragment),xe=s(),S=d("p"),S.innerHTML=st,Ee=s(),m(X.$$.fragment),Ce=s(),w=d("div"),m(O.$$.fragment),Pe=s(),re=d("p"),re.innerHTML=at,Be=s(),ie=d("p"),ie.innerHTML=rt,We=s(),m(L.$$.fragment),Ue=s(),m(D.$$.fragment),ze=s(),M=d("div"),m(Y.$$.fragment),Ge=s(),le=d("p"),le.textContent=it,qe=s(),de=d("p"),de.innerHTML=lt,He=s(),ce=d("p"),ce.innerHTML=dt,Ne=s(),z=d("div"),m(K.$$.fragment),Ve=s(),me=d("p"),me.innerHTML=ct,Re=s(),m(Z.$$.fragment),Ie=s(),m(ee.$$.fragment),je=s(),T=d("div"),m(te.$$.fragment),Ae=s(),pe=d("p"),pe.textContent=mt,Qe=s(),he=d("p"),he.innerHTML=pt,Se=s(),ue=d("p"),ue.innerHTML=ht,Xe=s(),E=d("div"),m(ne.$$.fragment),Oe=s(),fe=d("p"),fe.innerHTML=ut,De=s(),m(P.$$.fragment),Ye=s(),m(B.$$.fragment),Je=s(),m(oe.$$.fragment),Fe=s(),ge=d("p"),this.h()},l(e){const t=Mt("svelte-u9bgzb",document.head);o=c(t,"META",{name:!0,content:!0}),t.forEach(n),y=a(e),i=c(e,"P",{}),se(i).forEach(n),b=a(e),x=c(e,"P",{"data-svelte-h":!0}),_(x)!=="svelte-z3yyfn"&&(x.innerHTML=v),j=a(e),J=c(e,"DIV",{style:!0,"data-svelte-h":!0}),_(J)!=="svelte-11gpmgv"&&(J.innerHTML=et),ye=a(e),p(q.$$.fragment,e),be=a(e),p(H.$$.fragment,e),ve=a(e),N=c(e,"P",{"data-svelte-h":!0}),_(N)!=="svelte-1r3vik8"&&(N.innerHTML=tt),Me=a(e),V=c(e,"P",{"data-svelte-h":!0}),_(V)!=="svelte-rjlbzv"&&(V.innerHTML=nt),Te=a(e),F=c(e,"DIV",{class:!0,"data-svelte-h":!0}),_(F)!=="svelte-xmifer"&&(F.innerHTML=ot),we=a(e),p(R.$$.fragment,e),ke=a(e),p(A.$$.fragment,e),$e=a(e),p(Q.$$.fragment,e),xe=a(e),S=c(e,"P",{"data-svelte-h":!0}),_(S)!=="svelte-wizmvu"&&(S.innerHTML=st),Ee=a(e),p(X.$$.fragment,e),Ce=a(e),w=c(e,"DIV",{class:!0});var C=se(w);p(O.$$.fragment,C),Pe=a(C),re=c(C,"P",{"data-svelte-h":!0}),_(re)!=="svelte-xhmqtc"&&(re.innerHTML=at),Be=a(C),ie=c(C,"P",{"data-svelte-h":!0}),_(ie)!=="svelte-1ek1ss9"&&(ie.innerHTML=rt),We=a(C),p(L.$$.fragment,C),C.forEach(n),Ue=a(e),p(D.$$.fragment,e),ze=a(e),M=c(e,"DIV",{class:!0});var k=se(M);p(Y.$$.fragment,k),Ge=a(k),le=c(k,"P",{"data-svelte-h":!0}),_(le)!=="svelte-c5rec4"&&(le.textContent=it),qe=a(k),de=c(k,"P",{"data-svelte-h":!0}),_(de)!=="svelte-q52n56"&&(de.innerHTML=lt),He=a(k),ce=c(k,"P",{"data-svelte-h":!0}),_(ce)!=="svelte-hswkmf"&&(ce.innerHTML=dt),Ne=a(k),z=c(k,"DIV",{class:!0});var I=se(z);p(K.$$.fragment,I),Ve=a(I),me=c(I,"P",{"data-svelte-h":!0}),_(me)!=="svelte-bcjtxo"&&(me.innerHTML=ct),Re=a(I),p(Z.$$.fragment,I),I.forEach(n),k.forEach(n),Ie=a(e),p(ee.$$.fragment,e),je=a(e),T=c(e,"DIV",{class:!0});var $=se(T);p(te.$$.fragment,$),Ae=a($),pe=c($,"P",{"data-svelte-h":!0}),_(pe)!=="svelte-1xfcb6b"&&(pe.textContent=mt),Qe=a($),he=c($,"P",{"data-svelte-h":!0}),_(he)!=="svelte-q52n56"&&(he.innerHTML=pt),Se=a($),ue=c($,"P",{"data-svelte-h":!0}),_(ue)!=="svelte-hswkmf"&&(ue.innerHTML=ht),Xe=a($),E=c($,"DIV",{class:!0});var W=se(E);p(ne.$$.fragment,W),Oe=a(W),fe=c(W,"P",{"data-svelte-h":!0}),_(fe)!=="svelte-1jvutg4"&&(fe.innerHTML=ut),De=a(W),p(P.$$.fragment,W),Ye=a(W),p(B.$$.fragment,W),W.forEach(n),$.forEach(n),Je=a(e),p(oe.$$.fragment,e),Fe=a(e),ge=c(e,"P",{}),se(ge).forEach(n),this.h()},h(){G(o,"name","hf:doc:metadata"),G(o,"content",Ut),Tt(J,"float","right"),G(F,"class","flex justify-center"),G(w,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),G(z,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),G(M,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),G(E,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),G(T,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,t){l(document.head,o),r(e,y,t),r(e,i,t),r(e,b,t),r(e,x,t),r(e,j,t),r(e,J,t),r(e,ye,t),h(q,e,t),r(e,be,t),h(H,e,t),r(e,ve,t),r(e,N,t),r(e,Me,t),r(e,V,t),r(e,Te,t),r(e,F,t),r(e,we,t),h(R,e,t),r(e,ke,t),h(A,e,t),r(e,$e,t),h(Q,e,t),r(e,xe,t),r(e,S,t),r(e,Ee,t),h(X,e,t),r(e,Ce,t),r(e,w,t),h(O,w,null),l(w,Pe),l(w,re),l(w,Be),l(w,ie),l(w,We),h(L,w,null),r(e,Ue,t),h(D,e,t),r(e,ze,t),r(e,M,t),h(Y,M,null),l(M,Ge),l(M,le),l(M,qe),l(M,de),l(M,He),l(M,ce),l(M,Ne),l(M,z),h(K,z,null),l(z,Ve),l(z,me),l(z,Re),h(Z,z,null),r(e,Ie,t),h(ee,e,t),r(e,je,t),r(e,T,t),h(te,T,null),l(T,Ae),l(T,pe),l(T,Qe),l(T,he),l(T,Se),l(T,ue),l(T,Xe),l(T,E),h(ne,E,null),l(E,Oe),l(E,fe),l(E,De),h(P,E,null),l(E,Ye),h(B,E,null),r(e,Je,t),h(oe,e,t),r(e,Fe,t),r(e,ge,t),Le=!0},p(e,[t]){const C={};t&2&&(C.$$scope={dirty:t,ctx:e}),L.$set(C);const k={};t&2&&(k.$$scope={dirty:t,ctx:e}),Z.$set(k);const I={};t&2&&(I.$$scope={dirty:t,ctx:e}),P.$set(I);const $={};t&2&&($.$$scope={dirty:t,ctx:e}),B.$set($)},i(e){Le||(u(q.$$.fragment,e),u(H.$$.fragment,e),u(R.$$.fragment,e),u(A.$$.fragment,e),u(Q.$$.fragment,e),u(X.$$.fragment,e),u(O.$$.fragment,e),u(L.$$.fragment,e),u(D.$$.fragment,e),u(Y.$$.fragment,e),u(K.$$.fragment,e),u(Z.$$.fragment,e),u(ee.$$.fragment,e),u(te.$$.fragment,e),u(ne.$$.fragment,e),u(P.$$.fragment,e),u(B.$$.fragment,e),u(oe.$$.fragment,e),Le=!0)},o(e){f(q.$$.fragment,e),f(H.$$.fragment,e),f(R.$$.fragment,e),f(A.$$.fragment,e),f(Q.$$.fragment,e),f(X.$$.fragment,e),f(O.$$.fragment,e),f(L.$$.fragment,e),f(D.$$.fragment,e),f(Y.$$.fragment,e),f(K.$$.fragment,e),f(Z.$$.fragment,e),f(ee.$$.fragment,e),f(te.$$.fragment,e),f(ne.$$.fragment,e),f(P.$$.fragment,e),f(B.$$.fragment,e),f(oe.$$.fragment,e),Le=!1},d(e){e&&(n(y),n(i),n(b),n(x),n(j),n(J),n(ye),n(be),n(ve),n(N),n(Me),n(V),n(Te),n(F),n(we),n(ke),n($e),n(xe),n(S),n(Ee),n(Ce),n(w),n(Ue),n(ze),n(M),n(Ie),n(je),n(T),n(Je),n(Fe),n(ge)),n(o),g(q,e),g(H,e),g(R,e),g(A,e),g(Q,e),g(X,e),g(O),g(L),g(D,e),g(Y),g(K),g(Z),g(ee,e),g(te),g(ne),g(P),g(B),g(oe,e)}}}const Ut='{"title":"Ernie 4.5","local":"ernie-45","sections":[{"title":"Overview","local":"overview","sections":[],"depth":2},{"title":"Usage Tips","local":"usage-tips","sections":[{"title":"Generate text","local":"generate-text","sections":[],"depth":3}],"depth":2},{"title":"Ernie4_5Config","local":"transformers.Ernie4_5Config","sections":[],"depth":2},{"title":"Ernie4_5Model","local":"transformers.Ernie4_5Model","sections":[],"depth":2},{"title":"Ernie4_5ForCausalLM","local":"transformers.Ernie4_5ForCausalLM","sections":[],"depth":2}],"depth":1}';function zt(U){return yt(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class Bt extends bt{constructor(o){super(),vt(this,o,zt,Ct,_t,{})}}export{Bt as component};
