import{s as ao,o as so,n as Le}from"../chunks/scheduler.18a86fab.js";import{S as ro,i as io,g as l,s,r as f,A as lo,h as c,f as n,c as r,j as D,x as b,u as m,k as $,y as d,a as i,v as p,d as h,t as u,w as g}from"../chunks/index.98837b22.js";import{T as Je}from"../chunks/Tip.77304350.js";import{D as I}from"../chunks/Docstring.a1ef7999.js";import{C as no}from"../chunks/CodeBlock.8d0c2e8a.js";import{E as oo}from"../chunks/ExampleCodeBlock.8c3ee1f9.js";import{H as S,E as co}from"../chunks/getInferenceSnippets.06c2775f.js";function fo(T){let o,_;return o=new no({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMERpZmZMbGFtYU1vZGVsJTJDJTIwRGlmZkxsYW1hQ29uZmlnJTBBJTBBJTIzJTIwSW5pdGlhbGl6aW5nJTIwYSUyMERpZmZMbGFtYSUyMGRpZmZsbGFtYS03YiUyMHN0eWxlJTIwY29uZmlndXJhdGlvbiUwQWNvbmZpZ3VyYXRpb24lMjAlM0QlMjBEaWZmTGxhbWFDb25maWcoKSUwQSUwQSUyMyUyMEluaXRpYWxpemluZyUyMGElMjBtb2RlbCUyMGZyb20lMjB0aGUlMjBkaWZmbGxhbWEtN2IlMjBzdHlsZSUyMGNvbmZpZ3VyYXRpb24lMEFtb2RlbCUyMCUzRCUyMERpZmZMbGFtYU1vZGVsKGNvbmZpZ3VyYXRpb24pJTBBJTBBJTIzJTIwQWNjZXNzaW5nJTIwdGhlJTIwbW9kZWwlMjBjb25maWd1cmF0aW9uJTBBY29uZmlndXJhdGlvbiUyMCUzRCUyMG1vZGVsLmNvbmZpZw==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> DiffLlamaModel, DiffLlamaConfig

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a DiffLlama diffllama-7b style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = DiffLlamaConfig()

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a model from the diffllama-7b style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = DiffLlamaModel(configuration)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Accessing the model configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = model.config`,wrap:!1}}),{c(){f(o.$$.fragment)},l(a){m(o.$$.fragment,a)},m(a,v){p(o,a,v),_=!0},p:Le,i(a){_||(h(o.$$.fragment,a),_=!0)},o(a){u(o.$$.fragment,a),_=!1},d(a){g(o,a)}}}function mo(T){let o,_=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){o=l("p"),o.innerHTML=_},l(a){o=c(a,"P",{"data-svelte-h":!0}),b(o)!=="svelte-fincs2"&&(o.innerHTML=_)},m(a,v){i(a,o,v)},p:Le,d(a){a&&n(o)}}}function po(T){let o,_=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){o=l("p"),o.innerHTML=_},l(a){o=c(a,"P",{"data-svelte-h":!0}),b(o)!=="svelte-fincs2"&&(o.innerHTML=_)},m(a,v){i(a,o,v)},p:Le,d(a){a&&n(o)}}}function ho(T){let o,_="Example:",a,v,M;return v=new no({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBEaWZmTGxhbWFGb3JDYXVzYWxMTSUwQSUwQW1vZGVsJTIwJTNEJTIwRGlmZkxsYW1hRm9yQ2F1c2FsTE0uZnJvbV9wcmV0cmFpbmVkKCUyMmdvb2dsZSUyRmRpZmZsbGFtYS03YiUyMiklMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJnb29nbGUlMkZkaWZmbGxhbWEtN2IlMjIpJTBBJTBBcHJvbXB0JTIwJTNEJTIwJTIyV2hhdCUyMGlzJTIweW91ciUyMGZhdm9yaXRlJTIwY29uZGltZW50JTNGJTIyJTBBaW5wdXRzJTIwJTNEJTIwdG9rZW5pemVyKHByb21wdCUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpJTBBJTBBJTIzJTIwR2VuZXJhdGUlMEFnZW5lcmF0ZV9pZHMlMjAlM0QlMjBtb2RlbC5nZW5lcmF0ZShpbnB1dHMuaW5wdXRfaWRzJTJDJTIwbWF4X2xlbmd0aCUzRDMwKSUwQXRva2VuaXplci5iYXRjaF9kZWNvZGUoZ2VuZXJhdGVfaWRzJTJDJTIwc2tpcF9zcGVjaWFsX3Rva2VucyUzRFRydWUlMkMlMjBjbGVhbl91cF90b2tlbml6YXRpb25fc3BhY2VzJTNERmFsc2UpJTVCMCU1RA==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, DiffLlamaForCausalLM

<span class="hljs-meta">&gt;&gt;&gt; </span>model = DiffLlamaForCausalLM.from_pretrained(<span class="hljs-string">&quot;google/diffllama-7b&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;google/diffllama-7b&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>prompt = <span class="hljs-string">&quot;What is your favorite condiment?&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(prompt, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Generate</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>generate_ids = model.generate(inputs.input_ids, max_length=<span class="hljs-number">30</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer.batch_decode(generate_ids, skip_special_tokens=<span class="hljs-literal">True</span>, clean_up_tokenization_spaces=<span class="hljs-literal">False</span>)[<span class="hljs-number">0</span>]
<span class="hljs-string">&quot;What is your favorite condiment?&quot;</span>`,wrap:!1}}),{c(){o=l("p"),o.textContent=_,a=s(),f(v.$$.fragment)},l(y){o=c(y,"P",{"data-svelte-h":!0}),b(o)!=="svelte-11lpom8"&&(o.textContent=_),a=r(y),m(v.$$.fragment,y)},m(y,U){i(y,o,U),i(y,a,U),p(v,y,U),M=!0},p:Le,i(y){M||(h(v.$$.fragment,y),M=!0)},o(y){u(v.$$.fragment,y),M=!1},d(y){y&&(n(o),n(a)),g(v,y)}}}function uo(T){let o,_=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){o=l("p"),o.innerHTML=_},l(a){o=c(a,"P",{"data-svelte-h":!0}),b(o)!=="svelte-fincs2"&&(o.innerHTML=_)},m(a,v){i(a,o,v)},p:Le,d(a){a&&n(o)}}}function go(T){let o,_=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){o=l("p"),o.innerHTML=_},l(a){o=c(a,"P",{"data-svelte-h":!0}),b(o)!=="svelte-fincs2"&&(o.innerHTML=_)},m(a,v){i(a,o,v)},p:Le,d(a){a&&n(o)}}}function _o(T){let o,_=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){o=l("p"),o.innerHTML=_},l(a){o=c(a,"P",{"data-svelte-h":!0}),b(o)!=="svelte-fincs2"&&(o.innerHTML=_)},m(a,v){i(a,o,v)},p:Le,d(a){a&&n(o)}}}function bo(T){let o,_,a,v,M,y="<em>This model was released on 2024-10-07 and added to Hugging Face Transformers on 2025-01-07.</em>",U,K,Se,B,Ht='<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/> <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat"/> <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/>',Ue,ee,Be,te,Nt=`The DiffLlama model was proposed in <a href="https://huggingface.co/papers/2410.05258" rel="nofollow">Differential Transformer</a> by Kazuma Matsumoto and .
This model is combine Llama model and Differential Transformer’s Attention.`,Ee,oe,Zt="The abstract from the paper is the following:",Ge,ne,jt="<em>Transformer tends to overallocate attention to irrelevant context. In this work, we introduce Diff Transformer, which amplifies attention to the relevant context while canceling noise. Specifically, the differential attention mechanism calculates attention scores as the difference between two separate softmax attention maps. The subtraction cancels noise, promoting the emergence of sparse attention patterns. Experimental results on language modeling show that Diff Transformer outperforms Transformer in various settings of scaling up model size and training tokens. More intriguingly, it offers notable advantages in practical applications, such as long-context modeling, key information retrieval, hallucination mitigation, in-context learning, and reduction of activation outliers. By being less distracted by irrelevant context, Diff Transformer can mitigate hallucination in question answering and text summarization. For in-context learning, Diff Transformer not only enhances accuracy but is also more robust to order permutation, which was considered as a chronic robustness issue. The results position Diff Transformer as a highly effective and promising architecture to advance large language models.</em>",Re,ae,Ve,se,Jt="The hyperparameters of this model is the same as Llama model.",Qe,re,Ye,L,ie,pt,xe,St=`This is the configuration class to store the configuration of a <a href="/docs/transformers/v4.56.2/en/model_doc/diffllama#transformers.DiffLlamaModel">DiffLlamaModel</a>. It is used to instantiate an DiffLlama
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults
will yield a similar configuration to that of the <a href="https://huggingface.co/kajuma/DiffLlama-0.3B-handcut" rel="nofollow">kajuma/DiffLlama-0.3B-handcut</a>.`,ht,Ce,Ut=`Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> and can be used to control the model outputs. Read the
documentation from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> for more information.`,ut,E,Xe,de,Ke,k,le,gt,De,Bt="The bare Diffllama Model outputting raw hidden-states without any specific head on top.",_t,Me,Et=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,bt,ze,Gt=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,vt,P,ce,yt,Fe,Rt='The <a href="/docs/transformers/v4.56.2/en/model_doc/diffllama#transformers.DiffLlamaModel">DiffLlamaModel</a> forward method, overrides the <code>__call__</code> special method.',Tt,G,et,fe,tt,w,me,kt,qe,Vt="The Diffllama Model for causal language modeling.",wt,Ie,Qt=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,$t,Pe,Yt=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Lt,z,pe,xt,Oe,Xt='The <a href="/docs/transformers/v4.56.2/en/model_doc/diffllama#transformers.DiffLlamaForCausalLM">DiffLlamaForCausalLM</a> forward method, overrides the <code>__call__</code> special method.',Ct,R,Dt,V,ot,he,nt,H,ue,Mt,O,ge,zt,We,Kt="The <code>GenericForSequenceClassification</code> forward method, overrides the <code>__call__</code> special method.",Ft,Q,at,_e,st,N,be,qt,W,ve,It,Ae,eo="The <code>GenericForQuestionAnswering</code> forward method, overrides the <code>__call__</code> special method.",Pt,Y,rt,ye,it,Z,Te,Ot,A,ke,Wt,He,to="The <code>GenericForTokenClassification</code> forward method, overrides the <code>__call__</code> special method.",At,X,dt,we,lt,je,ct;return K=new S({props:{title:"DiffLlama",local:"diffllama",headingTag:"h1"}}),ee=new S({props:{title:"Overview",local:"overview",headingTag:"h2"}}),ae=new S({props:{title:"Usage tips",local:"usage-tips",headingTag:"h3"}}),re=new S({props:{title:"DiffLlamaConfig",local:"transformers.DiffLlamaConfig",headingTag:"h2"}}),ie=new I({props:{name:"class transformers.DiffLlamaConfig",anchor:"transformers.DiffLlamaConfig",parameters:[{name:"vocab_size",val:" = 32000"},{name:"hidden_size",val:" = 2048"},{name:"intermediate_size",val:" = 8192"},{name:"num_hidden_layers",val:" = 16"},{name:"num_attention_heads",val:" = 32"},{name:"num_key_value_heads",val:" = None"},{name:"hidden_act",val:" = 'silu'"},{name:"max_position_embeddings",val:" = 2048"},{name:"initializer_range",val:" = 0.02"},{name:"rms_norm_eps",val:" = 1e-05"},{name:"use_cache",val:" = True"},{name:"pad_token_id",val:" = None"},{name:"bos_token_id",val:" = 1"},{name:"eos_token_id",val:" = 2"},{name:"tie_word_embeddings",val:" = False"},{name:"rope_theta",val:" = 10000.0"},{name:"rope_scaling",val:" = None"},{name:"attention_bias",val:" = False"},{name:"attention_dropout",val:" = 0.0"},{name:"lambda_std_dev",val:" = 0.1"},{name:"head_dim",val:" = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.DiffLlamaConfig.vocab_size",description:`<strong>vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 32000) &#x2014;
Vocabulary size of the DiffLlama model. Defines the number of different tokens that can be represented by the
<code>inputs_ids</code> passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/diffllama#transformers.DiffLlamaModel">DiffLlamaModel</a>`,name:"vocab_size"},{anchor:"transformers.DiffLlamaConfig.hidden_size",description:`<strong>hidden_size</strong> (<code>int</code>, <em>optional</em>, defaults to 2048) &#x2014;
Dimension of the hidden representations.`,name:"hidden_size"},{anchor:"transformers.DiffLlamaConfig.intermediate_size",description:`<strong>intermediate_size</strong> (<code>int</code>, <em>optional</em>, defaults to 8192) &#x2014;
Dimension of the MLP representations.`,name:"intermediate_size"},{anchor:"transformers.DiffLlamaConfig.num_hidden_layers",description:`<strong>num_hidden_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 16) &#x2014;
Number of hidden layers in the Transformer decoder.`,name:"num_hidden_layers"},{anchor:"transformers.DiffLlamaConfig.num_attention_heads",description:`<strong>num_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 32) &#x2014;
Number of attention heads for each attention layer in the Transformer decoder.`,name:"num_attention_heads"},{anchor:"transformers.DiffLlamaConfig.num_key_value_heads",description:`<strong>num_key_value_heads</strong> (<code>int</code>, <em>optional</em>) &#x2014;
This is the number of key_value heads that should be used to implement Grouped Query Attention. If
<code>num_key_value_heads=num_attention_heads</code>, the model will use Multi Head Attention (MHA), if
<code>num_key_value_heads=1</code> the model will use Multi Query Attention (MQA) otherwise GQA is used. When
converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
by meanpooling all the original heads within that group. For more details, check out <a href="https://huggingface.co/papers/2305.13245" rel="nofollow">this
paper</a>. If it is not specified, will default to
<code>num_attention_heads</code>.`,name:"num_key_value_heads"},{anchor:"transformers.DiffLlamaConfig.hidden_act",description:`<strong>hidden_act</strong> (<code>str</code> or <code>function</code>, <em>optional</em>, defaults to <code>&quot;silu&quot;</code>) &#x2014;
The non-linear activation function (function or string) in the decoder.`,name:"hidden_act"},{anchor:"transformers.DiffLlamaConfig.max_position_embeddings",description:`<strong>max_position_embeddings</strong> (<code>int</code>, <em>optional</em>, defaults to 2048) &#x2014;
The maximum sequence length that this model might ever be used with.`,name:"max_position_embeddings"},{anchor:"transformers.DiffLlamaConfig.initializer_range",description:`<strong>initializer_range</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.`,name:"initializer_range"},{anchor:"transformers.DiffLlamaConfig.rms_norm_eps",description:`<strong>rms_norm_eps</strong> (<code>float</code>, <em>optional</em>, defaults to 1e-05) &#x2014;
The epsilon used by the rms normalization layers.`,name:"rms_norm_eps"},{anchor:"transformers.DiffLlamaConfig.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not the model should return the last key/values attentions (not used by all models). Only
relevant if <code>config.is_decoder=True</code>.`,name:"use_cache"},{anchor:"transformers.DiffLlamaConfig.pad_token_id",description:`<strong>pad_token_id</strong> (<code>int</code>, <em>optional</em>) &#x2014;
Padding token id.`,name:"pad_token_id"},{anchor:"transformers.DiffLlamaConfig.bos_token_id",description:`<strong>bos_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 1) &#x2014;
Beginning of stream token id.`,name:"bos_token_id"},{anchor:"transformers.DiffLlamaConfig.eos_token_id",description:`<strong>eos_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 2) &#x2014;
End of stream token id.`,name:"eos_token_id"},{anchor:"transformers.DiffLlamaConfig.tie_word_embeddings",description:`<strong>tie_word_embeddings</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether to tie weight embeddings`,name:"tie_word_embeddings"},{anchor:"transformers.DiffLlamaConfig.rope_theta",description:`<strong>rope_theta</strong> (<code>float</code>, <em>optional</em>, defaults to 10000.0) &#x2014;
The base period of the RoPE embeddings.`,name:"rope_theta"},{anchor:"transformers.DiffLlamaConfig.rope_scaling",description:`<strong>rope_scaling</strong> (<code>Dict</code>, <em>optional</em>) &#x2014;
Dictionary containing the scaling configuration for the RoPE embeddings. NOTE: if you apply new rope type
and you expect the model to work on longer <code>max_position_embeddings</code>, we recommend you to update this value
accordingly.
Expected contents:
<code>rope_type</code> (<code>str</code>):
The sub-variant of RoPE to use. Can be one of [&#x2018;default&#x2019;, &#x2018;linear&#x2019;, &#x2018;dynamic&#x2019;, &#x2018;yarn&#x2019;, &#x2018;longrope&#x2019;,
&#x2018;diffllama3&#x2019;], with &#x2018;default&#x2019; being the original RoPE implementation.
<code>factor</code> (<code>float</code>, <em>optional</em>):
Used with all rope types except &#x2018;default&#x2019;. The scaling factor to apply to the RoPE embeddings. In
most scaling types, a <code>factor</code> of x will enable the model to handle sequences of length x <em>
original maximum pre-trained length.
<code>original_max_position_embeddings</code> (<code>int</code>, </em>optional<em>):
Used with &#x2018;dynamic&#x2019;, &#x2018;longrope&#x2019; and &#x2018;diffllama3&#x2019;. The original max position embeddings used during
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
Only used with &#x2018;diffllama3&#x2019;. Scaling factor applied to low frequency components of the RoPE
<code>high_freq_factor</code> (<code>float</code>, </em>optional*):
Only used with &#x2018;diffllama3&#x2019;. Scaling factor applied to high frequency components of the RoPE`,name:"rope_scaling"},{anchor:"transformers.DiffLlamaConfig.attention_bias",description:`<strong>attention_bias</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether to use a bias in the query, key, value and output projection layers during self-attention.`,name:"attention_bias"},{anchor:"transformers.DiffLlamaConfig.attention_dropout",description:`<strong>attention_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The dropout ratio for the attention probabilities.`,name:"attention_dropout"},{anchor:"transformers.DiffLlamaConfig.lambda_std_dev",description:`<strong>lambda_std_dev</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The standard deviation for initialization of parameter lambda in attention layer.`,name:"lambda_std_dev"},{anchor:"transformers.DiffLlamaConfig.head_dim",description:`<strong>head_dim</strong> (<code>int</code>, <em>optional</em>) &#x2014;
The attention head dimension. If None, it will default to hidden_size // num_heads`,name:"head_dim"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/diffllama/configuration_diffllama.py#L24"}}),E=new oo({props:{anchor:"transformers.DiffLlamaConfig.example",$$slots:{default:[fo]},$$scope:{ctx:T}}}),de=new S({props:{title:"DiffLlamaModel",local:"transformers.DiffLlamaModel",headingTag:"h2"}}),le=new I({props:{name:"class transformers.DiffLlamaModel",anchor:"transformers.DiffLlamaModel",parameters:[{name:"config",val:": DiffLlamaConfig"}],parametersDescription:[{anchor:"transformers.DiffLlamaModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/diffllama#transformers.DiffLlamaConfig">DiffLlamaConfig</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/diffllama/modeling_diffllama.py#L594"}}),ce=new I({props:{name:"forward",anchor:"transformers.DiffLlamaModel.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.DiffLlamaModel.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.DiffLlamaModel.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.DiffLlamaModel.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.DiffLlamaModel.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.DiffLlamaModel.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.DiffLlamaModel.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"},{anchor:"transformers.DiffLlamaModel.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/diffllama/modeling_diffllama.py#L611",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast"
>transformers.modeling_outputs.BaseModelOutputWithPast</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/diffllama#transformers.DiffLlamaConfig"
>DiffLlamaConfig</a>) and inputs.</p>
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
`}}),G=new Je({props:{$$slots:{default:[mo]},$$scope:{ctx:T}}}),fe=new S({props:{title:"DiffLlamaForCausalLM",local:"transformers.DiffLlamaForCausalLM",headingTag:"h2"}}),me=new I({props:{name:"class transformers.DiffLlamaForCausalLM",anchor:"transformers.DiffLlamaForCausalLM",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.DiffLlamaForCausalLM.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/diffllama#transformers.DiffLlamaForCausalLM">DiffLlamaForCausalLM</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/diffllama/modeling_diffllama.py#L673"}}),pe=new I({props:{name:"forward",anchor:"transformers.DiffLlamaForCausalLM.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"},{name:"logits_to_keep",val:": typing.Union[int, torch.Tensor] = 0"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.DiffLlamaForCausalLM.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.DiffLlamaForCausalLM.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.DiffLlamaForCausalLM.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.DiffLlamaForCausalLM.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.DiffLlamaForCausalLM.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.DiffLlamaForCausalLM.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.DiffLlamaForCausalLM.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.DiffLlamaForCausalLM.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"},{anchor:"transformers.DiffLlamaForCausalLM.forward.logits_to_keep",description:`<strong>logits_to_keep</strong> (<code>Union[int, torch.Tensor]</code>, defaults to <code>0</code>) &#x2014;
If an <code>int</code>, compute logits for the last <code>logits_to_keep</code> tokens. If <code>0</code>, calculate logits for all
<code>input_ids</code> (special case). Only last token logits are needed for generation, and calculating them only for that
token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
If a <code>torch.Tensor</code>, must be 1D corresponding to the indices to keep in the sequence length dimension.
This is useful when using packed tensor format (single dimension for batch and sequence length).`,name:"logits_to_keep"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/diffllama/modeling_diffllama.py#L687",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast"
>transformers.modeling_outputs.CausalLMOutputWithPast</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/diffllama#transformers.DiffLlamaConfig"
>DiffLlamaConfig</a>) and inputs.</p>
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
`}}),R=new Je({props:{$$slots:{default:[po]},$$scope:{ctx:T}}}),V=new oo({props:{anchor:"transformers.DiffLlamaForCausalLM.forward.example",$$slots:{default:[ho]},$$scope:{ctx:T}}}),he=new S({props:{title:"DiffLlamaForSequenceClassification",local:"transformers.DiffLlamaForSequenceClassification",headingTag:"h2"}}),ue=new I({props:{name:"class transformers.DiffLlamaForSequenceClassification",anchor:"transformers.DiffLlamaForSequenceClassification",parameters:[{name:"config",val:""}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/diffllama/modeling_diffllama.py#L748"}}),ge=new I({props:{name:"forward",anchor:"transformers.DiffLlamaForSequenceClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.DiffLlamaForSequenceClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.DiffLlamaForSequenceClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.DiffLlamaForSequenceClassification.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.DiffLlamaForSequenceClassification.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.DiffLlamaForSequenceClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.DiffLlamaForSequenceClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.DiffLlamaForSequenceClassification.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
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
`}}),Q=new Je({props:{$$slots:{default:[uo]},$$scope:{ctx:T}}}),_e=new S({props:{title:"DiffLlamaForQuestionAnswering",local:"transformers.DiffLlamaForQuestionAnswering",headingTag:"h2"}}),be=new I({props:{name:"class transformers.DiffLlamaForQuestionAnswering",anchor:"transformers.DiffLlamaForQuestionAnswering",parameters:[{name:"config",val:""}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/diffllama/modeling_diffllama.py#L752"}}),ve=new I({props:{name:"forward",anchor:"transformers.DiffLlamaForQuestionAnswering.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"start_positions",val:": typing.Optional[torch.LongTensor] = None"},{name:"end_positions",val:": typing.Optional[torch.LongTensor] = None"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.DiffLlamaForQuestionAnswering.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.DiffLlamaForQuestionAnswering.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.DiffLlamaForQuestionAnswering.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.DiffLlamaForQuestionAnswering.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.DiffLlamaForQuestionAnswering.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.DiffLlamaForQuestionAnswering.forward.start_positions",description:`<strong>start_positions</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for position (index) of the start of the labelled span for computing the token classification loss.
Positions are clamped to the length of the sequence (<code>sequence_length</code>). Position outside of the sequence
are not taken into account for computing the loss.`,name:"start_positions"},{anchor:"transformers.DiffLlamaForQuestionAnswering.forward.end_positions",description:`<strong>end_positions</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
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
`}}),Y=new Je({props:{$$slots:{default:[go]},$$scope:{ctx:T}}}),ye=new S({props:{title:"DiffLlamaForTokenClassification",local:"transformers.DiffLlamaForTokenClassification",headingTag:"h2"}}),Te=new I({props:{name:"class transformers.DiffLlamaForTokenClassification",anchor:"transformers.DiffLlamaForTokenClassification",parameters:[{name:"config",val:""}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/diffllama/modeling_diffllama.py#L756"}}),ke=new I({props:{name:"forward",anchor:"transformers.DiffLlamaForTokenClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.DiffLlamaForTokenClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.DiffLlamaForTokenClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.DiffLlamaForTokenClassification.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.DiffLlamaForTokenClassification.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.DiffLlamaForTokenClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.DiffLlamaForTokenClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.DiffLlamaForTokenClassification.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
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
`}}),X=new Je({props:{$$slots:{default:[_o]},$$scope:{ctx:T}}}),we=new co({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/diffllama.md"}}),{c(){o=l("meta"),_=s(),a=l("p"),v=s(),M=l("p"),M.innerHTML=y,U=s(),f(K.$$.fragment),Se=s(),B=l("div"),B.innerHTML=Ht,Ue=s(),f(ee.$$.fragment),Be=s(),te=l("p"),te.innerHTML=Nt,Ee=s(),oe=l("p"),oe.textContent=Zt,Ge=s(),ne=l("p"),ne.innerHTML=jt,Re=s(),f(ae.$$.fragment),Ve=s(),se=l("p"),se.textContent=Jt,Qe=s(),f(re.$$.fragment),Ye=s(),L=l("div"),f(ie.$$.fragment),pt=s(),xe=l("p"),xe.innerHTML=St,ht=s(),Ce=l("p"),Ce.innerHTML=Ut,ut=s(),f(E.$$.fragment),Xe=s(),f(de.$$.fragment),Ke=s(),k=l("div"),f(le.$$.fragment),gt=s(),De=l("p"),De.textContent=Bt,_t=s(),Me=l("p"),Me.innerHTML=Et,bt=s(),ze=l("p"),ze.innerHTML=Gt,vt=s(),P=l("div"),f(ce.$$.fragment),yt=s(),Fe=l("p"),Fe.innerHTML=Rt,Tt=s(),f(G.$$.fragment),et=s(),f(fe.$$.fragment),tt=s(),w=l("div"),f(me.$$.fragment),kt=s(),qe=l("p"),qe.textContent=Vt,wt=s(),Ie=l("p"),Ie.innerHTML=Qt,$t=s(),Pe=l("p"),Pe.innerHTML=Yt,Lt=s(),z=l("div"),f(pe.$$.fragment),xt=s(),Oe=l("p"),Oe.innerHTML=Xt,Ct=s(),f(R.$$.fragment),Dt=s(),f(V.$$.fragment),ot=s(),f(he.$$.fragment),nt=s(),H=l("div"),f(ue.$$.fragment),Mt=s(),O=l("div"),f(ge.$$.fragment),zt=s(),We=l("p"),We.innerHTML=Kt,Ft=s(),f(Q.$$.fragment),at=s(),f(_e.$$.fragment),st=s(),N=l("div"),f(be.$$.fragment),qt=s(),W=l("div"),f(ve.$$.fragment),It=s(),Ae=l("p"),Ae.innerHTML=eo,Pt=s(),f(Y.$$.fragment),rt=s(),f(ye.$$.fragment),it=s(),Z=l("div"),f(Te.$$.fragment),Ot=s(),A=l("div"),f(ke.$$.fragment),Wt=s(),He=l("p"),He.innerHTML=to,At=s(),f(X.$$.fragment),dt=s(),f(we.$$.fragment),lt=s(),je=l("p"),this.h()},l(e){const t=lo("svelte-u9bgzb",document.head);o=c(t,"META",{name:!0,content:!0}),t.forEach(n),_=r(e),a=c(e,"P",{}),D(a).forEach(n),v=r(e),M=c(e,"P",{"data-svelte-h":!0}),b(M)!=="svelte-mk5q0x"&&(M.innerHTML=y),U=r(e),m(K.$$.fragment,e),Se=r(e),B=c(e,"DIV",{class:!0,"data-svelte-h":!0}),b(B)!=="svelte-b95w5j"&&(B.innerHTML=Ht),Ue=r(e),m(ee.$$.fragment,e),Be=r(e),te=c(e,"P",{"data-svelte-h":!0}),b(te)!=="svelte-17avutd"&&(te.innerHTML=Nt),Ee=r(e),oe=c(e,"P",{"data-svelte-h":!0}),b(oe)!=="svelte-vfdo9a"&&(oe.textContent=Zt),Ge=r(e),ne=c(e,"P",{"data-svelte-h":!0}),b(ne)!=="svelte-1ncjn49"&&(ne.innerHTML=jt),Re=r(e),m(ae.$$.fragment,e),Ve=r(e),se=c(e,"P",{"data-svelte-h":!0}),b(se)!=="svelte-1vygrzq"&&(se.textContent=Jt),Qe=r(e),m(re.$$.fragment,e),Ye=r(e),L=c(e,"DIV",{class:!0});var F=D(L);m(ie.$$.fragment,F),pt=r(F),xe=c(F,"P",{"data-svelte-h":!0}),b(xe)!=="svelte-1ewzq1q"&&(xe.innerHTML=St),ht=r(F),Ce=c(F,"P",{"data-svelte-h":!0}),b(Ce)!=="svelte-1ek1ss9"&&(Ce.innerHTML=Ut),ut=r(F),m(E.$$.fragment,F),F.forEach(n),Xe=r(e),m(de.$$.fragment,e),Ke=r(e),k=c(e,"DIV",{class:!0});var x=D(k);m(le.$$.fragment,x),gt=r(x),De=c(x,"P",{"data-svelte-h":!0}),b(De)!=="svelte-ah2m0q"&&(De.textContent=Bt),_t=r(x),Me=c(x,"P",{"data-svelte-h":!0}),b(Me)!=="svelte-q52n56"&&(Me.innerHTML=Et),bt=r(x),ze=c(x,"P",{"data-svelte-h":!0}),b(ze)!=="svelte-hswkmf"&&(ze.innerHTML=Gt),vt=r(x),P=c(x,"DIV",{class:!0});var j=D(P);m(ce.$$.fragment,j),yt=r(j),Fe=c(j,"P",{"data-svelte-h":!0}),b(Fe)!=="svelte-16mgbwh"&&(Fe.innerHTML=Rt),Tt=r(j),m(G.$$.fragment,j),j.forEach(n),x.forEach(n),et=r(e),m(fe.$$.fragment,e),tt=r(e),w=c(e,"DIV",{class:!0});var C=D(w);m(me.$$.fragment,C),kt=r(C),qe=c(C,"P",{"data-svelte-h":!0}),b(qe)!=="svelte-1otj62z"&&(qe.textContent=Vt),wt=r(C),Ie=c(C,"P",{"data-svelte-h":!0}),b(Ie)!=="svelte-q52n56"&&(Ie.innerHTML=Qt),$t=r(C),Pe=c(C,"P",{"data-svelte-h":!0}),b(Pe)!=="svelte-hswkmf"&&(Pe.innerHTML=Yt),Lt=r(C),z=c(C,"DIV",{class:!0});var q=D(z);m(pe.$$.fragment,q),xt=r(q),Oe=c(q,"P",{"data-svelte-h":!0}),b(Oe)!=="svelte-njpz9p"&&(Oe.innerHTML=Xt),Ct=r(q),m(R.$$.fragment,q),Dt=r(q),m(V.$$.fragment,q),q.forEach(n),C.forEach(n),ot=r(e),m(he.$$.fragment,e),nt=r(e),H=c(e,"DIV",{class:!0});var $e=D(H);m(ue.$$.fragment,$e),Mt=r($e),O=c($e,"DIV",{class:!0});var J=D(O);m(ge.$$.fragment,J),zt=r(J),We=c(J,"P",{"data-svelte-h":!0}),b(We)!=="svelte-1sal4ui"&&(We.innerHTML=Kt),Ft=r(J),m(Q.$$.fragment,J),J.forEach(n),$e.forEach(n),at=r(e),m(_e.$$.fragment,e),st=r(e),N=c(e,"DIV",{class:!0});var ft=D(N);m(be.$$.fragment,ft),qt=r(ft),W=c(ft,"DIV",{class:!0});var Ne=D(W);m(ve.$$.fragment,Ne),It=r(Ne),Ae=c(Ne,"P",{"data-svelte-h":!0}),b(Ae)!=="svelte-dyrov9"&&(Ae.innerHTML=eo),Pt=r(Ne),m(Y.$$.fragment,Ne),Ne.forEach(n),ft.forEach(n),rt=r(e),m(ye.$$.fragment,e),it=r(e),Z=c(e,"DIV",{class:!0});var mt=D(Z);m(Te.$$.fragment,mt),Ot=r(mt),A=c(mt,"DIV",{class:!0});var Ze=D(A);m(ke.$$.fragment,Ze),Wt=r(Ze),He=c(Ze,"P",{"data-svelte-h":!0}),b(He)!=="svelte-1py4aay"&&(He.innerHTML=to),At=r(Ze),m(X.$$.fragment,Ze),Ze.forEach(n),mt.forEach(n),dt=r(e),m(we.$$.fragment,e),lt=r(e),je=c(e,"P",{}),D(je).forEach(n),this.h()},h(){$(o,"name","hf:doc:metadata"),$(o,"content",vo),$(B,"class","flex flex-wrap space-x-1"),$(L,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(P,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(k,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(z,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(w,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(O,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(H,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(W,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(N,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(A,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(Z,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,t){d(document.head,o),i(e,_,t),i(e,a,t),i(e,v,t),i(e,M,t),i(e,U,t),p(K,e,t),i(e,Se,t),i(e,B,t),i(e,Ue,t),p(ee,e,t),i(e,Be,t),i(e,te,t),i(e,Ee,t),i(e,oe,t),i(e,Ge,t),i(e,ne,t),i(e,Re,t),p(ae,e,t),i(e,Ve,t),i(e,se,t),i(e,Qe,t),p(re,e,t),i(e,Ye,t),i(e,L,t),p(ie,L,null),d(L,pt),d(L,xe),d(L,ht),d(L,Ce),d(L,ut),p(E,L,null),i(e,Xe,t),p(de,e,t),i(e,Ke,t),i(e,k,t),p(le,k,null),d(k,gt),d(k,De),d(k,_t),d(k,Me),d(k,bt),d(k,ze),d(k,vt),d(k,P),p(ce,P,null),d(P,yt),d(P,Fe),d(P,Tt),p(G,P,null),i(e,et,t),p(fe,e,t),i(e,tt,t),i(e,w,t),p(me,w,null),d(w,kt),d(w,qe),d(w,wt),d(w,Ie),d(w,$t),d(w,Pe),d(w,Lt),d(w,z),p(pe,z,null),d(z,xt),d(z,Oe),d(z,Ct),p(R,z,null),d(z,Dt),p(V,z,null),i(e,ot,t),p(he,e,t),i(e,nt,t),i(e,H,t),p(ue,H,null),d(H,Mt),d(H,O),p(ge,O,null),d(O,zt),d(O,We),d(O,Ft),p(Q,O,null),i(e,at,t),p(_e,e,t),i(e,st,t),i(e,N,t),p(be,N,null),d(N,qt),d(N,W),p(ve,W,null),d(W,It),d(W,Ae),d(W,Pt),p(Y,W,null),i(e,rt,t),p(ye,e,t),i(e,it,t),i(e,Z,t),p(Te,Z,null),d(Z,Ot),d(Z,A),p(ke,A,null),d(A,Wt),d(A,He),d(A,At),p(X,A,null),i(e,dt,t),p(we,e,t),i(e,lt,t),i(e,je,t),ct=!0},p(e,[t]){const F={};t&2&&(F.$$scope={dirty:t,ctx:e}),E.$set(F);const x={};t&2&&(x.$$scope={dirty:t,ctx:e}),G.$set(x);const j={};t&2&&(j.$$scope={dirty:t,ctx:e}),R.$set(j);const C={};t&2&&(C.$$scope={dirty:t,ctx:e}),V.$set(C);const q={};t&2&&(q.$$scope={dirty:t,ctx:e}),Q.$set(q);const $e={};t&2&&($e.$$scope={dirty:t,ctx:e}),Y.$set($e);const J={};t&2&&(J.$$scope={dirty:t,ctx:e}),X.$set(J)},i(e){ct||(h(K.$$.fragment,e),h(ee.$$.fragment,e),h(ae.$$.fragment,e),h(re.$$.fragment,e),h(ie.$$.fragment,e),h(E.$$.fragment,e),h(de.$$.fragment,e),h(le.$$.fragment,e),h(ce.$$.fragment,e),h(G.$$.fragment,e),h(fe.$$.fragment,e),h(me.$$.fragment,e),h(pe.$$.fragment,e),h(R.$$.fragment,e),h(V.$$.fragment,e),h(he.$$.fragment,e),h(ue.$$.fragment,e),h(ge.$$.fragment,e),h(Q.$$.fragment,e),h(_e.$$.fragment,e),h(be.$$.fragment,e),h(ve.$$.fragment,e),h(Y.$$.fragment,e),h(ye.$$.fragment,e),h(Te.$$.fragment,e),h(ke.$$.fragment,e),h(X.$$.fragment,e),h(we.$$.fragment,e),ct=!0)},o(e){u(K.$$.fragment,e),u(ee.$$.fragment,e),u(ae.$$.fragment,e),u(re.$$.fragment,e),u(ie.$$.fragment,e),u(E.$$.fragment,e),u(de.$$.fragment,e),u(le.$$.fragment,e),u(ce.$$.fragment,e),u(G.$$.fragment,e),u(fe.$$.fragment,e),u(me.$$.fragment,e),u(pe.$$.fragment,e),u(R.$$.fragment,e),u(V.$$.fragment,e),u(he.$$.fragment,e),u(ue.$$.fragment,e),u(ge.$$.fragment,e),u(Q.$$.fragment,e),u(_e.$$.fragment,e),u(be.$$.fragment,e),u(ve.$$.fragment,e),u(Y.$$.fragment,e),u(ye.$$.fragment,e),u(Te.$$.fragment,e),u(ke.$$.fragment,e),u(X.$$.fragment,e),u(we.$$.fragment,e),ct=!1},d(e){e&&(n(_),n(a),n(v),n(M),n(U),n(Se),n(B),n(Ue),n(Be),n(te),n(Ee),n(oe),n(Ge),n(ne),n(Re),n(Ve),n(se),n(Qe),n(Ye),n(L),n(Xe),n(Ke),n(k),n(et),n(tt),n(w),n(ot),n(nt),n(H),n(at),n(st),n(N),n(rt),n(it),n(Z),n(dt),n(lt),n(je)),n(o),g(K,e),g(ee,e),g(ae,e),g(re,e),g(ie),g(E),g(de,e),g(le),g(ce),g(G),g(fe,e),g(me),g(pe),g(R),g(V),g(he,e),g(ue),g(ge),g(Q),g(_e,e),g(be),g(ve),g(Y),g(ye,e),g(Te),g(ke),g(X),g(we,e)}}}const vo='{"title":"DiffLlama","local":"diffllama","sections":[{"title":"Overview","local":"overview","sections":[{"title":"Usage tips","local":"usage-tips","sections":[],"depth":3}],"depth":2},{"title":"DiffLlamaConfig","local":"transformers.DiffLlamaConfig","sections":[],"depth":2},{"title":"DiffLlamaModel","local":"transformers.DiffLlamaModel","sections":[],"depth":2},{"title":"DiffLlamaForCausalLM","local":"transformers.DiffLlamaForCausalLM","sections":[],"depth":2},{"title":"DiffLlamaForSequenceClassification","local":"transformers.DiffLlamaForSequenceClassification","sections":[],"depth":2},{"title":"DiffLlamaForQuestionAnswering","local":"transformers.DiffLlamaForQuestionAnswering","sections":[],"depth":2},{"title":"DiffLlamaForTokenClassification","local":"transformers.DiffLlamaForTokenClassification","sections":[],"depth":2}],"depth":1}';function yo(T){return so(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class Do extends ro{constructor(o){super(),io(this,o,yo,bo,ao,{})}}export{Do as component};
