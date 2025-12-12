import{s as Yo,o as Ko,n as ke}from"../chunks/scheduler.18a86fab.js";import{S as et,i as ot,g as c,s as a,r as p,A as tt,h as l,f as n,c as r,j as O,u as h,x as v,k as M,y as i,a as d,v as u,d as m,t as f,w as g}from"../chunks/index.98837b22.js";import{T as Ue}from"../chunks/Tip.77304350.js";import{D as P}from"../chunks/Docstring.a1ef7999.js";import{C as Xo}from"../chunks/CodeBlock.8d0c2e8a.js";import{E as Go}from"../chunks/ExampleCodeBlock.8c3ee1f9.js";import{H as U,E as nt}from"../chunks/getInferenceSnippets.06c2775f.js";function st(T){let o,_;return o=new Xo({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMFNlZWRPc3NNb2RlbCUyQyUyMFNlZWRPc3NDb25maWclMEElMEElMjMlMjBJbml0aWFsaXppbmclMjBhJTIwU2VlZE9zcy0zNmIlMjBzdHlsZSUyMGNvbmZpZ3VyYXRpb24lMEFjb25maWd1cmF0aW9uJTIwJTNEJTIwU2VlZE9zc0NvbmZpZygpJTBBJTBBJTIzJTIwSW5pdGlhbGl6aW5nJTIwYSUyMG1vZGVsJTIwZnJvbSUyMHRoZSUyMFNlZWRPc3MtMzZiJTIwc3R5bGUlMjBjb25maWd1cmF0aW9uJTBBbW9kZWwlMjAlM0QlMjBTZWVkT3NzTW9kZWwoY29uZmlndXJhdGlvbiklMEElMEElMjMlMjBBY2Nlc3NpbmclMjB0aGUlMjBtb2RlbCUyMGNvbmZpZ3VyYXRpb24lMEFjb25maWd1cmF0aW9uJTIwJTNEJTIwbW9kZWwuY29uZmln",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> SeedOssModel, SeedOssConfig

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a SeedOss-36b style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = SeedOssConfig()

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a model from the SeedOss-36b style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = SeedOssModel(configuration)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Accessing the model configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = model.config`,wrap:!1}}),{c(){p(o.$$.fragment)},l(s){h(o.$$.fragment,s)},m(s,b){u(o,s,b),_=!0},p:ke,i(s){_||(m(o.$$.fragment,s),_=!0)},o(s){f(o.$$.fragment,s),_=!1},d(s){g(o,s)}}}function at(T){let o,_=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){o=c("p"),o.innerHTML=_},l(s){o=l(s,"P",{"data-svelte-h":!0}),v(o)!=="svelte-fincs2"&&(o.innerHTML=_)},m(s,b){d(s,o,b)},p:ke,d(s){s&&n(o)}}}function rt(T){let o,_=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){o=c("p"),o.innerHTML=_},l(s){o=l(s,"P",{"data-svelte-h":!0}),v(o)!=="svelte-fincs2"&&(o.innerHTML=_)},m(s,b){d(s,o,b)},p:ke,d(s){s&&n(o)}}}function it(T){let o,_="Example:",s,b,S;return b=new Xo({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBTZWVkT3NzRm9yQ2F1c2FsTE0lMEElMEFtb2RlbCUyMCUzRCUyMFNlZWRPc3NGb3JDYXVzYWxMTS5mcm9tX3ByZXRyYWluZWQoJTIyQnl0ZURhbmNlLVNlZWQlMkZTZWVkT3NzLTM2QiUyMiklMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJCeXRlRGFuY2UtU2VlZCUyRlNlZWRPc3MtMzZCJTIyKSUwQSUwQXByb21wdCUyMCUzRCUyMCUyMkhleSUyQyUyMGFyZSUyMHlvdSUyMGNvbnNjaW91cyUzRiUyMENhbiUyMHlvdSUyMHRhbGslMjB0byUyMG1lJTNGJTIyJTBBaW5wdXRzJTIwJTNEJTIwdG9rZW5pemVyKHByb21wdCUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpJTBBJTBBJTIzJTIwR2VuZXJhdGUlMEFnZW5lcmF0ZV9pZHMlMjAlM0QlMjBtb2RlbC5nZW5lcmF0ZShpbnB1dHMuaW5wdXRfaWRzJTJDJTIwbWF4X2xlbmd0aCUzRDMwKSUwQXRva2VuaXplci5iYXRjaF9kZWNvZGUoZ2VuZXJhdGVfaWRzJTJDJTIwc2tpcF9zcGVjaWFsX3Rva2VucyUzRFRydWUlMkMlMjBjbGVhbl91cF90b2tlbml6YXRpb25fc3BhY2VzJTNERmFsc2UpJTVCMCU1RA==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, SeedOssForCausalLM

<span class="hljs-meta">&gt;&gt;&gt; </span>model = SeedOssForCausalLM.from_pretrained(<span class="hljs-string">&quot;ByteDance-Seed/SeedOss-36B&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;ByteDance-Seed/SeedOss-36B&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>prompt = <span class="hljs-string">&quot;Hey, are you conscious? Can you talk to me?&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(prompt, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Generate</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>generate_ids = model.generate(inputs.input_ids, max_length=<span class="hljs-number">30</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer.batch_decode(generate_ids, skip_special_tokens=<span class="hljs-literal">True</span>, clean_up_tokenization_spaces=<span class="hljs-literal">False</span>)[<span class="hljs-number">0</span>]
<span class="hljs-string">&quot;Hey, are you conscious? Can you talk to me?\\nI&#x27;m not conscious, but I can talk to you.&quot;</span>`,wrap:!1}}),{c(){o=c("p"),o.textContent=_,s=a(),p(b.$$.fragment)},l(y){o=l(y,"P",{"data-svelte-h":!0}),v(o)!=="svelte-11lpom8"&&(o.textContent=_),s=r(y),h(b.$$.fragment,y)},m(y,F){d(y,o,F),d(y,s,F),u(b,y,F),S=!0},p:ke,i(y){S||(m(b.$$.fragment,y),S=!0)},o(y){f(b.$$.fragment,y),S=!1},d(y){y&&(n(o),n(s)),g(b,y)}}}function dt(T){let o,_=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){o=c("p"),o.innerHTML=_},l(s){o=l(s,"P",{"data-svelte-h":!0}),v(o)!=="svelte-fincs2"&&(o.innerHTML=_)},m(s,b){d(s,o,b)},p:ke,d(s){s&&n(o)}}}function ct(T){let o,_=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){o=c("p"),o.innerHTML=_},l(s){o=l(s,"P",{"data-svelte-h":!0}),v(o)!=="svelte-fincs2"&&(o.innerHTML=_)},m(s,b){d(s,o,b)},p:ke,d(s){s&&n(o)}}}function lt(T){let o,_=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){o=c("p"),o.innerHTML=_},l(s){o=l(s,"P",{"data-svelte-h":!0}),v(o)!=="svelte-fincs2"&&(o.innerHTML=_)},m(s,b){d(s,o,b)},p:ke,d(s){s&&n(o)}}}function pt(T){let o,_,s,b,S,y,F,De,Y,Po="To be released with the official model launch.",Ae,K,Be,ee,Io="To be released with the official model launch.",He,oe,Ee,te,Wo="To be released with the official model launch.",Ze,ne,Je,$,se,co,we,No=`This is the configuration class to store the configuration of a <a href="/docs/transformers/v4.56.2/en/model_doc/seed_oss#transformers.SeedOssModel">SeedOssModel</a>. It is used to instantiate an SeedOss
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the SeedOss-36B.
e.g. <a href="https://huggingface.co/ByteDance-Seed/SeedOss-36B" rel="nofollow">ByteDance-Seed/SeedOss-36B</a>`,lo,$e,jo=`Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> and can be used to control the model outputs. Read the
documentation from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> for more information.`,po,Z,Re,ae,Qe,k,re,ho,xe,Uo="The bare Seed Oss Model outputting raw hidden-states without any specific head on top.",uo,Ce,Do=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,mo,Oe,Ao=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,fo,I,ie,go,Me,Bo='The <a href="/docs/transformers/v4.56.2/en/model_doc/seed_oss#transformers.SeedOssModel">SeedOssModel</a> forward method, overrides the <code>__call__</code> special method.',_o,J,Ve,de,Ge,w,ce,bo,Se,Ho="The Seed Oss Model for causal language modeling.",vo,ze,Eo=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,yo,Fe,Zo=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,To,z,le,ko,qe,Jo='The <a href="/docs/transformers/v4.56.2/en/model_doc/seed_oss#transformers.SeedOssForCausalLM">SeedOssForCausalLM</a> forward method, overrides the <code>__call__</code> special method.',wo,R,$o,Q,Xe,pe,Ye,D,he,xo,W,ue,Co,Le,Ro="The <code>GenericForSequenceClassification</code> forward method, overrides the <code>__call__</code> special method.",Oo,V,Ke,me,eo,A,fe,Mo,N,ge,So,Pe,Qo="The <code>GenericForTokenClassification</code> forward method, overrides the <code>__call__</code> special method.",zo,G,oo,_e,to,B,be,Fo,j,ve,qo,Ie,Vo="The <code>GenericForQuestionAnswering</code> forward method, overrides the <code>__call__</code> special method.",Lo,X,no,ye,so,je,ao;return S=new U({props:{title:"SeedOss",local:"seedoss",headingTag:"h1"}}),F=new U({props:{title:"Overview",local:"overview",headingTag:"h2"}}),K=new U({props:{title:"Model Details",local:"model-details",headingTag:"h3"}}),oe=new U({props:{title:"Usage tips",local:"usage-tips",headingTag:"h2"}}),ne=new U({props:{title:"SeedOssConfig",local:"transformers.SeedOssConfig",headingTag:"h2"}}),se=new P({props:{name:"class transformers.SeedOssConfig",anchor:"transformers.SeedOssConfig",parameters:[{name:"vocab_size",val:" = 155136"},{name:"hidden_size",val:" = 4096"},{name:"intermediate_size",val:" = 27648"},{name:"num_hidden_layers",val:" = 64"},{name:"num_attention_heads",val:" = 80"},{name:"num_key_value_heads",val:" = 8"},{name:"hidden_act",val:" = 'silu'"},{name:"max_position_embeddings",val:" = 524288"},{name:"initializer_range",val:" = 0.02"},{name:"rms_norm_eps",val:" = 1e-06"},{name:"use_cache",val:" = True"},{name:"pad_token_id",val:" = 1"},{name:"bos_token_id",val:" = 0"},{name:"eos_token_id",val:" = 2"},{name:"pretraining_tp",val:" = 1"},{name:"tie_word_embeddings",val:" = False"},{name:"rope_theta",val:" = 10000.0"},{name:"rope_scaling",val:" = None"},{name:"attention_bias",val:" = True"},{name:"attention_out_bias",val:" = False"},{name:"attention_dropout",val:" = 0.1"},{name:"residual_dropout",val:" = 0.1"},{name:"mlp_bias",val:" = False"},{name:"head_dim",val:" = 128"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.SeedOssConfig.vocab_size",description:`<strong>vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 155136) &#x2014;
Vocabulary size of the SeedOss model. Defines the number of different tokens that can be represented by the
<code>inputs_ids</code> passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/seed_oss#transformers.SeedOssModel">SeedOssModel</a>`,name:"vocab_size"},{anchor:"transformers.SeedOssConfig.hidden_size",description:`<strong>hidden_size</strong> (<code>int</code>, <em>optional</em>, defaults to 4096) &#x2014;
Dimension of the hidden representations.`,name:"hidden_size"},{anchor:"transformers.SeedOssConfig.intermediate_size",description:`<strong>intermediate_size</strong> (<code>int</code>, <em>optional</em>, defaults to 27648) &#x2014;
Dimension of the MLP representations.`,name:"intermediate_size"},{anchor:"transformers.SeedOssConfig.num_hidden_layers",description:`<strong>num_hidden_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 64) &#x2014;
Number of hidden layers in the Transformer decoder.`,name:"num_hidden_layers"},{anchor:"transformers.SeedOssConfig.num_attention_heads",description:`<strong>num_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 80) &#x2014;
Number of attention heads for each attention layer in the Transformer decoder.`,name:"num_attention_heads"},{anchor:"transformers.SeedOssConfig.num_key_value_heads",description:`<strong>num_key_value_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 8) &#x2014;
This is the number of key_value heads that should be used to implement Grouped Query Attention. If
<code>num_key_value_heads=num_attention_heads</code>, the model will use Multi Head Attention (MHA), if
<code>num_key_value_heads=1</code> the model will use Multi Query Attention (MQA) otherwise GQA is used. When
converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
by meanpooling all the original heads within that group. For more details, check out <a href="https://huggingface.co/papers/2305.13245" rel="nofollow">this
paper</a>. If it is not specified, will default to <code>8</code>.`,name:"num_key_value_heads"},{anchor:"transformers.SeedOssConfig.hidden_act",description:`<strong>hidden_act</strong> (<code>str</code> or <code>function</code>, <em>optional</em>, defaults to <code>&quot;silu&quot;</code>) &#x2014;
The non-linear activation function (function or string) in the decoder.`,name:"hidden_act"},{anchor:"transformers.SeedOssConfig.max_position_embeddings",description:`<strong>max_position_embeddings</strong> (<code>int</code>, <em>optional</em>, defaults to 524288) &#x2014;
The maximum sequence length that this model might ever be used with.`,name:"max_position_embeddings"},{anchor:"transformers.SeedOssConfig.initializer_range",description:`<strong>initializer_range</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.`,name:"initializer_range"},{anchor:"transformers.SeedOssConfig.rms_norm_eps",description:`<strong>rms_norm_eps</strong> (<code>float</code>, <em>optional</em>, defaults to 1e-06) &#x2014;
The epsilon used by the rms normalization layers.`,name:"rms_norm_eps"},{anchor:"transformers.SeedOssConfig.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not the model should return the last key/values attentions (not used by all models). Only
relevant if <code>config.is_decoder=True</code>.`,name:"use_cache"},{anchor:"transformers.SeedOssConfig.pad_token_id",description:`<strong>pad_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 1) &#x2014;
Padding token id.`,name:"pad_token_id"},{anchor:"transformers.SeedOssConfig.bos_token_id",description:`<strong>bos_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 0) &#x2014;
Beginning of stream token id.`,name:"bos_token_id"},{anchor:"transformers.SeedOssConfig.eos_token_id",description:`<strong>eos_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 2) &#x2014;
End of stream token id.`,name:"eos_token_id"},{anchor:"transformers.SeedOssConfig.pretraining_tp",description:`<strong>pretraining_tp</strong> (<code>int</code>, <em>optional</em>, defaults to 1) &#x2014;
Experimental feature. Tensor parallelism rank used during pretraining. Please refer to <a href="https://huggingface.co/docs/transformers/main/perf_train_gpu_many#tensor-parallelism" rel="nofollow">this
document</a> to
understand more about it. This value is necessary to ensure exact reproducibility of the pretraining
results. Please refer to <a href="https://github.com/pytorch/pytorch/issues/76232" rel="nofollow">this issue</a>.`,name:"pretraining_tp"},{anchor:"transformers.SeedOssConfig.tie_word_embeddings",description:`<strong>tie_word_embeddings</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether to tie weight embeddings`,name:"tie_word_embeddings"},{anchor:"transformers.SeedOssConfig.rope_theta",description:`<strong>rope_theta</strong> (<code>float</code>, <em>optional</em>, defaults to 10000.0) &#x2014;
The base period of the RoPE embeddings.`,name:"rope_theta"},{anchor:"transformers.SeedOssConfig.rope_scaling",description:`<strong>rope_scaling</strong> (<code>Dict</code>, <em>optional</em>) &#x2014;
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
Only used with &#x2018;llama3&#x2019;. Scaling factor applied to high frequency components of the RoPE`,name:"rope_scaling"},{anchor:"transformers.SeedOssConfig.attention_bias",description:`<strong>attention_bias</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether to use a bias in the query, key, value layers during self-attention.`,name:"attention_bias"},{anchor:"transformers.SeedOssConfig.attention_out_bias",description:`<strong>attention_out_bias</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether to use a bias in the output projection layer during self-attention.`,name:"attention_out_bias"},{anchor:"transformers.SeedOssConfig.attention_dropout",description:`<strong>attention_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout ratio for the attention probabilities.`,name:"attention_dropout"},{anchor:"transformers.SeedOssConfig.residual_dropout",description:`<strong>residual_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
Residual connection dropout value.`,name:"residual_dropout"},{anchor:"transformers.SeedOssConfig.mlp_bias",description:`<strong>mlp_bias</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether to use a bias in up_proj, down_proj and gate_proj layers in the MLP layers.`,name:"mlp_bias"},{anchor:"transformers.SeedOssConfig.head_dim",description:`<strong>head_dim</strong> (<code>int</code>, <em>optional</em>, defaults to 128) &#x2014;
The attention head dimension.`,name:"head_dim"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/seed_oss/configuration_seed_oss.py#L20"}}),Z=new Go({props:{anchor:"transformers.SeedOssConfig.example",$$slots:{default:[st]},$$scope:{ctx:T}}}),ae=new U({props:{title:"SeedOssModel",local:"transformers.SeedOssModel",headingTag:"h2"}}),re=new P({props:{name:"class transformers.SeedOssModel",anchor:"transformers.SeedOssModel",parameters:[{name:"config",val:": SeedOssConfig"}],parametersDescription:[{anchor:"transformers.SeedOssModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/seed_oss#transformers.SeedOssConfig">SeedOssConfig</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/seed_oss/modeling_seed_oss.py#L335"}}),ie=new P({props:{name:"forward",anchor:"transformers.SeedOssModel.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.SeedOssModel.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.SeedOssModel.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.SeedOssModel.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.SeedOssModel.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.SeedOssModel.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.SeedOssModel.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"},{anchor:"transformers.SeedOssModel.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/seed_oss/modeling_seed_oss.py#L352",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast"
>transformers.modeling_outputs.BaseModelOutputWithPast</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/seed_oss#transformers.SeedOssConfig"
>SeedOssConfig</a>) and inputs.</p>
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
`}}),J=new Ue({props:{$$slots:{default:[at]},$$scope:{ctx:T}}}),de=new U({props:{title:"SeedOssForCausalLM",local:"transformers.SeedOssForCausalLM",headingTag:"h2"}}),ce=new P({props:{name:"class transformers.SeedOssForCausalLM",anchor:"transformers.SeedOssForCausalLM",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.SeedOssForCausalLM.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/seed_oss#transformers.SeedOssForCausalLM">SeedOssForCausalLM</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/seed_oss/modeling_seed_oss.py#L414"}}),le=new P({props:{name:"forward",anchor:"transformers.SeedOssForCausalLM.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"},{name:"logits_to_keep",val:": typing.Union[int, torch.Tensor] = 0"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.SeedOssForCausalLM.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.SeedOssForCausalLM.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.SeedOssForCausalLM.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.SeedOssForCausalLM.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.SeedOssForCausalLM.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.SeedOssForCausalLM.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.SeedOssForCausalLM.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.SeedOssForCausalLM.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"},{anchor:"transformers.SeedOssForCausalLM.forward.logits_to_keep",description:`<strong>logits_to_keep</strong> (<code>Union[int, torch.Tensor]</code>, defaults to <code>0</code>) &#x2014;
If an <code>int</code>, compute logits for the last <code>logits_to_keep</code> tokens. If <code>0</code>, calculate logits for all
<code>input_ids</code> (special case). Only last token logits are needed for generation, and calculating them only for that
token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
If a <code>torch.Tensor</code>, must be 1D corresponding to the indices to keep in the sequence length dimension.
This is useful when using packed tensor format (single dimension for batch and sequence length).`,name:"logits_to_keep"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/seed_oss/modeling_seed_oss.py#L428",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast"
>transformers.modeling_outputs.CausalLMOutputWithPast</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/seed_oss#transformers.SeedOssConfig"
>SeedOssConfig</a>) and inputs.</p>
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
`}}),R=new Ue({props:{$$slots:{default:[rt]},$$scope:{ctx:T}}}),Q=new Go({props:{anchor:"transformers.SeedOssForCausalLM.forward.example",$$slots:{default:[it]},$$scope:{ctx:T}}}),pe=new U({props:{title:"SeedOssForSequenceClassification",local:"transformers.SeedOssForSequenceClassification",headingTag:"h2"}}),he=new P({props:{name:"class transformers.SeedOssForSequenceClassification",anchor:"transformers.SeedOssForSequenceClassification",parameters:[{name:"config",val:""}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/seed_oss/modeling_seed_oss.py#L494"}}),ue=new P({props:{name:"forward",anchor:"transformers.SeedOssForSequenceClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.SeedOssForSequenceClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.SeedOssForSequenceClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.SeedOssForSequenceClassification.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.SeedOssForSequenceClassification.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.SeedOssForSequenceClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.SeedOssForSequenceClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.SeedOssForSequenceClassification.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
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
`}}),V=new Ue({props:{$$slots:{default:[dt]},$$scope:{ctx:T}}}),me=new U({props:{title:"SeedOssForTokenClassification",local:"transformers.SeedOssForTokenClassification",headingTag:"h2"}}),fe=new P({props:{name:"class transformers.SeedOssForTokenClassification",anchor:"transformers.SeedOssForTokenClassification",parameters:[{name:"config",val:""}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/seed_oss/modeling_seed_oss.py#L498"}}),ge=new P({props:{name:"forward",anchor:"transformers.SeedOssForTokenClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.SeedOssForTokenClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.SeedOssForTokenClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.SeedOssForTokenClassification.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.SeedOssForTokenClassification.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.SeedOssForTokenClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.SeedOssForTokenClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.SeedOssForTokenClassification.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
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
`}}),G=new Ue({props:{$$slots:{default:[ct]},$$scope:{ctx:T}}}),_e=new U({props:{title:"SeedOssForQuestionAnswering",local:"transformers.SeedOssForQuestionAnswering",headingTag:"h2"}}),be=new P({props:{name:"class transformers.SeedOssForQuestionAnswering",anchor:"transformers.SeedOssForQuestionAnswering",parameters:[{name:"config",val:""}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/seed_oss/modeling_seed_oss.py#L502"}}),ve=new P({props:{name:"forward",anchor:"transformers.SeedOssForQuestionAnswering.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"start_positions",val:": typing.Optional[torch.LongTensor] = None"},{name:"end_positions",val:": typing.Optional[torch.LongTensor] = None"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.SeedOssForQuestionAnswering.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.SeedOssForQuestionAnswering.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.SeedOssForQuestionAnswering.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.SeedOssForQuestionAnswering.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.SeedOssForQuestionAnswering.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.SeedOssForQuestionAnswering.forward.start_positions",description:`<strong>start_positions</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for position (index) of the start of the labelled span for computing the token classification loss.
Positions are clamped to the length of the sequence (<code>sequence_length</code>). Position outside of the sequence
are not taken into account for computing the loss.`,name:"start_positions"},{anchor:"transformers.SeedOssForQuestionAnswering.forward.end_positions",description:`<strong>end_positions</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
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
`}}),X=new Ue({props:{$$slots:{default:[lt]},$$scope:{ctx:T}}}),ye=new nt({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/seed_oss.md"}}),{c(){o=c("meta"),_=a(),s=c("p"),b=a(),p(S.$$.fragment),y=a(),p(F.$$.fragment),De=a(),Y=c("p"),Y.textContent=Po,Ae=a(),p(K.$$.fragment),Be=a(),ee=c("p"),ee.textContent=Io,He=a(),p(oe.$$.fragment),Ee=a(),te=c("p"),te.textContent=Wo,Ze=a(),p(ne.$$.fragment),Je=a(),$=c("div"),p(se.$$.fragment),co=a(),we=c("p"),we.innerHTML=No,lo=a(),$e=c("p"),$e.innerHTML=jo,po=a(),p(Z.$$.fragment),Re=a(),p(ae.$$.fragment),Qe=a(),k=c("div"),p(re.$$.fragment),ho=a(),xe=c("p"),xe.textContent=Uo,uo=a(),Ce=c("p"),Ce.innerHTML=Do,mo=a(),Oe=c("p"),Oe.innerHTML=Ao,fo=a(),I=c("div"),p(ie.$$.fragment),go=a(),Me=c("p"),Me.innerHTML=Bo,_o=a(),p(J.$$.fragment),Ve=a(),p(de.$$.fragment),Ge=a(),w=c("div"),p(ce.$$.fragment),bo=a(),Se=c("p"),Se.textContent=Ho,vo=a(),ze=c("p"),ze.innerHTML=Eo,yo=a(),Fe=c("p"),Fe.innerHTML=Zo,To=a(),z=c("div"),p(le.$$.fragment),ko=a(),qe=c("p"),qe.innerHTML=Jo,wo=a(),p(R.$$.fragment),$o=a(),p(Q.$$.fragment),Xe=a(),p(pe.$$.fragment),Ye=a(),D=c("div"),p(he.$$.fragment),xo=a(),W=c("div"),p(ue.$$.fragment),Co=a(),Le=c("p"),Le.innerHTML=Ro,Oo=a(),p(V.$$.fragment),Ke=a(),p(me.$$.fragment),eo=a(),A=c("div"),p(fe.$$.fragment),Mo=a(),N=c("div"),p(ge.$$.fragment),So=a(),Pe=c("p"),Pe.innerHTML=Qo,zo=a(),p(G.$$.fragment),oo=a(),p(_e.$$.fragment),to=a(),B=c("div"),p(be.$$.fragment),Fo=a(),j=c("div"),p(ve.$$.fragment),qo=a(),Ie=c("p"),Ie.innerHTML=Vo,Lo=a(),p(X.$$.fragment),no=a(),p(ye.$$.fragment),so=a(),je=c("p"),this.h()},l(e){const t=tt("svelte-u9bgzb",document.head);o=l(t,"META",{name:!0,content:!0}),t.forEach(n),_=r(e),s=l(e,"P",{}),O(s).forEach(n),b=r(e),h(S.$$.fragment,e),y=r(e),h(F.$$.fragment,e),De=r(e),Y=l(e,"P",{"data-svelte-h":!0}),v(Y)!=="svelte-u40g91"&&(Y.textContent=Po),Ae=r(e),h(K.$$.fragment,e),Be=r(e),ee=l(e,"P",{"data-svelte-h":!0}),v(ee)!=="svelte-u40g91"&&(ee.textContent=Io),He=r(e),h(oe.$$.fragment,e),Ee=r(e),te=l(e,"P",{"data-svelte-h":!0}),v(te)!=="svelte-u40g91"&&(te.textContent=Wo),Ze=r(e),h(ne.$$.fragment,e),Je=r(e),$=l(e,"DIV",{class:!0});var q=O($);h(se.$$.fragment,q),co=r(q),we=l(q,"P",{"data-svelte-h":!0}),v(we)!=="svelte-pmg5dz"&&(we.innerHTML=No),lo=r(q),$e=l(q,"P",{"data-svelte-h":!0}),v($e)!=="svelte-1ek1ss9"&&($e.innerHTML=jo),po=r(q),h(Z.$$.fragment,q),q.forEach(n),Re=r(e),h(ae.$$.fragment,e),Qe=r(e),k=l(e,"DIV",{class:!0});var x=O(k);h(re.$$.fragment,x),ho=r(x),xe=l(x,"P",{"data-svelte-h":!0}),v(xe)!=="svelte-1a4gvfq"&&(xe.textContent=Uo),uo=r(x),Ce=l(x,"P",{"data-svelte-h":!0}),v(Ce)!=="svelte-q52n56"&&(Ce.innerHTML=Do),mo=r(x),Oe=l(x,"P",{"data-svelte-h":!0}),v(Oe)!=="svelte-hswkmf"&&(Oe.innerHTML=Ao),fo=r(x),I=l(x,"DIV",{class:!0});var H=O(I);h(ie.$$.fragment,H),go=r(H),Me=l(H,"P",{"data-svelte-h":!0}),v(Me)!=="svelte-1jqglmk"&&(Me.innerHTML=Bo),_o=r(H),h(J.$$.fragment,H),H.forEach(n),x.forEach(n),Ve=r(e),h(de.$$.fragment,e),Ge=r(e),w=l(e,"DIV",{class:!0});var C=O(w);h(ce.$$.fragment,C),bo=r(C),Se=l(C,"P",{"data-svelte-h":!0}),v(Se)!=="svelte-w2f3m9"&&(Se.textContent=Ho),vo=r(C),ze=l(C,"P",{"data-svelte-h":!0}),v(ze)!=="svelte-q52n56"&&(ze.innerHTML=Eo),yo=r(C),Fe=l(C,"P",{"data-svelte-h":!0}),v(Fe)!=="svelte-hswkmf"&&(Fe.innerHTML=Zo),To=r(C),z=l(C,"DIV",{class:!0});var L=O(z);h(le.$$.fragment,L),ko=r(L),qe=l(L,"P",{"data-svelte-h":!0}),v(qe)!=="svelte-135u5ag"&&(qe.innerHTML=Jo),wo=r(L),h(R.$$.fragment,L),$o=r(L),h(Q.$$.fragment,L),L.forEach(n),C.forEach(n),Xe=r(e),h(pe.$$.fragment,e),Ye=r(e),D=l(e,"DIV",{class:!0});var Te=O(D);h(he.$$.fragment,Te),xo=r(Te),W=l(Te,"DIV",{class:!0});var E=O(W);h(ue.$$.fragment,E),Co=r(E),Le=l(E,"P",{"data-svelte-h":!0}),v(Le)!=="svelte-1sal4ui"&&(Le.innerHTML=Ro),Oo=r(E),h(V.$$.fragment,E),E.forEach(n),Te.forEach(n),Ke=r(e),h(me.$$.fragment,e),eo=r(e),A=l(e,"DIV",{class:!0});var ro=O(A);h(fe.$$.fragment,ro),Mo=r(ro),N=l(ro,"DIV",{class:!0});var We=O(N);h(ge.$$.fragment,We),So=r(We),Pe=l(We,"P",{"data-svelte-h":!0}),v(Pe)!=="svelte-1py4aay"&&(Pe.innerHTML=Qo),zo=r(We),h(G.$$.fragment,We),We.forEach(n),ro.forEach(n),oo=r(e),h(_e.$$.fragment,e),to=r(e),B=l(e,"DIV",{class:!0});var io=O(B);h(be.$$.fragment,io),Fo=r(io),j=l(io,"DIV",{class:!0});var Ne=O(j);h(ve.$$.fragment,Ne),qo=r(Ne),Ie=l(Ne,"P",{"data-svelte-h":!0}),v(Ie)!=="svelte-dyrov9"&&(Ie.innerHTML=Vo),Lo=r(Ne),h(X.$$.fragment,Ne),Ne.forEach(n),io.forEach(n),no=r(e),h(ye.$$.fragment,e),so=r(e),je=l(e,"P",{}),O(je).forEach(n),this.h()},h(){M(o,"name","hf:doc:metadata"),M(o,"content",ht),M($,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),M(I,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),M(k,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),M(z,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),M(w,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),M(W,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),M(D,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),M(N,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),M(A,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),M(j,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),M(B,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,t){i(document.head,o),d(e,_,t),d(e,s,t),d(e,b,t),u(S,e,t),d(e,y,t),u(F,e,t),d(e,De,t),d(e,Y,t),d(e,Ae,t),u(K,e,t),d(e,Be,t),d(e,ee,t),d(e,He,t),u(oe,e,t),d(e,Ee,t),d(e,te,t),d(e,Ze,t),u(ne,e,t),d(e,Je,t),d(e,$,t),u(se,$,null),i($,co),i($,we),i($,lo),i($,$e),i($,po),u(Z,$,null),d(e,Re,t),u(ae,e,t),d(e,Qe,t),d(e,k,t),u(re,k,null),i(k,ho),i(k,xe),i(k,uo),i(k,Ce),i(k,mo),i(k,Oe),i(k,fo),i(k,I),u(ie,I,null),i(I,go),i(I,Me),i(I,_o),u(J,I,null),d(e,Ve,t),u(de,e,t),d(e,Ge,t),d(e,w,t),u(ce,w,null),i(w,bo),i(w,Se),i(w,vo),i(w,ze),i(w,yo),i(w,Fe),i(w,To),i(w,z),u(le,z,null),i(z,ko),i(z,qe),i(z,wo),u(R,z,null),i(z,$o),u(Q,z,null),d(e,Xe,t),u(pe,e,t),d(e,Ye,t),d(e,D,t),u(he,D,null),i(D,xo),i(D,W),u(ue,W,null),i(W,Co),i(W,Le),i(W,Oo),u(V,W,null),d(e,Ke,t),u(me,e,t),d(e,eo,t),d(e,A,t),u(fe,A,null),i(A,Mo),i(A,N),u(ge,N,null),i(N,So),i(N,Pe),i(N,zo),u(G,N,null),d(e,oo,t),u(_e,e,t),d(e,to,t),d(e,B,t),u(be,B,null),i(B,Fo),i(B,j),u(ve,j,null),i(j,qo),i(j,Ie),i(j,Lo),u(X,j,null),d(e,no,t),u(ye,e,t),d(e,so,t),d(e,je,t),ao=!0},p(e,[t]){const q={};t&2&&(q.$$scope={dirty:t,ctx:e}),Z.$set(q);const x={};t&2&&(x.$$scope={dirty:t,ctx:e}),J.$set(x);const H={};t&2&&(H.$$scope={dirty:t,ctx:e}),R.$set(H);const C={};t&2&&(C.$$scope={dirty:t,ctx:e}),Q.$set(C);const L={};t&2&&(L.$$scope={dirty:t,ctx:e}),V.$set(L);const Te={};t&2&&(Te.$$scope={dirty:t,ctx:e}),G.$set(Te);const E={};t&2&&(E.$$scope={dirty:t,ctx:e}),X.$set(E)},i(e){ao||(m(S.$$.fragment,e),m(F.$$.fragment,e),m(K.$$.fragment,e),m(oe.$$.fragment,e),m(ne.$$.fragment,e),m(se.$$.fragment,e),m(Z.$$.fragment,e),m(ae.$$.fragment,e),m(re.$$.fragment,e),m(ie.$$.fragment,e),m(J.$$.fragment,e),m(de.$$.fragment,e),m(ce.$$.fragment,e),m(le.$$.fragment,e),m(R.$$.fragment,e),m(Q.$$.fragment,e),m(pe.$$.fragment,e),m(he.$$.fragment,e),m(ue.$$.fragment,e),m(V.$$.fragment,e),m(me.$$.fragment,e),m(fe.$$.fragment,e),m(ge.$$.fragment,e),m(G.$$.fragment,e),m(_e.$$.fragment,e),m(be.$$.fragment,e),m(ve.$$.fragment,e),m(X.$$.fragment,e),m(ye.$$.fragment,e),ao=!0)},o(e){f(S.$$.fragment,e),f(F.$$.fragment,e),f(K.$$.fragment,e),f(oe.$$.fragment,e),f(ne.$$.fragment,e),f(se.$$.fragment,e),f(Z.$$.fragment,e),f(ae.$$.fragment,e),f(re.$$.fragment,e),f(ie.$$.fragment,e),f(J.$$.fragment,e),f(de.$$.fragment,e),f(ce.$$.fragment,e),f(le.$$.fragment,e),f(R.$$.fragment,e),f(Q.$$.fragment,e),f(pe.$$.fragment,e),f(he.$$.fragment,e),f(ue.$$.fragment,e),f(V.$$.fragment,e),f(me.$$.fragment,e),f(fe.$$.fragment,e),f(ge.$$.fragment,e),f(G.$$.fragment,e),f(_e.$$.fragment,e),f(be.$$.fragment,e),f(ve.$$.fragment,e),f(X.$$.fragment,e),f(ye.$$.fragment,e),ao=!1},d(e){e&&(n(_),n(s),n(b),n(y),n(De),n(Y),n(Ae),n(Be),n(ee),n(He),n(Ee),n(te),n(Ze),n(Je),n($),n(Re),n(Qe),n(k),n(Ve),n(Ge),n(w),n(Xe),n(Ye),n(D),n(Ke),n(eo),n(A),n(oo),n(to),n(B),n(no),n(so),n(je)),n(o),g(S,e),g(F,e),g(K,e),g(oe,e),g(ne,e),g(se),g(Z),g(ae,e),g(re),g(ie),g(J),g(de,e),g(ce),g(le),g(R),g(Q),g(pe,e),g(he),g(ue),g(V),g(me,e),g(fe),g(ge),g(G),g(_e,e),g(be),g(ve),g(X),g(ye,e)}}}const ht='{"title":"SeedOss","local":"seedoss","sections":[{"title":"Overview","local":"overview","sections":[{"title":"Model Details","local":"model-details","sections":[],"depth":3}],"depth":2},{"title":"Usage tips","local":"usage-tips","sections":[],"depth":2},{"title":"SeedOssConfig","local":"transformers.SeedOssConfig","sections":[],"depth":2},{"title":"SeedOssModel","local":"transformers.SeedOssModel","sections":[],"depth":2},{"title":"SeedOssForCausalLM","local":"transformers.SeedOssForCausalLM","sections":[],"depth":2},{"title":"SeedOssForSequenceClassification","local":"transformers.SeedOssForSequenceClassification","sections":[],"depth":2},{"title":"SeedOssForTokenClassification","local":"transformers.SeedOssForTokenClassification","sections":[],"depth":2},{"title":"SeedOssForQuestionAnswering","local":"transformers.SeedOssForQuestionAnswering","sections":[],"depth":2}],"depth":1}';function ut(T){return Ko(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class Tt extends et{constructor(o){super(),ot(this,o,ut,pt,Yo,{})}}export{Tt as component};
