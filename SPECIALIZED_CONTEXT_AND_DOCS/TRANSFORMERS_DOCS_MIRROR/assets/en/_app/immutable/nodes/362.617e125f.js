import{s as on,o as nn,n as ke}from"../chunks/scheduler.18a86fab.js";import{S as tn,i as sn,g as c,s as a,r as p,A as an,h as l,f as t,c as r,j as C,x as b,u,k as Q,y as i,a as d,v as h,d as m,t as f,w as g}from"../chunks/index.98837b22.js";import{T as He}from"../chunks/Tip.77304350.js";import{D as I}from"../chunks/Docstring.a1ef7999.js";import{C as en}from"../chunks/CodeBlock.8d0c2e8a.js";import{E as Ko}from"../chunks/ExampleCodeBlock.8c3ee1f9.js";import{H as U,E as rn}from"../chunks/getInferenceSnippets.06c2775f.js";function dn(y){let o,_;return o=new en({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMFF3ZW4zTW9lTW9kZWwlMkMlMjBRd2VuM01vZUNvbmZpZyUwQSUwQSUyMyUyMEluaXRpYWxpemluZyUyMGElMjBRd2VuM01vRSUyMHN0eWxlJTIwY29uZmlndXJhdGlvbiUwQWNvbmZpZ3VyYXRpb24lMjAlM0QlMjBRd2VuM01vZUNvbmZpZygpJTBBJTBBJTIzJTIwSW5pdGlhbGl6aW5nJTIwYSUyMG1vZGVsJTIwZnJvbSUyMHRoZSUyMFF3ZW4zLTE1Qi1BMkIlMjIlMjBzdHlsZSUyMGNvbmZpZ3VyYXRpb24lMEFtb2RlbCUyMCUzRCUyMFF3ZW4zTW9lTW9kZWwoY29uZmlndXJhdGlvbiklMEElMEElMjMlMjBBY2Nlc3NpbmclMjB0aGUlMjBtb2RlbCUyMGNvbmZpZ3VyYXRpb24lMEFjb25maWd1cmF0aW9uJTIwJTNEJTIwbW9kZWwuY29uZmln",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> Qwen3MoeModel, Qwen3MoeConfig

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a Qwen3MoE style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = Qwen3MoeConfig()

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a model from the Qwen3-15B-A2B&quot; style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = Qwen3MoeModel(configuration)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Accessing the model configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = model.config`,wrap:!1}}),{c(){p(o.$$.fragment)},l(s){u(o.$$.fragment,s)},m(s,w){h(o,s,w),_=!0},p:ke,i(s){_||(m(o.$$.fragment,s),_=!0)},o(s){f(o.$$.fragment,s),_=!1},d(s){g(o,s)}}}function cn(y){let o,_=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){o=c("p"),o.innerHTML=_},l(s){o=l(s,"P",{"data-svelte-h":!0}),b(o)!=="svelte-fincs2"&&(o.innerHTML=_)},m(s,w){d(s,o,w)},p:ke,d(s){s&&t(o)}}}function ln(y){let o,_=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){o=c("p"),o.innerHTML=_},l(s){o=l(s,"P",{"data-svelte-h":!0}),b(o)!=="svelte-fincs2"&&(o.innerHTML=_)},m(s,w){d(s,o,w)},p:ke,d(s){s&&t(o)}}}function pn(y){let o,_="Example:",s,w,z;return w=new en({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBRd2VuM01vZUZvckNhdXNhbExNJTBBJTBBbW9kZWwlMjAlM0QlMjBRd2VuM01vZUZvckNhdXNhbExNLmZyb21fcHJldHJhaW5lZCglMjJRd2VuJTJGUXdlbjMtTW9FLTE1Qi1BMkIlMjIpJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIyUXdlbiUyRlF3ZW4zLU1vRS0xNUItQTJCJTIyKSUwQSUwQXByb21wdCUyMCUzRCUyMCUyMkhleSUyQyUyMGFyZSUyMHlvdSUyMGNvbnNjaW91cyUzRiUyMENhbiUyMHlvdSUyMHRhbGslMjB0byUyMG1lJTNGJTIyJTBBaW5wdXRzJTIwJTNEJTIwdG9rZW5pemVyKHByb21wdCUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpJTBBJTBBJTIzJTIwR2VuZXJhdGUlMEFnZW5lcmF0ZV9pZHMlMjAlM0QlMjBtb2RlbC5nZW5lcmF0ZShpbnB1dHMuaW5wdXRfaWRzJTJDJTIwbWF4X2xlbmd0aCUzRDMwKSUwQXRva2VuaXplci5iYXRjaF9kZWNvZGUoZ2VuZXJhdGVfaWRzJTJDJTIwc2tpcF9zcGVjaWFsX3Rva2VucyUzRFRydWUlMkMlMjBjbGVhbl91cF90b2tlbml6YXRpb25fc3BhY2VzJTNERmFsc2UpJTVCMCU1RA==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, Qwen3MoeForCausalLM

<span class="hljs-meta">&gt;&gt;&gt; </span>model = Qwen3MoeForCausalLM.from_pretrained(<span class="hljs-string">&quot;Qwen/Qwen3-MoE-15B-A2B&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;Qwen/Qwen3-MoE-15B-A2B&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>prompt = <span class="hljs-string">&quot;Hey, are you conscious? Can you talk to me?&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(prompt, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Generate</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>generate_ids = model.generate(inputs.input_ids, max_length=<span class="hljs-number">30</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer.batch_decode(generate_ids, skip_special_tokens=<span class="hljs-literal">True</span>, clean_up_tokenization_spaces=<span class="hljs-literal">False</span>)[<span class="hljs-number">0</span>]
<span class="hljs-string">&quot;Hey, are you conscious? Can you talk to me?\\nI&#x27;m not conscious, but I can talk to you.&quot;</span>`,wrap:!1}}),{c(){o=c("p"),o.textContent=_,s=a(),p(w.$$.fragment)},l(v){o=l(v,"P",{"data-svelte-h":!0}),b(o)!=="svelte-11lpom8"&&(o.textContent=_),s=r(v),u(w.$$.fragment,v)},m(v,B){d(v,o,B),d(v,s,B),h(w,v,B),z=!0},p:ke,i(v){z||(m(w.$$.fragment,v),z=!0)},o(v){f(w.$$.fragment,v),z=!1},d(v){v&&(t(o),t(s)),g(w,v)}}}function un(y){let o,_=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){o=c("p"),o.innerHTML=_},l(s){o=l(s,"P",{"data-svelte-h":!0}),b(o)!=="svelte-fincs2"&&(o.innerHTML=_)},m(s,w){d(s,o,w)},p:ke,d(s){s&&t(o)}}}function hn(y){let o,_=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){o=c("p"),o.innerHTML=_},l(s){o=l(s,"P",{"data-svelte-h":!0}),b(o)!=="svelte-fincs2"&&(o.innerHTML=_)},m(s,w){d(s,o,w)},p:ke,d(s){s&&t(o)}}}function mn(y){let o,_=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){o=c("p"),o.innerHTML=_},l(s){o=l(s,"P",{"data-svelte-h":!0}),b(o)!=="svelte-fincs2"&&(o.innerHTML=_)},m(s,w){d(s,o,w)},p:ke,d(s){s&&t(o)}}}function fn(y){let o,_,s,w,z,v="<em>This model was released on 2025-04-29 and added to Hugging Face Transformers on 2025-03-31.</em>",B,Y,je,K,Se,ee,No='<a href="https://huggingface.co/papers/2505.09388" rel="nofollow">Qwen3MoE</a> refers to the mixture of experts model architecture Qwen3-235B-A22B which was released with its dense variant <a href="qwen3">Qwen3</a> (<a href="https://qwenlm.github.io/blog/qwen3/" rel="nofollow">blog post</a>).',Be,oe,De,ne,Uo="To be released with the official model launch.",Ze,te,Je,se,Ao="To be released with the official model launch.",Re,ae,Ve,k,re,uo,$e,Eo=`This is the configuration class to store the configuration of a <a href="/docs/transformers/v4.56.2/en/model_doc/qwen3_moe#transformers.Qwen3MoeModel">Qwen3MoeModel</a>. It is used to instantiate a
Qwen3MoE model according to the specified arguments, defining the model architecture. Instantiating a configuration
with the defaults will yield a similar configuration to that of <a href="https://huggingface.co/Qwen/Qwen3-15B-A2B" rel="nofollow">Qwen/Qwen3-15B-A2B</a>.`,ho,xe,Ho=`Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> and can be used to control the model outputs. Read the
documentation from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> for more information.`,mo,D,Ge,ie,Xe,T,de,fo,Ce,jo="The bare Qwen3 Moe Model outputting raw hidden-states without any specific head on top.",go,Qe,So=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,_o,ze,Bo=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,wo,P,ce,bo,Fe,Do='The <a href="/docs/transformers/v4.56.2/en/model_doc/qwen3_moe#transformers.Qwen3MoeModel">Qwen3MoeModel</a> forward method, overrides the <code>__call__</code> special method.',vo,Z,Ye,le,Ke,M,pe,yo,qe,Zo="The Qwen3 Moe Model for causal language modeling.",To,Le,Jo=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Mo,Ie,Ro=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,ko,F,ue,$o,Pe,Vo='The <a href="/docs/transformers/v4.56.2/en/model_doc/qwen3_moe#transformers.Qwen3MoeForCausalLM">Qwen3MoeForCausalLM</a> forward method, overrides the <code>__call__</code> special method.',xo,J,Co,R,eo,he,oo,A,me,Qo,O,fe,zo,Oe,Go="The <code>GenericForSequenceClassification</code> forward method, overrides the <code>__call__</code> special method.",Fo,V,no,ge,to,E,_e,qo,W,we,Lo,We,Xo="The <code>GenericForTokenClassification</code> forward method, overrides the <code>__call__</code> special method.",Io,G,so,be,ao,H,ve,Po,N,ye,Oo,Ne,Yo="The <code>GenericForQuestionAnswering</code> forward method, overrides the <code>__call__</code> special method.",Wo,X,ro,Te,io,Ee,co;return Y=new U({props:{title:"Qwen3MoE",local:"qwen3moe",headingTag:"h1"}}),K=new U({props:{title:"Overview",local:"overview",headingTag:"h2"}}),oe=new U({props:{title:"Model Details",local:"model-details",headingTag:"h3"}}),te=new U({props:{title:"Usage tips",local:"usage-tips",headingTag:"h2"}}),ae=new U({props:{title:"Qwen3MoeConfig",local:"transformers.Qwen3MoeConfig",headingTag:"h2"}}),re=new I({props:{name:"class transformers.Qwen3MoeConfig",anchor:"transformers.Qwen3MoeConfig",parameters:[{name:"vocab_size",val:" = 151936"},{name:"hidden_size",val:" = 2048"},{name:"intermediate_size",val:" = 6144"},{name:"num_hidden_layers",val:" = 24"},{name:"num_attention_heads",val:" = 32"},{name:"num_key_value_heads",val:" = 4"},{name:"hidden_act",val:" = 'silu'"},{name:"max_position_embeddings",val:" = 32768"},{name:"initializer_range",val:" = 0.02"},{name:"rms_norm_eps",val:" = 1e-06"},{name:"use_cache",val:" = True"},{name:"tie_word_embeddings",val:" = False"},{name:"rope_theta",val:" = 10000.0"},{name:"rope_scaling",val:" = None"},{name:"attention_bias",val:" = False"},{name:"use_sliding_window",val:" = False"},{name:"sliding_window",val:" = 4096"},{name:"attention_dropout",val:" = 0.0"},{name:"decoder_sparse_step",val:" = 1"},{name:"moe_intermediate_size",val:" = 768"},{name:"num_experts_per_tok",val:" = 8"},{name:"num_experts",val:" = 128"},{name:"norm_topk_prob",val:" = False"},{name:"output_router_logits",val:" = False"},{name:"router_aux_loss_coef",val:" = 0.001"},{name:"mlp_only_layers",val:" = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.Qwen3MoeConfig.vocab_size",description:`<strong>vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 151936) &#x2014;
Vocabulary size of the Qwen3MoE model. Defines the number of different tokens that can be represented by the
<code>inputs_ids</code> passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/qwen3_moe#transformers.Qwen3MoeModel">Qwen3MoeModel</a>`,name:"vocab_size"},{anchor:"transformers.Qwen3MoeConfig.hidden_size",description:`<strong>hidden_size</strong> (<code>int</code>, <em>optional</em>, defaults to 2048) &#x2014;
Dimension of the hidden representations.`,name:"hidden_size"},{anchor:"transformers.Qwen3MoeConfig.intermediate_size",description:`<strong>intermediate_size</strong> (<code>int</code>, <em>optional</em>, defaults to 6144) &#x2014;
Dimension of the MLP representations.`,name:"intermediate_size"},{anchor:"transformers.Qwen3MoeConfig.num_hidden_layers",description:`<strong>num_hidden_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 24) &#x2014;
Number of hidden layers in the Transformer encoder.`,name:"num_hidden_layers"},{anchor:"transformers.Qwen3MoeConfig.num_attention_heads",description:`<strong>num_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 32) &#x2014;
Number of attention heads for each attention layer in the Transformer encoder.`,name:"num_attention_heads"},{anchor:"transformers.Qwen3MoeConfig.num_key_value_heads",description:`<strong>num_key_value_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 4) &#x2014;
This is the number of key_value heads that should be used to implement Grouped Query Attention. If
<code>num_key_value_heads=num_attention_heads</code>, the model will use Multi Head Attention (MHA), if
<code>num_key_value_heads=1</code> the model will use Multi Query Attention (MQA) otherwise GQA is used. When
converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
by meanpooling all the original heads within that group. For more details, check out <a href="https://huggingface.co/papers/2305.13245" rel="nofollow">this
paper</a>. If it is not specified, will default to <code>32</code>.`,name:"num_key_value_heads"},{anchor:"transformers.Qwen3MoeConfig.hidden_act",description:`<strong>hidden_act</strong> (<code>str</code> or <code>function</code>, <em>optional</em>, defaults to <code>&quot;silu&quot;</code>) &#x2014;
The non-linear activation function (function or string) in the decoder.`,name:"hidden_act"},{anchor:"transformers.Qwen3MoeConfig.max_position_embeddings",description:`<strong>max_position_embeddings</strong> (<code>int</code>, <em>optional</em>, defaults to 32768) &#x2014;
The maximum sequence length that this model might ever be used with.`,name:"max_position_embeddings"},{anchor:"transformers.Qwen3MoeConfig.initializer_range",description:`<strong>initializer_range</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.`,name:"initializer_range"},{anchor:"transformers.Qwen3MoeConfig.rms_norm_eps",description:`<strong>rms_norm_eps</strong> (<code>float</code>, <em>optional</em>, defaults to 1e-06) &#x2014;
The epsilon used by the rms normalization layers.`,name:"rms_norm_eps"},{anchor:"transformers.Qwen3MoeConfig.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not the model should return the last key/values attentions (not used by all models). Only
relevant if <code>config.is_decoder=True</code>.`,name:"use_cache"},{anchor:"transformers.Qwen3MoeConfig.tie_word_embeddings",description:`<strong>tie_word_embeddings</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether the model&#x2019;s input and output word embeddings should be tied.`,name:"tie_word_embeddings"},{anchor:"transformers.Qwen3MoeConfig.rope_theta",description:`<strong>rope_theta</strong> (<code>float</code>, <em>optional</em>, defaults to 10000.0) &#x2014;
The base period of the RoPE embeddings.`,name:"rope_theta"},{anchor:"transformers.Qwen3MoeConfig.rope_scaling",description:`<strong>rope_scaling</strong> (<code>Dict</code>, <em>optional</em>) &#x2014;
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
Only used with &#x2018;llama3&#x2019;. Scaling factor applied to high frequency components of the RoPE`,name:"rope_scaling"},{anchor:"transformers.Qwen3MoeConfig.attention_bias",description:`<strong>attention_bias</strong> (<code>bool</code>, defaults to <code>False</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether to use a bias in the query, key, value and output projection layers during self-attention.`,name:"attention_bias"},{anchor:"transformers.Qwen3MoeConfig.use_sliding_window",description:`<strong>use_sliding_window</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether to use sliding window attention.`,name:"use_sliding_window"},{anchor:"transformers.Qwen3MoeConfig.sliding_window",description:`<strong>sliding_window</strong> (<code>int</code>, <em>optional</em>, defaults to 4096) &#x2014;
Sliding window attention (SWA) window size. If not specified, will default to <code>4096</code>.`,name:"sliding_window"},{anchor:"transformers.Qwen3MoeConfig.attention_dropout",description:`<strong>attention_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The dropout ratio for the attention probabilities.`,name:"attention_dropout"},{anchor:"transformers.Qwen3MoeConfig.decoder_sparse_step",description:`<strong>decoder_sparse_step</strong> (<code>int</code>, <em>optional</em>, defaults to 1) &#x2014;
The frequency of the MoE layer.`,name:"decoder_sparse_step"},{anchor:"transformers.Qwen3MoeConfig.moe_intermediate_size",description:`<strong>moe_intermediate_size</strong> (<code>int</code>, <em>optional</em>, defaults to 768) &#x2014;
Intermediate size of the routed expert.`,name:"moe_intermediate_size"},{anchor:"transformers.Qwen3MoeConfig.num_experts_per_tok",description:`<strong>num_experts_per_tok</strong> (<code>int</code>, <em>optional</em>, defaults to 8) &#x2014;
Number of selected experts.`,name:"num_experts_per_tok"},{anchor:"transformers.Qwen3MoeConfig.num_experts",description:`<strong>num_experts</strong> (<code>int</code>, <em>optional</em>, defaults to 128) &#x2014;
Number of routed experts.`,name:"num_experts"},{anchor:"transformers.Qwen3MoeConfig.norm_topk_prob",description:`<strong>norm_topk_prob</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether to normalize the topk probabilities.`,name:"norm_topk_prob"},{anchor:"transformers.Qwen3MoeConfig.output_router_logits",description:`<strong>output_router_logits</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not the router logits should be returned by the model. Enabling this will also
allow the model to output the auxiliary loss, including load balancing loss and router z-loss.`,name:"output_router_logits"},{anchor:"transformers.Qwen3MoeConfig.router_aux_loss_coef",description:`<strong>router_aux_loss_coef</strong> (<code>float</code>, <em>optional</em>, defaults to 0.001) &#x2014;
The aux loss factor for the total loss.`,name:"router_aux_loss_coef"},{anchor:"transformers.Qwen3MoeConfig.mlp_only_layers",description:`<strong>mlp_only_layers</strong> (<code>list[int]</code>, <em>optional</em>, defaults to <code>[]</code>) &#x2014;
Indicate which layers use Qwen3MoeMLP rather than Qwen3MoeSparseMoeBlock
The list contains layer index, from 0 to num_layers-1 if we have num_layers layers
If <code>mlp_only_layers</code> is empty, <code>decoder_sparse_step</code> is used to determine the sparsity.`,name:"mlp_only_layers"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/qwen3_moe/configuration_qwen3_moe.py#L25"}}),D=new Ko({props:{anchor:"transformers.Qwen3MoeConfig.example",$$slots:{default:[dn]},$$scope:{ctx:y}}}),ie=new U({props:{title:"Qwen3MoeModel",local:"transformers.Qwen3MoeModel",headingTag:"h2"}}),de=new I({props:{name:"class transformers.Qwen3MoeModel",anchor:"transformers.Qwen3MoeModel",parameters:[{name:"config",val:": Qwen3MoeConfig"}],parametersDescription:[{anchor:"transformers.Qwen3MoeModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/qwen3_moe#transformers.Qwen3MoeConfig">Qwen3MoeConfig</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py#L424"}}),ce=new I({props:{name:"forward",anchor:"transformers.Qwen3MoeModel.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.Qwen3MoeModel.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.Qwen3MoeModel.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.Qwen3MoeModel.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.Qwen3MoeModel.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.Qwen3MoeModel.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.Qwen3MoeModel.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.Qwen3MoeModel.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py#L441",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <code>transformers.modeling_outputs.MoeModelOutputWithPast</code> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/qwen3_moe#transformers.Qwen3MoeConfig"
>Qwen3MoeConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>last_hidden_state</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>) — Sequence of hidden-states at the output of the last layer of the model.</p>
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
<li>
<p><strong>router_logits</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_router_probs=True</code> and <code>config.add_router_probs=True</code> is passed or when <code>config.output_router_probs=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, sequence_length, num_experts)</code>.</p>
<p>Raw router logtis (post-softmax) that are computed by MoE routers, these terms are used to compute the auxiliary
loss for Mixture of Experts models.</p>
</li>
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>transformers.modeling_outputs.MoeModelOutputWithPast</code> or <code>tuple(torch.FloatTensor)</code></p>
`}}),Z=new He({props:{$$slots:{default:[cn]},$$scope:{ctx:y}}}),le=new U({props:{title:"Qwen3MoeForCausalLM",local:"transformers.Qwen3MoeForCausalLM",headingTag:"h2"}}),pe=new I({props:{name:"class transformers.Qwen3MoeForCausalLM",anchor:"transformers.Qwen3MoeForCausalLM",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.Qwen3MoeForCausalLM.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/qwen3_moe#transformers.Qwen3MoeForCausalLM">Qwen3MoeForCausalLM</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py#L589"}}),ue=new I({props:{name:"forward",anchor:"transformers.Qwen3MoeForCausalLM.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_router_logits",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"},{name:"logits_to_keep",val:": typing.Union[int, torch.Tensor] = 0"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.Qwen3MoeForCausalLM.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.Qwen3MoeForCausalLM.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.Qwen3MoeForCausalLM.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.Qwen3MoeForCausalLM.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.Qwen3MoeForCausalLM.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.Qwen3MoeForCausalLM.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.Qwen3MoeForCausalLM.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.Qwen3MoeForCausalLM.forward.output_router_logits",description:`<strong>output_router_logits</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the logits of all the routers. They are useful for computing the router loss, and
should not be returned during inference.`,name:"output_router_logits"},{anchor:"transformers.Qwen3MoeForCausalLM.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"},{anchor:"transformers.Qwen3MoeForCausalLM.forward.logits_to_keep",description:`<strong>logits_to_keep</strong> (<code>Union[int, torch.Tensor]</code>, defaults to <code>0</code>) &#x2014;
If an <code>int</code>, compute logits for the last <code>logits_to_keep</code> tokens. If <code>0</code>, calculate logits for all
<code>input_ids</code> (special case). Only last token logits are needed for generation, and calculating them only for that
token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
If a <code>torch.Tensor</code>, must be 1D corresponding to the indices to keep in the sequence length dimension.
This is useful when using packed tensor format (single dimension for batch and sequence length).`,name:"logits_to_keep"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py#L606",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <code>transformers.modeling_outputs.MoeCausalLMOutputWithPast</code> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/qwen3_moe#transformers.Qwen3MoeConfig"
>Qwen3MoeConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>loss</strong> (<code>torch.FloatTensor</code> of shape <code>(1,)</code>, <em>optional</em>, returned when <code>labels</code> is provided) — Language modeling loss (for next-token prediction).</p>
</li>
<li>
<p><strong>logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, config.vocab_size)</code>) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).</p>
</li>
<li>
<p><strong>aux_loss</strong> (<code>torch.FloatTensor</code>, <em>optional</em>, returned when <code>labels</code> is provided) — aux_loss for the sparse modules.</p>
</li>
<li>
<p><strong>router_logits</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_router_probs=True</code> and <code>config.add_router_probs=True</code> is passed or when <code>config.output_router_probs=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, sequence_length, num_experts)</code>.</p>
<p>Raw router logtis (post-softmax) that are computed by MoE routers, these terms are used to compute the auxiliary
loss for Mixture of Experts models.</p>
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


<p><code>transformers.modeling_outputs.MoeCausalLMOutputWithPast</code> or <code>tuple(torch.FloatTensor)</code></p>
`}}),J=new He({props:{$$slots:{default:[ln]},$$scope:{ctx:y}}}),R=new Ko({props:{anchor:"transformers.Qwen3MoeForCausalLM.forward.example",$$slots:{default:[pn]},$$scope:{ctx:y}}}),he=new U({props:{title:"Qwen3MoeForSequenceClassification",local:"transformers.Qwen3MoeForSequenceClassification",headingTag:"h2"}}),me=new I({props:{name:"class transformers.Qwen3MoeForSequenceClassification",anchor:"transformers.Qwen3MoeForSequenceClassification",parameters:[{name:"config",val:""}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py#L693"}}),fe=new I({props:{name:"forward",anchor:"transformers.Qwen3MoeForSequenceClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.Qwen3MoeForSequenceClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.Qwen3MoeForSequenceClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.Qwen3MoeForSequenceClassification.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.Qwen3MoeForSequenceClassification.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.Qwen3MoeForSequenceClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.Qwen3MoeForSequenceClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.Qwen3MoeForSequenceClassification.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
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
`}}),V=new He({props:{$$slots:{default:[un]},$$scope:{ctx:y}}}),ge=new U({props:{title:"Qwen3MoeForTokenClassification",local:"transformers.Qwen3MoeForTokenClassification",headingTag:"h2"}}),_e=new I({props:{name:"class transformers.Qwen3MoeForTokenClassification",anchor:"transformers.Qwen3MoeForTokenClassification",parameters:[{name:"config",val:""}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py#L697"}}),we=new I({props:{name:"forward",anchor:"transformers.Qwen3MoeForTokenClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.Qwen3MoeForTokenClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.Qwen3MoeForTokenClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.Qwen3MoeForTokenClassification.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.Qwen3MoeForTokenClassification.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.Qwen3MoeForTokenClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.Qwen3MoeForTokenClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.Qwen3MoeForTokenClassification.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
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
`}}),G=new He({props:{$$slots:{default:[hn]},$$scope:{ctx:y}}}),be=new U({props:{title:"Qwen3MoeForQuestionAnswering",local:"transformers.Qwen3MoeForQuestionAnswering",headingTag:"h2"}}),ve=new I({props:{name:"class transformers.Qwen3MoeForQuestionAnswering",anchor:"transformers.Qwen3MoeForQuestionAnswering",parameters:[{name:"config",val:""}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py#L701"}}),ye=new I({props:{name:"forward",anchor:"transformers.Qwen3MoeForQuestionAnswering.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"start_positions",val:": typing.Optional[torch.LongTensor] = None"},{name:"end_positions",val:": typing.Optional[torch.LongTensor] = None"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.Qwen3MoeForQuestionAnswering.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.Qwen3MoeForQuestionAnswering.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.Qwen3MoeForQuestionAnswering.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.Qwen3MoeForQuestionAnswering.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.Qwen3MoeForQuestionAnswering.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.Qwen3MoeForQuestionAnswering.forward.start_positions",description:`<strong>start_positions</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for position (index) of the start of the labelled span for computing the token classification loss.
Positions are clamped to the length of the sequence (<code>sequence_length</code>). Position outside of the sequence
are not taken into account for computing the loss.`,name:"start_positions"},{anchor:"transformers.Qwen3MoeForQuestionAnswering.forward.end_positions",description:`<strong>end_positions</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
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
`}}),X=new He({props:{$$slots:{default:[mn]},$$scope:{ctx:y}}}),Te=new rn({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/qwen3_moe.md"}}),{c(){o=c("meta"),_=a(),s=c("p"),w=a(),z=c("p"),z.innerHTML=v,B=a(),p(Y.$$.fragment),je=a(),p(K.$$.fragment),Se=a(),ee=c("p"),ee.innerHTML=No,Be=a(),p(oe.$$.fragment),De=a(),ne=c("p"),ne.textContent=Uo,Ze=a(),p(te.$$.fragment),Je=a(),se=c("p"),se.textContent=Ao,Re=a(),p(ae.$$.fragment),Ve=a(),k=c("div"),p(re.$$.fragment),uo=a(),$e=c("p"),$e.innerHTML=Eo,ho=a(),xe=c("p"),xe.innerHTML=Ho,mo=a(),p(D.$$.fragment),Ge=a(),p(ie.$$.fragment),Xe=a(),T=c("div"),p(de.$$.fragment),fo=a(),Ce=c("p"),Ce.textContent=jo,go=a(),Qe=c("p"),Qe.innerHTML=So,_o=a(),ze=c("p"),ze.innerHTML=Bo,wo=a(),P=c("div"),p(ce.$$.fragment),bo=a(),Fe=c("p"),Fe.innerHTML=Do,vo=a(),p(Z.$$.fragment),Ye=a(),p(le.$$.fragment),Ke=a(),M=c("div"),p(pe.$$.fragment),yo=a(),qe=c("p"),qe.textContent=Zo,To=a(),Le=c("p"),Le.innerHTML=Jo,Mo=a(),Ie=c("p"),Ie.innerHTML=Ro,ko=a(),F=c("div"),p(ue.$$.fragment),$o=a(),Pe=c("p"),Pe.innerHTML=Vo,xo=a(),p(J.$$.fragment),Co=a(),p(R.$$.fragment),eo=a(),p(he.$$.fragment),oo=a(),A=c("div"),p(me.$$.fragment),Qo=a(),O=c("div"),p(fe.$$.fragment),zo=a(),Oe=c("p"),Oe.innerHTML=Go,Fo=a(),p(V.$$.fragment),no=a(),p(ge.$$.fragment),to=a(),E=c("div"),p(_e.$$.fragment),qo=a(),W=c("div"),p(we.$$.fragment),Lo=a(),We=c("p"),We.innerHTML=Xo,Io=a(),p(G.$$.fragment),so=a(),p(be.$$.fragment),ao=a(),H=c("div"),p(ve.$$.fragment),Po=a(),N=c("div"),p(ye.$$.fragment),Oo=a(),Ne=c("p"),Ne.innerHTML=Yo,Wo=a(),p(X.$$.fragment),ro=a(),p(Te.$$.fragment),io=a(),Ee=c("p"),this.h()},l(e){const n=an("svelte-u9bgzb",document.head);o=l(n,"META",{name:!0,content:!0}),n.forEach(t),_=r(e),s=l(e,"P",{}),C(s).forEach(t),w=r(e),z=l(e,"P",{"data-svelte-h":!0}),b(z)!=="svelte-hz2l3k"&&(z.innerHTML=v),B=r(e),u(Y.$$.fragment,e),je=r(e),u(K.$$.fragment,e),Se=r(e),ee=l(e,"P",{"data-svelte-h":!0}),b(ee)!=="svelte-1tke77b"&&(ee.innerHTML=No),Be=r(e),u(oe.$$.fragment,e),De=r(e),ne=l(e,"P",{"data-svelte-h":!0}),b(ne)!=="svelte-u40g91"&&(ne.textContent=Uo),Ze=r(e),u(te.$$.fragment,e),Je=r(e),se=l(e,"P",{"data-svelte-h":!0}),b(se)!=="svelte-u40g91"&&(se.textContent=Ao),Re=r(e),u(ae.$$.fragment,e),Ve=r(e),k=l(e,"DIV",{class:!0});var q=C(k);u(re.$$.fragment,q),uo=r(q),$e=l(q,"P",{"data-svelte-h":!0}),b($e)!=="svelte-9o116g"&&($e.innerHTML=Eo),ho=r(q),xe=l(q,"P",{"data-svelte-h":!0}),b(xe)!=="svelte-1ek1ss9"&&(xe.innerHTML=Ho),mo=r(q),u(D.$$.fragment,q),q.forEach(t),Ge=r(e),u(ie.$$.fragment,e),Xe=r(e),T=l(e,"DIV",{class:!0});var $=C(T);u(de.$$.fragment,$),fo=r($),Ce=l($,"P",{"data-svelte-h":!0}),b(Ce)!=="svelte-d1u953"&&(Ce.textContent=jo),go=r($),Qe=l($,"P",{"data-svelte-h":!0}),b(Qe)!=="svelte-q52n56"&&(Qe.innerHTML=So),_o=r($),ze=l($,"P",{"data-svelte-h":!0}),b(ze)!=="svelte-hswkmf"&&(ze.innerHTML=Bo),wo=r($),P=l($,"DIV",{class:!0});var j=C(P);u(ce.$$.fragment,j),bo=r(j),Fe=l(j,"P",{"data-svelte-h":!0}),b(Fe)!=="svelte-1ohiruj"&&(Fe.innerHTML=Do),vo=r(j),u(Z.$$.fragment,j),j.forEach(t),$.forEach(t),Ye=r(e),u(le.$$.fragment,e),Ke=r(e),M=l(e,"DIV",{class:!0});var x=C(M);u(pe.$$.fragment,x),yo=r(x),qe=l(x,"P",{"data-svelte-h":!0}),b(qe)!=="svelte-1aw34y0"&&(qe.textContent=Zo),To=r(x),Le=l(x,"P",{"data-svelte-h":!0}),b(Le)!=="svelte-q52n56"&&(Le.innerHTML=Jo),Mo=r(x),Ie=l(x,"P",{"data-svelte-h":!0}),b(Ie)!=="svelte-hswkmf"&&(Ie.innerHTML=Ro),ko=r(x),F=l(x,"DIV",{class:!0});var L=C(F);u(ue.$$.fragment,L),$o=r(L),Pe=l(L,"P",{"data-svelte-h":!0}),b(Pe)!=="svelte-wae8dv"&&(Pe.innerHTML=Vo),xo=r(L),u(J.$$.fragment,L),Co=r(L),u(R.$$.fragment,L),L.forEach(t),x.forEach(t),eo=r(e),u(he.$$.fragment,e),oo=r(e),A=l(e,"DIV",{class:!0});var Me=C(A);u(me.$$.fragment,Me),Qo=r(Me),O=l(Me,"DIV",{class:!0});var S=C(O);u(fe.$$.fragment,S),zo=r(S),Oe=l(S,"P",{"data-svelte-h":!0}),b(Oe)!=="svelte-1sal4ui"&&(Oe.innerHTML=Go),Fo=r(S),u(V.$$.fragment,S),S.forEach(t),Me.forEach(t),no=r(e),u(ge.$$.fragment,e),to=r(e),E=l(e,"DIV",{class:!0});var lo=C(E);u(_e.$$.fragment,lo),qo=r(lo),W=l(lo,"DIV",{class:!0});var Ue=C(W);u(we.$$.fragment,Ue),Lo=r(Ue),We=l(Ue,"P",{"data-svelte-h":!0}),b(We)!=="svelte-1py4aay"&&(We.innerHTML=Xo),Io=r(Ue),u(G.$$.fragment,Ue),Ue.forEach(t),lo.forEach(t),so=r(e),u(be.$$.fragment,e),ao=r(e),H=l(e,"DIV",{class:!0});var po=C(H);u(ve.$$.fragment,po),Po=r(po),N=l(po,"DIV",{class:!0});var Ae=C(N);u(ye.$$.fragment,Ae),Oo=r(Ae),Ne=l(Ae,"P",{"data-svelte-h":!0}),b(Ne)!=="svelte-dyrov9"&&(Ne.innerHTML=Yo),Wo=r(Ae),u(X.$$.fragment,Ae),Ae.forEach(t),po.forEach(t),ro=r(e),u(Te.$$.fragment,e),io=r(e),Ee=l(e,"P",{}),C(Ee).forEach(t),this.h()},h(){Q(o,"name","hf:doc:metadata"),Q(o,"content",gn),Q(k,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),Q(P,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),Q(T,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),Q(F,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),Q(M,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),Q(O,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),Q(A,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),Q(W,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),Q(E,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),Q(N,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),Q(H,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,n){i(document.head,o),d(e,_,n),d(e,s,n),d(e,w,n),d(e,z,n),d(e,B,n),h(Y,e,n),d(e,je,n),h(K,e,n),d(e,Se,n),d(e,ee,n),d(e,Be,n),h(oe,e,n),d(e,De,n),d(e,ne,n),d(e,Ze,n),h(te,e,n),d(e,Je,n),d(e,se,n),d(e,Re,n),h(ae,e,n),d(e,Ve,n),d(e,k,n),h(re,k,null),i(k,uo),i(k,$e),i(k,ho),i(k,xe),i(k,mo),h(D,k,null),d(e,Ge,n),h(ie,e,n),d(e,Xe,n),d(e,T,n),h(de,T,null),i(T,fo),i(T,Ce),i(T,go),i(T,Qe),i(T,_o),i(T,ze),i(T,wo),i(T,P),h(ce,P,null),i(P,bo),i(P,Fe),i(P,vo),h(Z,P,null),d(e,Ye,n),h(le,e,n),d(e,Ke,n),d(e,M,n),h(pe,M,null),i(M,yo),i(M,qe),i(M,To),i(M,Le),i(M,Mo),i(M,Ie),i(M,ko),i(M,F),h(ue,F,null),i(F,$o),i(F,Pe),i(F,xo),h(J,F,null),i(F,Co),h(R,F,null),d(e,eo,n),h(he,e,n),d(e,oo,n),d(e,A,n),h(me,A,null),i(A,Qo),i(A,O),h(fe,O,null),i(O,zo),i(O,Oe),i(O,Fo),h(V,O,null),d(e,no,n),h(ge,e,n),d(e,to,n),d(e,E,n),h(_e,E,null),i(E,qo),i(E,W),h(we,W,null),i(W,Lo),i(W,We),i(W,Io),h(G,W,null),d(e,so,n),h(be,e,n),d(e,ao,n),d(e,H,n),h(ve,H,null),i(H,Po),i(H,N),h(ye,N,null),i(N,Oo),i(N,Ne),i(N,Wo),h(X,N,null),d(e,ro,n),h(Te,e,n),d(e,io,n),d(e,Ee,n),co=!0},p(e,[n]){const q={};n&2&&(q.$$scope={dirty:n,ctx:e}),D.$set(q);const $={};n&2&&($.$$scope={dirty:n,ctx:e}),Z.$set($);const j={};n&2&&(j.$$scope={dirty:n,ctx:e}),J.$set(j);const x={};n&2&&(x.$$scope={dirty:n,ctx:e}),R.$set(x);const L={};n&2&&(L.$$scope={dirty:n,ctx:e}),V.$set(L);const Me={};n&2&&(Me.$$scope={dirty:n,ctx:e}),G.$set(Me);const S={};n&2&&(S.$$scope={dirty:n,ctx:e}),X.$set(S)},i(e){co||(m(Y.$$.fragment,e),m(K.$$.fragment,e),m(oe.$$.fragment,e),m(te.$$.fragment,e),m(ae.$$.fragment,e),m(re.$$.fragment,e),m(D.$$.fragment,e),m(ie.$$.fragment,e),m(de.$$.fragment,e),m(ce.$$.fragment,e),m(Z.$$.fragment,e),m(le.$$.fragment,e),m(pe.$$.fragment,e),m(ue.$$.fragment,e),m(J.$$.fragment,e),m(R.$$.fragment,e),m(he.$$.fragment,e),m(me.$$.fragment,e),m(fe.$$.fragment,e),m(V.$$.fragment,e),m(ge.$$.fragment,e),m(_e.$$.fragment,e),m(we.$$.fragment,e),m(G.$$.fragment,e),m(be.$$.fragment,e),m(ve.$$.fragment,e),m(ye.$$.fragment,e),m(X.$$.fragment,e),m(Te.$$.fragment,e),co=!0)},o(e){f(Y.$$.fragment,e),f(K.$$.fragment,e),f(oe.$$.fragment,e),f(te.$$.fragment,e),f(ae.$$.fragment,e),f(re.$$.fragment,e),f(D.$$.fragment,e),f(ie.$$.fragment,e),f(de.$$.fragment,e),f(ce.$$.fragment,e),f(Z.$$.fragment,e),f(le.$$.fragment,e),f(pe.$$.fragment,e),f(ue.$$.fragment,e),f(J.$$.fragment,e),f(R.$$.fragment,e),f(he.$$.fragment,e),f(me.$$.fragment,e),f(fe.$$.fragment,e),f(V.$$.fragment,e),f(ge.$$.fragment,e),f(_e.$$.fragment,e),f(we.$$.fragment,e),f(G.$$.fragment,e),f(be.$$.fragment,e),f(ve.$$.fragment,e),f(ye.$$.fragment,e),f(X.$$.fragment,e),f(Te.$$.fragment,e),co=!1},d(e){e&&(t(_),t(s),t(w),t(z),t(B),t(je),t(Se),t(ee),t(Be),t(De),t(ne),t(Ze),t(Je),t(se),t(Re),t(Ve),t(k),t(Ge),t(Xe),t(T),t(Ye),t(Ke),t(M),t(eo),t(oo),t(A),t(no),t(to),t(E),t(so),t(ao),t(H),t(ro),t(io),t(Ee)),t(o),g(Y,e),g(K,e),g(oe,e),g(te,e),g(ae,e),g(re),g(D),g(ie,e),g(de),g(ce),g(Z),g(le,e),g(pe),g(ue),g(J),g(R),g(he,e),g(me),g(fe),g(V),g(ge,e),g(_e),g(we),g(G),g(be,e),g(ve),g(ye),g(X),g(Te,e)}}}const gn='{"title":"Qwen3MoE","local":"qwen3moe","sections":[{"title":"Overview","local":"overview","sections":[{"title":"Model Details","local":"model-details","sections":[],"depth":3}],"depth":2},{"title":"Usage tips","local":"usage-tips","sections":[],"depth":2},{"title":"Qwen3MoeConfig","local":"transformers.Qwen3MoeConfig","sections":[],"depth":2},{"title":"Qwen3MoeModel","local":"transformers.Qwen3MoeModel","sections":[],"depth":2},{"title":"Qwen3MoeForCausalLM","local":"transformers.Qwen3MoeForCausalLM","sections":[],"depth":2},{"title":"Qwen3MoeForSequenceClassification","local":"transformers.Qwen3MoeForSequenceClassification","sections":[],"depth":2},{"title":"Qwen3MoeForTokenClassification","local":"transformers.Qwen3MoeForTokenClassification","sections":[],"depth":2},{"title":"Qwen3MoeForQuestionAnswering","local":"transformers.Qwen3MoeForQuestionAnswering","sections":[],"depth":2}],"depth":1}';function _n(y){return nn(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class $n extends tn{constructor(o){super(),sn(this,o,_n,fn,on,{})}}export{$n as component};
