import{s as no,o as oo,n as $e}from"../chunks/scheduler.18a86fab.js";import{S as to,i as so,g as c,s as a,r as p,A as ao,h as l,f as t,c as r,j as M,x as b,u as h,k as Q,y as i,a as d,v as u,d as m,t as f,w as g}from"../chunks/index.98837b22.js";import{T as Ue}from"../chunks/Tip.77304350.js";import{D as I}from"../chunks/Docstring.a1ef7999.js";import{C as eo}from"../chunks/CodeBlock.8d0c2e8a.js";import{E as Kn}from"../chunks/ExampleCodeBlock.8c3ee1f9.js";import{H as A,E as ro}from"../chunks/getInferenceSnippets.06c2775f.js";function io(y){let n,_;return n=new eo({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMFF3ZW4zTW9kZWwlMkMlMjBRd2VuM0NvbmZpZyUwQSUwQSUyMyUyMEluaXRpYWxpemluZyUyMGElMjBRd2VuMyUyMHN0eWxlJTIwY29uZmlndXJhdGlvbiUwQWNvbmZpZ3VyYXRpb24lMjAlM0QlMjBRd2VuM0NvbmZpZygpJTBBJTBBJTIzJTIwSW5pdGlhbGl6aW5nJTIwYSUyMG1vZGVsJTIwZnJvbSUyMHRoZSUyMFF3ZW4zLThCJTIwc3R5bGUlMjBjb25maWd1cmF0aW9uJTBBbW9kZWwlMjAlM0QlMjBRd2VuM01vZGVsKGNvbmZpZ3VyYXRpb24pJTBBJTBBJTIzJTIwQWNjZXNzaW5nJTIwdGhlJTIwbW9kZWwlMjBjb25maWd1cmF0aW9uJTBBY29uZmlndXJhdGlvbiUyMCUzRCUyMG1vZGVsLmNvbmZpZw==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> Qwen3Model, Qwen3Config

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a Qwen3 style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = Qwen3Config()

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a model from the Qwen3-8B style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = Qwen3Model(configuration)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Accessing the model configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = model.config`,wrap:!1}}),{c(){p(n.$$.fragment)},l(s){h(n.$$.fragment,s)},m(s,w){u(n,s,w),_=!0},p:$e,i(s){_||(m(n.$$.fragment,s),_=!0)},o(s){f(n.$$.fragment,s),_=!1},d(s){g(n,s)}}}function co(y){let n,_=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){n=c("p"),n.innerHTML=_},l(s){n=l(s,"P",{"data-svelte-h":!0}),b(n)!=="svelte-fincs2"&&(n.innerHTML=_)},m(s,w){d(s,n,w)},p:$e,d(s){s&&t(n)}}}function lo(y){let n,_=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){n=c("p"),n.innerHTML=_},l(s){n=l(s,"P",{"data-svelte-h":!0}),b(n)!=="svelte-fincs2"&&(n.innerHTML=_)},m(s,w){d(s,n,w)},p:$e,d(s){s&&t(n)}}}function po(y){let n,_="Example:",s,w,z;return w=new eo({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBRd2VuM0ZvckNhdXNhbExNJTBBJTBBbW9kZWwlMjAlM0QlMjBRd2VuM0ZvckNhdXNhbExNLmZyb21fcHJldHJhaW5lZCglMjJRd2VuJTJGUXdlbjMtOEIlMjIpJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIyUXdlbiUyRlF3ZW4zLThCJTIyKSUwQSUwQXByb21wdCUyMCUzRCUyMCUyMkhleSUyQyUyMGFyZSUyMHlvdSUyMGNvbnNjaW91cyUzRiUyMENhbiUyMHlvdSUyMHRhbGslMjB0byUyMG1lJTNGJTIyJTBBaW5wdXRzJTIwJTNEJTIwdG9rZW5pemVyKHByb21wdCUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpJTBBJTBBJTIzJTIwR2VuZXJhdGUlMEFnZW5lcmF0ZV9pZHMlMjAlM0QlMjBtb2RlbC5nZW5lcmF0ZShpbnB1dHMuaW5wdXRfaWRzJTJDJTIwbWF4X2xlbmd0aCUzRDMwKSUwQXRva2VuaXplci5iYXRjaF9kZWNvZGUoZ2VuZXJhdGVfaWRzJTJDJTIwc2tpcF9zcGVjaWFsX3Rva2VucyUzRFRydWUlMkMlMjBjbGVhbl91cF90b2tlbml6YXRpb25fc3BhY2VzJTNERmFsc2UpJTVCMCU1RA==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, Qwen3ForCausalLM

<span class="hljs-meta">&gt;&gt;&gt; </span>model = Qwen3ForCausalLM.from_pretrained(<span class="hljs-string">&quot;Qwen/Qwen3-8B&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;Qwen/Qwen3-8B&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>prompt = <span class="hljs-string">&quot;Hey, are you conscious? Can you talk to me?&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(prompt, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Generate</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>generate_ids = model.generate(inputs.input_ids, max_length=<span class="hljs-number">30</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer.batch_decode(generate_ids, skip_special_tokens=<span class="hljs-literal">True</span>, clean_up_tokenization_spaces=<span class="hljs-literal">False</span>)[<span class="hljs-number">0</span>]
<span class="hljs-string">&quot;Hey, are you conscious? Can you talk to me?\\nI&#x27;m not conscious, but I can talk to you.&quot;</span>`,wrap:!1}}),{c(){n=c("p"),n.textContent=_,s=a(),p(w.$$.fragment)},l(v){n=l(v,"P",{"data-svelte-h":!0}),b(n)!=="svelte-11lpom8"&&(n.textContent=_),s=r(v),h(w.$$.fragment,v)},m(v,J){d(v,n,J),d(v,s,J),u(w,v,J),z=!0},p:$e,i(v){z||(m(w.$$.fragment,v),z=!0)},o(v){f(w.$$.fragment,v),z=!1},d(v){v&&(t(n),t(s)),g(w,v)}}}function ho(y){let n,_=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){n=c("p"),n.innerHTML=_},l(s){n=l(s,"P",{"data-svelte-h":!0}),b(n)!=="svelte-fincs2"&&(n.innerHTML=_)},m(s,w){d(s,n,w)},p:$e,d(s){s&&t(n)}}}function uo(y){let n,_=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){n=c("p"),n.innerHTML=_},l(s){n=l(s,"P",{"data-svelte-h":!0}),b(n)!=="svelte-fincs2"&&(n.innerHTML=_)},m(s,w){d(s,n,w)},p:$e,d(s){s&&t(n)}}}function mo(y){let n,_=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){n=c("p"),n.innerHTML=_},l(s){n=l(s,"P",{"data-svelte-h":!0}),b(n)!=="svelte-fincs2"&&(n.innerHTML=_)},m(s,w){d(s,n,w)},p:$e,d(s){s&&t(n)}}}function fo(y){let n,_,s,w,z,v="<em>This model was released on 2025-04-29 and added to Hugging Face Transformers on 2025-03-31.</em>",J,Y,je,K,Se,ee,Nn='<a href="https://huggingface.co/papers/2505.09388" rel="nofollow">Qwen3</a> refers to the dense model architecture Qwen3-32B which was released with its mixture of experts variant <a href="qwen3_moe">Qwen3MoE</a> (<a href="https://qwenlm.github.io/blog/qwen3/" rel="nofollow">blog post</a>).',Je,ne,De,oe,An="To be released with the official model launch.",Ze,te,Re,se,Bn="To be released with the official model launch.",Ee,ae,Ve,$,re,hn,xe,Hn=`This is the configuration class to store the configuration of a <a href="/docs/transformers/v4.56.2/en/model_doc/qwen3#transformers.Qwen3Model">Qwen3Model</a>. It is used to instantiate a
Qwen3 model according to the specified arguments, defining the model architecture. Instantiating a configuration
with the defaults will yield a similar configuration to that of
Qwen3-8B <a href="https://huggingface.co/Qwen/Qwen3-8B" rel="nofollow">Qwen/Qwen3-8B</a>.`,un,Ce,Un=`Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> and can be used to control the model outputs. Read the
documentation from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> for more information.`,mn,D,Ge,ie,Xe,T,de,fn,Me,jn="The bare Qwen3 Model outputting raw hidden-states without any specific head on top.",gn,Qe,Sn=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,_n,ze,Jn=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,wn,P,ce,bn,Fe,Dn='The <a href="/docs/transformers/v4.56.2/en/model_doc/qwen3#transformers.Qwen3Model">Qwen3Model</a> forward method, overrides the <code>__call__</code> special method.',vn,Z,Ye,le,Ke,k,pe,yn,qe,Zn="The Qwen3 Model for causal language modeling.",Tn,Le,Rn=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,kn,Ie,En=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,$n,F,he,xn,Pe,Vn='The <a href="/docs/transformers/v4.56.2/en/model_doc/qwen3#transformers.Qwen3ForCausalLM">Qwen3ForCausalLM</a> forward method, overrides the <code>__call__</code> special method.',Cn,R,Mn,E,en,ue,nn,B,me,Qn,O,fe,zn,Oe,Gn="The <code>GenericForSequenceClassification</code> forward method, overrides the <code>__call__</code> special method.",Fn,V,on,ge,tn,H,_e,qn,W,we,Ln,We,Xn="The <code>GenericForTokenClassification</code> forward method, overrides the <code>__call__</code> special method.",In,G,sn,be,an,U,ve,Pn,N,ye,On,Ne,Yn="The <code>GenericForQuestionAnswering</code> forward method, overrides the <code>__call__</code> special method.",Wn,X,rn,Te,dn,He,cn;return Y=new A({props:{title:"Qwen3",local:"qwen3",headingTag:"h1"}}),K=new A({props:{title:"Overview",local:"overview",headingTag:"h2"}}),ne=new A({props:{title:"Model Details",local:"model-details",headingTag:"h3"}}),te=new A({props:{title:"Usage tips",local:"usage-tips",headingTag:"h2"}}),ae=new A({props:{title:"Qwen3Config",local:"transformers.Qwen3Config",headingTag:"h2"}}),re=new I({props:{name:"class transformers.Qwen3Config",anchor:"transformers.Qwen3Config",parameters:[{name:"vocab_size",val:" = 151936"},{name:"hidden_size",val:" = 4096"},{name:"intermediate_size",val:" = 22016"},{name:"num_hidden_layers",val:" = 32"},{name:"num_attention_heads",val:" = 32"},{name:"num_key_value_heads",val:" = 32"},{name:"head_dim",val:" = 128"},{name:"hidden_act",val:" = 'silu'"},{name:"max_position_embeddings",val:" = 32768"},{name:"initializer_range",val:" = 0.02"},{name:"rms_norm_eps",val:" = 1e-06"},{name:"use_cache",val:" = True"},{name:"tie_word_embeddings",val:" = False"},{name:"rope_theta",val:" = 10000.0"},{name:"rope_scaling",val:" = None"},{name:"attention_bias",val:" = False"},{name:"use_sliding_window",val:" = False"},{name:"sliding_window",val:" = 4096"},{name:"max_window_layers",val:" = 28"},{name:"layer_types",val:" = None"},{name:"attention_dropout",val:" = 0.0"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.Qwen3Config.vocab_size",description:`<strong>vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 151936) &#x2014;
Vocabulary size of the Qwen3 model. Defines the number of different tokens that can be represented by the
<code>inputs_ids</code> passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/qwen3#transformers.Qwen3Model">Qwen3Model</a>`,name:"vocab_size"},{anchor:"transformers.Qwen3Config.hidden_size",description:`<strong>hidden_size</strong> (<code>int</code>, <em>optional</em>, defaults to 4096) &#x2014;
Dimension of the hidden representations.`,name:"hidden_size"},{anchor:"transformers.Qwen3Config.intermediate_size",description:`<strong>intermediate_size</strong> (<code>int</code>, <em>optional</em>, defaults to 22016) &#x2014;
Dimension of the MLP representations.`,name:"intermediate_size"},{anchor:"transformers.Qwen3Config.num_hidden_layers",description:`<strong>num_hidden_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 32) &#x2014;
Number of hidden layers in the Transformer encoder.`,name:"num_hidden_layers"},{anchor:"transformers.Qwen3Config.num_attention_heads",description:`<strong>num_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 32) &#x2014;
Number of attention heads for each attention layer in the Transformer encoder.`,name:"num_attention_heads"},{anchor:"transformers.Qwen3Config.num_key_value_heads",description:`<strong>num_key_value_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 32) &#x2014;
This is the number of key_value heads that should be used to implement Grouped Query Attention. If
<code>num_key_value_heads=num_attention_heads</code>, the model will use Multi Head Attention (MHA), if
<code>num_key_value_heads=1</code> the model will use Multi Query Attention (MQA) otherwise GQA is used. When
converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
by meanpooling all the original heads within that group. For more details, check out <a href="https://huggingface.co/papers/2305.13245" rel="nofollow">this
paper</a>. If it is not specified, will default to <code>32</code>.`,name:"num_key_value_heads"},{anchor:"transformers.Qwen3Config.head_dim",description:`<strong>head_dim</strong> (<code>int</code>, <em>optional</em>, defaults to 128) &#x2014;
The attention head dimension.`,name:"head_dim"},{anchor:"transformers.Qwen3Config.hidden_act",description:`<strong>hidden_act</strong> (<code>str</code> or <code>function</code>, <em>optional</em>, defaults to <code>&quot;silu&quot;</code>) &#x2014;
The non-linear activation function (function or string) in the decoder.`,name:"hidden_act"},{anchor:"transformers.Qwen3Config.max_position_embeddings",description:`<strong>max_position_embeddings</strong> (<code>int</code>, <em>optional</em>, defaults to 32768) &#x2014;
The maximum sequence length that this model might ever be used with.`,name:"max_position_embeddings"},{anchor:"transformers.Qwen3Config.initializer_range",description:`<strong>initializer_range</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.`,name:"initializer_range"},{anchor:"transformers.Qwen3Config.rms_norm_eps",description:`<strong>rms_norm_eps</strong> (<code>float</code>, <em>optional</em>, defaults to 1e-06) &#x2014;
The epsilon used by the rms normalization layers.`,name:"rms_norm_eps"},{anchor:"transformers.Qwen3Config.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not the model should return the last key/values attentions (not used by all models). Only
relevant if <code>config.is_decoder=True</code>.`,name:"use_cache"},{anchor:"transformers.Qwen3Config.tie_word_embeddings",description:`<strong>tie_word_embeddings</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether the model&#x2019;s input and output word embeddings should be tied.`,name:"tie_word_embeddings"},{anchor:"transformers.Qwen3Config.rope_theta",description:`<strong>rope_theta</strong> (<code>float</code>, <em>optional</em>, defaults to 10000.0) &#x2014;
The base period of the RoPE embeddings.`,name:"rope_theta"},{anchor:"transformers.Qwen3Config.rope_scaling",description:`<strong>rope_scaling</strong> (<code>Dict</code>, <em>optional</em>) &#x2014;
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
Only used with &#x2018;llama3&#x2019;. Scaling factor applied to high frequency components of the RoPE`,name:"rope_scaling"},{anchor:"transformers.Qwen3Config.attention_bias",description:`<strong>attention_bias</strong> (<code>bool</code>, defaults to <code>False</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether to use a bias in the query, key, value and output projection layers during self-attention.`,name:"attention_bias"},{anchor:"transformers.Qwen3Config.use_sliding_window",description:`<strong>use_sliding_window</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether to use sliding window attention.`,name:"use_sliding_window"},{anchor:"transformers.Qwen3Config.sliding_window",description:`<strong>sliding_window</strong> (<code>int</code>, <em>optional</em>, defaults to 4096) &#x2014;
Sliding window attention (SWA) window size. If not specified, will default to <code>4096</code>.`,name:"sliding_window"},{anchor:"transformers.Qwen3Config.max_window_layers",description:`<strong>max_window_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 28) &#x2014;
The number of layers using full attention. The first <code>max_window_layers</code> layers will use full attention, while any
additional layer afterwards will use SWA (Sliding Window Attention).`,name:"max_window_layers"},{anchor:"transformers.Qwen3Config.layer_types",description:`<strong>layer_types</strong> (<code>list</code>, <em>optional</em>) &#x2014;
Attention pattern for each layer.`,name:"layer_types"},{anchor:"transformers.Qwen3Config.attention_dropout",description:`<strong>attention_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The dropout ratio for the attention probabilities.`,name:"attention_dropout"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/qwen3/configuration_qwen3.py#L25"}}),D=new Kn({props:{anchor:"transformers.Qwen3Config.example",$$slots:{default:[io]},$$scope:{ctx:y}}}),ie=new A({props:{title:"Qwen3Model",local:"transformers.Qwen3Model",headingTag:"h2"}}),de=new I({props:{name:"class transformers.Qwen3Model",anchor:"transformers.Qwen3Model",parameters:[{name:"config",val:": Qwen3Config"}],parametersDescription:[{anchor:"transformers.Qwen3Model.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/qwen3#transformers.Qwen3Config">Qwen3Config</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/qwen3/modeling_qwen3.py#L336"}}),ce=new I({props:{name:"forward",anchor:"transformers.Qwen3Model.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.Qwen3Model.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.Qwen3Model.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.Qwen3Model.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.Qwen3Model.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.Qwen3Model.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.Qwen3Model.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.Qwen3Model.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/qwen3/modeling_qwen3.py#L354",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast"
>transformers.modeling_outputs.BaseModelOutputWithPast</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/qwen3#transformers.Qwen3Config"
>Qwen3Config</a>) and inputs.</p>
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
`}}),Z=new Ue({props:{$$slots:{default:[co]},$$scope:{ctx:y}}}),le=new A({props:{title:"Qwen3ForCausalLM",local:"transformers.Qwen3ForCausalLM",headingTag:"h2"}}),pe=new I({props:{name:"class transformers.Qwen3ForCausalLM",anchor:"transformers.Qwen3ForCausalLM",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.Qwen3ForCausalLM.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/qwen3#transformers.Qwen3ForCausalLM">Qwen3ForCausalLM</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/qwen3/modeling_qwen3.py#L429"}}),he=new I({props:{name:"forward",anchor:"transformers.Qwen3ForCausalLM.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"},{name:"logits_to_keep",val:": typing.Union[int, torch.Tensor] = 0"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.Qwen3ForCausalLM.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.Qwen3ForCausalLM.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.Qwen3ForCausalLM.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.Qwen3ForCausalLM.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.Qwen3ForCausalLM.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.Qwen3ForCausalLM.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.Qwen3ForCausalLM.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.Qwen3ForCausalLM.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"},{anchor:"transformers.Qwen3ForCausalLM.forward.logits_to_keep",description:`<strong>logits_to_keep</strong> (<code>Union[int, torch.Tensor]</code>, defaults to <code>0</code>) &#x2014;
If an <code>int</code>, compute logits for the last <code>logits_to_keep</code> tokens. If <code>0</code>, calculate logits for all
<code>input_ids</code> (special case). Only last token logits are needed for generation, and calculating them only for that
token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
If a <code>torch.Tensor</code>, must be 1D corresponding to the indices to keep in the sequence length dimension.
This is useful when using packed tensor format (single dimension for batch and sequence length).`,name:"logits_to_keep"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/qwen3/modeling_qwen3.py#L443",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast"
>transformers.modeling_outputs.CausalLMOutputWithPast</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/qwen3#transformers.Qwen3Config"
>Qwen3Config</a>) and inputs.</p>
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
`}}),R=new Ue({props:{$$slots:{default:[lo]},$$scope:{ctx:y}}}),E=new Kn({props:{anchor:"transformers.Qwen3ForCausalLM.forward.example",$$slots:{default:[po]},$$scope:{ctx:y}}}),ue=new A({props:{title:"Qwen3ForSequenceClassification",local:"transformers.Qwen3ForSequenceClassification",headingTag:"h2"}}),me=new I({props:{name:"class transformers.Qwen3ForSequenceClassification",anchor:"transformers.Qwen3ForSequenceClassification",parameters:[{name:"config",val:""}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/qwen3/modeling_qwen3.py#L509"}}),fe=new I({props:{name:"forward",anchor:"transformers.Qwen3ForSequenceClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.Qwen3ForSequenceClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.Qwen3ForSequenceClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.Qwen3ForSequenceClassification.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.Qwen3ForSequenceClassification.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.Qwen3ForSequenceClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.Qwen3ForSequenceClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.Qwen3ForSequenceClassification.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
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
`}}),V=new Ue({props:{$$slots:{default:[ho]},$$scope:{ctx:y}}}),ge=new A({props:{title:"Qwen3ForTokenClassification",local:"transformers.Qwen3ForTokenClassification",headingTag:"h2"}}),_e=new I({props:{name:"class transformers.Qwen3ForTokenClassification",anchor:"transformers.Qwen3ForTokenClassification",parameters:[{name:"config",val:""}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/qwen3/modeling_qwen3.py#L513"}}),we=new I({props:{name:"forward",anchor:"transformers.Qwen3ForTokenClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.Qwen3ForTokenClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.Qwen3ForTokenClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.Qwen3ForTokenClassification.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.Qwen3ForTokenClassification.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.Qwen3ForTokenClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.Qwen3ForTokenClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.Qwen3ForTokenClassification.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
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
`}}),G=new Ue({props:{$$slots:{default:[uo]},$$scope:{ctx:y}}}),be=new A({props:{title:"Qwen3ForQuestionAnswering",local:"transformers.Qwen3ForQuestionAnswering",headingTag:"h2"}}),ve=new I({props:{name:"class transformers.Qwen3ForQuestionAnswering",anchor:"transformers.Qwen3ForQuestionAnswering",parameters:[{name:"config",val:""}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/qwen3/modeling_qwen3.py#L517"}}),ye=new I({props:{name:"forward",anchor:"transformers.Qwen3ForQuestionAnswering.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"start_positions",val:": typing.Optional[torch.LongTensor] = None"},{name:"end_positions",val:": typing.Optional[torch.LongTensor] = None"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.Qwen3ForQuestionAnswering.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.Qwen3ForQuestionAnswering.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.Qwen3ForQuestionAnswering.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.Qwen3ForQuestionAnswering.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.Qwen3ForQuestionAnswering.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.Qwen3ForQuestionAnswering.forward.start_positions",description:`<strong>start_positions</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for position (index) of the start of the labelled span for computing the token classification loss.
Positions are clamped to the length of the sequence (<code>sequence_length</code>). Position outside of the sequence
are not taken into account for computing the loss.`,name:"start_positions"},{anchor:"transformers.Qwen3ForQuestionAnswering.forward.end_positions",description:`<strong>end_positions</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
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
`}}),X=new Ue({props:{$$slots:{default:[mo]},$$scope:{ctx:y}}}),Te=new ro({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/qwen3.md"}}),{c(){n=c("meta"),_=a(),s=c("p"),w=a(),z=c("p"),z.innerHTML=v,J=a(),p(Y.$$.fragment),je=a(),p(K.$$.fragment),Se=a(),ee=c("p"),ee.innerHTML=Nn,Je=a(),p(ne.$$.fragment),De=a(),oe=c("p"),oe.textContent=An,Ze=a(),p(te.$$.fragment),Re=a(),se=c("p"),se.textContent=Bn,Ee=a(),p(ae.$$.fragment),Ve=a(),$=c("div"),p(re.$$.fragment),hn=a(),xe=c("p"),xe.innerHTML=Hn,un=a(),Ce=c("p"),Ce.innerHTML=Un,mn=a(),p(D.$$.fragment),Ge=a(),p(ie.$$.fragment),Xe=a(),T=c("div"),p(de.$$.fragment),fn=a(),Me=c("p"),Me.textContent=jn,gn=a(),Qe=c("p"),Qe.innerHTML=Sn,_n=a(),ze=c("p"),ze.innerHTML=Jn,wn=a(),P=c("div"),p(ce.$$.fragment),bn=a(),Fe=c("p"),Fe.innerHTML=Dn,vn=a(),p(Z.$$.fragment),Ye=a(),p(le.$$.fragment),Ke=a(),k=c("div"),p(pe.$$.fragment),yn=a(),qe=c("p"),qe.textContent=Zn,Tn=a(),Le=c("p"),Le.innerHTML=Rn,kn=a(),Ie=c("p"),Ie.innerHTML=En,$n=a(),F=c("div"),p(he.$$.fragment),xn=a(),Pe=c("p"),Pe.innerHTML=Vn,Cn=a(),p(R.$$.fragment),Mn=a(),p(E.$$.fragment),en=a(),p(ue.$$.fragment),nn=a(),B=c("div"),p(me.$$.fragment),Qn=a(),O=c("div"),p(fe.$$.fragment),zn=a(),Oe=c("p"),Oe.innerHTML=Gn,Fn=a(),p(V.$$.fragment),on=a(),p(ge.$$.fragment),tn=a(),H=c("div"),p(_e.$$.fragment),qn=a(),W=c("div"),p(we.$$.fragment),Ln=a(),We=c("p"),We.innerHTML=Xn,In=a(),p(G.$$.fragment),sn=a(),p(be.$$.fragment),an=a(),U=c("div"),p(ve.$$.fragment),Pn=a(),N=c("div"),p(ye.$$.fragment),On=a(),Ne=c("p"),Ne.innerHTML=Yn,Wn=a(),p(X.$$.fragment),rn=a(),p(Te.$$.fragment),dn=a(),He=c("p"),this.h()},l(e){const o=ao("svelte-u9bgzb",document.head);n=l(o,"META",{name:!0,content:!0}),o.forEach(t),_=r(e),s=l(e,"P",{}),M(s).forEach(t),w=r(e),z=l(e,"P",{"data-svelte-h":!0}),b(z)!=="svelte-hz2l3k"&&(z.innerHTML=v),J=r(e),h(Y.$$.fragment,e),je=r(e),h(K.$$.fragment,e),Se=r(e),ee=l(e,"P",{"data-svelte-h":!0}),b(ee)!=="svelte-vclv1i"&&(ee.innerHTML=Nn),Je=r(e),h(ne.$$.fragment,e),De=r(e),oe=l(e,"P",{"data-svelte-h":!0}),b(oe)!=="svelte-u40g91"&&(oe.textContent=An),Ze=r(e),h(te.$$.fragment,e),Re=r(e),se=l(e,"P",{"data-svelte-h":!0}),b(se)!=="svelte-u40g91"&&(se.textContent=Bn),Ee=r(e),h(ae.$$.fragment,e),Ve=r(e),$=l(e,"DIV",{class:!0});var q=M($);h(re.$$.fragment,q),hn=r(q),xe=l(q,"P",{"data-svelte-h":!0}),b(xe)!=="svelte-13cw57k"&&(xe.innerHTML=Hn),un=r(q),Ce=l(q,"P",{"data-svelte-h":!0}),b(Ce)!=="svelte-1ek1ss9"&&(Ce.innerHTML=Un),mn=r(q),h(D.$$.fragment,q),q.forEach(t),Ge=r(e),h(ie.$$.fragment,e),Xe=r(e),T=l(e,"DIV",{class:!0});var x=M(T);h(de.$$.fragment,x),fn=r(x),Me=l(x,"P",{"data-svelte-h":!0}),b(Me)!=="svelte-16tbpaa"&&(Me.textContent=jn),gn=r(x),Qe=l(x,"P",{"data-svelte-h":!0}),b(Qe)!=="svelte-q52n56"&&(Qe.innerHTML=Sn),_n=r(x),ze=l(x,"P",{"data-svelte-h":!0}),b(ze)!=="svelte-hswkmf"&&(ze.innerHTML=Jn),wn=r(x),P=l(x,"DIV",{class:!0});var j=M(P);h(ce.$$.fragment,j),bn=r(j),Fe=l(j,"P",{"data-svelte-h":!0}),b(Fe)!=="svelte-1192cdd"&&(Fe.innerHTML=Dn),vn=r(j),h(Z.$$.fragment,j),j.forEach(t),x.forEach(t),Ye=r(e),h(le.$$.fragment,e),Ke=r(e),k=l(e,"DIV",{class:!0});var C=M(k);h(pe.$$.fragment,C),yn=r(C),qe=l(C,"P",{"data-svelte-h":!0}),b(qe)!=="svelte-1f4yb3f"&&(qe.textContent=Zn),Tn=r(C),Le=l(C,"P",{"data-svelte-h":!0}),b(Le)!=="svelte-q52n56"&&(Le.innerHTML=Rn),kn=r(C),Ie=l(C,"P",{"data-svelte-h":!0}),b(Ie)!=="svelte-hswkmf"&&(Ie.innerHTML=En),$n=r(C),F=l(C,"DIV",{class:!0});var L=M(F);h(he.$$.fragment,L),xn=r(L),Pe=l(L,"P",{"data-svelte-h":!0}),b(Pe)!=="svelte-zo7cal"&&(Pe.innerHTML=Vn),Cn=r(L),h(R.$$.fragment,L),Mn=r(L),h(E.$$.fragment,L),L.forEach(t),C.forEach(t),en=r(e),h(ue.$$.fragment,e),nn=r(e),B=l(e,"DIV",{class:!0});var ke=M(B);h(me.$$.fragment,ke),Qn=r(ke),O=l(ke,"DIV",{class:!0});var S=M(O);h(fe.$$.fragment,S),zn=r(S),Oe=l(S,"P",{"data-svelte-h":!0}),b(Oe)!=="svelte-1sal4ui"&&(Oe.innerHTML=Gn),Fn=r(S),h(V.$$.fragment,S),S.forEach(t),ke.forEach(t),on=r(e),h(ge.$$.fragment,e),tn=r(e),H=l(e,"DIV",{class:!0});var ln=M(H);h(_e.$$.fragment,ln),qn=r(ln),W=l(ln,"DIV",{class:!0});var Ae=M(W);h(we.$$.fragment,Ae),Ln=r(Ae),We=l(Ae,"P",{"data-svelte-h":!0}),b(We)!=="svelte-1py4aay"&&(We.innerHTML=Xn),In=r(Ae),h(G.$$.fragment,Ae),Ae.forEach(t),ln.forEach(t),sn=r(e),h(be.$$.fragment,e),an=r(e),U=l(e,"DIV",{class:!0});var pn=M(U);h(ve.$$.fragment,pn),Pn=r(pn),N=l(pn,"DIV",{class:!0});var Be=M(N);h(ye.$$.fragment,Be),On=r(Be),Ne=l(Be,"P",{"data-svelte-h":!0}),b(Ne)!=="svelte-dyrov9"&&(Ne.innerHTML=Yn),Wn=r(Be),h(X.$$.fragment,Be),Be.forEach(t),pn.forEach(t),rn=r(e),h(Te.$$.fragment,e),dn=r(e),He=l(e,"P",{}),M(He).forEach(t),this.h()},h(){Q(n,"name","hf:doc:metadata"),Q(n,"content",go),Q($,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),Q(P,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),Q(T,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),Q(F,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),Q(k,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),Q(O,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),Q(B,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),Q(W,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),Q(H,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),Q(N,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),Q(U,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,o){i(document.head,n),d(e,_,o),d(e,s,o),d(e,w,o),d(e,z,o),d(e,J,o),u(Y,e,o),d(e,je,o),u(K,e,o),d(e,Se,o),d(e,ee,o),d(e,Je,o),u(ne,e,o),d(e,De,o),d(e,oe,o),d(e,Ze,o),u(te,e,o),d(e,Re,o),d(e,se,o),d(e,Ee,o),u(ae,e,o),d(e,Ve,o),d(e,$,o),u(re,$,null),i($,hn),i($,xe),i($,un),i($,Ce),i($,mn),u(D,$,null),d(e,Ge,o),u(ie,e,o),d(e,Xe,o),d(e,T,o),u(de,T,null),i(T,fn),i(T,Me),i(T,gn),i(T,Qe),i(T,_n),i(T,ze),i(T,wn),i(T,P),u(ce,P,null),i(P,bn),i(P,Fe),i(P,vn),u(Z,P,null),d(e,Ye,o),u(le,e,o),d(e,Ke,o),d(e,k,o),u(pe,k,null),i(k,yn),i(k,qe),i(k,Tn),i(k,Le),i(k,kn),i(k,Ie),i(k,$n),i(k,F),u(he,F,null),i(F,xn),i(F,Pe),i(F,Cn),u(R,F,null),i(F,Mn),u(E,F,null),d(e,en,o),u(ue,e,o),d(e,nn,o),d(e,B,o),u(me,B,null),i(B,Qn),i(B,O),u(fe,O,null),i(O,zn),i(O,Oe),i(O,Fn),u(V,O,null),d(e,on,o),u(ge,e,o),d(e,tn,o),d(e,H,o),u(_e,H,null),i(H,qn),i(H,W),u(we,W,null),i(W,Ln),i(W,We),i(W,In),u(G,W,null),d(e,sn,o),u(be,e,o),d(e,an,o),d(e,U,o),u(ve,U,null),i(U,Pn),i(U,N),u(ye,N,null),i(N,On),i(N,Ne),i(N,Wn),u(X,N,null),d(e,rn,o),u(Te,e,o),d(e,dn,o),d(e,He,o),cn=!0},p(e,[o]){const q={};o&2&&(q.$$scope={dirty:o,ctx:e}),D.$set(q);const x={};o&2&&(x.$$scope={dirty:o,ctx:e}),Z.$set(x);const j={};o&2&&(j.$$scope={dirty:o,ctx:e}),R.$set(j);const C={};o&2&&(C.$$scope={dirty:o,ctx:e}),E.$set(C);const L={};o&2&&(L.$$scope={dirty:o,ctx:e}),V.$set(L);const ke={};o&2&&(ke.$$scope={dirty:o,ctx:e}),G.$set(ke);const S={};o&2&&(S.$$scope={dirty:o,ctx:e}),X.$set(S)},i(e){cn||(m(Y.$$.fragment,e),m(K.$$.fragment,e),m(ne.$$.fragment,e),m(te.$$.fragment,e),m(ae.$$.fragment,e),m(re.$$.fragment,e),m(D.$$.fragment,e),m(ie.$$.fragment,e),m(de.$$.fragment,e),m(ce.$$.fragment,e),m(Z.$$.fragment,e),m(le.$$.fragment,e),m(pe.$$.fragment,e),m(he.$$.fragment,e),m(R.$$.fragment,e),m(E.$$.fragment,e),m(ue.$$.fragment,e),m(me.$$.fragment,e),m(fe.$$.fragment,e),m(V.$$.fragment,e),m(ge.$$.fragment,e),m(_e.$$.fragment,e),m(we.$$.fragment,e),m(G.$$.fragment,e),m(be.$$.fragment,e),m(ve.$$.fragment,e),m(ye.$$.fragment,e),m(X.$$.fragment,e),m(Te.$$.fragment,e),cn=!0)},o(e){f(Y.$$.fragment,e),f(K.$$.fragment,e),f(ne.$$.fragment,e),f(te.$$.fragment,e),f(ae.$$.fragment,e),f(re.$$.fragment,e),f(D.$$.fragment,e),f(ie.$$.fragment,e),f(de.$$.fragment,e),f(ce.$$.fragment,e),f(Z.$$.fragment,e),f(le.$$.fragment,e),f(pe.$$.fragment,e),f(he.$$.fragment,e),f(R.$$.fragment,e),f(E.$$.fragment,e),f(ue.$$.fragment,e),f(me.$$.fragment,e),f(fe.$$.fragment,e),f(V.$$.fragment,e),f(ge.$$.fragment,e),f(_e.$$.fragment,e),f(we.$$.fragment,e),f(G.$$.fragment,e),f(be.$$.fragment,e),f(ve.$$.fragment,e),f(ye.$$.fragment,e),f(X.$$.fragment,e),f(Te.$$.fragment,e),cn=!1},d(e){e&&(t(_),t(s),t(w),t(z),t(J),t(je),t(Se),t(ee),t(Je),t(De),t(oe),t(Ze),t(Re),t(se),t(Ee),t(Ve),t($),t(Ge),t(Xe),t(T),t(Ye),t(Ke),t(k),t(en),t(nn),t(B),t(on),t(tn),t(H),t(sn),t(an),t(U),t(rn),t(dn),t(He)),t(n),g(Y,e),g(K,e),g(ne,e),g(te,e),g(ae,e),g(re),g(D),g(ie,e),g(de),g(ce),g(Z),g(le,e),g(pe),g(he),g(R),g(E),g(ue,e),g(me),g(fe),g(V),g(ge,e),g(_e),g(we),g(G),g(be,e),g(ve),g(ye),g(X),g(Te,e)}}}const go='{"title":"Qwen3","local":"qwen3","sections":[{"title":"Overview","local":"overview","sections":[{"title":"Model Details","local":"model-details","sections":[],"depth":3}],"depth":2},{"title":"Usage tips","local":"usage-tips","sections":[],"depth":2},{"title":"Qwen3Config","local":"transformers.Qwen3Config","sections":[],"depth":2},{"title":"Qwen3Model","local":"transformers.Qwen3Model","sections":[],"depth":2},{"title":"Qwen3ForCausalLM","local":"transformers.Qwen3ForCausalLM","sections":[],"depth":2},{"title":"Qwen3ForSequenceClassification","local":"transformers.Qwen3ForSequenceClassification","sections":[],"depth":2},{"title":"Qwen3ForTokenClassification","local":"transformers.Qwen3ForTokenClassification","sections":[],"depth":2},{"title":"Qwen3ForQuestionAnswering","local":"transformers.Qwen3ForQuestionAnswering","sections":[],"depth":2}],"depth":1}';function _o(y){return oo(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class xo extends to{constructor(n){super(),so(this,n,_o,fo,no,{})}}export{xo as component};
