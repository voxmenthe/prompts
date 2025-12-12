import{s as yo,o as To,n as je}from"../chunks/scheduler.18a86fab.js";import{S as wo,i as xo,g as l,s as a,r as f,A as ko,h as d,f as t,c as r,j as ne,x as m,u as g,k as se,y as c,a as s,v as _,d as b,t as M,w as v}from"../chunks/index.98837b22.js";import{T as bo}from"../chunks/Tip.77304350.js";import{D as ge}from"../chunks/Docstring.a1ef7999.js";import{C as vo}from"../chunks/CodeBlock.8d0c2e8a.js";import{E as Mo}from"../chunks/ExampleCodeBlock.8c3ee1f9.js";import{H as _e,E as $o}from"../chunks/getInferenceSnippets.06c2775f.js";function Co(z){let n,p;return n=new vo({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEdsbTRNb2VNb2RlbCUyQyUyMEdsbTRNb2VDb25maWclMEElMEElMjMlMjBJbml0aWFsaXppbmclMjBhJTIwR2xtNE1vZSUyMHN0eWxlJTIwY29uZmlndXJhdGlvbiUwQWNvbmZpZ3VyYXRpb24lMjAlM0QlMjBHbG00TW9lQ29uZmlnKCklMEElMEElMjMlMjBJbml0aWFsaXppbmclMjBhJTIwbW9kZWwlMjBmcm9tJTIwdGhlJTIwR0xNLTQtTU9FLTEwMEItQTEwQiUyMHN0eWxlJTIwY29uZmlndXJhdGlvbiUwQW1vZGVsJTIwJTNEJTIwR2xtNE1vZU1vZGVsKGNvbmZpZ3VyYXRpb24pJTBBJTBBJTIzJTIwQWNjZXNzaW5nJTIwdGhlJTIwbW9kZWwlMjBjb25maWd1cmF0aW9uJTBBY29uZmlndXJhdGlvbiUyMCUzRCUyMG1vZGVsLmNvbmZpZw==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> Glm4MoeModel, Glm4MoeConfig

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a Glm4Moe style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = Glm4MoeConfig()

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a model from the GLM-4-MOE-100B-A10B style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = Glm4MoeModel(configuration)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Accessing the model configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = model.config`,wrap:!1}}),{c(){f(n.$$.fragment)},l(i){g(n.$$.fragment,i)},m(i,h){_(n,i,h),p=!0},p:je,i(i){p||(b(n.$$.fragment,i),p=!0)},o(i){M(n.$$.fragment,i),p=!1},d(i){v(n,i)}}}function Go(z){let n,p=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){n=l("p"),n.innerHTML=p},l(i){n=d(i,"P",{"data-svelte-h":!0}),m(n)!=="svelte-fincs2"&&(n.innerHTML=p)},m(i,h){s(i,n,h)},p:je,d(i){i&&t(n)}}}function zo(z){let n,p=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){n=l("p"),n.innerHTML=p},l(i){n=d(i,"P",{"data-svelte-h":!0}),m(n)!=="svelte-fincs2"&&(n.innerHTML=p)},m(i,h){s(i,n,h)},p:je,d(i){i&&t(n)}}}function Lo(z){let n,p="Example:",i,h,$;return h=new vo({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBHbG00TW9lRm9yQ2F1c2FsTE0lMEElMEFtb2RlbCUyMCUzRCUyMEdsbTRNb2VGb3JDYXVzYWxMTS5mcm9tX3ByZXRyYWluZWQoJTIybWV0YS1nbG00X21vZSUyRkdsbTRNb2UtMi03Yi1oZiUyMiklMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJtZXRhLWdsbTRfbW9lJTJGR2xtNE1vZS0yLTdiLWhmJTIyKSUwQSUwQXByb21wdCUyMCUzRCUyMCUyMkhleSUyQyUyMGFyZSUyMHlvdSUyMGNvbnNjaW91cyUzRiUyMENhbiUyMHlvdSUyMHRhbGslMjB0byUyMG1lJTNGJTIyJTBBaW5wdXRzJTIwJTNEJTIwdG9rZW5pemVyKHByb21wdCUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpJTBBJTBBJTIzJTIwR2VuZXJhdGUlMEFnZW5lcmF0ZV9pZHMlMjAlM0QlMjBtb2RlbC5nZW5lcmF0ZShpbnB1dHMuaW5wdXRfaWRzJTJDJTIwbWF4X2xlbmd0aCUzRDMwKSUwQXRva2VuaXplci5iYXRjaF9kZWNvZGUoZ2VuZXJhdGVfaWRzJTJDJTIwc2tpcF9zcGVjaWFsX3Rva2VucyUzRFRydWUlMkMlMjBjbGVhbl91cF90b2tlbml6YXRpb25fc3BhY2VzJTNERmFsc2UpJTVCMCU1RA==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, Glm4MoeForCausalLM

<span class="hljs-meta">&gt;&gt;&gt; </span>model = Glm4MoeForCausalLM.from_pretrained(<span class="hljs-string">&quot;meta-glm4_moe/Glm4Moe-2-7b-hf&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;meta-glm4_moe/Glm4Moe-2-7b-hf&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>prompt = <span class="hljs-string">&quot;Hey, are you conscious? Can you talk to me?&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(prompt, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Generate</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>generate_ids = model.generate(inputs.input_ids, max_length=<span class="hljs-number">30</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer.batch_decode(generate_ids, skip_special_tokens=<span class="hljs-literal">True</span>, clean_up_tokenization_spaces=<span class="hljs-literal">False</span>)[<span class="hljs-number">0</span>]
<span class="hljs-string">&quot;Hey, are you conscious? Can you talk to me?\\nI&#x27;m not conscious, but I can talk to you.&quot;</span>`,wrap:!1}}),{c(){n=l("p"),n.textContent=p,i=a(),f(h.$$.fragment)},l(u){n=d(u,"P",{"data-svelte-h":!0}),m(n)!=="svelte-11lpom8"&&(n.textContent=p),i=r(u),g(h.$$.fragment,u)},m(u,P){s(u,n,P),s(u,i,P),_(h,u,P),$=!0},p:je,i(u){$||(b(h.$$.fragment,u),$=!0)},o(u){M(h.$$.fragment,u),$=!1},d(u){u&&(t(n),t(i)),v(h,u)}}}function Fo(z){let n,p,i,h,$,u="<em>This model was released on 2025-07-28 and added to Hugging Face Transformers on 2025-07-21.</em>",P,U,be,J,Me,E,Ke='The <a href="https://huggingface.co/papers/2508.06471" rel="nofollow"><strong>GLM-4.5</strong></a> series models are foundation models designed for intelligent agents, MoE variants are documented here as Glm4Moe.',ve,N,eo="GLM-4.5 has <strong>355</strong> billion total parameters with <strong>32</strong> billion active parameters, while GLM-4.5-Air adopts a more compact design with <strong>106</strong> billion total parameters and <strong>12</strong> billion active parameters. GLM-4.5 models unify reasoning, coding, and intelligent agent capabilities to meet the complex demands of intelligent agent applications.",ye,R,oo="Both GLM-4.5 and GLM-4.5-Air are hybrid reasoning models that provide two modes: thinking mode for complex reasoning and tool usage, and non-thinking mode for immediate responses.",Te,B,to="We have open-sourced the base models, hybrid reasoning models, and FP8 versions of the hybrid reasoning models for both GLM-4.5 and GLM-4.5-Air. They are released under the MIT open-source license and can be used commercially and for secondary development.",we,Z,no="As demonstrated in our comprehensive evaluation across 12 industry-standard benchmarks, GLM-4.5 achieves exceptional performance with a score of <strong>63.2</strong>, in the <strong>3rd</strong> place among all the proprietary and open-source models. Notably, GLM-4.5-Air delivers competitive results at <strong>59.8</strong> while maintaining superior efficiency.",xe,O,so='<img src="https://raw.githubusercontent.com/zai-org/GLM-4.5/refs/heads/main/resources/bench.png" alt="bench"/>',ke,D,ao='For more eval results, show cases, and technical details, please visit our <a href="https://huggingface.co/papers/2508.06471" rel="nofollow">technical report</a> or <a href="https://z.ai/blog/glm-4.5" rel="nofollow">technical blog</a>.',$e,A,ro='The model code, tool parser and reasoning parser can be found in the implementation of <a href="https://github.com/huggingface/transformers/tree/main/src/transformers/models/glm4_moe" rel="nofollow">transformers</a>, <a href="https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/glm4_moe_mtp.py" rel="nofollow">vLLM</a> and <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/glm4_moe.py" rel="nofollow">SGLang</a>.',Ce,V,Ge,w,S,He,ae,io=`This is the configuration class to store the configuration of a <a href="/docs/transformers/v4.56.2/en/model_doc/glm4_moe#transformers.Glm4MoeModel">Glm4MoeModel</a>. It is used to instantiate a
Glm4Moe model according to the specified arguments, defining the model architecture. Instantiating a configuration
with the defaults will yield a similar configuration to that of <a href="https://huggingface.co/THUDM/GLM-4-100B-A10B" rel="nofollow">THUDM/GLM-4-100B-A10B</a>.`,Ue,re,lo=`Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> and can be used to control the model outputs. Read the
documentation from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> for more information.`,Je,I,ze,X,Le,y,Q,Ee,ie,co="The bare Glm4 Moe Model outputting raw hidden-states without any specific head on top.",Ne,le,mo=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Re,de,po=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Be,L,Y,Ze,ce,ho='The <a href="/docs/transformers/v4.56.2/en/model_doc/glm4_moe#transformers.Glm4MoeModel">Glm4MoeModel</a> forward method, overrides the <code>__call__</code> special method.',Oe,q,Fe,K,Pe,T,ee,De,me,uo="The Glm4 Moe Model for causal language modeling.",Ae,pe,fo=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Ve,he,go=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Se,C,oe,Xe,ue,_o='The <a href="/docs/transformers/v4.56.2/en/model_doc/glm4_moe#transformers.Glm4MoeForCausalLM">Glm4MoeForCausalLM</a> forward method, overrides the <code>__call__</code> special method.',Qe,W,Ye,j,Ie,te,qe,fe,We;return U=new _e({props:{title:"Glm4Moe",local:"glm4moe",headingTag:"h1"}}),J=new _e({props:{title:"Overview",local:"overview",headingTag:"h2"}}),V=new _e({props:{title:"Glm4MoeConfig",local:"transformers.Glm4MoeConfig",headingTag:"h2"}}),S=new ge({props:{name:"class transformers.Glm4MoeConfig",anchor:"transformers.Glm4MoeConfig",parameters:[{name:"vocab_size",val:" = 151552"},{name:"hidden_size",val:" = 4096"},{name:"intermediate_size",val:" = 10944"},{name:"num_hidden_layers",val:" = 46"},{name:"num_attention_heads",val:" = 96"},{name:"partial_rotary_factor",val:" = 0.5"},{name:"num_key_value_heads",val:" = 8"},{name:"hidden_act",val:" = 'silu'"},{name:"max_position_embeddings",val:" = 131072"},{name:"initializer_range",val:" = 0.02"},{name:"rms_norm_eps",val:" = 1e-05"},{name:"use_cache",val:" = True"},{name:"tie_word_embeddings",val:" = False"},{name:"rope_theta",val:" = 10000.0"},{name:"rope_scaling",val:" = None"},{name:"attention_bias",val:" = False"},{name:"attention_dropout",val:" = 0.0"},{name:"moe_intermediate_size",val:" = 1408"},{name:"num_experts_per_tok",val:" = 8"},{name:"n_shared_experts",val:" = 1"},{name:"n_routed_experts",val:" = 128"},{name:"routed_scaling_factor",val:" = 1.0"},{name:"n_group",val:" = 1"},{name:"topk_group",val:" = 1"},{name:"first_k_dense_replace",val:" = 1"},{name:"norm_topk_prob",val:" = True"},{name:"use_qk_norm",val:" = False"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.Glm4MoeConfig.vocab_size",description:`<strong>vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 151552) &#x2014;
Vocabulary size of the Glm4Moe model. Defines the number of different tokens that can be represented by the
<code>inputs_ids</code> passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/glm4_moe#transformers.Glm4MoeModel">Glm4MoeModel</a>`,name:"vocab_size"},{anchor:"transformers.Glm4MoeConfig.hidden_size",description:`<strong>hidden_size</strong> (<code>int</code>, <em>optional</em>, defaults to 4096) &#x2014;
Dimension of the hidden representations.`,name:"hidden_size"},{anchor:"transformers.Glm4MoeConfig.intermediate_size",description:`<strong>intermediate_size</strong> (<code>int</code>, <em>optional</em>, defaults to 10944) &#x2014;
Dimension of the MLP representations.`,name:"intermediate_size"},{anchor:"transformers.Glm4MoeConfig.num_hidden_layers",description:`<strong>num_hidden_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 46) &#x2014;
Number of hidden layers in the Transformer encoder.`,name:"num_hidden_layers"},{anchor:"transformers.Glm4MoeConfig.num_attention_heads",description:`<strong>num_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 96) &#x2014;
Number of attention heads for each attention layer in the Transformer encoder.`,name:"num_attention_heads"},{anchor:"transformers.Glm4MoeConfig.partial_rotary_factor",description:`<strong>partial_rotary_factor</strong> (<code>float</code>, <em>optional</em>, defaults to 0.5) &#x2014;
The factor of the partial rotary position.`,name:"partial_rotary_factor"},{anchor:"transformers.Glm4MoeConfig.num_key_value_heads",description:`<strong>num_key_value_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 8) &#x2014;
This is the number of key_value heads that should be used to implement Grouped Query Attention. If
<code>num_key_value_heads=num_attention_heads</code>, the model will use Multi Head Attention (MHA), if
<code>num_key_value_heads=1</code> the model will use Multi Query Attention (MQA) otherwise GQA is used. When
converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
by meanpooling all the original heads within that group. For more details, check out <a href="https://huggingface.co/papers/2305.13245" rel="nofollow">this
paper</a>. If it is not specified, will default to <code>32</code>.`,name:"num_key_value_heads"},{anchor:"transformers.Glm4MoeConfig.hidden_act",description:`<strong>hidden_act</strong> (<code>str</code> or <code>function</code>, <em>optional</em>, defaults to <code>&quot;silu&quot;</code>) &#x2014;
The non-linear activation function (function or string) in the decoder.`,name:"hidden_act"},{anchor:"transformers.Glm4MoeConfig.max_position_embeddings",description:`<strong>max_position_embeddings</strong> (<code>int</code>, <em>optional</em>, defaults to 131072) &#x2014;
The maximum sequence length that this model might ever be used with.`,name:"max_position_embeddings"},{anchor:"transformers.Glm4MoeConfig.initializer_range",description:`<strong>initializer_range</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.`,name:"initializer_range"},{anchor:"transformers.Glm4MoeConfig.rms_norm_eps",description:`<strong>rms_norm_eps</strong> (<code>float</code>, <em>optional</em>, defaults to 1e-05) &#x2014;
The epsilon used by the rms normalization layers.`,name:"rms_norm_eps"},{anchor:"transformers.Glm4MoeConfig.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not the model should return the last key/values attentions (not used by all models). Only
relevant if <code>config.is_decoder=True</code>.`,name:"use_cache"},{anchor:"transformers.Glm4MoeConfig.tie_word_embeddings",description:`<strong>tie_word_embeddings</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether the model&#x2019;s input and output word embeddings should be tied.`,name:"tie_word_embeddings"},{anchor:"transformers.Glm4MoeConfig.rope_theta",description:`<strong>rope_theta</strong> (<code>float</code>, <em>optional</em>, defaults to 10000.0) &#x2014;
The base period of the RoPE embeddings.`,name:"rope_theta"},{anchor:"transformers.Glm4MoeConfig.rope_scaling",description:`<strong>rope_scaling</strong> (<code>Dict</code>, <em>optional</em>) &#x2014;
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
Only used with &#x2018;llama3&#x2019;. Scaling factor applied to high frequency components of the RoPE`,name:"rope_scaling"},{anchor:"transformers.Glm4MoeConfig.attention_bias",description:`<strong>attention_bias</strong> (<code>bool</code>, defaults to <code>False</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether to use a bias in the query, key, value and output projection layers during self-attention.`,name:"attention_bias"},{anchor:"transformers.Glm4MoeConfig.attention_dropout",description:`<strong>attention_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The dropout ratio for the attention probabilities.`,name:"attention_dropout"},{anchor:"transformers.Glm4MoeConfig.moe_intermediate_size",description:`<strong>moe_intermediate_size</strong> (<code>int</code>, <em>optional</em>, defaults to 1408) &#x2014;
Intermediate size of the routed expert.`,name:"moe_intermediate_size"},{anchor:"transformers.Glm4MoeConfig.num_experts_per_tok",description:`<strong>num_experts_per_tok</strong> (<code>int</code>, <em>optional</em>, defaults to 8) &#x2014;
number of experts per token.`,name:"num_experts_per_tok"},{anchor:"transformers.Glm4MoeConfig.n_shared_experts",description:`<strong>n_shared_experts</strong> (<code>int</code>, <em>optional</em>, defaults to 1) &#x2014;
Number of shared experts.`,name:"n_shared_experts"},{anchor:"transformers.Glm4MoeConfig.n_routed_experts",description:`<strong>n_routed_experts</strong> (<code>int</code>, <em>optional</em>, defaults to 128) &#x2014;
Number of routed experts.`,name:"n_routed_experts"},{anchor:"transformers.Glm4MoeConfig.routed_scaling_factor",description:`<strong>routed_scaling_factor</strong> (<code>float</code>, <em>optional</em>, defaults to 1.0) &#x2014;
Scaling factor or routed experts.`,name:"routed_scaling_factor"},{anchor:"transformers.Glm4MoeConfig.n_group",description:`<strong>n_group</strong> (<code>int</code>, <em>optional</em>, defaults to 1) &#x2014;
Number of groups for routed experts.`,name:"n_group"},{anchor:"transformers.Glm4MoeConfig.topk_group",description:`<strong>topk_group</strong> (<code>int</code>, <em>optional</em>, defaults to 1) &#x2014;
Number of selected groups for each token(for each token, ensuring the selected experts is only within <code>topk_group</code> groups).`,name:"topk_group"},{anchor:"transformers.Glm4MoeConfig.first_k_dense_replace",description:`<strong>first_k_dense_replace</strong> (<code>int</code>, <em>optional</em>, defaults to 1) &#x2014;
Number of dense layers in shallow layers(embed-&gt;dense-&gt;dense-&gt;&#x2026;-&gt;dense-&gt;moe-&gt;moe&#x2026;-&gt;lm_head).
--k dense layers&#x2014;/`,name:"first_k_dense_replace"},{anchor:"transformers.Glm4MoeConfig.norm_topk_prob",description:`<strong>norm_topk_prob</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether to normalize the topk probabilities.`,name:"norm_topk_prob"},{anchor:"transformers.Glm4MoeConfig.use_qk_norm",description:`<strong>use_qk_norm</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether to use query-key normalization in the attention`,name:"use_qk_norm"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/glm4_moe/configuration_glm4_moe.py#L26"}}),I=new Mo({props:{anchor:"transformers.Glm4MoeConfig.example",$$slots:{default:[Co]},$$scope:{ctx:z}}}),X=new _e({props:{title:"Glm4MoeModel",local:"transformers.Glm4MoeModel",headingTag:"h2"}}),Q=new ge({props:{name:"class transformers.Glm4MoeModel",anchor:"transformers.Glm4MoeModel",parameters:[{name:"config",val:": Glm4MoeConfig"}],parametersDescription:[{anchor:"transformers.Glm4MoeModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/glm4_moe#transformers.Glm4MoeConfig">Glm4MoeConfig</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/glm4_moe/modeling_glm4_moe.py#L460"}}),Y=new ge({props:{name:"forward",anchor:"transformers.Glm4MoeModel.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.Glm4MoeModel.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.Glm4MoeModel.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.Glm4MoeModel.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.Glm4MoeModel.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.Glm4MoeModel.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.Glm4MoeModel.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"},{anchor:"transformers.Glm4MoeModel.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/glm4_moe/modeling_glm4_moe.py#L479",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast"
>transformers.modeling_outputs.BaseModelOutputWithPast</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/glm4_moe#transformers.Glm4MoeConfig"
>Glm4MoeConfig</a>) and inputs.</p>
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
`}}),q=new bo({props:{$$slots:{default:[Go]},$$scope:{ctx:z}}}),K=new _e({props:{title:"Glm4MoeForCausalLM",local:"transformers.Glm4MoeForCausalLM",headingTag:"h2"}}),ee=new ge({props:{name:"class transformers.Glm4MoeForCausalLM",anchor:"transformers.Glm4MoeForCausalLM",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.Glm4MoeForCausalLM.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/glm4_moe#transformers.Glm4MoeForCausalLM">Glm4MoeForCausalLM</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/glm4_moe/modeling_glm4_moe.py#L541"}}),oe=new ge({props:{name:"forward",anchor:"transformers.Glm4MoeForCausalLM.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"},{name:"logits_to_keep",val:": typing.Union[int, torch.Tensor] = 0"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.Glm4MoeForCausalLM.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.Glm4MoeForCausalLM.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.Glm4MoeForCausalLM.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.Glm4MoeForCausalLM.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.Glm4MoeForCausalLM.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.Glm4MoeForCausalLM.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.Glm4MoeForCausalLM.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.Glm4MoeForCausalLM.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"},{anchor:"transformers.Glm4MoeForCausalLM.forward.logits_to_keep",description:`<strong>logits_to_keep</strong> (<code>Union[int, torch.Tensor]</code>, defaults to <code>0</code>) &#x2014;
If an <code>int</code>, compute logits for the last <code>logits_to_keep</code> tokens. If <code>0</code>, calculate logits for all
<code>input_ids</code> (special case). Only last token logits are needed for generation, and calculating them only for that
token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
If a <code>torch.Tensor</code>, must be 1D corresponding to the indices to keep in the sequence length dimension.
This is useful when using packed tensor format (single dimension for batch and sequence length).`,name:"logits_to_keep"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/glm4_moe/modeling_glm4_moe.py#L555",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast"
>transformers.modeling_outputs.CausalLMOutputWithPast</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/glm4_moe#transformers.Glm4MoeConfig"
>Glm4MoeConfig</a>) and inputs.</p>
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
`}}),W=new bo({props:{$$slots:{default:[zo]},$$scope:{ctx:z}}}),j=new Mo({props:{anchor:"transformers.Glm4MoeForCausalLM.forward.example",$$slots:{default:[Lo]},$$scope:{ctx:z}}}),te=new $o({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/glm4_moe.md"}}),{c(){n=l("meta"),p=a(),i=l("p"),h=a(),$=l("p"),$.innerHTML=u,P=a(),f(U.$$.fragment),be=a(),f(J.$$.fragment),Me=a(),E=l("p"),E.innerHTML=Ke,ve=a(),N=l("p"),N.innerHTML=eo,ye=a(),R=l("p"),R.textContent=oo,Te=a(),B=l("p"),B.textContent=to,we=a(),Z=l("p"),Z.innerHTML=no,xe=a(),O=l("p"),O.innerHTML=so,ke=a(),D=l("p"),D.innerHTML=ao,$e=a(),A=l("p"),A.innerHTML=ro,Ce=a(),f(V.$$.fragment),Ge=a(),w=l("div"),f(S.$$.fragment),He=a(),ae=l("p"),ae.innerHTML=io,Ue=a(),re=l("p"),re.innerHTML=lo,Je=a(),f(I.$$.fragment),ze=a(),f(X.$$.fragment),Le=a(),y=l("div"),f(Q.$$.fragment),Ee=a(),ie=l("p"),ie.textContent=co,Ne=a(),le=l("p"),le.innerHTML=mo,Re=a(),de=l("p"),de.innerHTML=po,Be=a(),L=l("div"),f(Y.$$.fragment),Ze=a(),ce=l("p"),ce.innerHTML=ho,Oe=a(),f(q.$$.fragment),Fe=a(),f(K.$$.fragment),Pe=a(),T=l("div"),f(ee.$$.fragment),De=a(),me=l("p"),me.textContent=uo,Ae=a(),pe=l("p"),pe.innerHTML=fo,Ve=a(),he=l("p"),he.innerHTML=go,Se=a(),C=l("div"),f(oe.$$.fragment),Xe=a(),ue=l("p"),ue.innerHTML=_o,Qe=a(),f(W.$$.fragment),Ye=a(),f(j.$$.fragment),Ie=a(),f(te.$$.fragment),qe=a(),fe=l("p"),this.h()},l(e){const o=ko("svelte-u9bgzb",document.head);n=d(o,"META",{name:!0,content:!0}),o.forEach(t),p=r(e),i=d(e,"P",{}),ne(i).forEach(t),h=r(e),$=d(e,"P",{"data-svelte-h":!0}),m($)!=="svelte-1mwnviv"&&($.innerHTML=u),P=r(e),g(U.$$.fragment,e),be=r(e),g(J.$$.fragment,e),Me=r(e),E=d(e,"P",{"data-svelte-h":!0}),m(E)!=="svelte-7d0e4t"&&(E.innerHTML=Ke),ve=r(e),N=d(e,"P",{"data-svelte-h":!0}),m(N)!=="svelte-1dm0g9a"&&(N.innerHTML=eo),ye=r(e),R=d(e,"P",{"data-svelte-h":!0}),m(R)!=="svelte-1paojlh"&&(R.textContent=oo),Te=r(e),B=d(e,"P",{"data-svelte-h":!0}),m(B)!=="svelte-lonb1a"&&(B.textContent=to),we=r(e),Z=d(e,"P",{"data-svelte-h":!0}),m(Z)!=="svelte-1bi4kd9"&&(Z.innerHTML=no),xe=r(e),O=d(e,"P",{"data-svelte-h":!0}),m(O)!=="svelte-nd6oml"&&(O.innerHTML=so),ke=r(e),D=d(e,"P",{"data-svelte-h":!0}),m(D)!=="svelte-1kymb1u"&&(D.innerHTML=ao),$e=r(e),A=d(e,"P",{"data-svelte-h":!0}),m(A)!=="svelte-1nxi3fz"&&(A.innerHTML=ro),Ce=r(e),g(V.$$.fragment,e),Ge=r(e),w=d(e,"DIV",{class:!0});var G=ne(w);g(S.$$.fragment,G),He=r(G),ae=d(G,"P",{"data-svelte-h":!0}),m(ae)!=="svelte-477g5w"&&(ae.innerHTML=io),Ue=r(G),re=d(G,"P",{"data-svelte-h":!0}),m(re)!=="svelte-1ek1ss9"&&(re.innerHTML=lo),Je=r(G),g(I.$$.fragment,G),G.forEach(t),ze=r(e),g(X.$$.fragment,e),Le=r(e),y=d(e,"DIV",{class:!0});var x=ne(y);g(Q.$$.fragment,x),Ee=r(x),ie=d(x,"P",{"data-svelte-h":!0}),m(ie)!=="svelte-1ycoqi1"&&(ie.textContent=co),Ne=r(x),le=d(x,"P",{"data-svelte-h":!0}),m(le)!=="svelte-q52n56"&&(le.innerHTML=mo),Re=r(x),de=d(x,"P",{"data-svelte-h":!0}),m(de)!=="svelte-hswkmf"&&(de.innerHTML=po),Be=r(x),L=d(x,"DIV",{class:!0});var F=ne(L);g(Y.$$.fragment,F),Ze=r(F),ce=d(F,"P",{"data-svelte-h":!0}),m(ce)!=="svelte-r2634t"&&(ce.innerHTML=ho),Oe=r(F),g(q.$$.fragment,F),F.forEach(t),x.forEach(t),Fe=r(e),g(K.$$.fragment,e),Pe=r(e),T=d(e,"DIV",{class:!0});var k=ne(T);g(ee.$$.fragment,k),De=r(k),me=d(k,"P",{"data-svelte-h":!0}),m(me)!=="svelte-1ux6hb8"&&(me.textContent=uo),Ae=r(k),pe=d(k,"P",{"data-svelte-h":!0}),m(pe)!=="svelte-q52n56"&&(pe.innerHTML=fo),Ve=r(k),he=d(k,"P",{"data-svelte-h":!0}),m(he)!=="svelte-hswkmf"&&(he.innerHTML=go),Se=r(k),C=d(k,"DIV",{class:!0});var H=ne(C);g(oe.$$.fragment,H),Xe=r(H),ue=d(H,"P",{"data-svelte-h":!0}),m(ue)!=="svelte-1wa8n6t"&&(ue.innerHTML=_o),Qe=r(H),g(W.$$.fragment,H),Ye=r(H),g(j.$$.fragment,H),H.forEach(t),k.forEach(t),Ie=r(e),g(te.$$.fragment,e),qe=r(e),fe=d(e,"P",{}),ne(fe).forEach(t),this.h()},h(){se(n,"name","hf:doc:metadata"),se(n,"content",Po),se(w,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),se(L,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),se(y,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),se(C,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),se(T,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,o){c(document.head,n),s(e,p,o),s(e,i,o),s(e,h,o),s(e,$,o),s(e,P,o),_(U,e,o),s(e,be,o),_(J,e,o),s(e,Me,o),s(e,E,o),s(e,ve,o),s(e,N,o),s(e,ye,o),s(e,R,o),s(e,Te,o),s(e,B,o),s(e,we,o),s(e,Z,o),s(e,xe,o),s(e,O,o),s(e,ke,o),s(e,D,o),s(e,$e,o),s(e,A,o),s(e,Ce,o),_(V,e,o),s(e,Ge,o),s(e,w,o),_(S,w,null),c(w,He),c(w,ae),c(w,Ue),c(w,re),c(w,Je),_(I,w,null),s(e,ze,o),_(X,e,o),s(e,Le,o),s(e,y,o),_(Q,y,null),c(y,Ee),c(y,ie),c(y,Ne),c(y,le),c(y,Re),c(y,de),c(y,Be),c(y,L),_(Y,L,null),c(L,Ze),c(L,ce),c(L,Oe),_(q,L,null),s(e,Fe,o),_(K,e,o),s(e,Pe,o),s(e,T,o),_(ee,T,null),c(T,De),c(T,me),c(T,Ae),c(T,pe),c(T,Ve),c(T,he),c(T,Se),c(T,C),_(oe,C,null),c(C,Xe),c(C,ue),c(C,Qe),_(W,C,null),c(C,Ye),_(j,C,null),s(e,Ie,o),_(te,e,o),s(e,qe,o),s(e,fe,o),We=!0},p(e,[o]){const G={};o&2&&(G.$$scope={dirty:o,ctx:e}),I.$set(G);const x={};o&2&&(x.$$scope={dirty:o,ctx:e}),q.$set(x);const F={};o&2&&(F.$$scope={dirty:o,ctx:e}),W.$set(F);const k={};o&2&&(k.$$scope={dirty:o,ctx:e}),j.$set(k)},i(e){We||(b(U.$$.fragment,e),b(J.$$.fragment,e),b(V.$$.fragment,e),b(S.$$.fragment,e),b(I.$$.fragment,e),b(X.$$.fragment,e),b(Q.$$.fragment,e),b(Y.$$.fragment,e),b(q.$$.fragment,e),b(K.$$.fragment,e),b(ee.$$.fragment,e),b(oe.$$.fragment,e),b(W.$$.fragment,e),b(j.$$.fragment,e),b(te.$$.fragment,e),We=!0)},o(e){M(U.$$.fragment,e),M(J.$$.fragment,e),M(V.$$.fragment,e),M(S.$$.fragment,e),M(I.$$.fragment,e),M(X.$$.fragment,e),M(Q.$$.fragment,e),M(Y.$$.fragment,e),M(q.$$.fragment,e),M(K.$$.fragment,e),M(ee.$$.fragment,e),M(oe.$$.fragment,e),M(W.$$.fragment,e),M(j.$$.fragment,e),M(te.$$.fragment,e),We=!1},d(e){e&&(t(p),t(i),t(h),t($),t(P),t(be),t(Me),t(E),t(ve),t(N),t(ye),t(R),t(Te),t(B),t(we),t(Z),t(xe),t(O),t(ke),t(D),t($e),t(A),t(Ce),t(Ge),t(w),t(ze),t(Le),t(y),t(Fe),t(Pe),t(T),t(Ie),t(qe),t(fe)),t(n),v(U,e),v(J,e),v(V,e),v(S),v(I),v(X,e),v(Q),v(Y),v(q),v(K,e),v(ee),v(oe),v(W),v(j),v(te,e)}}}const Po='{"title":"Glm4Moe","local":"glm4moe","sections":[{"title":"Overview","local":"overview","sections":[],"depth":2},{"title":"Glm4MoeConfig","local":"transformers.Glm4MoeConfig","sections":[],"depth":2},{"title":"Glm4MoeModel","local":"transformers.Glm4MoeModel","sections":[],"depth":2},{"title":"Glm4MoeForCausalLM","local":"transformers.Glm4MoeForCausalLM","sections":[],"depth":2}],"depth":1}';function Io(z){return To(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class No extends wo{constructor(n){super(),xo(this,n,Io,Fo,yo,{})}}export{No as component};
