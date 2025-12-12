import{s as wo,o as $o,n as Te}from"../chunks/scheduler.18a86fab.js";import{S as Mo,i as Co,g as c,s as a,r as p,A as xo,h as l,f as n,c as r,j as W,x as v,u as m,k as P,y as d,a as i,v as h,d as u,t as f,w as g}from"../chunks/index.98837b22.js";import{T as ao}from"../chunks/Tip.77304350.js";import{D as de}from"../chunks/Docstring.a1ef7999.js";import{C as To}from"../chunks/CodeBlock.8d0c2e8a.js";import{E as yo}from"../chunks/ExampleCodeBlock.8c3ee1f9.js";import{H as ce,E as Do}from"../chunks/getInferenceSnippets.06c2775f.js";function Vo(M){let t,_;return t=new To({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMERlZXBzZWVrVjJNb2RlbCUyQyUyMERlZXBzZWVrVjJDb25maWclMEElMjMlMjBJbml0aWFsaXppbmclMjBhJTIwRGVlcFNlZWstVjIlMjBzdHlsZSUyMGNvbmZpZ3VyYXRpb24lMEFjb25maWd1cmF0aW9uJTIwJTNEJTIwRGVlcHNlZWtWMkNvbmZpZygpJTBBJTIzJTIwQWNjZXNzaW5nJTIwdGhlJTIwbW9kZWwlMjBjb25maWd1cmF0aW9uJTBBbW9kZWwlMjAlM0QlMjBEZWVwc2Vla1YyTW9kZWwoY29uZmlndXJhdGlvbiklMEFwcmludChtb2RlbC5jb25maWcp",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> DeepseekV2Model, DeepseekV2Config
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a DeepSeek-V2 style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = DeepseekV2Config()
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Accessing the model configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = DeepseekV2Model(configuration)
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">print</span>(model.config)`,wrap:!1}}),{c(){p(t.$$.fragment)},l(s){m(t.$$.fragment,s)},m(s,k){h(t,s,k),_=!0},p:Te,i(s){_||(u(t.$$.fragment,s),_=!0)},o(s){f(t.$$.fragment,s),_=!1},d(s){g(t,s)}}}function zo(M){let t,_=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=_},l(s){t=l(s,"P",{"data-svelte-h":!0}),v(t)!=="svelte-fincs2"&&(t.innerHTML=_)},m(s,k){i(s,t,k)},p:Te,d(s){s&&n(t)}}}function qo(M){let t,_=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=_},l(s){t=l(s,"P",{"data-svelte-h":!0}),v(t)!=="svelte-fincs2"&&(t.innerHTML=_)},m(s,k){i(s,t,k)},p:Te,d(s){s&&n(t)}}}function Fo(M){let t,_="Example:",s,k,C;return k=new To({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBEZWVwc2Vla1YyRm9yQ2F1c2FsTE0lMEElMEFtb2RlbCUyMCUzRCUyMERlZXBzZWVrVjJGb3JDYXVzYWxMTS5mcm9tX3ByZXRyYWluZWQoJTIybWV0YS1kZWVwc2Vla192MiUyRkRlZXBzZWVrVjItMi03Yi1oZiUyMiklMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJtZXRhLWRlZXBzZWVrX3YyJTJGRGVlcHNlZWtWMi0yLTdiLWhmJTIyKSUwQSUwQXByb21wdCUyMCUzRCUyMCUyMkhleSUyQyUyMGFyZSUyMHlvdSUyMGNvbnNjaW91cyUzRiUyMENhbiUyMHlvdSUyMHRhbGslMjB0byUyMG1lJTNGJTIyJTBBaW5wdXRzJTIwJTNEJTIwdG9rZW5pemVyKHByb21wdCUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpJTBBJTBBJTIzJTIwR2VuZXJhdGUlMEFnZW5lcmF0ZV9pZHMlMjAlM0QlMjBtb2RlbC5nZW5lcmF0ZShpbnB1dHMuaW5wdXRfaWRzJTJDJTIwbWF4X2xlbmd0aCUzRDMwKSUwQXRva2VuaXplci5iYXRjaF9kZWNvZGUoZ2VuZXJhdGVfaWRzJTJDJTIwc2tpcF9zcGVjaWFsX3Rva2VucyUzRFRydWUlMkMlMjBjbGVhbl91cF90b2tlbml6YXRpb25fc3BhY2VzJTNERmFsc2UpJTVCMCU1RA==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, DeepseekV2ForCausalLM

<span class="hljs-meta">&gt;&gt;&gt; </span>model = DeepseekV2ForCausalLM.from_pretrained(<span class="hljs-string">&quot;meta-deepseek_v2/DeepseekV2-2-7b-hf&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;meta-deepseek_v2/DeepseekV2-2-7b-hf&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>prompt = <span class="hljs-string">&quot;Hey, are you conscious? Can you talk to me?&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(prompt, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Generate</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>generate_ids = model.generate(inputs.input_ids, max_length=<span class="hljs-number">30</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer.batch_decode(generate_ids, skip_special_tokens=<span class="hljs-literal">True</span>, clean_up_tokenization_spaces=<span class="hljs-literal">False</span>)[<span class="hljs-number">0</span>]
<span class="hljs-string">&quot;Hey, are you conscious? Can you talk to me?\\nI&#x27;m not conscious, but I can talk to you.&quot;</span>`,wrap:!1}}),{c(){t=c("p"),t.textContent=_,s=a(),p(k.$$.fragment)},l(b){t=l(b,"P",{"data-svelte-h":!0}),v(t)!=="svelte-11lpom8"&&(t.textContent=_),s=r(b),m(k.$$.fragment,b)},m(b,S){i(b,t,S),i(b,s,S),h(k,b,S),C=!0},p:Te,i(b){C||(u(k.$$.fragment,b),C=!0)},o(b){f(k.$$.fragment,b),C=!1},d(b){b&&(n(t),n(s)),g(k,b)}}}function Lo(M){let t,_=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=_},l(s){t=l(s,"P",{"data-svelte-h":!0}),v(t)!=="svelte-fincs2"&&(t.innerHTML=_)},m(s,k){i(s,t,k)},p:Te,d(s){s&&n(t)}}}function Io(M){let t,_,s,k,C,b="<em>This model was released on 2024-05-07 and added to Hugging Face Transformers on 2025-07-09.</em>",S,Z,we,U,$e,J,ro='The DeepSeek-V2 model was proposed in <a href="https://huggingface.co/papers/2405.04434" rel="nofollow">DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model</a> by DeepSeek-AI Team.',Me,B,io=`The abstract from the paper is the following:
We present DeepSeek-V2, a strong Mixture-of-Experts (MoE) language model characterized by economical training and efficient inference. It comprises 236B total parameters, of which 21B are activated for each token, and supports a context length of 128K tokens. DeepSeek-V2 adopts innovative architectures including Multi-head Latent Attention (MLA) and DeepSeekMoE. MLA guarantees efficient inference through significantly compressing the Key-Value (KV) cache into a latent vector, while DeepSeekMoE enables training strong models at an economical cost through sparse computation. Compared with DeepSeek 67B, DeepSeek-V2 achieves significantly stronger performance, and meanwhile saves 42.5% of training costs, reduces the KV cache by 93.3%, and boosts the maximum generation throughput to 5.76 times. We pretrain DeepSeek-V2 on a high-quality and multi-source corpus consisting of 8.1T tokens, and further perform Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL) to fully unlock its potential. Evaluation results show that, even with only 21B activated parameters, DeepSeek-V2 and its chat versions still achieve top-tier performance among open-source models.`,Ce,O,co=`This model was contributed by <a href="https://github.com/VladOS95-cyber" rel="nofollow">VladOS95-cyber</a>.
The original code can be found <a href="https://huggingface.co/deepseek-ai/DeepSeek-V2" rel="nofollow">here</a>.`,xe,A,De,G,lo="The model uses Multi-head Latent Attention (MLA) and DeepSeekMoE architectures for efficient inference and cost-effective training. It employs an auxiliary-loss-free strategy for load balancing and multi-token prediction training objective. The model can be used for various language tasks after being pre-trained on 14.8 trillion tokens and going through Supervised Fine-Tuning and Reinforcement Learning stages.",Ve,X,ze,D,Q,He,le,po=`This is the configuration class to store the configuration of a <a href="/docs/transformers/v4.56.2/en/model_doc/deepseek_v2#transformers.DeepseekV2Model">DeepseekV2Model</a>. It is used to instantiate a DeepSeek
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of DeepSeek-V2-Lite” <a href="https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite%22" rel="nofollow">deepseek-ai/DeepSeek-V2-Lite”</a>.
Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> and can be used to control the model outputs. Read the
documentation from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> for more information.`,Re,j,qe,Y,Fe,y,K,Ze,pe,mo="The bare Deepseek V2 Model outputting raw hidden-states without any specific head on top.",Ue,me,ho=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Je,he,uo=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Be,z,ee,Oe,ue,fo='The <a href="/docs/transformers/v4.56.2/en/model_doc/deepseek_v2#transformers.DeepseekV2Model">DeepseekV2Model</a> forward method, overrides the <code>__call__</code> special method.',Ae,E,Le,oe,Ie,T,te,Ge,fe,go="The Deepseek V2 Model for causal language modeling.",Xe,ge,_o=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Qe,_e,ko=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Ye,x,ne,Ke,ke,vo='The <a href="/docs/transformers/v4.56.2/en/model_doc/deepseek_v2#transformers.DeepseekV2ForCausalLM">DeepseekV2ForCausalLM</a> forward method, overrides the <code>__call__</code> special method.',eo,N,oo,H,We,se,Pe,F,ae,to,q,re,no,ve,bo="The <code>GenericForSequenceClassification</code> forward method, overrides the <code>__call__</code> special method.",so,R,Se,ie,je,ye,Ee;return Z=new ce({props:{title:"DeepSeek-V2",local:"deepseek-v2",headingTag:"h1"}}),U=new ce({props:{title:"Overview",local:"overview",headingTag:"h2"}}),A=new ce({props:{title:"Usage tips",local:"usage-tips",headingTag:"h3"}}),X=new ce({props:{title:"DeepseekV2Config",local:"transformers.DeepseekV2Config",headingTag:"h2"}}),Q=new de({props:{name:"class transformers.DeepseekV2Config",anchor:"transformers.DeepseekV2Config",parameters:[{name:"vocab_size",val:" = 32000"},{name:"hidden_size",val:" = 4096"},{name:"intermediate_size",val:" = 11008"},{name:"num_hidden_layers",val:" = 32"},{name:"num_attention_heads",val:" = 32"},{name:"num_key_value_heads",val:" = None"},{name:"hidden_act",val:" = 'silu'"},{name:"max_position_embeddings",val:" = 2048"},{name:"initializer_range",val:" = 0.02"},{name:"rms_norm_eps",val:" = 1e-06"},{name:"use_cache",val:" = True"},{name:"pad_token_id",val:" = None"},{name:"bos_token_id",val:" = 1"},{name:"eos_token_id",val:" = 2"},{name:"tie_word_embeddings",val:" = False"},{name:"rope_theta",val:" = 10000.0"},{name:"rope_scaling",val:" = None"},{name:"attention_bias",val:" = False"},{name:"attention_dropout",val:" = 0.0"},{name:"mlp_bias",val:" = False"},{name:"aux_loss_alpha",val:" = 0.001"},{name:"first_k_dense_replace",val:" = 0"},{name:"kv_lora_rank",val:" = 512"},{name:"q_lora_rank",val:" = 1536"},{name:"n_group",val:" = None"},{name:"n_routed_experts",val:" = 64"},{name:"n_shared_experts",val:" = 2"},{name:"qk_nope_head_dim",val:" = 128"},{name:"qk_rope_head_dim",val:" = 64"},{name:"routed_scaling_factor",val:" = 1.0"},{name:"seq_aux",val:" = True"},{name:"topk_group",val:" = None"},{name:"topk_method",val:" = 'greedy'"},{name:"v_head_dim",val:" = 128"},{name:"num_experts_per_tok",val:" = None"},{name:"norm_topk_prob",val:" = False"},{name:"moe_intermediate_size",val:" = 1407"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.DeepseekV2Config.vocab_size",description:`<strong>vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 32000) &#x2014;
Vocabulary size of the DeepSeek model. Defines the number of different tokens that can be represented by the
<code>input_ids</code> passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/deepseek_v2#transformers.DeepseekV2Model">DeepseekV2Model</a>.`,name:"vocab_size"},{anchor:"transformers.DeepseekV2Config.hidden_size",description:`<strong>hidden_size</strong> (<code>int</code>, <em>optional</em>, defaults to 4096) &#x2014;
Dimension of the hidden representations.`,name:"hidden_size"},{anchor:"transformers.DeepseekV2Config.intermediate_size",description:`<strong>intermediate_size</strong> (<code>int</code>, <em>optional</em>, defaults to 11008) &#x2014;
Dimension of the MLP representations.`,name:"intermediate_size"},{anchor:"transformers.DeepseekV2Config.num_hidden_layers",description:`<strong>num_hidden_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 32) &#x2014;
Number of hidden layers in the Transformer decoder.`,name:"num_hidden_layers"},{anchor:"transformers.DeepseekV2Config.num_attention_heads",description:`<strong>num_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 32) &#x2014;
Number of attention heads for each attention layer in the Transformer decoder.`,name:"num_attention_heads"},{anchor:"transformers.DeepseekV2Config.num_key_value_heads",description:`<strong>num_key_value_heads</strong> (<code>int</code>, <em>optional</em>) &#x2014;
The number of key-value heads used to implement Grouped Query Attention (GQA). If
<code>num_key_value_heads=num_attention_heads</code>, the model will use Multi-Head Attention (MHA). If
<code>num_key_value_heads=1</code>, the model will use Multi-Query Attention (MQA). Otherwise, GQA is used.`,name:"num_key_value_heads"},{anchor:"transformers.DeepseekV2Config.hidden_act",description:`<strong>hidden_act</strong> (<code>str</code> or <code>function</code>, <em>optional</em>, defaults to <code>&quot;silu&quot;</code>) &#x2014;
The non-linear activation function (function or string) in the decoder.`,name:"hidden_act"},{anchor:"transformers.DeepseekV2Config.max_position_embeddings",description:`<strong>max_position_embeddings</strong> (<code>int</code>, <em>optional</em>, defaults to 2048) &#x2014;
The maximum sequence length that this model might ever be used with.`,name:"max_position_embeddings"},{anchor:"transformers.DeepseekV2Config.initializer_range",description:`<strong>initializer_range</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated normal initializer for initializing all weight matrices.`,name:"initializer_range"},{anchor:"transformers.DeepseekV2Config.rms_norm_eps",description:`<strong>rms_norm_eps</strong> (<code>float</code>, <em>optional</em>, defaults to 1e-06) &#x2014;
The epsilon value used by the RMS normalization layers.`,name:"rms_norm_eps"},{anchor:"transformers.DeepseekV2Config.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not the model should return the last key/value attentions (useful for inference optimization).`,name:"use_cache"},{anchor:"transformers.DeepseekV2Config.pad_token_id",description:`<strong>pad_token_id</strong> (<code>int</code>, <em>optional</em>) &#x2014;
Padding token ID.`,name:"pad_token_id"},{anchor:"transformers.DeepseekV2Config.bos_token_id",description:`<strong>bos_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 1) &#x2014;
Beginning-of-sequence token ID.`,name:"bos_token_id"},{anchor:"transformers.DeepseekV2Config.eos_token_id",description:`<strong>eos_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 2) &#x2014;
End-of-sequence token ID.`,name:"eos_token_id"},{anchor:"transformers.DeepseekV2Config.tie_word_embeddings",description:`<strong>tie_word_embeddings</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether to tie input and output embeddings.`,name:"tie_word_embeddings"},{anchor:"transformers.DeepseekV2Config.rope_theta",description:`<strong>rope_theta</strong> (<code>float</code>, <em>optional</em>, defaults to 10000.0) &#x2014;
The base period of the Rotary Position Embeddings (RoPE).`,name:"rope_theta"},{anchor:"transformers.DeepseekV2Config.rope_scaling",description:`<strong>rope_scaling</strong> (<code>Dict</code>, <em>optional</em>) &#x2014;
Configuration for scaling RoPE embeddings. Supports <code>linear</code> and <code>dynamic</code> scaling strategies.`,name:"rope_scaling"},{anchor:"transformers.DeepseekV2Config.attention_bias",description:`<strong>attention_bias</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether to use a bias in the query, key, value, and output projection layers during self-attention.`,name:"attention_bias"},{anchor:"transformers.DeepseekV2Config.attention_dropout",description:`<strong>attention_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The dropout probability applied to attention weights.`,name:"attention_dropout"},{anchor:"transformers.DeepseekV2Config.mlp_bias",description:`<strong>mlp_bias</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether to use a bias term in the MLP layers.`,name:"mlp_bias"},{anchor:"transformers.DeepseekV2Config.aux_loss_alpha",description:`<strong>aux_loss_alpha</strong> (<code>float</code>, <em>optional</em>, defaults to 0.001) &#x2014;
Weight coefficient for auxiliary loss in Mixture of Experts (MoE) models.`,name:"aux_loss_alpha"},{anchor:"transformers.DeepseekV2Config.first_k_dense_replace",description:`<strong>first_k_dense_replace</strong> (<code>int</code>, <em>optional</em>, defaults to 0) &#x2014;
Number of dense layers in the shallow layers before switching to MoE layers.`,name:"first_k_dense_replace"},{anchor:"transformers.DeepseekV2Config.kv_lora_rank",description:`<strong>kv_lora_rank</strong> (<code>int</code>, <em>optional</em>, defaults to 512) &#x2014;
Rank of the LoRA decomposition for key-value projections.`,name:"kv_lora_rank"},{anchor:"transformers.DeepseekV2Config.q_lora_rank",description:`<strong>q_lora_rank</strong> (<code>int</code>, <em>optional</em>, defaults to 1536) &#x2014;
Rank of the LoRA decomposition for query projections.
Specifically, it determines the dimensionality to which the query (q) vectors are compressed before being expanded back to their original size.
It reduces computational overhead while maintaining model performance.`,name:"q_lora_rank"},{anchor:"transformers.DeepseekV2Config.n_group",description:`<strong>n_group</strong> (<code>int</code>, <em>optional</em>) &#x2014;
Number of groups for routed experts.`,name:"n_group"},{anchor:"transformers.DeepseekV2Config.n_routed_experts",description:`<strong>n_routed_experts</strong> (<code>int</code>, <em>optional</em>, defaults to 64) &#x2014;
Number of routed experts (None indicates a dense model).`,name:"n_routed_experts"},{anchor:"transformers.DeepseekV2Config.n_shared_experts",description:`<strong>n_shared_experts</strong> (<code>int</code>, <em>optional</em>, defaults to 2) &#x2014;
Number of shared experts (None indicates a dense model).`,name:"n_shared_experts"},{anchor:"transformers.DeepseekV2Config.qk_nope_head_dim",description:`<strong>qk_nope_head_dim</strong> (<code>int</code>, <em>optional</em>, defaults to 128) &#x2014;
The head dimension for the QK (query-key) projections when using NOPE (Neural Operator Position Encoding).`,name:"qk_nope_head_dim"},{anchor:"transformers.DeepseekV2Config.qk_rope_head_dim",description:`<strong>qk_rope_head_dim</strong> (<code>int</code>, <em>optional</em>, defaults to 64) &#x2014;
The head dimension for QK projections when using RoPE.`,name:"qk_rope_head_dim"},{anchor:"transformers.DeepseekV2Config.routed_scaling_factor",description:`<strong>routed_scaling_factor</strong> (<code>float</code>, <em>optional</em>, defaults to 1.0) &#x2014;
Scaling factor for routed experts in MoE models.`,name:"routed_scaling_factor"},{anchor:"transformers.DeepseekV2Config.seq_aux",description:`<strong>seq_aux</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether to compute the auxiliary loss for each individual sequence.`,name:"seq_aux"},{anchor:"transformers.DeepseekV2Config.topk_group",description:`<strong>topk_group</strong> (<code>int</code>, <em>optional</em>) &#x2014;
Number of selected groups per token for expert selection.`,name:"topk_group"},{anchor:"transformers.DeepseekV2Config.topk_method",description:`<strong>topk_method</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;greedy&quot;</code>) &#x2014;
The method used for selecting top-k experts in the routed gate mechanism.`,name:"topk_method"},{anchor:"transformers.DeepseekV2Config.v_head_dim",description:`<strong>v_head_dim</strong> (<code>int</code>, <em>optional</em>, defaults to 128) &#x2014;
The dimension of value projections in the attention layers.`,name:"v_head_dim"},{anchor:"transformers.DeepseekV2Config.num_experts_per_tok",description:`<strong>num_experts_per_tok</strong> (<code>int</code>, <em>optional</em>) &#x2014;
The number of experts selected per token. If <code>None</code>, the model behaves as a dense Transformer.`,name:"num_experts_per_tok"},{anchor:"transformers.DeepseekV2Config.norm_topk_prob",description:`<strong>norm_topk_prob</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether to normalize the probability distribution over top-k selected experts.`,name:"norm_topk_prob"},{anchor:"transformers.DeepseekV2Config.moe_intermediate_size",description:`<strong>moe_intermediate_size</strong> (<code>int</code>, <em>optional</em>, defaults to 1407) &#x2014;
Dimension of the MoE (Mixture of Experts) representations.`,name:"moe_intermediate_size"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deepseek_v2/configuration_deepseek_v2.py#L27"}}),j=new yo({props:{anchor:"transformers.DeepseekV2Config.example",$$slots:{default:[Vo]},$$scope:{ctx:M}}}),Y=new ce({props:{title:"DeepseekV2Model",local:"transformers.DeepseekV2Model",headingTag:"h2"}}),K=new de({props:{name:"class transformers.DeepseekV2Model",anchor:"transformers.DeepseekV2Model",parameters:[{name:"config",val:": DeepseekV2Config"}],parametersDescription:[{anchor:"transformers.DeepseekV2Model.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/deepseek_v2#transformers.DeepseekV2Config">DeepseekV2Config</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deepseek_v2/modeling_deepseek_v2.py#L477"}}),ee=new de({props:{name:"forward",anchor:"transformers.DeepseekV2Model.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.DeepseekV2Model.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.DeepseekV2Model.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.DeepseekV2Model.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.DeepseekV2Model.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.DeepseekV2Model.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.DeepseekV2Model.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"},{anchor:"transformers.DeepseekV2Model.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deepseek_v2/modeling_deepseek_v2.py#L494",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast"
>transformers.modeling_outputs.BaseModelOutputWithPast</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/deepseek_v2#transformers.DeepseekV2Config"
>DeepseekV2Config</a>) and inputs.</p>
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
`}}),E=new ao({props:{$$slots:{default:[zo]},$$scope:{ctx:M}}}),oe=new ce({props:{title:"DeepseekV2ForCausalLM",local:"transformers.DeepseekV2ForCausalLM",headingTag:"h2"}}),te=new de({props:{name:"class transformers.DeepseekV2ForCausalLM",anchor:"transformers.DeepseekV2ForCausalLM",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.DeepseekV2ForCausalLM.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/deepseek_v2#transformers.DeepseekV2ForCausalLM">DeepseekV2ForCausalLM</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deepseek_v2/modeling_deepseek_v2.py#L556"}}),ne=new de({props:{name:"forward",anchor:"transformers.DeepseekV2ForCausalLM.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"},{name:"logits_to_keep",val:": typing.Union[int, torch.Tensor] = 0"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.DeepseekV2ForCausalLM.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.DeepseekV2ForCausalLM.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.DeepseekV2ForCausalLM.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.DeepseekV2ForCausalLM.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.DeepseekV2ForCausalLM.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.DeepseekV2ForCausalLM.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.DeepseekV2ForCausalLM.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.DeepseekV2ForCausalLM.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"},{anchor:"transformers.DeepseekV2ForCausalLM.forward.logits_to_keep",description:`<strong>logits_to_keep</strong> (<code>Union[int, torch.Tensor]</code>, defaults to <code>0</code>) &#x2014;
If an <code>int</code>, compute logits for the last <code>logits_to_keep</code> tokens. If <code>0</code>, calculate logits for all
<code>input_ids</code> (special case). Only last token logits are needed for generation, and calculating them only for that
token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
If a <code>torch.Tensor</code>, must be 1D corresponding to the indices to keep in the sequence length dimension.
This is useful when using packed tensor format (single dimension for batch and sequence length).`,name:"logits_to_keep"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deepseek_v2/modeling_deepseek_v2.py#L570",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast"
>transformers.modeling_outputs.CausalLMOutputWithPast</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/deepseek_v2#transformers.DeepseekV2Config"
>DeepseekV2Config</a>) and inputs.</p>
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
`}}),N=new ao({props:{$$slots:{default:[qo]},$$scope:{ctx:M}}}),H=new yo({props:{anchor:"transformers.DeepseekV2ForCausalLM.forward.example",$$slots:{default:[Fo]},$$scope:{ctx:M}}}),se=new ce({props:{title:"DeepseekV2ForSequenceClassification",local:"transformers.DeepseekV2ForSequenceClassification",headingTag:"h2"}}),ae=new de({props:{name:"class transformers.DeepseekV2ForSequenceClassification",anchor:"transformers.DeepseekV2ForSequenceClassification",parameters:[{name:"config",val:""}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deepseek_v2/modeling_deepseek_v2.py#L631"}}),re=new de({props:{name:"forward",anchor:"transformers.DeepseekV2ForSequenceClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.DeepseekV2ForSequenceClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.DeepseekV2ForSequenceClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.DeepseekV2ForSequenceClassification.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.DeepseekV2ForSequenceClassification.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.DeepseekV2ForSequenceClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.DeepseekV2ForSequenceClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.DeepseekV2ForSequenceClassification.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
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
`}}),R=new ao({props:{$$slots:{default:[Lo]},$$scope:{ctx:M}}}),ie=new Do({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/deepseek_v2.md"}}),{c(){t=c("meta"),_=a(),s=c("p"),k=a(),C=c("p"),C.innerHTML=b,S=a(),p(Z.$$.fragment),we=a(),p(U.$$.fragment),$e=a(),J=c("p"),J.innerHTML=ro,Me=a(),B=c("p"),B.textContent=io,Ce=a(),O=c("p"),O.innerHTML=co,xe=a(),p(A.$$.fragment),De=a(),G=c("p"),G.textContent=lo,Ve=a(),p(X.$$.fragment),ze=a(),D=c("div"),p(Q.$$.fragment),He=a(),le=c("p"),le.innerHTML=po,Re=a(),p(j.$$.fragment),qe=a(),p(Y.$$.fragment),Fe=a(),y=c("div"),p(K.$$.fragment),Ze=a(),pe=c("p"),pe.textContent=mo,Ue=a(),me=c("p"),me.innerHTML=ho,Je=a(),he=c("p"),he.innerHTML=uo,Be=a(),z=c("div"),p(ee.$$.fragment),Oe=a(),ue=c("p"),ue.innerHTML=fo,Ae=a(),p(E.$$.fragment),Le=a(),p(oe.$$.fragment),Ie=a(),T=c("div"),p(te.$$.fragment),Ge=a(),fe=c("p"),fe.textContent=go,Xe=a(),ge=c("p"),ge.innerHTML=_o,Qe=a(),_e=c("p"),_e.innerHTML=ko,Ye=a(),x=c("div"),p(ne.$$.fragment),Ke=a(),ke=c("p"),ke.innerHTML=vo,eo=a(),p(N.$$.fragment),oo=a(),p(H.$$.fragment),We=a(),p(se.$$.fragment),Pe=a(),F=c("div"),p(ae.$$.fragment),to=a(),q=c("div"),p(re.$$.fragment),no=a(),ve=c("p"),ve.innerHTML=bo,so=a(),p(R.$$.fragment),Se=a(),p(ie.$$.fragment),je=a(),ye=c("p"),this.h()},l(e){const o=xo("svelte-u9bgzb",document.head);t=l(o,"META",{name:!0,content:!0}),o.forEach(n),_=r(e),s=l(e,"P",{}),W(s).forEach(n),k=r(e),C=l(e,"P",{"data-svelte-h":!0}),v(C)!=="svelte-knw357"&&(C.innerHTML=b),S=r(e),m(Z.$$.fragment,e),we=r(e),m(U.$$.fragment,e),$e=r(e),J=l(e,"P",{"data-svelte-h":!0}),v(J)!=="svelte-aufh43"&&(J.innerHTML=ro),Me=r(e),B=l(e,"P",{"data-svelte-h":!0}),v(B)!=="svelte-1fs4lne"&&(B.textContent=io),Ce=r(e),O=l(e,"P",{"data-svelte-h":!0}),v(O)!=="svelte-a7cxt3"&&(O.innerHTML=co),xe=r(e),m(A.$$.fragment,e),De=r(e),G=l(e,"P",{"data-svelte-h":!0}),v(G)!=="svelte-aoi9cz"&&(G.textContent=lo),Ve=r(e),m(X.$$.fragment,e),ze=r(e),D=l(e,"DIV",{class:!0});var L=W(D);m(Q.$$.fragment,L),He=r(L),le=l(L,"P",{"data-svelte-h":!0}),v(le)!=="svelte-15216fy"&&(le.innerHTML=po),Re=r(L),m(j.$$.fragment,L),L.forEach(n),qe=r(e),m(Y.$$.fragment,e),Fe=r(e),y=l(e,"DIV",{class:!0});var w=W(y);m(K.$$.fragment,w),Ze=r(w),pe=l(w,"P",{"data-svelte-h":!0}),v(pe)!=="svelte-1tqc0yw"&&(pe.textContent=mo),Ue=r(w),me=l(w,"P",{"data-svelte-h":!0}),v(me)!=="svelte-q52n56"&&(me.innerHTML=ho),Je=r(w),he=l(w,"P",{"data-svelte-h":!0}),v(he)!=="svelte-hswkmf"&&(he.innerHTML=uo),Be=r(w),z=l(w,"DIV",{class:!0});var I=W(z);m(ee.$$.fragment,I),Oe=r(I),ue=l(I,"P",{"data-svelte-h":!0}),v(ue)!=="svelte-803pym"&&(ue.innerHTML=fo),Ae=r(I),m(E.$$.fragment,I),I.forEach(n),w.forEach(n),Le=r(e),m(oe.$$.fragment,e),Ie=r(e),T=l(e,"DIV",{class:!0});var $=W(T);m(te.$$.fragment,$),Ge=r($),fe=l($,"P",{"data-svelte-h":!0}),v(fe)!=="svelte-1ej4ohx"&&(fe.textContent=go),Xe=r($),ge=l($,"P",{"data-svelte-h":!0}),v(ge)!=="svelte-q52n56"&&(ge.innerHTML=_o),Qe=r($),_e=l($,"P",{"data-svelte-h":!0}),v(_e)!=="svelte-hswkmf"&&(_e.innerHTML=ko),Ye=r($),x=l($,"DIV",{class:!0});var V=W(x);m(ne.$$.fragment,V),Ke=r(V),ke=l(V,"P",{"data-svelte-h":!0}),v(ke)!=="svelte-maut42"&&(ke.innerHTML=vo),eo=r(V),m(N.$$.fragment,V),oo=r(V),m(H.$$.fragment,V),V.forEach(n),$.forEach(n),We=r(e),m(se.$$.fragment,e),Pe=r(e),F=l(e,"DIV",{class:!0});var Ne=W(F);m(ae.$$.fragment,Ne),to=r(Ne),q=l(Ne,"DIV",{class:!0});var be=W(q);m(re.$$.fragment,be),no=r(be),ve=l(be,"P",{"data-svelte-h":!0}),v(ve)!=="svelte-1sal4ui"&&(ve.innerHTML=bo),so=r(be),m(R.$$.fragment,be),be.forEach(n),Ne.forEach(n),Se=r(e),m(ie.$$.fragment,e),je=r(e),ye=l(e,"P",{}),W(ye).forEach(n),this.h()},h(){P(t,"name","hf:doc:metadata"),P(t,"content",Wo),P(D,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),P(z,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),P(y,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),P(x,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),P(T,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),P(q,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),P(F,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,o){d(document.head,t),i(e,_,o),i(e,s,o),i(e,k,o),i(e,C,o),i(e,S,o),h(Z,e,o),i(e,we,o),h(U,e,o),i(e,$e,o),i(e,J,o),i(e,Me,o),i(e,B,o),i(e,Ce,o),i(e,O,o),i(e,xe,o),h(A,e,o),i(e,De,o),i(e,G,o),i(e,Ve,o),h(X,e,o),i(e,ze,o),i(e,D,o),h(Q,D,null),d(D,He),d(D,le),d(D,Re),h(j,D,null),i(e,qe,o),h(Y,e,o),i(e,Fe,o),i(e,y,o),h(K,y,null),d(y,Ze),d(y,pe),d(y,Ue),d(y,me),d(y,Je),d(y,he),d(y,Be),d(y,z),h(ee,z,null),d(z,Oe),d(z,ue),d(z,Ae),h(E,z,null),i(e,Le,o),h(oe,e,o),i(e,Ie,o),i(e,T,o),h(te,T,null),d(T,Ge),d(T,fe),d(T,Xe),d(T,ge),d(T,Qe),d(T,_e),d(T,Ye),d(T,x),h(ne,x,null),d(x,Ke),d(x,ke),d(x,eo),h(N,x,null),d(x,oo),h(H,x,null),i(e,We,o),h(se,e,o),i(e,Pe,o),i(e,F,o),h(ae,F,null),d(F,to),d(F,q),h(re,q,null),d(q,no),d(q,ve),d(q,so),h(R,q,null),i(e,Se,o),h(ie,e,o),i(e,je,o),i(e,ye,o),Ee=!0},p(e,[o]){const L={};o&2&&(L.$$scope={dirty:o,ctx:e}),j.$set(L);const w={};o&2&&(w.$$scope={dirty:o,ctx:e}),E.$set(w);const I={};o&2&&(I.$$scope={dirty:o,ctx:e}),N.$set(I);const $={};o&2&&($.$$scope={dirty:o,ctx:e}),H.$set($);const V={};o&2&&(V.$$scope={dirty:o,ctx:e}),R.$set(V)},i(e){Ee||(u(Z.$$.fragment,e),u(U.$$.fragment,e),u(A.$$.fragment,e),u(X.$$.fragment,e),u(Q.$$.fragment,e),u(j.$$.fragment,e),u(Y.$$.fragment,e),u(K.$$.fragment,e),u(ee.$$.fragment,e),u(E.$$.fragment,e),u(oe.$$.fragment,e),u(te.$$.fragment,e),u(ne.$$.fragment,e),u(N.$$.fragment,e),u(H.$$.fragment,e),u(se.$$.fragment,e),u(ae.$$.fragment,e),u(re.$$.fragment,e),u(R.$$.fragment,e),u(ie.$$.fragment,e),Ee=!0)},o(e){f(Z.$$.fragment,e),f(U.$$.fragment,e),f(A.$$.fragment,e),f(X.$$.fragment,e),f(Q.$$.fragment,e),f(j.$$.fragment,e),f(Y.$$.fragment,e),f(K.$$.fragment,e),f(ee.$$.fragment,e),f(E.$$.fragment,e),f(oe.$$.fragment,e),f(te.$$.fragment,e),f(ne.$$.fragment,e),f(N.$$.fragment,e),f(H.$$.fragment,e),f(se.$$.fragment,e),f(ae.$$.fragment,e),f(re.$$.fragment,e),f(R.$$.fragment,e),f(ie.$$.fragment,e),Ee=!1},d(e){e&&(n(_),n(s),n(k),n(C),n(S),n(we),n($e),n(J),n(Me),n(B),n(Ce),n(O),n(xe),n(De),n(G),n(Ve),n(ze),n(D),n(qe),n(Fe),n(y),n(Le),n(Ie),n(T),n(We),n(Pe),n(F),n(Se),n(je),n(ye)),n(t),g(Z,e),g(U,e),g(A,e),g(X,e),g(Q),g(j),g(Y,e),g(K),g(ee),g(E),g(oe,e),g(te),g(ne),g(N),g(H),g(se,e),g(ae),g(re),g(R),g(ie,e)}}}const Wo='{"title":"DeepSeek-V2","local":"deepseek-v2","sections":[{"title":"Overview","local":"overview","sections":[{"title":"Usage tips","local":"usage-tips","sections":[],"depth":3}],"depth":2},{"title":"DeepseekV2Config","local":"transformers.DeepseekV2Config","sections":[],"depth":2},{"title":"DeepseekV2Model","local":"transformers.DeepseekV2Model","sections":[],"depth":2},{"title":"DeepseekV2ForCausalLM","local":"transformers.DeepseekV2ForCausalLM","sections":[],"depth":2},{"title":"DeepseekV2ForSequenceClassification","local":"transformers.DeepseekV2ForSequenceClassification","sections":[],"depth":2}],"depth":1}';function Po(M){return $o(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class Uo extends Mo{constructor(t){super(),Co(this,t,Po,Io,wo,{})}}export{Uo as component};
