import{s as vn,o as bn,n as Pe}from"../chunks/scheduler.18a86fab.js";import{S as yn,i as kn,g as c,s,r as u,A as Tn,h as l,f as t,c as a,j as q,u as p,x as b,k as I,y as r,a as i,v as h,d as m,t as f,w as g}from"../chunks/index.98837b22.js";import{T as tn}from"../chunks/Tip.77304350.js";import{D as re}from"../chunks/Docstring.a1ef7999.js";import{C as wn}from"../chunks/CodeBlock.8d0c2e8a.js";import{E as $n}from"../chunks/ExampleCodeBlock.8c3ee1f9.js";import{H as S,E as Cn}from"../chunks/getInferenceSnippets.06c2775f.js";function Mn(D){let o,y=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){o=c("p"),o.innerHTML=y},l(d){o=l(d,"P",{"data-svelte-h":!0}),b(o)!=="svelte-fincs2"&&(o.innerHTML=y)},m(d,v){i(d,o,v)},p:Pe,d(d){d&&t(o)}}}function Hn(D){let o,y=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){o=c("p"),o.innerHTML=y},l(d){o=l(d,"P",{"data-svelte-h":!0}),b(o)!=="svelte-fincs2"&&(o.innerHTML=y)},m(d,v){i(d,o,v)},p:Pe,d(d){d&&t(o)}}}function xn(D){let o,y="Example:",d,v,C;return v=new wn({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBIdW5ZdWFuRGVuc2VWMUZvckNhdXNhbExNJTBBJTBBbW9kZWwlMjAlM0QlMjBIdW5ZdWFuRGVuc2VWMUZvckNhdXNhbExNLmZyb21fcHJldHJhaW5lZCglMjJtZXRhLWh1bnl1YW5fdjFfZGVuc2UlMkZIdW5ZdWFuRGVuc2VWMS0yLTdiLWhmJTIyKSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMm1ldGEtaHVueXVhbl92MV9kZW5zZSUyRkh1bll1YW5EZW5zZVYxLTItN2ItaGYlMjIpJTBBJTBBcHJvbXB0JTIwJTNEJTIwJTIySGV5JTJDJTIwYXJlJTIweW91JTIwY29uc2Npb3VzJTNGJTIwQ2FuJTIweW91JTIwdGFsayUyMHRvJTIwbWUlM0YlMjIlMEFpbnB1dHMlMjAlM0QlMjB0b2tlbml6ZXIocHJvbXB0JTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiklMEElMEElMjMlMjBHZW5lcmF0ZSUwQWdlbmVyYXRlX2lkcyUyMCUzRCUyMG1vZGVsLmdlbmVyYXRlKGlucHV0cy5pbnB1dF9pZHMlMkMlMjBtYXhfbGVuZ3RoJTNEMzApJTBBdG9rZW5pemVyLmJhdGNoX2RlY29kZShnZW5lcmF0ZV9pZHMlMkMlMjBza2lwX3NwZWNpYWxfdG9rZW5zJTNEVHJ1ZSUyQyUyMGNsZWFuX3VwX3Rva2VuaXphdGlvbl9zcGFjZXMlM0RGYWxzZSklNUIwJTVE",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, HunYuanDenseV1ForCausalLM

<span class="hljs-meta">&gt;&gt;&gt; </span>model = HunYuanDenseV1ForCausalLM.from_pretrained(<span class="hljs-string">&quot;meta-hunyuan_v1_dense/HunYuanDenseV1-2-7b-hf&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;meta-hunyuan_v1_dense/HunYuanDenseV1-2-7b-hf&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>prompt = <span class="hljs-string">&quot;Hey, are you conscious? Can you talk to me?&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(prompt, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Generate</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>generate_ids = model.generate(inputs.input_ids, max_length=<span class="hljs-number">30</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer.batch_decode(generate_ids, skip_special_tokens=<span class="hljs-literal">True</span>, clean_up_tokenization_spaces=<span class="hljs-literal">False</span>)[<span class="hljs-number">0</span>]
<span class="hljs-string">&quot;Hey, are you conscious? Can you talk to me?\\nI&#x27;m not conscious, but I can talk to you.&quot;</span>`,wrap:!1}}),{c(){o=c("p"),o.textContent=y,d=s(),u(v.$$.fragment)},l(_){o=l(_,"P",{"data-svelte-h":!0}),b(o)!=="svelte-11lpom8"&&(o.textContent=y),d=a(_),p(v.$$.fragment,_)},m(_,H){i(_,o,H),i(_,d,H),h(v,_,H),C=!0},p:Pe,i(_){C||(m(v.$$.fragment,_),C=!0)},o(_){f(v.$$.fragment,_),C=!1},d(_){_&&(t(o),t(d)),g(v,_)}}}function Dn(D){let o,y=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){o=c("p"),o.innerHTML=y},l(d){o=l(d,"P",{"data-svelte-h":!0}),b(o)!=="svelte-fincs2"&&(o.innerHTML=y)},m(d,v){i(d,o,v)},p:Pe,d(d){d&&t(o)}}}function Vn(D){let o,y,d,v,C,_,H,ye,J,on="To be released with the official model launch.",ke,B,Te,Z,sn="To be released with the official model launch.",we,E,$e,U,an="To be released with the official model launch.",Ce,A,Me,x,G,We,ie,rn=`This is the configuration class to store the configuration of a <a href="/docs/transformers/v4.56.2/en/model_doc/hunyuan_v1_dense#transformers.HunYuanDenseV1Config">HunYuanDenseV1Config</a>. It is used to instantiate an
HunYuan model according to the specified arguments, defining the model architecture. Instantiating a configuration
with the defaults will yield a similar configuration to that of the HunYuan-7B.
Hunyuan-7B-Instruct <a href="https://huggingface.co/tencent/Hunyuan-7B-Instruct" rel="nofollow">tencent/Hunyuan-7B-Instruct</a>.`,Ne,de,dn=`Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> and can be used to control the model outputs. Read the
documentation from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> for more information.`,He,X,xe,k,R,Oe,ce,cn="The bare Hunyuan V1 Dense Model outputting raw hidden-states without any specific head on top.",je,le,ln=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Se,ue,un=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Je,V,Q,Be,pe,pn='The <a href="/docs/transformers/v4.56.2/en/model_doc/hunyuan_v1_dense#transformers.HunYuanDenseV1Model">HunYuanDenseV1Model</a> forward method, overrides the <code>__call__</code> special method.',Ze,P,De,K,Ve,T,ee,Ee,he,hn="The Hunyuan V1 Dense Model for causal language modeling.",Ue,me,mn=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Ae,fe,fn=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Ge,M,ne,Xe,ge,gn='The <a href="/docs/transformers/v4.56.2/en/model_doc/hunyuan_v1_dense#transformers.HunYuanDenseV1ForCausalLM">HunYuanDenseV1ForCausalLM</a> forward method, overrides the <code>__call__</code> special method.',Re,W,Qe,N,Ye,te,ze,z,oe,Ke,Y,se,en,_e,_n="The <code>GenericForSequenceClassification</code> forward method, overrides the <code>__call__</code> special method.",nn,O,Fe,ae,Le,be,qe;return C=new S({props:{title:"HunYuanDenseV1",local:"hunyuandensev1",headingTag:"h1"}}),H=new S({props:{title:"Overview",local:"overview",headingTag:"h2"}}),B=new S({props:{title:"Model Details",local:"model-details",headingTag:"h3"}}),E=new S({props:{title:"Usage tips",local:"usage-tips",headingTag:"h2"}}),A=new S({props:{title:"HunYuanDenseV1Config",local:"transformers.HunYuanDenseV1Config",headingTag:"h2"}}),G=new re({props:{name:"class transformers.HunYuanDenseV1Config",anchor:"transformers.HunYuanDenseV1Config",parameters:[{name:"vocab_size",val:" = 290943"},{name:"hidden_size",val:" = 4096"},{name:"intermediate_size",val:": int = 11008"},{name:"num_hidden_layers",val:" = 32"},{name:"num_attention_heads",val:" = 32"},{name:"num_key_value_heads",val:" = None"},{name:"hidden_act",val:" = 'silu'"},{name:"max_position_embeddings",val:" = 2048"},{name:"initializer_range",val:" = 0.02"},{name:"rms_norm_eps",val:" = 1e-05"},{name:"use_cache",val:" = True"},{name:"pad_token_id",val:" = 0"},{name:"bos_token_id",val:" = 1"},{name:"eos_token_id",val:" = 2"},{name:"eod_token_id",val:" = 3"},{name:"pretraining_tp",val:" = 1"},{name:"tie_word_embeddings",val:" = False"},{name:"rope_theta",val:" = 10000.0"},{name:"rope_scaling",val:" = None"},{name:"attention_bias",val:" = False"},{name:"attention_dropout",val:" = 0.0"},{name:"head_dim",val:" = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.HunYuanDenseV1Config.vocab_size",description:`<strong>vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 290943) &#x2014;
Vocabulary size of the HunYuan model. Defines the number of different tokens that can be represented by the
<code>inputs_ids</code> passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/hunyuan_v1_dense#transformers.HunYuanDenseV1Config">HunYuanDenseV1Config</a>`,name:"vocab_size"},{anchor:"transformers.HunYuanDenseV1Config.hidden_size",description:`<strong>hidden_size</strong> (<code>int</code>, <em>optional</em>, defaults to 4096) &#x2014;
Dimension of the hidden representations.`,name:"hidden_size"},{anchor:"transformers.HunYuanDenseV1Config.intermediate_size",description:`<strong>intermediate_size</strong> (<code>int</code>, <em>optional</em>, defaults to 11008) &#x2014;
Dimension of the MLP representations or shared MLP representations.`,name:"intermediate_size"},{anchor:"transformers.HunYuanDenseV1Config.num_hidden_layers",description:`<strong>num_hidden_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 32) &#x2014;
Number of hidden layers in the Transformer decoder.`,name:"num_hidden_layers"},{anchor:"transformers.HunYuanDenseV1Config.num_attention_heads",description:`<strong>num_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 32) &#x2014;
Number of attention heads for each attention layer in the Transformer decoder.`,name:"num_attention_heads"},{anchor:"transformers.HunYuanDenseV1Config.num_key_value_heads",description:`<strong>num_key_value_heads</strong> (<code>int</code>, <em>optional</em>) &#x2014;
This is the number of key_value heads that should be used to implement Grouped Query Attention. If
<code>num_key_value_heads=num_attention_heads</code>, the model will use Multi Head Attention (MHA), if
<code>num_key_value_heads=1 the model will use Multi Query Attention (MQA) otherwise GQA is used. When converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed by meanpooling all the original heads within that group. For more details checkout [this paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to </code>num_attention_heads\`.`,name:"num_key_value_heads"},{anchor:"transformers.HunYuanDenseV1Config.hidden_act",description:`<strong>hidden_act</strong> (<code>str</code> or <code>function</code>, <em>optional</em>, defaults to <code>&quot;silu&quot;</code>) &#x2014;
The non-linear activation function (function or string) in the decoder.`,name:"hidden_act"},{anchor:"transformers.HunYuanDenseV1Config.max_position_embeddings",description:`<strong>max_position_embeddings</strong> (<code>int</code>, <em>optional</em>, defaults to 2048) &#x2014;
The maximum sequence length that this model might ever be used with.`,name:"max_position_embeddings"},{anchor:"transformers.HunYuanDenseV1Config.initializer_range",description:`<strong>initializer_range</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.`,name:"initializer_range"},{anchor:"transformers.HunYuanDenseV1Config.rms_norm_eps",description:`<strong>rms_norm_eps</strong> (<code>float</code>, <em>optional</em>, defaults to 1e-05) &#x2014;
The epsilon used by the rms normalization layers.`,name:"rms_norm_eps"},{anchor:"transformers.HunYuanDenseV1Config.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not the model should return the last key/values attentions (not used by all models). Only
relevant if <code>config.is_decoder=True</code>.`,name:"use_cache"},{anchor:"transformers.HunYuanDenseV1Config.pad_token_id",description:`<strong>pad_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 0) &#x2014;
Padding token id.`,name:"pad_token_id"},{anchor:"transformers.HunYuanDenseV1Config.bos_token_id",description:`<strong>bos_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 1) &#x2014;
Beginning of stream token id.`,name:"bos_token_id"},{anchor:"transformers.HunYuanDenseV1Config.eos_token_id",description:`<strong>eos_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 2) &#x2014;
End of stream token id.`,name:"eos_token_id"},{anchor:"transformers.HunYuanDenseV1Config.eod_token_id",description:`<strong>eod_token_id</strong> (int, <em>optional</em>, defaults to 3) &#x2014;
Token ID representing the end-of-document marker. Used to indicate the termination of a text sequence.
Example: In multi-document processing, this token helps the model distinguish between separate documents.`,name:"eod_token_id"},{anchor:"transformers.HunYuanDenseV1Config.pretraining_tp",description:`<strong>pretraining_tp</strong> (<code>int</code>, <em>optional</em>, defaults to 1) &#x2014;
Experimental feature. Tensor parallelism rank used during pretraining. Please refer to <a href="https://huggingface.co/docs/transformers/parallelism" rel="nofollow">this
document</a> to understand more about it. This value is
necessary to ensure exact reproducibility of the pretraining results. Please refer to <a href="https://github.com/pytorch/pytorch/issues/76232" rel="nofollow">this
issue</a>.`,name:"pretraining_tp"},{anchor:"transformers.HunYuanDenseV1Config.tie_word_embeddings",description:`<strong>tie_word_embeddings</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether to tie weight embeddings`,name:"tie_word_embeddings"},{anchor:"transformers.HunYuanDenseV1Config.rope_theta",description:`<strong>rope_theta</strong> (<code>float</code>, <em>optional</em>, defaults to 10000.0) &#x2014;
The base period of the RoPE embeddings.`,name:"rope_theta"},{anchor:"transformers.HunYuanDenseV1Config.rope_scaling",description:`<strong>rope_scaling</strong> (<code>Dict</code>, <em>optional</em>) &#x2014;
Dictionary containing the scaling configuration for the RoPE embeddings. Currently supports two scaling
strategies: linear and dynamic. Their scaling factor must be a float greater than 1. The expected format is
<code>{&quot;type&quot;: strategy name, &quot;factor&quot;: scaling factor}</code>. When using this flag, don&#x2019;t update
<code>max_position_embeddings</code> to the expected new maximum. See the following thread for more information on how
these scaling strategies behave:
<a href="https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically_scaled_rope_further_increases/" rel="nofollow">https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically_scaled_rope_further_increases/</a>. This is an
experimental feature, subject to breaking API changes in future versions.`,name:"rope_scaling"},{anchor:"transformers.HunYuanDenseV1Config.attention_bias",description:`<strong>attention_bias</strong> (<code>bool</code>, defaults to <code>False</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether to use a bias in the query, key, value and output projection layers during self-attention.`,name:"attention_bias"},{anchor:"transformers.HunYuanDenseV1Config.attention_dropout",description:`<strong>attention_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The dropout ratio for the attention probabilities.`,name:"attention_dropout"},{anchor:"transformers.HunYuanDenseV1Config.head_dim",description:`<strong>head_dim</strong> (<code>int</code>, <em>optional</em>, defaults to 128) &#x2014;
The attention head dimension.`,name:"head_dim"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/hunyuan_v1_dense/configuration_hunyuan_v1_dense.py#L24"}}),X=new S({props:{title:"HunYuanModel",local:"transformers.HunYuanDenseV1Model",headingTag:"h2"}}),R=new re({props:{name:"class transformers.HunYuanDenseV1Model",anchor:"transformers.HunYuanDenseV1Model",parameters:[{name:"config",val:": HunYuanDenseV1Config"}],parametersDescription:[{anchor:"transformers.HunYuanDenseV1Model.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/hunyuan_v1_dense#transformers.HunYuanDenseV1Config">HunYuanDenseV1Config</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/hunyuan_v1_dense/modeling_hunyuan_v1_dense.py#L351"}}),Q=new re({props:{name:"forward",anchor:"transformers.HunYuanDenseV1Model.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.HunYuanDenseV1Model.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.HunYuanDenseV1Model.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.HunYuanDenseV1Model.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.HunYuanDenseV1Model.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.HunYuanDenseV1Model.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.HunYuanDenseV1Model.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"},{anchor:"transformers.HunYuanDenseV1Model.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/hunyuan_v1_dense/modeling_hunyuan_v1_dense.py#L368",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast"
>transformers.modeling_outputs.BaseModelOutputWithPast</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/hunyuan_v1_dense#transformers.HunYuanDenseV1Config"
>HunYuanDenseV1Config</a>) and inputs.</p>
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
`}}),P=new tn({props:{$$slots:{default:[Mn]},$$scope:{ctx:D}}}),K=new S({props:{title:"HunYuanDenseV1ForCausalLM",local:"transformers.HunYuanDenseV1ForCausalLM",headingTag:"h2"}}),ee=new re({props:{name:"class transformers.HunYuanDenseV1ForCausalLM",anchor:"transformers.HunYuanDenseV1ForCausalLM",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.HunYuanDenseV1ForCausalLM.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/hunyuan_v1_dense#transformers.HunYuanDenseV1ForCausalLM">HunYuanDenseV1ForCausalLM</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/hunyuan_v1_dense/modeling_hunyuan_v1_dense.py#L430"}}),ne=new re({props:{name:"forward",anchor:"transformers.HunYuanDenseV1ForCausalLM.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"},{name:"logits_to_keep",val:": typing.Union[int, torch.Tensor] = 0"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.HunYuanDenseV1ForCausalLM.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.HunYuanDenseV1ForCausalLM.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.HunYuanDenseV1ForCausalLM.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.HunYuanDenseV1ForCausalLM.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.HunYuanDenseV1ForCausalLM.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.HunYuanDenseV1ForCausalLM.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.HunYuanDenseV1ForCausalLM.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.HunYuanDenseV1ForCausalLM.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"},{anchor:"transformers.HunYuanDenseV1ForCausalLM.forward.logits_to_keep",description:`<strong>logits_to_keep</strong> (<code>Union[int, torch.Tensor]</code>, defaults to <code>0</code>) &#x2014;
If an <code>int</code>, compute logits for the last <code>logits_to_keep</code> tokens. If <code>0</code>, calculate logits for all
<code>input_ids</code> (special case). Only last token logits are needed for generation, and calculating them only for that
token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
If a <code>torch.Tensor</code>, must be 1D corresponding to the indices to keep in the sequence length dimension.
This is useful when using packed tensor format (single dimension for batch and sequence length).`,name:"logits_to_keep"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/hunyuan_v1_dense/modeling_hunyuan_v1_dense.py#L444",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast"
>transformers.modeling_outputs.CausalLMOutputWithPast</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/hunyuan_v1_dense#transformers.HunYuanDenseV1Config"
>HunYuanDenseV1Config</a>) and inputs.</p>
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
`}}),W=new tn({props:{$$slots:{default:[Hn]},$$scope:{ctx:D}}}),N=new $n({props:{anchor:"transformers.HunYuanDenseV1ForCausalLM.forward.example",$$slots:{default:[xn]},$$scope:{ctx:D}}}),te=new S({props:{title:"HunYuanDenseV1ForSequenceClassification",local:"transformers.HunYuanDenseV1ForSequenceClassification",headingTag:"h2"}}),oe=new re({props:{name:"class transformers.HunYuanDenseV1ForSequenceClassification",anchor:"transformers.HunYuanDenseV1ForSequenceClassification",parameters:[{name:"config",val:""}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/hunyuan_v1_dense/modeling_hunyuan_v1_dense.py#L505"}}),se=new re({props:{name:"forward",anchor:"transformers.HunYuanDenseV1ForSequenceClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.HunYuanDenseV1ForSequenceClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.HunYuanDenseV1ForSequenceClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.HunYuanDenseV1ForSequenceClassification.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.HunYuanDenseV1ForSequenceClassification.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.HunYuanDenseV1ForSequenceClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.HunYuanDenseV1ForSequenceClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.HunYuanDenseV1ForSequenceClassification.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
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
`}}),O=new tn({props:{$$slots:{default:[Dn]},$$scope:{ctx:D}}}),ae=new Cn({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/hunyuan_v1_dense.md"}}),{c(){o=c("meta"),y=s(),d=c("p"),v=s(),u(C.$$.fragment),_=s(),u(H.$$.fragment),ye=s(),J=c("p"),J.textContent=on,ke=s(),u(B.$$.fragment),Te=s(),Z=c("p"),Z.textContent=sn,we=s(),u(E.$$.fragment),$e=s(),U=c("p"),U.textContent=an,Ce=s(),u(A.$$.fragment),Me=s(),x=c("div"),u(G.$$.fragment),We=s(),ie=c("p"),ie.innerHTML=rn,Ne=s(),de=c("p"),de.innerHTML=dn,He=s(),u(X.$$.fragment),xe=s(),k=c("div"),u(R.$$.fragment),Oe=s(),ce=c("p"),ce.textContent=cn,je=s(),le=c("p"),le.innerHTML=ln,Se=s(),ue=c("p"),ue.innerHTML=un,Je=s(),V=c("div"),u(Q.$$.fragment),Be=s(),pe=c("p"),pe.innerHTML=pn,Ze=s(),u(P.$$.fragment),De=s(),u(K.$$.fragment),Ve=s(),T=c("div"),u(ee.$$.fragment),Ee=s(),he=c("p"),he.textContent=hn,Ue=s(),me=c("p"),me.innerHTML=mn,Ae=s(),fe=c("p"),fe.innerHTML=fn,Ge=s(),M=c("div"),u(ne.$$.fragment),Xe=s(),ge=c("p"),ge.innerHTML=gn,Re=s(),u(W.$$.fragment),Qe=s(),u(N.$$.fragment),Ye=s(),u(te.$$.fragment),ze=s(),z=c("div"),u(oe.$$.fragment),Ke=s(),Y=c("div"),u(se.$$.fragment),en=s(),_e=c("p"),_e.innerHTML=_n,nn=s(),u(O.$$.fragment),Fe=s(),u(ae.$$.fragment),Le=s(),be=c("p"),this.h()},l(e){const n=Tn("svelte-u9bgzb",document.head);o=l(n,"META",{name:!0,content:!0}),n.forEach(t),y=a(e),d=l(e,"P",{}),q(d).forEach(t),v=a(e),p(C.$$.fragment,e),_=a(e),p(H.$$.fragment,e),ye=a(e),J=l(e,"P",{"data-svelte-h":!0}),b(J)!=="svelte-u40g91"&&(J.textContent=on),ke=a(e),p(B.$$.fragment,e),Te=a(e),Z=l(e,"P",{"data-svelte-h":!0}),b(Z)!=="svelte-u40g91"&&(Z.textContent=sn),we=a(e),p(E.$$.fragment,e),$e=a(e),U=l(e,"P",{"data-svelte-h":!0}),b(U)!=="svelte-u40g91"&&(U.textContent=an),Ce=a(e),p(A.$$.fragment,e),Me=a(e),x=l(e,"DIV",{class:!0});var F=q(x);p(G.$$.fragment,F),We=a(F),ie=l(F,"P",{"data-svelte-h":!0}),b(ie)!=="svelte-hchcv"&&(ie.innerHTML=rn),Ne=a(F),de=l(F,"P",{"data-svelte-h":!0}),b(de)!=="svelte-1ek1ss9"&&(de.innerHTML=dn),F.forEach(t),He=a(e),p(X.$$.fragment,e),xe=a(e),k=l(e,"DIV",{class:!0});var w=q(k);p(R.$$.fragment,w),Oe=a(w),ce=l(w,"P",{"data-svelte-h":!0}),b(ce)!=="svelte-v2nh6w"&&(ce.textContent=cn),je=a(w),le=l(w,"P",{"data-svelte-h":!0}),b(le)!=="svelte-q52n56"&&(le.innerHTML=ln),Se=a(w),ue=l(w,"P",{"data-svelte-h":!0}),b(ue)!=="svelte-hswkmf"&&(ue.innerHTML=un),Je=a(w),V=l(w,"DIV",{class:!0});var L=q(V);p(Q.$$.fragment,L),Be=a(L),pe=l(L,"P",{"data-svelte-h":!0}),b(pe)!=="svelte-1ytpadd"&&(pe.innerHTML=pn),Ze=a(L),p(P.$$.fragment,L),L.forEach(t),w.forEach(t),De=a(e),p(K.$$.fragment,e),Ve=a(e),T=l(e,"DIV",{class:!0});var $=q(T);p(ee.$$.fragment,$),Ee=a($),he=l($,"P",{"data-svelte-h":!0}),b(he)!=="svelte-em1vlb"&&(he.textContent=hn),Ue=a($),me=l($,"P",{"data-svelte-h":!0}),b(me)!=="svelte-q52n56"&&(me.innerHTML=mn),Ae=a($),fe=l($,"P",{"data-svelte-h":!0}),b(fe)!=="svelte-hswkmf"&&(fe.innerHTML=fn),Ge=a($),M=l($,"DIV",{class:!0});var j=q(M);p(ne.$$.fragment,j),Xe=a(j),ge=l(j,"P",{"data-svelte-h":!0}),b(ge)!=="svelte-ufmd6d"&&(ge.innerHTML=gn),Re=a(j),p(W.$$.fragment,j),Qe=a(j),p(N.$$.fragment,j),j.forEach(t),$.forEach(t),Ye=a(e),p(te.$$.fragment,e),ze=a(e),z=l(e,"DIV",{class:!0});var Ie=q(z);p(oe.$$.fragment,Ie),Ke=a(Ie),Y=l(Ie,"DIV",{class:!0});var ve=q(Y);p(se.$$.fragment,ve),en=a(ve),_e=l(ve,"P",{"data-svelte-h":!0}),b(_e)!=="svelte-1sal4ui"&&(_e.innerHTML=_n),nn=a(ve),p(O.$$.fragment,ve),ve.forEach(t),Ie.forEach(t),Fe=a(e),p(ae.$$.fragment,e),Le=a(e),be=l(e,"P",{}),q(be).forEach(t),this.h()},h(){I(o,"name","hf:doc:metadata"),I(o,"content",Yn),I(x,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),I(V,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),I(k,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),I(M,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),I(T,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),I(Y,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),I(z,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,n){r(document.head,o),i(e,y,n),i(e,d,n),i(e,v,n),h(C,e,n),i(e,_,n),h(H,e,n),i(e,ye,n),i(e,J,n),i(e,ke,n),h(B,e,n),i(e,Te,n),i(e,Z,n),i(e,we,n),h(E,e,n),i(e,$e,n),i(e,U,n),i(e,Ce,n),h(A,e,n),i(e,Me,n),i(e,x,n),h(G,x,null),r(x,We),r(x,ie),r(x,Ne),r(x,de),i(e,He,n),h(X,e,n),i(e,xe,n),i(e,k,n),h(R,k,null),r(k,Oe),r(k,ce),r(k,je),r(k,le),r(k,Se),r(k,ue),r(k,Je),r(k,V),h(Q,V,null),r(V,Be),r(V,pe),r(V,Ze),h(P,V,null),i(e,De,n),h(K,e,n),i(e,Ve,n),i(e,T,n),h(ee,T,null),r(T,Ee),r(T,he),r(T,Ue),r(T,me),r(T,Ae),r(T,fe),r(T,Ge),r(T,M),h(ne,M,null),r(M,Xe),r(M,ge),r(M,Re),h(W,M,null),r(M,Qe),h(N,M,null),i(e,Ye,n),h(te,e,n),i(e,ze,n),i(e,z,n),h(oe,z,null),r(z,Ke),r(z,Y),h(se,Y,null),r(Y,en),r(Y,_e),r(Y,nn),h(O,Y,null),i(e,Fe,n),h(ae,e,n),i(e,Le,n),i(e,be,n),qe=!0},p(e,[n]){const F={};n&2&&(F.$$scope={dirty:n,ctx:e}),P.$set(F);const w={};n&2&&(w.$$scope={dirty:n,ctx:e}),W.$set(w);const L={};n&2&&(L.$$scope={dirty:n,ctx:e}),N.$set(L);const $={};n&2&&($.$$scope={dirty:n,ctx:e}),O.$set($)},i(e){qe||(m(C.$$.fragment,e),m(H.$$.fragment,e),m(B.$$.fragment,e),m(E.$$.fragment,e),m(A.$$.fragment,e),m(G.$$.fragment,e),m(X.$$.fragment,e),m(R.$$.fragment,e),m(Q.$$.fragment,e),m(P.$$.fragment,e),m(K.$$.fragment,e),m(ee.$$.fragment,e),m(ne.$$.fragment,e),m(W.$$.fragment,e),m(N.$$.fragment,e),m(te.$$.fragment,e),m(oe.$$.fragment,e),m(se.$$.fragment,e),m(O.$$.fragment,e),m(ae.$$.fragment,e),qe=!0)},o(e){f(C.$$.fragment,e),f(H.$$.fragment,e),f(B.$$.fragment,e),f(E.$$.fragment,e),f(A.$$.fragment,e),f(G.$$.fragment,e),f(X.$$.fragment,e),f(R.$$.fragment,e),f(Q.$$.fragment,e),f(P.$$.fragment,e),f(K.$$.fragment,e),f(ee.$$.fragment,e),f(ne.$$.fragment,e),f(W.$$.fragment,e),f(N.$$.fragment,e),f(te.$$.fragment,e),f(oe.$$.fragment,e),f(se.$$.fragment,e),f(O.$$.fragment,e),f(ae.$$.fragment,e),qe=!1},d(e){e&&(t(y),t(d),t(v),t(_),t(ye),t(J),t(ke),t(Te),t(Z),t(we),t($e),t(U),t(Ce),t(Me),t(x),t(He),t(xe),t(k),t(De),t(Ve),t(T),t(Ye),t(ze),t(z),t(Fe),t(Le),t(be)),t(o),g(C,e),g(H,e),g(B,e),g(E,e),g(A,e),g(G),g(X,e),g(R),g(Q),g(P),g(K,e),g(ee),g(ne),g(W),g(N),g(te,e),g(oe),g(se),g(O),g(ae,e)}}}const Yn='{"title":"HunYuanDenseV1","local":"hunyuandensev1","sections":[{"title":"Overview","local":"overview","sections":[{"title":"Model Details","local":"model-details","sections":[],"depth":3}],"depth":2},{"title":"Usage tips","local":"usage-tips","sections":[],"depth":2},{"title":"HunYuanDenseV1Config","local":"transformers.HunYuanDenseV1Config","sections":[],"depth":2},{"title":"HunYuanModel","local":"transformers.HunYuanDenseV1Model","sections":[],"depth":2},{"title":"HunYuanDenseV1ForCausalLM","local":"transformers.HunYuanDenseV1ForCausalLM","sections":[],"depth":2},{"title":"HunYuanDenseV1ForSequenceClassification","local":"transformers.HunYuanDenseV1ForSequenceClassification","sections":[],"depth":2}],"depth":1}';function zn(D){return bn(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class On extends yn{constructor(o){super(),kn(this,o,zn,Vn,vn,{})}}export{On as component};
