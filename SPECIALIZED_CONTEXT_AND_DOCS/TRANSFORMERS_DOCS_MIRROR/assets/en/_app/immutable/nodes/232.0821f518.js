import{s as vn,o as bn,n as Pe}from"../chunks/scheduler.18a86fab.js";import{S as yn,i as Mn,g as c,s,r as u,A as kn,h as l,f as o,c as a,j as q,u as p,x as b,k as I,y as r,a as i,v as m,d as h,t as f,w as g}from"../chunks/index.98837b22.js";import{T as on}from"../chunks/Tip.77304350.js";import{D as re}from"../chunks/Docstring.a1ef7999.js";import{C as Tn}from"../chunks/CodeBlock.8d0c2e8a.js";import{E as wn}from"../chunks/ExampleCodeBlock.8c3ee1f9.js";import{H as N,E as $n}from"../chunks/getInferenceSnippets.06c2775f.js";function Cn(V){let t,y=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=y},l(d){t=l(d,"P",{"data-svelte-h":!0}),b(t)!=="svelte-fincs2"&&(t.innerHTML=y)},m(d,v){i(d,t,v)},p:Pe,d(d){d&&o(t)}}}function xn(V){let t,y=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=y},l(d){t=l(d,"P",{"data-svelte-h":!0}),b(t)!=="svelte-fincs2"&&(t.innerHTML=y)},m(d,v){i(d,t,v)},p:Pe,d(d){d&&o(t)}}}function Hn(V){let t,y="Example:",d,v,$;return v=new Tn({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBIdW5ZdWFuTW9FVjFGb3JDYXVzYWxMTSUwQSUwQW1vZGVsJTIwJTNEJTIwSHVuWXVhbk1vRVYxRm9yQ2F1c2FsTE0uZnJvbV9wcmV0cmFpbmVkKCUyMm1ldGEtaHVueXVhbl92MV9tb2UlMkZIdW5ZdWFuTW9FVjEtMi03Yi1oZiUyMiklMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJtZXRhLWh1bnl1YW5fdjFfbW9lJTJGSHVuWXVhbk1vRVYxLTItN2ItaGYlMjIpJTBBJTBBcHJvbXB0JTIwJTNEJTIwJTIySGV5JTJDJTIwYXJlJTIweW91JTIwY29uc2Npb3VzJTNGJTIwQ2FuJTIweW91JTIwdGFsayUyMHRvJTIwbWUlM0YlMjIlMEFpbnB1dHMlMjAlM0QlMjB0b2tlbml6ZXIocHJvbXB0JTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiklMEElMEElMjMlMjBHZW5lcmF0ZSUwQWdlbmVyYXRlX2lkcyUyMCUzRCUyMG1vZGVsLmdlbmVyYXRlKGlucHV0cy5pbnB1dF9pZHMlMkMlMjBtYXhfbGVuZ3RoJTNEMzApJTBBdG9rZW5pemVyLmJhdGNoX2RlY29kZShnZW5lcmF0ZV9pZHMlMkMlMjBza2lwX3NwZWNpYWxfdG9rZW5zJTNEVHJ1ZSUyQyUyMGNsZWFuX3VwX3Rva2VuaXphdGlvbl9zcGFjZXMlM0RGYWxzZSklNUIwJTVE",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, HunYuanMoEV1ForCausalLM

<span class="hljs-meta">&gt;&gt;&gt; </span>model = HunYuanMoEV1ForCausalLM.from_pretrained(<span class="hljs-string">&quot;meta-hunyuan_v1_moe/HunYuanMoEV1-2-7b-hf&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;meta-hunyuan_v1_moe/HunYuanMoEV1-2-7b-hf&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>prompt = <span class="hljs-string">&quot;Hey, are you conscious? Can you talk to me?&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(prompt, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Generate</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>generate_ids = model.generate(inputs.input_ids, max_length=<span class="hljs-number">30</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer.batch_decode(generate_ids, skip_special_tokens=<span class="hljs-literal">True</span>, clean_up_tokenization_spaces=<span class="hljs-literal">False</span>)[<span class="hljs-number">0</span>]
<span class="hljs-string">&quot;Hey, are you conscious? Can you talk to me?\\nI&#x27;m not conscious, but I can talk to you.&quot;</span>`,wrap:!1}}),{c(){t=c("p"),t.textContent=y,d=s(),u(v.$$.fragment)},l(_){t=l(_,"P",{"data-svelte-h":!0}),b(t)!=="svelte-11lpom8"&&(t.textContent=y),d=a(_),p(v.$$.fragment,_)},m(_,x){i(_,t,x),i(_,d,x),m(v,_,x),$=!0},p:Pe,i(_){$||(h(v.$$.fragment,_),$=!0)},o(_){f(v.$$.fragment,_),$=!1},d(_){_&&(o(t),o(d)),g(v,_)}}}function Vn(V){let t,y=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=c("p"),t.innerHTML=y},l(d){t=l(d,"P",{"data-svelte-h":!0}),b(t)!=="svelte-fincs2"&&(t.innerHTML=y)},m(d,v){i(d,t,v)},p:Pe,d(d){d&&o(t)}}}function En(V){let t,y,d,v,$,_,x,ye,J,tn="To be released with the official model launch.",Me,S,ke,A,sn="To be released with the official model launch.",Te,B,we,U,an="To be released with the official model launch.",$e,Z,Ce,H,G,We,ie,rn=`This is the configuration class to store the configuration of a <a href="/docs/transformers/v4.56.2/en/model_doc/hunyuan_v1_moe#transformers.HunYuanMoEV1Model">HunYuanMoEV1Model</a>. It is used to instantiate an
HunYuan model according to the specified arguments, defining the model architecture. Instantiating a configuration
with the defaults will yield a similar configuration to that of the HunYuan-7B.
Hunyuan-A13B-Instruct <a href="https://huggingface.co/tencent/Hunyuan-A13B-Instruct" rel="nofollow">tencent/Hunyuan-A13B-Instruct</a>.`,De,de,dn=`Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> and can be used to control the model outputs. Read the
documentation from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> for more information.`,xe,X,He,M,R,Oe,ce,cn="The bare Hunyuan V1 Moe Model outputting raw hidden-states without any specific head on top.",je,le,ln=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Ne,ue,un=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Je,E,Q,Se,pe,pn='The <a href="/docs/transformers/v4.56.2/en/model_doc/hunyuan_v1_moe#transformers.HunYuanMoEV1Model">HunYuanMoEV1Model</a> forward method, overrides the <code>__call__</code> special method.',Ae,P,Ve,K,Ee,k,ee,Be,me,mn="The Hunyuan V1 Moe Model for causal language modeling.",Ue,he,hn=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Ze,fe,fn=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Ge,C,ne,Xe,ge,gn='The <a href="/docs/transformers/v4.56.2/en/model_doc/hunyuan_v1_moe#transformers.HunYuanMoEV1ForCausalLM">HunYuanMoEV1ForCausalLM</a> forward method, overrides the <code>__call__</code> special method.',Re,W,Qe,D,Ye,oe,ze,z,te,Ke,Y,se,en,_e,_n="The <code>GenericForSequenceClassification</code> forward method, overrides the <code>__call__</code> special method.",nn,O,Fe,ae,Le,be,qe;return $=new N({props:{title:"HunYuanMoEV1",local:"hunyuanmoev1",headingTag:"h1"}}),x=new N({props:{title:"Overview",local:"overview",headingTag:"h2"}}),S=new N({props:{title:"Model Details",local:"model-details",headingTag:"h3"}}),B=new N({props:{title:"Usage tips",local:"usage-tips",headingTag:"h2"}}),Z=new N({props:{title:"HunYuanMoEV1Config",local:"transformers.HunYuanMoEV1Config",headingTag:"h2"}}),G=new re({props:{name:"class transformers.HunYuanMoEV1Config",anchor:"transformers.HunYuanMoEV1Config",parameters:[{name:"vocab_size",val:" = 290943"},{name:"hidden_size",val:" = 4096"},{name:"intermediate_size",val:": int = 11008"},{name:"num_hidden_layers",val:" = 32"},{name:"num_attention_heads",val:" = 32"},{name:"num_key_value_heads",val:" = None"},{name:"hidden_act",val:" = 'silu'"},{name:"max_position_embeddings",val:" = 2048"},{name:"initializer_range",val:" = 0.02"},{name:"rms_norm_eps",val:" = 1e-05"},{name:"use_cache",val:" = True"},{name:"pad_token_id",val:" = 0"},{name:"bos_token_id",val:" = 1"},{name:"eos_token_id",val:" = 2"},{name:"eod_token_id",val:" = 3"},{name:"sep_token_id",val:" = 4"},{name:"pretraining_tp",val:" = 1"},{name:"tie_word_embeddings",val:" = False"},{name:"rope_theta",val:" = 10000.0"},{name:"rope_scaling",val:" = None"},{name:"attention_bias",val:" = False"},{name:"attention_dropout",val:" = 0.0"},{name:"num_experts",val:": typing.Union[int, list] = 1"},{name:"moe_topk",val:": typing.Union[int, list] = 1"},{name:"head_dim",val:" = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.HunYuanMoEV1Config.vocab_size",description:`<strong>vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 290943) &#x2014;
Vocabulary size of the HunYuan model. Defines the number of different tokens that can be represented by the
<code>inputs_ids</code> passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/hunyuan_v1_moe#transformers.HunYuanMoEV1Model">HunYuanMoEV1Model</a>`,name:"vocab_size"},{anchor:"transformers.HunYuanMoEV1Config.hidden_size",description:`<strong>hidden_size</strong> (<code>int</code>, <em>optional</em>, defaults to 4096) &#x2014;
Dimension of the hidden representations.`,name:"hidden_size"},{anchor:"transformers.HunYuanMoEV1Config.intermediate_size",description:`<strong>intermediate_size</strong> (<code>int</code>, <em>optional</em>, defaults to 11008) &#x2014;
Dimension of the MLP representations or shared MLP representations.`,name:"intermediate_size"},{anchor:"transformers.HunYuanMoEV1Config.num_hidden_layers",description:`<strong>num_hidden_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 32) &#x2014;
Number of hidden layers in the Transformer decoder.`,name:"num_hidden_layers"},{anchor:"transformers.HunYuanMoEV1Config.num_attention_heads",description:`<strong>num_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 32) &#x2014;
Number of attention heads for each attention layer in the Transformer decoder.`,name:"num_attention_heads"},{anchor:"transformers.HunYuanMoEV1Config.num_key_value_heads",description:`<strong>num_key_value_heads</strong> (<code>int</code>, <em>optional</em>) &#x2014;
This is the number of key_value heads that should be used to implement Grouped Query Attention. If
<code>num_key_value_heads=num_attention_heads</code>, the model will use Multi Head Attention (MHA), if
<code>num_key_value_heads=1 the model will use Multi Query Attention (MQA) otherwise GQA is used. When converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed by meanpooling all the original heads within that group. For more details checkout [this paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to </code>num_attention_heads\`.`,name:"num_key_value_heads"},{anchor:"transformers.HunYuanMoEV1Config.hidden_act",description:`<strong>hidden_act</strong> (<code>str</code> or <code>function</code>, <em>optional</em>, defaults to <code>&quot;silu&quot;</code>) &#x2014;
The non-linear activation function (function or string) in the decoder.`,name:"hidden_act"},{anchor:"transformers.HunYuanMoEV1Config.max_position_embeddings",description:`<strong>max_position_embeddings</strong> (<code>int</code>, <em>optional</em>, defaults to 2048) &#x2014;
The maximum sequence length that this model might ever be used with.`,name:"max_position_embeddings"},{anchor:"transformers.HunYuanMoEV1Config.initializer_range",description:`<strong>initializer_range</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.`,name:"initializer_range"},{anchor:"transformers.HunYuanMoEV1Config.rms_norm_eps",description:`<strong>rms_norm_eps</strong> (<code>float</code>, <em>optional</em>, defaults to 1e-05) &#x2014;
The epsilon used by the rms normalization layers.`,name:"rms_norm_eps"},{anchor:"transformers.HunYuanMoEV1Config.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not the model should return the last key/values attentions (not used by all models). Only
relevant if <code>config.is_decoder=True</code>.`,name:"use_cache"},{anchor:"transformers.HunYuanMoEV1Config.pad_token_id",description:`<strong>pad_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 0) &#x2014;
Padding token id.`,name:"pad_token_id"},{anchor:"transformers.HunYuanMoEV1Config.bos_token_id",description:`<strong>bos_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 1) &#x2014;
Beginning of stream token id.`,name:"bos_token_id"},{anchor:"transformers.HunYuanMoEV1Config.eos_token_id",description:`<strong>eos_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 2) &#x2014;
End of stream token id.`,name:"eos_token_id"},{anchor:"transformers.HunYuanMoEV1Config.eod_token_id",description:`<strong>eod_token_id</strong> (int, <em>optional</em>, defaults to 3) &#x2014;
Token ID representing the end-of-document marker. Used to indicate the termination of a text sequence.
Example: In multi-document processing, this token helps the model distinguish between separate documents.`,name:"eod_token_id"},{anchor:"transformers.HunYuanMoEV1Config.sep_token_id",description:`<strong>sep_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 4) &#x2014;
Token ID representing the separator token (<code>[SEP]</code>), used to demarcate boundaries between different text segments.`,name:"sep_token_id"},{anchor:"transformers.HunYuanMoEV1Config.pretraining_tp",description:`<strong>pretraining_tp</strong> (<code>int</code>, <em>optional</em>, defaults to 1) &#x2014;
Experimental feature. Tensor parallelism rank used during pretraining. Please refer to <a href="https://huggingface.co/docs/transformers/parallelism" rel="nofollow">this
document</a> to understand more about it. This value is
necessary to ensure exact reproducibility of the pretraining results. Please refer to <a href="https://github.com/pytorch/pytorch/issues/76232" rel="nofollow">this
issue</a>.`,name:"pretraining_tp"},{anchor:"transformers.HunYuanMoEV1Config.tie_word_embeddings",description:`<strong>tie_word_embeddings</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether to tie weight embeddings`,name:"tie_word_embeddings"},{anchor:"transformers.HunYuanMoEV1Config.rope_theta",description:`<strong>rope_theta</strong> (<code>float</code>, <em>optional</em>, defaults to 10000.0) &#x2014;
The base period of the RoPE embeddings.`,name:"rope_theta"},{anchor:"transformers.HunYuanMoEV1Config.rope_scaling",description:`<strong>rope_scaling</strong> (<code>Dict</code>, <em>optional</em>) &#x2014;
Dictionary containing the scaling configuration for the RoPE embeddings. Currently supports two scaling
strategies: linear and dynamic. Their scaling factor must be a float greater than 1. The expected format is
<code>{&quot;type&quot;: strategy name, &quot;factor&quot;: scaling factor}</code>. When using this flag, don&#x2019;t update
<code>max_position_embeddings</code> to the expected new maximum. See the following thread for more information on how
these scaling strategies behave:
<a href="https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically_scaled_rope_further_increases/" rel="nofollow">https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically_scaled_rope_further_increases/</a>. This is an
experimental feature, subject to breaking API changes in future versions.`,name:"rope_scaling"},{anchor:"transformers.HunYuanMoEV1Config.attention_bias",description:`<strong>attention_bias</strong> (<code>bool</code>, defaults to <code>False</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether to use a bias in the query, key, value and output projection layers during self-attention.`,name:"attention_bias"},{anchor:"transformers.HunYuanMoEV1Config.attention_dropout",description:`<strong>attention_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The dropout ratio for the attention probabilities.`,name:"attention_dropout"},{anchor:"transformers.HunYuanMoEV1Config.num_experts",description:`<strong>num_experts</strong> (<code>int</code> or <code>List</code>, <em>optional</em>, defaults to 1) &#x2014;
The number of experts for moe. If it is a list, it will be used as the number of experts for each layer.`,name:"num_experts"},{anchor:"transformers.HunYuanMoEV1Config.moe_topk",description:`<strong>moe_topk</strong> (int or List, <em>optional</em>, defaults to 1) &#x2014;
Number of experts selected per token (Top-K routing). List form enables layer-wise customization.`,name:"moe_topk"},{anchor:"transformers.HunYuanMoEV1Config.head_dim",description:`<strong>head_dim</strong> (<code>int</code>, <em>optional</em>, defaults to 128) &#x2014;
The attention head dimension.`,name:"head_dim"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/hunyuan_v1_moe/configuration_hunyuan_v1_moe.py#L26"}}),X=new N({props:{title:"HunYuanMoEV1Model",local:"transformers.HunYuanMoEV1Model",headingTag:"h2"}}),R=new re({props:{name:"class transformers.HunYuanMoEV1Model",anchor:"transformers.HunYuanMoEV1Model",parameters:[{name:"config",val:": HunYuanMoEV1Config"}],parametersDescription:[{anchor:"transformers.HunYuanMoEV1Model.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/hunyuan_v1_moe#transformers.HunYuanMoEV1Config">HunYuanMoEV1Config</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/hunyuan_v1_moe/modeling_hunyuan_v1_moe.py#L421"}}),Q=new re({props:{name:"forward",anchor:"transformers.HunYuanMoEV1Model.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.HunYuanMoEV1Model.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.HunYuanMoEV1Model.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.HunYuanMoEV1Model.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.HunYuanMoEV1Model.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.HunYuanMoEV1Model.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.HunYuanMoEV1Model.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"},{anchor:"transformers.HunYuanMoEV1Model.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/hunyuan_v1_moe/modeling_hunyuan_v1_moe.py#L438",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast"
>transformers.modeling_outputs.BaseModelOutputWithPast</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/hunyuan_v1_moe#transformers.HunYuanMoEV1Config"
>HunYuanMoEV1Config</a>) and inputs.</p>
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
`}}),P=new on({props:{$$slots:{default:[Cn]},$$scope:{ctx:V}}}),K=new N({props:{title:"HunYuanMoEV1ForCausalLM",local:"transformers.HunYuanMoEV1ForCausalLM",headingTag:"h2"}}),ee=new re({props:{name:"class transformers.HunYuanMoEV1ForCausalLM",anchor:"transformers.HunYuanMoEV1ForCausalLM",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.HunYuanMoEV1ForCausalLM.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/hunyuan_v1_moe#transformers.HunYuanMoEV1ForCausalLM">HunYuanMoEV1ForCausalLM</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/hunyuan_v1_moe/modeling_hunyuan_v1_moe.py#L500"}}),ne=new re({props:{name:"forward",anchor:"transformers.HunYuanMoEV1ForCausalLM.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"},{name:"logits_to_keep",val:": typing.Union[int, torch.Tensor] = 0"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.HunYuanMoEV1ForCausalLM.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.HunYuanMoEV1ForCausalLM.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.HunYuanMoEV1ForCausalLM.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.HunYuanMoEV1ForCausalLM.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.HunYuanMoEV1ForCausalLM.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.HunYuanMoEV1ForCausalLM.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.HunYuanMoEV1ForCausalLM.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.HunYuanMoEV1ForCausalLM.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"},{anchor:"transformers.HunYuanMoEV1ForCausalLM.forward.logits_to_keep",description:`<strong>logits_to_keep</strong> (<code>Union[int, torch.Tensor]</code>, defaults to <code>0</code>) &#x2014;
If an <code>int</code>, compute logits for the last <code>logits_to_keep</code> tokens. If <code>0</code>, calculate logits for all
<code>input_ids</code> (special case). Only last token logits are needed for generation, and calculating them only for that
token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
If a <code>torch.Tensor</code>, must be 1D corresponding to the indices to keep in the sequence length dimension.
This is useful when using packed tensor format (single dimension for batch and sequence length).`,name:"logits_to_keep"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/hunyuan_v1_moe/modeling_hunyuan_v1_moe.py#L514",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast"
>transformers.modeling_outputs.CausalLMOutputWithPast</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/hunyuan_v1_moe#transformers.HunYuanMoEV1Config"
>HunYuanMoEV1Config</a>) and inputs.</p>
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
`}}),W=new on({props:{$$slots:{default:[xn]},$$scope:{ctx:V}}}),D=new wn({props:{anchor:"transformers.HunYuanMoEV1ForCausalLM.forward.example",$$slots:{default:[Hn]},$$scope:{ctx:V}}}),oe=new N({props:{title:"HunYuanMoEV1ForSequenceClassification",local:"transformers.HunYuanMoEV1ForSequenceClassification",headingTag:"h2"}}),te=new re({props:{name:"class transformers.HunYuanMoEV1ForSequenceClassification",anchor:"transformers.HunYuanMoEV1ForSequenceClassification",parameters:[{name:"config",val:""}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/hunyuan_v1_moe/modeling_hunyuan_v1_moe.py#L575"}}),se=new re({props:{name:"forward",anchor:"transformers.HunYuanMoEV1ForSequenceClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.HunYuanMoEV1ForSequenceClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.HunYuanMoEV1ForSequenceClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.HunYuanMoEV1ForSequenceClassification.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.HunYuanMoEV1ForSequenceClassification.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.HunYuanMoEV1ForSequenceClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.HunYuanMoEV1ForSequenceClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.HunYuanMoEV1ForSequenceClassification.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
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
`}}),O=new on({props:{$$slots:{default:[Vn]},$$scope:{ctx:V}}}),ae=new $n({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/hunyuan_v1_moe.md"}}),{c(){t=c("meta"),y=s(),d=c("p"),v=s(),u($.$$.fragment),_=s(),u(x.$$.fragment),ye=s(),J=c("p"),J.textContent=tn,Me=s(),u(S.$$.fragment),ke=s(),A=c("p"),A.textContent=sn,Te=s(),u(B.$$.fragment),we=s(),U=c("p"),U.textContent=an,$e=s(),u(Z.$$.fragment),Ce=s(),H=c("div"),u(G.$$.fragment),We=s(),ie=c("p"),ie.innerHTML=rn,De=s(),de=c("p"),de.innerHTML=dn,xe=s(),u(X.$$.fragment),He=s(),M=c("div"),u(R.$$.fragment),Oe=s(),ce=c("p"),ce.textContent=cn,je=s(),le=c("p"),le.innerHTML=ln,Ne=s(),ue=c("p"),ue.innerHTML=un,Je=s(),E=c("div"),u(Q.$$.fragment),Se=s(),pe=c("p"),pe.innerHTML=pn,Ae=s(),u(P.$$.fragment),Ve=s(),u(K.$$.fragment),Ee=s(),k=c("div"),u(ee.$$.fragment),Be=s(),me=c("p"),me.textContent=mn,Ue=s(),he=c("p"),he.innerHTML=hn,Ze=s(),fe=c("p"),fe.innerHTML=fn,Ge=s(),C=c("div"),u(ne.$$.fragment),Xe=s(),ge=c("p"),ge.innerHTML=gn,Re=s(),u(W.$$.fragment),Qe=s(),u(D.$$.fragment),Ye=s(),u(oe.$$.fragment),ze=s(),z=c("div"),u(te.$$.fragment),Ke=s(),Y=c("div"),u(se.$$.fragment),en=s(),_e=c("p"),_e.innerHTML=_n,nn=s(),u(O.$$.fragment),Fe=s(),u(ae.$$.fragment),Le=s(),be=c("p"),this.h()},l(e){const n=kn("svelte-u9bgzb",document.head);t=l(n,"META",{name:!0,content:!0}),n.forEach(o),y=a(e),d=l(e,"P",{}),q(d).forEach(o),v=a(e),p($.$$.fragment,e),_=a(e),p(x.$$.fragment,e),ye=a(e),J=l(e,"P",{"data-svelte-h":!0}),b(J)!=="svelte-u40g91"&&(J.textContent=tn),Me=a(e),p(S.$$.fragment,e),ke=a(e),A=l(e,"P",{"data-svelte-h":!0}),b(A)!=="svelte-u40g91"&&(A.textContent=sn),Te=a(e),p(B.$$.fragment,e),we=a(e),U=l(e,"P",{"data-svelte-h":!0}),b(U)!=="svelte-u40g91"&&(U.textContent=an),$e=a(e),p(Z.$$.fragment,e),Ce=a(e),H=l(e,"DIV",{class:!0});var F=q(H);p(G.$$.fragment,F),We=a(F),ie=l(F,"P",{"data-svelte-h":!0}),b(ie)!=="svelte-1cl3ypd"&&(ie.innerHTML=rn),De=a(F),de=l(F,"P",{"data-svelte-h":!0}),b(de)!=="svelte-1ek1ss9"&&(de.innerHTML=dn),F.forEach(o),xe=a(e),p(X.$$.fragment,e),He=a(e),M=l(e,"DIV",{class:!0});var T=q(M);p(R.$$.fragment,T),Oe=a(T),ce=l(T,"P",{"data-svelte-h":!0}),b(ce)!=="svelte-2tze50"&&(ce.textContent=cn),je=a(T),le=l(T,"P",{"data-svelte-h":!0}),b(le)!=="svelte-q52n56"&&(le.innerHTML=ln),Ne=a(T),ue=l(T,"P",{"data-svelte-h":!0}),b(ue)!=="svelte-hswkmf"&&(ue.innerHTML=un),Je=a(T),E=l(T,"DIV",{class:!0});var L=q(E);p(Q.$$.fragment,L),Se=a(L),pe=l(L,"P",{"data-svelte-h":!0}),b(pe)!=="svelte-1b1pge1"&&(pe.innerHTML=pn),Ae=a(L),p(P.$$.fragment,L),L.forEach(o),T.forEach(o),Ve=a(e),p(K.$$.fragment,e),Ee=a(e),k=l(e,"DIV",{class:!0});var w=q(k);p(ee.$$.fragment,w),Be=a(w),me=l(w,"P",{"data-svelte-h":!0}),b(me)!=="svelte-o7jq73"&&(me.textContent=mn),Ue=a(w),he=l(w,"P",{"data-svelte-h":!0}),b(he)!=="svelte-q52n56"&&(he.innerHTML=hn),Ze=a(w),fe=l(w,"P",{"data-svelte-h":!0}),b(fe)!=="svelte-hswkmf"&&(fe.innerHTML=fn),Ge=a(w),C=l(w,"DIV",{class:!0});var j=q(C);p(ne.$$.fragment,j),Xe=a(j),ge=l(j,"P",{"data-svelte-h":!0}),b(ge)!=="svelte-1xmcylp"&&(ge.innerHTML=gn),Re=a(j),p(W.$$.fragment,j),Qe=a(j),p(D.$$.fragment,j),j.forEach(o),w.forEach(o),Ye=a(e),p(oe.$$.fragment,e),ze=a(e),z=l(e,"DIV",{class:!0});var Ie=q(z);p(te.$$.fragment,Ie),Ke=a(Ie),Y=l(Ie,"DIV",{class:!0});var ve=q(Y);p(se.$$.fragment,ve),en=a(ve),_e=l(ve,"P",{"data-svelte-h":!0}),b(_e)!=="svelte-1sal4ui"&&(_e.innerHTML=_n),nn=a(ve),p(O.$$.fragment,ve),ve.forEach(o),Ie.forEach(o),Fe=a(e),p(ae.$$.fragment,e),Le=a(e),be=l(e,"P",{}),q(be).forEach(o),this.h()},h(){I(t,"name","hf:doc:metadata"),I(t,"content",Yn),I(H,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),I(E,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),I(M,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),I(C,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),I(k,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),I(Y,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),I(z,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,n){r(document.head,t),i(e,y,n),i(e,d,n),i(e,v,n),m($,e,n),i(e,_,n),m(x,e,n),i(e,ye,n),i(e,J,n),i(e,Me,n),m(S,e,n),i(e,ke,n),i(e,A,n),i(e,Te,n),m(B,e,n),i(e,we,n),i(e,U,n),i(e,$e,n),m(Z,e,n),i(e,Ce,n),i(e,H,n),m(G,H,null),r(H,We),r(H,ie),r(H,De),r(H,de),i(e,xe,n),m(X,e,n),i(e,He,n),i(e,M,n),m(R,M,null),r(M,Oe),r(M,ce),r(M,je),r(M,le),r(M,Ne),r(M,ue),r(M,Je),r(M,E),m(Q,E,null),r(E,Se),r(E,pe),r(E,Ae),m(P,E,null),i(e,Ve,n),m(K,e,n),i(e,Ee,n),i(e,k,n),m(ee,k,null),r(k,Be),r(k,me),r(k,Ue),r(k,he),r(k,Ze),r(k,fe),r(k,Ge),r(k,C),m(ne,C,null),r(C,Xe),r(C,ge),r(C,Re),m(W,C,null),r(C,Qe),m(D,C,null),i(e,Ye,n),m(oe,e,n),i(e,ze,n),i(e,z,n),m(te,z,null),r(z,Ke),r(z,Y),m(se,Y,null),r(Y,en),r(Y,_e),r(Y,nn),m(O,Y,null),i(e,Fe,n),m(ae,e,n),i(e,Le,n),i(e,be,n),qe=!0},p(e,[n]){const F={};n&2&&(F.$$scope={dirty:n,ctx:e}),P.$set(F);const T={};n&2&&(T.$$scope={dirty:n,ctx:e}),W.$set(T);const L={};n&2&&(L.$$scope={dirty:n,ctx:e}),D.$set(L);const w={};n&2&&(w.$$scope={dirty:n,ctx:e}),O.$set(w)},i(e){qe||(h($.$$.fragment,e),h(x.$$.fragment,e),h(S.$$.fragment,e),h(B.$$.fragment,e),h(Z.$$.fragment,e),h(G.$$.fragment,e),h(X.$$.fragment,e),h(R.$$.fragment,e),h(Q.$$.fragment,e),h(P.$$.fragment,e),h(K.$$.fragment,e),h(ee.$$.fragment,e),h(ne.$$.fragment,e),h(W.$$.fragment,e),h(D.$$.fragment,e),h(oe.$$.fragment,e),h(te.$$.fragment,e),h(se.$$.fragment,e),h(O.$$.fragment,e),h(ae.$$.fragment,e),qe=!0)},o(e){f($.$$.fragment,e),f(x.$$.fragment,e),f(S.$$.fragment,e),f(B.$$.fragment,e),f(Z.$$.fragment,e),f(G.$$.fragment,e),f(X.$$.fragment,e),f(R.$$.fragment,e),f(Q.$$.fragment,e),f(P.$$.fragment,e),f(K.$$.fragment,e),f(ee.$$.fragment,e),f(ne.$$.fragment,e),f(W.$$.fragment,e),f(D.$$.fragment,e),f(oe.$$.fragment,e),f(te.$$.fragment,e),f(se.$$.fragment,e),f(O.$$.fragment,e),f(ae.$$.fragment,e),qe=!1},d(e){e&&(o(y),o(d),o(v),o(_),o(ye),o(J),o(Me),o(ke),o(A),o(Te),o(we),o(U),o($e),o(Ce),o(H),o(xe),o(He),o(M),o(Ve),o(Ee),o(k),o(Ye),o(ze),o(z),o(Fe),o(Le),o(be)),o(t),g($,e),g(x,e),g(S,e),g(B,e),g(Z,e),g(G),g(X,e),g(R),g(Q),g(P),g(K,e),g(ee),g(ne),g(W),g(D),g(oe,e),g(te),g(se),g(O),g(ae,e)}}}const Yn='{"title":"HunYuanMoEV1","local":"hunyuanmoev1","sections":[{"title":"Overview","local":"overview","sections":[{"title":"Model Details","local":"model-details","sections":[],"depth":3}],"depth":2},{"title":"Usage tips","local":"usage-tips","sections":[],"depth":2},{"title":"HunYuanMoEV1Config","local":"transformers.HunYuanMoEV1Config","sections":[],"depth":2},{"title":"HunYuanMoEV1Model","local":"transformers.HunYuanMoEV1Model","sections":[],"depth":2},{"title":"HunYuanMoEV1ForCausalLM","local":"transformers.HunYuanMoEV1ForCausalLM","sections":[],"depth":2},{"title":"HunYuanMoEV1ForSequenceClassification","local":"transformers.HunYuanMoEV1ForSequenceClassification","sections":[],"depth":2}],"depth":1}';function zn(V){return bn(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class On extends yn{constructor(t){super(),Mn(this,t,zn,En,vn,{})}}export{On as component};
