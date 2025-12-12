import{s as fo,o as go,n as ve}from"../chunks/scheduler.18a86fab.js";import{S as _o,i as bo,g as c,s as a,r as u,A as vo,h as l,f as n,c as r,j,x as M,u as h,k as L,y as d,a as i,v as m,d as f,t as g,w as _}from"../chunks/index.98837b22.js";import{T as Ke}from"../chunks/Tip.77304350.js";import{D as re}from"../chunks/Docstring.a1ef7999.js";import{C as mo}from"../chunks/CodeBlock.8d0c2e8a.js";import{E as ho}from"../chunks/ExampleCodeBlock.8c3ee1f9.js";import{H as _e,E as Mo}from"../chunks/getInferenceSnippets.06c2775f.js";function yo($){let o,p;return o=new mo({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEpldE1vZU1vZGVsJTJDJTIwSmV0TW9lQ29uZmlnJTBBJTBBJTIzJTIwSW5pdGlhbGl6aW5nJTIwYSUyMEpldE1vZSUyMDRCJTIwc3R5bGUlMjBjb25maWd1cmF0aW9uJTBBY29uZmlndXJhdGlvbiUyMCUzRCUyMEpldE1vZUNvbmZpZygpJTBBJTBBJTIzJTIwSW5pdGlhbGl6aW5nJTIwYSUyMG1vZGVsJTIwZnJvbSUyMHRoZSUyMEpldE1vZSUyMDRCJTIwc3R5bGUlMjBjb25maWd1cmF0aW9uJTBBbW9kZWwlMjAlM0QlMjBKZXRNb2VNb2RlbChjb25maWd1cmF0aW9uKSUwQSUwQSUyMyUyMEFjY2Vzc2luZyUyMHRoZSUyMG1vZGVsJTIwY29uZmlndXJhdGlvbiUwQWNvbmZpZ3VyYXRpb24lMjAlM0QlMjBtb2RlbC5jb25maWc=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> JetMoeModel, JetMoeConfig

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a JetMoe 4B style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = JetMoeConfig()

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a model from the JetMoe 4B style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = JetMoeModel(configuration)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Accessing the model configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = model.config`,wrap:!1}}),{c(){u(o.$$.fragment)},l(s){h(o.$$.fragment,s)},m(s,b){m(o,s,b),p=!0},p:ve,i(s){p||(f(o.$$.fragment,s),p=!0)},o(s){g(o.$$.fragment,s),p=!1},d(s){_(o,s)}}}function To($){let o,p=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){o=c("p"),o.innerHTML=p},l(s){o=l(s,"P",{"data-svelte-h":!0}),M(o)!=="svelte-fincs2"&&(o.innerHTML=p)},m(s,b){i(s,o,b)},p:ve,d(s){s&&n(o)}}}function ko($){let o,p=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){o=c("p"),o.innerHTML=p},l(s){o=l(s,"P",{"data-svelte-h":!0}),M(o)!=="svelte-fincs2"&&(o.innerHTML=p)},m(s,b){i(s,o,b)},p:ve,d(s){s&&n(o)}}}function wo($){let o,p="Example:",s,b,x;return b=new mo({props:{code:"",highlighted:"",wrap:!1}}),{c(){o=c("p"),o.textContent=p,s=a(),u(b.$$.fragment)},l(v){o=l(v,"P",{"data-svelte-h":!0}),M(o)!=="svelte-11lpom8"&&(o.textContent=p),s=r(v),h(b.$$.fragment,v)},m(v,S){i(v,o,S),i(v,s,S),m(b,v,S),x=!0},p:ve,i(v){x||(f(b.$$.fragment,v),x=!0)},o(v){g(b.$$.fragment,v),x=!1},d(v){v&&(n(o),n(s)),_(b,v)}}}function $o($){let o,p=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){o=c("p"),o.innerHTML=p},l(s){o=l(s,"P",{"data-svelte-h":!0}),M(o)!=="svelte-fincs2"&&(o.innerHTML=p)},m(s,b){i(s,o,b)},p:ve,d(s){s&&n(o)}}}function xo($){let o,p,s,b,x,v="<em>This model was released on 2023-06-07 and added to Hugging Face Transformers on 2024-05-14.</em>",S,A,Me,D,eo='<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/> <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat"/> <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/>',ye,B,Te,U,oo=`<strong>JetMoe-8B</strong> is an 8B Mixture-of-Experts (MoE) language model developed by <a href="https://scholar.google.com.hk/citations?user=qff5rRYAAAAJ" rel="nofollow">Yikang Shen</a> and <a href="https://myshell.ai/" rel="nofollow">MyShell</a>.
JetMoe project aims to provide a LLaMA2-level performance and efficient language model with a limited budget.
To achieve this goal, JetMoe uses a sparsely activated architecture inspired by the <a href="https://huggingface.co/papers/2306.04640" rel="nofollow">ModuleFormer</a>.
Each JetMoe block consists of two MoE layers: Mixture of Attention Heads and Mixture of MLP Experts.
Given the input tokens, it activates a subset of its experts to process them.
This sparse activation schema enables JetMoe to achieve much better training throughput than similar size dense models.
The training throughput of JetMoe-8B is around 100B tokens per day on a cluster of 96 H100 GPUs with a straightforward 3-way pipeline parallelism strategy.`,ke,Z,to='This model was contributed by <a href="https://huggingface.co/YikangS" rel="nofollow">Yikang Shen</a>.',we,G,$e,y,R,Se,ie,no=`This is the configuration class to store the configuration of a <a href="/docs/transformers/v4.56.2/en/model_doc/jetmoe#transformers.JetMoeModel">JetMoeModel</a>. It is used to instantiate a
JetMoe model according to the specified arguments, defining the model architecture. Instantiating a configuration
with the defaults will yield a configuration of the JetMoe-4B.`,De,de,so='<a href="https://huggingface.co/jetmoe/jetmoe-8b" rel="nofollow">jetmoe/jetmoe-8b</a>',Oe,ce,ao=`Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> and can be used to control the model outputs. Read the
documentation from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> for more information.`,Ee,O,xe,V,Ce,T,Y,He,le,ro="The bare Jetmoe Model outputting raw hidden-states without any specific head on top.",We,pe,io=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Ne,ue,co=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Ae,z,Q,Be,he,lo='The <a href="/docs/transformers/v4.56.2/en/model_doc/jetmoe#transformers.JetMoeModel">JetMoeModel</a> forward method, overrides the <code>__call__</code> special method.',Ue,E,Je,X,ze,q,K,Ze,C,ee,Ge,me,po='The <a href="/docs/transformers/v4.56.2/en/model_doc/jetmoe#transformers.JetMoeForCausalLM">JetMoeForCausalLM</a> forward method, overrides the <code>__call__</code> special method.',Re,H,Ve,W,Fe,oe,Le,P,te,Ye,F,ne,Qe,fe,uo="The <code>GenericForSequenceClassification</code> forward method, overrides the <code>__call__</code> special method.",Xe,N,qe,se,Pe,be,Ie;return A=new _e({props:{title:"JetMoe",local:"jetmoe",headingTag:"h1"}}),B=new _e({props:{title:"Overview",local:"overview",headingTag:"h2"}}),G=new _e({props:{title:"JetMoeConfig",local:"transformers.JetMoeConfig",headingTag:"h2"}}),R=new re({props:{name:"class transformers.JetMoeConfig",anchor:"transformers.JetMoeConfig",parameters:[{name:"vocab_size",val:" = 32000"},{name:"hidden_size",val:" = 2048"},{name:"num_hidden_layers",val:" = 12"},{name:"num_key_value_heads",val:" = 16"},{name:"kv_channels",val:" = 128"},{name:"intermediate_size",val:" = 5632"},{name:"max_position_embeddings",val:" = 4096"},{name:"activation_function",val:" = 'silu'"},{name:"num_local_experts",val:" = 8"},{name:"num_experts_per_tok",val:" = 2"},{name:"output_router_logits",val:" = False"},{name:"aux_loss_coef",val:" = 0.01"},{name:"use_cache",val:" = True"},{name:"bos_token_id",val:" = 1"},{name:"eos_token_id",val:" = 2"},{name:"tie_word_embeddings",val:" = True"},{name:"rope_theta",val:" = 10000.0"},{name:"rms_norm_eps",val:" = 1e-06"},{name:"initializer_range",val:" = 0.01"},{name:"attention_dropout",val:" = 0.0"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.JetMoeConfig.vocab_size",description:`<strong>vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 32000) &#x2014;
Vocabulary size of the JetMoe model. Defines the number of different tokens that can be represented by the
<code>inputs_ids</code> passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/jetmoe#transformers.JetMoeModel">JetMoeModel</a>`,name:"vocab_size"},{anchor:"transformers.JetMoeConfig.hidden_size",description:`<strong>hidden_size</strong> (<code>int</code>, <em>optional</em>, defaults to 2048) &#x2014;
Dimension of the hidden representations.`,name:"hidden_size"},{anchor:"transformers.JetMoeConfig.num_hidden_layers",description:`<strong>num_hidden_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 12) &#x2014;
Number of hidden layers in the Transformer encoder.`,name:"num_hidden_layers"},{anchor:"transformers.JetMoeConfig.num_key_value_heads",description:`<strong>num_key_value_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 16) &#x2014;
Number of attention heads for each key and value in the Transformer encoder.`,name:"num_key_value_heads"},{anchor:"transformers.JetMoeConfig.kv_channels",description:`<strong>kv_channels</strong> (<code>int</code>, <em>optional</em>, defaults to 128) &#x2014;
Defines the number of channels for the key and value tensors.`,name:"kv_channels"},{anchor:"transformers.JetMoeConfig.intermediate_size",description:`<strong>intermediate_size</strong> (<code>int</code>, <em>optional</em>, defaults to 5632) &#x2014;
Dimension of the MLP representations.`,name:"intermediate_size"},{anchor:"transformers.JetMoeConfig.max_position_embeddings",description:`<strong>max_position_embeddings</strong> (<code>int</code>, <em>optional</em>, defaults to 4096) &#x2014;
The maximum sequence length that this model might ever be used with. JetMoe&#x2019;s attention allows sequence of
up to 4096 tokens.`,name:"max_position_embeddings"},{anchor:"transformers.JetMoeConfig.activation_function",description:`<strong>activation_function</strong> (<code>string</code>, <em>optional</em>, defaults to <code>&quot;silu&quot;</code>) &#x2014;
Defines the activation function for MLP experts.`,name:"activation_function"},{anchor:"transformers.JetMoeConfig.num_local_experts",description:`<strong>num_local_experts</strong> (<code>int</code>, <em>optional</em>, defaults to 8) &#x2014;
Defines the number of experts in the MoE and MoA.`,name:"num_local_experts"},{anchor:"transformers.JetMoeConfig.num_experts_per_tok",description:"<strong>num_experts_per_tok</strong> (`int, <em>optional</em>, defaults to 2) &#x2014;\nThe number of experts to route per-token and for MoE and MoA.",name:"num_experts_per_tok"},{anchor:"transformers.JetMoeConfig.output_router_logits",description:`<strong>output_router_logits</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not the router logits should be returned by the model. Enabling this will also
allow the model to output the auxiliary loss.`,name:"output_router_logits"},{anchor:"transformers.JetMoeConfig.aux_loss_coef",description:`<strong>aux_loss_coef</strong> (<code>float</code>, <em>optional</em>, defaults to 0.01) &#x2014;
The coefficient for the auxiliary loss.`,name:"aux_loss_coef"},{anchor:"transformers.JetMoeConfig.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not the model should return the last key/values attentions (not used by all models). Only
relevant if <code>config.is_decoder=True</code>.`,name:"use_cache"},{anchor:"transformers.JetMoeConfig.bos_token_id",description:`<strong>bos_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 1) &#x2014;
The id of the &#x201C;beginning-of-sequence&#x201D; token.`,name:"bos_token_id"},{anchor:"transformers.JetMoeConfig.eos_token_id",description:`<strong>eos_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 2) &#x2014;
The id of the &#x201C;end-of-sequence&#x201D; token.`,name:"eos_token_id"},{anchor:"transformers.JetMoeConfig.tie_word_embeddings",description:`<strong>tie_word_embeddings</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether the model&#x2019;s input and output word embeddings should be tied.`,name:"tie_word_embeddings"},{anchor:"transformers.JetMoeConfig.rope_theta",description:`<strong>rope_theta</strong> (<code>float</code>, <em>optional</em>, defaults to 10000.0) &#x2014;
The base period of the RoPE embeddings.`,name:"rope_theta"},{anchor:"transformers.JetMoeConfig.rms_norm_eps",description:`<strong>rms_norm_eps</strong> (<code>float</code>, <em>optional</em>, defaults to 1e-06) &#x2014;
The epsilon used by the rms normalization layers.`,name:"rms_norm_eps"},{anchor:"transformers.JetMoeConfig.initializer_range",description:`<strong>initializer_range</strong> (<code>float</code>, <em>optional</em>, defaults to 0.01) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.`,name:"initializer_range"},{anchor:"transformers.JetMoeConfig.attention_dropout",description:`<strong>attention_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The dropout ratio for the attention probabilities.`,name:"attention_dropout"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/jetmoe/configuration_jetmoe.py#L24"}}),O=new ho({props:{anchor:"transformers.JetMoeConfig.example",$$slots:{default:[yo]},$$scope:{ctx:$}}}),V=new _e({props:{title:"JetMoeModel",local:"transformers.JetMoeModel",headingTag:"h2"}}),Y=new re({props:{name:"class transformers.JetMoeModel",anchor:"transformers.JetMoeModel",parameters:[{name:"config",val:": JetMoeConfig"}],parametersDescription:[{anchor:"transformers.JetMoeModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/jetmoe#transformers.JetMoeConfig">JetMoeConfig</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/jetmoe/modeling_jetmoe.py#L865"}}),Q=new re({props:{name:"forward",anchor:"transformers.JetMoeModel.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Union[list[torch.FloatTensor], transformers.cache_utils.Cache, NoneType] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"output_router_logits",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"}],parametersDescription:[{anchor:"transformers.JetMoeModel.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.JetMoeModel.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.JetMoeModel.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.JetMoeModel.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>Union[list[torch.FloatTensor], ~cache_utils.Cache, NoneType]</code>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.JetMoeModel.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.JetMoeModel.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.JetMoeModel.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.JetMoeModel.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.JetMoeModel.forward.output_router_logits",description:`<strong>output_router_logits</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the logits of all the routers. They are useful for computing the router loss, and
should not be returned during inference.`,name:"output_router_logits"},{anchor:"transformers.JetMoeModel.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/jetmoe/modeling_jetmoe.py#L888",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <code>transformers.modeling_outputs.MoeModelOutputWithPast</code> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/jetmoe#transformers.JetMoeConfig"
>JetMoeConfig</a>) and inputs.</p>
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
`}}),E=new Ke({props:{$$slots:{default:[To]},$$scope:{ctx:$}}}),X=new _e({props:{title:"JetMoeForCausalLM",local:"transformers.JetMoeForCausalLM",headingTag:"h2"}}),K=new re({props:{name:"class transformers.JetMoeForCausalLM",anchor:"transformers.JetMoeForCausalLM",parameters:[{name:"config",val:""}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/jetmoe/modeling_jetmoe.py#L1112"}}),ee=new re({props:{name:"forward",anchor:"transformers.JetMoeForCausalLM.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"output_router_logits",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"},{name:"logits_to_keep",val:": typing.Union[int, torch.Tensor] = 0"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.JetMoeForCausalLM.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.JetMoeForCausalLM.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.JetMoeForCausalLM.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.JetMoeForCausalLM.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.JetMoeForCausalLM.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.JetMoeForCausalLM.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.JetMoeForCausalLM.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.JetMoeForCausalLM.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.JetMoeForCausalLM.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.JetMoeForCausalLM.forward.output_router_logits",description:`<strong>output_router_logits</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the logits of all the routers. They are useful for computing the router loss, and
should not be returned during inference.`,name:"output_router_logits"},{anchor:"transformers.JetMoeForCausalLM.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"},{anchor:"transformers.JetMoeForCausalLM.forward.logits_to_keep",description:`<strong>logits_to_keep</strong> (<code>Union[int, torch.Tensor]</code>, defaults to <code>0</code>) &#x2014;
If an <code>int</code>, compute logits for the last <code>logits_to_keep</code> tokens. If <code>0</code>, calculate logits for all
<code>input_ids</code> (special case). Only last token logits are needed for generation, and calculating them only for that
token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
If a <code>torch.Tensor</code>, must be 1D corresponding to the indices to keep in the sequence length dimension.
This is useful when using packed tensor format (single dimension for batch and sequence length).`,name:"logits_to_keep"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/jetmoe/modeling_jetmoe.py#L1126",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <code>transformers.modeling_outputs.MoeCausalLMOutputWithPast</code> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/jetmoe#transformers.JetMoeConfig"
>JetMoeConfig</a>) and inputs.</p>
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
`}}),H=new Ke({props:{$$slots:{default:[ko]},$$scope:{ctx:$}}}),W=new ho({props:{anchor:"transformers.JetMoeForCausalLM.forward.example",$$slots:{default:[wo]},$$scope:{ctx:$}}}),oe=new _e({props:{title:"JetMoeForSequenceClassification",local:"transformers.JetMoeForSequenceClassification",headingTag:"h2"}}),te=new re({props:{name:"class transformers.JetMoeForSequenceClassification",anchor:"transformers.JetMoeForSequenceClassification",parameters:[{name:"config",val:""}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/jetmoe/modeling_jetmoe.py#L1205"}}),ne=new re({props:{name:"forward",anchor:"transformers.JetMoeForSequenceClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.JetMoeForSequenceClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.JetMoeForSequenceClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.JetMoeForSequenceClassification.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.JetMoeForSequenceClassification.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.JetMoeForSequenceClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.JetMoeForSequenceClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.JetMoeForSequenceClassification.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
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
`}}),N=new Ke({props:{$$slots:{default:[$o]},$$scope:{ctx:$}}}),se=new Mo({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/jetmoe.md"}}),{c(){o=c("meta"),p=a(),s=c("p"),b=a(),x=c("p"),x.innerHTML=v,S=a(),u(A.$$.fragment),Me=a(),D=c("div"),D.innerHTML=eo,ye=a(),u(B.$$.fragment),Te=a(),U=c("p"),U.innerHTML=oo,ke=a(),Z=c("p"),Z.innerHTML=to,we=a(),u(G.$$.fragment),$e=a(),y=c("div"),u(R.$$.fragment),Se=a(),ie=c("p"),ie.innerHTML=no,De=a(),de=c("p"),de.innerHTML=so,Oe=a(),ce=c("p"),ce.innerHTML=ao,Ee=a(),u(O.$$.fragment),xe=a(),u(V.$$.fragment),Ce=a(),T=c("div"),u(Y.$$.fragment),He=a(),le=c("p"),le.textContent=ro,We=a(),pe=c("p"),pe.innerHTML=io,Ne=a(),ue=c("p"),ue.innerHTML=co,Ae=a(),z=c("div"),u(Q.$$.fragment),Be=a(),he=c("p"),he.innerHTML=lo,Ue=a(),u(E.$$.fragment),Je=a(),u(X.$$.fragment),ze=a(),q=c("div"),u(K.$$.fragment),Ze=a(),C=c("div"),u(ee.$$.fragment),Ge=a(),me=c("p"),me.innerHTML=po,Re=a(),u(H.$$.fragment),Ve=a(),u(W.$$.fragment),Fe=a(),u(oe.$$.fragment),Le=a(),P=c("div"),u(te.$$.fragment),Ye=a(),F=c("div"),u(ne.$$.fragment),Qe=a(),fe=c("p"),fe.innerHTML=uo,Xe=a(),u(N.$$.fragment),qe=a(),u(se.$$.fragment),Pe=a(),be=c("p"),this.h()},l(e){const t=vo("svelte-u9bgzb",document.head);o=l(t,"META",{name:!0,content:!0}),t.forEach(n),p=r(e),s=l(e,"P",{}),j(s).forEach(n),b=r(e),x=l(e,"P",{"data-svelte-h":!0}),M(x)!=="svelte-lo722o"&&(x.innerHTML=v),S=r(e),h(A.$$.fragment,e),Me=r(e),D=l(e,"DIV",{class:!0,"data-svelte-h":!0}),M(D)!=="svelte-b95w5j"&&(D.innerHTML=eo),ye=r(e),h(B.$$.fragment,e),Te=r(e),U=l(e,"P",{"data-svelte-h":!0}),M(U)!=="svelte-jj72cb"&&(U.innerHTML=oo),ke=r(e),Z=l(e,"P",{"data-svelte-h":!0}),M(Z)!=="svelte-8foy02"&&(Z.innerHTML=to),we=r(e),h(G.$$.fragment,e),$e=r(e),y=l(e,"DIV",{class:!0});var k=j(y);h(R.$$.fragment,k),Se=r(k),ie=l(k,"P",{"data-svelte-h":!0}),M(ie)!=="svelte-kblae1"&&(ie.innerHTML=no),De=r(k),de=l(k,"P",{"data-svelte-h":!0}),M(de)!=="svelte-aqfqqx"&&(de.innerHTML=so),Oe=r(k),ce=l(k,"P",{"data-svelte-h":!0}),M(ce)!=="svelte-1ek1ss9"&&(ce.innerHTML=ao),Ee=r(k),h(O.$$.fragment,k),k.forEach(n),xe=r(e),h(V.$$.fragment,e),Ce=r(e),T=l(e,"DIV",{class:!0});var w=j(T);h(Y.$$.fragment,w),He=r(w),le=l(w,"P",{"data-svelte-h":!0}),M(le)!=="svelte-14wtgkq"&&(le.textContent=ro),We=r(w),pe=l(w,"P",{"data-svelte-h":!0}),M(pe)!=="svelte-q52n56"&&(pe.innerHTML=io),Ne=r(w),ue=l(w,"P",{"data-svelte-h":!0}),M(ue)!=="svelte-hswkmf"&&(ue.innerHTML=co),Ae=r(w),z=l(w,"DIV",{class:!0});var I=j(z);h(Q.$$.fragment,I),Be=r(I),he=l(I,"P",{"data-svelte-h":!0}),M(he)!=="svelte-e2pczp"&&(he.innerHTML=lo),Ue=r(I),h(E.$$.fragment,I),I.forEach(n),w.forEach(n),Je=r(e),h(X.$$.fragment,e),ze=r(e),q=l(e,"DIV",{class:!0});var ae=j(q);h(K.$$.fragment,ae),Ze=r(ae),C=l(ae,"DIV",{class:!0});var J=j(C);h(ee.$$.fragment,J),Ge=r(J),me=l(J,"P",{"data-svelte-h":!0}),M(me)!=="svelte-nr7b41"&&(me.innerHTML=po),Re=r(J),h(H.$$.fragment,J),Ve=r(J),h(W.$$.fragment,J),J.forEach(n),ae.forEach(n),Fe=r(e),h(oe.$$.fragment,e),Le=r(e),P=l(e,"DIV",{class:!0});var je=j(P);h(te.$$.fragment,je),Ye=r(je),F=l(je,"DIV",{class:!0});var ge=j(F);h(ne.$$.fragment,ge),Qe=r(ge),fe=l(ge,"P",{"data-svelte-h":!0}),M(fe)!=="svelte-1sal4ui"&&(fe.innerHTML=uo),Xe=r(ge),h(N.$$.fragment,ge),ge.forEach(n),je.forEach(n),qe=r(e),h(se.$$.fragment,e),Pe=r(e),be=l(e,"P",{}),j(be).forEach(n),this.h()},h(){L(o,"name","hf:doc:metadata"),L(o,"content",Co),L(D,"class","flex flex-wrap space-x-1"),L(y,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),L(z,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),L(T,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),L(C,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),L(q,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),L(F,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),L(P,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,t){d(document.head,o),i(e,p,t),i(e,s,t),i(e,b,t),i(e,x,t),i(e,S,t),m(A,e,t),i(e,Me,t),i(e,D,t),i(e,ye,t),m(B,e,t),i(e,Te,t),i(e,U,t),i(e,ke,t),i(e,Z,t),i(e,we,t),m(G,e,t),i(e,$e,t),i(e,y,t),m(R,y,null),d(y,Se),d(y,ie),d(y,De),d(y,de),d(y,Oe),d(y,ce),d(y,Ee),m(O,y,null),i(e,xe,t),m(V,e,t),i(e,Ce,t),i(e,T,t),m(Y,T,null),d(T,He),d(T,le),d(T,We),d(T,pe),d(T,Ne),d(T,ue),d(T,Ae),d(T,z),m(Q,z,null),d(z,Be),d(z,he),d(z,Ue),m(E,z,null),i(e,Je,t),m(X,e,t),i(e,ze,t),i(e,q,t),m(K,q,null),d(q,Ze),d(q,C),m(ee,C,null),d(C,Ge),d(C,me),d(C,Re),m(H,C,null),d(C,Ve),m(W,C,null),i(e,Fe,t),m(oe,e,t),i(e,Le,t),i(e,P,t),m(te,P,null),d(P,Ye),d(P,F),m(ne,F,null),d(F,Qe),d(F,fe),d(F,Xe),m(N,F,null),i(e,qe,t),m(se,e,t),i(e,Pe,t),i(e,be,t),Ie=!0},p(e,[t]){const k={};t&2&&(k.$$scope={dirty:t,ctx:e}),O.$set(k);const w={};t&2&&(w.$$scope={dirty:t,ctx:e}),E.$set(w);const I={};t&2&&(I.$$scope={dirty:t,ctx:e}),H.$set(I);const ae={};t&2&&(ae.$$scope={dirty:t,ctx:e}),W.$set(ae);const J={};t&2&&(J.$$scope={dirty:t,ctx:e}),N.$set(J)},i(e){Ie||(f(A.$$.fragment,e),f(B.$$.fragment,e),f(G.$$.fragment,e),f(R.$$.fragment,e),f(O.$$.fragment,e),f(V.$$.fragment,e),f(Y.$$.fragment,e),f(Q.$$.fragment,e),f(E.$$.fragment,e),f(X.$$.fragment,e),f(K.$$.fragment,e),f(ee.$$.fragment,e),f(H.$$.fragment,e),f(W.$$.fragment,e),f(oe.$$.fragment,e),f(te.$$.fragment,e),f(ne.$$.fragment,e),f(N.$$.fragment,e),f(se.$$.fragment,e),Ie=!0)},o(e){g(A.$$.fragment,e),g(B.$$.fragment,e),g(G.$$.fragment,e),g(R.$$.fragment,e),g(O.$$.fragment,e),g(V.$$.fragment,e),g(Y.$$.fragment,e),g(Q.$$.fragment,e),g(E.$$.fragment,e),g(X.$$.fragment,e),g(K.$$.fragment,e),g(ee.$$.fragment,e),g(H.$$.fragment,e),g(W.$$.fragment,e),g(oe.$$.fragment,e),g(te.$$.fragment,e),g(ne.$$.fragment,e),g(N.$$.fragment,e),g(se.$$.fragment,e),Ie=!1},d(e){e&&(n(p),n(s),n(b),n(x),n(S),n(Me),n(D),n(ye),n(Te),n(U),n(ke),n(Z),n(we),n($e),n(y),n(xe),n(Ce),n(T),n(Je),n(ze),n(q),n(Fe),n(Le),n(P),n(qe),n(Pe),n(be)),n(o),_(A,e),_(B,e),_(G,e),_(R),_(O),_(V,e),_(Y),_(Q),_(E),_(X,e),_(K),_(ee),_(H),_(W),_(oe,e),_(te),_(ne),_(N),_(se,e)}}}const Co='{"title":"JetMoe","local":"jetmoe","sections":[{"title":"Overview","local":"overview","sections":[],"depth":2},{"title":"JetMoeConfig","local":"transformers.JetMoeConfig","sections":[],"depth":2},{"title":"JetMoeModel","local":"transformers.JetMoeModel","sections":[],"depth":2},{"title":"JetMoeForCausalLM","local":"transformers.JetMoeForCausalLM","sections":[],"depth":2},{"title":"JetMoeForSequenceClassification","local":"transformers.JetMoeForSequenceClassification","sections":[],"depth":2}],"depth":1}';function Jo($){return go(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class So extends _o{constructor(o){super(),bo(this,o,Jo,xo,fo,{})}}export{So as component};
