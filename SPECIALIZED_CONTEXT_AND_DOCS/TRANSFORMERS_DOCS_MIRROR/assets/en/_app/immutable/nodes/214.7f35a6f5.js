import{s as Nt,o as Rt,n as ze}from"../chunks/scheduler.18a86fab.js";import{S as Ut,i as Dt,g as c,s as a,r as p,m as Ht,A as At,h as d,f as o,c as r,j as F,x as _,u,n as St,k as z,l as jt,y as i,a as s,v as h,d as m,t as f,w as g}from"../chunks/index.98837b22.js";import{T as Ke}from"../chunks/Tip.77304350.js";import{D as S}from"../chunks/Docstring.a1ef7999.js";import{C as Wt}from"../chunks/CodeBlock.8d0c2e8a.js";import{E as Bt}from"../chunks/ExampleCodeBlock.8c3ee1f9.js";import{H as fe,E as Vt}from"../chunks/getInferenceSnippets.06c2775f.js";function Jt(M){let n,T=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){n=c("p"),n.innerHTML=T},l(l){n=d(l,"P",{"data-svelte-h":!0}),_(n)!=="svelte-fincs2"&&(n.innerHTML=T)},m(l,v){s(l,n,v)},p:ze,d(l){l&&o(n)}}}function Xt(M){let n,T=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){n=c("p"),n.innerHTML=T},l(l){n=d(l,"P",{"data-svelte-h":!0}),_(n)!=="svelte-fincs2"&&(n.innerHTML=T)},m(l,v){s(l,n,v)},p:ze,d(l){l&&o(n)}}}function Yt(M){let n,T="Example:",l,v,C;return v=new Wt({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBHcHRPc3NGb3JDYXVzYWxMTSUwQSUwQW1vZGVsJTIwJTNEJTIwR3B0T3NzRm9yQ2F1c2FsTE0uZnJvbV9wcmV0cmFpbmVkKCUyMm1pc3RyYWxhaSUyRkdwdE9zcy04eDdCLXYwLjElMjIpJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIybWlzdHJhbGFpJTJGR3B0T3NzLTh4N0ItdjAuMSUyMiklMEElMEFwcm9tcHQlMjAlM0QlMjAlMjJIZXklMkMlMjBhcmUlMjB5b3UlMjBjb25zY2lvdXMlM0YlMjBDYW4lMjB5b3UlMjB0YWxrJTIwdG8lMjBtZSUzRiUyMiUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplcihwcm9tcHQlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQSUwQSUyMyUyMEdlbmVyYXRlJTBBZ2VuZXJhdGVfaWRzJTIwJTNEJTIwbW9kZWwuZ2VuZXJhdGUoaW5wdXRzLmlucHV0X2lkcyUyQyUyMG1heF9sZW5ndGglM0QzMCklMEF0b2tlbml6ZXIuYmF0Y2hfZGVjb2RlKGdlbmVyYXRlX2lkcyUyQyUyMHNraXBfc3BlY2lhbF90b2tlbnMlM0RUcnVlJTJDJTIwY2xlYW5fdXBfdG9rZW5pemF0aW9uX3NwYWNlcyUzREZhbHNlKSU1QjAlNUQ=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, GptOssForCausalLM

<span class="hljs-meta">&gt;&gt;&gt; </span>model = GptOssForCausalLM.from_pretrained(<span class="hljs-string">&quot;mistralai/GptOss-8x7B-v0.1&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;mistralai/GptOss-8x7B-v0.1&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>prompt = <span class="hljs-string">&quot;Hey, are you conscious? Can you talk to me?&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(prompt, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Generate</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>generate_ids = model.generate(inputs.input_ids, max_length=<span class="hljs-number">30</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer.batch_decode(generate_ids, skip_special_tokens=<span class="hljs-literal">True</span>, clean_up_tokenization_spaces=<span class="hljs-literal">False</span>)[<span class="hljs-number">0</span>]
<span class="hljs-string">&quot;Hey, are you conscious? Can you talk to me?\\nI&#x27;m not conscious, but I can talk to you.&quot;</span>`,wrap:!1}}),{c(){n=c("p"),n.textContent=T,l=a(),p(v.$$.fragment)},l(b){n=d(b,"P",{"data-svelte-h":!0}),_(n)!=="svelte-11lpom8"&&(n.textContent=T),l=r(b),u(v.$$.fragment,b)},m(b,N){s(b,n,N),s(b,l,N),h(v,b,N),C=!0},p:ze,i(b){C||(m(v.$$.fragment,b),C=!0)},o(b){f(v.$$.fragment,b),C=!1},d(b){b&&(o(n),o(l)),g(v,b)}}}function Zt(M){let n,T=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){n=c("p"),n.innerHTML=T},l(l){n=d(l,"P",{"data-svelte-h":!0}),_(n)!=="svelte-fincs2"&&(n.innerHTML=T)},m(l,v){s(l,n,v)},p:ze,d(l){l&&o(n)}}}function Qt(M){let n,T=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){n=c("p"),n.innerHTML=T},l(l){n=d(l,"P",{"data-svelte-h":!0}),_(n)!=="svelte-fincs2"&&(n.innerHTML=T)},m(l,v){s(l,n,v)},p:ze,d(l){l&&o(n)}}}function Kt(M){let n,T,l,v,C,b="<em>This model was released on 2025-08-05 and added to Hugging Face Transformers on 2025-08-05.</em>",N,R,bt='<div class="flex flex-wrap space-x-1"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/> <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat"/> <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/></div>',Ge,B,Le,V,qe,J,yt='The GptOss model was proposed in <a href="https://openai.com/index/introducing-gpt-oss/" rel="nofollow">blog post</a> by &lt;INSERT AUTHORS HERE&gt;.',Ie,X,kt="The abstract from the paper is the following:",Pe,Y,wt="<em>&lt;INSERT PAPER ABSTRACT HERE&gt;</em>",Ee,Z,$t="Tips:",He,Q,Mt=`This model was contributed by [INSERT YOUR HF USERNAME HERE](<a href="https://huggingface.co/" rel="nofollow">https://huggingface.co/</a>&lt;INSERT YOUR HF USERNAME HERE&gt;).
The original code can be found <a href="INSERT%20LINK%20TO%20GITHUB%20REPO%20HERE">here</a>.`,Se,K,Ne,I,ee,et,ge,Ct=`This will yield a configuration to that of the BERT
<a href="https://huggingface.co/google-bert/bert-base-uncased" rel="nofollow">google-bert/bert-base-uncased</a> architecture.`,Re,te,Ue,y,oe,tt,_e,xt="The bare Gpt Oss Model outputting raw hidden-states without any specific head on top.",ot,ve,Ot=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,nt,Te,Ft=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,st,G,ne,at,be,zt='The <a href="/docs/transformers/v4.56.2/en/model_doc/gpt_oss#transformers.GptOssModel">GptOssModel</a> forward method, overrides the <code>__call__</code> special method.',rt,U,De,se,Ae,k,ae,it,ye,Gt="The Gpt Oss Model for causal language modeling.",ct,ke,Lt=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,dt,we,qt=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,lt,x,re,pt,$e,It='The <a href="/docs/transformers/v4.56.2/en/model_doc/gpt_oss#transformers.GptOssForCausalLM">GptOssForCausalLM</a> forward method, overrides the <code>__call__</code> special method.',ut,D,ht,A,je,ie,We,P,ce,mt,L,de,ft,Me,Pt="The <code>GenericForSequenceClassification</code> forward method, overrides the <code>__call__</code> special method.",gt,j,Be,le,Ve,E,pe,_t,q,ue,vt,Ce,Et="The <code>GenericForTokenClassification</code> forward method, overrides the <code>__call__</code> special method.",Tt,W,Je,he,Xe,Fe,Ye;return B=new fe({props:{title:"GptOss",local:"gptoss",headingTag:"h1"}}),V=new fe({props:{title:"Overview",local:"overview",headingTag:"h2"}}),K=new fe({props:{title:"GptOssConfig",local:"transformers.GptOssConfig",headingTag:"h2"}}),ee=new S({props:{name:"class transformers.GptOssConfig",anchor:"transformers.GptOssConfig",parameters:[{name:"num_hidden_layers",val:": int = 36"},{name:"num_local_experts",val:": int = 128"},{name:"vocab_size",val:": int = 201088"},{name:"hidden_size",val:": int = 2880"},{name:"intermediate_size",val:": int = 2880"},{name:"head_dim",val:": int = 64"},{name:"num_attention_heads",val:": int = 64"},{name:"num_key_value_heads",val:": int = 8"},{name:"sliding_window",val:": int = 128"},{name:"rope_theta",val:": float = 150000.0"},{name:"tie_word_embeddings",val:" = False"},{name:"hidden_act",val:": str = 'silu'"},{name:"initializer_range",val:": float = 0.02"},{name:"max_position_embeddings",val:" = 131072"},{name:"rms_norm_eps",val:": float = 1e-05"},{name:"rope_scaling",val:" = {'rope_type': 'yarn', 'factor': 32.0, 'beta_fast': 32.0, 'beta_slow': 1.0, 'truncate': False, 'original_max_position_embeddings': 4096}"},{name:"attention_dropout",val:": float = 0.0"},{name:"num_experts_per_tok",val:" = 4"},{name:"router_aux_loss_coef",val:": float = 0.9"},{name:"output_router_logits",val:" = False"},{name:"use_cache",val:" = True"},{name:"layer_types",val:" = None"},{name:"**kwargs",val:""}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/gpt_oss/configuration_gpt_oss.py#L21"}}),te=new fe({props:{title:"GptOssModel",local:"transformers.GptOssModel",headingTag:"h2"}}),oe=new S({props:{name:"class transformers.GptOssModel",anchor:"transformers.GptOssModel",parameters:[{name:"config",val:": GptOssConfig"}],parametersDescription:[{anchor:"transformers.GptOssModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/gpt_oss#transformers.GptOssConfig">GptOssConfig</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/gpt_oss/modeling_gpt_oss.py#L435"}}),ne=new S({props:{name:"forward",anchor:"transformers.GptOssModel.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[list[torch.FloatTensor]] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.GptOssModel.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.GptOssModel.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.GptOssModel.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.GptOssModel.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>list[torch.FloatTensor]</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.GptOssModel.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.GptOssModel.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.GptOssModel.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/gpt_oss/modeling_gpt_oss.py#L454",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <code>transformers.modeling_outputs.MoeModelOutputWithPast</code> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/gpt_oss#transformers.GptOssConfig"
>GptOssConfig</a>) and inputs.</p>
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
`}}),U=new Ke({props:{$$slots:{default:[Jt]},$$scope:{ctx:M}}}),se=new fe({props:{title:"GptOssForCausalLM",local:"transformers.GptOssForCausalLM",headingTag:"h2"}}),ae=new S({props:{name:"class transformers.GptOssForCausalLM",anchor:"transformers.GptOssForCausalLM",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.GptOssForCausalLM.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/gpt_oss#transformers.GptOssForCausalLM">GptOssForCausalLM</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/gpt_oss/modeling_gpt_oss.py#L602"}}),re=new S({props:{name:"forward",anchor:"transformers.GptOssForCausalLM.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_router_logits",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"},{name:"logits_to_keep",val:": typing.Union[int, torch.Tensor] = 0"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.GptOssForCausalLM.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.GptOssForCausalLM.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.GptOssForCausalLM.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.GptOssForCausalLM.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.GptOssForCausalLM.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.GptOssForCausalLM.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.GptOssForCausalLM.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.GptOssForCausalLM.forward.output_router_logits",description:`<strong>output_router_logits</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the logits of all the routers. They are useful for computing the router loss, and
should not be returned during inference.`,name:"output_router_logits"},{anchor:"transformers.GptOssForCausalLM.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"},{anchor:"transformers.GptOssForCausalLM.forward.logits_to_keep",description:`<strong>logits_to_keep</strong> (<code>Union[int, torch.Tensor]</code>, defaults to <code>0</code>) &#x2014;
If an <code>int</code>, compute logits for the last <code>logits_to_keep</code> tokens. If <code>0</code>, calculate logits for all
<code>input_ids</code> (special case). Only last token logits are needed for generation, and calculating them only for that
token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
If a <code>torch.Tensor</code>, must be 1D corresponding to the indices to keep in the sequence length dimension.
This is useful when using packed tensor format (single dimension for batch and sequence length).`,name:"logits_to_keep"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/gpt_oss/modeling_gpt_oss.py#L619",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <code>transformers.modeling_outputs.MoeCausalLMOutputWithPast</code> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/gpt_oss#transformers.GptOssConfig"
>GptOssConfig</a>) and inputs.</p>
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
`}}),D=new Ke({props:{$$slots:{default:[Xt]},$$scope:{ctx:M}}}),A=new Bt({props:{anchor:"transformers.GptOssForCausalLM.forward.example",$$slots:{default:[Yt]},$$scope:{ctx:M}}}),ie=new fe({props:{title:"GptOssForSequenceClassification",local:"transformers.GptOssForSequenceClassification",headingTag:"h2"}}),ce=new S({props:{name:"class transformers.GptOssForSequenceClassification",anchor:"transformers.GptOssForSequenceClassification",parameters:[{name:"config",val:""}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/gpt_oss/modeling_gpt_oss.py#L706"}}),de=new S({props:{name:"forward",anchor:"transformers.GptOssForSequenceClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.GptOssForSequenceClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.GptOssForSequenceClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.GptOssForSequenceClassification.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.GptOssForSequenceClassification.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.GptOssForSequenceClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.GptOssForSequenceClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.GptOssForSequenceClassification.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
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
`}}),j=new Ke({props:{$$slots:{default:[Zt]},$$scope:{ctx:M}}}),le=new fe({props:{title:"GptOssForTokenClassification",local:"transformers.GptOssForTokenClassification",headingTag:"h2"}}),pe=new S({props:{name:"class transformers.GptOssForTokenClassification",anchor:"transformers.GptOssForTokenClassification",parameters:[{name:"config",val:""}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/gpt_oss/modeling_gpt_oss.py#L710"}}),ue=new S({props:{name:"forward",anchor:"transformers.GptOssForTokenClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.GptOssForTokenClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.GptOssForTokenClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.GptOssForTokenClassification.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.GptOssForTokenClassification.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.GptOssForTokenClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.GptOssForTokenClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.GptOssForTokenClassification.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
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
`}}),W=new Ke({props:{$$slots:{default:[Qt]},$$scope:{ctx:M}}}),he=new Vt({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/gpt_oss.md"}}),{c(){n=c("meta"),T=a(),l=c("p"),v=a(),C=c("p"),C.innerHTML=b,N=a(),R=c("div"),R.innerHTML=bt,Ge=a(),p(B.$$.fragment),Le=a(),p(V.$$.fragment),qe=a(),J=c("p"),J.innerHTML=yt,Ie=Ht(`
<INSERT SHORT SUMMARY HERE>
`),X=c("p"),X.textContent=kt,Pe=a(),Y=c("p"),Y.innerHTML=wt,Ee=a(),Z=c("p"),Z.textContent=$t,He=Ht(`
<INSERT TIPS ABOUT MODEL HERE>
`),Q=c("p"),Q.innerHTML=Mt,Se=a(),p(K.$$.fragment),Ne=a(),I=c("div"),p(ee.$$.fragment),et=a(),ge=c("p"),ge.innerHTML=Ct,Re=a(),p(te.$$.fragment),Ue=a(),y=c("div"),p(oe.$$.fragment),tt=a(),_e=c("p"),_e.textContent=xt,ot=a(),ve=c("p"),ve.innerHTML=Ot,nt=a(),Te=c("p"),Te.innerHTML=Ft,st=a(),G=c("div"),p(ne.$$.fragment),at=a(),be=c("p"),be.innerHTML=zt,rt=a(),p(U.$$.fragment),De=a(),p(se.$$.fragment),Ae=a(),k=c("div"),p(ae.$$.fragment),it=a(),ye=c("p"),ye.textContent=Gt,ct=a(),ke=c("p"),ke.innerHTML=Lt,dt=a(),we=c("p"),we.innerHTML=qt,lt=a(),x=c("div"),p(re.$$.fragment),pt=a(),$e=c("p"),$e.innerHTML=It,ut=a(),p(D.$$.fragment),ht=a(),p(A.$$.fragment),je=a(),p(ie.$$.fragment),We=a(),P=c("div"),p(ce.$$.fragment),mt=a(),L=c("div"),p(de.$$.fragment),ft=a(),Me=c("p"),Me.innerHTML=Pt,gt=a(),p(j.$$.fragment),Be=a(),p(le.$$.fragment),Ve=a(),E=c("div"),p(pe.$$.fragment),_t=a(),q=c("div"),p(ue.$$.fragment),vt=a(),Ce=c("p"),Ce.innerHTML=Et,Tt=a(),p(W.$$.fragment),Je=a(),p(he.$$.fragment),Xe=a(),Fe=c("p"),this.h()},l(e){const t=At("svelte-u9bgzb",document.head);n=d(t,"META",{name:!0,content:!0}),t.forEach(o),T=r(e),l=d(e,"P",{}),F(l).forEach(o),v=r(e),C=d(e,"P",{"data-svelte-h":!0}),_(C)!=="svelte-1aiwji2"&&(C.innerHTML=b),N=r(e),R=d(e,"DIV",{style:!0,"data-svelte-h":!0}),_(R)!=="svelte-2m0t7r"&&(R.innerHTML=bt),Ge=r(e),u(B.$$.fragment,e),Le=r(e),u(V.$$.fragment,e),qe=r(e),J=d(e,"P",{"data-svelte-h":!0}),_(J)!=="svelte-13r8ibo"&&(J.innerHTML=yt),Ie=St(e,`
<INSERT SHORT SUMMARY HERE>
`),X=d(e,"P",{"data-svelte-h":!0}),_(X)!=="svelte-vfdo9a"&&(X.textContent=kt),Pe=r(e),Y=d(e,"P",{"data-svelte-h":!0}),_(Y)!=="svelte-1d070d4"&&(Y.innerHTML=wt),Ee=r(e),Z=d(e,"P",{"data-svelte-h":!0}),_(Z)!=="svelte-axv494"&&(Z.textContent=$t),He=St(e,`
<INSERT TIPS ABOUT MODEL HERE>
`),Q=d(e,"P",{"data-svelte-h":!0}),_(Q)!=="svelte-1dsg3m1"&&(Q.innerHTML=Mt),Se=r(e),u(K.$$.fragment,e),Ne=r(e),I=d(e,"DIV",{class:!0});var me=F(I);u(ee.$$.fragment,me),et=r(me),ge=d(me,"P",{"data-svelte-h":!0}),_(ge)!=="svelte-y35bvn"&&(ge.innerHTML=Ct),me.forEach(o),Re=r(e),u(te.$$.fragment,e),Ue=r(e),y=d(e,"DIV",{class:!0});var w=F(y);u(oe.$$.fragment,w),tt=r(w),_e=d(w,"P",{"data-svelte-h":!0}),_(_e)!=="svelte-1w0y56c"&&(_e.textContent=xt),ot=r(w),ve=d(w,"P",{"data-svelte-h":!0}),_(ve)!=="svelte-q52n56"&&(ve.innerHTML=Ot),nt=r(w),Te=d(w,"P",{"data-svelte-h":!0}),_(Te)!=="svelte-hswkmf"&&(Te.innerHTML=Ft),st=r(w),G=d(w,"DIV",{class:!0});var H=F(G);u(ne.$$.fragment,H),at=r(H),be=d(H,"P",{"data-svelte-h":!0}),_(be)!=="svelte-1i5bxx4"&&(be.innerHTML=zt),rt=r(H),u(U.$$.fragment,H),H.forEach(o),w.forEach(o),De=r(e),u(se.$$.fragment,e),Ae=r(e),k=d(e,"DIV",{class:!0});var $=F(k);u(ae.$$.fragment,$),it=r($),ye=d($,"P",{"data-svelte-h":!0}),_(ye)!=="svelte-1aqouxp"&&(ye.textContent=Gt),ct=r($),ke=d($,"P",{"data-svelte-h":!0}),_(ke)!=="svelte-q52n56"&&(ke.innerHTML=Lt),dt=r($),we=d($,"P",{"data-svelte-h":!0}),_(we)!=="svelte-hswkmf"&&(we.innerHTML=qt),lt=r($),x=d($,"DIV",{class:!0});var O=F(x);u(re.$$.fragment,O),pt=r(O),$e=d(O,"P",{"data-svelte-h":!0}),_($e)!=="svelte-7bvuz8"&&($e.innerHTML=It),ut=r(O),u(D.$$.fragment,O),ht=r(O),u(A.$$.fragment,O),O.forEach(o),$.forEach(o),je=r(e),u(ie.$$.fragment,e),We=r(e),P=d(e,"DIV",{class:!0});var Ze=F(P);u(ce.$$.fragment,Ze),mt=r(Ze),L=d(Ze,"DIV",{class:!0});var xe=F(L);u(de.$$.fragment,xe),ft=r(xe),Me=d(xe,"P",{"data-svelte-h":!0}),_(Me)!=="svelte-1sal4ui"&&(Me.innerHTML=Pt),gt=r(xe),u(j.$$.fragment,xe),xe.forEach(o),Ze.forEach(o),Be=r(e),u(le.$$.fragment,e),Ve=r(e),E=d(e,"DIV",{class:!0});var Qe=F(E);u(pe.$$.fragment,Qe),_t=r(Qe),q=d(Qe,"DIV",{class:!0});var Oe=F(q);u(ue.$$.fragment,Oe),vt=r(Oe),Ce=d(Oe,"P",{"data-svelte-h":!0}),_(Ce)!=="svelte-1py4aay"&&(Ce.innerHTML=Et),Tt=r(Oe),u(W.$$.fragment,Oe),Oe.forEach(o),Qe.forEach(o),Je=r(e),u(he.$$.fragment,e),Xe=r(e),Fe=d(e,"P",{}),F(Fe).forEach(o),this.h()},h(){z(n,"name","hf:doc:metadata"),z(n,"content",eo),jt(R,"float","right"),z(I,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),z(G,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),z(y,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),z(x,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),z(k,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),z(L,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),z(P,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),z(q,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),z(E,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,t){i(document.head,n),s(e,T,t),s(e,l,t),s(e,v,t),s(e,C,t),s(e,N,t),s(e,R,t),s(e,Ge,t),h(B,e,t),s(e,Le,t),h(V,e,t),s(e,qe,t),s(e,J,t),s(e,Ie,t),s(e,X,t),s(e,Pe,t),s(e,Y,t),s(e,Ee,t),s(e,Z,t),s(e,He,t),s(e,Q,t),s(e,Se,t),h(K,e,t),s(e,Ne,t),s(e,I,t),h(ee,I,null),i(I,et),i(I,ge),s(e,Re,t),h(te,e,t),s(e,Ue,t),s(e,y,t),h(oe,y,null),i(y,tt),i(y,_e),i(y,ot),i(y,ve),i(y,nt),i(y,Te),i(y,st),i(y,G),h(ne,G,null),i(G,at),i(G,be),i(G,rt),h(U,G,null),s(e,De,t),h(se,e,t),s(e,Ae,t),s(e,k,t),h(ae,k,null),i(k,it),i(k,ye),i(k,ct),i(k,ke),i(k,dt),i(k,we),i(k,lt),i(k,x),h(re,x,null),i(x,pt),i(x,$e),i(x,ut),h(D,x,null),i(x,ht),h(A,x,null),s(e,je,t),h(ie,e,t),s(e,We,t),s(e,P,t),h(ce,P,null),i(P,mt),i(P,L),h(de,L,null),i(L,ft),i(L,Me),i(L,gt),h(j,L,null),s(e,Be,t),h(le,e,t),s(e,Ve,t),s(e,E,t),h(pe,E,null),i(E,_t),i(E,q),h(ue,q,null),i(q,vt),i(q,Ce),i(q,Tt),h(W,q,null),s(e,Je,t),h(he,e,t),s(e,Xe,t),s(e,Fe,t),Ye=!0},p(e,[t]){const me={};t&2&&(me.$$scope={dirty:t,ctx:e}),U.$set(me);const w={};t&2&&(w.$$scope={dirty:t,ctx:e}),D.$set(w);const H={};t&2&&(H.$$scope={dirty:t,ctx:e}),A.$set(H);const $={};t&2&&($.$$scope={dirty:t,ctx:e}),j.$set($);const O={};t&2&&(O.$$scope={dirty:t,ctx:e}),W.$set(O)},i(e){Ye||(m(B.$$.fragment,e),m(V.$$.fragment,e),m(K.$$.fragment,e),m(ee.$$.fragment,e),m(te.$$.fragment,e),m(oe.$$.fragment,e),m(ne.$$.fragment,e),m(U.$$.fragment,e),m(se.$$.fragment,e),m(ae.$$.fragment,e),m(re.$$.fragment,e),m(D.$$.fragment,e),m(A.$$.fragment,e),m(ie.$$.fragment,e),m(ce.$$.fragment,e),m(de.$$.fragment,e),m(j.$$.fragment,e),m(le.$$.fragment,e),m(pe.$$.fragment,e),m(ue.$$.fragment,e),m(W.$$.fragment,e),m(he.$$.fragment,e),Ye=!0)},o(e){f(B.$$.fragment,e),f(V.$$.fragment,e),f(K.$$.fragment,e),f(ee.$$.fragment,e),f(te.$$.fragment,e),f(oe.$$.fragment,e),f(ne.$$.fragment,e),f(U.$$.fragment,e),f(se.$$.fragment,e),f(ae.$$.fragment,e),f(re.$$.fragment,e),f(D.$$.fragment,e),f(A.$$.fragment,e),f(ie.$$.fragment,e),f(ce.$$.fragment,e),f(de.$$.fragment,e),f(j.$$.fragment,e),f(le.$$.fragment,e),f(pe.$$.fragment,e),f(ue.$$.fragment,e),f(W.$$.fragment,e),f(he.$$.fragment,e),Ye=!1},d(e){e&&(o(T),o(l),o(v),o(C),o(N),o(R),o(Ge),o(Le),o(qe),o(J),o(Ie),o(X),o(Pe),o(Y),o(Ee),o(Z),o(He),o(Q),o(Se),o(Ne),o(I),o(Re),o(Ue),o(y),o(De),o(Ae),o(k),o(je),o(We),o(P),o(Be),o(Ve),o(E),o(Je),o(Xe),o(Fe)),o(n),g(B,e),g(V,e),g(K,e),g(ee),g(te,e),g(oe),g(ne),g(U),g(se,e),g(ae),g(re),g(D),g(A),g(ie,e),g(ce),g(de),g(j),g(le,e),g(pe),g(ue),g(W),g(he,e)}}}const eo='{"title":"GptOss","local":"gptoss","sections":[{"title":"Overview","local":"overview","sections":[],"depth":2},{"title":"GptOssConfig","local":"transformers.GptOssConfig","sections":[],"depth":2},{"title":"GptOssModel","local":"transformers.GptOssModel","sections":[],"depth":2},{"title":"GptOssForCausalLM","local":"transformers.GptOssForCausalLM","sections":[],"depth":2},{"title":"GptOssForSequenceClassification","local":"transformers.GptOssForSequenceClassification","sections":[],"depth":2},{"title":"GptOssForTokenClassification","local":"transformers.GptOssForTokenClassification","sections":[],"depth":2}],"depth":1}';function to(M){return Rt(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class lo extends Ut{constructor(n){super(),Dt(this,n,to,Kt,Nt,{})}}export{lo as component};
