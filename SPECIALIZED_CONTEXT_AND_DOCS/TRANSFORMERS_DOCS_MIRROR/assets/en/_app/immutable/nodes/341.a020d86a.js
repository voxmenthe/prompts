import{s as at,o as rt,n as xe}from"../chunks/scheduler.18a86fab.js";import{S as it,i as dt,g as h,s as d,r as g,A as ct,h as m,f as s,c,j as V,x as k,u as _,k as Z,y as p,a as l,v as b,d as v,t as y,w as T}from"../chunks/index.98837b22.js";import{T as Ee}from"../chunks/Tip.77304350.js";import{D as le}from"../chunks/Docstring.a1ef7999.js";import{C as tt}from"../chunks/CodeBlock.8d0c2e8a.js";import{F as lt,M as pt}from"../chunks/Markdown.ae01904b.js";import{E as st}from"../chunks/ExampleCodeBlock.8c3ee1f9.js";import{H as pe,E as ht}from"../chunks/getInferenceSnippets.06c2775f.js";function mt(M){let t,f="Phi-3 has been integrated in the development version (4.40.0.dev) of <code>transformers</code>. Until the official version is released through <code>pip</code>, ensure that you are doing one of the following:",n,i,w="<li><p>When loading the model, ensure that <code>trust_remote_code=True</code> is passed as an argument of the <code>from_pretrained()</code> function.</p></li> <li><p>Update your local <code>transformers</code> to the development version: <code>pip uninstall -y transformers &amp;&amp; pip install git+https://github.com/huggingface/transformers</code>. The previous command is an alternative to cloning and installing from the source.</p></li>";return{c(){t=h("p"),t.innerHTML=f,n=d(),i=h("ul"),i.innerHTML=w},l(r){t=m(r,"P",{"data-svelte-h":!0}),k(t)!=="svelte-c51ot7"&&(t.innerHTML=f),n=c(r),i=m(r,"UL",{"data-svelte-h":!0}),k(i)!=="svelte-1ysocqo"&&(i.innerHTML=w)},m(r,$){l(r,t,$),l(r,n,$),l(r,i,$)},p:xe,d(r){r&&(s(t),s(n),s(i))}}}function ut(M){let t,f="Example:",n,i,w;return i=new tt({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMFBoaTNNb2RlbCUyQyUyMFBoaTNDb25maWclMEElMEElMjMlMjBJbml0aWFsaXppbmclMjBhJTIwUGhpLTMlMjBzdHlsZSUyMGNvbmZpZ3VyYXRpb24lMEFjb25maWd1cmF0aW9uJTIwJTNEJTIwUGhpM0NvbmZpZy5mcm9tX3ByZXRyYWluZWQoJTIybWljcm9zb2Z0JTJGUGhpLTMtbWluaS00ay1pbnN0cnVjdCUyMiklMEElMEElMjMlMjBJbml0aWFsaXppbmclMjBhJTIwbW9kZWwlMjBmcm9tJTIwdGhlJTIwY29uZmlndXJhdGlvbiUwQW1vZGVsJTIwJTNEJTIwUGhpM01vZGVsKGNvbmZpZ3VyYXRpb24pJTBBJTBBJTIzJTIwQWNjZXNzaW5nJTIwdGhlJTIwbW9kZWwlMjBjb25maWd1cmF0aW9uJTBBY29uZmlndXJhdGlvbiUyMCUzRCUyMG1vZGVsLmNvbmZpZw==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> Phi3Model, Phi3Config

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a Phi-3 style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = Phi3Config.from_pretrained(<span class="hljs-string">&quot;microsoft/Phi-3-mini-4k-instruct&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a model from the configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = Phi3Model(configuration)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Accessing the model configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = model.config`,wrap:!1}}),{c(){t=h("p"),t.textContent=f,n=d(),g(i.$$.fragment)},l(r){t=m(r,"P",{"data-svelte-h":!0}),k(t)!=="svelte-11lpom8"&&(t.textContent=f),n=c(r),_(i.$$.fragment,r)},m(r,$){l(r,t,$),l(r,n,$),b(i,r,$),w=!0},p:xe,i(r){w||(v(i.$$.fragment,r),w=!0)},o(r){y(i.$$.fragment,r),w=!1},d(r){r&&(s(t),s(n)),T(i,r)}}}function ft(M){let t,f=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=h("p"),t.innerHTML=f},l(n){t=m(n,"P",{"data-svelte-h":!0}),k(t)!=="svelte-fincs2"&&(t.innerHTML=f)},m(n,i){l(n,t,i)},p:xe,d(n){n&&s(t)}}}function gt(M){let t,f=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=h("p"),t.innerHTML=f},l(n){t=m(n,"P",{"data-svelte-h":!0}),k(t)!=="svelte-fincs2"&&(t.innerHTML=f)},m(n,i){l(n,t,i)},p:xe,d(n){n&&s(t)}}}function _t(M){let t,f="Example:",n,i,w;return i=new tt({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBQaGkzRm9yQ2F1c2FsTE0lMEElMEFtb2RlbCUyMCUzRCUyMFBoaTNGb3JDYXVzYWxMTS5mcm9tX3ByZXRyYWluZWQoJTIybWV0YS1waGkzJTJGUGhpMy0yLTdiLWhmJTIyKSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMm1ldGEtcGhpMyUyRlBoaTMtMi03Yi1oZiUyMiklMEElMEFwcm9tcHQlMjAlM0QlMjAlMjJIZXklMkMlMjBhcmUlMjB5b3UlMjBjb25zY2lvdXMlM0YlMjBDYW4lMjB5b3UlMjB0YWxrJTIwdG8lMjBtZSUzRiUyMiUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplcihwcm9tcHQlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQSUwQSUyMyUyMEdlbmVyYXRlJTBBZ2VuZXJhdGVfaWRzJTIwJTNEJTIwbW9kZWwuZ2VuZXJhdGUoaW5wdXRzLmlucHV0X2lkcyUyQyUyMG1heF9sZW5ndGglM0QzMCklMEF0b2tlbml6ZXIuYmF0Y2hfZGVjb2RlKGdlbmVyYXRlX2lkcyUyQyUyMHNraXBfc3BlY2lhbF90b2tlbnMlM0RUcnVlJTJDJTIwY2xlYW5fdXBfdG9rZW5pemF0aW9uX3NwYWNlcyUzREZhbHNlKSU1QjAlNUQ=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, Phi3ForCausalLM

<span class="hljs-meta">&gt;&gt;&gt; </span>model = Phi3ForCausalLM.from_pretrained(<span class="hljs-string">&quot;meta-phi3/Phi3-2-7b-hf&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;meta-phi3/Phi3-2-7b-hf&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>prompt = <span class="hljs-string">&quot;Hey, are you conscious? Can you talk to me?&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(prompt, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Generate</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>generate_ids = model.generate(inputs.input_ids, max_length=<span class="hljs-number">30</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer.batch_decode(generate_ids, skip_special_tokens=<span class="hljs-literal">True</span>, clean_up_tokenization_spaces=<span class="hljs-literal">False</span>)[<span class="hljs-number">0</span>]
<span class="hljs-string">&quot;Hey, are you conscious? Can you talk to me?\\nI&#x27;m not conscious, but I can talk to you.&quot;</span>`,wrap:!1}}),{c(){t=h("p"),t.textContent=f,n=d(),g(i.$$.fragment)},l(r){t=m(r,"P",{"data-svelte-h":!0}),k(t)!=="svelte-11lpom8"&&(t.textContent=f),n=c(r),_(i.$$.fragment,r)},m(r,$){l(r,t,$),l(r,n,$),b(i,r,$),w=!0},p:xe,i(r){w||(v(i.$$.fragment,r),w=!0)},o(r){y(i.$$.fragment,r),w=!1},d(r){r&&(s(t),s(n)),T(i,r)}}}function bt(M){let t,f=`Most generation-controlling parameters are set in <code>generation_config</code> which, if not passed, will be set to the
model’s default generation configuration. You can override any <code>generation_config</code> by passing the corresponding
parameters to generate(), e.g. <code>.generate(inputs, num_beams=4, do_sample=True)</code>.`,n,i,w=`For an overview of generation strategies and code examples, check out the <a href="../generation_strategies">following
guide</a>.`;return{c(){t=h("p"),t.innerHTML=f,n=d(),i=h("p"),i.innerHTML=w},l(r){t=m(r,"P",{"data-svelte-h":!0}),k(t)!=="svelte-1c5u34l"&&(t.innerHTML=f),n=c(r),i=m(r,"P",{"data-svelte-h":!0}),k(i)!=="svelte-fvlq1g"&&(i.innerHTML=w)},m(r,$){l(r,t,$),l(r,n,$),l(r,i,$)},p:xe,d(r){r&&(s(t),s(n),s(i))}}}function vt(M){let t,f=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=h("p"),t.innerHTML=f},l(n){t=m(n,"P",{"data-svelte-h":!0}),k(t)!=="svelte-fincs2"&&(t.innerHTML=f)},m(n,i){l(n,t,i)},p:xe,d(n){n&&s(t)}}}function yt(M){let t,f=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=h("p"),t.innerHTML=f},l(n){t=m(n,"P",{"data-svelte-h":!0}),k(t)!=="svelte-fincs2"&&(t.innerHTML=f)},m(n,i){l(n,t,i)},p:xe,d(n){n&&s(t)}}}function Tt(M){let t,f,n,i,w,r,$="The bare Phi3 Model outputting raw hidden-states without any specific head on top.",X,Q,Y=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,He,K,ae=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,ye,x,re,Te,N,ze='The <a href="/docs/transformers/v4.56.2/en/model_doc/phi3#transformers.Phi3Model">Phi3Model</a> forward method, overrides the <code>__call__</code> special method.',ee,te,he,I,Fe,C,j,Ge,oe,ie="The Phi3 Model for causal language modeling.",we,B,Ze=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,ke,D,Le=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,A,z,U,$e,J,qe='The <a href="/docs/transformers/v4.56.2/en/model_doc/phi3#transformers.Phi3ForCausalLM">Phi3ForCausalLM</a> forward method, overrides the <code>__call__</code> special method.',F,q,Ie,O,Ne,S,E,Be,de,ne="Generates sequences of token ids for models with a language modeling head.",Me,L,me,W,ue,H,R,e,a,P,De,fe,ot="The <code>GenericForSequenceClassification</code> forward method, overrides the <code>__call__</code> special method.",Qe,Ce,Ae,je,Re,ge,Ue,Ye,ce,Oe,Ke,Je,nt="The <code>GenericForTokenClassification</code> forward method, overrides the <code>__call__</code> special method.",et,Pe,Ve;return t=new pe({props:{title:"Phi3Model",local:"transformers.Phi3Model",headingTag:"h2"}}),i=new le({props:{name:"class transformers.Phi3Model",anchor:"transformers.Phi3Model",parameters:[{name:"config",val:": Phi3Config"}],parametersDescription:[{anchor:"transformers.Phi3Model.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/phi3#transformers.Phi3Config">Phi3Config</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/phi3/modeling_phi3.py#L339"}}),re=new le({props:{name:"forward",anchor:"transformers.Phi3Model.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.Phi3Model.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.Phi3Model.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.Phi3Model.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.Phi3Model.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.Phi3Model.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.Phi3Model.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.Phi3Model.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/phi3/modeling_phi3.py#L356",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast"
>transformers.modeling_outputs.BaseModelOutputWithPast</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/phi3#transformers.Phi3Config"
>Phi3Config</a>) and inputs.</p>
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
`}}),te=new Ee({props:{$$slots:{default:[ft]},$$scope:{ctx:M}}}),I=new pe({props:{title:"Phi3ForCausalLM",local:"transformers.Phi3ForCausalLM",headingTag:"h2"}}),j=new le({props:{name:"class transformers.Phi3ForCausalLM",anchor:"transformers.Phi3ForCausalLM",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.Phi3ForCausalLM.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/phi3#transformers.Phi3ForCausalLM">Phi3ForCausalLM</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/phi3/modeling_phi3.py#L419"}}),U=new le({props:{name:"forward",anchor:"transformers.Phi3ForCausalLM.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"},{name:"logits_to_keep",val:": typing.Union[int, torch.Tensor] = 0"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.Phi3ForCausalLM.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.Phi3ForCausalLM.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.Phi3ForCausalLM.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.Phi3ForCausalLM.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.Phi3ForCausalLM.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.Phi3ForCausalLM.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.Phi3ForCausalLM.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.Phi3ForCausalLM.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"},{anchor:"transformers.Phi3ForCausalLM.forward.logits_to_keep",description:`<strong>logits_to_keep</strong> (<code>Union[int, torch.Tensor]</code>, defaults to <code>0</code>) &#x2014;
If an <code>int</code>, compute logits for the last <code>logits_to_keep</code> tokens. If <code>0</code>, calculate logits for all
<code>input_ids</code> (special case). Only last token logits are needed for generation, and calculating them only for that
token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
If a <code>torch.Tensor</code>, must be 1D corresponding to the indices to keep in the sequence length dimension.
This is useful when using packed tensor format (single dimension for batch and sequence length).`,name:"logits_to_keep"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/phi3/modeling_phi3.py#L433",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast"
>transformers.modeling_outputs.CausalLMOutputWithPast</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/phi3#transformers.Phi3Config"
>Phi3Config</a>) and inputs.</p>
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
`}}),q=new Ee({props:{$$slots:{default:[gt]},$$scope:{ctx:M}}}),O=new st({props:{anchor:"transformers.Phi3ForCausalLM.forward.example",$$slots:{default:[_t]},$$scope:{ctx:M}}}),E=new le({props:{name:"generate",anchor:"transformers.Phi3ForCausalLM.generate",parameters:[{name:"inputs",val:": typing.Optional[torch.Tensor] = None"},{name:"generation_config",val:": typing.Optional[transformers.generation.configuration_utils.GenerationConfig] = None"},{name:"logits_processor",val:": typing.Optional[transformers.generation.logits_process.LogitsProcessorList] = None"},{name:"stopping_criteria",val:": typing.Optional[transformers.generation.stopping_criteria.StoppingCriteriaList] = None"},{name:"prefix_allowed_tokens_fn",val:": typing.Optional[typing.Callable[[int, torch.Tensor], list[int]]] = None"},{name:"synced_gpus",val:": typing.Optional[bool] = None"},{name:"assistant_model",val:": typing.Optional[ForwardRef('PreTrainedModel')] = None"},{name:"streamer",val:": typing.Optional[ForwardRef('BaseStreamer')] = None"},{name:"negative_prompt_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"negative_prompt_attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"use_model_defaults",val:": typing.Optional[bool] = None"},{name:"custom_generate",val:": typing.Union[str, typing.Callable, NoneType] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.Phi3ForCausalLM.generate.inputs",description:`<strong>inputs</strong> (<code>torch.Tensor</code> of varying shape depending on the modality, <em>optional</em>) &#x2014;
The sequence used as a prompt for the generation or as model inputs to the encoder. If <code>None</code> the
method initializes it with <code>bos_token_id</code> and a batch size of 1. For decoder-only models <code>inputs</code>
should be in the format of <code>input_ids</code>. For encoder-decoder models <em>inputs</em> can represent any of
<code>input_ids</code>, <code>input_values</code>, <code>input_features</code>, or <code>pixel_values</code>.`,name:"inputs"},{anchor:"transformers.Phi3ForCausalLM.generate.generation_config",description:`<strong>generation_config</strong> (<a href="/docs/transformers/v4.56.2/en/main_classes/text_generation#transformers.GenerationConfig">GenerationConfig</a>, <em>optional</em>) &#x2014;
The generation configuration to be used as base parametrization for the generation call. <code>**kwargs</code>
passed to generate matching the attributes of <code>generation_config</code> will override them. If
<code>generation_config</code> is not provided, the default will be used, which has the following loading
priority: 1) from the <code>generation_config.json</code> model file, if it exists; 2) from the model
configuration. Please note that unspecified parameters will inherit <a href="/docs/transformers/v4.56.2/en/main_classes/text_generation#transformers.GenerationConfig">GenerationConfig</a>&#x2019;s
default values, whose documentation should be checked to parameterize generation.`,name:"generation_config"},{anchor:"transformers.Phi3ForCausalLM.generate.logits_processor",description:`<strong>logits_processor</strong> (<code>LogitsProcessorList</code>, <em>optional</em>) &#x2014;
Custom logits processors that complement the default logits processors built from arguments and
generation config. If a logit processor is passed that is already created with the arguments or a
generation config an error is thrown. This feature is intended for advanced users.`,name:"logits_processor"},{anchor:"transformers.Phi3ForCausalLM.generate.stopping_criteria",description:`<strong>stopping_criteria</strong> (<code>StoppingCriteriaList</code>, <em>optional</em>) &#x2014;
Custom stopping criteria that complements the default stopping criteria built from arguments and a
generation config. If a stopping criteria is passed that is already created with the arguments or a
generation config an error is thrown. If your stopping criteria depends on the <code>scores</code> input, make
sure you pass <code>return_dict_in_generate=True, output_scores=True</code> to <code>generate</code>. This feature is
intended for advanced users.`,name:"stopping_criteria"},{anchor:"transformers.Phi3ForCausalLM.generate.prefix_allowed_tokens_fn",description:`<strong>prefix_allowed_tokens_fn</strong> (<code>Callable[[int, torch.Tensor], list[int]]</code>, <em>optional</em>) &#x2014;
If provided, this function constraints the beam search to allowed tokens only at each step. If not
provided no constraint is applied. This function takes 2 arguments: the batch ID <code>batch_id</code> and
<code>input_ids</code>. It has to return a list with the allowed tokens for the next generation step conditioned
on the batch ID <code>batch_id</code> and the previously generated tokens <code>inputs_ids</code>. This argument is useful
for constrained generation conditioned on the prefix, as described in <a href="https://huggingface.co/papers/2010.00904" rel="nofollow">Autoregressive Entity
Retrieval</a>.`,name:"prefix_allowed_tokens_fn"},{anchor:"transformers.Phi3ForCausalLM.generate.synced_gpus",description:`<strong>synced_gpus</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether to continue running the while loop until max_length. Unless overridden, this flag will be set
to <code>True</code> if using <code>FullyShardedDataParallel</code> or DeepSpeed ZeRO Stage 3 with multiple GPUs to avoid
deadlocking if one GPU finishes generating before other GPUs. Otherwise, defaults to <code>False</code>.`,name:"synced_gpus"},{anchor:"transformers.Phi3ForCausalLM.generate.assistant_model",description:`<strong>assistant_model</strong> (<code>PreTrainedModel</code>, <em>optional</em>) &#x2014;
An assistant model that can be used to accelerate generation. The assistant model must have the exact
same tokenizer. The acceleration is achieved when forecasting candidate tokens with the assistant model
is much faster than running generation with the model you&#x2019;re calling generate from. As such, the
assistant model should be much smaller.`,name:"assistant_model"},{anchor:"transformers.Phi3ForCausalLM.generate.streamer",description:`<strong>streamer</strong> (<code>BaseStreamer</code>, <em>optional</em>) &#x2014;
Streamer object that will be used to stream the generated sequences. Generated tokens are passed
through <code>streamer.put(token_ids)</code> and the streamer is responsible for any further processing.`,name:"streamer"},{anchor:"transformers.Phi3ForCausalLM.generate.negative_prompt_ids",description:`<strong>negative_prompt_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
The negative prompt needed for some processors such as CFG. The batch size must match the input batch
size. This is an experimental feature, subject to breaking API changes in future versions.`,name:"negative_prompt_ids"},{anchor:"transformers.Phi3ForCausalLM.generate.negative_prompt_attention_mask",description:`<strong>negative_prompt_attention_mask</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Attention_mask for <code>negative_prompt_ids</code>.`,name:"negative_prompt_attention_mask"},{anchor:"transformers.Phi3ForCausalLM.generate.use_model_defaults",description:`<strong>use_model_defaults</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
When it is <code>True</code>, unset parameters in <code>generation_config</code> will be set to the model-specific default
generation configuration (<code>model.generation_config</code>), as opposed to the global defaults
(<code>GenerationConfig()</code>). If unset, models saved starting from <code>v4.50</code> will consider this flag to be
<code>True</code>.`,name:"use_model_defaults"},{anchor:"transformers.Phi3ForCausalLM.generate.custom_generate",description:`<strong>custom_generate</strong> (<code>str</code> or <code>Callable</code>, <em>optional</em>) &#x2014;
One of the following:<ul>
<li><code>str</code> (Hugging Face Hub repository name): runs the custom <code>generate</code> function defined at
<code>custom_generate/generate.py</code> in that repository instead of the standard <code>generate</code> method. The
repository fully replaces the generation logic, and the return type may differ.</li>
<li><code>str</code> (local repository path): same as above but from a local path, <code>trust_remote_code</code> not required.</li>
<li><code>Callable</code>: <code>generate</code> will perform the usual input preparation steps, then call the provided callable to
run the decoding loop.
For more information, see <a href="../../generation_strategies#custom-generation-methods">the docs</a>.</li>
</ul>`,name:"custom_generate"},{anchor:"transformers.Phi3ForCausalLM.generate.kwargs",description:`<strong>kwargs</strong> (<code>dict[str, Any]</code>, <em>optional</em>) &#x2014;
Ad hoc parametrization of <code>generation_config</code> and/or additional model-specific kwargs that will be
forwarded to the <code>forward</code> function of the model. If the model is an encoder-decoder model, encoder
specific kwargs should not be prefixed and decoder specific kwargs should be prefixed with <em>decoder_</em>.`,name:"kwargs"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/generation/utils.py#L2140",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput"
>ModelOutput</a> (if <code>return_dict_in_generate=True</code>
or when <code>config.return_dict_in_generate=True</code>) or a <code>torch.LongTensor</code>.</p>
<p>If the model is <em>not</em> an encoder-decoder model (<code>model.config.is_encoder_decoder=False</code>), the possible
<a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput"
>ModelOutput</a> types are:</p>
<ul>
<li><a
  href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.generation.GenerateDecoderOnlyOutput"
>GenerateDecoderOnlyOutput</a>,</li>
<li><a
  href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.generation.GenerateBeamDecoderOnlyOutput"
>GenerateBeamDecoderOnlyOutput</a></li>
</ul>
<p>If the model is an encoder-decoder model (<code>model.config.is_encoder_decoder=True</code>), the possible
<a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput"
>ModelOutput</a> types are:</p>
<ul>
<li><a
  href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.generation.GenerateEncoderDecoderOutput"
>GenerateEncoderDecoderOutput</a>,</li>
<li><a
  href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.generation.GenerateBeamEncoderDecoderOutput"
>GenerateBeamEncoderDecoderOutput</a></li>
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput"
>ModelOutput</a> or <code>torch.LongTensor</code></p>
`}}),L=new Ee({props:{warning:!0,$$slots:{default:[bt]},$$scope:{ctx:M}}}),W=new pe({props:{title:"Phi3ForSequenceClassification",local:"transformers.Phi3ForSequenceClassification",headingTag:"h2"}}),R=new le({props:{name:"class transformers.Phi3ForSequenceClassification",anchor:"transformers.Phi3ForSequenceClassification",parameters:[{name:"config",val:""}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/phi3/modeling_phi3.py#L533"}}),P=new le({props:{name:"forward",anchor:"transformers.Phi3ForSequenceClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.Phi3ForSequenceClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.Phi3ForSequenceClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.Phi3ForSequenceClassification.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.Phi3ForSequenceClassification.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.Phi3ForSequenceClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.Phi3ForSequenceClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.Phi3ForSequenceClassification.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
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
`}}),Ce=new Ee({props:{$$slots:{default:[vt]},$$scope:{ctx:M}}}),je=new pe({props:{title:"Phi3ForTokenClassification",local:"transformers.Phi3ForTokenClassification",headingTag:"h2"}}),Ue=new le({props:{name:"class transformers.Phi3ForTokenClassification",anchor:"transformers.Phi3ForTokenClassification",parameters:[{name:"config",val:""}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/phi3/modeling_phi3.py#L537"}}),Oe=new le({props:{name:"forward",anchor:"transformers.Phi3ForTokenClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.Phi3ForTokenClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.Phi3ForTokenClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.Phi3ForTokenClassification.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.Phi3ForTokenClassification.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.Phi3ForTokenClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.Phi3ForTokenClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.Phi3ForTokenClassification.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
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
`}}),Pe=new Ee({props:{$$slots:{default:[yt]},$$scope:{ctx:M}}}),{c(){g(t.$$.fragment),f=d(),n=h("div"),g(i.$$.fragment),w=d(),r=h("p"),r.textContent=$,X=d(),Q=h("p"),Q.innerHTML=Y,He=d(),K=h("p"),K.innerHTML=ae,ye=d(),x=h("div"),g(re.$$.fragment),Te=d(),N=h("p"),N.innerHTML=ze,ee=d(),g(te.$$.fragment),he=d(),g(I.$$.fragment),Fe=d(),C=h("div"),g(j.$$.fragment),Ge=d(),oe=h("p"),oe.textContent=ie,we=d(),B=h("p"),B.innerHTML=Ze,ke=d(),D=h("p"),D.innerHTML=Le,A=d(),z=h("div"),g(U.$$.fragment),$e=d(),J=h("p"),J.innerHTML=qe,F=d(),g(q.$$.fragment),Ie=d(),g(O.$$.fragment),Ne=d(),S=h("div"),g(E.$$.fragment),Be=d(),de=h("p"),de.textContent=ne,Me=d(),g(L.$$.fragment),me=d(),g(W.$$.fragment),ue=d(),H=h("div"),g(R.$$.fragment),e=d(),a=h("div"),g(P.$$.fragment),De=d(),fe=h("p"),fe.innerHTML=ot,Qe=d(),g(Ce.$$.fragment),Ae=d(),g(je.$$.fragment),Re=d(),ge=h("div"),g(Ue.$$.fragment),Ye=d(),ce=h("div"),g(Oe.$$.fragment),Ke=d(),Je=h("p"),Je.innerHTML=nt,et=d(),g(Pe.$$.fragment),this.h()},l(o){_(t.$$.fragment,o),f=c(o),n=m(o,"DIV",{class:!0});var u=V(n);_(i.$$.fragment,u),w=c(u),r=m(u,"P",{"data-svelte-h":!0}),k(r)!=="svelte-defg3o"&&(r.textContent=$),X=c(u),Q=m(u,"P",{"data-svelte-h":!0}),k(Q)!=="svelte-q52n56"&&(Q.innerHTML=Y),He=c(u),K=m(u,"P",{"data-svelte-h":!0}),k(K)!=="svelte-hswkmf"&&(K.innerHTML=ae),ye=c(u),x=m(u,"DIV",{class:!0});var _e=V(x);_(re.$$.fragment,_e),Te=c(_e),N=m(_e,"P",{"data-svelte-h":!0}),k(N)!=="svelte-g3gsjf"&&(N.innerHTML=ze),ee=c(_e),_(te.$$.fragment,_e),_e.forEach(s),u.forEach(s),he=c(o),_(I.$$.fragment,o),Fe=c(o),C=m(o,"DIV",{class:!0});var G=V(C);_(j.$$.fragment,G),Ge=c(G),oe=m(G,"P",{"data-svelte-h":!0}),k(oe)!=="svelte-qm3u7f"&&(oe.textContent=ie),we=c(G),B=m(G,"P",{"data-svelte-h":!0}),k(B)!=="svelte-q52n56"&&(B.innerHTML=Ze),ke=c(G),D=m(G,"P",{"data-svelte-h":!0}),k(D)!=="svelte-hswkmf"&&(D.innerHTML=Le),A=c(G),z=m(G,"DIV",{class:!0});var se=V(z);_(U.$$.fragment,se),$e=c(se),J=m(se,"P",{"data-svelte-h":!0}),k(J)!=="svelte-1khcxlz"&&(J.innerHTML=qe),F=c(se),_(q.$$.fragment,se),Ie=c(se),_(O.$$.fragment,se),se.forEach(s),Ne=c(G),S=m(G,"DIV",{class:!0});var be=V(S);_(E.$$.fragment,be),Be=c(be),de=m(be,"P",{"data-svelte-h":!0}),k(de)!=="svelte-s5ko3x"&&(de.textContent=ne),Me=c(be),_(L.$$.fragment,be),be.forEach(s),G.forEach(s),me=c(o),_(W.$$.fragment,o),ue=c(o),H=m(o,"DIV",{class:!0});var We=V(H);_(R.$$.fragment,We),e=c(We),a=m(We,"DIV",{class:!0});var ve=V(a);_(P.$$.fragment,ve),De=c(ve),fe=m(ve,"P",{"data-svelte-h":!0}),k(fe)!=="svelte-1sal4ui"&&(fe.innerHTML=ot),Qe=c(ve),_(Ce.$$.fragment,ve),ve.forEach(s),We.forEach(s),Ae=c(o),_(je.$$.fragment,o),Re=c(o),ge=m(o,"DIV",{class:!0});var Xe=V(ge);_(Ue.$$.fragment,Xe),Ye=c(Xe),ce=m(Xe,"DIV",{class:!0});var Se=V(ce);_(Oe.$$.fragment,Se),Ke=c(Se),Je=m(Se,"P",{"data-svelte-h":!0}),k(Je)!=="svelte-1py4aay"&&(Je.innerHTML=nt),et=c(Se),_(Pe.$$.fragment,Se),Se.forEach(s),Xe.forEach(s),this.h()},h(){Z(x,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),Z(n,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),Z(z,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),Z(S,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),Z(C,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),Z(a,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),Z(H,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),Z(ce,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),Z(ge,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(o,u){b(t,o,u),l(o,f,u),l(o,n,u),b(i,n,null),p(n,w),p(n,r),p(n,X),p(n,Q),p(n,He),p(n,K),p(n,ye),p(n,x),b(re,x,null),p(x,Te),p(x,N),p(x,ee),b(te,x,null),l(o,he,u),b(I,o,u),l(o,Fe,u),l(o,C,u),b(j,C,null),p(C,Ge),p(C,oe),p(C,we),p(C,B),p(C,ke),p(C,D),p(C,A),p(C,z),b(U,z,null),p(z,$e),p(z,J),p(z,F),b(q,z,null),p(z,Ie),b(O,z,null),p(C,Ne),p(C,S),b(E,S,null),p(S,Be),p(S,de),p(S,Me),b(L,S,null),l(o,me,u),b(W,o,u),l(o,ue,u),l(o,H,u),b(R,H,null),p(H,e),p(H,a),b(P,a,null),p(a,De),p(a,fe),p(a,Qe),b(Ce,a,null),l(o,Ae,u),b(je,o,u),l(o,Re,u),l(o,ge,u),b(Ue,ge,null),p(ge,Ye),p(ge,ce),b(Oe,ce,null),p(ce,Ke),p(ce,Je),p(ce,et),b(Pe,ce,null),Ve=!0},p(o,u){const _e={};u&2&&(_e.$$scope={dirty:u,ctx:o}),te.$set(_e);const G={};u&2&&(G.$$scope={dirty:u,ctx:o}),q.$set(G);const se={};u&2&&(se.$$scope={dirty:u,ctx:o}),O.$set(se);const be={};u&2&&(be.$$scope={dirty:u,ctx:o}),L.$set(be);const We={};u&2&&(We.$$scope={dirty:u,ctx:o}),Ce.$set(We);const ve={};u&2&&(ve.$$scope={dirty:u,ctx:o}),Pe.$set(ve)},i(o){Ve||(v(t.$$.fragment,o),v(i.$$.fragment,o),v(re.$$.fragment,o),v(te.$$.fragment,o),v(I.$$.fragment,o),v(j.$$.fragment,o),v(U.$$.fragment,o),v(q.$$.fragment,o),v(O.$$.fragment,o),v(E.$$.fragment,o),v(L.$$.fragment,o),v(W.$$.fragment,o),v(R.$$.fragment,o),v(P.$$.fragment,o),v(Ce.$$.fragment,o),v(je.$$.fragment,o),v(Ue.$$.fragment,o),v(Oe.$$.fragment,o),v(Pe.$$.fragment,o),Ve=!0)},o(o){y(t.$$.fragment,o),y(i.$$.fragment,o),y(re.$$.fragment,o),y(te.$$.fragment,o),y(I.$$.fragment,o),y(j.$$.fragment,o),y(U.$$.fragment,o),y(q.$$.fragment,o),y(O.$$.fragment,o),y(E.$$.fragment,o),y(L.$$.fragment,o),y(W.$$.fragment,o),y(R.$$.fragment,o),y(P.$$.fragment,o),y(Ce.$$.fragment,o),y(je.$$.fragment,o),y(Ue.$$.fragment,o),y(Oe.$$.fragment,o),y(Pe.$$.fragment,o),Ve=!1},d(o){o&&(s(f),s(n),s(he),s(Fe),s(C),s(me),s(ue),s(H),s(Ae),s(Re),s(ge)),T(t,o),T(i),T(re),T(te),T(I,o),T(j),T(U),T(q),T(O),T(E),T(L),T(W,o),T(R),T(P),T(Ce),T(je,o),T(Ue),T(Oe),T(Pe)}}}function wt(M){let t,f;return t=new pt({props:{$$slots:{default:[Tt]},$$scope:{ctx:M}}}),{c(){g(t.$$.fragment)},l(n){_(t.$$.fragment,n)},m(n,i){b(t,n,i),f=!0},p(n,i){const w={};i&2&&(w.$$scope={dirty:i,ctx:n}),t.$set(w)},i(n){f||(v(t.$$.fragment,n),f=!0)},o(n){y(t.$$.fragment,n),f=!1},d(n){T(t,n)}}}function kt(M){let t,f,n,i,w,r="<em>This model was released on 2024-04-22 and added to Hugging Face Transformers on 2024-04-24.</em>",$,X,Q,Y,He='<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/> <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat"/> <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/> <img alt="Tensor parallelism" src="https://img.shields.io/badge/Tensor%20parallelism-06b6d4?style=flat&amp;logoColor=white"/>',K,ae,ye,x,re='The Phi-3 model was proposed in <a href="https://huggingface.co/papers/2404.14219" rel="nofollow">Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone</a> by Microsoft.',Te,N,ze,ee,te="The abstract from the Phi-3 paper is the following:",he,I,Fe="We introduce phi-3-mini, a 3.8 billion parameter language model trained on 3.3 trillion tokens, whose overall performance, as measured by both academic benchmarks and internal testing, rivals that of models such as Mixtral 8x7B and GPT-3.5 (e.g., phi-3-mini achieves 69% on MMLU and 8.38 on MT-bench), despite being small enough to be deployed on a phone. The innovation lies entirely in our dataset for training, a scaled-up version of the one used for phi-2, composed of heavily filtered web data and synthetic data. The model is also further aligned for robustness, safety, and chat format. We also provide some initial parameter-scaling results with a 7B and 14B models trained for 4.8T tokens, called phi-3-small and phi-3-medium, both significantly more capable than phi-3-mini (e.g., respectively 75% and 78% on MMLU, and 8.7 and 8.9 on MT-bench).",C,j,Ge='The original code for Phi-3 can be found <a href="https://huggingface.co/microsoft/Phi-3-mini-4k-instruct" rel="nofollow">here</a>.',oe,ie,we,B,Ze='<li>This model is very similar to <code>Llama</code> with the main difference of <code>Phi3SuScaledRotaryEmbedding</code> and <code>Phi3YarnScaledRotaryEmbedding</code>, where they are used to extend the context of the rotary embeddings. The query, key and values are fused, and the MLP’s up and gate projection layers are also fused.</li> <li>The tokenizer used for this model is identical to the <a href="/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaTokenizer">LlamaTokenizer</a>, with the exception of additional tokens.</li>',ke,D,Le,A,z,U,$e,J,qe,F,q,Ie,O,Ne=`This is the configuration class to store the configuration of a <a href="/docs/transformers/v4.56.2/en/model_doc/phi3#transformers.Phi3Model">Phi3Model</a>. It is used to instantiate a Phi-3
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the
<a href="https://huggingface.co/microsoft/Phi-3-mini-4k-instruct" rel="nofollow">microsoft/Phi-3-mini-4k-instruct</a>.`,S,E,Be=`Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> and can be used to control the model outputs. Read the
documentation from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> for more information.`,de,ne,Me,L,me,W,ue,H,R;return X=new pe({props:{title:"Phi-3",local:"phi-3",headingTag:"h1"}}),ae=new pe({props:{title:"Overview",local:"overview",headingTag:"h2"}}),N=new pe({props:{title:"Summary",local:"summary",headingTag:"h3"}}),ie=new pe({props:{title:"Usage tips",local:"usage-tips",headingTag:"h2"}}),D=new pe({props:{title:"How to use Phi-3",local:"how-to-use-phi-3",headingTag:"h2"}}),A=new Ee({props:{warning:!0,$$slots:{default:[mt]},$$scope:{ctx:M}}}),U=new tt({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Nb2RlbEZvckNhdXNhbExNJTJDJTIwQXV0b1Rva2VuaXplciUwQSUwQW1vZGVsJTIwJTNEJTIwQXV0b01vZGVsRm9yQ2F1c2FsTE0uZnJvbV9wcmV0cmFpbmVkKCUyMm1pY3Jvc29mdCUyRlBoaS0zLW1pbmktNGstaW5zdHJ1Y3QlMjIpJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIybWljcm9zb2Z0JTJGUGhpLTMtbWluaS00ay1pbnN0cnVjdCUyMiklMEElMEFtZXNzYWdlcyUyMCUzRCUyMCU1QiU3QiUyMnJvbGUlMjIlM0ElMjAlMjJ1c2VyJTIyJTJDJTIwJTIyY29udGVudCUyMiUzQSUyMCUyMkNhbiUyMHlvdSUyMHByb3ZpZGUlMjB3YXlzJTIwdG8lMjBlYXQlMjBjb21iaW5hdGlvbnMlMjBvZiUyMGJhbmFuYXMlMjBhbmQlMjBkcmFnb25mcnVpdHMlM0YlMjIlN0QlNUQlMEFpbnB1dHMlMjAlM0QlMjB0b2tlbml6ZXIuYXBwbHlfY2hhdF90ZW1wbGF0ZShtZXNzYWdlcyUyQyUyMGFkZF9nZW5lcmF0aW9uX3Byb21wdCUzRFRydWUlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQSUwQW91dHB1dHMlMjAlM0QlMjBtb2RlbC5nZW5lcmF0ZShpbnB1dHMlMkMlMjBtYXhfbmV3X3Rva2VucyUzRDMyKSUwQXRleHQlMjAlM0QlMjB0b2tlbml6ZXIuYmF0Y2hfZGVjb2RlKG91dHB1dHMpJTVCMCU1RCUwQXByaW50KHRleHQp",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoModelForCausalLM, AutoTokenizer

<span class="hljs-meta">&gt;&gt;&gt; </span>model = AutoModelForCausalLM.from_pretrained(<span class="hljs-string">&quot;microsoft/Phi-3-mini-4k-instruct&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;microsoft/Phi-3-mini-4k-instruct&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>messages = [{<span class="hljs-string">&quot;role&quot;</span>: <span class="hljs-string">&quot;user&quot;</span>, <span class="hljs-string">&quot;content&quot;</span>: <span class="hljs-string">&quot;Can you provide ways to eat combinations of bananas and dragonfruits?&quot;</span>}]
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=<span class="hljs-literal">True</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model.generate(inputs, max_new_tokens=<span class="hljs-number">32</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>text = tokenizer.batch_decode(outputs)[<span class="hljs-number">0</span>]
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">print</span>(text)
&lt;|user|&gt; Can you provide ways to eat combinations of bananas <span class="hljs-keyword">and</span> dragonfruits?&lt;|end|&gt;&lt;|assistant|&gt; Certainly! Bananas <span class="hljs-keyword">and</span> dragonfruits can be combined <span class="hljs-keyword">in</span> various delicious ways. Here are some creative ideas <span class="hljs-keyword">for</span> incorporating both fruits`,wrap:!1}}),J=new pe({props:{title:"Phi3Config",local:"transformers.Phi3Config",headingTag:"h2"}}),q=new le({props:{name:"class transformers.Phi3Config",anchor:"transformers.Phi3Config",parameters:[{name:"vocab_size",val:" = 32064"},{name:"hidden_size",val:" = 3072"},{name:"intermediate_size",val:" = 8192"},{name:"num_hidden_layers",val:" = 32"},{name:"num_attention_heads",val:" = 32"},{name:"num_key_value_heads",val:" = None"},{name:"resid_pdrop",val:" = 0.0"},{name:"embd_pdrop",val:" = 0.0"},{name:"attention_dropout",val:" = 0.0"},{name:"hidden_act",val:" = 'silu'"},{name:"max_position_embeddings",val:" = 4096"},{name:"original_max_position_embeddings",val:" = 4096"},{name:"initializer_range",val:" = 0.02"},{name:"rms_norm_eps",val:" = 1e-05"},{name:"use_cache",val:" = True"},{name:"tie_word_embeddings",val:" = False"},{name:"rope_theta",val:" = 10000.0"},{name:"rope_scaling",val:" = None"},{name:"partial_rotary_factor",val:" = 1.0"},{name:"bos_token_id",val:" = 1"},{name:"eos_token_id",val:" = 32000"},{name:"pad_token_id",val:" = 32000"},{name:"sliding_window",val:" = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.Phi3Config.vocab_size",description:`<strong>vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 32064) &#x2014;
Vocabulary size of the Phi-3 model. Defines the number of different tokens that can be represented by the
<code>inputs_ids</code> passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/phi3#transformers.Phi3Model">Phi3Model</a>.`,name:"vocab_size"},{anchor:"transformers.Phi3Config.hidden_size",description:`<strong>hidden_size</strong> (<code>int</code>, <em>optional</em>, defaults to 3072) &#x2014;
Dimension of the hidden representations.`,name:"hidden_size"},{anchor:"transformers.Phi3Config.intermediate_size",description:`<strong>intermediate_size</strong> (<code>int</code>, <em>optional</em>, defaults to 8192) &#x2014;
Dimension of the MLP representations.`,name:"intermediate_size"},{anchor:"transformers.Phi3Config.num_hidden_layers",description:`<strong>num_hidden_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 32) &#x2014;
Number of hidden layers in the Transformer decoder.`,name:"num_hidden_layers"},{anchor:"transformers.Phi3Config.num_attention_heads",description:`<strong>num_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 32) &#x2014;
Number of attention heads for each attention layer in the Transformer decoder.`,name:"num_attention_heads"},{anchor:"transformers.Phi3Config.num_key_value_heads",description:`<strong>num_key_value_heads</strong> (<code>int</code>, <em>optional</em>) &#x2014;
This is the number of key_value heads that should be used to implement Grouped Query Attention. If
<code>num_key_value_heads=num_attention_heads</code>, the model will use Multi Head Attention (MHA), if
<code>num_key_value_heads=1</code> the model will use Multi Query Attention (MQA) otherwise GQA is used. When
converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
by meanpooling all the original heads within that group. For more details, check out <a href="https://huggingface.co/papers/2305.13245" rel="nofollow">this
paper</a>. If it is not specified, will default to
<code>num_attention_heads</code>.`,name:"num_key_value_heads"},{anchor:"transformers.Phi3Config.resid_pdrop",description:`<strong>resid_pdrop</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
Dropout probability for mlp outputs.`,name:"resid_pdrop"},{anchor:"transformers.Phi3Config.embd_pdrop",description:`<strong>embd_pdrop</strong> (<code>int</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The dropout ratio for the embeddings.`,name:"embd_pdrop"},{anchor:"transformers.Phi3Config.attention_dropout",description:`<strong>attention_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The dropout ratio after computing the attention scores.`,name:"attention_dropout"},{anchor:"transformers.Phi3Config.hidden_act",description:`<strong>hidden_act</strong> (<code>str</code> or <code>function</code>, <em>optional</em>, defaults to <code>&quot;silu&quot;</code>) &#x2014;
The non-linear activation function (function or string) in the decoder.`,name:"hidden_act"},{anchor:"transformers.Phi3Config.max_position_embeddings",description:`<strong>max_position_embeddings</strong> (<code>int</code>, <em>optional</em>, defaults to 4096) &#x2014;
The maximum sequence length that this model might ever be used with.`,name:"max_position_embeddings"},{anchor:"transformers.Phi3Config.original_max_position_embeddings",description:`<strong>original_max_position_embeddings</strong> (<code>int</code>, <em>optional</em>, defaults to 4096) &#x2014;
The maximum sequence length that this model was trained with. This is used to determine the size of the
original RoPE embeddings when using long scaling.`,name:"original_max_position_embeddings"},{anchor:"transformers.Phi3Config.initializer_range",description:`<strong>initializer_range</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.`,name:"initializer_range"},{anchor:"transformers.Phi3Config.rms_norm_eps",description:`<strong>rms_norm_eps</strong> (<code>float</code>, <em>optional</em>, defaults to 1e-05) &#x2014;
The epsilon value used for the RMSNorm.`,name:"rms_norm_eps"},{anchor:"transformers.Phi3Config.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not the model should return the last key/values attentions (not used by all models). Only
relevant if <code>config.is_decoder=True</code>. Whether to tie weight embeddings or not.`,name:"use_cache"},{anchor:"transformers.Phi3Config.tie_word_embeddings",description:`<strong>tie_word_embeddings</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether to tie weight embeddings`,name:"tie_word_embeddings"},{anchor:"transformers.Phi3Config.rope_theta",description:`<strong>rope_theta</strong> (<code>float</code>, <em>optional</em>, defaults to 10000.0) &#x2014;
The base period of the RoPE embeddings.`,name:"rope_theta"},{anchor:"transformers.Phi3Config.rope_scaling",description:`<strong>rope_scaling</strong> (<code>dict</code>, <em>optional</em>) &#x2014;
The scaling strategy for the RoPE embeddings. If <code>None</code>, no scaling is applied. If a dictionary, it must
contain the following keys: <code>type</code>, <code>short_factor</code> and <code>long_factor</code>. The <code>type</code> must be <code>longrope</code> and
the <code>short_factor</code> and <code>long_factor</code> must be lists of numbers with the same length as the hidden size
divided by the number of attention heads divided by 2.`,name:"rope_scaling"},{anchor:"transformers.Phi3Config.partial_rotary_factor",description:`<strong>partial_rotary_factor</strong> (<code>float</code>, <em>optional</em>, defaults to 1.0) &#x2014;
Percentage of the query and keys which will have rotary embedding. Must be between 0.0 and 1.0.`,name:"partial_rotary_factor"},{anchor:"transformers.Phi3Config.bos_token_id",description:`<strong>bos_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 1) &#x2014;
The id of the &#x201C;beginning-of-sequence&#x201D; token.`,name:"bos_token_id"},{anchor:"transformers.Phi3Config.eos_token_id",description:`<strong>eos_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 32000) &#x2014;
The id of the &#x201C;end-of-sequence&#x201D; token.`,name:"eos_token_id"},{anchor:"transformers.Phi3Config.pad_token_id",description:`<strong>pad_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 32000) &#x2014;
The id of the padding token.`,name:"pad_token_id"},{anchor:"transformers.Phi3Config.sliding_window",description:`<strong>sliding_window</strong> (<code>int</code>, <em>optional</em>) &#x2014;
Sliding window attention window size. If <code>None</code>, no sliding window is applied.`,name:"sliding_window"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/phi3/configuration_phi3.py#L25"}}),ne=new st({props:{anchor:"transformers.Phi3Config.example",$$slots:{default:[ut]},$$scope:{ctx:M}}}),L=new lt({props:{pytorch:!0,tensorflow:!1,jax:!1,$$slots:{pytorch:[wt]},$$scope:{ctx:M}}}),W=new ht({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/phi3.md"}}),{c(){t=h("meta"),f=d(),n=h("p"),i=d(),w=h("p"),w.innerHTML=r,$=d(),g(X.$$.fragment),Q=d(),Y=h("div"),Y.innerHTML=He,K=d(),g(ae.$$.fragment),ye=d(),x=h("p"),x.innerHTML=re,Te=d(),g(N.$$.fragment),ze=d(),ee=h("p"),ee.textContent=te,he=d(),I=h("p"),I.textContent=Fe,C=d(),j=h("p"),j.innerHTML=Ge,oe=d(),g(ie.$$.fragment),we=d(),B=h("ul"),B.innerHTML=Ze,ke=d(),g(D.$$.fragment),Le=d(),g(A.$$.fragment),z=d(),g(U.$$.fragment),$e=d(),g(J.$$.fragment),qe=d(),F=h("div"),g(q.$$.fragment),Ie=d(),O=h("p"),O.innerHTML=Ne,S=d(),E=h("p"),E.innerHTML=Be,de=d(),g(ne.$$.fragment),Me=d(),g(L.$$.fragment),me=d(),g(W.$$.fragment),ue=d(),H=h("p"),this.h()},l(e){const a=ct("svelte-u9bgzb",document.head);t=m(a,"META",{name:!0,content:!0}),a.forEach(s),f=c(e),n=m(e,"P",{}),V(n).forEach(s),i=c(e),w=m(e,"P",{"data-svelte-h":!0}),k(w)!=="svelte-1y8xz3w"&&(w.innerHTML=r),$=c(e),_(X.$$.fragment,e),Q=c(e),Y=m(e,"DIV",{class:!0,"data-svelte-h":!0}),k(Y)!=="svelte-is43db"&&(Y.innerHTML=He),K=c(e),_(ae.$$.fragment,e),ye=c(e),x=m(e,"P",{"data-svelte-h":!0}),k(x)!=="svelte-1wjpxev"&&(x.innerHTML=re),Te=c(e),_(N.$$.fragment,e),ze=c(e),ee=m(e,"P",{"data-svelte-h":!0}),k(ee)!=="svelte-1iah8ch"&&(ee.textContent=te),he=c(e),I=m(e,"P",{"data-svelte-h":!0}),k(I)!=="svelte-10zying"&&(I.textContent=Fe),C=c(e),j=m(e,"P",{"data-svelte-h":!0}),k(j)!=="svelte-1deox1y"&&(j.innerHTML=Ge),oe=c(e),_(ie.$$.fragment,e),we=c(e),B=m(e,"UL",{"data-svelte-h":!0}),k(B)!=="svelte-a66zbt"&&(B.innerHTML=Ze),ke=c(e),_(D.$$.fragment,e),Le=c(e),_(A.$$.fragment,e),z=c(e),_(U.$$.fragment,e),$e=c(e),_(J.$$.fragment,e),qe=c(e),F=m(e,"DIV",{class:!0});var P=V(F);_(q.$$.fragment,P),Ie=c(P),O=m(P,"P",{"data-svelte-h":!0}),k(O)!=="svelte-qbqknh"&&(O.innerHTML=Ne),S=c(P),E=m(P,"P",{"data-svelte-h":!0}),k(E)!=="svelte-1ek1ss9"&&(E.innerHTML=Be),de=c(P),_(ne.$$.fragment,P),P.forEach(s),Me=c(e),_(L.$$.fragment,e),me=c(e),_(W.$$.fragment,e),ue=c(e),H=m(e,"P",{}),V(H).forEach(s),this.h()},h(){Z(t,"name","hf:doc:metadata"),Z(t,"content",$t),Z(Y,"class","flex flex-wrap space-x-1"),Z(F,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,a){p(document.head,t),l(e,f,a),l(e,n,a),l(e,i,a),l(e,w,a),l(e,$,a),b(X,e,a),l(e,Q,a),l(e,Y,a),l(e,K,a),b(ae,e,a),l(e,ye,a),l(e,x,a),l(e,Te,a),b(N,e,a),l(e,ze,a),l(e,ee,a),l(e,he,a),l(e,I,a),l(e,C,a),l(e,j,a),l(e,oe,a),b(ie,e,a),l(e,we,a),l(e,B,a),l(e,ke,a),b(D,e,a),l(e,Le,a),b(A,e,a),l(e,z,a),b(U,e,a),l(e,$e,a),b(J,e,a),l(e,qe,a),l(e,F,a),b(q,F,null),p(F,Ie),p(F,O),p(F,S),p(F,E),p(F,de),b(ne,F,null),l(e,Me,a),b(L,e,a),l(e,me,a),b(W,e,a),l(e,ue,a),l(e,H,a),R=!0},p(e,[a]){const P={};a&2&&(P.$$scope={dirty:a,ctx:e}),A.$set(P);const De={};a&2&&(De.$$scope={dirty:a,ctx:e}),ne.$set(De);const fe={};a&2&&(fe.$$scope={dirty:a,ctx:e}),L.$set(fe)},i(e){R||(v(X.$$.fragment,e),v(ae.$$.fragment,e),v(N.$$.fragment,e),v(ie.$$.fragment,e),v(D.$$.fragment,e),v(A.$$.fragment,e),v(U.$$.fragment,e),v(J.$$.fragment,e),v(q.$$.fragment,e),v(ne.$$.fragment,e),v(L.$$.fragment,e),v(W.$$.fragment,e),R=!0)},o(e){y(X.$$.fragment,e),y(ae.$$.fragment,e),y(N.$$.fragment,e),y(ie.$$.fragment,e),y(D.$$.fragment,e),y(A.$$.fragment,e),y(U.$$.fragment,e),y(J.$$.fragment,e),y(q.$$.fragment,e),y(ne.$$.fragment,e),y(L.$$.fragment,e),y(W.$$.fragment,e),R=!1},d(e){e&&(s(f),s(n),s(i),s(w),s($),s(Q),s(Y),s(K),s(ye),s(x),s(Te),s(ze),s(ee),s(he),s(I),s(C),s(j),s(oe),s(we),s(B),s(ke),s(Le),s(z),s($e),s(qe),s(F),s(Me),s(me),s(ue),s(H)),s(t),T(X,e),T(ae,e),T(N,e),T(ie,e),T(D,e),T(A,e),T(U,e),T(J,e),T(q),T(ne),T(L,e),T(W,e)}}}const $t='{"title":"Phi-3","local":"phi-3","sections":[{"title":"Overview","local":"overview","sections":[{"title":"Summary","local":"summary","sections":[],"depth":3}],"depth":2},{"title":"Usage tips","local":"usage-tips","sections":[],"depth":2},{"title":"How to use Phi-3","local":"how-to-use-phi-3","sections":[],"depth":2},{"title":"Phi3Config","local":"transformers.Phi3Config","sections":[],"depth":2},{"title":"Phi3Model","local":"transformers.Phi3Model","sections":[],"depth":2},{"title":"Phi3ForCausalLM","local":"transformers.Phi3ForCausalLM","sections":[],"depth":2},{"title":"Phi3ForSequenceClassification","local":"transformers.Phi3ForSequenceClassification","sections":[],"depth":2},{"title":"Phi3ForTokenClassification","local":"transformers.Phi3ForTokenClassification","sections":[],"depth":2}],"depth":1}';function Mt(M){return rt(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class jt extends it{constructor(t){super(),dt(this,t,Mt,kt,at,{})}}export{jt as component};
