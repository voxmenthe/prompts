import{s as Le,o as qe,n as Ue}from"../chunks/scheduler.18a86fab.js";import{S as Ge,i as We,g as m,s as d,r as _,A as Be,h,f as s,c,j as de,x as C,u as b,k as ie,y as g,a as r,v as y,d as M,t as v,w as T}from"../chunks/index.98837b22.js";import{T as Ie}from"../chunks/Tip.77304350.js";import{D as we}from"../chunks/Docstring.a1ef7999.js";import{C as Fe}from"../chunks/CodeBlock.8d0c2e8a.js";import{F as Se,M as Ee}from"../chunks/Markdown.ae01904b.js";import{E as ze}from"../chunks/ExampleCodeBlock.8c3ee1f9.js";import{H as fe,E as Ze}from"../chunks/getInferenceSnippets.06c2775f.js";function Ne($){let o,f="Phi-3.5-MoE-instruct has been integrated in the development version (4.44.2.dev) of <code>transformers</code>. Until the official version is released through <code>pip</code>, ensure that you are doing the following:",t,i,w="<li>When loading the model, ensure that <code>trust_remote_code=True</code> is passed as an argument of the <code>from_pretrained()</code> function.</li>",l,k,N="The current <code>transformers</code> version can be verified with: <code>pip list | grep transformers</code>.",z,U,ce="Examples of required packages:",L,P,O;return P=new Fe({props:{code:"Zmxhc2hfYXR0biUzRCUzRDIuNS44JTBBdG9yY2glM0QlM0QyLjMuMSUwQWFjY2VsZXJhdGUlM0QlM0QwLjMxLjAlMEF0cmFuc2Zvcm1lcnMlM0QlM0Q0LjQzLjA=",highlighted:`<span class="hljs-attribute">flash_attn</span>==<span class="hljs-number">2</span>.<span class="hljs-number">5</span>.<span class="hljs-number">8</span>
<span class="hljs-attribute">torch</span>==<span class="hljs-number">2</span>.<span class="hljs-number">3</span>.<span class="hljs-number">1</span>
<span class="hljs-attribute">accelerate</span>==<span class="hljs-number">0</span>.<span class="hljs-number">31</span>.<span class="hljs-number">0</span>
<span class="hljs-attribute">transformers</span>==<span class="hljs-number">4</span>.<span class="hljs-number">43</span>.<span class="hljs-number">0</span>`,wrap:!1}}),{c(){o=m("p"),o.innerHTML=f,t=d(),i=m("ul"),i.innerHTML=w,l=d(),k=m("p"),k.innerHTML=N,z=d(),U=m("p"),U.textContent=ce,L=d(),_(P.$$.fragment)},l(p){o=h(p,"P",{"data-svelte-h":!0}),C(o)!=="svelte-16ml7y6"&&(o.innerHTML=f),t=c(p),i=h(p,"UL",{"data-svelte-h":!0}),C(i)!=="svelte-b3j7h0"&&(i.innerHTML=w),l=c(p),k=h(p,"P",{"data-svelte-h":!0}),C(k)!=="svelte-pmzjti"&&(k.innerHTML=N),z=c(p),U=h(p,"P",{"data-svelte-h":!0}),C(U)!=="svelte-1njcx0o"&&(U.textContent=ce),L=c(p),b(P.$$.fragment,p)},m(p,x){r(p,o,x),r(p,t,x),r(p,i,x),r(p,l,x),r(p,k,x),r(p,z,x),r(p,U,x),r(p,L,x),y(P,p,x),O=!0},p:Ue,i(p){O||(M(P.$$.fragment,p),O=!0)},o(p){v(P.$$.fragment,p),O=!1},d(p){p&&(s(o),s(t),s(i),s(l),s(k),s(z),s(U),s(L)),T(P,p)}}}function Ae($){let o,f="Example:",t,i,w;return i=new Fe({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMFBoaW1vZU1vZGVsJTJDJTIwUGhpbW9lQ29uZmlnJTBBJTIzJTIwSW5pdGlhbGl6aW5nJTIwYSUyMFBoaS0zJTIwc3R5bGUlMjBjb25maWd1cmF0aW9uJTBBY29uZmlndXJhdGlvbiUyMCUzRCUyMFBoaW1vZUNvbmZpZy5mcm9tX3ByZXRyYWluZWQoJTIybWljcm9zb2Z0JTJGUGhpLTMuNS1Nb0UtaW5zdHJ1Y3QlMjIpJTBBJTIzJTIwSW5pdGlhbGl6aW5nJTIwYSUyMG1vZGVsJTIwZnJvbSUyMHRoZSUyMGNvbmZpZ3VyYXRpb24lMEFtb2RlbCUyMCUzRCUyMFBoaW1vZU1vZGVsKGNvbmZpZ3VyYXRpb24pJTBBJTIzJTIwQWNjZXNzaW5nJTIwdGhlJTIwbW9kZWwlMjBjb25maWd1cmF0aW9uJTBBY29uZmlndXJhdGlvbiUyMCUzRCUyMG1vZGVsLmNvbmZpZw==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> PhimoeModel, PhimoeConfig
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a Phi-3 style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = PhimoeConfig.from_pretrained(<span class="hljs-string">&quot;microsoft/Phi-3.5-MoE-instruct&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a model from the configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = PhimoeModel(configuration)
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Accessing the model configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = model.config`,wrap:!1}}),{c(){o=m("p"),o.textContent=f,t=d(),_(i.$$.fragment)},l(l){o=h(l,"P",{"data-svelte-h":!0}),C(o)!=="svelte-11lpom8"&&(o.textContent=f),t=c(l),b(i.$$.fragment,l)},m(l,k){r(l,o,k),r(l,t,k),y(i,l,k),w=!0},p:Ue,i(l){w||(M(i.$$.fragment,l),w=!0)},o(l){v(i.$$.fragment,l),w=!1},d(l){l&&(s(o),s(t)),T(i,l)}}}function Re($){let o,f=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){o=m("p"),o.innerHTML=f},l(t){o=h(t,"P",{"data-svelte-h":!0}),C(o)!=="svelte-fincs2"&&(o.innerHTML=f)},m(t,i){r(t,o,i)},p:Ue,d(t){t&&s(o)}}}function He($){let o,f=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){o=m("p"),o.innerHTML=f},l(t){o=h(t,"P",{"data-svelte-h":!0}),C(o)!=="svelte-fincs2"&&(o.innerHTML=f)},m(t,i){r(t,o,i)},p:Ue,d(t){t&&s(o)}}}function Ve($){let o,f="Example:",t,i,w;return i=new Fe({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBQaGltb2VGb3JDYXVzYWxMTSUwQW1vZGVsJTIwJTNEJTIwUGhpbW9lRm9yQ2F1c2FsTE0uZnJvbV9wcmV0cmFpbmVkKCUyMm1pY3Jvc29mdCUyRlBoaS0zLjUtTW9FLWluc3RydWN0JTIyKSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMm1pY3Jvc29mdCUyRlBoaS0zLjUtTW9FLWluc3RydWN0JTIyKSUwQXByb21wdCUyMCUzRCUyMCUyMkhleSUyQyUyMGFyZSUyMHlvdSUyMGNvbnNjaW91cyUzRiUyMENhbiUyMHlvdSUyMHRhbGslMjB0byUyMG1lJTNGJTIyJTBBaW5wdXRzJTIwJTNEJTIwdG9rZW5pemVyKHByb21wdCUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpJTBBJTIzJTIwR2VuZXJhdGUlMEFnZW5lcmF0ZV9pZHMlMjAlM0QlMjBtb2RlbC5nZW5lcmF0ZShpbnB1dHMuaW5wdXRfaWRzJTJDJTIwbWF4X2xlbmd0aCUzRDMwKSUwQXRva2VuaXplci5iYXRjaF9kZWNvZGUoZ2VuZXJhdGVfaWRzJTJDJTIwc2tpcF9zcGVjaWFsX3Rva2VucyUzRFRydWUlMkMlMjBjbGVhbl91cF90b2tlbml6YXRpb25fc3BhY2VzJTNERmFsc2UpJTVCMCU1RA==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, PhimoeForCausalLM
<span class="hljs-meta">&gt;&gt;&gt; </span>model = PhimoeForCausalLM.from_pretrained(<span class="hljs-string">&quot;microsoft/Phi-3.5-MoE-instruct&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;microsoft/Phi-3.5-MoE-instruct&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>prompt = <span class="hljs-string">&quot;Hey, are you conscious? Can you talk to me?&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(prompt, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Generate</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>generate_ids = model.generate(inputs.input_ids, max_length=<span class="hljs-number">30</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer.batch_decode(generate_ids, skip_special_tokens=<span class="hljs-literal">True</span>, clean_up_tokenization_spaces=<span class="hljs-literal">False</span>)[<span class="hljs-number">0</span>]
<span class="hljs-string">&quot;Hey, are you conscious? Can you talk to me?\\nI&#x27;m not conscious, but I can talk to you.&quot;</span>`,wrap:!1}}),{c(){o=m("p"),o.textContent=f,t=d(),_(i.$$.fragment)},l(l){o=h(l,"P",{"data-svelte-h":!0}),C(o)!=="svelte-11lpom8"&&(o.textContent=f),t=c(l),b(i.$$.fragment,l)},m(l,k){r(l,o,k),r(l,t,k),y(i,l,k),w=!0},p:Ue,i(l){w||(M(i.$$.fragment,l),w=!0)},o(l){v(i.$$.fragment,l),w=!1},d(l){l&&(s(o),s(t)),T(i,l)}}}function Oe($){let o,f=`Most generation-controlling parameters are set in <code>generation_config</code> which, if not passed, will be set to the
model’s default generation configuration. You can override any <code>generation_config</code> by passing the corresponding
parameters to generate(), e.g. <code>.generate(inputs, num_beams=4, do_sample=True)</code>.`,t,i,w=`For an overview of generation strategies and code examples, check out the <a href="../generation_strategies">following
guide</a>.`;return{c(){o=m("p"),o.innerHTML=f,t=d(),i=m("p"),i.innerHTML=w},l(l){o=h(l,"P",{"data-svelte-h":!0}),C(o)!=="svelte-1c5u34l"&&(o.innerHTML=f),t=c(l),i=h(l,"P",{"data-svelte-h":!0}),C(i)!=="svelte-fvlq1g"&&(i.innerHTML=w)},m(l,k){r(l,o,k),r(l,t,k),r(l,i,k)},p:Ue,d(l){l&&(s(o),s(t),s(i))}}}function Qe($){let o,f=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){o=m("p"),o.innerHTML=f},l(t){o=h(t,"P",{"data-svelte-h":!0}),C(o)!=="svelte-fincs2"&&(o.innerHTML=f)},m(t,i){r(t,o,i)},p:Ue,d(t){t&&s(o)}}}function De($){let o,f,t,i,w,l,k="The bare Phimoe Model outputting raw hidden-states without any specific head on top.",N,z,U=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,ce,L,P=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,O,p,x,ge,A,ke='The <a href="/docs/transformers/v4.56.2/en/model_doc/phimoe#transformers.PhimoeModel">PhimoeModel</a> forward method, overrides the <code>__call__</code> special method.',ee,oe,pe,W,Ce,q,B,Pe,I,S,_e,R,Je='The <a href="/docs/transformers/v4.56.2/en/model_doc/phimoe#transformers.PhimoeForCausalLM">PhimoeForCausalLM</a> forward method, overrides the <code>__call__</code> special method.',be,G,ye,F,Me,j,Q,te,ne,H="Generates sequences of token ids for models with a language modeling head.",se,D,ae,le,ve,J,X,Y,E,Z,Te,K,$e="The <code>GenericForSequenceClassification</code> forward method, overrides the <code>__call__</code> special method.",e,a,V;return o=new fe({props:{title:"PhimoeModel",local:"transformers.PhimoeModel",headingTag:"h2"}}),i=new we({props:{name:"class transformers.PhimoeModel",anchor:"transformers.PhimoeModel",parameters:[{name:"config",val:": PhimoeConfig"}],parametersDescription:[{anchor:"transformers.PhimoeModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/phimoe#transformers.PhimoeConfig">PhimoeConfig</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/phimoe/modeling_phimoe.py#L917"}}),x=new we({props:{name:"forward",anchor:"transformers.PhimoeModel.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"output_router_logits",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"}],parametersDescription:[{anchor:"transformers.PhimoeModel.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.PhimoeModel.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.PhimoeModel.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.PhimoeModel.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.PhimoeModel.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.PhimoeModel.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.PhimoeModel.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.PhimoeModel.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.PhimoeModel.forward.output_router_logits",description:`<strong>output_router_logits</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the logits of all the routers. They are useful for computing the router loss, and
should not be returned during inference.`,name:"output_router_logits"},{anchor:"transformers.PhimoeModel.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/phimoe/modeling_phimoe.py#L941",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <code>transformers.modeling_outputs.MoeModelOutputWithPast</code> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/phimoe#transformers.PhimoeConfig"
>PhimoeConfig</a>) and inputs.</p>
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
`}}),oe=new Ie({props:{$$slots:{default:[Re]},$$scope:{ctx:$}}}),W=new fe({props:{title:"PhimoeForCausalLM",local:"transformers.PhimoeForCausalLM",headingTag:"h2"}}),B=new we({props:{name:"class transformers.PhimoeForCausalLM",anchor:"transformers.PhimoeForCausalLM",parameters:[{name:"config",val:""}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/phimoe/modeling_phimoe.py#L1198"}}),S=new we({props:{name:"forward",anchor:"transformers.PhimoeForCausalLM.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"output_router_logits",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"},{name:"logits_to_keep",val:": typing.Union[int, torch.Tensor] = 0"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.PhimoeForCausalLM.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.PhimoeForCausalLM.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.PhimoeForCausalLM.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.PhimoeForCausalLM.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.PhimoeForCausalLM.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.PhimoeForCausalLM.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.PhimoeForCausalLM.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.PhimoeForCausalLM.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.PhimoeForCausalLM.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.PhimoeForCausalLM.forward.output_router_logits",description:`<strong>output_router_logits</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the logits of all the routers. They are useful for computing the router loss, and
should not be returned during inference.`,name:"output_router_logits"},{anchor:"transformers.PhimoeForCausalLM.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"},{anchor:"transformers.PhimoeForCausalLM.forward.logits_to_keep",description:`<strong>logits_to_keep</strong> (<code>Union[int, torch.Tensor]</code>, defaults to <code>0</code>) &#x2014;
If an <code>int</code>, compute logits for the last <code>logits_to_keep</code> tokens. If <code>0</code>, calculate logits for all
<code>input_ids</code> (special case). Only last token logits are needed for generation, and calculating them only for that
token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
If a <code>torch.Tensor</code>, must be 1D corresponding to the indices to keep in the sequence length dimension.
This is useful when using packed tensor format (single dimension for batch and sequence length).`,name:"logits_to_keep"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/phimoe/modeling_phimoe.py#L1212",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <code>transformers.modeling_outputs.MoeCausalLMOutputWithPast</code> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/phimoe#transformers.PhimoeConfig"
>PhimoeConfig</a>) and inputs.</p>
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
`}}),G=new Ie({props:{$$slots:{default:[He]},$$scope:{ctx:$}}}),F=new ze({props:{anchor:"transformers.PhimoeForCausalLM.forward.example",$$slots:{default:[Ve]},$$scope:{ctx:$}}}),Q=new we({props:{name:"generate",anchor:"transformers.PhimoeForCausalLM.generate",parameters:[{name:"inputs",val:": typing.Optional[torch.Tensor] = None"},{name:"generation_config",val:": typing.Optional[transformers.generation.configuration_utils.GenerationConfig] = None"},{name:"logits_processor",val:": typing.Optional[transformers.generation.logits_process.LogitsProcessorList] = None"},{name:"stopping_criteria",val:": typing.Optional[transformers.generation.stopping_criteria.StoppingCriteriaList] = None"},{name:"prefix_allowed_tokens_fn",val:": typing.Optional[typing.Callable[[int, torch.Tensor], list[int]]] = None"},{name:"synced_gpus",val:": typing.Optional[bool] = None"},{name:"assistant_model",val:": typing.Optional[ForwardRef('PreTrainedModel')] = None"},{name:"streamer",val:": typing.Optional[ForwardRef('BaseStreamer')] = None"},{name:"negative_prompt_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"negative_prompt_attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"use_model_defaults",val:": typing.Optional[bool] = None"},{name:"custom_generate",val:": typing.Union[str, typing.Callable, NoneType] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.PhimoeForCausalLM.generate.inputs",description:`<strong>inputs</strong> (<code>torch.Tensor</code> of varying shape depending on the modality, <em>optional</em>) &#x2014;
The sequence used as a prompt for the generation or as model inputs to the encoder. If <code>None</code> the
method initializes it with <code>bos_token_id</code> and a batch size of 1. For decoder-only models <code>inputs</code>
should be in the format of <code>input_ids</code>. For encoder-decoder models <em>inputs</em> can represent any of
<code>input_ids</code>, <code>input_values</code>, <code>input_features</code>, or <code>pixel_values</code>.`,name:"inputs"},{anchor:"transformers.PhimoeForCausalLM.generate.generation_config",description:`<strong>generation_config</strong> (<a href="/docs/transformers/v4.56.2/en/main_classes/text_generation#transformers.GenerationConfig">GenerationConfig</a>, <em>optional</em>) &#x2014;
The generation configuration to be used as base parametrization for the generation call. <code>**kwargs</code>
passed to generate matching the attributes of <code>generation_config</code> will override them. If
<code>generation_config</code> is not provided, the default will be used, which has the following loading
priority: 1) from the <code>generation_config.json</code> model file, if it exists; 2) from the model
configuration. Please note that unspecified parameters will inherit <a href="/docs/transformers/v4.56.2/en/main_classes/text_generation#transformers.GenerationConfig">GenerationConfig</a>&#x2019;s
default values, whose documentation should be checked to parameterize generation.`,name:"generation_config"},{anchor:"transformers.PhimoeForCausalLM.generate.logits_processor",description:`<strong>logits_processor</strong> (<code>LogitsProcessorList</code>, <em>optional</em>) &#x2014;
Custom logits processors that complement the default logits processors built from arguments and
generation config. If a logit processor is passed that is already created with the arguments or a
generation config an error is thrown. This feature is intended for advanced users.`,name:"logits_processor"},{anchor:"transformers.PhimoeForCausalLM.generate.stopping_criteria",description:`<strong>stopping_criteria</strong> (<code>StoppingCriteriaList</code>, <em>optional</em>) &#x2014;
Custom stopping criteria that complements the default stopping criteria built from arguments and a
generation config. If a stopping criteria is passed that is already created with the arguments or a
generation config an error is thrown. If your stopping criteria depends on the <code>scores</code> input, make
sure you pass <code>return_dict_in_generate=True, output_scores=True</code> to <code>generate</code>. This feature is
intended for advanced users.`,name:"stopping_criteria"},{anchor:"transformers.PhimoeForCausalLM.generate.prefix_allowed_tokens_fn",description:`<strong>prefix_allowed_tokens_fn</strong> (<code>Callable[[int, torch.Tensor], list[int]]</code>, <em>optional</em>) &#x2014;
If provided, this function constraints the beam search to allowed tokens only at each step. If not
provided no constraint is applied. This function takes 2 arguments: the batch ID <code>batch_id</code> and
<code>input_ids</code>. It has to return a list with the allowed tokens for the next generation step conditioned
on the batch ID <code>batch_id</code> and the previously generated tokens <code>inputs_ids</code>. This argument is useful
for constrained generation conditioned on the prefix, as described in <a href="https://huggingface.co/papers/2010.00904" rel="nofollow">Autoregressive Entity
Retrieval</a>.`,name:"prefix_allowed_tokens_fn"},{anchor:"transformers.PhimoeForCausalLM.generate.synced_gpus",description:`<strong>synced_gpus</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether to continue running the while loop until max_length. Unless overridden, this flag will be set
to <code>True</code> if using <code>FullyShardedDataParallel</code> or DeepSpeed ZeRO Stage 3 with multiple GPUs to avoid
deadlocking if one GPU finishes generating before other GPUs. Otherwise, defaults to <code>False</code>.`,name:"synced_gpus"},{anchor:"transformers.PhimoeForCausalLM.generate.assistant_model",description:`<strong>assistant_model</strong> (<code>PreTrainedModel</code>, <em>optional</em>) &#x2014;
An assistant model that can be used to accelerate generation. The assistant model must have the exact
same tokenizer. The acceleration is achieved when forecasting candidate tokens with the assistant model
is much faster than running generation with the model you&#x2019;re calling generate from. As such, the
assistant model should be much smaller.`,name:"assistant_model"},{anchor:"transformers.PhimoeForCausalLM.generate.streamer",description:`<strong>streamer</strong> (<code>BaseStreamer</code>, <em>optional</em>) &#x2014;
Streamer object that will be used to stream the generated sequences. Generated tokens are passed
through <code>streamer.put(token_ids)</code> and the streamer is responsible for any further processing.`,name:"streamer"},{anchor:"transformers.PhimoeForCausalLM.generate.negative_prompt_ids",description:`<strong>negative_prompt_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
The negative prompt needed for some processors such as CFG. The batch size must match the input batch
size. This is an experimental feature, subject to breaking API changes in future versions.`,name:"negative_prompt_ids"},{anchor:"transformers.PhimoeForCausalLM.generate.negative_prompt_attention_mask",description:`<strong>negative_prompt_attention_mask</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Attention_mask for <code>negative_prompt_ids</code>.`,name:"negative_prompt_attention_mask"},{anchor:"transformers.PhimoeForCausalLM.generate.use_model_defaults",description:`<strong>use_model_defaults</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
When it is <code>True</code>, unset parameters in <code>generation_config</code> will be set to the model-specific default
generation configuration (<code>model.generation_config</code>), as opposed to the global defaults
(<code>GenerationConfig()</code>). If unset, models saved starting from <code>v4.50</code> will consider this flag to be
<code>True</code>.`,name:"use_model_defaults"},{anchor:"transformers.PhimoeForCausalLM.generate.custom_generate",description:`<strong>custom_generate</strong> (<code>str</code> or <code>Callable</code>, <em>optional</em>) &#x2014;
One of the following:<ul>
<li><code>str</code> (Hugging Face Hub repository name): runs the custom <code>generate</code> function defined at
<code>custom_generate/generate.py</code> in that repository instead of the standard <code>generate</code> method. The
repository fully replaces the generation logic, and the return type may differ.</li>
<li><code>str</code> (local repository path): same as above but from a local path, <code>trust_remote_code</code> not required.</li>
<li><code>Callable</code>: <code>generate</code> will perform the usual input preparation steps, then call the provided callable to
run the decoding loop.
For more information, see <a href="../../generation_strategies#custom-generation-methods">the docs</a>.</li>
</ul>`,name:"custom_generate"},{anchor:"transformers.PhimoeForCausalLM.generate.kwargs",description:`<strong>kwargs</strong> (<code>dict[str, Any]</code>, <em>optional</em>) &#x2014;
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
`}}),D=new Ie({props:{warning:!0,$$slots:{default:[Oe]},$$scope:{ctx:$}}}),le=new fe({props:{title:"PhimoeForSequenceClassification",local:"transformers.PhimoeForSequenceClassification",headingTag:"h2"}}),X=new we({props:{name:"class transformers.PhimoeForSequenceClassification",anchor:"transformers.PhimoeForSequenceClassification",parameters:[{name:"config",val:""}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/phimoe/modeling_phimoe.py#L1351"}}),Z=new we({props:{name:"forward",anchor:"transformers.PhimoeForSequenceClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.PhimoeForSequenceClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.PhimoeForSequenceClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.PhimoeForSequenceClassification.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.PhimoeForSequenceClassification.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.PhimoeForSequenceClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.PhimoeForSequenceClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.PhimoeForSequenceClassification.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
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
`}}),a=new Ie({props:{$$slots:{default:[Qe]},$$scope:{ctx:$}}}),{c(){_(o.$$.fragment),f=d(),t=m("div"),_(i.$$.fragment),w=d(),l=m("p"),l.textContent=k,N=d(),z=m("p"),z.innerHTML=U,ce=d(),L=m("p"),L.innerHTML=P,O=d(),p=m("div"),_(x.$$.fragment),ge=d(),A=m("p"),A.innerHTML=ke,ee=d(),_(oe.$$.fragment),pe=d(),_(W.$$.fragment),Ce=d(),q=m("div"),_(B.$$.fragment),Pe=d(),I=m("div"),_(S.$$.fragment),_e=d(),R=m("p"),R.innerHTML=Je,be=d(),_(G.$$.fragment),ye=d(),_(F.$$.fragment),Me=d(),j=m("div"),_(Q.$$.fragment),te=d(),ne=m("p"),ne.textContent=H,se=d(),_(D.$$.fragment),ae=d(),_(le.$$.fragment),ve=d(),J=m("div"),_(X.$$.fragment),Y=d(),E=m("div"),_(Z.$$.fragment),Te=d(),K=m("p"),K.innerHTML=$e,e=d(),_(a.$$.fragment),this.h()},l(n){b(o.$$.fragment,n),f=c(n),t=h(n,"DIV",{class:!0});var u=de(t);b(i.$$.fragment,u),w=c(u),l=h(u,"P",{"data-svelte-h":!0}),C(l)!=="svelte-34hkmq"&&(l.textContent=k),N=c(u),z=h(u,"P",{"data-svelte-h":!0}),C(z)!=="svelte-q52n56"&&(z.innerHTML=U),ce=c(u),L=h(u,"P",{"data-svelte-h":!0}),C(L)!=="svelte-hswkmf"&&(L.innerHTML=P),O=c(u),p=h(u,"DIV",{class:!0});var me=de(p);b(x.$$.fragment,me),ge=c(me),A=h(me,"P",{"data-svelte-h":!0}),C(A)!=="svelte-lj8j1"&&(A.innerHTML=ke),ee=c(me),b(oe.$$.fragment,me),me.forEach(s),u.forEach(s),pe=c(n),b(W.$$.fragment,n),Ce=c(n),q=h(n,"DIV",{class:!0});var he=de(q);b(B.$$.fragment,he),Pe=c(he),I=h(he,"DIV",{class:!0});var re=de(I);b(S.$$.fragment,re),_e=c(re),R=h(re,"P",{"data-svelte-h":!0}),C(R)!=="svelte-1lufol5"&&(R.innerHTML=Je),be=c(re),b(G.$$.fragment,re),ye=c(re),b(F.$$.fragment,re),re.forEach(s),Me=c(he),j=h(he,"DIV",{class:!0});var ue=de(j);b(Q.$$.fragment,ue),te=c(ue),ne=h(ue,"P",{"data-svelte-h":!0}),C(ne)!=="svelte-s5ko3x"&&(ne.textContent=H),se=c(ue),b(D.$$.fragment,ue),ue.forEach(s),he.forEach(s),ae=c(n),b(le.$$.fragment,n),ve=c(n),J=h(n,"DIV",{class:!0});var xe=de(J);b(X.$$.fragment,xe),Y=c(xe),E=h(xe,"DIV",{class:!0});var je=de(E);b(Z.$$.fragment,je),Te=c(je),K=h(je,"P",{"data-svelte-h":!0}),C(K)!=="svelte-1sal4ui"&&(K.innerHTML=$e),e=c(je),b(a.$$.fragment,je),je.forEach(s),xe.forEach(s),this.h()},h(){ie(p,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),ie(t,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),ie(I,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),ie(j,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),ie(q,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),ie(E,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),ie(J,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(n,u){y(o,n,u),r(n,f,u),r(n,t,u),y(i,t,null),g(t,w),g(t,l),g(t,N),g(t,z),g(t,ce),g(t,L),g(t,O),g(t,p),y(x,p,null),g(p,ge),g(p,A),g(p,ee),y(oe,p,null),r(n,pe,u),y(W,n,u),r(n,Ce,u),r(n,q,u),y(B,q,null),g(q,Pe),g(q,I),y(S,I,null),g(I,_e),g(I,R),g(I,be),y(G,I,null),g(I,ye),y(F,I,null),g(q,Me),g(q,j),y(Q,j,null),g(j,te),g(j,ne),g(j,se),y(D,j,null),r(n,ae,u),y(le,n,u),r(n,ve,u),r(n,J,u),y(X,J,null),g(J,Y),g(J,E),y(Z,E,null),g(E,Te),g(E,K),g(E,e),y(a,E,null),V=!0},p(n,u){const me={};u&2&&(me.$$scope={dirty:u,ctx:n}),oe.$set(me);const he={};u&2&&(he.$$scope={dirty:u,ctx:n}),G.$set(he);const re={};u&2&&(re.$$scope={dirty:u,ctx:n}),F.$set(re);const ue={};u&2&&(ue.$$scope={dirty:u,ctx:n}),D.$set(ue);const xe={};u&2&&(xe.$$scope={dirty:u,ctx:n}),a.$set(xe)},i(n){V||(M(o.$$.fragment,n),M(i.$$.fragment,n),M(x.$$.fragment,n),M(oe.$$.fragment,n),M(W.$$.fragment,n),M(B.$$.fragment,n),M(S.$$.fragment,n),M(G.$$.fragment,n),M(F.$$.fragment,n),M(Q.$$.fragment,n),M(D.$$.fragment,n),M(le.$$.fragment,n),M(X.$$.fragment,n),M(Z.$$.fragment,n),M(a.$$.fragment,n),V=!0)},o(n){v(o.$$.fragment,n),v(i.$$.fragment,n),v(x.$$.fragment,n),v(oe.$$.fragment,n),v(W.$$.fragment,n),v(B.$$.fragment,n),v(S.$$.fragment,n),v(G.$$.fragment,n),v(F.$$.fragment,n),v(Q.$$.fragment,n),v(D.$$.fragment,n),v(le.$$.fragment,n),v(X.$$.fragment,n),v(Z.$$.fragment,n),v(a.$$.fragment,n),V=!1},d(n){n&&(s(f),s(t),s(pe),s(Ce),s(q),s(ae),s(ve),s(J)),T(o,n),T(i),T(x),T(oe),T(W,n),T(B),T(S),T(G),T(F),T(Q),T(D),T(le,n),T(X),T(Z),T(a)}}}function Xe($){let o,f;return o=new Ee({props:{$$slots:{default:[De]},$$scope:{ctx:$}}}),{c(){_(o.$$.fragment)},l(t){b(o.$$.fragment,t)},m(t,i){y(o,t,i),f=!0},p(t,i){const w={};i&2&&(w.$$scope={dirty:i,ctx:t}),o.$set(w)},i(t){f||(M(o.$$.fragment,t),f=!0)},o(t){v(o.$$.fragment,t),f=!1},d(t){T(o,t)}}}function Ye($){let o,f,t,i,w,l="<em>This model was released on 2024-04-22 and added to Hugging Face Transformers on 2024-10-04.</em>",k,N,z,U,ce='<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/> <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat"/> <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/>',L,P,O,p,x='The PhiMoE model was proposed in <a href="https://huggingface.co/papers/2404.14219" rel="nofollow">Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone</a> by Microsoft.',ge,A,ke,ee,oe="The abstract from the Phi-3 paper is the following:",pe,W,Ce="We introduce phi-3-mini, a 3.8 billion parameter language model trained on 3.3 trillion tokens, whose overall performance, as measured by both academic benchmarks and internal testing, rivals that of models such as Mixtral 8x7B and GPT-3.5 (e.g., phi-3-mini achieves 69% on MMLU and 8.38 on MT-bench), despite being small enough to be deployed on a phone. Our training dataset is a scaled-up version of the one used for phi-2, composed of heavily filtered publicly available web data and synthetic data. The model is also further aligned for robustness, safety, and chat format. We also provide parameter-scaling results with a 7B, 14B models trained for 4.8T tokens, called phi-3-small, phi-3-medium, both significantly more capable than phi-3-mini (e.g., respectively 75%, 78% on MMLU, and 8.7, 8.9 on MT-bench). To enhance multilingual, multimodal, and long-context capabilities, we introduce three models in the phi-3.5 series: phi-3.5-mini, phi-3.5-MoE, and phi-3.5-Vision. The phi-3.5-MoE, a 16 x 3.8B MoE model with 6.6 billion active parameters, achieves superior performance in language reasoning, math, and code tasks compared to other open-source models of similar scale, such as Llama 3.1 and the Mixtral series, and on par with Gemini-1.5-Flash and GPT-4o-mini. Meanwhile, phi-3.5-Vision, a 4.2 billion parameter model derived from phi-3.5-mini, excels in reasoning tasks and is adept at handling both single-image and text prompts, as well as multi-image and text prompts.",q,B,Pe='The original code for PhiMoE can be found <a href="https://huggingface.co/microsoft/Phi-3.5-MoE-instruct" rel="nofollow">here</a>.',I,S,_e,R,Je='<li>This model is very similar to <code>Mixtral</code> with the main difference of <code>Phi3LongRoPEScaledRotaryEmbedding</code>, where they are used to extend the context of the rotary embeddings. The query, key and values are fused, and the MLP’s up and gate projection layers are also fused.</li> <li>The tokenizer used for this model is identical to the <a href="/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaTokenizer">LlamaTokenizer</a>, with the exception of additional tokens.</li>',be,G,ye,F,Me,j,Q,te,ne,H,se,D,ae,le=`This is the configuration class to store the configuration of a <a href="/docs/transformers/v4.56.2/en/model_doc/phimoe#transformers.PhimoeModel">PhimoeModel</a>. It is used to instantiate a Phi-moe
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the
<a href="https://huggingface.co/microsoft/Phi-3.5-MoE-instruct" rel="nofollow">microsoft/Phi-3.5-MoE-instruct</a>.
Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> and can be used to control the model outputs. Read the
documentation from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> for more information.`,ve,J,X,Y,E,Z,Te,K,$e;return N=new fe({props:{title:"PhiMoE",local:"phimoe",headingTag:"h1"}}),P=new fe({props:{title:"Overview",local:"overview",headingTag:"h2"}}),A=new fe({props:{title:"Summary",local:"summary",headingTag:"h3"}}),S=new fe({props:{title:"Usage tips",local:"usage-tips",headingTag:"h2"}}),G=new fe({props:{title:"How to use PhiMoE",local:"how-to-use-phimoe",headingTag:"h2"}}),F=new Ie({props:{warning:!0,$$slots:{default:[Ne]},$$scope:{ctx:$}}}),j=new Fe({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b01vZGVsRm9yQ2F1c2FsTE0lMkMlMjBBdXRvVG9rZW5pemVyJTJDJTIwcGlwZWxpbmUlMjAlMEElMEF0b3JjaC5yYW5kb20ubWFudWFsX3NlZWQoMCklMjAlMEElMEFtb2RlbCUyMCUzRCUyMEF1dG9Nb2RlbEZvckNhdXNhbExNLmZyb21fcHJldHJhaW5lZCglMjAlMEElMjAlMjAlMjAlMjAlMjJtaWNyb3NvZnQlMkZQaGktMy41LU1vRS1pbnN0cnVjdCUyMiUyQyUyMCUyMCUwQSUyMCUyMCUyMCUyMGRldmljZV9tYXAlM0QlMjJhdXRvJTIyJTJDJTIwJTIwJTBBJTIwJTIwJTIwJTIwZHR5cGUlM0QlMjJhdXRvJTIyJTJDJTBBKSUyMCUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMm1pY3Jvc29mdCUyRlBoaS0zLjUtTW9FLWluc3RydWN0JTIyKSUyMCUwQSUwQW1lc3NhZ2VzJTIwJTNEJTIwJTVCJTIwJTBBJTIwJTIwJTIwJTIwJTdCJTIycm9sZSUyMiUzQSUyMCUyMnN5c3RlbSUyMiUyQyUyMCUyMmNvbnRlbnQlMjIlM0ElMjAlMjJZb3UlMjBhcmUlMjBhJTIwaGVscGZ1bCUyMEFJJTIwYXNzaXN0YW50LiUyMiU3RCUyQyUyMCUwQSUyMCUyMCUyMCUyMCU3QiUyMnJvbGUlMjIlM0ElMjAlMjJ1c2VyJTIyJTJDJTIwJTIyY29udGVudCUyMiUzQSUyMCUyMkNhbiUyMHlvdSUyMHByb3ZpZGUlMjB3YXlzJTIwdG8lMjBlYXQlMjBjb21iaW5hdGlvbnMlMjBvZiUyMGJhbmFuYXMlMjBhbmQlMjBkcmFnb25mcnVpdHMlM0YlMjIlN0QlMkMlMjAlMEElMjAlMjAlMjAlMjAlN0IlMjJyb2xlJTIyJTNBJTIwJTIyYXNzaXN0YW50JTIyJTJDJTIwJTIyY29udGVudCUyMiUzQSUyMCUyMlN1cmUhJTIwSGVyZSUyMGFyZSUyMHNvbWUlMjB3YXlzJTIwdG8lMjBlYXQlMjBiYW5hbmFzJTIwYW5kJTIwZHJhZ29uZnJ1aXRzJTIwdG9nZXRoZXIlM0ElMjAxLiUyMEJhbmFuYSUyMGFuZCUyMGRyYWdvbmZydWl0JTIwc21vb3RoaWUlM0ElMjBCbGVuZCUyMGJhbmFuYXMlMjBhbmQlMjBkcmFnb25mcnVpdHMlMjB0b2dldGhlciUyMHdpdGglMjBzb21lJTIwbWlsayUyMGFuZCUyMGhvbmV5LiUyMDIuJTIwQmFuYW5hJTIwYW5kJTIwZHJhZ29uZnJ1aXQlMjBzYWxhZCUzQSUyME1peCUyMHNsaWNlZCUyMGJhbmFuYXMlMjBhbmQlMjBkcmFnb25mcnVpdHMlMjB0b2dldGhlciUyMHdpdGglMjBzb21lJTIwbGVtb24lMjBqdWljZSUyMGFuZCUyMGhvbmV5LiUyMiU3RCUyQyUyMCUwQSUyMCUyMCUyMCUyMCU3QiUyMnJvbGUlMjIlM0ElMjAlMjJ1c2VyJTIyJTJDJTIwJTIyY29udGVudCUyMiUzQSUyMCUyMldoYXQlMjBhYm91dCUyMHNvbHZpbmclMjBhbiUyMDJ4JTIwJTJCJTIwMyUyMCUzRCUyMDclMjBlcXVhdGlvbiUzRiUyMiU3RCUyQyUyMCUwQSU1RCUyMCUwQSUwQXBpcGUlMjAlM0QlMjBwaXBlbGluZSglMjAlMEElMjAlMjAlMjAlMjAlMjJ0ZXh0LWdlbmVyYXRpb24lMjIlMkMlMjAlMEElMjAlMjAlMjAlMjBtb2RlbCUzRG1vZGVsJTJDJTIwJTBBJTIwJTIwJTIwJTIwdG9rZW5pemVyJTNEdG9rZW5pemVyJTJDJTIwJTBBKSUyMCUwQSUwQWdlbmVyYXRpb25fYXJncyUyMCUzRCUyMCU3QiUyMCUwQSUyMCUyMCUyMCUyMCUyMm1heF9uZXdfdG9rZW5zJTIyJTNBJTIwNTAwJTJDJTIwJTBBJTIwJTIwJTIwJTIwJTIycmV0dXJuX2Z1bGxfdGV4dCUyMiUzQSUyMEZhbHNlJTJDJTIwJTBBJTIwJTIwJTIwJTIwJTIydGVtcGVyYXR1cmUlMjIlM0ElMjAwLjAlMkMlMjAlMEElMjAlMjAlMjAlMjAlMjJkb19zYW1wbGUlMjIlM0ElMjBGYWxzZSUyQyUyMCUwQSU3RCUyMCUwQSUwQW91dHB1dCUyMCUzRCUyMHBpcGUobWVzc2FnZXMlMkMlMjAqKmdlbmVyYXRpb25fYXJncyklMjAlMEFwcmludChvdXRwdXQlNUIwJTVEJTVCJ2dlbmVyYXRlZF90ZXh0JyU1RCk=",highlighted:`<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoModelForCausalLM, AutoTokenizer, pipeline 

torch.random.manual_seed(<span class="hljs-number">0</span>) 

model = AutoModelForCausalLM.from_pretrained( 
    <span class="hljs-string">&quot;microsoft/Phi-3.5-MoE-instruct&quot;</span>,  
    device_map=<span class="hljs-string">&quot;auto&quot;</span>,  
    dtype=<span class="hljs-string">&quot;auto&quot;</span>,
) 

tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;microsoft/Phi-3.5-MoE-instruct&quot;</span>) 

messages = [ 
    {<span class="hljs-string">&quot;role&quot;</span>: <span class="hljs-string">&quot;system&quot;</span>, <span class="hljs-string">&quot;content&quot;</span>: <span class="hljs-string">&quot;You are a helpful AI assistant.&quot;</span>}, 
    {<span class="hljs-string">&quot;role&quot;</span>: <span class="hljs-string">&quot;user&quot;</span>, <span class="hljs-string">&quot;content&quot;</span>: <span class="hljs-string">&quot;Can you provide ways to eat combinations of bananas and dragonfruits?&quot;</span>}, 
    {<span class="hljs-string">&quot;role&quot;</span>: <span class="hljs-string">&quot;assistant&quot;</span>, <span class="hljs-string">&quot;content&quot;</span>: <span class="hljs-string">&quot;Sure! Here are some ways to eat bananas and dragonfruits together: 1. Banana and dragonfruit smoothie: Blend bananas and dragonfruits together with some milk and honey. 2. Banana and dragonfruit salad: Mix sliced bananas and dragonfruits together with some lemon juice and honey.&quot;</span>}, 
    {<span class="hljs-string">&quot;role&quot;</span>: <span class="hljs-string">&quot;user&quot;</span>, <span class="hljs-string">&quot;content&quot;</span>: <span class="hljs-string">&quot;What about solving an 2x + 3 = 7 equation?&quot;</span>}, 
] 

pipe = pipeline( 
    <span class="hljs-string">&quot;text-generation&quot;</span>, 
    model=model, 
    tokenizer=tokenizer, 
) 

generation_args = { 
    <span class="hljs-string">&quot;max_new_tokens&quot;</span>: <span class="hljs-number">500</span>, 
    <span class="hljs-string">&quot;return_full_text&quot;</span>: <span class="hljs-literal">False</span>, 
    <span class="hljs-string">&quot;temperature&quot;</span>: <span class="hljs-number">0.0</span>, 
    <span class="hljs-string">&quot;do_sample&quot;</span>: <span class="hljs-literal">False</span>, 
} 

output = pipe(messages, **generation_args) 
<span class="hljs-built_in">print</span>(output[<span class="hljs-number">0</span>][<span class="hljs-string">&#x27;generated_text&#x27;</span>])`,wrap:!1}}),te=new fe({props:{title:"PhimoeConfig",local:"transformers.PhimoeConfig",headingTag:"h2"}}),se=new we({props:{name:"class transformers.PhimoeConfig",anchor:"transformers.PhimoeConfig",parameters:[{name:"vocab_size",val:" = 32064"},{name:"hidden_size",val:" = 4096"},{name:"intermediate_size",val:" = 6400"},{name:"num_hidden_layers",val:" = 32"},{name:"num_attention_heads",val:" = 32"},{name:"num_key_value_heads",val:" = 8"},{name:"hidden_act",val:" = 'silu'"},{name:"max_position_embeddings",val:" = 131072"},{name:"initializer_range",val:" = 0.02"},{name:"rms_norm_eps",val:" = 1e-05"},{name:"use_cache",val:" = True"},{name:"pad_token_id",val:" = None"},{name:"bos_token_id",val:" = 1"},{name:"eos_token_id",val:" = 2"},{name:"tie_word_embeddings",val:" = False"},{name:"rope_theta",val:" = 1000000.0"},{name:"rope_scaling",val:" = None"},{name:"sliding_window",val:" = None"},{name:"attention_dropout",val:" = 0.0"},{name:"num_experts_per_tok",val:" = 2"},{name:"num_local_experts",val:" = 16"},{name:"output_router_logits",val:" = False"},{name:"router_aux_loss_coef",val:" = 0.001"},{name:"router_jitter_noise",val:" = 0.01"},{name:"input_jitter_noise",val:" = 0.0"},{name:"attention_bias",val:" = False"},{name:"lm_head_bias",val:" = False"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.PhimoeConfig.vocab_size",description:`<strong>vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 32064) &#x2014;
Vocabulary size of the Phimoe model. Defines the number of different tokens that can be represented by the
<code>inputs_ids</code> passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/phimoe#transformers.PhimoeModel">PhimoeModel</a>`,name:"vocab_size"},{anchor:"transformers.PhimoeConfig.hidden_size",description:`<strong>hidden_size</strong> (<code>int</code>, <em>optional</em>, defaults to 4096) &#x2014;
Dimension of the hidden representations.`,name:"hidden_size"},{anchor:"transformers.PhimoeConfig.intermediate_size",description:`<strong>intermediate_size</strong> (<code>int</code>, <em>optional</em>, defaults to 6400) &#x2014;
Dimension of the MLP representations.`,name:"intermediate_size"},{anchor:"transformers.PhimoeConfig.num_hidden_layers",description:`<strong>num_hidden_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 32) &#x2014;
Number of hidden layers in the Transformer encoder.`,name:"num_hidden_layers"},{anchor:"transformers.PhimoeConfig.num_attention_heads",description:`<strong>num_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 32) &#x2014;
Number of attention heads for each attention layer in the Transformer encoder.`,name:"num_attention_heads"},{anchor:"transformers.PhimoeConfig.num_key_value_heads",description:`<strong>num_key_value_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 8) &#x2014;
This is the number of key_value heads that should be used to implement Grouped Query Attention. If
<code>num_key_value_heads=num_attention_heads</code>, the model will use Multi Head Attention (MHA), if
<code>num_key_value_heads=1</code> the model will use Multi Query Attention (MQA) otherwise GQA is used. When
converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
by meanpooling all the original heads within that group. For more details, check out <a href="https://huggingface.co/papers/2305.13245" rel="nofollow">this
paper</a>. If it is not specified, will default to <code>8</code>.`,name:"num_key_value_heads"},{anchor:"transformers.PhimoeConfig.hidden_act",description:`<strong>hidden_act</strong> (<code>str</code> or <code>function</code>, <em>optional</em>, defaults to <code>&quot;silu&quot;</code>) &#x2014;
The non-linear activation function (function or string) in the decoder.`,name:"hidden_act"},{anchor:"transformers.PhimoeConfig.max_position_embeddings",description:`<strong>max_position_embeddings</strong> (<code>int</code>, <em>optional</em>, defaults to <code>4096*32</code>) &#x2014;
The maximum sequence length that this model might ever be used with. Mixtral&#x2019;s sliding window attention
allows sequence of up to 4096*32 tokens.`,name:"max_position_embeddings"},{anchor:"transformers.PhimoeConfig.initializer_range",description:`<strong>initializer_range</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.`,name:"initializer_range"},{anchor:"transformers.PhimoeConfig.rms_norm_eps",description:`<strong>rms_norm_eps</strong> (<code>float</code>, <em>optional</em>, defaults to 1e-05) &#x2014;
The epsilon used by the rms normalization layers.`,name:"rms_norm_eps"},{anchor:"transformers.PhimoeConfig.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not the model should return the last key/values attentions (not used by all models). Only
relevant if <code>config.is_decoder=True</code>.`,name:"use_cache"},{anchor:"transformers.PhimoeConfig.pad_token_id",description:`<strong>pad_token_id</strong> (<code>int</code>, <em>optional</em>) &#x2014;
The id of the padding token.`,name:"pad_token_id"},{anchor:"transformers.PhimoeConfig.bos_token_id",description:`<strong>bos_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 1) &#x2014;
The id of the &#x201C;beginning-of-sequence&#x201D; token.`,name:"bos_token_id"},{anchor:"transformers.PhimoeConfig.eos_token_id",description:`<strong>eos_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 2) &#x2014;
The id of the &#x201C;end-of-sequence&#x201D; token.`,name:"eos_token_id"},{anchor:"transformers.PhimoeConfig.tie_word_embeddings",description:`<strong>tie_word_embeddings</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether the model&#x2019;s input and output word embeddings should be tied.`,name:"tie_word_embeddings"},{anchor:"transformers.PhimoeConfig.rope_theta",description:`<strong>rope_theta</strong> (<code>float</code>, <em>optional</em>, defaults to 1000000.0) &#x2014;
The base period of the RoPE embeddings.`,name:"rope_theta"},{anchor:"transformers.PhimoeConfig.rope_scaling",description:`<strong>rope_scaling</strong> (<code>dict</code>, <em>optional</em>) &#x2014;
The scaling strategy for the RoPE embeddings. If <code>None</code>, no scaling is applied. If a dictionary, it must
contain the following keys: <code>type</code>, <code>short_factor</code>, <code>long_factor</code>, <code>short_mscale</code>, <code>long_mscale</code> and
<code>original_max_position_embeddings</code>. The <code>type</code> must be <code>longrope</code>, the <code>short_mscale</code> and <code>long_scale</code> must
be numbers, the <code>short_factor</code> and <code>long_factor</code> must be lists of numbers with the same length as half of
the attention head size and the <code>original_max_position_embeddings</code> must be an integer.`,name:"rope_scaling"},{anchor:"transformers.PhimoeConfig.sliding_window",description:`<strong>sliding_window</strong> (<code>int</code>, <em>optional</em>) &#x2014;
Sliding window attention window size. If not specified, will default to <code>262144</code>.`,name:"sliding_window"},{anchor:"transformers.PhimoeConfig.attention_dropout",description:`<strong>attention_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The dropout ratio for the attention probabilities.`,name:"attention_dropout"},{anchor:"transformers.PhimoeConfig.num_experts_per_tok",description:`<strong>num_experts_per_tok</strong> (<code>int</code>, <em>optional</em>, defaults to 2) &#x2014;
The number of experts to root per-token, can be also interpreted as the <code>top-p</code> routing
parameter`,name:"num_experts_per_tok"},{anchor:"transformers.PhimoeConfig.num_local_experts",description:`<strong>num_local_experts</strong> (<code>int</code>, <em>optional</em>, defaults to 16) &#x2014;
Number of experts per Sparse MLP layer.`,name:"num_local_experts"},{anchor:"transformers.PhimoeConfig.output_router_logits",description:`<strong>output_router_logits</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not the router logits should be returned by the model. Enabling this will also
allow the model to output the auxiliary loss. See <a href>here</a> for more details`,name:"output_router_logits"},{anchor:"transformers.PhimoeConfig.router_aux_loss_coef",description:`<strong>router_aux_loss_coef</strong> (<code>float</code>, <em>optional</em>, defaults to 0.001) &#x2014;
The aux loss factor for the total loss.`,name:"router_aux_loss_coef"},{anchor:"transformers.PhimoeConfig.router_jitter_noise",description:`<strong>router_jitter_noise</strong> (<code>float</code>, <em>optional</em>, defaults to 0.01) &#x2014;
Amount of noise to add to the router.`,name:"router_jitter_noise"},{anchor:"transformers.PhimoeConfig.input_jitter_noise",description:"<strong>input_jitter_noise</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014; Input jitter noise",name:"input_jitter_noise"},{anchor:"transformers.PhimoeConfig.attention_bias",description:"<strong>attention_bias</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014; Attention bias",name:"attention_bias"},{anchor:"transformers.PhimoeConfig.lm_head_bias",description:"<strong>lm_head_bias</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014; LM head bias",name:"lm_head_bias"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/phimoe/configuration_phimoe.py#L26"}}),J=new ze({props:{anchor:"transformers.PhimoeConfig.example",$$slots:{default:[Ae]},$$scope:{ctx:$}}}),Y=new Se({props:{pytorch:!0,tensorflow:!1,jax:!1,$$slots:{pytorch:[Xe]},$$scope:{ctx:$}}}),Z=new Ze({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/phimoe.md"}}),{c(){o=m("meta"),f=d(),t=m("p"),i=d(),w=m("p"),w.innerHTML=l,k=d(),_(N.$$.fragment),z=d(),U=m("div"),U.innerHTML=ce,L=d(),_(P.$$.fragment),O=d(),p=m("p"),p.innerHTML=x,ge=d(),_(A.$$.fragment),ke=d(),ee=m("p"),ee.textContent=oe,pe=d(),W=m("p"),W.textContent=Ce,q=d(),B=m("p"),B.innerHTML=Pe,I=d(),_(S.$$.fragment),_e=d(),R=m("ul"),R.innerHTML=Je,be=d(),_(G.$$.fragment),ye=d(),_(F.$$.fragment),Me=d(),_(j.$$.fragment),Q=d(),_(te.$$.fragment),ne=d(),H=m("div"),_(se.$$.fragment),D=d(),ae=m("p"),ae.innerHTML=le,ve=d(),_(J.$$.fragment),X=d(),_(Y.$$.fragment),E=d(),_(Z.$$.fragment),Te=d(),K=m("p"),this.h()},l(e){const a=Be("svelte-u9bgzb",document.head);o=h(a,"META",{name:!0,content:!0}),a.forEach(s),f=c(e),t=h(e,"P",{}),de(t).forEach(s),i=c(e),w=h(e,"P",{"data-svelte-h":!0}),C(w)!=="svelte-1xnu703"&&(w.innerHTML=l),k=c(e),b(N.$$.fragment,e),z=c(e),U=h(e,"DIV",{class:!0,"data-svelte-h":!0}),C(U)!=="svelte-b95w5j"&&(U.innerHTML=ce),L=c(e),b(P.$$.fragment,e),O=c(e),p=h(e,"P",{"data-svelte-h":!0}),C(p)!=="svelte-fwxnb0"&&(p.innerHTML=x),ge=c(e),b(A.$$.fragment,e),ke=c(e),ee=h(e,"P",{"data-svelte-h":!0}),C(ee)!=="svelte-1iah8ch"&&(ee.textContent=oe),pe=c(e),W=h(e,"P",{"data-svelte-h":!0}),C(W)!=="svelte-dp2ft6"&&(W.textContent=Ce),q=c(e),B=h(e,"P",{"data-svelte-h":!0}),C(B)!=="svelte-weehy4"&&(B.innerHTML=Pe),I=c(e),b(S.$$.fragment,e),_e=c(e),R=h(e,"UL",{"data-svelte-h":!0}),C(R)!=="svelte-j5d3z"&&(R.innerHTML=Je),be=c(e),b(G.$$.fragment,e),ye=c(e),b(F.$$.fragment,e),Me=c(e),b(j.$$.fragment,e),Q=c(e),b(te.$$.fragment,e),ne=c(e),H=h(e,"DIV",{class:!0});var V=de(H);b(se.$$.fragment,V),D=c(V),ae=h(V,"P",{"data-svelte-h":!0}),C(ae)!=="svelte-181n8ic"&&(ae.innerHTML=le),ve=c(V),b(J.$$.fragment,V),V.forEach(s),X=c(e),b(Y.$$.fragment,e),E=c(e),b(Z.$$.fragment,e),Te=c(e),K=h(e,"P",{}),de(K).forEach(s),this.h()},h(){ie(o,"name","hf:doc:metadata"),ie(o,"content",Ke),ie(U,"class","flex flex-wrap space-x-1"),ie(H,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,a){g(document.head,o),r(e,f,a),r(e,t,a),r(e,i,a),r(e,w,a),r(e,k,a),y(N,e,a),r(e,z,a),r(e,U,a),r(e,L,a),y(P,e,a),r(e,O,a),r(e,p,a),r(e,ge,a),y(A,e,a),r(e,ke,a),r(e,ee,a),r(e,pe,a),r(e,W,a),r(e,q,a),r(e,B,a),r(e,I,a),y(S,e,a),r(e,_e,a),r(e,R,a),r(e,be,a),y(G,e,a),r(e,ye,a),y(F,e,a),r(e,Me,a),y(j,e,a),r(e,Q,a),y(te,e,a),r(e,ne,a),r(e,H,a),y(se,H,null),g(H,D),g(H,ae),g(H,ve),y(J,H,null),r(e,X,a),y(Y,e,a),r(e,E,a),y(Z,e,a),r(e,Te,a),r(e,K,a),$e=!0},p(e,[a]){const V={};a&2&&(V.$$scope={dirty:a,ctx:e}),F.$set(V);const n={};a&2&&(n.$$scope={dirty:a,ctx:e}),J.$set(n);const u={};a&2&&(u.$$scope={dirty:a,ctx:e}),Y.$set(u)},i(e){$e||(M(N.$$.fragment,e),M(P.$$.fragment,e),M(A.$$.fragment,e),M(S.$$.fragment,e),M(G.$$.fragment,e),M(F.$$.fragment,e),M(j.$$.fragment,e),M(te.$$.fragment,e),M(se.$$.fragment,e),M(J.$$.fragment,e),M(Y.$$.fragment,e),M(Z.$$.fragment,e),$e=!0)},o(e){v(N.$$.fragment,e),v(P.$$.fragment,e),v(A.$$.fragment,e),v(S.$$.fragment,e),v(G.$$.fragment,e),v(F.$$.fragment,e),v(j.$$.fragment,e),v(te.$$.fragment,e),v(se.$$.fragment,e),v(J.$$.fragment,e),v(Y.$$.fragment,e),v(Z.$$.fragment,e),$e=!1},d(e){e&&(s(f),s(t),s(i),s(w),s(k),s(z),s(U),s(L),s(O),s(p),s(ge),s(ke),s(ee),s(pe),s(W),s(q),s(B),s(I),s(_e),s(R),s(be),s(ye),s(Me),s(Q),s(ne),s(H),s(X),s(E),s(Te),s(K)),s(o),T(N,e),T(P,e),T(A,e),T(S,e),T(G,e),T(F,e),T(j,e),T(te,e),T(se),T(J),T(Y,e),T(Z,e)}}}const Ke='{"title":"PhiMoE","local":"phimoe","sections":[{"title":"Overview","local":"overview","sections":[{"title":"Summary","local":"summary","sections":[],"depth":3}],"depth":2},{"title":"Usage tips","local":"usage-tips","sections":[],"depth":2},{"title":"How to use PhiMoE","local":"how-to-use-phimoe","sections":[],"depth":2},{"title":"PhimoeConfig","local":"transformers.PhimoeConfig","sections":[],"depth":2},{"title":"PhimoeModel","local":"transformers.PhimoeModel","sections":[],"depth":2},{"title":"PhimoeForCausalLM","local":"transformers.PhimoeForCausalLM","sections":[],"depth":2},{"title":"PhimoeForSequenceClassification","local":"transformers.PhimoeForSequenceClassification","sections":[],"depth":2}],"depth":1}';function eo($){return qe(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class co extends Ge{constructor(o){super(),We(this,o,eo,Ye,Le,{})}}export{co as component};
