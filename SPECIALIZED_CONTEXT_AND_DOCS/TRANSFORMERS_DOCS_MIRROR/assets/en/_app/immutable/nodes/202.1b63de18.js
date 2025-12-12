import{s as Bt,o as Wt,n as Fe}from"../chunks/scheduler.18a86fab.js";import{S as Ot,i as Dt,g as l,s as a,r as m,A as Nt,h as c,f as o,c as r,j as F,x as v,u as p,k as L,y as d,a as i,v as h,d as u,t as f,w as g}from"../chunks/index.98837b22.js";import{T as Ye}from"../chunks/Tip.77304350.js";import{D as W}from"../chunks/Docstring.a1ef7999.js";import{C as Ht}from"../chunks/CodeBlock.8d0c2e8a.js";import{E as jt}from"../chunks/ExampleCodeBlock.8c3ee1f9.js";import{H as ge,E as Et}from"../chunks/getInferenceSnippets.06c2775f.js";function St(w){let n,_;return n=new Ht({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEdsbTRNb2RlbCUyQyUyMEdsbTRDb25maWclMEElMjMlMjBJbml0aWFsaXppbmclMjBhJTIwR2xtNCUyMGdsbTQtNC05Yi1jaGF0JTIwc3R5bGUlMjBjb25maWd1cmF0aW9uJTBBY29uZmlndXJhdGlvbiUyMCUzRCUyMEdsbTRDb25maWcoKSUwQSUyMyUyMEluaXRpYWxpemluZyUyMGElMjBtb2RlbCUyMGZyb20lMjB0aGUlMjBnbG00LTQtOWItY2hhdCUyMHN0eWxlJTIwY29uZmlndXJhdGlvbiUwQW1vZGVsJTIwJTNEJTIwR2xtNE1vZGVsKGNvbmZpZ3VyYXRpb24pJTBBJTIzJTIwQWNjZXNzaW5nJTIwdGhlJTIwbW9kZWwlMjBjb25maWd1cmF0aW9uJTBBY29uZmlndXJhdGlvbiUyMCUzRCUyMG1vZGVsLmNvbmZpZw==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> Glm4Model, Glm4Config
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a Glm4 glm4-4-9b-chat style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = Glm4Config()
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a model from the glm4-4-9b-chat style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = Glm4Model(configuration)
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Accessing the model configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = model.config`,wrap:!1}}),{c(){m(n.$$.fragment)},l(s){p(n.$$.fragment,s)},m(s,b){h(n,s,b),_=!0},p:Fe,i(s){_||(u(n.$$.fragment,s),_=!0)},o(s){f(n.$$.fragment,s),_=!1},d(s){g(n,s)}}}function Rt(w){let n,_=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){n=l("p"),n.innerHTML=_},l(s){n=c(s,"P",{"data-svelte-h":!0}),v(n)!=="svelte-fincs2"&&(n.innerHTML=_)},m(s,b){i(s,n,b)},p:Fe,d(s){s&&o(n)}}}function Zt(w){let n,_=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){n=l("p"),n.innerHTML=_},l(s){n=c(s,"P",{"data-svelte-h":!0}),v(n)!=="svelte-fincs2"&&(n.innerHTML=_)},m(s,b){i(s,n,b)},p:Fe,d(s){s&&o(n)}}}function At(w){let n,_="Example:",s,b,C;return b=new Ht({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBHbG00Rm9yQ2F1c2FsTE0lMEElMEFtb2RlbCUyMCUzRCUyMEdsbTRGb3JDYXVzYWxMTS5mcm9tX3ByZXRyYWluZWQoJTIyVEhVRE0lMkZHTE0tNC05Qi0wNDE0JTIyKSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMlRIVURNJTJGR0xNLTQtOUItMDQxNCUyMiklMEElMEFwcm9tcHQlMjAlM0QlMjAlMjJIZXklMkMlMjBhcmUlMjB5b3UlMjBjb25zY2lvdXMlM0YlMjBDYW4lMjB5b3UlMjB0YWxrJTIwdG8lMjBtZSUzRiUyMiUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplcihwcm9tcHQlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQSUwQSUyMyUyMEdlbmVyYXRlJTBBZ2VuZXJhdGVfaWRzJTIwJTNEJTIwbW9kZWwuZ2VuZXJhdGUoaW5wdXRzLmlucHV0X2lkcyUyQyUyMG1heF9sZW5ndGglM0QzMCklMEF0b2tlbml6ZXIuYmF0Y2hfZGVjb2RlKGdlbmVyYXRlX2lkcyUyQyUyMHNraXBfc3BlY2lhbF90b2tlbnMlM0RUcnVlJTJDJTIwY2xlYW5fdXBfdG9rZW5pemF0aW9uX3NwYWNlcyUzREZhbHNlKSU1QjAlNUQ=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, Glm4ForCausalLM

<span class="hljs-meta">&gt;&gt;&gt; </span>model = Glm4ForCausalLM.from_pretrained(<span class="hljs-string">&quot;THUDM/GLM-4-9B-0414&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;THUDM/GLM-4-9B-0414&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>prompt = <span class="hljs-string">&quot;Hey, are you conscious? Can you talk to me?&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(prompt, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Generate</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>generate_ids = model.generate(inputs.input_ids, max_length=<span class="hljs-number">30</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer.batch_decode(generate_ids, skip_special_tokens=<span class="hljs-literal">True</span>, clean_up_tokenization_spaces=<span class="hljs-literal">False</span>)[<span class="hljs-number">0</span>]
<span class="hljs-string">&quot;Hey, are you conscious? Can you talk to me?\\nI&#x27;m not conscious, but I can talk to you.&quot;</span>`,wrap:!1}}),{c(){n=l("p"),n.textContent=_,s=a(),m(b.$$.fragment)},l(y){n=c(y,"P",{"data-svelte-h":!0}),v(n)!=="svelte-11lpom8"&&(n.textContent=_),s=r(y),p(b.$$.fragment,y)},m(y,O){i(y,n,O),i(y,s,O),h(b,y,O),C=!0},p:Fe,i(y){C||(u(b.$$.fragment,y),C=!0)},o(y){f(b.$$.fragment,y),C=!1},d(y){y&&(o(n),o(s)),g(b,y)}}}function Jt(w){let n,_=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){n=l("p"),n.innerHTML=_},l(s){n=c(s,"P",{"data-svelte-h":!0}),v(n)!=="svelte-fincs2"&&(n.innerHTML=_)},m(s,b){i(s,n,b)},p:Fe,d(s){s&&o(n)}}}function Vt(w){let n,_=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){n=l("p"),n.innerHTML=_},l(s){n=c(s,"P",{"data-svelte-h":!0}),v(n)!=="svelte-fincs2"&&(n.innerHTML=_)},m(s,b){i(s,n,b)},p:Fe,d(s){s&&o(n)}}}function Qt(w){let n,_,s,b,C,y="<em>This model was released on 2024-06-18 and added to Hugging Face Transformers on 2025-04-09.</em>",O,A,qe,J,Ie,V,yt='The GLM family welcomes new members <a href="https://huggingface.co/papers/2406.12793" rel="nofollow">GLM-4-0414</a> series models.',Pe,Q,Tt=`The <strong>GLM-4-32B-0414</strong> series models, featuring 32 billion parameters. Its performance is comparable to OpenAI’s GPT
series and DeepSeek’s V3/R1 series. It also supports very user-friendly local deployment features. GLM-4-32B-Base-0414
was pre-trained on 15T of high-quality data, including substantial reasoning-type synthetic data. This lays the
foundation for subsequent reinforcement learning extensions. In the post-training stage, we employed human preference
alignment for dialogue scenarios. Additionally, using techniques like rejection sampling and reinforcement learning, we
enhanced the model’s performance in instruction following, engineering code, and function calling, thus strengthening
the atomic capabilities required for agent tasks. GLM-4-32B-0414 achieves good results in engineering code, Artifact
generation, function calling, search-based Q&amp;A, and report generation. In particular, on several benchmarks, such as
code generation or specific Q&amp;A tasks, GLM-4-32B-Base-0414 achieves comparable performance with those larger models like
GPT-4o and DeepSeek-V3-0324 (671B).`,Ue,X,kt=`<strong>GLM-Z1-32B-0414</strong> is a reasoning model with deep thinking capabilities. This was developed based on GLM-4-32B-0414
through cold start, extended reinforcement learning, and further training on tasks including mathematics, code, and
logic. Compared to the base model, GLM-Z1-32B-0414 significantly improves mathematical abilities and the capability to
solve complex tasks. During training, we also introduced general reinforcement learning based on pairwise ranking
feedback, which enhances the model’s general capabilities.`,je,Y,wt=`<strong>GLM-Z1-Rumination-32B-0414</strong> is a deep reasoning model with rumination capabilities (against OpenAI’s Deep Research).
Unlike typical deep thinking models, the rumination model is capable of deeper and longer thinking to solve more
open-ended and complex problems (e.g., writing a comparative analysis of AI development in two cities and their future
development plans). Z1-Rumination is trained through scaling end-to-end reinforcement learning with responses graded by
the ground truth answers or rubrics and can make use of search tools during its deep thinking process to handle complex
tasks. The model shows significant improvements in research-style writing and complex tasks.`,He,K,$t=`Finally, <strong>GLM-Z1-9B-0414</strong> is a surprise. We employed all the aforementioned techniques to train a small model (9B).
GLM-Z1-9B-0414 exhibits excellent capabilities in mathematical reasoning and general tasks. Its overall performance is
top-ranked among all open-source models of the same size. Especially in resource-constrained scenarios, this model
achieves an excellent balance between efficiency and effectiveness, providing a powerful option for users seeking
lightweight deployment.`,Be,ee,We,x,te,Ke,_e,Mt=`This is the configuration class to store the configuration of a <a href="/docs/transformers/v4.56.2/en/model_doc/glm4#transformers.Glm4Model">Glm4Model</a>. It is used to instantiate an Glm4
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the Glm4-4-9b-chat.
e.g. <a href="https://huggingface.co/THUDM/GLM-4-9B-0414" rel="nofollow">THUDM/GLM-4-9B-0414</a>
Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> and can be used to control the model outputs. Read the
documentation from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> for more information.`,et,D,Oe,ne,De,T,oe,tt,be,Ct="The bare Glm4 Model outputting raw hidden-states without any specific head on top.",nt,ve,Gt=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,ot,ye,xt=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,st,q,se,at,Te,zt='The <a href="/docs/transformers/v4.56.2/en/model_doc/glm4#transformers.Glm4Model">Glm4Model</a> forward method, overrides the <code>__call__</code> special method.',rt,N,Ne,ae,Ee,k,re,it,ke,Ft="The Glm4 Model for causal language modeling.",dt,we,Lt=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,lt,$e,qt=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,ct,G,ie,mt,Me,It='The <a href="/docs/transformers/v4.56.2/en/model_doc/glm4#transformers.Glm4ForCausalLM">Glm4ForCausalLM</a> forward method, overrides the <code>__call__</code> special method.',pt,E,ht,S,Se,de,Re,U,le,ut,I,ce,ft,Ce,Pt="The <code>GenericForSequenceClassification</code> forward method, overrides the <code>__call__</code> special method.",gt,R,Ze,me,Ae,j,pe,_t,P,he,bt,Ge,Ut="The <code>GenericForTokenClassification</code> forward method, overrides the <code>__call__</code> special method.",vt,Z,Je,ue,Ve,Le,Qe;return A=new ge({props:{title:"Glm4",local:"glm4",headingTag:"h1"}}),J=new ge({props:{title:"Overview",local:"overview",headingTag:"h2"}}),ee=new ge({props:{title:"Glm4Config",local:"transformers.Glm4Config",headingTag:"h2"}}),te=new W({props:{name:"class transformers.Glm4Config",anchor:"transformers.Glm4Config",parameters:[{name:"vocab_size",val:" = 151552"},{name:"hidden_size",val:" = 4096"},{name:"intermediate_size",val:" = 13696"},{name:"num_hidden_layers",val:" = 40"},{name:"num_attention_heads",val:" = 32"},{name:"num_key_value_heads",val:" = 2"},{name:"partial_rotary_factor",val:" = 0.5"},{name:"head_dim",val:" = 128"},{name:"hidden_act",val:" = 'silu'"},{name:"attention_dropout",val:" = 0.0"},{name:"max_position_embeddings",val:" = 131072"},{name:"initializer_range",val:" = 0.02"},{name:"rms_norm_eps",val:" = 1.5625e-07"},{name:"use_cache",val:" = True"},{name:"tie_word_embeddings",val:" = False"},{name:"rope_theta",val:" = 10000.0"},{name:"pad_token_id",val:" = 151329"},{name:"eos_token_id",val:" = [151329, 151336, 151338]"},{name:"bos_token_id",val:" = None"},{name:"attention_bias",val:" = True"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.Glm4Config.vocab_size",description:`<strong>vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 151552) &#x2014;
Vocabulary size of the Glm4 model. Defines the number of different tokens that can be represented by the
<code>inputs_ids</code> passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/glm4#transformers.Glm4Model">Glm4Model</a>`,name:"vocab_size"},{anchor:"transformers.Glm4Config.hidden_size",description:`<strong>hidden_size</strong> (<code>int</code>, <em>optional</em>, defaults to 4096) &#x2014;
Dimension of the hidden representations.`,name:"hidden_size"},{anchor:"transformers.Glm4Config.intermediate_size",description:`<strong>intermediate_size</strong> (<code>int</code>, <em>optional</em>, defaults to 13696) &#x2014;
Dimension of the MLP representations.`,name:"intermediate_size"},{anchor:"transformers.Glm4Config.num_hidden_layers",description:`<strong>num_hidden_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 40) &#x2014;
Number of hidden layers in the Transformer decoder.`,name:"num_hidden_layers"},{anchor:"transformers.Glm4Config.num_attention_heads",description:`<strong>num_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 32) &#x2014;
Number of attention heads for each attention layer in the Transformer decoder.`,name:"num_attention_heads"},{anchor:"transformers.Glm4Config.num_key_value_heads",description:`<strong>num_key_value_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 2) &#x2014;
This is the number of key_value heads that should be used to implement Grouped Query Attention. If
<code>num_key_value_heads=num_attention_heads</code>, the model will use Multi Head Attention (MHA), if
<code>num_key_value_heads=1</code> the model will use Multi Query Attention (MQA) otherwise GQA is used. When
converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
by meanpooling all the original heads within that group. For more details, check out <a href="https://huggingface.co/papers/2305.13245" rel="nofollow">this
paper</a>. If it is not specified, will default to
<code>num_attention_heads</code>.`,name:"num_key_value_heads"},{anchor:"transformers.Glm4Config.partial_rotary_factor",description:"<strong>partial_rotary_factor</strong> (<code>float</code>, <em>optional</em>, defaults to 0.5) &#x2014; The factor of the partial rotary position.",name:"partial_rotary_factor"},{anchor:"transformers.Glm4Config.head_dim",description:`<strong>head_dim</strong> (<code>int</code>, <em>optional</em>, defaults to 128) &#x2014;
The attention head dimension.`,name:"head_dim"},{anchor:"transformers.Glm4Config.hidden_act",description:`<strong>hidden_act</strong> (<code>str</code> or <code>function</code>, <em>optional</em>, defaults to <code>&quot;silu&quot;</code>) &#x2014;
The legacy activation function. It is overwritten by the <code>hidden_activation</code>.`,name:"hidden_act"},{anchor:"transformers.Glm4Config.attention_dropout",description:`<strong>attention_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The dropout ratio for the attention probabilities.`,name:"attention_dropout"},{anchor:"transformers.Glm4Config.max_position_embeddings",description:`<strong>max_position_embeddings</strong> (<code>int</code>, <em>optional</em>, defaults to 131072) &#x2014;
The maximum sequence length that this model might ever be used with.`,name:"max_position_embeddings"},{anchor:"transformers.Glm4Config.initializer_range",description:`<strong>initializer_range</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.`,name:"initializer_range"},{anchor:"transformers.Glm4Config.rms_norm_eps",description:`<strong>rms_norm_eps</strong> (<code>float</code>, <em>optional</em>, defaults to 1.5625e-07) &#x2014;
The epsilon used by the rms normalization layers.`,name:"rms_norm_eps"},{anchor:"transformers.Glm4Config.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not the model should return the last key/values attentions (not used by all models). Only
relevant if <code>config.is_decoder=True</code>.`,name:"use_cache"},{anchor:"transformers.Glm4Config.tie_word_embeddings",description:`<strong>tie_word_embeddings</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether to tie weight embeddings`,name:"tie_word_embeddings"},{anchor:"transformers.Glm4Config.rope_theta",description:`<strong>rope_theta</strong> (<code>float</code>, <em>optional</em>, defaults to 10000.0) &#x2014;
The base period of the RoPE embeddings.`,name:"rope_theta"},{anchor:"transformers.Glm4Config.pad_token_id",description:`<strong>pad_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 151329) &#x2014;
Padding token id.`,name:"pad_token_id"},{anchor:"transformers.Glm4Config.eos_token_id",description:`<strong>eos_token_id</strong> (<code>int</code> | <code>list</code>, <em>optional</em>, defaults to <code>[151329, 151336, 151338]</code>) &#x2014;
End of stream token id.`,name:"eos_token_id"},{anchor:"transformers.Glm4Config.bos_token_id",description:`<strong>bos_token_id</strong> (<code>int</code>, <em>optional</em>) &#x2014;
Beginning of stream token id.`,name:"bos_token_id"},{anchor:"transformers.Glm4Config.attention_bias",description:`<strong>attention_bias</strong> (<code>bool</code>, defaults to <code>False</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether to use a bias in the query, key, value and output projection layers during self-attention.`,name:"attention_bias"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/glm4/configuration_glm4.py#L20"}}),D=new jt({props:{anchor:"transformers.Glm4Config.example",$$slots:{default:[St]},$$scope:{ctx:w}}}),ne=new ge({props:{title:"Glm4Model",local:"transformers.Glm4Model",headingTag:"h2"}}),oe=new W({props:{name:"class transformers.Glm4Model",anchor:"transformers.Glm4Model",parameters:[{name:"config",val:": Glm4Config"}],parametersDescription:[{anchor:"transformers.Glm4Model.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/glm4#transformers.Glm4Config">Glm4Config</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/glm4/modeling_glm4.py#L348"}}),se=new W({props:{name:"forward",anchor:"transformers.Glm4Model.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.Glm4Model.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.Glm4Model.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.Glm4Model.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.Glm4Model.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.Glm4Model.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.Glm4Model.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"},{anchor:"transformers.Glm4Model.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/glm4/modeling_glm4.py#L365",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast"
>transformers.modeling_outputs.BaseModelOutputWithPast</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/glm4#transformers.Glm4Config"
>Glm4Config</a>) and inputs.</p>
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
`}}),N=new Ye({props:{$$slots:{default:[Rt]},$$scope:{ctx:w}}}),ae=new ge({props:{title:"Glm4ForCausalLM",local:"transformers.Glm4ForCausalLM",headingTag:"h2"}}),re=new W({props:{name:"class transformers.Glm4ForCausalLM",anchor:"transformers.Glm4ForCausalLM",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.Glm4ForCausalLM.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/glm4#transformers.Glm4ForCausalLM">Glm4ForCausalLM</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/glm4/modeling_glm4.py#L427"}}),ie=new W({props:{name:"forward",anchor:"transformers.Glm4ForCausalLM.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"},{name:"logits_to_keep",val:": typing.Union[int, torch.Tensor] = 0"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.Glm4ForCausalLM.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.Glm4ForCausalLM.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.Glm4ForCausalLM.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.Glm4ForCausalLM.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.Glm4ForCausalLM.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.Glm4ForCausalLM.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.Glm4ForCausalLM.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.Glm4ForCausalLM.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"},{anchor:"transformers.Glm4ForCausalLM.forward.logits_to_keep",description:`<strong>logits_to_keep</strong> (<code>Union[int, torch.Tensor]</code>, defaults to <code>0</code>) &#x2014;
If an <code>int</code>, compute logits for the last <code>logits_to_keep</code> tokens. If <code>0</code>, calculate logits for all
<code>input_ids</code> (special case). Only last token logits are needed for generation, and calculating them only for that
token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
If a <code>torch.Tensor</code>, must be 1D corresponding to the indices to keep in the sequence length dimension.
This is useful when using packed tensor format (single dimension for batch and sequence length).`,name:"logits_to_keep"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/glm4/modeling_glm4.py#L441",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast"
>transformers.modeling_outputs.CausalLMOutputWithPast</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/glm4#transformers.Glm4Config"
>Glm4Config</a>) and inputs.</p>
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
`}}),E=new Ye({props:{$$slots:{default:[Zt]},$$scope:{ctx:w}}}),S=new jt({props:{anchor:"transformers.Glm4ForCausalLM.forward.example",$$slots:{default:[At]},$$scope:{ctx:w}}}),de=new ge({props:{title:"Glm4ForSequenceClassification",local:"transformers.Glm4ForSequenceClassification",headingTag:"h2"}}),le=new W({props:{name:"class transformers.Glm4ForSequenceClassification",anchor:"transformers.Glm4ForSequenceClassification",parameters:[{name:"config",val:""}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/glm4/modeling_glm4.py#L507"}}),ce=new W({props:{name:"forward",anchor:"transformers.Glm4ForSequenceClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.Glm4ForSequenceClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.Glm4ForSequenceClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.Glm4ForSequenceClassification.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.Glm4ForSequenceClassification.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.Glm4ForSequenceClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.Glm4ForSequenceClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.Glm4ForSequenceClassification.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
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
`}}),R=new Ye({props:{$$slots:{default:[Jt]},$$scope:{ctx:w}}}),me=new ge({props:{title:"Glm4ForTokenClassification",local:"transformers.Glm4ForTokenClassification",headingTag:"h2"}}),pe=new W({props:{name:"class transformers.Glm4ForTokenClassification",anchor:"transformers.Glm4ForTokenClassification",parameters:[{name:"config",val:""}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/glm4/modeling_glm4.py#L511"}}),he=new W({props:{name:"forward",anchor:"transformers.Glm4ForTokenClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.Glm4ForTokenClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.Glm4ForTokenClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.Glm4ForTokenClassification.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.Glm4ForTokenClassification.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.Glm4ForTokenClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.Glm4ForTokenClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.Glm4ForTokenClassification.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
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
`}}),Z=new Ye({props:{$$slots:{default:[Vt]},$$scope:{ctx:w}}}),ue=new Et({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/glm4.md"}}),{c(){n=l("meta"),_=a(),s=l("p"),b=a(),C=l("p"),C.innerHTML=y,O=a(),m(A.$$.fragment),qe=a(),m(J.$$.fragment),Ie=a(),V=l("p"),V.innerHTML=yt,Pe=a(),Q=l("p"),Q.innerHTML=Tt,Ue=a(),X=l("p"),X.innerHTML=kt,je=a(),Y=l("p"),Y.innerHTML=wt,He=a(),K=l("p"),K.innerHTML=$t,Be=a(),m(ee.$$.fragment),We=a(),x=l("div"),m(te.$$.fragment),Ke=a(),_e=l("p"),_e.innerHTML=Mt,et=a(),m(D.$$.fragment),Oe=a(),m(ne.$$.fragment),De=a(),T=l("div"),m(oe.$$.fragment),tt=a(),be=l("p"),be.textContent=Ct,nt=a(),ve=l("p"),ve.innerHTML=Gt,ot=a(),ye=l("p"),ye.innerHTML=xt,st=a(),q=l("div"),m(se.$$.fragment),at=a(),Te=l("p"),Te.innerHTML=zt,rt=a(),m(N.$$.fragment),Ne=a(),m(ae.$$.fragment),Ee=a(),k=l("div"),m(re.$$.fragment),it=a(),ke=l("p"),ke.textContent=Ft,dt=a(),we=l("p"),we.innerHTML=Lt,lt=a(),$e=l("p"),$e.innerHTML=qt,ct=a(),G=l("div"),m(ie.$$.fragment),mt=a(),Me=l("p"),Me.innerHTML=It,pt=a(),m(E.$$.fragment),ht=a(),m(S.$$.fragment),Se=a(),m(de.$$.fragment),Re=a(),U=l("div"),m(le.$$.fragment),ut=a(),I=l("div"),m(ce.$$.fragment),ft=a(),Ce=l("p"),Ce.innerHTML=Pt,gt=a(),m(R.$$.fragment),Ze=a(),m(me.$$.fragment),Ae=a(),j=l("div"),m(pe.$$.fragment),_t=a(),P=l("div"),m(he.$$.fragment),bt=a(),Ge=l("p"),Ge.innerHTML=Ut,vt=a(),m(Z.$$.fragment),Je=a(),m(ue.$$.fragment),Ve=a(),Le=l("p"),this.h()},l(e){const t=Nt("svelte-u9bgzb",document.head);n=c(t,"META",{name:!0,content:!0}),t.forEach(o),_=r(e),s=c(e,"P",{}),F(s).forEach(o),b=r(e),C=c(e,"P",{"data-svelte-h":!0}),v(C)!=="svelte-12y99gd"&&(C.innerHTML=y),O=r(e),p(A.$$.fragment,e),qe=r(e),p(J.$$.fragment,e),Ie=r(e),V=c(e,"P",{"data-svelte-h":!0}),v(V)!=="svelte-14stjzy"&&(V.innerHTML=yt),Pe=r(e),Q=c(e,"P",{"data-svelte-h":!0}),v(Q)!=="svelte-qgktol"&&(Q.innerHTML=Tt),Ue=r(e),X=c(e,"P",{"data-svelte-h":!0}),v(X)!=="svelte-8wxxjs"&&(X.innerHTML=kt),je=r(e),Y=c(e,"P",{"data-svelte-h":!0}),v(Y)!=="svelte-10i6pia"&&(Y.innerHTML=wt),He=r(e),K=c(e,"P",{"data-svelte-h":!0}),v(K)!=="svelte-13jg0ss"&&(K.innerHTML=$t),Be=r(e),p(ee.$$.fragment,e),We=r(e),x=c(e,"DIV",{class:!0});var H=F(x);p(te.$$.fragment,H),Ke=r(H),_e=c(H,"P",{"data-svelte-h":!0}),v(_e)!=="svelte-5vv09n"&&(_e.innerHTML=Mt),et=r(H),p(D.$$.fragment,H),H.forEach(o),Oe=r(e),p(ne.$$.fragment,e),De=r(e),T=c(e,"DIV",{class:!0});var $=F(T);p(oe.$$.fragment,$),tt=r($),be=c($,"P",{"data-svelte-h":!0}),v(be)!=="svelte-1s85vwu"&&(be.textContent=Ct),nt=r($),ve=c($,"P",{"data-svelte-h":!0}),v(ve)!=="svelte-q52n56"&&(ve.innerHTML=Gt),ot=r($),ye=c($,"P",{"data-svelte-h":!0}),v(ye)!=="svelte-hswkmf"&&(ye.innerHTML=xt),st=r($),q=c($,"DIV",{class:!0});var B=F(q);p(se.$$.fragment,B),at=r(B),Te=c(B,"P",{"data-svelte-h":!0}),v(Te)!=="svelte-ahbyh"&&(Te.innerHTML=zt),rt=r(B),p(N.$$.fragment,B),B.forEach(o),$.forEach(o),Ne=r(e),p(ae.$$.fragment,e),Ee=r(e),k=c(e,"DIV",{class:!0});var M=F(k);p(re.$$.fragment,M),it=r(M),ke=c(M,"P",{"data-svelte-h":!0}),v(ke)!=="svelte-1311zvd"&&(ke.textContent=Ft),dt=r(M),we=c(M,"P",{"data-svelte-h":!0}),v(we)!=="svelte-q52n56"&&(we.innerHTML=Lt),lt=r(M),$e=c(M,"P",{"data-svelte-h":!0}),v($e)!=="svelte-hswkmf"&&($e.innerHTML=qt),ct=r(M),G=c(M,"DIV",{class:!0});var z=F(G);p(ie.$$.fragment,z),mt=r(z),Me=c(z,"P",{"data-svelte-h":!0}),v(Me)!=="svelte-b175j1"&&(Me.innerHTML=It),pt=r(z),p(E.$$.fragment,z),ht=r(z),p(S.$$.fragment,z),z.forEach(o),M.forEach(o),Se=r(e),p(de.$$.fragment,e),Re=r(e),U=c(e,"DIV",{class:!0});var fe=F(U);p(le.$$.fragment,fe),ut=r(fe),I=c(fe,"DIV",{class:!0});var xe=F(I);p(ce.$$.fragment,xe),ft=r(xe),Ce=c(xe,"P",{"data-svelte-h":!0}),v(Ce)!=="svelte-1sal4ui"&&(Ce.innerHTML=Pt),gt=r(xe),p(R.$$.fragment,xe),xe.forEach(o),fe.forEach(o),Ze=r(e),p(me.$$.fragment,e),Ae=r(e),j=c(e,"DIV",{class:!0});var Xe=F(j);p(pe.$$.fragment,Xe),_t=r(Xe),P=c(Xe,"DIV",{class:!0});var ze=F(P);p(he.$$.fragment,ze),bt=r(ze),Ge=c(ze,"P",{"data-svelte-h":!0}),v(Ge)!=="svelte-1py4aay"&&(Ge.innerHTML=Ut),vt=r(ze),p(Z.$$.fragment,ze),ze.forEach(o),Xe.forEach(o),Je=r(e),p(ue.$$.fragment,e),Ve=r(e),Le=c(e,"P",{}),F(Le).forEach(o),this.h()},h(){L(n,"name","hf:doc:metadata"),L(n,"content",Xt),L(x,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),L(q,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),L(T,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),L(G,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),L(k,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),L(I,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),L(U,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),L(P,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),L(j,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,t){d(document.head,n),i(e,_,t),i(e,s,t),i(e,b,t),i(e,C,t),i(e,O,t),h(A,e,t),i(e,qe,t),h(J,e,t),i(e,Ie,t),i(e,V,t),i(e,Pe,t),i(e,Q,t),i(e,Ue,t),i(e,X,t),i(e,je,t),i(e,Y,t),i(e,He,t),i(e,K,t),i(e,Be,t),h(ee,e,t),i(e,We,t),i(e,x,t),h(te,x,null),d(x,Ke),d(x,_e),d(x,et),h(D,x,null),i(e,Oe,t),h(ne,e,t),i(e,De,t),i(e,T,t),h(oe,T,null),d(T,tt),d(T,be),d(T,nt),d(T,ve),d(T,ot),d(T,ye),d(T,st),d(T,q),h(se,q,null),d(q,at),d(q,Te),d(q,rt),h(N,q,null),i(e,Ne,t),h(ae,e,t),i(e,Ee,t),i(e,k,t),h(re,k,null),d(k,it),d(k,ke),d(k,dt),d(k,we),d(k,lt),d(k,$e),d(k,ct),d(k,G),h(ie,G,null),d(G,mt),d(G,Me),d(G,pt),h(E,G,null),d(G,ht),h(S,G,null),i(e,Se,t),h(de,e,t),i(e,Re,t),i(e,U,t),h(le,U,null),d(U,ut),d(U,I),h(ce,I,null),d(I,ft),d(I,Ce),d(I,gt),h(R,I,null),i(e,Ze,t),h(me,e,t),i(e,Ae,t),i(e,j,t),h(pe,j,null),d(j,_t),d(j,P),h(he,P,null),d(P,bt),d(P,Ge),d(P,vt),h(Z,P,null),i(e,Je,t),h(ue,e,t),i(e,Ve,t),i(e,Le,t),Qe=!0},p(e,[t]){const H={};t&2&&(H.$$scope={dirty:t,ctx:e}),D.$set(H);const $={};t&2&&($.$$scope={dirty:t,ctx:e}),N.$set($);const B={};t&2&&(B.$$scope={dirty:t,ctx:e}),E.$set(B);const M={};t&2&&(M.$$scope={dirty:t,ctx:e}),S.$set(M);const z={};t&2&&(z.$$scope={dirty:t,ctx:e}),R.$set(z);const fe={};t&2&&(fe.$$scope={dirty:t,ctx:e}),Z.$set(fe)},i(e){Qe||(u(A.$$.fragment,e),u(J.$$.fragment,e),u(ee.$$.fragment,e),u(te.$$.fragment,e),u(D.$$.fragment,e),u(ne.$$.fragment,e),u(oe.$$.fragment,e),u(se.$$.fragment,e),u(N.$$.fragment,e),u(ae.$$.fragment,e),u(re.$$.fragment,e),u(ie.$$.fragment,e),u(E.$$.fragment,e),u(S.$$.fragment,e),u(de.$$.fragment,e),u(le.$$.fragment,e),u(ce.$$.fragment,e),u(R.$$.fragment,e),u(me.$$.fragment,e),u(pe.$$.fragment,e),u(he.$$.fragment,e),u(Z.$$.fragment,e),u(ue.$$.fragment,e),Qe=!0)},o(e){f(A.$$.fragment,e),f(J.$$.fragment,e),f(ee.$$.fragment,e),f(te.$$.fragment,e),f(D.$$.fragment,e),f(ne.$$.fragment,e),f(oe.$$.fragment,e),f(se.$$.fragment,e),f(N.$$.fragment,e),f(ae.$$.fragment,e),f(re.$$.fragment,e),f(ie.$$.fragment,e),f(E.$$.fragment,e),f(S.$$.fragment,e),f(de.$$.fragment,e),f(le.$$.fragment,e),f(ce.$$.fragment,e),f(R.$$.fragment,e),f(me.$$.fragment,e),f(pe.$$.fragment,e),f(he.$$.fragment,e),f(Z.$$.fragment,e),f(ue.$$.fragment,e),Qe=!1},d(e){e&&(o(_),o(s),o(b),o(C),o(O),o(qe),o(Ie),o(V),o(Pe),o(Q),o(Ue),o(X),o(je),o(Y),o(He),o(K),o(Be),o(We),o(x),o(Oe),o(De),o(T),o(Ne),o(Ee),o(k),o(Se),o(Re),o(U),o(Ze),o(Ae),o(j),o(Je),o(Ve),o(Le)),o(n),g(A,e),g(J,e),g(ee,e),g(te),g(D),g(ne,e),g(oe),g(se),g(N),g(ae,e),g(re),g(ie),g(E),g(S),g(de,e),g(le),g(ce),g(R),g(me,e),g(pe),g(he),g(Z),g(ue,e)}}}const Xt='{"title":"Glm4","local":"glm4","sections":[{"title":"Overview","local":"overview","sections":[],"depth":2},{"title":"Glm4Config","local":"transformers.Glm4Config","sections":[],"depth":2},{"title":"Glm4Model","local":"transformers.Glm4Model","sections":[],"depth":2},{"title":"Glm4ForCausalLM","local":"transformers.Glm4ForCausalLM","sections":[],"depth":2},{"title":"Glm4ForSequenceClassification","local":"transformers.Glm4ForSequenceClassification","sections":[],"depth":2},{"title":"Glm4ForTokenClassification","local":"transformers.Glm4ForTokenClassification","sections":[],"depth":2}],"depth":1}';function Yt(w){return Wt(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class rn extends Ot{constructor(n){super(),Dt(this,n,Yt,Qt,Bt,{})}}export{rn as component};
