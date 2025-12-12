import{s as Zo,z as Bo,o as So,n as Ue}from"../chunks/scheduler.18a86fab.js";import{S as Vo,i as Po,g as l,s,r as p,A as Ho,h as c,f as t,c as a,j as J,x as b,u as m,k as D,y as d,a as r,v as u,d as h,t as g,w as f}from"../chunks/index.98837b22.js";import{T as wo}from"../chunks/Tip.77304350.js";import{D as he}from"../chunks/Docstring.a1ef7999.js";import{C as Ye}from"../chunks/CodeBlock.8d0c2e8a.js";import{E as Wo}from"../chunks/ExampleCodeBlock.8c3ee1f9.js";import{H as ge,E as Xo}from"../chunks/getInferenceSnippets.06c2775f.js";function Go(x){let n,_;return n=new Ye({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMERvZ2VDb25maWclMkMlMjBEb2dlTW9kZWwlMEElMEElMjMlMjBJbml0aWFsaXppbmclMjBhJTIwRG9nZS0zMjBNJTIwc3R5bGUlMjBjb25maWd1cmF0aW9uJTBBY29uZmlndXJhdGlvbiUyMCUzRCUyMERvZ2VDb25maWcoKSUwQSUwQSUyMyUyMEluaXRpYWxpemluZyUyMGElMjBtb2RlbCUyMGZyb20lMjB0aGUlMjBEb2dlLTMyME0lMjBzdHlsZSUyMGNvbmZpZ3VyYXRpb24lMEFtb2RlbCUyMCUzRCUyMERvZ2VNb2RlbChjb25maWd1cmF0aW9uKSUwQSUwQSUyMyUyMEFjY2Vzc2luZyUyMHRoZSUyMG1vZGVsJTIwY29uZmlndXJhdGlvbiUwQWNvbmZpZ3VyYXRpb24lMjAlM0QlMjBtb2RlbC5jb25maWc=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> DogeConfig, DogeModel

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a Doge-320M style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = DogeConfig()

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a model from the Doge-320M style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = DogeModel(configuration)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Accessing the model configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = model.config`,wrap:!1}}),{c(){p(n.$$.fragment)},l(i){m(n.$$.fragment,i)},m(i,y){u(n,i,y),_=!0},p:Ue,i(i){_||(h(n.$$.fragment,i),_=!0)},o(i){g(n.$$.fragment,i),_=!1},d(i){f(n,i)}}}function No(x){let n,_=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){n=l("p"),n.innerHTML=_},l(i){n=c(i,"P",{"data-svelte-h":!0}),b(n)!=="svelte-fincs2"&&(n.innerHTML=_)},m(i,y){r(i,n,y)},p:Ue,d(i){i&&t(n)}}}function Ao(x){let n,_=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){n=l("p"),n.innerHTML=_},l(i){n=c(i,"P",{"data-svelte-h":!0}),b(n)!=="svelte-fincs2"&&(n.innerHTML=_)},m(i,y){r(i,n,y)},p:Ue,d(i){i&&t(n)}}}function Qo(x){let n,_="Example:",i,y,C;return y=new Ye({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBEb2dlRm9yQ2F1c2FsTE0lMEElMEFtb2RlbCUyMCUzRCUyMERvZ2VGb3JDYXVzYWxMTS5mcm9tX3ByZXRyYWluZWQoJTIyU21hbGxEb2dlJTJGRG9nZS0zMjBNJTIyKSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMlNtYWxsRG9nZSUyRkRvZ2UtMzIwTSUyMiklMEElMEFwcm9tcHQlMjAlM0QlMjAlMjJIZXklMkMlMjBhcmUlMjB5b3UlMjBjb25zY2lvdXMlM0YlMjBDYW4lMjB5b3UlMjB0YWxrJTIwdG8lMjBtZSUzRiUyMiUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplcihwcm9tcHQlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQSUwQSUyMyUyMEdlbmVyYXRlJTBBZ2VuZXJhdGVfaWRzJTIwJTNEJTIwbW9kZWwuZ2VuZXJhdGUoaW5wdXRzLmlucHV0X2lkcyUyQyUyMG1heF9sZW5ndGglM0QzMCklMEF0b2tlbml6ZXIuYmF0Y2hfZGVjb2RlKGdlbmVyYXRlX2lkcyUyQyUyMHNraXBfc3BlY2lhbF90b2tlbnMlM0RUcnVlJTJDJTIwY2xlYW5fdXBfdG9rZW5pemF0aW9uX3NwYWNlcyUzREZhbHNlKSU1QjAlNUQ=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, DogeForCausalLM

<span class="hljs-meta">&gt;&gt;&gt; </span>model = DogeForCausalLM.from_pretrained(<span class="hljs-string">&quot;SmallDoge/Doge-320M&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;SmallDoge/Doge-320M&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>prompt = <span class="hljs-string">&quot;Hey, are you conscious? Can you talk to me?&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(prompt, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Generate</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>generate_ids = model.generate(inputs.input_ids, max_length=<span class="hljs-number">30</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer.batch_decode(generate_ids, skip_special_tokens=<span class="hljs-literal">True</span>, clean_up_tokenization_spaces=<span class="hljs-literal">False</span>)[<span class="hljs-number">0</span>]
<span class="hljs-string">&quot;Hey, are you conscious? Can you talk to me?\\nI&#x27;m not conscious, but I can talk to you.&quot;</span>`,wrap:!1}}),{c(){n=l("p"),n.textContent=_,i=s(),p(y.$$.fragment)},l(T){n=c(T,"P",{"data-svelte-h":!0}),b(n)!=="svelte-11lpom8"&&(n.textContent=_),i=a(T),m(y.$$.fragment,T)},m(T,L){r(T,n,L),r(T,i,L),u(y,T,L),C=!0},p:Ue,i(T){C||(h(y.$$.fragment,T),C=!0)},o(T){g(y.$$.fragment,T),C=!1},d(T){T&&(t(n),t(i)),f(y,T)}}}function Oo(x){let n,_=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){n=l("p"),n.innerHTML=_},l(i){n=c(i,"P",{"data-svelte-h":!0}),b(n)!=="svelte-fincs2"&&(n.innerHTML=_)},m(i,y){r(i,n,y)},p:Ue,d(i){i&&t(n)}}}function Yo(x){let n,_,i,y,C,T="<em>This model was released on 2024-12-27 and added to Hugging Face Transformers on 2025-07-08.</em>",L,X,Ie,G,Fe,N,vo='Doge is a series of small language models based on the <a href="https://github.com/SmallDoges/small-doge" rel="nofollow">Doge</a> architecture, aiming to combine the advantages of state-space and self-attention algorithms, calculate dynamic masks from cached value states using the zero-order hold method, and solve the problem of existing mainstream language models getting lost in context. It uses the <code>wsd_scheduler</code> scheduler to pre-train on the <code>smollm-corpus</code>, and can continue training on new datasets or add sparse activation feedforward networks from stable stage checkpoints.',je,E,Mo,qe,A,ko="As shown in the figure below, the sequence transformation part of the Doge architecture uses <code>Dynamic Mask Attention</code>, which can be understood as using self-attention related to value states during training, and using state-space without past state decay during inference, to solve the problem of existing Transformers or SSMs getting lost in long text. The state transformation part of Doge uses <code>Cross Domain Mixture of Experts</code>, which consists of dense linear layers and sparse embedding layers, and can additionally increase sparse parameters to continue training from dense weight checkpoints without retraining the entire model, thereby reducing the cost of continuous iteration of the model. In addition, Doge also uses <code>RMSNorm</code> and <code>Residual</code> with learnable parameters to adapt the gradient range of deep models.",Le,Q,xo='Checkout all Doge model checkpoints <a href="https://huggingface.co/collections/SmallDoge/doge-slm-679cc991f027c4a3abbded4a" rel="nofollow">here</a>.',Ee,O,Re,R,fe,Co="Using Doge-Base for text generation",Ke,Y,We,W,_e,$o="Using Doge-Instruct for question answering",eo,K,Ze,ee,Be,M,oe,oo,be,Do=`This is the configuration class to store the configuration of a <a href="/docs/transformers/v4.56.2/en/model_doc/doge#transformers.DogeModel">DogeModel</a>. It is used to instantiate an Doge
model according to the specified arguments, defining the model architecture like <a href="https://huggingface.co/SmallDoge/Doge-320M" rel="nofollow">SmallDoge/Doge-320M</a>.`,to,ye,zo=`Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> and can be used to control the model outputs. Read the
documentation from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> for more information.`,no,Z,Se,te,Ve,w,ne,so,Te,Jo="The bare Doge Model outputting raw hidden-states without any specific head on top.",ao,we,Uo=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,ro,ve,Io=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,io,U,se,lo,Me,Fo='The <a href="/docs/transformers/v4.56.2/en/model_doc/doge#transformers.DogeModel">DogeModel</a> forward method, overrides the <code>__call__</code> special method.',co,B,Pe,ae,He,v,re,po,ke,jo="The Doge Model for causal language modeling.",mo,xe,qo=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,uo,Ce,Lo=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,ho,$,ie,go,$e,Eo='The <a href="/docs/transformers/v4.56.2/en/model_doc/doge#transformers.DogeForCausalLM">DogeForCausalLM</a> forward method, overrides the <code>__call__</code> special method.',fo,S,_o,V,Xe,de,Ge,j,le,bo,I,ce,yo,De,Ro="The <code>GenericForSequenceClassification</code> forward method, overrides the <code>__call__</code> special method.",To,P,Ne,pe,Ae,Je,Qe;return X=new ge({props:{title:"Doge",local:"doge",headingTag:"h1"}}),G=new ge({props:{title:"Overview",local:"overview",headingTag:"h2"}}),O=new ge({props:{title:"Usage",local:"usage",headingTag:"h2"}}),Y=new Ye({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBBdXRvTW9kZWxGb3JDYXVzYWxMTSUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMlNtYWxsRG9nZSUyRkRvZ2UtMjBNJTIyKSUwQW1vZGVsJTIwJTNEJTIwQXV0b01vZGVsRm9yQ2F1c2FsTE0uZnJvbV9wcmV0cmFpbmVkKCUyMlNtYWxsRG9nZSUyRkRvZ2UtMjBNJTIyKSUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplciglMjJIZXklMjBob3clMjBhcmUlMjB5b3UlMjBkb2luZyUzRiUyMiUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpJTBBJTBBb3V0cHV0cyUyMCUzRCUyMG1vZGVsLmdlbmVyYXRlKCoqaW5wdXRzJTJDJTIwbWF4X25ld190b2tlbnMlM0QxMDApJTBBcHJpbnQodG9rZW5pemVyLmJhdGNoX2RlY29kZShvdXRwdXRzKSk=",highlighted:`<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;SmallDoge/Doge-20M&quot;</span>)
model = AutoModelForCausalLM.from_pretrained(<span class="hljs-string">&quot;SmallDoge/Doge-20M&quot;</span>)
inputs = tokenizer(<span class="hljs-string">&quot;Hey how are you doing?&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

outputs = model.generate(**inputs, max_new_tokens=<span class="hljs-number">100</span>)
<span class="hljs-built_in">print</span>(tokenizer.batch_decode(outputs))`,wrap:!1}}),K=new Ye({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBBdXRvTW9kZWxGb3JDYXVzYWxMTSUyQyUyMEdlbmVyYXRpb25Db25maWclMkMlMjBUZXh0U3RyZWFtZXIlMEElMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJTbWFsbERvZ2UlMkZEb2dlLTIwTS1JbnN0cnVjdCUyMiklMEFtb2RlbCUyMCUzRCUyMEF1dG9Nb2RlbEZvckNhdXNhbExNLmZyb21fcHJldHJhaW5lZCglMjJTbWFsbERvZ2UlMkZEb2dlLTIwTS1JbnN0cnVjdCUyMiklMEElMEFnZW5lcmF0aW9uX2NvbmZpZyUyMCUzRCUyMEdlbmVyYXRpb25Db25maWcoJTBBJTIwJTIwJTIwJTIwJTIwJTIwbWF4X25ld190b2tlbnMlM0QxMDAlMkMlMjAlMEElMjAlMjAlMjAlMjAlMjAlMjB1c2VfY2FjaGUlM0RUcnVlJTJDJTIwJTBBJTIwJTIwJTIwJTIwJTIwJTIwZG9fc2FtcGxlJTNEVHJ1ZSUyQyUyMCUwQSUyMCUyMCUyMCUyMCUyMCUyMHRlbXBlcmF0dXJlJTNEMC44JTJDJTIwJTBBJTIwJTIwJTIwJTIwJTIwJTIwdG9wX3AlM0QwLjklMkMlMEElMjAlMjAlMjAlMjAlMjAlMjByZXBldGl0aW9uX3BlbmFsdHklM0QxLjAlMEEpJTBBc3RlYW1lciUyMCUzRCUyMFRleHRTdHJlYW1lcih0b2tlbml6ZXIlM0R0b2tlbml6ZXIlMkMlMjBza2lwX3Byb21wdCUzRFRydWUpJTBBJTBBcHJvbXB0JTIwJTNEJTIwJTIySGklMkMlMjBob3clMjBhcmUlMjB5b3UlMjBkb2luZyUyMHRvZGF5JTNGJTIyJTBBY29udmVyc2F0aW9uJTIwJTNEJTIwJTVCJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTdCJTIycm9sZSUyMiUzQSUyMCUyMnVzZXIlMjIlMkMlMjAlMjJjb250ZW50JTIyJTNBJTIwcHJvbXB0JTdEJTBBJTVEJTBBaW5wdXRzJTIwJTNEJTIwdG9rZW5pemVyLmFwcGx5X2NoYXRfdGVtcGxhdGUoJTBBJTIwJTIwJTIwJTIwY29udmVyc2F0aW9uJTNEY29udmVyc2F0aW9uJTJDJTBBJTIwJTIwJTIwJTIwdG9rZW5pemUlM0RUcnVlJTJDJTBBJTIwJTIwJTIwJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiUyQyUwQSklMEElMEFvdXRwdXRzJTIwJTNEJTIwbW9kZWwuZ2VuZXJhdGUoJTBBJTIwJTIwJTIwJTIwaW5wdXRzJTJDJTIwJTBBJTIwJTIwJTIwJTIwdG9rZW5pemVyJTNEdG9rZW5pemVyJTJDJTBBJTIwJTIwJTIwJTIwZ2VuZXJhdGlvbl9jb25maWclM0RnZW5lcmF0aW9uX2NvbmZpZyUyQyUyMCUwQSUyMCUyMCUyMCUyMHN0cmVhbWVyJTNEc3RlYW1lciUwQSk=",highlighted:`<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, AutoModelForCausalLM, GenerationConfig, TextStreamer

tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;SmallDoge/Doge-20M-Instruct&quot;</span>)
model = AutoModelForCausalLM.from_pretrained(<span class="hljs-string">&quot;SmallDoge/Doge-20M-Instruct&quot;</span>)

generation_config = GenerationConfig(
      max_new_tokens=<span class="hljs-number">100</span>, 
      use_cache=<span class="hljs-literal">True</span>, 
      do_sample=<span class="hljs-literal">True</span>, 
      temperature=<span class="hljs-number">0.8</span>, 
      top_p=<span class="hljs-number">0.9</span>,
      repetition_penalty=<span class="hljs-number">1.0</span>
)
steamer = TextStreamer(tokenizer=tokenizer, skip_prompt=<span class="hljs-literal">True</span>)

prompt = <span class="hljs-string">&quot;Hi, how are you doing today?&quot;</span>
conversation = [
      {<span class="hljs-string">&quot;role&quot;</span>: <span class="hljs-string">&quot;user&quot;</span>, <span class="hljs-string">&quot;content&quot;</span>: prompt}
]
inputs = tokenizer.apply_chat_template(
    conversation=conversation,
    tokenize=<span class="hljs-literal">True</span>,
    return_tensors=<span class="hljs-string">&quot;pt&quot;</span>,
)

outputs = model.generate(
    inputs, 
    tokenizer=tokenizer,
    generation_config=generation_config, 
    streamer=steamer
)`,wrap:!1}}),ee=new ge({props:{title:"DogeConfig",local:"transformers.DogeConfig",headingTag:"h2"}}),oe=new he({props:{name:"class transformers.DogeConfig",anchor:"transformers.DogeConfig",parameters:[{name:"vocab_size",val:" = 32768"},{name:"hidden_size",val:" = 1024"},{name:"intermediate_size",val:" = 2048"},{name:"num_hidden_layers",val:" = 32"},{name:"hidden_dropout",val:" = 0.0"},{name:"hidden_act",val:" = 'silu'"},{name:"initializer_range",val:" = 0.02"},{name:"rms_norm_eps",val:" = 1e-06"},{name:"use_cache",val:" = True"},{name:"tie_word_embeddings",val:" = False"},{name:"max_position_embeddings",val:" = 2048"},{name:"rope_theta",val:" = 10000.0"},{name:"rope_scaling",val:" = None"},{name:"num_attention_heads",val:" = 8"},{name:"num_key_value_heads",val:" = None"},{name:"attention_bias",val:" = False"},{name:"attention_dropout",val:" = 0.0"},{name:"mlp_bias",val:" = False"},{name:"sliding_window",val:" = None"},{name:"keep_window_size",val:" = 2048"},{name:"is_moe",val:" = False"},{name:"num_experts",val:" = 16384"},{name:"num_experts_per_tok",val:" = 64"},{name:"norm_topk_prob",val:" = False"},{name:"output_router_logits",val:" = False"},{name:"router_aux_loss_coef",val:" = 0.001"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.DogeConfig.vocab_size",description:`<strong>vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 32768) &#x2014;
Vocabulary size of the Doge2 model. Defines the number of different tokens that can be represented by the <code>inputs_ids</code> passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/doge#transformers.DogeModel">DogeModel</a>`,name:"vocab_size"},{anchor:"transformers.DogeConfig.hidden_size",description:`<strong>hidden_size</strong> (<code>int</code>, <em>optional</em>, defaults to 1024) &#x2014;
Dimension of the hidden representations.`,name:"hidden_size"},{anchor:"transformers.DogeConfig.intermediate_size",description:`<strong>intermediate_size</strong> (<code>int</code>, <em>optional</em>, defaults to 2048) &#x2014;
Dimension of the MLP representations.`,name:"intermediate_size"},{anchor:"transformers.DogeConfig.num_hidden_layers",description:`<strong>num_hidden_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 32) &#x2014;
Number of hidden layers in the Transformer decoder.`,name:"num_hidden_layers"},{anchor:"transformers.DogeConfig.hidden_dropout",description:`<strong>hidden_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
Dropout probability for each sequence transformation and state transformation module.`,name:"hidden_dropout"},{anchor:"transformers.DogeConfig.hidden_act",description:`<strong>hidden_act</strong> (<code>str</code> or <code>function</code>, <em>optional</em>, defaults to <code>&quot;silu&quot;</code>) &#x2014;
The non-linear activation function (function or string) in the decoder.`,name:"hidden_act"},{anchor:"transformers.DogeConfig.initializer_range",description:`<strong>initializer_range</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.`,name:"initializer_range"},{anchor:"transformers.DogeConfig.rms_norm_eps",description:`<strong>rms_norm_eps</strong> (<code>float</code>, <em>optional</em>, defaults to 1e-06) &#x2014;
The epsilon used by the rms normalization layers.`,name:"rms_norm_eps"},{anchor:"transformers.DogeConfig.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not the model should return the last key/values attentions (not used by all models). Only
relevant if <code>config.is_decoder=True</code>.`,name:"use_cache"},{anchor:"transformers.DogeConfig.tie_word_embeddings",description:`<strong>tie_word_embeddings</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether the model&#x2019;s input and output word embeddings should be tied.`,name:"tie_word_embeddings"},{anchor:"transformers.DogeConfig.max_position_embeddings",description:`<strong>max_position_embeddings</strong> (<code>int</code>, <em>optional</em>, defaults to 2048) &#x2014;
The maximum sequence length that this model might ever be used with.`,name:"max_position_embeddings"},{anchor:"transformers.DogeConfig.rope_theta",description:`<strong>rope_theta</strong> (<code>float</code>, <em>optional</em>, defaults to 10000.0) &#x2014;
The base period of the RoPE embeddings.`,name:"rope_theta"},{anchor:"transformers.DogeConfig.rope_scaling",description:`<strong>rope_scaling</strong> (<code>Dict</code>, <em>optional</em>) &#x2014;
Dictionary containing the scaling configuration for the RoPE embeddings.
NOTE: if you apply new rope type and you expect the model to work on longer <code>max_position_embeddings</code>, we recommend you to update this value accordingly.
Doge family of small models use <code>{ &apos;rope_type&apos;: &apos;dynamic&apos;, &apos;factor&apos;: 4.0, &apos;original_max_position_embeddings&apos;: 2048 }</code> as the default value.
Expected contents:
<code>rope_type</code> (<code>str</code>):
The sub-variant of RoPE to use. Can be one of [&#x2018;default&#x2019;, &#x2018;linear&#x2019;, &#x2018;dynamic&#x2019;, &#x2018;yarn&#x2019;, &#x2018;longrope&#x2019;, &#x2018;llama3&#x2019;], with &#x2018;default&#x2019; being the original RoPE implementation.
<code>factor</code> (<code>float</code>, <em>optional</em>):
Used with all rope types except &#x2018;default&#x2019;. The scaling factor to apply to the RoPE embeddings.
In most scaling types, a <code>factor</code> of x will enable the model to handle sequences of length x <em> original maximum pre-trained length.
<code>original_max_position_embeddings</code> (<code>int</code>, </em>optional<em>):
Used with &#x2018;dynamic&#x2019;, &#x2018;longrope&#x2019; and &#x2018;llama3&#x2019;.
The original max position embeddings used during pretraining.
<code>attention_factor</code> (<code>float</code>, </em>optional<em>):
Used with &#x2018;yarn&#x2019; and &#x2018;longrope&#x2019;. The scaling factor to be applied on the attention
computation.
If unspecified, it defaults to value recommended by the implementation, using the <code>factor</code> field to infer the suggested value.
<code>beta_fast</code> (<code>float</code>, </em>optional<em>):
Only used with &#x2018;yarn&#x2019;. Parameter to set the boundary for extrapolation (only) in the linear
ramp function. If unspecified, it defaults to 32.
<code>beta_slow</code> (<code>float</code>, </em>optional<em>):
Only used with &#x2018;yarn&#x2019;. Parameter to set the boundary for interpolation (only) in the linear
ramp function. If unspecified, it defaults to 1.
<code>short_factor</code> (<code>List[float]</code>, </em>optional<em>):
Only used with &#x2018;longrope&#x2019;. The scaling factor to be applied to short contexts (&lt;<code>original_max_position_embeddings</code>).
Must be a list of numbers with the same length as the hidden size divided by the number of attention heads divided by 2
<code>long_factor</code> (<code>List[float]</code>, </em>optional<em>):
Only used with &#x2018;longrope&#x2019;. The scaling factor to be applied to long contexts (&lt;<code>original_max_position_embeddings</code>).
Must be a list of numbers with the same length as the hidden size divided by the number of attention heads divided by 2
<code>low_freq_factor</code> (<code>float</code>, </em>optional<em>):
Only used with &#x2018;llama3&#x2019;. Scaling factor applied to low frequency components of the RoPE
<code>high_freq_factor</code> (<code>float</code>, </em>optional*):
Only used with &#x2018;llama3&#x2019;. Scaling factor applied to high frequency components of the RoPE`,name:"rope_scaling"},{anchor:"transformers.DogeConfig.num_attention_heads",description:`<strong>num_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 8) &#x2014;
Number of attention heads for each attention layer in the Transformer decoder.`,name:"num_attention_heads"},{anchor:"transformers.DogeConfig.num_key_value_heads",description:`<strong>num_key_value_heads</strong> (<code>int</code>, <em>optional</em>) &#x2014;
This is the number of key_value heads that should be used to implement Grouped Query Attention.
If <code>num_key_value_heads=num_attention_heads</code>, the model will use Multi Head Attention (MHA), if
<code>num_key_value_heads=1</code> the model will use Multi Query Attention (MQA) otherwise GQA is used.
When converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed by meanpooling all the original heads within that group.
For more details checkout <a href="https://huggingface.co/papers/2305.13245" rel="nofollow">this paper</a>.
If it is not specified, will default to <code>num_attention_heads</code>.`,name:"num_key_value_heads"},{anchor:"transformers.DogeConfig.attention_bias",description:`<strong>attention_bias</strong> (<code>bool</code>, defaults to <code>False</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether to use a bias in the query, key, value and output projection layers during self-attention.`,name:"attention_bias"},{anchor:"transformers.DogeConfig.attention_dropout",description:`<strong>attention_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The dropout ratio for the attention probabilities.`,name:"attention_dropout"},{anchor:"transformers.DogeConfig.mlp_bias",description:`<strong>mlp_bias</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether to use a bias in up_proj, down_proj and gate_proj layers in the MLP layers.`,name:"mlp_bias"},{anchor:"transformers.DogeConfig.sliding_window",description:`<strong>sliding_window</strong> (<code>int</code>, <em>optional</em>) &#x2014;
Sliding window attention window size. If not specified, will default to <code>None</code>.`,name:"sliding_window"},{anchor:"transformers.DogeConfig.keep_window_size",description:`<strong>keep_window_size</strong> (<code>int</code>, <em>optional</em>, defaults to 2048) &#x2014;
The window size of tokens that are not dynamically masked, and dynamic masking is only performed when the sequence length exceeds this value.`,name:"keep_window_size"},{anchor:"transformers.DogeConfig.is_moe",description:`<strong>is_moe</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether to use the Cross Domain Mixture of Experts, if <code>True</code>, the MoE will inherit the MLP to initialize.`,name:"is_moe"},{anchor:"transformers.DogeConfig.num_experts",description:`<strong>num_experts</strong> (<code>int</code>, <em>optional</em>, defaults to 16384) &#x2014;
Number of routed experts in the model. This is only used when <code>is_moe=True</code>.`,name:"num_experts"},{anchor:"transformers.DogeConfig.num_experts_per_tok",description:`<strong>num_experts_per_tok</strong> (<code>int</code>, <em>optional</em>, defaults to 64) &#x2014;
Number of selected experts to route per-token.`,name:"num_experts_per_tok"},{anchor:"transformers.DogeConfig.norm_topk_prob",description:`<strong>norm_topk_prob</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether to normalize the topk probabilities.`,name:"norm_topk_prob"},{anchor:"transformers.DogeConfig.output_router_logits",description:`<strong>output_router_logits</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not the router logits should be returned by the model. Enabling this will also
allow the model to output the auxiliary loss, including load balancing loss and router z-loss.`,name:"output_router_logits"},{anchor:"transformers.DogeConfig.router_aux_loss_coef",description:`<strong>router_aux_loss_coef</strong> (<code>float</code>, <em>optional</em>, defaults to 0.001) &#x2014;
The aux loss factor for the total loss.`,name:"router_aux_loss_coef"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/doge/configuration_doge.py#L27"}}),Z=new Wo({props:{anchor:"transformers.DogeConfig.example",$$slots:{default:[Go]},$$scope:{ctx:x}}}),te=new ge({props:{title:"DogeModel",local:"transformers.DogeModel",headingTag:"h2"}}),ne=new he({props:{name:"class transformers.DogeModel",anchor:"transformers.DogeModel",parameters:[{name:"config",val:": DogeConfig"}],parametersDescription:[{anchor:"transformers.DogeModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/doge#transformers.DogeConfig">DogeConfig</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/doge/modeling_doge.py#L516"}}),se=new he({props:{name:"forward",anchor:"transformers.DogeModel.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.DogeModel.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.DogeModel.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.DogeModel.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.DogeModel.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.DogeModel.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.DogeModel.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.DogeModel.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/doge/modeling_doge.py#L533",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <code>transformers.modeling_outputs.MoeModelOutputWithPast</code> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/doge#transformers.DogeConfig"
>DogeConfig</a>) and inputs.</p>
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
`}}),B=new wo({props:{$$slots:{default:[No]},$$scope:{ctx:x}}}),ae=new ge({props:{title:"DogeForCausalLM",local:"transformers.DogeForCausalLM",headingTag:"h2"}}),re=new he({props:{name:"class transformers.DogeForCausalLM",anchor:"transformers.DogeForCausalLM",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.DogeForCausalLM.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/doge#transformers.DogeForCausalLM">DogeForCausalLM</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/doge/modeling_doge.py#L705"}}),ie=new he({props:{name:"forward",anchor:"transformers.DogeForCausalLM.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[list[torch.FloatTensor]] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"},{name:"logits_to_keep",val:": typing.Union[int, torch.Tensor] = 0"},{name:"output_router_logits",val:": typing.Optional[bool] = None"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.DogeForCausalLM.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.DogeForCausalLM.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.DogeForCausalLM.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.DogeForCausalLM.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>list[torch.FloatTensor]</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.DogeForCausalLM.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.DogeForCausalLM.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.DogeForCausalLM.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.DogeForCausalLM.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"},{anchor:"transformers.DogeForCausalLM.forward.logits_to_keep",description:`<strong>logits_to_keep</strong> (<code>Union[int, torch.Tensor]</code>, defaults to <code>0</code>) &#x2014;
If an <code>int</code>, compute logits for the last <code>logits_to_keep</code> tokens. If <code>0</code>, calculate logits for all
<code>input_ids</code> (special case). Only last token logits are needed for generation, and calculating them only for that
token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
If a <code>torch.Tensor</code>, must be 1D corresponding to the indices to keep in the sequence length dimension.
This is useful when using packed tensor format (single dimension for batch and sequence length).`,name:"logits_to_keep"},{anchor:"transformers.DogeForCausalLM.forward.output_router_logits",description:`<strong>output_router_logits</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the logits of all the routers. They are useful for computing the router loss, and
should not be returned during inference.`,name:"output_router_logits"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/doge/modeling_doge.py#L722",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <code>transformers.modeling_outputs.MoeCausalLMOutputWithPast</code> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/doge#transformers.DogeConfig"
>DogeConfig</a>) and inputs.</p>
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
`}}),S=new wo({props:{$$slots:{default:[Ao]},$$scope:{ctx:x}}}),V=new Wo({props:{anchor:"transformers.DogeForCausalLM.forward.example",$$slots:{default:[Qo]},$$scope:{ctx:x}}}),de=new ge({props:{title:"DogeForSequenceClassification",local:"transformers.DogeForSequenceClassification",headingTag:"h2"}}),le=new he({props:{name:"class transformers.DogeForSequenceClassification",anchor:"transformers.DogeForSequenceClassification",parameters:[{name:"config",val:""}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/doge/modeling_doge.py#L808"}}),ce=new he({props:{name:"forward",anchor:"transformers.DogeForSequenceClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.DogeForSequenceClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.DogeForSequenceClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.DogeForSequenceClassification.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.DogeForSequenceClassification.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.DogeForSequenceClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.DogeForSequenceClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.DogeForSequenceClassification.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
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
`}}),P=new wo({props:{$$slots:{default:[Oo]},$$scope:{ctx:x}}}),pe=new Xo({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/doge.md"}}),{c(){n=l("meta"),_=s(),i=l("p"),y=s(),C=l("p"),C.innerHTML=T,L=s(),p(X.$$.fragment),Ie=s(),p(G.$$.fragment),Fe=s(),N=l("p"),N.innerHTML=vo,je=s(),E=l("img"),qe=s(),A=l("p"),A.innerHTML=ko,Le=s(),Q=l("p"),Q.innerHTML=xo,Ee=s(),p(O.$$.fragment),Re=s(),R=l("details"),fe=l("summary"),fe.textContent=Co,Ke=s(),p(Y.$$.fragment),We=s(),W=l("details"),_e=l("summary"),_e.textContent=$o,eo=s(),p(K.$$.fragment),Ze=s(),p(ee.$$.fragment),Be=s(),M=l("div"),p(oe.$$.fragment),oo=s(),be=l("p"),be.innerHTML=Do,to=s(),ye=l("p"),ye.innerHTML=zo,no=s(),p(Z.$$.fragment),Se=s(),p(te.$$.fragment),Ve=s(),w=l("div"),p(ne.$$.fragment),so=s(),Te=l("p"),Te.textContent=Jo,ao=s(),we=l("p"),we.innerHTML=Uo,ro=s(),ve=l("p"),ve.innerHTML=Io,io=s(),U=l("div"),p(se.$$.fragment),lo=s(),Me=l("p"),Me.innerHTML=Fo,co=s(),p(B.$$.fragment),Pe=s(),p(ae.$$.fragment),He=s(),v=l("div"),p(re.$$.fragment),po=s(),ke=l("p"),ke.textContent=jo,mo=s(),xe=l("p"),xe.innerHTML=qo,uo=s(),Ce=l("p"),Ce.innerHTML=Lo,ho=s(),$=l("div"),p(ie.$$.fragment),go=s(),$e=l("p"),$e.innerHTML=Eo,fo=s(),p(S.$$.fragment),_o=s(),p(V.$$.fragment),Xe=s(),p(de.$$.fragment),Ge=s(),j=l("div"),p(le.$$.fragment),bo=s(),I=l("div"),p(ce.$$.fragment),yo=s(),De=l("p"),De.innerHTML=Ro,To=s(),p(P.$$.fragment),Ne=s(),p(pe.$$.fragment),Ae=s(),Je=l("p"),this.h()},l(e){const o=Ho("svelte-u9bgzb",document.head);n=c(o,"META",{name:!0,content:!0}),o.forEach(t),_=a(e),i=c(e,"P",{}),J(i).forEach(t),y=a(e),C=c(e,"P",{"data-svelte-h":!0}),b(C)!=="svelte-fjevtc"&&(C.innerHTML=T),L=a(e),m(X.$$.fragment,e),Ie=a(e),m(G.$$.fragment,e),Fe=a(e),N=c(e,"P",{"data-svelte-h":!0}),b(N)!=="svelte-70c1ea"&&(N.innerHTML=vo),je=a(e),E=c(e,"IMG",{src:!0,alt:!0,width:!0}),qe=a(e),A=c(e,"P",{"data-svelte-h":!0}),b(A)!=="svelte-1j3qh00"&&(A.innerHTML=ko),Le=a(e),Q=c(e,"P",{"data-svelte-h":!0}),b(Q)!=="svelte-1q2vvyp"&&(Q.innerHTML=xo),Ee=a(e),m(O.$$.fragment,e),Re=a(e),R=c(e,"DETAILS",{});var me=J(R);fe=c(me,"SUMMARY",{"data-svelte-h":!0}),b(fe)!=="svelte-8n4289"&&(fe.textContent=Co),Ke=a(me),m(Y.$$.fragment,me),me.forEach(t),We=a(e),W=c(e,"DETAILS",{});var ue=J(W);_e=c(ue,"SUMMARY",{"data-svelte-h":!0}),b(_e)!=="svelte-2o6qsz"&&(_e.textContent=$o),eo=a(ue),m(K.$$.fragment,ue),ue.forEach(t),Ze=a(e),m(ee.$$.fragment,e),Be=a(e),M=c(e,"DIV",{class:!0});var z=J(M);m(oe.$$.fragment,z),oo=a(z),be=c(z,"P",{"data-svelte-h":!0}),b(be)!=="svelte-1ga27zj"&&(be.innerHTML=Do),to=a(z),ye=c(z,"P",{"data-svelte-h":!0}),b(ye)!=="svelte-1ek1ss9"&&(ye.innerHTML=zo),no=a(z),m(Z.$$.fragment,z),z.forEach(t),Se=a(e),m(te.$$.fragment,e),Ve=a(e),w=c(e,"DIV",{class:!0});var k=J(w);m(ne.$$.fragment,k),so=a(k),Te=c(k,"P",{"data-svelte-h":!0}),b(Te)!=="svelte-16cvzex"&&(Te.textContent=Jo),ao=a(k),we=c(k,"P",{"data-svelte-h":!0}),b(we)!=="svelte-q52n56"&&(we.innerHTML=Uo),ro=a(k),ve=c(k,"P",{"data-svelte-h":!0}),b(ve)!=="svelte-hswkmf"&&(ve.innerHTML=Io),io=a(k),U=c(k,"DIV",{class:!0});var q=J(U);m(se.$$.fragment,q),lo=a(q),Me=c(q,"P",{"data-svelte-h":!0}),b(Me)!=="svelte-rma81i"&&(Me.innerHTML=Fo),co=a(q),m(B.$$.fragment,q),q.forEach(t),k.forEach(t),Pe=a(e),m(ae.$$.fragment,e),He=a(e),v=c(e,"DIV",{class:!0});var F=J(v);m(re.$$.fragment,F),po=a(F),ke=c(F,"P",{"data-svelte-h":!0}),b(ke)!=="svelte-rfcov4"&&(ke.textContent=jo),mo=a(F),xe=c(F,"P",{"data-svelte-h":!0}),b(xe)!=="svelte-q52n56"&&(xe.innerHTML=qo),uo=a(F),Ce=c(F,"P",{"data-svelte-h":!0}),b(Ce)!=="svelte-hswkmf"&&(Ce.innerHTML=Lo),ho=a(F),$=c(F,"DIV",{class:!0});var H=J($);m(ie.$$.fragment,H),go=a(H),$e=c(H,"P",{"data-svelte-h":!0}),b($e)!=="svelte-1nxxaxi"&&($e.innerHTML=Eo),fo=a(H),m(S.$$.fragment,H),_o=a(H),m(V.$$.fragment,H),H.forEach(t),F.forEach(t),Xe=a(e),m(de.$$.fragment,e),Ge=a(e),j=c(e,"DIV",{class:!0});var Oe=J(j);m(le.$$.fragment,Oe),bo=a(Oe),I=c(Oe,"DIV",{class:!0});var ze=J(I);m(ce.$$.fragment,ze),yo=a(ze),De=c(ze,"P",{"data-svelte-h":!0}),b(De)!=="svelte-1sal4ui"&&(De.innerHTML=Ro),To=a(ze),m(P.$$.fragment,ze),ze.forEach(t),Oe.forEach(t),Ne=a(e),m(pe.$$.fragment,e),Ae=a(e),Je=c(e,"P",{}),J(Je).forEach(t),this.h()},h(){D(n,"name","hf:doc:metadata"),D(n,"content",Ko),Bo(E.src,Mo="https://huggingface.co/datasets/huggingface/documentation-images/resolve/refs%2Fpr%2F426/transformers/model_doc/doge_architecture.png")||D(E,"src",Mo),D(E,"alt","drawing"),D(E,"width","600"),D(M,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),D(U,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),D(w,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),D($,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),D(v,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),D(I,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),D(j,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,o){d(document.head,n),r(e,_,o),r(e,i,o),r(e,y,o),r(e,C,o),r(e,L,o),u(X,e,o),r(e,Ie,o),u(G,e,o),r(e,Fe,o),r(e,N,o),r(e,je,o),r(e,E,o),r(e,qe,o),r(e,A,o),r(e,Le,o),r(e,Q,o),r(e,Ee,o),u(O,e,o),r(e,Re,o),r(e,R,o),d(R,fe),d(R,Ke),u(Y,R,null),r(e,We,o),r(e,W,o),d(W,_e),d(W,eo),u(K,W,null),r(e,Ze,o),u(ee,e,o),r(e,Be,o),r(e,M,o),u(oe,M,null),d(M,oo),d(M,be),d(M,to),d(M,ye),d(M,no),u(Z,M,null),r(e,Se,o),u(te,e,o),r(e,Ve,o),r(e,w,o),u(ne,w,null),d(w,so),d(w,Te),d(w,ao),d(w,we),d(w,ro),d(w,ve),d(w,io),d(w,U),u(se,U,null),d(U,lo),d(U,Me),d(U,co),u(B,U,null),r(e,Pe,o),u(ae,e,o),r(e,He,o),r(e,v,o),u(re,v,null),d(v,po),d(v,ke),d(v,mo),d(v,xe),d(v,uo),d(v,Ce),d(v,ho),d(v,$),u(ie,$,null),d($,go),d($,$e),d($,fo),u(S,$,null),d($,_o),u(V,$,null),r(e,Xe,o),u(de,e,o),r(e,Ge,o),r(e,j,o),u(le,j,null),d(j,bo),d(j,I),u(ce,I,null),d(I,yo),d(I,De),d(I,To),u(P,I,null),r(e,Ne,o),u(pe,e,o),r(e,Ae,o),r(e,Je,o),Qe=!0},p(e,[o]){const me={};o&2&&(me.$$scope={dirty:o,ctx:e}),Z.$set(me);const ue={};o&2&&(ue.$$scope={dirty:o,ctx:e}),B.$set(ue);const z={};o&2&&(z.$$scope={dirty:o,ctx:e}),S.$set(z);const k={};o&2&&(k.$$scope={dirty:o,ctx:e}),V.$set(k);const q={};o&2&&(q.$$scope={dirty:o,ctx:e}),P.$set(q)},i(e){Qe||(h(X.$$.fragment,e),h(G.$$.fragment,e),h(O.$$.fragment,e),h(Y.$$.fragment,e),h(K.$$.fragment,e),h(ee.$$.fragment,e),h(oe.$$.fragment,e),h(Z.$$.fragment,e),h(te.$$.fragment,e),h(ne.$$.fragment,e),h(se.$$.fragment,e),h(B.$$.fragment,e),h(ae.$$.fragment,e),h(re.$$.fragment,e),h(ie.$$.fragment,e),h(S.$$.fragment,e),h(V.$$.fragment,e),h(de.$$.fragment,e),h(le.$$.fragment,e),h(ce.$$.fragment,e),h(P.$$.fragment,e),h(pe.$$.fragment,e),Qe=!0)},o(e){g(X.$$.fragment,e),g(G.$$.fragment,e),g(O.$$.fragment,e),g(Y.$$.fragment,e),g(K.$$.fragment,e),g(ee.$$.fragment,e),g(oe.$$.fragment,e),g(Z.$$.fragment,e),g(te.$$.fragment,e),g(ne.$$.fragment,e),g(se.$$.fragment,e),g(B.$$.fragment,e),g(ae.$$.fragment,e),g(re.$$.fragment,e),g(ie.$$.fragment,e),g(S.$$.fragment,e),g(V.$$.fragment,e),g(de.$$.fragment,e),g(le.$$.fragment,e),g(ce.$$.fragment,e),g(P.$$.fragment,e),g(pe.$$.fragment,e),Qe=!1},d(e){e&&(t(_),t(i),t(y),t(C),t(L),t(Ie),t(Fe),t(N),t(je),t(E),t(qe),t(A),t(Le),t(Q),t(Ee),t(Re),t(R),t(We),t(W),t(Ze),t(Be),t(M),t(Se),t(Ve),t(w),t(Pe),t(He),t(v),t(Xe),t(Ge),t(j),t(Ne),t(Ae),t(Je)),t(n),f(X,e),f(G,e),f(O,e),f(Y),f(K),f(ee,e),f(oe),f(Z),f(te,e),f(ne),f(se),f(B),f(ae,e),f(re),f(ie),f(S),f(V),f(de,e),f(le),f(ce),f(P),f(pe,e)}}}const Ko='{"title":"Doge","local":"doge","sections":[{"title":"Overview","local":"overview","sections":[],"depth":2},{"title":"Usage","local":"usage","sections":[],"depth":2},{"title":"DogeConfig","local":"transformers.DogeConfig","sections":[],"depth":2},{"title":"DogeModel","local":"transformers.DogeModel","sections":[],"depth":2},{"title":"DogeForCausalLM","local":"transformers.DogeForCausalLM","sections":[],"depth":2},{"title":"DogeForSequenceClassification","local":"transformers.DogeForSequenceClassification","sections":[],"depth":2}],"depth":1}';function et(x){return So(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class dt extends Vo{constructor(n){super(),Po(this,n,et,Yo,Zo,{})}}export{dt as component};
