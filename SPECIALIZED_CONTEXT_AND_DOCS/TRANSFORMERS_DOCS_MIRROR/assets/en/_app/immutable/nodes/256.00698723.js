import{s as bt,o as vt,n as Ze}from"../chunks/scheduler.18a86fab.js";import{S as Tt,i as kt,g as d,s,r as m,A as wt,h as c,f as n,c as a,j as ae,x as _,u as p,k as P,y as l,a as r,v as f,d as h,t as u,w as g}from"../chunks/index.98837b22.js";import{T as yt}from"../chunks/Tip.77304350.js";import{D as ye}from"../chunks/Docstring.a1ef7999.js";import{C as tt}from"../chunks/CodeBlock.8d0c2e8a.js";import{E as Mt}from"../chunks/ExampleCodeBlock.8c3ee1f9.js";import{H as re,E as Ct}from"../chunks/getInferenceSnippets.06c2775f.js";function Lt(x){let o,y;return o=new tt({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMExmbTJNb2RlbCUyQyUyMExmbTJDb25maWclMEElMEElMjMlMjBJbml0aWFsaXppbmclMjBhJTIwTEZNMiUyMG1vZGVsJTBBY29uZmlndXJhdGlvbiUyMCUzRCUyMExmbTJDb25maWcoKSUwQSUwQSUyMyUyMEluaXRpYWxpemluZyUyMGElMjBtb2RlbCUyMGZyb20lMjB0aGUlMjBMRk0yLTEuMkIlMjBzdHlsZSUyMGNvbmZpZ3VyYXRpb24lMEFtb2RlbCUyMCUzRCUyMExmbTJNb2RlbChjb25maWd1cmF0aW9uKSUwQSUwQSUyMyUyMEFjY2Vzc2luZyUyMHRoZSUyMG1vZGVsJTIwY29uZmlndXJhdGlvbiUwQWNvbmZpZ3VyYXRpb24lMjAlM0QlMjBtb2RlbC5jb25maWc=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> Lfm2Model, Lfm2Config

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a LFM2 model</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = Lfm2Config()

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a model from the LFM2-1.2B style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = Lfm2Model(configuration)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Accessing the model configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = model.config`,wrap:!1}}),{c(){m(o.$$.fragment)},l(i){p(o.$$.fragment,i)},m(i,M){f(o,i,M),y=!0},p:Ze,i(i){y||(h(o.$$.fragment,i),y=!0)},o(i){u(o.$$.fragment,i),y=!1},d(i){g(o,i)}}}function $t(x){let o,y=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){o=d("p"),o.innerHTML=y},l(i){o=c(i,"P",{"data-svelte-h":!0}),_(o)!=="svelte-fincs2"&&(o.innerHTML=y)},m(i,M){r(i,o,M)},p:Ze,d(i){i&&n(o)}}}function Ut(x){let o,y=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){o=d("p"),o.innerHTML=y},l(i){o=c(i,"P",{"data-svelte-h":!0}),_(o)!=="svelte-fincs2"&&(o.innerHTML=y)},m(i,M){r(i,o,M)},p:Ze,d(i){i&&n(o)}}}function xt(x){let o,y="Example:",i,M,L;return M=new tt({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBMZm0yRm9yQ2F1c2FsTE0lMEElMEFtb2RlbCUyMCUzRCUyMExmbTJGb3JDYXVzYWxMTS5mcm9tX3ByZXRyYWluZWQoJTIybWV0YS1sZm0yJTJGTGZtMi0yLTdiLWhmJTIyKSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMm1ldGEtbGZtMiUyRkxmbTItMi03Yi1oZiUyMiklMEElMEFwcm9tcHQlMjAlM0QlMjAlMjJIZXklMkMlMjBhcmUlMjB5b3UlMjBjb25zY2lvdXMlM0YlMjBDYW4lMjB5b3UlMjB0YWxrJTIwdG8lMjBtZSUzRiUyMiUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplcihwcm9tcHQlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQSUwQSUyMyUyMEdlbmVyYXRlJTBBZ2VuZXJhdGVfaWRzJTIwJTNEJTIwbW9kZWwuZ2VuZXJhdGUoaW5wdXRzLmlucHV0X2lkcyUyQyUyMG1heF9sZW5ndGglM0QzMCklMEF0b2tlbml6ZXIuYmF0Y2hfZGVjb2RlKGdlbmVyYXRlX2lkcyUyQyUyMHNraXBfc3BlY2lhbF90b2tlbnMlM0RUcnVlJTJDJTIwY2xlYW5fdXBfdG9rZW5pemF0aW9uX3NwYWNlcyUzREZhbHNlKSU1QjAlNUQ=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, Lfm2ForCausalLM

<span class="hljs-meta">&gt;&gt;&gt; </span>model = Lfm2ForCausalLM.from_pretrained(<span class="hljs-string">&quot;meta-lfm2/Lfm2-2-7b-hf&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;meta-lfm2/Lfm2-2-7b-hf&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>prompt = <span class="hljs-string">&quot;Hey, are you conscious? Can you talk to me?&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(prompt, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Generate</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>generate_ids = model.generate(inputs.input_ids, max_length=<span class="hljs-number">30</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer.batch_decode(generate_ids, skip_special_tokens=<span class="hljs-literal">True</span>, clean_up_tokenization_spaces=<span class="hljs-literal">False</span>)[<span class="hljs-number">0</span>]
<span class="hljs-string">&quot;Hey, are you conscious? Can you talk to me?\\nI&#x27;m not conscious, but I can talk to you.&quot;</span>`,wrap:!1}}),{c(){o=d("p"),o.textContent=y,i=s(),m(M.$$.fragment)},l(b){o=c(b,"P",{"data-svelte-h":!0}),_(o)!=="svelte-11lpom8"&&(o.textContent=y),i=a(b),p(M.$$.fragment,b)},m(b,F){r(b,o,F),r(b,i,F),f(M,b,F),L=!0},p:Ze,i(b){L||(h(M.$$.fragment,b),L=!0)},o(b){u(M.$$.fragment,b),L=!1},d(b){b&&(n(o),n(i)),g(M,b)}}}function zt(x){let o,y,i,M,L,b="<em>This model was released on 2025-07-10 and added to Hugging Face Transformers on 2025-07-10.</em>",F,I,nt='<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/>',Me,W,be,B,ve,H,ot='<a href="https://www.liquid.ai/blog/liquid-foundation-models-v2-our-second-series-of-generative-ai-models" rel="nofollow">LFM2</a> represents a new generation of Liquid Foundation Models developed by <a href="https://liquid.ai/" rel="nofollow">Liquid AI</a>, specifically designed for edge AI and on-device deployment.',Te,G,st="The models are available in three sizes (350M, 700M, and 1.2B parameters) and are engineered to run efficiently on CPU, GPU, and NPU hardware, making them particularly well-suited for applications requiring low latency, offline operation, and privacy.",ke,Q,we,A,at="The architecture consists of 16 blocks total: 10 double-gated short-range convolution blocks and 6 blocks of grouped query attention. This design stems from the concept of dynamical systems, where linear operations are modulated by input-dependent gates, allowing for “liquid” dynamics that can adapt in real-time. The short convolutions are particularly optimized for embedded SoC CPUs, making them ideal for devices that require fast, local inference without relying on cloud connectivity.",Ce,X,rt="The key architectural innovation of LFM2 lies in its systematic approach to balancing quality, latency, and memory efficiency through our STAR neural architecture search engine. Using STAR, Liquid AI optimized the models for real-world performance on embedded hardware, measuring actual peak memory usage and inference speed on Qualcomm Snapdragon processors. This results in models that achieve 2x faster decode and prefill performance compared to similar-sized models, while maintaining superior benchmark performance across knowledge, mathematics, instruction following, and multilingual tasks.",Le,V,$e,S,it="The following example shows how to generate an answer using the <code>AutoModelForCausalLM</code> class.",Ue,N,xe,D,ze,k,O,Pe,ie,lt=`This is the configuration class to store the configuration of a <a href="/docs/transformers/v4.56.2/en/model_doc/lfm2#transformers.Lfm2Model">Lfm2Model</a>. It is used to instantiate a LFM2
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the LFM2-1.2B model.
e.g. <a href="https://huggingface.co/LiquidAI/LFM2-1.2B" rel="nofollow">LiquidAI/LFM2-1.2B</a>`,We,le,dt=`Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> and can be used to control the model outputs. Read the
documentation from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> for more information.`,Be,q,je,Y,Fe,v,K,He,de,ct="The bare Lfm2 Model outputting raw hidden-states without any specific head on top.",Ge,ce,mt=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Qe,me,pt=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Ae,z,ee,Xe,pe,ft='The <a href="/docs/transformers/v4.56.2/en/model_doc/lfm2#transformers.Lfm2Model">Lfm2Model</a> forward method, overrides the <code>__call__</code> special method.',Ve,E,Ie,te,qe,T,ne,Se,fe,ht="The Lfm2 Model for causal language modeling.",Ne,he,ut=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,De,ue,gt=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Oe,$,oe,Ye,ge,_t='The <a href="/docs/transformers/v4.56.2/en/model_doc/lfm2#transformers.Lfm2ForCausalLM">Lfm2ForCausalLM</a> forward method, overrides the <code>__call__</code> special method.',Ke,R,et,J,Ee,se,Re,_e,Je;return W=new re({props:{title:"LFM2",local:"lfm2",headingTag:"h1"}}),B=new re({props:{title:"Overview",local:"overview",headingTag:"h2"}}),Q=new re({props:{title:"Architecture",local:"architecture",headingTag:"h2"}}),V=new re({props:{title:"Example",local:"example",headingTag:"h2"}}),N=new tt({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Nb2RlbEZvckNhdXNhbExNJTJDJTIwQXV0b1Rva2VuaXplciUwQSUwQSUyMyUyMExvYWQlMjBtb2RlbCUyMGFuZCUyMHRva2VuaXplciUwQW1vZGVsX2lkJTIwJTNEJTIwJTIyTGlxdWlkQUklMkZMRk0yLTEuMkIlMjIlMEFtb2RlbCUyMCUzRCUyMEF1dG9Nb2RlbEZvckNhdXNhbExNLmZyb21fcHJldHJhaW5lZCglMEElMjAlMjAlMjAlMjBtb2RlbF9pZCUyQyUwQSUyMCUyMCUyMCUyMGRldmljZV9tYXAlM0QlMjJhdXRvJTIyJTJDJTBBJTIwJTIwJTIwJTIwZHR5cGUlM0QlMjJiZmxvYXQxNiUyMiUyQyUwQSklMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZChtb2RlbF9pZCklMEElMEElMjMlMjBHZW5lcmF0ZSUyMGFuc3dlciUwQXByb21wdCUyMCUzRCUyMCUyMldoYXQlMjBpcyUyMEMuJTIwZWxlZ2FucyUzRiUyMiUwQWlucHV0X2lkcyUyMCUzRCUyMHRva2VuaXplci5hcHBseV9jaGF0X3RlbXBsYXRlKCUwQSUyMCUyMCUyMCUyMCU1QiU3QiUyMnJvbGUlMjIlM0ElMjAlMjJ1c2VyJTIyJTJDJTIwJTIyY29udGVudCUyMiUzQSUyMHByb21wdCU3RCU1RCUyQyUwQSUyMCUyMCUyMCUyMGFkZF9nZW5lcmF0aW9uX3Byb21wdCUzRFRydWUlMkMlMEElMjAlMjAlMjAlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyJTJDJTBBJTIwJTIwJTIwJTIwdG9rZW5pemUlM0RUcnVlJTJDJTBBKSUwQSUwQW91dHB1dCUyMCUzRCUyMG1vZGVsLmdlbmVyYXRlKCUwQSUyMCUyMCUyMCUyMGlucHV0X2lkcyUyQyUwQSUyMCUyMCUyMCUyMGRvX3NhbXBsZSUzRFRydWUlMkMlMEElMjAlMjAlMjAlMjB0ZW1wZXJhdHVyZSUzRDAuMyUyQyUwQSUyMCUyMCUyMCUyMG1pbl9wJTNEMC4xNSUyQyUwQSUyMCUyMCUyMCUyMHJlcGV0aXRpb25fcGVuYWx0eSUzRDEuMDUlMkMlMEElMjAlMjAlMjAlMjBtYXhfbmV3X3Rva2VucyUzRDUxMiUyQyUwQSklMEElMEFwcmludCh0b2tlbml6ZXIuZGVjb2RlKG91dHB1dCU1QjAlNUQlMkMlMjBza2lwX3NwZWNpYWxfdG9rZW5zJTNERmFsc2UpKQ==",highlighted:`<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoModelForCausalLM, AutoTokenizer

<span class="hljs-comment"># Load model and tokenizer</span>
model_id = <span class="hljs-string">&quot;LiquidAI/LFM2-1.2B&quot;</span>
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map=<span class="hljs-string">&quot;auto&quot;</span>,
    dtype=<span class="hljs-string">&quot;bfloat16&quot;</span>,
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

<span class="hljs-comment"># Generate answer</span>
prompt = <span class="hljs-string">&quot;What is C. elegans?&quot;</span>
input_ids = tokenizer.apply_chat_template(
    [{<span class="hljs-string">&quot;role&quot;</span>: <span class="hljs-string">&quot;user&quot;</span>, <span class="hljs-string">&quot;content&quot;</span>: prompt}],
    add_generation_prompt=<span class="hljs-literal">True</span>,
    return_tensors=<span class="hljs-string">&quot;pt&quot;</span>,
    tokenize=<span class="hljs-literal">True</span>,
)

output = model.generate(
    input_ids,
    do_sample=<span class="hljs-literal">True</span>,
    temperature=<span class="hljs-number">0.3</span>,
    min_p=<span class="hljs-number">0.15</span>,
    repetition_penalty=<span class="hljs-number">1.05</span>,
    max_new_tokens=<span class="hljs-number">512</span>,
)

<span class="hljs-built_in">print</span>(tokenizer.decode(output[<span class="hljs-number">0</span>], skip_special_tokens=<span class="hljs-literal">False</span>))`,wrap:!1}}),D=new re({props:{title:"Lfm2Config",local:"transformers.Lfm2Config",headingTag:"h2"}}),O=new ye({props:{name:"class transformers.Lfm2Config",anchor:"transformers.Lfm2Config",parameters:[{name:"vocab_size",val:": int = 65536"},{name:"hidden_size",val:": int = 2560"},{name:"intermediate_size",val:": int = 12288"},{name:"num_hidden_layers",val:": int = 32"},{name:"num_attention_heads",val:": int = 32"},{name:"num_key_value_heads",val:": int = 8"},{name:"max_position_embeddings",val:": int = 128000"},{name:"initializer_range",val:": float = 0.02"},{name:"norm_eps",val:": float = 1e-05"},{name:"use_cache",val:": bool = True"},{name:"pad_token_id",val:": int = 0"},{name:"bos_token_id",val:": int = 1"},{name:"eos_token_id",val:": int = 2"},{name:"tie_word_embeddings",val:": bool = True"},{name:"rope_theta",val:": float = 1000000.0"},{name:"conv_bias",val:": bool = False"},{name:"conv_L_cache",val:": int = 3"},{name:"block_multiple_of",val:": int = 256"},{name:"block_ffn_dim_multiplier",val:": float = 1.0"},{name:"block_auto_adjust_ff_dim",val:": bool = True"},{name:"full_attn_idxs",val:": typing.Optional[list[int]] = None"},{name:"layer_types",val:": typing.Optional[list[str]] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.Lfm2Config.vocab_size",description:`<strong>vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 65536) &#x2014;
Vocabulary size of the LLaMA model. Defines the number of different tokens that can be represented by the
<code>inputs_ids</code> passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/lfm2#transformers.Lfm2Model">Lfm2Model</a>`,name:"vocab_size"},{anchor:"transformers.Lfm2Config.hidden_size",description:`<strong>hidden_size</strong> (<code>int</code>, <em>optional</em>, defaults to 2560) &#x2014;
Dimension of the hidden representations.`,name:"hidden_size"},{anchor:"transformers.Lfm2Config.intermediate_size",description:`<strong>intermediate_size</strong> (<code>int</code>, <em>optional</em>, defaults to 12288) &#x2014;
Dimension of the MLP representations.`,name:"intermediate_size"},{anchor:"transformers.Lfm2Config.num_hidden_layers",description:`<strong>num_hidden_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 32) &#x2014;
Number of hidden layers in the Transformer decoder.`,name:"num_hidden_layers"},{anchor:"transformers.Lfm2Config.num_attention_heads",description:`<strong>num_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 32) &#x2014;
Number of attention heads for each attention layer in the Transformer decoder.`,name:"num_attention_heads"},{anchor:"transformers.Lfm2Config.num_key_value_heads",description:`<strong>num_key_value_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 8) &#x2014;
This is the number of key_value heads that should be used to implement Grouped Query Attention. If
<code>num_key_value_heads=num_attention_heads</code>, the model will use Multi Head Attention (MHA), if
<code>num_key_value_heads=1</code> the model will use Multi Query Attention (MQA) otherwise GQA is used. When
converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
by meanpooling all the original heads within that group. For more details, check out <a href="https://huggingface.co/papers/2305.13245" rel="nofollow">this
paper</a>. If it is not specified, will default to
<code>num_attention_heads</code>.`,name:"num_key_value_heads"},{anchor:"transformers.Lfm2Config.max_position_embeddings",description:`<strong>max_position_embeddings</strong> (<code>int</code>, <em>optional</em>, defaults to 128000) &#x2014;
The maximum sequence length that this model might ever be used with. Lfm2 1 supports up to 2048 tokens,
Lfm2 2 up to 4096, CodeLfm2 up to 16384.`,name:"max_position_embeddings"},{anchor:"transformers.Lfm2Config.initializer_range",description:`<strong>initializer_range</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.`,name:"initializer_range"},{anchor:"transformers.Lfm2Config.norm_eps",description:`<strong>norm_eps</strong> (<code>float</code>, <em>optional</em>, defaults to 1e-05) &#x2014;
The epsilon used by the rms normalization layers.`,name:"norm_eps"},{anchor:"transformers.Lfm2Config.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not the model should return the last key/values attentions (not used by all models). Only
relevant if <code>config.is_decoder=True</code>.`,name:"use_cache"},{anchor:"transformers.Lfm2Config.pad_token_id",description:`<strong>pad_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 0) &#x2014;
Padding token id.`,name:"pad_token_id"},{anchor:"transformers.Lfm2Config.bos_token_id",description:`<strong>bos_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 1) &#x2014;
Beginning of stream token id.`,name:"bos_token_id"},{anchor:"transformers.Lfm2Config.eos_token_id",description:`<strong>eos_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 2) &#x2014;
End of stream token id.`,name:"eos_token_id"},{anchor:"transformers.Lfm2Config.tie_word_embeddings",description:`<strong>tie_word_embeddings</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether to tie weight embeddings`,name:"tie_word_embeddings"},{anchor:"transformers.Lfm2Config.rope_theta",description:`<strong>rope_theta</strong> (<code>float</code>, <em>optional</em>, defaults to 1000000.0) &#x2014;
The base period of the RoPE embeddings.`,name:"rope_theta"},{anchor:"transformers.Lfm2Config.conv_bias",description:`<strong>conv_bias</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether to use bias in the conv layers.`,name:"conv_bias"},{anchor:"transformers.Lfm2Config.conv_L_cache",description:`<strong>conv_L_cache</strong> (<code>int</code>, <em>optional</em>, defaults to 3) &#x2014;
L_cache dim in the conv layers.`,name:"conv_L_cache"},{anchor:"transformers.Lfm2Config.block_multiple_of",description:`<strong>block_multiple_of</strong> (<code>int</code>, <em>optional</em>, defaults to 256) &#x2014;
Multiple for the <code>intermediate_size</code>.`,name:"block_multiple_of"},{anchor:"transformers.Lfm2Config.block_ffn_dim_multiplier",description:`<strong>block_ffn_dim_multiplier</strong> (<code>float</code>, <em>optional</em>, defaults to 1.0) &#x2014;
Multiplier for the <code>intermediate_size</code>.`,name:"block_ffn_dim_multiplier"},{anchor:"transformers.Lfm2Config.block_auto_adjust_ff_dim",description:`<strong>block_auto_adjust_ff_dim</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether to adjust the dim of the <code>intermediate_size</code>.`,name:"block_auto_adjust_ff_dim"},{anchor:"transformers.Lfm2Config.full_attn_idxs",description:`<strong>full_attn_idxs</strong> (<code>Optional</code>, <em>optional</em>) &#x2014;
Index of the layers which use attention.`,name:"full_attn_idxs"},{anchor:"transformers.Lfm2Config.layer_types",description:`<strong>layer_types</strong> (<code>Optional</code>, <em>optional</em>) &#x2014;
Type of each layers.`,name:"layer_types"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/lfm2/configuration_lfm2.py#L19"}}),q=new Mt({props:{anchor:"transformers.Lfm2Config.example",$$slots:{default:[Lt]},$$scope:{ctx:x}}}),Y=new re({props:{title:"Lfm2Model",local:"transformers.Lfm2Model",headingTag:"h2"}}),K=new ye({props:{name:"class transformers.Lfm2Model",anchor:"transformers.Lfm2Model",parameters:[{name:"config",val:": Lfm2Config"}],parametersDescription:[{anchor:"transformers.Lfm2Model.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/lfm2#transformers.Lfm2Config">Lfm2Config</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/lfm2/modeling_lfm2.py#L594"}}),ee=new ye({props:{name:"forward",anchor:"transformers.Lfm2Model.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.models.lfm2.modeling_lfm2.Lfm2HybridConvCache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.Lfm2Model.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.Lfm2Model.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.Lfm2Model.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.Lfm2Model.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~models.lfm2.modeling_lfm2.Lfm2HybridConvCache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.Lfm2Model.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.Lfm2Model.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.Lfm2Model.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/lfm2/modeling_lfm2.py#L612",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast"
>transformers.modeling_outputs.BaseModelOutputWithPast</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/lfm2#transformers.Lfm2Config"
>Lfm2Config</a>) and inputs.</p>
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
`}}),E=new yt({props:{$$slots:{default:[$t]},$$scope:{ctx:x}}}),te=new re({props:{title:"Lfm2ForCausalLM",local:"transformers.Lfm2ForCausalLM",headingTag:"h2"}}),ne=new ye({props:{name:"class transformers.Lfm2ForCausalLM",anchor:"transformers.Lfm2ForCausalLM",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.Lfm2ForCausalLM.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/lfm2#transformers.Lfm2ForCausalLM">Lfm2ForCausalLM</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/lfm2/modeling_lfm2.py#L679"}}),oe=new ye({props:{name:"forward",anchor:"transformers.Lfm2ForCausalLM.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"},{name:"logits_to_keep",val:": typing.Union[int, torch.Tensor] = 0"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.Lfm2ForCausalLM.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.Lfm2ForCausalLM.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.Lfm2ForCausalLM.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.Lfm2ForCausalLM.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.Lfm2ForCausalLM.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.Lfm2ForCausalLM.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.Lfm2ForCausalLM.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.Lfm2ForCausalLM.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"},{anchor:"transformers.Lfm2ForCausalLM.forward.logits_to_keep",description:`<strong>logits_to_keep</strong> (<code>Union[int, torch.Tensor]</code>, defaults to <code>0</code>) &#x2014;
If an <code>int</code>, compute logits for the last <code>logits_to_keep</code> tokens. If <code>0</code>, calculate logits for all
<code>input_ids</code> (special case). Only last token logits are needed for generation, and calculating them only for that
token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
If a <code>torch.Tensor</code>, must be 1D corresponding to the indices to keep in the sequence length dimension.
This is useful when using packed tensor format (single dimension for batch and sequence length).`,name:"logits_to_keep"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/lfm2/modeling_lfm2.py#L693",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast"
>transformers.modeling_outputs.CausalLMOutputWithPast</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/lfm2#transformers.Lfm2Config"
>Lfm2Config</a>) and inputs.</p>
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
`}}),R=new yt({props:{$$slots:{default:[Ut]},$$scope:{ctx:x}}}),J=new Mt({props:{anchor:"transformers.Lfm2ForCausalLM.forward.example",$$slots:{default:[xt]},$$scope:{ctx:x}}}),se=new Ct({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/lfm2.md"}}),{c(){o=d("meta"),y=s(),i=d("p"),M=s(),L=d("p"),L.innerHTML=b,F=s(),I=d("div"),I.innerHTML=nt,Me=s(),m(W.$$.fragment),be=s(),m(B.$$.fragment),ve=s(),H=d("p"),H.innerHTML=ot,Te=s(),G=d("p"),G.textContent=st,ke=s(),m(Q.$$.fragment),we=s(),A=d("p"),A.textContent=at,Ce=s(),X=d("p"),X.textContent=rt,Le=s(),m(V.$$.fragment),$e=s(),S=d("p"),S.innerHTML=it,Ue=s(),m(N.$$.fragment),xe=s(),m(D.$$.fragment),ze=s(),k=d("div"),m(O.$$.fragment),Pe=s(),ie=d("p"),ie.innerHTML=lt,We=s(),le=d("p"),le.innerHTML=dt,Be=s(),m(q.$$.fragment),je=s(),m(Y.$$.fragment),Fe=s(),v=d("div"),m(K.$$.fragment),He=s(),de=d("p"),de.textContent=ct,Ge=s(),ce=d("p"),ce.innerHTML=mt,Qe=s(),me=d("p"),me.innerHTML=pt,Ae=s(),z=d("div"),m(ee.$$.fragment),Xe=s(),pe=d("p"),pe.innerHTML=ft,Ve=s(),m(E.$$.fragment),Ie=s(),m(te.$$.fragment),qe=s(),T=d("div"),m(ne.$$.fragment),Se=s(),fe=d("p"),fe.textContent=ht,Ne=s(),he=d("p"),he.innerHTML=ut,De=s(),ue=d("p"),ue.innerHTML=gt,Oe=s(),$=d("div"),m(oe.$$.fragment),Ye=s(),ge=d("p"),ge.innerHTML=_t,Ke=s(),m(R.$$.fragment),et=s(),m(J.$$.fragment),Ee=s(),m(se.$$.fragment),Re=s(),_e=d("p"),this.h()},l(e){const t=wt("svelte-u9bgzb",document.head);o=c(t,"META",{name:!0,content:!0}),t.forEach(n),y=a(e),i=c(e,"P",{}),ae(i).forEach(n),M=a(e),L=c(e,"P",{"data-svelte-h":!0}),_(L)!=="svelte-1ccd9v8"&&(L.innerHTML=b),F=a(e),I=c(e,"DIV",{class:!0,"data-svelte-h":!0}),_(I)!=="svelte-13t8s2t"&&(I.innerHTML=nt),Me=a(e),p(W.$$.fragment,e),be=a(e),p(B.$$.fragment,e),ve=a(e),H=c(e,"P",{"data-svelte-h":!0}),_(H)!=="svelte-1e78zdo"&&(H.innerHTML=ot),Te=a(e),G=c(e,"P",{"data-svelte-h":!0}),_(G)!=="svelte-w0iwc2"&&(G.textContent=st),ke=a(e),p(Q.$$.fragment,e),we=a(e),A=c(e,"P",{"data-svelte-h":!0}),_(A)!=="svelte-1yfdl74"&&(A.textContent=at),Ce=a(e),X=c(e,"P",{"data-svelte-h":!0}),_(X)!=="svelte-13r527b"&&(X.textContent=rt),Le=a(e),p(V.$$.fragment,e),$e=a(e),S=c(e,"P",{"data-svelte-h":!0}),_(S)!=="svelte-1aa4nvu"&&(S.innerHTML=it),Ue=a(e),p(N.$$.fragment,e),xe=a(e),p(D.$$.fragment,e),ze=a(e),k=c(e,"DIV",{class:!0});var U=ae(k);p(O.$$.fragment,U),Pe=a(U),ie=c(U,"P",{"data-svelte-h":!0}),_(ie)!=="svelte-1y0dzjm"&&(ie.innerHTML=lt),We=a(U),le=c(U,"P",{"data-svelte-h":!0}),_(le)!=="svelte-1ek1ss9"&&(le.innerHTML=dt),Be=a(U),p(q.$$.fragment,U),U.forEach(n),je=a(e),p(Y.$$.fragment,e),Fe=a(e),v=c(e,"DIV",{class:!0});var w=ae(v);p(K.$$.fragment,w),He=a(w),de=c(w,"P",{"data-svelte-h":!0}),_(de)!=="svelte-rq0s0d"&&(de.textContent=ct),Ge=a(w),ce=c(w,"P",{"data-svelte-h":!0}),_(ce)!=="svelte-q52n56"&&(ce.innerHTML=mt),Qe=a(w),me=c(w,"P",{"data-svelte-h":!0}),_(me)!=="svelte-hswkmf"&&(me.innerHTML=pt),Ae=a(w),z=c(w,"DIV",{class:!0});var j=ae(z);p(ee.$$.fragment,j),Xe=a(j),pe=c(j,"P",{"data-svelte-h":!0}),_(pe)!=="svelte-klxqna"&&(pe.innerHTML=ft),Ve=a(j),p(E.$$.fragment,j),j.forEach(n),w.forEach(n),Ie=a(e),p(te.$$.fragment,e),qe=a(e),T=c(e,"DIV",{class:!0});var C=ae(T);p(ne.$$.fragment,C),Se=a(C),fe=c(C,"P",{"data-svelte-h":!0}),_(fe)!=="svelte-7wrelc"&&(fe.textContent=ht),Ne=a(C),he=c(C,"P",{"data-svelte-h":!0}),_(he)!=="svelte-q52n56"&&(he.innerHTML=ut),De=a(C),ue=c(C,"P",{"data-svelte-h":!0}),_(ue)!=="svelte-hswkmf"&&(ue.innerHTML=gt),Oe=a(C),$=c(C,"DIV",{class:!0});var Z=ae($);p(oe.$$.fragment,Z),Ye=a(Z),ge=c(Z,"P",{"data-svelte-h":!0}),_(ge)!=="svelte-m5e7ie"&&(ge.innerHTML=_t),Ke=a(Z),p(R.$$.fragment,Z),et=a(Z),p(J.$$.fragment,Z),Z.forEach(n),C.forEach(n),Ee=a(e),p(se.$$.fragment,e),Re=a(e),_e=c(e,"P",{}),ae(_e).forEach(n),this.h()},h(){P(o,"name","hf:doc:metadata"),P(o,"content",jt),P(I,"class","flex flex-wrap space-x-1"),P(k,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),P(z,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),P(v,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),P($,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),P(T,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,t){l(document.head,o),r(e,y,t),r(e,i,t),r(e,M,t),r(e,L,t),r(e,F,t),r(e,I,t),r(e,Me,t),f(W,e,t),r(e,be,t),f(B,e,t),r(e,ve,t),r(e,H,t),r(e,Te,t),r(e,G,t),r(e,ke,t),f(Q,e,t),r(e,we,t),r(e,A,t),r(e,Ce,t),r(e,X,t),r(e,Le,t),f(V,e,t),r(e,$e,t),r(e,S,t),r(e,Ue,t),f(N,e,t),r(e,xe,t),f(D,e,t),r(e,ze,t),r(e,k,t),f(O,k,null),l(k,Pe),l(k,ie),l(k,We),l(k,le),l(k,Be),f(q,k,null),r(e,je,t),f(Y,e,t),r(e,Fe,t),r(e,v,t),f(K,v,null),l(v,He),l(v,de),l(v,Ge),l(v,ce),l(v,Qe),l(v,me),l(v,Ae),l(v,z),f(ee,z,null),l(z,Xe),l(z,pe),l(z,Ve),f(E,z,null),r(e,Ie,t),f(te,e,t),r(e,qe,t),r(e,T,t),f(ne,T,null),l(T,Se),l(T,fe),l(T,Ne),l(T,he),l(T,De),l(T,ue),l(T,Oe),l(T,$),f(oe,$,null),l($,Ye),l($,ge),l($,Ke),f(R,$,null),l($,et),f(J,$,null),r(e,Ee,t),f(se,e,t),r(e,Re,t),r(e,_e,t),Je=!0},p(e,[t]){const U={};t&2&&(U.$$scope={dirty:t,ctx:e}),q.$set(U);const w={};t&2&&(w.$$scope={dirty:t,ctx:e}),E.$set(w);const j={};t&2&&(j.$$scope={dirty:t,ctx:e}),R.$set(j);const C={};t&2&&(C.$$scope={dirty:t,ctx:e}),J.$set(C)},i(e){Je||(h(W.$$.fragment,e),h(B.$$.fragment,e),h(Q.$$.fragment,e),h(V.$$.fragment,e),h(N.$$.fragment,e),h(D.$$.fragment,e),h(O.$$.fragment,e),h(q.$$.fragment,e),h(Y.$$.fragment,e),h(K.$$.fragment,e),h(ee.$$.fragment,e),h(E.$$.fragment,e),h(te.$$.fragment,e),h(ne.$$.fragment,e),h(oe.$$.fragment,e),h(R.$$.fragment,e),h(J.$$.fragment,e),h(se.$$.fragment,e),Je=!0)},o(e){u(W.$$.fragment,e),u(B.$$.fragment,e),u(Q.$$.fragment,e),u(V.$$.fragment,e),u(N.$$.fragment,e),u(D.$$.fragment,e),u(O.$$.fragment,e),u(q.$$.fragment,e),u(Y.$$.fragment,e),u(K.$$.fragment,e),u(ee.$$.fragment,e),u(E.$$.fragment,e),u(te.$$.fragment,e),u(ne.$$.fragment,e),u(oe.$$.fragment,e),u(R.$$.fragment,e),u(J.$$.fragment,e),u(se.$$.fragment,e),Je=!1},d(e){e&&(n(y),n(i),n(M),n(L),n(F),n(I),n(Me),n(be),n(ve),n(H),n(Te),n(G),n(ke),n(we),n(A),n(Ce),n(X),n(Le),n($e),n(S),n(Ue),n(xe),n(ze),n(k),n(je),n(Fe),n(v),n(Ie),n(qe),n(T),n(Ee),n(Re),n(_e)),n(o),g(W,e),g(B,e),g(Q,e),g(V,e),g(N,e),g(D,e),g(O),g(q),g(Y,e),g(K),g(ee),g(E),g(te,e),g(ne),g(oe),g(R),g(J),g(se,e)}}}const jt='{"title":"LFM2","local":"lfm2","sections":[{"title":"Overview","local":"overview","sections":[],"depth":2},{"title":"Architecture","local":"architecture","sections":[],"depth":2},{"title":"Example","local":"example","sections":[],"depth":2},{"title":"Lfm2Config","local":"transformers.Lfm2Config","sections":[],"depth":2},{"title":"Lfm2Model","local":"transformers.Lfm2Model","sections":[],"depth":2},{"title":"Lfm2ForCausalLM","local":"transformers.Lfm2ForCausalLM","sections":[],"depth":2}],"depth":1}';function Ft(x){return vt(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class Wt extends Tt{constructor(o){super(),kt(this,o,Ft,zt,bt,{})}}export{Wt as component};
