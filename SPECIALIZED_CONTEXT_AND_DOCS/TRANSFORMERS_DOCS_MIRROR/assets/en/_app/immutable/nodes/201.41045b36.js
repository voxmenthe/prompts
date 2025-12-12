import{s as Qt,o as Yt,n as qe}from"../chunks/scheduler.18a86fab.js";import{S as Kt,i as eo,g as d,s,r as m,A as to,h as c,f as o,c as a,j as L,x as _,u as p,k as x,y as l,a as r,v as h,d as u,t as f,w as g}from"../chunks/index.98837b22.js";import{T as lt}from"../chunks/Tip.77304350.js";import{D as B}from"../chunks/Docstring.a1ef7999.js";import{C as Ft}from"../chunks/CodeBlock.8d0c2e8a.js";import{E as At}from"../chunks/ExampleCodeBlock.8c3ee1f9.js";import{H as D,E as oo}from"../chunks/getInferenceSnippets.06c2775f.js";function no(w){let n,b;return n=new Ft({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEdsbU1vZGVsJTJDJTIwR2xtQ29uZmlnJTBBJTIzJTIwSW5pdGlhbGl6aW5nJTIwYSUyMEdsbSUyMGdsbS00LTliLWNoYXQlMjBzdHlsZSUyMGNvbmZpZ3VyYXRpb24lMEFjb25maWd1cmF0aW9uJTIwJTNEJTIwR2xtQ29uZmlnKCklMEElMjMlMjBJbml0aWFsaXppbmclMjBhJTIwbW9kZWwlMjBmcm9tJTIwdGhlJTIwZ2xtLTQtOWItY2hhdCUyMHN0eWxlJTIwY29uZmlndXJhdGlvbiUwQW1vZGVsJTIwJTNEJTIwR2xtTW9kZWwoY29uZmlndXJhdGlvbiklMEElMjMlMjBBY2Nlc3NpbmclMjB0aGUlMjBtb2RlbCUyMGNvbmZpZ3VyYXRpb24lMEFjb25maWd1cmF0aW9uJTIwJTNEJTIwbW9kZWwuY29uZmln",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> GlmModel, GlmConfig
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a Glm glm-4-9b-chat style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = GlmConfig()
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a model from the glm-4-9b-chat style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = GlmModel(configuration)
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Accessing the model configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = model.config`,wrap:!1}}),{c(){m(n.$$.fragment)},l(i){p(n.$$.fragment,i)},m(i,y){h(n,i,y),b=!0},p:qe,i(i){b||(u(n.$$.fragment,i),b=!0)},o(i){f(n.$$.fragment,i),b=!1},d(i){g(n,i)}}}function so(w){let n,b=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){n=d("p"),n.innerHTML=b},l(i){n=c(i,"P",{"data-svelte-h":!0}),_(n)!=="svelte-fincs2"&&(n.innerHTML=b)},m(i,y){r(i,n,y)},p:qe,d(i){i&&o(n)}}}function ao(w){let n,b=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){n=d("p"),n.innerHTML=b},l(i){n=c(i,"P",{"data-svelte-h":!0}),_(n)!=="svelte-fincs2"&&(n.innerHTML=b)},m(i,y){r(i,n,y)},p:qe,d(i){i&&o(n)}}}function ro(w){let n,b="Example:",i,y,C;return y=new Ft({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBHbG1Gb3JDYXVzYWxMTSUwQSUwQW1vZGVsJTIwJTNEJTIwR2xtRm9yQ2F1c2FsTE0uZnJvbV9wcmV0cmFpbmVkKCUyMm1ldGEtZ2xtJTJGR2xtLTItN2ItaGYlMjIpJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIybWV0YS1nbG0lMkZHbG0tMi03Yi1oZiUyMiklMEElMEFwcm9tcHQlMjAlM0QlMjAlMjJIZXklMkMlMjBhcmUlMjB5b3UlMjBjb25zY2lvdXMlM0YlMjBDYW4lMjB5b3UlMjB0YWxrJTIwdG8lMjBtZSUzRiUyMiUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplcihwcm9tcHQlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQSUwQSUyMyUyMEdlbmVyYXRlJTBBZ2VuZXJhdGVfaWRzJTIwJTNEJTIwbW9kZWwuZ2VuZXJhdGUoaW5wdXRzLmlucHV0X2lkcyUyQyUyMG1heF9sZW5ndGglM0QzMCklMEF0b2tlbml6ZXIuYmF0Y2hfZGVjb2RlKGdlbmVyYXRlX2lkcyUyQyUyMHNraXBfc3BlY2lhbF90b2tlbnMlM0RUcnVlJTJDJTIwY2xlYW5fdXBfdG9rZW5pemF0aW9uX3NwYWNlcyUzREZhbHNlKSU1QjAlNUQ=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, GlmForCausalLM

<span class="hljs-meta">&gt;&gt;&gt; </span>model = GlmForCausalLM.from_pretrained(<span class="hljs-string">&quot;meta-glm/Glm-2-7b-hf&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;meta-glm/Glm-2-7b-hf&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>prompt = <span class="hljs-string">&quot;Hey, are you conscious? Can you talk to me?&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(prompt, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Generate</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>generate_ids = model.generate(inputs.input_ids, max_length=<span class="hljs-number">30</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer.batch_decode(generate_ids, skip_special_tokens=<span class="hljs-literal">True</span>, clean_up_tokenization_spaces=<span class="hljs-literal">False</span>)[<span class="hljs-number">0</span>]
<span class="hljs-string">&quot;Hey, are you conscious? Can you talk to me?\\nI&#x27;m not conscious, but I can talk to you.&quot;</span>`,wrap:!1}}),{c(){n=d("p"),n.textContent=b,i=s(),m(y.$$.fragment)},l(v){n=c(v,"P",{"data-svelte-h":!0}),_(n)!=="svelte-11lpom8"&&(n.textContent=b),i=a(v),p(y.$$.fragment,v)},m(v,H){r(v,n,H),r(v,i,H),h(y,v,H),C=!0},p:qe,i(v){C||(u(y.$$.fragment,v),C=!0)},o(v){f(y.$$.fragment,v),C=!1},d(v){v&&(o(n),o(i)),g(y,v)}}}function io(w){let n,b=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){n=d("p"),n.innerHTML=b},l(i){n=c(i,"P",{"data-svelte-h":!0}),_(n)!=="svelte-fincs2"&&(n.innerHTML=b)},m(i,y){r(i,n,y)},p:qe,d(i){i&&o(n)}}}function lo(w){let n,b=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){n=d("p"),n.innerHTML=b},l(i){n=c(i,"P",{"data-svelte-h":!0}),_(n)!=="svelte-fincs2"&&(n.innerHTML=b)},m(i,y){r(i,n,y)},p:qe,d(i){i&&o(n)}}}function co(w){let n,b,i,y,C,v="<em>This model was released on 2024-06-18 and added to Hugging Face Transformers on 2024-10-18.</em>",H,O,We,P,Lt='<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/> <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat"/> <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/> <img alt="Tensor parallelism" src="https://img.shields.io/badge/Tensor%20parallelism-06b6d4?style=flat&amp;logoColor=white"/>',Ze,A,Be,Q,Ut=`The GLM Model was proposed
in <a href="https://huggingface.co/papers/2406.12793" rel="nofollow">ChatGLM: A Family of Large Language Models from GLM-130B to GLM-4 All Tools</a>
by GLM Team, THUDM &amp; ZhipuAI.`,He,Y,It="The abstract from the paper is the following:",Pe,K,jt=`<em>We introduce ChatGLM, an evolving family of large language models that we have been developing over time. This report
primarily focuses on the GLM-4 language series, which includes GLM-4, GLM-4-Air, and GLM-4-9B. They represent our most
capable models that are trained with all the insights and lessons gained from the preceding three generations of
ChatGLM. To date, the GLM-4 models are pre-trained on ten trillions of tokens mostly in Chinese and English, along with
a small set of corpus from 24 languages, and aligned primarily for Chinese and English usage. The high-quality alignment
is achieved via a multi-stage post-training process, which involves supervised fine-tuning and learning from human
feedback. Evaluations show that GLM-4 1) closely rivals or outperforms GPT-4 in terms of general metrics such as MMLU,
GSM8K, MATH, BBH, GPQA, and HumanEval, 2) gets close to GPT-4-Turbo in instruction following as measured by IFEval, 3)
matches GPT-4 Turbo (128K) and Claude 3 for long context tasks, and 4) outperforms GPT-4 in Chinese alignments as
measured by AlignBench. The GLM-4 All Tools model is further aligned to understand user intent and autonomously decide
when and which tool(s) to use—including web browser, Python interpreter, text-to-image model, and user-defined
functions—to effectively complete complex tasks. In practical applications, it matches and even surpasses GPT-4 All
Tools in tasks like accessing online information via web browsing and solving math problems using Python interpreter.
Over the course, we have open-sourced a series of models, including ChatGLM-6B (three generations), GLM-4-9B (128K, 1M),
GLM-4V-9B, WebGLM, and CodeGeeX, attracting over 10 million downloads on Hugging face in the year 2023 alone.</em>`,Ve,ee,qt="Tips:",Re,te,Jt=`<li>This model was contributed by <a href="https://huggingface.co/THUDM" rel="nofollow">THUDM</a>. The most recent code can be
found <a href="https://github.com/thudm/GLM-4" rel="nofollow">here</a>.</li>`,Ne,oe,Xe,ne,Wt='<code>GLM-4</code> can be found on the <a href="https://huggingface.co/collections/THUDM/glm-4-665fcf188c414b03c2f7e3b7" rel="nofollow">Huggingface Hub</a>',Se,se,Zt="In the following, we demonstrate how to use <code>glm-4-9b-chat</code> for the inference. Note that we have used the ChatML format for dialog, in this demo we show how to leverage <code>apply_chat_template</code> for this purpose.",Ee,ae,De,re,Oe,z,ie,dt,ke,Bt=`This is the configuration class to store the configuration of a <a href="/docs/transformers/v4.56.2/en/model_doc/glm#transformers.GlmModel">GlmModel</a>. It is used to instantiate an Glm
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the Glm-4-9b-chat.
e.g. <a href="https://huggingface.co/THUDM/glm-4-9b-chat" rel="nofollow">THUDM/glm-4-9b-chat</a>
Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> and can be used to control the model outputs. Read the
documentation from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> for more information.`,ct,V,Ae,le,Qe,T,de,mt,we,Ht="The bare Glm Model outputting raw hidden-states without any specific head on top.",pt,Me,Pt=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,ht,$e,Vt=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,ut,U,ce,ft,Ce,Rt='The <a href="/docs/transformers/v4.56.2/en/model_doc/glm#transformers.GlmModel">GlmModel</a> forward method, overrides the <code>__call__</code> special method.',gt,R,Ye,me,Ke,k,pe,_t,Ge,Nt="The Glm Model for causal language modeling.",bt,xe,Xt=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,yt,ze,St=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,vt,G,he,Tt,Fe,Et='The <a href="/docs/transformers/v4.56.2/en/model_doc/glm#transformers.GlmForCausalLM">GlmForCausalLM</a> forward method, overrides the <code>__call__</code> special method.',kt,N,wt,X,et,ue,tt,q,fe,Mt,I,ge,$t,Le,Dt="The <code>GenericForSequenceClassification</code> forward method, overrides the <code>__call__</code> special method.",Ct,S,ot,_e,nt,J,be,Gt,j,ye,xt,Ue,Ot="The <code>GenericForTokenClassification</code> forward method, overrides the <code>__call__</code> special method.",zt,E,st,ve,at,Je,rt;return O=new D({props:{title:"GLM",local:"glm",headingTag:"h1"}}),A=new D({props:{title:"Overview",local:"overview",headingTag:"h2"}}),oe=new D({props:{title:"Usage tips",local:"usage-tips",headingTag:"h2"}}),ae=new Ft({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Nb2RlbEZvckNhdXNhbExNJTJDJTIwQXV0b1Rva2VuaXplciUyQyUyMGluZmVyX2RldmljZSUwQWRldmljZSUyMCUzRCUyMGluZmVyX2RldmljZSgpJTIwJTIzJTIwdGhlJTIwZGV2aWNlJTIwdG8lMjBsb2FkJTIwdGhlJTIwbW9kZWwlMjBvbnRvJTBBJTBBbW9kZWwlMjAlM0QlMjBBdXRvTW9kZWxGb3JDYXVzYWxMTS5mcm9tX3ByZXRyYWluZWQoJTIyVEhVRE0lMkZnbG0tNC05Yi1jaGF0JTIyJTJDJTIwZGV2aWNlX21hcCUzRCUyMmF1dG8lMjIlMkMlMjB0cnVzdF9yZW1vdGVfY29kZSUzRFRydWUpJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIyVEhVRE0lMkZnbG0tNC05Yi1jaGF0JTIyKSUwQSUwQXByb21wdCUyMCUzRCUyMCUyMkdpdmUlMjBtZSUyMGElMjBzaG9ydCUyMGludHJvZHVjdGlvbiUyMHRvJTIwbGFyZ2UlMjBsYW5ndWFnZSUyMG1vZGVsLiUyMiUwQSUwQW1lc3NhZ2VzJTIwJTNEJTIwJTVCJTdCJTIycm9sZSUyMiUzQSUyMCUyMnVzZXIlMjIlMkMlMjAlMjJjb250ZW50JTIyJTNBJTIwcHJvbXB0JTdEJTVEJTBBJTBBdGV4dCUyMCUzRCUyMHRva2VuaXplci5hcHBseV9jaGF0X3RlbXBsYXRlKG1lc3NhZ2VzJTJDJTIwdG9rZW5pemUlM0RGYWxzZSUyQyUyMGFkZF9nZW5lcmF0aW9uX3Byb21wdCUzRFRydWUpJTBBJTBBbW9kZWxfaW5wdXRzJTIwJTNEJTIwdG9rZW5pemVyKCU1QnRleHQlNUQlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKS50byhtb2RlbC5kZXZpY2UpJTBBJTBBZ2VuZXJhdGVkX2lkcyUyMCUzRCUyMG1vZGVsLmdlbmVyYXRlKG1vZGVsX2lucHV0cy5pbnB1dF9pZHMlMkMlMjBtYXhfbmV3X3Rva2VucyUzRDUxMiUyQyUyMGRvX3NhbXBsZSUzRFRydWUpJTBBJTBBZ2VuZXJhdGVkX2lkcyUyMCUzRCUyMCU1Qm91dHB1dF9pZHMlNUJsZW4oaW5wdXRfaWRzKSUzQSU1RCUyMGZvciUyMGlucHV0X2lkcyUyQyUyMG91dHB1dF9pZHMlMjBpbiUyMHppcChtb2RlbF9pbnB1dHMuaW5wdXRfaWRzJTJDJTIwZ2VuZXJhdGVkX2lkcyklNUQlMEElMEFyZXNwb25zZSUyMCUzRCUyMHRva2VuaXplci5iYXRjaF9kZWNvZGUoZ2VuZXJhdGVkX2lkcyUyQyUyMHNraXBfc3BlY2lhbF90b2tlbnMlM0RUcnVlKSU1QjAlNUQ=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoModelForCausalLM, AutoTokenizer, infer_device
<span class="hljs-meta">&gt;&gt;&gt; </span>device = infer_device() <span class="hljs-comment"># the device to load the model onto</span>

<span class="hljs-meta">&gt;&gt;&gt; </span>model = AutoModelForCausalLM.from_pretrained(<span class="hljs-string">&quot;THUDM/glm-4-9b-chat&quot;</span>, device_map=<span class="hljs-string">&quot;auto&quot;</span>, trust_remote_code=<span class="hljs-literal">True</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;THUDM/glm-4-9b-chat&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>prompt = <span class="hljs-string">&quot;Give me a short introduction to large language model.&quot;</span>

<span class="hljs-meta">&gt;&gt;&gt; </span>messages = [{<span class="hljs-string">&quot;role&quot;</span>: <span class="hljs-string">&quot;user&quot;</span>, <span class="hljs-string">&quot;content&quot;</span>: prompt}]

<span class="hljs-meta">&gt;&gt;&gt; </span>text = tokenizer.apply_chat_template(messages, tokenize=<span class="hljs-literal">False</span>, add_generation_prompt=<span class="hljs-literal">True</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>model_inputs = tokenizer([text], return_tensors=<span class="hljs-string">&quot;pt&quot;</span>).to(model.device)

<span class="hljs-meta">&gt;&gt;&gt; </span>generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=<span class="hljs-number">512</span>, do_sample=<span class="hljs-literal">True</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>generated_ids = [output_ids[<span class="hljs-built_in">len</span>(input_ids):] <span class="hljs-keyword">for</span> input_ids, output_ids <span class="hljs-keyword">in</span> <span class="hljs-built_in">zip</span>(model_inputs.input_ids, generated_ids)]

<span class="hljs-meta">&gt;&gt;&gt; </span>response = tokenizer.batch_decode(generated_ids, skip_special_tokens=<span class="hljs-literal">True</span>)[<span class="hljs-number">0</span>]`,wrap:!1}}),re=new D({props:{title:"GlmConfig",local:"transformers.GlmConfig",headingTag:"h2"}}),ie=new B({props:{name:"class transformers.GlmConfig",anchor:"transformers.GlmConfig",parameters:[{name:"vocab_size",val:" = 151552"},{name:"hidden_size",val:" = 4096"},{name:"intermediate_size",val:" = 13696"},{name:"num_hidden_layers",val:" = 40"},{name:"num_attention_heads",val:" = 32"},{name:"num_key_value_heads",val:" = 2"},{name:"partial_rotary_factor",val:" = 0.5"},{name:"head_dim",val:" = 128"},{name:"hidden_act",val:" = 'silu'"},{name:"attention_dropout",val:" = 0.0"},{name:"max_position_embeddings",val:" = 131072"},{name:"initializer_range",val:" = 0.02"},{name:"rms_norm_eps",val:" = 1.5625e-07"},{name:"use_cache",val:" = True"},{name:"tie_word_embeddings",val:" = False"},{name:"rope_theta",val:" = 10000.0"},{name:"pad_token_id",val:" = 151329"},{name:"eos_token_id",val:" = [151329, 151336, 151338]"},{name:"bos_token_id",val:" = None"},{name:"attention_bias",val:" = True"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.GlmConfig.vocab_size",description:`<strong>vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 151552) &#x2014;
Vocabulary size of the Glm model. Defines the number of different tokens that can be represented by the
<code>inputs_ids</code> passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/glm#transformers.GlmModel">GlmModel</a>`,name:"vocab_size"},{anchor:"transformers.GlmConfig.hidden_size",description:`<strong>hidden_size</strong> (<code>int</code>, <em>optional</em>, defaults to 4096) &#x2014;
Dimension of the hidden representations.`,name:"hidden_size"},{anchor:"transformers.GlmConfig.intermediate_size",description:`<strong>intermediate_size</strong> (<code>int</code>, <em>optional</em>, defaults to 13696) &#x2014;
Dimension of the MLP representations.`,name:"intermediate_size"},{anchor:"transformers.GlmConfig.num_hidden_layers",description:`<strong>num_hidden_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 40) &#x2014;
Number of hidden layers in the Transformer decoder.`,name:"num_hidden_layers"},{anchor:"transformers.GlmConfig.num_attention_heads",description:`<strong>num_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 32) &#x2014;
Number of attention heads for each attention layer in the Transformer decoder.`,name:"num_attention_heads"},{anchor:"transformers.GlmConfig.num_key_value_heads",description:`<strong>num_key_value_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 2) &#x2014;
This is the number of key_value heads that should be used to implement Grouped Query Attention. If
<code>num_key_value_heads=num_attention_heads</code>, the model will use Multi Head Attention (MHA), if
<code>num_key_value_heads=1</code> the model will use Multi Query Attention (MQA) otherwise GQA is used. When
converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
by meanpooling all the original heads within that group. For more details, check out <a href="https://huggingface.co/papers/2305.13245" rel="nofollow">this
paper</a>. If it is not specified, will default to
<code>num_attention_heads</code>.`,name:"num_key_value_heads"},{anchor:"transformers.GlmConfig.partial_rotary_factor",description:"<strong>partial_rotary_factor</strong> (<code>float</code>, <em>optional</em>, defaults to 0.5) &#x2014; The factor of the partial rotary position.",name:"partial_rotary_factor"},{anchor:"transformers.GlmConfig.head_dim",description:`<strong>head_dim</strong> (<code>int</code>, <em>optional</em>, defaults to 128) &#x2014;
The attention head dimension.`,name:"head_dim"},{anchor:"transformers.GlmConfig.hidden_act",description:`<strong>hidden_act</strong> (<code>str</code> or <code>function</code>, <em>optional</em>, defaults to <code>&quot;silu&quot;</code>) &#x2014;
The legacy activation function. It is overwritten by the <code>hidden_activation</code>.`,name:"hidden_act"},{anchor:"transformers.GlmConfig.attention_dropout",description:`<strong>attention_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The dropout ratio for the attention probabilities.`,name:"attention_dropout"},{anchor:"transformers.GlmConfig.max_position_embeddings",description:`<strong>max_position_embeddings</strong> (<code>int</code>, <em>optional</em>, defaults to 131072) &#x2014;
The maximum sequence length that this model might ever be used with.`,name:"max_position_embeddings"},{anchor:"transformers.GlmConfig.initializer_range",description:`<strong>initializer_range</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.`,name:"initializer_range"},{anchor:"transformers.GlmConfig.rms_norm_eps",description:`<strong>rms_norm_eps</strong> (<code>float</code>, <em>optional</em>, defaults to 1.5625e-07) &#x2014;
The epsilon used by the rms normalization layers.`,name:"rms_norm_eps"},{anchor:"transformers.GlmConfig.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not the model should return the last key/values attentions (not used by all models). Only
relevant if <code>config.is_decoder=True</code>.`,name:"use_cache"},{anchor:"transformers.GlmConfig.tie_word_embeddings",description:`<strong>tie_word_embeddings</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether to tie weight embeddings`,name:"tie_word_embeddings"},{anchor:"transformers.GlmConfig.rope_theta",description:`<strong>rope_theta</strong> (<code>float</code>, <em>optional</em>, defaults to 10000.0) &#x2014;
The base period of the RoPE embeddings.`,name:"rope_theta"},{anchor:"transformers.GlmConfig.pad_token_id",description:`<strong>pad_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 151329) &#x2014;
Padding token id.`,name:"pad_token_id"},{anchor:"transformers.GlmConfig.eos_token_id",description:`<strong>eos_token_id</strong> (<code>int</code> | <code>list</code>, <em>optional</em>, defaults to <code>[151329, 151336, 151338]</code>) &#x2014;
End of stream token id.`,name:"eos_token_id"},{anchor:"transformers.GlmConfig.bos_token_id",description:`<strong>bos_token_id</strong> (<code>int</code>, <em>optional</em>) &#x2014;
Beginning of stream token id.`,name:"bos_token_id"},{anchor:"transformers.GlmConfig.attention_bias",description:`<strong>attention_bias</strong> (<code>bool</code>, defaults to <code>False</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether to use a bias in the query, key, value and output projection layers during self-attention.`,name:"attention_bias"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/glm/configuration_glm.py#L20"}}),V=new At({props:{anchor:"transformers.GlmConfig.example",$$slots:{default:[no]},$$scope:{ctx:w}}}),le=new D({props:{title:"GlmModel",local:"transformers.GlmModel",headingTag:"h2"}}),de=new B({props:{name:"class transformers.GlmModel",anchor:"transformers.GlmModel",parameters:[{name:"config",val:": GlmConfig"}],parametersDescription:[{anchor:"transformers.GlmModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/glm#transformers.GlmConfig">GlmConfig</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/glm/modeling_glm.py#L344"}}),ce=new B({props:{name:"forward",anchor:"transformers.GlmModel.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.GlmModel.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.GlmModel.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.GlmModel.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.GlmModel.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.GlmModel.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.GlmModel.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"},{anchor:"transformers.GlmModel.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/glm/modeling_glm.py#L361",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast"
>transformers.modeling_outputs.BaseModelOutputWithPast</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/glm#transformers.GlmConfig"
>GlmConfig</a>) and inputs.</p>
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
`}}),R=new lt({props:{$$slots:{default:[so]},$$scope:{ctx:w}}}),me=new D({props:{title:"GlmForCausalLM",local:"transformers.GlmForCausalLM",headingTag:"h2"}}),pe=new B({props:{name:"class transformers.GlmForCausalLM",anchor:"transformers.GlmForCausalLM",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.GlmForCausalLM.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/glm#transformers.GlmForCausalLM">GlmForCausalLM</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/glm/modeling_glm.py#L423"}}),he=new B({props:{name:"forward",anchor:"transformers.GlmForCausalLM.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"},{name:"logits_to_keep",val:": typing.Union[int, torch.Tensor] = 0"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.GlmForCausalLM.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.GlmForCausalLM.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.GlmForCausalLM.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.GlmForCausalLM.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.GlmForCausalLM.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.GlmForCausalLM.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.GlmForCausalLM.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.GlmForCausalLM.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"},{anchor:"transformers.GlmForCausalLM.forward.logits_to_keep",description:`<strong>logits_to_keep</strong> (<code>Union[int, torch.Tensor]</code>, defaults to <code>0</code>) &#x2014;
If an <code>int</code>, compute logits for the last <code>logits_to_keep</code> tokens. If <code>0</code>, calculate logits for all
<code>input_ids</code> (special case). Only last token logits are needed for generation, and calculating them only for that
token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
If a <code>torch.Tensor</code>, must be 1D corresponding to the indices to keep in the sequence length dimension.
This is useful when using packed tensor format (single dimension for batch and sequence length).`,name:"logits_to_keep"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/glm/modeling_glm.py#L437",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast"
>transformers.modeling_outputs.CausalLMOutputWithPast</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/glm#transformers.GlmConfig"
>GlmConfig</a>) and inputs.</p>
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
`}}),N=new lt({props:{$$slots:{default:[ao]},$$scope:{ctx:w}}}),X=new At({props:{anchor:"transformers.GlmForCausalLM.forward.example",$$slots:{default:[ro]},$$scope:{ctx:w}}}),ue=new D({props:{title:"GlmForSequenceClassification",local:"transformers.GlmForSequenceClassification",headingTag:"h2"}}),fe=new B({props:{name:"class transformers.GlmForSequenceClassification",anchor:"transformers.GlmForSequenceClassification",parameters:[{name:"config",val:""}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/glm/modeling_glm.py#L498"}}),ge=new B({props:{name:"forward",anchor:"transformers.GlmForSequenceClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.GlmForSequenceClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.GlmForSequenceClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.GlmForSequenceClassification.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.GlmForSequenceClassification.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.GlmForSequenceClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.GlmForSequenceClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.GlmForSequenceClassification.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
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
`}}),S=new lt({props:{$$slots:{default:[io]},$$scope:{ctx:w}}}),_e=new D({props:{title:"GlmForTokenClassification",local:"transformers.GlmForTokenClassification",headingTag:"h2"}}),be=new B({props:{name:"class transformers.GlmForTokenClassification",anchor:"transformers.GlmForTokenClassification",parameters:[{name:"config",val:""}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/glm/modeling_glm.py#L502"}}),ye=new B({props:{name:"forward",anchor:"transformers.GlmForTokenClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.GlmForTokenClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.GlmForTokenClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.GlmForTokenClassification.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.GlmForTokenClassification.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.GlmForTokenClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.GlmForTokenClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.GlmForTokenClassification.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
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
`}}),E=new lt({props:{$$slots:{default:[lo]},$$scope:{ctx:w}}}),ve=new oo({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/glm.md"}}),{c(){n=d("meta"),b=s(),i=d("p"),y=s(),C=d("p"),C.innerHTML=v,H=s(),m(O.$$.fragment),We=s(),P=d("div"),P.innerHTML=Lt,Ze=s(),m(A.$$.fragment),Be=s(),Q=d("p"),Q.innerHTML=Ut,He=s(),Y=d("p"),Y.textContent=It,Pe=s(),K=d("p"),K.innerHTML=jt,Ve=s(),ee=d("p"),ee.textContent=qt,Re=s(),te=d("ul"),te.innerHTML=Jt,Ne=s(),m(oe.$$.fragment),Xe=s(),ne=d("p"),ne.innerHTML=Wt,Se=s(),se=d("p"),se.innerHTML=Zt,Ee=s(),m(ae.$$.fragment),De=s(),m(re.$$.fragment),Oe=s(),z=d("div"),m(ie.$$.fragment),dt=s(),ke=d("p"),ke.innerHTML=Bt,ct=s(),m(V.$$.fragment),Ae=s(),m(le.$$.fragment),Qe=s(),T=d("div"),m(de.$$.fragment),mt=s(),we=d("p"),we.textContent=Ht,pt=s(),Me=d("p"),Me.innerHTML=Pt,ht=s(),$e=d("p"),$e.innerHTML=Vt,ut=s(),U=d("div"),m(ce.$$.fragment),ft=s(),Ce=d("p"),Ce.innerHTML=Rt,gt=s(),m(R.$$.fragment),Ye=s(),m(me.$$.fragment),Ke=s(),k=d("div"),m(pe.$$.fragment),_t=s(),Ge=d("p"),Ge.textContent=Nt,bt=s(),xe=d("p"),xe.innerHTML=Xt,yt=s(),ze=d("p"),ze.innerHTML=St,vt=s(),G=d("div"),m(he.$$.fragment),Tt=s(),Fe=d("p"),Fe.innerHTML=Et,kt=s(),m(N.$$.fragment),wt=s(),m(X.$$.fragment),et=s(),m(ue.$$.fragment),tt=s(),q=d("div"),m(fe.$$.fragment),Mt=s(),I=d("div"),m(ge.$$.fragment),$t=s(),Le=d("p"),Le.innerHTML=Dt,Ct=s(),m(S.$$.fragment),ot=s(),m(_e.$$.fragment),nt=s(),J=d("div"),m(be.$$.fragment),Gt=s(),j=d("div"),m(ye.$$.fragment),xt=s(),Ue=d("p"),Ue.innerHTML=Ot,zt=s(),m(E.$$.fragment),st=s(),m(ve.$$.fragment),at=s(),Je=d("p"),this.h()},l(e){const t=to("svelte-u9bgzb",document.head);n=c(t,"META",{name:!0,content:!0}),t.forEach(o),b=a(e),i=c(e,"P",{}),L(i).forEach(o),y=a(e),C=c(e,"P",{"data-svelte-h":!0}),_(C)!=="svelte-wrtlsb"&&(C.innerHTML=v),H=a(e),p(O.$$.fragment,e),We=a(e),P=c(e,"DIV",{class:!0,"data-svelte-h":!0}),_(P)!=="svelte-is43db"&&(P.innerHTML=Lt),Ze=a(e),p(A.$$.fragment,e),Be=a(e),Q=c(e,"P",{"data-svelte-h":!0}),_(Q)!=="svelte-126337l"&&(Q.innerHTML=Ut),He=a(e),Y=c(e,"P",{"data-svelte-h":!0}),_(Y)!=="svelte-vfdo9a"&&(Y.textContent=It),Pe=a(e),K=c(e,"P",{"data-svelte-h":!0}),_(K)!=="svelte-57yk8e"&&(K.innerHTML=jt),Ve=a(e),ee=c(e,"P",{"data-svelte-h":!0}),_(ee)!=="svelte-axv494"&&(ee.textContent=qt),Re=a(e),te=c(e,"UL",{"data-svelte-h":!0}),_(te)!=="svelte-1mv2pve"&&(te.innerHTML=Jt),Ne=a(e),p(oe.$$.fragment,e),Xe=a(e),ne=c(e,"P",{"data-svelte-h":!0}),_(ne)!=="svelte-8dtdop"&&(ne.innerHTML=Wt),Se=a(e),se=c(e,"P",{"data-svelte-h":!0}),_(se)!=="svelte-grk520"&&(se.innerHTML=Zt),Ee=a(e),p(ae.$$.fragment,e),De=a(e),p(re.$$.fragment,e),Oe=a(e),z=c(e,"DIV",{class:!0});var W=L(z);p(ie.$$.fragment,W),dt=a(W),ke=c(W,"P",{"data-svelte-h":!0}),_(ke)!=="svelte-8eccwp"&&(ke.innerHTML=Bt),ct=a(W),p(V.$$.fragment,W),W.forEach(o),Ae=a(e),p(le.$$.fragment,e),Qe=a(e),T=c(e,"DIV",{class:!0});var M=L(T);p(de.$$.fragment,M),mt=a(M),we=c(M,"P",{"data-svelte-h":!0}),_(we)!=="svelte-6qqyu8"&&(we.textContent=Ht),pt=a(M),Me=c(M,"P",{"data-svelte-h":!0}),_(Me)!=="svelte-q52n56"&&(Me.innerHTML=Pt),ht=a(M),$e=c(M,"P",{"data-svelte-h":!0}),_($e)!=="svelte-hswkmf"&&($e.innerHTML=Vt),ut=a(M),U=c(M,"DIV",{class:!0});var Z=L(U);p(ce.$$.fragment,Z),ft=a(Z),Ce=c(Z,"P",{"data-svelte-h":!0}),_(Ce)!=="svelte-1sfelzb"&&(Ce.innerHTML=Rt),gt=a(Z),p(R.$$.fragment,Z),Z.forEach(o),M.forEach(o),Ye=a(e),p(me.$$.fragment,e),Ke=a(e),k=c(e,"DIV",{class:!0});var $=L(k);p(pe.$$.fragment,$),_t=a($),Ge=c($,"P",{"data-svelte-h":!0}),_(Ge)!=="svelte-xn4rm9"&&(Ge.textContent=Nt),bt=a($),xe=c($,"P",{"data-svelte-h":!0}),_(xe)!=="svelte-q52n56"&&(xe.innerHTML=Xt),yt=a($),ze=c($,"P",{"data-svelte-h":!0}),_(ze)!=="svelte-hswkmf"&&(ze.innerHTML=St),vt=a($),G=c($,"DIV",{class:!0});var F=L(G);p(he.$$.fragment,F),Tt=a(F),Fe=c(F,"P",{"data-svelte-h":!0}),_(Fe)!=="svelte-9s3snv"&&(Fe.innerHTML=Et),kt=a(F),p(N.$$.fragment,F),wt=a(F),p(X.$$.fragment,F),F.forEach(o),$.forEach(o),et=a(e),p(ue.$$.fragment,e),tt=a(e),q=c(e,"DIV",{class:!0});var Te=L(q);p(fe.$$.fragment,Te),Mt=a(Te),I=c(Te,"DIV",{class:!0});var Ie=L(I);p(ge.$$.fragment,Ie),$t=a(Ie),Le=c(Ie,"P",{"data-svelte-h":!0}),_(Le)!=="svelte-1sal4ui"&&(Le.innerHTML=Dt),Ct=a(Ie),p(S.$$.fragment,Ie),Ie.forEach(o),Te.forEach(o),ot=a(e),p(_e.$$.fragment,e),nt=a(e),J=c(e,"DIV",{class:!0});var it=L(J);p(be.$$.fragment,it),Gt=a(it),j=c(it,"DIV",{class:!0});var je=L(j);p(ye.$$.fragment,je),xt=a(je),Ue=c(je,"P",{"data-svelte-h":!0}),_(Ue)!=="svelte-1py4aay"&&(Ue.innerHTML=Ot),zt=a(je),p(E.$$.fragment,je),je.forEach(o),it.forEach(o),st=a(e),p(ve.$$.fragment,e),at=a(e),Je=c(e,"P",{}),L(Je).forEach(o),this.h()},h(){x(n,"name","hf:doc:metadata"),x(n,"content",mo),x(P,"class","flex flex-wrap space-x-1"),x(z,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),x(U,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),x(T,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),x(G,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),x(k,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),x(I,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),x(q,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),x(j,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),x(J,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,t){l(document.head,n),r(e,b,t),r(e,i,t),r(e,y,t),r(e,C,t),r(e,H,t),h(O,e,t),r(e,We,t),r(e,P,t),r(e,Ze,t),h(A,e,t),r(e,Be,t),r(e,Q,t),r(e,He,t),r(e,Y,t),r(e,Pe,t),r(e,K,t),r(e,Ve,t),r(e,ee,t),r(e,Re,t),r(e,te,t),r(e,Ne,t),h(oe,e,t),r(e,Xe,t),r(e,ne,t),r(e,Se,t),r(e,se,t),r(e,Ee,t),h(ae,e,t),r(e,De,t),h(re,e,t),r(e,Oe,t),r(e,z,t),h(ie,z,null),l(z,dt),l(z,ke),l(z,ct),h(V,z,null),r(e,Ae,t),h(le,e,t),r(e,Qe,t),r(e,T,t),h(de,T,null),l(T,mt),l(T,we),l(T,pt),l(T,Me),l(T,ht),l(T,$e),l(T,ut),l(T,U),h(ce,U,null),l(U,ft),l(U,Ce),l(U,gt),h(R,U,null),r(e,Ye,t),h(me,e,t),r(e,Ke,t),r(e,k,t),h(pe,k,null),l(k,_t),l(k,Ge),l(k,bt),l(k,xe),l(k,yt),l(k,ze),l(k,vt),l(k,G),h(he,G,null),l(G,Tt),l(G,Fe),l(G,kt),h(N,G,null),l(G,wt),h(X,G,null),r(e,et,t),h(ue,e,t),r(e,tt,t),r(e,q,t),h(fe,q,null),l(q,Mt),l(q,I),h(ge,I,null),l(I,$t),l(I,Le),l(I,Ct),h(S,I,null),r(e,ot,t),h(_e,e,t),r(e,nt,t),r(e,J,t),h(be,J,null),l(J,Gt),l(J,j),h(ye,j,null),l(j,xt),l(j,Ue),l(j,zt),h(E,j,null),r(e,st,t),h(ve,e,t),r(e,at,t),r(e,Je,t),rt=!0},p(e,[t]){const W={};t&2&&(W.$$scope={dirty:t,ctx:e}),V.$set(W);const M={};t&2&&(M.$$scope={dirty:t,ctx:e}),R.$set(M);const Z={};t&2&&(Z.$$scope={dirty:t,ctx:e}),N.$set(Z);const $={};t&2&&($.$$scope={dirty:t,ctx:e}),X.$set($);const F={};t&2&&(F.$$scope={dirty:t,ctx:e}),S.$set(F);const Te={};t&2&&(Te.$$scope={dirty:t,ctx:e}),E.$set(Te)},i(e){rt||(u(O.$$.fragment,e),u(A.$$.fragment,e),u(oe.$$.fragment,e),u(ae.$$.fragment,e),u(re.$$.fragment,e),u(ie.$$.fragment,e),u(V.$$.fragment,e),u(le.$$.fragment,e),u(de.$$.fragment,e),u(ce.$$.fragment,e),u(R.$$.fragment,e),u(me.$$.fragment,e),u(pe.$$.fragment,e),u(he.$$.fragment,e),u(N.$$.fragment,e),u(X.$$.fragment,e),u(ue.$$.fragment,e),u(fe.$$.fragment,e),u(ge.$$.fragment,e),u(S.$$.fragment,e),u(_e.$$.fragment,e),u(be.$$.fragment,e),u(ye.$$.fragment,e),u(E.$$.fragment,e),u(ve.$$.fragment,e),rt=!0)},o(e){f(O.$$.fragment,e),f(A.$$.fragment,e),f(oe.$$.fragment,e),f(ae.$$.fragment,e),f(re.$$.fragment,e),f(ie.$$.fragment,e),f(V.$$.fragment,e),f(le.$$.fragment,e),f(de.$$.fragment,e),f(ce.$$.fragment,e),f(R.$$.fragment,e),f(me.$$.fragment,e),f(pe.$$.fragment,e),f(he.$$.fragment,e),f(N.$$.fragment,e),f(X.$$.fragment,e),f(ue.$$.fragment,e),f(fe.$$.fragment,e),f(ge.$$.fragment,e),f(S.$$.fragment,e),f(_e.$$.fragment,e),f(be.$$.fragment,e),f(ye.$$.fragment,e),f(E.$$.fragment,e),f(ve.$$.fragment,e),rt=!1},d(e){e&&(o(b),o(i),o(y),o(C),o(H),o(We),o(P),o(Ze),o(Be),o(Q),o(He),o(Y),o(Pe),o(K),o(Ve),o(ee),o(Re),o(te),o(Ne),o(Xe),o(ne),o(Se),o(se),o(Ee),o(De),o(Oe),o(z),o(Ae),o(Qe),o(T),o(Ye),o(Ke),o(k),o(et),o(tt),o(q),o(ot),o(nt),o(J),o(st),o(at),o(Je)),o(n),g(O,e),g(A,e),g(oe,e),g(ae,e),g(re,e),g(ie),g(V),g(le,e),g(de),g(ce),g(R),g(me,e),g(pe),g(he),g(N),g(X),g(ue,e),g(fe),g(ge),g(S),g(_e,e),g(be),g(ye),g(E),g(ve,e)}}}const mo='{"title":"GLM","local":"glm","sections":[{"title":"Overview","local":"overview","sections":[],"depth":2},{"title":"Usage tips","local":"usage-tips","sections":[],"depth":2},{"title":"GlmConfig","local":"transformers.GlmConfig","sections":[],"depth":2},{"title":"GlmModel","local":"transformers.GlmModel","sections":[],"depth":2},{"title":"GlmForCausalLM","local":"transformers.GlmForCausalLM","sections":[],"depth":2},{"title":"GlmForSequenceClassification","local":"transformers.GlmForSequenceClassification","sections":[],"depth":2},{"title":"GlmForTokenClassification","local":"transformers.GlmForTokenClassification","sections":[],"depth":2}],"depth":1}';function po(w){return Yt(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class vo extends Kt{constructor(n){super(),eo(this,n,po,co,Qt,{})}}export{vo as component};
