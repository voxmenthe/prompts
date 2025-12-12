import{s as ao,o as ro,n as Ne}from"../chunks/scheduler.18a86fab.js";import{S as io,i as lo,g as l,s as a,r as m,A as co,h as d,f as o,c as r,j as F,x as y,u as h,k as C,y as c,a as n,v as u,d as f,t as g,w as _}from"../chunks/index.98837b22.js";import{T as yt}from"../chunks/Tip.77304350.js";import{D as G}from"../chunks/Docstring.a1ef7999.js";import{C as Ve}from"../chunks/CodeBlock.8d0c2e8a.js";import{E as so}from"../chunks/ExampleCodeBlock.8c3ee1f9.js";import{H as W,E as po}from"../chunks/getInferenceSnippets.06c2775f.js";function mo(w){let s,v="Example:",i,b,T;return b=new Ve({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMFN0YWJsZUxtTW9kZWwlMkMlMjBTdGFibGVMbUNvbmZpZyUwQSUwQSUyMyUyMEluaXRpYWxpemluZyUyMGElMjBTdGFibGVMTSUyMHN0YWJsZWxtLTNiJTIwc3R5bGUlMjBjb25maWd1cmF0aW9uJTBBY29uZmlndXJhdGlvbiUyMCUzRCUyMFN0YWJsZUxtQ29uZmlnKCk=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> StableLmModel, StableLmConfig

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a StableLM stablelm-3b style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = StableLmConfig()`,wrap:!1}}),{c(){s=l("p"),s.textContent=v,i=a(),m(b.$$.fragment)},l(p){s=d(p,"P",{"data-svelte-h":!0}),y(s)!=="svelte-11lpom8"&&(s.textContent=v),i=r(p),h(b.$$.fragment,p)},m(p,L){n(p,s,L),n(p,i,L),u(b,p,L),T=!0},p:Ne,i(p){T||(f(b.$$.fragment,p),T=!0)},o(p){g(b.$$.fragment,p),T=!1},d(p){p&&(o(s),o(i)),_(b,p)}}}function ho(w){let s,v=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){s=l("p"),s.innerHTML=v},l(i){s=d(i,"P",{"data-svelte-h":!0}),y(s)!=="svelte-fincs2"&&(s.innerHTML=v)},m(i,b){n(i,s,b)},p:Ne,d(i){i&&o(s)}}}function uo(w){let s,v=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){s=l("p"),s.innerHTML=v},l(i){s=d(i,"P",{"data-svelte-h":!0}),y(s)!=="svelte-fincs2"&&(s.innerHTML=v)},m(i,b){n(i,s,b)},p:Ne,d(i){i&&o(s)}}}function fo(w){let s,v="Example:",i,b,T;return b=new Ve({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBTdGFibGVMbUZvckNhdXNhbExNJTBBJTBBbW9kZWwlMjAlM0QlMjBTdGFibGVMbUZvckNhdXNhbExNLmZyb21fcHJldHJhaW5lZCglMjJhZGVwdCUyRnBlcnNpbW1vbi04Yi1iYXNlJTIyKSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMmFkZXB0JTJGcGVyc2ltbW9uLThiLWJhc2UlMjIpJTBBJTBBcHJvbXB0JTIwJTNEJTIwJTIyaHVtYW4lM0ElMjBIZXklMkMlMjB3aGF0JTIwc2hvdWxkJTIwSSUyMGVhdCUyMGZvciUyMGRpbm5lciUzRiUyMiUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplcihwcm9tcHQlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKSUwQSUwQSUyMyUyMEdlbmVyYXRlJTBBZ2VuZXJhdGVfaWRzJTIwJTNEJTIwbW9kZWwuZ2VuZXJhdGUoaW5wdXRzLmlucHV0X2lkcyUyQyUyMG1heF9sZW5ndGglM0QzMCklMEF0b2tlbml6ZXIuYmF0Y2hfZGVjb2RlKGdlbmVyYXRlX2lkcyUyQyUyMHNraXBfc3BlY2lhbF90b2tlbnMlM0RUcnVlJTJDJTIwY2xlYW5fdXBfdG9rZW5pemF0aW9uX3NwYWNlcyUzREZhbHNlKSU1QjAlNUQ=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, StableLmForCausalLM

<span class="hljs-meta">&gt;&gt;&gt; </span>model = StableLmForCausalLM.from_pretrained(<span class="hljs-string">&quot;adept/persimmon-8b-base&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;adept/persimmon-8b-base&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>prompt = <span class="hljs-string">&quot;human: Hey, what should I eat for dinner?&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(prompt, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Generate</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>generate_ids = model.generate(inputs.input_ids, max_length=<span class="hljs-number">30</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer.batch_decode(generate_ids, skip_special_tokens=<span class="hljs-literal">True</span>, clean_up_tokenization_spaces=<span class="hljs-literal">False</span>)[<span class="hljs-number">0</span>]
<span class="hljs-string">&#x27;human: Hey, what should I eat for dinner?\\n\\ncat: 🐱\\n\\nhuman: 😐\\n\\n&#x27;</span>`,wrap:!1}}),{c(){s=l("p"),s.textContent=v,i=a(),m(b.$$.fragment)},l(p){s=d(p,"P",{"data-svelte-h":!0}),y(s)!=="svelte-11lpom8"&&(s.textContent=v),i=r(p),h(b.$$.fragment,p)},m(p,L){n(p,s,L),n(p,i,L),u(b,p,L),T=!0},p:Ne,i(p){T||(f(b.$$.fragment,p),T=!0)},o(p){g(b.$$.fragment,p),T=!1},d(p){p&&(o(s),o(i)),_(b,p)}}}function go(w){let s,v=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){s=l("p"),s.innerHTML=v},l(i){s=d(i,"P",{"data-svelte-h":!0}),y(s)!=="svelte-fincs2"&&(s.innerHTML=v)},m(i,b){n(i,s,b)},p:Ne,d(i){i&&o(s)}}}function _o(w){let s,v=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){s=l("p"),s.innerHTML=v},l(i){s=d(i,"P",{"data-svelte-h":!0}),y(s)!=="svelte-fincs2"&&(s.innerHTML=v)},m(i,b){n(i,s,b)},p:Ne,d(i){i&&o(s)}}}function bo(w){let s,v,i,b,T,p="<em>This model was released on 2023-09-05 and added to Hugging Face Transformers on 2024-02-14.</em>",L,O,Pe,V,Nt='<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/> <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat"/> <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/>',Be,A,Re,D,Gt='StableLM 3B 4E1T (<a href="https://stability.ai/news/stable-lm-3b-sustainable-high-performance-language-models-smart-devices" rel="nofollow">blog post</a>) was proposed in <a href="https://stability.wandb.io/stability-llm/stable-lm/reports/StableLM-3B-4E1T--VmlldzoyMjU4?accessToken=u3zujipenkx5g7rtcj9qojjgxpconyjktjkli2po09nffrffdhhchq045vp0wyfo" rel="nofollow">StableLM 3B 4E1T: Technical Report</a> by Stability AI and is the first model in a series of multi-epoch pre-trained language models.',He,Q,Ee,Y,Vt=`StableLM 3B 4E1T is a decoder-only base language model pre-trained on 1 trillion tokens of diverse English and code datasets for four epochs.
The model architecture is transformer-based with partial Rotary Position Embeddings, SwiGLU activation, LayerNorm, etc.`,Xe,K,Pt="We also provide StableLM Zephyr 3B, an instruction fine-tuned version of the model that can be used for chat-based applications.",Oe,ee,Ae,te,Bt='<li>The architecture is similar to LLaMA but with RoPE applied to 25% of head embedding dimensions, LayerNorm instead of RMSNorm, and optional QKV bias terms.</li> <li><code>StableLM 3B 4E1T</code>-based models uses the same tokenizer as <a href="/docs/transformers/v4.56.2/en/model_doc/gpt_neox#transformers.GPTNeoXTokenizerFast">GPTNeoXTokenizerFast</a>.</li>',De,oe,Rt='<code>StableLM 3B 4E1T</code> and <code>StableLM Zephyr 3B</code> can be found on the <a href="https://huggingface.co/stabilityai" rel="nofollow">Huggingface Hub</a>',Qe,ne,Ht="The following code snippet demonstrates how to use <code>StableLM 3B 4E1T</code> for inference:",Ye,se,Ke,ae,et,re,Et="First, make sure to install the latest version of Flash Attention v2.",tt,ie,ot,le,Xt='Also make sure that your hardware is compatible with Flash-Attention 2. Read more about it in the official documentation of the <a href="https://github.com/Dao-AILab/flash-attention" rel="nofollow"><code>flash-attn</code></a> repository. Note: you must load your model in half-precision (e.g. <code>torch.bfloat16</code>).',nt,de,Ot="Now, to run the model with Flash Attention 2, refer to the snippet below:",st,ce,at,pe,rt,M,me,vt,Ce,At=`This is the configuration class to store the configuration of a <a href="/docs/transformers/v4.56.2/en/model_doc/stablelm#transformers.StableLmModel">~StableLmModel</a>.
It is used to instantiate an StableLM model according to the specified arguments, defining the model
architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of
the StableLM <a href="https://huggingface.co/stabilityai/stablelm-3b-4e1t" rel="nofollow">stabilityai/stablelm-3b-4e1t</a> architecture.`,Tt,Se,Dt=`Configuration objects inherit from  <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> and can be used
to control the model outputs. Read the documentation from  <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a>
for more information.`,kt,P,it,he,lt,k,ue,wt,ze,Qt="The bare Stablelm Model outputting raw hidden-states without any specific head on top.",Mt,Fe,Yt=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,$t,Ue,Kt=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Lt,U,fe,xt,je,eo='The <a href="/docs/transformers/v4.56.2/en/model_doc/stablelm#transformers.StableLmModel">StableLmModel</a> forward method, overrides the <code>__call__</code> special method.',Ct,B,dt,ge,ct,I,_e,St,x,be,zt,qe,to='The <a href="/docs/transformers/v4.56.2/en/model_doc/stablelm#transformers.StableLmForCausalLM">StableLmForCausalLM</a> forward method, overrides the <code>__call__</code> special method.',Ft,R,Ut,H,pt,ye,mt,J,ve,jt,j,Te,qt,We,oo="The <code>GenericForSequenceClassification</code> forward method, overrides the <code>__call__</code> special method.",Wt,E,ht,ke,ut,Z,we,It,q,Me,Jt,Ie,no="The <code>GenericForTokenClassification</code> forward method, overrides the <code>__call__</code> special method.",Zt,X,ft,$e,gt,Ge,_t;return O=new W({props:{title:"StableLM",local:"stablelm",headingTag:"h1"}}),A=new W({props:{title:"Overview",local:"overview",headingTag:"h2"}}),Q=new W({props:{title:"Model Details",local:"model-details",headingTag:"h3"}}),ee=new W({props:{title:"Usage Tips",local:"usage-tips",headingTag:"h3"}}),se=new Ve({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Nb2RlbEZvckNhdXNhbExNJTJDJTIwQXV0b1Rva2VuaXplciUyQyUyMGluZmVyX2RldmljZSUyQyUyMHNldF9zZWVkJTBBZGV2aWNlJTIwJTNEJTIwaW5mZXJfZGV2aWNlKCklMjAlMjMlMjB0aGUlMjBkZXZpY2UlMjB0byUyMGxvYWQlMjB0aGUlMjBtb2RlbCUyMG9udG8lMEElMEFzZXRfc2VlZCgwKSUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMnN0YWJpbGl0eWFpJTJGc3RhYmxlbG0tM2ItNGUxdCUyMiklMEFtb2RlbCUyMCUzRCUyMEF1dG9Nb2RlbEZvckNhdXNhbExNLmZyb21fcHJldHJhaW5lZCglMjJzdGFiaWxpdHlhaSUyRnN0YWJsZWxtLTNiLTRlMXQlMjIpJTBBbW9kZWwudG8oZGV2aWNlKSUwQW1vZGVsX2lucHV0cyUyMCUzRCUyMHRva2VuaXplciglMjJUaGUlMjB3ZWF0aGVyJTIwaXMlMjBhbHdheXMlMjB3b25kZXJmdWwlMjBpbiUyMiUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpLnRvKG1vZGVsLmRldmljZSklMEElMEFnZW5lcmF0ZWRfaWRzJTIwJTNEJTIwbW9kZWwuZ2VuZXJhdGUoKiptb2RlbF9pbnB1dHMlMkMlMjBtYXhfbGVuZ3RoJTNEMzIlMkMlMjBkb19zYW1wbGUlM0RUcnVlKSUwQXJlc3BvbnNlcyUyMCUzRCUyMHRva2VuaXplci5iYXRjaF9kZWNvZGUoZ2VuZXJhdGVkX2lkcyUyQyUyMHNraXBfc3BlY2lhbF90b2tlbnMlM0RUcnVlKSUwQXJlc3BvbnNlcw==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoModelForCausalLM, AutoTokenizer, infer_device, set_seed
<span class="hljs-meta">&gt;&gt;&gt; </span>device = infer_device() <span class="hljs-comment"># the device to load the model onto</span>

<span class="hljs-meta">&gt;&gt;&gt; </span>set_seed(<span class="hljs-number">0</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;stabilityai/stablelm-3b-4e1t&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = AutoModelForCausalLM.from_pretrained(<span class="hljs-string">&quot;stabilityai/stablelm-3b-4e1t&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model.to(device)
<span class="hljs-meta">&gt;&gt;&gt; </span>model_inputs = tokenizer(<span class="hljs-string">&quot;The weather is always wonderful in&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>).to(model.device)

<span class="hljs-meta">&gt;&gt;&gt; </span>generated_ids = model.generate(**model_inputs, max_length=<span class="hljs-number">32</span>, do_sample=<span class="hljs-literal">True</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=<span class="hljs-literal">True</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>responses
[<span class="hljs-string">&#x27;The weather is always wonderful in Costa Rica, which makes it a prime destination for retirees. That’s where the Pensionado program comes in, offering&#x27;</span>]`,wrap:!1}}),ae=new W({props:{title:"Combining StableLM and Flash Attention 2",local:"combining-stablelm-and-flash-attention-2",headingTag:"h2"}}),ie=new Ve({props:{code:"cGlwJTIwaW5zdGFsbCUyMC1VJTIwZmxhc2gtYXR0biUyMC0tbm8tYnVpbGQtaXNvbGF0aW9u",highlighted:"pip install -U flash-attn --no-build-isolation",wrap:!1}}),ce=new Ve({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b01vZGVsRm9yQ2F1c2FsTE0lMkMlMjBBdXRvVG9rZW5pemVyJTJDJTIwaW5mZXJfZGV2aWNlJTJDJTIwc2V0X3NlZWQlMEFkZXZpY2UlMjAlM0QlMjBpbmZlcl9kZXZpY2UoKSUyMCUyMyUyMHRoZSUyMGRldmljZSUyMHRvJTIwbG9hZCUyMHRoZSUyMG1vZGVsJTIwb250byUwQSUwQXNldF9zZWVkKDApJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIyc3RhYmlsaXR5YWklMkZzdGFibGVsbS0zYi00ZTF0JTIyKSUwQW1vZGVsJTIwJTNEJTIwQXV0b01vZGVsRm9yQ2F1c2FsTE0uZnJvbV9wcmV0cmFpbmVkKCUyMnN0YWJpbGl0eWFpJTJGc3RhYmxlbG0tM2ItNGUxdCUyMiUyQyUyMGR0eXBlJTNEdG9yY2guYmZsb2F0MTYlMkMlMjBhdHRuX2ltcGxlbWVudGF0aW9uJTNEJTIyZmxhc2hfYXR0ZW50aW9uXzIlMjIpJTBBbW9kZWwudG8oZGV2aWNlKSUwQW1vZGVsX2lucHV0cyUyMCUzRCUyMHRva2VuaXplciglMjJUaGUlMjB3ZWF0aGVyJTIwaXMlMjBhbHdheXMlMjB3b25kZXJmdWwlMjBpbiUyMiUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpLnRvKG1vZGVsLmRldmljZSklMEElMEFnZW5lcmF0ZWRfaWRzJTIwJTNEJTIwbW9kZWwuZ2VuZXJhdGUoKiptb2RlbF9pbnB1dHMlMkMlMjBtYXhfbGVuZ3RoJTNEMzIlMkMlMjBkb19zYW1wbGUlM0RUcnVlKSUwQXJlc3BvbnNlcyUyMCUzRCUyMHRva2VuaXplci5iYXRjaF9kZWNvZGUoZ2VuZXJhdGVkX2lkcyUyQyUyMHNraXBfc3BlY2lhbF90b2tlbnMlM0RUcnVlKSUwQXJlc3BvbnNlcw==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoModelForCausalLM, AutoTokenizer, infer_device, set_seed
<span class="hljs-meta">&gt;&gt;&gt; </span>device = infer_device() <span class="hljs-comment"># the device to load the model onto</span>

<span class="hljs-meta">&gt;&gt;&gt; </span>set_seed(<span class="hljs-number">0</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;stabilityai/stablelm-3b-4e1t&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = AutoModelForCausalLM.from_pretrained(<span class="hljs-string">&quot;stabilityai/stablelm-3b-4e1t&quot;</span>, dtype=torch.bfloat16, attn_implementation=<span class="hljs-string">&quot;flash_attention_2&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model.to(device)
<span class="hljs-meta">&gt;&gt;&gt; </span>model_inputs = tokenizer(<span class="hljs-string">&quot;The weather is always wonderful in&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>).to(model.device)

<span class="hljs-meta">&gt;&gt;&gt; </span>generated_ids = model.generate(**model_inputs, max_length=<span class="hljs-number">32</span>, do_sample=<span class="hljs-literal">True</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=<span class="hljs-literal">True</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>responses
[<span class="hljs-string">&#x27;The weather is always wonderful in Costa Rica, which makes it a prime destination for retirees. That’s where the Pensionado program comes in, offering&#x27;</span>]`,wrap:!1}}),pe=new W({props:{title:"StableLmConfig",local:"transformers.StableLmConfig",headingTag:"h2"}}),me=new G({props:{name:"class transformers.StableLmConfig",anchor:"transformers.StableLmConfig",parameters:[{name:"vocab_size",val:" = 50304"},{name:"intermediate_size",val:" = 6912"},{name:"hidden_size",val:" = 2560"},{name:"num_hidden_layers",val:" = 32"},{name:"num_attention_heads",val:" = 32"},{name:"num_key_value_heads",val:" = 32"},{name:"hidden_act",val:" = 'silu'"},{name:"max_position_embeddings",val:" = 4096"},{name:"initializer_range",val:" = 0.02"},{name:"layer_norm_eps",val:" = 1e-05"},{name:"use_cache",val:" = True"},{name:"tie_word_embeddings",val:" = False"},{name:"rope_theta",val:" = 10000"},{name:"rope_scaling",val:" = None"},{name:"use_qkv_bias",val:" = False"},{name:"qk_layernorm",val:" = False"},{name:"use_parallel_residual",val:" = False"},{name:"hidden_dropout",val:" = 0.0"},{name:"attention_dropout",val:" = 0.0"},{name:"partial_rotary_factor",val:" = 0.25"},{name:"bos_token_id",val:" = 0"},{name:"eos_token_id",val:" = 0"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.StableLmConfig.vocab_size",description:`<strong>vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 50304) &#x2014;
Vocabulary size of the StableLM model. Defines the number of different tokens that
can be represented by the <code>inputs_ids</code> passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/stablelm#transformers.StableLmModel">StableLmModel</a>.`,name:"vocab_size"},{anchor:"transformers.StableLmConfig.intermediate_size",description:`<strong>intermediate_size</strong> (<code>int</code>, <em>optional</em>, defaults to 6912) &#x2014;
Dimension of the MLP representations.`,name:"intermediate_size"},{anchor:"transformers.StableLmConfig.hidden_size",description:`<strong>hidden_size</strong> (<code>int</code>, <em>optional</em>, defaults to 2560) &#x2014;
Number of hidden layers in the Transformer decoder.`,name:"hidden_size"},{anchor:"transformers.StableLmConfig.num_hidden_layers",description:`<strong>num_hidden_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 32) &#x2014;
Number of hidden layers in the Transformer decoder.`,name:"num_hidden_layers"},{anchor:"transformers.StableLmConfig.num_attention_heads",description:`<strong>num_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 32) &#x2014;
Number of attention heads for each attention layer in the Transformer encoder.`,name:"num_attention_heads"},{anchor:"transformers.StableLmConfig.num_key_value_heads",description:`<strong>num_key_value_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 32) &#x2014;
This is the number of key_value heads that should be used to implement Grouped Query Attention. If
<code>num_key_value_heads=num_attention_heads</code>, the model will use Multi Head Attention (MHA), if
<code>num_key_value_heads=1</code> the model will use Multi Query Attention (MQA) otherwise GQA is used. When
converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
by meanpooling all the original heads within that group. For more details, check out <a href="https://huggingface.co/papers/2305.13245" rel="nofollow">this
paper</a>. If it is not specified, will default to
<code>num_attention_heads</code>.`,name:"num_key_value_heads"},{anchor:"transformers.StableLmConfig.hidden_act",description:`<strong>hidden_act</strong> (<code>str</code> or <code>function</code>, <em>optional</em>, defaults to <code>&quot;silu&quot;</code>) &#x2014;
The non-linear activation function (function or string).`,name:"hidden_act"},{anchor:"transformers.StableLmConfig.max_position_embeddings",description:`<strong>max_position_embeddings</strong> (<code>int</code>, <em>optional</em>, defaults to 4096) &#x2014;
The maximum sequence length that this model might ever be used with.
Typically set this to something large just in case (e.g., 512 or 1024 or 2048).`,name:"max_position_embeddings"},{anchor:"transformers.StableLmConfig.initializer_range",description:`<strong>initializer_range</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing
all weight matrices.`,name:"initializer_range"},{anchor:"transformers.StableLmConfig.layer_norm_eps",description:`<strong>layer_norm_eps</strong> (<code>float</code>, <em>optional</em>, defaults to 1e-05) &#x2014;
The epsilon used by the normalization layers.`,name:"layer_norm_eps"},{anchor:"transformers.StableLmConfig.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not the model should return the last key/values attentions
(not used by all models). Only relevant if <code>config.is_decoder=True</code>.`,name:"use_cache"},{anchor:"transformers.StableLmConfig.tie_word_embeddings",description:`<strong>tie_word_embeddings</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether the model&#x2019;s input and output word embeddings should be tied.`,name:"tie_word_embeddings"},{anchor:"transformers.StableLmConfig.rope_theta",description:`<strong>rope_theta</strong> (<code>float</code>, <em>optional</em>, defaults to <code>10000.0</code>) &#x2014;
The base period of the RoPE embeddings.`,name:"rope_theta"},{anchor:"transformers.StableLmConfig.rope_scaling",description:`<strong>rope_scaling</strong> (<code>Dict</code>, <em>optional</em>) &#x2014;
Dictionary containing the scaling configuration for the RoPE embeddings. NOTE: if you apply new rope type
and you expect the model to work on longer <code>max_position_embeddings</code>, we recommend you to update this value
accordingly.
Expected contents:
<code>rope_type</code> (<code>str</code>):
The sub-variant of RoPE to use. Can be one of [&#x2018;default&#x2019;, &#x2018;linear&#x2019;, &#x2018;dynamic&#x2019;, &#x2018;yarn&#x2019;, &#x2018;longrope&#x2019;,
&#x2018;llama3&#x2019;], with &#x2018;default&#x2019; being the original RoPE implementation.
<code>factor</code> (<code>float</code>, <em>optional</em>):
Used with all rope types except &#x2018;default&#x2019;. The scaling factor to apply to the RoPE embeddings. In
most scaling types, a <code>factor</code> of x will enable the model to handle sequences of length x <em>
original maximum pre-trained length.
<code>original_max_position_embeddings</code> (<code>int</code>, </em>optional<em>):
Used with &#x2018;dynamic&#x2019;, &#x2018;longrope&#x2019; and &#x2018;llama3&#x2019;. The original max position embeddings used during
pretraining.
<code>attention_factor</code> (<code>float</code>, </em>optional<em>):
Used with &#x2018;yarn&#x2019; and &#x2018;longrope&#x2019;. The scaling factor to be applied on the attention
computation. If unspecified, it defaults to value recommended by the implementation, using the
<code>factor</code> field to infer the suggested value.
<code>beta_fast</code> (<code>float</code>, </em>optional<em>):
Only used with &#x2018;yarn&#x2019;. Parameter to set the boundary for extrapolation (only) in the linear
ramp function. If unspecified, it defaults to 32.
<code>beta_slow</code> (<code>float</code>, </em>optional<em>):
Only used with &#x2018;yarn&#x2019;. Parameter to set the boundary for interpolation (only) in the linear
ramp function. If unspecified, it defaults to 1.
<code>short_factor</code> (<code>list[float]</code>, </em>optional<em>):
Only used with &#x2018;longrope&#x2019;. The scaling factor to be applied to short contexts (&lt;
<code>original_max_position_embeddings</code>). Must be a list of numbers with the same length as the hidden
size divided by the number of attention heads divided by 2
<code>long_factor</code> (<code>list[float]</code>, </em>optional<em>):
Only used with &#x2018;longrope&#x2019;. The scaling factor to be applied to long contexts (&lt;
<code>original_max_position_embeddings</code>). Must be a list of numbers with the same length as the hidden
size divided by the number of attention heads divided by 2
<code>low_freq_factor</code> (<code>float</code>, </em>optional<em>):
Only used with &#x2018;llama3&#x2019;. Scaling factor applied to low frequency components of the RoPE
<code>high_freq_factor</code> (<code>float</code>, </em>optional*):
Only used with &#x2018;llama3&#x2019;. Scaling factor applied to high frequency components of the RoPE`,name:"rope_scaling"},{anchor:"transformers.StableLmConfig.use_qkv_bias",description:`<strong>use_qkv_bias</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not the model should use bias for qkv layers.`,name:"use_qkv_bias"},{anchor:"transformers.StableLmConfig.qk_layernorm",description:`<strong>qk_layernorm</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to normalize, per head, the Queries and Keys after projecting the hidden states.`,name:"qk_layernorm"},{anchor:"transformers.StableLmConfig.use_parallel_residual",description:`<strong>use_parallel_residual</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether to use a &#x201C;parallel&#x201D; formulation in each Transformer layer, which can provide a slight training
speedup at large scales.`,name:"use_parallel_residual"},{anchor:"transformers.StableLmConfig.hidden_dropout",description:`<strong>hidden_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The dropout ratio after applying the MLP to the hidden states.`,name:"hidden_dropout"},{anchor:"transformers.StableLmConfig.attention_dropout",description:`<strong>attention_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The dropout ratio for the attention probabilities.`,name:"attention_dropout"},{anchor:"transformers.StableLmConfig.partial_rotary_factor",description:`<strong>partial_rotary_factor</strong> (<code>float</code>, <em>optional</em>, defaults to 0.25) &#x2014;
Percentage of the query and keys which will have rotary embedding.`,name:"partial_rotary_factor"},{anchor:"transformers.StableLmConfig.bos_token_id",description:`<strong>bos_token_id</strong> (int, <em>optional</em>, defaults to 0) &#x2014;
The id of the <code>BOS</code> token in the vocabulary.`,name:"bos_token_id"},{anchor:"transformers.StableLmConfig.eos_token_id",description:`<strong>eos_token_id</strong> (int, <em>optional</em>, defaults to 0) &#x2014;
The id of the <code>EOS</code> token in the vocabulary.`,name:"eos_token_id"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/stablelm/configuration_stablelm.py#L25"}}),P=new so({props:{anchor:"transformers.StableLmConfig.example",$$slots:{default:[mo]},$$scope:{ctx:w}}}),he=new W({props:{title:"StableLmModel",local:"transformers.StableLmModel",headingTag:"h2"}}),ue=new G({props:{name:"class transformers.StableLmModel",anchor:"transformers.StableLmModel",parameters:[{name:"config",val:": StableLmConfig"}],parametersDescription:[{anchor:"transformers.StableLmModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/stablelm#transformers.StableLmConfig">StableLmConfig</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/stablelm/modeling_stablelm.py#L648"}}),fe=new G({props:{name:"forward",anchor:"transformers.StableLmModel.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"}],parametersDescription:[{anchor:"transformers.StableLmModel.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.StableLmModel.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.StableLmModel.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.StableLmModel.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.StableLmModel.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.StableLmModel.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.StableLmModel.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.StableLmModel.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.StableLmModel.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/stablelm/modeling_stablelm.py#L673",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast"
>transformers.modeling_outputs.BaseModelOutputWithPast</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/stablelm#transformers.StableLmConfig"
>StableLmConfig</a>) and inputs.</p>
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
`}}),B=new yt({props:{$$slots:{default:[ho]},$$scope:{ctx:w}}}),ge=new W({props:{title:"StableLmForCausalLM",local:"transformers.StableLmForCausalLM",headingTag:"h2"}}),_e=new G({props:{name:"class transformers.StableLmForCausalLM",anchor:"transformers.StableLmForCausalLM",parameters:[{name:"config",val:""}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/stablelm/modeling_stablelm.py#L894"}}),be=new G({props:{name:"forward",anchor:"transformers.StableLmForCausalLM.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"},{name:"logits_to_keep",val:": typing.Union[int, torch.Tensor] = 0"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.StableLmForCausalLM.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.StableLmForCausalLM.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.StableLmForCausalLM.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.StableLmForCausalLM.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.StableLmForCausalLM.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.StableLmForCausalLM.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.StableLmForCausalLM.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.StableLmForCausalLM.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.StableLmForCausalLM.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.StableLmForCausalLM.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"},{anchor:"transformers.StableLmForCausalLM.forward.logits_to_keep",description:`<strong>logits_to_keep</strong> (<code>Union[int, torch.Tensor]</code>, defaults to <code>0</code>) &#x2014;
If an <code>int</code>, compute logits for the last <code>logits_to_keep</code> tokens. If <code>0</code>, calculate logits for all
<code>input_ids</code> (special case). Only last token logits are needed for generation, and calculating them only for that
token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
If a <code>torch.Tensor</code>, must be 1D corresponding to the indices to keep in the sequence length dimension.
This is useful when using packed tensor format (single dimension for batch and sequence length).`,name:"logits_to_keep"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/stablelm/modeling_stablelm.py#L907",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast"
>transformers.modeling_outputs.CausalLMOutputWithPast</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/stablelm#transformers.StableLmConfig"
>StableLmConfig</a>) and inputs.</p>
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
`}}),R=new yt({props:{$$slots:{default:[uo]},$$scope:{ctx:w}}}),H=new so({props:{anchor:"transformers.StableLmForCausalLM.forward.example",$$slots:{default:[fo]},$$scope:{ctx:w}}}),ye=new W({props:{title:"StableLmForSequenceClassification",local:"transformers.StableLmForSequenceClassification",headingTag:"h2"}}),ve=new G({props:{name:"class transformers.StableLmForSequenceClassification",anchor:"transformers.StableLmForSequenceClassification",parameters:[{name:"config",val:""}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/stablelm/modeling_stablelm.py#L989"}}),Te=new G({props:{name:"forward",anchor:"transformers.StableLmForSequenceClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.StableLmForSequenceClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.StableLmForSequenceClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.StableLmForSequenceClassification.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.StableLmForSequenceClassification.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.StableLmForSequenceClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.StableLmForSequenceClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.StableLmForSequenceClassification.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
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
`}}),E=new yt({props:{$$slots:{default:[go]},$$scope:{ctx:w}}}),ke=new W({props:{title:"StableLmForTokenClassification",local:"transformers.StableLmForTokenClassification",headingTag:"h2"}}),we=new G({props:{name:"class transformers.StableLmForTokenClassification",anchor:"transformers.StableLmForTokenClassification",parameters:[{name:"config",val:""}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/stablelm/modeling_stablelm.py#L992"}}),Me=new G({props:{name:"forward",anchor:"transformers.StableLmForTokenClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.StableLmForTokenClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.StableLmForTokenClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.StableLmForTokenClassification.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.StableLmForTokenClassification.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.StableLmForTokenClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.StableLmForTokenClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.StableLmForTokenClassification.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
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
`}}),X=new yt({props:{$$slots:{default:[_o]},$$scope:{ctx:w}}}),$e=new po({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/stablelm.md"}}),{c(){s=l("meta"),v=a(),i=l("p"),b=a(),T=l("p"),T.innerHTML=p,L=a(),m(O.$$.fragment),Pe=a(),V=l("div"),V.innerHTML=Nt,Be=a(),m(A.$$.fragment),Re=a(),D=l("p"),D.innerHTML=Gt,He=a(),m(Q.$$.fragment),Ee=a(),Y=l("p"),Y.textContent=Vt,Xe=a(),K=l("p"),K.textContent=Pt,Oe=a(),m(ee.$$.fragment),Ae=a(),te=l("ul"),te.innerHTML=Bt,De=a(),oe=l("p"),oe.innerHTML=Rt,Qe=a(),ne=l("p"),ne.innerHTML=Ht,Ye=a(),m(se.$$.fragment),Ke=a(),m(ae.$$.fragment),et=a(),re=l("p"),re.textContent=Et,tt=a(),m(ie.$$.fragment),ot=a(),le=l("p"),le.innerHTML=Xt,nt=a(),de=l("p"),de.textContent=Ot,st=a(),m(ce.$$.fragment),at=a(),m(pe.$$.fragment),rt=a(),M=l("div"),m(me.$$.fragment),vt=a(),Ce=l("p"),Ce.innerHTML=At,Tt=a(),Se=l("p"),Se.innerHTML=Dt,kt=a(),m(P.$$.fragment),it=a(),m(he.$$.fragment),lt=a(),k=l("div"),m(ue.$$.fragment),wt=a(),ze=l("p"),ze.textContent=Qt,Mt=a(),Fe=l("p"),Fe.innerHTML=Yt,$t=a(),Ue=l("p"),Ue.innerHTML=Kt,Lt=a(),U=l("div"),m(fe.$$.fragment),xt=a(),je=l("p"),je.innerHTML=eo,Ct=a(),m(B.$$.fragment),dt=a(),m(ge.$$.fragment),ct=a(),I=l("div"),m(_e.$$.fragment),St=a(),x=l("div"),m(be.$$.fragment),zt=a(),qe=l("p"),qe.innerHTML=to,Ft=a(),m(R.$$.fragment),Ut=a(),m(H.$$.fragment),pt=a(),m(ye.$$.fragment),mt=a(),J=l("div"),m(ve.$$.fragment),jt=a(),j=l("div"),m(Te.$$.fragment),qt=a(),We=l("p"),We.innerHTML=oo,Wt=a(),m(E.$$.fragment),ht=a(),m(ke.$$.fragment),ut=a(),Z=l("div"),m(we.$$.fragment),It=a(),q=l("div"),m(Me.$$.fragment),Jt=a(),Ie=l("p"),Ie.innerHTML=no,Zt=a(),m(X.$$.fragment),ft=a(),m($e.$$.fragment),gt=a(),Ge=l("p"),this.h()},l(e){const t=co("svelte-u9bgzb",document.head);s=d(t,"META",{name:!0,content:!0}),t.forEach(o),v=r(e),i=d(e,"P",{}),F(i).forEach(o),b=r(e),T=d(e,"P",{"data-svelte-h":!0}),y(T)!=="svelte-wgk7as"&&(T.innerHTML=p),L=r(e),h(O.$$.fragment,e),Pe=r(e),V=d(e,"DIV",{class:!0,"data-svelte-h":!0}),y(V)!=="svelte-b95w5j"&&(V.innerHTML=Nt),Be=r(e),h(A.$$.fragment,e),Re=r(e),D=d(e,"P",{"data-svelte-h":!0}),y(D)!=="svelte-c21z8l"&&(D.innerHTML=Gt),He=r(e),h(Q.$$.fragment,e),Ee=r(e),Y=d(e,"P",{"data-svelte-h":!0}),y(Y)!=="svelte-1jj98r9"&&(Y.textContent=Vt),Xe=r(e),K=d(e,"P",{"data-svelte-h":!0}),y(K)!=="svelte-1nrviiz"&&(K.textContent=Pt),Oe=r(e),h(ee.$$.fragment,e),Ae=r(e),te=d(e,"UL",{"data-svelte-h":!0}),y(te)!=="svelte-2kv2f3"&&(te.innerHTML=Bt),De=r(e),oe=d(e,"P",{"data-svelte-h":!0}),y(oe)!=="svelte-1a9jefj"&&(oe.innerHTML=Rt),Qe=r(e),ne=d(e,"P",{"data-svelte-h":!0}),y(ne)!=="svelte-udtc57"&&(ne.innerHTML=Ht),Ye=r(e),h(se.$$.fragment,e),Ke=r(e),h(ae.$$.fragment,e),et=r(e),re=d(e,"P",{"data-svelte-h":!0}),y(re)!=="svelte-pstkbw"&&(re.textContent=Et),tt=r(e),h(ie.$$.fragment,e),ot=r(e),le=d(e,"P",{"data-svelte-h":!0}),y(le)!=="svelte-8b9qy3"&&(le.innerHTML=Xt),nt=r(e),de=d(e,"P",{"data-svelte-h":!0}),y(de)!=="svelte-1w2ttra"&&(de.textContent=Ot),st=r(e),h(ce.$$.fragment,e),at=r(e),h(pe.$$.fragment,e),rt=r(e),M=d(e,"DIV",{class:!0});var S=F(M);h(me.$$.fragment,S),vt=r(S),Ce=d(S,"P",{"data-svelte-h":!0}),y(Ce)!=="svelte-1wpc3y1"&&(Ce.innerHTML=At),Tt=r(S),Se=d(S,"P",{"data-svelte-h":!0}),y(Se)!=="svelte-146nnu1"&&(Se.innerHTML=Dt),kt=r(S),h(P.$$.fragment,S),S.forEach(o),it=r(e),h(he.$$.fragment,e),lt=r(e),k=d(e,"DIV",{class:!0});var $=F(k);h(ue.$$.fragment,$),wt=r($),ze=d($,"P",{"data-svelte-h":!0}),y(ze)!=="svelte-o412yk"&&(ze.textContent=Qt),Mt=r($),Fe=d($,"P",{"data-svelte-h":!0}),y(Fe)!=="svelte-q52n56"&&(Fe.innerHTML=Yt),$t=r($),Ue=d($,"P",{"data-svelte-h":!0}),y(Ue)!=="svelte-hswkmf"&&(Ue.innerHTML=Kt),Lt=r($),U=d($,"DIV",{class:!0});var N=F(U);h(fe.$$.fragment,N),xt=r(N),je=d(N,"P",{"data-svelte-h":!0}),y(je)!=="svelte-18nzblb"&&(je.innerHTML=eo),Ct=r(N),h(B.$$.fragment,N),N.forEach(o),$.forEach(o),dt=r(e),h(ge.$$.fragment,e),ct=r(e),I=d(e,"DIV",{class:!0});var Le=F(I);h(_e.$$.fragment,Le),St=r(Le),x=d(Le,"DIV",{class:!0});var z=F(x);h(be.$$.fragment,z),zt=r(z),qe=d(z,"P",{"data-svelte-h":!0}),y(qe)!=="svelte-19xeudn"&&(qe.innerHTML=to),Ft=r(z),h(R.$$.fragment,z),Ut=r(z),h(H.$$.fragment,z),z.forEach(o),Le.forEach(o),pt=r(e),h(ye.$$.fragment,e),mt=r(e),J=d(e,"DIV",{class:!0});var xe=F(J);h(ve.$$.fragment,xe),jt=r(xe),j=d(xe,"DIV",{class:!0});var Je=F(j);h(Te.$$.fragment,Je),qt=r(Je),We=d(Je,"P",{"data-svelte-h":!0}),y(We)!=="svelte-1sal4ui"&&(We.innerHTML=oo),Wt=r(Je),h(E.$$.fragment,Je),Je.forEach(o),xe.forEach(o),ht=r(e),h(ke.$$.fragment,e),ut=r(e),Z=d(e,"DIV",{class:!0});var bt=F(Z);h(we.$$.fragment,bt),It=r(bt),q=d(bt,"DIV",{class:!0});var Ze=F(q);h(Me.$$.fragment,Ze),Jt=r(Ze),Ie=d(Ze,"P",{"data-svelte-h":!0}),y(Ie)!=="svelte-1py4aay"&&(Ie.innerHTML=no),Zt=r(Ze),h(X.$$.fragment,Ze),Ze.forEach(o),bt.forEach(o),ft=r(e),h($e.$$.fragment,e),gt=r(e),Ge=d(e,"P",{}),F(Ge).forEach(o),this.h()},h(){C(s,"name","hf:doc:metadata"),C(s,"content",yo),C(V,"class","flex flex-wrap space-x-1"),C(M,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),C(U,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),C(k,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),C(x,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),C(I,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),C(j,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),C(J,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),C(q,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),C(Z,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,t){c(document.head,s),n(e,v,t),n(e,i,t),n(e,b,t),n(e,T,t),n(e,L,t),u(O,e,t),n(e,Pe,t),n(e,V,t),n(e,Be,t),u(A,e,t),n(e,Re,t),n(e,D,t),n(e,He,t),u(Q,e,t),n(e,Ee,t),n(e,Y,t),n(e,Xe,t),n(e,K,t),n(e,Oe,t),u(ee,e,t),n(e,Ae,t),n(e,te,t),n(e,De,t),n(e,oe,t),n(e,Qe,t),n(e,ne,t),n(e,Ye,t),u(se,e,t),n(e,Ke,t),u(ae,e,t),n(e,et,t),n(e,re,t),n(e,tt,t),u(ie,e,t),n(e,ot,t),n(e,le,t),n(e,nt,t),n(e,de,t),n(e,st,t),u(ce,e,t),n(e,at,t),u(pe,e,t),n(e,rt,t),n(e,M,t),u(me,M,null),c(M,vt),c(M,Ce),c(M,Tt),c(M,Se),c(M,kt),u(P,M,null),n(e,it,t),u(he,e,t),n(e,lt,t),n(e,k,t),u(ue,k,null),c(k,wt),c(k,ze),c(k,Mt),c(k,Fe),c(k,$t),c(k,Ue),c(k,Lt),c(k,U),u(fe,U,null),c(U,xt),c(U,je),c(U,Ct),u(B,U,null),n(e,dt,t),u(ge,e,t),n(e,ct,t),n(e,I,t),u(_e,I,null),c(I,St),c(I,x),u(be,x,null),c(x,zt),c(x,qe),c(x,Ft),u(R,x,null),c(x,Ut),u(H,x,null),n(e,pt,t),u(ye,e,t),n(e,mt,t),n(e,J,t),u(ve,J,null),c(J,jt),c(J,j),u(Te,j,null),c(j,qt),c(j,We),c(j,Wt),u(E,j,null),n(e,ht,t),u(ke,e,t),n(e,ut,t),n(e,Z,t),u(we,Z,null),c(Z,It),c(Z,q),u(Me,q,null),c(q,Jt),c(q,Ie),c(q,Zt),u(X,q,null),n(e,ft,t),u($e,e,t),n(e,gt,t),n(e,Ge,t),_t=!0},p(e,[t]){const S={};t&2&&(S.$$scope={dirty:t,ctx:e}),P.$set(S);const $={};t&2&&($.$$scope={dirty:t,ctx:e}),B.$set($);const N={};t&2&&(N.$$scope={dirty:t,ctx:e}),R.$set(N);const Le={};t&2&&(Le.$$scope={dirty:t,ctx:e}),H.$set(Le);const z={};t&2&&(z.$$scope={dirty:t,ctx:e}),E.$set(z);const xe={};t&2&&(xe.$$scope={dirty:t,ctx:e}),X.$set(xe)},i(e){_t||(f(O.$$.fragment,e),f(A.$$.fragment,e),f(Q.$$.fragment,e),f(ee.$$.fragment,e),f(se.$$.fragment,e),f(ae.$$.fragment,e),f(ie.$$.fragment,e),f(ce.$$.fragment,e),f(pe.$$.fragment,e),f(me.$$.fragment,e),f(P.$$.fragment,e),f(he.$$.fragment,e),f(ue.$$.fragment,e),f(fe.$$.fragment,e),f(B.$$.fragment,e),f(ge.$$.fragment,e),f(_e.$$.fragment,e),f(be.$$.fragment,e),f(R.$$.fragment,e),f(H.$$.fragment,e),f(ye.$$.fragment,e),f(ve.$$.fragment,e),f(Te.$$.fragment,e),f(E.$$.fragment,e),f(ke.$$.fragment,e),f(we.$$.fragment,e),f(Me.$$.fragment,e),f(X.$$.fragment,e),f($e.$$.fragment,e),_t=!0)},o(e){g(O.$$.fragment,e),g(A.$$.fragment,e),g(Q.$$.fragment,e),g(ee.$$.fragment,e),g(se.$$.fragment,e),g(ae.$$.fragment,e),g(ie.$$.fragment,e),g(ce.$$.fragment,e),g(pe.$$.fragment,e),g(me.$$.fragment,e),g(P.$$.fragment,e),g(he.$$.fragment,e),g(ue.$$.fragment,e),g(fe.$$.fragment,e),g(B.$$.fragment,e),g(ge.$$.fragment,e),g(_e.$$.fragment,e),g(be.$$.fragment,e),g(R.$$.fragment,e),g(H.$$.fragment,e),g(ye.$$.fragment,e),g(ve.$$.fragment,e),g(Te.$$.fragment,e),g(E.$$.fragment,e),g(ke.$$.fragment,e),g(we.$$.fragment,e),g(Me.$$.fragment,e),g(X.$$.fragment,e),g($e.$$.fragment,e),_t=!1},d(e){e&&(o(v),o(i),o(b),o(T),o(L),o(Pe),o(V),o(Be),o(Re),o(D),o(He),o(Ee),o(Y),o(Xe),o(K),o(Oe),o(Ae),o(te),o(De),o(oe),o(Qe),o(ne),o(Ye),o(Ke),o(et),o(re),o(tt),o(ot),o(le),o(nt),o(de),o(st),o(at),o(rt),o(M),o(it),o(lt),o(k),o(dt),o(ct),o(I),o(pt),o(mt),o(J),o(ht),o(ut),o(Z),o(ft),o(gt),o(Ge)),o(s),_(O,e),_(A,e),_(Q,e),_(ee,e),_(se,e),_(ae,e),_(ie,e),_(ce,e),_(pe,e),_(me),_(P),_(he,e),_(ue),_(fe),_(B),_(ge,e),_(_e),_(be),_(R),_(H),_(ye,e),_(ve),_(Te),_(E),_(ke,e),_(we),_(Me),_(X),_($e,e)}}}const yo='{"title":"StableLM","local":"stablelm","sections":[{"title":"Overview","local":"overview","sections":[{"title":"Model Details","local":"model-details","sections":[],"depth":3},{"title":"Usage Tips","local":"usage-tips","sections":[],"depth":3}],"depth":2},{"title":"Combining StableLM and Flash Attention 2","local":"combining-stablelm-and-flash-attention-2","sections":[],"depth":2},{"title":"StableLmConfig","local":"transformers.StableLmConfig","sections":[],"depth":2},{"title":"StableLmModel","local":"transformers.StableLmModel","sections":[],"depth":2},{"title":"StableLmForCausalLM","local":"transformers.StableLmForCausalLM","sections":[],"depth":2},{"title":"StableLmForSequenceClassification","local":"transformers.StableLmForSequenceClassification","sections":[],"depth":2},{"title":"StableLmForTokenClassification","local":"transformers.StableLmForTokenClassification","sections":[],"depth":2}],"depth":1}';function vo(w){return ro(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class Co extends io{constructor(s){super(),lo(this,s,vo,bo,ao,{})}}export{Co as component};
