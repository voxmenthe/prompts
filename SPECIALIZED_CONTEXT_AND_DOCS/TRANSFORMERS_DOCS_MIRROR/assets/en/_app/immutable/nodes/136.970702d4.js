import{s as Ct,o as jt,n as Le}from"../chunks/scheduler.18a86fab.js";import{S as $t,i as It,g as l,s,r as h,A as Dt,h as d,f as o,c as a,j as le,x as m,u as f,k as X,y as c,a as n,v as g,d as b,t as _,w as y}from"../chunks/index.98837b22.js";import{T as kt}from"../chunks/Tip.77304350.js";import{D as Te}from"../chunks/Docstring.a1ef7999.js";import{C as we}from"../chunks/CodeBlock.8d0c2e8a.js";import{E as Ut}from"../chunks/ExampleCodeBlock.8c3ee1f9.js";import{H as ye,E as Zt}from"../chunks/getInferenceSnippets.06c2775f.js";function Ft($){let r,T="Example:",p,u,M;return u=new we({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMERicnhDb25maWclMkMlMjBEYnJ4TW9kZWwlMEElMEElMjMlMjBJbml0aWFsaXppbmclMjBhJTIwRGJyeCUyMGNvbmZpZ3VyYXRpb24lMEFjb25maWd1cmF0aW9uJTIwJTNEJTIwRGJyeENvbmZpZyhuX2xheWVycyUzRDIlMkMlMjBkX21vZGVsJTNEMjU2JTJDJTIwbl9oZWFkcyUzRDglMkMlMjB2b2NhYl9zaXplJTNEMTI4KSUwQSUwQSUyMyUyMEluaXRpYWxpemluZyUyMGElMjBtb2RlbCUyMCh3aXRoJTIwcmFuZG9tJTIwd2VpZ2h0cyklMjBmcm9tJTIwdGhlJTIwY29uZmlndXJhdGlvbiUwQW1vZGVsJTIwJTNEJTIwRGJyeE1vZGVsKGNvbmZpZ3VyYXRpb24pJTBBJTBBJTIzJTIwQWNjZXNzaW5nJTIwdGhlJTIwbW9kZWwlMjBjb25maWd1cmF0aW9uJTBBY29uZmlndXJhdGlvbiUyMCUzRCUyMG1vZGVsLmNvbmZpZw==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> DbrxConfig, DbrxModel

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a Dbrx configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = DbrxConfig(n_layers=<span class="hljs-number">2</span>, d_model=<span class="hljs-number">256</span>, n_heads=<span class="hljs-number">8</span>, vocab_size=<span class="hljs-number">128</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a model (with random weights) from the configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = DbrxModel(configuration)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Accessing the model configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = model.config`,wrap:!1}}),{c(){r=l("p"),r.textContent=T,p=s(),h(u.$$.fragment)},l(i){r=d(i,"P",{"data-svelte-h":!0}),m(r)!=="svelte-11lpom8"&&(r.textContent=T),p=a(i),f(u.$$.fragment,i)},m(i,U){n(i,r,U),n(i,p,U),g(u,i,U),M=!0},p:Le,i(i){M||(b(u.$$.fragment,i),M=!0)},o(i){_(u.$$.fragment,i),M=!1},d(i){i&&(o(r),o(p)),y(u,i)}}}function Rt($){let r,T=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){r=l("p"),r.innerHTML=T},l(p){r=d(p,"P",{"data-svelte-h":!0}),m(r)!=="svelte-fincs2"&&(r.innerHTML=T)},m(p,u){n(p,r,u)},p:Le,d(p){p&&o(r)}}}function zt($){let r,T=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){r=l("p"),r.innerHTML=T},l(p){r=d(p,"P",{"data-svelte-h":!0}),m(r)!=="svelte-fincs2"&&(r.innerHTML=T)},m(p,u){n(p,r,u)},p:Le,d(p){p&&o(r)}}}function Gt($){let r,T="Example:",p,u,M;return u=new we({props:{code:"JTNFJTNFJTIwZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBEYnJ4Rm9yQ2F1c2FsTE0lMEElMEElM0UlM0UlMjBtb2RlbCUyMCUzRCUyMERicnhGb3JDYXVzYWxMTS5mcm9tX3ByZXRyYWluZWQoJTIyZGF0YWJyaWNrcyUyRmRicngtaW5zdHJ1Y3QlMjIpJTBBJTNFJTNFJTIwdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIyZGF0YWJyaWNrcyUyRmRicngtaW5zdHJ1Y3QlMjIpJTBBJTBBJTNFJTNFJTIwcHJvbXB0JTIwJTNEJTIwJTIySGV5JTJDJTIwYXJlJTIweW91JTIwY29uc2Npb3VzJTNGJTIwQ2FuJTIweW91JTIwdGFsayUyMHRvJTIwbWUlM0YlMjIlMEElM0UlM0UlMjBpbnB1dHMlMjAlM0QlMjB0b2tlbml6ZXIocHJvbXB0JTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiklMEElMEElM0UlM0UlMjAlMjMlMjBHZW5lcmF0ZSUwQSUzRSUzRSUyMGdlbmVyYXRlX2lkcyUyMCUzRCUyMG1vZGVsLmdlbmVyYXRlKGlucHV0cy5pbnB1dF9pZHMlMkMlMjBtYXhfbGVuZ3RoJTNEMzApJTBBJTNFJTNFJTIwdG9rZW5pemVyLmJhdGNoX2RlY29kZShnZW5lcmF0ZV9pZHMlMkMlMjBza2lwX3NwZWNpYWxfdG9rZW5zJTNEVHJ1ZSUyQyUyMGNsZWFuX3VwX3Rva2VuaXphdGlvbl9zcGFjZXMlM0RGYWxzZSklNUIwJTVEJTBBJTIySGV5JTJDJTIwYXJlJTIweW91JTIwY29uc2Npb3VzJTNGJTIwQ2FuJTIweW91JTIwdGFsayUyMHRvJTIwbWUlM0YlNUNuSSdtJTIwbm90JTIwY29uc2Npb3VzJTJDJTIwYnV0JTIwSSUyMGNhbiUyMHRhbGslMjB0byUyMHlvdS4lMjI=",highlighted:`&gt;&gt; <span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, DbrxForCausalLM

&gt;&gt; model = DbrxForCausalLM.from_pretrained(<span class="hljs-string">&quot;databricks/dbrx-instruct&quot;</span>)
&gt;&gt; tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;databricks/dbrx-instruct&quot;</span>)

&gt;&gt; prompt = <span class="hljs-string">&quot;Hey, are you conscious? Can you talk to me?&quot;</span>
&gt;&gt; inputs = tokenizer(prompt, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

&gt;&gt; <span class="hljs-comment"># Generate</span>
&gt;&gt; generate_ids = model.generate(inputs.input_ids, max_length=<span class="hljs-number">30</span>)
&gt;&gt; tokenizer.batch_decode(generate_ids, skip_special_tokens=<span class="hljs-literal">True</span>, clean_up_tokenization_spaces=<span class="hljs-literal">False</span>)[<span class="hljs-number">0</span>]
<span class="hljs-string">&quot;Hey, are you conscious? Can you talk to me?\\nI&#x27;m not conscious, but I can talk to you.&quot;</span>`,wrap:!1}}),{c(){r=l("p"),r.textContent=T,p=s(),h(u.$$.fragment)},l(i){r=d(i,"P",{"data-svelte-h":!0}),m(r)!=="svelte-11lpom8"&&(r.textContent=T),p=a(i),f(u.$$.fragment,i)},m(i,U){n(i,r,U),n(i,p,U),g(u,i,U),M=!0},p:Le,i(i){M||(b(u.$$.fragment,i),M=!0)},o(i){_(u.$$.fragment,i),M=!1},d(i){i&&(o(r),o(p)),y(u,i)}}}function Wt($){let r,T,p,u,M,i="<em>This model was released on 2024-03-27 and added to Hugging Face Transformers on 2024-04-18.</em>",U,N,xe,Z,lt='<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/> <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat"/> <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/>',ve,q,Je,E,dt=`DBRX is a <a href="https://www.isattentionallyouneed.com/" rel="nofollow">transformer-based</a> decoder-only large language model (LLM) that was trained using next-token prediction.
It uses a <em>fine-grained</em> mixture-of-experts (MoE) architecture with 132B total parameters of which 36B parameters are active on any input.
It was pre-trained on 12T tokens of text and code data.
Compared to other open MoE models like Mixtral-8x7B and Grok-1, DBRX is fine-grained, meaning it uses a larger number of smaller experts. DBRX has 16 experts and chooses 4, while Mixtral-8x7B and Grok-1 have 8 experts and choose 2.
This provides 65x more possible combinations of experts and we found that this improves model quality.
DBRX uses rotary position encodings (RoPE), gated linear units (GLU), and grouped query attention (GQA).
It is a BPE based model and uses the GPT-4 tokenizer as described in the <a href="https://github.com/openai/tiktoken" rel="nofollow">tiktoken</a> repository.
We made these choices based on exhaustive evaluation and scaling experiments.`,ke,V,ct=`DBRX was pretrained on 12T tokens of carefully curated data and a maximum context length of 32K tokens.
We estimate that this data is at least 2x better token-for-token than the data we used to pretrain the MPT family of models.
This new dataset was developed using the full suite of Databricks tools, including Apache Spark™ and Databricks notebooks for data processing, and Unity Catalog for data management and governance.
We used curriculum learning for pretraining, changing the data mix during training in ways we found to substantially improve model quality.`,Ue,B,pt='More detailed information about DBRX Instruct and DBRX Base can be found in our <a href="https://www.databricks.com/blog/introducing-dbrx-new-state-art-open-llm" rel="nofollow">technical blog post</a>.',Ce,L,ut='This model was contributed by <a href="https://huggingface.co/eitanturok" rel="nofollow">eitan-turok</a> and <a href="https://huggingface.co/abhi-db" rel="nofollow">abhi-db</a>. The original code can be found <a href="https://github.com/databricks/dbrx-instruct" rel="nofollow">here</a>, though this may not be up to date.',je,H,$e,Q,mt="The <code>generate()</code> method can be used to generate text using DBRX. You can generate using the standard attention implementation, flash-attention, and the PyTorch scaled dot product attention. The last two attention implementations give speed ups.",Ie,S,De,Y,ht='If you have flash-attention installed (<code>pip install flash-attn</code>), it is possible to generate faster. (The HuggingFace documentation for flash-attention can be found <a href="https://huggingface.co/docs/transformers/perf_infer_gpu_one#flashattention-2" rel="nofollow">here</a>.)',Ze,P,Fe,A,ft='You can also generate faster using the PyTorch scaled dot product attention. (The HuggingFace documentation for scaled dot product attention can be found <a href="https://huggingface.co/docs/transformers/perf_infer_gpu_one#pytorch-scaled-dot-product-attention" rel="nofollow">here</a>.)',Re,O,ze,K,Ge,v,ee,He,de,gt=`This is the configuration class to store the configuration of a <a href="/docs/transformers/v4.56.2/en/model_doc/dbrx#transformers.DbrxModel">DbrxModel</a>. It is used to instantiate a Dbrx model according to the
specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a different configuration to that of the <a href="https://huggingface.co/databricks/dbrx-instruct" rel="nofollow">databricks/dbrx-instruct</a> architecture.`,Qe,ce,bt=`Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> and can be used to control the model outputs. Read the
documentation from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> for more information.`,Se,F,We,te,Xe,w,oe,Ye,pe,_t="The bare Dbrx Model outputting raw hidden-states without any specific head on top.",Pe,ue,yt=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Ae,me,Mt=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Oe,I,ne,Ke,he,Tt='The <a href="/docs/transformers/v4.56.2/en/model_doc/dbrx#transformers.DbrxModel">DbrxModel</a> forward method, overrides the <code>__call__</code> special method.',et,R,Ne,se,qe,x,ae,tt,fe,wt="The DBRX Model transformer for causal language modeling.",ot,ge,xt=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,nt,be,vt=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,st,C,re,at,_e,Jt='The <a href="/docs/transformers/v4.56.2/en/model_doc/dbrx#transformers.DbrxForCausalLM">DbrxForCausalLM</a> forward method, overrides the <code>__call__</code> special method.',rt,z,it,G,Ee,ie,Ve,Me,Be;return N=new ye({props:{title:"DBRX",local:"dbrx",headingTag:"h1"}}),q=new ye({props:{title:"Overview",local:"overview",headingTag:"h2"}}),H=new ye({props:{title:"Usage Examples",local:"usage-examples",headingTag:"h2"}}),S=new we({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMERicnhGb3JDYXVzYWxMTSUyQyUyMEF1dG9Ub2tlbml6ZXIlMEFpbXBvcnQlMjB0b3JjaCUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMmRhdGFicmlja3MlMkZkYnJ4LWluc3RydWN0JTIyJTJDJTIwdG9rZW4lM0QlMjJZT1VSX0hGX1RPS0VOJTIyKSUwQW1vZGVsJTIwJTNEJTIwRGJyeEZvckNhdXNhbExNLmZyb21fcHJldHJhaW5lZCglMEElMjAlMjAlMjAlMjAlMjJkYXRhYnJpY2tzJTJGZGJyeC1pbnN0cnVjdCUyMiUyQyUwQSUyMCUyMCUyMCUyMGRldmljZV9tYXAlM0QlMjJhdXRvJTIyJTJDJTBBJTIwJTIwJTIwJTIwZHR5cGUlM0R0b3JjaC5iZmxvYXQxNiUyQyUwQSUyMCUyMCUyMCUyMHRva2VuJTNEJTIyWU9VUl9IRl9UT0tFTiUyMiUyQyUwQSUyMCUyMCUyMCUyMCklMEElMEFpbnB1dF90ZXh0JTIwJTNEJTIwJTIyV2hhdCUyMGRvZXMlMjBpdCUyMHRha2UlMjB0byUyMGJ1aWxkJTIwYSUyMGdyZWF0JTIwTExNJTNGJTIyJTBBbWVzc2FnZXMlMjAlM0QlMjAlNUIlN0IlMjJyb2xlJTIyJTNBJTIwJTIydXNlciUyMiUyQyUyMCUyMmNvbnRlbnQlMjIlM0ElMjBpbnB1dF90ZXh0JTdEJTVEJTBBaW5wdXRfaWRzJTIwJTNEJTIwdG9rZW5pemVyLmFwcGx5X2NoYXRfdGVtcGxhdGUobWVzc2FnZXMlMkMlMjByZXR1cm5fZGljdCUzRFRydWUlMkMlMjB0b2tlbml6ZSUzRFRydWUlMkMlMjBhZGRfZ2VuZXJhdGlvbl9wcm9tcHQlM0RUcnVlJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMikudG8obW9kZWwuZGV2aWNlKSUwQSUwQW91dHB1dHMlMjAlM0QlMjBtb2RlbC5nZW5lcmF0ZSgqKmlucHV0X2lkcyUyQyUyMG1heF9uZXdfdG9rZW5zJTNEMjAwKSUwQXByaW50KHRva2VuaXplci5kZWNvZGUob3V0cHV0cyU1QjAlNUQpKQ==",highlighted:`<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> DbrxForCausalLM, AutoTokenizer
<span class="hljs-keyword">import</span> torch

tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;databricks/dbrx-instruct&quot;</span>, token=<span class="hljs-string">&quot;YOUR_HF_TOKEN&quot;</span>)
model = DbrxForCausalLM.from_pretrained(
    <span class="hljs-string">&quot;databricks/dbrx-instruct&quot;</span>,
    device_map=<span class="hljs-string">&quot;auto&quot;</span>,
    dtype=torch.bfloat16,
    token=<span class="hljs-string">&quot;YOUR_HF_TOKEN&quot;</span>,
    )

input_text = <span class="hljs-string">&quot;What does it take to build a great LLM?&quot;</span>
messages = [{<span class="hljs-string">&quot;role&quot;</span>: <span class="hljs-string">&quot;user&quot;</span>, <span class="hljs-string">&quot;content&quot;</span>: input_text}]
input_ids = tokenizer.apply_chat_template(messages, return_dict=<span class="hljs-literal">True</span>, tokenize=<span class="hljs-literal">True</span>, add_generation_prompt=<span class="hljs-literal">True</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>).to(model.device)

outputs = model.generate(**input_ids, max_new_tokens=<span class="hljs-number">200</span>)
<span class="hljs-built_in">print</span>(tokenizer.decode(outputs[<span class="hljs-number">0</span>]))`,wrap:!1}}),P=new we({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMERicnhGb3JDYXVzYWxMTSUyQyUyMEF1dG9Ub2tlbml6ZXIlMEFpbXBvcnQlMjB0b3JjaCUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMmRhdGFicmlja3MlMkZkYnJ4LWluc3RydWN0JTIyJTJDJTIwdG9rZW4lM0QlMjJZT1VSX0hGX1RPS0VOJTIyKSUwQW1vZGVsJTIwJTNEJTIwRGJyeEZvckNhdXNhbExNLmZyb21fcHJldHJhaW5lZCglMEElMjAlMjAlMjAlMjAlMjJkYXRhYnJpY2tzJTJGZGJyeC1pbnN0cnVjdCUyMiUyQyUwQSUyMCUyMCUyMCUyMGRldmljZV9tYXAlM0QlMjJhdXRvJTIyJTJDJTBBJTIwJTIwJTIwJTIwZHR5cGUlM0R0b3JjaC5iZmxvYXQxNiUyQyUwQSUyMCUyMCUyMCUyMHRva2VuJTNEJTIyWU9VUl9IRl9UT0tFTiUyMiUyQyUwQSUyMCUyMCUyMCUyMGF0dG5faW1wbGVtZW50YXRpb24lM0QlMjJmbGFzaF9hdHRlbnRpb25fMiUyMiUyQyUwQSUyMCUyMCUyMCUyMCklMEElMEFpbnB1dF90ZXh0JTIwJTNEJTIwJTIyV2hhdCUyMGRvZXMlMjBpdCUyMHRha2UlMjB0byUyMGJ1aWxkJTIwYSUyMGdyZWF0JTIwTExNJTNGJTIyJTBBbWVzc2FnZXMlMjAlM0QlMjAlNUIlN0IlMjJyb2xlJTIyJTNBJTIwJTIydXNlciUyMiUyQyUyMCUyMmNvbnRlbnQlMjIlM0ElMjBpbnB1dF90ZXh0JTdEJTVEJTBBaW5wdXRfaWRzJTIwJTNEJTIwdG9rZW5pemVyLmFwcGx5X2NoYXRfdGVtcGxhdGUobWVzc2FnZXMlMkMlMjByZXR1cm5fZGljdCUzRFRydWUlMkMlMjB0b2tlbml6ZSUzRFRydWUlMkMlMjBhZGRfZ2VuZXJhdGlvbl9wcm9tcHQlM0RUcnVlJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMikudG8obW9kZWwuZGV2aWNlKSUwQSUwQW91dHB1dHMlMjAlM0QlMjBtb2RlbC5nZW5lcmF0ZSgqKmlucHV0X2lkcyUyQyUyMG1heF9uZXdfdG9rZW5zJTNEMjAwKSUwQXByaW50KHRva2VuaXplci5kZWNvZGUob3V0cHV0cyU1QjAlNUQpKQ==",highlighted:`<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> DbrxForCausalLM, AutoTokenizer
<span class="hljs-keyword">import</span> torch

tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;databricks/dbrx-instruct&quot;</span>, token=<span class="hljs-string">&quot;YOUR_HF_TOKEN&quot;</span>)
model = DbrxForCausalLM.from_pretrained(
    <span class="hljs-string">&quot;databricks/dbrx-instruct&quot;</span>,
    device_map=<span class="hljs-string">&quot;auto&quot;</span>,
    dtype=torch.bfloat16,
    token=<span class="hljs-string">&quot;YOUR_HF_TOKEN&quot;</span>,
    attn_implementation=<span class="hljs-string">&quot;flash_attention_2&quot;</span>,
    )

input_text = <span class="hljs-string">&quot;What does it take to build a great LLM?&quot;</span>
messages = [{<span class="hljs-string">&quot;role&quot;</span>: <span class="hljs-string">&quot;user&quot;</span>, <span class="hljs-string">&quot;content&quot;</span>: input_text}]
input_ids = tokenizer.apply_chat_template(messages, return_dict=<span class="hljs-literal">True</span>, tokenize=<span class="hljs-literal">True</span>, add_generation_prompt=<span class="hljs-literal">True</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>).to(model.device)

outputs = model.generate(**input_ids, max_new_tokens=<span class="hljs-number">200</span>)
<span class="hljs-built_in">print</span>(tokenizer.decode(outputs[<span class="hljs-number">0</span>]))`,wrap:!1}}),O=new we({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMERicnhGb3JDYXVzYWxMTSUyQyUyMEF1dG9Ub2tlbml6ZXIlMEFpbXBvcnQlMjB0b3JjaCUwQSUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMmRhdGFicmlja3MlMkZkYnJ4LWluc3RydWN0JTIyJTJDJTIwdG9rZW4lM0QlMjJZT1VSX0hGX1RPS0VOJTIyKSUwQW1vZGVsJTIwJTNEJTIwRGJyeEZvckNhdXNhbExNLmZyb21fcHJldHJhaW5lZCglMEElMjAlMjAlMjAlMjAlMjJkYXRhYnJpY2tzJTJGZGJyeC1pbnN0cnVjdCUyMiUyQyUwQSUyMCUyMCUyMCUyMGRldmljZV9tYXAlM0QlMjJhdXRvJTIyJTJDJTBBJTIwJTIwJTIwJTIwZHR5cGUlM0R0b3JjaC5iZmxvYXQxNiUyQyUwQSUyMCUyMCUyMCUyMHRva2VuJTNEJTIyWU9VUl9IRl9UT0tFTiUyMiUyQyUwQSUyMCUyMCUyMCUyMGF0dG5faW1wbGVtZW50YXRpb24lM0QlMjJzZHBhJTIyJTJDJTBBJTIwJTIwJTIwJTIwKSUwQSUwQWlucHV0X3RleHQlMjAlM0QlMjAlMjJXaGF0JTIwZG9lcyUyMGl0JTIwdGFrZSUyMHRvJTIwYnVpbGQlMjBhJTIwZ3JlYXQlMjBMTE0lM0YlMjIlMEFtZXNzYWdlcyUyMCUzRCUyMCU1QiU3QiUyMnJvbGUlMjIlM0ElMjAlMjJ1c2VyJTIyJTJDJTIwJTIyY29udGVudCUyMiUzQSUyMGlucHV0X3RleHQlN0QlNUQlMEFpbnB1dF9pZHMlMjAlM0QlMjB0b2tlbml6ZXIuYXBwbHlfY2hhdF90ZW1wbGF0ZShtZXNzYWdlcyUyQyUyMHJldHVybl9kaWN0JTNEVHJ1ZSUyQyUyMHRva2VuaXplJTNEVHJ1ZSUyQyUyMGFkZF9nZW5lcmF0aW9uX3Byb21wdCUzRFRydWUlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKS50byhtb2RlbC5kZXZpY2UpJTBBJTBBb3V0cHV0cyUyMCUzRCUyMG1vZGVsLmdlbmVyYXRlKCoqaW5wdXRfaWRzJTJDJTIwbWF4X25ld190b2tlbnMlM0QyMDApJTBBcHJpbnQodG9rZW5pemVyLmRlY29kZShvdXRwdXRzJTVCMCU1RCkp",highlighted:`<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> DbrxForCausalLM, AutoTokenizer
<span class="hljs-keyword">import</span> torch

tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;databricks/dbrx-instruct&quot;</span>, token=<span class="hljs-string">&quot;YOUR_HF_TOKEN&quot;</span>)
model = DbrxForCausalLM.from_pretrained(
    <span class="hljs-string">&quot;databricks/dbrx-instruct&quot;</span>,
    device_map=<span class="hljs-string">&quot;auto&quot;</span>,
    dtype=torch.bfloat16,
    token=<span class="hljs-string">&quot;YOUR_HF_TOKEN&quot;</span>,
    attn_implementation=<span class="hljs-string">&quot;sdpa&quot;</span>,
    )

input_text = <span class="hljs-string">&quot;What does it take to build a great LLM?&quot;</span>
messages = [{<span class="hljs-string">&quot;role&quot;</span>: <span class="hljs-string">&quot;user&quot;</span>, <span class="hljs-string">&quot;content&quot;</span>: input_text}]
input_ids = tokenizer.apply_chat_template(messages, return_dict=<span class="hljs-literal">True</span>, tokenize=<span class="hljs-literal">True</span>, add_generation_prompt=<span class="hljs-literal">True</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>).to(model.device)

outputs = model.generate(**input_ids, max_new_tokens=<span class="hljs-number">200</span>)
<span class="hljs-built_in">print</span>(tokenizer.decode(outputs[<span class="hljs-number">0</span>]))`,wrap:!1}}),K=new ye({props:{title:"DbrxConfig",local:"transformers.DbrxConfig",headingTag:"h2"}}),ee=new Te({props:{name:"class transformers.DbrxConfig",anchor:"transformers.DbrxConfig",parameters:[{name:"d_model",val:": int = 2048"},{name:"n_heads",val:": int = 16"},{name:"n_layers",val:": int = 24"},{name:"max_seq_len",val:": int = 2048"},{name:"vocab_size",val:": int = 32000"},{name:"resid_pdrop",val:": float = 0.0"},{name:"emb_pdrop",val:": float = 0.0"},{name:"attn_config",val:": typing.Optional[transformers.models.dbrx.configuration_dbrx.DbrxAttentionConfig] = None"},{name:"ffn_config",val:": typing.Optional[transformers.models.dbrx.configuration_dbrx.DbrxFFNConfig] = None"},{name:"use_cache",val:": bool = True"},{name:"initializer_range",val:": float = 0.02"},{name:"output_router_logits",val:": bool = False"},{name:"**kwargs",val:": typing.Any"}],parametersDescription:[{anchor:"transformers.DbrxConfig.d_model",description:`<strong>d_model</strong> (<code>int</code>, <em>optional</em>, defaults to 2048) &#x2014;
Dimensionality of the embeddings and hidden states.`,name:"d_model"},{anchor:"transformers.DbrxConfig.n_heads",description:`<strong>n_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 16) &#x2014;
Number of attention heads for each attention layer in the Transformer encoder.`,name:"n_heads"},{anchor:"transformers.DbrxConfig.n_layers",description:`<strong>n_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 24) &#x2014;
Number of hidden layers in the Transformer encoder.`,name:"n_layers"},{anchor:"transformers.DbrxConfig.max_seq_len",description:`<strong>max_seq_len</strong> (<code>int</code>, <em>optional</em>, defaults to 2048) &#x2014;
The maximum sequence length of the model.`,name:"max_seq_len"},{anchor:"transformers.DbrxConfig.vocab_size",description:`<strong>vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 32000) &#x2014;
Vocabulary size of the Dbrx model. Defines the maximum number of different tokens that can be represented by
the <code>inputs_ids</code> passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/dbrx#transformers.DbrxModel">DbrxModel</a>.`,name:"vocab_size"},{anchor:"transformers.DbrxConfig.resid_pdrop",description:`<strong>resid_pdrop</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The dropout probability applied to the attention output before combining with residual.`,name:"resid_pdrop"},{anchor:"transformers.DbrxConfig.emb_pdrop",description:`<strong>emb_pdrop</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The dropout probability for the embedding layer.`,name:"emb_pdrop"},{anchor:"transformers.DbrxConfig.attn_config",description:`<strong>attn_config</strong> (<code>dict</code>, <em>optional</em>) &#x2014;
A dictionary used to configure the model&#x2019;s attention module.`,name:"attn_config"},{anchor:"transformers.DbrxConfig.ffn_config",description:`<strong>ffn_config</strong> (<code>dict</code>, <em>optional</em>) &#x2014;
A dictionary used to configure the model&#x2019;s FFN module.`,name:"ffn_config"},{anchor:"transformers.DbrxConfig.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not the model should return the last key/values attentions (not used by all models).`,name:"use_cache"},{anchor:"transformers.DbrxConfig.initializer_range",description:`<strong>initializer_range</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.`,name:"initializer_range"},{anchor:"transformers.DbrxConfig.output_router_logits",description:`<strong>output_router_logits</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not the router logits should be returned by the model. Enabling this will also
allow the model to output the auxiliary loss. See <a href>here</a> for more details.`,name:"output_router_logits"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/dbrx/configuration_dbrx.py#L119"}}),F=new Ut({props:{anchor:"transformers.DbrxConfig.example",$$slots:{default:[Ft]},$$scope:{ctx:$}}}),te=new ye({props:{title:"DbrxModel",local:"transformers.DbrxModel",headingTag:"h2"}}),oe=new Te({props:{name:"class transformers.DbrxModel",anchor:"transformers.DbrxModel",parameters:[{name:"config",val:": DbrxConfig"}],parametersDescription:[{anchor:"transformers.DbrxModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/dbrx#transformers.DbrxConfig">DbrxConfig</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/dbrx/modeling_dbrx.py#L844"}}),ne=new Te({props:{name:"forward",anchor:"transformers.DbrxModel.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"output_router_logits",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.DbrxModel.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.DbrxModel.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.DbrxModel.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.DbrxModel.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.DbrxModel.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.DbrxModel.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.DbrxModel.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.DbrxModel.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.DbrxModel.forward.output_router_logits",description:`<strong>output_router_logits</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the logits of all the routers. They are useful for computing the router loss, and
should not be returned during inference.`,name:"output_router_logits"},{anchor:"transformers.DbrxModel.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.DbrxModel.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/dbrx/modeling_dbrx.py#L873",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <code>transformers.modeling_outputs.MoeModelOutputWithPast</code> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/dbrx#transformers.DbrxConfig"
>DbrxConfig</a>) and inputs.</p>
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
`}}),R=new kt({props:{$$slots:{default:[Rt]},$$scope:{ctx:$}}}),se=new ye({props:{title:"DbrxForCausalLM",local:"transformers.DbrxForCausalLM",headingTag:"h2"}}),ae=new Te({props:{name:"class transformers.DbrxForCausalLM",anchor:"transformers.DbrxForCausalLM",parameters:[{name:"config",val:": DbrxConfig"}],parametersDescription:[{anchor:"transformers.DbrxForCausalLM.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/dbrx#transformers.DbrxConfig">DbrxConfig</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/dbrx/modeling_dbrx.py#L1115"}}),re=new Te({props:{name:"forward",anchor:"transformers.DbrxForCausalLM.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"output_router_logits",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"},{name:"logits_to_keep",val:": typing.Union[int, torch.Tensor] = 0"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.DbrxForCausalLM.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.DbrxForCausalLM.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.DbrxForCausalLM.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.DbrxForCausalLM.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.DbrxForCausalLM.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.DbrxForCausalLM.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.DbrxForCausalLM.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.DbrxForCausalLM.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.DbrxForCausalLM.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.DbrxForCausalLM.forward.output_router_logits",description:`<strong>output_router_logits</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the logits of all the routers. They are useful for computing the router loss, and
should not be returned during inference.`,name:"output_router_logits"},{anchor:"transformers.DbrxForCausalLM.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.DbrxForCausalLM.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"},{anchor:"transformers.DbrxForCausalLM.forward.logits_to_keep",description:`<strong>logits_to_keep</strong> (<code>Union[int, torch.Tensor]</code>, defaults to <code>0</code>) &#x2014;
If an <code>int</code>, compute logits for the last <code>logits_to_keep</code> tokens. If <code>0</code>, calculate logits for all
<code>input_ids</code> (special case). Only last token logits are needed for generation, and calculating them only for that
token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
If a <code>torch.Tensor</code>, must be 1D corresponding to the indices to keep in the sequence length dimension.
This is useful when using packed tensor format (single dimension for batch and sequence length).`,name:"logits_to_keep"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/dbrx/modeling_dbrx.py#L1146",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <code>transformers.modeling_outputs.MoeCausalLMOutputWithPast</code> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/dbrx#transformers.DbrxConfig"
>DbrxConfig</a>) and inputs.</p>
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
`}}),z=new kt({props:{$$slots:{default:[zt]},$$scope:{ctx:$}}}),G=new Ut({props:{anchor:"transformers.DbrxForCausalLM.forward.example",$$slots:{default:[Gt]},$$scope:{ctx:$}}}),ie=new Zt({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/dbrx.md"}}),{c(){r=l("meta"),T=s(),p=l("p"),u=s(),M=l("p"),M.innerHTML=i,U=s(),h(N.$$.fragment),xe=s(),Z=l("div"),Z.innerHTML=lt,ve=s(),h(q.$$.fragment),Je=s(),E=l("p"),E.innerHTML=dt,ke=s(),V=l("p"),V.textContent=ct,Ue=s(),B=l("p"),B.innerHTML=pt,Ce=s(),L=l("p"),L.innerHTML=ut,je=s(),h(H.$$.fragment),$e=s(),Q=l("p"),Q.innerHTML=mt,Ie=s(),h(S.$$.fragment),De=s(),Y=l("p"),Y.innerHTML=ht,Ze=s(),h(P.$$.fragment),Fe=s(),A=l("p"),A.innerHTML=ft,Re=s(),h(O.$$.fragment),ze=s(),h(K.$$.fragment),Ge=s(),v=l("div"),h(ee.$$.fragment),He=s(),de=l("p"),de.innerHTML=gt,Qe=s(),ce=l("p"),ce.innerHTML=bt,Se=s(),h(F.$$.fragment),We=s(),h(te.$$.fragment),Xe=s(),w=l("div"),h(oe.$$.fragment),Ye=s(),pe=l("p"),pe.textContent=_t,Pe=s(),ue=l("p"),ue.innerHTML=yt,Ae=s(),me=l("p"),me.innerHTML=Mt,Oe=s(),I=l("div"),h(ne.$$.fragment),Ke=s(),he=l("p"),he.innerHTML=Tt,et=s(),h(R.$$.fragment),Ne=s(),h(se.$$.fragment),qe=s(),x=l("div"),h(ae.$$.fragment),tt=s(),fe=l("p"),fe.textContent=wt,ot=s(),ge=l("p"),ge.innerHTML=xt,nt=s(),be=l("p"),be.innerHTML=vt,st=s(),C=l("div"),h(re.$$.fragment),at=s(),_e=l("p"),_e.innerHTML=Jt,rt=s(),h(z.$$.fragment),it=s(),h(G.$$.fragment),Ee=s(),h(ie.$$.fragment),Ve=s(),Me=l("p"),this.h()},l(e){const t=Dt("svelte-u9bgzb",document.head);r=d(t,"META",{name:!0,content:!0}),t.forEach(o),T=a(e),p=d(e,"P",{}),le(p).forEach(o),u=a(e),M=d(e,"P",{"data-svelte-h":!0}),m(M)!=="svelte-103x02p"&&(M.innerHTML=i),U=a(e),f(N.$$.fragment,e),xe=a(e),Z=d(e,"DIV",{class:!0,"data-svelte-h":!0}),m(Z)!=="svelte-b95w5j"&&(Z.innerHTML=lt),ve=a(e),f(q.$$.fragment,e),Je=a(e),E=d(e,"P",{"data-svelte-h":!0}),m(E)!=="svelte-14fqb4b"&&(E.innerHTML=dt),ke=a(e),V=d(e,"P",{"data-svelte-h":!0}),m(V)!=="svelte-6tnpsh"&&(V.textContent=ct),Ue=a(e),B=d(e,"P",{"data-svelte-h":!0}),m(B)!=="svelte-p78060"&&(B.innerHTML=pt),Ce=a(e),L=d(e,"P",{"data-svelte-h":!0}),m(L)!=="svelte-1y0gtxp"&&(L.innerHTML=ut),je=a(e),f(H.$$.fragment,e),$e=a(e),Q=d(e,"P",{"data-svelte-h":!0}),m(Q)!=="svelte-1n2ymew"&&(Q.innerHTML=mt),Ie=a(e),f(S.$$.fragment,e),De=a(e),Y=d(e,"P",{"data-svelte-h":!0}),m(Y)!=="svelte-45m1hv"&&(Y.innerHTML=ht),Ze=a(e),f(P.$$.fragment,e),Fe=a(e),A=d(e,"P",{"data-svelte-h":!0}),m(A)!=="svelte-1cyfh2c"&&(A.innerHTML=ft),Re=a(e),f(O.$$.fragment,e),ze=a(e),f(K.$$.fragment,e),Ge=a(e),v=d(e,"DIV",{class:!0});var j=le(v);f(ee.$$.fragment,j),He=a(j),de=d(j,"P",{"data-svelte-h":!0}),m(de)!=="svelte-1v2qxmx"&&(de.innerHTML=gt),Qe=a(j),ce=d(j,"P",{"data-svelte-h":!0}),m(ce)!=="svelte-1ek1ss9"&&(ce.innerHTML=bt),Se=a(j),f(F.$$.fragment,j),j.forEach(o),We=a(e),f(te.$$.fragment,e),Xe=a(e),w=d(e,"DIV",{class:!0});var J=le(w);f(oe.$$.fragment,J),Ye=a(J),pe=d(J,"P",{"data-svelte-h":!0}),m(pe)!=="svelte-1msk8jg"&&(pe.textContent=_t),Pe=a(J),ue=d(J,"P",{"data-svelte-h":!0}),m(ue)!=="svelte-q52n56"&&(ue.innerHTML=yt),Ae=a(J),me=d(J,"P",{"data-svelte-h":!0}),m(me)!=="svelte-hswkmf"&&(me.innerHTML=Mt),Oe=a(J),I=d(J,"DIV",{class:!0});var D=le(I);f(ne.$$.fragment,D),Ke=a(D),he=d(D,"P",{"data-svelte-h":!0}),m(he)!=="svelte-13g5cnj"&&(he.innerHTML=Tt),et=a(D),f(R.$$.fragment,D),D.forEach(o),J.forEach(o),Ne=a(e),f(se.$$.fragment,e),qe=a(e),x=d(e,"DIV",{class:!0});var k=le(x);f(ae.$$.fragment,k),tt=a(k),fe=d(k,"P",{"data-svelte-h":!0}),m(fe)!=="svelte-yq335u"&&(fe.textContent=wt),ot=a(k),ge=d(k,"P",{"data-svelte-h":!0}),m(ge)!=="svelte-q52n56"&&(ge.innerHTML=xt),nt=a(k),be=d(k,"P",{"data-svelte-h":!0}),m(be)!=="svelte-hswkmf"&&(be.innerHTML=vt),st=a(k),C=d(k,"DIV",{class:!0});var W=le(C);f(re.$$.fragment,W),at=a(W),_e=d(W,"P",{"data-svelte-h":!0}),m(_e)!=="svelte-142us2z"&&(_e.innerHTML=Jt),rt=a(W),f(z.$$.fragment,W),it=a(W),f(G.$$.fragment,W),W.forEach(o),k.forEach(o),Ee=a(e),f(ie.$$.fragment,e),Ve=a(e),Me=d(e,"P",{}),le(Me).forEach(o),this.h()},h(){X(r,"name","hf:doc:metadata"),X(r,"content",Xt),X(Z,"class","flex flex-wrap space-x-1"),X(v,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),X(I,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),X(w,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),X(C,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),X(x,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,t){c(document.head,r),n(e,T,t),n(e,p,t),n(e,u,t),n(e,M,t),n(e,U,t),g(N,e,t),n(e,xe,t),n(e,Z,t),n(e,ve,t),g(q,e,t),n(e,Je,t),n(e,E,t),n(e,ke,t),n(e,V,t),n(e,Ue,t),n(e,B,t),n(e,Ce,t),n(e,L,t),n(e,je,t),g(H,e,t),n(e,$e,t),n(e,Q,t),n(e,Ie,t),g(S,e,t),n(e,De,t),n(e,Y,t),n(e,Ze,t),g(P,e,t),n(e,Fe,t),n(e,A,t),n(e,Re,t),g(O,e,t),n(e,ze,t),g(K,e,t),n(e,Ge,t),n(e,v,t),g(ee,v,null),c(v,He),c(v,de),c(v,Qe),c(v,ce),c(v,Se),g(F,v,null),n(e,We,t),g(te,e,t),n(e,Xe,t),n(e,w,t),g(oe,w,null),c(w,Ye),c(w,pe),c(w,Pe),c(w,ue),c(w,Ae),c(w,me),c(w,Oe),c(w,I),g(ne,I,null),c(I,Ke),c(I,he),c(I,et),g(R,I,null),n(e,Ne,t),g(se,e,t),n(e,qe,t),n(e,x,t),g(ae,x,null),c(x,tt),c(x,fe),c(x,ot),c(x,ge),c(x,nt),c(x,be),c(x,st),c(x,C),g(re,C,null),c(C,at),c(C,_e),c(C,rt),g(z,C,null),c(C,it),g(G,C,null),n(e,Ee,t),g(ie,e,t),n(e,Ve,t),n(e,Me,t),Be=!0},p(e,[t]){const j={};t&2&&(j.$$scope={dirty:t,ctx:e}),F.$set(j);const J={};t&2&&(J.$$scope={dirty:t,ctx:e}),R.$set(J);const D={};t&2&&(D.$$scope={dirty:t,ctx:e}),z.$set(D);const k={};t&2&&(k.$$scope={dirty:t,ctx:e}),G.$set(k)},i(e){Be||(b(N.$$.fragment,e),b(q.$$.fragment,e),b(H.$$.fragment,e),b(S.$$.fragment,e),b(P.$$.fragment,e),b(O.$$.fragment,e),b(K.$$.fragment,e),b(ee.$$.fragment,e),b(F.$$.fragment,e),b(te.$$.fragment,e),b(oe.$$.fragment,e),b(ne.$$.fragment,e),b(R.$$.fragment,e),b(se.$$.fragment,e),b(ae.$$.fragment,e),b(re.$$.fragment,e),b(z.$$.fragment,e),b(G.$$.fragment,e),b(ie.$$.fragment,e),Be=!0)},o(e){_(N.$$.fragment,e),_(q.$$.fragment,e),_(H.$$.fragment,e),_(S.$$.fragment,e),_(P.$$.fragment,e),_(O.$$.fragment,e),_(K.$$.fragment,e),_(ee.$$.fragment,e),_(F.$$.fragment,e),_(te.$$.fragment,e),_(oe.$$.fragment,e),_(ne.$$.fragment,e),_(R.$$.fragment,e),_(se.$$.fragment,e),_(ae.$$.fragment,e),_(re.$$.fragment,e),_(z.$$.fragment,e),_(G.$$.fragment,e),_(ie.$$.fragment,e),Be=!1},d(e){e&&(o(T),o(p),o(u),o(M),o(U),o(xe),o(Z),o(ve),o(Je),o(E),o(ke),o(V),o(Ue),o(B),o(Ce),o(L),o(je),o($e),o(Q),o(Ie),o(De),o(Y),o(Ze),o(Fe),o(A),o(Re),o(ze),o(Ge),o(v),o(We),o(Xe),o(w),o(Ne),o(qe),o(x),o(Ee),o(Ve),o(Me)),o(r),y(N,e),y(q,e),y(H,e),y(S,e),y(P,e),y(O,e),y(K,e),y(ee),y(F),y(te,e),y(oe),y(ne),y(R),y(se,e),y(ae),y(re),y(z),y(G),y(ie,e)}}}const Xt='{"title":"DBRX","local":"dbrx","sections":[{"title":"Overview","local":"overview","sections":[],"depth":2},{"title":"Usage Examples","local":"usage-examples","sections":[],"depth":2},{"title":"DbrxConfig","local":"transformers.DbrxConfig","sections":[],"depth":2},{"title":"DbrxModel","local":"transformers.DbrxModel","sections":[],"depth":2},{"title":"DbrxForCausalLM","local":"transformers.DbrxForCausalLM","sections":[],"depth":2}],"depth":1}';function Nt($){return jt(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class St extends $t{constructor(r){super(),It(this,r,Nt,Wt,Ct,{})}}export{St as component};
