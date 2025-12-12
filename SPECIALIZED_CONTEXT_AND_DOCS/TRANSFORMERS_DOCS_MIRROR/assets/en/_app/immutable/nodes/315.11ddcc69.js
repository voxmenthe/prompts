import{s as sn,o as an,n as Oe}from"../chunks/scheduler.18a86fab.js";import{S as rn,i as dn,g as d,s,r as c,A as ln,h as l,f as o,c as a,j as C,x as _,u as p,k as $,y as g,a as n,v as m,d as h,t as u,w as f}from"../chunks/index.98837b22.js";import{T as lt}from"../chunks/Tip.77304350.js";import{D as J}from"../chunks/Docstring.a1ef7999.js";import{C as ao}from"../chunks/CodeBlock.8d0c2e8a.js";import{E as nn}from"../chunks/ExampleCodeBlock.8c3ee1f9.js";import{H as T,E as cn}from"../chunks/getInferenceSnippets.06c2775f.js";function pn(w){let r,b;return r=new ao({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyME5lbW90cm9uTW9kZWwlMkMlMjBOZW1vdHJvbkNvbmZpZyUwQSUwQSUyMyUyMEluaXRpYWxpemluZyUyMGElMjBOZW1vdHJvbiUyMG5lbW90cm9uLTE1YiUyMHN0eWxlJTIwY29uZmlndXJhdGlvbiUwQWNvbmZpZ3VyYXRpb24lMjAlM0QlMjBOZW1vdHJvbkNvbmZpZygpJTBBJTBBJTIzJTIwSW5pdGlhbGl6aW5nJTIwYSUyMG1vZGVsJTIwZnJvbSUyMHRoZSUyMG5lbW90cm9uLTE1YiUyMHN0eWxlJTIwY29uZmlndXJhdGlvbiUwQW1vZGVsJTIwJTNEJTIwTmVtb3Ryb25Nb2RlbChjb25maWd1cmF0aW9uKSUwQSUwQSUyMyUyMEFjY2Vzc2luZyUyMHRoZSUyMG1vZGVsJTIwY29uZmlndXJhdGlvbiUwQWNvbmZpZ3VyYXRpb24lMjAlM0QlMjBtb2RlbC5jb25maWc=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> NemotronModel, NemotronConfig

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a Nemotron nemotron-15b style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = NemotronConfig()

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a model from the nemotron-15b style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = NemotronModel(configuration)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Accessing the model configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = model.config`,wrap:!1}}),{c(){c(r.$$.fragment)},l(i){p(r.$$.fragment,i)},m(i,v){m(r,i,v),b=!0},p:Oe,i(i){b||(h(r.$$.fragment,i),b=!0)},o(i){u(r.$$.fragment,i),b=!1},d(i){f(r,i)}}}function mn(w){let r,b=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){r=d("p"),r.innerHTML=b},l(i){r=l(i,"P",{"data-svelte-h":!0}),_(r)!=="svelte-fincs2"&&(r.innerHTML=b)},m(i,v){n(i,r,v)},p:Oe,d(i){i&&o(r)}}}function hn(w){let r,b=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){r=d("p"),r.innerHTML=b},l(i){r=l(i,"P",{"data-svelte-h":!0}),_(r)!=="svelte-fincs2"&&(r.innerHTML=b)},m(i,v){n(i,r,v)},p:Oe,d(i){i&&o(r)}}}function un(w){let r,b="Example:",i,v,N;return v=new ao({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBOZW1vdHJvbkZvckNhdXNhbExNJTBBJTBBbW9kZWwlMjAlM0QlMjBOZW1vdHJvbkZvckNhdXNhbExNLmZyb21fcHJldHJhaW5lZCglMjJudmlkaWElMkZuZW1vdHJvbi0zLThiLWJhc2UtNGstaGYlMjIpJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIybnZpZGlhJTJGbmVtb3Ryb24tMy04Yi1iYXNlLTRrLWhmJTIyKSUwQSUwQXByb21wdCUyMCUzRCUyMCUyMkhleSUyQyUyMGFyZSUyMHlvdSUyMGNvbnNjaW91cyUzRiUyMENhbiUyMHlvdSUyMHRhbGslMjB0byUyMG1lJTNGJTIyJTBBaW5wdXRzJTIwJTNEJTIwdG9rZW5pemVyKHByb21wdCUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpJTBBJTBBJTIzJTIwR2VuZXJhdGUlMEFnZW5lcmF0ZV9pZHMlMjAlM0QlMjBtb2RlbC5nZW5lcmF0ZShpbnB1dHMuaW5wdXRfaWRzJTJDJTIwbWF4X2xlbmd0aCUzRDMwKSUwQXRva2VuaXplci5iYXRjaF9kZWNvZGUoZ2VuZXJhdGVfaWRzJTJDJTIwc2tpcF9zcGVjaWFsX3Rva2VucyUzRFRydWUlMkMlMjBjbGVhbl91cF90b2tlbml6YXRpb25fc3BhY2VzJTNERmFsc2UpJTVCMCU1RA==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, NemotronForCausalLM

<span class="hljs-meta">&gt;&gt;&gt; </span>model = NemotronForCausalLM.from_pretrained(<span class="hljs-string">&quot;nvidia/nemotron-3-8b-base-4k-hf&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;nvidia/nemotron-3-8b-base-4k-hf&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>prompt = <span class="hljs-string">&quot;Hey, are you conscious? Can you talk to me?&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(prompt, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Generate</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>generate_ids = model.generate(inputs.input_ids, max_length=<span class="hljs-number">30</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer.batch_decode(generate_ids, skip_special_tokens=<span class="hljs-literal">True</span>, clean_up_tokenization_spaces=<span class="hljs-literal">False</span>)[<span class="hljs-number">0</span>]
<span class="hljs-string">&quot;Hey, are you conscious? Can you talk to me?\\nI&#x27;m not conscious, but I can talk to you.&quot;</span>`,wrap:!1}}),{c(){r=d("p"),r.textContent=b,i=s(),c(v.$$.fragment)},l(y){r=l(y,"P",{"data-svelte-h":!0}),_(r)!=="svelte-11lpom8"&&(r.textContent=b),i=a(y),p(v.$$.fragment,y)},m(y,G){n(y,r,G),n(y,i,G),m(v,y,G),N=!0},p:Oe,i(y){N||(h(v.$$.fragment,y),N=!0)},o(y){u(v.$$.fragment,y),N=!1},d(y){y&&(o(r),o(i)),f(v,y)}}}function fn(w){let r,b=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){r=d("p"),r.innerHTML=b},l(i){r=l(i,"P",{"data-svelte-h":!0}),_(r)!=="svelte-fincs2"&&(r.innerHTML=b)},m(i,v){n(i,r,v)},p:Oe,d(i){i&&o(r)}}}function gn(w){let r,b=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){r=d("p"),r.innerHTML=b},l(i){r=l(i,"P",{"data-svelte-h":!0}),_(r)!=="svelte-fincs2"&&(r.innerHTML=b)},m(i,v){n(i,r,v)},p:Oe,d(i){i&&o(r)}}}function _n(w){let r,b=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){r=d("p"),r.innerHTML=b},l(i){r=l(i,"P",{"data-svelte-h":!0}),_(r)!=="svelte-fincs2"&&(r.innerHTML=b)},m(i,v){n(i,r,v)},p:Oe,d(i){i&&o(r)}}}function bn(w){let r,b,i,v,N,y="<em>This model was released on 2024-02-26 and added to Hugging Face Transformers on 2024-08-06.</em>",G,Y,ct,S,xo='<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/> <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat"/> <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/>',pt,K,mt,ee,zo='The use of this model is governed by the <a href="https://developer.nvidia.com/downloads/nv-ai-foundation-models-license" rel="nofollow">NVIDIA AI Foundation Models Community License Agreement</a>.',ht,te,ut,oe,Fo='Nemotron-4 is a family of enterprise ready generative text models compatible with <a href="https://www.nvidia.com/en-us/ai-data-science/generative-ai/nemo-framework/" rel="nofollow">NVIDIA NeMo Framework</a>.',ft,ne,Jo='NVIDIA NeMo is an end-to-end, cloud-native platform to build, customize, and deploy generative AI models anywhere. It includes training and inferencing frameworks, guardrailing toolkits, data curation tools, and pretrained models, offering enterprises an easy, cost-effective, and fast way to adopt generative AI. To get access to NeMo Framework, please sign up at <a href="https://developer.nvidia.com/nemo-framework/join" rel="nofollow">this link</a>.',gt,se,_t,ae,Lo='<a href="https://developer.nvidia.com/blog/nvidia-ai-foundation-models-build-custom-enterprise-chatbots-and-co-pilots-with-production-ready-llms/" rel="nofollow">Announcement Blog</a>',bt,re,vt,ie,Uo="<strong>Architecture Type:</strong> Transformer",yt,de,Io="<strong>Network Architecture:</strong> Transformer Decoder (auto-regressive language model).",Tt,le,wt,ce,kt,pe,Wo='Minitron is a family of small language models (SLMs) obtained by pruning NVIDIA’s <a href="https://huggingface.co/papers/2402.16819" rel="nofollow">Nemotron-4 15B</a> model. We prune model embedding size, attention heads, and MLP intermediate dimension, following which, we perform continued training with distillation to arrive at the final models.',$t,me,jo='Deriving the Minitron 8B and 4B models from the base 15B model using our approach requires up to <strong>40x fewer training tokens</strong> per model compared to training from scratch; this results in <strong>compute cost savings of 1.8x</strong> for training the full model family (15B, 8B, and 4B). Minitron models exhibit up to a 16% improvement in MMLU scores compared to training from scratch, perform comparably to other community models such as Mistral 7B, Gemma 7B and Llama-3 8B, and outperform state-of-the-art compression techniques from the literature. Please refer to our <a href="https://huggingface.co/papers/2407.14679" rel="nofollow">arXiv paper</a> for more details.',Mt,he,Ho="Minitron models are for research and development only.",Ct,ue,Nt,fe,Bo="The following code provides an example of how to load the Minitron-4B model and use it to perform text generation.",xt,ge,zt,_e,Ft,be,qo='Minitron is released under the <a href="https://developer.download.nvidia.com/licenses/nvidia-open-model-license-agreement-june-2024.pdf" rel="nofollow">NVIDIA Open Model License Agreement</a>.',Jt,ve,Lt,ye,Zo='<em>5-shot performance.</em> Language Understanding evaluated using <a href="https://huggingface.co/papers/2009.03300" rel="nofollow">Massive Multitask Language Understanding</a>:',Ut,Te,Po='<thead><tr><th align="left">Average</th></tr></thead> <tbody><tr><td align="left">58.6</td></tr></tbody>',It,we,Ao='<em>Zero-shot performance.</em> Evaluated using select datasets from the <a href="https://github.com/EleutherAI/lm-evaluation-harness" rel="nofollow">LM Evaluation Harness</a> with additions:',Wt,ke,Go='<thead><tr><th align="left">HellaSwag</th> <th align="left">Winogrande</th> <th align="left">GSM8K</th> <th align="left">ARC-C</th> <th align="left">XLSum</th></tr></thead> <tbody><tr><td align="left">75.0</td> <td align="left">74.0</td> <td align="left">24.1</td> <td align="left">50.9</td> <td align="left">29.5</td></tr></tbody>',jt,$e,So='<em>Code generation performance</em>. Evaluated using <a href="https://github.com/openai/human-eval" rel="nofollow">HumanEval</a>:',Ht,Me,Qo='<thead><tr><th align="left">p@1, 0-Shot</th></tr></thead> <tbody><tr><td align="left">23.3</td></tr></tbody>',Bt,Ce,Eo='Please refer to our <a href="https://huggingface.co/papers/2407.14679" rel="nofollow">paper</a> for the full set of results.',qt,Ne,Zt,xe,Ro="If you find our work helpful, please consider citing our paper:",Pt,ze,At,Fe,Gt,z,Je,ro,Ve,Do=`This is the configuration class to store the configuration of a <a href="/docs/transformers/v4.56.2/en/model_doc/nemotron#transformers.NemotronModel">NemotronModel</a>. It is used to instantiate an Nemotron
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the Nemotron-8B.
e.g. <a href="https://huggingface.co/nvidia/nemotron-3-8b-base-4k-hf" rel="nofollow">nvidia/nemotron-3-8b-base-4k-hf</a>.
Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> and can be used to control the model outputs. Read the
documentation from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> for more information.`,io,Q,St,Le,Qt,k,Ue,lo,Ye,Xo="The bare Nemotron Model outputting raw hidden-states without any specific head on top.",co,Ke,Oo=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,po,et,Vo=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,mo,L,Ie,ho,tt,Yo='The <a href="/docs/transformers/v4.56.2/en/model_doc/nemotron#transformers.NemotronModel">NemotronModel</a> forward method, overrides the <code>__call__</code> special method.',uo,E,Et,We,Rt,j,je,fo,x,He,go,ot,Ko='The <a href="/docs/transformers/v4.56.2/en/model_doc/nemotron#transformers.NemotronForCausalLM">NemotronForCausalLM</a> forward method, overrides the <code>__call__</code> special method.',_o,R,bo,D,Dt,Be,Xt,H,qe,vo,U,Ze,yo,nt,en="The <code>GenericForSequenceClassification</code> forward method, overrides the <code>__call__</code> special method.",To,X,Ot,Pe,Vt,B,Ae,wo,I,Ge,ko,st,tn="The <code>GenericForQuestionAnswering</code> forward method, overrides the <code>__call__</code> special method.",$o,O,Yt,Se,Kt,q,Qe,Mo,W,Ee,Co,at,on="The <code>GenericForTokenClassification</code> forward method, overrides the <code>__call__</code> special method.",No,V,eo,Re,to,dt,oo;return Y=new T({props:{title:"Nemotron",local:"nemotron",headingTag:"h1"}}),K=new T({props:{title:"License",local:"license",headingTag:"h3"}}),te=new T({props:{title:"Description",local:"description",headingTag:"h3"}}),se=new T({props:{title:"References",local:"references",headingTag:"h3"}}),re=new T({props:{title:"Model Architecture",local:"model-architecture",headingTag:"h3"}}),le=new T({props:{title:"Minitron",local:"minitron",headingTag:"h2"}}),ce=new T({props:{title:"Minitron 4B Base",local:"minitron-4b-base",headingTag:"h3"}}),ue=new T({props:{title:"HuggingFace Quickstart",local:"huggingface-quickstart",headingTag:"h3"}}),ge=new ao({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Rva2VuaXplciUyQyUyMEF1dG9Nb2RlbEZvckNhdXNhbExNJTJDJTIwaW5mZXJfZGV2aWNlJTBBJTBBJTIzJTIwTG9hZCUyMHRoZSUyMHRva2VuaXplciUyMGFuZCUyMG1vZGVsJTBBbW9kZWxfcGF0aCUyMCUzRCUyMCdudmlkaWElMkZNaW5pdHJvbi00Qi1CYXNlJyUwQXRva2VuaXplciUyMCUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKG1vZGVsX3BhdGgpJTBBJTBBZGV2aWNlJTIwJTNEJTIwaW5mZXJfZGV2aWNlKCklMEFkdHlwZSUyMCUyMCUzRCUyMHRvcmNoLmJmbG9hdDE2JTBBbW9kZWwlMjAlMjAlM0QlMjBBdXRvTW9kZWxGb3JDYXVzYWxMTS5mcm9tX3ByZXRyYWluZWQobW9kZWxfcGF0aCUyQyUyMGR0eXBlJTNEZHR5cGUlMkMlMjBkZXZpY2VfbWFwJTNEZGV2aWNlKSUwQSUwQSUyMyUyMFByZXBhcmUlMjB0aGUlMjBpbnB1dCUyMHRleHQlMEFwcm9tcHQlMjAlM0QlMjAnQ29tcGxldGUlMjB0aGUlMjBwYXJhZ3JhcGglM0ElMjBvdXIlMjBzb2xhciUyMHN5c3RlbSUyMGlzJyUwQWlucHV0cyUyMCUzRCUyMHRva2VuaXplci5lbmNvZGUocHJvbXB0JTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QncHQnKS50byhtb2RlbC5kZXZpY2UpJTBBJTBBJTIzJTIwR2VuZXJhdGUlMjB0aGUlMjBvdXRwdXQlMEFvdXRwdXRzJTIwJTNEJTIwbW9kZWwuZ2VuZXJhdGUoaW5wdXRzJTJDJTIwbWF4X2xlbmd0aCUzRDIwKSUwQSUwQSUyMyUyMERlY29kZSUyMGFuZCUyMHByaW50JTIwdGhlJTIwb3V0cHV0JTBBb3V0cHV0X3RleHQlMjAlM0QlMjB0b2tlbml6ZXIuZGVjb2RlKG91dHB1dHMlNUIwJTVEKSUwQXByaW50KG91dHB1dF90ZXh0KQ==",highlighted:`<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, AutoModelForCausalLM, infer_device

<span class="hljs-comment"># Load the tokenizer and model</span>
model_path = <span class="hljs-string">&#x27;nvidia/Minitron-4B-Base&#x27;</span>
tokenizer  = AutoTokenizer.from_pretrained(model_path)

device = infer_device()
dtype  = torch.bfloat16
model  = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype, device_map=device)

<span class="hljs-comment"># Prepare the input text</span>
prompt = <span class="hljs-string">&#x27;Complete the paragraph: our solar system is&#x27;</span>
inputs = tokenizer.encode(prompt, return_tensors=<span class="hljs-string">&#x27;pt&#x27;</span>).to(model.device)

<span class="hljs-comment"># Generate the output</span>
outputs = model.generate(inputs, max_length=<span class="hljs-number">20</span>)

<span class="hljs-comment"># Decode and print the output</span>
output_text = tokenizer.decode(outputs[<span class="hljs-number">0</span>])
<span class="hljs-built_in">print</span>(output_text)`,wrap:!1}}),_e=new T({props:{title:"License",local:"license",headingTag:"h3"}}),ve=new T({props:{title:"Evaluation Results",local:"evaluation-results",headingTag:"h3"}}),Ne=new T({props:{title:"Citation",local:"citation",headingTag:"h3"}}),ze=new ao({props:{code:"JTQwYXJ0aWNsZSU3Qm1pbml0cm9uMjAyNCUyQyUwQSUyMCUyMCUyMCUyMCUyMCUyMHRpdGxlJTNEJTdCQ29tcGFjdCUyMExhbmd1YWdlJTIwTW9kZWxzJTIwdmlhJTIwUHJ1bmluZyUyMGFuZCUyMEtub3dsZWRnZSUyMERpc3RpbGxhdGlvbiU3RCUyQyUwQSUyMCUyMCUyMCUyMCUyMCUyMGF1dGhvciUzRCU3QlNhdXJhdiUyME11cmFsaWRoYXJhbiUyMGFuZCUyMFNoYXJhdGglMjBUdXJ1dmVrZXJlJTIwU3JlZW5pdmFzJTIwYW5kJTIwUmF2aXJhaiUyMEpvc2hpJTIwYW5kJTIwTWFyY2luJTIwQ2hvY2hvd3NraSUyMGFuZCUyME1vc3RvZmElMjBQYXR3YXJ5JTIwYW5kJTIwTW9oYW1tYWQlMjBTaG9leWJpJTIwYW5kJTIwQnJ5YW4lMjBDYXRhbnphcm8lMjBhbmQlMjBKYW4lMjBLYXV0eiUyMGFuZCUyMFBhdmxvJTIwTW9sY2hhbm92JTdEJTJDJTBBJTIwJTIwJTIwJTIwJTIwJTIwam91cm5hbCUzRCU3QmFyWGl2JTIwcHJlcHJpbnQlMjBhclhpdiUzQTI0MDcuMTQ2NzklN0QlMkMlMEElMjAlMjAlMjAlMjAlMjAlMjB5ZWFyJTNEJTdCMjAyNCU3RCUyQyUwQSUyMCUyMCUyMCUyMCUyMCUyMHVybCUzRCU3Qmh0dHBzJTNBJTJGJTJGaHVnZ2luZ2ZhY2UuY28lMkZwYXBlcnMlMkYyNDA3LjE0Njc5JTdEJTJDJTBBJTdE",highlighted:`<span class="hljs-comment">@article{minitron2024,</span>
      title={Compact Language Models via Pruning <span class="hljs-keyword">and</span> Knowledge Distillation},
      author={Saurav Muralidharan <span class="hljs-keyword">and</span> Sharath Turuvekere Sreenivas <span class="hljs-keyword">and</span> Raviraj Joshi <span class="hljs-keyword">and</span> Marcin Chochowski <span class="hljs-keyword">and</span> Mostofa Patwary <span class="hljs-keyword">and</span> Mohammad Shoeybi <span class="hljs-keyword">and</span> Bryan Catanzaro <span class="hljs-keyword">and</span> Jan Kautz <span class="hljs-keyword">and</span> Pavlo Molchanov},
      journal={arXiv preprint arXiv:<span class="hljs-number">2407</span>.<span class="hljs-number">14679</span>},
      year={<span class="hljs-number">2024</span>},
      url={https:<span class="hljs-comment">//huggingface.co/papers/2407.14679},</span>
}`,wrap:!1}}),Fe=new T({props:{title:"NemotronConfig",local:"transformers.NemotronConfig",headingTag:"h2"}}),Je=new J({props:{name:"class transformers.NemotronConfig",anchor:"transformers.NemotronConfig",parameters:[{name:"vocab_size",val:" = 256000"},{name:"hidden_size",val:" = 6144"},{name:"intermediate_size",val:" = 24576"},{name:"num_hidden_layers",val:" = 32"},{name:"num_attention_heads",val:" = 48"},{name:"head_dim",val:" = None"},{name:"num_key_value_heads",val:" = None"},{name:"hidden_act",val:" = 'relu2'"},{name:"max_position_embeddings",val:" = 4096"},{name:"initializer_range",val:" = 0.0134"},{name:"norm_eps",val:" = 1e-05"},{name:"use_cache",val:" = True"},{name:"pad_token_id",val:" = None"},{name:"bos_token_id",val:" = 2"},{name:"eos_token_id",val:" = 3"},{name:"tie_word_embeddings",val:" = False"},{name:"rope_theta",val:" = 10000.0"},{name:"partial_rotary_factor",val:" = 0.5"},{name:"attention_bias",val:" = False"},{name:"attention_dropout",val:" = 0.0"},{name:"mlp_bias",val:" = False"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.NemotronConfig.vocab_size",description:`<strong>vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 256000) &#x2014;
Vocabulary size of the Nemotron model. Defines the number of different tokens that can be represented by the
<code>inputs_ids</code> passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/nemotron#transformers.NemotronModel">NemotronModel</a>`,name:"vocab_size"},{anchor:"transformers.NemotronConfig.hidden_size",description:`<strong>hidden_size</strong> (<code>int</code>, <em>optional</em>, defaults to 6144) &#x2014;
Dimension of the hidden representations.`,name:"hidden_size"},{anchor:"transformers.NemotronConfig.intermediate_size",description:`<strong>intermediate_size</strong> (<code>int</code>, <em>optional</em>, defaults to 24576) &#x2014;
Dimension of the MLP representations.`,name:"intermediate_size"},{anchor:"transformers.NemotronConfig.num_hidden_layers",description:`<strong>num_hidden_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 32) &#x2014;
Number of hidden layers in the Transformer decoder.`,name:"num_hidden_layers"},{anchor:"transformers.NemotronConfig.num_attention_heads",description:`<strong>num_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 48) &#x2014;
Number of attention heads for each attention layer in the Transformer decoder.`,name:"num_attention_heads"},{anchor:"transformers.NemotronConfig.head_dim",description:`<strong>head_dim</strong> (<code>int</code>, <em>optional</em>) &#x2014;
Projection weights dimension in multi-head attention. Set to hidden_size // num_attention_heads if None`,name:"head_dim"},{anchor:"transformers.NemotronConfig.num_key_value_heads",description:`<strong>num_key_value_heads</strong> (<code>int</code>, <em>optional</em>) &#x2014;
This is the number of key_value heads that should be used to implement Grouped Query Attention. If
<code>num_key_value_heads=num_attention_heads</code>, the model will use Multi Head Attention (MHA), if
<code>num_key_value_heads=1 the model will use Multi Query Attention (MQA) otherwise GQA is used. When converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed by meanpooling all the original heads within that group. For more details, check out [this paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to </code>num_attention_heads\`.`,name:"num_key_value_heads"},{anchor:"transformers.NemotronConfig.hidden_act",description:`<strong>hidden_act</strong> (<code>str</code> or <code>function</code>, <em>optional</em>, defaults to <code>&quot;relu2&quot;</code>) &#x2014;
The non-linear activation function (function or string) in the decoder.`,name:"hidden_act"},{anchor:"transformers.NemotronConfig.max_position_embeddings",description:`<strong>max_position_embeddings</strong> (<code>int</code>, <em>optional</em>, defaults to 4096) &#x2014;
The maximum sequence length that this model might ever be used with.`,name:"max_position_embeddings"},{anchor:"transformers.NemotronConfig.initializer_range",description:`<strong>initializer_range</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0134) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.`,name:"initializer_range"},{anchor:"transformers.NemotronConfig.norm_eps",description:`<strong>norm_eps</strong> (<code>float</code>, <em>optional</em>, defaults to 1e-05) &#x2014;
The epsilon used by the normalization layers.`,name:"norm_eps"},{anchor:"transformers.NemotronConfig.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not the model should return the last key/values attentions (not used by all models). Only
relevant if <code>config.is_decoder=True</code>.`,name:"use_cache"},{anchor:"transformers.NemotronConfig.pad_token_id",description:`<strong>pad_token_id</strong> (<code>int</code>, <em>optional</em>) &#x2014;
Padding token id.`,name:"pad_token_id"},{anchor:"transformers.NemotronConfig.bos_token_id",description:`<strong>bos_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 2) &#x2014;
Beginning of stream token id.`,name:"bos_token_id"},{anchor:"transformers.NemotronConfig.eos_token_id",description:`<strong>eos_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 3) &#x2014;
End of stream token id.`,name:"eos_token_id"},{anchor:"transformers.NemotronConfig.tie_word_embeddings",description:`<strong>tie_word_embeddings</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether to tie weight embeddings`,name:"tie_word_embeddings"},{anchor:"transformers.NemotronConfig.rope_theta",description:`<strong>rope_theta</strong> (<code>float</code>, <em>optional</em>, defaults to 10000.0) &#x2014;
The base period of the RoPE embeddings.`,name:"rope_theta"},{anchor:"transformers.NemotronConfig.partial_rotary_factor",description:"<strong>partial_rotary_factor</strong> (<code>float</code>, <em>optional</em>, defaults to 0.5) &#x2014; Percentage of the query and keys which will have rotary embedding.",name:"partial_rotary_factor"},{anchor:"transformers.NemotronConfig.attention_bias",description:`<strong>attention_bias</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether to use a bias in the query, key, value and output projection layers during self-attention.`,name:"attention_bias"},{anchor:"transformers.NemotronConfig.attention_dropout",description:`<strong>attention_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The dropout ratio for the attention probabilities.`,name:"attention_dropout"},{anchor:"transformers.NemotronConfig.mlp_bias",description:`<strong>mlp_bias</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether to use a bias in up_proj and down_proj layers in the MLP layers.`,name:"mlp_bias"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/nemotron/configuration_nemotron.py#L26"}}),Q=new nn({props:{anchor:"transformers.NemotronConfig.example",$$slots:{default:[pn]},$$scope:{ctx:w}}}),Le=new T({props:{title:"NemotronModel",local:"transformers.NemotronModel",headingTag:"h2"}}),Ue=new J({props:{name:"class transformers.NemotronModel",anchor:"transformers.NemotronModel",parameters:[{name:"config",val:": NemotronConfig"}],parametersDescription:[{anchor:"transformers.NemotronModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/nemotron#transformers.NemotronConfig">NemotronConfig</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/nemotron/modeling_nemotron.py#L615"}}),Ie=new J({props:{name:"forward",anchor:"transformers.NemotronModel.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Union[list[torch.FloatTensor], transformers.cache_utils.Cache, NoneType] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"}],parametersDescription:[{anchor:"transformers.NemotronModel.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.NemotronModel.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.NemotronModel.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.NemotronModel.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>Union[list[torch.FloatTensor], ~cache_utils.Cache, NoneType]</code>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.NemotronModel.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.NemotronModel.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.NemotronModel.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.NemotronModel.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.NemotronModel.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/nemotron/modeling_nemotron.py#L639",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast"
>transformers.modeling_outputs.BaseModelOutputWithPast</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/nemotron#transformers.NemotronConfig"
>NemotronConfig</a>) and inputs.</p>
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
`}}),E=new lt({props:{$$slots:{default:[mn]},$$scope:{ctx:w}}}),We=new T({props:{title:"NemotronForCausalLM",local:"transformers.NemotronForCausalLM",headingTag:"h2"}}),je=new J({props:{name:"class transformers.NemotronForCausalLM",anchor:"transformers.NemotronForCausalLM",parameters:[{name:"config",val:""}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/nemotron/modeling_nemotron.py#L859"}}),He=new J({props:{name:"forward",anchor:"transformers.NemotronForCausalLM.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Union[list[torch.FloatTensor], transformers.cache_utils.Cache, NoneType] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"},{name:"logits_to_keep",val:": typing.Union[int, torch.Tensor] = 0"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.NemotronForCausalLM.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.NemotronForCausalLM.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.NemotronForCausalLM.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.NemotronForCausalLM.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>Union[list[torch.FloatTensor], ~cache_utils.Cache, NoneType]</code>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.NemotronForCausalLM.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.NemotronForCausalLM.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.NemotronForCausalLM.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.NemotronForCausalLM.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.NemotronForCausalLM.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.NemotronForCausalLM.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"},{anchor:"transformers.NemotronForCausalLM.forward.logits_to_keep",description:`<strong>logits_to_keep</strong> (<code>Union[int, torch.Tensor]</code>, defaults to <code>0</code>) &#x2014;
If an <code>int</code>, compute logits for the last <code>logits_to_keep</code> tokens. If <code>0</code>, calculate logits for all
<code>input_ids</code> (special case). Only last token logits are needed for generation, and calculating them only for that
token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
If a <code>torch.Tensor</code>, must be 1D corresponding to the indices to keep in the sequence length dimension.
This is useful when using packed tensor format (single dimension for batch and sequence length).`,name:"logits_to_keep"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/nemotron/modeling_nemotron.py#L871",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast"
>transformers.modeling_outputs.CausalLMOutputWithPast</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/nemotron#transformers.NemotronConfig"
>NemotronConfig</a>) and inputs.</p>
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
`}}),R=new lt({props:{$$slots:{default:[hn]},$$scope:{ctx:w}}}),D=new nn({props:{anchor:"transformers.NemotronForCausalLM.forward.example",$$slots:{default:[un]},$$scope:{ctx:w}}}),Be=new T({props:{title:"NemotronForSequenceClassification",local:"transformers.NemotronForSequenceClassification",headingTag:"h2"}}),qe=new J({props:{name:"class transformers.NemotronForSequenceClassification",anchor:"transformers.NemotronForSequenceClassification",parameters:[{name:"config",val:""}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/nemotron/modeling_nemotron.py#L946"}}),Ze=new J({props:{name:"forward",anchor:"transformers.NemotronForSequenceClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.NemotronForSequenceClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.NemotronForSequenceClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.NemotronForSequenceClassification.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.NemotronForSequenceClassification.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.NemotronForSequenceClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.NemotronForSequenceClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.NemotronForSequenceClassification.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
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
`}}),X=new lt({props:{$$slots:{default:[fn]},$$scope:{ctx:w}}}),Pe=new T({props:{title:"NemotronForQuestionAnswering",local:"transformers.NemotronForQuestionAnswering",headingTag:"h2"}}),Ae=new J({props:{name:"class transformers.NemotronForQuestionAnswering",anchor:"transformers.NemotronForQuestionAnswering",parameters:[{name:"config",val:""}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/nemotron/modeling_nemotron.py#L949"}}),Ge=new J({props:{name:"forward",anchor:"transformers.NemotronForQuestionAnswering.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"start_positions",val:": typing.Optional[torch.LongTensor] = None"},{name:"end_positions",val:": typing.Optional[torch.LongTensor] = None"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.NemotronForQuestionAnswering.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.NemotronForQuestionAnswering.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.NemotronForQuestionAnswering.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.NemotronForQuestionAnswering.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.NemotronForQuestionAnswering.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.NemotronForQuestionAnswering.forward.start_positions",description:`<strong>start_positions</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for position (index) of the start of the labelled span for computing the token classification loss.
Positions are clamped to the length of the sequence (<code>sequence_length</code>). Position outside of the sequence
are not taken into account for computing the loss.`,name:"start_positions"},{anchor:"transformers.NemotronForQuestionAnswering.forward.end_positions",description:`<strong>end_positions</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for position (index) of the end of the labelled span for computing the token classification loss.
Positions are clamped to the length of the sequence (<code>sequence_length</code>). Position outside of the sequence
are not taken into account for computing the loss.`,name:"end_positions"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/modeling_layers.py#L191",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.QuestionAnsweringModelOutput"
>transformers.modeling_outputs.QuestionAnsweringModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<code>None</code>) and inputs.</p>
<ul>
<li>
<p><strong>loss</strong> (<code>torch.FloatTensor</code> of shape <code>(1,)</code>, <em>optional</em>, returned when <code>labels</code> is provided) — Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.</p>
</li>
<li>
<p><strong>start_logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>) — Span-start scores (before SoftMax).</p>
</li>
<li>
<p><strong>end_logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>) — Span-end scores (before SoftMax).</p>
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
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.QuestionAnsweringModelOutput"
>transformers.modeling_outputs.QuestionAnsweringModelOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),O=new lt({props:{$$slots:{default:[gn]},$$scope:{ctx:w}}}),Se=new T({props:{title:"NemotronForTokenClassification",local:"transformers.NemotronForTokenClassification",headingTag:"h2"}}),Qe=new J({props:{name:"class transformers.NemotronForTokenClassification",anchor:"transformers.NemotronForTokenClassification",parameters:[{name:"config",val:""}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/nemotron/modeling_nemotron.py#L953"}}),Ee=new J({props:{name:"forward",anchor:"transformers.NemotronForTokenClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.NemotronForTokenClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.NemotronForTokenClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.NemotronForTokenClassification.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.NemotronForTokenClassification.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.NemotronForTokenClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.NemotronForTokenClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.NemotronForTokenClassification.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
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
`}}),V=new lt({props:{$$slots:{default:[_n]},$$scope:{ctx:w}}}),Re=new cn({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/nemotron.md"}}),{c(){r=d("meta"),b=s(),i=d("p"),v=s(),N=d("p"),N.innerHTML=y,G=s(),c(Y.$$.fragment),ct=s(),S=d("div"),S.innerHTML=xo,pt=s(),c(K.$$.fragment),mt=s(),ee=d("p"),ee.innerHTML=zo,ht=s(),c(te.$$.fragment),ut=s(),oe=d("p"),oe.innerHTML=Fo,ft=s(),ne=d("p"),ne.innerHTML=Jo,gt=s(),c(se.$$.fragment),_t=s(),ae=d("p"),ae.innerHTML=Lo,bt=s(),c(re.$$.fragment),vt=s(),ie=d("p"),ie.innerHTML=Uo,yt=s(),de=d("p"),de.innerHTML=Io,Tt=s(),c(le.$$.fragment),wt=s(),c(ce.$$.fragment),kt=s(),pe=d("p"),pe.innerHTML=Wo,$t=s(),me=d("p"),me.innerHTML=jo,Mt=s(),he=d("p"),he.textContent=Ho,Ct=s(),c(ue.$$.fragment),Nt=s(),fe=d("p"),fe.textContent=Bo,xt=s(),c(ge.$$.fragment),zt=s(),c(_e.$$.fragment),Ft=s(),be=d("p"),be.innerHTML=qo,Jt=s(),c(ve.$$.fragment),Lt=s(),ye=d("p"),ye.innerHTML=Zo,Ut=s(),Te=d("table"),Te.innerHTML=Po,It=s(),we=d("p"),we.innerHTML=Ao,Wt=s(),ke=d("table"),ke.innerHTML=Go,jt=s(),$e=d("p"),$e.innerHTML=So,Ht=s(),Me=d("table"),Me.innerHTML=Qo,Bt=s(),Ce=d("p"),Ce.innerHTML=Eo,qt=s(),c(Ne.$$.fragment),Zt=s(),xe=d("p"),xe.textContent=Ro,Pt=s(),c(ze.$$.fragment),At=s(),c(Fe.$$.fragment),Gt=s(),z=d("div"),c(Je.$$.fragment),ro=s(),Ve=d("p"),Ve.innerHTML=Do,io=s(),c(Q.$$.fragment),St=s(),c(Le.$$.fragment),Qt=s(),k=d("div"),c(Ue.$$.fragment),lo=s(),Ye=d("p"),Ye.textContent=Xo,co=s(),Ke=d("p"),Ke.innerHTML=Oo,po=s(),et=d("p"),et.innerHTML=Vo,mo=s(),L=d("div"),c(Ie.$$.fragment),ho=s(),tt=d("p"),tt.innerHTML=Yo,uo=s(),c(E.$$.fragment),Et=s(),c(We.$$.fragment),Rt=s(),j=d("div"),c(je.$$.fragment),fo=s(),x=d("div"),c(He.$$.fragment),go=s(),ot=d("p"),ot.innerHTML=Ko,_o=s(),c(R.$$.fragment),bo=s(),c(D.$$.fragment),Dt=s(),c(Be.$$.fragment),Xt=s(),H=d("div"),c(qe.$$.fragment),vo=s(),U=d("div"),c(Ze.$$.fragment),yo=s(),nt=d("p"),nt.innerHTML=en,To=s(),c(X.$$.fragment),Ot=s(),c(Pe.$$.fragment),Vt=s(),B=d("div"),c(Ae.$$.fragment),wo=s(),I=d("div"),c(Ge.$$.fragment),ko=s(),st=d("p"),st.innerHTML=tn,$o=s(),c(O.$$.fragment),Yt=s(),c(Se.$$.fragment),Kt=s(),q=d("div"),c(Qe.$$.fragment),Mo=s(),W=d("div"),c(Ee.$$.fragment),Co=s(),at=d("p"),at.innerHTML=on,No=s(),c(V.$$.fragment),eo=s(),c(Re.$$.fragment),to=s(),dt=d("p"),this.h()},l(e){const t=ln("svelte-u9bgzb",document.head);r=l(t,"META",{name:!0,content:!0}),t.forEach(o),b=a(e),i=l(e,"P",{}),C(i).forEach(o),v=a(e),N=l(e,"P",{"data-svelte-h":!0}),_(N)!=="svelte-t6fsou"&&(N.innerHTML=y),G=a(e),p(Y.$$.fragment,e),ct=a(e),S=l(e,"DIV",{class:!0,"data-svelte-h":!0}),_(S)!=="svelte-b95w5j"&&(S.innerHTML=xo),pt=a(e),p(K.$$.fragment,e),mt=a(e),ee=l(e,"P",{"data-svelte-h":!0}),_(ee)!=="svelte-5ltir7"&&(ee.innerHTML=zo),ht=a(e),p(te.$$.fragment,e),ut=a(e),oe=l(e,"P",{"data-svelte-h":!0}),_(oe)!=="svelte-18vqkxz"&&(oe.innerHTML=Fo),ft=a(e),ne=l(e,"P",{"data-svelte-h":!0}),_(ne)!=="svelte-1inbdxl"&&(ne.innerHTML=Jo),gt=a(e),p(se.$$.fragment,e),_t=a(e),ae=l(e,"P",{"data-svelte-h":!0}),_(ae)!=="svelte-eh2pqk"&&(ae.innerHTML=Lo),bt=a(e),p(re.$$.fragment,e),vt=a(e),ie=l(e,"P",{"data-svelte-h":!0}),_(ie)!=="svelte-1afrym1"&&(ie.innerHTML=Uo),yt=a(e),de=l(e,"P",{"data-svelte-h":!0}),_(de)!=="svelte-8mone6"&&(de.innerHTML=Io),Tt=a(e),p(le.$$.fragment,e),wt=a(e),p(ce.$$.fragment,e),kt=a(e),pe=l(e,"P",{"data-svelte-h":!0}),_(pe)!=="svelte-1frpg5f"&&(pe.innerHTML=Wo),$t=a(e),me=l(e,"P",{"data-svelte-h":!0}),_(me)!=="svelte-3m3ent"&&(me.innerHTML=jo),Mt=a(e),he=l(e,"P",{"data-svelte-h":!0}),_(he)!=="svelte-1890ku0"&&(he.textContent=Ho),Ct=a(e),p(ue.$$.fragment,e),Nt=a(e),fe=l(e,"P",{"data-svelte-h":!0}),_(fe)!=="svelte-1ltrinh"&&(fe.textContent=Bo),xt=a(e),p(ge.$$.fragment,e),zt=a(e),p(_e.$$.fragment,e),Ft=a(e),be=l(e,"P",{"data-svelte-h":!0}),_(be)!=="svelte-1smp3z2"&&(be.innerHTML=qo),Jt=a(e),p(ve.$$.fragment,e),Lt=a(e),ye=l(e,"P",{"data-svelte-h":!0}),_(ye)!=="svelte-1i9c3my"&&(ye.innerHTML=Zo),Ut=a(e),Te=l(e,"TABLE",{"data-svelte-h":!0}),_(Te)!=="svelte-rg1rpw"&&(Te.innerHTML=Po),It=a(e),we=l(e,"P",{"data-svelte-h":!0}),_(we)!=="svelte-tkg6bs"&&(we.innerHTML=Ao),Wt=a(e),ke=l(e,"TABLE",{"data-svelte-h":!0}),_(ke)!=="svelte-kgjah5"&&(ke.innerHTML=Go),jt=a(e),$e=l(e,"P",{"data-svelte-h":!0}),_($e)!=="svelte-oe9pwo"&&($e.innerHTML=So),Ht=a(e),Me=l(e,"TABLE",{"data-svelte-h":!0}),_(Me)!=="svelte-xb08ow"&&(Me.innerHTML=Qo),Bt=a(e),Ce=l(e,"P",{"data-svelte-h":!0}),_(Ce)!=="svelte-kvmtym"&&(Ce.innerHTML=Eo),qt=a(e),p(Ne.$$.fragment,e),Zt=a(e),xe=l(e,"P",{"data-svelte-h":!0}),_(xe)!=="svelte-1viguyj"&&(xe.textContent=Ro),Pt=a(e),p(ze.$$.fragment,e),At=a(e),p(Fe.$$.fragment,e),Gt=a(e),z=l(e,"DIV",{class:!0});var Z=C(z);p(Je.$$.fragment,Z),ro=a(Z),Ve=l(Z,"P",{"data-svelte-h":!0}),_(Ve)!=="svelte-oac1mg"&&(Ve.innerHTML=Do),io=a(Z),p(Q.$$.fragment,Z),Z.forEach(o),St=a(e),p(Le.$$.fragment,e),Qt=a(e),k=l(e,"DIV",{class:!0});var M=C(k);p(Ue.$$.fragment,M),lo=a(M),Ye=l(M,"P",{"data-svelte-h":!0}),_(Ye)!=="svelte-hf9zau"&&(Ye.textContent=Xo),co=a(M),Ke=l(M,"P",{"data-svelte-h":!0}),_(Ke)!=="svelte-q52n56"&&(Ke.innerHTML=Oo),po=a(M),et=l(M,"P",{"data-svelte-h":!0}),_(et)!=="svelte-hswkmf"&&(et.innerHTML=Vo),mo=a(M),L=l(M,"DIV",{class:!0});var P=C(L);p(Ie.$$.fragment,P),ho=a(P),tt=l(P,"P",{"data-svelte-h":!0}),_(tt)!=="svelte-xukqs9"&&(tt.innerHTML=Yo),uo=a(P),p(E.$$.fragment,P),P.forEach(o),M.forEach(o),Et=a(e),p(We.$$.fragment,e),Rt=a(e),j=l(e,"DIV",{class:!0});var De=C(j);p(je.$$.fragment,De),fo=a(De),x=l(De,"DIV",{class:!0});var F=C(x);p(He.$$.fragment,F),go=a(F),ot=l(F,"P",{"data-svelte-h":!0}),_(ot)!=="svelte-16q4arp"&&(ot.innerHTML=Ko),_o=a(F),p(R.$$.fragment,F),bo=a(F),p(D.$$.fragment,F),F.forEach(o),De.forEach(o),Dt=a(e),p(Be.$$.fragment,e),Xt=a(e),H=l(e,"DIV",{class:!0});var Xe=C(H);p(qe.$$.fragment,Xe),vo=a(Xe),U=l(Xe,"DIV",{class:!0});var A=C(U);p(Ze.$$.fragment,A),yo=a(A),nt=l(A,"P",{"data-svelte-h":!0}),_(nt)!=="svelte-1sal4ui"&&(nt.innerHTML=en),To=a(A),p(X.$$.fragment,A),A.forEach(o),Xe.forEach(o),Ot=a(e),p(Pe.$$.fragment,e),Vt=a(e),B=l(e,"DIV",{class:!0});var no=C(B);p(Ae.$$.fragment,no),wo=a(no),I=l(no,"DIV",{class:!0});var rt=C(I);p(Ge.$$.fragment,rt),ko=a(rt),st=l(rt,"P",{"data-svelte-h":!0}),_(st)!=="svelte-dyrov9"&&(st.innerHTML=tn),$o=a(rt),p(O.$$.fragment,rt),rt.forEach(o),no.forEach(o),Yt=a(e),p(Se.$$.fragment,e),Kt=a(e),q=l(e,"DIV",{class:!0});var so=C(q);p(Qe.$$.fragment,so),Mo=a(so),W=l(so,"DIV",{class:!0});var it=C(W);p(Ee.$$.fragment,it),Co=a(it),at=l(it,"P",{"data-svelte-h":!0}),_(at)!=="svelte-1py4aay"&&(at.innerHTML=on),No=a(it),p(V.$$.fragment,it),it.forEach(o),so.forEach(o),eo=a(e),p(Re.$$.fragment,e),to=a(e),dt=l(e,"P",{}),C(dt).forEach(o),this.h()},h(){$(r,"name","hf:doc:metadata"),$(r,"content",vn),$(S,"class","flex flex-wrap space-x-1"),$(z,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(L,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(k,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(x,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(j,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(U,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(H,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(I,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(B,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(W,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(q,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,t){g(document.head,r),n(e,b,t),n(e,i,t),n(e,v,t),n(e,N,t),n(e,G,t),m(Y,e,t),n(e,ct,t),n(e,S,t),n(e,pt,t),m(K,e,t),n(e,mt,t),n(e,ee,t),n(e,ht,t),m(te,e,t),n(e,ut,t),n(e,oe,t),n(e,ft,t),n(e,ne,t),n(e,gt,t),m(se,e,t),n(e,_t,t),n(e,ae,t),n(e,bt,t),m(re,e,t),n(e,vt,t),n(e,ie,t),n(e,yt,t),n(e,de,t),n(e,Tt,t),m(le,e,t),n(e,wt,t),m(ce,e,t),n(e,kt,t),n(e,pe,t),n(e,$t,t),n(e,me,t),n(e,Mt,t),n(e,he,t),n(e,Ct,t),m(ue,e,t),n(e,Nt,t),n(e,fe,t),n(e,xt,t),m(ge,e,t),n(e,zt,t),m(_e,e,t),n(e,Ft,t),n(e,be,t),n(e,Jt,t),m(ve,e,t),n(e,Lt,t),n(e,ye,t),n(e,Ut,t),n(e,Te,t),n(e,It,t),n(e,we,t),n(e,Wt,t),n(e,ke,t),n(e,jt,t),n(e,$e,t),n(e,Ht,t),n(e,Me,t),n(e,Bt,t),n(e,Ce,t),n(e,qt,t),m(Ne,e,t),n(e,Zt,t),n(e,xe,t),n(e,Pt,t),m(ze,e,t),n(e,At,t),m(Fe,e,t),n(e,Gt,t),n(e,z,t),m(Je,z,null),g(z,ro),g(z,Ve),g(z,io),m(Q,z,null),n(e,St,t),m(Le,e,t),n(e,Qt,t),n(e,k,t),m(Ue,k,null),g(k,lo),g(k,Ye),g(k,co),g(k,Ke),g(k,po),g(k,et),g(k,mo),g(k,L),m(Ie,L,null),g(L,ho),g(L,tt),g(L,uo),m(E,L,null),n(e,Et,t),m(We,e,t),n(e,Rt,t),n(e,j,t),m(je,j,null),g(j,fo),g(j,x),m(He,x,null),g(x,go),g(x,ot),g(x,_o),m(R,x,null),g(x,bo),m(D,x,null),n(e,Dt,t),m(Be,e,t),n(e,Xt,t),n(e,H,t),m(qe,H,null),g(H,vo),g(H,U),m(Ze,U,null),g(U,yo),g(U,nt),g(U,To),m(X,U,null),n(e,Ot,t),m(Pe,e,t),n(e,Vt,t),n(e,B,t),m(Ae,B,null),g(B,wo),g(B,I),m(Ge,I,null),g(I,ko),g(I,st),g(I,$o),m(O,I,null),n(e,Yt,t),m(Se,e,t),n(e,Kt,t),n(e,q,t),m(Qe,q,null),g(q,Mo),g(q,W),m(Ee,W,null),g(W,Co),g(W,at),g(W,No),m(V,W,null),n(e,eo,t),m(Re,e,t),n(e,to,t),n(e,dt,t),oo=!0},p(e,[t]){const Z={};t&2&&(Z.$$scope={dirty:t,ctx:e}),Q.$set(Z);const M={};t&2&&(M.$$scope={dirty:t,ctx:e}),E.$set(M);const P={};t&2&&(P.$$scope={dirty:t,ctx:e}),R.$set(P);const De={};t&2&&(De.$$scope={dirty:t,ctx:e}),D.$set(De);const F={};t&2&&(F.$$scope={dirty:t,ctx:e}),X.$set(F);const Xe={};t&2&&(Xe.$$scope={dirty:t,ctx:e}),O.$set(Xe);const A={};t&2&&(A.$$scope={dirty:t,ctx:e}),V.$set(A)},i(e){oo||(h(Y.$$.fragment,e),h(K.$$.fragment,e),h(te.$$.fragment,e),h(se.$$.fragment,e),h(re.$$.fragment,e),h(le.$$.fragment,e),h(ce.$$.fragment,e),h(ue.$$.fragment,e),h(ge.$$.fragment,e),h(_e.$$.fragment,e),h(ve.$$.fragment,e),h(Ne.$$.fragment,e),h(ze.$$.fragment,e),h(Fe.$$.fragment,e),h(Je.$$.fragment,e),h(Q.$$.fragment,e),h(Le.$$.fragment,e),h(Ue.$$.fragment,e),h(Ie.$$.fragment,e),h(E.$$.fragment,e),h(We.$$.fragment,e),h(je.$$.fragment,e),h(He.$$.fragment,e),h(R.$$.fragment,e),h(D.$$.fragment,e),h(Be.$$.fragment,e),h(qe.$$.fragment,e),h(Ze.$$.fragment,e),h(X.$$.fragment,e),h(Pe.$$.fragment,e),h(Ae.$$.fragment,e),h(Ge.$$.fragment,e),h(O.$$.fragment,e),h(Se.$$.fragment,e),h(Qe.$$.fragment,e),h(Ee.$$.fragment,e),h(V.$$.fragment,e),h(Re.$$.fragment,e),oo=!0)},o(e){u(Y.$$.fragment,e),u(K.$$.fragment,e),u(te.$$.fragment,e),u(se.$$.fragment,e),u(re.$$.fragment,e),u(le.$$.fragment,e),u(ce.$$.fragment,e),u(ue.$$.fragment,e),u(ge.$$.fragment,e),u(_e.$$.fragment,e),u(ve.$$.fragment,e),u(Ne.$$.fragment,e),u(ze.$$.fragment,e),u(Fe.$$.fragment,e),u(Je.$$.fragment,e),u(Q.$$.fragment,e),u(Le.$$.fragment,e),u(Ue.$$.fragment,e),u(Ie.$$.fragment,e),u(E.$$.fragment,e),u(We.$$.fragment,e),u(je.$$.fragment,e),u(He.$$.fragment,e),u(R.$$.fragment,e),u(D.$$.fragment,e),u(Be.$$.fragment,e),u(qe.$$.fragment,e),u(Ze.$$.fragment,e),u(X.$$.fragment,e),u(Pe.$$.fragment,e),u(Ae.$$.fragment,e),u(Ge.$$.fragment,e),u(O.$$.fragment,e),u(Se.$$.fragment,e),u(Qe.$$.fragment,e),u(Ee.$$.fragment,e),u(V.$$.fragment,e),u(Re.$$.fragment,e),oo=!1},d(e){e&&(o(b),o(i),o(v),o(N),o(G),o(ct),o(S),o(pt),o(mt),o(ee),o(ht),o(ut),o(oe),o(ft),o(ne),o(gt),o(_t),o(ae),o(bt),o(vt),o(ie),o(yt),o(de),o(Tt),o(wt),o(kt),o(pe),o($t),o(me),o(Mt),o(he),o(Ct),o(Nt),o(fe),o(xt),o(zt),o(Ft),o(be),o(Jt),o(Lt),o(ye),o(Ut),o(Te),o(It),o(we),o(Wt),o(ke),o(jt),o($e),o(Ht),o(Me),o(Bt),o(Ce),o(qt),o(Zt),o(xe),o(Pt),o(At),o(Gt),o(z),o(St),o(Qt),o(k),o(Et),o(Rt),o(j),o(Dt),o(Xt),o(H),o(Ot),o(Vt),o(B),o(Yt),o(Kt),o(q),o(eo),o(to),o(dt)),o(r),f(Y,e),f(K,e),f(te,e),f(se,e),f(re,e),f(le,e),f(ce,e),f(ue,e),f(ge,e),f(_e,e),f(ve,e),f(Ne,e),f(ze,e),f(Fe,e),f(Je),f(Q),f(Le,e),f(Ue),f(Ie),f(E),f(We,e),f(je),f(He),f(R),f(D),f(Be,e),f(qe),f(Ze),f(X),f(Pe,e),f(Ae),f(Ge),f(O),f(Se,e),f(Qe),f(Ee),f(V),f(Re,e)}}}const vn='{"title":"Nemotron","local":"nemotron","sections":[{"title":"License","local":"license","sections":[],"depth":3},{"title":"Description","local":"description","sections":[],"depth":3},{"title":"References","local":"references","sections":[],"depth":3},{"title":"Model Architecture","local":"model-architecture","sections":[],"depth":3},{"title":"Minitron","local":"minitron","sections":[{"title":"Minitron 4B Base","local":"minitron-4b-base","sections":[],"depth":3},{"title":"HuggingFace Quickstart","local":"huggingface-quickstart","sections":[],"depth":3},{"title":"License","local":"license","sections":[],"depth":3},{"title":"Evaluation Results","local":"evaluation-results","sections":[],"depth":3},{"title":"Citation","local":"citation","sections":[],"depth":3}],"depth":2},{"title":"NemotronConfig","local":"transformers.NemotronConfig","sections":[],"depth":2},{"title":"NemotronModel","local":"transformers.NemotronModel","sections":[],"depth":2},{"title":"NemotronForCausalLM","local":"transformers.NemotronForCausalLM","sections":[],"depth":2},{"title":"NemotronForSequenceClassification","local":"transformers.NemotronForSequenceClassification","sections":[],"depth":2},{"title":"NemotronForQuestionAnswering","local":"transformers.NemotronForQuestionAnswering","sections":[],"depth":2},{"title":"NemotronForTokenClassification","local":"transformers.NemotronForTokenClassification","sections":[],"depth":2}],"depth":1}';function yn(w){return an(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class xn extends rn{constructor(r){super(),dn(this,r,yn,bn,sn,{})}}export{xn as component};
