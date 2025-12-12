import{s as ro,o as io,n as Fe}from"../chunks/scheduler.18a86fab.js";import{S as co,i as lo,g as d,s as a,r as p,A as mo,h as c,f as o,c as r,j as q,x as b,u as h,k as F,y as l,a as s,v as u,d as f,t as g,w as _}from"../chunks/index.98837b22.js";import{T as Be}from"../chunks/Tip.77304350.js";import{D as N}from"../chunks/Docstring.a1ef7999.js";import{C as Ne}from"../chunks/CodeBlock.8d0c2e8a.js";import{E as ao}from"../chunks/ExampleCodeBlock.8c3ee1f9.js";import{H as A,E as po}from"../chunks/getInferenceSnippets.06c2775f.js";function ho(w){let n,y=`The <code>Persimmon</code> models were trained using <code>bfloat16</code>, but the original inference uses <code>float16</code> The checkpoints uploaded on the hub use <code>dtype = &#39;float16&#39;</code> which will be
used by the <code>AutoModel</code> API to cast the checkpoints from <code>torch.float32</code> to <code>torch.float16</code>.`,i,m,$="The <code>dtype</code> of the online weights is mostly irrelevant, unless you are using <code>dtype=&quot;auto&quot;</code> when initializing a model using <code>model = AutoModelForCausalLM.from_pretrained(&quot;path&quot;, dtype = &quot;auto&quot;)</code>. The reason is that the model will first be downloaded ( using the <code>dtype</code> of the checkpoints online) then it will be cast to the default <code>dtype</code> of <code>torch</code> (becomes <code>torch.float32</code>). Users should specify the <code>dtype</code> they want, and if they don’t it will be <code>torch.float32</code>.",v,k,I="Finetuning the model in <code>float16</code> is not recommended and known to produce <code>nan</code>, as such the model should be fine-tuned in <code>bfloat16</code>.";return{c(){n=d("p"),n.innerHTML=y,i=a(),m=d("p"),m.innerHTML=$,v=a(),k=d("p"),k.innerHTML=I},l(T){n=c(T,"P",{"data-svelte-h":!0}),b(n)!=="svelte-3e9dts"&&(n.innerHTML=y),i=r(T),m=c(T,"P",{"data-svelte-h":!0}),b(m)!=="svelte-7cui76"&&(m.innerHTML=$),v=r(T),k=c(T,"P",{"data-svelte-h":!0}),b(k)!=="svelte-1p3drsu"&&(k.innerHTML=I)},m(T,C){s(T,n,C),s(T,i,C),s(T,m,C),s(T,v,C),s(T,k,C)},p:Fe,d(T){T&&(o(n),o(i),o(m),o(v),o(k))}}}function uo(w){let n,y;return n=new Ne({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMFBlcnNpbW1vbk1vZGVsJTJDJTIwUGVyc2ltbW9uQ29uZmlnJTBBJTBBJTIzJTIwSW5pdGlhbGl6aW5nJTIwYSUyMFBlcnNpbW1vbiUyMHBlcnNpbW1vbi03YiUyMHN0eWxlJTIwY29uZmlndXJhdGlvbiUwQWNvbmZpZ3VyYXRpb24lMjAlM0QlMjBQZXJzaW1tb25Db25maWcoKQ==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> PersimmonModel, PersimmonConfig

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a Persimmon persimmon-7b style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = PersimmonConfig()`,wrap:!1}}),{c(){p(n.$$.fragment)},l(i){h(n.$$.fragment,i)},m(i,m){u(n,i,m),y=!0},p:Fe,i(i){y||(f(n.$$.fragment,i),y=!0)},o(i){g(n.$$.fragment,i),y=!1},d(i){_(n,i)}}}function fo(w){let n,y=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){n=d("p"),n.innerHTML=y},l(i){n=c(i,"P",{"data-svelte-h":!0}),b(n)!=="svelte-fincs2"&&(n.innerHTML=y)},m(i,m){s(i,n,m)},p:Fe,d(i){i&&o(n)}}}function go(w){let n,y=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){n=d("p"),n.innerHTML=y},l(i){n=c(i,"P",{"data-svelte-h":!0}),b(n)!=="svelte-fincs2"&&(n.innerHTML=y)},m(i,m){s(i,n,m)},p:Fe,d(i){i&&o(n)}}}function _o(w){let n,y="Example:",i,m,$;return m=new Ne({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBQZXJzaW1tb25Gb3JDYXVzYWxMTSUwQSUwQW1vZGVsJTIwJTNEJTIwUGVyc2ltbW9uRm9yQ2F1c2FsTE0uZnJvbV9wcmV0cmFpbmVkKCUyMmFkZXB0JTJGcGVyc2ltbW9uLThiLWJhc2UlMjIpJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIyYWRlcHQlMkZwZXJzaW1tb24tOGItYmFzZSUyMiklMEElMEFwcm9tcHQlMjAlM0QlMjAlMjJodW1hbiUzQSUyMEhleSUyQyUyMHdoYXQlMjBzaG91bGQlMjBJJTIwZWF0JTIwZm9yJTIwZGlubmVyJTNGJTIyJTBBaW5wdXRzJTIwJTNEJTIwdG9rZW5pemVyKHByb21wdCUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpJTBBJTBBJTIzJTIwR2VuZXJhdGUlMEFnZW5lcmF0ZV9pZHMlMjAlM0QlMjBtb2RlbC5nZW5lcmF0ZShpbnB1dHMuaW5wdXRfaWRzJTJDJTIwbWF4X2xlbmd0aCUzRDMwKSUwQXRva2VuaXplci5iYXRjaF9kZWNvZGUoZ2VuZXJhdGVfaWRzJTJDJTIwc2tpcF9zcGVjaWFsX3Rva2VucyUzRFRydWUlMkMlMjBjbGVhbl91cF90b2tlbml6YXRpb25fc3BhY2VzJTNERmFsc2UpJTVCMCU1RA==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, PersimmonForCausalLM

<span class="hljs-meta">&gt;&gt;&gt; </span>model = PersimmonForCausalLM.from_pretrained(<span class="hljs-string">&quot;adept/persimmon-8b-base&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;adept/persimmon-8b-base&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>prompt = <span class="hljs-string">&quot;human: Hey, what should I eat for dinner?&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(prompt, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Generate</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>generate_ids = model.generate(inputs.input_ids, max_length=<span class="hljs-number">30</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer.batch_decode(generate_ids, skip_special_tokens=<span class="hljs-literal">True</span>, clean_up_tokenization_spaces=<span class="hljs-literal">False</span>)[<span class="hljs-number">0</span>]
<span class="hljs-string">&#x27;human: Hey, what should I eat for dinner?\\n\\ncat: 🐱\\n\\nhuman: 😐\\n\\n&#x27;</span>`,wrap:!1}}),{c(){n=d("p"),n.textContent=y,i=a(),p(m.$$.fragment)},l(v){n=c(v,"P",{"data-svelte-h":!0}),b(n)!=="svelte-11lpom8"&&(n.textContent=y),i=r(v),h(m.$$.fragment,v)},m(v,k){s(v,n,k),s(v,i,k),u(m,v,k),$=!0},p:Fe,i(v){$||(f(m.$$.fragment,v),$=!0)},o(v){g(m.$$.fragment,v),$=!1},d(v){v&&(o(n),o(i)),_(m,v)}}}function bo(w){let n,y=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){n=d("p"),n.innerHTML=y},l(i){n=c(i,"P",{"data-svelte-h":!0}),b(n)!=="svelte-fincs2"&&(n.innerHTML=y)},m(i,m){s(i,n,m)},p:Fe,d(i){i&&o(n)}}}function yo(w){let n,y=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){n=d("p"),n.innerHTML=y},l(i){n=c(i,"P",{"data-svelte-h":!0}),b(n)!=="svelte-fincs2"&&(n.innerHTML=y)},m(i,m){s(i,n,m)},p:Fe,d(i){i&&o(n)}}}function vo(w){let n,y,i,m,$,v="<em>This model was released on 2023-09-07 and added to Hugging Face Transformers on 2023-09-12.</em>",k,I,T,C,Ht='<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/>',Oe,Q,Ve,K,jt='The Persimmon model was created by <a href="https://www.adept.ai/blog/persimmon-8b" rel="nofollow">ADEPT</a>, and authored by Erich Elsen, Augustus Odena, Maxwell Nye, Sağnak Taşırlar, Tri Dao, Curtis Hawthorne, Deepak Moparthi, Arushi Somani.',Xe,ee,Rt="The authors introduced Persimmon-8B, a decoder model based on the classic transformers architecture, with query and key normalization. Persimmon-8B is a fully permissively-licensed model with approximately 8 billion parameters, released under the Apache license.  Some of the key attributes of Persimmon-8B are long context size (16K), performance, and capabilities for multimodal extensions.",Se,te,Bt="The authors showcase their approach to model evaluation, focusing on practical text generation, mirroring how users interact with language models. The work also includes a comparative analysis, pitting Persimmon-8B against other prominent models (MPT 7B Instruct and Llama 2 Base 7B 1-Shot), across various evaluation tasks. The results demonstrate Persimmon-8B’s competitive performance, even with limited training data.",De,oe,Nt="In terms of model details, the work outlines the architecture and training methodology of Persimmon-8B, providing insights into its design choices, sequence length, and dataset composition. The authors present a fast inference code that outperforms traditional implementations through operator fusion and CUDA graph utilization while maintaining code coherence. They express their anticipation of how the community will leverage this contribution to drive innovation, hinting at further upcoming releases as part of an ongoing series of developments.",Ye,ne,Ot=`This model was contributed by <a href="https://huggingface.co/ArthurZ" rel="nofollow">ArthurZ</a>.
The original code can be found <a href="https://github.com/persimmon-ai-labs/adept-inference" rel="nofollow">here</a>.`,Ee,se,Ae,O,Qe,ae,Vt="Tips:",Ke,re,Xt="<li>To convert the model, you need to clone the original repository using <code>git clone https://github.com/persimmon-ai-labs/adept-inference</code>, then get the checkpoints:</li>",et,ie,tt,de,St="For the chat model:",ot,ce,nt,le,Dt="Thereafter, models can be loaded via:",st,me,at,pe,Yt=`<li><p>Perismmon uses a <code>sentencepiece</code> based tokenizer, with a <code>Unigram</code> model. It supports bytefallback, which is only available in <code>tokenizers==0.14.0</code> for the fast tokenizer.
The <code>LlamaTokenizer</code> is used as it is a standard wrapper around sentencepiece. The <code>chat</code> template will be updated with the templating functions in a follow up PR!</p></li> <li><p>The authors suggest to use the following prompt format for the chat mode: <code>f&quot;human: {prompt}\\n\\nadept:&quot;</code></p></li>`,rt,he,it,P,ue,vt,Je,Et=`This is the configuration class to store the configuration of a <a href="/docs/transformers/v4.56.2/en/model_doc/persimmon#transformers.PersimmonModel">PersimmonModel</a>. It is used to instantiate an
Persimmon model according to the specified arguments, defining the model architecture. Instantiating a
configuration with the defaults will yield a similar configuration to that of the
<a href="https://huggingface.co/adept/persimmon-8b-base" rel="nofollow">adept/persimmon-8b-base</a>.`,Tt,Le,At=`Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> and can be used to control the model outputs. Read the
documentation from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> for more information.`,wt,V,dt,fe,ct,x,ge,kt,qe,Qt="The bare Persimmon Model outputting raw hidden-states without any specific head on top.",$t,Ie,Kt=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,xt,We,eo=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Ct,W,_e,Pt,Ue,to='The <a href="/docs/transformers/v4.56.2/en/model_doc/persimmon#transformers.PersimmonModel">PersimmonModel</a> forward method, overrides the <code>__call__</code> special method.',Mt,X,lt,be,mt,Z,ye,zt,z,ve,Ft,Ge,oo='The <a href="/docs/transformers/v4.56.2/en/model_doc/persimmon#transformers.PersimmonForCausalLM">PersimmonForCausalLM</a> forward method, overrides the <code>__call__</code> special method.',Jt,S,Lt,D,pt,Te,ht,H,we,qt,U,ke,It,Ze,no="The <code>GenericForSequenceClassification</code> forward method, overrides the <code>__call__</code> special method.",Wt,Y,ut,$e,ft,j,xe,Ut,G,Ce,Gt,He,so="The <code>GenericForTokenClassification</code> forward method, overrides the <code>__call__</code> special method.",Zt,E,gt,Pe,_t,Re,bt;return I=new A({props:{title:"Persimmon",local:"persimmon",headingTag:"h1"}}),Q=new A({props:{title:"Overview",local:"overview",headingTag:"h2"}}),se=new A({props:{title:"Usage tips",local:"usage-tips",headingTag:"h2"}}),O=new Be({props:{warning:!0,$$slots:{default:[ho]},$$scope:{ctx:w}}}),ie=new Ne({props:{code:"Z2l0JTIwY2xvbmUlMjBodHRwcyUzQSUyRiUyRmdpdGh1Yi5jb20lMkZwZXJzaW1tb24tYWktbGFicyUyRmFkZXB0LWluZmVyZW5jZSUwQXdnZXQlMjBodHRwcyUzQSUyRiUyRmF4dGtuNHhsNWNpcC5vYmplY3RzdG9yYWdlLnVzLXBob2VuaXgtMS5vY2kuY3VzdG9tZXItb2NpLmNvbSUyRm4lMkZheHRrbjR4bDVjaXAlMkZiJTJGYWRlcHQtcHVibGljLWRhdGElMkZvJTJGOGJfYmFzZV9tb2RlbF9yZWxlYXNlLnRhciUwQXRhciUyMC14dmYlMjA4Yl9iYXNlX21vZGVsX3JlbGVhc2UudGFyJTBBcHl0aG9uJTIwc3JjJTJGdHJhbnNmb3JtZXJzJTJGbW9kZWxzJTJGcGVyc2ltbW9uJTJGY29udmVydF9wZXJzaW1tb25fd2VpZ2h0c190b19oZi5weSUyMCUyMC0taW5wdXRfZGlyJTIwJTJGcGF0aCUyRnRvJTJGZG93bmxvYWRlZCUyRnBlcnNpbW1vbiUyRndlaWdodHMlMkYlMjAtLW91dHB1dF9kaXIlMjAlMkZvdXRwdXQlMkZwYXRoJTIwJTVDJTBBJTIwJTIwJTIwJTIwLS1wdF9tb2RlbF9wYXRoJTIwJTJGcGF0aCUyRnRvJTJGOGJfY2hhdF9tb2RlbF9yZWxlYXNlJTJGaXRlcl8wMDAxMjUxJTJGbXBfcmFua18wMCUyRm1vZGVsX29wdGltX3JuZy5wdCUwQSUyMCUyMCUyMCUyMC0tYWRhX2xpYl9wYXRoJTIwJTJGcGF0aCUyRnRvJTJGYWRlcHQtaW5mZXJlbmNl",highlighted:`git <span class="hljs-built_in">clone</span> https://github.com/persimmon-ai-labs/adept-inference
wget https://axtkn4xl5cip.objectstorage.us-phoenix-1.oci.customer-oci.com/n/axtkn4xl5cip/b/adept-public-data/o/8b_base_model_release.tar
tar -xvf 8b_base_model_release.tar
python src/transformers/models/persimmon/convert_persimmon_weights_to_hf.py  --input_dir /path/to/downloaded/persimmon/weights/ --output_dir /output/path \\
    --pt_model_path /path/to/8b_chat_model_release/iter_0001251/mp_rank_00/model_optim_rng.pt
    --ada_lib_path /path/to/adept-inference`,wrap:!1}}),ce=new Ne({props:{code:"d2dldCUyMGh0dHBzJTNBJTJGJTJGYXh0a240eGw1Y2lwLm9iamVjdHN0b3JhZ2UudXMtcGhvZW5peC0xLm9jaS5jdXN0b21lci1vY2kuY29tJTJGbiUyRmF4dGtuNHhsNWNpcCUyRmIlMkZhZGVwdC1wdWJsaWMtZGF0YSUyRm8lMkY4Yl9jaGF0X21vZGVsX3JlbGVhc2UudGFyJTBBdGFyJTIwLXh2ZiUyMDhiX2Jhc2VfbW9kZWxfcmVsZWFzZS50YXI=",highlighted:`wget https://axtkn4xl5cip.objectstorage.us-phoenix-1.oci.customer-oci.com/n/axtkn4xl5cip/b/adept-public-data/o/8b_chat_model_release.tar
tar -xvf 8b_base_model_release.tar`,wrap:!1}}),me=new Ne({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMFBlcnNpbW1vbkZvckNhdXNhbExNJTJDJTIwUGVyc2ltbW9uVG9rZW5pemVyJTBBJTBBbW9kZWwlMjAlM0QlMjBQZXJzaW1tb25Gb3JDYXVzYWxMTS5mcm9tX3ByZXRyYWluZWQoJTIyJTJGb3V0cHV0JTJGcGF0aCUyMiklMEF0b2tlbml6ZXIlMjAlM0QlMjBQZXJzaW1tb25Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMiUyRm91dHB1dCUyRnBhdGglMjIp",highlighted:`<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> PersimmonForCausalLM, PersimmonTokenizer

model = PersimmonForCausalLM.from_pretrained(<span class="hljs-string">&quot;/output/path&quot;</span>)
tokenizer = PersimmonTokenizer.from_pretrained(<span class="hljs-string">&quot;/output/path&quot;</span>)`,wrap:!1}}),he=new A({props:{title:"PersimmonConfig",local:"transformers.PersimmonConfig",headingTag:"h2"}}),ue=new N({props:{name:"class transformers.PersimmonConfig",anchor:"transformers.PersimmonConfig",parameters:[{name:"vocab_size",val:" = 262144"},{name:"hidden_size",val:" = 4096"},{name:"intermediate_size",val:" = 16384"},{name:"num_hidden_layers",val:" = 36"},{name:"num_attention_heads",val:" = 64"},{name:"hidden_act",val:" = 'relu2'"},{name:"max_position_embeddings",val:" = 16384"},{name:"initializer_range",val:" = 0.02"},{name:"layer_norm_eps",val:" = 1e-05"},{name:"use_cache",val:" = True"},{name:"tie_word_embeddings",val:" = False"},{name:"rope_theta",val:" = 25000.0"},{name:"rope_scaling",val:" = None"},{name:"qk_layernorm",val:" = True"},{name:"hidden_dropout",val:" = 0.0"},{name:"attention_dropout",val:" = 0.0"},{name:"partial_rotary_factor",val:" = 0.5"},{name:"pad_token_id",val:" = None"},{name:"bos_token_id",val:" = 1"},{name:"eos_token_id",val:" = 2"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.PersimmonConfig.vocab_size",description:`<strong>vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 262144) &#x2014;
Vocabulary size of the Persimmon model. Defines the number of different tokens that can be represented by
the <code>inputs_ids</code> passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/persimmon#transformers.PersimmonModel">PersimmonModel</a>`,name:"vocab_size"},{anchor:"transformers.PersimmonConfig.hidden_size",description:`<strong>hidden_size</strong> (<code>int</code>, <em>optional</em>, defaults to 4096) &#x2014;
Dimension of the hidden representations.`,name:"hidden_size"},{anchor:"transformers.PersimmonConfig.intermediate_size",description:`<strong>intermediate_size</strong> (<code>int</code>, <em>optional</em>, defaults to 16384) &#x2014;
Dimension of the MLP representations.`,name:"intermediate_size"},{anchor:"transformers.PersimmonConfig.num_hidden_layers",description:`<strong>num_hidden_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 36) &#x2014;
Number of hidden layers in the Transformer encoder.`,name:"num_hidden_layers"},{anchor:"transformers.PersimmonConfig.num_attention_heads",description:`<strong>num_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 64) &#x2014;
Number of attention heads for each attention layer in the Transformer encoder.`,name:"num_attention_heads"},{anchor:"transformers.PersimmonConfig.hidden_act",description:`<strong>hidden_act</strong> (<code>str</code> or <code>function</code>, <em>optional</em>, defaults to <code>&quot;relu2&quot;</code>) &#x2014;
The non-linear activation function (function or string) in the decoder.`,name:"hidden_act"},{anchor:"transformers.PersimmonConfig.max_position_embeddings",description:`<strong>max_position_embeddings</strong> (<code>int</code>, <em>optional</em>, defaults to 16384) &#x2014;
The maximum sequence length that this model might ever be used with.`,name:"max_position_embeddings"},{anchor:"transformers.PersimmonConfig.initializer_range",description:`<strong>initializer_range</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.`,name:"initializer_range"},{anchor:"transformers.PersimmonConfig.layer_norm_eps",description:`<strong>layer_norm_eps</strong> (<code>float</code>, <em>optional</em>, defaults to 1e-5) &#x2014;
The epsilon used by the rms normalization layers.`,name:"layer_norm_eps"},{anchor:"transformers.PersimmonConfig.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not the model should return the last key/values attentions (not used by all models). Only
relevant if <code>config.is_decoder=True</code>.`,name:"use_cache"},{anchor:"transformers.PersimmonConfig.tie_word_embeddings(bool,",description:`<strong>tie_word_embeddings(<code>bool</code>,</strong> <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether to tie weight embeddings`,name:"tie_word_embeddings(bool,"},{anchor:"transformers.PersimmonConfig.rope_theta",description:`<strong>rope_theta</strong> (<code>float</code>, <em>optional</em>, defaults to 25000.0) &#x2014;
The base period of the RoPE embeddings.`,name:"rope_theta"},{anchor:"transformers.PersimmonConfig.rope_scaling",description:`<strong>rope_scaling</strong> (<code>Dict</code>, <em>optional</em>) &#x2014;
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
Only used with &#x2018;llama3&#x2019;. Scaling factor applied to high frequency components of the RoPE`,name:"rope_scaling"},{anchor:"transformers.PersimmonConfig.qk_layernorm",description:`<strong>qk_layernorm</strong> (<code>bool</code>, <em>optional</em>, default to <code>True</code>) &#x2014;
Whether or not to normalize the Queries and Keys after projecting the hidden states`,name:"qk_layernorm"},{anchor:"transformers.PersimmonConfig.hidden_dropout",description:`<strong>hidden_dropout</strong> (<code>float</code>, <em>optional</em>, default to 0.0) &#x2014;
The dropout ratio after applying the MLP to the hidden states.`,name:"hidden_dropout"},{anchor:"transformers.PersimmonConfig.attention_dropout",description:`<strong>attention_dropout</strong> (<code>float</code>, <em>optional</em>, default to 0.0) &#x2014;
The dropout ratio after computing the attention scores.`,name:"attention_dropout"},{anchor:"transformers.PersimmonConfig.partial_rotary_factor",description:`<strong>partial_rotary_factor</strong> (<code>float</code>, <em>optional</em>, default to 0.5) &#x2014;
Percentage of the query and keys which will have rotary embedding.`,name:"partial_rotary_factor"},{anchor:"transformers.PersimmonConfig.Example",description:"<strong>Example</strong> &#x2014;",name:"Example"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/persimmon/configuration_persimmon.py#L25"}}),V=new ao({props:{anchor:"transformers.PersimmonConfig.example",$$slots:{default:[uo]},$$scope:{ctx:w}}}),fe=new A({props:{title:"PersimmonModel",local:"transformers.PersimmonModel",headingTag:"h2"}}),ge=new N({props:{name:"class transformers.PersimmonModel",anchor:"transformers.PersimmonModel",parameters:[{name:"config",val:": PersimmonConfig"}],parametersDescription:[{anchor:"transformers.PersimmonModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/persimmon#transformers.PersimmonConfig">PersimmonConfig</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/persimmon/modeling_persimmon.py#L419"}}),_e=new N({props:{name:"forward",anchor:"transformers.PersimmonModel.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.modeling_flash_attention_utils.FlashAttentionKwargs]"}],parametersDescription:[{anchor:"transformers.PersimmonModel.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.PersimmonModel.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.PersimmonModel.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.PersimmonModel.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.PersimmonModel.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.PersimmonModel.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.PersimmonModel.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.PersimmonModel.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.PersimmonModel.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/persimmon/modeling_persimmon.py#L444",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast"
>transformers.modeling_outputs.BaseModelOutputWithPast</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/persimmon#transformers.PersimmonConfig"
>PersimmonConfig</a>) and inputs.</p>
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
`}}),X=new Be({props:{$$slots:{default:[fo]},$$scope:{ctx:w}}}),be=new A({props:{title:"PersimmonForCausalLM",local:"transformers.PersimmonForCausalLM",headingTag:"h2"}}),ye=new N({props:{name:"class transformers.PersimmonForCausalLM",anchor:"transformers.PersimmonForCausalLM",parameters:[{name:"config",val:""}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/persimmon/modeling_persimmon.py#L666"}}),ve=new N({props:{name:"forward",anchor:"transformers.PersimmonForCausalLM.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"},{name:"logits_to_keep",val:": typing.Union[int, torch.Tensor] = 0"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.PersimmonForCausalLM.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.PersimmonForCausalLM.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.PersimmonForCausalLM.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.PersimmonForCausalLM.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.PersimmonForCausalLM.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.PersimmonForCausalLM.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.PersimmonForCausalLM.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.PersimmonForCausalLM.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.PersimmonForCausalLM.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.PersimmonForCausalLM.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"},{anchor:"transformers.PersimmonForCausalLM.forward.logits_to_keep",description:`<strong>logits_to_keep</strong> (<code>Union[int, torch.Tensor]</code>, defaults to <code>0</code>) &#x2014;
If an <code>int</code>, compute logits for the last <code>logits_to_keep</code> tokens. If <code>0</code>, calculate logits for all
<code>input_ids</code> (special case). Only last token logits are needed for generation, and calculating them only for that
token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
If a <code>torch.Tensor</code>, must be 1D corresponding to the indices to keep in the sequence length dimension.
This is useful when using packed tensor format (single dimension for batch and sequence length).`,name:"logits_to_keep"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/persimmon/modeling_persimmon.py#L679",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast"
>transformers.modeling_outputs.CausalLMOutputWithPast</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/persimmon#transformers.PersimmonConfig"
>PersimmonConfig</a>) and inputs.</p>
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
`}}),S=new Be({props:{$$slots:{default:[go]},$$scope:{ctx:w}}}),D=new ao({props:{anchor:"transformers.PersimmonForCausalLM.forward.example",$$slots:{default:[_o]},$$scope:{ctx:w}}}),Te=new A({props:{title:"PersimmonForSequenceClassification",local:"transformers.PersimmonForSequenceClassification",headingTag:"h2"}}),we=new N({props:{name:"class transformers.PersimmonForSequenceClassification",anchor:"transformers.PersimmonForSequenceClassification",parameters:[{name:"config",val:""}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/persimmon/modeling_persimmon.py#L761"}}),ke=new N({props:{name:"forward",anchor:"transformers.PersimmonForSequenceClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.PersimmonForSequenceClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.PersimmonForSequenceClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.PersimmonForSequenceClassification.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.PersimmonForSequenceClassification.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.PersimmonForSequenceClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.PersimmonForSequenceClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.PersimmonForSequenceClassification.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
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
`}}),Y=new Be({props:{$$slots:{default:[bo]},$$scope:{ctx:w}}}),$e=new A({props:{title:"PersimmonForTokenClassification",local:"transformers.PersimmonForTokenClassification",headingTag:"h2"}}),xe=new N({props:{name:"class transformers.PersimmonForTokenClassification",anchor:"transformers.PersimmonForTokenClassification",parameters:[{name:"config",val:""}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/persimmon/modeling_persimmon.py#L764"}}),Ce=new N({props:{name:"forward",anchor:"transformers.PersimmonForTokenClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.PersimmonForTokenClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.PersimmonForTokenClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.PersimmonForTokenClassification.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.PersimmonForTokenClassification.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.PersimmonForTokenClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.PersimmonForTokenClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.PersimmonForTokenClassification.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
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
`}}),E=new Be({props:{$$slots:{default:[yo]},$$scope:{ctx:w}}}),Pe=new po({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/persimmon.md"}}),{c(){n=d("meta"),y=a(),i=d("p"),m=a(),$=d("p"),$.innerHTML=v,k=a(),p(I.$$.fragment),T=a(),C=d("div"),C.innerHTML=Ht,Oe=a(),p(Q.$$.fragment),Ve=a(),K=d("p"),K.innerHTML=jt,Xe=a(),ee=d("p"),ee.textContent=Rt,Se=a(),te=d("p"),te.textContent=Bt,De=a(),oe=d("p"),oe.textContent=Nt,Ye=a(),ne=d("p"),ne.innerHTML=Ot,Ee=a(),p(se.$$.fragment),Ae=a(),p(O.$$.fragment),Qe=a(),ae=d("p"),ae.textContent=Vt,Ke=a(),re=d("ul"),re.innerHTML=Xt,et=a(),p(ie.$$.fragment),tt=a(),de=d("p"),de.textContent=St,ot=a(),p(ce.$$.fragment),nt=a(),le=d("p"),le.textContent=Dt,st=a(),p(me.$$.fragment),at=a(),pe=d("ul"),pe.innerHTML=Yt,rt=a(),p(he.$$.fragment),it=a(),P=d("div"),p(ue.$$.fragment),vt=a(),Je=d("p"),Je.innerHTML=Et,Tt=a(),Le=d("p"),Le.innerHTML=At,wt=a(),p(V.$$.fragment),dt=a(),p(fe.$$.fragment),ct=a(),x=d("div"),p(ge.$$.fragment),kt=a(),qe=d("p"),qe.textContent=Qt,$t=a(),Ie=d("p"),Ie.innerHTML=Kt,xt=a(),We=d("p"),We.innerHTML=eo,Ct=a(),W=d("div"),p(_e.$$.fragment),Pt=a(),Ue=d("p"),Ue.innerHTML=to,Mt=a(),p(X.$$.fragment),lt=a(),p(be.$$.fragment),mt=a(),Z=d("div"),p(ye.$$.fragment),zt=a(),z=d("div"),p(ve.$$.fragment),Ft=a(),Ge=d("p"),Ge.innerHTML=oo,Jt=a(),p(S.$$.fragment),Lt=a(),p(D.$$.fragment),pt=a(),p(Te.$$.fragment),ht=a(),H=d("div"),p(we.$$.fragment),qt=a(),U=d("div"),p(ke.$$.fragment),It=a(),Ze=d("p"),Ze.innerHTML=no,Wt=a(),p(Y.$$.fragment),ut=a(),p($e.$$.fragment),ft=a(),j=d("div"),p(xe.$$.fragment),Ut=a(),G=d("div"),p(Ce.$$.fragment),Gt=a(),He=d("p"),He.innerHTML=so,Zt=a(),p(E.$$.fragment),gt=a(),p(Pe.$$.fragment),_t=a(),Re=d("p"),this.h()},l(e){const t=mo("svelte-u9bgzb",document.head);n=c(t,"META",{name:!0,content:!0}),t.forEach(o),y=r(e),i=c(e,"P",{}),q(i).forEach(o),m=r(e),$=c(e,"P",{"data-svelte-h":!0}),b($)!=="svelte-9s07h6"&&($.innerHTML=v),k=r(e),h(I.$$.fragment,e),T=r(e),C=c(e,"DIV",{class:!0,"data-svelte-h":!0}),b(C)!=="svelte-13t8s2t"&&(C.innerHTML=Ht),Oe=r(e),h(Q.$$.fragment,e),Ve=r(e),K=c(e,"P",{"data-svelte-h":!0}),b(K)!=="svelte-1dqqllh"&&(K.innerHTML=jt),Xe=r(e),ee=c(e,"P",{"data-svelte-h":!0}),b(ee)!=="svelte-tohaaf"&&(ee.textContent=Rt),Se=r(e),te=c(e,"P",{"data-svelte-h":!0}),b(te)!=="svelte-gxqz2j"&&(te.textContent=Bt),De=r(e),oe=c(e,"P",{"data-svelte-h":!0}),b(oe)!=="svelte-l2kkzt"&&(oe.textContent=Nt),Ye=r(e),ne=c(e,"P",{"data-svelte-h":!0}),b(ne)!=="svelte-1kv44aj"&&(ne.innerHTML=Ot),Ee=r(e),h(se.$$.fragment,e),Ae=r(e),h(O.$$.fragment,e),Qe=r(e),ae=c(e,"P",{"data-svelte-h":!0}),b(ae)!=="svelte-axv494"&&(ae.textContent=Vt),Ke=r(e),re=c(e,"UL",{"data-svelte-h":!0}),b(re)!=="svelte-g9ue9k"&&(re.innerHTML=Xt),et=r(e),h(ie.$$.fragment,e),tt=r(e),de=c(e,"P",{"data-svelte-h":!0}),b(de)!=="svelte-1cs7acv"&&(de.textContent=St),ot=r(e),h(ce.$$.fragment,e),nt=r(e),le=c(e,"P",{"data-svelte-h":!0}),b(le)!=="svelte-nia5es"&&(le.textContent=Dt),st=r(e),h(me.$$.fragment,e),at=r(e),pe=c(e,"UL",{"data-svelte-h":!0}),b(pe)!=="svelte-yk84b4"&&(pe.innerHTML=Yt),rt=r(e),h(he.$$.fragment,e),it=r(e),P=c(e,"DIV",{class:!0});var J=q(P);h(ue.$$.fragment,J),vt=r(J),Je=c(J,"P",{"data-svelte-h":!0}),b(Je)!=="svelte-uj09f0"&&(Je.innerHTML=Et),Tt=r(J),Le=c(J,"P",{"data-svelte-h":!0}),b(Le)!=="svelte-1ek1ss9"&&(Le.innerHTML=At),wt=r(J),h(V.$$.fragment,J),J.forEach(o),dt=r(e),h(fe.$$.fragment,e),ct=r(e),x=c(e,"DIV",{class:!0});var M=q(x);h(ge.$$.fragment,M),kt=r(M),qe=c(M,"P",{"data-svelte-h":!0}),b(qe)!=="svelte-qpvziu"&&(qe.textContent=Qt),$t=r(M),Ie=c(M,"P",{"data-svelte-h":!0}),b(Ie)!=="svelte-q52n56"&&(Ie.innerHTML=Kt),xt=r(M),We=c(M,"P",{"data-svelte-h":!0}),b(We)!=="svelte-hswkmf"&&(We.innerHTML=eo),Ct=r(M),W=c(M,"DIV",{class:!0});var R=q(W);h(_e.$$.fragment,R),Pt=r(R),Ue=c(R,"P",{"data-svelte-h":!0}),b(Ue)!=="svelte-qt93r1"&&(Ue.innerHTML=to),Mt=r(R),h(X.$$.fragment,R),R.forEach(o),M.forEach(o),lt=r(e),h(be.$$.fragment,e),mt=r(e),Z=c(e,"DIV",{class:!0});var Me=q(Z);h(ye.$$.fragment,Me),zt=r(Me),z=c(Me,"DIV",{class:!0});var L=q(z);h(ve.$$.fragment,L),Ft=r(L),Ge=c(L,"P",{"data-svelte-h":!0}),b(Ge)!=="svelte-m370nd"&&(Ge.innerHTML=oo),Jt=r(L),h(S.$$.fragment,L),Lt=r(L),h(D.$$.fragment,L),L.forEach(o),Me.forEach(o),pt=r(e),h(Te.$$.fragment,e),ht=r(e),H=c(e,"DIV",{class:!0});var ze=q(H);h(we.$$.fragment,ze),qt=r(ze),U=c(ze,"DIV",{class:!0});var B=q(U);h(ke.$$.fragment,B),It=r(B),Ze=c(B,"P",{"data-svelte-h":!0}),b(Ze)!=="svelte-1sal4ui"&&(Ze.innerHTML=no),Wt=r(B),h(Y.$$.fragment,B),B.forEach(o),ze.forEach(o),ut=r(e),h($e.$$.fragment,e),ft=r(e),j=c(e,"DIV",{class:!0});var yt=q(j);h(xe.$$.fragment,yt),Ut=r(yt),G=c(yt,"DIV",{class:!0});var je=q(G);h(Ce.$$.fragment,je),Gt=r(je),He=c(je,"P",{"data-svelte-h":!0}),b(He)!=="svelte-1py4aay"&&(He.innerHTML=so),Zt=r(je),h(E.$$.fragment,je),je.forEach(o),yt.forEach(o),gt=r(e),h(Pe.$$.fragment,e),_t=r(e),Re=c(e,"P",{}),q(Re).forEach(o),this.h()},h(){F(n,"name","hf:doc:metadata"),F(n,"content",To),F(C,"class","flex flex-wrap space-x-1"),F(P,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),F(W,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),F(x,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),F(z,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),F(Z,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),F(U,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),F(H,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),F(G,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),F(j,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,t){l(document.head,n),s(e,y,t),s(e,i,t),s(e,m,t),s(e,$,t),s(e,k,t),u(I,e,t),s(e,T,t),s(e,C,t),s(e,Oe,t),u(Q,e,t),s(e,Ve,t),s(e,K,t),s(e,Xe,t),s(e,ee,t),s(e,Se,t),s(e,te,t),s(e,De,t),s(e,oe,t),s(e,Ye,t),s(e,ne,t),s(e,Ee,t),u(se,e,t),s(e,Ae,t),u(O,e,t),s(e,Qe,t),s(e,ae,t),s(e,Ke,t),s(e,re,t),s(e,et,t),u(ie,e,t),s(e,tt,t),s(e,de,t),s(e,ot,t),u(ce,e,t),s(e,nt,t),s(e,le,t),s(e,st,t),u(me,e,t),s(e,at,t),s(e,pe,t),s(e,rt,t),u(he,e,t),s(e,it,t),s(e,P,t),u(ue,P,null),l(P,vt),l(P,Je),l(P,Tt),l(P,Le),l(P,wt),u(V,P,null),s(e,dt,t),u(fe,e,t),s(e,ct,t),s(e,x,t),u(ge,x,null),l(x,kt),l(x,qe),l(x,$t),l(x,Ie),l(x,xt),l(x,We),l(x,Ct),l(x,W),u(_e,W,null),l(W,Pt),l(W,Ue),l(W,Mt),u(X,W,null),s(e,lt,t),u(be,e,t),s(e,mt,t),s(e,Z,t),u(ye,Z,null),l(Z,zt),l(Z,z),u(ve,z,null),l(z,Ft),l(z,Ge),l(z,Jt),u(S,z,null),l(z,Lt),u(D,z,null),s(e,pt,t),u(Te,e,t),s(e,ht,t),s(e,H,t),u(we,H,null),l(H,qt),l(H,U),u(ke,U,null),l(U,It),l(U,Ze),l(U,Wt),u(Y,U,null),s(e,ut,t),u($e,e,t),s(e,ft,t),s(e,j,t),u(xe,j,null),l(j,Ut),l(j,G),u(Ce,G,null),l(G,Gt),l(G,He),l(G,Zt),u(E,G,null),s(e,gt,t),u(Pe,e,t),s(e,_t,t),s(e,Re,t),bt=!0},p(e,[t]){const J={};t&2&&(J.$$scope={dirty:t,ctx:e}),O.$set(J);const M={};t&2&&(M.$$scope={dirty:t,ctx:e}),V.$set(M);const R={};t&2&&(R.$$scope={dirty:t,ctx:e}),X.$set(R);const Me={};t&2&&(Me.$$scope={dirty:t,ctx:e}),S.$set(Me);const L={};t&2&&(L.$$scope={dirty:t,ctx:e}),D.$set(L);const ze={};t&2&&(ze.$$scope={dirty:t,ctx:e}),Y.$set(ze);const B={};t&2&&(B.$$scope={dirty:t,ctx:e}),E.$set(B)},i(e){bt||(f(I.$$.fragment,e),f(Q.$$.fragment,e),f(se.$$.fragment,e),f(O.$$.fragment,e),f(ie.$$.fragment,e),f(ce.$$.fragment,e),f(me.$$.fragment,e),f(he.$$.fragment,e),f(ue.$$.fragment,e),f(V.$$.fragment,e),f(fe.$$.fragment,e),f(ge.$$.fragment,e),f(_e.$$.fragment,e),f(X.$$.fragment,e),f(be.$$.fragment,e),f(ye.$$.fragment,e),f(ve.$$.fragment,e),f(S.$$.fragment,e),f(D.$$.fragment,e),f(Te.$$.fragment,e),f(we.$$.fragment,e),f(ke.$$.fragment,e),f(Y.$$.fragment,e),f($e.$$.fragment,e),f(xe.$$.fragment,e),f(Ce.$$.fragment,e),f(E.$$.fragment,e),f(Pe.$$.fragment,e),bt=!0)},o(e){g(I.$$.fragment,e),g(Q.$$.fragment,e),g(se.$$.fragment,e),g(O.$$.fragment,e),g(ie.$$.fragment,e),g(ce.$$.fragment,e),g(me.$$.fragment,e),g(he.$$.fragment,e),g(ue.$$.fragment,e),g(V.$$.fragment,e),g(fe.$$.fragment,e),g(ge.$$.fragment,e),g(_e.$$.fragment,e),g(X.$$.fragment,e),g(be.$$.fragment,e),g(ye.$$.fragment,e),g(ve.$$.fragment,e),g(S.$$.fragment,e),g(D.$$.fragment,e),g(Te.$$.fragment,e),g(we.$$.fragment,e),g(ke.$$.fragment,e),g(Y.$$.fragment,e),g($e.$$.fragment,e),g(xe.$$.fragment,e),g(Ce.$$.fragment,e),g(E.$$.fragment,e),g(Pe.$$.fragment,e),bt=!1},d(e){e&&(o(y),o(i),o(m),o($),o(k),o(T),o(C),o(Oe),o(Ve),o(K),o(Xe),o(ee),o(Se),o(te),o(De),o(oe),o(Ye),o(ne),o(Ee),o(Ae),o(Qe),o(ae),o(Ke),o(re),o(et),o(tt),o(de),o(ot),o(nt),o(le),o(st),o(at),o(pe),o(rt),o(it),o(P),o(dt),o(ct),o(x),o(lt),o(mt),o(Z),o(pt),o(ht),o(H),o(ut),o(ft),o(j),o(gt),o(_t),o(Re)),o(n),_(I,e),_(Q,e),_(se,e),_(O,e),_(ie,e),_(ce,e),_(me,e),_(he,e),_(ue),_(V),_(fe,e),_(ge),_(_e),_(X),_(be,e),_(ye),_(ve),_(S),_(D),_(Te,e),_(we),_(ke),_(Y),_($e,e),_(xe),_(Ce),_(E),_(Pe,e)}}}const To='{"title":"Persimmon","local":"persimmon","sections":[{"title":"Overview","local":"overview","sections":[],"depth":2},{"title":"Usage tips","local":"usage-tips","sections":[],"depth":2},{"title":"PersimmonConfig","local":"transformers.PersimmonConfig","sections":[],"depth":2},{"title":"PersimmonModel","local":"transformers.PersimmonModel","sections":[],"depth":2},{"title":"PersimmonForCausalLM","local":"transformers.PersimmonForCausalLM","sections":[],"depth":2},{"title":"PersimmonForSequenceClassification","local":"transformers.PersimmonForSequenceClassification","sections":[],"depth":2},{"title":"PersimmonForTokenClassification","local":"transformers.PersimmonForTokenClassification","sections":[],"depth":2}],"depth":1}';function wo(w){return io(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class Fo extends co{constructor(n){super(),lo(this,n,wo,vo,ro,{})}}export{Fo as component};
