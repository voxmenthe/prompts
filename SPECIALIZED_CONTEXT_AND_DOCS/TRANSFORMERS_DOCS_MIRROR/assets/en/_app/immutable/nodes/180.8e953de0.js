import{s as Nn,o as Ln,n as Ye}from"../chunks/scheduler.18a86fab.js";import{S as Bn,i as Qn,g as i,s,r as p,A as Xn,h as l,f as n,c as a,j as A,x as M,u as m,k as U,y as d,a as o,v as u,d as h,t as f,w as g}from"../chunks/index.98837b22.js";import{T as Qt}from"../chunks/Tip.77304350.js";import{D as q}from"../chunks/Docstring.a1ef7999.js";import{C as tt}from"../chunks/CodeBlock.8d0c2e8a.js";import{E as zn}from"../chunks/ExampleCodeBlock.8c3ee1f9.js";import{H as k,E as Zn}from"../chunks/getInferenceSnippets.06c2775f.js";function Gn(j){let r,T="Example:",c,y,b;return y=new tt({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEV4YW9uZTRNb2RlbCUyQyUyMEV4YW9uZTRDb25maWclMEElMEElMjMlMjBJbml0aWFsaXppbmclMjBhJTIwRVhBT05FJTIwY29uZmlndXJhdGlvbiUwQWNvbmZpZ3VyYXRpb24lMjAlM0QlMjBFeGFvbmU0Q29uZmlnKCklMEElMEElMjMlMjBJbml0aWFsaXppbmclMjBhJTIwbW9kZWwlMjBmcm9tJTIwY29uZmlndXJhdGlvbiUwQW1vZGVsJTIwJTNEJTIwRXhhb25lNE1vZGVsKGNvbmZpZ3VyYXRpb24pJTBBJTBBJTIzJTIwQWNjZXNzaW5nJTIwdGhlJTIwbW9kZWwlMjBjb25maWd1cmF0aW9uJTBBY29uZmlndXJhdGlvbiUyMCUzRCUyMG1vZGVsLmNvbmZpZw==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> Exaone4Model, Exaone4Config

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a EXAONE configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = Exaone4Config()

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a model from configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = Exaone4Model(configuration)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Accessing the model configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = model.config`,wrap:!1}}),{c(){r=i("p"),r.textContent=T,c=s(),p(y.$$.fragment)},l(_){r=l(_,"P",{"data-svelte-h":!0}),M(r)!=="svelte-11lpom8"&&(r.textContent=T),c=a(_),m(y.$$.fragment,_)},m(_,I){o(_,r,I),o(_,c,I),u(y,_,I),b=!0},p:Ye,i(_){b||(h(y.$$.fragment,_),b=!0)},o(_){f(y.$$.fragment,_),b=!1},d(_){_&&(n(r),n(c)),g(y,_)}}}function Wn(j){let r,T=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){r=i("p"),r.innerHTML=T},l(c){r=l(c,"P",{"data-svelte-h":!0}),M(r)!=="svelte-fincs2"&&(r.innerHTML=T)},m(c,y){o(c,r,y)},p:Ye,d(c){c&&n(r)}}}function Rn(j){let r,T="Example:",c,y,b;return y=new tt({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Nb2RlbEZvckNhdXNhbExNJTJDJTIwQXV0b1Rva2VuaXplciUwQW1vZGVsJTIwJTNEJTIwQXV0b01vZGVsRm9yQ2F1c2FsTE0uZnJvbV9wcmV0cmFpbmVkKCUyMkxHQUktRVhBT05FJTJGRVhBT05FLTQuMC1JbnN0cnVjdCUyMiklMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJMR0FJLUVYQU9ORSUyRkVYQU9ORS00LjAtSW5zdHJ1Y3QlMjIpJTBBJTBBcHJvbXB0JTIwJTNEJTIwJTIyRXhwbGFpbiUyMGhvdyUyMHdvbmRlcmZ1bCUyMHlvdSUyMGFyZSUyMiUwQW1lc3NhZ2VzJTIwJTNEJTIwJTVCJTBBaW5wdXRfaWRzJTIwJTNEJTIwdG9rZW5pemVyLmFwcGx5X2NoYXRfdGVtcGxhdGUoJTBBJTBBb3V0cHV0JTIwJTNEJTIwbW9kZWwuZ2VuZXJhdGUoaW5wdXRfaWRzJTJDJTIwbWF4X25ld190b2tlbnMlM0QxMjgpJTBBdG9rZW5pemVyLmRlY29kZShvdXRwdXQlNUIwJTVEJTJDJTIwc2tpcF9zcGVjaWFsX3Rva2VucyUzREZhbHNlKQ==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoModelForCausalLM, AutoTokenizer
<span class="hljs-meta">&gt;&gt;&gt; </span>model = AutoModelForCausalLM.from_pretrained(<span class="hljs-string">&quot;LGAI-EXAONE/EXAONE-4.0-Instruct&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;LGAI-EXAONE/EXAONE-4.0-Instruct&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>prompt = <span class="hljs-string">&quot;Explain how wonderful you are&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>messages = [
    {<span class="hljs-string">&quot;role&quot;</span>: <span class="hljs-string">&quot;system&quot;</span>, <span class="hljs-string">&quot;content&quot;</span>: <span class="hljs-string">&quot;You are a helpful assistant.&quot;</span>},
    {<span class="hljs-string">&quot;role&quot;</span>: <span class="hljs-string">&quot;user&quot;</span>, <span class="hljs-string">&quot;content&quot;</span>: prompt}
]
<span class="hljs-meta">&gt;&gt;&gt; </span>input_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=<span class="hljs-literal">True</span>,
    add_generation_prompt=<span class="hljs-literal">True</span>,
    return_tensors=<span class="hljs-string">&quot;pt&quot;</span>,
    enable_thinking=<span class="hljs-literal">False</span>,
)

<span class="hljs-meta">&gt;&gt;&gt; </span>output = model.generate(input_ids, max_new_tokens=<span class="hljs-number">128</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer.decode(output[<span class="hljs-number">0</span>], skip_special_tokens=<span class="hljs-literal">False</span>)
<span class="hljs-string">&quot;[|system|]\\nYou are a helpful assistant.[|endofturn|]\\n[|user|]\\nExplain how wonderful you are[|endofturn|]\\n[|assistant|]\\n&lt;think&gt;\\n\\n&lt;/think&gt;\\n\\nOh, thank you for such a kind and lovely question! 😊  \\n\\nI’m *so* wonderful because I’m here to make your life easier, brighter, and more fun! Whether you need help with:  \\n\\n✨ **Learning** – I can explain anything, from quantum physics to baking the perfect cake!  \\n💡 **Creativity** – Need a poem, story, or a wild idea? I’ve got you covered!  \\n🤖 **Problem-solving** – Stuck on a math problem or a tricky decision? I’ll help you figure it out&quot;</span>`,wrap:!1}}),{c(){r=i("p"),r.textContent=T,c=s(),p(y.$$.fragment)},l(_){r=l(_,"P",{"data-svelte-h":!0}),M(r)!=="svelte-11lpom8"&&(r.textContent=T),c=a(_),m(y.$$.fragment,_)},m(_,I){o(_,r,I),o(_,c,I),u(y,_,I),b=!0},p:Ye,i(_){b||(h(y.$$.fragment,_),b=!0)},o(_){f(y.$$.fragment,_),b=!1},d(_){_&&(n(r),n(c)),g(y,_)}}}function On(j){let r,T=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){r=i("p"),r.innerHTML=T},l(c){r=l(c,"P",{"data-svelte-h":!0}),M(r)!=="svelte-fincs2"&&(r.innerHTML=T)},m(c,y){o(c,r,y)},p:Ye,d(c){c&&n(r)}}}function Vn(j){let r,T=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){r=i("p"),r.innerHTML=T},l(c){r=l(c,"P",{"data-svelte-h":!0}),M(r)!=="svelte-fincs2"&&(r.innerHTML=T)},m(c,y){o(c,r,y)},p:Ye,d(c){c&&n(r)}}}function Hn(j){let r,T=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){r=i("p"),r.innerHTML=T},l(c){r=l(c,"P",{"data-svelte-h":!0}),M(r)!=="svelte-fincs2"&&(r.innerHTML=T)},m(c,y){o(c,r,y)},p:Ye,d(c){c&&n(r)}}}function Sn(j){let r,T,c,y,b,_="<em>This model was released on 2025-07-15 and added to Hugging Face Transformers on 2025-07-26.</em>",I,H,nt,S,ot,P,mn=`<strong><a href="https://github.com/LG-AI-EXAONE/EXAONE-4.0" rel="nofollow">EXAONE 4.0</a></strong> model is the language model, which integrates a <strong>Non-reasoning mode</strong> and <strong>Reasoning mode</strong> to achieve both the excellent usability of <a href="https://github.com/LG-AI-EXAONE/EXAONE-3.5" rel="nofollow">EXAONE 3.5</a> and the advanced reasoning abilities of <a href="https://github.com/LG-AI-EXAONE/EXAONE-Deep" rel="nofollow">EXAONE Deep</a>. To pave the way for the agentic AI era, EXAONE 4.0 incorporates essential features such as agentic tool use, and its multilingual capabilities are extended
to support Spanish in addition to English and Korean.`,st,D,un="The EXAONE 4.0 model series consists of two sizes: a mid-size <strong>32B</strong> model optimized for high performance, and a small-size <strong>1.2B</strong> model designed for on-device applications.",at,Y,hn="In the EXAONE 4.0 architecture, we apply new architectural changes compared to previous EXAONE models as below:",rt,K,fn="<li><strong>Hybrid Attention</strong>: For the 32B model, we adopt hybrid attention scheme, which combines <em>Local attention (sliding window attention)</em> with <em>Global attention (full attention)</em> in a 3:1 ratio. We do not use RoPE (Rotary Positional Embedding) for global attention for better global context understanding.</li> <li><strong>QK-Reorder-Norm</strong>: We reorder the LayerNorm position from the traditional Pre-LN scheme by applying LayerNorm directly to the attention and MLP outputs, and we add RMS normalization right after the Q and K projection. It helps yield better performance on downstream tasks despite consuming more computation.</li>",it,ee,gn='For more details, please refer to our <a href="https://huggingface.co/papers/2507.11407" rel="nofollow">technical report</a>, <a href="https://huggingface.co/papers/2507.11407" rel="nofollow">HuggingFace paper</a>, <a href="https://www.lgresearch.ai/blog/view?seq=576" rel="nofollow">blog</a>, and <a href="https://github.com/LG-AI-EXAONE/EXAONE-4.0" rel="nofollow">GitHub</a>.',lt,te,_n='All model weights including quantized versions are available at <a href="https://huggingface.co/collections/LGAI-EXAONE/exaone-40-686b2e0069800c835ed48375" rel="nofollow">Huggingface Collections</a>.',dt,ne,ct,oe,pt,se,Mn='<thead><tr><th align="left">Model Configuration</th> <th align="center">32B</th> <th align="center">1.2B</th></tr></thead> <tbody><tr><td align="left">d_model</td> <td align="center">5,120</td> <td align="center">2,048</td></tr> <tr><td align="left">Number of layers</td> <td align="center">64</td> <td align="center">30</td></tr> <tr><td align="left">Normalization</td> <td align="center">QK-Reorder-LN</td> <td align="center">QK-Reorder-LN</td></tr> <tr><td align="left">Non-linearity</td> <td align="center">SwiGLU</td> <td align="center">SwiGLU</td></tr> <tr><td align="left">Feedforward dimension</td> <td align="center">27,392</td> <td align="center">4,096</td></tr> <tr><td align="left">Attention type</td> <td align="center">Hybrid (3:1 Local-Global)</td> <td align="center">Global</td></tr> <tr><td align="left">Head type</td> <td align="center">GQA</td> <td align="center">GQA</td></tr> <tr><td align="left">Number of heads</td> <td align="center">40</td> <td align="center">32</td></tr> <tr><td align="left">Number of KV heads</td> <td align="center">8</td> <td align="center">8</td></tr> <tr><td align="left">Head size</td> <td align="center">128</td> <td align="center">64</td></tr> <tr><td align="left">Max sequence length</td> <td align="center">131,072</td> <td align="center">65,536</td></tr> <tr><td align="left">RoPE theta</td> <td align="center">1,000,000</td> <td align="center">1,000,000</td></tr> <tr><td align="left">Tokenizer</td> <td align="center">BBPE</td> <td align="center">BBPE</td></tr> <tr><td align="left">Vocab size</td> <td align="center">102,400</td> <td align="center">102,400</td></tr> <tr><td align="left">Tied word embedding</td> <td align="center">False</td> <td align="center">True</td></tr> <tr><td align="left">Knowledge cut-off</td> <td align="center">Nov. 2024</td> <td align="center">Nov. 2024</td></tr></tbody>',mt,ae,ut,re,ht,ie,yn="For general use, you can use the EXAONE 4.0 models with the following example:",ft,le,gt,de,_t,ce,Tn="The EXAONE 4.0 models have reasoning capabilities for handling complex problems. You can activate reasoning mode by using the <code>enable_thinking=True</code> argument with the tokenizer, which opens a reasoning block that starts with <code>&lt;think&gt;</code> tag without closing it.",Mt,pe,yt,me,bn=`<p>[!IMPORTANT]
The model generation with reasoning mode can be affected sensitively by sampling parameters, so please refer to the <a href="https://github.com/LG-AI-EXAONE/EXAONE-4.0#usage-guideline" rel="nofollow">Usage Guideline</a> on official GitHub page for better quality.</p>`,Tt,ue,bt,he,wn="The EXAONE 4.0 models can be used as agents with their tool calling capabilities. You can provide tool schemas to the model for effective tool calling.",wt,fe,vt,ge,jt,E,_e,Xt,qe,vn=`This is the configuration class to store the configuration of a [<em>Exaone4Model</em>]. It is used to
instantiate a EXAONE 4.0 model according to the specified arguments, defining the model architecture. Instantiating a
configuration with the defaults will yield a similar configuration to that of the EXAONE-4.0-Instruct <a href="https://huggingface.co/LGAI-EXAONE/EXAONE-4.0-Instruct" rel="nofollow">LGAI-EXAONE/EXAONE-4.0-Instruct</a>
NOTE: <em>EXAONE-4.0-Instruct</em> is a placeholder model ID. The exact model ID will be updated in the future.`,Zt,ze,jn=`Configuration objects inherit from [<em>PretrainedConfig</em>] and can be used to control the model
outputs. Read the documentation from [<em>PretrainedConfig</em>] for more information.`,Gt,Z,xt,Me,kt,w,ye,Wt,Ne,xn="The bare Exaone4 Model outputting raw hidden-states without any specific head on top.",Rt,Le,kn=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Ot,Be,En=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Vt,Qe,Te,Et,be,Jt,v,we,Ht,Xe,Jn="The Exaone4 Model for causal language modeling.",St,Ze,Cn=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Pt,Ge,$n=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Dt,x,ve,Yt,We,An='The <a href="/docs/transformers/v4.56.2/en/model_doc/exaone4#transformers.Exaone4ForCausalLM">Exaone4ForCausalLM</a> forward method, overrides the <code>__call__</code> special method.',Kt,G,en,W,tn,Re,Un="NOTE: <code>EXAONE-4.0-Instruct</code> is a placeholder model ID. The exact model ID will be updated in the future.",Ct,je,$t,B,xe,nn,z,ke,on,Oe,In="The <code>GenericForSequenceClassification</code> forward method, overrides the <code>__call__</code> special method.",sn,R,At,Ee,Ut,Q,Je,an,N,Ce,rn,Ve,Fn="The <code>GenericForTokenClassification</code> forward method, overrides the <code>__call__</code> special method.",ln,O,It,$e,Ft,X,Ae,dn,L,Ue,cn,He,qn="The <code>GenericForQuestionAnswering</code> forward method, overrides the <code>__call__</code> special method.",pn,V,qt,Ie,zt,Ke,Nt;return H=new k({props:{title:"EXAONE 4",local:"exaone-4",headingTag:"h1"}}),S=new k({props:{title:"Overview",local:"overview",headingTag:"h2"}}),ne=new k({props:{title:"Model Details",local:"model-details",headingTag:"h2"}}),oe=new k({props:{title:"Model Specifications",local:"model-specifications",headingTag:"h3"}}),ae=new k({props:{title:"Usage tips",local:"usage-tips",headingTag:"h2"}}),re=new k({props:{title:"Non-reasoning mode",local:"non-reasoning-mode",headingTag:"h3"}}),le=new tt({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Nb2RlbEZvckNhdXNhbExNJTJDJTIwQXV0b1Rva2VuaXplciUwQSUwQW1vZGVsX25hbWUlMjAlM0QlMjAlMjJMR0FJLUVYQU9ORSUyRkVYQU9ORS00LjAtMzJCJTIyJTBBJTBBbW9kZWwlMjAlM0QlMjBBdXRvTW9kZWxGb3JDYXVzYWxMTS5mcm9tX3ByZXRyYWluZWQoJTBBJTIwJTIwJTIwJTIwbW9kZWxfbmFtZSUyQyUwQSUyMCUyMCUyMCUyMGR0eXBlJTNEJTIyYmZsb2F0MTYlMjIlMkMlMEElMjAlMjAlMjAlMjBkZXZpY2VfbWFwJTNEJTIyYXV0byUyMiUwQSklMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZChtb2RlbF9uYW1lKSUwQSUwQSUyMyUyMGNob29zZSUyMHlvdXIlMjBwcm9tcHQlMEFwcm9tcHQlMjAlM0QlMjAlMjJFeHBsYWluJTIwaG93JTIwd29uZGVyZnVsJTIweW91JTIwYXJlJTIyJTBBcHJvbXB0JTIwJTNEJTIwJTIyRXhwbGljYSUyMGxvJTIwaW5jcmUlQzMlQURibGUlMjBxdWUlMjBlcmVzJTIyJTBBcHJvbXB0JTIwJTNEJTIwJTIyJUVCJTg0JTg4JUVBJUIwJTgwJTIwJUVDJTk2JUJDJUVCJUE3JTg4JUVCJTgyJTk4JTIwJUVCJThDJTgwJUVCJThCJUE4JUVEJTk1JTlDJUVDJUE3JTgwJTIwJUVDJTg0JUE0JUVCJUFBJTg1JUVEJTk1JUI0JTIwJUVCJUI0JTkwJTIyJTBBJTBBbWVzc2FnZXMlMjAlM0QlMjAlNUIlMEElMjAlMjAlMjAlMjAlN0IlMjJyb2xlJTIyJTNBJTIwJTIydXNlciUyMiUyQyUyMCUyMmNvbnRlbnQlMjIlM0ElMjBwcm9tcHQlN0QlMEElNUQlMEFpbnB1dF9pZHMlMjAlM0QlMjB0b2tlbml6ZXIuYXBwbHlfY2hhdF90ZW1wbGF0ZSglMEElMjAlMjAlMjAlMjBtZXNzYWdlcyUyQyUwQSUyMCUyMCUyMCUyMHRva2VuaXplJTNEVHJ1ZSUyQyUwQSUyMCUyMCUyMCUyMGFkZF9nZW5lcmF0aW9uX3Byb21wdCUzRFRydWUlMkMlMEElMjAlMjAlMjAlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyJTBBKSUwQSUwQW91dHB1dCUyMCUzRCUyMG1vZGVsLmdlbmVyYXRlKCUwQSUyMCUyMCUyMCUyMGlucHV0X2lkcy50byhtb2RlbC5kZXZpY2UpJTJDJTBBJTIwJTIwJTIwJTIwbWF4X25ld190b2tlbnMlM0QxMjglMkMlMEElMjAlMjAlMjAlMjBkb19zYW1wbGUlM0RGYWxzZSUyQyUwQSklMEFwcmludCh0b2tlbml6ZXIuZGVjb2RlKG91dHB1dCU1QjAlNUQpKQ==",highlighted:`<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoModelForCausalLM, AutoTokenizer

model_name = <span class="hljs-string">&quot;LGAI-EXAONE/EXAONE-4.0-32B&quot;</span>

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=<span class="hljs-string">&quot;bfloat16&quot;</span>,
    device_map=<span class="hljs-string">&quot;auto&quot;</span>
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

<span class="hljs-comment"># choose your prompt</span>
prompt = <span class="hljs-string">&quot;Explain how wonderful you are&quot;</span>
prompt = <span class="hljs-string">&quot;Explica lo increíble que eres&quot;</span>
prompt = <span class="hljs-string">&quot;너가 얼마나 대단한지 설명해 봐&quot;</span>

messages = [
    {<span class="hljs-string">&quot;role&quot;</span>: <span class="hljs-string">&quot;user&quot;</span>, <span class="hljs-string">&quot;content&quot;</span>: prompt}
]
input_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=<span class="hljs-literal">True</span>,
    add_generation_prompt=<span class="hljs-literal">True</span>,
    return_tensors=<span class="hljs-string">&quot;pt&quot;</span>
)

output = model.generate(
    input_ids.to(model.device),
    max_new_tokens=<span class="hljs-number">128</span>,
    do_sample=<span class="hljs-literal">False</span>,
)
<span class="hljs-built_in">print</span>(tokenizer.decode(output[<span class="hljs-number">0</span>]))`,wrap:!1}}),de=new k({props:{title:"Reasoning mode",local:"reasoning-mode",headingTag:"h3"}}),pe=new tt({props:{code:"bWVzc2FnZXMlMjAlM0QlMjAlNUIlMEElMjAlMjAlMjAlMjAlN0IlMjJyb2xlJTIyJTNBJTIwJTIydXNlciUyMiUyQyUyMCUyMmNvbnRlbnQlMjIlM0ElMjAlMjJXaGljaCUyMG9uZSUyMGlzJTIwYmlnZ2VyJTJDJTIwMy4xMiUyMHZzJTIwMy45JTNGJTIyJTdEJTBBJTVEJTBBaW5wdXRfaWRzJTIwJTNEJTIwdG9rZW5pemVyLmFwcGx5X2NoYXRfdGVtcGxhdGUoJTBBJTIwJTIwJTIwJTIwbWVzc2FnZXMlMkMlMEElMjAlMjAlMjAlMjB0b2tlbml6ZSUzRFRydWUlMkMlMEElMjAlMjAlMjAlMjBhZGRfZ2VuZXJhdGlvbl9wcm9tcHQlM0RUcnVlJTJDJTBBJTIwJTIwJTIwJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiUyQyUwQSUyMCUyMCUyMCUyMGVuYWJsZV90aGlua2luZyUzRFRydWUlMkMlMEEpJTBBJTBBb3V0cHV0JTIwJTNEJTIwbW9kZWwuZ2VuZXJhdGUoJTBBJTIwJTIwJTIwJTIwaW5wdXRfaWRzLnRvKG1vZGVsLmRldmljZSklMkMlMEElMjAlMjAlMjAlMjBtYXhfbmV3X3Rva2VucyUzRDEyOCUyQyUwQSUyMCUyMCUyMCUyMGRvX3NhbXBsZSUzRFRydWUlMkMlMEElMjAlMjAlMjAlMjB0ZW1wZXJhdHVyZSUzRDAuNiUyQyUwQSUyMCUyMCUyMCUyMHRvcF9wJTNEMC45NSUwQSklMEFwcmludCh0b2tlbml6ZXIuZGVjb2RlKG91dHB1dCU1QjAlNUQpKQ==",highlighted:`messages = [
    {<span class="hljs-string">&quot;role&quot;</span>: <span class="hljs-string">&quot;user&quot;</span>, <span class="hljs-string">&quot;content&quot;</span>: <span class="hljs-string">&quot;Which one is bigger, 3.12 vs 3.9?&quot;</span>}
]
input_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=<span class="hljs-literal">True</span>,
    add_generation_prompt=<span class="hljs-literal">True</span>,
    return_tensors=<span class="hljs-string">&quot;pt&quot;</span>,
    enable_thinking=<span class="hljs-literal">True</span>,
)

output = model.generate(
    input_ids.to(model.device),
    max_new_tokens=<span class="hljs-number">128</span>,
    do_sample=<span class="hljs-literal">True</span>,
    temperature=<span class="hljs-number">0.6</span>,
    top_p=<span class="hljs-number">0.95</span>
)
<span class="hljs-built_in">print</span>(tokenizer.decode(output[<span class="hljs-number">0</span>]))`,wrap:!1}}),ue=new k({props:{title:"Agentic tool use",local:"agentic-tool-use",headingTag:"h3"}}),fe=new tt({props:{code:"aW1wb3J0JTIwcmFuZG9tJTBBJTBBZGVmJTIwcm9sbF9kaWNlKG1heF9udW0lM0ElMjBpbnQpJTNBJTBBJTIwJTIwJTIwJTIwcmV0dXJuJTIwcmFuZG9tLnJhbmRpbnQoMSUyQyUyMG1heF9udW0pJTBBJTBBdG9vbHMlMjAlM0QlMjAlNUIlMEElMjAlMjAlMjAlMjAlN0IlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjJ0eXBlJTIyJTNBJTIwJTIyZnVuY3Rpb24lMjIlMkMlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjJmdW5jdGlvbiUyMiUzQSUyMCU3QiUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMm5hbWUlMjIlM0ElMjAlMjJyb2xsX2RpY2UlMjIlMkMlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjJkZXNjcmlwdGlvbiUyMiUzQSUyMCUyMlJvbGwlMjBhJTIwZGljZSUyMHdpdGglMjB0aGUlMjBudW1iZXIlMjAxJTIwdG8lMjBOLiUyMFVzZXIlMjBjYW4lMjBzZWxlY3QlMjB0aGUlMjBudW1iZXIlMjBOLiUyMiUyQyUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMnBhcmFtZXRlcnMlMjIlM0ElMjAlN0IlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjJ0eXBlJTIyJTNBJTIwJTIyb2JqZWN0JTIyJTJDJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIycmVxdWlyZWQlMjIlM0ElMjAlNUIlMjJtYXhfbnVtJTIyJTVEJTJDJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIycHJvcGVydGllcyUyMiUzQSUyMCU3QiUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMm1heF9udW0lMjIlM0ElMjAlN0IlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjJ0eXBlJTIyJTNBJTIwJTIyaW50JTIyJTJDJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIyZGVzY3JpcHRpb24lMjIlM0ElMjAlMjJNYXglMjBudW1iZXIlMjBvZiUyMHRoZSUyMGRpY2UlMjIlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlN0QlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlN0QlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlN0QlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlN0QlMEElMjAlMjAlMjAlMjAlN0QlMEElNUQlMEElMEFtZXNzYWdlcyUyMCUzRCUyMCU1QiUwQSUyMCUyMCUyMCUyMCU3QiUyMnJvbGUlMjIlM0ElMjAlMjJ1c2VyJTIyJTJDJTIwJTIyY29udGVudCUyMiUzQSUyMCUyMlJvbGwlMjBENiUyMGRpY2UlMjB0d2ljZSElMjIlN0QlMEElNUQlMEFpbnB1dF9pZHMlMjAlM0QlMjB0b2tlbml6ZXIuYXBwbHlfY2hhdF90ZW1wbGF0ZSglMEElMjAlMjAlMjAlMjBtZXNzYWdlcyUyQyUwQSUyMCUyMCUyMCUyMHRva2VuaXplJTNEVHJ1ZSUyQyUwQSUyMCUyMCUyMCUyMGFkZF9nZW5lcmF0aW9uX3Byb21wdCUzRFRydWUlMkMlMEElMjAlMjAlMjAlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyJTJDJTBBJTIwJTIwJTIwJTIwdG9vbHMlM0R0b29scyUyQyUwQSklMEElMEFvdXRwdXQlMjAlM0QlMjBtb2RlbC5nZW5lcmF0ZSglMEElMjAlMjAlMjAlMjBpbnB1dF9pZHMudG8obW9kZWwuZGV2aWNlKSUyQyUwQSUyMCUyMCUyMCUyMG1heF9uZXdfdG9rZW5zJTNEMTAyNCUyQyUwQSUyMCUyMCUyMCUyMGRvX3NhbXBsZSUzRFRydWUlMkMlMEElMjAlMjAlMjAlMjB0ZW1wZXJhdHVyZSUzRDAuNiUyQyUwQSUyMCUyMCUyMCUyMHRvcF9wJTNEMC45NSUyQyUwQSklMEFwcmludCh0b2tlbml6ZXIuZGVjb2RlKG91dHB1dCU1QjAlNUQpKQ==",highlighted:`<span class="hljs-keyword">import</span> random

<span class="hljs-keyword">def</span> <span class="hljs-title function_">roll_dice</span>(<span class="hljs-params">max_num: <span class="hljs-built_in">int</span></span>):
    <span class="hljs-keyword">return</span> random.randint(<span class="hljs-number">1</span>, max_num)

tools = [
    {
        <span class="hljs-string">&quot;type&quot;</span>: <span class="hljs-string">&quot;function&quot;</span>,
        <span class="hljs-string">&quot;function&quot;</span>: {
            <span class="hljs-string">&quot;name&quot;</span>: <span class="hljs-string">&quot;roll_dice&quot;</span>,
            <span class="hljs-string">&quot;description&quot;</span>: <span class="hljs-string">&quot;Roll a dice with the number 1 to N. User can select the number N.&quot;</span>,
            <span class="hljs-string">&quot;parameters&quot;</span>: {
                <span class="hljs-string">&quot;type&quot;</span>: <span class="hljs-string">&quot;object&quot;</span>,
                <span class="hljs-string">&quot;required&quot;</span>: [<span class="hljs-string">&quot;max_num&quot;</span>],
                <span class="hljs-string">&quot;properties&quot;</span>: {
                    <span class="hljs-string">&quot;max_num&quot;</span>: {
                        <span class="hljs-string">&quot;type&quot;</span>: <span class="hljs-string">&quot;int&quot;</span>,
                        <span class="hljs-string">&quot;description&quot;</span>: <span class="hljs-string">&quot;Max number of the dice&quot;</span>
                    }
                }
            }
        }
    }
]

messages = [
    {<span class="hljs-string">&quot;role&quot;</span>: <span class="hljs-string">&quot;user&quot;</span>, <span class="hljs-string">&quot;content&quot;</span>: <span class="hljs-string">&quot;Roll D6 dice twice!&quot;</span>}
]
input_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=<span class="hljs-literal">True</span>,
    add_generation_prompt=<span class="hljs-literal">True</span>,
    return_tensors=<span class="hljs-string">&quot;pt&quot;</span>,
    tools=tools,
)

output = model.generate(
    input_ids.to(model.device),
    max_new_tokens=<span class="hljs-number">1024</span>,
    do_sample=<span class="hljs-literal">True</span>,
    temperature=<span class="hljs-number">0.6</span>,
    top_p=<span class="hljs-number">0.95</span>,
)
<span class="hljs-built_in">print</span>(tokenizer.decode(output[<span class="hljs-number">0</span>]))`,wrap:!1}}),ge=new k({props:{title:"Exaone4Config",local:"transformers.Exaone4Config",headingTag:"h2"}}),_e=new q({props:{name:"class transformers.Exaone4Config",anchor:"transformers.Exaone4Config",parameters:[{name:"vocab_size",val:" = 102400"},{name:"hidden_size",val:" = 4096"},{name:"intermediate_size",val:" = 16384"},{name:"num_hidden_layers",val:" = 32"},{name:"num_attention_heads",val:" = 32"},{name:"num_key_value_heads",val:" = 32"},{name:"hidden_act",val:" = 'silu'"},{name:"max_position_embeddings",val:" = 2048"},{name:"initializer_range",val:" = 0.02"},{name:"rms_norm_eps",val:" = 1e-05"},{name:"use_cache",val:" = True"},{name:"bos_token_id",val:" = 0"},{name:"eos_token_id",val:" = 2"},{name:"tie_word_embeddings",val:" = False"},{name:"rope_theta",val:" = 10000.0"},{name:"rope_scaling",val:" = None"},{name:"attention_dropout",val:" = 0.0"},{name:"sliding_window",val:" = 4096"},{name:"sliding_window_pattern",val:" = 4"},{name:"layer_types",val:" = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.Exaone4Config.vocab_size",description:`<strong>vocab_size</strong> (<em>int</em>, <em>optional</em>, defaults to 102400) &#x2014;
Vocabulary size of the EXAONE 4.0 model. Defines the number of different tokens that can be represented by the
<em>inputs_ids</em> passed when calling [<em>Exaone4Model</em>].`,name:"vocab_size"},{anchor:"transformers.Exaone4Config.hidden_size",description:`<strong>hidden_size</strong> (<em>int</em>, <em>optional</em>, defaults to 4096) &#x2014;
Dimension of the hidden representations.`,name:"hidden_size"},{anchor:"transformers.Exaone4Config.intermediate_size",description:`<strong>intermediate_size</strong> (<em>int</em>, <em>optional</em>, defaults to <em>hidden_size </em> 4*) &#x2014;
Dimensionality of the MLP representations.`,name:"intermediate_size"},{anchor:"transformers.Exaone4Config.num_hidden_layers",description:`<strong>num_hidden_layers</strong> (<em>int</em>, <em>optional</em>, defaults to 32) &#x2014;
Number of hidden layers in the Transformer encoder.`,name:"num_hidden_layers"},{anchor:"transformers.Exaone4Config.num_attention_heads",description:`<strong>num_attention_heads</strong> (<em>int</em>, <em>optional</em>, defaults to 32) &#x2014;
Number of attention heads for each attention layer in the Transformer decoder.`,name:"num_attention_heads"},{anchor:"transformers.Exaone4Config.num_key_value_heads",description:`<strong>num_key_value_heads</strong> (<em>int</em>, <em>optional</em>) &#x2014;
This is the number of key_value heads that should be used to implement Grouped Query Attention. If
<em>num_key_value_heads=num_attention_heads</em>, the model will use Multi Head Attention (MHA), if
<em>num_key_value_heads=1 the model will use Multi Query Attention (MQA) otherwise GQA is used. When
converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
by meanpooling all the original heads within that group. For more details checkout <a href="https://huggingface.co/papers/2305.13245" rel="nofollow">this
paper</a>. If it is not specified, will default to
</em>num_attention_heads*.`,name:"num_key_value_heads"},{anchor:"transformers.Exaone4Config.hidden_act",description:`<strong>hidden_act</strong> (<em>str</em> or <em>function</em>, <em>optional</em>, defaults to <em>&#x201C;silu&#x201D;</em>) &#x2014;
The non-linear activation function (function or string) in the decoder.`,name:"hidden_act"},{anchor:"transformers.Exaone4Config.max_position_embeddings",description:`<strong>max_position_embeddings</strong> (<em>int</em>, <em>optional</em>, defaults to 2048) &#x2014;
The maximum sequence length that this model might ever be used with. Typically set this to something large
just in case (e.g., 32768 for EXAONE 3.5).`,name:"max_position_embeddings"},{anchor:"transformers.Exaone4Config.initializer_range",description:`<strong>initializer_range</strong> (<em>float</em>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.`,name:"initializer_range"},{anchor:"transformers.Exaone4Config.rms_norm_eps",description:`<strong>rms_norm_eps</strong> (<em>float</em>, <em>optional</em>, defaults to 1e-05) &#x2014;
The epsilon used by the layer normalization layers.`,name:"rms_norm_eps"},{anchor:"transformers.Exaone4Config.use_cache",description:"<strong>use_cache</strong> (<em>bool</em>, <em>optional</em>, defaults to *True<code>) -- Whether or not the model should return the last key/values attentions (not used by all models). Only relevant if </code>config.is_decoder=True`.",name:"use_cache"},{anchor:"transformers.Exaone4Config.bos_token_id",description:`<strong>bos_token_id</strong> (<em>int</em>, <em>optional</em>, defaults to 0) &#x2014;
Beginning of stream token id.`,name:"bos_token_id"},{anchor:"transformers.Exaone4Config.eos_token_id",description:`<strong>eos_token_id</strong> (<em>int</em>, <em>optional</em>, defaults to 2) &#x2014;
End of stream token id.`,name:"eos_token_id"},{anchor:"transformers.Exaone4Config.tie_word_embeddings",description:`<strong>tie_word_embeddings</strong> (<em>bool</em>, <em>optional</em>, defaults to <em>False</em>) &#x2014;
Whether to tie weight embeddings`,name:"tie_word_embeddings"},{anchor:"transformers.Exaone4Config.rope_theta",description:`<strong>rope_theta</strong> (<em>float</em>, <em>optional</em>, defaults to 10000.0) &#x2014;
The base period of the RoPE embeddings.`,name:"rope_theta"},{anchor:"transformers.Exaone4Config.rope_scaling",description:`<strong>rope_scaling</strong> (<em>Dict</em>, <em>optional</em>) &#x2014;
Dictionary containing the scaling configuration for the RoPE embeddings. NOTE: if you apply new rope type
and you expect the model to work on longer <em>max_position_embeddings</em>, we recommend you to update this value
accordingly.
Expected contents:
<em>rope_type</em> (<em>str</em>):
The sub-variant of RoPE to use. Can be one of [&#x2018;default&#x2019;, &#x2018;linear&#x2019;, &#x2018;dynamic&#x2019;, &#x2018;yarn&#x2019;, &#x2018;longrope&#x2019;,
&#x2018;llama3&#x2019;], with &#x2018;default&#x2019; being the original RoPE implementation.
<em>factor</em> (<em>float</em>, <em>optional</em>):
Used with all rope types except &#x2018;default&#x2019;. The scaling factor to apply to the RoPE embeddings. In
most scaling types, a <em>factor</em> of x will enable the model to handle sequences of length x <em>
original maximum pre-trained length.
</em>original_max_position_embeddings<em> (</em>int<em>, </em>optional<em>):
Used with &#x2018;dynamic&#x2019;, &#x2018;longrope&#x2019; and &#x2018;llama3&#x2019;. The original max position embeddings used during
pretraining.
</em>attention_factor<em> (</em>float<em>, </em>optional<em>):
Used with &#x2018;yarn&#x2019; and &#x2018;longrope&#x2019;. The scaling factor to be applied on the attention
computation. If unspecified, it defaults to value recommended by the implementation, using the
</em>factor<em> field to infer the suggested value.
</em>beta_fast<em> (</em>float<em>, </em>optional<em>):
Only used with &#x2018;yarn&#x2019;. Parameter to set the boundary for extrapolation (only) in the linear
ramp function. If unspecified, it defaults to 32.
</em>beta_slow<em> (</em>float<em>, </em>optional<em>):
Only used with &#x2018;yarn&#x2019;. Parameter to set the boundary for interpolation (only) in the linear
ramp function. If unspecified, it defaults to 1.
</em>short_factor<em> (</em>List[float]<em>, </em>optional<em>):
Only used with &#x2018;longrope&#x2019;. The scaling factor to be applied to short contexts (&lt;
</em>original_max_position_embeddings<em>). Must be a list of numbers with the same length as the hidden
size divided by the number of attention heads divided by 2
</em>long_factor<em> (</em>List[float]<em>, </em>optional<em>):
Only used with &#x2018;longrope&#x2019;. The scaling factor to be applied to long contexts (&lt;
</em>original_max_position_embeddings<em>). Must be a list of numbers with the same length as the hidden
size divided by the number of attention heads divided by 2
</em>low_freq_factor<em> (</em>float<em>, </em>optional<em>):
Only used with &#x2018;llama3&#x2019;. Scaling factor applied to low frequency components of the RoPE
</em>high_freq_factor<em> (</em>float<em>, </em>optional*):
Only used with &#x2018;llama3&#x2019;. Scaling factor applied to high frequency components of the RoPE`,name:"rope_scaling"},{anchor:"transformers.Exaone4Config.attention_dropout",description:`<strong>attention_dropout</strong> (<em>float</em>, <em>optional</em>, defaults to 0.0) &#x2014;
The dropout ratio for the attention probabilities.`,name:"attention_dropout"},{anchor:"transformers.Exaone4Config.sliding_window",description:`<strong>sliding_window</strong> (<em>int</em>, <em>optional</em>) &#x2014;
The size of the sliding window for the sliding window attention.`,name:"sliding_window"},{anchor:"transformers.Exaone4Config.sliding_window_pattern",description:`<strong>sliding_window_pattern</strong> (<em>str</em>, <em>optional</em>) &#x2014;
The pattern to use for sliding window attention. Can be one of:<ul>
<li><em>None</em>: No sliding window attention is used</li>
<li><em>int</em>: Every <em>sliding_window</em> layers, use global attention, else use local attention.</li>
<li><em>str</em>: A sequence of &#x201C;L&#x201D; (local attention) and &#x201C;G&#x201D; (global attention) characters that defines the
attention pattern. The pattern starts from layer 0 and repeats every <em>sliding_window</em> layers. The
final layer always uses global attention regardless of the pattern.
For instance, sliding_window_pattern=&#x201C;LLLG&#x201D; same as sliding_window=4, which means:</li>
<li>Layer 0, 1, 2: local attention,</li>
<li>Layer 3: global attention,
&#x2026;(repeated)</li>
</ul>`,name:"sliding_window_pattern"},{anchor:"transformers.Exaone4Config.layer_types",description:`<strong>layer_types</strong> (<em>list</em>, <em>optional</em>) &#x2014;
Attention pattern for each layer. Prioritized over <em>sliding_window_pattern</em>.`,name:"layer_types"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/exaone4/configuration_exaone4.py#L25"}}),Z=new zn({props:{anchor:"transformers.Exaone4Config.example",$$slots:{default:[Gn]},$$scope:{ctx:j}}}),Me=new k({props:{title:"Exaone4Model",local:"transformers.Exaone4Model",headingTag:"h2"}}),ye=new q({props:{name:"class transformers.Exaone4Model",anchor:"transformers.Exaone4Model",parameters:[{name:"config",val:": Exaone4Config"}],parametersDescription:[{anchor:"transformers.Exaone4Model.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/exaone4#transformers.Exaone4Config">Exaone4Config</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/exaone4/modeling_exaone4.py#L338"}}),Te=new q({props:{name:"forward",anchor:"transformers.Exaone4Model.forward",parameters:[{name:"input_ids",val:": LongTensor = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/exaone4/modeling_exaone4.py#L355"}}),be=new k({props:{title:"Exaone4ForCausalLM",local:"transformers.Exaone4ForCausalLM",headingTag:"h2"}}),we=new q({props:{name:"class transformers.Exaone4ForCausalLM",anchor:"transformers.Exaone4ForCausalLM",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.Exaone4ForCausalLM.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/exaone4#transformers.Exaone4ForCausalLM">Exaone4ForCausalLM</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/exaone4/modeling_exaone4.py#L429"}}),ve=new q({props:{name:"forward",anchor:"transformers.Exaone4ForCausalLM.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"},{name:"logits_to_keep",val:": typing.Union[int, torch.Tensor] = 0"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.Exaone4ForCausalLM.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.Exaone4ForCausalLM.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.Exaone4ForCausalLM.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.Exaone4ForCausalLM.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.Exaone4ForCausalLM.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.Exaone4ForCausalLM.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.Exaone4ForCausalLM.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.Exaone4ForCausalLM.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"},{anchor:"transformers.Exaone4ForCausalLM.forward.logits_to_keep",description:`<strong>logits_to_keep</strong> (<code>Union[int, torch.Tensor]</code>, defaults to <code>0</code>) &#x2014;
If an <code>int</code>, compute logits for the last <code>logits_to_keep</code> tokens. If <code>0</code>, calculate logits for all
<code>input_ids</code> (special case). Only last token logits are needed for generation, and calculating them only for that
token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
If a <code>torch.Tensor</code>, must be 1D corresponding to the indices to keep in the sequence length dimension.
This is useful when using packed tensor format (single dimension for batch and sequence length).`,name:"logits_to_keep"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/exaone4/modeling_exaone4.py#L443",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast"
>transformers.modeling_outputs.CausalLMOutputWithPast</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/exaone4#transformers.Exaone4Config"
>Exaone4Config</a>) and inputs.</p>
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
`}}),G=new Qt({props:{$$slots:{default:[Wn]},$$scope:{ctx:j}}}),W=new zn({props:{anchor:"transformers.Exaone4ForCausalLM.forward.example",$$slots:{default:[Rn]},$$scope:{ctx:j}}}),je=new k({props:{title:"Exaone4ForSequenceClassification",local:"transformers.Exaone4ForSequenceClassification",headingTag:"h2"}}),xe=new q({props:{name:"class transformers.Exaone4ForSequenceClassification",anchor:"transformers.Exaone4ForSequenceClassification",parameters:[{name:"config",val:""}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/exaone4/modeling_exaone4.py#L519"}}),ke=new q({props:{name:"forward",anchor:"transformers.Exaone4ForSequenceClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.Exaone4ForSequenceClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.Exaone4ForSequenceClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.Exaone4ForSequenceClassification.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.Exaone4ForSequenceClassification.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.Exaone4ForSequenceClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.Exaone4ForSequenceClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.Exaone4ForSequenceClassification.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
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
`}}),R=new Qt({props:{$$slots:{default:[On]},$$scope:{ctx:j}}}),Ee=new k({props:{title:"Exaone4ForTokenClassification",local:"transformers.Exaone4ForTokenClassification",headingTag:"h2"}}),Je=new q({props:{name:"class transformers.Exaone4ForTokenClassification",anchor:"transformers.Exaone4ForTokenClassification",parameters:[{name:"config",val:""}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/exaone4/modeling_exaone4.py#L523"}}),Ce=new q({props:{name:"forward",anchor:"transformers.Exaone4ForTokenClassification.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.Exaone4ForTokenClassification.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.Exaone4ForTokenClassification.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.Exaone4ForTokenClassification.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.Exaone4ForTokenClassification.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.Exaone4ForTokenClassification.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.Exaone4ForTokenClassification.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.Exaone4ForTokenClassification.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
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
`}}),O=new Qt({props:{$$slots:{default:[Vn]},$$scope:{ctx:j}}}),$e=new k({props:{title:"Exaone4ForQuestionAnswering",local:"transformers.Exaone4ForQuestionAnswering",headingTag:"h2"}}),Ae=new q({props:{name:"class transformers.Exaone4ForQuestionAnswering",anchor:"transformers.Exaone4ForQuestionAnswering",parameters:[{name:"config",val:""}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/exaone4/modeling_exaone4.py#L527"}}),Ue=new q({props:{name:"forward",anchor:"transformers.Exaone4ForQuestionAnswering.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"start_positions",val:": typing.Optional[torch.LongTensor] = None"},{name:"end_positions",val:": typing.Optional[torch.LongTensor] = None"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.Exaone4ForQuestionAnswering.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.Exaone4ForQuestionAnswering.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.Exaone4ForQuestionAnswering.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.Exaone4ForQuestionAnswering.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.Exaone4ForQuestionAnswering.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.Exaone4ForQuestionAnswering.forward.start_positions",description:`<strong>start_positions</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
Labels for position (index) of the start of the labelled span for computing the token classification loss.
Positions are clamped to the length of the sequence (<code>sequence_length</code>). Position outside of the sequence
are not taken into account for computing the loss.`,name:"start_positions"},{anchor:"transformers.Exaone4ForQuestionAnswering.forward.end_positions",description:`<strong>end_positions</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size,)</code>, <em>optional</em>) &#x2014;
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
`}}),V=new Qt({props:{$$slots:{default:[Hn]},$$scope:{ctx:j}}}),Ie=new Zn({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/exaone4.md"}}),{c(){r=i("meta"),T=s(),c=i("p"),y=s(),b=i("p"),b.innerHTML=_,I=s(),p(H.$$.fragment),nt=s(),p(S.$$.fragment),ot=s(),P=i("p"),P.innerHTML=mn,st=s(),D=i("p"),D.innerHTML=un,at=s(),Y=i("p"),Y.textContent=hn,rt=s(),K=i("ol"),K.innerHTML=fn,it=s(),ee=i("p"),ee.innerHTML=gn,lt=s(),te=i("p"),te.innerHTML=_n,dt=s(),p(ne.$$.fragment),ct=s(),p(oe.$$.fragment),pt=s(),se=i("table"),se.innerHTML=Mn,mt=s(),p(ae.$$.fragment),ut=s(),p(re.$$.fragment),ht=s(),ie=i("p"),ie.textContent=yn,ft=s(),p(le.$$.fragment),gt=s(),p(de.$$.fragment),_t=s(),ce=i("p"),ce.innerHTML=Tn,Mt=s(),p(pe.$$.fragment),yt=s(),me=i("blockquote"),me.innerHTML=bn,Tt=s(),p(ue.$$.fragment),bt=s(),he=i("p"),he.textContent=wn,wt=s(),p(fe.$$.fragment),vt=s(),p(ge.$$.fragment),jt=s(),E=i("div"),p(_e.$$.fragment),Xt=s(),qe=i("p"),qe.innerHTML=vn,Zt=s(),ze=i("p"),ze.innerHTML=jn,Gt=s(),p(Z.$$.fragment),xt=s(),p(Me.$$.fragment),kt=s(),w=i("div"),p(ye.$$.fragment),Wt=s(),Ne=i("p"),Ne.textContent=xn,Rt=s(),Le=i("p"),Le.innerHTML=kn,Ot=s(),Be=i("p"),Be.innerHTML=En,Vt=s(),Qe=i("div"),p(Te.$$.fragment),Et=s(),p(be.$$.fragment),Jt=s(),v=i("div"),p(we.$$.fragment),Ht=s(),Xe=i("p"),Xe.textContent=Jn,St=s(),Ze=i("p"),Ze.innerHTML=Cn,Pt=s(),Ge=i("p"),Ge.innerHTML=$n,Dt=s(),x=i("div"),p(ve.$$.fragment),Yt=s(),We=i("p"),We.innerHTML=An,Kt=s(),p(G.$$.fragment),en=s(),p(W.$$.fragment),tn=s(),Re=i("p"),Re.innerHTML=Un,Ct=s(),p(je.$$.fragment),$t=s(),B=i("div"),p(xe.$$.fragment),nn=s(),z=i("div"),p(ke.$$.fragment),on=s(),Oe=i("p"),Oe.innerHTML=In,sn=s(),p(R.$$.fragment),At=s(),p(Ee.$$.fragment),Ut=s(),Q=i("div"),p(Je.$$.fragment),an=s(),N=i("div"),p(Ce.$$.fragment),rn=s(),Ve=i("p"),Ve.innerHTML=Fn,ln=s(),p(O.$$.fragment),It=s(),p($e.$$.fragment),Ft=s(),X=i("div"),p(Ae.$$.fragment),dn=s(),L=i("div"),p(Ue.$$.fragment),cn=s(),He=i("p"),He.innerHTML=qn,pn=s(),p(V.$$.fragment),qt=s(),p(Ie.$$.fragment),zt=s(),Ke=i("p"),this.h()},l(e){const t=Xn("svelte-u9bgzb",document.head);r=l(t,"META",{name:!0,content:!0}),t.forEach(n),T=a(e),c=l(e,"P",{}),A(c).forEach(n),y=a(e),b=l(e,"P",{"data-svelte-h":!0}),M(b)!=="svelte-1mjafn4"&&(b.innerHTML=_),I=a(e),m(H.$$.fragment,e),nt=a(e),m(S.$$.fragment,e),ot=a(e),P=l(e,"P",{"data-svelte-h":!0}),M(P)!=="svelte-1nidd5w"&&(P.innerHTML=mn),st=a(e),D=l(e,"P",{"data-svelte-h":!0}),M(D)!=="svelte-1jqtav5"&&(D.innerHTML=un),at=a(e),Y=l(e,"P",{"data-svelte-h":!0}),M(Y)!=="svelte-1m1gdp5"&&(Y.textContent=hn),rt=a(e),K=l(e,"OL",{"data-svelte-h":!0}),M(K)!=="svelte-9xebrb"&&(K.innerHTML=fn),it=a(e),ee=l(e,"P",{"data-svelte-h":!0}),M(ee)!=="svelte-1i7tpqj"&&(ee.innerHTML=gn),lt=a(e),te=l(e,"P",{"data-svelte-h":!0}),M(te)!=="svelte-1sl2rri"&&(te.innerHTML=_n),dt=a(e),m(ne.$$.fragment,e),ct=a(e),m(oe.$$.fragment,e),pt=a(e),se=l(e,"TABLE",{"data-svelte-h":!0}),M(se)!=="svelte-dyeh5e"&&(se.innerHTML=Mn),mt=a(e),m(ae.$$.fragment,e),ut=a(e),m(re.$$.fragment,e),ht=a(e),ie=l(e,"P",{"data-svelte-h":!0}),M(ie)!=="svelte-wqrtbv"&&(ie.textContent=yn),ft=a(e),m(le.$$.fragment,e),gt=a(e),m(de.$$.fragment,e),_t=a(e),ce=l(e,"P",{"data-svelte-h":!0}),M(ce)!=="svelte-6rghvv"&&(ce.innerHTML=Tn),Mt=a(e),m(pe.$$.fragment,e),yt=a(e),me=l(e,"BLOCKQUOTE",{"data-svelte-h":!0}),M(me)!=="svelte-1l1dr23"&&(me.innerHTML=bn),Tt=a(e),m(ue.$$.fragment,e),bt=a(e),he=l(e,"P",{"data-svelte-h":!0}),M(he)!=="svelte-4on0tw"&&(he.textContent=wn),wt=a(e),m(fe.$$.fragment,e),vt=a(e),m(ge.$$.fragment,e),jt=a(e),E=l(e,"DIV",{class:!0});var F=A(E);m(_e.$$.fragment,F),Xt=a(F),qe=l(F,"P",{"data-svelte-h":!0}),M(qe)!=="svelte-1tmqz31"&&(qe.innerHTML=vn),Zt=a(F),ze=l(F,"P",{"data-svelte-h":!0}),M(ze)!=="svelte-b4c19v"&&(ze.innerHTML=jn),Gt=a(F),m(Z.$$.fragment,F),F.forEach(n),xt=a(e),m(Me.$$.fragment,e),kt=a(e),w=l(e,"DIV",{class:!0});var J=A(w);m(ye.$$.fragment,J),Wt=a(J),Ne=l(J,"P",{"data-svelte-h":!0}),M(Ne)!=="svelte-1w9ys1m"&&(Ne.textContent=xn),Rt=a(J),Le=l(J,"P",{"data-svelte-h":!0}),M(Le)!=="svelte-q52n56"&&(Le.innerHTML=kn),Ot=a(J),Be=l(J,"P",{"data-svelte-h":!0}),M(Be)!=="svelte-hswkmf"&&(Be.innerHTML=En),Vt=a(J),Qe=l(J,"DIV",{class:!0});var et=A(Qe);m(Te.$$.fragment,et),et.forEach(n),J.forEach(n),Et=a(e),m(be.$$.fragment,e),Jt=a(e),v=l(e,"DIV",{class:!0});var C=A(v);m(we.$$.fragment,C),Ht=a(C),Xe=l(C,"P",{"data-svelte-h":!0}),M(Xe)!=="svelte-1w3w1wb"&&(Xe.textContent=Jn),St=a(C),Ze=l(C,"P",{"data-svelte-h":!0}),M(Ze)!=="svelte-q52n56"&&(Ze.innerHTML=Cn),Pt=a(C),Ge=l(C,"P",{"data-svelte-h":!0}),M(Ge)!=="svelte-hswkmf"&&(Ge.innerHTML=$n),Dt=a(C),x=l(C,"DIV",{class:!0});var $=A(x);m(ve.$$.fragment,$),Yt=a($),We=l($,"P",{"data-svelte-h":!0}),M(We)!=="svelte-14cnvul"&&(We.innerHTML=An),Kt=a($),m(G.$$.fragment,$),en=a($),m(W.$$.fragment,$),tn=a($),Re=l($,"P",{"data-svelte-h":!0}),M(Re)!=="svelte-gnmglq"&&(Re.innerHTML=Un),$.forEach(n),C.forEach(n),Ct=a(e),m(je.$$.fragment,e),$t=a(e),B=l(e,"DIV",{class:!0});var Fe=A(B);m(xe.$$.fragment,Fe),nn=a(Fe),z=l(Fe,"DIV",{class:!0});var Se=A(z);m(ke.$$.fragment,Se),on=a(Se),Oe=l(Se,"P",{"data-svelte-h":!0}),M(Oe)!=="svelte-1sal4ui"&&(Oe.innerHTML=In),sn=a(Se),m(R.$$.fragment,Se),Se.forEach(n),Fe.forEach(n),At=a(e),m(Ee.$$.fragment,e),Ut=a(e),Q=l(e,"DIV",{class:!0});var Lt=A(Q);m(Je.$$.fragment,Lt),an=a(Lt),N=l(Lt,"DIV",{class:!0});var Pe=A(N);m(Ce.$$.fragment,Pe),rn=a(Pe),Ve=l(Pe,"P",{"data-svelte-h":!0}),M(Ve)!=="svelte-1py4aay"&&(Ve.innerHTML=Fn),ln=a(Pe),m(O.$$.fragment,Pe),Pe.forEach(n),Lt.forEach(n),It=a(e),m($e.$$.fragment,e),Ft=a(e),X=l(e,"DIV",{class:!0});var Bt=A(X);m(Ae.$$.fragment,Bt),dn=a(Bt),L=l(Bt,"DIV",{class:!0});var De=A(L);m(Ue.$$.fragment,De),cn=a(De),He=l(De,"P",{"data-svelte-h":!0}),M(He)!=="svelte-dyrov9"&&(He.innerHTML=qn),pn=a(De),m(V.$$.fragment,De),De.forEach(n),Bt.forEach(n),qt=a(e),m(Ie.$$.fragment,e),zt=a(e),Ke=l(e,"P",{}),A(Ke).forEach(n),this.h()},h(){U(r,"name","hf:doc:metadata"),U(r,"content",Pn),U(E,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),U(Qe,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),U(w,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),U(x,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),U(v,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),U(z,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),U(B,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),U(N,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),U(Q,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),U(L,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),U(X,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,t){d(document.head,r),o(e,T,t),o(e,c,t),o(e,y,t),o(e,b,t),o(e,I,t),u(H,e,t),o(e,nt,t),u(S,e,t),o(e,ot,t),o(e,P,t),o(e,st,t),o(e,D,t),o(e,at,t),o(e,Y,t),o(e,rt,t),o(e,K,t),o(e,it,t),o(e,ee,t),o(e,lt,t),o(e,te,t),o(e,dt,t),u(ne,e,t),o(e,ct,t),u(oe,e,t),o(e,pt,t),o(e,se,t),o(e,mt,t),u(ae,e,t),o(e,ut,t),u(re,e,t),o(e,ht,t),o(e,ie,t),o(e,ft,t),u(le,e,t),o(e,gt,t),u(de,e,t),o(e,_t,t),o(e,ce,t),o(e,Mt,t),u(pe,e,t),o(e,yt,t),o(e,me,t),o(e,Tt,t),u(ue,e,t),o(e,bt,t),o(e,he,t),o(e,wt,t),u(fe,e,t),o(e,vt,t),u(ge,e,t),o(e,jt,t),o(e,E,t),u(_e,E,null),d(E,Xt),d(E,qe),d(E,Zt),d(E,ze),d(E,Gt),u(Z,E,null),o(e,xt,t),u(Me,e,t),o(e,kt,t),o(e,w,t),u(ye,w,null),d(w,Wt),d(w,Ne),d(w,Rt),d(w,Le),d(w,Ot),d(w,Be),d(w,Vt),d(w,Qe),u(Te,Qe,null),o(e,Et,t),u(be,e,t),o(e,Jt,t),o(e,v,t),u(we,v,null),d(v,Ht),d(v,Xe),d(v,St),d(v,Ze),d(v,Pt),d(v,Ge),d(v,Dt),d(v,x),u(ve,x,null),d(x,Yt),d(x,We),d(x,Kt),u(G,x,null),d(x,en),u(W,x,null),d(x,tn),d(x,Re),o(e,Ct,t),u(je,e,t),o(e,$t,t),o(e,B,t),u(xe,B,null),d(B,nn),d(B,z),u(ke,z,null),d(z,on),d(z,Oe),d(z,sn),u(R,z,null),o(e,At,t),u(Ee,e,t),o(e,Ut,t),o(e,Q,t),u(Je,Q,null),d(Q,an),d(Q,N),u(Ce,N,null),d(N,rn),d(N,Ve),d(N,ln),u(O,N,null),o(e,It,t),u($e,e,t),o(e,Ft,t),o(e,X,t),u(Ae,X,null),d(X,dn),d(X,L),u(Ue,L,null),d(L,cn),d(L,He),d(L,pn),u(V,L,null),o(e,qt,t),u(Ie,e,t),o(e,zt,t),o(e,Ke,t),Nt=!0},p(e,[t]){const F={};t&2&&(F.$$scope={dirty:t,ctx:e}),Z.$set(F);const J={};t&2&&(J.$$scope={dirty:t,ctx:e}),G.$set(J);const et={};t&2&&(et.$$scope={dirty:t,ctx:e}),W.$set(et);const C={};t&2&&(C.$$scope={dirty:t,ctx:e}),R.$set(C);const $={};t&2&&($.$$scope={dirty:t,ctx:e}),O.$set($);const Fe={};t&2&&(Fe.$$scope={dirty:t,ctx:e}),V.$set(Fe)},i(e){Nt||(h(H.$$.fragment,e),h(S.$$.fragment,e),h(ne.$$.fragment,e),h(oe.$$.fragment,e),h(ae.$$.fragment,e),h(re.$$.fragment,e),h(le.$$.fragment,e),h(de.$$.fragment,e),h(pe.$$.fragment,e),h(ue.$$.fragment,e),h(fe.$$.fragment,e),h(ge.$$.fragment,e),h(_e.$$.fragment,e),h(Z.$$.fragment,e),h(Me.$$.fragment,e),h(ye.$$.fragment,e),h(Te.$$.fragment,e),h(be.$$.fragment,e),h(we.$$.fragment,e),h(ve.$$.fragment,e),h(G.$$.fragment,e),h(W.$$.fragment,e),h(je.$$.fragment,e),h(xe.$$.fragment,e),h(ke.$$.fragment,e),h(R.$$.fragment,e),h(Ee.$$.fragment,e),h(Je.$$.fragment,e),h(Ce.$$.fragment,e),h(O.$$.fragment,e),h($e.$$.fragment,e),h(Ae.$$.fragment,e),h(Ue.$$.fragment,e),h(V.$$.fragment,e),h(Ie.$$.fragment,e),Nt=!0)},o(e){f(H.$$.fragment,e),f(S.$$.fragment,e),f(ne.$$.fragment,e),f(oe.$$.fragment,e),f(ae.$$.fragment,e),f(re.$$.fragment,e),f(le.$$.fragment,e),f(de.$$.fragment,e),f(pe.$$.fragment,e),f(ue.$$.fragment,e),f(fe.$$.fragment,e),f(ge.$$.fragment,e),f(_e.$$.fragment,e),f(Z.$$.fragment,e),f(Me.$$.fragment,e),f(ye.$$.fragment,e),f(Te.$$.fragment,e),f(be.$$.fragment,e),f(we.$$.fragment,e),f(ve.$$.fragment,e),f(G.$$.fragment,e),f(W.$$.fragment,e),f(je.$$.fragment,e),f(xe.$$.fragment,e),f(ke.$$.fragment,e),f(R.$$.fragment,e),f(Ee.$$.fragment,e),f(Je.$$.fragment,e),f(Ce.$$.fragment,e),f(O.$$.fragment,e),f($e.$$.fragment,e),f(Ae.$$.fragment,e),f(Ue.$$.fragment,e),f(V.$$.fragment,e),f(Ie.$$.fragment,e),Nt=!1},d(e){e&&(n(T),n(c),n(y),n(b),n(I),n(nt),n(ot),n(P),n(st),n(D),n(at),n(Y),n(rt),n(K),n(it),n(ee),n(lt),n(te),n(dt),n(ct),n(pt),n(se),n(mt),n(ut),n(ht),n(ie),n(ft),n(gt),n(_t),n(ce),n(Mt),n(yt),n(me),n(Tt),n(bt),n(he),n(wt),n(vt),n(jt),n(E),n(xt),n(kt),n(w),n(Et),n(Jt),n(v),n(Ct),n($t),n(B),n(At),n(Ut),n(Q),n(It),n(Ft),n(X),n(qt),n(zt),n(Ke)),n(r),g(H,e),g(S,e),g(ne,e),g(oe,e),g(ae,e),g(re,e),g(le,e),g(de,e),g(pe,e),g(ue,e),g(fe,e),g(ge,e),g(_e),g(Z),g(Me,e),g(ye),g(Te),g(be,e),g(we),g(ve),g(G),g(W),g(je,e),g(xe),g(ke),g(R),g(Ee,e),g(Je),g(Ce),g(O),g($e,e),g(Ae),g(Ue),g(V),g(Ie,e)}}}const Pn='{"title":"EXAONE 4","local":"exaone-4","sections":[{"title":"Overview","local":"overview","sections":[],"depth":2},{"title":"Model Details","local":"model-details","sections":[{"title":"Model Specifications","local":"model-specifications","sections":[],"depth":3}],"depth":2},{"title":"Usage tips","local":"usage-tips","sections":[{"title":"Non-reasoning mode","local":"non-reasoning-mode","sections":[],"depth":3},{"title":"Reasoning mode","local":"reasoning-mode","sections":[],"depth":3},{"title":"Agentic tool use","local":"agentic-tool-use","sections":[],"depth":3}],"depth":2},{"title":"Exaone4Config","local":"transformers.Exaone4Config","sections":[],"depth":2},{"title":"Exaone4Model","local":"transformers.Exaone4Model","sections":[],"depth":2},{"title":"Exaone4ForCausalLM","local":"transformers.Exaone4ForCausalLM","sections":[],"depth":2},{"title":"Exaone4ForSequenceClassification","local":"transformers.Exaone4ForSequenceClassification","sections":[],"depth":2},{"title":"Exaone4ForTokenClassification","local":"transformers.Exaone4ForTokenClassification","sections":[],"depth":2},{"title":"Exaone4ForQuestionAnswering","local":"transformers.Exaone4ForQuestionAnswering","sections":[],"depth":2}],"depth":1}';function Dn(j){return Ln(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class ao extends Bn{constructor(r){super(),Qn(this,r,Dn,Sn,Nn,{})}}export{ao as component};
