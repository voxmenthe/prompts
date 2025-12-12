import{s as Yt,o as St,n as L}from"../chunks/scheduler.18a86fab.js";import{S as At,i as Ot,g as u,s as r,r as g,A as Kt,h,f as s,c as a,j as C,x as v,u as _,k as x,l as en,y as p,a as l,v as b,d as y,t as T,w as M}from"../chunks/index.98837b22.js";import{T as Bt}from"../chunks/Tip.77304350.js";import{D as ge}from"../chunks/Docstring.a1ef7999.js";import{C as _e}from"../chunks/CodeBlock.8d0c2e8a.js";import{E as Pt}from"../chunks/ExampleCodeBlock.8c3ee1f9.js";import{H as Re,E as tn}from"../chunks/getInferenceSnippets.06c2775f.js";import{H as nn,a as Gt}from"../chunks/HfOption.6641485e.js";function on(k){let t,m='This model was contributed by <a href="https://huggingface.co/patrickvonplaten" rel="nofollow">patrickvonplaten</a>.',n,c,f="Click on the BertGeneration models in the right sidebar for more examples of how to apply BertGeneration to different sequence generation tasks.";return{c(){t=u("p"),t.innerHTML=m,n=r(),c=u("p"),c.textContent=f},l(i){t=h(i,"P",{"data-svelte-h":!0}),v(t)!=="svelte-vqdfz5"&&(t.innerHTML=m),n=a(i),c=h(i,"P",{"data-svelte-h":!0}),v(c)!=="svelte-7ihmr6"&&(c.textContent=f)},m(i,d){l(i,t,d),l(i,n,d),l(i,c,d)},p:L,d(i){i&&(s(t),s(n),s(c))}}}function sn(k){let t,m;return t=new _e({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwcGlwZWxpbmUlMEElMEFwaXBlbGluZSUyMCUzRCUyMHBpcGVsaW5lKCUwQSUyMCUyMCUyMCUyMHRhc2slM0QlMjJ0ZXh0MnRleHQtZ2VuZXJhdGlvbiUyMiUyQyUwQSUyMCUyMCUyMCUyMG1vZGVsJTNEJTIyZ29vZ2xlJTJGcm9iZXJ0YTJyb2JlcnRhX0wtMjRfZGlzY29mdXNlJTIyJTJDJTBBJTIwJTIwJTIwJTIwZHR5cGUlM0R0b3JjaC5mbG9hdDE2JTJDJTBBJTIwJTIwJTIwJTIwZGV2aWNlJTNEMCUwQSklMEFwaXBlbGluZSglMjJQbGFudHMlMjBjcmVhdGUlMjBlbmVyZ3klMjB0aHJvdWdoJTIwJTIyKQ==",highlighted:`<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> pipeline

pipeline = pipeline(
    task=<span class="hljs-string">&quot;text2text-generation&quot;</span>,
    model=<span class="hljs-string">&quot;google/roberta2roberta_L-24_discofuse&quot;</span>,
    dtype=torch.float16,
    device=<span class="hljs-number">0</span>
)
pipeline(<span class="hljs-string">&quot;Plants create energy through &quot;</span>)`,wrap:!1}}),{c(){g(t.$$.fragment)},l(n){_(t.$$.fragment,n)},m(n,c){b(t,n,c),m=!0},p:L,i(n){m||(y(t.$$.fragment,n),m=!0)},o(n){T(t.$$.fragment,n),m=!1},d(n){M(t,n)}}}function rn(k){let t,m;return t=new _e({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwRW5jb2RlckRlY29kZXJNb2RlbCUyQyUyMEF1dG9Ub2tlbml6ZXIlMEElMEFtb2RlbCUyMCUzRCUyMEVuY29kZXJEZWNvZGVyTW9kZWwuZnJvbV9wcmV0cmFpbmVkKCUyMmdvb2dsZSUyRnJvYmVydGEycm9iZXJ0YV9MLTI0X2Rpc2NvZnVzZSUyMiUyQyUyMGR0eXBlJTNEJTIyYXV0byUyMiklMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZCglMjJnb29nbGUlMkZyb2JlcnRhMnJvYmVydGFfTC0yNF9kaXNjb2Z1c2UlMjIpJTBBJTBBaW5wdXRfaWRzJTIwJTNEJTIwdG9rZW5pemVyKCUwQSUyMCUyMCUyMCUyMCUyMlBsYW50cyUyMGNyZWF0ZSUyMGVuZXJneSUyMHRocm91Z2glMjAlMjIlMkMlMjBhZGRfc3BlY2lhbF90b2tlbnMlM0RGYWxzZSUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIlMEEpLmlucHV0X2lkcyUwQSUwQW91dHB1dHMlMjAlM0QlMjBtb2RlbC5nZW5lcmF0ZShpbnB1dF9pZHMpJTBBcHJpbnQodG9rZW5pemVyLmRlY29kZShvdXRwdXRzJTVCMCU1RCkp",highlighted:`<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> EncoderDecoderModel, AutoTokenizer

model = EncoderDecoderModel.from_pretrained(<span class="hljs-string">&quot;google/roberta2roberta_L-24_discofuse&quot;</span>, dtype=<span class="hljs-string">&quot;auto&quot;</span>)
tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;google/roberta2roberta_L-24_discofuse&quot;</span>)

input_ids = tokenizer(
    <span class="hljs-string">&quot;Plants create energy through &quot;</span>, add_special_tokens=<span class="hljs-literal">False</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>
).input_ids

outputs = model.generate(input_ids)
<span class="hljs-built_in">print</span>(tokenizer.decode(outputs[<span class="hljs-number">0</span>]))`,wrap:!1}}),{c(){g(t.$$.fragment)},l(n){_(t.$$.fragment,n)},m(n,c){b(t,n,c),m=!0},p:L,i(n){m||(y(t.$$.fragment,n),m=!0)},o(n){T(t.$$.fragment,n),m=!1},d(n){M(t,n)}}}function an(k){let t,m;return t=new _e({props:{code:"ZWNobyUyMC1lJTIwJTIyUGxhbnRzJTIwY3JlYXRlJTIwZW5lcmd5JTIwdGhyb3VnaCUyMCUyMiUyMCU3QyUyMHRyYW5zZm9ybWVycyUyMHJ1biUyMC0tdGFzayUyMHRleHQydGV4dC1nZW5lcmF0aW9uJTIwLS1tb2RlbCUyMCUyMmdvb2dsZSUyRnJvYmVydGEycm9iZXJ0YV9MLTI0X2Rpc2NvZnVzZSUyMiUyMC0tZGV2aWNlJTIwMA==",highlighted:'<span class="hljs-built_in">echo</span> -e <span class="hljs-string">&quot;Plants create energy through &quot;</span> | transformers run --task text2text-generation --model <span class="hljs-string">&quot;google/roberta2roberta_L-24_discofuse&quot;</span> --device 0',wrap:!1}}),{c(){g(t.$$.fragment)},l(n){_(t.$$.fragment,n)},m(n,c){b(t,n,c),m=!0},p:L,i(n){m||(y(t.$$.fragment,n),m=!0)},o(n){T(t.$$.fragment,n),m=!1},d(n){M(t,n)}}}function ln(k){let t,m,n,c,f,i;return t=new Gt({props:{id:"usage",option:"Pipeline",$$slots:{default:[sn]},$$scope:{ctx:k}}}),n=new Gt({props:{id:"usage",option:"AutoModel",$$slots:{default:[rn]},$$scope:{ctx:k}}}),f=new Gt({props:{id:"usage",option:"transformers CLI",$$slots:{default:[an]},$$scope:{ctx:k}}}),{c(){g(t.$$.fragment),m=r(),g(n.$$.fragment),c=r(),g(f.$$.fragment)},l(d){_(t.$$.fragment,d),m=a(d),_(n.$$.fragment,d),c=a(d),_(f.$$.fragment,d)},m(d,w){b(t,d,w),l(d,m,w),b(n,d,w),l(d,c,w),b(f,d,w),i=!0},p(d,w){const P={};w&2&&(P.$$scope={dirty:w,ctx:d}),t.$set(P);const W={};w&2&&(W.$$scope={dirty:w,ctx:d}),n.$set(W);const Ie={};w&2&&(Ie.$$scope={dirty:w,ctx:d}),f.$set(Ie)},i(d){i||(y(t.$$.fragment,d),y(n.$$.fragment,d),y(f.$$.fragment,d),i=!0)},o(d){T(t.$$.fragment,d),T(n.$$.fragment,d),T(f.$$.fragment,d),i=!1},d(d){d&&(s(m),s(c)),M(t,d),M(n,d),M(f,d)}}}function dn(k){let t,m="Examples:",n,c,f;return c=new _e({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEJlcnRHZW5lcmF0aW9uQ29uZmlnJTJDJTIwQmVydEdlbmVyYXRpb25FbmNvZGVyJTBBJTBBJTIzJTIwSW5pdGlhbGl6aW5nJTIwYSUyMEJlcnRHZW5lcmF0aW9uJTIwY29uZmlnJTBBY29uZmlndXJhdGlvbiUyMCUzRCUyMEJlcnRHZW5lcmF0aW9uQ29uZmlnKCklMEElMEElMjMlMjBJbml0aWFsaXppbmclMjBhJTIwbW9kZWwlMjAod2l0aCUyMHJhbmRvbSUyMHdlaWdodHMpJTIwZnJvbSUyMHRoZSUyMGNvbmZpZyUwQW1vZGVsJTIwJTNEJTIwQmVydEdlbmVyYXRpb25FbmNvZGVyKGNvbmZpZ3VyYXRpb24pJTBBJTBBJTIzJTIwQWNjZXNzaW5nJTIwdGhlJTIwbW9kZWwlMjBjb25maWd1cmF0aW9uJTBBY29uZmlndXJhdGlvbiUyMCUzRCUyMG1vZGVsLmNvbmZpZw==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> BertGenerationConfig, BertGenerationEncoder

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a BertGeneration config</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = BertGenerationConfig()

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a model (with random weights) from the config</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = BertGenerationEncoder(configuration)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Accessing the model configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = model.config`,wrap:!1}}),{c(){t=u("p"),t.textContent=m,n=r(),g(c.$$.fragment)},l(i){t=h(i,"P",{"data-svelte-h":!0}),v(t)!=="svelte-kvfsh7"&&(t.textContent=m),n=a(i),_(c.$$.fragment,i)},m(i,d){l(i,t,d),l(i,n,d),b(c,i,d),f=!0},p:L,i(i){f||(y(c.$$.fragment,i),f=!0)},o(i){T(c.$$.fragment,i),f=!1},d(i){i&&(s(t),s(n)),M(c,i)}}}function cn(k){let t,m=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=u("p"),t.innerHTML=m},l(n){t=h(n,"P",{"data-svelte-h":!0}),v(t)!=="svelte-fincs2"&&(t.innerHTML=m)},m(n,c){l(n,t,c)},p:L,d(n){n&&s(t)}}}function pn(k){let t,m=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){t=u("p"),t.innerHTML=m},l(n){t=h(n,"P",{"data-svelte-h":!0}),v(t)!=="svelte-fincs2"&&(t.innerHTML=m)},m(n,c){l(n,t,c)},p:L,d(n){n&&s(t)}}}function mn(k){let t,m="Example:",n,c,f;return c=new _e({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBCZXJ0R2VuZXJhdGlvbkRlY29kZXIlMkMlMjBCZXJ0R2VuZXJhdGlvbkNvbmZpZyUwQWltcG9ydCUyMHRvcmNoJTBBJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIyZ29vZ2xlJTJGYmVydF9mb3Jfc2VxX2dlbmVyYXRpb25fTC0yNF9iYmNfZW5jb2RlciUyMiklMEFjb25maWclMjAlM0QlMjBCZXJ0R2VuZXJhdGlvbkNvbmZpZy5mcm9tX3ByZXRyYWluZWQoJTIyZ29vZ2xlJTJGYmVydF9mb3Jfc2VxX2dlbmVyYXRpb25fTC0yNF9iYmNfZW5jb2RlciUyMiklMEFjb25maWcuaXNfZGVjb2RlciUyMCUzRCUyMFRydWUlMEFtb2RlbCUyMCUzRCUyMEJlcnRHZW5lcmF0aW9uRGVjb2Rlci5mcm9tX3ByZXRyYWluZWQoJTBBJTIwJTIwJTIwJTIwJTIyZ29vZ2xlJTJGYmVydF9mb3Jfc2VxX2dlbmVyYXRpb25fTC0yNF9iYmNfZW5jb2RlciUyMiUyQyUyMGNvbmZpZyUzRGNvbmZpZyUwQSklMEElMEFpbnB1dHMlMjAlM0QlMjB0b2tlbml6ZXIoJTIySGVsbG8lMkMlMjBteSUyMGRvZyUyMGlzJTIwY3V0ZSUyMiUyQyUyMHJldHVybl90b2tlbl90eXBlX2lkcyUzREZhbHNlJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiklMEFvdXRwdXRzJTIwJTNEJTIwbW9kZWwoKippbnB1dHMpJTBBJTBBcHJlZGljdGlvbl9sb2dpdHMlMjAlM0QlMjBvdXRwdXRzLmxvZ2l0cw==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, BertGenerationDecoder, BertGenerationConfig
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;google/bert_for_seq_generation_L-24_bbc_encoder&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>config = BertGenerationConfig.from_pretrained(<span class="hljs-string">&quot;google/bert_for_seq_generation_L-24_bbc_encoder&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>config.is_decoder = <span class="hljs-literal">True</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = BertGenerationDecoder.from_pretrained(
<span class="hljs-meta">... </span>    <span class="hljs-string">&quot;google/bert_for_seq_generation_L-24_bbc_encoder&quot;</span>, config=config
<span class="hljs-meta">... </span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = tokenizer(<span class="hljs-string">&quot;Hello, my dog is cute&quot;</span>, return_token_type_ids=<span class="hljs-literal">False</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model(**inputs)

<span class="hljs-meta">&gt;&gt;&gt; </span>prediction_logits = outputs.logits`,wrap:!1}}),{c(){t=u("p"),t.textContent=m,n=r(),g(c.$$.fragment)},l(i){t=h(i,"P",{"data-svelte-h":!0}),v(t)!=="svelte-11lpom8"&&(t.textContent=m),n=a(i),_(c.$$.fragment,i)},m(i,d){l(i,t,d),l(i,n,d),b(c,i,d),f=!0},p:L,i(i){f||(y(c.$$.fragment,i),f=!0)},o(i){T(c.$$.fragment,i),f=!1},d(i){i&&(s(t),s(n)),M(c,i)}}}function un(k){let t,m,n,c,f,i='<div class="flex flex-wrap space-x-1"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/></div>',d,w,P,W,Ie='<a href="https://huggingface.co/papers/1907.12461" rel="nofollow">BertGeneration</a> leverages pretrained BERT checkpoints for sequence-to-sequence tasks with the <a href="/docs/transformers/v4.56.2/en/model_doc/encoder-decoder#transformers.EncoderDecoderModel">EncoderDecoderModel</a> architecture. BertGeneration adapts the <code>BERT</code> for generative tasks.',xe,Y,jt='You can find all the original BERT checkpoints under the <a href="https://huggingface.co/collections/google/bert-release-64ff5e7a4be99045d1896dbc" rel="nofollow">BERT</a> collection.',Xe,X,He,S,Zt='The example below demonstrates how to use BertGeneration with <a href="/docs/transformers/v4.56.2/en/model_doc/encoder-decoder#transformers.EncoderDecoderModel">EncoderDecoderModel</a> for sequence-to-sequence tasks.',Ve,H,Ne,A,zt='Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the <a href="../quantization/overview">Quantization</a> overview for more available quantization backends.',Fe,O,Ut='The example below uses <a href="../quantizationbitsandbytes">BitsAndBytesConfig</a> to quantize the weights to 4-bit.',De,K,Qe,ee,Le,R,te,be,Ct='<a href="/docs/transformers/v4.56.2/en/model_doc/bert-generation#transformers.BertGenerationEncoder">BertGenerationEncoder</a> and <a href="/docs/transformers/v4.56.2/en/model_doc/bert-generation#transformers.BertGenerationDecoder">BertGenerationDecoder</a> should be used in combination with <a href="/docs/transformers/v4.56.2/en/model_doc/encoder-decoder#transformers.EncoderDecoderModel">EncoderDecoderModel</a> for sequence-to-sequence tasks.',rt,ne,at,ye,Wt="<p>For summarization, sentence splitting, sentence fusion and translation, no special tokens are required for the input.</p>",it,Te,Rt="<p>No EOS token should be added to the end of the input for most generation tasks.</p>",Pe,oe,Ye,B,se,lt,Me,It=`This is the configuration class to store the configuration of a <code>BertGenerationPreTrainedModel</code>. It is used to
instantiate a BertGeneration model according to the specified arguments, defining the model architecture.
Instantiating a configuration with the defaults will yield a similar configuration to that of the BertGeneration
<a href="https://huggingface.co/google/bert_for_seq_generation_L-24_bbc_encoder" rel="nofollow">google/bert_for_seq_generation_L-24_bbc_encoder</a>
architecture.`,dt,ve,Et=`Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> and can be used to control the model outputs. Read the
documentation from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> for more information.`,ct,V,Se,re,Ae,G,ae,pt,ke,qt='Construct a BertGeneration tokenizer. Based on <a href="https://github.com/google/sentencepiece" rel="nofollow">SentencePiece</a>.',mt,we,xt=`This tokenizer inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer">PreTrainedTokenizer</a> which contains most of the main methods. Users should refer to
this superclass for more information regarding those methods.`,ut,Je,ie,Oe,le,Ke,J,de,ht,$e,Xt="The bare BertGeneration model transformer outputting raw hidden-states without any specific head on top.",ft,Be,Ht=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,gt,Ge,Vt=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,_t,I,ce,bt,je,Nt='The <a href="/docs/transformers/v4.56.2/en/model_doc/bert-generation#transformers.BertGenerationEncoder">BertGenerationEncoder</a> forward method, overrides the <code>__call__</code> special method.',yt,N,et,pe,tt,$,me,Tt,Ze,Ft="BertGeneration Model with a <code>language modeling</code> head on top for CLM fine-tuning.",Mt,ze,Dt=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,vt,Ue,Qt=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,kt,Z,ue,wt,Ce,Lt='The <a href="/docs/transformers/v4.56.2/en/model_doc/bert-generation#transformers.BertGenerationDecoder">BertGenerationDecoder</a> forward method, overrides the <code>__call__</code> special method.',Jt,F,$t,D,nt,he,ot,Ee,st;return w=new Re({props:{title:"BertGeneration",local:"bertgeneration",headingTag:"h1"}}),X=new Bt({props:{warning:!1,$$slots:{default:[on]},$$scope:{ctx:k}}}),H=new nn({props:{id:"usage",options:["Pipeline","AutoModel","transformers CLI"],$$slots:{default:[ln]},$$scope:{ctx:k}}}),K=new _e({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwRW5jb2RlckRlY29kZXJNb2RlbCUyQyUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBCaXRzQW5kQnl0ZXNDb25maWclMEElMEElMjMlMjBDb25maWd1cmUlMjA0LWJpdCUyMHF1YW50aXphdGlvbiUwQXF1YW50aXphdGlvbl9jb25maWclMjAlM0QlMjBCaXRzQW5kQnl0ZXNDb25maWcoJTBBJTIwJTIwJTIwJTIwbG9hZF9pbl80Yml0JTNEVHJ1ZSUyQyUwQSUyMCUyMCUyMCUyMGJuYl80Yml0X2NvbXB1dGVfZHR5cGUlM0R0b3JjaC5mbG9hdDE2JTBBKSUwQSUwQW1vZGVsJTIwJTNEJTIwRW5jb2RlckRlY29kZXJNb2RlbC5mcm9tX3ByZXRyYWluZWQoJTBBJTIwJTIwJTIwJTIwJTIyZ29vZ2xlJTJGcm9iZXJ0YTJyb2JlcnRhX0wtMjRfZGlzY29mdXNlJTIyJTJDJTBBJTIwJTIwJTIwJTIwcXVhbnRpemF0aW9uX2NvbmZpZyUzRHF1YW50aXphdGlvbl9jb25maWclMkMlMEElMjAlMjAlMjAlMjBkdHlwZSUzRCUyMmF1dG8lMjIlMEEpJTBBdG9rZW5pemVyJTIwJTNEJTIwQXV0b1Rva2VuaXplci5mcm9tX3ByZXRyYWluZWQoJTIyZ29vZ2xlJTJGcm9iZXJ0YTJyb2JlcnRhX0wtMjRfZGlzY29mdXNlJTIyKSUwQSUwQWlucHV0X2lkcyUyMCUzRCUyMHRva2VuaXplciglMEElMjAlMjAlMjAlMjAlMjJQbGFudHMlMjBjcmVhdGUlMjBlbmVyZ3klMjB0aHJvdWdoJTIwJTIyJTJDJTIwYWRkX3NwZWNpYWxfdG9rZW5zJTNERmFsc2UlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyJTBBKS5pbnB1dF9pZHMlMEElMEFvdXRwdXRzJTIwJTNEJTIwbW9kZWwuZ2VuZXJhdGUoaW5wdXRfaWRzKSUwQXByaW50KHRva2VuaXplci5kZWNvZGUob3V0cHV0cyU1QjAlNUQpKQ==",highlighted:`<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> EncoderDecoderModel, AutoTokenizer, BitsAndBytesConfig

<span class="hljs-comment"># Configure 4-bit quantization</span>
quantization_config = BitsAndBytesConfig(
    load_in_4bit=<span class="hljs-literal">True</span>,
    bnb_4bit_compute_dtype=torch.float16
)

model = EncoderDecoderModel.from_pretrained(
    <span class="hljs-string">&quot;google/roberta2roberta_L-24_discofuse&quot;</span>,
    quantization_config=quantization_config,
    dtype=<span class="hljs-string">&quot;auto&quot;</span>
)
tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;google/roberta2roberta_L-24_discofuse&quot;</span>)

input_ids = tokenizer(
    <span class="hljs-string">&quot;Plants create energy through &quot;</span>, add_special_tokens=<span class="hljs-literal">False</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>
).input_ids

outputs = model.generate(input_ids)
<span class="hljs-built_in">print</span>(tokenizer.decode(outputs[<span class="hljs-number">0</span>]))`,wrap:!1}}),ee=new Re({props:{title:"Notes",local:"notes",headingTag:"h2"}}),ne=new _e({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEJlcnRHZW5lcmF0aW9uRW5jb2RlciUyQyUyMEJlcnRHZW5lcmF0aW9uRGVjb2RlciUyQyUyMEJlcnRUb2tlbml6ZXIlMkMlMjBFbmNvZGVyRGVjb2Rlck1vZGVsJTBBJTBBJTIzJTIwbGV2ZXJhZ2UlMjBjaGVja3BvaW50cyUyMGZvciUyMEJlcnQyQmVydCUyMG1vZGVsJTBBJTIzJTIwdXNlJTIwQkVSVCdzJTIwY2xzJTIwdG9rZW4lMjBhcyUyMEJPUyUyMHRva2VuJTIwYW5kJTIwc2VwJTIwdG9rZW4lMjBhcyUyMEVPUyUyMHRva2VuJTBBZW5jb2RlciUyMCUzRCUyMEJlcnRHZW5lcmF0aW9uRW5jb2Rlci5mcm9tX3ByZXRyYWluZWQoJTIyZ29vZ2xlLWJlcnQlMkZiZXJ0LWxhcmdlLXVuY2FzZWQlMjIlMkMlMjBib3NfdG9rZW5faWQlM0QxMDElMkMlMjBlb3NfdG9rZW5faWQlM0QxMDIpJTBBJTIzJTIwYWRkJTIwY3Jvc3MlMjBhdHRlbnRpb24lMjBsYXllcnMlMjBhbmQlMjB1c2UlMjBCRVJUJ3MlMjBjbHMlMjB0b2tlbiUyMGFzJTIwQk9TJTIwdG9rZW4lMjBhbmQlMjBzZXAlMjB0b2tlbiUyMGFzJTIwRU9TJTIwdG9rZW4lMEFkZWNvZGVyJTIwJTNEJTIwQmVydEdlbmVyYXRpb25EZWNvZGVyLmZyb21fcHJldHJhaW5lZCglMEElMjAlMjAlMjAlMjAlMjJnb29nbGUtYmVydCUyRmJlcnQtbGFyZ2UtdW5jYXNlZCUyMiUyQyUyMGFkZF9jcm9zc19hdHRlbnRpb24lM0RUcnVlJTJDJTIwaXNfZGVjb2RlciUzRFRydWUlMkMlMjBib3NfdG9rZW5faWQlM0QxMDElMkMlMjBlb3NfdG9rZW5faWQlM0QxMDIlMEEpJTBBYmVydDJiZXJ0JTIwJTNEJTIwRW5jb2RlckRlY29kZXJNb2RlbChlbmNvZGVyJTNEZW5jb2RlciUyQyUyMGRlY29kZXIlM0RkZWNvZGVyKSUwQSUwQSUyMyUyMGNyZWF0ZSUyMHRva2VuaXplciUwQXRva2VuaXplciUyMCUzRCUyMEJlcnRUb2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMmdvb2dsZS1iZXJ0JTJGYmVydC1sYXJnZS11bmNhc2VkJTIyKSUwQSUwQWlucHV0X2lkcyUyMCUzRCUyMHRva2VuaXplciglMEElMjAlMjAlMjAlMjAlMjJUaGlzJTIwaXMlMjBhJTIwbG9uZyUyMGFydGljbGUlMjB0byUyMHN1bW1hcml6ZSUyMiUyQyUyMGFkZF9zcGVjaWFsX3Rva2VucyUzREZhbHNlJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiUwQSkuaW5wdXRfaWRzJTBBbGFiZWxzJTIwJTNEJTIwdG9rZW5pemVyKCUyMlRoaXMlMjBpcyUyMGElMjBzaG9ydCUyMHN1bW1hcnklMjIlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKS5pbnB1dF9pZHMlMEElMEElMjMlMjB0cmFpbiUwQWxvc3MlMjAlM0QlMjBiZXJ0MmJlcnQoaW5wdXRfaWRzJTNEaW5wdXRfaWRzJTJDJTIwZGVjb2Rlcl9pbnB1dF9pZHMlM0RsYWJlbHMlMkMlMjBsYWJlbHMlM0RsYWJlbHMpLmxvc3MlMEFsb3NzLmJhY2t3YXJkKCk=",highlighted:`<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> BertGenerationEncoder, BertGenerationDecoder, BertTokenizer, EncoderDecoderModel

<span class="hljs-comment"># leverage checkpoints for Bert2Bert model</span>
<span class="hljs-comment"># use BERT&#x27;s cls token as BOS token and sep token as EOS token</span>
encoder = BertGenerationEncoder.from_pretrained(<span class="hljs-string">&quot;google-bert/bert-large-uncased&quot;</span>, bos_token_id=<span class="hljs-number">101</span>, eos_token_id=<span class="hljs-number">102</span>)
<span class="hljs-comment"># add cross attention layers and use BERT&#x27;s cls token as BOS token and sep token as EOS token</span>
decoder = BertGenerationDecoder.from_pretrained(
    <span class="hljs-string">&quot;google-bert/bert-large-uncased&quot;</span>, add_cross_attention=<span class="hljs-literal">True</span>, is_decoder=<span class="hljs-literal">True</span>, bos_token_id=<span class="hljs-number">101</span>, eos_token_id=<span class="hljs-number">102</span>
)
bert2bert = EncoderDecoderModel(encoder=encoder, decoder=decoder)

<span class="hljs-comment"># create tokenizer</span>
tokenizer = BertTokenizer.from_pretrained(<span class="hljs-string">&quot;google-bert/bert-large-uncased&quot;</span>)

input_ids = tokenizer(
    <span class="hljs-string">&quot;This is a long article to summarize&quot;</span>, add_special_tokens=<span class="hljs-literal">False</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>
).input_ids
labels = tokenizer(<span class="hljs-string">&quot;This is a short summary&quot;</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>).input_ids

<span class="hljs-comment"># train</span>
loss = bert2bert(input_ids=input_ids, decoder_input_ids=labels, labels=labels).loss
loss.backward()`,wrap:!1}}),oe=new Re({props:{title:"BertGenerationConfig",local:"transformers.BertGenerationConfig",headingTag:"h2"}}),se=new ge({props:{name:"class transformers.BertGenerationConfig",anchor:"transformers.BertGenerationConfig",parameters:[{name:"vocab_size",val:" = 50358"},{name:"hidden_size",val:" = 1024"},{name:"num_hidden_layers",val:" = 24"},{name:"num_attention_heads",val:" = 16"},{name:"intermediate_size",val:" = 4096"},{name:"hidden_act",val:" = 'gelu'"},{name:"hidden_dropout_prob",val:" = 0.1"},{name:"attention_probs_dropout_prob",val:" = 0.1"},{name:"max_position_embeddings",val:" = 512"},{name:"initializer_range",val:" = 0.02"},{name:"layer_norm_eps",val:" = 1e-12"},{name:"pad_token_id",val:" = 0"},{name:"bos_token_id",val:" = 2"},{name:"eos_token_id",val:" = 1"},{name:"position_embedding_type",val:" = 'absolute'"},{name:"use_cache",val:" = True"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.BertGenerationConfig.vocab_size",description:`<strong>vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 50358) &#x2014;
Vocabulary size of the BERT model. Defines the number of different tokens that can be represented by the
<code>inputs_ids</code> passed when calling <code>BertGeneration</code>.`,name:"vocab_size"},{anchor:"transformers.BertGenerationConfig.hidden_size",description:`<strong>hidden_size</strong> (<code>int</code>, <em>optional</em>, defaults to 1024) &#x2014;
Dimensionality of the encoder layers and the pooler layer.`,name:"hidden_size"},{anchor:"transformers.BertGenerationConfig.num_hidden_layers",description:`<strong>num_hidden_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 24) &#x2014;
Number of hidden layers in the Transformer encoder.`,name:"num_hidden_layers"},{anchor:"transformers.BertGenerationConfig.num_attention_heads",description:`<strong>num_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 16) &#x2014;
Number of attention heads for each attention layer in the Transformer encoder.`,name:"num_attention_heads"},{anchor:"transformers.BertGenerationConfig.intermediate_size",description:`<strong>intermediate_size</strong> (<code>int</code>, <em>optional</em>, defaults to 4096) &#x2014;
Dimensionality of the &#x201C;intermediate&#x201D; (often called feed-forward) layer in the Transformer encoder.`,name:"intermediate_size"},{anchor:"transformers.BertGenerationConfig.hidden_act",description:`<strong>hidden_act</strong> (<code>str</code> or <code>function</code>, <em>optional</em>, defaults to <code>&quot;gelu&quot;</code>) &#x2014;
The non-linear activation function (function or string) in the encoder and pooler. If string, <code>&quot;gelu&quot;</code>,
<code>&quot;relu&quot;</code>, <code>&quot;silu&quot;</code> and <code>&quot;gelu_new&quot;</code> are supported.`,name:"hidden_act"},{anchor:"transformers.BertGenerationConfig.hidden_dropout_prob",description:`<strong>hidden_dropout_prob</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.`,name:"hidden_dropout_prob"},{anchor:"transformers.BertGenerationConfig.attention_probs_dropout_prob",description:`<strong>attention_probs_dropout_prob</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout ratio for the attention probabilities.`,name:"attention_probs_dropout_prob"},{anchor:"transformers.BertGenerationConfig.max_position_embeddings",description:`<strong>max_position_embeddings</strong> (<code>int</code>, <em>optional</em>, defaults to 512) &#x2014;
The maximum sequence length that this model might ever be used with. Typically set this to something large
just in case (e.g., 512 or 1024 or 2048).`,name:"max_position_embeddings"},{anchor:"transformers.BertGenerationConfig.initializer_range",description:`<strong>initializer_range</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.`,name:"initializer_range"},{anchor:"transformers.BertGenerationConfig.layer_norm_eps",description:`<strong>layer_norm_eps</strong> (<code>float</code>, <em>optional</em>, defaults to 1e-12) &#x2014;
The epsilon used by the layer normalization layers.`,name:"layer_norm_eps"},{anchor:"transformers.BertGenerationConfig.pad_token_id",description:`<strong>pad_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 0) &#x2014;
Padding token id.`,name:"pad_token_id"},{anchor:"transformers.BertGenerationConfig.bos_token_id",description:`<strong>bos_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 2) &#x2014;
Beginning of stream token id.`,name:"bos_token_id"},{anchor:"transformers.BertGenerationConfig.eos_token_id",description:`<strong>eos_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 1) &#x2014;
End of stream token id.`,name:"eos_token_id"},{anchor:"transformers.BertGenerationConfig.position_embedding_type",description:`<strong>position_embedding_type</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;absolute&quot;</code>) &#x2014;
Type of position embedding. Choose one of <code>&quot;absolute&quot;</code>, <code>&quot;relative_key&quot;</code>, <code>&quot;relative_key_query&quot;</code>. For
positional embeddings use <code>&quot;absolute&quot;</code>. For more information on <code>&quot;relative_key&quot;</code>, please refer to
<a href="https://huggingface.co/papers/1803.02155" rel="nofollow">Self-Attention with Relative Position Representations (Shaw et al.)</a>.
For more information on <code>&quot;relative_key_query&quot;</code>, please refer to <em>Method 4</em> in <a href="https://huggingface.co/papers/2009.13658" rel="nofollow">Improve Transformer Models
with Better Relative Position Embeddings (Huang et al.)</a>.`,name:"position_embedding_type"},{anchor:"transformers.BertGenerationConfig.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not the model should return the last key/values attentions (not used by all models). Only
relevant if <code>config.is_decoder=True</code>.`,name:"use_cache"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/bert_generation/configuration_bert_generation.py#L20"}}),V=new Pt({props:{anchor:"transformers.BertGenerationConfig.example",$$slots:{default:[dn]},$$scope:{ctx:k}}}),re=new Re({props:{title:"BertGenerationTokenizer",local:"transformers.BertGenerationTokenizer",headingTag:"h2"}}),ae=new ge({props:{name:"class transformers.BertGenerationTokenizer",anchor:"transformers.BertGenerationTokenizer",parameters:[{name:"vocab_file",val:""},{name:"bos_token",val:" = '<s>'"},{name:"eos_token",val:" = '</s>'"},{name:"unk_token",val:" = '<unk>'"},{name:"pad_token",val:" = '<pad>'"},{name:"sep_token",val:" = '<::::>'"},{name:"sp_model_kwargs",val:": typing.Optional[dict[str, typing.Any]] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.BertGenerationTokenizer.vocab_file",description:`<strong>vocab_file</strong> (<code>str</code>) &#x2014;
<a href="https://github.com/google/sentencepiece" rel="nofollow">SentencePiece</a> file (generally has a <em>.spm</em> extension) that
contains the vocabulary necessary to instantiate a tokenizer.`,name:"vocab_file"},{anchor:"transformers.BertGenerationTokenizer.bos_token",description:`<strong>bos_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;s&gt;&quot;</code>) &#x2014;
The begin of sequence token.`,name:"bos_token"},{anchor:"transformers.BertGenerationTokenizer.eos_token",description:`<strong>eos_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;/s&gt;&quot;</code>) &#x2014;
The end of sequence token.`,name:"eos_token"},{anchor:"transformers.BertGenerationTokenizer.unk_token",description:`<strong>unk_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;unk&gt;&quot;</code>) &#x2014;
The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
token instead.`,name:"unk_token"},{anchor:"transformers.BertGenerationTokenizer.pad_token",description:`<strong>pad_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;pad&gt;&quot;</code>) &#x2014;
The token used for padding, for example when batching sequences of different lengths.`,name:"pad_token"},{anchor:"transformers.BertGenerationTokenizer.sep_token",description:`<strong>sep_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt; --:::&gt;&quot;</code>):
The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
sequence classification or for a text and a question for question answering. It is also used as the last
token of a sequence built with special tokens.`,name:"sep_token"},{anchor:"transformers.BertGenerationTokenizer.sp_model_kwargs",description:`<strong>sp_model_kwargs</strong> (<code>dict</code>, <em>optional</em>) &#x2014;
Will be passed to the <code>SentencePieceProcessor.__init__()</code> method. The <a href="https://github.com/google/sentencepiece/tree/master/python" rel="nofollow">Python wrapper for
SentencePiece</a> can be used, among other things,
to set:</p>
<ul>
<li>
<p><code>enable_sampling</code>: Enable subword regularization.</p>
</li>
<li>
<p><code>nbest_size</code>: Sampling parameters for unigram. Invalid for BPE-Dropout.</p>
<ul>
<li><code>nbest_size = {0,1}</code>: No sampling is performed.</li>
<li><code>nbest_size &gt; 1</code>: samples from the nbest_size results.</li>
<li><code>nbest_size &lt; 0</code>: assuming that nbest_size is infinite and samples from the all hypothesis (lattice)
using forward-filtering-and-backward-sampling algorithm.</li>
</ul>
</li>
<li>
<p><code>alpha</code>: Smoothing parameter for unigram sampling, and dropout probability of merge operations for
BPE-dropout.</p>
</li>
</ul>`,name:"sp_model_kwargs"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/bert_generation/tokenization_bert_generation.py#L34"}}),ie=new ge({props:{name:"save_vocabulary",anchor:"transformers.BertGenerationTokenizer.save_vocabulary",parameters:[{name:"save_directory",val:": str"},{name:"filename_prefix",val:": typing.Optional[str] = None"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/bert_generation/tokenization_bert_generation.py#L159"}}),le=new Re({props:{title:"BertGenerationEncoder",local:"transformers.BertGenerationEncoder",headingTag:"h2"}}),de=new ge({props:{name:"class transformers.BertGenerationEncoder",anchor:"transformers.BertGenerationEncoder",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.BertGenerationEncoder.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/bert-generation#transformers.BertGenerationEncoder">BertGenerationEncoder</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/bert_generation/modeling_bert_generation.py#L592"}}),ce=new ge({props:{name:"forward",anchor:"transformers.BertGenerationEncoder.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_hidden_states",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"past_key_values",val:": typing.Optional[tuple[tuple[torch.FloatTensor]]] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.BertGenerationEncoder.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.BertGenerationEncoder.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.BertGenerationEncoder.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.BertGenerationEncoder.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.BertGenerationEncoder.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.BertGenerationEncoder.forward.encoder_hidden_states",description:`<strong>encoder_hidden_states</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
if the model is configured as a decoder.`,name:"encoder_hidden_states"},{anchor:"transformers.BertGenerationEncoder.forward.encoder_attention_mask",description:`<strong>encoder_attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
the cross-attention if the model is configured as a decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>`,name:"encoder_attention_mask"},{anchor:"transformers.BertGenerationEncoder.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>tuple[tuple[torch.FloatTensor]]</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.BertGenerationEncoder.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.BertGenerationEncoder.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.BertGenerationEncoder.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.BertGenerationEncoder.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/bert_generation/modeling_bert_generation.py#L633",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPastAndCrossAttentions"
>transformers.modeling_outputs.BaseModelOutputWithPastAndCrossAttentions</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/bert-generation#transformers.BertGenerationConfig"
>BertGenerationConfig</a>) and inputs.</p>
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
<li>
<p><strong>cross_attentions</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> and <code>config.add_cross_attention=True</code> is passed or when <code>config.output_attentions=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the
weighted average in the cross-attention heads.</p>
</li>
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPastAndCrossAttentions"
>transformers.modeling_outputs.BaseModelOutputWithPastAndCrossAttentions</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),N=new Bt({props:{$$slots:{default:[cn]},$$scope:{ctx:k}}}),pe=new Re({props:{title:"BertGenerationDecoder",local:"transformers.BertGenerationDecoder",headingTag:"h2"}}),me=new ge({props:{name:"class transformers.BertGenerationDecoder",anchor:"transformers.BertGenerationDecoder",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.BertGenerationDecoder.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/bert-generation#transformers.BertGenerationDecoder">BertGenerationDecoder</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/bert_generation/modeling_bert_generation.py#L765"}}),ue=new ge({props:{name:"forward",anchor:"transformers.BertGenerationDecoder.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_hidden_states",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"labels",val:": typing.Optional[torch.Tensor] = None"},{name:"past_key_values",val:": typing.Optional[tuple[tuple[torch.FloatTensor]]] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.BertGenerationDecoder.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.BertGenerationDecoder.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.BertGenerationDecoder.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.BertGenerationDecoder.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.BertGenerationDecoder.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.BertGenerationDecoder.forward.encoder_hidden_states",description:`<strong>encoder_hidden_states</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
if the model is configured as a decoder.`,name:"encoder_hidden_states"},{anchor:"transformers.BertGenerationDecoder.forward.encoder_attention_mask",description:`<strong>encoder_attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
the cross-attention if the model is configured as a decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>`,name:"encoder_attention_mask"},{anchor:"transformers.BertGenerationDecoder.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the left-to-right language modeling loss (next word prediction). Indices should be in
<code>[-100, 0, ..., config.vocab_size]</code> (see <code>input_ids</code> docstring) Tokens with indices set to <code>-100</code> are
ignored (masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>`,name:"labels"},{anchor:"transformers.BertGenerationDecoder.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>tuple[tuple[torch.FloatTensor]]</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.BertGenerationDecoder.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.BertGenerationDecoder.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.BertGenerationDecoder.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.BertGenerationDecoder.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/bert_generation/modeling_bert_generation.py#L787",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithCrossAttentions"
>transformers.modeling_outputs.CausalLMOutputWithCrossAttentions</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/bert-generation#transformers.BertGenerationConfig"
>BertGenerationConfig</a>) and inputs.</p>
<ul>
<li>
<p><strong>loss</strong> (<code>torch.FloatTensor</code> of shape <code>(1,)</code>, <em>optional</em>, returned when <code>labels</code> is provided) — Language modeling loss (for next-token prediction).</p>
</li>
<li>
<p><strong>logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, config.vocab_size)</code>) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).</p>
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
<p><strong>cross_attentions</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Cross attentions weights after the attention softmax, used to compute the weighted average in the
cross-attention heads.</p>
</li>
<li>
<p><strong>past_key_values</strong> (<code>Cache</code>, <em>optional</em>, returned when <code>use_cache=True</code> is passed or when <code>config.use_cache=True</code>) — It is a <a
  href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache"
>Cache</a> instance. For more details, see our <a
  href="https://huggingface.co/docs/transformers/en/kv_cache"
  rel="nofollow"
>kv cache guide</a>.</p>
<p>Contains pre-computed hidden-states (key and values in the attention blocks) that can be used (see
<code>past_key_values</code> input) to speed up sequential decoding.</p>
</li>
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithCrossAttentions"
>transformers.modeling_outputs.CausalLMOutputWithCrossAttentions</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),F=new Bt({props:{$$slots:{default:[pn]},$$scope:{ctx:k}}}),D=new Pt({props:{anchor:"transformers.BertGenerationDecoder.forward.example",$$slots:{default:[mn]},$$scope:{ctx:k}}}),he=new tn({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/bert-generation.md"}}),{c(){t=u("meta"),m=r(),n=u("p"),c=r(),f=u("div"),f.innerHTML=i,d=r(),g(w.$$.fragment),P=r(),W=u("p"),W.innerHTML=Ie,xe=r(),Y=u("p"),Y.innerHTML=jt,Xe=r(),g(X.$$.fragment),He=r(),S=u("p"),S.innerHTML=Zt,Ve=r(),g(H.$$.fragment),Ne=r(),A=u("p"),A.innerHTML=zt,Fe=r(),O=u("p"),O.innerHTML=Ut,De=r(),g(K.$$.fragment),Qe=r(),g(ee.$$.fragment),Le=r(),R=u("ul"),te=u("li"),be=u("p"),be.innerHTML=Ct,rt=r(),g(ne.$$.fragment),at=r(),ye=u("li"),ye.innerHTML=Wt,it=r(),Te=u("li"),Te.innerHTML=Rt,Pe=r(),g(oe.$$.fragment),Ye=r(),B=u("div"),g(se.$$.fragment),lt=r(),Me=u("p"),Me.innerHTML=It,dt=r(),ve=u("p"),ve.innerHTML=Et,ct=r(),g(V.$$.fragment),Se=r(),g(re.$$.fragment),Ae=r(),G=u("div"),g(ae.$$.fragment),pt=r(),ke=u("p"),ke.innerHTML=qt,mt=r(),we=u("p"),we.innerHTML=xt,ut=r(),Je=u("div"),g(ie.$$.fragment),Oe=r(),g(le.$$.fragment),Ke=r(),J=u("div"),g(de.$$.fragment),ht=r(),$e=u("p"),$e.textContent=Xt,ft=r(),Be=u("p"),Be.innerHTML=Ht,gt=r(),Ge=u("p"),Ge.innerHTML=Vt,_t=r(),I=u("div"),g(ce.$$.fragment),bt=r(),je=u("p"),je.innerHTML=Nt,yt=r(),g(N.$$.fragment),et=r(),g(pe.$$.fragment),tt=r(),$=u("div"),g(me.$$.fragment),Tt=r(),Ze=u("p"),Ze.innerHTML=Ft,Mt=r(),ze=u("p"),ze.innerHTML=Dt,vt=r(),Ue=u("p"),Ue.innerHTML=Qt,kt=r(),Z=u("div"),g(ue.$$.fragment),wt=r(),Ce=u("p"),Ce.innerHTML=Lt,Jt=r(),g(F.$$.fragment),$t=r(),g(D.$$.fragment),nt=r(),g(he.$$.fragment),ot=r(),Ee=u("p"),this.h()},l(e){const o=Kt("svelte-u9bgzb",document.head);t=h(o,"META",{name:!0,content:!0}),o.forEach(s),m=a(e),n=h(e,"P",{}),C(n).forEach(s),c=a(e),f=h(e,"DIV",{style:!0,"data-svelte-h":!0}),v(f)!=="svelte-wa5t4p"&&(f.innerHTML=i),d=a(e),_(w.$$.fragment,e),P=a(e),W=h(e,"P",{"data-svelte-h":!0}),v(W)!=="svelte-g9gvlb"&&(W.innerHTML=Ie),xe=a(e),Y=h(e,"P",{"data-svelte-h":!0}),v(Y)!=="svelte-z34ajl"&&(Y.innerHTML=jt),Xe=a(e),_(X.$$.fragment,e),He=a(e),S=h(e,"P",{"data-svelte-h":!0}),v(S)!=="svelte-1igwq5m"&&(S.innerHTML=Zt),Ve=a(e),_(H.$$.fragment,e),Ne=a(e),A=h(e,"P",{"data-svelte-h":!0}),v(A)!=="svelte-nf5ooi"&&(A.innerHTML=zt),Fe=a(e),O=h(e,"P",{"data-svelte-h":!0}),v(O)!=="svelte-72w4f8"&&(O.innerHTML=Ut),De=a(e),_(K.$$.fragment,e),Qe=a(e),_(ee.$$.fragment,e),Le=a(e),R=h(e,"UL",{});var q=C(R);te=h(q,"LI",{});var fe=C(te);be=h(fe,"P",{"data-svelte-h":!0}),v(be)!=="svelte-1objaji"&&(be.innerHTML=Ct),rt=a(fe),_(ne.$$.fragment,fe),fe.forEach(s),at=a(q),ye=h(q,"LI",{"data-svelte-h":!0}),v(ye)!=="svelte-d2m74m"&&(ye.innerHTML=Wt),it=a(q),Te=h(q,"LI",{"data-svelte-h":!0}),v(Te)!=="svelte-1s57c4d"&&(Te.innerHTML=Rt),q.forEach(s),Pe=a(e),_(oe.$$.fragment,e),Ye=a(e),B=h(e,"DIV",{class:!0});var z=C(B);_(se.$$.fragment,z),lt=a(z),Me=h(z,"P",{"data-svelte-h":!0}),v(Me)!=="svelte-1664b7p"&&(Me.innerHTML=It),dt=a(z),ve=h(z,"P",{"data-svelte-h":!0}),v(ve)!=="svelte-1ek1ss9"&&(ve.innerHTML=Et),ct=a(z),_(V.$$.fragment,z),z.forEach(s),Se=a(e),_(re.$$.fragment,e),Ae=a(e),G=h(e,"DIV",{class:!0});var U=C(G);_(ae.$$.fragment,U),pt=a(U),ke=h(U,"P",{"data-svelte-h":!0}),v(ke)!=="svelte-qeg36l"&&(ke.innerHTML=qt),mt=a(U),we=h(U,"P",{"data-svelte-h":!0}),v(we)!=="svelte-ntrhio"&&(we.innerHTML=xt),ut=a(U),Je=h(U,"DIV",{class:!0});var qe=C(Je);_(ie.$$.fragment,qe),qe.forEach(s),U.forEach(s),Oe=a(e),_(le.$$.fragment,e),Ke=a(e),J=h(e,"DIV",{class:!0});var j=C(J);_(de.$$.fragment,j),ht=a(j),$e=h(j,"P",{"data-svelte-h":!0}),v($e)!=="svelte-1g5c0ym"&&($e.textContent=Xt),ft=a(j),Be=h(j,"P",{"data-svelte-h":!0}),v(Be)!=="svelte-q52n56"&&(Be.innerHTML=Ht),gt=a(j),Ge=h(j,"P",{"data-svelte-h":!0}),v(Ge)!=="svelte-hswkmf"&&(Ge.innerHTML=Vt),_t=a(j),I=h(j,"DIV",{class:!0});var We=C(I);_(ce.$$.fragment,We),bt=a(We),je=h(We,"P",{"data-svelte-h":!0}),v(je)!=="svelte-1lcutvj"&&(je.innerHTML=Nt),yt=a(We),_(N.$$.fragment,We),We.forEach(s),j.forEach(s),et=a(e),_(pe.$$.fragment,e),tt=a(e),$=h(e,"DIV",{class:!0});var E=C($);_(me.$$.fragment,E),Tt=a(E),Ze=h(E,"P",{"data-svelte-h":!0}),v(Ze)!=="svelte-1avzda8"&&(Ze.innerHTML=Ft),Mt=a(E),ze=h(E,"P",{"data-svelte-h":!0}),v(ze)!=="svelte-q52n56"&&(ze.innerHTML=Dt),vt=a(E),Ue=h(E,"P",{"data-svelte-h":!0}),v(Ue)!=="svelte-hswkmf"&&(Ue.innerHTML=Qt),kt=a(E),Z=h(E,"DIV",{class:!0});var Q=C(Z);_(ue.$$.fragment,Q),wt=a(Q),Ce=h(Q,"P",{"data-svelte-h":!0}),v(Ce)!=="svelte-1rfc6hv"&&(Ce.innerHTML=Lt),Jt=a(Q),_(F.$$.fragment,Q),$t=a(Q),_(D.$$.fragment,Q),Q.forEach(s),E.forEach(s),nt=a(e),_(he.$$.fragment,e),ot=a(e),Ee=h(e,"P",{}),C(Ee).forEach(s),this.h()},h(){x(t,"name","hf:doc:metadata"),x(t,"content",hn),en(f,"float","right"),x(B,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),x(Je,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),x(G,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),x(I,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),x(J,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),x(Z,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),x($,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,o){p(document.head,t),l(e,m,o),l(e,n,o),l(e,c,o),l(e,f,o),l(e,d,o),b(w,e,o),l(e,P,o),l(e,W,o),l(e,xe,o),l(e,Y,o),l(e,Xe,o),b(X,e,o),l(e,He,o),l(e,S,o),l(e,Ve,o),b(H,e,o),l(e,Ne,o),l(e,A,o),l(e,Fe,o),l(e,O,o),l(e,De,o),b(K,e,o),l(e,Qe,o),b(ee,e,o),l(e,Le,o),l(e,R,o),p(R,te),p(te,be),p(te,rt),b(ne,te,null),p(R,at),p(R,ye),p(R,it),p(R,Te),l(e,Pe,o),b(oe,e,o),l(e,Ye,o),l(e,B,o),b(se,B,null),p(B,lt),p(B,Me),p(B,dt),p(B,ve),p(B,ct),b(V,B,null),l(e,Se,o),b(re,e,o),l(e,Ae,o),l(e,G,o),b(ae,G,null),p(G,pt),p(G,ke),p(G,mt),p(G,we),p(G,ut),p(G,Je),b(ie,Je,null),l(e,Oe,o),b(le,e,o),l(e,Ke,o),l(e,J,o),b(de,J,null),p(J,ht),p(J,$e),p(J,ft),p(J,Be),p(J,gt),p(J,Ge),p(J,_t),p(J,I),b(ce,I,null),p(I,bt),p(I,je),p(I,yt),b(N,I,null),l(e,et,o),b(pe,e,o),l(e,tt,o),l(e,$,o),b(me,$,null),p($,Tt),p($,Ze),p($,Mt),p($,ze),p($,vt),p($,Ue),p($,kt),p($,Z),b(ue,Z,null),p(Z,wt),p(Z,Ce),p(Z,Jt),b(F,Z,null),p(Z,$t),b(D,Z,null),l(e,nt,o),b(he,e,o),l(e,ot,o),l(e,Ee,o),st=!0},p(e,[o]){const q={};o&2&&(q.$$scope={dirty:o,ctx:e}),X.$set(q);const fe={};o&2&&(fe.$$scope={dirty:o,ctx:e}),H.$set(fe);const z={};o&2&&(z.$$scope={dirty:o,ctx:e}),V.$set(z);const U={};o&2&&(U.$$scope={dirty:o,ctx:e}),N.$set(U);const qe={};o&2&&(qe.$$scope={dirty:o,ctx:e}),F.$set(qe);const j={};o&2&&(j.$$scope={dirty:o,ctx:e}),D.$set(j)},i(e){st||(y(w.$$.fragment,e),y(X.$$.fragment,e),y(H.$$.fragment,e),y(K.$$.fragment,e),y(ee.$$.fragment,e),y(ne.$$.fragment,e),y(oe.$$.fragment,e),y(se.$$.fragment,e),y(V.$$.fragment,e),y(re.$$.fragment,e),y(ae.$$.fragment,e),y(ie.$$.fragment,e),y(le.$$.fragment,e),y(de.$$.fragment,e),y(ce.$$.fragment,e),y(N.$$.fragment,e),y(pe.$$.fragment,e),y(me.$$.fragment,e),y(ue.$$.fragment,e),y(F.$$.fragment,e),y(D.$$.fragment,e),y(he.$$.fragment,e),st=!0)},o(e){T(w.$$.fragment,e),T(X.$$.fragment,e),T(H.$$.fragment,e),T(K.$$.fragment,e),T(ee.$$.fragment,e),T(ne.$$.fragment,e),T(oe.$$.fragment,e),T(se.$$.fragment,e),T(V.$$.fragment,e),T(re.$$.fragment,e),T(ae.$$.fragment,e),T(ie.$$.fragment,e),T(le.$$.fragment,e),T(de.$$.fragment,e),T(ce.$$.fragment,e),T(N.$$.fragment,e),T(pe.$$.fragment,e),T(me.$$.fragment,e),T(ue.$$.fragment,e),T(F.$$.fragment,e),T(D.$$.fragment,e),T(he.$$.fragment,e),st=!1},d(e){e&&(s(m),s(n),s(c),s(f),s(d),s(P),s(W),s(xe),s(Y),s(Xe),s(He),s(S),s(Ve),s(Ne),s(A),s(Fe),s(O),s(De),s(Qe),s(Le),s(R),s(Pe),s(Ye),s(B),s(Se),s(Ae),s(G),s(Oe),s(Ke),s(J),s(et),s(tt),s($),s(nt),s(ot),s(Ee)),s(t),M(w,e),M(X,e),M(H,e),M(K,e),M(ee,e),M(ne),M(oe,e),M(se),M(V),M(re,e),M(ae),M(ie),M(le,e),M(de),M(ce),M(N),M(pe,e),M(me),M(ue),M(F),M(D),M(he,e)}}}const hn='{"title":"BertGeneration","local":"bertgeneration","sections":[{"title":"Notes","local":"notes","sections":[],"depth":2},{"title":"BertGenerationConfig","local":"transformers.BertGenerationConfig","sections":[],"depth":2},{"title":"BertGenerationTokenizer","local":"transformers.BertGenerationTokenizer","sections":[],"depth":2},{"title":"BertGenerationEncoder","local":"transformers.BertGenerationEncoder","sections":[],"depth":2},{"title":"BertGenerationDecoder","local":"transformers.BertGenerationDecoder","sections":[],"depth":2}],"depth":1}';function fn(k){return St(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class wn extends At{constructor(t){super(),Ot(this,t,fn,un,Yt,{})}}export{wn as component};
