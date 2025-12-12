import{s as gn,o as _n,n as jo}from"../chunks/scheduler.18a86fab.js";import{S as yn,i as bn,g as s,s as a,r as c,A as vn,h as i,f as o,c as r,j as y,x as g,u as l,k as b,l as Tn,y as n,a as d,v as m,d as p,t as h,w as u}from"../chunks/index.98837b22.js";import{T as fn}from"../chunks/Tip.77304350.js";import{D as w}from"../chunks/Docstring.a1ef7999.js";import{C as At}from"../chunks/CodeBlock.8d0c2e8a.js";import{E as wn}from"../chunks/ExampleCodeBlock.8c3ee1f9.js";import{H as $,E as xn}from"../chunks/getInferenceSnippets.06c2775f.js";function kn(P){let f,C="Example:",_,x,J;return x=new At({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMERpYUNvbmZpZyUyQyUyMERpYU1vZGVsJTBBJTBBJTIzJTIwSW5pdGlhbGl6aW5nJTIwYSUyMERpYUNvbmZpZyUyMHdpdGglMjBkZWZhdWx0JTIwdmFsdWVzJTBBY29uZmlndXJhdGlvbiUyMCUzRCUyMERpYUNvbmZpZygpJTBBJTBBJTIzJTIwSW5pdGlhbGl6aW5nJTIwYSUyMERpYU1vZGVsJTIwKHdpdGglMjByYW5kb20lMjB3ZWlnaHRzKSUyMGZyb20lMjB0aGUlMjBjb25maWd1cmF0aW9uJTBBbW9kZWwlMjAlM0QlMjBEaWFNb2RlbChjb25maWd1cmF0aW9uKSUwQSUwQSUyMyUyMEFjY2Vzc2luZyUyMHRoZSUyMG1vZGVsJTIwY29uZmlndXJhdGlvbiUwQWNvbmZpZ3VyYXRpb24lMjAlM0QlMjBtb2RlbC5jb25maWc=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> DiaConfig, DiaModel

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a DiaConfig with default values</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = DiaConfig()

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a DiaModel (with random weights) from the configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = DiaModel(configuration)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Accessing the model configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = model.config`,wrap:!1}}),{c(){f=s("p"),f.textContent=C,_=a(),c(x.$$.fragment)},l(v){f=i(v,"P",{"data-svelte-h":!0}),g(f)!=="svelte-11lpom8"&&(f.textContent=C),_=r(v),l(x.$$.fragment,v)},m(v,W){d(v,f,W),d(v,_,W),m(x,v,W),J=!0},p:jo,i(v){J||(p(x.$$.fragment,v),J=!0)},o(v){h(x.$$.fragment,v),J=!1},d(v){v&&(o(f),o(_)),u(x,v)}}}function Mn(P){let f,C=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){f=s("p"),f.innerHTML=C},l(_){f=i(_,"P",{"data-svelte-h":!0}),g(f)!=="svelte-fincs2"&&(f.innerHTML=C)},m(_,x){d(_,f,x)},p:jo,d(_){_&&o(f)}}}function Dn(P){let f,C=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){f=s("p"),f.innerHTML=C},l(_){f=i(_,"P",{"data-svelte-h":!0}),g(f)!=="svelte-fincs2"&&(f.innerHTML=C)},m(_,x){d(_,f,x)},p:jo,d(_){_&&o(f)}}}function $n(P){let f,C,_,x,J,v="<em>This model was released on 2025-04-21 and added to Hugging Face Transformers on 2025-06-26.</em>",W,te,gt,B,Go='<div class="flex flex-wrap space-x-1"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/> <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat"/> <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/></div>',_t,oe,yt,ne,No=`<a href="https://github.com/nari-labs/dia" rel="nofollow">Dia</a> is an open-source text-to-speech (TTS) model (1.6B parameters) developed by <a href="https://huggingface.co/nari-labs" rel="nofollow">Nari Labs</a>.
It can generate highly realistic dialogue from transcript including non-verbal communications such as laughter and coughing.
Furthermore, emotion and tone control is also possible via audio conditioning (voice cloning).`,bt,ae,Zo=`<strong>Model Architecture:</strong>
Dia is an encoder-decoder transformer based on the original transformer architecture. However, some more modern features such as
rotational positional embeddings (RoPE) are also included. For its text portion (encoder), a byte tokenizer is utilized while
for the audio portion (decoder), a pretrained codec model <a href="./dac">DAC</a> is used - DAC encodes speech into discrete codebook
tokens and decodes them back into audio.`,vt,re,Tt,se,wt,ie,xt,de,kt,ce,Mt,le,Dt,me,$t,pe,Po=`This model was contributed by <a href="https://huggingface.co/buttercrab" rel="nofollow">Jaeyong Sung</a>, <a href="https://huggingface.co/ArthurZ" rel="nofollow">Arthur Zucker</a>,
and <a href="https://huggingface.co/AntonV" rel="nofollow">Anton Vlasjuk</a>. The original code can be found <a href="https://github.com/nari-labs/dia/" rel="nofollow">here</a>.`,Ct,he,zt,k,ue,Kt,Ze,Wo=`This is the configuration class to store the configuration of a <a href="/docs/transformers/v4.56.2/en/model_doc/dia#transformers.DiaModel">DiaModel</a>. It is used to instantiate a
Dia model according to the specified arguments, defining the model architecture. Instantiating a configuration
with the defaults will yield a similar configuration to that of the
<a href="https://huggingface.co/nari-labs/Dia-1.6B" rel="nofollow">nari-labs/Dia-1.6B</a> architecture.`,eo,Pe,Bo=`Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> and can be used to control the model outputs. Read the
documentation from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> for more information.`,to,L,oo,S,fe,no,We,Lo="Defaulting to audio config as it’s the decoder in this case which is usually the text backbone",Ut,ge,qt,F,_e,ao,Be,So=`This is the configuration class to store the configuration of a <code>DiaDecoder</code>. It is used to instantiate a Dia
decoder according to the specified arguments, defining the decoder architecture.`,ro,Le,Ho=`Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> and can be used to control the model outputs. Read the
documentation from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> for more information.`,Jt,ye,It,E,be,so,Se,Vo=`This is the configuration class to store the configuration of a <code>DiaEncoder</code>. It is used to instantiate a Dia
encoder according to the specified arguments, defining the encoder architecture.`,io,He,Xo=`Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> and can be used to control the model outputs. Read the
documentation from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> for more information.`,Ft,ve,Et,z,Te,co,Ve,Qo="Construct a Dia tokenizer. Dia simply uses raw bytes utf-8 encoding except for special tokens <code>[S1]</code> and <code>[S2]</code>.",lo,Xe,Oo=`This tokenizer inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast">PreTrainedTokenizerFast</a> which contains most of the main methods. Users should
refer to this superclass for more information regarding those methods.`,mo,H,we,po,Qe,Yo=`Main method to tokenize and prepare for the model one or several sequence(s) or one or several pair(s) of
sequences.`,Rt,xe,jt,U,ke,ho,Oe,Ao="Constructs an Dia feature extractor.",uo,Ye,Ko=`This feature extractor inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/feature_extractor#transformers.SequenceFeatureExtractor">SequenceFeatureExtractor</a> which contains
most of the main methods. Users should refer to this superclass for more information regarding those methods.`,fo,V,Me,go,Ae,en="Main method to featurize and prepare for the model one or several sequence(s).",Gt,De,Nt,M,$e,_o,Ke,tn=`Constructs a Dia processor which wraps a <a href="/docs/transformers/v4.56.2/en/model_doc/dia#transformers.DiaFeatureExtractor">DiaFeatureExtractor</a>, <a href="/docs/transformers/v4.56.2/en/model_doc/dia#transformers.DiaTokenizer">DiaTokenizer</a>, and a <a href="/docs/transformers/v4.56.2/en/model_doc/dac#transformers.DacModel">DacModel</a> into
a single processor. It inherits, the audio feature extraction, tokenizer, and audio encode/decode functio-
nalities. See <a href="/docs/transformers/v4.56.2/en/model_doc/dia#transformers.DiaProcessor.__call__"><strong>call</strong>()</a>, <code>~DiaProcessor.encode</code>, and <a href="/docs/transformers/v4.56.2/en/model_doc/dia#transformers.DiaProcessor.decode">decode()</a> for more
information.`,yo,X,Ce,bo,et,on=`Main method to prepare text(s) and audio to be fed as input to the model. The <code>audio</code> argument is
forwarded to the DiaFeatureExtractor’s <a href="/docs/transformers/v4.56.2/en/model_doc/dia#transformers.DiaFeatureExtractor.__call__"><strong>call</strong>()</a> and subsequently to the
DacModel’s <a href="/docs/transformers/v4.56.2/en/model_doc/dac#transformers.DacModel.encode">encode()</a>. The <code>text</code> argument to <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__"><strong>call</strong>()</a>. Please refer
to the docstring of the above methods for more information.`,vo,Q,ze,To,tt,nn=`Decodes a batch of audio codebook sequences into their respective audio waveforms via the
<code>audio_tokenizer</code>. See <a href="/docs/transformers/v4.56.2/en/model_doc/dac#transformers.DacModel.decode">decode()</a> for more information.`,wo,O,Ue,xo,ot,an=`Decodes a single sequence of audio codebooks into the respective audio waveform via the
<code>audio_tokenizer</code>. See <a href="/docs/transformers/v4.56.2/en/model_doc/dac#transformers.DacModel.decode">decode()</a> and <a href="/docs/transformers/v4.56.2/en/model_doc/dia#transformers.DiaProcessor.batch_decode">batch_decode()</a> for more information.`,Zt,qe,Pt,D,Je,ko,nt,rn="The bare Dia model outputting raw hidden-states without any specific head on top.",Mo,at,sn=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Do,rt,dn=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,$o,R,Ie,Co,st,cn='The <a href="/docs/transformers/v4.56.2/en/model_doc/dia#transformers.DiaModel">DiaModel</a> forward method, overrides the <code>__call__</code> special method.',zo,Y,Wt,Fe,Bt,T,Ee,Uo,it,ln="The Dia model consisting of a (byte) text encoder and audio decoder with a prediction head on top.",qo,dt,mn=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Jo,ct,pn=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Io,j,Re,Fo,lt,hn='The <a href="/docs/transformers/v4.56.2/en/model_doc/dia#transformers.DiaForConditionalGeneration">DiaForConditionalGeneration</a> forward method, overrides the <code>__call__</code> special method.',Eo,A,Ro,mt,je,Lt,Ge,St,ft,Ht;return te=new $({props:{title:"Dia",local:"dia",headingTag:"h1"}}),oe=new $({props:{title:"Overview",local:"overview",headingTag:"h2"}}),re=new $({props:{title:"Usage Tips",local:"usage-tips",headingTag:"h2"}}),se=new $({props:{title:"Generation with Text",local:"generation-with-text",headingTag:"h3"}}),ie=new At({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Qcm9jZXNzb3IlMkMlMjBEaWFGb3JDb25kaXRpb25hbEdlbmVyYXRpb24lMkMlMjBpbmZlcl9kZXZpY2UlMEElMEF0b3JjaF9kZXZpY2UlMjAlM0QlMjBpbmZlcl9kZXZpY2UoKSUwQW1vZGVsX2NoZWNrcG9pbnQlMjAlM0QlMjAlMjJuYXJpLWxhYnMlMkZEaWEtMS42Qi0wNjI2JTIyJTBBJTBBdGV4dCUyMCUzRCUyMCU1QiUyMiU1QlMxJTVEJTIwRGlhJTIwaXMlMjBhbiUyMG9wZW4lMjB3ZWlnaHRzJTIwdGV4dCUyMHRvJTIwZGlhbG9ndWUlMjBtb2RlbC4lMjIlNUQlMEFwcm9jZXNzb3IlMjAlM0QlMjBBdXRvUHJvY2Vzc29yLmZyb21fcHJldHJhaW5lZChtb2RlbF9jaGVja3BvaW50KSUwQWlucHV0cyUyMCUzRCUyMHByb2Nlc3Nvcih0ZXh0JTNEdGV4dCUyQyUyMHBhZGRpbmclM0RUcnVlJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMikudG8odG9yY2hfZGV2aWNlKSUwQSUwQW1vZGVsJTIwJTNEJTIwRGlhRm9yQ29uZGl0aW9uYWxHZW5lcmF0aW9uLmZyb21fcHJldHJhaW5lZChtb2RlbF9jaGVja3BvaW50KS50byh0b3JjaF9kZXZpY2UpJTBBb3V0cHV0cyUyMCUzRCUyMG1vZGVsLmdlbmVyYXRlKCoqaW5wdXRzJTJDJTIwbWF4X25ld190b2tlbnMlM0QyNTYpJTIwJTIwJTIzJTIwY29ycmVzcG9uZHMlMjB0byUyMGFyb3VuZCUyMH4ycyUwQSUwQSUyMyUyMHNhdmUlMjBhdWRpbyUyMHRvJTIwYSUyMGZpbGUlMEFvdXRwdXRzJTIwJTNEJTIwcHJvY2Vzc29yLmJhdGNoX2RlY29kZShvdXRwdXRzKSUwQXByb2Nlc3Nvci5zYXZlX2F1ZGlvKG91dHB1dHMlMkMlMjAlMjJleGFtcGxlLndhdiUyMiklMEE=",highlighted:`<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoProcessor, DiaForConditionalGeneration, infer_device

torch_device = infer_device()
model_checkpoint = <span class="hljs-string">&quot;nari-labs/Dia-1.6B-0626&quot;</span>

text = [<span class="hljs-string">&quot;[S1] Dia is an open weights text to dialogue model.&quot;</span>]
processor = AutoProcessor.from_pretrained(model_checkpoint)
inputs = processor(text=text, padding=<span class="hljs-literal">True</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>).to(torch_device)

model = DiaForConditionalGeneration.from_pretrained(model_checkpoint).to(torch_device)
outputs = model.generate(**inputs, max_new_tokens=<span class="hljs-number">256</span>)  <span class="hljs-comment"># corresponds to around ~2s</span>

<span class="hljs-comment"># save audio to a file</span>
outputs = processor.batch_decode(outputs)
processor.save_audio(outputs, <span class="hljs-string">&quot;example.wav&quot;</span>)
`,wrap:!1}}),de=new $({props:{title:"Generation with Text and Audio (Voice Cloning)",local:"generation-with-text-and-audio-voice-cloning",headingTag:"h3"}}),ce=new At({props:{code:"ZnJvbSUyMGRhdGFzZXRzJTIwaW1wb3J0JTIwbG9hZF9kYXRhc2V0JTJDJTIwQXVkaW8lMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Byb2Nlc3NvciUyQyUyMERpYUZvckNvbmRpdGlvbmFsR2VuZXJhdGlvbiUyQyUyMGluZmVyX2RldmljZSUwQSUwQXRvcmNoX2RldmljZSUyMCUzRCUyMGluZmVyX2RldmljZSgpJTBBbW9kZWxfY2hlY2twb2ludCUyMCUzRCUyMCUyMm5hcmktbGFicyUyRkRpYS0xLjZCLTA2MjYlMjIlMEElMEFkcyUyMCUzRCUyMGxvYWRfZGF0YXNldCglMjJoZi1pbnRlcm5hbC10ZXN0aW5nJTJGZGFpbHl0YWxrLWR1bW15JTIyJTJDJTIwc3BsaXQlM0QlMjJ0cmFpbiUyMiklMEFkcyUyMCUzRCUyMGRzLmNhc3RfY29sdW1uKCUyMmF1ZGlvJTIyJTJDJTIwQXVkaW8oc2FtcGxpbmdfcmF0ZSUzRDQ0MTAwKSklMEFhdWRpbyUyMCUzRCUyMGRzJTVCLTElNUQlNUIlMjJhdWRpbyUyMiU1RCU1QiUyMmFycmF5JTIyJTVEJTBBJTIzJTIwdGV4dCUyMGlzJTIwYSUyMHRyYW5zY3JpcHQlMjBvZiUyMHRoZSUyMGF1ZGlvJTIwJTJCJTIwYWRkaXRpb25hbCUyMHRleHQlMjB5b3UlMjB3YW50JTIwYXMlMjBuZXclMjBhdWRpbyUwQXRleHQlMjAlM0QlMjAlNUIlMjIlNUJTMSU1RCUyMEklMjBrbm93LiUyMEl0J3MlMjBnb2luZyUyMHRvJTIwc2F2ZSUyMG1lJTIwYSUyMGxvdCUyMG9mJTIwbW9uZXklMkMlMjBJJTIwaG9wZS4lMjAlNUJTMiU1RCUyMEklMjBzdXJlJTIwaG9wZSUyMHNvJTIwZm9yJTIweW91LiUyMiU1RCUwQSUwQXByb2Nlc3NvciUyMCUzRCUyMEF1dG9Qcm9jZXNzb3IuZnJvbV9wcmV0cmFpbmVkKG1vZGVsX2NoZWNrcG9pbnQpJTBBaW5wdXRzJTIwJTNEJTIwcHJvY2Vzc29yKHRleHQlM0R0ZXh0JTJDJTIwYXVkaW8lM0RhdWRpbyUyQyUyMHBhZGRpbmclM0RUcnVlJTJDJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMikudG8odG9yY2hfZGV2aWNlKSUwQXByb21wdF9sZW4lMjAlM0QlMjBwcm9jZXNzb3IuZ2V0X2F1ZGlvX3Byb21wdF9sZW4oaW5wdXRzJTVCJTIyZGVjb2Rlcl9hdHRlbnRpb25fbWFzayUyMiU1RCklMEElMEFtb2RlbCUyMCUzRCUyMERpYUZvckNvbmRpdGlvbmFsR2VuZXJhdGlvbi5mcm9tX3ByZXRyYWluZWQobW9kZWxfY2hlY2twb2ludCkudG8odG9yY2hfZGV2aWNlKSUwQW91dHB1dHMlMjAlM0QlMjBtb2RlbC5nZW5lcmF0ZSgqKmlucHV0cyUyQyUyMG1heF9uZXdfdG9rZW5zJTNEMjU2KSUyMCUyMCUyMyUyMGNvcnJlc3BvbmRzJTIwdG8lMjBhcm91bmQlMjB+MnMlMEElMEElMjMlMjByZXRyaWV2ZSUyMGFjdHVhbGx5JTIwZ2VuZXJhdGVkJTIwYXVkaW8lMjBhbmQlMjBzYXZlJTIwdG8lMjBhJTIwZmlsZSUwQW91dHB1dHMlMjAlM0QlMjBwcm9jZXNzb3IuYmF0Y2hfZGVjb2RlKG91dHB1dHMlMkMlMjBhdWRpb19wcm9tcHRfbGVuJTNEcHJvbXB0X2xlbiklMEFwcm9jZXNzb3Iuc2F2ZV9hdWRpbyhvdXRwdXRzJTJDJTIwJTIyZXhhbXBsZV93aXRoX2F1ZGlvLndhdiUyMik=",highlighted:`<span class="hljs-keyword">from</span> datasets <span class="hljs-keyword">import</span> load_dataset, Audio
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoProcessor, DiaForConditionalGeneration, infer_device

torch_device = infer_device()
model_checkpoint = <span class="hljs-string">&quot;nari-labs/Dia-1.6B-0626&quot;</span>

ds = load_dataset(<span class="hljs-string">&quot;hf-internal-testing/dailytalk-dummy&quot;</span>, split=<span class="hljs-string">&quot;train&quot;</span>)
ds = ds.cast_column(<span class="hljs-string">&quot;audio&quot;</span>, Audio(sampling_rate=<span class="hljs-number">44100</span>))
audio = ds[-<span class="hljs-number">1</span>][<span class="hljs-string">&quot;audio&quot;</span>][<span class="hljs-string">&quot;array&quot;</span>]
<span class="hljs-comment"># text is a transcript of the audio + additional text you want as new audio</span>
text = [<span class="hljs-string">&quot;[S1] I know. It&#x27;s going to save me a lot of money, I hope. [S2] I sure hope so for you.&quot;</span>]

processor = AutoProcessor.from_pretrained(model_checkpoint)
inputs = processor(text=text, audio=audio, padding=<span class="hljs-literal">True</span>, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>).to(torch_device)
prompt_len = processor.get_audio_prompt_len(inputs[<span class="hljs-string">&quot;decoder_attention_mask&quot;</span>])

model = DiaForConditionalGeneration.from_pretrained(model_checkpoint).to(torch_device)
outputs = model.generate(**inputs, max_new_tokens=<span class="hljs-number">256</span>)  <span class="hljs-comment"># corresponds to around ~2s</span>

<span class="hljs-comment"># retrieve actually generated audio and save to a file</span>
outputs = processor.batch_decode(outputs, audio_prompt_len=prompt_len)
processor.save_audio(outputs, <span class="hljs-string">&quot;example_with_audio.wav&quot;</span>)`,wrap:!1}}),le=new $({props:{title:"Training",local:"training",headingTag:"h3"}}),me=new At({props:{code:"ZnJvbSUyMGRhdGFzZXRzJTIwaW1wb3J0JTIwbG9hZF9kYXRhc2V0JTJDJTIwQXVkaW8lMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b1Byb2Nlc3NvciUyQyUyMERpYUZvckNvbmRpdGlvbmFsR2VuZXJhdGlvbiUyQyUyMGluZmVyX2RldmljZSUwQSUwQXRvcmNoX2RldmljZSUyMCUzRCUyMGluZmVyX2RldmljZSgpJTBBbW9kZWxfY2hlY2twb2ludCUyMCUzRCUyMCUyMm5hcmktbGFicyUyRkRpYS0xLjZCLTA2MjYlMjIlMEElMEFkcyUyMCUzRCUyMGxvYWRfZGF0YXNldCglMjJoZi1pbnRlcm5hbC10ZXN0aW5nJTJGZGFpbHl0YWxrLWR1bW15JTIyJTJDJTIwc3BsaXQlM0QlMjJ0cmFpbiUyMiklMEFkcyUyMCUzRCUyMGRzLmNhc3RfY29sdW1uKCUyMmF1ZGlvJTIyJTJDJTIwQXVkaW8oc2FtcGxpbmdfcmF0ZSUzRDQ0MTAwKSklMEFhdWRpbyUyMCUzRCUyMGRzJTVCLTElNUQlNUIlMjJhdWRpbyUyMiU1RCU1QiUyMmFycmF5JTIyJTVEJTBBJTIzJTIwdGV4dCUyMGlzJTIwYSUyMHRyYW5zY3JpcHQlMjBvZiUyMHRoZSUyMGF1ZGlvJTBBdGV4dCUyMCUzRCUyMCU1QiUyMiU1QlMxJTVEJTIwSSUyMGtub3cuJTIwSXQncyUyMGdvaW5nJTIwdG8lMjBzYXZlJTIwbWUlMjBhJTIwbG90JTIwb2YlMjBtb25leSUyQyUyMEklMjBob3BlLiUyMiU1RCUwQSUwQXByb2Nlc3NvciUyMCUzRCUyMEF1dG9Qcm9jZXNzb3IuZnJvbV9wcmV0cmFpbmVkKG1vZGVsX2NoZWNrcG9pbnQpJTBBaW5wdXRzJTIwJTNEJTIwcHJvY2Vzc29yKCUwQSUyMCUyMCUyMCUyMHRleHQlM0R0ZXh0JTJDJTBBJTIwJTIwJTIwJTIwYXVkaW8lM0RhdWRpbyUyQyUwQSUyMCUyMCUyMCUyMGdlbmVyYXRpb24lM0RGYWxzZSUyQyUwQSUyMCUyMCUyMCUyMG91dHB1dF9sYWJlbHMlM0RUcnVlJTJDJTBBJTIwJTIwJTIwJTIwcGFkZGluZyUzRFRydWUlMkMlMEElMjAlMjAlMjAlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyJTBBKS50byh0b3JjaF9kZXZpY2UpJTBBJTBBbW9kZWwlMjAlM0QlMjBEaWFGb3JDb25kaXRpb25hbEdlbmVyYXRpb24uZnJvbV9wcmV0cmFpbmVkKG1vZGVsX2NoZWNrcG9pbnQpLnRvKHRvcmNoX2RldmljZSklMEFvdXQlMjAlM0QlMjBtb2RlbCgqKmlucHV0cyklMEFvdXQubG9zcy5iYWNrd2FyZCgp",highlighted:`<span class="hljs-keyword">from</span> datasets <span class="hljs-keyword">import</span> load_dataset, Audio
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoProcessor, DiaForConditionalGeneration, infer_device

torch_device = infer_device()
model_checkpoint = <span class="hljs-string">&quot;nari-labs/Dia-1.6B-0626&quot;</span>

ds = load_dataset(<span class="hljs-string">&quot;hf-internal-testing/dailytalk-dummy&quot;</span>, split=<span class="hljs-string">&quot;train&quot;</span>)
ds = ds.cast_column(<span class="hljs-string">&quot;audio&quot;</span>, Audio(sampling_rate=<span class="hljs-number">44100</span>))
audio = ds[-<span class="hljs-number">1</span>][<span class="hljs-string">&quot;audio&quot;</span>][<span class="hljs-string">&quot;array&quot;</span>]
<span class="hljs-comment"># text is a transcript of the audio</span>
text = [<span class="hljs-string">&quot;[S1] I know. It&#x27;s going to save me a lot of money, I hope.&quot;</span>]

processor = AutoProcessor.from_pretrained(model_checkpoint)
inputs = processor(
    text=text,
    audio=audio,
    generation=<span class="hljs-literal">False</span>,
    output_labels=<span class="hljs-literal">True</span>,
    padding=<span class="hljs-literal">True</span>,
    return_tensors=<span class="hljs-string">&quot;pt&quot;</span>
).to(torch_device)

model = DiaForConditionalGeneration.from_pretrained(model_checkpoint).to(torch_device)
out = model(**inputs)
out.loss.backward()`,wrap:!1}}),he=new $({props:{title:"DiaConfig",local:"transformers.DiaConfig",headingTag:"h2"}}),ue=new w({props:{name:"class transformers.DiaConfig",anchor:"transformers.DiaConfig",parameters:[{name:"encoder_config",val:": typing.Optional[transformers.models.dia.configuration_dia.DiaEncoderConfig] = None"},{name:"decoder_config",val:": typing.Optional[transformers.models.dia.configuration_dia.DiaDecoderConfig] = None"},{name:"norm_eps",val:": float = 1e-05"},{name:"is_encoder_decoder",val:": bool = True"},{name:"pad_token_id",val:": int = 1025"},{name:"eos_token_id",val:": int = 1024"},{name:"bos_token_id",val:": int = 1026"},{name:"delay_pattern",val:": typing.Optional[list[int]] = None"},{name:"initializer_range",val:": float = 0.02"},{name:"use_cache",val:": bool = True"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.DiaConfig.encoder_config",description:`<strong>encoder_config</strong> (<code>DiaEncoderConfig</code>, <em>optional</em>) &#x2014;
Configuration for the encoder part of the model. If not provided, a default <code>DiaEncoderConfig</code> will be used.`,name:"encoder_config"},{anchor:"transformers.DiaConfig.decoder_config",description:`<strong>decoder_config</strong> (<code>DiaDecoderConfig</code>, <em>optional</em>) &#x2014;
Configuration for the decoder part of the model. If not provided, a default <code>DiaDecoderConfig</code> will be used.`,name:"decoder_config"},{anchor:"transformers.DiaConfig.norm_eps",description:`<strong>norm_eps</strong> (<code>float</code>, <em>optional</em>, defaults to 1e-05) &#x2014;
The epsilon used by the normalization layers.`,name:"norm_eps"},{anchor:"transformers.DiaConfig.is_encoder_decoder",description:`<strong>is_encoder_decoder</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Indicating that this model uses an encoder-decoder architecture.`,name:"is_encoder_decoder"},{anchor:"transformers.DiaConfig.pad_token_id",description:`<strong>pad_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 1025) &#x2014;
Padding token id.`,name:"pad_token_id"},{anchor:"transformers.DiaConfig.eos_token_id",description:`<strong>eos_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 1024) &#x2014;
End of stream token id.`,name:"eos_token_id"},{anchor:"transformers.DiaConfig.bos_token_id",description:`<strong>bos_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 1026) &#x2014;
Beginning of stream token id.`,name:"bos_token_id"},{anchor:"transformers.DiaConfig.delay_pattern",description:`<strong>delay_pattern</strong> (<code>list[int]</code>, <em>optional</em>, defaults to <code>[0, 8, 9, 10, 11, 12, 13, 14, 15]</code>) &#x2014;
The delay pattern for the decoder. The length of this list must match <code>decoder_config.num_channels</code>.`,name:"delay_pattern"},{anchor:"transformers.DiaConfig.initializer_range",description:`<strong>initializer_range</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.`,name:"initializer_range"},{anchor:"transformers.DiaConfig.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not the model should return the last key/values attentions (not used by all models).`,name:"use_cache"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/dia/configuration_dia.py#L282"}}),L=new wn({props:{anchor:"transformers.DiaConfig.example",$$slots:{default:[kn]},$$scope:{ctx:P}}}),fe=new w({props:{name:"get_text_config",anchor:"transformers.DiaConfig.get_text_config",parameters:[{name:"*args",val:""},{name:"**kwargs",val:""}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/dia/configuration_dia.py#L371"}}),ge=new $({props:{title:"DiaDecoderConfig",local:"transformers.DiaDecoderConfig",headingTag:"h2"}}),_e=new w({props:{name:"class transformers.DiaDecoderConfig",anchor:"transformers.DiaDecoderConfig",parameters:[{name:"max_position_embeddings",val:": int = 3072"},{name:"num_hidden_layers",val:": int = 18"},{name:"hidden_size",val:": int = 2048"},{name:"intermediate_size",val:": int = 8192"},{name:"num_attention_heads",val:": int = 16"},{name:"num_key_value_heads",val:": int = 4"},{name:"head_dim",val:": int = 128"},{name:"cross_num_attention_heads",val:": int = 16"},{name:"cross_head_dim",val:": int = 128"},{name:"cross_num_key_value_heads",val:": int = 16"},{name:"cross_hidden_size",val:": int = 1024"},{name:"norm_eps",val:": float = 1e-05"},{name:"vocab_size",val:": int = 1028"},{name:"hidden_act",val:": str = 'silu'"},{name:"num_channels",val:": int = 9"},{name:"rope_theta",val:": float = 10000.0"},{name:"rope_scaling",val:": typing.Optional[dict] = None"},{name:"initializer_range",val:": float = 0.02"},{name:"use_cache",val:": bool = True"},{name:"is_encoder_decoder",val:": bool = True"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.DiaDecoderConfig.max_position_embeddings",description:`<strong>max_position_embeddings</strong> (<code>int</code>, <em>optional</em>, defaults to 3072) &#x2014;
The maximum sequence length that this model might ever be used with.`,name:"max_position_embeddings"},{anchor:"transformers.DiaDecoderConfig.num_hidden_layers",description:`<strong>num_hidden_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 18) &#x2014;
Number of hidden layers in the Transformer decoder.`,name:"num_hidden_layers"},{anchor:"transformers.DiaDecoderConfig.hidden_size",description:`<strong>hidden_size</strong> (<code>int</code>, <em>optional</em>, defaults to 2048) &#x2014;
Dimensionality of the decoder layers and the pooler layer.`,name:"hidden_size"},{anchor:"transformers.DiaDecoderConfig.intermediate_size",description:`<strong>intermediate_size</strong> (<code>int</code>, <em>optional</em>, defaults to 8192) &#x2014;
Dimensionality of the &#x201C;intermediate&#x201D; (often named feed-forward) layer in the Transformer decoder.`,name:"intermediate_size"},{anchor:"transformers.DiaDecoderConfig.num_attention_heads",description:`<strong>num_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 16) &#x2014;
Number of attention heads for each attention layer in the Transformer decoder.`,name:"num_attention_heads"},{anchor:"transformers.DiaDecoderConfig.num_key_value_heads",description:`<strong>num_key_value_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 4) &#x2014;
Number of key and value heads for each attention layer in the Transformer decoder.`,name:"num_key_value_heads"},{anchor:"transformers.DiaDecoderConfig.head_dim",description:`<strong>head_dim</strong> (<code>int</code>, <em>optional</em>, defaults to 128) &#x2014;
Dimensionality of the attention head.`,name:"head_dim"},{anchor:"transformers.DiaDecoderConfig.cross_num_attention_heads",description:`<strong>cross_num_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 16) &#x2014;
Number of attention heads for each cross-attention layer in the Transformer decoder.`,name:"cross_num_attention_heads"},{anchor:"transformers.DiaDecoderConfig.cross_head_dim",description:`<strong>cross_head_dim</strong> (<code>int</code>, <em>optional</em>, defaults to 128) &#x2014;
Dimensionality of the cross-attention head.`,name:"cross_head_dim"},{anchor:"transformers.DiaDecoderConfig.cross_num_key_value_heads",description:`<strong>cross_num_key_value_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 16) &#x2014;
Number of key and value heads for each cross-attention layer in the Transformer decoder.`,name:"cross_num_key_value_heads"},{anchor:"transformers.DiaDecoderConfig.cross_hidden_size",description:`<strong>cross_hidden_size</strong> (<code>int</code>, <em>optional</em>, defaults to 1024) &#x2014;
Dimensionality of the cross-attention layers.`,name:"cross_hidden_size"},{anchor:"transformers.DiaDecoderConfig.norm_eps",description:`<strong>norm_eps</strong> (<code>float</code>, <em>optional</em>, defaults to 1e-05) &#x2014;
The epsilon used by the normalization layers.`,name:"norm_eps"},{anchor:"transformers.DiaDecoderConfig.vocab_size",description:`<strong>vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 1028) &#x2014;
Vocabulary size of the Dia model. Defines the number of different tokens that can be represented by the
<code>inputs_ids</code> passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/dia#transformers.DiaModel">DiaModel</a>.`,name:"vocab_size"},{anchor:"transformers.DiaDecoderConfig.hidden_act",description:`<strong>hidden_act</strong> (<code>str</code> or <code>function</code>, <em>optional</em>, defaults to <code>&quot;silu&quot;</code>) &#x2014;
The non-linear activation function (function or string) in the decoder. If string, <code>&quot;gelu&quot;</code>, <code>&quot;relu&quot;</code>,
<code>&quot;swish&quot;</code> and <code>&quot;gelu_new&quot;</code> are supported.`,name:"hidden_act"},{anchor:"transformers.DiaDecoderConfig.num_channels",description:`<strong>num_channels</strong> (<code>int</code>, <em>optional</em>, defaults to 9) &#x2014;
Number of channels for the Dia decoder.`,name:"num_channels"},{anchor:"transformers.DiaDecoderConfig.rope_theta",description:`<strong>rope_theta</strong> (<code>float</code>, <em>optional</em>, defaults to 10000.0) &#x2014;
The base period of the RoPE embeddings.`,name:"rope_theta"},{anchor:"transformers.DiaDecoderConfig.rope_scaling",description:`<strong>rope_scaling</strong> (<code>dict</code>, <em>optional</em>) &#x2014;
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
<code>short_factor</code> (<code>List[float]</code>, </em>optional<em>):
Only used with &#x2018;longrope&#x2019;. The scaling factor to be applied to short contexts (&lt;
<code>original_max_position_embeddings</code>). Must be a list of numbers with the same length as the hidden
size divided by the number of attention heads divided by 2
<code>long_factor</code> (<code>List[float]</code>, </em>optional<em>):
Only used with &#x2018;longrope&#x2019;. The scaling factor to be applied to long contexts (&lt;
<code>original_max_position_embeddings</code>). Must be a list of numbers with the same length as the hidden
size divided by the number of attention heads divided by 2
<code>low_freq_factor</code> (<code>float</code>, </em>optional<em>):
Only used with &#x2018;llama3&#x2019;. Scaling factor applied to low frequency components of the RoPE
<code>high_freq_factor</code> (<code>float</code>, </em>optional*):
Only used with &#x2018;llama3&#x2019;. Scaling factor applied to high frequency components of the RoPE`,name:"rope_scaling"},{anchor:"transformers.DiaDecoderConfig.initializer_range",description:`<strong>initializer_range</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.`,name:"initializer_range"},{anchor:"transformers.DiaDecoderConfig.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not the model should return the last key/values attentions (not used by all models).`,name:"use_cache"},{anchor:"transformers.DiaDecoderConfig.is_encoder_decoder",description:`<strong>is_encoder_decoder</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Indicating that this model is part of an encoder-decoder architecture.`,name:"is_encoder_decoder"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/dia/configuration_dia.py#L141"}}),ye=new $({props:{title:"DiaEncoderConfig",local:"transformers.DiaEncoderConfig",headingTag:"h2"}}),be=new w({props:{name:"class transformers.DiaEncoderConfig",anchor:"transformers.DiaEncoderConfig",parameters:[{name:"max_position_embeddings",val:": int = 1024"},{name:"num_hidden_layers",val:": int = 12"},{name:"hidden_size",val:": int = 1024"},{name:"num_attention_heads",val:": int = 16"},{name:"num_key_value_heads",val:": int = 16"},{name:"head_dim",val:": int = 128"},{name:"intermediate_size",val:": int = 4096"},{name:"norm_eps",val:": float = 1e-05"},{name:"vocab_size",val:": int = 256"},{name:"hidden_act",val:": str = 'silu'"},{name:"rope_theta",val:": float = 10000.0"},{name:"rope_scaling",val:": typing.Optional[dict] = None"},{name:"initializer_range",val:": float = 0.02"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.DiaEncoderConfig.max_position_embeddings",description:`<strong>max_position_embeddings</strong> (<code>int</code>, <em>optional</em>, defaults to 1024) &#x2014;
The maximum sequence length that this model might ever be used with.`,name:"max_position_embeddings"},{anchor:"transformers.DiaEncoderConfig.num_hidden_layers",description:`<strong>num_hidden_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 12) &#x2014;
Number of hidden layers in the Transformer encoder.`,name:"num_hidden_layers"},{anchor:"transformers.DiaEncoderConfig.hidden_size",description:`<strong>hidden_size</strong> (<code>int</code>, <em>optional</em>, defaults to 1024) &#x2014;
Dimensionality of the encoder layers and the pooler layer.`,name:"hidden_size"},{anchor:"transformers.DiaEncoderConfig.num_attention_heads",description:`<strong>num_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 16) &#x2014;
Number of attention heads for each attention layer in the Transformer encoder.`,name:"num_attention_heads"},{anchor:"transformers.DiaEncoderConfig.num_key_value_heads",description:`<strong>num_key_value_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 16) &#x2014;
Number of key and value heads for each attention layer in the Transformer encoder.`,name:"num_key_value_heads"},{anchor:"transformers.DiaEncoderConfig.head_dim",description:`<strong>head_dim</strong> (<code>int</code>, <em>optional</em>, defaults to 128) &#x2014;
Dimensionality of the attention head.`,name:"head_dim"},{anchor:"transformers.DiaEncoderConfig.intermediate_size",description:`<strong>intermediate_size</strong> (<code>int</code>, <em>optional</em>, defaults to 4096) &#x2014;
Dimensionality of the &#x201C;intermediate&#x201D; (often named feed-forward) layer in the Transformer encoder.`,name:"intermediate_size"},{anchor:"transformers.DiaEncoderConfig.norm_eps",description:`<strong>norm_eps</strong> (<code>float</code>, <em>optional</em>, defaults to 1e-05) &#x2014;
The epsilon used by the normalization layers.`,name:"norm_eps"},{anchor:"transformers.DiaEncoderConfig.vocab_size",description:`<strong>vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 256) &#x2014;
Vocabulary size of the Dia model. Defines the number of different tokens that can be represented by the
<code>inputs_ids</code> passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/dia#transformers.DiaModel">DiaModel</a>.`,name:"vocab_size"},{anchor:"transformers.DiaEncoderConfig.hidden_act",description:`<strong>hidden_act</strong> (<code>str</code> or <code>function</code>, <em>optional</em>, defaults to <code>&quot;silu&quot;</code>) &#x2014;
The non-linear activation function (function or string) in the encoder and pooler. If string, <code>&quot;gelu&quot;</code>,
<code>&quot;relu&quot;</code>, <code>&quot;swish&quot;</code> and <code>&quot;gelu_new&quot;</code> are supported.`,name:"hidden_act"},{anchor:"transformers.DiaEncoderConfig.rope_theta",description:`<strong>rope_theta</strong> (<code>float</code>, <em>optional</em>, defaults to 10000.0) &#x2014;
The base period of the RoPE embeddings.`,name:"rope_theta"},{anchor:"transformers.DiaEncoderConfig.rope_scaling",description:`<strong>rope_scaling</strong> (<code>dict</code>, <em>optional</em>) &#x2014;
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
<code>short_factor</code> (<code>List[float]</code>, </em>optional<em>):
Only used with &#x2018;longrope&#x2019;. The scaling factor to be applied to short contexts (&lt;
<code>original_max_position_embeddings</code>). Must be a list of numbers with the same length as the hidden
size divided by the number of attention heads divided by 2
<code>long_factor</code> (<code>List[float]</code>, </em>optional<em>):
Only used with &#x2018;longrope&#x2019;. The scaling factor to be applied to long contexts (&lt;
<code>original_max_position_embeddings</code>). Must be a list of numbers with the same length as the hidden
size divided by the number of attention heads divided by 2
<code>low_freq_factor</code> (<code>float</code>, </em>optional<em>):
Only used with &#x2018;llama3&#x2019;. Scaling factor applied to low frequency components of the RoPE
<code>high_freq_factor</code> (<code>float</code>, </em>optional*):
Only used with &#x2018;llama3&#x2019;. Scaling factor applied to high frequency components of the RoPE`,name:"rope_scaling"},{anchor:"transformers.DiaEncoderConfig.initializer_range",description:`<strong>initializer_range</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.`,name:"initializer_range"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/dia/configuration_dia.py#L27"}}),ve=new $({props:{title:"DiaTokenizer",local:"transformers.DiaTokenizer",headingTag:"h2"}}),Te=new w({props:{name:"class transformers.DiaTokenizer",anchor:"transformers.DiaTokenizer",parameters:[{name:"pad_token",val:": typing.Optional[str] = '<pad>'"},{name:"unk_token",val:": typing.Optional[str] = '<pad>'"},{name:"max_length",val:": typing.Optional[int] = 1024"},{name:"offset",val:": int = 0"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.DiaTokenizer.pad_token",description:`<strong>pad_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;pad&gt;&quot;</code>) &#x2014;
The token used for padding, for example when batching sequences of different lengths.`,name:"pad_token"},{anchor:"transformers.DiaTokenizer.unk_token",description:`<strong>unk_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;pad&gt;&quot;</code>) &#x2014;
The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
token instead.`,name:"unk_token"},{anchor:"transformers.DiaTokenizer.max_length",description:`<strong>max_length</strong> (<code>int</code>, <em>optional</em>, defaults to 1024) &#x2014;
The maximum length of the sequences when encoding. Sequences longer than this will be truncated.`,name:"max_length"},{anchor:"transformers.DiaTokenizer.offset",description:`<strong>offset</strong> (<code>int</code>, <em>optional</em>, defaults to 0) &#x2014;
The offset of the tokenizer.`,name:"offset"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/dia/tokenization_dia.py#L26"}}),we=new w({props:{name:"__call__",anchor:"transformers.DiaTokenizer.__call__",parameters:[{name:"text",val:": typing.Union[str, list[str], list[list[str]], NoneType] = None"},{name:"text_pair",val:": typing.Union[str, list[str], list[list[str]], NoneType] = None"},{name:"text_target",val:": typing.Union[str, list[str], list[list[str]], NoneType] = None"},{name:"text_pair_target",val:": typing.Union[str, list[str], list[list[str]], NoneType] = None"},{name:"add_special_tokens",val:": bool = True"},{name:"padding",val:": typing.Union[bool, str, transformers.utils.generic.PaddingStrategy] = False"},{name:"truncation",val:": typing.Union[bool, str, transformers.tokenization_utils_base.TruncationStrategy, NoneType] = None"},{name:"max_length",val:": typing.Optional[int] = None"},{name:"stride",val:": int = 0"},{name:"is_split_into_words",val:": bool = False"},{name:"pad_to_multiple_of",val:": typing.Optional[int] = None"},{name:"padding_side",val:": typing.Optional[str] = None"},{name:"return_tensors",val:": typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None"},{name:"return_token_type_ids",val:": typing.Optional[bool] = None"},{name:"return_attention_mask",val:": typing.Optional[bool] = None"},{name:"return_overflowing_tokens",val:": bool = False"},{name:"return_special_tokens_mask",val:": bool = False"},{name:"return_offsets_mapping",val:": bool = False"},{name:"return_length",val:": bool = False"},{name:"verbose",val:": bool = True"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.DiaTokenizer.__call__.text",description:`<strong>text</strong> (<code>str</code>, <code>list[str]</code>, <code>list[list[str]]</code>, <em>optional</em>) &#x2014;
The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
(pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
<code>is_split_into_words=True</code> (to lift the ambiguity with a batch of sequences).`,name:"text"},{anchor:"transformers.DiaTokenizer.__call__.text_pair",description:`<strong>text_pair</strong> (<code>str</code>, <code>list[str]</code>, <code>list[list[str]]</code>, <em>optional</em>) &#x2014;
The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
(pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
<code>is_split_into_words=True</code> (to lift the ambiguity with a batch of sequences).`,name:"text_pair"},{anchor:"transformers.DiaTokenizer.__call__.text_target",description:`<strong>text_target</strong> (<code>str</code>, <code>list[str]</code>, <code>list[list[str]]</code>, <em>optional</em>) &#x2014;
The sequence or batch of sequences to be encoded as target texts. Each sequence can be a string or a
list of strings (pretokenized string). If the sequences are provided as list of strings (pretokenized),
you must set <code>is_split_into_words=True</code> (to lift the ambiguity with a batch of sequences).`,name:"text_target"},{anchor:"transformers.DiaTokenizer.__call__.text_pair_target",description:`<strong>text_pair_target</strong> (<code>str</code>, <code>list[str]</code>, <code>list[list[str]]</code>, <em>optional</em>) &#x2014;
The sequence or batch of sequences to be encoded as target texts. Each sequence can be a string or a
list of strings (pretokenized string). If the sequences are provided as list of strings (pretokenized),
you must set <code>is_split_into_words=True</code> (to lift the ambiguity with a batch of sequences).`,name:"text_pair_target"},{anchor:"transformers.DiaTokenizer.__call__.add_special_tokens",description:`<strong>add_special_tokens</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to add special tokens when encoding the sequences. This will use the underlying
<code>PretrainedTokenizerBase.build_inputs_with_special_tokens</code> function, which defines which tokens are
automatically added to the input ids. This is useful if you want to add <code>bos</code> or <code>eos</code> tokens
automatically.`,name:"add_special_tokens"},{anchor:"transformers.DiaTokenizer.__call__.padding",description:`<strong>padding</strong> (<code>bool</code>, <code>str</code> or <a href="/docs/transformers/v4.56.2/en/internal/file_utils#transformers.utils.PaddingStrategy">PaddingStrategy</a>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Activates and controls padding. Accepts the following values:</p>
<ul>
<li><code>True</code> or <code>&apos;longest&apos;</code>: Pad to the longest sequence in the batch (or no padding if only a single
sequence is provided).</li>
<li><code>&apos;max_length&apos;</code>: Pad to a maximum length specified with the argument <code>max_length</code> or to the maximum
acceptable input length for the model if that argument is not provided.</li>
<li><code>False</code> or <code>&apos;do_not_pad&apos;</code> (default): No padding (i.e., can output a batch with sequences of different
lengths).</li>
</ul>`,name:"padding"},{anchor:"transformers.DiaTokenizer.__call__.truncation",description:`<strong>truncation</strong> (<code>bool</code>, <code>str</code> or <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.tokenization_utils_base.TruncationStrategy">TruncationStrategy</a>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Activates and controls truncation. Accepts the following values:</p>
<ul>
<li><code>True</code> or <code>&apos;longest_first&apos;</code>: Truncate to a maximum length specified with the argument <code>max_length</code> or
to the maximum acceptable input length for the model if that argument is not provided. This will
truncate token by token, removing a token from the longest sequence in the pair if a pair of
sequences (or a batch of pairs) is provided.</li>
<li><code>&apos;only_first&apos;</code>: Truncate to a maximum length specified with the argument <code>max_length</code> or to the
maximum acceptable input length for the model if that argument is not provided. This will only
truncate the first sequence of a pair if a pair of sequences (or a batch of pairs) is provided.</li>
<li><code>&apos;only_second&apos;</code>: Truncate to a maximum length specified with the argument <code>max_length</code> or to the
maximum acceptable input length for the model if that argument is not provided. This will only
truncate the second sequence of a pair if a pair of sequences (or a batch of pairs) is provided.</li>
<li><code>False</code> or <code>&apos;do_not_truncate&apos;</code> (default): No truncation (i.e., can output batch with sequence lengths
greater than the model maximum admissible input size).</li>
</ul>`,name:"truncation"},{anchor:"transformers.DiaTokenizer.__call__.max_length",description:`<strong>max_length</strong> (<code>int</code>, <em>optional</em>) &#x2014;
Controls the maximum length to use by one of the truncation/padding parameters.</p>
<p>If left unset or set to <code>None</code>, this will use the predefined model maximum length if a maximum length
is required by one of the truncation/padding parameters. If the model has no specific maximum input
length (like XLNet) truncation/padding to a maximum length will be deactivated.`,name:"max_length"},{anchor:"transformers.DiaTokenizer.__call__.stride",description:`<strong>stride</strong> (<code>int</code>, <em>optional</em>, defaults to 0) &#x2014;
If set to a number along with <code>max_length</code>, the overflowing tokens returned when
<code>return_overflowing_tokens=True</code> will contain some tokens from the end of the truncated sequence
returned to provide some overlap between truncated and overflowing sequences. The value of this
argument defines the number of overlapping tokens.`,name:"stride"},{anchor:"transformers.DiaTokenizer.__call__.is_split_into_words",description:`<strong>is_split_into_words</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not the input is already pre-tokenized (e.g., split into words). If set to <code>True</code>, the
tokenizer assumes the input is already split into words (for instance, by splitting it on whitespace)
which it will tokenize. This is useful for NER or token classification.`,name:"is_split_into_words"},{anchor:"transformers.DiaTokenizer.__call__.pad_to_multiple_of",description:`<strong>pad_to_multiple_of</strong> (<code>int</code>, <em>optional</em>) &#x2014;
If set will pad the sequence to a multiple of the provided value. Requires <code>padding</code> to be activated.
This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
<code>&gt;= 7.5</code> (Volta).`,name:"pad_to_multiple_of"},{anchor:"transformers.DiaTokenizer.__call__.padding_side",description:`<strong>padding_side</strong> (<code>str</code>, <em>optional</em>) &#x2014;
The side on which the model should have padding applied. Should be selected between [&#x2018;right&#x2019;, &#x2018;left&#x2019;].
Default value is picked from the class attribute of the same name.`,name:"padding_side"},{anchor:"transformers.DiaTokenizer.__call__.return_tensors",description:`<strong>return_tensors</strong> (<code>str</code> or <a href="/docs/transformers/v4.56.2/en/internal/file_utils#transformers.TensorType">TensorType</a>, <em>optional</em>) &#x2014;
If set, will return tensors instead of list of python integers. Acceptable values are:</p>
<ul>
<li><code>&apos;tf&apos;</code>: Return TensorFlow <code>tf.constant</code> objects.</li>
<li><code>&apos;pt&apos;</code>: Return PyTorch <code>torch.Tensor</code> objects.</li>
<li><code>&apos;np&apos;</code>: Return Numpy <code>np.ndarray</code> objects.</li>
</ul>`,name:"return_tensors"},{anchor:"transformers.DiaTokenizer.__call__.return_token_type_ids",description:`<strong>return_token_type_ids</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether to return token type IDs. If left to the default, will return the token type IDs according to
the specific tokenizer&#x2019;s default, defined by the <code>return_outputs</code> attribute.</p>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a>`,name:"return_token_type_ids"},{anchor:"transformers.DiaTokenizer.__call__.return_attention_mask",description:`<strong>return_attention_mask</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether to return the attention mask. If left to the default, will return the attention mask according
to the specific tokenizer&#x2019;s default, defined by the <code>return_outputs</code> attribute.</p>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"return_attention_mask"},{anchor:"transformers.DiaTokenizer.__call__.return_overflowing_tokens",description:`<strong>return_overflowing_tokens</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to return overflowing token sequences. If a pair of sequences of input ids (or a batch
of pairs) is provided with <code>truncation_strategy = longest_first</code> or <code>True</code>, an error is raised instead
of returning overflowing tokens.`,name:"return_overflowing_tokens"},{anchor:"transformers.DiaTokenizer.__call__.return_special_tokens_mask",description:`<strong>return_special_tokens_mask</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to return special tokens mask information.`,name:"return_special_tokens_mask"},{anchor:"transformers.DiaTokenizer.__call__.return_offsets_mapping",description:`<strong>return_offsets_mapping</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to return <code>(char_start, char_end)</code> for each token.</p>
<p>This is only available on fast tokenizers inheriting from <a href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast">PreTrainedTokenizerFast</a>, if using
Python&#x2019;s tokenizer, this method will raise <code>NotImplementedError</code>.`,name:"return_offsets_mapping"},{anchor:"transformers.DiaTokenizer.__call__.return_length",description:`<strong>return_length</strong>  (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to return the lengths of the encoded inputs.`,name:"return_length"},{anchor:"transformers.DiaTokenizer.__call__.verbose",description:`<strong>verbose</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not to print more information and warnings.`,name:"verbose"},{anchor:"transformers.DiaTokenizer.__call__.*kwargs",description:"*<strong>*kwargs</strong> &#x2014; passed to the <code>self.tokenize()</code> method",name:"*kwargs"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L2828",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.BatchEncoding"
>BatchEncoding</a> with the following fields:</p>
<ul>
<li>
<p><strong>input_ids</strong> — List of token ids to be fed to a model.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a></p>
</li>
<li>
<p><strong>token_type_ids</strong> — List of token type ids to be fed to a model (when <code>return_token_type_ids=True</code> or
if <em>“token_type_ids”</em> is in <code>self.model_input_names</code>).</p>
<p><a href="../glossary#token-type-ids">What are token type IDs?</a></p>
</li>
<li>
<p><strong>attention_mask</strong> — List of indices specifying which tokens should be attended to by the model (when
<code>return_attention_mask=True</code> or if <em>“attention_mask”</em> is in <code>self.model_input_names</code>).</p>
<p><a href="../glossary#attention-mask">What are attention masks?</a></p>
</li>
<li>
<p><strong>overflowing_tokens</strong> — List of overflowing tokens sequences (when a <code>max_length</code> is specified and
<code>return_overflowing_tokens=True</code>).</p>
</li>
<li>
<p><strong>num_truncated_tokens</strong> — Number of tokens truncated (when a <code>max_length</code> is specified and
<code>return_overflowing_tokens=True</code>).</p>
</li>
<li>
<p><strong>special_tokens_mask</strong> — List of 0s and 1s, with 1 specifying added special tokens and 0 specifying
regular sequence tokens (when <code>add_special_tokens=True</code> and <code>return_special_tokens_mask=True</code>).</p>
</li>
<li>
<p><strong>length</strong> — The length of the inputs (when <code>return_length=True</code>)</p>
</li>
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><a
  href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.BatchEncoding"
>BatchEncoding</a></p>
`}}),xe=new $({props:{title:"DiaFeatureExtractor",local:"transformers.DiaFeatureExtractor",headingTag:"h2"}}),ke=new w({props:{name:"class transformers.DiaFeatureExtractor",anchor:"transformers.DiaFeatureExtractor",parameters:[{name:"feature_size",val:": int = 1"},{name:"sampling_rate",val:": int = 16000"},{name:"padding_value",val:": float = 0.0"},{name:"hop_length",val:": int = 512"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.DiaFeatureExtractor.feature_size",description:`<strong>feature_size</strong> (<code>int</code>, <em>optional</em>, defaults to 1) &#x2014;
The feature dimension of the extracted features. Use 1 for mono, 2 for stereo.`,name:"feature_size"},{anchor:"transformers.DiaFeatureExtractor.sampling_rate",description:`<strong>sampling_rate</strong> (<code>int</code>, <em>optional</em>, defaults to 16000) &#x2014;
The sampling rate at which the audio waveform should be digitalized, expressed in hertz (Hz).`,name:"sampling_rate"},{anchor:"transformers.DiaFeatureExtractor.padding_value",description:`<strong>padding_value</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The value that is used for padding.`,name:"padding_value"},{anchor:"transformers.DiaFeatureExtractor.hop_length",description:`<strong>hop_length</strong> (<code>int</code>, <em>optional</em>, defaults to 512) &#x2014;
Overlap length between successive windows.`,name:"hop_length"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/dia/feature_extraction_dia.py#L29"}}),Me=new w({props:{name:"__call__",anchor:"transformers.DiaFeatureExtractor.__call__",parameters:[{name:"raw_audio",val:": typing.Union[numpy.ndarray, list[float], list[numpy.ndarray], list[list[float]]]"},{name:"padding",val:": typing.Union[bool, str, transformers.utils.generic.PaddingStrategy, NoneType] = None"},{name:"truncation",val:": typing.Optional[bool] = False"},{name:"max_length",val:": typing.Optional[int] = None"},{name:"return_tensors",val:": typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None"},{name:"sampling_rate",val:": typing.Optional[int] = None"}],parametersDescription:[{anchor:"transformers.DiaFeatureExtractor.__call__.raw_audio",description:`<strong>raw_audio</strong> (<code>np.ndarray</code>, <code>list[float]</code>, <code>list[np.ndarray]</code>, <code>list[list[float]]</code>) &#x2014;
The sequence or batch of sequences to be processed. Each sequence can be a numpy array, a list of float
values, a list of numpy arrays or a list of list of float values. The numpy array must be of shape
<code>(num_samples,)</code> for mono audio (<code>feature_size = 1</code>), or <code>(2, num_samples)</code> for stereo audio
(<code>feature_size = 2</code>).`,name:"raw_audio"},{anchor:"transformers.DiaFeatureExtractor.__call__.padding",description:`<strong>padding</strong> (<code>bool</code>, <code>str</code> or <a href="/docs/transformers/v4.56.2/en/internal/file_utils#transformers.utils.PaddingStrategy">PaddingStrategy</a>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Select a strategy to pad the returned sequences (according to the model&#x2019;s padding side and padding
index) among:</p>
<ul>
<li><code>True</code> or <code>&apos;longest&apos;</code>: Pad to the longest sequence in the batch (or no padding if only a single
sequence if provided).</li>
<li><code>&apos;max_length&apos;</code>: Pad to a maximum length specified with the argument <code>max_length</code> or to the maximum
acceptable input length for the model if that argument is not provided.</li>
<li><code>False</code> or <code>&apos;do_not_pad&apos;</code> (default): No padding (i.e., can output a batch with sequences of different
lengths).</li>
</ul>`,name:"padding"},{anchor:"transformers.DiaFeatureExtractor.__call__.truncation",description:`<strong>truncation</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Activates truncation to cut input sequences longer than <code>max_length</code> to <code>max_length</code>.`,name:"truncation"},{anchor:"transformers.DiaFeatureExtractor.__call__.max_length",description:`<strong>max_length</strong> (<code>int</code>, <em>optional</em>) &#x2014;
Maximum length of the returned list and optionally padding length (see above).`,name:"max_length"},{anchor:"transformers.DiaFeatureExtractor.__call__.return_tensors",description:`<strong>return_tensors</strong> (<code>str</code> or <a href="/docs/transformers/v4.56.2/en/internal/file_utils#transformers.TensorType">TensorType</a>, <em>optional</em>, default to &#x2018;pt&#x2019;) &#x2014;
If set, will return tensors instead of list of python integers. Acceptable values are:</p>
<ul>
<li><code>&apos;tf&apos;</code>: Return TensorFlow <code>tf.constant</code> objects.</li>
<li><code>&apos;pt&apos;</code>: Return PyTorch <code>torch.Tensor</code> objects.</li>
<li><code>&apos;np&apos;</code>: Return Numpy <code>np.ndarray</code> objects.</li>
</ul>`,name:"return_tensors"},{anchor:"transformers.DiaFeatureExtractor.__call__.sampling_rate",description:`<strong>sampling_rate</strong> (<code>int</code>, <em>optional</em>) &#x2014;
The sampling rate at which the <code>audio</code> input was sampled. It is strongly recommended to pass
<code>sampling_rate</code> at the forward call to prevent silent errors.`,name:"sampling_rate"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/dia/feature_extraction_dia.py#L60"}}),De=new $({props:{title:"DiaProcessor",local:"transformers.DiaProcessor",headingTag:"h2"}}),$e=new w({props:{name:"class transformers.DiaProcessor",anchor:"transformers.DiaProcessor",parameters:[{name:"feature_extractor",val:""},{name:"tokenizer",val:""},{name:"audio_tokenizer",val:""}],parametersDescription:[{anchor:"transformers.DiaProcessor.feature_extractor",description:`<strong>feature_extractor</strong> (<code>DiaFeatureExtractor</code>) &#x2014;
An instance of <a href="/docs/transformers/v4.56.2/en/model_doc/dia#transformers.DiaFeatureExtractor">DiaFeatureExtractor</a>. The feature extractor is a required input.`,name:"feature_extractor"},{anchor:"transformers.DiaProcessor.tokenizer",description:`<strong>tokenizer</strong> (<code>DiaTokenizer</code>) &#x2014;
An instance of <a href="/docs/transformers/v4.56.2/en/model_doc/dia#transformers.DiaTokenizer">DiaTokenizer</a>. The tokenizer is a required input.`,name:"tokenizer"},{anchor:"transformers.DiaProcessor.audio_tokenizer",description:`<strong>audio_tokenizer</strong> (<code>DacModel</code>) &#x2014;
An instance of <a href="/docs/transformers/v4.56.2/en/model_doc/dac#transformers.DacModel">DacModel</a> used to encode/decode audio into/from codebooks. It is is a required input.`,name:"audio_tokenizer"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/dia/processing_dia.py#L62"}}),Ce=new w({props:{name:"__call__",anchor:"transformers.DiaProcessor.__call__",parameters:[{name:"text",val:": typing.Union[str, list[str]]"},{name:"audio",val:": typing.Union[numpy.ndarray, ForwardRef('torch.Tensor'), typing.Sequence[numpy.ndarray], typing.Sequence[ForwardRef('torch.Tensor')], NoneType] = None"},{name:"output_labels",val:": typing.Optional[bool] = False"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.models.dia.processing_dia.DiaProcessorKwargs]"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/dia/processing_dia.py#L85"}}),ze=new w({props:{name:"batch_decode",anchor:"transformers.DiaProcessor.batch_decode",parameters:[{name:"decoder_input_ids",val:": torch.Tensor"},{name:"audio_prompt_len",val:": typing.Optional[int] = None"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.models.dia.processing_dia.DiaProcessorKwargs]"}],parametersDescription:[{anchor:"transformers.DiaProcessor.batch_decode.decoder_input_ids",description:"<strong>decoder_input_ids</strong> (<code>torch.Tensor</code>) &#x2014; The complete output sequence of the decoder.",name:"decoder_input_ids"},{anchor:"transformers.DiaProcessor.batch_decode.audio_prompt_len",description:"<strong>audio_prompt_len</strong> (<code>int</code>) &#x2014; The audio prefix length (e.g. when using voice cloning).",name:"audio_prompt_len"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/dia/processing_dia.py#L258"}}),Ue=new w({props:{name:"decode",anchor:"transformers.DiaProcessor.decode",parameters:[{name:"decoder_input_ids",val:": torch.Tensor"},{name:"audio_prompt_len",val:": typing.Optional[int] = None"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.models.dia.processing_dia.DiaProcessorKwargs]"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/dia/processing_dia.py#L329"}}),qe=new $({props:{title:"DiaModel",local:"transformers.DiaModel",headingTag:"h2"}}),Je=new w({props:{name:"class transformers.DiaModel",anchor:"transformers.DiaModel",parameters:[{name:"config",val:": DiaConfig"}],parametersDescription:[{anchor:"transformers.DiaModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/dia#transformers.DiaConfig">DiaConfig</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/dia/modeling_dia.py#L719"}}),Ie=new w({props:{name:"forward",anchor:"transformers.DiaModel.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.LongTensor] = None"},{name:"decoder_input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"decoder_position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"decoder_attention_mask",val:": typing.Optional[torch.LongTensor] = None"},{name:"encoder_outputs",val:": typing.Union[transformers.modeling_outputs.BaseModelOutput, tuple, NoneType] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.EncoderDecoderCache] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.DiaModel.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.DiaModel.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.DiaModel.forward.decoder_input_ids",description:"<strong>decoder_input_ids</strong> (<code>torch.LongTensor</code> of shape `(batch_size * num_codebooks, target_sequence_length) &#x2014;",name:"decoder_input_ids"},{anchor:"transformers.DiaModel.forward.or",description:`<strong>or</strong> (batch_size, target_sequence_length, num_codebooks)\`, <em>optional</em>) &#x2014;</p>
<ol>
<li>
<p>(batch_size * num_codebooks, target_sequence_length): corresponds to the general use case where
the audio input codebooks are flattened into the batch dimension. This also aligns with the flat-
tened audio logits which are used to calculate the loss.</p>
</li>
<li>
<p>(batch_size, sequence_length, num_codebooks): corresponds to the internally used shape of
Dia to calculate embeddings and subsequent steps more efficiently.</p>
</li>
</ol>
<p>If no <code>decoder_input_ids</code> are provided, it will create a tensor of <code>bos_token_id</code> with shape
<code>(batch_size, 1, num_codebooks)</code>. Indices can be obtained using the <a href="/docs/transformers/v4.56.2/en/model_doc/dia#transformers.DiaProcessor">DiaProcessor</a>. See
<a href="/docs/transformers/v4.56.2/en/model_doc/dia#transformers.DiaProcessor.__call__">DiaProcessor.<strong>call</strong>()</a> for more details.</p>
<p><a href="../glossary#decoder-input-ids">What are decoder input IDs?</a>`,name:"or"},{anchor:"transformers.DiaModel.forward.decoder_position_ids",description:`<strong>decoder_position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, target_sequence_length)</code>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings.
Used to calculate the position embeddings up to <code>config.decoder_config.max_position_embeddings</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"decoder_position_ids"},{anchor:"transformers.DiaModel.forward.decoder_attention_mask",description:`<strong>decoder_attention_mask</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on certain token indices. By default, a causal mask will be used, to
make sure the model can only look at previous inputs in order to predict the future.`,name:"decoder_attention_mask"},{anchor:"transformers.DiaModel.forward.encoder_outputs",description:`<strong>encoder_outputs</strong> (<code>Union[~modeling_outputs.BaseModelOutput, tuple, NoneType]</code>) &#x2014;
Tuple consists of (<code>last_hidden_state</code>, <em>optional</em>: <code>hidden_states</code>, <em>optional</em>: <code>attentions</code>)
<code>last_hidden_state</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) is a sequence of
hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.`,name:"encoder_outputs"},{anchor:"transformers.DiaModel.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.EncoderDecoderCache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.DiaModel.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.DiaModel.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.DiaModel.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.DiaModel.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/dia/modeling_dia.py#L730",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqModelOutput"
>transformers.modeling_outputs.Seq2SeqModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<code>None</code>) and inputs.</p>
<ul>
<li>
<p><strong>last_hidden_state</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>) — Sequence of hidden-states at the output of the last layer of the decoder of the model.</p>
<p>If <code>past_key_values</code> is used only the last hidden-state of the sequences of shape <code>(batch_size, 1, hidden_size)</code> is output.</p>
</li>
<li>
<p><strong>past_key_values</strong> (<code>EncoderDecoderCache</code>, <em>optional</em>, returned when <code>use_cache=True</code> is passed or when <code>config.use_cache=True</code>) — It is a <a
  href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.EncoderDecoderCache"
>EncoderDecoderCache</a> instance. For more details, see our <a
  href="https://huggingface.co/docs/transformers/en/kv_cache"
  rel="nofollow"
>kv cache guide</a>.</p>
<p>Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used (see <code>past_key_values</code> input) to speed up sequential decoding.</p>
</li>
<li>
<p><strong>decoder_hidden_states</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings, if the model has an embedding layer, +
one for the output of each layer) of shape <code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the decoder at the output of each layer plus the optional initial embedding outputs.</p>
</li>
<li>
<p><strong>decoder_attentions</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
self-attention heads.</p>
</li>
<li>
<p><strong>cross_attentions</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the
weighted average in the cross-attention heads.</p>
</li>
<li>
<p><strong>encoder_last_hidden_state</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) — Sequence of hidden-states at the output of the last layer of the encoder of the model.</p>
</li>
<li>
<p><strong>encoder_hidden_states</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings, if the model has an embedding layer, +
one for the output of each layer) of shape <code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the encoder at the output of each layer plus the optional initial embedding outputs.</p>
</li>
<li>
<p><strong>encoder_attentions</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
self-attention heads.</p>
</li>
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqModelOutput"
>transformers.modeling_outputs.Seq2SeqModelOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),Y=new fn({props:{$$slots:{default:[Mn]},$$scope:{ctx:P}}}),Fe=new $({props:{title:"DiaForConditionalGeneration",local:"transformers.DiaForConditionalGeneration",headingTag:"h2"}}),Ee=new w({props:{name:"class transformers.DiaForConditionalGeneration",anchor:"transformers.DiaForConditionalGeneration",parameters:[{name:"config",val:": DiaConfig"}],parametersDescription:[{anchor:"transformers.DiaForConditionalGeneration.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/dia#transformers.DiaConfig">DiaConfig</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/dia/modeling_dia.py#L847"}}),Re=new w({props:{name:"forward",anchor:"transformers.DiaForConditionalGeneration.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.LongTensor] = None"},{name:"decoder_input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"decoder_position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"decoder_attention_mask",val:": typing.Optional[torch.LongTensor] = None"},{name:"encoder_outputs",val:": typing.Union[transformers.modeling_outputs.BaseModelOutput, tuple, NoneType] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.EncoderDecoderCache] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.DiaForConditionalGeneration.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.DiaForConditionalGeneration.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.DiaForConditionalGeneration.forward.decoder_input_ids",description:"<strong>decoder_input_ids</strong> (<code>torch.LongTensor</code> of shape `(batch_size * num_codebooks, target_sequence_length) &#x2014;",name:"decoder_input_ids"},{anchor:"transformers.DiaForConditionalGeneration.forward.or",description:`<strong>or</strong> (batch_size, target_sequence_length, num_codebooks)\`, <em>optional</em>) &#x2014;</p>
<ol>
<li>
<p>(batch_size * num_codebooks, target_sequence_length): corresponds to the general use case where
the audio input codebooks are flattened into the batch dimension. This also aligns with the flat-
tened audio logits which are used to calculate the loss.</p>
</li>
<li>
<p>(batch_size, sequence_length, num_codebooks): corresponds to the internally used shape of
Dia to calculate embeddings and subsequent steps more efficiently.</p>
</li>
</ol>
<p>If no <code>decoder_input_ids</code> are provided, it will create a tensor of <code>bos_token_id</code> with shape
<code>(batch_size, 1, num_codebooks)</code>. Indices can be obtained using the <a href="/docs/transformers/v4.56.2/en/model_doc/dia#transformers.DiaProcessor">DiaProcessor</a>. See
<a href="/docs/transformers/v4.56.2/en/model_doc/dia#transformers.DiaProcessor.__call__">DiaProcessor.<strong>call</strong>()</a> for more details.</p>
<p><a href="../glossary#decoder-input-ids">What are decoder input IDs?</a>`,name:"or"},{anchor:"transformers.DiaForConditionalGeneration.forward.decoder_position_ids",description:`<strong>decoder_position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, target_sequence_length)</code>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings.
Used to calculate the position embeddings up to <code>config.decoder_config.max_position_embeddings</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"decoder_position_ids"},{anchor:"transformers.DiaForConditionalGeneration.forward.decoder_attention_mask",description:`<strong>decoder_attention_mask</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on certain token indices. By default, a causal mask will be used, to
make sure the model can only look at previous inputs in order to predict the future.`,name:"decoder_attention_mask"},{anchor:"transformers.DiaForConditionalGeneration.forward.encoder_outputs",description:`<strong>encoder_outputs</strong> (<code>Union[~modeling_outputs.BaseModelOutput, tuple, NoneType]</code>) &#x2014;
Tuple consists of (<code>last_hidden_state</code>, <em>optional</em>: <code>hidden_states</code>, <em>optional</em>: <code>attentions</code>)
<code>last_hidden_state</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) is a sequence of
hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.`,name:"encoder_outputs"},{anchor:"transformers.DiaForConditionalGeneration.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.EncoderDecoderCache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.DiaForConditionalGeneration.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.DiaForConditionalGeneration.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.DiaForConditionalGeneration.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.DiaForConditionalGeneration.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size * num_codebooks,)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in
<code>[0, ..., config.decoder_config.vocab_size - 1]</code> or -100. Tokens with indices set to <code>-100</code>
are ignored (masked).`,name:"labels"},{anchor:"transformers.DiaForConditionalGeneration.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/dia/modeling_dia.py#L871",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqLMOutput"
>transformers.modeling_outputs.Seq2SeqLMOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<code>None</code>) and inputs.</p>
<ul>
<li>
<p><strong>loss</strong> (<code>torch.FloatTensor</code> of shape <code>(1,)</code>, <em>optional</em>, returned when <code>labels</code> is provided) — Language modeling loss.</p>
</li>
<li>
<p><strong>logits</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, config.vocab_size)</code>) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).</p>
</li>
<li>
<p><strong>past_key_values</strong> (<code>EncoderDecoderCache</code>, <em>optional</em>, returned when <code>use_cache=True</code> is passed or when <code>config.use_cache=True</code>) — It is a <a
  href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.EncoderDecoderCache"
>EncoderDecoderCache</a> instance. For more details, see our <a
  href="https://huggingface.co/docs/transformers/en/kv_cache"
  rel="nofollow"
>kv cache guide</a>.</p>
<p>Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used (see <code>past_key_values</code> input) to speed up sequential decoding.</p>
</li>
<li>
<p><strong>decoder_hidden_states</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings, if the model has an embedding layer, +
one for the output of each layer) of shape <code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.</p>
</li>
<li>
<p><strong>decoder_attentions</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
self-attention heads.</p>
</li>
<li>
<p><strong>cross_attentions</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the
weighted average in the cross-attention heads.</p>
</li>
<li>
<p><strong>encoder_last_hidden_state</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) — Sequence of hidden-states at the output of the last layer of the encoder of the model.</p>
</li>
<li>
<p><strong>encoder_hidden_states</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_hidden_states=True</code> is passed or when <code>config.output_hidden_states=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for the output of the embeddings, if the model has an embedding layer, +
one for the output of each layer) of shape <code>(batch_size, sequence_length, hidden_size)</code>.</p>
<p>Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.</p>
</li>
<li>
<p><strong>encoder_attentions</strong> (<code>tuple(torch.FloatTensor)</code>, <em>optional</em>, returned when <code>output_attentions=True</code> is passed or when <code>config.output_attentions=True</code>) — Tuple of <code>torch.FloatTensor</code> (one for each layer) of shape <code>(batch_size, num_heads, sequence_length, sequence_length)</code>.</p>
<p>Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
self-attention heads.</p>
</li>
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqLMOutput"
>transformers.modeling_outputs.Seq2SeqLMOutput</a> or <code>tuple(torch.FloatTensor)</code></p>
`}}),A=new fn({props:{$$slots:{default:[Dn]},$$scope:{ctx:P}}}),je=new w({props:{name:"generate",anchor:"transformers.DiaForConditionalGeneration.generate",parameters:[{name:"inputs",val:": typing.Optional[torch.Tensor] = None"},{name:"generation_config",val:": typing.Optional[transformers.generation.configuration_utils.GenerationConfig] = None"},{name:"logits_processor",val:": typing.Optional[transformers.generation.logits_process.LogitsProcessorList] = None"},{name:"stopping_criteria",val:": typing.Optional[transformers.generation.stopping_criteria.StoppingCriteriaList] = None"},{name:"prefix_allowed_tokens_fn",val:": typing.Optional[typing.Callable[[int, torch.Tensor], list[int]]] = None"},{name:"synced_gpus",val:": typing.Optional[bool] = None"},{name:"assistant_model",val:": typing.Optional[ForwardRef('PreTrainedModel')] = None"},{name:"streamer",val:": typing.Optional[ForwardRef('BaseStreamer')] = None"},{name:"negative_prompt_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"negative_prompt_attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"use_model_defaults",val:": typing.Optional[bool] = None"},{name:"custom_generate",val:": typing.Optional[str] = None"},{name:"**kwargs",val:""}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/dia/generation_dia.py#L406"}}),Ge=new xn({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/dia.md"}}),{c(){f=s("meta"),C=a(),_=s("p"),x=a(),J=s("p"),J.innerHTML=v,W=a(),c(te.$$.fragment),gt=a(),B=s("div"),B.innerHTML=Go,_t=a(),c(oe.$$.fragment),yt=a(),ne=s("p"),ne.innerHTML=No,bt=a(),ae=s("p"),ae.innerHTML=Zo,vt=a(),c(re.$$.fragment),Tt=a(),c(se.$$.fragment),wt=a(),c(ie.$$.fragment),xt=a(),c(de.$$.fragment),kt=a(),c(ce.$$.fragment),Mt=a(),c(le.$$.fragment),Dt=a(),c(me.$$.fragment),$t=a(),pe=s("p"),pe.innerHTML=Po,Ct=a(),c(he.$$.fragment),zt=a(),k=s("div"),c(ue.$$.fragment),Kt=a(),Ze=s("p"),Ze.innerHTML=Wo,eo=a(),Pe=s("p"),Pe.innerHTML=Bo,to=a(),c(L.$$.fragment),oo=a(),S=s("div"),c(fe.$$.fragment),no=a(),We=s("p"),We.textContent=Lo,Ut=a(),c(ge.$$.fragment),qt=a(),F=s("div"),c(_e.$$.fragment),ao=a(),Be=s("p"),Be.innerHTML=So,ro=a(),Le=s("p"),Le.innerHTML=Ho,Jt=a(),c(ye.$$.fragment),It=a(),E=s("div"),c(be.$$.fragment),so=a(),Se=s("p"),Se.innerHTML=Vo,io=a(),He=s("p"),He.innerHTML=Xo,Ft=a(),c(ve.$$.fragment),Et=a(),z=s("div"),c(Te.$$.fragment),co=a(),Ve=s("p"),Ve.innerHTML=Qo,lo=a(),Xe=s("p"),Xe.innerHTML=Oo,mo=a(),H=s("div"),c(we.$$.fragment),po=a(),Qe=s("p"),Qe.textContent=Yo,Rt=a(),c(xe.$$.fragment),jt=a(),U=s("div"),c(ke.$$.fragment),ho=a(),Oe=s("p"),Oe.textContent=Ao,uo=a(),Ye=s("p"),Ye.innerHTML=Ko,fo=a(),V=s("div"),c(Me.$$.fragment),go=a(),Ae=s("p"),Ae.textContent=en,Gt=a(),c(De.$$.fragment),Nt=a(),M=s("div"),c($e.$$.fragment),_o=a(),Ke=s("p"),Ke.innerHTML=tn,yo=a(),X=s("div"),c(Ce.$$.fragment),bo=a(),et=s("p"),et.innerHTML=on,vo=a(),Q=s("div"),c(ze.$$.fragment),To=a(),tt=s("p"),tt.innerHTML=nn,wo=a(),O=s("div"),c(Ue.$$.fragment),xo=a(),ot=s("p"),ot.innerHTML=an,Zt=a(),c(qe.$$.fragment),Pt=a(),D=s("div"),c(Je.$$.fragment),ko=a(),nt=s("p"),nt.textContent=rn,Mo=a(),at=s("p"),at.innerHTML=sn,Do=a(),rt=s("p"),rt.innerHTML=dn,$o=a(),R=s("div"),c(Ie.$$.fragment),Co=a(),st=s("p"),st.innerHTML=cn,zo=a(),c(Y.$$.fragment),Wt=a(),c(Fe.$$.fragment),Bt=a(),T=s("div"),c(Ee.$$.fragment),Uo=a(),it=s("p"),it.textContent=ln,qo=a(),dt=s("p"),dt.innerHTML=mn,Jo=a(),ct=s("p"),ct.innerHTML=pn,Io=a(),j=s("div"),c(Re.$$.fragment),Fo=a(),lt=s("p"),lt.innerHTML=hn,Eo=a(),c(A.$$.fragment),Ro=a(),mt=s("div"),c(je.$$.fragment),Lt=a(),c(Ge.$$.fragment),St=a(),ft=s("p"),this.h()},l(e){const t=vn("svelte-u9bgzb",document.head);f=i(t,"META",{name:!0,content:!0}),t.forEach(o),C=r(e),_=i(e,"P",{}),y(_).forEach(o),x=r(e),J=i(e,"P",{"data-svelte-h":!0}),g(J)!=="svelte-h0dmab"&&(J.innerHTML=v),W=r(e),l(te.$$.fragment,e),gt=r(e),B=i(e,"DIV",{style:!0,"data-svelte-h":!0}),g(B)!=="svelte-2m0t7r"&&(B.innerHTML=Go),_t=r(e),l(oe.$$.fragment,e),yt=r(e),ne=i(e,"P",{"data-svelte-h":!0}),g(ne)!=="svelte-1ol0lhz"&&(ne.innerHTML=No),bt=r(e),ae=i(e,"P",{"data-svelte-h":!0}),g(ae)!=="svelte-n65lpn"&&(ae.innerHTML=Zo),vt=r(e),l(re.$$.fragment,e),Tt=r(e),l(se.$$.fragment,e),wt=r(e),l(ie.$$.fragment,e),xt=r(e),l(de.$$.fragment,e),kt=r(e),l(ce.$$.fragment,e),Mt=r(e),l(le.$$.fragment,e),Dt=r(e),l(me.$$.fragment,e),$t=r(e),pe=i(e,"P",{"data-svelte-h":!0}),g(pe)!=="svelte-zqpp4w"&&(pe.innerHTML=Po),Ct=r(e),l(he.$$.fragment,e),zt=r(e),k=i(e,"DIV",{class:!0});var q=y(k);l(ue.$$.fragment,q),Kt=r(q),Ze=i(q,"P",{"data-svelte-h":!0}),g(Ze)!=="svelte-1g5ue0f"&&(Ze.innerHTML=Wo),eo=r(q),Pe=i(q,"P",{"data-svelte-h":!0}),g(Pe)!=="svelte-1ek1ss9"&&(Pe.innerHTML=Bo),to=r(q),l(L.$$.fragment,q),oo=r(q),S=i(q,"DIV",{class:!0});var Ne=y(S);l(fe.$$.fragment,Ne),no=r(Ne),We=i(Ne,"P",{"data-svelte-h":!0}),g(We)!=="svelte-u1gtai"&&(We.textContent=Lo),Ne.forEach(o),q.forEach(o),Ut=r(e),l(ge.$$.fragment,e),qt=r(e),F=i(e,"DIV",{class:!0});var Z=y(F);l(_e.$$.fragment,Z),ao=r(Z),Be=i(Z,"P",{"data-svelte-h":!0}),g(Be)!=="svelte-wmsndj"&&(Be.innerHTML=So),ro=r(Z),Le=i(Z,"P",{"data-svelte-h":!0}),g(Le)!=="svelte-1ek1ss9"&&(Le.innerHTML=Ho),Z.forEach(o),Jt=r(e),l(ye.$$.fragment,e),It=r(e),E=i(e,"DIV",{class:!0});var pt=y(E);l(be.$$.fragment,pt),so=r(pt),Se=i(pt,"P",{"data-svelte-h":!0}),g(Se)!=="svelte-1u444qz"&&(Se.innerHTML=Vo),io=r(pt),He=i(pt,"P",{"data-svelte-h":!0}),g(He)!=="svelte-1ek1ss9"&&(He.innerHTML=Xo),pt.forEach(o),Ft=r(e),l(ve.$$.fragment,e),Et=r(e),z=i(e,"DIV",{class:!0});var K=y(z);l(Te.$$.fragment,K),co=r(K),Ve=i(K,"P",{"data-svelte-h":!0}),g(Ve)!=="svelte-1195mn2"&&(Ve.innerHTML=Qo),lo=r(K),Xe=i(K,"P",{"data-svelte-h":!0}),g(Xe)!=="svelte-gxzj9w"&&(Xe.innerHTML=Oo),mo=r(K),H=i(K,"DIV",{class:!0});var Vt=y(H);l(we.$$.fragment,Vt),po=r(Vt),Qe=i(Vt,"P",{"data-svelte-h":!0}),g(Qe)!=="svelte-kpxj0c"&&(Qe.textContent=Yo),Vt.forEach(o),K.forEach(o),Rt=r(e),l(xe.$$.fragment,e),jt=r(e),U=i(e,"DIV",{class:!0});var ee=y(U);l(ke.$$.fragment,ee),ho=r(ee),Oe=i(ee,"P",{"data-svelte-h":!0}),g(Oe)!=="svelte-9n5l0n"&&(Oe.textContent=Ao),uo=r(ee),Ye=i(ee,"P",{"data-svelte-h":!0}),g(Ye)!=="svelte-ue5gbv"&&(Ye.innerHTML=Ko),fo=r(ee),V=i(ee,"DIV",{class:!0});var Xt=y(V);l(Me.$$.fragment,Xt),go=r(Xt),Ae=i(Xt,"P",{"data-svelte-h":!0}),g(Ae)!=="svelte-1a6wgfx"&&(Ae.textContent=en),Xt.forEach(o),ee.forEach(o),Gt=r(e),l(De.$$.fragment,e),Nt=r(e),M=i(e,"DIV",{class:!0});var G=y(M);l($e.$$.fragment,G),_o=r(G),Ke=i(G,"P",{"data-svelte-h":!0}),g(Ke)!=="svelte-1wyh2uo"&&(Ke.innerHTML=tn),yo=r(G),X=i(G,"DIV",{class:!0});var Qt=y(X);l(Ce.$$.fragment,Qt),bo=r(Qt),et=i(Qt,"P",{"data-svelte-h":!0}),g(et)!=="svelte-167mwnb"&&(et.innerHTML=on),Qt.forEach(o),vo=r(G),Q=i(G,"DIV",{class:!0});var Ot=y(Q);l(ze.$$.fragment,Ot),To=r(Ot),tt=i(Ot,"P",{"data-svelte-h":!0}),g(tt)!=="svelte-1gad9d9"&&(tt.innerHTML=nn),Ot.forEach(o),wo=r(G),O=i(G,"DIV",{class:!0});var Yt=y(O);l(Ue.$$.fragment,Yt),xo=r(Yt),ot=i(Yt,"P",{"data-svelte-h":!0}),g(ot)!=="svelte-rwq2t0"&&(ot.innerHTML=an),Yt.forEach(o),G.forEach(o),Zt=r(e),l(qe.$$.fragment,e),Pt=r(e),D=i(e,"DIV",{class:!0});var N=y(D);l(Je.$$.fragment,N),ko=r(N),nt=i(N,"P",{"data-svelte-h":!0}),g(nt)!=="svelte-rt9ofc"&&(nt.textContent=rn),Mo=r(N),at=i(N,"P",{"data-svelte-h":!0}),g(at)!=="svelte-q52n56"&&(at.innerHTML=sn),Do=r(N),rt=i(N,"P",{"data-svelte-h":!0}),g(rt)!=="svelte-hswkmf"&&(rt.innerHTML=dn),$o=r(N),R=i(N,"DIV",{class:!0});var ht=y(R);l(Ie.$$.fragment,ht),Co=r(ht),st=i(ht,"P",{"data-svelte-h":!0}),g(st)!=="svelte-5r9zf7"&&(st.innerHTML=cn),zo=r(ht),l(Y.$$.fragment,ht),ht.forEach(o),N.forEach(o),Wt=r(e),l(Fe.$$.fragment,e),Bt=r(e),T=i(e,"DIV",{class:!0});var I=y(T);l(Ee.$$.fragment,I),Uo=r(I),it=i(I,"P",{"data-svelte-h":!0}),g(it)!=="svelte-14w8l0g"&&(it.textContent=ln),qo=r(I),dt=i(I,"P",{"data-svelte-h":!0}),g(dt)!=="svelte-q52n56"&&(dt.innerHTML=mn),Jo=r(I),ct=i(I,"P",{"data-svelte-h":!0}),g(ct)!=="svelte-hswkmf"&&(ct.innerHTML=pn),Io=r(I),j=i(I,"DIV",{class:!0});var ut=y(j);l(Re.$$.fragment,ut),Fo=r(ut),lt=i(ut,"P",{"data-svelte-h":!0}),g(lt)!=="svelte-1mzybyh"&&(lt.innerHTML=hn),Eo=r(ut),l(A.$$.fragment,ut),ut.forEach(o),Ro=r(I),mt=i(I,"DIV",{class:!0});var un=y(mt);l(je.$$.fragment,un),un.forEach(o),I.forEach(o),Lt=r(e),l(Ge.$$.fragment,e),St=r(e),ft=i(e,"P",{}),y(ft).forEach(o),this.h()},h(){b(f,"name","hf:doc:metadata"),b(f,"content",Cn),Tn(B,"float","right"),b(S,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),b(k,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),b(F,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),b(E,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),b(H,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),b(z,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),b(V,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),b(U,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),b(X,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),b(Q,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),b(O,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),b(M,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),b(R,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),b(D,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),b(j,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),b(mt,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),b(T,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,t){n(document.head,f),d(e,C,t),d(e,_,t),d(e,x,t),d(e,J,t),d(e,W,t),m(te,e,t),d(e,gt,t),d(e,B,t),d(e,_t,t),m(oe,e,t),d(e,yt,t),d(e,ne,t),d(e,bt,t),d(e,ae,t),d(e,vt,t),m(re,e,t),d(e,Tt,t),m(se,e,t),d(e,wt,t),m(ie,e,t),d(e,xt,t),m(de,e,t),d(e,kt,t),m(ce,e,t),d(e,Mt,t),m(le,e,t),d(e,Dt,t),m(me,e,t),d(e,$t,t),d(e,pe,t),d(e,Ct,t),m(he,e,t),d(e,zt,t),d(e,k,t),m(ue,k,null),n(k,Kt),n(k,Ze),n(k,eo),n(k,Pe),n(k,to),m(L,k,null),n(k,oo),n(k,S),m(fe,S,null),n(S,no),n(S,We),d(e,Ut,t),m(ge,e,t),d(e,qt,t),d(e,F,t),m(_e,F,null),n(F,ao),n(F,Be),n(F,ro),n(F,Le),d(e,Jt,t),m(ye,e,t),d(e,It,t),d(e,E,t),m(be,E,null),n(E,so),n(E,Se),n(E,io),n(E,He),d(e,Ft,t),m(ve,e,t),d(e,Et,t),d(e,z,t),m(Te,z,null),n(z,co),n(z,Ve),n(z,lo),n(z,Xe),n(z,mo),n(z,H),m(we,H,null),n(H,po),n(H,Qe),d(e,Rt,t),m(xe,e,t),d(e,jt,t),d(e,U,t),m(ke,U,null),n(U,ho),n(U,Oe),n(U,uo),n(U,Ye),n(U,fo),n(U,V),m(Me,V,null),n(V,go),n(V,Ae),d(e,Gt,t),m(De,e,t),d(e,Nt,t),d(e,M,t),m($e,M,null),n(M,_o),n(M,Ke),n(M,yo),n(M,X),m(Ce,X,null),n(X,bo),n(X,et),n(M,vo),n(M,Q),m(ze,Q,null),n(Q,To),n(Q,tt),n(M,wo),n(M,O),m(Ue,O,null),n(O,xo),n(O,ot),d(e,Zt,t),m(qe,e,t),d(e,Pt,t),d(e,D,t),m(Je,D,null),n(D,ko),n(D,nt),n(D,Mo),n(D,at),n(D,Do),n(D,rt),n(D,$o),n(D,R),m(Ie,R,null),n(R,Co),n(R,st),n(R,zo),m(Y,R,null),d(e,Wt,t),m(Fe,e,t),d(e,Bt,t),d(e,T,t),m(Ee,T,null),n(T,Uo),n(T,it),n(T,qo),n(T,dt),n(T,Jo),n(T,ct),n(T,Io),n(T,j),m(Re,j,null),n(j,Fo),n(j,lt),n(j,Eo),m(A,j,null),n(T,Ro),n(T,mt),m(je,mt,null),d(e,Lt,t),m(Ge,e,t),d(e,St,t),d(e,ft,t),Ht=!0},p(e,[t]){const q={};t&2&&(q.$$scope={dirty:t,ctx:e}),L.$set(q);const Ne={};t&2&&(Ne.$$scope={dirty:t,ctx:e}),Y.$set(Ne);const Z={};t&2&&(Z.$$scope={dirty:t,ctx:e}),A.$set(Z)},i(e){Ht||(p(te.$$.fragment,e),p(oe.$$.fragment,e),p(re.$$.fragment,e),p(se.$$.fragment,e),p(ie.$$.fragment,e),p(de.$$.fragment,e),p(ce.$$.fragment,e),p(le.$$.fragment,e),p(me.$$.fragment,e),p(he.$$.fragment,e),p(ue.$$.fragment,e),p(L.$$.fragment,e),p(fe.$$.fragment,e),p(ge.$$.fragment,e),p(_e.$$.fragment,e),p(ye.$$.fragment,e),p(be.$$.fragment,e),p(ve.$$.fragment,e),p(Te.$$.fragment,e),p(we.$$.fragment,e),p(xe.$$.fragment,e),p(ke.$$.fragment,e),p(Me.$$.fragment,e),p(De.$$.fragment,e),p($e.$$.fragment,e),p(Ce.$$.fragment,e),p(ze.$$.fragment,e),p(Ue.$$.fragment,e),p(qe.$$.fragment,e),p(Je.$$.fragment,e),p(Ie.$$.fragment,e),p(Y.$$.fragment,e),p(Fe.$$.fragment,e),p(Ee.$$.fragment,e),p(Re.$$.fragment,e),p(A.$$.fragment,e),p(je.$$.fragment,e),p(Ge.$$.fragment,e),Ht=!0)},o(e){h(te.$$.fragment,e),h(oe.$$.fragment,e),h(re.$$.fragment,e),h(se.$$.fragment,e),h(ie.$$.fragment,e),h(de.$$.fragment,e),h(ce.$$.fragment,e),h(le.$$.fragment,e),h(me.$$.fragment,e),h(he.$$.fragment,e),h(ue.$$.fragment,e),h(L.$$.fragment,e),h(fe.$$.fragment,e),h(ge.$$.fragment,e),h(_e.$$.fragment,e),h(ye.$$.fragment,e),h(be.$$.fragment,e),h(ve.$$.fragment,e),h(Te.$$.fragment,e),h(we.$$.fragment,e),h(xe.$$.fragment,e),h(ke.$$.fragment,e),h(Me.$$.fragment,e),h(De.$$.fragment,e),h($e.$$.fragment,e),h(Ce.$$.fragment,e),h(ze.$$.fragment,e),h(Ue.$$.fragment,e),h(qe.$$.fragment,e),h(Je.$$.fragment,e),h(Ie.$$.fragment,e),h(Y.$$.fragment,e),h(Fe.$$.fragment,e),h(Ee.$$.fragment,e),h(Re.$$.fragment,e),h(A.$$.fragment,e),h(je.$$.fragment,e),h(Ge.$$.fragment,e),Ht=!1},d(e){e&&(o(C),o(_),o(x),o(J),o(W),o(gt),o(B),o(_t),o(yt),o(ne),o(bt),o(ae),o(vt),o(Tt),o(wt),o(xt),o(kt),o(Mt),o(Dt),o($t),o(pe),o(Ct),o(zt),o(k),o(Ut),o(qt),o(F),o(Jt),o(It),o(E),o(Ft),o(Et),o(z),o(Rt),o(jt),o(U),o(Gt),o(Nt),o(M),o(Zt),o(Pt),o(D),o(Wt),o(Bt),o(T),o(Lt),o(St),o(ft)),o(f),u(te,e),u(oe,e),u(re,e),u(se,e),u(ie,e),u(de,e),u(ce,e),u(le,e),u(me,e),u(he,e),u(ue),u(L),u(fe),u(ge,e),u(_e),u(ye,e),u(be),u(ve,e),u(Te),u(we),u(xe,e),u(ke),u(Me),u(De,e),u($e),u(Ce),u(ze),u(Ue),u(qe,e),u(Je),u(Ie),u(Y),u(Fe,e),u(Ee),u(Re),u(A),u(je),u(Ge,e)}}}const Cn='{"title":"Dia","local":"dia","sections":[{"title":"Overview","local":"overview","sections":[],"depth":2},{"title":"Usage Tips","local":"usage-tips","sections":[{"title":"Generation with Text","local":"generation-with-text","sections":[],"depth":3},{"title":"Generation with Text and Audio (Voice Cloning)","local":"generation-with-text-and-audio-voice-cloning","sections":[],"depth":3},{"title":"Training","local":"training","sections":[],"depth":3}],"depth":2},{"title":"DiaConfig","local":"transformers.DiaConfig","sections":[],"depth":2},{"title":"DiaDecoderConfig","local":"transformers.DiaDecoderConfig","sections":[],"depth":2},{"title":"DiaEncoderConfig","local":"transformers.DiaEncoderConfig","sections":[],"depth":2},{"title":"DiaTokenizer","local":"transformers.DiaTokenizer","sections":[],"depth":2},{"title":"DiaFeatureExtractor","local":"transformers.DiaFeatureExtractor","sections":[],"depth":2},{"title":"DiaProcessor","local":"transformers.DiaProcessor","sections":[],"depth":2},{"title":"DiaModel","local":"transformers.DiaModel","sections":[],"depth":2},{"title":"DiaForConditionalGeneration","local":"transformers.DiaForConditionalGeneration","sections":[],"depth":2}],"depth":1}';function zn(P){return _n(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class jn extends yn{constructor(f){super(),bn(this,f,zn,$n,gn,{})}}export{jn as component};
