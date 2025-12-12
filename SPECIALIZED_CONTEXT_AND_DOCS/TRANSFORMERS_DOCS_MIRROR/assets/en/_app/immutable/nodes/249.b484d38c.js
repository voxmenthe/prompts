import{s as oo,o as no,n as ct}from"../chunks/scheduler.18a86fab.js";import{S as ao,i as so,g as l,s as n,r as p,A as ro,h as c,f as o,c as a,j as F,x as y,u as h,k as U,y as s,a as r,v as m,d as u,t as f,w as g}from"../chunks/index.98837b22.js";import{T as eo}from"../chunks/Tip.77304350.js";import{D as B}from"../chunks/Docstring.a1ef7999.js";import{C as lt}from"../chunks/CodeBlock.8d0c2e8a.js";import{E as to}from"../chunks/ExampleCodeBlock.8c3ee1f9.js";import{H as z,E as io}from"../chunks/getInferenceSnippets.06c2775f.js";function lo(G){let i,M="Example:",_,T,b;return T=new lt({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEt5dXRhaVNwZWVjaFRvVGV4dENvbmZpZyUyQyUyMEt5dXRhaVNwZWVjaFRvVGV4dEZvckNvbmRpdGlvbmFsR2VuZXJhdGlvbiUwQSUwQSUyMyUyMEluaXRpYWxpemluZyUyMGElMjBLeXV0YWlTcGVlY2hUb1RleHRDb25maWclMEFjb25maWd1cmF0aW9uJTIwJTNEJTIwS3l1dGFpU3BlZWNoVG9UZXh0Q29uZmlnKCklMEElMEElMjMlMjBJbml0aWFsaXppbmclMjBhJTIwbW9kZWwlMEFtb2RlbCUyMCUzRCUyMEt5dXRhaVNwZWVjaFRvVGV4dEZvckNvbmRpdGlvbmFsR2VuZXJhdGlvbihjb25maWd1cmF0aW9uKSUwQSUwQSUyMyUyMEFjY2Vzc2luZyUyMHRoZSUyMG1vZGVsJTIwY29uZmlndXJhdGlvbiUwQWNvbmZpZ3VyYXRpb24lMjAlM0QlMjBtb2RlbC5jb25maWc=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> KyutaiSpeechToTextConfig, KyutaiSpeechToTextForConditionalGeneration

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a KyutaiSpeechToTextConfig</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = KyutaiSpeechToTextConfig()

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a model</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = KyutaiSpeechToTextForConditionalGeneration(configuration)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Accessing the model configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = model.config`,wrap:!1}}),{c(){i=l("p"),i.textContent=M,_=n(),p(T.$$.fragment)},l(d){i=c(d,"P",{"data-svelte-h":!0}),y(i)!=="svelte-11lpom8"&&(i.textContent=M),_=a(d),h(T.$$.fragment,d)},m(d,$){r(d,i,$),r(d,_,$),m(T,d,$),b=!0},p:ct,i(d){b||(u(T.$$.fragment,d),b=!0)},o(d){f(T.$$.fragment,d),b=!1},d(d){d&&(o(i),o(_)),g(T,d)}}}function co(G){let i,M=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){i=l("p"),i.innerHTML=M},l(_){i=c(_,"P",{"data-svelte-h":!0}),y(i)!=="svelte-fincs2"&&(i.innerHTML=M)},m(_,T){r(_,i,T)},p:ct,d(_){_&&o(i)}}}function po(G){let i,M="Example:",_,T,b;return T=new lt({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwZGF0YXNldHMlMjBpbXBvcnQlMjBsb2FkX2RhdGFzZXQlMkMlMjBBdWRpbyUwQWZyb20lMjB0cmFuc2Zvcm1lcnMlMjBpbXBvcnQlMjBLeXV0YWlTcGVlY2hUb1RleHRQcm9jZXNzb3IlMkMlMjBLeXV0YWlTcGVlY2hUb1RleHRGb3JDb25kaXRpb25hbEdlbmVyYXRpb24lMEElMEF0b3JjaF9kZXZpY2UlMjAlM0QlMjAlMjJjdWRhJTIyJTIwaWYlMjB0b3JjaC5jdWRhLmlzX2F2YWlsYWJsZSgpJTIwZWxzZSUyMCUyMmNwdSUyMiUwQW1vZGVsX2lkJTIwJTNEJTIwJTIya3l1dGFpJTJGc3R0LTIuNmItZW4tdHJmcyUyMiUwQSUwQXByb2Nlc3NvciUyMCUzRCUyMEt5dXRhaVNwZWVjaFRvVGV4dFByb2Nlc3Nvci5mcm9tX3ByZXRyYWluZWQobW9kZWxfaWQpJTBBbW9kZWwlMjAlM0QlMjBLeXV0YWlTcGVlY2hUb1RleHRGb3JDb25kaXRpb25hbEdlbmVyYXRpb24uZnJvbV9wcmV0cmFpbmVkKG1vZGVsX2lkJTJDJTIwZGV2aWNlX21hcCUzRHRvcmNoX2RldmljZSklMEElMEFkcyUyMCUzRCUyMGxvYWRfZGF0YXNldCglMEElMjAlMjAlMjAlMjAlMjJoZi1pbnRlcm5hbC10ZXN0aW5nJTJGbGlicmlzcGVlY2hfYXNyX2R1bW15JTIyJTJDJTIwJTIyY2xlYW4lMjIlMkMlMjBzcGxpdCUzRCUyMnZhbGlkYXRpb24lMjIlMEEpJTBBJTBBZHMlMjAlM0QlMjBkcy5jYXN0X2NvbHVtbiglMjJhdWRpbyUyMiUyQyUyMEF1ZGlvKHNhbXBsaW5nX3JhdGUlM0QyNDAwMCkpJTBBaW5wdXRzJTIwJTNEJTIwcHJvY2Vzc29yKCUwQSUyMCUyMCUyMCUyMGRzJTVCMCU1RCU1QiUyMmF1ZGlvJTIyJTVEJTVCJTIyYXJyYXklMjIlNUQlMkMlMEEpJTBBaW5wdXRzLnRvKHRvcmNoX2RldmljZSklMEElMEFvdXRwdXRfdG9rZW5zJTIwJTNEJTIwbW9kZWwuZ2VuZXJhdGUoKippbnB1dHMpJTBBcHJpbnQocHJvY2Vzc29yLmJhdGNoX2RlY29kZShvdXRwdXRfdG9rZW5zJTJDJTIwc2tpcF9zcGVjaWFsX3Rva2VucyUzRFRydWUpKQ==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> datasets <span class="hljs-keyword">import</span> load_dataset, Audio
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> KyutaiSpeechToTextProcessor, KyutaiSpeechToTextForConditionalGeneration

<span class="hljs-meta">&gt;&gt;&gt; </span>torch_device = <span class="hljs-string">&quot;cuda&quot;</span> <span class="hljs-keyword">if</span> torch.cuda.is_available() <span class="hljs-keyword">else</span> <span class="hljs-string">&quot;cpu&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model_id = <span class="hljs-string">&quot;kyutai/stt-2.6b-en-trfs&quot;</span>

<span class="hljs-meta">&gt;&gt;&gt; </span>processor = KyutaiSpeechToTextProcessor.from_pretrained(model_id)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = KyutaiSpeechToTextForConditionalGeneration.from_pretrained(model_id, device_map=torch_device)

<span class="hljs-meta">&gt;&gt;&gt; </span>ds = load_dataset(
<span class="hljs-meta">... </span>    <span class="hljs-string">&quot;hf-internal-testing/librispeech_asr_dummy&quot;</span>, <span class="hljs-string">&quot;clean&quot;</span>, split=<span class="hljs-string">&quot;validation&quot;</span>
<span class="hljs-meta">... </span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>ds = ds.cast_column(<span class="hljs-string">&quot;audio&quot;</span>, Audio(sampling_rate=<span class="hljs-number">24000</span>))
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = processor(
<span class="hljs-meta">... </span>    ds[<span class="hljs-number">0</span>][<span class="hljs-string">&quot;audio&quot;</span>][<span class="hljs-string">&quot;array&quot;</span>],
<span class="hljs-meta">... </span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>inputs.to(torch_device)

<span class="hljs-meta">&gt;&gt;&gt; </span>output_tokens = model.generate(**inputs)
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-built_in">print</span>(processor.batch_decode(output_tokens, skip_special_tokens=<span class="hljs-literal">True</span>))`,wrap:!1}}),{c(){i=l("p"),i.textContent=M,_=n(),p(T.$$.fragment)},l(d){i=c(d,"P",{"data-svelte-h":!0}),y(i)!=="svelte-11lpom8"&&(i.textContent=M),_=a(d),h(T.$$.fragment,d)},m(d,$){r(d,i,$),r(d,_,$),m(T,d,$),b=!0},p:ct,i(d){b||(u(T.$$.fragment,d),b=!0)},o(d){f(T.$$.fragment,d),b=!1},d(d){d&&(o(i),o(_)),g(T,d)}}}function ho(G){let i,M=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){i=l("p"),i.innerHTML=M},l(_){i=c(_,"P",{"data-svelte-h":!0}),y(i)!=="svelte-fincs2"&&(i.innerHTML=M)},m(_,T){r(_,i,T)},p:ct,d(_){_&&o(i)}}}function mo(G){let i,M,_,T,b,d="<em>This model was released on 2025-06-17 and added to Hugging Face Transformers on 2025-06-25.</em>",$,P,We,Q,Be,Y,Rt='<a href="https://kyutai.org/next/stt" rel="nofollow">Kyutai STT</a> is a speech-to-text model architecture based on the <a href="https://huggingface.co/docs/transformers/en/model_doc/mimi" rel="nofollow">Mimi codec</a>, which encodes audio into discrete tokens in a streaming fashion, and a <a href="https://huggingface.co/docs/transformers/en/model_doc/moshi" rel="nofollow">Moshi-like</a> autoregressive decoder. Kyutai’s lab has released two model checkpoints:',Ie,D,Zt='<li><a href="https://huggingface.co/kyutai/stt-1b-en_fr" rel="nofollow">kyutai/stt-1b-en_fr</a>: a 1B-parameter model capable of transcribing both English and French</li> <li><a href="https://huggingface.co/kyutai/stt-2.6b-en" rel="nofollow">kyutai/stt-2.6b-en</a>: a 2.6B-parameter model focused solely on English, optimized for maximum transcription accuracy</li>',Ve,I,zt='<img src="https://huggingface.co/datasets/eustlb/documentation-images/resolve/main/kyutai_stt.png"/>',Xe,A,Ee,O,Ne,ee,He,te,qe,oe,Le,ne,Kt=`This model was contributed by <a href="https://huggingface.co/eustlb" rel="nofollow">Eustache Le Bihan</a>.
The original code can be found <a href="https://github.com/kyutai-labs/moshi" rel="nofollow">here</a>.`,Pe,ae,Qe,k,se,dt,be,Wt=`This is the configuration class to store the configuration of a <a href="/docs/transformers/v4.56.2/en/model_doc/kyutai_speech_to_text#transformers.KyutaiSpeechToTextForConditionalGeneration">KyutaiSpeechToTextForConditionalGeneration</a>.
It is used to instantiate a Kyutai Speech-to-Text model according to the specified arguments, defining the model
architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the
2.6b-en model.`,pt,ve,Bt='e.g. <a href="https://huggingface.co/kyutai/stt-2.6b-en-trfs" rel="nofollow">kyutai/stt-2.6b-en-trfs</a>',ht,Me,It=`Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> and can be used to control the model outputs. Read the
documentation from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> for more information.`,mt,V,Ye,re,De,S,ie,ut,ke,Vt=`Constructs a Moshi ASR processor which wraps <a href="/docs/transformers/v4.56.2/en/model_doc/encodec#transformers.EncodecFeatureExtractor">EncodecFeatureExtractor</a> and
<a href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast">PreTrainedTokenizerFast</a> into a single processor that inherits both the audio feature extraction and
tokenizer functionalities. See the <a href="/docs/transformers/v4.56.2/en/model_doc/kyutai_speech_to_text#transformers.KyutaiSpeechToTextProcessor.__call__"><strong>call</strong>()</a> for more
information.`,ft,X,le,gt,xe,Xt=`Main method to prepare audio to be fed as input to the model. This method forwards the <code>audio</code>
arguments to KyutaiSpeechToTextFeatureExtractor’s <code>__call__()</code>. Please refer
to the docstring of the above method for more information.`,Ae,ce,Oe,J,de,_t,we,Et="Constructs an KyutaiSpeechToText feature extractor.",yt,$e,Nt=`This feature extractor inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/feature_extractor#transformers.SequenceFeatureExtractor">SequenceFeatureExtractor</a> which contains
most of the main methods. Users should refer to this superclass for more information regarding those methods.`,et,pe,tt,v,he,Tt,je,Ht="The Kyutai Speech To Text Model for token generation conditioned on other modalities (e.g. image-text-to-text generation).",bt,Ce,qt=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,vt,Ue,Lt=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Mt,j,me,kt,Se,Pt='The <a href="/docs/transformers/v4.56.2/en/model_doc/kyutai_speech_to_text#transformers.KyutaiSpeechToTextForConditionalGeneration">KyutaiSpeechToTextForConditionalGeneration</a> forward method, overrides the <code>__call__</code> special method.',xt,E,wt,N,$t,H,ue,jt,Je,Qt='This method forwards all its arguments to GenerationMixin’s <a href="/docs/transformers/v4.56.2/en/main_classes/text_generation#transformers.GenerationMixin.generate">generate()</a>. Please refer to the docstring of this method for more information.',ot,fe,nt,x,ge,Ct,Fe,Yt="The bare Kyutai Speech To Text Text Model outputting raw hidden-states without any specific head on to.",Ut,Ge,Dt=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,St,Re,At=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Jt,R,_e,Ft,Ze,Ot='The <a href="/docs/transformers/v4.56.2/en/model_doc/kyutai_speech_to_text#transformers.KyutaiSpeechToTextModel">KyutaiSpeechToTextModel</a> forward method, overrides the <code>__call__</code> special method.',Gt,q,at,ye,st,Ke,rt;return P=new z({props:{title:"Kyutai Speech-To-Text",local:"kyutai-speech-to-text",headingTag:"h1"}}),Q=new z({props:{title:"Overview",local:"overview",headingTag:"h2"}}),A=new z({props:{title:"Usage Tips",local:"usage-tips",headingTag:"h2"}}),O=new z({props:{title:"Inference",local:"inference",headingTag:"h3"}}),ee=new lt({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwZGF0YXNldHMlMjBpbXBvcnQlMjBsb2FkX2RhdGFzZXQlMkMlMjBBdWRpbyUwQWZyb20lMjB0cmFuc2Zvcm1lcnMlMjBpbXBvcnQlMjBpbmZlcl9kZXZpY2UlMkMlMjBLeXV0YWlTcGVlY2hUb1RleHRQcm9jZXNzb3IlMkMlMjBLeXV0YWlTcGVlY2hUb1RleHRGb3JDb25kaXRpb25hbEdlbmVyYXRpb24lMEElMEElMjMlMjAxLiUyMGxvYWQlMjB0aGUlMjBtb2RlbCUyMGFuZCUyMHRoZSUyMHByb2Nlc3NvciUwQXRvcmNoX2RldmljZSUyMCUzRCUyMGluZmVyX2RldmljZSgpJTBBbW9kZWxfaWQlMjAlM0QlMjAlMjJreXV0YWklMkZzdHQtMi42Yi1lbi10cmZzJTIyJTBBJTBBcHJvY2Vzc29yJTIwJTNEJTIwS3l1dGFpU3BlZWNoVG9UZXh0UHJvY2Vzc29yLmZyb21fcHJldHJhaW5lZChtb2RlbF9pZCklMEFtb2RlbCUyMCUzRCUyMEt5dXRhaVNwZWVjaFRvVGV4dEZvckNvbmRpdGlvbmFsR2VuZXJhdGlvbi5mcm9tX3ByZXRyYWluZWQobW9kZWxfaWQlMkMlMjBkZXZpY2VfbWFwJTNEdG9yY2hfZGV2aWNlJTJDJTIwZHR5cGUlM0QlMjJhdXRvJTIyKSUwQSUwQSUyMyUyMDIuJTIwbG9hZCUyMGF1ZGlvJTIwc2FtcGxlcyUwQWRzJTIwJTNEJTIwbG9hZF9kYXRhc2V0KCUwQSUyMCUyMCUyMCUyMCUyMmhmLWludGVybmFsLXRlc3RpbmclMkZsaWJyaXNwZWVjaF9hc3JfZHVtbXklMjIlMkMlMjAlMjJjbGVhbiUyMiUyQyUyMHNwbGl0JTNEJTIydmFsaWRhdGlvbiUyMiUwQSklMEFkcyUyMCUzRCUyMGRzLmNhc3RfY29sdW1uKCUyMmF1ZGlvJTIyJTJDJTIwQXVkaW8oc2FtcGxpbmdfcmF0ZSUzRDI0MDAwKSklMEElMEElMjMlMjAzLiUyMHByZXBhcmUlMjB0aGUlMjBtb2RlbCUyMGlucHV0cyUwQWlucHV0cyUyMCUzRCUyMHByb2Nlc3NvciglMEElMjAlMjAlMjAlMjBkcyU1QjAlNUQlNUIlMjJhdWRpbyUyMiU1RCU1QiUyMmFycmF5JTIyJTVEJTJDJTBBKSUwQWlucHV0cy50byhtb2RlbC5kZXZpY2UpJTBBJTBBJTIzJTIwNC4lMjBpbmZlciUyMHRoZSUyMG1vZGVsJTBBb3V0cHV0X3Rva2VucyUyMCUzRCUyMG1vZGVsLmdlbmVyYXRlKCoqaW5wdXRzKSUwQSUwQSUyMyUyMDUuJTIwZGVjb2RlJTIwdGhlJTIwZ2VuZXJhdGVkJTIwdG9rZW5zJTBBcHJpbnQocHJvY2Vzc29yLmJhdGNoX2RlY29kZShvdXRwdXRfdG9rZW5zJTJDJTIwc2tpcF9zcGVjaWFsX3Rva2VucyUzRFRydWUpKQ==",highlighted:`<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> datasets <span class="hljs-keyword">import</span> load_dataset, Audio
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> infer_device, KyutaiSpeechToTextProcessor, KyutaiSpeechToTextForConditionalGeneration

<span class="hljs-comment"># 1. load the model and the processor</span>
torch_device = infer_device()
model_id = <span class="hljs-string">&quot;kyutai/stt-2.6b-en-trfs&quot;</span>

processor = KyutaiSpeechToTextProcessor.from_pretrained(model_id)
model = KyutaiSpeechToTextForConditionalGeneration.from_pretrained(model_id, device_map=torch_device, dtype=<span class="hljs-string">&quot;auto&quot;</span>)

<span class="hljs-comment"># 2. load audio samples</span>
ds = load_dataset(
    <span class="hljs-string">&quot;hf-internal-testing/librispeech_asr_dummy&quot;</span>, <span class="hljs-string">&quot;clean&quot;</span>, split=<span class="hljs-string">&quot;validation&quot;</span>
)
ds = ds.cast_column(<span class="hljs-string">&quot;audio&quot;</span>, Audio(sampling_rate=<span class="hljs-number">24000</span>))

<span class="hljs-comment"># 3. prepare the model inputs</span>
inputs = processor(
    ds[<span class="hljs-number">0</span>][<span class="hljs-string">&quot;audio&quot;</span>][<span class="hljs-string">&quot;array&quot;</span>],
)
inputs.to(model.device)

<span class="hljs-comment"># 4. infer the model</span>
output_tokens = model.generate(**inputs)

<span class="hljs-comment"># 5. decode the generated tokens</span>
<span class="hljs-built_in">print</span>(processor.batch_decode(output_tokens, skip_special_tokens=<span class="hljs-literal">True</span>))`,wrap:!1}}),te=new z({props:{title:"Batched Inference",local:"batched-inference",headingTag:"h3"}}),oe=new lt({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwZGF0YXNldHMlMjBpbXBvcnQlMjBsb2FkX2RhdGFzZXQlMkMlMjBBdWRpbyUwQWZyb20lMjB0cmFuc2Zvcm1lcnMlMjBpbXBvcnQlMjBpbmZlcl9kZXZpY2UlMkMlMjBLeXV0YWlTcGVlY2hUb1RleHRQcm9jZXNzb3IlMkMlMjBLeXV0YWlTcGVlY2hUb1RleHRGb3JDb25kaXRpb25hbEdlbmVyYXRpb24lMEElMEElMjMlMjAxLiUyMGxvYWQlMjB0aGUlMjBtb2RlbCUyMGFuZCUyMHRoZSUyMHByb2Nlc3NvciUwQXRvcmNoX2RldmljZSUyMCUzRCUyMGluZmVyX2RldmljZSgpJTBBbW9kZWxfaWQlMjAlM0QlMjAlMjJreXV0YWklMkZzdHQtMi42Yi1lbi10cmZzJTIyJTBBJTBBcHJvY2Vzc29yJTIwJTNEJTIwS3l1dGFpU3BlZWNoVG9UZXh0UHJvY2Vzc29yLmZyb21fcHJldHJhaW5lZChtb2RlbF9pZCklMEFtb2RlbCUyMCUzRCUyMEt5dXRhaVNwZWVjaFRvVGV4dEZvckNvbmRpdGlvbmFsR2VuZXJhdGlvbi5mcm9tX3ByZXRyYWluZWQobW9kZWxfaWQlMkMlMjBkZXZpY2VfbWFwJTNEdG9yY2hfZGV2aWNlJTJDJTIwZHR5cGUlM0QlMjJhdXRvJTIyKSUwQSUwQSUyMyUyMDIuJTIwbG9hZCUyMGF1ZGlvJTIwc2FtcGxlcyUwQWRzJTIwJTNEJTIwbG9hZF9kYXRhc2V0KCUwQSUyMCUyMCUyMCUyMCUyMmhmLWludGVybmFsLXRlc3RpbmclMkZsaWJyaXNwZWVjaF9hc3JfZHVtbXklMjIlMkMlMjAlMjJjbGVhbiUyMiUyQyUyMHNwbGl0JTNEJTIydmFsaWRhdGlvbiUyMiUwQSklMEFkcyUyMCUzRCUyMGRzLmNhc3RfY29sdW1uKCUyMmF1ZGlvJTIyJTJDJTIwQXVkaW8oc2FtcGxpbmdfcmF0ZSUzRDI0MDAwKSklMEElMEElMjMlMjAzLiUyMHByZXBhcmUlMjB0aGUlMjBtb2RlbCUyMGlucHV0cyUwQWF1ZGlvX2FycmF5cyUyMCUzRCUyMCU1QmRzJTVCaSU1RCU1QiUyMmF1ZGlvJTIyJTVEJTVCJTIyYXJyYXklMjIlNUQlMjBmb3IlMjBpJTIwaW4lMjByYW5nZSg0KSU1RCUwQWlucHV0cyUyMCUzRCUyMHByb2Nlc3NvcihhdWRpb19hcnJheXMlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyJTJDJTIwcGFkZGluZyUzRFRydWUpJTBBaW5wdXRzJTIwJTNEJTIwaW5wdXRzLnRvKG1vZGVsLmRldmljZSklMEElMEElMjMlMjA0LiUyMGluZmVyJTIwdGhlJTIwbW9kZWwlMEFvdXRwdXRfdG9rZW5zJTIwJTNEJTIwbW9kZWwuZ2VuZXJhdGUoKippbnB1dHMpJTBBJTBBJTIzJTIwNS4lMjBkZWNvZGUlMjB0aGUlMjBnZW5lcmF0ZWQlMjB0b2tlbnMlMEFkZWNvZGVkX291dHB1dHMlMjAlM0QlMjBwcm9jZXNzb3IuYmF0Y2hfZGVjb2RlKG91dHB1dF90b2tlbnMlMkMlMjBza2lwX3NwZWNpYWxfdG9rZW5zJTNEVHJ1ZSklMEFmb3IlMjBvdXRwdXQlMjBpbiUyMGRlY29kZWRfb3V0cHV0cyUzQSUwQSUyMCUyMCUyMCUyMHByaW50KG91dHB1dCk=",highlighted:`<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> datasets <span class="hljs-keyword">import</span> load_dataset, Audio
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> infer_device, KyutaiSpeechToTextProcessor, KyutaiSpeechToTextForConditionalGeneration

<span class="hljs-comment"># 1. load the model and the processor</span>
torch_device = infer_device()
model_id = <span class="hljs-string">&quot;kyutai/stt-2.6b-en-trfs&quot;</span>

processor = KyutaiSpeechToTextProcessor.from_pretrained(model_id)
model = KyutaiSpeechToTextForConditionalGeneration.from_pretrained(model_id, device_map=torch_device, dtype=<span class="hljs-string">&quot;auto&quot;</span>)

<span class="hljs-comment"># 2. load audio samples</span>
ds = load_dataset(
    <span class="hljs-string">&quot;hf-internal-testing/librispeech_asr_dummy&quot;</span>, <span class="hljs-string">&quot;clean&quot;</span>, split=<span class="hljs-string">&quot;validation&quot;</span>
)
ds = ds.cast_column(<span class="hljs-string">&quot;audio&quot;</span>, Audio(sampling_rate=<span class="hljs-number">24000</span>))

<span class="hljs-comment"># 3. prepare the model inputs</span>
audio_arrays = [ds[i][<span class="hljs-string">&quot;audio&quot;</span>][<span class="hljs-string">&quot;array&quot;</span>] <span class="hljs-keyword">for</span> i <span class="hljs-keyword">in</span> <span class="hljs-built_in">range</span>(<span class="hljs-number">4</span>)]
inputs = processor(audio_arrays, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>, padding=<span class="hljs-literal">True</span>)
inputs = inputs.to(model.device)

<span class="hljs-comment"># 4. infer the model</span>
output_tokens = model.generate(**inputs)

<span class="hljs-comment"># 5. decode the generated tokens</span>
decoded_outputs = processor.batch_decode(output_tokens, skip_special_tokens=<span class="hljs-literal">True</span>)
<span class="hljs-keyword">for</span> output <span class="hljs-keyword">in</span> decoded_outputs:
    <span class="hljs-built_in">print</span>(output)`,wrap:!1}}),ae=new z({props:{title:"KyutaiSpeechToTextConfig",local:"transformers.KyutaiSpeechToTextConfig",headingTag:"h2"}}),se=new B({props:{name:"class transformers.KyutaiSpeechToTextConfig",anchor:"transformers.KyutaiSpeechToTextConfig",parameters:[{name:"codebook_vocab_size",val:" = 2049"},{name:"vocab_size",val:" = 4001"},{name:"hidden_size",val:" = 2048"},{name:"num_hidden_layers",val:" = 48"},{name:"num_attention_heads",val:" = 32"},{name:"num_key_value_heads",val:" = None"},{name:"max_position_embeddings",val:" = 750"},{name:"rope_theta",val:" = 100000.0"},{name:"hidden_act",val:" = 'silu'"},{name:"head_dim",val:" = None"},{name:"initializer_range",val:" = 0.02"},{name:"use_cache",val:" = True"},{name:"sliding_window",val:" = 375"},{name:"attention_dropout",val:" = 0.0"},{name:"ffn_dim",val:" = 11264"},{name:"rms_norm_eps",val:" = 1e-08"},{name:"num_codebooks",val:" = 32"},{name:"audio_bos_token_id",val:" = 2048"},{name:"audio_pad_token_id",val:" = 69569"},{name:"tie_word_embeddings",val:" = False"},{name:"pad_token_id",val:" = 3"},{name:"bos_token_id",val:" = 48000"},{name:"codec_config",val:" = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.KyutaiSpeechToTextConfig.codebook_vocab_size",description:`<strong>codebook_vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 2049) &#x2014;
Vocabulary size of the codebook. Defines the number of different audio tokens that can be represented by each codebook.`,name:"codebook_vocab_size"},{anchor:"transformers.KyutaiSpeechToTextConfig.vocab_size",description:`<strong>vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 4001) &#x2014;
Vocabulary size of the model. Defines the number of different tokens that can be represented by the
<code>input_ids</code> passed when calling the model.`,name:"vocab_size"},{anchor:"transformers.KyutaiSpeechToTextConfig.hidden_size",description:`<strong>hidden_size</strong> (<code>int</code>, <em>optional</em>, defaults to 2048) &#x2014;
Dimensionality of the layers and the pooler layer of the main decoder.`,name:"hidden_size"},{anchor:"transformers.KyutaiSpeechToTextConfig.num_hidden_layers",description:`<strong>num_hidden_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 48) &#x2014;
Number of decoder layers.`,name:"num_hidden_layers"},{anchor:"transformers.KyutaiSpeechToTextConfig.num_attention_heads",description:`<strong>num_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 32) &#x2014;
Number of attention heads for each attention layer in the main decoder block.`,name:"num_attention_heads"},{anchor:"transformers.KyutaiSpeechToTextConfig.num_key_value_heads",description:`<strong>num_key_value_heads</strong> (<code>int</code>, <em>optional</em>) &#x2014;
This is the number of key_value heads that should be used to implement Grouped Query Attention. If
<code>num_key_value_heads=num_attention_heads</code>, the model will use Multi Head Attention (MHA), if
<code>num_key_value_heads=1</code> the model will use Multi Query Attention (MQA) otherwise GQA is used. When
converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
by meanpooling all the original heads within that group. For more details checkout <a href="https://huggingface.co/papers/2305.13245" rel="nofollow">this
paper</a>. If it is not specified, will default to
<code>num_attention_heads</code>.`,name:"num_key_value_heads"},{anchor:"transformers.KyutaiSpeechToTextConfig.max_position_embeddings",description:`<strong>max_position_embeddings</strong> (<code>int</code>, <em>optional</em>, defaults to 750) &#x2014;
The maximum sequence length that this model might ever be used with. Typically, set this to something large
just in case (e.g., 512 or 1024 or 2048).`,name:"max_position_embeddings"},{anchor:"transformers.KyutaiSpeechToTextConfig.rope_theta",description:`<strong>rope_theta</strong> (<code>float</code>, <em>optional</em>, defaults to 100000.0) &#x2014;
The base period of the RoPE embeddings.`,name:"rope_theta"},{anchor:"transformers.KyutaiSpeechToTextConfig.hidden_act",description:`<strong>hidden_act</strong> (<code>str</code> or <code>function</code>, <em>optional</em>, defaults to <code>&quot;silu&quot;</code>) &#x2014;
The non-linear activation function (function or string) in the decoder.`,name:"hidden_act"},{anchor:"transformers.KyutaiSpeechToTextConfig.head_dim",description:`<strong>head_dim</strong> (<code>int</code>, <em>optional</em>, defaults to <code>hidden_size // num_attention_heads</code>) &#x2014;
The attention head dimension.`,name:"head_dim"},{anchor:"transformers.KyutaiSpeechToTextConfig.initializer_range",description:`<strong>initializer_range</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.`,name:"initializer_range"},{anchor:"transformers.KyutaiSpeechToTextConfig.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not the model should return the last key/values attentions (not used by all models). Only
relevant if <code>config.is_decoder=True</code>.`,name:"use_cache"},{anchor:"transformers.KyutaiSpeechToTextConfig.sliding_window",description:`<strong>sliding_window</strong> (<code>int</code>, <em>optional</em>, defaults to 375) &#x2014;
Sliding window attention window size. If not specified, will default to <code>3000</code>.`,name:"sliding_window"},{anchor:"transformers.KyutaiSpeechToTextConfig.attention_dropout",description:`<strong>attention_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The dropout ratio for the attention probabilities.`,name:"attention_dropout"},{anchor:"transformers.KyutaiSpeechToTextConfig.ffn_dim",description:`<strong>ffn_dim</strong> (<code>int</code>, <em>optional</em>, defaults to 11264) &#x2014;
Dimensionality of the &#x201C;intermediate&#x201D; (often named feed-forward) layer in the main decoder block. Must be even.`,name:"ffn_dim"},{anchor:"transformers.KyutaiSpeechToTextConfig.rms_norm_eps",description:`<strong>rms_norm_eps</strong> (<code>float</code>, <em>optional</em>, defaults to 1e-08) &#x2014;
The epsilon used by the rms normalization layers.`,name:"rms_norm_eps"},{anchor:"transformers.KyutaiSpeechToTextConfig.num_codebooks",description:`<strong>num_codebooks</strong> (<code>int</code>, <em>optional</em>, defaults to 32) &#x2014;
The number of audio codebooks for each audio channels.`,name:"num_codebooks"},{anchor:"transformers.KyutaiSpeechToTextConfig.audio_bos_token_id",description:`<strong>audio_bos_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 2048) &#x2014;
Beginning of stream token id for codebook tokens.`,name:"audio_bos_token_id"},{anchor:"transformers.KyutaiSpeechToTextConfig.audio_pad_token_id",description:`<strong>audio_pad_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 69569) &#x2014;
Padding token id for codebook tokens.`,name:"audio_pad_token_id"},{anchor:"transformers.KyutaiSpeechToTextConfig.tie_word_embeddings",description:`<strong>tie_word_embeddings</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether to tie weight embeddings.`,name:"tie_word_embeddings"},{anchor:"transformers.KyutaiSpeechToTextConfig.pad_token_id",description:`<strong>pad_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 3) &#x2014;
Padding token id.`,name:"pad_token_id"},{anchor:"transformers.KyutaiSpeechToTextConfig.bos_token_id",description:`<strong>bos_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 48000) &#x2014;
Beginning of stream token id for text tokens.`,name:"bos_token_id"},{anchor:"transformers.KyutaiSpeechToTextConfig.codec_config",description:`<strong>codec_config</strong> (<code>PretrainedConfig</code>, <em>optional</em>) &#x2014;
Configuration for the codec.`,name:"codec_config"},{anchor:"transformers.KyutaiSpeechToTextConfig.kwargs",description:`<strong>kwargs</strong> (<em>optional</em>) &#x2014;
Dictionary of keyword arguments. Notably:<ul>
<li><strong>audio_encoder_config</strong> (<a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a>, <em>optional</em>) &#x2014; An instance of a configuration object that
defines the audio encoder config.</li>
<li><strong>depth__config</strong> (<a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a>, <em>optional</em>) &#x2014; An instance of a configuration object that
defines the depth decoder config.</li>
</ul>`,name:"kwargs"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/kyutai_speech_to_text/configuration_kyutai_speech_to_text.py#L24"}}),V=new to({props:{anchor:"transformers.KyutaiSpeechToTextConfig.example",$$slots:{default:[lo]},$$scope:{ctx:G}}}),re=new z({props:{title:"KyutaiSpeechToTextProcessor",local:"transformers.KyutaiSpeechToTextProcessor",headingTag:"h2"}}),ie=new B({props:{name:"class transformers.KyutaiSpeechToTextProcessor",anchor:"transformers.KyutaiSpeechToTextProcessor",parameters:[{name:"*args",val:""},{name:"**kwargs",val:""}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/kyutai_speech_to_text/processing_kyutai_speech_to_text.py#L31"}}),le=new B({props:{name:"__call__",anchor:"transformers.KyutaiSpeechToTextProcessor.__call__",parameters:[{name:"audio",val:": typing.Union[numpy.ndarray, ForwardRef('torch.Tensor'), typing.Sequence[numpy.ndarray], typing.Sequence[ForwardRef('torch.Tensor')], NoneType] = None"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.models.kyutai_speech_to_text.processing_kyutai_speech_to_text.KyutaiSpeechToTextProcessorKwargs]"}],parametersDescription:[{anchor:"transformers.KyutaiSpeechToTextProcessor.__call__.audio",description:`<strong>audio</strong> (<code>np.ndarray</code>, <code>torch.Tensor</code>, <code>list[np.ndarray]</code>, <code>list[torch.Tensor]</code>) &#x2014;
The audio or batch of audio to be prepared. Each audio can be a NumPy array or PyTorch
tensor.`,name:"audio"},{anchor:"transformers.KyutaiSpeechToTextProcessor.__call__.return_tensors",description:`<strong>return_tensors</strong> (<code>str</code> or <a href="/docs/transformers/v4.56.2/en/internal/file_utils#transformers.TensorType">TensorType</a>, <em>optional</em>) &#x2014;
If set, will return tensors of a particular framework. Acceptable values are:<ul>
<li><code>&apos;tf&apos;</code>: Return TensorFlow <code>tf.constant</code> objects.</li>
<li><code>&apos;pt&apos;</code>: Return PyTorch <code>torch.Tensor</code> objects.</li>
<li><code>&apos;np&apos;</code>: Return NumPy <code>np.ndarray</code> objects.</li>
<li><code>&apos;jax&apos;</code>: Return JAX <code>jnp.ndarray</code> objects.</li>
</ul>`,name:"return_tensors"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/kyutai_speech_to_text/processing_kyutai_speech_to_text.py#L42",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/image_processor#transformers.BatchFeature"
>BatchFeature</a> with the following fields:</p>
<ul>
<li><strong>input_values</strong> — List of audio values to be fed to a model. Returned when <code>audio</code> is not <code>None</code>.</li>
<li><strong>padding_mask</strong> — List of indices specifying which input values should be ignored by the model.</li>
</ul>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><a
  href="/docs/transformers/v4.56.2/en/main_classes/image_processor#transformers.BatchFeature"
>BatchFeature</a></p>
`}}),ce=new z({props:{title:"KyutaiSpeechToTextFeatureExtractor",local:"transformers.KyutaiSpeechToTextFeatureExtractor",headingTag:"h2"}}),de=new B({props:{name:"class transformers.KyutaiSpeechToTextFeatureExtractor",anchor:"transformers.KyutaiSpeechToTextFeatureExtractor",parameters:[{name:"feature_size",val:": int = 1"},{name:"sampling_rate",val:": int = 24000"},{name:"padding_value",val:": float = 0.0"},{name:"chunk_length_s",val:": typing.Optional[float] = None"},{name:"overlap",val:": typing.Optional[float] = None"},{name:"audio_delay_seconds",val:": typing.Optional[float] = 0.0"},{name:"audio_silence_prefix_seconds",val:": typing.Optional[float] = 0.0"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.KyutaiSpeechToTextFeatureExtractor.feature_size",description:`<strong>feature_size</strong> (<code>int</code>, <em>optional</em>, defaults to 1) &#x2014;
The feature dimension of the extracted features. Use 1 for mono, 2 for stereo.`,name:"feature_size"},{anchor:"transformers.KyutaiSpeechToTextFeatureExtractor.sampling_rate",description:`<strong>sampling_rate</strong> (<code>int</code>, <em>optional</em>, defaults to 24000) &#x2014;
The sampling rate at which the audio waveform should be digitalized expressed in hertz (Hz).`,name:"sampling_rate"},{anchor:"transformers.KyutaiSpeechToTextFeatureExtractor.padding_value",description:`<strong>padding_value</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The value that is used to fill the padding values.`,name:"padding_value"},{anchor:"transformers.KyutaiSpeechToTextFeatureExtractor.chunk_length_s",description:`<strong>chunk_length_s</strong> (<code>float</code>, <em>optional</em>) &#x2014;
If defined the audio is pre-processed into chunks of lengths <code>chunk_length_s</code> and then encoded.`,name:"chunk_length_s"},{anchor:"transformers.KyutaiSpeechToTextFeatureExtractor.overlap",description:`<strong>overlap</strong> (<code>float</code>, <em>optional</em>) &#x2014;
Defines the overlap between each chunk. It is used to compute the <code>chunk_stride</code> using the following
formulae : <code>int((1.0 - self.overlap) * self.chunk_length)</code>.`,name:"overlap"},{anchor:"transformers.KyutaiSpeechToTextFeatureExtractor.audio_delay_seconds",description:`<strong>audio_delay_seconds</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The delay in seconds to add after the audio (right padding).`,name:"audio_delay_seconds"},{anchor:"transformers.KyutaiSpeechToTextFeatureExtractor.audio_silence_prefix_seconds",description:`<strong>audio_silence_prefix_seconds</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The silence prefix in seconds to add before the audio (left padding).`,name:"audio_silence_prefix_seconds"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/kyutai_speech_to_text/feature_extraction_kyutai_speech_to_text.py#L34"}}),pe=new z({props:{title:"KyutaiSpeechToTextForConditionalGeneration",local:"transformers.KyutaiSpeechToTextForConditionalGeneration",headingTag:"h2"}}),he=new B({props:{name:"class transformers.KyutaiSpeechToTextForConditionalGeneration",anchor:"transformers.KyutaiSpeechToTextForConditionalGeneration",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.KyutaiSpeechToTextForConditionalGeneration.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/kyutai_speech_to_text#transformers.KyutaiSpeechToTextForConditionalGeneration">KyutaiSpeechToTextForConditionalGeneration</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/kyutai_speech_to_text/modeling_kyutai_speech_to_text.py#L1071"}}),me=new B({props:{name:"forward",anchor:"transformers.KyutaiSpeechToTextForConditionalGeneration.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Optional[transformers.cache_utils.Cache] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"},{name:"logits_to_keep",val:": typing.Union[int, torch.Tensor] = 0"},{name:"**kwargs",val:": typing_extensions.Unpack[transformers.utils.generic.TransformersKwargs]"}],parametersDescription:[{anchor:"transformers.KyutaiSpeechToTextForConditionalGeneration.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.KyutaiSpeechToTextForConditionalGeneration.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.KyutaiSpeechToTextForConditionalGeneration.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.KyutaiSpeechToTextForConditionalGeneration.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>~cache_utils.Cache</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.KyutaiSpeechToTextForConditionalGeneration.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.KyutaiSpeechToTextForConditionalGeneration.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.KyutaiSpeechToTextForConditionalGeneration.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.KyutaiSpeechToTextForConditionalGeneration.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"},{anchor:"transformers.KyutaiSpeechToTextForConditionalGeneration.forward.logits_to_keep",description:`<strong>logits_to_keep</strong> (<code>Union[int, torch.Tensor]</code>, defaults to <code>0</code>) &#x2014;
If an <code>int</code>, compute logits for the last <code>logits_to_keep</code> tokens. If <code>0</code>, calculate logits for all
<code>input_ids</code> (special case). Only last token logits are needed for generation, and calculating them only for that
token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
If a <code>torch.Tensor</code>, must be 1D corresponding to the indices to keep in the sequence length dimension.
This is useful when using packed tensor format (single dimension for batch and sequence length).`,name:"logits_to_keep"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/kyutai_speech_to_text/modeling_kyutai_speech_to_text.py#L1092",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast"
>transformers.modeling_outputs.CausalLMOutputWithPast</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/kyutai_speech_to_text#transformers.KyutaiSpeechToTextConfig"
>KyutaiSpeechToTextConfig</a>) and inputs.</p>
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
`}}),E=new eo({props:{$$slots:{default:[co]},$$scope:{ctx:G}}}),N=new to({props:{anchor:"transformers.KyutaiSpeechToTextForConditionalGeneration.forward.example",$$slots:{default:[po]},$$scope:{ctx:G}}}),ue=new B({props:{name:"generate",anchor:"transformers.KyutaiSpeechToTextForConditionalGeneration.generate",parameters:[{name:"*args",val:""},{name:"**kwargs",val:""}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/kyutai_speech_to_text/modeling_kyutai_speech_to_text.py#L1347"}}),fe=new z({props:{title:"KyutaiSpeechToTextModel",local:"transformers.KyutaiSpeechToTextModel",headingTag:"h2"}}),ge=new B({props:{name:"class transformers.KyutaiSpeechToTextModel",anchor:"transformers.KyutaiSpeechToTextModel",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.KyutaiSpeechToTextModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/kyutai_speech_to_text#transformers.KyutaiSpeechToTextModel">KyutaiSpeechToTextModel</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/kyutai_speech_to_text/modeling_kyutai_speech_to_text.py#L805"}}),_e=new B({props:{name:"forward",anchor:"transformers.KyutaiSpeechToTextModel.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"past_key_values",val:": typing.Union[list[torch.FloatTensor], transformers.cache_utils.Cache, NoneType] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.LongTensor] = None"}],parametersDescription:[{anchor:"transformers.KyutaiSpeechToTextModel.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.KyutaiSpeechToTextModel.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.KyutaiSpeechToTextModel.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.KyutaiSpeechToTextModel.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>Union[list[torch.FloatTensor], ~cache_utils.Cache, NoneType]</code>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.KyutaiSpeechToTextModel.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.KyutaiSpeechToTextModel.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.KyutaiSpeechToTextModel.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.KyutaiSpeechToTextModel.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.KyutaiSpeechToTextModel.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.KyutaiSpeechToTextModel.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.LongTensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/kyutai_speech_to_text/modeling_kyutai_speech_to_text.py#L823",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast"
>transformers.modeling_outputs.BaseModelOutputWithPast</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/kyutai_speech_to_text#transformers.KyutaiSpeechToTextConfig"
>KyutaiSpeechToTextConfig</a>) and inputs.</p>
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
`}}),q=new eo({props:{$$slots:{default:[ho]},$$scope:{ctx:G}}}),ye=new io({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/kyutai_speech_to_text.md"}}),{c(){i=l("meta"),M=n(),_=l("p"),T=n(),b=l("p"),b.innerHTML=d,$=n(),p(P.$$.fragment),We=n(),p(Q.$$.fragment),Be=n(),Y=l("p"),Y.innerHTML=Rt,Ie=n(),D=l("ul"),D.innerHTML=Zt,Ve=n(),I=l("div"),I.innerHTML=zt,Xe=n(),p(A.$$.fragment),Ee=n(),p(O.$$.fragment),Ne=n(),p(ee.$$.fragment),He=n(),p(te.$$.fragment),qe=n(),p(oe.$$.fragment),Le=n(),ne=l("p"),ne.innerHTML=Kt,Pe=n(),p(ae.$$.fragment),Qe=n(),k=l("div"),p(se.$$.fragment),dt=n(),be=l("p"),be.innerHTML=Wt,pt=n(),ve=l("p"),ve.innerHTML=Bt,ht=n(),Me=l("p"),Me.innerHTML=It,mt=n(),p(V.$$.fragment),Ye=n(),p(re.$$.fragment),De=n(),S=l("div"),p(ie.$$.fragment),ut=n(),ke=l("p"),ke.innerHTML=Vt,ft=n(),X=l("div"),p(le.$$.fragment),gt=n(),xe=l("p"),xe.innerHTML=Xt,Ae=n(),p(ce.$$.fragment),Oe=n(),J=l("div"),p(de.$$.fragment),_t=n(),we=l("p"),we.textContent=Et,yt=n(),$e=l("p"),$e.innerHTML=Nt,et=n(),p(pe.$$.fragment),tt=n(),v=l("div"),p(he.$$.fragment),Tt=n(),je=l("p"),je.textContent=Ht,bt=n(),Ce=l("p"),Ce.innerHTML=qt,vt=n(),Ue=l("p"),Ue.innerHTML=Lt,Mt=n(),j=l("div"),p(me.$$.fragment),kt=n(),Se=l("p"),Se.innerHTML=Pt,xt=n(),p(E.$$.fragment),wt=n(),p(N.$$.fragment),$t=n(),H=l("div"),p(ue.$$.fragment),jt=n(),Je=l("p"),Je.innerHTML=Qt,ot=n(),p(fe.$$.fragment),nt=n(),x=l("div"),p(ge.$$.fragment),Ct=n(),Fe=l("p"),Fe.textContent=Yt,Ut=n(),Ge=l("p"),Ge.innerHTML=Dt,St=n(),Re=l("p"),Re.innerHTML=At,Jt=n(),R=l("div"),p(_e.$$.fragment),Ft=n(),Ze=l("p"),Ze.innerHTML=Ot,Gt=n(),p(q.$$.fragment),at=n(),p(ye.$$.fragment),st=n(),Ke=l("p"),this.h()},l(e){const t=ro("svelte-u9bgzb",document.head);i=c(t,"META",{name:!0,content:!0}),t.forEach(o),M=a(e),_=c(e,"P",{}),F(_).forEach(o),T=a(e),b=c(e,"P",{"data-svelte-h":!0}),y(b)!=="svelte-q80isj"&&(b.innerHTML=d),$=a(e),h(P.$$.fragment,e),We=a(e),h(Q.$$.fragment,e),Be=a(e),Y=c(e,"P",{"data-svelte-h":!0}),y(Y)!=="svelte-12capup"&&(Y.innerHTML=Rt),Ie=a(e),D=c(e,"UL",{"data-svelte-h":!0}),y(D)!=="svelte-71ak3b"&&(D.innerHTML=Zt),Ve=a(e),I=c(e,"DIV",{class:!0,"data-svelte-h":!0}),y(I)!=="svelte-urbq5v"&&(I.innerHTML=zt),Xe=a(e),h(A.$$.fragment,e),Ee=a(e),h(O.$$.fragment,e),Ne=a(e),h(ee.$$.fragment,e),He=a(e),h(te.$$.fragment,e),qe=a(e),h(oe.$$.fragment,e),Le=a(e),ne=c(e,"P",{"data-svelte-h":!0}),y(ne)!=="svelte-ju28dl"&&(ne.innerHTML=Kt),Pe=a(e),h(ae.$$.fragment,e),Qe=a(e),k=c(e,"DIV",{class:!0});var w=F(k);h(se.$$.fragment,w),dt=a(w),be=c(w,"P",{"data-svelte-h":!0}),y(be)!=="svelte-cbadsl"&&(be.innerHTML=Wt),pt=a(w),ve=c(w,"P",{"data-svelte-h":!0}),y(ve)!=="svelte-1rtojv5"&&(ve.innerHTML=Bt),ht=a(w),Me=c(w,"P",{"data-svelte-h":!0}),y(Me)!=="svelte-1ek1ss9"&&(Me.innerHTML=It),mt=a(w),h(V.$$.fragment,w),w.forEach(o),Ye=a(e),h(re.$$.fragment,e),De=a(e),S=c(e,"DIV",{class:!0});var K=F(S);h(ie.$$.fragment,K),ut=a(K),ke=c(K,"P",{"data-svelte-h":!0}),y(ke)!=="svelte-wmed76"&&(ke.innerHTML=Vt),ft=a(K),X=c(K,"DIV",{class:!0});var Te=F(X);h(le.$$.fragment,Te),gt=a(Te),xe=c(Te,"P",{"data-svelte-h":!0}),y(xe)!=="svelte-xkykzi"&&(xe.innerHTML=Xt),Te.forEach(o),K.forEach(o),Ae=a(e),h(ce.$$.fragment,e),Oe=a(e),J=c(e,"DIV",{class:!0});var W=F(J);h(de.$$.fragment,W),_t=a(W),we=c(W,"P",{"data-svelte-h":!0}),y(we)!=="svelte-1oxfvt8"&&(we.textContent=Et),yt=a(W),$e=c(W,"P",{"data-svelte-h":!0}),y($e)!=="svelte-ue5gbv"&&($e.innerHTML=Nt),W.forEach(o),et=a(e),h(pe.$$.fragment,e),tt=a(e),v=c(e,"DIV",{class:!0});var C=F(v);h(he.$$.fragment,C),Tt=a(C),je=c(C,"P",{"data-svelte-h":!0}),y(je)!=="svelte-1w9d6gf"&&(je.textContent=Ht),bt=a(C),Ce=c(C,"P",{"data-svelte-h":!0}),y(Ce)!=="svelte-q52n56"&&(Ce.innerHTML=qt),vt=a(C),Ue=c(C,"P",{"data-svelte-h":!0}),y(Ue)!=="svelte-hswkmf"&&(Ue.innerHTML=Lt),Mt=a(C),j=c(C,"DIV",{class:!0});var L=F(j);h(me.$$.fragment,L),kt=a(L),Se=c(L,"P",{"data-svelte-h":!0}),y(Se)!=="svelte-1zi01f"&&(Se.innerHTML=Pt),xt=a(L),h(E.$$.fragment,L),wt=a(L),h(N.$$.fragment,L),L.forEach(o),$t=a(C),H=c(C,"DIV",{class:!0});var it=F(H);h(ue.$$.fragment,it),jt=a(it),Je=c(it,"P",{"data-svelte-h":!0}),y(Je)!=="svelte-2b4xoe"&&(Je.innerHTML=Qt),it.forEach(o),C.forEach(o),ot=a(e),h(fe.$$.fragment,e),nt=a(e),x=c(e,"DIV",{class:!0});var Z=F(x);h(ge.$$.fragment,Z),Ct=a(Z),Fe=c(Z,"P",{"data-svelte-h":!0}),y(Fe)!=="svelte-19xqfsq"&&(Fe.textContent=Yt),Ut=a(Z),Ge=c(Z,"P",{"data-svelte-h":!0}),y(Ge)!=="svelte-q52n56"&&(Ge.innerHTML=Dt),St=a(Z),Re=c(Z,"P",{"data-svelte-h":!0}),y(Re)!=="svelte-hswkmf"&&(Re.innerHTML=At),Jt=a(Z),R=c(Z,"DIV",{class:!0});var ze=F(R);h(_e.$$.fragment,ze),Ft=a(ze),Ze=c(ze,"P",{"data-svelte-h":!0}),y(Ze)!=="svelte-1gajvan"&&(Ze.innerHTML=Ot),Gt=a(ze),h(q.$$.fragment,ze),ze.forEach(o),Z.forEach(o),at=a(e),h(ye.$$.fragment,e),st=a(e),Ke=c(e,"P",{}),F(Ke).forEach(o),this.h()},h(){U(i,"name","hf:doc:metadata"),U(i,"content",uo),U(I,"class","flex justify-center"),U(k,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),U(X,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),U(S,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),U(J,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),U(j,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),U(H,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),U(v,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),U(R,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),U(x,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,t){s(document.head,i),r(e,M,t),r(e,_,t),r(e,T,t),r(e,b,t),r(e,$,t),m(P,e,t),r(e,We,t),m(Q,e,t),r(e,Be,t),r(e,Y,t),r(e,Ie,t),r(e,D,t),r(e,Ve,t),r(e,I,t),r(e,Xe,t),m(A,e,t),r(e,Ee,t),m(O,e,t),r(e,Ne,t),m(ee,e,t),r(e,He,t),m(te,e,t),r(e,qe,t),m(oe,e,t),r(e,Le,t),r(e,ne,t),r(e,Pe,t),m(ae,e,t),r(e,Qe,t),r(e,k,t),m(se,k,null),s(k,dt),s(k,be),s(k,pt),s(k,ve),s(k,ht),s(k,Me),s(k,mt),m(V,k,null),r(e,Ye,t),m(re,e,t),r(e,De,t),r(e,S,t),m(ie,S,null),s(S,ut),s(S,ke),s(S,ft),s(S,X),m(le,X,null),s(X,gt),s(X,xe),r(e,Ae,t),m(ce,e,t),r(e,Oe,t),r(e,J,t),m(de,J,null),s(J,_t),s(J,we),s(J,yt),s(J,$e),r(e,et,t),m(pe,e,t),r(e,tt,t),r(e,v,t),m(he,v,null),s(v,Tt),s(v,je),s(v,bt),s(v,Ce),s(v,vt),s(v,Ue),s(v,Mt),s(v,j),m(me,j,null),s(j,kt),s(j,Se),s(j,xt),m(E,j,null),s(j,wt),m(N,j,null),s(v,$t),s(v,H),m(ue,H,null),s(H,jt),s(H,Je),r(e,ot,t),m(fe,e,t),r(e,nt,t),r(e,x,t),m(ge,x,null),s(x,Ct),s(x,Fe),s(x,Ut),s(x,Ge),s(x,St),s(x,Re),s(x,Jt),s(x,R),m(_e,R,null),s(R,Ft),s(R,Ze),s(R,Gt),m(q,R,null),r(e,at,t),m(ye,e,t),r(e,st,t),r(e,Ke,t),rt=!0},p(e,[t]){const w={};t&2&&(w.$$scope={dirty:t,ctx:e}),V.$set(w);const K={};t&2&&(K.$$scope={dirty:t,ctx:e}),E.$set(K);const Te={};t&2&&(Te.$$scope={dirty:t,ctx:e}),N.$set(Te);const W={};t&2&&(W.$$scope={dirty:t,ctx:e}),q.$set(W)},i(e){rt||(u(P.$$.fragment,e),u(Q.$$.fragment,e),u(A.$$.fragment,e),u(O.$$.fragment,e),u(ee.$$.fragment,e),u(te.$$.fragment,e),u(oe.$$.fragment,e),u(ae.$$.fragment,e),u(se.$$.fragment,e),u(V.$$.fragment,e),u(re.$$.fragment,e),u(ie.$$.fragment,e),u(le.$$.fragment,e),u(ce.$$.fragment,e),u(de.$$.fragment,e),u(pe.$$.fragment,e),u(he.$$.fragment,e),u(me.$$.fragment,e),u(E.$$.fragment,e),u(N.$$.fragment,e),u(ue.$$.fragment,e),u(fe.$$.fragment,e),u(ge.$$.fragment,e),u(_e.$$.fragment,e),u(q.$$.fragment,e),u(ye.$$.fragment,e),rt=!0)},o(e){f(P.$$.fragment,e),f(Q.$$.fragment,e),f(A.$$.fragment,e),f(O.$$.fragment,e),f(ee.$$.fragment,e),f(te.$$.fragment,e),f(oe.$$.fragment,e),f(ae.$$.fragment,e),f(se.$$.fragment,e),f(V.$$.fragment,e),f(re.$$.fragment,e),f(ie.$$.fragment,e),f(le.$$.fragment,e),f(ce.$$.fragment,e),f(de.$$.fragment,e),f(pe.$$.fragment,e),f(he.$$.fragment,e),f(me.$$.fragment,e),f(E.$$.fragment,e),f(N.$$.fragment,e),f(ue.$$.fragment,e),f(fe.$$.fragment,e),f(ge.$$.fragment,e),f(_e.$$.fragment,e),f(q.$$.fragment,e),f(ye.$$.fragment,e),rt=!1},d(e){e&&(o(M),o(_),o(T),o(b),o($),o(We),o(Be),o(Y),o(Ie),o(D),o(Ve),o(I),o(Xe),o(Ee),o(Ne),o(He),o(qe),o(Le),o(ne),o(Pe),o(Qe),o(k),o(Ye),o(De),o(S),o(Ae),o(Oe),o(J),o(et),o(tt),o(v),o(ot),o(nt),o(x),o(at),o(st),o(Ke)),o(i),g(P,e),g(Q,e),g(A,e),g(O,e),g(ee,e),g(te,e),g(oe,e),g(ae,e),g(se),g(V),g(re,e),g(ie),g(le),g(ce,e),g(de),g(pe,e),g(he),g(me),g(E),g(N),g(ue),g(fe,e),g(ge),g(_e),g(q),g(ye,e)}}}const uo='{"title":"Kyutai Speech-To-Text","local":"kyutai-speech-to-text","sections":[{"title":"Overview","local":"overview","sections":[],"depth":2},{"title":"Usage Tips","local":"usage-tips","sections":[{"title":"Inference","local":"inference","sections":[],"depth":3},{"title":"Batched Inference","local":"batched-inference","sections":[],"depth":3}],"depth":2},{"title":"KyutaiSpeechToTextConfig","local":"transformers.KyutaiSpeechToTextConfig","sections":[],"depth":2},{"title":"KyutaiSpeechToTextProcessor","local":"transformers.KyutaiSpeechToTextProcessor","sections":[],"depth":2},{"title":"KyutaiSpeechToTextFeatureExtractor","local":"transformers.KyutaiSpeechToTextFeatureExtractor","sections":[],"depth":2},{"title":"KyutaiSpeechToTextForConditionalGeneration","local":"transformers.KyutaiSpeechToTextForConditionalGeneration","sections":[],"depth":2},{"title":"KyutaiSpeechToTextModel","local":"transformers.KyutaiSpeechToTextModel","sections":[],"depth":2}],"depth":1}';function fo(G){return no(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class ko extends ao{constructor(i){super(),so(this,i,fo,mo,oo,{})}}export{ko as component};
