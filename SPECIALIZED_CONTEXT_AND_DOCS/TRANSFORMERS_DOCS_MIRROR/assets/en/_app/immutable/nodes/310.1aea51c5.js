import{s as gs,o as fs,n as Ft}from"../chunks/scheduler.18a86fab.js";import{S as _s,i as Ms,g as r,s,r as m,A as bs,h as i,f as n,c as a,j as F,x as c,u,k as $,y as d,a as o,v as h,d as g,t as f,w as _}from"../chunks/index.98837b22.js";import{T as yo}from"../chunks/Tip.77304350.js";import{D as B}from"../chunks/Docstring.a1ef7999.js";import{C as W}from"../chunks/CodeBlock.8d0c2e8a.js";import{E as vo}from"../chunks/ExampleCodeBlock.8c3ee1f9.js";import{H as J,E as ys}from"../chunks/getInferenceSnippets.06c2775f.js";function vs(U){let l,y="Example:",M,b,v;return b=new W({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMCglMEElMjAlMjAlMjAlMjBNdXNpY2dlbkNvbmZpZyUyQyUwQSUyMCUyMCUyMCUyME11c2ljZ2VuRGVjb2RlckNvbmZpZyUyQyUwQSUyMCUyMCUyMCUyMFQ1Q29uZmlnJTJDJTBBJTIwJTIwJTIwJTIwRW5jb2RlY0NvbmZpZyUyQyUwQSUyMCUyMCUyMCUyME11c2ljZ2VuRm9yQ29uZGl0aW9uYWxHZW5lcmF0aW9uJTJDJTBBKSUwQSUwQSUyMyUyMEluaXRpYWxpemluZyUyMHRleHQlMjBlbmNvZGVyJTJDJTIwYXVkaW8lMjBlbmNvZGVyJTJDJTIwYW5kJTIwZGVjb2RlciUyMG1vZGVsJTIwY29uZmlndXJhdGlvbnMlMEF0ZXh0X2VuY29kZXJfY29uZmlnJTIwJTNEJTIwVDVDb25maWcoKSUwQWF1ZGlvX2VuY29kZXJfY29uZmlnJTIwJTNEJTIwRW5jb2RlY0NvbmZpZygpJTBBZGVjb2Rlcl9jb25maWclMjAlM0QlMjBNdXNpY2dlbkRlY29kZXJDb25maWcoKSUwQSUwQWNvbmZpZ3VyYXRpb24lMjAlM0QlMjBNdXNpY2dlbkNvbmZpZy5mcm9tX3N1Yl9tb2RlbHNfY29uZmlnKCUwQSUyMCUyMCUyMCUyMHRleHRfZW5jb2Rlcl9jb25maWclMkMlMjBhdWRpb19lbmNvZGVyX2NvbmZpZyUyQyUyMGRlY29kZXJfY29uZmlnJTBBKSUwQSUwQSUyMyUyMEluaXRpYWxpemluZyUyMGElMjBNdXNpY2dlbkZvckNvbmRpdGlvbmFsR2VuZXJhdGlvbiUyMCh3aXRoJTIwcmFuZG9tJTIwd2VpZ2h0cyklMjBmcm9tJTIwdGhlJTIwZmFjZWJvb2slMkZtdXNpY2dlbi1zbWFsbCUyMHN0eWxlJTIwY29uZmlndXJhdGlvbiUwQW1vZGVsJTIwJTNEJTIwTXVzaWNnZW5Gb3JDb25kaXRpb25hbEdlbmVyYXRpb24oY29uZmlndXJhdGlvbiklMEElMEElMjMlMjBBY2Nlc3NpbmclMjB0aGUlMjBtb2RlbCUyMGNvbmZpZ3VyYXRpb24lMEFjb25maWd1cmF0aW9uJTIwJTNEJTIwbW9kZWwuY29uZmlnJTBBY29uZmlnX3RleHRfZW5jb2RlciUyMCUzRCUyMG1vZGVsLmNvbmZpZy50ZXh0X2VuY29kZXIlMEFjb25maWdfYXVkaW9fZW5jb2RlciUyMCUzRCUyMG1vZGVsLmNvbmZpZy5hdWRpb19lbmNvZGVyJTBBY29uZmlnX2RlY29kZXIlMjAlM0QlMjBtb2RlbC5jb25maWcuZGVjb2RlciUwQSUwQSUyMyUyMFNhdmluZyUyMHRoZSUyMG1vZGVsJTJDJTIwaW5jbHVkaW5nJTIwaXRzJTIwY29uZmlndXJhdGlvbiUwQW1vZGVsLnNhdmVfcHJldHJhaW5lZCglMjJtdXNpY2dlbi1tb2RlbCUyMiklMEElMEElMjMlMjBsb2FkaW5nJTIwbW9kZWwlMjBhbmQlMjBjb25maWclMjBmcm9tJTIwcHJldHJhaW5lZCUyMGZvbGRlciUwQW11c2ljZ2VuX2NvbmZpZyUyMCUzRCUyME11c2ljZ2VuQ29uZmlnLmZyb21fcHJldHJhaW5lZCglMjJtdXNpY2dlbi1tb2RlbCUyMiklMEFtb2RlbCUyMCUzRCUyME11c2ljZ2VuRm9yQ29uZGl0aW9uYWxHZW5lcmF0aW9uLmZyb21fcHJldHJhaW5lZCglMjJtdXNpY2dlbi1tb2RlbCUyMiUyQyUyMGNvbmZpZyUzRG11c2ljZ2VuX2NvbmZpZyk=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> (
<span class="hljs-meta">... </span>    MusicgenConfig,
<span class="hljs-meta">... </span>    MusicgenDecoderConfig,
<span class="hljs-meta">... </span>    T5Config,
<span class="hljs-meta">... </span>    EncodecConfig,
<span class="hljs-meta">... </span>    MusicgenForConditionalGeneration,
<span class="hljs-meta">... </span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing text encoder, audio encoder, and decoder model configurations</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>text_encoder_config = T5Config()
<span class="hljs-meta">&gt;&gt;&gt; </span>audio_encoder_config = EncodecConfig()
<span class="hljs-meta">&gt;&gt;&gt; </span>decoder_config = MusicgenDecoderConfig()

<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = MusicgenConfig.from_sub_models_config(
<span class="hljs-meta">... </span>    text_encoder_config, audio_encoder_config, decoder_config
<span class="hljs-meta">... </span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a MusicgenForConditionalGeneration (with random weights) from the facebook/musicgen-small style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = MusicgenForConditionalGeneration(configuration)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Accessing the model configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = model.config
<span class="hljs-meta">&gt;&gt;&gt; </span>config_text_encoder = model.config.text_encoder
<span class="hljs-meta">&gt;&gt;&gt; </span>config_audio_encoder = model.config.audio_encoder
<span class="hljs-meta">&gt;&gt;&gt; </span>config_decoder = model.config.decoder

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Saving the model, including its configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model.save_pretrained(<span class="hljs-string">&quot;musicgen-model&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># loading model and config from pretrained folder</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>musicgen_config = MusicgenConfig.from_pretrained(<span class="hljs-string">&quot;musicgen-model&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = MusicgenForConditionalGeneration.from_pretrained(<span class="hljs-string">&quot;musicgen-model&quot;</span>, config=musicgen_config)`,wrap:!1}}),{c(){l=r("p"),l.textContent=y,M=s(),m(b.$$.fragment)},l(p){l=i(p,"P",{"data-svelte-h":!0}),c(l)!=="svelte-11lpom8"&&(l.textContent=y),M=a(p),u(b.$$.fragment,p)},m(p,T){o(p,l,T),o(p,M,T),h(b,p,T),v=!0},p:Ft,i(p){v||(g(b.$$.fragment,p),v=!0)},o(p){f(b.$$.fragment,p),v=!1},d(p){p&&(n(l),n(M)),_(b,p)}}}function Ts(U){let l,y=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){l=r("p"),l.innerHTML=y},l(M){l=i(M,"P",{"data-svelte-h":!0}),c(l)!=="svelte-fincs2"&&(l.innerHTML=y)},m(M,b){o(M,l,b)},p:Ft,d(M){M&&n(l)}}}function ws(U){let l,y=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){l=r("p"),l.innerHTML=y},l(M){l=i(M,"P",{"data-svelte-h":!0}),c(l)!=="svelte-fincs2"&&(l.innerHTML=y)},m(M,b){o(M,l,b)},p:Ft,d(M){M&&n(l)}}}function ks(U){let l,y="Example:",M,b,v;return b=new W({props:{code:"",highlighted:"",wrap:!1}}),{c(){l=r("p"),l.textContent=y,M=s(),m(b.$$.fragment)},l(p){l=i(p,"P",{"data-svelte-h":!0}),c(l)!=="svelte-11lpom8"&&(l.textContent=y),M=a(p),u(b.$$.fragment,p)},m(p,T){o(p,l,T),o(p,M,T),h(b,p,T),v=!0},p:Ft,i(p){v||(g(b.$$.fragment,p),v=!0)},o(p){f(b.$$.fragment,p),v=!1},d(p){p&&(n(l),n(M)),_(b,p)}}}function Cs(U){let l,y=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){l=r("p"),l.innerHTML=y},l(M){l=i(M,"P",{"data-svelte-h":!0}),c(l)!=="svelte-fincs2"&&(l.innerHTML=y)},m(M,b){o(M,l,b)},p:Ft,d(M){M&&n(l)}}}function js(U){let l,y="Examples:",M,b,v;return b=new W({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Qcm9jZXNzb3IlMkMlMjBNdXNpY2dlbkZvckNvbmRpdGlvbmFsR2VuZXJhdGlvbiUwQWltcG9ydCUyMHRvcmNoJTBBJTBBcHJvY2Vzc29yJTIwJTNEJTIwQXV0b1Byb2Nlc3Nvci5mcm9tX3ByZXRyYWluZWQoJTIyZmFjZWJvb2slMkZtdXNpY2dlbi1zbWFsbCUyMiklMEFtb2RlbCUyMCUzRCUyME11c2ljZ2VuRm9yQ29uZGl0aW9uYWxHZW5lcmF0aW9uLmZyb21fcHJldHJhaW5lZCglMjJmYWNlYm9vayUyRm11c2ljZ2VuLXNtYWxsJTIyKSUwQSUwQWlucHV0cyUyMCUzRCUyMHByb2Nlc3NvciglMEElMjAlMjAlMjAlMjB0ZXh0JTNEJTVCJTIyODBzJTIwcG9wJTIwdHJhY2slMjB3aXRoJTIwYmFzc3klMjBkcnVtcyUyMGFuZCUyMHN5bnRoJTIyJTJDJTIwJTIyOTBzJTIwcm9jayUyMHNvbmclMjB3aXRoJTIwbG91ZCUyMGd1aXRhcnMlMjBhbmQlMjBoZWF2eSUyMGRydW1zJTIyJTVEJTJDJTBBJTIwJTIwJTIwJTIwcGFkZGluZyUzRFRydWUlMkMlMEElMjAlMjAlMjAlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyJTJDJTBBKSUwQSUwQXBhZF90b2tlbl9pZCUyMCUzRCUyMG1vZGVsLmdlbmVyYXRpb25fY29uZmlnLnBhZF90b2tlbl9pZCUwQWRlY29kZXJfaW5wdXRfaWRzJTIwJTNEJTIwKCUwQSUyMCUyMCUyMCUyMHRvcmNoLm9uZXMoKGlucHV0cy5pbnB1dF9pZHMuc2hhcGUlNUIwJTVEJTIwKiUyMG1vZGVsLmRlY29kZXIubnVtX2NvZGVib29rcyUyQyUyMDEpJTJDJTIwZHR5cGUlM0R0b3JjaC5sb25nKSUwQSUyMCUyMCUyMCUyMColMjBwYWRfdG9rZW5faWQlMEEpJTBBJTBBbG9naXRzJTIwJTNEJTIwbW9kZWwoKippbnB1dHMlMkMlMjBkZWNvZGVyX2lucHV0X2lkcyUzRGRlY29kZXJfaW5wdXRfaWRzKS5sb2dpdHMlMEFsb2dpdHMuc2hhcGUlMjAlMjAlMjMlMjAoYnN6JTIwKiUyMG51bV9jb2RlYm9va3MlMkMlMjB0Z3RfbGVuJTJDJTIwdm9jYWJfc2l6ZSk=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoProcessor, MusicgenForConditionalGeneration
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> torch

<span class="hljs-meta">&gt;&gt;&gt; </span>processor = AutoProcessor.from_pretrained(<span class="hljs-string">&quot;facebook/musicgen-small&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = MusicgenForConditionalGeneration.from_pretrained(<span class="hljs-string">&quot;facebook/musicgen-small&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = processor(
<span class="hljs-meta">... </span>    text=[<span class="hljs-string">&quot;80s pop track with bassy drums and synth&quot;</span>, <span class="hljs-string">&quot;90s rock song with loud guitars and heavy drums&quot;</span>],
<span class="hljs-meta">... </span>    padding=<span class="hljs-literal">True</span>,
<span class="hljs-meta">... </span>    return_tensors=<span class="hljs-string">&quot;pt&quot;</span>,
<span class="hljs-meta">... </span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>pad_token_id = model.generation_config.pad_token_id
<span class="hljs-meta">&gt;&gt;&gt; </span>decoder_input_ids = (
<span class="hljs-meta">... </span>    torch.ones((inputs.input_ids.shape[<span class="hljs-number">0</span>] * model.decoder.num_codebooks, <span class="hljs-number">1</span>), dtype=torch.long)
<span class="hljs-meta">... </span>    * pad_token_id
<span class="hljs-meta">... </span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>logits = model(**inputs, decoder_input_ids=decoder_input_ids).logits
<span class="hljs-meta">&gt;&gt;&gt; </span>logits.shape  <span class="hljs-comment"># (bsz * num_codebooks, tgt_len, vocab_size)</span>
torch.Size([<span class="hljs-number">8</span>, <span class="hljs-number">1</span>, <span class="hljs-number">2048</span>])`,wrap:!1}}),{c(){l=r("p"),l.textContent=y,M=s(),m(b.$$.fragment)},l(p){l=i(p,"P",{"data-svelte-h":!0}),c(l)!=="svelte-kvfsh7"&&(l.textContent=y),M=a(p),u(b.$$.fragment,p)},m(p,T){o(p,l,T),o(p,M,T),h(b,p,T),v=!0},p:Ft,i(p){v||(g(b.$$.fragment,p),v=!0)},o(p){f(b.$$.fragment,p),v=!1},d(p){p&&(n(l),n(M)),_(b,p)}}}function Js(U){let l,y,M,b,v,p="<em>This model was released on 2023-06-08 and added to Hugging Face Transformers on 2023-06-29.</em>",T,te,It,Y,To='<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/> <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat"/> <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/>',Nt,ne,zt,oe,wo=`The MusicGen model was proposed in the paper <a href="https://huggingface.co/papers/2306.05284" rel="nofollow">Simple and Controllable Music Generation</a>
by Jade Copet, Felix Kreuk, Itai Gat, Tal Remez, David Kant, Gabriel Synnaeve, Yossi Adi and Alexandre Défossez.`,Xt,se,ko=`MusicGen is a single stage auto-regressive Transformer model capable of generating high-quality music samples conditioned
on text descriptions or audio prompts. The text descriptions are passed through a frozen text encoder model to obtain a
sequence of hidden-state representations. MusicGen is then trained to predict discrete audio tokens, or <em>audio codes</em>,
conditioned on these hidden-states. These audio tokens are then decoded using an audio compression model, such as EnCodec,
to recover the audio waveform.`,Bt,ae,Co=`Through an efficient token interleaving pattern, MusicGen does not require a self-supervised semantic representation of
the text/audio prompts, thus eliminating the need to cascade multiple models to predict a set of codebooks (e.g.
hierarchically or upsampling). Instead, it is able to generate all the codebooks in a single forward pass.`,Rt,re,jo="The abstract from the paper is the following:",qt,ie,Jo=`<em>We tackle the task of conditional music generation. We introduce MusicGen, a single Language Model (LM) that operates
over several streams of compressed discrete music representation, i.e., tokens. Unlike prior work, MusicGen is comprised
of a single-stage transformer LM together with efficient token interleaving patterns, which eliminates the need for
cascading several models, e.g., hierarchically or upsampling. Following this approach, we demonstrate how MusicGen
can generate high-quality samples, while being conditioned on textual description or melodic features, allowing better
controls over the generated output. We conduct extensive empirical evaluation, considering both automatic and human
studies, showing the proposed approach is superior to the evaluated baselines on a standard text-to-music benchmark.
Through ablation studies, we shed light over the importance of each of the components comprising MusicGen.</em>`,Vt,le,Uo=`This model was contributed by <a href="https://huggingface.co/sanchit-gandhi" rel="nofollow">sanchit-gandhi</a>. The original code can be found
<a href="https://github.com/facebookresearch/audiocraft" rel="nofollow">here</a>. The pre-trained checkpoints can be found on the
<a href="https://huggingface.co/models?sort=downloads&amp;search=facebook%2Fmusicgen-" rel="nofollow">Hugging Face Hub</a>.`,Et,de,Yt,ce,$o=`<li>After downloading the original checkpoints from <a href="https://github.com/facebookresearch/audiocraft/blob/main/docs/MUSICGEN.md#importing--exporting-models" rel="nofollow">here</a> , you can convert them using the <strong>conversion script</strong> available at
<code>src/transformers/models/musicgen/convert_musicgen_transformers.py</code> with the following command:</li>`,Ht,pe,Lt,me,Zo=`<p>[!NOTE]
The <code>head_mask</code> argument is ignored when using all attention implementation other than “eager”. If you have a <code>head_mask</code> and want it to have effect, load the model with <code>XXXModel.from_pretrained(model_id, attn_implementation=&quot;eager&quot;)</code></p>`,Qt,ue,Pt,he,Go=`MusicGen is compatible with two generation modes: greedy and sampling. In practice, sampling leads to significantly
better results than greedy, thus we encourage sampling mode to be used where possible. Sampling is enabled by default,
and can be explicitly specified by setting <code>do_sample=True</code> in the call to <code>MusicgenForConditionalGeneration.generate()</code>,
or by overriding the model’s generation config (see below).`,St,ge,xo=`Generation is limited by the sinusoidal positional embeddings to 30 second inputs. Meaning, MusicGen cannot generate more
than 30 seconds of audio (1503 tokens), and input audio passed by Audio-Prompted Generation contributes to this limit so,
given an input of 20 seconds of audio, MusicGen cannot generate more than 10 seconds of additional audio.`,Dt,fe,Fo=`Transformers supports both mono (1-channel) and stereo (2-channel) variants of MusicGen. The mono channel versions
generate a single set of codebooks. The stereo versions generate 2 sets of codebooks, 1 for each channel (left/right),
and each set of codebooks is decoded independently through the audio compression model. The audio streams for each
channel are combined to give the final stereo output.`,At,_e,Ot,Me,Wo=`The inputs for unconditional (or ‘null’) generation can be obtained through the method
<code>MusicgenForConditionalGeneration.get_unconditional_inputs()</code>:`,Kt,be,en,ye,Io=`The audio outputs are a three-dimensional Torch tensor of shape <code>(batch_size, num_channels, sequence_length)</code>. To listen
to the generated audio samples, you can either play them in an ipynb notebook:`,tn,ve,nn,Te,No="Or save them as a <code>.wav</code> file using a third-party library, e.g. <code>scipy</code>:",on,we,sn,ke,an,Ce,zo=`The model can generate an audio sample conditioned on a text prompt through use of the <a href="/docs/transformers/v4.56.2/en/model_doc/musicgen#transformers.MusicgenProcessor">MusicgenProcessor</a> to pre-process
the inputs:`,rn,je,ln,Je,Xo=`The <code>guidance_scale</code> is used in classifier free guidance (CFG), setting the weighting between the conditional logits
(which are predicted from the text prompts) and the unconditional logits (which are predicted from an unconditional or
‘null’ prompt). Higher guidance scale encourages the model to generate samples that are more closely linked to the input
prompt, usually at the expense of poorer audio quality. CFG is enabled by setting <code>guidance_scale &gt; 1</code>. For best results,
use <code>guidance_scale=3</code> (default).`,dn,Ue,cn,$e,Bo=`The same <a href="/docs/transformers/v4.56.2/en/model_doc/musicgen#transformers.MusicgenProcessor">MusicgenProcessor</a> can be used to pre-process an audio prompt that is used for audio continuation. In the
following example, we load an audio file using the 🤗 Datasets library, which can be pip installed through the command
below:`,pn,Ze,mn,Ge,un,xe,Ro=`For batched audio-prompted generation, the generated <code>audio_values</code> can be post-processed to remove padding by using the
<a href="/docs/transformers/v4.56.2/en/model_doc/musicgen#transformers.MusicgenProcessor">MusicgenProcessor</a> class:`,hn,Fe,gn,We,fn,Ie,qo=`The default parameters that control the generation process, such as sampling, guidance scale and number of generated
tokens, can be found in the model’s generation config, and updated as desired:`,_n,Ne,Mn,ze,Vo=`Note that any arguments passed to the generate method will <strong>supersede</strong> those in the generation config, so setting
<code>do_sample=False</code> in the call to generate will supersede the setting of <code>model.generation_config.do_sample</code> in the
generation config.`,bn,Xe,yn,Be,Eo="The MusicGen model can be de-composed into three distinct stages:",vn,Re,Yo="<li>Text encoder: maps the text inputs to a sequence of hidden-state representations. The pre-trained MusicGen models use a frozen text encoder from either T5 or Flan-T5</li> <li>MusicGen decoder: a language model (LM) that auto-regressively generates audio tokens (or codes) conditional on the encoder hidden-state representations</li> <li>Audio encoder/decoder: used to encode an audio prompt to use as prompt tokens, and recover the audio waveform from the audio tokens predicted by the decoder</li>",Tn,qe,Ho=`Thus, the MusicGen model can either be used as a standalone decoder model, corresponding to the class <a href="/docs/transformers/v4.56.2/en/model_doc/musicgen#transformers.MusicgenForCausalLM">MusicgenForCausalLM</a>,
or as a composite model that includes the text encoder and audio encoder/decoder, corresponding to the class
<a href="/docs/transformers/v4.56.2/en/model_doc/musicgen#transformers.MusicgenForConditionalGeneration">MusicgenForConditionalGeneration</a>. If only the decoder needs to be loaded from the pre-trained checkpoint, it can be loaded by first
specifying the correct config, or be accessed through the <code>.decoder</code> attribute of the composite model:`,wn,Ve,kn,Ee,Lo=`Since the text encoder and audio encoder/decoder models are frozen during training, the MusicGen decoder <a href="/docs/transformers/v4.56.2/en/model_doc/musicgen#transformers.MusicgenForCausalLM">MusicgenForCausalLM</a>
can be trained standalone on a dataset of encoder hidden-states and audio codes. For inference, the trained decoder can
be combined with the frozen text encoder and audio encoder/decoders to recover the composite <a href="/docs/transformers/v4.56.2/en/model_doc/musicgen#transformers.MusicgenForConditionalGeneration">MusicgenForConditionalGeneration</a>
model.`,Cn,Ye,Qo="Tips:",jn,He,Po="<li>MusicGen is trained on the 32kHz checkpoint of Encodec. You should ensure you use a compatible version of the Encodec model.</li> <li>Sampling mode tends to deliver better results than greedy - you can toggle sampling with the variable <code>do_sample</code> in the call to <code>MusicgenForConditionalGeneration.generate()</code></li>",Jn,Le,Un,z,Qe,Vn,mt,So=`This is the configuration class to store the configuration of an <code>MusicgenDecoder</code>. It is used to instantiate a
MusicGen decoder according to the specified arguments, defining the model architecture. Instantiating a
configuration with the defaults will yield a similar configuration to that of the MusicGen
<a href="https://huggingface.co/facebook/musicgen-small" rel="nofollow">facebook/musicgen-small</a> architecture.`,En,ut,Do=`Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> and can be used to control the model outputs. Read the
documentation from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> for more information.`,$n,Pe,Zn,w,Se,Yn,ht,Ao=`This is the configuration class to store the configuration of a <a href="/docs/transformers/v4.56.2/en/model_doc/musicgen#transformers.MusicgenModel">MusicgenModel</a>. It is used to instantiate a
MusicGen model according to the specified arguments, defining the text encoder, audio encoder and MusicGen decoder
configs.`,Hn,gt,Oo=`Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> and can be used to control the model outputs. Read the
documentation from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> for more information.`,Ln,H,Qn,L,De,Pn,ft,Ko=`Instantiate a <a href="/docs/transformers/v4.56.2/en/model_doc/musicgen#transformers.MusicgenConfig">MusicgenConfig</a> (or a derived class) from text encoder, audio encoder and decoder
configurations.`,Gn,Ae,xn,Z,Oe,Sn,_t,es=`Constructs a MusicGen processor which wraps an EnCodec feature extractor and a T5 tokenizer into a single processor
class.`,Dn,Mt,ts=`<a href="/docs/transformers/v4.56.2/en/model_doc/musicgen#transformers.MusicgenProcessor">MusicgenProcessor</a> offers all the functionalities of <a href="/docs/transformers/v4.56.2/en/model_doc/encodec#transformers.EncodecFeatureExtractor">EncodecFeatureExtractor</a> and <code>TTokenizer</code>. See
<code>__call__()</code> and <a href="/docs/transformers/v4.56.2/en/main_classes/processors#transformers.ProcessorMixin.decode">decode()</a> for more information.`,An,Q,Ke,On,bt,ns=`This method is used to decode either batches of audio outputs from the MusicGen model, or batches of token ids
from the tokenizer. In the case of decoding token ids, this method forwards all its arguments to T5Tokenizer’s
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.batch_decode">batch_decode()</a>. Please refer to the docstring of this method for more information.`,Fn,et,Wn,k,tt,Kn,yt,os="The bare Musicgen Model outputting raw hidden-states without any specific head on top.",eo,vt,ss=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,to,Tt,as=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,no,R,nt,oo,wt,rs='The <a href="/docs/transformers/v4.56.2/en/model_doc/musicgen#transformers.MusicgenModel">MusicgenModel</a> forward method, overrides the <code>__call__</code> special method.',so,P,In,ot,Nn,C,st,ao,kt,is="The MusicGen decoder model with a language modelling head on top.",ro,Ct,ls=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,io,jt,ds=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,lo,I,at,co,Jt,cs='The <a href="/docs/transformers/v4.56.2/en/model_doc/musicgen#transformers.MusicgenForCausalLM">MusicgenForCausalLM</a> forward method, overrides the <code>__call__</code> special method.',po,S,mo,D,zn,rt,Xn,j,it,uo,Ut,ps="The composite MusicGen model with a text encoder, audio encoder and Musicgen decoder,",ho,$t,ms=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,go,Zt,us=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,fo,N,lt,_o,Gt,hs='The <a href="/docs/transformers/v4.56.2/en/model_doc/musicgen#transformers.MusicgenForConditionalGeneration">MusicgenForConditionalGeneration</a> forward method, overrides the <code>__call__</code> special method.',Mo,A,bo,O,Bn,dt,Rn,Wt,qn;return te=new J({props:{title:"MusicGen",local:"musicgen",headingTag:"h1"}}),ne=new J({props:{title:"Overview",local:"overview",headingTag:"h2"}}),de=new J({props:{title:"Usage tips",local:"usage-tips",headingTag:"h2"}}),pe=new W({props:{code:"cHl0aG9uJTIwc3JjJTJGdHJhbnNmb3JtZXJzJTJGbW9kZWxzJTJGbXVzaWNnZW4lMkZjb252ZXJ0X211c2ljZ2VuX3RyYW5zZm9ybWVycy5weSUyMCU1QyUwQSUyMCUyMCUyMCUyMC0tY2hlY2twb2ludCUyMHNtYWxsJTIwLS1weXRvcmNoX2R1bXBfZm9sZGVyJTIwJTJGb3V0cHV0JTJGcGF0aCUyMC0tc2FmZV9zZXJpYWxpemF0aW9uJTIw",highlighted:`python src/transformers/models/musicgen/convert_musicgen_transformers.py \\
    --checkpoint small --pytorch_dump_folder /output/path --safe_serialization `,wrap:!1}}),ue=new J({props:{title:"Generation",local:"generation",headingTag:"h2"}}),_e=new J({props:{title:"Unconditional Generation",local:"unconditional-generation",headingTag:"h3"}}),be=new W({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyME11c2ljZ2VuRm9yQ29uZGl0aW9uYWxHZW5lcmF0aW9uJTBBJTBBbW9kZWwlMjAlM0QlMjBNdXNpY2dlbkZvckNvbmRpdGlvbmFsR2VuZXJhdGlvbi5mcm9tX3ByZXRyYWluZWQoJTIyZmFjZWJvb2slMkZtdXNpY2dlbi1zbWFsbCUyMiklMEF1bmNvbmRpdGlvbmFsX2lucHV0cyUyMCUzRCUyMG1vZGVsLmdldF91bmNvbmRpdGlvbmFsX2lucHV0cyhudW1fc2FtcGxlcyUzRDEpJTBBJTBBYXVkaW9fdmFsdWVzJTIwJTNEJTIwbW9kZWwuZ2VuZXJhdGUoKip1bmNvbmRpdGlvbmFsX2lucHV0cyUyQyUyMGRvX3NhbXBsZSUzRFRydWUlMkMlMjBtYXhfbmV3X3Rva2VucyUzRDI1Nik=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> MusicgenForConditionalGeneration

<span class="hljs-meta">&gt;&gt;&gt; </span>model = MusicgenForConditionalGeneration.from_pretrained(<span class="hljs-string">&quot;facebook/musicgen-small&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>unconditional_inputs = model.get_unconditional_inputs(num_samples=<span class="hljs-number">1</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>audio_values = model.generate(**unconditional_inputs, do_sample=<span class="hljs-literal">True</span>, max_new_tokens=<span class="hljs-number">256</span>)`,wrap:!1}}),ve=new W({props:{code:"ZnJvbSUyMElQeXRob24uZGlzcGxheSUyMGltcG9ydCUyMEF1ZGlvJTBBJTBBc2FtcGxpbmdfcmF0ZSUyMCUzRCUyMG1vZGVsLmNvbmZpZy5hdWRpb19lbmNvZGVyLnNhbXBsaW5nX3JhdGUlMEFBdWRpbyhhdWRpb192YWx1ZXMlNUIwJTVELm51bXB5KCklMkMlMjByYXRlJTNEc2FtcGxpbmdfcmF0ZSk=",highlighted:`<span class="hljs-keyword">from</span> IPython.display <span class="hljs-keyword">import</span> Audio

sampling_rate = model.config.audio_encoder.sampling_rate
Audio(audio_values[<span class="hljs-number">0</span>].numpy(), rate=sampling_rate)`,wrap:!1}}),we=new W({props:{code:"aW1wb3J0JTIwc2NpcHklMEElMEFzYW1wbGluZ19yYXRlJTIwJTNEJTIwbW9kZWwuY29uZmlnLmF1ZGlvX2VuY29kZXIuc2FtcGxpbmdfcmF0ZSUwQXNjaXB5LmlvLndhdmZpbGUud3JpdGUoJTIybXVzaWNnZW5fb3V0LndhdiUyMiUyQyUyMHJhdGUlM0RzYW1wbGluZ19yYXRlJTJDJTIwZGF0YSUzRGF1ZGlvX3ZhbHVlcyU1QjAlMkMlMjAwJTVELm51bXB5KCkp",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">import</span> scipy

<span class="hljs-meta">&gt;&gt;&gt; </span>sampling_rate = model.config.audio_encoder.sampling_rate
<span class="hljs-meta">&gt;&gt;&gt; </span>scipy.io.wavfile.write(<span class="hljs-string">&quot;musicgen_out.wav&quot;</span>, rate=sampling_rate, data=audio_values[<span class="hljs-number">0</span>, <span class="hljs-number">0</span>].numpy())`,wrap:!1}}),ke=new J({props:{title:"Text-Conditional Generation",local:"text-conditional-generation",headingTag:"h3"}}),je=new W({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Qcm9jZXNzb3IlMkMlMjBNdXNpY2dlbkZvckNvbmRpdGlvbmFsR2VuZXJhdGlvbiUwQSUwQXByb2Nlc3NvciUyMCUzRCUyMEF1dG9Qcm9jZXNzb3IuZnJvbV9wcmV0cmFpbmVkKCUyMmZhY2Vib29rJTJGbXVzaWNnZW4tc21hbGwlMjIpJTBBbW9kZWwlMjAlM0QlMjBNdXNpY2dlbkZvckNvbmRpdGlvbmFsR2VuZXJhdGlvbi5mcm9tX3ByZXRyYWluZWQoJTIyZmFjZWJvb2slMkZtdXNpY2dlbi1zbWFsbCUyMiklMEElMEFpbnB1dHMlMjAlM0QlMjBwcm9jZXNzb3IoJTBBJTIwJTIwJTIwJTIwdGV4dCUzRCU1QiUyMjgwcyUyMHBvcCUyMHRyYWNrJTIwd2l0aCUyMGJhc3N5JTIwZHJ1bXMlMjBhbmQlMjBzeW50aCUyMiUyQyUyMCUyMjkwcyUyMHJvY2slMjBzb25nJTIwd2l0aCUyMGxvdWQlMjBndWl0YXJzJTIwYW5kJTIwaGVhdnklMjBkcnVtcyUyMiU1RCUyQyUwQSUyMCUyMCUyMCUyMHBhZGRpbmclM0RUcnVlJTJDJTBBJTIwJTIwJTIwJTIwcmV0dXJuX3RlbnNvcnMlM0QlMjJwdCUyMiUyQyUwQSklMEFhdWRpb192YWx1ZXMlMjAlM0QlMjBtb2RlbC5nZW5lcmF0ZSgqKmlucHV0cyUyQyUyMGRvX3NhbXBsZSUzRFRydWUlMkMlMjBndWlkYW5jZV9zY2FsZSUzRDMlMkMlMjBtYXhfbmV3X3Rva2VucyUzRDI1Nik=",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoProcessor, MusicgenForConditionalGeneration

<span class="hljs-meta">&gt;&gt;&gt; </span>processor = AutoProcessor.from_pretrained(<span class="hljs-string">&quot;facebook/musicgen-small&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = MusicgenForConditionalGeneration.from_pretrained(<span class="hljs-string">&quot;facebook/musicgen-small&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = processor(
<span class="hljs-meta">... </span>    text=[<span class="hljs-string">&quot;80s pop track with bassy drums and synth&quot;</span>, <span class="hljs-string">&quot;90s rock song with loud guitars and heavy drums&quot;</span>],
<span class="hljs-meta">... </span>    padding=<span class="hljs-literal">True</span>,
<span class="hljs-meta">... </span>    return_tensors=<span class="hljs-string">&quot;pt&quot;</span>,
<span class="hljs-meta">... </span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>audio_values = model.generate(**inputs, do_sample=<span class="hljs-literal">True</span>, guidance_scale=<span class="hljs-number">3</span>, max_new_tokens=<span class="hljs-number">256</span>)`,wrap:!1}}),Ue=new J({props:{title:"Audio-Prompted Generation",local:"audio-prompted-generation",headingTag:"h3"}}),Ze=new W({props:{code:"cGlwJTIwaW5zdGFsbCUyMC0tdXBncmFkZSUyMHBpcCUwQXBpcCUyMGluc3RhbGwlMjBkYXRhc2V0cyU1QmF1ZGlvJTVE",highlighted:`pip install --upgrade pip
pip install datasets[audio]`,wrap:!1}}),Ge=new W({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Qcm9jZXNzb3IlMkMlMjBNdXNpY2dlbkZvckNvbmRpdGlvbmFsR2VuZXJhdGlvbiUwQWZyb20lMjBkYXRhc2V0cyUyMGltcG9ydCUyMGxvYWRfZGF0YXNldCUwQSUwQXByb2Nlc3NvciUyMCUzRCUyMEF1dG9Qcm9jZXNzb3IuZnJvbV9wcmV0cmFpbmVkKCUyMmZhY2Vib29rJTJGbXVzaWNnZW4tc21hbGwlMjIpJTBBbW9kZWwlMjAlM0QlMjBNdXNpY2dlbkZvckNvbmRpdGlvbmFsR2VuZXJhdGlvbi5mcm9tX3ByZXRyYWluZWQoJTIyZmFjZWJvb2slMkZtdXNpY2dlbi1zbWFsbCUyMiklMEElMEFkYXRhc2V0JTIwJTNEJTIwbG9hZF9kYXRhc2V0KCUyMnNhbmNoaXQtZ2FuZGhpJTJGZ3R6YW4lMjIlMkMlMjBzcGxpdCUzRCUyMnRyYWluJTIyJTJDJTIwc3RyZWFtaW5nJTNEVHJ1ZSklMEFzYW1wbGUlMjAlM0QlMjBuZXh0KGl0ZXIoZGF0YXNldCkpJTVCJTIyYXVkaW8lMjIlNUQlMEElMEElMjMlMjB0YWtlJTIwdGhlJTIwZmlyc3QlMjBoYWxmJTIwb2YlMjB0aGUlMjBhdWRpbyUyMHNhbXBsZSUwQXNhbXBsZSU1QiUyMmFycmF5JTIyJTVEJTIwJTNEJTIwc2FtcGxlJTVCJTIyYXJyYXklMjIlNUQlNUIlM0ElMjBsZW4oc2FtcGxlJTVCJTIyYXJyYXklMjIlNUQpJTIwJTJGJTJGJTIwMiU1RCUwQSUwQWlucHV0cyUyMCUzRCUyMHByb2Nlc3NvciglMEElMjAlMjAlMjAlMjBhdWRpbyUzRHNhbXBsZSU1QiUyMmFycmF5JTIyJTVEJTJDJTBBJTIwJTIwJTIwJTIwc2FtcGxpbmdfcmF0ZSUzRHNhbXBsZSU1QiUyMnNhbXBsaW5nX3JhdGUlMjIlNUQlMkMlMEElMjAlMjAlMjAlMjB0ZXh0JTNEJTVCJTIyODBzJTIwYmx1ZXMlMjB0cmFjayUyMHdpdGglMjBncm9vdnklMjBzYXhvcGhvbmUlMjIlNUQlMkMlMEElMjAlMjAlMjAlMjBwYWRkaW5nJTNEVHJ1ZSUyQyUwQSUyMCUyMCUyMCUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIlMkMlMEEpJTBBYXVkaW9fdmFsdWVzJTIwJTNEJTIwbW9kZWwuZ2VuZXJhdGUoKippbnB1dHMlMkMlMjBkb19zYW1wbGUlM0RUcnVlJTJDJTIwZ3VpZGFuY2Vfc2NhbGUlM0QzJTJDJTIwbWF4X25ld190b2tlbnMlM0QyNTYp",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoProcessor, MusicgenForConditionalGeneration
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> datasets <span class="hljs-keyword">import</span> load_dataset

<span class="hljs-meta">&gt;&gt;&gt; </span>processor = AutoProcessor.from_pretrained(<span class="hljs-string">&quot;facebook/musicgen-small&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = MusicgenForConditionalGeneration.from_pretrained(<span class="hljs-string">&quot;facebook/musicgen-small&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>dataset = load_dataset(<span class="hljs-string">&quot;sanchit-gandhi/gtzan&quot;</span>, split=<span class="hljs-string">&quot;train&quot;</span>, streaming=<span class="hljs-literal">True</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>sample = <span class="hljs-built_in">next</span>(<span class="hljs-built_in">iter</span>(dataset))[<span class="hljs-string">&quot;audio&quot;</span>]

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># take the first half of the audio sample</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>sample[<span class="hljs-string">&quot;array&quot;</span>] = sample[<span class="hljs-string">&quot;array&quot;</span>][: <span class="hljs-built_in">len</span>(sample[<span class="hljs-string">&quot;array&quot;</span>]) // <span class="hljs-number">2</span>]

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = processor(
<span class="hljs-meta">... </span>    audio=sample[<span class="hljs-string">&quot;array&quot;</span>],
<span class="hljs-meta">... </span>    sampling_rate=sample[<span class="hljs-string">&quot;sampling_rate&quot;</span>],
<span class="hljs-meta">... </span>    text=[<span class="hljs-string">&quot;80s blues track with groovy saxophone&quot;</span>],
<span class="hljs-meta">... </span>    padding=<span class="hljs-literal">True</span>,
<span class="hljs-meta">... </span>    return_tensors=<span class="hljs-string">&quot;pt&quot;</span>,
<span class="hljs-meta">... </span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>audio_values = model.generate(**inputs, do_sample=<span class="hljs-literal">True</span>, guidance_scale=<span class="hljs-number">3</span>, max_new_tokens=<span class="hljs-number">256</span>)`,wrap:!1}}),Fe=new W({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Qcm9jZXNzb3IlMkMlMjBNdXNpY2dlbkZvckNvbmRpdGlvbmFsR2VuZXJhdGlvbiUwQWZyb20lMjBkYXRhc2V0cyUyMGltcG9ydCUyMGxvYWRfZGF0YXNldCUwQSUwQXByb2Nlc3NvciUyMCUzRCUyMEF1dG9Qcm9jZXNzb3IuZnJvbV9wcmV0cmFpbmVkKCUyMmZhY2Vib29rJTJGbXVzaWNnZW4tc21hbGwlMjIpJTBBbW9kZWwlMjAlM0QlMjBNdXNpY2dlbkZvckNvbmRpdGlvbmFsR2VuZXJhdGlvbi5mcm9tX3ByZXRyYWluZWQoJTIyZmFjZWJvb2slMkZtdXNpY2dlbi1zbWFsbCUyMiklMEElMEFkYXRhc2V0JTIwJTNEJTIwbG9hZF9kYXRhc2V0KCUyMnNhbmNoaXQtZ2FuZGhpJTJGZ3R6YW4lMjIlMkMlMjBzcGxpdCUzRCUyMnRyYWluJTIyJTJDJTIwc3RyZWFtaW5nJTNEVHJ1ZSklMEFzYW1wbGUlMjAlM0QlMjBuZXh0KGl0ZXIoZGF0YXNldCkpJTVCJTIyYXVkaW8lMjIlNUQlMEElMEElMjMlMjB0YWtlJTIwdGhlJTIwZmlyc3QlMjBxdWFydGVyJTIwb2YlMjB0aGUlMjBhdWRpbyUyMHNhbXBsZSUwQXNhbXBsZV8xJTIwJTNEJTIwc2FtcGxlJTVCJTIyYXJyYXklMjIlNUQlNUIlM0ElMjBsZW4oc2FtcGxlJTVCJTIyYXJyYXklMjIlNUQpJTIwJTJGJTJGJTIwNCU1RCUwQSUwQSUyMyUyMHRha2UlMjB0aGUlMjBmaXJzdCUyMGhhbGYlMjBvZiUyMHRoZSUyMGF1ZGlvJTIwc2FtcGxlJTBBc2FtcGxlXzIlMjAlM0QlMjBzYW1wbGUlNUIlMjJhcnJheSUyMiU1RCU1QiUzQSUyMGxlbihzYW1wbGUlNUIlMjJhcnJheSUyMiU1RCklMjAlMkYlMkYlMjAyJTVEJTBBJTBBaW5wdXRzJTIwJTNEJTIwcHJvY2Vzc29yKCUwQSUyMCUyMCUyMCUyMGF1ZGlvJTNEJTVCc2FtcGxlXzElMkMlMjBzYW1wbGVfMiU1RCUyQyUwQSUyMCUyMCUyMCUyMHNhbXBsaW5nX3JhdGUlM0RzYW1wbGUlNUIlMjJzYW1wbGluZ19yYXRlJTIyJTVEJTJDJTBBJTIwJTIwJTIwJTIwdGV4dCUzRCU1QiUyMjgwcyUyMGJsdWVzJTIwdHJhY2slMjB3aXRoJTIwZ3Jvb3Z5JTIwc2F4b3Bob25lJTIyJTJDJTIwJTIyOTBzJTIwcm9jayUyMHNvbmclMjB3aXRoJTIwbG91ZCUyMGd1aXRhcnMlMjBhbmQlMjBoZWF2eSUyMGRydW1zJTIyJTVEJTJDJTBBJTIwJTIwJTIwJTIwcGFkZGluZyUzRFRydWUlMkMlMEElMjAlMjAlMjAlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyJTJDJTBBKSUwQWF1ZGlvX3ZhbHVlcyUyMCUzRCUyMG1vZGVsLmdlbmVyYXRlKCoqaW5wdXRzJTJDJTIwZG9fc2FtcGxlJTNEVHJ1ZSUyQyUyMGd1aWRhbmNlX3NjYWxlJTNEMyUyQyUyMG1heF9uZXdfdG9rZW5zJTNEMjU2KSUwQSUwQSUyMyUyMHBvc3QtcHJvY2VzcyUyMHRvJTIwcmVtb3ZlJTIwcGFkZGluZyUyMGZyb20lMjB0aGUlMjBiYXRjaGVkJTIwYXVkaW8lMEFhdWRpb192YWx1ZXMlMjAlM0QlMjBwcm9jZXNzb3IuYmF0Y2hfZGVjb2RlKGF1ZGlvX3ZhbHVlcyUyQyUyMHBhZGRpbmdfbWFzayUzRGlucHV0cy5wYWRkaW5nX21hc2sp",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoProcessor, MusicgenForConditionalGeneration
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> datasets <span class="hljs-keyword">import</span> load_dataset

<span class="hljs-meta">&gt;&gt;&gt; </span>processor = AutoProcessor.from_pretrained(<span class="hljs-string">&quot;facebook/musicgen-small&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>model = MusicgenForConditionalGeneration.from_pretrained(<span class="hljs-string">&quot;facebook/musicgen-small&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span>dataset = load_dataset(<span class="hljs-string">&quot;sanchit-gandhi/gtzan&quot;</span>, split=<span class="hljs-string">&quot;train&quot;</span>, streaming=<span class="hljs-literal">True</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>sample = <span class="hljs-built_in">next</span>(<span class="hljs-built_in">iter</span>(dataset))[<span class="hljs-string">&quot;audio&quot;</span>]

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># take the first quarter of the audio sample</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>sample_1 = sample[<span class="hljs-string">&quot;array&quot;</span>][: <span class="hljs-built_in">len</span>(sample[<span class="hljs-string">&quot;array&quot;</span>]) // <span class="hljs-number">4</span>]

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># take the first half of the audio sample</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>sample_2 = sample[<span class="hljs-string">&quot;array&quot;</span>][: <span class="hljs-built_in">len</span>(sample[<span class="hljs-string">&quot;array&quot;</span>]) // <span class="hljs-number">2</span>]

<span class="hljs-meta">&gt;&gt;&gt; </span>inputs = processor(
<span class="hljs-meta">... </span>    audio=[sample_1, sample_2],
<span class="hljs-meta">... </span>    sampling_rate=sample[<span class="hljs-string">&quot;sampling_rate&quot;</span>],
<span class="hljs-meta">... </span>    text=[<span class="hljs-string">&quot;80s blues track with groovy saxophone&quot;</span>, <span class="hljs-string">&quot;90s rock song with loud guitars and heavy drums&quot;</span>],
<span class="hljs-meta">... </span>    padding=<span class="hljs-literal">True</span>,
<span class="hljs-meta">... </span>    return_tensors=<span class="hljs-string">&quot;pt&quot;</span>,
<span class="hljs-meta">... </span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>audio_values = model.generate(**inputs, do_sample=<span class="hljs-literal">True</span>, guidance_scale=<span class="hljs-number">3</span>, max_new_tokens=<span class="hljs-number">256</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># post-process to remove padding from the batched audio</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>audio_values = processor.batch_decode(audio_values, padding_mask=inputs.padding_mask)`,wrap:!1}}),We=new J({props:{title:"Generation Configuration",local:"generation-configuration",headingTag:"h3"}}),Ne=new W({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyME11c2ljZ2VuRm9yQ29uZGl0aW9uYWxHZW5lcmF0aW9uJTBBJTBBbW9kZWwlMjAlM0QlMjBNdXNpY2dlbkZvckNvbmRpdGlvbmFsR2VuZXJhdGlvbi5mcm9tX3ByZXRyYWluZWQoJTIyZmFjZWJvb2slMkZtdXNpY2dlbi1zbWFsbCUyMiklMEElMEElMjMlMjBpbnNwZWN0JTIwdGhlJTIwZGVmYXVsdCUyMGdlbmVyYXRpb24lMjBjb25maWclMEFtb2RlbC5nZW5lcmF0aW9uX2NvbmZpZyUwQSUwQSUyMyUyMGluY3JlYXNlJTIwdGhlJTIwZ3VpZGFuY2UlMjBzY2FsZSUyMHRvJTIwNC4wJTBBbW9kZWwuZ2VuZXJhdGlvbl9jb25maWcuZ3VpZGFuY2Vfc2NhbGUlMjAlM0QlMjA0LjAlMEElMEElMjMlMjBkZWNyZWFzZSUyMHRoZSUyMG1heCUyMGxlbmd0aCUyMHRvJTIwMjU2JTIwdG9rZW5zJTBBbW9kZWwuZ2VuZXJhdGlvbl9jb25maWcubWF4X2xlbmd0aCUyMCUzRCUyMDI1Ng==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> MusicgenForConditionalGeneration

<span class="hljs-meta">&gt;&gt;&gt; </span>model = MusicgenForConditionalGeneration.from_pretrained(<span class="hljs-string">&quot;facebook/musicgen-small&quot;</span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># inspect the default generation config</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model.generation_config

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># increase the guidance scale to 4.0</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model.generation_config.guidance_scale = <span class="hljs-number">4.0</span>

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># decrease the max length to 256 tokens</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model.generation_config.max_length = <span class="hljs-number">256</span>`,wrap:!1}}),Xe=new J({props:{title:"Model Structure",local:"model-structure",headingTag:"h2"}}),Ve=new W({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Db25maWclMkMlMjBNdXNpY2dlbkZvckNhdXNhbExNJTJDJTIwTXVzaWNnZW5Gb3JDb25kaXRpb25hbEdlbmVyYXRpb24lMEElMEElMjMlMjBPcHRpb24lMjAxJTNBJTIwZ2V0JTIwZGVjb2RlciUyMGNvbmZpZyUyMGFuZCUyMHBhc3MlMjB0byUyMCU2MC5mcm9tX3ByZXRyYWluZWQlNjAlMEFkZWNvZGVyX2NvbmZpZyUyMCUzRCUyMEF1dG9Db25maWcuZnJvbV9wcmV0cmFpbmVkKCUyMmZhY2Vib29rJTJGbXVzaWNnZW4tc21hbGwlMjIpLmRlY29kZXIlMEFkZWNvZGVyJTIwJTNEJTIwTXVzaWNnZW5Gb3JDYXVzYWxMTS5mcm9tX3ByZXRyYWluZWQoJTIyZmFjZWJvb2slMkZtdXNpY2dlbi1zbWFsbCUyMiUyQyUyMCoqZGVjb2Rlcl9jb25maWcpJTBBJTBBJTIzJTIwT3B0aW9uJTIwMiUzQSUyMGxvYWQlMjB0aGUlMjBlbnRpcmUlMjBjb21wb3NpdGUlMjBtb2RlbCUyQyUyMGJ1dCUyMG9ubHklMjByZXR1cm4lMjB0aGUlMjBkZWNvZGVyJTBBZGVjb2RlciUyMCUzRCUyME11c2ljZ2VuRm9yQ29uZGl0aW9uYWxHZW5lcmF0aW9uLmZyb21fcHJldHJhaW5lZCglMjJmYWNlYm9vayUyRm11c2ljZ2VuLXNtYWxsJTIyKS5kZWNvZGVy",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoConfig, MusicgenForCausalLM, MusicgenForConditionalGeneration

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Option 1: get decoder config and pass to \`.from_pretrained\`</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>decoder_config = AutoConfig.from_pretrained(<span class="hljs-string">&quot;facebook/musicgen-small&quot;</span>).decoder
<span class="hljs-meta">&gt;&gt;&gt; </span>decoder = MusicgenForCausalLM.from_pretrained(<span class="hljs-string">&quot;facebook/musicgen-small&quot;</span>, **decoder_config)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Option 2: load the entire composite model, but only return the decoder</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>decoder = MusicgenForConditionalGeneration.from_pretrained(<span class="hljs-string">&quot;facebook/musicgen-small&quot;</span>).decoder`,wrap:!1}}),Le=new J({props:{title:"MusicgenDecoderConfig",local:"transformers.MusicgenDecoderConfig",headingTag:"h2"}}),Qe=new B({props:{name:"class transformers.MusicgenDecoderConfig",anchor:"transformers.MusicgenDecoderConfig",parameters:[{name:"vocab_size",val:" = 2048"},{name:"max_position_embeddings",val:" = 2048"},{name:"num_hidden_layers",val:" = 24"},{name:"ffn_dim",val:" = 4096"},{name:"num_attention_heads",val:" = 16"},{name:"layerdrop",val:" = 0.0"},{name:"use_cache",val:" = True"},{name:"activation_function",val:" = 'gelu'"},{name:"hidden_size",val:" = 1024"},{name:"dropout",val:" = 0.1"},{name:"attention_dropout",val:" = 0.0"},{name:"activation_dropout",val:" = 0.0"},{name:"initializer_factor",val:" = 0.02"},{name:"scale_embedding",val:" = False"},{name:"num_codebooks",val:" = 4"},{name:"audio_channels",val:" = 1"},{name:"pad_token_id",val:" = 2048"},{name:"bos_token_id",val:" = 2048"},{name:"eos_token_id",val:" = None"},{name:"tie_word_embeddings",val:" = False"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.MusicgenDecoderConfig.vocab_size",description:`<strong>vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 2048) &#x2014;
Vocabulary size of the MusicgenDecoder model. Defines the number of different tokens that can be
represented by the <code>inputs_ids</code> passed when calling <code>MusicgenDecoder</code>.`,name:"vocab_size"},{anchor:"transformers.MusicgenDecoderConfig.hidden_size",description:`<strong>hidden_size</strong> (<code>int</code>, <em>optional</em>, defaults to 1024) &#x2014;
Dimensionality of the layers and the pooler layer.`,name:"hidden_size"},{anchor:"transformers.MusicgenDecoderConfig.num_hidden_layers",description:`<strong>num_hidden_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 24) &#x2014;
Number of decoder layers.`,name:"num_hidden_layers"},{anchor:"transformers.MusicgenDecoderConfig.num_attention_heads",description:`<strong>num_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 16) &#x2014;
Number of attention heads for each attention layer in the Transformer block.`,name:"num_attention_heads"},{anchor:"transformers.MusicgenDecoderConfig.ffn_dim",description:`<strong>ffn_dim</strong> (<code>int</code>, <em>optional</em>, defaults to 4096) &#x2014;
Dimensionality of the &#x201C;intermediate&#x201D; (often named feed-forward) layer in the Transformer block.`,name:"ffn_dim"},{anchor:"transformers.MusicgenDecoderConfig.activation_function",description:`<strong>activation_function</strong> (<code>str</code> or <code>function</code>, <em>optional</em>, defaults to <code>&quot;gelu&quot;</code>) &#x2014;
The non-linear activation function (function or string) in the decoder and pooler. If string, <code>&quot;gelu&quot;</code>,
<code>&quot;relu&quot;</code>, <code>&quot;silu&quot;</code> and <code>&quot;gelu_new&quot;</code> are supported.`,name:"activation_function"},{anchor:"transformers.MusicgenDecoderConfig.dropout",description:`<strong>dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout probability for all fully connected layers in the embeddings, text_encoder, and pooler.`,name:"dropout"},{anchor:"transformers.MusicgenDecoderConfig.attention_dropout",description:`<strong>attention_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The dropout ratio for the attention probabilities.`,name:"attention_dropout"},{anchor:"transformers.MusicgenDecoderConfig.activation_dropout",description:`<strong>activation_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The dropout ratio for activations inside the fully connected layer.`,name:"activation_dropout"},{anchor:"transformers.MusicgenDecoderConfig.max_position_embeddings",description:`<strong>max_position_embeddings</strong> (<code>int</code>, <em>optional</em>, defaults to 2048) &#x2014;
The maximum sequence length that this model might ever be used with. Typically, set this to something large
just in case (e.g., 512 or 1024 or 2048).`,name:"max_position_embeddings"},{anchor:"transformers.MusicgenDecoderConfig.initializer_factor",description:`<strong>initializer_factor</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.`,name:"initializer_factor"},{anchor:"transformers.MusicgenDecoderConfig.layerdrop",description:`<strong>layerdrop</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The LayerDrop probability for the decoder. See the [LayerDrop paper](see <a href="https://huggingface.co/papers/1909.11556" rel="nofollow">https://huggingface.co/papers/1909.11556</a>)
for more details.`,name:"layerdrop"},{anchor:"transformers.MusicgenDecoderConfig.scale_embedding",description:`<strong>scale_embedding</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Scale embeddings by diving by sqrt(hidden_size).`,name:"scale_embedding"},{anchor:"transformers.MusicgenDecoderConfig.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether the model should return the last key/values attentions (not used by all models)`,name:"use_cache"},{anchor:"transformers.MusicgenDecoderConfig.num_codebooks",description:`<strong>num_codebooks</strong> (<code>int</code>, <em>optional</em>, defaults to 4) &#x2014;
The number of parallel codebooks forwarded to the model.`,name:"num_codebooks"},{anchor:"transformers.MusicgenDecoderConfig.tie_word_embeddings(bool,",description:`<strong>tie_word_embeddings(<code>bool</code>,</strong> <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether input and output word embeddings should be tied.`,name:"tie_word_embeddings(bool,"},{anchor:"transformers.MusicgenDecoderConfig.audio_channels",description:`<strong>audio_channels</strong> (<code>int</code>, <em>optional</em>, defaults to 1 &#x2014;
Number of channels in the audio data. Either 1 for mono or 2 for stereo. Stereo models generate a separate
audio stream for the left/right output channels. Mono models generate a single audio stream output.`,name:"audio_channels"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/musicgen/configuration_musicgen.py#L25"}}),Pe=new J({props:{title:"MusicgenConfig",local:"transformers.MusicgenConfig",headingTag:"h2"}}),Se=new B({props:{name:"class transformers.MusicgenConfig",anchor:"transformers.MusicgenConfig",parameters:[{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.MusicgenConfig.kwargs",description:`<strong>kwargs</strong> (<em>optional</em>) &#x2014;
Dictionary of keyword arguments. Notably:</p>
<ul>
<li><strong>text_encoder</strong> (<a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a>, <em>optional</em>) &#x2014; An instance of a configuration object that
defines the text encoder config.</li>
<li><strong>audio_encoder</strong> (<a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a>, <em>optional</em>) &#x2014; An instance of a configuration object that
defines the audio encoder config.</li>
<li><strong>decoder</strong> (<a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a>, <em>optional</em>) &#x2014; An instance of a configuration object that defines
the decoder config.</li>
</ul>`,name:"kwargs"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/musicgen/configuration_musicgen.py#L135"}}),H=new vo({props:{anchor:"transformers.MusicgenConfig.example",$$slots:{default:[vs]},$$scope:{ctx:U}}}),De=new B({props:{name:"from_sub_models_config",anchor:"transformers.MusicgenConfig.from_sub_models_config",parameters:[{name:"text_encoder_config",val:": PretrainedConfig"},{name:"audio_encoder_config",val:": PretrainedConfig"},{name:"decoder_config",val:": MusicgenDecoderConfig"},{name:"**kwargs",val:""}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/musicgen/configuration_musicgen.py#L219",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>An instance of a configuration object</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><a
  href="/docs/transformers/v4.56.2/en/model_doc/musicgen#transformers.MusicgenConfig"
>MusicgenConfig</a></p>
`}}),Ae=new J({props:{title:"MusicgenProcessor",local:"transformers.MusicgenProcessor",headingTag:"h2"}}),Oe=new B({props:{name:"class transformers.MusicgenProcessor",anchor:"transformers.MusicgenProcessor",parameters:[{name:"feature_extractor",val:""},{name:"tokenizer",val:""}],parametersDescription:[{anchor:"transformers.MusicgenProcessor.feature_extractor",description:`<strong>feature_extractor</strong> (<code>EncodecFeatureExtractor</code>) &#x2014;
An instance of <a href="/docs/transformers/v4.56.2/en/model_doc/encodec#transformers.EncodecFeatureExtractor">EncodecFeatureExtractor</a>. The feature extractor is a required input.`,name:"feature_extractor"},{anchor:"transformers.MusicgenProcessor.tokenizer",description:`<strong>tokenizer</strong> (<code>T5Tokenizer</code>) &#x2014;
An instance of <a href="/docs/transformers/v4.56.2/en/model_doc/t5#transformers.T5Tokenizer">T5Tokenizer</a>. The tokenizer is a required input.`,name:"tokenizer"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/musicgen/processing_musicgen.py#L27"}}),Ke=new B({props:{name:"batch_decode",anchor:"transformers.MusicgenProcessor.batch_decode",parameters:[{name:"*args",val:""},{name:"**kwargs",val:""}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/musicgen/processing_musicgen.py#L91"}}),et=new J({props:{title:"MusicgenModel",local:"transformers.MusicgenModel",headingTag:"h2"}}),tt=new B({props:{name:"class transformers.MusicgenModel",anchor:"transformers.MusicgenModel",parameters:[{name:"config",val:": MusicgenDecoderConfig"}],parametersDescription:[{anchor:"transformers.MusicgenModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/musicgen#transformers.MusicgenDecoderConfig">MusicgenDecoderConfig</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/musicgen/modeling_musicgen.py#L731"}}),nt=new B({props:{name:"forward",anchor:"transformers.MusicgenModel.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_hidden_states",val:": typing.Optional[torch.FloatTensor] = None"},{name:"encoder_attention_mask",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"cross_attn_head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"past_key_values",val:": typing.Optional[tuple[tuple[torch.FloatTensor]]] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.Tensor] = None"}],parametersDescription:[{anchor:"transformers.MusicgenModel.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size * num_codebooks, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary, corresponding to the sequence of audio codes.</p>
<p>Indices can be obtained by encoding an audio prompt with an audio encoder model to predict audio codes,
such as with the <a href="/docs/transformers/v4.56.2/en/model_doc/encodec#transformers.EncodecModel">EncodecModel</a>. See <a href="/docs/transformers/v4.56.2/en/model_doc/encodec#transformers.EncodecModel.encode">EncodecModel.encode()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a></p>
<div class="course-tip course-tip-orange bg-gradient-to-br dark:bg-gradient-to-r before:border-orange-500 dark:before:border-orange-800 from-orange-50 dark:from-gray-900 to-white dark:to-gray-950 border border-orange-50 text-orange-700 dark:text-gray-400">
						
<p>The <code>input_ids</code> will automatically be converted from shape <code>(batch_size * num_codebooks, target_sequence_length)</code> to <code>(batch_size, num_codebooks, target_sequence_length)</code> in the forward pass. If
you obtain audio codes from an audio encoding model, such as <a href="/docs/transformers/v4.56.2/en/model_doc/encodec#transformers.EncodecModel">EncodecModel</a>, ensure that the number of
frames is equal to 1, and that you reshape the audio codes from <code>(frames, batch_size, num_codebooks, target_sequence_length)</code> to <code>(batch_size * num_codebooks, target_sequence_length)</code> prior to passing them as
<code>input_ids</code>.</p>

					</div>`,name:"input_ids"},{anchor:"transformers.MusicgenModel.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.MusicgenModel.forward.encoder_hidden_states",description:`<strong>encoder_hidden_states</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, encoder_sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention of
the decoder.`,name:"encoder_hidden_states"},{anchor:"transformers.MusicgenModel.forward.encoder_attention_mask",description:`<strong>encoder_attention_mask</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, encoder_sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing cross-attention on padding tokens indices of encoder input_ids. Mask values
selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"encoder_attention_mask"},{anchor:"transformers.MusicgenModel.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.MusicgenModel.forward.cross_attn_head_mask",description:`<strong>cross_attn_head_mask</strong> (<code>torch.Tensor</code> of shape <code>(decoder_layers, decoder_attention_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the cross-attention modules in the decoder to avoid performing
cross-attention on hidden heads. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"cross_attn_head_mask"},{anchor:"transformers.MusicgenModel.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>tuple[tuple[torch.FloatTensor]]</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.MusicgenModel.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.MusicgenModel.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.MusicgenModel.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.MusicgenModel.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.MusicgenModel.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.MusicgenModel.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.Tensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/musicgen/modeling_musicgen.py#L744",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPastAndCrossAttentions"
>transformers.modeling_outputs.BaseModelOutputWithPastAndCrossAttentions</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/musicgen#transformers.MusicgenConfig"
>MusicgenConfig</a>) and inputs.</p>
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
`}}),P=new yo({props:{$$slots:{default:[Ts]},$$scope:{ctx:U}}}),ot=new J({props:{title:"MusicgenForCausalLM",local:"transformers.MusicgenForCausalLM",headingTag:"h2"}}),st=new B({props:{name:"class transformers.MusicgenForCausalLM",anchor:"transformers.MusicgenForCausalLM",parameters:[{name:"config",val:": MusicgenDecoderConfig"}],parametersDescription:[{anchor:"transformers.MusicgenForCausalLM.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/musicgen#transformers.MusicgenDecoderConfig">MusicgenDecoderConfig</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/musicgen/modeling_musicgen.py#L839"}}),at=new B({props:{name:"forward",anchor:"transformers.MusicgenForCausalLM.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_hidden_states",val:": typing.Optional[torch.FloatTensor] = None"},{name:"encoder_attention_mask",val:": typing.Optional[torch.LongTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"cross_attn_head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"past_key_values",val:": typing.Optional[tuple[tuple[torch.FloatTensor]]] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.Tensor] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.MusicgenForCausalLM.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size * num_codebooks, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary, corresponding to the sequence of audio codes.</p>
<p>Indices can be obtained by encoding an audio prompt with an audio encoder model to predict audio codes,
such as with the <a href="/docs/transformers/v4.56.2/en/model_doc/encodec#transformers.EncodecModel">EncodecModel</a>. See <a href="/docs/transformers/v4.56.2/en/model_doc/encodec#transformers.EncodecModel.encode">EncodecModel.encode()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a></p>
<div class="course-tip course-tip-orange bg-gradient-to-br dark:bg-gradient-to-r before:border-orange-500 dark:before:border-orange-800 from-orange-50 dark:from-gray-900 to-white dark:to-gray-950 border border-orange-50 text-orange-700 dark:text-gray-400">
						
<p>The <code>input_ids</code> will automatically be converted from shape <code>(batch_size * num_codebooks, target_sequence_length)</code> to <code>(batch_size, num_codebooks, target_sequence_length)</code> in the forward pass. If
you obtain audio codes from an audio encoding model, such as <a href="/docs/transformers/v4.56.2/en/model_doc/encodec#transformers.EncodecModel">EncodecModel</a>, ensure that the number of
frames is equal to 1, and that you reshape the audio codes from <code>(frames, batch_size, num_codebooks, target_sequence_length)</code> to <code>(batch_size * num_codebooks, target_sequence_length)</code> prior to passing them as
<code>input_ids</code>.</p>

					</div>`,name:"input_ids"},{anchor:"transformers.MusicgenForCausalLM.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.MusicgenForCausalLM.forward.encoder_hidden_states",description:`<strong>encoder_hidden_states</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, encoder_sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention of
the decoder.`,name:"encoder_hidden_states"},{anchor:"transformers.MusicgenForCausalLM.forward.encoder_attention_mask",description:`<strong>encoder_attention_mask</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, encoder_sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing cross-attention on padding tokens indices of encoder input_ids. Mask values
selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"encoder_attention_mask"},{anchor:"transformers.MusicgenForCausalLM.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.MusicgenForCausalLM.forward.cross_attn_head_mask",description:`<strong>cross_attn_head_mask</strong> (<code>torch.Tensor</code> of shape <code>(decoder_layers, decoder_attention_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the cross-attention modules in the decoder to avoid performing
cross-attention on hidden heads. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"cross_attn_head_mask"},{anchor:"transformers.MusicgenForCausalLM.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>tuple[tuple[torch.FloatTensor]]</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.MusicgenForCausalLM.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.MusicgenForCausalLM.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length, num_codebooks)</code>, <em>optional</em>) &#x2014;
Labels for language modeling. Note that the labels <strong>are shifted</strong> inside the model, i.e. you can set
<code>labels = input_ids</code> Indices are selected in <code>[-100, 0, ..., config.vocab_size]</code> All labels set to <code>-100</code>
are ignored (masked), the loss is only computed for labels in <code>[0, ..., config.vocab_size]</code>`,name:"labels"},{anchor:"transformers.MusicgenForCausalLM.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.MusicgenForCausalLM.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.MusicgenForCausalLM.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.MusicgenForCausalLM.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.MusicgenForCausalLM.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.Tensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/musicgen/modeling_musicgen.py#L871",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithCrossAttentions"
>transformers.modeling_outputs.CausalLMOutputWithCrossAttentions</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/musicgen#transformers.MusicgenConfig"
>MusicgenConfig</a>) and inputs.</p>
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
`}}),S=new yo({props:{$$slots:{default:[ws]},$$scope:{ctx:U}}}),D=new vo({props:{anchor:"transformers.MusicgenForCausalLM.forward.example",$$slots:{default:[ks]},$$scope:{ctx:U}}}),rt=new J({props:{title:"MusicgenForConditionalGeneration",local:"transformers.MusicgenForConditionalGeneration",headingTag:"h2"}}),it=new B({props:{name:"class transformers.MusicgenForConditionalGeneration",anchor:"transformers.MusicgenForConditionalGeneration",parameters:[{name:"config",val:": typing.Optional[transformers.models.musicgen.configuration_musicgen.MusicgenConfig] = None"},{name:"text_encoder",val:": typing.Optional[transformers.modeling_utils.PreTrainedModel] = None"},{name:"audio_encoder",val:": typing.Optional[transformers.modeling_utils.PreTrainedModel] = None"},{name:"decoder",val:": typing.Optional[transformers.models.musicgen.modeling_musicgen.MusicgenForCausalLM] = None"}],parametersDescription:[{anchor:"transformers.MusicgenForConditionalGeneration.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/musicgen#transformers.MusicgenConfig">MusicgenConfig</a>, <em>optional</em>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"},{anchor:"transformers.MusicgenForConditionalGeneration.text_encoder",description:`<strong>text_encoder</strong> (<code>PreTrainedModel</code>, <em>optional</em>) &#x2014;
The text encoder model that encodes text into hidden states for conditioning.`,name:"text_encoder"},{anchor:"transformers.MusicgenForConditionalGeneration.audio_encoder",description:`<strong>audio_encoder</strong> (<code>PreTrainedModel</code>, <em>optional</em>) &#x2014;
The audio encoder model that encodes audio into hidden states for conditioning.`,name:"audio_encoder"},{anchor:"transformers.MusicgenForConditionalGeneration.decoder",description:`<strong>decoder</strong> (<code>MusicgenForCausalLM</code>, <em>optional</em>) &#x2014;
The decoder model that generates audio tokens based on conditioning signals.`,name:"decoder"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/musicgen/modeling_musicgen.py#L1352"}}),lt=new B({props:{name:"forward",anchor:"transformers.MusicgenForConditionalGeneration.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.BoolTensor] = None"},{name:"input_values",val:": typing.Optional[torch.FloatTensor] = None"},{name:"padding_mask",val:": typing.Optional[torch.BoolTensor] = None"},{name:"decoder_input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"decoder_attention_mask",val:": typing.Optional[torch.BoolTensor] = None"},{name:"encoder_outputs",val:": typing.Optional[tuple[torch.FloatTensor]] = None"},{name:"past_key_values",val:": typing.Optional[tuple[tuple[torch.FloatTensor]]] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"decoder_inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.MusicgenForConditionalGeneration.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.MusicgenForConditionalGeneration.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.BoolTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.MusicgenForConditionalGeneration.forward.input_values",description:`<strong>input_values</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Float values of input raw speech waveform. Values can be obtained by loading a <code>.flac</code> or <code>.wav</code> audio file
into an array of type <code>list[float]</code>, a <code>numpy.ndarray</code> or a <code>torch.Tensor</code>, <em>e.g.</em> via the torchcodec library
(<code>pip install torchcodec</code>) or the soundfile library (<code>pip install soundfile</code>).
To prepare the array into <code>input_values</code>, the <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoProcessor">AutoProcessor</a> should be used for padding and conversion
into a tensor of type <code>torch.FloatTensor</code>. See <code>processor_class.__call__</code> for details.`,name:"input_values"},{anchor:"transformers.MusicgenForConditionalGeneration.forward.padding_mask",description:`<strong>padding_mask</strong> (<code>torch.BoolTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"padding_mask"},{anchor:"transformers.MusicgenForConditionalGeneration.forward.decoder_input_ids",description:`<strong>decoder_input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size * num_codebooks, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of decoder input sequence tokens in the vocabulary, corresponding to the sequence of audio codes.</p>
<p>Indices can be obtained by encoding an audio prompt with an audio encoder model to predict audio codes,
such as with the <a href="/docs/transformers/v4.56.2/en/model_doc/encodec#transformers.EncodecModel">EncodecModel</a>. See <a href="/docs/transformers/v4.56.2/en/model_doc/encodec#transformers.EncodecModel.encode">EncodecModel.encode()</a> for details.</p>
<p><a href="../glossary#decoder-input-ids">What are decoder input IDs?</a></p>
<div class="course-tip course-tip-orange bg-gradient-to-br dark:bg-gradient-to-r before:border-orange-500 dark:before:border-orange-800 from-orange-50 dark:from-gray-900 to-white dark:to-gray-950 border border-orange-50 text-orange-700 dark:text-gray-400">
						
<p>The <code>decoder_input_ids</code> will automatically be converted from shape <code>(batch_size * num_codebooks, target_sequence_length)</code> to <code>(batch_size, num_codebooks, target_sequence_length)</code> in the forward pass. If
you obtain audio codes from an audio encoding model, such as <a href="/docs/transformers/v4.56.2/en/model_doc/encodec#transformers.EncodecModel">EncodecModel</a>, ensure that the number of
frames is equal to 1, and that you reshape the audio codes from <code>(frames, batch_size, num_codebooks, target_sequence_length)</code> to <code>(batch_size * num_codebooks, target_sequence_length)</code> prior to passing them as
<code>decoder_input_ids</code>.</p>

					</div>`,name:"decoder_input_ids"},{anchor:"transformers.MusicgenForConditionalGeneration.forward.decoder_attention_mask",description:`<strong>decoder_attention_mask</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Default behavior: generate a tensor that ignores pad tokens in <code>decoder_input_ids</code>. Causal mask will also
be used by default.`,name:"decoder_attention_mask"},{anchor:"transformers.MusicgenForConditionalGeneration.forward.encoder_outputs",description:`<strong>encoder_outputs</strong> (<code>tuple[torch.FloatTensor]</code>, <em>optional</em>) &#x2014;
Tuple consists of (<code>last_hidden_state</code>, <em>optional</em>: <code>hidden_states</code>, <em>optional</em>: <code>attentions</code>)
<code>last_hidden_state</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) is a sequence of
hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.`,name:"encoder_outputs"},{anchor:"transformers.MusicgenForConditionalGeneration.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>tuple[tuple[torch.FloatTensor]]</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.MusicgenForConditionalGeneration.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.MusicgenForConditionalGeneration.forward.decoder_inputs_embeds",description:`<strong>decoder_inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, target_sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>decoder_input_ids</code> you can choose to directly pass an embedded
representation. If <code>past_key_values</code> is used, optionally only the last <code>decoder_inputs_embeds</code> have to be
input (see <code>past_key_values</code>). This is useful if you want more control over how to convert
<code>decoder_input_ids</code> indices into associated vectors than the model&#x2019;s internal embedding lookup matrix.</p>
<p>If <code>decoder_input_ids</code> and <code>decoder_inputs_embeds</code> are both unset, <code>decoder_inputs_embeds</code> takes the value
of <code>inputs_embeds</code>.`,name:"decoder_inputs_embeds"},{anchor:"transformers.MusicgenForConditionalGeneration.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length, num_codebooks)</code>, <em>optional</em>) &#x2014;
Labels for language modeling. Note that the labels <strong>are shifted</strong> inside the model, i.e. you can set
<code>labels = input_ids</code> Indices are selected in <code>[-100, 0, ..., config.vocab_size]</code> All labels set to <code>-100</code>
are ignored (masked), the loss is only computed for labels in <code>[0, ..., config.vocab_size]</code>`,name:"labels"},{anchor:"transformers.MusicgenForConditionalGeneration.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.MusicgenForConditionalGeneration.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.MusicgenForConditionalGeneration.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.MusicgenForConditionalGeneration.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/musicgen/modeling_musicgen.py#L1691",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqLMOutput"
>transformers.modeling_outputs.Seq2SeqLMOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/musicgen#transformers.MusicgenConfig"
>MusicgenConfig</a>) and inputs.</p>
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
`}}),A=new yo({props:{$$slots:{default:[Cs]},$$scope:{ctx:U}}}),O=new vo({props:{anchor:"transformers.MusicgenForConditionalGeneration.forward.example",$$slots:{default:[js]},$$scope:{ctx:U}}}),dt=new ys({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/musicgen.md"}}),{c(){l=r("meta"),y=s(),M=r("p"),b=s(),v=r("p"),v.innerHTML=p,T=s(),m(te.$$.fragment),It=s(),Y=r("div"),Y.innerHTML=To,Nt=s(),m(ne.$$.fragment),zt=s(),oe=r("p"),oe.innerHTML=wo,Xt=s(),se=r("p"),se.innerHTML=ko,Bt=s(),ae=r("p"),ae.textContent=Co,Rt=s(),re=r("p"),re.textContent=jo,qt=s(),ie=r("p"),ie.innerHTML=Jo,Vt=s(),le=r("p"),le.innerHTML=Uo,Et=s(),m(de.$$.fragment),Yt=s(),ce=r("ul"),ce.innerHTML=$o,Ht=s(),m(pe.$$.fragment),Lt=s(),me=r("blockquote"),me.innerHTML=Zo,Qt=s(),m(ue.$$.fragment),Pt=s(),he=r("p"),he.innerHTML=Go,St=s(),ge=r("p"),ge.textContent=xo,Dt=s(),fe=r("p"),fe.textContent=Fo,At=s(),m(_e.$$.fragment),Ot=s(),Me=r("p"),Me.innerHTML=Wo,Kt=s(),m(be.$$.fragment),en=s(),ye=r("p"),ye.innerHTML=Io,tn=s(),m(ve.$$.fragment),nn=s(),Te=r("p"),Te.innerHTML=No,on=s(),m(we.$$.fragment),sn=s(),m(ke.$$.fragment),an=s(),Ce=r("p"),Ce.innerHTML=zo,rn=s(),m(je.$$.fragment),ln=s(),Je=r("p"),Je.innerHTML=Xo,dn=s(),m(Ue.$$.fragment),cn=s(),$e=r("p"),$e.innerHTML=Bo,pn=s(),m(Ze.$$.fragment),mn=s(),m(Ge.$$.fragment),un=s(),xe=r("p"),xe.innerHTML=Ro,hn=s(),m(Fe.$$.fragment),gn=s(),m(We.$$.fragment),fn=s(),Ie=r("p"),Ie.textContent=qo,_n=s(),m(Ne.$$.fragment),Mn=s(),ze=r("p"),ze.innerHTML=Vo,bn=s(),m(Xe.$$.fragment),yn=s(),Be=r("p"),Be.textContent=Eo,vn=s(),Re=r("ol"),Re.innerHTML=Yo,Tn=s(),qe=r("p"),qe.innerHTML=Ho,wn=s(),m(Ve.$$.fragment),kn=s(),Ee=r("p"),Ee.innerHTML=Lo,Cn=s(),Ye=r("p"),Ye.textContent=Qo,jn=s(),He=r("ul"),He.innerHTML=Po,Jn=s(),m(Le.$$.fragment),Un=s(),z=r("div"),m(Qe.$$.fragment),Vn=s(),mt=r("p"),mt.innerHTML=So,En=s(),ut=r("p"),ut.innerHTML=Do,$n=s(),m(Pe.$$.fragment),Zn=s(),w=r("div"),m(Se.$$.fragment),Yn=s(),ht=r("p"),ht.innerHTML=Ao,Hn=s(),gt=r("p"),gt.innerHTML=Oo,Ln=s(),m(H.$$.fragment),Qn=s(),L=r("div"),m(De.$$.fragment),Pn=s(),ft=r("p"),ft.innerHTML=Ko,Gn=s(),m(Ae.$$.fragment),xn=s(),Z=r("div"),m(Oe.$$.fragment),Sn=s(),_t=r("p"),_t.textContent=es,Dn=s(),Mt=r("p"),Mt.innerHTML=ts,An=s(),Q=r("div"),m(Ke.$$.fragment),On=s(),bt=r("p"),bt.innerHTML=ns,Fn=s(),m(et.$$.fragment),Wn=s(),k=r("div"),m(tt.$$.fragment),Kn=s(),yt=r("p"),yt.textContent=os,eo=s(),vt=r("p"),vt.innerHTML=ss,to=s(),Tt=r("p"),Tt.innerHTML=as,no=s(),R=r("div"),m(nt.$$.fragment),oo=s(),wt=r("p"),wt.innerHTML=rs,so=s(),m(P.$$.fragment),In=s(),m(ot.$$.fragment),Nn=s(),C=r("div"),m(st.$$.fragment),ao=s(),kt=r("p"),kt.textContent=is,ro=s(),Ct=r("p"),Ct.innerHTML=ls,io=s(),jt=r("p"),jt.innerHTML=ds,lo=s(),I=r("div"),m(at.$$.fragment),co=s(),Jt=r("p"),Jt.innerHTML=cs,po=s(),m(S.$$.fragment),mo=s(),m(D.$$.fragment),zn=s(),m(rt.$$.fragment),Xn=s(),j=r("div"),m(it.$$.fragment),uo=s(),Ut=r("p"),Ut.textContent=ps,ho=s(),$t=r("p"),$t.innerHTML=ms,go=s(),Zt=r("p"),Zt.innerHTML=us,fo=s(),N=r("div"),m(lt.$$.fragment),_o=s(),Gt=r("p"),Gt.innerHTML=hs,Mo=s(),m(A.$$.fragment),bo=s(),m(O.$$.fragment),Bn=s(),m(dt.$$.fragment),Rn=s(),Wt=r("p"),this.h()},l(e){const t=bs("svelte-u9bgzb",document.head);l=i(t,"META",{name:!0,content:!0}),t.forEach(n),y=a(e),M=i(e,"P",{}),F(M).forEach(n),b=a(e),v=i(e,"P",{"data-svelte-h":!0}),c(v)!=="svelte-1e8935v"&&(v.innerHTML=p),T=a(e),u(te.$$.fragment,e),It=a(e),Y=i(e,"DIV",{class:!0,"data-svelte-h":!0}),c(Y)!=="svelte-b95w5j"&&(Y.innerHTML=To),Nt=a(e),u(ne.$$.fragment,e),zt=a(e),oe=i(e,"P",{"data-svelte-h":!0}),c(oe)!=="svelte-uhqugf"&&(oe.innerHTML=wo),Xt=a(e),se=i(e,"P",{"data-svelte-h":!0}),c(se)!=="svelte-8d0j4l"&&(se.innerHTML=ko),Bt=a(e),ae=i(e,"P",{"data-svelte-h":!0}),c(ae)!=="svelte-cdry7g"&&(ae.textContent=Co),Rt=a(e),re=i(e,"P",{"data-svelte-h":!0}),c(re)!=="svelte-vfdo9a"&&(re.textContent=jo),qt=a(e),ie=i(e,"P",{"data-svelte-h":!0}),c(ie)!=="svelte-6qc9b"&&(ie.innerHTML=Jo),Vt=a(e),le=i(e,"P",{"data-svelte-h":!0}),c(le)!=="svelte-1o040cl"&&(le.innerHTML=Uo),Et=a(e),u(de.$$.fragment,e),Yt=a(e),ce=i(e,"UL",{"data-svelte-h":!0}),c(ce)!=="svelte-12e77r4"&&(ce.innerHTML=$o),Ht=a(e),u(pe.$$.fragment,e),Lt=a(e),me=i(e,"BLOCKQUOTE",{"data-svelte-h":!0}),c(me)!=="svelte-1fwzni2"&&(me.innerHTML=Zo),Qt=a(e),u(ue.$$.fragment,e),Pt=a(e),he=i(e,"P",{"data-svelte-h":!0}),c(he)!=="svelte-1rrub89"&&(he.innerHTML=Go),St=a(e),ge=i(e,"P",{"data-svelte-h":!0}),c(ge)!=="svelte-1tppmp9"&&(ge.textContent=xo),Dt=a(e),fe=i(e,"P",{"data-svelte-h":!0}),c(fe)!=="svelte-1e09glx"&&(fe.textContent=Fo),At=a(e),u(_e.$$.fragment,e),Ot=a(e),Me=i(e,"P",{"data-svelte-h":!0}),c(Me)!=="svelte-hslupw"&&(Me.innerHTML=Wo),Kt=a(e),u(be.$$.fragment,e),en=a(e),ye=i(e,"P",{"data-svelte-h":!0}),c(ye)!=="svelte-c9ap2j"&&(ye.innerHTML=Io),tn=a(e),u(ve.$$.fragment,e),nn=a(e),Te=i(e,"P",{"data-svelte-h":!0}),c(Te)!=="svelte-1endcdl"&&(Te.innerHTML=No),on=a(e),u(we.$$.fragment,e),sn=a(e),u(ke.$$.fragment,e),an=a(e),Ce=i(e,"P",{"data-svelte-h":!0}),c(Ce)!=="svelte-13n78e9"&&(Ce.innerHTML=zo),rn=a(e),u(je.$$.fragment,e),ln=a(e),Je=i(e,"P",{"data-svelte-h":!0}),c(Je)!=="svelte-78k595"&&(Je.innerHTML=Xo),dn=a(e),u(Ue.$$.fragment,e),cn=a(e),$e=i(e,"P",{"data-svelte-h":!0}),c($e)!=="svelte-1bvh2sv"&&($e.innerHTML=Bo),pn=a(e),u(Ze.$$.fragment,e),mn=a(e),u(Ge.$$.fragment,e),un=a(e),xe=i(e,"P",{"data-svelte-h":!0}),c(xe)!=="svelte-1u4x79a"&&(xe.innerHTML=Ro),hn=a(e),u(Fe.$$.fragment,e),gn=a(e),u(We.$$.fragment,e),fn=a(e),Ie=i(e,"P",{"data-svelte-h":!0}),c(Ie)!=="svelte-1qo0662"&&(Ie.textContent=qo),_n=a(e),u(Ne.$$.fragment,e),Mn=a(e),ze=i(e,"P",{"data-svelte-h":!0}),c(ze)!=="svelte-132s853"&&(ze.innerHTML=Vo),bn=a(e),u(Xe.$$.fragment,e),yn=a(e),Be=i(e,"P",{"data-svelte-h":!0}),c(Be)!=="svelte-52mell"&&(Be.textContent=Eo),vn=a(e),Re=i(e,"OL",{"data-svelte-h":!0}),c(Re)!=="svelte-12ipaq0"&&(Re.innerHTML=Yo),Tn=a(e),qe=i(e,"P",{"data-svelte-h":!0}),c(qe)!=="svelte-1qkvx2d"&&(qe.innerHTML=Ho),wn=a(e),u(Ve.$$.fragment,e),kn=a(e),Ee=i(e,"P",{"data-svelte-h":!0}),c(Ee)!=="svelte-1wsm5ay"&&(Ee.innerHTML=Lo),Cn=a(e),Ye=i(e,"P",{"data-svelte-h":!0}),c(Ye)!=="svelte-axv494"&&(Ye.textContent=Qo),jn=a(e),He=i(e,"UL",{"data-svelte-h":!0}),c(He)!=="svelte-1upux1z"&&(He.innerHTML=Po),Jn=a(e),u(Le.$$.fragment,e),Un=a(e),z=i(e,"DIV",{class:!0});var E=F(z);u(Qe.$$.fragment,E),Vn=a(E),mt=i(E,"P",{"data-svelte-h":!0}),c(mt)!=="svelte-13s74fx"&&(mt.innerHTML=So),En=a(E),ut=i(E,"P",{"data-svelte-h":!0}),c(ut)!=="svelte-1ek1ss9"&&(ut.innerHTML=Do),E.forEach(n),$n=a(e),u(Pe.$$.fragment,e),Zn=a(e),w=i(e,"DIV",{class:!0});var G=F(w);u(Se.$$.fragment,G),Yn=a(G),ht=i(G,"P",{"data-svelte-h":!0}),c(ht)!=="svelte-sohisf"&&(ht.innerHTML=Ao),Hn=a(G),gt=i(G,"P",{"data-svelte-h":!0}),c(gt)!=="svelte-1ek1ss9"&&(gt.innerHTML=Oo),Ln=a(G),u(H.$$.fragment,G),Qn=a(G),L=i(G,"DIV",{class:!0});var ct=F(L);u(De.$$.fragment,ct),Pn=a(ct),ft=i(ct,"P",{"data-svelte-h":!0}),c(ft)!=="svelte-buswo"&&(ft.innerHTML=Ko),ct.forEach(n),G.forEach(n),Gn=a(e),u(Ae.$$.fragment,e),xn=a(e),Z=i(e,"DIV",{class:!0});var X=F(Z);u(Oe.$$.fragment,X),Sn=a(X),_t=i(X,"P",{"data-svelte-h":!0}),c(_t)!=="svelte-12wttjh"&&(_t.textContent=es),Dn=a(X),Mt=i(X,"P",{"data-svelte-h":!0}),c(Mt)!=="svelte-dmempc"&&(Mt.innerHTML=ts),An=a(X),Q=i(X,"DIV",{class:!0});var pt=F(Q);u(Ke.$$.fragment,pt),On=a(pt),bt=i(pt,"P",{"data-svelte-h":!0}),c(bt)!=="svelte-1kuivfw"&&(bt.innerHTML=ns),pt.forEach(n),X.forEach(n),Fn=a(e),u(et.$$.fragment,e),Wn=a(e),k=i(e,"DIV",{class:!0});var x=F(k);u(tt.$$.fragment,x),Kn=a(x),yt=i(x,"P",{"data-svelte-h":!0}),c(yt)!=="svelte-1u3x61j"&&(yt.textContent=os),eo=a(x),vt=i(x,"P",{"data-svelte-h":!0}),c(vt)!=="svelte-q52n56"&&(vt.innerHTML=ss),to=a(x),Tt=i(x,"P",{"data-svelte-h":!0}),c(Tt)!=="svelte-hswkmf"&&(Tt.innerHTML=as),no=a(x),R=i(x,"DIV",{class:!0});var xt=F(R);u(nt.$$.fragment,xt),oo=a(xt),wt=i(xt,"P",{"data-svelte-h":!0}),c(wt)!=="svelte-11n6mck"&&(wt.innerHTML=rs),so=a(xt),u(P.$$.fragment,xt),xt.forEach(n),x.forEach(n),In=a(e),u(ot.$$.fragment,e),Nn=a(e),C=i(e,"DIV",{class:!0});var q=F(C);u(st.$$.fragment,q),ao=a(q),kt=i(q,"P",{"data-svelte-h":!0}),c(kt)!=="svelte-1l370y5"&&(kt.textContent=is),ro=a(q),Ct=i(q,"P",{"data-svelte-h":!0}),c(Ct)!=="svelte-q52n56"&&(Ct.innerHTML=ls),io=a(q),jt=i(q,"P",{"data-svelte-h":!0}),c(jt)!=="svelte-hswkmf"&&(jt.innerHTML=ds),lo=a(q),I=i(q,"DIV",{class:!0});var K=F(I);u(at.$$.fragment,K),co=a(K),Jt=i(K,"P",{"data-svelte-h":!0}),c(Jt)!=="svelte-1wzef4k"&&(Jt.innerHTML=cs),po=a(K),u(S.$$.fragment,K),mo=a(K),u(D.$$.fragment,K),K.forEach(n),q.forEach(n),zn=a(e),u(rt.$$.fragment,e),Xn=a(e),j=i(e,"DIV",{class:!0});var V=F(j);u(it.$$.fragment,V),uo=a(V),Ut=i(V,"P",{"data-svelte-h":!0}),c(Ut)!=="svelte-4vubq6"&&(Ut.textContent=ps),ho=a(V),$t=i(V,"P",{"data-svelte-h":!0}),c($t)!=="svelte-q52n56"&&($t.innerHTML=ms),go=a(V),Zt=i(V,"P",{"data-svelte-h":!0}),c(Zt)!=="svelte-hswkmf"&&(Zt.innerHTML=us),fo=a(V),N=i(V,"DIV",{class:!0});var ee=F(N);u(lt.$$.fragment,ee),_o=a(ee),Gt=i(ee,"P",{"data-svelte-h":!0}),c(Gt)!=="svelte-1yyx41c"&&(Gt.innerHTML=hs),Mo=a(ee),u(A.$$.fragment,ee),bo=a(ee),u(O.$$.fragment,ee),ee.forEach(n),V.forEach(n),Bn=a(e),u(dt.$$.fragment,e),Rn=a(e),Wt=i(e,"P",{}),F(Wt).forEach(n),this.h()},h(){$(l,"name","hf:doc:metadata"),$(l,"content",Us),$(Y,"class","flex flex-wrap space-x-1"),$(z,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(L,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(w,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(Q,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(Z,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(R,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(k,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(I,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(C,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(N,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),$(j,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,t){d(document.head,l),o(e,y,t),o(e,M,t),o(e,b,t),o(e,v,t),o(e,T,t),h(te,e,t),o(e,It,t),o(e,Y,t),o(e,Nt,t),h(ne,e,t),o(e,zt,t),o(e,oe,t),o(e,Xt,t),o(e,se,t),o(e,Bt,t),o(e,ae,t),o(e,Rt,t),o(e,re,t),o(e,qt,t),o(e,ie,t),o(e,Vt,t),o(e,le,t),o(e,Et,t),h(de,e,t),o(e,Yt,t),o(e,ce,t),o(e,Ht,t),h(pe,e,t),o(e,Lt,t),o(e,me,t),o(e,Qt,t),h(ue,e,t),o(e,Pt,t),o(e,he,t),o(e,St,t),o(e,ge,t),o(e,Dt,t),o(e,fe,t),o(e,At,t),h(_e,e,t),o(e,Ot,t),o(e,Me,t),o(e,Kt,t),h(be,e,t),o(e,en,t),o(e,ye,t),o(e,tn,t),h(ve,e,t),o(e,nn,t),o(e,Te,t),o(e,on,t),h(we,e,t),o(e,sn,t),h(ke,e,t),o(e,an,t),o(e,Ce,t),o(e,rn,t),h(je,e,t),o(e,ln,t),o(e,Je,t),o(e,dn,t),h(Ue,e,t),o(e,cn,t),o(e,$e,t),o(e,pn,t),h(Ze,e,t),o(e,mn,t),h(Ge,e,t),o(e,un,t),o(e,xe,t),o(e,hn,t),h(Fe,e,t),o(e,gn,t),h(We,e,t),o(e,fn,t),o(e,Ie,t),o(e,_n,t),h(Ne,e,t),o(e,Mn,t),o(e,ze,t),o(e,bn,t),h(Xe,e,t),o(e,yn,t),o(e,Be,t),o(e,vn,t),o(e,Re,t),o(e,Tn,t),o(e,qe,t),o(e,wn,t),h(Ve,e,t),o(e,kn,t),o(e,Ee,t),o(e,Cn,t),o(e,Ye,t),o(e,jn,t),o(e,He,t),o(e,Jn,t),h(Le,e,t),o(e,Un,t),o(e,z,t),h(Qe,z,null),d(z,Vn),d(z,mt),d(z,En),d(z,ut),o(e,$n,t),h(Pe,e,t),o(e,Zn,t),o(e,w,t),h(Se,w,null),d(w,Yn),d(w,ht),d(w,Hn),d(w,gt),d(w,Ln),h(H,w,null),d(w,Qn),d(w,L),h(De,L,null),d(L,Pn),d(L,ft),o(e,Gn,t),h(Ae,e,t),o(e,xn,t),o(e,Z,t),h(Oe,Z,null),d(Z,Sn),d(Z,_t),d(Z,Dn),d(Z,Mt),d(Z,An),d(Z,Q),h(Ke,Q,null),d(Q,On),d(Q,bt),o(e,Fn,t),h(et,e,t),o(e,Wn,t),o(e,k,t),h(tt,k,null),d(k,Kn),d(k,yt),d(k,eo),d(k,vt),d(k,to),d(k,Tt),d(k,no),d(k,R),h(nt,R,null),d(R,oo),d(R,wt),d(R,so),h(P,R,null),o(e,In,t),h(ot,e,t),o(e,Nn,t),o(e,C,t),h(st,C,null),d(C,ao),d(C,kt),d(C,ro),d(C,Ct),d(C,io),d(C,jt),d(C,lo),d(C,I),h(at,I,null),d(I,co),d(I,Jt),d(I,po),h(S,I,null),d(I,mo),h(D,I,null),o(e,zn,t),h(rt,e,t),o(e,Xn,t),o(e,j,t),h(it,j,null),d(j,uo),d(j,Ut),d(j,ho),d(j,$t),d(j,go),d(j,Zt),d(j,fo),d(j,N),h(lt,N,null),d(N,_o),d(N,Gt),d(N,Mo),h(A,N,null),d(N,bo),h(O,N,null),o(e,Bn,t),h(dt,e,t),o(e,Rn,t),o(e,Wt,t),qn=!0},p(e,[t]){const E={};t&2&&(E.$$scope={dirty:t,ctx:e}),H.$set(E);const G={};t&2&&(G.$$scope={dirty:t,ctx:e}),P.$set(G);const ct={};t&2&&(ct.$$scope={dirty:t,ctx:e}),S.$set(ct);const X={};t&2&&(X.$$scope={dirty:t,ctx:e}),D.$set(X);const pt={};t&2&&(pt.$$scope={dirty:t,ctx:e}),A.$set(pt);const x={};t&2&&(x.$$scope={dirty:t,ctx:e}),O.$set(x)},i(e){qn||(g(te.$$.fragment,e),g(ne.$$.fragment,e),g(de.$$.fragment,e),g(pe.$$.fragment,e),g(ue.$$.fragment,e),g(_e.$$.fragment,e),g(be.$$.fragment,e),g(ve.$$.fragment,e),g(we.$$.fragment,e),g(ke.$$.fragment,e),g(je.$$.fragment,e),g(Ue.$$.fragment,e),g(Ze.$$.fragment,e),g(Ge.$$.fragment,e),g(Fe.$$.fragment,e),g(We.$$.fragment,e),g(Ne.$$.fragment,e),g(Xe.$$.fragment,e),g(Ve.$$.fragment,e),g(Le.$$.fragment,e),g(Qe.$$.fragment,e),g(Pe.$$.fragment,e),g(Se.$$.fragment,e),g(H.$$.fragment,e),g(De.$$.fragment,e),g(Ae.$$.fragment,e),g(Oe.$$.fragment,e),g(Ke.$$.fragment,e),g(et.$$.fragment,e),g(tt.$$.fragment,e),g(nt.$$.fragment,e),g(P.$$.fragment,e),g(ot.$$.fragment,e),g(st.$$.fragment,e),g(at.$$.fragment,e),g(S.$$.fragment,e),g(D.$$.fragment,e),g(rt.$$.fragment,e),g(it.$$.fragment,e),g(lt.$$.fragment,e),g(A.$$.fragment,e),g(O.$$.fragment,e),g(dt.$$.fragment,e),qn=!0)},o(e){f(te.$$.fragment,e),f(ne.$$.fragment,e),f(de.$$.fragment,e),f(pe.$$.fragment,e),f(ue.$$.fragment,e),f(_e.$$.fragment,e),f(be.$$.fragment,e),f(ve.$$.fragment,e),f(we.$$.fragment,e),f(ke.$$.fragment,e),f(je.$$.fragment,e),f(Ue.$$.fragment,e),f(Ze.$$.fragment,e),f(Ge.$$.fragment,e),f(Fe.$$.fragment,e),f(We.$$.fragment,e),f(Ne.$$.fragment,e),f(Xe.$$.fragment,e),f(Ve.$$.fragment,e),f(Le.$$.fragment,e),f(Qe.$$.fragment,e),f(Pe.$$.fragment,e),f(Se.$$.fragment,e),f(H.$$.fragment,e),f(De.$$.fragment,e),f(Ae.$$.fragment,e),f(Oe.$$.fragment,e),f(Ke.$$.fragment,e),f(et.$$.fragment,e),f(tt.$$.fragment,e),f(nt.$$.fragment,e),f(P.$$.fragment,e),f(ot.$$.fragment,e),f(st.$$.fragment,e),f(at.$$.fragment,e),f(S.$$.fragment,e),f(D.$$.fragment,e),f(rt.$$.fragment,e),f(it.$$.fragment,e),f(lt.$$.fragment,e),f(A.$$.fragment,e),f(O.$$.fragment,e),f(dt.$$.fragment,e),qn=!1},d(e){e&&(n(y),n(M),n(b),n(v),n(T),n(It),n(Y),n(Nt),n(zt),n(oe),n(Xt),n(se),n(Bt),n(ae),n(Rt),n(re),n(qt),n(ie),n(Vt),n(le),n(Et),n(Yt),n(ce),n(Ht),n(Lt),n(me),n(Qt),n(Pt),n(he),n(St),n(ge),n(Dt),n(fe),n(At),n(Ot),n(Me),n(Kt),n(en),n(ye),n(tn),n(nn),n(Te),n(on),n(sn),n(an),n(Ce),n(rn),n(ln),n(Je),n(dn),n(cn),n($e),n(pn),n(mn),n(un),n(xe),n(hn),n(gn),n(fn),n(Ie),n(_n),n(Mn),n(ze),n(bn),n(yn),n(Be),n(vn),n(Re),n(Tn),n(qe),n(wn),n(kn),n(Ee),n(Cn),n(Ye),n(jn),n(He),n(Jn),n(Un),n(z),n($n),n(Zn),n(w),n(Gn),n(xn),n(Z),n(Fn),n(Wn),n(k),n(In),n(Nn),n(C),n(zn),n(Xn),n(j),n(Bn),n(Rn),n(Wt)),n(l),_(te,e),_(ne,e),_(de,e),_(pe,e),_(ue,e),_(_e,e),_(be,e),_(ve,e),_(we,e),_(ke,e),_(je,e),_(Ue,e),_(Ze,e),_(Ge,e),_(Fe,e),_(We,e),_(Ne,e),_(Xe,e),_(Ve,e),_(Le,e),_(Qe),_(Pe,e),_(Se),_(H),_(De),_(Ae,e),_(Oe),_(Ke),_(et,e),_(tt),_(nt),_(P),_(ot,e),_(st),_(at),_(S),_(D),_(rt,e),_(it),_(lt),_(A),_(O),_(dt,e)}}}const Us='{"title":"MusicGen","local":"musicgen","sections":[{"title":"Overview","local":"overview","sections":[],"depth":2},{"title":"Usage tips","local":"usage-tips","sections":[],"depth":2},{"title":"Generation","local":"generation","sections":[{"title":"Unconditional Generation","local":"unconditional-generation","sections":[],"depth":3},{"title":"Text-Conditional Generation","local":"text-conditional-generation","sections":[],"depth":3},{"title":"Audio-Prompted Generation","local":"audio-prompted-generation","sections":[],"depth":3},{"title":"Generation Configuration","local":"generation-configuration","sections":[],"depth":3}],"depth":2},{"title":"Model Structure","local":"model-structure","sections":[],"depth":2},{"title":"MusicgenDecoderConfig","local":"transformers.MusicgenDecoderConfig","sections":[],"depth":2},{"title":"MusicgenConfig","local":"transformers.MusicgenConfig","sections":[],"depth":2},{"title":"MusicgenProcessor","local":"transformers.MusicgenProcessor","sections":[],"depth":2},{"title":"MusicgenModel","local":"transformers.MusicgenModel","sections":[],"depth":2},{"title":"MusicgenForCausalLM","local":"transformers.MusicgenForCausalLM","sections":[],"depth":2},{"title":"MusicgenForConditionalGeneration","local":"transformers.MusicgenForConditionalGeneration","sections":[],"depth":2}],"depth":1}';function $s(U){return fs(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class zs extends _s{constructor(l){super(),Ms(this,l,$s,Js,gs,{})}}export{zs as component};
