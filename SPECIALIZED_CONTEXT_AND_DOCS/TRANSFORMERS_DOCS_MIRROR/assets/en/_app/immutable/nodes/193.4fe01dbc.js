import{s as ao,o as ro,n as nt}from"../chunks/scheduler.18a86fab.js";import{S as io,i as co,g as i,s,r as u,A as lo,h as d,f as n,c as a,j as S,x as m,u as f,k as z,y as t,a as c,v as g,d as _,t as T,w as b}from"../chunks/index.98837b22.js";import{T as oo}from"../chunks/Tip.77304350.js";import{D as W}from"../chunks/Docstring.a1ef7999.js";import{C as so}from"../chunks/CodeBlock.8d0c2e8a.js";import{E as no}from"../chunks/ExampleCodeBlock.8c3ee1f9.js";import{H as ge,E as mo}from"../chunks/getInferenceSnippets.06c2775f.js";function po(I){let r,M="Examples:",p,h,y;return h=new so({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEZTTVRDb25maWclMkMlMjBGU01UTW9kZWwlMEElMEElMjMlMjBJbml0aWFsaXppbmclMjBhJTIwRlNNVCUyMGZhY2Vib29rJTJGd210MTktZW4tcnUlMjBzdHlsZSUyMGNvbmZpZ3VyYXRpb24lMEFjb25maWclMjAlM0QlMjBGU01UQ29uZmlnKCklMEElMEElMjMlMjBJbml0aWFsaXppbmclMjBhJTIwbW9kZWwlMjAod2l0aCUyMHJhbmRvbSUyMHdlaWdodHMpJTIwZnJvbSUyMHRoZSUyMGNvbmZpZ3VyYXRpb24lMEFtb2RlbCUyMCUzRCUyMEZTTVRNb2RlbChjb25maWcpJTBBJTBBJTIzJTIwQWNjZXNzaW5nJTIwdGhlJTIwbW9kZWwlMjBjb25maWd1cmF0aW9uJTBBY29uZmlndXJhdGlvbiUyMCUzRCUyMG1vZGVsLmNvbmZpZw==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> FSMTConfig, FSMTModel

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a FSMT facebook/wmt19-en-ru style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>config = FSMTConfig()

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a model (with random weights) from the configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = FSMTModel(config)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Accessing the model configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = model.config`,wrap:!1}}),{c(){r=i("p"),r.textContent=M,p=s(),u(h.$$.fragment)},l(l){r=d(l,"P",{"data-svelte-h":!0}),m(r)!=="svelte-kvfsh7"&&(r.textContent=M),p=a(l),f(h.$$.fragment,l)},m(l,$){c(l,r,$),c(l,p,$),g(h,l,$),y=!0},p:nt,i(l){y||(_(h.$$.fragment,l),y=!0)},o(l){T(h.$$.fragment,l),y=!1},d(l){l&&(n(r),n(p)),b(h,l)}}}function ho(I){let r,M=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){r=i("p"),r.innerHTML=M},l(p){r=d(p,"P",{"data-svelte-h":!0}),m(r)!=="svelte-fincs2"&&(r.innerHTML=M)},m(p,h){c(p,r,h)},p:nt,d(p){p&&n(r)}}}function uo(I){let r,M=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){r=i("p"),r.innerHTML=M},l(p){r=d(p,"P",{"data-svelte-h":!0}),m(r)!=="svelte-fincs2"&&(r.innerHTML=M)},m(p,h){c(p,r,h)},p:nt,d(p){p&&n(r)}}}function fo(I){let r,M="Example Translation:",p,h,y;return h=new so({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBGU01URm9yQ29uZGl0aW9uYWxHZW5lcmF0aW9uJTBBJTBBbW5hbWUlMjAlM0QlMjAlMjJmYWNlYm9vayUyRndtdDE5LXJ1LWVuJTIyJTBBbW9kZWwlMjAlM0QlMjBGU01URm9yQ29uZGl0aW9uYWxHZW5lcmF0aW9uLmZyb21fcHJldHJhaW5lZChtbmFtZSklMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZChtbmFtZSklMEElMEFzcmNfdGV4dCUyMCUzRCUyMCUyMiVEMCU5QyVEMCVCMCVEMSU4OCVEMCVCOCVEMCVCRCVEMCVCRCVEMCVCRSVEMCVCNSUyMCVEMCVCRSVEMCVCMSVEMSU4MyVEMSU4NyVEMCVCNSVEMCVCRCVEMCVCOCVEMCVCNSUyMC0lMjAlRDElOEQlRDElODIlRDAlQkUlMjAlRDAlQjclRDAlQjQlRDAlQkUlRDElODAlRDAlQkUlRDAlQjIlRDAlQkUlMkMlMjAlRDAlQkQlRDAlQjUlMjAlRDElODIlRDAlQjAlRDAlQkElMjAlRDAlQkIlRDAlQjglM0YlMjIlMEFpbnB1dF9pZHMlMjAlM0QlMjB0b2tlbml6ZXIoc3JjX3RleHQlMkMlMjByZXR1cm5fdGVuc29ycyUzRCUyMnB0JTIyKS5pbnB1dF9pZHMlMEFvdXRwdXRzJTIwJTNEJTIwbW9kZWwuZ2VuZXJhdGUoaW5wdXRfaWRzJTJDJTIwbnVtX2JlYW1zJTNENSUyQyUyMG51bV9yZXR1cm5fc2VxdWVuY2VzJTNEMyklMEF0b2tlbml6ZXIuZGVjb2RlKG91dHB1dHMlNUIwJTVEJTJDJTIwc2tpcF9zcGVjaWFsX3Rva2VucyUzRFRydWUp",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, FSMTForConditionalGeneration

<span class="hljs-meta">&gt;&gt;&gt; </span>mname = <span class="hljs-string">&quot;facebook/wmt19-ru-en&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = FSMTForConditionalGeneration.from_pretrained(mname)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer = AutoTokenizer.from_pretrained(mname)

<span class="hljs-meta">&gt;&gt;&gt; </span>src_text = <span class="hljs-string">&quot;Машинное обучение - это здорово, не так ли?&quot;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>input_ids = tokenizer(src_text, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>).input_ids
<span class="hljs-meta">&gt;&gt;&gt; </span>outputs = model.generate(input_ids, num_beams=<span class="hljs-number">5</span>, num_return_sequences=<span class="hljs-number">3</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>tokenizer.decode(outputs[<span class="hljs-number">0</span>], skip_special_tokens=<span class="hljs-literal">True</span>)
<span class="hljs-string">&quot;Machine learning is great, isn&#x27;t it?&quot;</span>`,wrap:!1}}),{c(){r=i("p"),r.textContent=M,p=s(),u(h.$$.fragment)},l(l){r=d(l,"P",{"data-svelte-h":!0}),m(r)!=="svelte-hvxwgb"&&(r.textContent=M),p=a(l),f(h.$$.fragment,l)},m(l,$){c(l,r,$),c(l,p,$),g(h,l,$),y=!0},p:nt,i(l){y||(_(h.$$.fragment,l),y=!0)},o(l){T(h.$$.fragment,l),y=!1},d(l){l&&(n(r),n(p)),b(h,l)}}}function go(I){let r,M,p,h,y,l="<em>This model was released on 2019-07-15 and added to Hugging Face Transformers on 2020-11-16.</em>",$,R,Ne,B,Oe,Z,It='FSMT (FairSeq MachineTranslation) models were introduced in <a href="https://huggingface.co/papers/1907.06616" rel="nofollow">Facebook FAIR’s WMT19 News Translation Task Submission</a> by Nathan Ng, Kyra Yee, Alexei Baevski, Myle Ott, Michael Auli, Sergey Edunov.',He,J,Lt="The abstract of the paper is the following:",Ue,Q,Dt=`<em>This paper describes Facebook FAIR’s submission to the WMT19 shared news translation task. We participate in two
language pairs and four language directions, English &lt;-&gt; German and English &lt;-&gt; Russian. Following our submission from
last year, our baseline systems are large BPE-based transformer models trained with the Fairseq sequence modeling
toolkit which rely on sampled back-translations. This year we experiment with different bitext data filtering schemes,
as well as with adding filtered back-translated data. We also ensemble and fine-tune our models on domain-specific
data, then decode using noisy channel model reranking. Our submissions are ranked first in all four directions of the
human evaluation campaign. On En-&gt;De, our system significantly outperforms other systems as well as human translations.
This system improves upon our WMT’18 submission by 4.5 BLEU points.</em>`,Ve,X,Et=`This model was contributed by <a href="https://huggingface.co/stas" rel="nofollow">stas</a>. The original code can be found
<a href="https://github.com/pytorch/fairseq/tree/master/examples/wmt19" rel="nofollow">here</a>.`,Ae,Y,Ge,K,jt=`<li>FSMT uses source and target vocabulary pairs that aren’t combined into one. It doesn’t share embeddings tokens
either. Its tokenizer is very similar to <a href="/docs/transformers/v4.56.2/en/model_doc/xlm#transformers.XLMTokenizer">XLMTokenizer</a> and the main model is derived from
<a href="/docs/transformers/v4.56.2/en/model_doc/bart#transformers.BartModel">BartModel</a>.</li>`,Re,ee,Be,x,te,st,_e,Pt=`This is the configuration class to store the configuration of a <a href="/docs/transformers/v4.56.2/en/model_doc/fsmt#transformers.FSMTModel">FSMTModel</a>. It is used to instantiate a FSMT
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the FSMT
<a href="https://huggingface.co/facebook/wmt19-en-ru" rel="nofollow">facebook/wmt19-en-ru</a> architecture.`,at,Te,Wt=`Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> and can be used to control the model outputs. Read the
documentation from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> for more information.`,rt,O,Ze,oe,Je,k,ne,it,be,Nt="Construct an FAIRSEQ Transformer tokenizer. Based on Byte-Pair Encoding. The tokenization process is the following:",dt,ke,Ot=`<li>Moses preprocessing and tokenization.</li> <li>Normalizing all inputs text.</li> <li>The arguments <code>special_tokens</code> and the function <code>set_special_tokens</code>, can be used to add additional symbols (like
”<strong>classify</strong>”) to a vocabulary.</li> <li>The argument <code>langs</code> defines a pair of languages.</li>`,ct,ve,Ht=`This tokenizer inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer">PreTrainedTokenizer</a> which contains most of the main methods. Users should refer to
this superclass for more information regarding those methods.`,lt,L,se,mt,ye,Ut=`Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
adding special tokens. A FAIRSEQ Transformer sequence has the following format:`,pt,Me,Vt="<li>single sequence: <code>&lt;s&gt; X &lt;/s&gt;</code></li> <li>pair of sequences: <code>&lt;s&gt; A &lt;/s&gt; B &lt;/s&gt;</code></li>",ht,H,ae,ut,we,At=`Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
special tokens using the tokenizer <code>prepare_for_model</code> method.`,ft,D,re,gt,Fe,Gt=`Create the token type IDs corresponding to the sequences passed. <a href="../glossary#token-type-ids">What are token type
IDs?</a>`,_t,xe,Rt="Should be overridden in a subclass if the model has a special way of building those.",Tt,$e,ie,Qe,de,Xe,w,ce,bt,Ce,Bt="The bare Fsmt Model outputting raw hidden-states without any specific head on top.",kt,Se,Zt=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,vt,ze,Jt=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,yt,E,le,Mt,qe,Qt='The <a href="/docs/transformers/v4.56.2/en/model_doc/fsmt#transformers.FSMTModel">FSMTModel</a> forward method, overrides the <code>__call__</code> special method.',wt,U,Ye,me,Ke,F,pe,Ft,Ie,Xt="The FSMT Model with a language modeling head. Can be used for summarization.",xt,Le,Yt=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,$t,De,Kt=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Ct,C,he,St,Ee,eo='The <a href="/docs/transformers/v4.56.2/en/model_doc/fsmt#transformers.FSMTForConditionalGeneration">FSMTForConditionalGeneration</a> forward method, overrides the <code>__call__</code> special method.',zt,V,qt,A,et,ue,tt,We,ot;return R=new ge({props:{title:"FSMT",local:"fsmt",headingTag:"h1"}}),B=new ge({props:{title:"Overview",local:"overview",headingTag:"h2"}}),Y=new ge({props:{title:"Implementation Notes",local:"implementation-notes",headingTag:"h2"}}),ee=new ge({props:{title:"FSMTConfig",local:"transformers.FSMTConfig",headingTag:"h2"}}),te=new W({props:{name:"class transformers.FSMTConfig",anchor:"transformers.FSMTConfig",parameters:[{name:"langs",val:" = ['en', 'de']"},{name:"src_vocab_size",val:" = 42024"},{name:"tgt_vocab_size",val:" = 42024"},{name:"activation_function",val:" = 'relu'"},{name:"d_model",val:" = 1024"},{name:"max_length",val:" = 200"},{name:"max_position_embeddings",val:" = 1024"},{name:"encoder_ffn_dim",val:" = 4096"},{name:"encoder_layers",val:" = 12"},{name:"encoder_attention_heads",val:" = 16"},{name:"encoder_layerdrop",val:" = 0.0"},{name:"decoder_ffn_dim",val:" = 4096"},{name:"decoder_layers",val:" = 12"},{name:"decoder_attention_heads",val:" = 16"},{name:"decoder_layerdrop",val:" = 0.0"},{name:"attention_dropout",val:" = 0.0"},{name:"dropout",val:" = 0.1"},{name:"activation_dropout",val:" = 0.0"},{name:"init_std",val:" = 0.02"},{name:"decoder_start_token_id",val:" = 2"},{name:"is_encoder_decoder",val:" = True"},{name:"scale_embedding",val:" = True"},{name:"tie_word_embeddings",val:" = False"},{name:"num_beams",val:" = 5"},{name:"length_penalty",val:" = 1.0"},{name:"early_stopping",val:" = False"},{name:"use_cache",val:" = True"},{name:"pad_token_id",val:" = 1"},{name:"bos_token_id",val:" = 0"},{name:"eos_token_id",val:" = 2"},{name:"forced_eos_token_id",val:" = 2"},{name:"**common_kwargs",val:""}],parametersDescription:[{anchor:"transformers.FSMTConfig.langs",description:`<strong>langs</strong> (<code>list[str]</code>) &#x2014;
A list with source language and target_language (e.g., [&#x2018;en&#x2019;, &#x2018;ru&#x2019;]).`,name:"langs"},{anchor:"transformers.FSMTConfig.src_vocab_size",description:`<strong>src_vocab_size</strong> (<code>int</code>) &#x2014;
Vocabulary size of the encoder. Defines the number of different tokens that can be represented by the
<code>inputs_ids</code> passed to the forward method in the encoder.`,name:"src_vocab_size"},{anchor:"transformers.FSMTConfig.tgt_vocab_size",description:`<strong>tgt_vocab_size</strong> (<code>int</code>) &#x2014;
Vocabulary size of the decoder. Defines the number of different tokens that can be represented by the
<code>inputs_ids</code> passed to the forward method in the decoder.`,name:"tgt_vocab_size"},{anchor:"transformers.FSMTConfig.d_model",description:`<strong>d_model</strong> (<code>int</code>, <em>optional</em>, defaults to 1024) &#x2014;
Dimensionality of the layers and the pooler layer.`,name:"d_model"},{anchor:"transformers.FSMTConfig.encoder_layers",description:`<strong>encoder_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 12) &#x2014;
Number of encoder layers.`,name:"encoder_layers"},{anchor:"transformers.FSMTConfig.decoder_layers",description:`<strong>decoder_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 12) &#x2014;
Number of decoder layers.`,name:"decoder_layers"},{anchor:"transformers.FSMTConfig.encoder_attention_heads",description:`<strong>encoder_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 16) &#x2014;
Number of attention heads for each attention layer in the Transformer encoder.`,name:"encoder_attention_heads"},{anchor:"transformers.FSMTConfig.decoder_attention_heads",description:`<strong>decoder_attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 16) &#x2014;
Number of attention heads for each attention layer in the Transformer decoder.`,name:"decoder_attention_heads"},{anchor:"transformers.FSMTConfig.decoder_ffn_dim",description:`<strong>decoder_ffn_dim</strong> (<code>int</code>, <em>optional</em>, defaults to 4096) &#x2014;
Dimensionality of the &#x201C;intermediate&#x201D; (often named feed-forward) layer in decoder.`,name:"decoder_ffn_dim"},{anchor:"transformers.FSMTConfig.encoder_ffn_dim",description:`<strong>encoder_ffn_dim</strong> (<code>int</code>, <em>optional</em>, defaults to 4096) &#x2014;
Dimensionality of the &#x201C;intermediate&#x201D; (often named feed-forward) layer in decoder.`,name:"encoder_ffn_dim"},{anchor:"transformers.FSMTConfig.activation_function",description:`<strong>activation_function</strong> (<code>str</code> or <code>Callable</code>, <em>optional</em>, defaults to <code>&quot;relu&quot;</code>) &#x2014;
The non-linear activation function (function or string) in the encoder and pooler. If string, <code>&quot;gelu&quot;</code>,
<code>&quot;relu&quot;</code>, <code>&quot;silu&quot;</code> and <code>&quot;gelu_new&quot;</code> are supported.`,name:"activation_function"},{anchor:"transformers.FSMTConfig.dropout",description:`<strong>dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.`,name:"dropout"},{anchor:"transformers.FSMTConfig.attention_dropout",description:`<strong>attention_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The dropout ratio for the attention probabilities.`,name:"attention_dropout"},{anchor:"transformers.FSMTConfig.activation_dropout",description:`<strong>activation_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The dropout ratio for activations inside the fully connected layer.`,name:"activation_dropout"},{anchor:"transformers.FSMTConfig.max_position_embeddings",description:`<strong>max_position_embeddings</strong> (<code>int</code>, <em>optional</em>, defaults to 1024) &#x2014;
The maximum sequence length that this model might ever be used with. Typically set this to something large
just in case (e.g., 512 or 1024 or 2048).`,name:"max_position_embeddings"},{anchor:"transformers.FSMTConfig.init_std",description:`<strong>init_std</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.`,name:"init_std"},{anchor:"transformers.FSMTConfig.scale_embedding",description:`<strong>scale_embedding</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Scale embeddings by diving by sqrt(d_model).`,name:"scale_embedding"},{anchor:"transformers.FSMTConfig.bos_token_id",description:`<strong>bos_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 0) &#x2014;
Beginning of stream token id.`,name:"bos_token_id"},{anchor:"transformers.FSMTConfig.pad_token_id",description:`<strong>pad_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 1) &#x2014;
Padding token id.`,name:"pad_token_id"},{anchor:"transformers.FSMTConfig.eos_token_id",description:`<strong>eos_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 2) &#x2014;
End of stream token id.`,name:"eos_token_id"},{anchor:"transformers.FSMTConfig.decoder_start_token_id",description:`<strong>decoder_start_token_id</strong> (<code>int</code>, <em>optional</em>) &#x2014;
This model starts decoding with <code>eos_token_id</code>`,name:"decoder_start_token_id"},{anchor:"transformers.FSMTConfig.encoder_layerdrop",description:`<strong>encoder_layerdrop</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
Google &#x201C;layerdrop arxiv&#x201D;, as its not explainable in one line.`,name:"encoder_layerdrop"},{anchor:"transformers.FSMTConfig.decoder_layerdrop",description:`<strong>decoder_layerdrop</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
Google &#x201C;layerdrop arxiv&#x201D;, as its not explainable in one line.`,name:"decoder_layerdrop"},{anchor:"transformers.FSMTConfig.is_encoder_decoder",description:`<strong>is_encoder_decoder</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether this is an encoder/decoder model.`,name:"is_encoder_decoder"},{anchor:"transformers.FSMTConfig.tie_word_embeddings",description:`<strong>tie_word_embeddings</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether to tie input and output embeddings.`,name:"tie_word_embeddings"},{anchor:"transformers.FSMTConfig.num_beams",description:`<strong>num_beams</strong> (<code>int</code>, <em>optional</em>, defaults to 5) &#x2014;
Number of beams for beam search that will be used by default in the <code>generate</code> method of the model. 1 means
no beam search.`,name:"num_beams"},{anchor:"transformers.FSMTConfig.length_penalty",description:`<strong>length_penalty</strong> (<code>float</code>, <em>optional</em>, defaults to 1) &#x2014;
Exponential penalty to the length that is used with beam-based generation. It is applied as an exponent to
the sequence length, which in turn is used to divide the score of the sequence. Since the score is the log
likelihood of the sequence (i.e. negative), <code>length_penalty</code> &gt; 0.0 promotes longer sequences, while
<code>length_penalty</code> &lt; 0.0 encourages shorter sequences.`,name:"length_penalty"},{anchor:"transformers.FSMTConfig.early_stopping",description:`<strong>early_stopping</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Flag that will be used by default in the <code>generate</code> method of the model. Whether to stop the beam search
when at least <code>num_beams</code> sentences are finished per batch or not.`,name:"early_stopping"},{anchor:"transformers.FSMTConfig.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not the model should return the last key/values attentions (not used by all models).`,name:"use_cache"},{anchor:"transformers.FSMTConfig.forced_eos_token_id",description:`<strong>forced_eos_token_id</strong> (<code>int</code>, <em>optional</em>, defaults to 2) &#x2014;
The id of the token to force as the last generated token when <code>max_length</code> is reached. Usually set to
<code>eos_token_id</code>.`,name:"forced_eos_token_id"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/fsmt/configuration_fsmt.py#L38"}}),O=new no({props:{anchor:"transformers.FSMTConfig.example",$$slots:{default:[po]},$$scope:{ctx:I}}}),oe=new ge({props:{title:"FSMTTokenizer",local:"transformers.FSMTTokenizer",headingTag:"h2"}}),ne=new W({props:{name:"class transformers.FSMTTokenizer",anchor:"transformers.FSMTTokenizer",parameters:[{name:"langs",val:" = None"},{name:"src_vocab_file",val:" = None"},{name:"tgt_vocab_file",val:" = None"},{name:"merges_file",val:" = None"},{name:"do_lower_case",val:" = False"},{name:"unk_token",val:" = '<unk>'"},{name:"bos_token",val:" = '<s>'"},{name:"sep_token",val:" = '</s>'"},{name:"pad_token",val:" = '<pad>'"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.FSMTTokenizer.langs",description:`<strong>langs</strong> (<code>List[str]</code>, <em>optional</em>) &#x2014;
A list of two languages to translate from and to, for instance <code>[&quot;en&quot;, &quot;ru&quot;]</code>.`,name:"langs"},{anchor:"transformers.FSMTTokenizer.src_vocab_file",description:`<strong>src_vocab_file</strong> (<code>str</code>, <em>optional</em>) &#x2014;
File containing the vocabulary for the source language.`,name:"src_vocab_file"},{anchor:"transformers.FSMTTokenizer.tgt_vocab_file",description:`<strong>tgt_vocab_file</strong> (<code>st</code>, <em>optional</em>) &#x2014;
File containing the vocabulary for the target language.`,name:"tgt_vocab_file"},{anchor:"transformers.FSMTTokenizer.merges_file",description:`<strong>merges_file</strong> (<code>str</code>, <em>optional</em>) &#x2014;
File containing the merges.`,name:"merges_file"},{anchor:"transformers.FSMTTokenizer.do_lower_case",description:`<strong>do_lower_case</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not to lowercase the input when tokenizing.`,name:"do_lower_case"},{anchor:"transformers.FSMTTokenizer.unk_token",description:`<strong>unk_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;unk&gt;&quot;</code>) &#x2014;
The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
token instead.`,name:"unk_token"},{anchor:"transformers.FSMTTokenizer.bos_token",description:`<strong>bos_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;s&gt;&quot;</code>) &#x2014;
The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.</p>
<div class="course-tip  bg-gradient-to-br dark:bg-gradient-to-r before:border-green-500 dark:before:border-green-800 from-green-50 dark:from-gray-900 to-white dark:to-gray-950 border border-green-50 text-green-700 dark:text-gray-400">
						
<p>When building a sequence using special tokens, this is not the token that is used for the beginning of
sequence. The token used is the <code>cls_token</code>.</p>

					</div>`,name:"bos_token"},{anchor:"transformers.FSMTTokenizer.sep_token",description:`<strong>sep_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;/s&gt;&quot;</code>) &#x2014;
The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
sequence classification or for a text and a question for question answering. It is also used as the last
token of a sequence built with special tokens.`,name:"sep_token"},{anchor:"transformers.FSMTTokenizer.pad_token",description:`<strong>pad_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;pad&gt;&quot;</code>) &#x2014;
The token used for padding, for example when batching sequences of different lengths.`,name:"pad_token"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/fsmt/tokenization_fsmt.py#L114"}}),se=new W({props:{name:"build_inputs_with_special_tokens",anchor:"transformers.FSMTTokenizer.build_inputs_with_special_tokens",parameters:[{name:"token_ids_0",val:": list"},{name:"token_ids_1",val:": typing.Optional[list[int]] = None"}],parametersDescription:[{anchor:"transformers.FSMTTokenizer.build_inputs_with_special_tokens.token_ids_0",description:`<strong>token_ids_0</strong> (<code>List[int]</code>) &#x2014;
List of IDs to which the special tokens will be added.`,name:"token_ids_0"},{anchor:"transformers.FSMTTokenizer.build_inputs_with_special_tokens.token_ids_1",description:`<strong>token_ids_1</strong> (<code>List[int]</code>, <em>optional</em>) &#x2014;
Optional second list of IDs for sequence pairs.`,name:"token_ids_1"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/fsmt/tokenization_fsmt.py#L379",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>List of <a href="../glossary#input-ids">input IDs</a> with the appropriate special tokens.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>List[int]</code></p>
`}}),ae=new W({props:{name:"get_special_tokens_mask",anchor:"transformers.FSMTTokenizer.get_special_tokens_mask",parameters:[{name:"token_ids_0",val:": list"},{name:"token_ids_1",val:": typing.Optional[list[int]] = None"},{name:"already_has_special_tokens",val:": bool = False"}],parametersDescription:[{anchor:"transformers.FSMTTokenizer.get_special_tokens_mask.token_ids_0",description:`<strong>token_ids_0</strong> (<code>List[int]</code>) &#x2014;
List of IDs.`,name:"token_ids_0"},{anchor:"transformers.FSMTTokenizer.get_special_tokens_mask.token_ids_1",description:`<strong>token_ids_1</strong> (<code>List[int]</code>, <em>optional</em>) &#x2014;
Optional second list of IDs for sequence pairs.`,name:"token_ids_1"},{anchor:"transformers.FSMTTokenizer.get_special_tokens_mask.already_has_special_tokens",description:`<strong>already_has_special_tokens</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not the token list is already formatted with special tokens for the model.`,name:"already_has_special_tokens"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/fsmt/tokenization_fsmt.py#L405",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>List[int]</code></p>
`}}),re=new W({props:{name:"create_token_type_ids_from_sequences",anchor:"transformers.FSMTTokenizer.create_token_type_ids_from_sequences",parameters:[{name:"token_ids_0",val:": list"},{name:"token_ids_1",val:": typing.Optional[list[int]] = None"}],parametersDescription:[{anchor:"transformers.FSMTTokenizer.create_token_type_ids_from_sequences.token_ids_0",description:"<strong>token_ids_0</strong> (<code>list[int]</code>) &#x2014; The first tokenized sequence.",name:"token_ids_0"},{anchor:"transformers.FSMTTokenizer.create_token_type_ids_from_sequences.token_ids_1",description:"<strong>token_ids_1</strong> (<code>list[int]</code>, <em>optional</em>) &#x2014; The second tokenized sequence.",name:"token_ids_1"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L3432",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>The token type ids.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>list[int]</code></p>
`}}),ie=new W({props:{name:"save_vocabulary",anchor:"transformers.FSMTTokenizer.save_vocabulary",parameters:[{name:"save_directory",val:": str"},{name:"filename_prefix",val:": typing.Optional[str] = None"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/fsmt/tokenization_fsmt.py#L433"}}),de=new ge({props:{title:"FSMTModel",local:"transformers.FSMTModel",headingTag:"h2"}}),ce=new W({props:{name:"class transformers.FSMTModel",anchor:"transformers.FSMTModel",parameters:[{name:"config",val:": FSMTConfig"}],parametersDescription:[{anchor:"transformers.FSMTModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/fsmt#transformers.FSMTConfig">FSMTConfig</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/fsmt/modeling_fsmt.py#L890"}}),le=new W({props:{name:"forward",anchor:"transformers.FSMTModel.forward",parameters:[{name:"input_ids",val:": LongTensor"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"decoder_input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"decoder_attention_mask",val:": typing.Optional[torch.BoolTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"decoder_head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"cross_attn_head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_outputs",val:": typing.Optional[tuple[torch.FloatTensor]] = None"},{name:"past_key_values",val:": typing.Optional[tuple[torch.FloatTensor]] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"decoder_inputs_embeds",val:": typing.Optional[torch.FloatTensor] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.Tensor] = None"}],parametersDescription:[{anchor:"transformers.FSMTModel.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.FSMTModel.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.FSMTModel.forward.decoder_input_ids",description:`<strong>decoder_input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of decoder input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#decoder-input-ids">What are decoder input IDs?</a></p>
<p>FSMT uses the <code>eos_token_id</code> as the starting token for <code>decoder_input_ids</code> generation. If <code>past_key_values</code>
is used, optionally only the last <code>decoder_input_ids</code> have to be input (see <code>past_key_values</code>).`,name:"decoder_input_ids"},{anchor:"transformers.FSMTModel.forward.decoder_attention_mask",description:`<strong>decoder_attention_mask</strong> (<code>torch.BoolTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Default behavior: generate a tensor that ignores pad tokens in <code>decoder_input_ids</code>. Causal mask will also
be used by default.`,name:"decoder_attention_mask"},{anchor:"transformers.FSMTModel.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.FSMTModel.forward.decoder_head_mask",description:`<strong>decoder_head_mask</strong> (<code>torch.Tensor</code> of shape <code>(decoder_layers, decoder_attention_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the attention modules in the decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"decoder_head_mask"},{anchor:"transformers.FSMTModel.forward.cross_attn_head_mask",description:`<strong>cross_attn_head_mask</strong> (<code>torch.Tensor</code> of shape <code>(decoder_layers, decoder_attention_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the cross-attention modules in the decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"cross_attn_head_mask"},{anchor:"transformers.FSMTModel.forward.encoder_outputs",description:`<strong>encoder_outputs</strong> (<code>tuple[torch.FloatTensor]</code>, <em>optional</em>) &#x2014;
Tuple consists of (<code>last_hidden_state</code>, <em>optional</em>: <code>hidden_states</code>, <em>optional</em>: <code>attentions</code>)
<code>last_hidden_state</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) is a sequence of
hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.`,name:"encoder_outputs"},{anchor:"transformers.FSMTModel.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>tuple[torch.FloatTensor]</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.FSMTModel.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.FSMTModel.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.FSMTModel.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.FSMTModel.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.FSMTModel.forward.decoder_inputs_embeds",description:`<strong>decoder_inputs_embeds</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, target_sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>decoder_input_ids</code> you can choose to directly pass an embedded
representation. If <code>past_key_values</code> is used, optionally only the last <code>decoder_inputs_embeds</code> have to be
input (see <code>past_key_values</code>). This is useful if you want more control over how to convert
<code>decoder_input_ids</code> indices into associated vectors than the model&#x2019;s internal embedding lookup matrix.</p>
<p>If <code>decoder_input_ids</code> and <code>decoder_inputs_embeds</code> are both unset, <code>decoder_inputs_embeds</code> takes the value
of <code>inputs_embeds</code>.`,name:"decoder_inputs_embeds"},{anchor:"transformers.FSMTModel.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.FSMTModel.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.Tensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/fsmt/modeling_fsmt.py#L914",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqModelOutput"
>transformers.modeling_outputs.Seq2SeqModelOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/fsmt#transformers.FSMTConfig"
>FSMTConfig</a>) and inputs.</p>
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
`}}),U=new oo({props:{$$slots:{default:[ho]},$$scope:{ctx:I}}}),me=new ge({props:{title:"FSMTForConditionalGeneration",local:"transformers.FSMTForConditionalGeneration",headingTag:"h2"}}),pe=new W({props:{name:"class transformers.FSMTForConditionalGeneration",anchor:"transformers.FSMTForConditionalGeneration",parameters:[{name:"config",val:": FSMTConfig"}],parametersDescription:[{anchor:"transformers.FSMTForConditionalGeneration.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/fsmt#transformers.FSMTConfig">FSMTConfig</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/fsmt/modeling_fsmt.py#L1048"}}),he=new W({props:{name:"forward",anchor:"transformers.FSMTForConditionalGeneration.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"decoder_input_ids",val:": typing.Optional[torch.LongTensor] = None"},{name:"decoder_attention_mask",val:": typing.Optional[torch.BoolTensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"decoder_head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"cross_attn_head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_outputs",val:": typing.Optional[tuple[torch.FloatTensor]] = None"},{name:"past_key_values",val:": typing.Optional[tuple[torch.FloatTensor]] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"decoder_inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"labels",val:": typing.Optional[torch.LongTensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.Tensor] = None"}],parametersDescription:[{anchor:"transformers.FSMTForConditionalGeneration.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.FSMTForConditionalGeneration.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.FSMTForConditionalGeneration.forward.decoder_input_ids",description:`<strong>decoder_input_ids</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of decoder input sequence tokens in the vocabulary.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#decoder-input-ids">What are decoder input IDs?</a></p>
<p>FSMT uses the <code>eos_token_id</code> as the starting token for <code>decoder_input_ids</code> generation. If <code>past_key_values</code>
is used, optionally only the last <code>decoder_input_ids</code> have to be input (see <code>past_key_values</code>).`,name:"decoder_input_ids"},{anchor:"transformers.FSMTForConditionalGeneration.forward.decoder_attention_mask",description:`<strong>decoder_attention_mask</strong> (<code>torch.BoolTensor</code> of shape <code>(batch_size, target_sequence_length)</code>, <em>optional</em>) &#x2014;
Default behavior: generate a tensor that ignores pad tokens in <code>decoder_input_ids</code>. Causal mask will also
be used by default.`,name:"decoder_attention_mask"},{anchor:"transformers.FSMTForConditionalGeneration.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.FSMTForConditionalGeneration.forward.decoder_head_mask",description:`<strong>decoder_head_mask</strong> (<code>torch.Tensor</code> of shape <code>(decoder_layers, decoder_attention_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the attention modules in the decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"decoder_head_mask"},{anchor:"transformers.FSMTForConditionalGeneration.forward.cross_attn_head_mask",description:`<strong>cross_attn_head_mask</strong> (<code>torch.Tensor</code> of shape <code>(decoder_layers, decoder_attention_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the cross-attention modules in the decoder. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"cross_attn_head_mask"},{anchor:"transformers.FSMTForConditionalGeneration.forward.encoder_outputs",description:`<strong>encoder_outputs</strong> (<code>tuple[torch.FloatTensor]</code>, <em>optional</em>) &#x2014;
Tuple consists of (<code>last_hidden_state</code>, <em>optional</em>: <code>hidden_states</code>, <em>optional</em>: <code>attentions</code>)
<code>last_hidden_state</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) is a sequence of
hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.`,name:"encoder_outputs"},{anchor:"transformers.FSMTForConditionalGeneration.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>tuple[torch.FloatTensor]</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.FSMTForConditionalGeneration.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.FSMTForConditionalGeneration.forward.decoder_inputs_embeds",description:`<strong>decoder_inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, target_sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>decoder_input_ids</code> you can choose to directly pass an embedded
representation. If <code>past_key_values</code> is used, optionally only the last <code>decoder_inputs_embeds</code> have to be
input (see <code>past_key_values</code>). This is useful if you want more control over how to convert
<code>decoder_input_ids</code> indices into associated vectors than the model&#x2019;s internal embedding lookup matrix.</p>
<p>If <code>decoder_input_ids</code> and <code>decoder_inputs_embeds</code> are both unset, <code>decoder_inputs_embeds</code> takes the value
of <code>inputs_embeds</code>.`,name:"decoder_inputs_embeds"},{anchor:"transformers.FSMTForConditionalGeneration.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.FSMTForConditionalGeneration.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.FSMTForConditionalGeneration.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.FSMTForConditionalGeneration.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.FSMTForConditionalGeneration.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.FSMTForConditionalGeneration.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.Tensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/fsmt/modeling_fsmt.py#L1060",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqLMOutput"
>transformers.modeling_outputs.Seq2SeqLMOutput</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/fsmt#transformers.FSMTConfig"
>FSMTConfig</a>) and inputs.</p>
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
`}}),V=new oo({props:{$$slots:{default:[uo]},$$scope:{ctx:I}}}),A=new no({props:{anchor:"transformers.FSMTForConditionalGeneration.forward.example",$$slots:{default:[fo]},$$scope:{ctx:I}}}),ue=new mo({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/fsmt.md"}}),{c(){r=i("meta"),M=s(),p=i("p"),h=s(),y=i("p"),y.innerHTML=l,$=s(),u(R.$$.fragment),Ne=s(),u(B.$$.fragment),Oe=s(),Z=i("p"),Z.innerHTML=It,He=s(),J=i("p"),J.textContent=Lt,Ue=s(),Q=i("p"),Q.innerHTML=Dt,Ve=s(),X=i("p"),X.innerHTML=Et,Ae=s(),u(Y.$$.fragment),Ge=s(),K=i("ul"),K.innerHTML=jt,Re=s(),u(ee.$$.fragment),Be=s(),x=i("div"),u(te.$$.fragment),st=s(),_e=i("p"),_e.innerHTML=Pt,at=s(),Te=i("p"),Te.innerHTML=Wt,rt=s(),u(O.$$.fragment),Ze=s(),u(oe.$$.fragment),Je=s(),k=i("div"),u(ne.$$.fragment),it=s(),be=i("p"),be.textContent=Nt,dt=s(),ke=i("ul"),ke.innerHTML=Ot,ct=s(),ve=i("p"),ve.innerHTML=Ht,lt=s(),L=i("div"),u(se.$$.fragment),mt=s(),ye=i("p"),ye.textContent=Ut,pt=s(),Me=i("ul"),Me.innerHTML=Vt,ht=s(),H=i("div"),u(ae.$$.fragment),ut=s(),we=i("p"),we.innerHTML=At,ft=s(),D=i("div"),u(re.$$.fragment),gt=s(),Fe=i("p"),Fe.innerHTML=Gt,_t=s(),xe=i("p"),xe.textContent=Rt,Tt=s(),$e=i("div"),u(ie.$$.fragment),Qe=s(),u(de.$$.fragment),Xe=s(),w=i("div"),u(ce.$$.fragment),bt=s(),Ce=i("p"),Ce.textContent=Bt,kt=s(),Se=i("p"),Se.innerHTML=Zt,vt=s(),ze=i("p"),ze.innerHTML=Jt,yt=s(),E=i("div"),u(le.$$.fragment),Mt=s(),qe=i("p"),qe.innerHTML=Qt,wt=s(),u(U.$$.fragment),Ye=s(),u(me.$$.fragment),Ke=s(),F=i("div"),u(pe.$$.fragment),Ft=s(),Ie=i("p"),Ie.textContent=Xt,xt=s(),Le=i("p"),Le.innerHTML=Yt,$t=s(),De=i("p"),De.innerHTML=Kt,Ct=s(),C=i("div"),u(he.$$.fragment),St=s(),Ee=i("p"),Ee.innerHTML=eo,zt=s(),u(V.$$.fragment),qt=s(),u(A.$$.fragment),et=s(),u(ue.$$.fragment),tt=s(),We=i("p"),this.h()},l(e){const o=lo("svelte-u9bgzb",document.head);r=d(o,"META",{name:!0,content:!0}),o.forEach(n),M=a(e),p=d(e,"P",{}),S(p).forEach(n),h=a(e),y=d(e,"P",{"data-svelte-h":!0}),m(y)!=="svelte-1r4sfvc"&&(y.innerHTML=l),$=a(e),f(R.$$.fragment,e),Ne=a(e),f(B.$$.fragment,e),Oe=a(e),Z=d(e,"P",{"data-svelte-h":!0}),m(Z)!=="svelte-1p5kl8i"&&(Z.innerHTML=It),He=a(e),J=d(e,"P",{"data-svelte-h":!0}),m(J)!=="svelte-wu27l3"&&(J.textContent=Lt),Ue=a(e),Q=d(e,"P",{"data-svelte-h":!0}),m(Q)!=="svelte-75g4jk"&&(Q.innerHTML=Dt),Ve=a(e),X=d(e,"P",{"data-svelte-h":!0}),m(X)!=="svelte-1uemjgo"&&(X.innerHTML=Et),Ae=a(e),f(Y.$$.fragment,e),Ge=a(e),K=d(e,"UL",{"data-svelte-h":!0}),m(K)!=="svelte-vq04m1"&&(K.innerHTML=jt),Re=a(e),f(ee.$$.fragment,e),Be=a(e),x=d(e,"DIV",{class:!0});var q=S(x);f(te.$$.fragment,q),st=a(q),_e=d(q,"P",{"data-svelte-h":!0}),m(_e)!=="svelte-6fsdlh"&&(_e.innerHTML=Pt),at=a(q),Te=d(q,"P",{"data-svelte-h":!0}),m(Te)!=="svelte-1ek1ss9"&&(Te.innerHTML=Wt),rt=a(q),f(O.$$.fragment,q),q.forEach(n),Ze=a(e),f(oe.$$.fragment,e),Je=a(e),k=d(e,"DIV",{class:!0});var v=S(k);f(ne.$$.fragment,v),it=a(v),be=d(v,"P",{"data-svelte-h":!0}),m(be)!=="svelte-5bbjru"&&(be.textContent=Nt),dt=a(v),ke=d(v,"UL",{"data-svelte-h":!0}),m(ke)!=="svelte-stmybw"&&(ke.innerHTML=Ot),ct=a(v),ve=d(v,"P",{"data-svelte-h":!0}),m(ve)!=="svelte-ntrhio"&&(ve.innerHTML=Ht),lt=a(v),L=d(v,"DIV",{class:!0});var N=S(L);f(se.$$.fragment,N),mt=a(N),ye=d(N,"P",{"data-svelte-h":!0}),m(ye)!=="svelte-ym5sov"&&(ye.textContent=Ut),pt=a(N),Me=d(N,"UL",{"data-svelte-h":!0}),m(Me)!=="svelte-1w73b42"&&(Me.innerHTML=Vt),N.forEach(n),ht=a(v),H=d(v,"DIV",{class:!0});var fe=S(H);f(ae.$$.fragment,fe),ut=a(fe),we=d(fe,"P",{"data-svelte-h":!0}),m(we)!=="svelte-1f4f5kp"&&(we.innerHTML=At),fe.forEach(n),ft=a(v),D=d(v,"DIV",{class:!0});var je=S(D);f(re.$$.fragment,je),gt=a(je),Fe=d(je,"P",{"data-svelte-h":!0}),m(Fe)!=="svelte-zj1vf1"&&(Fe.innerHTML=Gt),_t=a(je),xe=d(je,"P",{"data-svelte-h":!0}),m(xe)!=="svelte-9vptpw"&&(xe.textContent=Rt),je.forEach(n),Tt=a(v),$e=d(v,"DIV",{class:!0});var to=S($e);f(ie.$$.fragment,to),to.forEach(n),v.forEach(n),Qe=a(e),f(de.$$.fragment,e),Xe=a(e),w=d(e,"DIV",{class:!0});var j=S(w);f(ce.$$.fragment,j),bt=a(j),Ce=d(j,"P",{"data-svelte-h":!0}),m(Ce)!=="svelte-1n3c9c6"&&(Ce.textContent=Bt),kt=a(j),Se=d(j,"P",{"data-svelte-h":!0}),m(Se)!=="svelte-q52n56"&&(Se.innerHTML=Zt),vt=a(j),ze=d(j,"P",{"data-svelte-h":!0}),m(ze)!=="svelte-hswkmf"&&(ze.innerHTML=Jt),yt=a(j),E=d(j,"DIV",{class:!0});var Pe=S(E);f(le.$$.fragment,Pe),Mt=a(Pe),qe=d(Pe,"P",{"data-svelte-h":!0}),m(qe)!=="svelte-1ukm7uh"&&(qe.innerHTML=Qt),wt=a(Pe),f(U.$$.fragment,Pe),Pe.forEach(n),j.forEach(n),Ye=a(e),f(me.$$.fragment,e),Ke=a(e),F=d(e,"DIV",{class:!0});var P=S(F);f(pe.$$.fragment,P),Ft=a(P),Ie=d(P,"P",{"data-svelte-h":!0}),m(Ie)!=="svelte-1449fju"&&(Ie.textContent=Xt),xt=a(P),Le=d(P,"P",{"data-svelte-h":!0}),m(Le)!=="svelte-q52n56"&&(Le.innerHTML=Yt),$t=a(P),De=d(P,"P",{"data-svelte-h":!0}),m(De)!=="svelte-hswkmf"&&(De.innerHTML=Kt),Ct=a(P),C=d(P,"DIV",{class:!0});var G=S(C);f(he.$$.fragment,G),St=a(G),Ee=d(G,"P",{"data-svelte-h":!0}),m(Ee)!=="svelte-htjc23"&&(Ee.innerHTML=eo),zt=a(G),f(V.$$.fragment,G),qt=a(G),f(A.$$.fragment,G),G.forEach(n),P.forEach(n),et=a(e),f(ue.$$.fragment,e),tt=a(e),We=d(e,"P",{}),S(We).forEach(n),this.h()},h(){z(r,"name","hf:doc:metadata"),z(r,"content",_o),z(x,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),z(L,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),z(H,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),z(D,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),z($e,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),z(k,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),z(E,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),z(w,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),z(C,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),z(F,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,o){t(document.head,r),c(e,M,o),c(e,p,o),c(e,h,o),c(e,y,o),c(e,$,o),g(R,e,o),c(e,Ne,o),g(B,e,o),c(e,Oe,o),c(e,Z,o),c(e,He,o),c(e,J,o),c(e,Ue,o),c(e,Q,o),c(e,Ve,o),c(e,X,o),c(e,Ae,o),g(Y,e,o),c(e,Ge,o),c(e,K,o),c(e,Re,o),g(ee,e,o),c(e,Be,o),c(e,x,o),g(te,x,null),t(x,st),t(x,_e),t(x,at),t(x,Te),t(x,rt),g(O,x,null),c(e,Ze,o),g(oe,e,o),c(e,Je,o),c(e,k,o),g(ne,k,null),t(k,it),t(k,be),t(k,dt),t(k,ke),t(k,ct),t(k,ve),t(k,lt),t(k,L),g(se,L,null),t(L,mt),t(L,ye),t(L,pt),t(L,Me),t(k,ht),t(k,H),g(ae,H,null),t(H,ut),t(H,we),t(k,ft),t(k,D),g(re,D,null),t(D,gt),t(D,Fe),t(D,_t),t(D,xe),t(k,Tt),t(k,$e),g(ie,$e,null),c(e,Qe,o),g(de,e,o),c(e,Xe,o),c(e,w,o),g(ce,w,null),t(w,bt),t(w,Ce),t(w,kt),t(w,Se),t(w,vt),t(w,ze),t(w,yt),t(w,E),g(le,E,null),t(E,Mt),t(E,qe),t(E,wt),g(U,E,null),c(e,Ye,o),g(me,e,o),c(e,Ke,o),c(e,F,o),g(pe,F,null),t(F,Ft),t(F,Ie),t(F,xt),t(F,Le),t(F,$t),t(F,De),t(F,Ct),t(F,C),g(he,C,null),t(C,St),t(C,Ee),t(C,zt),g(V,C,null),t(C,qt),g(A,C,null),c(e,et,o),g(ue,e,o),c(e,tt,o),c(e,We,o),ot=!0},p(e,[o]){const q={};o&2&&(q.$$scope={dirty:o,ctx:e}),O.$set(q);const v={};o&2&&(v.$$scope={dirty:o,ctx:e}),U.$set(v);const N={};o&2&&(N.$$scope={dirty:o,ctx:e}),V.$set(N);const fe={};o&2&&(fe.$$scope={dirty:o,ctx:e}),A.$set(fe)},i(e){ot||(_(R.$$.fragment,e),_(B.$$.fragment,e),_(Y.$$.fragment,e),_(ee.$$.fragment,e),_(te.$$.fragment,e),_(O.$$.fragment,e),_(oe.$$.fragment,e),_(ne.$$.fragment,e),_(se.$$.fragment,e),_(ae.$$.fragment,e),_(re.$$.fragment,e),_(ie.$$.fragment,e),_(de.$$.fragment,e),_(ce.$$.fragment,e),_(le.$$.fragment,e),_(U.$$.fragment,e),_(me.$$.fragment,e),_(pe.$$.fragment,e),_(he.$$.fragment,e),_(V.$$.fragment,e),_(A.$$.fragment,e),_(ue.$$.fragment,e),ot=!0)},o(e){T(R.$$.fragment,e),T(B.$$.fragment,e),T(Y.$$.fragment,e),T(ee.$$.fragment,e),T(te.$$.fragment,e),T(O.$$.fragment,e),T(oe.$$.fragment,e),T(ne.$$.fragment,e),T(se.$$.fragment,e),T(ae.$$.fragment,e),T(re.$$.fragment,e),T(ie.$$.fragment,e),T(de.$$.fragment,e),T(ce.$$.fragment,e),T(le.$$.fragment,e),T(U.$$.fragment,e),T(me.$$.fragment,e),T(pe.$$.fragment,e),T(he.$$.fragment,e),T(V.$$.fragment,e),T(A.$$.fragment,e),T(ue.$$.fragment,e),ot=!1},d(e){e&&(n(M),n(p),n(h),n(y),n($),n(Ne),n(Oe),n(Z),n(He),n(J),n(Ue),n(Q),n(Ve),n(X),n(Ae),n(Ge),n(K),n(Re),n(Be),n(x),n(Ze),n(Je),n(k),n(Qe),n(Xe),n(w),n(Ye),n(Ke),n(F),n(et),n(tt),n(We)),n(r),b(R,e),b(B,e),b(Y,e),b(ee,e),b(te),b(O),b(oe,e),b(ne),b(se),b(ae),b(re),b(ie),b(de,e),b(ce),b(le),b(U),b(me,e),b(pe),b(he),b(V),b(A),b(ue,e)}}}const _o='{"title":"FSMT","local":"fsmt","sections":[{"title":"Overview","local":"overview","sections":[],"depth":2},{"title":"Implementation Notes","local":"implementation-notes","sections":[],"depth":2},{"title":"FSMTConfig","local":"transformers.FSMTConfig","sections":[],"depth":2},{"title":"FSMTTokenizer","local":"transformers.FSMTTokenizer","sections":[],"depth":2},{"title":"FSMTModel","local":"transformers.FSMTModel","sections":[],"depth":2},{"title":"FSMTForConditionalGeneration","local":"transformers.FSMTForConditionalGeneration","sections":[],"depth":2}],"depth":1}';function To(I){return ro(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class xo extends io{constructor(r){super(),co(this,r,To,go,ao,{})}}export{xo as component};
