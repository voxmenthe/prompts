import{s as qo,o as Fo,n as vt}from"../chunks/scheduler.18a86fab.js";import{S as Po,i as Io,g as a,s,r as p,m as Do,A as Oo,h as i,f as o,c as r,j as G,x as m,u,n as No,k as w,y as t,a as d,v as h,d as f,t as g,w as _}from"../chunks/index.98837b22.js";import{T as zo}from"../chunks/Tip.77304350.js";import{D as C}from"../chunks/Docstring.a1ef7999.js";import{C as Xo}from"../chunks/CodeBlock.8d0c2e8a.js";import{E as Co}from"../chunks/ExampleCodeBlock.8c3ee1f9.js";import{H as Y,E as Ho}from"../chunks/getInferenceSnippets.06c2775f.js";function Wo(P){let l,y="Example:",k,b,T;return b=new Xo({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMFhHTE1Nb2RlbCUyQyUyMFhHTE1Db25maWclMEElMEElMjMlMjBJbml0aWFsaXppbmclMjBhJTIwWEdMTSUyMGZhY2Vib29rJTJGeGdsbS01NjRNJTIwc3R5bGUlMjBjb25maWd1cmF0aW9uJTBBY29uZmlndXJhdGlvbiUyMCUzRCUyMFhHTE1Db25maWcoKSUwQSUwQSUyMyUyMEluaXRpYWxpemluZyUyMGElMjBtb2RlbCUyMGZyb20lMjB0aGUlMjBmYWNlYm9vayUyRnhnbG0tNTY0TSUyMHN0eWxlJTIwY29uZmlndXJhdGlvbiUwQW1vZGVsJTIwJTNEJTIwWEdMTU1vZGVsKGNvbmZpZ3VyYXRpb24pJTBBJTBBJTIzJTIwQWNjZXNzaW5nJTIwdGhlJTIwbW9kZWwlMjBjb25maWd1cmF0aW9uJTBBY29uZmlndXJhdGlvbiUyMCUzRCUyMG1vZGVsLmNvbmZpZw==",highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> XGLMModel, XGLMConfig

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a XGLM facebook/xglm-564M style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = XGLMConfig()

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Initializing a model from the facebook/xglm-564M style configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model = XGLMModel(configuration)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Accessing the model configuration</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>configuration = model.config`,wrap:!1}}),{c(){l=a("p"),l.textContent=y,k=s(),p(b.$$.fragment)},l(c){l=i(c,"P",{"data-svelte-h":!0}),m(l)!=="svelte-11lpom8"&&(l.textContent=y),k=r(c),u(b.$$.fragment,c)},m(c,X){d(c,l,X),d(c,k,X),h(b,c,X),T=!0},p:vt,i(c){T||(f(b.$$.fragment,c),T=!0)},o(c){g(b.$$.fragment,c),T=!1},d(c){c&&(o(l),o(k)),_(b,c)}}}function Eo(P){let l,y=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){l=a("p"),l.innerHTML=y},l(k){l=i(k,"P",{"data-svelte-h":!0}),m(l)!=="svelte-fincs2"&&(l.innerHTML=y)},m(k,b){d(k,l,b)},p:vt,d(k){k&&o(l)}}}function So(P){let l,y=`Although the recipe for forward pass needs to be defined within this function, one should call the <code>Module</code>
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.`;return{c(){l=a("p"),l.innerHTML=y},l(k){l=i(k,"P",{"data-svelte-h":!0}),m(l)!=="svelte-fincs2"&&(l.innerHTML=y)},m(k,b){d(k,l,b)},p:vt,d(k){k&&o(l)}}}function Bo(P){let l,y="Example:",k,b,T;return b=new Xo({props:{code:"",highlighted:"",wrap:!1}}),{c(){l=a("p"),l.textContent=y,k=s(),p(b.$$.fragment)},l(c){l=i(c,"P",{"data-svelte-h":!0}),m(l)!=="svelte-11lpom8"&&(l.textContent=y),k=r(c),u(b.$$.fragment,c)},m(c,X){d(c,l,X),d(c,k,X),h(b,c,X),T=!0},p:vt,i(c){T||(f(b.$$.fragment,c),T=!0)},o(c){g(b.$$.fragment,c),T=!1},d(c){c&&(o(l),o(k)),_(b,c)}}}function Ao(P){let l,y,k,b,T,c="<em>This model was released on 2021-12-20 and added to Hugging Face Transformers on 2022-01-28.</em>",X,Q,Ye,S,eo='<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&amp;logo=pytorch&amp;logoColor=white"/>',Qe,K,Ke,ee,to=`The XGLM model was proposed in <a href="https://huggingface.co/papers/2112.10668" rel="nofollow">Few-shot Learning with Multilingual Language Models</a>
by Xi Victoria Lin, Todor Mihaylov, Mikel Artetxe, Tianlu Wang, Shuohui Chen, Daniel Simig, Myle Ott, Naman Goyal,
Shruti Bhosale, Jingfei Du, Ramakanth Pasunuru, Sam Shleifer, Punit Singh Koura, Vishrav Chaudhary, Brian O’Horo,
Jeff Wang, Luke Zettlemoyer, Zornitsa Kozareva, Mona Diab, Veselin Stoyanov, Xian Li.`,et,te,oo="The abstract from the paper is the following:",tt,oe,no=`<em>Large-scale autoregressive language models such as GPT-3 are few-shot learners that can perform a wide range of language
tasks without fine-tuning. While these models are known to be able to jointly represent many different languages,
their training data is dominated by English, potentially limiting their cross-lingual generalization.
In this work, we train multilingual autoregressive language models on a balanced corpus covering a diverse set of languages,
and study their few- and zero-shot learning capabilities in a wide range of tasks. Our largest model with 7.5 billion parameters
sets new state of the art in few-shot learning in more than 20 representative languages, outperforming GPT-3 of comparable size
in multilingual commonsense reasoning (with +7.4% absolute accuracy improvement in 0-shot settings and +9.4% in 4-shot settings)
and natural language inference (+5.4% in each of 0-shot and 4-shot settings). On the FLORES-101 machine translation benchmark,
our model outperforms GPT-3 on 171 out of 182 translation directions with 32 training examples, while surpassing the
official supervised baseline in 45 directions. We present a detailed analysis of where the model succeeds and fails,
showing in particular that it enables cross-lingual in-context learning on some tasks, while there is still room for improvement
on surface form robustness and adaptation to tasks that do not have a natural cloze form. Finally, we evaluate our models
in social value tasks such as hate speech detection in five languages and find it has limitations similar to comparable sized GPT-3 models.</em>`,ot,ne,so='This model was contributed by <a href="https://huggingface.co/valhalla" rel="nofollow">Suraj</a>. The original code can be found <a href="https://github.com/pytorch/fairseq/tree/main/examples/xglm" rel="nofollow">here</a>.',nt,se,st,re,ro='<li><a href="../tasks/language_modeling">Causal language modeling task guide</a></li>',rt,ae,at,z,ie,Tt,xe,ao=`This is the configuration class to store the configuration of a <a href="/docs/transformers/v4.56.2/en/model_doc/xglm#transformers.XGLMModel">XGLMModel</a>. It is used to instantiate an XGLM
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the XGLM
<a href="https://huggingface.co/facebook/xglm-564M" rel="nofollow">facebook/xglm-564M</a> architecture.`,Mt,$e,io=`Configuration objects inherit from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> and can be used to control the model outputs. Read the
documentation from <a href="/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig">PretrainedConfig</a> for more information.`,yt,B,it,de,dt,v,le,wt,Ge,lo=`Adapted from <a href="/docs/transformers/v4.56.2/en/model_doc/roberta#transformers.RobertaTokenizer">RobertaTokenizer</a> and <a href="/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.XLNetTokenizer">XLNetTokenizer</a>. Based on
<a href="https://github.com/google/sentencepiece" rel="nofollow">SentencePiece</a>.`,Lt,ze,co=`This tokenizer inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer">PreTrainedTokenizer</a> which contains most of the main methods. Users should refer to
this superclass for more information regarding those methods.`,xt,I,ce,$t,Ce,mo=`Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
adding special tokens. An XLM-RoBERTa sequence has the following format:`,Gt,Xe,po="<li>single sequence: <code>&lt;s&gt; X &lt;/s&gt;</code></li> <li>pair of sequences: <code>&lt;s&gt; A &lt;/s&gt;&lt;/s&gt; B &lt;/s&gt;</code></li>",zt,A,me,Ct,qe,uo=`Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
special tokens using the tokenizer <code>prepare_for_model</code> method.`,Xt,U,pe,qt,Fe,ho=`Create a mask from the two sequences passed to be used in a sequence-pair classification task. XLM-RoBERTa does
not make use of token type ids, therefore a list of zeros is returned.`,Ft,Pe,ue,lt,he,ct,L,fe,Pt,Ie,fo=`Construct a “fast” XGLM tokenizer (backed by HuggingFace’s <em>tokenizers</em> library). Adapted from <a href="/docs/transformers/v4.56.2/en/model_doc/roberta#transformers.RobertaTokenizer">RobertaTokenizer</a>
and <a href="/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.XLNetTokenizer">XLNetTokenizer</a>. Based on
<a href="https://huggingface.co/docs/tokenizers/python/latest/components.html?highlight=BPE#models" rel="nofollow">BPE</a>.`,It,De,go=`This tokenizer inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast">PreTrainedTokenizerFast</a> which contains most of the main methods. Users should
refer to this superclass for more information regarding those methods.`,Dt,D,ge,Ot,Oe,_o=`Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
adding special tokens. An XLM-RoBERTa sequence has the following format:`,Nt,Ne,ko="<li>single sequence: <code>&lt;s&gt; X &lt;/s&gt;</code></li> <li>pair of sequences: <code>&lt;s&gt; A &lt;/s&gt;&lt;/s&gt; B &lt;/s&gt;</code></li>",Ht,j,_e,Wt,He,bo=`Create a mask from the two sequences passed to be used in a sequence-pair classification task. XLM-RoBERTa does
not make use of token type ids, therefore a list of zeros is returned.`,mt,ke,pt,x,be,Et,We,vo="The bare Xglm Model outputting raw hidden-states without any specific head on top.",St,Ee,To=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Bt,Se,Mo=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,At,O,ve,Ut,Be,yo='The <a href="/docs/transformers/v4.56.2/en/model_doc/xglm#transformers.XGLMModel">XGLMModel</a> forward method, overrides the <code>__call__</code> special method.',jt,R,ut,Te,ht,$,Me,Rt,Ae,wo=`The XGLM Model transformer with a language modeling head on top (linear layer with weights tied to the input
embeddings).`,Vt,Ue,Lo=`This model inherits from <a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel">PreTrainedModel</a>. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)`,Jt,je,xo=`This model is also a PyTorch <a href="https://pytorch.org/docs/stable/nn.html#torch.nn.Module" rel="nofollow">torch.nn.Module</a> subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.`,Zt,q,ye,Yt,Re,$o='The <a href="/docs/transformers/v4.56.2/en/model_doc/xglm#transformers.XGLMForCausalLM">XGLMForCausalLM</a> forward method, overrides the <code>__call__</code> special method.',Qt,V,Kt,J,ft,we,gt,Ze,_t;return Q=new Y({props:{title:"XGLM",local:"xglm",headingTag:"h1"}}),K=new Y({props:{title:"Overview",local:"overview",headingTag:"h2"}}),se=new Y({props:{title:"Resources",local:"resources",headingTag:"h2"}}),ae=new Y({props:{title:"XGLMConfig",local:"transformers.XGLMConfig",headingTag:"h2"}}),ie=new C({props:{name:"class transformers.XGLMConfig",anchor:"transformers.XGLMConfig",parameters:[{name:"vocab_size",val:" = 256008"},{name:"max_position_embeddings",val:" = 2048"},{name:"d_model",val:" = 1024"},{name:"ffn_dim",val:" = 4096"},{name:"num_layers",val:" = 24"},{name:"attention_heads",val:" = 16"},{name:"activation_function",val:" = 'gelu'"},{name:"dropout",val:" = 0.1"},{name:"attention_dropout",val:" = 0.1"},{name:"activation_dropout",val:" = 0.0"},{name:"layerdrop",val:" = 0.0"},{name:"init_std",val:" = 0.02"},{name:"scale_embedding",val:" = True"},{name:"use_cache",val:" = True"},{name:"decoder_start_token_id",val:" = 2"},{name:"pad_token_id",val:" = 1"},{name:"bos_token_id",val:" = 0"},{name:"eos_token_id",val:" = 2"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.XGLMConfig.vocab_size",description:`<strong>vocab_size</strong> (<code>int</code>, <em>optional</em>, defaults to 256008) &#x2014;
Vocabulary size of the XGLM model. Defines the number of different tokens that can be represented by the
<code>inputs_ids</code> passed when calling <a href="/docs/transformers/v4.56.2/en/model_doc/xglm#transformers.XGLMModel">XGLMModel</a> or <code>FlaxXGLMModel</code>.`,name:"vocab_size"},{anchor:"transformers.XGLMConfig.max_position_embeddings",description:`<strong>max_position_embeddings</strong> (<code>int</code>, <em>optional</em>, defaults to 2048) &#x2014;
The maximum sequence length that this model might ever be used with. Typically set this to something large
just in case (e.g., 512 or 1024 or 2048).`,name:"max_position_embeddings"},{anchor:"transformers.XGLMConfig.d_model",description:`<strong>d_model</strong> (<code>int</code>, <em>optional</em>, defaults to 1024) &#x2014;
Dimension of the layers and the pooler layer.`,name:"d_model"},{anchor:"transformers.XGLMConfig.ffn_dim",description:`<strong>ffn_dim</strong> (<code>int</code>, <em>optional</em>, defaults to 4096) &#x2014;
Dimension of the &#x201C;intermediate&#x201D; (often named feed-forward) layer in decoder.`,name:"ffn_dim"},{anchor:"transformers.XGLMConfig.num_layers",description:`<strong>num_layers</strong> (<code>int</code>, <em>optional</em>, defaults to 24) &#x2014;
Number of hidden layers Transformer decoder.`,name:"num_layers"},{anchor:"transformers.XGLMConfig.attention_heads",description:`<strong>attention_heads</strong> (<code>int</code>, <em>optional</em>, defaults to 16) &#x2014;
Number of attention heads for each attention layer in the Transformer decoder.`,name:"attention_heads"},{anchor:"transformers.XGLMConfig.activation_function",description:`<strong>activation_function</strong> (<code>str</code> or <code>function</code>, <em>optional</em>, defaults to <code>&quot;gelu&quot;</code>) &#x2014;
The non-linear activation function (function or string) in the encoder and pooler. If string, <code>&quot;gelu&quot;</code>,
<code>&quot;relu&quot;</code>, <code>&quot;silu&quot;</code> and <code>&quot;gelu_new&quot;</code> are supported.`,name:"activation_function"},{anchor:"transformers.XGLMConfig.dropout",description:`<strong>dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout probability for all fully connected layers in the embeddings, dencoder, and pooler.`,name:"dropout"},{anchor:"transformers.XGLMConfig.attention_dropout",description:`<strong>attention_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.1) &#x2014;
The dropout ratio for the attention probabilities.`,name:"attention_dropout"},{anchor:"transformers.XGLMConfig.activation_dropout",description:`<strong>activation_dropout</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The dropout ratio for activations inside the fully connected layer.`,name:"activation_dropout"},{anchor:"transformers.XGLMConfig.layerdrop",description:`<strong>layerdrop</strong> (<code>float</code>, <em>optional</em>, defaults to 0.0) &#x2014;
The LayerDrop probability for the encoder. See the [LayerDrop paper](see <a href="https://huggingface.co/papers/1909.11556" rel="nofollow">https://huggingface.co/papers/1909.11556</a>)
for more details.`,name:"layerdrop"},{anchor:"transformers.XGLMConfig.init_std",description:`<strong>init_std</strong> (<code>float</code>, <em>optional</em>, defaults to 0.02) &#x2014;
The standard deviation of the truncated_normal_initializer for initializing all weight matrices.`,name:"init_std"},{anchor:"transformers.XGLMConfig.scale_embedding",description:`<strong>scale_embedding</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Scale embeddings by diving by sqrt(d_model).`,name:"scale_embedding"},{anchor:"transformers.XGLMConfig.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>True</code>) &#x2014;
Whether or not the model should return the last key/values attentions (not used by all models).`,name:"use_cache"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xglm/configuration_xglm.py#L24"}}),B=new Co({props:{anchor:"transformers.XGLMConfig.example",$$slots:{default:[Wo]},$$scope:{ctx:P}}}),de=new Y({props:{title:"XGLMTokenizer",local:"transformers.XGLMTokenizer",headingTag:"h2"}}),le=new C({props:{name:"class transformers.XGLMTokenizer",anchor:"transformers.XGLMTokenizer",parameters:[{name:"vocab_file",val:""},{name:"bos_token",val:" = '<s>'"},{name:"eos_token",val:" = '</s>'"},{name:"sep_token",val:" = '</s>'"},{name:"cls_token",val:" = '<s>'"},{name:"unk_token",val:" = '<unk>'"},{name:"pad_token",val:" = '<pad>'"},{name:"sp_model_kwargs",val:": typing.Optional[dict[str, typing.Any]] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.XGLMTokenizer.vocab_file",description:`<strong>vocab_file</strong> (<code>str</code>) &#x2014;
Path to the vocabulary file.`,name:"vocab_file"},{anchor:"transformers.XGLMTokenizer.bos_token",description:`<strong>bos_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;s&gt;&quot;</code>) &#x2014;
The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.</p>
<div class="course-tip  bg-gradient-to-br dark:bg-gradient-to-r before:border-green-500 dark:before:border-green-800 from-green-50 dark:from-gray-900 to-white dark:to-gray-950 border border-green-50 text-green-700 dark:text-gray-400">
						
<p>When building a sequence using special tokens, this is not the token that is used for the beginning of
sequence. The token used is the <code>cls_token</code>.</p>

					</div>`,name:"bos_token"},{anchor:"transformers.XGLMTokenizer.eos_token",description:`<strong>eos_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;/s&gt;&quot;</code>) &#x2014;
The end of sequence token.</p>
<div class="course-tip  bg-gradient-to-br dark:bg-gradient-to-r before:border-green-500 dark:before:border-green-800 from-green-50 dark:from-gray-900 to-white dark:to-gray-950 border border-green-50 text-green-700 dark:text-gray-400">
						
<p>When building a sequence using special tokens, this is not the token that is used for the end of sequence.
The token used is the <code>sep_token</code>.</p>

					</div>`,name:"eos_token"},{anchor:"transformers.XGLMTokenizer.sep_token",description:`<strong>sep_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;/s&gt;&quot;</code>) &#x2014;
The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
sequence classification or for a text and a question for question answering. It is also used as the last
token of a sequence built with special tokens.`,name:"sep_token"},{anchor:"transformers.XGLMTokenizer.cls_token",description:`<strong>cls_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;s&gt;&quot;</code>) &#x2014;
The classifier token which is used when doing sequence classification (classification of the whole sequence
instead of per-token classification). It is the first token of the sequence when built with special tokens.`,name:"cls_token"},{anchor:"transformers.XGLMTokenizer.unk_token",description:`<strong>unk_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;unk&gt;&quot;</code>) &#x2014;
The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
token instead.`,name:"unk_token"},{anchor:"transformers.XGLMTokenizer.pad_token",description:`<strong>pad_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;pad&gt;&quot;</code>) &#x2014;
The token used for padding, for example when batching sequences of different lengths.`,name:"pad_token"},{anchor:"transformers.XGLMTokenizer.sp_model_kwargs",description:`<strong>sp_model_kwargs</strong> (<code>dict</code>, <em>optional</em>) &#x2014;
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
</ul>`,name:"sp_model_kwargs"},{anchor:"transformers.XGLMTokenizer.sp_model",description:`<strong>sp_model</strong> (<code>SentencePieceProcessor</code>) &#x2014;
The <em>SentencePiece</em> processor that is used for every conversion (string, tokens and IDs).`,name:"sp_model"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xglm/tokenization_xglm.py#L36"}}),ce=new C({props:{name:"build_inputs_with_special_tokens",anchor:"transformers.XGLMTokenizer.build_inputs_with_special_tokens",parameters:[{name:"token_ids_0",val:": list"},{name:"token_ids_1",val:": typing.Optional[list[int]] = None"}],parametersDescription:[{anchor:"transformers.XGLMTokenizer.build_inputs_with_special_tokens.token_ids_0",description:`<strong>token_ids_0</strong> (<code>list[int]</code>) &#x2014;
List of IDs to which the special tokens will be added.`,name:"token_ids_0"},{anchor:"transformers.XGLMTokenizer.build_inputs_with_special_tokens.token_ids_1",description:`<strong>token_ids_1</strong> (<code>list[int]</code>, <em>optional</em>) &#x2014;
Optional second list of IDs for sequence pairs.`,name:"token_ids_1"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xglm/tokenization_xglm.py#L175",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>List of <a href="../glossary#input-ids">input IDs</a> with the appropriate special tokens.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>list[int]</code></p>
`}}),me=new C({props:{name:"get_special_tokens_mask",anchor:"transformers.XGLMTokenizer.get_special_tokens_mask",parameters:[{name:"token_ids_0",val:": list"},{name:"token_ids_1",val:": typing.Optional[list[int]] = None"},{name:"already_has_special_tokens",val:": bool = False"}],parametersDescription:[{anchor:"transformers.XGLMTokenizer.get_special_tokens_mask.token_ids_0",description:`<strong>token_ids_0</strong> (<code>list[int]</code>) &#x2014;
List of IDs.`,name:"token_ids_0"},{anchor:"transformers.XGLMTokenizer.get_special_tokens_mask.token_ids_1",description:`<strong>token_ids_1</strong> (<code>list[int]</code>, <em>optional</em>) &#x2014;
Optional second list of IDs for sequence pairs.`,name:"token_ids_1"},{anchor:"transformers.XGLMTokenizer.get_special_tokens_mask.already_has_special_tokens",description:`<strong>already_has_special_tokens</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
Whether or not the token list is already formatted with special tokens for the model.`,name:"already_has_special_tokens"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xglm/tokenization_xglm.py#L200",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>list[int]</code></p>
`}}),pe=new C({props:{name:"create_token_type_ids_from_sequences",anchor:"transformers.XGLMTokenizer.create_token_type_ids_from_sequences",parameters:[{name:"token_ids_0",val:": list"},{name:"token_ids_1",val:": typing.Optional[list[int]] = None"}],parametersDescription:[{anchor:"transformers.XGLMTokenizer.create_token_type_ids_from_sequences.token_ids_0",description:`<strong>token_ids_0</strong> (<code>list[int]</code>) &#x2014;
List of IDs.`,name:"token_ids_0"},{anchor:"transformers.XGLMTokenizer.create_token_type_ids_from_sequences.token_ids_1",description:`<strong>token_ids_1</strong> (<code>list[int]</code>, <em>optional</em>) &#x2014;
Optional second list of IDs for sequence pairs.`,name:"token_ids_1"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xglm/tokenization_xglm.py#L228",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>List of zeros.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>list[int]</code></p>
`}}),ue=new C({props:{name:"save_vocabulary",anchor:"transformers.XGLMTokenizer.save_vocabulary",parameters:[{name:"save_directory",val:": str"},{name:"filename_prefix",val:": typing.Optional[str] = None"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xglm/tokenization_xglm.py#L284"}}),he=new Y({props:{title:"XGLMTokenizerFast",local:"transformers.XGLMTokenizerFast",headingTag:"h2"}}),fe=new C({props:{name:"class transformers.XGLMTokenizerFast",anchor:"transformers.XGLMTokenizerFast",parameters:[{name:"vocab_file",val:" = None"},{name:"tokenizer_file",val:" = None"},{name:"bos_token",val:" = '<s>'"},{name:"eos_token",val:" = '</s>'"},{name:"sep_token",val:" = '</s>'"},{name:"cls_token",val:" = '<s>'"},{name:"unk_token",val:" = '<unk>'"},{name:"pad_token",val:" = '<pad>'"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.XGLMTokenizerFast.vocab_file",description:`<strong>vocab_file</strong> (<code>str</code>) &#x2014;
Path to the vocabulary file.`,name:"vocab_file"},{anchor:"transformers.XGLMTokenizerFast.bos_token",description:`<strong>bos_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;s&gt;&quot;</code>) &#x2014;
The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.</p>
<div class="course-tip  bg-gradient-to-br dark:bg-gradient-to-r before:border-green-500 dark:before:border-green-800 from-green-50 dark:from-gray-900 to-white dark:to-gray-950 border border-green-50 text-green-700 dark:text-gray-400">
						
<p>When building a sequence using special tokens, this is not the token that is used for the beginning of
sequence. The token used is the <code>cls_token</code>.</p>

					</div>`,name:"bos_token"},{anchor:"transformers.XGLMTokenizerFast.eos_token",description:`<strong>eos_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;/s&gt;&quot;</code>) &#x2014;
The end of sequence token.</p>
<div class="course-tip  bg-gradient-to-br dark:bg-gradient-to-r before:border-green-500 dark:before:border-green-800 from-green-50 dark:from-gray-900 to-white dark:to-gray-950 border border-green-50 text-green-700 dark:text-gray-400">
						
<p>When building a sequence using special tokens, this is not the token that is used for the end of sequence.
The token used is the <code>sep_token</code>.</p>

					</div>`,name:"eos_token"},{anchor:"transformers.XGLMTokenizerFast.sep_token",description:`<strong>sep_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;/s&gt;&quot;</code>) &#x2014;
The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
sequence classification or for a text and a question for question answering. It is also used as the last
token of a sequence built with special tokens.`,name:"sep_token"},{anchor:"transformers.XGLMTokenizerFast.cls_token",description:`<strong>cls_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;s&gt;&quot;</code>) &#x2014;
The classifier token which is used when doing sequence classification (classification of the whole sequence
instead of per-token classification). It is the first token of the sequence when built with special tokens.`,name:"cls_token"},{anchor:"transformers.XGLMTokenizerFast.unk_token",description:`<strong>unk_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;unk&gt;&quot;</code>) &#x2014;
The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
token instead.`,name:"unk_token"},{anchor:"transformers.XGLMTokenizerFast.pad_token",description:`<strong>pad_token</strong> (<code>str</code>, <em>optional</em>, defaults to <code>&quot;&lt;pad&gt;&quot;</code>) &#x2014;
The token used for padding, for example when batching sequences of different lengths.`,name:"pad_token"},{anchor:"transformers.XGLMTokenizerFast.additional_special_tokens",description:`<strong>additional_special_tokens</strong> (<code>list[str]</code>, <em>optional</em>, defaults to <code>[&quot;&lt;s&gt;NOTUSED&quot;, &quot;&lt;/s&gt;NOTUSED&quot;]</code>) &#x2014;
Additional special tokens used by the tokenizer.`,name:"additional_special_tokens"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xglm/tokenization_xglm_fast.py#L36"}}),ge=new C({props:{name:"build_inputs_with_special_tokens",anchor:"transformers.XGLMTokenizerFast.build_inputs_with_special_tokens",parameters:[{name:"token_ids_0",val:": list"},{name:"token_ids_1",val:": typing.Optional[list[int]] = None"}],parametersDescription:[{anchor:"transformers.XGLMTokenizerFast.build_inputs_with_special_tokens.token_ids_0",description:`<strong>token_ids_0</strong> (<code>list[int]</code>) &#x2014;
List of IDs to which the special tokens will be added.`,name:"token_ids_0"},{anchor:"transformers.XGLMTokenizerFast.build_inputs_with_special_tokens.token_ids_1",description:`<strong>token_ids_1</strong> (<code>list[int]</code>, <em>optional</em>) &#x2014;
Optional second list of IDs for sequence pairs.`,name:"token_ids_1"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xglm/tokenization_xglm_fast.py#L123",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>List of <a href="../glossary#input-ids">input IDs</a> with the appropriate special tokens.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>list[int]</code></p>
`}}),_e=new C({props:{name:"create_token_type_ids_from_sequences",anchor:"transformers.XGLMTokenizerFast.create_token_type_ids_from_sequences",parameters:[{name:"token_ids_0",val:": list"},{name:"token_ids_1",val:": typing.Optional[list[int]] = None"}],parametersDescription:[{anchor:"transformers.XGLMTokenizerFast.create_token_type_ids_from_sequences.token_ids_0",description:`<strong>token_ids_0</strong> (<code>list[int]</code>) &#x2014;
List of IDs.`,name:"token_ids_0"},{anchor:"transformers.XGLMTokenizerFast.create_token_type_ids_from_sequences.token_ids_1",description:`<strong>token_ids_1</strong> (<code>list[int]</code>, <em>optional</em>) &#x2014;
Optional second list of IDs for sequence pairs.`,name:"token_ids_1"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xglm/tokenization_xglm_fast.py#L148",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>List of zeros.</p>
`,returnType:`<script context="module">export const metadata = 'undefined';<\/script>


<p><code>list[int]</code></p>
`}}),ke=new Y({props:{title:"XGLMModel",local:"transformers.XGLMModel",headingTag:"h2"}}),be=new C({props:{name:"class transformers.XGLMModel",anchor:"transformers.XGLMModel",parameters:[{name:"config",val:": XGLMConfig"},{name:"embed_tokens",val:": typing.Optional[torch.nn.modules.sparse.Embedding] = None"}],parametersDescription:[{anchor:"transformers.XGLMModel.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/xglm#transformers.XGLMConfig">XGLMConfig</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"},{anchor:"transformers.XGLMModel.embed_tokens",description:`<strong>embed_tokens</strong> (<code>nn.Embedding</code>, <em>optional</em>) &#x2014;
output embeddings`,name:"embed_tokens"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xglm/modeling_xglm.py#L399"}}),ve=new C({props:{name:"forward",anchor:"transformers.XGLMModel.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_hidden_states",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"cross_attn_head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"past_key_values",val:": typing.Optional[list[torch.FloatTensor]] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.Tensor] = None"}],parametersDescription:[{anchor:"transformers.XGLMModel.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.XGLMModel.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.XGLMModel.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.XGLMModel.forward.encoder_hidden_states",description:`<strong>encoder_hidden_states</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, encoder_sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention of
the decoder.`,name:"encoder_hidden_states"},{anchor:"transformers.XGLMModel.forward.encoder_attention_mask",description:`<strong>encoder_attention_mask</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, encoder_sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing cross-attention on padding tokens indices of encoder input_ids. Mask values
selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"encoder_attention_mask"},{anchor:"transformers.XGLMModel.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.XGLMModel.forward.cross_attn_head_mask",description:`<strong>cross_attn_head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_layers, attention_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the cross-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"cross_attn_head_mask"},{anchor:"transformers.XGLMModel.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>list[torch.FloatTensor]</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.XGLMModel.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.XGLMModel.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.XGLMModel.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.XGLMModel.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.XGLMModel.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.XGLMModel.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.Tensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xglm/modeling_xglm.py#L431",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPastAndCrossAttentions"
>transformers.modeling_outputs.BaseModelOutputWithPastAndCrossAttentions</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/xglm#transformers.XGLMConfig"
>XGLMConfig</a>) and inputs.</p>
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
`}}),R=new zo({props:{$$slots:{default:[Eo]},$$scope:{ctx:P}}}),Te=new Y({props:{title:"XGLMForCausalLM",local:"transformers.XGLMForCausalLM",headingTag:"h2"}}),Me=new C({props:{name:"class transformers.XGLMForCausalLM",anchor:"transformers.XGLMForCausalLM",parameters:[{name:"config",val:""}],parametersDescription:[{anchor:"transformers.XGLMForCausalLM.config",description:`<strong>config</strong> (<a href="/docs/transformers/v4.56.2/en/model_doc/xglm#transformers.XGLMForCausalLM">XGLMForCausalLM</a>) &#x2014;
Model configuration class with all the parameters of the model. Initializing with a config file does not
load the weights associated with the model, only the configuration. Check out the
<a href="/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained">from_pretrained()</a> method to load the model weights.`,name:"config"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xglm/modeling_xglm.py#L606"}}),ye=new C({props:{name:"forward",anchor:"transformers.XGLMForCausalLM.forward",parameters:[{name:"input_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"position_ids",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_hidden_states",val:": typing.Optional[torch.Tensor] = None"},{name:"encoder_attention_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"cross_attn_head_mask",val:": typing.Optional[torch.Tensor] = None"},{name:"past_key_values",val:": typing.Optional[list[torch.FloatTensor]] = None"},{name:"inputs_embeds",val:": typing.Optional[torch.Tensor] = None"},{name:"labels",val:": typing.Optional[torch.Tensor] = None"},{name:"use_cache",val:": typing.Optional[bool] = None"},{name:"output_attentions",val:": typing.Optional[bool] = None"},{name:"output_hidden_states",val:": typing.Optional[bool] = None"},{name:"return_dict",val:": typing.Optional[bool] = None"},{name:"cache_position",val:": typing.Optional[torch.Tensor] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"transformers.XGLMForCausalLM.forward.input_ids",description:`<strong>input_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.</p>
<p>Indices can be obtained using <a href="/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer">AutoTokenizer</a>. See <a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode">PreTrainedTokenizer.encode()</a> and
<a href="/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__">PreTrainedTokenizer.<strong>call</strong>()</a> for details.</p>
<p><a href="../glossary#input-ids">What are input IDs?</a>`,name:"input_ids"},{anchor:"transformers.XGLMForCausalLM.forward.attention_mask",description:`<strong>attention_mask</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing attention on padding token indices. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"attention_mask"},{anchor:"transformers.XGLMForCausalLM.forward.position_ids",description:`<strong>position_ids</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Indices of positions of each input sequence tokens in the position embeddings. Selected in the range <code>[0, config.n_positions - 1]</code>.</p>
<p><a href="../glossary#position-ids">What are position IDs?</a>`,name:"position_ids"},{anchor:"transformers.XGLMForCausalLM.forward.encoder_hidden_states",description:`<strong>encoder_hidden_states</strong> (<code>torch.FloatTensor</code> of shape <code>(batch_size, encoder_sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention of
the decoder.`,name:"encoder_hidden_states"},{anchor:"transformers.XGLMForCausalLM.forward.encoder_attention_mask",description:`<strong>encoder_attention_mask</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, encoder_sequence_length)</code>, <em>optional</em>) &#x2014;
Mask to avoid performing cross-attention on padding tokens indices of encoder input_ids. Mask values
selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 for tokens that are <strong>not masked</strong>,</li>
<li>0 for tokens that are <strong>masked</strong>.</li>
</ul>
<p><a href="../glossary#attention-mask">What are attention masks?</a>`,name:"encoder_attention_mask"},{anchor:"transformers.XGLMForCausalLM.forward.head_mask",description:`<strong>head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_heads,)</code> or <code>(num_layers, num_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the self-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"head_mask"},{anchor:"transformers.XGLMForCausalLM.forward.cross_attn_head_mask",description:`<strong>cross_attn_head_mask</strong> (<code>torch.Tensor</code> of shape <code>(num_layers, attention_heads)</code>, <em>optional</em>) &#x2014;
Mask to nullify selected heads of the cross-attention modules. Mask values selected in <code>[0, 1]</code>:</p>
<ul>
<li>1 indicates the head is <strong>not masked</strong>,</li>
<li>0 indicates the head is <strong>masked</strong>.</li>
</ul>`,name:"cross_attn_head_mask"},{anchor:"transformers.XGLMForCausalLM.forward.past_key_values",description:`<strong>past_key_values</strong> (<code>list[torch.FloatTensor]</code>, <em>optional</em>) &#x2014;
Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
blocks) that can be used to speed up sequential decoding. This typically consists in the <code>past_key_values</code>
returned by the model at a previous stage of decoding, when <code>use_cache=True</code> or <code>config.use_cache=True</code>.</p>
<p>Only <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache">Cache</a> instance is allowed as input, see our <a href="https://huggingface.co/docs/transformers/en/kv_cache" rel="nofollow">kv cache guide</a>.
If no <code>past_key_values</code> are passed, <a href="/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache">DynamicCache</a> will be initialized by default.</p>
<p>The model will output the same cache format that is fed as input.</p>
<p>If <code>past_key_values</code> are used, the user is expected to input only unprocessed <code>input_ids</code> (those that don&#x2019;t
have their past key value states given to this model) of shape <code>(batch_size, unprocessed_length)</code> instead of all <code>input_ids</code>
of shape <code>(batch_size, sequence_length)</code>.`,name:"past_key_values"},{anchor:"transformers.XGLMForCausalLM.forward.inputs_embeds",description:`<strong>inputs_embeds</strong> (<code>torch.Tensor</code> of shape <code>(batch_size, sequence_length, hidden_size)</code>, <em>optional</em>) &#x2014;
Optionally, instead of passing <code>input_ids</code> you can choose to directly pass an embedded representation. This
is useful if you want more control over how to convert <code>input_ids</code> indices into associated vectors than the
model&#x2019;s internal embedding lookup matrix.`,name:"inputs_embeds"},{anchor:"transformers.XGLMForCausalLM.forward.labels",description:`<strong>labels</strong> (<code>torch.LongTensor</code> of shape <code>(batch_size, sequence_length)</code>, <em>optional</em>) &#x2014;
Labels for computing the masked language modeling loss. Indices should either be in <code>[0, ..., config.vocab_size]</code> or -100 (see <code>input_ids</code> docstring). Tokens with indices set to <code>-100</code> are ignored
(masked), the loss is only computed for the tokens with labels in <code>[0, ..., config.vocab_size]</code>.`,name:"labels"},{anchor:"transformers.XGLMForCausalLM.forward.use_cache",description:`<strong>use_cache</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
If set to <code>True</code>, <code>past_key_values</code> key value states are returned and can be used to speed up decoding (see
<code>past_key_values</code>).`,name:"use_cache"},{anchor:"transformers.XGLMForCausalLM.forward.output_attentions",description:`<strong>output_attentions</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the attentions tensors of all attention layers. See <code>attentions</code> under returned
tensors for more detail.`,name:"output_attentions"},{anchor:"transformers.XGLMForCausalLM.forward.output_hidden_states",description:`<strong>output_hidden_states</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return the hidden states of all layers. See <code>hidden_states</code> under returned tensors for
more detail.`,name:"output_hidden_states"},{anchor:"transformers.XGLMForCausalLM.forward.return_dict",description:`<strong>return_dict</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to return a <a href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput">ModelOutput</a> instead of a plain tuple.`,name:"return_dict"},{anchor:"transformers.XGLMForCausalLM.forward.cache_position",description:`<strong>cache_position</strong> (<code>torch.Tensor</code> of shape <code>(sequence_length)</code>, <em>optional</em>) &#x2014;
Indices depicting the position of the input sequence tokens in the sequence. Contrarily to <code>position_ids</code>,
this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
the complete sequence length.`,name:"cache_position"}],source:"https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/xglm/modeling_xglm.py#L618",returnDescription:`<script context="module">export const metadata = 'undefined';<\/script>


<p>A <a
  href="/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithCrossAttentions"
>transformers.modeling_outputs.CausalLMOutputWithCrossAttentions</a> or a tuple of
<code>torch.FloatTensor</code> (if <code>return_dict=False</code> is passed or when <code>config.return_dict=False</code>) comprising various
elements depending on the configuration (<a
  href="/docs/transformers/v4.56.2/en/model_doc/xglm#transformers.XGLMConfig"
>XGLMConfig</a>) and inputs.</p>
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
`}}),V=new zo({props:{$$slots:{default:[So]},$$scope:{ctx:P}}}),J=new Co({props:{anchor:"transformers.XGLMForCausalLM.forward.example",$$slots:{default:[Bo]},$$scope:{ctx:P}}}),we=new Ho({props:{source:"https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/xglm.md"}}),{c(){l=a("meta"),y=s(),k=a("p"),b=s(),T=a("p"),T.innerHTML=c,X=s(),p(Q.$$.fragment),Ye=s(),S=a("div"),S.innerHTML=eo,Qe=s(),p(K.$$.fragment),Ke=s(),ee=a("p"),ee.innerHTML=to,et=s(),te=a("p"),te.textContent=oo,tt=s(),oe=a("p"),oe.innerHTML=no,ot=s(),ne=a("p"),ne.innerHTML=so,nt=s(),p(se.$$.fragment),st=s(),re=a("ul"),re.innerHTML=ro,rt=s(),p(ae.$$.fragment),at=s(),z=a("div"),p(ie.$$.fragment),Tt=s(),xe=a("p"),xe.innerHTML=ao,Mt=s(),$e=a("p"),$e.innerHTML=io,yt=s(),p(B.$$.fragment),it=s(),p(de.$$.fragment),dt=s(),v=a("div"),p(le.$$.fragment),wt=s(),Ge=a("p"),Ge.innerHTML=lo,Lt=s(),ze=a("p"),ze.innerHTML=co,xt=s(),I=a("div"),p(ce.$$.fragment),$t=s(),Ce=a("p"),Ce.textContent=mo,Gt=s(),Xe=a("ul"),Xe.innerHTML=po,zt=s(),A=a("div"),p(me.$$.fragment),Ct=s(),qe=a("p"),qe.innerHTML=uo,Xt=s(),U=a("div"),p(pe.$$.fragment),qt=s(),Fe=a("p"),Fe.textContent=ho,Ft=s(),Pe=a("div"),p(ue.$$.fragment),lt=s(),p(he.$$.fragment),ct=s(),L=a("div"),p(fe.$$.fragment),Pt=s(),Ie=a("p"),Ie.innerHTML=fo,It=s(),De=a("p"),De.innerHTML=go,Dt=s(),D=a("div"),p(ge.$$.fragment),Ot=s(),Oe=a("p"),Oe.textContent=_o,Nt=s(),Ne=a("ul"),Ne.innerHTML=ko,Ht=s(),j=a("div"),p(_e.$$.fragment),Wt=s(),He=a("p"),He.textContent=bo,mt=Do(`
<frameworkcontent>
<pt>
`),p(ke.$$.fragment),pt=s(),x=a("div"),p(be.$$.fragment),Et=s(),We=a("p"),We.textContent=vo,St=s(),Ee=a("p"),Ee.innerHTML=To,Bt=s(),Se=a("p"),Se.innerHTML=Mo,At=s(),O=a("div"),p(ve.$$.fragment),Ut=s(),Be=a("p"),Be.innerHTML=yo,jt=s(),p(R.$$.fragment),ut=s(),p(Te.$$.fragment),ht=s(),$=a("div"),p(Me.$$.fragment),Rt=s(),Ae=a("p"),Ae.textContent=wo,Vt=s(),Ue=a("p"),Ue.innerHTML=Lo,Jt=s(),je=a("p"),je.innerHTML=xo,Zt=s(),q=a("div"),p(ye.$$.fragment),Yt=s(),Re=a("p"),Re.innerHTML=$o,Qt=s(),p(V.$$.fragment),Kt=s(),p(J.$$.fragment),ft=s(),p(we.$$.fragment),gt=s(),Ze=a("p"),this.h()},l(e){const n=Oo("svelte-u9bgzb",document.head);l=i(n,"META",{name:!0,content:!0}),n.forEach(o),y=r(e),k=i(e,"P",{}),G(k).forEach(o),b=r(e),T=i(e,"P",{"data-svelte-h":!0}),m(T)!=="svelte-17pttnt"&&(T.innerHTML=c),X=r(e),u(Q.$$.fragment,e),Ye=r(e),S=i(e,"DIV",{class:!0,"data-svelte-h":!0}),m(S)!=="svelte-13t8s2t"&&(S.innerHTML=eo),Qe=r(e),u(K.$$.fragment,e),Ke=r(e),ee=i(e,"P",{"data-svelte-h":!0}),m(ee)!=="svelte-mub7v5"&&(ee.innerHTML=to),et=r(e),te=i(e,"P",{"data-svelte-h":!0}),m(te)!=="svelte-vfdo9a"&&(te.textContent=oo),tt=r(e),oe=i(e,"P",{"data-svelte-h":!0}),m(oe)!=="svelte-21oekg"&&(oe.innerHTML=no),ot=r(e),ne=i(e,"P",{"data-svelte-h":!0}),m(ne)!=="svelte-1c16k7b"&&(ne.innerHTML=so),nt=r(e),u(se.$$.fragment,e),st=r(e),re=i(e,"UL",{"data-svelte-h":!0}),m(re)!=="svelte-162aebv"&&(re.innerHTML=ro),rt=r(e),u(ae.$$.fragment,e),at=r(e),z=i(e,"DIV",{class:!0});var F=G(z);u(ie.$$.fragment,F),Tt=r(F),xe=i(F,"P",{"data-svelte-h":!0}),m(xe)!=="svelte-1kloxkj"&&(xe.innerHTML=ao),Mt=r(F),$e=i(F,"P",{"data-svelte-h":!0}),m($e)!=="svelte-1ek1ss9"&&($e.innerHTML=io),yt=r(F),u(B.$$.fragment,F),F.forEach(o),it=r(e),u(de.$$.fragment,e),dt=r(e),v=i(e,"DIV",{class:!0});var M=G(v);u(le.$$.fragment,M),wt=r(M),Ge=i(M,"P",{"data-svelte-h":!0}),m(Ge)!=="svelte-19vr0qz"&&(Ge.innerHTML=lo),Lt=r(M),ze=i(M,"P",{"data-svelte-h":!0}),m(ze)!=="svelte-ntrhio"&&(ze.innerHTML=co),xt=r(M),I=i(M,"DIV",{class:!0});var E=G(I);u(ce.$$.fragment,E),$t=r(E),Ce=i(E,"P",{"data-svelte-h":!0}),m(Ce)!=="svelte-1ooxl9e"&&(Ce.textContent=mo),Gt=r(E),Xe=i(E,"UL",{"data-svelte-h":!0}),m(Xe)!=="svelte-rq8uot"&&(Xe.innerHTML=po),E.forEach(o),zt=r(M),A=i(M,"DIV",{class:!0});var Le=G(A);u(me.$$.fragment,Le),Ct=r(Le),qe=i(Le,"P",{"data-svelte-h":!0}),m(qe)!=="svelte-1f4f5kp"&&(qe.innerHTML=uo),Le.forEach(o),Xt=r(M),U=i(M,"DIV",{class:!0});var kt=G(U);u(pe.$$.fragment,kt),qt=r(kt),Fe=i(kt,"P",{"data-svelte-h":!0}),m(Fe)!=="svelte-bub0ru"&&(Fe.textContent=ho),kt.forEach(o),Ft=r(M),Pe=i(M,"DIV",{class:!0});var Go=G(Pe);u(ue.$$.fragment,Go),Go.forEach(o),M.forEach(o),lt=r(e),u(he.$$.fragment,e),ct=r(e),L=i(e,"DIV",{class:!0});var N=G(L);u(fe.$$.fragment,N),Pt=r(N),Ie=i(N,"P",{"data-svelte-h":!0}),m(Ie)!=="svelte-90lafr"&&(Ie.innerHTML=fo),It=r(N),De=i(N,"P",{"data-svelte-h":!0}),m(De)!=="svelte-gxzj9w"&&(De.innerHTML=go),Dt=r(N),D=i(N,"DIV",{class:!0});var Ve=G(D);u(ge.$$.fragment,Ve),Ot=r(Ve),Oe=i(Ve,"P",{"data-svelte-h":!0}),m(Oe)!=="svelte-1ooxl9e"&&(Oe.textContent=_o),Nt=r(Ve),Ne=i(Ve,"UL",{"data-svelte-h":!0}),m(Ne)!=="svelte-rq8uot"&&(Ne.innerHTML=ko),Ve.forEach(o),Ht=r(N),j=i(N,"DIV",{class:!0});var bt=G(j);u(_e.$$.fragment,bt),Wt=r(bt),He=i(bt,"P",{"data-svelte-h":!0}),m(He)!=="svelte-bub0ru"&&(He.textContent=bo),bt.forEach(o),N.forEach(o),mt=No(e,`
<frameworkcontent>
<pt>
`),u(ke.$$.fragment,e),pt=r(e),x=i(e,"DIV",{class:!0});var H=G(x);u(be.$$.fragment,H),Et=r(H),We=i(H,"P",{"data-svelte-h":!0}),m(We)!=="svelte-pmn556"&&(We.textContent=vo),St=r(H),Ee=i(H,"P",{"data-svelte-h":!0}),m(Ee)!=="svelte-q52n56"&&(Ee.innerHTML=To),Bt=r(H),Se=i(H,"P",{"data-svelte-h":!0}),m(Se)!=="svelte-hswkmf"&&(Se.innerHTML=Mo),At=r(H),O=i(H,"DIV",{class:!0});var Je=G(O);u(ve.$$.fragment,Je),Ut=r(Je),Be=i(Je,"P",{"data-svelte-h":!0}),m(Be)!=="svelte-3i10l"&&(Be.innerHTML=yo),jt=r(Je),u(R.$$.fragment,Je),Je.forEach(o),H.forEach(o),ut=r(e),u(Te.$$.fragment,e),ht=r(e),$=i(e,"DIV",{class:!0});var W=G($);u(Me.$$.fragment,W),Rt=r(W),Ae=i(W,"P",{"data-svelte-h":!0}),m(Ae)!=="svelte-11br4fd"&&(Ae.textContent=wo),Vt=r(W),Ue=i(W,"P",{"data-svelte-h":!0}),m(Ue)!=="svelte-q52n56"&&(Ue.innerHTML=Lo),Jt=r(W),je=i(W,"P",{"data-svelte-h":!0}),m(je)!=="svelte-hswkmf"&&(je.innerHTML=xo),Zt=r(W),q=i(W,"DIV",{class:!0});var Z=G(q);u(ye.$$.fragment,Z),Yt=r(Z),Re=i(Z,"P",{"data-svelte-h":!0}),m(Re)!=="svelte-rdfjgp"&&(Re.innerHTML=$o),Qt=r(Z),u(V.$$.fragment,Z),Kt=r(Z),u(J.$$.fragment,Z),Z.forEach(o),W.forEach(o),ft=r(e),u(we.$$.fragment,e),gt=r(e),Ze=i(e,"P",{}),G(Ze).forEach(o),this.h()},h(){w(l,"name","hf:doc:metadata"),w(l,"content",Uo),w(S,"class","flex flex-wrap space-x-1"),w(z,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),w(I,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),w(A,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),w(U,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),w(Pe,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),w(v,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),w(D,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),w(j,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),w(L,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),w(O,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),w(x,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),w(q,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),w($,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,n){t(document.head,l),d(e,y,n),d(e,k,n),d(e,b,n),d(e,T,n),d(e,X,n),h(Q,e,n),d(e,Ye,n),d(e,S,n),d(e,Qe,n),h(K,e,n),d(e,Ke,n),d(e,ee,n),d(e,et,n),d(e,te,n),d(e,tt,n),d(e,oe,n),d(e,ot,n),d(e,ne,n),d(e,nt,n),h(se,e,n),d(e,st,n),d(e,re,n),d(e,rt,n),h(ae,e,n),d(e,at,n),d(e,z,n),h(ie,z,null),t(z,Tt),t(z,xe),t(z,Mt),t(z,$e),t(z,yt),h(B,z,null),d(e,it,n),h(de,e,n),d(e,dt,n),d(e,v,n),h(le,v,null),t(v,wt),t(v,Ge),t(v,Lt),t(v,ze),t(v,xt),t(v,I),h(ce,I,null),t(I,$t),t(I,Ce),t(I,Gt),t(I,Xe),t(v,zt),t(v,A),h(me,A,null),t(A,Ct),t(A,qe),t(v,Xt),t(v,U),h(pe,U,null),t(U,qt),t(U,Fe),t(v,Ft),t(v,Pe),h(ue,Pe,null),d(e,lt,n),h(he,e,n),d(e,ct,n),d(e,L,n),h(fe,L,null),t(L,Pt),t(L,Ie),t(L,It),t(L,De),t(L,Dt),t(L,D),h(ge,D,null),t(D,Ot),t(D,Oe),t(D,Nt),t(D,Ne),t(L,Ht),t(L,j),h(_e,j,null),t(j,Wt),t(j,He),d(e,mt,n),h(ke,e,n),d(e,pt,n),d(e,x,n),h(be,x,null),t(x,Et),t(x,We),t(x,St),t(x,Ee),t(x,Bt),t(x,Se),t(x,At),t(x,O),h(ve,O,null),t(O,Ut),t(O,Be),t(O,jt),h(R,O,null),d(e,ut,n),h(Te,e,n),d(e,ht,n),d(e,$,n),h(Me,$,null),t($,Rt),t($,Ae),t($,Vt),t($,Ue),t($,Jt),t($,je),t($,Zt),t($,q),h(ye,q,null),t(q,Yt),t(q,Re),t(q,Qt),h(V,q,null),t(q,Kt),h(J,q,null),d(e,ft,n),h(we,e,n),d(e,gt,n),d(e,Ze,n),_t=!0},p(e,[n]){const F={};n&2&&(F.$$scope={dirty:n,ctx:e}),B.$set(F);const M={};n&2&&(M.$$scope={dirty:n,ctx:e}),R.$set(M);const E={};n&2&&(E.$$scope={dirty:n,ctx:e}),V.$set(E);const Le={};n&2&&(Le.$$scope={dirty:n,ctx:e}),J.$set(Le)},i(e){_t||(f(Q.$$.fragment,e),f(K.$$.fragment,e),f(se.$$.fragment,e),f(ae.$$.fragment,e),f(ie.$$.fragment,e),f(B.$$.fragment,e),f(de.$$.fragment,e),f(le.$$.fragment,e),f(ce.$$.fragment,e),f(me.$$.fragment,e),f(pe.$$.fragment,e),f(ue.$$.fragment,e),f(he.$$.fragment,e),f(fe.$$.fragment,e),f(ge.$$.fragment,e),f(_e.$$.fragment,e),f(ke.$$.fragment,e),f(be.$$.fragment,e),f(ve.$$.fragment,e),f(R.$$.fragment,e),f(Te.$$.fragment,e),f(Me.$$.fragment,e),f(ye.$$.fragment,e),f(V.$$.fragment,e),f(J.$$.fragment,e),f(we.$$.fragment,e),_t=!0)},o(e){g(Q.$$.fragment,e),g(K.$$.fragment,e),g(se.$$.fragment,e),g(ae.$$.fragment,e),g(ie.$$.fragment,e),g(B.$$.fragment,e),g(de.$$.fragment,e),g(le.$$.fragment,e),g(ce.$$.fragment,e),g(me.$$.fragment,e),g(pe.$$.fragment,e),g(ue.$$.fragment,e),g(he.$$.fragment,e),g(fe.$$.fragment,e),g(ge.$$.fragment,e),g(_e.$$.fragment,e),g(ke.$$.fragment,e),g(be.$$.fragment,e),g(ve.$$.fragment,e),g(R.$$.fragment,e),g(Te.$$.fragment,e),g(Me.$$.fragment,e),g(ye.$$.fragment,e),g(V.$$.fragment,e),g(J.$$.fragment,e),g(we.$$.fragment,e),_t=!1},d(e){e&&(o(y),o(k),o(b),o(T),o(X),o(Ye),o(S),o(Qe),o(Ke),o(ee),o(et),o(te),o(tt),o(oe),o(ot),o(ne),o(nt),o(st),o(re),o(rt),o(at),o(z),o(it),o(dt),o(v),o(lt),o(ct),o(L),o(mt),o(pt),o(x),o(ut),o(ht),o($),o(ft),o(gt),o(Ze)),o(l),_(Q,e),_(K,e),_(se,e),_(ae,e),_(ie),_(B),_(de,e),_(le),_(ce),_(me),_(pe),_(ue),_(he,e),_(fe),_(ge),_(_e),_(ke,e),_(be),_(ve),_(R),_(Te,e),_(Me),_(ye),_(V),_(J),_(we,e)}}}const Uo='{"title":"XGLM","local":"xglm","sections":[{"title":"Overview","local":"overview","sections":[],"depth":2},{"title":"Resources","local":"resources","sections":[],"depth":2},{"title":"XGLMConfig","local":"transformers.XGLMConfig","sections":[],"depth":2},{"title":"XGLMTokenizer","local":"transformers.XGLMTokenizer","sections":[],"depth":2},{"title":"XGLMTokenizerFast","local":"transformers.XGLMTokenizerFast","sections":[],"depth":2},{"title":"XGLMModel","local":"transformers.XGLMModel","sections":[],"depth":2},{"title":"XGLMForCausalLM","local":"transformers.XGLMForCausalLM","sections":[],"depth":2}],"depth":1}';function jo(P){return Fo(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class en extends Po{constructor(l){super(),Io(this,l,jo,Ao,qo,{})}}export{en as component};
